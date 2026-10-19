"""Locked, direct h0/hL cosine clustering for the integrated SO1/SO2 cohort.

The parent analysis owns registration, extraction verification, and publication.
This stage reads aligned frozen embeddings only. It writes one independent graph
and two fixed-seed Leiden partitions per representation; it never refits PCA or
touches a training run. Exact search means exhaustive FP32 dot products, with
TF32 disabled, in bounded query blocks rather than an N-by-N matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from spatial_benchmark.paths import current_paths
from spatial_benchmark.so12_relative_qkv_training import digest
from spatial_benchmark.relative_qkv_embedding_clustering import (
    KNNGraphResult,
    run_seeded_leiden,
)

N_CELLS = 407659
N_DIMENSIONS = 256
K = 30
RESOLUTION = 1.0
SEEDS = (0, 1)
SCHEMA = "so12_direct_hidden_clustering_v1"
DISK_RESERVE = 5 * 1024**3


def active_output(out: Path) -> Path:
    out = Path(out).resolve()
    active = (current_paths().scratch_root / "active_runs").resolve()
    if not out.is_relative_to(active) or out == active:
        raise ValueError("Clustering writes must be inside a separate scratch/active_runs run")
    return out


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def event(out: Path, representation: str, stage: str, **values) -> None:
    row = {"time": datetime.now(timezone.utc).isoformat(), "representation": representation,
           "stage": stage, **values}
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{representation}_events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row), flush=True)


def normalized_rows(values: np.ndarray) -> np.ndarray:
    """Normalize only the row direction; do not center or reduce dimensions."""
    values = np.asarray(values)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Embedding must be a finite two-dimensional array with at least two rows")
    norms = np.sqrt(np.einsum("ij,ij->i", values, values, dtype=np.float64))
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Cosine clustering cannot use zero-norm or nonfinite rows")
    return np.ascontiguousarray(values / norms[:, None], dtype=np.float32)


def exact_cosine_knn(
    normalized: np.ndarray,
    *,
    k: int,
    device: str,
    query_chunk: int = 512,
    query_indices: np.ndarray | None = None,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Exhaustive FP32 search with self exclusion and stable index tie breaks.

    Only query_chunk x N similarities are materialized. Boundary ties are
    resolved against the complete score row, avoiding torch.topk's arbitrary
    choice among equal scores. CPU is available for small synthetic checks.
    """
    z = np.asarray(normalized)
    if z.dtype != np.float32 or z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("Search requires a finite float32 matrix")
    n = len(z)
    if not 0 < k < n or query_chunk < 1:
        raise ValueError("Require 0 < k < number of cells and a positive query chunk")
    if not np.allclose(np.linalg.norm(z, axis=1), 1, atol=2e-6, rtol=2e-6):
        raise ValueError("Search rows must be L2 normalized")
    queries = np.arange(n, dtype=np.int64) if query_indices is None else np.asarray(query_indices, dtype=np.int64)
    if queries.ndim != 1 or not len(queries) or np.any((queries < 0) | (queries >= n)):
        raise ValueError("Invalid query indices")
    # A small synthetic input may fit in one block, but never materialize N x N.
    actual_chunk = min(int(query_chunk), max(1, n - 1))
    target_device = torch.device(device)
    cuda = target_device.type == "cuda"
    if cuda:
        torch.cuda.set_device(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    database = torch.as_tensor(z, dtype=torch.float32, device=target_device)
    neighbors = np.empty((len(queries), k), dtype=np.int32)
    distances = np.empty((len(queries), k), dtype=np.float32)
    tied_rows = 0
    largest_block = [0, n]
    with torch.inference_mode():
        for start in range(0, len(queries), actual_chunk):
            ids = queries[start:start + actual_chunk]
            query_tensor = torch.as_tensor(ids, device=target_device)
            scores = database[query_tensor] @ database.T
            scores[torch.arange(len(ids), device=target_device), query_tensor] = -torch.inf
            values, indices = torch.topk(scores, min(k + 1, n), dim=1, largest=True, sorted=True)
            candidate_indices = indices.cpu().numpy()
            candidate_scores = values.cpu().numpy()
            order = np.lexsort((candidate_indices, -candidate_scores), axis=1)
            candidate_indices = np.take_along_axis(candidate_indices, order, axis=1)
            candidate_scores = np.take_along_axis(candidate_scores, order, axis=1)
            boundary = candidate_scores[:, k - 1] == candidate_scores[:, k]
            for local in np.flatnonzero(boundary):
                full_row = scores[int(local)].cpu().numpy()
                selected = np.lexsort((np.arange(n), -full_row))[:k]
                candidate_indices[local, :k] = selected
                candidate_scores[local, :k] = full_row[selected]
            tied_rows += int(boundary.sum())
            neighbors[start:start + len(ids)] = candidate_indices[:, :k]
            distances[start:start + len(ids)] = np.clip(1.0 - candidate_scores[:, :k], 0.0, 2.0)
            largest_block[0] = max(largest_block[0], len(ids))
            del scores, values, indices
            if progress is not None:
                progress(min(start + len(ids), len(queries)), len(queries))
    if cuda:
        torch.cuda.synchronize(target_device)
    elapsed = time.monotonic() - started
    receipt = {
        "implementation": "torch_exhaustive_float32_inner_product",
        "approximate_index": False,
        "tf32": False,
        "mixed_precision": False,
        "metric": "cosine_via_L2_normalized_inner_product",
        "tie_break": "decreasing_computed_similarity_then_global_row_index",
        "boundary_tie_rows": tied_rows,
        "n_cells": n,
        "n_queries": len(queries),
        "dimensions": z.shape[1],
        "k": k,
        "query_chunk": actual_chunk,
        "maximum_explicit_pairwise_array_shape": largest_block,
        "cell_by_cell_matrix_constructed": False,
        "device": str(target_device),
        "gpu_name": torch.cuda.get_device_name(target_device) if cuda else None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "elapsed_seconds": elapsed,
        "queries_per_second": len(queries) / elapsed,
        "estimated_full_query_seconds": n / (len(queries) / elapsed),
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(target_device) if cuda else 0,
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(target_device) if cuda else 0,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    del database
    if cuda:
        torch.cuda.empty_cache()
    return neighbors, distances, receipt


def cpu_neighbor_audit(normalized, neighbors, distances, query_indices=None, query_count=64):
    """Independent float64 CPU dot products against the entire database.

    Exact neighbor recall must be one; no approximate-search gate or silent
    fallback is allowed. Any floating-point boundary disagreement stops the
    stage for diagnosis instead of hiding the discrepancy.
    """
    z = np.asarray(normalized)
    queries = np.arange(len(z), dtype=np.int64) if query_indices is None else np.asarray(query_indices)
    positions = np.unique(np.linspace(0, len(queries) - 1, min(query_count, len(queries)), dtype=np.int64))
    double_z = z.astype(np.float64)
    k = neighbors.shape[1]
    recalls, errors = [], []
    mismatches = []
    for position in positions:
        query = int(queries[position])
        similarities = double_z @ double_z[query]
        similarities[query] = -np.inf
        chosen = np.lexsort((np.arange(len(z)), -similarities))[:k]
        actual = neighbors[position]
        recall = len(set(chosen.tolist()) & set(actual.tolist())) / k
        recalls.append(recall)
        expected_distances = np.clip(1.0 - similarities[actual], 0, 2)
        errors.append(float(np.max(np.abs(expected_distances - distances[position]))))
        if recall < 1:
            mismatches.append({"row_index": query, "recall": recall})
    return {
        "reference": "independent_numpy_float64_exhaustive_dot_product",
        "query_count": len(positions),
        "query_indices": queries[positions].tolist(),
        "mean_recall_at_k": float(np.mean(recalls)),
        "minimum_recall_at_k": float(np.min(recalls)),
        "maximum_cosine_distance_error": max(errors),
        "distance_absolute_tolerance": 2e-6,
        "mismatched_queries": mismatches,
        "passed": bool(min(recalls) == 1.0 and max(errors) <= 2e-6),
    }


def undirected_union(neighbors: np.ndarray) -> np.ndarray:
    neighbors = np.asarray(neighbors)
    if neighbors.ndim != 2 or not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError("Neighbor indices must be a two-dimensional integer array")
    n, k = neighbors.shape
    if not 0 < k < n or np.any((neighbors < 0) | (neighbors >= n)):
        raise ValueError("Neighbor indices are out of bounds")
    if np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise ValueError("Neighbor lists contain duplicate entries")
    source = np.repeat(np.arange(n, dtype=np.int64), k)
    target = neighbors.ravel().astype(np.int64)
    if np.any(source == target):
        raise ValueError("Neighbor graph contains self edges")
    codes = np.minimum(source, target) * np.int64(n) + np.maximum(source, target)
    codes = np.unique(codes)
    return np.column_stack((codes // n, codes % n)).astype(np.uint32)


def load_embeddings(out: Path, representation: str):
    if representation not in ("h0", "hL"):
        raise ValueError("Representation must be h0 or hL")
    # The collector has already checked complete-core inference and prediction
    # invariance. Bind its manifest and recheck every array before clustering.
    inputs = json.loads((out / "extraction_inputs.json").read_text())
    unhashed = dict(inputs)
    expected_input_digest = unhashed.pop("inputs_sha256")
    if digest(unhashed) != expected_input_digest:
        raise ValueError("Extraction preparation receipt changed")
    manifest_path = out / "embeddings/extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not (manifest.get("verified") is True and manifest["n_cells"] == N_CELLS
            and manifest["dimensions"] == N_DIMENSIONS and manifest["core_count"] == 28
            and manifest["inputs_sha256"] == inputs["inputs_sha256"]
            and manifest["checkpoint_sha256"] == inputs["checkpoint_sha256"]
            and manifest["all_observed"] is True and manifest["tf32"] is False):
        raise ValueError("Complete all-observed extraction has not been verified")
    core_records = {record["core_number"]: record for record in inputs["cores"]}
    receipt_hashes = {record["core_number"]: record["receipt_sha256"] for record in manifest["cores"]}
    cells_path = out / "cells.parquet"
    cells = pd.read_parquet(cells_path, columns=["row_index", "core_number"])
    if len(cells) != N_CELLS or not np.array_equal(cells.row_index, np.arange(N_CELLS)):
        raise ValueError("Cells must cover the complete cohort in canonical row order")
    cores = cells.core_number.to_numpy()
    if not np.array_equal(np.unique(cores), np.arange(1, 29)) or np.any(np.diff(cores) < 0):
        raise ValueError("Expected contiguous ascending cores 1 through 28")
    values = np.empty((N_CELLS, N_DIMENSIONS), dtype=np.float32)
    bindings = {"cells.parquet": sha(cells_path), "extraction_inputs.json": sha(out / "extraction_inputs.json"),
                "embeddings/extraction_manifest.json": sha(manifest_path)}
    if bindings["cells.parquet"] != manifest["cells_sha256"]:
        raise ValueError("Cell order differs from the extraction manifest")
    for core in range(1, 29):
        rows = np.flatnonzero(cores == core)
        path = out / "embeddings" / f"core_{core}" / f"{representation}.npy"
        receipt_path = path.parent / "receipt.json"
        if sha(receipt_path) != receipt_hashes[core]:
            raise ValueError(f"Extraction receipt changed for core {core}")
        receipt = json.loads(receipt_path.read_text())
        if not (receipt["inputs_sha256"] == inputs["inputs_sha256"]
                and receipt["checkpoint_sha256"] == inputs["checkpoint_sha256"]
                and receipt["core"] == core_records[core]
                and receipt["audit"]["all_observed"] is True
                and receipt["audit"]["masked_entries"] == 0):
            raise ValueError(f"Unexpected extraction binding for core {core}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(rows), N_DIMENSIONS) or array.dtype != np.float32:
            raise ValueError(f"Unexpected embedding schema for core {core}: {array.shape}, {array.dtype}")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite embedding for core {core}")
        values[rows] = array
        bindings[path.relative_to(out).as_posix()] = sha(path)
        if bindings[path.relative_to(out).as_posix()] != receipt["files"][path.name]["sha256"]:
            raise ValueError(f"Extraction array checksum changed for core {core}")
        bindings[receipt_path.relative_to(out).as_posix()] = receipt_hashes[core]
    return cells, values, bindings


def verify_receipt(out: Path, receipt: dict, bindings: dict) -> None:
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "complete":
        raise ValueError("Invalid clustering completion receipt")
    if receipt["input_sha256"] != bindings:
        raise ValueError("Frozen embedding inputs changed since this stage completed")
    for relative, expected in receipt["output_sha256"].items():
        if sha(out / relative) != expected:
            raise ValueError(f"Clustering artifact checksum mismatch: {relative}")


def _core_composition(labels, cells):
    table = pd.crosstab(pd.Series(labels, name="cluster"), cells.core_number)
    table.columns = [f"core_{int(x)}" for x in table.columns]
    totals = table.sum(axis=1)
    maxima = table.max(axis=1) / totals
    flagged = [{"cluster": int(label), "largest_core_fraction": float(maxima.loc[label]),
                "largest_core": int(table.loc[label].idxmax().split("_")[1]),
                "cell_count": int(totals.loc[label])} for label in table.index if maxima.loc[label] > .9]
    return table, flagged


def pilot_representation(out: Path, representation: str, device: str, query_chunk=512, threads=4):
    out = active_output(out)
    torch.set_num_threads(threads)
    cells, values, bindings = load_embeddings(out, representation)
    del cells
    with threadpool_limits(limits=threads):
        normalized = normalized_rows(values)
        del values
        queries = np.unique(np.linspace(0, len(normalized) - 1, min(4096, len(normalized)), dtype=np.int64))
        nn, distances, receipt = exact_cosine_knn(normalized, k=K, device=device,
                                                query_chunk=query_chunk, query_indices=queries)
        audit = cpu_neighbor_audit(normalized, nn, distances, queries, query_count=64)
    result = {"schema": SCHEMA, "representation": representation, "input_sha256": bindings,
              "search": receipt, "cpu_exact_audit": audit, "passed": audit["passed"]}
    dump(out / "clustering" / f"{representation}_pilot.json", result)
    if not audit["passed"]:
        raise ValueError(f"Exact CPU neighbor audit failed: {audit}")
    print(json.dumps(result), flush=True)
    return result


def cluster_representation(out, representation, device, query_chunk=512, threads=4):
    out = active_output(out)
    torch.set_num_threads(threads)
    root = out / "clustering"
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cells, values, bindings = load_embeddings(out, representation)
    receipt_path = root / f"{representation}_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        verify_receipt(out, receipt, bindings)
        return receipt
    if shutil.disk_usage(out).free < DISK_RESERVE + 350 * 1024**2:
        raise RuntimeError("Insufficient disk space for graph while retaining 5 GiB reserve")
    folder = root / representation
    folder.mkdir(exist_ok=True)
    with threadpool_limits(limits=threads):
        normalized = normalized_rows(values)
        del values
        graph_path = folder / "graph_receipt.json"
        if graph_path.exists():
            graph_receipt = json.loads(graph_path.read_text())
            verify_receipt(out, graph_receipt, bindings)
            neighbors = np.load(folder / "indices.npy", allow_pickle=False)
            distances = np.load(folder / "cosine_distances.npy", allow_pickle=False)
            edges = np.load(folder / "undirected_edges.npy", allow_pickle=False)
        else:
            event(root, representation, "exact_cosine_search_started", n_cells=len(normalized), device=device)
            last_event = time.monotonic()

            def progress(done, total):
                nonlocal last_event
                if done == total or time.monotonic() - last_event >= 30:
                    event(root, representation, "exact_cosine_search_progress", queries_complete=done,
                          queries_total=total)
                    last_event = time.monotonic()

            neighbors, distances, search = exact_cosine_knn(normalized, k=K, device=device,
                                                           query_chunk=query_chunk, progress=progress)
            audit = cpu_neighbor_audit(normalized, neighbors, distances)
            dump(folder / "cpu_exact_audit.json", audit)
            if not audit["passed"]:
                raise ValueError(f"Exact CPU neighbor audit failed: {audit}")
            edges = undirected_union(neighbors)
            np.save(folder / "indices.npy", neighbors, allow_pickle=False)
            np.save(folder / "cosine_distances.npy", distances, allow_pickle=False)
            np.save(folder / "undirected_edges.npy", edges, allow_pickle=False)
            graph_receipt = {"schema": SCHEMA, "status": "complete", "representation": representation,
                             "input_sha256": bindings, "search": search, "cpu_exact_audit": audit,
                             "symmetrization": "undirected_union", "edge_weights": "unweighted",
                             "undirected_edges": len(edges), "directed_neighbor_entries": int(neighbors.size),
                             "output_sha256": {p.relative_to(out).as_posix(): sha(p) for p in
                                               (folder / "indices.npy", folder / "cosine_distances.npy",
                                                folder / "undirected_edges.npy", folder / "cpu_exact_audit.json")}}
            dump(graph_path, graph_receipt)
        del normalized, neighbors, distances
        graph = KNNGraphResult(edge_pairs=edges, receipt=graph_receipt)
        partitions, leiden_receipts = {}, {}
        for seed in SEEDS:
            event(root, representation, "leiden_started", seed=seed, edges=len(edges))
            labels_path = root / (f"{representation}_labels.npy" if seed == 0 else f"{representation}_labels_seed1.npy")
            seed_receipt_path = folder / f"leiden_seed{seed}_receipt.json"
            if seed_receipt_path.exists():
                saved = json.loads(seed_receipt_path.read_text())
                if saved["graph_receipt_sha256"] != sha(graph_path) or saved["labels_sha256"] != sha(labels_path):
                    raise ValueError("Existing Leiden partition does not match its frozen graph or checksum")
                labels = np.load(labels_path, allow_pickle=False)
                details = saved["leiden"]
            else:
                result = run_seeded_leiden(graph, n_cells=len(cells), resolution=RESOLUTION, random_seed=seed)
                labels, details = result.labels.astype(np.int32), dict(result.receipt)
                np.save(labels_path, labels, allow_pickle=False)
                dump(seed_receipt_path, {"graph_receipt_sha256": sha(graph_path),
                                         "labels_sha256": sha(labels_path), "leiden": details})
            if labels.shape != (len(cells),) or np.any(labels < 0):
                raise ValueError("Leiden did not cover every cell")
            partitions[seed], leiden_receipts[str(seed)] = labels, details
            event(root, representation, "leiden_complete", seed=seed,
                  cluster_count=int(np.unique(labels).size))
        table, flags = _core_composition(partitions[0], cells)
        table.to_csv(folder / "core_composition.csv")
        seed_ari = float(adjusted_rand_score(partitions[0], partitions[1]))
    output_files = list(folder.glob("*.npy")) + list(folder.glob("*.json")) + [folder / "core_composition.csv",
                    root / f"{representation}_labels.npy", root / f"{representation}_labels_seed1.npy"]
    receipt = {"schema": SCHEMA, "status": "complete", "representation": representation,
               "input_sha256": bindings, "n_cells": len(cells), "dimensions": N_DIMENSIONS,
               "pca": False, "mean_centering": False, "L2_normalization_for_cosine": True,
               "resolution": RESOLUTION, "primary_seed": 0, "stability_seed": 1,
               "leiden": leiden_receipts, "seed_ARI": seed_ari,
               "clusters_over_90_percent_one_core": flags,
               "elapsed_seconds": time.monotonic() - started,
               "output_sha256": {p.relative_to(out).as_posix(): sha(p) for p in output_files}}
    dump(receipt_path, receipt)
    verify_receipt(out, receipt, bindings)
    event(root, representation, "clustering_complete", seed_ARI=seed_ari,
          cluster_count=leiden_receipts["0"]["cluster_count"])
    return receipt


def collect_clustering(out):
    out = active_output(out)
    root = out / "clustering"
    cells = pd.read_parquet(out / "cells.parquet", columns=["row_index", "core_number"])
    receipts, labels = {}, {}
    for representation in ("h0", "hL"):
        receipt = json.loads((root / f"{representation}_receipt.json").read_text())
        bindings = {name: sha(out / name) for name in receipt["input_sha256"]}
        verify_receipt(out, receipt, bindings)
        receipts[representation] = receipt
        labels[representation] = np.load(root / f"{representation}_labels.npy", allow_pickle=False)
        if labels[representation].shape != (len(cells),):
            raise ValueError("Combined labels are not aligned with cells")
    if receipts["h0"]["input_sha256"]["cells.parquet"] != receipts["hL"]["input_sha256"]["cells.parquet"]:
        raise ValueError("Representations used different cell orders")
    summary = {"schema": SCHEMA, "status": "complete", "n_cells": len(cells),
               "estimand": "exploratory_transductive_all_observed_direct_hidden_partitions",
               "h0_hL_ARI": float(adjusted_rand_score(labels["h0"], labels["hL"])),
               "per_core_h0_hL_ARI": {str(core): float(adjusted_rand_score(
                   labels["h0"][cells.core_number == core], labels["hL"][cells.core_number == core]))
                   for core in sorted(cells.core_number.unique())},
               "representations": {rep: {key: receipts[rep][key] for key in
                   ("n_cells", "dimensions", "seed_ARI", "leiden", "clusters_over_90_percent_one_core")}
                   for rep in ("h0", "hL")},
               "receipt_sha256": {rep: sha(root / f"{rep}_receipt.json") for rep in ("h0", "hL")}}
    dump(root / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("h0", "hL"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        print(json.dumps(collect_clustering(args.output)), flush=True)
    else:
        if args.representation is None:
            parser.error("--representation is required unless --collect is supplied")
        function = pilot_representation if args.pilot else cluster_representation
        function(args.output, args.representation, args.device, args.query_chunk, args.threads)


if __name__ == "__main__":
    main()
