"""Independent, read-only numerical audit of the SO12 hidden-state analysis.

No inference, clustering, fitting, GPU access, or publication occurs here.
The caller can persist the returned dictionary; the CLI writes JSON to stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from spatial_benchmark.paths import current_paths


N_CELLS = 407659
DIMENSIONS = 256
CORES = tuple(range(1, 29))
DISPLAYED_CORES = (1, 9, 13, 15, 21, 23)
METHODS = ("h0", "hL", "pca", "kernel_pca", "spatial_pca")
KEYS = ["slide", "fov", "cell_ID"]
CELL_COLUMNS = ("row_index", "slide", "fov", "cell_ID", "core_number", "alias", "x_um", "y_um", "split")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=str).encode()).hexdigest()


def _record(path):
    return {"sha256": sha(path), "size_bytes": Path(path).stat().st_size}


def _safe_path(out, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe receipt path")
    return out / relative


def _verify_hashes(out, records):
    for relative, expected in records.items():
        path = _safe_path(out, relative)
        actual = _record(path) if isinstance(expected, dict) else sha(path)
        require(actual == expected, f"Checksum mismatch: {relative}")


def check_graph(neighbors, distances, edges, *, n_cells, k):
    """Independently reconstruct every undirected edge and audit all neighbors."""
    require(neighbors.shape == distances.shape == (n_cells, k), "Incorrect neighbor array shapes")
    require(np.issubdtype(neighbors.dtype, np.integer), "Neighbors must be integers")
    require(np.issubdtype(edges.dtype, np.integer) and edges.ndim == 2 and edges.shape[1] == 2,
            "Edges must be integer pairs")
    require(np.isfinite(distances).all() and np.all((distances >= 0) & (distances <= 2)),
            "Distances must be finite and lie in [0, 2]")
    require(np.all(np.diff(distances, axis=1) >= 0), "Neighbor distances are not nondecreasing")
    require(np.all((neighbors >= 0) & (neighbors < n_cells)), "Neighbor indices out of bounds")
    require(not np.any(neighbors == np.arange(n_cells)[:, None]), "Self-neighbor found")
    require(np.all(np.diff(np.sort(neighbors, axis=1), axis=1) > 0), "Duplicate neighbor found")
    require(np.all((edges >= 0) & (edges < n_cells)) and np.all(edges[:, 0] < edges[:, 1]),
            "Undirected edge indices are invalid")
    source = np.repeat(np.arange(n_cells, dtype=np.int64), k)
    target = np.asarray(neighbors).ravel().astype(np.int64)
    ordered_codes = np.where(source < target, source * n_cells + target, target * n_cells + source)
    expected = np.unique(ordered_codes)
    actual = edges[:, 0].astype(np.int64) * n_cells + edges[:, 1].astype(np.int64)
    require(np.array_equal(actual, expected), "Saved edges differ from the exact undirected neighbor union")
    degrees = np.bincount(edges.ravel().astype(np.int64), minlength=n_cells)
    require(np.all(degrees >= k), "Union graph lost a node's outgoing neighbors")
    return {"n_cells": n_cells, "k": k, "directed_entries": int(neighbors.size),
            "undirected_edges": int(len(edges)), "degree_min": int(degrees.min()),
            "degree_max": int(degrees.max()), "degree_mean": float(degrees.mean()),
            "all_neighbor_rows_unique_and_self_free": True, "union_exact": True}, degrees


def check_partition(labels, *, n_cells, details, edges, degrees):
    require(labels.shape == (n_cells,) and np.issubdtype(labels.dtype, np.integer),
            "Every cell must have an integer label")
    ids, sizes = np.unique(labels, return_counts=True)
    require(np.array_equal(ids, np.arange(len(ids))), "Cluster labels must be contiguous from zero")
    require(np.all(np.diff(sizes) <= 0), "Labels are not sorted by descending cluster size")
    first_indices = np.full(len(ids), n_cells, dtype=np.int64)
    np.minimum.at(first_indices, labels, np.arange(n_cells))
    require(np.all(np.diff(first_indices)[np.diff(sizes) == 0] > 0),
            "Equal-size cluster labels are not sorted by minimum cell row")
    require(details["cluster_count"] == len(ids) and details["cluster_size_min"] == int(sizes.min())
            and details["cluster_size_max"] == int(sizes.max()), "Recorded cluster sizes differ")
    require(details["resolution"] == 1 and details["n_iterations"] == -1,
            "Unexpected Leiden settings")
    degree_totals = np.bincount(labels, weights=degrees, minlength=len(ids))
    internal_edges = int(np.count_nonzero(labels[edges[:, 0]] == labels[edges[:, 1]]))
    modularity = internal_edges / len(edges) - float(np.square(degree_totals / (2 * len(edges))).sum())
    quality = 2 * len(edges) * modularity
    require(np.isclose(modularity, details["modularity"], atol=1e-12, rtol=1e-12),
            "Reported modularity disagrees with saved graph and labels")
    require(np.isclose(quality, details["quality"], atol=1e-6, rtol=1e-12),
            "Reported RBConfiguration quality disagrees with saved graph and labels")
    return {"clusters": len(ids), "sizes": sizes.tolist(), "modularity": modularity,
            "quality": quality, "complete_contiguous_sorted_labels": True}


def _source_audit(out, paths):
    execution = read(out / "provenance/execution.json")
    checked, intentionally_mutable = [], []
    for relative, expected in execution["code_files"].items():
        snapshot = out / "provenance/source" / relative
        require(_record(snapshot) == expected, f"Source snapshot differs from provenance: {relative}")
        if relative.endswith(".py"):
            require(_record(paths.project_root / relative) == expected,
                    f"Current executable source changed during execution: {relative}")
            checked.append(relative)
        else:
            intentionally_mutable.append(relative)
    inputs = read(out / "extraction_inputs.json")
    require(inputs["analysis_source_sha256"] == sha(out / "provenance/source/scripts/analysis/so12_hidden_extraction.py"),
            "Extraction used different source than its preserved snapshot")
    checkpoint = Path(inputs["source_run_dir"]) / inputs["checkpoint_relative_path"]
    require(sha(checkpoint) == inputs["checkpoint_sha256"], "Frozen upstream checkpoint changed")
    for source in inputs["source_files"]:
        stat = Path(source["path"]).stat()
        require(stat.st_size == source["size_bytes"] and stat.st_mtime_ns == source["mtime_ns"],
                f"Previously checksummed data source changed: {source['path']}")
    return {"executable_sources_current_and_snapshot_verified": checked,
            "snapshot_only_nonexecutable_sources": intentionally_mutable,
            "upstream_checkpoint_checksum_verified": True,
            "previously_checksummed_data_sources_size_mtime_unchanged": len(inputs["source_files"])}


def _extraction_audit(out, cells):
    inputs = read(out / "extraction_inputs.json")
    unhashed = dict(inputs)
    expected = unhashed.pop("inputs_sha256")
    require(_digest(unhashed) == expected, "Extraction input self-hash failed")
    require(tuple(cells.columns) == CELL_COLUMNS and
            _digest(cells.to_dict(orient="list")) == inputs["cell_rows_sha256"],
            "Extraction cell schema or canonical row digest differs")
    manifest = read(out / "embeddings/extraction_manifest.json")
    require(manifest["verified"] and manifest["n_cells"] == N_CELLS and manifest["dimensions"] == DIMENSIONS
            and manifest["core_count"] == 28 and manifest["inputs_sha256"] == expected
            and manifest["cells_sha256"] == inputs["cells_sha256"] == sha(out / "cells.parquet")
            and manifest["checkpoint_sha256"] == inputs["checkpoint_sha256"]
            and manifest["all_observed"] and manifest["tf32"] is False,
            "Extraction manifest identity, dimensions, or inference mode failed")
    require([x["core_number"] for x in inputs["cores"]] == list(CORES), "Expected all 28 ordered cores")
    require([x["core_number"] for x in manifest["cores"]] == list(CORES), "Manifest core order failed")
    result, offset = [], 0
    for record, saved in zip(inputs["cores"], manifest["cores"], strict=True):
        core = record["core_number"]
        directory = out / "embeddings" / f"core_{core}"
        require(sha(directory / "receipt.json") == saved["receipt_sha256"], f"Core {core} receipt hash failed")
        receipt = read(directory / "receipt.json")
        require(receipt["core"] == record and receipt["inputs_sha256"] == expected
                and receipt["checkpoint_sha256"] == inputs["checkpoint_sha256"]
                and receipt["dtype"] == "float32" and receipt["tf32"] is False
                and receipt["audit"]["all_observed"] and receipt["audit"]["masked_entries"] == 0
                and receipt["audit"]["complete_core_graph"] and receipt["audit"]["hL_equals_full_node_embedding"],
                f"Core {core} extraction binding or inference audit failed")
        n = record["n_cells"]
        require(record["row_start"] == offset and record["row_stop"] == offset + n
                and cells.iloc[offset:offset+n].core_number.eq(core).all()
                and _digest(cells.iloc[offset:offset+n].to_dict(orient="list")) == record["cell_rows_sha256"],
                f"Core {core} row coverage or digest failed")
        arrays = {}
        norms = {}
        for name in ("h0", "hL", "delta_l2"):
            path = directory / f"{name}.npy"
            require(_record(path) == receipt["files"][path.name], f"Core {core} {name} file hash failed")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            require(array.dtype == np.float32 and array.shape == ((n,) if name == "delta_l2" else (n, DIMENSIONS))
                    and np.isfinite(array).all(), f"Core {core} {name} array schema/finite check failed")
            if name != "delta_l2":
                row_norm = np.sqrt(np.einsum("ij,ij->i", array, array, dtype=np.float64))
                require(np.all(row_norm > 0), f"Core {core} {name} has zero-norm rows")
                norms[name] = {"minimum": float(row_norm.min()), "maximum": float(row_norm.max())}
            arrays[name] = array
        delta = np.linalg.norm(arrays["hL"].astype(np.float64) - arrays["h0"].astype(np.float64), axis=1).astype(np.float32)
        require(np.array_equal(delta, arrays["delta_l2"]), f"Core {core} delta norm is not hL minus h0")
        result.append({"core_number": core, "cells": n, "norms": norms,
                       "delta_l2_mean": float(delta.mean(dtype=np.float64))})
        offset += n
    require(offset == N_CELLS, "Incomplete final cell coverage")
    pilot = read(out / "diagnostics/extraction_pilot.json")
    require(pilot == read(out / "embeddings/core_23/receipt.json") and
            all(pilot["audit"].get(name) is True for name in ("prediction_invariance_checked", "h0_equals_direct_encoder",
                "predictions_exact", "ordinary_hL_exact", "hL_equals_full_node_embedding")),
            "Largest-core prediction/intermediate-state invariance failed")
    return {"all_28_cores_verified": True, "cells": offset, "dimensions": DIMENSIONS, "per_core": result}


def _clustering_audit(out, cells):
    labels, result = {}, {}
    summary = read(out / "clustering/summary.json")
    require(summary["status"] == "complete" and summary["n_cells"] == len(cells), "Clustering summary incomplete")
    for representation in ("h0", "hL"):
        directory = out / "clustering" / representation
        parent_path = out / "clustering" / f"{representation}_receipt.json"
        receipt = read(parent_path)
        require(summary["receipt_sha256"][representation] == sha(parent_path), "Summary receipt hash failed")
        require(receipt["status"] == "complete" and receipt["representation"] == representation
                and receipt["n_cells"] == N_CELLS and receipt["dimensions"] == DIMENSIONS
                and receipt["pca"] is False and receipt["mean_centering"] is False
                and receipt["L2_normalization_for_cosine"] and receipt["resolution"] == 1
                and receipt["primary_seed"] == 0 and receipt["stability_seed"] == 1,
                f"{representation}: unexpected representation or clustering contract")
        _verify_hashes(out, receipt["input_sha256"])
        _verify_hashes(out, receipt["output_sha256"])
        graph_path = directory / "graph_receipt.json"
        graph = read(graph_path)
        require(graph["input_sha256"] == receipt["input_sha256"] and graph["status"] == "complete"
                and graph["symmetrization"] == "undirected_union" and graph["edge_weights"] == "unweighted",
                f"{representation}: graph contract failed")
        _verify_hashes(out, graph["output_sha256"])
        neighbors = np.load(directory / "indices.npy", mmap_mode="r", allow_pickle=False)
        distances = np.load(directory / "cosine_distances.npy", mmap_mode="r", allow_pickle=False)
        edges = np.load(directory / "undirected_edges.npy", mmap_mode="r", allow_pickle=False)
        graph_audit, degrees = check_graph(neighbors, distances, edges, n_cells=N_CELLS, k=30)
        require(graph["undirected_edges"] == graph_audit["undirected_edges"] and
                graph["directed_neighbor_entries"] == graph_audit["directed_entries"], "Recorded graph sizes differ")
        exact_audit = read(directory / "cpu_exact_audit.json")
        require(exact_audit == graph["cpu_exact_audit"] and exact_audit["passed"]
                and exact_audit["mean_recall_at_k"] == exact_audit["minimum_recall_at_k"] == 1
                and exact_audit["maximum_cosine_distance_error"] <= exact_audit["distance_absolute_tolerance"],
                "CPU exact-neighbor audit failed")
        search = graph["search"]
        require(search["n_cells"] == search["n_queries"] == N_CELLS and search["dimensions"] == DIMENSIONS
                and search["k"] == 30 and search["approximate_index"] is False
                and search["tf32"] is False and search["mixed_precision"] is False
                and search["cell_by_cell_matrix_constructed"] is False
                and search["maximum_explicit_pairwise_array_shape"][0] < N_CELLS,
                "Exact search settings failed")
        partitions, seeds = {}, {}
        for seed in (0, 1):
            path = out / "clustering" / (f"{representation}_labels.npy" if seed == 0 else f"{representation}_labels_seed1.npy")
            part = np.load(path, mmap_mode="r", allow_pickle=False)
            saved = read(directory / f"leiden_seed{seed}_receipt.json")
            require(saved["graph_receipt_sha256"] == sha(graph_path) and saved["labels_sha256"] == sha(path)
                    and saved["leiden"] == receipt["leiden"][str(seed)] and saved["leiden"]["random_seed"] == seed,
                    "Seed label/graph provenance failed")
            seeds[str(seed)] = check_partition(part, n_cells=N_CELLS, details=saved["leiden"], edges=edges, degrees=degrees)
            partitions[seed] = part
        ari = float(adjusted_rand_score(partitions[0], partitions[1]))
        require(ari == receipt["seed_ARI"] == summary["representations"][representation]["seed_ARI"], "Seed ARI differs")
        composition = np.zeros((seeds["0"]["clusters"], 28), dtype=np.int64)
        np.add.at(composition, (partitions[0], cells.core_number.to_numpy() - 1), 1)
        saved_table = pd.read_csv(directory / "core_composition.csv", index_col=0)
        require(saved_table.columns.tolist() == [f"core_{core}" for core in CORES]
                and np.array_equal(saved_table.index, np.arange(len(composition)))
                and np.array_equal(saved_table.to_numpy(), composition), "Core composition table differs")
        fractions = composition.max(axis=1) / composition.sum(axis=1)
        flags = [{"cluster": int(i), "largest_core_fraction": float(fractions[i]),
                  "largest_core": int(composition[i].argmax()) + 1, "cell_count": int(composition[i].sum())}
                 for i in np.flatnonzero(fractions > .9)]
        require(flags == receipt["clusters_over_90_percent_one_core"], "Core-concentrated cluster flags differ")
        labels[representation] = partitions[0]
        result[representation] = {"graph": graph_audit, "seeds": seeds, "seed_ARI": ari,
                                  "core_composition_exact": True, "cpu_neighbor_audit_verified": True}
    cross = float(adjusted_rand_score(labels["h0"], labels["hL"]))
    require(cross == summary["h0_hL_ARI"], "Cross-representation ARI differs")
    for core in CORES:
        rows = cells.core_number.to_numpy() == core
        require(float(adjusted_rand_score(labels["h0"][rows], labels["hL"][rows])) == summary["per_core_h0_hL_ARI"][str(core)],
                f"Core {core} cross-representation ARI differs")
    return result, labels, cross


def _pca_and_render_audit(out, cells, labels, paths):
    config = read(out / "config.resolved.json")
    pca = paths.artifact_root / "runs/2026/10" / config["pca_run_id"]
    checksums = read(pca / "checksums.json")
    provenance = read(out / "provenance/pca_bundle_verification.json")
    require(provenance["valid"] and provenance["run_id"] == config["pca_run_id"]
            and provenance["checksums_file"] == _record(pca / "checksums.json"),
            "PCA manifest differs from the initially verified frozen bundle")
    keys = pd.MultiIndex.from_frame(cells[KEYS])
    source_cells = pd.read_csv(pca / "cells.csv.gz")
    source_keys = pd.MultiIndex.from_frame(source_cells[KEYS])
    require(keys.is_unique and source_keys.is_unique and len(source_keys) == N_CELLS
            and len(keys.difference(source_keys)) == 0, "PCA and model cell keys differ")
    positions = source_keys.get_indexer(keys)
    require(np.all(positions >= 0) and np.unique(positions).size == N_CELLS, "PCA keyed join is not bijective")
    require(np.array_equal(cells.alias, source_cells.core.iloc[positions])
            and np.array_equal(cells.split.eq("train"), source_cells.train.iloc[positions]),
            "PCA keyed core/split identities differ")
    require(np.allclose(cells[["x_um", "y_um"]], source_cells.iloc[positions][["x_um", "y_um"]], atol=1e-8, rtol=0),
            "PCA keyed coordinates differ")
    checked = ["cells.csv.gz", "_SUCCESS", "resolved_config.json"]
    for method in METHODS[2:]:
        relative = f"methods/{method}/clusters.csv.gz"
        clusters = pd.read_csv(pca / relative)
        require(clusters.row_index.is_unique and len(clusters) == N_CELLS, "PCA source label rows invalid")
        by_row = clusters.set_index("row_index")["leiden_r1.0_s0"].reindex(source_cells.row_index)
        require(not by_row.isna().any() and np.issubdtype(by_row.dtype, np.integer), "PCA label row alignment failed")
        labels[method] = by_row.to_numpy()[positions]
        require(np.array_equal(np.unique(labels[method]), np.arange(int(labels[method].max()) + 1)),
                "PCA labels are noncontiguous or invalid")
        checked.append(relative)
    for relative in checked:
        require(_record(pca / relative) == checksums[relative], f"Frozen PCA checksum differs: {relative}")
    metrics_path = out / "report/comparison_metrics.json"
    if not metrics_path.exists():
        return {"frozen_PCA_keyed_labels_verified": True, "rendering_checked": False,
                "reason": "Renderer has not written comparison_metrics.json"}
    saved = read(metrics_path)
    require(saved["method_order"] == list(METHODS), "Comparison method order differs")
    populations = {"all_cells": np.ones(N_CELLS, dtype=bool),
                   "requested_cores": cells.core_number.isin(DISPLAYED_CORES).to_numpy()}
    populations.update({f"core_{core:02d}": cells.core_number.to_numpy() == core for core in DISPLAYED_CORES})
    require(set(saved["populations"]) == set(populations), "Comparison populations differ")
    comparisons = 0
    for name, rows in populations.items():
        record = saved["populations"][name]
        require(record["cells"] == int(rows.sum()), f"{name}: reported cell count differs")
        expected = np.eye(5)
        for i, first in enumerate(METHODS):
            require(record["observed_cluster_counts"][first] == len(np.unique(labels[first][rows])),
                    f"{name}/{first}: reported cluster count differs")
            for j in range(i + 1, 5):
                expected[i, j] = expected[j, i] = adjusted_rand_score(labels[first][rows], labels[METHODS[j]][rows])
                comparisons += 1
        require(np.array_equal(expected, np.asarray(record["pairwise_ari"])), f"{name}: reported ARI matrix differs")
    colors = pd.read_csv(out / "report/cluster_color_keys.csv")
    for method in METHODS:
        records = colors.loc[colors.method == method].sort_values("cluster_id")
        require(np.array_equal(records.cluster_id, np.arange(int(labels[method].max()) + 1))
                and np.array_equal(records.all_cells, np.bincount(labels[method]))
                and np.array_equal(records.selected_cells, np.bincount(labels[method][populations["requested_cores"]], minlength=len(records))),
                f"{method}: legend population counts differ from exact labels")
    figure_path = out / "report/figure_manifest.json"
    require(figure_path.exists(), "Renderer metrics exist but figure completion manifest is absent")
    figures = read(figure_path)
    _verify_hashes(out, figures["files"])
    require(figures["inputs"]["cells"] == _record(out / "cells.parquet")
            and figures["inputs"]["h0_labels"] == _record(out / "clustering/h0_labels.npy")
            and figures["inputs"]["hL_labels"] == _record(out / "clustering/hL_labels.npy")
            and figures["inputs"]["clustering_summary"] == _record(out / "clustering/summary.json")
            and figures["inputs"]["pca_checksums"] == _record(pca / "checksums.json"),
            "Figure inputs differ from the independently audited arrays")
    for relative, expected in figures["inputs"]["pca_sources"].items():
        require(expected == _record(pca / relative) == checksums[relative], "Figure PCA source differs")
    require(figures["partition_comparison"] == saved and figures["status"] == "complete"
            and figures["cores"] == list(DISPLAYED_CORES) and figures["qc"]["selected_cell_count"] == 117996
            and figures["qc"]["polygons"]["covered_cells"] == 117996
            and figures["qc"]["polygons"]["alignment_mismatches"] == 0
            and figures["rendering"]["actual_cell_polygons"] is True,
            "Figure manifest coverage, comparison, or polygon alignment failed")
    return {"frozen_PCA_keyed_labels_verified": True, "rendering_checked": True,
            "pairwise_ARI_values_independently_recomputed": comparisons,
            "legend_population_counts_exact": True, "figure_files_checksum_verified": len(figures["files"])}


def audit(out: Path) -> dict:
    out = Path(out).resolve()
    paths = current_paths()
    started = time.monotonic()
    with threadpool_limits(limits=4):
        cells = pd.read_parquet(out / "cells.parquet")
        require(len(cells) == N_CELLS and np.array_equal(cells.row_index, np.arange(N_CELLS))
                and not cells.duplicated(KEYS).any() and np.isfinite(cells[["x_um", "y_um"]]).all().all()
                and np.array_equal(np.unique(cells.core_number), CORES), "Cell coverage, keys, coordinates, or order invalid")
        sources = _source_audit(out, paths)
        extraction = _extraction_audit(out, cells)
        clustering, labels, cross = _clustering_audit(out, cells)
        comparison = _pca_and_render_audit(out, cells, labels, paths)
    return {"schema": "so12_hidden_leiden_independent_numeric_audit_v1", "status": "passed",
            "audited_at": datetime.now(timezone.utc).isoformat(), "output": str(out),
            "audit_source_sha256": sha(Path(__file__)), "elapsed_seconds": time.monotonic() - started,
            "sources": sources, "extraction": extraction, "clustering": clustering,
            "h0_hL_ARI": cross, "comparison": comparison, "gpu_accessed": False,
            "inference_or_clustering_repeated": False, "visual_review_performed": False}


def smoke() -> dict:
    """Small hand-calculated graph plus intentional corruption checks."""
    neighbors = np.array([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]], dtype=np.int32)
    distances = np.tile([.1, .2], (6, 1)).astype(np.float32)
    edges = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]], dtype=np.uint32)
    graph, degrees = check_graph(neighbors, distances, edges, n_cells=6, k=2)
    details = {"cluster_count": 2, "cluster_size_min": 3, "cluster_size_max": 3,
               "resolution": 1, "n_iterations": -1, "modularity": .5, "quality": 6.}
    partition = check_partition(np.array([0, 0, 0, 1, 1, 1]), n_cells=6, details=details, edges=edges, degrees=degrees)
    invalid = [neighbors.copy(), neighbors.copy()]
    invalid[0][0, 0] = 0
    invalid[1][0, 1] = 1
    for bad in invalid:
        try:
            check_graph(bad, distances, edges, n_cells=6, k=2)
        except ValueError:
            continue
        raise AssertionError("Corrupt neighbors were accepted")
    try:
        check_graph(neighbors, distances, edges[:-1], n_cells=6, k=2)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing union edge was accepted")
    return {"status": "passed", "graph": graph, "partition": partition, "corruption_rejections": 3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and args.output is None:
        parser.error("--output is required unless --smoke is supplied")
    print(json.dumps(smoke() if args.smoke else audit(args.output), indent=2, allow_nan=False), flush=True)
