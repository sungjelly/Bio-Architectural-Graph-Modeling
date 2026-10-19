"""Common fixed test-cell cosine graphs and parallel Leiden resolution sweep."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

if __package__:
    from .so12_hidden_clustering import active_output, cpu_neighbor_audit, dump, event, exact_cosine_knn, normalized_rows, sha, undirected_union
    from .so12_cancer_metrics import seed_stability
else:
    from so12_hidden_clustering import active_output, cpu_neighbor_audit, dump, event, exact_cosine_knn, normalized_rows, sha, undirected_union
    from so12_cancer_metrics import seed_stability
from spatial_benchmark.paths import current_paths
from spatial_benchmark.relative_qkv_embedding_clustering import KNNGraphResult, run_seeded_leiden


SCHEMA = "so12_cancer_common_cosine_leiden_v1"
METHODS = ("h0", "h1", "h2", "h3", "h4", "pca", "scvi")
RESOLUTIONS = (.25, .5, 1., 2.)
SEEDS = (0, 1)
CORE_COUNTS = {15: 38145, 23: 43462}
K = 30
DISK_RESERVE = 5 * 1024**3
REFERENCES = {
    "h0": ("analyses/so12_paper_test_h0_leiden/ev_so12_paper_test_h0_s0_20261018", "h0"),
    "h4": ("analyses/so12_paper_test_leiden/ev_so12_paper_test_hL_s0_20261017", "hL"),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _code_binding():
    paths = {Path(__file__).resolve(), Path(inspect.getsourcefile(exact_cosine_knn)),
             Path(inspect.getsourcefile(run_seeded_leiden)), Path(inspect.getsourcefile(seed_stability))}
    return {str(path.relative_to(current_paths().project_root)): sha(path) for path in sorted(paths)}


def _record(path, output):
    path, output = Path(path).resolve(), Path(output).resolve()
    return {"path": str(path.relative_to(output)) if path.is_relative_to(output) else str(path),
            "relative": path.is_relative_to(output), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _resolve(record, output):
    path = Path(record["path"])
    if record["relative"]:
        _require(not path.is_absolute() and ".." not in path.parts, "Invalid relative artifact path")
        path = Path(output) / path
    _require(path.is_file() and sha(path) == record["sha256"], f"Artifact changed: {path}")
    return path


def _cells(output):
    path = Path(output) / "cells.csv.gz"
    cells = pd.read_csv(path, float_precision="round_trip")
    required = ["row_index", "core_number", "slide", "fov", "cell_ID", "x_um", "y_um"]
    _require(set(required).issubset(cells), "Cell identity columns are missing")
    _require(len(cells) == sum(CORE_COUNTS.values()) and np.array_equal(cells.row_index, np.arange(len(cells))),
             "Canonical test cell row coverage changed")
    _require(np.array_equal(cells.core_number, np.concatenate([np.full(n, core) for core, n in CORE_COUNTS.items()]))
             and cells.slide.eq(2).all(), "Unexpected test cores or slide")
    _require(not cells.duplicated(["slide", "fov", "cell_ID"]).any()
             and np.isfinite(cells[required].to_numpy()).all(), "Cell keys/coordinates invalid")
    if "split" in cells:
        _require(cells.split.eq("test").all(), "Non-test cells are not permitted")
    return cells


def _labels(values, n_cells):
    values = np.asarray(values)
    _require(values.shape == (n_cells,) and np.issubdtype(values.dtype, np.integer)
             and values.min(initial=0) >= 0, "Leiden label coverage invalid")
    _require(np.array_equal(np.unique(values), np.arange(values.max() + 1)), "Cluster IDs must be contiguous")
    return values.astype(np.int32, copy=False)


def _verified_reference(method, output, cells):
    relative, representation = REFERENCES[method]
    root = current_paths().report_root / relative
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _require(json.loads((root / "_SUCCESS").read_text())["manifest_sha256"] == sha(manifest_path),
             "Reference publication marker changed")
    requested = {"cells": "cells.csv.gz", "embedding": f"embeddings/{representation}.npy",
        "indices": f"clustering/{representation}_knn_indices.npy",
        "distances": f"clustering/{representation}_knn_cosine_distances.npy",
        "edges": f"clustering/{representation}_undirected_edges.npy",
        "clustering_manifest": "clustering/clustering_manifest.json"}
    requested.update({f"labels_seed{seed}": f"clustering/{representation}_labels_seed{seed}.npy" for seed in SEEDS})
    files = {}
    for name, relative_path in requested.items():
        path = root / relative_path
        _require(sha(path) == manifest["files"][relative_path]["sha256"], f"Reference checksum changed: {name}")
        files[name] = _record(path, output)
    source_cells = pd.read_csv(root / "cells.csv.gz", float_precision="round_trip")
    columns = ["row_index", "core_number", "slide", "fov", "cell_ID", "x_um", "y_um"]
    _require(source_cells[columns].equals(cells[columns]), "Reference cell order or geometry differs")
    graph_manifest = json.loads((root / "clustering/clustering_manifest.json").read_text())
    _require(graph_manifest["status"] == "complete" and graph_manifest["k"] == K
             and graph_manifest["symmetrization"] == "undirected_union"
             and graph_manifest["edge_weights"] == "unweighted"
             and graph_manifest["normalization"] == "row_L2_for_cosine_only"
             and graph_manifest["resolution"] == 1.0 and graph_manifest["primary_seed"] == 0
             and graph_manifest["stability_seed"] == 1
             and graph_manifest["cpu_exact_audit"]["passed"], "Reference graph settings/audit differ")
    return files, {"manifest": _record(manifest_path, output), "graph_manifest": graph_manifest}


def _verify_graph(output, receipt):
    _require(receipt.get("schema") == SCHEMA and receipt.get("status") == "complete", "Graph is incomplete")
    _require(receipt["code_binding"] == _code_binding(), "Graph scientific source changed")
    for record in receipt["files"].values():
        _resolve(record, output)
    _resolve(receipt["cells"], output)
    for record in receipt.get("input_manifests", {}).values():
        _resolve(record, output)


def graph(output, method, device="cuda:0", query_chunk=512, threads=2):
    _require(method in METHODS, "Unknown representation")
    output = active_output(Path(output))
    cells = _cells(output)
    folder = output / "clustering" / method
    receipt_path = folder / "graph_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        _verify_graph(output, receipt)
        return receipt
    _require(shutil.disk_usage(output).free >= DISK_RESERVE + 64 * 1024**2,
             "Insufficient graph space above 5 GiB reserve")
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    inputs = {}
    if method in REFERENCES:
        files, reference = _verified_reference(method, output, cells)
        audit = reference["graph_manifest"]["cpu_exact_audit"]
        search = reference["graph_manifest"]["search"]
        inputs["reference_manifest"] = reference["manifest"]
        reused = True
    else:
        array_path = output / "embeddings" / f"{method}.npy"
        manifest_path = output / (f"{method}_manifest.json" if method in ("pca", "scvi") else "layers_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        _require(manifest.get("status") == "complete" and manifest.get("passed") is True,
                 "Representation manifest is incomplete")
        inputs["representation_manifest"] = _record(manifest_path, output)
        values = np.load(array_path, mmap_mode="r", allow_pickle=False)
        dimensions = 50 if method in ("pca", "scvi") else 256
        _require(values.shape == (len(cells), dimensions) and values.dtype == np.float32,
                 "Representation shape or dtype changed")
        if method in ("pca", "scvi"):
            _require(manifest.get("test_used_for_fitting") is False,
                     "A comparator fitted on test cells is not permitted")
            _require(manifest["embedding_sha256"] == sha(array_path)
                     and manifest["cells_sha256"] == sha(output / "cells.csv.gz"), "Baseline embedding/cell checksum differs")
        else:
            _require(manifest["embeddings"][method]["path"] == f"embeddings/{method}.npy"
                     and manifest["embeddings"][method]["sha256"] == sha(array_path)
                     and manifest["cells_sha256"] == sha(output / "cells.csv.gz"), "Layer embedding/cell checksum differs")
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            z = normalized_rows(values)
            queries = np.unique(np.linspace(0, len(cells) - 1, min(128, len(cells)), dtype=np.int64))
            nn, distances, pilot_search = exact_cosine_knn(z, k=K, device=device, query_chunk=query_chunk, query_indices=queries)
            pilot = cpu_neighbor_audit(z, nn, distances, queries, query_count=len(queries))
            dump(folder / "pilot.json", {"search": pilot_search, "cpu_exact_audit": pilot, "passed": pilot["passed"]})
            _require(pilot["passed"], "CPU pilot failed before full search")
            nn, distances, search = exact_cosine_knn(z, k=K, device=device, query_chunk=query_chunk)
            audit = cpu_neighbor_audit(z, nn, distances, query_count=128)
            _require(audit["passed"], "Full-search CPU neighbor audit failed")
            edges = undirected_union(nn)
        np.save(folder / "indices.npy", nn, allow_pickle=False)
        np.save(folder / "cosine_distances.npy", distances, allow_pickle=False)
        np.save(folder / "undirected_edges.npy", edges, allow_pickle=False)
        files = {name: _record(path, output) for name, path in {
            "embedding": array_path, "indices": folder / "indices.npy", "distances": folder / "cosine_distances.npy",
            "edges": folder / "undirected_edges.npy", "pilot": folder / "pilot.json"}.items()}
        reused = False
    edges = np.load(_resolve(files["edges"], output), mmap_mode="r", allow_pickle=False)
    _require(edges.ndim == 2 and edges.shape[1] == 2 and np.issubdtype(edges.dtype, np.integer)
             and np.all(edges[:, 0] < edges[:, 1]) and edges.min() >= 0 and edges.max() < len(cells), "Invalid graph edge coverage")
    receipt = {"schema": SCHEMA, "status": "complete", "passed": True, "method": method,
        "n_cells": len(cells), "dimensions": 50 if method in ("pca", "scvi") else 256,
        "k": K, "distance": "cosine", "row_l2": True, "symmetrization": "undirected_union",
        "weights": "unweighted", "spatial_edges_added": False, "edges": len(edges),
        "reused_reference": reused, "files": files, "input_manifests": inputs,
        "cells": _record(output / "cells.csv.gz", output), "code_binding": _code_binding(),
        "search": search, "cpu_exact_audit": audit, "elapsed_seconds": time.monotonic() - started}
    dump(receipt_path, receipt)
    event(output, method, "graph_ready", reused=reused, edges=len(edges))
    return receipt


def _setting_name(resolution, seed):
    return f"r{resolution:g}_s{seed}".replace(".", "p")


def _partition_task(task):
    output, method, resolution, seed = task
    output = Path(output)
    folder = output / "clustering" / method
    graph_path = folder / "graph_receipt.json"
    receipt = json.loads(graph_path.read_text())
    _verify_graph(output, receipt)
    name = _setting_name(resolution, seed)
    target, result_path = folder / f"labels_{name}.npy", folder / f"leiden_{name}.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        _require(result["graph_receipt_sha256"] == sha(graph_path), "Frozen partition graph changed")
        _resolve(result["labels"], output)
        return result
    started = time.monotonic()
    if receipt["reused_reference"] and resolution == 1.0:
        label_record = receipt["files"][f"labels_seed{seed}"]
        labels = _labels(np.load(_resolve(label_record, output), allow_pickle=False), receipt["n_cells"])
        source = json.loads(_resolve(receipt["files"]["clustering_manifest"], output).read_text())
        details = source["leiden"][str(seed)]
        reused = True
    else:
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            edges = np.load(_resolve(receipt["files"]["edges"], output), mmap_mode="r", allow_pickle=False)
            partition = run_seeded_leiden(KNNGraphResult(edge_pairs=edges, receipt={}),
                n_cells=receipt["n_cells"], resolution=resolution, random_seed=seed)
            labels, details = _labels(partition.labels, receipt["n_cells"]), partition.receipt
        label_record = None
        reused = False
    # Tiny label vectors are copied for one uniform table interface; embeddings
    # and expensive graph arrays stay immutable references when reused.
    np.save(target, labels, allow_pickle=False)
    result = {"schema": SCHEMA, "status": "complete", "passed": True, "method": method,
        "resolution": resolution, "seed": seed, "n_cells": receipt["n_cells"],
        "graph_receipt_sha256": sha(graph_path), "labels": _record(target, output),
        "reused_resolution_one": reused, "reference_labels": label_record,
        "leiden": details, "elapsed_seconds": time.monotonic() - started}
    dump(result_path, result)
    return result


def leiden(output, workers=6, methods=None):
    output = active_output(Path(output))
    _require(1 <= workers <= 6, "Leiden workers must remain within the six-process budget")
    methods = tuple(METHODS if methods is None else methods)
    _require(bool(methods) and len(set(methods)) == len(methods) and set(methods).issubset(METHODS),
             "Leiden method subset is invalid")
    cells = _cells(output)
    for method in methods:
        _verify_graph(output, json.loads((output / "clustering" / method / "graph_receipt.json").read_text()))
    tasks = [(str(output), method, resolution, seed) for method in methods for resolution in RESOLUTIONS for seed in SEEDS]
    results = []
    # Explicit spawn avoids inheriting a live CUDA context. One BLAS/OpenMP thread
    # per worker keeps the six independent CPU Leiden tasks bounded.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(_partition_task, task) for task in tasks]):
            result = future.result()
            results.append(result)
            event(output, result["method"], "leiden_complete", resolution=result["resolution"], seed=result["seed"],
                  complete=len(results), total=len(tasks))
    stability = []
    for method in methods:
        for resolution in RESOLUTIONS:
            labels = [np.load(output / "clustering" / method / f"labels_{_setting_name(resolution, seed)}.npy", allow_pickle=False) for seed in SEEDS]
            stability.append({"method": method, "resolution": resolution,
                **seed_stability(labels[0], labels[1], cells.core_number.to_numpy())})
    summary = {"schema": SCHEMA, "status": "complete", "passed": True, "n_cells": len(cells),
        "methods": list(methods), "resolutions": list(RESOLUTIONS), "seeds": list(SEEDS),
        "primary_resolution": 1.0, "primary_seed": 0, "partitions": sorted(results, key=lambda x:(x["method"],x["resolution"],x["seed"])),
        "seed_stability": stability, "workers": workers, "code_binding": _code_binding()}
    summary["all_methods_complete"] = set(methods) == set(METHODS)
    destination = (output / "clustering_summary.json" if summary["all_methods_complete"] else
                   output / "clustering" / ("partial_summary_" + "_".join(methods) + ".json"))
    dump(destination, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    graph_parser = sub.add_parser("graph")
    graph_parser.add_argument("--output", type=Path, required=True)
    graph_parser.add_argument("--method", choices=METHODS, required=True)
    graph_parser.add_argument("--device", default="cuda:0")
    graph_parser.add_argument("--query-chunk", type=int, default=512)
    leiden_parser = sub.add_parser("leiden")
    leiden_parser.add_argument("--output", type=Path, required=True)
    leiden_parser.add_argument("--workers", type=int, default=6)
    leiden_parser.add_argument("--methods", nargs="+", choices=METHODS)
    args = parser.parse_args()
    if args.stage == "graph":
        graph(args.output, args.method, args.device, args.query_chunk)
    else:
        leiden(args.output, args.workers, args.methods)


if __name__ == "__main__":
    main()
