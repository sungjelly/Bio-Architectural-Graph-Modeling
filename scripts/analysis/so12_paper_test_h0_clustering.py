"""Fixed, test-only h0 clustering and original-cell polygon figures.

The parent controller owns the finalized-training gate, registration and
publication. This module accepts only a complete, checksummed extraction of
cores 15/23. It never loads a training/validation dataset or a checkpoint.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

if __package__:
    from .so12_hidden_clustering import (
        active_output, cpu_neighbor_audit, dump, event, exact_cosine_knn,
        normalized_rows, sha, undirected_union,
    )
    from .so12_hidden_maps import (
        _bounds, _draw, _save, _validate_labels, color_keys, polygon_vertices, plt,
    )
else:  # Direct script execution with the repository's PYTHONPATH=src convention.
    from so12_hidden_clustering import (
        active_output, cpu_neighbor_audit, dump, event, exact_cosine_knn,
        normalized_rows, sha, undirected_union,
    )
    from so12_hidden_maps import (
        _bounds, _draw, _save, _validate_labels, color_keys, polygon_vertices, plt,
    )
from spatial_benchmark.attention_niche_geometry import load_aligned_cosmx_polygons
from spatial_benchmark.data import resolve_nested_raw_path
from spatial_benchmark.paths import current_paths
from spatial_benchmark.relative_qkv_embedding_clustering import KNNGraphResult, run_seeded_leiden


SCHEMA = "so12_paper_test_h0_leiden_v1"
REPRESENTATION = "trained_node_encoder_output_before_graph_blocks"
SOURCE_RUN_ID = "r_20261017T044837Z_caf76350_s000_f00_a01_12df41be"
EXPECTED_CORE_COUNTS = {15: 38_145, 23: 43_462}
N_DIMENSIONS = 256
K = 30
SEEDS = (0, 1)
RESOLUTION = 1.0
KEYS = ["slide", "fov", "cell_ID"]
DISK_RESERVE = 5 * 1024**3
DISK_BUDGET = 512 * 1024**2
LIMITATION = (
    "h0 is the trained node encoder output before graph message passing. "
    "The h0 partition is independent of hL: matching cluster IDs or colors do not "
    "indicate matching cell groups across representations. "
    "Exploratory transductive grouping of the two test cores under a frozen model. "
    "Clusters are not validated cell types, cancer regions, independent donor "
    "replication, or causal interactions. Training permits donor overlap and "
    "historical sample exposure. Seed 0 is displayed; seed 1 only measures stability."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_test_cells(cells, *, expected_counts=None):
    """Require exact canonical coverage; never silently sort or drop cells."""
    expected_counts = EXPECTED_CORE_COUNTS if expected_counts is None else expected_counts
    required = KEYS + ["row_index", "core_number", "core_alias", "split", "x_um", "y_um"]
    _require(set(required).issubset(cells.columns), "Cells lack identity, core, split or coordinates")
    _require(len(cells) == sum(expected_counts.values()), "Test cell coverage changed")
    _require(not cells[required].isna().any().any(), "Missing test cell metadata")
    for column in KEYS + ["row_index", "core_number"]:
        _require(np.issubdtype(cells[column].dtype, np.integer), f"{column} must be an integer")
    _require(np.array_equal(cells.row_index.to_numpy(), np.arange(len(cells))),
             "Cells must retain canonical row order")
    expected_cores = np.concatenate([np.full(count, core) for core, count in sorted(expected_counts.items())])
    _require(np.array_equal(cells.core_number.to_numpy(), expected_cores),
             "Only ordered test cores 15 and 23 are permitted")
    _require(cells.slide.eq(2).all() and cells.split.eq("test").all(), "Non-test or wrong-slide cells")
    expected_alias = np.asarray([f"SO2-C{core}" for core in expected_cores])
    _require(np.array_equal(cells.core_alias.to_numpy(), expected_alias), "Core alias mismatch")
    _require(not cells.duplicated(KEYS).any(), "Duplicate slide-qualified cell keys")
    _require(np.isfinite(cells[["x_um", "y_um"]].to_numpy()).all(), "Nonfinite cell coordinates")


def _code_binding():
    # Bind the functions that determine neighbors, partitions and polygon maps.
    import inspect
    paths = {Path(__file__).resolve()}
    for function in (exact_cosine_knn, _draw, run_seeded_leiden,
                     load_aligned_cosmx_polygons, resolve_nested_raw_path):
        paths.add(Path(inspect.getsourcefile(function)).resolve())
    root = current_paths().project_root
    return {str(path.relative_to(root)): sha(path) for path in sorted(paths)}


def load_inputs(output):
    """Validate extraction identities/checksums before accessing its matrices."""
    output = Path(output)
    manifest_path = output / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _require(manifest.get("status") == "complete" and manifest.get("passed") is True
             and manifest.get("verified") is True, "Extraction is not complete and verified")
    _require(manifest.get("source_run_id") == SOURCE_RUN_ID, "Unexpected selected source run")
    _require(manifest.get("n_cells") == sum(EXPECTED_CORE_COUNTS.values())
             and manifest.get("dimensions") == N_DIMENSIONS, "Extraction shape differs from contract")
    _require(manifest.get("test_core_numbers") == [15, 23] and manifest.get("all_observed") is True,
             "Extraction is not all-observed test-only h0")
    _require(manifest.get("representation_name") == "h0"
             and manifest.get("representation") == REPRESENTATION,
             "Extraction must be trained node-encoder h0 before graph blocks")
    _require(manifest.get("h0_path") == "embeddings/h0.npy"
             and "hL_path" not in manifest and "hL_sha256" not in manifest,
             "h0 extraction must use its own explicit embedding path and identity")
    for field in ("checkpoint_sha256", "preprocessing_fingerprint", "fitted_parameters_sha256"):
        _require(isinstance(manifest.get(field), str) and len(manifest[field]) == 64,
                 f"Missing frozen extraction identity: {field}")
    cell_path, embedding_path = output / "cells.csv.gz", output / "embeddings/h0.npy"
    checksums = {"cells.csv.gz": sha(cell_path), "embeddings/h0.npy": sha(embedding_path)}
    _require(checksums["cells.csv.gz"] == manifest.get("cells_sha256"), "Cell checksum changed")
    _require(checksums["embeddings/h0.npy"] == manifest.get("embedding_sha256")
             == manifest.get("h0_sha256"), "Embedding checksum changed")
    core_records = manifest.get("cores", [])
    _require([core["core_number"] for core in core_records] == [15, 23], "Core extraction receipt coverage changed")
    row_start = 0
    for core in core_records:
        count = EXPECTED_CORE_COUNTS[core["core_number"]]
        _require(core["n_cells"] == count and core["row_start"] == row_start
                 and core["row_stop"] == row_start + count, "Core extraction row bounds changed")
        row_start += count
        relative = Path(core["receipt_path"])
        _require(not relative.is_absolute() and ".." not in relative.parts, "Invalid core receipt path")
        checksums[str(relative)] = sha(output / relative)
        _require(checksums[str(relative)] == core["receipt_sha256"], "Core extraction receipt checksum changed")
    cells = pd.read_csv(cell_path, float_precision="round_trip")
    if "alias" in cells and "core_alias" not in cells:
        cells = cells.rename(columns={"alias": "core_alias"})
    validate_test_cells(cells)
    values = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    _require(values.shape == (len(cells), N_DIMENSIONS) and values.dtype == np.float32,
             "h0 must be a cell-aligned float32 matrix with 256 dimensions")
    _require(np.isfinite(values).all(), "Nonfinite h0 values")
    _require(np.all(np.einsum("ij,ij->i", values, values, dtype=np.float64) > 0), "Zero-norm h0 row")
    checksums["extraction_manifest.json"] = sha(manifest_path)
    return cells, values, {"files": checksums, "code": _code_binding()}, manifest


def _verify_complete(output, receipt, bindings):
    _require(receipt.get("schema") == SCHEMA and receipt.get("status") == "complete",
             "Invalid clustering completion receipt")
    _require(receipt.get("input_binding") == bindings, "Clustering input/source binding changed")
    for relative, expected in receipt["output_sha256"].items():
        _require(sha(output / relative) == expected, f"Artifact checksum changed: {relative}")


def _summary(output, receipt, manifest_path, filename):
    summary = {"schema": SCHEMA, "status": "complete", "passed": True,
        "representation_name": "h0", "representation": REPRESENTATION,
        "n_cells": sum(EXPECTED_CORE_COUNTS.values()), "test_core_numbers": [15, 23],
        "cells_per_core": {str(core): count for core, count in EXPECTED_CORE_COUNTS.items()},
        "manifest_path": str(manifest_path.relative_to(output)), "manifest_sha256": sha(manifest_path),
        "output_sha256": receipt["output_sha256"]}
    if "metrics" in receipt:
        summary["metrics"] = receipt["metrics"]
    if "qc" in receipt:
        summary["qc"] = receipt["qc"]
    dump(output / filename, summary)
    return summary


def partition_tables(cells, primary, secondary):
    """Cell-aligned labels and descriptive counts; no biological validation."""
    primary = _validate_labels(primary, len(cells), "seed 0")
    secondary = _validate_labels(secondary, len(cells), "seed 1")
    labels = cells[["row_index", "core_number", "core_alias"] + KEYS].copy()
    labels["leiden_r1.0_s0"], labels["leiden_r1.0_s1"] = primary, secondary
    sizes = pd.DataFrame({"cluster_id": np.arange(primary.max() + 1), "cells": np.bincount(primary)})
    sizes["fraction_of_test_cells"] = sizes.cells / len(cells)
    counts = pd.crosstab(pd.Series(primary, name="cluster_id"), pd.Series(cells.core_number.to_numpy(), name="core_number"))
    composition = counts.stack().rename("cells").reset_index()
    composition["fraction_of_cluster"] = composition.cells / composition.cluster_id.map(counts.sum(axis=1))
    composition["fraction_of_core"] = composition.cells / composition.core_number.map(cells.core_number.value_counts())
    metrics = {
        "seed0_seed1_ari": float(adjusted_rand_score(primary, secondary)),
        "seed0_seed1_ari_per_core": {str(core): float(adjusted_rand_score(
            primary[cells.core_number.to_numpy() == core], secondary[cells.core_number.to_numpy() == core]))
            for core in sorted(cells.core_number.unique())},
        "cluster_count_by_seed": {"0": int(primary.max() + 1), "1": int(secondary.max() + 1)},
        "seed0_cluster_vs_core_ari_descriptive": float(adjusted_rand_score(primary, cells.core_number)),
        "clusters_over_90_percent_one_core": [int(x) for x in counts.index[(counts.max(axis=1) / counts.sum(axis=1)) > .9]],
        "interpretation": LIMITATION,
    }
    return labels, sizes, composition, metrics


def cluster(output, device="cuda:0", query_chunk=512, threads=4):
    output = active_output(Path(output))
    cells, values, bindings, extraction = load_inputs(output)
    folder = output / "clustering"
    receipt_path = folder / "clustering_manifest.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        _verify_complete(output, receipt, bindings)
        _summary(output, receipt, receipt_path, "clustering_summary.json")
        return receipt
    if shutil.disk_usage(output).free < DISK_RESERVE + DISK_BUDGET:
        raise RuntimeError("Clustering needs 0.5 GiB free above the 5 GiB reserve")
    folder.mkdir(parents=True, exist_ok=True)
    input_path = folder / "input_binding.json"
    if input_path.exists():
        _require(json.loads(input_path.read_text()) == bindings, "Partial clustering has different inputs")
    else:
        dump(input_path, bindings)
    started = time.monotonic()
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads):
        normalized = normalized_rows(values)
        del values
        # Actual-data pilot must pass before the full scaled neighbor search.
        queries = np.unique(np.linspace(0, len(cells) - 1, min(128, len(cells)), dtype=np.int64))
        event(folder, "h0", "neighbor_pilot_started", cells=len(cells), queries=len(queries))
        pilot_nn, pilot_distances, pilot_search = exact_cosine_knn(
            normalized, k=K, device=device, query_chunk=query_chunk, query_indices=queries)
        pilot_audit = cpu_neighbor_audit(normalized, pilot_nn, pilot_distances, queries, query_count=len(queries))
        dump(folder / "neighbor_pilot.json", {"input_binding": bindings, "search": pilot_search,
            "query_indices": queries.tolist(), "cpu_exact_audit": pilot_audit, "passed": pilot_audit["passed"]})
        _require(pilot_audit["passed"], "Pilot exact-neighbor CPU audit failed; full search was not started")
        del pilot_nn, pilot_distances
        last_event = time.monotonic()

        def progress(done, total):
            nonlocal last_event
            if done == total or time.monotonic() - last_event >= 30:
                event(folder, "h0", "neighbor_search_progress", queries_complete=done, queries_total=total)
                last_event = time.monotonic()

        neighbors, distances, search = exact_cosine_knn(
            normalized, k=K, device=device, query_chunk=query_chunk, progress=progress)
        audit = cpu_neighbor_audit(normalized, neighbors, distances, query_count=128)
        dump(folder / "cpu_neighbor_audit.json", audit)
        _require(audit["passed"], "Full exact-neighbor CPU audit failed")
        edges = undirected_union(neighbors)
        np.save(folder / "h0_knn_indices.npy", neighbors, allow_pickle=False)
        np.save(folder / "h0_knn_cosine_distances.npy", distances, allow_pickle=False)
        np.save(folder / "h0_undirected_edges.npy", edges, allow_pickle=False)
        del normalized, neighbors, distances
        graph = KNNGraphResult(edge_pairs=edges, receipt={"search": search, "edge_weights": "unweighted"})
        partitions, leiden = {}, {}
        for seed in SEEDS:
            event(folder, "h0", "leiden_started", seed=seed, edges=len(edges))
            result = run_seeded_leiden(graph, n_cells=len(cells), resolution=RESOLUTION, random_seed=seed)
            partitions[seed] = _validate_labels(result.labels, len(cells), f"seed {seed}")
            leiden[str(seed)] = result.receipt
            np.save(folder / f"h0_labels_seed{seed}.npy", partitions[seed], allow_pickle=False)
            event(folder, "h0", "leiden_complete", seed=seed, clusters=len(np.unique(partitions[seed])))
        labels, sizes, composition, metrics = partition_tables(cells, partitions[0], partitions[1])
        cores = cells.core_number.to_numpy()
        metrics["cross_core_undirected_edge_fraction"] = float(np.mean(cores[edges[:, 0]] != cores[edges[:, 1]]))
    labels.to_csv(folder / "clusters.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    sizes.to_csv(folder / "cluster_sizes.csv", index=False)
    composition.to_csv(folder / "core_composition.csv", index=False)
    dump(folder / "metrics.json", metrics)
    receipt = {"schema": SCHEMA, "status": "complete", "input_binding": bindings,
        "source_run_id": SOURCE_RUN_ID, "checkpoint_sha256": extraction["checkpoint_sha256"],
        "preprocessing_fingerprint": extraction["preprocessing_fingerprint"],
        "fitted_parameters_sha256": extraction["fitted_parameters_sha256"],
        "n_cells": len(cells), "dimensions": N_DIMENSIONS, "cores": [15, 23],
        "representation": REPRESENTATION, "representation_name": "h0", "all_observed": True,
        "normalization": "row_L2_for_cosine_only", "pca": False, "mean_centering": False,
        "k": K, "symmetrization": "undirected_union", "edge_weights": "unweighted",
        "resolution": RESOLUTION, "primary_seed": 0, "stability_seed": 1,
        "search": search, "cpu_exact_audit": audit, "pilot_cpu_exact_audit": pilot_audit,
        "undirected_edges": len(edges), "leiden": leiden, "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "output_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(folder.iterdir())
                          if p.is_file() and p.suffix != ".jsonl" and p != receipt_path}}
    dump(receipt_path, receipt)
    _summary(output, receipt, receipt_path, "clustering_summary.json")
    return receipt


def _preview(path):
    from PIL import Image
    with Image.open(path) as image:
        image.thumbnail((1600, 1200))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def render(output, paths=None):
    output = active_output(Path(output))
    paths = paths or current_paths()
    cells, values, bindings, extraction = load_inputs(output)
    del values
    cluster_path = output / "clustering/clustering_manifest.json"
    receipt = json.loads(cluster_path.read_text())
    _verify_complete(output, receipt, bindings)
    labels = _validate_labels(np.load(output / "clustering/h0_labels_seed0.npy", allow_pickle=False), len(cells), "seed 0")
    palette, palette_source = color_keys(labels, "h0")
    source = resolve_nested_raw_path(paths.data_root / "raw", "26040302SO_2-polygons.csv")
    alignment = load_aligned_cosmx_polygons({"SO_2": source}, cells[KEYS],
        prepared_coordinates_um=cells[["x_um", "y_um"]].to_numpy(), centroid_tolerance_um=5.0)
    expected_keys = tuple(("SO_2", int(row.fov), int(row.cell_ID)) for row in cells.itertuples())
    _require(alignment.keys == expected_keys and alignment.n_cells == len(cells), "Polygon key order/coverage mismatch")
    vertices = polygon_vertices(alignment.polygons_um)
    _require(len(vertices) == len(cells), "Polygon count differs from test cells")
    figures, reports = output / "figures", output / "report"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    palette_table = pd.DataFrame({"cluster_id": list(palette), "color_hex": list(palette.values()),
        "cells": np.bincount(labels), "cluster_label": [f"h0-{i:02d}" for i in palette]})
    palette_table.to_csv(reports / "cluster_color_keys.csv", index=False)
    saved, bounds, coverage = {}, {}, {}
    core_vertices, core_labels = {}, {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}):
        for core in EXPECTED_CORE_COUNTS:
            rows = np.flatnonzero(cells.core_number.to_numpy() == core)
            core_vertices[core] = tuple(vertices[i] for i in rows)
            core_labels[core] = labels[rows]
            bounds[core] = _bounds(core_vertices[core])
            coverage[str(core)] = len(rows)
            figure, axis = plt.subplots(figsize=(7, 7))
            _draw(axis, core_vertices[core], core_labels[core], palette, bounds[core])
            axis.set_title(f"Test core {core} · {len(rows):,} cells · trained h0 Leiden")
            figure.text(.5, .015, "Trained node encoder before graph blocks · h0 colors/IDs are independent of hL",
                        ha="center", fontsize=8)
            saved[f"core_{core}"] = _save(figure, figures, f"core_{core}_h0_leiden_segmentation", 250)
        figure, axes = plt.subplots(1, 2, figsize=(14, 7))
        for axis, core in zip(axes, EXPECTED_CORE_COUNTS, strict=True):
            _draw(axis, core_vertices[core], core_labels[core], palette, bounds[core])
            axis.set_title(f"Test core {core} · {coverage[str(core)]:,} cells")
        figure.suptitle("Trained h0 before graph blocks · Joint test-core Leiden · Shared h0 colors")
        figure.text(.5, .01, "h0 clusters are independent of hL; matching IDs/colors do not imply correspondence",
                    ha="center", fontsize=9)
        figure.tight_layout(rect=(0, .035, 1, .95))
        saved["overview"] = _save(figure, figures, "test_cores_h0_leiden", 220)
        n_columns = min(4, len(palette))
        n_rows = int(np.ceil(len(palette) / n_columns))
        figure, axis = plt.subplots(figsize=(max(4, n_columns * 3.2), max(1.7, .32 * n_rows + .6)))
        axis.axis("off")
        for i, row in enumerate(palette_table.itertuples()):
            column, line = divmod(i, n_rows)
            x, y = column / n_columns, 1 - (line + .5) / n_rows
            axis.scatter([x + .025], [y], s=90, marker="s", color=row.color_hex, transform=axis.transAxes)
            axis.text(x + .055, y, f"{row.cluster_label} · {row.cells:,}", va="center", transform=axis.transAxes)
        saved["legend"] = _save(figure, figures, "shared_cluster_color_key", 180)
    # Verify exported raster dimensions and PDF signatures; no visual-review claim.
    from PIL import Image
    export_checks = {}
    for name, filenames in saved.items():
        png, pdf = figures / filenames["png"], figures / filenames["pdf"]
        with Image.open(png) as image:
            image.verify()
        with Image.open(png) as image:
            dimensions = list(image.size)
        _require(min(dimensions) >= 100 and pdf.read_bytes()[:5] == b"%PDF-", "Figure export is invalid")
        export_checks[name] = {"png_dimensions": dimensions, "png_bytes": png.stat().st_size,
                              "pdf_bytes": pdf.stat().st_size, "verified": True}
    sizes = pd.read_csv(output / "clustering/cluster_sizes.csv")
    composition = pd.read_csv(output / "clustering/core_composition.csv")
    metrics = receipt["metrics"]
    sections = "".join(f'<section><h2>Test core {core}</h2><img alt="Core {core}: actual cell polygons" '
        f'src="data:image/png;base64,{_preview(figures / saved[f"core_{core}"]["png"])}"></section>' for core in EXPECTED_CORE_COUNTS)
    color_rows = "".join(f'<tr><td><span style="background:{row.color_hex}"></span>{row.cluster_label}</td>'
        f'<td>{row.cells:,}</td></tr>' for row in palette_table.itertuples())
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    document += '<title>Test cores 15 and 23: trained h0 Leiden</title><style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:auto;padding:24px;color:#172033}img{max-width:100%}p{line-height:1.5}table{border-collapse:collapse;margin:16px 0}td,th{padding:5px 12px;border-bottom:1px solid #ddd;text-align:right}span{display:inline-block;width:14px;height:14px;margin-right:8px}section{break-inside:avoid}</style><main>'
    document += '<h1>Test cores 15 and 23 · Trained h0 Leiden</h1>'
    document += f'<p>{len(cells):,} cells, clustered jointly from the frozen 256-dimensional trained node encoder before graph blocks. h0 is learned from expression and morphology; it is not raw expression or an untrained baseline. All genes observed; training-fitted Pearson and morphology statistics unchanged. Exact cosine k=30, unweighted undirected union, Leiden resolution 1.0.</p>'
    document += f'<p>Seed 0: {metrics["cluster_count_by_seed"]["0"]} clusters. Seed 1: {metrics["cluster_count_by_seed"]["1"]} clusters. Between-seed adjusted Rand index: <strong>{metrics["seed0_seed1_ari"]:.4f}</strong>. Cross-core graph-edge fraction: {metrics["cross_core_undirected_edge_fraction"]:.4f}.</p>'
    per_core_ari = "; ".join(f"core {core}: {value:.4f}" for core, value in metrics["seed0_seed1_ari_per_core"].items())
    concentrated = ", ".join(f"h0-{value:02d}" for value in metrics["clusters_over_90_percent_one_core"]) or "none"
    document += f'<p>Within-core between-seed ARI: {per_core_ari}. Clusters with more than 90% of their cells from one core: {concentrated}. These are descriptive stability and composition checks, without a biological pass threshold.</p>'
    document += f'<p>{html.escape(LIMITATION)}</p><p>Every original cell polygon is shown; the two h0 core maps use shared h0 cluster IDs and colors. Colors and IDs are not aligned with the independently clustered hL maps. Coordinates are micrometres; y follows the CosMx image convention. Each scale bar is 100 µm. Core panel extents differ; scale bars specify physical scale.</p>'
    document += sections + '<h2>Shared color key</h2><table><tr><th>Cluster</th><th>Test cells</th></tr>' + color_rows + '</table>'
    document += '<h2>Cluster sizes</h2>' + sizes.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    document += '<h2>Core composition</h2><p>Fractions describe these two sections; cells are not independent biological replicates.</p>' + composition.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    document += f'<p>Source: {SOURCE_RUN_ID}; selected checkpoint SHA-256: {extraction["checkpoint_sha256"]}.</p></main></html>'
    (reports / "index.html").write_text(document)
    centroid = alignment.centroid_alignment
    _require(centroid is not None and centroid.aligned, "Polygon centroid audit is absent or failed")
    qc = {"cells": len(cells), "covered_cells_per_core": coverage, "actual_cell_polygons": True,
        "polygons_sampled": False, "slide_qualified_key_order_verified": True,
        "repaired_cells": int(alignment.repaired.sum()), "selected_vertex_rows": alignment.selected_vertex_rows,
        "source_rows_scanned": alignment.source_rows_scanned, "pixel_size_um": alignment.pixel_size_um,
        "centroid_alignment_rule": centroid.alignment_rule, "centroid_tolerance_um": centroid.tolerance_um,
        "centroid_mismatch_count": len(centroid.mismatch_indices),
        "centroid_tolerance_exceeded_count": len(centroid.centroid_tolerance_exceeded_indices),
        "containment_accepted_count": len(centroid.containment_accepted_indices),
        "coordinate_outside_polygon_count": len(centroid.coordinate_outside_polygon_indices),
        "median_centroid_distance_um": centroid.median_distance_um,
        "maximum_centroid_distance_um": centroid.maximum_distance_um,
        "shared_palette": True, "palette_source": palette_source, "export_checks": export_checks,
        "palette_scope": "h0_only_shared_across_test_cores", "cross_representation_label_alignment": False,
        "manual_visual_review": "not_performed"}
    dump(reports / "segmentation_qc.json", qc)
    manifest = {"schema": SCHEMA + "_figures", "status": "complete", "input_binding": bindings,
        "representation_name": "h0", "representation": REPRESENTATION,
        "clustering_manifest_sha256": sha(cluster_path), "raw_polygons": {"path": str(source), "sha256": sha(source)},
        "qc": qc, "figures": saved, "bounds_um": bounds, "scale_bar_um": 100,
        "equal_physical_aspect": True, "invert_y": True, "polygon_layer_rasterized_in_pdf": True,
        "html_embedded_previews": True, "interpretation": LIMITATION,
        "output_sha256": {str(p.relative_to(output)): sha(p) for directory in (figures, reports)
            for p in sorted(directory.iterdir()) if p.is_file() and p.name != "figure_manifest.json"}}
    total_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    _require(total_bytes <= DISK_BUDGET, "Test analysis exceeds the 0.5 GiB disk budget")
    _require(shutil.disk_usage(output).free >= DISK_RESERVE, "Disk reserve fell below 5 GiB")
    manifest["active_output_bytes"] = total_bytes
    dump(reports / "figure_manifest.json", manifest)
    _summary(output, manifest, reports / "figure_manifest.json", "rendering_summary.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("cluster", "render", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.stage in ("cluster", "all"):
        cluster(args.output, device=args.device, query_chunk=args.query_chunk, threads=args.threads)
    if args.stage in ("render", "all"):
        render(args.output)


if __name__ == "__main__":
    main()
