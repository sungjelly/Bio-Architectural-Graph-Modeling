"""Matched cell-polygon maps for model h0/hL and existing SO12 PCA clusters.

Rendering is post-hoc only. Labels remain method-specific and the existing PCA
partitions are joined by slide-qualified cell keys, never by row position.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex, to_rgba
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from spatial_benchmark.attention_niche_geometry import load_aligned_cosmx_polygons
from spatial_benchmark.data import DEFAULT_PIXEL_SIZE_UM, resolve_nested_raw_path
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths


EXPECTED_CELLS = 407_659
DEFAULT_CORES = (1, 9, 13, 15, 21, 23)
EXPECTED_CORE_COUNTS = {1: 8924, 9: 17223, 13: 5345, 15: 38145, 21: 4897, 23: 43462}
METHODS = ("h0", "hL", "pca", "kernel_pca", "spatial_pca")
METHOD_NAMES = {
    "h0": "h0 · initial embedding", "hL": "hL · layer 4",
    "pca": "Pearson PCA", "kernel_pca": "RBF kernel PCA",
    "spatial_pca": "Spatial PCA",
}
NAMESPACES = {"h0": "h0", "hL": "hL", "pca": "PCA", "kernel_pca": "RBF", "spatial_pca": "SPCA"}
KEYS = ["slide", "fov", "cell_ID"]
PRIMARY_COLUMN = "leiden_r1.0_s0"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _validate_labels(values: np.ndarray, expected_cells: int, method: str) -> np.ndarray:
    labels = np.asarray(values)
    _require(labels.shape == (expected_cells,), f"{method}: label coverage does not match cells")
    _require(np.issubdtype(labels.dtype, np.integer), f"{method}: labels must be integers")
    _require(len(labels) > 0 and labels.min() == 0, f"{method}: labels must start at zero")
    _require(np.array_equal(np.unique(labels), np.arange(int(labels.max()) + 1)),
             f"{method}: cluster IDs must be contiguous")
    return labels.astype(np.int32, copy=False)


def _validate_cells(cells: pd.DataFrame, expected_cells: int) -> None:
    required = set(KEYS + ["row_index", "x_um", "y_um"])
    _require(required.issubset(cells.columns), "Observation table lacks identity or coordinates")
    _require(len(cells) == expected_cells, "Observation row coverage changed")
    _require(not cells[KEYS].isna().any().any() and not cells.duplicated(KEYS).any(),
             "Observation table has missing or duplicate slide-qualified keys")
    _require(not cells.row_index.isna().any() and not cells.row_index.duplicated().any(),
             "Observation row_index is missing or duplicated")
    for name in KEYS + ["row_index"]:
        _require(np.issubdtype(cells[name].dtype, np.integer), f"{name} must use integer identities")
    _require(set(cells.slide).issubset({1, 2}), "Unexpected slide identity")
    _require(np.isfinite(cells[["x_um", "y_um"]].to_numpy()).all(), "Nonfinite coordinates")


def align_source_labels(target_cells: pd.DataFrame, source_cells: pd.DataFrame,
                        source_labels: pd.DataFrame, *, expected_cells: int = EXPECTED_CELLS) -> np.ndarray:
    """Exact one-to-one key join, preserving target order and checking geometry."""
    _validate_cells(target_cells, expected_cells)
    _validate_cells(source_cells, expected_cells)
    _require(set(["row_index", PRIMARY_COLUMN]).issubset(source_labels.columns),
             "PCA cluster table lacks primary labels or row_index")
    _require(len(source_labels) == expected_cells and not source_labels.row_index.duplicated().any(),
             "PCA cluster row coverage or uniqueness failed")
    _require(set(source_labels.row_index) == set(source_cells.row_index),
             "PCA clusters and source observation row identities differ")
    _validate_labels(source_labels[PRIMARY_COLUMN].to_numpy(), expected_cells, "PCA source")
    linked = source_cells[KEYS + ["row_index", "x_um", "y_um"]].merge(
        source_labels[["row_index", PRIMARY_COLUMN]], on="row_index", validate="one_to_one")
    linked = linked.rename(columns={"x_um": "source_x_um", "y_um": "source_y_um"}).drop(columns="row_index")
    ordered = target_cells[KEYS + ["x_um", "y_um"]].copy()
    ordered["_target_order"] = np.arange(expected_cells)
    joined = ordered.merge(linked, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    _require(len(joined) == expected_cells and joined["_merge"].eq("both").all(),
             "PCA and target slide-qualified cell key coverage differs")
    joined = joined.sort_values("_target_order", kind="stable")
    _require(np.allclose(joined[["x_um", "y_um"]].to_numpy(),
                        joined[["source_x_um", "source_y_um"]].to_numpy(), rtol=0, atol=1e-8),
             "PCA and target keyed coordinates differ")
    return _validate_labels(joined[PRIMARY_COLUMN].to_numpy(), expected_cells, "PCA joined")


def color_keys(labels: np.ndarray, method: str, existing: pd.DataFrame | None = None) -> tuple[dict[int, str], str]:
    """Preserve verified historical turbo colors; never silently omit a label."""
    labels = _validate_labels(labels, len(labels), method)
    count = int(labels.max()) + 1
    reconstructed = {index: to_hex(color) for index, color in enumerate(
        plt.get_cmap("turbo")(np.linspace(0.04, 0.96, max(count, 2)))[:count])}
    if method in ("h0", "hL"):
        # Use the same deterministic scheme, without implying identity across methods.
        return reconstructed, "deterministic_turbo_0.04_to_0.96"
    if existing is not None:
        _require({"method", "cluster_id", "color_hex"}.issubset(existing.columns),
                 "Historical color key schema is invalid")
        selected = existing.loc[existing.method == method]
        _require(not selected.cluster_id.duplicated().any(), f"{method}: duplicate color key")
        if set(selected.cluster_id) == set(reconstructed):
            supplied = {int(row.cluster_id): to_hex(to_rgba(row.color_hex))
                        for row in selected.itertuples()}
            if supplied == reconstructed:
                return supplied, "existing_color_keys_verified_against_original_palette"
        return reconstructed, "reconstructed_original_palette_existing_key_inconsistent"
    return reconstructed, "reconstructed_original_palette_no_existing_key"


def polygon_vertices(polygons: Sequence) -> tuple[np.ndarray, ...]:
    """Fail rather than lose holes or MultiPolygon pieces in this simple renderer."""
    result = []
    for polygon in polygons:
        _require(polygon.geom_type == "Polygon" and len(polygon.interiors) == 0,
                 "Simple segmentation renderer cannot discard holes or MultiPolygon parts")
        _require(polygon.is_valid and not polygon.is_empty, "Invalid segmentation polygon")
        vertices = np.asarray(polygon.exterior.coords, dtype=np.float64)
        _require(np.isfinite(vertices).all(), "Nonfinite segmentation vertices")
        result.append(vertices)
    return tuple(result)


def compare_partitions(labels: dict[str, np.ndarray], core_numbers: np.ndarray,
                       cores: Sequence[int] = DEFAULT_CORES) -> dict:
    """Describe existing partitions on identical cells, without fitting anything."""
    core_numbers = np.asarray(core_numbers)
    _require(core_numbers.ndim == 1, "Core identities must be one-dimensional")
    _require(set(labels) == set(METHODS) and all(np.asarray(x).shape == core_numbers.shape
             for x in labels.values()), "Comparison label arrays do not share cell coverage")
    populations = {"all_cells": np.ones(len(core_numbers), dtype=bool),
                   "requested_cores": np.isin(core_numbers, cores)}
    populations.update({f"core_{core:02d}": core_numbers == core for core in cores})
    results = {}
    for population, selected in populations.items():
        _require(selected.any(), f"Comparison population is empty: {population}")
        matrix = np.eye(len(METHODS), dtype=np.float64)
        subset = {method: labels[method][selected] for method in METHODS}
        for first in range(len(METHODS)):
            for second in range(first + 1, len(METHODS)):
                matrix[first, second] = matrix[second, first] = adjusted_rand_score(
                    subset[METHODS[first]], subset[METHODS[second]])
        results[population] = {"cells": int(selected.sum()),
            "observed_cluster_counts": {method: int(len(np.unique(values))) for method, values in subset.items()},
            "pairwise_ari": matrix.tolist()}
    return {"schema": "so12_five_method_partition_comparison_v1", "method_order": list(METHODS),
        "metric": "adjusted_rand_index", "populations": results,
        "interpretation": "ARI measures agreement of existing partitions on the same cells; 1 is identical membership up to relabeling, approximately 0 is chance-level agreement, and negative values are possible. It is not a biological-validity or model-performance ranking.",
        "pipeline_comparability": {
            "shared": "All 407659 mapped cells; primary Leiden resolution 1.0 and seed 0. Subsets retain the full-cohort labels; no subset is reclustered.",
            "h0_hL": "256-dimensional direct hidden representations; cosine k30 exhaustive FP32 neighbor search; unweighted undirected-union graph; Leiden until convergence.",
            "previous_PCA_methods": "32-dimensional encoders; Euclidean k30 approximate HNSW neighbors; exponential distance weights and arithmetic-mean symmetrization; Leiden for two iterations. Original partitions are reused.",
            "spatial_PCA": "Training-fitted neighborhood-augmented PCA; 15 neighbors within 50 micrometres, within the same core and slide, spatial weight 0.3.",
            "limitation": "Observed differences combine representation and clustering-pipeline differences. These comparisons are not a controlled encoder ablation. Cluster IDs and colors are independent across methods."}}


def _comparison_html(comparison: dict) -> str:
    pieces = ['<h2>Partition agreement and cluster counts</h2><p>' +
              html.escape(comparison["interpretation"]) + '</p>']
    pieces.append('<p>' + html.escape(comparison["pipeline_comparability"]["limitation"]) + '</p>')
    header = ''.join(f'<th>{NAMESPACES[method]}</th>' for method in METHODS)
    title = {"all_cells": "All mapped cells", "requested_cores": "Six displayed cores pooled"}
    tables = []
    for population, values in comparison["populations"].items():
        label = title.get(population, f'Core {int(population.removeprefix("core_"))}' if population.startswith('core_') else population)
        rows = []
        for method, row in zip(METHODS, values["pairwise_ari"], strict=True):
            rows.append(f'<tr><th>{NAMESPACES[method]}</th>' + ''.join(f'<td>{value:.3f}</td>' for value in row) + '</tr>')
        counts = ''.join(f'<td>{values["observed_cluster_counts"][method]}</td>' for method in METHODS)
        tables.append(f'<section><h3>{label} · {values["cells"]:,} cells</h3><table><thead><tr><th>ARI</th>{header}</tr></thead>' +
                      '<tbody>' + ''.join(rows) + f'<tr><th>Clusters present</th>{counts}</tr></tbody></table></section>')
    pieces.append('<div class="comparisons">' + ''.join(tables) + '</div>')
    pieces.append('<h3>Pipeline differences retained in this comparison</h3><table><thead><tr><th>Pipeline</th><th>Definition</th></tr></thead><tbody>')
    for label, key in (("Shared", "shared"), ("h0 and hL", "h0_hL"), ("Previous PCA methods", "previous_PCA_methods"), ("Spatial PCA encoder", "spatial_PCA")):
        pieces.append(f'<tr><th>{label}</th><td style="text-align:left">{html.escape(comparison["pipeline_comparability"][key])}</td></tr>')
    pieces.append('</tbody></table>')
    return ''.join(pieces)


def _bounds(vertices: Sequence[np.ndarray]) -> tuple[float, float, float, float]:
    points = np.vstack(vertices)
    low, high = points.min(axis=0), points.max(axis=0)
    pad = max(float((high - low).max()) * .025, 12.0)
    return float(low[0]-pad), float(high[0]+pad), float(low[1]-pad), float(high[1]+pad)


def _draw(axis, vertices: Sequence[np.ndarray], labels: np.ndarray, palette: dict[int, str],
          bounds: tuple[float, float, float, float]) -> None:
    _require(len(vertices) == len(labels) and set(np.unique(labels)).issubset(palette),
             "Polygon/label/color coverage differs")
    collection = PolyCollection(vertices, facecolors=[palette[int(label)] for label in labels],
                                edgecolors="none", linewidths=0, antialiaseds=False, rasterized=True)
    axis.add_collection(collection)
    xmin, xmax, ymin, ymax = bounds
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymax, ymin)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    # Fixed physical length, positioned in whitespace at the lower left.
    x = xmin + .045 * (xmax-xmin)
    y = ymax - .035 * (ymax-ymin)
    axis.plot([x, x+100], [y, y], color="#111827", linewidth=2.0, solid_capstyle="butt", zorder=3)
    axis.text(x+50, y-.018*(ymax-ymin), "100 µm", ha="center", va="bottom", fontsize=7, zorder=4,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 0.5})


def _save(figure, directory: Path, basename: str, dpi: int) -> dict:
    paths = {extension: directory / f"{basename}.{extension}" for extension in ("png", "pdf")}
    try:
        for extension, path in paths.items():
            figure.savefig(path, format=extension, dpi=dpi, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(figure)
    return {extension: path.name for extension, path in paths.items()}


def _legend_figure(table: pd.DataFrame, figure_dir: Path) -> dict:
    maximum = int(table.groupby("method").size().max())
    figure, axes = plt.subplots(1, 5, figsize=(18, max(5, .24*maximum+1.5)))
    for method, axis in zip(METHODS, axes, strict=True):
        rows = table.loc[table.method == method]
        axis.set_title(METHOD_NAMES[method], fontsize=11)
        axis.set_xlim(0, 1)
        axis.set_ylim(maximum+1, -1)
        axis.axis("off")
        for index, row in enumerate(rows.itertuples()):
            axis.scatter(.05, index, s=80, c=row.color_hex, marker="s", edgecolors="none")
            axis.text(.12, index, f"{row.cluster_label}   {row.selected_cells:,}", fontsize=8, va="center")
    figure.suptitle("Complete cluster keys · counts in the six displayed cores", fontsize=14)
    figure.text(.5, .015, "Cluster IDs and colors identify groups within a method only; they are not matched across methods.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .035, 1, .96))
    return _save(figure, figure_dir, "cluster_color_keys", 220)


def render_comparison(out: Path, pca_run_dir: Path, cores: Sequence[int] = DEFAULT_CORES) -> dict:
    """Write matched polygon PNG/PDF maps, all legends, portable HTML and QC.

    Requires ``cells.parquet`` (all 407659 cells), ``clustering/h0_labels.npy``,
    ``clustering/hL_labels.npy`` and ``clustering/summary.json`` under ``out``.
    Writes only new files within ``out/figures`` and ``out/report``.
    """
    out, pca_run_dir = Path(out), Path(pca_run_dir)
    cores = tuple(int(core) for core in cores)
    _require(cores == DEFAULT_CORES, "This comparison is locked to cores 1, 9, 13, 15, 21, 23")
    figure_dir, report_dir = out / "figures", out / "report"
    for directory in (figure_dir, report_dir):
        _require(not directory.exists() or not any(directory.iterdir()),
                 f"Refusing to overwrite existing report output: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    paths = current_paths()
    cell_path = out / "cells.parquet"
    cells = pd.read_parquet(cell_path)
    _validate_cells(cells, EXPECTED_CELLS)
    _require({"core_number", "alias", "split"}.issubset(cells.columns), "Missing core or split columns")
    _require(np.array_equal(cells.row_index.to_numpy(), np.arange(EXPECTED_CELLS)),
             "Root observation rows do not follow all-cell embedding order")
    expected_alias = [f"SO{slide}-C{number:02d}" for slide, number in zip(cells.slide, cells.core_number)]
    _require(np.array_equal(cells.alias.to_numpy(), expected_alias), "Core aliases disagree with slide or core number")
    labels = {method: _validate_labels(np.load(out / "clustering" / f"{method}_labels.npy", allow_pickle=False),
                                      len(cells), method) for method in ("h0", "hL")}
    summary_path = out / "clustering" / "summary.json"
    json.loads(summary_path.read_text())  # Require complete, parseable clustering receipt.
    checksums = json.loads((pca_run_dir / "checksums.json").read_text())
    source_records = {}
    for relative in ("_SUCCESS", "cells.csv.gz", "resolved_config.json", *[f"methods/{m}/clusters.csv.gz" for m in METHODS[2:]]):
        source = pca_run_dir / relative
        record = _record(source)
        _require(relative in checksums and record["sha256"] == checksums[relative]["sha256"]
                 and record["size_bytes"] == checksums[relative]["size_bytes"],
                 f"Immutable PCA input checksum failed: {relative}")
        source_records[relative] = record
    pca_cells = pd.read_csv(pca_run_dir / "cells.csv.gz")
    identity_check = cells[KEYS + ["alias", "split"]].merge(
        pca_cells[KEYS + ["core", "train"]], on=KEYS, validate="one_to_one", how="outer", indicator=True)
    _require(len(identity_check) == EXPECTED_CELLS and identity_check["_merge"].eq("both").all(),
             "PCA and model cell keys differ")
    _require(identity_check.alias.eq(identity_check.core).all(), "PCA and model keyed core assignments differ")
    _require(set(cells.split).issubset({"train", "validation"}) and
             np.array_equal(identity_check.split.eq("train").to_numpy(), identity_check.train.to_numpy()),
             "PCA and model keyed training/validation splits differ")
    for method in METHODS[2:]:
        labels[method] = align_source_labels(cells, pca_cells,
            pd.read_csv(pca_run_dir / "methods" / method / "clusters.csv.gz"))
    comparison = compare_partitions(labels, cells.core_number.to_numpy(), cores)
    _write_json(report_dir / "comparison_metrics.json", comparison)
    pair_rows, count_rows = [], []
    for population, values in comparison["populations"].items():
        for first, method in enumerate(METHODS):
            count_rows.append({"population": population, "cells": values["cells"], "method": method,
                               "clusters_present": values["observed_cluster_counts"][method]})
            for second, other in enumerate(METHODS):
                pair_rows.append({"population": population, "cells": values["cells"], "method_1": method,
                                  "method_2": other, "adjusted_rand_index": values["pairwise_ari"][first][second]})
    pd.DataFrame(pair_rows).to_csv(report_dir / "comparison_ari.csv", index=False)
    pd.DataFrame(count_rows).to_csv(report_dir / "comparison_cluster_counts.csv", index=False)
    color_path = paths.report_root / "analyses" / "so12_embeddings_leiden" / "cluster_color_keys.csv"
    prior_colors = pd.read_csv(color_path) if color_path.is_file() else None
    palettes, color_sources = {}, {}
    for method in METHODS:
        palettes[method], color_sources[method] = color_keys(labels[method], method, prior_colors)
    selected_mask = cells.core_number.isin(cores).to_numpy()
    selected = cells.loc[selected_mask].copy()
    _require(len(selected) == 117_996, "Requested core coverage differs from the locked cohort")
    _require(selected.groupby("core_number").size().to_dict() == EXPECTED_CORE_COUNTS,
             "Requested per-core cell counts differ from the locked cohort")
    polygon_paths = {f"SO_{slide}": resolve_nested_raw_path(paths.data_root / "raw", f"26040302SO_{slide}-polygons.csv")
                     for slide in (1, 2)}
    alignment = load_aligned_cosmx_polygons(polygon_paths, selected[KEYS],
        prepared_coordinates_um=selected[["x_um", "y_um"]].to_numpy(), centroid_tolerance_um=5.0)
    vertices = polygon_vertices(alignment.polygons_um)
    _require(alignment.n_cells == len(selected), "Segmentation polygon coverage failed")
    selected_labels = {method: values[selected_mask] for method, values in labels.items()}
    keyed_colors = []
    for method in METHODS:
        counts = np.bincount(labels[method], minlength=len(palettes[method]))
        displayed = np.bincount(selected_labels[method], minlength=len(palettes[method]))
        for cluster, color in palettes[method].items():
            keyed_colors.append({"method": method, "cluster_id": cluster,
                "cluster_label": f"{NAMESPACES[method]}-{cluster:02d}", "color_hex": color,
                "all_cells": int(counts[cluster]), "selected_cells": int(displayed[cluster])})
    table = pd.DataFrame(keyed_colors)
    table.to_csv(report_dir / "cluster_color_keys.csv", index=False)
    table.to_json(report_dir / "cluster_color_keys.json", orient="records", indent=2)
    core_vertices, core_indices, bounds, counts = {}, {}, {}, {}
    for core in cores:
        indices = np.flatnonzero(selected.core_number.to_numpy() == core)
        core_indices[core] = indices
        core_vertices[core] = tuple(vertices[index] for index in indices)
        bounds[core] = _bounds(core_vertices[core])
        counts[core] = len(indices)
    saved = {}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        for core in cores:
            figure, axes = plt.subplots(1, 5, figsize=(22, 4.8))
            for method, axis in zip(METHODS, axes, strict=True):
                _draw(axis, core_vertices[core], selected_labels[method][core_indices[core]], palettes[method], bounds[core])
                axis.set_title(METHOD_NAMES[method], fontsize=12)
            split = str(selected.loc[selected.core_number == core, "split"].iloc[0])
            figure.suptitle(f"Core {core} · {counts[core]:,} cells · {split}", fontsize=16)
            figure.text(.5, .015, "All genes observed · Actual cell segmentation polygons · Cluster IDs/colors are independent across methods",
                        ha="center", fontsize=10)
            figure.tight_layout(rect=(0, .05, 1, .93))
            saved[f"core_{core:02d}"] = _save(figure, figure_dir, f"core_{core:02d}_five_method_segmentation", 300)
        figure, axes = plt.subplots(6, 5, figsize=(22, 26), squeeze=False)
        for row, core in enumerate(cores):
            for column, method in enumerate(METHODS):
                axis = axes[row, column]
                _draw(axis, core_vertices[core], selected_labels[method][core_indices[core]], palettes[method], bounds[core])
                axis.set_title(f"Core {core} · {METHOD_NAMES[method]}", fontsize=10)
        figure.suptitle("Same cells, five representations · SO1 + SO2", fontsize=20)
        figure.text(.5, .007, "All genes observed · Raw cell polygons · All scale bars 100 µm · Cluster IDs/colors are method-specific",
                    ha="center", fontsize=11)
        figure.tight_layout(rect=(0, .02, 1, .975))
        saved["overview"] = _save(figure, figure_dir, "six_core_five_method_segmentation", 220)
        saved["legend"] = _legend_figure(table, figure_dir)
    sections = []
    for core in cores:
        path = figure_dir / saved[f"core_{core:02d}"]["png"]
        # The portable report embeds reduced previews; full-resolution PNG/PDF are separate artifacts.
        from PIL import Image
        import io
        with Image.open(path) as image:
            image.thumbnail((2400, 1200))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        sections.append(f'<section><h2>Core {core} · {counts[core]:,} cells</h2>'
                        f'<img src="data:image/png;base64,{encoded}" alt="Core {core}: h0, hL, PCA, RBF kernel PCA and Spatial PCA cell segmentation maps"></section>')
    legend_sections = []
    for method in METHODS:
        rows = table.loc[table.method == method]
        entries = "".join(f'<tr><td><span class="swatch" style="background:{row.color_hex}"></span>'
                          f'{html.escape(row.cluster_label)}</td><td>{row.selected_cells:,}</td><td>{row.all_cells:,}</td></tr>'
                          for row in rows.itertuples())
        legend_sections.append(f'<section><h3>{html.escape(METHOD_NAMES[method])}</h3><table>'
            '<thead><tr><th>Cluster</th><th>Shown</th><th>All cells</th></tr></thead><tbody>'+entries+'</tbody></table></section>')
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    document += '<title>SO12 model and PCA cell segmentation comparison</title><style>'
    document += 'body{font-family:system-ui,sans-serif;color:#172033;max-width:1600px;margin:auto;padding:24px;background:white}img{width:100%;height:auto}h1{font-size:28px}p{line-height:1.5;max-width:1100px}.legends{display:grid;grid-template-columns:repeat(5,minmax(170px,1fr));gap:20px}.comparisons{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:24px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:4px;text-align:right;border-bottom:1px solid #e2e8f0}td:first-child,th:first-child{text-align:left}.swatch{display:inline-block;width:14px;height:14px;margin-right:6px;vertical-align:middle}@media(max-width:900px){.legends{grid-template-columns:repeat(2,minmax(170px,1fr))}.comparisons{display:block}}@media print{section{break-inside:avoid}body{padding:0}.legends{display:block}.legends section{display:inline-block;vertical-align:top;width:19%}}'
    document += '</style><main><h1>Cell segmentation: h0, hL and the previous PCA methods</h1>'
    document += '<p>All observed genes are supplied to the frozen four-layer SO1 + SO2 model. h0 is the initial embedding; hL is its final hidden state. The three PCA columns reuse their existing all-cell Leiden assignments at resolution 1.0, seed 0. No PCA model or partition was refitted.</p>'
    document += '<p>Every panel colors the same original cell polygons, with matching physical bounds, orientation and 100 µm scale bars. Colors and cluster IDs are independent across methods; the same color does not mean the same cell group. Each method uses one shared palette across all six cores. Historical PCA colors are preserved.</p>'
    document += '<p>Core 13 belongs to the model validation partition; cores 1, 9, 15, 21 and 23 belong to training. Clustering is exploratory and transductive. These maps are descriptive model-derived groups, not validated cell types, evidence of graph-specific predictive benefit, or causal interactions. Spatial PCA explicitly includes neighborhood information.</p>'
    document += ''.join(sections) + _comparison_html(comparison) + '<h2>Complete color keys</h2><p>Shown counts refer to the six displayed cores; all-cell counts refer to 407,659 cells. Clusters absent from these cores remain in the key.</p><div class="legends">'+''.join(legend_sections)+'</div></main></html>'
    (report_dir / "index.html").write_text(document)
    centroid = alignment.centroid_alignment
    qc = {
        "all_cell_count": len(cells), "selected_cell_count": len(selected), "cells_per_core": counts,
        "slide_qualified_join": {"keys": KEYS, "one_to_one": True, "all_source_and_target_cells_matched": True,
                                  "coordinate_atol_um": 1e-8},
        "polygons": {"covered_cells": alignment.n_cells, "selected_vertex_rows": alignment.selected_vertex_rows,
            "source_rows_scanned": alignment.source_rows_scanned, "repaired_cells": int(alignment.repaired.sum()),
            "simple_polygon_without_holes_count": len(vertices), "pixel_size_um": DEFAULT_PIXEL_SIZE_UM,
            "alignment_rule": centroid.alignment_rule, "centroid_tolerance_um": centroid.tolerance_um,
            "alignment_mismatches": len(centroid.mismatch_indices),
            "centroid_tolerance_exceeded": len(centroid.centroid_tolerance_exceeded_indices),
            "containment_accepted": len(centroid.containment_accepted_indices),
            "coordinate_outside_polygon_count": len(centroid.coordinate_outside_polygon_indices),
            "median_centroid_distance_um": centroid.median_distance_um,
            "maximum_centroid_distance_um": centroid.maximum_distance_um},
        "cluster_counts": {method: len(palette) for method, palette in palettes.items()},
        "color_sources": color_sources, "polygons_sampled": False,
    }
    _write_json(report_dir / "segmentation_qc.json", qc)
    manifest = {"schema": "so12_hidden_and_pca_segmentation_comparison_v1", "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(), "methods": list(METHODS), "cores": list(cores),
        "inputs": {"cells": _record(cell_path), "h0_labels": _record(out / "clustering/h0_labels.npy"),
            "hL_labels": _record(out / "clustering/hL_labels.npy"), "clustering_summary": _record(summary_path),
            "pca_run_dir": str(pca_run_dir.resolve()), "pca_sources": source_records,
            "pca_checksums": _record(pca_run_dir / "checksums.json"),
            "historical_color_keys": _record(color_path) if color_path.is_file() else None,
            "raw_polygons": {slide: _record(path) for slide, path in polygon_paths.items()},
            "renderer": _record(Path(__file__))},
        "qc": qc, "figures": saved, "partition_comparison": comparison,
        "rendering": {"actual_cell_polygons": True, "shared_bounds_per_core": bounds,
            "equal_physical_aspect": True, "invert_y": True,
            "orientation_basis": "repository CosMx global-image plotting convention",
            "coordinate_units": "micrometres", "scale_bar_um": 100, "per_core_png_dpi": 300,
            "overview_png_dpi": 220, "polygon_layer_rasterized_in_pdf": True,
            "method_namespaces_independent": True, "html_portable_embedded_images": True},
        "files": {str(path.relative_to(out)): _record(path)
            for directory in (figure_dir, report_dir) for path in sorted(directory.iterdir()) if path.is_file()}}
    _write_json(report_dir / "figure_manifest.json", manifest)
    return manifest
