"""Portable, descriptive figures for the joint SO1/SO2 representation analysis.

All clustering is performed upstream on the full cohort. Sampling here affects
the embedding display only; spatial maps include every mapped observation.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "pca": "Pearson PCA",
    "kernel_pca": "RBF kernel PCA (training landmarks)",
    "spatial_pca": "Neighborhood-augmented spatial PCA",
}
PRIMARY_COLUMN = "leiden_r1.0_s0"
SECONDARY_SEED_COLUMN = "leiden_r1.0_s1"


def _palette(count: int) -> ListedColormap:
    # Cluster numbers are local to a method, so colors are not identities shared
    # between representations. Use a fixed hue ordering within each method.
    colors = plt.get_cmap("turbo")(np.linspace(0.04, 0.96, max(count, 2)))
    return ListedColormap(colors)


def _read_primary_labels(path: Path, expected_rows: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(path)
    if "row_index" not in frame or not np.array_equal(frame.row_index.to_numpy(), expected_rows):
        raise ValueError("Report cluster rows do not align with the observation index.")
    if PRIMARY_COLUMN not in frame:
        raise ValueError(f"Missing primary clustering column: {PRIMARY_COLUMN}")
    values = frame[PRIMARY_COLUMN].to_numpy()
    if not np.isfinite(values).all() or not np.array_equal(values, values.astype(np.int64)):
        raise ValueError("Primary cluster labels must be finite integers.")
    labels = values.astype(np.int64)
    if np.any(labels < 0):
        raise ValueError("Negative cluster labels are not supported.")
    return labels, frame


def _save_figure(figure, directory: Path, basename: str) -> dict[str, str]:
    png = directory / f"{basename}.png"
    pdf = directory / f"{basename}.pdf"
    figure.savefig(png, dpi=175, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return {"png": png.name, "pdf": pdf.name}


def _embedding_figure(methods: list[str], embeddings: dict[str, np.ndarray],
                      labels: dict[str, np.ndarray], sample: np.ndarray, directory: Path) -> dict:
    figure, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5), squeeze=False)
    for method, axis in zip(methods, axes.flat):
        values = embeddings[method]
        ids = labels[method]
        axis.scatter(values[sample, 0], values[sample, 1], c=ids[sample],
                     cmap=_palette(int(ids.max()) + 1), vmin=-0.5,
                     vmax=int(ids.max()) + 0.5, s=1.6, linewidths=0,
                     alpha=0.65, rasterized=True)
        axis.set_title(METHOD_LABELS.get(method, method), fontsize=11)
        axis.set_xlabel("Embedding dimension 1")
        axis.set_ylabel("Embedding dimension 2")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(f"Primary Leiden clusters · {len(sample):,} cells shown", fontsize=14)
    figure.tight_layout()
    return _save_figure(figure, directory, "embedding_dimensions")


def _spatial_figure(cells: pd.DataFrame, labels: np.ndarray, method: str, directory: Path) -> dict:
    cores = list(dict.fromkeys(cells.core))
    columns = min(7, len(cores))
    rows = math.ceil(len(cores) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 2.8 * rows), squeeze=False)
    cmap = _palette(int(labels.max()) + 1)
    for core, axis in zip(cores, axes.flat):
        selector = cells.core.to_numpy() == core
        subset = cells.loc[selector]
        # Subtract the local origin only for display; preserve physical scale
        # and aspect ratio inside each panel. Panels have separate extents.
        x = subset.x_um.to_numpy()
        y = subset.y_um.to_numpy()
        axis.scatter(x - x.min(), y - y.min(), c=labels[selector], cmap=cmap,
                     vmin=-0.5, vmax=int(labels.max()) + 0.5, s=0.9,
                     linewidths=0, rasterized=True)
        axis.set_aspect("equal")
        axis.set_title(f"{core} · {len(subset):,}", fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for axis in list(axes.flat)[len(cores):]:
        axis.set_visible(False)
    figure.suptitle(f"{METHOD_LABELS.get(method, method)} · all {len(cells):,} cells", fontsize=14)
    figure.tight_layout()
    return _save_figure(figure, directory, f"spatial_{method}")


def _composition_figure(cells: pd.DataFrame, methods: list[str],
                        labels: dict[str, np.ndarray], directory: Path) -> dict:
    figure, axes = plt.subplots(2, len(methods), figsize=(6 * len(methods), 10), squeeze=False)
    slides = sorted(cells.slide.unique())
    for column, method in enumerate(methods):
        ids, counts = np.unique(labels[method], return_counts=True)
        chosen = np.argsort(-counts, kind="stable")[:40]
        ids, counts = ids[chosen], counts[chosen]
        axis = axes[0, column]
        colors = _palette(int(labels[method].max()) + 1)
        axis.bar(np.arange(len(ids)), counts, color=colors(ids), width=0.8)
        axis.set_title(METHOD_LABELS.get(method, method), fontsize=10)
        axis.set_ylabel("Cells per cluster")
        axis.set_xlabel("Cluster number (ordered by size)")
        axis.set_xticks(np.arange(len(ids)), ids, rotation=90, fontsize=6)
        axis.spines[["top", "right"]].set_visible(False)
        proportions = np.array([
            [np.mean(cells.slide.to_numpy()[labels[method] == cluster] == slide)
             for cluster in ids] for slide in slides
        ])
        axis = axes[1, column]
        image = axis.imshow(proportions, aspect="auto", vmin=0, vmax=1, cmap="Blues")
        axis.set_yticks(np.arange(len(slides)), [f"Slide {slide}" for slide in slides])
        axis.set_xticks(np.arange(len(ids)), ids, rotation=90, fontsize=6)
        axis.set_xlabel("Same cluster order as above")
        axis.set_title("Slide fraction within each cluster", fontsize=10)
        figure.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    figure.suptitle("Cluster size and slide composition · largest 40 clusters per method", fontsize=14)
    figure.tight_layout()
    return _save_figure(figure, directory, "cluster_composition")


def _image_block(directory: Path, record: dict, title: str, caption: str) -> str:
    data = base64.b64encode((directory / record["png"]).read_bytes()).decode("ascii")
    return (f'<figure><h2>{html.escape(title)}</h2><img src="data:image/png;base64,{data}" '
            f'alt="{html.escape(caption, quote=True)}"><figcaption>{html.escape(caption)}</figcaption></figure>')


def _safe_metric_rows(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Expose compact numerical diagnostics, never arbitrary metadata rows."""
    rows = []
    if not isinstance(value, dict):
        return rows
    for key, entry in value.items():
        label = f"{prefix}.{key}" if prefix else str(key)
        if any(term in label.lower() for term in ("donor", "patient", "vendor", "cell_id")):
            continue
        if isinstance(entry, dict):
            rows.extend(_safe_metric_rows(entry, label))
        elif isinstance(entry, (int, float, bool)) and not isinstance(entry, str):
            rows.append((label, entry))
    return rows


def write_report(out: Path, config: dict, summary: dict) -> dict:
    """Read verified analysis products and write a self-contained HTML report."""
    out = Path(out)
    directory = out / "report"
    directory.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(out / "cells.csv.gz")
    required = {"row_index", "core", "slide", "x_um", "y_um"}
    if not required.issubset(cells.columns) or cells.row_index.duplicated().any():
        raise ValueError("Observation index is incomplete or duplicated.")
    if not np.isfinite(cells[["x_um", "y_um"]].to_numpy()).all():
        raise ValueError("Nonfinite spatial coordinates in report inputs.")
    methods = list(config.get("methods", METHOD_LABELS))
    expected_rows = cells.row_index.to_numpy()
    embeddings, labels, diagnostics = {}, {}, {}
    records = []
    for method in methods:
        folder = out / "methods" / method
        values = np.load(folder / "embedding.npy", mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[0] != len(cells) or values.shape[1] < 2:
            raise ValueError(f"{method} embedding shape does not match report observations.")
        if not np.isfinite(values).all():
            raise ValueError(f"{method} embedding contains nonfinite values.")
        embeddings[method] = values
        ids, _ = _read_primary_labels(folder / "clusters.csv.gz", expected_rows)
        labels[method] = ids
        diagnostic_path = folder / "diagnostics.json"
        diagnostics[method] = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
        _, counts = np.unique(ids, return_counts=True)
        records.append({"method": method, "cells": len(cells), "dimensions": values.shape[1],
                        "clusters_resolution_1_seed_0": len(counts),
                        "smallest_cluster": int(counts.min()),
                        "median_cluster_size": float(np.median(counts)),
                        "largest_cluster": int(counts.max())})
    table = pd.DataFrame(records)
    table.to_csv(directory / "report_summary.csv", index=False)
    seed = int(config.get("seed", 0))
    sample = np.sort(np.random.default_rng(seed).choice(len(cells), min(30_000, len(cells)), replace=False))
    figures = {"embedding": _embedding_figure(methods, embeddings, labels, sample, directory)}
    blocks = [_image_block(
        directory, figures["embedding"], "Representation views",
        "Only this display is sampled. The first two dimensions are a projection of the full embedding, "
        "not UMAP; overlap here does not imply overlap in every dimension. Colors denote each method's "
        "own Leiden clusters at resolution 1.0, seed 0. Equal cluster numbers across methods are unrelated.")]
    for method in methods:
        figures[f"spatial_{method}"] = _spatial_figure(cells, labels[method], method, directory)
        blocks.append(_image_block(
            directory, figures[f"spatial_{method}"], f"Spatial maps: {METHOD_LABELS.get(method, method)}",
            "All cells are displayed at their measured coordinates, with each core shown separately. "
            "Panels preserve their own spatial aspect ratio and have separate extents. No spatial "
            "neighbors cross a core or slide. Spatial smoothness is partly constructed by the spatial "
            "encoder and is not independent evidence of biological accuracy."))
    figures["composition"] = _composition_figure(cells, methods, labels, directory)
    blocks.append(_image_block(
        directory, figures["composition"], "Cluster sizes and slide composition",
        "All cells contribute to the counts. At most the largest 40 clusters are displayed for each "
        "method. Slide composition is descriptive: slide and tissue composition are confounded, so "
        "increased mixing is not an optimization target or proof of successful biological integration."))
    metric_blocks = []
    for method in methods:
        rows = _safe_metric_rows(diagnostics[method])
        if rows:
            frame = pd.DataFrame(rows, columns=["Diagnostic", "Value"])
            metric_blocks.append(f"<h3>{html.escape(METHOD_LABELS.get(method, method))}</h3>"
                                 + frame.to_html(index=False, escape=True, border=0))
    spatial_weight = config.get("spatial_weight", 0.3)
    spatial_neighbors = config.get("spatial_k", config.get("spatial_neighbors", 15))
    spatial_radius = config.get("spatial_radius_um", 50.0)
    landmarks = config.get("kernel_landmarks", config.get("n_landmarks", 512))
    source_run = html.escape(str(config.get("upstream_run_id", "recorded in the resolved configuration")))
    comparison = summary.get("pairwise_cluster_ari", {})
    comparison_html = ""
    if comparison:
        comparison_html = ("<h2>Agreement between representations</h2><p>Adjusted Rand index "
                           "compares the primary cluster partitions. Agreement does not establish "
                           "biological accuracy.</p>" + pd.DataFrame(
                               list(comparison.items()), columns=["Representations", "Adjusted Rand index"]
                           ).to_html(index=False, escape=True, border=0))
    limitations_html = ""
    if summary.get("limitations"):
        limitations_html = "<ul>" + "".join(
            f"<li>{html.escape(str(value))}</li>" for value in summary["limitations"]
        ) + "</ul>"
    narrative = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SO1 + SO2 embeddings and Leiden</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;color:#17212e;background:#fff;max-width:1400px;margin:32px auto;padding:0 24px}}
h1,h2,h3{{line-height:1.2}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:32px}}h3{{font-size:18px}}
figure{{margin:30px 0;break-inside:avoid}}img{{max-width:100%;height:auto}}figcaption{{color:#465365;font-size:14px}}
table{{border-collapse:collapse;margin:18px 0;font-size:13px;max-width:100%;overflow-wrap:anywhere}}
th,td{{border-bottom:1px solid #dce1e7;padding:7px 10px;text-align:left}}th{{background:#eff3f7}}
.note{{background:#f3f6fa;border-left:4px solid #52728e;padding:12px 18px}}
@media print{{body{{margin:0;max-width:none;font-size:11px}}h2{{page-break-after:avoid}}figure{{page-break-inside:avoid}}}}
</style></head><body><h1>SO1 + SO2 representations and Leiden clustering</h1>
<p>{len(cells):,} mapped cells across {cells.core.nunique()} cores and {cells.slide.nunique()} slides.
Three representations use the normalized, fully observed integrated expression data.
Clustering includes the full cohort; the train/validation partition is retained for fitting encoder parameters.</p>
<p class="note">Exploratory, transductive cell grouping. These clusters are not validated cell types, independent
patient-level evidence, or proof of batch removal. No active-training checkpoint supplies these representations.</p>
<p>Pearson PCA is the linear baseline. RBF kernel PCA nonlinearly transforms its scores using {landmarks}
core-balanced training landmarks and a frozen out-of-sample extension; it is an approximation to full kernel PCA.
The spatial representation combines cell expression and local neighborhood information with weight {spatial_weight},
up to {spatial_neighbors} neighbors and radius {spatial_radius} µm, separately within each core.
All graph and scale details are recorded in the resolved configuration and diagnostics.</p>
<p>Primary display: Leiden resolution 1.0, seed 0. Resolution 0.5/1.5 and seed 1 at resolution 1.0
are retained as sensitivity analyses. No best resolution or seed is selected from the displays.</p>
{table.to_html(index=False, escape=True, border=0)}
{''.join(blocks)}
{comparison_html}
<h2>Recorded numerical diagnostics</h2>{''.join(metric_blocks)}
<h2>Interpretation and provenance</h2>
{limitations_html}
<p>Clustering separation is partly induced by the representation and graph. Spatial smoothness may reflect
smoothing, segmentation spillover, or tissue structure. RNA depth, slide composition, graph-neighbor accuracy,
resolution sensitivity and seed stability should be considered together; none is a biological ground truth.</p>
<p>The normalized Pearson values are signed residuals. scVI's count model would require the raw-count input
and separate training, so it is not fitted to these residuals. Existing GPU training remains separate.</p>
<p>Upstream preprocessing run: <code>{source_run}</code>. The source dataset retains all 1,000 biological
probes and excludes only the previously unmapped 340 cells from SO2 FOV246. Vendor annotations and direct donor
identifiers do not enter these encoders or these figures. This report embeds every displayed image and is portable;
matching PNG and PDF exports accompany it.</p>
</body></html>"""
    destination = directory / "index.html"
    destination.write_text(narrative, encoding="utf-8")
    receipt = {"html": "report/index.html", "summary": "report/report_summary.csv",
               "figures": figures, "embedding_display_cells": len(sample),
               "spatial_display_cells_per_method": len(cells), "sampling_affects_clustering": False}
    (directory / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
