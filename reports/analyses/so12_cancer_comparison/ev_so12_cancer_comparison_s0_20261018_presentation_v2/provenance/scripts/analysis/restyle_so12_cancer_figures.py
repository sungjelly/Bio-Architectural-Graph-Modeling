"""Render a presentation-only revision from the frozen comparison inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
import numpy as np
import pandas as pd
from PIL import Image

from so12_cancer_figures import (
    CORES, COUNTS, PRIMARY, TITLES, _panel, _save, color_raster,
    dump, load_partitions, sha,
)
from spatial_benchmark.paths import current_paths

# Alternate restrained hues instead of adjacent shades on a continuous rainbow.
# The lookup depends only on cluster ID, never region membership or method score.
COLORS = (
    "#4477AA", "#CC6677", "#228877", "#AA8833", "#8877AA",
    "#66AACC", "#AA7744", "#77AA66", "#AA4499", "#DD9966",
    "#667788", "#44AAAA", "#BB5544", "#999944", "#5566AA",
    "#BB88AA", "#88AAAA", "#BBAA77", "#777799", "#99AA44",
)
OUTLINE_COLOR = "#161616"
OUTLINE_WIDTH = 1.45
HALO_WIDTH = 2.15


def categorical_palette(labels):
    count = int(labels.max()) + 1
    if count > len(COLORS):
        raise ValueError("More clusters than distinct presentation colors")
    return np.rint(np.array([to_rgb(c) for c in COLORS[:count]]) * 255).astype(np.uint8)


def panel(axis, rgb, roi, core, title):
    _panel(axis, rgb, roi, core, title, annotation=False)
    # Overlay the same reviewed contour on histology and all cluster maps.
    # The halo covers the baked-in blue annotation without changing its source.
    axis.contour(roi, levels=[.5], colors=["white"], linewidths=HALO_WIDTH)
    axis.contour(roi, levels=[.5], colors=[OUTLINE_COLOR], linewidths=OUTLINE_WIDTH)


def render(source, output):
    paths = current_paths()
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or not output.is_relative_to(paths.scratch_root / "active_runs"):
        raise ValueError("Write the presentation revision to a separate active directory")
    figures = output / "figures"
    if figures.exists() and any(figures.iterdir()):
        raise ValueError("Presentation output must be empty")
    registration_path = source / "registration/registration_manifest.json"
    registration = json.loads(registration_path.read_text())
    if not registration.get("passed") or registration.get("status") != "verified_approximate_section":
        raise ValueError("Reviewed registration is required")
    if sha(source / "cells.csv.gz") != registration["canonical_cells"]["sha256"]:
        raise ValueError("Canonical cells changed")
    selected = load_partitions(source)
    palettes = {method: categorical_palette(labels) for method, (labels, _) in selected.items()}
    figures.mkdir(parents=True)
    color_keys = []
    for method, (labels, _) in selected.items():
        for cluster, color in enumerate(palettes[method]):
            color_keys.append({"method": method, "cluster_id": cluster,
                               "color_hex": to_hex(color / 255),
                               "cells": int(np.count_nonzero(labels == cluster))})
    pd.DataFrame(color_keys).to_csv(figures / "cluster_color_keys.csv", index=False)

    records, images, rois, audits = {}, {}, {}, []
    for item in registration["cores"]:
        core = item["core_number"]
        raster = source / item["raster_path"]
        if sha(raster) != item["raster_sha256"]:
            raise ValueError("Reviewed owner raster changed")
        with np.load(raster, allow_pickle=False) as arrays:
            owner, roi = arrays["owner"], arrays["roi"]
        image_path = Path(item["image"]["path"])
        if sha(image_path) != item["image"]["sha256"]:
            raise ValueError("Clinical image changed")
        with Image.open(image_path) as image:
            images[core] = {"gt": np.asarray(image.convert("RGB"))}
        rois[core] = roi
        occupied = owner >= 0
        for method in PRIMARY[1:]:
            labels = selected[method][0]
            rgb = color_raster(owner, labels, palettes[method])
            # Verify the presentation preserves every cell pixel and cluster ID.
            if not np.array_equal(rgb[occupied], palettes[method][labels[owner[occupied]]]):
                raise ValueError("Rendered colors do not match cluster IDs")
            if not np.all(rgb[~occupied] == 255):
                raise ValueError("Uncovered pixels changed")
            if len(np.unique(palettes[method], axis=0)) != len(palettes[method]):
                raise ValueError("Cluster colors must be distinct")
            images[core][method] = rgb
        audits.append({"core": core, "raster_sha256": sha(raster),
                       "histology_sha256": sha(image_path),
                       "occupied_pixels": int(occupied.sum()),
                       "all_cluster_pixels_preserved": True})
        figure, axes = plt.subplots(1, 5, figsize=(18, 6.5))
        for axis, method in zip(axes, PRIMARY, strict=True):
            panel(axis, images[core][method], roi, core, TITLES[method])
        figure.suptitle(f"Core {core} · {COUNTS[core]:,} cells · Leiden resolution 1, seed 0",
                       fontsize=15, y=.98)
        figure.text(.5, .015,
                    "Black: supplied region outlines · Approximate section registration · Independent cluster colors across methods",
                    ha="center", fontsize=9)
        figure.subplots_adjust(wspace=.035, top=.88, bottom=.08)
        name = f"core_{core}_five_panel"
        records[name] = _save(figure, figures, name)
        print(f"Rendered core {core}", flush=True)

    figure, axes = plt.subplots(2, 5, figsize=(20, 12.8))
    for row, core in enumerate(CORES):
        for column, method in enumerate(PRIMARY):
            panel(axes[row, column], images[core][method], rois[core], core,
                  f"Core {core} · {TITLES[method]}")
    figure.text(.5, .012,
                "Fixed resolution 1, seed 0 · Black: supplied region outlines · Approximate section alignment · Colors/IDs are method-specific",
                ha="center", fontsize=10)
    figure.subplots_adjust(wspace=.035, hspace=.12, bottom=.05, top=.96)
    records["combined_five_panel"] = _save(figure, figures, "combined_five_panel")

    manifest = {
        "schema": "so12_cancer_presentation_revision_v1", "status": "complete",
        "revision": "contrast_v2", "source": str(source),
        "source_manifest_sha256": sha(source / "manifest.json"),
        "source_figure_manifest_sha256": sha(source / "figures/figure_manifest.json"),
        "registration_manifest_sha256": sha(registration_path),
        "clustering_manifest_sha256": sha(source / "clustering_summary.json"),
        "renderer_sha256": sha(Path(__file__)),
        "partitions": {method: item for method, (_, item) in selected.items()},
        "resolution": 1., "seed": 0, "n_cells": sum(COUNTS.values()),
        "primary_methods": list(PRIMARY), "figures": records,
        "color_keys": {"path": "figures/cluster_color_keys.csv",
                       "sha256": sha(figures / "cluster_color_keys.csv")},
        "style": {"palette": list(COLORS), "outline": OUTLINE_COLOR,
                  "outline_width_pt": OUTLINE_WIDTH, "white_halo_width_pt": HALO_WIDTH,
                  "palette_rule": "Fixed categorical lookup by cluster ID; same across cores; no ROI-dependent recoloring"},
        "audits": audits, "annotations_shifted": False,
        "scientific_inputs_and_metrics_changed": False,
        "interpretation": "Presentation-only restyling; all labels, cell footprints, registration, metrics and scientific conclusions are unchanged.",
    }
    dump(figures / "figure_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = render(args.source, args.output)
    print(json.dumps({"status": result["status"], "figures": list(result["figures"])}))
