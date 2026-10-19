"""Restyle audited pooled F1 with slim, rounded bars and restrained palettes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgb
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from PIL import Image

from spatial_benchmark.paths import current_paths
from run_so12_cancer_limited_union import read, write, record, require, verify_inventory

SOURCE_ID = "ev_so12_cancer_pooled_union_s0_20261018"
SOURCE_SHA = "8727217f27775252d22d16bb5f14b06b2707b904aa7b2c1117e5c0b987f1d9b2"
FIGURE_ID = "viz_so12_pooled_f1_monochrome_20261019"
METHODS = ["h4", "h3", "h2", "h1", "h0", "pca", "scvi"]
LABELS = ["h4", "h3", "h2", "h1", "h0", "PCA", "scVI"]
COLORS = ["#3F3F3F", "#555555", "#6B6B6B", "#818181", "#979797", "#ADADAD", "#8B8B8B"]
MUTED_COLORS = ["#5E778B", "#6C8495", "#7D93A0", "#8DA1AC", "#9DAFB7", "#ACA293", "#80988D"]
SUBSCRIPT_LABELS = [r"$h_{4}$", r"$h_{3}$", r"$h_{2}$", r"$h_{1}$", r"$h_{0}$", "PCA", "scVI"]
WIDTH = .42
RADIUS_PT = 1.7
LIMITS = (.78, .88)


def rounded_top_bar(axis, x, value, color):
    """Round only the top corners in physical units, preserving exact extrema."""
    lo, hi = LIMITS
    left, right = x - WIDTH / 2, x + WIDTH / 2
    radius_px = RADIUS_PT * axis.figure.dpi / 72
    rx = radius_px / axis.bbox.width * np.diff(axis.get_xlim())[0]
    ry = radius_px / axis.bbox.height * (hi - lo)
    require(rx < WIDTH / 2 and ry < value - lo, "Corner radius exceeds the bar")
    vertices = [(left, lo), (left, value - ry), (left, value),
                (left + rx, value), (right - rx, value), (right, value),
                (right, value - ry), (right, lo), (left, lo)]
    codes = [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.CURVE3,
             PlotPath.CURVE3, PlotPath.LINETO, PlotPath.CURVE3,
             PlotPath.CURVE3, PlotPath.LINETO, PlotPath.CLOSEPOLY]
    patch = PathPatch(PlotPath(vertices, codes), facecolor=color,
                      edgecolor="none", linewidth=0, zorder=3)
    axis.add_patch(patch)
    require(patch.get_path().vertices[:, 1].max() == value, "Rounded bar changes F1")
    require(patch.get_path().vertices[:, 1].min() == lo, "Rounded bar changes baseline")
    return patch


def main(style="monochrome"):
    muted = style == "muted"
    figure_id = "viz_so12_pooled_f1_muted_20261019" if muted else FIGURE_ID
    colors = MUTED_COLORS if muted else COLORS
    labels = SUBSCRIPT_LABELS if muted else LABELS
    paths = current_paths()
    source = paths.report_root / "analyses/so12_cancer_pooled_union" / SOURCE_ID
    output = paths.scratch_root / "active_runs" / figure_id
    final = source.parent / figure_id
    require(not final.exists() and not output.exists(), "Use a new presentation version")
    verify_inventory(source, SOURCE_SHA)
    summary = read(source / "summary.json")
    metrics = read(source / "metrics/pooled_union.json")
    rows = {row["method"]: row for row in summary["primary_scores"]}
    require(len(rows) == len(summary["primary_scores"]) == 7, "Expected seven primary scores")
    require(summary["primary_resolution"] == 1 and summary["primary_seed"] == 0,
            "Primary setting changed")
    csv = pd.read_csv(source / "metrics/primary_report_table.csv")
    for method in METHODS:
        part = next(p for p in metrics["results"] if
                    (p["method"], p["resolution"], p["seed"]) == (method, 1., 0))
        curve = part["pooled"]
        best = curve["steps"][curve["best_step_index"]]
        row = rows[method]
        require(row["best_f1"] == best["footprint"]["dice_f1"], "F1 differs from pooled path")
        require(abs(row["best_f1"] - 2 * row["tp"] /
                    (2 * row["tp"] + row["fp"] + row["fn"])) < 1e-15, "Pooled F1 mismatch")
        require(abs(float(csv.loc[csv.method == method, "best_f1"].iloc[0]) -
                    row["best_f1"]) < 1e-14, "CSV mismatch")
    output.mkdir(parents=True)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(source / "metrics/primary_report_table.csv", output / "source_scores.csv")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#262626", "axes.labelcolor": "#333333",
                         "xtick.color": "#333333", "ytick.color": "#555555",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axis = plt.subplots(figsize=(10.4, 5.7))
    fig.subplots_adjust(left=.105, right=.975, bottom=.245, top=.79)
    fig.text(.105, .94, "Pooled F1", fontsize=21, weight="bold")
    fig.text(.105, .883, "Cores 15 + 23 · one shared cluster union", fontsize=11.5, color="#666666")
    axis.set(ylim=LIMITS, xlim=(-.65, 6.65), xticks=np.arange(7),
             xticklabels=labels, ylabel="Pooled pixel F1" if muted else "Pooled pixel F1 (Dice)")
    if muted:
        axis.set_xlabel("Cell representation", fontsize=12, labelpad=14)
    axis.set_yticks(np.linspace(*LIMITS, 6))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.labelpad = 12
    axis.grid(axis="y", color="#E7E7E7", linewidth=.7, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#B8B8B8")
        axis.spines[spine].set_linewidth(.7)
    axis.tick_params(axis="x", length=0, pad=10, labelsize=12)
    axis.tick_params(axis="y", length=0, pad=8)
    fig.canvas.draw()
    for x, (method, color) in enumerate(zip(METHODS, colors, strict=True)):
        value = rows[method]["best_f1"]
        rounded_top_bar(axis, x, value, color)
        axis.text(x, value + .0019, f"{value:.4f}", ha="center", va="bottom",
                  fontsize=12, color="#292929")
    for y in (.012, .034):
        axis.plot([-.007, .007], [y - .011, y + .011], transform=axis.transAxes,
                  color="#555555", lw=1.2, clip_on=False, zorder=5)
    if muted:
        axis.text(1, 1.035, "Zoomed y axis: 0.78–0.88 (zero omitted)",
                  transform=axis.transAxes, ha="right", va="bottom", fontsize=9.5, color="#666666")
    else:
        axis.text(.5, -.205, "Zoomed y axis: 0.78–0.88 (zero omitted)",
                  transform=axis.transAxes, ha="center", fontsize=9.5, color="#666666")
    fig.text(.105, .077, "Best pooled F1 after backward cluster removal · Leiden resolution 1, seed 0",
             fontsize=9.5, color="#606060")
    fig.text(.105, .035, "Annotation-selected agreement · approximate tissue registration",
             fontsize=9.5, color="#606060")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in [*fig.texts, *axis.texts, axis.xaxis.label, axis.yaxis.label, *axis.get_xticklabels()]:
        bbox = text.get_window_extent(renderer)
        require(bbox.x0 >= 0 and bbox.y0 >= 0 and bbox.x1 <= fig.bbox.width
                and bbox.y1 <= fig.bbox.height, "Label clipped by figure")
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"pooled_f1_bar.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    with Image.open(output / "pooled_f1_bar.png") as image:
        pixels = np.asarray(image.convert("RGB"))
        achromatic = np.array_equal(pixels[:, :, 0], pixels[:, :, 1]) and np.array_equal(pixels[:, :, 1], pixels[:, :, 2])
        if muted:
            require(not achromatic, "Muted color variation is missing")
            for color in colors:
                rgb = np.rint(np.array(to_rgb(color)) * 255).astype(np.uint8)
                require(np.any(np.all(pixels == rgb, axis=2)), "A bar color is missing")
        else:
            require(achromatic, "Non-neutral raster colors")
    write(output / "validation.json", {
        "passed": True, "presentation_only": True, "source_evaluation": SOURCE_ID,
        "source_manifest_sha256": SOURCE_SHA, "source_summary": record(source / "summary.json"),
        "script": record(Path(__file__)), "method_order": METHODS,
        "scores": {method: rows[method]["best_f1"] for method in METHODS},
        "primary_resolution": 1, "primary_seed": 0, "axis_limits": LIMITS,
        "bar_width": WIDTH, "original_bar_width": .66, "top_corner_radius_pt": RADIUS_PT,
        "colors": colors, "style": style, "all_pixels_achromatic": bool(achromatic), "label_bounds_verified": True,
        "axis_labels": {"y": "Pooled pixel F1" if muted else "Pooled pixel F1 (Dice)",
                        "x": "Cell representation" if muted else ""},
        "tick_labels": labels,
        "bar_extrema_exact": True, "visual_review": "pending",
        "matplotlib_version": matplotlib.__version__,
        "aggregation": "Pooled native-pixel micro F1 from summed TP/FP/FN, one shared selected cluster-ID union",
    })
    print(json.dumps({"output": str(output), "scores_verified": 7, "style": style}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=("monochrome", "muted"), default="monochrome")
    main(parser.parse_args().style)
