"""Render the completed backward-removal F1 table with explicitly zoomed axes."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

from spatial_benchmark.paths import current_paths
from run_so12_cancer_limited_union import read, write, record, require, verify_inventory

SOURCE_ID = "ev_so12_cancer_backward_union_s0_20261018"
SOURCE_SHA = "88f966f74307485e1c66c2728f41b4efe9bb51860823c63133976605c0483ffa"
FIGURE_ID = "viz_so12_backward_f1_bars_20261018"
METHODS = ["h4", "h3", "h2", "h1", "h0", "pca", "scvi"]
DISPLAY = {**{f"h{i}": f"h{i}" for i in range(5)}, "pca": "PCA", "scvi": "scVI"}
COLORS = ["#255B91", "#3572A5", "#4A85B4", "#639BC2", "#82AFCF", "#278573", "#A46A40"]
LIMITS = {15: (0.72, 0.80), 23: (0.86, 0.94)}


def save(fig, output, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    paths = current_paths()
    source = paths.report_root / "analyses/so12_cancer_backward_union" / SOURCE_ID
    output = paths.scratch_root / "active_runs" / FIGURE_ID
    final = source.parent / FIGURE_ID
    require(not final.exists(), "Published presentation already exists; use a new version")
    verify_inventory(source, SOURCE_SHA)
    summary = read(source / "summary.json")
    metrics = read(source / "metrics/backward_union.json")
    rows = {(r["method"], r["core"]): r for r in summary["primary_scores"]}
    require(len(rows) == 14 and len(summary["primary_scores"]) == 14, "Expected 14 unique primary rows")
    require(summary["primary_resolution"] == 1 and summary["primary_seed"] == 0, "Primary setting changed")
    csv = pd.read_csv(source / "metrics/primary_report_table.csv")
    for method in METHODS:
        part = next(p for p in metrics["results"] if (p["method"], p["resolution"], p["seed"]) == (method, 1, 0))
        for core in (15, 23):
            row = rows[method, core]
            path = part["cores"][str(core)]
            best = path["steps"][path["best_step_index"]]
            require(row["best_f1"] == best["footprint"]["dice_f1"], "F1 differs from audited path")
            for field in ("k", "roi_cells_retained", "roi_cells_total", "roi_cells_retained_pct"):
                require(row[field] == best[field], f"Path mismatch: {field}")
            require(row["initial_clusters"] == best["initial_overlap_clusters"], "Cluster denominator differs")
            old = csv[(csv.method == method) & (csv.core == core)].iloc[0]
            require(abs(float(old.best_f1) - row["best_f1"]) < 1e-14, "CSV F1 mismatch")
            lo, hi = LIMITS[core]
            require(lo < row["best_f1"] < hi - 0.004, "Bar or value label outside axis")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    shutil.copy2(source / "metrics/primary_report_table.csv", output / "f1_and_retention_full_precision.csv")
    wide = [{"method": DISPLAY[m], "core_15_f1": rows[m, 15]["best_f1"],
             "core_23_f1": rows[m, 23]["best_f1"]} for m in METHODS]
    pd.DataFrame(wide).to_csv(output / "f1_table.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": "#243447", "text.color": "#243447",
                         "xtick.color": "#243447", "ytick.color": "#243447",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.7))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.235, top=0.76, wspace=0.20)
    fig.text(0.065, 0.94, "Cancer-region agreement", fontsize=21, weight="bold")
    fig.text(0.065, 0.883, "Best pixel F1 after backward cluster removal  ·  Leiden resolution 1, seed 0", fontsize=12)
    for panel, (axis, core) in enumerate(zip(axes, (15, 23))):
        lo, hi = LIMITS[core]
        values = np.array([rows[m, core]["best_f1"] for m in METHODS])
        bars = axis.bar(np.arange(7), values - lo, bottom=lo, width=0.68,
                        color=COLORS, edgecolor="white", linewidth=0.8, zorder=3)
        require(np.allclose([b.get_y() + b.get_height() for b in bars], values, atol=1e-15, rtol=0), "Incorrect plotted tops")
        for x, value in enumerate(values):
            axis.text(x, value + 0.0013, f"{value:.4f}", ha="center", va="bottom", fontsize=10.5)
        axis.set(ylim=(lo, hi), xlim=(-0.65, 6.65), xticks=np.arange(7),
                 xticklabels=[DISPLAY[m] for m in METHODS], ylabel="Pixel F1 (Dice)")
        axis.set_yticks(np.linspace(lo, hi, 5))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axis.set_title(f"{'AB'[panel]}   Core {core}", loc="left", fontsize=14, weight="bold", pad=15)
        axis.grid(axis="y", color="#DDE3EA", lw=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#AAB5C0")
        axis.tick_params(axis="x", length=0, pad=8)
        axis.tick_params(axis="y", length=3, color="#AAB5C0")
        # Two explicit diagonal marks at the bottom of the y axis indicate omitted zero.
        for y in (0.012, 0.034):
            axis.plot([-0.009, 0.009], [y - 0.012, y + 0.012], transform=axis.transAxes,
                      color="#243447", lw=1.3, clip_on=False, zorder=5)
        axis.text(0.5, -0.22, f"Zoomed y axis: {lo:.2f}–{hi:.2f}", transform=axis.transAxes,
                  ha="center", fontsize=10, color="#526376")
    fig.text(0.065, 0.075, "Both y axes omit zero and span 0.08 F1 units. Compare the labeled scores, not bar-length ratios.", fontsize=10)
    fig.text(0.065, 0.035, "Clusters selected using each scored annotation; descriptive agreement on approximately aligned tissue sections.", fontsize=9.5, color="#526376")
    save(fig, output, "f1_bar_comparison")

    fig, axis = plt.subplots(figsize=(13.6, 5.0))
    axis.set_axis_off()
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.18, top=0.78)
    fig.text(0.035, 0.93, "F1 and region coverage at the best-F1 step", fontsize=19, weight="bold")
    fig.text(0.035, 0.865, "Fixed Leiden resolution 1, seed 0  ·  Cluster counts and cell retention refer to the same selected union", fontsize=11)
    table_rows = []
    for method in METHODS:
        cells = [DISPLAY[method]]
        for core in (15, 23):
            r = rows[method, core]
            cells.extend([f'{r["best_f1"]:.4f}', f'{r["k"]}/{r["initial_clusters"]} ({r["cluster_retained_pct"]:.1f}%)', f'{r["roi_cells_retained_pct"]:.1f}%'])
        table_rows.append(cells)
    columns = ["Method", "Core 15\nF1", "Core 15\nClusters kept", "Core 15\nCells kept", "Core 23\nF1", "Core 23\nClusters kept", "Core 23\nCells kept"]
    table = axis.table(cellText=table_rows, colLabels=columns, cellLoc="center",
                       colWidths=[.10, .11, .20, .14, .11, .20, .14], bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#DDE3EA")
        if row == 0:
            cell.set_facecolor("#243F59")
            cell.set_text_props(color="white", weight="bold", fontsize=10.5)
            cell.set_height(cell.get_height() * 1.25)
        else:
            cell.set_facecolor("#F0F4F8" if row % 2 else "white")
            if column == 0:
                cell.set_text_props(color=COLORS[row - 1], weight="bold")
    fig.text(.035, .10, "Clusters kept: selected / initially overlapping clusters (%). Cells kept: fraction of all annotation-overlapping cells.", fontsize=9.5)
    fig.text(.035, .052, "A cell overlaps if any valid owned pixel is inside the annotation. Approximate alignment; annotation-selected descriptive agreement.", fontsize=9, color="#526376")
    save(fig, output, "f1_retention_table")

    write(output / "validation.json", {"passed": True, "type": "presentation_only", "source_evaluation": SOURCE_ID,
          "source_manifest_sha256": SOURCE_SHA, "source_summary": record(source / "summary.json"),
          "script": record(Path(__file__)), "primary_resolution": 1, "primary_seed": 0,
          "verified_primary_rows": 14, "axis_limits": LIMITS, "equal_axis_span": 0.08,
          "checks": ["immutable source inventory", "F1 matches audited path and source CSV", "retention matches audited path",
                     "all bars and labels inside axes", "bar tops equal full precision F1"],
          "visual_review": "pending", "matplotlib_version": matplotlib.__version__})
    print(json.dumps({"output": str(output), "verified_primary_rows": 14, "files": len(list(output.iterdir()))}))


if __name__ == "__main__":
    main()
