"""Plot frozen-checkpoint residual/update summaries without running inference.

Run after ``analyze.py --summarize`` from the BAGM root:
``PYTHONPATH=src /venv/main/bin/python
reports/analyses/so2_residual_attention/v1/render.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ATTENTION = "#1c6284"
CENTERED = "#805598"
ALIGNMENT = "#277c70"
FFN = "#b4682e"
CORE_COLOR = "#7a8992"
X = np.arange(1, 5, dtype=float)
CORES = tuple(range(15, 29))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(root):
    summary = json.loads((root / "summary.json").read_text())
    blocks = sorted(summary["per_block"], key=lambda row: row["block"])
    require([r["block"] for r in blocks] == [1, 2, 3, 4], "Expected four block summaries.")
    with (root / "per_core_block.csv").open(newline="") as source:
        records = [{key: float(value) for key, value in row.items()}
                   for row in csv.DictReader(source)]
    require(len(records) == 56, "Expected 14 cores × 4 blocks.")
    lookup = {(int(r["core"]), int(r["block"])): r for r in records}
    require(set(lookup) == {(core, block) for core in CORES for block in range(1, 5)},
            "Core/block identities are incomplete or duplicated.")
    require(all(np.isfinite(value) for row in records for value in row.values()),
            "Non-finite per-core statistics.")
    require(sum(lookup[core, 1]["cells"] for core in CORES) == summary["total_cells"],
            "Cell totals disagree between summary and core records.")
    for row in blocks:
        for ratio in ("rms_attention_to_residual", "between_cell_rms_attention_to_residual",
                      "rms_ffn_to_post_attention"):
            require(np.isfinite(row[ratio]) and row[ratio] >= 0, f"Invalid {ratio}.")
        require(-1 <= row["mean_cosine"] <= 1, "Mean cosine is outside [-1, 1].")
        distribution = row["attention_to_residual"]
        require(0 <= distribution["mean_core_q10"] <= distribution["mean_core_median"]
                <= distribution["mean_core_q90"], "Mean core quantiles are not ordered.")
        sub = [lookup[core, row["block"]] for core in CORES]
        for source_key, summary_key in (("attention_to_residual_q10", "mean_core_q10"),
                                        ("attention_to_residual_median", "mean_core_median"),
                                        ("attention_to_residual_q90", "mean_core_q90")):
            require(np.isclose(np.mean([r[source_key] for r in sub]), distribution[summary_key],
                               rtol=1e-9, atol=1e-11), "Core quantiles disagree with their aggregate.")
        require(np.isclose(np.mean([r["mean_cosine"] for r in sub]), row["mean_cosine"],
                           rtol=1e-9, atol=1e-11), "Core mean cosines disagree with the summary.")
    return summary, blocks, lookup


def configure_blocks(ax, *, ylabel):
    ax.set_xticks(X, labels=[f"Block {block}" for block in range(1, 5)])
    ax.set_xlim(0.55, 4.45)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#939fa7", alpha=0.19, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=8)


def ratio_scale(ax, values):
    values = np.asarray(values, dtype=float).ravel()
    require(np.isfinite(values).all() and (values >= 0).all(), "Invalid magnitude ratios.")
    high = max(1.0, float(values.max()))
    positive = values[values > 0]
    if positive.size == values.size and high / float(positive.min()) > 25:
        ax.set_yscale("log")
        ax.set_ylim(float(positive.min()) / 1.35, high * 1.4)
    else:
        ax.set_ylim(0, high * 1.18)
    ax.axhline(1, color="#64717b", linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)


def dots_and_aggregate(ax, core_values, aggregate, *, color, offset=0, label=None):
    jitter = np.linspace(-0.075, 0.075, len(CORES))
    for block in range(4):
        ax.scatter(X[block] + offset + jitter, core_values[:, block], s=21,
                   facecolors="white", edgecolors=color, linewidths=0.7,
                   alpha=0.72, zorder=3)
    line, = ax.plot(X + offset, aggregate, color=color, marker="D", markersize=6.5,
                    linewidth=1.65, markeredgecolor="white", markeredgewidth=0.7,
                    label=label, zorder=5)
    return line


def core_matrix(lookup, key):
    return np.asarray([[lookup[core, block][key] for block in range(1, 5)] for core in CORES])


def render(root):
    summary, blocks, lookup = load_inputs(root)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12.5,
        "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2), layout="constrained")
    fig.get_layout_engine().set(hspace=0.10, wspace=0.07)

    raw = np.asarray([b["rms_attention_to_residual"] for b in blocks])
    centered = np.asarray([b["between_cell_rms_attention_to_residual"] for b in blocks])
    raw_cores = core_matrix(lookup, "rms_attention_to_residual")
    centered_cores = core_matrix(lookup, "between_cell_rms_attention_to_residual")
    ax = axes[0, 0]
    configure_blocks(ax, ylabel="Attention / residual RMS norm")
    ratio_scale(ax, np.r_[raw, centered, raw_cores.ravel(), centered_cores.ravel()])
    handles = [
        dots_and_aggregate(ax, raw_cores, raw, color=ATTENTION, offset=-0.14, label="Raw vectors"),
        dots_and_aggregate(ax, centered_cores, centered, color=CENTERED, offset=0.14,
                           label="Across-cell centered vectors"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor=CORE_COLOR,
               markerfacecolor="white", markersize=5, label="Each of 14 cores"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="best")
    ax.set_title("A  Attention update versus incoming residual\nDiamonds: ratio of aggregate RMS norms", loc="left", pad=13)

    ax = axes[0, 1]
    quantiles = np.asarray([[b["attention_to_residual"][key] for b in blocks]
                           for key in ("mean_core_q10", "mean_core_median", "mean_core_q90")])
    medians = core_matrix(lookup, "attention_to_residual_median")
    q10 = core_matrix(lookup, "attention_to_residual_q10")
    q90 = core_matrix(lookup, "attention_to_residual_q90")
    configure_blocks(ax, ylabel=r"Per-cell ratio  $\|u\|_2 / \|h\|_2$")
    ratio_scale(ax, np.r_[quantiles.ravel(), q10.ravel(), q90.ravel()])
    jitter = np.linspace(-0.14, 0.14, len(CORES))
    for block in range(4):
        ax.vlines(X[block] + jitter, q10[:, block], q90[:, block],
                  color=CORE_COLOR, alpha=0.18, linewidth=0.6, zorder=2)
        ax.scatter(X[block] + jitter, medians[:, block], s=18, color=CORE_COLOR, alpha=0.55, zorder=3)
    ax.errorbar(X, quantiles[1], yerr=np.vstack((quantiles[1] - quantiles[0], quantiles[2] - quantiles[1])),
                fmt="D", color=ATTENTION, markeredgecolor="white", markeredgewidth=0.7,
                markersize=7, elinewidth=2.1, capsize=7, capthick=2.1, zorder=5,
                label="Mean core q10 / median / q90")
    ax.legend(handles=[
        Line2D([0], [0], marker="D", color=ATTENTION, lw=2, markersize=6,
               label="Mean of core q10 / median / q90"),
        Line2D([0], [0], marker="o", color=CORE_COLOR, lw=0.6, markersize=4,
               label="Each core: q10–q90 and median"),
    ], frameon=False, fontsize=9, loc="best")
    ax.set_title("B  Variation across cells\nExact core quantiles; averages are not pooled quantiles", loc="left", pad=13)

    ax = axes[1, 0]
    cosines = core_matrix(lookup, "mean_cosine")
    mean_cosine = np.asarray([b["mean_cosine"] for b in blocks])
    configure_blocks(ax, ylabel="Mean cosine similarity  cos(h, u)")
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.axhline(0, color="#64717b", linestyle=(0, (3, 3)), linewidth=1)
    dots_and_aggregate(ax, cosines, mean_cosine, color=ALIGNMENT)
    ax.text(0.98, 0.98, "+1 aligned", transform=ax.transAxes, ha="right", va="top", color="#5b6972", fontsize=9)
    ax.text(0.98, 0.02, "−1 opposed", transform=ax.transAxes, ha="right", va="bottom", color="#5b6972", fontsize=9)
    ax.set_title("C  Does the attention update align with the residual?\nDiamonds: equal-core mean cosine; circles: core means", loc="left", pad=13)

    ax = axes[1, 1]
    ffn = np.asarray([b["rms_ffn_to_post_attention"] for b in blocks])
    ffn_cores = core_matrix(lookup, "rms_ffn_to_post_attention")
    configure_blocks(ax, ylabel="FFN / post-attention RMS norm")
    ratio_scale(ax, np.r_[ffn, ffn_cores.ravel()])
    dots_and_aggregate(ax, ffn_cores, ffn, color=FFN)
    ax.set_title("D  Size of the subsequent FFN update\nDiamonds: ratio of aggregate RMS norms; circles: cores", loc="left", pad=13)

    fig.suptitle("Attention updates compared with the residual stream", fontsize=18, fontweight="bold")
    fig.supxlabel(
        f"All {summary['total_cells']:,} cells · 14 cores · epoch 200, seed 0 · one fixed mask per core · FP32 CPU inference\n"
        "h: incoming residual; u: projected attention update; a = h + u; f: FFN update. Dashed ratio 1 means equal magnitude.\n"
        "RMS ratios use equal-core mean squared norms. Centering subtracts each core’s mean vector across cells.\n"
        "Core variation is descriptive, not a patient-level confidence interval. Magnitude ratios are not information fractions; h includes earlier blocks.",
        fontsize=9.5, linespacing=1.45,
    )
    png = root / "residual_attention.png"
    pdf = root / "residual_attention.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf, metadata={"Title": "Residual versus projected attention update magnitude"})
    plt.close(fig)
    provenance = {
        "schema": "so2_residual_attention_figure_provenance_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": summary["run_id"], "checkpoint_sha256": summary["checkpoint_sha256"],
        "render_script_sha256": sha256(Path(__file__).resolve()),
        "summary_sha256": sha256(root / "summary.json"),
        "per_core_block_csv_sha256": sha256(root / "per_core_block.csv"),
        "outputs": {png.name: sha256(png), pdf.name: sha256(pdf)},
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "total_cells": summary["total_cells"], "core_count": len(CORES),
        "weighting": summary["weighting"],
        "rms_aggregation": "Ratio after equal-core aggregation of mean squared vector norms; not a mean of core ratios.",
        "quantile_panel": "Mean of exact core q10, median, and q90; all individual core intervals and medians also shown. These are not pooled quantiles or confidence intervals.",
        "centered_vectors": "Subtract each core's mean vector across cells, separately for residual and attention.",
    }
    (root / "figure_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    render(args.root.resolve())
