"""Plot full-edge distance/attention summaries; see the adjacent README.md.

Run from the BAGM root with ``PYTHONPATH=src /venv/main/bin/python
reports/analyses/so2_distance_attention/v1/render.py`` after summarization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


OBSERVED = "#17577b"
HEAD = "#819aa9"
REFERENCE = "#bd6236"
CONTENT = "#78518d"
SCORE_BAND = "#547b91"
PRIMARY_RANGE = "10_450_um"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_finite(values, axis):
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    total = np.where(valid, values, 0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def range_finite(values, axis):
    masked = np.ma.masked_invalid(values)
    return masked.min(axis=axis).filled(np.nan), masked.max(axis=axis).filled(np.nan)


def load_inputs(root):
    summary = json.loads((root / "summary.json").read_text())
    selected = [row for row in summary["per_head"] if row["range"] == PRIMARY_RANGE]
    require(len(selected) == 32, "Expected 32 primary block/head fits.")
    heads = {(int(row["block"]), int(row["head"])): row for row in selected}
    require(set(heads) == {(b, h) for b in range(1, 5) for h in range(1, 9)},
            "Primary fits do not cover every block and head exactly once.")
    matrices = {}
    for name in ("power_exponent", "power_r2", "inverse_square_r2"):
        values = [float(heads[b, h][name]) for b in range(1, 5) for h in range(1, 9)]
        matrices[name] = np.asarray(values).reshape(4, 8)
        require(np.isfinite(matrices[name]).all(), f"Non-finite {name}.")
    ratios = []
    for b in range(1, 5):
        for h in range(1, 9):
            row = heads[b, h]
            require(row["flat_mse"] > 0, "Flat score MSE must be positive for an error ratio.")
            ratio = float(row["inverse_square_mse"]) / float(row["flat_mse"])
            require(ratio >= -1e-10 and np.isclose(ratio, 1 - row["inverse_square_r2"],
                    rtol=1e-8, atol=1e-10), "Fixed-power MSE and R² disagree.")
            ratios.append(max(0.0, ratio))
    matrices["inverse_square_error_ratio"] = np.asarray(ratios).reshape(4, 8)
    require(((matrices["power_r2"] >= -1e-8) & (matrices["power_r2"] <= 1 + 1e-8)).all(),
            "Fitted-power R² is outside its valid range.")

    with np.load(root / "curves.npz", allow_pickle=False) as data:
        curves = {name: np.asarray(data[name], dtype=np.float64)
                  for name in ("bin_edges", "mean", "core_min", "core_max", "per_core")}
        channels = [str(v) for v in data["channels"].tolist()]
    expected = ("attention", "degree_scaled_attention", "inverse_square_attention",
                "degree_scaled_inverse_square", "score", "content", "beta")
    require(len(channels) == 7 and set(channels) == set(expected), "Unexpected curve channels.")
    lookup = {name: channels.index(name) for name in expected}
    require(curves["bin_edges"].shape == (51,) and (np.diff(curves["bin_edges"]) > 0).all(),
            "Expected 50 ordered distance bins.")
    for name in ("mean", "core_min", "core_max"):
        require(curves[name].shape == (4, 7, 50, 8), f"Unexpected {name} dimensions.")
    require(curves["per_core"].shape == (14, 4, 7, 50, 8), "Expected curves for all 14 cores.")
    require(all(not np.isinf(value).any() for value in curves.values()), "Infinite curve values.")
    require(np.allclose(mean_finite(curves["per_core"], axis=0), curves["mean"],
                        rtol=1e-9, atol=1e-10, equal_nan=True), "Core curves and equal-core mean disagree.")
    for name in expected[:4]:
        values = curves["mean"][:, lookup[name]]
        require((values[np.isfinite(values)] > 0).all(), f"Nonpositive {name} cannot be log-plotted.")
    curves["centers"] = (curves["bin_edges"][:-1] + curves["bin_edges"][1:]) / 2
    require((curves["centers"] > 0).all(), "Distance centers must be positive.")
    return summary, curves, lookup, matrices


def per_block_curve(curves, lookup, block, channel):
    head_curves = curves["mean"][block, lookup[channel]]
    center = mean_finite(head_curves, axis=-1)
    core_means = mean_finite(curves["per_core"][:, block, lookup[channel]], axis=-1)
    lower, upper = range_finite(core_means, axis=0)
    return head_curves, center, lower, upper


def distance_axis(ax, centers, *, log_y=False, ylabel=None):
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(max(centers[0], 1e-3), 500)
    ax.set_xticks([10, 30, 100, 300, 500], labels=["10", "30", "100", "300", "500"])
    for boundary in (50, 150, 300):
        ax.axvline(boundary, color="#79858b", alpha=0.25, lw=0.65, zorder=0)
    ax.grid(axis="y", which="major", color="#9ca6ad", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Projected cell distance (μm)")
    if ylabel:
        ax.set_ylabel(ylabel)


def positive_limits(arrays, include=None):
    values = np.concatenate([np.asarray(a).ravel() for a in arrays])
    values = values[np.isfinite(values) & (values > 0)]
    require(len(values), "No positive values to plot.")
    low, high = float(values.min()), float(values.max())
    if include is not None:
        low, high = min(low, include), max(high, include)
    return low / 1.3, high * 1.3


def label_color(cmap, norm, value):
    red, green, blue, _ = cmap(norm(value))
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
              for v in (red, green, blue)]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return "white" if luminance < 0.179 else "#18232b"


def heatmap(fig, ax, values, *, title, caption, cmap, norm, formatter, colorbar_label,
            annotation_values=None):
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(8), labels=[str(h) for h in range(1, 9)])
    ax.set_yticks(range(4), labels=[f"Block {b}" for b in range(1, 5)])
    ax.tick_params(length=0)
    ax.set_xlabel("Attention head")
    ax.set_title(title + "\n" + caption, fontsize=11.5, pad=11)
    for b in range(4):
        for h in range(8):
            value = values[b, h]
            displayed = value if annotation_values is None else annotation_values[b, h]
            ax.text(h, b, formatter(displayed), ha="center", va="center", fontsize=9,
                    color=label_color(image.cmap, image.norm, value))
    bar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.025)
    bar.set_label(colorbar_label, fontsize=9)
    return bar


def main_figure(root, summary, curves, lookup, matrices):
    fig = plt.figure(figsize=(16, 10.3), layout="constrained")
    outer = fig.add_gridspec(3, 1, height_ratios=[0.095, 1, 1.12], hspace=0.10)
    legend = fig.add_subplot(outer[0]); legend.axis("off")
    legend.legend(handles=[
        Line2D([0], [0], color=HEAD, lw=0.9, label="Individual attention heads"),
        Line2D([0], [0], color=OBSERVED, lw=2, label="Mean over heads"),
        Patch(facecolor=SCORE_BAND, alpha=0.14, edgecolor="none", label="Range across cores of head means"),
        Line2D([0], [0], color=REFERENCE, lw=1.7, ls="--", label="Same-graph inverse-square reference"),
        Line2D([0], [0], color="#68737b", lw=1.1, ls=":", label="Uniform attention = 1"),
    ], loc="center", ncol=5, frameon=False, fontsize=8.5)
    top = outer[1].subgridspec(1, 4, wspace=0.025)
    bottom = outer[2].subgridspec(1, 3, wspace=0.06)
    centers = curves["centers"]
    prepared = [per_block_curve(curves, lookup, b, "degree_scaled_attention") for b in range(4)]
    references = [per_block_curve(curves, lookup, b, "degree_scaled_inverse_square")[1]
                  for b in range(4)]
    ylim = positive_limits([a for item in prepared for a in item] + references, include=1)
    for block in range(4):
        ax = fig.add_subplot(top[0, block])
        heads, center, lower, upper = prepared[block]
        ax.fill_between(centers, lower, upper, color=SCORE_BAND, alpha=0.13, linewidth=0)
        for head in range(8):
            ax.plot(centers, heads[:, head], color=HEAD, alpha=0.55, lw=0.85)
        ax.plot(centers, center, color=OBSERVED, lw=2.1)
        ax.plot(centers, references[block], color=REFERENCE, lw=1.7, ls="--")
        ax.axhline(1, color="#68737b", lw=1.05, ls=":")
        distance_axis(ax, centers, log_y=True, ylabel="Degree-scaled attention  nᵢ αᵢⱼ" if block == 0 else None)
        ax.set_ylim(*ylim)
        ax.set_title(f"Block {block + 1}", loc="left", fontweight="bold", pad=9)
        if block:
            ax.tick_params(labelleft=False)

    exponent = matrices["power_exponent"]
    low = min(0.0, float(exponent.min()))
    high = max(0.1, float(exponent.max()))
    if low < 0 < high:
        pnorm, pcmap = TwoSlopeNorm(vmin=low, vcenter=0, vmax=high), "RdBu_r"
    else:
        pnorm, pcmap = Normalize(vmin=low, vmax=high), "Blues"
    heatmap(fig, fig.add_subplot(bottom[0, 0]), exponent,
            title="Fitted power exponent p", caption="Inverse-square hypothesis: p = 2",
            cmap=pcmap, norm=pnorm, formatter=lambda v: f"{v:.2f}", colorbar_label="Exponent in r⁻ᵖ")
    heatmap(fig, fig.add_subplot(bottom[0, 1]), matrices["power_r2"],
            title="Variance explained by fitted power law", caption="Receiver-centered score R²",
            cmap="Blues", norm=Normalize(vmin=0, vmax=1), formatter=lambda v: f"{v:.2f}",
            colorbar_label="R² (0 to 1)")
    error_ratio = matrices["inverse_square_error_ratio"]
    positive = error_ratio[error_ratio > 0]
    low = max(min(1.0, float(positive.min()) if positive.size else 1e-3), 1e-8)
    high = max(1.01, float(error_ratio.max()))
    ratio_bar = heatmap(fig, fig.add_subplot(bottom[0, 2]), np.maximum(error_ratio, low),
            title="Fixed inverse-square error / flat error", caption="Above 1×: inverse-square fits worse",
            cmap="YlOrBr", norm=LogNorm(vmin=low, vmax=high),
            formatter=lambda v: f"{v:.1f}×" if v < 100 else f"{v:.0f}×",
            colorbar_label="Centered-score MSE ratio (log scale)", annotation_values=error_ratio)
    ticks = [v for v in (0.01, 0.03, 0.1, 0.3, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
             if low <= v <= high]
    if len(ticks) >= 2:
        ratio_bar.set_ticks(ticks, labels=[f"{v:g}×" for v in ticks])
    fig.suptitle("Does learned attention follow inverse-square distance?", fontsize=19, fontweight="bold")
    fig.supxlabel(
        f"All {summary['total_edges']:,} directed edges · 14 cores · 4 blocks × 8 heads · one fixed mask per core · FP32 CPU inference\n"
        "Curves: equal-edge means within core/bin, then equal available cores. Ranges describe core variation, not confidence intervals.\n"
        "Fits: 10–450 μm; equal receivers within core, then equal cores. Vertical guides: graph shells at 50, 150 and 300 μm.\n"
        "The inverse-square reference uses the same complete neighborhoods; pooled curves can bend even for an exact r⁻² kernel.",
        fontsize=9.5, linespacing=1.4,
    )
    png, pdf = root / "distance_attention.png", root / "distance_attention.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf, metadata={"Title": "Distance versus attention in the latest SO2 model"})
    plt.close(fig)
    return png, pdf


def companion_figure(root, summary, curves, lookup):
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    outer = fig.add_gridspec(3, 1, height_ratios=[0.12, 1, 1], hspace=0.08)
    legend = fig.add_subplot(outer[0]); legend.axis("off")
    legend.legend(handles=[
        Line2D([0], [0], color=HEAD, lw=0.85, label="Individual raw-attention heads"),
        Line2D([0], [0], color=OBSERVED, lw=2, label="Observed α / combined score"),
        Line2D([0], [0], color=REFERENCE, lw=1.7, ls="--", label="Inverse-square α reference"),
        Patch(facecolor=SCORE_BAND, alpha=0.14, edgecolor="none", label="Across-core range"),
        Line2D([0], [0], color=CONTENT, lw=1.8, label="Content score"),
        Line2D([0], [0], color=REFERENCE, lw=1.8, label="Geometry bias β"),
    ], loc="center", ncol=3, frameon=False, fontsize=9)
    grid = outer[1:].subgridspec(2, 4, wspace=0.025, hspace=0.12)
    centers = curves["centers"]
    observed = [per_block_curve(curves, lookup, b, "attention") for b in range(4)]
    reference = [per_block_curve(curves, lookup, b, "inverse_square_attention")[1]
                 for b in range(4)]
    attention_limits = positive_limits([a for item in observed for a in item] + reference)
    scores = {channel: [per_block_curve(curves, lookup, b, channel) for b in range(4)]
              for channel in ("score", "content", "beta")}
    score_values = [item[1] for group in scores.values() for item in group]
    score_values += [item[index] for item in scores["score"] for index in (2, 3)]
    finite_scores = np.concatenate([a.ravel() for a in score_values])
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    lo, hi = min(0, float(finite_scores.min())), max(0, float(finite_scores.max()))
    padding = max((hi - lo) * 0.08, 0.05)
    for block in range(4):
        top = fig.add_subplot(grid[0, block])
        heads, center, lower, upper = observed[block]
        top.fill_between(centers, lower, upper, color=SCORE_BAND, alpha=0.13, linewidth=0)
        for head in range(8):
            top.plot(centers, heads[:, head], color=HEAD, lw=0.85, alpha=0.55)
        top.plot(centers, center, color=OBSERVED, lw=2.1)
        top.plot(centers, reference[block], color=REFERENCE, lw=1.7, ls="--")
        distance_axis(top, centers, log_y=True, ylabel="Raw incoming attention α" if block == 0 else None)
        top.set_ylim(*attention_limits)
        top.set_title(f"Block {block + 1}", loc="left", fontweight="bold", pad=9)
        bottom = fig.add_subplot(grid[1, block])
        _, _, lower, upper = scores["score"][block]
        bottom.fill_between(centers, lower, upper, color=SCORE_BAND, alpha=0.13, linewidth=0)
        for channel, color in (("score", OBSERVED), ("content", CONTENT), ("beta", REFERENCE)):
            bottom.plot(centers, scores[channel][block][1], color=color,
                        lw=2.1 if channel == "score" else 1.8)
        bottom.axhline(0, color="#6e7880", ls=":", lw=0.9)
        distance_axis(bottom, centers, ylabel="Raw score / logit" if block == 0 else None)
        bottom.set_ylim(lo - padding, hi + padding)
        if block:
            top.tick_params(labelleft=False)
            bottom.tick_params(labelleft=False)
    fig.suptitle("Distance dependence of attention and its score components", fontsize=18, fontweight="bold")
    fig.supxlabel(
        f"All {summary['total_edges']:,} directed edges · 14 cores · one fixed mask per core · FP32 CPU inference\n"
        "Equal-edge means within core/bin, then equal available cores and heads. Shading shows across-core ranges of head means.\n"
        "Combined score = modulated Q–K content + β. Raw scores include receiver-specific offsets; the fitted exponents remove those offsets.\n"
        "The normalized inverse-square reference can curve after pooling. These are descriptive associations on the existing graph.",
        fontsize=9.5, linespacing=1.4,
    )
    png, pdf = root / "score_components.png", root / "score_components.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf, metadata={"Title": "Distance dependence of raw attention and score components"})
    plt.close(fig)
    return png, pdf


def render(root):
    summary, curves, lookup, matrices = load_inputs(root)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    files = (*main_figure(root, summary, curves, lookup, matrices),
             *companion_figure(root, summary, curves, lookup))
    provenance = {
        "schema": "so2_distance_attention_figure_provenance_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": summary["run_id"], "checkpoint_sha256": summary["checkpoint_sha256"],
        "render_script_sha256": sha256(Path(__file__).resolve()),
        "summary_sha256": sha256(root / "summary.json"),
        "curves_sha256": sha256(root / "curves.npz"),
        "outputs": {p.name: sha256(p) for p in files},
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "primary_fit_range": PRIMARY_RANGE, "total_edges": summary["total_edges"],
        "curve_weighting": summary["curve_weighting"],
        "regression_weighting": summary["regression_weighting"],
        "across_core_range": "Range of per-core, head-averaged curves in each bin; not a confidence interval.",
        "error_heatmap": "Fixed inverse-square centered-score MSE divided by flat centered-score MSE; greater than one favors flat over fixed inverse square.",
    }
    (root / "figure_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outputs": [str(p) for p in files]}, indent=2))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    render(args.root.resolve())
