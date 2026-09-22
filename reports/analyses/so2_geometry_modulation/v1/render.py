"""Render the geometry-only modulation audit from frozen summary artifacts.

Run from the BAGM root with ``PYTHONPATH=src /venv/main/bin/python
reports/analyses/so2_geometry_modulation/v1/render.py``.
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
from matplotlib.lines import Line2D
import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(root: Path):
    summary = json.loads((root / "summary.json").read_text())
    records = summary["per_head"]
    require(len(records) == 32, "Expected 32 block/head records.")
    ordered = {(int(r["block"]), int(r["head"])): r for r in records}
    require(
        set(ordered) == {(b, h) for b in range(1, 5) for h in range(1, 9)},
        "Block/head records must cover four blocks and eight heads exactly once.",
    )
    matrices = {}
    for key in ("total_rms", "dynamic_rms", "dynamic_fraction"):
        values = [ordered[b, h][key] for b in range(1, 5) for h in range(1, 9)]
        matrices[key] = np.asarray(
            [np.nan if v is None else float(v) for v in values]
        ).reshape(4, 8)
    for key in ("total_rms", "dynamic_rms"):
        require(np.isfinite(matrices[key]).all(), f"Non-finite {key}.")
        require((matrices[key] >= 0).all(), f"Negative {key}.")
    fraction = matrices["dynamic_fraction"]
    valid_fraction = fraction[np.isfinite(fraction)]
    require(
        ((valid_fraction >= -1e-7) & (valid_fraction <= 1 + 1e-7)).all(),
        "Dynamic fraction must lie between zero and one.",
    )
    for (block, head), record in ordered.items():
        total = float(record["total_ms"])
        dynamic = float(record["dynamic_ms"])
        components = sum(
            float(record[key])
            for key in ("fixed_ms", "within_core_ms", "between_core_ms")
        )
        require(
            np.isclose(total, components, rtol=1e-6, atol=1e-10),
            f"Departure decomposition failed at block {block}, head {head}.",
        )
        require(
            np.isclose(
                dynamic,
                float(record["within_core_ms"]) + float(record["between_core_ms"]),
                rtol=1e-6,
                atol=1e-10,
            ),
            f"Dynamic decomposition failed at block {block}, head {head}.",
        )
        if total > 0:
            require(
                np.isclose(
                    matrices["dynamic_fraction"][block - 1, head - 1],
                    dynamic / total,
                    rtol=1e-6,
                    atol=1e-10,
                ),
                f"Dynamic fraction failed at block {block}, head {head}.",
            )

    with np.load(root / "distributions.npz", allow_pickle=False) as data:
        edges = np.asarray(data["bin_edges"], dtype=np.float64)
        probability = np.asarray(data["probability"], dtype=np.float64)
    require(edges.shape == (301,), "Expected 301 histogram bin boundaries.")
    require(probability.shape == (4, 8, 300), "Unexpected histogram dimensions.")
    require(np.isfinite(edges).all() and np.all(np.diff(edges) > 0), "Invalid bins.")
    require(np.isfinite(probability).all(), "Non-finite histogram probabilities.")
    require((probability >= 0).all(), "Negative histogram probabilities.")
    require(
        np.allclose(probability.sum(axis=-1), 1, rtol=1e-6, atol=1e-7),
        "Each block/head histogram must sum to one.",
    )
    return summary, matrices, edges, probability


def annotation_color(cmap, norm, value: float) -> str:
    red, green, blue, _ = cmap(norm(value))
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
              for v in (red, green, blue)]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return "white" if luminance < 0.179 else "#17212b"


def render(root: Path) -> dict:
    summary, matrices, bin_edges, probability = load_inputs(root)
    edges_count = int(summary.get("total_edges", 55_980_536))
    cores_count = int(summary.get("core_count", 14))
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(16, 9.4), layout="constrained")
    layout = fig.add_gridspec(2, 1, height_ratios=[1, 1.08], hspace=0.13)
    top = layout[0].subgridspec(1, 4, wspace=0.035)
    bottom = layout[1].subgridspec(1, 3, wspace=0.08)

    density = probability / np.diff(bin_edges)[None, None, :]
    occupied = np.flatnonzero(np.any(probability > 0, axis=(0, 1)))
    require(len(occupied) > 0, "Empty histogram.")
    left = max(float(bin_edges[0]), float(bin_edges[occupied[0]]) - 0.08)
    right = min(float(bin_edges[-1]), float(bin_edges[occupied[-1] + 1]) + 0.08)
    ymax = float(density.max()) * 1.08
    distribution_axes = []
    for block in range(4):
        ax = fig.add_subplot(top[0, block])
        distribution_axes.append(ax)
        for head in range(8):
            ax.stairs(density[block, head], bin_edges, color="#7692a4",
                      alpha=0.43, linewidth=0.8)
        average = density[block].mean(axis=0)
        ax.stairs(average, bin_edges, color="#174f72", linewidth=1.9)
        ax.stairs(average, bin_edges, color="#174f72", alpha=0.075, fill=True)
        ax.axvline(1, color="#bb623b", linewidth=1.3, linestyle=(0, (4, 3)))
        ax.set(xlim=(left, right), ylim=(0, ymax), xlabel="Geometry multiplier g")
        ax.set_title(f"Block {block + 1}", loc="left", fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.16, linewidth=0.6)
        ax.set_axisbelow(True)
        if block == 0:
            ax.set_ylabel("Sampled probability density")
        else:
            ax.tick_params(labelleft=False)
    distribution_axes[0].legend(
        handles=[Line2D([0], [0], color="#7692a4", lw=0.8, label="Individual heads"),
                 Line2D([0], [0], color="#174f72", lw=1.9, label="Mean over heads"),
                 Line2D([0], [0], color="#bb623b", lw=1.3, ls="--", label="g = 1")],
        loc="upper left", fontsize=8, frameon=False,
    )

    scale_max = max(float(matrices["total_rms"].max()), 1e-6)
    panels = [
        ("total_rms", "Departure from g = 1", "RMS departure", "Blues", scale_max),
        ("dynamic_rms", "Variation across edges", "RMS around each dimension’s mean", "Blues", scale_max),
        ("dynamic_fraction", "Fraction due to edge variation", "Share of squared departure", "viridis", 1.0),
    ]
    for index, (key, title, units, cmap, vmax) in enumerate(panels):
        ax = fig.add_subplot(bottom[0, index])
        values = matrices[key]
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(8), [str(h) for h in range(1, 9)])
        ax.set_yticks(range(4), [f"Block {b}" for b in range(1, 5)])
        ax.set_xlabel("Attention head")
        ax.set_title(title, pad=12, fontweight="bold")
        ax.tick_params(length=0)
        for block in range(4):
            for head in range(8):
                value = values[block, head]
                label = "—" if not np.isfinite(value) else (
                    f"{value:.0%}" if key == "dynamic_fraction" else f"{value:.3f}"
                )
                color = "#17212b" if not np.isfinite(value) else annotation_color(im.cmap, im.norm, value)
                ax.text(head, block, label, ha="center", va="center", color=color,
                        fontsize=9, fontweight="normal")
        bar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.025)
        bar.set_label(units, fontsize=9)
        if key == "dynamic_fraction":
            bar.set_ticks([0, 0.25, 0.5, 0.75, 1], labels=["0%", "25%", "50%", "75%", "100%"])

    fig.suptitle("How geometry changes Q–K dimension weights", fontsize=19, fontweight="bold")
    fig.supxlabel(
        f"Top: deterministic sample of 4,096 edges per core.  Bottom: all {edges_count:,} directed edges across {cores_count} cores.\n"
        "Equal core and dimension weighting · geometry only · epoch 200, seed 0 · FP32\n"
        "Edge variation includes within-core and between-core variation; it does not directly measure the effect on attention.",
        fontsize=10, linespacing=1.55,
    )
    png = root / "g_variation.png"
    pdf = root / "g_variation.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf, metadata={"Title": "Geometry modulation variation in the latest trained SO2 model"})
    plt.close(fig)
    provenance = {
        "schema": "so2_geometry_modulation_figure_provenance_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "render_script_sha256": sha256(Path(__file__).resolve()),
        "summary_sha256": sha256(root / "summary.json"),
        "distributions_sha256": sha256(root / "distributions.npz"),
        "png_sha256": sha256(png), "pdf_sha256": sha256(pdf),
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
        "full_moment_directed_edges": edges_count, "core_count": cores_count,
        "distribution_sampling": "Deterministic 4096-edge sample per core; equal-core probabilities; all dimensions per head.",
        "heatmap_weighting": "Equal core and dimension; RMS after averaging squared values.",
        "first_two_heatmaps_share_rms_color_scale": True,
    }
    (root / "figure_provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    render(arguments.root.resolve())
