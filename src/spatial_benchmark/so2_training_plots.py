"""Deterministic plots derived from durable SO2 training scalar records."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping

import numpy as np


LOSS_FIGURE_RELATIVE_PATH = Path("figures/loss_vs_epoch.png")
GRADIENT_FIGURE_RELATIVE_PATH = Path("figures/gradient_direction_vs_epoch.png")


class SO2TrainingPlotError(RuntimeError):
    """Raised when durable scalar records cannot support an honest plot."""


def _read_rows(path: Path, *, required: Iterable[str]) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise SO2TrainingPlotError(f"Required scalar CSV is unavailable: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = sorted(set(required) - set(columns))
        if missing:
            raise SO2TrainingPlotError(
                f"Scalar CSV {path.name} lacks columns: {', '.join(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise SO2TrainingPlotError(f"Scalar CSV is empty: {path}")
    return rows


def _finite_float(raw: object, *, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise SO2TrainingPlotError(f"{field} is not numeric.") from error
    if not math.isfinite(value):
        raise SO2TrainingPlotError(f"{field} must be finite.")
    return value


def _optional_float(raw: object, *, field: str) -> float:
    if raw in {None, ""}:
        return math.nan
    return _finite_float(raw, field=field)


def _one_based_epochs(rows: list[Mapping[str, str]], *, path: Path) -> np.ndarray:
    try:
        epochs = np.asarray([int(row["global_epoch"]) for row in rows], dtype=int)
    except (KeyError, TypeError, ValueError) as error:
        raise SO2TrainingPlotError(
            f"Scalar CSV {path.name} contains an invalid global epoch."
        ) from error
    expected = np.arange(1, len(rows) + 1, dtype=int)
    if not np.array_equal(epochs, expected):
        raise SO2TrainingPlotError(
            f"Scalar CSV {path.name} must contain contiguous epochs from one."
        )
    return epochs


def _atomic_save_figure(figure: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png.writing", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_so2_training_plots(
    run_root: str | Path,
    *,
    expected_blocks: int = 8,
    cohort_label: str = "SO2 14-core",
) -> dict[str, str]:
    """Create loss and gradient-direction figures from finalized scalar CSVs.

    ``expected_blocks`` binds the exact per-epoch block-row count and heatmap
    dimensions. Eight remains the default for the original untied-depth run.
    The plots are descriptive artifacts only. They do not feed back into
    optimization, checkpoint choice, or the plateau stopping calculation.
    """

    if (
        isinstance(expected_blocks, bool)
        or not isinstance(expected_blocks, int)
        or expected_blocks <= 0
    ):
        raise SO2TrainingPlotError("expected_blocks must be a positive integer.")
    if not isinstance(cohort_label, str) or not cohort_label.strip():
        raise SO2TrainingPlotError("cohort_label must be a non-empty string.")
    label = cohort_label.strip()

    root = Path(run_root).resolve(strict=True)
    epoch_path = root / "results" / "epoch_metrics.csv"
    gradient_path = root / "results" / "gradient_direction_metrics.csv"
    block_path = root / "results" / "gradient_direction_by_block.csv"

    epoch_rows = _read_rows(
        epoch_path,
        required=("global_epoch", "equal_core_mean_masked_huber"),
    )
    epochs = _one_based_epochs(epoch_rows, path=epoch_path)
    losses = np.asarray(
        [
            _finite_float(
                row["equal_core_mean_masked_huber"],
                field="equal_core_mean_masked_huber",
            )
            for row in epoch_rows
        ],
        dtype=float,
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    loss_figure, loss_axis = plt.subplots(figsize=(9.0, 5.2))
    loss_axis.plot(epochs, losses, color="#1f4e79", linewidth=1.15, alpha=0.72)
    if losses.size >= 5:
        rolling = np.convolve(losses, np.ones(5, dtype=float) / 5.0, mode="valid")
        loss_axis.plot(
            epochs[4:],
            rolling,
            color="#c44e52",
            linewidth=2.0,
            label="5-epoch mean",
        )
    loss_axis.axvline(150, color="#777777", linestyle="--", linewidth=1.0)
    loss_axis.set(
        xlabel="Completed global epoch",
        ylabel="Equal-core masked Huber loss",
        title=f"{label} training loss",
    )
    loss_axis.grid(alpha=0.2)
    if losses.size >= 5:
        loss_axis.legend(frameon=False)
    loss_figure.tight_layout()
    loss_output = root / LOSS_FIGURE_RELATIVE_PATH
    _atomic_save_figure(loss_figure, loss_output)
    plt.close(loss_figure)

    gradient_rows = _read_rows(
        gradient_path,
        required=(
            "global_epoch",
            "gradient_norm_mean_before_clip",
            "consecutive_optimizer_step_cosine_mean",
            "epoch_aggregate_gradient_cosine_to_previous_epoch",
        ),
    )
    gradient_epochs = _one_based_epochs(gradient_rows, path=gradient_path)
    if not np.array_equal(gradient_epochs, epochs):
        raise SO2TrainingPlotError(
            "Loss and full-gradient CSVs cover different completed epochs."
        )
    norms = np.asarray(
        [
            _finite_float(
                row["gradient_norm_mean_before_clip"],
                field="gradient_norm_mean_before_clip",
            )
            for row in gradient_rows
        ],
        dtype=float,
    )
    update_cosines = np.asarray(
        [
            _optional_float(
                row["consecutive_optimizer_step_cosine_mean"],
                field="consecutive_optimizer_step_cosine_mean",
            )
            for row in gradient_rows
        ],
        dtype=float,
    )
    epoch_cosines = np.asarray(
        [
            _optional_float(
                row["epoch_aggregate_gradient_cosine_to_previous_epoch"],
                field="epoch_aggregate_gradient_cosine_to_previous_epoch",
            )
            for row in gradient_rows
        ],
        dtype=float,
    )

    block_rows = _read_rows(
        block_path,
        required=(
            "global_epoch",
            "block_index",
            "epoch_aggregate_gradient_cosine_to_previous_epoch",
        ),
    )
    if len(block_rows) != len(epochs) * expected_blocks:
        raise SO2TrainingPlotError(
            "Block-gradient CSV must contain exactly "
            f"{expected_blocks} rows per epoch."
        )
    block_cosines = np.full((expected_blocks, len(epochs)), np.nan, dtype=float)
    for offset, row in enumerate(block_rows):
        epoch = int(row["global_epoch"])
        block = int(row["block_index"])
        if (
            epoch != offset // expected_blocks + 1
            or block != offset % expected_blocks
        ):
            raise SO2TrainingPlotError(
                "Block-gradient CSV ordering must be epoch then block 0 through "
                f"{expected_blocks - 1}."
            )
        block_cosines[block, epoch - 1] = _optional_float(
            row["epoch_aggregate_gradient_cosine_to_previous_epoch"],
            field="block epoch aggregate gradient cosine",
        )

    gradient_figure, axes = plt.subplots(
        3,
        1,
        figsize=(10.0, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.25, 1.35)},
    )
    axes[0].plot(epochs, norms, color="#4c72b0", linewidth=1.25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean gradient norm\n(pre-clip, log scale)")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        epochs,
        update_cosines,
        color="#55a868",
        linewidth=1.1,
        label="Consecutive updates (mean)",
    )
    axes[1].plot(
        epochs,
        epoch_cosines,
        color="#c44e52",
        linewidth=1.1,
        label="Aggregate vs previous epoch",
    )
    axes[1].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].set_ylabel("Full-gradient cosine")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].grid(alpha=0.2)

    image = axes[2].imshow(
        block_cosines,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=(
            0.5,
            len(epochs) + 0.5,
            -0.5,
            expected_blocks - 0.5,
        ),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[2].set_yticks(
        range(expected_blocks),
        labels=[f"block {index}" for index in range(expected_blocks)],
    )
    axes[2].set_xlabel("Completed global epoch")
    axes[2].set_ylabel("Graph block")
    gradient_figure.colorbar(
        image,
        ax=axes[2],
        label="Aggregate gradient cosine vs previous epoch",
        pad=0.02,
    )
    gradient_figure.suptitle(
        f"{label} gradient magnitude and direction diagnostics"
    )
    gradient_figure.tight_layout()
    gradient_output = root / GRADIENT_FIGURE_RELATIVE_PATH
    _atomic_save_figure(gradient_figure, gradient_output)
    plt.close(gradient_figure)

    return {
        "loss_vs_epoch": str(LOSS_FIGURE_RELATIVE_PATH),
        "gradient_direction_vs_epoch": str(GRADIENT_FIGURE_RELATIVE_PATH),
    }


__all__ = [
    "GRADIENT_FIGURE_RELATIVE_PATH",
    "LOSS_FIGURE_RELATIVE_PATH",
    "SO2TrainingPlotError",
    "write_so2_training_plots",
]
