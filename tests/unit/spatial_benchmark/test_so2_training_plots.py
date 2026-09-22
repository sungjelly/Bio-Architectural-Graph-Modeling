from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from spatial_benchmark.so2_training_plots import (
    SO2TrainingPlotError,
    write_so2_training_plots,
)


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_scalar_records(
    root: Path,
    *,
    block_count: int,
    epoch_count: int = 6,
) -> None:
    _write_csv(
        root / "results/epoch_metrics.csv",
        ("global_epoch", "equal_core_mean_masked_huber"),
        [
            {"global_epoch": epoch, "equal_core_mean_masked_huber": 0.3 / epoch}
            for epoch in range(1, epoch_count + 1)
        ],
    )
    _write_csv(
        root / "results/gradient_direction_metrics.csv",
        (
            "global_epoch",
            "gradient_norm_mean_before_clip",
            "consecutive_optimizer_step_cosine_mean",
            "epoch_aggregate_gradient_cosine_to_previous_epoch",
        ),
        [
            {
                "global_epoch": epoch,
                "gradient_norm_mean_before_clip": 0.2 / epoch,
                "consecutive_optimizer_step_cosine_mean": 0.1 * epoch,
                "epoch_aggregate_gradient_cosine_to_previous_epoch": (
                    "" if epoch == 1 else -0.1 * epoch
                ),
            }
            for epoch in range(1, epoch_count + 1)
        ],
    )
    _write_csv(
        root / "results/gradient_direction_by_block.csv",
        (
            "global_epoch",
            "block_index",
            "epoch_aggregate_gradient_cosine_to_previous_epoch",
        ),
        [
            {
                "global_epoch": epoch,
                "block_index": block,
                "epoch_aggregate_gradient_cosine_to_previous_epoch": (
                    ""
                    if epoch == 1
                    else (block - (block_count - 1) / 2.0) / block_count
                ),
            }
            for epoch in range(1, epoch_count + 1)
            for block in range(block_count)
        ],
    )


def _assert_png_outputs(root: Path, result: dict[str, str]) -> None:
    assert result == {
        "loss_vs_epoch": "figures/loss_vs_epoch.png",
        "gradient_direction_vs_epoch": "figures/gradient_direction_vs_epoch.png",
    }
    for relative in result.values():
        path = root / relative
        assert path.is_file()
        assert path.stat().st_size > 10_000
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list((root / "figures").glob("*.writing"))


def test_default_writes_eight_block_loss_and_gradient_figures(
    tmp_path: Path,
) -> None:
    _write_scalar_records(tmp_path, block_count=8)

    result = write_so2_training_plots(tmp_path)

    _assert_png_outputs(tmp_path, result)


def test_writes_four_block_heatmap_with_matching_shape_and_extent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    _write_scalar_records(tmp_path, block_count=4)
    observed: dict[str, Any] = {}
    original_imshow = Axes.imshow
    original_set = Axes.set
    original_suptitle = Figure.suptitle

    def _capture_imshow(
        axis: Axes,
        values: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        observed["shape"] = values.shape
        observed["extent"] = kwargs.get("extent")
        return original_imshow(axis, values, *args, **kwargs)

    def _capture_set(axis: Axes, **kwargs: Any) -> Any:
        if "title" in kwargs:
            observed["loss_title"] = kwargs["title"]
        return original_set(axis, **kwargs)

    def _capture_suptitle(
        figure: Figure,
        title: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        observed["gradient_title"] = title
        return original_suptitle(figure, title, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", _capture_imshow)
    monkeypatch.setattr(Axes, "set", _capture_set)
    monkeypatch.setattr(Figure, "suptitle", _capture_suptitle)
    result = write_so2_training_plots(
        tmp_path,
        expected_blocks=4,
        cohort_label="SO1 14-core",
    )

    _assert_png_outputs(tmp_path, result)
    assert observed == {
        "shape": (4, 6),
        "extent": (0.5, 6.5, -0.5, 3.5),
        "loss_title": "SO1 14-core training loss",
        "gradient_title": (
            "SO1 14-core gradient magnitude and direction diagnostics"
        ),
    }


def test_four_block_rows_are_rejected_by_default_eight_block_contract(
    tmp_path: Path,
) -> None:
    _write_scalar_records(tmp_path, block_count=4)

    with pytest.raises(
        SO2TrainingPlotError,
        match="exactly 8 rows per epoch",
    ):
        write_so2_training_plots(tmp_path)


def test_four_block_order_validation_uses_the_configured_last_index(
    tmp_path: Path,
) -> None:
    _write_scalar_records(tmp_path, block_count=4)
    block_path = tmp_path / "results/gradient_direction_by_block.csv"
    with block_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[3]["block_index"] = "2"
    _write_csv(
        block_path,
        (
            "global_epoch",
            "block_index",
            "epoch_aggregate_gradient_cosine_to_previous_epoch",
        ),
        rows,
    )

    with pytest.raises(
        SO2TrainingPlotError,
        match="block 0 through 3",
    ):
        write_so2_training_plots(tmp_path, expected_blocks=4)


@pytest.mark.parametrize("expected_blocks", [0, -1, True, 4.0])
def test_expected_blocks_must_be_a_positive_integer(
    tmp_path: Path,
    expected_blocks: object,
) -> None:
    with pytest.raises(SO2TrainingPlotError, match="positive integer"):
        write_so2_training_plots(  # type: ignore[arg-type]
            tmp_path,
            expected_blocks=expected_blocks,
        )


@pytest.mark.parametrize("cohort_label", ["", "   ", None, 1])
def test_cohort_label_must_be_a_non_empty_string(
    tmp_path: Path,
    cohort_label: object,
) -> None:
    with pytest.raises(SO2TrainingPlotError, match="cohort_label"):
        write_so2_training_plots(  # type: ignore[arg-type]
            tmp_path,
            cohort_label=cohort_label,
        )
