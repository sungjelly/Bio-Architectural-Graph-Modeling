from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import spatial_benchmark.so2_hidden_layer_progression_figures as figures


def _spatial_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for offset, core_number in enumerate(figures.SO2_CORE_NUMBERS):
        for cell in range(3):
            rows.append(
                {
                    "core_number": core_number,
                    "x_um": float(offset * 100 + cell * 10),
                    "y_um": float(offset * 75 + cell * 6),
                }
            )
    return pd.DataFrame(rows)


def _progression_labels() -> dict[str, np.ndarray]:
    return {
        "h0": np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
        "h1": np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
        "h2": np.array([0, 1, 0, 1, 2, 2, 3, 3], dtype=np.int64),
        "h3": np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64),
        "hL": np.array([0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64),
    }


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_layer_spatial_png_is_layer_qualified_and_png_only(
    tmp_path: Path,
) -> None:
    frame = _spatial_frame()
    before = frame.copy(deep=True)
    labels = np.tile(np.array([0, 1, 0], dtype=np.int64), len(figures.SO2_CORE_NUMBERS))
    output_path = tmp_path / "contextual_h1_leiden_resolution_1p0_spatial_14cores.png"

    summary = figures.render_layer_spatial_png(
        frame,
        labels,
        {"C0": "#0072B2", "C1": "#D55E00"},
        output_path,
        layer="h1",
        dpi=24,
        expected_counts_by_core={core: 3 for core in figures.SO2_CORE_NUMBERS},
    )

    _assert_png(output_path)
    assert not list(tmp_path.glob("*.pdf"))
    assert summary.layer == "h1"
    assert summary.point_count == len(frame)
    assert summary.cluster_labels == ("H1C0", "H1C1")
    assert summary.core_numbers == tuple(range(15, 29))
    pd.testing.assert_frame_equal(frame, before)


def test_spatial_png_rejects_non_png_and_incomplete_palette(tmp_path: Path) -> None:
    frame = _spatial_frame()
    labels = np.zeros(len(frame), dtype=np.int64)
    with pytest.raises(figures.HiddenLayerProgressionFigureError, match=".png"):
        figures.render_layer_spatial_png(
            frame,
            labels,
            {0: "#0072B2"},
            tmp_path / "h1.pdf",
            layer="h1",
            dpi=24,
        )

    labels[-1] = 1
    with pytest.raises(figures.HiddenLayerProgressionFigureError, match="missing"):
        figures.render_layer_spatial_png(
            frame,
            labels,
            {0: "#0072B2"},
            tmp_path / "h1.png",
            layer="h1",
            dpi=24,
        )


def test_render_layer_spatial_png_accepts_intrinsic_h0(tmp_path: Path) -> None:
    frame = _spatial_frame()
    output_path = tmp_path / "intrinsic_h0_leiden_resolution_1p0_spatial_14cores.png"

    summary = figures.render_layer_spatial_png(
        frame,
        np.zeros(len(frame), dtype=np.int64),
        {0: "#0072B2"},
        output_path,
        layer="h0",
        dpi=12,
    )

    _assert_png(output_path)
    assert summary.layer == "h0"
    assert summary.cluster_labels == ("H0C0",)


def test_adjacent_transition_metrics_and_normalizations_are_aligned() -> None:
    transitions = figures.compute_adjacent_partition_transitions(_progression_labels())

    assert len(transitions) == 4
    assert transitions[0].source_layer == "h0"
    assert transitions[-1].target_layer == "hL"
    assert transitions[0].adjusted_rand_index == pytest.approx(1.0)
    assert transitions[0].normalized_mutual_information == pytest.approx(1.0)
    assert transitions[-1].adjusted_rand_index == pytest.approx(1.0)
    assert transitions[-1].normalized_mutual_information == pytest.approx(1.0)
    assert transitions[1].adjusted_rand_index < 1.0
    assert transitions[1].normalized_mutual_information < 1.0
    for transition in transitions:
        assert int(transition.counts.sum()) == 8
        np.testing.assert_allclose(transition.split_fractions.sum(axis=1), 1.0)
        np.testing.assert_allclose(transition.merge_fractions.sum(axis=0), 1.0)
        assert all(label.startswith("H") for label in transition.source_labels)
        assert all(label.startswith("H") for label in transition.target_labels)


def test_render_transition_heatmap_png_and_reject_misaligned_labels(
    tmp_path: Path,
) -> None:
    labels = _progression_labels()
    output_path = tmp_path / "hidden_layer_cluster_transition_heatmaps_h0_to_hL.png"

    summary = figures.render_transition_heatmap_png(
        labels,
        output_path,
        dpi=24,
        annotation_threshold=0.5,
    )

    _assert_png(output_path)
    assert not list(tmp_path.glob("*.pdf"))
    assert summary.point_count == 8
    assert summary.layer_order == ("h0", "h1", "h2", "h3", "hL")
    assert len(summary.transitions) == 4

    misaligned = dict(labels)
    misaligned["h3"] = misaligned["h3"][:-1]
    with pytest.raises(figures.HiddenLayerProgressionFigureError, match="same.*length"):
        figures.compute_adjacent_partition_transitions(misaligned)


def test_progression_requires_every_layer_and_rejects_missing_labels() -> None:
    labels = _progression_labels()
    labels.pop("h2")
    with pytest.raises(figures.HiddenLayerProgressionFigureError, match="missing layers"):
        figures.compute_adjacent_partition_transitions(labels)

    labels = _progression_labels()
    labels["h2"] = labels["h2"].astype(float)
    labels["h2"][0] = np.nan
    with pytest.raises(figures.HiddenLayerProgressionFigureError, match="missing values"):
        figures.compute_adjacent_partition_transitions(labels)
