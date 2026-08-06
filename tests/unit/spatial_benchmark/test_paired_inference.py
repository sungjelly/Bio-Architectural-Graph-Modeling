"""Tests for exact core-level paired inference utilities."""

from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark.paired_inference import (
    exact_one_sided_paired_sign_flip,
    holm_adjust,
    paired_relative_improvements,
)


def test_exact_sign_flip_enumerates_all_assignments_and_keeps_ties() -> None:
    favorable = exact_one_sided_paired_sign_flip([1.0] * 10)
    assert favorable.permutation_count == 2**10
    assert favorable.p_value == pytest.approx(1 / 2**10)
    assert favorable.favorable_units == 10
    assert favorable.tied_units == 0

    tied = exact_one_sided_paired_sign_flip([0.0] * 10)
    assert tied.p_value == 1.0
    assert tied.tied_units == 10


def test_holm_adjust_is_monotone_and_step_down() -> None:
    result = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
    assert result["a"]["holm_adjusted_p_value"] == pytest.approx(0.03)
    assert result["b"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert result["c"]["holm_adjusted_p_value"] == pytest.approx(0.06)
    assert result["a"]["reject"] is True
    assert result["b"]["reject"] is False
    assert result["c"]["reject"] is False


def test_paired_relative_improvement_preserves_one_value_per_unit() -> None:
    actual = paired_relative_improvements([2.0, 4.0], [1.0, 3.0])
    np.testing.assert_allclose(actual, [0.5, 0.25])
    with pytest.raises(ValueError, match="positive"):
        paired_relative_improvements([0.0], [0.0])
