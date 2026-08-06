from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark.pooled_references import (
    fit_equal_core_hybrid_count_references,
)


ALIASES = ("ANC-01", "ANC-02")


def _counts() -> dict[str, np.ndarray]:
    return {
        "ANC-01": np.asarray(
            [[0, 1], [1, 2], [1, 4], [2, 8]], dtype=np.int32
        ),
        "ANC-02": np.asarray(
            [[0, 1], [0, 2], [0, 4], [4, 16], [4, 32], [4, 32]],
            dtype=np.int32,
        ),
    }


def test_equal_core_reference_does_not_weight_by_cell_count() -> None:
    result = fit_equal_core_hybrid_count_references(
        _counts(),
        expression_mean=np.zeros(2),
        expression_scale=np.ones(2),
        expected_aliases=ALIASES,
    )

    # Gene zero: Jeffreys probabilities are 3.5/5 and 3.5/7.  Their equal-core
    # mean differs from the 6.5/11 cell-pooled probability.
    assert result.detection_probability[0] == pytest.approx(
        0.5 * (3.5 / 5.0 + 3.5 / 7.0)
    )
    assert result.detection_probability[0] != pytest.approx(6.5 / 11.0)
    assert result.audit["core_weighting"] == "equal"


def test_continuous_reference_is_median_of_core_medians() -> None:
    result = fit_equal_core_hybrid_count_references(
        _counts(),
        expression_mean=np.zeros(2),
        expression_scale=np.ones(2),
        expected_aliases=ALIASES,
    )
    first = np.median(np.log1p(np.asarray([1, 1, 2], dtype=np.float64)))
    second = np.median(np.log1p(np.asarray([4, 4, 4], dtype=np.float64)))
    assert result.positive_continuous_standardized[0] == pytest.approx(
        np.median([first, second])
    )


def test_reference_alias_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing"):
        fit_equal_core_hybrid_count_references(
            {"ANC-01": _counts()["ANC-01"]},
            expression_mean=np.zeros(2),
            expression_scale=np.ones(2),
            expected_aliases=ALIASES,
        )


def test_reference_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="positive"):
        fit_equal_core_hybrid_count_references(
            _counts(),
            expression_mean=np.zeros(2),
            expression_scale=np.asarray([1.0, 0.0]),
            expected_aliases=ALIASES,
        )
