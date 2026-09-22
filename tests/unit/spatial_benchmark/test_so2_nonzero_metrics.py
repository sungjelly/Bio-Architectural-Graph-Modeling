"""Synthetic correctness checks for the observed-positive accuracy audit."""

import json

import numpy as np
import pytest

from spatial_benchmark.so2_nonzero_metrics import NonzeroMetricAccumulator, STRATA


def _update(accumulator, counts, truth, prediction, predicted_log):
    accumulator.update(
        true_counts=counts,
        true_standardized=truth,
        pred_standardized=prediction,
        pred_log1p=predicted_log,
    )


def test_positive_membership_uses_observed_count_and_correct_denominator():
    accumulator = NonzeroMetricAccumulator()
    # A positive observed count can have negative standardized expression.
    _update(accumulator, [0, 1, 2, 0], [-2, -1, 0, -3], [-1, 1, 3, -3], [0, 0, 0, 0])
    result = accumulator.result()
    positive = result["strata"]["positive"]
    zero = result["strata"]["zero"]
    assert positive["n"] == 2
    assert positive["mse_standardized"] == pytest.approx(6.5)
    assert positive["mae_standardized"] == pytest.approx(2.5)
    assert positive["huber_standardized"] == pytest.approx(2.0)
    assert positive["bias_standardized"] == pytest.approx(2.5)
    assert positive["mse_log1p"] == pytest.approx((np.log(2)**2 + np.log(3)**2) / 2)
    assert zero["mse_standardized"] == pytest.approx(0.5)
    assert result["strata"]["all"]["mse_standardized"] == pytest.approx(3.5)
    assert positive["exact_count_accuracy"] == 0.0
    assert zero["exact_count_accuracy"] == 1.0
    assert result["detection"]["balanced_accuracy"] == 0.5


def test_perfect_predictions_recover_every_stratum_and_detection():
    counts = np.array([0, 1, 2, 3, 4, 7, 8, 30])
    truth = (np.log1p(counts) - 1.7) / 0.4
    accumulator = NonzeroMetricAccumulator()
    _update(accumulator, counts, truth, truth, np.log1p(counts))
    result = accumulator.result()
    for name in STRATA:
        assert result["strata"][name]["mse_standardized"] == 0
        assert result["strata"][name]["mse_log1p"] == 0
        assert result["strata"][name]["exact_count_accuracy"] == 1
    assert result["strata"]["count_4_7"]["n"] == 2
    assert result["strata"]["count_8_plus"]["n"] == 2
    assert result["detection"]["positive_recall"] == 1
    assert result["detection"]["zero_specificity"] == 1
    assert result["detection"]["positive_precision"] == 1


def test_half_up_boundaries_and_detection_use_fixed_half_count_threshold():
    boundary = np.log1p(0.5)
    below = np.nextafter(boundary, -np.inf)
    accumulator = NonzeroMetricAccumulator()
    # Decoded predictions: 0, 1, 1, 2, 2, 0, 0, 1. Half boundaries round up.
    _update(
        accumulator,
        [0, 0, 1, 1, 2, 0, 1, 1],
        np.zeros(8), np.zeros(8),
        [below, boundary, boundary, np.log1p(1.5), np.log1p(1.5), -50, below, boundary],
    )
    result = accumulator.result()
    assert result["strata"]["zero"]["n_exact_count"] == 2
    assert result["strata"]["positive"]["n_exact_count"] == 3
    detection = result["detection"]
    assert detection["true_positive_count"] == 4
    assert detection["false_positive_count"] == 1
    assert detection["true_negative_count"] == 2
    assert detection["false_negative_count"] == 1
    assert detection["positive_recall"] == 4 / 5
    assert detection["zero_specificity"] == 2 / 3
    assert detection["positive_precision"] == 4 / 5


def test_unclipped_log_errors_and_large_log_prediction_do_not_require_exp():
    accumulator = NonzeroMetricAccumulator()
    with np.errstate(over="raise"):
        _update(accumulator, [0, 1], [0, 1], [0, 1], [-2, 1000])
    result = accumulator.result()
    assert result["strata"]["zero"]["mse_log1p"] == 4
    assert result["strata"]["zero"]["exact_count_accuracy"] == 1
    assert result["strata"]["positive"]["exact_count_accuracy"] == 0
    assert result["strata"]["positive"]["mse_log1p"] == pytest.approx((1000 - np.log(2))**2)


def test_empty_strata_are_null_and_do_not_become_zero_error():
    accumulator = NonzeroMetricAccumulator()
    _update(accumulator, [], [], [], [])
    result = accumulator.result()
    for stratum in result["strata"].values():
        assert stratum["n"] == 0
        assert stratum["mse_standardized"] is None
        assert stratum["exact_count_accuracy"] is None
    assert result["detection"]["balanced_accuracy"] is None
    json.dumps(result, allow_nan=False)
    _update(accumulator, [0], [-2], [-1], [0])
    result = accumulator.result()
    assert result["strata"]["positive"]["mse_standardized"] is None
    assert result["detection"]["positive_recall"] is None
    assert result["detection"]["positive_precision"] is None
    assert result["detection"]["zero_specificity"] == 1


def test_streaming_merge_matches_direct_metrics_and_weighted_decomposition():
    rng = np.random.default_rng(19)
    counts = rng.integers(0, 12, 1031)
    truth = (np.log1p(counts) - 0.8) / 1.2
    prediction = truth + rng.normal(size=counts.size)
    log_prediction = prediction * 1.2 + 0.8
    direct = NonzeroMetricAccumulator()
    _update(direct, counts, truth, prediction, log_prediction)
    merged = NonzeroMetricAccumulator()
    for start, end in [(0, 2), (2, 17), (17, 430), (430, len(counts))]:
        part = NonzeroMetricAccumulator()
        _update(part, counts[start:end], truth[start:end], prediction[start:end], log_prediction[start:end])
        merged.merge(part)
    result = merged.result()
    direct_result = direct.result()
    for name in STRATA:
        assert result["strata"][name]["n"] == direct_result["strata"][name]["n"]
        for metric in ["mse_standardized", "mae_standardized", "huber_standardized", "mse_log1p", "exact_count_accuracy"]:
            assert result["strata"][name][metric] == pytest.approx(direct_result["strata"][name][metric])
    assert result["detection"] == direct_result["detection"]
    strata = result["strata"]
    weighted_mse = sum(strata[name]["n"] * strata[name]["mse_standardized"] for name in ["zero", "positive"]) / len(counts)
    assert weighted_mse == pytest.approx(np.mean((prediction - truth)**2))
    assert result["decomposition"]["metric_sum_partition_verified"]
    assert abs(result["decomposition"]["zero_positive_mse_residual"]) < 1e-12


@pytest.mark.parametrize("counts,truth,prediction,predicted_log", [
    ([-1], [0], [0], [0]),
    ([0.5], [0], [0], [0]),
    ([0], [np.nan], [0], [0]),
    ([0], [0], [np.inf], [0]),
    ([0], [0], [0], [np.inf]),
    ([0], [0, 1], [0], [0]),
    ([[0]], [[0]], [[0]], [[0]]),
    ([0], [0], [1e200], [0]),
])
def test_invalid_inputs_raise_without_dropping_entries_or_mutating(counts, truth, prediction, predicted_log):
    accumulator = NonzeroMetricAccumulator()
    _update(accumulator, [1], [0], [0], [np.log(2)])
    before = accumulator.result()
    with pytest.raises(ValueError):
        _update(accumulator, counts, truth, prediction, predicted_log)
    assert accumulator.result() == before


def test_merge_refuses_incompatible_huber_definitions():
    with pytest.raises(ValueError, match="different Huber"):
        NonzeroMetricAccumulator().merge(NonzeroMetricAccumulator(huber_delta=2))
