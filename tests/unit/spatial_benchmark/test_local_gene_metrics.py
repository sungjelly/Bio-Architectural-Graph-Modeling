"""Count metrics retain marginal/conditional semantics and pool by support."""

import json
import math

import pytest
import torch

from spatial_benchmark.hurdle_negative_binomial import (
    masked_hurdle_negative_binomial_nll,
)
from spatial_benchmark.local_gene_metrics import CountMetricAccumulator, LocalGeneCountMetrics
from spatial_benchmark.local_gene_residual import LocalGeneResidualOutput


def _fixture() -> tuple[LocalGeneResidualOutput, torch.Tensor, torch.Tensor]:
    probability = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    positive_mean = torch.tensor([[2.0, 4.0], [3.0, 4.0]])
    output = LocalGeneResidualOutput(
        nonzero_logits=torch.logit(probability), nb_mean=positive_mean - 1.0,
        theta=torch.ones(2), positive_conditional_mean=positive_mean,
        marginal_mean=probability * positive_mean,
    )
    return output, torch.tensor([[0, 4], [1, 0]]), torch.ones(2, 2, dtype=torch.bool)


def test_metrics_separate_marginal_and_positive_conditional_predictions() -> None:
    output, target, mask = _fixture()
    accumulator = CountMetricAccumulator()
    accumulator.update(output, target, mask)
    actual = accumulator.compute()
    assert actual["masked_entries"] == 4
    assert actual["positive_entries"] == 2
    assert actual["raw_count_mae"] == pytest.approx(1.0)
    assert actual["raw_count_rmse"] == pytest.approx(math.sqrt(5.5 / 4))
    assert actual["zero_raw_count_mae"] == pytest.approx(1.25)
    assert actual["positive_marginal_raw_count_mae"] == pytest.approx(0.75)
    assert actual["positive_marginal_raw_count_rmse"] == pytest.approx(math.sqrt(0.625))
    assert actual["positive_conditional_raw_count_mae"] == pytest.approx(1.0)
    assert actual["positive_conditional_raw_count_rmse"] == pytest.approx(math.sqrt(2))
    assert actual["observed_count_mean"] == pytest.approx(1.25)
    assert actual["predicted_count_mean"] == pytest.approx(1.75)
    assert actual["detection_brier"] == pytest.approx(0.15625)
    assert actual["detection_precision"] == pytest.approx(2 / 3)
    assert actual["detection_recall"] == 1.0
    assert actual["detection_f1"] == pytest.approx(0.8)
    assert actual["detection_average_precision"] == pytest.approx(5 / 6)
    assert actual["detection_auroc"] == pytest.approx(0.875)
    expected_nll = masked_hurdle_negative_binomial_nll(
        output.nonzero_logits, output.nb_mean, output.theta, target, mask
    )
    assert actual["count_nll"] == pytest.approx(expected_nll.item())
    log_error = torch.log1p(output.marginal_mean) - torch.log1p(target.float())
    assert actual["log1p_count_mae"] == pytest.approx(log_error.abs().mean().item())
    assert actual["log1p_count_rmse"] == pytest.approx(log_error.square().mean().sqrt().item())
    json.dumps(actual, allow_nan=False)


def test_merged_disjoint_masks_match_whole_mask_including_detection_histograms() -> None:
    output, target, mask = _fixture()
    whole = LocalGeneCountMetrics()
    whole.update(output, target, mask)
    first = LocalGeneCountMetrics()
    first_mask = mask.clone()
    first_mask[0, :] = False
    first.update(output, target, first_mask)
    second = LocalGeneCountMetrics()
    second.update(output, target, ~first_mask)
    first.merge(second)
    for name, expected in whole.finalize().items():
        assert first.finalize()[name] == pytest.approx(expected, abs=1e-7), name


@pytest.mark.parametrize("positive", [False, True])
def test_single_class_support_is_explicit_and_all_metrics_stay_finite(positive: bool) -> None:
    output, target, mask = _fixture()
    target.fill_(int(positive))
    accumulator = LocalGeneCountMetrics()
    accumulator.update(output, target, mask)
    actual = accumulator.finalize()
    assert actual["detection_auroc_defined"] == 0
    assert actual["detection_auroc"] == 0.0
    assert actual["positive_metrics_defined"] == int(positive)
    assert actual["zero_metrics_defined"] == int(not positive)
    assert actual["detection_average_precision"] == float(positive)
    assert all(math.isfinite(value) for value in actual.values())


def test_empty_subgroup_is_no_op_and_empty_finalization_is_rejected() -> None:
    output, target, mask = _fixture()
    accumulator = LocalGeneCountMetrics()
    accumulator.update(output, target, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="empty metric accumulator"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="bin counts must agree"):
        accumulator.merge(LocalGeneCountMetrics(256))

