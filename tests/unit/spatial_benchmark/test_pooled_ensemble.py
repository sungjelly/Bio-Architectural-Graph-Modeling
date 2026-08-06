from __future__ import annotations

import pytest
import torch

from spatial_benchmark.hybrid_count import (
    decode_count_states,
    split_hybrid_prediction,
)
from spatial_benchmark.hybrid_count_metrics import (
    evaluate_hybrid_count_output,
    fit_hybrid_count_references,
)
from spatial_benchmark.pooled_ensemble import (
    HybridCountEnsembleAccumulator,
    ensemble_hybrid_count_predictions,
)


def _prediction(
    *,
    detection: float,
    ordinal: float,
    continuous: float,
    nodes: int = 2,
    genes: int = 3,
) -> torch.Tensor:
    output = torch.empty((nodes, genes, 8), dtype=torch.float32)
    output[..., 0] = detection
    output[..., 1:7] = ordinal
    output[..., 7] = continuous
    return output


def test_probability_heads_are_averaged_as_probabilities_not_logits() -> None:
    first = _prediction(detection=-3.0, ordinal=-2.0, continuous=4.0)
    second = _prediction(detection=1.0, ordinal=3.0, continuous=-2.0)

    ensemble = ensemble_hybrid_count_predictions(
        [first, second], expected_member_count=2
    )
    detection, ordinal, continuous = split_hybrid_prediction(ensemble)

    expected_detection = torch.logit(
        (torch.sigmoid(first[..., 0]) + torch.sigmoid(second[..., 0])) / 2.0
    )
    expected_ordinal = torch.logit(
        (
            torch.sigmoid(first[..., 1:7])
            + torch.sigmoid(second[..., 1:7])
        )
        / 2.0
    )
    torch.testing.assert_close(detection, expected_detection)
    torch.testing.assert_close(ordinal, expected_ordinal)
    torch.testing.assert_close(continuous, torch.ones_like(continuous))
    assert not torch.allclose(
        detection, (first[..., 0] + second[..., 0]) / 2.0
    )
    assert not torch.allclose(
        ordinal, (first[..., 1:7] + second[..., 1:7]) / 2.0
    )


def test_decoding_is_applied_to_the_averaged_probability_prediction() -> None:
    logits = torch.logit(torch.tensor([0.99, 0.49, 0.49]))
    members = [
        _prediction(
            detection=float(logit),
            ordinal=float(logit),
            continuous=0.0,
            nodes=1,
            genes=1,
        )
        for logit in logits
    ]
    individual_states = torch.stack(
        [decode_count_states(member) for member in members]
    )
    assert int((individual_states > 0).sum()) == 1

    ensemble = ensemble_hybrid_count_predictions(
        members, expected_member_count=3
    )
    # Mean probability is > 0.5 even though a majority of individually
    # decoded members are below the hard threshold.
    assert decode_count_states(ensemble).item() == 7


def test_nonlinear_metrics_are_recomputed_from_ensemble_prediction() -> None:
    target = torch.tensor(
        [[0.0], [1.0], [2.0], [3.0], [4.0], [8.0], [16.0], [32.0]]
    )
    mask = torch.ones_like(target, dtype=torch.bool)
    expression_mean = torch.zeros(1)
    expression_scale = torch.ones(1)
    references = fit_hybrid_count_references(
        target.numpy(),
        expression_mean=expression_mean.numpy(),
        expression_scale=expression_scale.numpy(),
    )
    first = _prediction(
        detection=4.0,
        ordinal=3.0,
        continuous=3.0,
        nodes=8,
        genes=1,
    )
    second = _prediction(
        detection=-1.0,
        ordinal=-2.0,
        continuous=-1.0,
        nodes=8,
        genes=1,
    )
    member_evaluations = [
        evaluate_hybrid_count_output(
            prediction,
            target,
            mask,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
            references=references,
        )
        for prediction in (first, second)
    ]
    ensemble_prediction = ensemble_hybrid_count_predictions(
        [first, second], expected_member_count=2
    )
    ensemble_evaluation = evaluate_hybrid_count_output(
        ensemble_prediction,
        target,
        mask,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
        references=references,
    )

    mean_member_loss = sum(
        item.metrics["hybrid_loss"] for item in member_evaluations
    ) / 2.0
    assert ensemble_evaluation.metrics["hybrid_loss"] != pytest.approx(
        mean_member_loss
    )
    direct_loss = evaluate_hybrid_count_output(
        ensemble_prediction.clone(),
        target,
        mask,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
        references=references,
    ).metrics["hybrid_loss"]
    assert ensemble_evaluation.metrics["hybrid_loss"] == pytest.approx(
        direct_loss
    )


@pytest.mark.parametrize(
    "invalid",
    [
        torch.zeros((2, 3, 7), dtype=torch.float32),
        torch.zeros((2, 3, 9), dtype=torch.float32),
        torch.zeros((0, 3, 8), dtype=torch.float32),
    ],
)
def test_rejects_invalid_shape_or_cumulative_slot_count(
    invalid: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="\\[nodes, genes, 8\\]"):
        ensemble_hybrid_count_predictions(
            [invalid], expected_member_count=1
        )


def test_rejects_shape_nonfinite_and_member_count_failures() -> None:
    valid = _prediction(detection=0.0, ordinal=0.0, continuous=0.0)
    accumulator = HybridCountEnsembleAccumulator(2)
    accumulator.update(valid)
    with pytest.raises(ValueError, match="shape mismatch"):
        accumulator.update(
            _prediction(
                detection=0.0,
                ordinal=0.0,
                continuous=0.0,
                nodes=3,
            )
        )
    with pytest.raises(ValueError, match="member count mismatch"):
        accumulator.finalize()

    nonfinite = valid.clone()
    nonfinite[0, 0, 4] = torch.nan
    with pytest.raises(FloatingPointError, match="non-finite"):
        ensemble_hybrid_count_predictions(
            [nonfinite], expected_member_count=1
        )

    complete = HybridCountEnsembleAccumulator(1)
    complete.update(valid)
    with pytest.raises(ValueError, match="more ensemble members"):
        complete.update(valid)
    with pytest.raises(ValueError, match="member count mismatch"):
        ensemble_hybrid_count_predictions(
            [valid], expected_member_count=2
        )


@pytest.mark.parametrize("count", [0, -1])
def test_rejects_nonpositive_expected_member_count(count: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        HybridCountEnsembleAccumulator(count)


@pytest.mark.parametrize("count", [True, 1.5])
def test_rejects_noninteger_expected_member_count(count: object) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        HybridCountEnsembleAccumulator(count)  # type: ignore[arg-type]


def test_extreme_finite_member_logits_produce_finite_ensemble_logits() -> None:
    first = _prediction(
        detection=1.0e4, ordinal=-1.0e4, continuous=0.0
    )
    second = first.clone()
    ensemble = ensemble_hybrid_count_predictions(
        [first, second], expected_member_count=2
    )
    assert bool(torch.isfinite(ensemble).all())


def test_rare_interior_probabilities_are_not_inflated_to_float32_epsilon() -> None:
    rare_probability = torch.tensor(1.0e-12, dtype=torch.float32)
    rare_logit = float(torch.logit(rare_probability))
    member = _prediction(
        detection=rare_logit,
        ordinal=rare_logit,
        continuous=0.0,
        nodes=1,
        genes=1,
    )

    ensemble = ensemble_hybrid_count_predictions(
        [member], expected_member_count=1
    )
    detection, ordinal, _ = split_hybrid_prediction(ensemble)
    torch.testing.assert_close(
        torch.sigmoid(detection),
        rare_probability.expand_as(detection),
        rtol=1.0e-5,
        atol=0.0,
    )
    torch.testing.assert_close(
        torch.sigmoid(ordinal),
        rare_probability.expand_as(ordinal),
        rtol=1.0e-5,
        atol=0.0,
    )
    positive_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        detection,
        torch.ones_like(detection),
    )
    expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        member[..., 0],
        torch.ones_like(member[..., 0]),
    )
    assert positive_bce == pytest.approx(float(expected_bce), rel=1.0e-6)
    assert float(positive_bce) > 25.0


def test_exact_probability_endpoints_map_to_finite_adjacent_logits() -> None:
    endpoints = torch.tensor([0.0, 1.0], dtype=torch.float32)
    logits = __import__(
        "spatial_benchmark.pooled_ensemble",
        fromlist=["_finite_probability_logits"],
    )._finite_probability_logits(endpoints)
    assert bool(torch.isfinite(logits).all())
    assert float(logits[0]) < -100.0
    assert float(logits[1]) > 16.0


def test_streaming_accumulation_is_deterministic_and_matches_convenience_api() -> None:
    generator = torch.Generator().manual_seed(1776)
    members = [
        torch.randn((4, 5, 8), generator=generator) for _ in range(7)
    ]
    accumulator = HybridCountEnsembleAccumulator(
        7, accumulation_device="cpu"
    )
    for member in members:
        accumulator.update(member)
    streamed = accumulator.finalize()

    convenience = ensemble_hybrid_count_predictions(
        (member for member in members),
        expected_member_count=7,
        accumulation_device="cpu",
    )
    repeated = ensemble_hybrid_count_predictions(
        iter(members),
        expected_member_count=7,
        accumulation_device="cpu",
    )
    assert torch.equal(streamed, convenience)
    assert torch.equal(convenience, repeated)
