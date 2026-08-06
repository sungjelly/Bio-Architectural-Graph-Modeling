"""Focused contracts for the detection plus continuous hurdle objective."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch.nn import functional as F  # noqa: E402

from spatial_benchmark.hurdle_continuous import (  # noqa: E402
    decode_hurdle_continuous_prediction,
    evaluate_hurdle_continuous_output,
    hurdle_continuous_loss,
    reconstruct_positive_counts,
    round_nonnegative_half_up,
    split_hurdle_continuous_prediction,
)
from spatial_benchmark.hybrid_count import (  # noqa: E402
    tokenize_raw_count_tensor,
)


_STATE_COUNTS = torch.tensor(
    [0, 1, 2, 3, 4, 8, 16, 32],
    dtype=torch.float32,
).reshape(-1, 1)


def _perfect_prediction(
    raw_target: torch.Tensor,
    *,
    detection_logit: float | None = None,
) -> torch.Tensor:
    prediction = torch.empty((*raw_target.shape, 2), dtype=torch.float32)
    if detection_logit is None:
        prediction[..., 0] = torch.where(
            raw_target > 0,
            torch.full_like(raw_target, 20.0),
            torch.full_like(raw_target, -20.0),
        )
    else:
        prediction[..., 0] = detection_logit
    prediction[..., 1] = torch.log1p(raw_target)
    return prediction


def test_loss_balances_detection_strata_and_equally_weights_components() -> None:
    raw_target = torch.tensor(
        [0, 0, 0, 0, 0, 1, 2, 4],
        dtype=torch.float32,
    ).reshape(-1, 1)
    mask = torch.ones_like(raw_target, dtype=torch.bool)
    prediction = _perfect_prediction(raw_target)
    prediction[:5, 0, 0] = 2.0
    prediction[5:, 0, 0] = -1.0
    prediction[5:, 0, 1] += 0.4

    loss = hurdle_continuous_loss(
        prediction,
        raw_target,
        mask,
        expression_mean=torch.zeros(1),
        expression_scale=torch.ones(1),
    )
    detection = prediction[..., 0][mask]
    zero = raw_target[mask] == 0
    positive = ~zero
    expected_detection = 0.5 * (
        F.binary_cross_entropy_with_logits(
            detection[zero],
            torch.zeros_like(detection[zero]),
        )
        + F.binary_cross_entropy_with_logits(
            detection[positive],
            torch.ones_like(detection[positive]),
        )
    )
    expected_huber = F.huber_loss(
        prediction[..., 1][mask][positive],
        torch.log1p(raw_target)[mask][positive],
        reduction="mean",
        delta=1.0,
    )

    assert loss.detection.item() == pytest.approx(expected_detection.item())
    assert loss.detection_bce.item() == pytest.approx(
        expected_detection.item()
    )
    assert loss.positive_continuous_huber.item() == pytest.approx(
        expected_huber.item()
    )
    assert loss.total.item() == pytest.approx(
        0.5 * (expected_detection.item() + expected_huber.item())
    )
    assert (loss.n_masked, loss.n_zero, loss.n_positive) == (8, 5, 3)


def test_loss_is_differentiable_with_finite_gradients() -> None:
    raw_target = _STATE_COUNTS.clone()
    prediction = torch.zeros(
        (*raw_target.shape, 2),
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = hurdle_continuous_loss(
        prediction,
        raw_target,
        torch.ones_like(raw_target, dtype=torch.bool),
        expression_mean=torch.zeros(1),
        expression_scale=torch.ones(1),
    )
    loss.total.backward()

    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool((prediction.grad != 0).any())


@pytest.mark.parametrize(
    ("raw_target", "message"),
    [
        (torch.zeros(8, 1), "zero and positive strata"),
        (torch.ones(8, 1), "zero and positive strata"),
    ],
)
def test_loss_fails_when_detection_stratum_is_absent(
    raw_target: torch.Tensor,
    message: str,
) -> None:
    prediction = torch.zeros((*raw_target.shape, 2))
    with pytest.raises(ValueError, match=message):
        hurdle_continuous_loss(
            prediction,
            raw_target,
            torch.ones_like(raw_target, dtype=torch.bool),
            expression_mean=torch.zeros(1),
            expression_scale=torch.ones(1),
        )


def test_half_up_rounding_and_fixed_count_state_boundaries() -> None:
    values = torch.tensor(
        [0.0, 0.49, 0.5, 1.49, 1.5, 2.5, 7.5],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        round_nonnegative_half_up(values),
        torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 8.0]),
        rtol=0,
        atol=0,
    )

    raw_counts = torch.tensor(
        [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32],
        dtype=torch.float32,
    ).reshape(1, -1)
    reconstructed = reconstruct_positive_counts(
        torch.log1p(raw_counts),
        expression_mean=torch.zeros(raw_counts.shape[1]),
        expression_scale=torch.ones(raw_counts.shape[1]),
    )
    torch.testing.assert_close(reconstructed, raw_counts, rtol=0, atol=0)
    torch.testing.assert_close(
        tokenize_raw_count_tensor(reconstructed),
        torch.tensor([[0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7]]),
        rtol=0,
        atol=0,
    )


def test_positive_metrics_ignore_detection_but_full_state_applies_it() -> None:
    raw_target = _STATE_COUNTS.clone()
    prediction = _perfect_prediction(raw_target, detection_logit=-20.0)
    mask = torch.ones_like(raw_target, dtype=torch.bool)

    result = evaluate_hurdle_continuous_output(
        prediction,
        raw_target,
        mask,
        expression_mean=torch.zeros(1),
        expression_scale=torch.ones(1),
    )

    assert result.metrics[
        "positive_count_state_exact_accuracy"
    ] == pytest.approx(1.0)
    assert result.metrics["positive_count_state_mae"] == pytest.approx(0.0)
    assert result.metrics[
        "positive_count_state_within_one_accuracy"
    ] == pytest.approx(1.0)
    assert result.metrics["state8_exact_accuracy"] == pytest.approx(1 / 8)
    assert result.metrics["detection_sensitivity"] == pytest.approx(0.0)
    assert result.metrics["detection_specificity"] == pytest.approx(1.0)
    assert result.metrics["detection_balanced_accuracy"] == pytest.approx(0.5)
    assert result.metrics["state8_recall"][0] == pytest.approx(1.0)
    assert result.metrics["state8_recall"][1] == pytest.approx(0.0)


def test_perfect_prediction_has_perfect_fixed_metrics() -> None:
    raw_target = _STATE_COUNTS.clone()
    prediction = _perfect_prediction(raw_target)
    result = evaluate_hurdle_continuous_output(
        prediction,
        raw_target,
        torch.ones_like(raw_target, dtype=torch.bool),
        expression_mean=torch.zeros(1),
        expression_scale=torch.ones(1),
    )

    assert result.metrics["detection_balanced_accuracy"] == pytest.approx(1.0)
    assert result.metrics["positive_count_state_mae"] == pytest.approx(0.0)
    assert result.metrics["positive_continuous_huber"] == pytest.approx(0.0)
    assert result.metrics["positive_continuous_mae"] == pytest.approx(0.0)
    assert result.metrics["state8_exact_accuracy"] == pytest.approx(1.0)
    assert result.metrics["state8_balanced_accuracy"] == pytest.approx(1.0)
    assert result.metrics["reconstructed_count_log1p_mae"] == pytest.approx(
        0.0
    )
    assert result.metrics["state8_support"] == [1] * 8
    assert result.metrics["state8_recall"] == pytest.approx([1.0] * 8)
    torch.testing.assert_close(
        result.reconstructed_count,
        raw_target,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("prediction_shape", ValueError, "shape"),
        ("prediction_nonfinite", FloatingPointError, "non-finite"),
        ("mask_dtype", TypeError, "boolean"),
        ("target_fractional", ValueError, "integer-valued"),
        ("scale_zero", ValueError, "strictly positive"),
        ("wrong_delta", ValueError, "exactly 1.0"),
    ],
)
def test_loss_validation_fails_closed(
    case: str,
    error: type[Exception],
    message: str,
) -> None:
    target = torch.tensor([[0.0], [1.0]])
    prediction = _perfect_prediction(target)
    mask = torch.ones_like(target, dtype=torch.bool)
    mean = torch.zeros(1)
    scale = torch.ones(1)
    delta = 1.0
    if case == "prediction_shape":
        prediction = torch.zeros(2, 1, 3)
    elif case == "prediction_nonfinite":
        prediction[0, 0, 0] = float("nan")
    elif case == "mask_dtype":
        mask = mask.float()
    elif case == "target_fractional":
        target[1, 0] = 1.5
    elif case == "scale_zero":
        scale[0] = 0
    elif case == "wrong_delta":
        delta = 0.5

    with pytest.raises(error, match=message):
        hurdle_continuous_loss(
            prediction,
            target,
            mask,
            expression_mean=mean,
            expression_scale=scale,
            huber_delta=delta,
        )


def test_split_and_decode_enforce_two_channel_contract() -> None:
    target = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    prediction = _perfect_prediction(target)
    detection, continuous = split_hurdle_continuous_prediction(prediction)
    assert detection.shape == target.shape
    assert continuous.shape == target.shape

    decoded = decode_hurdle_continuous_prediction(
        prediction,
        expression_mean=torch.zeros(2),
        expression_scale=torch.ones(2),
    )
    torch.testing.assert_close(
        decoded.reconstructed_count,
        target,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        decoded.count_state,
        tokenize_raw_count_tensor(target),
        rtol=0,
        atol=0,
    )
