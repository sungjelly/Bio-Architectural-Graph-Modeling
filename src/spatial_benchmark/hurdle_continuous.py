"""Detection plus positive-continuous hurdle objective and decoding.

The prediction contract is exactly ``[nodes, genes, 2]``:

1. a detection logit; and
2. a standardized positive ``log1p(count)`` estimate.

The loss and decoder intentionally contain no fitted thresholds.  Positive
counts are inverted through the recorded per-gene standardization, clipped at
zero, and rounded with an explicit nonnegative half-up rule before assignment
to the fixed hybrid-count states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .hybrid_count import (
    NUM_COUNT_STATES,
    standardized_log1p_counts,
    tokenize_raw_count_tensor,
)


NUM_HURDLE_CONTINUOUS_CHANNELS = 2


@dataclass(frozen=True)
class HurdleContinuousLoss:
    """Differentiable equal-weight hurdle objective and support counts."""

    total: Tensor
    detection: Tensor
    positive_continuous_huber: Tensor
    n_masked: int
    n_zero: int
    n_positive: int

    @property
    def detection_bce(self) -> Tensor:
        """Alias with the metric's explicit name."""

        return self.detection


@dataclass(frozen=True)
class HurdleContinuousDecoded:
    """Fixed decoded outputs before evaluation masking."""

    detected: Tensor
    positive_reconstructed_count: Tensor
    reconstructed_count: Tensor
    positive_count_state: Tensor
    count_state: Tensor


@dataclass(frozen=True)
class HurdleContinuousEvaluation:
    """Decoded CPU tensors and metrics for one fixed evaluation mask."""

    count_state: Tensor
    positive_count_state: Tensor
    detected: Tensor
    positive_continuous_standardized: Tensor
    positive_reconstructed_count: Tensor
    reconstructed_count: Tensor
    metrics: Mapping[str, Any]


def _validate_prediction(prediction: Tensor) -> Tensor:
    if not isinstance(prediction, Tensor):
        raise TypeError("prediction must be a tensor")
    if (
        prediction.ndim != 3
        or prediction.shape[-1] != NUM_HURDLE_CONTINUOUS_CHANNELS
    ):
        raise ValueError("prediction must have shape [nodes, genes, 2]")
    if prediction.shape[0] == 0 or prediction.shape[1] == 0:
        raise ValueError("prediction node and gene dimensions must be nonempty")
    if prediction.dtype == torch.bool or prediction.is_complex():
        raise TypeError("prediction must contain real floating values")
    if not prediction.is_floating_point():
        raise TypeError("prediction must contain real floating values")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("prediction contains non-finite values")
    return prediction


def split_hurdle_continuous_prediction(
    prediction: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return detection logits and standardized positive estimates."""

    validated = _validate_prediction(prediction)
    return validated[..., 0], validated[..., 1]


def _validated_standardization(
    expression_mean: Any,
    expression_scale: Any,
    *,
    num_genes: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    mean = torch.as_tensor(expression_mean, device=device)
    scale = torch.as_tensor(expression_scale, device=device)
    if mean.dtype == torch.bool or scale.dtype == torch.bool:
        raise TypeError("expression standardization must be real numeric")
    if mean.is_complex() or scale.is_complex():
        raise TypeError("expression standardization must be real numeric")
    if mean.shape != (num_genes,) or scale.shape != (num_genes,):
        raise ValueError("expression standardization must have shape [genes]")
    mean = mean.float()
    scale = scale.float()
    if not bool(torch.isfinite(mean).all()) or not bool(
        torch.isfinite(scale).all()
    ):
        raise ValueError("expression standardization must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("expression_scale must be strictly positive")
    return mean, scale


def _validated_target_and_mask(
    raw_target: Tensor,
    target_mask: Tensor,
    *,
    prediction: Tensor,
) -> Tensor:
    if not isinstance(raw_target, Tensor):
        raise TypeError("raw_target must be a tensor")
    # Reuse the fixed raw-count validator and state boundaries.
    tokenize_raw_count_tensor(raw_target)
    if not isinstance(target_mask, Tensor):
        raise TypeError("target_mask must be a tensor")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must be boolean")
    if target_mask.shape != raw_target.shape:
        raise ValueError("target_mask must match raw_target")
    if (
        prediction.device != raw_target.device
        or target_mask.device != raw_target.device
    ):
        raise ValueError("prediction, raw_target, and target_mask must share a device")
    expected = (*raw_target.shape, NUM_HURDLE_CONTINUOUS_CHANNELS)
    if tuple(prediction.shape) != expected:
        raise ValueError(
            f"prediction shape mismatch: expected {expected}, got "
            f"{tuple(prediction.shape)}"
        )
    if not bool(target_mask.any()):
        raise ValueError("hurdle loss requires at least one masked target")
    return raw_target


def hurdle_continuous_loss(
    prediction: Tensor,
    raw_target: Tensor,
    target_mask: Tensor,
    *,
    expression_mean: Any,
    expression_scale: Any,
    huber_delta: float = 1.0,
) -> HurdleContinuousLoss:
    """Compute balanced detection BCE plus positive-only Huber.

    The two components receive equal weight.  Both zero and positive masked
    detection strata are mandatory; an absent stratum is an error rather than
    a zero-valued contribution.
    """

    if huber_delta != 1.0:
        raise ValueError("the frozen positive Huber delta is exactly 1.0")
    validated_prediction = _validate_prediction(prediction)
    target = _validated_target_and_mask(
        raw_target,
        target_mask,
        prediction=validated_prediction,
    )
    mean, scale = _validated_standardization(
        expression_mean,
        expression_scale,
        num_genes=int(target.shape[1]),
        device=target.device,
    )
    detection_logits, continuous_prediction = (
        split_hurdle_continuous_prediction(validated_prediction)
    )
    selected_counts = target[target_mask]
    selected_detection = detection_logits[target_mask].float()
    zero = selected_counts == 0
    positive = selected_counts > 0
    if not bool(zero.any()) or not bool(positive.any()):
        raise ValueError(
            "balanced detection loss requires zero and positive strata"
        )

    detection_loss = 0.5 * (
        F.binary_cross_entropy_with_logits(
            selected_detection[zero],
            torch.zeros_like(selected_detection[zero]),
        )
        + F.binary_cross_entropy_with_logits(
            selected_detection[positive],
            torch.ones_like(selected_detection[positive]),
        )
    )
    standardized_target = standardized_log1p_counts(
        target,
        mean,
        scale,
    )[target_mask]
    selected_continuous = continuous_prediction[target_mask].float()
    positive_continuous_huber = F.huber_loss(
        selected_continuous[positive],
        standardized_target[positive].float(),
        reduction="mean",
        delta=1.0,
    )
    total = 0.5 * (detection_loss + positive_continuous_huber)
    for name, value in (
        ("detection loss", detection_loss),
        ("positive continuous Huber", positive_continuous_huber),
        ("total hurdle loss", total),
    ):
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"{name} is non-finite")
    return HurdleContinuousLoss(
        total=total,
        detection=detection_loss,
        positive_continuous_huber=positive_continuous_huber,
        n_masked=int(selected_counts.numel()),
        n_zero=int(zero.sum().detach().cpu()),
        n_positive=int(positive.sum().detach().cpu()),
    )


def round_nonnegative_half_up(value: Tensor) -> Tensor:
    """Round finite nonnegative values with the fixed ``floor(x + 0.5)`` rule."""

    if not isinstance(value, Tensor):
        raise TypeError("value must be a tensor")
    if value.numel() == 0:
        raise ValueError("value must be nonempty")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError("value must contain real numeric values")
    numeric = value if value.is_floating_point() else value.float()
    if not bool(torch.isfinite(numeric).all()):
        raise FloatingPointError("value contains non-finite values")
    if bool((numeric < 0).any()):
        raise ValueError("value must be nonnegative")
    rounded = torch.floor(numeric + 0.5)
    if not bool(torch.isfinite(rounded).all()):
        raise FloatingPointError("rounded value is non-finite")
    return rounded


def reconstruct_positive_counts(
    positive_continuous_standardized: Tensor,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> Tensor:
    """Invert standardization and return fixed half-up rounded counts."""

    if not isinstance(positive_continuous_standardized, Tensor):
        raise TypeError("positive_continuous_standardized must be a tensor")
    value = positive_continuous_standardized
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(
            "positive_continuous_standardized must have shape [nodes, genes]"
        )
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError(
            "positive_continuous_standardized must contain real floating values"
        )
    if not value.is_floating_point():
        raise TypeError(
            "positive_continuous_standardized must contain real floating values"
        )
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(
            "positive_continuous_standardized contains non-finite values"
        )
    mean, scale = _validated_standardization(
        expression_mean,
        expression_scale,
        num_genes=int(value.shape[1]),
        device=value.device,
    )
    predicted_log1p = value.float() * scale.unsqueeze(0) + mean.unsqueeze(0)
    reconstructed = torch.expm1(predicted_log1p)
    if not bool(torch.isfinite(reconstructed).all()):
        raise FloatingPointError("inverse-standardized count is non-finite")
    return round_nonnegative_half_up(reconstructed.clamp_min(0.0))


def decode_hurdle_continuous_prediction(
    prediction: Tensor,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> HurdleContinuousDecoded:
    """Apply the fixed detection decision and continuous count decoder."""

    detection_logits, continuous = split_hurdle_continuous_prediction(
        prediction
    )
    detected = torch.sigmoid(detection_logits.float()) >= 0.5
    positive_count = reconstruct_positive_counts(
        continuous,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
    )
    positive_state = tokenize_raw_count_tensor(positive_count)
    count_state = torch.where(
        detected,
        positive_state,
        torch.zeros_like(positive_state),
    )
    reconstructed_count = torch.where(
        detected,
        positive_count,
        torch.zeros_like(positive_count),
    )
    return HurdleContinuousDecoded(
        detected=detected,
        positive_reconstructed_count=positive_count,
        reconstructed_count=reconstructed_count,
        positive_count_state=positive_state,
        count_state=count_state,
    )


def _binary_metrics(target: Tensor, prediction: Tensor) -> dict[str, Any]:
    if (
        target.dtype != torch.bool
        or prediction.dtype != torch.bool
        or target.shape != prediction.shape
        or target.ndim != 1
        or target.numel() == 0
    ):
        raise ValueError("detection metrics require aligned boolean vectors")
    positive = int(target.sum().detach().cpu())
    negative = int(target.numel() - positive)
    if positive == 0 or negative == 0:
        raise ValueError("detection metrics require zero and positive targets")
    true_positive = int((target & prediction).sum().detach().cpu())
    true_negative = int((~target & ~prediction).sum().detach().cpu())
    predicted_positive = int(prediction.sum().detach().cpu())
    sensitivity = true_positive / positive
    specificity = true_negative / negative
    return {
        "detection_balanced_accuracy": 0.5
        * (sensitivity + specificity),
        "detection_sensitivity": sensitivity,
        "detection_specificity": specificity,
        "detection_precision": (
            None
            if predicted_positive == 0
            else true_positive / predicted_positive
        ),
        "detection_positive_support": positive,
        "detection_zero_support": negative,
        "detection_predicted_positive": predicted_positive,
    }


def _state_metrics(
    target: Tensor,
    prediction: Tensor,
) -> dict[str, Any]:
    if (
        target.shape != prediction.shape
        or target.ndim != 1
        or target.numel() == 0
    ):
        raise ValueError("state metrics require aligned nonempty vectors")
    if bool((target < 0).any()) or bool((target >= NUM_COUNT_STATES).any()):
        raise ValueError("target count state is outside the fixed vocabulary")
    if bool((prediction < 0).any()) or bool(
        (prediction >= NUM_COUNT_STATES).any()
    ):
        raise ValueError(
            "predicted count state is outside the fixed vocabulary"
        )
    support: list[int] = []
    recall: list[float | None] = []
    for state in range(NUM_COUNT_STATES):
        selected = target == state
        state_support = int(selected.sum().detach().cpu())
        support.append(state_support)
        recall.append(
            None
            if state_support == 0
            else float(
                (prediction[selected] == state)
                .float()
                .mean()
                .detach()
                .cpu()
            )
        )
    finite_recall = [value for value in recall if value is not None]
    if not finite_recall:
        raise ValueError("state metrics have no supported class")
    return {
        "state8_exact_accuracy": float(
            (target == prediction).float().mean().detach().cpu()
        ),
        "state8_balanced_accuracy": float(
            sum(finite_recall) / len(finite_recall)
        ),
        "state8_support": support,
        "state8_recall": recall,
    }


def evaluate_hurdle_continuous_output(
    prediction: Tensor,
    raw_target: Tensor,
    target_mask: Tensor,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> HurdleContinuousEvaluation:
    """Evaluate one hurdle-continuous output on one immutable mask."""

    loss = hurdle_continuous_loss(
        prediction,
        raw_target,
        target_mask,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
        huber_delta=1.0,
    )
    mean, scale = _validated_standardization(
        expression_mean,
        expression_scale,
        num_genes=int(raw_target.shape[1]),
        device=raw_target.device,
    )
    detection_logits, continuous = split_hurdle_continuous_prediction(
        prediction
    )
    decoded = decode_hurdle_continuous_prediction(
        prediction,
        expression_mean=mean,
        expression_scale=scale,
    )
    target_state = tokenize_raw_count_tensor(raw_target)
    selected_target_count = raw_target[target_mask]
    selected_target_state = target_state[target_mask]
    selected_target_detected = selected_target_count > 0
    selected_detected = decoded.detected[target_mask]
    selected_count_state = decoded.count_state[target_mask]

    metrics: dict[str, Any] = {
        "n_masked": loss.n_masked,
        "n_zero": loss.n_zero,
        "n_positive": loss.n_positive,
        "hurdle_loss": float(loss.total.detach().float().cpu()),
        "detection_bce": float(loss.detection.detach().float().cpu()),
        "positive_continuous_huber": float(
            loss.positive_continuous_huber.detach().float().cpu()
        ),
    }
    metrics.update(
        _binary_metrics(selected_target_detected, selected_detected)
    )
    metrics.update(
        _state_metrics(selected_target_state, selected_count_state)
    )

    positive = selected_target_detected
    positive_target_state = selected_target_state[positive]
    positive_prediction_state = decoded.positive_count_state[target_mask][
        positive
    ]
    positive_state_error = (
        positive_prediction_state - positive_target_state
    ).abs()
    metrics.update(
        {
            "positive_count_state_exact_accuracy": float(
                (positive_state_error == 0).float().mean().detach().cpu()
            ),
            "positive_count_state_mae": float(
                positive_state_error.float().mean().detach().cpu()
            ),
            "positive_count_state_within_one_accuracy": float(
                (positive_state_error <= 1).float().mean().detach().cpu()
            ),
        }
    )

    standardized_target = standardized_log1p_counts(
        raw_target,
        mean,
        scale,
    )
    continuous_error = (
        continuous[target_mask].float()[positive]
        - standardized_target[target_mask].float()[positive]
    )
    metrics["positive_continuous_mae"] = float(
        continuous_error.abs().mean().detach().cpu()
    )
    reconstructed_log1p_error = (
        torch.log1p(decoded.reconstructed_count[target_mask].float())
        - torch.log1p(selected_target_count.float())
    )
    metrics["reconstructed_count_log1p_mae"] = float(
        reconstructed_log1p_error.abs().mean().detach().cpu()
    )
    for name, value in metrics.items():
        if isinstance(value, float) and not torch.isfinite(
            torch.tensor(value)
        ):
            raise FloatingPointError(f"metric {name} is non-finite")

    return HurdleContinuousEvaluation(
        count_state=decoded.count_state.detach().cpu(),
        positive_count_state=decoded.positive_count_state.detach().cpu(),
        detected=decoded.detected.detach().cpu(),
        positive_continuous_standardized=continuous.detach().float().cpu(),
        positive_reconstructed_count=decoded.positive_reconstructed_count.detach()
        .float()
        .cpu(),
        reconstructed_count=decoded.reconstructed_count.detach().float().cpu(),
        metrics=metrics,
    )


__all__ = [
    "HurdleContinuousDecoded",
    "HurdleContinuousEvaluation",
    "HurdleContinuousLoss",
    "NUM_HURDLE_CONTINUOUS_CHANNELS",
    "decode_hurdle_continuous_prediction",
    "evaluate_hurdle_continuous_output",
    "hurdle_continuous_loss",
    "reconstruct_positive_counts",
    "round_nonnegative_half_up",
    "split_hurdle_continuous_prediction",
]
