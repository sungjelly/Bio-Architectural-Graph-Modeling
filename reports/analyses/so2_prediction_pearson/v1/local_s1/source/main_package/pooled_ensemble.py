"""Prediction-level ensembling for pooled hybrid-count models.

The frozen pooled campaign combines detection and cumulative-ordinal heads in
probability space, while the continuous standardized-log-count head is
combined in its native space.  The resulting probabilities are converted back
to finite logits so the existing hybrid loss and metric APIs can recompute all
nonlinear quantities from the ensemble prediction.

This module is deliberately evaluation-only.  Inputs are detached while they
are accumulated, and no member predictions or member-level metrics are
retained.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import torch
from torch import Tensor

from .hybrid_count import (
    NUM_OUTPUT_CHANNELS,
    NUM_POSITIVE_ORDINAL_THRESHOLDS,
    split_hybrid_prediction,
)


def _validate_expected_member_count(expected_member_count: int) -> int:
    if isinstance(expected_member_count, bool) or not isinstance(
        expected_member_count, Integral
    ):
        raise TypeError("expected_member_count must be a positive integer")
    count = int(expected_member_count)
    if count <= 0:
        raise ValueError("expected_member_count must be a positive integer")
    return count


def _validate_member_prediction(prediction: Tensor) -> None:
    if not isinstance(prediction, Tensor):
        raise TypeError("each ensemble member prediction must be a tensor")
    if not prediction.is_floating_point():
        raise TypeError(
            "each ensemble member prediction must be a real floating tensor"
        )
    if (
        prediction.ndim != 3
        or prediction.shape[0] == 0
        or prediction.shape[1] == 0
        or prediction.shape[-1] != NUM_OUTPUT_CHANNELS
    ):
        raise ValueError(
            "each ensemble member prediction must have nonempty shape "
            f"[nodes, genes, {NUM_OUTPUT_CHANNELS}]"
        )
    _, ordinal_logits, _ = split_hybrid_prediction(prediction)
    if ordinal_logits.shape[-1] != NUM_POSITIVE_ORDINAL_THRESHOLDS:
        raise ValueError(
            "each ensemble member must contain exactly six cumulative "
            "ordinal slots"
        )
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError(
            "ensemble member prediction contains a non-finite value"
        )


def _finite_probability_logits(probability: Tensor) -> Tensor:
    """Convert a validated probability tensor to finite float32 logits."""

    if probability.dtype != torch.float32:
        raise TypeError("ensemble probabilities must use float32 accumulation")
    if not bool(torch.isfinite(probability).all()):
        raise FloatingPointError("ensemble probability is non-finite")
    if bool(((probability < 0.0) | (probability > 1.0)).any()):
        raise ValueError("ensemble probability lies outside [0, 1]")

    # A finite input logit can round to exactly zero or one after sigmoid.
    # Replace only those endpoints with the adjacent representable float32
    # values.  ``finfo.eps`` is the spacing below one, not a valid lower-tail
    # bound: using it as both bounds would inflate every finite probability
    # below about 1.19e-7 and cap rare-event BCE near 15.94.
    zero = torch.zeros((), dtype=probability.dtype, device=probability.device)
    one = torch.ones((), dtype=probability.dtype, device=probability.device)
    lower = torch.nextafter(zero, one)
    upper = torch.nextafter(one, zero)
    bounded = torch.maximum(torch.minimum(probability, upper), lower)
    return torch.logit(bounded)


class HybridCountEnsembleAccumulator:
    """Stream one core/mask's member predictions into a probability ensemble.

    Only three running sums are retained: detection probabilities, six
    threshold probabilities, and continuous standardized predictions.  The
    first member fixes the node/gene shape.  ``finalize`` fails unless exactly
    ``expected_member_count`` members were added.
    """

    def __init__(
        self,
        expected_member_count: int,
        *,
        accumulation_device: torch.device | str | None = None,
    ) -> None:
        self.expected_member_count = _validate_expected_member_count(
            expected_member_count
        )
        self._requested_device = (
            None
            if accumulation_device is None
            else torch.device(accumulation_device)
        )
        self._device: torch.device | None = None
        self._prediction_shape: tuple[int, int, int] | None = None
        self._member_count = 0
        self._detection_probability_sum: Tensor | None = None
        self._ordinal_probability_sum: Tensor | None = None
        self._continuous_sum: Tensor | None = None

    @property
    def member_count(self) -> int:
        """Number of member predictions accumulated so far."""

        return self._member_count

    @property
    def prediction_shape(self) -> tuple[int, int, int] | None:
        """Common member shape, or ``None`` before the first update."""

        return self._prediction_shape

    def update(self, prediction: Tensor) -> None:
        """Add one detached member prediction to the running ensemble."""

        if self._member_count >= self.expected_member_count:
            raise ValueError(
                "received more ensemble members than expected: "
                f"{self.expected_member_count}"
            )
        _validate_member_prediction(prediction)
        shape = tuple(int(value) for value in prediction.shape)
        if self._prediction_shape is not None and shape != self._prediction_shape:
            raise ValueError(
                "ensemble member shape mismatch: expected "
                f"{self._prediction_shape}, got {shape}"
            )

        if self._prediction_shape is None:
            self._prediction_shape = shape
            self._device = self._requested_device or prediction.device
        assert self._device is not None

        detached = prediction.detach().to(
            device=self._device,
            dtype=torch.float32,
        )
        detection_logits, ordinal_logits, continuous = split_hybrid_prediction(
            detached
        )
        detection_probability = torch.sigmoid(detection_logits)
        ordinal_probability = torch.sigmoid(ordinal_logits)

        if self._member_count == 0:
            self._detection_probability_sum = detection_probability.clone()
            self._ordinal_probability_sum = ordinal_probability.clone()
            self._continuous_sum = continuous.clone()
        else:
            assert self._detection_probability_sum is not None
            assert self._ordinal_probability_sum is not None
            assert self._continuous_sum is not None
            self._detection_probability_sum.add_(detection_probability)
            self._ordinal_probability_sum.add_(ordinal_probability)
            self._continuous_sum.add_(continuous)
        self._member_count += 1

    def finalize(self) -> Tensor:
        """Return one hybrid tensor after exact-count validation.

        Detection and ordinal channels are finite logits corresponding to the
        arithmetic mean probabilities.  Channel seven is the arithmetic mean
        continuous standardized prediction.  Decoding and every nonlinear
        metric must be performed on this returned tensor.
        """

        if self._member_count != self.expected_member_count:
            raise ValueError(
                "ensemble member count mismatch: expected "
                f"{self.expected_member_count}, received {self._member_count}"
            )
        assert self._prediction_shape is not None
        assert self._device is not None
        assert self._detection_probability_sum is not None
        assert self._ordinal_probability_sum is not None
        assert self._continuous_sum is not None

        divisor = float(self.expected_member_count)
        detection_probability = self._detection_probability_sum / divisor
        ordinal_probability = self._ordinal_probability_sum / divisor
        continuous = self._continuous_sum / divisor
        if not bool(torch.isfinite(continuous).all()):
            raise FloatingPointError(
                "ensemble continuous prediction is non-finite"
            )

        ensemble = torch.empty(
            self._prediction_shape,
            dtype=torch.float32,
            device=self._device,
        )
        ensemble[..., 0] = _finite_probability_logits(
            detection_probability
        )
        ensemble[..., 1:7] = _finite_probability_logits(ordinal_probability)
        ensemble[..., 7] = continuous
        if not bool(torch.isfinite(ensemble).all()):
            raise FloatingPointError("final ensemble prediction is non-finite")
        return ensemble


def ensemble_hybrid_count_predictions(
    member_predictions: Iterable[Tensor],
    *,
    expected_member_count: int,
    accumulation_device: torch.device | str | None = None,
) -> Tensor:
    """Combine a member iterable without stacking or retaining every member.

    Iteration order is preserved, making the floating-point accumulation
    deterministic for a fixed ordered iterable.  The caller must supply the
    frozen expected member count; missing or extra members fail closed.
    """

    accumulator = HybridCountEnsembleAccumulator(
        expected_member_count,
        accumulation_device=accumulation_device,
    )
    for prediction in member_predictions:
        accumulator.update(prediction)
    return accumulator.finalize()


__all__ = [
    "HybridCountEnsembleAccumulator",
    "ensemble_hybrid_count_predictions",
]
