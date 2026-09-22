"""Streaming descriptive accuracy on observed-count strata of fixed masks.

Callers select aligned masked entries before updating. Positive membership is
defined by observed raw count, never standardized expression. Sums use float64;
no prediction or target arrays are retained. These are descriptive entry-level
metrics, not an estimate of replication or held-out predictive gain.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


STRATA = (
    "all", "zero", "positive", "count_1", "count_2", "count_3",
    "count_4_7", "count_8_plus",
)
_COUNT_BINS = STRATA[3:]
_METRICS = (
    "mse_standardized", "mae_standardized", "huber_standardized",
    "bias_standardized", "mse_log1p", "mae_log1p",
)


@dataclass
class _Sums:
    n: int
    n_exact_count: int
    values: np.ndarray


class NonzeroMetricAccumulator:
    """Accumulate fixed-mask reconstruction metrics without retaining rows.

    ``update`` takes four same-length 1D arrays. ``pred_log1p`` is the inverse
    gene-standardized prediction, with negative values preserved for errors.
    Only decoded count diagnostics conceptually clip negative counts to zero.
    Decoding uses round-half-up and a fixed continuous count threshold of 0.5.
    ``merge`` pools sufficient statistics (not an equal-core average).
    """

    def __init__(self, *, huber_delta: float = 1.0) -> None:
        self.huber_delta = float(huber_delta)
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        self._strata = {
            name: _Sums(0, 0, np.zeros(len(_METRICS), dtype=np.float64))
            for name in STRATA
        }
        self._true_positive = 0
        self._false_positive = 0
        self._true_negative = 0
        self._false_negative = 0

    def update(
        self,
        *,
        true_counts: Any,
        true_standardized: Any,
        pred_standardized: Any,
        pred_log1p: Any,
    ) -> None:
        """Add already-masked entries; reject invalid arrays without skipping.

        True counts must be finite nonnegative integers. Empty aligned arrays
        are allowed. Nonfinite inputs or accumulated errors fail explicitly.
        """
        arrays = {
            "true_counts": np.asarray(true_counts, dtype=np.float64),
            "true_standardized": np.asarray(true_standardized, dtype=np.float64),
            "pred_standardized": np.asarray(pred_standardized, dtype=np.float64),
            "pred_log1p": np.asarray(pred_log1p, dtype=np.float64),
        }
        counts = arrays["true_counts"]
        for name, value in arrays.items():
            if value.ndim != 1 or value.shape != counts.shape:
                raise ValueError("metric inputs must be aligned one-dimensional arrays")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("true_counts must contain nonnegative integers")
        # Half-integer interval boundaries must remain representable in float64.
        if np.any(counts >= 2**52):
            raise ValueError("true_counts are too large for exact count decoding")
        if not counts.size:
            return

        positive = counts > 0
        selections = {
            "all": np.ones(counts.size, dtype=bool),
            "zero": ~positive,
            "positive": positive,
            "count_1": counts == 1,
            "count_2": counts == 2,
            "count_3": counts == 3,
            "count_4_7": (counts >= 4) & (counts <= 7),
            "count_8_plus": counts >= 8,
        }
        log_prediction = arrays["pred_log1p"]
        # Avoid exponentiating predictions: values far above float64's exp
        # limit can still be scored safely on the modeled log scale.
        lower = np.log1p(np.maximum(counts - 0.5, 0.0))
        upper = np.log1p(counts + 0.5)
        exact = ((~positive) | (log_prediction >= lower)) & (log_prediction < upper)
        detected = log_prediction >= np.log1p(0.5)
        try:
            with np.errstate(over="raise", invalid="raise"):
                error = arrays["pred_standardized"] - arrays["true_standardized"]
                absolute = np.abs(error)
                quadratic = np.minimum(absolute, self.huber_delta)
                huber = 0.5 * quadratic**2 + self.huber_delta * (absolute - quadratic)
                log_error = log_prediction - np.log1p(counts)
                values = (
                    error**2, absolute, huber, error,
                    log_error**2, np.abs(log_error),
                )
                pending = {}
                for name, selected in selections.items():
                    previous = self._strata[name]
                    added = np.asarray([
                        np.sum(value[selected], dtype=np.float64) for value in values
                    ], dtype=np.float64)
                    summed = previous.values + added
                    if not np.isfinite(summed).all():
                        raise ValueError("accumulated metric sums must be finite")
                    pending[name] = _Sums(
                        previous.n + int(np.count_nonzero(selected)),
                        previous.n_exact_count + int(np.count_nonzero(exact & selected)),
                        summed,
                    )
        except FloatingPointError as exc:
            raise ValueError("metric errors or sums exceeded finite float64 range") from exc
        self._strata = pending
        self._true_positive += int(np.count_nonzero(positive & detected))
        self._false_positive += int(np.count_nonzero(~positive & detected))
        self._true_negative += int(np.count_nonzero(~positive & ~detected))
        self._false_negative += int(np.count_nonzero(positive & ~detected))

    def merge(self, other: NonzeroMetricAccumulator) -> None:
        """Pool another accumulator using support-weighted sufficient sums."""
        if not isinstance(other, NonzeroMetricAccumulator):
            raise TypeError("other must be a NonzeroMetricAccumulator")
        if self.huber_delta != other.huber_delta:
            raise ValueError("cannot merge different Huber deltas")
        pending = {}
        for name in STRATA:
            left, right = self._strata[name], other._strata[name]
            with np.errstate(over="ignore"):
                summed = left.values + right.values
            if not np.isfinite(summed).all():
                raise ValueError("merged metric sums must be finite")
            pending[name] = _Sums(
                left.n + right.n, left.n_exact_count + right.n_exact_count, summed,
            )
        self._strata = pending
        self._true_positive += other._true_positive
        self._false_positive += other._false_positive
        self._true_negative += other._true_negative
        self._false_negative += other._false_negative

    def result(self) -> dict[str, Any]:
        """Return JSON-safe metrics and independently auditable decomposition.

        Unsupported metrics are ``None`` with support zero. Positive recall
        measures detection of count > 0; exact_count_accuracy requires the
        decoded count to match the particular observed integer count.
        """
        strata: dict[str, dict[str, Any]] = {}
        for name, values in self._strata.items():
            strata[name] = {
                "n": values.n,
                "n_exact_count": values.n_exact_count,
                "sums": {
                    key: float(value) for key, value in zip(_METRICS, values.values)
                },
                **{
                    key: float(value / values.n) if values.n else None
                    for key, value in zip(_METRICS, values.values)
                },
                "exact_count_accuracy": (
                    values.n_exact_count / values.n if values.n else None
                ),
            }
        total = self._strata["all"]
        positive = self._strata["positive"]
        zero = self._strata["zero"]
        bins = [self._strata[name] for name in _COUNT_BINS]
        if zero.n + positive.n != total.n or sum(item.n for item in bins) != positive.n:
            raise AssertionError("count stratum support partition mismatch")
        partition_sum = zero.values + positive.values
        bin_sum = np.sum([item.values for item in bins], axis=0, dtype=np.float64)
        # Signed bias can cancel; use accumulated absolute error to bound its
        # floating-point summation tolerance while keeping SSE checks strict.
        scale = np.maximum(np.abs(total.values), 1.0)
        scale[3] = max(total.values[1], 1.0)
        tolerance = 1e-11 * scale
        if np.any(np.abs(partition_sum - total.values) > tolerance):
            raise AssertionError("zero/positive metric sums do not reconstruct all entries")
        if np.any(np.abs(bin_sum - positive.values) > tolerance):
            raise AssertionError("count-bin metric sums do not reconstruct positive entries")
        tp, fp = self._true_positive, self._false_positive
        tn, fn = self._true_negative, self._false_negative
        if tp + fn != positive.n or tn + fp != zero.n:
            raise AssertionError("detection support does not match count strata")
        recall = tp / positive.n if positive.n else None
        specificity = tn / zero.n if zero.n else None
        return {
            "schema": "so2_nonzero_metrics_v1",
            "huber_delta": self.huber_delta,
            "strata": strata,
            "detection": {
                "continuous_count_threshold": 0.5,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "true_negative_count": tn,
                "false_negative_count": fn,
                "positive_recall": recall,
                "zero_specificity": specificity,
                "positive_precision": tp / (tp + fp) if tp + fp else None,
                "balanced_accuracy": (
                    (recall + specificity) / 2
                    if recall is not None and specificity is not None else None
                ),
            },
            "decomposition": {
                "support_partition_verified": True,
                "metric_sum_partition_verified": True,
                "all_squared_error_standardized": float(total.values[0]),
                "zero_plus_positive_squared_error_standardized": float(partition_sum[0]),
                "positive_count_bins_squared_error_standardized": float(bin_sum[0]),
                "zero_positive_mse_residual": (
                    float((total.values[0] - partition_sum[0]) / total.n)
                    if total.n else None
                ),
            },
        }
