"""Additive reconstruction metrics without retaining cell-level predictions.

Point metrics consistently use the marginal mean, except the explicitly named
``positive_conditional_*`` metrics.  Detection AP/AUROC use a fixed probability
histogram (4096 bins by default), so they are approximations with ties inside
each bin.  Each reported group owns an accumulator; ``merge`` permits exact
support-weighted pooling instead of averaging per-core means.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from .hurdle_negative_binomial import masked_hurdle_negative_binomial_nll_sums
from .local_gene_residual import LocalGeneResidualOutput


def _sum(value: Tensor) -> float:
    return float(value.sum(dtype=torch.float64).item())


class LocalGeneCountMetrics:
    """Accumulate only sufficient statistics for count and detection metrics.

    Zero-support metrics return 0.0; the associated support and ``*_defined``
    flags expose that convention.  An entirely empty accumulator cannot be
    finalized.  Updates detach model output and never retain autograd graphs.
    """

    def __init__(self, num_bins: int = 4096) -> None:
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer >= 2")
        self.num_bins = num_bins
        self.masked_entries = 0
        self.positive_entries = 0
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0
        self.positive_histogram = np.zeros(num_bins, dtype=np.int64)
        self.negative_histogram = np.zeros(num_bins, dtype=np.int64)
        self.sums = dict.fromkeys(
            (
                "joint_nll", "detection_nll", "positive_count_nll", "brier",
                "raw_absolute", "raw_squared", "log_absolute", "log_squared",
                "zero_raw_absolute", "zero_raw_squared",
                "positive_marginal_raw_absolute", "positive_marginal_raw_squared",
                "positive_conditional_raw_absolute", "positive_conditional_raw_squared",
                "observed_raw", "predicted_raw", "probability",
            ),
            0.0,
        )

    @torch.no_grad()
    def update(
        self,
        output: LocalGeneResidualOutput,
        raw_target: Tensor,
        mask: Tensor,
    ) -> None:
        """Score exactly the supplied mask; an empty subgroup is a no-op."""

        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean tensor")
        if mask.shape != raw_target.shape or mask.shape != output.nb_mean.shape:
            raise ValueError("mask, raw_target, and output must be shape-aligned")
        if not bool(mask.any()):
            return
        likelihood = masked_hurdle_negative_binomial_nll_sums(
            output.nonzero_logits, output.nb_mean, output.theta, raw_target, mask
        )
        target = raw_target[mask].float()
        prediction = output.marginal_mean[mask].float()
        positive_prediction = output.positive_conditional_mean[mask].float()
        probability = torch.sigmoid(output.nonzero_logits[mask].float())
        if not bool(torch.isfinite(prediction).all()) or bool((prediction < 0).any()):
            raise ValueError("marginal predictions must be finite and nonnegative")
        if not bool(torch.isfinite(positive_prediction).all()) or bool((positive_prediction < 1.0 - 1e-5).any()):
            raise ValueError("conditional positive means must be finite and >= 1")
        positive = target > 0
        zero = ~positive
        predicted_positive = probability >= 0.5
        self.masked_entries += likelihood.masked_entries
        self.positive_entries += likelihood.positive_entries
        self.true_positive += int((predicted_positive & positive).sum().item())
        self.false_positive += int((predicted_positive & zero).sum().item())
        self.false_negative += int((~predicted_positive & positive).sum().item())
        self.true_negative += int((~predicted_positive & zero).sum().item())
        error = prediction - target
        log_error = torch.log1p(prediction) - torch.log1p(target)
        positive_error = positive_prediction[positive] - target[positive]
        additions = {
            "joint_nll": float(likelihood.joint_nll_sum.item()),
            "detection_nll": float(likelihood.detection_nll_sum.item()),
            "positive_count_nll": float(likelihood.positive_count_nll_sum.item()),
            "brier": _sum((probability - positive.float()).square()),
            "raw_absolute": _sum(error.abs()),
            "raw_squared": _sum(error.square()),
            "log_absolute": _sum(log_error.abs()),
            "log_squared": _sum(log_error.square()),
            "zero_raw_absolute": _sum(error[zero].abs()),
            "zero_raw_squared": _sum(error[zero].square()),
            "positive_marginal_raw_absolute": _sum(error[positive].abs()),
            "positive_marginal_raw_squared": _sum(error[positive].square()),
            "positive_conditional_raw_absolute": _sum(positive_error.abs()),
            "positive_conditional_raw_squared": _sum(positive_error.square()),
            "observed_raw": _sum(target),
            "predicted_raw": _sum(prediction),
            "probability": _sum(probability),
        }
        if not all(math.isfinite(value) for value in additions.values()):
            raise FloatingPointError("count metric sufficient statistics are non-finite")
        for key, value in additions.items():
            self.sums[key] += value
        bin_index = (probability * self.num_bins).long().clamp_(0, self.num_bins - 1)
        self.positive_histogram += torch.bincount(
            bin_index[positive], minlength=self.num_bins
        ).cpu().numpy()
        self.negative_histogram += torch.bincount(
            bin_index[zero], minlength=self.num_bins
        ).cpu().numpy()

    def merge(self, other: LocalGeneCountMetrics) -> None:
        """Pool a disjoint shard/core/group with identical histogram resolution."""

        if not isinstance(other, LocalGeneCountMetrics):
            raise TypeError("other must be a LocalGeneCountMetrics accumulator")
        if other.num_bins != self.num_bins:
            raise ValueError("histogram bin counts must agree when merging")
        for name in (
            "masked_entries", "positive_entries", "true_positive", "false_positive",
            "false_negative", "true_negative",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for key in self.sums:
            self.sums[key] += other.sums[key]
        self.positive_histogram += other.positive_histogram
        self.negative_histogram += other.negative_histogram

    def finalize(self) -> dict[str, float | int]:
        """Return JSON-safe finite scalar metrics and their relevant supports."""

        total = self.masked_entries
        positive = self.positive_entries
        zero = total - positive
        if total == 0:
            raise ValueError("cannot finalize an empty metric accumulator")

        def divide(numerator: float, denominator: int) -> float:
            return numerator / denominator if denominator else 0.0

        def mean(key: str, support: int = total) -> float:
            return divide(self.sums[key], support)

        # Descending cumulative precision handles all members of a bin as ties.
        descending_positive = self.positive_histogram[::-1].astype(np.float64)
        descending_negative = self.negative_histogram[::-1].astype(np.float64)
        cumulative_positive = np.cumsum(descending_positive)
        cumulative_total = cumulative_positive + np.cumsum(descending_negative)
        precision = np.divide(
            cumulative_positive, cumulative_total,
            out=np.zeros_like(cumulative_total), where=cumulative_total > 0,
        )
        ap = float(np.dot(descending_positive, precision) / positive) if positive else 0.0
        ascending_negative = self.negative_histogram.astype(np.float64)
        lower_negative = np.cumsum(ascending_negative) - ascending_negative
        auc = (
            float(np.dot(self.positive_histogram, lower_negative + 0.5 * ascending_negative) / (positive * zero))
            if positive and zero else 0.0
        )
        true_positive = self.true_positive
        predicted_positive = true_positive + self.false_positive
        result: dict[str, float | int] = {
            "masked_entries": total,
            "positive_entries": positive,
            "zero_entries": zero,
            "count_nll": mean("joint_nll"),
            "detection_bce": mean("detection_nll"),
            "positive_conditional_count_nll": mean("positive_count_nll", positive),
            "raw_count_mae": mean("raw_absolute"),
            "raw_count_rmse": math.sqrt(mean("raw_squared")),
            "log1p_count_mae": mean("log_absolute"),
            "log1p_count_rmse": math.sqrt(mean("log_squared")),
            "zero_raw_count_mae": mean("zero_raw_absolute", zero),
            "zero_raw_count_rmse": math.sqrt(mean("zero_raw_squared", zero)),
            "positive_marginal_raw_count_mae": mean("positive_marginal_raw_absolute", positive),
            "positive_marginal_raw_count_rmse": math.sqrt(mean("positive_marginal_raw_squared", positive)),
            "positive_conditional_raw_count_mae": mean("positive_conditional_raw_absolute", positive),
            "positive_conditional_raw_count_rmse": math.sqrt(mean("positive_conditional_raw_squared", positive)),
            "observed_raw_count_mean": mean("observed_raw"),
            "predicted_raw_count_mean": mean("predicted_raw"),
            "observed_count_mean": mean("observed_raw"),
            "predicted_count_mean": mean("predicted_raw"),
            "observed_positive_fraction": positive / total,
            "predicted_positive_fraction": predicted_positive / total,
            "mean_nonzero_probability": mean("probability"),
            "detection_brier": mean("brier"),
            "detection_average_precision": ap,
            "detection_auroc": auc,
            "detection_accuracy": (true_positive + self.true_negative) / total,
            "detection_precision": divide(true_positive, predicted_positive),
            "detection_recall": divide(true_positive, positive),
            "detection_f1": divide(2 * true_positive, 2 * true_positive + self.false_positive + self.false_negative),
            "true_positive": true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
            "positive_metrics_defined": int(positive > 0),
            "zero_metrics_defined": int(zero > 0),
            "detection_precision_defined": int(predicted_positive > 0),
            "detection_average_precision_defined": int(positive > 0),
            "detection_auroc_defined": int(positive > 0 and zero > 0),
            "detection_histogram_bins": self.num_bins,
        }
        if not all(math.isfinite(value) for value in result.values()):
            raise FloatingPointError("final count metrics are non-finite")
        return result

    compute = finalize


CountMetricAccumulator = LocalGeneCountMetrics
