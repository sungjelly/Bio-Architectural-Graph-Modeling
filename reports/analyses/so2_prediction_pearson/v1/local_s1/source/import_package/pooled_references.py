"""Equal-core transductive references for pooled hybrid-count models."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

from .hybrid_count_metrics import (
    HybridCountReferences,
    fit_hybrid_count_references,
)


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def fit_equal_core_hybrid_count_references(
    counts_by_alias: Mapping[str, Any],
    *,
    expression_mean: Any,
    expression_scale: Any,
    expected_aliases: tuple[str, ...],
) -> HybridCountReferences:
    """Fit one reference while giving every core exactly equal weight.

    Detection and cumulative-ordinal probabilities are arithmetic means of
    the separately smoothed per-core probabilities.  The continuous reference
    is the median of the per-core positive medians on the common log-count
    scale.  It is then expressed in the shared model standardization.
    """

    if not expected_aliases or len(set(expected_aliases)) != len(
        expected_aliases
    ):
        raise ValueError("expected_aliases must be unique and nonempty")
    observed = set(counts_by_alias)
    expected = set(expected_aliases)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "pooled reference aliases do not match the frozen cohort "
            f"(missing={missing}, extra={extra})"
        )

    mean = np.asarray(expression_mean, dtype=np.float64)
    scale = np.asarray(expression_scale, dtype=np.float64)
    if mean.ndim != 1 or scale.shape != mean.shape:
        raise ValueError("shared expression moments must be aligned vectors")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("shared expression moments must be finite")
    if np.any(scale <= 0):
        raise ValueError("shared expression scale must be positive")

    per_core = [
        fit_hybrid_count_references(
            counts_by_alias[alias],
            expression_mean=mean,
            expression_scale=scale,
        )
        for alias in expected_aliases
    ]
    if any(
        reference.detection_probability.shape != mean.shape
        for reference in per_core
    ):
        raise ValueError("pooled reference gene schemas are not aligned")

    detection = np.mean(
        np.stack(
            [reference.detection_probability for reference in per_core],
            axis=0,
        ),
        axis=0,
        dtype=np.float64,
    )
    ordinal = np.mean(
        np.stack(
            [
                reference.positive_ordinal_probability
                for reference in per_core
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.float64,
    )
    per_core_median_log = np.stack(
        [
            reference.positive_continuous_standardized * scale + mean
            for reference in per_core
        ],
        axis=0,
    )
    median_log = np.median(per_core_median_log, axis=0)
    continuous = (median_log - mean) / scale

    if (
        np.any(detection <= 0)
        or np.any(detection >= 1)
        or np.any(ordinal <= 0)
        or np.any(ordinal >= 1)
    ):
        raise ValueError("smoothed pooled probabilities must lie inside (0, 1)")
    arrays = {
        "detection_probability": detection,
        "positive_ordinal_probability": ordinal,
        "positive_continuous_standardized": continuous,
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("pooled reference contains non-finite values")

    detected_state = detection >= 0.5
    positive_state = 1 + np.sum(ordinal >= 0.5, axis=1)
    count_state = np.where(detected_state, positive_state, 0).astype(
        np.int64, copy=False
    )
    arrays["count_state"] = count_state
    audit = {
        "schema": "hybrid_count_equal_core_pooled_references_v1",
        "fit_scope": "all_nodes_all_ten_cores_transductive",
        "core_weighting": "equal",
        "core_count": len(expected_aliases),
        "aliases": list(expected_aliases),
        "detection_aggregation": "mean_per_core_jeffreys_probability",
        "ordinal_aggregation": "mean_per_core_jeffreys_probability",
        "continuous_aggregation": "median_of_per_core_positive_medians",
        "shared_standardization": True,
        "reference_sha256": _array_digest(arrays),
    }
    return HybridCountReferences(
        detection_probability=detection,
        positive_ordinal_probability=ordinal,
        positive_continuous_standardized=continuous,
        detected_state=detected_state,
        positive_state=positive_state.astype(np.int64, copy=False),
        count_state=count_state,
        audit=audit,
    )


__all__ = ["fit_equal_core_hybrid_count_references"]
