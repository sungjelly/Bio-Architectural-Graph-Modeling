"""Pure aggregation and frozen decision gates for the pooled campaign."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
ARMS = ("pooled-hybrid-gat-k1000", "pooled-hybrid-matched-self")
SEEDS = tuple(range(7))
MASK_MODES = ("partial_gene", "whole_node", "spatial_block")
REPLICATES = tuple(range(3))
GAT_ARM, SELF_ARM = ARMS


class PooledCampaignGateError(ValueError):
    """Raised when conclusion-bearing coverage or a metric is invalid."""


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PooledCampaignGateError(f"{label} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PooledCampaignGateError(
            f"{label} must be finite numeric"
        ) from exc
    if not math.isfinite(result):
        raise PooledCampaignGateError(f"{label} must be finite numeric")
    return result


def _relative_gain(*, baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise PooledCampaignGateError(
            "relative-improvement baselines must be strictly positive"
        )
    return (baseline - candidate) / baseline


def collapse_member_replicates(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate complete member coverage and average mask replicates first."""

    required = tuple(metric_names)
    if not required or len(set(required)) != len(required):
        raise PooledCampaignGateError(
            "metric_names must be unique and nonempty"
        )
    grouped: dict[tuple[str, int, str, str], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("arm")),
            int(row.get("seed", -1)),
            str(row.get("core_alias")),
            str(row.get("mask_mode")),
        )
        replicate = int(row.get("mask_replicate", -1))
        if (
            key[0] not in ARMS
            or key[1] not in SEEDS
            or key[2] not in ALIASES
            or key[3] not in MASK_MODES
            or replicate not in REPLICATES
        ):
            raise PooledCampaignGateError(
                "member row has an unexpected arm, seed, alias, mode, or replicate"
            )
        bucket = grouped.setdefault(key, {})
        if replicate in bucket:
            raise PooledCampaignGateError(
                "member rows contain a duplicate mask replicate"
            )
        bucket[replicate] = row

    expected = {
        (arm, seed, alias, mode)
        for arm in ARMS
        for seed in SEEDS
        for alias in ALIASES
        for mode in MASK_MODES
    }
    if set(grouped) != expected:
        raise PooledCampaignGateError(
            "member rows do not have complete 2x7x10x3 coverage"
        )

    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        by_replicate = grouped[key]
        if set(by_replicate) != set(REPLICATES):
            raise PooledCampaignGateError(
                "member core/mode does not contain exactly three replicates"
            )
        item: dict[str, Any] = {
            "arm": key[0],
            "seed": key[1],
            "core_alias": key[2],
            "mask_mode": key[3],
            "mask_replicates": len(REPLICATES),
        }
        for metric in required:
            item[metric] = float(
                np.mean(
                    [
                        _finite(
                            by_replicate[replicate].get(metric),
                            label=f"member {metric}",
                        )
                        for replicate in REPLICATES
                    ],
                    dtype=np.float64,
                )
            )
        output.append(item)
    return output


def collapse_ensemble_replicates(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Validate complete ensemble coverage and average replicates by core."""

    required = tuple(metric_names)
    if not required or len(set(required)) != len(required):
        raise PooledCampaignGateError(
            "metric_names must be unique and nonempty"
        )
    grouped: dict[tuple[str, str, str], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("arm")),
            str(row.get("core_alias")),
            str(row.get("mask_mode")),
        )
        replicate = int(row.get("mask_replicate", -1))
        if (
            key[0] not in ARMS
            or key[1] not in ALIASES
            or key[2] not in MASK_MODES
            or replicate not in REPLICATES
        ):
            raise PooledCampaignGateError(
                "ensemble row has an unexpected arm, alias, mode, or replicate"
            )
        bucket = grouped.setdefault(key, {})
        if replicate in bucket:
            raise PooledCampaignGateError(
                "ensemble rows contain a duplicate mask replicate"
            )
        bucket[replicate] = row

    expected = {
        (arm, alias, mode)
        for arm in ARMS
        for alias in ALIASES
        for mode in MASK_MODES
    }
    if set(grouped) != expected:
        raise PooledCampaignGateError(
            "ensemble rows do not have complete 2x10x3 coverage"
        )

    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        by_replicate = grouped[key]
        if set(by_replicate) != set(REPLICATES):
            raise PooledCampaignGateError(
                "ensemble core/mode does not contain exactly three replicates"
            )
        item: dict[str, Any] = {
            "arm": key[0],
            "core_alias": key[1],
            "mask_mode": key[2],
            "mask_replicates": len(REPLICATES),
        }
        for metric in required:
            item[metric] = float(
                np.mean(
                    [
                        _finite(
                            by_replicate[replicate].get(metric),
                            label=f"ensemble {metric}",
                        )
                        for replicate in REPLICATES
                    ],
                    dtype=np.float64,
                )
            )
        output.append(item)
    return output


def _indexed(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if key in output:
            raise PooledCampaignGateError(f"duplicate aggregate key {key!r}")
        output[key] = row
    return output


def evaluate_frozen_gates(
    *,
    member_core_mode_rows: Sequence[Mapping[str, Any]],
    ensemble_core_mode_rows: Sequence[Mapping[str, Any]],
    prior_independent_seed0_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the four frozen pooled-campaign gates without inference claims."""

    member = _indexed(
        member_core_mode_rows,
        ("arm", "seed", "core_alias", "mask_mode"),
    )
    ensemble = _indexed(
        ensemble_core_mode_rows,
        ("arm", "core_alias", "mask_mode"),
    )
    prior = _indexed(
        prior_independent_seed0_rows,
        ("core_alias", "mask_mode"),
    )

    data_metrics = (
        "detection_bce",
        "positive_ordinal_mae",
        "reconstructed_count_log1p_mae",
    )
    data_checks: dict[str, Any] = {}
    for metric in data_metrics:
        gains = [
            _relative_gain(
                baseline=_finite(
                    prior[(alias, "whole_node")].get(metric),
                    label=f"prior {metric}",
                ),
                candidate=_finite(
                    member[(GAT_ARM, 0, alias, "whole_node")].get(metric),
                    label=f"pooled seed-0 {metric}",
                ),
            )
            for alias in ALIASES
        ]
        data_checks[metric] = {
            "mean_relative_improvement": float(np.mean(gains)),
            "favoring_core_count": int(np.count_nonzero(np.asarray(gains) > 0)),
            "passed": bool(
                float(np.mean(gains)) >= 0.02
                and np.count_nonzero(np.asarray(gains) > 0) >= 8
            ),
        }
    pooled_data_passed = all(
        check["passed"] for check in data_checks.values()
    )

    graph_gains = [
        _relative_gain(
            baseline=_finite(
                ensemble[(SELF_ARM, alias, "whole_node")].get("hybrid_loss"),
                label="self ensemble hybrid loss",
            ),
            candidate=_finite(
                ensemble[(GAT_ARM, alias, "whole_node")].get("hybrid_loss"),
                label="GAT ensemble hybrid loss",
            ),
        )
        for alias in ALIASES
    ]
    seed_pair_gains = []
    for seed in SEEDS:
        gat = np.mean(
            [
                _finite(
                    member[(GAT_ARM, seed, alias, "whole_node")].get(
                        "hybrid_loss"
                    ),
                    label="GAT member hybrid loss",
                )
                for alias in ALIASES
            ]
        )
        control = np.mean(
            [
                _finite(
                    member[(SELF_ARM, seed, alias, "whole_node")].get(
                        "hybrid_loss"
                    ),
                    label="self member hybrid loss",
                )
                for alias in ALIASES
            ]
        )
        seed_pair_gains.append(_relative_gain(baseline=control, candidate=gat))
    graph_noninferior = {
        metric: bool(
            np.mean(
                [
                    _finite(
                        ensemble[(GAT_ARM, alias, "whole_node")].get(metric),
                        label=f"GAT ensemble {metric}",
                    )
                    for alias in ALIASES
                ]
            )
            <= np.mean(
                [
                    _finite(
                        ensemble[(SELF_ARM, alias, "whole_node")].get(metric),
                        label=f"self ensemble {metric}",
                    )
                    for alias in ALIASES
                ]
            )
        )
        for metric in (
            "positive_ordinal_mae",
            "positive_continuous_huber",
        )
    }
    graph_gate = {
        "mean_relative_hybrid_loss_improvement": float(np.mean(graph_gains)),
        "favoring_core_count": int(
            np.count_nonzero(np.asarray(graph_gains) > 0)
        ),
        "favoring_seed_pair_count": int(
            np.count_nonzero(np.asarray(seed_pair_gains) > 0)
        ),
        "positive_metric_noninferiority": graph_noninferior,
    }
    graph_gate["passed"] = bool(
        graph_gate["mean_relative_hybrid_loss_improvement"] >= 0.02
        and graph_gate["favoring_core_count"] >= 8
        and graph_gate["favoring_seed_pair_count"] >= 5
        and all(graph_noninferior.values())
    )

    representation_checks: dict[str, Any] = {}
    for metric, reference_metric in (
        ("positive_ordinal_mae", "reference_per_gene_positive_ordinal_mae"),
        (
            "positive_continuous_huber",
            "reference_per_gene_positive_continuous_huber",
        ),
    ):
        gains = [
            _relative_gain(
                baseline=_finite(
                    ensemble[(GAT_ARM, alias, "whole_node")].get(
                        reference_metric
                    ),
                    label=reference_metric,
                ),
                candidate=_finite(
                    ensemble[(GAT_ARM, alias, "whole_node")].get(metric),
                    label=metric,
                ),
            )
            for alias in ALIASES
        ]
        representation_checks[metric] = {
            "mean_relative_improvement": float(np.mean(gains)),
            "favoring_core_count": int(
                np.count_nonzero(np.asarray(gains) > 0)
            ),
            "passed": bool(
                float(np.mean(gains)) >= 0.02
                and np.count_nonzero(np.asarray(gains) > 0) >= 8
            ),
        }
    model_detection = float(
        np.mean(
            [
                _finite(
                    ensemble[(GAT_ARM, alias, "whole_node")].get(
                        "detection_balanced_accuracy"
                    ),
                    label="model detection balanced accuracy",
                )
                for alias in ALIASES
            ]
        )
    )
    per_core_detection = float(
        np.mean(
            [
                _finite(
                    ensemble[(GAT_ARM, alias, "whole_node")].get(
                        "reference_per_gene_detection_balanced_accuracy"
                    ),
                    label="per-core reference detection balanced accuracy",
                )
                for alias in ALIASES
            ]
        )
    )
    pooled_detection = float(
        np.mean(
            [
                _finite(
                    ensemble[(GAT_ARM, alias, "whole_node")].get(
                        "reference_equal_core_detection_balanced_accuracy"
                    ),
                    label="pooled reference detection balanced accuracy",
                )
                for alias in ALIASES
            ]
        )
    )
    detection_passed = (
        model_detection > per_core_detection
        and model_detection > pooled_detection
    )
    representation_gate = {
        "positive_metrics": representation_checks,
        "detection_balanced_accuracy": {
            "model": model_detection,
            "per_core_reference": per_core_detection,
            "pooled_reference": pooled_detection,
            "passed": detection_passed,
        },
    }
    representation_gate["passed"] = bool(
        all(check["passed"] for check in representation_checks.values())
        and detection_passed
    )

    ensemble_checks: dict[str, Any] = {}
    for metric in (
        "hybrid_loss",
        "positive_ordinal_mae",
        "positive_continuous_huber",
    ):
        ensemble_mean = float(
            np.mean(
                [
                    _finite(
                        ensemble[(GAT_ARM, alias, "whole_node")].get(metric),
                        label=f"ensemble {metric}",
                    )
                    for alias in ALIASES
                ]
            )
        )
        individual_mean = float(
            np.mean(
                [
                    _finite(
                        member[(GAT_ARM, seed, alias, "whole_node")].get(
                            metric
                        ),
                        label=f"member {metric}",
                    )
                    for seed in SEEDS
                    for alias in ALIASES
                ]
            )
        )
        ensemble_checks[metric] = {
            "ensemble_equal_core_mean": ensemble_mean,
            "mean_individual_equal_core_metric": individual_mean,
            "passed": ensemble_mean <= individual_mean,
        }
    ensemble_gate = {
        "metrics": ensemble_checks,
        "passed": all(check["passed"] for check in ensemble_checks.values()),
    }
    return {
        "pooled_data_gate": {
            "metrics": data_checks,
            "passed": pooled_data_passed,
        },
        "graph_gate": graph_gate,
        "representation_gate": representation_gate,
        "ensemble_gate": ensemble_gate,
        "formal_core_level_inference": (
            "not_performed_shared_fitted_weights_couple_core_outcomes"
        ),
    }


__all__ = [
    "ALIASES",
    "ARMS",
    "GAT_ARM",
    "MASK_MODES",
    "PooledCampaignGateError",
    "REPLICATES",
    "SEEDS",
    "SELF_ARM",
    "collapse_ensemble_replicates",
    "collapse_member_replicates",
    "evaluate_frozen_gates",
]
