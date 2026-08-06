from __future__ import annotations

from copy import deepcopy

import pytest

from spatial_benchmark.pooled_campaign_gates import (
    ALIASES,
    ARMS,
    GAT_ARM,
    MASK_MODES,
    REPLICATES,
    SEEDS,
    SELF_ARM,
    PooledCampaignGateError,
    collapse_ensemble_replicates,
    collapse_member_replicates,
    evaluate_frozen_gates,
)


_METRICS = (
    "hybrid_loss",
    "detection_bce",
    "positive_ordinal_mae",
    "positive_continuous_huber",
    "reconstructed_count_log1p_mae",
    "detection_balanced_accuracy",
    "reference_per_gene_positive_ordinal_mae",
    "reference_per_gene_positive_continuous_huber",
    "reference_per_gene_detection_balanced_accuracy",
    "reference_equal_core_detection_balanced_accuracy",
)


def _member_core_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        for seed in SEEDS:
            for alias in ALIASES:
                for mode in MASK_MODES:
                    gat = arm == GAT_ARM
                    rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "core_alias": alias,
                            "mask_mode": mode,
                            "hybrid_loss": 0.90 if gat else 1.0,
                            "detection_bce": 0.90 if gat else 1.0,
                            "positive_ordinal_mae": 0.90 if gat else 1.0,
                            "positive_continuous_huber": 0.90 if gat else 1.0,
                            "reconstructed_count_log1p_mae": (
                                0.90 if gat else 1.0
                            ),
                        }
                    )
    return rows


def _ensemble_core_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        for alias in ALIASES:
            for mode in MASK_MODES:
                gat = arm == GAT_ARM
                rows.append(
                    {
                        "arm": arm,
                        "core_alias": alias,
                        "mask_mode": mode,
                        "hybrid_loss": 0.85 if gat else 1.0,
                        "positive_ordinal_mae": 0.85 if gat else 1.0,
                        "positive_continuous_huber": 0.85 if gat else 1.0,
                        "detection_balanced_accuracy": 0.70 if gat else 0.65,
                        "reference_per_gene_positive_ordinal_mae": 1.0,
                        "reference_per_gene_positive_continuous_huber": 1.0,
                        "reference_per_gene_detection_balanced_accuracy": 0.60,
                        "reference_equal_core_detection_balanced_accuracy": 0.65,
                    }
                )
    return rows


def _prior_rows() -> list[dict[str, object]]:
    return [
        {
            "core_alias": alias,
            "mask_mode": mode,
            "detection_bce": 1.0,
            "positive_ordinal_mae": 1.0,
            "reconstructed_count_log1p_mae": 1.0,
        }
        for alias in ALIASES
        for mode in MASK_MODES
    ]


def test_all_frozen_gates_pass_only_from_complete_aggregate_evidence() -> None:
    result = evaluate_frozen_gates(
        member_core_mode_rows=_member_core_rows(),
        ensemble_core_mode_rows=_ensemble_core_rows(),
        prior_independent_seed0_rows=_prior_rows(),
    )

    assert result["pooled_data_gate"]["passed"] is True
    assert result["graph_gate"]["passed"] is True
    assert result["representation_gate"]["passed"] is True
    assert result["ensemble_gate"]["passed"] is True
    assert result["graph_gate"]["favoring_core_count"] == 10
    assert result["graph_gate"]["favoring_seed_pair_count"] == 7
    assert result["formal_core_level_inference"].startswith("not_performed")


def test_high_overall_accuracy_cannot_rescue_a_failed_graph_gate() -> None:
    ensemble = _ensemble_core_rows()
    for row in ensemble:
        if row["arm"] == GAT_ARM and row["mask_mode"] == "whole_node":
            row["hybrid_loss"] = 1.01
            row["detection_balanced_accuracy"] = 0.999
    result = evaluate_frozen_gates(
        member_core_mode_rows=_member_core_rows(),
        ensemble_core_mode_rows=ensemble,
        prior_independent_seed0_rows=_prior_rows(),
    )

    assert result["graph_gate"]["passed"] is False
    assert result["graph_gate"]["favoring_core_count"] == 0


def test_replicate_collapsers_fail_closed_on_missing_or_duplicate_rows() -> None:
    member_rows: list[dict[str, object]] = []
    ensemble_rows: list[dict[str, object]] = []
    for arm in ARMS:
        for seed in SEEDS:
            for alias in ALIASES:
                for mode in MASK_MODES:
                    for replicate in REPLICATES:
                        member_rows.append(
                            {
                                "arm": arm,
                                "seed": seed,
                                "core_alias": alias,
                                "mask_mode": mode,
                                "mask_replicate": replicate,
                                **{metric: 1.0 for metric in _METRICS},
                            }
                        )
        for alias in ALIASES:
            for mode in MASK_MODES:
                for replicate in REPLICATES:
                    ensemble_rows.append(
                        {
                            "arm": arm,
                            "core_alias": alias,
                            "mask_mode": mode,
                            "mask_replicate": replicate,
                            **{metric: 1.0 for metric in _METRICS},
                        }
                    )

    assert len(collapse_member_replicates(member_rows, metric_names=_METRICS)) == (
        2 * 7 * 10 * 3
    )
    assert len(
        collapse_ensemble_replicates(ensemble_rows, metric_names=_METRICS)
    ) == (2 * 10 * 3)

    with pytest.raises(PooledCampaignGateError, match="three replicates"):
        collapse_member_replicates(member_rows[:-1], metric_names=_METRICS)
    duplicate = deepcopy(ensemble_rows)
    duplicate.append(deepcopy(ensemble_rows[0]))
    with pytest.raises(PooledCampaignGateError, match="duplicate"):
        collapse_ensemble_replicates(duplicate, metric_names=_METRICS)
