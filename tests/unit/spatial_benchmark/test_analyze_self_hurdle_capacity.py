from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.registry import Registry


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/analysis/analyze_self_hurdle_capacity.py"
_SPEC = importlib.util.spec_from_file_location(
    "analyze_self_hurdle_capacity_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

SelfHurdleAnalysisError = _MODULE.SelfHurdleAnalysisError


def _metric_fixture(
    *,
    state_mae_model: float = 0.99,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for mode in _MODULE.MASK_MODES:
        for replicate in range(_MODULE.EXPECTED_REPLICATES):
            row: dict[str, Any] = {
                "split": "fit",
                "mask_mode": mode,
                "mask_replicate": replicate,
            }
            for field in _MODULE.ACCURACY_FIELDS:
                row[field] = 0.6
                row[f"reference_per_gene_{field}"] = 0.5
                row[f"{field}_percent"] = 60.0
                row[f"reference_per_gene_{field}_percent"] = 50.0
            for field in _MODULE.ERROR_FIELDS:
                model = (
                    state_mae_model
                    if field == "positive_count_state_mae"
                    else 0.8
                )
                row[field] = model
                row[f"reference_per_gene_{field}"] = 1.0
            row[
                "detection_balanced_accuracy_gain_over_per_gene_reference"
            ] = 0.1
            for field in (
                "hurdle_loss",
                "positive_continuous_huber",
                "positive_count_state_mae",
            ):
                row[
                    f"{field}_relative_improvement_over_per_gene_reference"
                ] = 1.0 - float(row[field])
            rows.append(row)

        selected = [row for row in rows if row["mask_mode"] == mode]
        for field in (*_MODULE.ACCURACY_FIELDS, *_MODULE.ERROR_FIELDS):
            model = sum(float(row[field]) for row in selected) / 3.0
            reference = sum(
                float(row[f"reference_per_gene_{field}"])
                for row in selected
            ) / 3.0
            metrics[f"fit/{mode}/{field}"] = model
            metrics[f"fit/{mode}/reference_per_gene_{field}"] = reference
            if field in _MODULE.ACCURACY_FIELDS:
                metrics[f"fit/{mode}/{field}_percent"] = 100.0 * model
                metrics[
                    f"fit/{mode}/reference_per_gene_{field}_percent"
                ] = 100.0 * reference
        metrics[
            f"fit/{mode}/"
            "detection_balanced_accuracy_gain_over_per_gene_reference"
        ] = 0.1
        for field in (
            "hurdle_loss",
            "positive_continuous_huber",
            "positive_count_state_mae",
        ):
            metrics[
                f"fit/{mode}/{field}_relative_improvement_"
                "over_per_gene_reference"
            ] = 1.0 - float(selected[0][field])
    return metrics, rows


def test_gate_requires_all_three_whole_node_criteria() -> None:
    metrics, rows = _metric_fixture(state_mae_model=0.99)
    comparisons = _MODULE.build_mode_comparisons(metrics, rows)
    gate = _MODULE.evaluate_representation_gate(comparisons)
    assert gate["passed"] is False
    assert gate["checks"] == {
        "positive_detection_balanced_accuracy_gain": True,
        "positive_continuous_huber_improvement_at_least_2_percent": True,
        "positive_count_state_mae_improvement_at_least_2_percent": False,
    }

    passing_metrics, passing_rows = _metric_fixture(state_mae_model=0.97)
    passing = _MODULE.evaluate_representation_gate(
        _MODULE.build_mode_comparisons(passing_metrics, passing_rows)
    )
    assert passing["passed"] is True


def test_all_estimands_and_percentage_values_are_audited() -> None:
    metrics, rows = _metric_fixture()
    comparisons = _MODULE.build_mode_comparisons(metrics, rows)
    assert tuple(comparisons) == _MODULE.MASK_MODES
    for mode in _MODULE.MASK_MODES:
        assert set(comparisons[mode]) == set(
            (*_MODULE.ACCURACY_FIELDS, *_MODULE.ERROR_FIELDS)
        )
        assert comparisons[mode]["detection_balanced_accuracy"][
            "difference_percentage_points"
        ] == pytest.approx(10.0)

    metrics["fit/spatial_block/state8_exact_accuracy_percent"] = 61.0
    with pytest.raises(
        SelfHurdleAnalysisError, match="state8_exact_accuracy_percent"
    ):
        _MODULE.build_mode_comparisons(metrics, rows)


def test_replicate_relative_improvement_is_not_recomputed_as_best_only() -> None:
    metrics, rows = _metric_fixture()
    selected = [
        row for row in rows if row["mask_mode"] == "whole_node"
    ]
    selected[0][
        "positive_count_state_mae_relative_improvement_"
        "over_per_gene_reference"
    ] = 0.00
    selected[1][
        "positive_count_state_mae_relative_improvement_"
        "over_per_gene_reference"
    ] = 0.01
    selected[2][
        "positive_count_state_mae_relative_improvement_"
        "over_per_gene_reference"
    ] = 0.02
    metrics[
        "fit/whole_node/positive_count_state_mae_relative_improvement_"
        "over_per_gene_reference"
    ] = 0.01
    comparisons = _MODULE.build_mode_comparisons(metrics, rows)
    assert comparisons["whole_node"]["positive_count_state_mae"][
        "relative_improvement_percent"
    ] == pytest.approx(1.0)


def test_strict_json_and_embedded_receipt_checksum(tmp_path: Path) -> None:
    with pytest.raises(SelfHurdleAnalysisError, match="duplicate key"):
        _MODULE._strict_json_text('{"a": 1, "a": 2}', "fixture")
    with pytest.raises(SelfHurdleAnalysisError, match="non-finite"):
        _MODULE._strict_json_text('{"a": NaN}', "fixture")

    payload = {"schema_version": 1, "campaign_id": _MODULE.CAMPAIGN_ID}
    receipt = {**payload, "checksum": canonical_sha256(payload)}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert _MODULE._verified_receipt(path, "fixture")["checksum"] == receipt[
        "checksum"
    ]
    receipt["campaign_id"] = "tampered"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SelfHurdleAnalysisError, match="does not verify"):
        _MODULE._verified_receipt(path, "fixture")


def _config(alias: str, stage: str) -> dict[str, Any]:
    pilot = stage == "resource"
    return {
        "campaign": {
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "frozen_contract_sha256": _MODULE.CONTRACT_SHA256,
        },
        "experiment": {
            "biological_unit_alias": alias,
            "arm": "self-hurdle",
            "resource_pilot": pilot,
            "conclusion_eligible": not pilot,
            "graph_arms_authorized": False,
        },
        "classification": {
            "lifecycle_stage": (
                "diagnostic" if pilot else "exploratory_screen"
            )
        },
        "dataset": {
            "biological_unit_alias": alias,
            "validation_or_test_partition_present": False,
            "patient_generalization_supported": False,
            "dataset_id": f"fixture_{alias.lower()}",
            "version": "v1",
            "split_id": f"split_{alias.lower()}",
        },
        "model": {
            "name": "self-hurdle-count",
            "family": "self_only_hurdle_count",
            "uses_graph_inputs": False,
            "uses_edge_inputs": False,
            "expected_trainable_parameter_count": (
                _MODULE.EXPECTED_PARAMETER_COUNT
            ),
            "embedding_dim": 768,
        },
        "graph": {
            "enabled": False,
            "construction_performed": False,
            "expected_directed_edges": 0,
            "neighbor_k": 1,
        },
        "features": {"use_edge_features": False},
        "masking": {"type": "mixed_expression_masking"},
        "trainer": {
            "max_epochs": 2 if pilot else 200,
            "fixed_epoch_budget": True,
            "early_stopping": False,
            "restore_best": False,
            "checkpoint_policy": "last_only",
            "graph_execution": "none_graph_inputs_prohibited",
            "learning_rate": 0.0003,
            "batch_size": 1,
        },
        "evaluation": {
            "splits": ["fit"],
            "generalization_estimate": False,
            "validation_or_test_selection": False,
            "mask_modes": [
                "partial_gene",
                "whole_node",
                "spatial_block",
            ],
            "mask_replicates_per_mode": 1 if pilot else 3,
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }


def _registry_fixture(
    tmp_path: Path,
) -> tuple[
    Registry,
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    registry = Registry(tmp_path / "tracking.sqlite3")
    registry.create_campaign(_MODULE.CAMPAIGN_ID, name="fixture")
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    enqueue: dict[str, dict[str, Any]] = {}
    for stage in _MODULE.STAGES:
        receipt_jobs: list[dict[str, Any]] = []
        for alias in _MODULE.ALIASES:
            config = _config(alias, stage)
            config_sha = canonical_sha256(config)
            scientific_id = (
                f"sci_fixture_{stage}_{alias.lower().replace('-', '')}"
            )
            registry.register_variant(
                scientific_id,
                campaign_id=_MODULE.CAMPAIGN_ID,
                configuration=config,
            )
            run_id = f"r_fixture_{stage}_{alias.lower()}"
            registry.create_run(
                run_id,
                campaign_id=_MODULE.CAMPAIGN_ID,
                scientific_id=scientific_id,
                repro_id=f"rep_{run_id}",
                seed=0,
                fold=0,
                attempt=1,
                configuration=config,
                status="completed",
                artifact_path=tmp_path / run_id,
                duration_seconds=10.0,
                end_time="2026-07-29T00:00:10Z",
                parameter_count=_MODULE.EXPECTED_PARAMETER_COUNT,
                primary_metric_name=_MODULE.PRIMARY_METRIC,
                primary_metric_value=0.5,
            )
            job_id = f"q_fixture_{stage}_{alias.lower()}"
            registry.enqueue(
                campaign_id=_MODULE.CAMPAIGN_ID,
                configuration=config,
                command=["python", "fixture.py"],
                requested_gpu="0",
                job_id=job_id,
            )
            with registry.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    UPDATE queue_jobs
                    SET status = 'completed', run_id = ?, finished_at = ?
                    WHERE job_id = ?
                    """,
                    (run_id, "2026-07-29T00:00:10Z", job_id),
                )
            expected[(stage, alias)] = {
                "alias": alias,
                "stage": stage,
                "arm": "self-hurdle",
                "config_sha256": config_sha,
            }
            receipt_jobs.append(
                {
                    "alias": alias,
                    "config_sha256": config_sha,
                    "job_id": job_id,
                }
            )
        enqueue[stage] = {"jobs": receipt_jobs}
    return registry, expected, enqueue


def test_registry_inventory_requires_exact_four_locked_jobs(
    tmp_path: Path,
) -> None:
    registry, expected, enqueue = _registry_fixture(tmp_path)
    slots, failed = _MODULE.registry_inventory(
        registry=registry,
        project_root=tmp_path,
        expected=expected,
        enqueue=enqueue,
    )
    assert set(slots) == set(expected)
    assert failed == []

    registry.enqueue(
        campaign_id=_MODULE.CAMPAIGN_ID,
        configuration=_config("ANC-03", "science"),
        command=["python", "unexpected.py"],
        job_id="q_unexpected",
    )
    with pytest.raises(
        SelfHurdleAnalysisError, match="queue coverage differs"
    ):
        _MODULE.registry_inventory(
            registry=registry,
            project_root=tmp_path,
            expected=expected,
            enqueue=enqueue,
        )


def test_gpu_accounting_uses_full_registered_duration_for_all_four_runs() -> None:
    durations = [
        59.127588244038634,
        34.61043364799116,
        146.52503370004706,
        73.92951618903317,
    ]
    assert _MODULE.conservative_campaign_gpu_hours(durations) == pytest.approx(
        314.19257178111 / 3600.0
    )
    with pytest.raises(
        SelfHurdleAnalysisError, match="all four registered runs"
    ):
        _MODULE.conservative_campaign_gpu_hours(durations[1:])
