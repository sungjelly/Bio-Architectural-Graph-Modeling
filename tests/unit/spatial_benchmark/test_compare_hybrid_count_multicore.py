from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256, scientific_id
from spatial_benchmark.registry import Registry


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/analysis/compare_hybrid_count_multicore.py"
_SPEC = importlib.util.spec_from_file_location(
    "compare_hybrid_count_multicore_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

HybridCountComparisonError = _MODULE.HybridCountComparisonError
_ALIASES = _MODULE._CORE_ALIASES
_ARMS = _MODULE._ARMS


def _config(alias: str, arm: str) -> dict[str, Any]:
    model_name = {
        "hybrid-gat-k1000": "hybrid-count-gat",
        "hybrid-matched-self": "hybrid-count-matched-self",
    }[arm]
    token = {
        "hybrid-gat-k1000": "hybrid_gat_k1000",
        "hybrid-matched-self": "hybrid_matched_self",
    }[arm]
    graph_arm = arm == "hybrid-gat-k1000"
    return {
        "campaign": {"campaign_id": _MODULE._CAMPAIGN_ID},
        "classification": {"lifecycle_stage": "exploratory_screen"},
        "dataset": {
            "dataset_id": f"fixture_{alias.lower()}",
            "version": "v1",
            "split_id": f"split_{alias.lower()}",
            "biological_unit_alias": alias,
            "task": "masked_expression_hybrid_count",
            "target_scale": (
                "raw_biological_probe_counts_with_per_gene_all_fit_"
                "standardized_log1p"
            ),
        },
        "experiment": {
            "biological_unit_alias": alias,
            "arm": arm,
            "variant_label": (
                alias.lower().replace("-", "") + f"_{token}_full_core"
            ),
            "conclusion_eligible": True,
            "resource_pilot": False,
        },
        "model": {
            "name": model_name,
            "family": (
                "hybrid_count_edge_conditioned_gatv2"
                if graph_arm
                else "hybrid_count_parameter_matched_self_control"
            ),
            "count_representation_schema": _MODULE._REPRESENTATION_SCHEMA,
            "output_count_states": 8,
            "input_mask_token_id": 8,
            "detection_logits_per_gene": 1,
            "positive_ordinal_logits_per_gene": 6,
            "continuous_predictions_per_gene": 1,
            "embedding_dim": 512,
            "hidden_dim": 512,
            "graph_layers": 2,
            "attention_heads": 4,
            "ffn_dim": 512,
            "decoder_dim": 512,
            "uses_graph_inputs": graph_arm,
            "uses_edge_inputs": graph_arm,
        },
        "features": {
            "use_edge_features": graph_arm,
            "node_metadata": {"fields": list(_MODULE._NODE_METADATA_FIELDS)},
            "edge_features": (
                {"fields": list(_MODULE._EDGE_FIELDS)} if graph_arm else []
            ),
        },
        "graph": {
            "neighbor_k": 1000,
            "k": 1000,
            "symmetry": "mutual",
            "edge_dropout": 0.0,
            "self_loops": False,
            "coordinates_are_node_covariates": False,
        },
        "masking": {"type": "partial_gene_whole_node_spatial_block"},
        "trainer": {
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "huber_delta": 1.0,
            "max_epochs": 200,
            "fixed_epoch_budget": True,
            "early_stopping": False,
            "restore_best": False,
            "primary_checkpoint_role": "last",
            "checkpoint_policy": "last_only",
            "neighbor_sampling": False,
            "graph_execution": _MODULE._GRAPH_EXECUTION,
            "objective": (
                "equal_weight_balanced_detection_ordinal_positive_huber"
            ),
        },
        "evaluation": {
            "task_family": "masked_expression_hybrid_count",
            "protocol": _MODULE._PROTOCOL,
            "canonical_prediction_split": "fit",
            "primary_metric": _MODULE._PRIMARY_METRIC,
            "primary_direction": "minimize",
            "splits": ["fit"],
            "mask_modes": list(_MODULE._MASK_MODES),
            "mask_replicates_per_mode": 3,
            "generalization_estimate": False,
            "validation_or_test_selection": False,
            "conclusion_bearing": True,
            "diagnostic_only": False,
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }


def _registry(tmp_path: Path) -> Registry:
    registry = Registry(tmp_path / "tracking.sqlite3")
    registry.create_campaign(
        _MODULE._CAMPAIGN_ID,
        name="hybrid comparison fixture",
    )
    return registry


def _register_run(
    registry: Registry,
    *,
    alias: str,
    arm: str,
    suffix: str = "primary",
    status: str = "completed",
    attempt: int = 1,
    retry_of: str | None = None,
) -> str:
    config = _config(alias, arm)
    config["attempt"] = attempt
    identifier = scientific_id(config)
    registry.register_variant(
        identifier,
        campaign_id=_MODULE._CAMPAIGN_ID,
        configuration=config,
    )
    run_id = f"run_{alias.lower()}_{arm.replace('-', '_')}_{suffix}"
    registry.create_run(
        run_id,
        campaign_id=_MODULE._CAMPAIGN_ID,
        scientific_id=identifier,
        repro_id=f"rep_{run_id}",
        seed=0,
        fold=0,
        attempt=attempt,
        configuration=config,
        status=status,
        artifact_path=(tmp_path := registry.path.parent) / run_id,
        failure_category="fixture_failure" if status == "failed" else None,
        retry_of=retry_of,
    )
    return run_id


def test_registry_coverage_is_exact_and_exposes_failures_and_duplicates(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    for alias in _ALIASES:
        for arm in _ARMS:
            _register_run(registry, alias=alias, arm=arm)
    _register_run(
        registry,
        alias="ANC-01",
        arm="hybrid-gat-k1000",
        suffix="failed_retry",
        status="failed",
    )

    coverage = _MODULE._registry_inventory(registry)
    assert coverage["exact_primary_coverage"] is True
    assert coverage["selected_completed_run_count"] == 20
    assert len(coverage["failed_attempts"]) == 1
    assert coverage["duplicate_completed_slots"] == []

    duplicate_run = _register_run(
        registry,
        alias="ANC-01",
        arm="hybrid-gat-k1000",
        suffix="duplicate_success",
    )
    duplicate = _MODULE._registry_inventory(registry)
    assert duplicate["exact_primary_coverage"] is False
    assert duplicate["missing_slots"] == []
    assert duplicate["duplicate_completed_slots"] == [
        {
            "core_alias": "ANC-01",
            "arm": "hybrid-gat-k1000",
            "run_ids": sorted(
                [
                    "run_anc-01_hybrid_gat_k1000_primary",
                    duplicate_run,
                ]
            ),
        }
    ]


def test_attempt_two_recovery_is_eligible_and_attempt_one_failure_is_exposed(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    failed_run = _register_run(
        registry,
        alias="ANC-01",
        arm="hybrid-gat-k1000",
        suffix="attempt_1_failed",
        status="failed",
        attempt=1,
    )
    recovered_run = _register_run(
        registry,
        alias="ANC-01",
        arm="hybrid-gat-k1000",
        suffix="attempt_2_recovered",
        attempt=2,
        retry_of=failed_run,
    )
    for alias in _ALIASES:
        for arm in _ARMS:
            if alias == "ANC-01" and arm == "hybrid-gat-k1000":
                continue
            _register_run(registry, alias=alias, arm=arm)

    coverage = _MODULE._registry_inventory(registry)
    assert coverage["exact_primary_coverage"] is True
    assert coverage["selected"][("ANC-01", "hybrid-gat-k1000")][
        "run_id"
    ] == recovered_run
    assert any(
        row["run_id"] == failed_run for row in coverage["failed_attempts"]
    )
    assert next(
        row
        for row in coverage["attempt_inventory"]
        if row["run_id"] == recovered_run
    )["attempt"] == 2


def _raw_mask_row(
    *,
    mode: str,
    replicate: int,
    value: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": "fit",
        "mask_mode": mode,
        "mask_replicate": replicate,
        "mask_entry_id": f"fit-{mode}-{replicate}",
        "mask_seed": 100 + replicate,
        "mask_checksum": f"{replicate + 1:064x}",
        "n_masked": 1000 + replicate,
    }
    for metric in _MODULE._REQUIRED_SCALAR_METRICS:
        row[metric] = value
    row["detection_precision"] = None if replicate == 0 else value
    for metric, length in _MODULE._VECTOR_METRICS.items():
        row[metric] = [value + state for state in range(length)]
    return row


def test_mask_metrics_average_replicates_before_core_aggregation() -> None:
    raw = [
        _raw_mask_row(mode=mode, replicate=replicate, value=float(replicate + 1))
        for mode in _MODULE._MASK_MODES
        for replicate in range(3)
    ]
    normalized = _MODULE._normalize_per_mask_rows(
        raw,
        alias="ANC-01",
        arm="hybrid-gat-k1000",
        run_id="run_fixture",
    )
    aggregated = _MODULE.aggregate_replicates(normalized)

    assert len(normalized) == 9
    assert len(aggregated) == 3
    whole = next(row for row in aggregated if row["mask_mode"] == "whole_node")
    assert whole["hybrid_loss"] == pytest.approx(2.0)
    assert whole["state8_recall_7"] == pytest.approx(9.0)
    assert whole["detection_precision"] == pytest.approx(2.5)
    assert whole["detection_precision_defined_replicates"] == 2


def test_equal_core_aggregate_includes_per_state_support_and_recall() -> None:
    rows: list[dict[str, Any]] = []
    for index, alias in enumerate(_ALIASES, start=1):
        for arm in _ARMS:
            for mode in _MODULE._MASK_MODES:
                row: dict[str, Any] = {
                    "core_alias": alias,
                    "arm": arm,
                    "mask_mode": mode,
                }
                for metric in _MODULE._REQUIRED_SCALAR_METRICS:
                    row[metric] = float(index)
                for metric, length in _MODULE._VECTOR_METRICS.items():
                    for state in range(length):
                        row[f"{metric}_{state}"] = float(index + state)
                rows.append(row)

    aggregates = _MODULE._aggregate_across_cores(rows)
    selected = next(
        row
        for row in aggregates
        if row["arm"] == "hybrid-gat-k1000"
        and row["mask_mode"] == "whole_node"
        and row["metric"] == "state8_recall_7"
    )
    assert selected["independent_core_count"] == 10
    assert selected["defined_core_count"] == 10
    assert selected["equal_core_mean"] == pytest.approx(12.5)


def _gate_rows(*, favorable: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, alias in enumerate(_ALIASES, start=1):
        self_loss = 1.0 + 0.001 * index
        gat_loss = self_loss * (0.90 if favorable else 1.10)
        for arm in _ARMS:
            gat = arm == "hybrid-gat-k1000"
            rows.append(
                {
                    "core_alias": alias,
                    "arm": arm,
                    "run_id": f"run_{index}_{arm}",
                    "mask_mode": "whole_node",
                    "hybrid_loss": gat_loss if gat else self_loss,
                    "detection_bce": 0.8 if gat and favorable else 1.0,
                    "ordinal_bce": 0.8 if gat and favorable else 1.0,
                    "positive_continuous_huber": (
                        0.8 if gat and favorable else 1.0
                    ),
                    "positive_ordinal_mae": (
                        0.8 if gat and favorable else 0.9
                    ),
                    "reference_per_gene_positive_ordinal_mae": 1.0,
                    "reference_per_gene_positive_continuous_huber": 1.0,
                    "detection_balanced_accuracy": 0.7 if gat else 0.6,
                    "reference_per_gene_detection_balanced_accuracy": 0.5,
                }
            )
    return rows


def test_frozen_gates_use_ten_raw_component_pairs_and_holm_three() -> None:
    result = _MODULE.evaluate_frozen_gates(
        _gate_rows(), all_runs_verified=True
    )

    assert result["graph_gate"]["passes"] is True
    assert result["representation_gate"]["passes"] is True
    component = result["graph_gate"]["component_inference"]
    assert set(component) == set(_MODULE._COMPONENT_METRICS)
    for value in component.values():
        assert value["unit_count"] == 10
        assert value["permutation_count"] == 1024
        assert value["p_value"] == pytest.approx(1 / 1024)
        assert value["holm"]["reject"] is True
    criterion = result["graph_gate"]["criteria"][
        "mean_paired_relative_hybrid_loss_improvement_at_least_2_percent"
    ]
    assert criterion["observed"] == pytest.approx(0.10)

    negative = _MODULE.evaluate_frozen_gates(
        _gate_rows(favorable=False), all_runs_verified=True
    )
    assert negative["graph_gate"]["passes"] is False
    assert negative["graph_gate"]["criteria"][
        "at_least_8_of_10_cores_favor_gat"
    ]["passes"] is False


def _report_result() -> dict[str, Any]:
    gates = _MODULE.evaluate_frozen_gates(
        _gate_rows(), all_runs_verified=True
    )
    return {
        "status": "complete",
        "graph_gate": gates["graph_gate"],
        "representation_gate": gates["representation_gate"],
        "prior_categorical_comparison": {"available": False, "reason": "fixture"},
        "negative_evidence": [],
        "limitations": ["Fixture limitation."],
        "maximum_defensible_conclusion": "fixture conclusion",
        "tables": {
            "gate_criteria": [{"gate": "graph", "passes": True}],
            "per_mask_metrics": [],
        },
    }


def test_report_is_portable_complete_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    destination = _MODULE.write_comparison(
        _report_result(), tmp_path / "comparison"
    )
    names = {path.name for path in destination.iterdir()}
    assert {
        "comparison.json",
        "report.md",
        "report.html",
        "report_manifest.json",
        "gate_criteria.csv",
        "per_mask_metrics.csv",
    } <= names
    document = (destination / "report.html").read_text(encoding="utf-8")
    assert "<svg" in document
    assert "aria-label=" in document
    assert "<title>" in document
    assert "@media print" in document
    assert "http://" not in document
    assert "https://" not in document
    assert "<link" not in document
    assert "<script" not in document
    manifest = json.loads(
        (destination / "report_manifest.json").read_text(encoding="utf-8")
    )
    assert "report.html" in manifest["files"]
    with pytest.raises(HybridCountComparisonError, match="already exists"):
        _MODULE.write_comparison(_report_result(), destination)


def test_prior_categorical_json_is_labeled_and_deidentified(
    tmp_path: Path,
) -> None:
    prior = _MODULE._load_prior_categorical(
        _MODULE._PRIOR_CATEGORICAL_DEFAULT
    )
    assert prior["available"] is True
    assert prior["single_core"] is True
    assert prior["exchangeable_with_current_ten_core_study"] is False
    assert prior["comparison_is_descriptive_only"] is True
    assert prior["prior_tissue_context"] == "true_normal"
    assert "scope" not in prior
    assert "prior_gate" not in prior
    assert {row["variant"] for row in prior["rows"]} == set(
        _MODULE._PRIOR_SAFE_VARIANTS.values()
    )
    assert len(prior["rows"]) == 2

    source = json.loads(
        _MODULE._PRIOR_CATEGORICAL_DEFAULT.read_text(encoding="utf-8")
    )
    _, value = source["variant_aggregates"].popitem()
    source["variant_aggregates"]["unrecognized_variant"] = value
    unexpected = tmp_path / "unexpected_prior.json"
    unexpected.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(HybridCountComparisonError, match="variant identity"):
        _MODULE._load_prior_categorical(unexpected)


def _checksummed(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["checksum"] = canonical_sha256(result)
    return result


def test_pilot_gate_receipts_are_checksum_bound_and_fail_closed() -> None:
    frozen_sha = "a" * 64
    materialization = _checksummed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE._MATERIALIZATION_KIND,
            "campaign_id": _MODULE._CAMPAIGN_ID,
            "frozen_contract": {"sha256": frozen_sha},
            "parameter_count": _MODULE._EXPECTED_PARAMETER_COUNT,
            "counts": {
                "aliases": 10,
                "pilot_configs": 2,
                "production_configs": 20,
            },
            "registry_mutation_performed": False,
            "queue_mutation_performed": False,
            "training_performed": False,
        }
    )
    gate = _checksummed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE._PILOT_GATE_KIND,
            "campaign_id": _MODULE._CAMPAIGN_ID,
            "materialization_checksum": materialization["checksum"],
            "pilot_enqueue_receipt_checksum": "9" * 64,
            "frozen_contract_sha256": frozen_sha,
            "thresholds": {
                "peak_allocated_vram_gib_maximum": 20.5,
                "fp32_amp_absolute_total_loss_discrepancy_maximum": 0.001,
                "projected_gat_runtime_hours_per_core_maximum": 6.0,
            },
            "same_frozen_precision_batch": True,
            "same_evaluation_masks": True,
            "same_verified_graph": True,
            "gate_passed": True,
            "production_authorized": True,
            "failure_reasons": [],
            "jobs": [
                {
                    "alias": "ANC-01",
                    "arm": arm,
                    "verified_bundle": True,
                    "finite_losses_and_gradients": True,
                    "parameter_match": True,
                    "parameter_count": _MODULE._EXPECTED_PARAMETER_COUNT,
                    "checkpoint_verified": True,
                    "checkpoint_role": "last",
                    "checkpoint_epoch": 1,
                    "checkpoint_sha256": (
                        "e" * 64
                        if arm == "hybrid-gat-k1000"
                        else "f" * 64
                    ),
                    "precision_mask_checksum": "b" * 64,
                    "fp32_amp_absolute_total_loss_discrepancy": 0.0005,
                    "precision_equivalence_passed": True,
                    "peak_allocated_vram_gib": 20.0,
                    "peak_vram_passed": True,
                    "projected_200_epoch_runtime_hours": 5.0,
                    "projected_runtime_passed": True,
                    "runner_pilot_gate_passed": True,
                    "evaluation_mask_bundle_sha256": "c" * 64,
                    "graph_sha256": "d" * 64,
                }
                for arm in _ARMS
            ],
        }
    )

    verified = _MODULE.validate_pilot_gate_receipts(
        materialization, gate, frozen_contract_sha256=frozen_sha
    )
    assert verified["verified"] is True
    assert verified["pilot_gate_checksum"] == gate["checksum"]

    mutations = (
        lambda value: value.__setitem__("production_authorized", False),
        lambda value: value["thresholds"].__setitem__(
            "peak_allocated_vram_gib_maximum", 20.6
        ),
        lambda value: value.__setitem__("same_verified_graph", False),
        lambda value: value["jobs"][0].__setitem__(
            "precision_equivalence_passed", False
        ),
        lambda value: value["jobs"][0].__setitem__(
            "parameter_count", _MODULE._EXPECTED_PARAMETER_COUNT + 1
        ),
        lambda value: value["jobs"][0].__setitem__(
            "finite_losses_and_gradients", False
        ),
    )
    for mutate in mutations:
        invalid_gate = json.loads(json.dumps(gate))
        invalid_gate.pop("checksum")
        mutate(invalid_gate)
        invalid_gate = _checksummed(invalid_gate)
        with pytest.raises(HybridCountComparisonError, match="passing two-arm"):
            _MODULE.validate_pilot_gate_receipts(
                materialization,
                invalid_gate,
                frozen_contract_sha256=frozen_sha,
            )


def test_run_loader_matches_current_runner_artifact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias = "ANC-01"
    arm = "hybrid-gat-k1000"
    run_id = "run_current_runner_schema"
    root = tmp_path / run_id
    (root / "metrics").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "diagnostics").mkdir()
    (root / "config.resolved.yaml").write_text(
        yaml.safe_dump(_config(alias, arm)), encoding="utf-8"
    )

    raw_rows = [
        _raw_mask_row(
            mode=mode,
            replicate=replicate,
            value=float(replicate + 1),
        )
        for mode in _MODULE._MASK_MODES
        for replicate in range(3)
    ]
    with (root / "metrics/evaluation_replicates.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in raw_rows:
            handle.write(json.dumps(row) + "\n")
    with (root / "metrics/history.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in range(200):
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_hybrid_loss": 1.0,
                        "train_detection_bce": 1.0,
                        "train_ordinal_bce": 1.0,
                        "train_positive_continuous_huber": 1.0,
                        "gradient_norm": 1.0,
                        "duration_seconds": 0.5,
                    }
                )
                + "\n"
            )

    final_metrics: dict[str, Any] = {}
    for mode in _MODULE._MASK_MODES:
        mode_rows = [row for row in raw_rows if row["mask_mode"] == mode]
        for metric in _MODULE._REQUIRED_SCALAR_METRICS:
            values = [row[metric] for row in mode_rows if row[metric] is not None]
            final_metrics[f"fit/{mode}/{metric}"] = (
                sum(values) / len(values) if values else None
            )
    final_metrics.update(
        {
            "resource/total_duration_seconds": 123.0,
            "resource/peak_vram_gib": 4.5,
        }
    )
    summary = {
        "run_id": run_id,
        "status": "success",
        "training_exit_status": "success",
        "campaign_id": _MODULE._CAMPAIGN_ID,
        "biological_unit_alias": alias,
        "evaluation_protocol": _MODULE._PROTOCOL,
        "task_family": "masked_expression_hybrid_count",
        "public_variant": arm,
        "model_name": "hybrid-count-gat",
        "checkpoint_role": "last",
        "checkpoint": {
            "role": "last",
            "final_epoch": 199,
            "policy": _MODULE._CHECKPOINT_POLICY,
            "monitored_metric": None,
        },
        "primary_metric_name": _MODULE._PRIMARY_METRIC,
        "primary_metric_value": 2.0,
        "metrics": final_metrics,
        "parameter_count": _MODULE._EXPECTED_PARAMETER_COUNT,
        "exact_parameter_match": True,
        "final_epoch": 199,
        "fixed_epoch_budget": 200,
        "graph_sha256": "b" * 64,
        "graph_supplied_to_model": True,
        "graph_interpretation": "broad_regional_context_not_direct_interaction",
        "evaluation_mask_replicates_per_mode": 3,
        "evaluation_metrics_include_all_configured_replicates_per_mode": True,
        "diagnostic_resource_pilot": False,
        "conclusion_eligible": True,
        "generalization_estimate": False,
        "duration_seconds": 123.0,
        "peak_vram_gib": 4.5,
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "metrics/final.json").write_text(
        json.dumps(final_metrics), encoding="utf-8"
    )
    parameter_audit = {
        "schema": "hybrid_count_parameter_structure_audit_v1",
        "trainable_parameter_count_graph": _MODULE._EXPECTED_PARAMETER_COUNT,
        "trainable_parameter_count_self": _MODULE._EXPECTED_PARAMETER_COUNT,
        "exact_trainable_parameter_match": True,
        "encoder_initial_state_bit_identical": True,
        "decoder_initial_state_bit_identical": True,
        "graph_layer_parameter_budgets": [1_000_000, 1_000_000],
        "self_layer_parameter_budgets": [1_000_000, 1_000_000],
    }
    (root / "provenance/full_core_training.json").write_text(
        json.dumps(
            {
                "training_protocol": _MODULE._TRAINING_PROTOCOL,
                "graph_execution": _MODULE._GRAPH_EXECUTION,
                "checkpoint_policy": _MODULE._CHECKPOINT_POLICY,
                "final_epoch": 199,
                "fixed_epoch_budget": 200,
                "parameter_count": _MODULE._EXPECTED_PARAMETER_COUNT,
                "parameter_structure_audit": parameter_audit,
            }
        ),
        encoding="utf-8",
    )
    (root / "diagnostics/parameter_structure_audit.json").write_text(
        json.dumps(parameter_audit), encoding="utf-8"
    )
    (root / "provenance/fixed_evaluation_masks.json").write_text(
        json.dumps(
            {
                "used_for_gradient_updates": False,
                "used_for_checkpoint_selection": False,
                "technical_replicates_not_biological_replicates": True,
            }
        ),
        encoding="utf-8",
    )
    (root / "diagnostics/training_convergence.json").write_text(
        json.dumps(
            {
                "final_epoch": 199,
                "final_train_hybrid_loss": 1.0,
                "minimum_observed_train_hybrid_loss": 0.9,
                "last_20_epoch_loss_slope": -0.001,
                "all_epochs_completed": True,
                "all_losses_and_gradients_finite": True,
                "final_components": {
                    metric: 1.0 for metric in _MODULE._COMPONENT_METRICS
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "diagnostics/resource_usage.json").write_text(
        json.dumps(
            {
                "diagnostic_resource_pilot": False,
                "public_variant": arm,
                "biological_unit_alias": alias,
                "finite_losses_and_gradients": True,
                "epochs_completed": 200,
                "parameter_count": _MODULE._EXPECTED_PARAMETER_COUNT,
                "total_pilot_or_run_duration_seconds": 123.0,
                "training_duration_seconds": 100.0,
                "evaluation_duration_seconds": 10.0,
                "peak_allocated_vram_gib": 4.5,
                "device": "cuda:0",
                "cuda_device_name": "fixture GPU",
                "torch_version": "fixture",
                "torch_cuda_version": "fixture",
                "effective_training_amp": True,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        _MODULE, "verify_run_bundle", lambda path: {"status": "success"}
    )
    monkeypatch.setattr(
        _MODULE,
        "build_checkpoint_catalog",
        lambda registry, paths, run_ids: [
            {
                "checkpoint_role": "last",
                "best_epoch": 199,
                "verification_status": "verified",
                "checkpoint_sha256": "c" * 64,
                "content_duplicate_count": 1,
            }
        ],
    )
    monkeypatch.setattr(
        _MODULE, "verify_checkpoint_record", lambda checkpoint, paths: None
    )
    evidence = _MODULE._load_run(
        object(),
        type("Paths", (), {"project_root": tmp_path})(),
        {
            "core_alias": alias,
            "arm": arm,
            "run_id": run_id,
            "artifact_path": root,
        },
    )
    assert evidence.parameter_count == _MODULE._EXPECTED_PARAMETER_COUNT
    assert evidence.peak_vram_gib == pytest.approx(4.5)
    assert len(evidence.per_mask_rows) == 9
    assert len(evidence.core_mode_rows) == 3
    resource_rows = _MODULE._run_resource_rows({(alias, arm): evidence})
    assert resource_rows[0]["training_duration_seconds"] == pytest.approx(100.0)
    assert resource_rows[0]["all_epochs_completed"] is True

    invalid_audit = dict(parameter_audit)
    invalid_audit["encoder_initial_state_bit_identical"] = False
    (root / "diagnostics/parameter_structure_audit.json").write_text(
        json.dumps(invalid_audit), encoding="utf-8"
    )
    training_payload = json.loads(
        (root / "provenance/full_core_training.json").read_text(
            encoding="utf-8"
        )
    )
    training_payload["parameter_structure_audit"] = invalid_audit
    (root / "provenance/full_core_training.json").write_text(
        json.dumps(training_payload), encoding="utf-8"
    )
    with pytest.raises(HybridCountComparisonError, match="parameter structure"):
        _MODULE._load_run(
            object(),
            type("Paths", (), {"project_root": tmp_path})(),
            {
                "core_alias": alias,
                "arm": arm,
                "run_id": run_id,
                "artifact_path": root,
            },
        )
