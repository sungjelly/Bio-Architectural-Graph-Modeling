#!/usr/bin/env python3
"""Run the registered Stage-0 multiscale synthetic-recovery diagnostic.

The queue worker owns archive creation, registry transitions, hardware/code
provenance, publication, and checkpoint indexing.  This subprocess attaches to
that exact active archive and writes only identifier-free scientific outputs.
It never enqueues work or finalizes a run itself.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Optional

import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT))
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.multiscale_synthetic import (  # noqa: E402
    ARM_TRUE_LOCAL,
    AliasSafeObservedGeometry,
    MultiscaleSyntheticError,
    MultiscaleSyntheticRecoveryResult,
    SYNTHETIC_ARMS,
    SYNTHETIC_GEOMETRY_ALIAS,
    SYNTHETIC_PRIMARY_METRIC,
    SYNTHETIC_PROTOCOL,
    SYNTHETIC_SCHEMA,
    SYNTHETIC_TASK_FAMILY,
    SyntheticRecoveryConfig,
    assert_alias_safe_configuration,
    load_alias_safe_observed_geometry,
    run_multiscale_synthetic_recovery,
)
from spatial_benchmark.multiscale_hurdle_contract import (  # noqa: E402
    ACTIVE_CONTRACT_AMENDMENT_SHA256,
    REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.run_archive import RunArchive  # noqa: E402


CAMPAIGN_ID = "cmp_20260729_multiscale_hurdle_count_pilot"
RECEIPT_SCHEMA = "multiscale_synthetic_recovery_gate_receipt_v2"
PRERUN_NEGATIVE_DIAGNOSTIC_REFERENCE = (
    "experiments/campaigns/"
    f"{CAMPAIGN_ID}/stage0_prerun_negative_diagnostic_v1.json"
)
PRERUN_NEGATIVE_DIAGNOSTIC_FILE_SHA256 = (
    "87d21ba39de9972d787ebec185d4de433743bfc623349f57287275060bc22c72"
)
PRERUN_NEGATIVE_DIAGNOSTIC_PAYLOAD_CHECKSUM = (
    "4b06b1d3d08afe560ed727cb1e0da53254b9f358626fda067b2de6cc224dade3"
)


@dataclass(frozen=True)
class SyntheticArchiveRunResult:
    run_id: str
    primary_metric_name: str
    primary_metric_value: float
    gate_passed: bool
    checkpoint_path: Path
    prediction_path: Path
    receipt_path: Path
    summary: Mapping[str, Any]


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise MultiscaleSyntheticError(f"config.{name} must be a mapping")
    return dict(value)


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise MultiscaleSyntheticError(
            f"{field} must be {expected!r}; got {actual!r}"
        )


def _expected_prepared_sha(dataset: Mapping[str, Any]) -> str:
    direct = dataset.get("prepared_data_sha256")
    basis = dataset.get("dataset_fingerprint_basis")
    nested = (
        basis.get("source_prepared_data_sha256")
        if isinstance(basis, Mapping)
        else None
    )
    value = direct if direct is not None else nested
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MultiscaleSyntheticError(
            "dataset must bind the prepared-data SHA-256"
        )
    if direct is not None and nested is not None and direct != nested:
        raise MultiscaleSyntheticError(
            "dataset prepared-data checksum declarations disagree"
        )
    return value


def _synthetic_settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = config.get("metadata", {})
    experiment = config.get("experiment", {})
    metadata_settings = (
        metadata.get("synthetic_recovery")
        if isinstance(metadata, Mapping)
        else None
    )
    experiment_settings = (
        experiment.get("synthetic_recovery")
        if isinstance(experiment, Mapping)
        else None
    )
    if metadata_settings is not None and experiment_settings is not None:
        if canonical_sha256(metadata_settings) != canonical_sha256(
            experiment_settings
        ):
            raise MultiscaleSyntheticError(
                "metadata and experiment synthetic settings disagree"
            )
    selected = (
        metadata_settings
        if metadata_settings is not None
        else experiment_settings
    )
    if selected is None:
        return {}
    if not isinstance(selected, Mapping):
        raise MultiscaleSyntheticError(
            "synthetic_recovery settings must be a mapping"
        )
    return dict(selected)


def _expected_fixture_preflight(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = _section(config, "metadata")
    _require_equal(
        metadata.get("execution_role"),
        "stage0_synthetic_recovery",
        field="metadata.execution_role",
    )
    _require_equal(
        metadata.get("no_post_outcome_tuning"),
        True,
        field="metadata.no_post_outcome_tuning",
    )
    prerun = metadata.get("pre_run_negative_diagnostic")
    expected_prerun = {
        "reference": PRERUN_NEGATIVE_DIAGNOSTIC_REFERENCE,
        "file_sha256": PRERUN_NEGATIVE_DIAGNOSTIC_FILE_SHA256,
        "payload_checksum": PRERUN_NEGATIVE_DIAGNOSTIC_PAYLOAD_CHECKSUM,
        "registered_stage0_result_inferred": False,
    }
    if not isinstance(prerun, Mapping) or dict(prerun) != expected_prerun:
        raise MultiscaleSyntheticError(
            "metadata must bind the checksum-verified negative pre-run "
            "diagnostic without inferring the registered outcome"
        )
    preflight = metadata.get("expected_fixture_preflight")
    if not isinstance(preflight, Mapping):
        raise MultiscaleSyntheticError(
            "metadata.expected_fixture_preflight must be a mapping"
        )
    expected_fields = {
        "selected_node_count",
        "full_node_count",
        "prepared_manifest_sha256",
        "selection_index_sha256",
        "geometry_sha256",
        "macroblock_ids_sha256",
        "macroblock_count",
        "true_graph_bundle_sha256",
        "true_local_graph_sha256",
        "true_regional_graph_sha256",
        "sender_state_permutation_receipt_checksum",
        "sender_state_permutation_mapping_sha256",
        "effective_permuted_source_equals_receiver_count",
        "effective_permuted_source_equals_receiver_fraction",
        "affected_receiver_count",
        "affected_receiver_fraction",
        "fixture_checksum",
    }
    if set(preflight) != expected_fields:
        raise MultiscaleSyntheticError(
            "expected fixture preflight fields changed"
        )
    for field in (
        "prepared_manifest_sha256",
        "selection_index_sha256",
        "geometry_sha256",
        "macroblock_ids_sha256",
        "true_graph_bundle_sha256",
        "true_local_graph_sha256",
        "true_regional_graph_sha256",
        "sender_state_permutation_receipt_checksum",
        "sender_state_permutation_mapping_sha256",
        "fixture_checksum",
    ):
        value = preflight.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise MultiscaleSyntheticError(
                f"metadata.expected_fixture_preflight.{field} is not SHA-256"
            )
    return dict(preflight)


def _verify_fixture_preflight(
    *,
    expected: Mapping[str, Any],
    geometry: AliasSafeObservedGeometry,
    recovery: MultiscaleSyntheticRecoveryResult,
) -> None:
    geometry_receipt = geometry.receipt()
    bundle = recovery.fixture.graph_audit["bundle_checksums"]
    permutation = recovery.fixture.sender_state_permutation_audit
    permutation_qc = permutation["qc"]
    observed = {
        "selected_node_count": geometry_receipt["selected_node_count"],
        "full_node_count": geometry_receipt["full_node_count"],
        "prepared_manifest_sha256": geometry_receipt[
            "prepared_manifest_sha256"
        ],
        "selection_index_sha256": geometry_receipt["selection_index_sha256"],
        "geometry_sha256": geometry_receipt["geometry_sha256"],
        "macroblock_ids_sha256": geometry_receipt["macroblock_ids_sha256"],
        "macroblock_count": geometry_receipt["macroblock_count"],
        "true_graph_bundle_sha256": bundle["bundle_sha256"],
        "true_local_graph_sha256": bundle["local_graph_sha256"],
        "true_regional_graph_sha256": bundle["regional_graph_sha256"],
        "sender_state_permutation_receipt_checksum": permutation["checksum"],
        "sender_state_permutation_mapping_sha256": permutation[
            "source_index_by_node_sha256"
        ],
        "effective_permuted_source_equals_receiver_count": permutation_qc[
            "effective_permuted_source_equals_receiver_count"
        ],
        "effective_permuted_source_equals_receiver_fraction": permutation_qc[
            "effective_permuted_source_equals_receiver_fraction"
        ],
        "affected_receiver_count": permutation_qc[
            "effective_permuted_source_equals_receiver_affected_receiver_count"
        ],
        "affected_receiver_fraction": permutation_qc[
            "effective_permuted_source_equals_receiver_affected_receiver_fraction"
        ],
        "fixture_checksum": recovery.fixture.fixture_checksum,
    }
    if observed != dict(expected):
        mismatches = sorted(
            field
            for field in set(observed).union(expected)
            if observed.get(field) != expected.get(field)
        )
        raise MultiscaleSyntheticError(
            "locked Stage-0 fixture preflight changed: "
            + ", ".join(mismatches)
        )


def validate_synthetic_runner_config(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], SyntheticRecoveryConfig, str]:
    """Validate archive semantics and return dataset/settings/checksum."""

    assert_alias_safe_configuration(config)
    campaign = _section(config, "campaign")
    model = _section(config, "model")
    dataset = _section(config, "dataset")
    trainer = _section(config, "trainer")
    evaluation = _section(config, "evaluation")
    _require_equal(
        campaign.get("campaign_id"),
        CAMPAIGN_ID,
        field="campaign.campaign_id",
    )
    _require_equal(
        campaign.get("active_contract_amendment_sha256"),
        ACTIVE_CONTRACT_AMENDMENT_SHA256,
        field="campaign.active_contract_amendment_sha256",
    )
    _require_equal(
        campaign.get("required_contract_supplement_sha256"),
        REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
        field="campaign.required_contract_supplement_sha256",
    )
    _require_equal(
        model.get("name"),
        "multiscale-hurdle-count",
        field="model.name",
    )
    _require_equal(
        model.get("family"),
        "additive_multiscale_hurdle_count",
        field="model.family",
    )
    _require_equal(
        evaluation.get("task_family"),
        SYNTHETIC_TASK_FAMILY,
        field="evaluation.task_family",
    )
    _require_equal(
        evaluation.get("protocol"),
        SYNTHETIC_PROTOCOL,
        field="evaluation.protocol",
    )
    _require_equal(
        evaluation.get("canonical_prediction_split"),
        "fit",
        field="evaluation.canonical_prediction_split",
    )
    _require_equal(
        evaluation.get("primary_metric"),
        SYNTHETIC_PRIMARY_METRIC,
        field="evaluation.primary_metric",
    )
    _require_equal(
        evaluation.get("primary_direction"),
        "minimize",
        field="evaluation.primary_direction",
    )
    if trainer.get("restore_best") not in {False, None}:
        raise MultiscaleSyntheticError("trainer.restore_best must be false")
    _require_equal(
        trainer.get("primary_checkpoint_role"),
        "last",
        field="trainer.primary_checkpoint_role",
    )
    alias = dataset.get("biological_unit_alias")
    reference = dataset.get("prepared_artifact_reference")
    if not isinstance(alias, str) or not isinstance(reference, str):
        raise MultiscaleSyntheticError(
            "dataset requires opaque alias and prepared artifact reference"
        )
    _require_equal(
        alias,
        SYNTHETIC_GEOMETRY_ALIAS,
        field="dataset.biological_unit_alias",
    )
    settings = SyntheticRecoveryConfig.from_mapping(
        _synthetic_settings(config),
        seed=int(config.get("seed", 0)),
        trainer=trainer,
    )
    _expected_fixture_preflight(config)
    return dataset, settings, _expected_prepared_sha(dataset)


def _checkpoint_bytes(
    *,
    archive: RunArchive,
    recovery: MultiscaleSyntheticRecoveryResult,
) -> bytes:
    arm_states = {
        arm: dict(recovery.arms[arm].training.final_state_dict)
        for arm in SYNTHETIC_ARMS
    }
    payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "checkpoint_role": "last",
        "checkpoint_policy": "final_epoch_no_validation_selection",
        "training_protocol": SYNTHETIC_PROTOCOL,
        "task_family": SYNTHETIC_TASK_FAMILY,
        "diagnostic_schema": SYNTHETIC_SCHEMA,
        "active_contract_amendment_sha256": (
            ACTIVE_CONTRACT_AMENDMENT_SHA256
        ),
        "required_contract_supplement_sha256": (
            REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        ),
        "fixture_checksum": recovery.fixture.fixture_checksum,
        "sender_state_permutation_receipt_checksum": (
            recovery.fixture.sender_state_permutation_audit["checksum"]
        ),
        "arm_state_dicts": arm_states,
        "arm_state_dict_sha256": {
            arm: recovery.arms[arm].training.final_state_checksum
            for arm in SYNTHETIC_ARMS
        },
        "arm_final_epoch": {
            arm: recovery.arms[arm].training.final_epoch
            for arm in SYNTHETIC_ARMS
        },
        "parameter_count": recovery.arms[ARM_TRUE_LOCAL].parameter_count,
        "parameter_match_verified": recovery.parameter_match_verified,
        "monitored_metric": None,
        "selection_policy": "last_epoch_without_validation_selection",
        "gate_passed": recovery.gate.gate_passed,
        "direct_identifiers_emitted": False,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _prediction_rows(
    *,
    archive: RunArchive,
    recovery: MultiscaleSyntheticRecoveryResult,
) -> list[dict[str, Any]]:
    outcome = recovery.arms[ARM_TRUE_LOCAL].evaluation
    target_gene = recovery.fixture.target_gene_index
    target = outcome.target[:, target_gene].numpy()
    reconstructed = (
        outcome.evaluation.reconstructed_count[:, target_gene].numpy()
    )
    if target.shape != reconstructed.shape or target.ndim != 1:
        raise MultiscaleSyntheticError(
            "synthetic prediction arrays are not aligned"
        )
    rows: list[dict[str, Any]] = []
    for rank, (truth, prediction) in enumerate(
        zip(target.tolist(), reconstructed.tolist(), strict=True)
    ):
        digest = hashlib.sha256(
            (
                f"{archive.run_id}:{recovery.fixture.fixture_checksum}:"
                f"synthetic-receiver-{rank}"
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "run_id": archive.run_id,
                "sample_key": f"sk_{digest[:32]}",
                "dataset_id": "synthetic_observed_geometry_v1",
                "split": "fit",
                "graph_id": (
                    "multiscale_synthetic_"
                    f"{recovery.fixture.fixture_checksum[:16]}"
                ),
                "y_true": float(truth),
                "y_pred": float(prediction),
            }
        )
    return rows


def _history_rows(
    recovery: MultiscaleSyntheticRecoveryResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in SYNTHETIC_ARMS:
        outcome = recovery.arms[arm]
        for record in outcome.training.history_rows():
            rows.append(
                {
                    "arm": arm,
                    "regional_routing": outcome.regional_routing,
                    "local_routing": outcome.local_routing,
                    **record,
                }
            )
    if not rows:
        raise MultiscaleSyntheticError("synthetic training history is empty")
    return rows


def _receipt(
    *,
    archive: RunArchive,
    config: Mapping[str, Any],
    recovery: MultiscaleSyntheticRecoveryResult,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "receipt_kind": RECEIPT_SCHEMA,
        "run_id": archive.run_id,
        "campaign_id": CAMPAIGN_ID,
        "active_contract_amendment_sha256": (
            ACTIVE_CONTRACT_AMENDMENT_SHA256
        ),
        "required_contract_supplement_sha256": (
            REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        ),
        "pre_run_negative_diagnostic": dict(
            _section(config, "metadata")["pre_run_negative_diagnostic"]
        ),
        "no_post_outcome_tuning": True,
        "config_sha256": canonical_sha256(config),
        "fixture_checksum": recovery.fixture.fixture_checksum,
        "observed_geometry": recovery.fixture.audit["observed_geometry"],
        "pre_outcome_geometry_selection": recovery.fixture.audit[
            "pre_outcome_geometry_selection"
        ],
        "arm_final_state_sha256": {
            arm: recovery.arms[arm].training.final_state_checksum
            for arm in SYNTHETIC_ARMS
        },
        "arm_parameter_structure_sha256": {
            arm: recovery.arms[arm].parameter_structure_sha256
            for arm in SYNTHETIC_ARMS
        },
        "parameter_match_verified": recovery.parameter_match_verified,
        "sender_state_permutation_receipt_checksum": (
            recovery.fixture.sender_state_permutation_audit["checksum"]
        ),
        "sender_state_permutation_qc": (
            recovery.fixture.sender_state_permutation_audit["qc"]
        ),
        "contrast_estimand": (
            "predictive dependence on correctly aligned local sender state "
            "conditional on identical observed topology, receivers, and "
            "edge attributes"
        ),
        "planted_diagnostic": asdict(recovery.planted_diagnostic),
        "null_diagnostic": asdict(recovery.null_diagnostic),
        "gate": asdict(recovery.gate),
        "checkpoint_sha256": checkpoint_sha256,
        "registered_execution": True,
        "registration_and_finalization_owner": "queue_worker",
        "real_data_graph_interpretation_authorized": (
            recovery.gate.gate_passed
        ),
        "failed_gate_blocks": (
            []
            if recovery.gate.gate_passed
            else [
                "real_data_graph_arms",
                "real_data_graph_interpretation",
            ]
        ),
        "failed_gate_does_not_block": (
            []
            if recovery.gate.gate_passed
            else ["self_only_representation_and_resource_question"]
        ),
        "direct_identifiers_emitted": False,
    }
    payload["checksum"] = canonical_sha256(payload)
    return payload


def run_synthetic_recovery_archive(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    observed_geometry: Optional[AliasSafeObservedGeometry] = None,
    recovery_result: Optional[MultiscaleSyntheticRecoveryResult] = None,
) -> SyntheticArchiveRunResult:
    """Execute or archive one Stage-0 result in a worker-owned RunArchive."""

    started = time.monotonic()
    dataset, settings, expected_sha = validate_synthetic_runner_config(config)
    expected_preflight = _expected_fixture_preflight(config)
    geometry = (
        load_alias_safe_observed_geometry(
            str(dataset["prepared_artifact_reference"]),
            project_root=archive.paths.project_root,
            biological_unit_alias=str(dataset["biological_unit_alias"]),
            expected_prepared_data_sha256=expected_sha,
            selected_node_count=settings.selected_node_count,
        )
        if observed_geometry is None
        else observed_geometry
    )
    if (
        geometry.biological_unit_alias != dataset["biological_unit_alias"]
        or geometry.prepared_data_sha256 != expected_sha
        or geometry.selected_node_count != settings.selected_node_count
    ):
        raise MultiscaleSyntheticError(
            "injected observed geometry violates the frozen config"
        )
    recovery = (
        run_multiscale_synthetic_recovery(geometry, settings)
        if recovery_result is None
        else recovery_result
    )
    if (
        recovery.config != settings
        or recovery.fixture.audit["observed_geometry"] != geometry.receipt()
    ):
        raise MultiscaleSyntheticError(
            "recovery result is not bound to the configured geometry/settings"
        )
    _verify_fixture_preflight(
        expected=expected_preflight,
        geometry=geometry,
        recovery=recovery,
    )

    archive.write_json(
        "provenance/synthetic_inputs.json",
        {
            "schema": SYNTHETIC_SCHEMA,
            "configuration_sha256": canonical_sha256(config),
            "active_contract_amendment_sha256": (
                ACTIVE_CONTRACT_AMENDMENT_SHA256
            ),
            "required_contract_supplement_sha256": (
                REQUIRED_CONTRACT_SUPPLEMENT_SHA256
            ),
            "pre_run_negative_diagnostic": dict(
                _section(config, "metadata")["pre_run_negative_diagnostic"]
            ),
            "no_post_outcome_tuning": True,
            "observed_geometry": geometry.receipt(),
            "pre_outcome_geometry_selection": recovery.fixture.audit[
                "pre_outcome_geometry_selection"
            ],
            "expression_source": "generated_de_novo_no_observed_expression_used",
            "sender_state_null_construction_inputs": [
                "coordinates_um",
                "macroblock_ids",
                "stable_node_row_index",
            ],
            "target_expression_used_for_sender_state_null": False,
            "null_injection": (
                "marginal_preserving_target_permutation_correlation_bounded"
            ),
            "fit_scope": "single_synthetic_observed_shape_transductive",
            "generalization_estimate": False,
            "direct_identifiers_emitted": False,
        },
    )
    archive.write_json(
        "diagnostics/synthetic_fixture.json",
        dict(recovery.fixture.audit),
    )
    archive.write_json(
        "diagnostics/multiscale_graphs.json",
        dict(recovery.fixture.graph_audit),
    )
    archive.write_json(
        "diagnostics/sender_state_permutation.json",
        dict(recovery.fixture.sender_state_permutation_audit),
    )
    archive.write_json(
        "diagnostics/planted_edge_deletion.json",
        asdict(recovery.planted_diagnostic),
    )
    archive.write_json(
        "diagnostics/null_edge_deletion.json",
        asdict(recovery.null_diagnostic),
    )
    archive.write_table(
        "metrics/history",
        _history_rows(recovery),
        fallback="jsonl",
    )
    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt",
        _checkpoint_bytes(archive=archive, recovery=recovery),
    )
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    prediction_path = archive.write_predictions(
        "fit",
        _prediction_rows(archive=archive, recovery=recovery),
        fallback="jsonl",
    )
    final_metrics = recovery.final_metrics()
    for name, value in final_metrics.items():
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"final metric {name} is non-finite")
        archive.append_metric_event(
            {
                "name": name,
                "value": value,
                "phase": "synthetic_recovery_gate",
            }
        )
    archive.write_json("metrics/final.json", final_metrics)
    receipt = _receipt(
        archive=archive,
        config=config,
        recovery=recovery,
        checkpoint_sha256=checkpoint_sha,
    )
    receipt_path = archive.write_json(
        "diagnostics/synthetic_recovery_receipt.json",
        receipt,
    )
    peak_vram_bytes = max(
        record.peak_cuda_memory_bytes
        for outcome in recovery.arms.values()
        for record in outcome.training.history
    )
    duration = time.monotonic() - started
    parameter_count = recovery.arms[ARM_TRUE_LOCAL].parameter_count
    archive.write_json(
        "diagnostics/resource_usage.json",
        {
            "duration_seconds": duration,
            "peak_allocated_vram_bytes": peak_vram_bytes,
            "peak_allocated_vram_gib": peak_vram_bytes / (1024**3),
            "parameter_count_per_arm": parameter_count,
            "arm_count": len(recovery.arms),
            "fixed_epoch_budget": settings.max_epochs,
            "device": recovery.arms[ARM_TRUE_LOCAL].training.device,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        },
    )
    primary_value = float(final_metrics[SYNTHETIC_PRIMARY_METRIC])
    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "training_exit_status": "success",
        "campaign_id": CAMPAIGN_ID,
        "active_contract_amendment_sha256": (
            ACTIVE_CONTRACT_AMENDMENT_SHA256
        ),
        "required_contract_supplement_sha256": (
            REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        ),
        "pre_run_negative_diagnostic": dict(
            _section(config, "metadata")["pre_run_negative_diagnostic"]
        ),
        "no_post_outcome_tuning": True,
        "evaluation_protocol": SYNTHETIC_PROTOCOL,
        "task_family": SYNTHETIC_TASK_FAMILY,
        "canonical_prediction_split": "fit",
        "primary_metric_name": SYNTHETIC_PRIMARY_METRIC,
        "primary_metric_value": primary_value,
        "model_name": "multiscale-hurdle-count-synthetic-recovery",
        "final_epoch": settings.max_epochs - 1,
        "fixed_epoch_budget": settings.max_epochs,
        "checkpoint_role": "last",
        "checkpoint": {
            "role": "last",
            "policy": "final_epoch_no_validation_selection",
            "monitored_metric": None,
            "sha256": checkpoint_sha,
        },
        "parameter_count": parameter_count,
        "parameter_match_verified": recovery.parameter_match_verified,
        "duration_seconds": duration,
        "peak_vram_gib": peak_vram_bytes / (1024**3),
        "synthetic_recovery_gate_passed": recovery.gate.gate_passed,
        "sender_state_permutation_receipt_checksum": (
            recovery.fixture.sender_state_permutation_audit["checksum"]
        ),
        "sender_state_permutation_qc_passed": (
            recovery.gate.sender_state_permutation_qc_passed
        ),
        "pre_outcome_geometry_selection": recovery.fixture.audit[
            "pre_outcome_geometry_selection"
        ],
        "synthetic_recovery_failure_reasons": list(
            recovery.gate.failure_reasons
        ),
        "real_data_graph_interpretation_authorized": (
            recovery.gate.gate_passed
        ),
        "failed_gate_blocks": (
            []
            if recovery.gate.gate_passed
            else [
                "real_data_graph_arms",
                "real_data_graph_interpretation",
            ]
        ),
        "failed_gate_does_not_block": (
            []
            if recovery.gate.gate_passed
            else ["self_only_representation_and_resource_question"]
        ),
        "receipt_checksum": receipt["checksum"],
        "generalization_estimate": False,
        "diagnostic_only": True,
        "maximum_claim": (
            "synthetic recovery of correctly aligned local sender-state "
            "dependence conditional on identical topology"
        ),
        "prohibited_claims": [
            "real-cell predictive gain",
            "patient-held-out generalization",
            "sender-state contrast is a topology contrast",
            "biological mechanism",
            "causality",
        ],
        "direct_identifiers_emitted": False,
    }
    archive.write_summary(summary)
    return SyntheticArchiveRunResult(
        run_id=archive.run_id,
        primary_metric_name=SYNTHETIC_PRIMARY_METRIC,
        primary_metric_value=primary_value,
        gate_passed=recovery.gate.gate_passed,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        receipt_path=receipt_path,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one queue-worker-owned multiscale synthetic recovery diagnostic."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def _worker_archive_and_config(
    args: argparse.Namespace,
) -> tuple[RunArchive, dict[str, Any]]:
    run_id = os.environ.get("BAGM_RUN_ID", "").strip()
    environment_scratch = os.environ.get("BAGM_RUN_SCRATCH", "").strip()
    if not run_id or not environment_scratch:
        raise MultiscaleSyntheticError(
            "BAGM_RUN_ID and BAGM_RUN_SCRATCH are required; this diagnostic "
            "must execute under the queue worker"
        )
    supplied_scratch = args.run_scratch.resolve(strict=False)
    if supplied_scratch != Path(environment_scratch).resolve(strict=False):
        raise MultiscaleSyntheticError(
            "--run-scratch does not match BAGM_RUN_SCRATCH"
        )
    expected_config = supplied_scratch / "config.resolved.yaml"
    if args.config.resolve(strict=False) != expected_config.resolve(
        strict=False
    ):
        raise MultiscaleSyntheticError(
            "--config must be the worker-owned resolved configuration"
        )
    environment_config = os.environ.get("BAGM_CONFIG_PATH")
    if environment_config and Path(environment_config).resolve(
        strict=False
    ) != expected_config.resolve(strict=False):
        raise MultiscaleSyntheticError(
            "BAGM_CONFIG_PATH does not match the resolved configuration"
        )
    archive = RunArchive.attach_active(
        run_id,
        paths=current_paths(),
        scratch_path=supplied_scratch,
    )
    configuration = load_yaml_mapping(expected_config)
    validate_experiment_config(configuration)
    validate_synthetic_runner_config(configuration)
    return archive, configuration


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    result = run_synthetic_recovery_archive(config, archive)
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "primary_metric_name": result.primary_metric_name,
                "primary_metric_value": result.primary_metric_value,
                "gate_passed": result.gate_passed,
                "checkpoint": str(result.checkpoint_path),
                "predictions": str(result.prediction_path),
                "receipt": str(result.receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
