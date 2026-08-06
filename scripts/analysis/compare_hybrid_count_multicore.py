#!/usr/bin/env python3
"""Audit and compare the frozen ten-core hybrid-count campaign.

The registry and each run's resolved configuration are authoritative for
coverage.  Fixed mask replicates are averaged within one arm/mode/core before
the ten opaque core aliases are given equal weight.  No cell- or mask-level
quantity is treated as an independent biological replicate.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import yaml


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.checkpoint_catalog import (  # noqa: E402
    build_checkpoint_catalog,
    verify_checkpoint_record,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paired_inference import (  # noqa: E402
    exact_one_sided_paired_sign_flip,
    holm_adjust,
    paired_relative_improvements,
)
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import (  # noqa: E402
    RunValidationError,
    verify_run_bundle,
)


_CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
_CORE_ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
_ARMS = ("hybrid-gat-k1000", "hybrid-matched-self")
_MODEL_TO_ARM = {
    "hybrid-count-gat": "hybrid-gat-k1000",
    "hybrid-count-matched-self": "hybrid-matched-self",
}
_ARM_TO_VARIANT_TOKEN = {
    "hybrid-gat-k1000": "hybrid_gat_k1000",
    "hybrid-matched-self": "hybrid_matched_self",
}
_MASK_MODES = ("partial_gene", "whole_node", "spatial_block")
_EXPECTED_MASK_REPLICATES = 3
_EXPECTED_EPOCHS = 200
_EXPECTED_SEED = 0
_EXPECTED_FOLD = 0
_EXPECTED_ATTEMPTS = frozenset({1, 2})
_EXPECTED_PARAMETER_COUNT = 11_674_880
_PILOT_MAX_VRAM_GIB = 20.5
_PILOT_MAX_AMP_DISCREPANCY = 1e-3
_PILOT_MAX_GAT_RUNTIME_HOURS = 6.0
_PRIMARY_METRIC = "fit/whole_node/hybrid_loss"
_PROTOCOL = "held_in_full_core_fixed_budget"
_TRAINING_PROTOCOL = "held_in_full_core_hybrid_count_fixed_budget"
_GRAPH_EXECUTION = "full_core_exact_no_neighbor_sampling"
_CHECKPOINT_POLICY = "final_epoch_no_validation_selection"
_REPRESENTATION_SCHEMA = (
    "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1"
)
_PRIOR_CATEGORICAL_CAMPAIGN = (
    "cmp_20260726_full_core_g2_count_tokens_multiseed"
)
_PRIOR_SAFE_VARIANTS = {
    "g2_tokenized_width512_exact_k1000_full_core": (
        "prior-categorical-gat-width512"
    ),
    "g2_tokenized_width1024_exact_k1000_full_core": (
        "prior-categorical-gat-width1024"
    ),
}
_PRIOR_CATEGORICAL_DEFAULT = (
    _BOOTSTRAP_ROOT
    / "reports/analyses/full_core_g2_count_tokens_multiseed/"
    "comparison/comparison.json"
)
_LOCKED_CAMPAIGN_RELATIVE = Path(
    "locked_campaigns/cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
)
_MATERIALIZATION_KIND = "hybrid_count_locked_config_materialization_v1"
_PILOT_GATE_KIND = "hybrid_count_pilot_gate_v1"
_COMPONENT_METRICS = (
    "detection_bce",
    "ordinal_bce",
    "positive_continuous_huber",
)
_REQUIRED_SCALAR_METRICS = (
    "hybrid_loss",
    "detection_bce",
    "ordinal_bce",
    "positive_continuous_huber",
    "detection_balanced_accuracy",
    "detection_sensitivity",
    "detection_specificity",
    "detection_precision",
    "detection_positive_support",
    "detection_zero_support",
    "detection_predicted_positive",
    "state8_exact_accuracy",
    "state8_balanced_accuracy",
    "positive_state_exact_accuracy",
    "positive_ordinal_mae",
    "positive_within_one_state_accuracy",
    "positive_continuous_mae",
    "reconstructed_count_log1p_mae",
    "collapsed4_exact_accuracy",
    "collapsed4_balanced_accuracy",
    "collapsed4_positive_exact_accuracy",
    "reference_per_gene_hybrid_loss",
    "reference_per_gene_detection_bce",
    "reference_per_gene_ordinal_bce",
    "reference_per_gene_positive_continuous_huber",
    "reference_per_gene_detection_balanced_accuracy",
    "reference_per_gene_positive_ordinal_mae",
    "reference_per_gene_positive_continuous_mae",
    "reference_per_gene_state8_exact_accuracy",
    "reference_per_gene_state8_balanced_accuracy",
    "reference_all_zero_state8_exact_accuracy",
    "reference_all_zero_state8_balanced_accuracy",
    "reference_all_zero_collapsed4_exact_accuracy",
)
_NULLABLE_SCALAR_METRICS = frozenset({"detection_precision"})
_VECTOR_METRICS = {
    "state8_support": 8,
    "state8_recall": 8,
    "collapsed4_support": 4,
    "collapsed4_recall": 4,
    "reference_all_zero_state8_support": 8,
    "reference_all_zero_state8_recall": 8,
}
_NODE_METADATA_FIELDS = (
    "Area",
    "Area.um2",
    "AspectRatio",
    "Width",
    "Height",
    "Mean.PanCK",
    "Max.PanCK",
    "Mean.G",
    "Max.G",
    "Mean.Membrane",
    "Max.Membrane",
    "Mean.CD45",
    "Max.CD45",
    "Mean.DAPI",
    "Max.DAPI",
    "SplitRatioToLocal",
    "NucArea",
    "NucAspectRatio",
    "Circularity",
    "Eccentricity",
    "Perimeter",
    "Solidity",
)
_EDGE_FIELDS = (
    "distance_um",
    "distance_over_radius",
    "log1p_distance_um",
    "delta_x_over_radius",
    "delta_y_over_radius",
    "cos_theta",
    "sin_theta",
    "cos_2theta",
    "sin_2theta",
    "distance_rbf_0",
    "distance_rbf_1",
    "distance_rbf_2",
    "distance_rbf_3",
    "distance_rbf_4",
    "distance_rbf_5",
    "distance_rbf_6",
    "distance_rbf_7",
)
_TABLE_SUFFIXES = (".parquet", ".jsonl", ".csv")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class HybridCountComparisonError(RuntimeError):
    """Raised when registry or artifact structure is not auditable."""


@dataclass(frozen=True)
class ConfigSemantics:
    alias: str | None
    arm: str | None
    production_like: bool
    valid_production: bool
    errors: tuple[str, ...]


@dataclass(frozen=True)
class RunEvidence:
    alias: str
    arm: str
    run_id: str
    root: Path
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    training: Mapping[str, Any]
    parameter_audit: Mapping[str, Any]
    convergence: Mapping[str, Any]
    resource_usage: Mapping[str, Any]
    fixed_masks: Mapping[str, Any]
    per_mask_rows: tuple[Mapping[str, Any], ...]
    core_mode_rows: tuple[Mapping[str, Any], ...]
    parameter_count: int
    graph_sha256: str
    checkpoint_sha256: str
    checkpoint_duplicate_count: int
    duration_seconds: float
    peak_vram_gib: float


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HybridCountComparisonError(f"{label} must be a mapping")
    return value


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise HybridCountComparisonError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HybridCountComparisonError(
            f"{label} must be an integer"
        ) from error
    if isinstance(value, float) and not value.is_integer():
        raise HybridCountComparisonError(f"{label} must be an integer")
    return result


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise HybridCountComparisonError(f"{label} must be finite numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise HybridCountComparisonError(
            f"{label} must be finite numeric"
        ) from error
    if not math.isfinite(result):
        raise HybridCountComparisonError(f"{label} must be finite numeric")
    return result


def _nullable_float(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    return _finite_float(value, label)


def _sha256(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise HybridCountComparisonError(f"{label} must be a lowercase SHA-256")
    return text


def _json_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError) as error:
        raise HybridCountComparisonError(f"{label} is not valid JSON") from error
    return _mapping(decoded, label)


def _load_strict_json(path: Path, *, label: str) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise HybridCountComparisonError(
            f"{label} contains a non-finite JSON constant"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise HybridCountComparisonError(
                    f"{label} contains duplicate JSON keys"
                )
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except HybridCountComparisonError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HybridCountComparisonError(f"{label} is not strict JSON") from error
    return _mapping(value, label)


def _verified_receipt_checksum(
    value: Mapping[str, Any], *, label: str
) -> str:
    checksum = _sha256(value.get("checksum"), f"{label} checksum")
    payload = dict(value)
    payload.pop("checksum", None)
    if canonical_sha256(payload) != checksum:
        raise HybridCountComparisonError(f"{label} checksum does not verify")
    return checksum


def validate_pilot_gate_receipts(
    materialization: Mapping[str, Any],
    pilot_gate: Mapping[str, Any],
    *,
    frozen_contract_sha256: str,
) -> dict[str, Any]:
    """Validate the checksum-bound gate that authorized production."""

    materialization_checksum = _verified_receipt_checksum(
        materialization, label="locked materialization"
    )
    gate_checksum = _verified_receipt_checksum(
        pilot_gate, label="pilot gate receipt"
    )
    counts = _mapping(materialization.get("counts"), "materialization counts")
    frozen = _mapping(
        materialization.get("frozen_contract"),
        "materialization frozen contract",
    )
    if (
        materialization.get("schema_version") != 1
        or materialization.get("receipt_kind") != _MATERIALIZATION_KIND
        or materialization.get("campaign_id") != _CAMPAIGN_ID
        or counts.get("aliases") != 10
        or counts.get("pilot_configs") != 2
        or counts.get("production_configs") != 20
        or frozen.get("sha256") != frozen_contract_sha256
        or materialization.get("parameter_count") != _EXPECTED_PARAMETER_COUNT
        or materialization.get("registry_mutation_performed") is not False
        or materialization.get("queue_mutation_performed") is not False
        or materialization.get("training_performed") is not False
    ):
        raise HybridCountComparisonError(
            "locked materialization does not match the frozen campaign"
        )
    jobs = pilot_gate.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2 or not all(
        isinstance(job, Mapping) for job in jobs
    ):
        raise HybridCountComparisonError(
            "pilot gate must contain exactly two job mappings"
        )
    observed = {
        (str(job.get("alias")), str(job.get("arm"))) for job in jobs
    }
    thresholds = pilot_gate.get("thresholds")
    expected_thresholds = {
        "peak_allocated_vram_gib_maximum": _PILOT_MAX_VRAM_GIB,
        "fp32_amp_absolute_total_loss_discrepancy_maximum": (
            _PILOT_MAX_AMP_DISCREPANCY
        ),
        "projected_gat_runtime_hours_per_core_maximum": (
            _PILOT_MAX_GAT_RUNTIME_HOURS
        ),
    }
    if (
        pilot_gate.get("schema_version") != 1
        or pilot_gate.get("receipt_kind") != _PILOT_GATE_KIND
        or pilot_gate.get("campaign_id") != _CAMPAIGN_ID
        or pilot_gate.get("materialization_checksum") != materialization_checksum
        or pilot_gate.get("frozen_contract_sha256") != frozen_contract_sha256
        or pilot_gate.get("gate_passed") is not True
        or pilot_gate.get("production_authorized") is not True
        or pilot_gate.get("same_frozen_precision_batch") is not True
        or pilot_gate.get("same_evaluation_masks") is not True
        or pilot_gate.get("same_verified_graph") is not True
        or pilot_gate.get("failure_reasons") != []
        or thresholds != expected_thresholds
        or observed
        != {
            ("ANC-01", "hybrid-gat-k1000"),
            ("ANC-01", "hybrid-matched-self"),
        }
    ):
        raise HybridCountComparisonError(
            "production lacks the exact passing two-arm pilot gate"
        )
    pilot_enqueue_checksum = _sha256(
        pilot_gate.get("pilot_enqueue_receipt_checksum"),
        "pilot enqueue receipt checksum",
    )
    audited_jobs: list[dict[str, Any]] = []
    for job in jobs:
        arm = str(job["arm"])
        discrepancy = _finite_float(
            job.get("fp32_amp_absolute_total_loss_discrepancy"),
            "pilot FP32/AMP discrepancy",
        )
        peak_vram = _finite_float(
            job.get("peak_allocated_vram_gib"), "pilot peak VRAM"
        )
        projected_runtime = _finite_float(
            job.get("projected_200_epoch_runtime_hours"),
            "pilot projected runtime",
        )
        precision_mask = _sha256(
            job.get("precision_mask_checksum"), "pilot precision mask"
        )
        evaluation_masks = _sha256(
            job.get("evaluation_mask_bundle_sha256"),
            "pilot evaluation masks",
        )
        graph_sha = _sha256(job.get("graph_sha256"), "pilot graph")
        checkpoint_sha = _sha256(
            job.get("checkpoint_sha256"), "pilot checkpoint"
        )
        if (
            job.get("verified_bundle") is not True
            or job.get("finite_losses_and_gradients") is not True
            or job.get("parameter_match") is not True
            or job.get("parameter_count") != _EXPECTED_PARAMETER_COUNT
            or job.get("precision_equivalence_passed") is not True
            or job.get("peak_vram_passed") is not True
            or job.get("projected_runtime_passed") is not True
            or job.get("runner_pilot_gate_passed") is not True
            or job.get("checkpoint_verified") is not True
            or job.get("checkpoint_role") != "last"
            or job.get("checkpoint_epoch") != 1
            or discrepancy < 0
            or discrepancy > _PILOT_MAX_AMP_DISCREPANCY
            or peak_vram < 0
            or peak_vram > _PILOT_MAX_VRAM_GIB
            or projected_runtime < 0
            or (
                arm == "hybrid-gat-k1000"
                and projected_runtime > _PILOT_MAX_GAT_RUNTIME_HOURS
            )
        ):
            raise HybridCountComparisonError(
                "production lacks the exact passing two-arm pilot gate"
            )
        audited_jobs.append(
            {
                "alias": str(job["alias"]),
                "arm": arm,
                "verified_bundle": True,
                "checkpoint_verified": True,
                "checkpoint_sha256": checkpoint_sha,
                "finite_losses_and_gradients": True,
                "parameter_match": True,
                "parameter_count": _EXPECTED_PARAMETER_COUNT,
                "precision_mask_checksum": precision_mask,
                "fp32_amp_absolute_total_loss_discrepancy": discrepancy,
                "precision_equivalence_passed": True,
                "peak_allocated_vram_gib": peak_vram,
                "peak_vram_passed": True,
                "projected_200_epoch_runtime_hours": projected_runtime,
                "projected_runtime_passed": True,
                "runner_pilot_gate_passed": True,
                "evaluation_mask_bundle_sha256": evaluation_masks,
                "graph_sha256": graph_sha,
            }
        )
    if (
        len({job["precision_mask_checksum"] for job in audited_jobs}) != 1
        or len(
            {job["evaluation_mask_bundle_sha256"] for job in audited_jobs}
        )
        != 1
        or len({job["graph_sha256"] for job in audited_jobs}) != 1
    ):
        raise HybridCountComparisonError(
            "production lacks the exact passing two-arm pilot gate"
        )
    return {
        "verified": True,
        "materialization_checksum": materialization_checksum,
        "pilot_gate_checksum": gate_checksum,
        "pilot_enqueue_receipt_checksum": pilot_enqueue_checksum,
        "frozen_contract_sha256": frozen_contract_sha256,
        "gate_passed": True,
        "production_authorized": True,
        "thresholds": expected_thresholds,
        "same_frozen_precision_batch": True,
        "same_evaluation_masks": True,
        "same_verified_graph": True,
        "jobs": sorted(
            audited_jobs,
            key=lambda item: (str(item["alias"]), str(item["arm"])),
        ),
    }


def _load_campaign_receipts(paths: ProjectPaths) -> dict[str, Any]:
    locked = paths.scratch_root / _LOCKED_CAMPAIGN_RELATIVE
    materialization_path = locked / "locked_config_materialization.json"
    pilot_gate_path = locked / "pilot_gate_receipt.json"
    contract_path = (
        paths.project_root
        / "experiments/campaigns"
        / _CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    if not materialization_path.is_file() or not pilot_gate_path.is_file():
        return {
            "verified": False,
            "reason": "materialization_or_pilot_gate_receipt_missing",
        }
    if not contract_path.is_file() or contract_path.is_symlink():
        return {"verified": False, "reason": "frozen_contract_missing"}
    contract_sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    try:
        return validate_pilot_gate_receipts(
            _load_strict_json(
                materialization_path, label="locked materialization"
            ),
            _load_strict_json(pilot_gate_path, label="pilot gate receipt"),
            frozen_contract_sha256=contract_sha,
        )
    except HybridCountComparisonError as error:
        return {
            "verified": False,
            "reason": "pilot_gate_receipt_verification_failed",
            "error_type": type(error).__name__,
        }


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HybridCountComparisonError(
            f"required YAML artifact is unreadable: {path.name}"
        ) from error
    return _mapping(value, path.name)


def _logical_table_path(root: Path, stem: str) -> Path:
    matches = [
        root / f"{stem}{suffix}"
        for suffix in _TABLE_SUFFIXES
        if (root / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise HybridCountComparisonError(
            f"{root.name} requires exactly one {stem} table"
        )
    return matches[0]


def _load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.strip():
                        rows.append(
                            dict(
                                _mapping(
                                    json.loads(line),
                                    f"{path.name}:{line_number}",
                                )
                            )
                        )
        except (OSError, ValueError) as error:
            raise HybridCountComparisonError(
                f"required JSONL table is unreadable: {path.name}"
            ) from error
    elif path.suffix == ".csv":
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except (OSError, csv.Error) as error:
            raise HybridCountComparisonError(
                f"required CSV table is unreadable: {path.name}"
            ) from error
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except (ImportError, ModuleNotFoundError) as error:
            raise HybridCountComparisonError(
                f"reading Parquet requires pyarrow: {path.name}"
            ) from error
        try:
            rows = [
                dict(_mapping(row, f"row in {path.name}"))
                for row in parquet.read_table(path).to_pylist()
            ]
        except Exception as error:
            raise HybridCountComparisonError(
                f"required Parquet table is unreadable: {path.name}"
            ) from error
    else:
        raise HybridCountComparisonError(f"unsupported table: {path.name}")
    if not rows:
        raise HybridCountComparisonError(f"required table is empty: {path.name}")
    return rows


def _parse_vector(value: Any, *, length: int, label: str) -> list[float | None]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError as error:
            raise HybridCountComparisonError(
                f"{label} must be a JSON numeric vector"
            ) from error
    if not isinstance(parsed, (list, tuple)) or len(parsed) != length:
        raise HybridCountComparisonError(
            f"{label} must contain exactly {length} states"
        )
    return [
        _nullable_float(item, f"{label}[{index}]")
        for index, item in enumerate(parsed)
    ]


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    return value if isinstance(value, Mapping) else {}


def _safe_alias(config: Mapping[str, Any]) -> str | None:
    dataset_alias = _section(config, "dataset").get("biological_unit_alias")
    experiment_alias = _section(config, "experiment").get(
        "biological_unit_alias"
    )
    if dataset_alias == experiment_alias and dataset_alias in _CORE_ALIASES:
        return str(dataset_alias)
    return None


def _safe_arm(config: Mapping[str, Any]) -> str | None:
    model_name = str(_section(config, "model").get("name", ""))
    model_arm = _MODEL_TO_ARM.get(model_name)
    declared = _section(config, "experiment").get("arm")
    if model_arm is not None and declared == model_arm:
        return model_arm
    return None


def _config_semantics(config: Mapping[str, Any]) -> ConfigSemantics:
    """Classify one config without ever exporting an unrecognized sample label."""

    campaign = _section(config, "campaign")
    dataset = _section(config, "dataset")
    experiment = _section(config, "experiment")
    model = _section(config, "model")
    features = _section(config, "features")
    graph = _section(config, "graph")
    trainer = _section(config, "trainer")
    evaluation = _section(config, "evaluation")
    classification = _section(config, "classification")
    alias = _safe_alias(config)
    arm = _safe_arm(config)
    production_like = (
        trainer.get("max_epochs") == _EXPECTED_EPOCHS
        or experiment.get("resource_pilot") is False
        or evaluation.get("diagnostic_only") is False
    )
    errors: list[str] = []

    def expect(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    expect(campaign.get("campaign_id") == _CAMPAIGN_ID, "campaign_id")
    expect(alias is not None, "opaque_alias_pair")
    expect(arm is not None, "model_arm_pair")
    if alias is not None and arm is not None:
        expected_label = (
            alias.lower().replace("-", "")
            + "_"
            + _ARM_TO_VARIANT_TOKEN[arm]
            + "_full_core"
        )
        expect(experiment.get("variant_label") == expected_label, "variant_label")
    expect(experiment.get("conclusion_eligible") is True, "conclusion_eligible")
    expect(experiment.get("resource_pilot") is False, "resource_pilot")
    expect(
        experiment.get("excluded_from_primary_comparison") in {None, False},
        "primary_comparison_inclusion",
    )
    expect(
        classification.get("lifecycle_stage") == "exploratory_screen",
        "classification_stage",
    )
    expect(config.get("seed") == _EXPECTED_SEED, "seed")
    expect(config.get("fold") == _EXPECTED_FOLD, "fold")
    expect(config.get("attempt") in _EXPECTED_ATTEMPTS, "attempt")
    expect(dataset.get("task") == "masked_expression_hybrid_count", "dataset_task")
    expect(
        dataset.get("target_scale")
        == "raw_biological_probe_counts_with_per_gene_all_fit_standardized_log1p",
        "target_scale",
    )
    expect(model.get("count_representation_schema") == _REPRESENTATION_SCHEMA, "count_schema")
    for field, expected in (
        ("output_count_states", 8),
        ("input_mask_token_id", 8),
        ("detection_logits_per_gene", 1),
        ("positive_ordinal_logits_per_gene", 6),
        ("continuous_predictions_per_gene", 1),
        ("embedding_dim", 512),
        ("hidden_dim", 512),
        ("graph_layers", 2),
        ("attention_heads", 4),
        ("ffn_dim", 512),
        ("decoder_dim", 512),
    ):
        expect(model.get(field) == expected, f"model_{field}")
    if arm == "hybrid-gat-k1000":
        expect(
            model.get("family") == "hybrid_count_edge_conditioned_gatv2",
            "model_family",
        )
        expect(model.get("uses_graph_inputs") is True, "gat_graph_inputs")
        expect(model.get("uses_edge_inputs") is True, "gat_edge_inputs")
        expect(features.get("use_edge_features") is True, "gat_edge_features")
        edge = _section(features, "edge_features")
        expect(tuple(edge.get("fields", ())) == _EDGE_FIELDS, "edge_field_contract")
    elif arm == "hybrid-matched-self":
        expect(
            model.get("family")
            == "hybrid_count_parameter_matched_self_control",
            "model_family",
        )
        expect(model.get("uses_graph_inputs") is False, "self_graph_inputs")
        expect(model.get("uses_edge_inputs") is False, "self_edge_inputs")
        expect(features.get("use_edge_features") is False, "self_edge_features")
        expect(features.get("edge_features") == [], "self_edge_field_contract")
    node_metadata = _section(features, "node_metadata")
    expect(tuple(node_metadata.get("fields", ())) == _NODE_METADATA_FIELDS, "node_metadata_fields")
    expect(graph.get("neighbor_k") == 1000 and graph.get("k") == 1000, "graph_k")
    expect(graph.get("symmetry") == "mutual", "graph_symmetry")
    expect(graph.get("edge_dropout") == 0.0, "graph_edge_dropout")
    expect(graph.get("self_loops") is False, "graph_self_loops")
    expect(graph.get("coordinates_are_node_covariates") is False, "coordinate_inputs")
    for field, expected in (
        ("learning_rate", 0.0003),
        ("weight_decay", 0.0001),
        ("gradient_clip_norm", 1.0),
        ("huber_delta", 1.0),
        ("max_epochs", _EXPECTED_EPOCHS),
        ("fixed_epoch_budget", True),
        ("early_stopping", False),
        ("restore_best", False),
        ("primary_checkpoint_role", "last"),
        ("checkpoint_policy", "last_only"),
        ("neighbor_sampling", False),
        ("graph_execution", _GRAPH_EXECUTION),
        ("objective", "equal_weight_balanced_detection_ordinal_positive_huber"),
    ):
        expect(trainer.get(field) == expected, f"trainer_{field}")
    for field, expected in (
        ("task_family", "masked_expression_hybrid_count"),
        ("protocol", _PROTOCOL),
        ("canonical_prediction_split", "fit"),
        ("primary_metric", _PRIMARY_METRIC),
        ("primary_direction", "minimize"),
        ("splits", ["fit"]),
        ("mask_modes", list(_MASK_MODES)),
        ("mask_replicates_per_mode", _EXPECTED_MASK_REPLICATES),
        ("generalization_estimate", False),
        ("validation_or_test_selection", False),
        ("conclusion_bearing", True),
        ("diagnostic_only", False),
    ):
        expect(evaluation.get(field) == expected, f"evaluation_{field}")
    return ConfigSemantics(
        alias=alias,
        arm=arm,
        production_like=production_like,
        valid_production=production_like and not errors,
        errors=tuple(errors),
    )


def _registry_inventory(registry: Registry) -> dict[str, Any]:
    """Read campaign attempts and derive exact production slots."""

    with registry.connect() as connection:
        run_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT run_id, status, artifact_path, failure_category,
                       retry_of, seed, fold, attempt, config_json
                FROM runs WHERE campaign_id = ?
                ORDER BY created_at, run_id
                """,
                (_CAMPAIGN_ID,),
            )
        ]
        job_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT job_id, status, run_id, failure_category, retry_of,
                       attempt_count, maximum_attempts, canonical_config_json
                FROM queue_jobs WHERE campaign_id = ?
                ORDER BY created_at, job_id
                """,
                (_CAMPAIGN_ID,),
            )
        ]
        failure_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT f.run_id, f.job_id, f.category
                FROM failures f
                LEFT JOIN runs r ON r.run_id = f.run_id
                LEFT JOIN queue_jobs q ON q.job_id = f.job_id
                WHERE r.campaign_id = ? OR q.campaign_id = ?
                ORDER BY f.failure_id
                """,
                (_CAMPAIGN_ID, _CAMPAIGN_ID),
            )
        ]

    inventory: list[dict[str, Any]] = []
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {
        (alias, arm): [] for alias in _CORE_ALIASES for arm in _ARMS
    }
    for row in run_rows:
        config = _json_mapping(row["config_json"], "registry run config")
        semantics = _config_semantics(config)
        execution_errors = list(semantics.errors)
        for field in ("seed", "fold", "attempt"):
            if row.get(field) != config.get(field):
                execution_errors.append(f"registry_{field}_mismatch")
        valid_run_semantics = (
            semantics.production_like and not execution_errors
        )
        item = {
            "record_type": "run",
            "record_id": str(row["run_id"]),
            "run_id": str(row["run_id"]),
            "core_alias": semantics.alias or "UNRECOGNIZED",
            "arm": semantics.arm or "UNRECOGNIZED",
            "status": str(row["status"]),
            "production_like": semantics.production_like,
            "valid_production_semantics": valid_run_semantics,
            "semantic_errors": execution_errors,
            "failure_category": row.get("failure_category"),
            "retry_of": row.get("retry_of"),
            "attempt": row.get("attempt"),
        }
        inventory.append(item)
        if semantics.production_like and semantics.alias and semantics.arm:
            candidates[(semantics.alias, semantics.arm)].append(
                {**item, "artifact_path": row.get("artifact_path")}
            )
    for row in job_rows:
        config = _json_mapping(
            row["canonical_config_json"], "registry queue config"
        )
        semantics = _config_semantics(config)
        inventory.append(
            {
                "record_type": "queue_job",
                "record_id": str(row["job_id"]),
                "run_id": row.get("run_id"),
                "core_alias": semantics.alias or "UNRECOGNIZED",
                "arm": semantics.arm or "UNRECOGNIZED",
                "status": str(row["status"]),
                "production_like": semantics.production_like,
                "valid_production_semantics": semantics.valid_production,
                "semantic_errors": list(semantics.errors),
                "failure_category": row.get("failure_category"),
                "retry_of": row.get("retry_of"),
                "attempt": row.get("attempt_count"),
            }
        )

    slots: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    duplicates: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for alias in _CORE_ALIASES:
        for arm in _ARMS:
            rows = candidates[(alias, arm)]
            valid_completed = [
                row
                for row in rows
                if row["valid_production_semantics"]
                and row["status"] == "completed"
                and row.get("artifact_path")
            ]
            invalid_rows = [
                row for row in rows if not row["valid_production_semantics"]
            ]
            slot = {
                "core_alias": alias,
                "arm": arm,
                "attempt_count": len(rows),
                "completed_valid_run_count": len(valid_completed),
                "failed_attempt_count": sum(
                    row["status"] == "failed" for row in rows
                ),
                "pending_attempt_count": sum(
                    row["status"] not in {"completed", "failed", "pruned"}
                    for row in rows
                ),
                "invalid_semantics_count": len(invalid_rows),
                "selected_run_id": (
                    valid_completed[0]["run_id"]
                    if len(valid_completed) == 1
                    else None
                ),
            }
            slots.append(slot)
            if len(valid_completed) == 1:
                selected[(alias, arm)] = valid_completed[0]
            elif not valid_completed:
                missing.append({"core_alias": alias, "arm": arm})
            else:
                duplicates.append(
                    {
                        "core_alias": alias,
                        "arm": arm,
                        "run_ids": sorted(
                            str(row["run_id"]) for row in valid_completed
                        ),
                    }
                )
            invalid.extend(
                {
                    "core_alias": alias,
                    "arm": arm,
                    "run_id": row["run_id"],
                    "semantic_errors": row["semantic_errors"],
                }
                for row in invalid_rows
            )

    failed_attempts = [
        {
            "record_type": row["record_type"],
            "record_id": row["record_id"],
            "run_id": row.get("run_id"),
            "core_alias": row["core_alias"],
            "arm": row["arm"],
            "failure_category": row.get("failure_category"),
        }
        for row in inventory
        if row["status"] in {"failed", "stale"}
    ]
    registered_failures = [
        {
            "run_id": row.get("run_id"),
            "job_id": row.get("job_id"),
            "category": row.get("category"),
        }
        for row in failure_rows
    ]
    exact = (
        len(selected) == len(_CORE_ALIASES) * len(_ARMS)
        and not missing
        and not duplicates
        and len({row["run_id"] for row in selected.values()}) == 20
    )
    completed_candidates = [
        row
        for rows in candidates.values()
        for row in rows
        if row["valid_production_semantics"]
        and row["status"] == "completed"
        and row.get("artifact_path")
    ]
    return {
        "exact_primary_coverage": exact,
        "expected_slot_count": 20,
        "selected_completed_run_count": len(selected),
        "slots": slots,
        "attempt_inventory": inventory,
        "missing_slots": missing,
        "duplicate_completed_slots": duplicates,
        "invalid_production_attempts": invalid,
        "failed_attempts": failed_attempts,
        "registered_failures": registered_failures,
        "selected": selected,
        "completed_candidates": completed_candidates,
    }


def _normalize_per_mask_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    alias: str,
    arm: str,
    run_id: str,
) -> tuple[dict[str, Any], ...]:
    expected = {
        (mode, replicate)
        for mode in _MASK_MODES
        for replicate in range(_EXPECTED_MASK_REPLICATES)
    }
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if raw.get("split") != "fit":
            raise HybridCountComparisonError(
                f"{run_id} evaluation row {index} is not fit-only"
            )
        mode = str(raw.get("mask_mode", ""))
        replicate = _as_int(raw.get("mask_replicate"), "mask_replicate")
        key = (mode, replicate)
        if key not in expected or key in observed:
            raise HybridCountComparisonError(
                f"{run_id} has duplicate or unexpected evaluation masks"
            )
        row: dict[str, Any] = {
            "core_alias": alias,
            "arm": arm,
            "run_id": run_id,
            "split": "fit",
            "mask_mode": mode,
            "mask_replicate": replicate,
            "mask_entry_id": str(raw.get("mask_entry_id", "")),
            "mask_seed": _as_int(raw.get("mask_seed"), "mask_seed"),
            "mask_checksum": _sha256(
                raw.get("mask_checksum"), "mask_checksum"
            ),
            "n_masked": _as_int(raw.get("n_masked"), "n_masked"),
        }
        for metric in _REQUIRED_SCALAR_METRICS:
            if metric not in raw:
                raise HybridCountComparisonError(
                    f"{run_id} evaluation row omits {metric}"
                )
            if metric in _NULLABLE_SCALAR_METRICS:
                row[metric] = _nullable_float(raw[metric], metric)
            else:
                row[metric] = _finite_float(raw[metric], metric)
        for metric, length in _VECTOR_METRICS.items():
            if metric not in raw:
                raise HybridCountComparisonError(
                    f"{run_id} evaluation row omits {metric}"
                )
            values = _parse_vector(
                raw[metric], length=length, label=f"{run_id} {metric}"
            )
            for state, value in enumerate(values):
                row[f"{metric}_{state}"] = value
        observed[key] = row
    if set(observed) != expected:
        raise HybridCountComparisonError(
            f"{run_id} does not contain exactly three repeats of all modes"
        )
    return tuple(observed[key] for key in sorted(observed))


def aggregate_replicates(
    per_mask_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Average three technical masks within each arm/mode/core."""

    groups: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in per_mask_rows:
        key = (
            str(row["core_alias"]),
            str(row["arm"]),
            str(row["run_id"]),
            str(row["mask_mode"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    numeric_columns = tuple(
        _REQUIRED_SCALAR_METRICS
        + tuple(
            f"{metric}_{state}"
            for metric, length in _VECTOR_METRICS.items()
            for state in range(length)
        )
    )
    for key, rows in sorted(groups.items()):
        if len(rows) != _EXPECTED_MASK_REPLICATES or {
            _as_int(row["mask_replicate"], "mask_replicate") for row in rows
        } != set(range(_EXPECTED_MASK_REPLICATES)):
            raise HybridCountComparisonError(
                f"{key[0]} {key[1]} {key[3]} requires three technical repeats"
            )
        aggregate: dict[str, Any] = {
            "core_alias": key[0],
            "arm": key[1],
            "run_id": key[2],
            "mask_mode": key[3],
            "technical_replicate_count": _EXPECTED_MASK_REPLICATES,
        }
        for metric in numeric_columns:
            values = [
                _nullable_float(row.get(metric), metric) for row in rows
            ]
            finite = [value for value in values if value is not None]
            if metric not in _NULLABLE_SCALAR_METRICS and len(finite) != len(rows):
                raise HybridCountComparisonError(
                    f"{key[0]} {key[1]} {key[3]} has missing {metric}"
                )
            aggregate[metric] = (
                statistics.fmean(finite) if finite else None
            )
            aggregate[f"{metric}_defined_replicates"] = len(finite)
        output.append(aggregate)
    return tuple(output)


def _mask_identity(rows: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            row["mask_mode"],
            row["mask_replicate"],
            row["mask_entry_id"],
            row["mask_seed"],
            row["mask_checksum"],
            row["n_masked"],
        )
        for row in sorted(
            rows, key=lambda item: (item["mask_mode"], item["mask_replicate"])
        )
    )


def _load_run(
    registry: Registry,
    paths: ProjectPaths,
    candidate: Mapping[str, Any],
) -> RunEvidence:
    alias = str(candidate["core_alias"])
    arm = str(candidate["arm"])
    run_id = str(candidate["run_id"])
    root = Path(str(candidate["artifact_path"]))
    if not root.is_absolute():
        root = paths.project_root / root
    root = root.resolve(strict=False)
    try:
        verification = verify_run_bundle(root)
    except (RunValidationError, OSError, ValueError) as error:
        raise HybridCountComparisonError(
            f"{run_id} failed immutable bundle verification"
        ) from error
    if verification.get("status") != "success":
        raise HybridCountComparisonError(f"{run_id} bundle is not successful")
    config = _load_yaml(root / "config.resolved.yaml")
    semantics = _config_semantics(config)
    if (
        not semantics.valid_production
        or semantics.alias != alias
        or semantics.arm != arm
    ):
        raise HybridCountComparisonError(
            f"{run_id} resolved config no longer matches registry semantics"
        )
    summary = _load_strict_json(root / "summary.json", label="run summary")
    final_metrics = _load_strict_json(
        root / "metrics/final.json", label="final metrics"
    )
    training = _load_strict_json(
        root / "provenance/full_core_training.json",
        label="full-core training provenance",
    )
    fixed_masks = _load_strict_json(
        root / "provenance/fixed_evaluation_masks.json",
        label="fixed evaluation masks",
    )
    convergence = _load_strict_json(
        root / "diagnostics/training_convergence.json",
        label="training convergence",
    )
    resource_usage = _load_strict_json(
        root / "diagnostics/resource_usage.json", label="resource usage"
    )
    parameter_audit = _load_strict_json(
        root / "diagnostics/parameter_structure_audit.json",
        label="parameter structure audit",
    )
    history = _load_table(_logical_table_path(root, "metrics/history"))
    evaluation_rows = _load_table(
        _logical_table_path(root, "metrics/evaluation_replicates")
    )
    if (
        summary.get("run_id") != run_id
        or summary.get("status") != "success"
        or summary.get("training_exit_status") != "success"
        or summary.get("campaign_id") != _CAMPAIGN_ID
        or summary.get("biological_unit_alias") != alias
        or summary.get("evaluation_protocol") != _PROTOCOL
        or summary.get("task_family") != "masked_expression_hybrid_count"
        or summary.get("public_variant") != arm
        or summary.get("model_name")
        != {
            "hybrid-gat-k1000": "hybrid-count-gat",
            "hybrid-matched-self": "hybrid-count-matched-self",
        }[arm]
        or summary.get("checkpoint_role") != "last"
        or summary.get("primary_metric_name") != _PRIMARY_METRIC
        or summary.get("exact_parameter_match") is not True
        or summary.get("graph_supplied_to_model")
        is not (arm == "hybrid-gat-k1000")
        or summary.get("evaluation_mask_replicates_per_mode")
        != _EXPECTED_MASK_REPLICATES
        or summary.get(
            "evaluation_metrics_include_all_configured_replicates_per_mode"
        )
        is not True
        or summary.get("conclusion_eligible") is not True
        or summary.get("diagnostic_resource_pilot") is not False
    ):
        raise HybridCountComparisonError(
            f"{run_id} summary violates the production contract"
        )
    if any(
        _as_int(value, f"{run_id} final epoch") != _EXPECTED_EPOCHS - 1
        for value in (summary.get("final_epoch"), training.get("final_epoch"))
    ):
        raise HybridCountComparisonError(f"{run_id} is not a final-epoch run")
    if any(
        _as_int(value, f"{run_id} epoch budget") != _EXPECTED_EPOCHS
        for value in (
            summary.get("fixed_epoch_budget"),
            training.get("fixed_epoch_budget"),
        )
    ):
        raise HybridCountComparisonError(f"{run_id} did not complete 200 epochs")
    if (
        training.get("training_protocol") != _TRAINING_PROTOCOL
        or training.get("graph_execution") != _GRAPH_EXECUTION
        or training.get("checkpoint_policy") != _CHECKPOINT_POLICY
    ):
        raise HybridCountComparisonError(
            f"{run_id} training provenance violates the fixed protocol"
        )
    graph_budgets = parameter_audit.get("graph_layer_parameter_budgets")
    self_budgets = parameter_audit.get("self_layer_parameter_budgets")
    if (
        parameter_audit.get("schema")
        != "hybrid_count_parameter_structure_audit_v1"
        or parameter_audit.get("trainable_parameter_count_graph")
        != _EXPECTED_PARAMETER_COUNT
        or parameter_audit.get("trainable_parameter_count_self")
        != _EXPECTED_PARAMETER_COUNT
        or parameter_audit.get("exact_trainable_parameter_match") is not True
        or parameter_audit.get("encoder_initial_state_bit_identical") is not True
        or parameter_audit.get("decoder_initial_state_bit_identical") is not True
        or not isinstance(graph_budgets, list)
        or not isinstance(self_budgets, list)
        or len(graph_budgets) != 2
        or graph_budgets != self_budgets
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in graph_budgets
        )
        or training.get("parameter_structure_audit") != parameter_audit
    ):
        raise HybridCountComparisonError(
            f"{run_id} parameter structure audit is incomplete or mismatched"
        )
    summary_checkpoint = _mapping(summary.get("checkpoint"), "summary checkpoint")
    if (
        summary_checkpoint.get("role") != "last"
        or summary_checkpoint.get("final_epoch") != _EXPECTED_EPOCHS - 1
        or summary_checkpoint.get("policy") != _CHECKPOINT_POLICY
        or summary_checkpoint.get("monitored_metric") is not None
        or summary.get("graph_interpretation")
        != "broad_regional_context_not_direct_interaction"
        or summary.get("generalization_estimate") is not False
        or fixed_masks.get("used_for_gradient_updates") is not False
        or fixed_masks.get("used_for_checkpoint_selection") is not False
        or fixed_masks.get("technical_replicates_not_biological_replicates")
        is not True
    ):
        raise HybridCountComparisonError(
            f"{run_id} summary, checkpoint, or mask provenance is invalid"
        )
    if (
        convergence.get("final_epoch") != _EXPECTED_EPOCHS - 1
        or convergence.get("all_epochs_completed") is not True
        or convergence.get("all_losses_and_gradients_finite") is not True
        or resource_usage.get("diagnostic_resource_pilot") is not False
        or resource_usage.get("public_variant") != arm
        or resource_usage.get("biological_unit_alias") != alias
        or resource_usage.get("finite_losses_and_gradients") is not True
        or resource_usage.get("epochs_completed") != _EXPECTED_EPOCHS
    ):
        raise HybridCountComparisonError(
            f"{run_id} convergence or resource audit is incomplete"
        )
    for field in (
        "final_train_hybrid_loss",
        "minimum_observed_train_hybrid_loss",
        "last_20_epoch_loss_slope",
    ):
        _finite_float(convergence.get(field), f"convergence {field}")
    convergence_components = _mapping(
        convergence.get("final_components"), "convergence final components"
    )
    for metric in _COMPONENT_METRICS:
        _finite_float(
            convergence_components.get(metric),
            f"convergence final {metric}",
        )
    epochs = [_as_int(row.get("epoch"), "history epoch") for row in history]
    if epochs != list(range(_EXPECTED_EPOCHS)):
        raise HybridCountComparisonError(
            f"{run_id} history does not contain epochs 0 through 199"
        )
    for index, row in enumerate(history):
        for metric in (
            "train_hybrid_loss",
            "train_detection_bce",
            "train_ordinal_bce",
            "train_positive_continuous_huber",
            "gradient_norm",
            "duration_seconds",
        ):
            value = _finite_float(row.get(metric), f"history[{index}].{metric}")
            if value < 0:
                raise HybridCountComparisonError(
                    f"{run_id} history contains negative {metric}"
                )
    per_mask = _normalize_per_mask_rows(
        evaluation_rows, alias=alias, arm=arm, run_id=run_id
    )
    core_mode = aggregate_replicates(per_mask)
    for row in core_mode:
        mode = str(row["mask_mode"])
        for metric in _REQUIRED_SCALAR_METRICS:
            expected_name = f"fit/{mode}/{metric}"
            if expected_name not in final_metrics:
                raise HybridCountComparisonError(
                    f"{run_id} final metrics omit {expected_name}"
                )
            observed = _nullable_float(final_metrics[expected_name], expected_name)
            expected = row[metric]
            if observed is None or expected is None:
                if observed != expected:
                    raise HybridCountComparisonError(
                        f"{run_id} final {expected_name} does not match mask mean"
                    )
            elif not math.isclose(observed, float(expected), rel_tol=1e-8, abs_tol=1e-10):
                raise HybridCountComparisonError(
                    f"{run_id} final {expected_name} does not match mask mean"
                )
    checkpoint_records = build_checkpoint_catalog(
        registry, paths, run_ids=[run_id]
    )
    if len(checkpoint_records) != 1:
        raise HybridCountComparisonError(
            f"{run_id} requires exactly one registered checkpoint"
        )
    checkpoint = checkpoint_records[0]
    if (
        checkpoint.get("checkpoint_role") != "last"
        or checkpoint.get("best_epoch") != _EXPECTED_EPOCHS - 1
        or checkpoint.get("verification_status") != "verified"
    ):
        raise HybridCountComparisonError(
            f"{run_id} checkpoint catalog role, epoch, or status is invalid"
        )
    verify_checkpoint_record(checkpoint, paths=paths)
    parameter_count = _as_int(summary.get("parameter_count"), "parameter_count")
    if (
        parameter_count != _EXPECTED_PARAMETER_COUNT
        or _as_int(training.get("parameter_count"), "training parameter_count")
        != parameter_count
        or _as_int(resource_usage.get("parameter_count"), "resource parameter_count")
        != parameter_count
    ):
        raise HybridCountComparisonError(
            f"{run_id} parameter audit does not match the frozen count"
        )
    graph_sha = _sha256(summary.get("graph_sha256"), "graph_sha256")
    duration = _finite_float(summary.get("duration_seconds"), "duration_seconds")
    peak_vram = _finite_float(summary.get("peak_vram_gib"), "peak_vram_gib")
    resource_duration = _finite_float(
        resource_usage.get("total_pilot_or_run_duration_seconds"),
        "resource total duration",
    )
    resource_peak = _finite_float(
        resource_usage.get("peak_allocated_vram_gib"),
        "resource peak VRAM",
    )
    final_resource_duration = _finite_float(
        final_metrics.get("resource/total_duration_seconds"),
        "final resource total duration",
    )
    final_resource_peak = _finite_float(
        final_metrics.get("resource/peak_vram_gib"),
        "final resource peak VRAM",
    )
    primary_value = _finite_float(
        final_metrics.get(_PRIMARY_METRIC), _PRIMARY_METRIC
    )
    if (
        not math.isclose(duration, resource_duration, rel_tol=1e-10, abs_tol=1e-8)
        or not math.isclose(duration, final_resource_duration, rel_tol=1e-10, abs_tol=1e-8)
        or not math.isclose(peak_vram, resource_peak, rel_tol=1e-10, abs_tol=1e-10)
        or not math.isclose(peak_vram, final_resource_peak, rel_tol=1e-10, abs_tol=1e-10)
        or not math.isclose(
            _finite_float(summary.get("primary_metric_value"), "primary metric value"),
            primary_value,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        or _mapping(summary.get("metrics"), "summary metrics") != final_metrics
    ):
        raise HybridCountComparisonError(
            f"{run_id} summary and metric artifacts disagree"
        )
    if duration < 0 or peak_vram < 0:
        raise HybridCountComparisonError(
            f"{run_id} has negative runtime or peak memory"
        )
    return RunEvidence(
        alias=alias,
        arm=arm,
        run_id=run_id,
        root=root,
        config=config,
        summary=summary,
        training=training,
        parameter_audit=parameter_audit,
        convergence=convergence,
        resource_usage=resource_usage,
        fixed_masks=fixed_masks,
        per_mask_rows=per_mask,
        core_mode_rows=core_mode,
        parameter_count=parameter_count,
        graph_sha256=graph_sha,
        checkpoint_sha256=_sha256(
            checkpoint.get("checkpoint_sha256"), "checkpoint_sha256"
        ),
        checkpoint_duplicate_count=_as_int(
            checkpoint.get("content_duplicate_count"),
            "checkpoint_duplicate_count",
        ),
        duration_seconds=duration,
        peak_vram_gib=peak_vram,
    )


def _validate_pairs(runs: Mapping[tuple[str, str], RunEvidence]) -> None:
    parameter_counts = {run.parameter_count for run in runs.values()}
    if len(parameter_counts) != 1:
        raise HybridCountComparisonError(
            "the twenty production runs are not exactly parameter matched"
        )
    for alias in _CORE_ALIASES:
        gat = runs[(alias, "hybrid-gat-k1000")]
        control = runs[(alias, "hybrid-matched-self")]
        if gat.run_id == control.run_id:
            raise HybridCountComparisonError(f"{alias} arms share a run ID")
        if gat.graph_sha256 != control.graph_sha256:
            raise HybridCountComparisonError(
                f"{alias} arms do not audit the same k1000 graph"
            )
        if _mask_identity(gat.per_mask_rows) != _mask_identity(
            control.per_mask_rows
        ):
            raise HybridCountComparisonError(
                f"{alias} arms do not use identical fixed masks"
            )
        for section in ("dataset", "graph", "masking", "trainer", "evaluation"):
            if gat.config.get(section) != control.config.get(section):
                raise HybridCountComparisonError(
                    f"{alias} arms differ in config.{section}"
                )


def _whole_node_by_slot(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    selected = {
        (str(row["core_alias"]), str(row["arm"])): row
        for row in rows
        if row["mask_mode"] == "whole_node"
    }
    if set(selected) != {
        (alias, arm) for alias in _CORE_ALIASES for arm in _ARMS
    }:
        raise HybridCountComparisonError(
            "whole-node aggregation does not contain exactly twenty slots"
        )
    return selected


def evaluate_frozen_gates(
    core_mode_rows: Sequence[Mapping[str, Any]],
    *,
    all_runs_verified: bool,
) -> dict[str, Any]:
    """Compute both frozen gates from one core-level row per arm."""

    whole = _whole_node_by_slot(core_mode_rows)
    graph_rows: list[dict[str, Any]] = []
    component_results: dict[str, Any] = {}
    raw_component_p: dict[str, float] = {}
    for alias in _CORE_ALIASES:
        gat = whole[(alias, "hybrid-gat-k1000")]
        control = whole[(alias, "hybrid-matched-self")]
        self_loss = _finite_float(control["hybrid_loss"], "self hybrid loss")
        gat_loss = _finite_float(gat["hybrid_loss"], "GAT hybrid loss")
        if self_loss <= 0:
            raise HybridCountComparisonError(
                f"{alias} self hybrid loss must be positive"
            )
        row: dict[str, Any] = {
            "core_alias": alias,
            "gat_run_id": gat["run_id"],
            "self_run_id": control["run_id"],
            "gat_hybrid_loss": gat_loss,
            "self_hybrid_loss": self_loss,
            "relative_hybrid_loss_improvement": (self_loss - gat_loss) / self_loss,
            "gat_favors_total_loss": gat_loss < self_loss,
        }
        for metric in _COMPONENT_METRICS:
            row[f"raw_self_minus_gat_{metric}"] = _finite_float(
                control[metric], metric
            ) - _finite_float(gat[metric], metric)
        row["gat_positive_ordinal_mae"] = _finite_float(
            gat["positive_ordinal_mae"], "GAT positive ordinal MAE"
        )
        row["self_positive_ordinal_mae"] = _finite_float(
            control["positive_ordinal_mae"], "self positive ordinal MAE"
        )
        row["gat_positive_continuous_huber"] = _finite_float(
            gat["positive_continuous_huber"], "GAT positive Huber"
        )
        row["self_positive_continuous_huber"] = _finite_float(
            control["positive_continuous_huber"], "self positive Huber"
        )
        graph_rows.append(row)
    for metric in _COMPONENT_METRICS:
        differences = [
            float(row[f"raw_self_minus_gat_{metric}"])
            for row in graph_rows
        ]
        exact = exact_one_sided_paired_sign_flip(differences)
        if exact.unit_count != 10 or exact.permutation_count != 1024:
            raise HybridCountComparisonError(
                "component inference did not enumerate exactly 2^10 sign flips"
            )
        component_results[metric] = exact.to_dict()
        component_results[metric]["core_differences"] = differences
        raw_component_p[metric] = exact.p_value
    if set(raw_component_p) != set(_COMPONENT_METRICS):
        raise HybridCountComparisonError(
            "Holm family must contain exactly three loss components"
        )
    holm = holm_adjust(raw_component_p, alpha=0.05)
    for metric in _COMPONENT_METRICS:
        component_results[metric]["holm"] = holm[metric]

    total_gains = paired_relative_improvements(
        [float(row["self_hybrid_loss"]) for row in graph_rows],
        [float(row["gat_hybrid_loss"]) for row in graph_rows],
    ).tolist()
    for row, gain in zip(graph_rows, total_gains, strict=True):
        row["relative_hybrid_loss_improvement"] = float(gain)
        row["gat_favors_total_loss"] = bool(gain > 0)
    gat_ordinal = [float(row["gat_positive_ordinal_mae"]) for row in graph_rows]
    self_ordinal = [float(row["self_positive_ordinal_mae"]) for row in graph_rows]
    gat_continuous = [
        float(row["gat_positive_continuous_huber"]) for row in graph_rows
    ]
    self_continuous = [
        float(row["self_positive_continuous_huber"]) for row in graph_rows
    ]
    graph_criteria = {
        "all_twenty_production_runs_complete_and_verified": {
            "observed": all_runs_verified,
            "threshold": True,
            "passes": all_runs_verified,
        },
        "mean_paired_relative_hybrid_loss_improvement_at_least_2_percent": {
            "observed": statistics.fmean(total_gains),
            "threshold": 0.02,
            "passes": statistics.fmean(total_gains) >= 0.02,
        },
        "at_least_8_of_10_cores_favor_gat": {
            "observed": sum(value > 0 for value in total_gains),
            "threshold": 8,
            "passes": sum(value > 0 for value in total_gains) >= 8,
        },
        "all_three_holm_adjusted_component_tests_reject": {
            "observed": sum(bool(holm[name]["reject"]) for name in _COMPONENT_METRICS),
            "threshold": 3,
            "passes": all(bool(holm[name]["reject"]) for name in _COMPONENT_METRICS),
        },
        "mean_positive_ordinal_mae_not_worse": {
            "observed": statistics.fmean(gat_ordinal) - statistics.fmean(self_ordinal),
            "threshold": 0.0,
            "passes": statistics.fmean(gat_ordinal) <= statistics.fmean(self_ordinal),
        },
        "mean_positive_continuous_huber_not_worse": {
            "observed": statistics.fmean(gat_continuous) - statistics.fmean(self_continuous),
            "threshold": 0.0,
            "passes": statistics.fmean(gat_continuous) <= statistics.fmean(self_continuous),
        },
    }

    representation_rows: list[dict[str, Any]] = []
    for alias in _CORE_ALIASES:
        gat = whole[(alias, "hybrid-gat-k1000")]
        ordinal_reference = _finite_float(
            gat["reference_per_gene_positive_ordinal_mae"],
            "reference positive ordinal MAE",
        )
        continuous_reference = _finite_float(
            gat["reference_per_gene_positive_continuous_huber"],
            "reference positive continuous Huber",
        )
        if ordinal_reference <= 0 or continuous_reference <= 0:
            raise HybridCountComparisonError(
                f"{alias} positive reference losses must be positive"
            )
        ordinal = _finite_float(gat["positive_ordinal_mae"], "positive ordinal MAE")
        continuous = _finite_float(
            gat["positive_continuous_huber"], "positive continuous Huber"
        )
        detection = _finite_float(
            gat["detection_balanced_accuracy"], "detection balanced accuracy"
        )
        detection_reference = _finite_float(
            gat["reference_per_gene_detection_balanced_accuracy"],
            "reference detection balanced accuracy",
        )
        representation_rows.append(
            {
                "core_alias": alias,
                "gat_run_id": gat["run_id"],
                "positive_ordinal_mae": ordinal,
                "reference_positive_ordinal_mae": ordinal_reference,
                "relative_positive_ordinal_mae_improvement": (
                    ordinal_reference - ordinal
                )
                / ordinal_reference,
                "positive_continuous_huber": continuous,
                "reference_positive_continuous_huber": continuous_reference,
                "relative_positive_continuous_huber_improvement": (
                    continuous_reference - continuous
                )
                / continuous_reference,
                "detection_balanced_accuracy": detection,
                "reference_detection_balanced_accuracy": detection_reference,
                "detection_balanced_accuracy_difference": detection
                - detection_reference,
            }
        )
    ordinal_gains = paired_relative_improvements(
        [float(row["reference_positive_ordinal_mae"]) for row in representation_rows],
        [float(row["positive_ordinal_mae"]) for row in representation_rows],
    ).tolist()
    continuous_gains = paired_relative_improvements(
        [
            float(row["reference_positive_continuous_huber"])
            for row in representation_rows
        ],
        [float(row["positive_continuous_huber"]) for row in representation_rows],
    ).tolist()
    for row, ordinal_gain, continuous_gain in zip(
        representation_rows, ordinal_gains, continuous_gains, strict=True
    ):
        row["relative_positive_ordinal_mae_improvement"] = float(ordinal_gain)
        row["relative_positive_continuous_huber_improvement"] = float(
            continuous_gain
        )
    detection_differences = [
        float(row["detection_balanced_accuracy_difference"])
        for row in representation_rows
    ]
    representation_criteria = {
        "mean_positive_ordinal_mae_relative_improvement_at_least_2_percent": {
            "observed": statistics.fmean(ordinal_gains),
            "threshold": 0.02,
            "passes": statistics.fmean(ordinal_gains) >= 0.02,
        },
        "mean_positive_continuous_huber_relative_improvement_at_least_2_percent": {
            "observed": statistics.fmean(continuous_gains),
            "threshold": 0.02,
            "passes": statistics.fmean(continuous_gains) >= 0.02,
        },
        "at_least_8_of_10_cores_favor_gat_on_positive_ordinal_mae": {
            "observed": sum(value > 0 for value in ordinal_gains),
            "threshold": 8,
            "passes": sum(value > 0 for value in ordinal_gains) >= 8,
        },
        "at_least_8_of_10_cores_favor_gat_on_positive_continuous_huber": {
            "observed": sum(value > 0 for value in continuous_gains),
            "threshold": 8,
            "passes": sum(value > 0 for value in continuous_gains) >= 8,
        },
        "mean_detection_balanced_accuracy_exceeds_reference": {
            "observed": statistics.fmean(detection_differences),
            "threshold": 0.0,
            "passes": statistics.fmean(detection_differences) > 0.0,
        },
    }
    return {
        "graph_gate": {
            "assessable": True,
            "passes": all(item["passes"] for item in graph_criteria.values()),
            "criteria": graph_criteria,
            "core_rows": graph_rows,
            "component_inference": component_results,
            "direction": "raw matched-self minus GAT; positive favors GAT",
        },
        "representation_gate": {
            "assessable": True,
            "passes": all(
                item["passes"] for item in representation_criteria.values()
            ),
            "criteria": representation_criteria,
            "core_rows": representation_rows,
        },
    }


def _unassessable_gates(reason: str) -> dict[str, Any]:
    return {
        name: {
            "assessable": False,
            "passes": False,
            "reason": reason,
            "criteria": {},
            "core_rows": [],
        }
        for name in ("graph_gate", "representation_gate")
    }


def _aggregate_across_cores(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    aggregate_metrics = _REQUIRED_SCALAR_METRICS + tuple(
        f"{metric}_{state}"
        for metric, length in _VECTOR_METRICS.items()
        for state in range(length)
    )
    for arm in _ARMS:
        for mode in _MASK_MODES:
            selected = [
                row
                for row in rows
                if row["arm"] == arm and row["mask_mode"] == mode
            ]
            if len(selected) != 10:
                raise HybridCountComparisonError(
                    f"{arm} {mode} does not contain ten equal-weight core rows"
                )
            for metric in aggregate_metrics:
                values = [
                    _nullable_float(row.get(metric), metric) for row in selected
                ]
                finite = [value for value in values if value is not None]
                output.append(
                    {
                        "arm": arm,
                        "mask_mode": mode,
                        "metric": metric,
                        "independent_core_count": 10,
                        "defined_core_count": len(finite),
                        "equal_core_mean": statistics.fmean(finite) if finite else None,
                        "core_median": statistics.median(finite) if finite else None,
                        "core_sample_sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
                        "minimum": min(finite) if finite else None,
                        "maximum": max(finite) if finite else None,
                    }
                )
    return output


def _run_resource_rows(
    runs: Mapping[tuple[str, str], RunEvidence],
) -> list[dict[str, Any]]:
    """Flatten auditable runtime, memory, parameter, and convergence fields."""

    rows: list[dict[str, Any]] = []
    for alias in _CORE_ALIASES:
        for arm in _ARMS:
            run = runs.get((alias, arm))
            if run is None:
                continue
            resource = run.resource_usage
            convergence = run.convergence
            rows.append(
                {
                    "core_alias": alias,
                    "arm": arm,
                    "run_id": run.run_id,
                    "parameter_count": run.parameter_count,
                    "duration_seconds": run.duration_seconds,
                    "training_duration_seconds": _finite_float(
                        resource.get("training_duration_seconds"),
                        "training duration",
                    ),
                    "evaluation_duration_seconds": _finite_float(
                        resource.get("evaluation_duration_seconds"),
                        "evaluation duration",
                    ),
                    "peak_allocated_vram_gib": run.peak_vram_gib,
                    "device": str(resource.get("device")),
                    "cuda_device_name": resource.get("cuda_device_name"),
                    "torch_version": str(resource.get("torch_version")),
                    "torch_cuda_version": resource.get("torch_cuda_version"),
                    "effective_training_amp": resource.get(
                        "effective_training_amp"
                    ),
                    "final_epoch": _as_int(
                        convergence.get("final_epoch"), "final epoch"
                    ),
                    "all_epochs_completed": convergence.get(
                        "all_epochs_completed"
                    ),
                    "all_losses_and_gradients_finite": convergence.get(
                        "all_losses_and_gradients_finite"
                    ),
                    "final_train_hybrid_loss": _finite_float(
                        convergence.get("final_train_hybrid_loss"),
                        "final train hybrid loss",
                    ),
                    "minimum_observed_train_hybrid_loss": _finite_float(
                        convergence.get("minimum_observed_train_hybrid_loss"),
                        "minimum observed train hybrid loss",
                    ),
                    "last_20_epoch_loss_slope": _finite_float(
                        convergence.get("last_20_epoch_loss_slope"),
                        "last 20 epoch loss slope",
                    ),
                    "checkpoint_sha256": run.checkpoint_sha256,
                    "checkpoint_content_duplicate_count": (
                        run.checkpoint_duplicate_count
                    ),
                }
            )
    return rows


def _load_prior_categorical(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False, "reason": "not_requested"}
    if not path.is_file():
        return {"available": False, "reason": "file_missing"}
    source = _load_strict_json(path, label="prior categorical comparison")
    if (
        source.get("campaign_id") != _PRIOR_CATEGORICAL_CAMPAIGN
        or source.get("status") != "complete"
    ):
        raise HybridCountComparisonError(
            "prior categorical JSON has the wrong campaign identity"
        )
    variants = _mapping(source.get("variant_aggregates"), "categorical variants")
    if set(variants) != set(_PRIOR_SAFE_VARIANTS):
        raise HybridCountComparisonError(
            "prior categorical JSON has an unexpected variant identity"
        )
    prior_gate = _mapping(source.get("h1_width_gate"), "prior width gate")
    if not isinstance(prior_gate.get("passes"), bool):
        raise HybridCountComparisonError(
            "prior categorical JSON has no safe gate decision"
        )
    extracted: list[dict[str, Any]] = []
    for name, raw in sorted(variants.items()):
        variant = _mapping(raw, f"categorical variant {name}")
        if not isinstance(variant.get("model_seeds"), list) or len(
            variant["model_seeds"]
        ) != 3:
            raise HybridCountComparisonError(
                "prior categorical JSON has an unexpected seed count"
            )
        metrics = _mapping(variant.get("metrics"), f"categorical {name} metrics")
        baselines = _mapping(
            variant.get("baselines"), f"categorical {name} baselines"
        )
        recalls = _mapping(
            variant.get("token_recalls_percent"),
            f"categorical {name} recalls",
        )
        row: dict[str, Any] = {
            "source": "prior_one_core_categorical",
            "variant": _PRIOR_SAFE_VARIANTS[str(name)],
            "model_seed_count": 3,
            "exact_accuracy_percent": _finite_float(
                _mapping(metrics.get("exact_percent"), "exact percent").get("mean"),
                "categorical exact percent",
            ),
            "balanced_accuracy_percent": _finite_float(
                _mapping(metrics.get("balanced_percent"), "balanced percent").get("mean"),
                "categorical balanced percent",
            ),
            "positive_exact_accuracy_percent": _finite_float(
                _mapping(metrics.get("nonzero_percent"), "nonzero percent").get("mean"),
                "categorical nonzero percent",
            ),
            "all_zero_exact_accuracy_percent": _finite_float(
                _mapping(
                    baselines.get("baseline_always_zero_accuracy_percent"),
                    "all-zero baseline",
                ).get("mean"),
                "categorical all-zero exact percent",
            ),
            "all_zero_balanced_accuracy_percent": _finite_float(
                _mapping(
                    baselines.get("baseline_always_zero_balanced_accuracy_percent"),
                    "all-zero balanced baseline",
                ).get("mean"),
                "categorical all-zero balanced percent",
            ),
        }
        for state in range(4):
            row[f"state_{state}_recall_percent"] = _finite_float(
                _mapping(recalls.get(str(state)), f"recall state {state}").get("mean"),
                f"categorical state {state} recall",
            )
        extracted.append(row)
    return {
        "available": True,
        "campaign_id": _PRIOR_CATEGORICAL_CAMPAIGN,
        "status": "complete",
        "single_core": True,
        "prior_tissue_context": "true_normal",
        "current_tissue_context": "pathology_confirmed_adjacent_normal",
        "exchangeable_with_current_ten_core_study": False,
        "comparison_is_descriptive_only": True,
        "safe_scope": {
            "estimand": "held_in_whole_node_masked_count_reconstruction",
            "output_states": "0_1_2_3plus",
            "biological_core_count": 1,
            "model_seed_count": 3,
            "generalization_estimate": False,
        },
        "prior_width_gate_passed": bool(prior_gate["passes"]),
        "rows": extracted,
    }


def _current_collapsed_rows(
    core_mode_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gat = [
        row
        for row in core_mode_rows
        if row["arm"] == "hybrid-gat-k1000"
        and row["mask_mode"] == "whole_node"
    ]
    if len(gat) != 10:
        return []
    row: dict[str, Any] = {
        "source": "current_ten_core_hybrid_gat",
        "variant": "hybrid-gat-k1000",
        "independent_core_count": 10,
        "exact_accuracy_percent": 100.0
        * statistics.fmean(float(item["collapsed4_exact_accuracy"]) for item in gat),
        "balanced_accuracy_percent": 100.0
        * statistics.fmean(float(item["collapsed4_balanced_accuracy"]) for item in gat),
        "positive_exact_accuracy_percent": 100.0
        * statistics.fmean(
            float(item["collapsed4_positive_exact_accuracy"]) for item in gat
        ),
        "all_zero_exact_accuracy_percent": 100.0
        * statistics.fmean(
            float(item["reference_all_zero_collapsed4_exact_accuracy"])
            for item in gat
        ),
    }
    for state in range(4):
        recalls = [item[f"collapsed4_recall_{state}"] for item in gat]
        finite = [float(value) for value in recalls if value is not None]
        row[f"state_{state}_recall_percent"] = (
            100.0 * statistics.fmean(finite) if finite else None
        )
    return [row]


def compare_hybrid_count_multicore(
    registry: Registry,
    paths: ProjectPaths,
    *,
    categorical_json: str | Path | None = _PRIOR_CATEGORICAL_DEFAULT,
) -> dict[str, Any]:
    """Verify registered production evidence and compute frozen gates."""

    coverage = _registry_inventory(registry)
    selected = coverage.pop("selected")
    completed_candidates = coverage.pop("completed_candidates")
    verification_rows: list[dict[str, Any]] = []
    runs: dict[tuple[str, str], RunEvidence] = {}
    verification_failures: list[dict[str, Any]] = []
    pilot_gate = _load_campaign_receipts(paths)
    unique_candidates = {
        str(row["run_id"]): row for row in completed_candidates
    }
    loaded_by_id: dict[str, RunEvidence] = {}
    for run_id, candidate in sorted(unique_candidates.items()):
        try:
            evidence = _load_run(registry, paths, candidate)
        except Exception as error:
            verification_failures.append(
                {
                    "core_alias": candidate["core_alias"],
                    "arm": candidate["arm"],
                    "run_id": run_id,
                    "error_type": type(error).__name__,
                    "stage": "bundle_or_checkpoint_verification",
                }
            )
            verification_rows.append(
                {
                    "core_alias": candidate["core_alias"],
                    "arm": candidate["arm"],
                    "run_id": run_id,
                    "verified": False,
                    "checkpoint_sha256": None,
                    "checkpoint_duplicate_count": None,
                }
            )
            continue
        loaded_by_id[run_id] = evidence
        verification_rows.append(
            {
                "core_alias": evidence.alias,
                "arm": evidence.arm,
                "run_id": run_id,
                "verified": True,
                "checkpoint_sha256": evidence.checkpoint_sha256,
                "checkpoint_duplicate_count": evidence.checkpoint_duplicate_count,
            }
        )
    if coverage["exact_primary_coverage"]:
        for slot, candidate in selected.items():
            evidence = loaded_by_id.get(str(candidate["run_id"]))
            if evidence is not None:
                runs[slot] = evidence

    all_primary_artifacts_verified = (
        coverage["exact_primary_coverage"]
        and len(runs) == 20
        and not verification_failures
    )
    analysis_ready = (
        all_primary_artifacts_verified
        and pilot_gate.get("verified") is True
    )
    pairing_error: str | None = None
    if analysis_ready:
        try:
            _validate_pairs(runs)
        except HybridCountComparisonError as error:
            pairing_error = str(error)
            analysis_ready = False
    per_mask_rows = [
        dict(row)
        for run in runs.values()
        for row in run.per_mask_rows
    ] if analysis_ready else []
    core_mode_rows = [
        dict(row)
        for run in runs.values()
        for row in run.core_mode_rows
    ] if analysis_ready else []
    if analysis_ready:
        gates = evaluate_frozen_gates(
            core_mode_rows, all_runs_verified=True
        )
        aggregate_rows = _aggregate_across_cores(core_mode_rows)
        resource_rows = _run_resource_rows(runs)
    else:
        if pairing_error:
            reason = "pairing_contract_failed"
        elif pilot_gate.get("verified") is not True:
            reason = "verified_pilot_gate_receipt_unavailable"
        else:
            reason = "exact_verified_10_by_2_coverage_unavailable"
        gates = _unassessable_gates(reason)
        aggregate_rows = []
        resource_rows = []
    categorical_path = (
        Path(categorical_json).resolve(strict=False)
        if categorical_json is not None
        else None
    )
    prior = _load_prior_categorical(categorical_path)
    categorical_rows = _current_collapsed_rows(core_mode_rows)
    categorical_rows.extend(prior.get("rows", []))
    negative_evidence: list[str] = []
    if not coverage["exact_primary_coverage"]:
        negative_evidence.append("Exact 10-alias by 2-arm production coverage is absent or duplicated.")
    if verification_failures:
        negative_evidence.append("At least one completed production bundle or checkpoint failed verification.")
    if pilot_gate.get("verified") is not True:
        negative_evidence.append(
            "The checksum-bound two-arm pilot gate receipt is missing or invalid."
        )
    if (
        prior.get("available") is True
        and prior.get("prior_width_gate_passed") is False
    ):
        negative_evidence.append(
            "The prior one-core categorical width-increase gate failed; this is nonexchangeable descriptive context only."
        )
    if pairing_error:
        negative_evidence.append("The paired arms violate a frozen identity or parameter-matching contract.")
    for name in ("graph_gate", "representation_gate"):
        gate = gates[name]
        if gate.get("assessable") and not gate.get("passes"):
            negative_evidence.append(f"The frozen {name.replace('_', ' ')} failed.")
    checkpoint_duplicates = [
        row for row in verification_rows if (row.get("checkpoint_duplicate_count") or 0) > 1
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "hybrid_count_ten_core_comparison",
        "campaign_id": _CAMPAIGN_ID,
        "status": "complete" if analysis_ready else "incomplete_or_invalid",
        "exploratory": True,
        "analysis_ready": analysis_ready,
        "coverage": coverage,
        "pilot_gate": pilot_gate,
        "verification": {
            "all_selected_bundles_and_checkpoints_verified": (
                all_primary_artifacts_verified
            ),
            "rows": verification_rows,
            "failures": verification_failures,
            "checkpoint_content_duplicates": checkpoint_duplicates,
            "pairing_error": pairing_error,
        },
        "aggregation_contract": {
            "technical_replicates_averaged_within_mode_core_arm": 3,
            "independent_biological_unit_count": 10,
            "core_weighting": "equal",
            "cells_are_independent_replicates": False,
            "mask_replicates_are_independent_replicates": False,
        },
        "graph_gate": gates["graph_gate"],
        "representation_gate": gates["representation_gate"],
        "prior_categorical_comparison": prior,
        "negative_evidence": negative_evidence,
        "limitations": [
            "The ten opaque aliases denote pathology-confirmed adjacent-normal tissue, not true Normal tissue.",
            "All fitting, preprocessing, and reference estimation occur within each core; this is not patient-held-out generalization.",
            "The k=1000 graph represents broad regional context and cannot identify direct cell-cell communication.",
            "The prior categorical experiment fit three model seeds on one true-Normal core, whereas this study uses ten pathology-confirmed adjacent-normal cores; the designs are not exchangeable.",
            "Predictive reconstruction and graph gain do not establish biological mechanism or causality.",
        ],
        "maximum_defensible_conclusion": (
            "held-in masked-expression representation capacity and possible "
            "broad-context graph gain across ten adjacent-normal cores"
        ),
        "tables": {
            "attempt_inventory": coverage["attempt_inventory"],
            "coverage_slots": coverage["slots"],
            "missing_slots": coverage["missing_slots"],
            "duplicate_completed_slots": coverage[
                "duplicate_completed_slots"
            ],
            "invalid_production_attempts": coverage[
                "invalid_production_attempts"
            ],
            "registered_failures": coverage["registered_failures"],
            "verification": verification_rows,
            "verification_failures": verification_failures,
            "checkpoint_content_duplicates": checkpoint_duplicates,
            "run_resources_and_convergence": resource_rows,
            "per_mask_metrics": per_mask_rows,
            "core_mode_metrics": core_mode_rows,
            "equal_core_aggregates": aggregate_rows,
            "graph_core_comparisons": gates["graph_gate"].get("core_rows", []),
            "representation_core_comparisons": gates["representation_gate"].get("core_rows", []),
            "component_inference": [
                {"metric": name, **value}
                for name, value in gates["graph_gate"].get("component_inference", {}).items()
            ],
            "gate_criteria": [
                {
                    "gate": gate_name,
                    "criterion": criterion,
                    **values,
                }
                for gate_name in ("graph_gate", "representation_gate")
                for criterion, values in gates[gate_name].get("criteria", {}).items()
            ],
            "categorical_descriptive_comparison": categorical_rows,
        },
    }


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _gate_decision(gate: Mapping[str, Any]) -> str:
    if gate.get("assessable") is not True:
        return "NOT ASSESSABLE"
    return "PASS" if gate.get("passes") is True else "FAIL"


def _markdown_report(result: Mapping[str, Any]) -> str:
    graph = _mapping(result.get("graph_gate"), "graph gate")
    representation = _mapping(result.get("representation_gate"), "representation gate")
    tables_value = result.get("tables")
    tables = tables_value if isinstance(tables_value, Mapping) else {}
    coverage_value = result.get("coverage")
    coverage = coverage_value if isinstance(coverage_value, Mapping) else {}
    pilot_value = result.get("pilot_gate")
    pilot = pilot_value if isinstance(pilot_value, Mapping) else {}
    lines = [
        "# Ten-core hybrid raw-count GAT comparison",
        "",
        (
            f"Status: **{result['status']}**. This is an exploratory, held-in "
            "masked-expression study of ten pathology-confirmed adjacent-normal "
            "cores. Three fixed masks are averaged within mode/core; the ten "
            "core aliases receive equal weight."
        ),
        "",
        "## Frozen model and objective",
        "",
        "Counts use eight output states: `0`, `1`, `2`, `3`, `4–7`, `8–15`, `16–31`, and `>=32`; token 8 is input-only `MASK`. The encoder combines gene-specific token projections, exact per-gene standardized `log1p(raw count)`, and the permitted morphology/imaging covariates. Masking replaces the token and sets the continuous input to standardized zero.",
        "",
        "The graph arm is an exact receiver-partitioned edge-conditioned GATv2 (width 512, four heads, two graph layers, FFN 512, decoder 512) over the exact mutual `k=1000` graph and 17 geometry attributes. The matched-self arm has the same encoder, decoder, and trainable parameter count but receives no graph, neighbor, coordinate, or edge input.",
        "",
        "The fixed objective is the equal mean of balanced detection BCE, balanced six-threshold positive ordinal BCE, and positive-only standardized-`log1p` Huber loss (`delta=1.0`). Whole-node masking is the primary estimand.",
        "",
        "## Evidence coverage",
        "",
        f"Registry slots: {_format(coverage.get('selected_completed_run_count'))}/20 uniquely completed; pilot receipt verified: {_format(pilot.get('verified'))}; analysis ready: {_format(result.get('analysis_ready'))}.",
        (
            "Execution audit: "
            f"{len(coverage.get('failed_attempts', []))} failed/stale run or queue records; "
            f"{len(coverage.get('duplicate_completed_slots', []))} duplicate completed slots; "
            f"{len(coverage.get('invalid_production_attempts', []))} invalid production attempts. "
            "Attempt-2 recoveries remain eligible when they are the sole valid completed run for a slot; earlier failures remain visible in the audit tables."
        ),
        "",
        "## Frozen gate decisions",
        "",
        "| Gate | Assessable | Decision |",
        "|---|:---:|:---:|",
        f"| Graph gain | {_format(graph.get('assessable'))} | {_gate_decision(graph)} |",
        f"| Representation | {_format(representation.get('assessable'))} | {_gate_decision(representation)} |",
        "",
    ]
    for title, gate in (("Graph gate", graph), ("Representation gate", representation)):
        lines.extend([f"### {title}", "", "| Criterion | Observed | Threshold | Pass |", "|---|---:|---:|:---:|"])
        if not gate.get("criteria"):
            lines.append(f"| Not assessable | {gate.get('reason', 'unavailable')} | NA | no |")
        for name, values in gate.get("criteria", {}).items():
            lines.append(
                f"| {name.replace('_', ' ')} | {_format(values.get('observed'))} | "
                f"{_format(values.get('threshold'))} | {_format(values.get('passes'))} |"
            )
        lines.append("")
    graph_rows = graph.get("core_rows", [])
    if graph_rows:
        lines.extend(
            [
                "## Whole-node paired core results",
                "",
                "| Alias | GAT loss | Self loss | Relative gain (%) | GAT favored |",
                "|---|---:|---:|---:|:---:|",
            ]
        )
        for row in graph_rows:
            lines.append(
                f"| {row['core_alias']} | {_format(row['gat_hybrid_loss'], 6)} | "
                f"{_format(row['self_hybrid_loss'], 6)} | "
                f"{_format(100.0 * row['relative_hybrid_loss_improvement'])} | "
                f"{_format(row['gat_favors_total_loss'])} |"
            )
        lines.append("")
    representation_rows = representation.get("core_rows", [])
    if representation_rows:
        lines.extend(
            [
                "## Whole-node representation results",
                "",
                "| Alias | Ordinal MAE gain vs reference (%) | Continuous Huber gain vs reference (%) | Detection balanced-accuracy difference |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in representation_rows:
            lines.append(
                f"| {row['core_alias']} | "
                f"{_format(100.0 * row['relative_positive_ordinal_mae_improvement'])} | "
                f"{_format(100.0 * row['relative_positive_continuous_huber_improvement'])} | "
                f"{_format(row['detection_balanced_accuracy_difference'], 6)} |"
            )
        lines.append("")
    component = graph.get("component_inference", {})
    if component:
        lines.extend(
            [
                "## Exact paired component inference",
                "",
                "Raw differences are matched-self minus GAT. Each test enumerates all `2^10` sign flips; Holm correction covers exactly the three frozen loss components.",
                "",
                "| Component | Mean raw difference | Raw p | Holm p | Reject |",
                "|---|---:|---:|---:|:---:|",
            ]
        )
        for name in _COMPONENT_METRICS:
            value = component[name]
            lines.append(
                f"| {name} | {_format(value['statistic'], 6)} | "
                f"{_format(value['p_value'], 6)} | "
                f"{_format(value['holm']['holm_adjusted_p_value'], 6)} | "
                f"{_format(value['holm']['reject'])} |"
            )
        lines.append("")
    categorical = _mapping(result.get("prior_categorical_comparison"), "prior comparison")
    lines.extend(["## Descriptive prior categorical comparison", ""])
    if categorical.get("available"):
        lines.append(
            "The prior `0/1/2/>=3` study fit three model seeds on one true-Normal core, whereas the current study uses ten pathology-confirmed adjacent-normal cores. It is shown only for descriptive context; the observations are not exchangeable."
        )
        lines.append(
            "The current eight-state hybrid outputs are collapsed to the common four states only for this display. The prior model directly predicted four categorical states and lacked the current exact-value continuous channel and hurdle/ordinal objective, so numerical differences cannot be attributed to tokenization alone."
        )
        categorical_rows = tables.get("categorical_descriptive_comparison", [])
        if categorical_rows:
            lines.extend(
                [
                    "",
                    "| Source | Variant | Exact (%) | Balanced (%) | Positive exact (%) | All-zero exact (%) |",
                    "|---|---|---:|---:|---:|---:|",
                ]
            )
            for row in categorical_rows:
                lines.append(
                    f"| {row.get('source')} | {row.get('variant')} | "
                    f"{_format(row.get('exact_accuracy_percent'))} | "
                    f"{_format(row.get('balanced_accuracy_percent'))} | "
                    f"{_format(row.get('positive_exact_accuracy_percent'))} | "
                    f"{_format(row.get('all_zero_exact_accuracy_percent'))} |"
                )
    else:
        lines.append(f"Prior JSON unavailable: {categorical.get('reason', 'unknown')}.")
    resource_rows = tables.get("run_resources_and_convergence", [])
    if resource_rows:
        lines.extend(
            [
                "",
                "## Runtime and convergence",
                "",
                "| Alias | Arm | Parameters | Runtime (h) | Peak VRAM (GiB) | Final train loss | Last-20 slope | Complete/finite |",
                "|---|---|---:|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in resource_rows:
            lines.append(
                f"| {row['core_alias']} | {row['arm']} | {row['parameter_count']} | "
                f"{_format(float(row['duration_seconds']) / 3600.0, 3)} | "
                f"{_format(row['peak_allocated_vram_gib'], 3)} | "
                f"{_format(row['final_train_hybrid_loss'], 6)} | "
                f"{_format(row['last_20_epoch_loss_slope'], 8)} | "
                f"{_format(bool(row['all_epochs_completed']) and bool(row['all_losses_and_gradients_finite']))} |"
            )
    lines.extend(["", "## Negative evidence", ""])
    negative = result.get("negative_evidence", [])
    lines.extend(f"- {item}" for item in negative)
    if not negative:
        lines.append("- No frozen gate failure was observed; this does not remove the design limitations below.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.extend(
        [
            "",
            "Maximum defensible conclusion: "
            + str(result["maximum_defensible_conclusion"])
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    head = "".join(f"<th scope=\"col\">{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_format(item))}</td>" for item in row)
        + "</tr>"
        for row in rows
    )
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _gain_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "<p>No assessable core-level graph gains.</p>"
    width, height = 760, 300
    margin_left, baseline = 88, 660
    values = [100.0 * float(row["relative_hybrid_loss_improvement"]) for row in rows]
    limit = max(2.0, max(abs(value) for value in values))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Relative whole-node hybrid-loss improvement for each of ten adjacent-normal core aliases">',
        "<title>Per-core relative hybrid-loss improvement; positive values favor GAT</title>",
        f'<line x1="{margin_left}" y1="20" x2="{margin_left}" y2="280" stroke="#59636e"/>',
    ]
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        y = 30 + 25 * index
        length = 540.0 * value / (2.0 * limit)
        zero = margin_left + 270
        x = zero if length >= 0 else zero + length
        color = "#1b7f5a" if value > 0 else "#a33a3a"
        parts.extend(
            [
                f'<text x="4" y="{y + 10}" font-size="12">{html.escape(str(row["core_alias"]))}</text>',
                f'<line x1="{zero}" y1="{y - 5}" x2="{zero}" y2="{y + 12}" stroke="#9aa3ac"/>',
                f'<rect x="{x:.2f}" y="{y}" width="{abs(length):.2f}" height="10" fill="{color}"/>',
                f'<text x="{baseline}" y="{y + 10}" font-size="11">{value:.2f}%</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def _html_report(result: Mapping[str, Any]) -> str:
    graph = _mapping(result.get("graph_gate"), "graph gate")
    representation = _mapping(result.get("representation_gate"), "representation gate")
    tables_value = result.get("tables")
    tables = tables_value if isinstance(tables_value, Mapping) else {}
    coverage_value = result.get("coverage")
    coverage = coverage_value if isinstance(coverage_value, Mapping) else {}
    pilot_value = result.get("pilot_gate")
    pilot = pilot_value if isinstance(pilot_value, Mapping) else {}
    graph_rows = graph.get("core_rows", [])
    gate_rows = [
        (
            gate_name.replace("_", " "),
            criterion.replace("_", " "),
            values.get("observed"),
            values.get("threshold"),
            values.get("passes"),
        )
        for gate_name, gate in (("graph_gate", graph), ("representation_gate", representation))
        for criterion, values in gate.get("criteria", {}).items()
    ]
    core_rows = [
        (
            row["core_alias"],
            row["gat_hybrid_loss"],
            row["self_hybrid_loss"],
            100.0 * row["relative_hybrid_loss_improvement"],
            row["gat_favors_total_loss"],
        )
        for row in graph_rows
    ]
    representation_rows = [
        (
            row["core_alias"],
            100.0 * row["relative_positive_ordinal_mae_improvement"],
            100.0 * row["relative_positive_continuous_huber_improvement"],
            row["detection_balanced_accuracy_difference"],
        )
        for row in representation.get("core_rows", [])
    ]
    component_rows = [
        (
            metric,
            graph["component_inference"][metric]["statistic"],
            graph["component_inference"][metric]["p_value"],
            graph["component_inference"][metric]["holm"][
                "holm_adjusted_p_value"
            ],
            graph["component_inference"][metric]["holm"]["reject"],
        )
        for metric in _COMPONENT_METRICS
        if metric in graph.get("component_inference", {})
    ]
    categorical_rows = [
        (
            row.get("source"),
            row.get("variant"),
            row.get("exact_accuracy_percent"),
            row.get("balanced_accuracy_percent"),
            row.get("positive_exact_accuracy_percent"),
            row.get("all_zero_exact_accuracy_percent"),
        )
        for row in tables.get("categorical_descriptive_comparison", [])
    ]
    resource_rows = [
        (
            row["core_alias"],
            row["arm"],
            row["parameter_count"],
            float(row["duration_seconds"]) / 3600.0,
            row["peak_allocated_vram_gib"],
            row["final_train_hybrid_loss"],
            row["last_20_epoch_loss_slope"],
            bool(row["all_epochs_completed"])
            and bool(row["all_losses_and_gradients_finite"]),
        )
        for row in tables.get("run_resources_and_convergence", [])
    ]
    negative = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in result.get("negative_evidence", [])
    ) or "<li>No frozen gate failure observed; limitations still apply.</li>"
    limitations = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in result["limitations"]
    )
    graph_decision = _gate_decision(graph)
    representation_decision = _gate_decision(representation)
    graph_class = (
        "muted"
        if graph_decision == "NOT ASSESSABLE"
        else ("pass" if graph_decision == "PASS" else "fail")
    )
    representation_class = (
        "muted"
        if representation_decision == "NOT ASSESSABLE"
        else ("pass" if representation_decision == "PASS" else "fail")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ten-core hybrid raw-count GAT comparison</title>
<style>
:root{{--ink:#18212a;--muted:#59636e;--paper:#fff;--panel:#f4f7f8;--line:#cad2d8;--pass:#176b4d;--fail:#943737}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}}main{{max-width:1100px;margin:auto;padding:32px}}h1,h2{{line-height:1.2}}.lede{{font-size:1.05rem}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}}.pass{{color:var(--pass);font-weight:700}}.fail{{color:var(--fail);font-weight:700}}.muted{{color:var(--muted);font-weight:700}}.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}th,td{{border:1px solid var(--line);padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}svg{{width:100%;height:auto;background:#fff;border:1px solid var(--line)}}code{{background:var(--panel);padding:.1em .25em}}@media print{{main{{max-width:none;padding:0}}.card{{break-inside:avoid}}table{{font-size:10pt}}svg{{max-height:230px}}}}
</style></head><body><main>
<h1>Ten-core hybrid raw-count GAT comparison</h1>
<p class="lede">Exploratory held-in reconstruction across ten pathology-confirmed adjacent-normal cores. Three technical masks are averaged within each core; cores receive equal weight.</p>
<div class="cards"><div class="card"><strong>Analysis status</strong><br>{html.escape(str(result['status']))}</div><div class="card"><strong>Verified slots</strong><br>{html.escape(_format(coverage.get('selected_completed_run_count')))}/20<br>Pilot receipt: {html.escape(_format(pilot.get('verified')))}</div><div class="card"><strong>Graph gate</strong><br><span class="{graph_class}">{graph_decision}</span></div><div class="card"><strong>Representation gate</strong><br><span class="{representation_class}">{representation_decision}</span></div></div>
<p><strong>Execution audit:</strong> {len(coverage.get('failed_attempts', []))} failed/stale run or queue records; {len(coverage.get('duplicate_completed_slots', []))} duplicate completed slots; {len(coverage.get('invalid_production_attempts', []))} invalid production attempts. Attempt-2 recoveries remain eligible only when they are the sole valid completion for a slot; prior failures remain in the complete audit tables.</p>
<h2>Frozen model and objective</h2><p>Counts use eight output states: <code>0</code>, <code>1</code>, <code>2</code>, <code>3</code>, <code>4–7</code>, <code>8–15</code>, <code>16–31</code>, and <code>&gt;=32</code>; token 8 is input-only <code>MASK</code>. The encoder combines gene-specific token projections, exact per-gene standardized <code>log1p(raw count)</code>, and the permitted morphology/imaging covariates. Masking replaces the token and sets the continuous input to standardized zero.</p><p>The graph arm is an exact receiver-partitioned edge-conditioned GATv2 (width 512, four heads, two graph layers, FFN 512, decoder 512) over the exact mutual <code>k=1000</code> graph and 17 geometry attributes. The matched-self arm has the same encoder, decoder, and parameter count but no graph, neighbor, coordinate, or edge input. The objective equally averages balanced detection BCE, balanced six-threshold ordinal BCE, and positive-only standardized-log1p Huber loss (<code>delta=1.0</code>).</p>
<h2>Frozen criteria</h2>{_html_table(('Gate','Criterion','Observed','Threshold','Pass'), gate_rows) if gate_rows else '<p>Gates are not assessable because exact verified coverage is unavailable.</p>'}
<h2>Whole-node graph comparison</h2>{_gain_svg(graph_rows)}{_html_table(('Alias','GAT loss','Self loss','Relative gain (%)','GAT favored'), core_rows) if core_rows else ''}
<h2>Whole-node representation comparison</h2>{_html_table(('Alias','Ordinal MAE gain vs ref (%)','Continuous Huber gain vs ref (%)','Detection balanced-accuracy difference'), representation_rows) if representation_rows else '<p>No assessable representation comparison.</p>'}
<h2>Exact paired component inference</h2><p>Raw differences are matched-self minus GAT. Each test enumerates all <code>2^10</code> sign flips; Holm correction covers exactly three frozen loss components.</p>{_html_table(('Component','Mean raw difference','Raw p','Holm p','Reject'), component_rows) if component_rows else '<p>No assessable component inference.</p>'}
<h2>Negative evidence</h2><ul>{negative}</ul>
<h2>Descriptive prior comparison</h2><p>The prior <code>0/1/2/&gt;=3</code> study fit three model seeds on one true-Normal core, whereas the current study uses ten pathology-confirmed adjacent-normal cores. The observations are descriptive and not exchangeable. Current eight-state outputs are collapsed to four states only for this display; the prior model directly predicted four categorical states and lacked the current exact-value continuous channel and hurdle/ordinal objective. Numerical differences therefore cannot be attributed to tokenization alone.</p>{_html_table(('Source','Variant','Exact (%)','Balanced (%)','Positive exact (%)','All-zero exact (%)'), categorical_rows) if categorical_rows else '<p>Prior categorical evidence unavailable.</p>'}
<h2>Runtime and convergence</h2>{_html_table(('Alias','Arm','Parameters','Runtime (h)','Peak VRAM (GiB)','Final train loss','Last-20 slope','Complete/finite'), resource_rows) if resource_rows else '<p>Verified production resource evidence unavailable.</p>'}
<h2>Limitations</h2><ul>{limitations}</ul>
<p><strong>Maximum defensible conclusion:</strong> {html.escape(str(result['maximum_defensible_conclusion']))}.</p>
</main></body></html>"""


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(str(key))
                seen.add(str(key))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _rename_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise HybridCountComparisonError(
            "platform lacks renameat2; refusing non-atomic publication"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    status = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if status == 0:
        return
    number = ctypes.get_errno()
    if number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise HybridCountComparisonError(
            f"output directory already exists: {destination}"
        )
    raise OSError(number, os.strerror(number), destination.as_posix())


def write_comparison(result: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Publish complete tables plus Markdown and portable HTML atomically."""

    destination = Path(output_dir).resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise HybridCountComparisonError(
            f"output directory already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.writing-", dir=destination.parent
        )
    )
    try:
        (temporary / "comparison.json").write_text(
            json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        (temporary / "report.md").write_text(
            _markdown_report(result), encoding="utf-8"
        )
        (temporary / "report.html").write_text(
            _html_report(result), encoding="utf-8"
        )
        tables = _mapping(result.get("tables"), "result tables")
        for name, rows in tables.items():
            if not isinstance(rows, list):
                raise HybridCountComparisonError(f"table {name} must be a list")
            _write_csv(temporary / f"{name}.csv", rows)
        files = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        (temporary / "report_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign_id": _CAMPAIGN_ID,
                    "files": files,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _rename_no_replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and compare the frozen ten-core hybrid-count campaign."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument(
        "--categorical-json",
        type=Path,
        default=_PRIOR_CATEGORICAL_DEFAULT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = (
        ProjectPaths.from_environment({"BAGM_ROOT": str(args.root)})
        if args.root is not None
        else current_paths(anchor=__file__)
    )
    registry_path = args.registry or paths.state_root / "tracking/bagm.sqlite3"
    if not registry_path.is_file():
        print("comparison failed: registry does not exist", file=sys.stderr)
        return 2
    registry = Registry(registry_path, initialize=False)
    try:
        result = compare_hybrid_count_multicore(
            registry,
            paths,
            categorical_json=args.categorical_json,
        )
        output = write_comparison(result, args.output_dir)
    except HybridCountComparisonError as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
