#!/usr/bin/env python3
"""Audit both fixed self-hurdle science runs and render one deterministic report.

The registry, locked materialization, resource gate, and immutable run bundles
jointly define the analysis population.  The script refuses missing or duplicate
science slots and never selects a favorable run or masking estimand.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


CAMPAIGN_ID = "cmp_20260729_self_hurdle_full_core_capacity"
ALIASES = ("ANC-03", "ANC-05")
STAGES = ("resource", "science")
MASK_MODES = ("whole_node", "partial_gene", "spatial_block")
EXPECTED_REPLICATES = 3
EXPECTED_PARAMETER_COUNT = 16_917_200
EXPECTED_SCIENCE_EPOCHS = 200
CONTRACT_SHA256 = (
    "26cf4f094d843c4fa9020c52e1d45988f8e4a0c43f20beb186e4742b2dacba00"
)
MATERIALIZATION_KIND = "self_hurdle_locked_config_materialization_v1"
RESOURCE_GATE_KIND = "self_hurdle_resource_gate_v1"
ENQUEUE_KINDS = {
    "resource": "self_hurdle_resource_enqueue_v1",
    "science": "self_hurdle_science_enqueue_v1",
}
SCIENCE_PEAK_VRAM_GIB_MAX = 20.5
DISK_USED_DECIMAL_GB_MAX = 55.0
PREFERRED_GPU_HOURS = 12.0
ABSOLUTE_GPU_HOURS = 24.0
PRIMARY_METRIC = "fit/whole_node/hurdle_loss"
ACCURACY_FIELDS = (
    "detection_balanced_accuracy",
    "detection_sensitivity",
    "detection_specificity",
    "detection_precision",
    "positive_count_state_exact_accuracy",
    "positive_count_state_within_one_accuracy",
    "state8_exact_accuracy",
    "state8_balanced_accuracy",
)
ERROR_FIELDS = (
    "hurdle_loss",
    "detection_bce",
    "positive_continuous_huber",
    "positive_continuous_mae",
    "positive_count_state_mae",
    "reconstructed_count_log1p_mae",
)


class SelfHurdleAnalysisError(RuntimeError):
    """Raised when coverage, provenance, or immutable evidence is ambiguous."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    output: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in output:
            raise SelfHurdleAnalysisError(
                f"YAML contains duplicate key {key!r}"
            )
        output[key] = loader.construct_object(value_node, deep=deep)
    return output


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class RegistrySlot:
    alias: str
    stage: str
    expected_config_sha256: str
    job_id: str
    requested_gpu: str
    run_id: str
    artifact_path: Path
    registry_duration_seconds: float
    run_row: Mapping[str, Any]
    config: Mapping[str, Any]


@dataclass(frozen=True)
class RunEvidence:
    alias: str
    run_id: str
    root: Path
    config_sha256: str
    config_file_sha256: str
    checkpoint_sha256: str
    success_marker_sha256: str
    bundle_content_sha256: str
    metrics_sha256: str
    summary_sha256: str
    checkpoint_catalog_verified: bool
    summary: Mapping[str, Any]
    metrics: Mapping[str, Any]
    resource: Mapping[str, Any]
    hardware: Mapping[str, Any]
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]]
    gate: Mapping[str, Any]
    registry_duration_seconds: float
    requested_gpu: str
    dataset_fingerprint: str
    split_fingerprint: str
    mask_bundle_sha256: str
    per_gene_reference_sha256: str


@dataclass(frozen=True)
class ResourceEvidence:
    alias: str
    run_id: str
    root: Path
    config_sha256: str
    checkpoint_sha256: str
    success_marker_sha256: str
    bundle_content_sha256: str
    checkpoint_catalog_verified: bool
    resource: Mapping[str, Any]
    registry_duration_seconds: float
    requested_gpu: str


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelfHurdleAnalysisError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SelfHurdleAnalysisError(f"{label} must be a list")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelfHurdleAnalysisError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SelfHurdleAnalysisError(f"{label} must be finite numeric")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelfHurdleAnalysisError(f"{label} must be an integer")
    return value


def _sha(value: Any, label: str) -> str:
    text = str(value or "")
    if (
        len(text) != 64
        or text.lower() != text
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise SelfHurdleAnalysisError(
            f"{label} must be a lowercase SHA-256"
        )
    return text


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_text(text: str, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise SelfHurdleAnalysisError(
            f"{label} contains non-finite JSON constant {value}"
        )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise SelfHurdleAnalysisError(
                    f"{label} contains duplicate key {key!r}"
                )
            output[key] = value
        return output

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique,
        )
    except SelfHurdleAnalysisError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SelfHurdleAnalysisError(f"{label} is invalid JSON") from error


def _strict_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = _strict_json_text(path.read_text(encoding="utf-8"), label)
    except (OSError, UnicodeError) as error:
        raise SelfHurdleAnalysisError(f"{label} is unreadable") from error
    return _mapping(value, label)


def _strict_yaml(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeyLoader,
        )
    except SelfHurdleAnalysisError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SelfHurdleAnalysisError(f"{label} is invalid YAML") from error
    return _mapping(value, label)


def _verified_receipt(path: Path, label: str) -> Mapping[str, Any]:
    receipt = dict(_strict_json(path, label))
    observed = _sha(receipt.pop("checksum", None), f"{label} checksum")
    if canonical_sha256(receipt) != observed:
        raise SelfHurdleAnalysisError(f"{label} checksum does not verify")
    receipt["checksum"] = observed
    return receipt


def _inside_project(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SelfHurdleAnalysisError(f"{label} path is missing")
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as error:
        raise SelfHurdleAnalysisError(
            f"{label} escapes the project root"
        ) from error
    return path


def _expected_jobs(
    materialization: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for stage in STAGES:
        rows = _list(
            materialization.get(f"{stage}_jobs"),
            f"{stage} materialization jobs",
        )
        for raw in rows:
            row = _mapping(raw, f"{stage} materialization job")
            alias = str(row.get("alias"))
            key = (stage, alias)
            if (
                alias not in ALIASES
                or row.get("stage") != stage
                or row.get("arm") != "self-hurdle"
                or key in expected
            ):
                raise SelfHurdleAnalysisError(
                    f"invalid or duplicate materialization slot {key}"
                )
            _sha(row.get("config_sha256"), f"{key} config checksum")
            expected[key] = row
    required = {(stage, alias) for stage in STAGES for alias in ALIASES}
    if set(expected) != required:
        raise SelfHurdleAnalysisError(
            "locked materialization does not contain the exact 2x2 job matrix"
        )
    return expected


def validate_locked_receipts(
    *,
    project_root: Path,
    materialization_path: Path,
    resource_gate_path: Path,
    resource_enqueue_path: Path,
    science_enqueue_path: Path,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[tuple[str, str], Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    contract_path = (
        project_root
        / "experiments/campaigns"
        / CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    if _sha_file(contract_path) != CONTRACT_SHA256:
        raise SelfHurdleAnalysisError("frozen task contract checksum drifted")
    contract = _strict_yaml(contract_path, "frozen task contract")
    if (
        contract.get("campaign_id") != CAMPAIGN_ID
        or contract.get("frozen_before_new_training") is not True
    ):
        raise SelfHurdleAnalysisError("frozen task contract identity is invalid")

    materialization = _verified_receipt(
        materialization_path, "locked materialization"
    )
    frozen = _mapping(
        materialization.get("frozen_contract"),
        "materialization frozen contract",
    )
    parameters = _mapping(
        materialization.get("parameter_audit"),
        "materialization parameter audit",
    )
    graph = _mapping(
        materialization.get("graph_contract"),
        "materialization graph contract",
    )
    if (
        materialization.get("receipt_kind") != MATERIALIZATION_KIND
        or materialization.get("campaign_id") != CAMPAIGN_ID
        or frozen.get("sha256") != CONTRACT_SHA256
        or parameters.get("trainable_parameter_count")
        != EXPECTED_PARAMETER_COUNT
        or parameters.get("uses_graph_inputs") is not False
        or parameters.get("uses_edge_inputs") is not False
        or graph.get("construction_performed") is not False
        or graph.get("model_graph_inputs") is not False
        or graph.get("model_edge_inputs") is not False
        or graph.get("expected_directed_edges") != 0
        or materialization.get("training_performed") is not False
    ):
        raise SelfHurdleAnalysisError(
            "locked materialization violates the graphless contract"
        )
    expected = _expected_jobs(materialization)

    gate = _verified_receipt(resource_gate_path, "resource gate")
    gate_jobs = {
        str(_mapping(row, "resource gate job").get("alias")): _mapping(
            row, "resource gate job"
        )
        for row in _list(gate.get("jobs"), "resource gate jobs")
    }
    if (
        gate.get("receipt_kind") != RESOURCE_GATE_KIND
        or gate.get("campaign_id") != CAMPAIGN_ID
        or gate.get("materialization_checksum")
        != materialization.get("checksum")
        or gate.get("complete") is not True
        or gate.get("passed") is not True
        or gate.get("science_authorized") is not True
        or set(gate_jobs) != set(ALIASES)
    ):
        raise SelfHurdleAnalysisError(
            "resource gate did not authorize the exact science campaign"
        )
    for alias in ALIASES:
        row = gate_jobs[alias]
        if (
            row.get("passed") is not True
            or row.get("verified_bundle") is not True
            or row.get("config_sha256")
            != expected[("resource", alias)].get("config_sha256")
            or row.get("graph_construction_count") != 0
            or row.get("graph_input_tensor_count") != 0
            or row.get("edge_input_tensor_count") != 0
        ):
            raise SelfHurdleAnalysisError(
                f"resource gate evidence is invalid for {alias}"
            )

    enqueue: dict[str, Mapping[str, Any]] = {}
    for stage, path in (
        ("resource", resource_enqueue_path),
        ("science", science_enqueue_path),
    ):
        receipt = _verified_receipt(path, f"{stage} enqueue receipt")
        if (
            receipt.get("receipt_kind") != ENQUEUE_KINDS[stage]
            or receipt.get("campaign_id") != CAMPAIGN_ID
            or receipt.get("stage") != stage
            or receipt.get("complete") is not True
            or receipt.get("materialization_checksum")
            != materialization.get("checksum")
        ):
            raise SelfHurdleAnalysisError(
                f"{stage} enqueue receipt identity is invalid"
            )
        jobs = _list(receipt.get("jobs"), f"{stage} enqueue jobs")
        observed = {
            (
                str(_mapping(row, f"{stage} enqueue job").get("alias")),
                str(
                    _mapping(row, f"{stage} enqueue job").get(
                        "config_sha256"
                    )
                ),
            )
            for row in jobs
        }
        expected_pairs = {
            (
                alias,
                str(expected[(stage, alias)].get("config_sha256")),
            )
            for alias in ALIASES
        }
        if observed != expected_pairs or len(jobs) != len(ALIASES):
            raise SelfHurdleAnalysisError(
                f"{stage} enqueue receipt does not bind both locked jobs"
            )
        enqueue[stage] = receipt
    return materialization, gate, expected, enqueue


def _config_semantics(
    config: Mapping[str, Any],
    *,
    alias: str,
    stage: str,
) -> None:
    campaign = _mapping(config.get("campaign"), "campaign config")
    experiment = _mapping(config.get("experiment"), "experiment config")
    classification = _mapping(
        config.get("classification"), "classification config"
    )
    dataset = _mapping(config.get("dataset"), "dataset config")
    model = _mapping(config.get("model"), "model config")
    graph = _mapping(config.get("graph"), "graph config")
    features = _mapping(config.get("features"), "features config")
    trainer = _mapping(config.get("trainer"), "trainer config")
    evaluation = _mapping(config.get("evaluation"), "evaluation config")
    pilot = stage == "resource"
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("frozen_contract_sha256") != CONTRACT_SHA256
        or experiment.get("biological_unit_alias") != alias
        or experiment.get("arm") != "self-hurdle"
        or experiment.get("resource_pilot") is not pilot
        or experiment.get("conclusion_eligible") is pilot
        or experiment.get("graph_arms_authorized") is not False
        or classification.get("lifecycle_stage")
        != ("diagnostic" if pilot else "exploratory_screen")
        or dataset.get("biological_unit_alias") != alias
        or dataset.get("validation_or_test_partition_present") is not False
        or dataset.get("patient_generalization_supported") is not False
        or model.get("name") != "self-hurdle-count"
        or model.get("family") != "self_only_hurdle_count"
        or model.get("uses_graph_inputs") is not False
        or model.get("uses_edge_inputs") is not False
        or model.get("expected_trainable_parameter_count")
        != EXPECTED_PARAMETER_COUNT
        or graph.get("enabled") is not False
        or graph.get("construction_performed") is not False
        or graph.get("expected_directed_edges") != 0
        or features.get("use_edge_features") is not False
        or trainer.get("max_epochs") != (2 if pilot else 200)
        or trainer.get("fixed_epoch_budget") is not True
        or trainer.get("early_stopping") is not False
        or trainer.get("restore_best") is not False
        or trainer.get("checkpoint_policy") != "last_only"
        or trainer.get("graph_execution") != "none_graph_inputs_prohibited"
        or evaluation.get("splits") != ["fit"]
        or evaluation.get("generalization_estimate") is not False
        or evaluation.get("validation_or_test_selection") is not False
        or evaluation.get("mask_modes")
        != ["partial_gene", "whole_node", "spatial_block"]
        or evaluation.get("mask_replicates_per_mode")
        != (1 if pilot else EXPECTED_REPLICATES)
        or config.get("seed") != 0
        or config.get("fold") != 0
    ):
        raise SelfHurdleAnalysisError(
            f"resolved config violates frozen semantics for {stage}/{alias}"
        )


def registry_inventory(
    *,
    registry: Registry,
    project_root: Path,
    expected: Mapping[tuple[str, str], Mapping[str, Any]],
    enqueue: Mapping[str, Mapping[str, Any]],
) -> tuple[
    Mapping[tuple[str, str], RegistrySlot],
    list[Mapping[str, Any]],
]:
    with registry.connect() as connection:
        queue_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT q.*, r.status AS run_status,
                       r.artifact_path, r.duration_seconds,
                       r.primary_metric_name, r.primary_metric_value,
                       r.parameter_count, r.config_json AS run_config_json,
                       r.failure_category AS run_failure_category,
                       r.start_time, r.end_time
                FROM queue_jobs q
                LEFT JOIN runs r ON r.run_id = q.run_id
                WHERE q.campaign_id = ?
                ORDER BY q.created_at, q.job_id
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
        ]
        all_run_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT run_id, status, retry_of, failure_category, attempt,
                       start_time, end_time, config_json
                FROM runs
                WHERE campaign_id = ?
                ORDER BY created_at, run_id
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
        ]
    expected_job_ids = {
        str(_mapping(row, "enqueue job").get("job_id")): (stage, alias)
        for stage in STAGES
        for row in _list(
            enqueue[stage].get("jobs"), f"{stage} enqueue jobs"
        )
        for alias in [str(_mapping(row, "enqueue job").get("alias"))]
    }
    observed_job_ids = {str(row.get("job_id")) for row in queue_rows}
    if (
        observed_job_ids != set(expected_job_ids)
        or len(queue_rows) != len(expected_job_ids)
    ):
        raise SelfHurdleAnalysisError(
            "registry queue coverage differs from the four locked jobs"
        )

    slots: dict[tuple[str, str], RegistrySlot] = {}
    for row in queue_rows:
        job_id = str(row.get("job_id"))
        stage, alias = expected_job_ids[job_id]
        expected_sha = _sha(
            expected[(stage, alias)].get("config_sha256"),
            f"{stage}/{alias} expected config checksum",
        )
        config = _mapping(
            _strict_json_text(
                str(row.get("canonical_config_json")),
                f"{stage}/{alias} queue config",
            ),
            f"{stage}/{alias} queue config",
        )
        _config_semantics(config, alias=alias, stage=stage)
        run_config = _mapping(
            _strict_json_text(
                str(row.get("run_config_json")),
                f"{stage}/{alias} registered run config",
            ),
            f"{stage}/{alias} registered run config",
        )
        if config != run_config or canonical_sha256(config) != expected_sha:
            raise SelfHurdleAnalysisError(
                f"registered config drift for {stage}/{alias}"
            )
        run_id = str(row.get("run_id") or "")
        if (
            row.get("status") != "completed"
            or row.get("run_status") != "completed"
            or not run_id
            or row.get("run_failure_category") is not None
            or row.get("primary_metric_name") != PRIMARY_METRIC
            or row.get("parameter_count") != EXPECTED_PARAMETER_COUNT
            or row.get("end_time") is None
        ):
            raise SelfHurdleAnalysisError(
                f"{stage}/{alias} is not one completed registered run"
            )
        artifact = _inside_project(
            project_root,
            row.get("artifact_path"),
            f"{stage}/{alias} artifact",
        )
        duration = _finite(
            row.get("duration_seconds"),
            f"{stage}/{alias} registered duration",
        )
        key = (stage, alias)
        if key in slots:
            raise SelfHurdleAnalysisError(f"duplicate registry slot {key}")
        slots[key] = RegistrySlot(
            alias=alias,
            stage=stage,
            expected_config_sha256=expected_sha,
            job_id=job_id,
            requested_gpu=str(row.get("requested_gpu")),
            run_id=run_id,
            artifact_path=artifact,
            registry_duration_seconds=duration,
            run_row=row,
            config=config,
        )
    if set(slots) != set(expected):
        raise SelfHurdleAnalysisError(
            "registry does not contain the exact expected run matrix"
        )
    failed = [
        {
            "run_id": str(row["run_id"]),
            "status": str(row["status"]),
            "attempt": row["attempt"],
            "retry_of": row["retry_of"],
            "failure_category": row["failure_category"],
        }
        for row in all_run_rows
        if row.get("status") != "completed"
    ]
    return slots, failed


def _load_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise SelfHurdleAnalysisError(
                        f"{label} has blank line {line_number}"
                    )
                rows.append(
                    _mapping(
                        _strict_json_text(
                            line, f"{label} line {line_number}"
                        ),
                        f"{label} line {line_number}",
                    )
                )
    except (OSError, UnicodeError) as error:
        raise SelfHurdleAnalysisError(f"{label} is unreadable") from error
    return rows


def _validate_training_history(
    root: Path,
    *,
    alias: str,
    run_id: str,
    expected_epochs: int,
) -> None:
    rows = _load_jsonl(
        root / "metrics/history.jsonl", f"{alias} training history"
    )
    if (
        len(rows) != expected_epochs
        or [row.get("epoch") for row in rows]
        != list(range(expected_epochs))
        or any(
            row.get("run_id") != run_id
            or row.get("split") != "fit"
            or _integer(
                row.get("n_target_batches"),
                f"{alias} history target batches",
            )
            < 1
            for row in rows
        )
    ):
        raise SelfHurdleAnalysisError(
            f"{alias} training history does not contain the complete fixed budget"
        )
    for epoch, row in enumerate(rows):
        for field in (
            "duration_seconds",
            "gradient_norm",
            "train_detection_bce",
            "train_positive_continuous_huber",
            "train_hurdle_loss",
            "peak_cuda_memory_bytes",
        ):
            value = _finite(
                row.get(field), f"{alias} epoch {epoch} {field}"
            )
            if value < 0:
                raise SelfHurdleAnalysisError(
                    f"{alias} epoch {epoch} {field} cannot be negative"
                )


def _close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-10):
        raise SelfHurdleAnalysisError(
            f"{label} differs: observed={observed}, expected={expected}"
        )


def build_mode_comparisons(
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
    if len(rows) != len(MASK_MODES) * EXPECTED_REPLICATES:
        raise SelfHurdleAnalysisError(
            "science evaluation must contain exactly nine mask rows"
        )
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in MASK_MODES:
        selected = [row for row in rows if row.get("mask_mode") == mode]
        if (
            len(selected) != EXPECTED_REPLICATES
            or sorted(row.get("mask_replicate") for row in selected)
            != list(range(EXPECTED_REPLICATES))
            or any(row.get("split") != "fit" for row in selected)
        ):
            raise SelfHurdleAnalysisError(
                f"{mode} lacks exactly three fixed fit-mask replicates"
            )
        comparisons[mode] = {}
        for field in (*ACCURACY_FIELDS, *ERROR_FIELDS):
            model_key = f"fit/{mode}/{field}"
            reference_key = f"fit/{mode}/reference_per_gene_{field}"
            observed = _finite(metrics.get(model_key), model_key)
            reference = _finite(metrics.get(reference_key), reference_key)
            row_observed = [
                _finite(row.get(field), f"{mode}/{field} replicate")
                for row in selected
            ]
            row_reference = [
                _finite(
                    row.get(f"reference_per_gene_{field}"),
                    f"{mode}/reference/{field} replicate",
                )
                for row in selected
            ]
            _close(
                observed,
                sum(row_observed) / len(row_observed),
                f"{model_key} aggregate",
            )
            _close(
                reference,
                sum(row_reference) / len(row_reference),
                f"{reference_key} aggregate",
            )
            if field in ACCURACY_FIELDS:
                percent_key = f"{model_key}_percent"
                reference_percent_key = (
                    f"fit/{mode}/reference_per_gene_{field}_percent"
                )
                _close(
                    _finite(metrics.get(percent_key), percent_key),
                    100.0 * observed,
                    percent_key,
                )
                _close(
                    _finite(
                        metrics.get(reference_percent_key),
                        reference_percent_key,
                    ),
                    100.0 * reference,
                    reference_percent_key,
                )
                comparison = {
                    "direction": "maximize",
                    "model": observed,
                    "per_gene_reference": reference,
                    "model_percent": 100.0 * observed,
                    "per_gene_reference_percent": 100.0 * reference,
                    "difference_percentage_points": 100.0
                    * (observed - reference),
                    "relative_improvement_percent": None,
                }
            else:
                if reference <= 0:
                    raise SelfHurdleAnalysisError(
                        f"{reference_key} must be positive"
                    )
                comparison = {
                    "direction": "minimize",
                    "model": observed,
                    "per_gene_reference": reference,
                    "model_percent": None,
                    "per_gene_reference_percent": None,
                    "difference_percentage_points": None,
                    "relative_improvement_percent": 100.0
                    * (reference - observed)
                    / reference,
                }
            comparisons[mode][field] = comparison

        expected_gain = (
            comparisons[mode]["detection_balanced_accuracy"][
                "difference_percentage_points"
            ]
            / 100.0
        )
        stored_gain = _finite(
            metrics.get(
                f"fit/{mode}/"
                "detection_balanced_accuracy_gain_over_per_gene_reference"
            ),
            f"{mode} stored detection gain",
        )
        _close(stored_gain, expected_gain, f"{mode} detection gain")
        for field in (
            "hurdle_loss",
            "positive_continuous_huber",
            "positive_count_state_mae",
        ):
            stored = _finite(
                metrics.get(
                    f"fit/{mode}/{field}_relative_improvement_"
                    "over_per_gene_reference"
                ),
                f"{mode} stored {field} improvement",
            )
            replicate_improvements = [
                _finite(
                    row.get(
                        f"{field}_relative_improvement_"
                        "over_per_gene_reference"
                    ),
                    f"{mode}/{field} replicate improvement",
                )
                for row in selected
            ]
            expected_improvement = sum(replicate_improvements) / len(
                replicate_improvements
            )
            _close(
                stored,
                expected_improvement,
                f"{mode} {field} improvement",
            )
            comparisons[mode][field][
                "ratio_of_aggregate_means_relative_improvement_percent"
            ] = comparisons[mode][field]["relative_improvement_percent"]
            comparisons[mode][field][
                "relative_improvement_percent"
            ] = 100.0 * stored
    return comparisons


def evaluate_representation_gate(
    comparisons: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    whole = _mapping(comparisons.get("whole_node"), "whole-node comparisons")
    detection_gain = _finite(
        _mapping(
            whole.get("detection_balanced_accuracy"),
            "detection comparison",
        ).get("difference_percentage_points"),
        "detection gain",
    )
    huber_improvement = _finite(
        _mapping(
            whole.get("positive_continuous_huber"),
            "Huber comparison",
        ).get("relative_improvement_percent"),
        "Huber improvement",
    )
    state_improvement = _finite(
        _mapping(
            whole.get("positive_count_state_mae"),
            "state MAE comparison",
        ).get("relative_improvement_percent"),
        "state MAE improvement",
    )
    checks = {
        "positive_detection_balanced_accuracy_gain": detection_gain > 0.0,
        "positive_continuous_huber_improvement_at_least_2_percent": (
            huber_improvement >= 2.0
        ),
        "positive_count_state_mae_improvement_at_least_2_percent": (
            state_improvement >= 2.0
        ),
    }
    return {
        "thresholds": {
            "detection_balanced_accuracy_gain_percentage_points": "> 0",
            "positive_continuous_huber_relative_improvement_percent": ">= 2",
            "positive_count_state_mae_relative_improvement_percent": ">= 2",
        },
        "observed": {
            "detection_balanced_accuracy_gain_percentage_points": (
                detection_gain
            ),
            "positive_continuous_huber_relative_improvement_percent": (
                huber_improvement
            ),
            "positive_count_state_mae_relative_improvement_percent": (
                state_improvement
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _checkpoint_catalog_verified(
    registry: Registry,
    *,
    run_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    expected_epoch: int,
    retention_class: str,
) -> bool:
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT a.path, a.sha256, a.size_bytes, a.status,
                   c.role, c.best_epoch, c.monitored_metric,
                   c.monitored_mode, c.monitored_value,
                   c.retention_class, c.verification_status
            FROM artifacts a
            JOIN checkpoint_catalog c ON c.artifact_id = a.artifact_id
            WHERE a.run_id = ?
            """,
            (run_id,),
        ).fetchall()
    if len(rows) != 1:
        raise SelfHurdleAnalysisError(
            f"{run_id} must have exactly one indexed checkpoint"
        )
    row = dict(rows[0])
    if (
        Path(str(row["path"])).resolve() != checkpoint_path.resolve()
        or row["sha256"] != checkpoint_sha256
        or row["size_bytes"] != checkpoint_path.stat().st_size
        or row["status"] != "present"
        or row["role"] != "last"
        or row["best_epoch"] != expected_epoch
        or row["monitored_metric"] != PRIMARY_METRIC
        or row["monitored_mode"] != "min"
        or row["retention_class"] != retention_class
        or row["verification_status"] != "verified"
    ):
        raise SelfHurdleAnalysisError(
            f"{run_id} checkpoint catalog record is invalid"
        )
    return True


def load_science_evidence(
    *,
    registry: Registry,
    project_root: Path,
    slot: RegistrySlot,
    resource_gate_checksum: str,
) -> RunEvidence:
    root = slot.artifact_path
    verified = verify_run_bundle(root)
    if (
        verified.get("valid") is not True
        or verified.get("status") != "success"
        or root.name != slot.run_id
    ):
        raise SelfHurdleAnalysisError(
            f"{slot.alias} immutable run bundle did not verify"
        )
    checksums = _mapping(
        _strict_json(
            root / "provenance/artifact_checksums.json",
            f"{slot.alias} artifact checksums",
        ).get("files"),
        f"{slot.alias} artifact checksum files",
    )

    def artifact_sha(relative: str) -> str:
        record = _mapping(
            checksums.get(relative), f"{slot.alias} {relative} checksum"
        )
        observed = _sha(record.get("sha256"), f"{slot.alias} {relative}")
        if (
            record.get("type") != "file"
            or record.get("size") != (root / relative).stat().st_size
            or _sha_file(root / relative) != observed
        ):
            raise SelfHurdleAnalysisError(
                f"{slot.alias} artifact record changed for {relative}"
            )
        return observed

    config = _strict_yaml(
        root / "config.resolved.yaml", f"{slot.alias} resolved config"
    )
    _config_semantics(config, alias=slot.alias, stage="science")
    if (
        config != slot.config
        or canonical_sha256(config) != slot.expected_config_sha256
    ):
        raise SelfHurdleAnalysisError(
            f"{slot.alias} resolved config is not the locked config"
        )
    summary = _strict_json(root / "summary.json", f"{slot.alias} summary")
    metrics = _strict_json(
        root / "metrics/final.json", f"{slot.alias} final metrics"
    )
    resource = _strict_json(
        root / "diagnostics/resource_usage.json",
        f"{slot.alias} resource diagnostics",
    )
    hardware = _strict_json(
        root / "provenance/hardware.json", f"{slot.alias} hardware"
    )
    graph = _strict_json(
        root / "diagnostics/graph_prohibition.json",
        f"{slot.alias} graph prohibition",
    )
    gate_authorization = _strict_json(
        root / "provenance/external_gate_authorization.json",
        f"{slot.alias} gate authorization",
    )
    references = _strict_json(
        root / "provenance/per_gene_references.json",
        f"{slot.alias} per-gene references",
    )
    rows = _load_jsonl(
        root / "metrics/evaluation_replicates.jsonl",
        f"{slot.alias} evaluation replicates",
    )
    _validate_training_history(
        root,
        alias=slot.alias,
        run_id=slot.run_id,
        expected_epochs=EXPECTED_SCIENCE_EPOCHS,
    )
    if summary.get("metrics") != metrics:
        raise SelfHurdleAnalysisError(
            f"{slot.alias} summary metrics differ from final metrics"
        )
    if (
        summary.get("run_id") != slot.run_id
        or summary.get("status") != "success"
        or summary.get("campaign_id") != CAMPAIGN_ID
        or summary.get("biological_unit_alias") != slot.alias
        or summary.get("resource_pilot") is not False
        or summary.get("conclusion_eligible") is not True
        or summary.get("generalization_estimate") is not False
        or summary.get("final_epoch") != EXPECTED_SCIENCE_EPOCHS - 1
        or summary.get("fixed_epoch_budget") != EXPECTED_SCIENCE_EPOCHS
        or summary.get("checkpoint_role") != "last"
        or summary.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or summary.get("primary_metric_name") != PRIMARY_METRIC
        or summary.get("graph_constructed") is not False
        or summary.get("graph_inputs_used") is not False
        or summary.get("edge_inputs_used") is not False
        or resource.get("resource_pilot") is not False
        or resource.get("all_epochs_completed") is not True
        or resource.get("all_losses_and_gradients_finite") is not True
        or resource.get("graph_construction_count") != 0
        or resource.get("graph_input_tensor_count") != 0
        or resource.get("edge_input_tensor_count") != 0
        or graph.get("graph_construction_count") != 0
        or graph.get("graph_input_tensor_count") != 0
        or graph.get("edge_input_tensor_count") != 0
        or graph.get("coordinates_used_by_model") is not False
        or gate_authorization.get("resource_pilot") is not False
        or _mapping(
            gate_authorization.get("resource_gate"),
            f"{slot.alias} gate authorization identity",
        ).get("checksum")
        != resource_gate_checksum
    ):
        raise SelfHurdleAnalysisError(
            f"{slot.alias} science summary/provenance violates the contract"
        )
    peak = _finite(resource.get("peak_allocated_vram_gib"), "peak VRAM")
    disk = _finite(
        resource.get("filesystem_used_decimal_gb"), "recorded disk use"
    )
    if peak > SCIENCE_PEAK_VRAM_GIB_MAX or disk >= DISK_USED_DECIMAL_GB_MAX:
        raise SelfHurdleAnalysisError(
            f"{slot.alias} exceeded a science resource hard stop"
        )
    _close(
        _finite(summary.get("peak_vram_gib"), "summary peak VRAM"),
        peak,
        f"{slot.alias} peak VRAM",
    )
    _close(
        _finite(summary.get("filesystem_used_decimal_gb"), "summary disk"),
        disk,
        f"{slot.alias} disk use",
    )
    primary = _finite(metrics.get(PRIMARY_METRIC), PRIMARY_METRIC)
    _close(
        _finite(summary.get("primary_metric_value"), "summary primary metric"),
        primary,
        f"{slot.alias} primary metric",
    )
    comparisons = build_mode_comparisons(metrics, rows)
    gate = evaluate_representation_gate(comparisons)
    marker = _strict_json(root / "_SUCCESS", f"{slot.alias} success marker")
    marker_content = _sha(
        marker.get("content_sha256"),
        f"{slot.alias} bundle content checksum",
    )
    checkpoint_sha = artifact_sha("checkpoints/last.ckpt")
    catalog_verified = _checkpoint_catalog_verified(
        registry,
        run_id=slot.run_id,
        checkpoint_path=root / "checkpoints/last.ckpt",
        checkpoint_sha256=checkpoint_sha,
        expected_epoch=EXPECTED_SCIENCE_EPOCHS - 1,
        retention_class="retain_exploratory_evidence",
    )
    reference_audit = _mapping(
        references.get("audit"), f"{slot.alias} reference audit"
    )
    return RunEvidence(
        alias=slot.alias,
        run_id=slot.run_id,
        root=root,
        config_sha256=slot.expected_config_sha256,
        config_file_sha256=artifact_sha("config.resolved.yaml"),
        checkpoint_sha256=checkpoint_sha,
        success_marker_sha256=_sha_file(root / "_SUCCESS"),
        bundle_content_sha256=marker_content,
        metrics_sha256=artifact_sha("metrics/final.json"),
        summary_sha256=artifact_sha("summary.json"),
        checkpoint_catalog_verified=catalog_verified,
        summary=summary,
        metrics=metrics,
        resource=resource,
        hardware=hardware,
        comparisons=comparisons,
        gate=gate,
        registry_duration_seconds=slot.registry_duration_seconds,
        requested_gpu=slot.requested_gpu,
        dataset_fingerprint=_sha(
            _mapping(config.get("dataset"), "dataset").get(
                "dataset_fingerprint"
            ),
            f"{slot.alias} dataset fingerprint",
        ),
        split_fingerprint=_sha(
            _mapping(config.get("dataset"), "dataset").get(
                "split_fingerprint"
            ),
            f"{slot.alias} split fingerprint",
        ),
        mask_bundle_sha256=_sha(
            summary.get("evaluation_mask_bundle_sha256"),
            f"{slot.alias} mask bundle checksum",
        ),
        per_gene_reference_sha256=_sha(
            reference_audit.get("reference_sha256"),
            f"{slot.alias} per-gene reference checksum",
        ),
    )


def load_resource_evidence(
    *,
    registry: Registry,
    project_root: Path,
    slot: RegistrySlot,
    gate_job: Mapping[str, Any],
) -> ResourceEvidence:
    root = slot.artifact_path
    verified = verify_run_bundle(root)
    if (
        verified.get("valid") is not True
        or verified.get("status") != "success"
        or root.name != slot.run_id
        or gate_job.get("run_id") != slot.run_id
        or _inside_project(
            project_root,
            gate_job.get("artifact_bundle"),
            f"{slot.alias} gate resource bundle",
        )
        != root.resolve()
    ):
        raise SelfHurdleAnalysisError(
            f"{slot.alias} resource bundle identity did not verify"
        )
    config = _strict_yaml(
        root / "config.resolved.yaml",
        f"{slot.alias} resource resolved config",
    )
    _config_semantics(config, alias=slot.alias, stage="resource")
    if (
        config != slot.config
        or canonical_sha256(config) != slot.expected_config_sha256
        or gate_job.get("config_sha256") != slot.expected_config_sha256
    ):
        raise SelfHurdleAnalysisError(
            f"{slot.alias} resource config is not the locked config"
        )
    summary = _strict_json(
        root / "summary.json", f"{slot.alias} resource summary"
    )
    resource = _strict_json(
        root / "diagnostics/resource_usage.json",
        f"{slot.alias} resource diagnostics",
    )
    _validate_training_history(
        root,
        alias=slot.alias,
        run_id=slot.run_id,
        expected_epochs=2,
    )
    checksums = _mapping(
        _strict_json(
            root / "provenance/artifact_checksums.json",
            f"{slot.alias} resource artifact checksums",
        ).get("files"),
        f"{slot.alias} resource artifact checksum files",
    )
    checkpoint = root / "checkpoints/last.ckpt"
    checkpoint_record = _mapping(
        checksums.get("checkpoints/last.ckpt"),
        f"{slot.alias} resource checkpoint record",
    )
    checkpoint_sha = _sha(
        checkpoint_record.get("sha256"),
        f"{slot.alias} resource checkpoint checksum",
    )
    if (
        _sha_file(checkpoint) != checkpoint_sha
        or checkpoint_record.get("size") != checkpoint.stat().st_size
        or summary.get("run_id") != slot.run_id
        or summary.get("resource_pilot") is not True
        or summary.get("conclusion_eligible") is not False
        or summary.get("final_epoch") != 1
        or summary.get("fixed_epoch_budget") != 2
        or resource.get("resource_pilot") is not True
        or resource.get("runner_pilot_gate_passed") is not True
        or resource.get("all_epochs_completed") is not True
        or resource.get("all_losses_and_gradients_finite") is not True
        or resource.get("graph_construction_count") != 0
        or resource.get("graph_input_tensor_count") != 0
        or resource.get("edge_input_tensor_count") != 0
    ):
        raise SelfHurdleAnalysisError(
            f"{slot.alias} resource evidence violates the contract"
        )
    marker = _strict_json(
        root / "_SUCCESS", f"{slot.alias} resource success marker"
    )
    marker_sha = _sha_file(root / "_SUCCESS")
    if marker_sha != gate_job.get("success_marker_sha256"):
        raise SelfHurdleAnalysisError(
            f"{slot.alias} resource success marker differs from gate"
        )
    peak = _finite(
        resource.get("peak_allocated_vram_gib"),
        f"{slot.alias} resource peak VRAM",
    )
    discrepancy = _finite(
        resource.get("fp32_amp_absolute_loss_discrepancy"),
        f"{slot.alias} AMP discrepancy",
    )
    projected = _finite(
        resource.get("projected_gpu_hours_per_200_epochs"),
        f"{slot.alias} projected GPU hours",
    )
    disk = _finite(
        resource.get("filesystem_used_decimal_gb"),
        f"{slot.alias} resource disk",
    )
    if (
        peak != _finite(
            gate_job.get("peak_allocated_vram_gib"),
            f"{slot.alias} gate peak VRAM",
        )
        or discrepancy
        != _finite(
            gate_job.get("fp32_amp_absolute_loss_discrepancy"),
            f"{slot.alias} gate AMP discrepancy",
        )
        or projected
        != _finite(
            gate_job.get("projected_gpu_hours_per_200_epochs"),
            f"{slot.alias} gate projected hours",
        )
        or disk
        != _finite(
            gate_job.get("filesystem_used_decimal_gb"),
            f"{slot.alias} gate disk",
        )
    ):
        raise SelfHurdleAnalysisError(
            f"{slot.alias} resource diagnostics differ from the gate"
        )
    catalog_verified = _checkpoint_catalog_verified(
        registry,
        run_id=slot.run_id,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        expected_epoch=1,
        retention_class="retain_diagnostic",
    )
    return ResourceEvidence(
        alias=slot.alias,
        run_id=slot.run_id,
        root=root,
        config_sha256=slot.expected_config_sha256,
        checkpoint_sha256=checkpoint_sha,
        success_marker_sha256=marker_sha,
        bundle_content_sha256=_sha(
            marker.get("content_sha256"),
            f"{slot.alias} resource bundle content checksum",
        ),
        checkpoint_catalog_verified=catalog_verified,
        resource=resource,
        registry_duration_seconds=slot.registry_duration_seconds,
        requested_gpu=slot.requested_gpu,
    )


def conservative_campaign_gpu_hours(
    registry_duration_seconds: Sequence[float],
) -> float:
    """Return one-GPU-per-run accounting from all registered durations."""

    if len(registry_duration_seconds) != 4:
        raise SelfHurdleAnalysisError(
            "campaign GPU accounting requires all four registered runs"
        )
    durations = [
        _finite(value, "registered GPU-run duration")
        for value in registry_duration_seconds
    ]
    if any(value < 0 for value in durations):
        raise SelfHurdleAnalysisError(
            "registered GPU-run duration cannot be negative"
        )
    return sum(durations) / 3600.0


def _metric_rows(evidence: Sequence[RunEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in sorted(evidence, key=lambda item: item.alias):
        for mode in MASK_MODES:
            for field in (*ACCURACY_FIELDS, *ERROR_FIELDS):
                comparison = run.comparisons[mode][field]
                rows.append(
                    {
                        "core_alias": run.alias,
                        "run_id": run.run_id,
                        "mask_estimand": mode,
                        "metric": field,
                        "direction": comparison["direction"],
                        "model_value": comparison["model"],
                        "per_gene_reference_value": comparison[
                            "per_gene_reference"
                        ],
                        "model_percent": comparison["model_percent"],
                        "per_gene_reference_percent": comparison[
                            "per_gene_reference_percent"
                        ],
                        "difference_percentage_points": comparison[
                            "difference_percentage_points"
                        ],
                        "relative_improvement_percent": comparison[
                            "relative_improvement_percent"
                        ],
                        "frozen_whole_node_gate_metric": (
                            mode == "whole_node"
                            and field
                            in {
                                "detection_balanced_accuracy",
                                "positive_continuous_huber",
                                "positive_count_state_mae",
                            }
                        ),
                    }
                )
    return rows


def build_analysis_payload(
    *,
    materialization: Mapping[str, Any],
    resource_gate: Mapping[str, Any],
    enqueue: Mapping[str, Mapping[str, Any]],
    evidence: Sequence[RunEvidence],
    resource_evidence: Sequence[ResourceEvidence],
    failed_attempts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if {run.alias for run in evidence} != set(ALIASES) or len(evidence) != 2:
        raise SelfHurdleAnalysisError(
            "analysis requires both and only both science cores"
        )
    if (
        {run.alias for run in resource_evidence} != set(ALIASES)
        or len(resource_evidence) != 2
    ):
        raise SelfHurdleAnalysisError(
            "analysis requires both and only both resource pilots"
        )
    per_core_gate = {run.alias: dict(run.gate) for run in evidence}
    campaign_pass = all(
        bool(per_core_gate[alias]["passed"]) for alias in ALIASES
    )
    science_gpu_hours = sum(
        run.registry_duration_seconds for run in evidence
    ) / 3600.0
    resource_by_alias = {
        str(_mapping(row, "resource gate job").get("alias")): _mapping(
            row, "resource gate job"
        )
        for row in _list(resource_gate.get("jobs"), "resource gate jobs")
    }
    pilot_gpu_hours = sum(
        run.registry_duration_seconds for run in resource_evidence
    ) / 3600.0
    all_gpu_hours = conservative_campaign_gpu_hours(
        [
            *[run.registry_duration_seconds for run in resource_evidence],
            *[run.registry_duration_seconds for run in evidence],
        ]
    )
    resources = {
        "accounting_basis": (
            "one GPU per run; conservative registered start-to-end duration "
            "summed across both resource pilots and both science runs"
        ),
        "science_registered_gpu_hours": science_gpu_hours,
        "resource_pilot_registered_gpu_hours": pilot_gpu_hours,
        "campaign_registered_gpu_hours": all_gpu_hours,
        "preferred_budget_gpu_hours": PREFERRED_GPU_HOURS,
        "absolute_budget_gpu_hours": ABSOLUTE_GPU_HOURS,
        "within_preferred_budget": all_gpu_hours <= PREFERRED_GPU_HOURS,
        "within_absolute_budget": all_gpu_hours <= ABSOLUTE_GPU_HOURS,
        "maximum_science_peak_allocated_vram_gib": max(
            _finite(
                run.resource.get("peak_allocated_vram_gib"),
                f"{run.alias} peak VRAM",
            )
            for run in evidence
        ),
        "science_peak_vram_gib_limit": SCIENCE_PEAK_VRAM_GIB_MAX,
        "maximum_recorded_filesystem_used_decimal_gb": max(
            [
                *[
                    _finite(
                        run.resource.get("filesystem_used_decimal_gb"),
                        f"{run.alias} disk use",
                    )
                    for run in evidence
                ],
                *[
                    _finite(
                        resource_by_alias[alias].get(
                            "filesystem_used_decimal_gb"
                        ),
                        f"{alias} pilot disk use",
                    )
                    for alias in ALIASES
                ],
            ]
        ),
        "filesystem_used_decimal_gb_hard_stop": DISK_USED_DECIMAL_GB_MAX,
        "runs": {
            run.alias: {
                "run_id": run.run_id,
                "requested_gpu": run.requested_gpu,
                "gpu_model": str(run.hardware.get("gpu_model")),
                "cuda_visible_devices": str(
                    run.hardware.get("cuda_visible_devices")
                ),
                "registry_duration_seconds": run.registry_duration_seconds,
                "runner_duration_seconds": _finite(
                    run.summary.get("duration_seconds"),
                    f"{run.alias} runner duration",
                ),
                "training_duration_seconds": _finite(
                    run.metrics.get(
                        "resource/training_duration_seconds"
                    ),
                    f"{run.alias} training duration",
                ),
                "peak_allocated_vram_gib": _finite(
                    run.resource.get("peak_allocated_vram_gib"),
                    f"{run.alias} peak VRAM",
                ),
                "peak_host_memory_bytes": _integer(
                    run.summary.get("peak_host_memory_bytes"),
                    f"{run.alias} peak host memory",
                ),
                "filesystem_used_decimal_gb": _finite(
                    run.resource.get("filesystem_used_decimal_gb"),
                    f"{run.alias} disk use",
                ),
                "checkpoint_size_bytes": _integer(
                    run.metrics.get("resource/checkpoint_size_bytes"),
                    f"{run.alias} checkpoint size",
                ),
            }
            for run in evidence
        },
        "resource_pilots": {
            run.alias: {
                "run_id": run.run_id,
                "requested_gpu": run.requested_gpu,
                "registry_duration_seconds": run.registry_duration_seconds,
                "peak_allocated_vram_gib": _finite(
                    run.resource.get("peak_allocated_vram_gib"),
                    f"{run.alias} pilot peak VRAM",
                ),
                "fp32_amp_absolute_loss_discrepancy": _finite(
                    run.resource.get(
                        "fp32_amp_absolute_loss_discrepancy"
                    ),
                    f"{run.alias} AMP discrepancy",
                ),
                "projected_gpu_hours_per_200_epochs": _finite(
                    run.resource.get(
                        "projected_gpu_hours_per_200_epochs"
                    ),
                    f"{run.alias} projected GPU hours",
                ),
                "filesystem_used_decimal_gb": _finite(
                    run.resource.get("filesystem_used_decimal_gb"),
                    f"{run.alias} pilot disk use",
                ),
                "config_sha256": run.config_sha256,
                "checkpoint_sha256": run.checkpoint_sha256,
                "success_marker_sha256": run.success_marker_sha256,
                "bundle_content_sha256": run.bundle_content_sha256,
                "checkpoint_catalog_verified": (
                    run.checkpoint_catalog_verified
                ),
            }
            for run in resource_evidence
        },
    }
    maximum_claim = (
        "The fixed graphless model met the exploratory held-in, two-core "
        "within-cell representation-capacity gate relative to its all-fit "
        "per-gene reference."
        if campaign_pass
        else (
            "The prespecified two-core representation-capacity hypothesis "
            "was not supported; individual held-in metric gains are "
            "descriptive."
        )
    )
    return {
        "schema_version": 1,
        "analysis_kind": "self_hurdle_two_core_capacity_audit_v1",
        "campaign_id": CAMPAIGN_ID,
        "analysis_population": {
            "biological_units": list(ALIASES),
            "science_run_count": 2,
            "all_registered_science_runs_included": True,
            "all_registered_resource_pilots_included": True,
            "best_run_or_checkpoint_selection": False,
            "failed_attempts": list(failed_attempts),
        },
        "locked_receipts": {
            "frozen_task_contract_sha256": CONTRACT_SHA256,
            "materialization_checksum": materialization["checksum"],
            "resource_gate_checksum": resource_gate["checksum"],
            "resource_enqueue_checksum": enqueue["resource"]["checksum"],
            "science_enqueue_checksum": enqueue["science"]["checksum"],
        },
        "estimands": {
            "primary": "whole_node",
            "reported_separately": ["partial_gene", "spatial_block"],
            "fit_scope": "all_nodes_transductive",
            "validation_nodes": 0,
            "test_nodes": 0,
        },
        "representation_gate": {
            "rule": "all three whole-node criteria must pass on both cores",
            "per_core": per_core_gate,
            "campaign_passed": campaign_pass,
            "outcome": "supported" if campaign_pass else "negative",
        },
        "runs": {
            run.alias: {
                "run_id": run.run_id,
                "artifact_bundle": run.root.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "config_sha256": run.config_sha256,
                "config_file_sha256": run.config_file_sha256,
                "checkpoint_sha256": run.checkpoint_sha256,
                "checkpoint_catalog_verified": (
                    run.checkpoint_catalog_verified
                ),
                "success_marker_sha256": run.success_marker_sha256,
                "bundle_content_sha256": run.bundle_content_sha256,
                "metrics_sha256": run.metrics_sha256,
                "summary_sha256": run.summary_sha256,
                "dataset_fingerprint": run.dataset_fingerprint,
                "split_fingerprint": run.split_fingerprint,
                "evaluation_mask_bundle_sha256": run.mask_bundle_sha256,
                "per_gene_reference_sha256": (
                    run.per_gene_reference_sha256
                ),
                "mask_estimands": run.comparisons,
            }
            for run in sorted(evidence, key=lambda item: item.alias)
        },
        "resources": resources,
        "scientific_interpretation": {
            "maximum_defensible_claim": maximum_claim,
            "facts": [
                "The model is graphless and used no edge or coordinate input.",
                "Every reported metric is held-in on the same core used for fitting.",
                "The reference fits one prevalence and one positive-count median per gene on all nodes.",
                "Both resource-pilot and both science bundles verified against the registry and checkpoint catalog.",
            ],
            "strongest_alternative_explanations": [
                "transductive memorization or common within-cell co-expression",
                "morphology or technical intensity rather than biological state",
                "a weak fixed-threshold prevalence reference for rare-gene detection",
            ],
            "untested": [
                "unseen-patient or unseen-core generalization",
                "independently validated biological labels or programs",
                "graph-specific predictive gain or cell-cell interaction",
                "mechanism or causality",
            ],
            "prohibited_claims": [
                "unseen-patient generalization",
                "stable population biology",
                "spatial predictive dependency",
                "cell-cell communication",
                "biological mechanism",
                "causal influence",
            ],
        },
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    gate = _mapping(payload["representation_gate"], "representation gate")
    outcome = str(gate["outcome"]).upper()
    lines = [
        "# Self-only continuous-hurdle full-core capacity: results",
        "",
        f"Outcome: **{outcome}** for the frozen two-core capacity hypothesis.",
        "",
        (
            "This is an exploratory, graphless, held-in reconstruction result. "
            "It is not a validation/test estimate and does not test cell-cell "
            "interaction or biological mechanism."
        ),
        "",
        "## Prespecified whole-node gate",
        "",
        (
            "| Core | Detection BA model/ref | Gain (pp) | Positive Huber "
            "model/ref | Improvement | Positive-state MAE model/ref | "
            "Improvement | Core pass |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    runs = _mapping(payload["runs"], "runs")
    per_core = _mapping(gate["per_core"], "per-core gate")
    for alias in ALIASES:
        run = _mapping(runs[alias], f"{alias} run")
        mode = _mapping(
            _mapping(run["mask_estimands"], f"{alias} estimands")[
                "whole_node"
            ],
            f"{alias} whole-node",
        )
        ba = _mapping(mode["detection_balanced_accuracy"], "BA")
        huber = _mapping(mode["positive_continuous_huber"], "Huber")
        state = _mapping(mode["positive_count_state_mae"], "state MAE")
        core_gate = _mapping(per_core[alias], f"{alias} gate")
        lines.append(
            "| "
            + " | ".join(
                [
                    alias,
                    (
                        f"{_format_number(ba['model_percent'], 2)}% / "
                        f"{_format_number(ba['per_gene_reference_percent'], 2)}%"
                    ),
                    _format_number(ba["difference_percentage_points"], 2),
                    (
                        f"{_format_number(huber['model'], 4)} / "
                        f"{_format_number(huber['per_gene_reference'], 4)}"
                    ),
                    (
                        f"{_format_number(huber['relative_improvement_percent'], 2)}%"
                    ),
                    (
                        f"{_format_number(state['model'], 4)} / "
                        f"{_format_number(state['per_gene_reference'], 4)}"
                    ),
                    (
                        f"{_format_number(state['relative_improvement_percent'], 2)}%"
                    ),
                    "yes" if core_gate["passed"] else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                "The campaign passes only if all three criteria pass on both "
                "cores. Failure of one criterion on either core makes the "
                "frozen hypothesis negative."
            ),
            "",
            "## Estimands reported separately",
            "",
            (
                "Each percentage cell is `model / all-fit per-gene reference`. "
                "Loss/error columns report relative improvement; positive means "
                "lower error."
            ),
            "",
            (
                "| Core | Mask estimand | Detection BA | Sensitivity | "
                "Specificity | Precision | Positive exact | Positive ±1 | "
                "State-8 exact | State-8 balanced | Huber impr. | "
                "State-MAE impr. | Reconstructed log1p-MAE impr. |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for alias in ALIASES:
        estimands = _mapping(
            _mapping(runs[alias], f"{alias} run")["mask_estimands"],
            f"{alias} estimands",
        )
        for mode in MASK_MODES:
            values = _mapping(estimands[mode], f"{alias}/{mode}")

            def pair(field: str) -> str:
                item = _mapping(values[field], f"{alias}/{mode}/{field}")
                return (
                    f"{_format_number(item['model_percent'], 2)}% / "
                    f"{_format_number(item['per_gene_reference_percent'], 2)}%"
                )

            lines.append(
                "| "
                + " | ".join(
                    [
                        alias,
                        mode,
                        pair("detection_balanced_accuracy"),
                        pair("detection_sensitivity"),
                        pair("detection_specificity"),
                        pair("detection_precision"),
                        pair("positive_count_state_exact_accuracy"),
                        pair("positive_count_state_within_one_accuracy"),
                        pair("state8_exact_accuracy"),
                        pair("state8_balanced_accuracy"),
                        (
                            f"{_format_number(values['positive_continuous_huber']['relative_improvement_percent'], 2)}%"
                        ),
                        (
                            f"{_format_number(values['positive_count_state_mae']['relative_improvement_percent'], 2)}%"
                        ),
                        (
                            f"{_format_number(values['reconstructed_count_log1p_mae']['relative_improvement_percent'], 2)}%"
                        ),
                    ]
                )
                + " |"
            )
    resources = _mapping(payload["resources"], "resources")
    lines.extend(
        [
            "",
            "## Resource and provenance audit",
            "",
            (
                "| Core | Run ID | GPU | Registry runtime | Training runtime | "
                "Peak VRAM | Recorded disk | Checkpoint SHA-256 |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    resource_runs = _mapping(resources["runs"], "resource runs")
    for alias in ALIASES:
        run = _mapping(runs[alias], f"{alias} run")
        resource = _mapping(resource_runs[alias], f"{alias} resources")
        lines.append(
            "| "
            + " | ".join(
                [
                    alias,
                    f"`{run['run_id']}`",
                    str(resource["requested_gpu"]),
                    (
                        f"{_finite(resource['registry_duration_seconds'], 'runtime') / 60.0:.2f} min"
                    ),
                    (
                        f"{_finite(resource['training_duration_seconds'], 'training runtime') / 60.0:.2f} min"
                    ),
                    (
                        f"{_format_number(resource['peak_allocated_vram_gib'], 3)} GiB"
                    ),
                    (
                        f"{_format_number(resource['filesystem_used_decimal_gb'], 3)} GB"
                    ),
                    f"`{run['checkpoint_sha256']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Resource pilots:",
            "",
            (
                "| Core | Run ID | GPU | Registry runtime | Peak VRAM | "
                "AMP loss discrepancy | Recorded disk | Checkpoint SHA-256 |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    pilot_runs = _mapping(
        resources["resource_pilots"], "resource pilot runs"
    )
    for alias in ALIASES:
        pilot = _mapping(pilot_runs[alias], f"{alias} resource pilot")
        lines.append(
            "| "
            + " | ".join(
                [
                    alias,
                    f"`{pilot['run_id']}`",
                    str(pilot["requested_gpu"]),
                    (
                        f"{_finite(pilot['registry_duration_seconds'], 'pilot runtime') / 60.0:.2f} min"
                    ),
                    (
                        f"{_format_number(pilot['peak_allocated_vram_gib'], 3)} GiB"
                    ),
                    _format_number(
                        pilot["fp32_amp_absolute_loss_discrepancy"], 8
                    ),
                    (
                        f"{_format_number(pilot['filesystem_used_decimal_gb'], 3)} GB"
                    ),
                    f"`{pilot['checkpoint_sha256']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            (
                f"Conservative registered campaign GPU use was "
                f"{_format_number(resources['campaign_registered_gpu_hours'], 4)} "
                f"GPU-hours; the preferred and absolute limits were "
                f"{PREFERRED_GPU_HOURS:.0f} and {ABSOLUTE_GPU_HOURS:.0f}. "
                f"Maximum recorded disk use was "
                f"{_format_number(resources['maximum_recorded_filesystem_used_decimal_gb'], 3)} "
                f"GB, below the {DISK_USED_DECIMAL_GB_MAX:.0f} GB hard stop."
            ),
            "",
            "Both resource-pilot and both science bundles, final-epoch "
            "checkpoints, configuration hashes, fixed masks, per-gene "
            "references, and checkpoint-catalog records verified. No best run "
            "or checkpoint was selected.",
            "",
            "## Interpretation",
            "",
            str(
                _mapping(
                    payload["scientific_interpretation"],
                    "scientific interpretation",
                )["maximum_defensible_claim"]
            ),
            "",
            (
                "The per-gene reference is not a strong learned control: at a "
                "fixed 0.5 detection threshold it predicts almost all rare "
                "genes as absent. Detection balanced-accuracy gains therefore "
                "do not by themselves establish biological signal. Likewise, "
                "high state-8 exact accuracy is zero-dominated; its reference "
                "can exceed the model while balanced accuracy moves in the "
                "opposite direction."
            ),
            "",
            (
                "The strongest unresolved explanations are transductive "
                "memorization, common within-cell co-expression, and morphology "
                "or technical intensity. There is no unseen-core evaluation, "
                "morphology-only learned baseline, biological-label readout, "
                "graph input, interaction test, mechanism test, or causal test."
            ),
            "",
            "The complete machine-readable comparison is in `comparison.json`; "
            "all model/reference metric pairs are in `metrics.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_report(
    output_dir: Path,
    payload: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, str]:
    json_content = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    columns = [
        "core_alias",
        "run_id",
        "mask_estimand",
        "metric",
        "direction",
        "model_value",
        "per_gene_reference_value",
        "model_percent",
        "per_gene_reference_percent",
        "difference_percentage_points",
        "relative_improvement_percent",
        "frozen_whole_node_gate_metric",
    ]
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(metric_rows)
    csv_content = csv_buffer.getvalue().encode()
    markdown_content = render_markdown(payload).encode()
    files = {
        "comparison.json": json_content,
        "metrics.csv": csv_content,
        "RESULTS.md": markdown_content,
    }
    checksums: dict[str, str] = {}
    for name, content in files.items():
        _atomic_write(output_dir / name, content)
        checksums[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": 1,
        "analysis_kind": payload["analysis_kind"],
        "campaign_id": CAMPAIGN_ID,
        "files": {
            name: {"sha256": checksums[name], "size_bytes": len(files[name])}
            for name in sorted(files)
        },
    }
    manifest_content = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    _atomic_write(output_dir / "report_manifest.json", manifest_content)
    checksums["report_manifest.json"] = hashlib.sha256(
        manifest_content
    ).hexdigest()
    return checksums


def analyze(
    *,
    project_root: Path,
    registry: Registry,
    materialization_path: Path,
    resource_gate_path: Path,
    resource_enqueue_path: Path,
    science_enqueue_path: Path,
    output_dir: Path,
) -> Mapping[str, Any]:
    materialization, gate, expected, enqueue = validate_locked_receipts(
        project_root=project_root,
        materialization_path=materialization_path,
        resource_gate_path=resource_gate_path,
        resource_enqueue_path=resource_enqueue_path,
        science_enqueue_path=science_enqueue_path,
    )
    slots, failed_attempts = registry_inventory(
        registry=registry,
        project_root=project_root,
        expected=expected,
        enqueue=enqueue,
    )
    evidence = [
        load_science_evidence(
            registry=registry,
            project_root=project_root,
            slot=slots[("science", alias)],
            resource_gate_checksum=str(gate["checksum"]),
        )
        for alias in ALIASES
    ]
    gate_jobs = {
        str(_mapping(row, "resource gate job").get("alias")): _mapping(
            row, "resource gate job"
        )
        for row in _list(gate.get("jobs"), "resource gate jobs")
    }
    resource_evidence = [
        load_resource_evidence(
            registry=registry,
            project_root=project_root,
            slot=slots[("resource", alias)],
            gate_job=gate_jobs[alias],
        )
        for alias in ALIASES
    ]
    payload = build_analysis_payload(
        materialization=materialization,
        resource_gate=gate,
        enqueue=enqueue,
        evidence=evidence,
        resource_evidence=resource_evidence,
        failed_attempts=failed_attempts,
    )
    checksums = write_report(output_dir, payload, _metric_rows(evidence))
    return {
        "campaign_id": CAMPAIGN_ID,
        "outcome": payload["representation_gate"]["outcome"],
        "campaign_passed": payload["representation_gate"]["campaign_passed"],
        "science_run_ids": [run.run_id for run in evidence],
        "resource_run_ids": [run.run_id for run in resource_evidence],
        "output_dir": output_dir.as_posix(),
        "report_checksums": checksums,
    }


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking/bagm.sqlite3",
    )
    parser.add_argument(
        "--materialization",
        type=Path,
        default=locked / "locked_config_materialization.json",
    )
    parser.add_argument(
        "--resource-gate",
        type=Path,
        default=locked / "resource_gate_receipt.json",
    )
    parser.add_argument(
        "--resource-enqueue",
        type=Path,
        default=locked / "resource_enqueue_receipt.json",
    )
    parser.add_argument(
        "--science-enqueue",
        type=Path,
        default=locked / "science_enqueue_receipt.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            paths.project_root
            / "reports/analyses/self_hurdle_full_core_capacity"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze(
        project_root=PROJECT_ROOT,
        registry=Registry(args.database.resolve()),
        materialization_path=args.materialization.resolve(),
        resource_gate_path=args.resource_gate.resolve(),
        resource_enqueue_path=args.resource_enqueue.resolve(),
        science_enqueue_path=args.science_enqueue.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
