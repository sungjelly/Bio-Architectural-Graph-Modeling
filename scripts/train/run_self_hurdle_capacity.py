#!/usr/bin/env python3
"""Run one worker-owned graphless self-hurdle capacity experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train.run_full_core_capacity import (  # noqa: E402
    CapacityRunResult,
    _evaluation_masks,
    _peak_host_memory_bytes,
    _resolve_prepared_artifact,
    _training_config,
    _verify_materialized_identity,
    _worker_archive_and_config,
)
from spatial_benchmark.configuration import load_yaml_mapping  # noqa: E402
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.full_core import (  # noqa: E402
    ALLOWED_METADATA_COLUMNS,
    FullCoreData,
    load_and_refit_full_core,
)
from spatial_benchmark.hurdle_continuous import (  # noqa: E402
    evaluate_hurdle_continuous_output,
)
from spatial_benchmark.hybrid_count import (  # noqa: E402
    tokenize_raw_counts,
    validate_raw_counts,
)
from spatial_benchmark.hybrid_count_metrics import (  # noqa: E402
    json_safe_metrics,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    RunValidationError,
    deidentify_prediction_rows,
)
from spatial_benchmark.self_hurdle import (  # noqa: E402
    SelfHurdleFixedMaskResult,
    SelfHurdleModel,
    SelfHurdlePrecisionResult,
    SelfHurdleSplitView,
    SelfHurdleTrainingResult,
    compare_self_hurdle_fp32_amp_loss,
    evaluate_fixed_self_hurdle_mask,
    fit_full_core_self_hurdle_model,
    trainable_parameter_count,
)
from spatial_benchmark.training import set_deterministic_seed  # noqa: E402


CAMPAIGN_ID = "cmp_20260729_self_hurdle_full_core_capacity"
CONTRACT_RELATIVE = Path(
    "experiments/campaigns"
) / CAMPAIGN_ID / "frozen_task_contract.yaml"
CONTRACT_SHA256 = (
    "26cf4f094d843c4fa9020c52e1d45988f8e4a0c43f20beb186e4742b2dacba00"
)
RECEIPT_KIND = "self_hurdle_locked_config_materialization_v1"
RESOURCE_GATE_KIND = "self_hurdle_resource_gate_v1"
EXPECTED_PARAMETER_COUNT = 16_917_200
ALIASES = ("ANC-03", "ANC-05")
TARGET_BATCH_SIZE = 16_384
PILOT_PEAK_GIB_MAX = 12.0
SCIENCE_PEAK_GIB_MAX = 20.5
AMP_DISCREPANCY_MAX = 1e-3
PER_RUN_PROJECTED_HOURS_MAX = 6.0
AGGREGATE_PROJECTED_HOURS_MAX = 12.0
DISK_USED_DECIMAL_GB_MAX = 55.0
PRIMARY_METRIC = "fit/whole_node/hurdle_loss"
PERCENT_FIELDS = (
    "detection_balanced_accuracy",
    "detection_sensitivity",
    "detection_specificity",
    "detection_precision",
    "state8_exact_accuracy",
    "state8_balanced_accuracy",
    "positive_count_state_exact_accuracy",
    "positive_count_state_within_one_accuracy",
)
PUBLIC_MASK_NAMES = {
    "partial": "partial_gene",
    "node": "whole_node",
    "block": "spatial_block",
}
REQUIRED_MASKS = ("partial_gene", "whole_node", "spatial_block")
ROW_METADATA = {
    "split",
    "mask_mode",
    "mask_replicate",
    "mask_entry_id",
    "mask_seed",
    "mask_checksum",
}


class SelfHurdleRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunContract:
    alias: str
    resource_pilot: bool


@dataclass(frozen=True)
class PerGeneReferences:
    detection_probability: np.ndarray
    positive_continuous_standardized: np.ndarray
    audit: Mapping[str, Any]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelfHurdleRunnerError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise SelfHurdleRunnerError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SelfHurdleRunnerError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject,
            object_pairs_hook=unique,
        )
    except SelfHurdleRunnerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelfHurdleRunnerError(f"{label} is invalid") from exc
    return dict(_mapping(value, label))


def _verified_checksum(value: Mapping[str, Any], label: str) -> str:
    observed = value.get("checksum")
    if not isinstance(observed, str) or len(observed) != 64:
        raise SelfHurdleRunnerError(f"{label} checksum is malformed")
    canonical = dict(value)
    canonical.pop("checksum", None)
    if canonical_sha256(canonical) != observed:
        raise SelfHurdleRunnerError(f"{label} checksum does not verify")
    return observed


def _project_path(
    project_root: Path,
    reference: Any,
    *,
    label: str,
    require_file: bool = True,
) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise SelfHurdleRunnerError(f"{label} must be a relative path")
    path = (project_root / reference).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise SelfHurdleRunnerError(f"{label} escapes project root") from exc
    if require_file and not path.is_file():
        raise SelfHurdleRunnerError(f"{label} is missing")
    return path


def _disk(project_root: Path) -> dict[str, float | int]:
    usage = shutil.disk_usage(project_root)
    used = (usage.total - usage.free) / 1e9
    result = {
        "total_bytes": usage.total,
        "free_bytes": usage.free,
        "used_decimal_gb": used,
        "hard_stop_decimal_gb": DISK_USED_DECIMAL_GB_MAX,
    }
    if used >= DISK_USED_DECIMAL_GB_MAX:
        raise SelfHurdleRunnerError(
            f"filesystem uses {used:.3f} GB; hard stop is 55 GB"
        )
    return result


def _validate_config(config: Mapping[str, Any]) -> RunContract:
    campaign = _mapping(config.get("campaign"), "campaign")
    experiment = _mapping(config.get("experiment"), "experiment")
    model = _mapping(config.get("model"), "model")
    graph = _mapping(config.get("graph"), "graph")
    features = _mapping(config.get("features"), "features")
    trainer = _mapping(config.get("trainer"), "trainer")
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    metadata = _mapping(config.get("metadata"), "metadata")
    dataset = _mapping(config.get("dataset"), "dataset")
    alias = str(experiment.get("biological_unit_alias"))
    pilot = bool(experiment.get("resource_pilot"))
    epochs = 2 if pilot else 200
    replicates = 1 if pilot else 3
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("frozen_contract_sha256") != CONTRACT_SHA256
        or alias not in ALIASES
        or experiment.get("arm") != "self-hurdle"
        or experiment.get("graph_arms_authorized") is not False
        or model.get("name") != "self-hurdle-count"
        or model.get("family") != "self_only_hurdle_count"
        or model.get("uses_graph_inputs") is not False
        or model.get("uses_edge_inputs") is not False
        or int(model.get("expected_trainable_parameter_count", -1))
        != EXPECTED_PARAMETER_COUNT
        or features.get("use_edge_features") is not False
        or graph.get("kind") != "disabled_self_only_schema_placeholder"
        or graph.get("enabled") is not False
        or graph.get("construction_performed") is not False
        or int(graph.get("expected_directed_edges", -1)) != 0
        or graph.get("expected_graph_sha256") is not None
        or float(graph.get("edge_dropout", -1)) != 0.0
        or int(trainer.get("max_epochs", -1)) != epochs
        or trainer.get("restore_best") is not False
        or trainer.get("checkpoint_policy") != "last_only"
        or trainer.get("primary_checkpoint_role") != "last"
        or trainer.get("graph_execution")
        != "none_graph_inputs_prohibited"
        or int(trainer.get("target_node_batch_size", -1))
        != TARGET_BATCH_SIZE
        or evaluation.get("protocol") != "held_in_full_core_fixed_budget"
        or evaluation.get("task_family")
        != "masked_expression_hurdle_count"
        or evaluation.get("primary_metric") != PRIMARY_METRIC
        or list(evaluation.get("splits", [])) != ["fit"]
        or int(evaluation.get("mask_replicates_per_mode", -1))
        != replicates
        or metadata.get("execution_role")
        != ("resource_pilot" if pilot else "science")
        or metadata.get("no_graph_construction_or_input") is not True
        or dataset.get("biological_unit_alias") != alias
        or dataset.get("validation_or_test_partition_present") is not False
        or dataset.get("task") != "masked_expression_hurdle_count"
        or int(config.get("seed", -1)) != 0
        or int(config.get("fold", -1)) != 0
        or int(config.get("attempt", -1)) != 1
    ):
        raise SelfHurdleRunnerError(
            "resolved config differs from the frozen graphless campaign"
        )
    return RunContract(alias=alias, resource_pilot=pilot)


def _verify_contract(project_root: Path) -> Mapping[str, Any]:
    path = project_root / CONTRACT_RELATIVE
    if not path.is_file() or sha256_file(path) != CONTRACT_SHA256:
        raise SelfHurdleRunnerError("frozen task contract checksum drifted")
    contract = load_yaml_mapping(path)
    if (
        contract.get("campaign_id") != CAMPAIGN_ID
        or contract.get("frozen_before_new_training") is not True
    ):
        raise SelfHurdleRunnerError("frozen task contract is invalid")
    return contract


def _verify_materialization(
    project_root: Path,
    config: Mapping[str, Any],
    contract: RunContract,
) -> Mapping[str, Any]:
    metadata = _mapping(config.get("metadata"), "metadata")
    path = _project_path(
        project_root,
        metadata.get("locked_config_materialization_receipt"),
        label="locked materialization",
    )
    receipt = _strict_json(path, "locked materialization")
    _verified_checksum(receipt, "locked materialization")
    graph = _mapping(receipt.get("graph_contract"), "graph contract")
    parameters = _mapping(receipt.get("parameter_audit"), "parameter audit")
    if (
        receipt.get("receipt_kind") != RECEIPT_KIND
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or _mapping(receipt.get("frozen_contract"), "frozen contract").get(
            "sha256"
        )
        != CONTRACT_SHA256
        or parameters.get("trainable_parameter_count")
        != EXPECTED_PARAMETER_COUNT
        or parameters.get("uses_graph_inputs") is not False
        or parameters.get("uses_edge_inputs") is not False
        or graph.get("construction_performed") is not False
        or graph.get("model_graph_inputs") is not False
        or graph.get("model_edge_inputs") is not False
        or int(graph.get("expected_directed_edges", -1)) != 0
        or receipt.get("training_performed") is not False
    ):
        raise SelfHurdleRunnerError("locked materialization is incompatible")
    jobs = receipt.get(
        "resource_jobs" if contract.resource_pilot else "science_jobs"
    )
    if not isinstance(jobs, list):
        raise SelfHurdleRunnerError("locked job list is missing")
    matches = [
        _mapping(item, "locked job")
        for item in jobs
        if _mapping(item, "locked job").get("alias") == contract.alias
    ]
    if (
        len(matches) != 1
        or matches[0].get("config_sha256") != canonical_sha256(config)
    ):
        raise SelfHurdleRunnerError(
            "resolved config is not the locked materialized job"
        )
    return receipt


def _verify_science_gate(
    project_root: Path,
    materialization: Mapping[str, Any],
    contract: RunContract,
) -> Mapping[str, Any] | None:
    if contract.resource_pilot:
        return None
    path = _project_path(
        project_root,
        materialization.get("resource_gate_receipt_reference"),
        label="resource gate",
    )
    gate = _strict_json(path, "resource gate")
    _verified_checksum(gate, "resource gate")
    if (
        gate.get("receipt_kind") != RESOURCE_GATE_KIND
        or gate.get("campaign_id") != CAMPAIGN_ID
        or gate.get("materialization_checksum")
        != materialization.get("checksum")
        or gate.get("passed") is not True
        or gate.get("science_authorized") is not True
        or float(gate.get("projected_aggregate_science_gpu_hours", math.inf))
        > AGGREGATE_PROJECTED_HOURS_MAX
    ):
        raise SelfHurdleRunnerError("resource gate does not authorize science")
    jobs = gate.get("jobs")
    if not isinstance(jobs, list) or {
        str(_mapping(item, "gate job").get("alias")) for item in jobs
    } != set(ALIASES):
        raise SelfHurdleRunnerError("resource gate job matrix is incomplete")
    for raw in jobs:
        item = _mapping(raw, "gate job")
        bundle = _project_path(
            project_root,
            item.get("artifact_bundle"),
            label="resource run bundle",
            require_file=False,
        )
        marker = bundle / "_SUCCESS"
        config_path = bundle / "config.resolved.yaml"
        diagnostics_path = bundle / "diagnostics/resource_usage.json"
        summary_path = bundle / "summary.json"
        if (
            not bundle.is_dir()
            or not marker.is_file()
            or sha256_file(marker) != item.get("success_marker_sha256")
            or not config_path.is_file()
            or canonical_sha256(load_yaml_mapping(config_path))
            != item.get("config_sha256")
            or not diagnostics_path.is_file()
            or not summary_path.is_file()
        ):
            raise SelfHurdleRunnerError(
                "resource gate bundle evidence no longer verifies"
            )
        diagnostics = _strict_json(diagnostics_path, "resource diagnostics")
        summary = _strict_json(summary_path, "resource summary")
        if (
            summary.get("run_id") != item.get("run_id")
            or summary.get("status") != "success"
            or diagnostics.get("runner_pilot_gate_passed") is not True
            or float(diagnostics.get("peak_allocated_vram_gib", math.inf))
            > PILOT_PEAK_GIB_MAX
            or float(
                diagnostics.get(
                    "fp32_amp_absolute_loss_discrepancy", math.inf
                )
            )
            > AMP_DISCREPANCY_MAX
            or float(
                diagnostics.get(
                    "projected_gpu_hours_per_200_epochs", math.inf
                )
            )
            > PER_RUN_PROJECTED_HOURS_MAX
            or int(diagnostics.get("graph_construction_count", -1)) != 0
            or int(diagnostics.get("graph_input_tensor_count", -1)) != 0
        ):
            raise SelfHurdleRunnerError(
                "resource gate evidence fails frozen thresholds"
            )
    return gate


def _array_sha(name: str, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _fit_references(
    counts: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> PerGeneReferences:
    raw = validate_raw_counts(counts, name="expression_counts")
    mean = np.asarray(expression_mean, dtype=np.float64)
    scale = np.asarray(expression_scale, dtype=np.float64)
    positive = raw > 0
    support = positive.sum(axis=0, dtype=np.int64)
    if np.any(support == 0):
        raise SelfHurdleRunnerError(
            "every gene needs positive support for the per-gene reference"
        )
    probability = (support.astype(np.float64) + 0.5) / (
        float(raw.shape[0]) + 1.0
    )
    transformed = np.log1p(raw.astype(np.float64, copy=False))
    continuous = np.asarray(
        [
            (
                float(np.median(transformed[positive[:, gene], gene]))
                - mean[gene]
            )
            / scale[gene]
            for gene in range(raw.shape[1])
        ]
    )
    audit = {
        "schema": "self_hurdle_all_fit_per_gene_references_v1",
        "fit_scope": "all_nodes_transductive",
        "smoothing": "Jeffreys_add_half",
        "positive_continuous_statistic": "per_gene_positive_median_log1p",
        "positive_support_minimum": int(support.min()),
        "positive_support_maximum": int(support.max()),
        "reference_sha256": canonical_sha256(
            {
                "detection": _array_sha("detection", probability),
                "positive": _array_sha("positive", continuous),
            }
        ),
    }
    return PerGeneReferences(probability, continuous, audit)


def _reference_metrics(
    reference: PerGeneReferences,
    result: SelfHurdleFixedMaskResult,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> dict[str, Any]:
    probability = torch.from_numpy(
        reference.detection_probability.astype(np.float32)
    ).clamp(1e-7, 1.0 - 1e-7)
    continuous = torch.from_numpy(
        reference.positive_continuous_standardized.astype(np.float32)
    )
    one = torch.stack((torch.logit(probability), continuous), dim=-1)
    prediction = one.unsqueeze(0).expand(result.target.shape[0], -1, -1)
    evaluation = evaluate_hurdle_continuous_output(
        prediction,
        result.target,
        result.target_mask,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
    )
    return {
        f"reference_per_gene_{name}": value
        for name, value in evaluation.metrics.items()
    }


def _replicate_row(
    entry: Mapping[str, Any],
    result: SelfHurdleFixedMaskResult,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = json_safe_metrics(
        {**result.evaluation.metrics, **dict(reference)}
    )
    for field in (
        "hurdle_loss",
        "positive_continuous_huber",
        "positive_count_state_mae",
    ):
        baseline = float(metrics[f"reference_per_gene_{field}"])
        observed = float(metrics[field])
        metrics[f"{field}_relative_improvement_over_per_gene_reference"] = (
            None if baseline <= 0 else (baseline - observed) / baseline
        )
    metrics[
        "detection_balanced_accuracy_gain_over_per_gene_reference"
    ] = float(metrics["detection_balanced_accuracy"]) - float(
        metrics["reference_per_gene_detection_balanced_accuracy"]
    )
    for field in PERCENT_FIELDS:
        for prefix in ("", "reference_per_gene_"):
            name = f"{prefix}{field}"
            value = metrics.get(name)
            metrics[f"{name}_percent"] = (
                None if value is None else 100.0 * float(value)
            )
    return {
        "split": "fit",
        "mask_mode": PUBLIC_MASK_NAMES[str(entry["spec"]["mode"])],
        "mask_replicate": int(entry["replicate"]),
        "mask_entry_id": str(entry["entry_id"]),
        "mask_seed": int(entry["seed"]),
        "mask_checksum": str(entry["mask_checksum"]),
        **metrics,
    }


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    training: SelfHurdleTrainingResult,
    durations: Mapping[str, float],
    checkpoint_size: int,
    peak_bytes: int,
    projected_hours: float | None,
    disk: Mapping[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in ROW_METADATA
            and key not in {"n_masked", "n_zero", "n_positive"}
            and (_finite(value) or value is None)
        }
    )
    for mode in REQUIRED_MASKS:
        selected = [row for row in rows if row["mask_mode"] == mode]
        if len(selected) != replicates:
            raise SelfHurdleRunnerError(
                f"expected {replicates} rows for {mode}"
            )
        for field in fields:
            values = [
                float(row[field])
                for row in selected
                if _finite(row.get(field))
            ]
            output[f"fit/{mode}/{field}"] = (
                None if not values else float(np.mean(values))
            )
    output.update(
        {
            "resource/data_preparation_duration_seconds": durations["data"],
            "resource/graph_construction_duration_seconds": 0.0,
            "resource/training_duration_seconds": durations["training"],
            "resource/inference_duration_seconds": durations["evaluation"],
            "resource/total_duration_seconds": durations["total"],
            "resource/parameter_count": EXPECTED_PARAMETER_COUNT,
            "resource/checkpoint_size_bytes": checkpoint_size,
            "resource/peak_allocated_vram_bytes": peak_bytes,
            "resource/peak_allocated_vram_gib": peak_bytes / (1024**3),
            "resource/projected_gpu_hours_per_200_epochs": projected_hours,
            "resource/filesystem_used_decimal_gb": disk[
                "used_decimal_gb"
            ],
            "graph/construction_count": 0,
            "graph/input_tensor_count": 0,
            "training/final_epoch": training.final_epoch,
            "training/final_hurdle_loss": training.final_train_loss,
        }
    )
    return output


def _whole_node_zero(bundle: Any) -> np.ndarray:
    matches = [
        entry
        for entry in bundle.manifest["entries"]
        if str(entry["spec"]["mode"]) == "node"
        and int(entry["replicate"]) == 0
    ]
    if len(matches) != 1:
        raise SelfHurdleRunnerError(
            "fixed masks lack unique whole-node replicate zero"
        )
    return bundle.masks[str(matches[0]["entry_id"])]


def _checkpoint(
    *,
    archive: RunArchive,
    config: Mapping[str, Any],
    training: SelfHurdleTrainingResult,
    core: FullCoreData,
    mask_checksum: str,
    references: PerGeneReferences,
) -> bytes:
    payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "checkpoint_role": "last",
        "checkpoint_policy": training.checkpoint_policy,
        "training_protocol": training.training_protocol,
        "model_name": "self-hurdle-count",
        "model_config": dict(_mapping(config.get("model"), "model")),
        "epoch": training.final_epoch,
        "fixed_epoch_budget": training.fixed_epoch_budget,
        "model_state_dict": dict(training.final_state_dict),
        "state_dict_sha256": training.final_state_checksum,
        "full_core_preprocessing_sha256": (
            core.checksums.preprocessing_sha256
        ),
        "evaluation_mask_bundle_sha256": mask_checksum,
        "per_gene_reference_sha256": references.audit[
            "reference_sha256"
        ],
        "graph_inputs": False,
        "edge_inputs": False,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _prediction_rows(
    *,
    archive: RunArchive,
    dataset: Mapping[str, Any],
    entry: Mapping[str, Any],
    result: SelfHurdleFixedMaskResult,
    salt: str,
    fold: int,
) -> Iterator[dict[str, Any]]:
    target = result.target.numpy()
    mask = result.target_mask.numpy().astype(bool, copy=False)
    evaluation = result.evaluation
    reconstructed = evaluation.reconstructed_count.numpy()
    state = evaluation.count_state.numpy()
    positive_state = evaluation.positive_count_state.numpy()
    detected = evaluation.detected.numpy()
    continuous = evaluation.positive_continuous_standardized.numpy()
    positive_count = evaluation.positive_reconstructed_count.numpy()
    namespace = (
        f"bagm:{dataset['dataset_id']}:{dataset['version']}:self-fit"
    )
    batch: list[dict[str, Any]] = []
    for local, node in enumerate(result.target_nodes.numpy()):
        genes = np.flatnonzero(mask[local])
        truth = np.rint(target[local, genes]).astype(np.int32)
        batch.append(
            {
                "_protected_local_index": int(node),
                "run_id": archive.run_id,
                "graph_id": "self_only_no_graph",
                "dataset_id": str(dataset["dataset_id"]),
                "split": "fit",
                "fold": fold,
                "y_true": truth.astype(int).tolist(),
                "y_pred": reconstructed[local, genes].astype(float).tolist(),
                "target_indices": genes.astype(int).tolist(),
                "y_true_state": tokenize_raw_counts(
                    truth.reshape(1, -1)
                ).reshape(-1).astype(int).tolist(),
                "y_pred_state": state[local, genes].astype(int).tolist(),
                "y_pred_positive_state": positive_state[
                    local, genes
                ].astype(int).tolist(),
                "y_pred_detected": detected[
                    local, genes
                ].astype(int).tolist(),
                "y_pred_positive_count": positive_count[
                    local, genes
                ].astype(float).tolist(),
                "y_pred_standardized_log1p": continuous[
                    local, genes
                ].astype(float).tolist(),
                "node_count": int(result.full_mask.shape[0]),
                "edge_count": 0,
                "effective_mask_rate": float(
                    genes.size / result.full_mask.shape[1]
                ),
                "masking_type": "whole_node",
                "mask_replicate": int(entry["replicate"]),
            }
        )
        if len(batch) >= 128:
            yield from deidentify_prediction_rows(
                batch,
                identifier_fields=["_protected_local_index"],
                salt=salt,
                namespace=namespace,
            )
            batch.clear()
    if batch:
        yield from deidentify_prediction_rows(
            batch,
            identifier_fields=["_protected_local_index"],
            salt=salt,
            namespace=namespace,
        )


def run_self_hurdle_capacity(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    sample_key_salt: str,
    full_core_data: FullCoreData | None = None,
) -> CapacityRunResult:
    started = time.monotonic()
    if len(sample_key_salt.encode()) < 16:
        raise RunValidationError(
            "BAGM_SAMPLE_KEY_SALT must contain at least 16 bytes"
        )
    contract = _validate_config(config)
    project_root = archive.paths.project_root
    frozen = _verify_contract(project_root)
    materialization = _verify_materialization(
        project_root, config, contract
    )
    gate = _verify_science_gate(project_root, materialization, contract)
    initial_disk = _disk(project_root)
    dataset = _mapping(config.get("dataset"), "dataset")
    model_config = _mapping(config.get("model"), "model")
    evaluation = _mapping(config.get("evaluation"), "evaluation")

    data_started = time.monotonic()
    core = (
        full_core_data
        if full_core_data is not None
        else load_and_refit_full_core(
            _resolve_prepared_artifact(config, archive)
        )
    )
    data_duration = time.monotonic() - data_started
    identity = _verify_materialized_identity(core, dataset)
    counts = validate_raw_counts(
        core.expression_counts, name="expression_counts"
    )
    if (
        core.n_genes != 1000
        or core.node_covariates.shape[1] != len(ALLOWED_METADATA_COLUMNS)
        or tuple(core.metadata_names) != tuple(ALLOWED_METADATA_COLUMNS)
    ):
        raise SelfHurdleRunnerError(
            "prepared core does not match 1000-gene/22-covariate contract"
        )
    references = _fit_references(
        counts,
        expression_mean=core.expression_mean,
        expression_scale=core.expression_scale,
    )
    view = SelfHurdleSplitView(
        expression=torch.from_numpy(
            np.asarray(counts, dtype=np.float32)
        ),
        coordinates_um=torch.from_numpy(
            np.asarray(core.coordinates_um, dtype=np.float64)
        ),
        node_covariates=torch.from_numpy(
            np.asarray(core.node_covariates, dtype=np.float32)
        ),
        block_ids=np.asarray(core.macroblock_ids),
        name="fit",
    )
    training_config = _training_config(config)
    masks = _evaluation_masks(config, core, training_config)
    set_deterministic_seed(
        training_config.model_seed,
        deterministic=training_config.deterministic,
        warn_only=training_config.deterministic_warn_only,
    )
    model = SelfHurdleModel(
        num_genes=core.n_genes,
        expression_mean=core.expression_mean,
        expression_scale=core.expression_scale,
        node_covariate_dim=view.node_covariate_dim,
        hidden_dim=int(model_config["hidden_dim"]),
        decoder_dim=int(model_config["decoder_dim"]),
        ffn_dim=int(model_config["ffn_dim"]),
        residual_blocks=int(model_config["residual_blocks"]),
        dropout=float(model_config["dropout"]),
    )
    if trainable_parameter_count(model) != EXPECTED_PARAMETER_COUNT:
        raise SelfHurdleRunnerError("self-hurdle parameter count drifted")

    archive.write_json(
        "diagnostics/full_core_preprocessing.json",
        core.preprocessing_qc.to_dict(),
    )
    archive.write_json(
        "diagnostics/graph_prohibition.json",
        {
            "graph_construction_count": 0,
            "graph_input_tensor_count": 0,
            "edge_input_tensor_count": 0,
            "coordinates_used_by_model": False,
            "coordinates_used_only_for_spatial_block_masks": True,
        },
    )
    archive.write_json("diagnostics/disk_safety_start.json", initial_disk)
    archive.write_json("provenance/frozen_task_contract.json", frozen)
    archive.write_json(
        "provenance/locked_materialization_identity.json",
        {
            "receipt_kind": materialization["receipt_kind"],
            "checksum": materialization["checksum"],
            "parameter_audit": materialization["parameter_audit"],
            "graph_contract": materialization["graph_contract"],
        },
    )
    archive.write_json(
        "provenance/external_gate_authorization.json",
        {
            "resource_pilot": contract.resource_pilot,
            "resource_gate": (
                None
                if gate is None
                else {
                    "checksum": gate["checksum"],
                    "science_authorized": True,
                }
            ),
        },
    )
    archive.write_json(
        "provenance/full_core_inputs.json",
        {
            "biological_unit_alias": contract.alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "fit_scope": "all_nodes_transductive",
            "generalization_estimate": False,
            "prepared_artifact_reference": (
                dataset["prepared_artifact_reference"]
            ),
            "preprocessing_checksums": core.checksums.to_dict(),
            "materialized_identity_verification": identity,
            "graph_constructed": False,
            "data_preparation_duration_seconds": data_duration,
        },
    )
    archive.write_json(
        "provenance/per_gene_references.json",
        {
            "audit": dict(references.audit),
            "detection_probability": (
                references.detection_probability.tolist()
            ),
            "positive_continuous_standardized": (
                references.positive_continuous_standardized.tolist()
            ),
        },
    )
    archive.write_json(
        "provenance/fixed_evaluation_masks.json",
        {
            "bundle_manifest": masks.manifest,
            "used_for_gradient_updates": False,
            "used_for_checkpoint_selection": False,
            "technical_replicates_not_biological_replicates": True,
        },
    )

    precision: SelfHurdlePrecisionResult | None = None
    if contract.resource_pilot:
        precision = compare_self_hurdle_fp32_amp_loss(
            model,
            view,
            _whole_node_zero(masks),
            expression_mean=core.expression_mean,
            expression_scale=core.expression_scale,
            device=training_config.device or "cuda",
            target_node_batch_size=TARGET_BATCH_SIZE,
            amp_dtype=training_config.amp_dtype,
            maximum_discrepancy=AMP_DISCREPANCY_MAX,
        )
        archive.write_json(
            "diagnostics/fp32_amp_equivalence.json",
            {
                **asdict(precision),
                "maximum_allowed_discrepancy": AMP_DISCREPANCY_MAX,
            },
        )
        if (
            not precision.passed
            or precision.peak_cuda_memory_bytes / (1024**3)
            > PILOT_PEAK_GIB_MAX
        ):
            raise SelfHurdleRunnerError(
                "resource precision/memory smoke failed"
            )

    training_started = time.monotonic()
    training = fit_full_core_self_hurdle_model(
        model,
        view,
        training_config,
        expression_mean=core.expression_mean,
        expression_scale=core.expression_scale,
        target_node_batch_size=TARGET_BATCH_SIZE,
    )
    training_duration = time.monotonic() - training_started
    archive.write_table(
        "metrics/history",
        [
            {
                "run_id": archive.run_id,
                "split": "fit",
                "training_protocol": training.training_protocol,
                **row,
            }
            for row in training.history_rows()
        ],
        fallback="jsonl",
    )
    peak_bytes = max(
        max(record.peak_cuda_memory_bytes for record in training.history),
        0 if precision is None else precision.peak_cuda_memory_bytes,
    )
    projected = (
        float(np.mean([row.duration_seconds for row in training.history]))
        * 200.0
        / 3600.0
        if contract.resource_pilot
        else None
    )
    peak_limit = (
        PILOT_PEAK_GIB_MAX
        if contract.resource_pilot
        else SCIENCE_PEAK_GIB_MAX
    )
    if peak_bytes / (1024**3) > peak_limit:
        raise SelfHurdleRunnerError("run exceeds frozen VRAM limit")
    if contract.resource_pilot and (
        projected is None or projected > PER_RUN_PROJECTED_HOURS_MAX
    ):
        raise SelfHurdleRunnerError(
            "resource pilot exceeds projected per-run time limit"
        )
    _disk(project_root)
    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt",
        _checkpoint(
            archive=archive,
            config=config,
            training=training,
            core=core,
            mask_checksum=masks.checksum,
            references=references,
        ),
    )
    checkpoint_size = checkpoint_path.stat().st_size

    rows: list[dict[str, Any]] = []
    evaluation_started = time.monotonic()

    def predictions() -> Iterator[dict[str, Any]]:
        for entry in masks.manifest["entries"]:
            result = evaluate_fixed_self_hurdle_mask(
                model,
                view,
                masks.masks[str(entry["entry_id"])],
                expression_mean=core.expression_mean,
                expression_scale=core.expression_scale,
                target_node_batch_size=TARGET_BATCH_SIZE,
                device=training_config.device,
                amp=training_config.amp,
                amp_dtype=training_config.amp_dtype,
            )
            reference = _reference_metrics(
                references,
                result,
                expression_mean=core.expression_mean,
                expression_scale=core.expression_scale,
            )
            row = _replicate_row(entry, result, reference)
            rows.append(row)
            mode = str(row["mask_mode"])
            for name, value in row.items():
                if name in ROW_METADATA or not _finite(value):
                    continue
                archive.append_metric_event(
                    {
                        "name": f"fit/{mode}/{name}",
                        "value": value,
                        "mask_replicate": int(entry["replicate"]),
                    }
                )
            if mode == "whole_node" and int(entry["replicate"]) == 0:
                yield from _prediction_rows(
                    archive=archive,
                    dataset=dataset,
                    entry=entry,
                    result=result,
                    salt=sample_key_salt,
                    fold=int(config["fold"]),
                )

    prediction_path = archive.write_prediction_jsonl_stream(
        "fit", predictions()
    )
    evaluation_duration = time.monotonic() - evaluation_started
    archive.write_table(
        "metrics/evaluation_replicates", rows, fallback="jsonl"
    )
    final_disk = _disk(project_root)
    durations = {
        "data": data_duration,
        "training": training_duration,
        "evaluation": evaluation_duration,
        "total": time.monotonic() - started,
    }
    final_metrics = _aggregate_metrics(
        rows,
        replicates=int(evaluation["mask_replicates_per_mode"]),
        training=training,
        durations=durations,
        checkpoint_size=checkpoint_size,
        peak_bytes=peak_bytes,
        projected_hours=projected,
        disk=final_disk,
    )
    primary = final_metrics.get(PRIMARY_METRIC)
    if not _finite(primary):
        raise SelfHurdleRunnerError("primary metric is missing or non-finite")
    for name, value in final_metrics.items():
        if _finite(value):
            archive.append_metric_event(
                {"name": name, "value": value, "phase": "final_aggregate"}
            )
    archive.write_json("metrics/final.json", final_metrics)
    discrepancy = (
        None
        if precision is None
        else precision.absolute_total_loss_discrepancy
    )
    pilot_pass = (
        None
        if not contract.resource_pilot
        else (
            discrepancy is not None
            and discrepancy <= AMP_DISCREPANCY_MAX
            and peak_bytes / (1024**3) <= PILOT_PEAK_GIB_MAX
            and projected is not None
            and projected <= PER_RUN_PROJECTED_HOURS_MAX
            and final_disk["used_decimal_gb"] < DISK_USED_DECIMAL_GB_MAX
        )
    )
    resource = {
        "schema_version": 1,
        "receipt_kind": "self_hurdle_resource_usage_v1",
        "resource_pilot": contract.resource_pilot,
        "biological_unit_alias": contract.alias,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "peak_allocated_vram_gib": peak_bytes / (1024**3),
        "fp32_amp_absolute_loss_discrepancy": discrepancy,
        "projected_gpu_hours_per_200_epochs": projected,
        "filesystem_used_decimal_gb": final_disk["used_decimal_gb"],
        "graph_construction_count": 0,
        "graph_input_tensor_count": 0,
        "edge_input_tensor_count": 0,
        "all_losses_and_gradients_finite": True,
        "all_epochs_completed": len(training.history)
        == training.fixed_epoch_budget,
        "runner_pilot_gate_passed": pilot_pass,
        "thresholds": {
            "pilot_peak_vram_gib_maximum": PILOT_PEAK_GIB_MAX,
            "science_peak_vram_gib_maximum": SCIENCE_PEAK_GIB_MAX,
            "fp32_amp_loss_discrepancy_maximum": AMP_DISCREPANCY_MAX,
            "projected_gpu_hours_per_science_run_maximum": (
                PER_RUN_PROJECTED_HOURS_MAX
            ),
            "filesystem_used_decimal_gb_hard_stop": (
                DISK_USED_DECIMAL_GB_MAX
            ),
        },
    }
    archive.write_json("diagnostics/resource_usage.json", resource)
    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "training_exit_status": "success",
        "campaign_id": CAMPAIGN_ID,
        "biological_unit_alias": contract.alias,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "evaluation_protocol": "held_in_full_core_fixed_budget",
        "task_family": "masked_expression_hurdle_count",
        "canonical_prediction_split": "fit",
        "model_name": "self-hurdle-count",
        "arm": "self-hurdle",
        "model_seed": training_config.model_seed,
        "final_epoch": training.final_epoch,
        "fixed_epoch_budget": training.fixed_epoch_budget,
        "checkpoint_role": "last",
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_value": float(primary),
        "metrics": final_metrics,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "duration_seconds": durations["total"],
        "peak_vram_gib": peak_bytes / (1024**3),
        "peak_host_memory_bytes": _peak_host_memory_bytes(),
        "graph_constructed": False,
        "graph_inputs_used": False,
        "edge_inputs_used": False,
        "evaluation_mask_bundle_sha256": masks.checksum,
        "resource_pilot": contract.resource_pilot,
        "runner_pilot_gate_passed": pilot_pass,
        "fp32_amp_absolute_loss_discrepancy": discrepancy,
        "projected_gpu_hours_per_200_epochs": projected,
        "filesystem_used_decimal_gb": final_disk["used_decimal_gb"],
        "conclusion_eligible": not contract.resource_pilot,
        "generalization_estimate": False,
        "maximum_claim": (
            "diagnostic precision, runtime, and memory feasibility only"
            if contract.resource_pilot
            else (
                "exploratory held-in within-cell representation capacity "
                "for one adjacent-normal core"
            )
        ),
        "prohibited_claims": [
            "unseen-patient generalization",
            "spatial predictive dependency",
            "cell-cell communication",
            "biological mechanism",
            "causal influence",
        ],
    }
    archive.write_summary(summary)
    return CapacityRunResult(
        run_id=archive.run_id,
        model_name="self-hurdle-count",
        primary_metric_name=PRIMARY_METRIC,
        primary_metric_value=float(primary),
        final_epoch=training.final_epoch,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    result = run_self_hurdle_capacity(
        config,
        archive,
        sample_key_salt=os.environ.get("BAGM_SAMPLE_KEY_SALT", ""),
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "model_name": result.model_name,
                "primary_metric_name": result.primary_metric_name,
                "primary_metric_value": result.primary_metric_value,
                "final_epoch": result.final_epoch,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

