#!/usr/bin/env python3
"""Analyze the frozen grouped adjacent-normal adjacency ablation.

The independent units are the ten opaque donor/core aliases.  Evaluation-mask
replicates and model seeds are averaged within each core before the core
bootstrap; neither is promoted to a biological replicate.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import load_yaml_mapping  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.queueing import command_for_config  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


CAMPAIGN_ID = "cmp_20260802_adjacent_normal_grouped_adjacency_ablation"
CONTRACT_SHA256 = (
    "08e4040ce8b0a3535c7bef1cbf896c68bc5e11693cb13bbdfb3b992467742eff"
)
CONDITIONS = ("spatial", "isolated")
NULL_CONDITION = "position_permuted_null"
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
SEEDS = tuple(range(5))
FOLDS = tuple(range(5))
EXPECTED_STAGE_COUNTS = {"smoke": 2, "pilot": 2, "primary": 50, "null": 25}
MATERIALIZATION_KIND = "adjacency_ablation_config_materialization_v1"
NULL_TRIGGER_KIND = "adjacency_ablation_null_trigger_v1"
NULL_ENQUEUE_KIND = "adjacency_ablation_null_enqueue_v1"
RECOVERY_PLAN_KIND = "adjacency_ablation_cpu_recovery_plan_v1"
RECOVERY_ENQUEUE_KIND = "adjacency_ablation_cpu_recovery_enqueue_v1"
RECOVERY_CONTRACT_SHA256 = (
    "84230db441cd03d158a28bc79dfecee495fee1b0864dae2311a0a9041f7b6bb0"
)
DATASET_ID = "cosmx_adjacent_normal_grouped_adjacency_v1"
DATASET_VERSION = "adjacent_normal_grouped_adjacency_v1"
SPLIT_ID = "adjacent_normal_10donor_slide_balanced_5fold_v1"
EXPECTED_PARAMETER_COUNT = 645_736
TARGET_MASK_BINS = (
    "0_to_25",
    "25_to_50",
    "50_to_75",
    "75_to_100",
    "exactly_100",
)
NEIGHBOR_OBSERVED_BINS = (
    "0_to_25",
    "25_to_50",
    "50_to_75",
    "75_to_100",
)
PAIR_SUMMARY_FIELDS = (
    "initial_state_sha256",
    "training_mask_schedule_sha256",
    "validation_selection_mask_schedule_sha256",
    "validation_mask_identity_sha256",
    "test_mask_identity_sha256",
    "prepared_manifest_sha256",
    "prepared_content_sha256",
    "preprocessing_sha256",
    "parameter_count",
    "completed_global_epochs",
    "optimizer_steps",
)
PAIR_DIAGNOSTIC_FIELDS = (
    "train_aliases",
    "validation_aliases",
    "test_aliases",
    "preprocessing_fit_aliases",
    "preprocessing_sha256",
    "initial_state_sha256",
    "training_mask_schedule_sha256",
    "validation_selection_mask_schedule_sha256",
    "validation_evaluation_mask_identity_sha256",
    "test_evaluation_mask_identity_sha256",
    "model_inputs",
    "masked_neighbor_inputs_only",
)
SECONDARY_METRICS = {
    "standardized_huber": "lower",
    "log1p_mse": "lower",
    "log1p_rmse": "lower",
    "gene_pearson": "higher",
    "gene_spearman": "higher",
    "cell_pearson": "higher",
    "cell_spearman": "higher",
}
BOOTSTRAP_SEED = 20260802
BOOTSTRAP_RESAMPLES = 10_000
MEANINGFUL_RELATIVE_IMPROVEMENT = 0.02
LOCK_ROOT = (
    Path("scratch/locked_campaigns") / CAMPAIGN_ID
)
DEFAULT_RECOVERY_PLAN = LOCK_ROOT / "primary_cpu_recovery_plan_receipt.json"
DEFAULT_RECOVERY_ENQUEUE = LOCK_ROOT / "primary_cpu_recovery_enqueue_receipt.json"
DEFAULT_NULL_ENQUEUE = LOCK_ROOT / "null_enqueue_receipt.json"
DEFAULT_REPORT_ROOT = Path(
    "reports/analyses/adjacent_normal_grouped_adjacency_ablation"
)


class AdjacencyAnalysisError(RuntimeError):
    """Raised when incomplete or unpaired evidence cannot be analyzed."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdjacencyAnalysisError(f"{label} must be a mapping")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AdjacencyAnalysisError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def _hex_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AdjacencyAnalysisError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _project_file(reference: object, label: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise AdjacencyAnalysisError(f"{label} reference is missing")
    candidate = PROJECT_ROOT / reference
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(PROJECT_ROOT.resolve())
    except (OSError, ValueError) as error:
        raise AdjacencyAnalysisError(
            f"{label} must resolve to a file under the project root"
        ) from error
    if candidate.is_symlink() or not resolved.is_file():
        raise AdjacencyAnalysisError(f"{label} is not a regular immutable file")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AdjacencyAnalysisError(
            f"{path} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdjacencyAnalysisError(
                    f"{path} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except AdjacencyAnalysisError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AdjacencyAnalysisError(f"cannot read JSON: {path}") from error
    return dict(_mapping(value, str(path)))


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("checksum", None)
    value["checksum"] = canonical_sha256(value)
    return value


def _write_json_idempotent(path: Path, payload: Mapping[str, Any]) -> None:
    normalized = json.loads(
        json.dumps(payload, sort_keys=True, allow_nan=False)
    )
    if path.is_file():
        existing = _read_json(path)
        if existing != normalized:
            raise AdjacencyAnalysisError(
                f"refusing to overwrite a different receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        existing = _read_json(path)
        if existing != normalized:
            raise AdjacencyAnalysisError(
                f"refusing to overwrite a different receipt: {path}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def _materialization(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    checksum = _hex_digest(value.get("checksum"), "materialization checksum")
    unsigned = dict(value)
    unsigned.pop("checksum", None)
    if checksum != canonical_sha256(unsigned):
        raise AdjacencyAnalysisError("materialization receipt checksum mismatch")
    if (
        value.get("schema_version") != 1
        or value.get("receipt_kind") != MATERIALIZATION_KIND
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("counts") != EXPECTED_STAGE_COUNTS
        or value.get("dataset_id") != DATASET_ID
        or value.get("dataset_version") != DATASET_VERSION
        or value.get("split_id") != SPLIT_ID
    ):
        raise AdjacencyAnalysisError(
            "materialization differs from the frozen campaign identity"
        )
    frozen_contract = _mapping(
        value.get("frozen_contract"), "materialization frozen_contract"
    )
    contract_path = _project_file(
        frozen_contract.get("reference"), "materialization frozen contract"
    )
    if (
        frozen_contract.get("sha256") != CONTRACT_SHA256
        or _sha256_file(contract_path) != CONTRACT_SHA256
    ):
        raise AdjacencyAnalysisError("materialization contract binding changed")
    prepared = _mapping(
        value.get("prepared_manifest"), "materialization prepared_manifest"
    )
    manifest_path = _project_file(
        prepared.get("reference"), "materialization prepared manifest"
    )
    if _sha256_file(manifest_path) != prepared.get("file_sha256"):
        raise AdjacencyAnalysisError("prepared manifest file checksum changed")
    try:
        from scripts.train.prepare_adjacency_ablation import (
            verify_prepared_artifact,
        )

        manifest = dict(verify_prepared_artifact(manifest_path))
    except AdjacencyAnalysisError:
        raise
    except Exception as error:
        raise AdjacencyAnalysisError(
            "prepared artifact verification failed"
        ) from error
    manifest_dataset = _mapping(manifest.get("dataset"), "prepared dataset")
    manifest_split = _mapping(manifest.get("split"), "prepared split")
    if (
        manifest.get("content_sha256") != prepared.get("content_sha256")
        or manifest.get("artifact_id") != prepared.get("artifact_id")
        or manifest_dataset.get("dataset_fingerprint")
        != value.get("dataset_fingerprint")
        or manifest_split.get("assignment_fingerprint")
        != value.get("split_fingerprint")
    ):
        raise AdjacencyAnalysisError(
            "prepared artifact no longer matches materialization"
        )
    _planned_jobs(value, verify_files=True)
    return value


def _expected_job_slots() -> set[tuple[str, int, int, str]]:
    slots = {
        (stage, 0, 0, condition)
        for stage in ("smoke", "pilot")
        for condition in CONDITIONS
    }
    slots.update(
        ("primary", fold, seed, condition)
        for fold in FOLDS
        for seed in SEEDS
        for condition in CONDITIONS
    )
    slots.update(
        ("null", fold, seed, NULL_CONDITION)
        for fold in FOLDS
        for seed in SEEDS
    )
    return slots


def _planned_jobs(
    materialization: Mapping[str, Any], *, verify_files: bool = False
) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    jobs = materialization.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 79:
        raise AdjacencyAnalysisError("materialization does not contain 79 jobs")
    planned: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for raw in jobs:
        item = dict(_mapping(raw, "materialized job"))
        key = (
            str(item.get("stage")),
            int(item.get("fold", -1)),
            int(item.get("seed", -1)),
            str(item.get("condition")),
        )
        _hex_digest(item.get("config_sha256"), "materialized config checksum")
        _hex_digest(
            item.get("config_file_sha256"),
            "materialized config file checksum",
        )
        if key in planned:
            raise AdjacencyAnalysisError("materialization job identity is invalid")
        if verify_files:
            config_path = _project_file(
                item.get("config_reference"), f"locked config {key}"
            )
            if _sha256_file(config_path) != item.get("config_file_sha256"):
                raise AdjacencyAnalysisError(f"locked config changed for {key}")
            config = load_yaml_mapping(config_path)
            experiment = _mapping(config.get("experiment"), "config.experiment")
            graph = _mapping(config.get("graph"), "config.graph")
            if (
                canonical_sha256(config) != item.get("config_sha256")
                or str(experiment.get("stage")) != key[0]
                or int(config.get("fold", -1)) != key[1]
                or int(config.get("seed", -1)) != key[2]
                or str(graph.get("adjacency_condition")) != key[3]
            ):
                raise AdjacencyAnalysisError(
                    f"locked config identity changed for {key}"
                )
        planned[key] = item
    if set(planned) != _expected_job_slots():
        raise AdjacencyAnalysisError("materialization job grid changed")
    return planned


def _resolved_attempt_config(
    materialized_job: Mapping[str, Any], *, attempt: int
) -> tuple[dict[str, Any], str]:
    config_path = _project_file(
        materialized_job.get("config_reference"), "materialized recovery config"
    )
    config = dict(load_yaml_mapping(config_path))
    if canonical_sha256(config) != materialized_job.get("config_sha256"):
        raise AdjacencyAnalysisError("materialized recovery config changed")
    config["attempt"] = int(attempt)
    return config, canonical_sha256(config)


def _normalize_attempt(config: Mapping[str, Any], *, attempt: int = 1) -> dict[str, Any]:
    normalized = deepcopy(dict(config))
    normalized["attempt"] = int(attempt)
    return normalized


def _recovery_authorization(
    plan_path: Path,
    enqueue_path: Path,
    *,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the producer-verified recovery receipts and bind frozen configs."""

    try:
        from scripts.train.recover_adjacency_gpu_failure import (
            load_recovery_enqueue,
            load_recovery_plan,
        )

        plan = dict(load_recovery_plan(plan_path))
        enqueue = dict(load_recovery_enqueue(enqueue_path, plan=plan))
    except Exception as error:
        raise AdjacencyAnalysisError(
            "CPU recovery plan/enqueue receipt verification failed"
        ) from error
    if (
        plan.get("schema_version") != 1
        or plan.get("receipt_kind") != RECOVERY_PLAN_KIND
        or plan.get("campaign_id") != CAMPAIGN_ID
        or _mapping(plan.get("recovery_contract"), "recovery contract").get(
            "sha256"
        )
        != RECOVERY_CONTRACT_SHA256
        or _mapping(plan.get("materialization"), "recovery materialization").get(
            "checksum"
        )
        != materialization.get("checksum")
        or plan.get("scientific_contract_sha256") != CONTRACT_SHA256
        or enqueue.get("schema_version") != 1
        or enqueue.get("receipt_kind") != RECOVERY_ENQUEUE_KIND
        or enqueue.get("campaign_id") != CAMPAIGN_ID
        or enqueue.get("plan_checksum") != plan.get("checksum")
    ):
        raise AdjacencyAnalysisError(
            "CPU recovery receipts differ from the frozen campaign"
        )
    recovery_contract = _mapping(
        plan.get("recovery_contract"), "recovery contract"
    )
    recovery_contract_path = _project_file(
        recovery_contract.get("reference"), "recovery contract"
    )
    canonical_plan_path = _project_file(
        enqueue.get("plan_reference"), "canonical recovery plan"
    )
    if (
        _sha256_file(recovery_contract_path) != RECOVERY_CONTRACT_SHA256
        or _sha256_file(canonical_plan_path) != _sha256_file(plan_path)
    ):
        raise AdjacencyAnalysisError(
            "CPU recovery contract or canonical plan file changed"
        )
    planned = _planned_jobs(materialization)
    raw_primary = plan.get("primary_jobs")
    raw_null = plan.get("conditional_null_jobs")
    raw_enqueue = enqueue.get("primary_retries")
    if (
        not isinstance(raw_primary, list)
        or len(raw_primary) != 50
        or not isinstance(raw_null, list)
        or len(raw_null) != 25
        or not isinstance(raw_enqueue, list)
        or len(raw_enqueue) != 50
    ):
        raise AdjacencyAnalysisError("CPU recovery receipt job grids are incomplete")
    enqueue_by_slot: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for raw in raw_enqueue:
        item = _mapping(raw, "recovery enqueue job")
        key = (int(item.get("fold", -1)), int(item.get("seed", -1)), str(item.get("condition")))
        if key in enqueue_by_slot:
            raise AdjacencyAnalysisError("recovery enqueue repeats a primary slot")
        enqueue_by_slot[key] = item
    primary: dict[tuple[int, int, str], dict[str, Any]] = {}
    root_completed = 0
    root_failed = 0
    for raw in raw_primary:
        item = _mapping(raw, "recovery primary job")
        key = (int(item.get("fold", -1)), int(item.get("seed", -1)), str(item.get("condition")))
        planned_job = planned.get(("primary", *key))
        queued = enqueue_by_slot.get(key)
        if planned_job is None or queued is None or key in primary:
            raise AdjacencyAnalysisError("recovery primary slot is unauthorized")
        resolved_config, resolved_sha = _resolved_attempt_config(
            planned_job, attempt=2
        )
        root_status = str(item.get("root_status"))
        root_completed += int(root_status == "completed")
        root_failed += int(root_status == "failed")
        retry_job_id = str(item.get("retry_job_id", ""))
        if (
            item.get("canonical_config_sha256") != planned_job["config_sha256"]
            or item.get("config_reference") != planned_job["config_reference"]
            or item.get("command_sha256")
            != canonical_sha256(command_for_config(resolved_config))
            or int(item.get("attempt", 2)) != 2
            or not retry_job_id
            or queued.get("retry_job_id") != retry_job_id
            or queued.get("root_job_id") != item.get("root_job_id")
            or queued.get("canonical_config_sha256")
            != planned_job["config_sha256"]
            or queued.get("command_sha256") != item.get("command_sha256")
            or int(queued.get("attempt_count", -1)) != 2
            or int(queued.get("maximum_attempts", -1)) != 2
            or queued.get("retry_of") != item.get("root_job_id")
        ):
            raise AdjacencyAnalysisError(
                f"recovery primary identity changed for {key}"
            )
        primary[key] = {
            "plan": dict(item),
            "enqueue": dict(queued),
            "materialized_job": planned_job,
            "resolved_config": resolved_config,
            "resolved_config_sha256": resolved_sha,
            "retry_job_id": retry_job_id,
        }
    expected_primary = {
        (fold, seed, condition)
        for fold in FOLDS for seed in SEEDS for condition in CONDITIONS
    }
    if set(primary) != expected_primary or set(enqueue_by_slot) != expected_primary:
        raise AdjacencyAnalysisError("recovery primary grid differs from 50 slots")
    null: dict[tuple[int, int, str], dict[str, Any]] = {}
    for raw in raw_null:
        item = _mapping(raw, "conditional recovery null job")
        key = (int(item.get("fold", -1)), int(item.get("seed", -1)), str(item.get("condition")))
        planned_job = planned.get(("null", *key))
        if planned_job is None or key in null:
            raise AdjacencyAnalysisError("conditional recovery null slot is invalid")
        resolved_config, resolved_sha = _resolved_attempt_config(
            planned_job, attempt=1
        )
        if (
            key[2] != NULL_CONDITION
            or int(item.get("attempt", -1)) != 1
            or item.get("canonical_config_sha256") != planned_job["config_sha256"]
            or item.get("config_reference") != planned_job["config_reference"]
            or item.get("command_sha256")
            != canonical_sha256(command_for_config(resolved_config))
            or resolved_sha != planned_job["config_sha256"]
        ):
            raise AdjacencyAnalysisError(
                f"conditional recovery null identity changed for {key}"
            )
        null[key] = {
            "plan": dict(item),
            "materialized_job": planned_job,
            "resolved_config": resolved_config,
            "resolved_config_sha256": resolved_sha,
        }
    expected_null = {
        (fold, seed, NULL_CONDITION) for fold in FOLDS for seed in SEEDS
    }
    inventory = _mapping(plan.get("inventory"), "recovery inventory")
    if (
        set(null) != expected_null
        or root_completed != 7
        or root_failed != 43
        or inventory.get("primary_completed_attempt_1") != 7
        or inventory.get("primary_failed_attempt_1") != 43
        or inventory.get("primary_retry_slots") != 50
        or inventory.get("conditional_null_slots") != 25
    ):
        raise AdjacencyAnalysisError("recovery inventory differs from frozen 7/43/50/25")
    return {
        "plan": plan,
        "enqueue": enqueue,
        "recovery_contract_reference": recovery_contract["reference"],
        "recovery_contract_sha256": RECOVERY_CONTRACT_SHA256,
        "recovery_plan_reference": enqueue["plan_reference"],
        "recovery_plan_file_sha256": _sha256_file(canonical_plan_path),
        "primary": primary,
        "null": null,
        "attempt_1_completed": root_completed,
        "attempt_1_failed": root_failed,
    }


def _run_bundles(
    artifact_root: Path,
    materialization: Mapping[str, Any],
    *,
    recovery: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    planned = _planned_jobs(materialization)
    records: list[dict[str, Any]] = []
    for marker in sorted((artifact_root / "runs").glob("*/*/*/_SUCCESS")):
        root = marker.parent
        summary_path = root / "summary.json"
        config_path = root / "config.resolved.yaml"
        manifest_path = root / "manifest.yaml"
        if (
            not summary_path.is_file()
            or not config_path.is_file()
            or not manifest_path.is_file()
        ):
            continue
        summary = _read_json(summary_path)
        if summary.get("campaign_id") != CAMPAIGN_ID:
            continue
        config = load_yaml_mapping(config_path)
        if _mapping(config.get("campaign"), "config.campaign").get(
            "campaign_id"
        ) != CAMPAIGN_ID:
            raise AdjacencyAnalysisError(f"campaign mismatch in {root}")
        experiment = _mapping(config.get("experiment"), "config.experiment")
        graph = _mapping(config.get("graph"), "config.graph")
        run_manifest = load_yaml_mapping(manifest_path)
        stage = str(experiment.get("stage"))
        condition = str(graph.get("adjacency_condition"))
        fold = int(config["fold"])
        seed = int(config["seed"])
        attempt = int(config.get("attempt", -1))
        job_id = str(run_manifest.get("job_id", ""))
        config_sha = canonical_sha256(config)
        audit_errors: list[str] = []
        selection_role = "materialized"
        try:
            verify_run_bundle(root)
        except Exception as error:
            audit_errors.append(f"run bundle verification failed: {error}")
        key = (stage, fold, seed, condition)
        planned_job = planned.get(key)
        if planned_job is None:
            audit_errors.append(f"run is absent from materialization: {key}")
        elif stage == "primary" and attempt == 2:
            authorization = (
                None
                if recovery is None
                else _mapping(recovery.get("primary"), "recovery primary").get(
                    (fold, seed, condition)
                )
            )
            if authorization is None:
                selection_role = "unauthorized_primary_retry"
                audit_errors.append("attempt-2 primary run has no recovery authorization")
            else:
                authorized = _mapping(authorization, "recovery primary slot")
                if (
                    config_sha != authorized.get("resolved_config_sha256")
                    or _normalize_attempt(config)
                    != _normalize_attempt(
                        _mapping(
                            authorized.get("resolved_config"),
                            "authorized resolved config",
                        )
                    )
                    or job_id != authorized.get("retry_job_id")
                ):
                    selection_role = "unauthorized_primary_retry"
                    audit_errors.append(
                        "attempt-2 primary config or queue job differs from recovery receipts"
                    )
                else:
                    selection_role = "recovery_primary_attempt_2"
        elif stage == "primary" and attempt == 1 and recovery is not None:
            authorization = _mapping(
                recovery.get("primary"), "recovery primary"
            ).get((fold, seed, condition))
            if (
                authorization is None
                or config_sha != planned_job["config_sha256"]
                or str(_mapping(authorization, "recovery slot")["plan"].get(
                    "root_job_id"
                ))
                != job_id
                or str(_mapping(authorization, "recovery slot")["plan"].get(
                    "root_run_id"
                ))
                != root.name
                or _mapping(authorization, "recovery slot")["plan"].get(
                    "root_status"
                )
                != "completed"
            ):
                selection_role = "unauthorized_primary_attempt_1"
                audit_errors.append(
                    "attempt-1 primary success differs from recovery inventory"
                )
            else:
                selection_role = "historical_primary_attempt_1"
        elif stage == "null" and recovery is not None:
            authorization = _mapping(recovery.get("null"), "recovery null").get(
                (fold, seed, condition)
            )
            if (
                attempt != 1
                or authorization is None
                or config_sha
                != _mapping(authorization, "recovery null slot").get(
                    "resolved_config_sha256"
                )
                or config
                != _mapping(authorization, "recovery null slot").get(
                    "resolved_config"
                )
            ):
                selection_role = "unauthorized_recovery_null"
                audit_errors.append(
                    "null run differs from conditional CPU recovery authorization"
                )
            else:
                selection_role = "recovery_null_attempt_1"
        elif config_sha != planned_job["config_sha256"] or attempt != 1:
            audit_errors.append("resolved config differs from materialization")
        if stage != summary.get("stage"):
            audit_errors.append("summary stage differs from resolved config")
        if condition != summary.get("condition"):
            audit_errors.append("summary condition differs from resolved config")
        if fold != summary.get("fold"):
            audit_errors.append("summary fold differs from resolved config")
        if seed != summary.get("model_seed"):
            audit_errors.append("summary model seed differs from resolved config")
        if summary.get("config_sha256") != config_sha:
            audit_errors.append("summary config checksum mismatch")
        if summary.get("run_id") != root.name:
            audit_errors.append("summary run_id differs from bundle path")
        if summary.get("status") != "success":
            audit_errors.append("summary does not declare success")
        if run_manifest.get("run_id") != root.name or not job_id:
            audit_errors.append("run manifest identity or queue job is invalid")
        records.append(
            {
                "root": root,
                "run_id": summary.get("run_id", root.name),
                "summary": summary,
                "config": config,
                "config_sha256": config_sha,
                "materialized_job": planned_job,
                "stage": stage,
                "condition": condition,
                "fold": fold,
                "seed": seed,
                "attempt": attempt,
                "job_id": job_id,
                "selection_role": selection_role,
                "required_execution_device": (
                    "cpu"
                    if selection_role
                    in {"recovery_primary_attempt_2", "recovery_null_attempt_1"}
                    else None
                ),
                "required_recovery_audit": (
                    {
                        "execution_device": "cpu",
                        "execution_mode": "cpu_hardware_recovery",
                        "queue_job_id": job_id,
                        "torch_intraop_threads": 4,
                        "torch_interop_threads": 1,
                        "hardware_recovery_used": True,
                        "recovery_contract_reference": recovery[
                            "recovery_contract_reference"
                        ],
                        "recovery_contract_sha256": recovery[
                            "recovery_contract_sha256"
                        ],
                        "recovery_plan_reference": recovery[
                            "recovery_plan_reference"
                        ],
                        "recovery_plan_checksum": _mapping(
                            recovery.get("plan"), "recovery plan"
                        )["checksum"],
                        "recovery_plan_file_sha256": recovery[
                            "recovery_plan_file_sha256"
                        ],
                        "recovery_worker_slot": int(
                            _mapping(
                                _mapping(
                                    (
                                        recovery.get("primary")
                                        if selection_role
                                        == "recovery_primary_attempt_2"
                                        else recovery.get("null")
                                    ),
                                    "recovery slots",
                                ).get((fold, seed, condition)),
                                "recovery slot",
                            )["plan"]["worker_slot"]
                        ),
                        "recovery_root_job_id": (
                            _mapping(
                                _mapping(
                                    recovery.get("primary"), "recovery primary"
                                ).get((fold, seed, condition)),
                                "recovery primary slot",
                            )["plan"]["root_job_id"]
                            if selection_role == "recovery_primary_attempt_2"
                            else None
                        ),
                    }
                    if recovery is not None
                    and selection_role
                    in {"recovery_primary_attempt_2", "recovery_null_attempt_1"}
                    else None
                ),
                "audit_errors": audit_errors,
            }
        )
    return records


def _select_unique_runs(
    records: Iterable[Mapping[str, Any]],
    *,
    stage: str,
    conditions: Sequence[str],
    expected_keys: set[tuple[int, int, str]],
    selection_roles: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected: dict[tuple[int, int, str], dict[str, Any]] = {}
    for raw in records:
        row = dict(raw)
        if (
            row.get("stage") != stage
            or row.get("condition") not in conditions
            or (
                selection_roles is not None
                and row.get("selection_role") not in selection_roles
            )
        ):
            continue
        key = (int(row["fold"]), int(row["seed"]), str(row["condition"]))
        if key in selected:
            raise AdjacencyAnalysisError(
                f"multiple successful {stage} runs for {key}"
            )
        selected[key] = row
    missing = expected_keys.difference(selected)
    unexpected = set(selected).difference(expected_keys)
    if missing or unexpected:
        raise AdjacencyAnalysisError(
            f"{stage} run grid mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return [selected[key] for key in sorted(selected)]


def _read_table(root: Path, relative_stem: str) -> pd.DataFrame:
    candidates = (
        root / f"{relative_stem}.parquet",
        root / f"{relative_stem}.jsonl",
        root / f"{relative_stem}.csv",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise AdjacencyAnalysisError(
            f"expected one table for {root / relative_stem}; found {existing}"
        )
    path = existing[0]
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def _attach_run_columns(frame: pd.DataFrame, run: Mapping[str, Any]) -> pd.DataFrame:
    value = frame.copy()
    expected: dict[str, object] = {
        "run_id": str(run["run_id"]),
        "config_sha256": str(run["config_sha256"]),
        "fold": int(run["fold"]),
        "model_seed": int(run["seed"]),
        "condition": str(run["condition"]),
        "attempt": int(run.get("attempt", 1)),
        "job_id": str(run.get("job_id", "")),
    }
    for column, expected_value in expected.items():
        if column in value.columns:
            if value[column].isna().any():
                raise AdjacencyAnalysisError(
                    f"{run['run_id']} table contains missing {column} provenance"
                )
            if isinstance(expected_value, int):
                numeric = pd.to_numeric(value[column], errors="coerce")
                matches = numeric.notna() & (numeric == expected_value)
            else:
                matches = value[column].astype(str) == str(expected_value)
            if not bool(matches.all()):
                raise AdjacencyAnalysisError(
                    f"refusing to overwrite conflicting {column} provenance "
                    f"in {run['run_id']}"
                )
        else:
            value[column] = expected_value
    return value


def _load_primary_tables(
    runs: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main: list[pd.DataFrame] = []
    mask_bins: list[pd.DataFrame] = []
    neighbor_bins: list[pd.DataFrame] = []
    for run in runs:
        root = Path(str(run["root"]))
        main.append(
            _attach_run_columns(
                _read_table(root, "metrics/per_core_replicate"), run
            )
        )
        mask_bins.append(
            _attach_run_columns(
                _read_table(root, "metrics/per_target_mask_bin"), run
            )
        )
        neighbor_bins.append(
            _attach_run_columns(
                _read_table(root, "metrics/per_neighbor_observation_bin"), run
            )
        )
    return (
        pd.concat(main, ignore_index=True),
        pd.concat(mask_bins, ignore_index=True),
        pd.concat(neighbor_bins, ignore_index=True),
    )


def _bound_prepared_manifest(
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = _mapping(
        materialization.get("prepared_manifest"),
        "materialization prepared manifest",
    )
    path = _project_file(prepared.get("reference"), "prepared manifest")
    if _sha256_file(path) != prepared.get("file_sha256"):
        raise AdjacencyAnalysisError("prepared manifest file checksum changed")
    try:
        from scripts.train.prepare_adjacency_ablation import (
            verify_prepared_artifact,
        )

        manifest = dict(verify_prepared_artifact(path))
    except Exception as error:
        raise AdjacencyAnalysisError(
            "prepared artifact verification failed"
        ) from error
    if (
        manifest.get("content_sha256") != prepared.get("content_sha256")
        or manifest.get("artifact_id") != prepared.get("artifact_id")
    ):
        raise AdjacencyAnalysisError(
            "prepared manifest identity differs from materialization"
        )
    return manifest


def _shared_config_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "dataset",
        "features",
        "graph",
        "masking",
        "preprocessing",
        "model",
        "trainer",
        "evaluation",
        "seed",
        "fold",
    )
    result = {field: deepcopy(config.get(field)) for field in fields}
    graph = _mapping(result.get("graph"), "paired config graph")
    graph.pop("adjacency_condition", None)
    return result


def _scientific_run_audit(
    runs: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    paired_conditions: Sequence[str],
    require_pilot_resources: bool = False,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Audit immutable provenance and exact paired experimental controls."""

    errors: list[str] = []
    evidence: dict[str, dict[str, Any]] = {}
    _mapping(manifest.get("files"), "prepared files")
    folds = _mapping(manifest.get("folds"), "prepared folds")
    prepared_reference = str(
        _mapping(
            runs[0]["config"].get("dataset"), "config dataset"
        ).get("prepared_manifest", "")
    ) if runs else ""
    prepared_manifest_sha = (
        str(_mapping(runs[0]["config"].get("dataset"), "config dataset").get(
            "prepared_manifest_sha256", ""
        ))
        if runs
        else ""
    )
    for run in runs:
        run_id = str(run["run_id"])
        for message in run.get("audit_errors", []):
            errors.append(f"{run_id}: {message}")
        summary = _mapping(run.get("summary"), f"{run_id} summary")
        config = _mapping(run.get("config"), f"{run_id} config")
        root = Path(str(run["root"]))
        try:
            pairing = _read_json(
                root / "diagnostics/pairing_and_leakage_audit.json"
            )
            scientific = _read_json(root / "provenance/scientific_inputs.json")
            resources = (
                _read_json(root / "diagnostics/resources.json")
                if run.get("required_recovery_audit") is not None
                else {}
            )
        except AdjacencyAnalysisError as error:
            errors.append(f"{run_id}: {error}")
            continue
        fold = int(run["fold"])
        condition = str(run["condition"])
        seed = int(run["seed"])
        fold_record = _mapping(folds.get(str(fold)), f"prepared fold {fold}")
        expected_roles = {
            field: list(fold_record[field])
            for field in ("train_aliases", "validation_aliases", "test_aliases")
        }
        config_dataset = _mapping(config.get("dataset"), "config dataset")
        config_model = _mapping(config.get("model"), "config model")
        expected_preprocessing = str(fold_record["preprocessing_file_sha256"])
        identity_checks = {
            "status": summary.get("status") == "success",
            "campaign": summary.get("campaign_id") == CAMPAIGN_ID,
            "stage": summary.get("stage") == run["stage"],
            "condition": summary.get("condition") == condition,
            "fold": summary.get("fold") == fold,
            "seed": summary.get("model_seed") == seed,
            "attempt": int(config.get("attempt", -1))
            == int(run.get("attempt", config.get("attempt", -1))),
            "parameter_count": summary.get("parameter_count")
            == EXPECTED_PARAMETER_COUNT
            == config_model.get("expected_parameter_count"),
            "finite_metrics": summary.get("finite_metrics") is True,
            "coverage_complete": summary.get("coverage_complete") is True,
            "update_count_verified": summary.get("update_count_verified")
            is True,
            "all_gradients_finite": summary.get("all_gradients_finite") is True,
            "deterministic_algorithms": summary.get("deterministic_algorithms")
            is True,
            "prepared_content": summary.get("prepared_content_sha256")
            == manifest.get("content_sha256")
            == scientific.get("prepared_content_sha256"),
            "prepared_manifest": summary.get("prepared_manifest_sha256")
            == config_dataset.get("prepared_manifest_sha256")
            == scientific.get("prepared_manifest_sha256")
            == prepared_manifest_sha,
            "prepared_reference": scientific.get("prepared_manifest")
            == config_dataset.get("prepared_manifest")
            == prepared_reference,
            "contract": scientific.get("frozen_contract_sha256")
            == CONTRACT_SHA256,
            "scientific_fold": scientific.get("fold") == fold,
            "scientific_condition": scientific.get("condition") == condition,
            "pairing_fold": pairing.get("fold") == fold,
            "pairing_seed": pairing.get("model_seed") == seed,
            "pairing_condition": pairing.get("condition") == condition,
            "preprocessing": summary.get("preprocessing_sha256")
            == pairing.get("preprocessing_sha256")
            == scientific.get("preprocessing_sha256")
            == expected_preprocessing,
            "initial_state": summary.get("initial_state_sha256")
            == pairing.get("initial_state_sha256"),
            "training_masks": summary.get("training_mask_schedule_sha256")
            == pairing.get("training_mask_schedule_sha256"),
            "validation_selection_masks": summary.get(
                "validation_selection_mask_schedule_sha256"
            )
            == pairing.get("validation_selection_mask_schedule_sha256"),
            "validation_evaluation_masks": summary.get(
                "validation_mask_identity_sha256"
            )
            == pairing.get("validation_evaluation_mask_identity_sha256"),
            "test_evaluation_masks": summary.get("test_mask_identity_sha256")
            == pairing.get("test_evaluation_mask_identity_sha256"),
            "split_disjoint": pairing.get("split_disjoint") is True,
            "preprocessing_fit_scope": pairing.get(
                "preprocessing_fit_aliases"
            )
            == expected_roles["train_aliases"],
            "masked_neighbor_inputs_only": pairing.get(
                "masked_neighbor_inputs_only"
            )
            is True,
            "model_inputs": pairing.get("model_inputs")
            == ["masked_standardized_log1p_counts", "binary_mask"],
            "prohibited_inputs_absent": pairing.get("prohibited_inputs_absent")
            == [
                "cell_type",
                "cluster",
                "niche",
                "donor",
                "core",
                "tissue_stage",
                "coordinates",
                "target_derived_library_size",
            ],
        }
        for field, expected_value in expected_roles.items():
            identity_checks[f"role_{field}"] = pairing.get(field) == expected_value
        if require_pilot_resources:
            identity_checks["resource_limits_passed"] = (
                summary.get("resource_limits_passed") is True
            )
        if str(run["stage"]) in {"primary", "null"}:
            identity_checks.update(
                {
                    "conclusion_eligible": summary.get("conclusion_eligible")
                    is True,
                    "generalization_estimate": summary.get(
                        "generalization_estimate"
                    )
                    is True,
                    "completed_80_epochs": summary.get(
                        "completed_global_epochs"
                    )
                    == 80,
                    "completed_560_updates": summary.get("optimizer_steps")
                    == 560,
                    "config_80_epochs": _mapping(
                        config.get("trainer"), "config trainer"
                    ).get("max_epochs")
                    == 80,
                    "config_560_updates": _mapping(
                        config.get("trainer"), "config trainer"
                    ).get("expected_total_optimizer_updates")
                    == 560,
                }
            )
        recovery_audit = run.get("required_recovery_audit")
        if recovery_audit is not None:
            expected_recovery = _mapping(
                recovery_audit, f"{run_id} required recovery audit"
            )
            recovery_sources = {
                "summary": summary,
                "resources": resources,
                "pairing": pairing,
                "scientific_inputs": scientific,
            }
            for field, expected_value in expected_recovery.items():
                identity_checks[f"recovery_{field}_consistency"] = all(
                    field in source and source.get(field) == expected_value
                    for source in recovery_sources.values()
                )
            identity_checks["recovery_resource_device"] = (
                resources.get("device")
                == expected_recovery.get("execution_device")
            )
        for label, passed in identity_checks.items():
            if not passed:
                errors.append(f"{run_id}: scientific audit failed {label}")
        for field in PAIR_SUMMARY_FIELDS:
            value = summary.get(field)
            if field.endswith("sha256"):
                try:
                    _hex_digest(value, f"{run_id} summary {field}")
                except AdjacencyAnalysisError as error:
                    errors.append(str(error))
        for field in (
            "initial_state_sha256",
            "training_mask_schedule_sha256",
            "validation_selection_mask_schedule_sha256",
            "validation_evaluation_mask_identity_sha256",
            "test_evaluation_mask_identity_sha256",
            "preprocessing_sha256",
        ):
            try:
                _hex_digest(pairing.get(field), f"{run_id} diagnostic {field}")
            except AdjacencyAnalysisError as error:
                errors.append(str(error))
        evidence[run_id] = {
            "summary": dict(summary),
            "pairing": pairing,
            "scientific": scientific,
            "resources": resources,
            "shared_config": _shared_config_payload(config),
        }

    grouped: dict[tuple[int, int], dict[str, Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((int(run["fold"]), int(run["seed"])), {})[
            str(run["condition"])
        ] = run
    for pair_key, by_condition in sorted(grouped.items()):
        if set(by_condition) != set(paired_conditions):
            errors.append(
                f"fold/seed {pair_key}: paired condition set is incomplete"
            )
            continue
        reference_condition = paired_conditions[0]
        reference_run = by_condition[reference_condition]
        reference = evidence.get(str(reference_run["run_id"]))
        if reference is None:
            continue
        for condition in paired_conditions[1:]:
            candidate_run = by_condition[condition]
            candidate = evidence.get(str(candidate_run["run_id"]))
            if candidate is None:
                continue
            for field in PAIR_SUMMARY_FIELDS:
                if reference["summary"].get(field) != candidate["summary"].get(
                    field
                ):
                    errors.append(
                        f"fold/seed {pair_key}: paired summary {field} differs"
                    )
            for field in PAIR_DIAGNOSTIC_FIELDS:
                if reference["pairing"].get(field) != candidate["pairing"].get(
                    field
                ):
                    errors.append(
                        f"fold/seed {pair_key}: paired diagnostic {field} differs"
                    )
            if reference["shared_config"] != candidate["shared_config"]:
                errors.append(
                    f"fold/seed {pair_key}: paired scientific configs differ"
                )
    return sorted(set(errors)), evidence


def _expected_main_grid(
    runs: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> set[tuple[str, str, int]]:
    folds = _mapping(manifest.get("folds"), "prepared folds")
    return {
        (str(run["run_id"]), str(alias), replicate)
        for run in runs
        for alias in _mapping(
            folds.get(str(int(run["fold"]))), f"prepared fold {run['fold']}"
        )["test_aliases"]
        for replicate in range(3)
    }


def _validate_primary_rows(
    frame: pd.DataFrame,
    runs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    required = {
        "run_id",
        "config_sha256",
        "job_id",
        "attempt",
        "core_alias",
        "split",
        "mask_replicate",
        "mask_seed",
        "mask_checksum",
        "log1p_huber",
        "log1p_mae",
        "log1p_rmse",
        "standardized_huber",
        "condition",
        "fold",
        "model_seed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise AdjacencyAnalysisError(
            "per-core metric table lacks: " + ", ".join(sorted(missing))
        )
    if set(frame["split"]) != {"test"}:
        raise AdjacencyAnalysisError("primary table contains non-test rows")
    if not np.isfinite(
        frame[["log1p_huber", "log1p_mae", "log1p_rmse", "standardized_huber"]]
        .to_numpy(dtype=float)
    ).all():
        raise AdjacencyAnalysisError("primary metrics contain nonfinite values")
    key = ["run_id", "core_alias", "mask_replicate"]
    if frame.duplicated(key).any():
        raise AdjacencyAnalysisError("duplicate run/core/mask rows")
    expected = _expected_main_grid(runs, manifest)
    observed = {
        (str(row.run_id), str(row.core_alias), int(row.mask_replicate))
        for row in frame.itertuples(index=False)
    }
    if observed != expected:
        raise AdjacencyAnalysisError(
            "per-core table differs from exact fold test/mask grid; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    run_by_id = {str(run["run_id"]): run for run in runs}
    cores = _mapping(manifest.get("cores"), "prepared cores")
    for row in frame.itertuples(index=False):
        run = run_by_id[str(row.run_id)]
        replicate = int(row.mask_replicate)
        core = _mapping(cores.get(str(row.core_alias)), f"core {row.core_alias}")
        expected_seed = int(core["mask_seeds"][replicate])
        expected_checksum = str(core["mask_realization_checksums"][replicate])
        if (
            int(row.fold) != int(run["fold"])
            or int(row.model_seed) != int(run["seed"])
            or str(row.condition) != str(run["condition"])
            or str(row.config_sha256) != str(run["config_sha256"])
            or str(row.job_id) != str(run.get("job_id", ""))
            or int(row.attempt) != int(run.get("attempt", -1))
            or int(row.mask_seed) != expected_seed
            or str(row.mask_checksum) != expected_checksum
        ):
            raise AdjacencyAnalysisError(
                f"per-core provenance or mask metadata changed for "
                f"{row.run_id}/{row.core_alias}/{replicate}"
            )
        _hex_digest(str(row.mask_checksum), "per-core mask checksum")
    if set(frame["core_alias"]) != set(ALIASES):
        raise AdjacencyAnalysisError("primary table does not cover all ten cores")


def _validate_stratum_rows(
    frame: pd.DataFrame,
    *,
    main: pd.DataFrame,
    runs: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    bin_column: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    required = {
        "run_id",
        "config_sha256",
        "job_id",
        "attempt",
        "core_alias",
        "split",
        "mask_replicate",
        "mask_seed",
        "mask_checksum",
        "fold",
        "model_seed",
        "condition",
        bin_column,
        "n_masked",
        "log1p_huber",
        "log1p_mae",
        "log1p_mse",
        "log1p_rmse",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise AdjacencyAnalysisError(
            f"{bin_column} table lacks: {', '.join(sorted(missing))}"
        )
    if set(frame["split"]) != {"test"}:
        raise AdjacencyAnalysisError(f"{bin_column} table contains non-test rows")
    base_grid = _expected_main_grid(runs, manifest)
    expected = {
        (*base, label)
        for base in base_grid
        for label in labels
    }
    keys = ["run_id", "core_alias", "mask_replicate", bin_column]
    if frame.duplicated(keys).any():
        raise AdjacencyAnalysisError(f"duplicate {bin_column} grid rows")
    observed = {
        (
            str(row.run_id),
            str(row.core_alias),
            int(row.mask_replicate),
            str(getattr(row, bin_column)),
        )
        for row in frame.itertuples(index=False)
    }
    if observed != expected:
        raise AdjacencyAnalysisError(
            f"{bin_column} table differs from exact run/core/mask/bin grid; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    metadata = main[
        [
            "run_id",
            "core_alias",
            "mask_replicate",
            "mask_seed",
            "mask_checksum",
            "config_sha256",
            "job_id",
            "attempt",
            "fold",
            "model_seed",
            "condition",
            "n_cells",
            "n_masked",
        ]
    ].rename(
        columns={
            "mask_seed": "expected_mask_seed",
            "mask_checksum": "expected_mask_checksum",
            "config_sha256": "expected_config_sha256",
            "job_id": "expected_job_id",
            "attempt": "expected_attempt",
            "fold": "expected_fold",
            "model_seed": "expected_model_seed",
            "condition": "expected_condition",
            "n_cells": "overall_n_cells",
            "n_masked": "overall_n_masked",
        }
    )
    metadata_key = ["run_id", "core_alias", "mask_replicate"]
    if metadata.duplicated(metadata_key).any():
        raise AdjacencyAnalysisError(
            "main metric metadata repeats a run/core/mask key"
        )
    joined = frame.merge(
        metadata,
        on=metadata_key,
        how="left",
        validate="many_to_one",
    )
    if joined["expected_mask_seed"].isna().any():
        raise AdjacencyAnalysisError(f"{bin_column} rows do not join to main rows")
    exact = (
        (pd.to_numeric(joined["mask_seed"]) == joined["expected_mask_seed"])
        & (joined["mask_checksum"].astype(str) == joined["expected_mask_checksum"])
        & (joined["config_sha256"].astype(str) == joined["expected_config_sha256"])
        & (joined["job_id"].astype(str) == joined["expected_job_id"])
        & (pd.to_numeric(joined["attempt"]) == joined["expected_attempt"])
        & (pd.to_numeric(joined["fold"]) == joined["expected_fold"])
        & (pd.to_numeric(joined["model_seed"]) == joined["expected_model_seed"])
        & (joined["condition"].astype(str) == joined["expected_condition"])
    )
    if not bool(exact.all()):
        raise AdjacencyAnalysisError(
            f"{bin_column} rows have conflicting run or mask provenance"
        )
    if (pd.to_numeric(joined["n_masked"], errors="coerce") < 0).any():
        raise AdjacencyAnalysisError(f"{bin_column} contains negative support")
    supported = pd.to_numeric(joined["n_masked"], errors="coerce") > 0
    metric_columns = ["log1p_huber", "log1p_mae", "log1p_mse", "log1p_rmse"]
    if not np.isfinite(
        joined.loc[supported, metric_columns].to_numpy(dtype=float)
    ).all():
        raise AdjacencyAnalysisError(
            f"{bin_column} has nonfinite metrics with positive support"
        )
    grouped = joined.groupby(
        ["run_id", "core_alias", "mask_replicate"], observed=True
    )["n_masked"].sum()
    overall = metadata.set_index(
        ["run_id", "core_alias", "mask_replicate"]
    )["overall_n_masked"]
    grouped = grouped.sort_index()
    overall = overall.sort_index()
    if not grouped.index.equals(overall.index) or not np.array_equal(
        grouped.to_numpy(dtype=np.int64), overall.to_numpy(dtype=np.int64)
    ):
        raise AdjacencyAnalysisError(
            f"{bin_column} support does not partition the overall masked targets"
        )
    return joined.sort_values(keys).reset_index(drop=True)


def _core_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in (
            "log1p_huber",
            "log1p_mae",
            "log1p_rmse",
            "log1p_mse",
            "standardized_huber",
            "gene_pearson",
            "gene_spearman",
            "cell_pearson",
            "cell_spearman",
        )
        if column in frame.columns
    ]
    return (
        frame.groupby(
            ["core_alias", "model_seed", "condition"], as_index=False,
            observed=True,
        )[numeric]
        .mean()
        .sort_values(["core_alias", "model_seed", "condition"])
    )


def _paired_wide(
    aggregate: pd.DataFrame,
    *,
    left: str,
    right: str,
    metric: str,
) -> pd.DataFrame:
    filtered = aggregate[aggregate["condition"].isin((left, right))]
    wide = filtered.pivot(
        index=["core_alias", "model_seed"],
        columns="condition",
        values=metric,
    ).reset_index()
    if left not in wide or right not in wide or wide[[left, right]].isna().any().any():
        raise AdjacencyAnalysisError(f"unpaired {metric} rows for {left}/{right}")
    wide["difference"] = wide[left] - wide[right]
    wide["relative_improvement"] = (wide[right] - wide[left]) / wide[right]
    return wide


def _descriptive_metric_summary(
    aggregate: pd.DataFrame,
    *,
    left: str,
    right: str,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    """Core-equal descriptive aggregation for secondary metrics where defined."""

    filtered = aggregate[aggregate["condition"].isin((left, right))]
    wide = filtered.pivot(
        index=["core_alias", "model_seed"],
        columns="condition",
        values=metric,
    ).reset_index()
    if left not in wide or right not in wide:
        return {
            "direction": direction,
            "defined_core_count": 0,
            "defined_core_seed_pair_count": 0,
        }
    numeric = wide[[left, right]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    wide = wide.loc[valid].copy()
    if wide.empty:
        return {
            "direction": direction,
            "defined_core_count": 0,
            "defined_core_seed_pair_count": 0,
        }
    wide["difference"] = wide[left] - wide[right]
    core = wide.groupby("core_alias", as_index=False)[
        [left, right, "difference"]
    ].mean()
    favors = core["difference"] < 0 if direction == "lower" else core["difference"] > 0
    return {
        "direction": direction,
        "difference_definition": f"{left} minus {right}",
        "mean_left": float(core[left].mean()),
        "mean_right": float(core[right].mean()),
        "mean_difference": float(core["difference"].mean()),
        "defined_core_count": int(core["core_alias"].nunique()),
        "defined_core_seed_pair_count": int(len(wide)),
        "favoring_core_count": int(favors.sum()),
        "core_effects": core.to_dict(orient="records"),
    }


def _secondary_metric_summaries(
    aggregate: pd.DataFrame, *, left: str, right: str
) -> dict[str, Any]:
    return {
        metric: _descriptive_metric_summary(
            aggregate,
            left=left,
            right=right,
            metric=metric,
            direction=direction,
        )
        for metric, direction in SECONDARY_METRICS.items()
        if metric in aggregate.columns
    }


def _bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (10,) or not np.isfinite(vector).all():
        raise AdjacencyAnalysisError("core bootstrap requires ten finite values")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(vector), size=(resamples, len(vector)))
    means = vector[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _effect_summary(
    paired: pd.DataFrame,
    *,
    left: str,
    right: str,
) -> dict[str, Any]:
    core = paired.groupby("core_alias", as_index=False).agg(
        difference=("difference", "mean"),
        left_error=(left, "mean"),
        right_error=(right, "mean"),
    )
    core["relative_improvement"] = (
        core["right_error"] - core["left_error"]
    ) / core["right_error"]
    core = core.set_index("core_alias").loc[list(ALIASES)].reset_index()
    diff_low, diff_high = _bootstrap_mean(core["difference"].to_numpy())
    rel_low, rel_high = _bootstrap_mean(
        core["relative_improvement"].to_numpy(), seed=BOOTSTRAP_SEED + 1
    )
    seed = paired.groupby("model_seed", as_index=False)["difference"].mean()
    return {
        "left": left,
        "right": right,
        "mean_left_error": float(core["left_error"].mean()),
        "mean_right_error": float(core["right_error"].mean()),
        "mean_difference": float(core["difference"].mean()),
        "difference_ci95": [diff_low, diff_high],
        "mean_relative_improvement": float(
            core["relative_improvement"].mean()
        ),
        "relative_improvement_ci95": [rel_low, rel_high],
        "favoring_core_count": int((core["difference"] < 0).sum()),
        "favoring_seed_count": int((seed["difference"] < 0).sum()),
        "core_effects": core.to_dict(orient="records"),
        "seed_effects": seed.to_dict(orient="records"),
    }


def _bin_effects(
    frame: pd.DataFrame,
    *,
    bin_column: str,
    left: str,
    right: str,
) -> pd.DataFrame:
    required = {
        "core_alias",
        "model_seed",
        "condition",
        "mask_replicate",
        bin_column,
        "log1p_huber",
        "log1p_mae",
        "n_masked",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise AdjacencyAnalysisError(
            f"stratum table lacks {sorted(missing)}"
        )
    supported = frame[frame["n_masked"].astype(float) > 0].copy()
    grouped = (
        supported.groupby(
            ["core_alias", "model_seed", "condition", bin_column],
            as_index=False,
            observed=True,
        )[["log1p_huber", "log1p_mae"]]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    labels = TARGET_MASK_BINS if bin_column == "target_mask_bin" else NEIGHBOR_OBSERVED_BINS
    for metric in ("log1p_huber", "log1p_mae"):
        for label in labels:
            subset = grouped[grouped[bin_column].astype(str) == label]
            if subset.empty:
                raise AdjacencyAnalysisError(
                    f"no supported rows for {bin_column}={label}"
                )
            wide = subset.pivot(
                index=["core_alias", "model_seed"],
                columns="condition",
                values=metric,
            ).reset_index()
            if left not in wide or right not in wide:
                raise AdjacencyAnalysisError(
                    f"unpaired {bin_column}={label} rows"
                )
            wide["difference"] = wide[left] - wide[right]
            core = wide.groupby("core_alias", as_index=False)[
                [left, right, "difference"]
            ].mean()
            low, high = _bootstrap_mean(
                core.set_index("core_alias").loc[list(ALIASES)][
                    "difference"
                ].to_numpy(),
                seed=BOOTSTRAP_SEED + len(rows) + 10,
            )
            rows.append(
                {
                    bin_column: str(label),
                    "metric": metric,
                    f"{left}_mean": float(core[left].mean()),
                    f"{right}_mean": float(core[right].mean()),
                    "mean_difference": float(core["difference"].mean()),
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                    "favoring_cores": int((core["difference"] < 0).sum()),
                    "defined_core_seed_pairs": int(len(wide)),
                }
            )
    return pd.DataFrame(rows)


def _neighbor_mask_sanity(effects: pd.DataFrame) -> dict[str, Any]:
    subset = effects[effects["metric"] == "log1p_huber"].copy()
    subset = subset.set_index("neighbor_observed_bin").loc[
        list(NEIGHBOR_OBSERVED_BINS)
    ]
    differences = subset["mean_difference"].to_numpy(dtype=np.float64)
    if not np.isfinite(differences).all():
        raise AdjacencyAnalysisError("neighbor-mask sanity has nonfinite effects")
    trend = (
        None
        if np.ptp(differences) == 0
        else float(np.corrcoef(np.arange(len(differences)), differences)[0, 1])
    )
    return {
        "expected_pattern": (
            "graph-minus-control error becomes more negative as more true-neighbor "
            "expression is observed"
        ),
        "low_observation_difference": float(differences[0]),
        "high_observation_difference": float(differences[-1]),
        "high_minus_low_difference": float(differences[-1] - differences[0]),
        "ordered_bin_trend_correlation": trend,
        "expected_direction_supported": bool(differences[-1] < differences[0]),
        "strictly_monotone_supported": bool(np.all(np.diff(differences) < 0)),
        "interpretation": "descriptive model-use sanity check; not a decision gate",
    }


def _decision(
    huber: Mapping[str, Any],
    mae: Mapping[str, Any],
    mask_effects: pd.DataFrame,
) -> tuple[str, list[str]]:
    moderate = mask_effects[
        (mask_effects["metric"] == "log1p_huber")
        & mask_effects["target_mask_bin"].isin(("25_to_50", "50_to_75"))
    ]
    reasons = [
        f"mean relative Huber improvement={100*float(huber['mean_relative_improvement']):.3f}%",
        f"Huber difference CI95={huber['difference_ci95']}",
        f"favoring cores={huber['favoring_core_count']}/10",
        f"favoring seeds={huber['favoring_seed_count']}/5",
        f"mean MAE difference={float(mae['mean_difference']):.8g}",
        "moderate-bin Huber differences="
        + json.dumps(
            dict(zip(moderate["target_mask_bin"], moderate["mean_difference"])),
            sort_keys=True,
        ),
    ]
    supported = all(
        (
            float(huber["mean_relative_improvement"])
            >= MEANINGFUL_RELATIVE_IMPROVEMENT,
            float(huber["difference_ci95"][1]) < 0,
            int(huber["favoring_core_count"]) >= 8,
            int(huber["favoring_seed_count"]) >= 4,
            float(mae["mean_difference"]) < 0,
            len(moderate) == 2,
            bool((moderate["mean_difference"] < 0).all()),
        )
    )
    if supported:
        return "GRAPH CONTEXT SUPPORTED", reasons
    if float(huber["relative_improvement_ci95"][1]) < (
        MEANINGFUL_RELATIVE_IMPROVEMENT
    ):
        return "GRAPH CONTEXT NOT SUPPORTED", reasons
    return "INCONCLUSIVE", reasons


def _topology_decision(
    huber: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Apply only the prespecified real-versus-null topology criteria."""

    reasons = [
        "topology gate uses log1p Huber only; MAE and strata are descriptive",
        f"mean relative Huber improvement={100*float(huber['mean_relative_improvement']):.3f}%",
        f"relative improvement CI95={huber['relative_improvement_ci95']}",
        f"Huber difference CI95={huber['difference_ci95']}",
        f"favoring cores={huber['favoring_core_count']}/10",
        f"favoring seeds={huber['favoring_seed_count']}/5",
    ]
    supported = all(
        (
            float(huber["mean_relative_improvement"])
            >= MEANINGFUL_RELATIVE_IMPROVEMENT,
            float(huber["difference_ci95"][1]) < 0,
            int(huber["favoring_core_count"]) >= 8,
            int(huber["favoring_seed_count"]) >= 4,
        )
    )
    if supported:
        return "REAL TOPOLOGY SUPPORTED", reasons
    if float(huber["relative_improvement_ci95"][1]) < (
        MEANINGFUL_RELATIVE_IMPROVEMENT
    ):
        return "REAL TOPOLOGY NOT SUPPORTED", reasons
    return "REAL TOPOLOGY INCONCLUSIVE", reasons


def _save_effect_plot(
    paired: pd.DataFrame, path: Path, *, left: str, right: str
) -> None:
    core = paired.groupby("core_alias")["difference"].agg(["mean", "std"])
    core = core.loc[list(ALIASES)]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(core))
    axis.errorbar(
        positions,
        core["mean"],
        yerr=core["std"],
        fmt="o",
        capsize=3,
        color="#225ea8",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(positions, core.index, rotation=35, ha="right")
    axis.set_ylabel(f"{left} minus {right} log1p Huber")
    axis.set_title("Held-out core effects (mean and SD across five seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_bin_plot(
    effects: pd.DataFrame,
    path: Path,
    *,
    bin_column: str,
    title: str,
) -> None:
    subset = effects[effects["metric"] == "log1p_huber"].copy()
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    positions = np.arange(len(subset))
    means = subset["mean_difference"].to_numpy(dtype=float)
    low = subset["difference_ci95_low"].to_numpy(dtype=float)
    high = subset["difference_ci95_high"].to_numpy(dtype=float)
    axis.errorbar(
        positions,
        means,
        yerr=np.vstack((means - low, high - means)),
        fmt="o-",
        capsize=3,
        color="#238b45",
    )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xticks(positions, subset[bin_column], rotation=25, ha="right")
    axis.set_ylabel("graph minus isolated log1p Huber")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _atomic_publish_report(
    target: Path, builder: Any
) -> None:
    """Build one complete report beside its destination and publish once."""

    if target.exists() or target.is_symlink():
        raise AdjacencyAnalysisError(
            f"refusing to overwrite an existing report directory: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    try:
        builder(temporary)
        if target.exists() or target.is_symlink():
            raise AdjacencyAnalysisError(
                f"report destination appeared during publication: {target}"
            )
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _assert_receipt_compatible(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if not path.is_file() or _read_json(path) != dict(payload):
            raise AdjacencyAnalysisError(
                f"refusing to overwrite a different receipt: {path}"
            )


def _primary_job_identities(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            {
                "condition": str(run["condition"]),
                "fold": int(run["fold"]),
                "seed": int(run["seed"]),
                "run_id": str(run["run_id"]),
                "job_id": str(run.get("job_id", "")),
                "attempt": int(run.get("attempt", -1)),
                "config_sha256": str(run["config_sha256"]),
                "materialized_config_sha256": str(
                    _mapping(
                        run.get("materialized_job"), "run materialized job"
                    ).get("config_sha256", "")
                ),
            }
            for run in runs
        ],
        key=lambda row: (row["fold"], row["seed"], row["condition"]),
    )


def _null_trigger_receipt(
    *,
    materialization: Mapping[str, Any],
    recovery: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    triggered: bool,
    difference: float | None,
    scientific_audit_passed: bool,
    audit_errors: Sequence[str],
) -> dict[str, Any]:
    favorable = (
        scientific_audit_passed
        and difference is not None
        and math.isfinite(float(difference))
        and float(difference) < 0
    )
    if triggered != favorable:
        raise AdjacencyAnalysisError(
            "null trigger must equal the audited favorable primary comparison"
        )
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": NULL_TRIGGER_KIND,
            "campaign_id": CAMPAIGN_ID,
            "materialization_checksum": materialization["checksum"],
            "recovery_plan_checksum": _mapping(
                recovery.get("plan"), "recovery plan"
            )["checksum"],
            "recovery_enqueue_checksum": _mapping(
                recovery.get("enqueue"), "recovery enqueue"
            )["checksum"],
            "frozen_contract_sha256": CONTRACT_SHA256,
            "triggered": bool(triggered),
            "graph_huber_lower_than_isolated": bool(favorable),
            "graph_minus_isolated_huber": difference,
            "scientific_audit_passed": bool(scientific_audit_passed),
            "scientific_audit_errors": list(audit_errors),
            "primary_jobs": _primary_job_identities(runs),
        }
    )


def _verify_positive_null_trigger(
    path: Path,
    *,
    materialization: Mapping[str, Any],
    recovery: Mapping[str, Any],
    primary_runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    trigger = _read_json(path)
    checksum = _hex_digest(trigger.get("checksum"), "null trigger checksum")
    unsigned = dict(trigger)
    unsigned.pop("checksum", None)
    expected_jobs = _primary_job_identities(primary_runs)
    difference = trigger.get("graph_minus_isolated_huber")
    if (
        checksum != canonical_sha256(unsigned)
        or trigger.get("schema_version") != 1
        or trigger.get("receipt_kind") != NULL_TRIGGER_KIND
        or trigger.get("campaign_id") != CAMPAIGN_ID
        or trigger.get("materialization_checksum")
        != materialization.get("checksum")
        or trigger.get("recovery_plan_checksum")
        != _mapping(recovery.get("plan"), "recovery plan").get("checksum")
        or trigger.get("recovery_enqueue_checksum")
        != _mapping(recovery.get("enqueue"), "recovery enqueue").get("checksum")
        or trigger.get("frozen_contract_sha256") != CONTRACT_SHA256
        or trigger.get("triggered") is not True
        or trigger.get("graph_huber_lower_than_isolated") is not True
        or trigger.get("scientific_audit_passed") is not True
        or trigger.get("scientific_audit_errors") != []
        or isinstance(difference, bool)
        or not isinstance(difference, (int, float))
        or not math.isfinite(float(difference))
        or float(difference) >= 0
        or trigger.get("primary_jobs") != expected_jobs
    ):
        raise AdjacencyAnalysisError(
            "topology analysis requires the exact positive audited null trigger"
        )
    for job in expected_jobs:
        slot = (job["fold"], job["seed"], job["condition"])
        authorization = _mapping(
            _mapping(recovery.get("primary"), "recovery primary").get(slot),
            f"recovery primary {slot}",
        )
        if (
            job["attempt"] != 2
            or job["job_id"] != authorization.get("retry_job_id")
            or job["config_sha256"]
            != authorization.get("resolved_config_sha256")
            or job["materialized_config_sha256"]
            != _mapping(
                authorization.get("materialized_job"), "materialized job"
            ).get("config_sha256")
        ):
            raise AdjacencyAnalysisError(
                f"null trigger primary job differs from materialization: {slot}"
            )
    return trigger


def _null_enqueue_authorization(
    path: Path,
    *,
    materialization: Mapping[str, Any],
    recovery: Mapping[str, Any],
    trigger: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the conditional null queue receipt and its exact 25 job IDs."""

    receipt = _read_json(path)
    checksum = _hex_digest(receipt.get("checksum"), "null enqueue checksum")
    unsigned = dict(receipt)
    unsigned.pop("checksum", None)
    expected_keys = {
        "schema_version",
        "receipt_kind",
        "campaign_id",
        "stage",
        "materialization_checksum",
        "pilot_gate_checksum",
        "null_trigger_checksum",
        "recovery_plan_checksum",
        "recovery_enqueue_checksum",
        "execution_device",
        "maximum_attempts",
        "complete",
        "jobs",
        "checksum",
    }
    jobs = receipt.get("jobs")
    if (
        set(receipt) != expected_keys
        or canonical_sha256(unsigned) != checksum
        or receipt.get("schema_version") != 1
        or receipt.get("receipt_kind") != NULL_ENQUEUE_KIND
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("stage") != "null"
        or receipt.get("materialization_checksum")
        != materialization.get("checksum")
        or receipt.get("pilot_gate_checksum") is not None
        or receipt.get("null_trigger_checksum") != trigger.get("checksum")
        or receipt.get("recovery_plan_checksum")
        != _mapping(recovery.get("plan"), "recovery plan").get("checksum")
        or receipt.get("recovery_enqueue_checksum")
        != _mapping(recovery.get("enqueue"), "recovery enqueue").get(
            "checksum"
        )
        or receipt.get("execution_device") != "cpu"
        or receipt.get("maximum_attempts") != 1
        or receipt.get("complete") is not True
        or not isinstance(jobs, list)
        or len(jobs) != 25
    ):
        raise AdjacencyAnalysisError(
            "topology analysis requires the exact CPU null enqueue receipt"
        )
    planned = _planned_jobs(materialization)
    recovery_null = _mapping(recovery.get("null"), "recovery null")
    authorized: dict[tuple[int, int, str], dict[str, Any]] = {}
    job_ids: set[str] = set()
    job_keys = {
        "stage",
        "fold",
        "seed",
        "condition",
        "config_sha256",
        "scientific_id",
        "job_id",
        "requested_gpu",
        "maximum_attempts",
    }
    for raw in jobs:
        job = dict(_mapping(raw, "null enqueue job"))
        slot = (
            int(job.get("fold", -1)),
            int(job.get("seed", -1)),
            str(job.get("condition")),
        )
        materialized_job = planned.get(("null", *slot))
        recovery_slot = recovery_null.get(slot)
        job_id = str(job.get("job_id", ""))
        if (
            set(job) != job_keys
            or slot in authorized
            or not job_id
            or job_id in job_ids
            or materialized_job is None
            or recovery_slot is None
            or job.get("stage") != "null"
            or slot[2] != NULL_CONDITION
            or job.get("config_sha256")
            != materialized_job.get("config_sha256")
            or job.get("scientific_id")
            != materialized_job.get("scientific_id")
            or job.get("requested_gpu")
            != _mapping(
                _mapping(recovery_slot, "recovery null slot").get("plan"),
                "recovery null plan",
            ).get("worker_slot")
            or job.get("maximum_attempts") != 1
        ):
            raise AdjacencyAnalysisError(
                f"null enqueue identity changed for slot {slot}"
            )
        job_ids.add(job_id)
        authorized[slot] = job
    expected = {
        (fold, seed, NULL_CONDITION) for fold in FOLDS for seed in SEEDS
    }
    if set(authorized) != expected or len(job_ids) != 25:
        raise AdjacencyAnalysisError("null enqueue job grid is incomplete")
    return {"receipt": receipt, "jobs": authorized}


def _verify_null_run_identities(
    records: Sequence[Mapping[str, Any]],
    *,
    null_enqueue: Mapping[str, Any],
) -> None:
    expected = _mapping(null_enqueue.get("jobs"), "authorized null jobs")
    observed = [row for row in records if row.get("stage") == "null"]
    if len(observed) != 25:
        raise AdjacencyAnalysisError(
            "topology evidence does not contain exactly 25 null successes"
        )
    seen: set[tuple[int, int, str]] = set()
    for run in observed:
        slot = (
            int(run["fold"]),
            int(run["seed"]),
            str(run["condition"]),
        )
        authorized = expected.get(slot)
        if (
            slot in seen
            or authorized is None
            or run.get("selection_role") != "recovery_null_attempt_1"
            or int(run.get("attempt", -1)) != 1
            or run.get("job_id") != authorized.get("job_id")
            or run.get("config_sha256") != authorized.get("config_sha256")
        ):
            raise AdjacencyAnalysisError(
                f"null success is not the authorized completed queue job: {slot}"
            )
        seen.add(slot)
    if seen != set(expected):
        raise AdjacencyAnalysisError("authorized null queue jobs are incomplete")


def _publish_audit_failure(
    *,
    mode: str,
    output_root: Path,
    materialization: Mapping[str, Any],
    recovery: Mapping[str, Any] | None,
    runs: Sequence[Mapping[str, Any]],
    audit_errors: Sequence[str],
) -> dict[str, Any]:
    decision = "INCONCLUSIVE" if mode == "primary" else "REAL TOPOLOGY INCONCLUSIVE"
    result = _signed(
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "materialization_checksum": materialization["checksum"],
            "recovery_plan_checksum": (
                None
                if recovery is None
                else _mapping(recovery.get("plan"), "recovery plan")["checksum"]
            ),
            "recovery": (
                None
                if recovery is None
                else {
                    "execution_device": "cpu",
                    "required_primary_attempt": 2,
                    "excluded_attempt_1_successes": int(
                        recovery["attempt_1_completed"]
                    ),
                    "immutable_attempt_1_failures": int(
                        recovery["attempt_1_failed"]
                    ),
                }
            ),
            "scientific_audit_passed": False,
            "scientific_audit_errors": list(audit_errors),
            "decision": decision,
            "maximum_supported_claim": None,
            "claim_withheld_reason": "scientific preflight failed",
        }
    )
    target = output_root / mode
    receipt: dict[str, Any] | None = None
    receipt_path = output_root / "null_trigger_receipt.json"
    if mode == "primary":
        if recovery is None:
            raise AdjacencyAnalysisError(
                "primary audit failure still requires recovery receipt binding"
            )
        receipt = _null_trigger_receipt(
            materialization=materialization,
            recovery=recovery,
            runs=runs,
            triggered=False,
            difference=None,
            scientific_audit_passed=False,
            audit_errors=audit_errors,
        )
        _assert_receipt_compatible(receipt_path, receipt)

    def build(stage: Path) -> None:
        _write_json_file(stage / "aggregate_summary.json", result)
        (stage / "README.md").write_text(
            "\n".join(
                (
                    f"# {mode.title()} adjacency analysis",
                    "",
                    f"- Evidence classification: **{decision}**",
                    "- Scientific audit: **failed**",
                    "",
                    "No quantitative or positive scientific claim is reported. ",
                    "See `aggregate_summary.json` for the exact audit failures.",
                    "",
                )
            ),
            encoding="utf-8",
        )

    _atomic_publish_report(target, build)
    if receipt is not None:
        _write_json_idempotent(receipt_path, receipt)
    return result


def write_pilot_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    materialization: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    expected = {(0, 0, condition) for condition in CONDITIONS}
    runs = _select_unique_runs(
        records,
        stage="pilot",
        conditions=CONDITIONS,
        expected_keys=expected,
    )
    manifest = _bound_prepared_manifest(materialization)
    audit_errors, evidence = _scientific_run_audit(
        runs,
        manifest=manifest,
        paired_conditions=CONDITIONS,
        require_pilot_resources=True,
    )
    jobs: list[dict[str, Any]] = []
    for run in runs:
        summary = _mapping(run["summary"], "pilot summary")
        job = {
            "condition": run["condition"],
            "run_id": run["run_id"],
            "config_sha256": run["config_sha256"],
            "finite_metrics": summary.get("finite_metrics") is True,
            "coverage_complete": summary.get("coverage_complete") is True,
            "update_count_verified": summary.get("update_count_verified")
            is True,
            "resource_limits_passed": summary.get("resource_limits_passed")
            is True,
        }
        job.update({field: summary.get(field) for field in PAIR_SUMMARY_FIELDS})
        pairing = evidence.get(str(run["run_id"]), {}).get("pairing", {})
        job["validation_evaluation_mask_identity_sha256"] = pairing.get(
            "validation_evaluation_mask_identity_sha256"
        )
        job["test_evaluation_mask_identity_sha256"] = pairing.get(
            "test_evaluation_mask_identity_sha256"
        )
        jobs.append(job)
    passed = not audit_errors
    payload = _signed(
        {
            "schema_version": 1,
            "receipt_kind": "adjacency_ablation_paired_pilot_gate_v1",
            "campaign_id": CAMPAIGN_ID,
            "materialization_checksum": materialization["checksum"],
            "frozen_contract_sha256": CONTRACT_SHA256,
            "passed": passed,
            "scientific_audit_passed": passed,
            "scientific_audit_errors": audit_errors,
            "fold": 0,
            "seed": 0,
            "jobs": sorted(jobs, key=lambda row: row["condition"]),
        }
    )
    _write_json_idempotent(output, payload)
    if not passed:
        raise AdjacencyAnalysisError(
            "paired pilot gate did not pass: " + "; ".join(audit_errors)
        )
    return payload


def analyze_primary(
    records: Sequence[Mapping[str, Any]],
    *,
    materialization: Mapping[str, Any],
    recovery: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    expected = {
        (fold, seed, condition)
        for fold in FOLDS
        for seed in SEEDS
        for condition in CONDITIONS
    }
    runs = _select_unique_runs(
        records,
        stage="primary",
        conditions=CONDITIONS,
        expected_keys=expected,
        selection_roles=("recovery_primary_attempt_2",),
    )
    historical = [
        row for row in records
        if row.get("stage") == "primary"
        and row.get("selection_role") == "historical_primary_attempt_1"
    ]
    unauthorized = [
        row for row in records
        if row.get("stage") == "primary"
        and str(row.get("selection_role", "")).startswith("unauthorized")
    ]
    manifest = _bound_prepared_manifest(materialization)
    audit_errors, _ = _scientific_run_audit(
        runs, manifest=manifest, paired_conditions=CONDITIONS
    )
    if len(historical) != int(recovery["attempt_1_completed"]):
        audit_errors.append(
            "historical primary success count differs from recovery inventory"
        )
    for row in historical:
        for error in row.get("audit_errors", []):
            audit_errors.append(f"historical {row['run_id']}: {error}")
    for row in unauthorized:
        audit_errors.append(
            f"unauthorized primary retry/success {row['run_id']}: "
            + "; ".join(row.get("audit_errors", []))
        )
    if audit_errors:
        return _publish_audit_failure(
            mode="primary", output_root=output_root,
            materialization=materialization, recovery=recovery, runs=runs,
            audit_errors=audit_errors,
        )
    try:
        main, mask_bins, neighbor_bins = _load_primary_tables(runs)
        _validate_primary_rows(main, runs, manifest)
        mask_bins = _validate_stratum_rows(
            mask_bins, main=main, runs=runs, manifest=manifest,
            bin_column="target_mask_bin", labels=TARGET_MASK_BINS,
        )
        neighbor_bins = _validate_stratum_rows(
            neighbor_bins, main=main, runs=runs, manifest=manifest,
            bin_column="neighbor_observed_bin", labels=NEIGHBOR_OBSERVED_BINS,
        )
        for frame in (main, mask_bins, neighbor_bins):
            bindings = {
                "materialization_checksum": materialization["checksum"],
                "recovery_plan_checksum": _mapping(
                    recovery.get("plan"), "recovery plan"
                )["checksum"],
                "recovery_enqueue_checksum": _mapping(
                    recovery.get("enqueue"), "recovery enqueue"
                )["checksum"],
            }
            for column, expected_value in bindings.items():
                if column in frame and not bool(
                    (frame[column].astype(str) == str(expected_value)).all()
                ):
                    raise AdjacencyAnalysisError(
                        f"raw rows contain conflicting {column} provenance"
                    )
                frame[column] = expected_value
        core_seed = _core_seed_metrics(main)
        huber_paired = _paired_wide(
            core_seed, left="spatial", right="isolated", metric="log1p_huber"
        )
        mae_paired = _paired_wide(
            core_seed, left="spatial", right="isolated", metric="log1p_mae"
        )
        huber = _effect_summary(huber_paired, left="spatial", right="isolated")
        mae = _effect_summary(mae_paired, left="spatial", right="isolated")
        secondary = _secondary_metric_summaries(
            core_seed, left="spatial", right="isolated"
        )
        mask_effects = _bin_effects(
            mask_bins, bin_column="target_mask_bin",
            left="spatial", right="isolated",
        )
        neighbor_effects = _bin_effects(
            neighbor_bins, bin_column="neighbor_observed_bin",
            left="spatial", right="isolated",
        )
        neighbor_sanity = _neighbor_mask_sanity(neighbor_effects)
        decision, reasons = _decision(huber, mae, mask_effects)
    except (AdjacencyAnalysisError, KeyError, TypeError, ValueError) as error:
        return _publish_audit_failure(
            mode="primary", output_root=output_root,
            materialization=materialization, recovery=recovery, runs=runs,
            audit_errors=[f"metric-table scientific audit failed: {error}"],
        )
    trigger = float(huber["mean_difference"]) < 0
    trigger_receipt = _null_trigger_receipt(
        materialization=materialization, recovery=recovery,
        runs=runs, triggered=trigger,
        difference=float(huber["mean_difference"]),
        scientific_audit_passed=True, audit_errors=[],
    )
    maximum_claim = (
        "Spatial neighboring-cell information provides predictive value for "
        "masked transcript reconstruction in Adjacent Normal gastric tissue."
        if decision == "GRAPH CONTEXT SUPPORTED" else None
    )
    result = _signed({
        "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "materialization_checksum": materialization["checksum"],
        "recovery_plan_checksum": _mapping(
            recovery.get("plan"), "recovery plan"
        )["checksum"],
        "recovery_enqueue_checksum": _mapping(
            recovery.get("enqueue"), "recovery enqueue"
        )["checksum"],
        "recovery": {
            "execution_device": "cpu",
            "selected_primary_attempt": 2,
            "selected_attempt_2_successes": len(runs),
            "excluded_attempt_1_successes": len(historical),
            "immutable_attempt_1_failures": int(recovery["attempt_1_failed"]),
            "mixed_attempt_estimate_prohibited": True,
            "limitation": (
                "execution changed after a host-wide CUDA failure and before "
                "primary outcomes were examined"
            ),
        },
        "exploratory": True, "scientific_audit_passed": True,
        "scientific_audit_errors": [], "complete_primary_runs": len(runs),
        "independent_core_count": 10, "model_seed_count": 5,
        "evaluation_mask_replicates": 3,
        "graph_vs_isolated": {"log1p_huber": huber, "log1p_mae": mae,
                              "secondary_metrics": secondary},
        "target_mask_bin_effects": mask_effects.to_dict(orient="records"),
        "neighbor_observation_sanity": neighbor_sanity,
        "decision": decision, "decision_rationale": reasons,
        "null_triggered": trigger,
        "strongest_alternative_explanation": (
            "generic smoothing or composition, plus segmentation spillover"
        ),
        "maximum_supported_claim": maximum_claim,
        "claim_withheld_reason": (
            None if maximum_claim is not None else "support criteria were not met"
        ),
    })
    receipt_path = output_root / "null_trigger_receipt.json"
    _assert_receipt_compatible(receipt_path, trigger_receipt)

    def build(stage: Path) -> None:
        tables, plots = stage / "tables", stage / "plots"
        tables.mkdir(); plots.mkdir()
        main.to_csv(tables / "audited_per_core_replicate.csv", index=False)
        mask_bins.to_csv(tables / "audited_per_target_mask_bin.csv", index=False)
        neighbor_bins.to_csv(
            tables / "audited_per_neighbor_observation_bin.csv", index=False
        )
        core_seed.to_csv(tables / "per_core_seed_metrics.csv", index=False)
        huber_paired.to_csv(tables / "paired_core_seed_huber.csv", index=False)
        mae_paired.to_csv(tables / "paired_core_seed_mae.csv", index=False)
        mask_effects.to_csv(tables / "target_mask_bin_effects.csv", index=False)
        neighbor_effects.to_csv(
            tables / "neighbor_observation_bin_effects.csv", index=False
        )
        _save_effect_plot(huber_paired, plots / "paired_core_huber.png",
                          left="spatial", right="isolated")
        _save_bin_plot(mask_effects, plots / "target_mask_bins.png",
                       bin_column="target_mask_bin",
                       title="Adjacency effect by target-cell masking percentage")
        _save_bin_plot(neighbor_effects, plots / "neighbor_observation_bins.png",
                       bin_column="neighbor_observed_bin",
                       title="Adjacency effect by observed true-neighbor expression")
        _write_json_file(stage / "aggregate_summary.json", result)
        _write_json_file(stage / "null_trigger_receipt.json", trigger_receipt)
        (stage / "README.md").write_text(
            f"# Grouped adjacency ablation result\n\n"
            f"- Evidence classification: **{decision}**\n"
            f"- Scientific audit: **passed**\n"
            f"- Selected CPU attempt-2 runs: **{len(runs)}/50**\n"
            f"- Excluded GPU attempt-1 successes: **{len(historical)}**\n"
            f"- Preserved immutable attempt-1 failures: "
            f"**{int(recovery['attempt_1_failed'])}**\n"
            f"- Mean graph-minus-isolated log1p Huber: "
            f"{huber['mean_difference']:.8g}\n",
            encoding="utf-8",
        )

    _atomic_publish_report(output_root / "primary", build)
    _write_json_idempotent(receipt_path, trigger_receipt)
    return result


def analyze_topology(
    records: Sequence[Mapping[str, Any]],
    *,
    materialization: Mapping[str, Any],
    recovery: Mapping[str, Any],
    output_root: Path,
    null_trigger_path: Path | None = None,
    null_enqueue_path: Path | None = None,
) -> dict[str, Any]:
    unauthorized = [
        row for row in records
        if row.get("stage") in {"primary", "null"}
        and str(row.get("selection_role", "")).startswith("unauthorized")
    ]
    if unauthorized:
        raise AdjacencyAnalysisError(
            "topology evidence contains unauthorized recovery runs: "
            + ", ".join(str(row.get("run_id")) for row in unauthorized)
        )
    expected_primary = {
        (fold, seed, condition)
        for fold in FOLDS for seed in SEEDS for condition in CONDITIONS
    }
    primary_runs = _select_unique_runs(
        records, stage="primary", conditions=CONDITIONS,
        expected_keys=expected_primary,
        selection_roles=("recovery_primary_attempt_2",),
    )
    trigger = _verify_positive_null_trigger(
        null_trigger_path or output_root / "null_trigger_receipt.json",
        materialization=materialization,
        recovery=recovery,
        primary_runs=primary_runs,
    )
    if null_enqueue_path is None:
        raise AdjacencyAnalysisError(
            "topology analysis requires the immutable null enqueue receipt"
        )
    null_enqueue = _null_enqueue_authorization(
        null_enqueue_path,
        materialization=materialization,
        recovery=recovery,
        trigger=trigger,
    )
    _verify_null_run_identities(records, null_enqueue=null_enqueue)
    expected_spatial = {(fold, seed, "spatial") for fold in FOLDS for seed in SEEDS}
    expected_null = {
        (fold, seed, "position_permuted_null")
        for fold in FOLDS
        for seed in SEEDS
    }
    spatial = _select_unique_runs(
        records,
        stage="primary",
        conditions=("spatial",),
        expected_keys=expected_spatial,
        selection_roles=("recovery_primary_attempt_2",),
    )
    null = _select_unique_runs(
        records,
        stage="null",
        conditions=("position_permuted_null",),
        expected_keys=expected_null,
        selection_roles=("recovery_null_attempt_1",),
    )
    runs = [*spatial, *null]
    manifest = _bound_prepared_manifest(materialization)
    primary_errors, _ = _scientific_run_audit(
        primary_runs, manifest=manifest, paired_conditions=CONDITIONS
    )
    if primary_errors:
        raise AdjacencyAnalysisError(
            "positive null trigger no longer matches audited primary runs: "
            + "; ".join(primary_errors)
        )
    audit_errors, _ = _scientific_run_audit(
        runs, manifest=manifest,
        paired_conditions=("spatial", NULL_CONDITION),
    )
    if audit_errors:
        return _publish_audit_failure(
            mode="topology", output_root=output_root,
            materialization=materialization, recovery=recovery, runs=runs,
            audit_errors=audit_errors,
        )
    try:
        main, mask_bins, neighbor_bins = _load_primary_tables(runs)
        _validate_primary_rows(main, runs, manifest)
        mask_bins = _validate_stratum_rows(
            mask_bins, main=main, runs=runs, manifest=manifest,
            bin_column="target_mask_bin", labels=TARGET_MASK_BINS,
        )
        neighbor_bins = _validate_stratum_rows(
            neighbor_bins, main=main, runs=runs, manifest=manifest,
            bin_column="neighbor_observed_bin", labels=NEIGHBOR_OBSERVED_BINS,
        )
        for frame in (main, mask_bins, neighbor_bins):
            for column, expected_value in (
                ("materialization_checksum", materialization["checksum"]),
                (
                    "recovery_plan_checksum",
                    _mapping(recovery.get("plan"), "recovery plan")["checksum"],
                ),
                (
                    "recovery_enqueue_checksum",
                    _mapping(recovery.get("enqueue"), "recovery enqueue")[
                        "checksum"
                    ],
                ),
                ("null_trigger_checksum", trigger["checksum"]),
                (
                    "null_enqueue_checksum",
                    _mapping(
                        null_enqueue.get("receipt"), "null enqueue receipt"
                    )["checksum"],
                ),
            ):
                if column in frame and not bool(
                    (frame[column].astype(str) == str(expected_value)).all()
                ):
                    raise AdjacencyAnalysisError(
                        f"raw rows contain conflicting {column} provenance"
                    )
            frame["materialization_checksum"] = materialization["checksum"]
            frame["recovery_plan_checksum"] = _mapping(
                recovery.get("plan"), "recovery plan"
            )["checksum"]
            frame["recovery_enqueue_checksum"] = _mapping(
                recovery.get("enqueue"), "recovery enqueue"
            )["checksum"]
            frame["null_trigger_checksum"] = trigger["checksum"]
            frame["null_enqueue_checksum"] = _mapping(
                null_enqueue.get("receipt"), "null enqueue receipt"
            )["checksum"]
        core_seed = _core_seed_metrics(main)
        huber_paired = _paired_wide(
            core_seed, left="spatial", right=NULL_CONDITION,
            metric="log1p_huber",
        )
        mae_paired = _paired_wide(
            core_seed, left="spatial", right=NULL_CONDITION,
            metric="log1p_mae",
        )
        huber = _effect_summary(huber_paired, left="spatial", right=NULL_CONDITION)
        mae = _effect_summary(mae_paired, left="spatial", right=NULL_CONDITION)
        secondary = _secondary_metric_summaries(
            core_seed, left="spatial", right=NULL_CONDITION
        )
        mask_effects = _bin_effects(
            mask_bins, bin_column="target_mask_bin",
            left="spatial", right=NULL_CONDITION,
        )
        neighbor_effects = _bin_effects(
            neighbor_bins, bin_column="neighbor_observed_bin",
            left="spatial", right=NULL_CONDITION,
        )
        neighbor_sanity = _neighbor_mask_sanity(neighbor_effects)
        decision, reasons = _topology_decision(huber)
    except (AdjacencyAnalysisError, KeyError, TypeError, ValueError) as error:
        return _publish_audit_failure(
            mode="topology", output_root=output_root,
            materialization=materialization, recovery=recovery, runs=runs,
            audit_errors=[f"metric-table scientific audit failed: {error}"],
        )
    maximum_claim = (
        "The actual local spatial assignment of neighboring cells contains "
        "predictive information."
        if decision == "REAL TOPOLOGY SUPPORTED" else None
    )
    result = _signed({
        "schema_version": 1, "campaign_id": CAMPAIGN_ID,
        "materialization_checksum": materialization["checksum"],
        "recovery_plan_checksum": _mapping(
            recovery.get("plan"), "recovery plan"
        )["checksum"],
        "recovery_enqueue_checksum": _mapping(
            recovery.get("enqueue"), "recovery enqueue"
        )["checksum"],
        "null_trigger_checksum": trigger["checksum"],
        "null_enqueue_checksum": _mapping(
            null_enqueue.get("receipt"), "null enqueue receipt"
        )["checksum"],
        "scientific_audit_passed": True, "scientific_audit_errors": [],
        "complete_spatial_runs": len(spatial),
        "complete_position_permuted_null_runs": len(null),
        "spatial_vs_position_permuted_null": {
            "log1p_huber": huber, "log1p_mae": mae,
            "secondary_metrics": secondary,
        },
        "target_mask_bin_effects": mask_effects.to_dict(orient="records"),
        "neighbor_observation_sanity": neighbor_sanity,
        "decision": decision, "decision_rationale": reasons,
        "maximum_supported_claim": maximum_claim,
        "claim_withheld_reason": (
            None if maximum_claim is not None else "topology support criteria were not met"
        ),
    })

    def build(stage: Path) -> None:
        tables, plots = stage / "tables", stage / "plots"
        tables.mkdir(); plots.mkdir()
        main.to_csv(tables / "audited_per_core_replicate.csv", index=False)
        mask_bins.to_csv(tables / "audited_per_target_mask_bin.csv", index=False)
        neighbor_bins.to_csv(
            tables / "audited_per_neighbor_observation_bin.csv", index=False
        )
        huber_paired.to_csv(
            tables / "paired_core_seed_spatial_vs_null_huber.csv", index=False
        )
        mae_paired.to_csv(
            tables / "paired_core_seed_spatial_vs_null_mae.csv", index=False
        )
        mask_effects.to_csv(
            tables / "topology_target_mask_bin_effects.csv", index=False
        )
        neighbor_effects.to_csv(
            tables / "topology_neighbor_observation_bin_effects.csv", index=False
        )
        _save_effect_plot(
            huber_paired, plots / "paired_core_spatial_vs_null_huber.png",
            left="spatial", right=NULL_CONDITION,
        )
        _write_json_file(stage / "aggregate_summary.json", result)
        (stage / "README.md").write_text(
            f"# Conditional spatial-topology null\n\n"
            f"- Evidence classification: **{decision}**\n"
            f"- Scientific audit: **passed**\n",
            encoding="utf-8",
        )

    _atomic_publish_report(output_root / "topology", build)
    return result


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("pilot-gate", "primary", "topology"), help="analysis stage"
    )
    parser.add_argument(
        "--materialization",
        type=Path,
        default=paths.project_root / LOCK_ROOT / "materialization_receipt.json",
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=paths.artifact_root
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=paths.project_root / DEFAULT_REPORT_ROOT,
    )
    parser.add_argument(
        "--pilot-gate-output",
        type=Path,
        default=paths.project_root / LOCK_ROOT / "pilot_gate_receipt.json",
    )
    parser.add_argument(
        "--null-trigger",
        type=Path,
        default=(
            paths.project_root / DEFAULT_REPORT_ROOT / "null_trigger_receipt.json"
        ),
    )
    parser.add_argument(
        "--recovery-plan",
        type=Path,
        default=paths.project_root / DEFAULT_RECOVERY_PLAN,
    )
    parser.add_argument(
        "--recovery-enqueue",
        type=Path,
        default=paths.project_root / DEFAULT_RECOVERY_ENQUEUE,
    )
    parser.add_argument(
        "--null-enqueue",
        type=Path,
        default=paths.project_root / DEFAULT_NULL_ENQUEUE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    materialization = _materialization(args.materialization)
    recovery = (
        None
        if args.mode == "pilot-gate"
        else _recovery_authorization(
            args.recovery_plan,
            args.recovery_enqueue,
            materialization=materialization,
        )
    )
    records = _run_bundles(
        args.artifact_root, materialization, recovery=recovery
    )
    if args.mode == "pilot-gate":
        result = write_pilot_gate(
            records,
            materialization=materialization,
            output=args.pilot_gate_output,
        )
    elif args.mode == "primary":
        assert recovery is not None
        result = analyze_primary(
            records,
            materialization=materialization,
            recovery=recovery,
            output_root=args.output_root,
        )
    else:
        assert recovery is not None
        result = analyze_topology(
            records,
            materialization=materialization,
            recovery=recovery,
            output_root=args.output_root,
            null_trigger_path=args.null_trigger,
            null_enqueue_path=args.null_enqueue,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
