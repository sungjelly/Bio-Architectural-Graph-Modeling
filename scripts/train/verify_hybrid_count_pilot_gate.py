#!/usr/bin/env python3
"""Verify both ANC-01 pilots and write the production gate receipt.

Threshold failures are valid diagnostic results: when both pilot bundles are
structurally complete this command writes a checksum-bound receipt even when
the overall gate fails.  It returns a nonzero status for a failed gate so an
automation wrapper cannot silently continue to production.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.checkpoint_catalog import (  # noqa: E402
    build_checkpoint_catalog,
    verify_checkpoint_record,
)
from spatial_benchmark.configuration import validate_experiment_config  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
MATERIALIZATION_KIND = "hybrid_count_locked_config_materialization_v1"
PILOT_ENQUEUE_KIND = "hybrid_count_pilot_enqueue_v1"
PILOT_GATE_KIND = "hybrid_count_pilot_gate_v1"
EXPECTED_CONTRACT_SHA256 = (
    "2c4db92868ab37b806b82274f20f31e402b737714ff0a946f4954386db9eca1c"
)
EXPECTED_PARAMETER_COUNT = 11_674_880
EXPECTED_JOBS = {
    ("ANC-01", "hybrid-gat-k1000"),
    ("ANC-01", "hybrid-matched-self"),
}
MAX_VRAM_GIB = 20.5
MAX_AMP_DISCREPANCY = 1e-3
MAX_GAT_PROJECTED_HOURS = 6.0
_MODEL_TO_ARM = {
    "hybrid-count-gat": "hybrid-gat-k1000",
    "hybrid-count-matched-self": "hybrid-matched-self",
}


class HybridCountPilotGateError(RuntimeError):
    """Raised when pilot evidence is incomplete or internally inconsistent."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HybridCountPilotGateError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise HybridCountPilotGateError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HybridCountPilotGateError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except HybridCountPilotGateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HybridCountPilotGateError(f"{label} is not strict JSON") from exc
    return dict(_mapping(value, label))


def _verified_checksum(payload: Mapping[str, Any], *, label: str) -> str:
    checksum = payload.get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise HybridCountPilotGateError(f"{label} checksum is malformed")
    core = dict(payload)
    core.pop("checksum", None)
    if canonical_sha256(core) != checksum:
        raise HybridCountPilotGateError(f"{label} checksum does not verify")
    return checksum


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise HybridCountPilotGateError(f"{label} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise HybridCountPilotGateError(f"{label} must be finite") from exc
    if not math.isfinite(converted):
        raise HybridCountPilotGateError(f"{label} must be finite")
    return converted


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise HybridCountPilotGateError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise HybridCountPilotGateError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise HybridCountPilotGateError(f"{label} must be an integer")
    return converted


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HybridCountPilotGateError(
            "pilot resolved configuration is unreadable"
        ) from exc
    return dict(_mapping(value, "pilot resolved configuration"))


def _load_table(root: Path, stem: str) -> list[dict[str, Any]]:
    matches = [
        root / f"{stem}{suffix}"
        for suffix in (".jsonl", ".csv", ".parquet")
        if (root / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise HybridCountPilotGateError(
            f"pilot bundle requires exactly one {stem} table"
        )
    path = matches[0]
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        value = json.loads(line)
                    except ValueError as exc:
                        raise HybridCountPilotGateError(
                            f"{stem} row {number} is invalid JSON"
                        ) from exc
                    rows.append(dict(_mapping(value, f"{stem} row {number}")))
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    else:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise HybridCountPilotGateError(
                "pyarrow is required to verify a Parquet pilot table"
            ) from exc
        rows = [dict(row) for row in parquet.read_table(path).to_pylist()]
    if not rows:
        raise HybridCountPilotGateError(f"pilot {stem} table is empty")
    return rows


def _load_materialization(path: Path) -> dict[str, Any]:
    materialization = _strict_json(path, label="locked materialization")
    _verified_checksum(materialization, label="locked materialization")
    frozen = _mapping(
        materialization.get("frozen_contract"), "materialization frozen contract"
    )
    jobs = materialization.get("pilot_jobs")
    if not isinstance(jobs, list) or not all(
        isinstance(job, Mapping) for job in jobs
    ):
        raise HybridCountPilotGateError(
            "locked materialization pilot_jobs must be a list of mappings"
        )
    observed = {
        (str(job.get("alias")), str(job.get("arm")))
        for job in jobs
    }
    if (
        materialization.get("schema_version") != 1
        or materialization.get("receipt_kind") != MATERIALIZATION_KIND
        or materialization.get("campaign_id") != CAMPAIGN_ID
        or materialization.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or frozen.get("sha256") != EXPECTED_CONTRACT_SHA256
        or observed != EXPECTED_JOBS
        or len(jobs) != 2
    ):
        raise HybridCountPilotGateError(
            "locked materialization does not describe the exact two pilots"
        )
    return materialization


def _load_enqueue_receipt(
    path: Path,
    *,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _strict_json(path, label="pilot enqueue receipt")
    _verified_checksum(receipt, label="pilot enqueue receipt")
    jobs = receipt.get("jobs")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_kind") != PILOT_ENQUEUE_KIND
        or receipt.get("campaign_id") != CAMPAIGN_ID
        or receipt.get("stage") != "pilot"
        or receipt.get("materialization_checksum")
        != materialization.get("checksum")
        or receipt.get("pilot_gate_checksum") is not None
        or receipt.get("complete") is not True
        or not isinstance(jobs, list)
        or len(jobs) != 2
        or not all(isinstance(job, Mapping) for job in jobs)
        or {(str(job.get("alias")), str(job.get("arm"))) for job in jobs}
        != EXPECTED_JOBS
    ):
        raise HybridCountPilotGateError(
            "pilot enqueue receipt is incomplete or mismatched"
        )
    return receipt


def _materialized_job_map(
    materialization: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(job["alias"]), str(job["arm"])): job
        for job in materialization["pilot_jobs"]
        if isinstance(job, Mapping)
    }


def _campaign_queue_jobs(registry: Registry) -> list[dict[str, Any]]:
    """Read the complete campaign queue inventory without mutating it."""

    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT job_id, campaign_id, experiment_config_reference,
                   canonical_config_json, command_json, priority, status,
                   attempt_count, maximum_attempts, requested_gpu, retry_of,
                   run_id, failure_category
            FROM queue_jobs
            WHERE campaign_id = ?
            ORDER BY created_at, job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["canonical_config"] = json.loads(
                str(item.pop("canonical_config_json"))
            )
            item["command"] = json.loads(str(item.pop("command_json")))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise HybridCountPilotGateError(
                "campaign queue inventory contains invalid JSON"
            ) from exc
        if not isinstance(item["canonical_config"], Mapping) or not isinstance(
            item["command"], list
        ):
            raise HybridCountPilotGateError(
                "campaign queue inventory contains invalid job semantics"
            )
        result.append(item)
    return result


def _attempt_evidence(
    job: Mapping[str, Any],
    *,
    original_job_id: str,
    completed_job_id: str,
) -> dict[str, Any]:
    return {
        "job_id": str(job["job_id"]),
        "attempt": int(job["attempt_count"]),
        "maximum_attempts": int(job["maximum_attempts"]),
        "status": str(job["status"]),
        "run_id": (
            None if job.get("run_id") is None else str(job.get("run_id"))
        ),
        "failure_category": job.get("failure_category"),
        "retry_of": (
            None if job.get("retry_of") is None else str(job.get("retry_of"))
        ),
        "is_original_enqueue_job": str(job["job_id"]) == original_job_id,
        "selected_completed_attempt": str(job["job_id"]) == completed_job_id,
    }


def _resolve_pilot_attempt_lineages(
    registry: Registry,
    *,
    enqueue: Mapping[str, Any],
    planned: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Resolve exactly two unbranched pilot lineages through attempt two.

    The enqueue receipt remains the identity authority for each root job.
    Retry jobs are accepted only as exact, linearly linked execution attempts
    copied from that root. Any additional campaign job is therefore evidence
    of an unrelated or duplicate execution and fails the gate.
    """

    receipt_jobs = enqueue.get("jobs")
    if not isinstance(receipt_jobs, list):
        raise HybridCountPilotGateError(
            "pilot enqueue receipt jobs must be a list"
        )
    receipt_by_slot = {
        (str(job.get("alias")), str(job.get("arm"))): job
        for job in receipt_jobs
        if isinstance(job, Mapping)
    }
    if set(receipt_by_slot) != EXPECTED_JOBS or len(receipt_jobs) != 2:
        raise HybridCountPilotGateError(
            "pilot enqueue receipt does not identify exactly two roots"
        )
    root_ids = [str(job.get("job_id")) for job in receipt_jobs]
    if any(not value or value == "None" for value in root_ids) or len(
        set(root_ids)
    ) != 2:
        raise HybridCountPilotGateError(
            "pilot enqueue receipt root job identities are invalid"
        )

    inventory = _campaign_queue_jobs(registry)
    by_id = {str(job.get("job_id")): job for job in inventory}
    if len(by_id) != len(inventory):
        raise HybridCountPilotGateError(
            "campaign queue inventory contains duplicate job identities"
        )
    children: dict[str, list[str]] = {job_id: [] for job_id in by_id}
    for job_id, job in by_id.items():
        parent = job.get("retry_of")
        if parent is None:
            continue
        parent_id = str(parent)
        if parent_id not in by_id:
            raise HybridCountPilotGateError(
                f"pilot retry {job_id} has an unrelated parent"
            )
        children[parent_id].append(job_id)
    if any(len(values) > 1 for values in children.values()):
        raise HybridCountPilotGateError(
            "pilot retry lineage branches into multiple descendants"
        )

    expected_roots = set(root_ids)
    observed_roots: set[str] = set()
    for job_id in by_id:
        visited: set[str] = set()
        cursor = job_id
        while by_id[cursor].get("retry_of") is not None:
            if cursor in visited:
                raise HybridCountPilotGateError(
                    "pilot retry lineage contains a cycle"
                )
            visited.add(cursor)
            cursor = str(by_id[cursor]["retry_of"])
        observed_roots.add(cursor)
    if observed_roots != expected_roots:
        raise HybridCountPilotGateError(
            "campaign queue contains an unrelated or duplicate pilot root"
        )
    if any(root_id not in by_id for root_id in expected_roots):
        raise HybridCountPilotGateError(
            "pilot enqueue root is absent from the registry"
        )

    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for slot in sorted(EXPECTED_JOBS):
        receipt_job = receipt_by_slot[slot]
        planned_job = planned.get(slot)
        if planned_job is None:
            raise HybridCountPilotGateError(
                "pilot enqueue slot is absent from the materialized plan"
            )
        root_id = str(receipt_job["job_id"])
        chain: list[dict[str, Any]] = []
        cursor = root_id
        while True:
            chain.append(by_id[cursor])
            descendants = children[cursor]
            if not descendants:
                break
            cursor = descendants[0]
        if len(chain) > 2:
            raise HybridCountPilotGateError(
                f"pilot {slot[0]} {slot[1]} exceeds maximum_attempts=2"
            )

        planned_digest = planned_job.get("config_sha256")
        planned_gpu = planned_job.get("requested_gpu")
        if (
            receipt_job.get("config_sha256") != planned_digest
            or receipt_job.get("requested_gpu") != planned_gpu
            or receipt_job.get("maximum_attempts") != 2
        ):
            raise HybridCountPilotGateError(
                "pilot enqueue job differs from its materialized plan"
            )
        root = chain[0]
        root_command = root["command"]
        root_reference = root.get("experiment_config_reference")
        root_priority = root.get("priority")
        for index, job in enumerate(chain, start=1):
            if (
                int(job.get("attempt_count", -1)) != index
                or int(job.get("maximum_attempts", -1)) != 2
                or (
                    job.get("retry_of") is None
                    if index > 1
                    else job.get("retry_of") is not None
                )
                or (
                    index > 1
                    and str(job.get("retry_of"))
                    != str(chain[index - 2]["job_id"])
                )
                or canonical_sha256(dict(job["canonical_config"]))
                != planned_digest
                or str(job.get("requested_gpu")) != str(planned_gpu)
                or job["command"] != root_command
                or job.get("experiment_config_reference") != root_reference
                or job.get("priority") != root_priority
            ):
                raise HybridCountPilotGateError(
                    f"pilot {slot[0]} {slot[1]} retry semantics changed"
                )
        planned_reference = planned_job.get("config")
        if (
            str(root.get("job_id")) != root_id
            or (
                planned_reference is not None
                and str(root_reference) != str(planned_reference)
            )
        ):
            raise HybridCountPilotGateError(
                "pilot registry root differs from the enqueue receipt"
            )
        if any(
            str(job.get("status")) not in {"failed", "stale", "pruned"}
            for job in chain[:-1]
        ):
            raise HybridCountPilotGateError(
                "only failed terminal attempts may have a retry descendant"
            )
        terminal = chain[-1]
        if str(terminal.get("status")) != "completed" or not terminal.get(
            "run_id"
        ):
            raise HybridCountPilotGateError(
                f"pilot {slot[0]} {slot[1]} has no completed terminal attempt"
            )
        completed = [
            job for job in chain if str(job.get("status")) == "completed"
        ]
        if len(completed) != 1 or completed[0] is not terminal:
            raise HybridCountPilotGateError(
                "pilot retry lineage has duplicate completed attempts"
            )
        for failed in chain[:-1]:
            if not isinstance(failed.get("failure_category"), str) or not str(
                failed.get("failure_category")
            ).strip():
                raise HybridCountPilotGateError(
                    "failed pilot attempt lacks a failure category"
                )
        attempt_rows = [
            _attempt_evidence(
                job,
                original_job_id=root_id,
                completed_job_id=str(terminal["job_id"]),
            )
            for job in chain
        ]
        resolved[slot] = {
            "original_job_id": root_id,
            "completed_job": terminal,
            "attempts": attempt_rows,
            "failed_attempts": attempt_rows[:-1],
        }
    return resolved


def _bundle_marker(root: Path) -> str:
    marker = _strict_json(root / "_SUCCESS", label="bundle success marker")
    digest = marker.get("content_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise HybridCountPilotGateError("bundle success marker lacks content hash")
    return digest


def _verify_checkpoint(
    registry: Registry,
    *,
    run_id: str,
) -> dict[str, Any]:
    records = build_checkpoint_catalog(registry, current_paths(), run_ids=[run_id])
    if len(records) != 1:
        raise HybridCountPilotGateError(
            f"pilot run {run_id} requires one indexed checkpoint"
        )
    record = records[0]
    verified = verify_checkpoint_record(record, paths=current_paths())
    if (
        record.get("checkpoint_role") != "last"
        or record.get("best_epoch") != 1
        or record.get("historical") is not False
        or verified.get("verification_status") != "verified"
    ):
        raise HybridCountPilotGateError(
            f"pilot run {run_id} checkpoint role, epoch, or verification failed"
        )
    return {
        "checkpoint_id": str(record["checkpoint_id"]),
        "checkpoint_sha256": str(record["checkpoint_sha256"]),
        "checkpoint_epoch": 1,
        "checkpoint_role": "last",
        "checkpoint_verified": True,
    }


def _pilot_semantics(
    config: Mapping[str, Any],
    *,
    expected_attempt: int,
) -> tuple[str, str]:
    validate_experiment_config(config)
    campaign = _mapping(config.get("campaign"), "pilot campaign")
    dataset = _mapping(config.get("dataset"), "pilot dataset")
    experiment = _mapping(config.get("experiment"), "pilot experiment")
    model = _mapping(config.get("model"), "pilot model")
    trainer = _mapping(config.get("trainer"), "pilot trainer")
    evaluation = _mapping(config.get("evaluation"), "pilot evaluation")
    alias = str(dataset.get("biological_unit_alias"))
    arm = _MODEL_TO_ARM.get(str(model.get("name")))
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or alias != "ANC-01"
        or experiment.get("biological_unit_alias") != alias
        or experiment.get("arm") != arm
        or experiment.get("resource_pilot") is not True
        or trainer.get("optimizer") != "AdamW"
        or trainer.get("max_epochs") != 2
        or trainer.get("diagnostic_resource_pilot") is not True
        or evaluation.get("mask_replicates_per_mode") != 1
        or evaluation.get("diagnostic_only") is not True
        or config.get("seed") != 0
        or config.get("fold") != 0
        or config.get("attempt") != expected_attempt
        or arm is None
    ):
        raise HybridCountPilotGateError(
            "pilot resolved configuration violates the frozen diagnostic contract"
        )
    return alias, arm


def _verify_one_pilot(
    registry: Registry,
    receipt_job: Mapping[str, Any],
    planned_job: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    alias = str(receipt_job.get("alias"))
    arm = str(receipt_job.get("arm"))
    if (alias, arm) not in EXPECTED_JOBS:
        raise HybridCountPilotGateError("pilot receipt contains an unsafe slot")
    original_job_id = str(receipt_job.get("job_id"))
    if lineage.get("original_job_id") != original_job_id:
        raise HybridCountPilotGateError(
            "resolved retry lineage changed the original receipt identity"
        )
    job = _mapping(lineage.get("completed_job"), "completed pilot queue job")
    completed_job_id = str(job.get("job_id"))
    completed_attempt = _integer(
        job.get("attempt_count"), label="completed pilot attempt"
    )
    run_id = str(job["run_id"])
    run = registry.get_run(run_id)
    if (
        run is None
        or run.get("campaign_id") != CAMPAIGN_ID
        or run.get("status") != "completed"
        or not run.get("artifact_path")
        or run.get("attempt") != completed_attempt
    ):
        raise HybridCountPilotGateError(
            f"pilot {alias} {arm} is not a completed registered run"
        )
    attempts = lineage.get("attempts")
    failed_attempts = lineage.get("failed_attempts")
    if (
        not isinstance(attempts, list)
        or not isinstance(failed_attempts, list)
        or attempts[:-1] != failed_attempts
        or len(attempts) != completed_attempt
    ):
        raise HybridCountPilotGateError(
            "resolved pilot attempt evidence is inconsistent"
        )
    previous_run_id: str | None = None
    for attempt in attempts:
        attempt_run_id = attempt.get("run_id")
        if attempt_run_id is not None:
            attempt_run = registry.get_run(str(attempt_run_id))
            expected_statuses = (
                {"completed"}
                if attempt.get("selected_completed_attempt") is True
                else {"failed", "pruned"}
            )
            if (
                attempt_run is None
                or attempt_run.get("campaign_id") != CAMPAIGN_ID
                or attempt_run.get("status") not in expected_statuses
                or attempt_run.get("attempt") != attempt.get("attempt")
                or attempt_run.get("retry_of") != previous_run_id
            ):
                raise HybridCountPilotGateError(
                    "pilot retry run lineage differs from its queue attempts"
                )
            previous_run_id = str(attempt_run_id)
    root = Path(str(run["artifact_path"]))
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    root = root.resolve(strict=False)
    verification = verify_run_bundle(root)
    if verification.get("valid") is not True or verification.get("status") != "success":
        raise HybridCountPilotGateError(
            f"pilot {alias} {arm} bundle failed verification"
        )

    config = _load_yaml(root / "config.resolved.yaml")
    resolved_alias, resolved_arm = _pilot_semantics(
        config,
        expected_attempt=completed_attempt,
    )
    if (resolved_alias, resolved_arm) != (alias, arm):
        raise HybridCountPilotGateError("pilot bundle slot differs from queue receipt")
    normalized_config = dict(config)
    normalized_config["attempt"] = 1
    if canonical_sha256(normalized_config) != planned_job.get("config_sha256"):
        raise HybridCountPilotGateError("pilot resolved config digest changed")
    summary = _strict_json(root / "summary.json", label="pilot summary")
    resource = _strict_json(
        root / "diagnostics/resource_usage.json", label="pilot resource diagnostic"
    )
    precision = _strict_json(
        root / "diagnostics/fp32_amp_equivalence.json",
        label="pilot precision diagnostic",
    )
    parameter = _strict_json(
        root / "diagnostics/parameter_structure_audit.json",
        label="pilot parameter audit",
    )
    convergence = _strict_json(
        root / "diagnostics/training_convergence.json",
        label="pilot convergence diagnostic",
    )
    training = _strict_json(
        root / "provenance/full_core_training.json",
        label="pilot training provenance",
    )
    history = _load_table(root, "metrics/history")

    if (
        summary.get("run_id") != run_id
        or summary.get("status") != "success"
        or summary.get("training_exit_status") != "success"
        or summary.get("diagnostic_resource_pilot") is not True
        or summary.get("conclusion_eligible") is not False
        or summary.get("final_epoch") != 1
        or summary.get("fixed_epoch_budget") != 2
        or summary.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or summary.get("exact_parameter_match") is not True
        or resource.get("diagnostic_resource_pilot") is not True
        or resource.get("public_variant") != arm
        or resource.get("biological_unit_alias") != alias
        or resource.get("epochs_completed") != 2
        or resource.get("finite_losses_and_gradients") is not True
        or resource.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or parameter.get("exact_trainable_parameter_match") is not True
        or parameter.get("trainable_parameter_count_graph")
        != EXPECTED_PARAMETER_COUNT
        or parameter.get("trainable_parameter_count_self")
        != EXPECTED_PARAMETER_COUNT
        or parameter.get("encoder_initial_state_bit_identical") is not True
        or parameter.get("decoder_initial_state_bit_identical") is not True
        or convergence.get("all_epochs_completed") is not True
        or convergence.get("all_losses_and_gradients_finite") is not True
        or training.get("final_epoch") != 1
        or training.get("fixed_epoch_budget") != 2
    ):
        raise HybridCountPilotGateError(
            f"pilot {alias} {arm} structural diagnostics are inconsistent"
        )
    if len(history) != 2 or [
        _integer(row.get("epoch"), label="pilot history epoch") for row in history
    ] != [0, 1]:
        raise HybridCountPilotGateError("pilot history must contain epochs 0 and 1")
    for index, row in enumerate(history):
        for field in (
            "train_hybrid_loss",
            "train_detection_bce",
            "train_ordinal_bce",
            "train_positive_continuous_huber",
            "gradient_norm",
            "duration_seconds",
        ):
            if _finite(row.get(field), label=f"history[{index}].{field}") < 0:
                raise HybridCountPilotGateError("pilot history metric is negative")

    discrepancy = _finite(
        precision.get("absolute_total_loss_discrepancy"),
        label="FP32/AMP loss discrepancy",
    )
    peak = _finite(
        resource.get("peak_allocated_vram_gib"), label="peak allocated VRAM"
    )
    projected = _finite(
        resource.get("projected_200_epoch_runtime_hours"),
        label="projected 200-epoch runtime",
    )
    precision_pass = discrepancy <= MAX_AMP_DISCREPANCY
    vram_pass = peak <= MAX_VRAM_GIB
    runtime_pass = arm != "hybrid-gat-k1000" or projected <= MAX_GAT_PROJECTED_HOURS
    parameter_match = True
    finite_losses = True
    runner_pass = bool(summary.get("pilot_gate_passed"))
    calculated_pass = (
        precision_pass
        and vram_pass
        and runtime_pass
        and parameter_match
        and finite_losses
    )
    if (
        precision.get("passed") is not precision_pass
        or resource.get("pilot_gate_passed") is not calculated_pass
        or runner_pass is not calculated_pass
    ):
        raise HybridCountPilotGateError(
            f"pilot {alias} {arm} runner and independent gate decisions differ"
        )
    checkpoint = _verify_checkpoint(registry, run_id=run_id)
    return {
        "alias": alias,
        "arm": arm,
        "job_id": original_job_id,
        "original_enqueue_job_id": original_job_id,
        "completed_job_id": completed_job_id,
        "completed_attempt": completed_attempt,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "failed_attempts": failed_attempts,
        "run_id": run_id,
        "config_sha256": str(planned_job["config_sha256"]),
        "bundle_content_sha256": _bundle_marker(root),
        "verified_bundle": True,
        **checkpoint,
        "finite_losses_and_gradients": finite_losses,
        "parameter_match": parameter_match,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "encoder_initial_state_bit_identical": True,
        "decoder_initial_state_bit_identical": True,
        "precision_mask_checksum": str(precision.get("mask_checksum")),
        "fp32_total_loss": _finite(
            precision.get("fp32_total_loss"), label="FP32 total loss"
        ),
        "amp_total_loss": _finite(
            precision.get("amp_total_loss"), label="AMP total loss"
        ),
        "fp32_amp_absolute_total_loss_discrepancy": discrepancy,
        "precision_equivalence_passed": precision_pass,
        "peak_allocated_vram_gib": peak,
        "peak_vram_passed": vram_pass,
        "projected_200_epoch_runtime_hours": projected,
        "projected_runtime_passed": runtime_pass,
        "runner_pilot_gate_passed": runner_pass,
        "evaluation_mask_bundle_sha256": str(
            summary.get("evaluation_mask_bundle_sha256")
        ),
        "graph_sha256": str(summary.get("graph_sha256")),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            dict(payload), indent=2, sort_keys=True, ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def verify_pilot_gate(
    *,
    materialization_path: Path,
    enqueue_receipt_path: Path,
    output_path: Path,
    database_path: Path,
) -> dict[str, Any]:
    materialization = _load_materialization(materialization_path)
    enqueue = _load_enqueue_receipt(
        enqueue_receipt_path, materialization=materialization
    )
    registry = Registry(database_path)
    planned = _materialized_job_map(materialization)
    lineages = _resolve_pilot_attempt_lineages(
        registry,
        enqueue=enqueue,
        planned=planned,
    )
    jobs = [
        _verify_one_pilot(
            registry,
            receipt_job,
            planned_job=planned[
                (str(receipt_job["alias"]), str(receipt_job["arm"]))
            ],
            lineage=lineages[
                (str(receipt_job["alias"]), str(receipt_job["arm"]))
            ],
        )
        for receipt_job in sorted(
            enqueue["jobs"], key=lambda item: (str(item["alias"]), str(item["arm"]))
        )
    ]
    if len({job["parameter_count"] for job in jobs}) != 1:
        raise HybridCountPilotGateError("pilot arm parameter counts differ")
    if len({job["precision_mask_checksum"] for job in jobs}) != 1:
        raise HybridCountPilotGateError(
            "pilot FP32/AMP comparisons did not use the same frozen batch"
        )
    if len({job["evaluation_mask_bundle_sha256"] for job in jobs}) != 1:
        raise HybridCountPilotGateError("pilot arms used different evaluation masks")
    if len({job["graph_sha256"] for job in jobs}) != 1:
        raise HybridCountPilotGateError("pilot arms did not verify the same graph")
    failures: list[str] = []
    for job in jobs:
        for field in (
            "precision_equivalence_passed",
            "peak_vram_passed",
            "projected_runtime_passed",
            "runner_pilot_gate_passed",
        ):
            if job[field] is not True:
                failures.append(f"{job['arm']}:{field}")
    gate_passed = not failures
    gate: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": PILOT_GATE_KIND,
        "campaign_id": CAMPAIGN_ID,
        "materialization_checksum": materialization["checksum"],
        "pilot_enqueue_receipt_checksum": enqueue["checksum"],
        "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "thresholds": {
            "peak_allocated_vram_gib_maximum": MAX_VRAM_GIB,
            "fp32_amp_absolute_total_loss_discrepancy_maximum": (
                MAX_AMP_DISCREPANCY
            ),
            "projected_gat_runtime_hours_per_core_maximum": (
                MAX_GAT_PROJECTED_HOURS
            ),
        },
        "same_frozen_precision_batch": True,
        "same_evaluation_masks": True,
        "same_verified_graph": True,
        "jobs": jobs,
        "failure_reasons": failures,
        "gate_passed": gate_passed,
        "production_authorized": gate_passed,
    }
    gate["checksum"] = canonical_sha256(gate)
    _atomic_json(output_path, gate)
    return gate


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    locked = paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--materialization",
        type=Path,
        default=locked / "locked_config_materialization.json",
    )
    parser.add_argument(
        "--pilot-enqueue-receipt",
        type=Path,
        default=locked / "pilot_enqueue_receipt.json",
    )
    parser.add_argument(
        "--output", type=Path, default=locked / "pilot_gate_receipt.json"
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gate = verify_pilot_gate(
        materialization_path=args.materialization.resolve(),
        enqueue_receipt_path=args.pilot_enqueue_receipt.resolve(),
        output_path=args.output.resolve(),
        database_path=args.database.resolve(),
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "gate_passed": gate["gate_passed"],
                "failure_reasons": gate["failure_reasons"],
                "receipt": str(args.output),
                "checksum": gate["checksum"],
            },
            sort_keys=True,
        )
    )
    return 0 if gate["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
