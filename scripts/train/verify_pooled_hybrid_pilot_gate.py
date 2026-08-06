#!/usr/bin/env python3
"""Audit both pooled ten-core pilots and write the production gate receipt.

Scientific threshold failures are recorded as a checksum-bound negative gate.
Missing, malformed, or lineage-inconsistent evidence fails closed without
writing an authorization receipt.
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


CAMPAIGN_ID = "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
MATERIALIZATION_KIND = "pooled_hybrid_count_locked_config_materialization_v1"
PILOT_ENQUEUE_KIND = "pooled_hybrid_count_pilot_enqueue_v1"
PILOT_GATE_KIND = "pooled_hybrid_count_pilot_gate_v1"
EXPECTED_CONTRACT_SHA256 = (
    "c6af3dc756155ee502506f08304a7436ae99da36ad2b4ed8fae48672a312f6e2"
)
EXPECTED_PARAMETER_COUNT = 11_674_880
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
ARMS = ("pooled-hybrid-gat-k1000", "pooled-hybrid-matched-self")
EXPECTED_JOBS = {(arm, 0) for arm in ARMS}
MAXIMUM_ATTEMPTS = 2
EXPECTED_GLOBAL_EPOCHS = 2
EXPECTED_OPTIMIZER_STEPS = 20
MAX_AMP_DISCREPANCY = 1e-3
MAX_VRAM_GIB = 20.5
MAX_HOST_MEMORY_GIB = 40.0
MAX_GAT_PROJECTED_HOURS = 6.0
MIN_PROJECTED_FINAL_FREE_DISK_GIB = 27.5
PILOT_GATE_THRESHOLDS = {
    "fp32_amp_absolute_total_loss_discrepancy_each_core_maximum": (
        MAX_AMP_DISCREPANCY
    ),
    "peak_allocated_vram_gib_maximum": MAX_VRAM_GIB,
    "peak_host_memory_gib_per_process_maximum": MAX_HOST_MEMORY_GIB,
    "projected_200_epoch_gat_runtime_hours_maximum": (
        MAX_GAT_PROJECTED_HOURS
    ),
    "projected_final_free_disk_gib_minimum": (
        MIN_PROJECTED_FINAL_FREE_DISK_GIB
    ),
}
_MODEL_TO_ARM = {
    "hybrid-count-gat": "pooled-hybrid-gat-k1000",
    "hybrid-count-matched-self": "pooled-hybrid-matched-self",
}


class PooledHybridPilotGateError(RuntimeError):
    """Raised when pilot evidence is missing or internally inconsistent."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PooledHybridPilotGateError(f"{label} must be a mapping")
    return value


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PooledHybridPilotGateError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PooledHybridPilotGateError(
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
    except PooledHybridPilotGateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PooledHybridPilotGateError(f"{label} is not strict JSON") from exc
    return dict(_mapping(value, label))


def _verified_checksum(payload: Mapping[str, Any], *, label: str) -> str:
    checksum = payload.get("checksum")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
    ):
        raise PooledHybridPilotGateError(f"{label} checksum is malformed")
    unsigned = dict(payload)
    unsigned.pop("checksum", None)
    if canonical_sha256(unsigned) != checksum:
        raise PooledHybridPilotGateError(f"{label} checksum does not verify")
    return checksum


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise PooledHybridPilotGateError(f"{label} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise PooledHybridPilotGateError(f"{label} must be finite") from exc
    if not math.isfinite(converted):
        raise PooledHybridPilotGateError(f"{label} must be finite")
    return converted


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise PooledHybridPilotGateError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise PooledHybridPilotGateError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise PooledHybridPilotGateError(f"{label} must be an integer")
    return converted


def _digest(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PooledHybridPilotGateError(f"{label} must be a SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PooledHybridPilotGateError(
            "pilot resolved configuration is unreadable"
        ) from exc
    return dict(_mapping(value, "pilot resolved configuration"))


def _load_table(root: Path, stem: str) -> list[dict[str, Any]]:
    paths = [
        root / f"{stem}{suffix}"
        for suffix in (".jsonl", ".csv", ".parquet")
        if (root / f"{stem}{suffix}").is_file()
    ]
    if len(paths) != 1:
        raise PooledHybridPilotGateError(
            f"pilot bundle requires exactly one {stem} table"
        )
    path = paths[0]
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PooledHybridPilotGateError(
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
            raise PooledHybridPilotGateError(
                "pyarrow is required to verify a Parquet pilot table"
            ) from exc
        rows = [dict(row) for row in parquet.read_table(path).to_pylist()]
    if not rows:
        raise PooledHybridPilotGateError(f"pilot {stem} table is empty")
    return rows


def _slot(job: Mapping[str, Any]) -> tuple[str, int]:
    return (
        str(job.get("arm")),
        _integer(job.get("seed"), label="pilot seed"),
    )


def _load_materialization(path: Path) -> dict[str, Any]:
    materialization = _strict_json(path, label="locked materialization")
    _verified_checksum(materialization, label="locked materialization")
    frozen = _mapping(
        materialization.get("frozen_contract"), "materialization frozen contract"
    )
    counts = _mapping(materialization.get("counts"), "materialization counts")
    cohort = _mapping(materialization.get("cohort"), "materialization cohort")
    jobs = materialization.get("pilot_jobs")
    aliases = cohort.get("aliases")
    if (
        materialization.get("schema_version") != 1
        or materialization.get("receipt_kind") != MATERIALIZATION_KIND
        or materialization.get("campaign_id") != CAMPAIGN_ID
        or materialization.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or frozen.get("sha256") != EXPECTED_CONTRACT_SHA256
        or counts.get("aliases") != 10
        or counts.get("pilot_configs") != 2
        or counts.get("production_configs") != 14
        or counts.get("production_seeds") != 7
        or not isinstance(aliases, list)
        or tuple(aliases) != ALIASES
        or not isinstance(jobs, list)
        or len(jobs) != 2
        or not all(isinstance(job, Mapping) for job in jobs)
        or {_slot(job) for job in jobs} != EXPECTED_JOBS
    ):
        raise PooledHybridPilotGateError(
            "locked materialization does not describe the exact pooled pilots"
        )
    return materialization


def _load_enqueue_receipt(
    path: Path, *, materialization: Mapping[str, Any]
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
        or {_slot(job) for job in jobs} != EXPECTED_JOBS
    ):
        raise PooledHybridPilotGateError(
            "pilot enqueue receipt is incomplete or mismatched"
        )
    return receipt


def _materialized_job_map(
    materialization: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        _slot(job): job
        for job in materialization["pilot_jobs"]
        if isinstance(job, Mapping)
    }


def _campaign_queue_jobs(registry: Registry) -> list[dict[str, Any]]:
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
            raise PooledHybridPilotGateError(
                "campaign queue inventory contains invalid JSON"
            ) from exc
        if not isinstance(item["canonical_config"], Mapping) or not isinstance(
            item["command"], list
        ):
            raise PooledHybridPilotGateError(
                "campaign queue inventory contains invalid job semantics"
            )
        result.append(item)
    return result


def _attempt_evidence(
    job: Mapping[str, Any], *, original_job_id: str, completed_job_id: str
) -> dict[str, Any]:
    return {
        "job_id": str(job["job_id"]),
        "attempt": int(job["attempt_count"]),
        "maximum_attempts": int(job["maximum_attempts"]),
        "status": str(job["status"]),
        "run_id": None if job.get("run_id") is None else str(job["run_id"]),
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
    planned: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    receipt_jobs = enqueue.get("jobs")
    if not isinstance(receipt_jobs, list):
        raise PooledHybridPilotGateError("pilot enqueue jobs must be a list")
    receipt_by_slot = {
        _slot(job): job for job in receipt_jobs if isinstance(job, Mapping)
    }
    if set(receipt_by_slot) != EXPECTED_JOBS or len(receipt_jobs) != 2:
        raise PooledHybridPilotGateError(
            "pilot enqueue receipt does not identify exactly two roots"
        )
    root_ids = [str(job.get("job_id")) for job in receipt_jobs]
    if any(not value or value == "None" for value in root_ids) or len(
        set(root_ids)
    ) != 2:
        raise PooledHybridPilotGateError(
            "pilot enqueue root identities are invalid"
        )

    inventory = _campaign_queue_jobs(registry)
    by_id = {str(job.get("job_id")): job for job in inventory}
    if len(by_id) != len(inventory):
        raise PooledHybridPilotGateError(
            "campaign queue contains duplicate job identities"
        )
    children: dict[str, list[str]] = {job_id: [] for job_id in by_id}
    for job_id, job in by_id.items():
        parent = job.get("retry_of")
        if parent is None:
            continue
        parent_id = str(parent)
        if parent_id not in by_id:
            raise PooledHybridPilotGateError(
                f"pilot retry {job_id} has an unrelated parent"
            )
        children[parent_id].append(job_id)
    if any(len(descendants) > 1 for descendants in children.values()):
        raise PooledHybridPilotGateError("pilot retry lineage branches")
    if any(root_id not in by_id for root_id in root_ids):
        raise PooledHybridPilotGateError(
            "pilot enqueue root is absent from the registry"
        )
    observed_roots: set[str] = set()
    for job_id in by_id:
        cursor = job_id
        visited: set[str] = set()
        while by_id[cursor].get("retry_of") is not None:
            if cursor in visited:
                raise PooledHybridPilotGateError("pilot retry lineage cycles")
            visited.add(cursor)
            cursor = str(by_id[cursor]["retry_of"])
        observed_roots.add(cursor)
    if observed_roots != set(root_ids):
        raise PooledHybridPilotGateError(
            "campaign queue contains an unrelated or duplicate pilot root"
        )

    resolved: dict[tuple[str, int], dict[str, Any]] = {}
    for slot in sorted(EXPECTED_JOBS):
        receipt_job = receipt_by_slot[slot]
        planned_job = planned[slot]
        root_id = str(receipt_job["job_id"])
        chain: list[dict[str, Any]] = []
        cursor = root_id
        while True:
            chain.append(by_id[cursor])
            descendants = children[cursor]
            if not descendants:
                break
            cursor = descendants[0]
        if len(chain) > MAXIMUM_ATTEMPTS:
            raise PooledHybridPilotGateError(
                f"pilot {slot[0]} exceeds maximum_attempts=2"
            )
        if (
            receipt_job.get("config_sha256")
            != planned_job.get("config_sha256")
            or receipt_job.get("requested_gpu")
            != planned_job.get("requested_gpu")
            or receipt_job.get("maximum_attempts") != MAXIMUM_ATTEMPTS
        ):
            raise PooledHybridPilotGateError(
                "pilot enqueue job differs from its materialized plan"
            )
        root = chain[0]
        for index, job in enumerate(chain, start=1):
            if (
                int(job.get("attempt_count", -1)) != index
                or int(job.get("maximum_attempts", -1)) != MAXIMUM_ATTEMPTS
                or (index == 1 and job.get("retry_of") is not None)
                or (
                    index > 1
                    and str(job.get("retry_of"))
                    != str(chain[index - 2].get("job_id"))
                )
                or canonical_sha256(dict(job["canonical_config"]))
                != planned_job.get("config_sha256")
                or str(job.get("requested_gpu"))
                != str(planned_job.get("requested_gpu"))
                or job["command"] != root["command"]
                or job.get("experiment_config_reference")
                != root.get("experiment_config_reference")
                or job.get("priority") != root.get("priority")
            ):
                raise PooledHybridPilotGateError(
                    f"pilot {slot[0]} retry semantics changed"
                )
        if str(root.get("experiment_config_reference")) != str(
            planned_job.get("config")
        ):
            raise PooledHybridPilotGateError(
                "pilot registry root differs from materialized config"
            )
        if any(
            str(job.get("status")) not in {"failed", "stale", "pruned"}
            for job in chain[:-1]
        ):
            raise PooledHybridPilotGateError(
                "only failed terminal attempts may have a retry"
            )
        terminal = chain[-1]
        if str(terminal.get("status")) != "completed" or not terminal.get("run_id"):
            raise PooledHybridPilotGateError(
                f"pilot {slot[0]} has no completed terminal attempt"
            )
        if (
            sum(str(job.get("status")) == "completed" for job in chain) != 1
        ):
            raise PooledHybridPilotGateError(
                "pilot retry lineage has duplicate completions"
            )
        for failed in chain[:-1]:
            if not isinstance(failed.get("failure_category"), str) or not str(
                failed["failure_category"]
            ).strip():
                raise PooledHybridPilotGateError(
                    "failed pilot attempt lacks a failure category"
                )
        attempts = [
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
            "attempts": attempts,
            "failed_attempts": attempts[:-1],
        }
    return resolved


def _bundle_marker(root: Path) -> str:
    marker = _strict_json(root / "_SUCCESS", label="bundle success marker")
    digest = marker.get("content_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PooledHybridPilotGateError(
            "bundle success marker lacks content hash"
        )
    return digest


def _verify_checkpoint(registry: Registry, *, run_id: str) -> dict[str, Any]:
    records = build_checkpoint_catalog(registry, current_paths(), run_ids=[run_id])
    if len(records) != 1:
        raise PooledHybridPilotGateError(
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
        raise PooledHybridPilotGateError(
            f"pilot run {run_id} checkpoint verification failed"
        )
    return {
        "checkpoint_id": str(record["checkpoint_id"]),
        "checkpoint_sha256": str(record["checkpoint_sha256"]),
        "checkpoint_epoch": 1,
        "checkpoint_role": "last",
        "checkpoint_verified": True,
    }


def _pilot_semantics(
    config: Mapping[str, Any], *, expected_attempt: int
) -> tuple[str, int]:
    validate_experiment_config(config)
    campaign = _mapping(config.get("campaign"), "pilot campaign")
    experiment = _mapping(config.get("experiment"), "pilot experiment")
    model = _mapping(config.get("model"), "pilot model")
    trainer = _mapping(config.get("trainer"), "pilot trainer")
    evaluation = _mapping(config.get("evaluation"), "pilot evaluation")
    dataset = _mapping(config.get("dataset"), "pilot dataset")
    arm = _MODEL_TO_ARM.get(str(model.get("name")))
    aliases = dataset.get("core_aliases")
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or experiment.get("arm") != arm
        or experiment.get("resource_pilot") is not True
        or trainer.get("optimizer") != "AdamW"
        or trainer.get("max_epochs") != 2
        or trainer.get("total_optimizer_steps") != 20
        or trainer.get("diagnostic_resource_pilot") is not True
        or evaluation.get("protocol")
        != "held_in_pooled_10core_fixed_budget"
        or evaluation.get("mask_replicates_per_mode") != 3
        or evaluation.get("diagnostic_only") is not True
        or not isinstance(aliases, list)
        or tuple(aliases) != ALIASES
        or config.get("seed") != 0
        or config.get("fold") != 0
        or config.get("attempt") != expected_attempt
        or arm is None
    ):
        raise PooledHybridPilotGateError(
            "pilot resolved config violates the frozen pooled contract"
        )
    return arm, 0


def _verify_checksum_bindings(
    evidence: Mapping[str, Any],
    *,
    label: str,
    materialization_checksum: str,
    config_sha256: str,
) -> None:
    if (
        evidence.get("materialization_checksum") != materialization_checksum
        or evidence.get("config_sha256") != config_sha256
    ):
        raise PooledHybridPilotGateError(
            f"{label} is not bound to the materialization and config"
        )


def _verify_core_step_history(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_OPTIMIZER_STEPS:
        raise PooledHybridPilotGateError(
            "pilot core-step history must contain exactly 20 optimizer steps"
        )
    by_epoch: dict[int, list[str]] = {0: [], 1: []}
    observed_steps: list[int] = []
    for index, row in enumerate(rows):
        epoch = _integer(
            row.get("global_epoch", row.get("epoch")),
            label=f"core_step_history[{index}].global_epoch",
        )
        alias = str(row.get("alias", row.get("core_alias")))
        step = _integer(
            row.get("optimizer_step", row.get("step")),
            label=f"core_step_history[{index}].optimizer_step",
        )
        if epoch not in by_epoch or alias not in ALIASES:
            raise PooledHybridPilotGateError(
                "pilot core-step history contains an invalid epoch or alias"
            )
        by_epoch[epoch].append(alias)
        observed_steps.append(step)
        metric_fields = (
            ("hybrid_loss", "train_hybrid_loss", "total_loss"),
            ("detection_bce", "train_detection_bce", "detection_loss"),
            ("ordinal_bce", "train_ordinal_bce", "ordinal_loss"),
            (
                "positive_continuous_huber",
                "train_positive_continuous_huber",
                "continuous_loss",
            ),
            ("gradient_norm",),
        )
        for alternatives in metric_fields:
            present = next((name for name in alternatives if name in row), None)
            if present is None:
                raise PooledHybridPilotGateError(
                    f"core-step history lacks {alternatives[0]}"
                )
            if _finite(
                row[present], label=f"core_step_history[{index}].{present}"
            ) < 0:
                raise PooledHybridPilotGateError(
                    "pilot core-step history contains a negative diagnostic"
                )
    if observed_steps not in [list(range(20)), list(range(1, 21))]:
        raise PooledHybridPilotGateError(
            "pilot optimizer-step indices are not contiguous"
        )
    if any(
        len(values) != 10 or set(values) != set(ALIASES)
        for values in by_epoch.values()
    ):
        raise PooledHybridPilotGateError(
            "every alias must occur exactly once in each pilot global epoch"
        )


def _verify_precision(
    precision: Mapping[str, Any],
) -> tuple[dict[str, str], float, bool, dict[str, dict[str, float]]]:
    raw = precision.get("per_core")
    if raw is None:
        raw = precision.get("aliases")
    per_core = _mapping(raw, "per-core precision diagnostics")
    if set(per_core) != set(ALIASES):
        raise PooledHybridPilotGateError(
            "precision diagnostics must contain exactly ten aliases"
        )
    checksums: dict[str, str] = {}
    values: dict[str, dict[str, float]] = {}
    discrepancies: list[float] = []
    for alias in ALIASES:
        row = _mapping(per_core[alias], f"{alias} precision diagnostic")
        fp32 = _finite(row.get("fp32_total_loss"), label=f"{alias} FP32 loss")
        amp = _finite(row.get("amp_total_loss"), label=f"{alias} AMP loss")
        discrepancy = _finite(
            row.get("absolute_total_loss_discrepancy"),
            label=f"{alias} precision discrepancy",
        )
        if discrepancy < 0:
            raise PooledHybridPilotGateError(
                f"{alias} precision discrepancy is negative"
            )
        if not math.isclose(discrepancy, abs(fp32 - amp), abs_tol=1e-12):
            raise PooledHybridPilotGateError(
                f"{alias} precision discrepancy was not recomputed correctly"
            )
        passed = discrepancy <= MAX_AMP_DISCREPANCY
        if row.get("passed") is not passed:
            raise PooledHybridPilotGateError(
                f"{alias} precision pass flag is inconsistent"
            )
        checksum = _digest(
            row.get("mask_checksum"), label=f"{alias} precision mask checksum"
        )
        checksums[alias] = checksum
        values[alias] = {
            "fp32_total_loss": fp32,
            "amp_total_loss": amp,
            "absolute_total_loss_discrepancy": discrepancy,
        }
        discrepancies.append(discrepancy)
    maximum = max(discrepancies)
    declared_maximum = _finite(
        precision.get("maximum_observed_discrepancy"),
        label="maximum FP32/AMP discrepancy",
    )
    if not math.isclose(maximum, declared_maximum, abs_tol=1e-12):
        raise PooledHybridPilotGateError(
            "maximum precision discrepancy is inconsistent"
        )
    passed = maximum <= MAX_AMP_DISCREPANCY
    if precision.get("all_cores_passed") is not passed:
        raise PooledHybridPilotGateError(
            "aggregate precision pass flag is inconsistent"
        )
    return checksums, maximum, passed, values


def _verify_one_pilot(
    registry: Registry,
    receipt_job: Mapping[str, Any],
    *,
    planned_job: Mapping[str, Any],
    lineage: Mapping[str, Any],
    materialization_checksum: str,
) -> dict[str, Any]:
    arm, seed = _slot(receipt_job)
    if (arm, seed) not in EXPECTED_JOBS:
        raise PooledHybridPilotGateError("pilot receipt contains an unsafe slot")
    original_job_id = str(receipt_job.get("job_id"))
    if lineage.get("original_job_id") != original_job_id:
        raise PooledHybridPilotGateError(
            "resolved lineage changed the pilot root identity"
        )
    queue_job = _mapping(
        lineage.get("completed_job"), "completed pilot queue job"
    )
    completed_job_id = str(queue_job.get("job_id"))
    completed_attempt = _integer(
        queue_job.get("attempt_count"), label="completed pilot attempt"
    )
    run_id = str(queue_job.get("run_id"))
    run = registry.get_run(run_id)
    if (
        run is None
        or run.get("campaign_id") != CAMPAIGN_ID
        or run.get("status") != "completed"
        or not run.get("artifact_path")
        or run.get("attempt") != completed_attempt
    ):
        raise PooledHybridPilotGateError(
            f"pilot {arm} is not a completed registered run"
        )
    attempts = lineage.get("attempts")
    failed_attempts = lineage.get("failed_attempts")
    if (
        not isinstance(attempts, list)
        or not isinstance(failed_attempts, list)
        or attempts[:-1] != failed_attempts
        or len(attempts) != completed_attempt
    ):
        raise PooledHybridPilotGateError(
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
                raise PooledHybridPilotGateError(
                    "pilot retry run lineage differs from queue lineage"
                )
            previous_run_id = str(attempt_run_id)

    root = Path(str(run["artifact_path"]))
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    root = root.resolve(strict=False)
    verification = verify_run_bundle(root)
    if verification.get("valid") is not True or verification.get("status") != "success":
        raise PooledHybridPilotGateError(f"pilot {arm} bundle failed verification")
    config = _load_yaml(root / "config.resolved.yaml")
    if _pilot_semantics(config, expected_attempt=completed_attempt) != (arm, seed):
        raise PooledHybridPilotGateError(
            "pilot bundle slot differs from its queue receipt"
        )
    normalized = dict(config)
    normalized["attempt"] = 1
    config_sha256 = canonical_sha256(normalized)
    if config_sha256 != planned_job.get("config_sha256"):
        raise PooledHybridPilotGateError("pilot resolved config digest changed")

    summary = _strict_json(root / "summary.json", label="pilot summary")
    resource = _strict_json(
        root / "diagnostics/resource_usage.json",
        label="pilot resource diagnostic",
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
        root / "provenance/pooled_training.json",
        label="pilot training provenance",
    )
    for evidence, label in (
        (summary, "pilot summary"),
        (resource, "pilot resource diagnostic"),
        (precision, "pilot precision diagnostic"),
        (parameter, "pilot parameter audit"),
        (convergence, "pilot convergence diagnostic"),
        (training, "pilot training provenance"),
    ):
        _verify_checksum_bindings(
            evidence,
            label=label,
            materialization_checksum=materialization_checksum,
            config_sha256=config_sha256,
        )

    core_steps = _load_table(root, "metrics/core_step_history")
    global_epochs = _load_table(root, "metrics/global_epoch_history")
    _verify_core_step_history(core_steps)
    if len(global_epochs) != 2 or [
        _integer(
            row.get("global_epoch", row.get("epoch")),
            label="global epoch history index",
        )
        for row in global_epochs
    ] != [0, 1]:
        raise PooledHybridPilotGateError(
            "pilot global-epoch history must contain epochs 0 and 1"
        )
    for index, row in enumerate(global_epochs):
        for field in (
            "mean_hybrid_loss",
            "mean_detection_bce",
            "mean_ordinal_bce",
            "mean_positive_continuous_huber",
            "duration_seconds",
        ):
            if _finite(
                row.get(field), label=f"global_epoch_history[{index}].{field}"
            ) < 0:
                raise PooledHybridPilotGateError(
                    "pilot global-epoch diagnostic is negative"
                )

    pilot_checks = _mapping(resource.get("pilot_checks"), "runner pilot checks")
    required_runner_checks = (
        "all_20_optimizer_steps_completed",
        "every_core_once_each_epoch",
        "precision_equivalence_passed",
        "peak_vram_passed",
        "peak_host_memory_passed",
        "projected_runtime_passed",
        "projected_disk_passed",
        "parameter_match_passed",
        "paired_initialization_passed",
        "finite_losses_and_gradients",
    )
    if (
        summary.get("run_id") != run_id
        or summary.get("status") != "success"
        or summary.get("training_exit_status") != "success"
        or summary.get("diagnostic_resource_pilot") is not True
        or summary.get("conclusion_eligible") is not False
        or summary.get("final_epoch") != 1
        or summary.get("fixed_epoch_budget") != 2
        or summary.get("optimizer_steps_completed") != 20
        or summary.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or summary.get("exact_parameter_match") is not True
        or summary.get("paired_initialization_match") is not True
        or resource.get("diagnostic_resource_pilot") is not True
        or resource.get("public_variant") != arm
        or resource.get("optimizer_steps_completed") != 20
        or resource.get("aliases") != list(ALIASES)
        or resource.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or parameter.get("exact_trainable_parameter_match") is not True
        or parameter.get("trainable_parameter_count_graph")
        != EXPECTED_PARAMETER_COUNT
        or parameter.get("trainable_parameter_count_self")
        != EXPECTED_PARAMETER_COUNT
        or parameter.get("encoder_initial_state_bit_identical") is not True
        or parameter.get("decoder_initial_state_bit_identical") is not True
        or convergence.get("all_global_epochs_completed") is not True
        or convergence.get("all_losses_and_gradients_finite") is not True
        or convergence.get("optimizer_steps_completed") != 20
        or training.get("final_epoch") != 1
        or training.get("fixed_epoch_budget") != 2
        or training.get("optimizer_steps_completed") != 20
        or training.get("aliases") != list(ALIASES)
        or set(pilot_checks) != set(required_runner_checks)
        or any(
            not isinstance(pilot_checks.get(field), bool)
            for field in required_runner_checks
        )
    ):
        raise PooledHybridPilotGateError(
            f"pilot {arm} structural diagnostics are inconsistent"
        )

    precision_checksums, maximum_discrepancy, precision_pass, precision_values = (
        _verify_precision(precision)
    )
    peak_vram = _finite(
        resource.get("peak_allocated_vram_gib"), label="peak allocated VRAM"
    )
    peak_host = _finite(
        resource.get("peak_host_memory_gib"), label="peak host memory"
    )
    projected_hours = _finite(
        resource.get("projected_200_epoch_runtime_hours"),
        label="projected 200-epoch runtime",
    )
    projected_disk = _finite(
        resource.get("projected_final_free_disk_gib"),
        label="projected final free disk",
    )
    if min(peak_vram, peak_host, projected_hours, projected_disk) < 0:
        raise PooledHybridPilotGateError(
            f"pilot {arm} contains a negative resource diagnostic"
        )
    all_steps_pass = True
    visits_pass = True
    finite_pass = True
    parameter_pass = True
    paired_initialization_pass = True
    vram_pass = peak_vram <= MAX_VRAM_GIB
    host_pass = peak_host <= MAX_HOST_MEMORY_GIB
    runtime_pass = (
        arm != "pooled-hybrid-gat-k1000"
        or projected_hours <= MAX_GAT_PROJECTED_HOURS
    )
    disk_pass = projected_disk >= MIN_PROJECTED_FINAL_FREE_DISK_GIB
    calculated_checks = {
        "all_20_optimizer_steps_completed": all_steps_pass,
        "every_core_once_each_epoch": visits_pass,
        "precision_equivalence_passed": precision_pass,
        "peak_vram_passed": vram_pass,
        "peak_host_memory_passed": host_pass,
        "projected_runtime_passed": runtime_pass,
        "projected_disk_passed": disk_pass,
        "parameter_match_passed": parameter_pass,
        "paired_initialization_passed": paired_initialization_pass,
        "finite_losses_and_gradients": finite_pass,
    }
    if any(
        pilot_checks[field] is not calculated_checks[field]
        for field in calculated_checks
    ):
        raise PooledHybridPilotGateError(
            f"pilot {arm} runner and independent check flags differ"
        )
    runner_pass = all(calculated_checks.values())
    if (
        resource.get("pilot_gate_passed") is not runner_pass
        or summary.get("pilot_gate_passed") is not runner_pass
    ):
        raise PooledHybridPilotGateError(
            f"pilot {arm} runner and independent gate decisions differ"
        )

    checkpoint = _verify_checkpoint(registry, run_id=run_id)
    return {
        "arm": arm,
        "seed": seed,
        "job_id": original_job_id,
        "original_enqueue_job_id": original_job_id,
        "completed_job_id": completed_job_id,
        "completed_attempt": completed_attempt,
        "attempt_count": len(attempts),
        "attempts": attempts,
        "failed_attempts": failed_attempts,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "materialization_checksum": materialization_checksum,
        "bundle_content_sha256": _bundle_marker(root),
        "verified_bundle": True,
        **checkpoint,
        "finite_losses_and_gradients": finite_pass,
        "all_20_optimizer_steps_completed": all_steps_pass,
        "every_core_once_each_epoch": visits_pass,
        "parameter_match": parameter_pass,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "paired_initialization_match": paired_initialization_pass,
        "encoder_initial_state_bit_identical": True,
        "decoder_initial_state_bit_identical": True,
        "encoder_initial_state_sha256": _digest(
            parameter.get("encoder_initial_state_sha256"),
            label="encoder initial state checksum",
        ),
        "decoder_initial_state_sha256": _digest(
            parameter.get("decoder_initial_state_sha256"),
            label="decoder initial state checksum",
        ),
        "precision_mask_checksums": precision_checksums,
        "per_core_precision": precision_values,
        "maximum_fp32_amp_absolute_total_loss_discrepancy": (
            maximum_discrepancy
        ),
        "precision_equivalence_passed": precision_pass,
        "peak_allocated_vram_gib": peak_vram,
        "peak_vram_passed": vram_pass,
        "peak_host_memory_gib": peak_host,
        "peak_host_memory_passed": host_pass,
        "projected_200_epoch_runtime_hours": projected_hours,
        "projected_runtime_passed": runtime_pass,
        "projected_final_free_disk_gib": projected_disk,
        "projected_disk_passed": disk_pass,
        "runner_pilot_gate_passed": runner_pass,
        "evaluation_mask_bundle_sha256": _digest(
            summary.get("evaluation_mask_bundle_sha256"),
            label="evaluation mask bundle checksum",
        ),
        "graph_bundle_sha256": _digest(
            summary.get("graph_bundle_sha256"),
            label="graph bundle checksum",
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(
            dict(payload), indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
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
        if temporary_name and Path(temporary_name).exists():
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
        registry, enqueue=enqueue, planned=planned
    )
    jobs = [
        _verify_one_pilot(
            registry,
            receipt_job,
            planned_job=planned[_slot(receipt_job)],
            lineage=lineages[_slot(receipt_job)],
            materialization_checksum=str(materialization["checksum"]),
        )
        for receipt_job in sorted(enqueue["jobs"], key=_slot)
    ]
    if len({job["parameter_count"] for job in jobs}) != 1:
        raise PooledHybridPilotGateError("pilot arm parameter counts differ")
    if len({job["evaluation_mask_bundle_sha256"] for job in jobs}) != 1:
        raise PooledHybridPilotGateError(
            "pilot arms used different evaluation masks"
        )
    if len({job["graph_bundle_sha256"] for job in jobs}) != 1:
        raise PooledHybridPilotGateError(
            "pilot arms did not verify the same ten graphs"
        )
    for alias in ALIASES:
        if len(
            {job["precision_mask_checksums"][alias] for job in jobs}
        ) != 1:
            raise PooledHybridPilotGateError(
                f"pilot arms used different frozen precision mask for {alias}"
            )
    for field in (
        "encoder_initial_state_sha256",
        "decoder_initial_state_sha256",
    ):
        values = {job[field] for job in jobs}
        if None in values or len(values) != 1:
            raise PooledHybridPilotGateError(
                f"pilot paired initialization digest {field} differs"
            )

    failures: list[str] = []
    checked_fields = (
        "finite_losses_and_gradients",
        "all_20_optimizer_steps_completed",
        "every_core_once_each_epoch",
        "parameter_match",
        "paired_initialization_match",
        "precision_equivalence_passed",
        "peak_vram_passed",
        "peak_host_memory_passed",
        "projected_runtime_passed",
        "projected_disk_passed",
        "runner_pilot_gate_passed",
    )
    for job in jobs:
        for field in checked_fields:
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
        "thresholds": dict(PILOT_GATE_THRESHOLDS),
        "same_frozen_precision_batches_all_cores": True,
        "same_evaluation_masks": True,
        "same_verified_graph_bundle": True,
        "paired_initialization_digests_match": True,
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
