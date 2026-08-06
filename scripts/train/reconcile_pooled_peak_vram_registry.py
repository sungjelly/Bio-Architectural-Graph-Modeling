#!/usr/bin/env python3
"""Plan or apply the pooled-campaign peak-VRAM registry reconciliation.

Planning is the default and does not update the registry or any run bundle.
Application requires both an existing checksum-bound plan and its exact
reviewed checksum.  The implementation follows the frozen procedure in
``registry_peak_vram_reconciliation.md`` and deliberately has no generic
campaign mode.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.identifiers import canonical_sha256, scientific_id  # noqa: E402
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.registry import Registry, utc_now  # noqa: E402
from spatial_benchmark.run_archive import (  # noqa: E402
    COMPLETION_MARKERS,
    verify_run_bundle,
)


CAMPAIGN_ID = "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
EXPECTED_PROTOCOL = "held_in_pooled_10core_fixed_budget"
EXPECTED_CONTRACT_SHA256 = (
    "c6af3dc756155ee502506f08304a7436ae99da36ad2b4ed8fae48672a312f6e2"
)
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
ARMS = ("pooled-hybrid-gat-k1000", "pooled-hybrid-matched-self")
SEEDS = tuple(range(7))
MAXIMUM_ATTEMPTS = 2
PLAN_KIND = "pooled_peak_vram_registry_reconciliation_plan_v1"
APPLICATION_KIND = "pooled_peak_vram_registry_reconciliation_application_v1"
MATERIALIZATION_KIND = "pooled_hybrid_count_locked_config_materialization_v1"
PILOT_ENQUEUE_KIND = "pooled_hybrid_count_pilot_enqueue_v1"
PILOT_GATE_KIND = "pooled_hybrid_count_pilot_gate_v1"
PRODUCTION_ENQUEUE_KIND = "pooled_hybrid_count_production_enqueue_v1"
_ACTIVE_QUEUE_STATUSES = frozenset({"queued", "claimed", "running", "finalizing"})
_INCOMPLETE_RUN_STATUSES = frozenset({"pending", "running", "finalizing"})
_FAILED_ATTEMPT_QUEUE_STATUSES = frozenset({"failed", "stale", "pruned"})
_DIGEST_CHARS = frozenset("0123456789abcdef")
_TOLERANCE = 1e-12


class PeakVramReconciliationError(RuntimeError):
    """Raised when reconciliation evidence is incomplete or inconsistent."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PeakVramReconciliationError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PeakVramReconciliationError(f"{label} must be a list")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise PeakVramReconciliationError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise PeakVramReconciliationError(f"{label} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise PeakVramReconciliationError(f"{label} must be an integer")
    return converted


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PeakVramReconciliationError(
            f"{label} must be a finite nonnegative number"
        )
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise PeakVramReconciliationError(
            f"{label} must be a finite nonnegative number"
        ) from exc
    if not math.isfinite(converted) or converted < 0:
        raise PeakVramReconciliationError(
            f"{label} must be a finite nonnegative number"
        )
    return converted


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARS for character in value)
    ):
        raise PeakVramReconciliationError(f"{label} must be a lowercase SHA-256")
    return value


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PeakVramReconciliationError(f"cannot hash required file: {path}") from exc
    return digest.hexdigest()


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PeakVramReconciliationError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PeakVramReconciliationError(
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
    except PeakVramReconciliationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PeakVramReconciliationError(f"{label} is not strict JSON") from exc
    return dict(_mapping(value, label))


def _loads_strict(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise PeakVramReconciliationError(f"{label} is not serialized JSON")

    def reject_constant(constant: str) -> None:
        raise PeakVramReconciliationError(
            f"{label} contains non-finite JSON constant {constant!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PeakVramReconciliationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except PeakVramReconciliationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PeakVramReconciliationError(f"{label} is not strict JSON") from exc
    return dict(_mapping(parsed, label))


def _verified_signed_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _strict_json(path, label=label)
    checksum = _digest(payload.get("checksum"), f"{label} checksum")
    unsigned = dict(payload)
    unsigned.pop("checksum", None)
    if not hmac.compare_digest(canonical_sha256(unsigned), checksum):
        raise PeakVramReconciliationError(f"{label} checksum does not verify")
    return payload


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)
    return result


def _resolve_project_file(
    paths: ProjectPaths, reference: Any, *, label: str
) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise PeakVramReconciliationError(f"{label} reference is missing")
    path = Path(reference)
    if not path.is_absolute():
        path = paths.project_root / path
    path = path.resolve(strict=False)
    if not path.is_file():
        raise PeakVramReconciliationError(f"{label} is missing: {path}")
    return path


def _slot(
    row: Mapping[str, Any], *, stage: str, label: str
) -> tuple[str, str, int]:
    arm = str(row.get("arm", ""))
    seed = _integer(row.get("seed"), f"{label} seed")
    expected_seeds = (0,) if stage == "pilot" else SEEDS
    if arm not in ARMS or seed not in expected_seeds:
        raise PeakVramReconciliationError(
            f"{label} has an invalid {stage} arm/seed slot"
        )
    return stage, arm, seed


def _validate_config_slot(
    config: Mapping[str, Any],
    *,
    stage: str,
    arm: str,
    seed: int,
    label: str,
    require_root_attempt: bool,
) -> None:
    campaign = _mapping(config.get("campaign"), f"{label} campaign")
    dataset = _mapping(config.get("dataset"), f"{label} dataset")
    experiment = _mapping(config.get("experiment"), f"{label} experiment")
    trainer = _mapping(config.get("trainer"), f"{label} trainer")
    evaluation = _mapping(config.get("evaluation"), f"{label} evaluation")
    classification = _mapping(
        config.get("classification"), f"{label} classification"
    )
    metadata = _mapping(config.get("metadata"), f"{label} metadata")
    is_pilot = stage == "pilot"
    expected_lifecycle = "diagnostic" if is_pilot else "exploratory_screen"
    expected_role = "resource_pilot" if is_pilot else "production"
    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or campaign.get("frozen_contract_sha256") != EXPECTED_CONTRACT_SHA256
        or dataset.get("core_aliases") != list(ALIASES)
        or experiment.get("core_aliases") != list(ALIASES)
        or experiment.get("arm") != arm
        or experiment.get("resource_pilot") is not is_pilot
        or trainer.get("diagnostic_resource_pilot") is not is_pilot
        or evaluation.get("protocol") != EXPECTED_PROTOCOL
        or classification.get("scientific_variant") != arm
        or classification.get("lifecycle_stage") != expected_lifecycle
        or metadata.get("execution_role") != expected_role
        or _integer(config.get("seed"), f"{label} seed") != seed
    ):
        raise PeakVramReconciliationError(
            f"{label} does not match the frozen pooled campaign slot"
        )
    attempt = _integer(config.get("attempt"), f"{label} attempt")
    if attempt < 1 or (require_root_attempt and attempt != 1):
        raise PeakVramReconciliationError(f"{label} has an invalid attempt")


def _normalized_config_sha256(config: Mapping[str, Any]) -> str:
    normalized = deepcopy(dict(config))
    normalized["attempt"] = 1
    return canonical_sha256(normalized)


def _materialized_jobs(
    *,
    paths: ProjectPaths,
    materialization: Mapping[str, Any],
    key: str,
    stage: str,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    raw = _sequence(materialization.get(key), f"materialization {key}")
    expected_count = 2 if stage == "pilot" else 14
    if len(raw) != expected_count:
        raise PeakVramReconciliationError(
            f"materialization must contain {expected_count} {stage} jobs"
        )
    jobs: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, value in enumerate(raw):
        row = dict(_mapping(value, f"materialization {key}[{index}]"))
        slot = _slot(row, stage=stage, label=f"materialization {key}[{index}]")
        if slot in jobs:
            raise PeakVramReconciliationError(
                f"materialization duplicates slot {slot}"
            )
        config_path = _resolve_project_file(
            paths, row.get("config"), label=f"materialized config {slot}"
        )
        if _sha256_file(config_path) != _digest(
            row.get("file_sha256"), f"materialized config file {slot}"
        ):
            raise PeakVramReconciliationError(
                f"materialized config file checksum changed for {slot}"
            )
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise PeakVramReconciliationError(
                f"materialized config is unreadable for {slot}"
            ) from exc
        config = dict(_mapping(config, f"materialized config {slot}"))
        _validate_config_slot(
            config,
            stage=stage,
            arm=slot[1],
            seed=slot[2],
            label=f"materialized config {slot}",
            require_root_attempt=True,
        )
        config_sha256 = _digest(
            row.get("config_sha256"), f"materialized config digest {slot}"
        )
        if canonical_sha256(config) != config_sha256:
            raise PeakVramReconciliationError(
                f"materialized config canonical checksum changed for {slot}"
            )
        requested_gpu = _integer(
            row.get("requested_gpu"), f"materialized requested GPU {slot}"
        )
        if requested_gpu not in {0, 1, 2, 3, 5, 6, 7}:
            raise PeakVramReconciliationError(
                f"materialized config uses an unsafe GPU for {slot}"
            )
        row.update(
            {
                "slot": slot,
                "config_reference": str(row.get("config")),
                "config_path": config_path.as_posix(),
                "config_payload": config,
                "config_sha256": config_sha256,
                "requested_gpu": requested_gpu,
            }
        )
        jobs[slot] = row
    expected_slots = (
        {("pilot", arm, 0) for arm in ARMS}
        if stage == "pilot"
        else {("production", arm, seed) for arm in ARMS for seed in SEEDS}
    )
    if set(jobs) != expected_slots:
        raise PeakVramReconciliationError(
            f"materialization lacks the exact frozen {stage} slots"
        )
    return jobs


def _enqueue_jobs(
    receipt: Mapping[str, Any],
    *,
    stage: str,
    materialized: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows = _sequence(receipt.get("jobs"), f"{stage} enqueue jobs")
    if len(rows) != len(materialized):
        raise PeakVramReconciliationError(
            f"{stage} enqueue receipt has the wrong job count"
        )
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    root_ids: set[str] = set()
    for index, value in enumerate(rows):
        row = dict(_mapping(value, f"{stage} enqueue job[{index}]"))
        slot = _slot(row, stage=stage, label=f"{stage} enqueue job[{index}]")
        if slot in result:
            raise PeakVramReconciliationError(
                f"{stage} enqueue receipt duplicates slot {slot}"
            )
        root_id = str(row.get("job_id", ""))
        if not root_id or root_id in root_ids:
            raise PeakVramReconciliationError(
                f"{stage} enqueue receipt has an invalid root job ID"
            )
        root_ids.add(root_id)
        planned = materialized.get(slot)
        if (
            planned is None
            or row.get("config_sha256") != planned.get("config_sha256")
            or _integer(
                row.get("requested_gpu"), f"{stage} enqueue requested GPU"
            )
            != planned.get("requested_gpu")
            or _integer(
                row.get("maximum_attempts"), f"{stage} enqueue attempt budget"
            )
            != MAXIMUM_ATTEMPTS
        ):
            raise PeakVramReconciliationError(
                f"{stage} enqueue identity changed for {slot}"
            )
        row["slot"] = slot
        result[slot] = row
    if set(result) != set(materialized):
        raise PeakVramReconciliationError(
            f"{stage} enqueue receipt lacks frozen slot coverage"
        )
    return result


def _validate_receipts(
    *,
    paths: ProjectPaths,
    materialization_path: Path,
    pilot_enqueue_path: Path,
    pilot_gate_path: Path,
    production_enqueue_path: Path,
) -> dict[str, Any]:
    receipt_paths = {
        "materialization": materialization_path.resolve(strict=False),
        "pilot_enqueue": pilot_enqueue_path.resolve(strict=False),
        "pilot_gate": pilot_gate_path.resolve(strict=False),
        "production_enqueue": production_enqueue_path.resolve(strict=False),
    }
    materialization = _verified_signed_json(
        receipt_paths["materialization"], label="materialization receipt"
    )
    counts = _mapping(materialization.get("counts"), "materialization counts")
    cohort = _mapping(materialization.get("cohort"), "materialization cohort")
    frozen = _mapping(
        materialization.get("frozen_contract"), "materialization frozen contract"
    )
    if (
        materialization.get("schema_version") != 1
        or materialization.get("receipt_kind") != MATERIALIZATION_KIND
        or materialization.get("campaign_id") != CAMPAIGN_ID
        or frozen.get("sha256") != EXPECTED_CONTRACT_SHA256
        or counts.get("aliases") != 10
        or counts.get("pilot_configs") != 2
        or counts.get("production_configs") != 14
        or counts.get("production_seeds") != 7
        or cohort.get("aliases") != list(ALIASES)
        or materialization.get("queue_mutation_performed") is not False
        or materialization.get("registry_mutation_performed") is not False
        or materialization.get("training_performed") is not False
    ):
        raise PeakVramReconciliationError(
            "materialization receipt violates the frozen campaign identity"
        )
    contract_path = _resolve_project_file(
        paths, frozen.get("reference"), label="frozen task contract"
    )
    if _sha256_file(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise PeakVramReconciliationError("frozen task contract checksum changed")
    pilot_materialized = _materialized_jobs(
        paths=paths,
        materialization=materialization,
        key="pilot_jobs",
        stage="pilot",
    )
    production_materialized = _materialized_jobs(
        paths=paths,
        materialization=materialization,
        key="production_jobs",
        stage="production",
    )

    pilot_enqueue = _verified_signed_json(
        receipt_paths["pilot_enqueue"], label="pilot enqueue receipt"
    )
    if (
        pilot_enqueue.get("schema_version") != 1
        or pilot_enqueue.get("receipt_kind") != PILOT_ENQUEUE_KIND
        or pilot_enqueue.get("campaign_id") != CAMPAIGN_ID
        or pilot_enqueue.get("stage") != "pilot"
        or pilot_enqueue.get("materialization_checksum")
        != materialization.get("checksum")
        or pilot_enqueue.get("pilot_gate_checksum") is not None
        or pilot_enqueue.get("complete") is not True
    ):
        raise PeakVramReconciliationError("pilot enqueue receipt is not frozen")
    pilot_roots = _enqueue_jobs(
        pilot_enqueue, stage="pilot", materialized=pilot_materialized
    )

    pilot_gate = _verified_signed_json(
        receipt_paths["pilot_gate"], label="pilot gate receipt"
    )
    gate_jobs = _sequence(pilot_gate.get("jobs"), "pilot gate jobs")
    if (
        pilot_gate.get("schema_version") != 1
        or pilot_gate.get("receipt_kind") != PILOT_GATE_KIND
        or pilot_gate.get("campaign_id") != CAMPAIGN_ID
        or pilot_gate.get("frozen_contract_sha256") != EXPECTED_CONTRACT_SHA256
        or pilot_gate.get("materialization_checksum")
        != materialization.get("checksum")
        or pilot_gate.get("pilot_enqueue_receipt_checksum")
        != pilot_enqueue.get("checksum")
        or pilot_gate.get("gate_passed") is not True
        or pilot_gate.get("production_authorized") is not True
        or pilot_gate.get("failure_reasons") != []
        or len(gate_jobs) != 2
    ):
        raise PeakVramReconciliationError(
            "pilot gate does not authorize the frozen production campaign"
        )
    gate_by_slot: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, value in enumerate(gate_jobs):
        row = dict(_mapping(value, f"pilot gate job[{index}]"))
        slot = _slot(row, stage="pilot", label=f"pilot gate job[{index}]")
        root = pilot_roots.get(slot)
        if (
            root is None
            or slot in gate_by_slot
            or row.get("original_enqueue_job_id") != root.get("job_id")
            or row.get("config_sha256") != root.get("config_sha256")
            or row.get("verified_bundle") is not True
            or row.get("checkpoint_verified") is not True
        ):
            raise PeakVramReconciliationError(
                f"pilot gate lineage changed for {slot}"
            )
        gate_by_slot[slot] = row
    if set(gate_by_slot) != set(pilot_roots):
        raise PeakVramReconciliationError(
            "pilot gate lacks the exact paired pilot slots"
        )

    production_enqueue = _verified_signed_json(
        receipt_paths["production_enqueue"], label="production enqueue receipt"
    )
    if (
        production_enqueue.get("schema_version") != 1
        or production_enqueue.get("receipt_kind") != PRODUCTION_ENQUEUE_KIND
        or production_enqueue.get("campaign_id") != CAMPAIGN_ID
        or production_enqueue.get("stage") != "production"
        or production_enqueue.get("materialization_checksum")
        != materialization.get("checksum")
        or production_enqueue.get("pilot_gate_checksum")
        != pilot_gate.get("checksum")
        or production_enqueue.get("complete") is not True
    ):
        raise PeakVramReconciliationError(
            "production enqueue receipt is not bound to the passing pilot gate"
        )
    production_roots = _enqueue_jobs(
        production_enqueue,
        stage="production",
        materialized=production_materialized,
    )
    evidence = {
        name: {
            "path": path.as_posix(),
            "file_sha256": _sha256_file(path),
            "payload_checksum": {
                "materialization": materialization,
                "pilot_enqueue": pilot_enqueue,
                "pilot_gate": pilot_gate,
                "production_enqueue": production_enqueue,
            }[name]["checksum"],
        }
        for name, path in receipt_paths.items()
    }
    return {
        "materialization": materialization,
        "pilot_enqueue": pilot_enqueue,
        "pilot_gate": pilot_gate,
        "production_enqueue": production_enqueue,
        "materialized": {**pilot_materialized, **production_materialized},
        "roots": {**pilot_roots, **production_roots},
        "pilot_gate_by_slot": gate_by_slot,
        "receipt_evidence": evidence,
        "frozen_contract_path": contract_path.as_posix(),
    }


def _campaign_inventory(registry: Registry) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with registry.connect() as connection:
        queue_rows = connection.execute(
            """
            SELECT job_id, campaign_id, experiment_config_reference,
                   canonical_config_json, command_json, priority, status,
                   attempt_count, maximum_attempts, requested_gpu, retry_of,
                   run_id, failure_category, created_at
            FROM queue_jobs
            WHERE campaign_id = ?
            ORDER BY created_at, job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
        run_rows = connection.execute(
            """
            SELECT run_id, campaign_id, scientific_id, status, seed, fold,
                   attempt, peak_vram_gb, artifact_path, retry_of, config_json,
                   created_at
            FROM runs
            WHERE campaign_id = ?
            ORDER BY created_at, run_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    queue: list[dict[str, Any]] = []
    for raw in queue_rows:
        row = dict(raw)
        row["canonical_config"] = _loads_strict(
            row.pop("canonical_config_json"),
            label=f"queue config {row.get('job_id')}",
        )
        command = _loads_json_list(
            row.pop("command_json"), label=f"queue command {row.get('job_id')}"
        )
        row["command"] = command
        queue.append(row)
    runs: list[dict[str, Any]] = []
    for raw in run_rows:
        row = dict(raw)
        row["config"] = _loads_strict(
            row.pop("config_json"), label=f"run config {row.get('run_id')}"
        )
        runs.append(row)
    return queue, runs


def _assert_campaign_inactive(registry: Registry) -> None:
    with registry.connect() as connection:
        active_queue = [
            str(row["job_id"])
            for row in connection.execute(
                """
                SELECT job_id FROM queue_jobs
                WHERE campaign_id = ?
                  AND status IN ('queued', 'claimed', 'running', 'finalizing')
                ORDER BY created_at, job_id
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
        ]
        incomplete_runs = [
            str(row["run_id"])
            for row in connection.execute(
                """
                SELECT run_id FROM runs
                WHERE campaign_id = ?
                  AND status IN ('pending', 'running', 'finalizing')
                ORDER BY created_at, run_id
                """,
                (CAMPAIGN_ID,),
            ).fetchall()
        ]
    if active_queue or incomplete_runs:
        raise PeakVramReconciliationError(
            "campaign still contains active or incomplete work; "
            f"queue={active_queue}, runs={incomplete_runs}"
        )


def _loads_json_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, str):
        raise PeakVramReconciliationError(f"{label} is not serialized JSON")
    try:
        parsed = json.loads(
            value,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                PeakVramReconciliationError(
                    f"{label} contains non-finite JSON constant {constant!r}"
                )
            ),
        )
    except PeakVramReconciliationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PeakVramReconciliationError(f"{label} is not valid JSON") from exc
    return list(_sequence(parsed, label))


def _queue_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": str(row["job_id"]),
        "status": str(row["status"]),
        "attempt": _integer(row["attempt_count"], "queue attempt"),
        "maximum_attempts": _integer(
            row["maximum_attempts"], "queue maximum attempts"
        ),
        "requested_gpu": str(row["requested_gpu"]),
        "retry_of": None if row.get("retry_of") is None else str(row["retry_of"]),
        "run_id": None if row.get("run_id") is None else str(row["run_id"]),
        "config_sha256": canonical_sha256(row["canonical_config"]),
    }


def _run_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("peak_vram_gb")
    return {
        "run_id": str(row["run_id"]),
        "status": str(row["status"]),
        "attempt": _integer(row["attempt"], "run attempt"),
        "retry_of": None if row.get("retry_of") is None else str(row["retry_of"]),
        "artifact_path": (
            None if row.get("artifact_path") is None else str(row["artifact_path"])
        ),
        "registry_peak_vram_gb": (
            None
            if value is None
            else _finite_nonnegative(value, "registered peak VRAM")
        ),
        "normalized_config_sha256": _normalized_config_sha256(row["config"]),
    }


def _resolve_lineages(
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    run_rows: Sequence[Mapping[str, Any]],
    receipts: Mapping[str, Any],
) -> dict[str, Any]:
    active_queue = [
        str(row["job_id"])
        for row in queue_rows
        if str(row.get("status")) in _ACTIVE_QUEUE_STATUSES
    ]
    incomplete_runs = [
        str(row["run_id"])
        for row in run_rows
        if str(row.get("status")) in _INCOMPLETE_RUN_STATUSES
    ]
    if active_queue or incomplete_runs:
        raise PeakVramReconciliationError(
            "campaign still contains active or incomplete work; "
            f"queue={active_queue}, runs={incomplete_runs}"
        )
    by_job = {str(row["job_id"]): row for row in queue_rows}
    by_run = {str(row["run_id"]): row for row in run_rows}
    if len(by_job) != len(queue_rows) or len(by_run) != len(run_rows):
        raise PeakVramReconciliationError(
            "campaign registry inventory contains duplicate identities"
        )
    children: dict[str, list[str]] = {job_id: [] for job_id in by_job}
    for job_id, row in by_job.items():
        parent = row.get("retry_of")
        if parent is None:
            continue
        parent_id = str(parent)
        if parent_id not in by_job:
            raise PeakVramReconciliationError(
                f"queue retry {job_id} has an unrelated parent"
            )
        children[parent_id].append(job_id)
    if any(len(items) > 1 for items in children.values()):
        raise PeakVramReconciliationError("campaign retry lineage branches")

    expected_roots = {
        str(row["job_id"]): slot for slot, row in receipts["roots"].items()
    }
    if len(expected_roots) != 16:
        raise PeakVramReconciliationError(
            "signed receipts do not identify exactly sixteen roots"
        )
    observed_roots: set[str] = set()
    for start in by_job:
        cursor = start
        visited: set[str] = set()
        while by_job[cursor].get("retry_of") is not None:
            if cursor in visited:
                raise PeakVramReconciliationError("campaign retry lineage cycles")
            visited.add(cursor)
            cursor = str(by_job[cursor]["retry_of"])
        observed_roots.add(cursor)
    if observed_roots != set(expected_roots):
        raise PeakVramReconciliationError(
            "campaign queue contains an omitted, duplicate, or unrelated root"
        )

    selected: list[dict[str, Any]] = []
    queue_snapshots: list[dict[str, Any]] = []
    referenced_run_ids: set[str] = set()
    for root_id, slot in sorted(
        expected_roots.items(), key=lambda item: item[1]
    ):
        stage, arm, seed = slot
        root = by_job.get(root_id)
        if root is None or root.get("retry_of") is not None:
            raise PeakVramReconciliationError(f"slot {slot} lacks its signed root")
        planned = receipts["materialized"][slot]
        receipt_root = receipts["roots"][slot]
        chain: list[Mapping[str, Any]] = []
        cursor = root_id
        while True:
            chain.append(by_job[cursor])
            descendants = children[cursor]
            if not descendants:
                break
            cursor = descendants[0]
        if len(chain) > MAXIMUM_ATTEMPTS:
            raise PeakVramReconciliationError(
                f"slot {slot} exceeds its frozen retry budget"
            )
        root_command = root["command"]
        root_reference = root.get("experiment_config_reference")
        root_priority = root.get("priority")
        if str(root_reference) != str(planned.get("config_reference")):
            raise PeakVramReconciliationError(
                f"queue root config reference changed for slot {slot}"
            )
        previous_run_id: str | None = None
        for index, row in enumerate(chain, start=1):
            config = _mapping(row["canonical_config"], f"queue config {row['job_id']}")
            _validate_config_slot(
                config,
                stage=stage,
                arm=arm,
                seed=seed,
                label=f"queue config {row['job_id']}",
                require_root_attempt=True,
            )
            if (
                canonical_sha256(config) != planned["config_sha256"]
                or _integer(row.get("attempt_count"), "queue attempt") != index
                or _integer(row.get("maximum_attempts"), "queue retry budget")
                != MAXIMUM_ATTEMPTS
                or str(row.get("requested_gpu")) != str(planned["requested_gpu"])
                or row.get("command") != root_command
                or row.get("experiment_config_reference") != root_reference
                or row.get("priority") != root_priority
                or (
                    index == 1
                    and (
                        row.get("retry_of") is not None
                        or str(row.get("job_id")) != str(receipt_root["job_id"])
                    )
                )
                or (
                    index > 1
                    and str(row.get("retry_of"))
                    != str(chain[index - 2].get("job_id"))
                )
            ):
                raise PeakVramReconciliationError(
                    f"queue retry semantics changed for slot {slot}"
                )
            run_id = row.get("run_id")
            if run_id is not None:
                run_id = str(run_id)
                run = by_run.get(run_id)
                if run is None or run_id in referenced_run_ids:
                    raise PeakVramReconciliationError(
                        f"queue attempt has an absent or duplicate run: {run_id}"
                    )
                referenced_run_ids.add(run_id)
                run_config = _mapping(run["config"], f"registered run config {run_id}")
                _validate_config_slot(
                    run_config,
                    stage=stage,
                    arm=arm,
                    seed=seed,
                    label=f"registered run config {run_id}",
                    require_root_attempt=False,
                )
                if (
                    run.get("campaign_id") != CAMPAIGN_ID
                    or run.get("scientific_id") != scientific_id(run_config)
                    or _integer(run.get("seed"), "registered run seed") != seed
                    or _integer(run.get("fold"), "registered run fold") != 0
                    or _integer(run.get("attempt"), "registered run attempt") != index
                    or _integer(run_config.get("attempt"), "run config attempt")
                    != index
                    or _normalized_config_sha256(run_config)
                    != planned["config_sha256"]
                    or run.get("retry_of") != previous_run_id
                ):
                    raise PeakVramReconciliationError(
                        f"registered run lineage changed for {run_id}"
                    )
                if str(row.get("status")) == "completed":
                    if str(run.get("status")) != "completed":
                        raise PeakVramReconciliationError(
                            f"completed queue/run status disagrees for {run_id}"
                        )
                elif str(run.get("status")) in _INCOMPLETE_RUN_STATUSES:
                    raise PeakVramReconciliationError(
                        f"failed queue attempt retains incomplete run {run_id}"
                    )
                previous_run_id = run_id
            queue_snapshots.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "seed": seed,
                    **_queue_snapshot(row),
                }
            )
        completed = [row for row in chain if row.get("status") == "completed"]
        if (
            len(completed) != 1
            or completed[0] is not chain[-1]
            or any(
                str(row.get("status")) not in _FAILED_ATTEMPT_QUEUE_STATUSES
                for row in chain[:-1]
            )
        ):
            raise PeakVramReconciliationError(
                f"slot {slot} lacks one terminal completed attempt"
            )
        terminal = completed[0]
        run_id = terminal.get("run_id")
        if not isinstance(run_id, str) or run_id not in by_run:
            raise PeakVramReconciliationError(
                f"slot {slot} completion has no registered run"
            )
        selected.append(
            {
                "stage": stage,
                "arm": arm,
                "seed": seed,
                "root_job_id": root_id,
                "completed_job_id": str(terminal["job_id"]),
                "selected_attempt": _integer(
                    terminal["attempt_count"], "selected queue attempt"
                ),
                "run_id": run_id,
                "failed_attempts": [
                    {
                        "job_id": str(row["job_id"]),
                        "attempt": _integer(row["attempt_count"], "failed attempt"),
                        "status": str(row["status"]),
                        "run_id": (
                            None
                            if row.get("run_id") is None
                            else str(row.get("run_id"))
                        ),
                    }
                    for row in chain[:-1]
                ],
            }
        )
    if referenced_run_ids != set(by_run):
        raise PeakVramReconciliationError(
            "campaign contains an unqueued or omitted registered run"
        )
    if len(selected) != 16 or sum(row["stage"] == "pilot" for row in selected) != 2:
        raise PeakVramReconciliationError(
            "registry does not resolve exactly two pilots and fourteen production runs"
        )

    selected_by_slot = {
        (row["stage"], row["arm"], row["seed"]): row for row in selected
    }
    for slot, gate in receipts["pilot_gate_by_slot"].items():
        chosen = selected_by_slot.get(slot)
        attempts = _sequence(gate.get("attempts"), f"pilot gate attempts {slot}")
        expected_attempts = [
            row
            for row in queue_snapshots
            if (row["stage"], row["arm"], row["seed"]) == slot
        ]
        gate_attempts = [
            {
                "job_id": str(_mapping(value, "pilot gate attempt").get("job_id")),
                "attempt": _integer(
                    _mapping(value, "pilot gate attempt").get("attempt"),
                    "pilot gate attempt",
                ),
                "status": str(
                    _mapping(value, "pilot gate attempt").get("status", "")
                ),
                "run_id": _mapping(value, "pilot gate attempt").get("run_id"),
                "retry_of": _mapping(value, "pilot gate attempt").get("retry_of"),
            }
            for value in attempts
        ]
        expected_gate_attempts = [
            {
                "job_id": row["job_id"],
                "attempt": row["attempt"],
                "status": row["status"],
                "run_id": row["run_id"],
                "retry_of": row["retry_of"],
            }
            for row in expected_attempts
        ]
        if (
            chosen is None
            or gate.get("completed_job_id") != chosen["completed_job_id"]
            or gate.get("run_id") != chosen["run_id"]
            or _integer(gate.get("completed_attempt"), "gate completed attempt")
            != chosen["selected_attempt"]
            or gate_attempts != expected_gate_attempts
        ):
            raise PeakVramReconciliationError(
                f"pilot gate no longer matches registry lineage for {slot}"
            )
    return {
        "selected": sorted(
            selected, key=lambda row: (row["stage"], row["arm"], row["seed"])
        ),
        "queue_attempts": sorted(
            queue_snapshots,
            key=lambda row: (row["stage"], row["arm"], row["seed"], row["attempt"]),
        ),
        "run_attempts": sorted(
            (_run_snapshot(row) for row in run_rows),
            key=lambda row: row["run_id"],
        ),
    }


def _registered_summary_artifact(
    registry: Registry, *, run_id: str, summary_path: Path
) -> dict[str, Any]:
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT artifact_id, kind, path, sha256, size_bytes, status
            FROM artifacts
            WHERE run_id = ? AND path = ?
            ORDER BY artifact_id
            """,
            (run_id, summary_path.as_posix()),
        ).fetchall()
    if len(rows) != 1:
        raise PeakVramReconciliationError(
            f"{run_id} must register exactly one canonical summary.json artifact"
        )
    row = dict(rows[0])
    actual_size = summary_path.stat().st_size
    actual_sha256 = _sha256_file(summary_path)
    if (
        row.get("status") != "present"
        or row.get("sha256") != actual_sha256
        or _integer(row.get("size_bytes"), f"{run_id} summary artifact size")
        != actual_size
    ):
        raise PeakVramReconciliationError(
            f"{run_id} registered summary metadata disagrees with the file"
        )
    return {
        "artifact_id": int(row["artifact_id"]),
        "kind": str(row["kind"]),
        "path": summary_path.as_posix(),
        "sha256": actual_sha256,
        "size_bytes": actual_size,
        "status": "present",
    }


def _source_peak_values(
    *,
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    resource: Mapping[str, Any],
    run_id: str,
) -> dict[str, float]:
    summary_values: list[tuple[str, float]] = []
    for key in ("peak_vram_gib", "peak_vram_gb"):
        if key in summary and summary.get(key) is not None:
            summary_values.append(
                (key, _finite_nonnegative(summary[key], f"{run_id} summary {key}"))
            )
    if not summary_values:
        raise PeakVramReconciliationError(
            f"{run_id} summary lacks a peak-VRAM measurement"
        )
    if any(not _same(summary_values[0][1], value) for _, value in summary_values[1:]):
        raise PeakVramReconciliationError(
            f"{run_id} summary peak_vram_gib and peak_vram_gb disagree"
        )
    if "resource/peak_vram_gib" not in final_metrics:
        raise PeakVramReconciliationError(
            f"{run_id} final metrics lack resource/peak_vram_gib"
        )
    if "peak_allocated_vram_gib" not in resource:
        raise PeakVramReconciliationError(
            f"{run_id} resource diagnostic lacks peak_allocated_vram_gib"
        )
    values = {
        "summary": summary_values[0][1],
        "metrics_final": _finite_nonnegative(
            final_metrics["resource/peak_vram_gib"],
            f"{run_id} final metric peak VRAM",
        ),
        "resource_diagnostic": _finite_nonnegative(
            resource["peak_allocated_vram_gib"],
            f"{run_id} diagnostic peak VRAM",
        ),
    }
    reference = values["summary"]
    if any(not _same(reference, value) for value in values.values()):
        raise PeakVramReconciliationError(
            f"{run_id} summary/final/resource peak-VRAM values disagree"
        )
    return values


def _audit_bundle(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run: Mapping[str, Any],
    selected: Mapping[str, Any],
    materialization_checksum: str,
    bundle_verifier: Callable[..., Mapping[str, Any]],
) -> dict[str, Any]:
    run_id = str(selected["run_id"])
    artifact_value = run.get("artifact_path")
    if not isinstance(artifact_value, str) or not Path(artifact_value).is_absolute():
        raise PeakVramReconciliationError(
            f"{run_id} artifact path must be an absolute immutable bundle path"
        )
    root = Path(artifact_value).resolve(strict=False)
    expected_base = (paths.artifact_root / "runs").resolve(strict=False)
    try:
        relative = root.relative_to(expected_base)
    except ValueError as exc:
        raise PeakVramReconciliationError(
            f"{run_id} artifact path is outside the immutable run store"
        ) from exc
    if (
        len(relative.parts) != 3
        or relative.parts[-1] != run_id
        or len(relative.parts[0]) != 4
        or not relative.parts[0].isdigit()
        or len(relative.parts[1]) != 2
        or not relative.parts[1].isdigit()
    ):
        raise PeakVramReconciliationError(
            f"{run_id} artifact path does not match runs/YYYY/MM/<run_id>"
        )
    markers = [name for name in COMPLETION_MARKERS if (root / name).is_file()]
    if markers != ["_SUCCESS"]:
        raise PeakVramReconciliationError(
            f"{run_id} must contain one and only one _SUCCESS marker"
        )
    verification = bundle_verifier(root)
    if verification.get("valid") is not True or verification.get("status") != "success":
        raise PeakVramReconciliationError(f"{run_id} bundle verification failed")
    artifact_issues = registry.verify_artifacts(run_id=run_id)
    if artifact_issues:
        raise PeakVramReconciliationError(
            f"{run_id} registered artifact verification failed: "
            f"{artifact_issues[:3]}"
        )

    summary_path = root / "summary.json"
    final_path = root / "metrics/final.json"
    resource_path = root / "diagnostics/resource_usage.json"
    checksum_path = root / "provenance/artifact_checksums.json"
    marker_path = root / "_SUCCESS"
    summary = _strict_json(summary_path, label=f"{run_id} summary")
    final_metrics = _strict_json(final_path, label=f"{run_id} final metrics")
    resource = _strict_json(resource_path, label=f"{run_id} resource diagnostic")
    checksum_manifest = _strict_json(
        checksum_path, label=f"{run_id} artifact checksum manifest"
    )
    marker = _strict_json(marker_path, label=f"{run_id} success marker")
    files = _mapping(
        checksum_manifest.get("files"), f"{run_id} artifact checksum files"
    )
    if (
        marker.get("status") != "success"
        or marker.get("run_id") != run_id
        or marker.get("content_sha256") != canonical_sha256(files)
    ):
        raise PeakVramReconciliationError(
            f"{run_id} success marker does not bind the bundle content"
        )
    source_paths = {
        "summary": summary_path,
        "metrics_final": final_path,
        "resource_diagnostic": resource_path,
    }
    source_hashes: dict[str, dict[str, Any]] = {}
    for label, path in source_paths.items():
        relative_path = path.relative_to(root).as_posix()
        manifest_entry = _mapping(
            files.get(relative_path), f"{run_id} checksum entry {relative_path}"
        )
        actual_sha256 = _sha256_file(path)
        actual_size = path.stat().st_size
        if (
            manifest_entry.get("type") != "file"
            or manifest_entry.get("sha256") != actual_sha256
            or _integer(
                manifest_entry.get("size"),
                f"{run_id} checksum size {relative_path}",
            )
            != actual_size
        ):
            raise PeakVramReconciliationError(
                f"{run_id} checksum manifest disagrees for {relative_path}"
            )
        source_hashes[label] = {
            "path": path.as_posix(),
            "sha256": actual_sha256,
            "size_bytes": actual_size,
        }
    registered_summary = _registered_summary_artifact(
        registry, run_id=run_id, summary_path=summary_path
    )
    stage = str(selected["stage"])
    arm = str(selected["arm"])
    seed = int(selected["seed"])
    if (
        run.get("campaign_id") != CAMPAIGN_ID
        or run.get("status") != "completed"
        or summary.get("run_id") != run_id
        or summary.get("status") != "success"
        or summary.get("campaign_id") != CAMPAIGN_ID
        or summary.get("evaluation_protocol") != EXPECTED_PROTOCOL
        or summary.get("aliases") != list(ALIASES)
        or summary.get("public_variant") != arm
        or _integer(summary.get("model_seed"), f"{run_id} summary seed") != seed
        or summary.get("diagnostic_resource_pilot") is not (stage == "pilot")
        or _integer(summary.get("run_attempt"), f"{run_id} summary attempt")
        != int(selected["selected_attempt"])
        or summary.get("materialization_checksum") != materialization_checksum
        or resource.get("aliases") != list(ALIASES)
        or resource.get("public_variant") != arm
        or resource.get("diagnostic_resource_pilot") is not (stage == "pilot")
        or resource.get("schema")
        != "pooled_hybrid_count_resource_diagnostic_v1"
    ):
        raise PeakVramReconciliationError(
            f"{run_id} bundle identity does not match its registered pooled slot"
        )
    values = _source_peak_values(
        summary=summary,
        final_metrics=final_metrics,
        resource=resource,
        run_id=run_id,
    )
    proposed = values["summary"]
    current_raw = run.get("peak_vram_gb")
    if current_raw is None:
        current: float | None = None
        action = "backfill"
    else:
        current = _finite_nonnegative(
            current_raw, f"{run_id} registered peak_vram_gb"
        )
        if not _same(current, proposed):
            raise PeakVramReconciliationError(
                f"{run_id} has a conflicting non-null registry peak_vram_gb"
            )
        action = "already_consistent"
    return {
        **dict(selected),
        "artifact_path": root.as_posix(),
        "bundle_content_sha256": _digest(
            marker.get("content_sha256"), f"{run_id} bundle content checksum"
        ),
        "artifact_checksum_manifest": {
            "path": checksum_path.as_posix(),
            "sha256": _sha256_file(checksum_path),
            "size_bytes": checksum_path.stat().st_size,
        },
        "registered_summary_artifact": registered_summary,
        "source_files": source_hashes,
        "source_values_binary_gib": values,
        "registry_value_binary_gib": current,
        "proposed_value_binary_gib": proposed,
        "action": action,
    }


def build_reconciliation_plan(
    *,
    paths: ProjectPaths,
    database_path: Path,
    materialization_path: Path,
    pilot_enqueue_path: Path,
    pilot_gate_path: Path,
    production_enqueue_path: Path,
    created_at: str | None = None,
    registry_factory: Callable[..., Registry] = Registry,
    bundle_verifier: Callable[..., Mapping[str, Any]] = verify_run_bundle,
) -> dict[str, Any]:
    """Build the checksum-bound plan without changing registry run metadata."""

    registry = registry_factory(database_path, initialize=False)
    _assert_campaign_inactive(registry)
    receipts = _validate_receipts(
        paths=paths,
        materialization_path=materialization_path,
        pilot_enqueue_path=pilot_enqueue_path,
        pilot_gate_path=pilot_gate_path,
        production_enqueue_path=production_enqueue_path,
    )
    campaign = registry.get_campaign(CAMPAIGN_ID)
    if campaign is None:
        raise PeakVramReconciliationError(
            "pooled campaign is absent from the authoritative registry"
        )
    campaign_config = _mapping(campaign.get("config"), "registered campaign config")
    if (
        campaign_config.get("frozen_contract_sha256")
        != EXPECTED_CONTRACT_SHA256
        or campaign_config.get("materialization_checksum")
        != receipts["materialization"].get("checksum")
        or campaign_config.get("production_run_count") != 14
        or campaign_config.get("production_seeds") != list(SEEDS)
    ):
        raise PeakVramReconciliationError(
            "registered campaign record is not bound to the frozen receipts"
        )
    queue_rows, run_rows = _campaign_inventory(registry)
    lineage = _resolve_lineages(
        queue_rows=queue_rows,
        run_rows=run_rows,
        receipts=receipts,
    )
    run_by_id = {str(row["run_id"]): row for row in run_rows}
    items = [
        _audit_bundle(
            registry=registry,
            paths=paths,
            run=run_by_id[str(selected["run_id"])],
            selected=selected,
            materialization_checksum=str(receipts["materialization"]["checksum"]),
            bundle_verifier=bundle_verifier,
        )
        for selected in lineage["selected"]
    ]
    if len(items) != 16:
        raise PeakVramReconciliationError(
            "bundle audit did not resolve all sixteen completed runs"
        )
    # Detect registry changes that raced the multi-file read-only audit.
    after_queue, after_runs = _campaign_inventory(registry)
    if (
        [_queue_snapshot(row) for row in after_queue]
        != [_queue_snapshot(row) for row in queue_rows]
        or [_run_snapshot(row) for row in after_runs]
        != [_run_snapshot(row) for row in run_rows]
    ):
        raise PeakVramReconciliationError(
            "registry changed while the read-only reconciliation plan was built"
        )
    actions = {
        "backfill": sum(item["action"] == "backfill" for item in items),
        "already_consistent": sum(
            item["action"] == "already_consistent" for item in items
        ),
        "conflict": 0,
    }
    payload = {
        "schema_version": 1,
        "plan_kind": PLAN_KIND,
        "campaign_id": CAMPAIGN_ID,
        "created_at": created_at or utc_now(),
        "mode": "read_only_plan",
        "database_path": database_path.resolve(strict=False).as_posix(),
        "frozen_contract": {
            "path": receipts["frozen_contract_path"],
            "sha256": EXPECTED_CONTRACT_SHA256,
        },
        "registry_campaign_record_sha256": canonical_sha256(campaign),
        "receipts": receipts["receipt_evidence"],
        "expected_completed_runs": {"pilot": 2, "production": 14, "total": 16},
        "lineage": {
            "queue_attempts": lineage["queue_attempts"],
            "run_attempts": lineage["run_attempts"],
        },
        "items": items,
        "actions": actions,
        "registry_mutation_performed": False,
        "run_bundles_modified": False,
        "active_or_incomplete_runs_touched": False,
    }
    return _signed(payload)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create one JSON file and refuse overwrite."""

    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"append-only output already exists: {path}")
    serialized = json.dumps(
        dict(payload), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_reconciliation_plan(path: Path, plan: Mapping[str, Any]) -> None:
    checksum = _digest(plan.get("checksum"), "plan checksum")
    unsigned = dict(plan)
    unsigned.pop("checksum", None)
    if canonical_sha256(unsigned) != checksum:
        raise PeakVramReconciliationError("refusing to write an invalid plan")
    _write_new_json(path, plan)


def _online_backup(*, database_path: Path, backup_path: Path) -> str:
    if not database_path.is_file():
        raise PeakVramReconciliationError(
            f"registry database is missing: {database_path}"
        )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists() or backup_path.is_symlink():
        raise FileExistsError(f"database backup already exists: {backup_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{backup_path.name}.", suffix=".tmp", dir=backup_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source_uri = f"file:{database_path.resolve(strict=False).as_posix()}?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source:
            with sqlite3.connect(temporary) as destination:
                source.backup(destination)
                destination.execute("PRAGMA wal_checkpoint")
        os.link(temporary, backup_path)
    except BaseException:
        backup_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(backup_path)


def _snapshot_from_database(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queue: list[dict[str, Any]] = []
    for row in connection.execute(
            """
            SELECT job_id, status, attempt_count, maximum_attempts,
                   requested_gpu, retry_of, run_id, canonical_config_json
            FROM queue_jobs
            WHERE campaign_id = ?
            ORDER BY created_at, job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall():
        config = _loads_strict(
            row["canonical_config_json"],
            label=f"queue config {row['job_id']}",
        )
        trainer = _mapping(
            config.get("trainer"), f"queue config {row['job_id']} trainer"
        )
        experiment = _mapping(
            config.get("experiment"), f"queue config {row['job_id']} experiment"
        )
        stage = (
            "pilot"
            if trainer.get("diagnostic_resource_pilot") is True
            else "production"
        )
        queue.append(
            {
                "stage": stage,
                "arm": str(experiment.get("arm", "")),
                "seed": _integer(
                    config.get("seed"), f"queue config {row['job_id']} seed"
                ),
                "job_id": str(row["job_id"]),
                "status": str(row["status"]),
                "attempt": int(row["attempt_count"]),
                "maximum_attempts": int(row["maximum_attempts"]),
                "requested_gpu": str(row["requested_gpu"]),
                "retry_of": (
                    None if row["retry_of"] is None else str(row["retry_of"])
                ),
                "run_id": None if row["run_id"] is None else str(row["run_id"]),
                "config_sha256": canonical_sha256(config),
            }
        )
    runs = []
    for row in connection.execute(
        """
        SELECT run_id, status, attempt, retry_of, artifact_path, peak_vram_gb,
               config_json
        FROM runs
        WHERE campaign_id = ?
        ORDER BY created_at, run_id
        """,
        (CAMPAIGN_ID,),
    ).fetchall():
        value = row["peak_vram_gb"]
        runs.append(
            {
                "run_id": str(row["run_id"]),
                "status": str(row["status"]),
                "attempt": int(row["attempt"]),
                "retry_of": (
                    None if row["retry_of"] is None else str(row["retry_of"])
                ),
                "artifact_path": (
                    None
                    if row["artifact_path"] is None
                    else str(row["artifact_path"])
                ),
                "registry_peak_vram_gb": (
                    None
                    if value is None
                    else _finite_nonnegative(value, "registered peak VRAM")
                ),
                "normalized_config_sha256": _normalized_config_sha256(
                    _loads_strict(
                        row["config_json"],
                        label=f"run config {row['run_id']}",
                    )
                ),
            }
        )
    queue.sort(
        key=lambda row: (row["stage"], row["arm"], row["seed"], row["attempt"])
    )
    runs.sort(key=lambda row: row["run_id"])
    return queue, runs


def _transactional_backfill(
    *,
    registry: Registry,
    plan: Mapping[str, Any],
    applied_at: str,
) -> list[dict[str, Any]]:
    lineage = _mapping(plan.get("lineage"), "reviewed plan lineage")
    expected_queue = _sequence(
        lineage.get("queue_attempts"), "reviewed queue attempt snapshot"
    )
    expected_runs = _sequence(
        lineage.get("run_attempts"), "reviewed run attempt snapshot"
    )
    items = _sequence(plan.get("items"), "reviewed plan items")
    changes: list[dict[str, Any]] = []
    with registry.transaction(immediate=True) as connection:
        queue_snapshot, run_snapshot = _snapshot_from_database(connection)
        if queue_snapshot != expected_queue or run_snapshot != expected_runs:
            raise PeakVramReconciliationError(
                "registry lineage/current values changed before BEGIN IMMEDIATE apply"
            )
        if any(row["status"] in _ACTIVE_QUEUE_STATUSES for row in queue_snapshot):
            raise PeakVramReconciliationError(
                "campaign became active before reconciliation apply"
            )
        if any(row["status"] in _INCOMPLETE_RUN_STATUSES for row in run_snapshot):
            raise PeakVramReconciliationError(
                "campaign gained an incomplete run before reconciliation apply"
            )
        for raw in items:
            item = _mapping(raw, "reviewed plan item")
            action = item.get("action")
            run_id = str(item.get("run_id", ""))
            proposed = _finite_nonnegative(
                item.get("proposed_value_binary_gib"),
                f"{run_id} proposed peak VRAM",
            )
            if action == "already_consistent":
                row = connection.execute(
                    """
                    SELECT peak_vram_gb FROM runs
                    WHERE run_id = ? AND campaign_id = ? AND status = 'completed'
                    """,
                    (run_id, CAMPAIGN_ID),
                ).fetchone()
                if (
                    row is None
                    or row["peak_vram_gb"] is None
                    or not _same(float(row["peak_vram_gb"]), proposed)
                ):
                    raise PeakVramReconciliationError(
                        f"already-consistent registry value changed for {run_id}"
                    )
                continue
            if action != "backfill" or item.get("registry_value_binary_gib") is not None:
                raise PeakVramReconciliationError(
                    f"reviewed plan contains an invalid action for {run_id}"
                )
            cursor = connection.execute(
                """
                UPDATE runs
                SET peak_vram_gb = ?, updated_at = ?
                WHERE run_id = ?
                  AND campaign_id = ?
                  AND status = 'completed'
                  AND peak_vram_gb IS NULL
                """,
                (proposed, applied_at, run_id, CAMPAIGN_ID),
            )
            if cursor.rowcount != 1:
                raise PeakVramReconciliationError(
                    f"conditional backfill changed {cursor.rowcount} rows for {run_id}"
                )
            changes.append(
                {
                    "run_id": run_id,
                    "old_value_binary_gib": None,
                    "new_value_binary_gib": proposed,
                }
            )
    return changes


def _run_post_apply_verification(
    *, paths: ProjectPaths, database_path: Path
) -> dict[str, Any]:
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    source_path = (paths.project_root / "src").as_posix()
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else source_path + os.pathsep + existing_pythonpath
    )
    commands = [
        [
            sys.executable,
            "-m",
            "spatial_benchmark",
            "--database",
            database_path.as_posix(),
            "verify-artifacts",
        ],
        [
            sys.executable,
            "-m",
            "spatial_benchmark",
            "--database",
            database_path.as_posix(),
            "doctor",
        ],
    ]
    records: list[dict[str, Any]] = []
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=paths.project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        record = {
            "command": command,
            "working_directory": paths.project_root.as_posix(),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        records.append(record)
        if completed.returncode != 0:
            raise PeakVramReconciliationError(
                f"post-apply verification failed: {' '.join(command)}"
            )
        try:
            result = json.loads(completed.stdout)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PeakVramReconciliationError(
                f"post-apply command did not emit JSON: {' '.join(command)}"
            ) from exc
        if command[-1] == "verify-artifacts" and result.get("valid") is not True:
            raise PeakVramReconciliationError(
                "post-apply artifact verification did not pass"
            )
        if command[-1] == "doctor" and result.get("ok") is not True:
            raise PeakVramReconciliationError(
                "post-apply project doctor did not pass"
            )
    payload = {"commands": records, "all_passed": True}
    return {**payload, "checksum": canonical_sha256(payload)}


@contextmanager
def _application_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _reconciliation_report_root(paths: ProjectPaths) -> Path:
    return (
        paths.report_root
        / "analyses"
        / "adjacent_normal_10core_pooled_hybrid_ensemble"
        / "registry_reconciliation"
    ).resolve(strict=False)


def _require_unapplied_plan(
    *,
    paths: ProjectPaths,
    receipt_output_path: Path,
    plan_checksum: str,
) -> None:
    report_root = _reconciliation_report_root(paths)
    resolved_output = receipt_output_path.resolve(strict=False)
    try:
        resolved_output.relative_to(report_root)
    except ValueError as exc:
        raise PeakVramReconciliationError(
            "application receipt must remain in the campaign report "
            "registry_reconciliation directory"
        ) from exc
    if not report_root.exists():
        return
    for candidate in sorted(report_root.rglob("*.json")):
        payload = _strict_json(
            candidate, label=f"existing reconciliation report {candidate.name}"
        )
        if payload.get("application_kind") != APPLICATION_KIND:
            continue
        existing = _verified_signed_json(
            candidate, label=f"existing application receipt {candidate.name}"
        )
        reviewed = _mapping(
            existing.get("reviewed_plan"),
            f"existing application receipt {candidate.name} reviewed plan",
        )
        if hmac.compare_digest(str(reviewed.get("checksum", "")), plan_checksum):
            raise PeakVramReconciliationError(
                f"reviewed plan {plan_checksum} has already been applied"
            )


def apply_reconciliation_plan(
    *,
    paths: ProjectPaths,
    database_path: Path,
    plan_path: Path,
    expected_plan_checksum: str,
    receipt_output_path: Path,
    materialization_path: Path,
    pilot_enqueue_path: Path,
    pilot_gate_path: Path,
    production_enqueue_path: Path,
    registry_factory: Callable[..., Registry] = Registry,
    bundle_verifier: Callable[..., Mapping[str, Any]] = verify_run_bundle,
    post_apply_verifier: Callable[..., Mapping[str, Any]] = (
        _run_post_apply_verification
    ),
    backup_path: Path | None = None,
) -> dict[str, Any]:
    """Apply one reviewed plan using a backup and one conditional transaction."""

    expected = _digest(expected_plan_checksum, "expected reviewed plan checksum")
    reviewed = _verified_signed_json(plan_path, label="reviewed reconciliation plan")
    if (
        reviewed.get("schema_version") != 1
        or reviewed.get("plan_kind") != PLAN_KIND
        or reviewed.get("campaign_id") != CAMPAIGN_ID
        or reviewed.get("mode") != "read_only_plan"
        or reviewed.get("registry_mutation_performed") is not False
        or reviewed.get("run_bundles_modified") is not False
        or not hmac.compare_digest(str(reviewed.get("checksum")), expected)
    ):
        raise PeakVramReconciliationError(
            "reviewed plan identity or expected checksum is invalid"
        )
    if receipt_output_path.exists() or receipt_output_path.is_symlink():
        raise FileExistsError(
            f"append-only application receipt exists: {receipt_output_path}"
        )
    lock_path = (
        paths.state_root
        / "tracking"
        / f".{CAMPAIGN_ID}.peak_vram_reconciliation.lock"
    )
    with _application_lock(lock_path):
        _require_unapplied_plan(
            paths=paths,
            receipt_output_path=receipt_output_path,
            plan_checksum=expected,
        )
        current = build_reconciliation_plan(
            paths=paths,
            database_path=database_path,
            materialization_path=materialization_path,
            pilot_enqueue_path=pilot_enqueue_path,
            pilot_gate_path=pilot_gate_path,
            production_enqueue_path=production_enqueue_path,
            created_at=str(reviewed.get("created_at")),
            registry_factory=registry_factory,
            bundle_verifier=bundle_verifier,
        )
        if current != reviewed or not hmac.compare_digest(
            str(current.get("checksum")), expected
        ):
            raise PeakVramReconciliationError(
                "current evidence no longer matches the exact reviewed plan"
            )
        applied_at = utc_now()
        if backup_path is None:
            stamp = (
                applied_at.replace("-", "")
                .replace(":", "")
                .replace(".", "")
                .replace("Z", "Z")
            )
            backup_path = (
                paths.state_root
                / "tracking"
                / "backups"
                / (
                    f"bagm.before_pooled_peak_vram_reconciliation."
                    f"{stamp}.{expected[:12]}.sqlite3"
                )
            )
        backup_path = backup_path.resolve(strict=False)
        required_backup_root = (
            paths.state_root / "tracking" / "backups"
        ).resolve(strict=False)
        try:
            backup_path.relative_to(required_backup_root)
        except ValueError as exc:
            raise PeakVramReconciliationError(
                "database backup must remain under state/tracking/backups"
            ) from exc
        backup_sha256 = _online_backup(
            database_path=database_path, backup_path=backup_path
        )
        registry = registry_factory(database_path, initialize=False)
        changes = _transactional_backfill(
            registry=registry, plan=reviewed, applied_at=applied_at
        )
        post_plan = build_reconciliation_plan(
            paths=paths,
            database_path=database_path,
            materialization_path=materialization_path,
            pilot_enqueue_path=pilot_enqueue_path,
            pilot_gate_path=pilot_gate_path,
            production_enqueue_path=production_enqueue_path,
            registry_factory=registry_factory,
            bundle_verifier=bundle_verifier,
        )
        if (
            post_plan.get("actions")
            != {"backfill": 0, "already_consistent": 16, "conflict": 0}
            or any(
                _mapping(item, "post-apply plan item").get("action")
                != "already_consistent"
                for item in _sequence(post_plan.get("items"), "post-apply items")
            )
        ):
            raise PeakVramReconciliationError(
                "post-apply plan is not fully registry-consistent"
            )
        post_verification = dict(
            post_apply_verifier(paths=paths, database_path=database_path)
        )
        post_verification_checksum = canonical_sha256(post_verification)
        receipt = _signed(
            {
                "schema_version": 1,
                "application_kind": APPLICATION_KIND,
                "campaign_id": CAMPAIGN_ID,
                "applied_at": applied_at,
                "reviewed_plan": {
                    "path": plan_path.resolve(strict=False).as_posix(),
                    "file_sha256": _sha256_file(plan_path),
                    "checksum": expected,
                },
                "database_backup": {
                    "path": backup_path.as_posix(),
                    "sha256": backup_sha256,
                    "size_bytes": backup_path.stat().st_size,
                },
                "transaction": {
                    "mode": "BEGIN IMMEDIATE",
                    "conditional_null_only_updates": True,
                    "changed_run_ids": [row["run_id"] for row in changes],
                    "changes": changes,
                    "changed_count": len(changes),
                },
                "post_reconciliation_plan_checksum": post_plan["checksum"],
                "post_verification": post_verification,
                "post_verification_checksum": post_verification_checksum,
                "run_bundles_modified": False,
                "active_or_incomplete_runs_touched": False,
            }
        )
        _write_new_json(receipt_output_path, receipt)
        return receipt


def _default_inputs(paths: ProjectPaths) -> dict[str, Path]:
    locked = (
        paths.scratch_root
        / "locked_campaigns"
        / CAMPAIGN_ID
    )
    return {
        "database": paths.state_root / "tracking" / "bagm.sqlite3",
        "materialization": locked / "locked_config_materialization.json",
        "pilot_enqueue": locked / "pilot_enqueue_receipt.json",
        "pilot_gate": locked / "pilot_gate_receipt.json",
        "production_enqueue": locked / "production_enqueue_receipt.json",
    }


def _timestamp_slug() -> str:
    return (
        utc_now()
        .replace("-", "")
        .replace(":", "")
        .replace(".", "")
        .replace("Z", "Z")
    )


def _parser(paths: ProjectPaths) -> argparse.ArgumentParser:
    defaults = _default_inputs(paths)
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only pooled peak-VRAM reconciliation plan by default; "
            "apply only an explicitly checksum-approved plan."
        )
    )
    parser.add_argument("--database", type=Path, default=defaults["database"])
    parser.add_argument(
        "--materialization", type=Path, default=defaults["materialization"]
    )
    parser.add_argument(
        "--pilot-enqueue", type=Path, default=defaults["pilot_enqueue"]
    )
    parser.add_argument("--pilot-gate", type=Path, default=defaults["pilot_gate"])
    parser.add_argument(
        "--production-enqueue", type=Path, default=defaults["production_enqueue"]
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help=(
            "Append-only plan path. Defaults to a timestamped campaign report "
            "path in planning mode."
        ),
    )
    parser.add_argument(
        "--apply-plan",
        type=Path,
        help="Explicitly apply this existing reviewed plan instead of planning.",
    )
    parser.add_argument(
        "--expected-plan-checksum",
        help="Required exact canonical checksum when --apply-plan is supplied.",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        help=(
            "Append-only application receipt. Defaults to a timestamped campaign "
            "report path in application mode."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    paths = current_paths()
    parser = _parser(paths)
    arguments = parser.parse_args(argv)
    report_dir = _reconciliation_report_root(paths)
    try:
        if arguments.apply_plan is None:
            if arguments.expected_plan_checksum is not None:
                parser.error(
                    "--expected-plan-checksum is valid only with --apply-plan"
                )
            if arguments.receipt_output is not None:
                parser.error("--receipt-output is valid only with --apply-plan")
            output = arguments.plan_output or (
                report_dir / f"peak_vram_plan_{_timestamp_slug()}.json"
            )
            plan = build_reconciliation_plan(
                paths=paths,
                database_path=arguments.database,
                materialization_path=arguments.materialization,
                pilot_enqueue_path=arguments.pilot_enqueue,
                pilot_gate_path=arguments.pilot_gate,
                production_enqueue_path=arguments.production_enqueue,
            )
            write_reconciliation_plan(output, plan)
            print(
                json.dumps(
                    {
                        "mode": "read_only_plan",
                        "output": output.resolve(strict=False).as_posix(),
                        "checksum": plan["checksum"],
                        "actions": plan["actions"],
                        "registry_mutation_performed": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if arguments.plan_output is not None:
            parser.error("--plan-output is not valid with --apply-plan")
        if arguments.expected_plan_checksum is None:
            parser.error("--apply-plan requires --expected-plan-checksum")
        receipt_output = arguments.receipt_output or (
            report_dir / f"peak_vram_application_{_timestamp_slug()}.json"
        )
        receipt = apply_reconciliation_plan(
            paths=paths,
            database_path=arguments.database,
            plan_path=arguments.apply_plan,
            expected_plan_checksum=arguments.expected_plan_checksum,
            receipt_output_path=receipt_output,
            materialization_path=arguments.materialization,
            pilot_enqueue_path=arguments.pilot_enqueue,
            pilot_gate_path=arguments.pilot_gate,
            production_enqueue_path=arguments.production_enqueue,
        )
        print(
            json.dumps(
                {
                    "mode": "applied",
                    "output": receipt_output.resolve(strict=False).as_posix(),
                    "checksum": receipt["checksum"],
                    "changed_count": receipt["transaction"]["changed_count"],
                    "backup": receipt["database_backup"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        FileExistsError,
        PeakVramReconciliationError,
        sqlite3.Error,
    ) as exc:
        print(f"peak-VRAM reconciliation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
