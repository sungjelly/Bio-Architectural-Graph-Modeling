"""Registry-aware retention for large, reproducible experiment payloads.

This module deliberately does not delete runs, metrics, provenance, reports, or
registry rows.  It records an immutable plan, verifies every selected payload,
marks the corresponding rows pending, unlinks only those exact files, and then
records an explicit tombstone status while retaining path/checksum metadata.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable

from .paths import ProjectPaths, current_paths
from .registry import (
    ARTIFACT_STATUS_DELETED_BY_RETENTION,
    ARTIFACT_STATUS_RETENTION_PENDING,
    Registry,
)


DELETED_BY_RETENTION = ARTIFACT_STATUS_DELETED_BY_RETENTION
RETENTION_DELETION_PENDING = ARTIFACT_STATUS_RETENTION_PENDING
DEFAULT_CHECKPOINT_MIN_BYTES = 128 * 1024 * 1024
DEFAULT_PREDICTION_MIN_BYTES = 8 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_LIVE_QUEUE_STATES = ("queued", "claimed", "running")
_LIVE_RUN_STATES = ("pending", "running", "finalizing")


class ArtifactRetentionError(RuntimeError):
    """Raised when retention cannot proceed without weakening an audit."""


@dataclass(frozen=True, slots=True)
class RetentionCandidate:
    """One exact registered payload authorized by a retention decision."""

    artifact_id: int
    run_id: str
    campaign_id: str
    run_status: str
    lifecycle_stage: str
    retention_class: str
    kind: str
    path: str
    size_bytes: int
    sha256: str
    artifact_root: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_decision_id(value: str) -> str:
    decision_id = str(value).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,79}", decision_id):
        raise ValueError(
            "decision_id must be 3-80 lowercase letters, digits, underscores, "
            "or hyphens"
        )
    return decision_id


def select_retention_candidates(
    registry: Registry,
    *,
    checkpoint_min_bytes: int = DEFAULT_CHECKPOINT_MIN_BYTES,
    prediction_min_bytes: int = DEFAULT_PREDICTION_MIN_BYTES,
) -> list[RetentionCandidate]:
    """Select only large modern diagnostic/exploratory payloads.

    Failed-run checkpoints and predictions are selected regardless of size.
    Completed-run payloads use the explicit size thresholds.  Legacy,
    promoted, confirmation, and locked-final artifacts are excluded by query.
    """

    if checkpoint_min_bytes < 1 or prediction_min_bytes < 1:
        raise ValueError("retention size thresholds must be positive")
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT a.artifact_id, a.run_id, a.kind, a.path, a.size_bytes,
                   a.sha256, r.campaign_id, r.status AS run_status,
                   r.artifact_path AS artifact_root, rc.lifecycle_stage,
                   rc.retention_class
            FROM artifacts a
            JOIN runs r ON r.run_id = a.run_id
            JOIN run_categories rc ON rc.run_id = r.run_id
            WHERE a.status = 'present'
              AND a.kind IN ('checkpoints', 'predictions')
              AND r.run_id GLOB 'r_*'
              AND r.promoted_at IS NULL
              AND rc.lifecycle_stage IN ('diagnostic', 'exploratory_screen')
              AND (
                    r.status = 'failed'
                    OR (
                        r.status = 'completed'
                        AND (
                            (a.kind = 'checkpoints' AND a.size_bytes >= ?)
                            OR
                            (a.kind = 'predictions' AND a.size_bytes >= ?)
                        )
                    )
              )
            ORDER BY a.artifact_id
            """,
            (int(checkpoint_min_bytes), int(prediction_min_bytes)),
        ).fetchall()
    candidates: list[RetentionCandidate] = []
    for row in rows:
        sha256 = str(row["sha256"] or "").lower()
        if not _SHA256.fullmatch(sha256):
            raise ArtifactRetentionError(
                f"Selected artifact {row['artifact_id']} has no valid SHA-256"
            )
        if row["size_bytes"] is None or int(row["size_bytes"]) < 0:
            raise ArtifactRetentionError(
                f"Selected artifact {row['artifact_id']} has no valid size"
            )
        if not row["artifact_root"]:
            raise ArtifactRetentionError(
                f"Selected run {row['run_id']} has no artifact root"
            )
        candidates.append(
            RetentionCandidate(
                artifact_id=int(row["artifact_id"]),
                run_id=str(row["run_id"]),
                campaign_id=str(row["campaign_id"]),
                run_status=str(row["run_status"]),
                lifecycle_stage=str(row["lifecycle_stage"]),
                retention_class=str(row["retention_class"]),
                kind=str(row["kind"]),
                path=str(row["path"]),
                size_bytes=int(row["size_bytes"]),
                sha256=sha256,
                artifact_root=str(row["artifact_root"]),
            )
        )
    return candidates


def assert_no_live_experiments(registry: Registry) -> None:
    """Fail closed if queue or run state permits an active writer."""

    queue_placeholders = ",".join("?" for _ in _LIVE_QUEUE_STATES)
    run_placeholders = ",".join("?" for _ in _LIVE_RUN_STATES)
    with registry.connect() as connection:
        queue_rows = connection.execute(
            f"""
            SELECT job_id, status, run_id FROM queue_jobs
            WHERE status IN ({queue_placeholders})
            ORDER BY job_id
            """,
            _LIVE_QUEUE_STATES,
        ).fetchall()
        run_rows = connection.execute(
            f"""
            SELECT run_id, status FROM runs
            WHERE status IN ({run_placeholders})
            ORDER BY run_id
            """,
            _LIVE_RUN_STATES,
        ).fetchall()
    if queue_rows or run_rows:
        raise ArtifactRetentionError(
            "Refusing retention while an experiment may be active: "
            f"queue={[(row['job_id'], row['status']) for row in queue_rows]}, "
            f"runs={[(row['run_id'], row['status']) for row in run_rows]}"
        )


def _resolve_candidate_paths(
    candidate: RetentionCandidate,
    *,
    paths: ProjectPaths,
) -> tuple[Path, Path]:
    payload = Path(candidate.path)
    run_root = Path(candidate.artifact_root)
    if not payload.is_absolute():
        payload = paths.project_root / payload
    if not run_root.is_absolute():
        run_root = paths.project_root / run_root
    if payload.is_symlink() or run_root.is_symlink():
        raise ArtifactRetentionError(
            f"Retention target or run root may not be a symlink: {payload}"
        )
    modern_root = (paths.artifact_root / "runs").resolve(strict=True)
    resolved_run_root = run_root.resolve(strict=True)
    resolved_payload = payload.resolve(strict=True)
    if not resolved_run_root.is_relative_to(modern_root):
        raise ArtifactRetentionError(
            f"Run root is outside the modern artifact root: {resolved_run_root}"
        )
    if not resolved_payload.is_relative_to(resolved_run_root):
        raise ArtifactRetentionError(
            f"Payload is outside its registered run root: {resolved_payload}"
        )
    marker = {"completed": "_SUCCESS", "failed": "_FAILED"}.get(
        candidate.run_status
    )
    if marker is None or not (resolved_run_root / marker).is_file():
        raise ArtifactRetentionError(
            f"Run {candidate.run_id} lacks its expected terminal marker {marker}"
        )
    if not resolved_payload.is_file():
        raise ArtifactRetentionError(f"Retention target is not a file: {payload}")
    return resolved_payload, resolved_run_root


def verify_candidates(
    candidates: Iterable[RetentionCandidate],
    *,
    paths: ProjectPaths,
    verify_sha256: bool,
) -> list[Path]:
    """Validate exact path, size, and optionally SHA-256 for every target."""

    verified: list[Path] = []
    seen_paths: set[Path] = set()
    for index, candidate in enumerate(candidates, start=1):
        payload, _ = _resolve_candidate_paths(candidate, paths=paths)
        if payload in seen_paths:
            raise ArtifactRetentionError(f"Duplicate retention target: {payload}")
        seen_paths.add(payload)
        observed_size = payload.stat().st_size
        if observed_size != candidate.size_bytes:
            raise ArtifactRetentionError(
                f"Size mismatch for {payload}: registered={candidate.size_bytes}, "
                f"observed={observed_size}"
            )
        if verify_sha256:
            observed_sha = _sha256_file(payload)
            if observed_sha != candidate.sha256:
                raise ArtifactRetentionError(
                    f"SHA-256 mismatch for {payload}: "
                    f"registered={candidate.sha256}, observed={observed_sha}"
                )
            print(
                f"verified {index} payload(s), latest={candidate.run_id}/{payload.name}",
                flush=True,
            )
        verified.append(payload)
    return verified


def _plan_bytes(candidates: Iterable[RetentionCandidate]) -> bytes:
    lines = [_canonical_json(candidate.record()) for candidate in candidates]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise ArtifactRetentionError(
                f"Refusing to overwrite a different retention record: {path}"
            )
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_retention_plan(
    candidates: list[RetentionCandidate],
    *,
    decision_id: str,
    output_dir: Path,
    checkpoint_min_bytes: int,
    prediction_min_bytes: int,
) -> dict[str, Any]:
    """Write deterministic plan, summary, and checksum without overwriting."""

    decision_id = _safe_decision_id(decision_id)
    output_dir = output_dir.resolve(strict=False)
    plan_path = output_dir / "deletion_plan.jsonl"
    payload = _plan_bytes(candidates)
    digest = hashlib.sha256(payload).hexdigest()
    _write_immutable(plan_path, payload)
    _write_immutable(
        output_dir / "deletion_plan.sha256",
        f"{digest}  deletion_plan.jsonl\n".encode("utf-8"),
    )
    by_kind = Counter(candidate.kind for candidate in candidates)
    by_status = Counter(candidate.run_status for candidate in candidates)
    by_campaign = Counter(candidate.campaign_id for candidate in candidates)
    summary = {
        "schema_version": 1,
        "decision_id": decision_id,
        "selection": {
            "checkpoint_min_bytes": int(checkpoint_min_bytes),
            "prediction_min_bytes": int(prediction_min_bytes),
            "modern_run_prefix": "r_",
            "lifecycle_stages": ["diagnostic", "exploratory_screen"],
            "promoted_runs_excluded": True,
            "failed_checkpoint_and_prediction_size_floor": 0,
        },
        "artifact_count": len(candidates),
        "run_count": len({candidate.run_id for candidate in candidates}),
        "total_bytes": sum(candidate.size_bytes for candidate in candidates),
        "plan_sha256": digest,
        "by_kind": dict(sorted(by_kind.items())),
        "by_run_status": dict(sorted(by_status.items())),
        "by_campaign": dict(sorted(by_campaign.items())),
    }
    summary_bytes = (_canonical_json(summary) + "\n").encode("utf-8")
    _write_immutable(output_dir / "deletion_plan_summary.json", summary_bytes)
    return summary


def _sqlite_backup(registry: Registry, *, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ArtifactRetentionError(f"Registry backup already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ArtifactRetentionError(f"Stale registry backup staging file: {temporary}")
    source = sqlite3.connect(registry.path)
    destination = sqlite3.connect(temporary)
    try:
        source.backup(destination)
        result = destination.execute("PRAGMA integrity_check").fetchall()
        if [str(row[0]) for row in result] != ["ok"]:
            raise ArtifactRetentionError(
                f"Registry backup integrity check failed: {result}"
            )
        destination.commit()
    finally:
        destination.close()
        source.close()
    os.replace(temporary, path)


def _assert_registry_plan_current(
    registry: Registry, candidates: list[RetentionCandidate]
) -> None:
    if not candidates:
        raise ArtifactRetentionError("Retention plan is empty")
    identifiers = [candidate.artifact_id for candidate in candidates]
    placeholders = ",".join("?" for _ in identifiers)
    with registry.connect() as connection:
        rows = connection.execute(
            f"""
            SELECT artifact_id, run_id, kind, path, size_bytes, sha256, status
            FROM artifacts WHERE artifact_id IN ({placeholders})
            ORDER BY artifact_id
            """,
            identifiers,
        ).fetchall()
    if len(rows) != len(candidates):
        raise ArtifactRetentionError("Registry artifact set changed after planning")
    for candidate, row in zip(candidates, rows, strict=True):
        observed = {
            "artifact_id": int(row["artifact_id"]),
            "run_id": str(row["run_id"]),
            "kind": str(row["kind"]),
            "path": str(row["path"]),
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"] or "").lower(),
            "status": str(row["status"]),
        }
        expected = {
            "artifact_id": candidate.artifact_id,
            "run_id": candidate.run_id,
            "kind": candidate.kind,
            "path": candidate.path,
            "size_bytes": candidate.size_bytes,
            "sha256": candidate.sha256,
            "status": "present",
        }
        if observed != expected:
            raise ArtifactRetentionError(
                f"Registry row changed after planning for artifact "
                f"{candidate.artifact_id}: expected={expected}, observed={observed}"
            )


def _mark_pending(registry: Registry, candidates: list[RetentionCandidate]) -> None:
    identifiers = [candidate.artifact_id for candidate in candidates]
    with registry.transaction(immediate=True) as connection:
        for artifact_id in identifiers:
            cursor = connection.execute(
                """
                UPDATE artifacts SET status = ?
                WHERE artifact_id = ? AND status = 'present'
                """,
                (RETENTION_DELETION_PENDING, artifact_id),
            )
            if cursor.rowcount != 1:
                raise ArtifactRetentionError(
                    f"Artifact {artifact_id} was not present when marking pending"
                )
        connection.executemany(
            """
            UPDATE checkpoint_catalog SET verification_status = ?, updated_at = ?
            WHERE artifact_id = ?
            """,
            [
                (RETENTION_DELETION_PENDING, _utc_now(), artifact_id)
                for artifact_id in identifiers
            ],
        )


def _mark_deleted(
    registry: Registry,
    candidates: list[RetentionCandidate],
    payloads: list[Path],
) -> None:
    with registry.transaction(immediate=True) as connection:
        for candidate, payload in zip(candidates, payloads, strict=True):
            if payload.exists() or payload.is_symlink():
                raise ArtifactRetentionError(
                    f"Cannot finalize retention while payload exists: {payload}"
                )
            cursor = connection.execute(
                """
                UPDATE artifacts SET status = ?
                WHERE artifact_id = ? AND status = ?
                """,
                (
                    DELETED_BY_RETENTION,
                    candidate.artifact_id,
                    RETENTION_DELETION_PENDING,
                ),
            )
            if cursor.rowcount != 1:
                raise ArtifactRetentionError(
                    f"Artifact {candidate.artifact_id} is not pending deletion"
                )
        connection.executemany(
            """
            UPDATE checkpoint_catalog SET verification_status = ?, updated_at = ?
            WHERE artifact_id = ?
            """,
            [
                (DELETED_BY_RETENTION, _utc_now(), candidate.artifact_id)
                for candidate in candidates
            ],
        )


def verify_retention_result(
    registry: Registry,
    candidates: list[RetentionCandidate],
    *,
    paths: ProjectPaths,
) -> None:
    """Verify every planned path is absent and every tombstone is recorded."""

    identifiers = [candidate.artifact_id for candidate in candidates]
    placeholders = ",".join("?" for _ in identifiers)
    with registry.connect() as connection:
        rows = connection.execute(
            f"""
            SELECT a.artifact_id, a.status, c.verification_status
            FROM artifacts a
            LEFT JOIN checkpoint_catalog c ON c.artifact_id = a.artifact_id
            WHERE a.artifact_id IN ({placeholders})
            ORDER BY a.artifact_id
            """,
            identifiers,
        ).fetchall()
    if len(rows) != len(candidates):
        raise ArtifactRetentionError("Retention verification lost registry rows")
    for candidate, row in zip(candidates, rows, strict=True):
        payload = Path(candidate.path)
        if not payload.is_absolute():
            payload = paths.project_root / payload
        if payload.exists() or payload.is_symlink():
            raise ArtifactRetentionError(f"Deleted payload still exists: {payload}")
        if str(row["status"]) != DELETED_BY_RETENTION:
            raise ArtifactRetentionError(
                f"Artifact {candidate.artifact_id} lacks deletion tombstone"
            )
        if candidate.kind == "checkpoints" and row["verification_status"] != (
            DELETED_BY_RETENTION
        ):
            raise ArtifactRetentionError(
                f"Checkpoint {candidate.artifact_id} lacks catalog tombstone"
            )


def compact_registered_artifacts(
    registry: Registry,
    *,
    decision_id: str,
    output_dir: Path,
    paths: ProjectPaths | None = None,
    checkpoint_min_bytes: int = DEFAULT_CHECKPOINT_MIN_BYTES,
    prediction_min_bytes: int = DEFAULT_PREDICTION_MIN_BYTES,
    apply: bool = False,
) -> dict[str, Any]:
    """Plan or apply one explicit registry-aware retention decision."""

    decision_id = _safe_decision_id(decision_id)
    selected_paths = paths or current_paths()
    selected_paths.validate()
    registry.initialize()
    assert_no_live_experiments(registry)
    candidates = select_retention_candidates(
        registry,
        checkpoint_min_bytes=checkpoint_min_bytes,
        prediction_min_bytes=prediction_min_bytes,
    )
    verify_candidates(candidates, paths=selected_paths, verify_sha256=False)
    summary = write_retention_plan(
        candidates,
        decision_id=decision_id,
        output_dir=output_dir,
        checkpoint_min_bytes=checkpoint_min_bytes,
        prediction_min_bytes=prediction_min_bytes,
    )
    if not apply:
        return {"applied": False, **summary}

    _assert_registry_plan_current(registry, candidates)
    verified_payloads = verify_candidates(
        candidates,
        paths=selected_paths,
        verify_sha256=True,
    )
    backup_path = (
        selected_paths.state_root
        / "backups"
        / f"bagm_pre_{decision_id}.sqlite3"
    )
    _sqlite_backup(registry, path=backup_path)
    _mark_pending(registry, candidates)
    for index, (candidate, payload) in enumerate(
        zip(candidates, verified_payloads, strict=True),
        start=1,
    ):
        payload.unlink()
        print(
            f"deleted {index}/{len(candidates)}: "
            f"{candidate.run_id}/{payload.name} ({candidate.size_bytes} bytes)",
            flush=True,
        )
    _mark_deleted(registry, candidates, verified_payloads)
    verify_retention_result(registry, candidates, paths=selected_paths)
    receipt = {
        "schema_version": 1,
        "decision_id": decision_id,
        "completed_at": _utc_now(),
        "artifact_count": len(candidates),
        "run_count": len({candidate.run_id for candidate in candidates}),
        "deleted_bytes": sum(candidate.size_bytes for candidate in candidates),
        "plan_sha256": summary["plan_sha256"],
        "registry_backup": str(backup_path),
        "artifact_status": DELETED_BY_RETENTION,
        "checkpoint_verification_status": DELETED_BY_RETENTION,
    }
    receipt_bytes = (_canonical_json(receipt) + "\n").encode("utf-8")
    receipt_path = output_dir.resolve(strict=False) / "application_receipt.json"
    _write_immutable(receipt_path, receipt_bytes)
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    _write_immutable(
        output_dir.resolve(strict=False) / "application_receipt.sha256",
        f"{receipt_sha}  application_receipt.json\n".encode("utf-8"),
    )
    return {"applied": True, **receipt, "receipt_sha256": receipt_sha}


__all__ = [
    "ArtifactRetentionError",
    "DELETED_BY_RETENTION",
    "DEFAULT_CHECKPOINT_MIN_BYTES",
    "DEFAULT_PREDICTION_MIN_BYTES",
    "RETENTION_DELETION_PENDING",
    "RetentionCandidate",
    "assert_no_live_experiments",
    "compact_registered_artifacts",
    "select_retention_candidates",
    "verify_candidates",
    "verify_retention_result",
    "write_retention_plan",
]
