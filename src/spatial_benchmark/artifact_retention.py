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
import shutil
import sqlite3
from typing import Any, Iterable, Sequence

from .paths import ProjectPaths, current_paths
from .registry import (
    ARTIFACT_STATUS_DELETED_BY_RETENTION,
    ARTIFACT_STATUS_RETENTION_PENDING,
    FULL_RUN_RETENTION_RECEIPT_KIND,
    Registry,
    full_run_artifact_inventory_sha256,
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


@dataclass(frozen=True, slots=True)
class RunBundleRetirement:
    """One exact terminal run bundle approved for complete retirement."""

    run_id: str
    campaign_id: str
    run_status: str
    artifact_root: str
    completion_marker: str
    completion_marker_size: int
    completion_marker_sha256: str
    completion_marker_content_sha256: str
    checksum_manifest_sha256: str
    present_artifact_count: int
    present_artifact_bytes: int
    artifact_inventory_count: int
    artifact_inventory_bytes: int
    artifact_inventory_sha256: str
    prior_tombstone_count: int
    prior_tombstone_bytes: int
    prior_tombstone_artifact_ids: tuple[int, ...]

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunArtifactRetirementRecord:
    """One original artifact identity, including any prior tombstone state."""

    artifact_id: int
    run_id: str
    kind: str
    path: str
    size_bytes: int
    sha256: str
    pre_retirement_status: str

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


def _select_full_run_retirement(
    registry: Registry,
    *,
    run_ids: Sequence[str],
    expected_campaign_id: str,
    paths: ProjectPaths,
) -> tuple[
    list[RetentionCandidate],
    list[RunBundleRetirement],
    list[RunArtifactRetirementRecord],
]:
    """Resolve an exact, closed inventory for complete terminal-run retirement."""

    requested = [str(run_id).strip() for run_id in run_ids]
    if not requested or any(not run_id for run_id in requested):
        raise ValueError("run_ids must contain at least one non-empty run ID")
    if len(requested) != len(set(requested)):
        raise ValueError("run_ids must not contain duplicates")
    if any(not run_id.startswith("r_") for run_id in requested):
        raise ValueError("full run retirement accepts only modern r_* run IDs")
    campaign_id = str(expected_campaign_id).strip()
    if not campaign_id:
        raise ValueError("expected_campaign_id must be non-empty")

    modern_root = (paths.artifact_root / "runs").resolve(strict=True)
    candidates: list[RetentionCandidate] = []
    bundles: list[RunBundleRetirement] = []
    inventory_records: list[RunArtifactRetirementRecord] = []
    seen_roots: set[Path] = set()

    with registry.connect() as connection:
        campaign_run_ids = {
            str(row["run_id"])
            for row in connection.execute(
                "SELECT run_id FROM runs WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchall()
        }
        if set(requested) != campaign_run_ids:
            raise ArtifactRetentionError(
                "Explicit run IDs must equal the campaign's complete run set; "
                f"missing={sorted(campaign_run_ids - set(requested))}, "
                f"extra={sorted(set(requested) - campaign_run_ids)}"
            )
        for run_id in sorted(requested):
            run = connection.execute(
                """
                SELECT r.run_id, r.campaign_id, r.status, r.artifact_path,
                       rc.lifecycle_stage, rc.retention_class
                FROM runs r
                JOIN run_categories rc ON rc.run_id = r.run_id
                WHERE r.run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                raise ArtifactRetentionError(f"Unknown run selected: {run_id}")
            if str(run["campaign_id"]) != campaign_id:
                raise ArtifactRetentionError(
                    f"Run {run_id} belongs to {run['campaign_id']}, not "
                    f"{campaign_id}"
                )
            run_status = str(run["status"])
            marker_name = {
                "completed": "_SUCCESS",
                "failed": "_FAILED",
                "pruned": "_PRUNED",
            }.get(run_status)
            if marker_name is None:
                raise ArtifactRetentionError(
                    f"Run {run_id} is not terminally archivable: {run_status}"
                )
            if not run["artifact_path"]:
                raise ArtifactRetentionError(f"Run {run_id} has no artifact root")
            root = Path(str(run["artifact_path"]))
            if not root.is_absolute():
                root = paths.project_root / root
            if root.is_symlink() or not root.is_dir():
                raise ArtifactRetentionError(
                    f"Run root is missing, not a directory, or a symlink: {root}"
                )
            root = root.resolve(strict=True)
            if not root.is_relative_to(modern_root):
                raise ArtifactRetentionError(
                    f"Run root is outside the modern archive: {root}"
                )
            relative_root = root.relative_to(modern_root)
            if (
                len(relative_root.parts) != 3
                or relative_root.parts[-1] != run_id
                or not relative_root.parts[0].isdigit()
                or not relative_root.parts[1].isdigit()
            ):
                raise ArtifactRetentionError(
                    f"Run root does not match YYYY/MM/<run_id>: {root}"
                )
            if root in seen_roots:
                raise ArtifactRetentionError(f"Duplicate run root: {root}")
            seen_roots.add(root)
            symlinks = [item for item in root.rglob("*") if item.is_symlink()]
            if symlinks:
                raise ArtifactRetentionError(
                    f"Run {run_id} contains symlinks: {symlinks[:5]}"
                )

            rows = connection.execute(
                """
                SELECT artifact_id, kind, path, size_bytes, sha256, status
                FROM artifacts
                WHERE run_id = ?
                ORDER BY artifact_id
                """,
                (run_id,),
            ).fetchall()
            if not rows:
                raise ArtifactRetentionError(
                    f"Run {run_id} has no registered artifact inventory"
                )
            if any(
                str(row["kind"]) == FULL_RUN_RETENTION_RECEIPT_KIND
                and str(row["status"]) == "present"
                for row in rows
            ):
                raise ArtifactRetentionError(
                    f"Run {run_id} already has a full-retirement receipt"
                )

            present_paths: set[Path] = set()
            prior_tombstone_ids: list[int] = []
            prior_tombstone_bytes = 0
            run_candidates: list[RetentionCandidate] = []
            run_inventory_records: list[RunArtifactRetirementRecord] = []
            for row in rows:
                status = str(row["status"])
                registered_path = str(row["path"])
                payload = Path(registered_path)
                if not payload.is_absolute():
                    payload = paths.project_root / payload
                resolved_payload = payload.resolve(strict=False)
                if not resolved_payload.is_relative_to(root):
                    raise ArtifactRetentionError(
                        f"Artifact {row['artifact_id']} is outside run root: "
                        f"{payload}"
                    )
                if status == DELETED_BY_RETENTION:
                    sha256 = str(row["sha256"] or "").lower()
                    if not _SHA256.fullmatch(sha256):
                        raise ArtifactRetentionError(
                            f"Prior tombstone {row['artifact_id']} has no valid SHA-256"
                        )
                    if row["size_bytes"] is None or int(row["size_bytes"]) < 0:
                        raise ArtifactRetentionError(
                            f"Prior tombstone {row['artifact_id']} has no valid size"
                        )
                    if payload.exists() or payload.is_symlink():
                        raise ArtifactRetentionError(
                            f"Prior tombstone path still exists: {payload}"
                        )
                    prior_tombstone_ids.append(int(row["artifact_id"]))
                    prior_tombstone_bytes += int(row["size_bytes"] or 0)
                elif status != "present":
                    raise ArtifactRetentionError(
                        f"Run {run_id} has unsupported artifact state "
                        f"{status!r} for {payload}"
                    )
                else:
                    sha256 = str(row["sha256"] or "").lower()
                    if not _SHA256.fullmatch(sha256):
                        raise ArtifactRetentionError(
                            f"Artifact {row['artifact_id']} has no valid SHA-256"
                        )
                    if row["size_bytes"] is None or int(row["size_bytes"]) < 0:
                        raise ArtifactRetentionError(
                            f"Artifact {row['artifact_id']} has no valid size"
                        )
                    if payload.is_symlink() or not payload.is_file():
                        raise ArtifactRetentionError(
                            "Registered artifact is missing, not a file, or a "
                            f"symlink: {payload}"
                        )
                    resolved_payload = payload.resolve(strict=True)
                    if resolved_payload in present_paths:
                        raise ArtifactRetentionError(
                            f"Duplicate registered artifact path: {resolved_payload}"
                        )
                    present_paths.add(resolved_payload)
                    run_candidates.append(
                        RetentionCandidate(
                            artifact_id=int(row["artifact_id"]),
                            run_id=run_id,
                            campaign_id=campaign_id,
                            run_status=run_status,
                            lifecycle_stage=str(run["lifecycle_stage"]),
                            retention_class=str(run["retention_class"]),
                            kind=str(row["kind"]),
                            path=registered_path,
                            size_bytes=int(row["size_bytes"]),
                            sha256=sha256,
                            artifact_root=str(root),
                        )
                    )
                run_inventory_records.append(
                    RunArtifactRetirementRecord(
                        artifact_id=int(row["artifact_id"]),
                        run_id=run_id,
                        kind=str(row["kind"]),
                        path=registered_path,
                        size_bytes=int(row["size_bytes"]),
                        sha256=sha256,
                        pre_retirement_status=status,
                    )
                )

            marker = root / marker_name
            if marker.is_symlink() or not marker.is_file():
                raise ArtifactRetentionError(
                    f"Run {run_id} lacks its expected terminal marker {marker_name}"
                )
            marker = marker.resolve(strict=True)
            live_files = {
                item.resolve(strict=True)
                for item in root.rglob("*")
                if item.is_file() and not item.is_symlink()
            }
            expected_files = present_paths | {marker}
            if live_files != expected_files:
                missing = sorted(str(path) for path in expected_files - live_files)
                unregistered = sorted(str(path) for path in live_files - expected_files)
                raise ArtifactRetentionError(
                    f"Run {run_id} inventory is not closed; missing={missing}, "
                    f"unregistered={unregistered}"
                )
            marker_size = marker.stat().st_size
            try:
                marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArtifactRetentionError(
                    f"Run {run_id} has an unreadable terminal marker"
                ) from error
            expected_marker_status = marker_name.removeprefix("_").lower()
            if (
                not isinstance(marker_payload, dict)
                or marker_payload.get("run_id") != run_id
                or marker_payload.get("status") != expected_marker_status
                or not _SHA256.fullmatch(
                    str(marker_payload.get("content_sha256", "")).lower()
                )
            ):
                raise ArtifactRetentionError(
                    f"Run {run_id} terminal marker identity is invalid"
                )

            manifest_path = root / "provenance/artifact_checksums.json"
            manifest_rows = [
                record
                for record in run_inventory_records
                if Path(record.path).resolve(strict=False) == manifest_path
            ]
            if len(manifest_rows) != 1 or not manifest_path.is_file():
                raise ArtifactRetentionError(
                    f"Run {run_id} has no unique registered checksum manifest"
                )
            try:
                manifest_payload = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ArtifactRetentionError(
                    f"Run {run_id} checksum manifest is unreadable"
                ) from error
            manifest_files = (
                manifest_payload.get("files")
                if isinstance(manifest_payload, dict)
                else None
            )
            if not isinstance(manifest_files, dict):
                raise ArtifactRetentionError(
                    f"Run {run_id} checksum manifest is malformed"
                )
            manifest_digest = hashlib.sha256(
                _canonical_json(manifest_files).encode("utf-8")
            ).hexdigest()
            if marker_payload["content_sha256"] != manifest_digest:
                raise ArtifactRetentionError(
                    f"Run {run_id} marker does not bind its checksum manifest"
                )
            registered_manifest_entries: dict[str, dict[str, Any]] = {}
            for record in run_inventory_records:
                record_path = Path(record.path)
                if not record_path.is_absolute():
                    record_path = paths.project_root / record_path
                relative = record_path.resolve(strict=False).relative_to(root).as_posix()
                if relative == "provenance/artifact_checksums.json":
                    continue
                registered_manifest_entries[relative] = {
                    "type": "file",
                    "size": record.size_bytes,
                    "sha256": record.sha256,
                }
            if registered_manifest_entries != manifest_files:
                missing = sorted(set(manifest_files) - set(registered_manifest_entries))
                extra = sorted(set(registered_manifest_entries) - set(manifest_files))
                changed = sorted(
                    key
                    for key in set(manifest_files).intersection(
                        registered_manifest_entries
                    )
                    if manifest_files[key] != registered_manifest_entries[key]
                )
                raise ArtifactRetentionError(
                    f"Run {run_id} registry/manifest mismatch; missing={missing}, "
                    f"extra={extra}, changed={changed}"
                )

            inventory_dicts = [record.record() for record in run_inventory_records]
            candidates.extend(run_candidates)
            inventory_records.extend(run_inventory_records)
            bundles.append(
                RunBundleRetirement(
                    run_id=run_id,
                    campaign_id=campaign_id,
                    run_status=run_status,
                    artifact_root=str(root),
                    completion_marker=str(marker),
                    completion_marker_size=marker_size,
                    completion_marker_sha256=_sha256_file(marker),
                    completion_marker_content_sha256=manifest_digest,
                    checksum_manifest_sha256=manifest_rows[0].sha256,
                    present_artifact_count=len(run_candidates),
                    present_artifact_bytes=sum(
                        candidate.size_bytes for candidate in run_candidates
                    ),
                    artifact_inventory_count=len(run_inventory_records),
                    artifact_inventory_bytes=sum(
                        record.size_bytes for record in run_inventory_records
                    ),
                    artifact_inventory_sha256=full_run_artifact_inventory_sha256(
                        inventory_dicts
                    ),
                    prior_tombstone_count=len(prior_tombstone_ids),
                    prior_tombstone_bytes=prior_tombstone_bytes,
                    prior_tombstone_artifact_ids=tuple(prior_tombstone_ids),
                )
            )

    for left in seen_roots:
        for right in seen_roots:
            if left != right and left.is_relative_to(right):
                raise ArtifactRetentionError(
                    f"Nested run roots are not permitted: {left} within {right}"
                )
    return candidates, bundles, inventory_records


def _full_retirement_plan_bytes(
    bundles: Sequence[RunBundleRetirement],
    inventory_records: Sequence[RunArtifactRetirementRecord],
) -> bytes:
    records = [
        {"record_type": "registered_artifact", **record.record()}
        for record in inventory_records
    ]
    records.extend(
        {"record_type": "completion_marker", **bundle.record()}
        for bundle in bundles
    )
    records.sort(
        key=lambda record: (
            str(record["run_id"]),
            str(record["record_type"]),
            int(record.get("artifact_id", -1)),
        )
    )
    return ("\n".join(_canonical_json(record) for record in records) + "\n").encode(
        "utf-8"
    )


def _remove_empty_run_root(root: Path) -> None:
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.rmdir()
    root.rmdir()


def retire_registered_run_bundles(
    registry: Registry,
    *,
    run_ids: Sequence[str],
    expected_campaign_id: str,
    decision_id: str,
    output_dir: Path,
    prior_retention_decision_ids: Sequence[str] = (),
    paths: ProjectPaths | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Retire every bundle in one explicitly enumerated campaign run set.

    The operation retains historical run and artifact rows, writes a complete
    checksum-bound inventory, and registers one external receipt per retired
    root. It deliberately cannot select a wildcard, a partial campaign, or an
    upstream run merely because it shares a model or scientific identifier.
    """

    decision_id = _safe_decision_id(decision_id)
    selected_paths = paths or current_paths()
    selected_paths.validate()
    selected_paths.ensure_runtime_directories()
    registry.initialize()
    assert_no_live_experiments(registry)

    resolved_output = output_dir
    if not resolved_output.is_absolute():
        resolved_output = selected_paths.project_root / resolved_output
    resolved_output = resolved_output.resolve(strict=False)
    report_root = selected_paths.report_root.resolve(strict=True)
    if not resolved_output.is_relative_to(report_root) or resolved_output == report_root:
        raise ArtifactRetentionError(
            f"Full-retirement records must use a subdirectory of {report_root}"
        )

    candidates, bundles, inventory_records = _select_full_run_retirement(
        registry,
        run_ids=run_ids,
        expected_campaign_id=expected_campaign_id,
        paths=selected_paths,
    )
    if not candidates or not bundles:
        raise ArtifactRetentionError("Full-run retirement selection is empty")
    if any(
        resolved_output.is_relative_to(Path(bundle.artifact_root))
        for bundle in bundles
    ):
        raise ArtifactRetentionError(
            "Retention records may not be written inside a retired run root"
        )
    prior_decisions = sorted(
        {str(value).strip() for value in prior_retention_decision_ids if str(value).strip()}
    )
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,79}", value) for value in prior_decisions):
        raise ValueError("prior retention decision IDs must use safe decision syntax")

    verify_candidates(candidates, paths=selected_paths, verify_sha256=False)
    plan_payload = _full_retirement_plan_bytes(
        bundles,
        inventory_records,
    )
    plan_sha256 = hashlib.sha256(plan_payload).hexdigest()
    plan_path = resolved_output / "deletion_plan.jsonl"
    _write_immutable(plan_path, plan_payload)
    _write_immutable(
        resolved_output / "deletion_plan.sha256",
        f"{plan_sha256}  deletion_plan.jsonl\n".encode("utf-8"),
    )
    summary = {
        "schema_version": 1,
        "decision_id": decision_id,
        "decision_type": "complete_registered_run_bundle_retirement",
        "expected_campaign_id": str(expected_campaign_id),
        "run_ids": sorted(bundle.run_id for bundle in bundles),
        "run_count": len(bundles),
        "original_artifact_row_count": len(inventory_records),
        "present_artifact_count": len(candidates),
        "prior_tombstone_count": sum(
            bundle.prior_tombstone_count for bundle in bundles
        ),
        "present_registered_bytes": sum(
            candidate.size_bytes for candidate in candidates
        ),
        "terminal_marker_count": len(bundles),
        "terminal_marker_bytes": sum(
            bundle.completion_marker_size for bundle in bundles
        ),
        "logical_bytes_planned_now": sum(
            candidate.size_bytes for candidate in candidates
        )
        + sum(bundle.completion_marker_size for bundle in bundles),
        "prior_tombstone_bytes": sum(
            bundle.prior_tombstone_bytes for bundle in bundles
        ),
        "prior_retention_decision_ids": prior_decisions,
        "plan_sha256": plan_sha256,
    }
    _write_immutable(
        resolved_output / "deletion_plan_summary.json",
        (_canonical_json(summary) + "\n").encode("utf-8"),
    )
    if not apply:
        return {"applied": False, **summary}

    _assert_registry_plan_current(registry, candidates)
    verified_payloads = verify_candidates(
        candidates,
        paths=selected_paths,
        verify_sha256=True,
    )
    for bundle in bundles:
        marker = Path(bundle.completion_marker)
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.stat().st_size != bundle.completion_marker_size
            or _sha256_file(marker) != bundle.completion_marker_sha256
        ):
            raise ArtifactRetentionError(
                f"Completion marker changed after planning: {marker}"
            )

    backup_path = (
        selected_paths.state_root
        / "backups"
        / f"bagm_pre_{decision_id}.sqlite3"
    )
    free_bytes_before = shutil.disk_usage(selected_paths.artifact_root).free
    _sqlite_backup(registry, path=backup_path)
    _mark_pending(registry, candidates)
    for index, payload in enumerate(verified_payloads, start=1):
        payload.unlink()
        if index == 1 or index % 25 == 0 or index == len(verified_payloads):
            print(
                f"deleted {index}/{len(verified_payloads)} registered files",
                flush=True,
            )
    for bundle in bundles:
        marker = Path(bundle.completion_marker)
        marker.unlink()
        _remove_empty_run_root(Path(bundle.artifact_root))
        print(f"retired bundle root: {bundle.run_id}", flush=True)
    _mark_deleted(registry, candidates, verified_payloads)

    retired_at = _utc_now()
    receipt_rows: list[tuple[str, Path, str, int]] = []
    run_receipt_records: list[dict[str, Any]] = []
    for bundle in bundles:
        receipt = {
            "schema_version": 1,
            "receipt_kind": FULL_RUN_RETENTION_RECEIPT_KIND,
            "decision_id": decision_id,
            "retired_at": retired_at,
            "run_id": bundle.run_id,
            "campaign_id": bundle.campaign_id,
            "run_status": bundle.run_status,
            "artifact_root": bundle.artifact_root,
            "completion_marker": {
                "name": Path(bundle.completion_marker).name,
                "path": bundle.completion_marker,
                "size_bytes": bundle.completion_marker_size,
                "file_sha256": bundle.completion_marker_sha256,
                "content_sha256": bundle.completion_marker_content_sha256,
            },
            "checksum_manifest_sha256": bundle.checksum_manifest_sha256,
            "artifact_inventory": {
                "count": bundle.artifact_inventory_count,
                "size_bytes": bundle.artifact_inventory_bytes,
                "sha256": bundle.artifact_inventory_sha256,
            },
            "deleted_now": {
                "registered_artifact_count": bundle.present_artifact_count,
                "registered_artifact_bytes": bundle.present_artifact_bytes,
                "terminal_marker_count": 1,
                "terminal_marker_bytes": bundle.completion_marker_size,
            },
            "prior_tombstones": {
                "artifact_ids": list(bundle.prior_tombstone_artifact_ids),
                "count": bundle.prior_tombstone_count,
                "size_bytes": bundle.prior_tombstone_bytes,
                "decision_ids": prior_decisions,
            },
            "deletion_plan_path": str(plan_path),
            "deletion_plan_sha256": plan_sha256,
            "registry_backup": str(backup_path),
        }
        receipt_content = (_canonical_json(receipt) + "\n").encode("utf-8")
        receipt_path = resolved_output / "run_receipts" / f"{bundle.run_id}.json"
        _write_immutable(receipt_path, receipt_content)
        receipt_sha256 = hashlib.sha256(receipt_content).hexdigest()
        receipt_rows.append(
            (bundle.run_id, receipt_path, receipt_sha256, len(receipt_content))
        )
        run_receipt_records.append(
            {
                "run_id": bundle.run_id,
                "path": str(receipt_path),
                "sha256": receipt_sha256,
                "size_bytes": len(receipt_content),
            }
        )

    with registry.transaction(immediate=True) as connection:
        connection.executemany(
            """
            INSERT INTO artifacts(
                run_id, evaluation_id, kind, path, sha256, size_bytes,
                status, created_at
            ) VALUES (?, NULL, ?, ?, ?, ?, 'present', ?)
            """,
            [
                (
                    receipt_run_id,
                    FULL_RUN_RETENTION_RECEIPT_KIND,
                    str(receipt_path),
                    receipt_sha256,
                    receipt_size,
                    retired_at,
                )
                for receipt_run_id, receipt_path, receipt_sha256, receipt_size in receipt_rows
            ],
        )

    verified_retirements, retirement_issues = registry.verify_full_run_retirements()
    expected_retirements = {bundle.run_id for bundle in bundles}
    if not expected_retirements.issubset(verified_retirements) or retirement_issues:
        raise ArtifactRetentionError(
            "Full-run retirement receipts failed verification: "
            f"verified={sorted(verified_retirements)}, issues={retirement_issues}"
        )
    for candidate, payload in zip(candidates, verified_payloads, strict=True):
        if payload.exists() or payload.is_symlink():
            raise ArtifactRetentionError(f"Deleted payload reappeared: {payload}")
        with registry.connect() as connection:
            status = connection.execute(
                "SELECT status FROM artifacts WHERE artifact_id = ?",
                (candidate.artifact_id,),
            ).fetchone()[0]
        if str(status) != DELETED_BY_RETENTION:
            raise ArtifactRetentionError(
                f"Artifact {candidate.artifact_id} lacks its final tombstone"
            )

    free_bytes_after = shutil.disk_usage(selected_paths.artifact_root).free
    application_receipt = {
        "schema_version": 1,
        "decision_id": decision_id,
        "decision_type": "complete_registered_run_bundle_retirement",
        "completed_at": retired_at,
        "expected_campaign_id": str(expected_campaign_id),
        "run_ids": sorted(expected_retirements),
        "deleted_registered_artifact_count": len(candidates),
        "deleted_terminal_marker_count": len(bundles),
        "logical_deleted_bytes": summary["logical_bytes_planned_now"],
        "measured_free_bytes_before": free_bytes_before,
        "measured_free_bytes_after": free_bytes_after,
        "measured_free_bytes_delta": free_bytes_after - free_bytes_before,
        "prior_tombstone_count": summary["prior_tombstone_count"],
        "prior_retention_decision_ids": prior_decisions,
        "deletion_plan_path": str(plan_path),
        "deletion_plan_sha256": plan_sha256,
        "registry_backup": str(backup_path),
        "run_receipts": run_receipt_records,
        "artifact_status": DELETED_BY_RETENTION,
    }
    application_bytes = (_canonical_json(application_receipt) + "\n").encode(
        "utf-8"
    )
    application_path = resolved_output / "application_receipt.json"
    _write_immutable(application_path, application_bytes)
    application_sha256 = hashlib.sha256(application_bytes).hexdigest()
    _write_immutable(
        resolved_output / "application_receipt.sha256",
        f"{application_sha256}  application_receipt.json\n".encode("utf-8"),
    )
    return {
        "applied": True,
        **application_receipt,
        "receipt_sha256": application_sha256,
    }


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
    "RunArtifactRetirementRecord",
    "RunBundleRetirement",
    "assert_no_live_experiments",
    "compact_registered_artifacts",
    "retire_registered_run_bundles",
    "select_retention_candidates",
    "verify_candidates",
    "verify_retention_result",
    "write_retention_plan",
]
