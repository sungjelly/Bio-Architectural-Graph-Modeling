#!/usr/bin/env python3
"""Finalize the pooled ten-core campaign in the authoritative registry.

This deliberately campaign-specific command accepts only the exact published
comparison manifest checksum supplied by the caller. It verifies every
manifested report file, takes an online SQLite backup, rechecks that no queue
job or run is active inside ``BEGIN IMMEDIATE``, and conditionally changes only
the campaign lifecycle status from ``planned`` to ``complete``. Re-running
against an already-complete campaign is a verified no-op.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.registry import Registry, utc_now  # noqa: E402


CAMPAIGN_ID = "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
COMPARISON_KIND = "pooled_hybrid_ensemble_comparison"
RECEIPT_KIND = "pooled_campaign_registry_finalization_v1"
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
REQUIRED_GATES = (
    "pooled_data_gate",
    "graph_gate",
    "representation_gate",
    "ensemble_gate",
)
TERMINAL_QUEUE_STATUSES = frozenset(
    {"completed", "failed", "pruned", "cancelled", "stale"}
)
TERMINAL_RUN_STATUSES = frozenset(
    {"completed", "failed", "pruned", "cancelled"}
)
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")


class PooledCampaignFinalizationError(RuntimeError):
    """Raised when finalization evidence or registry state is invalid."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PooledCampaignFinalizationError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PooledCampaignFinalizationError(f"{label} must be a list")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PooledCampaignFinalizationError(f"{label} must be an integer")
    return value


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise PooledCampaignFinalizationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PooledCampaignFinalizationError(
            f"cannot hash required file: {path}"
        ) from exc
    return digest.hexdigest()


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PooledCampaignFinalizationError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PooledCampaignFinalizationError(
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
    except PooledCampaignFinalizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PooledCampaignFinalizationError(
            f"{label} is not strict JSON"
        ) from exc
    return dict(_mapping(value, label))


def _expected_comparison_directory(paths: ProjectPaths) -> Path:
    return (
        paths.report_root
        / "analyses"
        / "adjacent_normal_10core_pooled_hybrid_ensemble"
        / "comparison"
    ).resolve(strict=False)


def _reconciliation_directory(paths: ProjectPaths) -> Path:
    return (
        paths.report_root
        / "analyses"
        / "adjacent_normal_10core_pooled_hybrid_ensemble"
        / "registry_reconciliation"
    ).resolve(strict=False)


def _require_child(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise PooledCampaignFinalizationError(
            f"{label} must remain under {root.resolve(strict=False)}"
        ) from exc
    return resolved


def _validate_comparison(
    *,
    paths: ProjectPaths,
    comparison_directory: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    expected_digest = _digest(
        expected_manifest_sha256, "expected comparison manifest checksum"
    )
    directory = comparison_directory.resolve(strict=False)
    if directory != _expected_comparison_directory(paths):
        raise PooledCampaignFinalizationError(
            "comparison directory is not the canonical pooled campaign report"
        )
    if not directory.is_dir() or directory.is_symlink():
        raise PooledCampaignFinalizationError(
            f"comparison directory is missing or is a symlink: {directory}"
        )
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PooledCampaignFinalizationError(
            f"comparison manifest is missing or is a symlink: {manifest_path}"
        )
    manifest_sha256 = _sha256_file(manifest_path)
    if not hmac.compare_digest(manifest_sha256, expected_digest):
        raise PooledCampaignFinalizationError(
            "comparison manifest does not match the explicitly approved checksum"
        )
    manifest = _strict_json(manifest_path, label="comparison manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_kind") != COMPARISON_KIND
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("portable_html") is not True
        or manifest.get("protected_identifiers_emitted") is not False
    ):
        raise PooledCampaignFinalizationError(
            "comparison manifest identity or privacy contract is invalid"
        )

    files = _mapping(manifest.get("files"), "comparison manifest files")
    if not files or "manifest.json" in files or "comparison.json" not in files:
        raise PooledCampaignFinalizationError(
            "comparison manifest has an invalid file inventory"
        )
    verified_files: dict[str, dict[str, Any]] = {}
    for reference, raw_entry in sorted(files.items()):
        if not isinstance(reference, str) or not reference:
            raise PooledCampaignFinalizationError(
                "comparison manifest contains an invalid file reference"
            )
        pure = PurePosixPath(reference)
        if (
            pure.is_absolute()
            or reference != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise PooledCampaignFinalizationError(
                f"unsafe comparison file reference: {reference!r}"
            )
        candidate = _require_child(
            directory.joinpath(*pure.parts),
            directory,
            label=f"comparison file {reference}",
        )
        if not candidate.is_file() or candidate.is_symlink():
            raise PooledCampaignFinalizationError(
                f"comparison file is missing or is a symlink: {reference}"
            )
        entry = _mapping(raw_entry, f"comparison manifest entry {reference}")
        expected_size = _integer(
            entry.get("size_bytes"), f"comparison file size {reference}"
        )
        if expected_size < 0:
            raise PooledCampaignFinalizationError(
                f"comparison file size is negative: {reference}"
            )
        expected_file_digest = _digest(
            entry.get("sha256"), f"comparison file checksum {reference}"
        )
        actual_size = candidate.stat().st_size
        actual_digest = _sha256_file(candidate)
        if actual_size != expected_size or not hmac.compare_digest(
            actual_digest, expected_file_digest
        ):
            raise PooledCampaignFinalizationError(
                f"comparison file size or checksum mismatch: {reference}"
            )
        verified_files[reference] = {
            "sha256": actual_digest,
            "size_bytes": actual_size,
        }

    actual_files: set[str] = set()
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            raise PooledCampaignFinalizationError(
                f"comparison report contains a symlink: {candidate}"
            )
        if candidate.is_file() and candidate != manifest_path:
            actual_files.add(candidate.relative_to(directory).as_posix())
    if actual_files != set(files):
        raise PooledCampaignFinalizationError(
            "comparison directory and manifest file inventories differ"
        )

    comparison_entry = verified_files["comparison.json"]
    comparison_digest = _digest(
        manifest.get("comparison_sha256"), "manifest comparison checksum"
    )
    if not hmac.compare_digest(
        comparison_digest, str(comparison_entry["sha256"])
    ):
        raise PooledCampaignFinalizationError(
            "manifest comparison checksum disagrees with its file entry"
        )
    provenance_entry = verified_files.get("provenance.json")
    if provenance_entry is None or not hmac.compare_digest(
        _digest(
            manifest.get("provenance_sha256"),
            "manifest provenance checksum",
        ),
        str(provenance_entry["sha256"]),
    ):
        raise PooledCampaignFinalizationError(
            "manifest provenance checksum disagrees with its file entry"
        )
    comparison = _strict_json(
        directory / "comparison.json", label="pooled comparison"
    )
    if (
        comparison.get("schema_version") != 1
        or comparison.get("artifact_kind") != COMPARISON_KIND
        or comparison.get("campaign_id") != CAMPAIGN_ID
        or comparison.get("status") != "complete"
        or comparison.get("exploratory") is not True
        or comparison.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or comparison.get("tissue_context_is_true_normal") is not False
    ):
        raise PooledCampaignFinalizationError(
            "comparison campaign, status, or study identity is invalid"
        )
    gates = _mapping(comparison.get("frozen_gates"), "comparison frozen gates")
    failed_gates: list[str] = []
    for gate_name in REQUIRED_GATES:
        gate = _mapping(gates.get(gate_name), f"comparison gate {gate_name}")
        if not isinstance(gate.get("passed"), bool):
            raise PooledCampaignFinalizationError(
                f"comparison gate {gate_name} lacks a boolean result"
            )
        if gate["passed"] is False:
            failed_gates.append(gate_name)
    reported_failed = _list(
        comparison.get("failed_gates"), "comparison failed gates"
    )
    if (
        any(not isinstance(value, str) for value in reported_failed)
        or sorted(reported_failed) != sorted(failed_gates)
    ):
        raise PooledCampaignFinalizationError(
            "comparison failed-gate inventory is inconsistent"
        )
    expected_outcome = "supported" if not failed_gates else "negative"
    if comparison.get("outcome") != expected_outcome:
        raise PooledCampaignFinalizationError(
            "comparison outcome is inconsistent with the frozen gates"
        )
    coverage = _mapping(comparison.get("coverage"), "comparison coverage")
    if (
        _integer(
            coverage.get("production_slots_expected"),
            "expected production slots",
        )
        != 14
        or _integer(
            coverage.get("production_slots_completed"),
            "completed production slots",
        )
        != 14
        or _integer(
            coverage.get("failed_production_attempts_before_completion"),
            "failed production attempts",
        )
        != 0
    ):
        raise PooledCampaignFinalizationError(
            "comparison does not document exact successful production coverage"
        )
    training_scope = _mapping(
        comparison.get("training_scope"), "comparison training scope"
    )
    if (
        training_scope.get("one_shared_model_per_member") is not True
        or training_scope.get("cross_core_edges") is not False
        or _integer(
            training_scope.get("production_run_count"),
            "comparison production run count",
        )
        != 14
        or training_scope.get("core_aliases") != list(ALIASES)
    ):
        raise PooledCampaignFinalizationError(
            "comparison pooled-training scope is invalid"
        )
    return {
        "directory": directory.as_posix(),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_sha256,
        "manifest_size_bytes": manifest_path.stat().st_size,
        "comparison_sha256": comparison_digest,
        "comparison_size_bytes": comparison_entry["size_bytes"],
        "verified_file_count": len(verified_files),
        "verified_files_checksum": canonical_sha256(verified_files),
        "status": "complete",
        "outcome": expected_outcome,
        "failed_gates": failed_gates,
    }


def _status_counts(
    connection: sqlite3.Connection, *, table: str
) -> dict[str, int]:
    if table not in {"queue_jobs", "runs"}:
        raise ValueError(f"unsupported status table: {table}")
    rows = connection.execute(
        f"""
        SELECT status, COUNT(*) AS count
        FROM {table}
        WHERE campaign_id = ?
        GROUP BY status
        ORDER BY status
        """,
        (CAMPAIGN_ID,),
    ).fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def _registry_snapshot(connection: sqlite3.Connection) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT campaign_id, status, config_json, created_at, updated_at
        FROM campaigns
        WHERE campaign_id = ?
        """,
        (CAMPAIGN_ID,),
    ).fetchone()
    if row is None:
        raise PooledCampaignFinalizationError(
            "pooled campaign is absent from the authoritative registry"
        )
    status = str(row["status"])
    if status not in {"planned", "complete"}:
        raise PooledCampaignFinalizationError(
            f"campaign status cannot be finalized safely: {status!r}"
        )
    queue_counts = _status_counts(connection, table="queue_jobs")
    run_counts = _status_counts(connection, table="runs")
    nonterminal_queue = set(queue_counts) - TERMINAL_QUEUE_STATUSES
    nonterminal_runs = set(run_counts) - TERMINAL_RUN_STATUSES
    if nonterminal_queue or nonterminal_runs:
        raise PooledCampaignFinalizationError(
            "campaign has active or unknown queue/run states: "
            f"queue={sorted(nonterminal_queue)}, runs={sorted(nonterminal_runs)}"
        )
    config_json = str(row["config_json"])
    return {
        "campaign": {
            "campaign_id": str(row["campaign_id"]),
            "status": status,
            "config_json": config_json,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        },
        "queue_status_counts": queue_counts,
        "run_status_counts": run_counts,
    }


def _read_registry_snapshot(database_path: Path) -> dict[str, Any]:
    if not database_path.is_file():
        raise PooledCampaignFinalizationError(
            f"registry database is missing: {database_path}"
        )
    uri = database_path.resolve(strict=False).as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            integrity = [
                str(row[0])
                for row in connection.execute("PRAGMA integrity_check").fetchall()
            ]
            if integrity != ["ok"]:
                raise PooledCampaignFinalizationError(
                    f"registry integrity check failed: {integrity}"
                )
            return _registry_snapshot(connection)
    except PooledCampaignFinalizationError:
        raise
    except sqlite3.Error as exc:
        raise PooledCampaignFinalizationError(
            "cannot inspect the authoritative registry"
        ) from exc


def _online_backup(*, database_path: Path, backup_path: Path) -> dict[str, Any]:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists() or backup_path.is_symlink():
        raise FileExistsError(f"database backup already exists: {backup_path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{backup_path.name}.", suffix=".tmp", dir=backup_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source_uri = database_path.resolve(strict=False).as_uri() + "?mode=ro"
        with sqlite3.connect(source_uri, uri=True) as source:
            with sqlite3.connect(temporary) as destination:
                source.backup(destination)
        with sqlite3.connect(temporary) as check:
            integrity = [
                str(row[0])
                for row in check.execute("PRAGMA integrity_check").fetchall()
            ]
        if integrity != ["ok"]:
            raise PooledCampaignFinalizationError(
                f"database backup integrity check failed: {integrity}"
            )
        os.link(temporary, backup_path)
    except BaseException:
        backup_path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": backup_path.as_posix(),
        "sha256": _sha256_file(backup_path),
        "size_bytes": backup_path.stat().st_size,
        "integrity_check": "ok",
    }


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


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"append-only receipt already exists: {path}")
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


def _timestamp_slug(timestamp: str) -> str:
    return (
        timestamp.replace("-", "")
        .replace(":", "")
        .replace(".", "")
        .replace("Z", "Z")
    )


def finalize_campaign(
    *,
    paths: ProjectPaths,
    database_path: Path,
    comparison_directory: Path,
    expected_manifest_sha256: str,
    receipt_output_path: Path,
    backup_path: Path | None = None,
    finalized_at: str | None = None,
) -> dict[str, Any]:
    """Finalize the exact pooled campaign and return its signed receipt."""

    database = database_path.resolve(strict=False)
    canonical_database = (
        paths.state_root / "tracking" / "bagm.sqlite3"
    ).resolve(strict=False)
    if database != canonical_database:
        raise PooledCampaignFinalizationError(
            "only the authoritative BAGM registry may be finalized"
        )
    receipt_path = _require_child(
        receipt_output_path,
        _reconciliation_directory(paths),
        label="finalization receipt",
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(
            f"append-only receipt already exists: {receipt_path}"
        )
    timestamp = finalized_at or utc_now()
    lock_path = (
        paths.state_root / "tracking" / f".{CAMPAIGN_ID}.finalization.lock"
    )
    with _application_lock(lock_path):
        comparison = _validate_comparison(
            paths=paths,
            comparison_directory=comparison_directory,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        preflight = _read_registry_snapshot(database)
        if backup_path is None:
            backup_path = (
                paths.state_root
                / "tracking"
                / "backups"
                / (
                    "bagm.before_pooled_campaign_finalization."
                    f"{_timestamp_slug(timestamp)}."
                    f"{comparison['manifest_sha256'][:12]}.sqlite3"
                )
            )
        backup = _require_child(
            backup_path,
            paths.state_root / "tracking" / "backups",
            label="registry backup",
        )
        backup_evidence = _online_backup(
            database_path=database, backup_path=backup
        )
        backup_snapshot = _read_registry_snapshot(backup)
        if backup_snapshot != preflight:
            raise PooledCampaignFinalizationError(
                "campaign registry state changed while the online backup was made"
            )

        registry = Registry(database, initialize=False)
        changed_count = 0
        with registry.transaction(immediate=True) as connection:
            current = _registry_snapshot(connection)
            if current != backup_snapshot:
                raise PooledCampaignFinalizationError(
                    "campaign registry state changed before BEGIN IMMEDIATE"
                )
            campaign = current["campaign"]
            config_json = str(campaign["config_json"])
            if campaign["status"] == "planned":
                cursor = connection.execute(
                    """
                    UPDATE campaigns
                    SET status = 'complete', updated_at = ?
                    WHERE campaign_id = ?
                      AND status = 'planned'
                      AND config_json = ?
                      AND updated_at = ?
                    """,
                    (
                        timestamp,
                        CAMPAIGN_ID,
                        config_json,
                        campaign["updated_at"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise PooledCampaignFinalizationError(
                        "campaign planned-to-complete CAS changed an invalid "
                        f"number of rows: {cursor.rowcount}"
                    )
                changed_count = 1
            post = _registry_snapshot(connection)
            if (
                post["campaign"]["status"] != "complete"
                or post["campaign"]["config_json"] != config_json
                or post["campaign"]["created_at"] != campaign["created_at"]
                or post["queue_status_counts"] != current["queue_status_counts"]
                or post["run_status_counts"] != current["run_status_counts"]
            ):
                raise PooledCampaignFinalizationError(
                    "campaign finalization changed prohibited registry state"
                )
            if changed_count == 0 and post != current:
                raise PooledCampaignFinalizationError(
                    "already-complete idempotent verification changed registry state"
                )

        config_sha256 = hashlib.sha256(
            str(backup_snapshot["campaign"]["config_json"]).encode("utf-8")
        ).hexdigest()
        receipt_payload = {
            "schema_version": 1,
            "receipt_kind": RECEIPT_KIND,
            "campaign_id": CAMPAIGN_ID,
            "finalized_at": timestamp,
            "comparison": comparison,
            "database": {
                "path": database.as_posix(),
                "backup": backup_evidence,
                "integrity_check_before": "ok",
            },
            "transaction": {
                "mode": "BEGIN IMMEDIATE",
                "compare_and_swap": "planned->complete",
                "pre_status": backup_snapshot["campaign"]["status"],
                "post_status": "complete",
                "changed_count": changed_count,
                "idempotent_already_complete": changed_count == 0,
                "config_json_sha256_before": config_sha256,
                "config_json_sha256_after": config_sha256,
                "config_json_unchanged": True,
                "queue_status_counts": backup_snapshot[
                    "queue_status_counts"
                ],
                "run_status_counts": backup_snapshot["run_status_counts"],
                "active_queue_or_run_states": False,
                "run_or_queue_rows_modified": False,
            },
            "checksum_algorithm": "canonical-json-sha256",
        }
        receipt = {
            **receipt_payload,
            "checksum": canonical_sha256(receipt_payload),
        }
        _write_new_json(receipt_path, receipt)
        return receipt


def _parser(paths: ProjectPaths) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=paths.state_root / "tracking" / "bagm.sqlite3",
    )
    parser.add_argument(
        "--comparison-directory",
        type=Path,
        default=_expected_comparison_directory(paths),
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="Exact explicitly approved SHA-256 of comparison/manifest.json.",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        help="Append-only finalization receipt path.",
    )
    parser.add_argument(
        "--backup-output",
        type=Path,
        help="Append-only online SQLite backup path.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    paths = current_paths()
    arguments = _parser(paths).parse_args(argv)
    timestamp = utc_now()
    receipt_output = arguments.receipt_output or (
        _reconciliation_directory(paths)
        / f"campaign_finalization_{_timestamp_slug(timestamp)}.json"
    )
    receipt = finalize_campaign(
        paths=paths,
        database_path=arguments.database,
        comparison_directory=arguments.comparison_directory,
        expected_manifest_sha256=arguments.expected_manifest_sha256,
        receipt_output_path=receipt_output,
        backup_path=arguments.backup_output,
        finalized_at=timestamp,
    )
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "status": receipt["transaction"]["post_status"],
                "outcome": receipt["comparison"]["outcome"],
                "changed_count": receipt["transaction"]["changed_count"],
                "receipt": receipt_output.resolve(strict=False).as_posix(),
                "checksum": receipt["checksum"],
                "backup": receipt["database"]["backup"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
