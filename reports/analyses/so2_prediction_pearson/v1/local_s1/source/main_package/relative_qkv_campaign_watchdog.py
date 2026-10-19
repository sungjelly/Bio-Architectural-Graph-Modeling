"""Read-only health and checkpoint watchdog for the four-seed Relative-QKV run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Mapping, Sequence


CAMPAIGN_ID = "cmp_20260824_cancer_6core_relative_qkv_multiseed"
EXPECTED_GPU_BY_SEED = {0: "0", 1: "1", 2: "2", 3: "3"}
ACTIVE_STATUSES = frozenset({"queued", "claimed", "running"})
_EPOCH_CHECKPOINT = re.compile(r"^epoch_(\d{4,})\.ckpt$")


class RelativeQKVWatchdogError(RuntimeError):
    """Raised when a watchdog snapshot cannot be constructed safely."""


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _checkpoint_inventory(
    run_root: Path | None,
    *,
    project_root: Path,
    cached: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if run_root is None:
        return []
    directory = run_root / "checkpoints"
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.ckpt")):
        before = path.stat()
        relative = _relative_path(path, project_root)
        prior = cached.get(relative, {})
        if (
            prior.get("size_bytes") == before.st_size
            and prior.get("mtime_ns") == before.st_mtime_ns
            and isinstance(prior.get("sha256"), str)
        ):
            checksum = str(prior["sha256"])
        else:
            checksum = _sha256_file(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (
                before.st_size,
                before.st_mtime_ns,
            ):
                # A writer was still publishing the checkpoint.  Omit it until
                # the next poll rather than recording a partial-file digest.
                continue
        match = _EPOCH_CHECKPOINT.fullmatch(path.name)
        records.append(
            {
                "name": path.name,
                "path": relative,
                "epoch": None if match is None else int(match.group(1)),
                "size_bytes": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "sha256": checksum,
            }
        )
    return records


def _cached_checkpoints(previous: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    seeds = previous.get("seeds")
    if not isinstance(seeds, Mapping):
        return result
    for seed in seeds.values():
        if not isinstance(seed, Mapping):
            continue
        checkpoints = seed.get("checkpoints")
        if not isinstance(checkpoints, list):
            continue
        for checkpoint in checkpoints:
            if isinstance(checkpoint, Mapping) and isinstance(
                checkpoint.get("path"), str
            ):
                result[str(checkpoint["path"])] = checkpoint
    return result


def _run_root(
    project_root: Path,
    run_id: str | None,
    artifact_path: str | None,
) -> Path | None:
    if run_id is None:
        return None
    active = project_root / "scratch" / "active_runs" / run_id
    if active.is_dir():
        return active
    if artifact_path:
        artifact = Path(artifact_path)
        if not artifact.is_absolute():
            artifact = project_root / artifact
        if artifact.is_dir():
            return artifact
    return None


def build_snapshot(
    *,
    database: Path,
    project_root: Path,
    previous: Mapping[str, Any] | None = None,
    stale_after_seconds: int = 180,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build one read-only campaign health snapshot."""

    observed_at = now or datetime.now(timezone.utc)
    prior = previous or {}
    cached = _cached_checkpoints(prior)
    uri = f"file:{database.resolve()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise RelativeQKVWatchdogError(
            f"Cannot open registry read-only: {database}"
        ) from exc
    connection.row_factory = sqlite3.Row
    try:
        jobs = connection.execute(
            """
            SELECT job_id, status, created_at, heartbeat_at, run_id, worker_id,
                   requested_gpu, last_error, attempt_count, canonical_config_json
            FROM queue_jobs
            WHERE campaign_id = ?
            ORDER BY created_at, job_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
        run_rows = connection.execute(
            "SELECT run_id, artifact_path FROM runs WHERE campaign_id = ?",
            (CAMPAIGN_ID,),
        ).fetchall()
    finally:
        connection.close()
    artifacts = {
        str(row["run_id"]): row["artifact_path"] for row in run_rows
    }
    grouped: dict[int, list[sqlite3.Row]] = {
        seed: [] for seed in EXPECTED_GPU_BY_SEED
    }
    for row in jobs:
        try:
            config = json.loads(str(row["canonical_config_json"]))
        except (TypeError, ValueError):
            continue
        if not isinstance(config, Mapping):
            continue
        seed = config.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and seed in grouped:
            grouped[seed].append(row)

    alerts: list[str] = []
    seed_records: dict[str, Any] = {}
    for seed, expected_gpu in EXPECTED_GPU_BY_SEED.items():
        candidates = grouped[seed]
        active = [row for row in candidates if row["status"] in ACTIVE_STATUSES]
        completed = [row for row in candidates if row["status"] == "completed"]
        if len(active) > 1:
            alerts.append(f"seed_{seed}:multiple_active_jobs")
        selected = active[-1] if active else None
        if selected is None and completed:
            selected = completed[-1]
        if selected is None and candidates:
            selected = candidates[-1]
        if selected is None:
            alerts.append(f"seed_{seed}:missing_job")
            seed_records[str(seed)] = {
                "expected_gpu": expected_gpu,
                "status": "missing",
                "checkpoints": [],
            }
            continue
        status = str(selected["status"])
        requested_gpu = str(selected["requested_gpu"])
        if requested_gpu != expected_gpu:
            alerts.append(
                f"seed_{seed}:gpu_mismatch_expected_{expected_gpu}_got_{requested_gpu}"
            )
        heartbeat_age: float | None = None
        heartbeat = selected["heartbeat_at"]
        if status == "running":
            if not heartbeat:
                alerts.append(f"seed_{seed}:missing_heartbeat")
            else:
                heartbeat_age = max(
                    0.0,
                    (observed_at - _parse_utc(str(heartbeat))).total_seconds(),
                )
                if heartbeat_age > stale_after_seconds:
                    alerts.append(f"seed_{seed}:stale_heartbeat")
        elif status not in {"queued", "claimed", "completed"}:
            alerts.append(f"seed_{seed}:terminal_status_{status}")
        run_id = None if selected["run_id"] is None else str(selected["run_id"])
        root = _run_root(
            project_root,
            run_id,
            None if run_id is None else artifacts.get(run_id),
        )
        if status == "running" and root is None:
            alerts.append(f"seed_{seed}:missing_run_root")
        checkpoints = _checkpoint_inventory(
            root, project_root=project_root, cached=cached
        )
        epochs = [
            int(record["epoch"])
            for record in checkpoints
            if record["epoch"] is not None
        ]
        markers = (
            []
            if root is None
            else [
                name
                for name in ("_SUCCESS", "_FAILED", "_PRUNED")
                if (root / name).is_file()
            ]
        )
        if status == "completed" and markers != ["_SUCCESS"]:
            alerts.append(f"seed_{seed}:completed_without_unique_success_marker")
        seed_records[str(seed)] = {
            "expected_gpu": expected_gpu,
            "requested_gpu": requested_gpu,
            "job_id": str(selected["job_id"]),
            "run_id": run_id,
            "attempt_count": int(selected["attempt_count"]),
            "worker_id": selected["worker_id"],
            "status": status,
            "heartbeat_at": heartbeat,
            "heartbeat_age_seconds": heartbeat_age,
            "last_error": selected["last_error"],
            "run_root": None if root is None else _relative_path(root, project_root),
            "completion_markers": markers,
            "latest_periodic_epoch": max(epochs, default=0),
            "checkpoints": checkpoints,
        }

    event_state = {
        "alerts": sorted(alerts),
        "seeds": {
            seed: {
                "job_id": record.get("job_id"),
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "requested_gpu": record.get("requested_gpu"),
                "latest_periodic_epoch": record.get("latest_periodic_epoch", 0),
                "completion_markers": record.get("completion_markers", []),
                "checkpoints": [
                    (item["name"], item["sha256"])
                    for item in record.get("checkpoints", [])
                ],
            }
            for seed, record in seed_records.items()
        },
    }
    return {
        "schema": "relative_qkv_campaign_watchdog_v1",
        "campaign_id": CAMPAIGN_ID,
        "observed_at": observed_at.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "read_only_monitor": True,
        "stale_after_seconds": stale_after_seconds,
        "healthy": not alerts,
        "alerts": sorted(alerts),
        "all_four_completed": all(
            record.get("status") == "completed"
            and record.get("completion_markers") == ["_SUCCESS"]
            for record in seed_records.values()
        ),
        "seeds": seed_records,
        "state_digest": _canonical_sha256(event_state),
    }


def persist_snapshot(
    snapshot: Mapping[str, Any],
    *,
    latest_path: Path,
    events_path: Path,
) -> bool:
    """Persist latest state and append an event only when meaningful state changes."""

    previous = _load_json(latest_path)
    changed = previous.get("state_digest") != snapshot.get("state_digest")
    _atomic_json(latest_path, snapshot)
    if changed:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "observed_at": snapshot["observed_at"],
            "state_digest": snapshot["state_digest"],
            "healthy": snapshot["healthy"],
            "alerts": snapshot["alerts"],
            "all_four_completed": snapshot["all_four_completed"],
            "seeds": {
                seed: {
                    "job_id": record.get("job_id"),
                    "run_id": record.get("run_id"),
                    "status": record.get("status"),
                    "requested_gpu": record.get("requested_gpu"),
                    "latest_periodic_epoch": record.get(
                        "latest_periodic_epoch", 0
                    ),
                    "completion_markers": record.get("completion_markers", []),
                    "checkpoint_sha256": {
                        item["name"]: item["sha256"]
                        for item in record.get("checkpoints", [])
                    },
                }
                for seed, record in snapshot["seeds"].items()
            },
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--stale-after-seconds", type=int, default=180)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    database = (
        args.database or root / "state" / "tracking" / "bagm.sqlite3"
    ).resolve()
    output = (
        args.output_dir
        or root / "state" / "monitoring" / "relative_qkv_four_seed"
    ).resolve()
    latest = output / "latest.json"
    events = output / "events.jsonl"
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    while True:
        previous = _load_json(latest)
        snapshot = build_snapshot(
            database=database,
            project_root=root,
            previous=previous,
            stale_after_seconds=args.stale_after_seconds,
        )
        changed = persist_snapshot(snapshot, latest_path=latest, events_path=events)
        if changed or args.once:
            print(json.dumps(snapshot, sort_keys=True), flush=True)
        if args.once:
            return 0 if snapshot["healthy"] else 2
        if snapshot["all_four_completed"]:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
