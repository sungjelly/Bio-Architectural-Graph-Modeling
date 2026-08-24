from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

from spatial_benchmark.relative_qkv_campaign_watchdog import (
    CAMPAIGN_ID,
    build_snapshot,
    persist_snapshot,
)


def _registry(path: Path, *, stale_seed: int | None = None) -> None:
    now = datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc)
    with sqlite3.connect(path) as database:
        database.executescript(
            """
            CREATE TABLE queue_jobs(
                job_id TEXT, campaign_id TEXT, status TEXT, created_at TEXT,
                heartbeat_at TEXT, run_id TEXT, worker_id TEXT,
                requested_gpu TEXT, last_error TEXT, attempt_count INTEGER,
                canonical_config_json TEXT
            );
            CREATE TABLE runs(
                run_id TEXT, campaign_id TEXT, artifact_path TEXT
            );
            """
        )
        for seed in range(4):
            heartbeat = now - (
                timedelta(minutes=10)
                if seed == stale_seed
                else timedelta(seconds=20)
            )
            database.execute(
                "INSERT INTO queue_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"q_{seed}",
                    CAMPAIGN_ID,
                    "running",
                    now.isoformat(),
                    heartbeat.isoformat(),
                    f"r_{seed}",
                    f"worker-{seed}",
                    str(seed),
                    None,
                    1,
                    json.dumps({"seed": seed}),
                ),
            )


def test_snapshot_resolves_four_healthy_dedicated_gpu_runs(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite3"
    _registry(database)
    for seed in range(4):
        (tmp_path / "scratch" / "active_runs" / f"r_{seed}").mkdir(
            parents=True
        )
    checkpoint = (
        tmp_path / "scratch" / "active_runs" / "r_0" / "checkpoints"
        / "epoch_0025.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    snapshot = build_snapshot(
        database=database,
        project_root=tmp_path,
        stale_after_seconds=180,
        now=datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc),
    )

    assert snapshot["healthy"] is True
    assert snapshot["alerts"] == []
    assert set(snapshot["seeds"]) == {"0", "1", "2", "3"}
    assert snapshot["seeds"]["0"]["latest_periodic_epoch"] == 25
    assert snapshot["seeds"]["0"]["checkpoints"][0]["sha256"]


def test_snapshot_reports_stale_heartbeat_and_gpu_mismatch(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite3"
    _registry(database, stale_seed=2)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE queue_jobs SET requested_gpu='0' WHERE job_id='q_3'"
        )

    snapshot = build_snapshot(
        database=database,
        project_root=tmp_path,
        stale_after_seconds=180,
        now=datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc),
    )

    assert snapshot["healthy"] is False
    assert "seed_2:stale_heartbeat" in snapshot["alerts"]
    assert "seed_3:gpu_mismatch_expected_3_got_0" in snapshot["alerts"]


def test_persistence_appends_only_meaningful_state_changes(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite3"
    _registry(database)
    for seed in range(4):
        (tmp_path / "scratch" / "active_runs" / f"r_{seed}").mkdir(
            parents=True
        )
    latest = tmp_path / "monitor" / "latest.json"
    events = tmp_path / "monitor" / "events.jsonl"
    first = build_snapshot(
        database=database,
        project_root=tmp_path,
        now=datetime(2026, 8, 24, 13, 0, tzinfo=timezone.utc),
    )
    assert persist_snapshot(first, latest_path=latest, events_path=events)
    second = build_snapshot(
        database=database,
        project_root=tmp_path,
        previous=json.loads(latest.read_text()),
        now=datetime(2026, 8, 24, 13, 1, tzinfo=timezone.utc),
    )
    assert not persist_snapshot(second, latest_path=latest, events_path=events)
    assert len(events.read_text().splitlines()) == 1

    checkpoint = (
        tmp_path / "scratch" / "active_runs" / "r_1" / "checkpoints"
        / "epoch_0025.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"new checkpoint")
    third = build_snapshot(
        database=database,
        project_root=tmp_path,
        previous=json.loads(latest.read_text()),
        now=datetime(2026, 8, 24, 13, 2, tzinfo=timezone.utc),
    )
    assert persist_snapshot(third, latest_path=latest, events_path=events)
    assert len(events.read_text().splitlines()) == 2
