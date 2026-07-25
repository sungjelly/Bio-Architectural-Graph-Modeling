from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import pytest

from spatial_benchmark.cli import _import_legacy
from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import QueueWorker, WorkerSettings
from spatial_benchmark.registry import (
    InvalidTransitionError,
    Registry,
    RegistryError,
)


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=root,
        config_root=root / "configs",
        data_root=root / "data",
        artifact_root=root / "artifacts",
        state_root=root / "state",
        scratch_root=root / "scratch",
        cache_root=root / "cache",
        export_root=root / "exports",
        report_root=root / "reports",
    )


def _config(*, campaign_id: str = "cmp_test") -> dict[str, object]:
    return {
        "version": 1,
        "campaign": {"campaign_id": campaign_id},
        "model": {
            "name": "g1",
            "family": "topology_only",
            "embedding_dim": 32,
        },
        "masking": {"type": "whole_node", "rate": 0.1},
        "dataset": {
            "dataset_id": "synthetic",
            "version": "v1",
            "split_id": "split_1",
            "dataset_fingerprint": "dataset-fingerprint",
            "split_fingerprint": "split-fingerprint",
            "preprocessing_version": "test-v1",
        },
        "features": {"use_edge_features": False, "edge_features": []},
        "graph": {"neighbor_k": 4, "radius_um": 10.0, "symmetry": "mutual"},
        "trainer": {"learning_rate": 0.001, "batch_size": 1},
        "evaluation": {
            "primary_metric": "val/masked_huber",
            "primary_direction": "minimize",
        },
        "seed": 3,
        "fold": 2,
        "attempt": 1,
        "metadata": {"test_only_dummy": True},
    }


def _registry(root: Path) -> tuple[Registry, ProjectPaths]:
    paths = _paths(root)
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    registry.create_campaign("cmp_test", name="Test campaign")
    registry.register_dataset(
        "synthetic",
        "v1",
        display_name="Synthetic non-clinical fixture",
        raw_fingerprint="dataset-fingerprint",
        processed_fingerprint="dataset-fingerprint",
        preprocessing_version="test-v1",
        verification_status="verified",
    )
    registry.register_split(
        "split_1",
        dataset_id="synthetic",
        dataset_version="v1",
        method="fixed",
        unit="sample",
        fingerprint="split-fingerprint",
        verification_status="verified",
    )
    return registry, paths


def _success_command() -> list[str]:
    code = (
        "import os,pathlib;"
        "p=pathlib.Path(os.environ['BAGM_RUN_SCRATCH']);"
        "(p/'checkpoints').mkdir(parents=True,exist_ok=True);"
        "(p/'checkpoints'/'best.ckpt').write_bytes(b'dummy-checkpoint');"
        "print('cpu dummy complete')"
    )
    return [sys.executable, "-c", code]


def test_schema_integrity_and_required_tables(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite3")
    assert registry.schema_version() == 3
    assert registry.integrity_check() == ["ok"]
    assert {
        "campaigns",
        "variants",
        "campaign_variants",
        "runs",
        "run_aliases",
        "run_categories",
        "evaluations",
        "metrics",
        "artifacts",
        "checkpoint_catalog",
        "datasets",
        "splits",
        "queue_jobs",
        "failures",
    }.issubset(registry.table_names())
    with registry.connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000


def test_atomic_claim_and_transition_guards(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    config = _config()
    registry.register_variant(
        scientific_id(config),
        campaign_id="cmp_test",
        configuration=config,
    )
    job = registry.enqueue(
        campaign_id="cmp_test",
        configuration=config,
        command=_success_command(),
        requested_gpu=None,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(
            pool.map(lambda worker: registry.claim_next(worker), ("worker-a", "worker-b"))
        )
    winners = [record for record in claimed if record is not None]
    assert len(winners) == 1
    assert winners[0]["job_id"] == job["job_id"]
    with pytest.raises(InvalidTransitionError):
        registry.transition_job(
            job["job_id"],
            "completed",
            worker_id=str(winners[0]["worker_id"]),
        )


def test_variant_identity_can_belong_to_multiple_campaigns(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    registry.create_campaign("cmp_second", name="Second campaign")
    config = _config()
    variant = scientific_id(config)
    registry.register_variant(
        variant, campaign_id="cmp_test", configuration=config
    )
    registry.register_variant(
        variant, campaign_id="cmp_second", configuration=config
    )
    with registry.connect() as connection:
        memberships = connection.execute(
            """
            SELECT campaign_id FROM campaign_variants
            WHERE scientific_id = ? ORDER BY campaign_id
            """,
            (variant,),
        ).fetchall()
    assert [row["campaign_id"] for row in memberships] == [
        "cmp_second",
        "cmp_test",
    ]
    registry.create_run(
        "test_second_campaign",
        campaign_id="cmp_second",
        scientific_id=variant,
        repro_id="rep_second",
        seed=1,
        fold=0,
        attempt=1,
        configuration=config,
    )


def test_stale_job_atomically_fails_associated_run(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    config = _config()
    variant = scientific_id(config)
    registry.register_variant(
        variant, campaign_id="cmp_test", configuration=config
    )
    job = registry.enqueue(
        campaign_id="cmp_test",
        configuration=config,
        command=_success_command(),
        requested_gpu=None,
    )
    claimed = registry.claim_next("abandoned-worker")
    assert claimed is not None
    registry.register_run_for_job(
        job["job_id"],
        run_id="test_stale_run",
        scientific_id=variant,
        repro_id="rep_stale",
        artifact_path=tmp_path / "artifacts" / "test_stale_run",
    )
    registry.transition_run("test_stale_run", "running")
    with registry.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE queue_jobs SET heartbeat_at = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00.000000Z", job["job_id"]),
        )

    assert registry.mark_stale(
        heartbeat_before="2001-01-01T00:00:00.000000Z"
    ) == [job["job_id"]]
    assert registry.get_job(job["job_id"])["status"] == "stale"
    stale_run = registry.get_run("test_stale_run")
    assert stale_run is not None
    assert stale_run["status"] == "failed"
    assert stale_run["failure_category"] == "missing_heartbeat"


def test_cpu_dummy_worker_finalizes_canonical_bundle(tmp_path: Path) -> None:
    registry, paths = _registry(tmp_path)
    config = _config()
    job = registry.enqueue(
        campaign_id="cmp_test",
        configuration=config,
        command=_success_command(),
        maximum_attempts=1,
        requested_gpu=None,
    )
    worker = QueueWorker(
        registry,
        settings=WorkerSettings(
            worker_id="test-worker",
            gpu=None,
            once=True,
            min_free_gb=0,
            heartbeat_seconds=0.05,
            allow_test_jobs=True,
        ),
        paths=paths,
    )
    assert worker.run() == 1
    completed = registry.get_job(job["job_id"])
    assert completed is not None
    assert completed["status"] == "completed"
    run = registry.get_run(str(completed["run_id"]))
    assert run is not None
    assert run["status"] == "completed"
    assert run["seed"] == 3
    assert run["fold"] == 2
    artifact = Path(str(run["artifact_path"]))
    assert artifact.is_relative_to(paths.artifact_root / "runs")
    assert (artifact / "_SUCCESS").is_file()
    assert (artifact / "checkpoints" / "best.ckpt").read_bytes()
    assert "cpu dummy complete" in (artifact / "logs" / "stdout.log").read_text()
    assert registry.verify_artifacts(run_id=run["run_id"]) == []
    assert not (paths.scratch_root / "active_runs" / run["run_id"]).exists()


def test_worker_reconciles_crash_window_after_success_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, paths = _registry(tmp_path)
    job = registry.enqueue(
        campaign_id="cmp_test",
        configuration=_config(),
        command=_success_command(),
        maximum_attempts=1,
        requested_gpu=None,
    )
    original_complete = registry.complete_run_and_job
    calls = 0

    def fail_once(**arguments: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RegistryError("synthetic hard-crash boundary")
        original_complete(**arguments)

    monkeypatch.setattr(registry, "complete_run_and_job", fail_once)
    worker = QueueWorker(
        registry,
        settings=WorkerSettings(
            worker_id="recovery-worker",
            gpu=None,
            once=True,
            min_free_gb=0,
            heartbeat_seconds=0.05,
            allow_test_jobs=True,
        ),
        paths=paths,
    )
    assert worker.run() == 1
    recovered = registry.get_job(job["job_id"])
    assert recovered is not None and recovered["status"] == "completed"
    run = registry.get_run(str(recovered["run_id"]))
    assert run is not None and run["status"] == "completed"
    assert (Path(str(run["artifact_path"])) / "_SUCCESS").is_file()
    assert calls == 2


def test_zero_exit_without_checkpoint_is_finalization_failure(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    job = registry.enqueue(
        campaign_id="cmp_test",
        configuration=_config(),
        command=[sys.executable, "-c", "print('no checkpoint')"],
        requested_gpu=None,
    )
    worker = QueueWorker(
        registry,
        settings=WorkerSettings(
            worker_id="test-worker",
            gpu=None,
            once=True,
            min_free_gb=0,
            allow_test_jobs=True,
        ),
        paths=paths,
    )
    assert worker.run() == 1
    failed = registry.get_job(job["job_id"])
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["failure_category"] == "artifact_finalization_failure"
    run = registry.get_run(str(failed["run_id"]))
    assert run is not None and run["status"] == "failed"
    artifact = Path(str(run["artifact_path"]))
    assert (artifact / "_FAILED").is_file()
    assert (artifact / "logs" / "exception.txt").is_file()


def test_retry_copies_exact_config_and_creates_new_run(tmp_path: Path) -> None:
    registry, paths = _registry(tmp_path)
    config = _config()
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    original = registry.enqueue(
        campaign_id="cmp_test",
        configuration=config,
        command=command,
        maximum_attempts=2,
        requested_gpu=None,
    )
    first_worker = QueueWorker(
        registry,
        settings=WorkerSettings(
            worker_id="retry-worker-a",
            gpu=None,
            once=True,
            min_free_gb=0,
            auto_retry=True,
            allow_test_jobs=True,
        ),
        paths=paths,
    )
    assert first_worker.run() == 1
    first = registry.get_job(original["job_id"])
    assert first is not None and first["status"] == "failed"
    queued = registry.list_queue(statuses=["queued"])
    assert len(queued) == 1
    retry = queued[0]
    assert retry["retry_of"] == first["job_id"]
    assert retry["attempt_count"] == 2
    assert retry["canonical_config"] == config
    assert retry["command"] == command

    second_worker = QueueWorker(
        registry,
        settings=WorkerSettings(
            worker_id="retry-worker-b",
            gpu=None,
            once=True,
            min_free_gb=0,
            auto_retry=True,
            allow_test_jobs=True,
        ),
        paths=paths,
    )
    assert second_worker.run() == 1
    second = registry.get_job(str(retry["job_id"]))
    assert second is not None and second["status"] == "failed"
    assert first["run_id"] != second["run_id"]
    second_run = registry.get_run(str(second["run_id"]))
    assert second_run is not None
    assert second_run["retry_of"] == first["run_id"]
    assert registry.list_queue(statuses=["queued"]) == []
    summary = registry.summarize_variants(campaign_id="cmp_test")
    assert len(summary) == 1
    assert summary[0]["n_expected_runs"] == 1
    assert summary[0]["n_failed_attempts"] == 2
    assert summary[0]["n_failed_runs"] == 1


def test_variant_aggregation_uses_all_completed_runs(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    config = _config()
    variant = scientific_id(config)
    registry.register_variant(
        variant, campaign_id="cmp_test", configuration=config
    )
    for index, value in enumerate((0.6, 0.8), start=1):
        run_id = f"test_run_{index}"
        registry.create_run(
            run_id,
            campaign_id="cmp_test",
            scientific_id=variant,
            repro_id="rep_same",
            seed=index,
            fold=0,
            attempt=1,
            configuration=config,
            status="completed",
            primary_metric_name="val/loss",
            primary_metric_value=value,
        )
    summary = registry.summarize_variants(campaign_id="cmp_test")
    assert len(summary) == 1
    assert summary[0]["n_completed_runs"] == 2
    assert summary[0]["primary_mean"] == pytest.approx(0.7)
    assert summary[0]["primary_min"] == pytest.approx(0.6)
    assert summary[0]["primary_max"] == pytest.approx(0.8)
    # val/masked_huber is minimized, so the lower run is selected as best.
    assert summary[0]["best_run_id"] == "test_run_1"


def test_variant_summary_counts_unstarted_queue_plan(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    config = _config()
    variant = scientific_id(config)
    registry.register_variant(
        variant, campaign_id="cmp_test", configuration=config
    )
    for index in range(3):
        registry.enqueue(
            campaign_id="cmp_test",
            configuration={**config, "seed": index},
            command=_success_command(),
            requested_gpu=None,
        )
    summary = registry.summarize_variants(campaign_id="cmp_test")
    assert len(summary) == 1
    assert summary[0]["scientific_id"] == variant
    assert summary[0]["repro_id"] is None
    assert summary[0]["n_expected_runs"] == 3


def test_promotion_requires_registered_verified_artifacts(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    config = _config()
    variant = scientific_id(config)
    registry.register_variant(
        variant, campaign_id="cmp_test", configuration=config
    )
    registry.create_run(
        "test_unverifiable_promotion",
        campaign_id="cmp_test",
        scientific_id=variant,
        repro_id="rep_test",
        seed=1,
        fold=0,
        attempt=1,
        configuration=config,
        status="completed",
        artifact_path=tmp_path / "missing-artifacts",
    )
    with pytest.raises(RegistryError, match="no registered artifacts"):
        registry.promote_run("test_unverifiable_promotion")


def test_legacy_import_prefers_audited_index_over_manifest_rglob(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite3")
    legacy = tmp_path / "artifacts" / "legacy_runs" / "batch"
    run_a = legacy / "original" / "a"
    run_b = legacy / "original" / "b"
    decoy = legacy / "not_a_run"
    for directory in (run_a, run_b, decoy):
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
    records = []
    for name, directory in (("lr_stable_a", run_a), ("lr_stable_b", run_b)):
        manifest = directory / "manifest.json"
        records.append(
            {
                "legacy_run_id": name,
                "manifest_path": str(manifest),
                "archived_run_path": str(directory),
                "status": "complete",
                "model_family": "g1",
                "masking_type": "P+N+B",
                "dataset_id": "legacy_dataset",
                "dataset_version": "v1",
                "split_id": "split",
                "seed": 0,
                "fold": None,
                "use_edge_features": False,
                "canonical_config_sha256": name.removeprefix("lr_"),
                "manifest_sha256": None,
                "hyperparameters": {"embedding_dim": 32},
                "graph": {"neighbor_k": 4},
            }
        )
    (legacy / "index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    result = _import_legacy(
        registry, path=tmp_path / "artifacts" / "legacy_runs", campaign_id="cmp_legacy"
    )
    assert result["source_mode"] == "audited_index"
    assert result["imported"] == 2
    assert result["run_ids"] == ["lr_stable_a", "lr_stable_b"]
    assert registry.get_run("lr_stable_a") is not None
