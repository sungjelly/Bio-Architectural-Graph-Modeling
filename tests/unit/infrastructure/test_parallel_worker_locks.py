from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import spatial_benchmark.queueing as queueing
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import (
    ProjectWorkerLock,
    QueueWorker,
    QueueWorkerError,
    WorkerSettings,
)
from spatial_benchmark.registry import Registry


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths.from_environment({"BAGM_ROOT": str(root)})


def _registry(paths: ProjectPaths) -> Registry:
    return Registry(paths.state_root / "tracking" / "bagm.sqlite3")


def test_explicit_gpu_parallel_workers_have_disjoint_lock_files(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    registry = _registry(paths)
    first = QueueWorker(
        registry,
        settings=WorkerSettings(
            worker_id="gpu-0",
            gpu="0",
            parallel_gpu_workers=True,
        ),
        paths=paths,
    )
    second = QueueWorker(
        registry,
        settings=WorkerSettings(
            worker_id="gpu-1",
            gpu="1",
            parallel_gpu_workers=True,
        ),
        paths=paths,
    )

    assert first.lock_path.name == "bagm-worker-gpu-0.lock"
    assert second.lock_path.name == "bagm-worker-gpu-1.lock"
    with ProjectWorkerLock(first.lock_path), ProjectWorkerLock(
        second.lock_path
    ):
        assert first.lock_path.is_file()
        assert second.lock_path.is_file()


def test_default_worker_retains_project_global_lock(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    worker = QueueWorker(
        _registry(paths),
        settings=WorkerSettings(worker_id="legacy", gpu="0"),
        paths=paths,
    )
    assert worker.lock_path.name == "bagm-worker.lock"


@pytest.mark.parametrize("gpu", [None, "", "0,1"])
def test_parallel_worker_requires_one_explicit_gpu(gpu: str | None) -> None:
    settings = WorkerSettings(
        worker_id="invalid",
        gpu=gpu,
        parallel_gpu_workers=True,
    )
    with pytest.raises(ValueError, match="one explicit GPU|exactly one GPU"):
        settings.validate()


def _worker(
    tmp_path: Path, *, gpu: str = "1", parallel: bool = True
) -> QueueWorker:
    paths = _paths(tmp_path)
    return QueueWorker(
        _registry(paths),
        settings=WorkerSettings(
            worker_id=f"gpu-{gpu}",
            gpu=gpu,
            parallel_gpu_workers=parallel,
        ),
        paths=paths,
    )


def _write_child_record(
    worker: QueueWorker,
    *,
    job_id: str = "q_legacy",
    pid: int = 1234,
    start_ticks: int = 5678,
    requested_gpu: object = ...,
) -> Path:
    directory = worker.paths.state_root / "pids"
    directory.mkdir(parents=True, exist_ok=True)
    record = {
        "job_id": job_id,
        "run_id": "r_test",
        "pid": pid,
        "proc_start_ticks": start_ticks,
    }
    if requested_gpu is not ...:
        record["requested_gpu"] = requested_gpu
    path = directory / f"{job_id}.json"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_new_child_record_includes_worker_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path, gpu="2")
    worker._child = SimpleNamespace(pid=4321)  # type: ignore[assignment]
    monkeypatch.setattr(queueing, "_process_start_ticks", lambda _pid: 8765)

    worker._record_child_process(
        job_id="q_new", run_id="r_new", command=["python", "train.py"]
    )

    record = json.loads(
        (worker.paths.state_root / "pids" / "q_new.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["requested_gpu"] == "2"


def test_parallel_worker_ignores_live_child_on_other_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path, gpu="1")
    record = _write_child_record(worker, requested_gpu="0")
    monkeypatch.setattr(queueing, "_process_start_ticks", lambda _pid: 5678)

    worker._assert_no_surviving_children()

    assert record.is_file()


def test_parallel_worker_resolves_legacy_other_gpu_from_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path, gpu="1")
    record = _write_child_record(worker, job_id="q_old")
    monkeypatch.setattr(
        worker.registry,
        "get_job",
        lambda job_id: {"job_id": job_id, "requested_gpu": "0"},
    )
    monkeypatch.setattr(queueing, "_process_start_ticks", lambda _pid: 5678)

    worker._assert_no_surviving_children()

    assert record.is_file()


@pytest.mark.parametrize("legacy", [False, True])
def test_parallel_worker_blocks_live_child_on_same_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    worker = _worker(tmp_path, gpu="1")
    _write_child_record(
        worker, requested_gpu=... if legacy else "1"
    )
    if legacy:
        monkeypatch.setattr(
            worker.registry,
            "get_job",
            lambda job_id: {"job_id": job_id, "requested_gpu": "1"},
        )
    monkeypatch.setattr(queueing, "_process_start_ticks", lambda _pid: 5678)

    with pytest.raises(QueueWorkerError, match="is still alive"):
        worker._assert_no_surviving_children()


def test_parallel_worker_blocks_unresolved_legacy_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path, gpu="1")
    _write_child_record(worker)
    monkeypatch.setattr(worker.registry, "get_job", lambda _job_id: None)
    monkeypatch.setattr(queueing, "_process_start_ticks", lambda _pid: 5678)

    with pytest.raises(QueueWorkerError, match="no resolvable GPU assignment"):
        worker._assert_no_surviving_children()


def test_default_worker_retains_global_surviving_child_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(tmp_path, gpu="0", parallel=False)
    _write_child_record(worker, requested_gpu="1")
    monkeypatch.setattr(queueing, "_process_start_ticks", lambda _pid: 5678)

    with pytest.raises(QueueWorkerError, match="is still alive"):
        worker._assert_no_surviving_children()
