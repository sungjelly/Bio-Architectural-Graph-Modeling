from __future__ import annotations

from pathlib import Path

import pytest

from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import (
    ProjectWorkerLock,
    QueueWorker,
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
