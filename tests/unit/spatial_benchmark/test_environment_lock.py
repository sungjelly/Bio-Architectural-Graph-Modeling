from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from spatial_benchmark.environment_lock import (
    EnvironmentLockError,
    verify_live_environment,
)


def _lock() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "python": "3.12.13",
        "python_implementation": "CPython",
        "cuda_runtime": "13.0",
        "packages": {
            "numpy": "2.4.6",
            "torch": "2.12.0+cu130",
        },
        "hardware": {
            "launcher_visible_device_count": 4,
            "job_visible_device_count": 1,
            "homogeneous_device_name": "NVIDIA GeForce RTX 3090",
            "compute_capability": [8, 6],
            "minimum_total_memory_bytes_per_device": 25_000_000_000,
            "job_cuda_smoke_test_required": True,
        },
    }


def _observation(*, mode: str = "launcher") -> dict[str, Any]:
    count = 1 if mode == "job" else 4
    return {
        "python": "3.12.13",
        "python_implementation": "CPython",
        "packages": {
            "numpy": "2.4.6",
            "torch": "2.12.0+cu130",
        },
        "torch_version": "2.12.0+cu130",
        "cuda_runtime": "13.0",
        "cuda_available": True,
        "visible_device_count": count,
        "gpus": [
            {
                "logical_index": index,
                "name": "NVIDIA GeForce RTX 3090",
                "total_memory_bytes": 25_298_141_184,
                "compute_capability": [8, 6],
            }
            for index in range(count)
        ],
        "cuda_smoke_test_passed": True if mode == "job" else None,
    }


def _write_lock(path: Path, payload: dict[str, Any] | None = None) -> None:
    path.write_text(
        json.dumps(_lock() if payload is None else payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_exact_launcher_observation_produces_a_deterministic_receipt(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "environment_lock.json"
    _write_lock(lock_path)

    first = verify_live_environment(
        lock_path,
        visibility_mode="launcher",
        observation=_observation(),
    )
    second = verify_live_environment(
        lock_path,
        visibility_mode="launcher",
        observation=_observation(),
    )

    assert first == second
    assert first["verified"] is True
    assert first["visibility_mode"] == "launcher"
    assert len(first["environment_lock_sha256"]) == 64
    assert len(first["verification_sha256"]) == 64


def test_job_mode_requires_one_visible_gpu_and_the_cuda_smoke_test(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "environment_lock.json"
    _write_lock(lock_path)
    receipt = verify_live_environment(
        lock_path,
        visibility_mode="job",
        observation=_observation(mode="job"),
    )
    assert receipt["observation"]["cuda_smoke_test_passed"] is True

    missing_smoke = _observation(mode="job")
    missing_smoke["cuda_smoke_test_passed"] = False
    with pytest.raises(EnvironmentLockError, match="smoke test"):
        verify_live_environment(
            lock_path,
            visibility_mode="job",
            observation=missing_smoke,
        )


def test_exact_package_version_drift_fails_closed(tmp_path: Path) -> None:
    lock_path = tmp_path / "environment_lock.json"
    _write_lock(lock_path)
    observation = _observation()
    observation["packages"]["numpy"] = "2.4.5"

    with pytest.raises(EnvironmentLockError, match="numpy drifted"):
        verify_live_environment(
            lock_path,
            visibility_mode="launcher",
            observation=observation,
        )


def test_missing_locked_package_fails_closed(tmp_path: Path) -> None:
    lock_path = tmp_path / "environment_lock.json"
    _write_lock(lock_path)
    observation = _observation()
    observation["packages"]["numpy"] = None

    with pytest.raises(EnvironmentLockError, match="package is missing: numpy"):
        verify_live_environment(
            lock_path,
            visibility_mode="launcher",
            observation=observation,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("count", "GPU count"),
        ("name", "identity/capability/memory"),
        ("capability", "identity/capability/memory"),
        ("memory", "identity/capability/memory"),
    ],
)
def test_gpu_identity_count_capability_and_memory_drift_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    lock_path = tmp_path / "environment_lock.json"
    _write_lock(lock_path)
    observation = _observation()
    if mutation == "count":
        observation["visible_device_count"] = 3
    elif mutation == "name":
        observation["gpus"][0]["name"] = "different GPU"
    elif mutation == "capability":
        observation["gpus"][0]["compute_capability"] = [8, 0]
    else:
        observation["gpus"][0]["total_memory_bytes"] = 24_999_999_999

    with pytest.raises(EnvironmentLockError, match=message):
        verify_live_environment(
            lock_path,
            visibility_mode="launcher",
            observation=observation,
        )


def test_lock_schema_is_exact_and_rejects_unknown_fields(tmp_path: Path) -> None:
    lock_path = tmp_path / "environment_lock.json"
    payload = deepcopy(_lock())
    payload["unfrozen_extra"] = True
    _write_lock(lock_path, payload)

    with pytest.raises(EnvironmentLockError, match="top-level schema"):
        verify_live_environment(
            lock_path,
            visibility_mode="launcher",
            observation=_observation(),
        )
