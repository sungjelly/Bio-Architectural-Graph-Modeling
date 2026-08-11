"""Strict live-environment verification for frozen GPU campaigns."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence


_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "python",
        "python_implementation",
        "cuda_runtime",
        "packages",
        "hardware",
    }
)
_HARDWARE_KEYS = frozenset(
    {
        "launcher_visible_device_count",
        "job_visible_device_count",
        "homogeneous_device_name",
        "compute_capability",
        "minimum_total_memory_bytes_per_device",
        "job_cuda_smoke_test_required",
    }
)
_OBSERVATION_KEYS = frozenset(
    {
        "python",
        "python_implementation",
        "packages",
        "torch_version",
        "cuda_runtime",
        "cuda_available",
        "visible_device_count",
        "gpus",
        "cuda_smoke_test_passed",
    }
)
_GPU_KEYS = frozenset(
    {"logical_index", "name", "total_memory_bytes", "compute_capability"}
)
_MODES = frozenset({"launcher", "job", "analysis"})


class EnvironmentLockError(RuntimeError):
    """Raised when the lock or live software/GPU environment differs."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise EnvironmentLockError(
            f"environment lock contains non-finite JSON value {value!r}"
        )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EnvironmentLockError(
                    f"environment lock contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique,
        )
    except EnvironmentLockError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EnvironmentLockError(
            f"environment lock is not strict readable JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise EnvironmentLockError("environment lock must contain one object")
    return value


def load_environment_lock(path: str | Path) -> tuple[Path, dict[str, Any]]:
    raw = Path(path)
    if raw.is_symlink():
        raise EnvironmentLockError("environment lock may not be a symlink")
    try:
        lock_path = raw.resolve(strict=True)
    except OSError as error:
        raise EnvironmentLockError(f"environment lock is missing: {raw}") from error
    if not lock_path.is_file():
        raise EnvironmentLockError("environment lock must be a regular file")
    lock = _strict_json(lock_path)
    if set(lock) != _LOCK_KEYS or lock.get("schema_version") != 1:
        raise EnvironmentLockError(
            "environment lock top-level schema differs from version 1"
        )
    for field in ("python", "python_implementation", "cuda_runtime"):
        if not isinstance(lock[field], str) or not lock[field]:
            raise EnvironmentLockError(f"environment lock {field} must be nonempty")
    packages = lock["packages"]
    if (
        not isinstance(packages, Mapping)
        or not packages
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in packages.items()
        )
    ):
        raise EnvironmentLockError(
            "environment lock packages must map names to exact versions"
        )
    hardware = lock["hardware"]
    if not isinstance(hardware, Mapping) or set(hardware) != _HARDWARE_KEYS:
        raise EnvironmentLockError("environment lock hardware schema differs")
    for field in ("launcher_visible_device_count", "job_visible_device_count"):
        value = hardware[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EnvironmentLockError(f"environment lock hardware.{field} is invalid")
    name = hardware["homogeneous_device_name"]
    capability = hardware["compute_capability"]
    memory = hardware["minimum_total_memory_bytes_per_device"]
    if not isinstance(name, str) or not name:
        raise EnvironmentLockError("environment lock GPU name is invalid")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in capability
        )
    ):
        raise EnvironmentLockError("environment lock compute capability is invalid")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory < 1:
        raise EnvironmentLockError("environment lock GPU memory floor is invalid")
    if not isinstance(hardware["job_cuda_smoke_test_required"], bool):
        raise EnvironmentLockError("environment lock CUDA smoke-test rule is invalid")
    return lock_path, lock


def collect_live_environment(
    lock: Mapping[str, Any], *, visibility_mode: str
) -> dict[str, Any]:
    """Collect only fields declared by the lock, plus GPU compatibility facts."""

    if visibility_mode not in _MODES:
        raise EnvironmentLockError(
            f"unknown environment visibility mode: {visibility_mode}"
        )
    package_versions: dict[str, str | None] = {}
    for name in lock["packages"]:
        try:
            package_versions[str(name)] = importlib.metadata.version(str(name))
        except importlib.metadata.PackageNotFoundError:
            package_versions[str(name)] = None
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised as missing package above
        raise EnvironmentLockError("PyTorch cannot be imported") from error
    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        gpus = []
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            gpus.append(
                {
                    "logical_index": index,
                    "name": str(properties.name),
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(capability[0]), int(capability[1])],
                }
            )
        smoke_passed: bool | None = None
        if visibility_mode == "job" and lock["hardware"][
            "job_cuda_smoke_test_required"
        ]:
            if not available or count != 1:
                smoke_passed = False
            else:
                value = torch.ones(1, dtype=torch.float32, device="cuda:0")
                smoke_passed = bool((value + value).item() == 2.0)
                torch.cuda.synchronize(0)
                del value
    except Exception as error:
        raise EnvironmentLockError("live CUDA environment cannot be probed") from error
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": package_versions,
        "torch_version": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": available,
        "visible_device_count": count,
        "gpus": gpus,
        "cuda_smoke_test_passed": smoke_passed,
    }


def verify_environment_observation(
    lock: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    visibility_mode: str,
) -> None:
    """Compare a collected observation with every declared lock field."""

    if visibility_mode not in _MODES:
        raise EnvironmentLockError(
            f"unknown environment visibility mode: {visibility_mode}"
        )
    if set(observation) != _OBSERVATION_KEYS:
        raise EnvironmentLockError("live environment observation schema differs")
    exact = {
        "python": lock["python"],
        "python_implementation": lock["python_implementation"],
        "cuda_runtime": lock["cuda_runtime"],
    }
    for name, expected in exact.items():
        if observation.get(name) != expected:
            raise EnvironmentLockError(
                f"live {name} drifted: expected {expected!r}, observed {observation.get(name)!r}"
            )
    observed_packages = observation.get("packages")
    if not isinstance(observed_packages, Mapping):
        raise EnvironmentLockError("live package inventory is missing")
    if set(observed_packages) != set(lock["packages"]):
        raise EnvironmentLockError("live package inventory keys differ from the lock")
    for name, expected in lock["packages"].items():
        observed = observed_packages.get(name)
        if observed is None:
            raise EnvironmentLockError(f"locked package is missing: {name}")
        if observed != expected:
            raise EnvironmentLockError(
                f"locked package {name} drifted: expected {expected}, observed {observed}"
            )
    if observation.get("torch_version") != lock["packages"].get("torch"):
        raise EnvironmentLockError("imported torch version differs from its package lock")
    if observation.get("cuda_available") is not True:
        raise EnvironmentLockError("CUDA is not available")
    hardware = lock["hardware"]
    count_field = (
        "job_visible_device_count"
        if visibility_mode == "job"
        else "launcher_visible_device_count"
    )
    expected_count = int(hardware[count_field])
    if observation.get("visible_device_count") != expected_count:
        raise EnvironmentLockError(
            f"visible GPU count drifted for {visibility_mode}: expected {expected_count}, "
            f"observed {observation.get('visible_device_count')!r}"
        )
    gpus = observation.get("gpus")
    if not isinstance(gpus, list) or len(gpus) != expected_count:
        raise EnvironmentLockError("live GPU inventory length differs")
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, Mapping) or set(gpu) != _GPU_KEYS:
            raise EnvironmentLockError(f"live GPU record {index} is malformed")
        memory = gpu.get("total_memory_bytes")
        if (
            gpu.get("logical_index") != index
            or gpu.get("name") != hardware["homogeneous_device_name"]
            or gpu.get("compute_capability") != hardware["compute_capability"]
            or isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory < hardware["minimum_total_memory_bytes_per_device"]
        ):
            raise EnvironmentLockError(f"live GPU {index} identity/capability/memory differs")
    smoke = observation.get("cuda_smoke_test_passed")
    if visibility_mode == "job" and hardware["job_cuda_smoke_test_required"]:
        if smoke is not True:
            raise EnvironmentLockError("job CUDA smoke test did not pass")
    elif smoke is not None:
        raise EnvironmentLockError("CUDA smoke test unexpectedly ran outside job mode")


def verify_live_environment(
    lock_path: str | Path,
    *,
    visibility_mode: str,
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the lock, verify live state, and return a deterministic receipt."""

    path, lock = load_environment_lock(lock_path)
    observed = (
        collect_live_environment(lock, visibility_mode=visibility_mode)
        if observation is None
        else dict(observation)
    )
    verify_environment_observation(lock, observed, visibility_mode=visibility_mode)
    payload = {
        "schema_version": 1,
        "verified": True,
        "visibility_mode": visibility_mode,
        "environment_lock_sha256": _sha256_file(path),
        "observation": observed,
    }
    return {**payload, "verification_sha256": _canonical_sha256(payload)}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(_MODES), required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _arguments(argv)
    try:
        result = verify_live_environment(
            arguments.lock,
            visibility_mode=arguments.mode,
        )
    except EnvironmentLockError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EnvironmentLockError",
    "collect_live_environment",
    "load_environment_lock",
    "verify_environment_observation",
    "verify_live_environment",
]
