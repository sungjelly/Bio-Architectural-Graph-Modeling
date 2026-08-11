#!/usr/bin/env python3
"""Run a checksum-bound same-gene job plan on four isolated GPU lanes.

The launcher is deliberately ignorant of scientific results.  It schedules
already-materialized argv vectors, captures each job's logs, and considers a
job successful only when its process exits zero and its declared completion
marker (plus an optional verifier command) validates.

Plan schema (all relative paths are resolved under ``working_directory``)::

    {
      "schema_version": 1,
      "plan_id": "same_gene_robustness_v1",
      "working_directory": ".",
      "source_manifest": {"path": "...", "sha256": "<64 hex>"},
      "disk_path": ".",
      "minimum_free_disk_gb": 40,
      "data_preflight_argv": ["/venv/main/bin/python", "...", "--verify-only"],
      "jobs": [
        {
          "job_id": "baseline_s810_f0",
          "argv": ["/venv/main/bin/python", "...", "--config", "..."],
          "gpu": "auto",
          "stdout_path": "state/logs/...stdout.log",
          "stderr_path": "state/logs/...stderr.log",
          "expected_success_marker": "artifacts/.../_SUCCESS",
          "expected_config_sha256": "<64 hex>",
          "verify_argv": ["/venv/main/bin/python", "...", "--verify-only"]
        }
      ]
    }

``gpu`` may be an integer from 0 through 3, ``"auto"``, or JSON null.
``verify_argv`` is optional.  The coordinator owns one global non-blocking
``flock`` and never invokes a shell.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, BinaryIO, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = str(PROJECT_ROOT / "src")
if not sys.path or sys.path[0] != _SOURCE_ROOT:
    sys.path.insert(0, _SOURCE_ROOT)

from spatial_benchmark.environment_lock import (
    EnvironmentLockError,
    verify_live_environment,
)


GPU_IDS = (0, 1, 2, 3)
MINIMUM_SAFE_FREE_DISK_GB = 40.0
DEFAULT_POLL_SECONDS = 0.2
JOB_VERIFIER_TIMEOUT_SECONDS = 1800
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PLAN_KEYS = {
    "schema_version",
    "plan_id",
    "working_directory",
    "source_manifest",
    "disk_path",
    "minimum_free_disk_gb",
    "projected_output_bytes",
    "data_preflight_argv",
    "environment_lock",
    "jobs",
}
_JOB_KEYS = {
    "job_id",
    "argv",
    "gpu",
    "stdout_path",
    "stderr_path",
    "expected_success_marker",
    "expected_config_sha256",
    "verify_argv",
}
_SUCCESS_STATUSES = frozenset({"completed", "skipped"})
_RESUMABLE_STATUSES = frozenset(
    {
        "launching",
        "running",
        "completed",
        "skipped",
        "failed",
        "interrupted",
        "retry_required",
    }
)


class RobustnessLauncherError(RuntimeError):
    """Base error for a malformed plan or unsafe launch state."""


class PlanValidationError(RobustnessLauncherError):
    """Raised when the materialized plan is not safe and self-consistent."""


class PreflightError(RobustnessLauncherError):
    """Raised when source, configuration, or disk preflight fails."""


class LauncherLockHeldError(RobustnessLauncherError):
    """Raised when another coordinator owns the global campaign lock."""


class LiveChildError(RobustnessLauncherError):
    """Raised when a prior coordinator may still have a live child."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise PlanValidationError(f"{label} contains non-finite JSON value {value!r}")

    def unique_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PlanValidationError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_mapping,
        )
    except PlanValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PlanValidationError(f"{label} is not strict readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise PlanValidationError(f"{label} must contain a JSON object")
    return value


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise PlanValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _argv(value: Any, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise PlanValidationError(f"{label} must be a nonempty list of strings")
    if "\x00" in "".join(value):
        raise PlanValidationError(f"{label} may not contain NUL bytes")
    return tuple(value)


def _resolve_under(
    project_root: Path,
    value: Any,
    *,
    label: str,
    base: Path | None = None,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{label} must be a nonempty path string")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else (base or project_root) / raw
    if candidate.is_symlink():
        raise PlanValidationError(f"{label} may not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(project_root)
    except ValueError as error:
        raise PlanValidationError(
            f"{label} must remain under project root {project_root}: {candidate}"
        ) from error
    if require_file and (not resolved.is_file() or resolved.is_symlink()):
        raise PlanValidationError(f"{label} is not a regular file: {resolved}")
    if require_directory and (not resolved.is_dir() or resolved.is_symlink()):
        raise PlanValidationError(f"{label} is not a directory: {resolved}")
    return resolved


def _config_argument(argv: Sequence[str]) -> str:
    values: list[str] = []
    for index, value in enumerate(argv):
        if value == "--config":
            if index + 1 >= len(argv):
                raise PlanValidationError("job argv ends after --config")
            values.append(str(argv[index + 1]))
        elif value.startswith("--config="):
            values.append(value.split("=", 1)[1])
    if len(values) != 1 or not values[0]:
        raise PlanValidationError(
            "each job argv must contain exactly one --config path"
        )
    return values[0]


def _gpu(value: Any, *, label: str) -> int | None:
    if value is None or (
        isinstance(value, str) and value in {"auto", "automatic"}
    ):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value not in GPU_IDS:
        raise PlanValidationError(f"{label} must be 0, 1, 2, 3, auto, or null")
    return value


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    argv: tuple[str, ...]
    requested_gpu: int | None
    stdout_path: Path
    stderr_path: Path
    success_marker: Path
    expected_config_sha256: str
    config_path: Path
    verify_argv: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class MaterializedPlan:
    path: Path
    sha256: str
    plan_id: str
    project_root: Path
    working_directory: Path
    source_manifest_path: Path
    source_manifest_sha256: str
    disk_path: Path
    minimum_free_disk_gb: float
    projected_output_bytes: int
    data_preflight_argv: tuple[str, ...] | None
    environment_lock_path: Path
    environment_lock_sha256: str
    jobs: tuple[Job, ...]


@dataclass(slots=True)
class RunningJob:
    job: Job
    gpu: int
    process: subprocess.Popen[bytes]
    stdout_handle: BinaryIO
    stderr_handle: BinaryIO
    marker_before: tuple[int, int, int, int, int, str] | None


def load_plan(path: str | Path, *, project_root: str | Path = PROJECT_ROOT) -> MaterializedPlan:
    plan_path = Path(path).resolve(strict=True)
    root = Path(project_root).resolve(strict=True)
    if not root.is_dir():
        raise PlanValidationError(f"project root is not a directory: {root}")
    payload = _strict_json(plan_path, label="materialized job plan")
    unknown = set(payload).difference(_PLAN_KEYS)
    if unknown:
        raise PlanValidationError(f"unknown materialized plan keys: {sorted(unknown)}")
    required = _PLAN_KEYS.difference(
        {
            "working_directory",
            "disk_path",
            "data_preflight_argv",
            "projected_output_bytes",
        }
    )
    missing = required.difference(payload)
    if missing:
        raise PlanValidationError(f"materialized plan is missing keys: {sorted(missing)}")
    if payload.get("schema_version") != 1:
        raise PlanValidationError("materialized plan schema_version must equal 1")
    plan_id = payload.get("plan_id")
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise PlanValidationError("materialized plan plan_id must be nonempty")
    working_directory = _resolve_under(
        root,
        payload.get("working_directory", "."),
        label="working_directory",
        require_directory=True,
    )
    disk_path = _resolve_under(
        root,
        payload.get("disk_path", "."),
        label="disk_path",
        require_directory=True,
    )
    minimum_raw = payload.get("minimum_free_disk_gb")
    if isinstance(minimum_raw, bool) or not isinstance(minimum_raw, (int, float)):
        raise PlanValidationError("minimum_free_disk_gb must be numeric")
    minimum = float(minimum_raw)
    if not math.isfinite(minimum) or minimum < MINIMUM_SAFE_FREE_DISK_GB:
        raise PlanValidationError(
            f"minimum_free_disk_gb must be at least {MINIMUM_SAFE_FREE_DISK_GB:g}"
        )
    projected_raw = payload.get("projected_output_bytes", 0)
    if (
        isinstance(projected_raw, bool)
        or not isinstance(projected_raw, int)
        or projected_raw < 0
    ):
        raise PlanValidationError("projected_output_bytes must be a nonnegative integer")
    source = payload.get("source_manifest")
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise PlanValidationError(
            "source_manifest must contain exactly path and sha256"
        )
    source_path = _resolve_under(
        root,
        source.get("path"),
        label="source_manifest.path",
        base=working_directory,
        require_file=True,
    )
    source_sha = _sha256(source.get("sha256"), label="source_manifest.sha256")
    data_preflight_raw = payload.get("data_preflight_argv")
    data_preflight_argv = (
        None
        if data_preflight_raw is None
        else _argv(data_preflight_raw, label="data_preflight_argv")
    )
    environment = payload.get("environment_lock")
    if not isinstance(environment, Mapping) or set(environment) != {"path", "sha256"}:
        raise PlanValidationError(
            "environment_lock must contain exactly path and sha256"
        )
    environment_lock_path = _resolve_under(
        root,
        environment.get("path"),
        label="environment_lock.path",
        base=working_directory,
        require_file=True,
    )
    environment_lock_sha = _sha256(
        environment.get("sha256"), label="environment_lock.sha256"
    )

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise PlanValidationError("materialized plan jobs must be a nonempty list")
    jobs: list[Job] = []
    seen_ids: set[str] = set()
    output_paths: set[Path] = set()
    marker_paths: set[Path] = set()
    for index, raw_job in enumerate(raw_jobs):
        label = f"jobs[{index}]"
        if not isinstance(raw_job, Mapping):
            raise PlanValidationError(f"{label} must be an object")
        unknown_job = set(raw_job).difference(_JOB_KEYS)
        if unknown_job:
            raise PlanValidationError(f"{label} has unknown keys: {sorted(unknown_job)}")
        missing_job = _JOB_KEYS.difference({"verify_argv"}).difference(raw_job)
        if missing_job:
            raise PlanValidationError(f"{label} is missing keys: {sorted(missing_job)}")
        job_id = raw_job.get("job_id")
        if not isinstance(job_id, str) or _SAFE_JOB_ID.fullmatch(job_id) is None:
            raise PlanValidationError(f"{label}.job_id is not safe")
        if job_id in seen_ids:
            raise PlanValidationError(f"duplicate job_id: {job_id}")
        seen_ids.add(job_id)
        argv = _argv(raw_job.get("argv"), label=f"{label}.argv")
        stdout_path = _resolve_under(
            root,
            raw_job.get("stdout_path"),
            label=f"{label}.stdout_path",
            base=working_directory,
        )
        stderr_path = _resolve_under(
            root,
            raw_job.get("stderr_path"),
            label=f"{label}.stderr_path",
            base=working_directory,
        )
        marker = _resolve_under(
            root,
            raw_job.get("expected_success_marker"),
            label=f"{label}.expected_success_marker",
            base=working_directory,
        )
        if stdout_path == stderr_path:
            raise PlanValidationError(f"{label} stdout and stderr paths must differ")
        if stdout_path in output_paths or stderr_path in output_paths:
            raise PlanValidationError("job stdout/stderr paths must be globally unique")
        output_paths.update((stdout_path, stderr_path))
        if marker in marker_paths:
            raise PlanValidationError("expected success markers must be unique")
        marker_paths.add(marker)
        config_value = _config_argument(argv)
        config_path = _resolve_under(
            root,
            config_value,
            label=f"{label} --config",
            base=working_directory,
            require_file=True,
        )
        verify_raw = raw_job.get("verify_argv")
        verify = (
            None
            if verify_raw is None
            else _argv(verify_raw, label=f"{label}.verify_argv")
        )
        jobs.append(
            Job(
                job_id=job_id,
                argv=argv,
                requested_gpu=_gpu(raw_job.get("gpu"), label=f"{label}.gpu"),
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                success_marker=marker,
                expected_config_sha256=_sha256(
                    raw_job.get("expected_config_sha256"),
                    label=f"{label}.expected_config_sha256",
                ),
                config_path=config_path,
                verify_argv=verify,
            )
        )
    overlap = output_paths.intersection(marker_paths)
    if overlap:
        raise PlanValidationError(
            "stdout/stderr paths and success markers must be disjoint: "
            f"{sorted(map(str, overlap))}"
        )
    write_targets = output_paths.union(marker_paths)
    read_authorities = {
        plan_path,
        source_path,
        environment_lock_path,
        *(job.config_path for job in jobs),
    }
    authority_overlap = write_targets.intersection(read_authorities)
    if authority_overlap:
        raise PlanValidationError(
            "job outputs may not overwrite the plan, source manifest, or configs: "
            f"{sorted(map(str, authority_overlap))}"
        )
    return MaterializedPlan(
        path=plan_path,
        sha256=_sha256_file(plan_path),
        plan_id=plan_id,
        project_root=root,
        working_directory=working_directory,
        source_manifest_path=source_path,
        source_manifest_sha256=source_sha,
        disk_path=disk_path,
        minimum_free_disk_gb=minimum,
        projected_output_bytes=projected_raw,
        data_preflight_argv=data_preflight_argv,
        environment_lock_path=environment_lock_path,
        environment_lock_sha256=environment_lock_sha,
        jobs=tuple(jobs),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.writing-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


class GlobalLauncherLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> "GlobalLauncherLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise LauncherLockHeldError(f"launcher lock may not be a symlink: {self.path}")
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._handle.close()
            self._handle = None
            raise LauncherLockHeldError(
                f"another robustness coordinator owns {self.path}"
            ) from error
        self._handle.seek(0)
        self._handle.truncate()
        json.dump(
            {"pid": os.getpid(), "host": socket.gethostname(), "acquired_at": _utc_now()},
            self._handle,
            sort_keys=True,
        )
        self._handle.write("\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


def _process_start_ticks(pid: int) -> int | None:
    try:
        content = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        tail = content[content.rindex(")") + 2 :].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


class FourGpuCoordinator:
    """Coordinate one subprocess per GPU for an immutable materialized plan."""

    def __init__(
        self,
        plan: MaterializedPlan,
        *,
        ledger_path: str | Path,
        lock_path: str | Path,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self.plan = plan
        self.ledger_path = _resolve_under(
            plan.project_root,
            str(ledger_path),
            label="ledger_path",
            base=plan.working_directory,
        )
        self.lock_path = _resolve_under(
            plan.project_root,
            str(lock_path),
            label="lock_path",
            base=plan.working_directory,
        )
        if self.ledger_path == self.lock_path:
            raise PlanValidationError("ledger and lock paths must differ")
        job_write_targets = {
            path
            for job in plan.jobs
            for path in (job.stdout_path, job.stderr_path, job.success_marker)
        }
        launcher_targets = {self.ledger_path, self.lock_path}
        collision = job_write_targets.intersection(launcher_targets)
        if collision:
            raise PlanValidationError(
                "ledger/lock paths may not overlap job write targets: "
                f"{sorted(map(str, collision))}"
            )
        if not math.isfinite(poll_seconds) or poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive and finite")
        self.poll_seconds = float(poll_seconds)
        self._stop = threading.Event()
        self._signal: int | None = None
        self._active: dict[int, RunningJob] = {}
        self._ledger: dict[str, Any] = {}
        self._resuming = False

    def request_stop(self, signum: int = signal.SIGTERM) -> None:
        self._signal = int(signum)
        self._stop.set()
        for running in list(self._active.values()):
            if running.process.poll() is None:
                running.process.terminate()

    def _signal_handler(self, signum: int, _frame: object) -> None:
        self.request_stop(signum)

    def _initial_ledger(self) -> dict[str, Any]:
        now = _utc_now()
        return {
            "schema_version": 1,
            "plan_id": self.plan.plan_id,
            "plan_path": str(self.plan.path),
            "plan_sha256": self.plan.sha256,
            "source_manifest": {
                "path": str(self.plan.source_manifest_path),
                "sha256": self.plan.source_manifest_sha256,
            },
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "jobs": {
                job.job_id: {"status": "pending", "attempts": []}
                for job in self.plan.jobs
            },
        }

    def _load_ledger(self) -> None:
        if not self.ledger_path.exists():
            self._resuming = False
            self._ledger = self._initial_ledger()
            self._write_ledger()
            return
        self._resuming = True
        if self.ledger_path.is_symlink() or not self.ledger_path.is_file():
            raise PlanValidationError(f"ledger is not a regular file: {self.ledger_path}")
        ledger = _strict_json(self.ledger_path, label="launcher ledger")
        if (
            ledger.get("schema_version") != 1
            or ledger.get("plan_id") != self.plan.plan_id
            or ledger.get("plan_sha256") != self.plan.sha256
        ):
            raise PlanValidationError("ledger is not bound to the exact materialized plan")
        states = ledger.get("jobs")
        expected = {job.job_id for job in self.plan.jobs}
        if not isinstance(states, dict) or set(states) != expected:
            raise PlanValidationError("ledger job inventory differs from the plan")
        for job_id, state in states.items():
            if not isinstance(state, dict) or not isinstance(state.get("attempts"), list):
                raise PlanValidationError(f"ledger state is malformed for {job_id}")
        self._ledger = ledger

    def _write_ledger(self) -> None:
        self._ledger["updated_at"] = _utc_now()
        _atomic_json(self.ledger_path, self._ledger)

    def _verify_source_manifest(self) -> None:
        path = self.plan.source_manifest_path
        if not path.is_file() or path.is_symlink():
            raise PreflightError(f"source manifest is missing or unsafe: {path}")
        observed = _sha256_file(path)
        if observed != self.plan.source_manifest_sha256:
            raise PreflightError(
                "source manifest SHA changed: "
                f"expected {self.plan.source_manifest_sha256}, observed {observed}"
            )

    def _verify_config(self, job: Job) -> None:
        if not job.config_path.is_file() or job.config_path.is_symlink():
            raise PreflightError(f"job {job.job_id} config is missing or unsafe")
        observed = _sha256_file(job.config_path)
        if observed != job.expected_config_sha256:
            raise PreflightError(
                f"job {job.job_id} config SHA changed: expected "
                f"{job.expected_config_sha256}, observed {observed}"
            )

    def _remaining_projected_output_bytes(self) -> int:
        """Conservatively scale the frozen projection to unfinished jobs."""

        if self.plan.projected_output_bytes == 0:
            return 0
        states = self._ledger.get("jobs")
        if not isinstance(states, Mapping) or len(states) != len(self.plan.jobs):
            return self.plan.projected_output_bytes
        unfinished_statuses = {"pending", "launching", "running"}
        known_statuses = unfinished_statuses.union(
            {
                "completed",
                "skipped",
                "failed",
                "interrupted",
                "retry_required",
            }
        )
        observed_statuses = {str(state.get("status")) for state in states.values()}
        if not observed_statuses.issubset(known_statuses):
            return self.plan.projected_output_bytes
        unfinished = sum(
            str(state.get("status")) in unfinished_statuses
            for state in states.values()
        )
        return math.ceil(
            self.plan.projected_output_bytes * unfinished / len(self.plan.jobs)
        )

    def _verify_disk(self, *, projected_output_bytes: int | None = None) -> None:
        free = shutil.disk_usage(self.plan.disk_path).free / (1024**3)
        projection = (
            self.plan.projected_output_bytes
            if projected_output_bytes is None
            else projected_output_bytes
        )
        projected = projection / (1024**3)
        ending_free = free - projected
        if ending_free < self.plan.minimum_free_disk_gb:
            raise PreflightError(
                f"free disk {free:.2f} GiB minus projected outputs "
                f"{projected:.2f} GiB leaves {ending_free:.2f} GiB, below the "
                f"frozen {self.plan.minimum_free_disk_gb:.2f} GiB floor"
            )

    def _verify_environment(self) -> None:
        path = self.plan.environment_lock_path
        if not path.is_file() or path.is_symlink():
            raise PreflightError(f"environment lock is missing or unsafe: {path}")
        observed = _sha256_file(path)
        if observed != self.plan.environment_lock_sha256:
            raise PreflightError(
                "environment lock SHA changed: "
                f"expected {self.plan.environment_lock_sha256}, observed {observed}"
            )
        try:
            report = verify_live_environment(path, visibility_mode="launcher")
        except EnvironmentLockError as error:
            raise PreflightError(
                "live environment differs from the frozen environment lock"
            ) from error
        self._ledger["environment_verification"] = report

    def _verify_prepared_data(self) -> None:
        if self.plan.data_preflight_argv is None:
            return
        try:
            result = subprocess.run(
                list(self.plan.data_preflight_argv),
                cwd=self.plan.working_directory,
                env=self._verification_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                check=False,
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PreflightError(
                "prepared-data verification could not complete"
            ) from error
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-2000:]
            raise PreflightError(
                "prepared-data verification failed before GPU launch: " + stderr
            )

    def _preflight(self, job: Job | None = None) -> None:
        self._verify_source_manifest()
        self._verify_disk(
            projected_output_bytes=(
                self._remaining_projected_output_bytes()
                if self._resuming or job is not None
                else self.plan.projected_output_bytes
            )
        )
        self._verify_environment()
        if job is None:
            self._verify_prepared_data()
            for planned in self.plan.jobs:
                self._verify_config(planned)
        else:
            self._verify_config(job)

    def _verification_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(self.plan.project_root / "src"),
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
            }
        )
        return environment

    def _job_verification_environment(self, gpu: int) -> dict[str, str]:
        if isinstance(gpu, bool) or gpu not in GPU_IDS:
            raise PreflightError(f"job verifier GPU assignment is invalid: {gpu!r}")
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(self.plan.project_root / "src"),
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
            }
        )
        return environment

    def _success_contract(self, job: Job, *, gpu: int | None) -> tuple[bool, str]:
        marker = job.success_marker
        if job.verify_argv is None:
            return (
                (True, "marker_present_no_optional_verifier")
                if marker.is_file() and not marker.is_symlink()
                else (False, "expected_success_marker_missing")
            )
        if isinstance(gpu, bool) or gpu not in GPU_IDS:
            return False, "optional_verifier_gpu_assignment_missing"
        try:
            result = subprocess.run(
                list(job.verify_argv),
                cwd=self.plan.working_directory,
                env=self._job_verification_environment(gpu),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                check=False,
                timeout=JOB_VERIFIER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return False, "optional_verifier_timed_out"
        except OSError:
            return False, "optional_verifier_could_not_start"
        if result.returncode != 0:
            return False, f"optional_verifier_exit_{result.returncode}"
        if not marker.is_file() or marker.is_symlink():
            return False, "optional_verifier_passed_without_safe_marker"
        return True, "marker_and_optional_verifier_passed"

    @staticmethod
    def _marker_fingerprint(
        marker: Path,
    ) -> tuple[int, int, int, int, int, str] | None:
        if marker.is_symlink():
            raise PreflightError(f"success marker may not be a symlink: {marker}")
        if not marker.exists():
            return None
        if not marker.is_file():
            raise PreflightError(f"success marker is not a regular file: {marker}")
        metadata = marker.stat()
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
            _sha256_file(marker),
        )

    def _prepare_resume(self) -> list[Job]:
        pending: list[Job] = []
        states = self._ledger["jobs"]
        for job in self.plan.jobs:
            state = states[job.job_id]
            prior_status = state.get("status")
            if prior_status in {"launching", "running"}:
                pid = state.get("pid")
                ticks = state.get("proc_start_ticks")
                if isinstance(pid, int):
                    observed_ticks = _process_start_ticks(pid)
                    if observed_ticks is not None and (
                        not isinstance(ticks, int) or observed_ticks == ticks
                    ):
                        raise LiveChildError(
                            f"prior child for {job.job_id} is still alive as PID {pid}"
                        )
                attempts = state["attempts"]
                if attempts and isinstance(attempts[-1], dict):
                    attempts[-1].update(
                        {"status": "abandoned", "finished_at": _utc_now()}
                    )
            attempts = state["attempts"]
            reusable = (
                self._resuming
                and prior_status in _RESUMABLE_STATUSES
                and bool(attempts)
            )
            assigned_gpu: int | None = None
            if attempts and isinstance(attempts[-1], Mapping):
                attempt_gpu = attempts[-1].get("gpu")
                state_gpu = state.get("gpu")
                if (
                    isinstance(attempt_gpu, int)
                    and not isinstance(attempt_gpu, bool)
                    and attempt_gpu in GPU_IDS
                    and (state_gpu is None or state_gpu == attempt_gpu)
                ):
                    assigned_gpu = attempt_gpu
            passed, reason = (
                self._success_contract(job, gpu=assigned_gpu)
                if reusable
                else (False, "pristine_attempt_has_not_been_launched")
            )
            if passed:
                state.update(
                    {
                        "status": "skipped",
                        "resume_reason": reason,
                        "finished_at": state.get("finished_at") or _utc_now(),
                    }
                )
                for key in ("pid", "proc_start_ticks", "gpu"):
                    state.pop(key, None)
            elif not attempts and prior_status == "pending":
                state.update({"status": "pending", "resume_reason": reason})
                for key in ("pid", "proc_start_ticks", "gpu", "finished_at"):
                    state.pop(key, None)
                pending.append(job)
            elif (
                attempts
                and isinstance(attempts[-1], Mapping)
                and (
                    attempts[-1].get("status") == "failed_to_start"
                    or reason == "optional_verifier_exit_75"
                )
            ):
                state.update(
                    {
                        "status": "pending",
                        "resume_reason": "scientific_attempt_unclaimed_safe_relaunch",
                    }
                )
                for key in ("pid", "proc_start_ticks", "gpu", "finished_at"):
                    state.pop(key, None)
                pending.append(job)
            else:
                state.update(
                    {
                        "status": "retry_required",
                        "resume_reason": reason,
                        "finished_at": state.get("finished_at") or _utc_now(),
                    }
                )
                for key in ("pid", "proc_start_ticks", "gpu"):
                    state.pop(key, None)
        self._write_ledger()
        return pending

    @staticmethod
    def _compatible(job: Job, gpu: int) -> bool:
        return job.requested_gpu is None or job.requested_gpu == gpu

    def _next_for_gpu(self, pending: list[Job], gpu: int) -> Job | None:
        for index, job in enumerate(pending):
            if self._compatible(job, gpu):
                return pending.pop(index)
        return None

    @staticmethod
    def _open_log(path: Path) -> BinaryIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise PreflightError(f"log path may not be a symlink: {path}")
        return path.open("ab", buffering=0)

    def _launch(self, job: Job, gpu: int) -> None:
        self._preflight(job)
        marker_before = self._marker_fingerprint(job.success_marker)
        stdout_handle = self._open_log(job.stdout_path)
        try:
            stderr_handle = self._open_log(job.stderr_path)
        except BaseException:
            stdout_handle.close()
            raise
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": str(self.plan.project_root / "src"),
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "8",
                "PYTHONUNBUFFERED": "1",
            }
        )
        state = self._ledger["jobs"][job.job_id]
        attempt = {
            "number": len(state["attempts"]) + 1,
            "status": "launching",
            "gpu": gpu,
            "started_at": _utc_now(),
        }
        state["attempts"].append(attempt)
        state.update(
            {
                "status": "launching",
                "gpu": gpu,
                "started_at": attempt["started_at"],
            }
        )
        self._write_ledger()
        try:
            process = subprocess.Popen(
                list(job.argv),
                cwd=self.plan.working_directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
            )
        except BaseException as error:
            stdout_handle.close()
            stderr_handle.close()
            finished = _utc_now()
            attempt.update(
                {
                    "status": "failed_to_start",
                    "finished_at": finished,
                    "failure_type": type(error).__name__,
                }
            )
            state.update(
                {
                    "status": "retry_required",
                    "finished_at": finished,
                    "success_contract": "process_could_not_start",
                }
            )
            self._write_ledger()
            raise
        ticks = _process_start_ticks(process.pid)
        if ticks is None and process.poll() is None:
            process.terminate()
            process.wait()
            stdout_handle.close()
            stderr_handle.close()
            raise PreflightError(
                f"could not bind job {job.job_id} to a stable child identity"
            )
        attempt.update(
            {
                "status": "running",
                "pid": process.pid,
                "proc_start_ticks": ticks,
            }
        )
        state.update(
            {
                "status": "running",
                "gpu": gpu,
                "pid": process.pid,
                "proc_start_ticks": ticks,
                "started_at": attempt["started_at"],
            }
        )
        self._active[gpu] = RunningJob(
            job=job,
            gpu=gpu,
            process=process,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            marker_before=marker_before,
        )
        self._write_ledger()

    def _finish(self, gpu: int, return_code: int) -> None:
        running = self._active.pop(gpu)
        running.stdout_handle.close()
        running.stderr_handle.close()
        state = self._ledger["jobs"][running.job.job_id]
        attempt = state["attempts"][-1]
        finished = _utc_now()
        if self._stop.is_set():
            status = "interrupted"
            reason = f"coordinator_signal_{self._signal or signal.SIGTERM}"
        elif return_code != 0:
            status = "failed"
            reason = f"process_exit_{return_code}"
        else:
            passed, reason = self._success_contract(running.job, gpu=running.gpu)
            if passed:
                marker_after = self._marker_fingerprint(running.job.success_marker)
                if marker_after == running.marker_before:
                    passed = False
                    reason = "success_marker_not_created_or_refreshed_by_attempt"
            status = "completed" if passed else "failed"
        attempt.update(
            {
                "status": status,
                "return_code": int(return_code),
                "finished_at": finished,
                "success_contract": reason,
            }
        )
        state.update(
            {
                "status": status,
                "return_code": int(return_code),
                "finished_at": finished,
                "success_contract": reason,
            }
        )
        for key in ("pid", "proc_start_ticks"):
            state.pop(key, None)
        self._write_ledger()

    def _wait_for_active(self) -> None:
        while self._active:
            for gpu, running in list(self._active.items()):
                return_code = running.process.poll()
                if return_code is not None:
                    self._finish(gpu, int(return_code))
            if self._active:
                time.sleep(self.poll_seconds)

    def _summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for state in self._ledger["jobs"].values():
            status = str(state.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        return counts

    def run(self) -> int:
        old_handlers: dict[int, Any] = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal_handler)
        try:
            with GlobalLauncherLock(self.lock_path):
                self._load_ledger()
                try:
                    self._preflight()
                    pending = self._prepare_resume()
                    self._ledger["status"] = "running"
                    self._ledger["started_at"] = (
                        self._ledger.get("started_at") or _utc_now()
                    )
                    self._write_ledger()
                    while pending or self._active:
                        if not self._stop.is_set():
                            for gpu in GPU_IDS:
                                if self._stop.is_set():
                                    break
                                if gpu in self._active:
                                    continue
                                job = self._next_for_gpu(pending, gpu)
                                if job is not None:
                                    self._launch(job, gpu)
                        for gpu, running in list(self._active.items()):
                            return_code = running.process.poll()
                            if return_code is not None:
                                self._finish(gpu, int(return_code))
                        if self._stop.is_set() and self._active:
                            for running in list(self._active.values()):
                                if running.process.poll() is None:
                                    running.process.terminate()
                            self._wait_for_active()
                        if self._stop.is_set() and not self._active:
                            break
                        if pending or self._active:
                            time.sleep(self.poll_seconds)
                except BaseException:
                    self._stop.set()
                    for running in list(self._active.values()):
                        if running.process.poll() is None:
                            running.process.terminate()
                    self._wait_for_active()
                    self._ledger["status"] = "failed"
                    self._ledger["finished_at"] = _utc_now()
                    self._write_ledger()
                    raise
                counts = self._summary()
                if self._stop.is_set():
                    final_status = "interrupted"
                    exit_code = 128 + int(self._signal or signal.SIGTERM)
                elif any(
                    status not in _SUCCESS_STATUSES
                    for status in (
                        str(state.get("status"))
                        for state in self._ledger["jobs"].values()
                    )
                ):
                    final_status = "failed"
                    exit_code = 1
                else:
                    final_status = "completed"
                    exit_code = 0
                self._ledger.update(
                    {
                        "status": final_status,
                        "finished_at": _utc_now(),
                        "status_counts": counts,
                    }
                )
                self._write_ledger()
                print(
                    json.dumps(
                        {
                            "plan_id": self.plan.plan_id,
                            "status": final_status,
                            "status_counts": counts,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return exit_code
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    plan = load_plan(arguments.plan, project_root=arguments.project_root)
    ledger = arguments.ledger or plan.path.with_name(
        f"{plan.path.stem}.ledger.json"
    )
    lock = arguments.lock or (
        plan.project_root / "state/locks/same-gene-robustness-launcher.lock"
    )
    coordinator = FourGpuCoordinator(
        plan,
        ledger_path=ledger,
        lock_path=lock,
        poll_seconds=arguments.poll_seconds,
    )
    return coordinator.run()


if __name__ == "__main__":
    raise SystemExit(main())
