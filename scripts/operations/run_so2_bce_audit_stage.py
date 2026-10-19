#!/usr/bin/env python3
"""Execute one declared SO2 BCE audit stage; never select, retry, or resume jobs.

The registered audit launcher retains GPU admission, locking, registry and
artifact ownership. This process only sequences its launcher children, one
per allowed GPU, and persists operational status under the state directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from spatial_benchmark.paths import current_paths
from spatial_benchmark.so2_bce_optimization_audit import (
    BOUNDARY_LEARNING_RATES, DROPOUT_VALUES, GRID_LEARNING_RATES,
    PROFILE_EPOCHS, TARGET_BATCH_SIZES, WEIGHT_DECAYS,
)

_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
_RUN_ID = re.compile(r"r_[A-Za-z0-9_-]+\Z")
_JOB_FIELDS = {"label", "gpu", "profile", "batch_mode", "batch_size",
               "learning_rate", "dropout", "weight_decay", "seed", "smoke_run"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_plan(plan: Any) -> dict[str, Any]:
    require(isinstance(plan, dict) and set(plan) == {"stage_id", "allowed_gpus", "jobs"},
            "Plan must contain exactly stage_id, allowed_gpus and jobs")
    require(isinstance(plan["stage_id"], str) and bool(_SAFE_NAME.fullmatch(plan["stage_id"])),
            "stage_id must be a safe, nonempty name of at most 96 characters")
    allowed = plan["allowed_gpus"]
    require(isinstance(allowed, list) and bool(allowed)
            and all(isinstance(gpu, str) and gpu in {"0", "1", "2", "3"} for gpu in allowed)
            and len(set(allowed)) == len(allowed), "allowed_gpus must list unique physical GPU strings 0..3")
    require(isinstance(plan["jobs"], list) and bool(plan["jobs"]), "Plan needs at least one job")
    labels: set[str] = set()
    executions: set[tuple[Any, ...]] = set()
    for job in plan["jobs"]:
        require(isinstance(job, dict) and set(job) == _JOB_FIELDS, "Every job must contain exactly the declared job fields")
        label = job["label"]
        require(isinstance(label, str) and bool(_SAFE_NAME.fullmatch(label)) and label not in labels,
                "Job labels must be safe and unique")
        labels.add(label)
        gpu, profile, mode, batch = (job[key] for key in ("gpu", "profile", "batch_mode", "batch_size"))
        require(isinstance(gpu, str) and gpu in allowed, f"{label}: GPU is outside allowed_gpus")
        require(isinstance(profile, str) and profile in PROFILE_EPOCHS, f"{label}: unknown profile")
        require(mode in ("target_window", "core_pair"), f"{label}: unknown batch mode")
        require((mode == "target_window" and type(batch) is int and batch in TARGET_BATCH_SIZES)
                or (mode == "core_pair" and batch is None), f"{label}: invalid target batch size")
        lr, dropout, decay = (job[key] for key in ("learning_rate", "dropout", "weight_decay"))
        for key, value, choices in (("learning_rate", lr, GRID_LEARNING_RATES + BOUNDARY_LEARNING_RATES),
                                    ("dropout", dropout, DROPOUT_VALUES), ("weight_decay", decay, WEIGHT_DECAYS)):
            require(type(value) in (int, float) and math.isfinite(value) and value in choices,
                    f"{label}: invalid {key}")
        seed = job["seed"]
        require(type(seed) is int and seed in (0, 1) and (profile == "confirmation" or seed == 0),
                f"{label}: seed1 is only declared for confirmation")
        require(dropout == 0.1 or decay == 1e-5, f"{label}: regularization controls change one factor")
        require(lr not in BOUNDARY_LEARNING_RATES or (dropout == 0.1 and decay == 1e-5),
                f"{label}: boundary LR retains incumbent regularization")
        require(mode != "core_pair" or (lr == 1e-4 and dropout == 0.1 and decay == 1e-5
                and seed == 0 and profile != "confirmation"), f"{label}: paired reference settings changed")
        if profile == "smoke":
            require(job["smoke_run"] is None and lr == 1e-4 and dropout == 0.1 and decay == 1e-5
                    and (mode == "core_pair" or batch == 1024), f"{label}: smoke settings changed")
        else:
            require(isinstance(job["smoke_run"], str) and bool(_RUN_ID.fullmatch(job["smoke_run"])),
                    f"{label}: screen/confirmation requires a smoke run ID")
        identity = (profile, mode, batch, lr, dropout, decay, seed)
        require(identity not in executions, f"{label}: duplicate scientific job within the stage")
        executions.add(identity)
    # Freeze a JSON-only copy so caller mutation cannot change an active stage.
    return json.loads(json.dumps(plan, allow_nan=False))


def launcher_argv(job: Mapping[str, Any], project_root: Path) -> list[str]:
    command = [sys.executable, str(project_root / "scripts/train/launch_so2_bce_optimization_audit.py"),
               "--profile", job["profile"], "--batch-mode", job["batch_mode"],
               "--gpu", job["gpu"], "--learning-rate", str(job["learning_rate"]),
               "--dropout", str(job["dropout"]), "--weight-decay", str(job["weight_decay"]),
               "--seed", str(job["seed"])]
    if job["batch_size"] is not None:
        command += ["--target-batch-size", str(job["batch_size"])]
    if job["smoke_run"] is not None:
        command += ["--smoke-run", job["smoke_run"]]
    return command


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".state-", delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def run_stage(plan: Any, paths: Any) -> int:
    plan = validate_plan(plan)
    root = paths.state_root / "logs/so2_bce_optimization_audit" / plan["stage_id"]
    # Exclusive directory creation rejects every previous stage, even if an
    # earlier process stopped before it could finish writing its first state.
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    _atomic_json(root / "plan.json", plan)
    state: dict[str, Any] = {
        "schema": "so2_bce_audit_operational_stage_v1", "stage_id": plan["stage_id"],
        "status": "running", "runner_pid": os.getpid(), "started_at": _now(), "ended_at": None,
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "allowed_gpus": plan["allowed_gpus"], "hold_reason": None, "stop_signal": None,
        "jobs": [{"label": job["label"], "gpu": job["gpu"], "status": "pending", "pid": None,
                  "exit_code": None, "started_at": None, "ended_at": None,
                  "argv": launcher_argv(job, paths.project_root), "launcher_log": str(root / (job["label"] + ".log"))}
                 for job in plan["jobs"]],
    }
    running: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    requested_signal: int | None = None
    forwarded = False

    def save() -> None:
        state["updated_at"] = _now()
        _atomic_json(root / "state.json", state)

    def stop(number: int, _frame: Any) -> None:
        nonlocal requested_signal
        requested_signal = requested_signal or number

    def signal_children(number: int) -> None:
        for process, _, _ in running.values():
            try:
                # Every PID here was returned by our own start_new_session
                # Popen call. No discovered or externally supplied PID is used.
                os.killpg(process.pid, number)
            except ProcessLookupError:
                pass

    save()
    handlers = {number: signal.signal(number, stop) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        while True:
            if requested_signal is not None and not forwarded:
                state["stop_signal"] = requested_signal
                state["hold_reason"] = f"received_signal_{requested_signal}"
                signal_children(requested_signal)
                forwarded = True
                save()
            for gpu, (process, handle, job) in tuple(running.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                job.update(exit_code=code, ended_at=_now(), status="completed" if code == 0 else "failed")
                del running[gpu]
                if code != 0 and state["hold_reason"] is None:
                    state["hold_reason"] = f"launcher_failed:{job['label']}"
                save()
            if state["hold_reason"] is not None:
                newly_held = False
                for job in state["jobs"]:
                    if job["status"] == "pending":
                        job["status"] = "held"
                        newly_held = True
                if newly_held:
                    save()
            else:
                for gpu in plan["allowed_gpus"]:
                    if gpu in running or requested_signal is not None:
                        continue
                    job = next((job for job in state["jobs"] if job["gpu"] == gpu and job["status"] == "pending"), None)
                    if job is None:
                        continue
                    handle = None
                    try:
                        handle = Path(job["launcher_log"]).open("x")
                        process = subprocess.Popen(job["argv"], cwd=paths.project_root,
                            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                            shell=False, start_new_session=True)
                    except OSError as error:
                        if handle is not None:
                            handle.close()
                        job.update(status="launch_failed", ended_at=_now(), error=str(error))
                        state["hold_reason"] = f"launcher_start_failed:{job['label']}"
                        save()
                        break
                    job.update(status="running", pid=process.pid, started_at=_now())
                    running[gpu] = (process, handle, job)
                    save()
            if not running and not any(job["status"] == "pending" for job in state["jobs"]):
                break
            time.sleep(0.25)
        state["status"] = "interrupted" if requested_signal is not None else ("failed" if state["hold_reason"] else "completed")
        state["ended_at"] = _now()
        save()
        return 128 + requested_signal if requested_signal is not None else (2 if state["hold_reason"] else 0)
    except BaseException as error:
        state["status"] = "failed"
        state["hold_reason"] = state["hold_reason"] or "operational_exception"
        state["operational_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Unexpected operational exceptions must not orphan our launchers.
        if running:
            signal_children(signal.SIGTERM)
            for process, handle, job in running.values():
                try:
                    code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = process.wait()
                handle.close()
                job.update(exit_code=code, ended_at=_now(), status="completed" if code == 0 else "failed")
        if state["status"] == "failed" and state["ended_at"] is None:
            for job in state["jobs"]:
                if job["status"] == "pending":
                    job["status"] = "held"
            state["ended_at"] = _now()
            try:
                save()
            except OSError:
                # Preserve the original failure if storage itself is failing.
                pass
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan = json.loads(args.plan.read_text())
        return run_stage(plan, current_paths())
    except (ValueError, OSError) as error:
        print(f"Audit stage refused or stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
