#!/usr/bin/env python3
"""Schedule independent benchmark jobs across safely selected GPUs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.paths import PROJECT_ROOT  # noqa: E402

RUN_SCRIPT = PROJECT_ROOT / "scripts" / "train" / "run_spatial_benchmark.py"

from spatial_benchmark.standards_lock import (  # noqa: E402
    StandardsLockError,
    authorize_locked_test_job,
    verify_locked_test_matrix,
)


@dataclass(frozen=True)
class Job:
    job_id: str
    values: Mapping[str, Any]
    output: Path
    log: Path
    authorization: Mapping[str, Any] | None = None


def _canonical_id(values: Mapping[str, Any]) -> str:
    text = json.dumps(values, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    pieces = [
        str(values.get("model", "model")),
        f"s{values.get('seed', 0)}",
        f"k{values.get('k', 'base')}",
        f"r{values.get('radius_um', 'base')}",
        str(values.get("symmetry", "base")),
        str(values.get("curriculum", "base")).replace("+", ""),
        f"d{values.get('hidden_dim', 'base')}",
    ]
    prefix = "_".join(piece.replace(".", "p") for piece in pieces)
    return f"{prefix}_{digest}"


def _load_matrix(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Matrix YAML must contain a mapping.")
    allowed = {
        "name",
        "factors",
        "fixed",
        "include",
        "exclude",
        "open_test",
        "save_predictions",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown matrix keys: {sorted(unknown)}")
    if not isinstance(value.get("factors"), Mapping):
        raise ValueError("Matrix YAML requires a factors mapping.")
    return dict(value)


def _matches(values: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    return all(values.get(key) == expected for key, expected in rule.items())


def _expand(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    factors = dict(matrix["factors"])
    names = sorted(factors)
    choices = []
    for name in names:
        value = factors[name]
        if not isinstance(value, list) or not value:
            raise ValueError(f"Factor {name!r} must be a non-empty list.")
        choices.append(value)
    fixed = dict(matrix.get("fixed", {}))
    excluded = list(matrix.get("exclude", []))
    jobs: list[dict[str, Any]] = []
    for combination in itertools.product(*choices):
        values = {**fixed, **dict(zip(names, combination))}
        if any(_matches(values, rule) for rule in excluded):
            continue
        jobs.append(values)
    jobs.extend(dict(item) for item in matrix.get("include", []))
    unique: dict[str, dict[str, Any]] = {}
    for values in jobs:
        unique[json.dumps(values, sort_keys=True)] = values
    return [unique[key] for key in sorted(unique)]


def _arg_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def _command(
    prepared: Path,
    job: Job,
    matrix: Mapping[str, Any],
    standards_lock: Path | None,
) -> list[str]:
    values = dict(job.values)
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--prepared",
        str(prepared),
        "--output",
        str(job.output),
        "--model",
        str(values.pop("model")),
        "--seed",
        str(values.pop("seed")),
        "--device",
        "cuda",
    ]
    flags = {"rewired", "amp"}
    for name, value in sorted(values.items()):
        if value is None:
            continue
        if name in flags:
            if bool(value):
                command.append(_arg_name(name))
            elif name == "amp":
                command.append("--no-amp")
        else:
            command.extend([_arg_name(name), str(value)])
    if bool(matrix.get("open_test", False)):
        command.append("--open-test")
        if standards_lock is None:
            raise ValueError("Open-test jobs require a standards lock.")
        command.extend(["--standards-lock", str(standards_lock)])
    if not bool(matrix.get("save_predictions", True)):
        command.append("--no-save-predictions")
    return command


def _is_complete(
    path: Path,
    authorization: Mapping[str, Any] | None = None,
) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("status") != "complete":
        return False
    if authorization is None:
        return value.get("sealed_test_opened") is False
    observed = value.get("standards_lock")
    if not isinstance(observed, Mapping):
        return False
    required = (
        "lock_id",
        "artifact_id",
        "final_matrix_file",
        "final_matrix_sha256",
        "canonical_job_hash",
    )
    return (
        value.get("sealed_test_opened") is True
        and all(observed.get(key) == authorization.get(key) for key in required)
    )


def _job_succeeded(exit_code: int, job: Job) -> bool:
    """Require the same lock authorization after execution as before skipping."""
    return exit_code == 0 and _is_complete(job.output, job.authorization)


def _parse_gpu_selection(selection: str) -> list[str]:
    devices = [part.strip() for part in selection.split(",") if part.strip()]
    if not devices:
        raise ValueError("At least one GPU must be selected.")
    if len(set(devices)) != len(devices):
        raise ValueError("GPU selection must not contain duplicate devices.")
    return devices


def _default_gpu_selection(
    environ: Mapping[str, str] | None = None,
) -> str:
    values = os.environ if environ is None else environ
    for variable in ("BAGM_GPU_IDS", "CUDA_VISIBLE_DEVICES"):
        configured = values.get(variable, "").strip()
        if configured:
            _parse_gpu_selection(configured)
            return configured

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "0"
    detected = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]
    return ",".join(detected) if detected else "0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--standards-lock",
        type=Path,
        help=(
            "Immutable standards-lock artifact. Required exactly when the "
            "matrix opens the sealed test."
        ),
    )
    parser.add_argument(
        "--gpus",
        default=_default_gpu_selection(),
        help=(
            "Comma-separated physical GPU indices allocated to this task. "
            "Defaults to BAGM_GPU_IDS, CUDA_VISIBLE_DEVICES, or nvidia-smi "
            "discovery, in that order."
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    matrix = _load_matrix(args.matrix)
    values = _expand(matrix)
    opens_test = matrix.get("open_test") is True
    if opens_test and args.standards_lock is None:
        parser.error("--standards-lock is required when matrix open_test=true")
    if not opens_test and args.standards_lock is not None:
        parser.error(
            "--standards-lock is accepted only when matrix open_test=true"
        )
    lock_record: dict[str, Any] | None = None
    authorizations: list[dict[str, Any] | None]
    if opens_test:
        assert args.standards_lock is not None
        try:
            lock_record = verify_locked_test_matrix(
                args.standards_lock,
                args.matrix,
            )
            authorized = [
                authorize_locked_test_job(args.standards_lock, item)
                for item in values
            ]
        except (FileNotFoundError, StandardsLockError) as exc:
            parser.error(f"standards-lock verification failed: {exc}")
        if any(
            record["final_matrix_file"] != lock_record["final_matrix_file"]
            or record["final_matrix_sha256"]
            != lock_record["final_matrix_sha256"]
            for record in authorized
        ):
            parser.error(
                "matrix jobs are not authorized by the supplied locked "
                "final matrix"
            )
        authorizations = authorized
    else:
        authorizations = [None] * len(values)
    output_root = args.output_root.resolve()
    logs = output_root / "logs"
    runs = output_root / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    jobs = [
        Job(
            job_id=_canonical_id(item),
            values=item,
            output=runs / _canonical_id(item),
            log=logs / f"{_canonical_id(item)}.log",
            authorization=authorization,
        )
        for item, authorization in zip(values, authorizations)
    ]
    pending = [
        job
        for job in jobs
        if not _is_complete(job.output, job.authorization)
    ]
    skipped = len(jobs) - len(pending)
    devices = _parse_gpu_selection(args.gpus)

    active: dict[str, tuple[subprocess.Popen[str], Job, Any]] = {}
    failures: list[dict[str, Any]] = []
    completed = 0
    while pending or active:
        free = [device for device in devices if device not in active]
        while free and pending:
            device = free.pop(0)
            job = pending.pop(0)
            handle = job.log.open("w", encoding="utf-8")
            command = _command(
                args.prepared.resolve(),
                job,
                matrix,
                (
                    args.standards_lock.resolve()
                    if args.standards_lock is not None
                    else None
                ),
            )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = device
            handle.write(json.dumps({"device": device, "command": command}) + "\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[device] = (process, job, handle)

        time.sleep(max(0.1, float(args.poll_seconds)))
        for device, (process, job, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            del active[device]
            if _job_succeeded(code, job):
                completed += 1
            else:
                failures.append(
                    {
                        "job_id": job.job_id,
                        "device": device,
                        "exit_code": code,
                        "log": str(job.log),
                        "values": dict(job.values),
                    }
                )
        print(
            json.dumps(
                {
                    "matrix": matrix.get("name", args.matrix.stem),
                    "pending": len(pending),
                    "active": len(active),
                    "completed_this_invocation": completed,
                    "skipped_complete": skipped,
                    "failed": len(failures),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "matrix": matrix.get("name", args.matrix.stem),
        "standards_lock": lock_record,
        "jobs": len(jobs),
        "completed_this_invocation": completed,
        "skipped_complete": skipped,
        "failures": failures,
    }
    (output_root / "launcher_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
