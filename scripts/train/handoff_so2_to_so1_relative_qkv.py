#!/usr/bin/env python3
"""Safely hand all four GPUs from the fixed SO2 run to the SO1 plateau run.

This is an operational coordinator only.  It never trains a model directly,
creates an unregistered process, or relaxes either campaign contract.  The
handoff waits for the exact SO2 recovery run to finalize successfully, checks
its epoch-300 artifacts, waits for all four GPUs to become idle, runs the
registered SO1 DDP preflight, enqueues exactly one SO1 job through the normal
CLI, starts the one-shot supervisor worker, and verifies its first durable
epoch row and rolling checkpoint.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from scripts.train.run_so1_14core_relative_qkv import (  # noqa: E402
    CAMPAIGN_ID as SO1_CAMPAIGN_ID,
    _section,
    _validate_bound_preparation,
    _validate_hardware_preflight,
)
from spatial_benchmark.configuration import (  # noqa: E402
    compose_config,
    load_yaml_mapping,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.so1_training_observability import (  # noqa: E402
    EPOCH_METRICS_SCHEMA,
    SO1_CORE_ALIASES,
)


SO2_RUN_ID = "r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2"
SO2_JOB_ID = "q_c1311a5e2b200f644a76"
SO1_PREFLIGHT_SERVICE = "bagm_so1_14core_preflight"
SO1_WORKER_SERVICE = "bagm_so1_14core_ddp4_worker"
SO1_EXPERIMENT = Path(
    "configs/experiment/so1_14core_relative_qkv_seed0_batch2_plateau_min150.yaml"
)
SO1_CAMPAIGN_PLAN = Path(
    "experiments/campaigns/"
    "cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150/"
    "campaign.yaml"
)
SO1_PREFLIGHT_RECEIPT = Path(
    "state/preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json"
)
SO2_FINAL_EPOCH = 300
SO2_FINAL_OPTIMIZER_STEPS = 2100
SO1_FIRST_EPOCH_OPTIMIZER_STEPS = 7
SO1_FIRST_EPOCH_GRAPH_VIEWS = 140
MIN_FREE_GIB = 25.0


class HandoffError(RuntimeError):
    """Raised when an operational or scientific handoff gate fails."""


class _BlockingWorkerLock:
    """Wait for and exclusively hold the repository's global worker lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None

    def __enter__(self) -> "_BlockingWorkerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "role": "so2_to_so1_handoff",
                    "acquired_at": _utc_now(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".writing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record_phase(path: Path, phase: str, **details: Any) -> None:
    history: list[dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, Mapping) and isinstance(existing.get("history"), list):
                history = [
                    dict(item)
                    for item in existing["history"]
                    if isinstance(item, Mapping)
                ]
        except (OSError, ValueError):
            history = []
    event = {"at": _utc_now(), "phase": phase, **details}
    if not history or history[-1] != event:
        history.append(event)
    _atomic_json(
        path,
        {
            "schema": "so2_to_so1_relative_qkv_handoff_v1",
            "so2_run_id": SO2_RUN_ID,
            "so1_campaign_id": SO1_CAMPAIGN_ID,
            "phase": phase,
            "updated_at": event["at"],
            "details": details,
            "history": history,
        },
    )
    print(json.dumps(event, sort_keys=True), flush=True)


def _json_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffError(f"Cannot read {label}: {path}.") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"{label} must be a JSON mapping.")
    return value


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise HandoffError(
            f"{field} must be {expected!r}; received {actual!r}."
        )


def validate_so2_final_bundle(artifact_root: Path) -> dict[str, Any]:
    """Validate the exact semantic completion needed before releasing SO2."""

    root = artifact_root.resolve(strict=True)
    if not (root / "_SUCCESS").is_file():
        raise HandoffError("SO2 finalized bundle lacks _SUCCESS.")
    if (root / "_FAILED").exists():
        raise HandoffError("SO2 finalized bundle contains _FAILED.")
    summary = _json_file(root / "summary.json", label="SO2 summary")
    for field, expected in {
        "run_id": SO2_RUN_ID,
        "status": "success",
        "final_epoch": SO2_FINAL_EPOCH,
        "optimizer_steps": SO2_FINAL_OPTIMIZER_STEPS,
        "fixed_epoch_target_completed": True,
        "checkpoint_reload_verified": True,
        "checkpoint": "checkpoints/last.ckpt",
        "world_size": 4,
    }.items():
        _require_equal(summary.get(field), expected, field=f"so2.summary.{field}")

    final_metrics = _json_file(
        root / "metrics/final.json", label="SO2 final metrics"
    )
    _require_equal(
        final_metrics.get("fit/training/final_global_epoch"),
        float(SO2_FINAL_EPOCH),
        field="so2.metrics.final_global_epoch",
    )
    reload_receipt = _json_file(
        root / "diagnostics/final_checkpoint_reload_verification.json",
        label="SO2 final checkpoint reload receipt",
    )
    _require_equal(reload_receipt.get("verified"), True, field="so2.reload.verified")
    payload = reload_receipt.get("payload")
    replay = reload_receipt.get("fixed_prediction_replay")
    if not isinstance(payload, Mapping) or not isinstance(replay, Mapping):
        raise HandoffError("SO2 reload receipt is incomplete.")
    for field, expected in {
        "completed_global_epochs": SO2_FINAL_EPOCH,
        "optimizer_updates_completed": SO2_FINAL_OPTIMIZER_STEPS,
        "full_resume_payload_validated": True,
        "fixed_completion_payload_validated": True,
    }.items():
        _require_equal(payload.get(field), expected, field=f"so2.reload.{field}")
    _require_equal(replay.get("verified"), True, field="so2.replay.verified")

    checkpoints = sorted((root / "checkpoints").glob("*.ckpt"))
    if [path.name for path in checkpoints] != ["last.ckpt"]:
        raise HandoffError("SO2 final checkpoint layout is not last.ckpt-only.")
    checkpoint_sha256 = sha256_file(checkpoints[0])
    _require_equal(
        reload_receipt.get("checkpoint_file_sha256"),
        checkpoint_sha256,
        field="so2.reload.checkpoint_file_sha256",
    )

    csv_path = root / "results/epoch_metrics.csv"
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise HandoffError("SO2 epoch metrics CSV is unavailable.") from exc
    epochs = [int(row["global_epoch"]) for row in rows]
    if epochs != list(range(1, SO2_FINAL_EPOCH + 1)):
        raise HandoffError("SO2 epoch metrics are not the exact contiguous 1--300 history.")
    if any(row.get("schema") != "so2_14core_epoch_metrics_v1" for row in rows):
        raise HandoffError("SO2 epoch metrics schema drifted.")
    return {
        "artifact_root": str(root),
        "checkpoint": str(checkpoints[0]),
        "checkpoint_sha256": checkpoint_sha256,
        "epoch_metrics_csv": str(csv_path),
        "epoch_rows": len(rows),
        "final_epoch": epochs[-1],
    }


def validate_so1_first_epoch_row(row: Mapping[str, str]) -> dict[str, Any]:
    """Require the first live row to encode one complete 14-core epoch."""

    for field, expected in {
        "schema": EPOCH_METRICS_SCHEMA,
        "model_seed": "0",
        "global_epoch": "1",
        "optimizer_updates_this_epoch": str(SO1_FIRST_EPOCH_OPTIMIZER_STEPS),
        "cumulative_optimizer_updates": str(SO1_FIRST_EPOCH_OPTIMIZER_STEPS),
        "complete_graph_mask_views": str(SO1_FIRST_EPOCH_GRAPH_VIEWS),
    }.items():
        _require_equal(row.get(field), expected, field=f"so1.first_epoch.{field}")
    for alias in SO1_CORE_ALIASES:
        field = "loss_" + alias.lower().replace("-", "_")
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise HandoffError(f"SO1 first epoch lacks finite {field}.") from exc
        if not math.isfinite(value):
            raise HandoffError(f"SO1 first epoch has non-finite {field}.")
    return {
        "run_id": row.get("run_id"),
        "global_epoch": 1,
        "optimizer_updates": SO1_FIRST_EPOCH_OPTIMIZER_STEPS,
        "complete_graph_mask_views": SO1_FIRST_EPOCH_GRAPH_VIEWS,
        "equal_core_mean_masked_huber": float(
            row["equal_core_mean_masked_huber"]
        ),
    }


def _service_state(name: str) -> str:
    result = subprocess.run(
        ["supervisorctl", "status", name],
        text=True,
        capture_output=True,
        check=False,
    )
    output = (result.stdout or result.stderr).strip()
    fields = output.split()
    if result.returncode != 0 or len(fields) < 2:
        raise HandoffError(f"Cannot resolve supervisor service {name!r}: {output}")
    return fields[1]


def _start_service(name: str) -> None:
    state = _service_state(name)
    if state in {"RUNNING", "STARTING"}:
        return
    result = subprocess.run(
        ["supervisorctl", "start", name],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise HandoffError(
            f"Failed to start supervisor service {name!r}: "
            f"{(result.stdout or result.stderr).strip()}"
        )


def _run_cli_json(
    arguments: Sequence[str],
    *,
    database: Path,
    source_root: Path,
) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spatial_benchmark",
            "--database",
            str(database),
            *arguments,
        ],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise HandoffError(
            f"Registry CLI failed ({' '.join(arguments)}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    try:
        value = json.loads(result.stdout)
    except ValueError as exc:
        raise HandoffError("Registry CLI returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise HandoffError("Registry CLI result must be a mapping.")
    return value


def _gpu_snapshot() -> dict[str, Any]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if gpu_query.returncode != 0:
        raise HandoffError("nvidia-smi GPU query failed.")
    gpus = []
    for line in gpu_query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise HandoffError("Unexpected nvidia-smi GPU query output.")
        gpus.append(
            {
                "index": int(fields[0]),
                "memory_used_mib": int(fields[1]),
                "utilization_percent": int(fields[2]),
            }
        )
    if [gpu["index"] for gpu in gpus] != [0, 1, 2, 3]:
        raise HandoffError("The handoff requires exact physical GPUs 0,1,2,3.")
    process_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if process_query.returncode != 0:
        raise HandoffError("nvidia-smi compute-process query failed.")
    processes = [line.strip() for line in process_query.stdout.splitlines() if line.strip()]
    return {"gpus": gpus, "compute_processes": processes}


def _gpu_snapshot_is_idle(snapshot: Mapping[str, Any]) -> bool:
    gpus = snapshot.get("gpus")
    processes = snapshot.get("compute_processes")
    return bool(
        isinstance(gpus, list)
        and len(gpus) == 4
        and isinstance(processes, list)
        and not processes
        and all(
            isinstance(gpu, Mapping)
            and int(gpu["memory_used_mib"]) <= 512
            and int(gpu["utilization_percent"]) <= 10
            for gpu in gpus
        )
    )


def _wait_for_two_idle_snapshots(*, poll_seconds: float, runtime_root: Path) -> dict[str, Any]:
    consecutive = 0
    latest: dict[str, Any] = {}
    while consecutive < 2:
        free_gib = shutil.disk_usage(runtime_root).free / (1024**3)
        if free_gib < MIN_FREE_GIB:
            raise HandoffError(
                f"Only {free_gib:.2f} GiB is free; at least {MIN_FREE_GIB:.0f} is required."
            )
        latest = _gpu_snapshot()
        if _gpu_snapshot_is_idle(latest):
            consecutive += 1
        else:
            consecutive = 0
        if consecutive < 2:
            time.sleep(poll_seconds)
    return {**latest, "free_disk_gib": free_gib, "consecutive_idle_checks": 2}


def _validate_registered_campaign(
    registry: Registry,
    *,
    source_root: Path,
) -> None:
    campaign = registry.get_campaign(SO1_CAMPAIGN_ID)
    if campaign is None:
        raise HandoffError("SO1 campaign is not registered.")
    plan = load_yaml_mapping(source_root / SO1_CAMPAIGN_PLAN)
    _require_equal(campaign.get("config"), plan, field="registry.so1_campaign_plan")


def _matching_so1_job(
    registry: Registry,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    matches = [
        job
        for job in registry.list_queue(limit=10_000)
        if job.get("campaign_id") == SO1_CAMPAIGN_ID
    ]
    if len(matches) > 1:
        raise HandoffError("More than one SO1 campaign queue job exists.")
    if not matches:
        return None
    job = matches[0]
    for field, expected in {
        "canonical_config": dict(config),
        "requested_gpu": "0,1,2,3",
        "priority": 100,
        "maximum_attempts": 2,
    }.items():
        _require_equal(job.get(field), expected, field=f"so1.queue.{field}")
    if job.get("status") != "queued":
        raise HandoffError(
            "A recovered SO1 handoff may reuse only one matching queued job; "
            f"the existing job is {job.get('status')!r}."
        )
    return job


def _enqueue_so1(
    registry: Registry,
    *,
    config: Mapping[str, Any],
    database: Path,
    source_root: Path,
) -> dict[str, Any]:
    existing = _matching_so1_job(registry, config=config)
    if existing is not None:
        return existing
    active_eligible = [
        job
        for job in registry.list_queue(
            statuses=("queued", "claimed", "running"), limit=10_000
        )
        if job.get("requested_gpu") in {None, "0,1,2,3"}
    ]
    if active_eligible:
        raise HandoffError(
            "Another queue job is eligible for the four-GPU SO1 worker: "
            + ", ".join(str(job["job_id"]) for job in active_eligible)
        )
    result = _run_cli_json(
        (
            "enqueue-experiment",
            "--campaign-id",
            SO1_CAMPAIGN_ID,
            "--config",
            str(SO1_EXPERIMENT),
            "--priority",
            "100",
            "--max-attempts",
            "2",
            "--gpu",
            "0,1,2,3",
        ),
        database=database,
        source_root=source_root,
    )
    job_id = result.get("job_id")
    if not isinstance(job_id, str):
        raise HandoffError("SO1 enqueue did not return a job ID.")
    job = registry.get_job(job_id)
    if job is None:
        raise HandoffError("SO1 queue job disappeared after enqueue.")
    return job


def _wait_for_first_so1_epoch(
    registry: Registry,
    *,
    job_id: str,
    paths: ProjectPaths,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    run_id: str | None = None
    while time.monotonic() < deadline:
        job = registry.get_job(job_id)
        if job is None:
            raise HandoffError("SO1 queue job disappeared.")
        if job.get("status") in {"failed", "cancelled", "interrupted"}:
            raise HandoffError(
                f"SO1 job failed before its first epoch: {job.get('last_error')!r}."
            )
        if isinstance(job.get("run_id"), str):
            run_id = str(job["run_id"])
            csv_path = (
                paths.scratch_root
                / "active_runs"
                / run_id
                / "results/epoch_metrics.csv"
            )
            if csv_path.is_file():
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if rows:
                    receipt = validate_so1_first_epoch_row(rows[0])
                    checkpoint = csv_path.parents[1] / "checkpoints/latest.ckpt"
                    if checkpoint.is_file():
                        return {
                            **receipt,
                            "job_id": job_id,
                            "run_id": run_id,
                            "epoch_metrics_csv": str(csv_path),
                            "latest_checkpoint": str(checkpoint),
                            "latest_checkpoint_sha256": sha256_file(checkpoint),
                        }
        time.sleep(poll_seconds)
    raise HandoffError(
        f"Timed out after {timeout_seconds:.0f}s waiting for the first SO1 epoch "
        f"(run_id={run_id!r})."
    )


def run_handoff(args: argparse.Namespace) -> dict[str, Any]:
    paths = current_paths()
    source_root = paths.project_root.resolve(strict=True)
    runtime_root = paths.data_root.parent.resolve(strict=True)
    database = args.database.resolve(strict=True)
    state_path = args.state.resolve()
    registry = Registry(database)

    config = compose_config(
        source_root / SO1_EXPERIMENT,
        config_root=paths.config_root,
        validate=True,
    )
    _validate_registered_campaign(registry, source_root=source_root)
    dataset = _section(config, "dataset")
    cohort_dir = paths.data_root / Path(str(dataset["prepared_artifact"])).relative_to(
        "data"
    )
    graph_dir = paths.data_root / Path(
        str(dataset["prepared_graph_artifact"])
    ).relative_to("data")
    preparation = _validate_bound_preparation(
        dataset,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    _record_phase(state_path, "so1_immutable_readiness_verified", **preparation)
    worker_lock = paths.state_root / "locks/bagm-worker.lock"
    _record_phase(
        state_path,
        "waiting_for_so2_worker_lock",
        worker_lock=str(worker_lock),
        so2_run_id=SO2_RUN_ID,
        so2_job_id=SO2_JOB_ID,
    )
    with _BlockingWorkerLock(worker_lock):
        _record_phase(state_path, "exclusive_worker_lock_acquired")
        so2_job = registry.get_job(SO2_JOB_ID)
        so2_run = registry.get_run(SO2_RUN_ID)
        if so2_job is None or so2_run is None:
            raise HandoffError("Required SO2 queue job or run is not registered.")
        _require_equal(
            so2_job.get("run_id"), SO2_RUN_ID, field="so2.queue.run_id"
        )
        _require_equal(so2_job.get("status"), "completed", field="so2.queue.status")
        _require_equal(so2_run.get("status"), "completed", field="so2.run.status")
        if (paths.scratch_root / "active_runs" / SO2_RUN_ID).exists():
            raise HandoffError("SO2 active scratch directory remains after completion.")

        artifact_root = Path(str(so2_run["artifact_path"]))
        so2_bundle = validate_so2_final_bundle(artifact_root)
        verification = _run_cli_json(
            ("verify-artifacts", "--run-id", SO2_RUN_ID),
            database=database,
            source_root=source_root,
        )
        _require_equal(verification.get("valid"), True, field="so2.bundle.valid")
        _require_equal(
            verification.get("bundle_issues"), [], field="so2.bundle.issues"
        )
        _require_equal(
            verification.get("registry_issues"), [], field="so2.registry.issues"
        )
        _record_phase(state_path, "so2_epoch300_bundle_verified", **so2_bundle)

        idle = _wait_for_two_idle_snapshots(
            poll_seconds=args.idle_confirmation_seconds,
            runtime_root=runtime_root,
        )
        _record_phase(state_path, "four_gpus_idle_after_so2", **idle)

        receipt_path = paths.state_root / SO1_PREFLIGHT_RECEIPT.relative_to("state")
        try:
            _validate_hardware_preflight(
                config,
                paths=paths,
                cohort_dir=cohort_dir,
                graph_dir=graph_dir,
            )
            preflight_valid = True
        except Exception:
            preflight_valid = False
        if not preflight_valid:
            _record_phase(state_path, "so1_preflight_starting")
            _start_service(SO1_PREFLIGHT_SERVICE)
            deadline = time.monotonic() + args.preflight_timeout_seconds
            while time.monotonic() < deadline:
                service_state = _service_state(SO1_PREFLIGHT_SERVICE)
                if service_state in {"EXITED", "STOPPED", "FATAL", "BACKOFF"}:
                    break
                time.sleep(args.poll_seconds)
            else:
                raise HandoffError("SO1 preflight exceeded its timeout.")
        preflight = _validate_hardware_preflight(
            config,
            paths=paths,
            cohort_dir=cohort_dir,
            graph_dir=graph_dir,
        )
        _record_phase(
            state_path,
            "so1_preflight_verified",
            receipt=str(receipt_path),
            receipt_content_sha256=preflight["receipt_content_sha256"],
            peak_vram_gib_all_ranks=preflight["peak_vram_gib_all_ranks"],
            optimizer_updates=preflight["optimizer_updates"],
            complete_graph_mask_views=preflight["complete_graph_mask_views"],
        )

        idle = _wait_for_two_idle_snapshots(
            poll_seconds=args.idle_confirmation_seconds,
            runtime_root=runtime_root,
        )
        job = _enqueue_so1(
            registry,
            config=config,
            database=database,
            source_root=source_root,
        )
        _record_phase(
            state_path,
            "so1_job_queued",
            job_id=job["job_id"],
            queue_status=job["status"],
            requested_gpu=job["requested_gpu"],
            free_disk_gib=idle["free_disk_gib"],
        )
    _record_phase(state_path, "exclusive_worker_lock_released")
    _start_service(SO1_WORKER_SERVICE)
    _record_phase(
        state_path,
        "so1_worker_started",
        job_id=job["job_id"],
        supervisor_service=SO1_WORKER_SERVICE,
    )
    first_epoch = _wait_for_first_so1_epoch(
        registry,
        job_id=str(job["job_id"]),
        paths=paths,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.first_epoch_timeout_seconds,
    )
    _record_phase(state_path, "so1_first_epoch_verified", **first_epoch)
    return first_epoch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely hand four GPUs from the exact SO2 recovery to SO1."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("/workspace/BAGM/state/tracking/bagm.sqlite3"),
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=Path(
            "/workspace/BAGM/state/handoffs/"
            "so2_epoch300_to_so1_plateau_min150.json"
        ),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--idle-confirmation-seconds", type=float, default=30.0)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--first-epoch-timeout-seconds", type=float, default=3600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0 or args.idle_confirmation_seconds <= 0:
        raise SystemExit("Polling intervals must be positive.")
    try:
        result = run_handoff(args)
    except BaseException as exc:
        try:
            _record_phase(args.state.resolve(), "failed", error=repr(exc))
        finally:
            raise
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
