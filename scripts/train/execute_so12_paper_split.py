#!/usr/bin/env python3
"""Wait for allocated GPUs, preflight, then execute only the approved paper-split job."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spatial_benchmark.configuration import load_yaml_mapping, validate_experiment_config
from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import ProjectWorkerLock, QueueWorker, WorkerLockError, WorkerSettings, command_for_config
from spatial_benchmark.registry import Registry

CAMPAIGN = "cmp_20261017_so12_paper_core_split"
WORKER = "bagm-so12-paper-split-ddp4"


def now():
    return datetime.now(timezone.utc).isoformat()


def claim_exact_job(registry, job_id, worker_id):
    """Claim the named campaign job atomically without consuming another queue item."""
    stamp = now()
    with registry.transaction(immediate=True) as connection:
        cursor = connection.execute(
            "UPDATE queue_jobs SET status='claimed', claimed_at=?, heartbeat_at=?, worker_id=? "
            "WHERE job_id=? AND campaign_id=? AND status='queued' AND requested_gpu='0,1,2,3'",
            (stamp, stamp, worker_id, job_id, CAMPAIGN),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("Approved job is missing, already claimed, or has a different campaign/allocation")
    result = registry.get_job(job_id)
    if result is None:
        raise RuntimeError("Claimed job disappeared")
    return result


def gpu_processes():
    return subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
    ).strip()


def wait_for_preflight(child, stop):
    """Drain the child before the caller can release its GPU locks."""
    termination_requested = False
    while child.poll() is None:
        if stop.is_set() and not termination_requested:
            child.terminate()
            termination_requested = True
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    if stop.is_set():
        raise InterruptedError("Controller interrupted during preflight; child has exited")
    return child.returncode


def reserve_launch(path, config):
    """A crash after this durable intent requires review, never another enqueue."""
    with path.open("x") as stream:
        json.dump({"time": now(), "campaign_id": CAMPAIGN,
                   "scientific_id": scientific_id(config)}, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = current_paths()
    config = load_yaml_mapping(args.config)
    validate_experiment_config(config)
    if config["campaign"]["campaign_id"] != CAMPAIGN:
        raise ValueError("This controller only executes the approved paper-core campaign")
    state = paths.scratch_root / "active_runs/ctrl_so12_paper_split_seed0"
    state.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    child = None
    worker = QueueWorker(Registry(), settings=WorkerSettings(
        worker_id=WORKER, gpu="0,1,2,3", once=True, min_free_gb=5,
        heartbeat_seconds=30, stale_after_seconds=900), paths=paths)

    def event(status, **values):
        row = {"time": now(), "status": status, **values}
        temporary = state / "status.json.writing"
        temporary.write_text(json.dumps(row, indent=2) + "\n")
        temporary.replace(state / "status.json")
        with (state / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row), flush=True)

    def interrupt(signum, frame):
        stop.set()
        worker.request_stop()
        if child is not None and child.poll() is None:
            child.terminate()

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)

    try:
        # A unique controller lock prevents two service starts from producing
        # duplicate approved runs. Existing global/device owners are respected.
        with ProjectWorkerLock(paths.state_root / "locks/so12-paper-split-controller.lock"):
            launch_record = state / "launch.json"
            launch_intent = state / "launch_intent.json"
            if launch_record.exists() or launch_intent.exists():
                raise RuntimeError("A launch was previously reserved; inspect launch_intent.json and launch.json before recovery")
            while not stop.is_set():
                worker._assert_disk_space()
                locks = ExitStack()
                try:
                    for name in ["bagm-worker.lock", *[f"bagm-worker-gpu-{i}.lock" for i in range(4)]]:
                        locks.enter_context(ProjectWorkerLock(paths.state_root / "locks" / name))
                except WorkerLockError:
                    locks.close()
                    event("waiting_for_existing_gpu_owner", gpu_ids=[0, 1, 2, 3])
                    stop.wait(30)
                    continue
                with locks:
                    if gpu_processes():
                        event("waiting_for_existing_gpu_processes", gpu_ids=[0, 1, 2, 3])
                        stop.wait(30)
                        continue
                    worker._assert_no_surviving_children()
                    worker._assert_disk_space()
                    receipt = Path(config["launcher"]["preflight_receipt"])
                    output = receipt.parent
                    output.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                        "--nnodes=1", "--nproc-per-node=4", "--max-restarts=0",
                        str(paths.project_root / "scripts/train/run_so12_paper_split.py"),
                        "--config", str(args.config.resolve()), "--run-scratch", str(output), "--preflight"]
                    event("preflight_running", command=command, receipt=str(receipt))
                    env = dict(os.environ)
                    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3", PYTHONPATH=str(paths.project_root / "src"),
                        CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1",
                        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                    with (output / "controller.log").open("a") as log:
                        child = subprocess.Popen(command, cwd=paths.project_root, env=env,
                                                 stdout=log, stderr=subprocess.STDOUT)
                        code = wait_for_preflight(child, stop)
                        if code != 0:
                            raise RuntimeError(f"Preflight failed with exit code {code}; see {output / 'controller.log'}")
                    child = None
                    if not receipt.exists() or not json.loads(receipt.read_text()).get("all_required_gates_passed"):
                        raise RuntimeError("Preflight did not publish a passing receipt")
                    worker._assert_disk_space()
                    registry = worker.registry
                    if stop.is_set():
                        raise InterruptedError("Controller stopped before enqueue")
                    reserve_launch(launch_intent, config)
                    registry.register_variant(scientific_id(config), campaign_id=CAMPAIGN, configuration=config)
                    job = registry.enqueue(campaign_id=CAMPAIGN, configuration=config,
                        command=command_for_config(config, paths=paths), requested_gpu="0,1,2,3",
                        experiment_config_reference=args.config.resolve(), maximum_attempts=1)
                    launch_record.write_text(json.dumps({"time": now(), "job_id": job["job_id"],
                        "campaign_id": CAMPAIGN, "config": str(args.config.resolve())}, indent=2) + "\n")
                    claimed = claim_exact_job(registry, job["job_id"], WORKER)
                    if stop.is_set():
                        registry.transition_job(job["job_id"], "cancelled", worker_id=WORKER)
                        raise InterruptedError("Controller stopped before training; reserved job cancelled")
                    event("training_running", job_id=job["job_id"])
                    result = worker.execute_claimed(claimed)
                    event("training_finished", job_id=job["job_id"], queue_status=result["status"],
                          run_id=result.get("run_id"))
                    if result["status"] != "completed":
                        raise RuntimeError(f"Registered training run ended with {result['status']}")
                    return 0
            event("interrupted_before_launch")
            return 130
    except BaseException as error:
        event("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
