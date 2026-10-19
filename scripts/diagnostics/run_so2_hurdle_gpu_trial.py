#!/usr/bin/env python3
"""Register and supervise the bounded SO2 hurdle runtime diagnostic.

Run from the source-worktree root with explicit shared BAGM path overrides.
This process must remain supervised until its child and archive complete.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Iterator, Mapping
import uuid

import yaml

from spatial_benchmark.configuration import validate_experiment_config
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import (
    ProjectWorkerLock, QueueWorker, WorkerLockError, WorkerSettings,
    command_for_config,
)
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle
from spatial_benchmark.so2_hurdle_runtime_trial import (
    CAMPAIGN, SOURCE_RUN, SOURCE_SUCCESS_SHA, adapt_outputs, build_configuration,
)


@contextmanager
def global_worker_lease(path: Path) -> Iterator[None]:
    """Share the global lock with pilots, excluding legacy exclusive workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkerLockError("An active global BAGM worker prevents this GPU trial.") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def inspect_gpu(gpu: str) -> dict[str, Any]:
    def query(arguments: list[str]) -> str:
        return subprocess.run(["nvidia-smi", *arguments], check=True, text=True,
                              capture_output=True, timeout=20).stdout.strip()
    rows = query(["--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu",
                  "--format=csv,noheader,nounits"])
    devices = [line.split(",") for line in rows.splitlines() if line.strip()]
    selected = next((row for row in devices if row[0].strip() == gpu), None)
    if selected is None:
        raise RuntimeError(f"GPU {gpu} is not visible to nvidia-smi.")
    processes = query(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    if any(line.split(",")[0].strip() == selected[1].strip()
           for line in processes.splitlines() if line.strip()):
        raise RuntimeError(f"GPU {gpu} already has a compute process; no job was launched.")
    if int(selected[5]) != 0 or int(selected[4]) < 22 * 1024:
        raise RuntimeError(f"GPU {gpu} is not idle with at least 22 GiB free.")
    return {"all_devices_csv": rows, "compute_processes_csv": processes,
            "selected_gpu": gpu}


def source_configuration(registry: Registry, paths: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    record = registry.get_run(SOURCE_RUN)
    source = RunArchive.artifact_path_for(SOURCE_RUN, paths)
    if not record or record["status"] != "completed" or Path(record["artifact_path"]).resolve() != source.resolve():
        raise RuntimeError("Frozen source run is not completed at its canonical registry path.")
    verify_run_bundle(source)
    if sha256_file(source / "_SUCCESS") != SOURCE_SUCCESS_SHA:
        raise RuntimeError("Frozen source completion receipt differs from its pinned checksum.")
    names = {
        "config_file_sha256": "config.resolved.yaml", "manifest_sha256": "manifest.yaml",
        "success_marker_sha256": "_SUCCESS",
        "checksum_manifest_sha256": "provenance/artifact_checksums.json",
        "data_receipt_sha256": "provenance/data_fingerprints.json",
        "split_receipt_sha256": "provenance/split_fingerprint.json",
    }
    receipt = {"run_id": SOURCE_RUN, "artifact_path": str(source),
               **{key: sha256_file(source / relative) for key, relative in names.items()}}
    return yaml.safe_load((source / "config.resolved.yaml").read_text()), receipt


class RuntimeTrialWorker(QueueWorker):
    """Use standard lifecycle/provenance; adapt only this diagnostic's outputs."""

    admission: Mapping[str, Any]

    def _subprocess_environment(self, **kwargs: Any) -> dict[str, str]:
        environment = super()._subprocess_environment(**kwargs)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            environment[name] = "4"
        # Prevent inherited distributed launch state changing a one-process trial.
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                     "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(name, None)
        return environment

    def _write_provenance(self, archive: RunArchive, **kwargs: Any) -> None:
        super()._write_provenance(archive, **kwargs)
        config = kwargs["configuration"]
        archive.write_json("config.resolved.json", config)
        archive.write_json("provenance/source_bundle.json", config["source_bundle"])
        archive.write_json("provenance/gpu_admission.json", dict(self.admission))
        root = self.paths.project_root
        files = sorted(set((root / "src/spatial_benchmark").rglob("*.py"))
                       | set((root / "scripts/train").glob("*.py"))
                       | set((root / "scripts/diagnostics").glob("*.py")))
        archive.write_json("provenance/source_files.json", {
            "files": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in files],
            "note": "Code hashes only; no datasets or clinical inputs copied.",
        })
        for relative in ("scripts/diagnostics/run_so2_hurdle_gpu_trial.py",
                         "scripts/diagnostics/so2_hurdle_gpu_trial_worker.py",
                         "src/spatial_benchmark/so2_hurdle_runtime_trial.py"):
            archive.copy_file(root / relative, "provenance/source_snapshot/" + relative)
        archive.copy_file(root / "experiments/campaigns" / CAMPAIGN / "README.md",
                          "provenance/runtime_contract.md")

    def _adapt_spatial_benchmark_output(self, archive: RunArchive, command: Any,
                                        *, configuration: Mapping[str, Any]) -> None:
        adapt_outputs(archive, configuration)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), default="0")
    parser.add_argument("--epochs", type=int, choices=(2,), default=2)
    parser.add_argument("--check-only", action="store_true", help="Verify source/config/devices without writes or launch")
    args = parser.parse_args(argv)
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    source, receipt = source_configuration(registry, paths)
    config = build_configuration(source, receipt, gpu=args.gpu, epochs=args.epochs)
    validate_experiment_config(config)
    command = command_for_config(config, paths=paths)
    if args.check_only:
        admission = inspect_gpu(args.gpu)
        print(json.dumps({"check_only": True, "campaign": CAMPAIGN, "source_run": SOURCE_RUN,
                          "gpu": args.gpu, "epochs": args.epochs, "admission": admission}, sort_keys=True))
        return 0
    worker_id = f"so2-hurdle-trial-gpu{args.gpu}-{uuid.uuid4().hex[:8]}"
    worker = RuntimeTrialWorker(registry, settings=WorkerSettings(
        worker_id=worker_id, gpu=args.gpu, once=True, parallel_gpu_workers=True,
        min_free_gb=20, heartbeat_seconds=15), paths=paths)
    # Hold both locks before admission/registration. Global workers require an
    # exclusive lease; independent pilots may share it on separate GPU locks.
    with global_worker_lease(paths.state_root / "locks/bagm-worker.lock"), ProjectWorkerLock(worker.lock_path):
        worker._assert_no_surviving_children()
        worker._assert_disk_space()
        worker.admission = inspect_gpu(args.gpu)
        registry.create_campaign(CAMPAIGN, name="SO2 hurdle GPU runtime trial",
            scientific_question="Can the unchanged SO2 hurdle model run safely on one GPU?",
            config={"protocol": config["evaluation"]["protocol"], "source_run_id": SOURCE_RUN}, status="pilot")
        job = registry.enqueue(campaign_id=CAMPAIGN, configuration=config, command=command,
            requested_gpu=args.gpu, maximum_attempts=1,
            job_id=f"q_so2_hurdle_gpu_trial_{uuid.uuid4().hex}")
        claimed = registry.claim_next(worker_id, requested_gpu=args.gpu, job_id=job["job_id"])
        if claimed is None:
            raise RuntimeError("New runtime trial could not claim its exact registered job.")
        old_handlers = {number: signal.signal(number, worker._signal_handler)
                        for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            print(json.dumps({"job_id": job["job_id"], "worker_id": worker_id,
                              "gpu": args.gpu, "status": "claimed"}), flush=True)
            result = worker.execute_claimed(claimed)
        finally:
            for number, handler in old_handlers.items():
                signal.signal(number, handler)
        print(json.dumps({key: result.get(key) for key in
                          ("job_id", "run_id", "status", "failure_category")}), flush=True)
        return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
