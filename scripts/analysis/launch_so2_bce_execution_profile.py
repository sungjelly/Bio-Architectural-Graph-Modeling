#!/usr/bin/env python3
"""Register and supervise the bounded, one-epoch BCE execution profile.

See experiments/campaigns/cmp_20261001_so2_bce_execution_profile/README.md.
Run only on the idle device explicitly allocated by the owning supervisor.
The parent smoke, frozen training files and three additive instrumentation
files are checksum-bound. Normal predictive publication and checkpoint
catalog indexing remain mandatory. --check-only performs read-only admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
from typing import Any
import uuid

import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from scripts.analysis.so2_bce_execution_profile_contract import (
    CAMPAIGN, PARENT_CONFIG_SHA, PARENT_RUN, PARENT_SUCCESS_SHA,
    build_configuration, command_for_configuration, validate_configuration,
)
from scripts.diagnostics.run_so2_hurdle_gpu_trial import global_worker_lease, inspect_gpu
from scripts.train.launch_so2_bce_optimization_audit import validate_registered_inputs, verify_smoke
from scripts.train.launch_so2_binary_target_minibatch import source_files
from spatial_benchmark.configuration import ConfigurationError
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import ProjectWorkerLock, QueueWorker, WorkerSettings
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive

INSTRUMENTATION_FILES = (
    "scripts/analysis/launch_so2_bce_execution_profile.py",
    "scripts/analysis/so2_bce_execution_profile_contract.py",
    "scripts/analysis/profile_so2_bce_execution.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def instrumentation_hashes(paths: Any) -> dict[str, str]:
    return {relative: sha256_file(paths.project_root / relative) for relative in INSTRUMENTATION_FILES}


def load_parent(registry: Registry, paths: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    root = RunArchive.artifact_path_for(PARENT_RUN, paths)
    parent = yaml.safe_load((root / "config.resolved.yaml").read_text())
    require(canonical_sha256(parent) == PARENT_CONFIG_SHA, "Pinned parent scientific configuration changed.")
    require(sha256_file(root / "_SUCCESS") == PARENT_SUCCESS_SHA, "Pinned parent completion marker changed.")
    smoke = verify_smoke(registry, paths, PARENT_RUN, mode="target_window",
                         contract_sha=parent["metadata"]["frozen_task_contract_sha256"])
    recorded = json.loads((root / "provenance/source_files.json").read_text())
    require(recorded == source_files(paths), "Frozen training fingerprint changed after the parent smoke.")
    require(smoke["code_fingerprint_sha256"] == canonical_sha256(recorded), "Parent code fingerprint is inconsistent.")
    names = {"config_file_sha256": "config.resolved.yaml", "manifest_sha256": "manifest.yaml",
             "success_marker_sha256": "_SUCCESS", "checksum_manifest_sha256": "provenance/artifact_checksums.json",
             "epoch_metrics_sha256": "results/epoch_metrics.csv"}
    reference = {"run_id": PARENT_RUN, "artifact_path": str(root), "configuration_sha256": PARENT_CONFIG_SHA,
        "receipt_path": smoke["receipt_path"], "receipt_sha256": smoke["receipt_sha256"],
        "code_fingerprint_sha256": smoke["code_fingerprint_sha256"],
        **{key: sha256_file(root / relative) for key, relative in names.items()}}
    return parent, reference


class ExecutionProfileWorker(QueueWorker):
    admission: dict[str, Any]

    def _validate_job(self, configuration: Any, *, queued_command: Any, requested_gpu: Any) -> None:
        try:
            validate_configuration(configuration)
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        require(str(requested_gpu) == str(self.settings.gpu) == configuration["launcher"]["requested_gpu"],
                "Profiler worker and job GPU assignments differ.")
        require(list(queued_command) == command_for_configuration(configuration, self.paths.project_root),
                "Profiler argv differs from its guarded configuration-derived command.")
        validate_registered_inputs(self.registry, self.paths, configuration)
        parent, reference = load_parent(self.registry, self.paths)
        require(parent == configuration["source_profile_configuration"] and reference == configuration["metadata"]["profile_parent"],
                "Profiler parent provenance changed after registration.")
        require(instrumentation_hashes(self.paths) == configuration["metadata"]["instrumentation_hashes"],
                "Profiler instrumentation changed after registration.")
        frozen = self.paths.project_root / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
        require(sha256_file(frozen) == configuration["metadata"]["frozen_task_contract_sha256"],
                "Profiler frozen contract changed after registration.")

    def _subprocess_environment(self, **kwargs: Any) -> dict[str, str]:
        environment = super()._subprocess_environment(**kwargs)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            environment[name] = "4"
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(name, None)
        return environment

    def _write_provenance(self, archive: RunArchive, **kwargs: Any) -> None:
        super()._write_provenance(archive, **kwargs)
        config = kwargs["configuration"]
        require(instrumentation_hashes(self.paths) == config["metadata"]["instrumentation_hashes"],
                "Instrumentation changed before provenance snapshot.")
        archive.write_json("provenance/parent_reference.json", config["metadata"]["profile_parent"])
        archive.write_json("provenance/source_files.json", source_files(self.paths))
        archive.write_json("provenance/instrumentation_files.json", config["metadata"]["instrumentation_hashes"])
        archive.write_json("provenance/gpu_admission.json", self.admission)
        frozen = self.paths.project_root / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
        require(sha256_file(frozen) == config["metadata"]["frozen_task_contract_sha256"], "Profiler frozen contract changed.")
        archive.copy_file(frozen, "provenance/frozen_task_contract.md")
        for relative in INSTRUMENTATION_FILES:
            archive.copy_file(self.paths.project_root / relative, "provenance/source_snapshot/" + relative)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    require(args.attempt > 0, "--attempt must be positive.")
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    parent, reference = load_parent(registry, paths)
    frozen = paths.project_root / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
    contract_sha = sha256_file(frozen)
    config = build_configuration(parent, reference, gpu=args.gpu, task_contract_sha256=contract_sha,
                                 instrumentation_hashes=instrumentation_hashes(paths))
    config["attempt"] = args.attempt
    validate_configuration(config)
    validate_registered_inputs(registry, paths, config)
    command = command_for_configuration(config, paths.project_root)
    if args.check_only:
        print(json.dumps({"check_only": True, "parent_run_id": PARENT_RUN, "gpu": args.gpu,
            "epochs": config["trainer"]["max_epochs"], "optimizer_updates": 209, "disk_floor_gib": 5,
            "command": command, "instrumentation_hashes": config["metadata"]["instrumentation_hashes"],
            "admission": inspect_gpu(args.gpu)}, sort_keys=True))
        return 0
    worker_id = f"so2-bce-execution-profile-gpu{args.gpu}-{uuid.uuid4().hex[:8]}"
    worker = ExecutionProfileWorker(registry, paths=paths, settings=WorkerSettings(worker_id=worker_id,
        gpu=args.gpu, once=True, parallel_gpu_workers=True, min_free_gb=5, heartbeat_seconds=15))
    with global_worker_lease(paths.state_root / "locks/bagm-worker.lock"), ProjectWorkerLock(worker.lock_path):
        worker._assert_no_surviving_children()
        worker._assert_disk_space()
        worker.admission = inspect_gpu(args.gpu)
        registry.create_campaign(CAMPAIGN, name="SO2 BCE bounded execution profile",
            scientific_question="Which measured CPU/CUDA execution costs explain the observed target-window training utilization?",
            config={"parent_run_id": PARENT_RUN, "frozen_task_contract_sha256": contract_sha}, status="pilot")
        job = registry.enqueue(campaign_id=CAMPAIGN, configuration=config, command=command,
            requested_gpu=args.gpu, maximum_attempts=args.attempt, attempt_count=args.attempt,
            job_id=f"q_so2_bce_execution_profile_{uuid.uuid4().hex}")
        claimed = registry.claim_next(worker_id, requested_gpu=args.gpu, job_id=job["job_id"])
        require(claimed is not None, "Cannot claim the exact registered profiler job.")
        handlers = {number: signal.signal(number, worker._signal_handler) for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            print(json.dumps({"job_id": job["job_id"], "worker_id": worker_id, "gpu": args.gpu,
                              "attempt": args.attempt, "parent_run_id": PARENT_RUN}), flush=True)
            result = worker.execute_claimed(claimed)
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
        print(json.dumps({key: result.get(key) for key in ("job_id", "run_id", "status", "failure_category")}), flush=True)
        return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
