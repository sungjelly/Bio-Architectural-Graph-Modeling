#!/usr/bin/env python3
"""Register/supervise fresh SO2 BCE target-minibatch pilots or full training.

Use the owning campaign README's BAGM path overrides. Full training requires
a verified, same-batch pilot from unchanged executable source and contract.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.diagnostics.run_so2_hurdle_gpu_trial import global_worker_lease, inspect_gpu
from spatial_benchmark.configuration import validate_experiment_config
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import ProjectWorkerLock, QueueWorker, WorkerSettings, command_for_config
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle
from spatial_benchmark.so2_binary_target_minibatch import (
    CAMPAIGN, EVALUATION_PROTOCOL, PREFLIGHT_RELATIVE_PATH, SOURCE_CONFIG_SHA,
    SOURCE_RUN, SOURCE_SUCCESS_SHA, TARGET_BATCH_SIZES, build_configuration,
    optimizer_updates, validate_configuration,
)

GIB = 1024**3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_files(paths: Any) -> dict[str, Any]:
    root = paths.project_root
    files = sorted(set((root / "src/spatial_benchmark").rglob("*.py"))
                   | set((root / "scripts/train").glob("*.py"))
                   | set((root / "scripts/diagnostics").glob("*.py")))
    return {"files": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path)} for path in files],
            "scope": "package, training and diagnostic Python sources; excludes evolving documentation and reports"}


def verified_bundle(registry: Registry, paths: Any, run_id: str) -> Path:
    record = registry.get_run(run_id)
    root = RunArchive.artifact_path_for(run_id, paths)
    require(bool(record) and record["status"] == "completed", f"Run is not completed: {run_id}")
    require(Path(record["artifact_path"]).resolve() == root.resolve() and (root / "_SUCCESS").is_file(),
            "Run is not at its canonical successful published path.")
    verify_run_bundle(root)
    return root


def load_source(registry: Registry, paths: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    root = verified_bundle(registry, paths, SOURCE_RUN)
    require(sha256_file(root / "_SUCCESS") == SOURCE_SUCCESS_SHA, "Frozen source completion marker changed.")
    source = yaml.safe_load((root / "config.resolved.yaml").read_text())
    require(canonical_sha256(source) == SOURCE_CONFIG_SHA, "Frozen source configuration changed.")
    names = {"config_file_sha256": "config.resolved.yaml", "manifest_sha256": "manifest.yaml",
             "success_marker_sha256": "_SUCCESS", "checksum_manifest_sha256": "provenance/artifact_checksums.json",
             "data_receipt_sha256": "provenance/data_fingerprints.json", "split_receipt_sha256": "provenance/split_fingerprint.json"}
    return source, {"run_id": SOURCE_RUN, "artifact_path": str(root), "configuration_sha256": SOURCE_CONFIG_SHA,
                    **{key: sha256_file(root / name) for key, name in names.items()}}


def verify_pilot(registry: Registry, paths: Any, run_id: str, *, batch_size: int,
                 contract_sha: str) -> dict[str, Any]:
    root = verified_bundle(registry, paths, run_id)
    config = yaml.safe_load((root / "config.resolved.yaml").read_text())
    validate_configuration(config)
    require(config["metadata"]["execution_profile"] == "pilot", "Full training must reference a diagnostic pilot.")
    require(config["trainer"]["target_cells_per_optimizer_update"] == batch_size, "Pilot batch size differs.")
    require(config["metadata"]["frozen_task_contract_sha256"] == contract_sha, "Frozen contract changed since the pilot.")
    require(json.loads((root / "provenance/source_files.json").read_text()) == source_files(paths),
            "Executable source changed since the pilot; audit and revalidate before full training.")
    receipt_path = root / PREFLIGHT_RELATIVE_PATH
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("schema") == "so2_binary_target_minibatch_preflight_v1"
            and receipt.get("run_id") == run_id
            and receipt.get("passed") is True and receipt.get("coverage_verified") is True
            and receipt.get("finite_gradients") is True and receipt.get("completed_epochs") == 2,
            "Pilot did not pass the required two-epoch correctness gates.")
    require(receipt.get("source_configuration_sha256") == SOURCE_CONFIG_SHA
            and receipt.get("frozen_task_contract_sha256") == contract_sha
            and receipt.get("target_cells_per_optimizer_update") == batch_size
            and receipt.get("optimizer_updates_per_global_epoch") == optimizer_updates(batch_size),
            "Pilot receipt scientific source or optimizer batching differs.")
    resources = receipt["resource_gates"]
    require(resources.get("resource_gate_passed") is True
            and 0 < resources["peak_allocated_bytes"] <= resources["peak_reserved_bytes"] <= 22 * GIB
            and 2 * GIB <= resources["headroom_bytes"] <= resources["minimum_sampled_physical_free_bytes"]
            and resources["headroom_bytes"] <= resources["total_device_bytes"] - resources["peak_reserved_bytes"],
            "Pilot did not pass the measured GPU memory gate.")
    require(0 < resources.get("maximum_sampled_cgroup_memory_bytes", 0) < 160 * GIB,
            "Pilot failed or omitted the combined sampled/boundary cgroup-memory gate.")
    samples = [json.loads(line) for line in (root / "diagnostics/target_minibatch_resource_samples.jsonl").read_text().splitlines() if line.strip()]
    memory = [sample["cgroup"]["memory.current"] for sample in samples
              if isinstance(sample.get("cgroup", {}).get("memory.current"), int)]
    require(bool(memory) and max(memory) < 160 * GIB, "Pilot failed or omitted shared cgroup-memory measurements.")
    summary = json.loads((root / "summary.json").read_text())
    require(summary.get("final_epoch") == 2 and summary.get("checkpoint_reload_verified") is True,
            "Pilot selected-checkpoint reload is not verified.")
    checkpoint_sha = sha256_file(root / "checkpoints/best.ckpt")
    checkpoints = registry.list_checkpoint_catalog(run_id=run_id, role="best", verification_status="verified")
    require(len(checkpoints) == 1 and checkpoints[0]["sha256"] == checkpoint_sha,
            "Pilot checkpoint is absent or inconsistent in the verified checkpoint catalog.")
    return {"run_id": run_id, "receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
            "success_marker_sha256": sha256_file(root / "_SUCCESS"), "checkpoint_sha256": checkpoint_sha,
            "source_configuration_sha256": SOURCE_CONFIG_SHA, "target_cells_per_optimizer_update": batch_size,
            "code_fingerprint_sha256": canonical_sha256(source_files(paths)), "maximum_sampled_cgroup_memory_bytes": max(memory)}


class MinibatchWorker(QueueWorker):
    admission: dict[str, Any]

    def _validate_job(self, configuration: Any, **kwargs: Any) -> None:
        super()._validate_job(configuration, **kwargs)
        metadata = configuration["metadata"]
        if metadata["execution_profile"] == "full":
            actual = verify_pilot(self.registry, self.paths, metadata["pilot_preflight"]["run_id"],
                batch_size=configuration["trainer"]["target_cells_per_optimizer_update"],
                contract_sha=metadata["frozen_task_contract_sha256"])
            require(actual == metadata["pilot_preflight"], "Pilot gate provenance changed after registration.")

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
        archive.write_json("provenance/source_bundle.json", config["source_bundle"])
        archive.write_json("provenance/gpu_admission.json", self.admission)
        archive.write_json("provenance/source_files.json", source_files(self.paths))
        contract = self.paths.project_root / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
        require(sha256_file(contract) == config["metadata"]["frozen_task_contract_sha256"], "Frozen contract changed before execution.")
        archive.copy_file(contract, "provenance/frozen_task_contract.md")
        for relative in ("scripts/train/launch_so2_binary_target_minibatch.py",
                         "scripts/train/run_so2_target_minibatch_binary_expression_bce.py",
                         "src/spatial_benchmark/so2_binary_target_minibatch.py",
                         "scripts/diagnostics/so2_hurdle_gpu_trial_worker.py"):
            archive.copy_file(self.paths.project_root / relative, "provenance/source_snapshot/" + relative)
        if config["metadata"]["pilot_preflight"]:
            archive.write_json("provenance/pilot_preflight_reference.json", config["metadata"]["pilot_preflight"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), required=True)
    parser.add_argument("--target-batch-size", type=int, choices=TARGET_BATCH_SIZES, default=1024)
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), default="0")
    parser.add_argument("--attempt", type=int, default=1, help="Positive execution attempt; failed attempts remain archived")
    parser.add_argument("--pilot-run", help="Required verified same-batch pilot for a fresh full run")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    require(args.attempt > 0, "--attempt must be a positive integer.")
    require((args.profile == "full") == bool(args.pilot_run), "Use --pilot-run exactly when --profile full.")
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    source, source_bundle = load_source(registry, paths)
    contract = paths.project_root / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
    contract_sha = sha256_file(contract)
    pilot = verify_pilot(registry, paths, args.pilot_run, batch_size=args.target_batch_size,
                         contract_sha=contract_sha) if args.pilot_run else None
    config = build_configuration(source, source_bundle, gpu=args.gpu, profile=args.profile,
        batch_size=args.target_batch_size, task_contract_sha256=contract_sha, pilot_preflight=pilot)
    config["attempt"] = args.attempt
    validate_experiment_config(config)
    command = command_for_config(config, paths=paths)
    if args.check_only:
        print(json.dumps({"check_only": True, "campaign": CAMPAIGN, "profile": args.profile,
                          "attempt": args.attempt,
                          "target_batch_size": args.target_batch_size,
                          "optimizer_updates_per_epoch": optimizer_updates(args.target_batch_size),
                          "gpu": args.gpu, "admission": inspect_gpu(args.gpu)}, sort_keys=True))
        return 0
    worker_id = f"so2-bce-minibatch-{args.profile}-gpu{args.gpu}-{uuid.uuid4().hex[:8]}"
    worker = MinibatchWorker(registry, settings=WorkerSettings(worker_id=worker_id, gpu=args.gpu,
        once=True, parallel_gpu_workers=True, min_free_gb=20, heartbeat_seconds=15), paths=paths)
    with global_worker_lease(paths.state_root / "locks/bagm-worker.lock"), ProjectWorkerLock(worker.lock_path):
        worker._assert_no_surviving_children()
        worker._assert_disk_space()
        worker.admission = inspect_gpu(args.gpu)
        registry.create_campaign(CAMPAIGN, name="SO2 BCE target-cell optimizer minibatches",
            scientific_question="How does core-local target minibatching affect BCE optimization within the fixed SO2 split?",
            config={"protocol": EVALUATION_PROTOCOL, "source_run_id": SOURCE_RUN,
                    "frozen_task_contract_sha256": contract_sha}, status="pilot")
        job = registry.enqueue(campaign_id=CAMPAIGN, configuration=config, command=command,
            requested_gpu=args.gpu, maximum_attempts=args.attempt, attempt_count=args.attempt,
            job_id=f"q_so2_bce_minibatch_{uuid.uuid4().hex}")
        claimed = registry.claim_next(worker_id, requested_gpu=args.gpu, job_id=job["job_id"])
        require(claimed is not None, "Could not claim the exact registered minibatch job.")
        handlers = {number: signal.signal(number, worker._signal_handler) for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            print(json.dumps({"job_id": job["job_id"], "worker_id": worker_id, "profile": args.profile,
                              "attempt": args.attempt,
                              "target_batch_size": args.target_batch_size, "gpu": args.gpu}), flush=True)
            result = worker.execute_claimed(claimed)
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
        print(json.dumps({key: result.get(key) for key in ("job_id", "run_id", "status", "failure_category")}), flush=True)
        return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
