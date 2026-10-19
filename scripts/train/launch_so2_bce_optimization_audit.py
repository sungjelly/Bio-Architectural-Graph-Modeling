#!/usr/bin/env python3
"""Registered isolated launcher for the additive fixed-budget BCE audit.

This launcher owns validation/command derivation for its new protocol, leaving
the frozen generic configuration and command routes unchanged. Predictive
artifact requirements, registry provenance and checkpoint indexing are reused.
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
from scripts.train.launch_so2_binary_target_minibatch import load_source, source_files, verified_bundle
from spatial_benchmark.configuration import ConfigurationError
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import (
    MissingDatasetError, ProjectWorkerLock, QueueWorker, WorkerSettings,
    _resolve_prepared_artifact_reference,
)
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive
from spatial_benchmark.so2_bce_optimization_audit import (
    BOUNDARY_LEARNING_RATES, CAMPAIGN, DROPOUT_VALUES, EVALUATION_PROTOCOL,
    GRID_LEARNING_RATES, PREFLIGHT_RELATIVE_PATH, PREFLIGHT_SCHEMA, PROFILE_EPOCHS,
    SOURCE_CONFIG_SHA, SOURCE_RUN, TARGET_BATCH_SIZES, WEIGHT_DECAYS,
    build_configuration, command_for_configuration, validate_configuration,
)

GIB = 1024**3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def verify_smoke(registry: Registry, paths: Any, run_id: str, *, mode: str,
                 contract_sha: str) -> dict[str, Any]:
    root = verified_bundle(registry, paths, run_id)
    config = yaml.safe_load((root / "config.resolved.yaml").read_text())
    validate_configuration(config)
    require(config["metadata"]["execution_profile"] == "smoke", "Audit requires its own completed smoke profile.")
    require(config["trainer"]["optimizer_batch_mode"] == mode, "Smoke optimizer mode differs.")
    require(config["metadata"]["frozen_task_contract_sha256"] == contract_sha, "Audit contract changed after smoke.")
    require(json.loads((root / "provenance/source_files.json").read_text()) == source_files(paths),
            "Executable source changed after smoke; audit/revalidate before screening.")
    path = root / PREFLIGHT_RELATIVE_PATH
    receipt = json.loads(path.read_text())
    require(receipt.get("schema") == PREFLIGHT_SCHEMA and receipt.get("run_id") == run_id
            and receipt.get("passed") is True and receipt.get("coverage_verified") is True
            and receipt.get("finite_gradients") is True and receipt.get("completed_epochs") == 2,
            "New-protocol smoke correctness gates are incomplete.")
    require(receipt.get("source_configuration_sha256") == SOURCE_CONFIG_SHA
            and receipt.get("frozen_task_contract_sha256") == contract_sha
            and receipt.get("optimizer_batch_mode") == mode, "Smoke receipt source, contract or mode differs.")
    resources = receipt["resource_gates"]
    require(resources.get("resource_gate_passed") is True
            and 0 < resources["peak_allocated_bytes"] <= resources["peak_reserved_bytes"] <= 22 * GIB
            and 2 * GIB <= resources["headroom_bytes"] <= resources["minimum_sampled_physical_free_bytes"]
            and resources["headroom_bytes"] <= resources["total_device_bytes"] - resources["peak_reserved_bytes"]
            and 0 < resources["maximum_sampled_cgroup_memory_bytes"] < 160 * GIB,
            "Smoke resource gates failed or lack measured headroom.")
    baseline = json.loads((root / "diagnostics/train_fitted_gene_prevalence_validation.json").read_text())
    require(baseline.get("schema") == "so2_train_fitted_gene_prevalence_validation_v1"
            and baseline.get("fit_scope") == "training_cores_only" and baseline.get("validation_fitted") is False
            and canonical_sha256(baseline) == receipt.get("prevalence_reference_sha256"),
            "Smoke is missing its verified train-fitted per-gene prevalence reference.")
    samples = [json.loads(line) for line in
               (root / "diagnostics/optimization_audit_resource_samples.jsonl").read_text().splitlines()]
    require(samples and any("physical_memory_free_bytes" in row for row in samples),
            "Smoke lacks sampled physical GPU memory evidence.")
    require(all(0 < row["cgroup"]["memory.current"] <= resources["maximum_sampled_cgroup_memory_bytes"]
                for row in samples), "Smoke cgroup samples disagree with its resource gate.")
    summary = json.loads((root / "summary.json").read_text())
    require(summary.get("final_epoch") == 2 and summary.get("checkpoint_reload_verified") is True,
            "Smoke selected checkpoint was not reloaded and verified.")
    checkpoint_sha = sha256_file(root / "checkpoints/best.ckpt")
    indexed = registry.list_checkpoint_catalog(run_id=run_id, role="best", verification_status="verified")
    require(len(indexed) == 1 and indexed[0]["sha256"] == checkpoint_sha, "Smoke best checkpoint is not verified in the catalog.")
    return {"run_id": run_id, "receipt_path": str(path), "receipt_sha256": sha256_file(path),
            "success_marker_sha256": sha256_file(root / "_SUCCESS"), "checkpoint_sha256": checkpoint_sha,
            "source_configuration_sha256": SOURCE_CONFIG_SHA, "optimizer_batch_mode": mode,
            "code_fingerprint_sha256": canonical_sha256(source_files(paths))}


def validate_registered_inputs(registry: Registry, paths: Any, config: dict[str, Any]) -> None:
    """Retain every generic worker dataset/split/prepared-input admission gate."""
    dataset = config["dataset"]
    record = registry.get_dataset(str(dataset["dataset_id"]), str(dataset["version"]))
    if record is None:
        raise MissingDatasetError("Audit dataset/version is not registered.")
    fingerprints = {str(record[key]) for key in ("raw_fingerprint", "processed_fingerprint") if record.get(key)}
    require(bool(dataset.get("dataset_fingerprint")) and dataset["dataset_fingerprint"] in fingerprints,
            "Audit dataset fingerprint differs from the authoritative registry.")
    require(str(dataset.get("preprocessing_version", "")) == str(record.get("preprocessing_version", "")),
            "Audit preprocessing version differs from the registry.")
    split = registry.get_split(str(dataset["split_id"]))
    if split is None:
        raise MissingDatasetError("Audit split is not registered.")
    require(split["dataset_id"] == dataset["dataset_id"] and split["dataset_version"] == dataset["version"],
            "Audit split does not belong to the configured dataset/version.")
    require(str(dataset.get("split_fingerprint", "")) == str(split.get("fingerprint", "")),
            "Audit split fingerprint differs from the registry.")
    reference = dataset.get("prepared_artifact_reference")
    if reference and not _resolve_prepared_artifact_reference(reference, paths=paths).exists():
        raise MissingDatasetError("Audit prepared input artifact does not exist.")


class AuditWorker(QueueWorker):
    admission: dict[str, Any]

    def _validate_job(self, configuration: Any, *, queued_command: Any, requested_gpu: Any) -> None:
        try:
            validate_configuration(configuration)
        except ValueError as error:
            raise ConfigurationError(str(error)) from error
        require(str(requested_gpu) == str(self.settings.gpu) == configuration["launcher"]["requested_gpu"],
                "Audit worker and job GPU assignments differ.")
        require(list(queued_command) == command_for_configuration(configuration, self.paths.project_root),
                "Audit queued argv differs from the guarded configuration-derived command.")
        validate_registered_inputs(self.registry, self.paths, configuration)
        metadata = configuration["metadata"]
        if metadata["execution_profile"] != "smoke":
            actual = verify_smoke(self.registry, self.paths, metadata["smoke_preflight"]["run_id"],
                mode=configuration["trainer"]["optimizer_batch_mode"], contract_sha=metadata["frozen_task_contract_sha256"])
            require(actual == metadata["smoke_preflight"], "Smoke provenance changed after registration.")

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
        archive.write_json("provenance/source_files.json", source_files(self.paths))
        archive.write_json("provenance/gpu_admission.json", self.admission)
        frozen = self.paths.project_root / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
        require(sha256_file(frozen) == config["metadata"]["frozen_task_contract_sha256"], "Audit contract changed before execution.")
        archive.copy_file(frozen, "provenance/frozen_task_contract.md")
        for relative in ("scripts/train/launch_so2_bce_optimization_audit.py",
                         "scripts/train/run_so2_bce_optimization_audit.py",
                         "src/spatial_benchmark/so2_bce_optimization_audit.py"):
            archive.copy_file(self.paths.project_root / relative, "provenance/source_snapshot/" + relative)
        if config["metadata"]["smoke_preflight"]:
            archive.write_json("provenance/smoke_preflight_reference.json", config["metadata"]["smoke_preflight"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_EPOCHS), required=True)
    parser.add_argument("--batch-mode", choices=("target_window", "core_pair"), default="target_window")
    parser.add_argument("--target-batch-size", type=int, choices=TARGET_BATCH_SIZES)
    parser.add_argument("--learning-rate", type=float, choices=GRID_LEARNING_RATES + BOUNDARY_LEARNING_RATES, default=1e-4)
    parser.add_argument("--dropout", type=float, choices=DROPOUT_VALUES, default=0.1)
    parser.add_argument("--weight-decay", type=float, choices=WEIGHT_DECAYS, default=1e-5)
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--gpu", choices=("0", "1", "2", "3"), default="0")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--smoke-run")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    require(args.attempt > 0, "Execution attempt must be positive.")
    require((args.profile != "smoke") == bool(args.smoke_run), "Screen/confirmation require --smoke-run; smoke must omit it.")
    require(args.batch_mode != "core_pair" or args.target_batch_size is None, "Paired-core mode must omit --target-batch-size.")
    batch = (args.target_batch_size or 1024) if args.batch_mode == "target_window" else None
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    source, bundle = load_source(registry, paths)
    contract = paths.project_root / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
    contract_sha = sha256_file(contract)
    smoke = verify_smoke(registry, paths, args.smoke_run, mode=args.batch_mode, contract_sha=contract_sha) if args.smoke_run else None
    config = build_configuration(source, bundle, profile=args.profile, batch_mode=args.batch_mode,
        batch_size=batch, learning_rate=args.learning_rate, dropout=args.dropout, weight_decay=args.weight_decay,
        seed=args.seed, gpu=args.gpu, task_contract_sha256=contract_sha, smoke_preflight=smoke)
    config["attempt"] = args.attempt
    validate_configuration(config)
    validate_registered_inputs(registry, paths, config)
    command = command_for_configuration(config, paths.project_root)
    if args.check_only:
        print(json.dumps({"check_only": True, "profile": args.profile, "batch_mode": args.batch_mode,
                          "batch_size": batch, "seed": args.seed, "epochs": config["trainer"]["max_epochs"],
                          "optimizer_updates_per_epoch": config["trainer"]["optimizer_updates_per_global_epoch"],
                          "disk_floor_gib": 5, "admission": inspect_gpu(args.gpu)}, sort_keys=True))
        return 0
    worker_id = f"so2-bce-audit-{args.profile}-gpu{args.gpu}-{uuid.uuid4().hex[:8]}"
    worker = AuditWorker(registry, settings=WorkerSettings(worker_id=worker_id, gpu=args.gpu,
        once=True, parallel_gpu_workers=True, min_free_gb=5, heartbeat_seconds=15), paths=paths)
    with global_worker_lease(paths.state_root / "locks/bagm-worker.lock"), ProjectWorkerLock(worker.lock_path):
        worker._assert_no_surviving_children()
        worker._assert_disk_space()
        worker.admission = inspect_gpu(args.gpu)
        registry.create_campaign(CAMPAIGN, name="SO2 BCE optimization audit",
            scientific_question="Do batch size, learning rate or regularization materially improve fixed-split BCE at matched exposure?",
            config={"protocol": EVALUATION_PROTOCOL, "source_run_id": SOURCE_RUN,
                    "frozen_task_contract_sha256": contract_sha}, status="pilot")
        job = registry.enqueue(campaign_id=CAMPAIGN, configuration=config, command=command,
            requested_gpu=args.gpu, maximum_attempts=args.attempt, attempt_count=args.attempt,
            job_id=f"q_so2_bce_audit_{uuid.uuid4().hex}")
        claimed = registry.claim_next(worker_id, requested_gpu=args.gpu, job_id=job["job_id"])
        require(claimed is not None, "Cannot claim the exact newly registered audit job.")
        handlers = {number: signal.signal(number, worker._signal_handler) for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            print(json.dumps({"job_id": job["job_id"], "worker_id": worker_id, "profile": args.profile,
                              "batch_mode": args.batch_mode, "batch_size": batch, "seed": args.seed,
                              "attempt": args.attempt, "gpu": args.gpu}), flush=True)
            result = worker.execute_claimed(claimed)
        finally:
            for number, handler in handlers.items():
                signal.signal(number, handler)
        print(json.dumps({key: result.get(key) for key in ("job_id", "run_id", "status", "failure_category")}), flush=True)
        return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
