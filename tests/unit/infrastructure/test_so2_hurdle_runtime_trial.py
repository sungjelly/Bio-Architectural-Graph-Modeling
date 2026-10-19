from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from spatial_benchmark.configuration import ConfigurationError, validate_experiment_config
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256, create_run_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import ProjectWorkerLock, WorkerLockError, command_for_config
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, RunValidationError, verify_run_bundle
import spatial_benchmark.so2_hurdle_runtime_trial as trial


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> dict:
    source = {"seed": 0, "fold": 0, "attempt": 1, "model": {"name": "hurdle", "embedding_dim": 256},
              "dataset": {"dataset_id": "synthetic"}, "trainer": {"learning_rate": 0.0001},
              "masking": {"type": "partial_gene"}, "features": {}, "graph": {}}
    monkeypatch.setattr(trial, "SOURCE_CONFIG_SHA", canonical_sha256(source))
    receipt = {"run_id": trial.SOURCE_RUN, "success_marker_sha256": trial.SOURCE_SUCCESS_SHA,
               **{key: "a" * 64 for key in ("config_file_sha256", "manifest_sha256",
                  "checksum_manifest_sha256", "data_receipt_sha256", "split_receipt_sha256")}}
    return trial.build_configuration(source, receipt, gpu="0")


def test_bounded_config_command_and_frozen_source_pin(config: dict, tmp_path: Path) -> None:
    validate_experiment_config(config)
    command = command_for_config(config, paths=ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)}))
    assert command[-2:] == ["--epochs", "2"]
    assert "{run_scratch}/config.resolved.json" in command
    assert command[1].endswith("scripts/diagnostics/so2_hurdle_gpu_trial_worker.py")
    changed = deepcopy(config)
    changed["source_configuration"]["model"]["embedding_dim"] = 128
    with pytest.raises(ConfigurationError, match="exact frozen"):
        validate_experiment_config(changed)


@pytest.mark.parametrize("section,key,value", [
    ("runtime", "gpu", "0,1"), ("runtime", "epochs", 3),
    ("runtime", "cpu_threads", 8), ("trainer", "learning_rate", 0.1),
    ("classification", "lifecycle_stage", "locked_final"),
    ("launcher", "process_count", 4), ("evaluation", "artifact_contract", "predictive"),
])
def test_config_rejects_runtime_or_scientific_drift(config: dict, section: str, key: str, value: object) -> None:
    config[section][key] = value
    with pytest.raises(ConfigurationError):
        validate_experiment_config(config)


def _archive(tmp_path: Path, config: dict) -> RunArchive:
    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})
    archive = RunArchive.create(create_run_id(seed=0, fold=0, attempt=1), paths=paths, resolved_config=config)
    archive.write_manifest({"run_id": archive.run_id, "status": "completed"})
    archive.prepare_log_files()
    for name in ("git.json", "hardware.json", "data_fingerprints.json", "split_fingerprint.json"):
        archive.write_json("provenance/" + name, {"fixture": True})
    for name in ("uncommitted_changes.patch", "environment.txt", "command.txt"):
        archive.write_text("provenance/" + name, "synthetic\n")
    archive.write_json("provenance/source_bundle.json", config["source_bundle"])
    archive.write_json("diagnostics/runtime_environment.json", {"synthetic": True})
    archive.write_text("diagnostics/runtime_resources.jsonl", '{"synthetic":true}\n')
    archive.write_text("metrics/runtime_progress.jsonl", '{"synthetic":true}\n')
    epochs = []
    for number in (1, 2):
        receipt = f"diagnostics/coverage_epoch_{number:03d}.json"
        archive.write_json(receipt, {"synthetic": True, "epoch": number})
        epochs.append({"epoch": number, "optimizer_updates": 6, "train_target_cells": 208696,
                       "validation_target_cells": 37367, "finite_gradients": True,
                       "resource_gate_passed": True, "duration_seconds": 100.0,
                       "training_duration_seconds": 80.0, "validation_duration_seconds": 20.0,
                       "train_loss": 0.5, "validation_loss": 0.4,
                       "train_masked_entries": 100000, "validation_masked_entries": 30000,
                       "coverage_file": receipt, "coverage_sha256": sha256_file(archive.scratch_path / receipt),
                       "validation_mask_receipts_sha256": "b" * 64,
                       "peak_allocated_bytes": 10 * trial.GIB, "peak_reserved_bytes": 11 * trial.GIB,
                       "total_device_bytes": 24 * trial.GIB, "headroom_bytes": 13 * trial.GIB})
    archive.write_text("metrics/runtime_epoch_metrics.jsonl", "".join(json.dumps(row) + "\n" for row in epochs))
    archive.write_json("diagnostics/runtime_trial_result.json", {
        "schema": "so2_hurdle_gpu_runtime_trial_result_v1", "run_id": archive.run_id,
        "status": "passed", "epochs_requested": 2, "epochs_completed": 2,
        "effective_world_size": 1, "correctness_gate_passed": True,
        "resource_gate_passed": True, "epochs": epochs,
        **{key: epochs[-1][key] for key in ("peak_allocated_bytes", "peak_reserved_bytes", "total_device_bytes", "headroom_bytes")}})
    return archive


def test_runtime_bundle_finalizes_without_predictions_or_checkpoints(tmp_path: Path, config: dict) -> None:
    archive = _archive(tmp_path, config)
    trial.adapt_outputs(archive, config)
    path = archive.finalize_success()
    verify_run_bundle(path)
    assert (path / "_SUCCESS").is_file()
    assert not (path / "checkpoints").exists()
    assert json.loads((path / "metrics/final.json").read_text())[trial.PRIMARY_METRIC] == 100.0


@pytest.mark.parametrize("failure", ["receipt", "history", "memory", "missing", "checkpoint"])
def test_runtime_bundle_fail_closed(tmp_path: Path, config: dict, failure: str) -> None:
    archive = _archive(tmp_path, config)
    root = archive.scratch_path
    if failure == "receipt":
        (root / "diagnostics/coverage_epoch_001.json").write_text("{}")
    elif failure == "history":
        (root / "metrics/runtime_epoch_metrics.jsonl").write_text("{}\n")
    elif failure == "memory":
        path = root / "diagnostics/runtime_trial_result.json"
        result = json.loads(path.read_text())
        result["peak_reserved_bytes"] = 23 * trial.GIB
        path.write_text(json.dumps(result))
    elif failure == "missing":
        (root / "diagnostics/runtime_resources.jsonl").unlink()
    else:
        archive.write_bytes("checkpoints/last.ckpt", b"not allowed")
    with pytest.raises(ValueError):
        trial.adapt_outputs(archive, config)


def test_runtime_exception_cannot_be_reused_by_unrelated_protocol(tmp_path: Path, config: dict) -> None:
    archive = _archive(tmp_path, config)
    trial.adapt_outputs(archive, config)
    path = archive.scratch_path / "config.resolved.yaml"
    path.write_text(path.read_text().replace(trial.PROTOCOL, "unrelated"))
    with pytest.raises(RunValidationError):
        archive.finalize_success()


def test_exact_job_claim_keeps_other_jobs_queued(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry.sqlite3")
    registry.create_campaign("cmp_test", name="test")
    for name in ("q_first", "q_ours"):
        registry.enqueue(campaign_id="cmp_test", configuration={}, command=["unused"], job_id=name, requested_gpu="0")
    claimed = registry.claim_next("worker", requested_gpu="0", job_id="q_ours")
    assert claimed["job_id"] == "q_ours"
    assert registry.get_job("q_first")["status"] == "queued"
    assert registry.claim_next("worker2", requested_gpu="0", job_id="q_ours") is None
    assert registry.claim_next("worker3", requested_gpu="1", job_id="q_first") is None


def _launcher():
    file = Path(__file__).resolve().parents[3] / "scripts/diagnostics/run_so2_hurdle_gpu_trial.py"
    spec = importlib.util.spec_from_file_location("trial_launcher_test", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_lease_shares_between_pilots_but_excludes_global_worker(tmp_path: Path) -> None:
    launcher = _launcher()
    path = tmp_path / "bagm-worker.lock"
    with launcher.global_worker_lease(path), launcher.global_worker_lease(path):
        with pytest.raises(WorkerLockError):
            with ProjectWorkerLock(path):
                pass
    with ProjectWorkerLock(path):
        with pytest.raises(WorkerLockError):
            with launcher.global_worker_lease(path):
                pass
