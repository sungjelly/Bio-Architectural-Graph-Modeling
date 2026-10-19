from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from spatial_benchmark.configuration import ConfigurationError, validate_experiment_config
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import _analysis_only_artifact_contract, command_for_config
import spatial_benchmark.so2_binary_target_minibatch as contract


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> dict:
    value = {"seed": 0, "fold": 0, "attempt": 1,
             "campaign": {"campaign_id": "frozen_original"},
             "metadata": {"gradient_diagnostics": {"enabled": True}},
             "model": {"name": "binary_bce", "embedding_dim": 256},
             "dataset": {"dataset_id": "synthetic"}, "masking": {"count_min": 1, "count_max": 1000},
             "evaluation": {"primary_metric": contract.PRIMARY_METRIC, "primary_direction": "minimize"},
             "trainer": {"learning_rate": 1e-4, "primary_checkpoint_role": "best", "restore_best": True,
                         "early_stopping": True, "early_stopping_patience": 25,
                         "optimizer_steps_per_paired_core_update": 1}}
    monkeypatch.setattr(contract, "SOURCE_CONFIG_SHA", canonical_sha256(value))
    return value


def config(source: dict, *, profile: str = "pilot", batch: int = 1024, pilot: dict | None = None) -> dict:
    bundle = {"run_id": contract.SOURCE_RUN, "success_marker_sha256": contract.SOURCE_SUCCESS_SHA,
              **{key: "a" * 64 for key in ("config_file_sha256", "manifest_sha256",
                  "checksum_manifest_sha256", "data_receipt_sha256", "split_receipt_sha256")}}
    return contract.build_configuration(source, bundle, gpu="0", profile=profile, batch_size=batch,
        task_contract_sha256="b" * 64, pilot_preflight=pilot)


@pytest.mark.parametrize("batch,steps", [(512, 413), (1024, 209), (2048, 108)])
def test_pilot_contract_changes_batching_preserves_science(source: dict, batch: int, steps: int) -> None:
    result = config(source, batch=batch)
    validate_experiment_config(result)
    for key in ("model", "dataset", "masking", "source_configuration"):
        assert result[key] == (source if key == "source_configuration" else source[key])
    assert result["trainer"]["optimizer_updates_per_global_epoch"] == steps
    assert result["trainer"]["target_cells_per_optimizer_update"] == batch
    assert result["trainer"]["max_epochs"] == 2
    assert result["trainer"]["minimum_global_epochs"] == 1
    assert result["trainer"]["primary_checkpoint_role"] == "best"
    assert "optimizer_steps_per_paired_core_update" not in result["trainer"]
    assert not _analysis_only_artifact_contract(result)
    assert source["trainer"]["optimizer_steps_per_paired_core_update"] == 1


@pytest.mark.parametrize("group,key,value", [("trainer", "target_cells_per_optimizer_update", 1536),
    ("trainer", "learning_rate", 0.01), ("trainer", "optimizer_updates_per_global_epoch", 6),
    ("trainer", "max_epochs", 300), ("launcher", "requested_gpu", "0,1"),
    ("evaluation", "artifact_contract", "runtime_diagnostic"), ("model", "embedding_dim", 128)])
def test_config_rejects_unplanned_changes(source: dict, group: str, key: str, value: object) -> None:
    result = config(source)
    result[group][key] = value
    with pytest.raises(ConfigurationError):
        validate_experiment_config(result)


def test_full_requires_matching_pilot_and_keeps_original_stopping(source: dict) -> None:
    with pytest.raises(ConfigurationError, match="verified pilot"):
        validate_experiment_config(config(source, profile="full"))
    pilot = {"run_id": "r_pilot", "source_configuration_sha256": contract.SOURCE_CONFIG_SHA,
             "target_cells_per_optimizer_update": 1024, "receipt_path": "/artifact/" + contract.PREFLIGHT_RELATIVE_PATH,
             "receipt_sha256": "a" * 64, "success_marker_sha256": "b" * 64, "checkpoint_sha256": "c" * 64}
    result = config(source, profile="full", pilot=pilot)
    validate_experiment_config(result)
    assert result["trainer"]["max_epochs"] == 300
    assert result["trainer"]["minimum_global_epochs"] == 50
    assert result["trainer"]["early_stopping_patience"] == 25
    assert result["classification"]["lifecycle_stage"] == "exploratory_screen"
    pilot["target_cells_per_optimizer_update"] = 2048
    with pytest.raises(ConfigurationError, match="incompatible"):
        validate_experiment_config(config(source, profile="full", pilot=pilot))


def test_command_is_guarded_one_rank_fresh_predictive_entrypoint(source: dict, tmp_path: Path) -> None:
    result = config(source)
    command = command_for_config(result, paths=ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)}))
    assert "--nproc-per-node=1" in command
    assert any(part.endswith("run_so2_target_minibatch_binary_expression_bce.py") for part in command)
    assert command[-4:] == ["--config", "{run_scratch}/config.resolved.yaml", "--run-scratch", "{run_scratch}"]
    assert "--resume-checkpoint" not in command


def test_second_attempt_is_execution_metadata_and_validated(source: dict) -> None:
    result = config(source)
    result["attempt"] = 2
    validate_experiment_config(result)
    assert result["source_configuration"]["attempt"] == 1
    result["attempt"] = 0
    with pytest.raises(ConfigurationError, match="positive execution attempt"):
        validate_experiment_config(result)


def launcher():
    path = Path(__file__).resolve().parents[3] / "scripts/train/launch_so2_binary_target_minibatch.py"
    spec = importlib.util.spec_from_file_location("minibatch_launcher_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pilot_gate(tmp_path: Path, source: dict, monkeypatch: pytest.MonkeyPatch):
    module = launcher()
    root = tmp_path / "pilot"
    (root / "diagnostics").mkdir(parents=True)
    (root / "provenance").mkdir()
    (root / "checkpoints").mkdir()
    (root / "config.resolved.yaml").write_text(yaml.safe_dump(config(source)))
    (root / "_SUCCESS").write_text("fixture")
    (root / "checkpoints/best.ckpt").write_bytes(b"synthetic fixture")
    (root / "provenance/source_files.json").write_text('{"files":[]}')
    (root / "summary.json").write_text(json.dumps({"final_epoch": 2, "checkpoint_reload_verified": True}))
    receipt = {"schema": "so2_binary_target_minibatch_preflight_v1", "passed": True,
               "run_id": "r_pilot", "frozen_task_contract_sha256": "b" * 64,
               "source_configuration_sha256": contract.SOURCE_CONFIG_SHA,
               "target_cells_per_optimizer_update": 1024, "optimizer_updates_per_global_epoch": 209,
               "completed_epochs": 2, "coverage_verified": True, "finite_gradients": True,
               "resource_gates": {"resource_gate_passed": True, "peak_allocated_bytes": 10 * module.GIB,
                   "peak_reserved_bytes": 12 * module.GIB, "headroom_bytes": 11 * module.GIB,
                   "minimum_sampled_physical_free_bytes": 11 * module.GIB, "total_device_bytes": 24 * module.GIB,
                   "maximum_sampled_cgroup_memory_bytes": 40 * module.GIB}}
    (root / contract.PREFLIGHT_RELATIVE_PATH).write_text(json.dumps(receipt))
    (root / "diagnostics/target_minibatch_resource_samples.jsonl").write_text(json.dumps({"cgroup": {"memory.current": 40 * module.GIB}}) + "\n")
    registry = SimpleNamespace(list_checkpoint_catalog=lambda **_: [{"sha256": sha256_file(root / "checkpoints/best.ckpt")}])
    monkeypatch.setattr(module, "verified_bundle", lambda *_: root)
    monkeypatch.setattr(module, "source_files", lambda _: {"files": []})
    return module, root, registry


def test_completed_pilot_gate_binds_code_resource_and_catalog(pilot_gate) -> None:
    module, root, registry = pilot_gate
    receipt = module.verify_pilot(registry, object(), "r_pilot", batch_size=1024, contract_sha="b" * 64)
    assert receipt["checkpoint_sha256"] == sha256_file(root / "checkpoints/best.ckpt")
    assert receipt["target_cells_per_optimizer_update"] == 1024


@pytest.mark.parametrize("failure", ["code", "batch", "vram", "cgroup", "checkpoint", "contract"])
def test_pilot_gate_rejects_invalid_full_run_preconditions(pilot_gate, failure: str) -> None:
    module, root, registry = pilot_gate
    if failure == "code":
        (root / "provenance/source_files.json").write_text('{"files":["changed"]}')
    elif failure == "vram":
        path = root / contract.PREFLIGHT_RELATIVE_PATH
        receipt = json.loads(path.read_text())
        receipt["resource_gates"]["headroom_bytes"] = module.GIB
        path.write_text(json.dumps(receipt))
    elif failure == "cgroup":
        (root / "diagnostics/target_minibatch_resource_samples.jsonl").write_text(json.dumps({"cgroup": {"memory.current": 161 * module.GIB}}) + "\n")
    elif failure == "checkpoint":
        registry.list_checkpoint_catalog = lambda **_: []
    with pytest.raises(ValueError):
        module.verify_pilot(registry, object(), "r_pilot", batch_size=2048 if failure == "batch" else 1024,
                            contract_sha="c" * 64 if failure == "contract" else "b" * 64)
