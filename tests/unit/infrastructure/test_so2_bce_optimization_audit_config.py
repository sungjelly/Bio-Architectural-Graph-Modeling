"""Synthetic admission and ordinary predictive lifecycle checks for the audit."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from spatial_benchmark.checkpoint_catalog import index_checkpoint_catalog
from spatial_benchmark.configuration import ConfigurationError
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256, create_run_id, scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import WorkerSettings, _analysis_only_artifact_contract
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, RunValidationError, verify_run_bundle
import spatial_benchmark.so2_bce_optimization_audit as contract


@pytest.fixture
def source(monkeypatch):
    value = {"seed": 0, "fold": 0, "attempt": 1,
        "campaign": {"campaign_id": "frozen_synthetic"},
        "metadata": {"gradient_diagnostics": {"enabled": True}},
        "model": {"name": "binary_bce", "embedding_dim": 256},
        "dataset": {"dataset_id": "synthetic", "version": "v1", "dataset_fingerprint": "dataset-sha",
                    "preprocessing_version": "prep1", "split_id": "split1", "split_fingerprint": "split-sha"},
        "masking": {"count_min": 1, "count_max": 1000},
        "evaluation": {"primary_metric": contract.PRIMARY_METRIC, "primary_direction": "minimize"},
        "trainer": {"learning_rate": 1e-4, "primary_checkpoint_role": "best", "restore_best": True,
                    "early_stopping": True, "early_stopping_patience": 25,
                    "optimizer_steps_per_paired_core_update": 1}}
    monkeypatch.setattr(contract, "SOURCE_CONFIG_SHA", canonical_sha256(value))
    return value


def configuration(source, *, profile="smoke", mode="target_window", batch=1024,
                  lr=1e-4, dropout=.1, wd=1e-5, seed=0):
    bundle = {"run_id": contract.SOURCE_RUN, "success_marker_sha256": contract.SOURCE_SUCCESS_SHA,
        **{key: "a"*64 for key in ("config_file_sha256", "manifest_sha256", "checksum_manifest_sha256",
                                  "data_receipt_sha256", "split_receipt_sha256")}}
    smoke = None if profile == "smoke" else {"run_id": "r_smoke",
        "source_configuration_sha256": contract.SOURCE_CONFIG_SHA, "optimizer_batch_mode": mode,
        "receipt_path": "/synthetic/"+contract.PREFLIGHT_RELATIVE_PATH,
        "receipt_sha256": "b"*64, "success_marker_sha256": "c"*64, "checkpoint_sha256": "d"*64}
    return contract.build_configuration(source, bundle, profile=profile, batch_mode=mode,
        batch_size=None if mode == "core_pair" else batch, learning_rate=lr,
        dropout=dropout, weight_decay=wd, seed=seed, gpu="0", task_contract_sha256="e"*64,
        smoke_preflight=smoke)


@pytest.mark.parametrize("profile,epochs", [("smoke",2), ("screen",8), ("confirmation",32)])
def test_budget_and_predictive_contract(source, profile, epochs):
    config = configuration(source, profile=profile)
    contract.validate_configuration(config)
    assert config["trainer"]["minimum_global_epochs"] == 1
    assert config["trainer"]["max_epochs"] == epochs
    assert config["trainer"]["early_stopping_patience"] > epochs
    assert config["launcher"]["disk_safety_min_free_gb"] == 5
    assert config["metadata"]["preflight_acceptance"]["minimum_free_disk_gib"] == 5
    assert config["metadata"]["evaluate_fixed_training_at_best"] == (profile == "confirmation")
    assert not _analysis_only_artifact_contract(config)
    for key in ("dataset", "masking"):
        assert config[key] == source[key]
    assert config["source_configuration"] == source


@pytest.mark.parametrize("batch,steps", [(512,413), (1024,209), (2048,108)])
@pytest.mark.parametrize("lr", contract.GRID_LEARNING_RATES)
def test_full_grid_is_guarded_and_exposure_budget_fixed(source, batch, steps, lr):
    config = configuration(source, profile="screen", batch=batch, lr=lr)
    contract.validate_configuration(config)
    assert config["trainer"]["optimizer_updates_per_global_epoch"] == steps
    assert config["experiment"]["screening_endpoint_epochs"] == [7,8]


def test_core_pair_uses_original_six_updates_and_mode_matched_smoke(source):
    config = configuration(source, profile="screen", mode="core_pair")
    contract.validate_configuration(config)
    assert config["trainer"]["optimizer_updates_per_global_epoch"] == 6
    assert config["trainer"]["target_cells_per_optimizer_update"] is None
    assert config["trainer"]["cores_staged_sequentially_per_optimizer_update"] is True
    config["metadata"]["smoke_preflight"]["optimizer_batch_mode"] = "target_window"
    with pytest.raises(ValueError, match="incompatible"):
        contract.validate_configuration(config)


@pytest.mark.parametrize("kwargs", [{"profile":"smoke","batch":512}, {"profile":"screen","seed":1},
    {"profile":"confirmation","mode":"core_pair"}, {"profile":"screen","dropout":.3,"wd":.01},
    {"profile":"screen","lr":1e-3,"dropout":0.}, {"profile":"screen","batch":256}])
def test_rejects_unplanned_axes(source, kwargs):
    with pytest.raises(ValueError):
        contract.validate_configuration(configuration(source, **kwargs))


@pytest.mark.parametrize("group,key,value", [("model","embedding_dim",128), ("trainer","max_epochs",48),
    ("launcher","disk_safety_min_free_gb",1), ("evaluation","artifact_contract","runtime_diagnostic"),
    ("trainer","optimizer_updates_per_global_epoch",6), ("dataset","split_id","different")])
def test_rejects_scientific_or_artifact_contract_drift(source, group, key, value):
    config = configuration(source)
    config[group][key] = value
    with pytest.raises(ValueError):
        contract.validate_configuration(config)


def launcher():
    path = Path(__file__).resolve().parents[3]/"scripts/train/launch_so2_bce_optimization_audit.py"
    spec = importlib.util.spec_from_file_location("audit_launcher_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registry(tmp_path):
    paths = ProjectPaths.from_environment({"BAGM_ROOT":str(tmp_path)})
    registry = Registry(paths.state_root/"tracking/bagm.sqlite3")
    registry.create_campaign(contract.CAMPAIGN, name="Synthetic audit fixture")
    registry.register_dataset("synthetic","v1",display_name="Synthetic fixture",processed_fingerprint="dataset-sha",
                              preprocessing_version="prep1")
    registry.register_split("split1",dataset_id="synthetic",dataset_version="v1",method="fixed",
                            unit="sample",fingerprint="split-sha")
    return registry,paths


def test_registered_admission_enforces_exact_command_gpu_and_dataset(source, registry):
    db,paths = registry
    module = launcher()
    config = configuration(source)
    command = contract.command_for_configuration(config, paths.project_root)
    worker = module.AuditWorker(db, paths=paths, settings=WorkerSettings(worker_id="fixture", gpu="0",
        parallel_gpu_workers=True, min_free_gb=5))
    worker._validate_job(config,queued_command=command,requested_gpu="0")
    with pytest.raises(ConfigurationError,match="argv"):
        worker._validate_job(config,queued_command=["unapproved"],requested_gpu="0")
    with pytest.raises(ConfigurationError,match="GPU"):
        worker._validate_job(config,queued_command=command,requested_gpu="1")
    for field in ("dataset_fingerprint","split_fingerprint","preprocessing_version"):
        changed = deepcopy(config)
        changed["dataset"][field] = "different"
        with pytest.raises(ConfigurationError):
            module.validate_registered_inputs(db,paths,changed)
    changed = deepcopy(config)
    changed["dataset"]["prepared_artifact_reference"] = str(paths.data_root/"absent")
    with pytest.raises(module.MissingDatasetError):
        module.validate_registered_inputs(db,paths,changed)


def make_archive(source, registry, *, include_checkpoint=True, include_predictions=True):
    db,paths = registry
    config = configuration(source)
    sid = scientific_id(config)
    run_id = create_run_id(seed=0,fold=0,attempt=1,scientific_id_value=sid)
    archive = RunArchive.create(run_id,paths=paths,resolved_config=config,
                               manifest={"lifecycle_status_source":"registry_and_completion_marker"})
    metric = contract.PRIMARY_METRIC
    archive.write_summary({"status":"completed", "primary_metric_name":metric,"primary_metric_value":.25,
                           "best_epoch":2,"final_epoch":2})
    archive.append_metric_event({"name":metric,"value":.25,"step":2})
    archive.write_json("metrics/final.json",{metric:.25})
    archive.write_table("metrics/history",[{"epoch":2,metric:.25}])
    archive.prepare_log_files()
    for name in ("git","hardware","data_fingerprints","split_fingerprint"):
        archive.write_json("provenance/"+name+".json",{"synthetic":True})
    archive.write_text("provenance/uncommitted_changes.patch","")
    archive.write_text("provenance/environment.txt","synthetic CPU fixture\n")
    archive.write_text("provenance/command.txt",'{"argv":["fixture"],"cwd":"/synthetic"}\n')
    if include_checkpoint:
        archive.write_bytes("checkpoints/best.ckpt",b"synthetic checkpoint for checksum lifecycle only")
    if include_predictions:
        archive.write_predictions("validation",[{"run_id":run_id,"sample_key":"sk_0123456789abcdef",
            "graph_id":"synthetic","dataset_id":"synthetic","split":"validation","fold":0,
            "y_true":[0.,1.],"y_pred":[.1,.8],"sample_loss":.25,"node_count":2,
            "edge_count":2,"effective_neighbor_count":1.,"effective_mask_rate":.5}])
    return archive,config


def test_new_protocol_publishes_predictive_bundle_and_indexes_best(source, registry):
    db,paths = registry
    archive,config = make_archive(source,registry)
    final = archive.finalize_success()
    assert verify_run_bundle(final)["status"] == "success"
    sid = scientific_id(config)
    db.register_variant(sid,campaign_id=contract.CAMPAIGN,configuration=config)
    db.create_run(archive.run_id,campaign_id=contract.CAMPAIGN,scientific_id=sid,repro_id="synthetic",
        seed=0,fold=0,attempt=1,configuration=config,status="completed",artifact_path=final,
        primary_metric_name=contract.PRIMARY_METRIC,primary_metric_value=.25)
    checkpoint = final/"checkpoints/best.ckpt"
    db.record_artifacts(archive.run_id,[{"kind":"checkpoint_best","path":checkpoint,
        "sha256":sha256_file(checkpoint),"size_bytes":checkpoint.stat().st_size}])
    indexed = index_checkpoint_catalog(db,paths,run_reference=archive.run_id,verify=True)
    assert indexed["indexed_checkpoints"] == 1
    row, = db.list_checkpoint_catalog(run_id=archive.run_id,role="best",verification_status="verified")
    assert row["sha256"] == sha256_file(checkpoint)


@pytest.mark.parametrize("omitted", ["checkpoint","predictions"])
def test_new_protocol_cannot_finalize_without_predictive_artifacts(source,registry,omitted):
    archive,_ = make_archive(source,registry,include_checkpoint=omitted!="checkpoint",
                             include_predictions=omitted!="predictions")
    with pytest.raises(RunValidationError):
        archive.finalize_success()


@pytest.fixture
def smoke_gate(source,tmp_path,monkeypatch):
    module = launcher()
    monkeypatch.setattr(module,"SOURCE_CONFIG_SHA",contract.SOURCE_CONFIG_SHA)
    root = tmp_path/"smoke"
    for name in ("diagnostics","provenance","checkpoints"):
        (root/name).mkdir(parents=True)
    (root/"config.resolved.yaml").write_text(yaml.safe_dump(configuration(source)))
    (root/"_SUCCESS").write_text("synthetic marker")
    (root/"checkpoints/best.ckpt").write_bytes(b"synthetic checkpoint")
    (root/"provenance/source_files.json").write_text('{"files":[]}')
    (root/"summary.json").write_text(json.dumps({"final_epoch":2,"checkpoint_reload_verified":True}))
    baseline = {"schema":"so2_train_fitted_gene_prevalence_validation_v1",
                "fit_scope":"training_cores_only","validation_fitted":False}
    (root/"diagnostics/train_fitted_gene_prevalence_validation.json").write_text(json.dumps(baseline))
    (root/"diagnostics/optimization_audit_resource_samples.jsonl").write_text(json.dumps({
        "physical_memory_free_bytes":11*module.GIB,"cgroup":{"memory.current":40*module.GIB}})+"\n")
    receipt = {"schema":contract.PREFLIGHT_SCHEMA,"run_id":"r_smoke","passed":True,
        "coverage_verified":True,"finite_gradients":True,"completed_epochs":2,
        "source_configuration_sha256":contract.SOURCE_CONFIG_SHA,"frozen_task_contract_sha256":"e"*64,
        "optimizer_batch_mode":"target_window","prevalence_reference_sha256":canonical_sha256(baseline),
        "resource_gates":{"resource_gate_passed":True,"peak_allocated_bytes":10*module.GIB,
            "peak_reserved_bytes":12*module.GIB,"total_device_bytes":24*module.GIB,
            "headroom_bytes":11*module.GIB,"minimum_sampled_physical_free_bytes":11*module.GIB,
            "maximum_sampled_cgroup_memory_bytes":40*module.GIB}}
    (root/contract.PREFLIGHT_RELATIVE_PATH).write_text(json.dumps(receipt))
    db = SimpleNamespace(list_checkpoint_catalog=lambda **_: [{"sha256":sha256_file(root/"checkpoints/best.ckpt")}])
    monkeypatch.setattr(module,"verified_bundle",lambda *_:root)
    monkeypatch.setattr(module,"source_files",lambda _:{"files":[]})
    return module,root,db


def test_mode_matched_completed_smoke_gate_accepts_verified_evidence(smoke_gate):
    module,root,db = smoke_gate
    reference = module.verify_smoke(db,object(),"r_smoke",mode="target_window",contract_sha="e"*64)
    assert reference["checkpoint_sha256"] == sha256_file(root/"checkpoints/best.ckpt")
    assert reference["optimizer_batch_mode"] == "target_window"


@pytest.mark.parametrize("failure",["code","mode","contract","coverage","vram","cgroup","baseline","catalog"])
def test_smoke_gate_rejects_invalid_evidence(smoke_gate,failure):
    module,root,db = smoke_gate
    path = root/contract.PREFLIGHT_RELATIVE_PATH
    receipt = json.loads(path.read_text())
    if failure == "code":
        (root/"provenance/source_files.json").write_text('{"files":["changed"]}')
    elif failure == "coverage":
        receipt["coverage_verified"] = False
    elif failure == "vram":
        receipt["resource_gates"]["headroom_bytes"] = module.GIB
    elif failure == "cgroup":
        receipt["resource_gates"]["maximum_sampled_cgroup_memory_bytes"] = 161*module.GIB
    elif failure == "baseline":
        receipt["prevalence_reference_sha256"] = "0"*64
    elif failure == "catalog":
        db.list_checkpoint_catalog = lambda **_:[]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ConfigurationError):
        module.verify_smoke(db,object(),"r_smoke",mode="core_pair" if failure=="mode" else "target_window",
                           contract_sha="f"*64 if failure=="contract" else "e"*64)
