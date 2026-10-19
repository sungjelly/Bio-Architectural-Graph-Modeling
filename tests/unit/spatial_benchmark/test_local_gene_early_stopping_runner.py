"""CPU-only control/checkpoint/admission checks for the prospective runner."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts.train import run_local_gene_early_stopping as runner
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256, scientific_id
from spatial_benchmark.local_gene_stopping import LocalGeneEarlyStopping


def config(*, profile="full", variant="local", seed=0):
    return runner.build_config(SimpleNamespace(profile=profile, variant=variant, seed=seed,
        attempt=1, gpu=str(seed), smoke_run="pilot" if profile == "full" else None))


def test_frozen_schedule_and_fresh_paired_variants():
    full = config()
    trainer = full["trainer"]
    assert trainer["epochs"] == trainer["maximum_epochs"] == 500
    assert trainer["minimum_epochs"] == 50
    assert trainer["early_stopping"] == {"patience": 30, "min_delta": .0001, "min_epochs": 50, "max_epochs": 500}
    assert trainer["minimum_learning_rate"] == 1e-5
    assert trainer["scheduler"] == "reduce_on_plateau"
    assert trainer["scheduler_factor"] == .5 and trainer["scheduler_patience"] == 7
    assert trainer["scheduler_threshold"] == .0001 and trainer["scheduler_threshold_mode"] == "abs"
    assert trainer["retained_checkpoint_roles"] == ["best", "last"]
    assert trainer["initialization"] == "fresh_seed_matched_not_resumed"
    assert full["classification"]["lifecycle_stage"] == "exploratory_validation"
    assert full["classification"]["study_axis"] == "early_stopping"
    assert scientific_id(full) == scientific_id(config(seed=1))
    assert scientific_id(full) != scientific_id(config(variant="self"))
    smoke = config(profile="smoke")
    assert smoke["trainer"]["epochs"] == 2
    assert smoke["trainer"]["early_stopping"] == trainer["early_stopping"]


def test_scheduler_uses_absolute_validation_plateau_and_minimum_lr():
    model = torch.nn.Linear(1, 1)
    optimizer, scheduler = runner.make_optimizer_scheduler(model, config())
    for _ in range(8):
        scheduler.step(.3)
    assert optimizer.param_groups[0]["lr"] == .001
    scheduler.step(.3)
    assert optimizer.param_groups[0]["lr"] == .0005
    for _ in range(200):
        scheduler.step(.3)
    assert optimizer.param_groups[0]["lr"] == 1e-5


def test_best_and_last_preserve_optimizer_scheduler_policy_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    model = torch.nn.Linear(1, 1)
    resolved = config()
    optimizer, scheduler = runner.make_optimizer_scheduler(model, resolved)
    policy = LocalGeneEarlyStopping(patience=2, min_delta=1e-4, min_epochs=1, max_epochs=10)
    archive = SimpleNamespace(run_id="synthetic", scratch_path=tmp_path)
    values = [.3, .3002, .3003]
    for epoch, nll in enumerate(values, 1):
        optimizer.zero_grad()
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        decision = runner.commit_validation_epoch(archive=archive, config=resolved, epoch=epoch,
            model=model, optimizer=optimizer, scheduler=scheduler, policy=policy,
            metrics={"count_nll": nll}, hashes={"runner": "fixed"})
    assert decision.should_stop and decision.stop_reason == "patience"
    best = torch.load(tmp_path / "checkpoints/best.ckpt", weights_only=False)
    last = torch.load(tmp_path / "checkpoints/last.ckpt", weights_only=False)
    assert best["epoch"] == 1 and last["epoch"] == 3
    for checkpoint in (best, last):
        assert set(("model", "optimizer", "scheduler", "early_stopping", "torch_rng", "cuda_rng", "numpy_rng", "python_rng")).issubset(checkpoint)
        assert checkpoint["optimizer"]["state"]
        assert checkpoint["source_hashes"] == {"runner": "fixed"}
        restored = LocalGeneEarlyStopping.from_state_dict(checkpoint["early_stopping"])
        assert restored.completed_epoch == checkpoint["epoch"]
        assert checkpoint["scheduler"]["last_epoch"] == checkpoint["epoch"]


def test_small_strict_improvements_select_best_without_resetting_anchor(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    model = torch.nn.Linear(1, 1)
    resolved = config()
    optimizer, scheduler = runner.make_optimizer_scheduler(model, resolved)
    policy = LocalGeneEarlyStopping(patience=2, min_delta=1e-4, min_epochs=1, max_epochs=10)
    archive = SimpleNamespace(run_id="synthetic", scratch_path=tmp_path)
    for epoch, nll in enumerate([.3, .29998, .29997], 1):
        decision = runner.commit_validation_epoch(archive=archive, config=resolved, epoch=epoch,
            model=model, optimizer=optimizer, scheduler=scheduler, policy=policy,
            metrics={"count_nll": nll}, hashes={})
    assert decision.improved_best and not decision.improved_anchor
    assert decision.should_stop and decision.anchor_epoch == 1
    assert torch.load(tmp_path / "checkpoints/best.ckpt", weights_only=False)["epoch"] == 3


def test_smoke_and_safety_cap_never_claim_patience_stopping():
    policy = LocalGeneEarlyStopping()
    for epoch in (1, 2):
        policy.observe(epoch=epoch, count_nll=.3)
    smoke = runner.stopping_receipt(policy, "smoke")
    assert smoke["stop_reason"] == "diagnostic_budget"
    assert not smoke["early_stopping_triggered"] and not smoke["safety_cap_reached"]
    with pytest.raises(RuntimeError, match="without its declared stopping condition"):
        runner.stopping_receipt(policy, "full")
    capped = LocalGeneEarlyStopping(patience=30, min_epochs=2, max_epochs=3)
    for epoch in (1, 2, 3):
        capped.observe(epoch=epoch, count_nll=1. / epoch)
    outcome = runner.stopping_receipt(capped, "full")
    assert outcome["stop_reason"] == "maximum_epochs"
    assert outcome["safety_cap_reached"] and not outcome["early_stopping_triggered"]


def test_replay_gate_rejects_schema_drift_and_metric_mismatch():
    assert runner.replay_delta({"nll": .3}, {"nll": .3}) == 0
    with pytest.raises(RuntimeError, match="schema"):
        runner.replay_delta({"nll": .3}, {"loss": .3})
    with pytest.raises(RuntimeError, match="mismatch"):
        runner.replay_delta({"nll": .3}, {"nll": .31})


def dependency_fixture(tmp_path, monkeypatch):
    jobs, runs, dependencies = {}, {}, []
    for method in ("whole_cell", "partial_gene", "spatial_block"):
        for seed in (0, 1, 2):
            run_id, job_id = f"run_{method}_{seed}", f"job_{method}_{seed}"
            root = tmp_path / run_id
            (root / "provenance").mkdir(parents=True)
            value = {"campaign": {"campaign_id": runner.DEPENDENCY_CAMPAIGN},
                "masking": {"method": method}, "seed": seed,
                "metadata": {"execution_role": "production"}, "trainer": {"max_epochs": 200}}
            (root / "config.resolved.yaml").write_text(yaml.safe_dump({**value, "attempt": 1}))
            for name in ("_SUCCESS", "summary.json", "provenance/artifact_checksums.json"):
                (root / name).write_text("{}")
            jobs[job_id] = {"campaign_id": runner.DEPENDENCY_CAMPAIGN, "status": "completed", "run_id": run_id,
                "canonical_config": value}
            runs[run_id] = {"campaign_id": runner.DEPENDENCY_CAMPAIGN, "status": "completed",
                "artifact_path": str(root), "config": {**deepcopy(value), "attempt": 1}}
            dependencies.append({"job_id": job_id, "run_id": run_id, "config_sha256": canonical_sha256(value),
                "artifact_path": str(root), "success_sha256": sha256_file(root / "_SUCCESS"),
                "summary_sha256": sha256_file(root / "summary.json"),
                "checksum_manifest_sha256": sha256_file(root / "provenance/artifact_checksums.json")})
    receipt = {"schema": "local_gene_dependency_receipt_v1", "campaign_id": runner.DEPENDENCY_CAMPAIGN,
        "plan_sha256": "a" * 64, "verified_at": "2026-10-05T00:00:00Z", "dependencies": dependencies}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    active = {"count": 0}
    @contextmanager
    def connect():
        yield SimpleNamespace(execute=lambda *args: SimpleNamespace(fetchone=lambda: [active["count"]]))
    registry = SimpleNamespace(get_job=jobs.get, get_run=runs.get, connect=connect)
    monkeypatch.setattr(runner.RunArchive, "artifact_path_for", lambda run_id, paths: tmp_path / run_id)
    monkeypatch.setattr(runner, "verify_run_bundle", lambda *args: {"ok": True})
    return path, registry, jobs, runs, receipt, active


def test_dependency_receipt_binds_nine_unique_completed_full_slots(tmp_path, monkeypatch):
    path, registry, _, _, _, _ = dependency_fixture(tmp_path, monkeypatch)
    metadata, receipt = runner.verify_dependency_receipt(path, object(), registry)
    assert len(metadata["dependency_run_ids"]) == 9
    assert metadata["sha256"] == sha256_file(path)
    assert receipt["campaign_id"] == runner.DEPENDENCY_CAMPAIGN


@pytest.mark.parametrize("change", ["registry_slot", "published_slot", "unfinished", "duplicated", "active_queue", "checksum"])
def test_dependency_admission_rejects_incomplete_or_wrong_slot_sources(tmp_path, monkeypatch, change):
    path, registry, jobs, runs, receipt, active = dependency_fixture(tmp_path, monkeypatch)
    first = receipt["dependencies"][0]
    if change == "registry_slot":
        runs[first["run_id"]]["config"]["masking"]["method"] = "partial_gene"
    elif change == "published_slot":
        published = tmp_path / first["run_id"] / "config.resolved.yaml"
        value = yaml.safe_load(published.read_text())
        value["seed"] = 1
        published.write_text(yaml.safe_dump(value))
    elif change == "unfinished":
        jobs[first["job_id"]]["status"] = "running"
    elif change == "duplicated":
        receipt["dependencies"][-1] = deepcopy(first)
        path.write_text(json.dumps(receipt))
    elif change == "active_queue":
        active["count"] = 1
    else:
        (tmp_path / first["run_id"] / "summary.json").write_text('{"changed":true}')
    with pytest.raises(RuntimeError):
        runner.verify_dependency_receipt(path, object(), registry)


def test_check_only_never_probes_gpu_and_missing_dependency_blocks_launch(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(runner.DEPENDENCY_RECEIPT_ENV, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(runner, "current_paths", lambda: SimpleNamespace(state_root=tmp_path))
    monkeypatch.setattr(runner, "Registry", lambda *args, **kwargs: object())
    monkeypatch.setattr(runner, "source_hashes", lambda: {})
    def unexpected_gpu(*args):
        raise AssertionError("A CPU-only admission check touched a GPU")
    monkeypatch.setattr(runner, "inspect_gpu", unexpected_gpu)
    assert runner.main(["--profile", "smoke", "--gpu", "0", "--check-only"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["admission"] == {"admitted": False, "cpu_only_check": True,
        "dependency_verified": False, "gpu_probed": False}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="Normal admission requires"):
        runner.main(["--profile", "smoke", "--gpu", "0"])


@pytest.mark.parametrize("change", ["source", "contract", "terminal_replay", "nondiagnostic"])
def test_new_pilot_gate_cannot_reuse_old_or_failed_sources(tmp_path, monkeypatch, change):
    root = tmp_path / "pilot"
    (root / "diagnostics").mkdir(parents=True)
    receipt = {"passed": True, "source_hashes": {"source": "fixed"},
        "frozen_contract_sha256": "contract", "terminal_checkpoint_replay_passed": True}
    summary = {"final_epoch": 2, "stop_reason": "diagnostic_budget", "early_stopping_triggered": False}
    if change == "source": receipt["source_hashes"] = {}
    elif change == "contract": receipt["frozen_contract_sha256"] = "other"
    elif change == "terminal_replay": receipt["terminal_checkpoint_replay_passed"] = False
    else: summary["early_stopping_triggered"] = True
    (root / "diagnostics/preflight.json").write_text(json.dumps(receipt))
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "config.resolved.yaml").write_text(yaml.safe_dump({"metadata": {"profile": "smoke"}}))
    registry = SimpleNamespace(get_run=lambda run_id: {"status": "completed", "campaign_id": runner.CAMPAIGN,
        "artifact_path": str(root)})
    monkeypatch.setattr(runner.RunArchive, "artifact_path_for", lambda *args: root)
    monkeypatch.setattr(runner, "verify_run_bundle", lambda *args: {"ok": True})
    with pytest.raises(RuntimeError):
        runner.verify_pilot(registry, object(), "pilot", {"source": "fixed"},
            {"metadata": {"frozen_contract_sha256": "contract"}})
