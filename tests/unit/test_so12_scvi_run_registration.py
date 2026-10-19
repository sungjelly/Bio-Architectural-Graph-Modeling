from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis import register_so12_scvi_run as module
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import verify_run_bundle


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})
    output = paths.scratch_root / "active_runs" / module.EVALUATION
    source = {"source_run_id": module.PARENT, "checkpoint_epoch": 194,
              "dataset_fingerprint": "a" * 64, "split_fingerprint": "b" * 64,
              "preprocessing_fingerprint": "c" * 64, "fitted_parameters_sha256": "d" * 64,
              "config": {"dataset": {"dataset_id": "so12_paper_core_split", "version": "v1",
                                     "split_id": "so12_paper_21train_5val_2test_v1"}}}
    plan = {"training_cells": 264491, "validation_cells": 61561, "test_cells_in_fit": 0,
            "training_core_numbers": [1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26],
            "validation_core_numbers": [2, 5, 13, 27, 28], "batch_size": 1024,
            "learning_rate": .001, "max_epochs": 100}
    put(output / "provenance/source_verified.json", source)
    put(output / "baselines/scvi_training_plan.json", plan)
    code = tmp_path / "scripts/analysis/baseline.py"
    code.parent.mkdir(parents=True)
    code.write_text("# frozen baseline\n")
    put(output / "provenance/scvi_source_binding.json", {"scripts/analysis/baseline.py": module.sha256_file(code)})
    (output / "provenance/task_contract_at_start.md").write_text("Exploratory baseline, no test fit.\n")
    stage = {"status": "running", "time": "2026-10-18T05:20:52+00:00"}
    put(output / "stages/scvi.json", stage)
    (output / "stages/scvi.jsonl").write_text(json.dumps(stage) + "\n")
    actual_read = module.read
    monkeypatch.setattr(module, "read", lambda path: {"scvi": "1.5.1"}
                        if str(path) == "/venv/so12-scvi/dependency_verification.json" else actual_read(path))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "testgit\n")
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    registry.create_campaign(module.CAMPAIGN, name="test", scientific_question="test", config={})
    # The comparison evaluation belongs to the separate BAGM parent.
    registry.register_variant("sci_parent", campaign_id=module.CAMPAIGN, configuration={"model": {"family": "relative-qkv"}})
    registry.create_run(module.PARENT, campaign_id=module.CAMPAIGN, scientific_id="sci_parent",
                        repro_id="rep_parent", seed=0, fold=0, attempt=1,
                        configuration={"model": {"family": "relative-qkv"}}, status="completed")
    registry.create_evaluation(module.EVALUATION, run_id=module.PARENT, checkpoint_name="best",
                               dataset_id="so12_paper_core_split", split_id="test", status="running")
    return output, paths, registry


def complete_fake_fit(output):
    weights = output / "baselines/scvi_model/model.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic scVI selected state")
    digest = module.sha256_file(weights)
    put(output / "stages/scvi.json", {"status": "complete"})
    put(output / "scvi_manifest.json", {"status": "complete", "passed": True,
        "representation_name": "scvi", "n_cells": 81607, "test_used_for_fitting": False,
        "model_sha256": digest, "created_at": "2026-10-18T05:30:52+00:00", "elapsed_seconds": 600.})
    put(output / "baselines/scvi_frozen_before_test.json", {"model_sha256": digest,
        "state_sha256": "e" * 64, "selected_epoch_zero_based": 1, "validation_elbo": 375.})
    (output / "baselines/scvi_training.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"epoch": 0, "elbo_validation": 400.}, {"epoch": 1, "elbo_validation": 375.},
        {"epoch": 2, "elbo_validation": 376.}]))
    (output / "logs").mkdir()
    (output / "logs/scvi.log").write_text("scVI complete\n")


def test_registration_is_separate_scvi_and_idempotent(prepared):
    output, paths, registry = prepared
    first = module.initialize(output, paths, registry)
    second = module.initialize(output, paths, registry)
    assert first[0].run_id == second[0].run_id != module.PARENT
    run = registry.get_run(first[0].run_id)
    assert run["model_family"] == "scvi"
    assert run["embedding_dim"] == 50
    assert run["status"] == "running"
    assert first[1]["dataset"]["input_scale"] == "nonnegative_integer_raw_counts"
    assert first[1]["metadata"]["sealed_test_claim"] is False
    assert registry.get_run(module.PARENT)["model_family"] == "relative-qkv"


def test_cannot_finalize_incomplete_fit(prepared, monkeypatch):
    output, paths, registry = prepared
    archive, config, receipt = module.initialize(output, paths, registry)
    monkeypatch.setattr(module, "validation_predictions", lambda *a: pytest.fail("Premature model/data access"))
    with pytest.raises(ValueError, match="has not completed"):
        module.finalize(output, paths, registry, archive, config, receipt)


def test_native_bundle_and_catalog_verify_without_parent_changes(prepared, monkeypatch):
    output, paths, registry = prepared
    archive, config, receipt = module.initialize(output, paths, registry)
    complete_fake_fit(output)
    parent_before = registry.get_run(module.PARENT)
    def predictions(*args):
        assert args[3] == archive.run_id
        return [{"run_id": archive.run_id, "sample_key": "validation_aggregate_0",
                 "dataset_id": "so12_paper_core_split", "split": "validation",
                 "y_true": [1., 2.], "y_pred": [1.1, 1.9]}], {"test_cells_loaded": 0}
    monkeypatch.setattr(module, "validation_predictions", predictions)
    result = module.finalize(output, paths, registry, archive, config, receipt)
    assert result["status"] == "complete"
    assert result["catalog_model_name"] == "scvi"
    assert result["catalog_verified"] is True
    assert result["validation_negative_elbo"] == 375.
    assert result["selected_epoch_zero_based"] == 1
    assert verify_run_bundle(archive.artifact_path)["valid"] is True
    assert registry.get_run(module.PARENT) == parent_before
    assert registry.get_run(archive.run_id)["status"] == "completed"
    # Recovery/repeated registration cannot duplicate checkpoints or rerun inference.
    monkeypatch.setattr(module, "validation_predictions", lambda *a: pytest.fail("Duplicate diagnostic"))
    archive2, config2, receipt2 = module.initialize(output, paths, registry)
    again = module.finalize(output, paths, registry, archive2, config2, receipt2)
    assert again["checkpoint_artifact_id"] == result["checkpoint_artifact_id"]


def test_selected_model_hash_and_best_epoch_are_enforced(prepared, monkeypatch):
    output, paths, registry = prepared
    archive, config, receipt = module.initialize(output, paths, registry)
    complete_fake_fit(output)
    monkeypatch.setattr(module, "validation_predictions", lambda *a: pytest.fail("Invalid model accepted"))
    weights = output / "baselines/scvi_model/model.pt"
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash differs"):
        module.finalize(output, paths, registry, archive, config, receipt)


def test_actual_scvi_cpu_reload_and_nb_means_api(tmp_path):
    scvi = pytest.importorskip("scvi")
    import anndata
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(1)
    scvi.settings.seed = 0
    adata = anndata.AnnData(X=np.arange(24, dtype=np.float32).reshape(8, 3) % 4,
                           obs=pd.DataFrame({"slide": pd.Categorical(["1", "2"] * 4)}))
    scvi.model.SCVI.setup_anndata(adata, batch_key="slide")
    model = scvi.model.SCVI(adata, n_latent=2, n_hidden=4, gene_likelihood="nb", use_observed_lib_size=True)
    model.is_trained_ = True  # This test verifies serialization/inference, not fit quality.
    model.module.eval()
    model.save(tmp_path / "saved", save_anndata=False)
    restored = scvi.model.SCVI.load(tmp_path / "saved", adata=adata, accelerator="cpu", device="auto")
    restored.module.eval()
    parameters = restored.get_likelihood_parameters(adata, indices=np.arange(3),
                                                     n_samples=1, give_mean=True, batch_size=2)
    assert parameters["mean"].shape == (3, 3)
    assert np.isfinite(parameters["mean"]).all()
    assert (parameters["mean"] >= 0).all()
    assert all(torch.equal(tensor, restored.module.state_dict()[key])
               for key, tensor in model.module.state_dict().items())
