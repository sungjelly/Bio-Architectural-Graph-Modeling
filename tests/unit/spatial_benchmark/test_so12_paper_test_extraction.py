"""Synthetic finalization, frozen-source, cell-alignment, and hL semantics checks."""
from copy import deepcopy
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from spatial_benchmark.paths import current_paths

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/analysis"))
import so12_paper_test_extraction as extraction


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    paths = current_paths({"BAGM_ROOT": str(tmp_path)})
    source = paths.artifact_root / "runs/2026/10" / extraction.SOURCE_RUN_ID
    (source / "diagnostics").mkdir(parents=True)
    (source / "checkpoints").mkdir()
    (source / "_SUCCESS").write_text("success")
    registry_row = {"status": "completed", "artifact_path": str(source)}
    monkeypatch.setattr(extraction, "Registry", lambda _: SimpleNamespace(get_run=lambda _: registry_row))
    monkeypatch.setattr(extraction, "verify_run_bundle", lambda _: {"valid": True, "status": "success"})
    code_binding = {"frozen_training.py": "a" * 64}
    monkeypatch.setattr(extraction, "training_source_binding", lambda: code_binding)
    config = {"dataset": {"dataset_id": extraction.DATASET_ID, "version": extraction.DATASET_VERSION,
                          "split_id": extraction.SPLIT_ID},
              "model": {"graph_layers": 4, "hidden_dim": 256}}
    (source / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    cache = paths.data_root / "processed/fake_test_fixture"
    cache.mkdir(parents=True)
    manifest = {"dataset_fingerprint": "d" * 64, "split_fingerprint": "s" * 64,
                "preprocessing_fingerprint": "p" * 64,
                "files": {"fitted_parameters.npz": "f" * 64, "metadata_statistics.json": "m" * 64}}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(extraction, "_read_cache", lambda paths, verify: (cache, manifest))
    provenance = {key: manifest[key] for key in ("dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint")}
    provenance.update(split=extraction.split_contract(), fitted_parameters_sha256="f" * 64,
                      source_manifest_sha256=extraction.file_hash(cache / "manifest.json"))
    dataset = {"provenance": provenance}
    binding = {"source_sha256": code_binding, "config_sha256": extraction.scientific_binding(config),
               "dataset_sha256": extraction.digest(dataset)}
    checkpoint = {"run_id": extraction.SOURCE_RUN_ID, "checkpoint_schema": extraction.CHECKPOINT_SCHEMA,
                  "checkpoint_role": "best", "protocol": extraction.PROTOCOL,
                  "primary_metric_name": extraction.PRIMARY, "primary_metric_value": .1,
                  "epoch": 55, "binding": binding, "model_state_dict": {}}
    torch.save(checkpoint, source / "checkpoints/best.ckpt")
    extraction._write(source / "diagnostics/training_start.json", {"binding": binding, "dataset": dataset})
    selection = {"checkpoint": "checkpoints/best.ckpt", "checkpoint_sha256": extraction.file_hash(source / "checkpoints/best.ckpt"),
                 "best_epoch": 55, "selection_completed": True, "validation_replay_verified": True,
                 "test_used_for_selection": False, "primary_metric_name": extraction.PRIMARY,
                 "primary_metric_value": .1}
    extraction._write(source / "diagnostics/selection_frozen_before_test.json", selection)
    test = {**selection, "test_loaded_after_selection": True, "test_evaluation_count": 1,
            "mask_namespace": extraction.TEST_NAMESPACE, "test_core_aliases": list(extraction.CORE_ALIASES),
            "test_cells": extraction.N_CELLS, "mask_views": 3, "evaluated_core_view_count": 6,
            "core_view_coverage": [{"core_alias": alias, "view": view}
                                   for alias in extraction.CORE_ALIASES for view in range(3)],
            "test_data_provenance": {**provenance, "loaded_roles": ["test"],
                                     "opened_core_aliases": list(extraction.CORE_ALIASES)}}
    extraction._write(source / "diagnostics/test_evaluation.json", test)
    return paths, source, registry_row, manifest, test


def test_source_requires_finalized_exact_run_before_test_data_access(source_fixture, monkeypatch):
    paths, source, registry, _, _ = source_fixture
    monkeypatch.setattr(extraction, "load_so12_paper_data", lambda *a, **k: pytest.fail("Test data opened before gate"))
    registry["status"] = "running"
    with pytest.raises(ValueError, match="not completed"):
        extraction.extract(source, paths.scratch_root / "active_runs/fixture", device="cpu", paths=paths)
    registry["status"] = "completed"
    with pytest.raises(ValueError, match="exact approved"):
        extraction.verify_source(source.with_name("old_run"), paths)
    monkeypatch.setattr(extraction, "verify_run_bundle", lambda _: {"valid": False, "status": "success"})
    with pytest.raises(ValueError, match="bundle verification"):
        extraction.verify_source(source, paths)


def test_valid_source_binds_checkpoint_final_test_and_frozen_parameters(source_fixture):
    paths, source, _, _, _ = source_fixture
    verified = extraction.verify_source(source, paths)
    assert verified["source_run_id"] == extraction.SOURCE_RUN_ID
    assert verified["final_test_completed"] and verified["source_bundle_verified"]
    assert verified["fitted_parameters_sha256"] == "f" * 64
    assert verified["checkpoint_sha256"] == extraction.file_hash(source / "checkpoints/best.ckpt")
    assert verified["source_identity_sha256"] == extraction.digest(
        {k: v for k, v in verified.items() if k != "source_identity_sha256"})


@pytest.mark.parametrize("mutation", ["coverage", "checkpoint", "selection", "parameters", "role"])
def test_final_test_receipt_mutations_are_rejected(source_fixture, mutation):
    paths, source, _, _, receipt = source_fixture
    receipt = deepcopy(receipt)
    if mutation == "coverage":
        receipt["core_view_coverage"][-1] = receipt["core_view_coverage"][0]
    elif mutation == "checkpoint":
        receipt["checkpoint_sha256"] = "changed"
    elif mutation == "selection":
        receipt["test_used_for_selection"] = True
    elif mutation == "parameters":
        receipt["test_data_provenance"]["fitted_parameters_sha256"] = "changed"
    elif mutation == "role":
        receipt["test_data_provenance"]["loaded_roles"] = ["train", "test"]
    extraction._write(source / "diagnostics/test_evaluation.json", receipt)
    with pytest.raises(ValueError):
        extraction.verify_source(source, paths)


def test_changed_training_sources_or_preprocessing_are_rejected(source_fixture, monkeypatch):
    paths, source, _, manifest, _ = source_fixture
    manifest["preprocessing_fingerprint"] = "different"
    with pytest.raises(ValueError, match="preprocessing_fingerprint"):
        extraction.verify_source(source, paths)
    manifest["preprocessing_fingerprint"] = "p" * 64
    monkeypatch.setattr(extraction, "training_source_binding", lambda: {"frozen_training.py": "changed"})
    with pytest.raises(ValueError, match="source code"):
        extraction.verify_source(source, paths)


def _small_model_core():
    rng = np.random.default_rng(9)
    n, genes = 7, 8
    core = SimpleNamespace(raw_counts=rng.poisson(2, size=(n, genes)).astype(np.int32),
                           node_covariates=rng.normal(size=(n, 3)).astype(np.float32),
                           edge_index=np.stack([np.arange(n), np.roll(np.arange(n), 1)]),
                           relative_geometry=rng.normal(size=(n, 70)).astype(np.float32),
                           target_expression=object())
    config = {"model": {"hidden_dim": 16, "attention_heads": 2, "attention_head_dim": 8,
                         "graph_layers": 4, "ffn_dim": 32, "decoder_dim": 32, "dropout": 0.,
                         "receiver_chunk_size": 3, "max_edges_per_chunk": 16}}
    model = extraction.build_model(config, genes=genes, covariates=3).float().eval()
    normalization = {"gene_p": torch.full((genes,), 1 / genes),
                     "residual_mean": torch.zeros(genes), "residual_sd": torch.ones(genes)}
    return model, core, normalization


def test_hL_is_direct_fourth_block_with_unchanged_predictions_and_zero_masks():
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model, core, normalization = _small_model_core()
    masks = []
    handle = model.encoder.register_forward_pre_hook(
        lambda module, args, kwargs: masks.append(kwargs["gene_mask"].clone()), with_kwargs=True)
    array, audit = extraction.extract_core_array(model, core, normalization, torch.device("cpu"))
    handle.remove()
    assert array.shape == (7, 16) and array.dtype == np.float32
    assert np.isfinite(array).all() and np.all(np.linalg.norm(array, axis=1) > 0)
    assert len(masks) == 2 and all(not mask.any() for mask in masks)
    assert audit["direct_fourth_block_exact"] and audit["predictions_exact"] and audit["ordinary_hL_exact"]
    assert audit["maximum_absolute_prediction_difference"] == 0
    assert "_chunked_layer" not in model.__dict__
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        extraction.extract_core_array(model, core, normalization, torch.device("cpu"))


def test_cell_export_excludes_other_cores_and_rejects_coordinate_mismatch(tmp_path, monkeypatch):
    paths = current_paths({"BAGM_ROOT": str(tmp_path)})
    monkeypatch.setattr(extraction, "CORE_CELLS", {15: 2, 23: 2})
    monkeypatch.setattr(extraction, "N_CELLS", 4)
    source = paths.data_root / "processed/so12_pearson_integration_v1/observations.csv.gz"
    source.parent.mkdir(parents=True)
    rows = []
    cores = []
    for number in (1, 15, 23):
        coordinates = np.array([[number, 1.], [number, 2.]])
        for index in range(2):
            rows.append({"slide": 1 if number == 1 else 2, "fov": number, "cell_ID": index + 1,
                         "core_number": number, "core": f"SO{1 if number == 1 else 2}-C{number:02d}",
                         "x_um": coordinates[index, 0], "y_um": coordinates[index, 1]})
        if number != 1:
            cores.append(SimpleNamespace(alias=f"SO2-C{number}", split="test", slide=2, n_cells=2,
                                         coordinates_um=coordinates))
    pd.DataFrame(rows).to_csv(source, index=False, compression="gzip")
    monkeypatch.setattr(extraction, "UPSTREAM_OBSERVATIONS_SHA256", extraction.file_hash(source))
    dataset = SimpleNamespace(cores=cores)
    frame = extraction.prepare_cells(dataset, tmp_path / "output", paths)
    assert set(frame.core_number) == {15, 23}
    assert frame.row_index.tolist() == list(range(4))
    assert frame.split.eq("test").all()
    assert list(frame.columns) == list(extraction.CELL_COLUMNS)
    assert set(pd.read_csv(tmp_path / "output/cells.csv.gz").core_number) == {15, 23}
    cores[0].coordinates_um[0, 0] += 1
    with pytest.raises(ValueError, match="alignment"):
        extraction.prepare_cells(dataset, tmp_path / "other", paths)


def test_synthetic_extraction_manifest_is_consumed_by_clustering(tmp_path, monkeypatch):
    """Exercise the real exporter and downstream input reader without real cells."""
    import so12_paper_test_clustering as clustering
    paths = current_paths({"BAGM_ROOT": str(tmp_path)})
    model, small_core, normalization = _small_model_core()
    n = len(small_core.raw_counts)
    expected_counts = {15: n, 23: n}
    monkeypatch.setattr(extraction, "CORE_CELLS", expected_counts)
    monkeypatch.setattr(extraction, "N_CELLS", 2 * n)
    monkeypatch.setattr(extraction, "DIMENSIONS", 16)
    monkeypatch.setattr(clustering, "EXPECTED_CORE_COUNTS", expected_counts)
    monkeypatch.setattr(clustering, "N_DIMENSIONS", 16)
    monkeypatch.setattr(extraction.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    config = {"model": {"hidden_dim": 16, "attention_heads": 2, "attention_head_dim": 8,
                         "graph_layers": 4, "ffn_dim": 32, "decoder_dim": 32, "dropout": 0.,
                         "receiver_chunk_size": 3, "max_edges_per_chunk": 16}}
    run = paths.artifact_root / "runs/2026/10" / extraction.SOURCE_RUN_ID
    (run / "checkpoints").mkdir(parents=True)
    torch.save({"model_state_dict": model.state_dict()}, run / "checkpoints/best.ckpt")
    binding = {"synthetic_training.py": "b" * 64}
    monkeypatch.setattr(extraction, "training_source_binding", lambda: binding)
    source = {"config": config, "source_run_dir": str(run), "checkpoint_relative_path": "checkpoints/best.ckpt",
              "checkpoint_sha256": extraction.file_hash(run / "checkpoints/best.ckpt"),
              "source_identity_sha256": "i" * 64, "binding": {"source_sha256": binding},
              **{name: "f" * 64 for name in ("dataset_fingerprint", "split_fingerprint",
                 "preprocessing_fingerprint", "fitted_parameters_sha256", "source_manifest_sha256")}}
    monkeypatch.setattr(extraction, "verify_source", lambda run_dir, paths: source)
    rows, cores = [], []
    for number in expected_counts:
        coordinates = np.column_stack([np.arange(n, dtype=float), np.full(n, number, dtype=float)])
        cores.append(SimpleNamespace(**vars(small_core), alias=f"SO2-C{number}", slide=2,
                                     split="test", n_cells=n, coordinates_um=coordinates))
        for index in range(n):
            rows.append({"slide": 2, "fov": number, "cell_ID": index + 1, "core_number": number,
                         "core": f"SO2-C{number}", "x_um": coordinates[index, 0], "y_um": coordinates[index, 1]})
    observations = paths.data_root / "processed/so12_pearson_integration_v1/observations.csv.gz"
    observations.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(observations, index=False, compression="gzip")
    monkeypatch.setattr(extraction, "UPSTREAM_OBSERVATIONS_SHA256", extraction.file_hash(observations))
    dataset = SimpleNamespace(cores=cores, genes=tuple(f"g{i}" for i in range(8)),
        gene_p=normalization["gene_p"].numpy(), residual_mean=normalization["residual_mean"].numpy(),
        residual_sd=normalization["residual_sd"].numpy(),
        provenance={key: source[key] for key in ("dataset_fingerprint", "split_fingerprint",
            "preprocessing_fingerprint", "fitted_parameters_sha256", "source_manifest_sha256")})
    calls = []
    def load(config, paths, *, roles, **kwargs):
        calls.append(roles)
        return dataset
    monkeypatch.setattr(extraction, "load_so12_paper_data", load)
    out = paths.scratch_root / "active_runs/synthetic_test_leiden"
    result = extraction.extract(run, out, device="cpu", paths=paths)
    cells, values, bindings, manifest = clustering.load_inputs(out)
    assert calls == [("test",)]
    assert result["status"] == "complete" and result["passed"]
    assert result["cells_per_core"] == {"15": n, "23": n}
    assert values.shape == (2 * n, 16) and len(cells) == 2 * n
    assert set(cells.core_alias) == {"SO2-C15", "SO2-C23"}
    assert manifest["embedding_sha256"] == extraction.file_hash(out / "embeddings/hL.npy")
    assert len(manifest["cores"]) == 2
    assert all(item["receipt_path"] in bindings["files"] for item in manifest["cores"])
