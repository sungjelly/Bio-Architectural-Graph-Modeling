"""Synthetic h0/reference identity and downstream schema checks, no real cells."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from spatial_benchmark.paths import current_paths

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/analysis"))
import so12_paper_test_h0_extraction as extraction


def small_model_core():
    rng = np.random.default_rng(18)
    n, genes = 7, 8
    config = {"model": {"hidden_dim": 16, "attention_heads": 2, "attention_head_dim": 8,
        "graph_layers": 4, "ffn_dim": 32, "decoder_dim": 32, "dropout": 0.,
        "receiver_chunk_size": 3, "max_edges_per_chunk": 16}}
    core = SimpleNamespace(raw_counts=rng.poisson(2, (n, genes)).astype(np.int32),
        node_covariates=rng.normal(size=(n, 3)).astype(np.float32),
        edge_index=np.stack([np.arange(n), np.roll(np.arange(n), 1)]),
        relative_geometry=rng.normal(size=(n, 70)).astype(np.float32),
        target_expression=object())
    model = extraction.base.build_model(config, genes=genes, covariates=3).float().eval()
    norm = {"gene_p": torch.full((genes,), 1 / genes),
            "residual_mean": torch.zeros(genes), "residual_sd": torch.ones(genes)}
    return model, core, norm, config


def test_h0_direct_encoder_and_published_hL_are_exact_with_unchanged_predictions():
    torch.set_num_threads(1)
    torch.manual_seed(10)
    model, core, norm, _ = small_model_core()
    reference, _ = extraction.base.extract_core_array(model, core, norm, torch.device("cpu"))
    masks = []
    handle = model.encoder.register_forward_pre_hook(
        lambda module, args, kwargs: masks.append(kwargs["gene_mask"].clone()), with_kwargs=True)
    array, audit = extraction.extract_core_array(model, core, norm, torch.device("cpu"), reference)
    handle.remove()
    assert array.shape == reference.shape == (7, 16)
    assert array.dtype == np.float32 and np.isfinite(array).all()
    assert np.all(np.linalg.norm(array, axis=1) > 0)
    assert not np.array_equal(array, reference)
    assert len(masks) == 3 and all(not mask.any() for mask in masks)
    assert all(audit[key] is True for key in
               ("h0_equals_direct_encoder", "published_hL_exact", "predictions_exact", "direct_fourth_block_exact"))
    assert audit["representation"] == extraction.REPRESENTATION
    assert audit["maximum_absolute_reference_hL_difference"] == 0
    assert not model.encoder._forward_hooks


def test_hL_reference_mismatch_is_rejected_without_leaving_model_hooks():
    model, core, norm, _ = small_model_core()
    reference, _ = extraction.base.extract_core_array(model, core, norm, torch.device("cpu"))
    reference[0, 0] = np.nextafter(reference[0, 0], np.float32(np.inf))
    with pytest.raises(ValueError, match="published hL exactly"):
        extraction.extract_core_array(model, core, norm, torch.device("cpu"), reference)
    assert not model.encoder._forward_hooks
    assert "_chunked_layer" not in model.__dict__


def test_training_source_epoch_is_locked(monkeypatch):
    monkeypatch.setattr(extraction.base, "verify_source", lambda *args: {"checkpoint_epoch": 193})
    with pytest.raises(ValueError, match="epoch changed"):
        extraction.verify_source(Path("unused"))


@pytest.fixture
def reference_fixture(tmp_path, monkeypatch):
    base = extraction.base
    paths = current_paths({"BAGM_ROOT": str(tmp_path)})
    monkeypatch.setattr(base, "CORE_CELLS", {15: 2, 23: 2})
    monkeypatch.setattr(base, "N_CELLS", 4)
    monkeypatch.setattr(base, "DIMENSIONS", 3)
    reference = paths.report_root / "analyses/so12_paper_test_leiden" / extraction.REFERENCE_EVALUATION_ID
    (reference / "embeddings").mkdir(parents=True)
    row = {"status": "completed", "run_id": extraction.SOURCE_RUN_ID, "artifact_path": str(reference)}
    monkeypatch.setattr(extraction, "_reference_evaluation", lambda paths: row)
    cells = pd.DataFrame([dict(row_index=i, slide=2, fov=core, cell_ID=i % 2 + 1,
        core_number=core, alias=f"SO2-C{core}", x_um=float(i) + .120281000000003,
        y_um=float(core) + np.pi / 10, split="test")
        for i, core in enumerate((15, 15, 23, 23))], columns=base.CELL_COLUMNS)
    cells.to_csv(reference / "cells.csv.gz", index=False, compression="gzip")
    np.save(reference / "embeddings/hL.npy", np.arange(12, dtype=np.float32).reshape(4, 3) + 1)
    code = paths.project_root / "fake_reference.py"
    code.write_text("# frozen synthetic source\n")
    config = {"analysis_source_sha256": {code.name: base.file_hash(code)}}
    base._write(reference / "config.resolved.json", config)
    base._write(reference / "manifest.json", {"evaluation_id": extraction.REFERENCE_EVALUATION_ID})
    calls = []
    def verify(root, saved):
        calls.append((root, saved))
        return {"evaluation_id": extraction.REFERENCE_EVALUATION_ID}
    monkeypatch.setattr(extraction.hl_controller, "verify_published", verify)
    source = {name: "f" * 64 for name in ("checkpoint_sha256", "preprocessing_fingerprint",
        "fitted_parameters_sha256", "source_identity_sha256", "metadata_statistics_sha256",
        "source_manifest_sha256", "dataset_fingerprint", "split_fingerprint")}
    source.update(checkpoint_epoch=194, binding={"source_sha256": {"model.py": "m" * 64}})
    base._write(reference / "source_verified.json", source)
    manifest = {"source_run_id": extraction.SOURCE_RUN_ID, "representation": "fourth_graph_block_output_before_decoder",
        "hL_path": "embeddings/hL.npy", "status": "complete", "passed": True, "verified": True,
        "all_observed": True, "test_only": True, "test_core_numbers": [15, 23], "n_cells": 4,
        "dimensions": 3, "source_code_sha256": source["binding"]["source_sha256"],
        "cells_sha256": base.file_hash(reference / "cells.csv.gz"), "cell_rows_sha256": base._row_digest(cells),
        "embedding_sha256": base.file_hash(reference / "embeddings/hL.npy"),
        **{key: source[key] for key in ("checkpoint_sha256", "preprocessing_fingerprint",
           "fitted_parameters_sha256", "source_identity_sha256")}}
    base._write(reference / "extraction_manifest.json", manifest)
    return paths, reference, row, source, manifest, calls, config


def test_reference_uses_saved_config_and_exact_frozen_source(reference_fixture):
    paths, reference, _, source, _, calls, config = reference_fixture
    info, cells, values = extraction.verify_reference(reference, source, paths)
    assert calls == [(reference, config)]
    assert info["evaluation_id"] == extraction.REFERENCE_EVALUATION_ID
    assert info["path"] == str(reference) and info["verified"]
    assert isinstance(values, np.memmap) and values.shape == (4, 3)
    assert cells.core_number.tolist() == [15, 15, 23, 23]


@pytest.mark.parametrize("mutation", ["status", "checkpoint", "parameters", "cells", "source_code"])
def test_reference_changes_fail_before_raw_test_loader(reference_fixture, monkeypatch, mutation):
    paths, reference, row, source, manifest, _, _ = reference_fixture
    monkeypatch.setattr(extraction.base, "load_so12_paper_data", lambda *a, **k: pytest.fail("Raw test loader called"))
    if mutation == "status":
        row["status"] = "running"
    elif mutation == "checkpoint":
        manifest["checkpoint_sha256"] = "different"
    elif mutation == "parameters":
        source["fitted_parameters_sha256"] = "different"
    elif mutation == "cells":
        with (reference / "cells.csv.gz").open("ab") as stream:
            stream.write(b"corrupt")
    elif mutation == "source_code":
        (paths.project_root / "fake_reference.py").write_text("# changed\n")
    extraction.base._write(reference / "extraction_manifest.json", manifest)
    with pytest.raises(ValueError):
        extraction.verify_reference(reference, source, paths)


def test_complete_synthetic_h0_export_is_consumable_by_h0_clustering(tmp_path, monkeypatch):
    import so12_paper_test_h0_clustering as clustering
    base = extraction.base
    paths = current_paths({"BAGM_ROOT": str(tmp_path)})
    model, core, norm, config = small_model_core()
    n = len(core.raw_counts)
    monkeypatch.setattr(base, "CORE_CELLS", {15: n, 23: n})
    monkeypatch.setattr(base, "N_CELLS", 2 * n)
    monkeypatch.setattr(base, "DIMENSIONS", 16)
    monkeypatch.setattr(clustering, "EXPECTED_CORE_COUNTS", {15: n, 23: n})
    monkeypatch.setattr(clustering, "N_DIMENSIONS", 16)
    monkeypatch.setattr(extraction.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    run = paths.artifact_root / "runs/2026/10" / extraction.SOURCE_RUN_ID
    (run / "checkpoints").mkdir(parents=True)
    torch.save({"model_state_dict": model.state_dict()}, run / "checkpoints/best.ckpt")
    source = {name: "f" * 64 for name in ("dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint",
        "fitted_parameters_sha256", "source_manifest_sha256", "source_identity_sha256")}
    source.update(config=config, source_run_dir=str(run), checkpoint_epoch=194,
        checkpoint_relative_path="checkpoints/best.ckpt", checkpoint_sha256=base.file_hash(run / "checkpoints/best.ckpt"),
        binding={"source_sha256": {"synthetic.py": "s" * 64}})
    monkeypatch.setattr(extraction, "verify_source", lambda *args: source)
    monkeypatch.setattr(base, "training_source_binding", lambda: source["binding"]["source_sha256"])
    rows, cores = [], []
    for number in (15, 23):
        coordinates = np.column_stack([np.arange(n, dtype=float), np.full(n, number, dtype=float)])
        cores.append(SimpleNamespace(**vars(core), alias=f"SO2-C{number}", slide=2, split="test", n_cells=n,
                                     coordinates_um=coordinates))
        rows.extend(dict(slide=2, fov=number, cell_ID=i + 1, core_number=number, core=f"SO2-C{number}",
                         x_um=coordinates[i, 0], y_um=coordinates[i, 1]) for i in range(n))
    observations = paths.data_root / "processed/so12_pearson_integration_v1/observations.csv.gz"
    observations.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(observations, index=False, compression="gzip")
    monkeypatch.setattr(base, "UPSTREAM_OBSERVATIONS_SHA256", base.file_hash(observations))
    dataset = SimpleNamespace(cores=cores, genes=tuple(f"g{i}" for i in range(8)),
        gene_p=norm["gene_p"].numpy(), residual_mean=norm["residual_mean"].numpy(), residual_sd=norm["residual_sd"].numpy(),
        provenance={key: source[key] for key in ("dataset_fingerprint", "split_fingerprint",
            "preprocessing_fingerprint", "fitted_parameters_sha256", "source_manifest_sha256")})
    reference_dir = paths.report_root / "analyses/so12_paper_test_leiden" / extraction.REFERENCE_EVALUATION_ID
    (reference_dir / "embeddings").mkdir(parents=True)
    reference_cells = base.prepare_cells(dataset, reference_dir, paths)
    reference_hL, _ = base.extract_core_array(model, core, norm, torch.device("cpu"))
    reference_hL = np.concatenate([reference_hL, reference_hL])
    np.save(reference_dir / "embeddings/hL.npy", reference_hL)
    base._write(reference_dir / "manifest.json", {"synthetic": True})
    reference = {"evaluation_id": extraction.REFERENCE_EVALUATION_ID, "path": str(reference_dir),
        "manifest_sha256": base.file_hash(reference_dir / "manifest.json"),
        "cells_sha256": base.file_hash(reference_dir / "cells.csv.gz"),
        "cell_rows_sha256": base._row_digest(reference_cells),
        "embedding_sha256": base.file_hash(reference_dir / "embeddings/hL.npy")}
    monkeypatch.setattr(extraction, "verify_reference", lambda *args: (reference, reference_cells, reference_hL))
    calls = []
    def load(*args, roles, **kwargs):
        calls.append(roles)
        return dataset
    monkeypatch.setattr(base, "load_so12_paper_data", load)
    out = paths.scratch_root / "active_runs/synthetic_h0"
    result = extraction.extract(run, out, device="cpu", paths=paths, reference_hl_dir=reference_dir)
    cells, array, bindings, manifest = clustering.load_inputs(out)
    assert calls == [("test",)]
    assert result["representation_name"] == "h0" and result["representation"] == extraction.REPRESENTATION
    assert "hL_path" not in result and "hL_sha256" not in result
    assert not (out / "embeddings/hL.npy").exists()
    assert array.shape == (2 * n, 16) and len(cells) == 2 * n
    assert result["audit"] == {"h0_equals_direct_encoder": True, "published_hL_exact": True, "predictions_exact": True}
    assert all(item["receipt_path"] in bindings["files"] for item in manifest["cores"])
    assert extraction.base._read(out / "diagnostics/extraction_pilot.json")["core_number"] == 23
