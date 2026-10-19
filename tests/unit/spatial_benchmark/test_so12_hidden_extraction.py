"""State semantics, source binding, and saved-array integrity for extraction."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/analysis"))
import so12_hidden_extraction as analysis


def _small_core():
    rng = np.random.default_rng(33)
    n, genes = 7, 8
    core = SimpleNamespace(raw_counts=rng.poisson(2., (n, genes)).astype(np.int32),
        node_covariates=rng.normal(size=(n, 3)).astype(np.float32),
        edge_index=np.stack([np.arange(n), np.roll(np.arange(n), 1)]),
        relative_geometry=rng.normal(size=(n, 70)).astype(np.float32))
    normalization = {"gene_p": torch.full((genes,), 1 / genes),
                     "residual_mean": torch.zeros(genes), "residual_sd": torch.ones(genes)}
    config = {"model": {"hidden_dim": 16, "attention_heads": 2, "attention_head_dim": 8,
        "graph_layers": 4, "ffn_dim": 32, "decoder_dim": 32, "dropout": 0.,
        "receiver_chunk_size": 3, "max_edges_per_chunk": 16}}
    model = analysis.build_model(config, genes=genes, covariates=3).eval()
    return model, core, normalization


def test_full_observed_extraction_preserves_state_and_prediction_semantics():
    torch.set_num_threads(1)
    torch.manual_seed(15)
    model, core, norm = _small_core()
    observed_masks = []
    hook = model.encoder.register_forward_pre_hook(
        lambda _module, _args, kwargs: observed_masks.append(kwargs["gene_mask"].clone()),
        with_kwargs=True)
    arrays, audit = analysis.extract_core_arrays(model, core, norm, torch.device("cpu"),
                                                verify_invariance=True)
    hook.remove()
    assert len(observed_masks) == 3 and all(not m.any() for m in observed_masks)
    assert arrays["h0"].shape == arrays["hL"].shape == (7, 16)
    assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in arrays.values())
    np.testing.assert_array_equal(arrays["delta_l2"], np.linalg.norm(
        arrays["hL"].astype(np.float64) - arrays["h0"].astype(np.float64), axis=1).astype(np.float32))
    assert audit["predictions_exact"] and audit["h0_equals_direct_encoder"]
    assert audit["ordinary_hL_exact"] and audit["hL_equals_full_node_embedding"]
    assert not np.array_equal(arrays["h0"], arrays["hL"])
    model.train()
    with pytest.raises(ValueError, match="eval mode"):
        analysis.extract_core_arrays(model, core, norm, torch.device("cpu"))


def test_inference_enables_training_determinism_without_relaxing_precision(monkeypatch):
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.deterministic,
              torch.backends.cudnn.benchmark, torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    try:
        result = analysis.configure_deterministic_inference()
        assert result == {"deterministic_algorithms": True, "cudnn_deterministic": True,
                          "cublas_workspace_config": ":4096:8"}
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.use_deterministic_algorithms(before[0])
        torch.backends.cudnn.deterministic = before[1]
        torch.backends.cudnn.benchmark = before[2]
        torch.backends.cuda.matmul.allow_tf32 = before[3]
        torch.backends.cudnn.allow_tf32 = before[4]
        torch.set_float32_matmul_precision(before[5])


def test_verification_flag_normalization_preserves_scientific_binding():
    raw = {"source_file_checksums_verified": False, "files": [
        {"checksum_verified": False, "sha256": "original", "path": "input"}],
        "training_ready": False, "model": {"width": 256}}
    before = deepcopy(raw)
    verified = analysis._verified_metadata(raw)
    assert raw == before
    assert verified["source_file_checksums_verified"] and verified["files"][0]["checksum_verified"]
    assert verified["training_ready"] is False and verified["model"] == raw["model"]
    changed = deepcopy(raw)
    changed["files"][0]["sha256"] = "different"
    assert analysis.digest(analysis._verified_metadata(changed)) != analysis.digest(verified)


def _write_core(root, record, inputs):
    path = root / "embeddings" / f"core_{record['core_number']}"
    path.mkdir(parents=True)
    h0 = np.arange(record["n_cells"] * inputs["dimensions"], dtype=np.float32).reshape(
        record["n_cells"], inputs["dimensions"])
    hL = h0 + 1
    arrays = {"h0": h0, "hL": hL, "delta_l2": np.linalg.norm(
        hL.astype(np.float64) - h0.astype(np.float64), axis=1).astype(np.float32)}
    for name, value in arrays.items():
        analysis._save_array(path / f"{name}.npy", value)
    receipt = {"schema": analysis.SCHEMA, "core": record, "inputs_sha256": inputs["inputs_sha256"],
        "checkpoint_sha256": inputs["checkpoint_sha256"], "dtype": "float32", "tf32": False,
        "deterministic_algorithms": True, "cudnn_deterministic": True,
        "elapsed_seconds": 1., "peak_allocated_bytes": 100,
        "audit": {"all_observed": True, "masked_entries": 0, "complete_core_graph": True,
            "prediction_invariance_checked": True, "h0_equals_direct_encoder": True,
            "predictions_exact": True, "ordinary_hL_exact": True, "hL_equals_full_node_embedding": True},
        "files": {f"{name}.npy": {"sha256": analysis.file_hash(path / f"{name}.npy")}
                  for name in arrays}}
    analysis._write_json(path / "receipt.json", receipt)
    return receipt


def test_receipt_verification_detects_mutation_even_with_same_shape(tmp_path):
    record = {"core_number": 1, "n_cells": 2, "cell_rows_sha256": "ordered"}
    inputs = {"inputs_sha256": "inputs", "checkpoint_sha256": "checkpoint", "dimensions": 4}
    _write_core(tmp_path, record, inputs)
    analysis._verify_core(tmp_path, record, inputs)
    path = tmp_path / "embeddings/core_1/h0.npy"
    h0 = np.load(path)
    h0[0, 0] += 1
    np.save(path, h0)
    with pytest.raises(ValueError, match="checksum changed"):
        analysis._verify_core(tmp_path, record, inputs)


def test_collect_requires_all_cores_and_bound_order(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "EXPECTED_CELLS", 56)
    rows = []
    for core in analysis.CORE_NUMBERS:
        for index in range(2):
            rows.append({"row_index": len(rows), "slide": 1 if core <= 14 else 2,
                "fov": core, "cell_ID": index, "core_number": core,
                "alias": f"SO{1 if core <= 14 else 2}-C{core:02d}",
                "x_um": float(index), "y_um": float(core), "split": "train"})
    cells = pd.DataFrame(rows, columns=analysis.CELL_COLUMNS)
    cells.to_parquet(tmp_path / "cells.parquet", index=False)
    records = [{"core_number": c, "alias": cells.iloc[(c-1)*2].alias, "n_cells": 2,
        "row_start": (c-1)*2, "row_stop": c*2,
        "cell_rows_sha256": analysis._row_digest(cells.iloc[(c-1)*2:c*2])}
        for c in analysis.CORE_NUMBERS]
    inputs = {"schema": analysis.SCHEMA, "cores": records, "dimensions": 4,
        "checkpoint_sha256": "checkpoint", "checkpoint_epoch": 151,
        "cells_sha256": analysis.file_hash(tmp_path / "cells.parquet"),
        "cell_rows_sha256": analysis._row_digest(cells)}
    inputs["inputs_sha256"] = analysis.digest(inputs)
    analysis._write_json(tmp_path / "extraction_inputs.json", inputs)
    for record in records[:-1]:
        receipt = _write_core(tmp_path, record, inputs)
        if record["core_number"] == 23:
            analysis._write_json(tmp_path / "diagnostics/extraction_pilot.json", receipt)
    with pytest.raises(ValueError, match="missing extraction"):
        analysis.collect_extraction(tmp_path)
    _write_core(tmp_path, records[-1], inputs)
    result = analysis.collect_extraction(tmp_path)
    assert result["verified"] and result["n_cells"] == 56 and result["core_count"] == 28
    # A file with valid row count but reassigned cell keys must fail provenance.
    cells.loc[[0, 1], "cell_ID"] = cells.loc[[1, 0], "cell_ID"].to_numpy()
    cells.to_parquet(tmp_path / "cells.parquet", index=False)
    with pytest.raises(ValueError, match="Cell index changed"):
        analysis.collect_extraction(tmp_path)


def test_audited_sources_cannot_change_between_prepare_and_worker(tmp_path):
    path = tmp_path / "counts.npy"
    path.write_bytes(b"original")
    record = {"path": str(path), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    analysis._check_source_stats([record])
    path.write_bytes(b"different")
    with pytest.raises(ValueError, match="Audited source changed"):
        analysis._check_source_stats([record])
