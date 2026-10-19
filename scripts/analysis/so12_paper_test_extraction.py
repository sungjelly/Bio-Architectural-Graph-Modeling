"""Deferred, frozen-checkpoint hL extraction for paper test cores 15 and 23.

Only the completed, checksum-verified source run can unlock the test loader.
The caller owns scheduling, GPU locks, registration, and report publication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from spatial_benchmark.paths import ProjectPaths, current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import verify_run_bundle
from spatial_benchmark.so12_paper_split_data import (
    DATASET_ID, DATASET_VERSION, SPLIT_ID, UPSTREAM_OBSERVATIONS_SHA256,
    _read_cache, load_so12_paper_data, split_contract, visible_count_pearson_input,
)
from spatial_benchmark.so12_paper_split_training import (
    CHECKPOINT_SCHEMA, PRIMARY, PROTOCOL, TEST_NAMESPACE, build_model, digest,
    file_hash, normalization_tensors, scientific_binding,
    source_binding as training_source_binding,
)

SOURCE_RUN_ID = "r_20261017T044837Z_caf76350_s000_f00_a01_12df41be"
SCHEMA = "so12_paper_test_hL_extraction_v1"
CORE_CELLS = {15: 38145, 23: 43462}
CORE_ALIASES = ("SO2-C15", "SO2-C23")
N_CELLS = 81607
DIMENSIONS = 256
CELL_COLUMNS = ("row_index", "slide", "fov", "cell_ID", "core_number", "alias",
                "x_um", "y_um", "split")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "Expected a JSON mapping.")
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def analysis_source_binding() -> dict[str, str]:
    """Callback for the orchestration receipt; training sources are bound separately."""
    return {"scripts/analysis/so12_paper_test_extraction.py": file_hash(Path(__file__))}


def verify_source(run_dir: Path, paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Fail closed before accessing test arrays unless the exact source is finalized."""
    paths = paths or current_paths()
    source = Path(run_dir).resolve()
    expected = (paths.artifact_root / "runs/2026/10" / SOURCE_RUN_ID).resolve()
    _require(source == expected, "Extraction requires the exact approved immutable source run.")
    run = Registry(paths.state_root / "tracking/bagm.sqlite3").get_run(SOURCE_RUN_ID)
    _require(run is not None and run["status"] == "completed"
             and bool(run.get("artifact_path"))
             and Path(run["artifact_path"]).resolve() == source,
             "Source run is not completed and published; keep the analysis waiting.")
    _require((source / "_SUCCESS").is_file(), "Source run has no success marker.")
    verification = verify_run_bundle(source)
    _require(verification.get("valid") and verification.get("status") == "success",
             "Source bundle verification failed.")
    config = yaml.safe_load((source / "config.resolved.yaml").read_text())
    _require(config["dataset"]["dataset_id"] == DATASET_ID
             and config["dataset"]["version"] == DATASET_VERSION
             and config["dataset"]["split_id"] == SPLIT_ID
             and config["model"]["graph_layers"] == 4
             and config["model"]["hidden_dim"] == DIMENSIONS,
             "Source model or split is outside the fixed extraction contract.")
    startup = _read(source / "diagnostics/training_start.json")
    checkpoint_path = source / "checkpoints/best.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_sha = file_hash(checkpoint_path)
    _require(checkpoint["run_id"] == SOURCE_RUN_ID
             and checkpoint["checkpoint_schema"] == CHECKPOINT_SCHEMA
             and checkpoint["protocol"] == PROTOCOL and checkpoint["checkpoint_role"] == "best"
             and checkpoint["primary_metric_name"] == PRIMARY
             and checkpoint["binding"] == startup["binding"],
             "Selected checkpoint ownership, role, schema, or startup binding changed.")
    binding = checkpoint["binding"]
    _require(binding["config_sha256"] == scientific_binding(config)
             and binding["source_sha256"] == training_source_binding()
             and binding["dataset_sha256"] == digest(startup["dataset"]),
             "Frozen configuration, source code, or training-data binding changed.")
    selection_path = source / "diagnostics/selection_frozen_before_test.json"
    test_path = source / "diagnostics/test_evaluation.json"
    selection, test = _read(selection_path), _read(test_path)
    for receipt in (selection, test):
        _require(receipt.get("checkpoint") == "checkpoints/best.ckpt"
                 and receipt.get("checkpoint_sha256") == checkpoint_sha
                 and receipt.get("best_epoch") == checkpoint["epoch"]
                 and receipt.get("selection_completed") is True
                 and receipt.get("validation_replay_verified") is True
                 and receipt.get("test_used_for_selection") is False
                 and receipt.get("primary_metric_name") == PRIMARY
                 and receipt.get("primary_metric_value") == checkpoint["primary_metric_value"],
                 "Final selection/test receipt does not identify the selected checkpoint.")
    expected_coverage = {(alias, view) for alias in CORE_ALIASES for view in range(3)}
    coverage = test.get("core_view_coverage", [])
    _require(test.get("test_loaded_after_selection") is True
             and test.get("test_evaluation_count") == 1
             and test.get("mask_namespace") == TEST_NAMESPACE
             and test.get("test_core_aliases") == list(CORE_ALIASES)
             and test.get("test_cells") == N_CELLS
             and test.get("mask_views") == 3 and test.get("evaluated_core_view_count") == 6
             and len(coverage) == 6
             and {(row["core_alias"], row["view"]) for row in coverage} == expected_coverage,
             "Final test coverage is incomplete or belongs to another partition.")
    train_provenance = startup["dataset"]["provenance"]
    test_provenance = test["test_data_provenance"]
    _require(test_provenance.get("loaded_roles") == ["test"]
             and test_provenance.get("opened_core_aliases") == list(CORE_ALIASES)
             and train_provenance["split"] == test_provenance["split"] == split_contract(),
             "Recorded data roles or split do not match the approved test cores.")
    cache, manifest = _read_cache(paths, True)  # Small frozen parameters/metadata only.
    for name in ("dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint"):
        _require(train_provenance[name] == test_provenance[name] == manifest[name],
                 f"Frozen {name} changed since training/final test.")
    _require(train_provenance["fitted_parameters_sha256"]
             == test_provenance["fitted_parameters_sha256"]
             == manifest["files"]["fitted_parameters.npz"],
             "Training-fitted normalization parameters changed.")
    _require(test_provenance["source_manifest_sha256"] == file_hash(cache / "manifest.json"),
             "Frozen preprocessing manifest changed.")
    result = {
        "schema": SCHEMA, "source_run_id": SOURCE_RUN_ID, "source_run_dir": str(source),
        "checkpoint_relative_path": "checkpoints/best.ckpt", "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_validation_huber": float(checkpoint["primary_metric_value"]),
        "binding": binding, "config": config,
        "test_receipt_sha256": file_hash(test_path), "selection_receipt_sha256": file_hash(selection_path),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "split_fingerprint": manifest["split_fingerprint"],
        "preprocessing_fingerprint": manifest["preprocessing_fingerprint"],
        "fitted_parameters_sha256": manifest["files"]["fitted_parameters.npz"],
        "metadata_statistics_sha256": manifest["files"]["metadata_statistics.json"],
        "source_manifest_sha256": file_hash(cache / "manifest.json"),
        "final_test_completed": True, "source_bundle_verified": True,
    }
    result["source_identity_sha256"] = digest(result)
    return result


def _check_test_dataset(dataset: Any) -> None:
    _require([core.alias for core in dataset.cores] == list(CORE_ALIASES)
             and all(core.split == "test" and core.slide == 2 for core in dataset.cores)
             and {int(core.alias[-2:]): core.n_cells for core in dataset.cores} == CORE_CELLS,
             "Extraction dataset must contain exactly test cores 15 and 23 in canonical order.")


def _row_digest(frame: pd.DataFrame) -> str:
    return digest(frame[list(CELL_COLUMNS)].to_dict(orient="list"))


def prepare_cells(dataset: Any, output: Path, paths: ProjectPaths | None = None) -> pd.DataFrame:
    """Save only canonical test-cell keys/coordinates; no clinical fields."""
    paths = paths or current_paths()
    _check_test_dataset(dataset)
    observations = paths.data_root / "processed/so12_pearson_integration_v1/observations.csv.gz"
    _require(file_hash(observations) == UPSTREAM_OBSERVATIONS_SHA256,
             "Canonical observation index changed.")
    selected = []
    columns = ["slide", "fov", "cell_ID", "core_number", "core", "x_um", "y_um"]
    for batch in pd.read_csv(observations, usecols=columns, chunksize=50000):
        selected.append(batch.loc[batch.core.isin(CORE_ALIASES)])
    obs = pd.concat(selected, ignore_index=True)
    frames = []
    for core in dataset.cores:
        rows = obs.loc[obs.core == core.alias].copy()
        _require(len(rows) == core.n_cells and rows.slide.eq(2).all()
                 and rows.core_number.eq(int(core.alias[-2:])).all()
                 and rows[["fov", "cell_ID"]].equals(
                     rows[["fov", "cell_ID"]].sort_values(["fov", "cell_ID"], kind="mergesort"))
                 and np.allclose(rows[["x_um", "y_um"]].to_numpy(), core.coordinates_um,
                                 rtol=0, atol=1e-9), f"{core.alias} cell alignment changed.")
        rows["split"] = "test"
        frames.append(rows.rename(columns={"core": "alias"}))
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "row_index", np.arange(len(frame), dtype=np.int64))
    frame = frame[list(CELL_COLUMNS)]
    _require(len(frame) == N_CELLS and not frame.duplicated(["slide", "fov", "cell_ID"]).any(),
             "Test-cell keys are duplicated or incomplete.")
    destination = Path(output) / "cells.csv.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(not destination.exists(), "Refusing to overwrite an existing cell index.")
    temporary = destination.with_name(f".cells.{os.getpid()}.writing")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False)
    os.replace(temporary, destination)
    return frame


def configure_deterministic_inference() -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _require(os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"),
             "CUBLAS workspace is incompatible with deterministic inference.")
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"dtype": "float32", "tf32": False, "deterministic_algorithms": True,
            "cudnn_deterministic": True, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def extract_core_array(model: Any, core: Any, normalization: Mapping[str, torch.Tensor],
                       device: torch.device, *, verify_invariance: bool = True,
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """Full-core FP32 hL; synthetic fixtures exercise the identical forward path."""
    _require(not any(module.training for module in model.modules()), "Extraction requires eval mode.")
    _require(len(model.blocks) == 4 and all(p.dtype == torch.float32 for p in model.parameters()),
             "Extraction requires four FP32 graph blocks.")
    raw = torch.tensor(np.asarray(core.raw_counts), dtype=torch.int32, device=device)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    covariates = torch.tensor(np.asarray(core.node_covariates), dtype=torch.float32, device=device)
    # Preserve all original edges/geometry in FP32. This requires no graph copy
    # on disk and no sampled/subgraph receptive fields.
    edges = torch.tensor(np.asarray(core.edge_index), dtype=torch.int64, device=device)
    geometry = torch.tensor(np.asarray(core.relative_geometry), dtype=torch.float32, device=device)
    calls, direct_last = [], []
    original = model._chunked_layer
    had_override = "_chunked_layer" in model.__dict__
    previous_override = model.__dict__.get("_chunked_layer")
    def capture_last(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(args[0])
        if len(calls) == 4:
            direct_last.append(value[0].detach().clone())
        return value
    model._chunked_layer = capture_last
    try:
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            visible = visible_count_pearson_input(raw, mask, **normalization)
            arguments = dict(input_expression=visible, gene_mask=mask, node_covariates=covariates,
                             edge_index=edges, relative_geometry=geometry)
            output = model(**arguments, return_intermediate_embeddings=True)
            hL = output.final_graph_embedding
            _require(len(calls) == 4 and all(a is b for a, b in zip(calls, model.blocks, strict=True)),
                     "The four original graph blocks were not executed exactly once.")
            _require(hL is not None and hL.shape == (len(raw), model.hidden_dim)
                     and hL.dtype == torch.float32 and bool(torch.isfinite(hL).all())
                     and bool((torch.linalg.vector_norm(hL, dim=1) > 0).all())
                     and bool(torch.isfinite(output.prediction).all()),
                     "Hidden states are nonfinite, zero, or have an invalid schema.")
            _require(torch.equal(hL, output.full_node_embedding)
                     and torch.equal(hL, direct_last[0]),
                     "Returned hL differs from the direct fourth graph-block output.")
            audit = {"all_observed": True, "masked_entries": 0, "complete_core_graph": True,
                     "graph_blocks": 4, "direct_fourth_block_exact": True,
                     "hL_equals_full_node_embedding": True,
                     "prediction_invariance_checked": verify_invariance}
            if verify_invariance:
                plain = model(**arguments, return_intermediate_embeddings=False)
                difference = float((plain.prediction - output.prediction).abs().max())
                _require(torch.equal(plain.prediction, output.prediction)
                         and torch.equal(plain.full_node_embedding, hL),
                         f"Extraction changed ordinary predictions or states: max difference {difference:.9g}.")
                audit.update(predictions_exact=True, ordinary_hL_exact=True,
                             maximum_absolute_prediction_difference=difference)
            array = hL.cpu().numpy().copy()
    finally:
        if had_override:
            model._chunked_layer = previous_override
        else:
            delattr(model, "_chunked_layer")
        model.clear_edge_layout_cache()
    return array, audit


def extract(source_run_dir: Path, output: Path, device: str = "cuda:0",
            paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Verify source, run the deferred full-core pilot, and export only test hL."""
    paths = paths or current_paths()
    destination = Path(output).resolve()
    active = (paths.scratch_root / "active_runs").resolve()
    _require(destination.is_relative_to(active) and destination != active,
             "Extraction output must belong to a separate active analysis run.")
    source = verify_source(source_run_dir, paths)  # First gate, before the test loader.
    execution = configure_deterministic_inference()
    destination.mkdir(parents=True, exist_ok=True)
    _require(shutil.disk_usage(destination).free >= (5.5 * 1024**3),
             "Extraction would violate the 5 GiB reserve plus analysis budget.")
    _require(not (destination / "embeddings/hL.npy").exists()
             and not (destination / "extraction_manifest.json").exists(),
             "Refusing to overwrite an existing extraction.")
    started = time.monotonic()
    dataset = load_so12_paper_data(source["config"], paths, roles=("test",),
                                    verify_checksums=True, load_graphs=True)
    _check_test_dataset(dataset)
    for key in ("dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint",
                "fitted_parameters_sha256", "source_manifest_sha256"):
        _require(dataset.provenance[key] == source[key], f"Loaded test {key} changed.")
    cells = prepare_cells(dataset, destination, paths)
    _write(destination / "source_verified.json", source)
    _write(destination / "test_data_provenance.json", dataset.provenance)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = build_model(source["config"], genes=len(dataset.genes),
                        covariates=dataset.cores[0].node_covariates.shape[1]).to(torch_device).float().eval()
    checkpoint_path = Path(source["source_run_dir"]) / source["checkpoint_relative_path"]
    _require(file_hash(checkpoint_path) == source["checkpoint_sha256"], "Checkpoint changed after verification.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    normalization = normalization_tensors(dataset, torch_device)
    embedding_dir = destination / "embeddings"
    embedding_dir.mkdir(exist_ok=True)
    combined = np.lib.format.open_memmap(embedding_dir / "hL.npy", mode="w+", dtype=np.float32,
                                         shape=(N_CELLS, DIMENSIONS))
    receipts = []
    # Largest complete graph first: a real deferred pilot, then reuse its output.
    for core in sorted(dataset.cores, key=lambda value: -value.n_cells):
        core_started = time.monotonic()
        array, audit = extract_core_array(model, core, normalization, torch_device, verify_invariance=True)
        rows = cells.loc[cells.alias == core.alias]
        start, stop = int(rows.row_index.iloc[0]), int(rows.row_index.iloc[-1]) + 1
        _require(array.shape == (stop - start, DIMENSIONS), "Extracted cell count changed.")
        combined[start:stop] = array
        combined.flush()
        receipt = {"schema": SCHEMA, "core_number": int(core.alias[-2:]), "alias": core.alias,
                   "n_cells": core.n_cells, "row_start": start, "row_stop": stop,
                   "cell_rows_sha256": _row_digest(rows), "dimensions": DIMENSIONS,
                   "embedding_slice_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                   "checkpoint_sha256": source["checkpoint_sha256"],
                   "preprocessing_fingerprint": source["preprocessing_fingerprint"],
                   "elapsed_seconds": time.monotonic() - core_started, "execution": execution,
                   "edges": int(core.edge_index.shape[1]), "audit": audit}
        _write(embedding_dir / f"core_{int(core.alias[-2:])}_receipt.json", receipt)
        if core.alias == "SO2-C23":
            _write(destination / "diagnostics/extraction_pilot.json", {**receipt, "passed": True})
        receipts.append({**receipt,
                         "receipt_path": f"embeddings/core_{int(core.alias[-2:])}_receipt.json",
                         "receipt_sha256": file_hash(embedding_dir / f"core_{int(core.alias[-2:])}_receipt.json")})
        del array
    _require(np.isfinite(combined).all() and np.all(np.linalg.norm(combined, axis=1) > 0),
             "Final embedding coverage is nonfinite or zero.")
    del combined
    _require(file_hash(checkpoint_path) == source["checkpoint_sha256"]
             and training_source_binding() == source["binding"]["source_sha256"],
             "Frozen checkpoint or source changed during extraction.")
    result = {"schema": SCHEMA, "verified": True, "passed": True, "status": "complete",
              "source_run_id": SOURCE_RUN_ID,
              "created_at": datetime.now(timezone.utc).isoformat(), "n_cells": N_CELLS,
              "dimensions": DIMENSIONS, "core_numbers": list(CORE_CELLS),
              "test_core_numbers": list(CORE_CELLS), "all_observed": True,
              "cells_per_core": {str(number): count for number, count in CORE_CELLS.items()},
              "representation": "fourth_graph_block_output_before_decoder",
              "checkpoint_sha256": source["checkpoint_sha256"],
              "preprocessing_fingerprint": source["preprocessing_fingerprint"],
              "fitted_parameters_sha256": source["fitted_parameters_sha256"],
              "source_code_sha256": source["binding"]["source_sha256"],
              "source_identity_sha256": source["source_identity_sha256"],
              "cells_path": "cells.csv.gz", "hL_path": "embeddings/hL.npy",
              "cells_sha256": file_hash(destination / "cells.csv.gz"),
              "cell_rows_sha256": _row_digest(cells),
              "hL_sha256": file_hash(embedding_dir / "hL.npy"),
              "cores": sorted(receipts, key=lambda value: value["core_number"]),
              "analysis_source_sha256": analysis_source_binding(), "execution": execution,
              "elapsed_seconds": time.monotonic() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(torch_device) if torch_device.type == "cuda" else 0,
              "test_only": True, "normalization_refit": False, "source_modified": False}
    result["embedding_sha256"] = result["hL_sha256"]
    _write(destination / "extraction_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = extract(args.source_run_dir, args.output, args.device)
    print(json.dumps({"n_cells": result["n_cells"], "hL_path": str(args.output / result["hL_path"]),
                      "verified": result["verified"]}))


if __name__ == "__main__":
    main()
