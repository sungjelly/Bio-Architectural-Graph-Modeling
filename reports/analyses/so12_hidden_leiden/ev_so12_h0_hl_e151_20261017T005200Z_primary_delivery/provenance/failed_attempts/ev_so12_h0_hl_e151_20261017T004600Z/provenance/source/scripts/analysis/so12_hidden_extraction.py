"""Verified all-observed h0/hL extraction from the completed SO12 checkpoint.

The caller owns experiment registration/publication. This module writes only to
the caller's active run and never changes the frozen training bundle or data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import verify_run_bundle
from spatial_benchmark.so12_relative_qkv_data import (
    EXPECTED_CELLS, load_so12_training_data, visible_count_pearson_input,
)
from spatial_benchmark.so12_relative_qkv_training import (
    CHECKPOINT_SCHEMA, _metadata, build_model, digest, file_hash,
    normalization_tensors, scientific_binding, source_binding,
)

SCHEMA = "so12_all_observed_hidden_extraction_v1"
CORE_NUMBERS = tuple(range(1, 29))
CELL_COLUMNS = ("row_index", "slide", "fov", "cell_ID", "core_number", "alias",
                "x_um", "y_um", "split")


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.writing")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, path)


def _active_destination(out: Path) -> Path:
    out = out.resolve()
    active = (current_paths().scratch_root / "active_runs").resolve()
    _require(out.is_relative_to(active) and out != active,
             "Extraction output must be inside a separate scratch/active_runs run.")
    return out


def _verified_metadata(value: Any) -> Any:
    """Normalize verification-report flags, never scientific values or hashes."""
    if isinstance(value, dict):
        return {k: True if k in ("checksum_verified", "source_file_checksums_verified")
                else _verified_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_verified_metadata(v) for v in value]
    return value


def _read_source(source: Path, *, full_verify: bool) -> tuple[dict, dict, dict]:
    source = source.resolve()
    _require((source / "_SUCCESS").is_file(), "Source training has no success marker.")
    if full_verify:
        verified = verify_run_bundle(source)
        _require(verified["valid"] and verified["status"] == "success",
                 "Training bundle verification failed.")
        run = Registry().get_run(source.name)
        _require(run is not None and run["status"] == "completed"
                 and Path(run["artifact_path"]).resolve() == source,
                 "Training registry does not identify this completed bundle.")
    config = yaml.safe_load((source / "config.resolved.yaml").read_text())
    startup = _json(source / "diagnostics/training_start.json")
    checkpoint = torch.load(source / "checkpoints/best.ckpt", map_location="cpu", weights_only=False)
    _require(checkpoint["checkpoint_schema"] == CHECKPOINT_SCHEMA
             and checkpoint["checkpoint_role"] == "best"
             and checkpoint["run_id"] == source.name
             and checkpoint["binding"] == startup["binding"],
             "Best checkpoint ownership, schema, or startup binding changed.")
    binding = checkpoint["binding"]
    _require(binding["config_sha256"] == scientific_binding(config)
             and binding["source_sha256"] == source_binding(),
             "Current training code/configuration differs from the checkpoint.")
    _require(config["model"]["graph_layers"] == 4
             and config["model"]["hidden_dim"] == 256,
             "This analysis requires the four-layer 256-dimensional model.")
    return config, startup, checkpoint


def _cell_frame(dataset: Any) -> pd.DataFrame:
    path = current_paths().data_root / "processed/so12_pearson_integration_v1/observations.csv.gz"
    obs = pd.read_csv(path, usecols=["slide", "fov", "cell_ID", "core_number", "core",
                                        "x_um", "y_um", "train"])
    frames = []
    for core in dataset.cores:
        rows = obs.loc[obs.core == core.alias].copy()
        _require(len(rows) == core.n_cells and rows[["fov", "cell_ID"]].equals(
            rows[["fov", "cell_ID"]].sort_values(["fov", "cell_ID"], kind="mergesort")),
            f"Cell order differs for {core.alias}.")
        _require(np.allclose(rows[["x_um", "y_um"]], core.coordinates_um, rtol=0, atol=1e-9),
                 f"Cell coordinates differ for {core.alias}.")
        rows["split"] = np.where(rows.train, "train", "validation")
        _require(rows.split.eq(core.split).all(), "Cell split differs from training.")
        frames.append(rows.rename(columns={"core": "alias"}).drop(columns="train"))
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "row_index", np.arange(len(frame), dtype=np.int64))
    _require(len(frame) == EXPECTED_CELLS
             and not frame.duplicated(["slide", "fov", "cell_ID"]).any(),
             "Cell keys are duplicated or incomplete.")
    return frame[list(CELL_COLUMNS)]


def _row_digest(frame: pd.DataFrame) -> str:
    return digest(frame[list(CELL_COLUMNS)].to_dict(orient="list"))


def _source_files(dataset: Any) -> list[dict]:
    records: dict[str, str] = {}
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                records[value["path"]] = value["sha256"]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(dataset.provenance)
    for core in dataset.cores:
        visit(core.source)
    observations = current_paths().data_root / "processed/so12_pearson_integration_v1/observations.csv.gz"
    records[str(observations)] = dataset.provenance["observations_sha256"]
    return [{"path": name, "sha256": checksum, "size_bytes": Path(name).stat().st_size,
             "mtime_ns": Path(name).stat().st_mtime_ns}
            for name, checksum in sorted(records.items())]


def _check_source_stats(records: list[dict]) -> None:
    for record in records:
        stat = Path(record["path"]).stat()
        _require(stat.st_size == record["size_bytes"] and stat.st_mtime_ns == record["mtime_ns"],
                 f"Audited source changed since preparation: {Path(record['path']).name}")


def prepare_extraction(source_run_dir: Path, out: Path) -> dict:
    """Verify the frozen sources once and write the anonymous observation index."""
    out, source = _active_destination(out), source_run_dir.resolve()
    _require(not (out / "extraction_inputs.json").exists(), "Extraction was already prepared.")
    out.mkdir(parents=True, exist_ok=True)
    config, startup, checkpoint = _read_source(source, full_verify=True)
    dataset = load_so12_training_data(config, verify_checksums=True)
    metadata = _metadata(dataset)
    _require(metadata == startup["dataset"]
             and digest(metadata) == checkpoint["binding"]["dataset_sha256"],
             "Reconstructed dataset does not match the checkpoint binding.")
    cells = _cell_frame(dataset)
    temporary = out / f".cells.{os.getpid()}.parquet"
    cells.to_parquet(temporary, index=False)
    os.replace(temporary, out / "cells.parquet")
    core_records = []
    for core in dataset.cores:
        rows = cells.loc[cells.alias == core.alias]
        core_records.append({"core_number": int(rows.core_number.iloc[0]), "alias": core.alias,
            "n_cells": core.n_cells, "row_start": int(rows.row_index.iloc[0]),
            "row_stop": int(rows.row_index.iloc[-1]) + 1, "cell_rows_sha256": _row_digest(rows),
            "edges": int(core.edge_index.shape[1]), "split": core.split})
    result = {"schema": SCHEMA, "prepared_at": _now(), "source_run_id": source.name,
        "source_run_dir": str(source), "checkpoint_relative_path": "checkpoints/best.ckpt",
        "checkpoint_sha256": file_hash(source / "checkpoints/best.ckpt"),
        "checkpoint_epoch": int(checkpoint["completed_epoch"]),
        "checkpoint_validation_huber": float(checkpoint["primary_metric_value"]),
        "binding": checkpoint["binding"], "dataset_metadata": metadata,
        "source_files": _source_files(dataset), "source_files_checksums_verified": True,
        "cells_sha256": file_hash(out / "cells.parquet"), "cell_rows_sha256": _row_digest(cells),
        "n_cells": len(cells), "dimensions": 256, "cores": core_records,
        "all_observed": True, "inference_dtype": "float32", "tf32": False,
        "analysis_source_sha256": file_hash(Path(__file__)),
        "representation_definitions": {"h0": "trained_node_encoder_output_before_graph_blocks",
            "hL": "output_after_four_graph_blocks_before_expression_decoder"}}
    result["inputs_sha256"] = digest(result)
    _write_json(out / "extraction_inputs.json", result)
    return result


def _load_inputs(out: Path) -> dict:
    inputs = _json(out / "extraction_inputs.json")
    unhashed = dict(inputs)
    expected = unhashed.pop("inputs_sha256")
    _require(digest(unhashed) == expected and inputs["schema"] == SCHEMA,
             "Extraction preparation receipt changed.")
    _require(file_hash(out / "cells.parquet") == inputs["cells_sha256"], "Cell index changed.")
    return inputs


def extract_core_arrays(model: Any, core: Any, normalization: dict,
                        device: torch.device, *, verify_invariance: bool = False) -> tuple[dict, dict]:
    """Full-graph FP32 extraction; small CPU fixtures can exercise the same path."""
    _require(not any(module.training for module in model.modules()), "Extraction requires eval mode.")
    _require(all(p.dtype == torch.float32 for p in model.parameters()), "Model weights must be FP32.")
    raw = torch.tensor(np.asarray(core.raw_counts), device=device, dtype=torch.int32)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    covariates = torch.tensor(np.asarray(core.node_covariates), device=device, dtype=torch.float32)
    edges = torch.tensor(np.asarray(core.edge_index), device=device, dtype=torch.long)
    geometry = torch.tensor(np.asarray(core.relative_geometry), device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        visible = visible_count_pearson_input(raw, mask, **normalization)
        args = dict(input_expression=visible, gene_mask=mask, node_covariates=covariates,
                    edge_index=edges, relative_geometry=geometry)
        output = model(**args, return_intermediate_embeddings=True)
        h0, hL = output.node_encoder_embedding, output.final_graph_embedding
        expected = (len(raw), model.hidden_dim)
        _require(h0 is not None and hL is not None and h0.shape == hL.shape == expected,
                 "Intermediate outputs are incomplete or have the wrong shape.")
        _require(h0.dtype == hL.dtype == torch.float32
                 and bool(torch.isfinite(h0).all()) and bool(torch.isfinite(hL).all())
                 and bool(torch.isfinite(output.prediction).all()), "Nonfinite/non-FP32 model outputs.")
        _require(torch.equal(hL, output.full_node_embedding), "hL differs from full node states.")
        audit = {"all_observed": True, "masked_entries": int(mask.sum()), "complete_core_graph": True,
                 "prediction_invariance_checked": verify_invariance, "hL_equals_full_node_embedding": True}
        if verify_invariance:
            direct_h0 = model.encoder(input_expression=visible, gene_mask=mask, node_covariates=covariates)
            plain = model(**args, return_intermediate_embeddings=False)
            _require(torch.equal(direct_h0, h0), "h0 differs from the direct trained encoder.")
            _require(torch.equal(plain.prediction, output.prediction),
                     "Enabling intermediate states changed predictions.")
            _require(torch.equal(plain.full_node_embedding, hL), "Ordinary forward changed hL.")
            audit.update(h0_equals_direct_encoder=True, predictions_exact=True, ordinary_hL_exact=True,
                         maximum_absolute_prediction_difference=0.0)
        a0, aL = h0.cpu().numpy().copy(), hL.cpu().numpy().copy()
    delta = np.linalg.norm(aL.astype(np.float64) - a0.astype(np.float64), axis=1).astype(np.float32)
    return {"h0": a0, "hL": aL, "delta_l2": delta}, audit


def _verify_core(out: Path, expected: dict, inputs: dict) -> dict:
    root = out / "embeddings" / f"core_{expected['core_number']}"
    receipt = _json(root / "receipt.json")
    _require(receipt["schema"] == SCHEMA and receipt["core"] == expected
             and receipt["inputs_sha256"] == inputs["inputs_sha256"]
             and receipt["checkpoint_sha256"] == inputs["checkpoint_sha256"]
             and receipt["dtype"] == "float32" and receipt["tf32"] is False
             and receipt["audit"]["all_observed"] is True and receipt["audit"]["masked_entries"] == 0
             and receipt["audit"]["complete_core_graph"] is True,
             "Core receipt belongs to different inputs or row order.")
    arrays = {}
    for name in ("h0", "hL", "delta_l2"):
        path = root / f"{name}.npy"
        _require(file_hash(path) == receipt["files"][path.name]["sha256"], f"{name} checksum changed.")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = (expected["n_cells"],) if name == "delta_l2" else (expected["n_cells"], inputs["dimensions"])
        _require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                 f"Invalid {name} shape, dtype or finite values.")
        arrays[name] = value
    delta = np.linalg.norm(arrays["hL"].astype(np.float64) - arrays["h0"].astype(np.float64), axis=1).astype(np.float32)
    _require(np.array_equal(delta, arrays["delta_l2"]), "Saved delta norms differ from hL-h0.")
    return receipt


def run_worker(source_run_dir: Path, out: Path, device_name: str, cores: list[int], *, pilot: bool = False) -> dict:
    out, source = _active_destination(out), source_run_dir.resolve()
    inputs = _load_inputs(out)
    _require(len(cores) == len(set(cores)) and set(cores).issubset(CORE_NUMBERS) and bool(cores),
             "Core selection is empty, duplicated, or outside 1 through 28.")
    _require(not pilot or cores == [23], "The representative pilot must use largest core 23.")
    _require(str(source) == inputs["source_run_dir"]
             and file_hash(source / "checkpoints/best.ckpt") == inputs["checkpoint_sha256"]
             and file_hash(Path(__file__)) == inputs["analysis_source_sha256"],
             "Source checkpoint or extraction code changed after preparation.")
    _check_source_stats(inputs["source_files"])
    config, startup, checkpoint = _read_source(source, full_verify=False)
    _require(checkpoint["binding"] == inputs["binding"], "Checkpoint binding changed.")
    dataset = load_so12_training_data(config, verify_checksums=False)
    _require(_verified_metadata(_metadata(dataset)) == inputs["dataset_metadata"],
             "Worker dataset differs from fully audited preparation.")
    device = torch.device(device_name)
    _require(device.type == "cuda" and torch.cuda.is_available(), "Full extraction requires an available GPU.")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model(config, genes=len(dataset.gene_p), covariates=dataset.cores[0].node_covariates.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    del checkpoint
    norm = normalization_tensors(dataset, device)
    records = {record["core_number"]: record for record in inputs["cores"]}
    core_data = {int(core.alias.split("-C")[1]): core for core in dataset.cores}
    completed = []
    for number in cores:
        root = out / "embeddings" / f"core_{number}"
        if (root / "receipt.json").exists():
            existing = _verify_core(out, records[number], inputs)
            _require(not pilot or existing["audit"]["prediction_invariance_checked"],
                     "Existing extraction does not contain pilot invariance checks.")
            completed.append(number)
            print(json.dumps({"stage": "core_reused", "core": number}), flush=True)
            continue
        _require(not root.exists() or not any(root.iterdir()), "Partial core output exists without a receipt.")
        root.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        arrays, audit = extract_core_arrays(model, core_data[number], norm, device, verify_invariance=pilot)
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        for name, array in arrays.items():
            _save_array(root / f"{name}.npy", array)
        receipt = {"schema": SCHEMA, "completed_at": _now(), "core": records[number],
            "inputs_sha256": inputs["inputs_sha256"], "checkpoint_sha256": inputs["checkpoint_sha256"],
            "checkpoint_epoch": inputs["checkpoint_epoch"], "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "tf32": False, "dtype": "float32",
            "elapsed_seconds": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "audit": audit,
            "files": {f"{name}.npy": {"sha256": file_hash(root / f"{name}.npy"),
                "size_bytes": (root / f"{name}.npy").stat().st_size} for name in arrays}}
        _write_json(root / "receipt.json", receipt)
        _verify_core(out, records[number], inputs)
        if pilot:
            _write_json(out / "diagnostics/extraction_pilot.json", receipt)
        completed.append(number)
        print(json.dumps({"stage": "core_complete", "core": number, "seconds": elapsed,
                          "peak_allocated_gib": receipt["peak_allocated_bytes"] / 2**30}), flush=True)
        del arrays
        model.clear_edge_layout_cache()
    return {"completed_cores": completed, "device": str(device), "pilot": pilot}


def collect_extraction(out: Path) -> dict:
    """Verify the complete, nonoverlapping, checkpoint-bound all-cell export."""
    inputs = _load_inputs(out)
    records = inputs["cores"]
    _require([r["core_number"] for r in records] == list(CORE_NUMBERS), "Expected all 28 ordered cores.")
    actual = {p.name for p in (out / "embeddings").iterdir() if p.is_dir()}
    _require(actual == {f"core_{n}" for n in CORE_NUMBERS}, "Unexpected or missing extraction core directories.")
    cells = pd.read_parquet(out / "cells.parquet")
    _require(tuple(cells.columns) == CELL_COLUMNS and len(cells) == EXPECTED_CELLS
             and np.array_equal(cells.row_index, np.arange(EXPECTED_CELLS))
             and not cells.duplicated(["slide", "fov", "cell_ID"]).any()
             and _row_digest(cells) == inputs["cell_rows_sha256"], "Exported observation index is invalid.")
    receipts, offset = [], 0
    for record in records:
        _require(record["row_start"] == offset and record["row_stop"] == offset + record["n_cells"],
                 "Core rows overlap or leave a gap.")
        rows = cells.iloc[record["row_start"]:record["row_stop"]]
        _require(rows.core_number.eq(record["core_number"]).all()
                 and rows.alias.eq(record["alias"]).all()
                 and _row_digest(rows) == record["cell_rows_sha256"], "Core-to-cell order changed.")
        receipt = _verify_core(out, record, inputs)
        receipts.append({"core_number": record["core_number"],
            "receipt_sha256": file_hash(out / "embeddings" / f"core_{record['core_number']}" / "receipt.json"),
            "elapsed_seconds": receipt["elapsed_seconds"], "peak_allocated_bytes": receipt["peak_allocated_bytes"]})
        offset = record["row_stop"]
    _require(offset == EXPECTED_CELLS, "Cell coverage is incomplete.")
    pilot = _json(out / "diagnostics/extraction_pilot.json")
    _require(pilot == _json(out / "embeddings/core_23/receipt.json")
             and all(pilot["audit"].get(key) is True for key in (
                 "prediction_invariance_checked", "h0_equals_direct_encoder", "predictions_exact",
                 "ordinary_hL_exact", "hL_equals_full_node_embedding")),
             "Representative pilot invariance receipt is absent or inconsistent.")
    result = {"schema": SCHEMA, "verified": True, "verified_at": _now(),
        "n_cells": offset, "dimensions": inputs["dimensions"], "core_count": len(records),
        "inputs_sha256": inputs["inputs_sha256"], "cells_sha256": inputs["cells_sha256"],
        "checkpoint_sha256": inputs["checkpoint_sha256"], "checkpoint_epoch": inputs["checkpoint_epoch"],
        "all_observed": True, "dtype": "float32", "tf32": False, "cores": receipts}
    _write_json(out / "embeddings/extraction_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cores", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    result = run_worker(args.source_run_dir, args.output, args.device,
                        [int(number) for number in args.cores.split(",")], pilot=args.pilot)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
