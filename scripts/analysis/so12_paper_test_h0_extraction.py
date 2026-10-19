"""Extract trained pre-graph h0 while reproducing the published test hL exactly.

The frozen training and hL analysis modules are reused without modifications.
The caller owns the managed process, GPU locks, registry and publication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

import so12_paper_test_extraction as base
import run_so12_paper_test_leiden as hl_controller
from spatial_benchmark.paths import ProjectPaths, current_paths

SCHEMA = "so12_paper_test_h0_extraction_v1"
REPRESENTATION = "trained_node_encoder_output_before_graph_blocks"
SOURCE_RUN_ID = base.SOURCE_RUN_ID
REFERENCE_EVALUATION_ID = "ev_so12_paper_test_hL_s0_20261017"
SELECTED_EPOCH = 194


def verify_source(run_dir: Path, paths: ProjectPaths | None = None) -> dict[str, Any]:
    source = base.verify_source(run_dir, paths)
    base._require(source["checkpoint_epoch"] == SELECTED_EPOCH, "Selected checkpoint epoch changed.")
    return source


def analysis_source_binding() -> dict[str, str]:
    return {"scripts/analysis/" + path.name: base.file_hash(path) for path in
            (Path(__file__), Path(base.__file__), Path(hl_controller.__file__))}


def _reference_evaluation(paths: ProjectPaths) -> dict[str, Any] | None:
    registry = base.Registry(paths.state_root / "tracking/bagm.sqlite3")
    with registry.connect() as connection:
        row = connection.execute("SELECT * FROM evaluations WHERE evaluation_id=?",
                                 (REFERENCE_EVALUATION_ID,)).fetchone()
    return dict(row) if row else None


def verify_reference(reference_hl_dir: Path, source: Mapping[str, Any],
                     paths: ProjectPaths | None = None
                     ) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Validate published hL before opening its arrays or loading any raw test data."""
    paths = paths or current_paths()
    reference = Path(reference_hl_dir).resolve()
    expected = (paths.report_root / "analyses/so12_paper_test_leiden" / REFERENCE_EVALUATION_ID).resolve()
    base._require(reference == expected, "Use the exact published hL reference directory.")
    row = _reference_evaluation(paths)
    base._require(row is not None and row["status"] == "completed"
                  and row["run_id"] == SOURCE_RUN_ID
                  and Path(row["artifact_path"]).resolve() == reference,
                  "The reference hL evaluation is not completed at its canonical path.")
    config = base._read(reference / "config.resolved.json")
    published = hl_controller.verify_published(reference, config)
    for relative, expected_hash in config["analysis_source_sha256"].items():
        path = Path(relative)
        base._require(not path.is_absolute() and ".." not in path.parts
                      and base.file_hash(paths.project_root / path) == expected_hash,
                      "Published hL source code differs from its frozen binding.")
    manifest = base._read(reference / "extraction_manifest.json")
    base._require(published["evaluation_id"] == REFERENCE_EVALUATION_ID
                  and manifest["source_run_id"] == SOURCE_RUN_ID
                  and manifest.get("representation") == "fourth_graph_block_output_before_decoder"
                  and manifest.get("hL_path") == "embeddings/hL.npy"
                  and manifest.get("status") == "complete"
                  and manifest.get("passed") is True and manifest.get("verified") is True
                  and manifest.get("all_observed") is True and manifest.get("test_only") is True
                  and manifest.get("test_core_numbers") == [15, 23]
                  and manifest.get("n_cells") == base.N_CELLS
                  and manifest.get("dimensions") == base.DIMENSIONS,
                  "Reference is not the complete all-observed test hL extraction.")
    for key in ("checkpoint_sha256", "preprocessing_fingerprint", "fitted_parameters_sha256",
                "source_identity_sha256"):
        base._require(manifest[key] == source[key], f"Reference {key} differs from the selected source.")
    base._require(manifest["source_code_sha256"] == source["binding"]["source_sha256"],
                  "Reference training-code binding changed.")
    reference_source = base._read(reference / "source_verified.json")
    base._require(reference_source["checkpoint_epoch"] == source["checkpoint_epoch"] == SELECTED_EPOCH,
                  "h0 must use the same selected epoch-194 checkpoint as published hL.")
    for key in ("metadata_statistics_sha256", "source_manifest_sha256", "dataset_fingerprint",
                "split_fingerprint", "source_identity_sha256"):
        base._require(reference_source[key] == source[key], f"Reference frozen {key} changed.")
    cells_path, embeddings_path = reference / "cells.csv.gz", reference / "embeddings/hL.npy"
    base._require(base.file_hash(cells_path) == manifest["cells_sha256"]
                  and base.file_hash(embeddings_path) == manifest["embedding_sha256"],
                  "Reference cells or hidden-state checksum changed.")
    cells = pd.read_csv(cells_path, float_precision="round_trip")
    expected_cores = np.concatenate([np.full(count, core) for core, count in base.CORE_CELLS.items()])
    base._require(list(cells.columns) == list(base.CELL_COLUMNS)
                  and len(cells) == base.N_CELLS
                  and np.array_equal(cells.row_index.to_numpy(), np.arange(base.N_CELLS))
                  and np.array_equal(cells.core_number.to_numpy(), expected_cores)
                  and cells.slide.eq(2).all() and cells.split.eq("test").all()
                  and not cells.duplicated(["slide", "fov", "cell_ID"]).any()
                  and base._row_digest(cells) == manifest["cell_rows_sha256"],
                  "Reference cell identities/order differ from canonical test coverage.")
    hL = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    base._require(hL.shape == (base.N_CELLS, base.DIMENSIONS) and hL.dtype == np.float32
                  and np.isfinite(hL).all() and np.all(np.linalg.norm(hL, axis=1) > 0),
                  "Reference hL matrix is invalid.")
    info = {"evaluation_id": REFERENCE_EVALUATION_ID, "path": str(reference),
            "manifest_sha256": base.file_hash(reference / "manifest.json"),
            "extraction_manifest_sha256": base.file_hash(reference / "extraction_manifest.json"),
            "config_sha256": base.file_hash(reference / "config.resolved.json"),
            "cells_sha256": manifest["cells_sha256"],
            "cell_rows_sha256": manifest["cell_rows_sha256"],
            "embedding_sha256": manifest["embedding_sha256"],
            "checkpoint_sha256": source["checkpoint_sha256"], "checkpoint_epoch": SELECTED_EPOCH,
            "preprocessing_fingerprint": source["preprocessing_fingerprint"],
            "fitted_parameters_sha256": source["fitted_parameters_sha256"],
            "verified": True}
    return info, cells, hL


def extract_core_array(model: Any, core: Any, normalization: Mapping[str, torch.Tensor],
                       device: torch.device, reference_hL: np.ndarray
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse audited full forwards; additionally audit h0 and the published hL."""
    expected = (len(core.raw_counts), model.hidden_dim)
    base._require(reference_hL.shape == expected and reference_hL.dtype == np.float32
                  and np.isfinite(reference_hL).all(), "Reference hL slice is invalid.")
    captured = []
    def capture_encoder(module, args, output):
        captured.append(output.detach().clone())
    handle = model.encoder.register_forward_hook(capture_encoder)
    try:
        hL, audit = base.extract_core_array(model, core, normalization, device, verify_invariance=True)
    finally:
        handle.remove()
    base._require(len(captured) == 2 and torch.equal(captured[0], captured[1]),
                  "Intermediate and ordinary forwards disagree about the trained encoder output.")
    h0 = captured[0]
    base._require(h0.shape == expected and h0.dtype == torch.float32
                  and bool(torch.isfinite(h0).all())
                  and bool((torch.linalg.vector_norm(h0, dim=1) > 0).all()),
                  "h0 contains invalid or zero hidden states.")
    raw = torch.tensor(np.asarray(core.raw_counts), dtype=torch.int32, device=device)
    mask = torch.zeros_like(raw, dtype=torch.bool)
    covariates = torch.tensor(np.asarray(core.node_covariates), dtype=torch.float32, device=device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        visible = base.visible_count_pearson_input(raw, mask, **normalization)
        direct = model.encoder(input_expression=visible, gene_mask=mask, node_covariates=covariates)
        base._require(torch.equal(h0, direct), "h0 differs from the direct trained node encoder.")
    base._require(np.array_equal(hL, reference_hL),
                  "New full forward does not reproduce published hL exactly for every cell.")
    audit.update(h0_equals_direct_encoder=True, published_hL_exact=True,
                 maximum_absolute_reference_hL_difference=0.0,
                 representation_name="h0", representation=REPRESENTATION)
    return h0.cpu().numpy().copy(), audit


def extract(source_run_dir: Path, output: Path, device: str = "cuda:0",
            paths: ProjectPaths | None = None, reference_hl_dir: Path | None = None
            ) -> dict[str, Any]:
    """Extract only h0, retaining source and published-reference identity receipts."""
    paths = paths or current_paths()
    destination = Path(output).resolve()
    active = (paths.scratch_root / "active_runs").resolve()
    base._require(destination.is_relative_to(active) and destination != active,
                  "Output must belong to a separate active analysis run.")
    source = verify_source(source_run_dir, paths)
    reference_hl_dir = reference_hl_dir or (paths.report_root / "analyses/so12_paper_test_leiden" / REFERENCE_EVALUATION_ID)
    reference, reference_cells, reference_hL = verify_reference(reference_hl_dir, source, paths)
    execution = base.configure_deterministic_inference()
    destination.mkdir(parents=True, exist_ok=True)
    base._require(shutil.disk_usage(destination).free >= 5.5 * 1024**3,
                  "h0 analysis would violate the 5 GiB reserve plus 0.5 GiB budget.")
    base._require(not (destination / "embeddings/h0.npy").exists()
                  and not (destination / "extraction_manifest.json").exists(),
                  "Refusing to overwrite existing h0 extraction.")
    started = time.monotonic()
    dataset = base.load_so12_paper_data(source["config"], paths, roles=("test",),
                                      verify_checksums=True, load_graphs=True)
    base._check_test_dataset(dataset)
    for key in ("dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint",
                "fitted_parameters_sha256", "source_manifest_sha256"):
        base._require(dataset.provenance[key] == source[key], f"Loaded test {key} changed.")
    cells = base.prepare_cells(dataset, destination, paths)
    base._require(base.file_hash(destination / "cells.csv.gz") == reference["cells_sha256"]
                  and base._row_digest(cells) == reference["cell_rows_sha256"]
                  and cells.equals(reference_cells),
                  "New h0 rows do not exactly match the published hL cell order and coordinates.")
    base._write(destination / "source_verified.json", {"schema": "so12_paper_test_h0_source_verification_v1",
                                                       "training_source": source, "reference_hl": reference})
    base._write(destination / "test_data_provenance.json", dataset.provenance)
    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        torch.cuda.reset_peak_memory_stats(target_device)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = base.build_model(source["config"], genes=len(dataset.genes),
                             covariates=dataset.cores[0].node_covariates.shape[1]).to(target_device).float().eval()
    checkpoint_path = Path(source["source_run_dir"]) / source["checkpoint_relative_path"]
    base._require(base.file_hash(checkpoint_path) == source["checkpoint_sha256"], "Checkpoint changed after verification.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    normalization = base.normalization_tensors(dataset, target_device)
    embeddings = destination / "embeddings"
    embeddings.mkdir(exist_ok=True)
    combined = np.lib.format.open_memmap(embeddings / "h0.npy", mode="w+", dtype=np.float32,
                                         shape=(base.N_CELLS, base.DIMENSIONS))
    receipts = []
    # Always run complete core 23 first as the representative real-data pilot.
    for core in sorted(dataset.cores, key=lambda core: (core.alias != "SO2-C23", core.alias)):
        core_started = time.monotonic()
        rows = cells.loc[cells.alias == core.alias]
        start, stop = int(rows.row_index.iloc[0]), int(rows.row_index.iloc[-1]) + 1
        array, audit = extract_core_array(model, core, normalization, target_device, reference_hL[start:stop])
        base._require(array.shape == (stop - start, base.DIMENSIONS), "h0 row coverage changed.")
        combined[start:stop] = array
        combined.flush()
        receipt = {"schema": SCHEMA, "representation_name": "h0", "representation": REPRESENTATION,
                   "core_number": int(core.alias[-2:]), "alias": core.alias,
                   "n_cells": core.n_cells, "row_start": start, "row_stop": stop,
                   "cell_rows_sha256": base._row_digest(rows), "dimensions": base.DIMENSIONS,
                   "embedding_slice_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                   "checkpoint_sha256": source["checkpoint_sha256"],
                   "preprocessing_fingerprint": source["preprocessing_fingerprint"],
                   "reference_hl_manifest_sha256": reference["manifest_sha256"],
                   "reference_hl_slice_sha256": hashlib.sha256(reference_hL[start:stop].tobytes(order="C")).hexdigest(),
                   "elapsed_seconds": time.monotonic() - core_started,
                   "edges": int(core.edge_index.shape[1]), "execution": execution, "audit": audit}
        name = f"embeddings/core_{int(core.alias[-2:])}_receipt.json"
        base._write(destination / name, receipt)
        if core.alias == "SO2-C23":
            base._write(destination / "diagnostics/extraction_pilot.json", {**receipt, "passed": True})
        receipts.append({**receipt, "receipt_path": name, "receipt_sha256": base.file_hash(destination / name)})
        del array
    base._require(np.isfinite(combined).all() and np.all(np.linalg.norm(combined, axis=1) > 0),
                  "Final h0 coverage is nonfinite or zero.")
    del combined
    base._require(base.file_hash(checkpoint_path) == source["checkpoint_sha256"]
                  and base.training_source_binding() == source["binding"]["source_sha256"],
                  "Frozen checkpoint or training source changed during extraction.")
    base._require(base.file_hash(Path(reference["path"]) / "manifest.json") == reference["manifest_sha256"]
                  and base.file_hash(Path(reference["path"]) / "embeddings/hL.npy") == reference["embedding_sha256"],
                  "Published hL reference changed during extraction.")
    result = {"schema": SCHEMA, "status": "complete", "passed": True, "verified": True,
              "source_run_id": SOURCE_RUN_ID, "created_at": datetime.now(timezone.utc).isoformat(),
              "n_cells": base.N_CELLS, "dimensions": base.DIMENSIONS,
              "core_numbers": list(base.CORE_CELLS), "test_core_numbers": list(base.CORE_CELLS),
              "cells_per_core": {str(core): count for core, count in base.CORE_CELLS.items()},
              "all_observed": True, "representation_name": "h0", "representation": REPRESENTATION,
              "checkpoint_sha256": source["checkpoint_sha256"], "checkpoint_epoch": SELECTED_EPOCH,
              "preprocessing_fingerprint": source["preprocessing_fingerprint"],
              "fitted_parameters_sha256": source["fitted_parameters_sha256"],
              "source_code_sha256": source["binding"]["source_sha256"],
              "source_identity_sha256": source["source_identity_sha256"],
              "cells_path": "cells.csv.gz", "h0_path": "embeddings/h0.npy",
              "cells_sha256": base.file_hash(destination / "cells.csv.gz"),
              "cell_rows_sha256": base._row_digest(cells),
              "h0_sha256": base.file_hash(embeddings / "h0.npy"),
              "cores": sorted(receipts, key=lambda receipt: receipt["core_number"]),
              "reference_hl": reference,
              "audit": {"h0_equals_direct_encoder": True, "published_hL_exact": True,
                        "predictions_exact": True},
              "analysis_source_sha256": analysis_source_binding(), "execution": execution,
              "elapsed_seconds": time.monotonic() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(target_device) if target_device.type == "cuda" else 0,
              "test_only": True, "normalization_refit": False, "source_modified": False}
    result["embedding_sha256"] = result["h0_sha256"]
    base._write(destination / "extraction_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reference-hl-dir", type=Path, required=True)
    args = parser.parse_args()
    result = extract(args.source_run_dir, args.output, args.device, reference_hl_dir=args.reference_hl_dir)
    print({"n_cells": result["n_cells"], "h0_path": str(args.output / result["h0_path"]), "passed": result["passed"]})


if __name__ == "__main__":
    main()
