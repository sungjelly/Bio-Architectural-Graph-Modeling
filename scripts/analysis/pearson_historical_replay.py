"""Post-hoc Pearson replay of the frozen SO2 continued-e300 checkpoint.

The caller owns the registered evaluation, task contract, resource admission,
and final publication. This helper only reads historical inputs and writes an
isolated staging directory. It never trains or changes an existing run bundle.
"""
from __future__ import annotations

import gc
import inspect
import json
import math
from pathlib import Path
import resource
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from spatial_benchmark.adjacency_ablation import sample_uniform_mask_numpy
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.myjju_genemae import MaskedRegressionAccumulator
from spatial_benchmark.paths import current_paths
from spatial_benchmark.pooled_relative_qkv_training import _tree_sha256
from spatial_benchmark.relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.so2_recurrent_hl_clustering import _load_core_batch
from spatial_benchmark.training import _autocast_context, set_deterministic_seed


SOURCE_RUN = "r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2"
REPLAY_ATOL = 2e-6
SCALES = ("standardized", "log1p")


def _read(path):
    return json.loads(Path(path).read_text())


def _finite_json(value):
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(_finite_json(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def _verify(path, expected, hashes):
    path = Path(path)
    digest = sha256_file(path)
    if digest != expected:
        raise ValueError(f"Historical source checksum mismatch: {path}")
    hashes[str(path)] = digest
    return path


def _metrics(accumulator, cells):
    values = accumulator.finalize()
    values["masked_rmse"] = math.sqrt(values["masked_mse"])
    values["n_cells"] = cells
    values["n_undefined_cells"] = cells - values["n_valid_cells"]
    values["n_undefined_genes"] = accumulator.num_genes - values["n_valid_genes"]
    return _finite_json(values)


def _differences(actual, expected, label):
    differences = {key: abs(float(actual[key]) - float(value))
                   for key, value in expected.items()}
    if any(not math.isfinite(v) or v > REPLAY_ATOL for v in differences.values()):
        raise ValueError(f"Historical replay mismatch ({label}): {differences}")
    return differences


def evaluate_historical(stage_dir, checkpoint_path):
    """Replay all fourteen original masks; return and save JSON-ready metrics.

    Standardized targets and log1p(raw counts) are both scored. Log predictions
    are inverse-standardized without clipping, preserving the native decoder.
    Cell-wise means exclude undefined correlations; exclusion counts are saved.
    Global metrics pool sufficient statistics, while equal-core metrics average
    the fourteen separately calculated values. These are fitted-cohort results.
    """
    started = time.monotonic()
    paths = current_paths()
    stage = Path(stage_dir)
    destination = stage / "historical_e300_metrics.json"
    receipt_dir = stage / "historical_e300"
    if destination.exists() or receipt_dir.exists():
        raise FileExistsError("Historical Pearson output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("Historical full inference requires the allocated CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    if (torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] <= 2*1024**3
            or shutil.disk_usage(paths.scratch_root).free <= 5*1024**3):
        raise RuntimeError("Historical replay resource admission failed")
    if float((torch.ones(4, device=device)*2).sum()) != 8.:
        raise RuntimeError("CUDA preflight operation failed")
    set_deterministic_seed(0, deterministic=True, warn_only=False)
    checkpoint = Path(checkpoint_path).resolve()
    source = checkpoint.parent.parent
    if source.name != SOURCE_RUN or checkpoint.name != "last.ckpt":
        raise ValueError("This replay is bound to the continued-e300 last checkpoint")

    hashes = {}
    archive_manifest_path = source / "provenance/artifact_checksums.json"
    archive_manifest = _read(archive_manifest_path)["files"]
    hashes[str(archive_manifest_path)] = sha256_file(archive_manifest_path)
    for relative in ("checkpoints/last.ckpt", "config.resolved.yaml",
                     "diagnostics/held_in_fit_metrics_by_core.json", "metrics/final.json"):
        _verify(source / relative, archive_manifest[relative]["sha256"], hashes)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["run_id"] != SOURCE_RUN or payload["model_seed"] != 0
            or payload["completed_global_epochs"] != 300
            or _tree_sha256(payload["model_state_dict"]) != payload["model_state_checksum"]):
        raise ValueError("Historical checkpoint identity mismatch")
    construction = payload["model_construction"]
    cls = ReceiverChunkedRelativeGeometryQKVGraphTransformer
    if construction["class"] != cls.__name__:
        raise ValueError("Unexpected historical architecture")
    model = cls(**{key: construction[key] for key in inspect.signature(cls).parameters})
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if sum(p.numel() for p in model.parameters()) != payload["parameter_count"]:
        raise ValueError("Historical parameter count mismatch")
    configuration = payload["resolved_config"]
    model.to(device).eval()
    del payload

    cohort = paths.data_root / "processed/so2_14core_relative_qkv_v1"
    graphs = paths.data_root / "processed/so2_14core_relative_qkv_graphs_v1"
    dataset = configuration["dataset"]
    _verify(cohort / "manifest.json", dataset["cohort_manifest_file_sha256"], hashes)
    _verify(graphs / "manifest.json", dataset["graph_manifest_file_sha256"], hashes)
    cm, gm = _read(cohort / "manifest.json"), _read(graphs / "manifest.json")
    _verify(cohort / "cohort_statistics.npz", cm["files"]["cohort_statistics.npz"], hashes)
    with np.load(cohort / "cohort_statistics.npz", allow_pickle=False) as stats:
        means, scales = stats["expression_mean"], stats["expression_scale"]
    if not ((scales > 0).all() and np.isfinite(means).all() and np.isfinite(scales).all()):
        raise ValueError("Invalid native normalization statistics")
    expected_by_alias = {r["alias"]: r for r in _read(source / "diagnostics/held_in_fit_metrics_by_core.json")}
    if set(expected_by_alias) != {f"SO2-C{k}" for k in range(15, 29)}:
        raise ValueError("Historical diagnostic core coverage changed")

    prior_report = paths.report_root / "analyses/so2_nonzero_metrics" / SOURCE_RUN / "v1"
    prior_manifest_path = prior_report / "manifest.json"
    prior_manifest = _read(prior_manifest_path)
    prior_success = _read(prior_report / "_SUCCESS")
    _verify(prior_manifest_path, prior_success["manifest_sha256"], hashes)
    source_files = {Path(__file__).resolve()}
    for item in (MaskedRegressionAccumulator, cls, _load_core_batch,
                 sample_uniform_mask_numpy, _autocast_context, _tree_sha256):
        source_files.add(Path(inspect.getfile(item)).resolve())
    for item in source_files:
        hashes[str(item)] = sha256_file(item)

    pooled = {scale: MaskedRegressionAccumulator(1000) for scale in SCALES}
    rows = []
    native_rows = []
    trainer = configuration["trainer"]
    torch.cuda.reset_peak_memory_stats(device)
    for core in range(15, 29):
        core_started = time.monotonic()
        alias = f"SO2-C{core}"
        expected = expected_by_alias[alias]
        cr = next(r for r in cm["cores"] if r["alias"] == alias)
        gr = next(r for r in gm["cores"] if r["alias"] == alias)
        relative = f"cores/{alias}.npz"
        _verify(cohort / relative, cm["files"][relative], hashes)
        for filename in ("edge_index.npy", "relative_geometry.npy"):
            _verify(graphs / "cores" / alias / filename, gr["files"][filename], hashes)
        record = dict(cr, directed_edges=gr["graph"]["qc"]["n_directed_edges"],
                      receiver_major_canonical_order=gr["graph"]["qc"]["receiver_major_canonical_order"])
        batch, _ = _load_core_batch(cohort_dir=cohort, graph_dir=graphs, record=record)
        realization = sample_uniform_mask_numpy(batch.n_nodes, batch.n_genes, seed=expected["mask_seed"])
        if (realization.checksum != expected["mask_checksum"]
                or int(realization.masked_gene_counts.sum()) != expected["n_masked_entries"]):
            raise ValueError(f"Original masking mismatch: {alias}")
        target = batch.target_expression.to(device)
        covariates = batch.node_covariates.to(device)
        mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(device=device, dtype=torch.bool)
        torch.cuda.synchronize(device)
        inference_started = time.monotonic()
        with torch.no_grad(), _autocast_context(enabled=bool(trainer["amp"]), device=device,
                                               dtype_name=str(trainer["amp_dtype"])):
            output = model(input_expression=target.masked_fill(mask, 0.), gene_mask=mask,
                           edge_index=batch.edge_index, relative_geometry=batch.relative_geometry,
                           node_covariates=covariates)
        prediction = output.prediction.float()
        if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
            raise ValueError(f"Invalid reconstructed values: {alias}")
        torch.cuda.synchronize(device)
        inference_seconds = time.monotonic() - inference_started
        pt, yt = prediction[mask], target[mask]
        residual = pt - yt
        native = {"masked_huber": float(F.huber_loss(pt, yt, delta=1., reduction="mean")),
                  "masked_mae": float(residual.abs().mean()),
                  "masked_mse": float(residual.square().mean()),
                  "masked_r2": float(1 - residual.square().sum() / (yt - yt.mean()).square().sum()),
                  "mean_y_true": float(yt.mean()), "mean_y_pred": float(pt.mean())}
        native_differences = _differences(native, {key: expected[key] for key in native}, alias)
        native_rows.append(native)
        predicted = prediction.cpu().numpy()
        with np.load(cohort / relative, allow_pickle=False) as raw:
            counts = np.array(raw["expression_counts"], copy=True)
        if counts.shape != predicted.shape or (counts < 0).any():
            raise ValueError("Raw counts do not align with model output")
        accumulators = {scale: MaskedRegressionAccumulator(1000) for scale in SCALES}
        true_z = batch.target_expression.numpy()
        for start in range(0, batch.n_nodes, 1024):
            stop = min(start + 1024, batch.n_nodes)
            selected = realization.mask[start:stop]
            true_log = np.log1p(counts[start:stop].astype(np.float64))
            np.testing.assert_allclose(true_z[start:stop], (true_log - means) / scales,
                                       atol=2e-5, rtol=2e-6)
            pred_z = predicted[start:stop].astype(np.float64)
            pred_log = pred_z * scales + means
            for scale, true, pred in (("standardized", true_z[start:stop], pred_z),
                                      ("log1p", true_log, pred_log)):
                accumulators[scale].update(true, pred, selected)
                pooled[scale].update(true, pred, selected)
        values = {scale: _metrics(accumulators[scale], batch.n_nodes) for scale in SCALES}
        if any(value["n_masked"] != expected["n_masked_entries"] for value in values.values()):
            raise ValueError(f"Historical metric mask coverage mismatch: {alias}")
        prior_relative = f"core_{core}.json"
        _verify(prior_report / prior_relative, prior_manifest["files"][prior_relative]["sha256"], hashes)
        prior = _read(prior_report / prior_relative)
        if prior["mask_checksum"] != realization.checksum or prior["checkpoint_sha256"] != hashes[str(checkpoint)]:
            raise ValueError("Prior nonzero evaluation identity mismatch")
        prior_all = prior["metrics"]["model"]["strata"]["all"]
        replay64 = {}
        for scale, suffix in (("standardized", "standardized"), ("log1p", "log1p")):
            replay64[scale] = _differences(values[scale], {
                "masked_mae": prior_all[f"mae_{suffix}"],
                "masked_mse": prior_all[f"mse_{suffix}"],
            }, f"{alias} {scale} prior float64 errors")
        row = {"core": alias, "n_cells": batch.n_nodes, "mask_seed": expected["mask_seed"],
               "mask_checksum": realization.checksum, "masked_entries": expected["n_masked_entries"],
               "metrics": values, "native_metric_replay": native,
               "native_absolute_differences": native_differences,
               "prior_float64_absolute_differences": replay64,
               "inference_seconds": inference_seconds, "duration_seconds": time.monotonic() - core_started}
        rows.append(row)
        _write_new(receipt_dir / f"core_{core}.json", row)
        print(json.dumps({"phase": "historical_pearson_core_completed", "core": alias,
                          "seconds": row["duration_seconds"], "cell_pearson_log1p": values["log1p"]["cell_pearson_mean"]}), flush=True)
        del target, covariates, mask, output, prediction, pt, yt, residual, predicted, counts, batch
        gc.collect()
        torch.cuda.empty_cache()

    total_cells = sum(row["n_cells"] for row in rows)
    if total_cells != 246063:
        raise ValueError("Historical all-core cell coverage changed")
    expected_aggregate = _read(source / "metrics/final.json")
    native_aggregate = {key: float(np.mean([row[key] for row in native_rows]))
                        for key in ("masked_huber", "masked_mae", "masked_mse", "masked_r2")}
    aggregate_differences = _differences(native_aggregate, {
        key: expected_aggregate[f"fit/uniform_per_cell/{key}"] for key in native_aggregate
    }, "equal-core native aggregate")
    equal_core = {}
    for scale in SCALES:
        keys = ("masked_mae", "masked_mse", "masked_rmse", "masked_huber", "masked_r2",
                "pooled_pearson", "cell_pearson_mean", "gene_pearson_mean", "gene_pearson_median")
        equal_core[scale] = {}
        for key in keys:
            finite = [row["metrics"][scale][key] for row in rows if row["metrics"][scale][key] is not None]
            equal_core[scale][key] = float(np.mean(finite)) if finite else None
    result = {"schema": "so2_historical_e300_pearson_v1", "source_run_id": SOURCE_RUN,
              "status": "completed", "verified": True,
              "checkpoint_sha256": hashes[str(checkpoint)], "epoch": 300, "seed": 0,
              "split": "fitted_cohort_all14", "mask_protocol": "original_simultaneous_uniform_per_cell",
              "target_scales": ["gene_standardized_log1p_counts", "log1p_raw_counts"],
              "log_predictions_clipped": False, "normalization_fit_scope": "all14_fitted_cores",
              "per_core": rows, "global": {scale: _metrics(pooled[scale], total_cells) for scale in SCALES},
              "equal_core": equal_core, "verification": {"passed": True, "atol": REPLAY_ATOL,
                  "native_equal_core_absolute_differences": aggregate_differences,
                  "all_original_masks_and_sources_verified": True},
              "source_hashes": hashes, "row_level_predictions_saved": False,
              "duration_seconds": time.monotonic() - started,
              "hardware": {"gpu": torch.cuda.get_device_name(device), "device": str(device),
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                  "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                  "amp": bool(trainer["amp"]), "amp_dtype": trainer["amp_dtype"]}}
    result = _finite_json(result)
    _write_new(destination, result)
    del model
    torch.cuda.empty_cache()
    return result
