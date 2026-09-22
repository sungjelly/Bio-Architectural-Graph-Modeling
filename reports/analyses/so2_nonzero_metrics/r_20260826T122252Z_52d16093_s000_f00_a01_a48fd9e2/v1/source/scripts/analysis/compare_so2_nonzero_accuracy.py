#!/usr/bin/env python3
"""Frozen-checkpoint SO2 nonzero evaluation; see cmp_20260905_so2_nonzero_accuracy."""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from spatial_benchmark.adjacency_ablation import sample_uniform_mask_numpy
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer
from spatial_benchmark.paths import current_paths
from spatial_benchmark.pooled_relative_qkv_training import _tree_sha256
from spatial_benchmark.registry import Registry
from spatial_benchmark.relative_qkv_graph_transformer import ReceiverChunkedRelativeGeometryQKVGraphTransformer, ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer
from spatial_benchmark.so2_nonzero_metrics import NonzeroMetricAccumulator
from spatial_benchmark.so2_recurrent_hl_clustering import _load_core_batch
from spatial_benchmark.training import _autocast_context, set_deterministic_seed

PATHS = current_paths()
CAMPAIGN = "cmp_20260905_so2_nonzero_accuracy"
PROTOCOL = "so2_nonzero_fixed_masks_v1"
RUNS = {
    "original": "r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6",
    "continued": "r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2",
    "recurrent": "r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf",
    "geometry": "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf",
}
DEVICES = dict(zip(RUNS, ("cuda:0", "cuda:1", "cuda:2", "cuda:3"), strict=True))
CLASSES = {cls.__name__: cls for cls in (ReceiverChunkedRelativeGeometryQKVGraphTransformer,
           ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer, ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer)}
SOURCE_NAMES = ("so2_nonzero_metrics.py", "geometry_modulated_relative_qkv_graph_transformer.py",
                "relative_qkv_graph_transformer.py", "models.py", "so2_recurrent_hl_clustering.py",
                "pooled_relative_qkv_training.py", "pooled_relative_qkv_training_v2.py", "masking.py",
                "adjacency_ablation.py", "training.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def locations(name):
    rid = RUNS[name]
    source = PATHS.artifact_root / "runs" / rid[2:6] / rid[6:8] / rid
    stage = PATHS.scratch_root / "active_runs" / rid / "posthoc_reports/so2_nonzero_metrics/v1"
    final = PATHS.report_root / "analyses/so2_nonzero_metrics" / rid / "v1"
    return rid, source, stage, final


def event(stage, phase, **fields):
    value = {"time": now(), "phase": phase, **fields}
    line = json.dumps(value, allow_nan=False)
    for rel in ("metrics/events.jsonl", "logs/stdout.log"):
        with (stage / rel).open("a") as f:
            f.write(line + "\n"); f.flush(); os.fsync(f.fileno())
    print(line, flush=True)


def verified_file(root, rel, expected):
    path = root / rel
    assert sha256_file(path) == expected, f"Source checksum mismatch: {path}"
    return path


def prepare(name):
    rid, source, stage, final = locations(name)
    registry = Registry(initialize=False)
    contract = PATHS.project_root / "experiments/campaigns" / CAMPAIGN / "README.md"
    if registry.get_campaign(CAMPAIGN) is None:
        registry.create_campaign(CAMPAIGN, name="SO2 nonzero accuracy on fixed masks",
            scientific_question="Do similar overall errors conceal different nonzero reconstruction accuracy?",
            status="pilot", config={"protocol": PROTOCOL, "source_run_ids": list(RUNS.values()),
                                    "contract_sha256": sha256_file(contract), "expected_seeds": [0],
                                    "expected_folds": [], "analysis_design": "exploratory_posthoc"})
    if final.exists():
        return verify(final)
    assert not (stage / "config.resolved.json").exists(), "Evaluation already prepared; use pilot/full"
    for directory in ("source", "provenance", "metrics", "logs"):
        (stage / directory).mkdir(parents=True, exist_ok=True)
    manifest = read(source / "provenance/artifact_checksums.json")["files"]
    checkpoint = verified_file(source, "checkpoints/last.ckpt", manifest["checkpoints/last.ckpt"]["sha256"])
    verified_file(source, "diagnostics/held_in_fit_metrics_by_core.json", manifest["diagnostics/held_in_fit_metrics_by_core.json"]["sha256"])
    verified_file(source, "config.resolved.yaml", manifest["config.resolved.yaml"]["sha256"])
    source_run = registry.get_run(rid)
    assert source_run is not None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["run_id"] == rid and payload["model_seed"] == 0
    assert _tree_sha256(payload["model_state_dict"]) == payload["model_state_checksum"]
    evaluation_id = "ev_so2_nonzero_v1_" + rid
    source_paths = [Path(__file__).resolve(), *[PATHS.project_root / "src/spatial_benchmark" / n for n in SOURCE_NAMES]]
    hashes = {}
    for path in source_paths:
        rel = path.relative_to(PATHS.project_root)
        destination = stage / "source" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(rel)] = sha256_file(path)
    shutil.copyfile(contract, stage / "provenance/frozen_task_contract.md")
    cfg = {"protocol": PROTOCOL, "campaign_id": CAMPAIGN, "evaluation_id": evaluation_id,
           "model_name": name, "source_run_id": rid, "source_status": source_run["status"],
           "source_epoch": payload["completed_global_epochs"], "checkpoint_sha256": sha256_file(checkpoint),
           "parameter_count": payload["parameter_count"], "model_state_checksum": payload["model_state_checksum"],
           "source_resolved_config": payload["resolved_config"], "device": DEVICES[name],
           "classification": {"lifecycle_stage": "exploratory_screen", "study_axis": "nonzero_reconstruction",
                              "scientific_variant": name, "model_seed": 0, "fold_known": False},
           "code_sha256": hashes, "contract_sha256": sha256_file(contract),
           "metric_replay_atol": 2e-6, "pilot_cores": [21, 23], "all_cores": list(range(15, 29)),
           "mask_policy": "exact_source_held_in_diagnostic", "count_detection_threshold": .5,
           "primary_metric": "equal_core_positive_mse_standardized", "new_training": False}
    write(stage / "config.resolved.json", cfg)
    write(stage / "provenance/environment.json", {"python": platform.python_version(), "platform": platform.platform(),
          "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__, "cwd": str(Path.cwd()),
          "gpu_schedule": DEVICES, "cpu_threads_per_worker": 4, "prepared_at": now()})
    for rel, cmd in (("provenance/dependencies.txt", [sys.executable, "-m", "pip", "freeze"]),
                     ("provenance/git_commit.txt", ["git", "rev-parse", "HEAD"]),
                     ("provenance/git_status.txt", ["git", "status", "--short"]),
                     ("provenance/uncommitted_changes.patch", ["git", "diff", "HEAD"]),
                     ("provenance/nvidia-smi.txt", ["nvidia-smi"]),
                     ("provenance/nvidia-topology.txt", ["nvidia-smi", "topo", "-m"])):
        (stage / rel).write_text(subprocess.check_output(cmd, text=True))
    (stage / "logs/stderr.log").touch()
    registry.create_evaluation(evaluation_id, run_id=rid, checkpoint_name="last",
        dataset_id=payload["resolved_config"]["dataset"]["dataset_id"],
        split_id=payload["resolved_config"]["dataset"]["split_id"], status="running", metrics=cfg, artifact_path=stage)
    event(stage, "prepared", model=name, source_run_id=rid, evaluation_id=evaluation_id)


def verify(root):
    manifest = read(root / "manifest.json")
    for rel, expected in manifest["files"].items():
        assert sha256_file(root / rel) == expected["sha256"], rel
    assert read(root / "_SUCCESS")["manifest_sha256"] == sha256_file(root / "manifest.json")
    assert len(manifest["cores"]) == 14
    return manifest


def update_registry(cfg, root, metrics, status):
    registry = Registry(initialize=False)
    with registry.transaction(immediate=True) as connection:
        connection.execute("UPDATE evaluations SET status=?,metrics_json=?,artifact_path=?,finished_at=? WHERE evaluation_id=? AND run_id=?",
            (status, json.dumps(metrics, allow_nan=False), str(root), now(), cfg["evaluation_id"], cfg["source_run_id"]))
    if status == "completed":
        registry.record_artifact(cfg["source_run_id"], evaluation_id=cfg["evaluation_id"], kind="posthoc_nonzero_metrics_manifest_v1",
            path=root / "manifest.json", sha256=sha256_file(root / "manifest.json"), size_bytes=(root / "manifest.json").stat().st_size, status="present")


def evaluate_core(name, core, model, cfg, cm, gm, means, scales, stage, source):
    began = time.monotonic()
    alias = f"SO2-C{core}"
    cohort = PATHS.data_root / "processed/so2_14core_relative_qkv_v1"
    graphs = PATHS.data_root / "processed/so2_14core_relative_qkv_graphs_v1"
    cr = next(r for r in cm["cores"] if r["alias"] == alias)
    gr = next(r for r in gm["cores"] if r["alias"] == alias)
    source_files = {}
    rel = f"cores/{alias}.npz"
    verified_file(cohort, rel, cm["files"][rel]); source_files["cohort/" + rel] = cm["files"][rel]
    for fn in ("edge_index.npy", "relative_geometry.npy"):
        verified_file(graphs / "cores" / alias, fn, gr["files"][fn]); source_files[f"graph/cores/{alias}/{fn}"] = gr["files"][fn]
    record = dict(cr, directed_edges=gr["graph"]["qc"]["n_directed_edges"],
                  receiver_major_canonical_order=gr["graph"]["qc"]["receiver_major_canonical_order"])
    batch, _ = _load_core_batch(cohort_dir=cohort, graph_dir=graphs, record=record)
    expected = next(r for r in read(source / "diagnostics/held_in_fit_metrics_by_core.json") if r["alias"] == alias)
    realization = sample_uniform_mask_numpy(batch.n_nodes, batch.n_genes, seed=expected["mask_seed"])
    assert realization.checksum == expected["mask_checksum"]
    assert int(realization.masked_gene_counts.sum()) == expected["n_masked_entries"]
    device = torch.device(cfg["device"])
    target = batch.target_expression.to(device)
    covariates = batch.node_covariates.to(device)
    mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(device=device, dtype=torch.bool)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    infer_start = time.monotonic()
    trainer = cfg["source_resolved_config"]["trainer"]
    with torch.no_grad(), _autocast_context(enabled=bool(trainer["amp"]), device=device, dtype_name=str(trainer["amp_dtype"])):
        output = model(input_expression=target.masked_fill(mask, 0.), gene_mask=mask,
                       edge_index=batch.edge_index, relative_geometry=batch.relative_geometry, node_covariates=covariates)
    prediction = output.prediction.float()
    assert prediction.shape == target.shape and torch.isfinite(prediction).all()
    torch.cuda.synchronize(device)
    inference_seconds = time.monotonic() - infer_start
    pt, yt = prediction[mask], target[mask]
    residual = pt - yt
    replay = {"masked_huber": float(F.huber_loss(pt, yt, delta=1., reduction="mean")),
              "masked_mae": float(residual.abs().mean()), "masked_mse": float(residual.square().mean()),
              "masked_r2": float(1 - residual.square().sum() / (yt - yt.mean()).square().sum()),
              "mean_y_true": float(yt.mean()), "mean_y_pred": float(pt.mean())}
    replay_differences = {k: abs(v - expected[k]) for k, v in replay.items()}
    assert max(replay_differences.values()) <= cfg["metric_replay_atol"], (alias, replay_differences)
    predicted = prediction.cpu().numpy()
    with np.load(cohort / rel, allow_pickle=False) as raw:
        counts = np.array(raw["expression_counts"], copy=True)
    assert counts.shape == predicted.shape and (counts >= 0).all()
    accumulators = {key: NonzeroMetricAccumulator() for key in ("model", "zero_count", "gene_mean")}
    true_z = batch.target_expression.numpy()
    mask_np = realization.mask
    for start in range(0, batch.n_nodes, 1024):
        stop = min(start + 1024, batch.n_nodes)
        selected = mask_np[start:stop]
        raw = counts[start:stop]
        z = true_z[start:stop]
        reconstructed = np.log1p(raw.astype(np.float64))
        np.testing.assert_allclose(z, (reconstructed - means) / scales, atol=2e-5, rtol=2e-6)
        pred_z = predicted[start:stop].astype(np.float64)
        pred_log = pred_z * scales + means
        common = {"true_counts": raw[selected], "true_standardized": z[selected]}
        accumulators["model"].update(**common, pred_standardized=pred_z[selected], pred_log1p=pred_log[selected])
        zero_z = np.broadcast_to(-means / scales, raw.shape)
        gene_log = np.broadcast_to(means, raw.shape)
        accumulators["zero_count"].update(**common, pred_standardized=zero_z[selected], pred_log1p=np.zeros(int(selected.sum())))
        accumulators["gene_mean"].update(**common, pred_standardized=np.zeros(int(selected.sum())), pred_log1p=gene_log[selected])
    metrics = {key: accumulator.result() for key, accumulator in accumulators.items()}
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    assert peak < 21, peak
    receipt = {"schema": PROTOCOL, "model_name": name, "source_run_id": cfg["source_run_id"],
               "source_status": cfg["source_status"], "source_epoch": cfg["source_epoch"], "evaluation_id": cfg["evaluation_id"],
               "checkpoint_sha256": cfg["checkpoint_sha256"], "core_alias": alias, "n_cells": batch.n_nodes,
               "mask_seed": expected["mask_seed"], "mask_checksum": expected["mask_checksum"],
               "n_masked_entries": expected["n_masked_entries"], "source_files": source_files,
               "metrics": metrics, "original_all_entry_metrics_replay": replay,
               "replay_absolute_differences": replay_differences, "replay_verified": True,
               "runtime_seconds": time.monotonic() - began, "inference_seconds": inference_seconds,
               "peak_vram_gib": peak, "peak_reserved_vram_gib": torch.cuda.max_memory_reserved(device) / 2**30,
               "peak_host_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
               "device": cfg["device"], "gpu_model": torch.cuda.get_device_name(device),
               "amp_enabled": bool(trainer["amp"]), "amp_dtype_config": trainer["amp_dtype"], "completed_at": now()}
    write(stage / f"core_{core}.json", receipt)
    event(stage, "core_completed", model=name, core=core, runtime_seconds=receipt["runtime_seconds"], peak_vram_gib=peak)
    Registry(initialize=False).record_metric(cfg["source_run_id"], "fit/nonzero/mse_standardized",
        metrics["model"]["strata"]["positive"]["mse_standardized"], step=core, split="fit", evaluation_id=cfg["evaluation_id"])
    del output, prediction, target, covariates, mask, pt, yt, residual, batch, counts, predicted
    gc.collect(); torch.cuda.empty_cache()
    return receipt


def execute(name, phase):
    rid, source, stage, final = locations(name)
    if phase == "prepare":
        return prepare(name)
    if phase == "verify" or final.exists():
        verify(final)
        print(json.dumps({"model": name, "status": "verified", "path": str(final)}))
        return
    cfg = read(stage / "config.resolved.json")
    try:
        for rel, expected in cfg["code_sha256"].items():
            assert sha256_file(PATHS.project_root / rel) == expected, f"Evaluation source changed: {rel}"
        verified_file(source, "checkpoints/last.ckpt", cfg["checkpoint_sha256"])
        device = torch.device(cfg["device"])
        torch.cuda.set_device(device); torch.set_num_threads(4)
        set_deterministic_seed(0, deterministic=True, warn_only=False)
        payload = torch.load(source / "checkpoints/last.ckpt", map_location="cpu", weights_only=True)
        constructor = payload["model_construction"]
        cls = CLASSES[constructor["class"]]
        model = cls(**{key: constructor[key] for key in inspect.signature(cls).parameters})
        model.load_state_dict(payload["model_state_dict"], strict=True)
        assert _tree_sha256(model.state_dict()) == cfg["model_state_checksum"]
        assert sum(p.numel() for p in model.parameters()) == cfg["parameter_count"]
        model.to(device).eval()
        del payload
        cohort = PATHS.data_root / "processed/so2_14core_relative_qkv_v1"
        graphs = PATHS.data_root / "processed/so2_14core_relative_qkv_graphs_v1"
        dataset = cfg["source_resolved_config"]["dataset"]
        verified_file(cohort, "manifest.json", dataset["cohort_manifest_file_sha256"])
        verified_file(graphs, "manifest.json", dataset["graph_manifest_file_sha256"])
        cm, gm = read(cohort / "manifest.json"), read(graphs / "manifest.json")
        verified_file(cohort, "cohort_statistics.npz", cm["files"]["cohort_statistics.npz"])
        with np.load(cohort / "cohort_statistics.npz", allow_pickle=False) as stats:
            means, scales = stats["expression_mean"], stats["expression_scale"]
        assert (scales > 0).all() and np.isfinite(means).all() and np.isfinite(scales).all()
        if phase == "full":
            for core in cfg["pilot_cores"]:
                receipt = read(stage / f"core_{core}.json")
                assert receipt["replay_verified"] and receipt["peak_vram_gib"] < 21
        event(stage, phase + "_started", model=name, command=sys.argv)
        cores = cfg["pilot_cores"] if phase == "pilot" else cfg["all_cores"]
        for core in cores:
            if (stage / f"core_{core}.json").exists():
                receipt = read(stage / f"core_{core}.json")
                assert receipt["checkpoint_sha256"] == cfg["checkpoint_sha256"] and receipt["replay_verified"]
                continue
            evaluate_core(name, core, model, cfg, cm, gm, means, scales, stage, source)
        if phase == "pilot":
            event(stage, "pilot_passed", model=name, cores=cores)
            return
        receipts = [read(stage / f"core_{c}.json") for c in cfg["all_cores"]]
        assert sum(r["n_cells"] for r in receipts) == 246063
        primary = float(np.mean([r["metrics"]["model"]["strata"]["positive"]["mse_standardized"] for r in receipts]))
        summary = {"model_name": name, "source_run_id": rid, "evaluation_id": cfg["evaluation_id"],
                   "source_status": cfg["source_status"], "source_epoch": cfg["source_epoch"],
                   "equal_core_positive_mse_standardized": primary, "cores": 14, "total_cells": 246063,
                   "runtime_seconds": sum(r["runtime_seconds"] for r in receipts),
                   "peak_vram_gib": max(r["peak_vram_gib"] for r in receipts), "status": "completed"}
        write(stage / "summary.json", summary)
        event(stage, "evaluation_complete", **summary)
        files = {str(f.relative_to(stage)): {"sha256": sha256_file(f), "size_bytes": f.stat().st_size}
                 for f in sorted(stage.rglob("*")) if f.is_file() and f.name not in ("manifest.json", "_SUCCESS")}
        write(stage / "manifest.json", {"schema": PROTOCOL, "evaluation_id": cfg["evaluation_id"],
              "source_run_id": rid, "checkpoint_sha256": cfg["checkpoint_sha256"], "cores": cfg["all_cores"], "files": files})
        write(stage / "_SUCCESS", {"status": "success", "manifest_sha256": sha256_file(stage / "manifest.json"), "completed_at": now()})
        verify(stage)
        final.parent.mkdir(parents=True, exist_ok=True)
        assert not final.exists()
        os.rename(stage, final)
        verify(final)
        update_registry(cfg, final, summary, "completed")
        print(json.dumps({"model": name, "status": "published", "path": str(final)}), flush=True)
    except Exception:
        error = traceback.format_exc()
        if stage.exists():
            with (stage / "logs/stderr.log").open("a") as f: f.write(error)
            write(stage / "_FAILED", {"failed_at": now(), "phase": phase, "traceback": error})
            update_registry(cfg, stage, {"phase": phase, "error": error}, "failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=RUNS)
    parser.add_argument("--phase", required=True, choices=("prepare", "pilot", "full", "verify"))
    args = parser.parse_args()
    execute(args.model, args.phase)


if __name__ == "__main__":
    main()
