#!/usr/bin/env python3
"""CPU-only exploratory SO2 constant-loss and observed-neighbor diagnostics."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import argparse
import csv
import json
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
import scipy
from scipy import sparse
from spatial_benchmark.adjacency_ablation import sample_uniform_mask_numpy
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.so2_diagnostic_baselines import huber_constant, local_adjacency, observed_neighbor_mean
from spatial_benchmark.so2_nonzero_metrics import NonzeroMetricAccumulator

P = current_paths()
CAMPAIGN = "cmp_20260906_so2_failure_diagnosis"
RID = "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf"
EID = "ev_so2_diagnostic_baselines_v1_" + RID
STAGE = P.scratch_root / "active_runs" / RID / "posthoc_reports/so2_failure_diagnosis/v1"
FINAL = P.report_root / "analyses/so2_failure_diagnosis/baselines/v1"
PRIOR = P.report_root / "analyses/so2_nonzero_metrics"
COHORT = P.data_root / "processed/so2_14core_relative_qkv_v1"
GRAPHS = P.data_root / "processed/so2_14core_relative_qkv_graphs_v1"
CONTRACT = P.project_root / "experiments/campaigns" / CAMPAIGN / "README.md"
NAMES = ("gene_mean", "zero_count", "huber_constant", "local_log_mean", "local_count_mean", "full_graph_log_mean")
SOURCES = ("scripts/analysis/diagnose_so2_baselines.py", "src/spatial_benchmark/so2_diagnostic_baselines.py",
           "src/spatial_benchmark/so2_nonzero_metrics.py", "src/spatial_benchmark/adjacency_ablation.py")

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)

def check(path, checksum):
    assert sha256_file(path) == checksum, f"Checksum mismatch: {path}"

def event(phase, **fields):
    record = {"time": now(), "phase": phase, **fields}
    with (STAGE / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())
    print(json.dumps(record), flush=True)

def table(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def verify(root):
    manifest = read(root / "manifest.json")
    for rel, item in manifest["files"].items():
        check(root / rel, item["sha256"])
    assert read(root / "_SUCCESS")["manifest_sha256"] == sha256_file(root / "manifest.json")
    assert len(manifest["cores"]) == 14
    print(json.dumps({"verified": str(root), "files": len(manifest["files"])}))

def register_status(status, root, metrics):
    reg = Registry(initialize=False)
    with reg.transaction(immediate=True) as c:
        c.execute("UPDATE evaluations SET status=?,artifact_path=?,metrics_json=?,finished_at=? WHERE evaluation_id=?",
                  (status, str(root), json.dumps(metrics), now(), EID))
    if status == "completed":
        reg.record_artifact(RID, evaluation_id=EID, kind="posthoc_so2_diagnostic_baselines_v1",
            path=root / "manifest.json", sha256=sha256_file(root / "manifest.json"),
            size_bytes=(root / "manifest.json").stat().st_size, status="present")

def prepare():
    assert not STAGE.exists() and not FINAL.exists(), "Use a new version instead of overwriting"
    prior = PRIOR / RID / "v1"
    source_cfg = read(prior / "config.resolved.json")
    dataset = source_cfg["source_resolved_config"]["dataset"]
    check(COHORT / "manifest.json", dataset["cohort_manifest_file_sha256"])
    check(GRAPHS / "manifest.json", dataset["graph_manifest_file_sha256"])
    sources = {}
    for manifest in sorted(PRIOR.glob("r_*/v1/manifest.json")):
        for rel, item in read(manifest)["files"].items():
            check(manifest.parent / rel, item["sha256"])
        check(manifest, read(manifest.parent / "_SUCCESS")["manifest_sha256"])
        sources[str(manifest.relative_to(P.report_root))] = sha256_file(manifest)
    assert len(sources) == 4
    cfg = {"campaign_id": CAMPAIGN, "evaluation_id": EID, "source_run_id": RID,
           "protocol": "so2_diagnostic_baselines_v1", "analysis_design": "exploratory_transductive_all_fit",
           "cores": list(range(15, 29)), "pilot_cores": [21, 23], "device": "cpu",
           "local_k": 16, "local_radius_um": 75, "huber_delta": 1,
           "gene_chunk": 64, "source_report_manifests": sources,
           "dataset": dataset, "contract_sha256": sha256_file(CONTRACT),
           "code_sha256": {rel: sha256_file(P.project_root / rel) for rel in SOURCES},
           "classification": {"lifecycle_stage": "exploratory_screen", "study_axis": "loss_and_spatial_baselines",
                              "scientific_variant": "fixed_baseline_diagnostics", "model_seed": 0, "fold_known": False}}
    STAGE.mkdir(parents=True)
    for rel in SOURCES:
        target = STAGE / "source" / rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(P.project_root / rel, target)
    shutil.copyfile(CONTRACT, STAGE / "frozen_task_contract.md")
    write(STAGE / "config.resolved.json", cfg)
    write(STAGE / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "command": sys.argv,
        "cwd": str(P.project_root), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "cpu_reason": "GPU devices occupied by existing independent training; sparse CPU baseline evaluation",
        "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True)})
    reg = Registry(initialize=False)
    reg.create_campaign(CAMPAIGN, name="SO2 reconstruction diagnostic and external comparison",
        scientific_question="Does loss preference or simple spatial averaging explain endpoint limitations?",
        status="pilot", config=cfg)
    reg.create_evaluation(EID, run_id=RID, checkpoint_name="last", dataset_id=dataset["dataset_id"],
        split_id=dataset["split_id"], status="running", metrics=cfg, artifact_path=STAGE)
    event("prepared")
    fit_constants()

def fit_constants():
    started = time.monotonic(); cm = read(COHORT / "manifest.json")
    check(COHORT / "cohort_statistics.npz", cm["files"]["cohort_statistics.npz"])
    with np.load(COHORT / "cohort_statistics.npz", allow_pickle=False) as a:
        means, scales = a["expression_mean"], a["expression_scale"]
    histograms = [np.zeros(1, dtype=np.float64) for _ in range(1000)]
    for core in range(15, 29):
        rel = f"cores/SO2-C{core}.npz"; check(COHORT / rel, cm["files"][rel])
        with np.load(COHORT / rel, allow_pickle=False) as a:
            counts = a["expression_counts"]
        assert counts.shape[1] == 1000 and counts.min() >= 0
        for gene in range(1000):
            h = np.bincount(counts[:, gene]).astype(np.float64) / (14 * len(counts))
            if len(h) > len(histograms[gene]):
                histograms[gene] = np.pad(histograms[gene], (0, len(h) - len(histograms[gene])))
            histograms[gene][:len(h)] += h
        event("histogram_core", core=core, cells=len(counts))
    rows = []; constants = []
    for gene, weights in enumerate(histograms):
        assert abs(weights.sum() - 1) < 1e-12
        logs = np.log1p(np.arange(len(weights)))
        assert abs(np.dot(weights, logs) - means[gene]) < 2e-7
        values = (logs - means[gene]) / scales[gene]
        c = huber_constant(values, weights)
        psi = float(np.dot(weights, np.clip(c - values, -1, 1)))
        assert abs(psi) < 1e-9
        def risk(pred):
            a = np.abs(pred - values); return float(np.dot(weights, np.where(a <= 1, .5*a*a, a-.5)))
        rh, rm, rz = risk(c), risk(0), risk(values[0])
        assert rh <= min(rm, rz) + 1e-10
        constants.append(c)
        rows.append({"gene_index": gene, "zero_fraction": weights[0], "mean_log1p": means[gene],
            "scale_log1p": scales[gene], "huber_constant_z": c, "huber_constant_log1p": c*scales[gene]+means[gene],
            "huber_stationarity_residual": psi, "huber_risk_constant": rh, "huber_risk_gene_mean": rm,
            "huber_risk_zero_count": rz})
    np.savez(STAGE / "constants.npz", huber_z=np.asarray(constants), means=means, scales=scales)
    table(STAGE / "constant_by_gene.csv", rows)
    write(STAGE / "constant_fit.json", {"genes": 1000, "below_mean_count": sum(c < -1e-9 for c in constants),
        "max_stationarity_error": max(abs(r["huber_stationarity_residual"]) for r in rows),
        "mean_huber_z": float(np.mean(constants)), "median_huber_z": float(np.median(constants)),
        "mean_fitted_huber_risk": float(np.mean([r["huber_risk_constant"] for r in rows])),
        "runtime_seconds": time.monotonic() - started, "constants_sha256": sha256_file(STAGE / "constants.npz")})
    event("constants_fitted", seconds=time.monotonic() - started)

def evaluate(core):
    out = STAGE / f"core_{core}.json"
    if out.exists():
        return
    started = time.monotonic(); alias = f"SO2-C{core}"
    cm, gm = read(COHORT / "manifest.json"), read(GRAPHS / "manifest.json")
    rel = f"cores/{alias}.npz"; check(COHORT / rel, cm["files"][rel])
    with np.load(COHORT / rel, allow_pickle=False) as a:
        counts, targets, coords = a["expression_counts"], a["target_expression"], a["coordinates_um"]
    prior = read(PRIOR / RID / "v1" / f"core_{core}.json")
    realization = sample_uniform_mask_numpy(len(counts), 1000, seed=prior["mask_seed"])
    assert realization.checksum == prior["mask_checksum"]
    mask = realization.mask
    assert mask.sum() == prior["n_masked_entries"]
    with np.load(STAGE / "constants.npz", allow_pickle=False) as a:
        means, scales, constant = a["means"], a["scales"], a["huber_z"]
    graph_rec = next(r for r in gm["cores"] if r["alias"] == alias)
    graph_path = GRAPHS / "cores" / alias / "edge_index.npy"
    check(graph_path, graph_rec["files"]["edge_index.npy"])
    edges = np.load(graph_path, mmap_mode="r")
    assert edges.shape[0] == 2 and not np.any(edges[0] == edges[1])
    full = sparse.csr_matrix((np.ones(edges.shape[1]), (edges[1], edges[0])), shape=(len(counts), len(counts)))
    assert full.nnz == edges.shape[1] and np.all(full.data == 1)
    local = local_adjacency(coords, k=16, radius=75.)
    accum = {name: NonzeroMetricAccumulator() for name in NAMES}
    fallback = {name: 0 for name in NAMES if "mean" in name and name != "gene_mean"}
    for start in range(0, 1000, 64):
        stop = min(start+64, 1000); sl = slice(start, stop)
        raw = counts[:, sl]; logs = np.log1p(raw.astype(np.float64)); z = targets[:, sl]
        np.testing.assert_allclose(z, (logs-means[sl])/scales[sl], atol=2e-5, rtol=2e-6)
        selected = mask[:, sl]; observed = ~selected
        predictions = {"gene_mean": np.broadcast_to(means[sl], raw.shape),
            "zero_count": np.zeros(raw.shape),
            "huber_constant": np.broadcast_to(constant[sl]*scales[sl]+means[sl], raw.shape)}
        for name, graph, values in (("local_log_mean", local, logs), ("local_count_mean", local, raw),
                                     ("full_graph_log_mean", full, logs)):
            pred, fellback = observed_neighbor_mean(graph, values, observed)
            if name == "local_count_mean": pred = np.log1p(pred)
            predictions[name] = pred
            fallback[name] += int(np.count_nonzero(fellback & selected))
        for name, pred in predictions.items():
            accum[name].update(true_counts=raw[selected], true_standardized=z[selected],
                pred_standardized=((pred-means[sl])/scales[sl])[selected], pred_log1p=pred[selected])
    metrics = {name: a.result() for name, a in accum.items()}
    differences = []
    for name in ("gene_mean", "zero_count"):
        for stratum in ("all", "zero", "positive"):
            for key in ("mse_standardized", "mse_log1p", "huber_standardized", "exact_count_accuracy"):
                differences.append(abs(metrics[name]["strata"][stratum][key] - prior["metrics"][name]["strata"][stratum][key]))
    assert max(differences) <= 2e-6
    for name, result in metrics.items():
        s = result["strata"]
        assert s["all"]["n"] == s["zero"]["n"] + s["positive"]["n"] == prior["n_masked_entries"]
        np.testing.assert_allclose(s["all"]["sums"]["mse_standardized"],
            s["zero"]["sums"]["mse_standardized"] + s["positive"]["sums"]["mse_standardized"], rtol=1e-12)
    runtime = time.monotonic() - started; rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    assert rss < 12, rss
    if core in (21, 23): assert runtime < 300, runtime
    receipt = {"core_alias": alias, "n_cells": len(counts), "mask_seed": prior["mask_seed"],
        "mask_checksum": realization.checksum, "n_masked_entries": int(mask.sum()),
        "source_npz_sha256": cm["files"][rel], "edge_sha256": graph_rec["files"]["edge_index.npy"],
        "full_edges": full.nnz, "local_edges": local.nnz, "metrics": metrics,
        "fallback_masked_entries": fallback, "baseline_replay_max_difference": max(differences),
        "entry_weighted_mask_fraction": float(np.square(realization.masked_gene_counts).sum() / (1000*mask.sum())),
        "scored_entries_from_ge900_masked": int(realization.masked_gene_counts[realization.masked_gene_counts >= 900].sum()),
        "runtime_seconds": runtime, "peak_host_rss_gib": rss, "completed_at": now()}
    write(out, receipt); event("core_complete", core=core, seconds=runtime, peak_rss_gib=rss)

def finish():
    receipts = [read(STAGE / f"core_{core}.json") for core in range(15, 29)]
    assert sum(r["n_cells"] for r in receipts) == 246063
    rows = []
    for r in receipts:
        for name, result in r["metrics"].items():
            for stratum, m in result["strata"].items():
                rows.append({"core": r["core_alias"], "predictor": name, "stratum": stratum, "n": m["n"],
                    **{key: m[key] for key in ("mse_standardized", "mse_log1p", "huber_standardized", "bias_standardized", "exact_count_accuracy")}})
    table(STAGE / "per_core.csv", rows)
    summary = []
    for name in NAMES:
        for stratum in receipts[0]["metrics"][name]["strata"]:
            subset = [r for r in rows if r["predictor"] == name and r["stratum"] == stratum]
            summary.append({"predictor": name, "stratum": stratum, "cores": len(subset),
                **{key: float(np.mean([r[key] for r in subset])) for key in subset[0] if key not in ("core", "predictor", "stratum", "n")}})
    table(STAGE / "summary.csv", summary)
    totals = {"status": "completed", "evaluation_id": EID, "cores": 14,
        "masked_entries": sum(r["n_masked_entries"] for r in receipts),
        "runtime_seconds": sum(r["runtime_seconds"] for r in receipts),
        "peak_host_rss_gib": max(r["peak_host_rss_gib"] for r in receipts),
        "replay_max_difference": max(r["baseline_replay_max_difference"] for r in receipts),
        "constant_fit": read(STAGE / "constant_fit.json"), "summary": summary}
    write(STAGE / "summary.json", totals); event("completed", cores=14)
    write(STAGE / "manifest.json", {"evaluation_id": EID, "source_run_id": RID, "cores": list(range(15,29)),
        "files": {str(f.relative_to(STAGE)): {"sha256": sha256_file(f), "size_bytes": f.stat().st_size}
                  for f in sorted(STAGE.rglob("*")) if f.is_file() and f.name not in ("manifest.json", "_SUCCESS")}})
    write(STAGE / "_SUCCESS", {"manifest_sha256": sha256_file(STAGE / "manifest.json"), "completed_at": now()})
    verify(STAGE); FINAL.parent.mkdir(parents=True, exist_ok=True); assert not FINAL.exists()
    os.rename(STAGE, FINAL); verify(FINAL); register_status("completed", FINAL, totals)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "pilot", "full", "verify"))
    args = parser.parse_args()
    if args.phase == "verify": return verify(FINAL)
    try:
        if args.phase == "prepare": return prepare()
        cfg = read(STAGE / "config.resolved.json")
        for rel, checksum in cfg["code_sha256"].items(): check(P.project_root / rel, checksum)
        check(STAGE / "constants.npz", read(STAGE / "constant_fit.json")["constants_sha256"])
        check(COHORT / "manifest.json", cfg["dataset"]["cohort_manifest_file_sha256"])
        check(GRAPHS / "manifest.json", cfg["dataset"]["graph_manifest_file_sha256"])
        if args.phase == "full":
            assert all((STAGE / f"core_{c}.json").exists() for c in (21,23)), "Pilot must pass first"
        event(args.phase + "_started", command=sys.argv)
        for core in ((21,23) if args.phase == "pilot" else range(15,29)): evaluate(core)
        if args.phase == "full": finish()
        else: event("pilot_passed")
    except Exception:
        if STAGE.exists():
            error = traceback.format_exc(); write(STAGE / "_FAILED", {"traceback": error, "time": now()})
            register_status("failed", STAGE, {"error": error})
        raise

if __name__ == "__main__":
    main()
