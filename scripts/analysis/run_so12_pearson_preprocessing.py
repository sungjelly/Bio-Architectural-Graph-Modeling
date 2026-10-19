"""Registered exploratory joint-slide preprocessing; see owning campaign README."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from spatial_benchmark.paths import current_paths
from spatial_benchmark.identifiers import scientific_id, repro_id, create_run_id, canonical_sha256, semantic_run_alias
from spatial_benchmark.registry import Registry, utc_now

CAMPAIGN = "cmp_20261016_so12_pearson_preprocessing"
METHODS = ("log1p_raw", "log1p_cp10k", "pearson", "signed_log_pearson")
VERSION = "so12_quasipoisson_v1"
CONFIG = {
    "protocol": VERSION, "campaign_id": CAMPAIGN,
    "classification": {"lifecycle_stage": "pilot", "study_axis": "preprocessing"},
    "model": {"family": "preprocessing_comparison"},
    "methods": list(METHODS), "seed": 0, "fold": 0, "attempt": 1,
    "train_cores": list(range(1, 11)) + list(range(15, 25)),
    "diagnostic_cores": list(range(11, 15)) + list(range(25, 29)),
    "donor_groups": "verified_consecutive_odd_even_core_pairs",
    "phi": 1.01, "upper_clip": 10.0, "baseline_target_total": 10000,
    "sample_sd_ddof": 1, "pca_components": 32, "sample_cells_per_core": 1000,
    "fit_population": "all_mapped_training_cells", "gene_panel": "all_1000_biological_genes",
    "pca": "common_centered_covariance_on_all_transformed_training_cells_no_whitening",
    "gpu_schedule": dict(zip(METHODS, range(4))), "gpu_covariance_chunk": 8192,
    "claim": "exploratory_preprocessing_plausibility_only",
    "acceptance": {"depth_r2": "below_log1p_raw", "max_within_slide_label_accuracy_drop": 0.05},
    "vendor_source": "https://github.com/Nanostring-Biostats/CosMx-Analysis-Scratch-Space/blob/41c499026d789ff35288de2ddc62f506a9d51cf0/_code/scPearsonPCA/R/sparse_quasipoisson_pca_seurat.R",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fit_transform(counts, train, method, device):
    """Fit only selected rows; all cell totals intentionally use fully observed counts."""
    x = torch.as_tensor(counts, dtype=torch.float32, device=device)
    select = torch.as_tensor(train, dtype=torch.bool, device=device)
    n = x.sum(1).clamp_min(1)
    stats = {}
    if method in ("pearson", "signed_log_pearson"):
        gene_sum = x[select].sum(0, dtype=torch.float64)
        p = (gene_sum / gene_sum.sum()).float()
        if not torch.all(p > 0).item():
            raise ValueError("Zero-frequency training gene")
        for start in range(0, len(x), 8192):
            mu = n[start:start+8192, None] * p[None, :]
            x[start:start+8192] = (x[start:start+8192] - mu) / torch.sqrt(CONFIG["phi"] * mu)
        stats["gene_frequency"] = p.cpu().numpy()
    elif method == "log1p_cp10k":
        x = torch.log1p(x / n[:, None] * CONFIG["baseline_target_total"])
    else:
        x = torch.log1p(x)
    indices = torch.where(select)[0]
    sum_x = torch.zeros(x.shape[1], dtype=torch.float64, device=device)
    sum_x2 = torch.zeros_like(sum_x)
    for start in range(0, len(indices), 8192):
        batch = x[indices[start:start+8192]].double()
        sum_x += batch.sum(0)
        sum_x2 += batch.square().sum(0)
    mean = (sum_x / len(indices)).float()
    sd = torch.sqrt((sum_x2 - sum_x.square()/len(indices))/(len(indices)-1)).float()
    if not torch.all(sd > 0).item():
        raise ValueError("Constant training gene")
    x.sub_(mean).div_(sd)
    stats.update(mean=mean.cpu().numpy(), sd=sd.cpu().numpy())
    stats["upper_clip_fraction"] = np.array(float((x > 10).float().mean().item()))
    if method in ("pearson", "signed_log_pearson"):
        x.clamp_(max=CONFIG["upper_clip"])
        stats["ordinary_log1p_invalid_fraction"] = np.array(float((x <= -1).float().mean().item()))
    if method == "signed_log_pearson":
        for start in range(0,len(x),8192):
            chunk = x[start:start+8192]
            x[start:start+8192] = chunk.sign() * torch.log1p(chunk.abs())
    if not torch.isfinite(x).all().item():
        raise ValueError("Nonfinite transformed features")
    return x, stats


def pca_and_export(method, gpu, counts, meta, selected, out):
    t0 = time.monotonic()
    device = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    torch.cuda.reset_peak_memory_stats(gpu)
    x, stats = fit_transform(counts, meta.train.to_numpy(), method, device)
    train_idx = np.flatnonzero(meta.train.to_numpy())
    pca_mean = torch.zeros(x.shape[1], dtype=torch.float64, device=device)
    gram = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float64, device=device)
    for start in range(0, len(train_idx), CONFIG["gpu_covariance_chunk"]):
        batch = x[torch.as_tensor(train_idx[start:start+CONFIG["gpu_covariance_chunk"]], device=device)]
        pca_mean += batch.sum(0, dtype=torch.float64)
        gram += (batch.T @ batch).double()
    pca_mean /= len(train_idx)
    cov = (gram - len(train_idx) * pca_mean[:, None] * pca_mean[None, :]) / (len(train_idx) - 1)
    eig, vec = torch.linalg.eigh(cov)
    components = vec[:, -CONFIG["pca_components"]:].flip(1).float()
    emb = ((x[torch.as_tensor(selected, device=device)] - pca_mean.float()) @ components).cpu().numpy()
    stats.update(pca_mean=pca_mean.cpu().numpy(), components=components.cpu().numpy(),
                 explained_variance=eig[-CONFIG["pca_components"]:].flip(0).cpu().numpy(),
                 explained_variance_ratio=(eig[-CONFIG["pca_components"]:].flip(0)/eig.sum()).cpu().numpy())
    method_dir = out / "methods" / method
    method_dir.mkdir(parents=True)
    np.savez(method_dir / "fitted_parameters.npz", **stats)
    np.save(method_dir / "sample_pcs.npy", emb)
    diagnostics = {"method": method, "gpu": gpu, "finite_fraction": 1.0,
                   "min": float(x.min().item()), "max": float(x.max().item()),
                   "negative_fraction": float((x < 0).float().mean().item()),
                   "upper_clip_fraction": float(stats["upper_clip_fraction"]),
                   "retained_pc_variance": float(stats["explained_variance_ratio"].sum())}
    if "ordinary_log1p_invalid_fraction" in stats:
        diagnostics["ordinary_log1p_invalid_fraction"] = float(stats["ordinary_log1p_invalid_fraction"])
    if method in ("pearson", "signed_log_pearson"):
        folder = out / "derived" / method
        folder.mkdir(parents=True)
        for core, group in meta.groupby("core", sort=True):
            indices = group.index.to_numpy()
            arr = x[torch.as_tensor(indices, device=device)].cpu().numpy()
            np.save(folder / f"{core}.npy", arr)
    diagnostics.update(runtime_seconds=time.monotonic()-t0,
                       peak_vram_gib=torch.cuda.max_memory_allocated(gpu)/2**30)
    write_json(method_dir / "diagnostics.json", diagnostics)
    print(json.dumps(diagnostics), flush=True)
    return method, emb, diagnostics


def smoke():
    """Independent formula, heldout fit invariance, and planted signal/depth controls."""
    rng = np.random.default_rng(3)
    labels = np.tile([0, 1], 400)
    library = np.exp(rng.normal(3, 0.8, len(labels)))
    rates = np.ones((2, 40))
    rates[0, :10] = 5
    rates[1, 10:20] = 5
    counts = rng.poisson(library[:, None] * rates[labels]).astype("int32")
    train = np.arange(len(counts)) < 600
    x, stats = fit_transform(counts, train, "pearson", "cuda:0")
    ref = counts.astype("float64")
    p = ref[train].sum(0) / ref[train].sum()
    mu = ref.sum(1)[:, None] * p
    r = (ref - mu) / np.sqrt(1.01 * mu)
    expected = np.minimum((r-r[train].mean(0))/r[train].std(0, ddof=1), 10)
    error = float(np.max(np.abs(x.cpu().numpy()-expected)))
    assert error < 1e-5, error
    perturbed = counts.copy()
    perturbed[~train] *= 5
    changed, changed_stats = fit_transform(perturbed, train, "pearson", "cuda:0")
    np.testing.assert_array_equal(stats["gene_frequency"], changed_stats["gene_frequency"])
    np.testing.assert_array_equal(stats["mean"], changed_stats["mean"])
    np.testing.assert_array_equal(x[train].cpu().numpy(), changed[train].cpu().numpy())
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, r2_score
    from sklearn.preprocessing import StandardScaler
    results = {}
    for method in METHODS:
        features, _ = fit_transform(counts, train, method, "cuda:0")
        features = features.cpu().numpy()
        scale = StandardScaler().fit(features[train])
        standardized = scale.transform(features)
        clf = LogisticRegression(max_iter=500).fit(standardized[train], labels[train])
        acc = balanced_accuracy_score(labels[~train], clf.predict(standardized[~train]))
        depth = np.log1p(counts.sum(1))
        reg = Ridge(alpha=1.0).fit(features[train], depth[train])
        results[method] = {"type_balanced_accuracy": float(acc),
                           "heldout_depth_r2": float(r2_score(depth[~train], reg.predict(features[~train])))}
    assert results["pearson"]["type_balanced_accuracy"] >= 0.95
    assert results["pearson"]["heldout_depth_r2"] < results["log1p_raw"]["heldout_depth_r2"]
    # One hidden gene changes other genes through the full-cell total: a deliberate leakage warning.
    altered = counts.copy()
    altered[-1, 0] += 100
    altered_x, _ = fit_transform(altered, train, "pearson", "cuda:0")
    assert not torch.equal(x[-1, 1:], altered_x[-1, 1:])
    return {"formula_max_abs_error": error, "heldout_fit_invariance": True,
            "hidden_gene_changes_other_features": True, "synthetic_controls": results,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cuda": torch.version.cuda, "peak_vram_gib": torch.cuda.max_memory_allocated(0)/2**30}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry-of")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    control = smoke()
    if args.smoke:
        print(json.dumps(control, indent=2))
        return
    from so12_preprocessing_data import load_inputs
    from so12_preprocessing_metrics import evaluate
    paths = current_paths()
    derived_target = paths.data_root / "processed/so12_pearson_integration_v1"
    if derived_target.exists():
        raise FileExistsError(derived_target)
    if shutil.disk_usage(paths.project_root).free < 9 * 2**30:
        raise RuntimeError("Need 9 GiB free before creating 3.3 GiB features; preserve 5 GiB reserve")
    assert torch.cuda.device_count() >= 4
    for gpu in range(4):
        assert torch.cuda.mem_get_info(gpu)[0] > 12*2**30
        # CUDA linalg lazily loads a process-global extension: initialize serially.
        torch.linalg.eigh(torch.eye(2, dtype=torch.float64, device=f"cuda:{gpu}"))
    source_files = sorted(Path(__file__).parent.glob("*so12*preprocessing*.py"))
    source_hashes = {str(p.relative_to(paths.project_root)): sha(p) for p in source_files}
    environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "packages": {p: importlib.metadata.version(p) for p in
                    ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn")},
                   "gpus": {str(i): torch.cuda.get_device_name(i) for i in range(4)}}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=paths.project_root, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=paths.project_root, text=True)
    diff = subprocess.check_output(["git", "diff"], cwd=paths.project_root)
    source_manifests = {f"so{s}": sha(paths.data_root / f"processed/so{s}_14core_relative_qkv_v1/manifest.json") for s in (1,2)}
    sci = scientific_id(CONFIG)
    rep = repro_id(CONFIG, git_commit=commit,
        dirty_fingerprint=canonical_sha256({"sources": source_hashes,"git_diff_sha256": hashlib.sha256(diff).hexdigest()}),
        dataset_fingerprint=canonical_sha256(source_manifests),
        split_fingerprint=canonical_sha256({"train": CONFIG["train_cores"], "diagnostic": CONFIG["diagnostic_cores"]}),
        preprocessing_version=VERSION, environment_fingerprint=canonical_sha256(environment))
    run_config = {**CONFIG,"attempt":args.attempt}
    run_id = create_run_id(seed=0, fold=0, attempt=args.attempt, scientific_id_value=sci)
    out = paths.scratch_root / "active_runs" / run_id
    out.mkdir(parents=True)
    reg = Registry()
    reg.create_campaign(CAMPAIGN, name="SO1 SO2 Pearson preprocessing trial",
        scientific_question="Does pooled Pearson preprocessing reduce depth dependence while retaining annotation consistency?", config=CONFIG, status="pilot")
    reg.register_variant(sci, campaign_id=CAMPAIGN, configuration=CONFIG)
    reg.create_run(run_id, campaign_id=CAMPAIGN, scientific_id=sci, repro_id=rep,
        seed=0, fold=0, attempt=args.attempt, configuration=run_config, artifact_path=out, retry_of=args.retry_of,
        git_commit=commit, dirty_status="dirty", preprocessing_version=VERSION)
    alias = semantic_run_alias(primary_run_id=run_id, historical=False, categories={
        "lifecycle_stage":"pilot", "study_axis":"preprocessing", "model":"four-transforms",
        "graph":"none", "mask":"none", "edge_feature_state":"none", "embedding_dim":32,
        "seed":0,"fold":0,"attempt":args.attempt,"variant_evidence":{"scientific_id":sci}})
    reg.register_run_semantics(run_id, lifecycle_stage="pilot", study_axis="preprocessing", source_batch="so12_joint",
        seed_known=True, fold_known=True, attempt_known=True, retention_class="analysis_evidence",
        category_key="preprocessing.pearson", classification_confidence="high", timestamp_basis="execution",
        variant_label="four_transform_comparison", model_key="preprocessing", dataset_key="so12_28core",
        masking_key="none", graph_key="none", feature_key="biological_1000", embedding_key="pca32",
        semantic_alias=alias, preferred_alias_type="semantic")
    reg.transition_run(run_id, "running", start_time=utc_now())
    t0 = time.monotonic()
    try:
        write_json(out/"resolved_config.json", run_config)
        write_json(out/"environment.json", environment)
        write_json(out/"smoke_checks.json", control)
        write_json(out/"identity.json", {"run_id":run_id,"semantic_alias":alias,"scientific_id":sci,"repro_id":rep,"git_commit":commit,
                                      "command":sys.argv,"cwd":str(paths.project_root)})
        (out/"git_status.txt").write_text(git_status)
        (out/"git_diff.patch").write_bytes(diff)
        (out/"code").mkdir()
        for f in source_files:
            shutil.copy2(f, out/"code"/f.name)
        shutil.copy2(paths.project_root/"experiments/campaigns"/CAMPAIGN/"README.md", out/"task_contract.md")
        print(f"RUN_ID={run_id}; loading source data", flush=True)
        counts, meta, genes, provenance = load_inputs(paths)
        assert counts.shape == (407659,1000)
        assert int(meta.train.sum()) == 301198
        write_json(out/"input_provenance.json", provenance)
        rng = np.random.default_rng(CONFIG["seed"])
        selected = np.concatenate([np.sort(rng.choice(group.index.to_numpy(), min(len(group),CONFIG["sample_cells_per_core"]),replace=False)) for _,group in meta.groupby("core",sort=True)])
        sampled = meta.iloc[selected].reset_index(drop=True)
        np.save(out/"sample_indices.npy", selected)
        sampled.to_csv(out/"sample_metadata.csv.gz",index=False)
        (out/"derived").mkdir(exist_ok=True)
        meta.to_csv(out/"derived/observations.csv.gz",index=False)
        write_json(out/"derived/genes.json", genes)
        print(f"Loaded {counts.shape}; fitting four transforms on GPUs 0–3", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(pca_and_export,m,g,counts,meta,selected,out) for g,m in enumerate(METHODS)]
            result = [f.result() for f in futures]
        embeddings = {m:e for m,e,d in result}
        transform_diagnostics = {m:d for m,e,d in result}
        write_json(out/"transform_diagnostics.json", transform_diagnostics)
        print("Evaluating fixed held-out diagnostics", flush=True)
        metrics = evaluate(embeddings,sampled,out/"diagnostics")
        write_json(out/"metrics.json", metrics)
        with (out/"metrics.jsonl").open("a") as handle:
            def record(prefix, value):
                if isinstance(value,dict):
                    for k,v in value.items(): record(f"{prefix}.{k}" if prefix else k,v)
                elif isinstance(value,(float,int)) and np.isfinite(value):
                    entry={"name":prefix,"value":float(value),"recorded_at":utc_now()}
                    handle.write(json.dumps(entry)+"\n")
                    reg.record_metric(run_id,prefix,float(value),split="diagnostic")
            record("",metrics)
        payloads=[]
        for method in ("pearson","signed_log_pearson"):
            shutil.copy2(out/"methods"/method/"fitted_parameters.npz",out/"derived"/method/"fitted_parameters.npz")
            for core,group in meta.groupby("core",sort=True):
                p=out/"derived"/method/f"{core}.npy"
                arr=np.load(p,mmap_mode="r")
                assert arr.shape==(len(group),len(genes)) and np.isfinite(arr).all()
                payloads.append({"method":method,"core":core,"rows":len(group),"columns":len(genes),
                                 "path":str(p.relative_to(out/"derived")),"sha256":sha(p),"size_bytes":p.stat().st_size})
        manifest={"schema_version":1,"kind":"fully_observed_joint_slide_preprocessing", "run_id":run_id,
                  "configuration":CONFIG,"cells":len(meta),"genes":len(genes),"matrices":payloads,
                  "observations":"observations.csv.gz","gene_order":"genes.json",
                  "raw_counts":"Reference existing per-core expression_counts; see input_provenance.json",
                  "graph_policy":"Separate core spatial graphs; no cross-core or cross-slide edges",
                  "training_ready_for_masked_counts":False,"provenance":provenance}
        write_json(out/"derived/manifest.json",manifest)
        write_json(out/"verification.json",{"all_arrays_finite":True,"all_shapes_verified":True,
            "count_row_alignment": "verified_all_28_cores", "new_observations_deleted":0,
            "runtime_seconds":time.monotonic()-t0,"free_disk_gib":shutil.disk_usage(out).free/2**30,
            "derived_bytes":sum(p.stat().st_size for p in (out/"derived").rglob("*") if p.is_file()),
            "checkpoint":"not_applicable_nontraining_preprocessing", "source_code_sha256":source_hashes})
        assert shutil.disk_usage(out).free > 5*2**30
        reg.transition_run(run_id,"finalizing")
        derived_target.parent.mkdir(parents=True,exist_ok=True)
        (out/"derived").rename(derived_target)
        write_json(out/"derived_dataset_reference.json",{"path":str(derived_target),"manifest_sha256":sha(derived_target/"manifest.json")})
        write_json(out/"checksums.json",{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
        (out/"_SUCCESS").write_text(run_id+"\n")
        partition=datetime.now(timezone.utc).strftime("%Y/%m")
        final=paths.artifact_root/"runs"/partition/run_id
        final.parent.mkdir(parents=True,exist_ok=True)
        out.rename(final)
        for p in final.rglob("*"):
            if p.is_file(): reg.record_artifact(run_id,kind="preprocessing_evidence",path=p,sha256=sha(p),size_bytes=p.stat().st_size,status="verified")
        reg.record_artifact(run_id,kind="derived_dataset_manifest",path=derived_target/"manifest.json",sha256=sha(derived_target/"manifest.json"),size_bytes=(derived_target/"manifest.json").stat().st_size,status="verified")
        reg.transition_run(run_id,"completed",artifact_path=final,end_time=utc_now(),duration_seconds=time.monotonic()-t0)
        print(f"COMPLETED {final}",flush=True)
    except BaseException:
        if out.exists(): (out/"failure.txt").write_text(traceback.format_exc())
        reg.transition_run(run_id,"failed",end_time=utc_now(),failure_category="preprocessing_exception")
        raise


if __name__ == "__main__":
    main()
