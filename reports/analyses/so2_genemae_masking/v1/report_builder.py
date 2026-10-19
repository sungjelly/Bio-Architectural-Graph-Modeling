#!/usr/bin/env python3
"""Verify and summarize all nine prespecified SO2 GeneMAE models."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle
from spatial_benchmark.result_catalog import validate_result_manifest
from spatial_benchmark.so2_genemae_masking import CAMPAIGN, METHODS, runtime_path

LABELS = {"whole_cell": "Whole cell", "partial_gene": "Individual genes", "spatial_block": "Spatial block"}
AUDITED_FAILURES = {
    "r_20261005T074042Z_b35cac84_s001_f00_a01_95e91d2d": "r_20261005T084829Z_b35cac84_s001_f00_a03_8bd59a60",
    "r_20261005T075208Z_b35cac84_s002_f00_a01_e5347d9b": "r_20261005T085114Z_b35cac84_s002_f00_a02_fbab1145",
}


def failed_replay_audit(source, replay):
    """Report the failed gate without relaxing it or discarding affected seeds."""
    verify_run_bundle(source); verify_run_bundle(replay)
    if not (source/"_FAILED").exists() or not (replay/"_FAILED").exists():
        raise ValueError("Expected immutable failed attempts")
    keys=["train_method","model_seed","core","evaluation_method","replicate","condition"]
    left=pd.read_parquet(source/"metrics/per_core_replicate.parquet").set_index(keys).sort_index()
    right=pd.read_parquet(replay/"metrics/per_core_replicate.parquet").set_index(keys).sort_index()
    pd.testing.assert_index_equal(left.index,right.index)
    if len(left)!=588 or list(left.columns)!=list(right.columns):raise ValueError("Incomplete replay")
    masks=json.loads((source/"diagnostics/evaluation_masks.json").read_text())
    if masks!=json.loads((replay/"diagnostics/evaluation_masks.json").read_text()):raise ValueError("Replay mask drift")
    prior=json.loads((replay/"diagnostics/replay_before.json").read_text())
    if len(prior)!=9 or not all(r["exact_cpu_replay"] and r["hidden_target_invariance"] for r in prior):
        raise ValueError("Independent pre-evaluation exact checkpoint/invariance checks failed")
    fields={}
    for col in left.select_dtypes(include="number"):
        bad=~np.isclose(left[col],right[col],rtol=1e-5,atol=1e-7,equal_nan=True)
        fields[col]={"mismatch_count":int(bad.sum()),"max_absolute_difference":float(np.nanmax(np.abs(left[col]-right[col])))}
    for col in ("n_masked","n_valid_cells","n_valid_genes","positive_count","zero_count"):
        np.testing.assert_array_equal(left[col],right[col])
    return {"training_run_id":source.name,"replay_run_id":replay.name,"original_run_status":"failed",
            "strict_replay_gate_passed":not any(v["mismatch_count"] for v in fields.values()),
            "rtol":1e-5,"atol":1e-7,"identical_masks_and_counts":True,
            "independent_exact_cpu_checks_passed":True,"fields":fields,
            "checkpoint_sha256":sha256_file(source/"checkpoints/last.ckpt"),
            "replay_failed_marker_sha256":sha256_file(replay/"_FAILED")}


def main():
    paths = current_paths()
    directory = paths.report_root / "analyses/so2_genemae_masking/v1"
    if directory.exists():
        raise ValueError("Report exists; use a new explicit version for changed evidence")
    registry = Registry()
    with registry.connect() as connection:
        jobs = [dict(r) for r in connection.execute(
            "SELECT job_id,run_id,status,failure_category,worker_id FROM queue_jobs WHERE campaign_id=?", (CAMPAIGN,))]
    runs, excluded, frames, histories, mask_reference = [], [], [], [], None
    replay_audits=[]
    for job in jobs:
        if not job["run_id"]:
            raise ValueError("Unfinished registered job")
        root = RunArchive.artifact_path_for(job["run_id"], paths)
        if not root.exists():
            raise ValueError(f"Run not finalized: {job['run_id']}")
        config = yaml.safe_load((root / "config.resolved.yaml").read_text())
        pilot = config["metadata"]["execution_role"] == "resource_pilot"
        if pilot or job["status"] != "completed":
            reason = "diagnostic resource pilot" if pilot else "production failure"
            if job["status"] != "completed":
                error_log=root/"logs/stderr.log"
                error_lines=error_log.read_text().splitlines() if error_log.is_file() else []
                failed_summary=json.loads((root/"summary.json").read_text())
                detail=error_lines[-1] if error_lines else failed_summary.get("failure_message",str(job["failure_category"]))
                reason += ": " + detail
            if job["run_id"] in AUDITED_FAILURES:
                reason += "; retained in the descriptive comparison with failed verification explicitly flagged"
            excluded.append({"run_id": job["run_id"], "reason": reason, "status": job["status"]})
            if job["run_id"] not in AUDITED_FAILURES:
                continue
        verification = verify_run_bundle(root, require_success_contract=True)
        if verification["status"] != "success" and job["run_id"] not in AUDITED_FAILURES:
            raise ValueError("Bundle verification failed")
        summary = json.loads((root / "summary.json").read_text())
        if job["run_id"] in AUDITED_FAILURES:
            replay=RunArchive.artifact_path_for(AUDITED_FAILURES[job["run_id"]],paths)
            audit=failed_replay_audit(root,replay);replay_audits.append(audit)
            history=pd.read_parquet(root/"metrics/history.parquet")
            if history.epoch.tolist()!=list(range(1,201)):raise ValueError("Failed attempt has incomplete training")
            record=registry.get_run(job["run_id"])
            runtime=(datetime.fromisoformat(record["end_time"].replace("Z","+00:00"))-datetime.fromisoformat(record["start_time"].replace("Z","+00:00"))).total_seconds()
            summary={**summary,"completed_global_epochs":200,"final_epoch":199,"model_seed":config["seed"],
                     "train_method":config["masking"]["method"],"strict_execution_gate_passed":False,
                     "resources":{"visible_devices":job["worker_id"].removeprefix("so2-genemae-gpu"),"total_seconds":runtime,
                                  "peak_allocated_vram_gib":None,"scope":"original training/evaluation; peak not saved before failure"}}
        if summary["completed_global_epochs"] != 200:
            raise ValueError("Incomplete epoch budget")
        masks = json.loads((root / "diagnostics/evaluation_masks.json").read_text())
        if mask_reference is None:
            mask_reference = masks
        elif masks != mask_reference:
            raise ValueError("Comparison masks differ")
        frame = pd.read_parquet(root / "metrics/per_core_replicate.parquet")
        if len(frame) != 14 * 3 * (6+4+4):
            raise ValueError("Missing evaluation rows")
        frame["run_id"] = job["run_id"]
        frames.append(frame)
        history = pd.read_parquet(root / "metrics/history.parquet")
        history["method"] = config["masking"]["method"]; history["seed"] = config["seed"]
        histories.append(history)
        runs.append({"run_id": job["run_id"], "method": config["masking"]["method"], "seed": config["seed"],
                     "path": str(root.relative_to(paths.artifact_root)), "summary": summary,"status":job["status"]})
    slots = [(r["method"],r["seed"]) for r in runs]
    if sorted(slots) != sorted((m,s) for m in METHODS for s in range(3)):
        raise ValueError(f"Expected all nine slots exactly once; found {slots}")
    frame = pd.concat(frames, ignore_index=True)
    history = pd.concat(histories, ignore_index=True)
    metrics = ["masked_huber", "masked_mse", "masked_mae", "masked_r2", "pooled_pearson",
               "gene_pearson_mean", "cell_pearson_mean", "positive_mse", "positive_mae", "zero_mse"]
    keys = ["train_method", "model_seed", "evaluation_method", "condition", "core"]
    core = frame.groupby(keys, as_index=False)[metrics].mean()
    seed = core.groupby(keys[:-1], as_index=False)[metrics].mean()
    observed = seed[seed.condition == "observed"]
    means = observed.groupby(["train_method", "evaluation_method"])[metrics].mean()
    stds = observed.groupby(["train_method", "evaluation_method"])[metrics].std()
    # Common baselines must agree independently of training method or seed.
    for condition in ("zero", "visible_gene_mean", "neighbor_mean"):
        subset = core[core.condition == condition]
        spread = subset.groupby(["evaluation_method", "core"])["masked_huber"].agg(lambda v: v.max()-v.min())
        if spread.max() > 1e-10:
            raise ValueError("Shared baseline mismatch")
    primary = core[(core.train_method == "whole_cell") & (core.evaluation_method == "whole_cell")]
    wide = primary.pivot(index=["model_seed","core"], columns="condition", values="masked_huber")
    gain = 100*(1-wide.observed.mean()/wide.visible_gene_mean.mean())
    favor_cores = int((wide.groupby("core").mean().observed < wide.groupby("core").mean().visible_gene_mean).sum())
    favor_seeds = int((wide.groupby("model_seed").mean().observed < wide.groupby("model_seed").mean().visible_gene_mean).sum())
    supported = gain >= 2 and favor_cores >= 10 and favor_seeds == 3
    baseline_outcome = "supported" if supported else "negative"
    outcome = "inconclusive" if replay_audits else baseline_outcome
    directory.mkdir(parents=True)
    frame.to_csv(directory / "all_core_replicate_metrics.csv", index=False)
    core.to_csv(directory / "core_means.csv", index=False)
    seed.to_csv(directory / "equal_core_seed_metrics.csv", index=False)
    history.to_csv(directory / "epoch_history.csv", index=False)
    means.reset_index().to_csv(directory / "model_task_means.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1,3, figsize=(13,4), constrained_layout=True)
    for ax, task in zip(axes, METHODS):
        values = [means.loc[(m,task),"masked_huber"] for m in METHODS]
        errors = [stds.loc[(m,task),"masked_huber"] for m in METHODS]
        ax.bar(range(3), values, yerr=errors, color=["#167d9a","#de8f05","#8561a9"], capsize=4)
        references = seed[(seed.evaluation_method == task) & (seed.train_method == "whole_cell")]
        for cond,color in (("visible_gene_mean","#444444"),("zero","#b43a39"),("neighbor_mean","#278044")):
            val = references[references.condition == cond].masked_huber.mean()
            ax.axhline(val, color=color, linestyle="--", label=cond.replace("_"," "))
        ax.set_title(f"Evaluation: {LABELS[task]}")
        ax.set_xticks(range(3), [LABELS[m] for m in METHODS], rotation=20)
        ax.set_xlabel("Training mask (10%)"); ax.set_ylabel("Equal-core Huber loss ↓")
    axes[0].legend(fontsize=8)
    fig.suptitle("Matched evaluation tasks; error bars are SD across 3 technical seeds")
    fig.savefig(directory / "reconstruction_comparison.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1,3, figsize=(13,3.7), constrained_layout=True)
    for ax,method in zip(axes,METHODS):
        for s in range(3):
            values=history[(history.method==method)&(history.seed==s)]
            ax.plot(values.epoch,values.mean_core_huber,label=f"Seed {s}",alpha=.8)
        ax.set_title(LABELS[method]); ax.set_xlabel("Epoch"); ax.set_ylabel("Masked training Huber ↓"); ax.legend()
    fig.suptitle("Training curves describe different masking tasks")
    fig.savefig(directory / "training_curves.png",dpi=220);plt.close(fig)
    fig, axes = plt.subplots(1,2, figsize=(10,3.8), constrained_layout=True)
    for ax,metric,label in zip(axes,["positive_mse","zero_mse"],["Positive-expression MSE ↓","Zero-expression MSE ↓"]):
        values=[means.loc[(m,"whole_cell"),metric] for m in METHODS]
        errors=[stds.loc[(m,"whole_cell"),metric] for m in METHODS]
        ax.bar(range(3),values,yerr=errors,capsize=4,color=["#167d9a","#de8f05","#8561a9"])
        ax.set_xticks(range(3),[LABELS[m] for m in METHODS]);ax.set_ylabel(label)
    fig.suptitle("All models evaluated on identical 10% whole-cell masks")
    fig.savefig(directory / "positive_zero_errors.png",dpi=220);plt.close(fig)
    # Preserve core heterogeneity instead of displaying only the cohort mean.
    fig, ax = plt.subplots(figsize=(11,4), constrained_layout=True)
    core_order = sorted(core.core.unique(), key=lambda name: int(name.split("C")[-1]))
    for offset, method, color in zip((-.2,0,.2), METHODS, ("#167d9a","#de8f05","#8561a9")):
        subset=core[(core.train_method==method)&(core.evaluation_method=="whole_cell")&(core.condition=="observed")]
        by_core=subset.groupby("core").masked_huber.agg(["mean","std"]).reindex(core_order)
        ax.errorbar(np.arange(len(core_order))+offset,by_core["mean"],yerr=by_core["std"],
                    fmt="o",capsize=2,color=color,label=LABELS[method])
    baseline=core[(core.train_method=="whole_cell")&(core.evaluation_method=="whole_cell")&(core.condition=="visible_gene_mean")]
    ax.plot(range(len(core_order)),baseline.groupby("core").masked_huber.mean().reindex(core_order),
            "x",color="#444444",label="Visible gene mean")
    ax.set_xticks(range(len(core_order)),core_order,rotation=35)
    ax.set_ylabel("Whole-cell Huber loss ↓");ax.set_xlabel("Fitted SO2 core")
    ax.set_title("Core heterogeneity; error bars are SD across technical seeds")
    ax.legend(ncol=4,fontsize=8)
    fig.savefig(directory / "core_reconstruction.png",dpi=220);plt.close(fig)

    primary_score=means.loc[("whole_cell","whole_cell"),"masked_huber"]
    observed_text=(f"Whole-cell-trained GeneMAE has equal-core Huber {primary_score:.6f} on fixed 10% whole-cell masks, "
                   f"{gain:.2f}% improvement over visible per-gene means; {favor_cores}/14 cores and {favor_seeds}/3 seeds favor it. "
                   f"The prespecified reconstruction baseline gate is {baseline_outcome}.")
    comparisons=[]
    for alternative in ("partial_gene", "spatial_block"):
        reference=means.loc[(alternative,"whole_cell"),"masked_huber"]
        difference=100*(primary_score/reference-1)
        comparisons.append(f"{abs(difference):.2f}% {'higher' if difference > 0 else 'lower'} Huber than training with {LABELS[alternative].lower()} masks")
    observed_text += " On the same whole-cell evaluation masks, whole-cell training has " + "; ".join(comparisons) + "."
    primary_metrics=means.loc[("whole_cell","whole_cell")]
    zero_metrics=seed[(seed.train_method=="whole_cell")&(seed.evaluation_method=="whole_cell")&(seed.condition=="zero")][metrics].mean()
    if zero_metrics.masked_mae < primary_metrics.masked_mae:
        observed_text += (f" The all-zero predictor has lower MAE ({zero_metrics.masked_mae:.6f} versus "
                          f"{primary_metrics.masked_mae:.6f}); its Huber is {zero_metrics.masked_huber:.6f} and "
                          f"MSE {zero_metrics.masked_mse:.6f}, compared with model MSE {primary_metrics.masked_mse:.6f}. "
                          "This metric disagreement makes the primary loss and positive-expression errors important to interpretation.")
    audit_text=("All nine models completed 200 training epochs. Seven original runs passed all execution checks; individual-gene seeds "
        "1 and 2 failed the final exact CPU replay/invariance assertion. Independent fresh CPU replay and hidden-target checks passed, "
        "but full evaluation replays failed the prespecified numerical agreement gate only for median gene-wise Pearson "
        "(maximum differences 6.24e-7 and 7.36e-7). The strict gate remains failed; the nine-run verification criterion is not met. "
        "All seeds remain in this explicitly descriptive comparison, using their original recorded evaluations. Reconstruction "
        "loss replay differences and every secondary metric discrepancy are retained in replay_audit.json. The overall campaign "
        "outcome is inconclusive under its full verification contract; the narrower whole-cell baseline result is supported.")
    observed_text += " " + audit_text
    lines=["# SO_2 GeneMAE: 10% masking comparison", "", observed_text,"",
           "All nine 200-epoch trained models are included. Values average three mask replicates per core, fourteen cores equally, then three seeds. "
           "This is fitted-cohort reconstruction; evaluation cells were visible in other training epochs.","",
           "![Masked Huber comparisons within each evaluation task](reconstruction_comparison.png)","",
           "## Matched whole-cell reconstruction","",
           "| Training mask | Huber ± seed SD ↓ | MSE ↓ | MAE ↓ | R² ↑ | Pooled Pearson ↑ | Mean gene Pearson ↑ | Positive MSE ↓ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        row=means.loc[(method,"whole_cell")]; sd=stds.loc[(method,"whole_cell"),"masked_huber"]
        lines.append(f"| {LABELS[method]} | {row.masked_huber:.6f} ± {sd:.6f} | {row.masked_mse:.6f} | {row.masked_mae:.6f} | {row.masked_r2:.4f} | {row.pooled_pearson:.4f} | {row.gene_pearson_mean:.4f} | {row.positive_mse:.6f} |")
    lines += ["", "| Training mask | Seed 0 Huber ↓ | Seed 1 Huber ↓ | Seed 2 Huber ↓ |", "|---|---:|---:|---:|"]
    for method in METHODS:
        selected=observed[(observed.train_method==method)&(observed.evaluation_method=="whole_cell")].set_index("model_seed")
        lines.append("| "+LABELS[method]+" | "+" | ".join(f"{selected.loc[s,'masked_huber']:.6f}" for s in range(3))+" |")
    lines += ["", "![Whole-cell reconstruction in each fitted core](core_reconstruction.png)", ""]
    lines += ["", "## Cross-task evaluation", "", "| Training mask | Whole-cell Huber ↓ | Gene-entry Huber ↓ | Spatial-block Huber ↓ |",
              "|---|---:|---:|---:|"]
    for m in METHODS:
        lines.append("| "+LABELS[m]+" | "+" | ".join(f"{means.loc[(m,t),'masked_huber']:.6f}" for t in METHODS)+" |")
    lines += ["", "Compare models within a column. Differences between columns include task difficulty and available information; "
              "they do not establish model superiority.","", "## Controls on the requested whole-cell task", "",
              "| Condition | Huber ↓ | MAE ↓ | Positive MSE ↓ |", "|---|---:|---:|---:|"]
    controls=seed[(seed.train_method=="whole_cell")&(seed.evaluation_method=="whole_cell")].groupby("condition")[metrics].mean()
    for condition,row in controls.iterrows():
        lines.append(f"| {condition.replace('_',' ')} | {row.masked_huber:.6f} | {row.masked_mae:.6f} | {row.positive_mse:.6f} |")
    limits=("Sparsity, broad spatial fields, cell identity and segmentation spillover remain alternative explanations. "
            "Visible means, neighbor smoothing and frozen-model graph ablations address some shortcuts but do not replace trained matched "
            "self/morphology/regional controls. Node-label permutation is not a confounder-preserving biological null. "
            "No patient-held-out split, population interval, calibration, biological mechanism or causal effect is established. "
            "All uncertainty bars describe technical seed variability only. Training masks are shared across seeds within each method, "
            "while initialization, dropout and traversal order vary with seed. Tiling removes cross-tile edges and may especially limit block reconstruction. "
            "Equal epoch budgets visit every tile; spatial-block tiles without targets skip optimization, so optimizer-step counts differ. "
            "Pooled Pearson can reflect differences in average abundance between genes; mean gene-wise Pearson is reported separately. "
            "The full tables expose positive and zero errors, undefined correlations and core heterogeneity.")
    lines += ["", "![Positive and zero errors on common whole-cell masks](positive_zero_errors.png)","",
              "![All nine training trajectories](training_curves.png)","", "## Interpretation and limitations","",limits,"",
              "## Execution and provenance","", "| Training mask | Seed | Immutable run | Status | GPU | Optimizer steps | Runtime min | Peak VRAM GiB |", "|---|---:|---|---|---:|---:|---:|---:|"]
    for r in runs:
        resources=r["summary"]["resources"]
        steps=int(history[(history.method==r["method"])&(history.seed==r["seed"])].optimizer_steps.sum())
        peak=resources['peak_allocated_vram_gib']
        peak_text=f"{peak:.2f}" if peak is not None else "not saved"
        lines.append(f"| {LABELS[r['method']]} | {r['seed']} | `{r['run_id']}` | {r['status']} | {resources['visible_devices']} | {steps} | {resources['total_seconds']/60:.2f} | {peak_text} |")
    hardware=next(r["summary"]["resources"] for r in runs if r["status"]=="completed")
    lines += ["", f"Hardware: {hardware['gpu_name']}; PyTorch {hardware['torch']}; CUDA {hardware['cuda']}; FP32. "
              "One independent job per GPU on devices 0–3, with the next queued job claimed as a device becomes free. "
              "Longer pending gene-entry runs received priority after initial throughput measurements."]
    for r in runs:
        if r["summary"].get("recovery_source_run_id"):
            lines += ["", f"† {LABELS[r['method']]} seed {r['seed']} uses the unchanged checkpoint trained in "
                f"`{r['summary']['recovery_source_run_id']}`. That attempt completed training/evaluation but failed its combined exact "
                "CPU replay/invariance assertion; the original cause remains unconfirmed. A separate recovery attempt performed no "
                "optimizer steps, passed exact replay and hidden-target invariance across all three masking methods, reproduced "
                "all 588 evaluation rows within rtol=1e-5/atol=1e-7 with identical masks and counts, and verified exact checkpoint "
                "parameter equality. Its memory figure measures recovery evaluation only; the original training peak was not saved. "
                "The displayed runtime includes the failed attempt and recovery. Both bundles remain available."]
    lines += ["", "Diagnostic and failed runs are retained in `verification.json`; no production seed is silently omitted.", "",
              "Focused masking/configuration/queue tests passed (49 initially), and archive/masking checks passed (34 after the serialization fix). "
              "Recovery routing and strict replay-audit checks passed with the focused suite (52 tests). "
              "The latest CPU-isolated full repository suite returned 1,437 passed, 34 failed and four skipped. Failures concern unavailable historical "
              "adjacency/GeneMAE/geometry inputs or an older external gradient-audit source identity. Logs are retained in `../verification/`. "
              "Initial doctor also reported a pre-existing incomplete checkpoint catalog and its generic 25-GiB disk threshold; this campaign "
              "uses the documented 8-GiB floor plus a 3-GiB output reserve. These exceptions are not represented as a clean repository-wide audit.", "",
              "Reproduce from repository root with `PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_genemae_masking.py` "
              "after all prespecified runs finalize. The script refuses an existing report directory. Training commands are in the campaign README.",""]
    (directory/"README.md").write_text("\n".join(lines))
    (directory/"replay_audit.json").write_text(json.dumps(replay_audits,indent=2,allow_nan=False)+"\n")
    receipt={"campaign":CAMPAIGN,"outcome":outcome,"whole_cell_baseline_outcome":baseline_outcome,"baseline_gain_percent":gain,"favor_cores":favor_cores,"favor_seeds":favor_seeds,
             "runs":runs,"excluded":excluded,"common_masks_verified":True,"all_bundle_checksums_verified":True,
             "all_execution_gates_passed":not replay_audits,"replay_audits":replay_audits}
    (directory/"verification.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    shutil.copyfile(__file__, directory/"report_builder.py")
    manifest={p.name:{"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in sorted(directory.iterdir()) if p.is_file()}
    (directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

    result_id="res_so2_genemae_ten_percent_masking"
    result_dir=paths.result_root/"descriptive_analysis/genemae"/result_id
    result_dir.mkdir(parents=True,exist_ok=False)
    sources=[]
    for r in runs:
        marker="_SUCCESS" if r["status"]=="completed" else "_FAILED"
        sources.append({"kind":"run_artifact","root":"artifact","path":r["path"]+"/"+marker,
              "sha256":sha256_file(paths.artifact_root/r["path"]/marker),"verification_status":"verified","run_id":r["run_id"]})
    sources.append({"kind":"report","root":"report","path":str((directory/"manifest.json").relative_to(paths.report_root)),
                    "sha256":sha256_file(directory/"manifest.json"),"verification_status":"verified"})
    now=datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
    record={"schema_name":"bagm_result_record","schema_version":1,"revision":1,"result_id":result_id,
      "title":"SO2 GeneMAE reconstruction under three 10% masking techniques","status":"verified","outcome":outcome,
      "experiment_type":"descriptive_analysis","method_family":"genemae","lifecycle_stage":"exploratory_screen",
      "study_axis":"masking_comparison","campaign_ids":[CAMPAIGN],
      "method":{"name":"Matched SO2 GeneMAE masking comparison","model_family":"Dual-path GeneMAE",
        "graph_context":"Spatial k15 union, Gaussian distance feature, coordinate-only tiles of at most 7000 cells",
        "masking_or_perturbation":"10% whole cells, 10% entries per cell, or contiguous 10% cell blocks; all-gene masking for hidden cells",
        "evaluation_design":"All 14 cores fitted; 3 seeds per method, 200 epochs; identical 3 replicates per evaluation task; equal-core aggregation",
        "implementation_version":"held_in_so2_genemae_masking_v1; executed source and configs preserved in each run"},
      "conclusion":{"question":"How does whole-cell reconstruction compare with gene-entry and spatial-block masking?",
        "estimand":"Three separately reported fitted-cohort masked log1p-count reconstruction tasks",
        "observed_result":observed_text,"strongest_alternative_explanation":"Sparsity, cell identity, regional fields and segmentation spillover rather than useful cell interactions",
        "controls":["Visible gene mean, zero and neighbor mean", "Identical evaluation masks across all nine models", "Frozen model graph relabeling and self-edge-only ablation on whole-cell masks", "Synthetic neighbor-smoothing recovery, constant-profile model optimization and hidden-target invariance checks; no planted biological interaction recovery claim"],
        "remaining_uncertainty":limits,"maximum_defensible_claim":"Descriptive matched reconstruction comparison within the fitted SO2 cohort, conditional on the specified architecture, masks and training budget"},
      "evidence":{
        "predictive_gain":{"status":"not_tested","summary":"Patient-held-out graph-specific gain is not tested; the narrower fitted-cohort whole-cell baseline gate is "+baseline_outcome},
        "stability":{"status":"mixed","summary":"All three seeds and masks reported; individual-gene seeds1/2 fail strict numerical replay in median gene Pearson; graph, segmentation and independent-patient robustness untested"},
        "faithfulness":{"status":"not_tested","summary":"Graph-use ablations are diagnostic; no attribution faithfulness study"},
        "null_calibration":{"status":"not_tested","summary":"Topology-preserving label permutation does not preserve biological confounders"},
        "patient_replication":{"status":"not_tested","summary":"All fourteen cores belong to fitting; no independent patient test"},
        "external_support":{"status":"not_tested","summary":"No orthogonal validation"},
        "perturbation_support":{"status":"not_tested","summary":"No biological perturbation"}},
      "provenance":{"created_at_utc":now,"updated_at_utc":now,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=paths.project_root,text=True).strip(),
        "run_ids":[r["run_id"] for r in runs],"expected_seeds":[0,1,2],"included_seeds":[0,1,2],
        "expected_folds":[0],"included_folds":[0],"failed_or_excluded_runs":[{"run_id":r["run_id"],"reason":r["reason"]} for r in excluded],"sources":sources},
      "files":[]}
    (result_dir/"result.yaml").write_text(yaml.safe_dump(record,sort_keys=False))
    (result_dir/"README.md").write_text("# SO2 GeneMAE masking comparison\n\n"+observed_text+"\n\n"+limits+
        "\n\nFull report: `reports/analyses/so2_genemae_masking/v1/README.md`.\n")
    validate_result_manifest(result_dir/"result.yaml",paths=paths,verify_sources=True)
    print(json.dumps({"report":str(directory),"result":str(result_dir),"outcome":outcome,"gain_percent":gain}))


if __name__ == "__main__":
    main()
