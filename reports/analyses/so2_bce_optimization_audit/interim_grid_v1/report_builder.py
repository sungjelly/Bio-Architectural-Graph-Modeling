#!/usr/bin/env python3
"""Build immutable stage reports from verified, registered BCE audit bundles."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import mean
import shutil
import subprocess
import uuid

import yaml

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, _rename_no_replace, verify_run_bundle
from spatial_benchmark.so2_bce_optimization_audit import (
    CAMPAIGN, GRID_LEARNING_RATES, PREFLIGHT_RELATIVE_PATH, PROFILE_EPOCHS,
    SOURCE_RUN, TARGET_BATCH_SIZES, validate_configuration,
)

GIB = 1024**3
METRIC = "validation_pooled_masked_binary_cross_entropy"
SECONDARY = ("detection_brier_score", "detection_auprc_histogram_4096", "detection_auroc_histogram_4096",
             "detection_expected_calibration_error_histogram_4096", "detection_precision", "detection_recall",
             "detection_f1", "detection_specificity", "predicted_positive_class_rate", "observed_positive_rate")
INCUMBENT = {"batch_mode":"target_window", "target_batch_size":1024, "learning_rate":1e-4,
             "dropout":.1, "weight_decay":1e-5}
SETTING_FIELDS = tuple(INCUMBENT)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value):
    value = float(value)
    require(math.isfinite(value), "Report source contains a nonfinite scalar")
    return value


def setting(row):
    return {key:row[key] for key in SETTING_FIELDS}


def setting_key(row):
    return tuple(row[key] for key in SETTING_FIELDS)


def run_key(row):
    return (row["profile"], *setting_key(row), row["seed"])


def descriptor(config):
    trainer = config["trainer"]
    return {"profile":config["metadata"]["execution_profile"],
            "batch_mode":trainer["optimizer_batch_mode"],
            "target_batch_size":trainer["target_cells_per_optimizer_update"],
            "learning_rate":trainer["learning_rate"], "dropout":config["model"]["dropout"],
            "weight_decay":trainer["weight_decay"], "seed":config["seed"], "fold":config["fold"],
            "attempt":config["attempt"], "physical_gpu":config["launcher"]["requested_gpu"],
            "optimizer_steps_per_epoch":trainer["optimizer_updates_per_global_epoch"]}


def selected_metrics(final):
    result = {"bce":finite(final["val/target_only_masked_binary_cross_entropy"])}
    result.update({key:finite(final["val/unseen_donor/"+key]) for key in SECONDARY})
    return result


def load_completed(record, paths, registry):
    run_id = record["run_id"]
    root = RunArchive.artifact_path_for(run_id,paths)
    require(Path(record["artifact_path"]).resolve() == root.resolve(), "Registered bundle path differs")
    verify_run_bundle(root)
    require((root/"_SUCCESS").is_file(), "Completed run lacks success marker")
    config = yaml.safe_load((root/"config.resolved.yaml").read_text())
    validate_configuration(config)
    row = {"run_id":run_id, "status":"completed", "bundle_path":str(root.relative_to(paths.artifact_root)), **descriptor(config)}
    summary = read_json(root/"summary.json")
    receipt = read_json(root/PREFLIGHT_RELATIVE_PATH)
    endpoint = read_json(root/"results/optimization_audit_endpoint.json")
    baseline = read_json(root/"diagnostics/train_fitted_gene_prevalence_validation.json")
    epochs = read_csv(root/"results/epoch_metrics.csv")
    expected_epochs = PROFILE_EPOCHS[row["profile"]]
    require([int(x["global_epoch"]) for x in epochs] == list(range(1,expected_epochs+1)), "Incomplete epoch history")
    require(summary["final_epoch"] == expected_epochs and summary["checkpoint_reload_verified"] is True,
            "Final budget/replay is incomplete")
    require(receipt["passed"] is True and receipt["coverage_verified"] is True and receipt["finite_gradients"] is True
            and receipt["completed_epochs"] == expected_epochs, "Numerical/coverage gates failed")
    require(canonical_sha256(baseline) == receipt["prevalence_reference_sha256"], "Prior reference hash differs")
    require(baseline["fit_scope"] == "training_cores_only" and baseline["validation_fitted"] is False,
            "Prior reference used validation fitting")
    wanted = config["experiment"]["screening_endpoint_epochs"]
    terminal = [epochs[i-1] for i in wanted]
    value = mean(finite(x[METRIC]) for x in terminal)
    require(endpoint["endpoint_epochs"] == wanted and abs(value-finite(endpoint["mean_validation_bce"])) <= 1e-12,
            "Stored endpoint differs from prespecified terminal history")
    baseline_mask_sha = baseline["validation_coverage_receipts_sha256"]
    require(all(canonical_sha256(json.loads(x["validation_coverage_receipts_json"])) == baseline_mask_sha
                for x in epochs), "Model/prior validation masks differ")
    require(all(int(x["train_target_cells"]) == 208696 and int(x["validation_target_cells"]) == 37367
                and int(x["optimizer_updates"]) == row["optimizer_steps_per_epoch"]
                and int(x["cumulative_optimizer_updates"]) == int(x["global_epoch"])*row["optimizer_steps_per_epoch"]
                and finite(x["validation_detection_threshold"]) == .5 for x in epochs), "Coverage/steps/threshold differs")
    core_history = [x for x in read_csv(root/"results/per_core_epoch_metrics.csv") if x["split"] == "validation"]
    aliases = sorted({x["core_alias"] for x in core_history})
    require(len(aliases) == 2 and len(core_history) == 2*expected_epochs, "Per-core history incomplete")
    cores = {}
    for alias in aliases:
        observed = [x for x in core_history if x["core_alias"] == alias]
        require([int(x["global_epoch"]) for x in observed] == list(range(1,expected_epochs+1)), "Duplicate/missing core epochs")
        selected = [x for x in observed if int(x["global_epoch"]) in wanted]
        cores[alias] = {"bce":mean(finite(x["pooled_masked_binary_cross_entropy"]) for x in selected),
                       **{key:mean(finite(x[key]) for x in selected) for key in SECONDARY}}
    final = read_json(root/"metrics/final.json")
    checkpoint = root/"checkpoints/best.ckpt"
    checkpoint_sha = sha256_file(checkpoint)
    catalog = registry.list_checkpoint_catalog(run_id=run_id,role="best",verification_status="verified")
    require(len(catalog) == 1 and catalog[0]["sha256"] == checkpoint_sha, "Selected checkpoint not verified/indexed")
    best = selected_metrics(final)
    require(abs(best["bce"]-finite(summary["primary_metric_value"])) <= 1e-12, "Summary/final selected BCE differs")
    epoch_seconds = [finite(x["epoch_duration_seconds"]) for x in epochs]
    require(all(x > 0 for x in epoch_seconds), "Invalid epoch duration")
    gate = receipt["resource_gates"]
    require(gate["resource_gate_passed"] is True, "Resource gate failed")
    fixed_path = root/"diagnostics/selected_best_fixed_training_evaluation.json"
    fixed = read_json(fixed_path) if fixed_path.exists() else None
    require(row["profile"] != "confirmation" or fixed is not None, "Confirmation missing fixed-training diagnostic")
    if fixed:
        require(fixed["checkpoint_sha256"] == checkpoint_sha and fixed["used_for_model_selection"] is False,
                "Fixed-training diagnostic does not match selected best")
    source_names = ["_SUCCESS","manifest.yaml","config.resolved.yaml","summary.json","metrics/final.json",
        "results/epoch_metrics.csv","results/per_core_epoch_metrics.csv","results/optimization_audit_endpoint.json",
        PREFLIGHT_RELATIVE_PATH,"diagnostics/train_fitted_gene_prevalence_validation.json",
        "provenance/source_files.json","provenance/artifact_checksums.json"]
    if fixed:
        source_names.append("diagnostics/selected_best_fixed_training_evaluation.json")
    row.update(final_epoch=expected_epochs, selected_best_epoch=summary["best_epoch"],
        selected_best_checkpoint_sha256=checkpoint_sha, selected_best=best,
        endpoint_epochs=wanted, endpoint_bce=value,
        endpoint_secondary={key:mean(finite(x["validation_"+key]) for x in terminal) for key in SECONDARY},
        endpoint_per_core=cores, baseline=baseline["metrics"], baseline_per_core=baseline["per_core"],
        endpoint_bce_reduction_from_gene_prior=finite(baseline["metrics"]["pooled_masked_binary_cross_entropy"])-value,
        epoch_time_seconds=sum(epoch_seconds), mean_epoch_seconds=mean(epoch_seconds),
        elapsed_run_seconds=finite(summary["duration_seconds"]), total_optimizer_updates=expected_epochs*row["optimizer_steps_per_epoch"],
        training_target_visits=208696*expected_epochs,
        peak_allocated_gib=gate["peak_allocated_bytes"]/GIB, peak_reserved_gib=gate["peak_reserved_bytes"]/GIB,
        minimum_physical_free_gib=gate["minimum_sampled_physical_free_bytes"]/GIB,
        peak_shared_cgroup_gib=gate["maximum_sampled_cgroup_memory_bytes"]/GIB,
        fixed_training=None if fixed is None else {"metrics":{k:v for k,v in fixed["metrics"].items()
            if isinstance(v,(int,float))}, "train_fitted_gene_prevalence_reference":fixed["train_fitted_gene_prevalence_reference"],
            "bce_reduction_from_train_fitted_gene_prevalence":fixed["bce_reduction_from_train_fitted_gene_prevalence"],
            "mask_contract":fixed["mask_contract"]},
        source_hashes={name:sha256_file(root/name) for name in source_names},
        compatibility={"source_configuration_sha256":receipt["source_configuration_sha256"],
            "frozen_task_contract_sha256":receipt["frozen_task_contract_sha256"],
            "executable_source_sha256":canonical_sha256(read_json(root/"provenance/source_files.json")),
            "validation_mask_sha256":baseline_mask_sha,"prior_probability_sha256":baseline["probability_values_sha256"]})
    if row["profile"] == "confirmation":
        previous = mean(finite(epochs[i-1][METRIC]) for i in range(25,29))
        row["late_bce_improvement"] = previous-value
        row["endpoint_unconverged"] = previous-value > .0005
    elapsed = 0.
    history = []
    for entry in epochs:
        elapsed += finite(entry["epoch_duration_seconds"])
        history.append({"run_id":run_id,"epoch":int(entry["global_epoch"]),"optimizer_steps":int(entry["cumulative_optimizer_updates"]),
            "training_target_visits":int(entry["global_epoch"])*208696,"epoch_time_seconds":elapsed,
            "train_online_bce":finite(entry["train_pooled_masked_binary_cross_entropy"]),"validation_bce":finite(entry[METRIC]),
            "learning_rate":finite(entry["learning_rate"]),
            **{key:finite(entry["validation_"+key]) for key in SECONDARY}})
    return row,history


def load_campaign(paths,registry):
    with registry.connect() as connection:
        records = [dict(row) for row in connection.execute(
            "SELECT run_id,status,artifact_path,failure_category,config_json FROM runs WHERE campaign_id=? ORDER BY run_id",(CAMPAIGN,))]
    completed,other,history = [],[],[]
    for record in records:
        if record["status"] == "completed":
            row,curve = load_completed(record,paths,registry)
            completed.append(row); history.extend(curve)
            continue
        config = json.loads(record["config_json"])
        row = {"run_id":record["run_id"],"status":record["status"],**descriptor(config),
               "failure_category":record["failure_category"],"source_hashes":{}}
        if record["status"] in {"failed","cancelled","pruned"}:
            root = RunArchive.artifact_path_for(record["run_id"],paths)
            require(Path(record["artifact_path"]).resolve() == root.resolve(), "Failed run path differs")
            verify_run_bundle(root)
            row["source_hashes"] = {name:sha256_file(root/name) for name in
                ("_FAILED","_PRUNED","manifest.yaml","summary.json","provenance/artifact_checksums.json","logs/stderr.log")
                if (root/name).is_file()}
            error = (root/"logs/stderr.log").read_text(errors="replace") if (root/"logs/stderr.log").is_file() else ""
            if re.search(r"non[- ]?finite (?:loss|gradient|parameter)|(?:loss|gradient|parameter)[^\n]{0,80}non[- ]?finite",error,re.I):
                row["diagnosed_failure_type"] = "nonfinite"
            elif re.search(r"CUDA out of memory|resource gate failed|(?:headroom|free disk|cgroup memory)[^\n]{0,120}(?:insufficient|below|requires|failed|exceed)",error,re.I):
                row["diagnosed_failure_type"] = "resource"
            else:
                row["diagnosed_failure_type"] = "implementation_or_unknown"
        other.append(row)
    if completed:
        require(len({canonical_sha256(row["compatibility"]) for row in completed}) == 1,
                "Audit runs have mismatched executable source, source config, masks, prior or frozen contract")
    return completed,other,history


def read_parent(path,expected_stage):
    require(path.name == "summary.json", "Parent must be an immutable report summary.json")
    root = path.parent
    manifest = read_json(root/"manifest.json")
    for name,digest in manifest["files"].items():
        candidate = root/name
        require(candidate.resolve().is_relative_to(root.resolve()) and sha256_file(candidate)==digest,
                "Parent report checksum mismatch")
    require("summary.json" in manifest["files"], "Parent manifest does not bind summary")
    parent = read_json(path)
    require(parent["schema"] == "so2_bce_optimization_audit_report_v1" and parent.get("campaign_id") == CAMPAIGN
            and parent["stage"] == expected_stage
            and parent["required_coverage_complete"] is True, "Parent stage is incomplete or incompatible")
    return parent,{"path":str(path),"sha256":sha256_file(path),"manifest_sha256":sha256_file(root/"manifest.json")}


def historical_reference(paths,registry):
    root = RunArchive.artifact_path_for(SOURCE_RUN,paths)
    record = registry.get_run(SOURCE_RUN)
    require(record and record["status"]=="completed" and Path(record["artifact_path"]).resolve()==root.resolve(),
            "Historical reference is not a registered completed bundle")
    verify_run_bundle(root)
    summary = read_json(root/"summary.json")
    return {"run_id":SOURCE_RUN,"selected_best_epoch":summary["best_epoch"],"final_epoch":summary["final_epoch"],
            "selected_best_bce":finite(summary["primary_metric_value"]),"used_for_matched_stage_selection":False,
            "source_hashes":{name:sha256_file(root/name) for name in
                ("_SUCCESS","manifest.yaml","summary.json","config.resolved.yaml","provenance/artifact_checksums.json")}}


def expected_rows(stage,parent):
    profile = "confirmation" if stage == "confirmation" else "screen"
    if stage == "grid":
        settings = [{**INCUMBENT,"target_batch_size":batch,"learning_rate":lr}
                    for batch in TARGET_BATCH_SIZES for lr in GRID_LEARNING_RATES]
        settings.append({**INCUMBENT,"batch_mode":"core_pair","target_batch_size":None})
    elif stage == "regularization":
        winner = parent["grid_selected_setting"]
        settings = [winner,{**winner,"dropout":0.},{**winner,"dropout":.3},{**winner,"weight_decay":.01}]
        if parent["boundary_setting"]:
            settings.append(parent["boundary_setting"])
    else:
        settings = [INCUMBENT,parent["confirmation_comparator_setting"]]
    return [{"profile":profile,**setting,"seed":seed,"fold":0}
            for setting in settings for seed in ((0,1) if stage=="confirmation" else (0,))]


def proximity(row):
    return (abs(math.log2(row["target_batch_size"]/1024)), abs(math.log10(row["learning_rate"]/1e-4)),
            abs(row["dropout"]-.1),abs(math.log10(row["weight_decay"]/1e-5)))


def rank(rows):
    """Predeclared practical tie: terminal BCE within .0001, then measured time."""
    require(rows,"No completed candidates to rank")
    low = min(row["endpoint_bce"] for row in rows)
    tied = [row for row in rows if row["endpoint_bce"] <= low+.0001]
    return min(tied,key=lambda row:(row["epoch_time_seconds"],proximity(row),row["run_id"]))


def compare_confirmation(rows):
    incumbent = {row["seed"]:row for row in rows if setting(row)==INCUMBENT}
    comparator = {row["seed"]:row for row in rows if setting(row)!=INCUMBENT}
    require(set(incumbent)==set(comparator)=={0,1},"Confirmation needs both seeds for each distinct variant")
    effects = []
    for seed in (0,1):
        before,after = incumbent[seed],comparator[seed]
        effects.append({"seed":seed,"incumbent_run_id":before["run_id"],"comparator_run_id":after["run_id"],
            "bce_reduction":before["endpoint_bce"]-after["endpoint_bce"],
            "f1_increase":after["endpoint_secondary"]["detection_f1"]-before["endpoint_secondary"]["detection_f1"],
            "per_core_bce_reduction":{alias:before["endpoint_per_core"][alias]["bce"]-after["endpoint_per_core"][alias]["bce"]
                                      for alias in before["endpoint_per_core"]}})
    values = [row["bce_reduction"] for row in effects]
    supported = mean(values)>=.002 and all(value>0 for value in values)
    unconverged = any(row["endpoint_unconverged"] for row in rows)
    return {"paired_seed_effects":effects,"mean_paired_bce_reduction":mean(values),
        "paired_seed_effect_range":[min(values),max(values)],"material_gain_supported":supported,
        "fixed_budget_outcome":"supported" if supported else "negative",
        "any_endpoint_unconverged":unconverged,
        "eventual_performance_outcome":"inconclusive" if unconverged else "not_established_beyond_fixed_budget",
        "confidence_interval":None,"confidence_interval_reason":"One held-out donor; two optimization seeds do not supply independent biological replication."}


def stage_decision(stage,selected,parent,coverage_complete,completed):
    if not coverage_complete:
        return {"selection_status":"incomplete_no_selection"}
    if stage == "confirmation":
        if len(selected)!=4:
            return {"selection_status":"complete_with_failed_confirmation", "material_gain_supported":False,
                    "fixed_budget_outcome":"inconclusive", "eventual_performance_outcome":"inconclusive",
                    "paired_seed_effects":[], "mean_paired_bce_reduction":None,
                    "any_endpoint_unconverged":any(row.get("endpoint_unconverged",False) for row in selected)}
        return {"selection_status":"complete",**compare_confirmation(selected)}
    if stage == "grid":
        candidates = [row for row in selected if row["batch_mode"]=="target_window"]
        if not candidates:
            return {"selection_status":"all_candidates_failed", "grid_selected_run_id":None,
                    "grid_selected_setting":None,"grid_bce_reduction_from_incumbent":None,"boundary_setting":None}
        winner = rank(candidates)
        incumbent = next((row for row in candidates if setting(row)==INCUMBENT),None)
        gain = incumbent["endpoint_bce"]-winner["endpoint_bce"] if incumbent else None
        boundary = None
        if gain is not None and gain>=.002 and winner["learning_rate"] in (min(GRID_LEARNING_RATES),max(GRID_LEARNING_RATES)):
            boundary = {**setting(winner),"learning_rate":1e-5 if winner["learning_rate"]==min(GRID_LEARNING_RATES) else 1e-3}
        paired = next((row for row in selected if row["batch_mode"]=="core_pair"),None)
        early = None
        if paired and incumbent:
            early = {"paired_core_run_id":paired["run_id"],"target_window_incumbent_run_id":incumbent["run_id"],
                "endpoint_epochs":[7,8],"bce_reduction_with_target_windows":paired["endpoint_bce"]-incumbent["endpoint_bce"],
                "f1_increase_with_target_windows":incumbent["endpoint_secondary"]["detection_f1"]-paired["endpoint_secondary"]["detection_f1"],
                "paired_core_optimizer_steps":48,"target_window_optimizer_steps":1672,
                "training_target_visits_each":1669568,
                "interpretation_limit":"Matched exposure isolates the combined target-window/update-regimen change; unequal steps cannot establish equal-step or eventual superiority."}
        return {"selection_status":"complete","grid_selected_run_id":winner["run_id"],"grid_selected_setting":setting(winner),
                "grid_bce_reduction_from_incumbent":gain,"boundary_setting":boundary,"early_update_regime_comparison":early}
    # The regularization stage retains every initial grid candidate in the
    # finalist pool; an incumbent win still requires a distinct comparator.
    grid_ids = set(parent["stage_run_ids"])
    candidates = [row for row in completed if (row["run_id"] in grid_ids or row in selected)
                  and row["profile"]=="screen" and row["batch_mode"]=="target_window"]
    winner = rank(candidates)
    comparator = rank([row for row in candidates if setting(row)!=INCUMBENT]) if setting(winner)==INCUMBENT else winner
    return {"selection_status":"complete","selected_finalist_run_id":winner["run_id"],"selected_finalist_setting":setting(winner),
            "incumbent_won_screen":setting(winner)==INCUMBENT,
            "confirmation_comparator_run_id":comparator["run_id"],"confirmation_comparator_setting":setting(comparator)}


def write_csv(path,rows,fields):
    with path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        writer.writerows({key:row.get(key) for key in fields} for row in rows)


def plot_curves(root,history,rows,*,filename="validation_curves.png",title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(15,4.5),layout="constrained")
    if not rows:
        for axis,label in zip(axes,("Epoch view","Optimizer-update view","Elapsed-time view")):
            axis.set_axis_off()
            axis.set_title(label)
            axis.text(.5,.5,"Awaiting completed stage runs\nSelection remains pending",ha="center",va="center",
                      transform=axis.transAxes,color="#475569")
        fig.savefig(root/filename,dpi=160); plt.close(fig)
        return
    for row in rows:
        curve = [x for x in history if x["run_id"]==row["run_id"]]
        label = ("pair" if row["batch_mode"]=="core_pair" else f"B{row['target_batch_size']}")
        label += f" lr{row['learning_rate']:g} d{row['dropout']:g} wd{row['weight_decay']:g} s{row['seed']}"
        for axis,key in zip(axes,("epoch","optimizer_steps","epoch_time_seconds")):
            axis.plot([x[key]/(60 if key=="epoch_time_seconds" else 1) for x in curve],
                      [x["validation_bce"] for x in curve],label=label,linewidth=1.25)
    for axis,label in zip(axes,("Epochs (matched target exposure)","Optimizer updates","Epoch time including validation (min)")):
        axis.set(xlabel=label,ylabel="Fixed-mask validation BCE"); axis.grid(alpha=.2)
        if rows:
            axis.axhline(rows[0]["baseline"]["pooled_masked_binary_cross_entropy"],color="black",linestyle=":",
                         linewidth=1,label="Train-fitted gene prior")
    if rows:
        axes[-1].legend(fontsize=6,loc="best")
    if title:
        fig.suptitle(title)
    fig.savefig(root/filename,dpi=160); plt.close(fig)


def plot_endpoint_comparison(root,stage,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if stage=="grid":
        fig,axis = plt.subplots(figsize=(6,4),layout="constrained")
        data = [[next((row["endpoint_bce"] for row in rows if row["target_batch_size"]==batch
                       and row["learning_rate"]==lr),float("nan")) for lr in GRID_LEARNING_RATES]
                for batch in TARGET_BATCH_SIZES]
        any_scores = any(math.isfinite(value) for values in data for value in values)
        if any_scores:
            shown = axis.imshow(data,cmap="viridis_r",aspect="auto")
        else:
            axis.set(xlim=(-.5,2.5),ylim=(2.5,-.5))
            axis.set_facecolor("#f1f5f9")
        axis.set(xticks=range(3),xticklabels=[f"{lr:g}" for lr in GRID_LEARNING_RATES],
                 yticks=range(3),yticklabels=TARGET_BATCH_SIZES,xlabel="Initial learning rate",ylabel="Target batch cap",
                 title="Mean validation BCE, epochs 7–8")
        for i in range(3):
            for j in range(3):
                if math.isfinite(data[i][j]):
                    axis.text(j,i,f"{data[i][j]:.5f}",ha="center",va="center",color="white",
                              bbox={"facecolor":"black","alpha":.5,"edgecolor":"none"})
                else:
                    axis.text(j,i,"No finalized\nscore",ha="center",va="center",color="#475569",fontsize=9)
        if any_scores:
            fig.colorbar(shown,ax=axis,label="BCE (lower better)")
    elif stage=="confirmation":
        fig,axis = plt.subplots(figsize=(6,4),layout="constrained")
        for seed in (0,1):
            left = next((row for row in rows if row["seed"]==seed and setting(row)==INCUMBENT),None)
            right = next((row for row in rows if row["seed"]==seed and setting(row)!=INCUMBENT),None)
            if left and right:
                axis.plot([0,1],[left["endpoint_bce"],right["endpoint_bce"]],"o-",label=f"Seed {seed}")
        axis.set(xticks=[0,1],xticklabels=["Incumbent","Selected comparator"],ylabel="Mean validation BCE, epochs 29–32",
                 title="Paired optimization seeds on the same held-out donor")
        if axis.lines:
            axis.legend()
        axis.grid(alpha=.2)
    else:
        return False
    fig.savefig(root/"endpoint_comparison.png",dpi=160); plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("grid","regularization","confirmation"),required=True)
    parser.add_argument("--parent-report",type=Path)
    parser.add_argument("--version",required=True)
    parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--accepted-failure-run",action="append",default=[],metavar="RUN_ID",
        help="Explicitly accept a diagnosed immutable nonfinite/resource failure as resolved coverage, never as a ranked score")
    args = parser.parse_args(argv)
    require(args.version.replace("_","").replace("-","").isalnum(),"Unsafe report version")
    require((args.stage!="grid")==bool(args.parent_report),"Later stages require --parent-report; grid must omit it")
    paths = current_paths(); registry = Registry(paths.state_root/"tracking/bagm.sqlite3",initialize=False)
    parent,parent_source = read_parent(args.parent_report,"grid" if args.stage=="regularization" else "regularization") if args.parent_report else (None,None)
    completed,other,history = load_campaign(paths,registry)
    require(len(set(args.accepted_failure_run))==len(args.accepted_failure_run),"Duplicate accepted failure ID")
    accepted_failures = []
    for run_id in args.accepted_failure_run:
        matches = [row for row in other if row["run_id"]==run_id and row["status"]=="failed"]
        require(len(matches)==1 and matches[0].get("diagnosed_failure_type") in {"nonfinite","resource"},
                "Accepted failure must have verified numerical/resource evidence; coding/config/infrastructure failures remain blockers")
        accepted_failures.extend(matches)
    expected = expected_rows(args.stage,parent)
    selected,coverage = [],[]
    for expected_row in expected:
        matches = [row for row in completed if run_key(row)==run_key(expected_row)]
        require(len(matches)<=1,"Multiple completed attempts for one planned setting; explicit audit required before selection")
        attempts = [row for row in completed+other if run_key(row)==run_key(expected_row)]
        failures = [row for row in accepted_failures if run_key(row)==run_key(expected_row)]
        active = [row for row in attempts if row["status"] not in {"completed","failed","cancelled","pruned"}]
        coverage.append({**expected_row,"completed_run_id":matches[0]["run_id"] if matches else None,
                         "accepted_failure_run_ids":[row["run_id"] for row in failures],
                         "resolved":bool(matches or failures) and not active,
                         "attempts":[{"run_id":r["run_id"],"status":r["status"],"attempt":r["attempt"]} for r in attempts]})
        selected.extend(matches)
    require(all(any(run_key(row)==run_key(x) for x in expected) for row in accepted_failures),
            "Accepted failure is outside this stage's declared matrix")
    coverage_complete = all(row["resolved"] for row in coverage)
    require(coverage_complete or args.allow_incomplete,"Required matrix is incomplete; use --allow-incomplete for a clearly provisional report")
    decision = stage_decision(args.stage,selected,parent,coverage_complete,completed)
    report = {"schema":"so2_bce_optimization_audit_report_v1","campaign_id":CAMPAIGN,"stage":args.stage,
        "created_at":datetime.now(timezone.utc).isoformat(),"required_coverage_complete":coverage_complete,
        "complete":args.stage=="confirmation" and coverage_complete,"budget_branch":"full",
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=paths.project_root,text=True).strip(),
        "question":"Do target-cell batches, learning rate or regularization improve SO2 binary-expression prediction under controlled exposure budgets?",
        "estimand":"Target-isolated partial-gene binary-expression reconstruction on the fixed held-out donor.",
        "maximum_defensible_claim":"Exploratory fixed-split optimization behavior; neither count-magnitude reconstruction nor graph-specific or biological causal evidence.",
        "strongest_alternative":"Additional updates accelerate early fitting without improving eventual predictive performance; validation tuning may exploit this single donor.",
        "controls":["Fresh single-GPU paired-core reference at matched target exposure", "Fixed masks, graph, data and permitted covariates",
                    "Train-fitted per-gene prevalence baseline", "One-factor regularization controls", "Fresh paired optimization seeds in confirmation"],
        "expected_coverage":coverage,"expected_seeds":[0,1] if args.stage=="confirmation" else [0],"expected_folds":[0],
        "included_seeds":sorted({row["seed"] for row in selected}),"included_folds":sorted({row["fold"] for row in selected}),
        "stage_run_ids":[row["run_id"] for row in selected],"completed_runs":completed,"other_attempts":other,
        "accepted_failure_runs":[{"run_id":row["run_id"],"failure_type":row["diagnosed_failure_type"],
            "interpretation":"numerical instability at the tested setting" if row["diagnosed_failure_type"]=="nonfinite"
                             else "operational infeasibility; no predictive inferiority established"} for row in accepted_failures],
        "parent_report":parent_source,"historical_long_trained_reference":historical_reference(paths,registry),**decision,
        "limitations":["Exploratory validation tuning on one held-out donor; no independent test cohort or patient confidence interval.",
            "Seeds measure optimization variability. The two validation cores do not establish patient replication.",
            "Across batch sizes equal epochs match target exposure, while optimizer updates differ; within batch size LR comparisons match both.",
            "Selected-best replay metrics and fixed-training diagnostics are separate from the prespecified terminal endpoint.",
            "Online training BCE uses changing masks and dropout; the raw train-validation gap is not a controlled fitting estimate.",
            "Model AP/AUROC use 4096-bin histograms; the train-fitted per-gene reference has exact score-tie handling.",
            "Timing includes validation in epoch histories; startup, checkpoint I/O, final replay and diagnostic work additionally contribute to run duration.",
            "No graph-specific predictive gain, faithfulness, biological null, independent replication, mechanism or causality is tested."]}
    if args.stage!="grid":
        report["early_update_regime_comparison"] = parent.get("early_update_regime_comparison")
    output = paths.report_root/"analyses/so2_bce_optimization_audit"/args.version
    require(not output.exists(),"Immutable report version already exists")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+".tmp-"+uuid.uuid4().hex); temporary.mkdir()
    (temporary/"summary.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    fields = ["run_id","profile",*SETTING_FIELDS,"seed","fold","attempt","physical_gpu","final_epoch",
        "endpoint_bce","selected_best_epoch","total_optimizer_updates","training_target_visits","epoch_time_seconds",
        "mean_epoch_seconds","elapsed_run_seconds","peak_allocated_gib","peak_reserved_gib","minimum_physical_free_gib",
        "peak_shared_cgroup_gib","endpoint_bce_reduction_from_gene_prior","late_bce_improvement","endpoint_unconverged"]
    write_csv(temporary/"runs.csv",completed,fields)
    secondary_rows = [{"run_id":row["run_id"],"evaluation_point":point,**values}
                      for row in completed for point,values in
                      (("terminal_epoch_mean",row["endpoint_secondary"]),("selected_best_checkpoint",row["selected_best"]))]
    write_csv(temporary/"secondary_metrics.csv",secondary_rows,["run_id","evaluation_point",*SECONDARY])
    write_csv(temporary/"attempts.csv",[{**row,"status":"completed"} for row in completed]+other,
              ["run_id","status","failure_category","profile",*SETTING_FIELDS,"seed","fold","attempt"])
    write_csv(temporary/"history.csv",history,list(history[0]) if history else ["run_id","epoch"])
    core_rows = [{"run_id":row["run_id"],"core_alias":alias,**values}
                 for row in completed for alias,values in row["endpoint_per_core"].items()]
    write_csv(temporary/"per_core_endpoints.csv",core_rows,["run_id","core_alias","bce",*SECONDARY])
    plot_curves(temporary,history,selected)
    completed_smokes = [row for row in completed if row["profile"]=="smoke"]
    if not selected and completed_smokes:
        plot_curves(temporary,history,completed_smokes,filename="smoke_validation_curves.png",
                    title="Completed readiness smokes: excluded from scientific selection")
    endpoint_figure = plot_endpoint_comparison(temporary,args.stage,selected)
    lines = ["# SO2 BCE optimization audit: "+args.stage,"",
        f"Required completed coverage: **{len(selected)}/{len(expected)}**. Endpoint selection: **{decision['selection_status']}**.","",
        "The audit tests binary expression presence under target-isolated partial-gene masking. It separates the early effect of changing the update regimen from longer fixed-budget hyperparameter tuning; all validation comes from one held-out donor.","",
        "![Validation BCE against epochs, optimizer steps and epoch time](validation_curves.png)","",
        "| Mode / batch | LR | Dropout | WD | Seed | Terminal BCE | Best BCE (epoch) | Steps | Epoch min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in selected:
        lines.append(f"| {row['batch_mode']} / {row['target_batch_size']} | {row['learning_rate']:g} | {row['dropout']:g} | {row['weight_decay']:g} | {row['seed']} | {row['endpoint_bce']:.8f} | {row['selected_best']['bce']:.8f} ({row['selected_best_epoch']}) | {row['total_optimizer_updates']} | {row['epoch_time_seconds']/60:.1f} |")
    if endpoint_figure:
        lines += ["", "![Prespecified terminal endpoint comparison](endpoint_comparison.png)"]
    if not selected and completed_smokes:
        lines += ["", "Completed smoke runs verify readiness and are excluded from screen ranking:", "",
                  "![Completed two-epoch readiness smoke curves](smoke_validation_curves.png)"]
    if args.stage=="grid" and coverage_complete:
        gain = decision["grid_bce_reduction_from_incumbent"]
        lines += ["",f"Grid selection: `{decision['grid_selected_run_id']}`. BCE reduction from matched incumbent: {gain if gain is None else format(gain,'.8f')}."]
    elif args.stage=="regularization" and coverage_complete:
        lines += ["",f"Finalist: `{decision['selected_finalist_run_id']}`. Distinct confirmation comparator: `{decision['confirmation_comparator_run_id']}`."]
    elif args.stage=="confirmation" and coverage_complete:
        effect = decision["mean_paired_bce_reduction"]
        lines += ["",f"Mean paired-seed BCE reduction: **{effect if effect is None else format(effect,'.8f')}**; material-gain criterion: **{decision['material_gain_supported']}**.",
            f"Any endpoint still improving by >0.0005: **{decision['any_endpoint_unconverged']}**. Eventual-performance conclusion: **{decision['eventual_performance_outcome']}**.","",
            *[f"- Seed {x['seed']}: BCE reduction {x['bce_reduction']:.8f}; core effects {json.dumps(x['per_core_bce_reduction'],sort_keys=True)}." for x in decision["paired_seed_effects"]]]
    lines += ["", "Prespecified endpoints are means of epochs 7–8 for screens and 29–32 for confirmation. Best-checkpoint metrics are reported separately. Positive BCE reductions favor the comparator. Historical selected-best BCE 0.24017953 follows 129 epochs and is not a matched-budget control.","",
        "Train-fitted per-gene prevalence metrics, fixed-training diagnostics when required, per-core directions, all attempts and source checksums are in summary.json and the companion tables.","",
        f"Other attempts (including preserved failures and unfinished runs): {len(other)}.","",*['- '+x for x in report["limitations"]]]
    if selected:
        prior = selected[0]["baseline"]["pooled_masked_binary_cross_entropy"]
        lines += ["", f"Train-fitted per-gene prior BCE on the same validation masks: **{prior:.8f}**."]
    if report.get("early_update_regime_comparison"):
        early = report["early_update_regime_comparison"]
        lines += ["", f"At eight matched epochs, B=1,024 target windows reduce terminal BCE by {early['bce_reduction_with_target_windows']:.8f} relative to paired-core updates; F1 change is {early['f1_increase_with_target_windows']:.8f}. Each run sees 1,669,568 targets, with 1,672 versus 48 optimizer steps.",
            "This is evidence about early optimization under different update regimens. It is separate from the 32-epoch tuned-versus-incumbent contrast. A negative tuning result does not erase an observed early update-regimen benefit; early benefit alone does not establish eventual BCE or F1 superiority."]
    if other:
        lines += ["", "Preserved failed or unfinished attempts:", "",
                  *[f"- `{row['run_id']}`: {row['status']}; category {row.get('failure_category') or 'none'}." for row in other]]
    if report["accepted_failure_runs"]:
        lines += ["", "Explicitly accepted failure outcomes:", "",
                  *[f"- `{row['run_id']}`: {row['interpretation']}." for row in report["accepted_failure_runs"]]]
    if any(row["fixed_training"] for row in selected):
        lines += ["", "Selected-best fixed-training diagnostics (evaluation mode, epoch-one training masks; diagnostic only):", "",
            "| Run | Fixed training BCE | Training BCE reduction from gene prior | Selected-best validation BCE |",
            "|---|---:|---:|---:|"]
        for row in selected:
            fixed = row["fixed_training"]
            if fixed:
                lines.append(f"| `{row['run_id']}` | {fixed['metrics']['pooled_masked_binary_cross_entropy']:.8f} | {fixed['bce_reduction_from_train_fitted_gene_prevalence']:.8f} | {row['selected_best']['bce']:.8f} |")
    (temporary/"report.md").write_text("\n".join(lines)+"\n")
    shutil.copyfile(Path(__file__),temporary/"report_builder.py")
    manifest = {"schema":"checksummed_local_report_v1","files":{file.name:sha256_file(file)
                 for file in sorted(temporary.iterdir()) if file.is_file()}}
    (temporary/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    _rename_no_replace(temporary,output)
    print(json.dumps({"report":str(output),"stage":args.stage,"required_coverage_complete":coverage_complete,**decision},sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
