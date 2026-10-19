#!/usr/bin/env python3
"""Publish an aggregate-only human report from a verified BCE execution profile."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"src"))
import yaml

from scripts.analysis.so2_bce_execution_profile_contract import CAMPAIGN, PARENT_RUN, SCRIPT_PATHS, validate_configuration
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, _rename_no_replace, verify_run_bundle

GIB=1024**3
PROFILE="diagnostics/execution_operator_profile.json"
OBSERVATION="diagnostics/execution_profile_summary.json"
RECEIPT="diagnostics/execution_profile_preflight.json"
RESOURCES="diagnostics/execution_profile_resource_samples.jsonl"
METRIC="val/target_only_masked_binary_cross_entropy"
SCALARS=("train_pooled_masked_binary_cross_entropy","validation_pooled_masked_binary_cross_entropy",
    "validation_detection_brier_score","validation_detection_f1","validation_detection_auprc_histogram_4096",
    "validation_detection_auroc_histogram_4096")
COUNTS=("train_target_cells","validation_target_cells","train_masked_entries","validation_masked_entries",
    "train_positive_entries","validation_positive_entries","optimizer_updates","cumulative_optimizer_updates",
    "validation_detection_true_positive","validation_detection_true_negative",
    "validation_detection_false_positive","validation_detection_false_negative")
MASKS=("core_order_json","training_coverage_receipts_json","validation_coverage_receipts_json")


def require(value,message):
    if not value:raise ValueError(message)


def number(value):
    value=float(value)
    require(math.isfinite(value),"Nonfinite scalar evidence")
    return value


def replay_close(a,b):return math.isclose(number(a),number(b),rel_tol=1e-6,abs_tol=1e-6)


def read_json(path):return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:return list(csv.DictReader(handle))


def check_parent_epoch(actual,parent,stored):
    require(stored["passed"] is True and stored["absolute_scalar_tolerance"]==1e-6
            and stored["integer_counts_and_mask_receipts_exact"] is True,"Parent equivalence receipt failed")
    for field in SCALARS:
        a,b=number(actual[field]),number(parent[field])
        require(abs(a-b)<=1e-6,"Parent scalar disagreement: "+field)
        require(stored["scalar_comparisons"][field]=={"observed":a,"parent":b,"absolute_difference":abs(a-b)},
                "Parent comparison receipt differs from epoch evidence")
    require(all(int(actual[key])==int(parent[key]) for key in COUNTS),"Parent integer count mismatch")
    require(all(json.loads(actual[key])==json.loads(parent[key]) for key in MASKS),"Parent mask/order mismatch")


def validate_observation(profile,observation):
    require(profile["schema"]=="so2_bce_execution_operator_profile_v1"
            and observation["schema"]=="so2_bce_execution_profile_summary_v1","Unknown profiling schema")
    require(profile["raw_events_or_tensor_metadata_exported"] is False,"Unrestricted profiler export is prohibited")
    indices=profile["profiled_update_indices"]
    require(indices in ([3],[3,4],[3,4,5]),"Unexpected profiler capture window")
    require(profile["profile_coverage_complete"] is (len(indices)==3),"Capture completeness flag differs")
    require(len(indices)==3 or profile["capture_limited_by_budget"] is True,"Unexplained partial capture")
    updates=observation["all_update_timings"]
    require([row["update"] for row in updates]==list(range(1,210)),"Canonical 209 updates are incomplete")
    require(sum(row["target_cells"] for row in updates)==208696
            and all(0<row["target_cells"]<=1024 and number(row["wall_seconds"])>0 for row in updates),
            "Update target coverage/timing is invalid")
    measured=[row for row in updates if row["profiled"]]
    adjacent=[row for row in updates if indices[-1]<row["update"]<=indices[-1]+3]
    require([row["update"] for row in measured]==indices and measured==observation["profiled_updates"]
            and adjacent==observation["adjacent_unprofiled_updates"],"Stored timing windows differ")
    mean_profile=sum(row["wall_seconds"] for row in measured)/len(measured)
    mean_adjacent=sum(row["wall_seconds"] for row in adjacent)/len(adjacent)
    for key,value in (("profiled_mean_update_seconds",mean_profile),("adjacent_unprofiled_mean_update_seconds",mean_adjacent),
                      ("descriptive_profiled_to_adjacent_wall_ratio",mean_profile/mean_adjacent)):
        require(abs(number(observation[key])-value)<=1e-12,"Stored timing calculation differs: "+key)
    require(number(profile["capture_wall_seconds"])>0,"Empty capture duration")
    for key,other in (("startup_wall_seconds","profile_startup_seconds"),
                      ("stop_and_aggregate_wall_seconds","profile_stop_and_aggregate_seconds")):
        require(number(profile[key])>=0 and profile[key]==observation[other],"Profiler boundary duration differs")
    operators=profile["aggregate_operator_rows"]
    require(operators and any(number(row["self_device_microseconds"])>0 for row in operators),"No CUDA timing evidence")
    for row in operators:
        require(row["device_type"] in {"CPU","CUDA"} and row["kind"] in {"operator","named_range"}
                and int(row["calls"])>0,"Invalid aggregate operator identity")
        require(row["kind"]==("named_range" if row["operator"].startswith("execution/") else "operator"),"Range classification differs")
        require(re.fullmatch(r"[0-9a-f]{64}",row["operator_full_label_sha256"])
            and len(row["operator"])<=500 and type(row["operator_label_truncated"]) is bool
            and type(row["operator_label_control_characters_normalized"]) is bool,"Invalid static operator label identity")
        if not row["operator_label_truncated"] and not row["operator_label_control_characters_normalized"]:
            require(hashlib.sha256(row["operator"].encode()).hexdigest()==row["operator_full_label_sha256"],"Operator label checksum differs")
        require(all(number(row[key])>=0 for key in ("self_cpu_microseconds","inclusive_cpu_microseconds",
                "self_device_microseconds","inclusive_device_microseconds")),"Negative operator time")
    for row in observation["named_wall_ranges"]:
        require(int(row["calls"])>0 and 0<=number(row["maximum_call_wall_seconds"])<=number(row["inclusive_blocking_wall_seconds"]),
                "Invalid named wall range")
    return {"capture_complete":len(indices)==3,"profiled_update_indices":indices,
            "profiled_mean_seconds":mean_profile,"adjacent_mean_seconds":mean_adjacent,
            "descriptive_profiled_to_adjacent_ratio":mean_profile/mean_adjacent}


def operator_groups(rows):
    """Keep CPU attribution and device kernels separate: their device times overlap."""
    cpu=[row for row in rows if row["device_type"]=="CPU" and row["kind"]=="operator"]
    cuda=[row for row in rows if row["device_type"]=="CUDA" and row["kind"]=="operator"]
    return {"cpu_self":sorted(cpu,key=lambda row:-row["self_cpu_microseconds"]),
        "cpu_attributed_device":sorted([row for row in cpu if row["self_device_microseconds"]>0],key=lambda row:-row["self_device_microseconds"]),
        "device_kernels":sorted(cuda,key=lambda row:-row["self_device_microseconds"]),
        "named_ranges":[row for row in rows if row["kind"]=="named_range"],
        "scalar_sync_copy_candidates":[row for row in cpu if any(part in row["operator"].lower()
            for part in ("item","local_scalar_dense","synchronize","memcpy","copy_","cudalaunch"))]}


def resource_summary(receipt,samples):
    require(receipt["schema"]=="so2_bce_execution_profile_preflight_v1" and receipt["passed"] is True
        and receipt["completed_epochs"]==1 and receipt["coverage_verified"] is True
        and receipt["finite_gradients"] is True and receipt["optimizer_updates_per_global_epoch"]==209,
        "Profiling correctness gate failed")
    gate=receipt["resource_gates"]
    require(gate["resource_gate_passed"] is True and receipt["minimum_free_disk_gib"]==5,"Resource/disk contract differs")
    require(0<gate["peak_allocated_bytes"]<=gate["peak_reserved_bytes"]<=22*GIB
        and gate["headroom_bytes"]>=2*GIB and gate["headroom_bytes"]<=gate["minimum_sampled_physical_free_bytes"]
        and gate["maximum_sampled_cgroup_memory_bytes"]<160*GIB,"Declared resource gate failed")
    physical=[row for row in samples if "physical_memory_used_bytes" in row and "physical_memory_free_bytes" in row]
    require(physical and min(row["physical_memory_free_bytes"] for row in physical)>=2*GIB,"Physical headroom samples failed")
    host=max(int(row["cgroup"]["memory.current"]) for row in samples)
    require(host<=gate["maximum_sampled_cgroup_memory_bytes"]<160*GIB,"Shared host memory gate failed")
    require(gate["minimum_sampled_physical_free_bytes"]<=min(row["physical_memory_free_bytes"] for row in physical),
            "Declared minimum headroom exceeds an actual sample")
    return {"peak_allocated_gib":gate["peak_allocated_bytes"]/GIB,"peak_reserved_gib":gate["peak_reserved_bytes"]/GIB,
        "peak_sampled_physical_used_gib":max(row["physical_memory_used_bytes"] for row in physical)/GIB,
        "minimum_sampled_physical_free_gib":min(row["physical_memory_free_bytes"] for row in physical)/GIB,
        "conservative_headroom_gib":gate["headroom_bytes"]/GIB,
        "maximum_sampled_shared_cgroup_gib":gate["maximum_sampled_cgroup_memory_bytes"]/GIB,
        "peak_sampled_process_rss_gib":max(max(row.get("process_peak_rss_bytes",0),
            row.get("rss",{}).get("VmHWM_bytes",0),row.get("rss",{}).get("VmRSS_bytes",0)) for row in samples)/GIB,
        "resource_gate_passed":True,"physical_sampling_is_continuous_peak":False}


def checked_bundle(run_id,paths,registry):
    record=registry.get_run(run_id)
    root=RunArchive.artifact_path_for(run_id,paths)
    require(record and record["status"]=="completed" and Path(record["artifact_path"]).resolve()==root.resolve(),
            "Run is not a registered completed canonical bundle")
    require(verify_run_bundle(root)["status"]=="success","Bundle success verification failed")
    checkpoint_sha=sha256_file(root/"checkpoints/best.ckpt")
    rows=registry.list_checkpoint_catalog(run_id=run_id,role="best",verification_status="verified")
    require(len(rows)==1 and rows[0]["sha256"]==checkpoint_sha,"Best checkpoint catalog verification differs")
    summary=read_json(root/"summary.json");replay=read_json(root/"diagnostics/final_checkpoint_reload_verification.json")
    require(record["campaign_id"]==summary["campaign_id"] and summary["run_id"]==run_id,"Run campaign/identity differs")
    require(rows[0]["best_epoch"]==summary["best_epoch"]==replay["best_epoch"]
        and rows[0]["monitored_metric"]==METRIC and rows[0]["monitored_mode"]=="min"
        and replay_close(rows[0]["monitored_value"],summary["primary_metric_value"])
        and replay_close(replay["best_validation_metric"],replay["replayed_validation_metric"])
        and number(replay["replayed_validation_metric"])==number(summary["primary_metric_value"])
        and replay["verified"] is True and replay["checkpoint_sha256"]==checkpoint_sha,
        "Checkpoint catalog/replay semantics differ")
    selected=read_csv(root/"results/epoch_metrics.csv")[summary["best_epoch"]-1]
    require(int(selected["global_epoch"])==summary["best_epoch"]
        and replay_close(selected[SCALARS[1]],replay["best_validation_metric"])
        and canonical_sha256(json.loads(selected["validation_coverage_receipts_json"]))==replay["validation_coverage_receipts_sha256"],
        "Selected epoch replay or fixed masks differ")
    return root,record,checkpoint_sha


def load_run(run_id,paths,registry):
    root,record,checkpoint_sha=checked_bundle(run_id,paths,registry)
    config=yaml.safe_load((root/"config.resolved.yaml").read_text());validate_configuration(config)
    require(record["campaign_id"]==CAMPAIGN,"Registry campaign differs")
    parent,_,parent_checkpoint_sha=checked_bundle(PARENT_RUN,paths,registry)
    ref=config["metadata"]["profile_parent"]
    require(Path(ref["artifact_path"]).resolve()==parent.resolve(),"Parent reference path differs")
    parent_names={"configuration_sha256":None,"config_file_sha256":"config.resolved.yaml","success_marker_sha256":"_SUCCESS",
        "manifest_sha256":"manifest.yaml","checksum_manifest_sha256":"provenance/artifact_checksums.json",
        "epoch_metrics_sha256":"results/epoch_metrics.csv","receipt_sha256":"diagnostics/optimization_audit_preflight.json"}
    for key,name in parent_names.items():
        actual=sha256_file(parent/name) if name else canonical_sha256(yaml.safe_load((parent/"config.resolved.yaml").read_text()))
        require(ref[key]==actual,"Parent source checksum differs: "+key)
    require(read_json(root/"provenance/parent_reference.json")==ref,"Recorded parent reference differs")
    source=read_json(root/"provenance/source_files.json")
    require(source==read_json(parent/"provenance/source_files.json") and canonical_sha256(source)==ref["code_fingerprint_sha256"],
            "Frozen training source fingerprints differ")
    metadata=config["metadata"]
    require(read_json(root/"provenance/instrumentation_files.json")==metadata["instrumentation_hashes"],"Instrumentation receipt differs")
    for name,digest in metadata["instrumentation_hashes"].items():
        require(sha256_file(root/"provenance/source_snapshot"/name)==digest,"Preserved instrumentation differs")
    require(sha256_file(root/"provenance/frozen_task_contract.md")==metadata["frozen_task_contract_sha256"],"Frozen contract differs")
    summary=read_json(root/"summary.json");final=read_json(root/"metrics/final.json")
    require(summary["run_id"]==run_id and summary["status"]=="success" and summary["final_epoch"]==summary["best_epoch"]==1
        and summary["checkpoint_reload_verified"] is True and summary["optimizer_steps"]==209,"Predictive lifecycle incomplete")
    epochs=read_csv(root/"results/epoch_metrics.csv");require(len(epochs)==1 and int(epochs[0]["global_epoch"])==1,"History budget differs")
    require(int(epochs[0]["train_target_cells"])==208696 and int(epochs[0]["validation_target_cells"])==37367,"Target coverage differs")
    require(number(final[METRIC])==number(summary["primary_metric_value"])
        ==number(record["primary_metric_value"]) and replay_close(final[METRIC],epochs[0][SCALARS[1]]),
        "Selected BCE differs between lifecycle sources")
    profile=read_json(root/PROFILE);observation=read_json(root/OBSERVATION);receipt=read_json(root/RECEIPT)
    require(all(value["run_id"]==run_id for value in (profile,observation,receipt)),"Profile run identity differs")
    timing=validate_observation(profile,observation)
    check_parent_epoch(epochs[0],read_csv(parent/"results/epoch_metrics.csv")[0],observation["parent_epoch_one_comparison"])
    samples=[json.loads(line) for line in (root/RESOURCES).read_text().splitlines() if line.strip()]
    resources=resource_summary(receipt,samples)
    require(receipt["frozen_task_contract_sha256"]==metadata["frozen_task_contract_sha256"]
        and receipt["source_configuration_sha256"]==canonical_sha256(config["source_configuration"]),"Preflight contract/source differs")
    expected_code={row["path"]:row["sha256"] for row in source["files"]}|metadata["instrumentation_hashes"]
    require(receipt["code_hashes"]==expected_code,"Receipt source code differs")
    names=["_SUCCESS","manifest.yaml","config.resolved.yaml","summary.json","metrics/final.json",
        "provenance/artifact_checksums.json","provenance/source_files.json","provenance/parent_reference.json",
        "provenance/instrumentation_files.json","provenance/frozen_task_contract.md","results/epoch_metrics.csv",
        "diagnostics/execution_profile_environment.json","diagnostics/final_checkpoint_reload_verification.json",
        "provenance/git.json","provenance/data_fingerprints.json","provenance/split_fingerprint.json",PROFILE,OBSERVATION,RECEIPT,RESOURCES,
        *("provenance/source_snapshot/"+name for name in SCRIPT_PATHS)]
    sources=[{"run_id":run_id,"bundle_path":str(root.relative_to(paths.artifact_root)),"checkpoint_sha256":checkpoint_sha,
        "files":{name:sha256_file(root/name) for name in names}},
        {"run_id":PARENT_RUN,"bundle_path":str(parent.relative_to(paths.artifact_root)),"checkpoint_sha256":parent_checkpoint_sha,
        "files":{name:sha256_file(parent/name) for name in [*filter(None,parent_names.values()),"provenance/source_files.json",
            "summary.json","diagnostics/final_checkpoint_reload_verification.json"]}}]
    return {"run_id":run_id,"sources":sources,"profile":profile,"observation":observation,"timing":timing,
        "resources":resources,"environment":read_json(root/"diagnostics/execution_profile_environment.json"),
        "seed":config["seed"],"fold":config["fold"],"attempt":config["attempt"],"classification":config["classification"],
        "epoch_seconds":number(epochs[0]["epoch_duration_seconds"]),"runner_seconds":number(summary["duration_seconds"]),
        "selected_epoch_one_validation_bce":number(final[METRIC]),"frozen_training_source_sha256":ref["code_fingerprint_sha256"]}


def save_csv(path,rows):
    if not rows:return
    with path.open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_label(row):
    label=row["operator"]
    digest=row.get("operator_full_label_sha256",hashlib.sha256(label.encode()).hexdigest())
    return label if len(label)<=65 else label[:46]+"… ["+digest[:10]+"]"


def figures(out,run,groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(19,6),layout="constrained")
    for ax,key,field,title in zip(axes,("cpu_self","cpu_attributed_device","device_kernels"),
        ("self_cpu_microseconds","self_device_microseconds","self_device_microseconds"),
        ("CPU operators: self CPU time","CPU operators: attributed device time","CUDA kernels: self device time")):
        rows=groups[key][:10][::-1]
        if rows:
            ax.barh([plot_label(row) for row in rows],[row[field]/1000 for row in rows],color="#315b86")
        else:ax.text(.5,.5,"No rows in this category",ha="center",transform=ax.transAxes)
        ax.set_title(title,fontsize=11);ax.set_xlabel("Cumulative milliseconds in capture");ax.tick_params(axis="y",labelsize=7)
    fig.suptitle("Separate attribution views — overlapping times; no wall-time fractions",fontsize=13)
    fig.savefig(out/"operators.png",dpi=150);fig.savefig(out/"operators.svg");plt.close(fig)
    ranges=sorted(run["observation"]["named_wall_ranges"],key=lambda row:row["inclusive_blocking_wall_seconds"])[-20:]
    fig,ax=plt.subplots(figsize=(12,8),layout="constrained")
    ax.barh([row["range"] for row in ranges],[row["inclusive_blocking_wall_seconds"] for row in ranges],color="#6b5185")
    ax.set_title("Observed named blocking wall ranges across the run (nested, nonadditive)")
    ax.set_xlabel("Inclusive seconds; not isolated GPU duration");ax.tick_params(axis="y",labelsize=8)
    fig.savefig(out/"wall_ranges.png",dpi=150);fig.savefig(out/"wall_ranges.svg");plt.close(fig)
    rows=run["observation"]["all_update_timings"]
    fig,ax=plt.subplots(figsize=(12,4),layout="constrained")
    ax.plot([row["update"] for row in rows],[row["wall_seconds"] for row in rows],color="#8c9198",lw=1,label="Other observed updates")
    for key,color,label in (("profiled_updates","#ba3d40","Profiled"),("adjacent_unprofiled_updates","#267c64","Adjacent unprofiled")):
        selected=run["observation"][key];ax.scatter([r["update"] for r in selected],[r["wall_seconds"] for r in selected],color=color,label=label,zorder=3)
    ax.set(xlabel="Canonical optimizer update",ylabel="Observed interval seconds",title="209 unchanged updates; windows and weights vary")
    ax.legend();fig.savefig(out/"update_times.png",dpi=150);fig.savefig(out/"update_times.svg");plt.close(fig)


def table(rows,field,limit=8):
    lines=["| Operator | Calls | Milliseconds |","|---|---:|---:|"]
    lines += [f"| `{row['operator'].replace('|','/').replace('`','')}` | {row['calls']} | {row[field]/1000:.3f} |" for row in rows[:limit]]
    return "\n".join(lines) if rows else "No recorded rows in this category."


def markdown(run,groups):
    t=run["timing"];o=run["observation"];r=run["resources"]
    return f"""# BCE execution-cost profile

Run `{run['run_id']}` completed the unchanged one-epoch workload (209 updates), fixed validation and selected-checkpoint replay. Parent epoch-one counts and masks match exactly; six scalar metrics match within 0.000001. This is an exploratory execution-cost observation, not an optimization comparison or a biological result.

The study asks which measured costs plausibly explain low useful GPU utilization. Alternatives include serial host checks and synchronization, receiver-chunk transfers and launches, repeated full-core encoder/projection work, and backward recomputation. Profiler overhead, these particular early windows and concurrent shared-host load also affect observed timing; this study does not isolate their causal contributions.

![Separate CPU and device attribution views](operators.png)

Capture covered updates {t['profiled_update_indices']} ({'complete three-update capture' if t['capture_complete'] else 'budget-limited partial capture'}). Mean profiled interval: **{t['profiled_mean_seconds']:.3f} s**; adjacent unprofiled interval: **{t['adjacent_mean_seconds']:.3f} s**; descriptive ratio **{t['descriptive_profiled_to_adjacent_ratio']:.3f}×**. Profiler startup took {o['profile_startup_seconds']:.3f} s and stopping/aggregation took {o['profile_stop_and_aggregate_seconds']:.3f} s, separately from update intervals. Different windows and changing weights prevent a causal overhead estimate.

## CPU operator self time

{table(groups['cpu_self'],'self_cpu_microseconds')}

## Device time attributed to CPU operators

{table(groups['cpu_attributed_device'],'self_device_microseconds')}

## CUDA kernel self time

{table(groups['device_kernels'],'self_device_microseconds')}

These tables are separate attribution views. CPU operators can carry device time also represented by CUDA kernel rows; nested ranges overlap; CUDA activity may overlap CPU work or other CUDA activity. No rows are summed into a GPU wall-time fraction or pie chart. The scalar/synchronization/copy candidate table is an operator-name filter for follow-up inspection, not a proven bottleneck classification.

Long static kernel names use bounded display labels with full-name SHA-256 values in the aggregate table. Shortened figure labels retain a checksum suffix to distinguish otherwise similar template names. No tensor values, shapes or raw traces are included.

![Named blocking wall ranges](wall_ranges.png)

Named ranges cover initialization, training, validation and final artifact preparation. They measure observed blocking wall duration without adding CUDA synchronization. Parent and child ranges overlap. Receiver-chunk calls include forward execution and checkpoint recomputation. Core-first update intervals exclude mask realization and staging, which have separate named ranges.

![Observed update intervals](update_times.png)

The epoch took {run['epoch_seconds']:.2f} s; the trainer's recorded run duration was {run['runner_seconds']:.2f} s. The observation clock reached {o['elapsed_through_final_artifacts_seconds']:.2f} s through final artifact preparation; this excludes subsequent archive publication/indexing. The 600 s soft observation budget was {'exceeded' if o['soft_ten_minute_budget_exceeded'] else 'not exceeded'}. These clocks cover different scopes and do not estimate an uninstrumented epoch.

Resource and correctness gates passed: allocated/reserved peaks {r['peak_allocated_gib']:.3f}/{r['peak_reserved_gib']:.3f} GiB; sampled physical GPU use {r['peak_sampled_physical_used_gib']:.3f} GiB; sampled physical free minimum {r['minimum_sampled_physical_free_gib']:.3f} GiB; conservative headroom {r['conservative_headroom_gib']:.3f} GiB. Maximum observed shared cgroup memory was {r['maximum_sampled_shared_cgroup_gib']:.3f} GiB. Physical samples are not continuous peak measurements. Device: {run['environment']['gpu']}; physical GPU {run['environment']['visible_physical_gpu']}; Torch {run['environment']['torch']}, CUDA {run['environment']['cuda']}.

Maximum defensible claim: these are measured costs of a few early updates and named scopes in one instrumented execution. They do not establish every core's typical cost, the fraction of GPU idle time explained, an attainable speedup, eventual predictive superiority, graph-specific gain or biological mechanism. A proposed implementation change needs a separate state-controlled equivalence and throughput comparison. Complete aggregate tables and checksum-bound sources accompany this report; no raw traces or cell/patient identifiers are exported.
"""


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True);parser.add_argument("--version",default="v1")
    args=parser.parse_args(argv);require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",args.version),"Unsafe report version")
    paths=current_paths();output=paths.report_root/"runtime/so2_bce_execution_profile"/args.run_id/args.version
    require(not output.exists(),"Report output already exists")
    registry=Registry(paths.state_root/"tracking/bagm.sqlite3",initialize=False)
    run=load_run(args.run_id,paths,registry);groups=operator_groups(run["profile"]["aggregate_operator_rows"])
    output.parent.mkdir(parents=True,exist_ok=True);staging=output.parent/("."+args.version+"-building-"+uuid.uuid4().hex);staging.mkdir()
    figures(staging,run,groups)
    for name,rows in {"operators.csv":run["profile"]["aggregate_operator_rows"],"wall_ranges.csv":run["observation"]["named_wall_ranges"],
        "update_timings.csv":run["observation"]["all_update_timings"],"scalar_sync_copy_candidates.csv":groups["scalar_sync_copy_candidates"]}.items():save_csv(staging/name,rows)
    summary={"schema":"so2_bce_execution_profile_report_v1","campaign_id":CAMPAIGN,"complete":True,
        "maximum_claim":"observed_runtime_costs_only_no_speedup_or_biological_claim",**run,"operator_groups":groups}
    (staging/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (staging/"report.md").write_text(markdown(run,groups));(staging/"report_builder.py").write_bytes(Path(__file__).read_bytes())
    guard=ROOT/"scripts/analysis/so2_bce_execution_profile_contract.py"
    (staging/"report_configuration_guard.py").write_bytes(guard.read_bytes())
    manifest={"schema":"so2_bce_execution_profile_report_manifest_v1","created_at_utc":datetime.now(timezone.utc).isoformat(),
        "command":[sys.executable,str(Path(__file__).resolve()),*(sys.argv[1:] if argv is None else argv)],
        "working_directory":str(Path.cwd()),"source_runs":run["sources"],
        "files":{path.name:sha256_file(path) for path in sorted(staging.iterdir())}}
    (staging/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    _rename_no_replace(staging,output);print(output);return 0


if __name__=="__main__":raise SystemExit(main())
