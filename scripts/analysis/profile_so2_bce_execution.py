#!/usr/bin/env python3
"""Observe a bounded CUDA profile during one unchanged canonical BCE epoch.

All instrumentation is process-local. The original update body, masking,
finite checks, gradient diagnostics and durable writes execute unchanged.
Only aggregate operator/range data are persisted; raw traces are never exported.
"""
from __future__ import annotations

from contextlib import nullcontext
import csv
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))

import torch

from scripts.analysis.so2_bce_execution_profile_contract import (
    CAMPAIGN, PROTOCOL, SCRIPT_PATHS, validate_configuration,
)
from scripts.train import run_so2_bce_optimization_audit as audit
from scripts.train.launch_so2_binary_target_minibatch import source_files
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.run_archive import RunArchive

base,bce,minibatch = audit.base,audit.bce,audit.minibatch
PROFILE_OUTPUT = "diagnostics/execution_operator_profile.json"
SUMMARY_OUTPUT = "diagnostics/execution_profile_summary.json"
PREFLIGHT_PATH = "diagnostics/execution_profile_preflight.json"
RESOURCE_PATH = "diagnostics/execution_profile_resource_samples.jsonl"
_CONFIG = None
_OBSERVER = None
_ORIGINAL_FINAL = audit._write_final_artifacts
_ORIGINAL_TRAIN = audit._train_epoch
_ORIGINAL_PROGRESS = minibatch._progress


def require(value, message):
    if not value:
        raise base.SO2NBTrainingError(message)


def verify_instrumentation(config,paths):
    validate_configuration(config)
    for relative,digest in config["metadata"]["instrumentation_hashes"].items():
        require(sha256_file(paths.project_root/relative)==digest,"Profiling source changed: "+relative)
    require(canonical_sha256(source_files(paths))==config["metadata"]["profile_parent"]["code_fingerprint_sha256"],
            "Frozen training source fingerprint changed")
    contract = paths.project_root/"experiments/campaigns"/CAMPAIGN/"frozen_task_contract.md"
    require(sha256_file(contract)==config["metadata"]["frozen_task_contract_sha256"],"Profiling contract changed")


def aggregate_events(events):
    """Export only safe aggregate fields, never event arguments or tensor data."""
    rows = []
    for event in events:
        key = str(event.key)
        # CUDA template names can be very long. Bound only the human display,
        # retaining a digest of the complete static name for disambiguation.
        label_sha256=hashlib.sha256(key.encode("utf-8")).hexdigest()
        clean_key=key.replace("\n"," ").replace("\r"," ").replace("\t"," ")
        truncated=len(clean_key)>500
        display=clean_key[:440]+"… [sha256:"+label_sha256[:16]+"]" if truncated else clean_key
        device_type=str(event.device_type).rsplit(".",1)[-1]
        require(device_type in {"CPU","CUDA"},"Unexpected profiler device type")
        values = {"self_cpu_microseconds":float(event.self_cpu_time_total),
            "inclusive_cpu_microseconds":float(event.cpu_time_total),
            "self_device_microseconds":float(getattr(event,"self_device_time_total",0.)),
            "inclusive_device_microseconds":float(getattr(event,"device_time_total",0.))}
        require(all(math.isfinite(value) and value>=0 for value in values.values()),"Invalid aggregate profiler time")
        rows.append({"operator":display,"operator_full_label_sha256":label_sha256,
                     "operator_label_truncated":truncated,"operator_label_control_characters_normalized":clean_key!=key,
                     "kind":"named_range" if key.startswith("execution/") else "operator",
                     "device_type":device_type,"calls":int(event.count),**values})
    return sorted(rows,key=lambda row:(-row["self_cpu_microseconds"],row["operator"]))


class Observer:
    def __init__(self,config,*,clock=time.perf_counter):
        self.config=config; self.settings=config["metadata"]["profiler"]; self.clock=clock
        self.started=clock(); self.phase="initialization"; self.active=False
        self.profiler=None; self.capture_started=None; self.capture_stopped=None
        self.startup_seconds=0.; self.finalization_seconds=0.; self.profiled_updates=[]
        self.capture_limited=False; self.operator_rows=[]; self.ranges={}; self.updates=[]
        self.restorations=[]; self.last_update_end=None; self.encoder_index=0

    def record_wall(self,phase,label,elapsed):
        key=phase+"/"+label
        row=self.ranges.setdefault(key,{"range":key,"calls":0,"inclusive_blocking_wall_seconds":0.,"maximum_call_wall_seconds":0.})
        row["calls"]+=1; row["inclusive_blocking_wall_seconds"]+=elapsed
        row["maximum_call_wall_seconds"]=max(row["maximum_call_wall_seconds"],elapsed)

    def wrap(self,obj,name,label,*,phase=None):
        original=getattr(obj,name)
        @wraps(original)
        def observed(*args,**kwargs):
            selected=label() if callable(label) else label
            previous_phase=self.phase
            if phase is not None:
                self.phase=phase
            measured_phase=self.phase
            before=self.clock()
            context=torch.profiler.record_function("execution/"+selected) if self.active else nullcontext()
            try:
                with context:
                    return original(*args,**kwargs)
            finally:
                elapsed=self.clock()-before
                self.record_wall(measured_phase,selected,elapsed)
                self.phase=previous_phase
        setattr(obj,name,observed)
        self.restorations.append((obj,name,original))
        return observed

    def restore(self):
        for obj,name,original in reversed(self.restorations):
            setattr(obj,name,original)
        self.restorations.clear()

    def encoder_label(self):
        label="encoder_full_core" if self.encoder_index%2==0 else "encoder_targets"
        self.encoder_index+=1
        return label

    def start_capture(self):
        require(self.profiler is None and self.capture_started is None,"Profiler capture restarted")
        before=self.clock()
        self.profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,profile_memory=False,with_stack=False,with_flops=False,with_modules=False)
        self.profiler.start()
        self.startup_seconds=self.clock()-before
        self.capture_started=self.clock(); self.active=True

    def stop_capture(self,*,limited=False):
        if self.profiler is None:
            return
        self.active=False; self.capture_stopped=self.clock(); self.capture_limited=limited
        before=self.clock()
        self.profiler.stop()
        self.operator_rows=aggregate_events(self.profiler.key_averages())
        self.profiler=None
        self.finalization_seconds=self.clock()-before
        require(any(row["self_device_microseconds"]>0 for row in self.operator_rows),"CUDA profiler produced no device timing evidence")
        require(self.profiled_updates,"Profiler captured no complete real optimizer update")
        archive=RunArchive.attach_active(os.environ["BAGM_RUN_ID"],paths=current_paths(),scratch_path=minibatch._SCRATCH)
        archive.write_json(PROFILE_OUTPUT,{"schema":"so2_bce_execution_operator_profile_v1",
            "run_id":os.environ["BAGM_RUN_ID"],"profiled_update_indices":self.profiled_updates,
            "profile_coverage_complete":len(self.profiled_updates)==self.settings["profiled_updates"],
            "capture_limited_by_budget":limited,"capture_wall_seconds":self.capture_stopped-self.capture_started,
            "startup_wall_seconds":self.startup_seconds,"stop_and_aggregate_wall_seconds":self.finalization_seconds,
            "aggregate_operator_rows":self.operator_rows,"raw_events_or_tensor_metadata_exported":False,
            "interpretation":"Inclusive nested ranges and CPU/device times overlap; do not sum them into wall-time percentages."})

    def progress(self,event,**values):
        # Close this scope before start/stop: stopping with an open named range
        # would truncate that range and make its inclusive time misleading.
        before=self.clock()
        context=torch.profiler.record_function("execution/progress_write_and_stdout_flush") if self.active else nullcontext()
        try:
            with context:
                _ORIGINAL_PROGRESS(event,**values)
        finally:
            self.record_wall(self.phase,"progress_write_and_stdout_flush",self.clock()-before)
        if event=="epoch_training_started":
            self.phase="training"
        elif event=="core_started":
            self.last_update_end=self.clock()
        elif event=="target_update_completed":
            now=self.clock(); update=int(values["update"])
            require(self.last_update_end is not None,"Update timing lacks its core boundary")
            self.updates.append({"update":update,"target_cells":int(values["target_cells"]),
                "wall_seconds":now-self.last_update_end,"profiled":self.active})
            if self.active:
                self.profiled_updates.append(update)
                limited=now-self.capture_started>=self.settings["max_profile_seconds_at_update_boundary"]
                if len(self.profiled_updates)==self.settings["profiled_updates"] or limited:
                    self.stop_capture(limited=limited)
            elif update==self.settings["warmup_updates"]:
                self.start_capture()
            self.last_update_end=self.clock()


def _build_model_wrapper(original):
    def build(config,bundle):
        model=original(config,bundle)
        observer=_OBSERVER
        observer.wrap(model.encoder,"forward",observer.encoder_label)
        for name,label in (("_project_target_queries_and_clean_sources","source_kv_target_q_projection"),
            ("_target_isolated_layer","graph_layer_with_chunk_staging"),("_output_partition","receiver_chunk_and_checkpoint_recompute"),
            ("_validate_target_input_expression","target_input_correctness_checks"),
            ("_validate_cpu_receiver_major_graph","cpu_graph_order_checks")):
            observer.wrap(model,name,label)
        return model
    return build


def _train_epoch(**kwargs):
    observer=_OBSERVER; observer.phase="training"
    for name,label in (("global_tracker","gradient_full_model_diagnostics"),("block_tracker","gradient_block_diagnostics"),
                       ("theta_tracker","gradient_detection_head_diagnostics")):
        observer.wrap(kwargs[name],"observe",label)
    return _ORIGINAL_TRAIN(**kwargs)


def _load_preflight(config,paths,bundle,model):
    verify_instrumentation(config,paths)
    reference=config["metadata"]["profile_parent"]
    for key,relative in (("success_marker_sha256","_SUCCESS"),("epoch_metrics_sha256","results/epoch_metrics.csv"),
                         ("receipt_sha256","diagnostics/optimization_audit_preflight.json")):
        require(sha256_file(Path(reference["artifact_path"])/relative)==reference[key],"Profiling parent evidence changed")
    audit._BUNDLE=bundle; audit._MODEL=model
    minibatch._SCRATCH=Path(os.environ["BAGM_RUN_SCRATCH"]).resolve(strict=True)
    audit._disk_headroom(minibatch._SCRATCH)
    minibatch._MIN_CUDA_FREE,minibatch._DEVICE_TOTAL=torch.cuda.mem_get_info()
    require(minibatch._MIN_CUDA_FREE>=2*minibatch.GIB,"Initial GPU headroom is insufficient")
    audit._BASELINE=audit._prevalence_reference(bundle,model)
    frozen=source_files(paths)
    hashes={row["path"]:row["sha256"] for row in frozen["files"]}
    hashes.update(config["metadata"]["instrumentation_hashes"])
    initial={"schema":"so2_bce_execution_profile_initialization_v1","protocol":PROTOCOL,
        "configuration_sha256":canonical_sha256(config),"code_hashes":hashes,
        "frozen_task_contract_sha256":config["metadata"]["frozen_task_contract_sha256"],
        "source_configuration_sha256":canonical_sha256(config["source_configuration"]),
        "profile_parent":reference,"model_seed":0,"world_size":1,
        "parameter_count":sum(p.numel() for p in model.parameters()),
        "prevalence_reference_sha256":canonical_sha256(audit._BASELINE),"minimum_free_disk_gib":5.,
        "original_four_rank_preflight_claimed":False}
    initial["receipt_content_sha256"]=canonical_sha256(initial)
    minibatch._INITIALIZATION_RECEIPT=initial
    archive=RunArchive.attach_active(os.environ["BAGM_RUN_ID"],paths=paths,scratch_path=minibatch._SCRATCH)
    archive.write_json("diagnostics/execution_profile_initialization.json",initial)
    archive.write_json("diagnostics/execution_profile_environment.json",{"torch":torch.__version__,"cuda":torch.version.cuda,
        "gpu":torch.cuda.get_device_name(),"visible_physical_gpu":os.environ["CUDA_VISIBLE_DEVICES"],
        "cpu_threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),
        "total_device_bytes":minibatch._DEVICE_TOTAL,"initial_cuda_free_bytes":minibatch._MIN_CUDA_FREE})
    minibatch._SAMPLER=minibatch.ResourceSampler(minibatch._SCRATCH/RESOURCE_PATH,os.environ["CUDA_VISIBLE_DEVICES"])
    minibatch._SAMPLER.start()
    return initial


def parent_epoch_comparison(actual,parent,absolute_tolerance):
    comparisons={}
    for field in ("train_pooled_masked_binary_cross_entropy","validation_pooled_masked_binary_cross_entropy",
                  "validation_detection_brier_score","validation_detection_f1",
                  "validation_detection_auprc_histogram_4096","validation_detection_auroc_histogram_4096"):
        difference=abs(float(actual[field])-float(parent[field]))
        require(math.isfinite(difference) and difference<=absolute_tolerance,"Parent epoch-one scalar mismatch: "+field)
        comparisons[field]={"observed":float(actual[field]),"parent":float(parent[field]),"absolute_difference":difference}
    for field in ("train_target_cells","validation_target_cells","train_masked_entries","validation_masked_entries",
                  "train_positive_entries","validation_positive_entries","optimizer_updates","cumulative_optimizer_updates",
                  "validation_detection_true_positive","validation_detection_true_negative",
                  "validation_detection_false_positive","validation_detection_false_negative"):
        require(int(actual[field])==int(parent[field]),"Parent epoch-one integer mismatch: "+field)
    for field in ("core_order_json","training_coverage_receipts_json","validation_coverage_receipts_json"):
        require(json.loads(actual[field])==json.loads(parent[field]),"Parent epoch-one mask/order mismatch: "+field)
    return {"passed":True,"absolute_scalar_tolerance":absolute_tolerance,"integer_counts_and_mask_receipts_exact":True,
            "scalar_comparisons":comparisons,"weight_state_equality_tested":False}


def _write_final_artifacts(archive,validation):
    _OBSERVER.phase="finalization"
    _ORIGINAL_FINAL(archive,validation)
    require(len(_OBSERVER.updates)==209 and [x["update"] for x in _OBSERVER.updates]==list(range(1,210)),
            "Profiler run did not execute the canonical complete epoch")
    require(_OBSERVER.operator_rows and _OBSERVER.profiler is None,"Aggregate profiling did not complete")
    with (minibatch._SCRATCH/"results/epoch_metrics.csv").open(newline="") as handle:
        actual=list(csv.DictReader(handle))
    with (Path(_CONFIG["metadata"]["profile_parent"]["artifact_path"])/"results/epoch_metrics.csv").open(newline="") as handle:
        parent=list(csv.DictReader(handle))
    require(len(actual)==1 and int(actual[0]["global_epoch"])==int(parent[0]["global_epoch"])==1,"Invalid one-epoch history")
    comparison=parent_epoch_comparison(actual[0],parent[0],_CONFIG["metadata"]["profiler"]["parent_scalar_absolute_tolerance"])
    profiled=[row for row in _OBSERVER.updates if row["profiled"]]
    last=max(row["update"] for row in profiled)
    adjacent=[row for row in _OBSERVER.updates if last<row["update"]<=last+3]
    profile_mean=sum(row["wall_seconds"] for row in profiled)/len(profiled)
    adjacent_mean=sum(row["wall_seconds"] for row in adjacent)/len(adjacent)
    elapsed=_OBSERVER.clock()-_OBSERVER.started
    archive.write_json(SUMMARY_OUTPUT,{"schema":"so2_bce_execution_profile_summary_v1","run_id":os.environ["BAGM_RUN_ID"],
        "profiled_updates":profiled,"adjacent_unprofiled_updates":adjacent,
        "profiled_mean_update_seconds":profile_mean,"adjacent_unprofiled_mean_update_seconds":adjacent_mean,
        "descriptive_profiled_to_adjacent_wall_ratio":profile_mean/adjacent_mean,
        "profile_startup_seconds":_OBSERVER.startup_seconds,"profile_stop_and_aggregate_seconds":_OBSERVER.finalization_seconds,
        "elapsed_through_final_artifacts_seconds":elapsed,"soft_ten_minute_budget_exceeded":elapsed>600,
        "canonical_epoch_updates":209,"full_epoch_validation_and_replay_required":True,
        "parent_epoch_one_comparison":comparison,"named_wall_ranges":sorted(_OBSERVER.ranges.values(),key=lambda x:x["range"]),
        "all_update_timings":_OBSERVER.updates,"profile_operator_file":PROFILE_OUTPUT,
        "update_interval_scope":"Core-first intervals start after core_started; they exclude core mask realization and staging. Subsequent intervals begin after the prior progress boundary. Profiler startup/finalization are excluded and recorded separately.",
        "limitations":["Adjacent updates use different windows and weights; timing ratio is not a controlled causal overhead estimate.",
            "Nested named ranges overlap; CPU and CUDA work can overlap. Aggregate times are not additive wall-time percentages.",
            "Only three early updates are targeted; results need not represent every core or later training stage.",
            "Wall ranges add no new CUDA synchronization and measure observed blocking duration.",
            "The profile demonstrates observed costs, not a validated optimization or speedup."]})


def install_contract(config):
    global _CONFIG,_OBSERVER
    validate_configuration(config); _CONFIG=config
    audit.validate_configuration=validate_configuration
    audit.CAMPAIGN_ID=CAMPAIGN; audit.PROTOCOL=PROTOCOL
    audit.CHECKPOINT_SCHEMA="so2_bce_execution_profile_checkpoint_v1"
    audit.PREFLIGHT_SCHEMA="so2_bce_execution_profile_preflight_v1"
    audit.PREFLIGHT_PATH=PREFLIGHT_PATH; audit.RESOURCE_SAMPLES_PATH=RESOURCE_PATH
    audit.ENDPOINT_PATH="results/execution_profile_endpoint.json"
    audit.PROVENANCE_PATH="provenance/so2_bce_execution_profile.json"
    audit.install_contract(config)
    _OBSERVER=Observer(config)
    base._build_model=_build_model_wrapper(base._build_model)
    base._train_epoch=_train_epoch
    base._load_preflight_receipt=_load_preflight
    base._write_additional_final_artifacts=_write_final_artifacts
    base.ADDITIONAL_REQUIRED_FINAL_OUTPUTS=(*base.ADDITIONAL_REQUIRED_FINAL_OUTPUTS,PROFILE_OUTPUT,SUMMARY_OUTPUT)
    for obj,name,label in ((base,"_load_data","data_loading"),(base,"_stage_core","full_core_staging"),
        (base,"_partition_and_mask","core_target_mask_realization"),
        (base,"_make_target_tensors","target_mask_staging_and_checks"),(base,"_assert_finite_gradients","finite_gradient_checks"),
        (minibatch,"_resource_snapshot","resource_checks"),
        (minibatch,"append_json","durable_json_fsync"),(base.AtomicBestLatestCheckpointStore,"save_best","save_best_checkpoint"),
        (base.AtomicBestLatestCheckpointStore,"save_latest","save_latest_checkpoint")):
        _OBSERVER.wrap(obj,name,label)
    _OBSERVER.wrap(base,"_evaluate_validation","validation",phase="validation")
    original_forward=bce._forward_target_shard
    def forward(**kwargs):
        _OBSERVER.encoder_index=0
        return original_forward(**kwargs)
    bce._forward_target_shard=forward
    _OBSERVER.wrap(bce,"_forward_target_shard","forward_target_window")
    minibatch._progress=_OBSERVER.progress


def main(argv=None):
    parser=base.build_parser(); parser.description=__doc__
    args=parser.parse_args(argv)
    require(args.resume_checkpoint is None,"Execution profiling requires a fresh run, not a partial-trace resume")
    require(args.run_scratch is not None,"Execution profiling requires registered run scratch")
    config=base.load_yaml_mapping(args.config)
    verify_instrumentation(config,current_paths())
    require(not (Path(args.run_scratch)/"checkpoints/latest.ckpt").exists(),"Profiling scratch contains a prior checkpoint")
    install_contract(config)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    try:
        return base.main(argv)
    finally:
        already_failing=sys.exc_info()[0] is not None
        try:
            if _OBSERVER.profiler is not None:
                try:
                    _OBSERVER.stop_capture(limited=True)
                except Exception:
                    if not already_failing:
                        raise
        finally:
            try:
                if minibatch._SAMPLER is not None:
                    minibatch._SAMPLER.stop()
            finally:
                _OBSERVER.restore()


if __name__=="__main__":
    raise SystemExit(main())
