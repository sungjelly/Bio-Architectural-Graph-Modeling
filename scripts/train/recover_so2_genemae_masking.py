#!/usr/bin/env python3
"""Recover a fully trained checkpoint only after a complete matched replay audit."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train.run_so2_genemae_masking import load_inputs, evaluate, verify_pilot
from spatial_benchmark.configuration import load_yaml_mapping
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.myjju_genemae import make_source_model, build_symmetric_knn_graph
from spatial_benchmark.paths import current_paths
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle
from spatial_benchmark.so2_genemae_masking import METHODS, PRIMARY, make_mask, runtime_path, validate_protocol


def compare_evaluations(original, recovered):
    keys=["train_method","model_seed","core","evaluation_method","replicate","condition"]
    left=original.set_index(keys).sort_index()
    right=recovered.set_index(keys).sort_index()
    pd.testing.assert_index_equal(left.index,right.index)
    if len(left)!=588 or list(left.columns)!=list(right.columns):
        raise ValueError("Evaluation row/column drift")
    counts=[c for c in left if c.startswith("n_") or c.endswith("_count")]
    np.testing.assert_array_equal(left[counts].to_numpy(),right[counts].to_numpy())
    numeric=left.select_dtypes(include="number").columns
    np.testing.assert_allclose(left[numeric],right[numeric],rtol=1e-5,atol=1e-7,equal_nan=True)
    delta=np.abs(left[numeric].to_numpy()-right[numeric].to_numpy())
    return {"rows":len(left),"rtol":1e-5,"atol":1e-7,"maximum_metric_absolute_difference":float(np.nanmax(delta)),
            "all_counts_exact":True,"all_metrics_reproduced":True}


def exact_replay(model, state, cores):
    model.cpu().eval()
    fresh=make_source_model(mask_rate=.1).eval()
    fresh.load_state_dict(state,strict=True)
    if not all(torch.equal(value.cpu(),state[key]) for key,value in model.state_dict().items()):
        raise ValueError("Recovered model weights changed")
    x=torch.from_numpy(cores[0][1][:64].copy());coords=cores[0][2][:64]
    e,a=map(torch.from_numpy,build_symmetric_knn_graph(coords,15))
    receipts=[]
    with torch.no_grad():
        for method in METHODS:
            mask=torch.from_numpy(make_mask(coords,1000,method,7432))
            changed=x.clone();changed[mask]=999
            reference=model(x,e,a,mask)[0]
            for repeat in range(3):
                replay=fresh(x,e,a,mask)[0];hidden=fresh(changed,e,a,mask)[0]
                receipts.append({"method":method,"repeat":repeat,"exact_cpu_replay":torch.equal(reference,replay),
                    "hidden_target_invariance":torch.equal(replay,hidden),
                    "replay_max_abs":float((reference-replay).abs().max()),
                    "invariance_max_abs":float((replay-hidden).abs().max())})
    return receipts


def run(config,archive):
    if validate_protocol(config):
        raise ValueError("Recovery is restricted to a production run")
    gate=config["metadata"]["runtime"]["checkpoint_recovery"]
    if sha256_file(__file__)!=gate["recovery_script_sha256"]:
        raise ValueError("Recovery implementation drift")
    for name,digest in config["metadata"]["implementation_sha256"].items():
        if sha256_file(runtime_path(name))!=digest:
            raise ValueError("Original implementation drift")
    source=runtime_path(gate["source_path"])
    if sha256_file(source/"_FAILED")!=gate["failed_marker_sha256"] or sha256_file(source/"checkpoints/last.ckpt")!=gate["checkpoint_sha256"]:
        raise ValueError("Recovery source drift")
    verify_run_bundle(source)
    old=load_yaml_mapping(source/"config.resolved.yaml")
    if scientific_id(old)!=scientific_id(config) or old["seed"]!=config["seed"]:
        raise ValueError("Recovery changes scientific configuration or seed")
    verify_pilot(config)
    payload=torch.load(source/"checkpoints/last.ckpt",map_location="cpu",weights_only=True)
    history=pd.read_parquet(source/"metrics/history.parquet")
    if payload["final_epoch"]!=199 or history.epoch.tolist()!=list(range(1,201)):
        raise ValueError("Source did not complete all 200 epochs")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for complete evaluation replay")
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    cores=load_inputs(config,archive)
    io_seconds=time.monotonic()-start
    model=make_source_model(mask_rate=.1).eval()
    model.load_state_dict(payload["model_state_dict"],strict=True)
    before=exact_replay(model,payload["model_state_dict"],cores)
    archive.write_json("diagnostics/replay_before.json",before)
    if not all(r["exact_cpu_replay"] and r["hidden_target_invariance"] for r in before):
        raise ValueError("Pre-evaluation exact replay failed")
    for relative in ("metrics/history.parquet","metrics/epoch_history.jsonl","metrics/events.jsonl"):
        archive.copy_file(source/relative,relative)
    archive.copy_file(source/"logs/stdout.log","provenance/source_training_stdout.log")
    archive.copy_file(source/"logs/stderr.log","provenance/source_failure_stderr.log")
    archive.copy_file(source/"config.resolved.yaml","provenance/source_config.yaml")
    archive.copy_file(source/"manifest.yaml","provenance/source_manifest.yaml")
    evaluation_start=time.monotonic()
    final=evaluate(model.cuda(),cores,archive,config,torch.device("cuda:0"))
    evaluation_seconds=time.monotonic()-evaluation_start
    current=pd.read_parquet(archive.scratch_path/"metrics/per_core_replicate.parquet")
    agreement=compare_evaluations(pd.read_parquet(source/"metrics/per_core_replicate.parquet"),current)
    if json.loads((source/"diagnostics/evaluation_masks.json").read_text())!=json.loads((archive.scratch_path/"diagnostics/evaluation_masks.json").read_text()):
        raise ValueError("Recovery evaluation masks differ")
    after=exact_replay(model,payload["model_state_dict"],cores)
    archive.write_json("diagnostics/replay_after.json",after)
    if not all(r["exact_cpu_replay"] and r["hidden_target_invariance"] for r in after):
        raise ValueError("Post-evaluation exact replay failed")
    payload.update(run_id=archive.run_id, resolved_config=config, training_source_run_id=gate["source_run_id"],
                   training_source_checkpoint_sha256=gate["checkpoint_sha256"])
    buffer=io.BytesIO();torch.save(payload,buffer)
    checkpoint=archive.write_bytes("checkpoints/last.ckpt",buffer.getvalue())
    reread=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if not all(torch.equal(value,reread["model_state_dict"][key]) for key,value in payload["model_state_dict"].items()):
        raise ValueError("Final checkpoint weight mismatch")
    archive.write_json("diagnostics/replay.json",{"exact_cpu_replay":True,"hidden_target_invariance":True,
        "checkpoint_sha256":sha256_file(checkpoint),"source_weights_exact":True})
    archive.write_json("diagnostics/recovery.json",{**agreement,**gate,"new_optimizer_steps":0,
        "original_failure_cause":"Unconfirmed: combined exact replay/invariance assertion; fresh CPU and GPU-roundtrip checks pass",
        "common_masks_exact":True,"source_weights_exact":True})
    resources={"gpu_name":torch.cuda.get_device_name(),"visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch":str(torch.__version__),"cuda":torch.version.cuda,"precision":"fp32",
        "peak_allocated_vram_gib":torch.cuda.max_memory_allocated()/1024**3,
        "peak_reserved_vram_gib":torch.cuda.max_memory_reserved()/1024**3,
        "peak_host_memory_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        "total_seconds":gate["source_runtime_seconds"]+time.monotonic()-start,
        "recovery_seconds":time.monotonic()-start,"source_runtime_seconds":gate["source_runtime_seconds"],
        "io_and_graph_seconds":io_seconds,"evaluation_seconds":evaluation_seconds,
        "free_disk_gib":shutil.disk_usage(archive.scratch_path).free/1024**3,
        "memory_measurement_scope":"recovery evaluation only; original training peak unavailable",
        "original_training_peak_allocated_vram_gib":None}
    archive.write_json("diagnostics/resource.json",resources)
    archive.write_summary({"run_id":archive.run_id,"status":"success","primary_metric_name":PRIMARY,
        "primary_metric_value":final[PRIMARY],"completed_global_epochs":200,"final_epoch":199,
        "checkpoint_role":"last","checkpoint_sha256":sha256_file(checkpoint),"parameter_count":6888016,
        "model_seed":config["seed"],"train_method":config["masking"]["method"],"mask_rate":.1,
        "prediction_granularity":"per_core_gene_masked_means_not_individual_cells",
        "generalization_claim_supported":False,"resources":resources,"recovery_source_run_id":gate["source_run_id"]})
    print(json.dumps({"complete":True,"recovered":True,**agreement}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--run-scratch",type=Path,required=True)
    args=parser.parse_args()
    if args.run_scratch.resolve()!=Path(os.environ["BAGM_RUN_SCRATCH"]).resolve() or args.config.resolve()!=(args.run_scratch/"config.resolved.yaml").resolve():
        raise ValueError("Worker paths differ")
    run(load_yaml_mapping(args.config),RunArchive.attach_active(os.environ["BAGM_RUN_ID"],paths=current_paths(),scratch_path=args.run_scratch))
