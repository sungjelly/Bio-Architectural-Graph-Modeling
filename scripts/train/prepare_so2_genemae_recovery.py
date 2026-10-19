#!/usr/bin/env python3
"""Bind a completed failed checkpoint to a separate, reproducible recovery config."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from spatial_benchmark.configuration import load_yaml_mapping
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import verify_run_bundle
from spatial_benchmark.so2_genemae_masking import CAMPAIGN,runtime_path,validate_protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source-run",required=True)
    parser.add_argument("--attempt",type=int,default=2)
    args=parser.parse_args();paths=current_paths();source=runtime_path(args.source_run)
    verify_run_bundle(source)
    if not (source/"_FAILED").is_file():raise ValueError("Source must remain a failed immutable attempt")
    config=load_yaml_mapping(source/"config.resolved.yaml")
    if validate_protocol(config):raise ValueError("Production checkpoint required")
    record=Registry().get_run(source.name)
    if record is None or record["status"]!="failed":raise ValueError("Failed source is not registered")
    elapsed=(datetime.fromisoformat(record["end_time"].replace("Z","+00:00"))-datetime.fromisoformat(record["start_time"].replace("Z","+00:00"))).total_seconds()
    if args.attempt<2:raise ValueError("Recovery requires a later attempt")
    config["attempt"]=args.attempt
    config["metadata"].setdefault("runtime",{})["checkpoint_recovery"]={
        "source_run_id":source.name,"source_path":"artifacts/"+str(source.relative_to(paths.artifact_root)),
        "failed_marker_sha256":sha256_file(source/"_FAILED"),
        "checkpoint_sha256":sha256_file(source/"checkpoints/last.ckpt"),
        "recovery_script_sha256":sha256_file(paths.project_root/"scripts/train/recover_so2_genemae_masking.py"),
        "source_runtime_seconds":elapsed}
    validate_protocol(config)
    target=paths.project_root/"experiments/campaigns"/CAMPAIGN/"configs"/f"{config['masking']['method']}_s{config['seed']}_recovery_a{args.attempt}.yaml"
    content=yaml.safe_dump(config,sort_keys=False)
    if target.exists() and target.read_text()!=content:raise ValueError("Recovery config exists with different provenance")
    target.write_text(content)
    print(target)


if __name__=="__main__":main()
