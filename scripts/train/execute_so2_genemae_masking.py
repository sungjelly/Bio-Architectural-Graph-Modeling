#!/usr/bin/env python3
"""Launch the prespecified production matrix after the signed pilot gate."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from spatial_benchmark.configuration import load_yaml_mapping
from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.paths import current_paths
from spatial_benchmark.queueing import command_for_config
from spatial_benchmark.registry import Registry
from spatial_benchmark.so2_genemae_masking import CAMPAIGN, METHODS, runtime_path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--pilot-run",required=True)
    parser.add_argument("--wait-for-pilot",action="store_true")
    args=parser.parse_args()
    paths=current_paths()
    pilot_root=runtime_path(args.pilot_run)
    if args.wait_for_pilot:
        print("Waiting for the registered pilot to finalize successfully",flush=True)
        while not (pilot_root/"_SUCCESS").is_file():
            if (pilot_root/"_FAILED").exists():
                raise RuntimeError("Pilot failed; production remains unqueued")
            time.sleep(10)
    env=dict(os.environ, PYTHONPATH=str(paths.project_root/"src"), OMP_NUM_THREADS="4",
             OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", BAGM_MIN_FREE_DISK_GB="8")
    # Resource inspection only; never kill or displace another process.
    active=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
    if active:
        raise RuntimeError("GPU compute processes are active; inspect allocation before launching")
    subprocess.run([sys.executable,str(paths.project_root/"scripts/train/prepare_so2_genemae_masking.py"),
                    "--pilot-run",args.pilot_run],cwd=paths.project_root,env=env,check=True)
    registry=Registry()
    directory=paths.project_root/"experiments/campaigns"/CAMPAIGN/"configs"
    slots=[(method,seed) for seed in range(3) for method in METHODS]
    job_ids=[]
    for index,(method,seed) in enumerate(slots):
        path=directory/f"{method}_s{seed}.yaml"
        config=load_yaml_mapping(path)
        registry.register_variant(scientific_id(config),campaign_id=CAMPAIGN,configuration=config)
        row=registry.enqueue(campaign_id=CAMPAIGN,configuration=config,command=command_for_config(config),
                             experiment_config_reference=path,requested_gpu=None,maximum_attempts=1,
                             priority={"partial_gene":2,"whole_cell":1,"spatial_block":0}[method])
        job_ids.append(row["job_id"])
    print(json.dumps({"ordered_jobs":job_ids,"gpus":[0,1,2,3],"schedule":"next free GPU; one independent job per GPU"}),flush=True)
    stop=threading.Event()
    def run_gpu(gpu):
        while not stop.is_set():
            with Registry().connect() as connection:
                queued=connection.execute("SELECT count(*) FROM queue_jobs WHERE campaign_id=? AND status='queued'",(CAMPAIGN,)).fetchone()[0]
            if not queued:
                return
            code=subprocess.call([sys.executable,"-m","spatial_benchmark","worker","--gpu",str(gpu),
                "--once","--parallel-gpu-workers","--min-free-gb","8",
                "--worker-id",f"so2-genemae-gpu{gpu}"],cwd=paths.project_root,env=env)
            with Registry().connect() as connection:
                records=[dict(r) for r in connection.execute("SELECT job_id,status,run_id,failure_category FROM queue_jobs WHERE campaign_id=? AND worker_id=?",(CAMPAIGN,f"so2-genemae-gpu{gpu}"))]
            print(json.dumps({"gpu":gpu,"jobs":records}),flush=True)
            if code or any(r["status"] == "failed" for r in records):
                stop.set()
                raise RuntimeError(f"A job on GPU {gpu} failed; no blind retry")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run_gpu,gpu) for gpu in range(4)]
        for future in futures:
            future.result()
    subprocess.run([sys.executable,str(paths.project_root/"scripts/analysis/report_so2_genemae_masking.py")],
                   cwd=paths.project_root,env=env,check=True)


if __name__ == "__main__":
    main()
