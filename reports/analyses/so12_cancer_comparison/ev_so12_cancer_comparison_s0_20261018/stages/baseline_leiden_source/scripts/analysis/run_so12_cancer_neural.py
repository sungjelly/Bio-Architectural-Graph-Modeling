"""Managed neural graph precomputation, followed by a GPU-free Leiden subset."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import traceback

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths


EVALUATION = "ev_so12_cancer_comparison_s0_20261018"
SOURCE_RUN = "r_20261017T044837Z_caf76350_s000_f00_a01_12df41be"
METHODS = ("h0", "h1", "h2", "h3", "h4")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _lock(stack, path, mode):
    handle = stack.enter_context(path.open("a"))
    fcntl.flock(handle, mode | fcntl.LOCK_NB)


def _wait_child(child, stop):
    try:
        while child.poll() is None:
            if stop.wait(.5):
                raise RuntimeError("Neural precomputation stopped")
        if child.returncode != 0:
            raise RuntimeError(f"Neural child command exited with code {child.returncode}")
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def run():
    paths = current_paths()
    output = paths.scratch_root / "active_runs" / EVALUATION
    config = json.loads((output / "config.resolved.json").read_text())
    if (config.get("evaluation_id") != EVALUATION or config.get("source_run_id") != SOURCE_RUN
            or config.get("n_cells") != 81607):
        raise RuntimeError("Existing parent evaluation identity differs")
    if not (output / "cells.csv.gz").is_file() or not (output / "layers_manifest.json").is_file():
        raise RuntimeError("Verified canonical test cells and layer extraction are required")
    stages = output / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    locks = paths.state_root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: stop.set())

    def event(status, **values):
        record = {"stage": "neural_precompute", "status": status,
            "time": datetime.now(timezone.utc).isoformat(), **values}
        _write(stages / "neural_precompute.json", record)
        with (stages / "neural_precompute.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)

    def disk_gate():
        if shutil.disk_usage(output).free < 5.1 * 2**30:
            raise RuntimeError("Insufficient free disk above the 5 GiB reserve")

    from so12_cancer_clustering import _code_binding
    binding = _code_binding()
    for relative in ("scripts/analysis/run_so12_cancer_neural.py",
                     "ops/supervisor/bagm-so12-cancer-neural.sh",
                     "ops/supervisor/bagm-so12-cancer-neural.conf"):
        binding[relative] = sha256_file(paths.project_root / relative)

    def verify_source():
        for relative, expected in binding.items():
            if sha256_file(paths.project_root / relative) != expected:
                raise RuntimeError(f"Frozen neural-stage source changed: {relative}")

    def child(arguments, gpu):
        if stop.is_set():
            raise RuntimeError("Stopped before launching neural command")
        disk_gate()
        verify_source()
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="0" if gpu else "", PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=f"{paths.project_root}:{paths.project_root / 'src'}:{paths.project_root / 'scripts/analysis'}")
        command = [sys.executable, "-u", str(paths.project_root / "scripts/analysis/so12_cancer_clustering.py"), *arguments]
        event("graph_running" if gpu else "leiden_running", command=command, gpu=0 if gpu else None)
        with (stages / "neural_precompute.log").open("a") as stream:
            process = subprocess.Popen(command, cwd=paths.project_root, env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            _wait_child(process, stop)
        verify_source()
        disk_gate()

    with ExitStack() as controller:
        _lock(controller, locks / "so12-cancer-neural-precompute.lock", fcntl.LOCK_EX)
        try:
            binding_path = stages / "neural_precompute_source_binding.json"
            if binding_path.exists():
                if json.loads(binding_path.read_text()) != binding:
                    raise RuntimeError("Restart would change the frozen precomputation source")
            else:
                for relative in binding:
                    destination = stages / "neural_precompute_source" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(paths.project_root / relative, destination)
                _write(binding_path, binding)
            with ExitStack() as graph_locks:
                _lock(graph_locks, locks / "so12-cancer-graphs.lock", fcntl.LOCK_EX)
                _lock(graph_locks, locks / "bagm-worker.lock", fcntl.LOCK_SH)
                _lock(graph_locks, locks / "bagm-worker-gpu-0.lock", fcntl.LOCK_EX)
                processes = subprocess.check_output(["nvidia-smi", "--id=0", "--query-compute-apps=pid",
                                                     "--format=csv,noheader"], text=True)
                if processes.strip():
                    raise RuntimeError("GPU 0 already has a compute process; no preemption permitted")
                event("graphs_started", methods=list(METHODS), gpu=0)
                for method in METHODS:
                    child(["graph", "--output", str(output), "--method", method, "--device", "cuda:0"], gpu=True)
            event("graphs_complete_gpu_released", methods=list(METHODS), gpu=None)
            # Share the parent's Leiden-stage mutex so a premature full sweep
            # cannot write the same partitions concurrently.
            with ExitStack() as leiden_lock:
                _lock(leiden_lock, locks / "so12-cancer-leiden.lock", fcntl.LOCK_EX)
                child(["leiden", "--output", str(output), "--methods", *METHODS, "--workers", "6"], gpu=False)
            partial = output / "clustering/partial_summary_h0_h1_h2_h3_h4.json"
            receipt = json.loads(partial.read_text())
            if (not receipt.get("passed") or receipt.get("status") != "complete"
                    or len(receipt.get("partitions", [])) != 40 or receipt.get("all_methods_complete") is not False):
                raise RuntimeError("Neural subset completion receipt is invalid")
            event("complete", partial_summary=str(partial.relative_to(output)),
                  partial_summary_sha256=sha256_file(partial), partitions=40,
                  gpu_released=True, free_bytes=shutil.disk_usage(output).free)
        except BaseException as error:
            event("failed", error=str(error), traceback=traceback.format_exc())
            raise
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(run())
