"""Two CPU workers for disjoint PCA/scVI partitions beside the neural sweep."""
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
from run_so12_cancer_neural import EVALUATION, SOURCE_RUN, _lock, _wait_child, _write


METHODS = ("pca", "scvi")


def run():
    paths = current_paths()
    output = paths.scratch_root / "active_runs" / EVALUATION
    config = json.loads((output / "config.resolved.json").read_text())
    if config.get("evaluation_id") != EVALUATION or config.get("source_run_id") != SOURCE_RUN:
        raise RuntimeError("Parent evaluation identity changed")
    stages, locks = output / "stages", paths.state_root / "locks"
    stages.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: stop.set())

    def event(status, **values):
        row = {"stage": "baseline_leiden", "status": status,
               "time": datetime.now(timezone.utc).isoformat(), **values}
        _write(stages / "baseline_leiden.json", row)
        with (stages / "baseline_leiden.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    from so12_cancer_clustering import _code_binding, _verify_graph
    binding = _code_binding()
    for relative in ("scripts/analysis/run_so12_cancer_baseline_leiden.py",
                     "scripts/analysis/run_so12_cancer_neural.py",
                     "ops/supervisor/bagm-so12-cancer-baseline-leiden.sh",
                     "ops/supervisor/bagm-so12-cancer-baseline-leiden.conf"):
        binding[relative] = sha256_file(paths.project_root / relative)

    def verify_sources():
        for relative, expected in binding.items():
            if sha256_file(paths.project_root / relative) != expected:
                raise RuntimeError(f"Frozen baseline Leiden source changed: {relative}")

    # This unique mutex permits the neural stage's global Leiden mutex to remain
    # held. Method directories are disjoint. The final full sweep awaits BOTH.
    with ExitStack() as controller:
        _lock(controller, locks / "so12-cancer-baseline-leiden.lock", fcntl.LOCK_EX)
        try:
            binding_path = stages / "baseline_leiden_source_binding.json"
            if binding_path.exists():
                if json.loads(binding_path.read_text()) != binding:
                    raise RuntimeError("Restart would change the frozen baseline Leiden source")
            else:
                for relative in binding:
                    destination = stages / "baseline_leiden_source" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(paths.project_root / relative, destination)
                _write(binding_path, binding)
            quota_path = Path("/sys/fs/cgroup/cpu.max")
            _write(stages / "baseline_leiden_execution_plan.json", {
                "methods": list(METHODS), "cpu_workers": 2, "concurrent_neural_workers": 6,
                "total_leiden_worker_budget": 8, "logical_cpu_affinity": len(os.sched_getaffinity(0)),
                "cgroup_cpu_quota": quota_path.read_text().strip() if quota_path.is_file() else None,
                "change_scope": "execution_concurrency_only", "scientific_settings_unchanged": True,
                "output_scope": ["clustering/pca", "clustering/scvi", "clustering/partial_summary_pca_scvi.json"],
                "gpu_used": False, "final_full_sweep_requires_both_partial_sweeps_complete": True})
            while not stop.is_set():
                graph_stage_path = stages / "graphs.json"
                if graph_stage_path.is_file():
                    graph_stage = json.loads(graph_stage_path.read_text())
                    if graph_stage.get("status") == "failed":
                        raise RuntimeError("Parent graph stage failed; baseline Leiden cannot start")
                    if graph_stage.get("status") == "complete":
                        break
                event("waiting_for_full_graphs", gpu=None)
                stop.wait(10)
            if stop.is_set():
                raise RuntimeError("Baseline Leiden stopped before graph completion")
            verify_sources()
            for method in METHODS:
                receipt = json.loads((output / "clustering" / method / "graph_receipt.json").read_text())
                if receipt.get("method") != method:
                    raise RuntimeError("Baseline graph method mismatch")
                _verify_graph(output, receipt)
            if shutil.disk_usage(output).free < 5.1 * 2**30:
                raise RuntimeError("Insufficient disk above the 5 GiB reserve")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=f"{paths.project_root}:{paths.project_root / 'src'}:{paths.project_root / 'scripts/analysis'}")
            command = [sys.executable, "-u", str(paths.project_root / "scripts/analysis/so12_cancer_clustering.py"),
                "leiden", "--output", str(output), "--methods", *METHODS, "--workers", "2"]
            event("running", methods=list(METHODS), workers=2, gpu=None, command=command)
            with (stages / "baseline_leiden.log").open("a") as stream:
                child = subprocess.Popen(command, cwd=paths.project_root, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                _wait_child(child, stop)
            verify_sources()
            partial = output / "clustering/partial_summary_pca_scvi.json"
            receipt = json.loads(partial.read_text())
            if (receipt.get("status") != "complete" or not receipt.get("passed")
                    or set(receipt.get("methods", [])) != set(METHODS)
                    or len(receipt.get("partitions", [])) != 16 or receipt.get("all_methods_complete") is not False):
                raise RuntimeError("Baseline subset completion receipt is invalid")
            if shutil.disk_usage(output).free < 5 * 2**30:
                raise RuntimeError("Disk reserve breached")
            event("complete", partitions=16, workers=2, gpu=None,
                partial_summary=str(partial.relative_to(output)), partial_summary_sha256=sha256_file(partial))
        except BaseException as error:
            event("failed", error=str(error), traceback=traceback.format_exc())
            raise
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(run())
