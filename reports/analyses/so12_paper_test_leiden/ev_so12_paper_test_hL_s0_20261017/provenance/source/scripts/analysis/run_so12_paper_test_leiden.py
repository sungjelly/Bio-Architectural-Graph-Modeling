#!/usr/bin/env python3
"""Managed, source-gated Leiden analysis of the two reserved SO12 test cores."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry

RUN = "r_20261017T044837Z_caf76350_s000_f00_a01_12df41be"
CAMPAIGN = "cmp_20261017_so12_paper_test_leiden"
EVALUATION = "ev_so12_paper_test_hL_s0_20261017"
DATASET = "so12_paper_core_split"
SPLIT = "so12_paper_21train_5val_2test_v1"
CORE_COUNTS = {"15": 38145, "23": 43462}
REQUIRED_OUTPUTS = {
    "config.resolved.json", "summary.json", "cells.csv.gz", "embeddings/hL.npy",
    "extraction_manifest.json", "clustering_summary.json", "rendering_summary.json",
    "embeddings/core_15_receipt.json", "embeddings/core_23_receipt.json",
    "diagnostics/extraction_pilot.json", "provenance/source_verification.json",
    "clustering/hL_labels_seed0.npy", "clustering/hL_labels_seed1.npy",
    "clustering/clusters.csv.gz", "clustering/cluster_sizes.csv", "clustering/core_composition.csv",
    "clustering/clustering_manifest.json", "report/index.html", "report/figure_manifest.json",
    "figures/core_15_hL_leiden_segmentation.png", "figures/core_15_hL_leiden_segmentation.pdf",
    "figures/core_23_hL_leiden_segmentation.png", "figures/core_23_hL_leiden_segmentation.pdf",
    "figures/test_cores_hL_leiden.png", "figures/test_cores_hL_leiden.pdf",
    "figures/shared_cluster_color_key.png", "figures/shared_cluster_color_key.pdf",
}
SOURCE_FILES = (
    "scripts/analysis/run_so12_paper_test_leiden.py",
    "scripts/analysis/so12_paper_test_extraction.py",
    "scripts/analysis/so12_paper_test_clustering.py",
    "scripts/analysis/so12_hidden_extraction.py",
    "scripts/analysis/so12_hidden_clustering.py",
    "scripts/analysis/so12_hidden_maps.py",
    "src/spatial_benchmark/attention_niche_geometry.py",
    "src/spatial_benchmark/relative_qkv_embedding_clustering.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def record(path):
    return {"sha256": sha256_file(path), "size_bytes": Path(path).stat().st_size}


def scientific_config(paths):
    return {
        "schema": "so12_paper_test_leiden_v1", "evaluation_id": EVALUATION,
        "campaign_id": CAMPAIGN, "source_run_id": RUN,
        "dataset_id": DATASET, "split_id": SPLIT,
        "test_cores": [15, 23], "test_cells": 81607,
        "representation": "hL", "dimensions": 256, "all_observed_genes": True,
        "normalization": "unchanged_training_fitted_parameters",
        "joint_test_clustering": True, "knn": 30, "metric": "cosine",
        "graph": "unweighted_undirected_union", "leiden_resolution": 1.0,
        "primary_seed": 0, "stability_seed": 1, "leiden_iterations": -1,
        "minimum_free_gib": 5, "additional_storage_budget_gib": 0.5,
        "classification": {"lifecycle_stage": "exploratory_screen",
                           "study_axis": "reserved_test_core_representation",
                           "scientific_variant": "fully_observed_hL_joint_test_leiden_r1"},
        "analysis_source_sha256": {name: sha256_file(paths.project_root / name) for name in SOURCE_FILES},
    }


def completed_source(registry, paths):
    """Registry-only polling: never inspect live weights or test arrays."""
    run = registry.get_run(RUN)
    require(run is not None, "The exact source training run is not registered")
    status = run["status"]
    if status in {"failed", "cancelled", "pruned", "interrupted", "stale"}:
        raise RuntimeError(f"Source training ended without success: {status}")
    if status != "completed":
        return None
    expected = (paths.artifact_root / "runs/2026/10" / RUN).resolve()
    require(run.get("artifact_path") and Path(run["artifact_path"]).resolve() == expected,
            "Completed source archive path does not match the exact source run")
    require((expected / "_SUCCESS").is_file(), "Completed source is missing its success marker")
    return expected


def evaluation_row(registry):
    with registry.connect() as connection:
        row = connection.execute("SELECT * FROM evaluations WHERE evaluation_id=?", (EVALUATION,)).fetchone()
    return dict(row) if row else None


def update_evaluation(registry, out, status, stage, **metrics):
    payload = {"campaign_id": CAMPAIGN, "stage": stage, "test_cores": [15, 23], **metrics}
    with registry.transaction(immediate=True) as connection:
        cursor = connection.execute(
            "UPDATE evaluations SET status=?,artifact_path=?,metrics_json=?,finished_at=? "
            "WHERE evaluation_id=? AND run_id=?",
            (status, str(out), json.dumps(payload, allow_nan=False),
             now() if status in {"failed", "completed"} else None, EVALUATION, RUN))
        require(cursor.rowcount == 1, "Evaluation registry identity changed")


def initialize_evaluation(registry, paths, out, config):
    row = evaluation_row(registry)
    if row is None:
        require(not out.exists(), "Unregistered evaluation output exists; preserve and inspect it")
        out.mkdir(parents=True)
        write(out / "config.resolved.json", config)
        registry.create_evaluation(EVALUATION, run_id=RUN, checkpoint_name="best", dataset_id=DATASET,
            split_id=SPLIT, status="pending", artifact_path=out,
            metrics={"campaign_id": CAMPAIGN, "stage": "waiting_for_training", "test_cores": [15, 23]})
        return
    require(row["run_id"] == RUN and row["dataset_id"] == DATASET and row["split_id"] == SPLIT
            and row["checkpoint_name"] == "best",
            "Existing evaluation has a different source or split")
    require(row["artifact_path"] and Path(row["artifact_path"]).resolve() == out.resolve(),
            "Existing evaluation artifact path differs from its owned output")
    require(row["status"] == "pending" and out.is_dir(), "Existing evaluation needs explicit failure/recovery review")
    require(read(out / "config.resolved.json") == config, "Pending analysis configuration/source changed")
    require(not (out / "COMPUTATION_STARTED.json").exists(), "Partial computation cannot be silently restarted")


def acquire_gpu(stack, paths):
    locks = paths.state_root / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    for name, mode in (("bagm-worker.lock", fcntl.LOCK_SH), ("bagm-worker-gpu-0.lock", fcntl.LOCK_EX)):
        handle = stack.enter_context((locks / name).open("a"))
        fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)


def gpu_zero_free():
    output = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    return not output.strip()


def wait_child(child, stop):
    """Keep resource locks until all stages' supervised process has exited."""
    terminated = False
    while child.poll() is None:
        if stop.is_set() and not terminated:
            os.killpg(child.pid, signal.SIGTERM)
            terminated = True
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    if stop.is_set():
        raise InterruptedError("Analysis interrupted after supervised child exit")
    require(child.returncode == 0, f"Analysis stage exited with code {child.returncode}")


def validate_stages(root):
    stages = {name: read(root / filename) for name, filename in (
        ("extraction", "extraction_manifest.json"), ("clustering", "clustering_summary.json"),
        ("rendering", "rendering_summary.json"))}
    for name, receipt in stages.items():
        require(receipt.get("passed") is True and receipt.get("status") == "complete",
                f"Incomplete {name} receipt")
        require(receipt.get("n_cells") == 81607 and receipt.get("test_core_numbers") == [15, 23]
                and receipt.get("cells_per_core") == CORE_COUNTS,
                f"Incorrect {name} test-cell coverage")
    extraction = stages["extraction"]
    require(extraction.get("source_run_id") == RUN and extraction.get("verified") is True
            and extraction.get("dimensions") == 256 and extraction.get("all_observed") is True
            and extraction.get("test_only") is True and extraction.get("normalization_refit") is False
            and extraction.get("source_modified") is False,
            "Extraction source or frozen representation differs from the contract")
    summary = read(root / "summary.json")
    require(all(summary.get(name) == receipt for name, receipt in stages.items()),
            "Summary differs from completed stage receipts")
    require(summary.get("test_cores") == [15, 23] and summary.get("test_cells") == 81607,
            "Summary test population changed")
    return summary


def verify_published(root, config):
    root = Path(root)
    require(not root.is_symlink(), "Published root cannot be a symlink")
    entries = list(root.rglob("*"))
    require(not any(path.is_symlink() for path in entries), "Published tree contains a symlink")
    manifest = read(root / "manifest.json")
    require(manifest["evaluation_id"] == EVALUATION and manifest["source_run_id"] == RUN,
            "Published evaluation identity mismatch")
    require(read(root / "config.resolved.json") == config, "Published analysis configuration mismatch")
    require(read(root / "_SUCCESS")["manifest_sha256"] == sha256_file(root / "manifest.json"),
            "Published completion marker mismatch")
    actual = {str(path.relative_to(root)) for path in entries if path.is_file()}
    actual -= {"manifest.json", "_SUCCESS"}
    require(REQUIRED_OUTPUTS.issubset(actual), "Published analysis is missing required artifacts")
    require(set(manifest["files"]) == actual, "Published manifest is not a complete file inventory")
    for relative, expected in manifest["files"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "Unsafe path in published manifest")
        require(record(root / relative) == expected, f"Published file changed: {relative}")
    require(manifest.get("summary") == validate_stages(root), "Published summary identity mismatch")
    return manifest


def register_publication(registry, final, manifest):
    path = final / "manifest.json"
    stamp = now()
    with registry.transaction(immediate=True) as connection:
        existing = connection.execute("SELECT sha256 FROM artifacts WHERE evaluation_id=? AND path=?",
                                      (EVALUATION, str(path))).fetchall()
        expected = sha256_file(path)
        require(all(row["sha256"] == expected for row in existing), "Registered manifest identity changed")
        if not existing:
            connection.execute("INSERT INTO artifacts(run_id,evaluation_id,kind,path,sha256,size_bytes,status,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)", (RUN, EVALUATION, "so12_paper_test_leiden", str(path), expected,
                                           path.stat().st_size, "present", stamp))
        cursor = connection.execute("UPDATE evaluations SET status='completed',artifact_path=?,metrics_json=?,finished_at=? "
            "WHERE evaluation_id=? AND run_id=?", (str(final), json.dumps(manifest["summary"], allow_nan=False),
                                                   stamp, EVALUATION, RUN))
        require(cursor.rowcount == 1, "Evaluation publication row missing")


def snapshot(paths, out, config):
    for name in config["analysis_source_sha256"]:
        require(sha256_file(paths.project_root / name) == config["analysis_source_sha256"][name],
                f"Analysis code changed while waiting: {name}")
        target = out / "provenance/source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths.project_root / name, target)
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pandas", "scipy", "scikit-learn", "igraph", "leidenalg", "matplotlib", "shapely")}
    write(out / "provenance/environment.json", {"python": sys.version, "packages": packages,
        "captured_at": now(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                                   cwd=paths.project_root, text=True).strip()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    config = scientific_config(paths)
    out = paths.scratch_root / "active_runs" / EVALUATION
    final = paths.report_root / "analyses/so12_paper_test_leiden" / EVALUATION
    control = paths.scratch_root / "active_runs/ctrl_so12_paper_test_leiden"
    control.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stop.set())

    def event(stage, **values):
        row = {"time": now(), "stage": stage, "evaluation_id": EVALUATION, "source_run_id": RUN, **values}
        write(control / "status.json", row)
        with (control / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps(row), flush=True)

    def disk_gate(additional=0):
        free = shutil.disk_usage(out.parent).free
        require(free >= (5 + additional) * 2**30, "Insufficient disk for analysis plus 5 GiB reserve")
        return free

    def stage(name, arguments):
        require(not stop.is_set(), "Analysis stopped before stage launch")
        log = out / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", *arguments]
        event(name + "_running", command=command)
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES="0", PYTHONPATH=f"{paths.project_root}:{paths.project_root / 'src'}:{paths.project_root / 'scripts/analysis'}",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
        with log.open("a") as stream:
            child = subprocess.Popen(command, cwd=paths.project_root, env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                wait_child(child, stop)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
        disk_gate()
        event(name + "_complete")

    locks_root = paths.state_root / "locks"
    locks_root.mkdir(parents=True, exist_ok=True)
    with (locks_root / "so12-paper-test-leiden.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if final.exists():
            manifest = verify_published(final, config)
            register_publication(registry, final, manifest)
            event("completed", artifact_path=str(final), reconciled=True)
            return 0
        registry.create_campaign(CAMPAIGN, name="Reserved SO12 paper test-core hL Leiden analysis",
            scientific_question="Which descriptive frozen-model embedding groups occur in test cores 15 and 23?",
            config={"source_run_id": RUN, "evaluation_id": EVALUATION,
                    "contract": f"experiments/campaigns/{CAMPAIGN}/README.md"})
        initialize_evaluation(registry, paths, out, config)
        try:
            while not stop.is_set():
                source = completed_source(registry, paths)
                if source is None:
                    event("waiting_for_training_and_final_test")
                    stop.wait(30)
                    continue
                stack = ExitStack()
                try:
                    acquire_gpu(stack, paths)
                except BlockingIOError:
                    stack.close()
                    event("waiting_for_gpu_lock")
                    stop.wait(30)
                    continue
                with stack:
                    if not gpu_zero_free():
                        event("waiting_for_existing_gpu_process")
                        stop.wait(30)
                        continue
                    require(completed_source(registry, paths) == source, "Source completion changed")
                    disk_gate(additional=0.5)
                    snapshot(paths, out, config)
                    from so12_paper_test_extraction import verify_source
                    verified = verify_source(source, paths)
                    write(out / "provenance/source_verification.json", verified)
                    require(not stop.is_set(), "Analysis stopped before test extraction")
                    write(out / "COMPUTATION_STARTED.json", {"time": now(), "source": str(source)})
                    update_evaluation(registry, out, "running", "extracting")
                    stage("extraction", [str(paths.project_root / "scripts/analysis/so12_paper_test_extraction.py"),
                        "--source-run-dir", str(source), "--output", str(out), "--device", "cuda:0"])
                    stage("clustering", [str(paths.project_root / "scripts/analysis/so12_paper_test_clustering.py"),
                        "--output", str(out), "--device", "cuda:0", "--stage", "cluster"])
                # Plotting is CPU-only and releases the selected GPU allocation.
                stage("rendering", [str(paths.project_root / "scripts/analysis/so12_paper_test_clustering.py"),
                    "--output", str(out), "--stage", "render"])
                extraction = read(out / "extraction_manifest.json")
                clustering = read(out / "clustering_summary.json")
                rendering = read(out / "rendering_summary.json")
                summary = {"campaign_id": CAMPAIGN, "test_cores": [15, 23], "test_cells": 81607,
                    "representation": "hL", "extraction": extraction, "clustering": clustering,
                    "rendering": rendering, "manual_visual_review": "not_performed",
                    "scope": "descriptive_transductive_clustering_of_reserved_test_embeddings"}
                write(out / "summary.json", summary)
                event("publication_started")
                shutil.copyfile(control / "events.jsonl", out / "events.jsonl")
                manifest = {"schema": "so12_paper_test_leiden_report_v1", "evaluation_id": EVALUATION,
                    "source_run_id": RUN, "completed_at": now(), "summary": summary,
                    "files": {str(path.relative_to(out)): record(path) for path in sorted(out.rglob("*"))
                              if path.is_file() and str(path.relative_to(out)) not in {"manifest.json", "_SUCCESS"}}}
                write(out / "manifest.json", manifest)
                write(out / "_SUCCESS", {"manifest_sha256": sha256_file(out / "manifest.json"), "completed_at": now()})
                verify_published(out, config)
                final.parent.mkdir(parents=True, exist_ok=True)
                require(not final.exists(), "Final analysis path already exists")
                require(out.stat().st_dev == final.parent.stat().st_dev, "Publication requires same-filesystem rename")
                out.rename(final)
                manifest = verify_published(final, config)
                register_publication(registry, final, manifest)
                event("completed", artifact_path=str(final))
                return 0
            event("stopped_while_waiting")
            return 130
        except BaseException as error:
            if out.exists():
                write(out / "FAILURE.json", {"time": now(), "error": str(error), "traceback": traceback.format_exc()})
                update_evaluation(registry, out, "failed", "failed", error=str(error))
            event("failed", error=str(error))
            raise


if __name__ == "__main__":
    raise SystemExit(main())
