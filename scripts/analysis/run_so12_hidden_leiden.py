"""Registered, supervised all-observed SO12 hidden-state segmentation analysis."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry

RUN = "r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b"
PCA_RUN = "r_20261016T181735Z_e4e8f852_s000_f00_a02_d8caf6c4"
CAMPAIGN = "cmp_20261017_so12_h0_hl_leiden_segmentation"
CORES = [1, 9, 13, 15, 21, 23]
CHECKPOINT_SHA = "6b4526508d223fb8a286084f57fc5f4b36129db6afeffaa328672d2c06226af4"


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def record(path):
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def event(out, event_name, **values):
    row = {"time": now(), "event": event_name, **values}
    with (out / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(row), flush=True)


def disk_gate(out):
    free = shutil.disk_usage(out).free
    require(free >= 5 * 2**30, f"Disk reserve breached: {free / 2**30:.2f} GiB available")
    return free


def gpu_inventory():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader"], text=True)


def verify_pca_source(pca):
    run = Registry().get_run(PCA_RUN)
    require(run is not None and run["status"] == "completed"
            and Path(run["artifact_path"]).resolve() == pca.resolve(), "PCA registry mismatch")
    require((pca / "_SUCCESS").read_text().strip() == PCA_RUN, "PCA completion marker mismatch")
    require(read(pca / "identity.json")["run_id"] == PCA_RUN
            and read(pca / "status.json")["status"] == "completed"
            and read(pca / "status.json")["run_id"] == PCA_RUN, "PCA identity/status mismatch")
    files = read(pca / "checksums.json")
    for relative, expected in files.items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe PCA path")
        require(record(pca / relative) == expected, f"PCA source checksum mismatch: {relative}")
    return {"valid": True, "run_id": PCA_RUN, "files_verified": len(files),
            "checksums_file": record(pca / "checksums.json")}


def gpu_locks(stack, paths):
    root = paths.state_root / "locks"
    root.mkdir(parents=True, exist_ok=True)
    for name in ["bagm-worker.lock", *[f"bagm-worker-gpu-{i}.lock" for i in range(4)]]:
        handle = stack.enter_context((root / name).open("a"))
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True).strip()
    require(not active, "GPU compute processes already active; will not displace them")


def jobs(out, tasks):
    """Run supervised children concurrently, preserving exact commands and logs."""
    running = []
    try:
        for name, arguments in tasks:
            log = (out / "logs" / f"{name}.log")
            log.parent.mkdir(exist_ok=True)
            stream = log.open("a")
            command = [sys.executable, "-u", *arguments]
            event(out, "job_started", name=name, command=command)
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            running.append((name, process, stream))
        while running:
            for name, process, stream in running[:]:
                code = process.poll()
                if code is not None:
                    stream.close()
                    running.remove((name, process, stream))
                    event(out, "job_finished", name=name, exit_code=code)
                    require(code == 0, f"{name} failed with {code}; inspect logs/{name}.log")
            if running:
                time.sleep(2)
        disk_gate(out)
    finally:
        for _, process, stream in running:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()


def snapshot(out, paths):
    sources = [
        "scripts/analysis/run_so12_hidden_leiden.py",
        "scripts/analysis/so12_hidden_extraction.py",
        "scripts/analysis/so12_hidden_clustering.py",
        "scripts/analysis/so12_hidden_maps.py",
        "scripts/analysis/audit_so12_hidden_leiden.py",
        "src/spatial_benchmark/relative_qkv_embedding_clustering.py",
        "src/spatial_benchmark/attention_niche_geometry.py",
        "src/spatial_benchmark/so12_relative_qkv_training.py",
        "src/spatial_benchmark/so12_relative_qkv_data.py",
        "src/spatial_benchmark/relative_qkv_graph_transformer.py",
        "src/spatial_benchmark/models.py",
        "src/spatial_benchmark/pooled_relative_qkv_training_v2.py",
        "tests/unit/spatial_benchmark/test_so12_hidden_extraction.py",
        "tests/unit/spatial_benchmark/test_so12_hidden_clustering.py",
        "tests/unit/spatial_benchmark/test_so12_hidden_maps.py",
        "tests/unit/spatial_benchmark/test_so12_hidden_orchestration.py",
        f"experiments/campaigns/{CAMPAIGN}/README.md",
    ]
    records = {}
    for relative in sources:
        source = paths.project_root / relative
        dest = out / "provenance/source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        records[relative] = record(source)
    packages = {}
    for name in ["torch", "numpy", "pandas", "scipy", "scikit-learn", "igraph", "leidenalg", "matplotlib", "shapely", "pyarrow"]:
        packages[name] = importlib.metadata.version(name)
    write(out / "provenance/execution.json", {"created_at": now(), "python": sys.version,
          "packages": packages, "gpu_inventory": gpu_inventory(), "code_files": records,
          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "git_status": subprocess.check_output(["git", "status", "--short"], text=True)})


def update_status(eid, out, status, metrics):
    registry = Registry()
    with registry.transaction(immediate=True) as connection:
        cursor = connection.execute("UPDATE evaluations SET status=?,artifact_path=?,metrics_json=?,finished_at=? WHERE evaluation_id=? AND run_id=?",
            (status, str(out), json.dumps(metrics, allow_nan=False), now() if status != "running" else None, eid, RUN))
        require(cursor.rowcount == 1, "Evaluation registry row missing")


def execute(eid, retry_of=None):
    from so12_hidden_extraction import prepare_extraction, collect_extraction
    from so12_hidden_clustering import collect_clustering
    from so12_hidden_maps import render_comparison

    paths = current_paths()
    out = paths.scratch_root / "active_runs" / eid
    source = paths.artifact_root / "runs/2026/10" / RUN
    pca = paths.artifact_root / "runs/2026/10" / PCA_RUN
    require(not out.exists(), "Evaluation destination exists; inspect it before a new attempt")
    if retry_of is not None:
        with Registry().connect() as connection:
            prior = connection.execute("SELECT * FROM evaluations WHERE evaluation_id=?", (retry_of,)).fetchone()
        require(prior is not None and prior["status"] == "failed" and prior["run_id"] == RUN,
                "Retry must reference a preserved failed evaluation of this training run")
    out.mkdir(parents=True)
    Registry().create_evaluation(eid, run_id=RUN, checkpoint_name="best", dataset_id="so12_pearson_integration",
        split_id="so12_10train_4diagnostic_donors_v1", status="running", artifact_path=out,
        metrics={"campaign_id": CAMPAIGN, "stage": "preparing", "retry_of": retry_of})
    try:
        with ExitStack() as stack:
            gpu_locks(stack, paths)
            disk_gate(out)
            snapshot(out, paths)
            require(sha256_file(source / "checkpoints/best.ckpt") == CHECKPOINT_SHA, "Unexpected checkpoint")
            pca_verification = verify_pca_source(pca)
            write(out / "provenance/pca_bundle_verification.json", pca_verification)
            config = {"evaluation_id": eid, "retry_of": retry_of, "campaign_id": CAMPAIGN, "source_run_id": RUN,
                "pca_run_id": PCA_RUN, "checkpoint_sha256": CHECKPOINT_SHA,
                "displayed_cores": CORES, "all_observed_genes": True, "dimensions": 256,
                "knn": 30, "metric": "cosine", "graph": "unweighted_undirected_union",
                "leiden_resolution": 1.0, "primary_seed": 0, "stability_seed": 1,
                "leiden_iterations": -1, "classification": {"lifecycle_stage": "exploratory_screen",
                "study_axis": "fully_observed_representation_comparison", "scientific_variant": "all_observed_h0_hL_e151"}}
            write(out / "config.resolved.json", config)
            event(out, "preparation_started")
            inputs = prepare_extraction(source, out)
            require(inputs["checkpoint_epoch"] == 151 and inputs["n_cells"] == 407659, "Unexpected source dimensions/epoch")
            event(out, "preparation_complete", cells=inputs["n_cells"])
            base = ["scripts/analysis/so12_hidden_extraction.py", "--source-run-dir", str(source), "--output", str(out)]
            jobs(out, [("extraction_pilot", base + ["--device", "cuda:0", "--cores", "23", "--pilot"])])
            pilot = read(out / "diagnostics/extraction_pilot.json")
            require(pilot["peak_reserved_bytes"] < 22 * 2**30, "Largest-core GPU pilot has insufficient memory margin")
            event(out, "extraction_pilot_passed", seconds=pilot["elapsed_seconds"], peak_reserved_bytes=pilot["peak_reserved_bytes"])
            assignments = [[] for _ in range(4)]
            totals = [0] * 4
            for core in sorted(inputs["cores"], key=lambda x: -x["edges"]):
                if core["core_number"] == 23:
                    continue
                index = min(range(4), key=lambda x: totals[x])
                assignments[index].append(core["core_number"])
                totals[index] += core["edges"]
            write(out / "diagnostics/gpu_assignments.json", {"cores": assignments, "edge_counts": totals})
            jobs(out, [(f"extraction_gpu_{i}", base + ["--device", f"cuda:{i}", "--cores", ",".join(map(str, cores))]) for i, cores in enumerate(assignments)])
            extraction = collect_extraction(out)
            event(out, "extraction_complete", cells=extraction["n_cells"])
            cluster_base = ["scripts/analysis/so12_hidden_clustering.py", "--output", str(out)]
            jobs(out, [(f"knn_pilot_{rep}", cluster_base + ["--representation", rep, "--device", f"cuda:{i}", "--pilot"]) for i, rep in enumerate(("h0", "hL"))])
            event(out, "knn_pilots_passed")
            jobs(out, [(f"clustering_{rep}", cluster_base + ["--representation", rep, "--device", f"cuda:{i}"]) for i, rep in enumerate(("h0", "hL"))])
            clustering = collect_clustering(out)
            event(out, "clustering_complete")
        event(out, "rendering_started")
        figures = render_comparison(out, pca, cores=tuple(CORES))
        disk_gate(out)
        write(out / "READY_FOR_REVIEW.json", {"status": "awaiting_visual_verification", "time": now(),
              "evaluation_id": eid, "extraction": extraction, "clustering": clustering, "figures": figures})
        update_status(eid, out, "running", {"stage": "awaiting_visual_verification", "campaign_id": CAMPAIGN})
        event(out, "ready_for_visual_review")
    except BaseException as error:
        write(out / "FAILURE.json", {"time": now(), "error": str(error), "traceback": traceback.format_exc()})
        event(out, "failed", error=str(error))
        update_status(eid, out, "failed", {"error": str(error), "campaign_id": CAMPAIGN})
        raise


def verify_bundle(root):
    manifest = read(root / "manifest.json")
    require(read(root / "_SUCCESS")["manifest_sha256"] == sha256_file(root / "manifest.json"), "Completion marker mismatch")
    for relative, expected in manifest["files"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe manifest path")
        require(record(root / relative) == expected, f"Artifact checksum mismatch: {relative}")
    return manifest


def finalize(eid):
    paths = current_paths()
    out = paths.scratch_root / "active_runs" / eid
    final = paths.report_root / "analyses/so12_hidden_leiden" / eid
    require(not final.exists(), "Final destination already exists")
    ready = read(out / "READY_FOR_REVIEW.json")
    audit = read(out / "diagnostics/final_review.json")
    require(audit["status"] == "passed" and audit["visually_reviewed_cores"] == CORES,
            "All six images require a successful final review")
    event(out, "publication_verification_started")
    manifest = {"schema": "so12_hidden_leiden_report_v1", "evaluation_id": eid,
        "source_run_id": RUN, "pca_run_id": PCA_RUN, "checkpoint_sha256": CHECKPOINT_SHA,
        "completed_at": now(), "files": {str(p.relative_to(out)): record(p)
            for p in sorted(out.rglob("*")) if p.is_file() and p.name not in ("manifest.json", "_SUCCESS")}}
    write(out / "manifest.json", manifest)
    write(out / "_SUCCESS", {"status": "success", "manifest_sha256": sha256_file(out / "manifest.json"), "completed_at": now()})
    verify_bundle(out)
    final.parent.mkdir(parents=True, exist_ok=True)
    out.rename(final)
    verify_bundle(final)
    update_status(eid, final, "completed", {"campaign_id": CAMPAIGN, "checkpoint_epoch": 151,
        "n_cells": 407659, "displayed_cells": 117996, "clustering": ready["clustering"]})
    Registry().record_artifact(RUN, evaluation_id=eid, kind="so12_hidden_leiden_comparison",
        path=final / "manifest.json", sha256=sha256_file(final / "manifest.json"),
        size_bytes=(final / "manifest.json").stat().st_size, status="present")
    print(json.dumps({"status": "completed", "evaluation_id": eid, "artifact_path": str(final)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--retry-of")
    args = parser.parse_args()
    require(args.evaluation_id.startswith("ev_so12_h0_hl_") and "/" not in args.evaluation_id,
            "Invalid evaluation identifier")
    if args.finalize:
        finalize(args.evaluation_id)
    else:
        execute(args.evaluation_id, args.retry_of)
