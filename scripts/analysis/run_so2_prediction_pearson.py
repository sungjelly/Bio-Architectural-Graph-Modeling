#!/usr/bin/env python3
"""Registered, immutable Pearson replays of three existing SO2 checkpoints.

See experiments/campaigns/cmp_20261015_so2_prediction_pearson/frozen_task_contract.md.
Preparation does not run inference. Helpers execute in fresh Python processes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry

CAMPAIGN = "cmp_20261015_so2_prediction_pearson"
PROTOCOL = "so2_prediction_pearson_v1"
RUNS = {
    "local_s0": "r_20261004T072107Z_9d2b00cb_s000_f00_a01_f41786d5",
    "local_s1": "r_20261004T072107Z_9d2b00cb_s001_f00_a01_1caff983",
    "historical_e300": "r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def locations(key):
    paths = current_paths()
    run_id = RUNS[key]
    source = paths.artifact_root / "runs" / run_id[2:6] / run_id[6:8] / run_id
    stage = paths.scratch_root / "active_runs" / run_id / "posthoc_reports/so2_prediction_pearson/v1"
    final = paths.report_root / "analyses/so2_prediction_pearson/v1" / key
    checkpoint = source / "checkpoints" / ("last.ckpt" if key == "historical_e300" else "best.ckpt")
    return paths, run_id, source, stage, final, checkpoint


def event(stage, phase, **fields):
    value = {"time": now(), "phase": phase, **fields}
    line = json.dumps(value, allow_nan=False)
    with (stage / "metrics/events.jsonl").open("a") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(line, flush=True)


def verified(path, expected):
    if not Path(path).is_file() or sha(path) != expected:
        raise RuntimeError(f"Source checksum mismatch: {path}")


def source_files(project_root):
    """Freeze both actual import package and main package, including helper sources."""
    package_spec = importlib.util.find_spec("spatial_benchmark")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError("Cannot resolve spatial_benchmark source package")
    actual = Path(next(iter(package_spec.submodule_search_locations))).resolve()
    trees = {"import_package": actual, "main_package": project_root / "src/spatial_benchmark"}
    files = {}
    for label, root in trees.items():
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                files[str(Path(label) / path.relative_to(root))] = path.resolve()
    for filename in ("run_so2_prediction_pearson.py", "pearson_local_replay.py", "pearson_historical_replay.py"):
        path = project_root / "scripts/analysis" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Complete replay helper before preparation: {path}")
        files[str(Path("analysis") / filename)] = path.resolve()
    for filename in ("bagm-prediction-pearson.sh", "bagm-prediction-pearson.conf"):
        path = project_root / "ops/supervisor" / filename
        files[str(Path("supervisor") / filename)] = path.resolve()
    return files


def source_audit(cfg, stage):
    for absolute, record in cfg["source_files"].items():
        verified(absolute, record["sha256"])
        verified(stage / record["snapshot"], record["sha256"])
    for record in cfg["upstream_files"]:
        verified(record["path"], record["sha256"])
        if record.get("snapshot"):
            verified(stage / record["snapshot"], record["sha256"])
    verified(cfg["contract_path"], cfg["contract_sha256"])
    verified(stage / "provenance/frozen_task_contract.md", cfg["contract_sha256"])


def evaluation_row(registry, evaluation_id):
    with registry.connect() as connection:
        row = connection.execute("SELECT * FROM evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone()
    return None if row is None else dict(row)


def prepare(key):
    paths, run_id, source, stage, final, checkpoint = locations(key)
    if final.exists():
        return verify(final)
    registry = Registry(initialize=False)
    if (stage / "config.resolved.json").exists():
        cfg = read(stage / "config.resolved.json")
        source_audit(cfg, stage)
        existing = evaluation_row(registry, cfg["evaluation_id"])
        if existing is None or existing["status"] != "pending" or (stage / "_FAILED").exists():
            raise RuntimeError("Existing preparation is not pending; do not overwrite or silently retry")
        print(json.dumps({"model": key, "status": "already_prepared"}), flush=True)
        return cfg
    if stage.exists() and any(stage.iterdir()):
        raise RuntimeError(f"Incomplete prior preparation retained at {stage}; inspect before a new version")
    contract = paths.project_root / "experiments/campaigns" / CAMPAIGN / "frozen_task_contract.md"
    contract_hash = sha(contract)
    files = source_files(paths.project_root)
    run = registry.get_run(run_id)
    if run is None or run["status"] != "completed":
        raise RuntimeError(f"Expected completed registered source run: {run_id}")
    upstream_manifest = source / "provenance/artifact_checksums.json"
    archive = read(upstream_manifest)["files"]
    checkpoint_rel = str(checkpoint.relative_to(source))
    expected_checkpoint = archive[checkpoint_rel]["sha256"]
    verified(checkpoint, expected_checkpoint)
    config_rel = "config.resolved.yaml"
    verified(source / config_rel, archive[config_rel]["sha256"])
    campaign_cfg = {"protocol": PROTOCOL, "source_run_ids": list(RUNS.values()),
                    "contract_sha256": contract_hash, "analysis_design": "exploratory_posthoc",
                    "new_training": False}
    existing_campaign = registry.get_campaign(CAMPAIGN)
    if existing_campaign is None:
        registry.create_campaign(CAMPAIGN, name="SO2 masked-expression Pearson replay",
            scientific_question="What are cell-wise, gene-wise and pooled masked-expression correlations of selected existing predictors?",
            status="pilot", config=campaign_cfg)
    elif existing_campaign["config"] != campaign_cfg:
        raise RuntimeError("Existing campaign contract/configuration differs")
    for directory in ("source", "provenance", "metrics", "logs"):
        (stage / directory).mkdir(parents=True, exist_ok=True)
    frozen_sources = {}
    for relative, path in files.items():
        snapshot = Path("source") / relative
        destination = stage / snapshot
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = sha(path)
        shutil.copyfile(path, destination)
        verified(path, before)
        verified(destination, before)
        frozen_sources[str(path)] = {"sha256": before, "snapshot": str(snapshot)}
    shutil.copyfile(contract, stage / "provenance/frozen_task_contract.md")
    upstream = [{"path": str(checkpoint), "sha256": expected_checkpoint}]
    for path, snapshot in ((source / config_rel, "provenance/source_config.resolved.yaml"),
                           (upstream_manifest, "provenance/source_artifact_checksums.json")):
        shutil.copyfile(path, stage / snapshot)
        upstream.append({"path": str(path), "sha256": sha(path), "snapshot": snapshot})
    evaluation_id = "ev_so2_prediction_pearson_v1_" + run_id
    cfg = {"schema": PROTOCOL, "campaign_id": CAMPAIGN, "evaluation_id": evaluation_id,
           "model_key": key, "source_run_id": run_id, "source_status": run["status"],
           "checkpoint_path": str(checkpoint), "checkpoint_sha256": expected_checkpoint,
           "checkpoint_role": checkpoint.stem, "source_files": frozen_sources,
           "upstream_files": upstream, "contract_path": str(contract), "contract_sha256": contract_hash,
           "dataset_id": run["dataset_id"], "split_id": run["split_id"],
           "seed": run["seed"], "new_training": False, "prepared_at": now(),
           "helper": "pearson_historical_replay" if key == "historical_e300" else "pearson_local_replay",
           "helper_function": "evaluate_historical" if key == "historical_e300" else "evaluate_local",
           "analysis_design": "exploratory_posthoc",
           "evaluated_population": "fitted_cohort" if key == "historical_e300" else "reused_validation_donor",
           "command": [sys.executable, str(Path(__file__).resolve()), "--model", key]}
    write(stage / "config.resolved.json", cfg)
    write(stage / "provenance/environment.json", {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "cwd": str(Path.cwd()), "sys_path": sys.path, "prepared_at": now(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    })
    commands = {"dependencies.txt": [sys.executable, "-m", "pip", "freeze"],
                "git_commit.txt": ["git", "rev-parse", "HEAD"],
                "git_status.txt": ["git", "status", "--short"],
                "git_diff.patch": ["git", "diff", "HEAD"],
                "nvidia-smi.txt": ["nvidia-smi"]}
    for filename, command in commands.items():
        result = subprocess.run(command, cwd=paths.project_root, text=True, capture_output=True, check=False)
        (stage / "provenance" / filename).write_text(result.stdout + result.stderr)
        if result.returncode and filename != "nvidia-smi.txt":
            raise RuntimeError(f"Provenance command failed: {command}")
    (stage / "logs/stderr.log").touch()
    source_audit(cfg, stage)
    registry.create_evaluation(evaluation_id, run_id=run_id, checkpoint_name=checkpoint.stem,
        dataset_id=run["dataset_id"], split_id=run["split_id"], status="pending", metrics=cfg, artifact_path=stage)
    event(stage, "prepared", model=key, source_run_id=run_id, evaluation_id=evaluation_id)
    return cfg


def set_status(cfg, root, status, metrics):
    registry = Registry(initialize=False)
    with registry.transaction(immediate=True) as connection:
        sql = "UPDATE evaluations SET status=?,metrics_json=?,artifact_path=?,finished_at=? WHERE evaluation_id=? AND run_id=?"
        if status == "running":
            sql += " AND status='pending'"
        cursor = connection.execute(
            sql,
            (status, json.dumps(metrics, allow_nan=False), str(root), None if status == "running" else now(),
             cfg["evaluation_id"], cfg["source_run_id"]))
        if cursor.rowcount != 1:
            raise RuntimeError("Expected one registered evaluation")


def verify(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    if (root / "_FAILED").exists():
        raise RuntimeError("Failure receipt prohibits successful verification")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*")
                    if path.is_file() and path.name not in ("manifest.json", "_SUCCESS")}
    if actual_files != set(manifest["files"]):
        raise RuntimeError("Manifest file inventory mismatch")
    if manifest["schema"] != PROTOCOL:
        raise RuntimeError("Incorrect manifest schema")
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Invalid manifest path")
        verified(path, record["sha256"])
        if path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"Size mismatch: {path}")
    success = read(root / "_SUCCESS")
    if success["status"] != "success" or success["manifest_sha256"] != sha(root / "manifest.json"):
        raise RuntimeError("Invalid success receipt")
    result = read(root / "result.json")
    if not isinstance(result, dict) or not result:
        raise RuntimeError("Missing evaluation result")
    if result.get("status") != "completed" or result.get("verified") is not True:
        raise RuntimeError("Helper result does not certify completion")
    if result.get("source_run_id") != manifest["source_run_id"]:
        raise RuntimeError("Helper result and manifest source identity differ")
    json.dumps(result, allow_nan=False)
    return manifest


_HELPER_CODE = r'''
import importlib,json,os,sys
from pathlib import Path
module=importlib.import_module(sys.argv[1])
result=getattr(module,sys.argv[2])(Path(sys.argv[3]),Path(sys.argv[4]))
if not isinstance(result,dict) or not result:
    raise RuntimeError("Replay helper must return a nonempty result dictionary")
if result.get("status") != "completed" or result.get("verified") is not True:
    raise RuntimeError("Replay helper did not complete verification")
path=Path(sys.argv[3])/"result.json"
temporary=path.with_suffix(".json.tmp")
with temporary.open("w") as stream:
    json.dump(result,stream,indent=2,allow_nan=False)
    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
os.replace(temporary,path)
'''


def execute(key):
    paths, run_id, source, stage, final, checkpoint = locations(key)
    if final.exists():
        verify(final)
        print(json.dumps({"model": key, "status": "verified", "path": str(final)}), flush=True)
        return
    cfg = read(stage / "config.resolved.json")
    registry = Registry(initialize=False)
    row = evaluation_row(registry, cfg["evaluation_id"])
    if row is None or row["status"] != "pending" or (stage / "_FAILED").exists():
        raise RuntimeError("Expected pending prepared evaluation; previous failure/running evidence must be preserved")
    if cfg["model_key"] != key or cfg["source_run_id"] != run_id:
        raise RuntimeError("Prepared source identity mismatch")
    try:
        source_audit(cfg, stage)
        set_status(cfg, stage, "running", cfg)
        write(stage / "provenance/execution_environment.json", {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "started_at": now(), "command": sys.argv})
        event(stage, "started", model=key, command=sys.argv)
        command = [sys.executable, "-c", _HELPER_CODE, cfg["helper"], cfg["helper_function"], str(stage), str(checkpoint)]
        with (stage / "logs/stdout.log").open("a") as stdout, (stage / "logs/stderr.log").open("a") as stderr:
            subprocess.run(command, cwd=paths.project_root, check=True, stdout=stdout, stderr=stderr)
        source_audit(cfg, stage)
        result = read(stage / "result.json")
        if result.get("source_run_id") != run_id:
            raise RuntimeError("Helper returned a different source run")
        if result.get("status") != "completed" or result.get("verified") is not True:
            raise RuntimeError("Helper did not pass its completion checks")
        event(stage, "evaluation_complete", model=key, evaluation_id=cfg["evaluation_id"])
        files = {str(path.relative_to(stage)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
                 for path in sorted(stage.rglob("*")) if path.is_file() and path.name not in ("manifest.json", "_SUCCESS")}
        write(stage / "manifest.json", {"schema": PROTOCOL, "evaluation_id": cfg["evaluation_id"],
            "source_run_id": run_id, "model_key": key, "checkpoint_sha256": cfg["checkpoint_sha256"], "files": files})
        write(stage / "_SUCCESS", {"status": "success", "manifest_sha256": sha(stage / "manifest.json"), "completed_at": now()})
        verify(stage)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise RuntimeError("Immutable publication already exists")
        os.rename(stage, final)
        verify(final)
        set_status(cfg, final, "completed", result)
        registry.record_artifact(run_id, evaluation_id=cfg["evaluation_id"], kind="posthoc_prediction_pearson_manifest_v1",
            path=final / "manifest.json", sha256=sha(final / "manifest.json"),
            size_bytes=(final / "manifest.json").stat().st_size, status="present")
        print(json.dumps({"model": key, "status": "published", "path": str(final)}), flush=True)
    except BaseException:
        error = traceback.format_exc()
        if stage.exists():
            with (stage / "logs/stderr.log").open("a") as stream:
                stream.write(error)
            write(stage / "_FAILED", {"failed_at": now(), "traceback": error})
            set_status(cfg, stage, "failed", {"error": error, "model_key": key})
        else:
            # A published immutable bundle is never edited after a registry failure.
            print("Immutable publication exists; registry completion may require reconciliation: " + str(final), file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Register and freeze all models, or only --model")
    parser.add_argument("--model", choices=RUNS)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.prepare and args.verify:
        parser.error("--prepare and --verify are mutually exclusive")
    if args.prepare:
        for key in ([args.model] if args.model else RUNS):
            prepare(key)
    elif args.verify:
        for key in ([args.model] if args.model else RUNS):
            verify(locations(key)[4])
            print(json.dumps({"model": key, "status": "verified"}), flush=True)
    elif args.model:
        execute(args.model)
    else:
        parser.error("Supply --prepare, --verify, or --model")


if __name__ == "__main__":
    main()
