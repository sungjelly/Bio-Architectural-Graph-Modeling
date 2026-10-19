"""Verify the fixed BCE run's completed lifecycle without altering its sources.

Task contract: distinguish verified publication from stale status or a finished
training process. Require terminal registry/queue records, a checksum-bound
successful immutable bundle, and a verified cataloged best checkpoint whose
identity, metric and run semantics agree. Stop on any missing or inconsistent
evidence. This audits existing replay evidence; it performs no model inference,
raw-data access, GPU work, registry writes, or scientific hypothesis test.

After completion, run from the BAGM root (BAGM_* path overrides apply)::

    PYTHONPATH=src /venv/main/bin/python state/monitoring/verify_bce_completion.py --check-only
    PYTHONPATH=src /venv/main/bin/python state/monitoring/verify_bce_completion.py

The second command creates an exclusive completion_verification_v1.json receipt
and an exact verifier-source snapshot beside it, outside the immutable bundle.
The separate report builder audits histories, coverage and secondary metrics;
the monitoring task owns report inspection and campaign/result documentation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

import yaml

from spatial_benchmark.paths import current_paths
from spatial_benchmark.run_archive import RunArchiveError, verify_run_bundle


RUN_ID = "r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863"
JOB_ID = "q_a3649dc5f738ae679625"
CAMPAIGN_ID = "cmp_20260930_so2_target_isolated_geometry_modulated_binary_expression_bce_once_per_cell_seed0"
PRIMARY = "val/target_only_masked_binary_cross_entropy"
STUDY_AXIS = "so2_donor_grouped_target_isolated_geometry_modulated_binary_expression_bce"
RETENTION = "retain_exploratory_validation_selected_binary_expression_model"
RECEIPT_NAME = "completion_verification_v1.json"
SNAPSHOT_NAME = "verify_bce_completion_v1.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def one(connection: sqlite3.Connection, query: str, parameters: tuple = ()) -> dict[str, Any]:
    records = connection.execute(query, parameters).fetchall()
    require(len(records) == 1, "Expected exactly one matching registry record")
    return dict(records[0])


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"Missing or linked source: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def same_metric(left: Any, right: Any, label: str, tolerance: float = 1e-12) -> None:
    a, b = float(left), float(right)
    require(math.isfinite(a) and math.isfinite(b) and
            math.isclose(a, b, rel_tol=0, abs_tol=tolerance), f"Metric mismatch: {label}")


def terminal_records(run: dict, job: dict) -> None:
    require(run["run_id"] == job["run_id"] == RUN_ID and job["job_id"] == JOB_ID,
            "Run/job identity mismatch")
    require(run["campaign_id"] == job["campaign_id"] == CAMPAIGN_ID, "Campaign mismatch")
    require(run["status"] == job["status"] == "completed", "Run and queue must both be completed")
    require(run["failure_category"] is None and job["failure_category"] is None, "Terminal failure recorded")
    require((run["seed"], run["fold"], run["attempt"], job["attempt_count"]) == (0, 0, 1, 1),
            "Seed/fold/attempt changed")
    for record, first, last in ((run, "start_time", "end_time"), (job, "started_at", "finished_at")):
        require(bool(record[first]) and bool(record[last]), "Terminal timestamps missing")
        start = datetime.fromisoformat(record[first].replace("Z", "+00:00"))
        end = datetime.fromisoformat(record[last].replace("Z", "+00:00"))
        require(start.tzinfo is not None and end.tzinfo is not None and end >= start,
                "Invalid terminal timestamps")
    require(math.isfinite(float(run["duration_seconds"])) and run["duration_seconds"] > 0,
            "Registry runtime is invalid")


def verify_completion() -> dict[str, Any]:
    paths = current_paths()
    database = paths.state_root / "tracking/bagm.sqlite3"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        run = one(connection, """SELECT run_id,campaign_id,status,seed,fold,attempt,start_time,end_time,
            duration_seconds,primary_metric_name,primary_metric_value,artifact_path,failure_category
            FROM runs WHERE run_id=?""", (RUN_ID,))
        job = one(connection, """SELECT job_id,run_id,campaign_id,status,attempt_count,started_at,
            finished_at,failure_category FROM queue_jobs WHERE run_id=?""", (RUN_ID,))
        terminal_records(run, job)  # Fail before reading any in-progress bundle.
        schema = one(connection, "SELECT MAX(version) AS version FROM schema_migrations")["version"]
        require(schema >= 3, "Registry lacks schema-v3 checkpoint semantics")
        checkpoint = one(connection, """SELECT a.artifact_id,a.kind,a.path,a.status AS artifact_status,
            a.sha256,a.size_bytes,c.role,c.best_epoch,c.monitored_metric,c.monitored_mode,c.monitored_value,
            c.retention_class,c.verification_status FROM artifacts a LEFT JOIN checkpoint_catalog c
            ON c.artifact_id=a.artifact_id AND c.run_id=a.run_id
            WHERE a.run_id=? AND (a.kind='checkpoints' OR a.path LIKE '%.ckpt')""", (RUN_ID,))
        catalog_count = one(connection, "SELECT COUNT(*) AS n FROM checkpoint_catalog WHERE run_id=?", (RUN_ID,))["n"]
        require(catalog_count == 1, "Expected exactly one cataloged checkpoint")
        semantics = one(connection, """SELECT a.alias_id,a.alias_type,a.preferred,c.lifecycle_stage,
            c.study_axis,c.classification_confidence,c.seed_known,c.fold_known,c.attempt_known,
            c.retention_class,c.category_key FROM run_categories c LEFT JOIN run_aliases a
            ON a.run_id=c.run_id AND a.preferred=1 WHERE c.run_id=?""", (RUN_ID,))
    matches = list((paths.artifact_root / "runs").glob(f"*/*/{RUN_ID}"))
    require(len(matches) == 1, "Expected exactly one published run bundle")
    bundle = matches[0]
    require(not bundle.is_symlink() and Path(run["artifact_path"]) == bundle, "Registered artifact path mismatch")
    require(not (paths.scratch_root / "active_runs" / RUN_ID).exists(), "Active scratch still exists")
    require((bundle / "_SUCCESS").is_file(), "Success marker missing")
    verification = verify_run_bundle(bundle, require_success_contract=True)
    require(verification["valid"] and verification["status"] == "success", "Bundle verification failed")
    read = lambda name: json.loads((bundle / name).read_text(encoding="utf-8"))
    summary, replay, metrics = (read(name) for name in (
        "summary.json", "diagnostics/final_checkpoint_reload_verification.json", "metrics/final.json"))
    provenance = read("provenance/so2_target_isolated_binary_expression_training.json")
    manifest = yaml.safe_load((bundle / "manifest.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((bundle / "config.resolved.yaml").read_text(encoding="utf-8"))
    require(summary["run_id"] == manifest["run_id"] == RUN_ID and summary["status"] == "success",
            "Summary/manifest identity or status mismatch")
    require(summary["campaign_id"] == manifest["campaign_id"] == CAMPAIGN_ID and manifest["job_id"] == JOB_ID,
            "Bundle campaign/job mismatch")
    require(summary["model_seed"] == 0 and summary["checkpoint_count"] == 1 and
            summary["checkpoint"] == "checkpoints/best.ckpt", "Summary checkpoint or seed mismatch")
    require(summary["checkpoint_reload_verified"] is True and replay["verified"] is True,
            "Checkpoint replay unverified")
    require(summary["test_artifacts_present"] is False and config["dataset"]["test_partition_present"] is False,
            "Unexpected test scope")
    require(1 <= summary["best_epoch"] <= summary["final_epoch"] <= 300 and summary["final_epoch"] >= 50,
            "Invalid best/final epochs")
    require(replay["best_epoch"] == checkpoint["best_epoch"] == summary["best_epoch"], "Best epoch mismatch")
    require(replay["checkpoint"] == "checkpoints/best.ckpt", "Replay checkpoint path mismatch")
    require(checkpoint["kind"] == "checkpoints" and checkpoint["artifact_status"] == "present" and
            checkpoint["role"] == "best" and checkpoint["verification_status"] == "verified" and
            checkpoint["monitored_mode"] == "min", "Checkpoint catalog is incomplete or unverified")
    require(checkpoint["monitored_metric"] == run["primary_metric_name"] == summary["primary_metric_name"] ==
            manifest["primary_metric_name"] == PRIMARY, "Primary metric name mismatch")
    for label, value in (("registry", run["primary_metric_value"]), ("catalog", checkpoint["monitored_value"]),
                         ("manifest", manifest["primary_metric_value"]), ("summary", summary["primary_metric_value"]),
                         ("replayed", replay["replayed_validation_metric"])):
        same_metric(value, metrics[PRIMARY], label)
    same_metric(replay["best_validation_metric"], metrics[PRIMARY], "selected/replayed", tolerance=1e-6)
    best = bundle / "checkpoints/best.ckpt"
    require(sorted(p.name for p in best.parent.iterdir()) == ["best.ckpt"], "Unexpected retained checkpoints")
    actual = file_record(best)
    require(Path(checkpoint["path"]) == best and checkpoint["sha256"] == actual["sha256"] ==
            replay["checkpoint_sha256"] == provenance["checkpoint_sha256"] and
            checkpoint["size_bytes"] == actual["size_bytes"], "Checkpoint content registration mismatch")
    require(semantics["alias_type"] == "semantic" and semantics["preferred"] == 1 and
            bool(semantics["alias_id"]), "Preferred semantic alias missing")
    require((semantics["seed_known"], semantics["fold_known"], semantics["attempt_known"]) == (1, 1, 1),
            "Seed/fold/attempt semantics unknown")
    for key, expected in (("lifecycle_stage", "exploratory_screen"), ("study_axis", STUDY_AXIS),
                          ("retention_class", RETENTION), ("classification_confidence", "high")):
        require(semantics[key] == manifest["checkpoint_catalog"][key] == config["classification"][key] == expected,
                f"Classification mismatch: {key}")
    require(checkpoint["retention_class"] == RETENTION and semantics["category_key"] ==
            manifest["checkpoint_catalog"]["category_key"] and semantics["alias_id"] ==
            manifest["checkpoint_catalog"]["semantic_alias"], "Checkpoint/run semantics disagree")
    sources = {name: file_record(bundle / name) for name in (
        "_SUCCESS", "summary.json", "manifest.yaml", "config.resolved.yaml", "metrics/final.json",
        "diagnostics/final_checkpoint_reload_verification.json", "provenance/artifact_checksums.json",
        "provenance/so2_target_isolated_binary_expression_training.json", "checkpoints/best.ckpt")}
    return {"schema": "so2_binary_expression_bce_completion_verification_v1", "status": "verified",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID,
            "registry_database": str(database), "registry_access": "mode=ro; query_only; transaction_snapshot",
            "registry_schema_version": schema, "run": run, "queue_job": job, "checkpoint_catalog": checkpoint,
            "semantics": semantics, "bundle_verification": verification, "sources": sources,
            "scope": "Lifecycle, existing best-checkpoint replay evidence and catalog verification only; histories, secondary metrics, report inspection and documentation are verified separately."}


def exclusive_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="Verify and print; create no files")
    args = parser.parse_args()
    try:
        receipt = verify_completion()
        script = Path(__file__).read_bytes()
        receipt["verifier"] = {"path": SNAPSHOT_NAME, "sha256": hashlib.sha256(script).hexdigest(), "size_bytes": len(script)}
        payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if args.check_only:
            print(payload.decode(), end="")
        else:
            directory = current_paths().report_root / "analyses/so2_binary_expression_bce" / RUN_ID
            target, snapshot = directory / RECEIPT_NAME, directory / SNAPSHOT_NAME
            require(not target.exists() and not target.is_symlink(), "Completion receipt already exists")
            directory.mkdir(parents=True, exist_ok=True)
            if snapshot.exists() or snapshot.is_symlink():
                require(snapshot.is_file() and not snapshot.is_symlink() and snapshot.read_bytes() == script,
                        "Existing verifier snapshot differs")
            else:
                exclusive_bytes(snapshot, script)
            exclusive_bytes(target, payload)
            print(json.dumps({"status": "verified", "receipt": str(target), **file_record(target)}, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, RunArchiveError, yaml.YAMLError) as error:
        print(f"Completion verification failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
