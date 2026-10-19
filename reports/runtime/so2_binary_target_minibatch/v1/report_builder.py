#!/usr/bin/env python3
"""Compare verified SO2 BCE target-minibatch pilots using resource metrics only."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
import shutil
import uuid

import yaml

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, _rename_no_replace, verify_run_bundle
from spatial_benchmark.so2_binary_target_minibatch import (
    CAMPAIGN, PREFLIGHT_RELATIVE_PATH, SOURCE_RUN, optimizer_updates,
    validate_configuration,
)

GIB = 1024 ** 3


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def load_pilot(run_id: str, paths, registry: Registry) -> dict:
    root = RunArchive.artifact_path_for(run_id, paths)
    record = registry.get_run(run_id)
    require(bool(record) and record["status"] == "completed", f"Unfinished run: {run_id}")
    require(Path(record["artifact_path"]).resolve() == root.resolve(), "Registry path mismatch")
    verify_run_bundle(root)
    config = yaml.safe_load((root / "config.resolved.yaml").read_text())
    validate_configuration(config)
    require(config["metadata"]["execution_profile"] == "pilot", "Expected a pilot")
    receipt = json.loads((root / PREFLIGHT_RELATIVE_PATH).read_text())
    summary = json.loads((root / "summary.json").read_text())
    batch = config["trainer"]["target_cells_per_optimizer_update"]
    require(receipt.get("passed") is True and receipt.get("coverage_verified") is True
            and receipt.get("finite_gradients") is True, "Pilot failed correctness checks")
    require(receipt["target_cells_per_optimizer_update"] == batch, "Batch mismatch")
    require(receipt["optimizer_updates_per_global_epoch"] == optimizer_updates(batch), "Update count mismatch")
    require(receipt["completed_epochs"] == summary["final_epoch"] == 2, "Pilot must finish two epochs")
    require(summary["checkpoint_reload_verified"] is True, "Checkpoint replay was not verified")
    with (root / "results/epoch_metrics.csv").open() as handle:
        epochs = list(csv.DictReader(handle))
    require(len(epochs) == 2 and [int(row["global_epoch"]) for row in epochs] == [1, 2], "Incomplete epoch history")
    require(all(int(row["train_target_cells"]) == 208696 and int(row["validation_target_cells"]) == 37367
                for row in epochs), "Incomplete target coverage")
    durations = [float(row["epoch_duration_seconds"]) for row in epochs]
    require(all(math.isfinite(value) and value > 0 for value in durations), "Invalid timing")
    gate = receipt["resource_gates"]
    require(gate["resource_gate_passed"] is True and gate["headroom_bytes"] >= 2 * GIB,
            "Resource gate failed")
    require(gate["peak_allocated_bytes"] <= 22 * GIB and gate["peak_reserved_bytes"] <= 22 * GIB,
            "Memory maximum failed")
    require(gate["minimum_sampled_physical_free_bytes"] >= 2 * GIB, "Missing physical memory headroom")
    samples = [json.loads(line) for line in (root / "diagnostics/target_minibatch_resource_samples.jsonl").read_text().splitlines() if line.strip()]
    require(bool(samples), "Missing resource samples")
    rss = max(max(row.get("process_peak_rss_bytes", 0), row.get("rss", {}).get("VmRSS_bytes", 0)) for row in samples)
    cgroup = max(row.get("cgroup", {}).get("memory.current", 0) for row in samples)
    require(rss > 0 and 0 < cgroup < 160 * GIB, "Host-memory gate failed")
    source_files = ("_SUCCESS", "config.resolved.yaml", "summary.json", PREFLIGHT_RELATIVE_PATH,
                    "results/epoch_metrics.csv", "diagnostics/target_minibatch_resource_samples.jsonl",
                    "provenance/artifact_checksums.json")
    return {
        "run_id": run_id, "target_batch_size": batch,
        "physical_gpu": config["launcher"]["requested_gpu"],
        "optimizer_steps_per_epoch": optimizer_updates(batch),
        "epoch_seconds": durations, "mean_epoch_seconds": mean(durations),
        "training_targets_per_second_including_validation": 208696 / mean(durations),
        "peak_allocated_gib": gate["peak_allocated_bytes"] / GIB,
        "peak_reserved_gib": gate["peak_reserved_bytes"] / GIB,
        "minimum_physical_free_gib": gate["minimum_sampled_physical_free_bytes"] / GIB,
        "peak_process_rss_gib": rss / GIB, "peak_shared_cgroup_gib": cgroup / GIB,
        "estimated_50_epoch_hours": mean(durations) * 50 / 3600,
        "estimated_180_epoch_hours": mean(durations) * 180 / 3600,
        "correctness_and_resource_gates_passed": True,
        "source_hashes": {name: sha256_file(root / name) for name in source_files},
    }


def load_failures(paths, registry: Registry) -> list[dict]:
    with registry.connect() as connection:
        records = connection.execute(
            "SELECT run_id, artifact_path FROM runs WHERE campaign_id = ? AND status = 'failed' ORDER BY run_id",
            (CAMPAIGN,),
        ).fetchall()
    failures = []
    for record in records:
        root = Path(record["artifact_path"])
        verify_run_bundle(root)
        require((root / "_FAILED").is_file(), "Failed registry run has no failure marker")
        summary = json.loads((root / "summary.json").read_text())
        stderr = (root / "logs/stderr.log").read_text()
        reason = ("Pilot minimum_epochs equaled maximum_epochs; the epoch controller requires a strict inequality. "
                  "The attempt completed its first training and validation pass, then failed before checkpoint commit.") \
                 if "minimum_epochs must be below maximum_epochs" in stderr else summary.get("failure_message", "See preserved stderr")
        failures.append({"run_id": record["run_id"], "reason": reason,
                         "source_hashes": {name: sha256_file(root / name) for name in
                            ("_FAILED", "summary.json", "logs/stderr.log", "provenance/artifact_checksums.json")}})
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-runs", nargs=3, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(argv)
    require(args.version.isalnum(), "Report version must be alphanumeric")
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    rows = sorted((load_pilot(run_id, paths, registry) for run_id in args.pilot_runs), key=lambda row: row["target_batch_size"])
    require([row["target_batch_size"] for row in rows] == [512, 1024, 2048], "Expected each prespecified candidate once")
    fastest = min(rows, key=lambda row: row["mean_epoch_seconds"])
    preferred = rows[1]
    selected = preferred if preferred["mean_epoch_seconds"] <= 1.25 * fastest["mean_epoch_seconds"] else fastest
    failures = load_failures(paths, registry)
    report = {
        "schema": "so2_binary_target_minibatch_resource_selection_v1", "campaign_id": CAMPAIGN,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_run_id": SOURCE_RUN,
        "selection_rule": "prefer_1024_if_passing_and_within_25_percent_of_fastest_otherwise_fastest_passing",
        "selection_uses_validation_score": False, "selected_pilot_run_id": selected["run_id"],
        "selected_target_batch_size": selected["target_batch_size"], "pilots": rows,
        "failed_attempts": failures,
        "limitations": ["Two epochs only; timings include validation but exclude startup, checkpoint I/O and final replay.",
            "Candidates ran on different cards with concurrent host and I/O load.",
            "Runtime pilots are not independent scientific replicates or evidence of predictive superiority.",
            "ETA scales observed epoch time and does not predict early stopping or future contention."]}
    output = paths.report_root / "runtime/so2_binary_target_minibatch" / args.version
    require(not output.exists(), "Versioned report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    (temporary / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = [key for key in rows[0] if key not in {"source_hashes", "epoch_seconds"}]
    with (temporary / "table.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    lines = ["# SO2 BCE target-cell batch resource selection", "",
        f"Selected **{selected['target_batch_size']:,} target cells per optimizer update**, using the prespecified resource rule.",
        "All three registered two-epoch pilots passed coverage, numerical, memory and best-checkpoint replay checks.", "",
        "| Targets/update | Updates/epoch | GPU | Epoch 1 / 2 (s) | Mean (s) | Peak allocated / reserved (GiB) | Min free (GiB) |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['target_batch_size']} | {row['optimizer_steps_per_epoch']} | {row['physical_gpu']} | "
            f"{row['epoch_seconds'][0]:.1f} / {row['epoch_seconds'][1]:.1f} | {row['mean_epoch_seconds']:.1f} | "
            f"{row['peak_allocated_gib']:.2f} / {row['peak_reserved_gib']:.2f} | {row['minimum_physical_free_gib']:.2f} |")
    lines += ["", "Every epoch targets all 208,696 training cells once and evaluates all 37,367 validation cells with fixed masks. "
        "Balanced contiguous windows are shuffled within each shuffled core. The context graph remains complete; learned source embeddings are recomputed after each update.", "",
        "The historical paired-core run used six optimizer steps per epoch. These candidates change the optimization budget and stochastic trajectory. "
        "An eventual comparison must show epochs, optimizer steps and elapsed time; these pilot scores do not establish predictive improvement.", "",
        "Timing limitations: " + " ".join(report["limitations"]), "", "Immutable source runs:", ""]
    lines += [f"- B={row['target_batch_size']}: `{row['run_id']}`" for row in rows]
    if failures:
        lines += ["", "Failed attempts (preserved, excluded from successful timing comparisons):", ""]
        lines += [f"- `{row['run_id']}`: {row['reason']}" for row in failures]
    (temporary / "report.md").write_text("\n".join(lines) + "\n")
    shutil.copyfile(Path(__file__), temporary / "report_builder.py")
    manifest = {"schema": "checksummed_local_report_v1", "files": {
        file.name: sha256_file(file) for file in sorted(temporary.iterdir()) if file.is_file()}}
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _rename_no_replace(temporary, output)
    print(json.dumps({"report": str(output), "selected_target_batch_size": selected["target_batch_size"],
                      "selected_pilot_run_id": selected["run_id"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
