#!/usr/bin/env python3
"""Report one isolated and four concurrent, finalized SO2 hurdle runtime trials.

Run from the source worktree with the campaign README's shared BAGM path
overrides. Only verified immutable successful bundles are accepted. This
builder never reads expression matrices, checkpoints, or clinical records.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean
import sys
from typing import Any
import uuid

import yaml

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.run_archive import RunArchive, _rename_no_replace, verify_run_bundle
from spatial_benchmark.so2_hurdle_runtime_trial import (
    GIB, SOURCE_CONFIG_SHA, SOURCE_RUN, validate_outputs,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(rows) and all(isinstance(row, dict) for row in rows), f"Empty/malformed scalar evidence: {path.name}")
    return rows


def timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "Runtime timestamps must have a time zone.")
    return parsed.timestamp()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def load_run(run_id: str, *, role: str, paths: Any) -> dict[str, Any]:
    root = RunArchive.artifact_path_for(run_id, paths)
    require(root.is_dir() and (root / "_SUCCESS").is_file(), f"Run is not a published successful bundle: {run_id}")
    verify_run_bundle(root)
    config = yaml.safe_load((root / "config.resolved.yaml").read_text())
    result = validate_outputs(root, run_id, config)
    environment = read_json(root / "diagnostics/runtime_environment.json")
    resources = read_jsonl(root / "diagnostics/runtime_resources.jsonl")
    progress = read_jsonl(root / "metrics/runtime_progress.jsonl")
    epochs = result["epochs"]
    require(finite(result) and finite(resources), f"Nonfinite scalar evidence: {run_id}")
    require(canonical_sha256(config["source_configuration"]) == SOURCE_CONFIG_SHA, "Scientific source differs.")
    require(environment["visible_physical_gpu"] == config["runtime"]["gpu"], "GPU identity differs between config and environment.")
    for item in [result, *epochs]:
        physical_free = item.get("minimum_sampled_physical_free_bytes")
        require(isinstance(physical_free, int) and physical_free >= 2 * GIB,
                f"Missing or failed sampled physical headroom gate: {run_id}")
        require(item["headroom_bytes"] <= physical_free,
                f"Reported headroom exceeds sampled physical headroom: {run_id}")
    physical = [row for row in resources if "physical_memory_used_bytes" in row and "physical_memory_free_bytes" in row]
    require(bool(physical), f"No nvidia-smi physical memory samples: {run_id}")
    minimum_physical_free = min(row["physical_memory_free_bytes"] for row in physical)
    require(minimum_physical_free >= 2 * GIB, f"Physical sample falls below the 2 GiB gate: {run_id}")
    starts = [row for row in progress if row.get("event") == "paired_update_started"
              and row.get("epoch") == 1 and row.get("update") == 1]
    ends = [row for row in progress if row.get("event") == "epoch_completed" and row.get("epoch") == 2]
    require(len(starts) == len(ends) == 1, f"Training-window boundary events missing/duplicated: {run_id}")
    start, end = timestamp(starts[0]["timestamp"]), timestamp(ends[0]["timestamp"])
    require(end > start, "Training-window timestamps are not ordered.")
    host = [*resources, *(epoch["host"] for epoch in epochs), result["final_host"]]
    peak_rss = max(max(row.get("process_peak_rss_bytes", 0), row.get("rss", {}).get("VmHWM_bytes", 0),
                       row.get("rss", {}).get("VmRSS_bytes", 0)) for row in host)
    cgroup_samples = [row["cgroup"]["memory.current"] for row in host
                     if isinstance(row.get("cgroup", {}).get("memory.current"), int)]
    require(peak_rss > 0 and bool(cgroup_samples), "Host-memory measurements are absent.")
    cgroup_limits = {row["cgroup"]["memory.max"] for row in host
                     if isinstance(row.get("cgroup", {}).get("memory.max"), int)}
    require(len(cgroup_limits) == 1, "Cgroup memory limit is missing or changed during the trial.")
    utilization = []
    for sample in physical:
        fields = sample.get("nvidia_smi_sample", "").split(",")
        if len(fields) == 7:
            utilization.append(float(fields[1].strip()))
    epoch_seconds = [epoch["duration_seconds"] for epoch in epochs]
    row = {
        "role": role, "run_id": run_id, "physical_gpu": config["runtime"]["gpu"],
        "epoch_1_seconds": epoch_seconds[0], "epoch_2_seconds": epoch_seconds[1],
        "mean_epoch_seconds": mean(epoch_seconds),
        "mean_training_seconds": mean(epoch["training_duration_seconds"] for epoch in epochs),
        "mean_validation_seconds": mean(epoch["validation_duration_seconds"] for epoch in epochs),
        "eta_180_epochs_hours": mean(epoch_seconds) * 180 / 3600,
        "peak_allocated_gib": result["peak_allocated_bytes"] / GIB,
        "peak_reserved_gib": result["peak_reserved_bytes"] / GIB,
        "peak_sampled_physical_used_gib": max(item["physical_memory_used_bytes"] for item in physical) / GIB,
        "minimum_sampled_physical_free_gib": minimum_physical_free / GIB,
        "minimum_combined_cuda_and_physical_free_gib": result["minimum_sampled_physical_free_bytes"] / GIB,
        "conservative_headroom_gib": result["headroom_bytes"] / GIB,
        "peak_process_rss_gib": peak_rss / GIB,
        "maximum_sampled_shared_cgroup_current_gib": max(cgroup_samples) / GIB,
        "cgroup_memory_limit_gib": next(iter(cgroup_limits)) / GIB,
        "mean_sampled_gpu_utilization_percent_including_data_loading_and_initialization": mean(utilization) if utilization else None,
        "coverage_passed": True, "finite_gradients_passed": True, "resource_gate_passed": True,
        "runtime_window_seconds": end - start,
    }
    files = ("_SUCCESS", "manifest.yaml", "config.resolved.yaml", "provenance/artifact_checksums.json",
             "provenance/source_bundle.json", "provenance/source_files.json",
             "diagnostics/runtime_trial_result.json", "diagnostics/runtime_environment.json",
             "diagnostics/runtime_resources.jsonl", "metrics/runtime_epoch_metrics.jsonl",
             "metrics/runtime_progress.jsonl", "diagnostics/coverage_epoch_001.json", "diagnostics/coverage_epoch_002.json")
    provenance = {"run_id": run_id, "artifact_path": str(root),
                  "files": {name: sha256_file(root / name) for name in files},
                  "source_bundle": config["source_bundle"],
                  "source_configuration_canonical_sha256": canonical_sha256(config["source_configuration"]),
                  "dataset_fingerprints": result["source_data_fingerprints"],
                  "validation_mask_receipts_sha256": [item["validation_mask_receipts_sha256"] for item in epochs],
                  "coverage_file_sha256": [item["coverage_sha256"] for item in epochs]}
    return {"table": row, "source": provenance,
            "environment": {"cpu_threads": config["runtime"]["cpu_threads"],
                            **{key: environment.get(key) for key in ("gpu_name", "torch", "cuda_build", "cudnn")}},
            "window": {"first_training_update_start": starts[0]["timestamp"],
                       "second_epoch_complete": ends[0]["timestamp"], "start_unix_seconds": start, "end_unix_seconds": end},
            "source_code": read_json(root / "provenance/source_files.json"),
            "total_training_targets": sum(item["train_target_cells"] for item in epochs),
            "total_validation_targets": sum(item["validation_target_cells"] for item in epochs)}


def summarize(isolated: dict[str, Any], parallel: list[dict[str, Any]]) -> dict[str, Any]:
    all_runs = [isolated, *parallel]
    require(isolated["table"]["physical_gpu"] == "0", "This report requires the declared isolated GPU 0 baseline.")
    require({run["table"]["physical_gpu"] for run in parallel} == {"0", "1", "2", "3"},
            "Parallel runs must cover four distinct physical GPUs 0,1,2,3.")
    for key in ("source_configuration_canonical_sha256", "source_bundle", "dataset_fingerprints",
                "validation_mask_receipts_sha256", "coverage_file_sha256"):
        require(all(run["source"][key] == isolated["source"][key] for run in parallel), f"Run source/coverage mismatch: {key}")
    require(all(run["source_code"] == isolated["source_code"] for run in parallel), "Source code differs between isolated and parallel trials.")
    require(all(run["environment"] == isolated["environment"] for run in parallel), "Runtime environments differ between trials.")
    for run in all_runs:
        run["table"]["slowdown_vs_isolated"] = run["table"]["mean_epoch_seconds"] / isolated["table"]["mean_epoch_seconds"]
    first_start = min(run["window"]["start_unix_seconds"] for run in parallel)
    last_end = max(run["window"]["end_unix_seconds"] for run in parallel)
    common_overlap = min(run["window"]["end_unix_seconds"] for run in parallel) - max(run["window"]["start_unix_seconds"] for run in parallel)
    require(common_overlap > 0, "Four supposedly parallel training windows did not overlap.")
    span = last_end - first_start
    iso_rate = 2 * 3600 / isolated["table"]["runtime_window_seconds"]
    parallel_rate = 8 * 3600 / span
    max_cgroup = max(run["table"]["maximum_sampled_shared_cgroup_current_gib"] for run in parallel)
    return {
        "schema": "so2_hurdle_gpu_scaling_report_v1", "source_scientific_run_id": SOURCE_RUN,
        "scope": "Two-epoch runtime diagnostics; four same-seed replicas are not independent scientific seeds.",
        "isolated_run_id": isolated["table"]["run_id"],
        "parallel_run_ids": [run["table"]["run_id"] for run in parallel],
        "runs": [run["table"] for run in all_runs],
        "environment": isolated["environment"],
        "comparison": {
            "mean_parallel_epoch_seconds": mean(run["table"]["mean_epoch_seconds"] for run in parallel),
            "mean_parallel_slowdown_vs_isolated": mean(run["table"]["slowdown_vs_isolated"] for run in parallel),
            "parallel_training_and_validation_window_seconds": span,
            "four_way_training_window_overlap_seconds": common_overlap,
            "isolated_completed_epochs_per_hour": iso_rate,
            "parallel_aggregate_completed_epochs_per_hour": parallel_rate,
            "aggregate_epoch_throughput_speedup": parallel_rate / iso_rate,
            "parallel_aggregate_training_targets_per_second_including_validation_time": sum(run["total_training_targets"] for run in parallel) / span,
            "parallel_aggregate_all_target_visits_per_second": sum(run["total_training_targets"] + run["total_validation_targets"] for run in parallel) / span,
            "maximum_sampled_shared_cgroup_current_gib": max_cgroup,
            "prespecified_160_gib_cgroup_gate_passed": max_cgroup <= 160,
            "four_parallel_180_epoch_makespan_estimate_hours": max(run["table"]["eta_180_epochs_hours"] for run in parallel),
        },
        "windows": {run["table"]["run_id"]: run["window"] for run in all_runs},
        "sources": [run["source"] for run in all_runs],
        "limitations": [
            "ETA is 180 times the arithmetic mean of two train+validation epoch durations; excludes startup, final publication, checkpoints and unmeasured later-epoch variation or early stopping.",
            "The aggregate window runs from the earliest first training update to the latest second-epoch completion; it includes inter-job staggering and validation but excludes earlier data loading.",
            "GPU physical memory and utilization are sampled every five seconds, with additional CUDA free-memory measurements at phase boundaries; unsampled peaks can be missed.",
            "Mean sampled GPU utilization uses the entire resource-sampling period, including data loading and initialization; it is not a training-window-only utilization estimate.",
            "Only GPU 0 has an isolated baseline. Ratios for GPUs 1–3 conflate concurrency with possible device differences, so their full differences cannot be attributed to contention.",
            "Cgroup memory is a shared container measurement including caches and other processes; values across runs are never summed. Historical memory.peak is not used.",
            "These fresh, same-seed resource replicas have no checkpoints or prediction matrices and establish neither converged accuracy nor independent biological replication.",
            "One-rank and historical four-rank training preserve the pooled objective but change dropout draws; identical trajectories or checkpoint-resume equivalence are not claimed.",
            "No historical four-GPU timing is included in the controlled isolated-versus-concurrent comparison.",
        ],
    }


def markdown(summary: dict[str, Any]) -> str:
    comparison = summary["comparison"]
    allocation_claim = (
        "The tested allocation passed the measured two-epoch resource gates."
        if comparison["prespecified_160_gib_cgroup_gate_passed"]
        else "The tested allocation failed the prespecified aggregate host-memory gate."
    )
    lines = ["# SO2 hurdle model: isolated and concurrent GPU trials", "",
             f"All five registered two-epoch trials passed coverage, finite-gradient and GPU-memory checks. "
             f"Four concurrent jobs had an observed mean epoch-time ratio of {comparison['mean_parallel_slowdown_vs_isolated']:.2f}× relative to the isolated GPU 0 baseline; "
             f"aggregate measured epoch throughput was {comparison['aggregate_epoch_throughput_speedup']:.2f}× the isolated trial.", "",
             "| Trial | GPU | Epoch 1 / 2 (s) | Mean (s) | Ratio vs isolated GPU 0 | 180 epochs (h) | Allocated / reserved peak (GiB) | Sampled physical used peak (GiB) | Conservative free (GiB) | Peak RSS (GiB) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary["runs"]:
        lines.append(f"| {row['role']} | {row['physical_gpu']} | {row['epoch_1_seconds']:.1f} / {row['epoch_2_seconds']:.1f} | "
                     f"{row['mean_epoch_seconds']:.1f} | {row['slowdown_vs_isolated']:.2f}× | {row['eta_180_epochs_hours']:.2f} | "
                     f"{row['peak_allocated_gib']:.2f} / {row['peak_reserved_gib']:.2f} | {row['peak_sampled_physical_used_gib']:.2f} | "
                     f"{row['conservative_headroom_gib']:.2f} | {row['peak_process_rss_gib']:.2f} |")
    lines += ["", f"Concurrent shared cgroup memory reached a sampled maximum of **{comparison['maximum_sampled_shared_cgroup_current_gib']:.2f} GiB** "
              f"against the prespecified 160 GiB stop threshold (gate: {'passed' if comparison['prespecified_160_gib_cgroup_gate_passed'] else 'failed'}). "
              "This aggregate includes shared caches and other processes; per-run cgroup values are not additive.", "",
              f"The four-way training windows overlap for {comparison['four_way_training_window_overlap_seconds']:.1f} seconds. "
              f"Across the {comparison['parallel_training_and_validation_window_seconds']:.1f}-second combined window, "
              f"throughput was {comparison['parallel_aggregate_completed_epochs_per_hour']:.1f} completed epochs/hour "
              f"versus {comparison['isolated_completed_epochs_per_hour']:.1f} in isolation.", "",
              f"If later epoch durations remain similar, four 180-epoch jobs would finish in about "
              f"**{comparison['four_parallel_180_epoch_makespan_estimate_hours']:.2f} hours** from their training starts. "
              "This is an extrapolation from two epochs, not an observed full-training duration; startup, checkpoint writing and finalization add time.", "",
              allocation_claim + " The claim is limited to runtime feasibility. All four concurrent replicas use seed 0; they are not a multi-seed performance result.", "",
              "## Limitations", ""]
    lines += ["- " + item for item in summary["limitations"]]
    lines += ["", "## Immutable sources", "", "Every bundle was checksum-verified before reporting. Full source hashes, windows and measurements are in `summary.json` and `table.csv`.", ""]
    lines += [f"- {row['role']}, GPU {row['physical_gpu']}: `{row['run_id']}`" for row in summary["runs"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-run", required=True)
    parser.add_argument("--parallel-runs", nargs=4, required=True, metavar="RUN_ID")
    args = parser.parse_args(argv)
    ids = [args.isolated_run, *args.parallel_runs]
    require(len(set(ids)) == 5, "Expected five distinct immutable run IDs.")
    paths = current_paths()
    output = paths.report_root / "runtime/so2_hurdle_gpu_scaling/v1"
    require(not output.exists() and not output.is_symlink(), f"Report already exists: {output}")
    isolated = load_run(args.isolated_run, role="isolated", paths=paths)
    parallel = [load_run(run_id, role="parallel", paths=paths) for run_id in args.parallel_runs]
    parallel.sort(key=lambda run: run["table"]["physical_gpu"])
    summary = summarize(isolated, parallel)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".v1-building-{uuid.uuid4().hex}"
    staging.mkdir()
    (staging / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (staging / "report.md").write_text(markdown(summary))
    with (staging / "table.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["runs"][0]))
        writer.writeheader(); writer.writerows(summary["runs"])
    (staging / "report_builder.py").write_bytes(Path(__file__).read_bytes())
    manifest = {
        "schema": "so2_hurdle_gpu_scaling_report_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "working_directory": str(Path.cwd()), "source_run_ids": ids,
        "files": {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                  for path in sorted(staging.iterdir())},
        "completion": "All named run bundles verified; finalized report published without replacement.",
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _rename_no_replace(staging, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
