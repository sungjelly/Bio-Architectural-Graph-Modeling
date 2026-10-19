#!/usr/bin/env python3
"""Audit retained aggregate gradients after an optimization-audit stage ends.

Exploratory diagnostic declared after first-wave epoch-one gradients were
viewed. It tests finite/activity/clip evidence, not biological usefulness or
every internal parameter path. Require all ten completed grid settings or all
four completed confirmation settings. Read immutable artifacts only; never
run inference. First-update upstream zeros are expected from the zero output
head and are distinguished from later zeros. A detected zero is reported,
not suppressed. Stop on missing coverage, nonfinite values or source drift.

From the source root with BAGM_* overrides::

    PYTHONPATH=src:. /venv/main/bin/python scripts/analysis/report_so2_bce_gradient_audit.py \
      --stage grid --version grid_v1 --run-id <id> --run-id <remaining ids>

Produces a new checksum-bound JSON/CSV/Markdown receipt under
reports/runtime/so2_bce_optimization_audit/gradient_<version>. Existing output
fails closed. Reproduce checks with test_so2_bce_gradient_audit.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import yaml

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, _rename_no_replace, verify_run_bundle

CAMPAIGN = "cmp_20261001_so2_bce_optimization_audit"
INCUMBENT = ("target_window", 1024, 1e-4, .1, 1e-5)
NORM_FIELDS = tuple("gradient_norm_"+part+"_before_clip" for part in ("min", "mean", "max"))
GRADIENT_FILES = ("results/gradient_direction_metrics.csv", "results/gradient_direction_by_block.csv",
                  "results/gradient_direction_detection_head.csv")


def require(value, message):
    if not value:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text())


def finite(value):
    value = float(value)
    require(math.isfinite(value), "Nonfinite gradient evidence")
    return value


def check_csv_copy(payload, stored):
    """The dedicated CSV must agree with its copy in the epoch record."""
    require(set(payload) == set(stored), "Gradient CSV fields differ from epoch evidence")
    for key, value in payload.items():
        if value is None:
            require(stored[key] == "", "Nullable gradient CSV field differs")
        elif isinstance(value, bool):
            require(stored[key] == str(value), "Gradient boolean differs")
        elif isinstance(value, (int, float)):
            require(finite(stored[key]) == finite(value), "Gradient numeric CSV field differs")
        else:
            require(stored[key] == value, "Gradient identity differs")


def summarize_gradients(epochs, updates, *, run_id, seed, steps, clip_norm):
    require([int(row["global_epoch"]) for row in epochs] == list(range(1, len(epochs)+1)),
            "Gradient epoch coverage is incomplete")
    expected = [(epoch, step) for epoch in range(1, len(epochs)+1) for step in range(1, steps+1)]
    require([(row["epoch"], row["update"]) for row in updates] == expected,
            "Per-update norm coverage is missing, duplicated or out of order")
    require(all(row["run_id"] == run_id and row["cumulative_optimizer_update"] == i
                for i, row in enumerate(updates, 1)), "Per-update run/count identity differs")
    norms = [finite(row["gradient_norm"]) for row in updates]
    require(all(value >= 0 for value in norms) and finite(clip_norm) > 0, "Invalid preclip gradient norm")
    rows = []
    for epoch_record in epochs:
        epoch = int(epoch_record["global_epoch"])
        require(int(epoch_record["optimizer_updates"]) == steps, "Epoch update count differs")
        blocks = json.loads(epoch_record["block_gradients_json"])
        require([row["block_index"] for row in blocks] == [0, 1, 2, 3]
                and [row["block_name"] for row in blocks] == [f"blocks.{i}" for i in range(4)],
                "Four intended graph blocks are not present")
        payloads = [("global", json.loads(epoch_record["global_gradient_json"])),
                    *[(row["block_name"], row) for row in blocks],
                    ("detection_head", json.loads(epoch_record["detection_head_gradient_json"]))]
        for scope, value in payloads:
            require(value["run_id"] == run_id and value["model_seed"] == seed
                    and value["global_epoch"] == epoch and value["optimizer_updates_observed"] == steps
                    and value["resume_boundary_unavailable"] is False, "Gradient tracker coverage/identity differs")
            low, mean, high = [finite(value[key]) for key in NORM_FIELDS]
            require(0 <= low <= mean <= high and value["trainable_parameter_count"] > 0,
                    "Invalid aggregate gradient norm or parameter count")
            for key, item in value.items():
                if "cosine" in key and not key.endswith("valid_pairs") and item is not None:
                    require(-1.000001 <= finite(item) <= 1.000001, "Invalid gradient cosine")
            bootstrap = epoch == 1 and scope.startswith("blocks.")
            expected_pairs = steps - (2 if bootstrap else 1) if epoch == 1 else steps
            pairs = value["consecutive_optimizer_step_cosine_valid_pairs"]
            require(type(pairs) is int and 0 <= pairs <= (steps-1 if epoch == 1 else steps),
                    "Invalid adjacent-gradient pair count")
            rows.append({"run_id": run_id, "epoch": epoch, "scope": scope,
                "optimizer_updates": steps, "norm_min": low, "norm_mean": mean, "norm_max": high,
                "positive_mean_and_max": mean > 0 and high > 0,
                "expected_first_update_zero_context": bootstrap,
                "zero_minimum_beyond_expected_bootstrap": low == 0 and not bootstrap,
                "valid_adjacent_gradient_pairs": pairs, "expected_pairs_under_zero_head_bootstrap": expected_pairs,
                "adjacent_pair_count_matches_expected": pairs == expected_pairs})
        global_value = json.loads(epoch_record["global_gradient_json"])
        actual = norms[(epoch-1)*steps:epoch*steps]
        for retained, computed in ((global_value[NORM_FIELDS[0]], min(actual)),
                (global_value[NORM_FIELDS[1]], math.fsum(actual)/steps),
                (global_value[NORM_FIELDS[2]], max(actual))):
            require(math.isclose(float(retained), computed, rel_tol=1e-5, abs_tol=1e-7),
                    "Tracker and clip_grad_norm_ evidence disagree")
    clips = sum(value > clip_norm for value in norms)
    return {"total_updates": len(norms), "preclip_norm_min": min(norms), "preclip_norm_max": max(norms),
        "clip_threshold": clip_norm, "updates_above_clip_threshold": clips,
        "clipping_activation_fraction": clips/len(norms), "all_global_update_norms_positive": all(v > 0 for v in norms),
        "all_scope_epoch_means_positive": all(row["positive_mean_and_max"] for row in rows),
        "later_zero_minimum_detected": any(row["zero_minimum_beyond_expected_bootstrap"] for row in rows),
        "adjacent_pair_counts_match_bootstrap_expectation": all(row["adjacent_pair_count_matches_expected"] for row in rows),
        "scope_epochs": rows}


def settings(config):
    trainer = config["trainer"]
    return (trainer["optimizer_batch_mode"], trainer["target_cells_per_optimizer_update"],
            trainer["learning_rate"], config["model"]["dropout"], trainer["weight_decay"])


def validate_matrix(configs, stage):
    require(len(configs) == (10 if stage == "grid" else 4), "Entire terminal stage is required")
    profile, epochs = ("screen", 8) if stage == "grid" else ("confirmation", 32)
    require(all(c["metadata"]["execution_profile"] == profile and c["trainer"]["max_epochs"] == epochs
                and c["fold"] == 0 for c in configs), "Stage profile/budget/split differs")
    observed = {(settings(c), c["seed"]) for c in configs}
    if stage == "grid":
        expected = {(("target_window", b, lr, .1, 1e-5), 0)
                    for b in (512, 1024, 2048) for lr in (3e-5, 1e-4, 3e-4)}
        expected.add((("core_pair", None, 1e-4, .1, 1e-5), 0))
    else:
        variants = {settings(c) for c in configs}
        require(len(variants) == 2 and INCUMBENT in variants, "Confirmation variants differ")
        expected = {(v, s) for v in variants for s in (0, 1)}
    require(observed == expected, "Stage configuration matrix is incomplete or duplicated")


def load_run(run_id, paths, registry):
    root = RunArchive.artifact_path_for(run_id, paths)
    record = registry.get_run(run_id)
    require(record and record["status"] == "completed" and record["campaign_id"] == CAMPAIGN
            and Path(record["artifact_path"]).resolve() == root.resolve(), "Run is not finalized in its canonical registry location")
    require(verify_run_bundle(root)["status"] == "success", "Immutable bundle verification failed")
    config = yaml.safe_load((root/"config.resolved.yaml").read_text())
    epochs = read_csv(root/"results/epoch_metrics.csv")
    require(len(epochs) == config["trainer"]["max_epochs"], "Incomplete terminal epoch history")
    copies = [read_csv(root/name) for name in GRADIENT_FILES]
    require([len(rows) for rows in copies] == [len(epochs), 4*len(epochs), len(epochs)], "Dedicated gradient histories incomplete")
    for i, row in enumerate(epochs):
        check_csv_copy(json.loads(row["global_gradient_json"]), copies[0][i])
        for j, block in enumerate(json.loads(row["block_gradients_json"])):
            check_csv_copy(block, copies[1][4*i+j])
        check_csv_copy(json.loads(row["detection_head_gradient_json"]), copies[2][i])
    event = "paired_update_completed" if settings(config)[0] == "core_pair" else "target_update_completed"
    progress_path = root/"metrics/target_minibatch_progress.jsonl"
    updates = [row for line in progress_path.read_text().splitlines() if line.strip()
               and (row := json.loads(line))["event"] == event]
    audit = summarize_gradients(epochs, updates, run_id=run_id, seed=config["seed"],
        steps=config["trainer"]["optimizer_updates_per_global_epoch"], clip_norm=config["trainer"]["gradient_clip_norm"])
    names = ("_SUCCESS", "manifest.yaml", "config.resolved.yaml", "provenance/artifact_checksums.json",
             "results/epoch_metrics.csv", *GRADIENT_FILES, "metrics/target_minibatch_progress.jsonl")
    return config, {"run_id": run_id, "seed": config["seed"], "settings": list(settings(config)), **audit,
        "source_hashes": {name: sha256_file(root/name) for name in names}}


def stage_binding(plan_path, state_path, configs, runs, paths):
    require(plan_path.is_relative_to(paths.state_root) and state_path.is_relative_to(paths.state_root),
            "Operational stage sources must be under BAGM_STATE_ROOT")
    plan, state = read_json(plan_path), read_json(state_path)
    require(state["status"] == "completed" and state["ended_at"]
            and state["plan_sha256"] == canonical_sha256(plan)
            and state["stage_id"] == plan["stage_id"], "Operational stage is not complete or its plan changed")
    require(len(state["jobs"]) == len(plan["jobs"]) == len(runs)
            and all(job["status"] == "completed" and job["exit_code"] == 0 for job in state["jobs"])
            and {job["label"] for job in state["jobs"]} == {job["label"] for job in plan["jobs"]},
            "Operational stage job coverage differs")
    matched = []
    for job in plan["jobs"]:
        expected = (job["batch_mode"], job.get("batch_size"), job["learning_rate"], job["dropout"], job["weight_decay"])
        selected = [(c, r) for c, r in zip(configs, runs) if settings(c) == expected and c["seed"] == job["seed"]]
        require(len(selected) == 1 and selected[0][0]["metadata"]["execution_profile"] == job["profile"]
                and str(selected[0][0]["launcher"]["requested_gpu"]) == str(job["gpu"]),
                "Run does not uniquely match the authoritative stage job")
        matched.append({"label": job["label"], "run_id": selected[0][1]["run_id"]})
    return {"stage_id": plan["stage_id"], "jobs": matched,
        "sources": [{"root": "state", "path": p.relative_to(paths.state_root).as_posix(), "sha256": sha256_file(p)}
                    for p in (plan_path, state_path)]}


def build(stage, run_ids, version, paths, plan_path=None, state_path=None):
    require(len(run_ids) == len(set(run_ids)), "Repeated run ID")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version), "Unsafe receipt version")
    destination = paths.report_root/"runtime/so2_bce_optimization_audit"/("gradient_"+version)
    require(not destination.exists(), "Immutable gradient receipt already exists")
    registry = Registry(paths.state_root/"tracking/bagm.sqlite3", initialize=False)
    loaded = [load_run(run_id, paths, registry) for run_id in sorted(run_ids)]
    validate_matrix([item[0] for item in loaded], stage)
    runs = [item[1] for item in loaded]
    require(bool(plan_path) == bool(state_path), "Provide both operational plan and state")
    binding = stage_binding(plan_path.resolve(strict=True), state_path.resolve(strict=True),
        [item[0] for item in loaded], runs, paths) if plan_path else None
    count = sum(run["total_updates"] for run in runs)
    clips = sum(run["updates_above_clip_threshold"] for run in runs)
    summary = {"schema": "so2_bce_terminal_gradient_audit_v1", "campaign_id": CAMPAIGN, "stage": stage,
        "analysis_status": "exploratory_after_initial_gradient_inspection", "source_stage_complete": True,
        "operational_stage_binding": binding,
        "runs": runs, "total_updates": count, "updates_above_clip_threshold": clips,
        "clipping_activation_fraction": clips/count,
        "all_retained_gradient_evidence_finite": True,
        "all_scope_epoch_means_positive": all(run["all_scope_epoch_means_positive"] for run in runs),
        "later_zero_minimum_detected": any(run["later_zero_minimum_detected"] for run in runs),
        "adjacent_pair_counts_match_bootstrap_expectation": all(run["adjacent_pair_counts_match_bootstrap_expectation"] for run in runs),
        "limitations": ["Four graph-block parameter aggregates and the detection head are observed; internal encoder/geometry/feed-forward paths are not separately identified.",
            "Finite nonzero block gradients establish aggregate training activity, not useful spatial context, successful optimization, graph-specific gain or biology.",
            "The first upstream zero is expected from the zero-initialized output head. Retained epoch minima and valid-pair counts support this interpretation; individual block vectors are not exported.",
            "Clipping activation counts use the norm returned by clip_grad_norm_ before clipping, not the postclip parameter-gradient magnitude.",
            "Negative gradient cosines can arise from changing masks/core-local batches; they do not by themselves establish an optimization flaw."]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gradient-", dir=destination.parent) as temporary:
        root = Path(temporary)/"receipt"; root.mkdir()
        (root/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
        rows = [row for run in runs for row in run["scope_epochs"]]
        with (root/"scope_epochs.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        narrative = ["# Terminal aggregate gradient audit", "", f"Stage: {stage}; {len(runs)} complete runs and {count:,} optimizer updates.", "",
            f"Finite gradient evidence passed. All scope/epoch mean norms positive: {summary['all_scope_epoch_means_positive']}. "
            f"Zero minima beyond expected bootstrap: {summary['later_zero_minimum_detected']}. "
            f"Adjacent-pair counts match expected bootstrap: {summary['adjacent_pair_counts_match_bootstrap_expectation']}.", "",
            f"Preclip norms exceeded the configured clipping threshold in {clips:,}/{count:,} updates ({100*clips/count:.6f}%).", "",
            "| Run | Updates | Largest preclip norm | Clip activations | Later zero minimum |", "|---|---:|---:|---:|---|"]
        narrative += [f"| {run['run_id']} | {run['total_updates']} | {run['preclip_norm_max']:.6f} | {run['updates_above_clip_threshold']} | {run['later_zero_minimum_detected']} |" for run in runs]
        narrative += ["", *["- "+item for item in summary["limitations"]], "", "All source hashes and scope/epoch values are retained in summary.json and scope_epochs.csv. This is a post-hoc diagnostic and does not alter model selection."]
        (root/"README.md").write_text("\n".join(narrative)+"\n")
        shutil.copyfile(Path(__file__), root/"receipt_builder.py")
        if plan_path:
            shutil.copyfile(plan_path, root/"stage_plan.json")
            shutil.copyfile(state_path, root/"stage_state.json")
        manifest = {"schema": "checksummed_local_report_v1", "files": {p.name: sha256_file(p) for p in sorted(root.iterdir())}}
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        _rename_no_replace(root, destination)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("grid", "confirmation"), required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--stage-plan", type=Path)
    parser.add_argument("--stage-state", type=Path)
    args = parser.parse_args(argv)
    print(build(args.stage, args.run_id, args.version, current_paths(), args.stage_plan, args.stage_state))


if __name__ == "__main__":
    main()
