#!/usr/bin/env python3
"""Verify all four early-stopped local/self runs and optionally publish a result.

Run with PYTHONPATH=src:. and the shared BAGM_* roots. No output is written
until every full run and its immutable comparison sources pass verification.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import fcntl
import tempfile
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import uuid

import numpy as np
import yaml

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, _rename_no_replace, verify_run_bundle


CAMPAIGN = "cmp_20261005_local_gene_early_stopping"
RESULT_ID = "res_so2_local_gene_early_stopping"
ORIGINAL_RUNS = (
    "r_20261004T072107Z_9d2b00cb_s000_f00_a01_f41786d5",
    "r_20261004T072107Z_9d2b00cb_s001_f00_a01_1caff983",
    "r_20261004T072106Z_dd0aa417_s000_f00_a01_e0116c2b",
    "r_20261004T072107Z_dd0aa417_s001_f00_a01_ea627eca",
)
EXPECTED_PAIRS = {(variant, seed) for variant in ("local", "self") for seed in (0, 1)}
TRAIN_ALIASES = {f"SO2-C{core:02d}" for core in range(15, 27)}
VALIDATION_ALIASES = {"SO2-C27", "SO2-C28"}
MASKED_ENTRIES = 18_708_207
PRIMARY_NLL_THRESHOLD = 0.3262417527
RMSE_THRESHOLD = 1.1801696537
HISTORICAL_RUNS = {
    "BAGM NB": "r_20260926T104132Z_47d710fa_s000_f00_a01_b967dc7c",
    "BAGM hurdle": "r_20260928T021117Z_9ea46874_s000_f00_a01_41b54097",
}
GIB = 1024 ** 3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(first: float, second: float, *, tolerance: float = 1e-7) -> bool:
    return math.isfinite(first) and math.isfinite(second) and abs(first - second) <= tolerance


def _verified_root(run_id, paths, registry):
    root = RunArchive.artifact_path_for(run_id, paths)
    record = registry.get_run(run_id)
    require(bool(record) and record["status"] == "completed", f"Run is not completed: {run_id}")
    require(Path(record["artifact_path"]).resolve() == root.resolve(), f"Registry path differs: {run_id}")
    verify_run_bundle(root)
    require((root / "_SUCCESS").is_file(), f"Run lacks success marker: {run_id}")
    return root, record


def _hashes(root: Path, names) -> dict:
    return {name: sha256_file(root / name) for name in names}


def load_run(run_id, paths, registry) -> dict:
    root, record = _verified_root(run_id, paths, registry)
    config = yaml.safe_load((root / "config.resolved.yaml").read_text())
    summary = read_json(root / "summary.json")
    require(record["campaign_id"] == config["campaign"]["campaign_id"] == CAMPAIGN, "Unexpected campaign")
    require(config["metadata"]["profile"] == "full", "Scientific report excludes pilot runs")
    require(config["model"]["variant"] in {"local", "self"} and config["seed"] in {0, 1}, "Unexpected variant/seed")
    require(config["model"]["family"] == "local_gene_residual", "Unexpected model family")
    require(config["model"]["use_local"] == (config["model"]["variant"] == "local"), "Local ablation flag differs")
    for name, expected in {"hidden_dim": 512, "hidden_layers": 2, "dropout": .1,
            "output_distribution": "bernoulli_hurdle_zero_truncated_nb2"}.items():
        require(config["model"].get(name) == expected, f"Prespecified model field changed: {name}")
    require(config["evaluation"]["primary_metric"] == summary["primary_metric_name"] == "val/count_nll", "Primary metric changed")
    require(summary["run_id"] == run_id and summary["status"] == "completed", "Summary identity/status differs")
    require(summary["variant"] == config["model"]["variant"] and summary["seed"] == config["seed"], "Summary variant/seed differs")
    require(config["trainer"]["epochs"] == 500 and 50 <= summary["final_epoch"] <= 500, "Full early-stopping budget differs")
    require(config["trainer"]["batch_size"] == 1024, "Prespecified target batch changed")
    for name, expected in {"learning_rate": .001, "minimum_learning_rate": .00001,
            "weight_decay": .00001, "gradient_clip": 1., "ordering_seed": 2026100401,
            "scheduler": "reduce_on_plateau", "scheduler_factor": .5, "scheduler_patience": 7,
            "scheduler_threshold": .0001, "scheduler_threshold_mode": "abs", "precision": "bfloat16_autocast_fp32_likelihood"}.items():
        require(config["trainer"].get(name) == expected, f"Prespecified trainer field changed: {name}")
    require(config["trainer"]["early_stopping"] == {"patience": 30, "min_delta": .0001, "min_epochs": 50, "max_epochs": 500}, "Stopping policy changed")
    require(config["trainer"]["minimum_epochs"] == 50 and config["trainer"]["maximum_epochs"] == 500, "Minimum/maximum epoch guard changed")
    require(config.get("masking") == {"kind": "target_isolated_uniform_1_1000",
        "training_base_seed": 2026092601,
        "validation_namespace": "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"}, "Inherited masking configuration changed")
    require(summary["checkpoint_reload_verified"] is True and summary["terminal_checkpoint_reload_verified"] is True, "Selected or terminal checkpoint replay absent")
    require(summary["parameter_count"] == 2_329_432, "Architecture parameter count changed")
    final = {key.removeprefix("val/"): value for key, value in read_json(root / "metrics/final.json").items()}
    require(final["masked_entries"] == MASKED_ENTRIES, "Validation masked-entry support changed")
    require(close(final["count_nll"], summary["primary_metric_value"]), "Summary primary value differs")
    history = [json.loads(line) for line in (root / "metrics/epoch_history.jsonl").read_text().splitlines() if line.strip()]
    require([row["epoch"] for row in history] == list(range(1, summary["final_epoch"] + 1)), "Epoch history incomplete or duplicated")
    require(all(row["target_visits"] == 208_696 for row in history), "Incomplete training target exposure")
    require(all(row["val/masked_entries"] == MASKED_ENTRIES for row in history), "Validation support differs across epochs")
    verify_optimizer_history(history, summary)
    selected = min(history, key=lambda row: row["val/count_nll"])
    require(selected["epoch"] == summary["best_epoch"], "Checkpoint is not strict earliest validation NLL minimum")
    for key in ("count_nll", "raw_count_mae", "raw_count_rmse"):
        require(close(final[key], selected["val/" + key]), f"Selected metric replay differs: {key}")
    recent = read_json(root / "metrics/recent_epochs.json")
    require(recent["epochs"] == summary["recent_epoch_ids"] == [row["epoch"] for row in history[-4:]], "Recent epoch IDs differ")
    require(recent["aggregation"] == "unweighted_epoch_mean_last_up_to_four_complete_epochs", "Recent aggregation changed")
    require(set(recent["metrics"]) == set(final), "Recent-epoch metric schema differs")
    for key, value in recent["metrics"].items():
        require(close(value, float(np.mean([row["val/" + key] for row in history[-4:]]))), f"Recent-epoch metric differs: {key}")
    terminal = float(np.mean([row["val/count_nll"] for row in history[-4:]]))
    require(close(terminal, summary["terminal_mean_count_nll"]), "Recent four-epoch summary differs")
    stopping = verify_stopping_history(history, summary, read_json(root / "diagnostics/stopping.json"))
    terminal_metrics = {key.removeprefix("val/"): value for key, value in read_json(root / "metrics/terminal.json").items()}
    require(set(terminal_metrics) == set(final), "Final-epoch metric schema differs")
    for key, value in terminal_metrics.items():
        require(close(value, history[-1]["val/" + key], tolerance=1e-5), f"Final-epoch metric differs: {key}")
    require(summary["selected_metrics"] == final and summary["terminal_metrics"] == terminal_metrics, "Selected/final summary metrics differ")
    data = read_json(root / "provenance/data_fingerprints.json")
    require(data["overlay_manifest_content_sha256"] == config["dataset"]["dataset_fingerprint"], "Dataset fingerprint differs")
    require(data["split_fingerprint"] == config["dataset"]["split_fingerprint"], "Split fingerprint differs")
    require(data["preprocessing_fingerprint"] == config["dataset"]["preprocessing_fingerprint"], "Preprocessing fingerprint differs")
    require(data["source_core_checksums_verified"] is True and data["local_self_exclusion"] is True, "Data correctness receipt failed")
    require(data["training_cells"] == 208_696 and data["validation_cells"] == 37_367, "Data cell coverage differs")
    masks = read_json(root / "diagnostics/validation_masks.json")
    require(set(masks) == VALIDATION_ALIASES, "Validation aliases changed")
    require(sum(row["masked_entry_count"] for row in masks.values()) == MASKED_ENTRIES, "Mask receipts differ from metric support")
    require(all(row["covers_all_cells"] and row["fixed_across_epochs"] for row in masks.values()), "Fixed validation masks incomplete")
    coverage = read_json(root / "diagnostics/coverage.json")
    require([row["epoch"] for row in coverage] == list(range(1, summary["final_epoch"] + 1)), "Training coverage receipts incomplete")
    for row in coverage:
        require(set(row["cores"]) == TRAIN_ALIASES, "Training core coverage differs")
        require(sum(core["target_visits"] for core in row["cores"].values()) == 208_696, "Training visits incomplete")
        require(all(core["masked_entries"] > 0 and len(core["mask_sha256"]) == 64 and len(core["order_sha256"]) == 64 for core in row["cores"].values()), "Training mask receipt invalid")
    coverage_log = [json.loads(line) for line in (root / "diagnostics/coverage.jsonl").read_text().splitlines() if line.strip()]
    require(coverage_log == coverage, "Append-only coverage history differs")
    replay = read_json(root / "diagnostics/checkpoint_replay.json")
    require(replay["passed"] is True and replay["maximum_absolute_delta"] <= 1e-5, "Selected-checkpoint replay failed")
    require(replay["best_epoch"] == summary["best_epoch"], "Replay checkpoint epoch differs")
    require(replay["checkpoint_sha256"] == sha256_file(root / "checkpoints/best.ckpt"), "Replay checkpoint checksum differs")
    terminal_replay = read_json(root / "diagnostics/terminal_checkpoint_replay.json")
    require(terminal_replay["passed"] is True and terminal_replay["maximum_absolute_delta"] <= 1e-5, "Terminal-checkpoint replay failed")
    require(terminal_replay["final_epoch"] == summary["final_epoch"], "Terminal checkpoint epoch differs")
    require(terminal_replay["checkpoint_sha256"] == sha256_file(root / "checkpoints/last.ckpt"), "Terminal checkpoint checksum differs")
    indexed = registry.list_checkpoint_catalog(run_id=run_id, verification_status="verified", limit=None)
    for role, filename in (("best", "best.ckpt"), ("last", "last.ckpt")):
        require(any(Path(item["path"]).name == filename and item["sha256"] == sha256_file(root / "checkpoints" / filename) for item in indexed), f"{role} checkpoint is not checksum verified in catalog")
    preflight = read_json(root / "diagnostics/preflight.json")
    source_hashes = read_json(root / "provenance/source_hashes.json")
    require(preflight["passed"] is True and preflight["source_hashes"] == source_hashes, "Source or correctness receipt differs")
    require(preflight["frozen_contract_sha256"] == config["metadata"]["frozen_contract_sha256"], "Frozen contract fingerprint differs")
    precision = {role: read_json(root / f"diagnostics/precision_{role}.json") for role in ("initial", "selected")}
    require(all(row["passed"] is True and row["absolute_delta"] < row["tolerance"] == .001 for row in precision.values()), "FP32/BF16 likelihood check failed")
    per_core = read_json(root / "results/per_core.json")
    require(set(per_core) == VALIDATION_ALIASES, "Per-core metrics incomplete")
    require(sum(row["masked_entries"] for row in per_core.values()) == MASKED_ENTRIES, "Per-core metric supports differ")
    strata = read_json(root / "results/mask_strata.json")
    require(set(strata) == {"1_250", "251_500", "501_750", "751_1000"}, "Mask severity strata incomplete")
    require(sum(row["masked_entries"] for row in strata.values()) == MASKED_ENTRIES, "Mask severity support differs")
    resources = read_json(root / "diagnostics/resources.json")
    require(bool(resources), "No resource measurements")
    for metric in ("count_nll", "raw_count_mae", "raw_count_rmse"):
        weighted_core = sum(row["masked_entries"] * (row[metric] ** 2 if metric.endswith("rmse") else row[metric]) for row in per_core.values()) / MASKED_ENTRIES
        require(close(final[metric], math.sqrt(weighted_core) if metric.endswith("rmse") else weighted_core, tolerance=1e-6), "Pooled metrics disagree with per-core metrics")
    return {
        "run_id": run_id, "variant": config["model"]["variant"], "seed": config["seed"],
        "best_epoch": summary["best_epoch"], "final_epoch": summary["final_epoch"], "metrics": final,
        "terminal_metrics": terminal_metrics, "stopping": stopping,
        "recent_epochs": read_json(root / "metrics/recent_epochs.json"), "training_coverage": coverage,
        "terminal_mean_count_nll": terminal, "history": history, "per_core": per_core,
        "mask_strata": strata, "baseline": read_json(root / "diagnostics/train_fitted_constant_baseline.json"),
        "parameter_count": summary["parameter_count"], "duration_seconds": summary["duration_seconds"],
        "mean_epoch_seconds": float(np.mean([row["seconds"] for row in history])),
        "optimization_steps": summary["optimization_steps"], "peak_vram_gib": summary["peak_vram_gb"],
        "peak_rss_gib": max(row["rss_peak_bytes"] for row in resources) / GIB,
        "physical_gpu": config["launcher"]["requested_gpu"],
        "precision_checks": precision,
        "source_hashes": source_hashes, "data_provenance": data, "validation_masks": masks,
        "training_coverage_sha256": canonical_sha256(coverage), "config": config,
        "artifact_hashes": _hashes(root, ("_SUCCESS", "config.resolved.yaml", "summary.json", "metrics/final.json",
            "metrics/epoch_history.jsonl", "provenance/artifact_checksums.json", "diagnostics/checkpoint_replay.json",
            "diagnostics/train_fitted_constant_baseline.json", "diagnostics/coverage.json",
            "diagnostics/validation_masks.json", "diagnostics/precision_initial.json", "diagnostics/precision_selected.json",
            "results/per_core.json", "results/mask_strata.json", "metrics/terminal.json",
            "metrics/recent_epochs.json", "diagnostics/stopping.json", "diagnostics/terminal_checkpoint_replay.json",
            "diagnostics/coverage.jsonl", "provenance/source_hashes.json", "provenance/data_fingerprints.json")),
    }


def verify_comparison(rows: list[dict]) -> None:
    require(len(rows) == 4 and len({row["run_id"] for row in rows}) == 4, "Exactly four distinct runs are required")
    require({(row["variant"], row["seed"]) for row in rows} == EXPECTED_PAIRS, "Expected local/self crossed with seeds 0/1")
    for row in rows:
        expected_graph = {"kind": "fixed_local_summary" if row["variant"] == "local" else "none",
            "neighbor_k": 16 if row["variant"] == "local" else 0, "radius_um": 75.0, "self_loops": False}
        require(row["config"]["graph"] == expected_graph, "Variant-specific local graph contract changed")
    for row in rows:
        require(row["optimization_steps"] == 209 * row["final_epoch"], "Optimizer count differs from exhaustive target windows")
    first = rows[0]
    for row in rows[1:]:
        for key in ("source_hashes", "data_provenance", "validation_masks"):
            require(row[key] == first[key], f"Runs differ in paired comparison field: {key}")
        for key in ("dataset", "masking", "trainer", "evaluation", "metadata"):
            require(row["config"][key] == first["config"][key], f"Runs differ in scientific configuration: {key}")
        for key in ("family", "hidden_dim", "hidden_layers", "dropout", "output_distribution"):
            require(row["config"]["model"][key] == first["config"]["model"][key], f"Model ablation changed unrelated field: {key}")
        common = min(row["final_epoch"], first["final_epoch"])
        require(row["training_coverage"][:common] == first["training_coverage"][:common], "Paired training coverage/masks/order differ within common prefix")
        for key in ("count_nll", "raw_count_mae", "raw_count_rmse"):
            require(close(row["baseline"]["pooled"][key], first["baseline"]["pooled"][key]), "Train-fitted constant baseline differs")


def load_historical(paths, registry, reference=None) -> list[dict]:
    result = []
    for label, run_id in HISTORICAL_RUNS.items():
        root, _ = _verified_root(run_id, paths, registry)
        final = read_json(root / "metrics/final.json")
        summary = read_json(root / "summary.json")
        if reference is not None:
            config = yaml.safe_load((root / "config.resolved.yaml").read_text())
            for key in ("dataset_fingerprint", "split_fingerprint", "preprocessing_fingerprint"):
                require(config["dataset"][key] == reference["config"]["dataset"][key], f"Historical {key} differs")
            require(final["val/masked_entries"] == MASKED_ENTRIES, "Historical validation support differs")
        nll_key = "val/target_only_masked_" + ("hurdle_" if label.endswith("hurdle") else "") + "negative_binomial_nll"
        result.append({
            "label": label, "run_id": run_id, "seed": 0,
            "count_nll": final[nll_key], "raw_count_mae": final["val/unseen_donor/raw_count_mae"],
            "raw_count_rmse": final["val/unseen_donor/raw_count_rmse"],
            "best_epoch": summary["best_epoch"], "final_epoch": summary["final_epoch"],
            "parameter_count": summary["parameter_count"], "duration_seconds": summary["duration_seconds"],
            "artifact_hashes": _hashes(root, ("_SUCCESS", "metrics/final.json", "summary.json", "config.resolved.yaml", "provenance/artifact_checksums.json")),
        })
    return result


def assess(rows: list[dict], historical: list[dict]) -> dict:
    """Evaluate prespecified endpoint and paired-local criteria separately."""
    require({(row["variant"], row["seed"]) for row in rows} == EXPECTED_PAIRS, "Missing prespecified variant/seed")
    hurdle = next(row for row in historical if row["label"] == "BAGM hurdle")
    local = sorted((row for row in rows if row["variant"] == "local"), key=lambda row: row["seed"])
    means = {variant: float(np.mean([row["metrics"]["count_nll"] for row in rows if row["variant"] == variant])) for variant in ("local", "self")}
    nll_threshold = hurdle["count_nll"] - .002
    rmse_threshold = hurdle["raw_count_rmse"] * 1.01
    margin = means["local"] <= nll_threshold
    each_nll = all(row["metrics"]["count_nll"] < hurdle["count_nll"] for row in local)
    each_rmse = all(row["metrics"]["raw_count_rmse"] <= rmse_threshold for row in local)
    paired = []
    for row in local:
        comparator = next(other for other in rows if other["variant"] == "self" and other["seed"] == row["seed"])
        paired.append({"seed": row["seed"], "local_minus_self_count_nll": row["metrics"]["count_nll"] - comparator["metrics"]["count_nll"],
            "local_minus_self_raw_count_rmse": row["metrics"]["raw_count_rmse"] - comparator["metrics"]["raw_count_rmse"],
            "local_minus_self_raw_count_mae": row["metrics"]["raw_count_mae"] - comparator["metrics"]["raw_count_mae"],
            "per_core_count_nll_differences": {alias: row["per_core"][alias]["count_nll"] - comparator["per_core"][alias]["count_nll"] for alias in sorted(VALIDATION_ALIASES)}})
    return {
        "local_mean_selected_count_nll": means["local"], "self_mean_selected_count_nll": means["self"],
        "variant_mean_point_metrics": {variant: {metric: float(np.mean([row["metrics"][metric] for row in rows if row["variant"] == variant]))
            for metric in ("raw_count_mae", "raw_count_rmse")} for variant in ("local", "self")},
        "local_mean_minus_historical_hurdle_nll": means["local"] - hurdle["count_nll"],
        "nll_margin_threshold": nll_threshold, "rmse_upper_threshold": rmse_threshold,
        "local_mean_nll_margin_passed": margin, "both_local_seeds_improve_historical_nll": each_nll,
        "both_local_seeds_rmse_guard_passed": each_rmse,
        "historical_endpoint_criterion_passed": margin and each_nll and each_rmse,
        "both_seeds_local_beats_self_nll": all(row["local_minus_self_count_nll"] < 0 for row in paired),
        "paired_local_self_differences": paired,
        "statistical_significance_claimed": False,
    }


def load_failures(registry) -> list[dict]:
    with registry.connect() as connection:
        records = connection.execute("SELECT run_id, artifact_path, failure_category FROM runs WHERE campaign_id = ? AND status = 'failed' ORDER BY run_id", (CAMPAIGN,)).fetchall()
    failures = []
    for record in records:
        root = Path(record["artifact_path"])
        verify_run_bundle(root)
        require((root / "_FAILED").is_file(), "Failed attempt lacks a failure marker")
        config = yaml.safe_load((root / "config.resolved.yaml").read_text())
        failures.append({"run_id": record["run_id"], "profile": config["metadata"]["profile"],
            "variant": config["model"]["variant"], "seed": config["seed"],
            "failure_category": record["failure_category"],
            "artifact_hashes": _hashes(root, ("_FAILED", "summary.json", "provenance/artifact_checksums.json"))})
    return failures


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def learning_curve(rows, historical, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    colors = {"local": "#007F86", "self": "#A95721"}
    for row in rows:
        for axis, metric in zip(axes, ("count_nll", "raw_count_rmse")):
            axis.plot([value["epoch"] for value in row["history"]],
                [value["val/" + metric] for value in row["history"]], color=colors[row["variant"]],
                linestyle="-" if row["seed"] == 0 else "--", linewidth=1.8,
                label=f"{row['variant']} / seed {row['seed']}")
            axis.scatter(row["best_epoch"], row["metrics"][metric], s=24, color=colors[row["variant"]], zorder=4)
    hurdle = next(row for row in historical if row["label"] == "BAGM hurdle")
    for axis, metric, label in zip(axes, ("count_nll", "raw_count_rmse"), ("Count NLL ↓", "Raw-count RMSE ↓")):
        axis.axhline(hurdle[metric], color="#646464", linestyle=":", linewidth=1.3, label="Historical hurdle endpoint")
        axis.set(xlabel="Completed training epoch", ylabel=label)
        axis.grid(alpha=.16)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Fixed validation masks • one held-out donor • points select lowest NLL", fontsize=11)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def verify_optimizer_history(history, summary):
    """Check complete update counts and the recorded plateau LR transitions."""
    lr, scheduler_best, scheduler_bad = .001, math.inf, 0
    for row in history:
        require(row["steps"] == 209 and row["cumulative_steps"] == row["epoch"] * 209, "Incomplete optimizer-step history")
        require(close(row["learning_rate"], lr, tolerance=1e-12), "Learning-rate history before scheduling differs")
        value = row["val/count_nll"]
        if value < scheduler_best - .0001:
            scheduler_best, scheduler_bad = value, 0
        else:
            scheduler_bad += 1
        if scheduler_bad > 7:
            lr, scheduler_bad = max(lr * .5, .00001), 0
        require(close(row["learning_rate_after_scheduler"], lr, tolerance=1e-12), "Learning-rate history after scheduling differs")
    require(summary["optimization_steps"] == len(history) * 209, "Total optimizer steps differ")
    require(close(summary["last_learning_rate"], lr, tolerance=1e-12), "Terminal learning rate differs")


def verify_stopping_history(history, summary, receipt):
    """Reconstruct significant-improvement accounting independently of the trainer."""
    best, best_epoch, anchor, anchor_epoch, bad = math.inf, None, math.inf, None, 0
    last_expected = None
    for row in history:
        epoch, value = row["epoch"], float(row["val/count_nll"])
        require(math.isfinite(value), "Nonfinite stopping metric")
        strict = value < best
        if strict:
            best, best_epoch = value, epoch
        significant = value <= anchor - .0001
        if significant:
            anchor, anchor_epoch, bad = value, epoch, 0
        else:
            bad += 1
        reason = "patience" if epoch >= 50 and bad >= 30 else ("maximum_epochs" if epoch >= 500 else None)
        require(not reason or epoch == history[-1]["epoch"], "Training continued after its first stopping decision")
        decision = row["stopping_decision"]
        require(decision["stop_reason"] == reason and decision["should_stop"] == (reason is not None), "Recorded stopping decision differs from independent replay")
        last_expected = {"stop_reason": reason, "early_stopping_triggered": reason == "patience",
            "safety_cap_reached": epoch >= 500, "best_epoch": best_epoch,
            "best_value": best, "anchor_epoch": anchor_epoch, "anchor_value": anchor,
            "epochs_without_improvement": bad}
    require(last_expected is not None and last_expected["stop_reason"] is not None, "Full run has no actual terminal stopping decision")
    for key in ("stop_reason", "early_stopping_triggered", "safety_cap_reached"):
        require(summary[key] == receipt[key] == last_expected[key], f"Stopping receipt or summary differs: {key}")
    require(summary["best_epoch"] == best_epoch and close(summary["primary_metric_value"], best), "Strict best differs from stopping replay")
    require(receipt["policy_state"] == summary["stopping_state"] == history[-1]["stopping_state"], "Terminal stopping state is not preserved")
    # The policy state/decision complete schema is checked by replaying the public
    # serializable policy as well; the independent arithmetic above checks its rule.
    from spatial_benchmark.local_gene_stopping import LocalGeneEarlyStopping
    policy = LocalGeneEarlyStopping(min_epochs=50, max_epochs=500, patience=30, min_delta=.0001)
    from dataclasses import asdict
    for row in history:
        decision = policy.observe(epoch=row["epoch"], count_nll=row["val/count_nll"])
        require(asdict(decision) == row["stopping_decision"], "Stopping decision schema/state differs")
        require(policy.state_dict() == row["stopping_state"], "Stopping state history differs")
    return {**last_expected, "policy_state": receipt["policy_state"],
        "global_convergence_established": False,
        "interpretation": "Validation patience criterion reached; not global convergence." if last_expected["early_stopping_triggered"] else "Operational maximum_epochs reached; early stopping did not occur and convergence is unestablished."}


def load_original(paths, registry, rows):
    from scripts.analysis import report_local_gene_residual as original
    previous = [original.load_run(run_id, paths, registry) for run_id in ORIGINAL_RUNS]
    original.verify_comparison(previous)
    for row in rows:
        old = next(item for item in previous if (item["variant"], item["seed"]) == (row["variant"], row["seed"]))
        for key in ("data_provenance", "validation_masks"):
            require(row[key] == old[key], f"Original 32-epoch comparison changed {key}")
        for key in ("dataset", "masking", "model", "graph"):
            require(row["config"][key] == old["config"][key], f"Original comparison changed {key}")
        # Both fresh protocols consume the same masks and target ordering, despite
        # their different learning-rate schedules and final training durations.
        old_root = RunArchive.artifact_path_for(old["run_id"], paths)
        coverage = read_json(old_root / "diagnostics/coverage.json")
        require(row["training_coverage"][:32] == coverage, "Original first-32-epoch coverage/masks/order differ")
    return previous


def compare_original(rows, previous):
    comparisons = []
    for row in rows:
        old = next(item for item in previous if (item["variant"], item["seed"]) == (row["variant"], row["seed"]))
        comparisons.append({"variant": row["variant"], "seed": row["seed"],
            "new_run_id": row["run_id"], "original_run_id": old["run_id"],
            **{f"new_minus_original_{metric}": row["metrics"][metric] - old["metrics"][metric]
               for metric in ("count_nll", "raw_count_mae", "raw_count_rmse")}})
    return comparisons


def render_report(report):
    rows, assessment = report["runs"], report["assessment"]
    capped = [row for row in rows if not row["stopping"]["early_stopping_triggered"]]
    lines = ["# Local Gene Residual validation early stopping", "",
        f"All four full runs completed their declared policy. The historical .002 mean NLL improvement criterion **{'passed' if assessment['historical_endpoint_criterion_passed'] else 'did not pass'}**; local context **{'improved' if assessment['both_seeds_local_beats_self_nll'] else 'did not improve'}** selected NLL in both paired seeds.", "",
        f"Actual validation patience stops: **{4-len(capped)}/4**. Operational safety-cap stops: **{len(capped)}/4**. A `maximum_epochs` stop is not convergence. Reaching validation patience is not guaranteed global convergence either.", "",
        "This prospective follow-up reuses one held-out validation donor and has no sealed test set. Seeds measure optimization variation, not independent patient replication. These are fresh runs with a plateau scheduler, not uninterrupted resumptions of the original cosine schedule; differences combine training duration and schedule.", "",
        "![All-seed validation count NLL and RMSE; points mark strict selected checkpoints](learning_curves.png)", "",
        "| Variant | Seed | Best / final epoch | Stop reason | Selected NLL | Final NLL | Last-four mean NLL | Selected MAE | Selected RMSE |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['variant']} | {row['seed']} | {row['best_epoch']} / {row['final_epoch']} | {row['stopping']['stop_reason']} | {m['count_nll']:.6f} | {row['terminal_metrics']['count_nll']:.6f} | {row['terminal_mean_count_nll']:.6f} | {m['raw_count_mae']:.6f} | {m['raw_count_rmse']:.6f} |")
    lines += ["", "The strict lowest-NLL checkpoint is selected independently from the significant-improvement anchor. Patience counts 30 validations without an accumulated ≥0.0001 improvement, stopping no earlier than epoch 50; epoch 500 is a safety ceiling. Different stop epochs mean unequal final target exposure. Complete paired masks/order are verified through each common prefix, and all runs target every training cell once per epoch.", "",
        "| Historical reference | Best / final epoch | NLL | Raw MAE | Raw RMSE |",
        "|---|---:|---:|---:|---:|"]
    for row in report["historical"]:
        lines.append(f"| {row['label']} | {row['best_epoch']} / {row['final_epoch']} | {row['count_nll']:.6f} | {row['raw_count_mae']:.6f} | {row['raw_count_rmse']:.6f} |")
    for row in report["original_32_epoch_runs"]:
        m = row["metrics"]
        lines.append(f"| Original {row['variant']}, seed {row['seed']} | {row['best_epoch']} / 32 | {m['count_nll']:.6f} | {m['raw_count_mae']:.6f} | {m['raw_count_rmse']:.6f} |")
    lines += ["", "| Variant | Seed | New − original NLL | New − original MAE | New − original RMSE |",
        "|---|---:|---:|---:|---:|"]
    for row in report["original_comparison"]:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['new_minus_original_count_nll']:+.6f} | {row['new_minus_original_raw_count_mae']:+.6f} | {row['new_minus_original_raw_count_rmse']:+.6f} |")
    lines += ["", "The original 32-epoch experiment's criterion and outcome remain unchanged. Historical BAGM uses a different architecture and optimizer regimen; historical endpoint differences do not isolate an architecture effect. Local/self controls use the same stopping rule and optimizer configuration, but their realized LR trajectories and exposure may differ.", "",
        f"Mean local selected NLL: **{assessment['local_mean_selected_count_nll']:.6f}**. Historical-margin threshold: **{PRIMARY_NLL_THRESHOLD:.10f}**. Both local seeds improve historical NLL: **{assessment['both_local_seeds_improve_historical_nll']}**; both satisfy the ≤1% RMSE guard: **{assessment['both_local_seeds_rmse_guard_passed']}**.", "",
        "| Variant | Seed | Positive marginal MAE | Conditional-positive MAE | Detection BCE | Detection AP |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["metrics"]
        lines.append(f"| {row['variant']} | {row['seed']} | {m['positive_marginal_raw_count_mae']:.6f} | {m['positive_conditional_raw_count_mae']:.6f} | {m['detection_bce']:.6f} | {m['detection_average_precision']:.6f} |")
    lines += ["", "Positive marginal metrics score the complete predictor on observed-positive entries; conditional-positive metrics score the separate count component conditional on positivity. They must not be interchanged. Zero and log-scale errors, all detection metrics, and per-core/mask-severity strata are retained in the tables. AP/AUROC use 4,096-bin approximations. No threshold was tuned.", "",
        f"Always-zero raw MAE: **{report['zero_predictor_mae']:.6f}**. Train-fitted hurdle constants use training prevalence/positive means with theta fixed at 1; see `summary.json`. Low aggregate MAE alone is insufficient evidence of useful count reconstruction.", "",
        "Sources, frozen configuration, stopping histories, final and selected checkpoint checksums/replays/catalog entries, fixed masks, exhaustive coverage, and per-core metric decompositions were verified before writing this report. Initial/selected FP32-versus-BF16 minibatch likelihood checks do not establish whole-cohort precision equivalence.", "",
        "No sealed test, additional held-out donors, segmentation perturbations, realistic spatial nulls, or causal evidence are provided. Local gain here is predictive on this validation split. Neither a safety cap nor patience proves a global optimum.", "",
        "Files: [selected metrics](comparison.csv), [final metrics](terminal_metrics.csv), [original comparison](original_comparison.csv), [per-core](per_core.csv), [mask severity](mask_strata.csv), [all histories](learning_curves.csv), [stopping states](stopping.json), [summary](summary.json), [checksums](manifest.json).", "",
        "## Preserved failed attempts", ""]
    lines += ([f"- `{row['run_id']}`: {row['failure_category']} ({row['profile']}); excluded from completed-run means." for row in report["failed_attempts"]] or ["No failed attempts were registered at report time."])
    lines += ["", "## Immutable run sources", ""]
    lines += [f"- `{row['run_id']}`" for group in (rows, report["original_32_epoch_runs"], report["historical"]) for row in group]
    return "\n".join(lines) + "\n"


def result_record(report, sources):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    a = report["assessment"]
    capped = any(not row["stopping"]["early_stopping_triggered"] for row in report["runs"])
    outcome = "supported" if a["historical_endpoint_criterion_passed"] else ("inconclusive" if capped else "negative")
    observed = (f"Mean selected local count NLL {a['local_mean_selected_count_nll']:.9f}; mean-minus-historical hurdle NLL {a['local_mean_minus_historical_hurdle_nll']:+.9f}. "
        f"The predeclared .002 improvement plus both-seed NLL/RMSE guards {'passed' if a['historical_endpoint_criterion_passed'] else 'did not pass'}. "
        f"Local beats self-only NLL in both seeds: {a['both_seeds_local_beats_self_nll']}. "
        + "; ".join(f"{row['variant']} seed {row['seed']}: selected epoch {row['best_epoch']}, final epoch {row['final_epoch']}, {row['stopping']['stop_reason']}" for row in report["runs"]) + ".")
    limit = "One reused validation donor, no sealed test. Fresh plateau-scheduled runs differ from the original 32-epoch cosine schedule. Historical BAGM changes architecture and optimization. Safety-cap endpoints do not establish convergence; patience also does not prove global optimality."
    evidence = {key: {"status": "not_tested", "summary": "Not evaluated in this validation optimization follow-up."} for key in ("faithfulness", "null_calibration", "patient_replication", "external_support", "perturbation_support")}
    evidence.update(predictive_gain={"status": "supported" if a["historical_endpoint_criterion_passed"] else "mixed", "summary": observed},
        stability={"status": "supported" if a["both_seeds_local_beats_self_nll"] else "mixed", "summary": "Local/self direction across two optimization seeds; no independent patient replication."})
    return {"schema_name": "bagm_result_record", "schema_version": 1, "revision": 1,
        "result_id": RESULT_ID, "title": "Local Gene Residual reconstruction with validation early stopping",
        "status": "verified", "outcome": outcome, "experiment_type": "predictive_benchmark",
        "method_family": "local_gene_residual", "lifecycle_stage": "exploratory_validation", "study_axis": "early_stopping",
        "campaign_ids": [CAMPAIGN], "method": {"name": "Local Gene Residual with validation patience", "model_family": "Two-layer MLP and gene-wise local residual with hurdle NB2 output",
            "graph_context": "Up to 16 non-self neighbors within 75 micrometers; matched self-only control",
            "masking_or_perturbation": "Inherited target-isolated Uniform1..1000 gene masks and fixed validation masks",
            "evaluation_design": "Local/self crossed with seeds0/1; min50, patience30, min_delta .0001, max500; strict best and terminal metrics; historical and original32epoch comparisons",
            "implementation_version": report["git_commit"]},
        "conclusion": {"question": "Does longer optimization improve local count reconstruction and preserve local predictive gain?",
            "estimand": "Validation-selected target-isolated partial-gene count reconstruction on one reused held-out donor",
            "observed_result": observed, "strongest_alternative_explanation": "Validation reuse, adaptive exposure, and donor-specific local copying or segmentation effects can explain gains without wider generalization.",
            "controls": ["Both local/self model seeds and all registered failures retained", "Identical source/data/fixed masks and complete common-prefix training masks/order", "Independent stopping replay, strict best and final checkpoint replay/catalog checks", "Original32epoch and historical BAGM references; train-fitted constants and zero MAE"],
            "remaining_uncertainty": limit, "maximum_defensible_claim": "Selected-checkpoint performance on this reused validation split; no sealed-test, independent-patient, global-convergence, or causal claim."},
        "evidence": evidence, "provenance": {"created_at_utc": now, "updated_at_utc": now,
            "git_commit": report["git_commit"], "run_ids": sorted({s["run_id"] for s in sources if "run_id" in s}),
            "expected_seeds": [0, 1], "included_seeds": [0, 1], "expected_folds": [0], "included_folds": [0],
            "failed_or_excluded_runs": [{"run_id": row["run_id"], "reason": str(row["failure_category"])} for row in report["failed_attempts"]], "sources": sources}, "files": []}


def publish_result(report, output, paths):
    from spatial_benchmark.result_catalog import validate_result_manifest, validate_result_tree, write_or_check_catalog
    sources = []
    for group in (report["runs"], report["original_32_epoch_runs"], report["historical"], report["failed_attempts"]):
        for row in group:
            root = RunArchive.artifact_path_for(row["run_id"], paths)
            for name in ("provenance/artifact_checksums.json", "summary.json"):
                sources.append({"kind": "run_artifact", "root": "artifact", "path": str((root / name).relative_to(paths.artifact_root)),
                    "sha256": sha256_file(root / name), "verification_status": "verified", "run_id": row["run_id"]})
    for name in ("manifest.json", "summary.json"):
        sources.append({"kind": "report", "root": "report", "path": str((output / name).relative_to(paths.report_root)),
            "sha256": sha256_file(output / name), "verification_status": "verified"})
    payload = result_record(report, sources)
    destination = paths.result_root / "predictive_benchmark/local_gene_residual" / RESULT_ID
    if destination.exists():
        validated = validate_result_manifest(destination / "result.yaml", paths=paths, verify_sources=True, verify_payloads=True)
        require(validated.payload["provenance"]["sources"] == sources, "Existing result binds different immutable sources")
        require(validated.payload["conclusion"] == payload["conclusion"], "Existing result has a different scientific conclusion")
        write_or_check_catalog(paths=paths)
        write_or_check_catalog(paths=paths, check=True)
        return str(destination)
    paths.result_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".local-early-result-", dir=paths.result_root.parent) as temp:
        staged_paths = replace(paths, result_root=Path(temp))
        directory = staged_paths.result_root / "predictive_benchmark/local_gene_residual" / RESULT_ID
        directory.mkdir(parents=True)
        (directory / "result.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        (directory / "README.md").write_text(f"# {payload['title']}\n\nOutcome: **{payload['outcome']}**.\n\n" + payload["conclusion"]["observed_result"] + "\n\n" + payload["conclusion"]["remaining_uncertainty"] + "\n\n" + payload["conclusion"]["maximum_defensible_claim"] + "\n")
        validate_result_manifest(directory / "result.yaml", paths=staged_paths, verify_sources=True, verify_payloads=True)
        paths.result_root.mkdir(parents=True, exist_ok=True)
        with (paths.result_root / ".promotion.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            require(not any(r.payload["result_id"] == RESULT_ID for r in validate_result_tree(paths=paths)), "Result ID already registered")
            destination.parent.mkdir(parents=True, exist_ok=True)
            _rename_no_replace(directory, destination)
            write_or_check_catalog(paths=paths)
            write_or_check_catalog(paths=paths, check=True)
    return str(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", nargs=4, required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--publish-result", action="store_true")
    args = parser.parse_args(argv)
    require(args.version.isascii() and args.version.isalnum(), "Version must contain ASCII letters/digits only")
    paths = current_paths()
    output = paths.report_root / "local_gene_early_stopping" / args.version
    require(not output.exists() or args.publish_result, "Immutable report version already exists")
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    rows = sorted((load_run(run, paths, registry) for run in args.run_ids), key=lambda row: (row["variant"], row["seed"]))
    verify_comparison(rows)
    historical, original = load_historical(paths, registry, rows[0]), load_original(paths, registry, rows)
    zero_mae = rows[0]["metrics"]["observed_count_mean"]
    require(all(close(row["metrics"]["observed_count_mean"], zero_mae) for row in rows), "Validation targets differ")
    first_root = RunArchive.artifact_path_for(rows[0]["run_id"], paths)
    report = {"schema": "local_gene_early_stopping_report_v1", "campaign_id": CAMPAIGN,
        "created_at": datetime.now(timezone.utc).isoformat(), "version": args.version,
        "git_commit": read_json(first_root / "provenance/git.json")["commit"],
        "runs": rows, "historical": historical, "original_32_epoch_runs": original,
        "original_comparison": compare_original(rows, original), "assessment": assess(rows, historical),
        "failed_attempts": load_failures(registry), "zero_predictor_mae": zero_mae,
        "validation_donor_groups": 1, "no_sealed_test": True, "global_convergence_established": False,
        "seed_aggregate_policy": "all_prespecified_seeds_equal_weight_no_best_seed_selection"}
    if output.exists():
        manifest = read_json(output / "manifest.json")
        for name, digest in manifest["files"].items():
            file = (output / name).resolve()
            require(file.is_relative_to(output.resolve()) and sha256_file(file) == digest, "Existing report checksum differs")
        saved = read_json(output / "summary.json")
        require(saved["schema"] == report["schema"] and saved["campaign_id"] == CAMPAIGN, "Existing report belongs to another protocol")
        for key in ("runs", "historical", "original_32_epoch_runs", "assessment", "failed_attempts"):
            require(saved[key] == report[key], f"Existing report scientific source differs: {key}")
        result = publish_result(saved, output, paths)
        print(json.dumps({"report": str(output), "result": result, "already_verified": True}))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        (temporary / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (temporary / "report.md").write_text(render_report(report))
        for filename, values in (("comparison.csv", [{"run_id": r["run_id"], "variant": r["variant"], "seed": r["seed"], "best_epoch": r["best_epoch"], "final_epoch": r["final_epoch"], "stop_reason": r["stopping"]["stop_reason"], **r["metrics"]} for r in rows]),
                ("terminal_metrics.csv", [{"run_id": r["run_id"], "variant": r["variant"], "seed": r["seed"], **r["terminal_metrics"]} for r in rows]),
                ("original_comparison.csv", report["original_comparison"]),
                ("per_core.csv", [{"run_id": r["run_id"], "core_alias": alias, **metric} for r in rows for alias, metric in sorted(r["per_core"].items())]),
                ("mask_strata.csv", [{"run_id": r["run_id"], "masked_gene_range": name, **metric} for r in rows for name, metric in sorted(r["mask_strata"].items())]),
                ("learning_curves.csv", [{"run_id": r["run_id"], **{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in epoch.items()}} for r in rows for epoch in r["history"]])):
            _csv(temporary / filename, values)
        (temporary / "stopping.json").write_text(json.dumps({row["run_id"]: row["stopping"] for row in rows}, indent=2, sort_keys=True) + "\n")
        learning_curve(rows, historical, temporary / "learning_curves.png")
        shutil.copyfile(Path(__file__), temporary / "report_builder.py")
        from scripts.analysis import report_local_gene_residual as old_report
        shutil.copyfile(Path(old_report.__file__), temporary / "original_report_builder.py")
        manifest = {"schema": "checksummed_local_report_v1", "files": {file.name: sha256_file(file) for file in sorted(temporary.iterdir()) if file.is_file()}}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _rename_no_replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    result = publish_result(report, output, paths) if args.publish_result else None
    print(json.dumps({"report": str(output), "result": result, "assessment": report["assessment"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
