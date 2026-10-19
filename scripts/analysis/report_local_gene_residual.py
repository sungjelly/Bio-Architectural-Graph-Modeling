#!/usr/bin/env python3
"""Verify and report the complete, prespecified local/self count experiment."""
from __future__ import annotations

import argparse
import csv
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


CAMPAIGN = "cmp_20261004_local_gene_residual"
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
    require(config["trainer"]["epochs"] == summary["final_epoch"] == 32, "Prespecified 32-epoch budget incomplete")
    require(config["trainer"]["batch_size"] == 1024, "Prespecified target batch changed")
    for name, expected in {"learning_rate": .001, "minimum_learning_rate": .0001,
            "weight_decay": .00001, "gradient_clip": 1., "ordering_seed": 2026100401,
            "scheduler": "fixed_cosine", "precision": "bfloat16_autocast_fp32_likelihood"}.items():
        require(config["trainer"].get(name) == expected, f"Prespecified trainer field changed: {name}")
    require(config.get("masking") == {"kind": "target_isolated_uniform_1_1000",
        "training_base_seed": 2026092601,
        "validation_namespace": "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"}, "Inherited masking configuration changed")
    require(summary["checkpoint_reload_verified"] is True, "Selected checkpoint replay absent")
    final = {key.removeprefix("val/"): value for key, value in read_json(root / "metrics/final.json").items()}
    require(final["masked_entries"] == MASKED_ENTRIES, "Validation masked-entry support changed")
    require(close(final["count_nll"], summary["primary_metric_value"]), "Summary primary value differs")
    history = [json.loads(line) for line in (root / "metrics/epoch_history.jsonl").read_text().splitlines() if line.strip()]
    require([row["epoch"] for row in history] == list(range(1, 33)), "Epoch history incomplete or duplicated")
    require(all(row["target_visits"] == 208_696 for row in history), "Incomplete training target exposure")
    require(all(row["val/masked_entries"] == MASKED_ENTRIES for row in history), "Validation support differs across epochs")
    selected = min(history, key=lambda row: row["val/count_nll"])
    require(selected["epoch"] == summary["best_epoch"], "Checkpoint is not strict earliest validation NLL minimum")
    for key in ("count_nll", "raw_count_mae", "raw_count_rmse"):
        require(close(final[key], selected["val/" + key]), f"Selected metric replay differs: {key}")
    terminal = float(np.mean([row["val/count_nll"] for row in history[-4:]]))
    require(close(terminal, summary["terminal_mean_count_nll"]), "Terminal epochs 29-32 summary differs")
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
    require([row["epoch"] for row in coverage] == list(range(1, 33)), "Training coverage receipts incomplete")
    for row in coverage:
        require(set(row["cores"]) == TRAIN_ALIASES, "Training core coverage differs")
        require(sum(core["target_visits"] for core in row["cores"].values()) == 208_696, "Training visits incomplete")
        require(all(core["masked_entries"] > 0 and len(core["mask_sha256"]) == 64 for core in row["cores"].values()), "Training mask receipt invalid")
    replay = read_json(root / "diagnostics/checkpoint_replay.json")
    require(replay["passed"] is True and replay["maximum_absolute_delta"] <= 1e-5, "Selected-checkpoint replay failed")
    require(replay["best_epoch"] == summary["best_epoch"], "Replay checkpoint epoch differs")
    require(replay["checkpoint_sha256"] == sha256_file(root / "checkpoints/best.ckpt"), "Replay checkpoint checksum differs")
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
        "best_epoch": summary["best_epoch"], "final_epoch": 32, "metrics": final,
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
            "results/per_core.json", "results/mask_strata.json")),
    }


def verify_comparison(rows: list[dict]) -> None:
    require(len(rows) == 4 and len({row["run_id"] for row in rows}) == 4, "Exactly four distinct runs are required")
    require({(row["variant"], row["seed"]) for row in rows} == EXPECTED_PAIRS, "Expected local/self crossed with seeds 0/1")
    for row in rows:
        expected_graph = {"kind": "fixed_local_summary" if row["variant"] == "local" else "none",
            "neighbor_k": 16 if row["variant"] == "local" else 0, "radius_um": 75.0, "self_loops": False}
        require(row["config"]["graph"] == expected_graph, "Variant-specific local graph contract changed")
    first = rows[0]
    for row in rows[1:]:
        for key in ("source_hashes", "data_provenance", "validation_masks", "training_coverage_sha256"):
            require(row[key] == first[key], f"Runs differ in paired comparison field: {key}")
        for key in ("dataset", "masking", "trainer", "evaluation", "metadata"):
            require(row["config"][key] == first["config"][key], f"Runs differ in scientific configuration: {key}")
        for key in ("family", "hidden_dim", "hidden_layers", "dropout", "output_distribution"):
            require(row["config"]["model"][key] == first["config"]["model"][key], f"Model ablation changed unrelated field: {key}")
        require(row["optimization_steps"] == first["optimization_steps"], "Optimizer exposure differs")
        for key in ("count_nll", "raw_count_mae", "raw_count_rmse"):
            require(close(row["baseline"]["pooled"][key], first["baseline"]["pooled"][key]), "Train-fitted constant baseline differs")


def load_historical(paths, registry) -> list[dict]:
    result = []
    for label, run_id in HISTORICAL_RUNS.items():
        root, _ = _verified_root(run_id, paths, registry)
        final = read_json(root / "metrics/final.json")
        summary = read_json(root / "summary.json")
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
    margin = means["local"] <= PRIMARY_NLL_THRESHOLD
    each_nll = all(row["metrics"]["count_nll"] < hurdle["count_nll"] for row in local)
    each_rmse = all(row["metrics"]["raw_count_rmse"] <= RMSE_THRESHOLD for row in local)
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
        "nll_margin_threshold": PRIMARY_NLL_THRESHOLD, "rmse_upper_threshold": RMSE_THRESHOLD,
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


def render_report(report):
    assessment, rows = report["assessment"], report["runs"]
    passed = assessment["historical_endpoint_criterion_passed"]
    local_gain = assessment["both_seeds_local_beats_self_nll"]
    lines = ["# Local gene residual count reconstruction", "",
        f"The prespecified historical endpoint criterion **{'passed' if passed else 'did not pass'}**. "
        f"Mean selected local-model NLL across seeds 0/1 was **{assessment['local_mean_selected_count_nll']:.6f}**, "
        f"against the threshold **{PRIMARY_NLL_THRESHOLD:.6f}**. "
        f"The local component **{'improved' if local_gain else 'did not improve'}** NLL over the matched self-only model in both seeds.", "",
        "This is an exploratory validation comparison on one previously used held-out donor. The registry uses `validation_confirmation` as a catalog bucket; this experiment is not an independent or confirmatory study. It does not establish independent patient generalization, biological mechanism, or causality.", "",
        "![Validation count NLL and RMSE across all training epochs and all four runs](learning_curves.png)", "",
        "## Selected checkpoints and controls", "",
        "All point errors use the marginal predictive mean. Selection minimizes pooled count NLL; it does not select the best MAE or the best seed.", "",
        "| Model | Seed | Selected / final epoch | Count NLL ↓ | Raw MAE ↓ | Raw RMSE ↓ | Epochs 29–32 NLL |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for historical in report["historical"]:
        lines.append(f"| {historical['label']} | 0 | {historical['best_epoch']} / {historical['final_epoch']} | {historical['count_nll']:.6f} | {historical['raw_count_mae']:.6f} | {historical['raw_count_rmse']:.6f} | — |")
    for row in rows:
        metric = row["metrics"]
        lines.append(f"| {'Local residual' if row['variant']=='local' else 'Self only'} | {row['seed']} | {row['best_epoch']} / 32 | {metric['count_nll']:.6f} | {metric['raw_count_mae']:.6f} | {metric['raw_count_rmse']:.6f} | {row['terminal_mean_count_nll']:.6f} |")
    baseline = rows[0]["baseline"]["pooled"]
    lines += [f"| Train-fitted gene constants, θ=1 | — | — | {baseline['count_nll']:.6f} | {baseline['raw_count_mae']:.6f} | {baseline['raw_count_rmse']:.6f} | — |",
        f"| Always zero | — | — | — | {report['zero_predictor_mae']:.6f} | — | — |", "",
        "The constant hurdle baseline fits prevalence and conditional-positive mean on training cells, with dispersion fixed at θ=1. It is not a maximum-likelihood fit of all hurdle parameters. Always-zero MAE is the observed masked-count mean (approximately 0.15125); low MAE alone does not establish useful expression recovery.", "",
        "## Prespecified decisions", "",
        f"- Local mean NLL ≤ {PRIMARY_NLL_THRESHOLD:.10f}: **{assessment['local_mean_nll_margin_passed']}**.",
        f"- Both local seeds improve historical hurdle NLL: **{assessment['both_local_seeds_improve_historical_nll']}**.",
        f"- Both local seeds have RMSE ≤ {RMSE_THRESHOLD:.10f}: **{assessment['both_local_seeds_rmse_guard_passed']}**.",
        f"- Local NLL beats matched self-only NLL in both seeds: **{local_gain}**.", "",
        "These are practical exploratory thresholds, not tests of statistical significance. Two optimizer seeds do not supply two independent biological replicates.", "",
        "| Seed | Local − self NLL | Local − self RMSE | SO2-C27 NLL difference | SO2-C28 NLL difference |",
        "|---:|---:|---:|---:|---:|"]
    for pair in assessment["paired_local_self_differences"]:
        lines.append(f"| {pair['seed']} | {pair['local_minus_self_count_nll']:+.6f} | {pair['local_minus_self_raw_count_rmse']:+.6f} | {pair['per_core_count_nll_differences']['SO2-C27']:+.6f} | {pair['per_core_count_nll_differences']['SO2-C28']:+.6f} |")
    lines += ["", "Negative differences favor local context. Both validation cores come from the same donor.", "",
        "## Additional reconstruction metrics", "",
        "| Model | Seed | Log1p MAE | Observed-positive marginal MAE | Conditional-positive MAE | Detection BCE | Detection AP |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        metric = row["metrics"]
        lines.append(f"| {row['variant']} | {row['seed']} | {metric['log1p_count_mae']:.6f} | {metric['positive_marginal_raw_count_mae']:.6f} | {metric['positive_conditional_raw_count_mae']:.6f} | {metric['detection_bce']:.6f} | {metric['detection_average_precision']:.6f} |")
    lines += ["", "Observed-positive marginal errors score the complete predictor on entries known to be positive. Conditional-positive errors score the count component after conditioning on positivity; these are different estimands. Detection uses the fixed threshold 0.5 for precision/recall/F1, and AP is threshold-free with histogram approximation.", "",
        "## Runtime and model size", "",
        "| Model | Seed | Parameters | Runtime (min) | Mean epoch (s) | Optimizer steps | Peak allocated GPU / RSS (GiB) |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["historical"]:
        lines.append(f"| {row['label']} | 0 | {row['parameter_count']:,} | {row['duration_seconds']/60:.1f} | — | — | — |")
    for row in rows:
        lines.append(f"| {row['variant']} | {row['seed']} | {row['parameter_count']:,} | {row['duration_seconds']/60:.1f} | {row['mean_epoch_seconds']:.1f} | {row['optimization_steps']:,} | {row['peak_vram_gib']:.2f} / {row['peak_rss_gib']:.2f} |")
    lines += ["", "Runtime includes data loading, baseline scoring, training and final replay. Epoch timing includes validation and checkpoint writes. Concurrent jobs and differing epoch budgets prevent a controlled speedup claim against historical BAGM runs.", "",
        "## Evidence and limits", "",
        "All four 32-epoch runs passed immutable-bundle verification, matched source/data/split/normalization fingerprints, identical validation and training masks/order, full target coverage, and selected-checkpoint replay. Validation comprises 37,367 cells and 18,708,207 masked entries. Training comprises 208,696 cells; each is visited once per epoch. The configured batch size 1,024 is the maximum: each core uses balanced batches formed with `numpy.array_split`. Initial and selected checkpoints passed the FP32/BF16 likelihood check on one fixed training minibatch with an absolute NLL tolerance of 0.001; this is a limited numerical check, not full-cohort precision equivalence. Biological generalization remains limited to one held-out donor and no sealed test set.", "",
        "The local model is a two-hidden-layer masked-cell MLP plus four learned same-gene neighborhood weights per gene, with a Bernoulli hurdle/zero-truncated NB2 output. Up to 16 clean, non-self neighbors within 75 μm supply log1p mean counts and positive fractions. Raw physical density is recorded by the loader but is not an input to this model. The self-only ablation removes the local residual.", "",
        "Historical NB and hurdle checkpoints used different stopping epochs and six updates per epoch. These new runs change architecture and optimization together. A favorable endpoint comparison cannot attribute the gain solely to architecture. Local-versus-self comparisons share masks, target order, optimizer schedule and budget and isolate the added local inputs within the new design.", "",
        "Likelihood includes its normalization constants. Positive-marginal and conditional-positive errors are distinct in `comparison.csv`; per-core metrics and mask-severity metrics are retained in their own CSVs and `summary.json`. Detection AP/AUROC use 4,096 probability bins, so they are approximations. No gene-level or patient-population uncertainty is inferred from pooled cell counts.", "",
        "Files: [full metrics](comparison.csv), [per-core metrics](per_core.csv), [mask severity](mask_strata.csv), [learning curves](learning_curves.csv), [machine-readable summary](summary.json), [checksums](manifest.json).", "",
        "## Immutable sources", ""]
    lines += [f"- {row['variant']}, seed {row['seed']}: `{row['run_id']}`" for row in rows]
    lines += [f"- {row['label']}: `{row['run_id']}`" for row in report["historical"]]
    lines += ["", "## Failed attempts", ""]
    lines += ([f"- `{row['run_id']}` ({row['profile']}, {row['variant']}, seed {row['seed']}): {row['failure_category']}; preserved and excluded from completed-run metrics." for row in report["failed_attempts"]]
              or ["No failed attempts were registered in this campaign at report creation."])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", nargs=4, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(argv)
    require(args.version.isascii() and args.version.isalnum(), "Version must contain ASCII letters/digits only")
    paths = current_paths()
    output = paths.report_root / "local_gene_residual" / args.version
    require(not output.exists(), "Versioned report exists; use a new version")
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    # No report directory is created until every source has passed verification.
    rows = sorted((load_run(run_id, paths, registry) for run_id in args.run_ids), key=lambda row: (row["variant"], row["seed"]))
    verify_comparison(rows)
    historical = load_historical(paths, registry)
    assessment = assess(rows, historical)
    zero_mae = rows[0]["metrics"]["observed_count_mean"]
    require(all(close(row["metrics"]["observed_count_mean"], zero_mae) for row in rows), "Validation raw targets differ")
    report = {
        "schema": "local_gene_residual_report_v1", "campaign_id": CAMPAIGN,
        "created_at": datetime.now(timezone.utc).isoformat(), "version": args.version,
        "analysis_type": "exploratory_previous_validation_reuse", "runs": rows,
        "historical": historical, "assessment": assessment, "zero_predictor_mae": zero_mae,
        "failed_attempts": load_failures(registry), "validation_donor_groups": 1,
        "validation_cells": 37_367, "validation_masked_entries": MASKED_ENTRIES,
        "seed_aggregate_policy": "all_prespecified_seeds_equal_weight_no_best_seed_selection",
        "maximum_claim": "Validation endpoint performance on one previously used held-out donor; no independent generalization or causal claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        (temporary / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (temporary / "report.md").write_text(render_report(report), encoding="utf-8")
        _csv(temporary / "comparison.csv", [{"run_id": row["run_id"], "variant": row["variant"], "seed": row["seed"],
            "best_epoch": row["best_epoch"], "final_epoch": row["final_epoch"], "terminal_mean_count_nll": row["terminal_mean_count_nll"],
            "parameter_count": row["parameter_count"], "duration_seconds": row["duration_seconds"], **row["metrics"]} for row in rows])
        _csv(temporary / "per_core.csv", [{"run_id": row["run_id"], "variant": row["variant"], "seed": row["seed"], "core_alias": alias, **metric}
            for row in rows for alias, metric in sorted(row["per_core"].items())])
        _csv(temporary / "mask_strata.csv", [{"run_id": row["run_id"], "variant": row["variant"], "seed": row["seed"], "masked_gene_range": name, **metric}
            for row in rows for name, metric in sorted(row["mask_strata"].items())])
        _csv(temporary / "learning_curves.csv", [{"run_id": row["run_id"], "variant": row["variant"], "seed": row["seed"],
            **{key: value for key, value in epoch.items() if key != "per_core"}} for row in rows for epoch in row["history"]])
        learning_curve(rows, historical, temporary / "learning_curves.png")
        shutil.copyfile(Path(__file__), temporary / "report_builder.py")
        manifest = {"schema": "checksummed_local_report_v1", "files": {file.name: sha256_file(file) for file in sorted(temporary.iterdir()) if file.is_file()}}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        _rename_no_replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"report": str(output), "assessment": assessment}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
