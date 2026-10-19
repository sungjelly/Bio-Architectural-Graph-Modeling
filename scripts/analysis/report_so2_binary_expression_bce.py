"""Audit finalized SO2 binary-expression BCE training and render its report.

Task contract (exploratory, specified after viewing the learning curve): consume
only immutable, checksum-verified successful runs; summarize the locked primary
optimization criterion and all available occurrence metrics without additional
model evaluation, checkpoint selection, threshold tuning, or raw-data access.
The alternative is that majority-zero prediction or count-supervision
regularization explains apparently favorable performance. Shared detection
metrics and class-imbalance references discriminate these descriptions, but do
not establish graph-specific gain. Validation-selected masked-expression
prediction is the estimand; one donor group, one seed and no test partition
permit only a descriptive exploratory comparison. The frozen optimization
criterion is >1e-4 validation BCE improvement over epoch 1 with finite losses
and finite, nonzero gradients after the first update. Epoch 1 is already trained.

Acceptance: complete contiguous scalar histories, fixed/exhaustive masking
receipts, finite gradients, exact minimum checkpoint selection, replay agreement
within 1e-6, matching comparator split/mask/preprocessing fingerprints, and
successful bundle checksum verification. Stop on any failed contract. GPU and
raw/clinical data access are unnecessary. Outputs are a portable HTML report,
Markdown report, PNG figures, compact comparison CSV, machine summary with
source SHA-256 values and a report artifact manifest. Existing versions are
never overwritten. Registry/catalog status is verified by the monitoring task.

Run from the BAGM repository root after finalization (BAGM_* overrides apply)::

    PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_binary_expression_bce.py \
      --run-id r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863 --version v1

Safe implementation checks before finalization::

    PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_report_so2_binary_expression_bce.py
    /venv/main/bin/python -m py_compile scripts/analysis/report_so2_binary_expression_bce.py
    PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_binary_expression_bce.py --help
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import yaml

from spatial_benchmark.paths import current_paths
from spatial_benchmark.run_archive import verify_run_bundle


DEFAULT_COMPARATOR = "r_20260928T021117Z_9ea46874_s000_f00_a01_41b54097"
PRIMARY = "val/target_only_masked_binary_cross_entropy"
PREFIX = "val/unseen_donor/"
TRAIN_ALIASES = {f"SO2-C{i}" for i in range(15, 27)}
VAL_ALIASES = {"SO2-C27", "SO2-C28"}
PROVENANCE = "provenance/so2_target_isolated_binary_expression_training.json"
HURDLE_PROVENANCE = "provenance/so2_target_isolated_hurdle_nb_training.json"
SHARED = (
    ("bce", "Detection BCE", "lower"),
    ("detection_brier_score", "Brier score", "lower"),
    ("detection_precision", "Precision at 0.5", "higher"),
    ("detection_recall", "Recall at 0.5", "higher"),
    ("detection_f1", "F1 at 0.5", "higher"),
    ("detection_specificity", "Specificity at 0.5", "higher"),
    ("detection_accuracy", "Accuracy at 0.5", "higher"),
    ("detection_balanced_accuracy", "Balanced accuracy at 0.5", "higher"),
    ("detection_matthews_correlation_coefficient", "MCC at 0.5", "higher"),
    ("predicted_positive_probability_mean", "Mean predicted probability", "descriptive"),
    ("predicted_positive_class_rate", "Predicted positive fraction at 0.5", "descriptive"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"Missing or linked source: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def finite(value: Any, label: str) -> float:
    result = float(value)
    require(math.isfinite(result), f"Nonfinite metric: {label}")
    return result


def near(left: Any, right: Any, label: str, tolerance: float = 1e-6) -> None:
    require(math.isclose(finite(left, label), finite(right, label), abs_tol=tolerance,
                         rel_tol=tolerance), f"Metric disagreement: {label}")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def locate(run_id: str) -> Path:
    require(bool(re.fullmatch(r"r_[A-Za-z0-9_]+", run_id)), "Invalid immutable run ID")
    matches = list((current_paths().artifact_root / "runs").glob(f"*/*/{run_id}"))
    require(len(matches) == 1, f"Expected one finalized bundle for {run_id}, found {len(matches)}")
    return matches[0]


def load_bundle(run_id: str, provenance_path: str) -> dict[str, Any]:
    root = locate(run_id)
    require((root / "_SUCCESS").is_file(), f"Successful completion marker absent: {run_id}")
    verification = verify_run_bundle(root, require_success_contract=True)
    summary = read_json(root / "summary.json")
    require(summary["run_id"] == run_id and summary["status"] == "success", "Summary identity/status")
    require(summary["model_seed"] == 0 and summary["test_artifacts_present"] is False,
            "Run is outside the locked seed-0/no-test scope")
    receipt = read_json(root / "diagnostics/final_checkpoint_reload_verification.json")
    provenance = read_json(root / provenance_path)
    require(receipt["verified"] is True and summary["checkpoint_reload_verified"] is True,
            "Checkpoint replay is unverified")
    checkpoint = record(root / "checkpoints/best.ckpt")
    require(checkpoint["sha256"] == receipt["checkpoint_sha256"] == provenance["checkpoint_sha256"],
            "Best-checkpoint checksum mismatch")
    require(sorted(p.name for p in (root / "checkpoints").iterdir()) == ["best.ckpt"],
            "Final checkpoint layout must retain best only")
    epoch_rows = rows(root / "results/epoch_metrics.csv")
    require([int(r["global_epoch"]) for r in epoch_rows] == list(range(1, summary["final_epoch"] + 1)),
            "Missing, duplicate or unordered epoch history")
    require(receipt["best_epoch"] == summary["best_epoch"], "Selected epoch mismatch")
    common_sources = ["_SUCCESS", "manifest.yaml", "summary.json", "config.resolved.yaml",
                      "metrics/final.json", "results/epoch_metrics.csv", "results/per_core_epoch_metrics.csv",
                      "diagnostics/final_checkpoint_reload_verification.json", provenance_path,
                      "provenance/artifact_checksums.json", "provenance/hardware.json"]
    sources = {name: record(root / name) for name in common_sources}
    sources["checkpoints/best.ckpt"] = checkpoint
    return {"root": root, "summary": summary, "metrics": read_json(root / "metrics/final.json"),
            "config": yaml.safe_load((root / "config.resolved.yaml").read_text(encoding="utf-8")),
            "provenance": provenance, "receipt": receipt, "epochs": epoch_rows,
            "cores": rows(root / "results/per_core_epoch_metrics.csv"), "sources": sources,
            "bundle_verification": verification}


def coverage_checks(epoch_rows: list[dict[str, str]]) -> None:
    fixed = epoch_rows[0]["validation_coverage_receipts_json"]
    for row in epoch_rows:
        epoch = int(row["global_epoch"])
        require(int(row["train_target_cells"]) == 208696 and int(row["validation_target_cells"]) == 37367,
                f"Unexpected target count in epoch {epoch}")
        require(row["validation_coverage_receipts_json"] == fixed, "Fixed validation receipts changed")
        for role, aliases in (("training", TRAIN_ALIASES), ("validation", VAL_ALIASES)):
            receipts = json.loads(row[f"{role}_coverage_receipts_json"])
            require(set(receipts) == aliases, f"Coverage aliases: {role}/{epoch}")
            for alias, receipt in receipts.items():
                require(receipt["target_visit_count_min"] == receipt["target_visit_count_max"] == 1,
                        f"Target was not visited exactly once: {alias}/{epoch}")
                require(receipt["neighbor_masked_entry_count"] == 0, "Neighbor mask leakage")
                partition = receipt["partition_coverage_receipt"]
                require(partition["coverage_exhaustive"] and partition["partitions_disjoint"],
                        "Target partition coverage failure")
                require(partition["total_target_count"] == partition["unique_target_count"] == receipt["target_cells"],
                        "Coverage count disagreement")
                for shard in receipt["rank_shards"]:
                    mask = shard["mask"]
                    require(mask["all_masks_nonempty"] and mask["targets_unique"] and mask["targets_in_bounds"],
                            "Invalid target mask receipt")
                    require(1 <= mask["masked_gene_count_min"] <= mask["masked_gene_count_max"] <= 1000,
                            "Masked gene count outside locked bounds")
                    require(shard["neighbor_masked_entry_count"] == 0, "Shard neighbor mask leakage")


def audit_replayed_detection(best: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Check saved final occurrence metrics against the selected epoch."""
    for key, value in metrics.items():
        finite(value, key)
        if not key.startswith(PREFIX):
            continue
        suffix = key[len(PREFIX):]
        column = "validation_" + ("detection_mcc" if suffix == "detection_matthews_correlation_coefficient" else suffix)
        require(column in best, f"Unknown selected-epoch metric mapping: {key}")
        if suffix in {"detection_true_positive", "detection_true_negative", "detection_false_positive", "detection_false_negative"}:
            require(int(best[column]) == value, f"Selected/replayed confusion count differs: {suffix}")
        elif suffix == "detection_threshold":
            require(float(best[column]) == value == 0.5, "Fixed decision threshold differs")
        else:
            near(best[column], value, f"Selected/replayed {suffix}")


def audit_current(bundle: dict[str, Any]) -> dict[str, Any]:
    summary, metrics, epoch_rows = bundle["summary"], bundle["metrics"], bundle["epochs"]
    require(summary["primary_metric_name"] == PRIMARY, "Wrong primary metric")
    final_epoch, best_epoch = summary["final_epoch"], summary["best_epoch"]
    require(50 <= final_epoch <= 300 and summary["optimizer_steps"] == final_epoch * 6,
            "Epoch/update counts violate locked controls")
    control = summary["patience_control_state"]
    require(control["should_stop"] is True, "Training has no terminal stopping decision")
    require((summary["stop_reason"] == "patience" and control["bad_validations"] == 25) or
            (summary["stop_reason"] == "maximum_epochs" and final_epoch == 300), "Unexpected stopping decision")
    for row in epoch_rows:
        for key in ("train_pooled_masked_binary_cross_entropy", "validation_pooled_masked_binary_cross_entropy",
                    "epoch_duration_seconds", "learning_rate", "peak_vram_gib_all_ranks"):
            require(finite(row[key], key) > 0, f"Invalid positive scalar: {key}")
    best = min(epoch_rows, key=lambda r: float(r["validation_pooled_masked_binary_cross_entropy"]))
    require(int(best["global_epoch"]) == best_epoch, "Selected checkpoint is not the first strict BCE minimum")
    near(summary["primary_metric_value"], metrics[PRIMARY], "Summary/final BCE", 1e-12)
    near(best["validation_pooled_masked_binary_cross_entropy"], metrics[PRIMARY], "Selected/replayed BCE")
    near(bundle["receipt"]["replayed_validation_metric"], metrics[PRIMARY], "Replay receipt/final BCE")
    audit_replayed_detection(best, metrics)
    for short in ("masked_entries", "positive_entries", "target_cells"):
        require(int(best[f"validation_{short}"]) == metrics[f"val/{short}"], f"Final support mismatch: {short}")
    coverage_checks(epoch_rows)
    require(canonical_hash(json.loads(best["validation_coverage_receipts_json"])) ==
            bundle["provenance"]["validation_coverage_receipts_sha256"], "Final mask checksum differs")
    core_rows = bundle["cores"]
    require(len(core_rows) == final_epoch * 14, "Incomplete core histories")
    for epoch in range(1, final_epoch + 1):
        current = [r for r in core_rows if int(r["global_epoch"]) == epoch]
        require(len(current) == 14, "Wrong rows per core/epoch")
        for split, aliases in (("training", TRAIN_ALIASES), ("validation", VAL_ALIASES)):
            require({r["core_alias"] for r in current if r["split"] == split} == aliases, "Core split changed")
        for row in current:
            finite(row["pooled_masked_binary_cross_entropy"], "Core BCE")
    boundaries: set[int] = set()
    for name, multiplier in (("gradient_direction_metrics.csv", 1), ("gradient_direction_by_block.csv", 4),
                              ("gradient_direction_detection_head.csv", 1)):
        relative = "results/" + name
        gradients = rows(bundle["root"] / relative)
        require([int(r["global_epoch"]) for r in gradients] ==
                [e for e in range(1, final_epoch + 1) for _ in range(multiplier)], "Gradient history missing")
        for row in gradients:
            epoch = int(row["global_epoch"])
            require(int(row["optimizer_updates_observed"]) == 6, "Missing gradient updates")
            for key in ("gradient_norm_min_before_clip", "gradient_norm_mean_before_clip", "gradient_norm_max_before_clip"):
                value = finite(row[key], key)
                require(value >= 0 and ((epoch == 1 and key == "gradient_norm_min_before_clip") or value > 0),
                        "Required gradient norm is zero")
            if epoch == 1:
                require(int(row["consecutive_optimizer_step_cosine_valid_pairs"]) >= 4,
                        "Insufficient nonzero post-first-update gradients in epoch 1")
            for key, value in row.items():
                if "cosine" in key and value:
                    finite(value, key)
            if row["resume_boundary_unavailable"] == "True":
                boundaries.add(epoch)
        bundle["sources"][relative] = record(bundle["root"] / relative)
    improvement = float(epoch_rows[0]["validation_pooled_masked_binary_cross_entropy"]) - metrics[PRIMARY]
    return {"primary_optimization_criterion_passed": improvement > 1e-4,
            "absolute_bce_improvement_from_epoch1": improvement,
            "relative_bce_improvement_from_epoch1": improvement / float(epoch_rows[0]["validation_pooled_masked_binary_cross_entropy"]),
            "histories_contiguous": True, "losses_and_gradient_norms_finite": True,
            "post_first_epoch_gradient_norms_positive": True,
            "target_coverage_and_fixed_masks_verified": True, "secondary_replay_tolerance": 1e-6,
            "secondary_replay_verified": True, "resume_gradient_boundary_epochs": sorted(boundaries)}


def detection_metrics(bundle: dict[str, Any], *, binary: bool) -> dict[str, float]:
    raw = bundle["metrics"]
    out = {key[len(PREFIX):]: finite(value, key) for key, value in raw.items() if key.startswith(PREFIX)}
    out["bce"] = raw[PRIMARY] if binary else out["detection_bce"]
    tp, tn, fp, fn = (out["detection_" + key] for key in
                      ("true_positive", "true_negative", "false_positive", "false_negative"))
    require(all(value >= 0 and value.is_integer() for value in (tp, tn, fp, fn)),
            "Confusion counts must be nonnegative integers")
    total = tp + tn + fp + fn
    require(total > 0 and tp + fn > 0 and tn + fp > 0, "Detection comparison requires both observed classes")
    require(total == raw["val/masked_entries"] and tp + fn == raw["val/positive_entries"],
            "Confusion matrix support differs from masked-entry support")
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    derived = {"detection_accuracy": (tp + tn) / total,
               "detection_precision": tp / (tp + fp) if tp + fp else 0.0,
               "detection_recall": tp / (tp + fn),
               "detection_specificity": tn / (tn + fp),
               "detection_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
               "observed_positive_rate": (tp + fn) / total,
               "detection_balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
               "detection_matthews_correlation_coefficient": (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0,
               "predicted_positive_class_rate": (tp + fp) / total}
    for key, value in derived.items():
        if key in out:
            near(out[key], value, "Derived " + key)
        out[key] = value
    return out


def compare(current: dict[str, Any], hurdle: dict[str, Any]) -> list[dict[str, Any]]:
    require(current["config"]["graph"] == hurdle["config"]["graph"], "Resolved graph configuration differs")
    require(current["config"]["dataset"]["dataset_fingerprint"] == hurdle["config"]["dataset"]["dataset_fingerprint"],
            "Dataset fingerprint differs")
    for key in ("preprocessing_fingerprint", "split_fingerprint", "overlay_manifest_sha256",
                "validation_coverage_receipts_sha256", "training_cells", "validation_cells",
                "training_aliases", "validation_aliases", "detection_threshold", "detection_initialization"):
        require(current["provenance"][key] == hurdle["provenance"][key], f"Comparator mismatch: {key}")
    for key in ("val/masked_entries", "val/positive_entries", "val/target_cells"):
        require(current["metrics"][key] == hurdle["metrics"][key], f"Comparator support mismatch: {key}")
    for bundle in (current, hurdle):
        require(bundle["metrics"][PREFIX + "detection_threshold"] == 0.5, "Final detection threshold is not 0.5")
    for left_epoch, right_epoch in zip(current["epochs"], hurdle["epochs"]):
        for key in ("global_epoch", "core_order_json", "training_coverage_receipts_json", "validation_coverage_receipts_json"):
            require(left_epoch[key] == right_epoch[key], f"Matched epoch protocol differs: {key}")
    current["comparison_checks"] = {"dataset_and_graph_match": True, "final_thresholds_equal_0_5": True,
                                    "exact_mask_receipts_and_core_order_match_through_epoch": min(len(current["epochs"]), len(hurdle["epochs"]))}
    left, right = detection_metrics(current, binary=True), detection_metrics(hurdle, binary=False)
    return [{"metric": name, "label": label, "preferred_direction": direction,
             "binary_bce": left[name], "hurdle_nb": right[name], "binary_minus_hurdle": left[name] - right[name],
             "hurdle_metric_source": "derived_from_confusion_counts" if name in
                 {"detection_accuracy", "detection_balanced_accuracy", "detection_matthews_correlation_coefficient",
                  "predicted_positive_class_rate"} else "metrics/final.json"}
            for name, label, direction in SHARED]


def histogram(bundle: dict[str, Any]) -> dict[str, Any]:
    relative = "results/detection_curve_histogram_4096.json"
    data = read_json(bundle["root"] / relative)
    require(data["bin_count"] == 4096, "Histogram resolution changed")
    for key in ("positive_counts_by_ascending_probability_bin", "negative_counts_by_ascending_probability_bin",
                "probability_sums_by_ascending_probability_bin"):
        require(len(data[key]) == 4096 and all(finite(v, key) >= 0 for v in data[key]), "Invalid histogram")
    require(sum(data["positive_counts_by_ascending_probability_bin"]) == bundle["metrics"]["val/positive_entries"],
            "Histogram positives mismatch")
    require(sum(data["negative_counts_by_ascending_probability_bin"]) + data["positive_entries"] ==
            bundle["metrics"]["val/masked_entries"], "Histogram total mismatch")
    for short, final in (("auroc", "detection_auroc"), ("auprc", "detection_auprc"),
                         ("expected_calibration_error", "detection_expected_calibration_error")):
        near(data[short + "_histogram_4096"], bundle["metrics"][PREFIX + final + "_histogram_4096"], short)
    bundle["sources"][relative] = record(bundle["root"] / relative)
    return data


def runtime_info(bundle: dict[str, Any]) -> dict[str, Any]:
    root = bundle["root"]
    recoveries = []
    for path in sorted((root / "provenance").glob("recovery_*_*.json")):
        if not (path.name.endswith("_start.json") or path.name.endswith("_end.json")):
            continue
        source = read_json(path)
        keep = {key: value for key, value in source.items() if key in {
            "event", "started_at", "ended_at", "recovery_duration_seconds",
            "prior_completed_epoch_duration_seconds", "prior_checkpoint_completed_epoch", "duration_policy",
            "wall_elapsed_including_interruption_seconds"}}
        recoveries.append(keep)
        bundle["sources"][path.relative_to(root).as_posix()] = record(path)
    hardware = read_json(root / "provenance/hardware.json")
    wall_time = next((r["wall_elapsed_including_interruption_seconds"] for r in reversed(recoveries)
                      if "wall_elapsed_including_interruption_seconds" in r), None)
    return {"runner_duration_seconds": bundle["summary"]["duration_seconds"],
            "sum_completed_epoch_seconds": sum(float(r["epoch_duration_seconds"]) for r in bundle["epochs"]),
            "wall_elapsed_including_interruption_seconds": wall_time,
            "recovery_receipts": recoveries,
            "duration_scope": "Runner duration covers resumed process; epoch sum excludes setup, partial epochs and interruption downtime.",
            "devices": [{k: g[k] for k in ("index", "name", "memory_total_mib")} for g in hardware["gpus"]],
            "framework": hardware["torch"], "peak_allocated_vram_gib": max(bundle["summary"]["peak_vram_gib"],
                max(float(r["peak_vram_gib_all_ranks"]) for r in bundle["epochs"])),
            "parallel_schedule": "Four DDP ranks, one per GPU; four disjoint target shards per core; six two-core optimizer updates per epoch."}


def make_figures(output: Path, bundle: dict[str, Any], hurdle: dict[str, Any], hist: dict[str, Any]) -> list[tuple[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    data = bundle["epochs"]
    epoch = [int(r["global_epoch"]) for r in data]
    best = bundle["summary"]["best_epoch"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes[0, 0].plot(epoch, [float(r["train_pooled_masked_binary_cross_entropy"]) for r in data], label="Training")
    axes[0, 0].plot(epoch, [float(r["validation_pooled_masked_binary_cross_entropy"]) for r in data], label="Validation")
    axes[0, 0].set(title="Masked-entry BCE", ylabel="BCE"); axes[0, 0].legend()
    axes[0, 1].plot(epoch, [float(r["validation_pooled_masked_binary_cross_entropy"]) for r in data], label="BCE validation")
    axes[0, 1].axhline(hurdle["metrics"][PREFIX + "detection_bce"], color="#a65f00", ls="--", label="Hurdle selected checkpoint")
    axes[0, 1].set(title="Validation occurrence loss", ylabel="BCE"); axes[0, 1].legend()
    axes[1, 0].step(epoch, [float(r["learning_rate"]) for r in data], where="post", color="#3d5a80")
    axes[1, 0].set(yscale="log", ylabel="Learning rate", title="Learning rate and stopping patience")
    twin = axes[1, 0].twinx()
    twin.plot(epoch, [int(r["bad_validations"]) for r in data], color="#a65f00", alpha=.7)
    twin.axhline(25, color="#a65f00", ls=":"); twin.set_ylabel("Patience counter", color="#a65f00")
    for column, label in (("validation_detection_auprc_histogram_4096", "AUPRC (binned)"),
                          ("validation_detection_recall", "Recall at 0.5"), ("validation_detection_f1", "F1 at 0.5")):
        axes[1, 1].plot(epoch, [float(r[column]) for r in data], label=label)
    axes[1, 1].set(title="Occurrence discrimination", ylabel="Score"); axes[1, 1].legend()
    for ax in axes.flat:
        ax.axvline(best, color="#444444", ls=":", alpha=.7); ax.set_xlabel("Global epoch"); ax.grid(alpha=.15)
        ax.axvline(17.5, color="#b34545", ls="--", alpha=.5)
    figure.savefig(output / "learning.png", dpi=150); plt.close(figure)
    pos = np.asarray(hist["positive_counts_by_ascending_probability_bin"], dtype=float)
    neg = np.asarray(hist["negative_counts_by_ascending_probability_bin"], dtype=float)
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    recall = tp / pos.sum()
    precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    axes[0].plot(np.r_[0, fp / neg.sum()], np.r_[0, recall], color="#28666e")
    axes[0].plot([0, 1], [0, 1], color="grey", ls="--")
    axes[0].set(title="ROC (4,096 bins)", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].step(recall, precision, where="pre", color="#28666e")
    prevalence = pos.sum() / (pos.sum() + neg.sum())
    axes[1].axhline(prevalence, color="grey", ls="--", label="Validation prevalence")
    axes[1].set(title="Precision–recall (4,096 bins)", xlabel="Recall", ylabel="Precision", ylim=(0, 1)); axes[1].legend()
    totals = pos + neg
    probabilities = np.asarray(hist["probability_sums_by_ascending_probability_bin"], dtype=float)
    groups = np.array_split(np.arange(4096), 20)
    supported = [g for g in groups if totals[g].sum() > 0]
    x = [probabilities[g].sum() / totals[g].sum() for g in supported]
    y = [pos[g].sum() / totals[g].sum() for g in supported]
    axes[2].plot(x, y, marker="o", color="#28666e"); axes[2].plot([0, 1], [0, 1], color="grey", ls="--")
    axes[2].set(title="Calibration (20 display bins)", xlabel="Mean predicted probability", ylabel="Observed positive fraction")
    for ax in axes:
        ax.grid(alpha=.15)
    figure.savefig(output / "detection.png", dpi=150); plt.close(figure)
    return [("learning.png", "Training history. Dotted vertical line: selected BCE checkpoint; red dashed line: interruption after epoch 17. Training and validation have different core distributions and masking realizations; their gap is not an overfitting estimate."),
            ("detection.png", "Selected-checkpoint occurrence curves reconstructed from the retained 4,096-bin aggregate histogram. AUPRC means descending-threshold average precision, not trapezoidal PR area. The calibration display combines those bins into 20 groups; reported ECE uses all 4,096 bins. No per-entry predictions or confidence intervals are inferred.")]


def table(headers: list[str], values: list[list[str]], *, html_mode: bool) -> str:
    if html_mode:
        head = "".join("<th>" + html.escape(h) + "</th>" for h in headers)
        body = "".join("<tr>" + "".join("<td>" + html.escape(v) + "</td>" for v in row) + "</tr>" for row in values)
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(row) + " |" for row in values])


def write_reports(output: Path, machine: dict[str, Any], figures: list[tuple[str, str]]) -> None:
    summary, audit, detection = machine["run_summary"], machine["audit"], machine["detection_metrics"]
    prevalence = detection["observed_positive_rate"]
    comparison = machine["comparison"]
    bce_delta = next(r["binary_minus_hurdle"] for r in comparison if r["metric"] == "bce")
    title = "SO2 binary-expression BCE: completed training report"
    lead = (f"Run {summary['run_id']} completed {summary['final_epoch']} epochs; the selected checkpoint is epoch "
            f"{summary['best_epoch']} (stop reason: {summary['stop_reason']}). Replayed validation BCE is "
            f"{detection['bce']:.6f}. The frozen optimization criterion "
            f"{'passed' if audit['primary_optimization_criterion_passed'] else 'failed'}: BCE improved by "
            f"{audit['absolute_bce_improvement_from_epoch1']:.6f} from epoch 1. All report integrity checks passed.")
    design = ("This report is exploratory and was designed after viewing the learning curve. It summarizes masked-gene "
              "occurrence prediction (raw count > 0) with observed target genes, morphology and fully observed neighboring "
              "expression available. Training: 208,696 cells from 12 cores/six donor groups. Validation: 37,367 cells from "
              "two cores/one donor group; 18,708,207 masked entries, including 1,518,702 positives. Each target is visited "
              "once per epoch; validation masks are fixed. Seed 0 only. Validation controls scheduling, stopping and checkpoint selection.")
    imbalance = (f"Observed positive prevalence is {prevalence:.2%}. At the fixed 0.5 threshold, precision is "
                 f"{detection['detection_precision']:.2%}, recall {detection['detection_recall']:.2%}, F1 "
                 f"{detection['detection_f1']:.4f}, and balanced accuracy {detection['detection_balanced_accuracy']:.2%}. "
                 f"Accuracy is {detection['detection_accuracy']:.2%}, compared with {1-prevalence:.2%} for predicting every "
                 "entry negative. Most observed positive entries are missed at this threshold; high accuracy largely reflects class imbalance. "
                 "No threshold was tuned for this report.")
    matched = (f"Binary BCE minus hurdle detector BCE is {bce_delta:+.6f} (negative favors binary-only training). "
               "The split, preprocessing, initialization prevalence and fixed validation mask receipts match. The hurdle "
               "checkpoint was selected on joint hurdle-NB likelihood, while the binary checkpoint was selected on BCE; "
               "this comparison retains that selection asymmetry. Joint hurdle-NB NLL and binary BCE are different objectives "
               "and are not compared numerically. Hurdle AUROC, AUPRC and ECE are unavailable in its saved final artifacts. "
               "Hurdle accuracy, balanced accuracy, MCC and predicted-positive fraction are derived from saved confusion counts.")
    references = ("Constant-prevalence BCE, Brier and AUPRC references below use observed validation prevalence. They are "
                  "retrospective descriptive references, not training-fitted independent baselines. Epoch 1 already contains "
                  "six optimizer updates; it is not an untrained prevalence baseline.")
    runtime = machine["runtime"]
    interruption = (f"The original execution was interrupted after epoch 17 around 09:40 UTC on September 30; recorded host "
                    "and container restart observations support an environment interruption, whose initiating cause is unknown. "
                    "The same run resumed from its atomic checkpoint under supervisor at 17:13 UTC with model, optimizer, "
                    "scheduler and RNG state restored. Gradient-direction continuity at epoch 18 is unavailable. No epochs are "
                    f"missing or duplicated. Runner duration is {runtime['runner_duration_seconds']/3600:.2f} hours for the "
                    f"resumed process; the sum of completed epoch durations is {runtime['sum_completed_epoch_seconds']/3600:.2f} "
                    "hours across both segments. These are not total elapsed wall time: downtime and unrecoverable initial/partial-epoch "
                    f"work are excluded. Four RTX 3090 GPUs ran one DDP rank each; peak allocated VRAM was "
                    f"{runtime['peak_allocated_vram_gib']:.2f} GiB. Framework/CUDA and recovery receipts are retained in summary.json.")
    if runtime["wall_elapsed_including_interruption_seconds"] is not None:
        interruption += (f" Elapsed time from original launch through training completion, including the interruption, was "
                         f"{runtime['wall_elapsed_including_interruption_seconds']/3600:.2f} hours.")
    limits = ("The strongest alternatives are that observed same-cell genes, morphology and gene prevalence explain performance "
              "without useful spatial context, that segmentation spillover mimics neighborhood information, or that count "
              "supervision regularizes the hurdle detector. Target-isolated routing, "
              "training-only preprocessing, donor-grouped splitting, matching masks, and replay/coverage checks address "
              "specific leakage and comparability risks. They do not demonstrate useful graph-specific information. "
              "Graphless/self/context baselines, realistic spatial nulls, segmentation robustness, multiple seeds/folds, "
              "independent-patient replication and a sealed test remain absent. Cells and the two validation cores are not "
              "independent patient replicates; no patient-level uncertainty interval is justified. Maximum claim: this "
              "validation-selected run learns masked occurrence prediction and supports an exploratory matched comparison. "
              "No mechanism or causal claim follows. The next discriminating test is a locked comparison against train-fitted "
              "gene-prevalence and self/context models across independent donor-held-out folds.")
    validation = ("Both immutable input bundles passed verify_run_bundle. The best checkpoint checksum, strict minimum epoch, "
                  "replayed primary and secondary metrics (tolerance 1e-6), contiguous histories, finite gradients, once-per-target "
                  "coverage and fixed mask receipts were checked without re-running a model. Registry/checkpoint-catalog lifecycle "
                  "verification is performed separately by the monitoring task. Source SHA-256 records and all artifact hashes "
                  "are available in summary.json and report_manifest.json. No raw data or patient identifiers are exported.")
    comparisons = [[r["label"], f"{r['binary_bce']:.6f}", f"{r['hurdle_nb']:.6f}",
                    f"{r['binary_minus_hurdle']:+.6f}", r["preferred_direction"]] for r in comparison]
    extras = [[label, f"{detection[key]:.6f}"] for key, label in (
        ("detection_auroc_histogram_4096", "AUROC (4,096 bins)"), ("detection_auprc_histogram_4096", "AUPRC / average precision (4,096 bins)"),
        ("detection_expected_calibration_error_histogram_4096", "ECE (4,096 bins)"))]
    extras += [["Validation-prevalence BCE reference", f"{machine['retrospective_references']['bce']:.6f}"],
               ["Validation-prevalence Brier reference", f"{machine['retrospective_references']['brier']:.6f}"],
               ["Validation-prevalence AUPRC reference", f"{prevalence:.6f}"]]
    sections = [("Training and selected checkpoint", [lead, design]), ("Occurrence prediction", [imbalance]),
                ("Matched hurdle comparison", [matched]), ("Descriptive references", [references]),
                ("Interruption and resources", [interruption]), ("Interpretation and limits", [limits]),
                ("Verification and provenance", [validation])]
    md = ["# " + title]
    markup = ["<h1>" + title + "</h1>"]
    for index, (heading, paragraphs) in enumerate(sections):
        md.extend(["", "## " + heading, ""]); markup.append("<section><h2>" + heading + "</h2>")
        for paragraph in paragraphs:
            md.extend([paragraph, ""]); markup.append("<p>" + html.escape(paragraph) + "</p>")
        if index in (0, 1):
            name, caption = figures[index]
            encoded = base64.b64encode((output / name).read_bytes()).decode()
            md.extend([f"![{caption}]({name})", "", caption, ""])
            markup.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        if index == 2:
            headers = ["Metric", "Binary BCE", "Hurdle NB", "Binary − hurdle", "Preferred"]
            md.extend([table(headers, comparisons, html_mode=False), ""]); markup.append(table(headers, comparisons, html_mode=True))
        if index == 3:
            md.extend([table(["Metric/reference", "Value"], extras, html_mode=False), ""])
            markup.append(table(["Metric/reference", "Value"], extras, html_mode=True))
        markup.append("</section>")
    command = (f"PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_binary_expression_bce.py "
               f"--run-id {summary['run_id']} --comparator-id {machine['comparator_summary']['run_id']} --version {output.name}")
    md.extend(["## Reproduction", "", "Run from the repository root; choose a new version if the report already exists.", "", "```bash", command, "```", ""])
    markup.append("<section><h2>Reproduction</h2><p>Run from the repository root; choose a new version if the report already exists.</p><pre>" + html.escape(command) + "</pre></section>")
    (output / "report.md").write_text("\n".join(md), encoding="utf-8")
    style = "body{font:16px/1.55 system-ui,sans-serif;color:#16202b;background:#fff;max-width:1120px;margin:2rem auto;padding:0 1rem}h1{font-size:1.8rem}h2{font-size:1.3rem;margin-top:2rem}img{width:100%;height:auto}figure{margin:1.3rem 0}figcaption{color:#425466;font-size:.9rem}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{padding:.5rem;text-align:right;border-bottom:1px solid #c6ced6}th:first-child,td:first-child{text-align:left}thead{background:#eef2f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f5;padding:1rem}@media print{body{margin:0;max-width:none;font-size:10pt}figure,table{break-inside:avoid}h2{break-after:avoid}img{max-height:18cm;object-fit:contain}}"
    (output / "report.html").write_text("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>" + title + "</title><style>" + style + "</style><main>" + "".join(markup) + "</main></html>", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--comparator-id", default=DEFAULT_COMPARATOR)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    require(bool(re.fullmatch(r"v[1-9][0-9]*", args.version)), "Version must be v1, v2, ...")
    require(args.run_id != args.comparator_id, "Current and comparator must differ")
    output = current_paths().report_root / "analyses/so2_binary_expression_bce" / args.run_id / args.version
    require(not output.exists(), f"Report version already exists: {output}")
    current = load_bundle(args.run_id, PROVENANCE)
    hurdle = load_bundle(args.comparator_id, HURDLE_PROVENANCE)
    audit = audit_current(current)
    comparison = compare(current, hurdle)
    hist = histogram(current)
    detection = detection_metrics(current, binary=True)
    prevalence = detection["observed_positive_rate"]
    runtime = runtime_info(current)
    script = Path(__file__).resolve()
    machine = {"schema": "so2_binary_expression_bce_completion_report_v1",
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "analysis_status": "exploratory_designed_after_learning_curve_observation",
               "run_summary": current["summary"], "comparator_summary": hurdle["summary"],
               "audit": audit, "detection_metrics": detection, "comparison": comparison, "runtime": runtime,
               "comparison_checks": current["comparison_checks"],
               "retrospective_references": {"fit_scope": "observed_validation_prevalence_not_training_fitted_baseline",
                   "bce": -prevalence * math.log(prevalence) - (1-prevalence) * math.log(1-prevalence),
                   "brier": prevalence * (1-prevalence), "all_negative_accuracy": 1-prevalence,
                   "all_negative_balanced_accuracy": .5, "auroc": .5, "auprc": prevalence},
               "evidence": {key: "not_established" for key in ("graph_specific_predictive_gain", "seed_stability",
                   "faithfulness", "null_calibration", "patient_replication", "external_support", "perturbation")},
               "source_bundles": {args.run_id: {"path": str(current["root"]), "sources": current["sources"]},
                                  args.comparator_id: {"path": str(hurdle["root"]), "sources": hurdle["sources"]}},
               "builder": {"path": str(script), "snapshot": "report_builder.py", **record(script)},
               "report_environment": {"python": sys.version, "numpy": package_version("numpy"),
                                      "matplotlib": package_version("matplotlib")},
               "validation_core_metrics_at_selected_epoch": [{k: v for k, v in r.items() if not k.endswith("_json")}
                    for r in current["cores"] if r["split"] == "validation" and int(r["global_epoch"]) == current["summary"]["best_epoch"]],
               "comparison_limits": ["One validation donor group, one seed, no test split",
                   "Hurdle selected on joint NLL; binary selected on BCE", "No new inference or threshold selection",
                   "Hurdle AUROC/AUPRC/ECE unavailable", "Registry/checkpoint catalog verified separately by monitor"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "report_builder.py").write_bytes(script.read_bytes())
    figures = make_figures(output, current, hurdle, hist)
    write_reports(output, machine, figures)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
    (output / "summary.json").write_text(json.dumps(machine, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {"schema": "so2_binary_expression_bce_report_manifest_v1", "status": "complete",
                "run_id": args.run_id, "version": args.version,
                "files": {p.name: record(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "report_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output / "report.html"), "audit": audit}, sort_keys=True))


if __name__ == "__main__":
    main()
