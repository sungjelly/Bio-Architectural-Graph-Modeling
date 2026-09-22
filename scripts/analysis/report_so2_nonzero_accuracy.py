"""Verify and compare the four frozen SO2 nonzero-accuracy evaluations.

This CPU-only report consumes completed, checksummed aggregate receipts. It
does not evaluate checkpoints, read clinical data, or mutate the registry.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

from spatial_benchmark.paths import current_paths


MODELS = {
    "original": ("r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6", 175, "completed"),
    "continued": ("r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2", 300, "completed"),
    "recurrent": ("r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf", 175, "failed_artifact_finalization"),
    "geometry": ("r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf", 200, "completed"),
}
PREDICTORS = (*MODELS, "zero_count", "gene_mean")
LABELS = {
    "original": "Original e175", "continued": "Continued e300",
    "recurrent": "Recurrent e175*", "geometry": "Geometry e200",
    "zero_count": "Zero count", "gene_mean": "Gene mean",
}
CORES = tuple(range(15, 29))
STRATA = ("all", "zero", "positive", "count_1", "count_2", "count_3", "count_4_7", "count_8_plus")
ERRORS = ("mse_standardized", "mae_standardized", "huber_standardized", "bias_standardized", "mse_log1p", "mae_log1p")
RATES = ("positive_recall", "zero_specificity", "positive_precision", "balanced_accuracy")
PRIMARY = "positive/mse_standardized"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"Missing or linked file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_matches(path: Path, expected: Any) -> bool:
    record = file_record(path)
    if isinstance(expected, str):
        return record["sha256"] == expected
    return record["sha256"] == expected.get("sha256") and (
        "size_bytes" not in expected or record["size_bytes"] == expected["size_bytes"]
    )


def close(left: float, right: float, scale: float | None = None) -> bool:
    return abs(left - right) <= 1e-10 * max(abs(left), abs(right), abs(scale or 0), 1.0)


def numeric(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"Nonfinite/non-numeric {label}")
    return float(value)


def validate_metrics(result: dict[str, Any]) -> None:
    """Recompute support, means, detection and decomposition from saved sums."""
    require(result["schema"] == "so2_nonzero_metrics_v1", "Metric schema drift")
    require(result["huber_delta"] == 1.0, "Huber delta drift")
    strata = result["strata"]
    require(set(strata) == set(STRATA), "Count stratum drift")
    for name in STRATA:
        row = strata[name]
        n, exact = row["n"], row["n_exact_count"]
        require(isinstance(n, int) and n >= 0 and isinstance(exact, int) and 0 <= exact <= n, "Invalid stratum support")
        for metric in ERRORS:
            total = numeric(row["sums"][metric], metric)
            if metric != "bias_standardized":
                require(total >= 0, "Negative error sum")
            if n:
                require(close(numeric(row[metric], metric), total / n), f"Mean/sum mismatch: {name}/{metric}")
            else:
                require(row[metric] is None and total == 0, "Empty stratum is not explicit")
        require(row["exact_count_accuracy"] is None if not n else close(row["exact_count_accuracy"], exact / n), "Exact-count rate mismatch")
    require(strata["all"]["n"] > 0, "Empty evaluation")
    for total_name, parts in (("all", ("zero", "positive")), ("positive", STRATA[3:])):
        total = strata[total_name]
        for field in ("n", "n_exact_count"):
            require(sum(strata[s][field] for s in parts) == total[field], f"{field} partition mismatch")
        for metric in ERRORS:
            require(close(sum(strata[s]["sums"][metric] for s in parts), total["sums"][metric], total["sums"]["mae_standardized"]), "Metric partition mismatch")
    det = result["detection"]
    require(det["continuous_count_threshold"] == 0.5, "Detection threshold drift")
    tp, fp, tn, fn = [det[key + "_count"] for key in ("true_positive", "false_positive", "true_negative", "false_negative")]
    require(all(isinstance(v, int) and v >= 0 for v in (tp, fp, tn, fn)), "Invalid detection counts")
    require(tp + fn == strata["positive"]["n"] and fp + tn == strata["zero"]["n"], "Detection support mismatch")
    ratios = {"positive_recall": (tp, tp + fn), "zero_specificity": (tn, tn + fp), "positive_precision": (tp, tp + fp)}
    for key, (numerator, denominator) in ratios.items():
        require(det[key] is None if not denominator else close(det[key], numerator / denominator), "Detection rate mismatch")
    balanced = None if not (tp + fn and tn + fp) else 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    require(det["balanced_accuracy"] is None if balanced is None else close(det["balanced_accuracy"], balanced), "Balanced-accuracy mismatch")
    dec = result["decomposition"]
    require(dec["support_partition_verified"] and dec["metric_sum_partition_verified"], "Unverified receipt decomposition")
    sse = strata["all"]["sums"]["mse_standardized"]
    require(close(dec["all_squared_error_standardized"], sse), "Decomposition total mismatch")
    require(close(dec["zero_plus_positive_squared_error_standardized"], sse), "Decomposition partition mismatch")
    require(close(dec["positive_count_bins_squared_error_standardized"], strata["positive"]["sums"]["mse_standardized"]), "Decomposition bins mismatch")
    require(close(dec["zero_positive_mse_residual"], 0), "Nonzero MSE decomposition residual")


def verify_input_bundle(root: Path, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("source_run_id", manifest.get("run_id")) == run_id, "Evaluation manifest run mismatch")
    marker = root / "_SUCCESS"
    completion = read_json(marker)
    require(completion.get("status") == "success", "Evaluation is not completed")
    require(completion.get("manifest_sha256") == file_record(manifest_path)["sha256"], "Evaluation completion marker mismatch")
    require(manifest.get("cores") == list(CORES), "Evaluation core coverage drift")
    for hash_key in ("manifest_content_sha256", "content_sha256"):
        if hash_key in manifest:
            payload = {k: v for k, v in manifest.items() if k != hash_key}
            require(hashlib.sha256(canonical(payload)).hexdigest() == manifest[hash_key], "Evaluation manifest self-hash mismatch")
    inventory = manifest.get("files", {})
    require(isinstance(inventory, dict) and inventory, "Missing evaluation file inventory")
    sources = {}
    for relative, expected in inventory.items():
        path = root / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe manifest path")
        require(file_matches(path, expected), f"Evaluation artifact changed: {path}")
        sources[str(path.resolve())] = file_record(path)
    for core in CORES:
        require(f"core_{core}.json" in inventory, f"Core {core} missing from manifest")
    sources[str(manifest_path.resolve())] = file_record(manifest_path)
    sources[str(marker.resolve())] = file_record(marker)
    return manifest, sources


def load_inputs(input_root: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_predictor: dict[str, list[dict[str, Any]]] = {p: [] for p in PREDICTORS}
    provenance: dict[str, Any] = {"files": {}, "models": {}}
    anchors = {}
    for model, (run_id, epoch, source_status) in MODELS.items():
        root = input_root / run_id / "v1"
        manifest, sources = verify_input_bundle(root, run_id)
        provenance["files"].update(sources)
        provenance["models"][model] = {"source_run_id": run_id, "epoch": epoch, "source_training_status": source_status, "evaluation_manifest": str(root / "manifest.json"), "manifest": manifest}
        cells = 0
        for core in CORES:
            receipt = read_json(root / f"core_{core}.json")
            require(receipt["model_name"] == model and receipt["source_run_id"] == run_id, "Receipt model identity mismatch")
            require(receipt["core_alias"] == f"SO2-C{core}", "Receipt core mismatch")
            require(receipt["source_epoch"] == epoch, "Source checkpoint epoch mismatch")
            require(receipt["source_status"] == ("failed" if model == "recurrent" else "completed"), "Source lifecycle status drift")
            require(receipt["checkpoint_sha256"] == manifest["checkpoint_sha256"], "Checkpoint identity drift")
            require(receipt["evaluation_id"] == manifest["evaluation_id"], "Evaluation identity drift")
            require(receipt["replay_verified"] and all(numeric(v, "replay difference") <= 2e-6 for v in receipt["replay_absolute_differences"].values()), "Original all-entry diagnostic replay failed")
            cells += receipt["n_cells"]
            require(set(receipt["metrics"]) == {"model", "zero_count", "gene_mean"}, "Predictor set drift")
            for metric in receipt["metrics"].values():
                validate_metrics(metric)
            signature = {"n_cells": receipt["n_cells"], "mask_checksum": receipt["mask_checksum"], "mask_seed": receipt["mask_seed"], "source_files": receipt["source_files"], "supports": {s: receipt["metrics"]["model"]["strata"][s]["n"] for s in STRATA}}
            require(receipt["n_masked_entries"] == signature["supports"]["all"], "Original diagnostic mask support mismatch")
            for name, metric in receipt["metrics"].items():
                require(signature["supports"] == {s: metric["strata"][s]["n"] for s in STRATA}, f"Baseline support mismatch: {name}")
            if model == "original":
                anchors[core] = {"signature": signature, "baselines": {p: receipt["metrics"][p] for p in ("zero_count", "gene_mean")}}
                for predictor in ("zero_count", "gene_mean"):
                    by_predictor[predictor].append({"core_alias": receipt["core_alias"], "metrics": receipt["metrics"][predictor]})
            else:
                require(signature == anchors[core]["signature"], "Masks, cells, or observed-count supports differ across models")
                for predictor in ("zero_count", "gene_mean"):
                    require(receipt["metrics"][predictor] == anchors[core]["baselines"][predictor], f"Shared baseline differs across models: {predictor}")
            by_predictor[model].append({"core_alias": receipt["core_alias"], "metrics": receipt["metrics"]["model"]})
        require(cells == 246063, "Total cell count drift")
    provenance["support_by_core"] = {f"SO2-C{core}": anchors[core]["signature"] for core in CORES}
    return by_predictor, provenance


def average(values: list[Any]) -> dict[str, Any]:
    present = [numeric(v, "aggregated metric") for v in values if v is not None]
    return {"mean": sum(present) / len(present) if present else None, "n_cores_supported": len(present), "n_cores_expected": len(values)}


def flatten(result: dict[str, Any]) -> dict[str, Any]:
    flat = {}
    for stratum in STRATA:
        row = result["strata"][stratum]
        flat[stratum + "/support"] = row["n"]
        for key in (*ERRORS, "exact_count_accuracy"):
            flat[stratum + "/" + key] = row[key]
    for key in RATES:
        flat["detection/" + key] = result["detection"][key]
    n = result["strata"]["all"]["n"]
    for stratum in ("zero", "positive"):
        flat[f"decomposition/{stratum}_mse_contribution"] = result["strata"][stratum]["sums"]["mse_standardized"] / n
        flat[f"decomposition/{stratum}_support_fraction"] = result["strata"][stratum]["n"] / n
    return flat


def aggregate(by_predictor: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Average complete per-core metrics; never pool entries across cores."""
    summaries = {}
    flattened = {p: [flatten(row["metrics"]) for row in rows] for p, rows in by_predictor.items()}
    for predictor, rows in flattened.items():
        values = {key: average([row[key] for row in rows]) for key in rows[0] if not key.endswith("/support")}
        supports = {s: sum(row[s + "/support"] for row in rows) for s in STRATA}
        require(close(values["all/mse_standardized"]["mean"], values["decomposition/zero_mse_contribution"]["mean"] + values["decomposition/positive_mse_contribution"]["mean"]), "Equal-core decomposition failed")
        summaries[predictor] = {"metrics": values, "support_totals": supports}
    comparisons = []
    comparison_metrics = (PRIMARY, "positive/mae_standardized", "positive/exact_count_accuracy", "zero/mse_standardized", "all/mse_standardized", "detection/balanced_accuracy")
    for reference in ("original", "continued"):
        for candidate in MODELS:
            if candidate == reference:
                continue
            for metric in comparison_metrics:
                direction = "higher" if "accuracy" in metric else "lower"
                deltas = []
                for i, core in enumerate(CORES):
                    c, r = flattened[candidate][i][metric], flattened[reference][i][metric]
                    delta = c - r if c is not None and r is not None else None
                    deltas.append({"core_alias": f"SO2-C{core}", "candidate": c, "reference": r, "difference_candidate_minus_reference": delta, "relative_change_percent": 100 * delta / r if delta is not None and r != 0 else None})
                valid = [r for r in deltas if r["difference_candidate_minus_reference"] is not None]
                comparisons.append({"candidate": candidate, "reference": reference, "metric": metric, "improvement_direction": direction, "mean_paired_difference": average([r["difference_candidate_minus_reference"] for r in deltas]), "favorable_core_count": sum(r["difference_candidate_minus_reference"] < 0 if direction == "lower" else r["difference_candidate_minus_reference"] > 0 for r in valid), "tie_core_count": sum(r["difference_candidate_minus_reference"] == 0 for r in valid), "per_core": deltas})
    return {"schema": "so2_nonzero_comparison_v1", "status": "complete", "exploratory": True, "primary_metric": PRIMARY, "aggregation": "arithmetic mean of per-core metrics; each core receives equal weight; unsupported metrics are null and supported-core counts are explicit", "n_cores": 14, "n_cells": 246063, "model_seeds": [0], "models": {p: {"source_run_id": spec[0], "epoch": spec[1], "source_training_status": spec[2]} for p, spec in MODELS.items()}, "excluded_runs": [{"run_id": "r_20260826T032515Z_52d16093_s000_f00_a01_149a165b", "reason": "failed duplicate continuation; not an independent model seed"}], "predictors": summaries, "paired_comparisons": comparisons, "uncertainty": "No formal confidence intervals or significance tests; cores are fitted descriptive strata, and all models have seed 0.", "maximum_claim": "Exploratory fitted-cohort masked observed-count reconstruction comparison; no graph-specific gain, generalization, mechanism, or causal claim."}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tables(root: Path, inputs: dict[str, Any], summary: dict[str, Any]) -> None:
    model_rows, core_rows, bin_rows, paired_rows = [], [], [], []
    for predictor in PREDICTORS:
        source = {
            "source_run_id": MODELS[predictor][0] if predictor in MODELS else "shared_descriptive_reference",
            "source_status": ("failed" if predictor == "recurrent" else "completed") if predictor in MODELS else "all_fit_descriptive_reference",
            "source_failure_category": "artifact_finalization_failure" if predictor == "recurrent" else None,
            "source_epoch": MODELS[predictor][1] if predictor in MODELS else None,
            "evaluation_id": "ev_so2_nonzero_v1_" + MODELS[predictor][0] if predictor in MODELS else None,
            "baseline_receipt_source_run_id": None if predictor in MODELS else MODELS["original"][0],
        }
        row = {"predictor": predictor, "label": LABELS[predictor], **source}
        for key, value in summary["predictors"][predictor]["metrics"].items():
            row[key] = value["mean"]
            row[key + "/n_cores_supported"] = value["n_cores_supported"]
        row.update({s + "/support_total": n for s, n in summary["predictors"][predictor]["support_totals"].items()})
        model_rows.append(row)
        for core in inputs[predictor]:
            core_rows.append({"predictor": predictor, **source, "core_alias": core["core_alias"], **flatten(core["metrics"])})
            for stratum in STRATA[3:]:
                data = core["metrics"]["strata"][stratum]
                bin_rows.append({"predictor": predictor, **source, "core_alias": core["core_alias"], "count_bin": stratum, "support": data["n"], **{k: data[k] for k in (*ERRORS, "exact_count_accuracy")}})
        for stratum in STRATA[3:]:
            values = summary["predictors"][predictor]
            bin_rows.append({"predictor": predictor, **source, "core_alias": "equal_core_mean", "count_bin": stratum, "support": values["support_totals"][stratum], **{k: values["metrics"][stratum + "/" + k]["mean"] for k in (*ERRORS, "exact_count_accuracy")}, "n_cores_supported": values["metrics"][stratum + "/mse_standardized"]["n_cores_supported"]})
    for contrast in summary["paired_comparisons"]:
        for row in contrast["per_core"]:
            provenance = {}
            for role in ("candidate", "reference"):
                run_id, epoch, status = MODELS[contrast[role]]
                provenance.update({role + "_source_run_id": run_id, role + "_source_status": "failed" if contrast[role] == "recurrent" else status, role + "_source_epoch": epoch, role + "_source_failure_category": "artifact_finalization_failure" if contrast[role] == "recurrent" else None})
            paired_rows.append({"candidate_model": contrast["candidate"], "reference_model": contrast["reference"], **provenance, "metric": contrast["metric"], "improvement_direction": contrast["improvement_direction"], **row})
    write_csv(root / "modelsummary.csv", model_rows)
    write_csv(root / "percore.csv", core_rows)
    write_csv(root / "countbins.csv", bin_rows)
    write_csv(root / "paired_core_deltas.csv", paired_rows)


def mean(summary: dict[str, Any], predictor: str, metric: str) -> Any:
    return summary["predictors"][predictor]["metrics"][metric]["mean"]


def display(value: Any, percent: bool = False) -> str:
    if value is None:
        return "undefined"
    return f"{value * 100:.2f}%" if percent else f"{value:.6f}"


def render_report(root: Path, summary: dict[str, Any]) -> None:
    geo = mean(summary, "geometry", PRIMARY)
    lines = ["# SO2 masked nonzero reconstruction", "", "Exploratory evaluation of four frozen seed-0 endpoints on the same fixed masks across all 14 fitted cores (246,063 cells). Lower error is better. Each core receives equal weight.", "", "![Nonzero accuracy, detection, error decomposition, and paired core differences](accuracy_comparison.png)", "", "The figure shows positive-entry errors, supplementary rounded count accuracy, detection tradeoffs, and each stratum's contribution to total squared error. Grey bars are the shared descriptive references.", "", "| Predictor | Positive standardized MSE | Positive standardized MAE | Positive exact count | Positive detection recall | Zero specificity | Detection balanced accuracy |", "|---|---:|---:|---:|---:|---:|---:|"]
    for predictor in PREDICTORS:
        keys = (PRIMARY, "positive/mae_standardized", "positive/exact_count_accuracy", "detection/positive_recall", "detection/zero_specificity", "detection/balanced_accuracy")
        lines.append("| " + LABELS[predictor] + " | " + " | ".join(display(mean(summary, predictor, k), i >= 2) for i, k in enumerate(keys)) + " |")
    winner = min(MODELS, key=lambda p: mean(summary, p, PRIMARY))
    lines += ["", f"**{LABELS[winner]} has the lowest positive-entry standardized MSE among these four endpoints.** Nonzero means observed raw count > 0. Exact count accuracy requires the decoded integer to match the observed count; detection only asks whether the decoded count is positive.", "", f"Observed zeros account for **{100 * mean(summary, 'zero_count', 'decomposition/zero_support_fraction'):.2f}%** of masked entries on an equal-core basis. Consequently, always predicting zero achieves that same overall exact-count accuracy. The learned models' overall exact-count accuracy ranges from **{100 * min(mean(summary, p, 'all/exact_count_accuracy') for p in MODELS):.2f}% to {100 * max(mean(summary, p, 'all/exact_count_accuracy') for p in MODELS):.2f}%**. This makes overall exact accuracy a poor standalone measure of reconstruction quality; the models were trained for continuous Huber loss, not integer classification."]
    lines += ["", "## Sensitivity to the error scale", "", "Gene standardization weights squared log errors by the inverse of each gene's normalization variance. The unstandardized log1p metric therefore answers a differently weighted reconstruction question. Keep both scales visible when comparing with the target-derived gene-mean reference.", "", "| Predictor | Positive standardized MSE | Positive log1p MSE |", "|---|---:|---:|"]
    for predictor in PREDICTORS:
        lines.append("| " + LABELS[predictor] + " | " + display(mean(summary, predictor, PRIMARY)) + " | " + display(mean(summary, predictor, "positive/mse_log1p")) + " |")
    lines += ["", "## Paired positive-error comparisons", ""]
    for reference in ("original", "continued"):
        prior = mean(summary, reference, PRIMARY)
        contrast = next(r for r in summary["paired_comparisons"] if r["candidate"] == "geometry" and r["reference"] == reference and r["metric"] == PRIMARY)
        delta = geo - prior
        rel = f" ({100 * delta / prior:+.3f}%)" if prior else ""
        lines.append(f"Geometry minus {LABELS[reference]} positive standardized MSE is **{delta:+.6f}{rel}**; geometry has lower MSE in **{contrast['favorable_core_count']}/14 cores**. These are descriptive paired differences, without a post-hoc superiority threshold or significance claim.")
        lines.append("")
    lines += ["## Zero versus positive errors", "", "Each core's all-entry MSE equals its zero squared-error sum divided by all masked entries plus its positive squared-error sum divided by all masked entries. Averaging those contributions across cores preserves the exact decomposition; weighting equal-core stratum means by a pooled support fraction would not.", "", "| Predictor | All MSE | Zero MSE | Positive contribution to all MSE | Zero contribution to all MSE | Positive signed error |", "|---|---:|---:|---:|---:|---:|"]
    for predictor in PREDICTORS:
        keys = ("all/mse_standardized", "zero/mse_standardized", "decomposition/positive_mse_contribution", "decomposition/zero_mse_contribution", "positive/bias_standardized")
        lines.append("| " + LABELS[predictor] + " | " + " | ".join(display(mean(summary, predictor, k)) for k in keys) + " |")
    lines += ["", "Signed error is prediction minus target; negative positive-entry bias describes underprediction on the standardized log1p scale. Detailed count bins (1, 2, 3, 4–7, 8+) and errors on the unstandardized log1p scale are in `countbins.csv` and `percore.csv`.", "", "## Scope and limitations", "", "- The estimand is partial-gene reconstruction of fitted cells. Observed same-cell genes and permitted morphology remain available. These results do not measure patient-held-out prediction or isolate graph use.", "- Every inclusion mask uses observed raw count, not standardized target sign. Observed positive counts are not latent ground-truth expression; zero counts can reflect nondetection.", "- Original e175 and continued e300 are one training lineage. Recurrent ends at e175 and geometry at e200. Training duration and the complete attention-score change limit attribution to geometry modulation.", "- *Recurrent training and checkpoint reload completed, but its original bundle has failed artifact-finalization status. That source status is retained. The failed duplicate continuation is excluded rather than counted as another seed.", "- Zero count and shared equal-core gene-mean log1p are all-fit descriptive references. The gene mean is target-derived and is not a leakage-free held-out baseline.", "- Rounded-count diagnostics use fixed half-up decoding and a 0.5 continuous-count detection threshold. Positive recall, zero specificity, precision, and balanced accuracy must be interpreted together. Undefined precision is retained as null.", "- Only seed 0 is available. Core variation is descriptive; no formal confidence interval, significance test, graph-specific gain, biological mechanism, or causal claim is supported.", "", "## Files and verification", "", "`modelsummary.csv` contains equal-core means and supported-core counts. `percore.csv` retains every core. `countbins.csv` separates observed count ranges. `paired_core_deltas.csv` preserves every contrast against original and continued models. `source_references.json`, `verification.json`, and `manifest.json` bind all source and report files by checksum.", "", "```bash", "PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_nonzero_accuracy.py --check", "```", ""]
    (root / "report.md").write_text("\n".join(lines))


def plot(root: Path, summary: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), constrained_layout=True)
    names = list(PREDICTORS)
    labels = [LABELS[p].replace(" ", "\n", 1) for p in names]
    colors = ["#4477AA", "#228833", "#AA3377", "#EE7733", "#999999", "#BBBBBB"]
    x = np.arange(len(names))
    specs = [(axes[0, 0], PRIMARY, "Positive-entry standardized MSE ↓", 1), (axes[0, 1], "positive/mae_standardized", "Positive-entry standardized MAE ↓", 1), (axes[0, 2], "positive/exact_count_accuracy", "Positive exact rounded count ↑", 100)]
    for ax, metric, title, factor in specs:
        values = [mean(summary, p, metric) for p in names]
        displayed = [v * factor if v is not None else float("nan") for v in values]
        bars = ax.bar(x, displayed, color=colors)
        ax.bar_label(
            bars,
            labels=[("undefined" if v is None else f"{v * factor:.2f}%" if factor == 100 else f"{v:.3f}") for v in values],
            padding=3,
            fontsize=9,
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        finite_values = [v for v in displayed if math.isfinite(v)]
        maximum = max(finite_values, default=0.0)
        ax.set_ylim(0, maximum * 1.18 if maximum > 0 else 1.0)
        if factor == 100:
            ax.set_ylabel("Percent")
    ax = axes[1, 0]
    for shift, metric, label, color in ((-0.18, "zero_specificity", "Zero specificity", "#4477AA"), (0.18, "balanced_accuracy", "Detection balanced accuracy", "#EE7733")):
        ax.bar(x + shift, [100 * mean(summary, p, "detection/" + metric) for p in names], width=0.36, label=label, color=color)
    ax.set_title("Detection at fixed count threshold 0.5 ↑", loc="left", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1, 1]
    zero = [mean(summary, p, "decomposition/zero_mse_contribution") for p in names]
    positive = [mean(summary, p, "decomposition/positive_mse_contribution") for p in names]
    ax.bar(x, zero, color="#BBBBBB", label="Zero-entry contribution")
    ax.bar(x, positive, bottom=zero, color="#EE7733", label="Positive-entry contribution")
    ax.set_title("Contribution to all-entry standardized MSE", loc="left", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    for reference, color in (("original", "#4477AA"), ("continued", "#228833")):
        contrast = next(r for r in summary["paired_comparisons"] if r["candidate"] == "geometry" and r["reference"] == reference and r["metric"] == PRIMARY)
        ax.plot(CORES, [r["difference_candidate_minus_reference"] for r in contrast["per_core"]], marker="o", markersize=4, label=f"Geometry − {LABELS[reference]}", color=color)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Positive MSE difference by fitted core", loc="left", fontweight="bold")
    ax.set_xlabel("SO2 core; below zero favors geometry")
    ax.set_xticks(CORES)
    ax.legend(fontsize=8)
    fig.suptitle("SO2 nonzero reconstruction: four frozen seed-0 endpoints\nEqual-core descriptive metrics; fitted cells, fixed masks; *recurrent source finalization failed", fontsize=15, fontweight="bold")
    fig.savefig(root / "accuracy_comparison.png", dpi=180)
    fig.savefig(root / "accuracy_comparison.pdf")
    plt.close(fig)


def verify_report(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    require(manifest["schema"] == "so2_nonzero_comparison_manifest_v1" and manifest["status"] == "complete", "Wrong/incomplete report manifest")
    require((root / "_SUCCESS").read_text().strip() == file_record(root / "manifest.json")["sha256"], "Report completion marker mismatch")
    for relative, record in manifest["files"].items():
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe report manifest path")
        require(file_matches(root / relative, record), f"Report artifact changed: {relative}")
    sources = read_json(root / "source_references.json")
    for source, record in sources["files"].items():
        require(file_matches(Path(source), record), f"Source evaluation changed: {source}")
    summary = read_json(root / "summary.json")
    require(summary["n_cores"] == 14 and set(summary["predictors"]) == set(PREDICTORS), "Report coverage drift")
    return {"valid": True, "files_verified": len(manifest["files"]), "source_files_verified": len(sources["files"]), "report": str(root / "report.md")}


def main() -> None:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=paths.report_root / "analyses/so2_nonzero_metrics")
    parser.add_argument("--check", action="store_true", help="Verify published report and all referenced source evaluation files.")
    args = parser.parse_args()
    final = args.input_root / "comparison/v1"
    if args.check or final.exists():
        print(json.dumps(verify_report(final), sort_keys=True))
        return
    inputs, provenance = load_inputs(args.input_root)
    stage = paths.scratch_root / "active_runs" / MODELS["geometry"][0] / "posthoc_reports/so2_nonzero_metrics/comparison/v1"
    require(not stage.exists(), f"Incomplete report staging already exists; preserve and inspect it before a new attempt: {stage}")
    stage.mkdir(parents=True)
    write_json(stage / "config.resolved.json", {"protocol": "so2_nonzero_comparison_v1", "models": MODELS, "strata": STRATA, "primary_metric": PRIMARY, "aggregation": "equal_core_mean", "formal_inference": False, "input_root": str(args.input_root.resolve())})
    shutil.copyfile(Path(__file__), stage / "report_source.py")
    summary = aggregate(inputs)
    write_json(stage / "source_references.json", provenance)
    write_json(stage / "summary.json", summary)
    tables(stage, inputs, summary)
    render_report(stage, summary)
    plot(stage, summary)
    verification = {"status": "passed", "source_evaluation_count": 4, "core_receipt_count": 56, "source_manifest_inventory_verified": True, "fixed_masks_and_observed_count_supports_match": True, "shared_baselines_equal_across_models": True, "per_core_sums_means_detection_and_decomposition_recomputed": True, "equal_core_mse_decomposition_verified": True, "row_level_predictions_exported": False, "formal_inference_performed": False}
    write_json(stage / "verification.json", verification)
    manifest = {"schema": "so2_nonzero_comparison_manifest_v1", "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(), "files": {str(p.relative_to(stage)): file_record(p) for p in sorted(stage.rglob("*")) if p.is_file()}, "source_evaluation_ids": [p["manifest"]["evaluation_id"] for p in provenance["models"].values()], "source_run_ids": [r[0] for r in MODELS.values()]}
    write_json(stage / "manifest.json", manifest)
    (stage / "_SUCCESS").write_text(file_record(stage / "manifest.json")["sha256"] + "\n")
    verify_report(stage)
    final.parent.mkdir(parents=True, exist_ok=True)
    require(not final.exists(), "Completed report destination exists")
    os.rename(stage, final)
    print(json.dumps(verify_report(final), sort_keys=True))


if __name__ == "__main__":
    main()
