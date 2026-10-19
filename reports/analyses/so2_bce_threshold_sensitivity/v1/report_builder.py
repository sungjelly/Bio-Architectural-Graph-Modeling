#!/usr/bin/env python3
"""Exploratory threshold sensitivity from retained aggregate detection bins.

Task contract: diagnose whether the selected-best model's operating point
contributes to low recall/F1, using only retained 4096-bin validation histograms.
This plan was written after inspecting historical scores. The alternative is
that discrimination limits the attainable precision/recall tradeoff. The unit
is a masked entry nested in the one validation donor; four runs represent two
optimization seeds of two variants, not independent biological replication.

Fixed exploratory cutoffs declared in this plan are 0.1, 0.2, 0.3 and 0.5. No
threshold is optimized, selected or used to change the primary experiment.
Always-positive and always-negative count-derived references provide post-hoc
F1 context; they are not additional model thresholds or fitted baselines.
Metrics are precision, recall, F1 and predicted-positive frequency. Report
exact metrics at the next higher histogram-aligned cutoff and conservative
histogram discretization bounds at each requested cutoff. These bounds are
not statistical confidence intervals. The 0.5 boundary must reproduce saved confusion
counts exactly. Stop on incomplete confirmation coverage, source/checksum
drift, invalid support or mismatched 0.5 counts. No inference or training runs.

Maximum claim: operating-point sensitivity of selected-best predictions on
the reused validation split. This does not change the terminal-epoch primary
comparison, demonstrate an optimization gain, or provide independent threshold
validation. All four successful confirmation runs are required.

Run from the source repository with BAGM_* overrides as appropriate::

    PYTHONPATH=src:. /venv/main/bin/python scripts/analysis/report_so2_bce_threshold_sensitivity.py \
      --parent-report <final-confirmation-summary.json> --version v1

Creates a new immutable directory under reports/analyses/
so2_bce_threshold_sensitivity/<version> with summary/CSV/Markdown, an embedded
single-file HTML report, PNG and checksums. Existing versions fail closed.
Test with tests/unit/spatial_benchmark/test_so2_bce_threshold_sensitivity.py.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import RunArchive, _rename_no_replace, verify_run_bundle

THRESHOLDS = (.1, .2, .3, .5)
BIN_COUNT = 4096
HISTOGRAM_PATH = "results/detection_curve_histogram_4096.json"
CAMPAIGN = "cmp_20261001_so2_bce_optimization_audit"
METRICS = ("precision", "recall", "f1", "predicted_positive_rate")


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.


def metrics(tp: int, fp: int, positives: int, negatives: int) -> dict[str, Any]:
    return {"true_positive": tp, "false_positive": fp, "false_negative": positives-tp,
        "true_negative": negatives-fp, "precision": ratio(tp, tp+fp), "recall": ratio(tp, positives),
        "f1": ratio(2*tp, positives+tp+fp), "predicted_positive_rate": ratio(tp+fp, positives+negatives)}


def constant_classifier_references(positives: int, masked_entries: int) -> dict[str, Any]:
    """Exact hard-label references from observed counts, with explicit zero conventions."""
    require(type(positives) is int and type(masked_entries) is int
            and masked_entries > 0 and 0 <= positives <= masked_entries,
            "Reference support must be nonempty integer counts with 0 <= positives <= total")
    negatives = masked_entries-positives
    return {"positive_entries": positives, "negative_entries": negatives, "masked_entries": masked_entries,
        "observed_positive_rate": positives/masked_entries,
        "constant_score_average_precision": positives/masked_entries,
        "interpretation": "Retrospective count-derived context on validation outcomes; not a training-fitted or independent baseline and not used for selection",
        "undefined_rate_convention": "Precision with no predicted positives and recall with no observed positives are reported as zero, with defined flags retained",
        "classifiers": [
            {"classifier": "always_positive", **metrics(positives, negatives, positives, negatives),
                "precision_defined": True, "recall_defined": positives > 0},
            {"classifier": "always_negative", **metrics(0, 0, positives, negatives),
                "precision_defined": False, "recall_defined": positives > 0}]}


def threshold_rows(histogram: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive exact aligned operating points and conservative nominal bounds."""
    require(histogram.get("schema") == "binary_expression_detection_curve_histogram_v1"
            and histogram.get("bin_count") == BIN_COUNT and histogram.get("bin_width") == 1/BIN_COUNT
            and histogram.get("bin_index_definition") == "floor(nonzero_probability * bin_count), clamped to [0, bin_count - 1]",
            "Histogram schema or binning definition differs")
    positive = histogram["positive_counts_by_ascending_probability_bin"]
    negative = histogram["negative_counts_by_ascending_probability_bin"]
    require(isinstance(positive, list) and isinstance(negative, list)
            and len(positive) == len(negative) == BIN_COUNT
            and all(type(x) is int and x >= 0 for x in positive+negative), "Histogram counts must be nonnegative integers")
    p, n = sum(positive), sum(negative)
    require(p > 0 and n > 0 and p == histogram["positive_entries"] and p+n == histogram["masked_entries"],
            "Histogram support differs or lacks a class")
    result = []
    for threshold in THRESHOLDS:
        low, high = math.floor(threshold*BIN_COUNT), math.ceil(threshold*BIN_COUNT)
        tp_min, fp_min = sum(positive[high:]), sum(negative[high:])
        tp_max, fp_max = sum(positive[low:]), sum(negative[low:])
        point = metrics(tp_min, fp_min, p, n)
        # Within the ambiguous bin the unknown positive/negative allocations
        # can vary independently. Precision/F1 extrema are opposite corners;
        # the two observed aligned endpoint metrics alone are not bounds.
        bounds = {"precision": [ratio(tp_min, tp_min+fp_max), ratio(tp_max, tp_max+fp_min)],
            "recall": [tp_min/p, tp_max/p],
            "f1": [ratio(2*tp_min, p+tp_min+fp_max), ratio(2*tp_max, p+tp_max+fp_min)],
            "predicted_positive_rate": [(tp_min+fp_min)/(p+n), (tp_max+fp_max)/(p+n)]}
        result.append({"requested_threshold": threshold, "reported_aligned_threshold": high/BIN_COUNT,
            "lower_aligned_threshold": low/BIN_COUNT, "exact_at_requested_threshold": low == high,
            "alignment": "next_higher_bin_boundary", "ambiguous_bin_positive_entries": tp_max-tp_min,
            "ambiguous_bin_negative_entries": fp_max-fp_min, "positive_entries": p, "masked_entries": p+n,
            **point, "requested_threshold_bounds": bounds})
    return result


def check_primary_metrics(rows: Sequence[Mapping[str, Any]], final: Mapping[str, Any]) -> None:
    row, = [r for r in rows if r["requested_threshold"] == .5]
    prefix = "val/unseen_donor/"
    require(final[prefix+"detection_threshold"] == .5, "Primary threshold differs from 0.5")
    for key in ("true_positive", "true_negative", "false_positive", "false_negative"):
        expected = final[prefix+"detection_"+key]
        require(type(expected) is int and row[key] == expected, "Histogram disagrees with exact saved 0.5 confusion counts")
    for key in ("precision", "recall", "f1"):
        require(math.isclose(row[key], float(final[prefix+"detection_"+key]), rel_tol=0, abs_tol=1e-12),
                "Histogram disagrees with primary saved rate")
    require(math.isclose(row["predicted_positive_rate"], float(final[prefix+"predicted_positive_class_rate"]),
                        rel_tol=0, abs_tol=1e-12), "Histogram disagrees with saved predicted-positive frequency")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_sources(parent_path: Path, paths: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    parent_path = parent_path.resolve(strict=True)
    require(parent_path.name == "summary.json" and parent_path.is_relative_to(paths.report_root.resolve()),
            "Parent must be a finalized report summary under BAGM_REPORT_ROOT")
    manifest_path = parent_path.parent/"manifest.json"
    manifest = read_json(manifest_path)
    require("summary.json" in manifest["files"], "Parent manifest omits summary")
    for relative, digest in manifest["files"].items():
        source = (parent_path.parent/relative).resolve(strict=True)
        require(source.is_relative_to(parent_path.parent) and sha256_file(source) == digest, "Parent report checksum mismatch")
    parent = read_json(parent_path)
    require(parent.get("schema") == "so2_bce_optimization_audit_report_v1" and parent.get("campaign_id") == CAMPAIGN
            and parent.get("stage") == "confirmation" and parent.get("complete") is True
            and parent.get("required_coverage_complete") is True, "Final confirmation report is incomplete")
    ids = parent["stage_run_ids"]
    require(len(ids) == len(set(ids)) == 4, "Threshold comparison requires all four successful confirmation runs")
    chosen = [r for r in parent["completed_runs"] if r["run_id"] in ids]
    require(len(chosen) == 4 and all(r["profile"] == "confirmation" and r["final_epoch"] == 32 for r in chosen),
            "Smoke/screen predictions cannot substitute for confirmation")
    setting = lambda r: (r["batch_mode"], r["target_batch_size"], r["learning_rate"], r["dropout"], r["weight_decay"])
    variants = {setting(r) for r in chosen}
    require(len(variants) == 2 and {(setting(r), r["seed"]) for r in chosen} == {(v, s) for v in variants for s in (0, 1)},
            "Two variants with both seeds are required")
    incumbent = ("target_window", 1024, 1e-4, .1, 1e-5)
    require(incumbent in variants, "Matched incumbent is absent")
    provenance = [{"root": "report", "path": p.relative_to(paths.report_root).as_posix(), "sha256": sha256_file(p)}
                  for p in (parent_path, manifest_path)]
    registry = Registry(paths.state_root/"tracking/bagm.sqlite3", initialize=False)
    runs = []
    masks = set()
    for row in chosen:
        root = RunArchive.artifact_path_for(row["run_id"], paths)
        record = registry.get_run(row["run_id"])
        require(record is not None and record["status"] == "completed" and record["campaign_id"] == CAMPAIGN
                and Path(record["artifact_path"]).resolve() == root.resolve(), "Confirmation registry lifecycle differs")
        require(verify_run_bundle(root)["status"] == "success" and (root/"_SUCCESS").is_file(), "Confirmation bundle is not finalized")
        config = yaml.safe_load((root/"config.resolved.yaml").read_text())
        require(config["metadata"]["execution_profile"] == "confirmation" and config["seed"] == row["seed"], "Run profile/seed differs")
        trainer = config["trainer"]
        actual_setting = (trainer["optimizer_batch_mode"], trainer["target_cells_per_optimizer_update"],
                          trainer["learning_rate"], config["model"]["dropout"], trainer["weight_decay"])
        require(actual_setting == setting(row), "Parent report settings differ from the immutable run")
        hist = read_json(root/HISTOGRAM_PATH)
        values = threshold_rows(hist)
        final = read_json(root/"metrics/final.json")
        check_primary_metrics(values, final)
        require(math.isclose(float(final["val/target_only_masked_binary_cross_entropy"]),
                             float(row["selected_best"]["bce"]), rel_tol=0, abs_tol=1e-12), "Parent selected-best BCE differs")
        replay = read_json(root/"diagnostics/final_checkpoint_reload_verification.json")
        require(replay["verified"] is True and replay["best_epoch"] == row["selected_best_epoch"], "Selected-best replay differs")
        masks.add(replay["validation_coverage_receipts_sha256"])
        checkpoint_sha = sha256_file(root/"checkpoints/best.ckpt")
        catalog = registry.list_checkpoint_catalog(run_id=row["run_id"], role="best", verification_status="verified")
        require(len(catalog) == 1 and catalog[0]["sha256"] == checkpoint_sha == replay["checkpoint_sha256"]
                == row["selected_best_checkpoint_sha256"], "Selected-best checkpoint provenance differs")
        label = ("incumbent" if setting(row) == incumbent else "comparator") + f" seed {row['seed']}"
        runs.append({"run_id": row["run_id"], "label": label, "seed": row["seed"],
            "selected_best_epoch": replay["best_epoch"], "checkpoint_sha256": checkpoint_sha,
            "selected_best_bce": final["val/target_only_masked_binary_cross_entropy"],
            "selected_best_average_precision_histogram_4096": final["val/unseen_donor/detection_auprc_histogram_4096"],
            "selected_best_auroc_histogram_4096": final["val/unseen_donor/detection_auroc_histogram_4096"],
            "selected_best_brier_score": final["val/unseen_donor/detection_brier_score"],
            "terminal_primary_bce": row["endpoint_bce"], "thresholds": values})
        for relative in ("_SUCCESS", "config.resolved.yaml", "provenance/artifact_checksums.json", HISTOGRAM_PATH,
                         "metrics/final.json", "diagnostics/final_checkpoint_reload_verification.json"):
            source = root/relative
            provenance.append({"root": "artifact", "run_id": row["run_id"],
                "path": source.relative_to(paths.artifact_root).as_posix(), "sha256": sha256_file(source)})
    require(len(masks) == 1, "Confirmation runs used different validation masks")
    require(len({(run["thresholds"][0]["positive_entries"], run["thresholds"][0]["masked_entries"])
                 for run in runs}) == 1, "Confirmation histogram supports differ")
    return parent, runs, provenance


def plot(path: Path, runs: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for axis, key in zip(axes.flat, METRICS):
        for run in sorted(runs, key=lambda r: r["label"]):
            rows = run["thresholds"]
            axis.plot([r["reported_aligned_threshold"] for r in rows], [r[key] for r in rows], marker="o", label=run["label"])
        if key == "f1":
            support = runs[0]["thresholds"][0]
            reference = constant_classifier_references(support["positive_entries"], support["masked_entries"])
            axis.axhline(reference["classifiers"][0]["f1"], color="#666666", linestyle="--", linewidth=1,
                         label="Always-positive F1 reference")
            axis.legend(fontsize=7)
        axis.set(xlabel="Reported histogram-aligned cutoff", ylabel=key.replace("_", " "), ylim=(0, 1))
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Selected-best validation predictions: exploratory operating points")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def display_metric(row: Mapping[str, Any], metric: str) -> str:
    value = f"{100*row[metric]:.3f}%"
    if not row["exact_at_requested_threshold"]:
        value += f" [{100*row[metric+'_lower_bound']:.3f}, {100*row[metric+'_upper_bound']:.3f}]"
    return value


def build_report(parent_path: Path, version: str, *, paths: Any) -> Path:
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", version)), "Unsafe report version")
    parent, runs, provenance = load_sources(parent_path, paths)
    support = runs[0]["thresholds"][0]
    references = constant_classifier_references(support["positive_entries"], support["masked_entries"])
    output = paths.report_root/"analyses/so2_bce_threshold_sensitivity"/version
    require(not output.exists(), "Immutable threshold report version already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".threshold-", dir=output.parent) as temporary:
        root = Path(temporary)/"report"
        root.mkdir()
        summary = {"schema": "so2_bce_histogram_threshold_sensitivity_v1", "campaign_id": CAMPAIGN,
            "analysis_status": "post_hoc_exploratory_designed_after_historical_scores_inspected",
            "requested_thresholds": list(THRESHOLDS), "primary_threshold_unchanged": .5,
            "checkpoint_role": "selected_validation_best", "terminal_primary_selection_changed": False,
            "threshold_optimized_or_selected": False, "new_inference_performed": False,
            "bin_count": BIN_COUNT, "bin_width": 1/BIN_COUNT, "runs": runs, "sources": provenance,
            "count_derived_references": references,
            "bounds_type": "histogram_discretization_bounds_not_statistical_confidence_intervals",
            "nominal_bounds_method": "Independent extrema of unknown positive and negative counts in the single boundary bin; conservative because within-bin probability sums are not used",
            "interpretation": "Operating-point sensitivity on the reused validation donor; this does not establish optimization gain or independent threshold validity."}
        (root/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
        flat = []
        for run in runs:
            for row in run["thresholds"]:
                flat.append({"run_id": run["run_id"], "label": run["label"], "selected_best_epoch": run["selected_best_epoch"],
                    **{k: v for k, v in row.items() if k != "requested_threshold_bounds"},
                    **{f"{metric}_{bound}": limits[index] for metric, limits in row["requested_threshold_bounds"].items()
                       for index, bound in enumerate(("lower_bound", "upper_bound"))}})
        with (root/"thresholds.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        with (root/"constant_classifier_references.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(references["classifiers"][0]))
            writer.writeheader(); writer.writerows(references["classifiers"])
        plot(root/"threshold_sensitivity.png", runs)
        text = ["# Exploratory threshold sensitivity", "",
            "This post-hoc diagnostic uses retained selected-best validation histograms for all four confirmation runs. The primary threshold remains 0.5; terminal-epoch model selection is unchanged. No threshold was optimized or selected.", "",
            "![Precision, recall, F1 and positive prediction frequency at four declared cutoffs](threshold_sensitivity.png)", "",
            "The table reports exact metrics at the displayed upward-aligned cutoff. Brackets give histogram discretization bounds in percentage units at the requested nominal cutoff; these are not statistical confidence intervals. Within-bin positions were not retained. Requested cutoffs 0.1, 0.2 and 0.3 fall inside bins; the 0.5 row exactly matches saved confusion counts. Full-precision bounds are also in thresholds.csv and summary.json.", "",
            "| Run | Requested | Actual cutoff | Precision | Recall | F1 | Predicted positive |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for row in sorted(flat, key=lambda r: (r["label"], r["requested_threshold"])):
            text.append(f"| {row['label']} | {row['requested_threshold']:.1f} | {row['reported_aligned_threshold']:.9f} | "
                + " | ".join(display_metric(row, k) for k in METRICS) + " |")
        p = references["observed_positive_rate"]
        context = (f"Observed validation prevalence is {p:.8f}. Predicting every entry positive gives precision {100*p:.3f}%, "
            f"recall 100%, and F1 = 2p/(1+p) = {references['classifiers'][0]['f1']:.8f}. "
            "This can exceed a model's F1 at 0.5 while producing many more false positives. F1 ignores true negatives and depends on the operating point; "
            "a 0.5 cutoff is not generally F1-optimal for rare positives. A lower F1 than this trivial rule does not by itself establish poor probability ranking or failed optimization. "
            "Read precision, recall and prediction frequency together with AP and probability scores. AP summarizes ranking across cutoffs; the constant-score, tied-score AP reference equals prevalence. "
            "These retrospective count references do not choose a threshold or constitute an independent fitted baseline. Always-negative precision is undefined and displayed as zero by convention.")
        text += ["", context, "", "| Count-derived reference | TP | FP | TN | FN | Precision | Recall | F1 | Predicted positive |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in references["classifiers"]:
            text.append(f"| {row['classifier']} | {row['true_positive']} | {row['false_positive']} | {row['true_negative']} | {row['false_negative']} | "
                        + " | ".join(f"{100*row[key]:.3f}%" for key in METRICS)+" |")
        text += ["", "Selected-best probability metrics (AP is 4096-bin average precision; all are threshold-independent):", "",
            "| Run | Selected epoch | BCE | Brier | AP | AUROC |", "|---|---:|---:|---:|---:|---:|"]
        for run in sorted(runs, key=lambda r: r["label"]):
            text.append(f"| {run['label']} | {run['selected_best_epoch']} | {run['selected_best_bce']:.8f} | {run['selected_best_brier_score']:.8f} | "
                f"{run['selected_best_average_precision_histogram_4096']:.8f} | {run['selected_best_auroc_histogram_4096']:.8f} |")
        text += ["", "Lowering a cutoff changes the precision/recall tradeoff; it does not improve the model's probability estimates or prove failed optimization. "
            "These same-donor, selected-checkpoint observations provide no independent validation of a deployment threshold, graph-specific gain, count reconstruction, or causality. "
            "All four runs are shown; optimization seeds are not independent biological replicates."]
        markdown = "\n".join(text)+"\n"
        (root/"report.md").write_text(markdown)
        png = base64.b64encode((root/"threshold_sensitivity.png").read_bytes()).decode("ascii")
        table = "<table><thead><tr>"+"".join(f"<th>{html.escape(k)}</th>" for k in ("Run", "Requested", "Actual cutoff", *METRICS))+"</tr></thead><tbody>"
        for row in sorted(flat, key=lambda r: (r["label"], r["requested_threshold"])):
            values = [row["label"], f"{row['requested_threshold']:.1f}", f"{row['reported_aligned_threshold']:.9f}", *[display_metric(row, k) for k in METRICS]]
            table += "<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in values)+"</tr>"
        table += "</tbody></table>"
        reference_table = "<table><thead><tr>"+"".join(f"<th>{key}</th>" for key in
            ("Count-derived reference", "TP", "FP", "TN", "FN", *METRICS))+"</tr></thead><tbody>"
        for row in references["classifiers"]:
            values = [row["classifier"], *[row[key] for key in ("true_positive", "false_positive", "true_negative", "false_negative")],
                      *[f"{100*row[key]:.3f}%" for key in METRICS]]
            reference_table += "<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in values)+"</tr>"
        reference_table += "</tbody></table>"
        quality_table = "<table><thead><tr>"+"".join(f"<th>{key}</th>" for key in
            ("Run", "Selected epoch", "BCE", "Brier", "AP (4096 bins)", "AUROC (4096 bins)"))+"</tr></thead><tbody>"
        for run in sorted(runs, key=lambda r: r["label"]):
            values = [run["label"], run["selected_best_epoch"], *[f"{run[key]:.8f}" for key in
                ("selected_best_bce", "selected_best_brier_score", "selected_best_average_precision_histogram_4096", "selected_best_auroc_histogram_4096")]]
            quality_table += "<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in values)+"</tr>"
        quality_table += "</tbody></table>"
        page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Exploratory threshold sensitivity</title><style>body{font:16px system-ui;color:#172331;background:#fff;max-width:1050px;margin:auto;padding:2rem}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{padding:.4rem;border-bottom:1px solid #ccd3dc;text-align:right}th:first-child,td:first-child{text-align:left}@media print{body{padding:0}tr,img{break-inside:avoid}}</style><main><h1>Exploratory threshold sensitivity</h1>'
        page += "<p>"+html.escape(text[2])+"</p><figure><img alt=\"Four operating-point panels showing precision, recall, F1 and predicted-positive frequency for all confirmation runs\" src=\"data:image/png;base64,"+png+'\"><figcaption>Selected-best validation predictions. Lines connect the four declared aligned operating points; no intermediate thresholds were evaluated.</figcaption></figure>'
        page += "<p>"+html.escape(text[6])+"</p>"+table+"<p>"+html.escape(context)+"</p>"+reference_table
        page += "<p>Selected-best probability metrics; these do not change when the decision cutoff changes.</p>"+quality_table+"<p>"+html.escape(text[-1])+"</p></main></html>"
        (root/"report.html").write_text(page)
        shutil.copyfile(Path(__file__), root/"report_builder.py")
        manifest = {"schema": "checksummed_local_report_v1", "files": {p.name: sha256_file(p) for p in sorted(root.iterdir())}}
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        _rename_no_replace(root, output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args(argv)
    paths = current_paths()
    source = args.parent_report if args.parent_report.is_absolute() else paths.project_root/args.parent_report
    print(json.dumps({"report": str(build_report(source, args.version, paths=paths))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
