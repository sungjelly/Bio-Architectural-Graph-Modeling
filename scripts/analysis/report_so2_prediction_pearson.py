#!/usr/bin/env python3
"""Publish a descriptive comparison of three verified frozen-checkpoint replays.

No inference, fitting, threshold selection, or best-seed selection is performed.
Use --check to verify an existing immutable comparison and its three inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid

from spatial_benchmark.paths import current_paths
from run_so2_prediction_pearson import RUNS, now, read, sha, verify as verify_replay, write

SCHEMA = "so2_prediction_pearson_comparison_v1"
METRICS = (
    "cell_pearson_mean", "gene_pearson_mean", "pooled_pearson", "masked_mae",
    "masked_mse", "masked_rmse", "masked_r2", "n_cells", "n_masked",
    "n_valid_cells", "n_undefined_cells", "n_valid_genes", "n_undefined_genes",
)
CORRELATIONS = ("cell_pearson_mean", "gene_pearson_mean", "pooled_pearson")
LABELS = {"local_s0": "Local residual, seed 0", "local_s1": "Local residual, seed 1",
          "local_seed_mean": "Local residual, mean of both seeds", "historical_e300": "Historical BAGM, seed 0"}


def mean(values):
    # Undefined metric values remain explicitly undefined; never silently select seeds.
    if not values or any(value is None for value in values):
        return None
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("Nonfinite value in input metric")
    return sum(float(value) for value in values) / len(values)


def fmt(value, digits=4):
    return "undefined" if value is None else f"{value:.{digits}f}"


def count(value):
    return "undefined" if value is None else f"{int(value):,}"


def load_inputs(root):
    inputs, provenance = {}, {}
    for key, run_id in RUNS.items():
        source = root / key
        manifest = verify_replay(source)
        result = read(source / "result.json")
        if manifest["source_run_id"] != run_id or result["source_run_id"] != run_id:
            raise ValueError(f"Unexpected source identity: {key}")
        if result["verification"].get("passed") is not True:
            raise ValueError(f"Replay verification failed: {key}")
        for metric in result["global"].values():
            if metric["n_valid_cells"] + metric["n_undefined_cells"] != metric["n_cells"]:
                raise ValueError(f"Cell denominator does not partition: {key}")
            if metric["n_valid_genes"] + metric["n_undefined_genes"] != 1000:
                raise ValueError(f"Gene denominator does not partition: {key}")
        inputs[key] = result
        provenance[key] = {"run_id": run_id, "input_directory": str(source),
            "manifest_sha256": sha(source / "manifest.json"),
            "result_sha256": sha(source / "result.json"),
            "checkpoint_sha256": result["checkpoint_sha256"]}
    if inputs["local_s0"]["seed"] != 0 or inputs["local_s1"]["seed"] != 1:
        raise ValueError("Both prespecified local seeds are required")
    for key in ("local_s0", "local_s1"):
        if inputs[key]["global"]["log1p"]["n_cells"] != 37367:
            raise ValueError("Local validation population changed")
        if inputs[key]["global"]["log1p"]["n_masked"] != 18708207:
            raise ValueError("Local validation masking support changed")
    if inputs["historical_e300"]["global"]["log1p"]["n_cells"] != 246063:
        raise ValueError("Historical fitted population changed")
    return inputs, provenance


def equal_core(result, scale):
    keys = (*CORRELATIONS, "masked_mae", "masked_mse", "masked_rmse", "masked_r2")
    values = {key: mean([row["metrics"][scale][key] for row in result["per_core"]]) for key in keys}
    # Validate the historical helper's independently saved equal-core values.
    if "equal_core" in result:
        for key, value in values.items():
            expected = result["equal_core"][scale][key]
            if value is None or expected is None:
                if value is not expected:
                    raise ValueError("Undefined equal-core metric differs")
            elif abs(value - expected) > 1e-12:
                raise ValueError("Equal-core aggregation differs from replay receipt")
    return values


def make_rows(inputs):
    rows = []
    for key, result in inputs.items():
        scope = "reused_validation_one_donor" if key.startswith("local") else "fitted_cohort_all14"
        for scale, metrics in result["global"].items():
            for aggregation, values in (("global", metrics), ("equal_core", equal_core(result, scale))):
                rows.append({"endpoint": key, "source_run_id": result["source_run_id"],
                    "seed": result["seed"], "scope": scope, "aggregation": aggregation,
                    "scale": scale, **{metric: values.get(metric) for metric in METRICS}})
    for scale in ("raw_count", "log1p"):
        for aggregation in ("global", "equal_core"):
            members = [row for row in rows if row["endpoint"] in ("local_s0", "local_s1")
                       and row["scale"] == scale and row["aggregation"] == aggregation]
            if len(members) != 2:
                raise ValueError("Missing prespecified seed in mean")
            rows.append({"endpoint": "local_seed_mean", "source_run_id": "|".join(RUNS[key] for key in ("local_s0", "local_s1")),
                "seed": "mean_of_0_and_1", "scope": "reused_validation_one_donor",
                "aggregation": aggregation, "scale": scale,
                **{metric: mean([row[metric] for row in members]) for metric in METRICS}})
    return rows


def find_row(rows, key, scale, aggregation="global"):
    return next(row for row in rows if row["endpoint"] == key and row["scale"] == scale
                and row["aggregation"] == aggregation)


def pearson_table(rows, keys, scale, aggregation="global"):
    lines = ["| Model | Mean cell Pearson | Mean gene Pearson | Pooled Pearson | MAE | RMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in keys:
        row = find_row(rows, key, scale, aggregation)
        lines.append("| " + LABELS[key] + " | " + " | ".join(fmt(row[m]) for m in
            (*CORRELATIONS, "masked_mae", "masked_rmse")) + " |")
    return "\n".join(lines)


def report_text(inputs, rows):
    local = find_row(rows, "local_seed_mean", "log1p")
    historical = find_row(rows, "historical_e300", "log1p")
    lines = [
        "# Masked-expression Pearson correlation of existing SO2 predictors", "",
        f"On log1p observed counts, the current Local Gene Residual model has mean cell-wise Pearson **{fmt(local['cell_pearson_mean'])}** "
        f"across its two seeds on the reused validation donor. The historical BAGM endpoint has **{fmt(historical['cell_pearson_mean'])}** "
        "on its fitted cohort. These values describe different populations and masking protocols; they are not a matched model ranking.", "",
        "Both local seeds and the historical endpoint were fixed from existing error reports before this analysis. "
        "No model was retrained and no checkpoint, seed or threshold was selected using these correlations.", "",
        "## Primary scale: log1p observed counts", "",
        pearson_table(rows, ("local_s0", "local_s1", "local_seed_mean", "historical_e300"), "log1p"), "",
        "Cell-wise Pearson correlates predictions and targets across the masked genes within each cell, then averages valid cells. "
        "Gene-wise Pearson correlates across masked cells for each gene, then averages valid genes. "
        "Pooled Pearson uses all masked entries together and can be influenced by differences in average abundance between genes. "
        "These three quantities answer different questions. All original masked zero and positive observations are included.", "",
        "The local mean is an arithmetic mean of the two model scores, not an ensemble prediction. "
        "Its seed range for cell-wise Pearson is " +
        f"{fmt(min(find_row(rows,k,'log1p')['cell_pearson_mean'] for k in ('local_s0','local_s1')))}–" +
        f"{fmt(max(find_row(rows,k,'log1p')['cell_pearson_mean'] for k in ('local_s0','local_s1')))}. "
        "Seeds are technical replicates and do not provide independent-donor uncertainty.", "",
        "Local log-scale predictions are log1p of the marginal expected count, not expected log1p counts. "
        "Historical log predictions are inverse-standardized model outputs without clipping negative values.", "",
        "## Native and additional output scales", "",
        "Local model, raw observed counts:", "",
        pearson_table(rows, ("local_s0", "local_s1", "local_seed_mean"), "raw_count"), "",
        "Historical BAGM, its native gene-standardized log1p scale:", "",
        pearson_table(rows, ("historical_e300",), "standardized"), "",
        "MAE/RMSE units change with the output scale. Gene-specific standardization can also change cell-wise and pooled Pearson; "
        "a common log1p scale alone does not make these experiments comparable.", "",
        "## Equal-core sensitivity analysis", "",
        "Each core is scored separately, then core scores are averaged equally. The local row additionally averages both seeds. "
        "This differs from the global table's valid-cell weighting and global per-gene correlations.", "",
        pearson_table(rows, ("local_s0", "local_s1", "local_seed_mean", "historical_e300"), "log1p", "equal_core"), "",
        "Equal-core RMSE is the arithmetic mean of per-core RMSE, not the square root of mean per-core MSE. "
        "The complete CSV keeps these aggregation labels explicit.", "",
        "## Correlation support and undefined values", "",
        "Fewer than two selected values or a centered sum of squares at most 1e-12 in either vector makes correlation undefined. "
        "Undefined correlations are excluded from the means and counted below; they are never replaced with zero.", "",
        "| Model | Scale | Cells | Masked entries | Valid / undefined cell correlations | Valid / undefined gene correlations |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, result in inputs.items():
        for scale, metric in result["global"].items():
            lines.append(f"| {LABELS[key]} | {scale} | {count(metric['n_cells'])} | {count(metric['n_masked'])} | "
                         f"{count(metric['n_valid_cells'])} / {count(metric['n_undefined_cells'])} | "
                         f"{count(metric['n_valid_genes'])} / {count(metric['n_undefined_genes'])} |")
    lines += ["", "## Detection remains a separate question", "",
        "Pearson measures co-variation and does not guarantee correct zero/nonzero decisions. "
        "Targets `[0, 1, 2]` and predictions `[1, 3, 5]` have Pearson 1, although the observed zero is predicted positive and every magnitude is wrong. "
        "The following existing local detection metrics were replayed unchanged, using the probability head and threshold 0.5. "
        "Average precision is approximated using 4,096 probability bins; no threshold or metric was optimized here.", "",
        "| Model | Precision | Recall | F1 | Average precision |",
        "|---|---:|---:|---:|---:|"]
    detection_keys = ("detection_precision", "detection_recall", "detection_f1", "detection_average_precision")
    for key in ("local_s0", "local_s1", "local_seed_mean"):
        if key == "local_seed_mean":
            values = {metric: mean([inputs[k]["native_metrics"][metric] for k in ("local_s0", "local_s1")]) for metric in detection_keys}
        else:
            values = inputs[key]["native_metrics"]
            if values["detection_histogram_bins"] != 4096:
                raise ValueError("Unexpected detection approximation")
        lines.append("| " + LABELS[key] + " | " + " | ".join(fmt(values[metric]) for metric in detection_keys) + " |")
    lines += ["", "These detection labels mean observed count greater than zero in this assay; an observed zero does not establish biological RNA absence. "
        "Historical BAGM has no probability-head detection result in this replay.", "",
        "## Selection, populations and permitted conclusions", "",
        "The Local Gene Residual family was chosen because its completed two-seed aggregate had the lowest raw-count RMSE among the current count-model comparisons. "
        "Its training loss was hurdle count negative log-likelihood, and both saved epoch-30 checkpoints were originally selected by validation NLL. "
        "It was fitted on 208,696 cells from twelve cores/six donors and evaluated here on 37,367 cells from two cores of one different donor. "
        "That validation donor was repeatedly used for selection; this is not a sealed test estimate.", "",
        "Historical Relative-QKV uses its fixed epoch-300 endpoint, chosen from the previous four-endpoint comparison by standardized MSE. "
        "It was Huber-trained on all fourteen cores, including the two cores later used for validation. "
        "All 246,063 replayed cells therefore belong to its fitted cohort; only one seed is available.", "",
        "The local protocol masks Uniform{1,…,1000} target genes and allows clean non-self neighbor summaries. "
        "The historical protocol masks Uniform{0,…,1000} genes simultaneously in every graph cell. "
        "Different available neighbor information, populations, fitting scopes and normalization preclude attributing score differences to architecture or training-set size. "
        "These results add descriptive reconstruction metrics, without establishing external benchmark superiority, independent-test generalization, biological mechanism or causal effects.", "",
        "## Verification and immutable sources", "",
        "All three input bundles must pass checksum and completion verification before this report is written. "
        "The local replays match archived native metrics within absolute tolerance 1e-5; the historical replay uses 2e-6. "
        "Checkpoint, mask and data identities are verified by the input replays. Float64 correlation accumulation was checked against SciPy on a fixture with masking, singleton and constant vectors. "
        "No row-level prediction matrices are exported.", "",
        "| Endpoint | Immutable source run | Replay bundle |",
        "|---|---|---|"]
    for key in RUNS:
        lines.append(f"| {LABELS[key]} | `{RUNS[key]}` | [Metrics](../{key}/result.json), [manifest](../{key}/manifest.json) |")
    lines += ["", "[Complete summary CSV](summary.csv) retains all scales, seeds and aggregation rules. "
        "[Input provenance](input_provenance.json) records source manifest/result/checkpoint hashes; "
        "[manifest](manifest.json) binds this report and its source snapshot.", "",
        "Reproduce from the project root with `BAGM_ROOT=/workspace/BAGM PYTHONPATH=/workspace/BAGM-relative-qkv-analysis/src:/workspace/BAGM/scripts/analysis "
        "/venv/main/bin/python scripts/analysis/report_so2_prediction_pearson.py`; use `--check` to verify the existing immutable report. "
        "An existing comparison is never overwritten.", ""]
    return "\n".join(lines)


def verify_comparison(destination):
    manifest = read(destination / "manifest.json")
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unexpected comparison schema")
    success = read(destination / "_SUCCESS")
    if success.get("status") != "success" or success["manifest_sha256"] != sha(destination / "manifest.json"):
        raise ValueError("Comparison completion checksum mismatch")
    actual = {str(path.relative_to(destination)) for path in destination.rglob("*")
              if path.is_file() and path.name not in ("manifest.json", "_SUCCESS")}
    if actual != set(manifest["files"]):
        raise ValueError("Comparison file inventory differs")
    for relative, expected in manifest["files"].items():
        path = destination / relative
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError("Invalid manifest path")
        if sha(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"Comparison payload mismatch: {relative}")
    for key, expected in read(destination / "input_provenance.json").items():
        source = Path(expected["input_directory"])
        verify_replay(source)
        if sha(source / "manifest.json") != expected["manifest_sha256"] or sha(source / "result.json") != expected["result_sha256"]:
            raise ValueError(f"Input manifest/result changed: {key}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paths = current_paths()
    root = paths.report_root / "analyses/so2_prediction_pearson/v1"
    destination = root / "comparison"
    if args.check:
        verify_comparison(destination)
        print(json.dumps({"status": "verified", "path": str(destination)}))
        return
    if destination.exists():
        raise FileExistsError(f"Immutable report already exists; use --check: {destination}")
    inputs, provenance = load_inputs(root)
    rows = make_rows(inputs)
    report = report_text(inputs, rows)
    stage = paths.scratch_root / "active_runs" / ("so2_prediction_pearson_comparison_" + uuid.uuid4().hex[:12])
    stage.mkdir(parents=True, exist_ok=False)
    try:
        with (stage / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("endpoint", "source_run_id", "seed", "scope", "aggregation", "scale", *METRICS))
            writer.writeheader()
            writer.writerows(rows)
        (stage / "report.md").write_text(report)
        write(stage / "input_provenance.json", provenance)
        shutil.copyfile(Path(__file__), stage / "report_builder.py")
        write(stage / "execution.json", {"created_at": now(), "command": sys.argv,
            "python_executable": sys.executable, "source_sha256": sha(Path(__file__)),
            "source_file": str(Path(__file__).resolve()), "input_keys": list(RUNS),
            "new_training": False, "analysis_design": "descriptive_posthoc_extension"})
        files = {str(path.relative_to(stage)): {"sha256": sha(path), "size_bytes": path.stat().st_size}
                 for path in sorted(stage.rglob("*")) if path.is_file()}
        write(stage / "manifest.json", {"schema": SCHEMA, "files": files,
            "input_manifest_sha256": {key: value["manifest_sha256"] for key, value in provenance.items()}})
        write(stage / "_SUCCESS", {"status": "success", "completed_at": now(), "manifest_sha256": sha(stage / "manifest.json")})
        verify_comparison(stage)
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(stage, destination)
        verify_comparison(destination)
        print(json.dumps({"status": "published", "path": str(destination), "rows": len(rows)}))
    except BaseException:
        print(f"Comparison staging evidence retained at {stage}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
