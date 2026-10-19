#!/usr/bin/env python3
"""Report a completed joint-slide preprocessing run without changing its bundle.

Run from the project root with PYTHONPATH=src. Only aggregate diagnostics and
an existing figure enter the report; no cell-level or direct donor identifiers
are exported. Missing or undefined gate metrics cannot count as a pass.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil

from spatial_benchmark.paths import current_paths


METHODS = ("log1p_raw", "log1p_cp10k", "pearson", "signed_log_pearson")
LABELS = {
    "log1p_raw": "Log1p raw counts",
    "log1p_cp10k": "Log1p counts per 10,000",
    "pearson": "Quasi-Poisson Pearson",
    "signed_log_pearson": "Signed-log Pearson",
}
FIRST_ATTEMPT = "r_20261016T082320Z_660b87cb_s000_f00_a01_cd1c1544"
VENDOR_TUTORIAL = "https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/pearsonpca/"


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def fmt(value, digits=3):
    value = number(value)
    return "undefined" if value is None else f"{value:.{digits}f}"


def slide_scores(method, control):
    scores = {}
    for row in method["within_slide_vendor_label_recovery"]:
        if row.get("control") != control or row.get("status") != "ok":
            continue
        slide = str(row["slide"]).replace("SO_", "").replace("SO", "")
        if slide in scores:
            raise ValueError(f"Duplicate {control} metric for slide {slide}")
        scores[slide] = number(row.get("balanced_accuracy"))
    return scores


def comparison_rows(metrics, config):
    """Apply the prespecified strict depth and inclusive 0.05-drop guardrails."""
    if config["acceptance"]["depth_r2"] != "below_log1p_raw":
        raise ValueError("Unsupported primary gate; refusing to reinterpret it")
    max_drop = float(config["acceptance"]["max_within_slide_label_accuracy_drop"])
    baseline = metrics["methods"]["log1p_raw"]
    baseline_depth = number(baseline["heldout_core_median_depth_weighted_r2"])
    baseline_scores = slide_scores(baseline, "observed_labels")
    rows = []
    for method in METHODS:
        result = metrics["methods"][method]
        depth = number(result["heldout_core_median_depth_weighted_r2"])
        observed = slide_scores(result, "observed_labels")
        null = slide_scores(result, "within_core_train_label_shuffle")
        complete = baseline_depth is not None and depth is not None and all(
            baseline_scores.get(slide) is not None and observed.get(slide) is not None
            for slide in ("1", "2")
        )
        drops = {slide: (baseline_scores[slide] - observed[slide] if complete else None)
                 for slide in ("1", "2")}
        depth_pass = bool(complete and depth < baseline_depth)
        label_pass = bool(complete and all(
            drop <= max_drop or math.isclose(drop, max_drop, rel_tol=0, abs_tol=1e-12)
            for drop in drops.values()
        ))
        gate = ("reference" if method == "log1p_raw" else
                "incomplete" if not complete else "pass" if depth_pass and label_pass else "fail")
        rows.append({
            "method": method, "depth_r2": depth,
            "qc_pass_depth_r2": number(result.get("heldout_qc_pass_core_median_depth_weighted_r2")),
            "so1_balanced_accuracy": observed.get("1"),
            "so2_balanced_accuracy": observed.get("2"),
            "so1_null_balanced_accuracy": null.get("1"),
            "so2_null_balanced_accuracy": null.get("2"),
            "slide_auc": number(result["slide_diagnostics"].get("heldout_slide_auc")),
            "so1_accuracy_drop": drops["1"], "so2_accuracy_drop": drops["2"],
            "depth_pass": depth_pass, "label_guardrail_pass": label_pass, "gate": gate,
        })
    return rows


def relative_link(target, report_dir):
    return Path(os.path.relpath(target, report_dir)).as_posix()


def table(headers, rows, output="markdown"):
    if output == "html":
        head = "".join(f"<th scope='col'>{html.escape(str(cell))}</th>" for cell in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>" for row in rows)
        return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def load_bundle(paths, run_id):
    match = re.fullmatch(r"r_(\d{4})(\d{2})\d{2}T[^/]+", run_id)
    if match is None:
        raise ValueError("Expected an immutable r_YYYYMMDDT... run ID")
    bundle = paths.artifact_root / "runs" / match[1] / match[2] / run_id
    if not (bundle / "_SUCCESS").is_file():
        raise FileNotFoundError(f"No completed immutable bundle: {bundle}")
    if (bundle / "_SUCCESS").read_text().strip() != run_id:
        raise ValueError("Completion marker does not match requested run")
    checksums = read(bundle / "checksums.json")
    names = ("identity.json", "resolved_config.json", "metrics.json", "verification.json",
             "environment.json", "input_provenance.json", "transform_diagnostics.json",
             "smoke_checks.json", "derived_dataset_reference.json", "diagnostics/preprocessing_diagnostics.png")
    for name in names:
        if sha256(bundle / name) != checksums.get(name):
            raise ValueError(f"Run evidence checksum mismatch: {name}")
    objects = {name.removesuffix(".json"): read(bundle / name) for name in names if name.endswith(".json")}
    if objects["identity"]["run_id"] != run_id:
        raise ValueError("Run identity mismatch")
    verification = objects["verification"]
    if not verification.get("all_arrays_finite") or not verification.get("all_shapes_verified"):
        raise ValueError("Input run lacks finite, shape-verified outputs")
    reference = objects["derived_dataset_reference"]
    dataset = paths.data_root / "processed" / "so12_pearson_integration_v1"
    if not (dataset / "manifest.json").is_file():
        dataset = Path(reference["path"])
    if sha256(dataset / "manifest.json") != reference["manifest_sha256"]:
        raise ValueError("Derived dataset manifest checksum mismatch")
    manifest = read(dataset / "manifest.json")
    if manifest["run_id"] != run_id or manifest.get("training_ready_for_masked_counts") is not False:
        raise ValueError("Unexpected derived dataset identity or masking contract")
    if (manifest["cells"], manifest["genes"]) != (407659, 1000):
        raise ValueError("Unexpected cohort dimensions")
    return bundle, dataset, objects, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--failed-run-id", action="append", default=[])
    args = parser.parse_args()
    paths = current_paths()
    bundle, dataset, data, manifest = load_bundle(paths, args.run_id)
    config, metrics = data["resolved_config"], data["metrics"]
    rows = comparison_rows(metrics, config)
    report_dir = paths.report_root / "analyses" / "so12_pearson_preprocessing"
    report_dir.mkdir(parents=True, exist_ok=True)
    headers = ["Transform", "Depth R²", "SO_1 BA", "SO_2 BA", "SO_1 null BA", "SO_2 null BA", "Slide AUROC", "Gate"]
    display = [[LABELS[row["method"]], *[fmt(row[key]) for key in
                ("depth_r2", "so1_balanced_accuracy", "so2_balanced_accuracy", "so1_null_balanced_accuracy",
                 "so2_null_balanced_accuracy", "slide_auc")], row["gate"]] for row in rows]
    passed = [LABELS[row["method"]] for row in rows if row["gate"] == "pass"]
    outcome = ("The locked preprocessing plausibility gate passes for " + ", ".join(passed) + "."
               if passed else "None of the candidate transforms passes the locked preprocessing plausibility gate.")
    pearson_row = next(row for row in rows if row["method"] == "pearson")
    recommendation = ("Use plain quasi-Poisson Pearson as the primary candidate in a subsequent controlled training comparison, "
                      "with signed-log Pearson retained as an ablation. This prioritizes the documented residual transform without "
                      "claiming that its training performance is established. Passing this gate does not establish an advantage "
                      "in neural-network training."
                      if pearson_row["gate"] == "pass" else
                      "Plain Pearson did not pass the gate; retain the baselines and investigate the failed criterion before training.")
    signed_row = next(row for row in rows if row["method"] == "signed_log_pearson")
    pairs = [(pearson_row[key], signed_row[key]) for key in ("so1_balanced_accuracy", "so2_balanced_accuracy")]
    if all(a is not None and b is not None for a, b in pairs) and (pairs[0][1]-pairs[0][0]) * (pairs[1][1]-pairs[1][0]) < 0:
        recommendation += " Signed log trades annotation agreement between slides and does not demonstrate a clear overall advantage."
    gate_text = (f"A candidate must have lower median held-out-core depth R² than log1p(raw), and its held-out "
                 f"vendor-label balanced accuracy must drop by no more than {config['acceptance']['max_within_slide_label_accuracy_drop']:.2f} "
                 "on each slide. Both conditions are required. Slide AUROC and label-null accuracy do not select a winner.")
    split = data["input_provenance"]["split"]
    cohort_text = (f"The dataset retains {manifest['cells']:,} mapped cells × {manifest['genes']:,} biological probes across 28 cores. "
                   f"Training: {split['training_cells']:,} cells from {split['training_donors']} donors; held-out diagnostics: "
                   f"{split['diagnostic_cells']:,} cells from {split['diagnostic_donors']} donors. "
                   f"Transform parameters and 32-PC projections were fitted on all training cells pooled across slides; "
                   f"diagnostics use the same {metrics['n_sampled_cells']:,} sampled cells for every method, up to 1,000 per core. "
                   "Paired cores remain in the same partition. The existing exclusion of 340 unmapped SO_2 FOV246 cells is retained.")
    formula = (f"Pearson residuals are defined from raw counts x, not log counts: nᵢ = Σg xᵢg; "
               f"p̂g = Σtraining xᵢg / Σtraining,g xᵢg; μᵢg = nᵢp̂g; "
               f"rᵢg = (xᵢg − μᵢg) / √({config['phi']} μᵢg). "
               f"Use training gene means and sample SDs to obtain z, then cap only the upper tail at {config['upper_clip']:g}. "
               "The signed-log ablation applies sign(z)·log1p(|z|) afterward. It is nonstandard and is not claimed to be an AtoMx option. "
               "Ordinary log1p(z) is invalid when z ≤ −1. Baselines also use frozen training gene standardization. "
               "PCA re-centers each transformed training matrix; this is not a bit-for-bit reproduction of vendor PCA.")
    vendor = config["vendor_source"]
    sources_text = ("The residual formula follows the pinned vendor scPearsonPCA implementation. "
                    "The exact built-in AtoMx residual implementation is not publicly established. "
                    "The vendor tutorial discusses Pearson-residual PCA and batch-correction alternatives; neither source establishes "
                    "that this trial removes technical slide effects.")
    limitations = [
        "Maximum defensible claim: exploratory preprocessing plausibility on one donor-held-out split. No neural network was trained; masked-expression gain, biological integration, and causal mechanisms were not tested.",
        "Opaque vendor labels are expression-derived and their namespaces are evaluated separately by slide. Accuracy is circular annotation agreement, not independent cell-type or biological validation.",
        "Donors do not cross slides, and slide is confounded with tissue composition and disease. Lower slide AUROC or apparent mixing is not proof of technical batch removal and can erase biology.",
        "Lower depth association can reflect removal of real RNA-content biology or mere scale compression. Four held-out donors and one split do not establish robustness or population-level replication.",
        "The label null permutes training annotations within each core, preserving composition. It is one diagnostic permutation, not a significance test; no cell-independent p-values are reported.",
        "Cell totals use every measured gene. Masking after these transforms leaks hidden-count information into visible features. These files are suitable for fully observed representation experiments; strict masked-count prediction needs observed-only preprocessing and raw integer targets.",
        "Vendor QC flags are retained. All mapped cells are primary; QC-pass depth association is a sensitivity check. No new cells or old user files were deleted.",
    ]
    next_test = ("Next discriminating test: compare the passing candidate(s), if any, against both baselines with a locked, "
                 "mask-aware preprocessing protocol, raw-count prediction targets, donor-held-out splits, and per-slide/core results. "
                 "Independent labels or orthogonal measurements are needed before a biological integration claim.")
    diagnostics = data["transform_diagnostics"]
    invalid_fraction = diagnostics["pearson"].get("ordinary_log1p_invalid_fraction")
    if invalid_fraction is not None:
        formula += f" In this dataset, {100 * invalid_fraction:.2f}% of plain-Pearson feature entries are ≤ −1, so ordinary log1p would be nonfinite."
    resource_headers = ["Transform", "GPU", "Transform/PCA seconds", "Peak allocated VRAM (GiB)", "QC-pass depth R²"]
    resource_rows = [[LABELS[row["method"]], diagnostics[row["method"]]["gpu"],
                      fmt(diagnostics[row["method"]]["runtime_seconds"], 1),
                      fmt(diagnostics[row["method"]]["peak_vram_gib"], 2), fmt(row["qc_pass_depth_r2"])] for row in rows]
    verification = data["verification"]
    environment = data["environment"]
    hardware = ", ".join(f"GPU {key}: {value}" for key, value in environment["gpus"].items())
    resource_text = (f"Verified run runtime: {fmt(verification['runtime_seconds'], 1)} seconds; parallel schedule: one transform per GPU. "
                     f"{hardware}. PyTorch {environment['torch']}; CUDA {environment['cuda']}. "
                     f"Derived dataset: {verification['derived_bytes'] / 2**30:.3f} GiB; free disk at verification: "
                     f"{verification['free_disk_gib']:.2f} GiB; free disk when reporting: {shutil.disk_usage(dataset).free / 2**30:.2f} GiB. "
                     "Peak disk usage was not sampled; these are footprint/free-space measurements. No cleanup was performed by this report.")
    failed = list(dict.fromkeys(args.failed_run_id))
    failed_archive = paths.artifact_root / "runs" / "2026" / "10" / FIRST_ATTEMPT
    if not failed and args.run_id != FIRST_ATTEMPT and (
        (failed_archive / "_FAILED").exists()
        or (paths.scratch_root / "active_runs" / FIRST_ATTEMPT / "failure.txt").exists()
    ):
        failed = [FIRST_ATTEMPT]
    failure_text = ("Prior failed attempt(s): " + ", ".join(failed) + ". The first attempt encountered a CUDA lazy linear-algebra "
                    "initialization race during parallel setup; it did not supply scientific results to this report. "
                    "The successful retry initialized libraries serially with unchanged scientific settings. "
                    "The partial failed-run bundle is retained for audit." if failed else "No earlier failed attempt was supplied to this report.")
    dataset_text = (f"Reusable matrices are under {dataset}. Each of pearson/ and signed_log_pearson/ contains 28 float32 per-core .npy "
                    "matrices totaling 407,659 × 1,000 values, plus fitted_parameters.npz. observations.csv.gz preserves row order "
                    "within each core; genes.json supplies column order; manifest.json records shapes and checksums. "
                    "Raw counts remain in the source per-core expression_counts arrays. Preserve (slide, fov, cell_ID) keys and core IDs; "
                    "reuse independent core spatial graphs without cross-core or cross-slide edges.")
    example = '''import json
import numpy as np
import pandas as pd
from spatial_benchmark.paths import current_paths

root = current_paths().data_root / "processed/so12_pearson_integration_v1"
core = "SO1-C01"
features = np.load(root / "pearson" / f"{core}.npy", mmap_mode="r")
metadata = pd.read_csv(root / "observations.csv.gz",
                      usecols=["core", "slide", "fov", "cell_ID", "train", "qc_pass", "x_um", "y_um"])
cells = metadata.loc[metadata["core"] == core].reset_index(drop=True)
genes = json.loads((root / "genes.json").read_text())
assert features.shape == (len(cells), len(genes))
assert not cells.duplicated(["slide", "fov", "cell_ID"]).any()
# Row i in features belongs to row i in cells; preserve this alignment.
'''
    figure = bundle / "diagnostics" / "preprocessing_diagnostics.png"
    figure_alt = "Four preprocessing comparisons: PCA colored by slide and depth association for every held-out core."
    figure_caption = ("Top: sampled first two PCs, colored by slide; this descriptive separation is confounded. "
                      "Bottom: variance-weighted depth R² for every held-out core; dashed line marks the median. "
                      "The primary metric uses all 32 PCs, not just the plotted two.")
    report_links = {"Immutable run": bundle, "Dataset manifest": dataset / "manifest.json",
                    "Per-core depth diagnostics": bundle / "diagnostics/per_core_depth_association.csv",
                    "Per-donor annotation agreement": bundle / "diagnostics/per_donor_vendor_label_recovery.csv"}
    provenance_text = (f"Run: {args.run_id}; seed {config['seed']}; fold {config['fold']}; attempt {config['attempt']}. "
                       f"Metrics SHA256: {sha256(bundle / 'metrics.json')}. "
                       f"Dataset-manifest SHA256: {sha256(dataset / 'manifest.json')}. "
                       "Consumed evidence files were checked against the immutable bundle checksums before rendering.")
    warnings = metrics.get("warnings", [])
    warning_text = "Diagnostic warnings: " + ("; ".join(warnings) if warnings else "none recorded.")
    command = f"PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so12_pearson_preprocessing.py --run-id {args.run_id}"
    markdown = ["# SO_1 + SO_2 preprocessing comparison", outcome,
                "**Maximum claim: exploratory preprocessing plausibility only.**", cohort_text,
                "## Comparison", table(headers, display),
                "Depth R² is the median held-out-core, variance-weighted squared correlation of PCs with log1p biological-probe depth. BA is balanced accuracy. Lower depth R² supports depth reduction; slide AUROC is confounded.",
                gate_text, recommendation, f"![{figure_alt}]({relative_link(figure, report_dir)})", figure_caption,
                "## Transform definitions", formula, sources_text,
                f"Sources: [Pinned vendor R implementation]({vendor}); [Vendor Pearson-PCA tutorial]({VENDOR_TUTORIAL}).",
                "## Reusable dataset", dataset_text, "```python\n" + example + "```",
                "## Interpretation and limits", "\n".join(f"- {item}" for item in limitations), next_test,
                "## Compute and verification", table(resource_headers, resource_rows), resource_text,
                failure_text, warning_text, provenance_text,
                "\n".join(f"- [{name}]({relative_link(path, report_dir)})" for name, path in report_links.items()),
                "Rebuild the report:\n\n```bash\n" + command + "\n```"]
    (report_dir / "README.md").write_text("\n\n".join(markdown) + "\n")
    b64 = base64.b64encode(figure.read_bytes()).decode("ascii")
    paragraph = lambda text: "<p>" + html.escape(text) + "</p>"
    body = ["<h1>SO_1 + SO_2 preprocessing comparison</h1>", paragraph(outcome),
            "<p class='claim'><strong>Exploratory preprocessing plausibility only.</strong></p>", paragraph(cohort_text),
            "<h2>Comparison</h2>", table(headers, display, "html"), paragraph(gate_text), paragraph(recommendation),
            f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(figure_alt)}'><figcaption>{html.escape(figure_caption)}</figcaption></figure>",
            "<h2>Transform definitions</h2>", paragraph(formula), paragraph(sources_text),
            f"<p>Sources: <a href='{html.escape(vendor, quote=True)}'>Pinned vendor R implementation</a>; "
            f"<a href='{VENDOR_TUTORIAL}'>vendor Pearson-PCA tutorial</a>.</p>",
            "<h2>Reusable dataset</h2>", paragraph(dataset_text), "<pre><code>" + html.escape(example) + "</code></pre>",
            "<h2>Interpretation and limits</h2><ul>",
            *["<li>" + html.escape(item) + "</li>" for item in limitations], "</ul>", paragraph(next_test),
            "<h2>Compute and verification</h2>", table(resource_headers, resource_rows, "html"),
            paragraph(resource_text), paragraph(failure_text), paragraph(warning_text), paragraph(provenance_text),
            "<h2>Reproduce</h2><pre><code>" + html.escape(command) + "</code></pre>"]
    style = """body{font:16px/1.5 system-ui,sans-serif;color:#172b3a;background:#fff;max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:2rem}h2{font-size:1.3rem;margin-top:2rem}.claim{border-left:4px solid #24708b;padding:10px 16px;background:#eef6f8}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border-bottom:1px solid #cbd5df;text-align:left;padding:9px}th{background:#edf3f7}img{width:100%;height:auto}figure{margin:24px 0}figcaption{color:#3e5262;font-size:.9rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f6;padding:16px;font-size:.85rem}p{overflow-wrap:anywhere}li{margin:.5rem 0}a{color:#075f8d}@media print{body{max-width:none;margin:0;padding:0;font-size:10pt}.table-wrap{overflow:visible}figure,table{break-inside:avoid}h2{break-after:avoid}}"""
    page = "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>SO_1 + SO_2 preprocessing comparison</title><style>" + style + "</style></head><body><main>" + "\n".join(body) + "</main></body></html>\n"
    (report_dir / "report.html").write_text(page)
    receipt = {"run_id": args.run_id, "maximum_defensible_claim": "exploratory_preprocessing_plausibility_only",
               "gate": config["acceptance"], "comparison": rows,
               "input_metrics_sha256": sha256(bundle / "metrics.json"),
               "input_manifest_sha256": sha256(dataset / "manifest.json"),
               "report_script_sha256": sha256(Path(__file__)), "failed_attempts": failed,
               "outputs": {name: sha256(report_dir / name) for name in ("README.md", "report.html")}}
    (report_dir / "report_manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(outcome)
    print(table(headers, display))
    print(f"Report: {report_dir / 'README.md'}")
    print(f"Portable HTML: {report_dir / 'report.html'}")


if __name__ == "__main__":
    main()
