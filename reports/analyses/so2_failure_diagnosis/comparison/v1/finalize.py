#!/usr/bin/env python3
"""Publish the audited SO2 synthesis and curate its explicitly limited conclusion."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import yaml
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry

P = current_paths()
RID = "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf"
CAMPAIGN = "cmp_20260906_so2_failure_diagnosis"
STAGE = P.scratch_root / "active_runs" / RID / "posthoc_reports/so2_failure_comparison/v1"
FINAL = P.report_root / "analyses/so2_failure_diagnosis/comparison/v1"
BASE = P.report_root / "analyses/so2_failure_diagnosis/baselines/v1"
RESULT_ID = "res_so2_reconstruction_failure_diagnosis"
RESULT = P.result_root / "descriptive_analysis/relative_qkv" / RESULT_ID

def read(p): return json.loads(p.read_text())
def write(p, obj): p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")

def main():
    assert not FINAL.exists() and not RESULT.exists(), "Do not overwrite completed evidence"
    assert (STAGE / "report.md").is_file() and (STAGE / "independent_verification.json").is_file()
    audit = read(STAGE / "independent_verification.json")
    assert audit["status"] == "passed", audit
    visual = read(STAGE / "visual_verification.json")
    assert visual["status"] == "passed"
    assert visual["image_sha256"] == sha256_file(STAGE / "diagnostic_comparison.png")
    local = read(STAGE / "local_sources.json")
    for rel, expected in local["report_manifests"].items():
        f = P.report_root / rel; assert sha256_file(f) == expected
        for child, info in read(f)["files"].items(): assert sha256_file(f.parent / child) == info["sha256"]
    for source in local["diagnostic_sources"]:
        root = P.report_root if source["root"] == "report" else P.artifact_root
        assert sha256_file(root / source["path"]) == source["sha256"]
    shutil.copyfile(Path(__file__), STAGE / "finalize.py")
    write(STAGE / "manifest.json", {"schema": "so2_failure_diagnosis_synthesis_v1", "campaign_id": CAMPAIGN,
        "source_run_ids": local["source_run_ids"], "files": {str(f.relative_to(STAGE)): {"sha256": sha256_file(f), "size_bytes": f.stat().st_size}
        for f in sorted(STAGE.rglob("*")) if f.is_file() and f.name not in ("manifest.json", "_SUCCESS")}})
    write(STAGE / "_SUCCESS", {"manifest_sha256": sha256_file(STAGE / "manifest.json"), "completed_at": datetime.now(timezone.utc).isoformat()})
    FINAL.parent.mkdir(parents=True, exist_ok=True); os.rename(STAGE, FINAL)
    a = read(FINAL / "synthesis_metrics.json")["aggregate"]
    geometry = a["geometry"]; huber = a["huber_constant"]
    runtime = read(BASE / "summary.json")
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "schema_name": "bagm_result_record", "schema_version": 1, "revision": 1,
        "result_id": RESULT_ID, "title": "SO2 objective mismatch and matched spatial baselines; neural failure cause remains unresolved",
        "status": "verified", "outcome": "inconclusive", "experiment_type": "descriptive_analysis", "method_family": "relative_qkv",
        "lifecycle_stage": "exploratory_screen", "study_axis": "loss_and_spatial_baselines", "campaign_ids": [CAMPAIGN],
        "method": {"name": "Fixed-mask four-endpoint reconstruction comparison with six constant/spatial references and primary-source literature audit",
            "model_family": "Relative-QKV original, continued, recurrent and geometry; Huber-optimal constants and observed-only spatial means",
            "graph_context": "Original full graph (~227.5 incoming neighbors/cell globally within 500 micrometers) and fixed local k16 within 75 micrometers; no self edges",
            "masking_or_perturbation": "Exact original per-core uniform 0-through-1000 gene-mask seeds/checksums; no neural parameter intervention",
            "evaluation_design": "Exploratory transductive all-fit evaluation of 246063 cells, 1000 genes and 14 cores; one trained seed; equal-core metrics; no independent patient split",
            "implementation_version": "so2_diagnostic_baselines_v1 and so2_failure_diagnosis_synthesis_v1; executed-source snapshots and independent audit retained"},
        "conclusion": {
            "question": "Do loss preference or simple spatial averaging explain the limited reconstruction gains of the current SO2 models?",
            "estimand": "Common fixed-mask all-entry Huber/MSE and observed-positive standardized/log1p MSE; mathematical fitted constant-risk minimizers",
            "observed_result": f"999/1000 per-gene Huber constants are below the fitted log mean; mean standardized shift -0.194874. Geometry all-entry Huber {geometry['all_huber']:.9f} versus Huber constant {huber['all_huber']:.9f}. Geometry positive standardized MSE {geometry['positive_mse_z']:.9f}; local log mean {a['local_log_mean']['positive_mse_z']:.9f}, local count mean {a['local_count_mean']['positive_mse_z']:.9f}, original-full-graph log mean {a['full_graph_log_mean']['positive_mse_z']:.9f}. Full matched tables expose all-entry and zero/positive tradeoffs. Published methods differ in data scale, objective and task, and supply no directly comparable SOTA threshold.",
            "strongest_alternative_explanation": "Observed-positive errors condition on the realized outcome; conservative conditional predictions can be calibrated. Missing information, spatial mixing, decoder capacity and optimization remain competing explanations. Constants are all-fit target-derived references, and endpoints have different training durations.",
            "controls": ["Exact same masks, source counts and gene transform as the four verified endpoint evaluations.",
                "All-zero, gene-mean and convex Huber-optimal constants with root and risk verification.",
                "Observed-only local log/count means and original full-graph log mean; include observed zeros and prohibit hidden-target fallback leakage.",
                "29 focused metric/helper tests; representative pilots; all 14 cores; independent provenance, metric, source and registry audit."],
            "remaining_uncertainty": "No retrained cell-only control, isolated neural loss/head/graph ablation, data-size learning curve, mask-stratified neural evaluation, calibrated count likelihood, graph-use null, independent patient test or repeated trained seeds. Huber constant shrinkage does not prove the neural cause. Recurrent source finalization/catalog defect remains unchanged; predictions were independently verified. Literature sources include preliminary workshop and preprint results with distinct tasks.",
            "maximum_defensible_claim": "The objective demonstrably favors lower constants than MSE, and matched spatial baselines reveal metric-dependent performance. Current evidence cannot identify one neural failure cause, establish graph-specific predictive gain, or support a biological/causal claim."},
        "evidence": {"predictive_gain": {"status": "mixed", "summary": "Common-mask tables show modest endpoint gains and metric-specific tradeoffs against fixed constants/spatial means; no isolated neural graph gain."},
            **{key: {"status": "not_tested", "summary": text} for key,text in {
                "stability": "One trained seed and one fixed mask/core; descriptive paired cores are not seed or patient replication.",
                "faithfulness": "Numerical and leakage checks verify computation; no learned-attribution faithfulness experiment.",
                "null_calibration": "Simple predictors are baselines, not mechanism-breaking spatial nulls.",
                "patient_replication": "No patient-held-out evaluation; 14 cores are not asserted independent patients.",
                "external_support": "Methodological literature comparison supplies no independent biological validation of BAGM.",
                "perturbation_support": "No controlled biological perturbation or causal inference."}.items()}},
        "provenance": {"created_at_utc": now, "updated_at_utc": now,
            "git_commit": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
            "run_ids": local["source_run_ids"], "expected_seeds": [0], "included_seeds": [0], "expected_folds": [], "included_folds": [],
            "failed_or_excluded_runs": [
                {"run_id": "r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf", "reason": "Included verified recurrent endpoint despite original artifact-finalization failure and pre-existing missing checkpoint-catalog entry; source status remains failed."},
                {"run_id": "r_20260826T032515Z_52d16093_s000_f00_a01_149a165b", "reason": "Excluded failed duplicate continuation; not an independent model lineage or seed."}],
            "sources": [{"kind": "report", "root": "report", "path": str((root / 'manifest.json').relative_to(P.report_root)),
                "sha256": sha256_file(root / "manifest.json"), "verification_status": "verified"} for root in (BASE, FINAL)]},
        "files": []}
    RESULT.mkdir(parents=True)
    (RESULT / "result.yaml").write_text(yaml.safe_dump(record, sort_keys=False, width=110))
    (RESULT / "README.md").write_text(f"# SO2 reconstruction diagnostic\n\nOutcome: **inconclusive about the neural failure cause**; objective mismatch is supported in the constant-predictor class.\n\n{record['conclusion']['observed_result']}\n\n{record['conclusion']['maximum_defensible_claim']}\n\nSee the [full diagnostic report](../../../../reports/analyses/so2_failure_diagnosis/comparison/v1/report.md) for matched tables, literature, alternative explanations, tests and source provenance. This supplements the earlier nonzero comparison; it does not supersede its endpoint results.\n")
    for args in (("validate", "--verify-sources", "--verify-payloads"), ("catalog",), ("catalog", "--check")):
        subprocess.run([sys.executable, str(P.project_root / "scripts/results/manage_results.py"), *args], check=True)
    reg = Registry(initialize=False)
    with reg.connect() as c:
        raw = c.execute("SELECT config_json FROM campaigns WHERE campaign_id=?", (CAMPAIGN,)).fetchone()[0]
    cfg = json.loads(raw); cfg["completion"] = {"outcome": "inconclusive", "constant_loss_mismatch": "supported",
        "result_id": RESULT_ID, "report_manifest_sha256": sha256_file(FINAL / "manifest.json"), "cores": 14}
    reg.update_campaign(CAMPAIGN, config=cfg, expected_config_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        reason="All fixed-mask baselines and independent checks completed; evidence-limited synthesis curated", actor="codex", status="complete")
    print(json.dumps({"report": str(FINAL / "report.md"), "result": str(RESULT), "baseline_runtime_seconds": runtime["runtime_seconds"]}))

if __name__ == "__main__":
    main()
