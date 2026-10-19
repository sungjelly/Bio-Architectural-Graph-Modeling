"""Publish the verified, narrowly scoped preprocessing conclusion and catalog."""
from datetime import datetime, timezone
import json
from pathlib import Path
import yaml
from spatial_benchmark.paths import current_paths
from spatial_benchmark.result_catalog import sha256_file, validate_result_manifest, write_or_check_catalog


def main():
    paths=current_paths()
    run_id="r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410"
    failed="r_20261016T082320Z_660b87cb_s000_f00_a01_cd1c1544"
    run=paths.artifact_root/"runs/2026/10"/run_id
    report=paths.report_root/"analyses/so12_pearson_preprocessing"
    assert (run/"_SUCCESS").is_file()
    verification=json.loads((report/"independent_verification.json").read_text())
    assert verification["status"] == "passed"
    metrics=json.loads((run/"metrics.json").read_text())["methods"]
    identity=json.loads((run/"identity.json").read_text())
    baseline=metrics["log1p_raw"]
    for method in ("pearson","signed_log_pearson"):
        assert metrics[method]["heldout_core_median_depth_weighted_r2"] < baseline["heldout_core_median_depth_weighted_r2"]
        observed={v["slide"]:v["balanced_accuracy"] for v in metrics[method]["within_slide_vendor_label_recovery"] if v.get("control")=="observed_labels"}
        for row in baseline["within_slide_vendor_label_recovery"]:
            if row.get("control")=="observed_labels":
                assert observed[row["slide"]]>=row["balanced_accuracy"]-0.05
    result_id="res_so12_pearson_preprocessing_plausibility"
    dest=paths.result_root/"data_qc/pearson_preprocessing"/result_id
    dest.mkdir(parents=True,exist_ok=True)
    text="""# SO1 + SO2 Pearson preprocessing is plausible in a donor-separated pilot

Vendor-style quasi-Poisson Pearson residuals reduce the held-out-core depth-association
score from 0.3721 to 0.0331, with improvement in all eight diagnostic cores. Signed-log
Pearson reaches 0.0307; both pass the fixed provisional annotation guardrail. Plain
Pearson is the preferred next training baseline because the extra signed-log step
has inconsistent benefits and lower SO2 annotation recovery than plain Pearson.

This is shared preprocessing, not demonstrated batch correction or model-training
improvement. The twelve vendor letters are expression-derived and not independent
biological truth. Four diagnostic donors, one fixed split and one seed support only
an exploratory conclusion. Full-cell totals mean these exported features cannot be
masked afterward for a leakage-safe hidden-count benchmark.

The full report, plots, exact methods and derived dataset pointers are in
`reports/analyses/so12_pearson_preprocessing/`. Both transforms cover all 407,659 mapped
cells and 1,000 biological genes. Source observations and checkpoints were retained.
One CUDA initialization failure preceded the successful, scientifically unchanged retry.
"""
    (dest/"README.md").write_text(text)
    now=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
    observed=("Heldout-core median variance-weighted PC/depth R²: raw log 0.3720763, CP10k log 0.3260569, "
              "Pearson 0.0330754, signed-log Pearson 0.0307229. Pearson improves all eight heldout cores. "
              "SO1/SO2 balanced annotation accuracy: raw log 0.706224/0.734916, CP10k log 0.678751/0.729620, "
              "Pearson 0.716627/0.760118, signed-log Pearson 0.736762/0.726686. Both Pearson variants pass "
              "the fixed no-more-than-0.05 within-slide loss criterion. Extra signed log improves depth "
              "in only four of eight cores relative to Pearson. All output features finite.")
    controls=["Training-only pooled fitting; verified donor-disjoint split; all 28 core count/metadata alignment checks.",
              "Raw-log and CP10k-log baselines; nonstandard signed-log ablation; same balanced per-core sample and 32-PC protocol.",
              "Independent CPU formula and GPU comparison, heldout perturbation fit invariance, synthetic class/depth recovery.",
              "Within-training-core annotation permutation null; QC-pass depth sensitivity; independent exported-array replay and checksums."]
    sources=[]
    for filename in ("metrics.json","verification.json","resolved_config.json","smoke_checks.json","input_provenance.json","transform_diagnostics.json"):
        p=run/filename
        sources.append(dict(kind="run_artifact",root="artifact",path=p.relative_to(paths.artifact_root).as_posix(),
                            sha256=sha256_file(p),verification_status="verified",run_id=run_id))
    for filename in ("README.md","report.html","independent_verification.json","attempt1_failure.log","attempt2_execution.log"):
        p=report/filename
        sources.append(dict(kind="report",root="report",path=p.relative_to(paths.report_root).as_posix(),
                            sha256=sha256_file(p),verification_status="verified"))
    record={"schema_name":"bagm_result_record","schema_version":1,"revision":1,"result_id":result_id,
        "title":"Pearson preprocessing passes a joint-slide donor-separated plausibility pilot",
        "status":"verified","outcome":"supported","experiment_type":"data_qc","method_family":"pearson_preprocessing",
        "lifecycle_stage":"pilot","study_axis":"preprocessing","campaign_ids":["cmp_20261016_so12_pearson_preprocessing"],
        "method":{"name":"Pooled training-fitted quasi-Poisson Pearson preprocessing comparison", "model_family":"No neural model; fixed preprocessing, PCA and diagnostic logistic classifiers",
            "graph_context":"No new graph or cross-core edges; source independent core graphs referenced",
            "masking_or_perturbation":"Fully observed expression; synthetic heldout and hidden-gene perturbation checks only",
            "evaluation_design":"10 training donors, 4 diagnostic donors, 28 cores; 1000 sampled cells/core; all training cells fit transforms/PCA; one seed and one split",
            "implementation_version":"so12_quasipoisson_v1; vendor-source pin 41c499026d789ff35288de2ddc62f506a9d51cf0; common PCA recentered after clipping"},
        "conclusion":{"question":"Is pooled Pearson, with or without signed log, plausible shared preprocessing for SO1 and SO2?",
            "estimand":"Descriptive depth association and circular vendor-annotation consistency in heldout donors",
            "observed_result":observed,"strongest_alternative_explanation":"Depth can encode biology; compression changes PC variance weighting; vendor labels derive from expression; slide and disease composition are confounded.",
            "controls":controls,"remaining_uncertainty":"No neural training or batch-removal validation; only four diagnostic donors, one split/seed; vendor letters have unverified biological meanings; no external or independent biological validation. Full-cell totals leak hidden counts if masking is applied afterward.",
            "maximum_defensible_claim":"Both Pearson variants are plausible fully observed shared preprocessing candidates; prefer plain Pearson as the next training baseline and retain signed log as an ablation. This does not establish batch correction, improved masked prediction or preserved biological mechanisms."},
        "evidence":{
            "predictive_gain":{"status":"not_tested","summary":"No masked-expression or neural graph model comparison."},
            "stability":{"status":"mixed","summary":"Pearson versus raw log improves depth in all eight cores and QC-pass subset; extra signed-log benefit occurs in four of eight cores; no split/seed replication."},
            "faithfulness":{"status":"not_applicable","summary":"No model attribution or inferred interaction tested."},
            "null_calibration":{"status":"mixed","summary":"One within-training-core label permutation produces lower agreement than observed labels, but is not a calibrated significance test."},
            "patient_replication":{"status":"not_tested","summary":"Four heldout donor groups provide descriptive consistency only; no independent cohort or confirmatory replication."},
            "external_support":{"status":"not_tested","summary":"Vendor method source supports implementation, not independent biological validation."},
            "perturbation_support":{"status":"not_tested","summary":"Synthetic numerical controls only; no biological intervention."}},
        "provenance":{"created_at_utc":now,"updated_at_utc":now,"git_commit":identity["git_commit"],"run_ids":[run_id,failed],
            "expected_seeds":[0],"included_seeds":[0],"expected_folds":[0],"included_folds":[0],
            "failed_or_excluded_runs":[{"run_id":failed,"reason":"Concurrent CUDA eigensolver lazy-initialization failure; retried with serial initialization and unchanged scientific protocol; no failed-attempt metrics used."}],"sources":sources},
        "files":[]}
    (dest/"result.yaml").write_text(yaml.safe_dump(record,sort_keys=False,allow_unicode=True))
    validate_result_manifest(dest/"result.yaml",paths=paths,verify_payloads=True,verify_sources=True)
    catalog=write_or_check_catalog(paths=paths,verify_payloads=True,verify_sources=False)
    print(json.dumps({"result":str(dest),"validated":True,"catalog_rebuilt":True}))


if __name__=="__main__":
    main()
