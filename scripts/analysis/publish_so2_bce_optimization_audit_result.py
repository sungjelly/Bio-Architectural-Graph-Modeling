#!/usr/bin/env python3
"""Publish a completed, checksum-bound optimization audit as a curated result.

Contract: experiments/campaigns/cmp_20261001_so2_bce_optimization_audit/README.md
and docs/result_management.md. Only a complete confirmation report, chained
to complete regularization and grid reports, is eligible. Smoke and screen
artifacts alone cannot be promoted. This script performs no model evaluation.

Run from the repository root with BAGM_* roots set as needed::

    PYTHONPATH=src:. /venv/main/bin/python \
      scripts/analysis/publish_so2_bce_optimization_audit_result.py \
      --report reports/analyses/so2_bce_optimization_audit/confirmation_v1/summary.json \
      --result-id res_so2_bce_optimization_audit --check-only

Omit --check-only to publish after reviewing the completed report. Existing
result IDs fail closed. New metadata is schema-validated before an atomic
directory publication; catalogs are regenerated from all existing records.
No existing result record is edited or superseded by this command.

Verification::

    PYTHONPATH=src:. /venv/main/bin/python -m pytest -q \
      tests/unit/spatial_benchmark/test_publish_so2_bce_optimization_audit_result.py
    PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate --verify-payloads --verify-sources
    PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog --check
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.result_catalog import (
    sha256_file, validate_result_manifest, validate_result_tree, write_or_check_catalog,
)
from spatial_benchmark.run_archive import RunArchive, _rename_no_replace, verify_run_bundle
from spatial_benchmark.so2_bce_optimization_audit import CAMPAIGN, validate_configuration

SCHEMA = "so2_bce_optimization_audit_report_v1"
METRIC = "validation_pooled_masked_binary_cross_entropy"
SETTINGS = ("batch_mode", "target_batch_size", "learning_rate", "dropout", "weight_decay")
INCUMBENT = ("target_window", 1024, 1e-4, .1, 1e-5)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def finite(value: Any) -> float:
    number = float(value)
    require(math.isfinite(number) and number >= 0, "BCE must be finite and nonnegative")
    return number


def setting(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[key] for key in SETTINGS)


def analyze_confirmation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Independently apply the fixed-budget paired-seed decision criterion."""
    require(len(rows) == 4, "Confirmation requires exactly four distinct completed runs")
    require(len({r["run_id"] for r in rows}) == 4, "Repeated confirmation run")
    require(all(r["profile"] == "confirmation" and r["status"] == "completed" for r in rows),
            "Smoke, screen, failed, or active runs cannot establish confirmation")
    require(all(r["final_epoch"] == 32 and r["endpoint_epochs"] == [29, 30, 31, 32]
                and r["fold"] == 0 and type(r["seed"]) is int for r in rows),
            "Confirmation requires the declared 32-epoch endpoint and fixed split")
    variants = {setting(r) for r in rows}
    require(len(variants) == 2 and INCUMBENT in variants, "Need incumbent and one distinct comparator")
    comparator = next(v for v in variants if v != INCUMBENT)
    by_key = {(setting(r), r["seed"]): r for r in rows}
    require(set(by_key) == {(v, s) for v in variants for s in (0, 1)},
            "Both distinct variants require model seeds zero and one")
    effects = []
    for seed in (0, 1):
        before, after = by_key[(INCUMBENT, seed)], by_key[(comparator, seed)]
        effects.append({"seed": seed, "incumbent_run_id": before["run_id"],
            "comparator_run_id": after["run_id"],
            "bce_reduction": finite(before["endpoint_bce"]) - finite(after["endpoint_bce"])})
    mean_effect = math.fsum(r["bce_reduction"] for r in effects) / 2
    supported = mean_effect >= .002 and all(r["bce_reduction"] > 0 for r in effects)
    late = {r["run_id"]: finite(r["previous_window_bce"]) - finite(r["endpoint_bce"]) for r in rows}
    unconverged = any(value > .0005 for value in late.values())
    return {"paired_seed_effects": effects, "mean_paired_bce_reduction": mean_effect,
        "material_gain_supported": supported, "fixed_budget_outcome": "supported" if supported else "negative",
        "any_endpoint_unconverged": unconverged,
        "eventual_performance_outcome": "inconclusive" if unconverged else "not_established_beyond_fixed_budget",
        "late_bce_improvement_by_run": late,
        "comparator_setting": dict(zip(SETTINGS, comparator)), "confirmation_runs": list(rows)}


def analyze_complete_coverage(rows: Sequence[Mapping[str, Any]], coverage: Sequence[Mapping[str, Any]],
                              failures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A diagnosed failed slot can complete the audit, never a numeric effect."""
    require(len(coverage) == 4, "Need four prespecified confirmation slots")
    variants = {setting(r) for r in coverage}
    require(len(variants) == 2 and INCUMBENT in variants, "Confirmation variants changed")
    require({(setting(r), r["seed"]) for r in coverage} == {(v, s) for v in variants for s in (0, 1)},
            "Both variants must retain both expected seeds")
    require(all(r["profile"] == "confirmation" and r["fold"] == 0 and r.get("resolved") is True for r in coverage),
            "Only fully resolved confirmation coverage is eligible")
    successful = {r["run_id"]: r for r in rows}
    failed_ids = {r["run_id"] for r in failures}
    expected_success = {r["completed_run_id"] for r in coverage if r.get("completed_run_id")}
    require(set(successful) == expected_success and len(successful) == len(rows), "Successful confirmation coverage differs")
    for slot in coverage:
        if slot.get("completed_run_id"):
            observed = successful[slot["completed_run_id"]]
            require(setting(observed) == setting(slot) and observed["seed"] == slot["seed"],
                    "Successful run occupies the wrong planned confirmation slot")
        else:
            accepted = slot.get("accepted_failure_run_ids", [])
            require(accepted and set(accepted).issubset(failed_ids), "Missing slot lacks verified accepted failure")
    if len(rows) == 4:
        return analyze_confirmation(rows)
    require(all(r["profile"] == "confirmation" and r["status"] == "completed" and r["final_epoch"] == 32
                and r["endpoint_epochs"] == [29, 30, 31, 32] for r in rows), "Invalid partial confirmation endpoint")
    late = {r["run_id"]: finite(r["previous_window_bce"]) - finite(r["endpoint_bce"]) for r in rows}
    return {"paired_seed_effects": [], "mean_paired_bce_reduction": None, "material_gain_supported": False,
        "fixed_budget_outcome": "inconclusive", "eventual_performance_outcome": "inconclusive",
        "any_endpoint_unconverged": any(v > .0005 for v in late.values()),
        "late_bce_improvement_by_run": late, "confirmation_runs": list(rows),
        "comparator_setting": dict(zip(SETTINGS, next(v for v in variants if v != INCUMBENT))),
        "failure_interpretation": "Missing successful confirmation slots prohibit the paired-seed comparison; resource failure does not establish predictive inferiority."}


def confined(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root), "Source path escapes its declared BAGM root")
    return resolved


def load_report(path: Path, paths: Any, expected_stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = confined(paths.report_root, path)
    require(path.name == "summary.json", "Report input must be a finalized summary.json")
    manifest_path = path.parent / "manifest.json"
    manifest = read_json(manifest_path)
    files = manifest.get("files", {})
    require(all(name in files for name in ("summary.json", "report.md", "report_builder.py")),
            "Report manifest does not bind required review artifacts")
    for name, digest in files.items():
        candidate = confined(path.parent, path.parent / name)
        require(candidate.is_file() and sha256_file(candidate) == digest, "Report checksum mismatch")
    report = read_json(path)
    require(report.get("schema") == SCHEMA and report.get("campaign_id") == CAMPAIGN
            and report.get("stage") == expected_stage and report.get("required_coverage_complete") is True,
            "Report stage is incomplete or belongs to another campaign")
    expected = report["expected_coverage"]
    ids = [row["completed_run_id"] for row in expected if row.get("completed_run_id")]
    require(expected and all(row.get("resolved") is True and
            (row.get("completed_run_id") or row.get("accepted_failure_run_ids")) for row in expected)
            and len(set(ids)) == len(ids)
            and set(ids) == set(report["stage_run_ids"]), "Required report coverage is missing or duplicated")
    accepted = {r["run_id"]: r for r in report.get("accepted_failure_runs", [])}
    other = {r["run_id"]: r for r in report["other_attempts"]}
    completed = {r["run_id"]: r for r in report["completed_runs"]}
    for slot in expected:
        require(not slot.get("completed_run_id") or slot["completed_run_id"] in completed,
                "Resolved slot does not identify a completed report run")
        for run_id in slot.get("accepted_failure_run_ids", []):
            require(run_id in accepted and accepted[run_id]["failure_type"] in ("nonfinite", "resource")
                    and run_id in other and other[run_id]["status"] == "failed"
                    and other[run_id].get("diagnosed_failure_type") == accepted[run_id]["failure_type"]
                    and setting(other[run_id]) == setting(slot) and other[run_id]["seed"] == slot["seed"],
                    "Accepted failure lacks matching diagnosed planned-slot evidence")
    sources = [{"kind": "report", "root": "report", "path": item.relative_to(paths.report_root.resolve()).as_posix(),
        "sha256": sha256_file(item), "verification_status": "verified",
        "description": f"Completed {expected_stage} audit report {item.name}"}
        for item in (path, manifest_path, path.parent / "report.md")]
    return report, sources


def load_report_chain(path: Path, paths: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report, sources = load_report(path, paths, "confirmation")
    require(report.get("complete") is True and report.get("budget_branch") == "full",
            "Only the completed full audit is eligible for promotion")
    child = report
    regularization = None
    for stage in ("regularization", "grid"):
        ref = child["parent_report"]
        parent_path = Path(ref["path"])
        if not parent_path.is_absolute():
            parent_path = paths.report_root / parent_path
        require(sha256_file(parent_path) == ref["sha256"]
                and sha256_file(parent_path.parent / "manifest.json") == ref["manifest_sha256"],
                "Parent report binding changed")
        parent, parent_sources = load_report(parent_path, paths, stage)
        sources.extend(parent_sources)
        if stage == "regularization":
            regularization = parent
        all_ids = {r["run_id"] for r in report["completed_runs"] + report["other_attempts"]}
        require(set(parent["stage_run_ids"]).issubset(all_ids), "Final report omits a required parent-stage run")
        child = parent
    # The original screen is nine declared batch/LR settings plus paired cores.
    expected_grid = {(b, lr, "target_window") for b in (512, 1024, 2048) for lr in (3e-5, 1e-4, 3e-4)}
    expected_grid.add((None, 1e-4, "core_pair"))
    require({(r["target_batch_size"], r["learning_rate"], r["batch_mode"]) for r in child["expected_coverage"]}
            == expected_grid and len(child["expected_coverage"]) == 10,
            "Full grid and paired-core reference coverage are required")
    winner = child["grid_selected_setting"]
    expected_regularization = [winner, {**winner, "dropout": 0.}, {**winner, "dropout": .3}, {**winner, "weight_decay": .01}]
    if child["boundary_setting"]:
        expected_regularization.append(child["boundary_setting"])
    require(regularization is not None and len(regularization["expected_coverage"]) == len(expected_regularization)
            and {setting(r) for r in regularization["expected_coverage"]} == {setting(r) for r in expected_regularization},
            "Single-factor regularization/boundary coverage differs from declared grid decision")
    return report, sources


def verify_execution_identity(run_id: str, status: str, root: Path, registry: Any) -> None:
    """Archive markers and registry lifecycle use different status vocabularies."""
    marker_status = {"completed": "success", "failed": "failed", "cancelled": "failed", "pruned": "pruned"}
    require(status in marker_status, "Unfinished run cannot be a final source")
    verification = verify_run_bundle(root)
    require(verification["status"] == marker_status[status], "Report and immutable run status differ")
    registered = registry.get_run(run_id)
    require(registered is not None and registered["status"] == status
            and registered["campaign_id"] == CAMPAIGN
            and Path(registered["artifact_path"]).resolve() == root.resolve(),
            "Registry lifecycle, campaign or canonical artifact path differs")


def verify_sources(report: Mapping[str, Any], paths: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """Bind report scalars to immutable execution records and checkpoint catalog."""
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3", initialize=False)
    sources: list[dict[str, Any]] = []
    rows = list(report["completed_runs"]) + list(report["other_attempts"])
    require(len({r["run_id"] for r in rows}) == len(rows), "Duplicate campaign run")
    confirmation = []
    failures = []
    for row in rows:
        run_id = row["run_id"]
        require(row["status"] in ("completed", "failed", "cancelled", "pruned"),
                "An unfinished campaign attempt prevents final promotion")
        root = RunArchive.artifact_path_for(run_id, paths)
        confined(paths.artifact_root, root)
        verify_execution_identity(run_id, row["status"], root, registry)
        config = yaml.safe_load((root / "config.resolved.yaml").read_text())
        validate_configuration(config)
        trainer = config["trainer"]
        actual = {"batch_mode": trainer["optimizer_batch_mode"], "target_batch_size": trainer["target_cells_per_optimizer_update"],
            "learning_rate": trainer["learning_rate"], "weight_decay": trainer["weight_decay"],
            "dropout": config["model"]["dropout"], "seed": config["seed"], "fold": config["fold"],
            "profile": config["metadata"]["execution_profile"]}
        require(all(row[k] == v for k, v in actual.items()), "Report settings differ from immutable run configuration")
        for relative, digest in row["source_hashes"].items():
            source = confined(root, root / relative)
            require(sha256_file(source) == digest, "Report run-source checksum mismatch")
        marker = "_SUCCESS" if row["status"] == "completed" else ("_PRUNED" if (root / "_PRUNED").exists() else "_FAILED")
        for name in (marker, "provenance/artifact_checksums.json", "config.resolved.yaml"):
            source = root / name
            sources.append({"kind": "run_artifact", "root": "artifact", "run_id": run_id,
                "path": source.relative_to(paths.artifact_root).as_posix(), "sha256": sha256_file(source),
                "verification_status": "verified", "description": f"{row['profile']} {row['status']} audit run; smoke is readiness evidence only"})
        if row["status"] != "completed":
            failures.append({"run_id": run_id, "reason": str(row.get("failure_category") or row["status"])})
            continue
        checkpoint = registry.list_checkpoint_catalog(run_id=run_id, role="best", verification_status="verified")
        require(len(checkpoint) == 1 and checkpoint[0]["sha256"] == sha256_file(root / "checkpoints/best.ckpt"),
                "Retained best checkpoint lacks verified catalog binding")
        if run_id not in report["stage_run_ids"]:
            continue
        with (root / "results/epoch_metrics.csv").open(newline="") as handle:
            history = list(csv.DictReader(handle))
        require([int(r["global_epoch"]) for r in history] == list(range(1, 33)), "Confirmation history must cover exactly32epochs")
        endpoint = math.fsum(finite(history[i-1][METRIC]) for i in range(29, 33)) / 4
        previous = math.fsum(finite(history[i-1][METRIC]) for i in range(25, 29)) / 4
        stored = read_json(root / "results/optimization_audit_endpoint.json")
        require(stored["endpoint_epochs"] == [29, 30, 31, 32]
                and abs(endpoint - finite(stored["mean_validation_bce"])) <= 1e-12
                and abs(endpoint - finite(row["endpoint_bce"])) <= 1e-12,
                "Reported confirmation endpoint differs from immutable history")
        confirmation.append({"run_id": run_id, "status": "completed", **actual,
            "final_epoch": 32, "endpoint_epochs": [29, 30, 31, 32], "endpoint_bce": endpoint,
            "previous_window_bce": previous})
    return confirmation, sources, failures


def early_update_comparison(report: Mapping[str, Any], paths: Any) -> dict[str, Any] | None:
    """Preserve the separate matched-exposure early batching comparison."""
    source = report.get("early_update_regime_comparison")
    if source is None:
        return None
    values = {}
    f1 = {}
    counts = {}
    for key in ("paired_core_run_id", "target_window_incumbent_run_id"):
        root = RunArchive.artifact_path_for(source[key], paths)
        with (root / "results/epoch_metrics.csv").open(newline="") as handle:
            history = list(csv.DictReader(handle))
        require([int(r["global_epoch"]) for r in history] == list(range(1, 9)), "Early reference must cover eight epochs")
        values[key] = math.fsum(finite(history[i-1][METRIC]) for i in (7, 8)) / 2
        f1[key] = math.fsum(finite(history[i-1]["validation_detection_f1"]) for i in (7, 8)) / 2
        counts[key] = (int(history[-1]["cumulative_optimizer_updates"]), sum(int(r["train_target_cells"]) for r in history))
    effect = values["paired_core_run_id"] - values["target_window_incumbent_run_id"]
    f1_effect = f1["target_window_incumbent_run_id"] - f1["paired_core_run_id"]
    require(abs(effect - float(source["bce_reduction_with_target_windows"])) <= 1e-12
            and abs(f1_effect - float(source["f1_increase_with_target_windows"])) <= 1e-12
            and counts["paired_core_run_id"] == (48, 1669568)
            and counts["target_window_incumbent_run_id"] == (1672, 1669568), "Early reference effect or budget differs")
    return {"paired_core_run_id": source["paired_core_run_id"],
        "target_window_incumbent_run_id": source["target_window_incumbent_run_id"],
        "endpoint_epochs": [7, 8], "bce_reduction_with_target_windows": effect,
        "f1_increase_with_target_windows": f1_effect,
        "paired_core_optimizer_steps": 48, "target_window_optimizer_steps": 1672,
        "training_target_visits_each": 1669568,
        "interpretation_limit": "The combined target-window/update-regimen change at eight epochs; unequal optimizer steps do not establish equal-step or eventual superiority."}


def result_record(result_id: str, report: Mapping[str, Any], decision: Mapping[str, Any],
                  sources: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    effects = decision["paired_seed_effects"]
    observed = (f"Across model seeds 0 and 1, the incumbent-minus-comparator reduction in mean validation BCE at epochs 29–32 "
        f"was {effects[0]['bce_reduction']:.8f} and {effects[1]['bce_reduction']:.8f}; mean reduction "
        f"{decision['mean_paired_bce_reduction']:.8f}. The prespecified material-gain criterion was "
        f"{'met' if decision['material_gain_supported'] else 'not met'}. "
        f"Endpoint convergence diagnostic: {'still improving in at least one run' if decision['any_endpoint_unconverged'] else 'no run exceeded the declared late-improvement threshold'}.") if effects else (
        f"Only {len(decision['confirmation_runs'])} of the four planned confirmation runs produced a successful 32-epoch endpoint. "
        "Preserved, explicitly accepted failures resolve the remaining slots operationally. The paired-seed predictive comparison is inconclusive; no aggregate gain is estimated.")
    remaining = ("Only one held-out validation donor, two optimization seeds, repeated validation selection and no sealed test. "
        "Exposure is matched; optimizer updates differ across batch sizes. "
        + ("At least one endpoint remains unconverged, so eventual-performance comparison is inconclusive. " if decision["any_endpoint_unconverged"] else "The convergence diagnostic does not establish global optimality. ")
        + "Graph-specific gain, attribution stability/faithfulness, biological nulls, patient replication and mechanism remain untested.")
    early = decision.get("early_update_regime_comparison")
    if early is not None:
        observed += (f" Separately, the eight-epoch target-window incumbent reduced BCE versus paired cores by "
            f"{early['bce_reduction_with_target_windows']:.8f}, with F1 change {early['f1_increase_with_target_windows']:+.8f}, "
            "at equal 1,669,568 target visits and 1,672 versus 48 optimizer updates. "
            "This early update-regimen result is distinct from the tuning comparison and does not establish eventual superiority.")
    unsupported = {"status": "not_tested", "summary": "Not tested by this optimization audit."}
    return {"schema_name": "bagm_result_record", "schema_version": 1, "revision": 1,
        "result_id": result_id, "title": "SO2 BCE optimization audit at matched target exposure",
        "status": "verified", "outcome": decision["fixed_budget_outcome"], "experiment_type": "predictive_benchmark",
        "method_family": "relative_qkv", "lifecycle_stage": "validation_confirmation", "study_axis": "optimization",
        "campaign_ids": [CAMPAIGN], "method": {"name": "Target-isolated binary-expression BCE optimization audit",
            "model_family": "Geometry-modulated relative-QKV graph transformer",
            "graph_context": "Unchanged clean within-core neighbor expression with permitted geometry and imaging/morphology",
            "masking_or_perturbation": "Partial-gene per-cell masking; every training cell once per epoch; fixed validation masks",
            "evaluation_design": "Exploratory grid and regularization screen, then fresh incumbent/comparator model seeds 0/1 at 32 epochs on one held-out donor group. Fold 0 denotes this single fixed split, not independent cross-validation folds.",
            "implementation_version": report["git_commit"]},
        "conclusion": {"question": "Do the declared batch, learning-rate or regularization settings improve fixed-mask SO2 BCE prediction?",
            "estimand": "Partial-gene binary-expression reconstruction from observed same-cell genes, imaging/morphology and permitted clean neighborhood context",
            "observed_result": observed,
            "strongest_alternative_explanation": "Validation selection and one-donor distribution may explain apparent gains; gene prevalence and observed same-cell features may explain prediction without useful spatial context.",
            "controls": ["Complete batch/LR screen and single-factor dropout/weight-decay controls with immutable failed attempts retained",
                "Fresh incumbent and comparator seeds0/1, fixed masks/order and equal target exposure; terminal mean chosen independently of best checkpoint",
                "One-GPU paired-core reference, exact train-fitted gene prevalence validation reference and selected-best fixed-training diagnostics",
                "Coverage, finite-gradient, resource, selected-best replay and artifact/checkpoint checksum gates"],
            "remaining_uncertainty": remaining,
            "maximum_defensible_claim": "Exploratory validation-selected optimization behavior at the tested32-epoch budget on this split; no claim of global optimality, graph-specific information, independent patient replication or causality."},
        "evidence": {"predictive_gain": {"status": "supported" if decision["material_gain_supported"] else ("failed" if effects else "mixed"),
            "summary": "Prespecified optimization gain over the matched incumbent " + ("passed." if decision["material_gain_supported"] else ("did not pass." if effects else "could not be resolved because confirmation runs failed.")) + " This does not test graph-specific gain."},
            "stability": {"status": "supported" if effects and all(x["bce_reduction"] > 0 for x in effects) else "mixed",
                "summary": "Direction of optimization effect across two model seeds only; attribution and biological stability were not evaluated."},
            **{key: dict(unsupported) for key in ("faithfulness", "null_calibration", "patient_replication", "external_support", "perturbation_support")}},
        "provenance": {"created_at_utc": now, "updated_at_utc": now, "git_commit": report["git_commit"],
            "run_ids": sorted({s["run_id"] for s in sources if "run_id" in s}), "expected_seeds": [0, 1],
            "included_seeds": sorted({r["seed"] for r in decision["confirmation_runs"]}), "expected_folds": [0], "included_folds": [0] if decision["confirmation_runs"] else [],
            "failed_or_excluded_runs": failures, "sources": sources}, "files": []}


def publish(report_path: Path, result_id: str, *, paths: Any, check_only: bool = False) -> dict[str, Any]:
    require(bool(re.fullmatch(r"res_[a-z0-9][a-z0-9_]*", result_id)), "Unsafe result ID")
    report, sources = load_report_chain(report_path, paths)
    rows, run_sources, failures = verify_sources(report, paths)
    decision = analyze_complete_coverage(rows, report["expected_coverage"], failures)
    decision["early_update_regime_comparison"] = early_update_comparison(report, paths)
    for key in ("material_gain_supported", "fixed_budget_outcome", "any_endpoint_unconverged", "eventual_performance_outcome"):
        require(report[key] == decision[key], f"Report decision mismatch: {key}")
    mean_effect = decision["mean_paired_bce_reduction"]
    require((report["mean_paired_bce_reduction"] is None if mean_effect is None else
             abs(float(report["mean_paired_bce_reduction"]) - mean_effect) <= 1e-12),
            "Report paired-seed effect differs from immutable histories")
    payload = result_record(result_id, report, decision, sources + run_sources, failures)
    destination = paths.result_root / "predictive_benchmark/relative_qkv" / result_id
    require(not destination.exists(), "Result ID already exists; use a new explicitly reviewed revision")
    paths.result_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bce-result-validation-", dir=paths.result_root.parent) as staging:
        staging_root = Path(staging)
        staged_paths = replace(paths, result_root=staging_root)
        directory = staging_root / "predictive_benchmark/relative_qkv" / result_id
        (directory / "tables").mkdir(parents=True)
        numeric = directory / "tables/confirmation.json"
        numeric.write_text(json.dumps(decision, sort_keys=True, indent=2, allow_nan=False) + "\n")
        payload["files"] = [{"path": "tables/confirmation.json", "role": "numeric_source", "sha256": sha256_file(numeric),
                             "size_bytes": numeric.stat().st_size, "description": "Four run-level endpoint means and independently recomputed paired-seed criterion"}]
        readme = (f"# {payload['title']}\n\nOutcome: **{payload['outcome']}** at the fixed32-epoch budget.\n\n"
            + payload["conclusion"]["observed_result"] + "\n\n" + payload["conclusion"]["remaining_uncertainty"]
            + "\n\n" + payload["conclusion"]["maximum_defensible_claim"] + "\n\n"
            + "Detailed report and all immutable run sources are checksum-bound in result.yaml; tables/confirmation.json contains the four run-level endpoints.\n")
        (directory / "README.md").write_text(readme)
        (directory / "result.yaml").write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))
        validate_result_manifest(directory / "result.yaml", paths=staged_paths, verify_payloads=True, verify_sources=True)
        if not check_only:
            paths.result_root.mkdir(parents=True, exist_ok=True)
            with (paths.result_root / ".promotion.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                existing = validate_result_tree(paths=paths)
                require(not any(r.payload["result_id"] == result_id for r in existing), "Result ID exists in another directory")
                destination.parent.mkdir(parents=True, exist_ok=True)
                _rename_no_replace(directory, destination)
                write_or_check_catalog(paths=paths)
                write_or_check_catalog(paths=paths, check=True)
    return {"valid": True, "published": not check_only, "result_directory": str(destination),
            "outcome": decision["fixed_budget_outcome"], "mean_paired_bce_reduction": decision["mean_paired_bce_reduction"],
            "any_endpoint_unconverged": decision["any_endpoint_unconverged"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--result-id", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    paths = current_paths()
    path = args.report if args.report.is_absolute() else paths.project_root / args.report
    print(json.dumps(publish(path, args.result_id, paths=paths, check_only=args.check_only), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
