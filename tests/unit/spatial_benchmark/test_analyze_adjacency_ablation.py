from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

from spatial_benchmark.adjacency_ablation import build_five_fold_splits


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/analysis/analyze_adjacency_ablation.py"
_SPEC = importlib.util.spec_from_file_location(
    "analyze_adjacency_ablation_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

AdjacencyAnalysisError = _MODULE.AdjacencyAnalysisError


def _manifest() -> dict[str, Any]:
    folds = {
        str(fold.fold_index): {
            **fold.to_dict(),
            "preprocessing_file_sha256": "d" * 64,
        }
        for fold in build_five_fold_splits()
    }
    cores = {
        alias: {
            "mask_seeds": [1000 + replicate for replicate in range(3)],
            "mask_realization_checksums": [
                f"{index:x}" * 64 for index in range(1, 4)
            ],
        }
        for alias in _MODULE.ALIASES
    }
    return {
        "content_sha256": "c" * 64,
        "files": {},
        "folds": folds,
        "cores": cores,
    }


def _run_record(
    tmp_path: Path, *, condition: str, invalid_inputs: bool = False
) -> dict[str, Any]:
    root = tmp_path / condition
    (root / "diagnostics").mkdir(parents=True)
    (root / "provenance").mkdir()
    fold = _manifest()["folds"]["0"]
    digest = "a" * 64
    preprocessing = "d" * 64
    pairing = {
        "fold": 0,
        "model_seed": 0,
        "condition": condition,
        "train_aliases": fold["train_aliases"],
        "validation_aliases": fold["validation_aliases"],
        "test_aliases": fold["test_aliases"],
        "split_disjoint": True,
        "preprocessing_fit_aliases": fold["train_aliases"],
        "preprocessing_sha256": preprocessing,
        "initial_state_sha256": digest,
        "training_mask_schedule_sha256": digest,
        "validation_selection_mask_schedule_sha256": digest,
        "validation_evaluation_mask_identity_sha256": digest,
        "test_evaluation_mask_identity_sha256": digest,
        "model_inputs": (
            ["clean_expression", "binary_mask"]
            if invalid_inputs
            else ["masked_standardized_log1p_counts", "binary_mask"]
        ),
        "prohibited_inputs_absent": [
            "cell_type", "cluster", "niche", "donor", "core",
            "tissue_stage", "coordinates", "target_derived_library_size",
        ],
        "masked_neighbor_inputs_only": True,
    }
    scientific = {
        "prepared_manifest": "prepared/manifest.json",
        "prepared_manifest_sha256": "b" * 64,
        "prepared_content_sha256": "c" * 64,
        "preprocessing_sha256": preprocessing,
        "frozen_contract_sha256": _MODULE.CONTRACT_SHA256,
        "fold": 0,
        "condition": condition,
    }
    (root / "diagnostics/pairing_and_leakage_audit.json").write_text(
        json.dumps(pairing), encoding="utf-8"
    )
    (root / "provenance/scientific_inputs.json").write_text(
        json.dumps(scientific), encoding="utf-8"
    )
    summary = {
        "status": "success",
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "stage": "primary",
        "condition": condition,
        "fold": 0,
        "model_seed": 0,
        "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
        "finite_metrics": True,
        "coverage_complete": True,
        "update_count_verified": True,
        "all_gradients_finite": True,
        "deterministic_algorithms": True,
        "conclusion_eligible": True,
        "generalization_estimate": True,
        "completed_global_epochs": 80,
        "optimizer_steps": 560,
        "prepared_content_sha256": "c" * 64,
        "prepared_manifest_sha256": "b" * 64,
        "preprocessing_sha256": preprocessing,
        "initial_state_sha256": digest,
        "training_mask_schedule_sha256": digest,
        "validation_selection_mask_schedule_sha256": digest,
        "validation_mask_identity_sha256": digest,
        "test_mask_identity_sha256": digest,
    }
    config = {
        "dataset": {
            "prepared_manifest": "prepared/manifest.json",
            "prepared_manifest_sha256": "b" * 64,
        },
        "features": {"node_features": ["masked_expression", "binary_mask"]},
        "graph": {"adjacency_condition": condition, "k": 12},
        "masking": {"evaluation_replicates": 3},
        "preprocessing": {"feature_selection": "none"},
        "model": {"expected_parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT},
        "trainer": {"max_epochs": 80, "expected_total_optimizer_updates": 560},
        "evaluation": {"primary_metric": "val/masked_huber"},
        "seed": 0,
        "fold": 0,
    }
    return {
        "root": root,
        "run_id": f"r_{condition}",
        "summary": summary,
        "config": config,
        "config_sha256": "f" * 64,
        "stage": "primary",
        "condition": condition,
        "fold": 0,
        "seed": 0,
        "audit_errors": [],
    }


def test_scientific_preflight_requires_exact_inputs_pairing_and_budget(
    tmp_path: Path,
) -> None:
    runs = [
        _run_record(tmp_path, condition="spatial"),
        _run_record(tmp_path, condition="isolated"),
    ]
    errors, _ = _MODULE._scientific_run_audit(
        runs, manifest=_manifest(), paired_conditions=_MODULE.CONDITIONS
    )
    assert errors == []

    invalid = [
        _run_record(tmp_path / "bad", condition="spatial", invalid_inputs=True),
        _run_record(tmp_path / "bad", condition="isolated"),
    ]
    invalid[1]["summary"]["optimizer_steps"] = 559
    errors, _ = _MODULE._scientific_run_audit(
        invalid, manifest=_manifest(), paired_conditions=_MODULE.CONDITIONS
    )
    assert any("model_inputs" in error for error in errors)
    assert any("completed_560_updates" in error for error in errors)
    assert any("paired summary optimizer_steps differs" in error for error in errors)


def test_scientific_preflight_binds_all_cpu_recovery_provenance(
    tmp_path: Path,
) -> None:
    runs = [
        _run_record(tmp_path, condition="spatial"),
        _run_record(tmp_path, condition="isolated"),
    ]
    for index, run in enumerate(runs):
        expected = {
            "execution_device": "cpu",
            "execution_mode": "cpu_hardware_recovery",
            "queue_job_id": f"q_retry_{index}",
            "torch_intraop_threads": 4,
            "torch_interop_threads": 1,
            "hardware_recovery_used": True,
            "recovery_contract_reference": "recovery_contract.yaml",
            "recovery_contract_sha256": "1" * 64,
            "recovery_plan_reference": "recovery_plan.json",
            "recovery_plan_checksum": "2" * 64,
            "recovery_plan_file_sha256": "3" * 64,
            "recovery_worker_slot": index,
            "recovery_root_job_id": f"q_root_{index}",
        }
        run["required_recovery_audit"] = expected
        run["required_execution_device"] = "cpu"
        run["attempt"] = 2
        run["job_id"] = expected["queue_job_id"]
        run["config"]["attempt"] = 2
        run["summary"].update(expected)
        for relative in (
            "diagnostics/pairing_and_leakage_audit.json",
            "provenance/scientific_inputs.json",
        ):
            path = run["root"] / relative
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.update(expected)
            path.write_text(json.dumps(payload), encoding="utf-8")
        (run["root"] / "diagnostics/resources.json").write_text(
            json.dumps({**expected, "device": "cpu"}), encoding="utf-8"
        )
    errors, _ = _MODULE._scientific_run_audit(
        runs, manifest=_manifest(), paired_conditions=_MODULE.CONDITIONS
    )
    assert errors == []

    scientific_path = runs[0]["root"] / "provenance/scientific_inputs.json"
    tampered = json.loads(scientific_path.read_text(encoding="utf-8"))
    tampered["recovery_plan_checksum"] = "4" * 64
    scientific_path.write_text(json.dumps(tampered), encoding="utf-8")
    errors, _ = _MODULE._scientific_run_audit(
        runs, manifest=_manifest(), paired_conditions=_MODULE.CONDITIONS
    )
    assert any(
        "recovery_recovery_plan_checksum_consistency" in error
        for error in errors
    )


def test_run_provenance_columns_cannot_be_overwritten() -> None:
    run = {
        "run_id": "r_expected", "config_sha256": "a" * 64,
        "fold": 0, "seed": 2, "condition": "spatial",
    }
    frame = pd.DataFrame({"run_id": ["r_other"], "value": [1.0]})
    with pytest.raises(AdjacencyAnalysisError, match="refusing to overwrite"):
        _MODULE._attach_run_columns(frame, run)


def test_exact_main_and_bin_grids_bind_prepared_masks() -> None:
    manifest = _manifest()
    runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        run = {
            "run_id": f"r_{fold}", "config_sha256": "f" * 64,
            "fold": fold, "seed": 0, "condition": "spatial",
            "attempt": 2, "job_id": f"q_{fold}",
        }
        runs.append(run)
        for alias in manifest["folds"][str(fold)]["test_aliases"]:
            for replicate in range(3):
                rows.append({
                    **run, "model_seed": 0, "core_alias": alias,
                    "split": "test", "mask_replicate": replicate,
                    "mask_seed": 1000 + replicate,
                    "mask_checksum": f"{replicate + 1:x}" * 64,
                    "n_cells": 10, "n_masked": 100,
                    "standardized_huber": 1.0, "log1p_huber": 1.0,
                    "log1p_mae": 1.0, "log1p_mse": 1.0,
                    "log1p_rmse": 1.0,
                })
    main = pd.DataFrame(rows).rename(columns={"seed": "unused_seed"})
    _MODULE._validate_primary_rows(main, runs, manifest)

    strata: list[dict[str, Any]] = []
    for row in rows:
        for index, label in enumerate(_MODULE.TARGET_MASK_BINS):
            strata.append({
                **row, "target_mask_bin": label,
                "n_masked": 100 if index == 0 else 0,
                "log1p_huber": 1.0 if index == 0 else np.nan,
                "log1p_mae": 1.0 if index == 0 else np.nan,
                "log1p_mse": 1.0 if index == 0 else np.nan,
                "log1p_rmse": 1.0 if index == 0 else np.nan,
            })
    bins = pd.DataFrame(strata).rename(columns={"seed": "unused_seed"})
    joined = _MODULE._validate_stratum_rows(
        bins, main=main, runs=runs, manifest=manifest,
        bin_column="target_mask_bin", labels=_MODULE.TARGET_MASK_BINS,
    )
    assert len(joined) == len(main) * 5
    corrupted = bins.copy()
    corrupted.loc[0, "mask_checksum"] = "0" * 64
    with pytest.raises(AdjacencyAnalysisError, match="conflicting run or mask"):
        _MODULE._validate_stratum_rows(
            corrupted, main=main, runs=runs, manifest=manifest,
            bin_column="target_mask_bin", labels=_MODULE.TARGET_MASK_BINS,
        )


def test_positive_null_trigger_is_exact_and_audit_bound(tmp_path: Path) -> None:
    jobs = []
    for stage, fold, seed, condition in sorted(_MODULE._expected_job_slots()):
        jobs.append({
            "stage": stage, "fold": fold, "seed": seed,
            "condition": condition, "config_sha256": "a" * 64,
            "config_file_sha256": "b" * 64,
        })
    materialization = {"checksum": "c" * 64, "jobs": jobs}
    primary = [
        {
            "condition": condition, "fold": fold, "seed": seed,
            "run_id": f"r_{fold}_{seed}_{condition}",
            "job_id": f"q_{fold}_{seed}_{condition}", "attempt": 2,
            "config_sha256": "d" * 64,
            "materialized_job": {"config_sha256": "a" * 64},
        }
        for fold in _MODULE.FOLDS for seed in _MODULE.SEEDS
        for condition in _MODULE.CONDITIONS
    ]
    recovery = {
        "plan": {"checksum": "e" * 64},
        "enqueue": {"checksum": "f" * 64},
        "primary": {
            (row["fold"], row["seed"], row["condition"]): {
                "retry_job_id": row["job_id"],
                "resolved_config_sha256": row["config_sha256"],
                "materialized_job": row["materialized_job"],
            }
            for row in primary
        },
    }
    receipt = _MODULE._null_trigger_receipt(
        materialization=materialization, recovery=recovery,
        runs=primary, triggered=True,
        difference=-0.01, scientific_audit_passed=True, audit_errors=[],
    )
    path = tmp_path / "trigger.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert _MODULE._verify_positive_null_trigger(
        path, materialization=materialization, recovery=recovery,
        primary_runs=primary
    )["checksum"] == receipt["checksum"]
    receipt["scientific_audit_passed"] = False
    receipt = _MODULE._signed(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(AdjacencyAnalysisError, match="positive audited"):
        _MODULE._verify_positive_null_trigger(
            path, materialization=materialization, recovery=recovery,
            primary_runs=primary
        )


def test_topology_decision_uses_only_the_frozen_huber_gate() -> None:
    supported = {
        "mean_relative_improvement": 0.02,
        "relative_improvement_ci95": [0.005, 0.03],
        "difference_ci95": [-0.02, -0.001],
        "favoring_core_count": 8,
        "favoring_seed_count": 4,
    }
    assert _MODULE._topology_decision(supported)[0] == "REAL TOPOLOGY SUPPORTED"

    too_small = {
        **supported,
        "mean_relative_improvement": 0.01,
        "relative_improvement_ci95": [0.001, 0.019],
    }
    assert (
        _MODULE._topology_decision(too_small)[0]
        == "REAL TOPOLOGY NOT SUPPORTED"
    )
    uncertain = {
        **supported,
        "difference_ci95": [-0.02, 0.001],
        "relative_improvement_ci95": [0.001, 0.03],
    }
    assert (
        _MODULE._topology_decision(uncertain)[0]
        == "REAL TOPOLOGY INCONCLUSIVE"
    )


def test_topology_requires_exact_null_enqueue_job_ids(tmp_path: Path) -> None:
    materialization_jobs = [
        {
            "stage": stage,
            "fold": fold,
            "seed": seed,
            "condition": condition,
            "config_sha256": "a" * 64,
            "config_file_sha256": "b" * 64,
            "scientific_id": f"variant-{stage}-{fold}-{seed}-{condition}",
        }
        for stage, fold, seed, condition in sorted(_MODULE._expected_job_slots())
    ]
    materialization = {"checksum": "c" * 64, "jobs": materialization_jobs}
    recovery = {
        "plan": {"checksum": "d" * 64},
        "enqueue": {"checksum": "e" * 64},
        "null": {
            (fold, seed, _MODULE.NULL_CONDITION): {
                "plan": {
                    "worker_slot": (fold * len(_MODULE.SEEDS) + seed) % 8
                }
            }
            for fold in _MODULE.FOLDS
            for seed in _MODULE.SEEDS
        },
    }
    trigger = {"checksum": "f" * 64}
    null_jobs = []
    for item in materialization_jobs:
        if item["stage"] != "null":
            continue
        null_jobs.append(
            {
                "stage": "null",
                "fold": item["fold"],
                "seed": item["seed"],
                "condition": item["condition"],
                "config_sha256": item["config_sha256"],
                "scientific_id": item["scientific_id"],
                "job_id": f"q_null_{item['fold']}_{item['seed']}",
                "requested_gpu": (
                    int(item["fold"]) * len(_MODULE.SEEDS)
                    + int(item["seed"])
                )
                % 8,
                "maximum_attempts": 1,
            }
        )
    receipt = _MODULE._signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.NULL_ENQUEUE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "stage": "null",
            "materialization_checksum": materialization["checksum"],
            "pilot_gate_checksum": None,
            "null_trigger_checksum": trigger["checksum"],
            "recovery_plan_checksum": recovery["plan"]["checksum"],
            "recovery_enqueue_checksum": recovery["enqueue"]["checksum"],
            "execution_device": "cpu",
            "maximum_attempts": 1,
            "complete": True,
            "jobs": null_jobs,
        }
    )
    path = tmp_path / "null_enqueue_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    authorization = _MODULE._null_enqueue_authorization(
        path,
        materialization=materialization,
        recovery=recovery,
        trigger=trigger,
    )
    records = [
        {
            "stage": "null",
            "fold": item["fold"],
            "seed": item["seed"],
            "condition": item["condition"],
            "attempt": 1,
            "selection_role": "recovery_null_attempt_1",
            "job_id": item["job_id"],
            "config_sha256": item["config_sha256"],
        }
        for item in null_jobs
    ]
    _MODULE._verify_null_run_identities(records, null_enqueue=authorization)
    records[0]["job_id"] = "q_unauthorized"
    with pytest.raises(AdjacencyAnalysisError, match="authorized completed"):
        _MODULE._verify_null_run_identities(
            records, null_enqueue=authorization
        )


def test_run_bundle_accepts_only_the_signed_attempt_two_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialization_jobs = [
        {
            "stage": stage,
            "fold": fold,
            "seed": seed,
            "condition": condition,
            "config_sha256": "a" * 64,
            "config_file_sha256": "b" * 64,
        }
        for stage, fold, seed, condition in sorted(_MODULE._expected_job_slots())
    ]
    target = next(
        item
        for item in materialization_jobs
        if (
            item["stage"], item["fold"], item["seed"], item["condition"]
        )
        == ("primary", 0, 0, "spatial")
    )
    materialized_config = {
        "campaign": {"campaign_id": _MODULE.CAMPAIGN_ID},
        "experiment": {"stage": "primary"},
        "graph": {"adjacency_condition": "spatial"},
        "fold": 0,
        "seed": 0,
        "attempt": 1,
    }
    target["config_sha256"] = _MODULE.canonical_sha256(materialized_config)
    resolved_config = {**materialized_config, "attempt": 2}
    resolved_sha = _MODULE.canonical_sha256(resolved_config)
    run_id = "r_authorized_attempt_2"
    job_id = "q_authorized_retry"
    run_root = tmp_path / "artifacts/runs/2026/08" / run_id
    run_root.mkdir(parents=True)
    (run_root / "_SUCCESS").write_text("", encoding="utf-8")
    (run_root / "manifest.yaml").write_text(
        json.dumps({"run_id": run_id, "job_id": job_id}), encoding="utf-8"
    )

    def write_bundle(config: dict[str, Any]) -> None:
        config_sha = _MODULE.canonical_sha256(config)
        (run_root / "config.resolved.yaml").write_text(
            json.dumps(config), encoding="utf-8"
        )
        (run_root / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "success",
                    "campaign_id": _MODULE.CAMPAIGN_ID,
                    "stage": "primary",
                    "condition": "spatial",
                    "fold": 0,
                    "model_seed": 0,
                    "config_sha256": config_sha,
                }
            ),
            encoding="utf-8",
        )

    write_bundle(resolved_config)
    monkeypatch.setattr(_MODULE, "verify_run_bundle", lambda _: None)
    materialization = {"jobs": materialization_jobs}
    recovery = {
        "plan": {"checksum": "c" * 64},
        "recovery_contract_reference": "recovery_contract.yaml",
        "recovery_contract_sha256": "d" * 64,
        "recovery_plan_reference": "recovery_plan.json",
        "recovery_plan_file_sha256": "e" * 64,
        "primary": {
            (0, 0, "spatial"): {
                "plan": {"worker_slot": 0, "root_job_id": "q_root"},
                "resolved_config": resolved_config,
                "resolved_config_sha256": resolved_sha,
                "retry_job_id": job_id,
            }
        }
    }
    records = _MODULE._run_bundles(
        tmp_path / "artifacts", materialization, recovery=recovery
    )
    assert len(records) == 1
    assert records[0]["selection_role"] == "recovery_primary_attempt_2"
    assert records[0]["audit_errors"] == []

    write_bundle({**resolved_config, "unauthorized_change": True})
    records = _MODULE._run_bundles(
        tmp_path / "artifacts", materialization, recovery=recovery
    )
    assert records[0]["selection_role"] == "unauthorized_primary_retry"
    assert any("differs from recovery receipts" in item for item in records[0]["audit_errors"])


def test_atomic_report_publish_refuses_existing_destination(tmp_path: Path) -> None:
    target = tmp_path / "primary"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(AdjacencyAnalysisError, match="refusing to overwrite"):
        _MODULE._atomic_publish_report(target, lambda stage: None)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_executable_entrypoint_can_import_preparation_verifier_from_any_cwd(
    tmp_path: Path,
) -> None:
    receipt = (
        _ROOT
        / "scratch/locked_campaigns"
        / _MODULE.CAMPAIGN_ID
        / "materialization_receipt.json"
    )
    code = (
        "import pathlib, runpy; "
        f"module=runpy.run_path({str(_SCRIPT)!r}); "
        f"value=module['_materialization'](pathlib.Path({str(receipt)!r})); "
        "print(value['checksum'])"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.stdout.strip() == "0cc606d73a4979808581a032138cc18ef629952ed1a4884903ecd229bdfc4303"
