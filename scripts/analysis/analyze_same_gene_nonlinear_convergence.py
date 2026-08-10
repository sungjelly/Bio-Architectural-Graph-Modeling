#!/usr/bin/env python3
"""Verify and aggregate the frozen long-horizon nonlinear convergence audit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

# Reuse the already audited bundle, metric, and manifest primitives.  Make the
# sibling import explicit so this file also works when loaded through runpy.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
import analyze_same_gene_nonlinear as base  # noqa: E402

from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import canonical_json, canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.registry import Registry, utc_now  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402
from spatial_benchmark.same_gene_jacobian import diagonal_summary  # noqa: E402


CAMPAIGN_ID = "cmp_20260810_same_gene_nonlinear_convergence"
REPORT_NAME = "same_gene_nonlinear_convergence_20260810"
EVALUATION_ID = "eval_same_gene_nonlinear_convergence_20260810_v1"
FROZEN_CONTRACT_SHA256 = (
    "b8669ed2f0f60c45e248f1cee1dab781bbc50dd90ed97e1628050f5875a333d8"
)
DATASET_ID = base.DATASET_ID
SPLIT_ID = base.SPLIT_ID
ARMS = base.ARMS
MATRIX_ARMS = base.MATRIX_ARMS
MATRIX_PARTS = base.MATRIX_PARTS
PRIMARY_METRIC = base.PRIMARY_METRIC
ATTEMPT = 1
EPOCHS = (12, 24, 48, 96, 192)
ANCHOR_EPOCH = 12
BOOTSTRAP_SEED = 20260821
ANTECEDENT_REPORT = "same_gene_nonlinear_replication_20260810"
RIDGE_REPORT = "same_gene_cross_cell_jacobian_20260810_v2"
ANTECEDENT_EVALUATION_ID = "eval_same_gene_nonlinear_replication_20260810_v1"
RIDGE_EVALUATION_ID = "eval_same_gene_cross_cell_jacobian_20260810_v2"
ANTECEDENT_SCIENTIFIC_ID = "sci_12587b9d9e7f69f7"
ANTECEDENT_RUN_IDS = {
    0: "r_20260810T180451Z_12587b9d_s810_f00_a02_e272ac34",
    1: "r_20260810T180451Z_12587b9d_s810_f01_a02_b1fb76b9",
    2: "r_20260810T180451Z_12587b9d_s810_f02_a02_4d06f899",
    3: "r_20260810T180451Z_12587b9d_s810_f03_a02_2994062a",
}
ANTECEDENT_CHECKPOINT_SHA256 = {
    0: "88615d377d6fd2decc6d8e040a2aea8adc1e12c593e30e5e92b76f66756bd5fe",
    1: "b239b1898e6ea8c54b03733233b1b89459ae581b9dc5dde8222b1e9fa244f776",
    2: "0ebd683a47a63c27d455be151319e8ea3cbfb7daab7a5e050b8649d1123d7f25",
    3: "859aef14bafac845dff42e59f0efc91aefb36076199c2dfe1cf3993eff6c2b8b",
}
ANTECEDENT_AGGREGATE_SHA256 = (
    "e82e01314e680b98dee494e540a0d59166e82f918ea208a7214b5627a670004a"
)
ANTECEDENT_RESULTS_SHA256 = (
    "962a1e91129cb1cb74009422e3663b4014c3926f0b49bbf826ae537032a46c6f"
)
RIDGE_AGGREGATE_SHA256 = (
    "22d71833af192b03c4130b5105e98628232afd36ae06a8e4a66b2360e0321883"
)
RIDGE_RESULTS_SHA256 = (
    "253b1dbb4b43992c1d9ac8427f7501cb50e94ecd5730e362ff27fbeb4ef57b40"
)
GENE_AXIS_SHA256 = "046eb86c7ea8f1fe6977598a0190132340400fc61802fcde63ab5ac0e9502b03"
ELIGIBLE_MASK_SHA256 = "2420a9d160894a78e6a1db1cc4ff2c52757211cf31dec859856f7c3cf231712d"
RAW_FINGERPRINT = "e1513d598d4ea910386842cdf4a6d9bd58e21318d484bdb34dfb3962f1490ea5"
PROCESSED_FINGERPRINT = "6304132b4a57699c81b8616324dbeb2faee24b58ce70490be552595d84af34ce"
SPLIT_FINGERPRINT = "12c0d46244ed443a482586fc85422672f9f04132c7def49a741c40ba48bf4264"


class ConvergenceAnalysisError(RuntimeError):
    """Raised when an input, frozen comparison, or publication is invalid."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _gene_hash(genes: Iterable[str]) -> str:
    return sha256(canonical_json(list(genes)).encode("utf-8")).hexdigest()


def _mask_hash(mask: np.ndarray) -> str:
    return sha256(np.asarray(mask, dtype=np.uint8).tobytes(order="C")).hexdigest()


def _configure_reused_primitives() -> None:
    base.CAMPAIGN_ID = CAMPAIGN_ID
    base.REPORT_NAME = REPORT_NAME
    base.EVALUATION_ID = EVALUATION_ID
    base.FROZEN_CONTRACT_SHA256 = FROZEN_CONTRACT_SHA256


def _config(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["config_json"]))


def _discover_convergence_runs(registry: Registry) -> list[dict[str, Any]]:
    with registry.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM runs WHERE campaign_id = ? ORDER BY created_at",
            (CAMPAIGN_ID,),
        ).fetchall()
    selected: dict[int, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        config = _config(row)
        if config.get("profile") != "full":
            continue
        if int(row["attempt"]) != ATTEMPT:
            raise ConvergenceAnalysisError(
                f"unexpected convergence full attempt: {row['run_id']}"
            )
        if row["status"] != "completed":
            raise ConvergenceAnalysisError(
                f"convergence full run is not complete: {row['run_id']}"
            )
        fold = int(row["fold"])
        if fold in selected:
            raise ConvergenceAnalysisError(f"duplicate convergence fold {fold}")
        trainer = config.get("trainer", {})
        campaign = config.get("campaign", {})
        if (
            campaign.get("campaign_id") != CAMPAIGN_ID
            or campaign.get("experiment_flavor") != "convergence"
            or campaign.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256
            or tuple(trainer.get("effective_epoch_candidates", ())) != EPOCHS
            or int(trainer.get("anchor_epoch", -1)) != ANCHOR_EPOCH
        ):
            raise ConvergenceAnalysisError(
                f"convergence execution contract mismatch: {row['run_id']}"
            )
        selected[fold] = row
    if set(selected) != {0, 1, 2, 3}:
        raise ConvergenceAnalysisError(
            f"expected attempt-1 folds 0-3, found {sorted(selected)}"
        )
    runs = [selected[fold] for fold in range(4)]
    scientific_ids = {str(row["scientific_id"]) for row in runs}
    if len(scientific_ids) != 1:
        raise ConvergenceAnalysisError(
            f"convergence folds lack a common scientific_id: {scientific_ids}"
        )
    return runs


def _discover_antecedent_runs(registry: Registry) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    campaign = "cmp_20260810_same_gene_nonlinear_replication"
    with registry.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM runs WHERE campaign_id = ? ORDER BY created_at", (campaign,)
        ).fetchall()
    selected: dict[int, dict[str, Any]] = {}
    excluded: dict[int, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if _config(row).get("profile") != "full" or row["status"] != "completed":
            continue
        fold, attempt = int(row["fold"]), int(row["attempt"])
        if attempt == 2:
            if fold in selected:
                raise ConvergenceAnalysisError(f"duplicate antecedent fold {fold}")
            selected[fold] = row
        elif attempt == 1:
            excluded[fold] = row
    if set(selected) != {0, 1, 2, 3} or set(excluded) != {0, 1, 2, 3}:
        raise ConvergenceAnalysisError(
            "antecedent attempt-2 production or metadata-invalid attempt-1 set is absent"
        )
    runs = [selected[fold] for fold in range(4)]
    for fold, row in enumerate(runs):
        if (
            str(row["scientific_id"]) != ANTECEDENT_SCIENTIFIC_ID
            or str(row["run_id"]) != ANTECEDENT_RUN_IDS[fold]
        ):
            raise ConvergenceAnalysisError(
                f"frozen antecedent identity changed at fold {fold}"
            )
    return runs, [excluded[fold] for fold in range(4)]


def _load_arrays(bundle: Path, *, convergence: bool) -> dict[str, np.ndarray]:
    with np.load(bundle / "nonlinear_jacobians.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    expected = {"genes", "eligible_morphology_only"}
    for arm in MATRIX_ARMS:
        expected.add(f"eligible_{arm}")
        for part in (*MATRIX_PARTS, "mean_hidden_derivative"):
            expected.add(f"{arm}_{part}")
            if convergence:
                expected.add(f"{arm}_anchor12_{part}")
    if set(arrays) != expected:
        raise ConvergenceAnalysisError(
            f"unexpected {'convergence' if convergence else 'antecedent'} NPZ keys: "
            f"{sorted(set(arrays) ^ expected)}"
        )
    for arm in MATRIX_ARMS:
        for prefix in (("", "anchor12_") if convergence else ("",)):
            for part in MATRIX_PARTS:
                value = arrays[f"{arm}_{prefix}{part}"]
                if value.shape != (1000, 1000) or not bool(np.isfinite(value).all()):
                    raise ConvergenceAnalysisError(f"invalid Jacobian {arm}/{prefix}{part}")
            hidden = arrays[f"{arm}_{prefix}mean_hidden_derivative"]
            if hidden.shape != (64,) or not bool(np.isfinite(hidden).all()):
                raise ConvergenceAnalysisError(f"invalid hidden derivative {arm}/{prefix}")
    return arrays


def _verify_convergence_bundle(
    row: dict[str, Any], registry: Registry
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    _configure_reused_primitives()
    result, check = base._verify_bundle(row, registry)
    bundle = Path(str(row["artifact_path"]))
    arrays = _load_arrays(bundle, convergence=True)
    checkpoint = torch.load(
        bundle / "checkpoints/last.ckpt", map_location="cpu", weights_only=True
    )
    for arm in ARMS:
        arm_result = result["arms"][arm]
        history = arm_result.get("validation_history", [])
        if tuple(int(item["epoch"]) for item in history) != EPOCHS:
            raise ConvergenceAnalysisError(f"validation trajectory changed: {arm}")
        if not all(
            np.isfinite(float(item["component_equal_mse"])) for item in history
        ):
            raise ConvergenceAnalysisError(f"nonfinite validation trajectory: {arm}")
        expected_selected = int(
            min(
                history,
                key=lambda item: (
                    float(item["component_equal_mse"]),
                    int(item["epoch"]),
                ),
            )["epoch"]
        )
        selected_epoch = int(arm_result["selected_epoch"])
        if selected_epoch != expected_selected:
            raise ConvergenceAnalysisError(
                f"validation argmin/earliest-tie selection changed: {arm}"
            )
        if int(arm_result.get("refit_epoch", -1)) != selected_epoch:
            raise ConvergenceAnalysisError(f"final refit epoch changed: {arm}")
        anchor = arm_result.get("anchor")
        if anchor is None or int(anchor.get("epoch", -1)) != ANCHOR_EPOCH:
            raise ConvergenceAnalysisError(f"missing epoch-12 anchor: {arm}")
        state = checkpoint["arms"][arm]
        if (
            int(state.get("selected_epoch", -1)) != selected_epoch
            or int(state.get("refit_epoch", -1)) != selected_epoch
        ):
            raise ConvergenceAnalysisError(
                f"checkpoint selection/refit epoch changed: {arm}"
            )
        if int(state.get("anchor_epoch", -1)) != ANCHOR_EPOCH:
            raise ConvergenceAnalysisError(f"checkpoint anchor identity changed: {arm}")
        anchor_state = state.get("anchor_state_dict")
        if not isinstance(anchor_state, dict) or not anchor_state:
            raise ConvergenceAnalysisError(f"checkpoint anchor state absent: {arm}")
        if any(not bool(torch.isfinite(value).all()) for value in anchor_state.values()):
            raise ConvergenceAnalysisError(f"checkpoint anchor nonfinite: {arm}")
    check["npz_sha256"] = sha256_file(bundle / "nonlinear_jacobians.npz")
    check["anchor12_present"] = True
    return result, check, arrays


def _verify_antecedent_bundle(
    row: dict[str, Any], registry: Registry
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    bundle = Path(str(row["artifact_path"]))
    canonical = verify_run_bundle(bundle)
    native = base._verify_native_manifest(bundle)
    if registry.verify_artifacts(run_id=str(row["run_id"])):
        raise ConvergenceAnalysisError(f"antecedent registry artifacts failed: {row['run_id']}")
    result = json.loads((bundle / "results.json").read_text(encoding="utf-8"))
    arrays = _load_arrays(bundle, convergence=False)
    checkpoint_sha256 = sha256_file(bundle / "checkpoints/last.ckpt")
    fold = int(row["fold"])
    if checkpoint_sha256 != ANTECEDENT_CHECKPOINT_SHA256[fold]:
        raise ConvergenceAnalysisError(
            f"frozen antecedent checkpoint changed at fold {fold}"
        )
    return result, {
        "run_id": row["run_id"],
        "fold": int(row["fold"]),
        "attempt": int(row["attempt"]),
        "scientific_id": row["scientific_id"],
        "bundle": str(bundle),
        "canonical_bundle": canonical,
        "native_manifest": native,
        "checkpoint_sha256": checkpoint_sha256,
        "npz_sha256": sha256_file(bundle / "nonlinear_jacobians.npz"),
        "registry_artifacts_verified": True,
    }, arrays


def _max_prediction_difference(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> float:
    one = {int(item["geometry_group"]): item for item in first}
    two = {int(item["geometry_group"]): item for item in second}
    if set(one) != set(two):
        raise ConvergenceAnalysisError("anchor component prediction sets differ")
    maximum = 0.0
    for group in one:
        for key in ("y_true", "y_pred"):
            a = np.asarray(one[group][key], dtype=np.float64)
            b = np.asarray(two[group][key], dtype=np.float64)
            if a.shape != (1000,) or b.shape != (1000,):
                raise ConvergenceAnalysisError("component prediction gene axis changed")
            maximum = max(maximum, float(np.max(np.abs(a - b))))
    return maximum


def _verify_antecedent_equivalence(
    convergence_results: list[dict[str, Any]],
    convergence_arrays: list[dict[str, np.ndarray]],
    antecedent_results: list[dict[str, Any]],
    antecedent_arrays: list[dict[str, np.ndarray]],
) -> dict[str, Any]:
    candidate_checks: list[dict[str, Any]] = []
    anchor_checks: list[dict[str, Any]] = []
    for fold in range(4):
        for arm in ARMS:
            current = convergence_results[fold]["arms"][arm]
            prior = antecedent_results[fold]["arms"][arm]
            candidate12 = next(
                item for item in current["validation_history"] if int(item["epoch"]) == 12
            )
            prior12 = next(
                item for item in prior["validation_history"] if int(item["epoch"]) == 12
            )
            validation_delta = abs(
                float(candidate12["component_equal_mse"])
                - float(prior12["component_equal_mse"])
            )
            candidate_checks.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "absolute_mse_difference": validation_delta,
                    "passed": bool(validation_delta <= 1e-7),
                }
            )
            anchor = current["anchor"]
            metric_delta = max(
                abs(float(anchor["evaluation"][name]) - float(prior["evaluation"][name]))
                for name in ("component_equal_mse", "component_equal_mae")
            )
            anchor_rows = {int(x["geometry_group"]): x for x in anchor["evaluation"]["per_component"]}
            prior_rows = {int(x["geometry_group"]): x for x in prior["evaluation"]["per_component"]}
            if set(anchor_rows) != set(prior_rows):
                raise ConvergenceAnalysisError(f"anchor component sets changed: f{fold}/{arm}")
            component_delta = max(
                abs(float(anchor_rows[group][metric]) - float(prior_rows[group][metric]))
                for group in anchor_rows for metric in ("mse", "mae")
            )
            prediction_delta = _max_prediction_difference(
                anchor["component_predictions"], prior["component_predictions"]
            )
            jacobian_delta = 0.0
            if arm in MATRIX_ARMS:
                jacobian_delta = max(
                    float(
                        np.max(
                            np.abs(
                                convergence_arrays[fold][f"{arm}_anchor12_{part}"]
                                - antecedent_arrays[fold][f"{arm}_{part}"]
                            )
                        )
                    )
                    for part in MATRIX_PARTS
                )
            metric_prediction_passed = bool(
                max(metric_delta, component_delta, prediction_delta) <= 1e-7
            )
            jacobian_passed = bool(jacobian_delta <= 1e-6)
            anchor_checks.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "maximum_metric_difference": max(metric_delta, component_delta),
                    "maximum_prediction_difference": prediction_delta,
                    "maximum_jacobian_difference": jacobian_delta,
                    "metric_prediction_passed": metric_prediction_passed,
                    "jacobian_passed": jacobian_passed,
                    "passed": bool(metric_prediction_passed and jacobian_passed),
                }
            )
    if len(candidate_checks) != 16 or len(anchor_checks) != 16:
        raise ConvergenceAnalysisError("antecedent equivalence coverage is incomplete")
    return {
        "candidate12_validation_checks": candidate_checks,
        "candidate12_check_count": 16,
        "candidate12_tolerance": 1e-7,
        "anchor12_checks": anchor_checks,
        "anchor12_check_count": 16,
        "anchor_metric_prediction_tolerance": 1e-7,
        "anchor_jacobian_tolerance": 1e-6,
        "passed": bool(
            all(item["passed"] for item in candidate_checks)
            and all(item["passed"] for item in anchor_checks)
        ),
    }


def _arm_metrics(results: list[dict[str, Any]], arm: str, *, anchor: bool = False) -> dict[str, Any]:
    rows, fold_mse, fold_mae = [], [], []
    for result in results:
        block = result["arms"][arm]["anchor" if anchor else "evaluation"]
        evaluation = block["evaluation"] if anchor else block
        rows.extend(evaluation["per_component"])
        fold_mse.append(float(evaluation["component_equal_mse"]))
        fold_mae.append(float(evaluation["component_equal_mae"]))
    groups = [int(item["geometry_group"]) for item in rows]
    if len(groups) != 27 or len(set(groups)) != 27:
        raise ConvergenceAnalysisError(f"{arm} lacks 27 paired components")
    return {
        "component_equal_mse": float(np.mean([item["mse"] for item in rows])),
        "component_equal_mae": float(np.mean([item["mae"] for item in rows])),
        "fold_mse": fold_mse,
        "fold_mae": fold_mae,
        "per_component": rows,
    }


def _convergence_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories = []
    for fold, result in enumerate(results):
        for arm in ("morphology_only", "observed_near"):
            block = result["arms"][arm]
            history = {int(item["epoch"]): float(item["component_equal_mse"]) for item in block["validation_history"]}
            if set(history) != set(EPOCHS):
                raise ConvergenceAnalysisError(f"incomplete convergence curve: f{fold}/{arm}")
            gain = (history[96] - history[192]) / history[96]
            selected = int(block["selected_epoch"])
            saturated = selected < 192 or gain <= 0.001
            trajectories.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "selected_epoch": selected,
                    "validation_mse_96": history[96],
                    "validation_mse_192": history[192],
                    "relative_validation_mse_gain_96_to_192": gain,
                    "saturated": bool(saturated),
                }
            )
    return {
        "passed": bool(all(item["saturated"] for item in trajectories)),
        "trajectory_count": len(trajectories),
        "saturated_trajectory_count": int(sum(item["saturated"] for item in trajectories)),
        "relative_gain_maximum": 0.001,
        "trajectories": trajectories,
    }


def _component_map(rows: Iterable[dict[str, Any]]) -> dict[int, float]:
    values = {
        int(item["geometry_group"]): float(item["mse"])
        for item in rows
    }
    if len(values) != 27:
        raise ConvergenceAnalysisError("attribution requires 27 unique components")
    return values


def _prediction_attribution_gate(
    selected_metrics: dict[str, dict[str, Any]],
    anchor_metrics: dict[str, dict[str, Any]],
    selected_scientific_gate: dict[str, Any],
) -> dict[str, Any]:
    """Bootstrap selected-minus-anchor near-vs-morph relative gain.

    One index matrix is used for all four MSE vectors, exactly implementing the
    frozen same-resample requirement rather than subtracting two independent
    confidence intervals.
    """

    selected_morph = _component_map(
        selected_metrics["morphology_only"]["per_component"]
    )
    selected_near = _component_map(
        selected_metrics["observed_near"]["per_component"]
    )
    anchor_morph = _component_map(
        anchor_metrics["morphology_only"]["per_component"]
    )
    anchor_near = _component_map(
        anchor_metrics["observed_near"]["per_component"]
    )
    groups = sorted(selected_morph)
    if not (
        set(selected_near) == set(anchor_morph) == set(anchor_near) == set(groups)
    ):
        raise ConvergenceAnalysisError("selected/anchor attribution components differ")
    arrays = {
        "selected_morph": np.asarray([selected_morph[group] for group in groups]),
        "selected_near": np.asarray([selected_near[group] for group in groups]),
        "anchor_morph": np.asarray([anchor_morph[group] for group in groups]),
        "anchor_near": np.asarray([anchor_near[group] for group in groups]),
    }

    def relative_gain(morphology: np.ndarray, near: np.ndarray) -> float:
        denominator = float(np.mean(morphology))
        if denominator <= 0 or not np.isfinite(denominator):
            raise ConvergenceAnalysisError("invalid morphology MSE denominator")
        return float((denominator - float(np.mean(near))) / denominator)

    selected_gain = relative_gain(arrays["selected_morph"], arrays["selected_near"])
    anchor_gain = relative_gain(arrays["anchor_morph"], arrays["anchor_near"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = 10_000
    sampled_delta = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        index = rng.integers(0, len(groups), size=len(groups))
        sampled_delta[draw] = relative_gain(
            arrays["selected_morph"][index], arrays["selected_near"][index]
        ) - relative_gain(
            arrays["anchor_morph"][index], arrays["anchor_near"][index]
        )
    selected_fold = np.asarray(
        selected_metrics["observed_near"]["fold_mse"], dtype=np.float64
    )
    anchor_fold = np.asarray(
        anchor_metrics["observed_near"]["fold_mse"], dtype=np.float64
    )
    folds_improved = int(np.sum(selected_fold < anchor_fold))
    lower = float(np.quantile(sampled_delta, 0.025))
    upper = float(np.quantile(sampled_delta, 0.975))
    return {
        "passed": bool(
            selected_scientific_gate["passed"]
            and lower > 0
            and folds_improved >= 3
        ),
        "selected_near_vs_morphology_gate_passed": bool(
            selected_scientific_gate["passed"]
        ),
        "selected_relative_gain": selected_gain,
        "anchor12_relative_gain": anchor_gain,
        "selected_minus_anchor12_relative_gain": selected_gain - anchor_gain,
        "bootstrap_2_5_percentile": lower,
        "bootstrap_97_5_percentile": upper,
        "bootstrap_draws": draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "same_resample_for_selected_and_anchor12": True,
        "geometry_component_count": len(groups),
        "fold_relative_near_mse_gains": (
            (anchor_fold - selected_fold) / anchor_fold
        ).tolist(),
        "folds_selected_near_better_than_anchor12": folds_improved,
        "minimum_improved_folds": 3,
    }


def _row_attribution_gate(
    selected_fold_matrices: list[np.ndarray],
    anchor_fold_matrices: list[np.ndarray],
    eligible: np.ndarray,
    selected_scientific_gate: dict[str, Any],
) -> dict[str, Any]:
    if len(selected_fold_matrices) != 4 or len(anchor_fold_matrices) != 4:
        raise ConvergenceAnalysisError("row attribution requires four fold matrices")
    selected_aggregate = np.mean(selected_fold_matrices, axis=0)
    anchor_aggregate = np.mean(anchor_fold_matrices, axis=0)
    selected_summary = diagonal_summary(selected_aggregate, eligible).as_dict()
    anchor_summary = diagonal_summary(anchor_aggregate, eligible).as_dict()
    selected_folds = [
        diagonal_summary(matrix, eligible).as_dict()
        for matrix in selected_fold_matrices
    ]
    anchor_folds = [
        diagonal_summary(matrix, eligible).as_dict()
        for matrix in anchor_fold_matrices
    ]
    top1_improved = [
        bool(selected["row_top1_fraction"] > anchor["row_top1_fraction"])
        for selected, anchor in zip(selected_folds, anchor_folds, strict=True)
    ]
    top1_percent_improved = [
        bool(
            selected["row_top1_percent_fraction"]
            > anchor["row_top1_percent_fraction"]
        )
        for selected, anchor in zip(selected_folds, anchor_folds, strict=True)
    ]
    fixed_top1 = 0.24356223175965666
    fixed_top1_percent = 0.4248927038626609
    aggregate_top1_improved = bool(
        selected_summary["row_top1_fraction"] > fixed_top1
    )
    aggregate_top1_percent_improved = bool(
        selected_summary["row_top1_percent_fraction"] > fixed_top1_percent
    )
    top1_folds = int(sum(top1_improved))
    top1_percent_folds = int(sum(top1_percent_improved))
    return {
        "passed": bool(
            selected_scientific_gate["passed"]
            and aggregate_top1_improved
            and aggregate_top1_percent_improved
            and top1_folds >= 3
            and top1_percent_folds >= 3
        ),
        "selected_strict_row_selectivity_gate_passed": bool(
            selected_scientific_gate["passed"]
        ),
        "selected_aggregate": selected_summary,
        "anchor12_aggregate": anchor_summary,
        "fixed12_row_top1_fraction": fixed_top1,
        "fixed12_row_top1_percent_fraction": fixed_top1_percent,
        "aggregate_row_top1_fraction_exceeds_fixed12": aggregate_top1_improved,
        "aggregate_row_top1_percent_fraction_exceeds_fixed12": (
            aggregate_top1_percent_improved
        ),
        "selected_fold_summaries": selected_folds,
        "anchor12_fold_summaries": anchor_folds,
        "fold_row_top1_improved": top1_improved,
        "fold_row_top1_percent_improved": top1_percent_improved,
        "folds_row_top1_improved": top1_folds,
        "folds_row_top1_percent_improved": top1_percent_folds,
        "minimum_improved_folds": 3,
    }


def _scientific_analysis(
    results: list[dict[str, Any]], arrays: list[dict[str, np.ndarray]], eligible: np.ndarray
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    fold_matrices = {
        arm: {part: [item[f"{arm}_{part}"].astype(np.float64) for item in arrays] for part in MATRIX_PARTS}
        for arm in MATRIX_ARMS
    }
    matrices = {
        arm: {part: np.mean(values, axis=0) for part, values in parts.items()}
        for arm, parts in fold_matrices.items()
    }
    summaries = {
        arm: {part: diagonal_summary(value, eligible).as_dict() for part, value in parts.items()}
        for arm, parts in matrices.items()
    }
    metrics = {arm: _arm_metrics(results, arm) for arm in ARMS}
    near_vs_morph = base._paired_gain(
        metrics["morphology_only"]["per_component"], metrics["observed_near"]["per_component"], seed=BOOTSTRAP_SEED
    )
    near_vs_perm = base._paired_gain(
        metrics["within_fov_permuted_near"]["per_component"], metrics["observed_near"]["per_component"], seed=BOOTSTRAP_SEED
    )
    annular_vs_morph = base._paired_gain(
        metrics["morphology_only"]["per_component"], metrics["observed_annular"]["per_component"], seed=BOOTSTRAP_SEED
    )
    near_fold = np.asarray(metrics["observed_near"]["fold_mse"])
    morph_fold = np.asarray(metrics["morphology_only"]["fold_mse"])
    perm_fold = np.asarray(metrics["within_fov_permuted_near"]["fold_mse"])
    fold_diagonal = np.asarray([np.diag(value)[eligible] for value in fold_matrices["observed_near"]["total"]])
    correlations = [
        float(spearmanr(fold_diagonal[a], fold_diagonal[b]).statistic)
        for a in range(4) for b in range(a + 1, 4)
    ]
    if not bool(np.isfinite(correlations).all()):
        raise ConvergenceAnalysisError("fold diagonal stability is nonfinite")
    positive, negative = np.sum(fold_diagonal > 0, axis=0), np.sum(fold_diagonal < 0, axis=0)
    sign_fraction = float(np.mean(np.maximum(positive, negative) >= 3))
    fold_ratios = [
        float(np.median(np.abs(np.diag(near)[eligible])) / np.median(np.abs(np.diag(perm)[eligible])))
        for near, perm in zip(
            fold_matrices["observed_near"]["total"],
            fold_matrices["within_fov_permuted_near"]["total"], strict=True
        )
    ]
    near_summary = summaries["observed_near"]["total"]
    perm_summary = summaries["within_fov_permuted_near"]["total"]
    ratio = float(near_summary["median_absolute_diagonal"] / perm_summary["median_absolute_diagonal"])
    favors_morph, favors_perm = int(np.sum(near_fold < morph_fold)), int(np.sum(near_fold < perm_fold))
    technical = (
        all(result["controls"]["all_outputs_finite"] is True for result in results)
        and all(result["controls"]["analytical_nonlinear_jacobian"]["passed"] is True for result in results)
        and all(result["controls"]["identity_oracle_row_top1_fraction"] == 1.0 for result in results)
        and all(result["controls"]["receiver_expression_input"] is False for result in results)
        and all(float(result["peak_vram_gb"]) <= 20.5 for result in results)
        and all(bool(np.isfinite(value).all()) for parts in matrices.values() for value in parts.values())
    )
    gates = {
        "near_vs_morphology_prediction": {"passed": bool(near_vs_morph["relative_mse_gain"] >= .02 and favors_morph >= 3), "gain": near_vs_morph["relative_mse_gain"], "minimum": .02, "folds": favors_morph, "minimum_folds": 3},
        "near_vs_permuted_prediction": {"passed": bool(near_vs_perm["relative_mse_gain"] >= .01 and favors_perm >= 3), "gain": near_vs_perm["relative_mse_gain"], "minimum": .01, "folds": favors_perm, "minimum_folds": 3},
        "diagonal_enrichment": {"passed": bool(near_summary["diagonal_offdiagonal_ratio"] >= 2), "observed": near_summary["diagonal_offdiagonal_ratio"], "minimum": 2.0},
        "diagonal_row_selectivity": {"passed": bool(near_summary["row_top1_fraction"] >= .25 and near_summary["row_top1_percent_fraction"] >= .5), "top1": near_summary["row_top1_fraction"], "top1_minimum": .25, "top1_percent": near_summary["row_top1_percent_fraction"], "top1_percent_minimum": .5},
        "near_vs_permuted_diagonal": {"passed": bool(ratio >= 1.25 and sum(x >= 1.25 for x in fold_ratios) >= 3), "aggregate_ratio": ratio, "minimum": 1.25, "fold_ratios": fold_ratios, "folds_at_or_above": sum(x >= 1.25 for x in fold_ratios)},
        "fold_stability": {"passed": bool(np.median(correlations) >= .70 and sign_fraction >= .75), "median_spearman": float(np.median(correlations)), "minimum_spearman": .70, "sign_fraction": sign_fraction, "minimum_sign_fraction": .75},
        "technical_controls": {"passed": bool(technical)},
    }
    payload = {
        "arm_metrics": metrics,
        "near_vs_morphology": near_vs_morph,
        "near_vs_permuted": near_vs_perm,
        "annular_vs_morphology": annular_vs_morph,
        "near_favors_morphology_folds": favors_morph,
        "near_favors_permuted_folds": favors_perm,
        "aggregate_summaries": summaries,
        "fold_pairwise_signed_diagonal_spearman": correlations,
        "sign_consistent_gene_fraction": sign_fraction,
        "near_vs_permuted_diagonal_ratio": ratio,
        "fold_near_vs_permuted_diagonal_ratios": fold_ratios,
        "positive_fold_count": positive,
        "negative_fold_count": negative,
    }
    return payload, matrices, gates


def _comparison(
    current: dict[str, Any], baseline_results: list[dict[str, Any]], baseline_matrix: np.ndarray,
    current_matrix: np.ndarray, eligible: np.ndarray, *, label: str
) -> dict[str, Any]:
    baseline_metrics = _arm_metrics(baseline_results, "observed_near")
    current_metrics = current["arm_metrics"]["observed_near"]
    gain = base._paired_gain(
        baseline_metrics["per_component"], current_metrics["per_component"], seed=BOOTSTRAP_SEED
    )
    old_fold, new_fold = np.asarray(baseline_metrics["fold_mse"]), np.asarray(current_metrics["fold_mse"])
    diagonal_new, diagonal_old = np.diag(current_matrix)[eligible], np.diag(baseline_matrix)[eligible]
    correlation = float(spearmanr(diagonal_new, diagonal_old).statistic)
    if not np.isfinite(correlation):
        raise ConvergenceAnalysisError(f"{label} diagonal comparison is nonfinite")
    folds = int(np.sum(new_fold < old_fold))
    return {
        "baseline": label,
        "prediction": gain,
        "fold_relative_mse_gains": ((old_fold - new_fold) / old_fold).tolist(),
        "folds_improved": folds,
        "prediction_superiority_gate": {
            "passed": bool(gain["bootstrap_2_5_percentile"] > 0 and folds >= 3),
            "bootstrap_lower_bound": gain["bootstrap_2_5_percentile"],
            "minimum_folds": 3,
        },
        "signed_diagonal_spearman": correlation,
        "signed_diagonal_sign_agreement": float(np.mean(np.sign(diagonal_new) == np.sign(diagonal_old))),
    }


def _verdict(
    *,
    antecedent_anchor_valid: bool,
    convergence: dict[str, Any],
    gates: dict[str, Any],
    prediction_attribution: dict[str, Any],
    row_attribution: dict[str, Any],
) -> str:
    """Apply all nine frozen, mutually-exclusive verdict branches in order."""

    if not antecedent_anchor_valid:
        return "attribution_invalid_anchor_mismatch"
    if not convergence["passed"]:
        return "optimization_inconclusive_at_192_epochs"
    all_scientific = all(bool(item["passed"]) for item in gates.values())
    prediction_explained = bool(prediction_attribution["passed"])
    row_explained = bool(row_attribution["passed"])
    if all_scientific and prediction_explained and row_explained:
        return "strict_selectivity_supported_and_12_epoch_budget_explains_both"
    if all_scientific:
        return "strict_selectivity_supported_without_full_budget_attribution"
    if prediction_explained and row_explained:
        return "budget_explains_both_prior_failures_but_other_gate_failed"
    if prediction_explained:
        return "budget_explains_prediction_failure_only"
    if row_explained:
        return "budget_explains_row_failure_only"
    non_row_passed = all(
        bool(item["passed"])
        for name, item in gates.items()
        if name != "diagonal_row_selectivity"
    )
    if non_row_passed and not gates["diagonal_row_selectivity"]["passed"]:
        return "robust_nonexclusive_enrichment_after_convergence"
    return "evidence_against_12_epoch_budget_explanation"


def _analysis_inventory(directory: Path) -> list[dict[str, Any]]:
    return base._analysis_inventory(directory)


def _verify_native_inputs(verification: dict[str, Any]) -> None:
    for item in verification["input_files"]:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]) or sha256_file(path) != item["sha256"]:
            raise ConvergenceAnalysisError(f"frozen input hash mismatch: {path}")
    for item in (
        *verification["input_bundle_checks"],
        *verification["antecedent_bundle_checks"],
        *verification["ridge_bundle_checks"],
    ):
        verify_run_bundle(Path(item["bundle"]))
        native = base._verify_native_manifest(Path(item["bundle"]))
        if native["manifest_sha256"] != item["native_manifest"]["manifest_sha256"]:
            raise ConvergenceAnalysisError(f"native manifest changed: {item['bundle']}")


def _verify_output(directory: Path) -> dict[str, Any]:
    required = {
        "aggregate_results.json", "aggregate_jacobians.npz", "eligible_gene_diagonal_summary.csv",
        "summary.png", "report.md", "provenance.json", "verification.json",
        "registry_evaluation.json", "analysis_manifest.json",
    }
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        raise ConvergenceAnalysisError(f"analysis files missing: {missing}")
    manifest = json.loads((directory / "analysis_manifest.json").read_text(encoding="utf-8"))
    if canonical_sha256(manifest["files"]) != manifest["manifest_payload_sha256"]:
        raise ConvergenceAnalysisError("output manifest digest changed")
    for item in manifest["files"]:
        path = directory / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]) or sha256_file(path) != item["sha256"]:
            raise ConvergenceAnalysisError(f"output hash mismatch: {path}")
    verification = json.loads((directory / "verification.json").read_text(encoding="utf-8"))
    if verification.get("verified") is not True or verification.get("input_run_count") != 4:
        raise ConvergenceAnalysisError("stored verification is incomplete")
    _verify_native_inputs(verification)
    with np.load(directory / "aggregate_jacobians.npz", allow_pickle=False) as archive:
        genes = tuple(str(value) for value in archive["genes"])
        eligible = archive["eligible"].astype(bool)
        if _gene_hash(genes) != GENE_AXIS_SHA256 or _mask_hash(eligible) != ELIGIBLE_MASK_SHA256 or int(eligible.sum()) != 932:
            raise ConvergenceAnalysisError("published gene axis/eligibility changed")
        for arm in MATRIX_ARMS:
            for prefix in ("", "anchor12_"):
                for part in MATRIX_PARTS:
                    matrix = archive[f"jacobian_{arm}_{prefix}{part}"]
                    if matrix.shape != (1000, 1000) or not bool(np.isfinite(matrix).all()):
                        raise ConvergenceAnalysisError(f"published matrix invalid: {arm}/{prefix}{part}")
    return verification


def _verify_registry(registry: Registry, directory: Path) -> dict[str, Any]:
    with registry.connect() as connection:
        evaluation = connection.execute(
            "SELECT * FROM evaluations WHERE evaluation_id = ?", (EVALUATION_ID,)
        ).fetchone()
        campaign = connection.execute(
            "SELECT status FROM campaigns WHERE campaign_id = ?", (CAMPAIGN_ID,)
        ).fetchone()
        artifacts = connection.execute(
            "SELECT * FROM artifacts WHERE evaluation_id = ? ORDER BY path", (EVALUATION_ID,)
        ).fetchall()
    if evaluation is None or evaluation["status"] != "completed" or Path(evaluation["artifact_path"]) != directory:
        raise ConvergenceAnalysisError("registry evaluation is incomplete")
    if campaign is None or campaign["status"] != "complete":
        raise ConvergenceAnalysisError("registry campaign is incomplete")
    expected = {path.resolve() for path in directory.iterdir() if path.is_file()}
    if {Path(item["path"]).resolve() for item in artifacts} != expected:
        raise ConvergenceAnalysisError("registry artifact inventory differs")
    for item in artifacts:
        path = Path(item["path"])
        if item["status"] != "present" or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise ConvergenceAnalysisError(f"registry artifact hash mismatch: {path}")
    return {"evaluation_id": EVALUATION_ID, "artifact_count": len(artifacts)}


def _register(registry: Registry, directory: Path, payload: dict[str, Any]) -> str:
    primary = str(payload["input_runs"][0]["run_id"])
    now = utc_now()
    with registry.transaction(immediate=True) as connection:
        existing = connection.execute(
            "SELECT status, artifact_path FROM evaluations WHERE evaluation_id = ?", (EVALUATION_ID,)
        ).fetchone()
        if existing is not None:
            if existing["status"] == "completed" and Path(existing["artifact_path"]) == directory:
                return "already_registered"
            raise ConvergenceAnalysisError("conflicting convergence evaluation row")
        previous = connection.execute(
            "SELECT status FROM campaigns WHERE campaign_id = ?", (CAMPAIGN_ID,)
        ).fetchone()
        if previous is None:
            raise ConvergenceAnalysisError("convergence campaign registry row absent")
        metrics = {
            "verdict": payload["verdict"],
            "convergence_gate": payload["convergence_gate"]["passed"],
            "scientific_gates_passed": sum(x["passed"] for x in payload["scientific_gates"].values()),
            "prediction_attribution_gate": payload["attribution_gates"][
                "prediction_failure_explained"
            ]["passed"],
            "row_attribution_gate": payload["attribution_gates"][
                "row_failure_explained"
            ]["passed"],
            "near_gain_vs_morphology": payload["prediction"]["near_vs_morphology"]["relative_mse_gain"],
            "row_top1_fraction": payload["jacobian"]["aggregate_summaries"]["observed_near"]["total"]["row_top1_fraction"],
        }
        connection.execute(
            """INSERT INTO evaluations(evaluation_id,run_id,checkpoint_name,dataset_id,split_id,status,metrics_json,artifact_path,created_at,finished_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (EVALUATION_ID, primary, "four_fold_convergence_aggregate", DATASET_ID, SPLIT_ID,
             "completed", json.dumps(metrics, sort_keys=True, separators=(",", ":"), allow_nan=False), str(directory), now, now),
        )
        files = sorted(path for path in directory.iterdir() if path.is_file())
        connection.executemany(
            """INSERT INTO artifacts(run_id,evaluation_id,kind,path,sha256,size_bytes,status,created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            [(primary, EVALUATION_ID, "aggregate_analysis", str(path), sha256_file(path), path.stat().st_size, "present", now) for path in files],
        )
        connection.execute(
            "UPDATE campaigns SET status=?,updated_at=? WHERE campaign_id=?", ("complete", now, CAMPAIGN_ID)
        )
    return str(previous["status"])


def _compensate_registration(registry: Registry, previous_campaign_status: str) -> None:
    with registry.transaction(immediate=True) as connection:
        connection.execute("DELETE FROM artifacts WHERE evaluation_id=?", (EVALUATION_ID,))
        connection.execute("DELETE FROM evaluations WHERE evaluation_id=?", (EVALUATION_ID,))
        connection.execute(
            "UPDATE campaigns SET status=?,updated_at=? WHERE campaign_id=?",
            (previous_campaign_status, utc_now(), CAMPAIGN_ID),
        )


def _input_file(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ConvergenceAnalysisError(f"frozen reference hash changed: {path}")
    return {"path": str(path), "sha256": observed, "size_bytes": path.stat().st_size}


def _build_analysis(paths: Any, registry: Registry, output: Path) -> dict[str, Any]:
    contract = paths.project_root / "experiments/campaigns" / CAMPAIGN_ID / "frozen_task_contract.yaml"
    input_files = [_input_file(contract, FROZEN_CONTRACT_SHA256)]
    antecedent_root = paths.report_root / "analyses" / ANTECEDENT_REPORT
    ridge_root = paths.report_root / "analyses" / RIDGE_REPORT
    antecedent_npz = antecedent_root / "aggregate_jacobians.npz"
    antecedent_json = antecedent_root / "aggregate_results.json"
    ridge_npz, ridge_json = ridge_root / "aggregate_jacobians.npz", ridge_root / "aggregate_results.json"
    input_files.extend(
        [_input_file(antecedent_npz, ANTECEDENT_AGGREGATE_SHA256), _input_file(antecedent_json, ANTECEDENT_RESULTS_SHA256),
         _input_file(ridge_npz, RIDGE_AGGREGATE_SHA256), _input_file(ridge_json, RIDGE_RESULTS_SHA256)]
    )
    # The prior aggregate is itself hash-manifested and registry-complete.
    base._verify_output_files(antecedent_root)
    with registry.connect() as connection:
        antecedent_evaluation = connection.execute(
            "SELECT status,artifact_path FROM evaluations WHERE evaluation_id=?", (ANTECEDENT_EVALUATION_ID,)
        ).fetchone()
        ridge_evaluation = connection.execute(
            "SELECT status,artifact_path FROM evaluations WHERE evaluation_id=?", (RIDGE_EVALUATION_ID,)
        ).fetchone()
    if antecedent_evaluation is None or antecedent_evaluation["status"] != "completed" or Path(antecedent_evaluation["artifact_path"]) != antecedent_root:
        raise ConvergenceAnalysisError("antecedent evaluation registry state changed")
    if ridge_evaluation is None or ridge_evaluation["status"] != "completed" or Path(ridge_evaluation["artifact_path"]) != ridge_root:
        raise ConvergenceAnalysisError("ridge evaluation registry state changed")

    runs = _discover_convergence_runs(registry)
    antecedent_runs, antecedent_attempt1 = _discover_antecedent_runs(registry)
    results, checks, arrays = [], [], []
    prior_results, prior_checks, prior_arrays = [], [], []
    for row in runs:
        result, check, fold_arrays = _verify_convergence_bundle(row, registry)
        results.append(result); checks.append(check); arrays.append(fold_arrays)
    for row in antecedent_runs:
        result, check, fold_arrays = _verify_antecedent_bundle(row, registry)
        prior_results.append(result); prior_checks.append(check); prior_arrays.append(fold_arrays)
    equivalence = _verify_antecedent_equivalence(results, arrays, prior_results, prior_arrays)

    with np.load(antecedent_npz, allow_pickle=False) as archive:
        genes = tuple(str(value) for value in archive["genes"])
        frozen_eligible = archive["eligible"].astype(bool)
        antecedent_matrix = archive["jacobian_observed_near_total"].astype(np.float64)
    if _gene_hash(genes) != GENE_AXIS_SHA256 or _mask_hash(frozen_eligible) != ELIGIBLE_MASK_SHA256 or int(frozen_eligible.sum()) != 932:
        raise ConvergenceAnalysisError("frozen antecedent gene axis or 932-mask changed")
    if any(tuple(str(value) for value in item["genes"]) != genes for item in arrays):
        raise ConvergenceAnalysisError("convergence gene order differs from frozen axis")
    eligible = np.logical_and.reduce([item["eligible_observed_near"].astype(bool) for item in arrays])
    if not np.array_equal(eligible, frozen_eligible):
        raise ConvergenceAnalysisError("convergence common eligibility differs from frozen mask")
    for fold in range(4):
        for arm in ARMS:
            if not np.array_equal(arrays[fold][f"eligible_{arm}"], prior_arrays[fold][f"eligible_{arm}"]):
                raise ConvergenceAnalysisError(f"fold eligibility changed: f{fold}/{arm}")

    scientific, matrices, gates = _scientific_analysis(results, arrays, eligible)
    convergence = _convergence_gate(results)
    anchor_matrices = {
        arm: {part: np.mean([item[f"{arm}_anchor12_{part}"].astype(np.float64) for item in arrays], axis=0) for part in MATRIX_PARTS}
        for arm in MATRIX_ARMS
    }
    antecedent_comparison = _comparison(
        scientific, prior_results, antecedent_matrix, matrices["observed_near"]["total"], eligible, label="antecedent_12_epoch"
    )
    ridge_payload = json.loads(ridge_json.read_text(encoding="utf-8"))
    ridge_results = []
    ridge_checks: list[dict[str, Any]] = []
    for item in ridge_payload["input_runs"]:
        bundle = Path(item["bundle"])
        canonical = verify_run_bundle(bundle)
        native = base._verify_native_manifest(bundle)
        result = json.loads((bundle / "results.json").read_text(encoding="utf-8"))
        run_id = str(result["run_id"])
        registry_issues = registry.verify_artifacts(run_id=run_id)
        if registry_issues:
            raise ConvergenceAnalysisError(
                f"ridge registry artifacts failed: {run_id}: {registry_issues}"
            )
        ridge_results.append(result)
        ridge_checks.append(
            {
                "run_id": run_id,
                "fold": int(result["outer_fold"]),
                "bundle": str(bundle),
                "canonical_bundle": canonical,
                "native_manifest": native,
                "registry_artifacts_verified": True,
            }
        )
    ridge_results.sort(key=lambda item: int(item["outer_fold"]))
    # Normalize the ridge schema to the nonlinear evaluation schema used by _arm_metrics.
    normalized_ridge = []
    for item in ridge_results:
        block = item["arms"]["observed_near"].get("test", item["arms"]["observed_near"].get("evaluation"))
        normalized_ridge.append({"arms": {"observed_near": {"evaluation": block}}})
    with np.load(ridge_npz, allow_pickle=False) as archive:
        if tuple(str(value) for value in archive["genes"]) != genes:
            raise ConvergenceAnalysisError("ridge gene order changed")
        ridge_matrix = archive["jacobian_observed_near"].astype(np.float64)
        ridge_mask = archive["eligible"].astype(bool)
    if not np.array_equal(ridge_mask, eligible):
        raise ConvergenceAnalysisError("ridge/nonlinear frozen eligibility differs")
    ridge_comparison = _comparison(
        scientific, normalized_ridge, ridge_matrix, matrices["observed_near"]["total"], eligible, label="ridge"
    )
    anchor_metrics = {
        arm: _arm_metrics(results, arm, anchor=True) for arm in ARMS
    }
    prediction_attribution = _prediction_attribution_gate(
        scientific["arm_metrics"],
        anchor_metrics,
        gates["near_vs_morphology_prediction"],
    )
    selected_near_fold_matrices = [
        item["observed_near_total"].astype(np.float64) for item in arrays
    ]
    anchor_near_fold_matrices = [
        item["observed_near_anchor12_total"].astype(np.float64)
        for item in arrays
    ]
    row_attribution = _row_attribution_gate(
        selected_near_fold_matrices,
        anchor_near_fold_matrices,
        eligible,
        gates["diagonal_row_selectivity"],
    )
    row_attribution["selected_vs_anchor_signed_diagonal_spearman"] = float(
        spearmanr(
            np.diag(matrices["observed_near"]["total"])[eligible],
            np.diag(anchor_matrices["observed_near"]["total"])[eligible],
        ).statistic
    )
    attribution_gates = {
        "prediction_failure_explained": prediction_attribution,
        "row_failure_explained": row_attribution,
        "strong_budget_explanation": {
            "passed": bool(
                equivalence["passed"]
                and convergence["passed"]
                and prediction_attribution["passed"]
                and row_attribution["passed"]
            ),
            "antecedent_anchor_valid": bool(equivalence["passed"]),
            "convergence_gate_passed": bool(convergence["passed"]),
            "prediction_failure_explained": bool(
                prediction_attribution["passed"]
            ),
            "row_failure_explained": bool(row_attribution["passed"]),
        },
    }
    verdict = _verdict(
        antecedent_anchor_valid=bool(equivalence["passed"]),
        convergence=convergence,
        gates=gates,
        prediction_attribution=prediction_attribution,
        row_attribution=row_attribution,
    )
    failed = [name for name, value in gates.items() if not value["passed"]]
    payload = {
        "campaign_id": CAMPAIGN_ID, "evaluation_id": EVALUATION_ID, "verdict": verdict,
        "convergence_gate": convergence, "scientific_gates": gates,
        "scientific_gates_passed": int(sum(value["passed"] for value in gates.values())),
        "failed_scientific_gates": failed, "eligible_gene_count": 932, "total_gene_count": 1000,
        "prediction": {key: scientific[key] for key in (
            "arm_metrics", "near_vs_morphology", "near_vs_permuted", "annular_vs_morphology",
            "near_favors_morphology_folds", "near_favors_permuted_folds")},
        "jacobian": {key: scientific[key] for key in (
            "aggregate_summaries", "fold_pairwise_signed_diagonal_spearman",
            "sign_consistent_gene_fraction", "near_vs_permuted_diagonal_ratio",
            "fold_near_vs_permuted_diagonal_ratios")},
        "attribution_gates": attribution_gates,
        "row_attribution": row_attribution,
        "secondary_comparisons": {"antecedent_12_epoch": antecedent_comparison, "ridge": ridge_comparison},
        "antecedent_equivalence": equivalence,
        "input_runs": checks, "antecedent_input_runs": prior_checks,
        "ridge_input_runs": ridge_checks,
        "antecedent_metadata_invalid_attempt1": [
            {"run_id": row["run_id"], "fold": int(row["fold"]), "scientific_id": row["scientific_id"],
             "reason": "metadata-invalid fold-specific scientific identity; not a scientific failure"}
            for row in antecedent_attempt1
        ],
        "common_scientific_id": str(runs[0]["scientific_id"]), "attempt": ATTEMPT,
        "frozen_gene_axis_sha256": GENE_AXIS_SHA256, "frozen_eligible_mask_sha256": ELIGIBLE_MASK_SHA256,
        "adverse_limitations": [
            "the same outer test folds were already observed; this is not independent confirmation",
            "the optimization conclusion is inconclusive whenever any prespecified trajectory fails saturation",
            "geometry components are leakage units rather than biological replicates",
        ],
        "maximum_claim": "exploratory convergence-corrected held-out-geometry model sensitivity within two previously observed slides",
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "aggregate_results.json", payload)
    np.savez_compressed(
        output / "aggregate_jacobians.npz", genes=np.asarray(genes), eligible=eligible,
        **{f"jacobian_{arm}_{part}": matrix.astype(np.float32) for arm, parts in matrices.items() for part, matrix in parts.items()},
        **{f"jacobian_{arm}_anchor12_{part}": matrix.astype(np.float32) for arm, parts in anchor_matrices.items() for part, matrix in parts.items()},
    )
    ranks = base._row_ranks(matrices["observed_near"]["total"], eligible)
    index = np.flatnonzero(eligible)
    table = pd.DataFrame({
        "gene": [genes[position] for position in index],
        "selected_near_diagonal": np.diag(matrices["observed_near"]["total"])[eligible],
        "anchor12_near_diagonal": np.diag(anchor_matrices["observed_near"]["total"])[eligible],
        "ridge_near_diagonal": np.diag(ridge_matrix)[eligible],
        "selected_absolute_row_rank": ranks,
        "positive_fold_count": scientific["positive_fold_count"],
        "negative_fold_count": scientific["negative_fold_count"],
    })
    table.to_csv(output / "eligible_gene_diagonal_summary.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for arm, color in (("morphology_only", "#555555"), ("observed_near", "#1f77b4")):
        for fold, result in enumerate(results):
            history = result["arms"][arm]["validation_history"]
            axes[0].plot([x["epoch"] for x in history], [x["component_equal_mse"] for x in history], color=color, alpha=.55, label=arm if fold == 0 else None)
    axes[0].set(xlabel="Epoch", ylabel="Validation component-equal MSE", title="Frozen convergence trajectories"); axes[0].legend(frameon=False)
    axes[1].scatter(np.diag(anchor_matrices["observed_near"]["total"])[eligible], np.diag(matrices["observed_near"]["total"])[eligible], s=8, alpha=.4)
    axes[1].set(xlabel="Anchor-12 signed diagonal", ylabel="Selected signed diagonal", title="Long-horizon attribution")
    baseline = np.asarray(scientific["arm_metrics"]["morphology_only"]["fold_mse"])
    for arm, label in (("observed_near", "near"), ("observed_annular", "annular"), ("within_fov_permuted_near", "permuted")):
        values = np.asarray(scientific["arm_metrics"][arm]["fold_mse"])
        axes[2].plot(range(4), 100 * (baseline - values) / baseline, marker="o", label=label)
    axes[2].axhline(2, color="black", linestyle=":", linewidth=.8); axes[2].set(xlabel="Outer fold", ylabel="MSE gain vs morphology (%)", title="Held-out prediction"); axes[2].legend(frameon=False)
    figure.tight_layout(); figure.savefig(output / "summary.png", dpi=180); plt.close(figure)
    near_summary = scientific["aggregate_summaries"]["observed_near"]["total"]
    report = f"""# Nonlinear same-gene convergence audit

## Outcome

- Verdict: **{verdict}**
- Practical convergence: **{convergence['saturated_trajectory_count']}/8 trajectories saturated**
- Original scientific gates: **{sum(value['passed'] for value in gates.values())}/7 passed**
- Failed scientific gates: `{', '.join(failed) if failed else 'none'}`
- Frozen common probes: **932/1,000**

The nine mutually-exclusive frozen verdict branches were applied in order:
anchor validity, convergence, strict support with/without full attribution,
both/one attribution gates, robust nonexclusive enrichment, then evidence
against a budget explanation. The near model changed component-equal MSE
by {100 * scientific['near_vs_morphology']['relative_mse_gain']:.3f}% versus
morphology and {100 * scientific['near_vs_permuted']['relative_mse_gain']:.3f}%
versus the within-FOV permutation. Its diagonal/off-diagonal ratio was
{near_summary['diagonal_offdiagonal_ratio']:.3f}, while same-name row top-1 and
top-1% fractions were {near_summary['row_top1_fraction']:.3f} and
{near_summary['row_top1_percent_fraction']:.3f}.

## Convergence gate

All four morphology and four observed-near tuning paths were traversed through
epoch 192. A path counts as saturated only if its selected epoch is below 192
or its relative validation-MSE improvement from 96 to 192 is at most 0.1%.
This audit does not extend the horizon after inspecting those outcomes.

## Frozen antecedent controls

The complete 16-value candidate-12 and 16-arm anchor-12 audit
**{'passed' if equivalence['passed'] else 'failed'}**. Candidate validation has
absolute tolerance 1e-7; fresh-refit component metrics and predictions use
1e-7; and the 12 neighbor-arm Jacobians use maximum absolute tolerance 1e-6.
The exact canonical-JSON gene-order hash and 932-probe mask hash also matched.
These are execution-equivalence controls, not independent evidence. An anchor
failure takes the first verdict branch and invalidates budget attribution.

## Secondary prediction and row attribution

The selected-minus-anchor12 **near-vs-morphology relative-gain difference** was
{100 * prediction_attribution['selected_minus_anchor12_relative_gain']:.3f}%.
Its same-resample 27-component bootstrap interval was
[{100 * prediction_attribution['bootstrap_2_5_percentile']:.3f}%,
 {100 * prediction_attribution['bootstrap_97_5_percentile']:.3f}%], and selected
near MSE improved in
{prediction_attribution['folds_selected_near_better_than_anchor12']}/4 folds.
Prediction-failure attribution therefore
**{'passed' if prediction_attribution['passed'] else 'failed'}**.

Row-failure attribution **{'passed' if row_attribution['passed'] else 'failed'}**:
selected aggregate top-1/top-1% were
{row_attribution['selected_aggregate']['row_top1_fraction']:.3f}/
{row_attribution['selected_aggregate']['row_top1_percent_fraction']:.3f} versus
fixed-12 values {row_attribution['fixed12_row_top1_fraction']:.3f}/
{row_attribution['fixed12_row_top1_percent_fraction']:.3f}, with foldwise
improvement in {row_attribution['folds_row_top1_improved']}/4 and
{row_attribution['folds_row_top1_percent_improved']}/4 matrices. The original
strict row gate must also pass; no post-outcome threshold is relaxed.

Descriptively, selected near MSE changed by
{100 * antecedent_comparison['prediction']['relative_mse_gain']:.3f}% versus the
12-epoch near model and {100 * ridge_comparison['prediction']['relative_mse_gain']:.3f}%
versus ridge. The ridge superiority check is secondary and does not change the
frozen attribution verdict.

## Adverse interpretation

The outer test folds and two slides were already observed in the antecedent,
so this is a post-hoc optimization audit rather than independent confirmation.
If even one convergence trajectory is unsaturated, the result is optimization
inconclusive regardless of favorable test metrics. Geometry components are
leakage-control units, not patients or biological replicates. The output is an
exploratory model-implied sensitivity and does not establish cell-cell
communication, mechanism, causality, patient generalization, or clinical validity.
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.project_root, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
    status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=paths.project_root, check=True, text=True, stdout=subprocess.PIPE).stdout
    _write_json(output / "provenance.json", {
        "git_commit": commit, "git_status": status, "command": sys.argv,
        "analysis_source": {"path": str(Path(__file__).relative_to(paths.project_root)), "sha256": sha256_file(Path(__file__))},
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256, "input_files": input_files,
        "input_runs": checks, "antecedent_input_runs": prior_checks,
        "ridge_input_runs": ridge_checks,
    })
    verification = {
        "verified": True, "input_run_count": 4, "folds": [0, 1, 2, 3], "attempt": ATTEMPT,
        "scientific_id": str(runs[0]["scientific_id"]), "input_bundle_checks": checks,
        "antecedent_bundle_checks": prior_checks,
        "ridge_bundle_checks": ridge_checks,
        "antecedent_attempt1_count": 4,
        "input_files": input_files, "antecedent_equivalence": equivalence,
        "gene_axis_sha256": GENE_AXIS_SHA256, "eligible_mask_sha256": ELIGIBLE_MASK_SHA256,
        "eligible_gene_count": 932, "convergence_gate_passed": convergence["passed"],
        "scientific_gates_passed": sum(value["passed"] for value in gates.values()),
        "antecedent_anchor_valid": bool(equivalence["passed"]),
        "prediction_attribution_gate_passed": bool(
            prediction_attribution["passed"]
        ),
        "row_attribution_gate_passed": bool(row_attribution["passed"]),
        "verdict": verdict,
    }
    _write_json(output / "verification.json", verification)
    _write_json(output / "registry_evaluation.json", {"evaluation_id": EVALUATION_ID, "registered": True})
    inventory = _analysis_inventory(output)
    _write_json(output / "analysis_manifest.json", {
        "evaluation_id": EVALUATION_ID, "files": inventory,
        "manifest_payload_sha256": canonical_sha256(inventory),
    })
    _verify_output(output)
    return payload


def _verify_existing(paths: Any, registry: Registry, output: Path) -> dict[str, Any]:
    verification = _verify_output(output)
    runs = _discover_convergence_runs(registry)
    if {str(item["run_id"]) for item in runs} != {str(item["run_id"]) for item in verification["input_bundle_checks"]}:
        raise ConvergenceAnalysisError("current convergence run set differs from published inputs")
    antecedent_runs, antecedent_attempt1 = _discover_antecedent_runs(registry)
    if len(antecedent_attempt1) != 4:
        raise ConvergenceAnalysisError("antecedent metadata-invalid attempt-1 audit changed")
    stored_antecedent = {str(item["run_id"]) for item in verification["antecedent_bundle_checks"]}
    if {str(item["run_id"]) for item in antecedent_runs} != stored_antecedent:
        raise ConvergenceAnalysisError("antecedent attempt-2 run set differs from published inputs")
    for item in (
        *verification["input_bundle_checks"],
        *verification["antecedent_bundle_checks"],
        *verification["ridge_bundle_checks"],
    ):
        if registry.verify_artifacts(run_id=str(item["run_id"])):
            raise ConvergenceAnalysisError(
                f"run registry artifacts changed: {item['run_id']}"
            )
    with registry.connect() as connection:
        reference_evaluations = {
            evaluation_id: connection.execute(
                "SELECT status,artifact_path FROM evaluations WHERE evaluation_id=?",
                (evaluation_id,),
            ).fetchone()
            for evaluation_id in (ANTECEDENT_EVALUATION_ID, RIDGE_EVALUATION_ID)
        }
    expected_reference_paths = {
        ANTECEDENT_EVALUATION_ID: paths.report_root / "analyses" / ANTECEDENT_REPORT,
        RIDGE_EVALUATION_ID: paths.report_root / "analyses" / RIDGE_REPORT,
    }
    for evaluation_id, row in reference_evaluations.items():
        if (
            row is None
            or row["status"] != "completed"
            or Path(row["artifact_path"]) != expected_reference_paths[evaluation_id]
        ):
            raise ConvergenceAnalysisError(
                f"reference evaluation registry changed: {evaluation_id}"
            )
    registry_check = _verify_registry(registry, output)
    return {
        "verified": True, "evaluation_id": EVALUATION_ID, "input_runs": 4,
        "eligible_genes": 932, "report": str(output / "report.md"),
        "manifest_sha256": sha256_file(output / "analysis_manifest.json"),
        "registry": registry_check, "input_and_output_hashes_verified": True,
    }


def main() -> None:
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    final = paths.report_root / "analyses" / REPORT_NAME
    if arguments.verify_only:
        print(canonical_json(_verify_existing(paths, registry, final)))
        return
    if final.exists():
        print(canonical_json(_verify_existing(paths, registry, final)))
        return
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{REPORT_NAME}.staging-", dir=final.parent))
    published = False
    previous_status: str | None = None
    try:
        payload = _build_analysis(paths, registry, staging)
        os.replace(staging, final)
        published = True
        try:
            previous_status = _register(registry, final, payload)
            result = _verify_existing(paths, registry, final)
        except BaseException:
            if previous_status not in {None, "already_registered"}:
                _compensate_registration(registry, previous_status)
            if final.exists() and not staging.exists():
                os.replace(final, staging)
                published = False
            raise
        print(canonical_json({**result, "verdict": payload["verdict"]}))
    except BaseException:
        # Preserve a failed staging bundle for forensic inspection; it is never
        # registered or mistaken for the atomic final report.
        raise
    finally:
        if not published and staging.exists() and not any(staging.iterdir()):
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
