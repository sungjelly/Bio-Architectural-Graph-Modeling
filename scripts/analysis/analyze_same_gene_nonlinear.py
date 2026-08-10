#!/usr/bin/env python3
"""Aggregate and verify the frozen nonlinear same-gene replication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_json, canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry, utc_now
from spatial_benchmark.run_archive import verify_run_bundle
from spatial_benchmark.same_gene_jacobian import diagonal_summary
from spatial_benchmark.same_gene_nonlinear import AdditiveNeighborMLP


CAMPAIGN_ID = "cmp_20260810_same_gene_nonlinear_replication"
REPORT_NAME = "same_gene_nonlinear_replication_20260810"
EVALUATION_ID = "eval_same_gene_nonlinear_replication_20260810_v1"
DATASET_ID = "gastric_cosmx_drive_public"
SPLIT_ID = "opaque_geometry_components_075mm_4fold_v1"
FROZEN_CONTRACT_SHA256 = (
    "5b9f0058e011084cf28ea3e1fc78ebba787e0cf07dd35adde19381c5f6feb66e"
)
ARMS = (
    "morphology_only",
    "observed_near",
    "observed_annular",
    "within_fov_permuted_near",
)
MATRIX_ARMS = ARMS[1:]
MATRIX_PARTS = ("total", "linear", "nonlinear")
PRIMARY_METRIC = "validation/observed_near_component_equal_mse"


class NonlinearAnalysisError(RuntimeError):
    """Raised when a frozen analysis input or output fails verification."""


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _configuration(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return json.loads(str(row["config_json"]))


def _discover_runs(registry: Registry) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with registry.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM runs WHERE campaign_id = ? ORDER BY created_at",
            (CAMPAIGN_ID,),
        ).fetchall()
    production: dict[int, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        config = _configuration(row)
        if config.get("profile") != "full":
            continue
        attempt = int(row["attempt"])
        fold = int(row["fold"])
        if attempt == 2 and row["status"] == "completed":
            if fold in production:
                raise NonlinearAnalysisError(f"duplicate production fold {fold}")
            production[fold] = row
        elif attempt == 1:
            excluded.append(
                {
                    "run_id": row["run_id"],
                    "fold": fold,
                    "attempt": attempt,
                    "status": row["status"],
                    "scientific_id": row["scientific_id"],
                    "reason": (
                        "metadata-invalid: trainer.fold_seed was not excluded from "
                        "scientific identity; numerical result retained only for "
                        "determinism comparison"
                    ),
                    "artifact_path": row["artifact_path"],
                }
            )
    if set(production) != {0, 1, 2, 3}:
        raise NonlinearAnalysisError(
            f"expected completed attempt-2 folds 0-3; found {sorted(production)}"
        )
    selected = [production[fold] for fold in range(4)]
    scientific_ids = {str(row["scientific_id"]) for row in selected}
    if scientific_ids != {"sci_12587b9d9e7f69f7"}:
        raise NonlinearAnalysisError(
            f"production folds do not share the corrected identity: {scientific_ids}"
        )
    if len(excluded) != 4 or {int(item["fold"]) for item in excluded} != {0, 1, 2, 3}:
        raise NonlinearAnalysisError("the four metadata-invalid attempt-1 runs are absent")
    return selected, sorted(excluded, key=lambda item: int(item["fold"]))


def _verify_native_manifest(bundle: Path) -> dict[str, Any]:
    path = bundle / "artifact_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise NonlinearAnalysisError(f"invalid native manifest: {path}")
    for item in files:
        relative = str(item["path"])
        target = bundle / relative
        if (
            not target.is_file()
            or target.stat().st_size != int(item["size_bytes"])
            or sha256_file(target) != str(item["sha256"])
        ):
            raise NonlinearAnalysisError(f"native payload mismatch: {target}")
    if canonical_sha256(files) != manifest.get("manifest_payload_sha256"):
        raise NonlinearAnalysisError("native artifact manifest digest is invalid")
    return {
        "file_count": len(files),
        "manifest_sha256": sha256_file(path),
        "payload_digest": manifest["manifest_payload_sha256"],
    }


def _verify_checkpoint(bundle: Path, result: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = bundle / "checkpoints/last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("run_id") != result["run_id"]
        or checkpoint.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256
    ):
        raise NonlinearAnalysisError("checkpoint run or contract identity is invalid")
    genes = tuple(str(value) for value in checkpoint["genes"])
    if len(genes) != 1000 or len(set(genes)) != 1000:
        raise NonlinearAnalysisError("checkpoint gene axis is invalid")
    if set(checkpoint["arms"]) != set(ARMS):
        raise NonlinearAnalysisError("checkpoint arm set is invalid")
    for arm in ARMS:
        state = checkpoint["arms"][arm]
        expected_neighbor = arm != "morphology_only"
        if bool(state["model_kwargs"]["use_neighbor"]) != expected_neighbor:
            raise NonlinearAnalysisError(f"checkpoint neighbor mode is invalid: {arm}")
        model = AdditiveNeighborMLP(**state["model_kwargs"])
        model.load_state_dict(state["state_dict"], strict=True)
        for value in state["state_dict"].values():
            if not bool(torch.isfinite(value).all().item()):
                raise NonlinearAnalysisError(f"checkpoint has nonfinite weights: {arm}")
        expected_epoch = int(result["arms"][arm]["selected_epoch"])
        if int(state["selected_epoch"]) != expected_epoch:
            raise NonlinearAnalysisError(f"checkpoint epoch mismatch: {arm}")
        for name, shape in (
            ("target_mean", (1000,)),
            ("target_std", (1000,)),
            ("morphology_median", (22,)),
            ("morphology_mean", (22,)),
            ("morphology_std", (22,)),
        ):
            tensor = state[name]
            if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                raise NonlinearAnalysisError(f"checkpoint normalizer is invalid: {arm}/{name}")
    return {"sha256": sha256_file(checkpoint_path), "genes": genes}


def _verify_bundle(row: dict[str, Any], registry: Registry) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = Path(str(row["artifact_path"]))
    canonical = verify_run_bundle(bundle)
    native = _verify_native_manifest(bundle)
    result = json.loads((bundle / "results.json").read_text(encoding="utf-8"))
    if (
        result.get("run_id") != row["run_id"]
        or result.get("profile") != "full"
        or int(result.get("outer_fold", -1)) != int(row["fold"])
        or result.get("status") != "completed"
    ):
        raise NonlinearAnalysisError(f"result identity mismatch: {bundle}")
    if result.get("dataset_fingerprints") != {
        "raw": "e1513d598d4ea910386842cdf4a6d9bd58e21318d484bdb34dfb3962f1490ea5",
        "processed": "6304132b4a57699c81b8616324dbeb2faee24b58ce70490be552595d84af34ce",
        "split": "12c0d46244ed443a482586fc85422672f9f04132c7def49a741c40ba48bf4264",
    }:
        raise NonlinearAnalysisError("result data fingerprints are invalid")
    predictions = [
        json.loads(line)
        for line in (bundle / "predictions/validation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    final_metrics = json.loads((bundle / "metrics/final.json").read_text(encoding="utf-8"))
    reproduced = float(np.mean([float(item["sample_loss"]) for item in predictions]))
    if not np.isclose(reproduced, final_metrics[PRIMARY_METRIC], rtol=1e-12, atol=0.0):
        raise NonlinearAnalysisError("canonical predictions do not reproduce primary MSE")
    groups = [int(item["sample_key"].rsplit("_", 1)[1]) for item in predictions]
    if len(groups) != len(set(groups)) or not predictions:
        raise NonlinearAnalysisError("canonical prediction component keys are invalid")
    checkpoint = _verify_checkpoint(bundle, result)
    registry_issues = registry.verify_artifacts(run_id=str(row["run_id"]))
    if registry_issues:
        raise NonlinearAnalysisError(f"registry artifact verification failed: {registry_issues}")
    check = {
        "run_id": row["run_id"],
        "fold": int(row["fold"]),
        "attempt": int(row["attempt"]),
        "scientific_id": row["scientific_id"],
        "bundle": str(bundle),
        "canonical_bundle": canonical,
        "native_manifest": native,
        "checkpoint_sha256": checkpoint["sha256"],
        "canonical_prediction_rows": len(predictions),
        "canonical_primary_reproduced": True,
        "registry_artifacts_verified": True,
    }
    return result, {**check, "genes": checkpoint["genes"]}


def _load_fold_arrays(bundle: Path) -> dict[str, np.ndarray]:
    with np.load(bundle / "nonlinear_jacobians.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    expected = {"genes"}
    for arm in MATRIX_ARMS:
        expected.update(
            {
                f"{arm}_total",
                f"{arm}_linear",
                f"{arm}_nonlinear",
                f"{arm}_mean_hidden_derivative",
                f"eligible_{arm}",
            }
        )
    expected.add("eligible_morphology_only")
    if set(arrays) != expected:
        raise NonlinearAnalysisError(
            f"unexpected nonlinear array keys: {sorted(set(arrays) ^ expected)}"
        )
    for arm in MATRIX_ARMS:
        for part in MATRIX_PARTS:
            value = arrays[f"{arm}_{part}"]
            if value.shape != (1000, 1000) or not bool(np.isfinite(value).all()):
                raise NonlinearAnalysisError(f"invalid matrix {arm}/{part}")
        hidden = arrays[f"{arm}_mean_hidden_derivative"]
        if hidden.shape != (64,) or not bool(np.isfinite(hidden).all()):
            raise NonlinearAnalysisError(f"invalid hidden derivative {arm}")
    return arrays


def _verify_attempt_replay(
    selected: list[dict[str, Any]], excluded: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    old_by_fold = {int(item["fold"]): item for item in excluded}
    checks: list[dict[str, Any]] = []
    for row in selected:
        fold = int(row["fold"])
        old = old_by_fold[fold]
        old_arrays = _load_fold_arrays(Path(str(old["artifact_path"])))
        new_arrays = _load_fold_arrays(Path(str(row["artifact_path"])))
        equal = set(old_arrays) == set(new_arrays) and all(
            np.array_equal(old_arrays[name], new_arrays[name]) for name in old_arrays
        )
        if not equal:
            raise NonlinearAnalysisError(
                f"metadata-only attempt correction changed numerical arrays for fold {fold}"
            )
        checks.append(
            {
                "fold": fold,
                "excluded_run_id": old["run_id"],
                "selected_run_id": row["run_id"],
                "all_npz_arrays_exactly_equal": True,
            }
        )
    return checks


def _arm_metrics(results: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fold_mse: list[float] = []
    fold_mae: list[float] = []
    for result in results:
        evaluation = result["arms"][arm]["evaluation"]
        fold_mse.append(float(evaluation["component_equal_mse"]))
        fold_mae.append(float(evaluation["component_equal_mae"]))
        rows.extend(evaluation["per_component"])
    groups = [int(item["geometry_group"]) for item in rows]
    if len(groups) != 27 or len(set(groups)) != 27:
        raise NonlinearAnalysisError(f"{arm} does not cover 27 unique components")
    return {
        "component_equal_mse": float(np.mean([item["mse"] for item in rows])),
        "component_equal_mae": float(np.mean([item["mae"] for item in rows])),
        "fold_mse": fold_mse,
        "fold_mae": fold_mae,
        "per_component": rows,
    }


def _paired_gain(
    baseline_rows: Iterable[dict[str, Any]],
    model_rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    draws: int = 10_000,
) -> dict[str, float | int]:
    baseline = {
        int(item["geometry_group"]): float(item["mse"]) for item in baseline_rows
    }
    model = {int(item["geometry_group"]): float(item["mse"]) for item in model_rows}
    if set(baseline) != set(model) or len(baseline) != 27:
        raise NonlinearAnalysisError("paired component sets differ")
    groups = sorted(baseline)
    base = np.asarray([baseline[group] for group in groups], dtype=np.float64)
    fitted = np.asarray([model[group] for group in groups], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        index = rng.integers(0, len(groups), size=len(groups))
        base_mean = float(np.mean(base[index]))
        samples[draw] = (base_mean - float(np.mean(fitted[index]))) / base_mean
    return {
        "relative_mse_gain": float((base.mean() - fitted.mean()) / base.mean()),
        "bootstrap_2_5_percentile": float(np.quantile(samples, 0.025)),
        "bootstrap_97_5_percentile": float(np.quantile(samples, 0.975)),
        "bootstrap_draws": draws,
        "geometry_component_count": len(groups),
    }


def _row_ranks(matrix: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    index = np.flatnonzero(eligible)
    absolute = np.abs(matrix[np.ix_(index, index)])
    return np.asarray(
        [1 + int(np.sum(row > row[position])) for position, row in enumerate(absolute)],
        dtype=np.int32,
    )


def _random_alignment_null(
    matrix: np.ndarray, eligible: np.ndarray, *, draws: int = 1000
) -> dict[str, float | int]:
    index = np.flatnonzero(eligible)
    absolute = np.abs(matrix[np.ix_(index, index)])
    off = float(np.median(absolute[~np.eye(len(index), dtype=bool)]))
    observed = float(np.median(np.diag(absolute)) / off)
    maximum = absolute.max(axis=1)
    rng = np.random.default_rng(20260810)
    ratios = np.empty(draws, dtype=np.float64)
    top1 = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        columns = rng.integers(0, len(index), size=len(index))
        values = absolute[np.arange(len(index)), columns]
        ratios[draw] = float(np.median(values) / off)
        top1[draw] = float(np.mean(values >= maximum))
    return {
        "draws": draws,
        "observed_ratio": observed,
        "null_ratio_median": float(np.median(ratios)),
        "null_ratio_95th_percentile": float(np.quantile(ratios, 0.95)),
        "empirical_upper_p": float((1 + np.sum(ratios >= observed)) / (draws + 1)),
        "null_top1_fraction_median": float(np.median(top1)),
    }


def _ridge_comparison(
    paths: Any,
    nonlinear_metrics: dict[str, dict[str, Any]],
    nonlinear_matrix: np.ndarray,
    eligible: np.ndarray,
    genes: tuple[str, ...],
) -> dict[str, Any]:
    ridge_root = (
        paths.report_root
        / "analyses/same_gene_cross_cell_jacobian_20260810_v2"
    )
    ridge_results = json.loads(
        (ridge_root / "aggregate_results.json").read_text(encoding="utf-8")
    )
    ridge_runs: list[dict[str, Any]] = []
    for item in ridge_results["input_runs"]:
        verify_run_bundle(Path(item["bundle"]))
        result = json.loads(
            (Path(item["bundle"]) / "results.json").read_text(encoding="utf-8")
        )
        ridge_runs.append(result)
    ridge_runs.sort(key=lambda item: int(item["outer_fold"]))
    ridge_rows: list[dict[str, Any]] = []
    ridge_fold_mse: list[float] = []
    for result in ridge_runs:
        block = result["arms"]["observed_near"].get("test")
        if block is None:
            block = result["arms"]["observed_near"]["evaluation"]
        ridge_rows.extend(block["per_component"])
        ridge_fold_mse.append(float(block["component_equal_mse"]))
    prediction = _paired_gain(
        ridge_rows,
        nonlinear_metrics["observed_near"]["per_component"],
        seed=20260813,
    )
    nonlinear_fold = np.asarray(
        nonlinear_metrics["observed_near"]["fold_mse"], dtype=np.float64
    )
    ridge_fold = np.asarray(ridge_fold_mse, dtype=np.float64)
    with np.load(ridge_root / "aggregate_jacobians.npz", allow_pickle=False) as archive:
        ridge_genes = tuple(str(value) for value in archive["genes"])
        if ridge_genes != genes:
            raise NonlinearAnalysisError("ridge and nonlinear gene axes differ")
        ridge_matrix = archive["jacobian_observed_near"].astype(np.float64)
        ridge_eligible = archive["eligible"].astype(bool)
    if (
        ridge_matrix.shape != (1000, 1000)
        or ridge_eligible.shape != (1000,)
        or not bool(np.isfinite(ridge_matrix).all())
    ):
        raise NonlinearAnalysisError("ridge aggregate matrix or eligibility is invalid")
    common = eligible & ridge_eligible
    nonlinear_summary = diagonal_summary(nonlinear_matrix, common).as_dict()
    ridge_summary = diagonal_summary(ridge_matrix, common).as_dict()
    nonlinear_diagonal = np.diag(nonlinear_matrix)[common]
    ridge_diagonal = np.diag(ridge_matrix)[common]
    return {
        "baseline_report": str(ridge_root / "report.md"),
        "baseline_bundles_verified": True,
        "gene_axis_identical": True,
        "common_eligible_genes": int(np.sum(common)),
        "nonlinear_relative_mse_gain_vs_ridge": prediction,
        "fold_relative_gains_vs_ridge": (
            (ridge_fold - nonlinear_fold) / ridge_fold
        ).tolist(),
        "folds_nonlinear_better": int(np.sum(nonlinear_fold < ridge_fold)),
        "nonlinear_outperforms_ridge": bool(
            prediction["bootstrap_2_5_percentile"] > 0
            and int(np.sum(nonlinear_fold < ridge_fold)) >= 3
        ),
        "aggregate_jacobian_summary_nonlinear": nonlinear_summary,
        "aggregate_jacobian_summary_ridge": ridge_summary,
        "signed_diagonal_spearman": float(
            spearmanr(nonlinear_diagonal, ridge_diagonal).statistic
        ),
        "signed_diagonal_sign_agreement": float(
            np.mean(np.sign(nonlinear_diagonal) == np.sign(ridge_diagonal))
        ),
        "ridge_matrix": ridge_matrix,
        "common_mask": common,
    }


def _plot(
    output: Path,
    metrics: dict[str, dict[str, Any]],
    matrices: dict[str, dict[str, np.ndarray]],
    eligible: np.ndarray,
    ridge: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    baseline = np.asarray(metrics["morphology_only"]["fold_mse"])
    for arm, label, color in (
        ("observed_near", "0–25 µm", "#1f77b4"),
        ("observed_annular", "25–50 µm", "#ff7f0e"),
        ("within_fov_permuted_near", "permuted near", "#7f7f7f"),
    ):
        value = np.asarray(metrics[arm]["fold_mse"])
        axes[0].plot(
            range(4),
            100 * (baseline - value) / baseline,
            marker="o",
            label=label,
            color=color,
        )
    axes[0].axhline(2, color="black", linestyle=":", linewidth=0.8)
    axes[0].set(
        xlabel="Outer geometry fold",
        ylabel="MSE gain vs morphology (%)",
        title="Held-out predictive gain",
    )
    axes[0].set_xticks(range(4))
    axes[0].legend(frameon=False, fontsize=8)

    index = np.flatnonzero(eligible)
    rng = np.random.default_rng(20260810)
    near = np.abs(matrices["observed_near"]["total"][np.ix_(index, index)])
    axes[1].hist(
        np.diag(near), bins=60, density=True, histtype="step", linewidth=1.6,
        label="same-name diagonal", color="#1f77b4"
    )
    off = near[~np.eye(len(index), dtype=bool)]
    axes[1].hist(
        rng.choice(off, size=20_000, replace=False), bins=60, density=True,
        histtype="step", linewidth=1.4, linestyle="--",
        label="off-diagonal", color="#2ca02c"
    )
    axes[1].set(
        xlabel="Absolute standardized derivative",
        ylabel="Density",
        title="Enriched, not exclusive",
    )
    axes[1].legend(frameon=False, fontsize=8)

    common = ridge["common_mask"]
    x = np.diag(ridge["ridge_matrix"])[common]
    y = np.diag(matrices["observed_near"]["total"])[common]
    axes[2].scatter(x, y, s=8, alpha=0.45, color="#9467bd")
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].axvline(0, color="black", linewidth=0.6)
    axes[2].set(
        xlabel="Ridge signed diagonal",
        ylabel="Nonlinear signed diagonal",
        title="Nonlinear vs ridge",
    )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _analysis_inventory(output: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in output.iterdir() if item.is_file()):
        if path.name == "analysis_manifest.json":
            continue
        records.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _verify_output_files(output: Path) -> dict[str, Any]:
    required = {
        "aggregate_results.json",
        "aggregate_jacobians.npz",
        "eligible_gene_diagonal_summary.csv",
        "summary.png",
        "report.md",
        "provenance.json",
        "verification.json",
        "registry_evaluation.json",
        "analysis_manifest.json",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise NonlinearAnalysisError(f"analysis files are missing: {missing}")
    manifest = json.loads((output / "analysis_manifest.json").read_text())
    files = manifest["files"]
    if canonical_sha256(files) != manifest["manifest_payload_sha256"]:
        raise NonlinearAnalysisError("analysis manifest digest is invalid")
    for item in files:
        path = output / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(item["size_bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            raise NonlinearAnalysisError(f"analysis payload mismatch: {path}")
    verification = json.loads((output / "verification.json").read_text())
    if verification.get("verified") is not True or verification.get("input_run_count") != 4:
        raise NonlinearAnalysisError("stored analysis verification is incomplete")
    with np.load(output / "aggregate_jacobians.npz", allow_pickle=False) as archive:
        if int(np.sum(archive["eligible"])) != 932:
            raise NonlinearAnalysisError("stored common eligibility changed")
        for arm in MATRIX_ARMS:
            for part in MATRIX_PARTS:
                matrix = archive[f"jacobian_{arm}_{part}"]
                if matrix.shape != (1000, 1000) or not bool(np.isfinite(matrix).all()):
                    raise NonlinearAnalysisError(f"stored matrix is invalid: {arm}/{part}")
    for item in verification["input_bundle_checks"]:
        verify_run_bundle(Path(item["bundle"]))
    return verification


def _register_analysis(registry: Registry, output: Path) -> bool:
    """Atomically register an already verified, atomically published analysis."""

    aggregate = json.loads((output / "aggregate_results.json").read_text())
    input_runs = aggregate["input_runs"]
    primary_run_id = str(input_runs[0]["run_id"])
    gates = aggregate["gates"]
    metrics = {
        "verdict": aggregate["verdict"],
        "frozen_gates_passed": int(
            sum(bool(item["passed"]) for item in gates.values())
        ),
        "near_gain_vs_morphology": aggregate["prediction"]["near_vs_morphology"][
            "relative_mse_gain"
        ],
        "near_gain_vs_permuted": aggregate["prediction"]["near_vs_permuted"][
            "relative_mse_gain"
        ],
        "diagonal_offdiagonal_ratio": aggregate["jacobian"]["aggregate_summaries"][
            "observed_near"
        ]["total"]["diagonal_offdiagonal_ratio"],
        "row_top1_fraction": aggregate["jacobian"]["aggregate_summaries"][
            "observed_near"
        ]["total"]["row_top1_fraction"],
    }
    files = sorted(item for item in output.iterdir() if item.is_file())
    now = utc_now()
    with registry.transaction(immediate=True) as connection:
        existing = connection.execute(
            "SELECT status, artifact_path FROM evaluations WHERE evaluation_id = ?",
            (EVALUATION_ID,),
        ).fetchone()
        if existing is not None:
            if (
                existing["status"] == "completed"
                and Path(existing["artifact_path"]) == output
            ):
                return False
            raise NonlinearAnalysisError(
                f"conflicting evaluation registry row: {EVALUATION_ID}"
            )
        connection.execute(
            """
            INSERT INTO evaluations(
                evaluation_id, run_id, checkpoint_name, dataset_id, split_id,
                status, metrics_json, artifact_path, created_at, finished_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                EVALUATION_ID,
                primary_run_id,
                "four_fold_nonlinear_aggregate",
                DATASET_ID,
                SPLIT_ID,
                "completed",
                json.dumps(
                    metrics, sort_keys=True, separators=(",", ":"), allow_nan=False
                ),
                str(output),
                now,
                now,
            ),
        )
        connection.executemany(
            """
            INSERT INTO artifacts(
                run_id, evaluation_id, kind, path, sha256, size_bytes,
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    primary_run_id,
                    EVALUATION_ID,
                    "aggregate_analysis",
                    str(path),
                    sha256_file(path),
                    path.stat().st_size,
                    "present",
                    now,
                )
                for path in files
            ],
        )
        cursor = connection.execute(
            "UPDATE campaigns SET status = ?, updated_at = ? WHERE campaign_id = ?",
            ("complete", now, CAMPAIGN_ID),
        )
        if cursor.rowcount != 1:
            raise NonlinearAnalysisError("campaign registry row is absent")
    return True


def _verify_existing(
    output: Path, registry: Registry, *, verify_data: bool = True
) -> dict[str, Any]:
    verification = _verify_output_files(output)
    paths = current_paths()
    if verify_data:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/data/prepare_same_gene_jacobian.py",
                "--verify-only",
                "--verify-raw",
            ],
            cwd=paths.project_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        data_check = json.loads(completed.stdout.strip())
        if data_check.get("verified") is not True:
            raise NonlinearAnalysisError("raw/prepared data re-verification failed")
    runs, excluded = _discover_runs(registry)
    _verify_attempt_replay(runs, excluded)
    for run in runs:
        if registry.verify_artifacts(run_id=str(run["run_id"])):
            raise NonlinearAnalysisError(
                f"registered run artifacts failed verification: {run['run_id']}"
            )
    ridge_root = (
        paths.report_root / "analyses/same_gene_cross_cell_jacobian_20260810_v2"
    )
    ridge_results = json.loads(
        (ridge_root / "aggregate_results.json").read_text(encoding="utf-8")
    )
    for item in ridge_results["input_runs"]:
        verify_run_bundle(Path(item["bundle"]))
    with np.load(output / "aggregate_jacobians.npz", allow_pickle=False) as nonlinear:
        nonlinear_genes = tuple(str(value) for value in nonlinear["genes"])
    with np.load(ridge_root / "aggregate_jacobians.npz", allow_pickle=False) as ridge:
        ridge_genes = tuple(str(value) for value in ridge["genes"])
        ridge_matrix = ridge["jacobian_observed_near"]
        if ridge_matrix.shape != (1000, 1000) or not bool(
            np.isfinite(ridge_matrix).all()
        ):
            raise NonlinearAnalysisError("ridge aggregate matrix failed re-verification")
    if nonlinear_genes != ridge_genes:
        raise NonlinearAnalysisError("ridge/nonlinear gene axes changed")
    with registry.connect() as connection:
        row = connection.execute(
            "SELECT status, artifact_path FROM evaluations WHERE evaluation_id = ?",
            (EVALUATION_ID,),
        ).fetchone()
        campaign = connection.execute(
            "SELECT status FROM campaigns WHERE campaign_id = ?", (CAMPAIGN_ID,)
        ).fetchone()
        artifacts = connection.execute(
            """
            SELECT path, sha256, size_bytes, status FROM artifacts
            WHERE evaluation_id = ? ORDER BY path
            """,
            (EVALUATION_ID,),
        ).fetchall()
    if (
        row is None
        or row["status"] != "completed"
        or Path(row["artifact_path"]) != output
        or campaign is None
        or campaign["status"] != "complete"
    ):
        raise NonlinearAnalysisError("analysis registry state is incomplete")
    expected_paths = {path.resolve() for path in output.iterdir() if path.is_file()}
    registered_paths = {Path(item["path"]).resolve() for item in artifacts}
    if registered_paths != expected_paths:
        raise NonlinearAnalysisError("evaluation artifact inventory is incomplete")
    for item in artifacts:
        path = Path(item["path"])
        if (
            item["status"] != "present"
            or not path.is_file()
            or path.stat().st_size != int(item["size_bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            raise NonlinearAnalysisError(f"evaluation artifact mismatch: {path}")
    return {
        "verified": True,
        "evaluation_id": EVALUATION_ID,
        "input_runs": 4,
        "eligible_genes": 932,
        "report": str(output / "report.md"),
        "manifest_sha256": sha256_file(output / "analysis_manifest.json"),
        "raw_and_prepared_reverified": verify_data,
        "attempt1_to_attempt2_replay_reverified": True,
        "evaluation_artifacts_verified": len(artifacts),
    }


def main() -> None:
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    final_output = paths.report_root / "analyses" / REPORT_NAME
    if arguments.verify_only:
        print(canonical_json(_verify_existing(final_output, registry)))
        return
    if final_output.exists():
        _verify_output_files(final_output)
        _register_analysis(registry, final_output)
        print(canonical_json(_verify_existing(final_output, registry)))
        return
    output = final_output.parent / f".{REPORT_NAME}.staging-{os.getpid()}"
    if output.exists():
        raise FileExistsError(f"analysis staging path already exists: {output}")

    contract = (
        paths.project_root
        / "experiments/campaigns"
        / CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    if sha256_file(contract) != FROZEN_CONTRACT_SHA256:
        raise NonlinearAnalysisError("frozen task contract changed")
    verification_command = [
        sys.executable,
        "scripts/data/prepare_same_gene_jacobian.py",
        "--verify-only",
        "--verify-raw",
    ]
    verified_data = subprocess.run(
        verification_command,
        cwd=paths.project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )
    data_verification = json.loads(verified_data.stdout.strip())
    if data_verification.get("verified") is not True:
        raise NonlinearAnalysisError("prepared/raw data verification failed")

    runs, excluded = _discover_runs(registry)
    replay_checks = _verify_attempt_replay(runs, excluded)
    results: list[dict[str, Any]] = []
    bundle_checks: list[dict[str, Any]] = []
    genes: tuple[str, ...] | None = None
    fold_arrays: list[dict[str, np.ndarray]] = []
    for row in runs:
        result, check = _verify_bundle(row, registry)
        run_genes = tuple(check.pop("genes"))
        if genes is None:
            genes = run_genes
        elif run_genes != genes:
            raise NonlinearAnalysisError("production gene axes differ")
        results.append(result)
        bundle_checks.append(check)
        fold_arrays.append(_load_fold_arrays(Path(str(row["artifact_path"]))))
    assert genes is not None
    if any(tuple(str(value) for value in arrays["genes"]) != genes for arrays in fold_arrays):
        raise NonlinearAnalysisError("Jacobian and checkpoint gene axes differ")

    eligible = np.logical_and.reduce(
        [arrays["eligible_observed_near"].astype(bool) for arrays in fold_arrays]
    )
    if int(np.sum(eligible)) != 932:
        raise NonlinearAnalysisError("common eligible gene count changed")
    fold_matrices = {
        arm: {
            part: [
                arrays[f"{arm}_{part}"].astype(np.float64) for arrays in fold_arrays
            ]
            for part in MATRIX_PARTS
        }
        for arm in MATRIX_ARMS
    }
    matrices = {
        arm: {
            part: np.mean(values, axis=0)
            for part, values in arm_parts.items()
        }
        for arm, arm_parts in fold_matrices.items()
    }
    hidden_derivatives = {
        arm: np.mean(
            [arrays[f"{arm}_mean_hidden_derivative"] for arrays in fold_arrays],
            axis=0,
        )
        for arm in MATRIX_ARMS
    }
    summaries = {
        arm: {
            part: diagonal_summary(matrix, eligible).as_dict()
            for part, matrix in arm_parts.items()
        }
        for arm, arm_parts in matrices.items()
    }
    fold_summaries = {
        arm: [
            diagonal_summary(matrix, eligible).as_dict()
            for matrix in fold_matrices[arm]["total"]
        ]
        for arm in MATRIX_ARMS
    }
    metrics = {arm: _arm_metrics(results, arm) for arm in ARMS}
    near_vs_morph = _paired_gain(
        metrics["morphology_only"]["per_component"],
        metrics["observed_near"]["per_component"],
        seed=20260810,
    )
    near_vs_perm = _paired_gain(
        metrics["within_fov_permuted_near"]["per_component"],
        metrics["observed_near"]["per_component"],
        seed=20260811,
    )
    annular_vs_morph = _paired_gain(
        metrics["morphology_only"]["per_component"],
        metrics["observed_annular"]["per_component"],
        seed=20260812,
    )
    near_fold = np.asarray(metrics["observed_near"]["fold_mse"])
    morph_fold = np.asarray(metrics["morphology_only"]["fold_mse"])
    perm_fold = np.asarray(metrics["within_fov_permuted_near"]["fold_mse"])
    near_favors_morph = int(np.sum(near_fold < morph_fold))
    near_favors_perm = int(np.sum(near_fold < perm_fold))

    fold_diagonal = np.asarray(
        [np.diag(matrix)[eligible] for matrix in fold_matrices["observed_near"]["total"]]
    )
    correlations = [
        float(spearmanr(fold_diagonal[first], fold_diagonal[second]).statistic)
        for first in range(4)
        for second in range(first + 1, 4)
    ]
    positive = np.sum(fold_diagonal > 0, axis=0)
    negative = np.sum(fold_diagonal < 0, axis=0)
    sign_fraction = float(np.mean(np.maximum(positive, negative) >= 3))
    fold_near_perm_ratios = [
        float(
            np.median(np.abs(np.diag(near)[eligible]))
            / np.median(np.abs(np.diag(permuted)[eligible]))
        )
        for near, permuted in zip(
            fold_matrices["observed_near"]["total"],
            fold_matrices["within_fov_permuted_near"]["total"],
            strict=True,
        )
    ]
    near_summary = summaries["observed_near"]["total"]
    permuted_summary = summaries["within_fov_permuted_near"]["total"]
    near_perm_ratio = float(
        near_summary["median_absolute_diagonal"]
        / permuted_summary["median_absolute_diagonal"]
    )
    random_null = _random_alignment_null(
        matrices["observed_near"]["total"], eligible
    )
    selected_epochs = {
        arm: [int(result["arms"][arm]["selected_epoch"]) for result in results]
        for arm in ARMS
    }
    epoch_boundary = all(epoch == 12 for values in selected_epochs.values() for epoch in values)
    technical_passed = (
        data_verification["verified"] is True
        and all(result["controls"]["all_outputs_finite"] is True for result in results)
        and all(
            result["controls"]["analytical_nonlinear_jacobian"]["passed"] is True
            for result in results
        )
        and all(
            result["controls"]["identity_oracle_row_top1_fraction"] == 1.0
            for result in results
        )
        and all(result["controls"]["receiver_expression_input"] is False for result in results)
        and all(float(result["peak_vram_gb"]) <= 20.5 for result in results)
        and all(bool(np.isfinite(matrix).all()) for arm in matrices.values() for matrix in arm.values())
    )
    gates = {
        "near_vs_morphology_prediction": {
            "passed": bool(
                near_vs_morph["relative_mse_gain"] >= 0.02
                and near_favors_morph >= 3
            ),
            "gain": near_vs_morph["relative_mse_gain"],
            "minimum": 0.02,
            "folds": near_favors_morph,
            "minimum_folds": 3,
        },
        "near_vs_permuted_prediction": {
            "passed": bool(
                near_vs_perm["relative_mse_gain"] >= 0.01
                and near_favors_perm >= 3
            ),
            "gain": near_vs_perm["relative_mse_gain"],
            "minimum": 0.01,
            "folds": near_favors_perm,
            "minimum_folds": 3,
        },
        "diagonal_enrichment": {
            "passed": bool(near_summary["diagonal_offdiagonal_ratio"] >= 2.0),
            "observed": near_summary["diagonal_offdiagonal_ratio"],
            "minimum": 2.0,
        },
        "diagonal_row_selectivity": {
            "passed": bool(
                near_summary["row_top1_fraction"] >= 0.25
                and near_summary["row_top1_percent_fraction"] >= 0.50
            ),
            "top1": near_summary["row_top1_fraction"],
            "top1_minimum": 0.25,
            "top1_percent": near_summary["row_top1_percent_fraction"],
            "top1_percent_minimum": 0.50,
        },
        "near_vs_permuted_diagonal": {
            "passed": bool(
                near_perm_ratio >= 1.25
                and int(np.sum(np.asarray(fold_near_perm_ratios) >= 1.25)) >= 3
            ),
            "aggregate_ratio": near_perm_ratio,
            "minimum": 1.25,
            "fold_ratios": fold_near_perm_ratios,
            "folds_at_or_above": int(
                np.sum(np.asarray(fold_near_perm_ratios) >= 1.25)
            ),
        },
        "fold_stability": {
            "passed": bool(
                float(np.median(correlations)) >= 0.70 and sign_fraction >= 0.75
            ),
            "median_spearman": float(np.median(correlations)),
            "minimum_spearman": 0.70,
            "sign_fraction": sign_fraction,
            "minimum_sign_fraction": 0.75,
        },
        "technical_controls": {"passed": bool(technical_passed)},
    }
    failed = [name for name, item in gates.items() if not item["passed"]]
    all_passed = not failed
    verdict = (
        "strict_same_gene_selectivity_supported"
        if all_passed
        else "strict_same_gene_selectivity_not_supported"
    )

    ridge = _ridge_comparison(
        paths, metrics, matrices["observed_near"]["total"], eligible, genes
    )
    ridge_serializable = {
        key: value
        for key, value in ridge.items()
        if key not in {"ridge_matrix", "common_mask"}
    }
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "evaluation_id": EVALUATION_ID,
        "verdict": verdict,
        "all_frozen_gates_passed": all_passed,
        "frozen_gates_passed": int(sum(item["passed"] for item in gates.values())),
        "frozen_gate_count": len(gates),
        "failed_gates": failed,
        "eligible_gene_count": int(np.sum(eligible)),
        "total_gene_count": len(genes),
        "prediction": {
            "arm_metrics": metrics,
            "near_vs_morphology": near_vs_morph,
            "near_vs_permuted": near_vs_perm,
            "annular_vs_morphology": annular_vs_morph,
            "near_favors_morphology_folds": near_favors_morph,
            "near_favors_permuted_folds": near_favors_perm,
        },
        "jacobian": {
            "aggregate_summaries": summaries,
            "fold_total_summaries": fold_summaries,
            "fold_pairwise_signed_diagonal_spearman": correlations,
            "median_pairwise_signed_diagonal_spearman": float(np.median(correlations)),
            "sign_consistent_gene_fraction": sign_fraction,
            "near_vs_permuted_diagonal_ratio": near_perm_ratio,
            "fold_near_vs_permuted_diagonal_ratios": fold_near_perm_ratios,
            "random_alignment_null": random_null,
        },
        "ridge_comparison": ridge_serializable,
        "selected_epochs": selected_epochs,
        "all_selected_epochs_at_upper_boundary": epoch_boundary,
        "gates": gates,
        "input_runs": bundle_checks,
        "excluded_metadata_invalid_attempt1": excluded,
        "attempt_correction_determinism_checks": replay_checks,
        "data_verification": data_verification,
        "maximum_claim": (
            "exploratory held-out-geometry graph-alignment-dependent nonlinear "
            "model sensitivity within two slides"
        ),
    }

    output.mkdir(parents=True)
    _write_json(output / "aggregate_results.json", payload)
    np.savez_compressed(
        output / "aggregate_jacobians.npz",
        genes=np.asarray(genes),
        eligible=eligible,
        **{
            f"jacobian_{arm}_{part}": matrix.astype(np.float32)
            for arm, parts in matrices.items()
            for part, matrix in parts.items()
        },
        **{
            f"mean_hidden_derivative_{arm}": value.astype(np.float32)
            for arm, value in hidden_derivatives.items()
        },
    )
    ranks = _row_ranks(matrices["observed_near"]["total"], eligible)
    eligible_index = np.flatnonzero(eligible)
    common = ridge["common_mask"]
    ridge_by_index = np.full(1000, np.nan, dtype=np.float64)
    ridge_by_index[common] = np.diag(ridge["ridge_matrix"])[common]
    table = pd.DataFrame(
        {
            "gene": [genes[index] for index in eligible_index],
            "near_total_diagonal": np.diag(matrices["observed_near"]["total"])[eligible],
            "near_linear_diagonal": np.diag(matrices["observed_near"]["linear"])[eligible],
            "near_nonlinear_diagonal": np.diag(matrices["observed_near"]["nonlinear"])[eligible],
            "annular_total_diagonal": np.diag(matrices["observed_annular"]["total"])[eligible],
            "permuted_total_diagonal": np.diag(matrices["within_fov_permuted_near"]["total"])[eligible],
            "ridge_near_diagonal": ridge_by_index[eligible_index],
            "near_absolute_row_rank": ranks,
            "positive_fold_count": positive,
            "negative_fold_count": negative,
        }
    )
    table.to_csv(output / "eligible_gene_diagonal_summary.csv", index=False)
    _plot(output / "summary.png", metrics, matrices, eligible, ridge)

    top = table.loc[
        table["near_total_diagonal"].abs().sort_values(ascending=False).index
    ].head(10)
    top_lines = "\n".join(
        f"| {row.gene} | {row.near_total_diagonal:.5f} | "
        f"{row.near_linear_diagonal:.5f} | {row.near_nonlinear_diagonal:.5f} | "
        f"{int(row.near_absolute_row_rank)} |"
        for row in top.itertuples()
    )
    report = f"""# Nonlinear same-gene cross-cell replication

## Outcome

- Verdict: **{verdict}**
- Frozen gates passed: **{sum(item['passed'] for item in gates.values())}/{len(gates)}**
- Failed gates: `{', '.join(failed) if failed else 'none'}`
- Data: 407,999 cells, 451 FOVs, 27 opaque geometry groups, 1,000 probes
- Common eligible probes: {int(np.sum(eligible))}
- Corrected production identity: `sci_12587b9d9e7f69f7` (attempt 2, folds 0–3)

The narrow answer is **no: matching RNA names were enriched, but high
sensitivities were not exclusive to matching names**.  The aggregate absolute
same-name diagonal was {near_summary['diagonal_offdiagonal_ratio']:.3f} times
the off-diagonal median.  Yet the matching source was row top-1 for only
{100 * near_summary['row_top1_fraction']:.2f}% of eligible targets and in the
row top 1% for {100 * near_summary['row_top1_percent_fraction']:.2f}%, below
the frozen 25% and 50% thresholds.

## Frozen gates

| Gate | Observed | Threshold | Result |
|---|---:|---:|---|
| near vs morphology prediction | {100 * near_vs_morph['relative_mse_gain']:.2f}%; {near_favors_morph}/4 folds | ≥2%; ≥3/4 | {'pass' if gates['near_vs_morphology_prediction']['passed'] else 'fail'} |
| near vs permuted prediction | {100 * near_vs_perm['relative_mse_gain']:.2f}%; {near_favors_perm}/4 folds | ≥1%; ≥3/4 | {'pass' if gates['near_vs_permuted_prediction']['passed'] else 'fail'} |
| diagonal / off-diagonal | {near_summary['diagonal_offdiagonal_ratio']:.3f} | ≥2.0 | {'pass' if gates['diagonal_enrichment']['passed'] else 'fail'} |
| row top-1; row top-1% | {near_summary['row_top1_fraction']:.3f}; {near_summary['row_top1_percent_fraction']:.3f} | ≥0.25; ≥0.50 | {'pass' if gates['diagonal_row_selectivity']['passed'] else 'fail'} |
| near / permuted diagonal | {near_perm_ratio:.3f}; {sum(value >= 1.25 for value in fold_near_perm_ratios)}/4 folds | ≥1.25; ≥3/4 | {'pass' if gates['near_vs_permuted_diagonal']['passed'] else 'fail'} |
| fold stability | Spearman {float(np.median(correlations)):.3f}; sign {sign_fraction:.3f} | ≥0.70; ≥0.75 | {'pass' if gates['fold_stability']['passed'] else 'fail'} |
| technical controls | exact Jacobian, identity oracle, finite outputs, ≤20.5 GiB | all | {'pass' if gates['technical_controls']['passed'] else 'fail'} |

The component bootstrap intervals were
[{100 * near_vs_morph['bootstrap_2_5_percentile']:.2f}%,
{100 * near_vs_morph['bootstrap_97_5_percentile']:.2f}%] for near versus
morphology and [{100 * near_vs_perm['bootstrap_2_5_percentile']:.2f}%,
{100 * near_vs_perm['bootstrap_97_5_percentile']:.2f}%] for near versus the
permutation.  These resample spatial geometry components, not patients.

## What the nonlinearity added

The total near Jacobian had diagonal/off-diagonal ratio
{near_summary['diagonal_offdiagonal_ratio']:.3f}.  Its full linear skip alone
had ratio {summaries['observed_near']['linear']['diagonal_offdiagonal_ratio']:.3f};
the nonlinear residual alone had ratio
{summaries['observed_near']['nonlinear']['diagonal_offdiagonal_ratio']:.3f}.
Thus the stronger total diagonal enrichment mainly came from the nested linear
map, not an exclusively same-name nonlinear residual.

Against the frozen ridge baseline, the nonlinear model's held-out MSE change
was {100 * ridge_serializable['nonlinear_relative_mse_gain_vs_ridge']['relative_mse_gain']:.2f}%
(95% component bootstrap
[{100 * ridge_serializable['nonlinear_relative_mse_gain_vs_ridge']['bootstrap_2_5_percentile']:.2f}%,
{100 * ridge_serializable['nonlinear_relative_mse_gain_vs_ridge']['bootstrap_97_5_percentile']:.2f}%]);
it was better in {ridge_serializable['folds_nonlinear_better']}/4 folds.  A
negative value means worse MSE, so this run does not establish nonlinear
superiority.  Signed same-name diagonals were nevertheless highly rank-aligned
with ridge (Spearman {ridge_serializable['signed_diagonal_spearman']:.3f}).

All arms in every fold selected epoch 12, the frozen upper boundary.  This is
an adverse convergence limitation: the prespecified budget cannot distinguish
an optimum at 12 from continued improvement.  The candidate set was not
expanded after seeing outcomes.

## Largest aggregate same-name sensitivities

| Probe | Total | Linear | Nonlinear residual | Absolute row rank |
|---|---:|---:|---:|---:|
{top_lines}

## Verification and interpretation boundary

The raw and processed fingerprints were reverified, all four canonical run
bundles and native payload manifests passed hashes, canonical prediction rows
reproduced the registered primary metrics, checkpoints were structurally
replayable, and attempt 2 reproduced every attempt-1 NPZ array exactly.  The
four attempt-1 runs are excluded only because a fold-specific seed key polluted
their scientific IDs.

This is an exploratory model-implied derivative under held-out spatial
geometry.  Geometry groups are not biological or patient replicates, center
distance is not cell contact, and verified patient/core/cell-type labels are
unavailable.  Spatial autocorrelation, shared cell state, tissue compartment,
technical field effects, and segmentation spillover remain viable.  The result
does not establish cell-cell communication, mechanism, causality, patient
generalization, or clinical validity.
"""
    (output / "report.md").write_text(report, encoding="utf-8")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=paths.project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=paths.project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    _write_json(
        output / "provenance.json",
        {
            "git_commit": commit,
            "git_status": status,
            "command": sys.argv,
            "working_directory": str(paths.project_root),
            "analysis_source": {
                "path": "scripts/analysis/analyze_same_gene_nonlinear.py",
                "sha256": sha256_file(Path(__file__)),
            },
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "data_verification_command": verification_command,
            "data_verification": data_verification,
            "input_runs": bundle_checks,
            "excluded_runs": excluded,
        },
    )
    verification = {
        "verified": True,
        "input_run_count": 4,
        "folds": [0, 1, 2, 3],
        "attempt": 2,
        "scientific_id": "sci_12587b9d9e7f69f7",
        "input_bundle_checks": bundle_checks,
        "attempt_correction_determinism_checks": replay_checks,
        "gene_axis_identical": True,
        "eligible_gene_count": int(np.sum(eligible)),
        "matrix_shapes": {
            f"{arm}_{part}": list(matrix.shape)
            for arm, parts in matrices.items()
            for part, matrix in parts.items()
        },
        "all_matrices_finite": True,
        "test_geometry_component_coverage": 27,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "all_frozen_gates_passed": all_passed,
    }
    _write_json(output / "verification.json", verification)

    _write_json(
        output / "registry_evaluation.json",
        {"evaluation_id": EVALUATION_ID, "registered": True},
    )
    inventory = _analysis_inventory(output)
    _write_json(
        output / "analysis_manifest.json",
        {
            "evaluation_id": EVALUATION_ID,
            "files": inventory,
            "manifest_payload_sha256": canonical_sha256(inventory),
        },
    )
    _verify_output_files(output)
    output.rename(final_output)
    output = final_output
    _register_analysis(registry, output)
    _verify_existing(output, registry, verify_data=False)
    print(
        canonical_json(
            {
                "verdict": verdict,
                "failed_gates": failed,
                "gates_passed": int(sum(item["passed"] for item in gates.values())),
                "report": str(output / "report.md"),
                "near_gain_vs_morphology": near_vs_morph["relative_mse_gain"],
                "near_gain_vs_permuted": near_vs_perm["relative_mse_gain"],
                "near_diagonal_summary": near_summary,
                "nonlinear_vs_ridge": ridge_serializable[
                    "nonlinear_relative_mse_gain_vs_ridge"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
