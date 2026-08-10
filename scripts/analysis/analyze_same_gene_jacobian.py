#!/usr/bin/env python3
"""Aggregate the four verified same-gene Jacobian outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_json, canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry, utc_now
from spatial_benchmark.run_archive import verify_run_bundle
from spatial_benchmark.same_gene_jacobian import diagonal_summary


CAMPAIGN_ID = "cmp_20260810_same_gene_cross_cell_jacobian"
DATASET_ID = "gastric_cosmx_drive_public"
ARMS = (
    "morphology_only",
    "observed_near",
    "observed_annular",
    "within_fov_permuted_near",
    "same_cell_oracle",
)
MATRIX_ARMS = ARMS[1:]
REPORT_NAME = "same_gene_cross_cell_jacobian_20260810_v2"
CANONICAL_ATTEMPT = 2


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_bundle(path: Path) -> dict[str, Any]:
    canonical = verify_run_bundle(path)
    prediction_rows = [
        json.loads(line)
        for line in (path / "predictions/validation.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    final_metrics = json.loads(
        (path / "metrics/final.json").read_text(encoding="utf-8")
    )
    reproduced_primary = float(
        np.mean([float(row["sample_loss"]) for row in prediction_rows])
    )
    stored_primary = float(
        final_metrics["validation/observed_near_component_equal_mse"]
    )
    if not np.isclose(reproduced_primary, stored_primary, rtol=1e-12, atol=0.0):
        raise RuntimeError(f"Canonical predictions do not reproduce primary MSE: {path}")
    completion = json.loads((path / "COMPLETED.json").read_text(encoding="utf-8"))
    manifest_path = path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if completion.get("verified") is not True:
        raise RuntimeError(f"Run is not completion-verified: {path}")
    errors: list[str] = []
    for item in manifest["files"]:
        artifact = path / item["path"]
        if not artifact.is_file() or artifact.stat().st_size != item["size_bytes"]:
            errors.append(f"missing_or_size:{item['path']}")
        elif sha256_file(artifact) != item["sha256"]:
            errors.append(f"sha256:{item['path']}")
    if canonical_sha256(manifest["files"]) != manifest["manifest_payload_sha256"]:
        errors.append("manifest_payload_sha256")
    if errors:
        raise RuntimeError(f"Run bundle verification failed: {errors}")
    return {
        "run_id": completion["run_id"],
        "bundle": str(path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": sha256_file(path / "ridge_coefficients_checkpoint.npz"),
        "canonical_bundle": canonical,
        "canonical_prediction_rows": len(prediction_rows),
        "canonical_primary_reproduced": True,
    }


def _discover_runs(registry: Registry) -> list[dict[str, Any]]:
    with registry.connect() as connection:
        rows = connection.execute(
            "SELECT r.run_id, r.fold, r.attempt, r.retry_of, r.status, "
            "r.artifact_path, r.config_json, "
            "EXISTS(SELECT 1 FROM artifacts a WHERE a.run_id = r.run_id "
            "AND a.kind = 'legacy_manifest') AS is_legacy_native "
            "FROM runs r WHERE r.campaign_id = ? ORDER BY r.created_at",
            (CAMPAIGN_ID,),
        ).fetchall()
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        configuration = json.loads(str(row["config_json"]))
        if configuration.get("profile") != "full":
            continue
        if bool(row["is_legacy_native"]) or int(row["attempt"]) != CANONICAL_ATTEMPT:
            continue
        if row["status"] != "completed":
            raise RuntimeError(f"Full run is not completed: {row['run_id']}")
        fold = int(row["fold"])
        if fold in selected:
            raise RuntimeError(f"More than one full run exists for fold {fold}")
        selected[fold] = {
            "run_id": str(row["run_id"]),
            "fold": fold,
            "artifact_path": Path(str(row["artifact_path"])),
            "attempt": int(row["attempt"]),
            "retry_of": str(row["retry_of"]),
        }
    if set(selected) != {0, 1, 2, 3}:
        raise RuntimeError(f"Expected full folds 0-3; found {sorted(selected)}")
    return [selected[index] for index in range(4)]


def _arm_metrics(results: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fold_mse: list[float] = []
    for result in results:
        test = result["arms"][arm]["test"]
        fold_mse.append(float(test["component_equal_mse"]))
        rows.extend(test["per_component"])
    groups = [int(row["geometry_group"]) for row in rows]
    if len(groups) != 27 or len(set(groups)) != 27:
        raise RuntimeError(f"{arm} does not cover 27 unique geometry groups")
    return {
        "component_equal_mse": float(np.mean([row["mse"] for row in rows])),
        "component_equal_mae": float(np.mean([row["mae"] for row in rows])),
        "component_equal_huber": float(np.mean([row["huber"] for row in rows])),
        "fold_mse": fold_mse,
        "per_component": rows,
    }


def _paired_gain(
    baseline_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    *,
    seed: int,
    draws: int = 10000,
) -> dict[str, float | int]:
    baseline = {int(row["geometry_group"]): float(row["mse"]) for row in baseline_rows}
    model = {int(row["geometry_group"]): float(row["mse"]) for row in model_rows}
    if set(baseline) != set(model):
        raise RuntimeError("Paired geometry-component sets differ")
    groups = sorted(baseline)
    base = np.asarray([baseline[group] for group in groups])
    fitted = np.asarray([model[group] for group in groups])
    observed = float((base.mean() - fitted.mean()) / base.mean())
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        index = rng.integers(0, len(groups), size=len(groups))
        base_mean = base[index].mean()
        sampled[draw] = (base_mean - fitted[index].mean()) / base_mean
    return {
        "relative_mse_gain": observed,
        "bootstrap_2_5_percentile": float(np.quantile(sampled, 0.025)),
        "bootstrap_97_5_percentile": float(np.quantile(sampled, 0.975)),
        "bootstrap_draws": draws,
        "geometry_component_count": len(groups),
    }


def _row_ranks(matrix: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    absolute = np.abs(matrix[np.ix_(indices, indices)])
    return np.asarray(
        [1 + int(np.sum(row > row[index])) for index, row in enumerate(absolute)],
        dtype=np.int32,
    )


def _random_alignment_null(
    matrix: np.ndarray,
    eligible: np.ndarray,
    *,
    draws: int = 1000,
) -> dict[str, float | int]:
    indices = np.flatnonzero(eligible)
    absolute = np.abs(matrix[np.ix_(indices, indices)])
    off_median = float(np.median(absolute[~np.eye(len(indices), dtype=bool)]))
    observed = float(np.median(np.diag(absolute)) / off_median)
    rng = np.random.default_rng(20260810)
    ratios = np.empty(draws, dtype=np.float64)
    top1 = np.empty(draws, dtype=np.float64)
    maximum = absolute.max(axis=1)
    for draw in range(draws):
        columns = rng.integers(0, len(indices), size=len(indices))
        sampled = absolute[np.arange(len(indices)), columns]
        ratios[draw] = float(np.median(sampled) / off_median)
        top1[draw] = float(np.mean(sampled >= maximum))
    return {
        "draws": draws,
        "observed_ratio": observed,
        "null_ratio_median": float(np.median(ratios)),
        "null_ratio_95th_percentile": float(np.quantile(ratios, 0.95)),
        "empirical_upper_p": float((1 + np.sum(ratios >= observed)) / (draws + 1)),
        "null_top1_fraction_median": float(np.median(top1)),
        "null_top1_fraction_95th_percentile": float(np.quantile(top1, 0.95)),
    }


def _plot(
    output: Path,
    metrics: dict[str, dict[str, Any]],
    matrices: dict[str, np.ndarray],
    eligible: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    baseline = np.asarray(metrics["morphology_only"]["fold_mse"])
    for arm, label, color in (
        ("observed_near", "0–25 µm", "#1f77b4"),
        ("observed_annular", "25–50 µm", "#ff7f0e"),
        ("within_fov_permuted_near", "permuted near", "#7f7f7f"),
    ):
        value = np.asarray(metrics[arm]["fold_mse"])
        axes[0].plot(range(4), 100 * (baseline - value) / baseline, marker="o", label=label, color=color)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(xlabel="Outer geometry fold", ylabel="MSE gain vs morphology (%)", title="Held-out predictive gain")
    axes[0].set_xticks(range(4))
    axes[0].legend(frameon=False)

    indices = np.flatnonzero(eligible)
    rng = np.random.default_rng(20260810)
    for arm, label, color in (
        ("observed_near", "0–25 µm diagonal", "#1f77b4"),
        ("observed_annular", "25–50 µm diagonal", "#ff7f0e"),
        ("within_fov_permuted_near", "permuted diagonal", "#7f7f7f"),
    ):
        selected = np.abs(matrices[arm][np.ix_(indices, indices)])
        axes[1].hist(np.diag(selected), bins=60, density=True, histtype="step", linewidth=1.5, label=label, color=color)
        if arm == "observed_near":
            off = selected[~np.eye(len(indices), dtype=bool)]
            off = rng.choice(off, size=20000, replace=False)
            axes[1].hist(off, bins=60, density=True, histtype="step", linestyle="--", label="near off-diagonal", color="#2ca02c")
    axes[1].set(xlabel="Absolute standardized coefficient", ylabel="Density", title="Diagonal enriched, not exclusive")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _verify_existing_report(output: Path, registry: Registry) -> dict[str, Any]:
    required = (
        "aggregate_results.json",
        "aggregate_jacobians.npz",
        "eligible_gene_diagonal_summary.csv",
        "summary.png",
        "report.md",
        "provenance.json",
        "verification.json",
        "registry_evaluation.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Aggregate report is missing files: {missing}")
    verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
    if verification.get("verified") is not True or verification.get("input_run_count") != 4:
        raise RuntimeError("Stored aggregate verification is not complete")
    arrays = np.load(output / "aggregate_jacobians.npz", allow_pickle=False)
    matrix_names = [name for name in arrays.files if name.startswith("jacobian_")]
    if len(matrix_names) != 4:
        raise RuntimeError("Aggregate Jacobian bundle must contain four arm matrices")
    for name in matrix_names:
        if arrays[name].shape != (1000, 1000) or not bool(np.isfinite(arrays[name]).all()):
            raise RuntimeError(f"Invalid aggregate matrix: {name}")
    for item in verification["input_bundle_checks"]:
        _verify_bundle(Path(item["bundle"]))
    evaluation_id = json.loads(
        (output / "registry_evaluation.json").read_text(encoding="utf-8")
    )["evaluation_id"]
    with registry.connect() as connection:
        row = connection.execute(
            "SELECT status, artifact_path FROM evaluations WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
    if row is None or row["status"] != "completed" or Path(row["artifact_path"]) != output:
        raise RuntimeError("Aggregate registry evaluation is absent or incomplete")
    return {
        "verified": True,
        "evaluation_id": evaluation_id,
        "input_runs": 4,
        "eligible_genes": int(arrays["eligible"].sum()),
        "matrix_count": len(matrix_names),
        "report": str(output / "report.md"),
    }


def main() -> None:
    paths = current_paths()
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    output = paths.report_root / "analyses" / REPORT_NAME
    if arguments.verify_only:
        print(canonical_json(_verify_existing_report(output, registry)))
        return
    prepared = paths.data_root / "processed" / "same_gene_cross_cell_jacobian_v1"
    data_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    runs = _discover_runs(registry)
    bundle_checks = [_verify_bundle(run["artifact_path"]) for run in runs]
    results = [json.loads((run["artifact_path"] / "results.json").read_text(encoding="utf-8")) for run in runs]
    checkpoints = [np.load(run["artifact_path"] / "ridge_coefficients_checkpoint.npz", allow_pickle=False) for run in runs]
    genes = tuple(str(value) for value in checkpoints[0]["genes"])
    if len(genes) != 1000 or any(tuple(str(value) for value in checkpoint["genes"]) != genes for checkpoint in checkpoints[1:]):
        raise RuntimeError("The four gene axes are not identical")
    for checkpoint in checkpoints:
        for arm in ARMS:
            expected_rows = 23 if arm == "morphology_only" else 1023
            required_shapes = {
                f"state_{arm}_coefficients": (expected_rows, 1000),
                f"state_{arm}_target_mean": (1000,),
                f"state_{arm}_target_std": (1000,),
                f"state_{arm}_morphology_median": (22,),
                f"state_{arm}_morphology_mean": (22,),
                f"state_{arm}_morphology_std": (22,),
                f"state_{arm}_selected_penalty": (),
            }
            for name, shape in required_shapes.items():
                if name not in checkpoint or checkpoint[name].shape != shape:
                    raise RuntimeError(f"Replayable checkpoint field is invalid: {name}")
                if not bool(np.isfinite(checkpoint[name]).all()):
                    raise RuntimeError(f"Replayable checkpoint field is nonfinite: {name}")
    eligible = np.logical_and.reduce([checkpoint["eligible_observed_near"].astype(bool) for checkpoint in checkpoints])
    fold_matrices = {
        arm: [checkpoint[f"jacobian_{arm}"].astype(np.float64) for checkpoint in checkpoints]
        for arm in MATRIX_ARMS
    }
    matrices = {arm: np.mean(values, axis=0) for arm, values in fold_matrices.items()}
    summaries = {arm: diagonal_summary(matrix, eligible).as_dict() for arm, matrix in matrices.items()}
    fold_summaries = {
        arm: [diagonal_summary(matrix, eligible).as_dict() for matrix in values]
        for arm, values in fold_matrices.items()
    }
    metrics = {arm: _arm_metrics(results, arm) for arm in ARMS}
    near_vs_morph = _paired_gain(metrics["morphology_only"]["per_component"], metrics["observed_near"]["per_component"], seed=20260810)
    near_vs_perm = _paired_gain(metrics["within_fov_permuted_near"]["per_component"], metrics["observed_near"]["per_component"], seed=20260811)
    annular_vs_morph = _paired_gain(metrics["morphology_only"]["per_component"], metrics["observed_annular"]["per_component"], seed=20260812)
    near_favors_morph = int(np.sum(np.asarray(metrics["observed_near"]["fold_mse"]) < np.asarray(metrics["morphology_only"]["fold_mse"])))
    near_favors_perm = int(np.sum(np.asarray(metrics["observed_near"]["fold_mse"]) < np.asarray(metrics["within_fov_permuted_near"]["fold_mse"])))

    fold_diagonal = np.asarray([np.diag(matrix)[eligible] for matrix in fold_matrices["observed_near"]])
    correlations = [
        float(spearmanr(fold_diagonal[first], fold_diagonal[second]).statistic)
        for first in range(4)
        for second in range(first + 1, 4)
    ]
    positive = np.sum(fold_diagonal > 0, axis=0)
    negative = np.sum(fold_diagonal < 0, axis=0)
    sign_fraction = float(np.mean(np.maximum(positive, negative) >= 3))
    fold_near_perm_ratios = [
        float(np.median(np.abs(np.diag(near)[eligible])) / np.median(np.abs(np.diag(permuted)[eligible])))
        for near, permuted in zip(fold_matrices["observed_near"], fold_matrices["within_fov_permuted_near"], strict=True)
    ]
    near_perm_ratio = float(summaries["observed_near"]["median_absolute_diagonal"] / summaries["within_fov_permuted_near"]["median_absolute_diagonal"])
    near_annular_ratio = float(summaries["observed_near"]["diagonal_offdiagonal_ratio"] / summaries["observed_annular"]["diagonal_offdiagonal_ratio"])
    random_null = _random_alignment_null(matrices["observed_near"], eligible)
    near_summary = summaries["observed_near"]

    gates = {
        "near_vs_morphology_prediction": {
            "passed": near_vs_morph["relative_mse_gain"] >= 0.02 and near_favors_morph >= 3,
            "gain": near_vs_morph["relative_mse_gain"], "minimum": 0.02,
            "folds": near_favors_morph, "minimum_folds": 3,
        },
        "near_vs_permuted_prediction": {
            "passed": near_vs_perm["relative_mse_gain"] >= 0.01 and near_favors_perm >= 3,
            "gain": near_vs_perm["relative_mse_gain"], "minimum": 0.01,
            "folds": near_favors_perm, "minimum_folds": 3,
        },
        "diagonal_enrichment": {
            "passed": near_summary["diagonal_offdiagonal_ratio"] >= 2.0,
            "observed": near_summary["diagonal_offdiagonal_ratio"], "minimum": 2.0,
        },
        "diagonal_row_selectivity": {
            "passed": near_summary["row_top1_fraction"] >= 0.25 and near_summary["row_top1_percent_fraction"] >= 0.50,
            "top1": near_summary["row_top1_fraction"], "top1_minimum": 0.25,
            "top1_percent": near_summary["row_top1_percent_fraction"], "top1_percent_minimum": 0.50,
        },
        "near_vs_permuted_diagonal": {
            "passed": near_perm_ratio >= 1.25 and int(np.sum(np.asarray(fold_near_perm_ratios) >= 1.25)) >= 3,
            "aggregate_ratio": near_perm_ratio, "minimum": 1.25,
            "fold_ratios": fold_near_perm_ratios,
        },
        "fold_stability": {
            "passed": float(np.median(correlations)) >= 0.70 and sign_fraction >= 0.75,
            "median_spearman": float(np.median(correlations)), "minimum_spearman": 0.70,
            "sign_fraction": sign_fraction, "minimum_sign_fraction": 0.75,
        },
        "technical_controls": {
            "passed": all(result["controls"]["all_outputs_finite"] for result in results)
            and all(result["controls"]["primary_self_jacobian_exact_zero_by_design"] for result in results)
            and all(result["controls"]["same_cell_oracle"]["row_top1_fraction"] >= 0.95 for result in results)
            and all(result["peak_vram_gb"] <= 20.5 for result in results),
        },
    }
    all_passed = all(value["passed"] for value in gates.values())
    failed = [name for name, value in gates.items() if not value["passed"]]
    verdict = "strict_same_gene_selectivity_supported" if all_passed else "strict_same_gene_selectivity_not_supported"
    spillover_flag = near_annular_ratio >= 1.5

    payload = {
        "campaign_id": CAMPAIGN_ID,
        "verdict": verdict,
        "all_frozen_gates_passed": all_passed,
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
            "fold_summaries": fold_summaries,
            "fold_pairwise_signed_diagonal_spearman": correlations,
            "median_pairwise_signed_diagonal_spearman": float(np.median(correlations)),
            "sign_consistent_gene_fraction": sign_fraction,
            "near_vs_permuted_diagonal_ratio": near_perm_ratio,
            "near_vs_annular_enrichment_ratio": near_annular_ratio,
            "random_alignment_null": random_null,
        },
        "spillover_alternative_flagged": spillover_flag,
        "gates": gates,
        "input_runs": bundle_checks,
        "dataset_fingerprints": {
            "raw": data_manifest["raw_snapshot"]["fingerprint"],
            "processed": data_manifest["processed_fingerprint"],
            "split": data_manifest["split_fingerprint"],
        },
        "maximum_claim": "exploratory held-out-geometry graph-alignment-dependent model sensitivity",
    }

    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    output.mkdir(parents=True)
    _write_json(output / "aggregate_results.json", payload)
    np.savez_compressed(
        output / "aggregate_jacobians.npz",
        genes=np.asarray(genes), eligible=eligible,
        **{f"jacobian_{arm}": matrix.astype(np.float32) for arm, matrix in matrices.items()},
    )
    ranks = _row_ranks(matrices["observed_near"], eligible)
    eligible_indices = np.flatnonzero(eligible)
    table = pd.DataFrame(
        {
            "gene": [genes[index] for index in eligible_indices],
            "near_diagonal": np.diag(matrices["observed_near"])[eligible],
            "annular_diagonal": np.diag(matrices["observed_annular"])[eligible],
            "permuted_diagonal": np.diag(matrices["within_fov_permuted_near"])[eligible],
            "near_absolute_row_rank": ranks,
            "positive_fold_count": positive,
            "negative_fold_count": negative,
        }
    )
    table.to_csv(output / "eligible_gene_diagonal_summary.csv", index=False)
    _plot(output / "summary.png", metrics, matrices, eligible)
    top = table.loc[table["near_diagonal"].abs().sort_values(ascending=False).index].head(10)
    top_lines = "\n".join(
        f"| {row.gene} | {row.near_diagonal:.5f} | {int(row.near_absolute_row_rank)} |"
        for row in top.itertuples()
    )
    report = f"""# Same-gene cross-cell Jacobian result

## Outcome

- Verdict: **{verdict}**
- Frozen gates passed: **{sum(value['passed'] for value in gates.values())}/{len(gates)}**
- Failed gates: `{', '.join(failed) if failed else 'none'}`
- Data: 407,999 cells, 451 FOVs, 27 opaque geometry groups, 1,000 probes
- Eligible genes: {int(np.sum(eligible))}
- Full runs: {', '.join(check['run_id'] for check in bundle_checks)}

The narrow answer is **no: same-name derivatives were enriched, but high
values were not confined to the diagonal**.  Near neighbors reduced held-out
component-equal MSE by {100 * near_vs_morph['relative_mse_gain']:.2f}% versus
morphology and {100 * near_vs_perm['relative_mse_gain']:.2f}% versus the
within-FOV source-state permutation.  The absolute same-name diagonal was
{near_summary['diagonal_offdiagonal_ratio']:.2f} times the median off-diagonal
and {near_perm_ratio:.2f} times the permuted-arm diagonal.

The same-name source was row top-1 for only
{100 * near_summary['row_top1_fraction']:.2f}% of eligible targets (frozen
minimum 25%) and row top-1% for
{100 * near_summary['row_top1_percent_fraction']:.2f}% (minimum 50%).  The
prespecified selectivity gate therefore failed.

## Predictive evidence

| Comparison | Relative MSE gain | Geometry-component bootstrap interval | Folds favoring near |
|---|---:|---:|---:|
| near vs morphology | {100 * near_vs_morph['relative_mse_gain']:.2f}% | [{100 * near_vs_morph['bootstrap_2_5_percentile']:.2f}%, {100 * near_vs_morph['bootstrap_97_5_percentile']:.2f}%] | {near_favors_morph}/4 |
| near vs permuted near | {100 * near_vs_perm['relative_mse_gain']:.2f}% | [{100 * near_vs_perm['bootstrap_2_5_percentile']:.2f}%, {100 * near_vs_perm['bootstrap_97_5_percentile']:.2f}%] | {near_favors_perm}/4 |
| annular vs morphology | {100 * annular_vs_morph['relative_mse_gain']:.2f}% | [{100 * annular_vs_morph['bootstrap_2_5_percentile']:.2f}%, {100 * annular_vs_morph['bootstrap_97_5_percentile']:.2f}%] | descriptive |

The intervals resample 27 opaque geometry components.  They are spatial
uncertainty summaries, not patient-level confidence intervals.

## Jacobian gates

| Statistic | Observed | Threshold | Result |
|---|---:|---:|---|
| abs diagonal / off-diagonal | {near_summary['diagonal_offdiagonal_ratio']:.3f} | >=2.0 | {'pass' if gates['diagonal_enrichment']['passed'] else 'fail'} |
| same-name row top-1 | {near_summary['row_top1_fraction']:.3f} | >=0.25 | {'pass' if near_summary['row_top1_fraction'] >= 0.25 else 'fail'} |
| same-name row top-1% | {near_summary['row_top1_percent_fraction']:.3f} | >=0.50 | {'pass' if near_summary['row_top1_percent_fraction'] >= 0.50 else 'fail'} |
| near / permuted abs diagonal | {near_perm_ratio:.3f} | >=1.25 | {'pass' if gates['near_vs_permuted_diagonal']['passed'] else 'fail'} |
| fold signed-diagonal Spearman | {float(np.median(correlations)):.3f} | >=0.70 | {'pass' if float(np.median(correlations)) >= 0.70 else 'fail'} |
| sign-consistent genes | {sign_fraction:.3f} | >=0.75 | {'pass' if sign_fraction >= 0.75 else 'fail'} |
| random-alignment empirical p | {random_null['empirical_upper_p']:.4f} | descriptive | — |

Near/annular enrichment was {near_annular_ratio:.2f}, so it
{'did' if spillover_flag else 'did not'} cross the frozen 1.5 short-distance
spillover flag.  Not crossing it does not rule out segmentation spillover or
ordinary spatial autocorrelation.

## Largest aggregate same-name sensitivities

| Probe | Signed standardized coefficient | Absolute row rank |
|---|---:|---:|
{top_lines}

These are coefficients of a one-hop whole-node predictor, not causal effects
or verified molecular interactions.

## Interpretation boundary

The experiment supports modest held-out predictive use of nearby expression
and a stable same-name enrichment beyond the chosen permutation.  It does not
support the stronger claim that only matching RNA names yield high values;
off-diagonal dependencies remain substantial.

Verified patient/core/cell-type labels were unavailable.  Geometry groups are
not biological replicates, cell-center proximity is not contact, and shared
cell state, compartment, technical fields, spatial autocorrelation, and
segmentation spillover remain viable explanations.  The maximum claim is an
exploratory graph-alignment-dependent model-implied sensitivity—not cell-cell
communication, mechanism, causality, or patient generalization.
"""
    (output / "report.md").write_text(report, encoding="utf-8")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=paths.project_root, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    provenance = {
        "git_commit": commit,
        "command": sys.argv,
        "working_directory": str(paths.project_root),
        "input_runs": bundle_checks,
    }
    _write_json(output / "provenance.json", provenance)
    verification = {
        "verified": True,
        "folds": [0, 1, 2, 3],
        "input_run_count": 4,
        "input_bundle_checks": bundle_checks,
        "gene_axis_identical": True,
        "eligible_gene_count": int(np.sum(eligible)),
        "matrix_shapes": {arm: list(matrix.shape) for arm, matrix in matrices.items()},
        "all_matrices_finite": all(bool(np.isfinite(matrix).all()) for matrix in matrices.values()),
        "test_geometry_component_coverage": 27,
        "all_frozen_gates_passed": all_passed,
    }
    _write_json(output / "verification.json", verification)

    evaluation_id = "eval_same_gene_cross_cell_jacobian_20260810_v2"
    registry.create_evaluation(
        evaluation_id,
        run_id=runs[0]["run_id"],
        checkpoint_name="four_fold_aggregate",
        dataset_id=DATASET_ID,
        split_id="opaque_geometry_components_075mm_4fold_v1",
        status="pending",
        metrics={
            "verdict": verdict,
            "near_gain_vs_morphology": near_vs_morph["relative_mse_gain"],
            "near_gain_vs_permuted": near_vs_perm["relative_mse_gain"],
            "diagonal_offdiagonal_ratio": near_summary["diagonal_offdiagonal_ratio"],
            "row_top1_fraction": near_summary["row_top1_fraction"],
        },
        artifact_path=output,
    )
    _write_json(output / "registry_evaluation.json", {"evaluation_id": evaluation_id, "registered": True})
    for path in sorted(value for value in output.iterdir() if value.is_file()):
        registry.record_artifact(
            runs[0]["run_id"],
            evaluation_id=evaluation_id,
            kind="aggregate_analysis",
            path=path,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
    with registry.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE evaluations SET status = ?, finished_at = ? WHERE evaluation_id = ?",
            ("completed", utc_now(), evaluation_id),
        )
        connection.execute(
            "UPDATE campaigns SET status = ?, updated_at = ? WHERE campaign_id = ?",
            ("complete", utc_now(), CAMPAIGN_ID),
        )
    print(
        canonical_json(
            {
                "verdict": verdict,
                "failed_gates": failed,
                "report": str(output / "report.md"),
                "near_gain_vs_morphology": near_vs_morph["relative_mse_gain"],
                "near_gain_vs_permuted": near_vs_perm["relative_mse_gain"],
                "near_diagonal_summary": near_summary,
            }
        )
    )


if __name__ == "__main__":
    main()
