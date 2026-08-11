#!/usr/bin/env python3
"""Run-bundle authority audit for the same-gene robustness publication.

This closes the quantities that are not identifiable from the four aggregate
files alone: seed/fold Jacobian gates, fold stability, row-budget support,
validation-trajectory saturation, component source records, technical
controls, the complete gene axis, and the matched label-null strata.  It does
not import the campaign analyzer or ``spatial_benchmark``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
import torch

from audit_same_gene_robustness_standalone import (
    ARMS,
    ATOL,
    AuditRecorder,
    CAMPAIGN_ID,
    CONTRACT_SHA256,
    ELIGIBLE_MASK_SHA256,
    FOLDS,
    GATES,
    RTOL,
    SEEDS,
    VARIANTS,
    deterministic_derangement,
    json_safe,
    markdown_report as four_file_markdown_report,
    matrix_summary,
    read_component_rows,
    read_gene_rows,
    strict_json,
)


GENE_ORDER_SHA256 = "046eb86c7ea8f1fe6977598a0190132340400fc61802fcde63ab5ac0e9502b03"
ENVIRONMENT_LOCK_SHA256 = "0620ce9ba9914d1626ba1e27a4f2370f6752d52e91eecdec447da9ffa2075fac"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def fold_stability(diagonals: Sequence[np.ndarray]) -> dict[str, Any]:
    if len(diagonals) != 4:
        raise ValueError("four fold diagonals required")
    values = np.asarray(diagonals, dtype=np.float64)
    correlations: list[float] = []
    for first in range(4):
        for second in range(first + 1, 4):
            correlations.append(
                float(spearmanr(values[first], values[second]).statistic)
            )
    positive = np.sum(values > 0, axis=0)
    negative = np.sum(values < 0, axis=0)
    consistent = np.maximum(positive, negative) >= 3
    return {
        "pairwise_signed_diagonal_spearman": correlations,
        "median_pairwise_signed_diagonal_spearman": float(np.median(correlations)),
        "genes_same_sign_in_at_least_3_folds": int(np.sum(consistent)),
        "sign_consistent_gene_fraction": float(np.mean(consistent)),
    }


def compare_summary_subset(
    audit: AuditRecorder,
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    name: str,
    section: str,
) -> None:
    for field, value in expected.items():
        if field not in observed:
            continue
        if value is None or isinstance(value, (bool, int)):
            audit.equal(observed.get(field), value, f"{name}/{field}", section=section)
        else:
            audit.close(observed.get(field), value, f"{name}/{field}", section=section)


def compare_jacobian_gate_block(
    audit: AuditRecorder,
    observed: Mapping[str, Any],
    near_matrix: np.ndarray,
    perm_matrix: np.ndarray,
    near_fold_diagonals: Sequence[np.ndarray],
    perm_fold_diagonals: Sequence[np.ndarray],
    eligible_local: np.ndarray,
    *,
    name: str,
) -> dict[str, Any]:
    local_indices = np.arange(len(eligible_local), dtype=np.int64)
    near_summary = matrix_summary(near_matrix, local_indices)
    perm_summary = matrix_summary(perm_matrix, local_indices)
    diagonal = observed["same_name_diagonal_enrichment"]
    audit.close(diagonal.get("observed"), near_summary["diagonal_offdiagonal_ratio"], f"{name}/diagonal observed", section="run_jacobian_gates")
    audit.equal(diagonal.get("passed"), bool(near_summary["diagonal_offdiagonal_ratio"] >= 2.0), f"{name}/diagonal pass", section="run_jacobian_gates")
    row = observed["strict_row_selectivity"]
    row_pass = bool(
        near_summary["row_top1_fraction"] >= 0.25
        and near_summary["row_top1_percent_fraction"] >= 0.50
    )
    for field in (
        "row_top1_fraction", "row_top1_count",
        "row_top1_percent_fraction", "row_top1_percent_count",
    ):
        if isinstance(near_summary[field], int):
            audit.equal(row.get(field), near_summary[field], f"{name}/row {field}", section="run_jacobian_gates")
        else:
            audit.close(row.get(field), near_summary[field], f"{name}/row {field}", section="run_jacobian_gates")
    audit.equal(row.get("passed"), row_pass, f"{name}/row pass", section="run_jacobian_gates")
    ratio = float(
        near_summary["median_absolute_diagonal"]
        / perm_summary["median_absolute_diagonal"]
    )
    fold_ratios = [
        float(np.median(np.abs(near)) / np.median(np.abs(permuted)))
        for near, permuted in zip(
            near_fold_diagonals, perm_fold_diagonals, strict=True
        )
    ]
    folds_at = int(sum(value >= 1.25 for value in fold_ratios))
    near_perm = observed["near_vs_permutation_diagonal"]
    audit.close(near_perm.get("observed"), ratio, f"{name}/near-perm ratio", section="run_jacobian_gates")
    audit.equal(len(near_perm.get("fold_ratios", [])), 4, f"{name}/near-perm fold count", section="run_jacobian_gates")
    for fold, value in enumerate(fold_ratios):
        audit.close(near_perm.get("fold_ratios", [None] * 4)[fold], value, f"{name}/near-perm fold {fold}", section="run_jacobian_gates")
    audit.equal(near_perm.get("folds_at_or_above"), folds_at, f"{name}/near-perm folds at", section="run_jacobian_gates")
    audit.equal(near_perm.get("passed"), bool(ratio >= 1.25 and folds_at >= 3), f"{name}/near-perm pass", section="run_jacobian_gates")
    stability = fold_stability(near_fold_diagonals)
    encoded_stability = observed["fold_stability"]
    for index, value in enumerate(stability["pairwise_signed_diagonal_spearman"]):
        audit.close(encoded_stability.get("pairwise_signed_diagonal_spearman", [None] * 6)[index], value, f"{name}/stability pair {index}", section="run_jacobian_gates")
    for field in (
        "median_pairwise_signed_diagonal_spearman",
        "sign_consistent_gene_fraction",
    ):
        audit.close(encoded_stability.get(field), stability[field], f"{name}/stability {field}", section="run_jacobian_gates")
    audit.equal(encoded_stability.get("genes_same_sign_in_at_least_3_folds"), stability["genes_same_sign_in_at_least_3_folds"], f"{name}/stability gene count", section="run_jacobian_gates")
    stability_pass = bool(
        stability["median_pairwise_signed_diagonal_spearman"] >= 0.70
        and stability["sign_consistent_gene_fraction"] >= 0.75
    )
    audit.equal(encoded_stability.get("passed"), stability_pass, f"{name}/stability pass", section="run_jacobian_gates")
    return {
        "near_summary": near_summary,
        "permuted_summary": perm_summary,
        "near_permuted_diagonal_ratio": ratio,
        "fold_ratios": fold_ratios,
        "fold_stability": stability,
    }


def expected_component_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str, bool, int], Mapping[str, Any]]:
    return {
        (
            str(row["variant_id"]), int(row["model_seed"]), int(row["fold"]),
            str(row["arm"]), bool(row["anchor"]), int(row["geometry_group"]),
        ): row
        for row in rows
    }


def audit_run_results(
    audit: AuditRecorder,
    payload: Mapping[str, Any],
    component_rows: Sequence[Mapping[str, Any]],
    runs: Mapping[tuple[str, int, int], Path],
) -> tuple[dict[tuple[int, int, str], dict[str, Any]], dict[str, Any]]:
    components = expected_component_map(component_rows)
    observed_component_keys: set[tuple[str, int, int, str, bool, int]] = set()
    trajectories: dict[tuple[int, int, str], dict[str, Any]] = {}
    technical_max = {
        "peak_vram_gb": 0.0,
        "analytical_autograd_error": 0.0,
        "analytical_finite_difference_error": 0.0,
        "checkpoint_replay_metric_error": 0.0,
        "checkpoint_replay_prediction_error": 0.0,
    }
    for (variant, seed, fold), root in sorted(runs.items()):
        result = strict_json(root / "results.json")
        audit.require(
            result.get("campaign_id") == CAMPAIGN_ID
            and result.get("profile") == "full"
            and result.get("status") == "completed"
            and result.get("outer_fold") == fold
            and result.get("statistical_evaluation_role") == "outer_geometry_test"
            and set(result.get("arms", {})) == set(ARMS),
            f"run result identity {variant}/{seed}/{fold}",
            section="run_identity",
        )
        for arm in ARMS:
            block = result["arms"][arm]
            audit.equal(block.get("evaluation_role"), "test", f"run split role {variant}/{seed}/{fold}/{arm}", section="run_identity")
            for anchor, evaluation in (
                (False, block["evaluation"]),
                (True, block["anchor"]["evaluation"]),
            ):
                source_rows = evaluation.get("per_component", [])
                audit.equal(evaluation.get("component_count"), len(source_rows), f"run component count {variant}/{seed}/{fold}/{arm}/{anchor}", section="run_components")
                for source in source_rows:
                    group = int(source["geometry_group"])
                    key = (variant, seed, fold, arm, anchor, group)
                    observed_component_keys.add(key)
                    published = components.get(key)
                    audit.require(published is not None, f"published component source {key}", section="run_components")
                    if published is None:
                        continue
                    audit.equal(published["cell_count"], int(source["cell_count"]), f"component cells {key}", section="run_components")
                    audit.close(published["mse"], float(source["mse"]), f"component MSE {key}", section="run_components")
                    audit.close(published["mae"], float(source["mae"]), f"component MAE {key}", section="run_components")
        controls = result.get("controls", {})
        analytical = controls.get("analytical_nonlinear_jacobian", {})
        split = controls.get("split_overlap_control", {})
        metric_error = float(controls.get("checkpoint_gpu_replay_max_abs_metric_error", math.inf))
        prediction_error = float(controls.get("checkpoint_gpu_replay_max_abs_prediction_error", math.inf))
        peak = float(controls.get("peak_vram_gb", math.inf))
        autograd_error = float(analytical.get("maximum_autograd_error", math.inf))
        finite_difference_error = float(analytical.get("maximum_finite_difference_error", math.inf))
        technical_max["peak_vram_gb"] = max(technical_max["peak_vram_gb"], peak)
        technical_max["analytical_autograd_error"] = max(technical_max["analytical_autograd_error"], autograd_error)
        technical_max["analytical_finite_difference_error"] = max(technical_max["analytical_finite_difference_error"], finite_difference_error)
        technical_max["checkpoint_replay_metric_error"] = max(technical_max["checkpoint_replay_metric_error"], metric_error)
        technical_max["checkpoint_replay_prediction_error"] = max(technical_max["checkpoint_replay_prediction_error"], prediction_error)
        overlap_counts = split.get("cell_overlap_counts", {})
        overlap_ids = split.get("component_overlap_ids", {})
        technical_pass = bool(
            controls.get("all_outputs_finite") is True
            and controls.get("receiver_expression_input") is False
            and controls.get("receiver_rna_or_derived_covariate_model_input") is False
            and analytical.get("passed") is True
            and autograd_error <= 1e-10
            and finite_difference_error <= 1e-8
            and controls.get("identity_oracle_actually_executed") is True
            and controls.get("identity_oracle_row_top1_fraction") == 1.0
            and controls.get("graph_specific_invariants") is True
            and controls.get("checkpoint_replay_device_type") == "cuda"
            and metric_error <= 1e-7
            and prediction_error <= 1e-7
            and controls.get("canonical_production_split_label") == "test"
            and controls.get("source_config_data_hashes_verified") is True
            and controls.get("outer_test_untouched") is True
            and controls.get("environment_lock_verified") is True
            and controls.get("environment_visibility_mode") == "job"
            and controls.get("environment_lock_sha256") == ENVIRONMENT_LOCK_SHA256
            and controls.get("peak_vram_gate_passed") is True
            and peak <= 20.5
            and all(int(value) == 0 for value in overlap_counts.values())
            and all(value == [] for value in overlap_ids.values())
        )
        audit.require(technical_pass, f"technical controls {variant}/{seed}/{fold}", section="run_technical", observed=controls)
        for arm in ("morphology_only", "observed_near"):
            history = result["arms"][arm]["validation_history"]
            epochs = [int(row["epoch"]) for row in history]
            audit.equal(epochs, [12, 24, 48, 96, 192], f"trajectory epochs {variant}/{seed}/{fold}/{arm}", section="run_trajectories")
            selected = min(history, key=lambda row: (float(row["component_equal_mse"]), int(row["epoch"])))
            selected_epoch = int(result["arms"][arm]["selected_epoch"])
            audit.equal(selected_epoch, int(selected["epoch"]), f"trajectory selection {variant}/{seed}/{fold}/{arm}", section="run_trajectories")
            if variant == "V0":
                by_epoch = {int(row["epoch"]): float(row["component_equal_mse"]) for row in history}
                late_gain = (by_epoch[96] - by_epoch[192]) / by_epoch[96]
                trajectories[(seed, fold, arm)] = {
                    "run_id": result["run_id"],
                    "selected_epoch": selected_epoch,
                    "literal_relative_gain_96_to_192": late_gain,
                    "saturated": bool(selected_epoch < 192 or late_gain <= 0.001),
                }
    audit.equal(observed_component_keys, set(components), "all component rows sourced from run results", section="run_components")
    encoded_trajectories = {
        (int(row["model_seed"]), int(row["fold"]), str(row["arm"])): row
        for row in payload["budget_attribution"]["trajectories"]
    }
    audit.equal(set(encoded_trajectories), set(trajectories), "V0 trajectory source inventory", section="run_trajectories")
    for key, source in trajectories.items():
        encoded = encoded_trajectories[key]
        audit.equal(encoded.get("run_id"), source["run_id"], f"trajectory run ID {key}", section="run_trajectories")
        audit.equal(encoded.get("selected_epoch"), source["selected_epoch"], f"trajectory selected epoch {key}", section="run_trajectories")
        audit.close(encoded.get("literal_relative_gain_96_to_192"), source["literal_relative_gain_96_to_192"], f"trajectory late gain {key}", section="run_trajectories")
        audit.equal(encoded.get("saturated"), source["saturated"], f"trajectory saturation {key}", section="run_trajectories")
    return trajectories, technical_max


def audit_run_matrices(
    audit: AuditRecorder,
    payload: Mapping[str, Any],
    analysis_dir: Path,
    runs: Mapping[tuple[str, int, int], Path],
    eligible_indices: np.ndarray,
    published_gene_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, tuple[str, ...]]:
    published_archive = np.load(analysis_dir / "aggregate_jacobians.npz", allow_pickle=False)
    max_consensus_error = 0.0
    v0_consensus: np.ndarray | None = None
    gene_axis: tuple[str, ...] | None = None
    row_budget_counts: dict[str, int] = {}
    variant_results: dict[str, Any] = {}
    for variant in VARIANTS:
        seed_near: dict[int, np.ndarray] = {}
        seed_perm: dict[int, np.ndarray] = {}
        near_diagonals: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        perm_diagonals: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        for seed in SEEDS:
            fold_near: list[np.ndarray] = []
            fold_perm: list[np.ndarray] = []
            improved_rows = 0
            for fold in FOLDS:
                root = runs[(variant, seed, fold)]
                with np.load(root / "nonlinear_jacobians.npz", allow_pickle=False) as archive:
                    genes = tuple(str(value) for value in archive["genes"])
                    if gene_axis is None:
                        gene_axis = genes
                    else:
                        audit.equal(genes, gene_axis, f"run gene axis {variant}/{seed}/{fold}", section="run_gene_axis")
                    for arm in ARMS:
                        mask = np.asarray(archive[f"eligible_{arm}"], dtype=bool)
                        audit.equal(sha256(mask.astype(np.uint8).tobytes(order="C")).hexdigest(), ELIGIBLE_MASK_SHA256, f"run eligible mask {variant}/{seed}/{fold}/{arm}", section="run_gene_axis")
                    near_full = np.asarray(archive["observed_near_total"], dtype=np.float64)
                    perm_full = np.asarray(archive["within_fov_permuted_near_total"], dtype=np.float64)
                    near_local = near_full[np.ix_(eligible_indices, eligible_indices)]
                    perm_local = perm_full[np.ix_(eligible_indices, eligible_indices)]
                    fold_near.append(near_local)
                    fold_perm.append(perm_local)
                    near_diagonals[seed][fold] = np.diag(near_local).copy()
                    perm_diagonals[seed][fold] = np.diag(perm_local).copy()
                    if variant == "V0":
                        anchor_full = np.asarray(archive["observed_near_anchor12_total"], dtype=np.float64)
                        anchor_local = anchor_full[np.ix_(eligible_indices, eligible_indices)]
                        selected_summary = matrix_summary(near_local, np.arange(932))
                        anchor_summary = matrix_summary(anchor_local, np.arange(932))
                        improved_rows += int(
                            selected_summary["row_top1_fraction"] > anchor_summary["row_top1_fraction"]
                            and selected_summary["row_top1_percent_fraction"] > anchor_summary["row_top1_percent_fraction"]
                        )
            seed_near[seed] = np.mean(fold_near, axis=0)
            seed_perm[seed] = np.mean(fold_perm, axis=0)
            if variant == "V0":
                row_budget_counts[str(seed)] = improved_rows
            encoded_seed = payload["variants"][variant]["seed_specific_gates"][str(seed)]
            compare_jacobian_gate_block(
                audit,
                encoded_seed,
                seed_near[seed],
                seed_perm[seed],
                [near_diagonals[seed][fold] for fold in FOLDS],
                [perm_diagonals[seed][fold] for fold in FOLDS],
                np.ones(932, dtype=bool),
                name=f"{variant}/seed{seed}",
            )
            audit.equal(encoded_seed["technical_validity"].get("passed"), True, f"{variant}/seed{seed}/technical gate", section="run_jacobian_gates")
        consensus_near = np.mean([seed_near[seed] for seed in SEEDS], axis=0)
        consensus_perm = np.mean([seed_perm[seed] for seed in SEEDS], axis=0)
        published_near = np.asarray(published_archive[f"{variant}_observed_near_selected_total"])[np.ix_(eligible_indices, eligible_indices)]
        published_perm = np.asarray(published_archive[f"{variant}_within_fov_permuted_near_selected_total"])[np.ix_(eligible_indices, eligible_indices)]
        near_error = float(np.max(np.abs(consensus_near - published_near)))
        perm_error = float(np.max(np.abs(consensus_perm - published_perm)))
        max_consensus_error = max(max_consensus_error, near_error, perm_error)
        audit.require(near_error <= 1e-15, f"aggregate near matrix from runs {variant}", section="run_matrix_aggregation", observed=near_error, expected="<=1e-15")
        audit.require(perm_error <= 1e-15, f"aggregate perm matrix from runs {variant}", section="run_matrix_aggregation", observed=perm_error, expected="<=1e-15")
        fold_consensus_near = [
            np.mean([near_diagonals[seed][fold] for seed in SEEDS], axis=0)
            for fold in FOLDS
        ]
        fold_consensus_perm = [
            np.mean([perm_diagonals[seed][fold] for seed in SEEDS], axis=0)
            for fold in FOLDS
        ]
        reconstructed = compare_jacobian_gate_block(
            audit,
            payload["variants"][variant]["consensus_gates"],
            consensus_near,
            consensus_perm,
            fold_consensus_near,
            fold_consensus_perm,
            np.ones(932, dtype=bool),
            name=f"{variant}/consensus",
        )
        variant_results[variant] = {
            "aggregate_near_matrix_max_abs_error": near_error,
            "aggregate_permuted_matrix_max_abs_error": perm_error,
            "fold_ratios": reconstructed["fold_ratios"],
            "fold_stability": reconstructed["fold_stability"],
        }
        if variant == "V0":
            v0_consensus = consensus_near.copy()
    published_archive.close()
    assert gene_axis is not None and v0_consensus is not None
    audit.equal(len(gene_axis), 1000, "complete gene axis length", section="run_gene_axis")
    audit.equal(len(set(gene_axis)), 1000, "complete gene axis unique", section="run_gene_axis")
    gene_hash = sha256(canonical_json(list(gene_axis)).encode("utf-8")).hexdigest()
    audit.equal(gene_hash, GENE_ORDER_SHA256, "complete gene axis SHA", section="run_gene_axis")
    for row in published_gene_rows:
        audit.equal(row["gene"], gene_axis[int(row["gene_index"])], f"eligible gene label {row['gene_index']}", section="run_gene_axis")
    audit.equal(payload["budget_attribution"].get("seed_folds_improving_both_row_metrics"), row_budget_counts, "budget row fold support from run matrices", section="run_budget_rows")
    return {
        "maximum_published_consensus_matrix_abs_error": max_consensus_error,
        "variants": variant_results,
        "budget_row_fold_support": row_budget_counts,
        "complete_gene_axis_sha256": gene_hash,
    }, v0_consensus, gene_axis


def matched_strata(prevalence: np.ndarray, target_std: np.ndarray) -> list[np.ndarray]:
    first = np.asarray(prevalence, dtype=np.float64)
    second = np.asarray(target_std, dtype=np.float64)
    order_first = np.argsort(np.argsort(first, kind="stable"), kind="stable")
    order_second = np.argsort(np.argsort(second, kind="stable"), kind="stable")
    bins_first = np.minimum(9, (10 * order_first) // len(first))
    bins_second = np.minimum(9, (10 * order_second) // len(second))
    codes = bins_first * 10 + bins_second
    groups = [np.flatnonzero(codes == code) for code in sorted(np.unique(codes))]
    large = [value for value in groups if len(value) >= 2]
    singletons = [int(value[0]) for value in groups if len(value) == 1]
    for index in singletons:
        distances = [
            abs(float(first[index] - np.mean(first[group])))
            + abs(float(second[index] - np.mean(second[group])))
            for group in large
        ]
        selected = int(np.argmin(distances))
        large[selected] = np.sort(np.append(large[selected], index))
    return [np.asarray(value, dtype=np.int64) for value in large]


def v0_prevalence(prepared_root: Path) -> np.ndarray:
    total_nonzero = np.zeros(1000, dtype=np.int64)
    fold_nonzero = np.zeros((4, 1000), dtype=np.int64)
    total_cells = 0
    fold_cells = np.zeros(4, dtype=np.int64)
    for slide in ("SO_1", "SO_2"):
        expression = np.load(prepared_root / slide / "expression_log1p.npy", mmap_mode="r", allow_pickle=False)
        folds = np.load(prepared_root / slide / "fold.npy", mmap_mode="r", allow_pickle=False)
        eligible = np.load(prepared_root / slide / "eligible_primary.npy", mmap_mode="r", allow_pickle=False)
        for start in range(0, len(expression), 4096):
            stop = min(start + 4096, len(expression))
            chunk_eligible = np.asarray(eligible[start:stop], dtype=bool)
            if not np.any(chunk_eligible):
                continue
            chunk_expression = np.asarray(expression[start:stop][chunk_eligible])
            chunk_folds = np.asarray(folds[start:stop][chunk_eligible], dtype=np.int8)
            nonzero = chunk_expression > 0
            total_nonzero += np.sum(nonzero, axis=0, dtype=np.int64)
            total_cells += len(chunk_expression)
            for fold in FOLDS:
                selected = chunk_folds == fold
                fold_cells[fold] += int(np.sum(selected))
                if np.any(selected):
                    fold_nonzero[fold] += np.sum(nonzero[selected], axis=0, dtype=np.int64)
    values = [
        (total_nonzero - fold_nonzero[fold]) / float(total_cells - fold_cells[fold])
        for fold in FOLDS
    ]
    return np.mean(values, axis=0)


def v0_target_std(
    runs: Mapping[tuple[str, int, int], Path]
) -> np.ndarray:
    values: list[np.ndarray] = []
    for seed in SEEDS:
        for fold in FOLDS:
            checkpoint = torch.load(
                runs[("V0", seed, fold)] / "checkpoints/last.ckpt",
                map_location="cpu",
                weights_only=True,
            )
            tensor = checkpoint["arms"]["observed_near"]["target_std"]
            values.append(tensor.detach().cpu().double().numpy())
            del checkpoint
    return np.mean(values, axis=0)


def audit_matched_null(
    audit: AuditRecorder,
    payload: Mapping[str, Any],
    analysis_dir: Path,
    v0_matrix: np.ndarray,
    eligible_indices: np.ndarray,
    prevalence: np.ndarray,
    target_std: np.ndarray,
) -> dict[str, Any]:
    selected = v0_matrix
    absolute = np.abs(selected)
    count = len(selected)
    ranks = np.empty((count, count), dtype=np.int32)
    for row in range(count):
        order = np.argsort(-absolute[row], kind="stable")
        sorted_values = absolute[row, order]
        local_ranks = np.empty(count, dtype=np.int32)
        position = 0
        while position < count:
            end = position + 1
            while end < count and sorted_values[end] == sorted_values[position]:
                end += 1
            local_ranks[order[position:end]] = position + 1
            position = end
        ranks[row] = local_ranks
    strata = matched_strata(prevalence[eligible_indices], target_std[eligible_indices])
    audit.equal(len(strata), payload["gene_label_null"]["prevalence_sd_decile_matched"]["stratum_count"], "matched null stratum count", section="matched_null")
    audit.equal(min(len(value) for value in strata), payload["gene_label_null"]["prevalence_sd_decile_matched"]["minimum_stratum_size"], "matched null minimum stratum", section="matched_null")
    values = {
        "median_absolute_diagonal": np.empty(10_000, dtype=np.float64),
        "row_top1_fraction": np.empty(10_000, dtype=np.float64),
        "row_top1_percent_fraction": np.empty(10_000, dtype=np.float64),
    }
    row_indices = np.arange(count)
    for draw in range(10_000):
        mapping = np.empty(count, dtype=np.int64)
        for stratum_index, stratum in enumerate(strata):
            local = deterministic_derangement(
                len(stratum),
                seed=20261019 + 10_000_019 + draw * 100_003 + stratum_index,
            )
            mapping[stratum] = stratum[local]
        selected_absolute = absolute[row_indices, mapping]
        selected_rank = ranks[row_indices, mapping]
        values["median_absolute_diagonal"][draw] = np.median(selected_absolute)
        values["row_top1_fraction"][draw] = np.mean(selected_rank <= 1)
        values["row_top1_percent_fraction"][draw] = np.mean(selected_rank <= 10)
    errors: dict[str, float] = {}
    with np.load(analysis_dir / "aggregate_jacobians.npz", allow_pickle=False) as archive:
        for statistic, recomputed in values.items():
            published = np.asarray(archive[f"gene_label_null_prevalence_sd_decile_matched_{statistic}"])
            error = float(np.max(np.abs(published - recomputed)))
            errors[statistic] = error
            audit.require(error <= 1e-15, f"matched null draw reproduction/{statistic}", section="matched_null", observed=error, expected="<=1e-15")
    return {
        "stratum_count": len(strata),
        "minimum_stratum_size": min(len(value) for value in strata),
        "maximum_distribution_abs_error": max(errors.values()),
        "distribution_abs_errors": errors,
        "prevalence_minimum": float(np.min(prevalence[eligible_indices])),
        "prevalence_maximum": float(np.max(prevalence[eligible_indices])),
        "target_std_minimum": float(np.min(target_std[eligible_indices])),
        "target_std_maximum": float(np.max(target_std[eligible_indices])),
    }


def markdown(result: Mapping[str, Any]) -> str:
    audit = result["audit"]
    matrix = result["run_matrix_authority"]
    matched = result["matched_null_authority"]
    technical = result["technical_maxima"]
    lines = [
        "# Run-authority audit: same-gene robustness multiverse v1",
        "",
        f"Overall: **{audit['status'].upper()}** — {audit['passed']}/{audit['checks']} checks passed; {audit['failed']} failed.",
        "",
        "This audit independently read the 140 selected immutable run bundles and V0 prepared arrays. It imported neither the campaign analyzer nor `spatial_benchmark`.",
        "",
        "## Authority-complete checks",
        "",
        f"- Run-derived consensus near/permuted matrices reproduced the published eligible 932×932 matrices with maximum absolute error `{matrix['maximum_published_consensus_matrix_abs_error']:.3g}`.",
        f"- The matched prevalence/target-SD label-null reproduced all 30,000 stored values with maximum absolute error `{matched['maximum_distribution_abs_error']:.3g}`; strata={matched['stratum_count']}, minimum size={matched['minimum_stratum_size']}.",
        f"- Complete 1,000-gene order SHA-256: `{matrix['complete_gene_axis_sha256']}`.",
        "- All selected and anchor component MSE/MAE/cell-count records were reconciled to run `results.json`.",
        "- All seed/fold Jacobian diagonal, row-rank, observed/permuted ratio, fold-Spearman, and sign-consistency gates were reconstructed from run NPZ matrices.",
        "- All 40 V0 96→192 validation gains and selected epochs were reconstructed from validation histories.",
        "- The five seed-level row-budget fold-support counts were reconstructed from selected and anchor fold matrices.",
        "",
        "## Technical maxima across 140 runs",
        "",
        f"- peak VRAM: {technical['peak_vram_gb']:.6f} GiB (gate 20.5 GiB)",
        f"- analytical/autograd error: {technical['analytical_autograd_error']:.3g}",
        f"- analytical/finite-difference error: {technical['analytical_finite_difference_error']:.3g}",
        f"- checkpoint replay metric/prediction errors: {technical['checkpoint_replay_metric_error']:.3g} / {technical['checkpoint_replay_prediction_error']:.3g}",
        "",
        "## Remaining boundary",
        "",
        "- This run-authority audit still does not turn geometry components into biological or patient replicates.",
        "- Per-slide component-weighted Jacobian summaries require checkpoint-weight reconstruction; the separately published slide-equal aggregate was already checked by the four-file audit, while per-slide summaries remain a provenance-level rather than independent numerical check here.",
        "- Artifact manifests, registry identities, and source/environment hashes are covered by the publication verifier/provenance audit; this file focuses on numerical scientific authority.",
        "",
    ]
    if audit["failures"]:
        lines.extend(["## Failures", ""])
        lines.extend(
            f"- `{item['section']}/{item['check']}`: observed={item['observed']!r}, expected={item['expected']!r}"
            for item in audit["failures"]
        )
        lines.append("")
    return "\n".join(lines)


def run(analysis_dir: Path, prepared_v0_root: Path) -> dict[str, Any]:
    audit = AuditRecorder()
    payload = strict_json(analysis_dir / "aggregate_results.json")
    component_rows = read_component_rows(analysis_dir / "component_metrics.csv", audit)
    gene_rows = read_gene_rows(analysis_dir / "eligible_gene_summary.csv", audit)
    eligible_indices = np.asarray([row["gene_index"] for row in gene_rows], dtype=np.int64)
    runs = {
        (str(row["variant_id"]), int(row["model_seed"]), int(row["fold"])):
        Path(str(row["artifact_path"])).resolve(strict=True)
        for row in payload["attempt_history"]
        if row.get("selected") is True
    }
    audit.equal(len(runs), 140, "selected run authority count", section="run_identity")
    _, technical = audit_run_results(audit, payload, component_rows, runs)
    matrix_result, v0_matrix, _ = audit_run_matrices(
        audit, payload, analysis_dir, runs, eligible_indices, gene_rows
    )
    prevalence = v0_prevalence(prepared_v0_root)
    target_std = v0_target_std(runs)
    matched = audit_matched_null(
        audit,
        payload,
        analysis_dir,
        v0_matrix,
        eligible_indices,
        prevalence,
        target_std,
    )
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "audit_kind": "standalone_run_bundle_numerical_authority_audit",
        "independence": {
            "campaign_analyzer_imported": False,
            "spatial_benchmark_imported": False,
            "selected_run_bundles": 140,
            "prepared_v0_root": str(prepared_v0_root),
            "numeric_tolerance": {"rtol": RTOL, "atol": ATOL},
        },
        "audit": audit.summary(),
        "run_matrix_authority": matrix_result,
        "matched_null_authority": matched,
        "technical_maxima": technical,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--prepared-v0-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    result = run(
        args.analysis_dir.resolve(strict=True),
        args.prepared_v0_root.resolve(strict=True),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({
        "status": result["audit"]["status"],
        "checks": result["audit"]["checks"],
        "failed": result["audit"]["failed"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, sort_keys=True))
    return 0 if result["audit"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
