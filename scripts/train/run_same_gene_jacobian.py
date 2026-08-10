#!/usr/bin/env python3
"""Fit one outer fold of the same-gene additive cross-cell experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import torch

from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import (
    canonical_json,
    canonical_sha256,
    create_run_id,
    repro_id,
    scientific_id,
    semantic_run_alias,
)
from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry, utc_now
from spatial_benchmark.run_archive import (
    RunArchive,
    verify_run_bundle,
    verify_unmarked_run_bundle,
)
from spatial_benchmark.same_gene_jacobian import (
    SameGeneJacobianError,
    component_equal_weights,
    diagonal_summary,
    planted_recovery_control,
)


CAMPAIGN_ID = "cmp_20260810_same_gene_cross_cell_jacobian"
DATASET_ID = "gastric_cosmx_drive_public"
DATASET_VERSION = "drive_snapshot_20260810"
SPLIT_ID = "opaque_geometry_components_075mm_4fold_v1"
PREPROCESSING_VERSION = "same_gene_neighbor_log1p_v1"
SLIDES = ("SO_1", "SO_2")
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
ARMS = (
    "morphology_only",
    "observed_near",
    "observed_annular",
    "within_fov_permuted_near",
    "same_cell_oracle",
)
FEATURE_FILE = {
    "observed_near": "neighbor_near_mean.npy",
    "observed_annular": "neighbor_annular_mean.npy",
    "within_fov_permuted_near": "neighbor_permuted_near_mean.npy",
}
PROVENANCE_SOURCE_PATHS = (
    "src/spatial_benchmark/same_gene_jacobian.py",
    "scripts/data/prepare_same_gene_jacobian.py",
    "scripts/train/run_same_gene_jacobian.py",
    "scripts/analysis/analyze_same_gene_jacobian.py",
    "tests/unit/spatial_benchmark/test_same_gene_jacobian.py",
    "experiments/campaigns/cmp_20260810_same_gene_cross_cell_jacobian/README.md",
    "experiments/campaigns/cmp_20260810_same_gene_cross_cell_jacobian/campaign.yaml",
    "experiments/campaigns/cmp_20260810_same_gene_cross_cell_jacobian/frozen_task_contract.yaml",
)


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def _provenance(root: Path, configuration: dict[str, Any]) -> dict[str, Any]:
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(root, "status", "--short", "--untracked-files=all")
    diff = _git_output(root, "diff", "--binary", "HEAD")
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    ).stdout
    untracked_files = []
    for raw_relative in untracked_raw.split(b"\0"):
        if not raw_relative:
            continue
        relative = raw_relative.decode("utf-8", errors="strict")
        path = root / relative
        if path.is_symlink():
            target = os.readlink(path)
            untracked_files.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                }
            )
        elif path.is_file():
            untracked_files.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    untracked_files.sort(key=lambda item: str(item["path"]))
    source_snapshot_manifest = [
        {
            "path": relative,
            "size_bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in PROVENANCE_SOURCE_PATHS
        if (root / relative).is_file()
    ]
    dirty = (
        hashlib.sha256(
            canonical_json(
                {
                    "status": status,
                    "tracked_diff": diff,
                    "untracked_files": untracked_files,
                    "source_snapshot_manifest": source_snapshot_manifest,
                }
            ).encode("utf-8")
        ).hexdigest()
        if status or diff or untracked_files
        else None
    )
    hardware = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_count_visible": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
        "numpy": np.__version__,
    }
    environment_fingerprint = canonical_sha256(hardware)
    return {
        "git_commit": commit,
        "dirty_fingerprint": dirty,
        "git_status": status,
        "tracked_diff": diff,
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "untracked_files": untracked_files,
        "source_snapshot_manifest": source_snapshot_manifest,
        "hardware": hardware,
        "environment_fingerprint": environment_fingerprint,
        "configuration_sha256": canonical_sha256(configuration),
        "command": sys.argv,
        "working_directory": str(root),
    }


def _load_cpu_vector(prepared: Path, filename: str) -> np.ndarray:
    return np.concatenate(
        [np.load(prepared / slide / filename, allow_pickle=False) for slide in SLIDES]
    )


def _load_gpu_matrix(prepared: Path, filename: str, device: torch.device) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for slide in SLIDES:
        array = np.load(prepared / slide / filename, allow_pickle=False)
        if array.ndim != 2:
            raise SameGeneJacobianError(f"{filename} is not a matrix")
        parts.append(torch.from_numpy(array).to(device=device, dtype=torch.float32))
        del array
    result = torch.cat(parts, dim=0)
    del parts
    if not bool(torch.isfinite(result).all().item()):
        raise SameGeneJacobianError(f"{filename} contains nonfinite values")
    return result


def _capped_mask(
    base_mask: np.ndarray,
    groups: np.ndarray,
    maximum: int | None,
    *,
    seed: int,
) -> np.ndarray:
    if maximum is None or int(np.sum(base_mask)) <= maximum:
        return base_mask.copy()
    selected = np.zeros_like(base_mask, dtype=bool)
    unique_groups = np.unique(groups[base_mask])
    per_group = max(1, maximum // len(unique_groups))
    for group in unique_groups:
        candidates = np.flatnonzero(base_mask & (groups == group))
        rng = np.random.default_rng(seed + int(group) * 1009)
        take = min(len(candidates), per_group)
        chosen = rng.choice(candidates, size=take, replace=False)
        selected[chosen] = True
    remaining = maximum - int(np.sum(selected))
    if remaining > 0:
        candidates = np.flatnonzero(base_mask & ~selected)
        rng = np.random.default_rng(seed + 99173)
        chosen = rng.choice(candidates, size=min(remaining, len(candidates)), replace=False)
        selected[chosen] = True
    return selected


def _split_masks(
    folds: np.ndarray,
    groups: np.ndarray,
    eligible: np.ndarray,
    outer_fold: int,
    *,
    profile: str,
) -> dict[str, np.ndarray]:
    test = (folds == outer_fold) & eligible
    validation_fold = (outer_fold + 1) % 4
    validation = (folds == validation_fold) & eligible
    tuning_train = (folds != outer_fold) & (folds != validation_fold) & eligible
    final_train = (folds != outer_fold) & eligible
    caps = (
        {"tuning_train": 24000, "validation": 12000, "final_train": 36000, "test": 12000}
        if profile == "pilot"
        else {"tuning_train": None, "validation": None, "final_train": None, "test": None}
    )
    masks = {}
    for offset, (name, mask) in enumerate(
        (
            ("tuning_train", tuning_train),
            ("validation", validation),
            ("final_train", final_train),
            ("test", test),
        )
    ):
        masks[name] = _capped_mask(
            mask,
            groups,
            caps[name],
            seed=20260810 + outer_fold * 100 + offset,
        )
        if not np.any(masks[name]):
            raise SameGeneJacobianError(f"{name} mask is empty")
    if np.any(masks["test"] & masks["final_train"]):
        raise SameGeneJacobianError("Outer test cells leaked into final training")
    if set(np.unique(groups[masks["test"]])) & set(
        np.unique(groups[masks["final_train"]])
    ):
        raise SameGeneJacobianError("A geometry component crosses train and test")
    return masks


def _target_statistics(
    target: torch.Tensor, mask: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    indices = torch.from_numpy(np.flatnonzero(mask)).to(device)
    selected = target.index_select(0, indices)
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, correction=0)
    prevalence = (selected > 0).float().mean(dim=0).cpu().numpy()
    if not bool(torch.isfinite(mean).all().item()) or not bool(torch.isfinite(std).all().item()):
        raise SameGeneJacobianError("Target normalization statistics are nonfinite")
    safe_std = std.clamp_min(1e-6)
    return mean, safe_std, prevalence


def _morphology_statistics(
    morphology: torch.Tensor,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.from_numpy(np.flatnonzero(mask)).to(device)
    selected = morphology.index_select(0, indices)
    median = torch.nanmedian(selected, dim=0).values
    if not bool(torch.isfinite(median).all().item()):
        raise SameGeneJacobianError("A morphology column has no finite training values")
    imputed = torch.where(torch.isfinite(selected), selected, median)
    mean = imputed.mean(dim=0)
    std = imputed.std(dim=0, correction=0).clamp_min(1e-6)
    return median, mean, std


def _design_batch(
    indices: torch.Tensor,
    morphology: torch.Tensor,
    morphology_statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    feature: torch.Tensor | None,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    median, morphology_mean, morphology_std = morphology_statistics
    morph = morphology.index_select(0, indices)
    morph = torch.where(torch.isfinite(morph), morph, median)
    morph = (morph - morphology_mean) / morphology_std
    pieces = [torch.ones((len(indices), 1), device=indices.device), morph]
    if feature is not None:
        neighbor = (feature.index_select(0, indices) - target_mean) / target_std
        pieces.append(neighbor)
    design = torch.cat(pieces, dim=1)
    if not bool(torch.isfinite(design).all().item()):
        raise SameGeneJacobianError("A design batch contains nonfinite values")
    return design


def _sufficient_statistics(
    target: torch.Tensor,
    morphology: torch.Tensor,
    feature: torch.Tensor | None,
    mask: np.ndarray,
    groups: np.ndarray,
    target_statistics: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    morphology_statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    target_mean, target_std, _ = target_statistics
    positions = np.flatnonzero(mask)
    weights = component_equal_weights(groups, mask)[positions].astype(np.float32)
    feature_count = 1 + morphology.shape[1] + (0 if feature is None else target.shape[1])
    xtx = torch.zeros((feature_count, feature_count), dtype=torch.float64, device=device)
    xty = torch.zeros((feature_count, target.shape[1]), dtype=torch.float64, device=device)
    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        index = torch.from_numpy(positions[start:stop]).to(device)
        batch_weights = torch.from_numpy(weights[start:stop]).to(device)
        design = _design_batch(
            index,
            morphology,
            morphology_statistics,
            feature,
            target_mean,
            target_std,
        )
        response = (target.index_select(0, index) - target_mean) / target_std
        root_weight = torch.sqrt(batch_weights).unsqueeze(1)
        weighted_design = design * root_weight
        weighted_response = response * root_weight
        xtx += (weighted_design.T @ weighted_design).double()
        xty += (weighted_design.T @ weighted_response).double()
    if not bool(torch.isfinite(xtx).all().item()) or not bool(torch.isfinite(xty).all().item()):
        raise SameGeneJacobianError("Ridge sufficient statistics are nonfinite")
    return xtx, xty, feature_count


def _solve(xtx: torch.Tensor, xty: torch.Tensor, penalty: float) -> torch.Tensor:
    regularizer = torch.eye(xtx.shape[0], dtype=torch.float64, device=xtx.device)
    regularizer[0, 0] = 0.0
    system = xtx + float(penalty) * regularizer
    coefficients = torch.linalg.solve(system, xty).float()
    if not bool(torch.isfinite(coefficients).all().item()):
        raise SameGeneJacobianError("Ridge solve returned nonfinite coefficients")
    return coefficients


def _evaluate(
    coefficients: torch.Tensor,
    target: torch.Tensor,
    morphology: torch.Tensor,
    feature: torch.Tensor | None,
    mask: np.ndarray,
    groups: np.ndarray,
    target_statistics: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    morphology_statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    chunk_size: int,
    device: torch.device,
    collect_component_predictions: bool = False,
) -> dict[str, Any]:
    target_mean, target_std, _ = target_statistics
    positions = np.flatnonzero(mask)
    component_values = sorted(int(value) for value in np.unique(groups[mask]))
    component_lookup = {value: index for index, value in enumerate(component_values)}
    sum_mse = np.zeros(len(component_values), dtype=np.float64)
    sum_mae = np.zeros(len(component_values), dtype=np.float64)
    sum_huber = np.zeros(len(component_values), dtype=np.float64)
    counts = np.zeros(len(component_values), dtype=np.int64)
    genes = target.shape[1]
    component_sum_y = (
        torch.zeros((len(component_values), genes), dtype=torch.float64, device=device)
        if collect_component_predictions
        else None
    )
    component_sum_p = (
        torch.zeros((len(component_values), genes), dtype=torch.float64, device=device)
        if collect_component_predictions
        else None
    )
    sum_y = torch.zeros(genes, dtype=torch.float64, device=device)
    sum_p = torch.zeros(genes, dtype=torch.float64, device=device)
    sum_y2 = torch.zeros(genes, dtype=torch.float64, device=device)
    sum_p2 = torch.zeros(genes, dtype=torch.float64, device=device)
    sum_yp = torch.zeros(genes, dtype=torch.float64, device=device)
    sum_gene_squared_error = torch.zeros(genes, dtype=torch.float64, device=device)
    total_rows = 0
    for start in range(0, len(positions), chunk_size):
        stop = min(start + chunk_size, len(positions))
        cpu_index = positions[start:stop]
        index = torch.from_numpy(cpu_index).to(device)
        design = _design_batch(
            index,
            morphology,
            morphology_statistics,
            feature,
            target_mean,
            target_std,
        )
        response = (target.index_select(0, index) - target_mean) / target_std
        prediction = design @ coefficients
        if not bool(torch.isfinite(prediction).all().item()):
            raise SameGeneJacobianError("Evaluation predictions are nonfinite")
        residual = prediction - response
        absolute = residual.abs()
        row_mse = residual.square().mean(dim=1).double().cpu().numpy()
        row_mae = absolute.mean(dim=1).double().cpu().numpy()
        row_huber = torch.where(
            absolute <= 1.0, 0.5 * residual.square(), absolute - 0.5
        ).mean(dim=1).double().cpu().numpy()
        for group in np.unique(groups[cpu_index]):
            local = groups[cpu_index] == group
            slot = component_lookup[int(group)]
            sum_mse[slot] += float(np.sum(row_mse[local]))
            sum_mae[slot] += float(np.sum(row_mae[local]))
            sum_huber[slot] += float(np.sum(row_huber[local]))
            counts[slot] += int(np.sum(local))
            if component_sum_y is not None and component_sum_p is not None:
                local_index = torch.from_numpy(np.flatnonzero(local)).to(device)
                component_sum_y[slot] += response.double().index_select(
                    0, local_index
                ).sum(dim=0)
                component_sum_p[slot] += prediction.double().index_select(
                    0, local_index
                ).sum(dim=0)
        response64 = response.double()
        prediction64 = prediction.double()
        sum_y += response64.sum(dim=0)
        sum_p += prediction64.sum(dim=0)
        sum_y2 += response64.square().sum(dim=0)
        sum_p2 += prediction64.square().sum(dim=0)
        sum_yp += (response64 * prediction64).sum(dim=0)
        sum_gene_squared_error += residual.double().square().sum(dim=0)
        total_rows += len(cpu_index)
    if np.any(counts == 0):
        raise SameGeneJacobianError("An evaluation component has zero rows")
    per_component = [
        {
            "geometry_group": group,
            "cell_count": int(count),
            "mse": float(sum_mse[index] / count),
            "mae": float(sum_mae[index] / count),
            "huber": float(sum_huber[index] / count),
        }
        for index, (group, count) in enumerate(zip(component_values, counts, strict=True))
    ]
    denominator = torch.sqrt(
        (total_rows * sum_y2 - sum_y.square()).clamp_min(0)
        * (total_rows * sum_p2 - sum_p.square()).clamp_min(0)
    )
    correlation = torch.where(
        denominator > 0,
        (total_rows * sum_yp - sum_y * sum_p) / denominator,
        torch.full_like(denominator, float("nan")),
    )
    result = {
        "cell_count": total_rows,
        "component_count": len(per_component),
        "component_equal_mse": float(np.mean([row["mse"] for row in per_component])),
        "component_equal_mae": float(np.mean([row["mae"] for row in per_component])),
        "component_equal_huber": float(np.mean([row["huber"] for row in per_component])),
        "per_component": per_component,
        "gene_mse": (sum_gene_squared_error / total_rows).cpu().numpy(),
        "gene_pearson": correlation.cpu().numpy(),
    }
    if component_sum_y is not None and component_sum_p is not None:
        result["component_mean_predictions"] = [
            {
                "geometry_group": group,
                "cell_count": int(count),
                "y_true": (component_sum_y[index] / int(count)).cpu().tolist(),
                "y_pred": (component_sum_p[index] / int(count)).cpu().tolist(),
            }
            for index, (group, count) in enumerate(
                zip(component_values, counts, strict=True)
            )
        ]
    return result


def _fit_arm(
    arm: str,
    feature: torch.Tensor | None,
    target: torch.Tensor,
    morphology: torch.Tensor,
    masks: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    chunk_size: int,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    tuning_target_stats = _target_statistics(target, masks["tuning_train"], device)
    tuning_morph_stats = _morphology_statistics(morphology, masks["tuning_train"], device)
    xtx, xty, parameter_rows = _sufficient_statistics(
        target,
        morphology,
        feature,
        masks["tuning_train"],
        groups,
        tuning_target_stats,
        tuning_morph_stats,
        chunk_size=chunk_size,
        device=device,
    )
    validation_rows = []
    best: tuple[float, float] | None = None
    for penalty in RIDGE_GRID:
        coefficients = _solve(xtx, xty, penalty)
        evaluation = _evaluate(
            coefficients,
            target,
            morphology,
            feature,
            masks["validation"],
            groups,
            tuning_target_stats,
            tuning_morph_stats,
            chunk_size=chunk_size,
            device=device,
        )
        mse = float(evaluation["component_equal_mse"])
        validation_rows.append({"penalty": penalty, "component_equal_mse": mse})
        candidate = (mse, penalty)
        if best is None or candidate < best:
            best = candidate
        del coefficients
    assert best is not None
    selected_penalty = best[1]
    del xtx, xty

    final_target_stats = _target_statistics(target, masks["final_train"], device)
    final_morph_stats = _morphology_statistics(morphology, masks["final_train"], device)
    xtx, xty, parameter_rows = _sufficient_statistics(
        target,
        morphology,
        feature,
        masks["final_train"],
        groups,
        final_target_stats,
        final_morph_stats,
        chunk_size=chunk_size,
        device=device,
    )
    coefficients = _solve(xtx, xty, selected_penalty)
    test = _evaluate(
        coefficients,
        target,
        morphology,
        feature,
        masks["test"],
        groups,
        final_target_stats,
        final_morph_stats,
        chunk_size=chunk_size,
        device=device,
        collect_component_predictions=True,
    )
    component_predictions = list(test.pop("component_mean_predictions"))
    gene_mse = np.asarray(test.pop("gene_mse"), dtype=np.float64)
    gene_pearson = np.asarray(test.pop("gene_pearson"), dtype=np.float64)
    target_mean, target_std, prevalence = final_target_stats
    eligible_genes = (prevalence >= 0.05) & (target_std.cpu().numpy() > 1e-6)
    neighbor_matrix = np.empty((0, 0), dtype=np.float32)
    summary: dict[str, Any] | None = None
    if feature is not None:
        offset = 1 + morphology.shape[1]
        neighbor_matrix = coefficients[offset:, :].T.cpu().numpy().astype(np.float32)
        summary = diagonal_summary(neighbor_matrix, eligible_genes).as_dict()
    morphology_median, morphology_mean, morphology_std = final_morph_stats
    fitted_state = {
        "coefficients": coefficients.cpu().numpy().astype(np.float32),
        "target_mean": target_mean.cpu().numpy().astype(np.float32),
        "target_std": target_std.cpu().numpy().astype(np.float32),
        "morphology_median": morphology_median.cpu().numpy().astype(np.float32),
        "morphology_mean": morphology_mean.cpu().numpy().astype(np.float32),
        "morphology_std": morphology_std.cpu().numpy().astype(np.float32),
        "selected_penalty": np.asarray(selected_penalty, dtype=np.float64),
    }
    result = {
        "arm": arm,
        "selected_penalty": selected_penalty,
        "validation_grid": validation_rows,
        "test": test,
        "parameter_count": int(parameter_rows * target.shape[1]),
        "eligible_gene_count": int(np.sum(eligible_genes)),
        "diagonal_summary": summary,
    }
    return (
        result,
        neighbor_matrix,
        eligible_genes,
        gene_mse,
        gene_pearson,
        component_predictions,
        fitted_state,
    )


def _artifact_inventory(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(value for value in directory.rglob("*") if value.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative in {"artifact_manifest.json", "COMPLETED.json"}:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _canonical_final_metrics(result_payload: dict[str, Any]) -> dict[str, float]:
    arms = result_payload["arms"]
    return {
        "validation/observed_near_component_equal_mse": float(
            arms["observed_near"]["test"]["component_equal_mse"]
        ),
        "validation/morphology_only_component_equal_mse": float(
            arms["morphology_only"]["test"]["component_equal_mse"]
        ),
        "validation/observed_annular_component_equal_mse": float(
            arms["observed_annular"]["test"]["component_equal_mse"]
        ),
        "validation/permuted_near_component_equal_mse": float(
            arms["within_fov_permuted_near"]["test"]["component_equal_mse"]
        ),
        "validation/near_gain_vs_morphology": float(
            arms["observed_near"]["test"]["relative_mse_gain_vs_morphology"]
        ),
        "validation/near_gain_vs_permuted": float(
            arms["observed_near"]["test"]["relative_mse_gain_vs_permuted"]
        ),
    }


def _canonical_prediction_rows(
    run_id: str,
    fold: int,
    component_predictions: list[dict[str, Any]],
    component_losses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    loss_by_group = {
        int(item["geometry_group"]): float(item["mse"])
        for item in component_losses
    }
    rows = []
    for item in component_predictions:
        group = int(item["geometry_group"])
        if group not in loss_by_group:
            raise RuntimeError(f"Missing held-out component loss for group {group}")
        rows.append(
            {
                "run_id": run_id,
                "sample_key": f"geometry_group_{group:03d}",
                "dataset_id": DATASET_ID,
                "split": "validation",
                "y_true": item["y_true"],
                "y_pred": item["y_pred"],
                "graph_id": "within_fov_k12_0_25um",
                "fold": int(fold),
                "node_count": int(item["cell_count"]),
                "sample_loss": loss_by_group[group],
            }
        )
    if not rows:
        raise RuntimeError("Canonical held-out component predictions are empty")
    return rows


def _write_canonical_contract(
    archive: RunArchive,
    *,
    configuration: dict[str, Any],
    provenance: dict[str, Any],
    result_payload: dict[str, Any],
    component_predictions: list[dict[str, Any]],
) -> None:
    """Materialize the platform success contract before immutable publication."""

    run_id = str(result_payload["run_id"])
    fold = int(result_payload["outer_fold"])
    final_metrics = _canonical_final_metrics(result_payload)
    configured_primary = configuration.get("evaluation", {}).get("primary_metric")
    canonical_primary = "validation/observed_near_component_equal_mse"
    summary_primary = canonical_primary if configured_primary == canonical_primary else None
    archive.write_manifest(
        {
            "run_id": run_id,
            "status": "completed",
            "campaign_id": CAMPAIGN_ID,
            "profile": result_payload["profile"],
            "fold": fold,
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "split_id": SPLIT_ID,
            "prediction_semantics": (
                "held-out geometry-component mean over evaluated matched-eligible "
                "cells in final-train-standardized log1p units; pilot profiles may "
                "subsample each component, full profiles do not; sample_loss is "
                "component cellwise MSE; validation is only the canonical platform "
                "role for the statistical outer test"
            ),
        }
    )
    archive.write_resolved_config(configuration)
    archive.write_summary(
        {
            "run_id": run_id,
            "status": "completed",
            "profile": result_payload["profile"],
            "fold": fold,
            "primary_metric_name": summary_primary,
            "primary_metric_value": (
                final_metrics[canonical_primary] if summary_primary else None
            ),
            "duration_seconds": float(result_payload["duration_seconds"]),
            "peak_vram_gb": float(result_payload["peak_vram_gb"]),
            "claim_boundary": result_payload["claim_boundary"],
        }
    )
    archive.write_json("metrics/final.json", final_metrics)
    for name, value in final_metrics.items():
        archive.append_metric_event(
            {"name": name, "value": value, "step": 0, "split": "validation"}
        )
    archive.write_text(
        "metrics/history.jsonl",
        canonical_json(
            {
                "fold": fold,
                "profile": result_payload["profile"],
                **final_metrics,
            }
        )
        + "\n",
    )
    prediction_rows = _canonical_prediction_rows(
        run_id,
        fold,
        component_predictions,
        result_payload["arms"]["observed_near"]["test"]["per_component"],
    )
    if not np.isclose(
        np.mean([row["sample_loss"] for row in prediction_rows]),
        final_metrics[canonical_primary],
        rtol=1e-12,
        atol=0.0,
    ):
        raise RuntimeError("Canonical component losses do not reproduce the primary metric")
    archive.write_prediction_jsonl_stream("validation", prediction_rows)
    archive.copy_file(
        archive.scratch_path / "ridge_coefficients_checkpoint.npz",
        "checkpoints/best.ckpt",
    )
    archive.prepare_log_files()
    archive.write_json(
        "provenance/git.json",
        {
            "commit": provenance.get("git_commit"),
            "dirty_fingerprint": provenance.get("dirty_fingerprint"),
            "git_status": provenance.get("git_status", ""),
            "tracked_diff_sha256": provenance.get("tracked_diff_sha256"),
            "untracked_files": provenance.get("untracked_files", []),
            "source_snapshot_manifest": provenance.get(
                "source_snapshot_manifest", []
            ),
        },
    )
    tracked_diff = provenance.get("tracked_diff")
    if not isinstance(tracked_diff, str):
        tracked_diff = (
            "# Original runner retained only tracked_diff_sha256; "
            "see provenance.json.\n"
        )
    archive.write_text("provenance/uncommitted_changes.patch", tracked_diff)
    hardware = dict(provenance.get("hardware", {}))
    archive.write_json("provenance/hardware.json", hardware)
    archive.write_text(
        "provenance/environment.txt",
        "\n".join(f"{key}={hardware[key]}" for key in sorted(hardware)) + "\n",
    )
    fingerprints = result_payload["dataset_fingerprints"]
    archive.write_json(
        "provenance/data_fingerprints.json",
        {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "raw": fingerprints["raw"],
            "processed": fingerprints["processed"],
        },
    )
    archive.write_json(
        "provenance/split_fingerprint.json",
        {"split_id": SPLIT_ID, "sha256": fingerprints["split"]},
    )
    command = provenance.get("command", [])
    archive.write_text(
        "provenance/command.txt",
        " ".join(str(value) for value in command) + "\n",
    )
    project_root = Path(str(provenance.get("working_directory", "")))
    for item in provenance.get("source_snapshot_manifest", []):
        relative = str(item["path"])
        source = project_root / relative
        if (
            not source.is_file()
            or source.stat().st_size != int(item["size_bytes"])
            or sha256_file(source) != str(item["sha256"])
        ):
            raise RuntimeError(f"Provenance source changed during the run: {relative}")
        archive.copy_file(source, Path("provenance/source_snapshot") / relative)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each fold process requires exactly one visible CUDA device; set CUDA_VISIBLE_DEVICES"
        )
    if arguments.fold not in range(4):
        raise ValueError("fold must be 0, 1, 2, or 3")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("highest")
    paths = current_paths()
    prepared = arguments.prepared.resolve(strict=True)
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["processed_fingerprint"] != arguments.processed_fingerprint:
        raise RuntimeError("Prepared fingerprint does not match the reviewed invocation")
    if manifest["split_fingerprint"] != arguments.split_fingerprint:
        raise RuntimeError("Split fingerprint does not match the reviewed invocation")
    if manifest["raw_snapshot"]["fingerprint"] != arguments.raw_fingerprint:
        raise RuntimeError("Raw fingerprint does not match the reviewed invocation")
    control = planted_recovery_control()
    if control["passed"] is not True:
        raise RuntimeError(f"Analytical control failed: {control}")

    configuration = {
        "campaign_id": CAMPAIGN_ID,
        "profile": arguments.profile,
        "dataset": {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "dataset_fingerprint": arguments.raw_fingerprint,
            "processed_fingerprint": arguments.processed_fingerprint,
            "split_id": SPLIT_ID,
            "split_fingerprint": arguments.split_fingerprint,
            "preprocessing_version": PREPROCESSING_VERSION,
        },
        "model": {
            "family": "additive_multivariate_ridge",
            "ridge_grid": list(RIDGE_GRID),
            "arms": list(ARMS),
            "gene_count": 1000,
            "morphology_feature_count": 22,
        },
        "graph": {
            "partition": "fov",
            "k": 12,
            "near_um": [0, 25],
            "annular_um": [25, 50],
            "self_edges": False,
        },
        "masking_type": "whole_node",
        "trainer": {
            "method": "closed_form_ridge",
            "primary_checkpoint_role": "best",
            "restore_best": True,
        },
        "evaluation": {
            "fold": arguments.fold,
            "component_equal": True,
            "canonical_prediction_split": "validation",
            "statistical_partition": "outer_geometry_test",
            "canonical_role_note": (
                "platform validation role stores predictions from the frozen outer test"
            ),
            "primary_metric": "validation/observed_near_component_equal_mse",
        },
        "classification": {
            "lifecycle_stage": "diagnostic" if arguments.profile == "pilot" else "exploratory_screen",
            "study_axis": "interpretation",
            "variant_label": f"same_gene_ridge_{arguments.profile}",
            "seed": 810,
            "fold": arguments.fold,
            "attempt": arguments.attempt,
        },
    }
    provenance = _provenance(paths.project_root, configuration)
    sci_id = scientific_id(configuration)
    rep_id = repro_id(
        configuration,
        git_commit=provenance["git_commit"],
        dirty_fingerprint=provenance["dirty_fingerprint"],
        dataset_fingerprint=arguments.raw_fingerprint,
        split_fingerprint=arguments.split_fingerprint,
        preprocessing_version=PREPROCESSING_VERSION,
        environment_fingerprint=provenance["environment_fingerprint"],
    )
    run_id = create_run_id(
        seed=810,
        fold=arguments.fold,
        attempt=arguments.attempt,
        scientific_id_value=sci_id,
    )
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    if arguments.retry_of:
        prior = registry.get_run(arguments.retry_of)
        if (
            prior is None
            or str(prior["campaign_id"]) != CAMPAIGN_ID
            or int(prior["fold"]) != arguments.fold
        ):
            raise RuntimeError("--retry-of must name the same campaign and fold")
    archive = RunArchive.create(run_id, paths=paths)
    scratch = archive.scratch_path
    final = archive.artifact_path
    registry.register_variant(
        sci_id,
        campaign_id=CAMPAIGN_ID,
        configuration=configuration,
        model_family="additive_multivariate_ridge",
        masking_type="whole_node",
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        split_id=SPLIT_ID,
        use_edge_features=False,
        embedding_dim=1000,
        neighbor_k=12,
    )
    registry.create_run(
        run_id,
        campaign_id=CAMPAIGN_ID,
        scientific_id=sci_id,
        repro_id=rep_id,
        seed=810,
        fold=arguments.fold,
        attempt=arguments.attempt,
        configuration=configuration,
        status="pending",
        artifact_path=final,
        git_commit=provenance["git_commit"],
        dirty_status=provenance["dirty_fingerprint"] is not None,
        preprocessing_version=PREPROCESSING_VERSION,
        dataset_fingerprint=arguments.raw_fingerprint,
        split_fingerprint=arguments.split_fingerprint,
        host=socket.gethostname(),
        gpu_model=provenance["hardware"]["gpu_name"],
        retry_of=arguments.retry_of,
    )
    categories = {
        "lifecycle_stage": "diagnostic" if arguments.profile == "pilot" else "exploratory-screen",
        "study_axis": "interpretation",
        "model": "ridge",
        "graph": "near-annular-null",
        "mask": "whole-node",
        "edge_feature_state": False,
        "embedding_dim": 1000,
        "seed": 810,
        "fold": arguments.fold,
        "attempt": arguments.attempt,
        "variant_evidence": {
            "profile": arguments.profile,
            "processed_fingerprint": arguments.processed_fingerprint,
        },
    }
    alias = semantic_run_alias(primary_run_id=run_id, categories=categories, historical=False)
    registry.register_run_semantics(
        run_id,
        lifecycle_stage=categories["lifecycle_stage"],
        study_axis="interpretation",
        source_batch="same_gene_cross_cell_jacobian_20260810",
        seed_known=True,
        fold_known=True,
        attempt_known=True,
        retention_class="diagnostic" if arguments.profile == "pilot" else "exploratory_screen",
        category_key=f"same_gene_jacobian.{arguments.profile}.fold{arguments.fold}",
        classification_confidence="high",
        timestamp_basis="execution_start",
        variant_label=f"same_gene_ridge_{arguments.profile}",
        model_key="ridge",
        dataset_key=DATASET_ID,
        masking_key="whole_node",
        graph_key="near_annular_null",
        feature_key="neighbor_expression_plus_morphology",
        embedding_key="g1000",
        semantic_alias=alias,
        preferred_alias_type="semantic",
    )
    registry.transition_run(run_id, "running", start_time=utc_now())

    started = time.perf_counter()
    try:
        folds = _load_cpu_vector(prepared, "fold.npy").astype(np.int8, copy=False)
        groups = _load_cpu_vector(prepared, "geometry_group.npy").astype(np.int16, copy=False)
        eligible = _load_cpu_vector(prepared, "matched_eligible.npy").astype(bool, copy=False)
        masks = _split_masks(folds, groups, eligible, arguments.fold, profile=arguments.profile)
        target = _load_gpu_matrix(prepared, "expression_log1p.npy", device)
        morphology = _load_gpu_matrix(prepared, "metadata.npy", device)
        if len(folds) != target.shape[0] or target.shape[1] != 1000 or morphology.shape[1] != 22:
            raise SameGeneJacobianError("Prepared training shapes violate the frozen contract")

        results: dict[str, Any] = {}
        matrices: dict[str, np.ndarray] = {}
        eligibility_by_arm: dict[str, np.ndarray] = {}
        gene_metrics: dict[str, np.ndarray] = {}
        fitted_states: dict[str, dict[str, np.ndarray]] = {}
        canonical_component_predictions: list[dict[str, Any]] | None = None

        (
            result,
            matrix,
            gene_eligible,
            gene_mse,
            gene_pearson,
            _component_predictions,
            fitted_state,
        ) = _fit_arm(
            "morphology_only",
            None,
            target,
            morphology,
            masks,
            groups,
            chunk_size=arguments.chunk_size,
            device=device,
        )
        results["morphology_only"] = result
        eligibility_by_arm["morphology_only"] = gene_eligible
        gene_metrics["morphology_only_mse"] = gene_mse
        gene_metrics["morphology_only_pearson"] = gene_pearson
        fitted_states["morphology_only"] = fitted_state

        for arm in ("observed_near", "observed_annular", "within_fov_permuted_near"):
            feature = _load_gpu_matrix(prepared, FEATURE_FILE[arm], device)
            (
                result,
                matrix,
                gene_eligible,
                gene_mse,
                gene_pearson,
                arm_component_predictions,
                fitted_state,
            ) = _fit_arm(
                arm,
                feature,
                target,
                morphology,
                masks,
                groups,
                chunk_size=arguments.chunk_size,
                device=device,
            )
            results[arm] = result
            matrices[arm] = matrix
            eligibility_by_arm[arm] = gene_eligible
            gene_metrics[f"{arm}_mse"] = gene_mse
            gene_metrics[f"{arm}_pearson"] = gene_pearson
            fitted_states[arm] = fitted_state
            if arm == "observed_near":
                canonical_component_predictions = arm_component_predictions
            del feature
            torch.cuda.empty_cache()

        (
            result,
            matrix,
            gene_eligible,
            gene_mse,
            gene_pearson,
            _component_predictions,
            fitted_state,
        ) = _fit_arm(
            "same_cell_oracle",
            target,
            target,
            morphology,
            masks,
            groups,
            chunk_size=arguments.chunk_size,
            device=device,
        )
        results["same_cell_oracle"] = result
        matrices["same_cell_oracle"] = matrix
        eligibility_by_arm["same_cell_oracle"] = gene_eligible
        gene_metrics["same_cell_oracle_mse"] = gene_mse
        gene_metrics["same_cell_oracle_pearson"] = gene_pearson
        fitted_states["same_cell_oracle"] = fitted_state
        if canonical_component_predictions is None:
            raise RuntimeError("Observed-near component predictions were not captured")

        baseline_mse = results["morphology_only"]["test"]["component_equal_mse"]
        for arm in ("observed_near", "observed_annular", "within_fov_permuted_near", "same_cell_oracle"):
            value = results[arm]["test"]["component_equal_mse"]
            results[arm]["test"]["relative_mse_gain_vs_morphology"] = (
                (baseline_mse - value) / baseline_mse
            )
        permuted_mse = results["within_fov_permuted_near"]["test"]["component_equal_mse"]
        near_mse = results["observed_near"]["test"]["component_equal_mse"]
        results["observed_near"]["test"]["relative_mse_gain_vs_permuted"] = (
            (permuted_mse - near_mse) / permuted_mse
        )

        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        duration = time.perf_counter() - started
        full_base_counts = {
            "tuning_train": int(np.sum((folds != arguments.fold) & (folds != (arguments.fold + 1) % 4) & eligible)),
            "validation": int(np.sum((folds == (arguments.fold + 1) % 4) & eligible)),
            "final_train": int(np.sum((folds != arguments.fold) & eligible)),
            "test": int(np.sum((folds == arguments.fold) & eligible)),
        }
        used_counts = {name: int(np.sum(mask)) for name, mask in masks.items()}
        maximum_scale = max(full_base_counts[name] / used_counts[name] for name in used_counts)
        projected_hours = duration * maximum_scale / 3600
        controls = {
            "analytical_planted_recovery": control,
            "primary_receiver_expression_input": False,
            "primary_self_jacobian_exact_zero_by_design": True,
            "same_cell_oracle": results["same_cell_oracle"]["diagonal_summary"],
            "all_outputs_finite": True,
            "peak_vram_gb": peak_vram_gb,
            "peak_vram_gate_passed": peak_vram_gb <= 20.5,
            "projected_full_hours_per_fold": projected_hours,
            "runtime_gate_passed": projected_hours <= 2.0,
        }
        if not controls["peak_vram_gate_passed"] or not controls["runtime_gate_passed"]:
            raise RuntimeError(f"Pilot/full resource gate failed: {controls}")
        if results["same_cell_oracle"]["diagonal_summary"]["row_top1_fraction"] < 0.95:
            raise RuntimeError("Same-cell oracle failed to recover the diagonal")

        result_payload = {
            "run_id": run_id,
            "semantic_alias": alias,
            "campaign_id": CAMPAIGN_ID,
            "profile": arguments.profile,
            "outer_fold": arguments.fold,
            "status": "completed",
            "duration_seconds": duration,
            "peak_vram_gb": peak_vram_gb,
            "sample_counts": used_counts,
            "full_profile_sample_counts": full_base_counts,
            "controls": controls,
            "arms": results,
            "dataset_fingerprints": {
                "raw": arguments.raw_fingerprint,
                "processed": arguments.processed_fingerprint,
                "split": arguments.split_fingerprint,
            },
            "claim_boundary": (
                "exploratory held-out-geometry model sensitivity; not communication, "
                "mechanism, causality, or patient generalization"
            ),
        }
        _json_write(scratch / "resolved_configuration.json", configuration)
        _json_write(scratch / "provenance.json", provenance)
        _json_write(scratch / "results.json", result_payload)
        np.savez_compressed(
            scratch / "ridge_coefficients_checkpoint.npz",
            genes=np.asarray(json.loads((prepared / "genes.json").read_text(encoding="utf-8"))),
            **{f"jacobian_{key}": value for key, value in matrices.items()},
            **{f"eligible_{key}": value for key, value in eligibility_by_arm.items()},
            **gene_metrics,
            **{
                f"state_{arm}_{name}": value
                for arm, state in fitted_states.items()
                for name, value in state.items()
            },
        )
        inventory = _artifact_inventory(scratch)
        _json_write(
            scratch / "artifact_manifest.json",
            {
                "run_id": run_id,
                "file_count": len(inventory),
                "files": inventory,
                "manifest_payload_sha256": canonical_sha256(inventory),
            },
        )
        _json_write(
            scratch / "COMPLETED.json",
            {"run_id": run_id, "completed_at": utc_now(), "verified": True},
        )
        _write_canonical_contract(
            archive,
            configuration=configuration,
            provenance=provenance,
            result_payload=result_payload,
            component_predictions=canonical_component_predictions,
        )
        archive.publish_success_pending()
        verify_unmarked_run_bundle(final)

        registry.transition_run(
            run_id,
            "finalizing",
            end_time=utc_now(),
            duration_seconds=duration,
            peak_vram_gb=peak_vram_gb,
            parameter_count=sum(int(value["parameter_count"]) for value in results.values()),
            primary_metric_name="validation/observed_near_component_equal_mse",
            primary_metric_value=near_mse,
            artifact_path=final,
        )
        artifact_ids: dict[str, int] = {}
        for path in sorted(value for value in final.rglob("*") if value.is_file()):
            relative = path.relative_to(final).as_posix()
            top_level = relative.split("/", 1)[0]
            kind = (
                top_level
                if top_level
                in {
                    "checkpoints",
                    "predictions",
                    "metrics",
                    "diagnostics",
                    "interpretation",
                    "provenance",
                    "logs",
                }
                else "run_metadata"
            )
            artifact_ids[relative] = registry.record_artifact(
                run_id,
                kind=kind,
                path=path,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        registry.register_checkpoint_metadata(
            artifact_ids["checkpoints/best.ckpt"],
            run_id=run_id,
            role="best",
            retention_class="diagnostic" if arguments.profile == "pilot" else "exploratory_screen",
            verification_status="verified",
            monitored_metric="validation/observed_near_component_equal_mse",
            monitored_mode="min",
            monitored_value=near_mse,
            metadata={
                "closed_form_ridge": True,
                "prediction_replayable": True,
                "arms": list(ARMS),
                "state_fields": [
                    "coefficients",
                    "target_mean",
                    "target_std",
                    "morphology_median",
                    "morphology_mean",
                    "morphology_std",
                    "selected_penalty",
                ],
            },
        )
        for arm, value in results.items():
            registry.record_metric(
                run_id,
                f"validation/{arm}_component_equal_mse",
                value["test"]["component_equal_mse"],
                split="validation",
            )
        registry.record_metric(
            run_id,
            "validation/observed_near_relative_mse_gain_vs_permuted",
            results["observed_near"]["test"]["relative_mse_gain_vs_permuted"],
            split="validation",
        )
        archive.mark_success()
        verify_run_bundle(final)
        registry.transition_run(run_id, "completed")
        registry_issues = registry.verify_artifacts(run_id=run_id)
        if registry_issues:
            raise RuntimeError(f"Registered artifact verification failed: {registry_issues}")
        return {
            "run_id": run_id,
            "semantic_alias": alias,
            "profile": arguments.profile,
            "fold": arguments.fold,
            "artifact_path": str(final),
            "duration_seconds": duration,
            "peak_vram_gb": peak_vram_gb,
            "observed_near_mse": near_mse,
            "near_gain_vs_morphology": results["observed_near"]["test"]["relative_mse_gain_vs_morphology"],
            "near_gain_vs_permuted": results["observed_near"]["test"]["relative_mse_gain_vs_permuted"],
            "near_diagonal_summary": results["observed_near"]["diagonal_summary"],
        }
    except BaseException as error:
        registry.record_failure(
            run_id=run_id,
            category="same_gene_jacobian_run_failure",
            message=str(error),
            details={"type": type(error).__name__, "profile": arguments.profile, "fold": arguments.fold},
        )
        current = registry.get_run(run_id)
        success_marker = final / "_SUCCESS"
        if (
            current
            and current["status"] in {"pending", "running", "finalizing"}
            and not success_marker.is_file()
        ):
            registry.transition_run(
                run_id,
                "failed",
                end_time=utc_now(),
                duration_seconds=time.perf_counter() - started,
                failure_category="same_gene_jacobian_run_failure",
            )
            try:
                if final.is_dir() and not any(
                    (final / marker).exists()
                    for marker in ("_SUCCESS", "_FAILED", "_PRUNED")
                ):
                    RunArchive.from_published(run_id, paths=paths).mark_published_failure()
                elif archive.scratch_path.is_dir():
                    archive.finalize_failure(
                        error,
                        failure_category="same_gene_jacobian_run_failure",
                    )
            except BaseException as finalization_error:
                registry.record_failure(
                    run_id=run_id,
                    category="same_gene_jacobian_failure_compensation",
                    message=str(finalization_error),
                    details={"type": type(finalization_error).__name__},
                )
        raise


def parse_args() -> argparse.Namespace:
    paths = current_paths()
    prepared = paths.data_root / "processed" / "same_gene_cross_cell_jacobian_v1"
    manifest = (
        json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        if (prepared / "manifest.json").is_file()
        else {}
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--profile", choices=("pilot", "full"), default="full")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--retry-of")
    parser.add_argument("--prepared", type=Path, default=prepared)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--raw-fingerprint",
        default=manifest.get("raw_snapshot", {}).get("fingerprint"),
        required=not bool(manifest),
    )
    parser.add_argument(
        "--processed-fingerprint",
        default=manifest.get("processed_fingerprint"),
        required=not bool(manifest),
    )
    parser.add_argument(
        "--split-fingerprint",
        default=manifest.get("split_fingerprint"),
        required=not bool(manifest),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.chunk_size < 256:
        raise ValueError("chunk-size must be at least 256")
    if arguments.attempt < 1:
        raise ValueError("attempt must be positive")
    result = run(arguments)
    print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
