#!/usr/bin/env python3
"""Train one fold of the frozen nonlinear same-gene replication campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time
from typing import Any, Mapping

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
    component_equal_weights,
    diagonal_summary,
)
from spatial_benchmark.same_gene_nonlinear import (
    AdditiveNeighborMLP,
    planted_nonlinear_jacobian_control,
)


CAMPAIGN_ID = "cmp_20260810_same_gene_nonlinear_replication"
CAMPAIGN_DISPLAY_NAME = "Nonlinear same-gene cross-cell replication"
CAMPAIGN_SCIENTIFIC_QUESTION = (
    "Does nonlinear neighbor modeling replicate same-gene enrichment?"
)
EXPERIMENT_FLAVOR = "replication"
FROZEN_CONTRACT_SHA256 = (
    "5b9f0058e011084cf28ea3e1fc78ebba787e0cf07dd35adde19381c5f6feb66e"
)
DATASET_ID = "gastric_cosmx_drive_public"
DATASET_VERSION = "drive_snapshot_20260810"
SPLIT_ID = "opaque_geometry_components_075mm_4fold_v1"
PREPROCESSING_VERSION = "same_gene_neighbor_log1p_v1"
RAW_FINGERPRINT = "e1513d598d4ea910386842cdf4a6d9bd58e21318d484bdb34dfb3962f1490ea5"
PROCESSED_FINGERPRINT = (
    "6304132b4a57699c81b8616324dbeb2faee24b58ce70490be552595d84af34ce"
)
SPLIT_FINGERPRINT = (
    "12c0d46244ed443a482586fc85422672f9f04132c7def49a741c40ba48bf4264"
)
SLIDES = ("SO_1", "SO_2")
FULL_ARMS = (
    "morphology_only",
    "observed_near",
    "observed_annular",
    "within_fov_permuted_near",
)
FEATURE_FILE = {
    "observed_near": "neighbor_near_mean.npy",
    "observed_annular": "neighbor_annular_mean.npy",
    "within_fov_permuted_near": "neighbor_permuted_near_mean.npy",
}
EPOCH_CANDIDATES = (1, 2, 4, 8, 12)
PILOT_EPOCH_CANDIDATES = (1, 2)
PILOT_ARMS = ("observed_near",)
PILOT_REFIT_EPOCH_OVERRIDE: int | None = None
ANCHOR_EPOCH: int | None = None
HIDDEN_COUNT = 64
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED_BASE = 20260810
TRACKING_SEED = 810
PRIMARY_METRIC_SUFFIX = "observed_near_component_equal_mse"
PROVENANCE_SOURCE_PATHS = (
    "src/spatial_benchmark/same_gene_jacobian.py",
    "src/spatial_benchmark/same_gene_nonlinear.py",
    "scripts/data/prepare_same_gene_jacobian.py",
    "scripts/train/run_same_gene_nonlinear.py",
    "tests/unit/spatial_benchmark/test_same_gene_nonlinear.py",
    "experiments/campaigns/cmp_20260810_same_gene_nonlinear_replication/README.md",
    "experiments/campaigns/cmp_20260810_same_gene_nonlinear_replication/campaign.yaml",
    "experiments/campaigns/cmp_20260810_same_gene_nonlinear_replication/frozen_task_contract.yaml",
)


class NonlinearRunError(RuntimeError):
    """Raised when a frozen nonlinear run violates its execution contract."""


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _all_numeric_values_finite(value: Any) -> bool:
    """Recursively verify every numeric value that will enter a result bundle."""

    if isinstance(value, Mapping):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_numeric_values_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


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
    tracked_diff = _git_output(root, "diff", "--binary", "HEAD")
    untracked_raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    ).stdout
    untracked_files: list[dict[str, Any]] = []
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
    missing_sources = [
        relative
        for relative in PROVENANCE_SOURCE_PATHS
        if not (root / relative).is_file()
    ]
    if missing_sources:
        raise NonlinearRunError(
            f"declared provenance sources are missing: {missing_sources}"
        )
    source_snapshot_manifest = [
        {
            "path": relative,
            "size_bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for relative in PROVENANCE_SOURCE_PATHS
    ]
    dirty_payload = {
        "status": status,
        "tracked_diff": tracked_diff,
        "untracked_files": untracked_files,
        "source_snapshot_manifest": source_snapshot_manifest,
    }
    dirty = canonical_sha256(dirty_payload) if status or tracked_diff else None
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
    return {
        "git_commit": commit,
        "dirty_fingerprint": dirty,
        "git_status": status,
        "tracked_diff": tracked_diff,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff.encode()).hexdigest(),
        "untracked_files": untracked_files,
        "source_snapshot_manifest": source_snapshot_manifest,
        "hardware": hardware,
        "environment_fingerprint": canonical_sha256(hardware),
        "configuration_sha256": canonical_sha256(configuration),
        "command": sys.argv,
        "working_directory": str(root),
    }


def _load_cpu_vector(prepared: Path, filename: str) -> np.ndarray:
    return np.concatenate(
        [np.load(prepared / slide / filename, allow_pickle=False) for slide in SLIDES]
    )


def _load_gpu_matrix(
    prepared: Path, filename: str, device: torch.device
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for slide in SLIDES:
        array = np.load(prepared / slide / filename, allow_pickle=False)
        if array.ndim != 2:
            raise NonlinearRunError(f"{filename} is not a matrix")
        parts.append(torch.from_numpy(array).to(device=device, dtype=torch.float32))
        del array
    result = torch.cat(parts, dim=0)
    if not bool(torch.isfinite(result).all().item()):
        raise NonlinearRunError(f"{filename} contains nonfinite values")
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
        chosen = rng.choice(
            candidates, size=min(len(candidates), per_group), replace=False
        )
        selected[chosen] = True
    remaining = maximum - int(np.sum(selected))
    if remaining > 0:
        candidates = np.flatnonzero(base_mask & ~selected)
        rng = np.random.default_rng(seed + 99173)
        chosen = rng.choice(
            candidates, size=min(remaining, len(candidates)), replace=False
        )
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
    validation_fold = (outer_fold + 1) % 4
    base = {
        "tuning_train": (
            (folds != outer_fold) & (folds != validation_fold) & eligible
        ),
        "validation": (folds == validation_fold) & eligible,
        "final_train": (folds != outer_fold) & eligible,
        "test": (folds == outer_fold) & eligible,
    }
    caps = (
        {
            "tuning_train": 24000,
            "validation": 12000,
            "final_train": 36000,
            "test": 12000,
        }
        if profile == "pilot"
        else {name: None for name in base}
    )
    masks = {
        name: _capped_mask(
            mask,
            groups,
            caps[name],
            seed=SEED_BASE + outer_fold * 100 + offset,
        )
        for offset, (name, mask) in enumerate(base.items())
    }
    for name, mask in masks.items():
        if not np.any(mask):
            raise NonlinearRunError(f"{name} mask is empty")
    if np.any(masks["test"] & masks["final_train"]):
        raise NonlinearRunError("outer test cells leaked into final training")
    if set(np.unique(groups[masks["test"]])) & set(
        np.unique(groups[masks["final_train"]])
    ):
        raise NonlinearRunError("a geometry component crosses train and test")
    component_sets = {
        name: set(int(value) for value in np.unique(groups[mask]))
        for name, mask in masks.items()
    }
    for first, second in (
        ("tuning_train", "validation"),
        ("tuning_train", "test"),
        ("validation", "test"),
        ("final_train", "test"),
    ):
        overlap = component_sets[first] & component_sets[second]
        if overlap:
            raise NonlinearRunError(
                f"geometry components overlap between {first} and {second}: "
                f"{sorted(overlap)}"
            )
    return masks


def _target_statistics(
    target: torch.Tensor, mask: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    index = torch.from_numpy(np.flatnonzero(mask)).to(device)
    selected = target.index_select(0, index)
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, correction=0).clamp_min(1e-6)
    prevalence = (selected > 0).float().mean(dim=0).cpu().numpy()
    return mean, std, prevalence


def _morphology_statistics(
    morphology: torch.Tensor, mask: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.from_numpy(np.flatnonzero(mask)).to(device)
    selected = morphology.index_select(0, index).detach().cpu().numpy()
    median_array = np.nanmedian(selected, axis=0).astype(np.float32, copy=False)
    if not bool(np.isfinite(median_array).all()):
        raise NonlinearRunError("a morphology column has no finite training value")
    imputed = np.where(np.isfinite(selected), selected, median_array)
    mean_array = np.mean(imputed, axis=0, dtype=np.float64).astype(np.float32)
    std_array = np.std(imputed, axis=0, dtype=np.float64).astype(np.float32)
    median = torch.from_numpy(median_array).to(device)
    mean = torch.from_numpy(mean_array).to(device)
    std = torch.from_numpy(np.maximum(std_array, 1e-6)).to(device)
    return median, mean, std


def _normalized_batch(
    cpu_indices: np.ndarray,
    *,
    target: torch.Tensor,
    morphology: torch.Tensor,
    feature: torch.Tensor | None,
    target_stats: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    morphology_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    index = torch.from_numpy(cpu_indices).to(device)
    target_mean, target_std, _ = target_stats
    median, morphology_mean, morphology_std = morphology_stats
    response = (target.index_select(0, index) - target_mean) / target_std
    morph = morphology.index_select(0, index)
    morph = torch.where(torch.isfinite(morph), morph, median)
    morph = (morph - morphology_mean) / morphology_std
    neighbor = (
        None
        if feature is None
        else (feature.index_select(0, index) - target_mean) / target_std
    )
    return response, morph, neighbor


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _new_model(*, use_neighbor: bool, seed: int, device: torch.device) -> AdditiveNeighborMLP:
    _set_seed(seed)
    return AdditiveNeighborMLP(
        gene_count=1000,
        morphology_count=22,
        hidden_count=HIDDEN_COUNT,
        use_neighbor=use_neighbor,
    ).to(device)


def _train_epochs(
    model: AdditiveNeighborMLP,
    *,
    optimizer: torch.optim.Optimizer,
    target: torch.Tensor,
    morphology: torch.Tensor,
    feature: torch.Tensor | None,
    train_mask: np.ndarray,
    groups: np.ndarray,
    target_stats: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    morphology_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    epochs: int,
    start_epoch: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float]]:
    positions = np.flatnonzero(train_mask)
    global_weights = component_equal_weights(groups, train_mask)
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(start_epoch, start_epoch + epochs):
        order = np.random.default_rng(seed + epoch * 104729).permutation(positions)
        weighted_loss_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            cpu_index = order[start : start + BATCH_SIZE]
            response, morph, neighbor = _normalized_batch(
                cpu_index,
                target=target,
                morphology=morphology,
                feature=feature,
                target_stats=target_stats,
                morphology_stats=morphology_stats,
                device=device,
            )
            prediction = model(morph, neighbor)
            row_mse = (prediction - response).square().mean(dim=1)
            weights = torch.from_numpy(
                global_weights[cpu_index].astype(np.float32, copy=False)
            ).to(device)
            loss = (len(positions) / len(cpu_index)) * torch.sum(weights * row_mse)
            if not bool(torch.isfinite(loss).item()):
                raise NonlinearRunError("training loss is nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted_loss_sum += float(torch.sum(weights * row_mse.detach()).item())
        history.append(
            {
                "epoch": float(epoch + 1),
                "component_equal_training_mse_path_sum": weighted_loss_sum,
            }
        )
    return history


@torch.no_grad()
def _evaluate(
    model: AdditiveNeighborMLP,
    *,
    target: torch.Tensor,
    morphology: torch.Tensor,
    feature: torch.Tensor | None,
    mask: np.ndarray,
    groups: np.ndarray,
    target_stats: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    morphology_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    detailed: bool,
) -> dict[str, Any]:
    model.eval()
    positions = np.flatnonzero(mask)
    component_values = sorted(int(value) for value in np.unique(groups[mask]))
    lookup = {value: index for index, value in enumerate(component_values)}
    counts = np.zeros(len(component_values), dtype=np.int64)
    sum_mse = np.zeros(len(component_values), dtype=np.float64)
    sum_mae = np.zeros(len(component_values), dtype=np.float64)
    sum_y = np.zeros((len(component_values), 1000), dtype=np.float64) if detailed else None
    sum_p = np.zeros((len(component_values), 1000), dtype=np.float64) if detailed else None
    gene_squared_error = torch.zeros(1000, dtype=torch.float64, device=device)
    gene_sum_y = torch.zeros(1000, dtype=torch.float64, device=device)
    gene_sum_p = torch.zeros(1000, dtype=torch.float64, device=device)
    gene_sum_y2 = torch.zeros(1000, dtype=torch.float64, device=device)
    gene_sum_p2 = torch.zeros(1000, dtype=torch.float64, device=device)
    gene_sum_yp = torch.zeros(1000, dtype=torch.float64, device=device)
    for start in range(0, len(positions), BATCH_SIZE):
        cpu_index = positions[start : start + BATCH_SIZE]
        response, morph, neighbor = _normalized_batch(
            cpu_index,
            target=target,
            morphology=morphology,
            feature=feature,
            target_stats=target_stats,
            morphology_stats=morphology_stats,
            device=device,
        )
        prediction = model(morph, neighbor)
        residual = prediction - response
        row_mse = residual.square().mean(dim=1).double().cpu().numpy()
        row_mae = residual.abs().mean(dim=1).double().cpu().numpy()
        response_cpu = response.double().cpu().numpy() if detailed else None
        prediction_cpu = prediction.double().cpu().numpy() if detailed else None
        for group in np.unique(groups[cpu_index]):
            local = groups[cpu_index] == group
            slot = lookup[int(group)]
            counts[slot] += int(np.sum(local))
            sum_mse[slot] += float(np.sum(row_mse[local]))
            sum_mae[slot] += float(np.sum(row_mae[local]))
            if detailed and sum_y is not None and sum_p is not None:
                assert response_cpu is not None and prediction_cpu is not None
                sum_y[slot] += np.sum(response_cpu[local], axis=0)
                sum_p[slot] += np.sum(prediction_cpu[local], axis=0)
        if detailed:
            response64 = response.double()
            prediction64 = prediction.double()
            gene_squared_error += residual.double().square().sum(dim=0)
            gene_sum_y += response64.sum(dim=0)
            gene_sum_p += prediction64.sum(dim=0)
            gene_sum_y2 += response64.square().sum(dim=0)
            gene_sum_p2 += prediction64.square().sum(dim=0)
            gene_sum_yp += (response64 * prediction64).sum(dim=0)
    if np.any(counts == 0):
        raise NonlinearRunError("an evaluation component has zero cells")
    per_component = [
        {
            "geometry_group": group,
            "cell_count": int(count),
            "mse": float(sum_mse[index] / count),
            "mae": float(sum_mae[index] / count),
        }
        for index, (group, count) in enumerate(zip(component_values, counts, strict=True))
    ]
    result: dict[str, Any] = {
        "cell_count": int(np.sum(counts)),
        "component_count": len(per_component),
        "component_equal_mse": float(np.mean([row["mse"] for row in per_component])),
        "component_equal_mae": float(np.mean([row["mae"] for row in per_component])),
        "per_component": per_component,
    }
    if detailed:
        assert sum_y is not None and sum_p is not None
        result["component_mean_predictions"] = [
            {
                "geometry_group": group,
                "cell_count": int(count),
                "y_true": (sum_y[index] / count).tolist(),
                "y_pred": (sum_p[index] / count).tolist(),
            }
            for index, (group, count) in enumerate(
                zip(component_values, counts, strict=True)
            )
        ]
        total = int(np.sum(counts))
        denominator = torch.sqrt(
            (total * gene_sum_y2 - gene_sum_y.square()).clamp_min(0)
            * (total * gene_sum_p2 - gene_sum_p.square()).clamp_min(0)
        )
        pearson = torch.where(
            denominator > 0,
            (total * gene_sum_yp - gene_sum_y * gene_sum_p) / denominator,
            torch.full_like(denominator, float("nan")),
        )
        result["gene_mse"] = (gene_squared_error / total).cpu().numpy()
        result["gene_pearson"] = pearson.cpu().numpy()
    return result


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _evaluate_with_jacobian(
    model: AdditiveNeighborMLP,
    *,
    target: torch.Tensor,
    morphology: torch.Tensor,
    feature: torch.Tensor | None,
    mask: np.ndarray,
    groups: np.ndarray,
    target_stats: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    morphology_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eligible_genes: np.ndarray,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, np.ndarray] | None,
    dict[str, Any] | None,
]:
    evaluation = _evaluate(
        model,
        target=target,
        morphology=morphology,
        feature=feature,
        mask=mask,
        groups=groups,
        target_stats=target_stats,
        morphology_stats=morphology_stats,
        device=device,
        detailed=True,
    )
    gene_mse = np.asarray(evaluation.pop("gene_mse"), dtype=np.float64)
    gene_pearson = np.asarray(evaluation.pop("gene_pearson"), dtype=np.float64)
    component_predictions = list(evaluation.pop("component_mean_predictions"))
    jacobian_parts: dict[str, np.ndarray] | None = None
    jacobian_summary: dict[str, Any] | None = None
    if feature is not None:
        target_mean, target_std, _ = target_stats
        eval_positions = np.flatnonzero(mask)
        eval_index = torch.from_numpy(eval_positions).to(device)
        normalized_feature = (
            feature.index_select(0, eval_index) - target_mean
        ) / target_std
        jacobian_weights = component_equal_weights(groups, mask)[eval_positions].astype(
            np.float64, copy=False
        )
        jacobian_parts = model.mean_neighbor_jacobian_parts(
            normalized_feature,
            weights=torch.from_numpy(jacobian_weights).to(device),
            chunk_size=BATCH_SIZE,
        )
        jacobian_summary = diagonal_summary(
            jacobian_parts["total"], eligible_genes
        ).as_dict()
        del normalized_feature, eval_index
    return (
        evaluation,
        gene_mse,
        gene_pearson,
        component_predictions,
        jacobian_parts,
        jacobian_summary,
    )


def _fit_arm(
    arm: str,
    *,
    target: torch.Tensor,
    morphology: torch.Tensor,
    feature: torch.Tensor | None,
    masks: dict[str, np.ndarray],
    groups: np.ndarray,
    profile: str,
    fold: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray] | None]:
    use_neighbor = feature is not None
    seed = SEED_BASE + fold
    tuning_target_stats = _target_statistics(target, masks["tuning_train"], device)
    tuning_morph_stats = _morphology_statistics(
        morphology, masks["tuning_train"], device
    )
    tuning_model = _new_model(use_neighbor=use_neighbor, seed=seed, device=device)
    tuning_optimizer = torch.optim.AdamW(
        tuning_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    candidates = (
        PILOT_EPOCH_CANDIDATES if profile == "pilot" else EPOCH_CANDIDATES
    )
    validation_history: list[dict[str, float]] = []
    training_history: list[dict[str, float]] = []
    completed_epochs = 0
    for candidate in candidates:
        training_history.extend(
            _train_epochs(
                tuning_model,
                optimizer=tuning_optimizer,
                target=target,
                morphology=morphology,
                feature=feature,
                train_mask=masks["tuning_train"],
                groups=groups,
                target_stats=tuning_target_stats,
                morphology_stats=tuning_morph_stats,
                epochs=candidate - completed_epochs,
                start_epoch=completed_epochs,
                seed=seed,
                device=device,
            )
        )
        completed_epochs = candidate
        validation = _evaluate(
            tuning_model,
            target=target,
            morphology=morphology,
            feature=feature,
            mask=masks["validation"],
            groups=groups,
            target_stats=tuning_target_stats,
            morphology_stats=tuning_morph_stats,
            device=device,
            detailed=False,
        )
        validation_history.append(
            {
                "epoch": float(candidate),
                "component_equal_mse": float(validation["component_equal_mse"]),
            }
        )
    selected_epoch = int(
        min(
            validation_history,
            key=lambda item: (item["component_equal_mse"], item["epoch"]),
        )["epoch"]
    )
    del tuning_model, tuning_optimizer
    torch.cuda.empty_cache()

    fit_mask_name = "tuning_train" if profile == "pilot" else "final_train"
    evaluation_mask_name = "validation" if profile == "pilot" else "test"
    final_target_stats = _target_statistics(target, masks[fit_mask_name], device)
    final_morph_stats = _morphology_statistics(
        morphology, masks[fit_mask_name], device
    )
    final_model = _new_model(use_neighbor=use_neighbor, seed=seed, device=device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    refit_epoch = (
        int(PILOT_REFIT_EPOCH_OVERRIDE)
        if profile == "pilot" and PILOT_REFIT_EPOCH_OVERRIDE is not None
        else selected_epoch
    )
    if refit_epoch < 1:
        raise NonlinearRunError("refit epoch must be positive")
    refit_history: list[dict[str, float]] = []
    anchor_state_dict: dict[str, torch.Tensor] | None = None
    if ANCHOR_EPOCH is not None:
        anchor_epoch = int(ANCHOR_EPOCH)
        if anchor_epoch < 1 or anchor_epoch > refit_epoch:
            raise NonlinearRunError("anchor epoch must fall within the refit path")
        refit_history.extend(
            _train_epochs(
                final_model,
                optimizer=final_optimizer,
                target=target,
                morphology=morphology,
                feature=feature,
                train_mask=masks[fit_mask_name],
                groups=groups,
                target_stats=final_target_stats,
                morphology_stats=final_morph_stats,
                epochs=anchor_epoch,
                start_epoch=0,
                seed=seed + 1_000_000,
                device=device,
            )
        )
        anchor_state_dict = _cpu_state_dict(final_model)
        if refit_epoch > anchor_epoch:
            refit_history.extend(
                _train_epochs(
                    final_model,
                    optimizer=final_optimizer,
                    target=target,
                    morphology=morphology,
                    feature=feature,
                    train_mask=masks[fit_mask_name],
                    groups=groups,
                    target_stats=final_target_stats,
                    morphology_stats=final_morph_stats,
                    epochs=refit_epoch - anchor_epoch,
                    start_epoch=anchor_epoch,
                    seed=seed + 1_000_000,
                    device=device,
                )
            )
    else:
        refit_history.extend(
            _train_epochs(
                final_model,
                optimizer=final_optimizer,
                target=target,
                morphology=morphology,
                feature=feature,
                train_mask=masks[fit_mask_name],
                groups=groups,
                target_stats=final_target_stats,
                morphology_stats=final_morph_stats,
                epochs=refit_epoch,
                start_epoch=0,
                seed=seed + 1_000_000,
                device=device,
            )
        )
    target_mean, target_std, prevalence = final_target_stats
    eligible_genes = (prevalence >= 0.05) & (target_std.cpu().numpy() > 1e-6)
    (
        evaluation,
        gene_mse,
        gene_pearson,
        component_predictions,
        jacobian_parts,
        jacobian_summary,
    ) = _evaluate_with_jacobian(
        final_model,
        target=target,
        morphology=morphology,
        feature=feature,
        mask=masks[evaluation_mask_name],
        groups=groups,
        target_stats=final_target_stats,
        morphology_stats=final_morph_stats,
        eligible_genes=eligible_genes,
        device=device,
    )
    combined_jacobian_parts = (
        None if jacobian_parts is None else dict(jacobian_parts)
    )
    anchor_result: dict[str, Any] | None = None
    if anchor_state_dict is not None:
        assert ANCHOR_EPOCH is not None
        anchor_model = _new_model(use_neighbor=use_neighbor, seed=seed, device=device)
        anchor_model.load_state_dict(anchor_state_dict, strict=True)
        (
            anchor_evaluation,
            anchor_gene_mse,
            anchor_gene_pearson,
            anchor_component_predictions,
            anchor_jacobian_parts,
            anchor_jacobian_summary,
        ) = _evaluate_with_jacobian(
            anchor_model,
            target=target,
            morphology=morphology,
            feature=feature,
            mask=masks[evaluation_mask_name],
            groups=groups,
            target_stats=final_target_stats,
            morphology_stats=final_morph_stats,
            eligible_genes=eligible_genes,
            device=device,
        )
        if anchor_jacobian_parts is not None:
            if combined_jacobian_parts is None:
                combined_jacobian_parts = {}
            combined_jacobian_parts.update(
                {
                    f"anchor{int(ANCHOR_EPOCH)}_{name}": value
                    for name, value in anchor_jacobian_parts.items()
                }
            )
        anchor_result = {
            "epoch": int(ANCHOR_EPOCH),
            "evaluation": anchor_evaluation,
            "component_predictions": anchor_component_predictions,
            "gene_mse": anchor_gene_mse,
            "gene_pearson": anchor_gene_pearson,
            "jacobian_summary": anchor_jacobian_summary,
        }
        del anchor_model
    state = {
        "model_kwargs": {
            "gene_count": 1000,
            "morphology_count": 22,
            "hidden_count": HIDDEN_COUNT,
            "use_neighbor": use_neighbor,
        },
        "state_dict": _cpu_state_dict(final_model),
        "anchor_state_dict": anchor_state_dict,
        "anchor_epoch": ANCHOR_EPOCH,
        "selected_epoch": selected_epoch,
        "refit_epoch": refit_epoch,
        "target_mean": target_mean.detach().cpu(),
        "target_std": target_std.detach().cpu(),
        "morphology_median": final_morph_stats[0].detach().cpu(),
        "morphology_mean": final_morph_stats[1].detach().cpu(),
        "morphology_std": final_morph_stats[2].detach().cpu(),
        "fit_mask": fit_mask_name,
        "evaluation_mask": evaluation_mask_name,
    }
    result = {
        "arm": arm,
        "selected_epoch": selected_epoch,
        "refit_epoch": refit_epoch,
        "validation_history": validation_history,
        "tuning_training_history": training_history,
        "refit_training_history": refit_history,
        "evaluation_role": evaluation_mask_name,
        "evaluation": evaluation,
        "eligible_gene_count": int(np.sum(eligible_genes)),
        "jacobian_summary": jacobian_summary,
        "component_predictions": component_predictions,
        "parameter_count": sum(parameter.numel() for parameter in final_model.parameters()),
        "gene_mse": gene_mse,
        "gene_pearson": gene_pearson,
        "eligible_genes": eligible_genes,
        "anchor": anchor_result,
    }
    del final_model, final_optimizer
    torch.cuda.empty_cache()
    return result, state, combined_jacobian_parts


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


def _canonical_prediction_rows(
    run_id: str,
    fold: int,
    split: str,
    component_predictions: list[dict[str, Any]],
    component_losses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    loss_by_group = {
        int(item["geometry_group"]): float(item["mse"]) for item in component_losses
    }
    rows = []
    for item in component_predictions:
        group = int(item["geometry_group"])
        rows.append(
            {
                "run_id": run_id,
                "sample_key": f"geometry_group_{group:03d}",
                "dataset_id": DATASET_ID,
                "split": split,
                "y_true": item["y_true"],
                "y_pred": item["y_pred"],
                "graph_id": "within_fov_k12_0_25um",
                "fold": fold,
                "node_count": int(item["cell_count"]),
                "sample_loss": loss_by_group[group],
            }
        )
    return rows


def _write_canonical_contract(
    archive: RunArchive,
    *,
    configuration: dict[str, Any],
    provenance: dict[str, Any],
    result_payload: dict[str, Any],
) -> None:
    run_id = str(result_payload["run_id"])
    fold = int(result_payload["outer_fold"])
    observed = result_payload["arms"]["observed_near"]
    canonical_split = str(
        configuration["evaluation"]["canonical_prediction_split"]
    )
    expected_split = (
        "validation"
        if result_payload["statistical_evaluation_role"] == "resource_validation"
        else "test"
    )
    if canonical_split != expected_split:
        raise NonlinearRunError(
            "canonical prediction split disagrees with statistical evaluation role"
        )
    primary_metric = str(configuration["evaluation"]["primary_metric"])
    final_metrics: dict[str, float] = {
        primary_metric: float(observed["evaluation"]["component_equal_mse"])
    }
    for arm, result in result_payload["arms"].items():
        final_metrics[f"{canonical_split}/{arm}_component_equal_mse"] = float(
            result["evaluation"]["component_equal_mse"]
        )
    if "relative_mse_gain_vs_morphology" in observed["evaluation"]:
        final_metrics[f"{canonical_split}/near_gain_vs_morphology"] = float(
            observed["evaluation"]["relative_mse_gain_vs_morphology"]
        )
        final_metrics[f"{canonical_split}/near_gain_vs_permuted"] = float(
            observed["evaluation"]["relative_mse_gain_vs_permuted"]
        )
    archive.write_manifest(
        {
            "run_id": run_id,
            "status": "completed",
            "campaign_id": CAMPAIGN_ID,
            "profile": result_payload["profile"],
            "fold": fold,
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "statistical_evaluation_role": result_payload["statistical_evaluation_role"],
            "prediction_semantics": (
                "component mean standardized target/prediction over the evaluated "
                "matched-eligible population; sample_loss is component cellwise MSE"
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
            "primary_metric_name": primary_metric,
            "primary_metric_value": final_metrics[primary_metric],
            "duration_seconds": result_payload["duration_seconds"],
            "peak_vram_gb": result_payload["peak_vram_gb"],
            "parameter_count": result_payload["parameter_count"],
            "claim_boundary": result_payload["claim_boundary"],
        }
    )
    archive.write_json("metrics/final.json", final_metrics)
    for name, value in final_metrics.items():
        archive.append_metric_event(
            {"name": name, "value": value, "step": 0, "split": canonical_split}
        )
    archive.write_text(
        "metrics/history.jsonl",
        "\n".join(
            canonical_json({"arm": arm, **row})
            for arm, result in result_payload["arms"].items()
            for row in result["validation_history"]
        )
        + "\n",
    )
    prediction_rows = _canonical_prediction_rows(
        run_id,
        fold,
        canonical_split,
        observed["component_predictions"],
        observed["evaluation"]["per_component"],
    )
    if not np.isclose(
        np.mean([row["sample_loss"] for row in prediction_rows]),
        final_metrics[primary_metric],
        rtol=1e-12,
        atol=0.0,
    ):
        raise NonlinearRunError("canonical predictions do not reproduce primary MSE")
    archive.write_prediction_jsonl_stream(canonical_split, prediction_rows)
    archive.copy_file(
        archive.scratch_path / "nonlinear_checkpoint.pt", "checkpoints/last.ckpt"
    )
    archive.prepare_log_files()
    archive.write_json(
        "provenance/git.json",
        {
            "commit": provenance["git_commit"],
            "dirty_fingerprint": provenance["dirty_fingerprint"],
            "git_status": provenance["git_status"],
            "tracked_diff_sha256": provenance["tracked_diff_sha256"],
            "untracked_files": provenance["untracked_files"],
            "source_snapshot_manifest": provenance["source_snapshot_manifest"],
        },
    )
    archive.write_text(
        "provenance/uncommitted_changes.patch", provenance["tracked_diff"]
    )
    hardware = provenance["hardware"]
    archive.write_json("provenance/hardware.json", hardware)
    archive.write_text(
        "provenance/environment.txt",
        "\n".join(f"{key}={hardware[key]}" for key in sorted(hardware)) + "\n",
    )
    archive.write_json(
        "provenance/data_fingerprints.json",
        {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "raw": RAW_FINGERPRINT,
            "processed": PROCESSED_FINGERPRINT,
        },
    )
    archive.write_json(
        "provenance/split_fingerprint.json",
        {"split_id": SPLIT_ID, "sha256": SPLIT_FINGERPRINT},
    )
    archive.write_text(
        "provenance/command.txt",
        " ".join(str(value) for value in provenance["command"]) + "\n",
    )
    project_root = Path(provenance["working_directory"])
    for item in provenance["source_snapshot_manifest"]:
        relative = str(item["path"])
        source = project_root / relative
        if (
            source.stat().st_size != int(item["size_bytes"])
            or sha256_file(source) != str(item["sha256"])
        ):
            raise NonlinearRunError(f"source changed during run: {relative}")
        archive.copy_file(source, Path("provenance/source_snapshot") / relative)


def _configuration(*, profile: str, fold: int, attempt: int) -> dict[str, Any]:
    if profile not in {"pilot", "full"}:
        raise ValueError(f"unsupported profile: {profile}")
    return {
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "experiment_flavor": EXPERIMENT_FLAVOR,
        },
        "profile": profile,
        "dataset": {
            "dataset_id": DATASET_ID,
            "version": DATASET_VERSION,
            "dataset_version": DATASET_VERSION,
            "dataset_fingerprint": RAW_FINGERPRINT,
            "processed_fingerprint": PROCESSED_FINGERPRINT,
            "split_id": SPLIT_ID,
            "split_fingerprint": SPLIT_FINGERPRINT,
            "preprocessing_version": PREPROCESSING_VERSION,
        },
        "model": {
            "family": "additive_neighbor_mlp",
            "embedding_dim": HIDDEN_COUNT,
            "gene_count": 1000,
            "morphology_count": 22,
            "hidden_count": HIDDEN_COUNT,
            "full_linear_neighbor_skip": True,
            "activation": "exact_gelu",
        },
        "graph": {
            "partition": "fov",
            "k": 12,
            "neighbor_k": 12,
            "near_um": [0, 25],
            "annular_um": [25, 50],
            "self_edges": False,
        },
        "features": {"use_edge_features": False},
        "masking": {"type": "whole_node", "receiver_expression_input": False},
        "trainer": {
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "learning_rate_schedule": "constant",
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epoch_candidates": list(EPOCH_CANDIDATES),
            "effective_epoch_candidates": list(
                PILOT_EPOCH_CANDIDATES if profile == "pilot" else EPOCH_CANDIDATES
            ),
            "anchor_epoch": ANCHOR_EPOCH,
            "pilot_refit_epoch_override": (
                PILOT_REFIT_EPOCH_OVERRIDE if profile == "pilot" else None
            ),
            "effective_arms": list(PILOT_ARMS if profile == "pilot" else FULL_ARMS),
            "primary_checkpoint_role": "last",
            "restore_best": False,
            "selection": "validation_epoch_count_then_fresh_final_refit",
            "seed_base": SEED_BASE,
            "model_seed": SEED_BASE + fold,
        },
        "evaluation": {
            "fold": fold,
            "component_equal": True,
            "canonical_prediction_split": (
                "validation" if profile == "pilot" else "test"
            ),
            "protocol": (
                "resource_validation"
                if profile == "pilot"
                else "held_out_geometry_masked_reconstruction"
            ),
            "statistical_partition": (
                "resource_validation" if profile == "pilot" else "outer_geometry_test"
            ),
            "primary_metric": (
                f"{'validation' if profile == 'pilot' else 'test'}/"
                f"{PRIMARY_METRIC_SUFFIX}"
            ),
        },
        "classification": {
            "lifecycle_stage": (
                "diagnostic" if profile == "pilot" else "exploratory_screen"
            ),
            "study_axis": "interpretation",
            "variant_label": f"same_gene_nonlinear_{EXPERIMENT_FLAVOR}_{profile}",
            "experiment_flavor": EXPERIMENT_FLAVOR,
            "seed": TRACKING_SEED,
            "execution_seed": SEED_BASE + fold,
            "fold": fold,
            "attempt": attempt,
        },
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise NonlinearRunError(
            "each fold process requires exactly one visible CUDA device"
        )
    if arguments.fold not in range(4):
        raise ValueError("fold must be 0, 1, 2, or 3")
    paths = current_paths()
    contract = (
        paths.project_root
        / "experiments/campaigns"
        / CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    if sha256_file(contract) != FROZEN_CONTRACT_SHA256:
        raise NonlinearRunError("frozen task contract identity changed")
    prepared = arguments.prepared.resolve(strict=True)
    prepared_manifest = json.loads(
        (prepared / "manifest.json").read_text(encoding="utf-8")
    )
    observed_fingerprints = (
        prepared_manifest["raw_snapshot"]["fingerprint"],
        prepared_manifest["processed_fingerprint"],
        prepared_manifest["split_fingerprint"],
    )
    if observed_fingerprints != (
        RAW_FINGERPRINT,
        PROCESSED_FINGERPRINT,
        SPLIT_FINGERPRINT,
    ):
        raise NonlinearRunError("prepared data fingerprints changed")
    control = planted_nonlinear_jacobian_control()
    if control["passed"] is not True:
        raise NonlinearRunError(f"nonlinear analytical control failed: {control}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_float32_matmul_precision("highest")
    configuration = _configuration(
        profile=arguments.profile,
        fold=arguments.fold,
        attempt=arguments.attempt,
    )
    primary_metric = str(configuration["evaluation"]["primary_metric"])
    provenance = _provenance(paths.project_root, configuration)
    scientific_identifier = scientific_id(configuration)
    reproduction_identifier = repro_id(
        configuration,
        git_commit=provenance["git_commit"],
        dirty_fingerprint=provenance["dirty_fingerprint"],
        dataset_fingerprint=RAW_FINGERPRINT,
        split_fingerprint=SPLIT_FINGERPRINT,
        preprocessing_version=PREPROCESSING_VERSION,
        environment_fingerprint=provenance["environment_fingerprint"],
    )
    run_id = create_run_id(
        seed=TRACKING_SEED,
        fold=arguments.fold,
        attempt=arguments.attempt,
        scientific_id_value=scientific_identifier,
    )
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    registry.create_campaign(
        CAMPAIGN_ID,
        name=CAMPAIGN_DISPLAY_NAME,
        scientific_question=CAMPAIGN_SCIENTIFIC_QUESTION,
        config={
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "exploratory": True,
        },
        status="pilot",
    )
    archive = RunArchive.create(run_id, paths=paths)
    scratch = archive.scratch_path
    final = archive.artifact_path
    registry.register_variant(
        scientific_identifier,
        campaign_id=CAMPAIGN_ID,
        configuration=configuration,
        model_family="additive_neighbor_mlp",
        masking_type="whole_node",
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        split_id=SPLIT_ID,
        use_edge_features=False,
        embedding_dim=HIDDEN_COUNT,
        neighbor_k=12,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
    )
    registry.create_run(
        run_id,
        campaign_id=CAMPAIGN_ID,
        scientific_id=scientific_identifier,
        repro_id=reproduction_identifier,
        seed=TRACKING_SEED,
        fold=arguments.fold,
        attempt=arguments.attempt,
        configuration=configuration,
        status="pending",
        artifact_path=final,
        git_commit=provenance["git_commit"],
        dirty_status=provenance["dirty_fingerprint"] is not None,
        preprocessing_version=PREPROCESSING_VERSION,
        dataset_fingerprint=RAW_FINGERPRINT,
        split_fingerprint=SPLIT_FINGERPRINT,
        host=socket.gethostname(),
        gpu_model=provenance["hardware"]["gpu_name"],
    )
    categories = {
        "lifecycle_stage": (
            "diagnostic" if arguments.profile == "pilot" else "exploratory-screen"
        ),
        "study_axis": "interpretation",
        "model": "additive-neighbor-mlp",
        "graph": "near-annular-null",
        "mask": "whole-node",
        "edge_feature_state": False,
        "embedding_dim": HIDDEN_COUNT,
        "seed": TRACKING_SEED,
        "fold": arguments.fold,
        "attempt": arguments.attempt,
        "variant_evidence": {
            "profile": arguments.profile,
            "experiment_flavor": EXPERIMENT_FLAVOR,
        },
    }
    alias = semantic_run_alias(
        primary_run_id=run_id, categories=categories, historical=False
    )
    registry.register_run_semantics(
        run_id,
        lifecycle_stage=categories["lifecycle_stage"],
        study_axis="interpretation",
        source_batch=f"same_gene_nonlinear_{EXPERIMENT_FLAVOR}_20260810",
        seed_known=True,
        fold_known=True,
        attempt_known=True,
        retention_class=(
            "diagnostic" if arguments.profile == "pilot" else "exploratory_screen"
        ),
        category_key=(
            f"same_gene_nonlinear.{EXPERIMENT_FLAVOR}."
            f"{arguments.profile}.fold{arguments.fold}"
        ),
        classification_confidence="high",
        timestamp_basis="execution_start",
        variant_label=(
            f"same_gene_nonlinear_{EXPERIMENT_FLAVOR}_{arguments.profile}"
        ),
        model_key="additive_neighbor_mlp",
        dataset_key=DATASET_ID,
        masking_key="whole_node",
        graph_key="near_annular_null",
        feature_key="neighbor_expression_plus_morphology",
        embedding_key="h64_g1000",
        semantic_alias=alias,
        preferred_alias_type="semantic",
    )
    registry.transition_run(run_id, "running", start_time=utc_now())

    started = time.perf_counter()
    try:
        folds = _load_cpu_vector(prepared, "fold.npy").astype(np.int8, copy=False)
        groups = _load_cpu_vector(prepared, "geometry_group.npy").astype(
            np.int16, copy=False
        )
        eligible = _load_cpu_vector(prepared, "matched_eligible.npy").astype(
            bool, copy=False
        )
        masks = _split_masks(
            folds, groups, eligible, arguments.fold, profile=arguments.profile
        )
        target = _load_gpu_matrix(prepared, "expression_log1p.npy", device)
        morphology = _load_gpu_matrix(prepared, "metadata.npy", device)
        arms = PILOT_ARMS if arguments.profile == "pilot" else FULL_ARMS
        results: dict[str, Any] = {}
        checkpoint_arms: dict[str, Any] = {}
        matrices: dict[str, np.ndarray] = {}
        for arm in arms:
            feature = (
                None
                if arm == "morphology_only"
                else _load_gpu_matrix(prepared, FEATURE_FILE[arm], device)
            )
            result, state, jacobian_parts = _fit_arm(
                arm,
                target=target,
                morphology=morphology,
                feature=feature,
                masks=masks,
                groups=groups,
                profile=arguments.profile,
                fold=arguments.fold,
                device=device,
            )
            checkpoint_arms[arm] = state
            if jacobian_parts is not None:
                for name, value in jacobian_parts.items():
                    matrices[f"{arm}_{name}"] = np.asarray(value)
            result["gene_mse"] = result["gene_mse"].tolist()
            result["gene_pearson"] = result["gene_pearson"].tolist()
            result["eligible_genes"] = result["eligible_genes"].tolist()
            if result["anchor"] is not None:
                result["anchor"]["gene_mse"] = result["anchor"][
                    "gene_mse"
                ].tolist()
                result["anchor"]["gene_pearson"] = result["anchor"][
                    "gene_pearson"
                ].tolist()
            results[arm] = result
            if feature is not None:
                del feature
            torch.cuda.empty_cache()

        if arguments.profile == "full":
            morphology_mse = results["morphology_only"]["evaluation"][
                "component_equal_mse"
            ]
            permuted_mse = results["within_fov_permuted_near"]["evaluation"][
                "component_equal_mse"
            ]
            near_mse = results["observed_near"]["evaluation"][
                "component_equal_mse"
            ]
            results["observed_near"]["evaluation"][
                "relative_mse_gain_vs_morphology"
            ] = (morphology_mse - near_mse) / morphology_mse
            results["observed_near"]["evaluation"][
                "relative_mse_gain_vs_permuted"
            ] = (permuted_mse - near_mse) / permuted_mse
            results["observed_annular"]["evaluation"][
                "relative_mse_gain_vs_morphology"
            ] = (
                morphology_mse
                - results["observed_annular"]["evaluation"]["component_equal_mse"]
            ) / morphology_mse
            if all(results[arm]["anchor"] is not None for arm in FULL_ARMS):
                anchor_morphology_mse = results["morphology_only"]["anchor"][
                    "evaluation"
                ]["component_equal_mse"]
                anchor_permuted_mse = results["within_fov_permuted_near"][
                    "anchor"
                ]["evaluation"]["component_equal_mse"]
                anchor_near_mse = results["observed_near"]["anchor"][
                    "evaluation"
                ]["component_equal_mse"]
                results["observed_near"]["anchor"]["evaluation"][
                    "relative_mse_gain_vs_morphology"
                ] = (anchor_morphology_mse - anchor_near_mse) / anchor_morphology_mse
                results["observed_near"]["anchor"]["evaluation"][
                    "relative_mse_gain_vs_permuted"
                ] = (anchor_permuted_mse - anchor_near_mse) / anchor_permuted_mse
        else:
            near_mse = results["observed_near"]["evaluation"][
                "component_equal_mse"
            ]

        duration = time.perf_counter() - started
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        parameter_count = sum(
            int(result["parameter_count"]) for result in results.values()
        )
        if arguments.profile == "pilot":
            validation_fold = (arguments.fold + 1) % 4
            full_tuning_cells = int(
                np.sum(
                    (folds != arguments.fold)
                    & (folds != validation_fold)
                    & eligible
                )
            )
            full_final_cells = int(
                np.sum((folds != arguments.fold) & eligible)
            )
            pilot_tuning_cells = int(np.sum(masks["tuning_train"]))
            pilot_refit_cells = pilot_tuning_cells
            pilot_work = (
                len(arms)
                * pilot_tuning_cells
                * max(PILOT_EPOCH_CANDIDATES)
                + pilot_refit_cells
                * sum(int(result["refit_epoch"]) for result in results.values())
            )
            full_work = (
                len(FULL_ARMS)
                * (full_tuning_cells + full_final_cells)
                * max(EPOCH_CANDIDATES)
            )
            projected_hours = duration * full_work / max(1, pilot_work) / 3600
        else:
            projected_hours = duration / 3600
        oracle = diagonal_summary(
            np.eye(1000, dtype=np.float64), np.ones(1000, dtype=bool)
        ).as_dict()
        all_outputs_finite = _all_numeric_values_finite(results) and all(
            bool(np.isfinite(value).all()) for value in matrices.values()
        )
        controls = {
            "analytical_nonlinear_jacobian": control,
            "receiver_expression_input": False,
            "identity_oracle": oracle,
            "identity_oracle_row_top1_fraction": oracle["row_top1_fraction"],
            "all_outputs_finite": all_outputs_finite,
            "peak_vram_gb": peak_vram_gb,
            "peak_vram_gate_passed": peak_vram_gb <= 20.5,
            "projected_full_hours_per_fold": projected_hours,
            "runtime_advisory_under_two_hours": projected_hours <= 2.0,
        }
        if not controls["peak_vram_gate_passed"]:
            raise NonlinearRunError(f"resource gate failed: {controls}")
        if not controls["all_outputs_finite"]:
            raise NonlinearRunError("result finite-value control failed")
        if controls["identity_oracle_row_top1_fraction"] != 1.0:
            raise NonlinearRunError(f"identity-oracle control failed: {oracle}")
        if (
            arguments.profile == "pilot"
            and not controls["runtime_advisory_under_two_hours"]
        ):
            raise NonlinearRunError(f"pilot projected-runtime gate failed: {controls}")
        result_payload = {
            "run_id": run_id,
            "semantic_alias": alias,
            "campaign_id": CAMPAIGN_ID,
            "profile": arguments.profile,
            "outer_fold": arguments.fold,
            "status": "completed",
            "duration_seconds": duration,
            "peak_vram_gb": peak_vram_gb,
            "parameter_count": parameter_count,
            "statistical_evaluation_role": (
                "resource_validation" if arguments.profile == "pilot" else "outer_geometry_test"
            ),
            "sample_counts": {name: int(np.sum(mask)) for name, mask in masks.items()},
            "controls": controls,
            "arms": results,
            "dataset_fingerprints": {
                "raw": RAW_FINGERPRINT,
                "processed": PROCESSED_FINGERPRINT,
                "split": SPLIT_FINGERPRINT,
            },
            "claim_boundary": (
                "exploratory held-out-geometry nonlinear model sensitivity; not "
                "communication, mechanism, causality, or patient generalization"
            ),
        }
        checkpoint = {
            "format_version": 1,
            "run_id": run_id,
            "campaign_id": CAMPAIGN_ID,
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "configuration": configuration,
            "genes": json.loads((prepared / "genes.json").read_text(encoding="utf-8")),
            "arms": checkpoint_arms,
        }
        torch.save(checkpoint, scratch / "nonlinear_checkpoint.pt")
        np.savez_compressed(
            scratch / "nonlinear_jacobians.npz",
            genes=np.asarray(checkpoint["genes"]),
            **{name: value for name, value in matrices.items()},
            **{
                f"eligible_{arm}": np.asarray(result["eligible_genes"], dtype=bool)
                for arm, result in results.items()
            },
        )
        _json_write(scratch / "resolved_configuration.json", configuration)
        _json_write(scratch / "provenance.json", provenance)
        _json_write(scratch / "results.json", result_payload)
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
        )
        archive.publish_success_pending()
        verify_unmarked_run_bundle(final)
        registry.transition_run(
            run_id,
            "finalizing",
            end_time=utc_now(),
            duration_seconds=duration,
            peak_vram_gb=peak_vram_gb,
            parameter_count=parameter_count,
            primary_metric_name=primary_metric,
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
            artifact_ids["checkpoints/last.ckpt"],
            run_id=run_id,
            role="last",
            retention_class=(
                "diagnostic" if arguments.profile == "pilot" else "exploratory_screen"
            ),
            verification_status="verified",
            monitored_metric=primary_metric,
            monitored_mode="min",
            monitored_value=near_mse,
            metadata={
                "prediction_replayable": True,
                "selected_epoch_then_fresh_refit": arguments.profile == "full",
                "forced_resource_refit_epoch": (
                    PILOT_REFIT_EPOCH_OVERRIDE
                    if arguments.profile == "pilot"
                    else None
                ),
                "arms": list(arms),
            },
        )
        for arm, result in results.items():
            registry.record_metric(
                run_id,
                f"{configuration['evaluation']['canonical_prediction_split']}/"
                f"{arm}_component_equal_mse",
                result["evaluation"]["component_equal_mse"],
                split=configuration["evaluation"]["canonical_prediction_split"],
            )
        archive.mark_success()
        verify_run_bundle(final)
        registry.transition_run(run_id, "completed")
        issues = registry.verify_artifacts(run_id=run_id)
        if issues:
            raise NonlinearRunError(f"registered artifact verification failed: {issues}")
        return {
            "run_id": run_id,
            "profile": arguments.profile,
            "fold": arguments.fold,
            "artifact_path": str(final),
            "duration_seconds": duration,
            "peak_vram_gb": peak_vram_gb,
            "projected_full_hours_per_fold": projected_hours,
            "observed_near_mse": near_mse,
            "observed_near_jacobian_summary": results["observed_near"][
                "jacobian_summary"
            ],
        }
    except BaseException as error:
        registry.record_failure(
            run_id=run_id,
            category="same_gene_nonlinear_run_failure",
            message=str(error),
            details={"type": type(error).__name__, "profile": arguments.profile},
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
                failure_category="same_gene_nonlinear_run_failure",
            )
            try:
                if final.is_dir() and not any(
                    (final / marker).exists()
                    for marker in ("_SUCCESS", "_FAILED", "_PRUNED")
                ):
                    RunArchive.from_published(run_id, paths=paths).mark_published_failure()
                elif archive.scratch_path.is_dir():
                    archive.finalize_failure(
                        error, failure_category="same_gene_nonlinear_run_failure"
                    )
            except BaseException as compensation_error:
                registry.record_failure(
                    run_id=run_id,
                    category="same_gene_nonlinear_failure_compensation",
                    message=str(compensation_error),
                    details={"type": type(compensation_error).__name__},
                )
        raise


def parse_args() -> argparse.Namespace:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--profile", choices=("pilot", "full"), default="full")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument(
        "--prepared",
        type=Path,
        default=paths.data_root / "processed/same_gene_cross_cell_jacobian_v1",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.attempt < 1:
        raise ValueError("attempt must be positive")
    print(canonical_json(run(arguments)), flush=True)


if __name__ == "__main__":
    main()
