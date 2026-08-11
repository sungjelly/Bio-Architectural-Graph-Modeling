#!/usr/bin/env python3
"""Train one fold of the frozen nonlinear same-gene replication campaign."""

from __future__ import annotations

import argparse
import hashlib
import io
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = str(PROJECT_ROOT / "src")
if not sys.path or sys.path[0] != _SOURCE_ROOT:
    sys.path.insert(0, _SOURCE_ROOT)

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
ELIGIBILITY_FILE = "matched_eligible.npy"
FROZEN_GENE_ELIGIBILITY_FILE: str | None = None
FROZEN_GENE_ELIGIBILITY_SHA256: str | None = None
EPOCH_CANDIDATES = (1, 2, 4, 8, 12)
PILOT_EPOCH_CANDIDATES = (1, 2)
PILOT_ARMS = ("observed_near",)
PILOT_REFIT_EPOCH_OVERRIDE: int | None = None
ANCHOR_EPOCH: int | None = None
MAXIMUM_PROJECTED_HOURS_PER_FOLD = 2.0
MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR = 1e-7
CANONICAL_PRODUCTION_SPLIT_LABEL = "test"
# These authorities are set by wrappers only after their source/config/data and
# prepared-graph verification has completed.  A pilot refuses to authorize
# production while either remains false.
SOURCE_CONFIG_DATA_HASHES_VERIFIED = False
GRAPH_SPECIFIC_INVARIANTS_VERIFIED = False
ENVIRONMENT_LOCK_REQUIRED = False
ENVIRONMENT_LOCK_REPORT: Mapping[str, Any] | None = None
# Optional campaign wrapper hook.  It may construct train-population-fitted
# target/source transforms, but must return both tuning and final phase tensors
# explicitly so validation preprocessing cannot borrow final-train outcomes.
PHASE_INPUT_BUILDER: Any = None
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


class AttemptConsumedError(NonlinearRunError):
    """Raised when an immutable attempt exists but is not recoverable success."""


def _process_start_ticks(pid: int) -> int | None:
    try:
        content = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        tail = content[content.rindex(")") + 2 :].split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


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
    if ENVIRONMENT_LOCK_REPORT is not None:
        hardware["frozen_environment_verification"] = dict(
            ENVIRONMENT_LOCK_REPORT
        )
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


def _load_frozen_gene_eligibility(prepared: Path) -> np.ndarray | None:
    if FROZEN_GENE_ELIGIBILITY_FILE is None:
        if FROZEN_GENE_ELIGIBILITY_SHA256 is not None:
            raise NonlinearRunError(
                "frozen gene-eligibility SHA requires a configured file"
            )
        return None
    filename = str(FROZEN_GENE_ELIGIBILITY_FILE)
    if Path(filename).name != filename or filename in {".", ".."}:
        raise NonlinearRunError("frozen gene-eligibility filename is unsafe")
    path = prepared / filename
    value = np.load(path, allow_pickle=False)
    if value.shape != (1000,) or value.dtype != np.bool_:
        raise NonlinearRunError(
            "frozen gene eligibility must be a boolean vector of length 1000"
        )
    contiguous = np.ascontiguousarray(value, dtype=np.uint8)
    observed = hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()
    if FROZEN_GENE_ELIGIBILITY_SHA256 is None or (
        observed != FROZEN_GENE_ELIGIBILITY_SHA256
    ):
        raise NonlinearRunError("frozen gene-eligibility SHA-256 mismatch")
    if int(value.sum()) < 1:
        raise NonlinearRunError("frozen gene eligibility is empty")
    return value.copy()


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


def _split_component_overlap_control(
    masks: Mapping[str, np.ndarray], groups: np.ndarray
) -> dict[str, Any]:
    """Report the disjoint scientific partitions without conflating refit nesting."""

    required = {"tuning_train", "validation", "final_train", "test"}
    if set(masks) != required:
        raise NonlinearRunError("split control received an unexpected mask inventory")
    component_sets = {
        name: set(int(value) for value in np.unique(groups[mask]))
        for name, mask in masks.items()
    }
    checked_pairs = (
        ("tuning_train", "validation"),
        ("tuning_train", "test"),
        ("validation", "test"),
        ("final_train", "test"),
    )
    cell_overlap = {
        f"{first}__{second}": int(np.sum(masks[first] & masks[second]))
        for first, second in checked_pairs
    }
    component_overlap = {
        f"{first}__{second}": sorted(component_sets[first] & component_sets[second])
        for first, second in checked_pairs
    }
    passed = not any(cell_overlap.values()) and not any(component_overlap.values())
    result = {
        "checked_pairs": [list(pair) for pair in checked_pairs],
        "cell_overlap_counts": cell_overlap,
        "component_overlap_ids": component_overlap,
        "passed": bool(passed),
    }
    if not passed:
        raise NonlinearRunError(f"split overlap control failed: {result}")
    return result


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


@torch.no_grad()
def _checkpoint_replay_control(
    model: AdditiveNeighborMLP,
    *,
    state: Mapping[str, Any],
    target: torch.Tensor,
    morphology: torch.Tensor,
    feature: torch.Tensor | None,
    mask: np.ndarray,
    groups: np.ndarray,
    target_stats: tuple[torch.Tensor, torch.Tensor, np.ndarray],
    morphology_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """Round-trip a checkpoint and replay its GPU forward path.

    The control compares both raw predictions and the component-equal MSE
    computed from the live model against a model reconstructed solely from a
    ``torch.save``/``torch.load`` byte stream.  It intentionally does not touch
    any mask other than the arm's already-authorized evaluation mask.
    """

    model_kwargs = state.get("model_kwargs")
    state_dict = state.get("state_dict")
    if not isinstance(model_kwargs, Mapping) or not isinstance(state_dict, Mapping):
        raise NonlinearRunError("checkpoint replay state is malformed")
    buffer = io.BytesIO()
    torch.save(
        {"model_kwargs": dict(model_kwargs), "state_dict": dict(state_dict)},
        buffer,
    )
    buffer.seek(0)
    serialized = torch.load(buffer, map_location="cpu", weights_only=False)
    if not isinstance(serialized, Mapping) or set(serialized) != {
        "model_kwargs",
        "state_dict",
    }:
        raise NonlinearRunError("checkpoint replay serialization is malformed")
    replay = AdditiveNeighborMLP(**dict(serialized["model_kwargs"])).to(device)
    replay.load_state_dict(serialized["state_dict"], strict=True)
    model.eval()
    replay.eval()

    positions = np.flatnonzero(mask)
    component_values = sorted(int(value) for value in np.unique(groups[mask]))
    lookup = {value: index for index, value in enumerate(component_values)}
    counts = np.zeros(len(component_values), dtype=np.int64)
    live_squared_error = np.zeros(len(component_values), dtype=np.float64)
    replay_squared_error = np.zeros(len(component_values), dtype=np.float64)
    maximum_prediction_error = 0.0
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
        live_prediction = model(morph, neighbor)
        replay_prediction = replay(morph, neighbor)
        maximum_prediction_error = max(
            maximum_prediction_error,
            float(
                torch.max(torch.abs(live_prediction - replay_prediction)).item()
            ),
        )
        live_row_mse = (
            (live_prediction - response).square().mean(dim=1).double().cpu().numpy()
        )
        replay_row_mse = (
            (replay_prediction - response)
            .square()
            .mean(dim=1)
            .double()
            .cpu()
            .numpy()
        )
        for group in np.unique(groups[cpu_index]):
            local = groups[cpu_index] == group
            slot = lookup[int(group)]
            counts[slot] += int(np.sum(local))
            live_squared_error[slot] += float(np.sum(live_row_mse[local]))
            replay_squared_error[slot] += float(np.sum(replay_row_mse[local]))
    if np.any(counts == 0):
        raise NonlinearRunError("checkpoint replay component has zero cells")
    live_component_mse = live_squared_error / counts
    replay_component_mse = replay_squared_error / counts
    maximum_metric_error = max(
        float(np.max(np.abs(live_component_mse - replay_component_mse))),
        abs(
            float(np.mean(live_component_mse))
            - float(np.mean(replay_component_mse))
        ),
    )
    passed = (
        np.isfinite(maximum_prediction_error)
        and np.isfinite(maximum_metric_error)
        and maximum_prediction_error <= MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR
        and maximum_metric_error <= MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR
    )
    result = {
        "serialization": "torch_save_load_bytes",
        "replay_device_type": device.type,
        "maximum_prediction_abs_error": maximum_prediction_error,
        "maximum_metric_abs_error": maximum_metric_error,
        "tolerance": MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR,
        "passed": bool(passed),
    }
    del replay, serialized, buffer
    if not passed:
        raise NonlinearRunError(f"checkpoint GPU replay control failed: {result}")
    return result


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
        component_groups = np.asarray(
            sorted(int(value) for value in np.unique(groups[mask])), dtype=np.int16
        )
        component_hidden_derivatives = np.stack(
            [
                model.mean_hidden_derivative(
                    normalized_feature[
                        torch.from_numpy(
                            np.flatnonzero(groups[eval_positions] == group)
                        ).to(device)
                    ],
                    chunk_size=BATCH_SIZE,
                )
                for group in component_groups
            ],
            axis=0,
        )
        jacobian_parts["component_geometry_groups"] = component_groups
        jacobian_parts[
            "component_mean_hidden_derivative"
        ] = component_hidden_derivatives
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
    phase_inputs: Mapping[str, torch.Tensor | None] | None = None,
    frozen_gene_eligibility: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray] | None]:
    if phase_inputs is None:
        tuning_target = target
        final_target = target
        tuning_feature = feature
        final_feature = feature
    else:
        expected = {
            "tuning_target",
            "final_target",
            "tuning_feature",
            "final_feature",
        }
        if set(phase_inputs) != expected:
            raise NonlinearRunError(
                f"phase inputs require exactly {sorted(expected)}"
            )
        tuning_target = phase_inputs["tuning_target"]
        final_target = phase_inputs["final_target"]
        tuning_feature = phase_inputs["tuning_feature"]
        final_feature = phase_inputs["final_feature"]
        if not isinstance(tuning_target, torch.Tensor) or not isinstance(
            final_target, torch.Tensor
        ):
            raise NonlinearRunError("phase targets must be tensors")
    use_neighbor = tuning_feature is not None
    if (final_feature is not None) != use_neighbor:
        raise NonlinearRunError(
            "tuning and final phase neighbor-feature presence must agree"
        )
    if tuning_target.shape != target.shape or final_target.shape != target.shape:
        raise NonlinearRunError("phase targets must preserve the target shape")
    for name, value in (
        ("tuning_target", tuning_target),
        ("final_target", final_target),
        ("tuning_feature", tuning_feature),
        ("final_feature", final_feature),
    ):
        if value is not None and not bool(torch.isfinite(value).all().item()):
            raise NonlinearRunError(f"{name} contains nonfinite values")
    seed = SEED_BASE + fold
    tuning_target_stats = _target_statistics(
        tuning_target, masks["tuning_train"], device
    )
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
                target=tuning_target,
                morphology=morphology,
                feature=tuning_feature,
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
            target=tuning_target,
            morphology=morphology,
            feature=tuning_feature,
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
    final_target_stats = _target_statistics(
        final_target, masks[fit_mask_name], device
    )
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
                target=final_target,
                morphology=morphology,
                feature=final_feature,
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
                    target=final_target,
                    morphology=morphology,
                    feature=final_feature,
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
                target=final_target,
                morphology=morphology,
                feature=final_feature,
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
    raw_eligible_genes = (prevalence >= 0.05) & (
        target_std.cpu().numpy() > 1e-6
    )
    if frozen_gene_eligibility is None:
        eligible_genes = raw_eligible_genes
        eligibility_mode = "fit_population_derived"
    else:
        frozen = np.asarray(frozen_gene_eligibility)
        if frozen.shape != raw_eligible_genes.shape or frozen.dtype != np.bool_:
            raise NonlinearRunError("frozen gene-eligibility mask is malformed")
        if np.any(frozen & ~raw_eligible_genes):
            missing = np.flatnonzero(frozen & ~raw_eligible_genes)
            raise NonlinearRunError(
                "frozen genes fail this fit population's numerical eligibility: "
                f"{missing[:20].tolist()}"
            )
        eligible_genes = frozen.copy()
        eligibility_mode = "frozen_common_mask"
    (
        evaluation,
        gene_mse,
        gene_pearson,
        component_predictions,
        jacobian_parts,
        jacobian_summary,
    ) = _evaluate_with_jacobian(
        final_model,
        target=final_target,
        morphology=morphology,
        feature=final_feature,
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
            target=final_target,
            morphology=morphology,
            feature=final_feature,
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
            "gene_count": int(final_model.gene_count),
            "morphology_count": int(final_model.morphology_count),
            "hidden_count": int(final_model.hidden_count),
            "use_neighbor": bool(final_model.use_neighbor),
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
        "eligibility_mode": eligibility_mode,
        "raw_eligible_genes": torch.from_numpy(raw_eligible_genes.copy()),
        "eligible_genes": torch.from_numpy(eligible_genes.copy()),
    }
    checkpoint_replay = _checkpoint_replay_control(
        final_model,
        state=state,
        target=final_target,
        morphology=morphology,
        feature=final_feature,
        mask=masks[evaluation_mask_name],
        groups=groups,
        target_stats=final_target_stats,
        morphology_stats=final_morph_stats,
        device=device,
    )
    state["checkpoint_replay"] = checkpoint_replay
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
        "raw_eligible_genes": raw_eligible_genes,
        "eligibility_mode": eligibility_mode,
        "checkpoint_replay": checkpoint_replay,
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
        "seed": TRACKING_SEED,
        "fold": fold,
        "attempt": attempt,
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
            "maximum_projected_hours_per_fold": (
                MAXIMUM_PROJECTED_HOURS_PER_FOLD
            ),
            "effective_arms": list(PILOT_ARMS if profile == "pilot" else FULL_ARMS),
            "primary_checkpoint_role": "last",
            "restore_best": False,
            "selection": "validation_epoch_count_then_fresh_final_refit",
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


def _attempt_rows(
    registry: Registry,
    *,
    configuration: Mapping[str, Any],
    fold: int,
    attempt: int,
) -> list[dict[str, Any]]:
    """Return the complete registry history for one declared execution slot."""

    identifier = scientific_id(configuration)
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT run_id FROM runs
            WHERE campaign_id = ? AND scientific_id = ?
              AND seed = ? AND fold = ? AND attempt = ?
            ORDER BY created_at, run_id
            """,
            (CAMPAIGN_ID, identifier, TRACKING_SEED, int(fold), int(attempt)),
        ).fetchall()
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = registry.get_run(str(raw["run_id"]))
        if row is None:
            raise NonlinearRunError("attempt registry row disappeared during lookup")
        if canonical_json(row.get("config")) != canonical_json(configuration):
            raise NonlinearRunError(
                "registry attempt has the same slot identity but different config"
            )
        result.append(row)
    return result


def _result_from_bundle(
    artifact: Path,
    *,
    run_id: str,
    configuration: Mapping[str, Any],
    profile: str,
    fold: int,
) -> dict[str, Any]:
    try:
        payload = json.loads((artifact / "results.json").read_text(encoding="utf-8"))
        resolved = json.loads(
            (artifact / "resolved_configuration.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise NonlinearRunError("published attempt metadata is unreadable") from error
    if not isinstance(payload, Mapping) or not isinstance(resolved, Mapping):
        raise NonlinearRunError("published attempt metadata is malformed")
    if canonical_json(resolved) != canonical_json(configuration):
        raise NonlinearRunError("published attempt configuration does not match the slot")
    if (
        payload.get("run_id") != run_id
        or payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("profile") != profile
        or payload.get("outer_fold") != fold
        or payload.get("status") != "completed"
    ):
        raise NonlinearRunError("published attempt result identity is inconsistent")
    return dict(payload)


def _artifact_kind(relative: str) -> str:
    top_level = relative.split("/", 1)[0]
    return (
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


def _finish_published_success(
    *,
    registry: Registry,
    archive: RunArchive,
    configuration: Mapping[str, Any],
    profile: str,
    fold: int,
    failpoint: Any = None,
) -> dict[str, Any]:
    """Idempotently reconcile publish -> registry -> marker -> completed.

    ``failpoint`` is a test-only callback invoked at durable boundaries.  A
    replay after any callback failure must either complete the exact immutable
    bundle or reject it; it never creates another run for the same attempt.
    """

    def hit(name: str) -> None:
        if failpoint is not None:
            failpoint(name)

    final = archive.artifact_path
    run_id = archive.run_id
    row = registry.get_run(run_id)
    if row is None:
        raise NonlinearRunError("published attempt has no registry authority")
    if row.get("status") not in {"running", "finalizing", "completed"}:
        raise AttemptConsumedError(
            f"attempt registry status {row.get('status')!r} is not recoverable"
        )
    markers = [
        name
        for name in ("_SUCCESS", "_FAILED", "_PRUNED")
        if (final / name).exists() or (final / name).is_symlink()
    ]
    if markers == ["_SUCCESS"]:
        verify_run_bundle(final)
    elif not markers:
        verify_unmarked_run_bundle(final)
    else:
        raise AttemptConsumedError(
            f"published attempt has non-success completion markers: {markers}"
        )
    result = _result_from_bundle(
        final,
        run_id=run_id,
        configuration=configuration,
        profile=profile,
        fold=fold,
    )
    arms = result.get("arms")
    if not isinstance(arms, Mapping) or "observed_near" not in arms:
        raise NonlinearRunError("published attempt has no observed-near result")
    try:
        near_mse = float(arms["observed_near"]["evaluation"]["component_equal_mse"])
        duration = float(result["duration_seconds"])
        peak_vram_gb = float(result["peak_vram_gb"])
        parameter_count = int(result["parameter_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise NonlinearRunError("published attempt lifecycle metrics are malformed") from error
    if not all(np.isfinite(value) for value in (near_mse, duration, peak_vram_gb)):
        raise NonlinearRunError("published attempt lifecycle metrics are nonfinite")
    primary_metric = str(configuration["evaluation"]["primary_metric"])
    split = str(configuration["evaluation"]["canonical_prediction_split"])
    expected_updates = {
        "duration_seconds": duration,
        "peak_vram_gb": peak_vram_gb,
        "parameter_count": parameter_count,
        "primary_metric_name": primary_metric,
        "primary_metric_value": near_mse,
        "artifact_path": str(final),
    }
    row = registry.get_run(run_id)
    assert row is not None
    for key, expected in expected_updates.items():
        observed = row.get(key)
        if observed is not None and (
            (isinstance(expected, float) and float(observed) != expected)
            or (not isinstance(expected, float) and str(observed) != str(expected))
        ):
            raise NonlinearRunError(
                f"registry {key} conflicts with immutable published metadata"
            )
    if row["status"] == "running":
        registry.transition_run(
            run_id,
            "finalizing",
            end_time=utc_now(),
            **expected_updates,
        )
    elif row["status"] == "finalizing":
        registry.transition_run(run_id, "finalizing", **expected_updates)
    hit("registry_finalizing")

    files = {
        path.relative_to(final).as_posix(): path
        for path in sorted(final.rglob("*"))
        if path.is_file() and path.name not in {"_SUCCESS", "_FAILED", "_PRUNED"}
    }
    with registry.connect() as connection:
        existing_rows = connection.execute(
            """
            SELECT artifact_id, kind, path, sha256, size_bytes, status
            FROM artifacts WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    existing_by_relative: dict[str, Any] = {}
    for existing in existing_rows:
        path = Path(str(existing["path"]))
        try:
            relative = path.resolve(strict=False).relative_to(
                final.resolve(strict=True)
            ).as_posix()
        except ValueError as error:
            raise NonlinearRunError(
                "registry artifact path escapes the immutable run bundle"
            ) from error
        if relative in existing_by_relative:
            raise NonlinearRunError("registry contains duplicate artifact authorities")
        existing_by_relative[relative] = existing
    extra = set(existing_by_relative).difference(files)
    if extra:
        raise NonlinearRunError(f"registry contains undeclared artifacts: {sorted(extra)}")
    artifact_ids: dict[str, int] = {}
    for relative, path in files.items():
        expected_sha = sha256_file(path)
        expected_size = path.stat().st_size
        expected_kind = _artifact_kind(relative)
        existing = existing_by_relative.get(relative)
        if existing is None:
            artifact_ids[relative] = registry.record_artifact(
                run_id,
                kind=expected_kind,
                path=path,
                sha256=expected_sha,
                size_bytes=expected_size,
            )
        else:
            if (
                str(existing["kind"]) != expected_kind
                or str(existing["path"]) != str(path)
                or str(existing["sha256"]) != expected_sha
                or int(existing["size_bytes"]) != expected_size
                or str(existing["status"]) != "present"
            ):
                raise NonlinearRunError(
                    f"registry artifact conflicts with immutable file {relative}"
                )
            artifact_ids[relative] = int(existing["artifact_id"])
    hit("artifacts_recorded")

    checkpoint_relative = "checkpoints/last.ckpt"
    if checkpoint_relative not in artifact_ids:
        raise NonlinearRunError("published attempt has no canonical last checkpoint")
    registry.register_checkpoint_metadata(
        artifact_ids[checkpoint_relative],
        run_id=run_id,
        role="last",
        retention_class=("diagnostic" if profile == "pilot" else "exploratory_screen"),
        verification_status="verified",
        monitored_metric=primary_metric,
        monitored_mode="min",
        monitored_value=near_mse,
        metadata={
            "prediction_replayable": True,
            "selected_epoch_then_fresh_refit": profile == "full",
            "forced_resource_refit_epoch": (
                PILOT_REFIT_EPOCH_OVERRIDE if profile == "pilot" else None
            ),
            "arms": list(arms),
        },
    )
    hit("checkpoint_recorded")

    expected_metrics = {
        f"{split}/{arm}_component_equal_mse": float(
            value["evaluation"]["component_equal_mse"]
        )
        for arm, value in arms.items()
    }
    with registry.connect() as connection:
        metric_rows = connection.execute(
            "SELECT name, value, split FROM metrics WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    observed_metrics: dict[str, Any] = {}
    for metric in metric_rows:
        name = str(metric["name"])
        if name in observed_metrics:
            raise NonlinearRunError("registry contains duplicate lifecycle metrics")
        observed_metrics[name] = metric
    if set(observed_metrics).difference(expected_metrics):
        raise NonlinearRunError("registry contains unexpected lifecycle metrics")
    for name, value in expected_metrics.items():
        existing = observed_metrics.get(name)
        if existing is None:
            registry.record_metric(run_id, name, value, split=split)
        elif str(existing["split"]) != split or float(existing["value"]) != value:
            raise NonlinearRunError(f"registry metric conflicts for {name}")
    hit("metrics_recorded")

    archive.ensure_success()
    hit("success_marked")
    row = registry.get_run(run_id)
    assert row is not None
    if row["status"] == "finalizing":
        registry.transition_run(run_id, "completed")
    elif row["status"] != "completed":
        raise NonlinearRunError("registry left finalizing state unexpectedly")
    hit("registry_completed")
    issues = registry.verify_artifacts(run_id=run_id)
    if issues:
        raise NonlinearRunError(f"registered artifact verification failed: {issues}")
    return {
        "run_id": run_id,
        "profile": profile,
        "fold": fold,
        "artifact_path": str(final),
        "duration_seconds": duration,
        "peak_vram_gb": peak_vram_gb,
        "projected_full_hours_per_fold": float(
            result["controls"]["projected_full_hours_per_fold"]
        ),
        "observed_near_mse": near_mse,
        "observed_near_jacobian_summary": arms["observed_near"]["jacobian_summary"],
        "lifecycle_reconciled": True,
    }


def reconcile_existing_attempt(arguments: argparse.Namespace) -> dict[str, Any] | None:
    """Recover the one exact successful attempt, or declare it consumed.

    The function performs no numerical data load. The wrapper invokes it only
    after its source/data and one-GPU live-environment gates have passed.
    """

    paths = current_paths()
    configuration = _configuration(
        profile=arguments.profile,
        fold=arguments.fold,
        attempt=arguments.attempt,
    )
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    rows = _attempt_rows(
        registry,
        configuration=configuration,
        fold=arguments.fold,
        attempt=arguments.attempt,
    )
    if not rows:
        return None
    if len(rows) != 1:
        raise NonlinearRunError(
            "multiple registry runs claim one variant/seed/fold/attempt slot"
        )
    row = rows[0]
    artifact = RunArchive.artifact_path_for(str(row["run_id"]), paths)
    if not artifact.is_dir() or artifact.is_symlink():
        raise AttemptConsumedError(
            "declared attempt has no recoverable immutable published bundle"
        )
    archive = RunArchive.from_published(str(row["run_id"]), paths=paths)
    return _finish_published_success(
        registry=registry,
        archive=archive,
        configuration=configuration,
        profile=arguments.profile,
        fold=arguments.fold,
    )


def abandon_incomplete_attempt(arguments: argparse.Namespace) -> dict[str, Any]:
    """Explicitly seal one dead partial attempt as immutable failure.

    This operation is intentionally separate from reconciliation and retry
    materialization.  It refuses a live local owner and never changes a
    successful or success-ready published bundle.
    """

    paths = current_paths()
    configuration = _configuration(
        profile=arguments.profile,
        fold=arguments.fold,
        attempt=arguments.attempt,
    )
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    rows = _attempt_rows(
        registry,
        configuration=configuration,
        fold=arguments.fold,
        attempt=arguments.attempt,
    )
    if len(rows) != 1:
        raise NonlinearRunError(
            "explicit abandonment requires exactly one declared registry attempt"
        )
    row = rows[0]
    run_id = str(row["run_id"])
    initial_status = str(row.get("status"))
    if initial_status == "completed":
        return {
            "run_id": run_id,
            "status": initial_status,
            "already_terminal": True,
        }
    if initial_status not in {
        "pending",
        "running",
        "finalizing",
        "failed",
        "cancelled",
        "pruned",
    }:
        raise NonlinearRunError("attempt status cannot be explicitly abandoned")
    artifact = RunArchive.artifact_path_for(run_id, paths)
    scratch = paths.scratch_root / "active_runs" / run_id
    if artifact.is_symlink():
        raise NonlinearRunError("attempt artifact path may not be a symlink")
    if artifact.is_dir():
        if (artifact / "_SUCCESS").exists() or not any(
            (artifact / marker).exists() for marker in ("_FAILED", "_PRUNED")
        ):
            raise NonlinearRunError(
                "published attempt may be recoverable success; reconcile it instead"
            )
        verify_run_bundle(artifact, require_success_contract=False)
    else:
        if artifact.exists() or artifact.is_symlink():
            raise NonlinearRunError("attempt artifact path is unsafe")
        if scratch.is_dir():
            process_path = scratch / "diagnostics/attempt_process.json"
            if process_path.is_file() and not process_path.is_symlink():
                try:
                    process = json.loads(process_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as error:
                    raise NonlinearRunError(
                        "attempt process authority is unreadable"
                    ) from error
                if not isinstance(process, Mapping):
                    raise NonlinearRunError("attempt process authority is malformed")
                pid = process.get("pid")
                ticks = process.get("proc_start_ticks")
                host = process.get("host")
                if host != socket.gethostname():
                    raise NonlinearRunError(
                        "cannot prove a partial attempt on another host is dead"
                    )
                if isinstance(pid, int) and isinstance(ticks, int):
                    if _process_start_ticks(pid) == ticks:
                        raise NonlinearRunError(
                            f"attempt owner is still live as PID {pid}"
                        )
            archive = RunArchive.attach_active(run_id, paths=paths)
        elif not scratch.exists() and not scratch.is_symlink():
            archive = RunArchive.create(run_id, paths=paths)
        else:
            raise NonlinearRunError("attempt scratch path is unsafe")
        archive.finalize_failure(
            "explicitly abandoned after a dead/incomplete execution",
            failure_category="explicit_dead_attempt_abandonment",
        )
        verify_run_bundle(artifact, require_success_contract=False)
    if initial_status in {"pending", "running", "finalizing"}:
        registry.record_failure(
            run_id=run_id,
            category="explicit_dead_attempt_abandonment",
            message="dead/incomplete attempt sealed before retry materialization",
            details={"profile": arguments.profile, "attempt": arguments.attempt},
        )
        registry.transition_run(
            run_id,
            "failed",
            end_time=utc_now(),
            failure_category="explicit_dead_attempt_abandonment",
        )
    return {
        "run_id": run_id,
        "status": (
            "failed" if initial_status in {"pending", "running", "finalizing"}
            else initial_status
        ),
        "artifact_path": str(artifact),
        "already_terminal": initial_status in {"failed", "cancelled", "pruned"},
    }


def _retry_parent(
    registry: Registry,
    *,
    configuration: Mapping[str, Any],
    fold: int,
    attempt: int,
) -> str | None:
    if attempt == 1:
        return None
    identifier = scientific_id(configuration)
    # Attempt-specific immutable authorities (materialized config SHA, marker
    # path, launch job ID) may legitimately differ between a1 and a2 while the
    # scientific payload remains identical. Query the predecessor by that
    # stable scientific identity and exact execution coordinates; the analyzer
    # later binds each full configuration to its separately declared plan.
    with registry.connect() as connection:
        identities = connection.execute(
            """
            SELECT run_id FROM runs
            WHERE campaign_id = ? AND scientific_id = ?
              AND seed = ? AND fold = ? AND attempt = ?
            ORDER BY created_at, run_id
            """,
            (
                CAMPAIGN_ID,
                identifier,
                TRACKING_SEED,
                int(fold),
                int(attempt - 1),
            ),
        ).fetchall()
    previous: list[dict[str, Any]] = []
    for identity in identities:
        row = registry.get_run(str(identity["run_id"]))
        if row is None:
            raise NonlinearRunError(
                "preceding retry attempt disappeared during lookup"
            )
        row_configuration = row.get("config")
        if (
            not isinstance(row_configuration, Mapping)
            or scientific_id(row_configuration) != identifier
        ):
            raise NonlinearRunError(
                "preceding retry attempt has a different scientific payload"
            )
        previous.append(row)
    if len(previous) != 1 or previous[0].get("status") not in {
        "failed",
        "cancelled",
        "pruned",
    }:
        raise NonlinearRunError(
            "attempt >1 requires exactly one terminal unsuccessful preceding attempt"
        )
    return str(previous[0]["run_id"])


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise NonlinearRunError(
            "each fold process requires exactly one visible CUDA device"
        )
    if arguments.fold not in range(4):
        raise ValueError("fold must be 0, 1, 2, or 3")
    if ENVIRONMENT_LOCK_REQUIRED and (
        not isinstance(ENVIRONMENT_LOCK_REPORT, Mapping)
        or ENVIRONMENT_LOCK_REPORT.get("verified") is not True
        or ENVIRONMENT_LOCK_REPORT.get("visibility_mode") != "job"
        or not isinstance(
            ENVIRONMENT_LOCK_REPORT.get("environment_lock_sha256"), str
        )
        or len(ENVIRONMENT_LOCK_REPORT["environment_lock_sha256"]) != 64
        or not isinstance(
            ENVIRONMENT_LOCK_REPORT.get("verification_sha256"), str
        )
        or len(ENVIRONMENT_LOCK_REPORT["verification_sha256"]) != 64
    ):
        raise NonlinearRunError("frozen live-environment verification is missing")
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
    if (
        not np.isfinite(MAXIMUM_PROJECTED_HOURS_PER_FOLD)
        or MAXIMUM_PROJECTED_HOURS_PER_FOLD <= 0
    ):
        raise NonlinearRunError("projected-runtime threshold must be finite and positive")
    expected_evaluation_split = (
        "validation" if arguments.profile == "pilot" else CANONICAL_PRODUCTION_SPLIT_LABEL
    )
    if configuration["evaluation"]["canonical_prediction_split"] != expected_evaluation_split:
        raise NonlinearRunError("canonical evaluation split changed")
    if configuration["masking"].get("receiver_expression_input") is not False:
        raise NonlinearRunError("receiver RNA input must remain disabled")
    if arguments.profile == "pilot" and (
        not SOURCE_CONFIG_DATA_HASHES_VERIFIED
        or not GRAPH_SPECIFIC_INVARIANTS_VERIFIED
    ):
        raise NonlinearRunError(
            "pilot source/config/data/graph authorities were not verified"
        )
    recovered = reconcile_existing_attempt(arguments)
    if recovered is not None:
        return recovered
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
    retry_of = _retry_parent(
        registry,
        configuration=configuration,
        fold=arguments.fold,
        attempt=arguments.attempt,
    )
    final = RunArchive.artifact_path_for(run_id, paths)
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
        retry_of=retry_of,
        enforce_unique_attempt=True,
        git_commit=provenance["git_commit"],
        dirty_status=provenance["dirty_fingerprint"] is not None,
        preprocessing_version=PREPROCESSING_VERSION,
        dataset_fingerprint=RAW_FINGERPRINT,
        split_fingerprint=SPLIT_FINGERPRINT,
        host=socket.gethostname(),
        gpu_model=provenance["hardware"]["gpu_name"],
    )
    # The registry intent precedes scratch creation.  A crash at either side of
    # this boundary consumes the declared attempt instead of allowing a second
    # immutable bundle to be created for the same slot.
    archive = RunArchive.create(run_id, paths=paths)
    scratch = archive.scratch_path
    process_ticks = _process_start_ticks(os.getpid())
    if process_ticks is None:
        raise NonlinearRunError("could not bind run to a stable local process identity")
    archive.write_json(
        "diagnostics/attempt_process.json",
        {
            "run_id": run_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "proc_start_ticks": process_ticks,
            "recorded_at": utc_now(),
        },
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
        eligible = _load_cpu_vector(prepared, ELIGIBILITY_FILE).astype(
            bool, copy=False
        )
        masks = _split_masks(
            folds, groups, eligible, arguments.fold, profile=arguments.profile
        )
        split_control = _split_component_overlap_control(masks, groups)
        target = _load_gpu_matrix(prepared, "expression_log1p.npy", device)
        morphology = _load_gpu_matrix(prepared, "metadata.npy", device)
        frozen_gene_eligibility = _load_frozen_gene_eligibility(prepared)
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
            phase_inputs = None
            phase_metadata = None
            if PHASE_INPUT_BUILDER is not None:
                built = PHASE_INPUT_BUILDER(
                    arm=arm,
                    target=target,
                    feature=feature,
                    prepared=prepared,
                    masks=masks,
                    groups=groups,
                    profile=arguments.profile,
                    fold=arguments.fold,
                    device=device,
                )
                if (
                    not isinstance(built, tuple)
                    or len(built) != 2
                    or not isinstance(built[0], Mapping)
                    or not isinstance(built[1], Mapping)
                ):
                    raise NonlinearRunError(
                        "phase input builder must return (tensor mapping, metadata mapping)"
                    )
                phase_inputs, phase_metadata = built
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
                phase_inputs=phase_inputs,
                frozen_gene_eligibility=frozen_gene_eligibility,
            )
            if phase_metadata is not None:
                state["phase_transform"] = dict(phase_metadata)
                result["phase_transform"] = dict(phase_metadata)
            checkpoint_arms[arm] = state
            if jacobian_parts is not None:
                for name, value in jacobian_parts.items():
                    matrices[f"{arm}_{name}"] = np.asarray(value)
            result["gene_mse"] = result["gene_mse"].tolist()
            result["gene_pearson"] = result["gene_pearson"].tolist()
            result["eligible_genes"] = result["eligible_genes"].tolist()
            result["raw_eligible_genes"] = result[
                "raw_eligible_genes"
            ].tolist()
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
            if phase_inputs is not None:
                del phase_inputs
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
        checkpoint_replay_metric_error = max(
            float(result["checkpoint_replay"]["maximum_metric_abs_error"])
            for result in results.values()
        )
        checkpoint_replay_prediction_error = max(
            float(result["checkpoint_replay"]["maximum_prediction_abs_error"])
            for result in results.values()
        )
        checkpoint_replay_device_types = {
            str(result["checkpoint_replay"]["replay_device_type"])
            for result in results.values()
        }
        checkpoint_replay_device_type = (
            next(iter(checkpoint_replay_device_types))
            if len(checkpoint_replay_device_types) == 1
            else "mixed"
        )
        runtime_gate_passed = (
            projected_hours <= MAXIMUM_PROJECTED_HOURS_PER_FOLD
        )
        outer_test_untouched = (
            arguments.profile != "pilot"
            or (
                configuration["evaluation"]["canonical_prediction_split"]
                == "validation"
                and all(
                    result["evaluation_role"] == "validation"
                    for result in results.values()
                )
                and all(
                    state["evaluation_mask"] == "validation"
                    for state in checkpoint_arms.values()
                )
            )
        )
        controls = {
            "analytical_nonlinear_jacobian": control,
            "receiver_expression_input": False,
            "receiver_rna_or_derived_covariate_model_input": False,
            "identity_oracle": oracle,
            "identity_oracle_actually_executed": True,
            "identity_oracle_row_top1_fraction": oracle["row_top1_fraction"],
            "all_outputs_finite": all_outputs_finite,
            "train_validation_test_component_overlap": not split_control["passed"],
            "split_overlap_control": split_control,
            "graph_specific_invariants": bool(GRAPH_SPECIFIC_INVARIANTS_VERIFIED),
            "source_config_data_hashes_verified": bool(
                SOURCE_CONFIG_DATA_HASHES_VERIFIED
            ),
            "checkpoint_gpu_replay_max_abs_metric_error": (
                checkpoint_replay_metric_error
            ),
            "checkpoint_gpu_replay_max_abs_prediction_error": (
                checkpoint_replay_prediction_error
            ),
            "checkpoint_replay_device_type": checkpoint_replay_device_type,
            "canonical_production_split_label": (
                CANONICAL_PRODUCTION_SPLIT_LABEL
            ),
            "outer_test_untouched": outer_test_untouched,
            "peak_vram_gb": peak_vram_gb,
            "peak_vram_gate_passed": peak_vram_gb <= 20.5,
            "projected_full_hours_per_fold": projected_hours,
            "maximum_projected_hours_per_fold": (
                MAXIMUM_PROJECTED_HOURS_PER_FOLD
            ),
            "projected_runtime_gate_passed": runtime_gate_passed,
        }
        if ENVIRONMENT_LOCK_REQUIRED:
            assert ENVIRONMENT_LOCK_REPORT is not None
            controls.update(
                {
                    "environment_lock_verified": True,
                    "environment_lock_sha256": ENVIRONMENT_LOCK_REPORT[
                        "environment_lock_sha256"
                    ],
                    "environment_verification_sha256": ENVIRONMENT_LOCK_REPORT[
                        "verification_sha256"
                    ],
                    "environment_visibility_mode": ENVIRONMENT_LOCK_REPORT[
                        "visibility_mode"
                    ],
                }
            )
        if not controls["peak_vram_gate_passed"]:
            raise NonlinearRunError(f"resource gate failed: {controls}")
        if not controls["all_outputs_finite"]:
            raise NonlinearRunError("result finite-value control failed")
        if controls["identity_oracle_row_top1_fraction"] != 1.0:
            raise NonlinearRunError(f"identity-oracle control failed: {oracle}")
        if arguments.profile == "pilot":
            required_pilot_controls = {
                "train_validation_test_component_overlap": False,
                "receiver_rna_or_derived_covariate_model_input": False,
                "identity_oracle_actually_executed": True,
                "graph_specific_invariants": True,
                "source_config_data_hashes_verified": True,
                "checkpoint_replay_device_type": "cuda",
                "canonical_production_split_label": "test",
                "outer_test_untouched": True,
                "projected_runtime_gate_passed": True,
            }
            if ENVIRONMENT_LOCK_REQUIRED:
                required_pilot_controls.update(
                    {
                        "environment_lock_verified": True,
                        "environment_visibility_mode": "job",
                    }
                )
            failed = {
                name: controls[name]
                for name, expected in required_pilot_controls.items()
                if controls[name] != expected
            }
            if failed:
                raise NonlinearRunError(
                    f"pilot technical gate failed: {failed}"
                )
            if (
                checkpoint_replay_metric_error
                > MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR
                or checkpoint_replay_prediction_error
                > MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR
            ):
                raise NonlinearRunError(
                    f"pilot checkpoint replay gate failed: {controls}"
                )
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
            "sample_counts": {
                name: int(np.sum(mask))
                for name, mask in masks.items()
                if not (arguments.profile == "pilot" and name == "test")
            },
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
        return _finish_published_success(
            registry=registry,
            archive=archive,
            configuration=configuration,
            profile=arguments.profile,
            fold=arguments.fold,
        )
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
            try:
                if final.is_dir() and not any(
                    (final / marker).exists()
                    for marker in ("_SUCCESS", "_FAILED", "_PRUNED")
                ):
                    # A publish-success-pending bundle is already immutable and
                    # scientifically complete. A transient registry/finalizer
                    # error must leave it recoverable, not relabel it failed.
                    try:
                        verify_unmarked_run_bundle(final)
                    except BaseException:
                        RunArchive.from_published(
                            run_id, paths=paths
                        ).mark_published_failure()
                    else:
                        raise
                elif archive.scratch_path.is_dir():
                    archive.finalize_failure(
                        error, failure_category="same_gene_nonlinear_run_failure"
                    )
                verify_run_bundle(final, require_success_contract=False)
                registry.transition_run(
                    run_id,
                    "failed",
                    end_time=utc_now(),
                    duration_seconds=time.perf_counter() - started,
                    failure_category="same_gene_nonlinear_run_failure",
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
