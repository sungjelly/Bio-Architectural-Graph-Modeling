"""Masked-expression metrics and block-level paired inference.

Predictions from locked model seeds are averaged before the primary score is
computed.  Spatial blocks are the resampling/sign-flip units.  Cells, masked
entries, and model seeds are never promoted to independent replicates.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import rankdata


_MAX_EXACT_SIGN_FLIP_BLOCKS = 24


def _torch_module() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _is_torch_tensor(value: Any) -> bool:
    torch = _torch_module()
    return torch is not None and torch.is_tensor(value)


def _to_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if _is_torch_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _validate_reduction(reduction: str) -> str:
    reduction = str(reduction).lower()
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")
    return reduction


def _masked_loss(
    y_true: Any,
    y_pred: Any,
    mask: Any,
    *,
    kind: str,
    delta: float = 1.0,
    reduction: str = "mean",
) -> Any:
    """Backend-preserving loss used by the public masked loss functions."""

    reduction = _validate_reduction(reduction)
    delta = float(delta)
    if kind == "huber" and (not math.isfinite(delta) or delta <= 0):
        raise ValueError("delta must be finite and positive")

    torch = _torch_module()
    if torch is not None and (
        torch.is_tensor(y_true) or torch.is_tensor(y_pred)
    ):
        if torch.is_tensor(y_pred):
            prediction = y_pred
        else:
            prediction = torch.as_tensor(y_pred)
        target = torch.as_tensor(
            y_true,
            dtype=prediction.dtype,
            device=prediction.device,
        )
        selected_mask = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=prediction.device,
        )
        if target.shape != prediction.shape or selected_mask.shape != prediction.shape:
            raise ValueError("y_true, y_pred, and mask must have the same shape")
        valid = selected_mask & torch.isfinite(target) & torch.isfinite(prediction)
        if not bool(valid.any().item()):
            raise ValueError("mask selects no finite prediction/target pairs")
        # Sanitising before subtraction prevents NaNs outside the selected mask
        # from contaminating gradients through an unused branch.
        difference = torch.where(
            valid,
            prediction - target,
            torch.zeros_like(prediction),
        )
        absolute = torch.abs(difference)
        if kind == "mse":
            loss = difference.square()
        elif kind == "mae":
            loss = absolute
        else:
            loss = torch.where(
                absolute <= delta,
                0.5 * difference.square(),
                delta * (absolute - 0.5 * delta),
            )
        if reduction == "none":
            return loss
        selected = loss[valid]
        return selected.sum() if reduction == "sum" else selected.mean()

    target_np = _to_numpy(y_true)
    prediction_np = _to_numpy(y_pred)
    selected_mask_np = _to_numpy(mask, dtype=bool)
    if (
        target_np.shape != prediction_np.shape
        or selected_mask_np.shape != target_np.shape
    ):
        raise ValueError("y_true, y_pred, and mask must have the same shape")
    valid_np = (
        selected_mask_np
        & np.isfinite(target_np)
        & np.isfinite(prediction_np)
    )
    if not np.any(valid_np):
        raise ValueError("mask selects no finite prediction/target pairs")
    difference_np = np.zeros(
        target_np.shape,
        dtype=np.result_type(target_np.dtype, prediction_np.dtype, np.float64),
    )
    np.subtract(
        prediction_np,
        target_np,
        out=difference_np,
        where=valid_np,
    )
    absolute_np = np.abs(difference_np)
    if kind == "mse":
        loss_np = difference_np**2
    elif kind == "mae":
        loss_np = absolute_np
    else:
        loss_np = np.where(
            absolute_np <= delta,
            0.5 * difference_np**2,
            delta * (absolute_np - 0.5 * delta),
        )
    loss_np[~valid_np] = 0.0
    if reduction == "none":
        return loss_np
    selected_np = loss_np[valid_np]
    return float(selected_np.sum() if reduction == "sum" else selected_np.mean())


def masked_huber_loss(
    y_true: Any,
    y_pred: Any,
    mask: Any,
    *,
    delta: float = 1.0,
    reduction: str = "mean",
) -> Any:
    """Huber loss over masked finite entries only."""

    return _masked_loss(
        y_true,
        y_pred,
        mask,
        kind="huber",
        delta=delta,
        reduction=reduction,
    )


def masked_mse_loss(
    y_true: Any,
    y_pred: Any,
    mask: Any,
    *,
    reduction: str = "mean",
) -> Any:
    """Mean/summed squared error over masked finite entries only."""

    return _masked_loss(y_true, y_pred, mask, kind="mse", reduction=reduction)


def masked_mae_loss(
    y_true: Any,
    y_pred: Any,
    mask: Any,
    *,
    reduction: str = "mean",
) -> Any:
    """Mean/summed absolute error over masked finite entries only."""

    return _masked_loss(y_true, y_pred, mask, kind="mae", reduction=reduction)


# Concise aliases for metric-only call sites.
masked_huber = masked_huber_loss
masked_mse = masked_mse_loss
masked_mae = masked_mae_loss


def ensemble_predictions(
    predictions: Any,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> Any:
    """Average locked-seed predictions without treating seeds as replicates.

    Input may be one ``[cells, genes]`` prediction, a ``[seeds, cells, genes]``
    array/tensor, or a sequence of equally shaped predictions.  The return
    backend matches the inputs.
    """

    torch = _torch_module()
    is_sequence = isinstance(predictions, (list, tuple))
    uses_torch = (
        torch is not None
        and (
            torch.is_tensor(predictions)
            or (
                is_sequence
                and len(predictions) > 0
                and any(torch.is_tensor(item) for item in predictions)
            )
        )
    )
    if uses_torch:
        if is_sequence:
            if not predictions:
                raise ValueError("predictions may not be empty")
            reference = next(
                item for item in predictions if torch.is_tensor(item)
            )
            stack = torch.stack(
                [
                    item.to(device=reference.device, dtype=reference.dtype)
                    if torch.is_tensor(item)
                    else torch.as_tensor(
                        item,
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                    for item in predictions
                ],
                dim=0,
            )
        else:
            stack = predictions
        if stack.ndim == 2:
            if weights is not None:
                supplied = np.asarray(weights).reshape(-1)
                if supplied.size != 1:
                    raise ValueError("one prediction accepts exactly one weight")
            return stack.clone()
        if stack.ndim != 3 or stack.shape[0] == 0:
            raise ValueError(
                "predictions must have shape [cells, genes] or "
                "[seeds, cells, genes]"
            )
        if weights is None:
            return stack.mean(dim=0)
        weight = torch.as_tensor(weights, dtype=stack.dtype, device=stack.device)
        if weight.ndim != 1 or weight.numel() != stack.shape[0]:
            raise ValueError("weights must have one entry per prediction seed")
        if not bool(torch.isfinite(weight).all().item()) or bool(
            (weight < 0).any().item()
        ):
            raise ValueError("weights must be finite and non-negative")
        if not bool((weight.sum() > 0).item()):
            raise ValueError("weights must have a positive sum")
        weight = weight / weight.sum()
        return (stack * weight[:, None, None]).sum(dim=0)

    if is_sequence:
        if not predictions:
            raise ValueError("predictions may not be empty")
        stack_np = np.stack([_to_numpy(item) for item in predictions], axis=0)
    else:
        stack_np = _to_numpy(predictions)
    if stack_np.ndim == 2:
        if weights is not None and np.asarray(weights).reshape(-1).size != 1:
            raise ValueError("one prediction accepts exactly one weight")
        return np.array(stack_np, copy=True)
    if stack_np.ndim != 3 or stack_np.shape[0] == 0:
        raise ValueError(
            "predictions must have shape [cells, genes] or "
            "[seeds, cells, genes]"
        )
    if weights is None:
        return np.mean(stack_np, axis=0)
    weight_np = np.asarray(weights, dtype=np.float64)
    if weight_np.ndim != 1 or weight_np.size != stack_np.shape[0]:
        raise ValueError("weights must have one entry per prediction seed")
    if (
        not np.all(np.isfinite(weight_np))
        or np.any(weight_np < 0)
        or weight_np.sum() <= 0
    ):
        raise ValueError("weights must be finite, non-negative, and sum above zero")
    return np.average(stack_np, axis=0, weights=weight_np)


def _prediction_stack_and_ensemble(
    predictions: Any,
    weights: Sequence[float] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(predictions, (list, tuple)):
        if not predictions:
            raise ValueError("predictions may not be empty")
        stack = np.stack(
            [_to_numpy(item, dtype=np.float64) for item in predictions],
            axis=0,
        )
    else:
        stack = _to_numpy(predictions, dtype=np.float64)
        if stack.ndim == 2:
            stack = stack[None, ...]
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError(
            "predictions must have shape [cells, genes] or "
            "[seeds, cells, genes]"
        )
    ensemble = _to_numpy(
        ensemble_predictions(stack, weights=weights),
        dtype=np.float64,
    )
    return stack, ensemble


def _safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _safe_median(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    min_pairs: int,
) -> tuple[float, int]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=np.float64)
    y = np.asarray(y[finite], dtype=np.float64)
    n = int(x.size)
    if n < min_pairs:
        return float("nan"), n
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), n
    if method == "spearman":
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator == 0 or not math.isfinite(denominator):
        return float("nan"), n
    correlation = float(np.dot(x, y) / denominator)
    # Numerical roundoff can otherwise produce values such as 1+2e-16.
    return float(np.clip(correlation, -1.0, 1.0)), n


def _correlation_profiles(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    min_pairs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    n_cells, n_genes = target.shape
    gene_pearson = np.full(n_genes, np.nan, dtype=np.float64)
    gene_spearman = np.full(n_genes, np.nan, dtype=np.float64)
    gene_counts = np.zeros(n_genes, dtype=np.int64)
    for gene in range(n_genes):
        selected = mask[:, gene]
        if np.any(selected):
            gene_pearson[gene], gene_counts[gene] = _safe_correlation(
                target[selected, gene],
                prediction[selected, gene],
                method="pearson",
                min_pairs=min_pairs,
            )
            gene_spearman[gene], _ = _safe_correlation(
                target[selected, gene],
                prediction[selected, gene],
                method="spearman",
                min_pairs=min_pairs,
            )

    cell_pearson = np.full(n_cells, np.nan, dtype=np.float64)
    cell_spearman = np.full(n_cells, np.nan, dtype=np.float64)
    cell_counts = np.zeros(n_cells, dtype=np.int64)
    for cell in range(n_cells):
        selected = mask[cell]
        if np.any(selected):
            cell_pearson[cell], cell_counts[cell] = _safe_correlation(
                target[cell, selected],
                prediction[cell, selected],
                method="pearson",
                min_pairs=min_pairs,
            )
            cell_spearman[cell], _ = _safe_correlation(
                target[cell, selected],
                prediction[cell, selected],
                method="spearman",
                min_pairs=min_pairs,
            )

    def pack(
        pearson: np.ndarray,
        spearman: np.ndarray,
        counts: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "pearson": pearson,
            "spearman": spearman,
            "n_pairs": counts,
            "n_valid_pearson": int(np.isfinite(pearson).sum()),
            "n_valid_spearman": int(np.isfinite(spearman).sum()),
            "mean_pearson": _safe_mean(pearson),
            "median_pearson": _safe_median(pearson),
            "mean_spearman": _safe_mean(spearman),
            "median_spearman": _safe_median(spearman),
        }

    return (
        pack(gene_pearson, gene_spearman, gene_counts),
        pack(cell_pearson, cell_spearman, cell_counts),
    )


def _loss_summary(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    huber_delta: float,
) -> dict[str, Any]:
    valid = mask & np.isfinite(target) & np.isfinite(prediction)
    count = int(valid.sum())
    if count == 0:
        return {
            "n_masked": 0,
            "huber": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
        }
    difference = prediction[valid] - target[valid]
    absolute = np.abs(difference)
    huber = np.where(
        absolute <= huber_delta,
        0.5 * difference**2,
        huber_delta * (absolute - 0.5 * huber_delta),
    )
    return {
        "n_masked": count,
        "huber": float(huber.mean()),
        "mse": float(np.mean(difference**2)),
        "mae": float(absolute.mean()),
    }


def _normalise_block_ids(block_ids: Any, n_cells: int) -> np.ndarray:
    blocks = _to_numpy(block_ids)
    if blocks.shape != (n_cells,):
        raise ValueError("block_ids must have shape [n_cells]")
    return blocks


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ordered_block_groups(block_ids: np.ndarray) -> list[tuple[Any, np.ndarray]]:
    order: list[Any] = []
    indices: dict[str, list[int]] = {}
    labels: dict[str, Any] = {}
    for index, raw_label in enumerate(block_ids):
        label = _python_scalar(raw_label)
        if isinstance(label, float) and math.isnan(label):
            key = "__nan__"
            label = None
        else:
            key = f"{type(label).__name__}:{label!r}"
        if key not in indices:
            order.append(key)
            indices[key] = []
            labels[key] = label
        indices[key].append(index)
    return [
        (labels[key], np.asarray(indices[key], dtype=np.int64))
        for key in order
    ]


def evaluate_masked_predictions(
    y_true: Any,
    predictions: Any,
    mask: Any,
    block_ids: Any | None = None,
    *,
    huber_delta: float = 1.0,
    min_correlation_pairs: int = 2,
    seed_weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one prediction or an ensemble on masked entries.

    Correlation profiles return ``NaN`` for groups with too few finite pairs or
    zero variance.  Per-seed losses are descriptive diagnostics only; the
    primary metrics and block summaries use the mean prediction ensemble.
    """

    target = _to_numpy(y_true, dtype=np.float64)
    selected_mask = _to_numpy(mask, dtype=bool)
    if target.ndim != 2 or selected_mask.shape != target.shape:
        raise ValueError("y_true and mask must have the same [cells, genes] shape")
    if not math.isfinite(float(huber_delta)) or float(huber_delta) <= 0:
        raise ValueError("huber_delta must be finite and positive")
    min_correlation_pairs = int(min_correlation_pairs)
    if min_correlation_pairs < 2:
        raise ValueError("min_correlation_pairs must be at least two")

    stack, ensemble = _prediction_stack_and_ensemble(predictions, seed_weights)
    if stack.shape[1:] != target.shape:
        raise ValueError("prediction and target shapes differ")

    result = _loss_summary(
        target,
        ensemble,
        selected_mask,
        huber_delta=float(huber_delta),
    )
    gene_metrics, cell_metrics = _correlation_profiles(
        target,
        ensemble,
        selected_mask,
        min_pairs=min_correlation_pairs,
    )
    result.update(
        {
            "gene": gene_metrics,
            "cell": cell_metrics,
            "n_prediction_seeds": int(stack.shape[0]),
            "prediction_aggregation": "mean_across_model_seeds_before_scoring",
            "per_seed": [
                {
                    "seed_index": seed_index,
                    **_loss_summary(
                        target,
                        stack[seed_index],
                        selected_mask,
                        huber_delta=float(huber_delta),
                    ),
                    "technical_only": True,
                }
                for seed_index in range(stack.shape[0])
            ],
        }
    )

    block_metrics: list[dict[str, Any]] = []
    if block_ids is not None:
        blocks = _normalise_block_ids(block_ids, target.shape[0])
        for label, rows in _ordered_block_groups(blocks):
            block_mask = selected_mask[rows]
            summary = _loss_summary(
                target[rows],
                ensemble[rows],
                block_mask,
                huber_delta=float(huber_delta),
            )
            valid = (
                block_mask
                & np.isfinite(target[rows])
                & np.isfinite(ensemble[rows])
            )
            flat_pearson, n_pairs = _safe_correlation(
                target[rows][valid],
                ensemble[rows][valid],
                method="pearson",
                min_pairs=min_correlation_pairs,
            )
            flat_spearman, _ = _safe_correlation(
                target[rows][valid],
                ensemble[rows][valid],
                method="spearman",
                min_pairs=min_correlation_pairs,
            )
            block_metrics.append(
                {
                    "block_id": label,
                    "n_cells": int(rows.size),
                    **summary,
                    "pearson_flat": flat_pearson,
                    "spearman_flat": flat_spearman,
                    "n_correlation_pairs": n_pairs,
                }
            )
    result["blocks"] = block_metrics
    result["inference_warning"] = (
        "Cells and model seeds are not independent replicates; aggregate within "
        "spatial blocks before uncertainty estimation."
    )
    return result


def block_bootstrap_ci(
    values: Any,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Deterministic percentile bootstrap over spatial-block values."""

    sample = _to_numpy(values, dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("values contains no finite spatial-block estimates")
    confidence_level = float(confidence_level)
    n_resamples = int(n_resamples)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    estimate = float(statistic(sample))
    if sample.size == 1:
        draws = np.full(n_resamples, estimate, dtype=np.float64)
    else:
        rng = np.random.default_rng(int(seed))
        draws = np.empty(n_resamples, dtype=np.float64)
        # Chunking bounds memory for large requested resample counts.
        chunk_size = max(1, min(n_resamples, 2_000_000 // sample.size))
        for start in range(0, n_resamples, chunk_size):
            stop = min(start + chunk_size, n_resamples)
            indices = rng.integers(
                0,
                sample.size,
                size=(stop - start, sample.size),
            )
            for offset, resample in enumerate(sample[indices]):
                draws[start + offset] = float(statistic(resample))
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        draws,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return {
        "estimate": estimate,
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "seed": int(seed),
        "n_blocks": int(sample.size),
        "resampling_unit": "spatial_block",
        "method": "percentile_block_bootstrap",
    }


def _is_extreme(
    statistics: np.ndarray,
    observed: float,
    alternative: str,
) -> np.ndarray:
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(observed)) * 16
    if alternative == "greater":
        return statistics >= observed - tolerance
    if alternative == "less":
        return statistics <= observed + tolerance
    return np.abs(statistics) >= abs(observed) - tolerance


def paired_sign_flip_test(
    differences: Any,
    *,
    alternative: str = "greater",
    exact_max_blocks: int = 20,
    n_resamples: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired sign-flip test on spatial-block differences.

    Exact enumeration is used when the number of non-zero finite blocks is at
    most ``exact_max_blocks`` (and the implementation feasibility cap of 24).
    Otherwise a deterministic Monte Carlo estimate with the standard plus-one
    correction is returned.
    """

    alternative = str(alternative).lower()
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    exact_max_blocks = int(exact_max_blocks)
    n_resamples = int(n_resamples)
    if exact_max_blocks < 0 or n_resamples <= 0:
        raise ValueError(
            "exact_max_blocks must be non-negative and n_resamples positive"
        )
    values = _to_numpy(differences, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("differences contains no finite spatial-block values")
    n_zero = int(np.count_nonzero(values == 0))
    nonzero = values[values != 0]
    observed = float(values.mean())
    if nonzero.size == 0:
        return {
            "statistic": observed,
            "p_value": 1.0,
            "alternative": alternative,
            "method": "exact_sign_flip",
            "n_blocks": int(values.size),
            "n_nonzero_blocks": 0,
            "n_zero_blocks": n_zero,
            "n_permutations": 1,
            "seed": None,
            "resampling_unit": "spatial_block",
        }

    # Zeros never change a sign-flipped statistic.  Dividing by the original
    # block count retains the observed mean while avoiding duplicate patterns.
    denominator = values.size
    if nonzero.size <= min(exact_max_blocks, _MAX_EXACT_SIGN_FLIP_BLOCKS):
        total = 1 << int(nonzero.size)
        extreme = 0
        bit_positions = np.arange(nonzero.size, dtype=np.uint64)
        chunk_size = min(total, 65_536)
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            codes = np.arange(start, stop, dtype=np.uint64)
            bits = ((codes[:, None] >> bit_positions[None, :]) & 1).astype(
                np.int8,
                copy=False,
            )
            signs = bits * 2 - 1
            statistics = (signs @ nonzero) / denominator
            extreme += int(_is_extreme(statistics, observed, alternative).sum())
        p_value = extreme / total
        method = "exact_sign_flip"
        permutations = total
        result_seed: int | None = None
    else:
        rng = np.random.default_rng(int(seed))
        extreme = 0
        chunk_size = min(n_resamples, max(1, 2_000_000 // nonzero.size))
        completed = 0
        while completed < n_resamples:
            current = min(chunk_size, n_resamples - completed)
            signs = rng.integers(
                0,
                2,
                size=(current, nonzero.size),
                dtype=np.int8,
            )
            signs = signs * 2 - 1
            statistics = (signs @ nonzero) / denominator
            extreme += int(_is_extreme(statistics, observed, alternative).sum())
            completed += current
        p_value = (extreme + 1) / (n_resamples + 1)
        method = "monte_carlo_sign_flip"
        permutations = n_resamples
        result_seed = int(seed)
    return {
        "statistic": observed,
        "p_value": float(p_value),
        "alternative": alternative,
        "method": method,
        "n_blocks": int(values.size),
        "n_nonzero_blocks": int(nonzero.size),
        "n_zero_blocks": n_zero,
        "n_permutations": int(permutations),
        "seed": result_seed,
        "resampling_unit": "spatial_block",
    }


def _loss_vector(
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    *,
    loss: str,
    huber_delta: float,
) -> np.ndarray:
    difference = prediction[valid] - target[valid]
    absolute = np.abs(difference)
    if loss == "mse":
        return difference**2
    if loss == "mae":
        return absolute
    return np.where(
        absolute <= huber_delta,
        0.5 * difference**2,
        huber_delta * (absolute - 0.5 * huber_delta),
    )


def _paired_relative_bootstrap_ci(
    baseline: np.ndarray,
    spatial: np.ndarray,
    *,
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    if baseline.size == 1:
        relative = (
            (baseline[0] - spatial[0]) / baseline[0]
            if abs(baseline[0]) > np.finfo(np.float64).eps
            else float("nan")
        )
        return float(relative), float(relative)
    rng = np.random.default_rng(int(seed))
    values = np.empty(n_resamples, dtype=np.float64)
    chunk_size = max(1, min(n_resamples, 2_000_000 // baseline.size))
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(
            0,
            baseline.size,
            size=(stop - start, baseline.size),
        )
        baseline_mean = baseline[indices].mean(axis=1)
        spatial_mean = spatial[indices].mean(axis=1)
        values[start:stop] = np.divide(
            baseline_mean - spatial_mean,
            baseline_mean,
            out=np.full(stop - start, np.nan, dtype=np.float64),
            where=np.abs(baseline_mean) > np.finfo(np.float64).eps,
        )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        values,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return float(lower), float(upper)


def paired_spatial_gain(
    y_true: Any,
    baseline_predictions: Any,
    spatial_predictions: Any,
    mask: Any,
    block_ids: Any,
    *,
    loss: str = "huber",
    huber_delta: float = 1.0,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
    sign_flip_alternative: str = "greater",
    exact_max_blocks: int = 20,
    n_sign_flips: int = 100_000,
    sign_flip_seed: int = 1,
    baseline_seed_weights: Sequence[float] | np.ndarray | None = None,
    spatial_seed_weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Primary paired block-level gain: baseline loss minus spatial loss.

    Positive delta and relative gain favour the spatial model.  Model seeds are
    ensembled before cell losses are aggregated within blocks.  The block
    bootstrap and sign-flip test then operate on one paired estimate per block.
    """

    loss = str(loss).lower()
    if loss not in {"huber", "mse", "mae"}:
        raise ValueError("loss must be 'huber', 'mse', or 'mae'")
    huber_delta = float(huber_delta)
    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
    target = _to_numpy(y_true, dtype=np.float64)
    selected_mask = _to_numpy(mask, dtype=bool)
    if target.ndim != 2 or selected_mask.shape != target.shape:
        raise ValueError("y_true and mask must have the same [cells, genes] shape")
    baseline_stack, baseline = _prediction_stack_and_ensemble(
        baseline_predictions,
        baseline_seed_weights,
    )
    spatial_stack, spatial = _prediction_stack_and_ensemble(
        spatial_predictions,
        spatial_seed_weights,
    )
    if baseline.shape != target.shape or spatial.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    blocks = _normalise_block_ids(block_ids, target.shape[0])

    block_records: list[dict[str, Any]] = []
    for label, rows in _ordered_block_groups(blocks):
        common_valid = (
            selected_mask[rows]
            & np.isfinite(target[rows])
            & np.isfinite(baseline[rows])
            & np.isfinite(spatial[rows])
        )
        n_masked = int(common_valid.sum())
        if n_masked:
            baseline_loss = float(
                _loss_vector(
                    target[rows],
                    baseline[rows],
                    common_valid,
                    loss=loss,
                    huber_delta=huber_delta,
                ).mean()
            )
            spatial_loss = float(
                _loss_vector(
                    target[rows],
                    spatial[rows],
                    common_valid,
                    loss=loss,
                    huber_delta=huber_delta,
                ).mean()
            )
            delta = baseline_loss - spatial_loss
            relative = (
                delta / baseline_loss
                if abs(baseline_loss) > np.finfo(np.float64).eps
                else float("nan")
            )
        else:
            baseline_loss = spatial_loss = delta = relative = float("nan")
        block_records.append(
            {
                "block_id": label,
                "n_cells": int(rows.size),
                "n_masked": n_masked,
                "baseline_loss": baseline_loss,
                "spatial_loss": spatial_loss,
                "delta": delta,
                "relative_gain": relative,
            }
        )

    valid_records = [
        record
        for record in block_records
        if math.isfinite(record["baseline_loss"])
        and math.isfinite(record["spatial_loss"])
    ]
    if not valid_records:
        raise ValueError("no spatial block has a finite paired masked loss")
    baseline_losses = np.asarray(
        [record["baseline_loss"] for record in valid_records],
        dtype=np.float64,
    )
    spatial_losses = np.asarray(
        [record["spatial_loss"] for record in valid_records],
        dtype=np.float64,
    )
    differences = baseline_losses - spatial_losses
    mean_baseline = float(baseline_losses.mean())
    mean_spatial = float(spatial_losses.mean())
    mean_delta = float(differences.mean())
    relative_gain = (
        mean_delta / mean_baseline
        if abs(mean_baseline) > np.finfo(np.float64).eps
        else float("nan")
    )
    delta_ci = block_bootstrap_ci(
        differences,
        confidence_level=confidence_level,
        n_resamples=n_bootstrap,
        seed=bootstrap_seed,
    )
    relative_lower, relative_upper = _paired_relative_bootstrap_ci(
        baseline_losses,
        spatial_losses,
        confidence_level=float(confidence_level),
        n_resamples=int(n_bootstrap),
        seed=int(bootstrap_seed),
    )
    sign_flip = paired_sign_flip_test(
        differences,
        alternative=sign_flip_alternative,
        exact_max_blocks=exact_max_blocks,
        n_resamples=n_sign_flips,
        seed=sign_flip_seed,
    )
    return {
        "loss": loss,
        "huber_delta": huber_delta if loss == "huber" else None,
        "n_blocks": len(valid_records),
        "n_total_blocks": len(block_records),
        "n_baseline_prediction_seeds": int(baseline_stack.shape[0]),
        "n_spatial_prediction_seeds": int(spatial_stack.shape[0]),
        "prediction_aggregation": "mean_across_model_seeds_before_block_scoring",
        "block_aggregation": "equal_weight_mean_of_spatial_block_losses",
        "baseline_loss": mean_baseline,
        "spatial_loss": mean_spatial,
        "delta": mean_delta,
        "relative_gain": float(relative_gain),
        "relative_gain_percent": float(relative_gain * 100.0),
        "delta_ci": delta_ci,
        "relative_gain_ci": {
            "lower": relative_lower,
            "upper": relative_upper,
            "confidence_level": float(confidence_level),
            "method": "paired_percentile_block_bootstrap",
            "n_resamples": int(n_bootstrap),
            "seed": int(bootstrap_seed),
            "resampling_unit": "spatial_block",
        },
        "sign_flip": sign_flip,
        "blocks": block_records,
        "inference_unit": "spatial_block",
        "technical_units_not_replicates": ["cell", "masked_entry", "model_seed"],
        "descriptive_within_core": True,
    }


def benjamini_hochberg(p_values: Any) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving shape and NaNs."""

    values = _to_numpy(p_values, dtype=np.float64)
    flat = values.reshape(-1)
    adjusted = np.full(flat.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(flat))
    finite = flat[finite_indices]
    if np.any((finite < 0) | (finite > 1)):
        raise ValueError("finite p-values must lie in [0, 1]")
    if finite.size:
        order = np.argsort(finite, kind="mergesort")
        ranked = finite[order]
        raw_adjusted = ranked * finite.size / np.arange(1, finite.size + 1)
        monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
        monotone = np.clip(monotone, 0.0, 1.0)
        unsorted = np.empty_like(monotone)
        unsorted[order] = monotone
        adjusted[finite_indices] = unsorted
    return adjusted.reshape(values.shape)


def bh_fdr(p_values: Any, *, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Return BH adjusted p-values and the corresponding rejection mask."""

    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    adjusted = benjamini_hochberg(p_values)
    reject = np.isfinite(adjusted) & (adjusted <= alpha)
    return {"adjusted_p": adjusted, "reject": reject}
