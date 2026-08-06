"""Evaluation and transductive references for the hybrid count campaign."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from .hybrid_count import (
    NUM_COUNT_STATES,
    NUM_OUTPUT_CHANNELS,
    NUM_POSITIVE_ORDINAL_THRESHOLDS,
    decode_count_states,
    decode_positive_states,
    hybrid_count_hurdle_loss,
    split_hybrid_prediction,
    tokenize_raw_counts,
    validate_raw_counts,
)


@dataclass(frozen=True)
class HybridCountReferences:
    """All-fit, per-gene transductive reference predictions."""

    detection_probability: np.ndarray
    positive_ordinal_probability: np.ndarray
    positive_continuous_standardized: np.ndarray
    detected_state: np.ndarray
    positive_state: np.ndarray
    count_state: np.ndarray
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class HybridCountEvaluation:
    """Decoded arrays and metrics for one fixed mask."""

    count_state: Tensor
    detected: Tensor
    positive_continuous_standardized: Tensor
    reconstructed_count: Tensor
    metrics: Mapping[str, Any]


def _standardization_arrays(
    expression_mean: Any,
    expression_scale: Any,
    *,
    num_genes: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(expression_mean, dtype=np.float64)
    scale = np.asarray(expression_scale, dtype=np.float64)
    if mean.shape != (num_genes,) or scale.shape != (num_genes,):
        raise ValueError("expression standardization must have shape [genes]")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("expression standardization must be finite")
    if np.any(scale <= 0):
        raise ValueError("expression_scale must be strictly positive")
    return mean, scale


def _array_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def fit_hybrid_count_references(
    raw_counts: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> HybridCountReferences:
    """Fit the frozen all-fit references independently within one core."""

    counts = validate_raw_counts(raw_counts)
    n_nodes, n_genes = counts.shape
    mean, scale = _standardization_arrays(
        expression_mean, expression_scale, num_genes=n_genes
    )
    tokens = tokenize_raw_counts(counts)
    positive = counts > 0
    positive_support = positive.sum(axis=0, dtype=np.int64)
    if np.any(positive_support == 0):
        raise ValueError(
            "per-gene positive references require a positive observation for every gene"
        )

    detection_probability = (
        positive_support.astype(np.float64) + 0.5
    ) / (float(n_nodes) + 1.0)
    ordinal_probability = np.empty(
        (n_genes, NUM_POSITIVE_ORDINAL_THRESHOLDS), dtype=np.float64
    )
    positive_continuous = np.empty(n_genes, dtype=np.float64)
    log_counts = np.log1p(counts.astype(np.float64, copy=False))
    for gene_index in range(n_genes):
        gene_positive = positive[:, gene_index]
        positive_tokens = tokens[gene_positive, gene_index]
        support = int(positive_support[gene_index])
        for threshold_index in range(NUM_POSITIVE_ORDINAL_THRESHOLDS):
            threshold_state = threshold_index + 1
            above = int(np.count_nonzero(positive_tokens > threshold_state))
            ordinal_probability[gene_index, threshold_index] = (
                above + 0.5
            ) / (support + 1.0)
        median_log = float(np.median(log_counts[gene_positive, gene_index]))
        positive_continuous[gene_index] = (
            median_log - mean[gene_index]
        ) / scale[gene_index]

    detected_state = detection_probability >= 0.5
    positive_state = 1 + np.sum(ordinal_probability >= 0.5, axis=1)
    count_state = np.where(detected_state, positive_state, 0).astype(
        np.int64, copy=False
    )
    state_support = np.bincount(
        tokens.reshape(-1), minlength=NUM_COUNT_STATES
    ).astype(np.int64, copy=False)
    arrays = {
        "detection_probability": detection_probability,
        "positive_ordinal_probability": ordinal_probability,
        "positive_continuous_standardized": positive_continuous,
        "count_state": count_state,
    }
    audit = {
        "schema": "hybrid_count_all_fit_references_v1",
        "fit_scope": "all_nodes_transductive",
        "n_nodes": int(n_nodes),
        "n_genes": int(n_genes),
        "state_support": [int(value) for value in state_support],
        "state_prevalence": [
            float(value / counts.size) for value in state_support
        ],
        "zero_prevalence": float(state_support[0] / counts.size),
        "smoothing": "Jeffreys_add_half",
        "thresholds_fitted": False,
        "reference_sha256": _array_digest(arrays),
    }
    return HybridCountReferences(
        detection_probability=detection_probability,
        positive_ordinal_probability=ordinal_probability,
        positive_continuous_standardized=positive_continuous,
        detected_state=detected_state,
        positive_state=positive_state.astype(np.int64, copy=False),
        count_state=count_state,
        audit=audit,
    )


def _binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if target.dtype != np.bool_:
        target = target.astype(bool, copy=False)
    if prediction.dtype != np.bool_:
        prediction = prediction.astype(bool, copy=False)
    positive = int(np.count_nonzero(target))
    negative = int(target.size - positive)
    if positive == 0 or negative == 0:
        raise ValueError("detection metrics require zero and positive targets")
    true_positive = int(np.count_nonzero(target & prediction))
    true_negative = int(np.count_nonzero(~target & ~prediction))
    predicted_positive = int(np.count_nonzero(prediction))
    sensitivity = true_positive / positive
    specificity = true_negative / negative
    precision = (
        None if predicted_positive == 0 else true_positive / predicted_positive
    )
    return {
        "detection_balanced_accuracy": 0.5 * (sensitivity + specificity),
        "detection_sensitivity": sensitivity,
        "detection_specificity": specificity,
        "detection_precision": precision,
        "detection_positive_support": positive,
        "detection_zero_support": negative,
        "detection_predicted_positive": predicted_positive,
    }


def _state_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    num_states: int,
    prefix: str,
) -> dict[str, Any]:
    if target.shape != prediction.shape or target.ndim != 1 or target.size == 0:
        raise ValueError("state metrics require aligned nonempty vectors")
    support: list[int] = []
    recall: list[float | None] = []
    for state in range(num_states):
        selected = target == state
        state_support = int(np.count_nonzero(selected))
        support.append(state_support)
        recall.append(
            None
            if state_support == 0
            else float(np.mean(prediction[selected] == state))
        )
    finite_recall = [value for value in recall if value is not None]
    if not finite_recall:
        raise ValueError("state metrics have no supported class")
    result: dict[str, Any] = {
        f"{prefix}_exact_accuracy": float(np.mean(target == prediction)),
        f"{prefix}_balanced_accuracy": float(np.mean(finite_recall)),
        f"{prefix}_support": support,
        f"{prefix}_recall": recall,
    }
    return result


def _huber_numpy(error: np.ndarray, *, delta: float = 1.0) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(
        absolute <= delta,
        0.5 * absolute**2,
        delta * (absolute - 0.5 * delta),
    )


def _probability_logit(probability: np.ndarray) -> np.ndarray:
    if np.any(probability <= 0) or np.any(probability >= 1):
        raise ValueError("smoothed probabilities must lie strictly inside (0, 1)")
    return np.log(probability) - np.log1p(-probability)


def reference_prediction_tensor(
    references: HybridCountReferences,
    *,
    num_nodes: int,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Materialize reference heads for exact loss reconciliation."""

    n_genes = int(references.detection_probability.shape[0])
    output = torch.empty(
        (num_nodes, n_genes, NUM_OUTPUT_CHANNELS),
        dtype=torch.float32,
        device=device,
    )
    output[..., 0] = torch.as_tensor(
        _probability_logit(references.detection_probability),
        dtype=torch.float32,
        device=device,
    )
    output[..., 1:7] = torch.as_tensor(
        _probability_logit(references.positive_ordinal_probability),
        dtype=torch.float32,
        device=device,
    )
    output[..., 7] = torch.as_tensor(
        references.positive_continuous_standardized,
        dtype=torch.float32,
        device=device,
    )
    return output


def evaluate_hybrid_count_output(
    prediction: Tensor,
    raw_target: Tensor,
    target_mask: Tensor,
    *,
    expression_mean: Any,
    expression_scale: Any,
    references: HybridCountReferences,
) -> HybridCountEvaluation:
    """Evaluate one model output on one immutable mask."""

    if prediction.device != raw_target.device or target_mask.device != raw_target.device:
        raise ValueError("prediction, target, and mask must share a device")
    n_nodes, n_genes = raw_target.shape
    mean, scale = _standardization_arrays(
        expression_mean, expression_scale, num_genes=n_genes
    )
    mean_tensor = torch.as_tensor(
        mean, dtype=torch.float32, device=raw_target.device
    )
    scale_tensor = torch.as_tensor(
        scale, dtype=torch.float32, device=raw_target.device
    )
    loss = hybrid_count_hurdle_loss(
        prediction,
        raw_target,
        target_mask,
        expression_mean=mean_tensor,
        expression_scale=scale_tensor,
        huber_delta=1.0,
    )
    detection_logits, ordinal_logits, continuous_prediction = split_hybrid_prediction(
        prediction
    )
    detected = torch.sigmoid(detection_logits) >= 0.5
    count_state = decode_count_states(prediction)
    positive_state = decode_positive_states(ordinal_logits)
    predicted_log1p = (
        continuous_prediction.float() * scale_tensor.unsqueeze(0)
        + mean_tensor.unsqueeze(0)
    )
    reconstructed_log1p = torch.where(
        detected,
        predicted_log1p.clamp_min(0.0),
        torch.zeros_like(predicted_log1p),
    )
    reconstructed_count = torch.expm1(reconstructed_log1p)
    if not bool(torch.isfinite(reconstructed_count).all()):
        raise FloatingPointError("reconstructed count is non-finite")

    selected_target_count = raw_target[target_mask].detach().float().cpu().numpy()
    selected_target_state = tokenize_raw_counts(
        selected_target_count.reshape(-1, 1)
    ).reshape(-1)
    selected_prediction_state = (
        count_state[target_mask].detach().cpu().numpy().astype(np.int64, copy=False)
    )
    selected_target_detected = selected_target_count > 0
    selected_predicted_detected = (
        detected[target_mask].detach().cpu().numpy().astype(bool, copy=False)
    )
    metrics: dict[str, Any] = {
        "n_masked": loss.n_masked,
        "hybrid_loss": float(loss.total.detach().float().cpu()),
        "detection_bce": float(loss.detection.detach().float().cpu()),
        "ordinal_bce": float(loss.ordinal.detach().float().cpu()),
        "positive_continuous_huber": float(
            loss.positive_continuous_huber.detach().float().cpu()
        ),
    }
    metrics.update(
        _binary_metrics(selected_target_detected, selected_predicted_detected)
    )
    metrics.update(
        _state_metrics(
            selected_target_state,
            selected_prediction_state,
            num_states=NUM_COUNT_STATES,
            prefix="state8",
        )
    )

    positive = selected_target_detected
    positive_target_state = selected_target_state[positive]
    positive_prediction_state = (
        positive_state[target_mask]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)[positive]
    )
    state_error = np.abs(positive_prediction_state - positive_target_state)
    metrics.update(
        {
            "positive_state_exact_accuracy": float(np.mean(state_error == 0)),
            "positive_ordinal_mae": float(np.mean(state_error)),
            "positive_within_one_state_accuracy": float(
                np.mean(state_error <= 1)
            ),
        }
    )
    standardized_target = (
        torch.log1p(raw_target.float()) - mean_tensor.unsqueeze(0)
    ) / scale_tensor.unsqueeze(0)
    selected_continuous = continuous_prediction[target_mask].detach().float().cpu().numpy()
    selected_standardized = standardized_target[target_mask].detach().float().cpu().numpy()
    continuous_error = selected_continuous[positive] - selected_standardized[positive]
    metrics["positive_continuous_mae"] = float(
        np.mean(np.abs(continuous_error))
    )
    metrics["reconstructed_count_log1p_mae"] = float(
        torch.mean(
            torch.abs(
                reconstructed_log1p[target_mask]
                - torch.log1p(raw_target.float())[target_mask]
            )
        )
        .detach()
        .float()
        .cpu()
    )

    collapsed_target = np.minimum(selected_target_state, 3)
    collapsed_prediction = np.minimum(selected_prediction_state, 3)
    metrics.update(
        _state_metrics(
            collapsed_target,
            collapsed_prediction,
            num_states=4,
            prefix="collapsed4",
        )
    )
    collapsed_positive = collapsed_target > 0
    metrics["collapsed4_positive_exact_accuracy"] = float(
        np.mean(
            collapsed_target[collapsed_positive]
            == collapsed_prediction[collapsed_positive]
        )
    )

    # Reference heads are evaluated through the same fail-closed objective.
    reference_output = reference_prediction_tensor(
        references, num_nodes=n_nodes, device=raw_target.device
    )
    reference_loss = hybrid_count_hurdle_loss(
        reference_output,
        raw_target,
        target_mask,
        expression_mean=mean_tensor,
        expression_scale=scale_tensor,
        huber_delta=1.0,
    )
    gene_indices = (
        torch.arange(n_genes, device=raw_target.device)
        .unsqueeze(0)
        .expand(n_nodes, -1)[target_mask]
        .detach()
        .cpu()
        .numpy()
    )
    reference_state = references.count_state[gene_indices]
    reference_detected = references.detected_state[gene_indices]
    reference_positive_state = references.positive_state[gene_indices]
    reference_state_error = np.abs(
        reference_positive_state[positive] - positive_target_state
    )
    reference_continuous = references.positive_continuous_standardized[
        gene_indices
    ]
    reference_continuous_error = (
        reference_continuous[positive] - selected_standardized[positive]
    )
    reference_binary = _binary_metrics(
        selected_target_detected, reference_detected
    )
    metrics.update(
        {
            "reference_per_gene_hybrid_loss": float(
                reference_loss.total.detach().float().cpu()
            ),
            "reference_per_gene_detection_bce": float(
                reference_loss.detection.detach().float().cpu()
            ),
            "reference_per_gene_ordinal_bce": float(
                reference_loss.ordinal.detach().float().cpu()
            ),
            "reference_per_gene_positive_continuous_huber": float(
                reference_loss.positive_continuous_huber.detach().float().cpu()
            ),
            "reference_per_gene_detection_balanced_accuracy": float(
                reference_binary["detection_balanced_accuracy"]
            ),
            "reference_per_gene_positive_ordinal_mae": float(
                np.mean(reference_state_error)
            ),
            "reference_per_gene_positive_continuous_mae": float(
                np.mean(np.abs(reference_continuous_error))
            ),
            "reference_per_gene_state8_exact_accuracy": float(
                np.mean(reference_state == selected_target_state)
            ),
            "reference_all_zero_state8_exact_accuracy": float(
                np.mean(selected_target_state == 0)
            ),
            "reference_all_zero_collapsed4_exact_accuracy": float(
                np.mean(collapsed_target == 0)
            ),
        }
    )
    reference_state_metrics = _state_metrics(
        selected_target_state,
        reference_state,
        num_states=NUM_COUNT_STATES,
        prefix="reference_per_gene_state8",
    )
    metrics["reference_per_gene_state8_balanced_accuracy"] = (
        reference_state_metrics["reference_per_gene_state8_balanced_accuracy"]
    )
    all_zero_state_metrics = _state_metrics(
        selected_target_state,
        np.zeros_like(selected_target_state),
        num_states=NUM_COUNT_STATES,
        prefix="reference_all_zero_state8",
    )
    metrics["reference_all_zero_state8_balanced_accuracy"] = (
        all_zero_state_metrics["reference_all_zero_state8_balanced_accuracy"]
    )
    metrics["reference_all_zero_state8_support"] = all_zero_state_metrics[
        "reference_all_zero_state8_support"
    ]
    metrics["reference_all_zero_state8_recall"] = all_zero_state_metrics[
        "reference_all_zero_state8_recall"
    ]

    return HybridCountEvaluation(
        count_state=count_state.detach().cpu(),
        detected=detected.detach().cpu(),
        positive_continuous_standardized=continuous_prediction.detach()
        .float()
        .cpu(),
        reconstructed_count=reconstructed_count.detach().float().cpu(),
        metrics=metrics,
    )


def json_safe_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Reject NaN/Inf and convert NumPy scalars for archive serialization."""

    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("metric contains NaN or infinity")
            return value
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        raise TypeError(f"unsupported metric value type: {type(value).__name__}")

    converted = {str(key): convert(value) for key, value in metrics.items()}
    json.dumps(converted, sort_keys=True, allow_nan=False)
    return converted


__all__ = [
    "HybridCountEvaluation",
    "HybridCountReferences",
    "evaluate_hybrid_count_output",
    "fit_hybrid_count_references",
    "json_safe_metrics",
    "reference_prediction_tensor",
]
