"""Fixed-budget training and evaluation for hybrid raw-count full-core models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor, nn

from .hybrid_count import HybridCountLoss, hybrid_count_hurdle_loss
from .hybrid_count_metrics import (
    HybridCountEvaluation,
    HybridCountReferences,
    evaluate_hybrid_count_output,
)
from .masking import derive_mask_seed
from .training import (
    GraphSplitView,
    TrainingConfig,
    _array_checksum,
    _autocast_context,
    _forward_masked_targets,
    _make_grad_scaler,
    _model_dtype,
    _prepare_fixed_mask,
    _resolve_device,
    _to_device_view,
    apply_edge_dropout,
    make_epoch_mask,
    set_deterministic_seed,
)


@dataclass(frozen=True)
class HybridCountEpochRecord:
    epoch: int
    mask_mode: str
    mask_seed: int
    mask_checksum: str
    edge_dropout_seed: int
    edge_checksum: str
    n_masked_entries: int
    n_zero_targets: int
    n_positive_targets: int
    n_target_nodes: int
    n_edges_used: int
    train_hybrid_loss: float
    train_detection_bce: float
    train_ordinal_bce: float
    train_positive_continuous_huber: float
    gradient_norm: float
    duration_seconds: float
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class HybridCountTrainingResult:
    history: tuple[HybridCountEpochRecord, ...]
    final_epoch: int
    final_train_loss: float
    final_state_dict: Mapping[str, Tensor]
    final_state_checksum: str
    device: str
    fixed_epoch_budget: int
    checkpoint_policy: str = "final_epoch_no_validation_selection"
    graph_execution: str = "full_core_exact_no_neighbor_sampling"
    training_protocol: str = "held_in_full_core_hybrid_count_fixed_budget"

    def history_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.history]


@dataclass(frozen=True)
class HybridFixedMaskResult:
    target_nodes: Tensor
    target: Tensor
    target_mask: Tensor
    full_mask: Tensor
    evaluation: HybridCountEvaluation


@dataclass(frozen=True)
class PrecisionEquivalenceResult:
    mask_checksum: str
    fp32_total_loss: float
    amp_total_loss: float
    absolute_total_loss_discrepancy: float
    fp32_components: Mapping[str, float]
    amp_components: Mapping[str, float]
    amp_dtype: str
    peak_cuda_memory_bytes: int
    passed: bool


def _state_dict_checksum(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = torch.as_tensor(state_dict[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _loss_components(loss: HybridCountLoss) -> dict[str, float]:
    result = {
        "hybrid_loss": float(loss.total.detach().float().cpu()),
        "detection_bce": float(loss.detection.detach().float().cpu()),
        "ordinal_bce": float(loss.ordinal.detach().float().cpu()),
        "positive_continuous_huber": float(
            loss.positive_continuous_huber.detach().float().cpu()
        ),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("hybrid loss component is non-finite")
    return result


def fit_full_core_hybrid_count_model(
    model: nn.Module,
    fit_view: GraphSplitView,
    config: TrainingConfig,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> HybridCountTrainingResult:
    """Fit exactly the configured epochs and retain only the final state."""

    if fit_view.name not in {"fit", "full_core", "full-core"}:
        raise ValueError("fit_view must explicitly identify the held-in fit role")
    if config.restore_best:
        raise ValueError("hybrid full-core training requires restore_best=False")
    if config.huber_delta != 1.0:
        raise ValueError("the frozen positive Huber delta is exactly 1.0")
    if config.edge_dropout != 0.0:
        raise ValueError("the frozen hybrid campaign prohibits edge dropout")
    mean = torch.as_tensor(expression_mean, dtype=torch.float32)
    scale = torch.as_tensor(expression_scale, dtype=torch.float32)
    if mean.shape != (fit_view.expression.shape[1],) or scale.shape != (
        fit_view.expression.shape[1],
    ):
        raise ValueError("expression standardization must match fit genes")

    set_deterministic_seed(
        config.model_seed,
        deterministic=config.deterministic,
        warn_only=config.deterministic_warn_only,
    )
    device = _resolve_device(model, config.device)
    model.to(device)
    device_view = _to_device_view(
        fit_view, device=device, dtype=_model_dtype(model)
    )
    device_mean = mean.to(device=device)
    device_scale = scale.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = _make_grad_scaler(device, config.amp)
    history: list[HybridCountEpochRecord] = []
    undropped_edge_checksum = _array_checksum(
        np.ones(int(device_view.edge_index.shape[1]), dtype=np.bool_)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(config.max_epochs):
        started = time.monotonic()
        model.train()
        mask_batch = make_epoch_mask(fit_view, config, epoch)
        mask = torch.from_numpy(np.array(mask_batch.mask, copy=True)).to(
            device=device
        )
        edge_dropout_seed = derive_mask_seed(
            config.mask_seed, "edge-dropout", epoch
        )
        # The identity-preserving fast path keeps the exact receiver-layout
        # cache valid across all epochs.
        edges = device_view.edge_index
        edge_attributes = device_view.edge_attributes
        edge_checksum = undropped_edge_checksum

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(
            enabled=config.amp,
            device=device,
            dtype_name=config.amp_dtype,
        ):
            output, target, target_mask = _forward_masked_targets(
                model,
                device_view,
                mask,
                edge_index=edges,
                edge_attributes=edge_attributes,
            )
            loss = hybrid_count_hurdle_loss(
                output.prediction,
                target,
                target_mask,
                expression_mean=device_mean,
                expression_scale=device_scale,
                huber_delta=1.0,
            )
        components = _loss_components(loss)
        scaler.scale(loss.total).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm).detach().cpu()):
            raise FloatingPointError(
                f"non-finite gradient norm at epoch {epoch}"
            )
        scaler.step(optimizer)
        scaler.update()
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        history.append(
            HybridCountEpochRecord(
                epoch=epoch,
                mask_mode=mask_batch.spec.mode,
                mask_seed=mask_batch.seed,
                mask_checksum=_array_checksum(mask_batch.mask),
                edge_dropout_seed=edge_dropout_seed,
                edge_checksum=edge_checksum,
                n_masked_entries=loss.n_masked,
                n_zero_targets=loss.n_zero,
                n_positive_targets=loss.n_positive,
                n_target_nodes=mask_batch.n_selected_nodes,
                n_edges_used=int(edges.shape[1]),
                train_hybrid_loss=components["hybrid_loss"],
                train_detection_bce=components["detection_bce"],
                train_ordinal_bce=components["ordinal_bce"],
                train_positive_continuous_huber=components[
                    "positive_continuous_huber"
                ],
                gradient_norm=float(gradient_norm.detach().float().cpu()),
                duration_seconds=time.monotonic() - started,
                peak_cuda_memory_bytes=peak_memory,
            )
        )

    if len(history) != config.max_epochs:
        raise RuntimeError("fixed-budget training did not complete every epoch")
    final_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    return HybridCountTrainingResult(
        history=tuple(history),
        final_epoch=config.max_epochs - 1,
        final_train_loss=history[-1].train_hybrid_loss,
        final_state_dict=final_state,
        final_state_checksum=_state_dict_checksum(final_state),
        device=str(device),
        fixed_epoch_budget=config.max_epochs,
    )


def evaluate_fixed_hybrid_count_mask(
    model: nn.Module,
    view: GraphSplitView,
    mask: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
    references: HybridCountReferences,
    device: Optional[str] = None,
    amp: bool = False,
    amp_dtype: str = "auto",
) -> HybridFixedMaskResult:
    """Evaluate one immutable mask without using it for model selection."""

    fixed_mask = _prepare_fixed_mask(view, mask)
    resolved_device = _resolve_device(model, device)
    model.to(resolved_device)
    device_view = _to_device_view(
        view, device=resolved_device, dtype=_model_dtype(model)
    )
    mask_device = fixed_mask.to(device=resolved_device)
    was_training = model.training
    model.eval()
    with torch.no_grad(), _autocast_context(
        enabled=amp,
        device=resolved_device,
        dtype_name=str(amp_dtype).lower(),
    ):
        output, target, target_mask = _forward_masked_targets(
            model,
            device_view,
            mask_device,
            edge_index=device_view.edge_index,
            edge_attributes=device_view.edge_attributes,
        )
        evaluation = evaluate_hybrid_count_output(
            output.prediction,
            target,
            target_mask,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
            references=references,
        )
    model.train(was_training)
    target_nodes = mask_device.any(dim=1).nonzero(
        as_tuple=False
    ).flatten()
    return HybridFixedMaskResult(
        target_nodes=target_nodes.detach().cpu(),
        target=target.detach().float().cpu(),
        target_mask=target_mask.detach().cpu(),
        full_mask=fixed_mask.clone(),
        evaluation=evaluation,
    )


def compare_fp32_amp_loss(
    model: nn.Module,
    view: GraphSplitView,
    mask: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
    device: str,
    amp_dtype: str = "auto",
    maximum_discrepancy: float = 1e-3,
) -> PrecisionEquivalenceResult:
    """Compare identical weights and a frozen mask under FP32 and AMP."""

    if maximum_discrepancy < 0 or not math.isfinite(maximum_discrepancy):
        raise ValueError("maximum_discrepancy must be finite and nonnegative")
    fixed_mask = _prepare_fixed_mask(view, mask)
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("the production precision diagnostic requires CUDA")
    model.to(resolved_device)
    device_view = _to_device_view(
        view, device=resolved_device, dtype=_model_dtype(model)
    )
    mask_device = fixed_mask.to(device=resolved_device)
    mean = torch.as_tensor(
        expression_mean, dtype=torch.float32, device=resolved_device
    )
    scale = torch.as_tensor(
        expression_scale, dtype=torch.float32, device=resolved_device
    )
    was_training = model.training
    model.eval()
    torch.cuda.reset_peak_memory_stats(resolved_device)

    def one(enabled: bool) -> HybridCountLoss:
        with torch.no_grad(), _autocast_context(
            enabled=enabled,
            device=resolved_device,
            dtype_name=amp_dtype,
        ):
            output, target, target_mask = _forward_masked_targets(
                model,
                device_view,
                mask_device,
                edge_index=device_view.edge_index,
                edge_attributes=device_view.edge_attributes,
            )
            return hybrid_count_hurdle_loss(
                output.prediction,
                target,
                target_mask,
                expression_mean=mean,
                expression_scale=scale,
                huber_delta=1.0,
            )

    fp32 = one(False)
    amp = one(True)
    model.train(was_training)
    fp32_values = _loss_components(fp32)
    amp_values = _loss_components(amp)
    discrepancy = abs(fp32_values["hybrid_loss"] - amp_values["hybrid_loss"])
    return PrecisionEquivalenceResult(
        mask_checksum=_array_checksum(fixed_mask),
        fp32_total_loss=fp32_values["hybrid_loss"],
        amp_total_loss=amp_values["hybrid_loss"],
        absolute_total_loss_discrepancy=discrepancy,
        fp32_components=fp32_values,
        amp_components=amp_values,
        amp_dtype=amp_dtype,
        peak_cuda_memory_bytes=int(
            torch.cuda.max_memory_allocated(resolved_device)
        ),
        passed=discrepancy <= maximum_discrepancy,
    )


__all__ = [
    "HybridCountEpochRecord",
    "HybridCountTrainingResult",
    "HybridFixedMaskResult",
    "PrecisionEquivalenceResult",
    "compare_fp32_amp_loss",
    "evaluate_fixed_hybrid_count_mask",
    "fit_full_core_hybrid_count_model",
]
