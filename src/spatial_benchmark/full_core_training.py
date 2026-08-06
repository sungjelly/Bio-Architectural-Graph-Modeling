"""Fixed-budget training for explicitly held-in full-core capacity studies.

This module is intentionally separate from :mod:`spatial_benchmark.training`.
The standard trainer selects a checkpoint with a disjoint validation graph.
Here every node belongs to the fit graph, the epoch budget is fixed, and the
final epoch is retained.  Optional fixed-mask diagnostics never select or
restore a checkpoint.
"""

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
from torch.nn import functional as F

from .masking import derive_mask_seed
from .metrics import masked_huber_loss
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
class FullCoreEpochRecord:
    """One fixed-budget optimization epoch."""

    epoch: int
    mask_mode: str
    mask_seed: int
    mask_checksum: str
    edge_dropout_seed: int
    edge_checksum: str
    n_masked_entries: int
    n_target_nodes: int
    n_edges_used: int
    train_loss: float
    diagnostic_loss: Optional[float]
    gradient_norm: float
    duration_seconds: float
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class FullCoreTrainingResult:
    """Result of a no-validation, final-checkpoint training run."""

    history: tuple[FullCoreEpochRecord, ...]
    final_epoch: int
    final_train_loss: float
    final_state_dict: Mapping[str, Tensor]
    final_state_checksum: str
    device: str
    fixed_epoch_budget: int
    checkpoint_policy: str = "final_epoch_no_validation_selection"
    graph_execution: str = "full_core_exact_no_neighbor_sampling"
    training_protocol: str = "held_in_full_core_fixed_budget"
    diagnostic_mask_checksum: Optional[str] = None
    diagnostic_every: Optional[int] = None

    def history_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.history]


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


def _diagnostic_loss(
    model: nn.Module,
    device_view: Any,
    fixed_mask_cpu: Tensor,
    *,
    config: TrainingConfig,
    device: torch.device,
    objective: str,
    num_expression_tokens: int | None,
) -> float:
    was_training = model.training
    model.eval()
    mask = fixed_mask_cpu.to(device=device)
    with torch.no_grad(), _autocast_context(
        enabled=config.amp,
        device=device,
        dtype_name=config.amp_dtype,
    ):
        output, target, target_mask = _forward_masked_targets(
            model,
            device_view,
            mask,
            edge_index=device_view.edge_index,
            edge_attributes=device_view.edge_attributes,
        )
        loss = _masked_objective_loss(
            output=output,
            target=target,
            target_mask=target_mask,
            objective=objective,
            huber_delta=config.huber_delta,
            num_expression_tokens=num_expression_tokens,
        )
    model.train(was_training)
    value = float(loss.detach().float().cpu())
    if not math.isfinite(value):
        raise FloatingPointError("non-finite held-in diagnostic loss")
    return value


def _masked_objective_loss(
    *,
    output: Any,
    target: Tensor,
    target_mask: Tensor,
    objective: str,
    huber_delta: float,
    num_expression_tokens: int | None,
) -> Tensor:
    """Evaluate one explicitly selected masked-entry objective."""

    if objective == "masked_huber":
        if num_expression_tokens is not None:
            raise ValueError(
                "num_expression_tokens is valid only for token cross-entropy"
            )
        return masked_huber_loss(
            target,
            output.prediction,
            target_mask,
            delta=huber_delta,
        )
    if objective != "masked_token_cross_entropy":
        raise ValueError(
            "objective must be 'masked_huber' or "
            "'masked_token_cross_entropy'"
        )
    if (
        isinstance(num_expression_tokens, bool)
        or not isinstance(num_expression_tokens, int)
        or num_expression_tokens < 2
    ):
        raise ValueError(
            "token cross-entropy requires num_expression_tokens >= 2"
        )
    expected_shape = (*target.shape, num_expression_tokens)
    if tuple(output.prediction.shape) != expected_shape:
        raise ValueError(
            "token logits shape mismatch: expected "
            f"{expected_shape}, got {tuple(output.prediction.shape)}"
        )
    if target_mask.shape != target.shape or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be boolean and match token targets")
    rounded = target.round()
    if not bool(torch.equal(target, rounded)):
        raise ValueError("token targets must contain integral IDs")
    token_target = rounded.to(dtype=torch.long)
    if bool((token_target < 0).any()) or bool(
        (token_target >= num_expression_tokens).any()
    ):
        raise ValueError("token target lies outside the output vocabulary")
    selected_logits = output.prediction[target_mask]
    selected_target = token_target[target_mask]
    if selected_target.numel() == 0:
        raise ValueError("token objective requires at least one masked target")
    return F.cross_entropy(selected_logits, selected_target)


def fit_full_core_model(
    model: nn.Module,
    fit_view: GraphSplitView,
    config: TrainingConfig,
    *,
    diagnostic_mask: Any | None = None,
    diagnostic_every: int | None = None,
    objective: str = "masked_huber",
    num_expression_tokens: int | None = None,
) -> FullCoreTrainingResult:
    """Fit exactly ``max_epochs`` on one graph and retain the final state.

    There is no validation view, early stopping, best-epoch restoration, or
    checkpoint selection.  ``diagnostic_mask`` is optional held-in monitoring
    over the same cells and is never used to choose model state.
    """

    if objective not in {"masked_huber", "masked_token_cross_entropy"}:
        raise ValueError(
            "objective must be 'masked_huber' or "
            "'masked_token_cross_entropy'"
        )
    if objective == "masked_huber":
        if num_expression_tokens is not None:
            raise ValueError(
                "num_expression_tokens is valid only for token cross-entropy"
            )
    elif (
        isinstance(num_expression_tokens, bool)
        or not isinstance(num_expression_tokens, int)
        or num_expression_tokens < 2
    ):
        raise ValueError(
            "token cross-entropy requires num_expression_tokens >= 2"
        )

    if fit_view.name not in {"fit", "full_core", "full-core"}:
        raise ValueError(
            "fit_view.name must explicitly identify the held-in fit role"
        )
    if config.restore_best:
        raise ValueError(
            "full-core fixed-budget training requires restore_best=False"
        )
    if diagnostic_every is not None:
        diagnostic_every = int(diagnostic_every)
        if diagnostic_every <= 0:
            raise ValueError("diagnostic_every must be positive")
        if diagnostic_mask is None:
            raise ValueError(
                "diagnostic_every requires an explicit held-in diagnostic mask"
            )
    elif diagnostic_mask is not None:
        raise ValueError(
            "diagnostic_mask requires diagnostic_every so its cost is explicit"
        )

    fixed_diagnostic = (
        None
        if diagnostic_mask is None
        else _prepare_fixed_mask(fit_view, diagnostic_mask)
    )
    set_deterministic_seed(
        config.model_seed,
        deterministic=config.deterministic,
        warn_only=config.deterministic_warn_only,
    )
    device = _resolve_device(model, config.device)
    model.to(device)
    device_view = _to_device_view(
        fit_view,
        device=device,
        dtype=_model_dtype(model),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = _make_grad_scaler(device, config.amp)
    history: list[FullCoreEpochRecord] = []
    undropped_edge_checksum = (
        _array_checksum(
            np.ones(int(device_view.edge_index.shape[1]), dtype=np.bool_)
        )
        if config.edge_dropout == 0.0
        else None
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
        if config.edge_dropout == 0.0:
            # Preserve tensor identity so dense receiver-layout caches remain
            # valid, and avoid copying every edge through a boolean index.
            edges = device_view.edge_index
            edge_attributes = device_view.edge_attributes
            edge_checksum = str(undropped_edge_checksum)
        else:
            edges, edge_attributes, keep = apply_edge_dropout(
                device_view.edge_index,
                device_view.edge_attributes,
                probability=config.edge_dropout,
                seed=edge_dropout_seed,
            )
            edge_checksum = _array_checksum(keep)

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
            loss = _masked_objective_loss(
                output=output,
                target=target,
                target_mask=target_mask,
                objective=objective,
                huber_delta=config.huber_delta,
                num_expression_tokens=num_expression_tokens,
            )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
        scaler.scale(loss).backward()
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

        diagnostic_loss: float | None = None
        should_diagnose = (
            fixed_diagnostic is not None
            and diagnostic_every is not None
            and (
                epoch == 0
                or (epoch + 1) % diagnostic_every == 0
                or epoch + 1 == config.max_epochs
            )
        )
        if should_diagnose:
            diagnostic_loss = _diagnostic_loss(
                model,
                device_view,
                fixed_diagnostic,
                config=config,
                device=device,
                objective=objective,
                num_expression_tokens=num_expression_tokens,
            )

        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        history.append(
            FullCoreEpochRecord(
                epoch=epoch,
                mask_mode=mask_batch.spec.mode,
                mask_seed=mask_batch.seed,
                mask_checksum=_array_checksum(mask_batch.mask),
                edge_dropout_seed=edge_dropout_seed,
                edge_checksum=edge_checksum,
                n_masked_entries=mask_batch.n_masked_entries,
                n_target_nodes=mask_batch.n_selected_nodes,
                n_edges_used=int(edges.shape[1]),
                train_loss=float(loss.detach().float().cpu()),
                diagnostic_loss=diagnostic_loss,
                gradient_norm=float(gradient_norm.detach().float().cpu()),
                duration_seconds=time.monotonic() - started,
                peak_cuda_memory_bytes=peak_memory,
            )
        )

    if len(history) != config.max_epochs:
        raise RuntimeError("fixed-budget training did not complete every epoch")
    final_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    return FullCoreTrainingResult(
        history=tuple(history),
        final_epoch=config.max_epochs - 1,
        final_train_loss=history[-1].train_loss,
        final_state_dict=final_state,
        final_state_checksum=_state_dict_checksum(final_state),
        device=str(device),
        fixed_epoch_budget=config.max_epochs,
        training_protocol=(
            "held_in_full_core_fixed_budget_token_classification"
            if objective == "masked_token_cross_entropy"
            else "held_in_full_core_fixed_budget"
        ),
        diagnostic_mask_checksum=(
            None
            if fixed_diagnostic is None
            else _array_checksum(fixed_diagnostic)
        ),
        diagnostic_every=diagnostic_every,
    )


__all__ = [
    "FullCoreEpochRecord",
    "FullCoreTrainingResult",
    "fit_full_core_model",
]
