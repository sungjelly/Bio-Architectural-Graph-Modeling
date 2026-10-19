"""Graphless full-core continuous-hurdle representation model and training.

This module intentionally has no edge-index, edge-attribute, or graph-builder
surface.  It estimates held-in masked-expression capacity from the observed
genes and permitted morphology/imaging covariates of the same cell.  Spatial
coordinates are retained only for constructing separately reported block
masks; they are never passed to the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hurdle_continuous import (
    HurdleContinuousEvaluation,
    HurdleContinuousLoss,
    NUM_HURDLE_CONTINUOUS_CHANNELS,
    evaluate_hurdle_continuous_output,
    split_hurdle_continuous_prediction,
)
from .hybrid_count import (
    HybridCountNodeEncoder,
    standardized_log1p_counts,
    tokenize_raw_count_tensor,
)
from .models import ResidualFeedForward
from .training import (
    TrainingConfig,
    _autocast_context,
    _make_grad_scaler,
    _model_dtype,
    _prepare_fixed_mask,
    _resolve_device,
    make_epoch_mask,
    set_deterministic_seed,
)


@dataclass(frozen=True)
class SelfHurdleSplitView:
    """One graphless, held-in core used for masking and self-only prediction."""

    expression: Tensor
    coordinates_um: Tensor
    node_covariates: Optional[Tensor] = None
    block_ids: Optional[Any] = None
    name: str = "fit"

    def __post_init__(self) -> None:
        expression = torch.as_tensor(self.expression)
        if (
            expression.ndim != 2
            or not expression.is_floating_point()
            or expression.shape[0] == 0
            or expression.shape[1] == 0
        ):
            raise TypeError(
                "expression must be a nonempty floating [nodes, genes] tensor"
            )
        tokenize_raw_count_tensor(expression)
        coordinates = torch.as_tensor(
            self.coordinates_um, dtype=torch.float64, device="cpu"
        )
        if coordinates.shape != (expression.shape[0], 2) or not bool(
            torch.isfinite(coordinates).all()
        ):
            raise ValueError(
                "coordinates_um must be finite with shape [nodes, 2]"
            )
        covariates = self.node_covariates
        if covariates is not None:
            covariates = torch.as_tensor(covariates)
            if (
                covariates.ndim != 2
                or covariates.shape[0] != expression.shape[0]
                or not covariates.is_floating_point()
                or not bool(torch.isfinite(covariates).all())
            ):
                raise TypeError(
                    "node_covariates must be finite floating [nodes, features]"
                )
        blocks = self.block_ids
        if blocks is not None:
            blocks = np.asarray(
                blocks.detach().cpu().numpy()
                if torch.is_tensor(blocks)
                else blocks
            )
            if blocks.shape != (expression.shape[0],):
                raise ValueError("block_ids must have shape [nodes]")
            blocks = np.array(blocks, copy=True)
        name = str(self.name).strip()
        if name not in {"fit", "full_core", "full-core"}:
            raise ValueError("self-hurdle view must explicitly identify fit")
        object.__setattr__(self, "expression", expression.contiguous())
        object.__setattr__(self, "coordinates_um", coordinates.contiguous())
        object.__setattr__(
            self,
            "node_covariates",
            None if covariates is None else covariates.contiguous(),
        )
        object.__setattr__(self, "block_ids", blocks)
        object.__setattr__(self, "name", name)

    @property
    def num_nodes(self) -> int:
        return int(self.expression.shape[0])

    @property
    def num_genes(self) -> int:
        return int(self.expression.shape[1])

    @property
    def node_covariate_dim(self) -> int:
        return (
            0
            if self.node_covariates is None
            else int(self.node_covariates.shape[1])
        )


class _SelfHurdleDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        decoder_dim: int,
        num_genes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_genes = int(num_genes)
        self.linear_in = nn.Linear(hidden_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(
            decoder_dim,
            num_genes * NUM_HURDLE_CONTINUOUS_CHANNELS,
        )

    def forward(self, embedding: Tensor) -> Tensor:
        flat = self.linear_out(
            self.dropout(F.gelu(self.linear_in(embedding)))
        )
        return flat.reshape(
            embedding.shape[0],
            self.num_genes,
            NUM_HURDLE_CONTINUOUS_CHANNELS,
        )


class SelfHurdleModel(nn.Module):
    """Large graphless raw-count encoder with an exact two-channel decoder."""

    def __init__(
        self,
        *,
        num_genes: int,
        expression_mean: Any,
        expression_scale: Any,
        node_covariate_dim: int = 0,
        hidden_dim: int = 768,
        decoder_dim: int = 768,
        ffn_dim: int = 1536,
        residual_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_genes", num_genes),
            ("hidden_dim", hidden_dim),
            ("decoder_dim", decoder_dim),
            ("ffn_dim", ffn_dim),
            ("residual_blocks", residual_blocks),
        ):
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if int(node_covariate_dim) < 0:
            raise ValueError("node_covariate_dim cannot be negative")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        self.num_genes = int(num_genes)
        self.node_covariate_dim = int(node_covariate_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_channels = NUM_HURDLE_CONTINUOUS_CHANNELS
        self.uses_graph_inputs = False
        self.uses_edge_inputs = False
        self.encoder = HybridCountNodeEncoder(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
            dropout=dropout,
        )
        self.residual_blocks = nn.ModuleList(
            [
                ResidualFeedForward(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(int(residual_blocks))
            ]
        )
        self.decoder = _SelfHurdleDecoder(
            hidden_dim=hidden_dim,
            decoder_dim=decoder_dim,
            num_genes=num_genes,
            dropout=dropout,
        )

    @staticmethod
    def _target_selection(
        target_nodes: Optional[Tensor | Sequence[int]],
        *,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[Tensor]:
        if target_nodes is None:
            return None
        selected = torch.as_tensor(target_nodes, device=device)
        if (
            selected.ndim != 1
            or selected.dtype == torch.bool
            or selected.is_floating_point()
            or selected.is_complex()
        ):
            raise TypeError("target_nodes must be a one-dimensional integer")
        selected = selected.to(dtype=torch.long)
        if selected.numel() == 0:
            raise ValueError("target_nodes cannot be empty")
        if bool((selected < 0).any()) or bool((selected >= num_nodes).any()):
            raise ValueError("target_nodes contains an out-of-range index")
        if torch.unique(selected).numel() != selected.numel():
            raise ValueError("target_nodes cannot contain duplicates")
        return selected

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        *,
        node_covariates: Optional[Tensor] = None,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
    ) -> Tensor:
        """Predict selected cells using only their own permitted inputs."""

        if input_expression.ndim != 2:
            raise ValueError("input_expression must have shape [nodes, genes]")
        selected = self._target_selection(
            target_nodes,
            num_nodes=int(input_expression.shape[0]),
            device=input_expression.device,
        )
        if selected is not None:
            input_expression = input_expression.index_select(0, selected)
            gene_mask = gene_mask.index_select(0, selected)
            if node_covariates is not None:
                node_covariates = node_covariates.index_select(0, selected)
        embedding = self.encoder(
            input_expression,
            gene_mask,
            node_covariates,
        )
        for block in self.residual_blocks:
            embedding = block(embedding)
        return self.decoder(embedding)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@dataclass(frozen=True)
class SelfHurdleEpochRecord:
    epoch: int
    mask_mode: str
    mask_seed: int
    mask_checksum: str
    n_masked_entries: int
    n_zero_targets: int
    n_positive_targets: int
    n_target_nodes: int
    n_target_batches: int
    train_hurdle_loss: float
    train_detection_bce: float
    train_positive_continuous_huber: float
    gradient_norm: float
    duration_seconds: float
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class SelfHurdleTrainingResult:
    history: tuple[SelfHurdleEpochRecord, ...]
    final_epoch: int
    final_train_loss: float
    final_state_dict: Mapping[str, Tensor]
    final_state_checksum: str
    device: str
    fixed_epoch_budget: int
    target_node_batch_size: int
    checkpoint_policy: str = "final_epoch_no_validation_selection"
    training_protocol: str = "held_in_self_hurdle_fixed_budget"
    graph_execution: str = "none_graph_inputs_prohibited"

    def history_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.history]


@dataclass(frozen=True)
class SelfHurdleFixedMaskResult:
    target_nodes: Tensor
    target: Tensor
    target_mask: Tensor
    full_mask: Tensor
    evaluation: HurdleContinuousEvaluation


@dataclass(frozen=True)
class SelfHurdlePrecisionResult:
    mask_checksum: str
    fp32_total_loss: float
    amp_total_loss: float
    absolute_total_loss_discrepancy: float
    amp_dtype: str
    target_node_batch_size: int
    peak_cuda_memory_bytes: int
    passed: bool


@dataclass(frozen=True)
class _DeviceSelfView:
    expression: Tensor
    node_covariates: Optional[Tensor]


def _array_checksum(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _state_dict_checksum(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _validated_standardization(
    expression_mean: Any,
    expression_scale: Any,
    *,
    num_genes: int,
) -> tuple[Tensor, Tensor]:
    mean = torch.as_tensor(expression_mean, dtype=torch.float32, device="cpu")
    scale = torch.as_tensor(expression_scale, dtype=torch.float32, device="cpu")
    if mean.shape != (num_genes,) or scale.shape != (num_genes,):
        raise ValueError("expression standardization must have shape [genes]")
    if not bool(torch.isfinite(mean).all()) or not bool(
        torch.isfinite(scale).all()
    ):
        raise ValueError("expression standardization must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("expression_scale must be strictly positive")
    return mean.contiguous(), scale.contiguous()


def _to_device_view(
    view: SelfHurdleSplitView,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _DeviceSelfView:
    return _DeviceSelfView(
        expression=view.expression.to(device=device, dtype=dtype),
        node_covariates=(
            None
            if view.node_covariates is None
            else view.node_covariates.to(device=device, dtype=dtype)
        ),
    )


def _target_node_batches(mask: Tensor, batch_size: int) -> list[Tensor]:
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("target_node_batch_size must be positive")
    nodes = mask.any(dim=1).nonzero(as_tuple=False).flatten()
    if nodes.numel() == 0:
        raise ValueError("mask selects no target node")
    return list(nodes.split(int(batch_size)))


def _global_mask_support(
    expression: Tensor,
    mask: Tensor,
) -> tuple[int, int, int]:
    selected = expression[mask]
    n_masked = int(selected.numel())
    n_positive = int((selected > 0).sum().detach().cpu())
    n_zero = n_masked - n_positive
    if n_zero <= 0 or n_positive <= 0:
        raise ValueError("balanced hurdle training needs zero and positive strata")
    return n_masked, n_zero, n_positive


def _weighted_batch_loss(
    prediction: Tensor,
    target: Tensor,
    target_mask: Tensor,
    *,
    expression_mean: Tensor,
    expression_scale: Tensor,
    global_zero: int,
    global_positive: int,
) -> HurdleContinuousLoss:
    detection, continuous = split_hurdle_continuous_prediction(prediction)
    selected_counts = target[target_mask]
    zero = selected_counts == 0
    positive = selected_counts > 0
    selected_detection = detection[target_mask].float()
    zero_sum = selected_detection.sum() * 0.0
    positive_sum = selected_detection.sum() * 0.0
    if bool(zero.any()):
        zero_sum = F.binary_cross_entropy_with_logits(
            selected_detection[zero],
            torch.zeros_like(selected_detection[zero]),
            reduction="sum",
        )
    if bool(positive.any()):
        positive_sum = F.binary_cross_entropy_with_logits(
            selected_detection[positive],
            torch.ones_like(selected_detection[positive]),
            reduction="sum",
        )
    detection_loss = 0.5 * (
        zero_sum / float(global_zero)
        + positive_sum / float(global_positive)
    )
    standardized = standardized_log1p_counts(
        target,
        expression_mean,
        expression_scale,
    )[target_mask]
    selected_continuous = continuous[target_mask].float()
    huber_sum = selected_continuous.sum() * 0.0
    if bool(positive.any()):
        huber_sum = F.huber_loss(
            selected_continuous[positive],
            standardized[positive].float(),
            delta=1.0,
            reduction="sum",
        )
    huber = huber_sum / float(global_positive)
    total = 0.5 * (detection_loss + huber)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("self-hurdle batch loss is non-finite")
    return HurdleContinuousLoss(
        total=total,
        detection=detection_loss,
        positive_continuous_huber=huber,
        n_masked=int(selected_counts.numel()),
        n_zero=int(zero.sum().detach().cpu()),
        n_positive=int(positive.sum().detach().cpu()),
    )


def _forward_batch(
    model: nn.Module,
    view: _DeviceSelfView,
    mask: Tensor,
    target_nodes: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    prediction = model(
        view.expression,
        mask,
        node_covariates=view.node_covariates,
        target_nodes=target_nodes,
    )
    target = view.expression.index_select(0, target_nodes)
    target_mask = mask.index_select(0, target_nodes)
    return prediction, target, target_mask


def fit_full_core_self_hurdle_model(
    model: nn.Module,
    fit_view: SelfHurdleSplitView,
    config: TrainingConfig,
    *,
    expression_mean: Any,
    expression_scale: Any,
    target_node_batch_size: int = 16_384,
) -> SelfHurdleTrainingResult:
    """Fit every configured epoch without validation or checkpoint selection."""

    if config.restore_best:
        raise ValueError("self-hurdle training requires restore_best=False")
    if config.edge_dropout != 0.0:
        raise ValueError("self-hurdle training prohibits edge dropout")
    if config.huber_delta != 1.0:
        raise ValueError("self-hurdle positive Huber delta must equal 1.0")
    mean, scale = _validated_standardization(
        expression_mean,
        expression_scale,
        num_genes=fit_view.num_genes,
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
    mean = mean.to(device=device)
    scale = scale.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = _make_grad_scaler(device, config.amp)
    history: list[SelfHurdleEpochRecord] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(config.max_epochs):
        started = time.monotonic()
        model.train()
        batch = make_epoch_mask(fit_view, config, epoch)
        mask = torch.from_numpy(np.array(batch.mask, copy=True)).to(
            device=device
        )
        n_masked, n_zero, n_positive = _global_mask_support(
            device_view.expression,
            mask,
        )
        target_batches = _target_node_batches(mask, target_node_batch_size)
        totals = {
            "total": 0.0,
            "detection": 0.0,
            "positive": 0.0,
        }
        observed = [0, 0, 0]
        optimizer.zero_grad(set_to_none=True)
        for target_nodes in target_batches:
            with _autocast_context(
                enabled=config.amp,
                device=device,
                dtype_name=config.amp_dtype,
            ):
                prediction, target, target_mask = _forward_batch(
                    model,
                    device_view,
                    mask,
                    target_nodes,
                )
                loss = _weighted_batch_loss(
                    prediction,
                    target,
                    target_mask,
                    expression_mean=mean,
                    expression_scale=scale,
                    global_zero=n_zero,
                    global_positive=n_positive,
                )
            totals["total"] += float(loss.total.detach().float().cpu())
            totals["detection"] += float(
                loss.detection.detach().float().cpu()
            )
            totals["positive"] += float(
                loss.positive_continuous_huber.detach().float().cpu()
            )
            observed[0] += loss.n_masked
            observed[1] += loss.n_zero
            observed[2] += loss.n_positive
            scaler.scale(loss.total).backward()
        if tuple(observed) != (n_masked, n_zero, n_positive):
            raise RuntimeError("self-hurdle target batches did not cover mask")
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm).detach().cpu()):
            raise FloatingPointError("self-hurdle gradient norm is non-finite")
        scaler.step(optimizer)
        scaler.update()
        peak = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        history.append(
            SelfHurdleEpochRecord(
                epoch=epoch,
                mask_mode=batch.spec.mode,
                mask_seed=batch.seed,
                mask_checksum=_array_checksum(batch.mask),
                n_masked_entries=n_masked,
                n_zero_targets=n_zero,
                n_positive_targets=n_positive,
                n_target_nodes=batch.n_selected_nodes,
                n_target_batches=len(target_batches),
                train_hurdle_loss=totals["total"],
                train_detection_bce=totals["detection"],
                train_positive_continuous_huber=totals["positive"],
                gradient_norm=float(gradient_norm.detach().float().cpu()),
                duration_seconds=time.monotonic() - started,
                peak_cuda_memory_bytes=peak,
            )
        )

    final_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    for name, value in final_state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"final state {name!r} is non-finite")
    return SelfHurdleTrainingResult(
        history=tuple(history),
        final_epoch=config.max_epochs - 1,
        final_train_loss=history[-1].train_hurdle_loss,
        final_state_dict=final_state,
        final_state_checksum=_state_dict_checksum(final_state),
        device=str(device),
        fixed_epoch_budget=config.max_epochs,
        target_node_batch_size=int(target_node_batch_size),
    )


def _collect_predictions(
    model: nn.Module,
    view: _DeviceSelfView,
    mask: Tensor,
    *,
    target_node_batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    target_nodes = _target_node_batches(mask, target_node_batch_size)
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    target_masks: list[Tensor] = []
    nodes: list[Tensor] = []
    for selected in target_nodes:
        prediction, target, target_mask = _forward_batch(
            model, view, mask, selected
        )
        predictions.append(prediction.detach().float().cpu())
        targets.append(target.detach().float().cpu())
        target_masks.append(target_mask.detach().cpu())
        nodes.append(selected.detach().cpu())
    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(target_masks, dim=0),
        torch.cat(nodes, dim=0),
    )


def evaluate_fixed_self_hurdle_mask(
    model: nn.Module,
    fit_view: SelfHurdleSplitView,
    fixed_mask: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
    target_node_batch_size: int = 16_384,
    device: Optional[str] = None,
    amp: bool = False,
    amp_dtype: str = "auto",
) -> SelfHurdleFixedMaskResult:
    """Evaluate one immutable held-in mask without graph construction."""

    mask = _prepare_fixed_mask(fit_view, fixed_mask)
    resolved_device = _resolve_device(model, device)
    model.to(resolved_device)
    device_view = _to_device_view(
        fit_view,
        device=resolved_device,
        dtype=_model_dtype(model),
    )
    device_mask = mask.to(device=resolved_device)
    model.eval()
    with torch.no_grad(), _autocast_context(
        enabled=amp,
        device=resolved_device,
        dtype_name=amp_dtype,
    ):
        prediction, target, target_mask, target_nodes = _collect_predictions(
            model,
            device_view,
            device_mask,
            target_node_batch_size=target_node_batch_size,
        )
    evaluation = evaluate_hurdle_continuous_output(
        prediction,
        target,
        target_mask,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
    )
    return SelfHurdleFixedMaskResult(
        target_nodes=target_nodes,
        target=target,
        target_mask=target_mask,
        full_mask=mask,
        evaluation=evaluation,
    )


def compare_self_hurdle_fp32_amp_loss(
    model: nn.Module,
    fit_view: SelfHurdleSplitView,
    fixed_mask: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
    device: str = "cuda",
    target_node_batch_size: int = 16_384,
    amp_dtype: str = "auto",
    maximum_discrepancy: float = 1e-3,
) -> SelfHurdlePrecisionResult:
    """Compare deterministic initial FP32 and AMP loss on one fixed mask."""

    if maximum_discrepancy < 0 or not math.isfinite(maximum_discrepancy):
        raise ValueError("maximum_discrepancy must be finite and nonnegative")
    mask = _prepare_fixed_mask(fit_view, fixed_mask)
    resolved_device = _resolve_device(model, device)
    model.to(resolved_device)
    view = _to_device_view(
        fit_view,
        device=resolved_device,
        dtype=_model_dtype(model),
    )
    device_mask = mask.to(device=resolved_device)
    model.eval()
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    losses: list[float] = []
    for amp in (False, True):
        with torch.no_grad(), _autocast_context(
            enabled=amp,
            device=resolved_device,
            dtype_name=amp_dtype,
        ):
            prediction, target, target_mask, _ = _collect_predictions(
                model,
                view,
                device_mask,
                target_node_batch_size=target_node_batch_size,
            )
        loss = evaluate_hurdle_continuous_output(
            prediction,
            target,
            target_mask,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
        ).metrics["hurdle_loss"]
        losses.append(float(loss))
    discrepancy = abs(losses[0] - losses[1])
    peak = (
        int(torch.cuda.max_memory_allocated(resolved_device))
        if resolved_device.type == "cuda"
        else 0
    )
    return SelfHurdlePrecisionResult(
        mask_checksum=_array_checksum(mask.numpy()),
        fp32_total_loss=losses[0],
        amp_total_loss=losses[1],
        absolute_total_loss_discrepancy=discrepancy,
        amp_dtype=amp_dtype,
        target_node_batch_size=int(target_node_batch_size),
        peak_cuda_memory_bytes=peak,
        passed=discrepancy <= maximum_discrepancy,
    )

