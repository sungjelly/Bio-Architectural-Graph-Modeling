"""Fixed-budget training for multiscale continuous-hurdle models.

The local and regional graphs remain separate throughout execution.  Target
nodes are processed in deterministic batches, while loss contributions are
weighted by the global masked zero/positive support so that batching is
exactly equivalent to the frozen full-mask hurdle objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import time
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .hurdle_continuous import (
    HurdleContinuousEvaluation,
    HurdleContinuousLoss,
    evaluate_hurdle_continuous_output,
    hurdle_continuous_loss,
    split_hurdle_continuous_prediction,
)
from .hybrid_count import (
    standardized_log1p_counts,
    tokenize_raw_count_tensor,
)
from .masking import derive_mask_seed
from .training import (
    GraphSplitView,
    TrainingConfig,
    _array_checksum,
    _autocast_context,
    _make_grad_scaler,
    _model_dtype,
    _prepare_fixed_mask,
    _resolve_device,
    make_epoch_mask,
    set_deterministic_seed,
)


_ROUTING_MODES = frozenset({"surrogate", "true", "permuted"})


def _validate_edge_shard(
    edge_index: Any,
    edge_attributes: Any,
    *,
    name: str,
    num_nodes: int,
) -> tuple[Tensor, Tensor]:
    index = torch.as_tensor(edge_index)
    if index.ndim != 2 or index.shape[0] != 2:
        raise ValueError(f"{name}_edge_index must have shape [2, edges]")
    if (
        index.dtype == torch.bool
        or index.is_floating_point()
        or index.is_complex()
    ):
        raise TypeError(f"{name}_edge_index must use an integer dtype")
    index = index.detach().to(dtype=torch.long).contiguous()
    if index.shape[1] == 0:
        raise ValueError(f"{name} graph must contain at least one edge")
    if bool((index < 0).any()) or bool((index >= num_nodes).any()):
        raise ValueError(f"{name}_edge_index contains an out-of-range node")
    if bool((index[0] == index[1]).any()):
        raise ValueError(f"{name} graph cannot contain self loops")

    attributes = torch.as_tensor(edge_attributes)
    if (
        attributes.ndim != 2
        or attributes.shape[0] != index.shape[1]
        or attributes.shape[1] == 0
        or not attributes.is_floating_point()
    ):
        raise TypeError(
            f"{name}_edge_attributes must be floating with shape "
            "[edges, attributes]"
        )
    if not bool(torch.isfinite(attributes).all()):
        raise ValueError(f"{name}_edge_attributes must be finite")
    attributes = attributes.detach().contiguous()

    keys = (
        index[0].detach().cpu() * int(num_nodes)
        + index[1].detach().cpu()
    )
    if int(torch.unique(keys).numel()) != int(keys.numel()):
        raise ValueError(f"{name} graph contains duplicate directed edges")
    return index, attributes


def _edge_keys(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    return (
        edge_index[0].detach().cpu() * int(num_nodes)
        + edge_index[1].detach().cpu()
    )


def _assert_disjoint_edge_shards(
    local_edge_index: Tensor,
    regional_edge_index: Tensor,
    *,
    num_nodes: int,
) -> None:
    local = torch.sort(
        _edge_keys(local_edge_index, num_nodes=num_nodes)
    ).values
    regional = torch.sort(
        _edge_keys(regional_edge_index, num_nodes=num_nodes)
    ).values
    positions = torch.searchsorted(regional, local)
    within = positions < regional.numel()
    if bool(
        (
            regional[
                positions.clamp_max(max(0, regional.numel() - 1))
            ]
            == local
        )[within].any()
    ):
        raise ValueError("local and regional directed edge sets must be disjoint")


@dataclass(frozen=True)
class MultiscaleGraphSplitView:
    """One core with separate sparse local and regional graph shards."""

    expression: Tensor
    coordinates_um: Tensor
    local_edge_index: Tensor
    local_edge_attributes: Tensor
    regional_edge_index: Tensor
    regional_edge_attributes: Tensor
    local_source_index_by_node: Optional[Any] = None
    node_covariates: Optional[Tensor] = None
    block_ids: Optional[Any] = None
    name: str = "split"
    _masking_view: GraphSplitView = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        base = GraphSplitView(
            expression=self.expression,
            coordinates_um=self.coordinates_um,
            edge_index=empty_edges,
            node_covariates=self.node_covariates,
            edge_attributes=None,
            block_ids=self.block_ids,
            name=self.name,
        )
        # A raw-count view must fail before any optimizer or GPU mutation.
        tokenize_raw_count_tensor(base.expression)
        local_index, local_attributes = _validate_edge_shard(
            self.local_edge_index,
            self.local_edge_attributes,
            name="local",
            num_nodes=base.num_nodes,
        )
        regional_index, regional_attributes = _validate_edge_shard(
            self.regional_edge_index,
            self.regional_edge_attributes,
            name="regional",
            num_nodes=base.num_nodes,
        )
        _assert_disjoint_edge_shards(
            local_index,
            regional_index,
            num_nodes=base.num_nodes,
        )
        source_index = self.local_source_index_by_node
        if source_index is not None:
            source_index = torch.as_tensor(source_index)
            if (
                source_index.ndim != 1
                or source_index.numel() != base.num_nodes
                or source_index.dtype == torch.bool
                or source_index.is_floating_point()
                or source_index.is_complex()
            ):
                raise TypeError(
                    "local_source_index_by_node must be an integer node "
                    "permutation"
                )
            source_index = source_index.detach().to(
                dtype=torch.long,
            ).contiguous()
            if (
                bool((source_index < 0).any())
                or bool((source_index >= base.num_nodes).any())
                or int(torch.unique(source_index).numel()) != base.num_nodes
            ):
                raise ValueError(
                    "local_source_index_by_node must be a complete node "
                    "permutation"
                )
        object.__setattr__(self, "expression", base.expression)
        object.__setattr__(self, "coordinates_um", base.coordinates_um)
        object.__setattr__(self, "node_covariates", base.node_covariates)
        object.__setattr__(self, "block_ids", base.block_ids)
        object.__setattr__(self, "name", base.name)
        object.__setattr__(self, "local_edge_index", local_index)
        object.__setattr__(
            self,
            "local_edge_attributes",
            local_attributes,
        )
        object.__setattr__(self, "regional_edge_index", regional_index)
        object.__setattr__(
            self,
            "regional_edge_attributes",
            regional_attributes,
        )
        object.__setattr__(
            self,
            "local_source_index_by_node",
            source_index,
        )
        object.__setattr__(self, "_masking_view", base)

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

    @property
    def masking_view(self) -> GraphSplitView:
        return self._masking_view


@dataclass(frozen=True)
class MultiscaleHurdleEpochRecord:
    epoch: int
    mask_mode: str
    mask_seed: int
    mask_checksum: str
    local_edge_checksum: str
    regional_edge_checksum: str
    n_masked_entries: int
    n_zero_targets: int
    n_positive_targets: int
    n_target_nodes: int
    n_target_batches: int
    n_local_edges_used: int
    n_regional_edges_used: int
    train_hurdle_loss: float
    train_detection_bce: float
    train_positive_continuous_huber: float
    gradient_norm: float
    duration_seconds: float
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class MultiscaleHurdleTrainingResult:
    history: tuple[MultiscaleHurdleEpochRecord, ...]
    final_epoch: int
    final_train_loss: float
    final_state_dict: Mapping[str, Tensor]
    final_state_checksum: str
    device: str
    fixed_epoch_budget: int
    target_node_batch_size: int
    regional_routing: str
    local_routing: str
    checkpoint_policy: str = "final_epoch_no_validation_selection"
    graph_execution: str = "separate_local_regional_exact_no_sampling"
    training_protocol: str = "held_in_multiscale_hurdle_fixed_budget"

    def history_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.history]


@dataclass(frozen=True)
class MultiscaleHurdleFixedMaskResult:
    target_nodes: Tensor
    target: Tensor
    target_mask: Tensor
    full_mask: Tensor
    evaluation: HurdleContinuousEvaluation


@dataclass(frozen=True)
class MultiscalePrecisionEquivalenceResult:
    mask_checksum: str
    fp32_total_loss: float
    amp_total_loss: float
    absolute_total_loss_discrepancy: float
    fp32_components: Mapping[str, float]
    amp_components: Mapping[str, float]
    amp_dtype: str
    target_node_batch_size: int
    peak_cuda_memory_bytes: int
    passed: bool


@dataclass(frozen=True)
class _DeviceMultiscaleView:
    expression: Tensor
    node_covariates: Optional[Tensor]
    local_edge_index: Tensor
    local_edge_attributes: Tensor
    local_source_index_by_node: Optional[Tensor]
    regional_edge_index: Tensor
    regional_edge_attributes: Tensor


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


def _edge_shard_checksum(
    edge_index: Tensor,
    edge_attributes: Tensor,
) -> str:
    digest = hashlib.sha256()
    digest.update(_array_checksum(edge_index).encode("ascii"))
    digest.update(b"\0")
    digest.update(_array_checksum(edge_attributes).encode("ascii"))
    return digest.hexdigest()


def _validated_standardization(
    expression_mean: Any,
    expression_scale: Any,
    *,
    num_genes: int,
) -> tuple[Tensor, Tensor]:
    mean = torch.as_tensor(expression_mean)
    scale = torch.as_tensor(expression_scale)
    if mean.dtype == torch.bool or scale.dtype == torch.bool:
        raise TypeError("expression standardization must be real numeric")
    if mean.is_complex() or scale.is_complex():
        raise TypeError("expression standardization must be real numeric")
    if mean.shape != (num_genes,) or scale.shape != (num_genes,):
        raise ValueError("expression standardization must match fit genes")
    mean = mean.float()
    scale = scale.float()
    if not bool(torch.isfinite(mean).all()) or not bool(
        torch.isfinite(scale).all()
    ):
        raise ValueError("expression standardization must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("expression_scale must be strictly positive")
    return mean, scale


def _validate_target_node_batch_size(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError("target_node_batch_size must be a positive integer")
    return int(value)


def _routing_modes(model: nn.Module) -> tuple[str, str]:
    regional = str(getattr(model, "regional_routing", "")).strip().lower()
    local = str(getattr(model, "local_routing", "")).strip().lower()
    if regional not in _ROUTING_MODES:
        raise ValueError(
            "model.regional_routing must be surrogate, true, or permuted"
        )
    if local not in _ROUTING_MODES:
        raise ValueError(
            "model.local_routing must be surrogate, true, or permuted"
        )
    if regional == "permuted":
        raise ValueError("regional_routing cannot use the local source null")
    return regional, local


def _to_device_view(
    view: MultiscaleGraphSplitView,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _DeviceMultiscaleView:
    return _DeviceMultiscaleView(
        expression=view.expression.to(device=device, dtype=dtype),
        node_covariates=(
            None
            if view.node_covariates is None
            else view.node_covariates.to(device=device, dtype=dtype)
        ),
        local_edge_index=view.local_edge_index.to(
            device=device,
            dtype=torch.long,
        ),
        local_edge_attributes=view.local_edge_attributes.to(
            device=device,
            dtype=dtype,
        ),
        local_source_index_by_node=(
            None
            if view.local_source_index_by_node is None
            else view.local_source_index_by_node.to(
                device=device,
                dtype=torch.long,
            )
        ),
        regional_edge_index=view.regional_edge_index.to(
            device=device,
            dtype=torch.long,
        ),
        regional_edge_attributes=view.regional_edge_attributes.to(
            device=device,
            dtype=dtype,
        ),
    )


def _routing_graph_kwargs(
    model: nn.Module,
    view: _DeviceMultiscaleView,
) -> dict[str, Optional[Tensor]]:
    regional, local = _routing_modes(model)
    return {
        "regional_edge_index": (
            None if regional == "surrogate" else view.regional_edge_index
        ),
        "regional_edge_attributes": (
            None
            if regional == "surrogate"
            else view.regional_edge_attributes
        ),
        "local_edge_index": (
            None if local == "surrogate" else view.local_edge_index
        ),
        "local_edge_attributes": (
            None if local == "surrogate" else view.local_edge_attributes
        ),
        "local_source_index_by_node": (
            view.local_source_index_by_node
            if local == "permuted"
            else None
        ),
    }


def _target_node_batches(mask: Tensor, batch_size: int) -> tuple[Tensor, ...]:
    nodes = mask.any(dim=1).nonzero(as_tuple=False).flatten()
    if nodes.numel() == 0:
        raise ValueError("mask contains no target nodes")
    return tuple(nodes.split(batch_size))


def _forward_target_batch(
    model: nn.Module,
    view: _DeviceMultiscaleView,
    mask: Tensor,
    target_nodes: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    masked_expression = view.expression.masked_fill(mask, 0.0)
    output = model(
        masked_expression,
        mask,
        node_covariates=view.node_covariates,
        target_nodes=target_nodes,
        **_routing_graph_kwargs(model, view),
    )
    if not hasattr(output, "prediction"):
        raise TypeError("multiscale model output must expose prediction")
    prediction = output.prediction
    split_hurdle_continuous_prediction(prediction)
    target = view.expression.index_select(0, target_nodes)
    target_mask = mask.index_select(0, target_nodes)
    expected = (*target.shape, 2)
    if tuple(prediction.shape) != expected:
        raise ValueError(
            f"model prediction shape mismatch: expected {expected}, got "
            f"{tuple(prediction.shape)}"
        )
    return prediction, target, target_mask


def _global_mask_support(
    expression: Tensor,
    mask: Tensor,
) -> tuple[int, int, int]:
    selected = expression[mask]
    n_masked = int(selected.numel())
    n_zero = int((selected == 0).sum().detach().cpu())
    n_positive = int((selected > 0).sum().detach().cpu())
    if n_masked == 0:
        raise ValueError("hurdle loss requires at least one masked target")
    if n_zero == 0 or n_positive == 0:
        raise ValueError(
            "balanced detection loss requires zero and positive strata"
        )
    return n_masked, n_zero, n_positive


def _weighted_batch_loss(
    prediction: Tensor,
    raw_target: Tensor,
    target_mask: Tensor,
    *,
    expression_mean: Tensor,
    expression_scale: Tensor,
    global_zero: int,
    global_positive: int,
) -> HurdleContinuousLoss:
    """Return this batch's exact contribution to the full-mask objective."""

    if global_zero <= 0 or global_positive <= 0:
        raise ValueError("global hurdle support must contain both strata")
    detection_logits, continuous = split_hurdle_continuous_prediction(
        prediction
    )
    if target_mask.dtype != torch.bool or target_mask.shape != raw_target.shape:
        raise ValueError("target_mask must be boolean and match raw_target")
    if (
        prediction.device != raw_target.device
        or target_mask.device != raw_target.device
    ):
        raise ValueError("prediction, target, and mask must share a device")
    if tuple(prediction.shape) != (*raw_target.shape, 2):
        raise ValueError("prediction and target shapes are not aligned")
    tokenize_raw_count_tensor(raw_target)
    selected_counts = raw_target[target_mask]
    if selected_counts.numel() == 0:
        raise ValueError("target-node batch contains no masked entries")
    selected_detection = detection_logits[target_mask].float()
    zero = selected_counts == 0
    positive = selected_counts > 0

    zero_detection_sum = selected_detection.sum() * 0.0
    if bool(zero.any()):
        zero_detection_sum = F.binary_cross_entropy_with_logits(
            selected_detection[zero],
            torch.zeros_like(selected_detection[zero]),
            reduction="sum",
        )
    positive_detection_sum = selected_detection.sum() * 0.0
    if bool(positive.any()):
        positive_detection_sum = F.binary_cross_entropy_with_logits(
            selected_detection[positive],
            torch.ones_like(selected_detection[positive]),
            reduction="sum",
        )
    detection_contribution = 0.5 * (
        zero_detection_sum / float(global_zero)
        + positive_detection_sum / float(global_positive)
    )

    standardized_target = standardized_log1p_counts(
        raw_target,
        expression_mean,
        expression_scale,
    )[target_mask]
    selected_continuous = continuous[target_mask].float()
    continuous_huber_sum = selected_continuous.sum() * 0.0
    if bool(positive.any()):
        continuous_huber_sum = F.huber_loss(
            selected_continuous[positive],
            standardized_target[positive].float(),
            reduction="sum",
            delta=1.0,
        )
    continuous_contribution = continuous_huber_sum / float(global_positive)
    total = 0.5 * (detection_contribution + continuous_contribution)
    for name, value in (
        ("batch detection contribution", detection_contribution),
        ("batch continuous contribution", continuous_contribution),
        ("batch total contribution", total),
    ):
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"{name} is non-finite")
    return HurdleContinuousLoss(
        total=total,
        detection=detection_contribution,
        positive_continuous_huber=continuous_contribution,
        n_masked=int(selected_counts.numel()),
        n_zero=int(zero.sum().detach().cpu()),
        n_positive=int(positive.sum().detach().cpu()),
    )


def _loss_components(loss: HurdleContinuousLoss) -> dict[str, float]:
    result = {
        "hurdle_loss": float(loss.total.detach().float().cpu()),
        "detection_bce": float(loss.detection.detach().float().cpu()),
        "positive_continuous_huber": float(
            loss.positive_continuous_huber.detach().float().cpu()
        ),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("hurdle loss component is non-finite")
    return result


def _used_edge_counts(
    model: nn.Module,
    view: _DeviceMultiscaleView,
) -> tuple[int, int]:
    kwargs = _routing_graph_kwargs(model, view)
    local = kwargs["local_edge_index"]
    regional = kwargs["regional_edge_index"]
    return (
        0 if local is None else int(local.shape[1]),
        0 if regional is None else int(regional.shape[1]),
    )


def fit_full_core_multiscale_hurdle_model(
    model: nn.Module,
    fit_view: MultiscaleGraphSplitView,
    config: TrainingConfig,
    *,
    expression_mean: Any,
    expression_scale: Any,
    target_node_batch_size: int = 512,
) -> MultiscaleHurdleTrainingResult:
    """Fit exactly the configured epochs and retain only the final state."""

    if fit_view.name not in {"fit", "full_core", "full-core"}:
        raise ValueError("fit_view must explicitly identify the held-in fit role")
    if config.restore_best:
        raise ValueError(
            "multiscale full-core training requires restore_best=False"
        )
    if config.huber_delta != 1.0:
        raise ValueError("the frozen positive Huber delta is exactly 1.0")
    if config.edge_dropout != 0.0:
        raise ValueError("the frozen multiscale campaign prohibits edge dropout")
    batch_size = _validate_target_node_batch_size(target_node_batch_size)
    mean, scale = _validated_standardization(
        expression_mean,
        expression_scale,
        num_genes=fit_view.num_genes,
    )
    regional_routing, local_routing = _routing_modes(model)

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
    device_mean = mean.to(device=device)
    device_scale = scale.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = _make_grad_scaler(device, config.amp)
    history: list[MultiscaleHurdleEpochRecord] = []
    local_checksum = _edge_shard_checksum(
        fit_view.local_edge_index,
        fit_view.local_edge_attributes,
    )
    regional_checksum = _edge_shard_checksum(
        fit_view.regional_edge_index,
        fit_view.regional_edge_attributes,
    )
    n_local_edges_used, n_regional_edges_used = _used_edge_counts(
        model,
        device_view,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(config.max_epochs):
        started = time.monotonic()
        model.train()
        mask_batch = make_epoch_mask(fit_view.masking_view, config, epoch)
        mask = torch.from_numpy(np.array(mask_batch.mask, copy=True)).to(
            device=device
        )
        n_masked, n_zero, n_positive = _global_mask_support(
            device_view.expression,
            mask,
        )
        target_batches = _target_node_batches(mask, batch_size)
        optimizer.zero_grad(set_to_none=True)
        totals = {
            "hurdle_loss": 0.0,
            "detection_bce": 0.0,
            "positive_continuous_huber": 0.0,
        }
        observed_masked = 0
        observed_zero = 0
        observed_positive = 0
        for target_nodes in target_batches:
            with _autocast_context(
                enabled=config.amp,
                device=device,
                dtype_name=config.amp_dtype,
            ):
                prediction, target, target_mask = _forward_target_batch(
                    model,
                    device_view,
                    mask,
                    target_nodes,
                )
                batch_loss = _weighted_batch_loss(
                    prediction,
                    target,
                    target_mask,
                    expression_mean=device_mean,
                    expression_scale=device_scale,
                    global_zero=n_zero,
                    global_positive=n_positive,
                )
            components = _loss_components(batch_loss)
            for name in totals:
                totals[name] += components[name]
            observed_masked += batch_loss.n_masked
            observed_zero += batch_loss.n_zero
            observed_positive += batch_loss.n_positive
            scaler.scale(batch_loss.total).backward()
        if (observed_masked, observed_zero, observed_positive) != (
            n_masked,
            n_zero,
            n_positive,
        ):
            raise RuntimeError("target-node batches did not cover the mask exactly")

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip_norm,
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
            MultiscaleHurdleEpochRecord(
                epoch=epoch,
                mask_mode=mask_batch.spec.mode,
                mask_seed=mask_batch.seed,
                mask_checksum=_array_checksum(mask_batch.mask),
                local_edge_checksum=local_checksum,
                regional_edge_checksum=regional_checksum,
                n_masked_entries=n_masked,
                n_zero_targets=n_zero,
                n_positive_targets=n_positive,
                n_target_nodes=mask_batch.n_selected_nodes,
                n_target_batches=len(target_batches),
                n_local_edges_used=n_local_edges_used,
                n_regional_edges_used=n_regional_edges_used,
                train_hurdle_loss=totals["hurdle_loss"],
                train_detection_bce=totals["detection_bce"],
                train_positive_continuous_huber=totals[
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
    for name, value in final_state.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"final state tensor {name!r} is non-finite")
    return MultiscaleHurdleTrainingResult(
        history=tuple(history),
        final_epoch=config.max_epochs - 1,
        final_train_loss=history[-1].train_hurdle_loss,
        final_state_dict=final_state,
        final_state_checksum=_state_dict_checksum(final_state),
        device=str(device),
        fixed_epoch_budget=config.max_epochs,
        target_node_batch_size=batch_size,
        regional_routing=regional_routing,
        local_routing=local_routing,
    )


def _collect_fixed_mask_predictions(
    model: nn.Module,
    view: _DeviceMultiscaleView,
    mask: Tensor,
    *,
    target_node_batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batches = _target_node_batches(mask, target_node_batch_size)
    target_nodes: list[Tensor] = []
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    target_masks: list[Tensor] = []
    for nodes in batches:
        prediction, target, target_mask = _forward_target_batch(
            model,
            view,
            mask,
            nodes,
        )
        target_nodes.append(nodes)
        predictions.append(prediction)
        targets.append(target)
        target_masks.append(target_mask)
    return (
        torch.cat(target_nodes, dim=0),
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(target_masks, dim=0),
    )


def evaluate_fixed_multiscale_hurdle_mask(
    model: nn.Module,
    view: MultiscaleGraphSplitView,
    mask: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
    target_node_batch_size: int = 512,
    device: Optional[str] = None,
    amp: bool = False,
    amp_dtype: str = "auto",
) -> MultiscaleHurdleFixedMaskResult:
    """Evaluate one immutable mask without using it for model selection."""

    batch_size = _validate_target_node_batch_size(target_node_batch_size)
    fixed_mask = _prepare_fixed_mask(view.masking_view, mask)
    mean, scale = _validated_standardization(
        expression_mean,
        expression_scale,
        num_genes=view.num_genes,
    )
    resolved_device = _resolve_device(model, device)
    model.to(resolved_device)
    device_view = _to_device_view(
        view,
        device=resolved_device,
        dtype=_model_dtype(model),
    )
    mask_device = fixed_mask.to(device=resolved_device)
    _global_mask_support(device_view.expression, mask_device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), _autocast_context(
            enabled=amp,
            device=resolved_device,
            dtype_name=str(amp_dtype).lower(),
        ):
            target_nodes, prediction, target, target_mask = (
                _collect_fixed_mask_predictions(
                    model,
                    device_view,
                    mask_device,
                    target_node_batch_size=batch_size,
                )
            )
            evaluation = evaluate_hurdle_continuous_output(
                prediction,
                target,
                target_mask,
                expression_mean=mean,
                expression_scale=scale,
            )
    finally:
        model.train(was_training)
    return MultiscaleHurdleFixedMaskResult(
        target_nodes=target_nodes.detach().cpu(),
        target=target.detach().float().cpu(),
        target_mask=target_mask.detach().cpu(),
        full_mask=fixed_mask.clone(),
        evaluation=evaluation,
    )


def compare_multiscale_fp32_amp_loss(
    model: nn.Module,
    view: MultiscaleGraphSplitView,
    mask: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
    device: str,
    target_node_batch_size: int = 512,
    amp_dtype: str = "auto",
    maximum_discrepancy: float = 1e-3,
) -> MultiscalePrecisionEquivalenceResult:
    """Compare identical weights and a frozen mask under FP32 and AMP."""

    if maximum_discrepancy < 0 or not math.isfinite(maximum_discrepancy):
        raise ValueError("maximum_discrepancy must be finite and nonnegative")
    batch_size = _validate_target_node_batch_size(target_node_batch_size)
    fixed_mask = _prepare_fixed_mask(view.masking_view, mask)
    mean, scale = _validated_standardization(
        expression_mean,
        expression_scale,
        num_genes=view.num_genes,
    )
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("the production precision diagnostic requires CUDA")
    model.to(resolved_device)
    device_view = _to_device_view(
        view,
        device=resolved_device,
        dtype=_model_dtype(model),
    )
    mask_device = fixed_mask.to(device=resolved_device)
    _global_mask_support(device_view.expression, mask_device)
    mean = mean.to(device=resolved_device)
    scale = scale.to(device=resolved_device)
    was_training = model.training
    model.eval()
    torch.cuda.reset_peak_memory_stats(resolved_device)

    def one(enabled: bool) -> HurdleContinuousLoss:
        with torch.no_grad(), _autocast_context(
            enabled=enabled,
            device=resolved_device,
            dtype_name=amp_dtype,
        ):
            _, prediction, target, target_mask = (
                _collect_fixed_mask_predictions(
                    model,
                    device_view,
                    mask_device,
                    target_node_batch_size=batch_size,
                )
            )
            return hurdle_continuous_loss(
                prediction,
                target,
                target_mask,
                expression_mean=mean,
                expression_scale=scale,
                huber_delta=1.0,
            )

    try:
        fp32 = one(False)
        amp = one(True)
    finally:
        model.train(was_training)
    fp32_values = _loss_components(fp32)
    amp_values = _loss_components(amp)
    discrepancy = abs(
        fp32_values["hurdle_loss"] - amp_values["hurdle_loss"]
    )
    return MultiscalePrecisionEquivalenceResult(
        mask_checksum=_array_checksum(fixed_mask),
        fp32_total_loss=fp32_values["hurdle_loss"],
        amp_total_loss=amp_values["hurdle_loss"],
        absolute_total_loss_discrepancy=discrepancy,
        fp32_components=fp32_values,
        amp_components=amp_values,
        amp_dtype=amp_dtype,
        target_node_batch_size=batch_size,
        peak_cuda_memory_bytes=int(
            torch.cuda.max_memory_allocated(resolved_device)
        ),
        passed=discrepancy <= maximum_discrepancy,
    )


__all__ = [
    "MultiscaleGraphSplitView",
    "MultiscaleHurdleEpochRecord",
    "MultiscaleHurdleFixedMaskResult",
    "MultiscaleHurdleTrainingResult",
    "MultiscalePrecisionEquivalenceResult",
    "compare_multiscale_fp32_amp_loss",
    "evaluate_fixed_multiscale_hurdle_mask",
    "fit_full_core_multiscale_hurdle_model",
]
