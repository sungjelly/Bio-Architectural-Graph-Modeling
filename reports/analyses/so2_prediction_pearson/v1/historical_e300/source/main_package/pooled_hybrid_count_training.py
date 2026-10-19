"""Deterministic equal-core training for the pooled hybrid-count campaign.

One global epoch is ten sequential complete-graph batches, one for each
``ANC-01`` through ``ANC-10``.  The graphs are never concatenated and only the
current core is staged on the training device.  Consequently each core
contributes one optimizer step per global epoch regardless of its cell count.

Mask generation and core order use isolated seeds that do not depend on the
model seed.  This pairs masks and order across model initialisations and the
graph/self arms while retaining deterministic model-side randomness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .hybrid_count import HybridCountLoss, hybrid_count_hurdle_loss
from .hybrid_count_training import (
    _loss_components,
    _state_dict_checksum,
)
from .masking import (
    MaskBatch,
    MaskSpec,
    curriculum_mode,
    derive_mask_seed,
    generate_mask,
)
from .training import (
    GraphSplitView,
    TrainingConfig,
    _array_checksum,
    _autocast_context,
    _forward_masked_targets,
    _make_grad_scaler,
    _model_dtype,
    _resolve_device,
    _to_device_view,
    set_deterministic_seed,
)


POOLED_CORE_ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
CORE_ORDER_SEED = 271828


@dataclass(frozen=True)
class PooledCoreBatch:
    """One alias-safe, complete CPU graph participating in pooled training."""

    alias: str
    view: GraphSplitView

    def __post_init__(self) -> None:
        alias = str(self.alias).strip().upper()
        if alias not in POOLED_CORE_ALIASES:
            raise ValueError(
                "pooled core alias must be one of ANC-01 through ANC-10"
            )
        if not isinstance(self.view, GraphSplitView):
            raise TypeError("view must be a GraphSplitView")
        if self.view.name not in {"fit", "full_core", "full-core"}:
            raise ValueError(
                "pooled core view must explicitly identify the held-in fit role"
            )
        tensors = {
            "expression": self.view.expression,
            "coordinates_um": self.view.coordinates_um,
            "edge_index": self.view.edge_index,
            "node_covariates": self.view.node_covariates,
            "edge_attributes": self.view.edge_attributes,
        }
        non_cpu = [
            name
            for name, value in tensors.items()
            if value is not None
            and torch.as_tensor(value).device.type != "cpu"
        ]
        if non_cpu:
            raise ValueError(
                "pooled graph inputs must remain on CPU for staged transfer; "
                f"non-CPU tensors: {', '.join(non_cpu)}"
            )
        object.__setattr__(self, "alias", alias)


@dataclass(frozen=True)
class PooledCoreEpochRecord:
    """One complete-core optimizer step within a global epoch."""

    global_epoch: int
    step_in_epoch: int
    optimizer_step: int
    alias: str
    n_nodes: int
    n_edges: int
    mask_mode: str
    mask_seed: int
    mask_checksum: str
    n_masked_entries: int
    n_zero_targets: int
    n_positive_targets: int
    n_target_nodes: int
    train_hybrid_loss: float
    train_detection_bce: float
    train_ordinal_bce: float
    train_positive_continuous_huber: float
    gradient_norm: float
    duration_seconds: float
    peak_cuda_memory_bytes: int


@dataclass(frozen=True)
class PooledGlobalEpochRecord:
    """Equal-core aggregate of the ten sequential optimizer steps."""

    epoch: int
    ordered_aliases: tuple[str, ...]
    cores_visited: int
    optimizer_steps: int
    mean_hybrid_loss: float
    mean_detection_bce: float
    mean_ordinal_bce: float
    mean_positive_continuous_huber: float
    duration_seconds: float
    peak_cuda_memory_bytes: int
    aggregation: str = "equal_core_arithmetic_mean"


@dataclass(frozen=True)
class PooledEpochBoundaryResume:
    """State sufficient to resume after a completed global epoch only."""

    completed_global_epochs: int
    model_state_dict: Mapping[str, Tensor]
    model_state_checksum: str
    optimizer_state_dict: Mapping[str, Any]
    scaler_state_dict: Mapping[str, Any]

    def __post_init__(self) -> None:
        if int(self.completed_global_epochs) <= 0:
            raise ValueError(
                "resume state must follow at least one completed global epoch"
            )
        if _state_dict_checksum(self.model_state_dict) != str(
            self.model_state_checksum
        ):
            raise ValueError("resume model state checksum does not match")


@dataclass(frozen=True)
class PooledHybridCountTrainingResult:
    """Final shared state and auditable per-core/global training histories."""

    core_history: tuple[PooledCoreEpochRecord, ...]
    global_history: tuple[PooledGlobalEpochRecord, ...]
    completed_global_epochs: int
    final_epoch: int
    final_train_loss: float
    final_state_dict: Mapping[str, Tensor]
    final_state_checksum: str
    final_optimizer_state_dict: Mapping[str, Any]
    final_scaler_state_dict: Mapping[str, Any]
    device: str
    fixed_epoch_budget: int
    optimizer_steps_completed: int
    checkpoint_policy: str = "final_global_epoch_no_validation_selection"
    graph_execution: str = (
        "ten_sequential_complete_core_graphs_staged_one_at_a_time"
    )
    training_protocol: str = (
        "held_in_pooled_ten_core_hybrid_count_equal_core_fixed_budget"
    )

    def core_history_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.core_history]

    def global_history_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.global_history]

    def histories_by_core(
        self,
    ) -> Mapping[str, tuple[PooledCoreEpochRecord, ...]]:
        return {
            alias: tuple(
                record for record in self.core_history if record.alias == alias
            )
            for alias in POOLED_CORE_ALIASES
        }

    def epoch_boundary_resume(self) -> PooledEpochBoundaryResume:
        """Return a detached CPU resume payload at the final epoch boundary."""

        return PooledEpochBoundaryResume(
            completed_global_epochs=self.completed_global_epochs,
            model_state_dict=_clone_tree_to_cpu(self.final_state_dict),
            model_state_checksum=self.final_state_checksum,
            optimizer_state_dict=_clone_tree_to_cpu(
                self.final_optimizer_state_dict
            ),
            scaler_state_dict=_clone_tree_to_cpu(
                self.final_scaler_state_dict
            ),
        )


def _clone_tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_tree_to_cpu(child) for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_tree_to_cpu(child) for child in value)
    if isinstance(value, list):
        return [_clone_tree_to_cpu(child) for child in value]
    return value


def _validate_pooled_batches(
    batches: Sequence[PooledCoreBatch],
) -> Mapping[str, PooledCoreBatch]:
    materialized = tuple(batches)
    aliases = tuple(batch.alias for batch in materialized)
    if len(materialized) != len(POOLED_CORE_ALIASES):
        raise ValueError(
            "pooled training requires exactly ten core batches, ANC-01 "
            "through ANC-10"
        )
    if len(set(aliases)) != len(aliases):
        raise ValueError("pooled training core aliases must be unique")
    missing = sorted(set(POOLED_CORE_ALIASES) - set(aliases))
    unexpected = sorted(set(aliases) - set(POOLED_CORE_ALIASES))
    if missing or unexpected:
        raise ValueError(
            "pooled training requires exactly ANC-01 through ANC-10; "
            f"missing={missing}, unexpected={unexpected}"
        )

    first = materialized[0].view
    expected = (
        first.num_genes,
        first.node_covariate_dim,
        first.edge_attribute_dim,
    )
    for batch in materialized[1:]:
        observed = (
            batch.view.num_genes,
            batch.view.node_covariate_dim,
            batch.view.edge_attribute_dim,
        )
        if observed != expected:
            raise ValueError(
                "all pooled graphs must share gene, node-covariate, and "
                "edge-attribute dimensions"
            )
    return {batch.alias: batch for batch in materialized}


def pooled_core_order(
    aliases: Sequence[str],
    epoch: int,
    *,
    seed: int = CORE_ORDER_SEED,
) -> tuple[str, ...]:
    """Return the fixed deterministic epoch shuffle of the ten aliases."""

    if int(epoch) < 0:
        raise ValueError("epoch must be nonnegative")
    if int(seed) != CORE_ORDER_SEED:
        raise ValueError(
            f"the frozen pooled core-order seed is {CORE_ORDER_SEED}"
        )
    canonical = tuple(sorted(str(alias).strip().upper() for alias in aliases))
    if canonical != POOLED_CORE_ALIASES:
        raise ValueError(
            "core order requires exactly ANC-01 through ANC-10 once each"
        )
    epoch_seed = derive_mask_seed(
        CORE_ORDER_SEED, "pooled-core-order", int(epoch)
    )
    indices = np.random.default_rng(epoch_seed).permutation(len(canonical))
    return tuple(canonical[int(index)] for index in indices)


def make_pooled_epoch_mask(
    batch: PooledCoreBatch,
    config: TrainingConfig,
    epoch: int,
) -> MaskBatch:
    """Generate one alias-specific mask independent of ``model_seed``."""

    if int(epoch) < 0:
        raise ValueError("epoch must be nonnegative")
    mode = curriculum_mode(
        epoch=int(epoch),
        seed=int(config.mask_seed),
        curriculum=config.curriculum,
        warmup_epochs=int(config.warmup_epochs),
    )
    spec = MaskSpec(
        mode=mode,
        partial_gene_rate=config.partial_gene_rate,
        node_rate=config.node_rate,
        block_node_rate=config.block_node_rate,
        block_width_um=config.block_width_um,
        block_shape=config.block_shape,
    )
    core_seed = derive_mask_seed(
        int(config.mask_seed),
        "pooled-core-train-mask",
        batch.alias,
        int(epoch),
    )
    return generate_mask(
        spec,
        n_genes=batch.view.num_genes,
        coordinates_um=batch.view.coordinates_um,
        seed=core_seed,
    )


def _clear_graph_layout_caches(model: nn.Module) -> None:
    """Release caller-owned edge tensors cached by exact graph executors."""

    for module in model.modules():
        clear = getattr(module, "clear_edge_layout_cache", None)
        if callable(clear):
            clear()


def _seed_model_step(model_seed: int, epoch: int, alias: str) -> None:
    """Make dropout/random model operations resumable at epoch boundaries."""

    step_seed = derive_mask_seed(
        int(model_seed), "pooled-model-step", int(epoch), alias
    )
    torch.manual_seed(step_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(step_seed)


def _assert_finite_gradients(model: nn.Module, *, epoch: int, alias: str) -> None:
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(
                "non-finite gradient at global epoch "
                f"{epoch}, alias {alias}, parameter {name}"
            )


def _assert_finite_parameters(model: nn.Module, *, epoch: int, alias: str) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise FloatingPointError(
                "non-finite parameter after optimizer step at global epoch "
                f"{epoch}, alias {alias}, parameter {name}"
            )


def _mean(records: Sequence[PooledCoreEpochRecord], field: str) -> float:
    values = [float(getattr(record, field)) for record in records]
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(result):
        raise FloatingPointError(f"non-finite equal-core mean for {field}")
    return result


def fit_pooled_hybrid_count_model(
    model: nn.Module,
    core_batches: Sequence[PooledCoreBatch],
    config: TrainingConfig,
    *,
    expression_mean: Any,
    expression_scale: Any,
    resume: PooledEpochBoundaryResume | None = None,
) -> PooledHybridCountTrainingResult:
    """Fit one shared model with ten equal-core optimizer steps per epoch.

    ``config.max_epochs`` is the total global-epoch budget.  A resume payload
    can start only after a fully completed global epoch; partial-epoch state is
    deliberately unrepresentable.
    """

    batches_by_alias = _validate_pooled_batches(core_batches)
    if config.restore_best:
        raise ValueError("pooled training requires restore_best=False")
    if config.huber_delta != 1.0:
        raise ValueError("the frozen positive Huber delta is exactly 1.0")
    if config.edge_dropout != 0.0:
        raise ValueError("the frozen pooled campaign prohibits edge dropout")

    num_genes = next(iter(batches_by_alias.values())).view.num_genes
    mean = torch.as_tensor(expression_mean, dtype=torch.float32)
    scale = torch.as_tensor(expression_scale, dtype=torch.float32)
    if mean.shape != (num_genes,) or scale.shape != (num_genes,):
        raise ValueError("expression standardization must match pooled genes")
    if not bool(torch.isfinite(mean).all()) or not bool(
        torch.isfinite(scale).all()
    ):
        raise ValueError("expression standardization must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("expression_scale must be strictly positive")

    set_deterministic_seed(
        config.model_seed,
        deterministic=config.deterministic,
        warn_only=config.deterministic_warn_only,
    )
    device = _resolve_device(model, config.device)
    _clear_graph_layout_caches(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = _make_grad_scaler(device, config.amp)
    start_epoch = 0
    if resume is not None:
        start_epoch = int(resume.completed_global_epochs)
        if start_epoch >= int(config.max_epochs):
            raise ValueError(
                "resume boundary must precede the configured epoch budget"
            )
        model.load_state_dict(resume.model_state_dict, strict=True)
        optimizer.load_state_dict(
            _clone_tree_to_cpu(resume.optimizer_state_dict)
        )
        scaler.load_state_dict(dict(resume.scaler_state_dict))
        if _state_dict_checksum(model.state_dict()) != (
            resume.model_state_checksum
        ):
            raise ValueError("loaded resume model state checksum does not match")

    device_mean = mean.to(device=device)
    device_scale = scale.to(device=device)
    core_history: list[PooledCoreEpochRecord] = []
    global_history: list[PooledGlobalEpochRecord] = []
    aliases = tuple(batches_by_alias)

    for epoch in range(start_epoch, int(config.max_epochs)):
        epoch_started = time.monotonic()
        order = pooled_core_order(aliases, epoch)
        epoch_records: list[PooledCoreEpochRecord] = []
        for step_in_epoch, alias in enumerate(order):
            step_started = time.monotonic()
            batch = batches_by_alias[alias]
            mask_batch = make_pooled_epoch_mask(batch, config, epoch)
            _seed_model_step(config.model_seed, epoch, alias)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _clear_graph_layout_caches(model)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            device_view = None
            output = None
            loss: HybridCountLoss | None = None
            try:
                # This is the only graph transferred to the device in this
                # step.  Every other PooledCoreBatch remains CPU-resident.
                device_view = _to_device_view(
                    batch.view,
                    device=device,
                    dtype=_model_dtype(model),
                )
                mask = torch.from_numpy(
                    np.array(mask_batch.mask, copy=True)
                ).to(device=device)
                with _autocast_context(
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
                _assert_finite_gradients(model, epoch=epoch, alias=alias)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                if not bool(torch.isfinite(gradient_norm).detach().cpu()):
                    raise FloatingPointError(
                        "non-finite gradient norm at global epoch "
                        f"{epoch}, alias {alias}"
                    )
                scaler.step(optimizer)
                scaler.update()
                _assert_finite_parameters(model, epoch=epoch, alias=alias)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                peak_memory = (
                    int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else 0
                )
                record = PooledCoreEpochRecord(
                    global_epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    optimizer_step=(
                        epoch * len(POOLED_CORE_ALIASES)
                        + step_in_epoch
                        + 1
                    ),
                    alias=alias,
                    n_nodes=batch.view.num_nodes,
                    n_edges=batch.view.num_edges,
                    mask_mode=mask_batch.spec.mode,
                    mask_seed=mask_batch.seed,
                    mask_checksum=_array_checksum(mask_batch.mask),
                    n_masked_entries=loss.n_masked,
                    n_zero_targets=loss.n_zero,
                    n_positive_targets=loss.n_positive,
                    n_target_nodes=mask_batch.n_selected_nodes,
                    train_hybrid_loss=components["hybrid_loss"],
                    train_detection_bce=components["detection_bce"],
                    train_ordinal_bce=components["ordinal_bce"],
                    train_positive_continuous_huber=components[
                        "positive_continuous_huber"
                    ],
                    gradient_norm=float(
                        gradient_norm.detach().float().cpu()
                    ),
                    duration_seconds=time.monotonic() - step_started,
                    peak_cuda_memory_bytes=peak_memory,
                )
                core_history.append(record)
                epoch_records.append(record)
            finally:
                # Exact receiver-partitioned models cache the current edge
                # tensor by identity.  Clear it before the next graph so this
                # step cannot retain a prior graph on the device.
                _clear_graph_layout_caches(model)
                del output, loss, device_view

        if tuple(record.alias for record in epoch_records) != order:
            raise RuntimeError("pooled epoch visit order changed during training")
        if len(epoch_records) != len(POOLED_CORE_ALIASES):
            raise RuntimeError("pooled epoch did not complete ten optimizer steps")
        if set(record.alias for record in epoch_records) != set(
            POOLED_CORE_ALIASES
        ):
            raise RuntimeError(
                "pooled epoch did not visit every core exactly once"
            )
        global_history.append(
            PooledGlobalEpochRecord(
                epoch=epoch,
                ordered_aliases=order,
                cores_visited=len(epoch_records),
                optimizer_steps=len(epoch_records),
                mean_hybrid_loss=_mean(
                    epoch_records, "train_hybrid_loss"
                ),
                mean_detection_bce=_mean(
                    epoch_records, "train_detection_bce"
                ),
                mean_ordinal_bce=_mean(
                    epoch_records, "train_ordinal_bce"
                ),
                mean_positive_continuous_huber=_mean(
                    epoch_records, "train_positive_continuous_huber"
                ),
                duration_seconds=time.monotonic() - epoch_started,
                peak_cuda_memory_bytes=max(
                    record.peak_cuda_memory_bytes
                    for record in epoch_records
                ),
            )
        )

    expected_new_steps = (
        int(config.max_epochs) - start_epoch
    ) * len(POOLED_CORE_ALIASES)
    if len(core_history) != expected_new_steps:
        raise RuntimeError(
            "fixed-budget pooled training did not complete every optimizer step"
        )
    if len(global_history) != int(config.max_epochs) - start_epoch:
        raise RuntimeError(
            "fixed-budget pooled training did not complete every global epoch"
        )

    final_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    final_optimizer_state = _clone_tree_to_cpu(optimizer.state_dict())
    final_scaler_state = _clone_tree_to_cpu(scaler.state_dict())
    return PooledHybridCountTrainingResult(
        core_history=tuple(core_history),
        global_history=tuple(global_history),
        completed_global_epochs=int(config.max_epochs),
        final_epoch=int(config.max_epochs) - 1,
        final_train_loss=global_history[-1].mean_hybrid_loss,
        final_state_dict=final_state,
        final_state_checksum=_state_dict_checksum(final_state),
        final_optimizer_state_dict=final_optimizer_state,
        final_scaler_state_dict=final_scaler_state,
        device=str(device),
        fixed_epoch_budget=int(config.max_epochs),
        optimizer_steps_completed=int(config.max_epochs)
        * len(POOLED_CORE_ALIASES),
    )


__all__ = [
    "CORE_ORDER_SEED",
    "POOLED_CORE_ALIASES",
    "PooledCoreBatch",
    "PooledCoreEpochRecord",
    "PooledEpochBoundaryResume",
    "PooledGlobalEpochRecord",
    "PooledHybridCountTrainingResult",
    "fit_pooled_hybrid_count_model",
    "make_pooled_epoch_mask",
    "pooled_core_order",
]
