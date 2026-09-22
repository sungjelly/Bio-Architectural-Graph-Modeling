"""Cohort-parameterized staged training for relative-geometric QKV models.

This module is deliberately additive.  The six-core trainer and its serialized
``pooled_relative_qkv_epoch_boundary_resume_v1`` payload remain unchanged.
The v2 protocol supports an explicitly named, even-sized cohort and averages
the gradients from two complete cores and ten independently masked views per
core into one optimizer update.  Only one complete core is staged at a time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Integral
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .adjacency_ablation import mask_realization_sha256, sample_uniform_mask_numpy
from .masking import derive_mask_seed
from .pooled_relative_qkv_training import (
    CORE_ORDER_SEED,
    MASK_BASE_SEED,
    MASK_VIEWS_PER_CORE_STEP,
    ExactUniformTrainingMask,
    PooledRelativeQKVMaskViewRecord,
    PooledRelativeQKVTrainingError,
    _assert_finite_gradients,
    _clear_graph_layout_caches,
    _clone_tree_to_cpu,
    _prediction_tensor,
    _tree_sha256,
    masked_huber_reconstruction_loss,
)
from .training import (
    _autocast_context,
    _make_grad_scaler,
    _model_dtype,
    _resolve_device,
    set_deterministic_seed,
)


SO2_14CORE_ALIASES = tuple(f"SO2-C{core:02d}" for core in range(15, 29))
SO2_CORES_PER_OPTIMIZER_UPDATE = 2
SO2_OPTIMIZER_UPDATES_PER_EPOCH = 7
SO2_LOSSES_PER_OPTIMIZER_UPDATE = 20
COHORT_RESUME_SCHEMA_V2 = "pooled_relative_qkv_epoch_boundary_resume_v2"
COHORT_HISTORY_SCHEMA_V2 = "pooled_relative_qkv_cumulative_history_v2"
COHORT_MODEL_STEP_RNG_DERIVATION = (
    "model_seed+global_epoch+optimizer_update_in_epoch+core_alias+mask_view_index"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple(str(alias).strip().upper() for alias in aliases)
    if not canonical or any(not alias for alias in canonical):
        raise PooledRelativeQKVTrainingError("Cohort aliases must be non-empty.")
    if len(set(canonical)) != len(canonical):
        raise PooledRelativeQKVTrainingError("Cohort aliases must be unique.")
    return canonical


@dataclass(frozen=True)
class CohortRelativeQKVCoreBatch:
    """One complete disconnected core graph, kept CPU-resident between steps."""

    alias: str
    target_expression: Tensor = field(repr=False)
    edge_index: Tensor = field(repr=False)
    relative_geometry: Tensor = field(repr=False)
    node_covariates: Tensor = field(repr=False)

    def __post_init__(self) -> None:
        alias = str(self.alias).strip().upper()
        if not alias:
            raise PooledRelativeQKVTrainingError("Core alias must be non-empty.")
        expression = torch.as_tensor(self.target_expression)
        edges = torch.as_tensor(self.edge_index)
        geometry = torch.as_tensor(self.relative_geometry)
        covariates = torch.as_tensor(self.node_covariates)
        for name, tensor in (
            ("target_expression", expression),
            ("edge_index", edges),
            ("relative_geometry", geometry),
            ("node_covariates", covariates),
        ):
            if tensor.device.type != "cpu":
                raise PooledRelativeQKVTrainingError(
                    f"{name} must be supplied as CPU-resident campaign data."
                )
        if (
            expression.ndim != 2
            or expression.shape[0] == 0
            or expression.shape[1] == 0
            or not expression.is_floating_point()
            or not bool(torch.isfinite(expression).all())
        ):
            raise PooledRelativeQKVTrainingError(
                "target_expression must be finite floating [nodes, genes]."
            )
        if (
            edges.ndim != 2
            or edges.shape[0] != 2
            or edges.dtype == torch.bool
            or edges.is_floating_point()
            or edges.is_complex()
        ):
            raise PooledRelativeQKVTrainingError(
                "edge_index must be integral with shape [2, edges]."
            )
        edges = edges.to(dtype=torch.long)
        if edges.numel():
            if bool((edges < 0).any()) or bool((edges >= expression.shape[0]).any()):
                raise PooledRelativeQKVTrainingError(
                    "edge_index contains an out-of-range node."
                )
            if bool((edges[0] == edges[1]).any()):
                raise PooledRelativeQKVTrainingError("Self-loops are prohibited.")
        if (
            geometry.ndim != 2
            or geometry.shape[0] != edges.shape[1]
            or geometry.shape[1] == 0
            or not geometry.is_floating_point()
            or not bool(torch.isfinite(geometry).all())
        ):
            raise PooledRelativeQKVTrainingError(
                "relative_geometry must be finite floating [edges, features]."
            )
        if (
            covariates.ndim != 2
            or covariates.shape[0] != expression.shape[0]
            or covariates.shape[1] == 0
            or not covariates.is_floating_point()
            or not bool(torch.isfinite(covariates).all())
        ):
            raise PooledRelativeQKVTrainingError(
                "node_covariates must be finite floating [nodes, covariates]."
            )
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "target_expression", expression.contiguous())
        object.__setattr__(self, "edge_index", edges.contiguous())
        object.__setattr__(self, "relative_geometry", geometry.contiguous())
        object.__setattr__(self, "node_covariates", covariates.contiguous())

    @property
    def n_nodes(self) -> int:
        return int(self.target_expression.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.target_expression.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass(frozen=True)
class CohortRelativeQKVTrainingConfig:
    """Versioned contract for one bounded cohort-training segment."""

    model_seed: int
    cohort_aliases: tuple[str, ...] = SO2_14CORE_ALIASES
    segment_start_global_epoch: int = 0
    segment_end_global_epoch: int = 150
    cores_per_optimizer_update: int = SO2_CORES_PER_OPTIMIZER_UPDATE
    mask_views_per_core: int = MASK_VIEWS_PER_CORE_STEP
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    huber_delta: float = 1.0
    mask_base_seed: int = MASK_BASE_SEED
    core_order_seed: int = CORE_ORDER_SEED
    amp: bool = False
    amp_dtype: str = "auto"
    deterministic: bool = True
    deterministic_warn_only: bool = False
    device: str | None = None
    stage_complete_core_graph_on_device: bool = False
    staged_relative_geometry_dtype: str = "float32"
    checkpoint_interval_global_epochs: int = 1
    distributed_world_size: int = 1
    distributed_rank: int = 0

    def __post_init__(self) -> None:
        aliases = _canonical_aliases(self.cohort_aliases)
        for name in (
            "model_seed",
            "segment_start_global_epoch",
            "segment_end_global_epoch",
            "cores_per_optimizer_update",
            "mask_views_per_core",
            "checkpoint_interval_global_epochs",
            "distributed_world_size",
            "distributed_rank",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise PooledRelativeQKVTrainingError(f"{name} must be an integer.")
        if int(self.model_seed) < 0 or int(self.segment_start_global_epoch) < 0:
            raise PooledRelativeQKVTrainingError(
                "model_seed and segment start must be non-negative."
            )
        if int(self.segment_end_global_epoch) <= int(self.segment_start_global_epoch):
            raise PooledRelativeQKVTrainingError(
                "Segment end must be greater than segment start."
            )
        if int(self.cores_per_optimizer_update) != 2:
            raise PooledRelativeQKVTrainingError(
                "The v2 smooth-gradient contract requires two cores per update."
            )
        if len(aliases) % int(self.cores_per_optimizer_update) != 0:
            raise PooledRelativeQKVTrainingError(
                "Cohort size must be divisible by cores_per_optimizer_update."
            )
        if int(self.mask_views_per_core) != MASK_VIEWS_PER_CORE_STEP:
            raise PooledRelativeQKVTrainingError(
                "The v2 protocol requires exactly ten mask views per core."
            )
        if int(self.checkpoint_interval_global_epochs) <= 0:
            raise PooledRelativeQKVTrainingError(
                "Checkpoint interval must be a positive integer."
            )
        world_size = int(self.distributed_world_size)
        rank = int(self.distributed_rank)
        if world_size not in {1, 4}:
            raise PooledRelativeQKVTrainingError(
                "v2 supports either single-process execution or exactly four DDP ranks."
            )
        if rank not in range(world_size):
            raise PooledRelativeQKVTrainingError(
                "distributed_rank must be in [0, distributed_world_size)."
            )
        for name, allow_zero in (
            ("learning_rate", False),
            ("weight_decay", True),
            ("gradient_clip_norm", False),
            ("huber_delta", False),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
                raise PooledRelativeQKVTrainingError(
                    f"{name} must be finite and "
                    f"{'non-negative' if allow_zero else 'positive'}."
                )
        if float(self.huber_delta) != 1.0:
            raise PooledRelativeQKVTrainingError("Huber delta is locked to 1.0.")
        if int(self.mask_base_seed) != MASK_BASE_SEED:
            raise PooledRelativeQKVTrainingError(
                f"Mask base seed is locked to {MASK_BASE_SEED}."
            )
        if int(self.core_order_seed) != CORE_ORDER_SEED:
            raise PooledRelativeQKVTrainingError(
                f"Core order seed is locked to {CORE_ORDER_SEED}."
            )
        dtype = str(self.amp_dtype).lower()
        if dtype not in {"auto", "float16", "bfloat16"}:
            raise PooledRelativeQKVTrainingError(
                "amp_dtype must be auto, float16, or bfloat16."
            )
        geometry_dtype = str(self.staged_relative_geometry_dtype).lower()
        if geometry_dtype not in {"float16", "bfloat16", "float32"}:
            raise PooledRelativeQKVTrainingError(
                "staged_relative_geometry_dtype must be float16, bfloat16, or float32."
            )
        object.__setattr__(self, "cohort_aliases", aliases)
        object.__setattr__(self, "model_seed", int(self.model_seed))
        object.__setattr__(
            self, "segment_start_global_epoch", int(self.segment_start_global_epoch)
        )
        object.__setattr__(
            self, "segment_end_global_epoch", int(self.segment_end_global_epoch)
        )
        object.__setattr__(self, "amp_dtype", dtype)
        object.__setattr__(self, "staged_relative_geometry_dtype", geometry_dtype)
        object.__setattr__(self, "distributed_world_size", world_size)
        object.__setattr__(self, "distributed_rank", rank)

    @property
    def optimizer_updates_per_global_epoch(self) -> int:
        return len(self.cohort_aliases) // self.cores_per_optimizer_update

    @property
    def losses_per_optimizer_update(self) -> int:
        return self.cores_per_optimizer_update * self.mask_views_per_core


def _validate_batches(
    batches: Sequence[CohortRelativeQKVCoreBatch],
    config: CohortRelativeQKVTrainingConfig,
) -> Mapping[str, CohortRelativeQKVCoreBatch]:
    materialized = tuple(batches)
    aliases = tuple(batch.alias for batch in materialized)
    if len(materialized) != len(config.cohort_aliases) or len(set(aliases)) != len(
        aliases
    ):
        raise PooledRelativeQKVTrainingError(
            "Cohort training requires exactly one unique batch per configured alias."
        )
    if set(aliases) != set(config.cohort_aliases):
        raise PooledRelativeQKVTrainingError(
            "Core batches do not match the configured cohort alias contract."
        )
    first = materialized[0]
    expected = (
        first.n_genes,
        int(first.node_covariates.shape[1]),
        int(first.relative_geometry.shape[1]),
    )
    for batch in materialized[1:]:
        observed = (
            batch.n_genes,
            int(batch.node_covariates.shape[1]),
            int(batch.relative_geometry.shape[1]),
        )
        if observed != expected:
            raise PooledRelativeQKVTrainingError(
                "All cohort cores must share gene, metadata, and geometry schemas."
            )
    return {batch.alias: batch for batch in materialized}


def cohort_relative_qkv_core_order(
    global_epoch: int,
    *,
    aliases: Sequence[str] = SO2_14CORE_ALIASES,
    core_order_seed: int = CORE_ORDER_SEED,
) -> tuple[str, ...]:
    """Return one deterministic permutation of the complete configured cohort."""

    if int(global_epoch) < 0:
        raise PooledRelativeQKVTrainingError("global_epoch must be non-negative.")
    canonical = _canonical_aliases(aliases)
    seed = derive_mask_seed(
        int(core_order_seed),
        "relative-qkv-cohort-core-order-v2",
        int(global_epoch),
        *sorted(canonical),
    )
    permutation = np.random.default_rng(seed).permutation(len(canonical))
    return tuple(canonical[int(index)] for index in permutation)


def cohort_relative_qkv_core_pairs(
    global_epoch: int,
    *,
    aliases: Sequence[str] = SO2_14CORE_ALIASES,
    core_order_seed: int = CORE_ORDER_SEED,
) -> tuple[tuple[str, str], ...]:
    order = cohort_relative_qkv_core_order(
        global_epoch, aliases=aliases, core_order_seed=core_order_seed
    )
    if len(order) % 2:
        raise PooledRelativeQKVTrainingError("Paired core order requires even size.")
    return tuple((order[index], order[index + 1]) for index in range(0, len(order), 2))


def cohort_relative_qkv_mask_seed(
    alias: str,
    global_epoch: int,
    *,
    view_index: int,
    mask_base_seed: int = MASK_BASE_SEED,
    resample_attempt: int = 0,
) -> int:
    """Derive a mask seed from alias/epoch/view and never from model seed."""

    canonical_alias = str(alias).strip().upper()
    if not canonical_alias or int(global_epoch) < 0 or int(resample_attempt) < 0:
        raise PooledRelativeQKVTrainingError(
            "Alias must be non-empty and epoch/resample attempt non-negative."
        )
    if int(view_index) not in range(MASK_VIEWS_PER_CORE_STEP):
        raise PooledRelativeQKVTrainingError("view_index must be 0 through 9.")
    parts: tuple[object, ...] = (
        "relative-qkv-uniform-per-cell-train-mask",
        canonical_alias,
        int(global_epoch),
        int(view_index),
    )
    if int(resample_attempt) > 0:
        parts += ("nonzero-total-resample", int(resample_attempt))
    return derive_mask_seed(int(mask_base_seed), *parts)


def make_cohort_exact_uniform_training_mask(
    batch: CohortRelativeQKVCoreBatch,
    global_epoch: int,
    *,
    view_index: int,
    mask_base_seed: int = MASK_BASE_SEED,
    chunk_cells: int = 2048,
    maximum_zero_total_resamples: int = 1024,
) -> ExactUniformTrainingMask:
    if not isinstance(maximum_zero_total_resamples, Integral) or int(
        maximum_zero_total_resamples
    ) <= 0:
        raise PooledRelativeQKVTrainingError(
            "maximum_zero_total_resamples must be a positive integer."
        )
    initial_seed = cohort_relative_qkv_mask_seed(
        batch.alias,
        global_epoch,
        view_index=view_index,
        mask_base_seed=mask_base_seed,
    )
    for attempt in range(int(maximum_zero_total_resamples) + 1):
        seed = cohort_relative_qkv_mask_seed(
            batch.alias,
            global_epoch,
            view_index=view_index,
            mask_base_seed=mask_base_seed,
            resample_attempt=attempt,
        )
        realization = sample_uniform_mask_numpy(
            batch.n_nodes, batch.n_genes, seed=seed, chunk_cells=chunk_cells
        )
        if int(realization.masked_gene_counts.sum()) == 0:
            continue
        checksum = _payload_sha256(
            {
                "schema": "pooled_relative_qkv_exact_uniform_mask_v1",
                "initial_seed": initial_seed,
                "effective_seed": seed,
                "zero_total_resample_count": attempt,
                "realization_checksum": mask_realization_sha256(
                    realization.mask,
                    realization.masked_gene_counts,
                    seed=int(seed),
                ),
            }
        )
        return ExactUniformTrainingMask(
            mask=realization.mask,
            masked_gene_counts=realization.masked_gene_counts,
            initial_seed=initial_seed,
            effective_seed=seed,
            zero_total_resample_count=attempt,
            checksum_sha256=checksum,
        )
    raise PooledRelativeQKVTrainingError(
        "Deterministic zero-total mask resampling exhausted its fixed attempt cap."
    )


def cohort_relative_qkv_model_step_seed(
    model_seed: int,
    global_epoch: int,
    optimizer_update_in_epoch: int,
    alias: str,
    view_index: int,
) -> int:
    if min(int(model_seed), int(global_epoch), int(optimizer_update_in_epoch)) < 0:
        raise PooledRelativeQKVTrainingError("Model RNG inputs must be non-negative.")
    if int(view_index) not in range(MASK_VIEWS_PER_CORE_STEP):
        raise PooledRelativeQKVTrainingError(
            "Model RNG view_index must be 0 through 9."
        )
    canonical_alias = str(alias).strip().upper()
    if not canonical_alias:
        raise PooledRelativeQKVTrainingError("Model RNG alias must be non-empty.")
    return derive_mask_seed(
        int(model_seed),
        "relative-qkv-cohort-model-step-v2",
        int(global_epoch),
        int(optimizer_update_in_epoch),
        canonical_alias,
        int(view_index),
    )


@dataclass(frozen=True)
class CohortRelativeQKVDistributedAssignment:
    """The exact local work assigned to one rank for a paired update."""

    rank: int
    world_size: int
    alias: str
    position_in_optimizer_update: int
    view_indices: tuple[int, ...]
    local_loss_divisor: int


def cohort_relative_qkv_distributed_assignment(
    pair: tuple[str, str],
    *,
    rank: int,
    world_size: int,
) -> CohortRelativeQKVDistributedAssignment:
    """Partition 20 pair losses so DDP's rank mean is their exact global mean.

    Ranks 0/1 receive views 0--4/5--9 of the first core; ranks 2/3 receive
    views 0--4/5--9 of the second.  Every rank averages its five local losses,
    then ordinary DDP gradient averaging across four ranks produces the mean of
    all 20 losses.  Single-process execution receives both cores elsewhere in
    the trainer and therefore is intentionally not represented by this helper.
    """

    if len(pair) != 2 or not all(str(alias).strip() for alias in pair):
        raise PooledRelativeQKVTrainingError(
            "A distributed assignment needs two aliases."
        )
    if int(world_size) != 4 or int(rank) not in range(4):
        raise PooledRelativeQKVTrainingError(
            "Distributed pair assignment requires exactly ranks 0 through 3."
        )
    core_position = 0 if int(rank) < 2 else 1
    view_half = int(rank) % 2
    first_view = view_half * 5
    return CohortRelativeQKVDistributedAssignment(
        rank=int(rank),
        world_size=4,
        alias=str(pair[core_position]).strip().upper(),
        position_in_optimizer_update=core_position,
        view_indices=tuple(range(first_view, first_view + 5)),
        local_loss_divisor=5,
    )


@dataclass(frozen=True)
class CohortRelativeQKVCoreRecord:
    global_epoch: int
    completed_global_epoch: int
    position_in_epoch: int
    optimizer_update_in_epoch: int
    cumulative_optimizer_update: int
    position_in_optimizer_update: int
    alias: str
    n_nodes: int
    n_edges: int
    mask_views: tuple[PooledRelativeQKVMaskViewRecord, ...]
    n_mask_views: int
    n_masked_entries_across_views: int
    model_step_seed: int
    model_view_seeds: tuple[int, ...]
    masked_huber_loss: float


@dataclass(frozen=True)
class CohortRelativeQKVOptimizerUpdateRecord:
    global_epoch: int
    completed_global_epoch: int
    optimizer_update_in_epoch: int
    cumulative_optimizer_update: int
    aliases: tuple[str, str]
    n_cores: int
    mask_views_per_core: int
    losses_averaged: int
    equal_core_mean_masked_huber: float
    gradient_norm: float


@dataclass(frozen=True)
class CohortRelativeQKVGlobalEpochRecord:
    global_epoch: int
    completed_global_epochs: int
    ordered_aliases: tuple[str, ...]
    ordered_core_pairs: tuple[tuple[str, str], ...]
    core_records_this_epoch: int
    optimizer_updates_this_epoch: int
    cumulative_optimizer_updates: int
    equal_core_mean_masked_huber: float
    mean_gradient_norm: float
    max_gradient_norm: float
    aggregation: str = "equal_core_arithmetic_mean"


def _history_checksum(
    core_history: Sequence[CohortRelativeQKVCoreRecord],
    update_history: Sequence[CohortRelativeQKVOptimizerUpdateRecord],
    global_history: Sequence[CohortRelativeQKVGlobalEpochRecord],
) -> str:
    return _payload_sha256(
        {
            "schema": COHORT_HISTORY_SCHEMA_V2,
            "core_history": [asdict(record) for record in core_history],
            "optimizer_update_history": [asdict(record) for record in update_history],
            "global_history": [asdict(record) for record in global_history],
        }
    )


@dataclass(frozen=True)
class CohortRelativeQKVEpochBoundaryResume:
    """Checksum-validated deterministic continuation state for the v2 protocol."""

    completed_global_epochs: int
    optimizer_updates_completed: int
    model_seed: int
    mask_base_seed: int
    core_order_seed: int
    cohort_aliases: tuple[str, ...]
    cores_per_optimizer_update: int
    optimizer_updates_per_global_epoch: int
    mask_views_per_core: int
    losses_per_optimizer_update: int
    model_state_dict: Mapping[str, Tensor] = field(repr=False)
    model_state_checksum: str
    optimizer_state_dict: Mapping[str, Any] = field(repr=False)
    optimizer_state_checksum: str
    scaler_state_dict: Mapping[str, Any] = field(repr=False)
    scaler_state_checksum: str
    core_history: tuple[CohortRelativeQKVCoreRecord, ...] = field(repr=False)
    optimizer_update_history: tuple[
        CohortRelativeQKVOptimizerUpdateRecord, ...
    ] = field(repr=False)
    global_history: tuple[CohortRelativeQKVGlobalEpochRecord, ...] = field(repr=False)
    history_checksum: str
    resume_checksum: str
    model_step_rng_derivation: str = COHORT_MODEL_STEP_RNG_DERIVATION
    resume_schema: str = COHORT_RESUME_SCHEMA_V2

    def __post_init__(self) -> None:
        completed = int(self.completed_global_epochs)
        aliases = _canonical_aliases(self.cohort_aliases)
        cores = len(aliases)
        per_update = int(self.cores_per_optimizer_update)
        updates_per_epoch = int(self.optimizer_updates_per_global_epoch)
        if completed <= 0:
            raise PooledRelativeQKVTrainingError(
                "Resume state must follow at least one complete global epoch."
            )
        if self.resume_schema != COHORT_RESUME_SCHEMA_V2:
            raise PooledRelativeQKVTrainingError("Unsupported cohort resume schema.")
        if per_update != 2 or updates_per_epoch != cores // per_update:
            raise PooledRelativeQKVTrainingError("Resume core/update contract drifted.")
        if int(self.mask_views_per_core) != MASK_VIEWS_PER_CORE_STEP:
            raise PooledRelativeQKVTrainingError("Resume mask-view contract drifted.")
        if int(self.losses_per_optimizer_update) != (
            per_update * int(self.mask_views_per_core)
        ):
            raise PooledRelativeQKVTrainingError(
                "Resume loss divisor contract drifted."
            )
        if int(self.optimizer_updates_completed) != completed * updates_per_epoch:
            raise PooledRelativeQKVTrainingError("Resume update count is inconsistent.")
        if (
            len(self.core_history) != completed * cores
            or len(self.optimizer_update_history) != completed * updates_per_epoch
            or len(self.global_history) != completed
        ):
            raise PooledRelativeQKVTrainingError(
                "Resume histories are not cumulative from epoch zero."
            )
        for epoch in range(completed):
            core_slice = self.core_history[epoch * cores : (epoch + 1) * cores]
            update_slice = self.optimizer_update_history[
                epoch * updates_per_epoch : (epoch + 1) * updates_per_epoch
            ]
            epoch_record = self.global_history[epoch]
            observed_order = tuple(record.alias for record in core_slice)
            observed_pairs = tuple(record.aliases for record in update_slice)
            if (
                observed_order != epoch_record.ordered_aliases
                or observed_pairs != epoch_record.ordered_core_pairs
                or set(observed_order) != set(aliases)
                or len(set(observed_order)) != cores
            ):
                raise PooledRelativeQKVTrainingError(
                    "Resume alias/order/update history is inconsistent."
                )
        for record in self.core_history:
            if (
                record.n_mask_views != MASK_VIEWS_PER_CORE_STEP
                or len(record.mask_views) != MASK_VIEWS_PER_CORE_STEP
                or len(record.model_view_seeds) != MASK_VIEWS_PER_CORE_STEP
                or tuple(view.view_index for view in record.mask_views)
                != tuple(range(MASK_VIEWS_PER_CORE_STEP))
            ):
                raise PooledRelativeQKVTrainingError(
                    "Resume core history lacks ten ordered mask views."
                )
        for update in self.optimizer_update_history:
            if (
                update.n_cores != per_update
                or update.mask_views_per_core != MASK_VIEWS_PER_CORE_STEP
                or update.losses_averaged != self.losses_per_optimizer_update
            ):
                raise PooledRelativeQKVTrainingError(
                    "Resume optimizer-update averaging contract drifted."
                )
        if _tree_sha256(self.model_state_dict) != self.model_state_checksum:
            raise PooledRelativeQKVTrainingError("Resume model checksum mismatch.")
        if _tree_sha256(self.optimizer_state_dict) != self.optimizer_state_checksum:
            raise PooledRelativeQKVTrainingError("Resume optimizer checksum mismatch.")
        if _tree_sha256(self.scaler_state_dict) != self.scaler_state_checksum:
            raise PooledRelativeQKVTrainingError("Resume scaler checksum mismatch.")
        if _history_checksum(
            self.core_history, self.optimizer_update_history, self.global_history
        ) != self.history_checksum:
            raise PooledRelativeQKVTrainingError("Resume history checksum mismatch.")
        if _payload_sha256(_resume_identity_payload(self)) != self.resume_checksum:
            raise PooledRelativeQKVTrainingError("Resume payload checksum mismatch.")
        if not all(
            _is_sha256(checksum)
            for checksum in (
                self.model_state_checksum,
                self.optimizer_state_checksum,
                self.scaler_state_checksum,
                self.history_checksum,
                self.resume_checksum,
            )
        ):
            raise PooledRelativeQKVTrainingError("Resume checksum is malformed.")


def _resume_identity_payload(
    resume: CohortRelativeQKVEpochBoundaryResume | Mapping[str, Any],
) -> dict[str, object]:
    def value(name: str) -> Any:
        if isinstance(resume, Mapping):
            return resume[name]
        return getattr(resume, name)

    return {
        "schema": COHORT_RESUME_SCHEMA_V2,
        "completed_global_epochs": int(value("completed_global_epochs")),
        "optimizer_updates_completed": int(value("optimizer_updates_completed")),
        "model_seed": int(value("model_seed")),
        "mask_base_seed": int(value("mask_base_seed")),
        "core_order_seed": int(value("core_order_seed")),
        "cohort_aliases": list(value("cohort_aliases")),
        "cores_per_optimizer_update": int(value("cores_per_optimizer_update")),
        "optimizer_updates_per_global_epoch": int(
            value("optimizer_updates_per_global_epoch")
        ),
        "mask_views_per_core": int(value("mask_views_per_core")),
        "losses_per_optimizer_update": int(value("losses_per_optimizer_update")),
        "model_state_checksum": value("model_state_checksum"),
        "optimizer_state_checksum": value("optimizer_state_checksum"),
        "scaler_state_checksum": value("scaler_state_checksum"),
        "history_checksum": value("history_checksum"),
        "model_step_rng_derivation": value("model_step_rng_derivation"),
    }


def _build_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    config: CohortRelativeQKVTrainingConfig,
    completed_global_epochs: int,
    core_history: Sequence[CohortRelativeQKVCoreRecord],
    update_history: Sequence[CohortRelativeQKVOptimizerUpdateRecord],
    global_history: Sequence[CohortRelativeQKVGlobalEpochRecord],
) -> CohortRelativeQKVEpochBoundaryResume:
    model_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    optimizer_state = _clone_tree_to_cpu(optimizer.state_dict())
    scaler_state = _clone_tree_to_cpu(scaler.state_dict())
    history_checksum = _history_checksum(core_history, update_history, global_history)
    values: dict[str, Any] = {
        "completed_global_epochs": int(completed_global_epochs),
        "optimizer_updates_completed": int(completed_global_epochs)
        * config.optimizer_updates_per_global_epoch,
        "model_seed": config.model_seed,
        "mask_base_seed": config.mask_base_seed,
        "core_order_seed": config.core_order_seed,
        "cohort_aliases": config.cohort_aliases,
        "cores_per_optimizer_update": config.cores_per_optimizer_update,
        "optimizer_updates_per_global_epoch": config.optimizer_updates_per_global_epoch,
        "mask_views_per_core": config.mask_views_per_core,
        "losses_per_optimizer_update": config.losses_per_optimizer_update,
        "model_state_dict": model_state,
        "model_state_checksum": _tree_sha256(model_state),
        "optimizer_state_dict": optimizer_state,
        "optimizer_state_checksum": _tree_sha256(optimizer_state),
        "scaler_state_dict": scaler_state,
        "scaler_state_checksum": _tree_sha256(scaler_state),
        "core_history": tuple(core_history),
        "optimizer_update_history": tuple(update_history),
        "global_history": tuple(global_history),
        "history_checksum": history_checksum,
        "model_step_rng_derivation": COHORT_MODEL_STEP_RNG_DERIVATION,
    }
    values["resume_checksum"] = _payload_sha256(_resume_identity_payload(values))
    return CohortRelativeQKVEpochBoundaryResume(**values)


def cohort_epoch_boundary_resume_from_checkpoint(
    payload: Mapping[str, Any],
) -> CohortRelativeQKVEpochBoundaryResume:
    """Reconstruct and checksum-validate a serialized v2 resume payload."""

    def mapping(value: Any, location: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise PooledRelativeQKVTrainingError(
                f"Serialized resume {location} must be a mapping."
            )
        return value

    try:
        core_records = []
        for raw_core in payload["core_history"]:
            core = dict(mapping(raw_core, "core history record"))
            core["mask_views"] = tuple(
                PooledRelativeQKVMaskViewRecord(**dict(mapping(view, "mask view")))
                for view in core["mask_views"]
            )
            core["model_view_seeds"] = tuple(
                int(seed) for seed in core["model_view_seeds"]
            )
            core_records.append(CohortRelativeQKVCoreRecord(**core))
        update_records = []
        for raw_update in payload["optimizer_update_history"]:
            update = dict(mapping(raw_update, "optimizer update record"))
            update["aliases"] = tuple(str(alias) for alias in update["aliases"])
            update_records.append(CohortRelativeQKVOptimizerUpdateRecord(**update))
        global_records = []
        for raw_epoch in payload["global_history"]:
            epoch = dict(mapping(raw_epoch, "global epoch record"))
            epoch["ordered_aliases"] = tuple(
                str(alias) for alias in epoch["ordered_aliases"]
            )
            epoch["ordered_core_pairs"] = tuple(
                tuple(str(alias) for alias in pair)
                for pair in epoch["ordered_core_pairs"]
            )
            global_records.append(CohortRelativeQKVGlobalEpochRecord(**epoch))
        return CohortRelativeQKVEpochBoundaryResume(
            completed_global_epochs=int(payload["completed_global_epochs"]),
            optimizer_updates_completed=int(payload["optimizer_updates_completed"]),
            model_seed=int(payload["model_seed"]),
            mask_base_seed=int(payload["mask_base_seed"]),
            core_order_seed=int(payload["core_order_seed"]),
            cohort_aliases=tuple(str(alias) for alias in payload["cohort_aliases"]),
            cores_per_optimizer_update=int(payload["cores_per_optimizer_update"]),
            optimizer_updates_per_global_epoch=int(
                payload["optimizer_updates_per_global_epoch"]
            ),
            mask_views_per_core=int(payload["mask_views_per_core"]),
            losses_per_optimizer_update=int(payload["losses_per_optimizer_update"]),
            model_state_dict=_clone_tree_to_cpu(
                mapping(payload["model_state_dict"], "model_state_dict")
            ),
            model_state_checksum=str(payload["model_state_checksum"]),
            optimizer_state_dict=_clone_tree_to_cpu(
                mapping(payload["optimizer_state_dict"], "optimizer_state_dict")
            ),
            optimizer_state_checksum=str(payload["optimizer_state_checksum"]),
            scaler_state_dict=_clone_tree_to_cpu(
                mapping(
                    payload.get(
                        "amp_scaler_state_dict", payload.get("scaler_state_dict")
                    ),
                    "amp_scaler_state_dict",
                )
            ),
            scaler_state_checksum=str(
                payload.get(
                    "amp_scaler_state_checksum", payload.get("scaler_state_checksum")
                )
            ),
            core_history=tuple(core_records),
            optimizer_update_history=tuple(update_records),
            global_history=tuple(global_records),
            history_checksum=str(payload["history_checksum"]),
            resume_checksum=str(payload["resume_checksum"]),
            model_step_rng_derivation=str(
                payload.get(
                    "model_step_rng_derivation", COHORT_MODEL_STEP_RNG_DERIVATION
                )
            ),
            resume_schema=str(payload.get("resume_schema", payload.get("schema", ""))),
        )
    except PooledRelativeQKVTrainingError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PooledRelativeQKVTrainingError(
            "Serialized cohort epoch-boundary resume payload is invalid."
        ) from exc


@dataclass(frozen=True)
class CohortRelativeQKVTrainingResult:
    core_history: tuple[CohortRelativeQKVCoreRecord, ...]
    optimizer_update_history: tuple[CohortRelativeQKVOptimizerUpdateRecord, ...]
    global_history: tuple[CohortRelativeQKVGlobalEpochRecord, ...]
    segment_start_global_epoch: int
    segment_end_global_epoch: int
    completed_global_epochs: int
    optimizer_updates_completed: int
    optimizer_updates_this_segment: int
    cores_per_optimizer_update: int
    mask_views_per_core: int
    losses_per_optimizer_update: int
    final_train_loss: float
    final_state_dict: Mapping[str, Tensor] = field(repr=False)
    final_state_checksum: str
    final_optimizer_state_dict: Mapping[str, Any] = field(repr=False)
    final_scaler_state_dict: Mapping[str, Any] = field(repr=False)
    history_checksum: str
    resume: CohortRelativeQKVEpochBoundaryResume
    device: str
    maximum_simultaneously_staged_cores: int
    optimizer: str = "AdamW"
    loss: str = "masked_huber_standardized_log1p"
    validation_or_test_partition_present: bool = False
    checkpoint_selection: str = "none_latest_epoch_boundary_only"


CheckpointCallback = Callable[[CohortRelativeQKVEpochBoundaryResume], None]
EpochCallback = Callable[
    [
        CohortRelativeQKVGlobalEpochRecord,
        tuple[CohortRelativeQKVCoreRecord, ...],
        tuple[CohortRelativeQKVOptimizerUpdateRecord, ...],
        float,
        float,
    ],
    None,
]


def fit_cohort_relative_qkv_segment(
    model: nn.Module,
    core_batches: Sequence[CohortRelativeQKVCoreBatch],
    config: CohortRelativeQKVTrainingConfig,
    *,
    resume: CohortRelativeQKVEpochBoundaryResume | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    epoch_callback: EpochCallback | None = None,
) -> CohortRelativeQKVTrainingResult:
    """Fit a deterministic segment with two cores and 20 losses per update."""

    batches_by_alias = _validate_batches(core_batches, config)
    if resume is None:
        if config.segment_start_global_epoch != 0:
            raise PooledRelativeQKVTrainingError(
                "A nonzero segment start requires a v2 epoch-boundary resume."
            )
        core_history: list[CohortRelativeQKVCoreRecord] = []
        update_history: list[CohortRelativeQKVOptimizerUpdateRecord] = []
        global_history: list[CohortRelativeQKVGlobalEpochRecord] = []
    else:
        if resume.completed_global_epochs != config.segment_start_global_epoch:
            raise PooledRelativeQKVTrainingError(
                "Resume epoch does not equal the configured segment start."
            )
        if (
            resume.model_seed != config.model_seed
            or resume.mask_base_seed != config.mask_base_seed
            or resume.core_order_seed != config.core_order_seed
            or resume.cohort_aliases != config.cohort_aliases
            or resume.cores_per_optimizer_update != config.cores_per_optimizer_update
            or resume.mask_views_per_core != config.mask_views_per_core
            or resume.optimizer_updates_per_global_epoch
            != config.optimizer_updates_per_global_epoch
        ):
            raise PooledRelativeQKVTrainingError(
                "Resume seed/cohort/core/update contract does not match config."
            )
        core_history = list(resume.core_history)
        update_history = list(resume.optimizer_update_history)
        global_history = list(resume.global_history)

    set_deterministic_seed(
        config.model_seed,
        deterministic=config.deterministic,
        warn_only=config.deterministic_warn_only,
    )
    device = _resolve_device(model, config.device)
    if device.type == "cuda":
        # Select this process's device once. Per-view seeding below then touches
        # only this generator and does not create contexts on the other ranks'
        # GPUs (unlike manual_seed_all).
        torch.cuda.set_device(device)
    model.to(device)
    if resume is not None:
        model.load_state_dict(resume.model_state_dict, strict=True)
        if _tree_sha256(model.state_dict()) != resume.model_state_checksum:
            raise PooledRelativeQKVTrainingError(
                "Loaded resume model checksum drifted."
            )

    distributed = config.distributed_world_size == 4
    if distributed:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() != config.distributed_world_size
            or torch.distributed.get_rank() != config.distributed_rank
        ):
            raise PooledRelativeQKVTrainingError(
                "Configured four-rank execution requires a matching initialized "
                "torch.distributed process group."
            )
        ddp_kwargs: dict[str, Any] = {"forward_sync_buffers": False}
        if device.type == "cuda":
            if device.index is None:
                raise PooledRelativeQKVTrainingError(
                    "Distributed CUDA execution requires an indexed CUDA device."
                )
            ddp_kwargs.update(device_ids=[device.index], output_device=device.index)
        training_model: nn.Module = DistributedDataParallel(model, **ddp_kwargs)
    else:
        training_model = model
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = _make_grad_scaler(device, config.amp)
    if resume is not None:
        optimizer.load_state_dict(_clone_tree_to_cpu(resume.optimizer_state_dict))
        scaler.load_state_dict(dict(resume.scaler_state_dict))

    staged_cores = 0
    maximum_staged_cores = 0
    _clear_graph_layout_caches(training_model)
    for global_epoch in range(
        config.segment_start_global_epoch, config.segment_end_global_epoch
    ):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.monotonic()
        order = cohort_relative_qkv_core_order(
            global_epoch,
            aliases=config.cohort_aliases,
            core_order_seed=config.core_order_seed,
        )
        pairs = tuple(
            (order[index], order[index + 1]) for index in range(0, len(order), 2)
        )
        epoch_core_records: list[CohortRelativeQKVCoreRecord] = []
        epoch_update_records: list[CohortRelativeQKVOptimizerUpdateRecord] = []
        for update_index, pair in enumerate(pairs):
            training_model.train()
            optimizer.zero_grad(set_to_none=True)
            pair_records: list[CohortRelativeQKVCoreRecord] = []
            if distributed:
                assignment = cohort_relative_qkv_distributed_assignment(
                    pair,
                    rank=config.distributed_rank,
                    world_size=config.distributed_world_size,
                )
                local_work = (
                    (
                        assignment.position_in_optimizer_update,
                        assignment.alias,
                        assignment.view_indices,
                        assignment.local_loss_divisor,
                    ),
                )
            else:
                local_work = tuple(
                    (
                        pair_position,
                        alias,
                        tuple(range(config.mask_views_per_core)),
                        config.losses_per_optimizer_update,
                    )
                    for pair_position, alias in enumerate(pair)
                )
            local_payloads: list[dict[str, Any]] = []
            for pair_position, alias, view_indices, local_loss_divisor in local_work:
                batch = batches_by_alias[alias]
                _clear_graph_layout_caches(training_model)
                target = edge_index = relative_geometry = node_covariates = None
                target_nodes = input_expression = gene_mask = output = loss = None
                view_records: list[PooledRelativeQKVMaskViewRecord] = []
                model_view_seeds: list[int] = []
                staged_cores += 1
                maximum_staged_cores = max(maximum_staged_cores, staged_cores)
                if staged_cores != 1:
                    raise RuntimeError(
                        "More than one complete core was staged at once."
                    )
                try:
                    dtype = _model_dtype(training_model)
                    target = batch.target_expression.to(device=device, dtype=dtype)
                    if config.stage_complete_core_graph_on_device:
                        edge_index = batch.edge_index.to(
                            device=device, dtype=torch.long
                        )
                        geometry_dtype = {
                            "float16": torch.float16,
                            "bfloat16": torch.bfloat16,
                            "float32": torch.float32,
                        }[config.staged_relative_geometry_dtype]
                        relative_geometry = batch.relative_geometry.to(
                            device=device, dtype=geometry_dtype
                        )
                    else:
                        edge_index = batch.edge_index
                        relative_geometry = batch.relative_geometry
                    node_covariates = batch.node_covariates.to(
                        device=device, dtype=dtype
                    )
                    target_nodes = torch.arange(
                        batch.n_nodes, device=device, dtype=torch.long
                    )
                    for view_index in view_indices:
                        view_seed = cohort_relative_qkv_model_step_seed(
                            config.model_seed,
                            global_epoch,
                            update_index,
                            alias,
                            view_index,
                        )
                        # ``torch.manual_seed`` also seeds every visible CUDA
                        # device. Seed the CPU generator explicitly, then only
                        # this rank's selected CUDA generator.
                        torch.default_generator.manual_seed(view_seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed(view_seed)
                        model_view_seeds.append(view_seed)
                        realization = make_cohort_exact_uniform_training_mask(
                            batch,
                            global_epoch,
                            view_index=view_index,
                            mask_base_seed=config.mask_base_seed,
                        )
                        gene_mask = torch.from_numpy(
                            np.array(realization.mask, copy=True)
                        ).to(device=device, dtype=torch.bool)
                        input_expression = target.masked_fill(gene_mask, 0.0)
                        with _autocast_context(
                            enabled=config.amp,
                            device=device,
                            dtype_name=config.amp_dtype,
                        ):
                            output = training_model(
                                input_expression=input_expression,
                                gene_mask=gene_mask,
                                edge_index=edge_index,
                                relative_geometry=relative_geometry,
                                node_covariates=node_covariates,
                                target_nodes=target_nodes,
                            )
                            prediction = _prediction_tensor(output)
                            if prediction.shape != target.shape:
                                raise PooledRelativeQKVTrainingError(
                                    "Model must decode all genes for every node."
                                )
                            loss = masked_huber_reconstruction_loss(
                                prediction,
                                target,
                                gene_mask,
                                delta=config.huber_delta,
                            )
                        scaler.scale(
                            loss / float(local_loss_divisor)
                        ).backward()
                        counts = realization.masked_gene_counts
                        view_records.append(
                            PooledRelativeQKVMaskViewRecord(
                                view_index=view_index,
                                initial_mask_seed=realization.initial_seed,
                                effective_mask_seed=realization.effective_seed,
                                zero_total_mask_resamples=(
                                    realization.zero_total_resample_count
                                ),
                                mask_checksum_sha256=realization.checksum_sha256,
                                n_masked_entries=realization.n_masked_entries,
                                masked_count_min=int(counts.min()),
                                masked_count_mean=float(counts.mean()),
                                masked_count_median=float(np.median(counts)),
                                masked_count_max=int(counts.max()),
                                zero_mask_cells=int(np.sum(counts == 0)),
                                full_mask_cells=int(np.sum(counts == batch.n_genes)),
                                masked_huber_loss=float(loss.detach().float().cpu()),
                            )
                        )
                        del input_expression, gene_mask, output, loss
                        input_expression = gene_mask = output = loss = None
                    if len(view_records) != len(view_indices):
                        raise RuntimeError(
                            "Rank did not complete its assigned mask views."
                        )
                    local_payloads.append(
                        {
                            "alias": alias,
                            "position": pair_position,
                            "n_nodes": batch.n_nodes,
                            "n_edges": batch.n_edges,
                            "view_records": tuple(view_records),
                            "model_view_seeds": tuple(model_view_seeds),
                        }
                    )
                finally:
                    _clear_graph_layout_caches(training_model)
                    staged_cores -= 1
                    del (
                        target,
                        input_expression,
                        gene_mask,
                        edge_index,
                        relative_geometry,
                        node_covariates,
                        target_nodes,
                        output,
                        loss,
                    )

            if distributed:
                gathered_payloads: list[dict[str, Any] | None] = [
                    None for _ in range(config.distributed_world_size)
                ]
                torch.distributed.all_gather_object(
                    gathered_payloads, local_payloads[0]
                )
                raw_payloads = tuple(
                    payload for payload in gathered_payloads if payload is not None
                )
            else:
                raw_payloads = tuple(local_payloads)
            cumulative_update = (
                global_epoch * config.optimizer_updates_per_global_epoch
                + update_index
                + 1
            )
            for pair_position, alias in enumerate(pair):
                matching = tuple(
                    payload for payload in raw_payloads if payload["alias"] == alias
                )
                if not matching:
                    raise RuntimeError("A core has no gathered mask-view payload.")
                views = tuple(
                    sorted(
                        (
                            view
                            for payload in matching
                            for view in payload["view_records"]
                        ),
                        key=lambda view: view.view_index,
                    )
                )
                model_seeds_by_view = {
                    view.view_index: seed
                    for payload in matching
                    for view, seed in zip(
                        payload["view_records"],
                        payload["model_view_seeds"],
                        strict=True,
                    )
                }
                if tuple(view.view_index for view in views) != tuple(
                    range(config.mask_views_per_core)
                ) or len(model_seeds_by_view) != config.mask_views_per_core:
                    raise RuntimeError(
                        "Gathered DDP payload does not cover ten unique mask views."
                    )
                if len({int(payload["n_nodes"]) for payload in matching}) != 1 or len(
                    {int(payload["n_edges"]) for payload in matching}
                ) != 1:
                    raise RuntimeError("Gathered core shape metadata disagrees.")
                ordered_model_seeds = tuple(
                    model_seeds_by_view[index]
                    for index in range(config.mask_views_per_core)
                )
                record = CohortRelativeQKVCoreRecord(
                    global_epoch=global_epoch,
                    completed_global_epoch=global_epoch + 1,
                    position_in_epoch=update_index * 2 + pair_position,
                    optimizer_update_in_epoch=update_index,
                    cumulative_optimizer_update=cumulative_update,
                    position_in_optimizer_update=pair_position,
                    alias=alias,
                    n_nodes=int(matching[0]["n_nodes"]),
                    n_edges=int(matching[0]["n_edges"]),
                    mask_views=views,
                    n_mask_views=config.mask_views_per_core,
                    n_masked_entries_across_views=sum(
                        view.n_masked_entries for view in views
                    ),
                    model_step_seed=ordered_model_seeds[0],
                    model_view_seeds=ordered_model_seeds,
                    masked_huber_loss=float(
                        np.mean(
                            np.asarray(
                                [view.masked_huber_loss for view in views],
                                dtype=np.float64,
                            )
                        )
                    ),
                )
                pair_records.append(record)
                core_history.append(record)
                epoch_core_records.append(record)
            if tuple(record.alias for record in pair_records) != pair:
                raise RuntimeError("Core pair traversal changed during accumulation.")
            scaler.unscale_(optimizer)
            _assert_finite_gradients(
                training_model, epoch=global_epoch, alias="+".join(pair)
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                training_model.parameters(),
                config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            for name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(
                        "Non-finite parameter after paired optimizer update at "
                        f"epoch {global_epoch}, pair {pair}, {name}."
                    )
            update_record = CohortRelativeQKVOptimizerUpdateRecord(
                global_epoch=global_epoch,
                completed_global_epoch=global_epoch + 1,
                optimizer_update_in_epoch=update_index,
                cumulative_optimizer_update=(
                    global_epoch * config.optimizer_updates_per_global_epoch
                    + update_index
                    + 1
                ),
                aliases=pair,
                n_cores=config.cores_per_optimizer_update,
                mask_views_per_core=config.mask_views_per_core,
                losses_averaged=config.losses_per_optimizer_update,
                equal_core_mean_masked_huber=float(
                    np.mean([record.masked_huber_loss for record in pair_records])
                ),
                gradient_norm=float(gradient_norm.detach().float().cpu()),
            )
            update_history.append(update_record)
            epoch_update_records.append(update_record)

        if (
            len(epoch_core_records) != len(config.cohort_aliases)
            or len(epoch_update_records) != config.optimizer_updates_per_global_epoch
            or tuple(record.alias for record in epoch_core_records) != order
            or tuple(record.aliases for record in epoch_update_records) != pairs
        ):
            raise RuntimeError("Global epoch violated its core/update contract.")
        losses = np.asarray(
            [record.masked_huber_loss for record in epoch_core_records],
            dtype=np.float64,
        )
        gradients = np.asarray(
            [record.gradient_norm for record in epoch_update_records], dtype=np.float64
        )
        if not np.isfinite(losses).all() or not np.isfinite(gradients).all():
            raise FloatingPointError("Global-epoch metrics are non-finite.")
        completed = global_epoch + 1
        epoch_record = CohortRelativeQKVGlobalEpochRecord(
            global_epoch=global_epoch,
            completed_global_epochs=completed,
            ordered_aliases=order,
            ordered_core_pairs=pairs,
            core_records_this_epoch=len(epoch_core_records),
            optimizer_updates_this_epoch=len(epoch_update_records),
            cumulative_optimizer_updates=(
                completed * config.optimizer_updates_per_global_epoch
            ),
            equal_core_mean_masked_huber=float(losses.mean()),
            mean_gradient_norm=float(gradients.mean()),
            max_gradient_norm=float(gradients.max()),
        )
        global_history.append(epoch_record)
        epoch_duration = time.monotonic() - epoch_started
        if not math.isfinite(epoch_duration) or epoch_duration < 0:
            raise RuntimeError("Global-epoch wall-clock duration is invalid.")
        local_peak_bytes = (
            float(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0.0
        )
        if distributed:
            peak_tensor = torch.tensor(
                [local_peak_bytes], device=device, dtype=torch.float64
            )
            torch.distributed.all_reduce(
                peak_tensor, op=torch.distributed.ReduceOp.MAX
            )
            peak_bytes_all_ranks = float(peak_tensor.item())
        else:
            peak_bytes_all_ranks = local_peak_bytes
        peak_vram_gib_all_ranks = peak_bytes_all_ranks / float(1024**3)

        # Rank zero writes observability first, then the durable latest-only
        # checkpoint.  A resume-time CSV reconciler trims any row newer than the
        # checkpoint if a crash occurs between these callbacks.
        if epoch_callback is not None and config.distributed_rank == 0:
            epoch_callback(
                epoch_record,
                tuple(epoch_core_records),
                tuple(epoch_update_records),
                float(epoch_duration),
                float(peak_vram_gib_all_ranks),
            )
        if (
            checkpoint_callback is not None
            and config.distributed_rank == 0
            and completed % config.checkpoint_interval_global_epochs == 0
        ):
            checkpoint_callback(
                _build_resume(
                    model,
                    optimizer,
                    scaler,
                    config=config,
                    completed_global_epochs=completed,
                    core_history=core_history,
                    update_history=update_history,
                    global_history=global_history,
                )
            )
        if distributed:
            torch.distributed.barrier()

    expected_cores = config.segment_end_global_epoch * len(config.cohort_aliases)
    expected_updates = (
        config.segment_end_global_epoch * config.optimizer_updates_per_global_epoch
    )
    if (
        len(core_history) != expected_cores
        or len(update_history) != expected_updates
        or len(global_history) != config.segment_end_global_epoch
    ):
        raise RuntimeError("Cumulative histories are incomplete at segment end.")
    final_resume = _build_resume(
        model,
        optimizer,
        scaler,
        config=config,
        completed_global_epochs=config.segment_end_global_epoch,
        core_history=core_history,
        update_history=update_history,
        global_history=global_history,
    )
    return CohortRelativeQKVTrainingResult(
        core_history=tuple(core_history),
        optimizer_update_history=tuple(update_history),
        global_history=tuple(global_history),
        segment_start_global_epoch=config.segment_start_global_epoch,
        segment_end_global_epoch=config.segment_end_global_epoch,
        completed_global_epochs=config.segment_end_global_epoch,
        optimizer_updates_completed=expected_updates,
        optimizer_updates_this_segment=(
            (config.segment_end_global_epoch - config.segment_start_global_epoch)
            * config.optimizer_updates_per_global_epoch
        ),
        cores_per_optimizer_update=config.cores_per_optimizer_update,
        mask_views_per_core=config.mask_views_per_core,
        losses_per_optimizer_update=config.losses_per_optimizer_update,
        final_train_loss=global_history[-1].equal_core_mean_masked_huber,
        final_state_dict=final_resume.model_state_dict,
        final_state_checksum=final_resume.model_state_checksum,
        final_optimizer_state_dict=final_resume.optimizer_state_dict,
        final_scaler_state_dict=final_resume.scaler_state_dict,
        history_checksum=final_resume.history_checksum,
        resume=final_resume,
        device=str(device),
        maximum_simultaneously_staged_cores=maximum_staged_cores,
    )


__all__ = [
    "COHORT_HISTORY_SCHEMA_V2",
    "COHORT_RESUME_SCHEMA_V2",
    "CohortRelativeQKVCoreBatch",
    "CohortRelativeQKVCoreRecord",
    "CohortRelativeQKVDistributedAssignment",
    "CohortRelativeQKVEpochBoundaryResume",
    "CohortRelativeQKVGlobalEpochRecord",
    "CohortRelativeQKVOptimizerUpdateRecord",
    "CohortRelativeQKVTrainingConfig",
    "CohortRelativeQKVTrainingResult",
    "SO2_14CORE_ALIASES",
    "SO2_CORES_PER_OPTIMIZER_UPDATE",
    "SO2_LOSSES_PER_OPTIMIZER_UPDATE",
    "SO2_OPTIMIZER_UPDATES_PER_EPOCH",
    "cohort_epoch_boundary_resume_from_checkpoint",
    "cohort_relative_qkv_core_order",
    "cohort_relative_qkv_core_pairs",
    "cohort_relative_qkv_distributed_assignment",
    "cohort_relative_qkv_mask_seed",
    "cohort_relative_qkv_model_step_seed",
    "fit_cohort_relative_qkv_segment",
    "make_cohort_exact_uniform_training_mask",
]
