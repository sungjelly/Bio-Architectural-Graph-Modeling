"""Deterministic staged six-core training for relative-geometric QKV models.

One global epoch contains exactly one complete-core optimizer step for each of
``CAN-01``, ``CAN-09``, ``CAN-13``, ``CAN-15``, ``CAN-21``, and ``CAN-23``.
Core payloads remain CPU-resident.  The production hardware profile stages the
current core's complete graph and node tensors on the requested device, while
the receiver-chunked operator still bounds exact attention intermediates; a
CPU graph path remains available for lower-memory hardware.  Each core step
averages exactly ten
independently masked full-graph views before its one optimizer step.  Masks and
core order never depend on model seed; model-side randomness is re-derived for
each ``(model seed, epoch, alias)`` so epoch-boundary continuation is
reproducible.

The trainer executes caller-bounded continuation segments.  The active seed-0
campaign audits its training-loss plateau at 25-epoch boundaries, with the
first eligible stop at epoch 150, two consecutive passing audits, and no
scientific maximum cap.  A joint helper remains available for a deferred
five-seed expansion.  The segment trainer itself never makes a stop decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adjacency_ablation import (
    mask_realization_sha256,
    sample_uniform_mask_numpy,
)
from .masking import derive_mask_seed
from .training import (
    _autocast_context,
    _make_grad_scaler,
    _model_dtype,
    _resolve_device,
    set_deterministic_seed,
)


CAN_ALIASES = ("CAN-01", "CAN-09", "CAN-13", "CAN-15", "CAN-21", "CAN-23")
MASK_BASE_SEED = 2026082401
CORE_ORDER_SEED = 2026082402
STEPS_PER_GLOBAL_EPOCH = len(CAN_ALIASES)
MASK_VIEWS_PER_CORE_STEP = 10
CHECKPOINT_INTERVAL_GLOBAL_EPOCHS = 25
PLATEAU_FIRST_ALLOWED_STOP_EPOCH = 150
PLATEAU_WINDOW_GLOBAL_EPOCHS = 50
PLATEAU_HALF_WINDOW_GLOBAL_EPOCHS = 25
PLATEAU_CONSECUTIVE_PASSING_AUDITS = 2
PLATEAU_RELATIVE_IMPROVEMENT_MAX = 0.002
PLATEAU_NORMALIZED_ABSOLUTE_SLOPE_MAX = 0.0001
MODEL_STEP_RNG_DERIVATION = "model_seed+global_epoch+core_alias"


class PooledRelativeQKVTrainingError(ValueError):
    """Raised when pooled relative-QKV training violates its fixed contract."""


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


def _clone_tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree_to_cpu(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree_to_cpu(child) for child in value)
    if isinstance(value, list):
        return [_clone_tree_to_cpu(child) for child in value]
    return value


def _update_tree_digest(digest: Any, value: Any, *, path: str) -> None:
    digest.update(path.encode("utf-8"))
    digest.update(b"\0")
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_tree_digest(digest, key, path=f"{path}.key")
            _update_tree_digest(digest, value[key], path=f"{path}[{key!r}]")
        return
    if isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii") + b"\0")
        for index, child in enumerate(value):
            _update_tree_digest(digest, child, path=f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        scalar = value.item() if isinstance(value, np.generic) else value
        if isinstance(scalar, float) and not math.isfinite(scalar):
            raise PooledRelativeQKVTrainingError(
                f"Non-finite scalar cannot be checksummed at {path}."
            )
        digest.update(type(scalar).__name__.encode("ascii") + b"\0")
        digest.update(_canonical_json(scalar))
        return
    raise PooledRelativeQKVTrainingError(
        f"Unsupported checkpoint value at {path}: {type(value).__name__}."
    )


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_tree_digest(digest, value, path="root")
    return digest.hexdigest()


@dataclass(frozen=True)
class PooledRelativeQKVCoreBatch:
    """One complete disconnected graph whose tensors must reside on CPU."""

    alias: str
    target_expression: Tensor = field(repr=False)
    edge_index: Tensor = field(repr=False)
    relative_geometry: Tensor = field(repr=False)
    node_covariates: Tensor = field(repr=False)

    def __post_init__(self) -> None:
        alias = str(self.alias).strip().upper()
        if alias not in CAN_ALIASES:
            raise PooledRelativeQKVTrainingError(
                "Core alias must be one of the six locked CAN aliases."
            )
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


def _validate_core_batches(
    batches: Sequence[PooledRelativeQKVCoreBatch],
) -> Mapping[str, PooledRelativeQKVCoreBatch]:
    materialized = tuple(batches)
    aliases = tuple(batch.alias for batch in materialized)
    if len(materialized) != STEPS_PER_GLOBAL_EPOCH or len(set(aliases)) != len(
        aliases
    ):
        raise PooledRelativeQKVTrainingError(
            "Pooled training requires exactly six unique complete-core batches."
        )
    if set(aliases) != set(CAN_ALIASES):
        raise PooledRelativeQKVTrainingError(
            f"Pooled training requires exactly {CAN_ALIASES}."
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
                "All six cores must share gene, metadata, and geometry schemas."
            )
    return {batch.alias: batch for batch in materialized}


@dataclass(frozen=True)
class ExactUniformTrainingMask:
    """One paired mask with deterministic whole-step zero-mask resampling."""

    mask: np.ndarray = field(repr=False)
    masked_gene_counts: np.ndarray = field(repr=False)
    initial_seed: int
    effective_seed: int
    zero_total_resample_count: int
    checksum_sha256: str

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=np.bool_)
        counts = np.asarray(self.masked_gene_counts, dtype=np.int64)
        if mask.ndim != 2 or counts.shape != (mask.shape[0],):
            raise PooledRelativeQKVTrainingError(
                "Mask and per-cell count shapes do not align."
            )
        if not np.array_equal(mask.sum(axis=1, dtype=np.int64), counts):
            raise PooledRelativeQKVTrainingError(
                "Per-cell masked counts do not equal exact row sums."
            )
        if int(mask.sum()) <= 0:
            raise PooledRelativeQKVTrainingError(
                "Training masks must contain at least one masked entry."
            )
        expected = _payload_sha256(
            {
                "schema": "pooled_relative_qkv_exact_uniform_mask_v1",
                "initial_seed": int(self.initial_seed),
                "effective_seed": int(self.effective_seed),
                "zero_total_resample_count": int(self.zero_total_resample_count),
                "realization_checksum": _mask_realization_checksum(
                    mask, counts, seed=int(self.effective_seed)
                ),
            }
        )
        if self.checksum_sha256 != expected:
            raise PooledRelativeQKVTrainingError("Training mask checksum mismatch.")
        mask = np.ascontiguousarray(mask)
        counts = np.ascontiguousarray(counts)
        mask.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "masked_gene_counts", counts)

    @property
    def n_masked_entries(self) -> int:
        return int(self.masked_gene_counts.sum())


def _mask_realization_checksum(
    mask: np.ndarray,
    counts: np.ndarray,
    *,
    seed: int,
) -> str:
    return mask_realization_sha256(mask, counts, seed=int(seed))


def relative_qkv_mask_seed(
    alias: str,
    global_epoch: int,
    *,
    view_index: int,
    mask_base_seed: int = MASK_BASE_SEED,
    resample_attempt: int = 0,
) -> int:
    """Derive a mask seed without accepting or incorporating model seed."""

    canonical_alias = str(alias).strip().upper()
    if canonical_alias not in CAN_ALIASES:
        raise PooledRelativeQKVTrainingError("Unknown CAN alias for mask seed.")
    if (
        int(global_epoch) < 0
        or int(view_index) not in range(MASK_VIEWS_PER_CORE_STEP)
        or int(resample_attempt) < 0
    ):
        raise PooledRelativeQKVTrainingError(
            "global_epoch/resample_attempt must be non-negative and view_index "
            "must be 0 through 9."
        )
    parts: tuple[object, ...] = (
        "relative-qkv-uniform-per-cell-train-mask",
        canonical_alias,
        int(global_epoch),
        int(view_index),
    )
    if int(resample_attempt) > 0:
        parts += ("nonzero-total-resample", int(resample_attempt))
    return derive_mask_seed(int(mask_base_seed), *parts)


def make_exact_uniform_training_mask(
    batch: PooledRelativeQKVCoreBatch,
    global_epoch: int,
    *,
    view_index: int,
    mask_base_seed: int = MASK_BASE_SEED,
    chunk_cells: int = 2048,
    maximum_zero_total_resamples: int = 1024,
) -> ExactUniformTrainingMask:
    """Sample exact Uniform{0,...,G} row masks and reject only all-zero steps."""

    if not isinstance(maximum_zero_total_resamples, Integral) or int(
        maximum_zero_total_resamples
    ) <= 0:
        raise PooledRelativeQKVTrainingError(
            "maximum_zero_total_resamples must be a positive integer."
        )
    initial_seed = relative_qkv_mask_seed(
        batch.alias,
        global_epoch,
        view_index=view_index,
        mask_base_seed=mask_base_seed,
    )
    for attempt in range(int(maximum_zero_total_resamples) + 1):
        seed = relative_qkv_mask_seed(
            batch.alias,
            global_epoch,
            view_index=view_index,
            mask_base_seed=mask_base_seed,
            resample_attempt=attempt,
        )
        realization = sample_uniform_mask_numpy(
            batch.n_nodes,
            batch.n_genes,
            seed=seed,
            chunk_cells=chunk_cells,
        )
        if int(realization.masked_gene_counts.sum()) == 0:
            continue
        checksum = _payload_sha256(
            {
                "schema": "pooled_relative_qkv_exact_uniform_mask_v1",
                "initial_seed": initial_seed,
                "effective_seed": seed,
                "zero_total_resample_count": attempt,
                "realization_checksum": realization.checksum,
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


def relative_qkv_core_order(
    global_epoch: int,
    *,
    aliases: Sequence[str] = CAN_ALIASES,
    core_order_seed: int = CORE_ORDER_SEED,
) -> tuple[str, ...]:
    """Return the deterministic six-core permutation for one global epoch."""

    if int(global_epoch) < 0:
        raise PooledRelativeQKVTrainingError("global_epoch must be non-negative.")
    if int(core_order_seed) != CORE_ORDER_SEED:
        raise PooledRelativeQKVTrainingError(
            f"The locked core-order seed is {CORE_ORDER_SEED}."
        )
    canonical = tuple(sorted(str(alias).strip().upper() for alias in aliases))
    if canonical != CAN_ALIASES:
        raise PooledRelativeQKVTrainingError(
            "Core order requires each locked CAN alias exactly once."
        )
    seed = derive_mask_seed(
        int(core_order_seed), "relative-qkv-core-order", int(global_epoch)
    )
    permutation = np.random.default_rng(seed).permutation(len(canonical))
    return tuple(canonical[int(index)] for index in permutation)


def relative_qkv_model_step_seed(
    model_seed: int,
    global_epoch: int,
    alias: str,
) -> int:
    """Derive resumable model-side RNG independently of the mask schedule."""

    if int(model_seed) < 0 or int(global_epoch) < 0:
        raise PooledRelativeQKVTrainingError(
            "model_seed and global_epoch must be non-negative."
        )
    canonical_alias = str(alias).strip().upper()
    if canonical_alias not in CAN_ALIASES:
        raise PooledRelativeQKVTrainingError("Unknown CAN alias for model RNG.")
    return derive_mask_seed(
        int(model_seed),
        "relative-qkv-model-step",
        int(global_epoch),
        canonical_alias,
    )


@dataclass(frozen=True)
class PooledRelativeQKVTrainingConfig:
    """One bounded queued segment of the plateau-controlled protocol."""

    model_seed: int
    segment_start_global_epoch: int = 0
    segment_end_global_epoch: int = PLATEAU_FIRST_ALLOWED_STOP_EPOCH
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
    checkpoint_interval_global_epochs: int = CHECKPOINT_INTERVAL_GLOBAL_EPOCHS

    def __post_init__(self) -> None:
        for name in (
            "model_seed",
            "segment_start_global_epoch",
            "segment_end_global_epoch",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise PooledRelativeQKVTrainingError(f"{name} must be an integer.")
        if int(self.model_seed) < 0 or int(self.segment_start_global_epoch) < 0:
            raise PooledRelativeQKVTrainingError(
                "model_seed and segment start must be non-negative."
            )
        if int(self.segment_end_global_epoch) <= int(
            self.segment_start_global_epoch
        ):
            raise PooledRelativeQKVTrainingError(
                "Segment end must be greater than segment start."
            )
        for name, allow_zero in (
            ("learning_rate", False),
            ("weight_decay", True),
            ("gradient_clip_norm", False),
            ("huber_delta", False),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (
                value < 0.0 if allow_zero else value <= 0.0
            ):
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
        if int(self.checkpoint_interval_global_epochs) != (
            CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
        ):
            raise PooledRelativeQKVTrainingError(
                "Checkpoint interval is locked to every 25 global epochs."
            )
        dtype = str(self.amp_dtype).lower()
        if dtype not in {"auto", "float16", "bfloat16"}:
            raise PooledRelativeQKVTrainingError(
                "amp_dtype must be auto, float16, or bfloat16."
            )
        object.__setattr__(self, "model_seed", int(self.model_seed))
        object.__setattr__(
            self, "segment_start_global_epoch", int(self.segment_start_global_epoch)
        )
        object.__setattr__(
            self, "segment_end_global_epoch", int(self.segment_end_global_epoch)
        )
        object.__setattr__(self, "amp_dtype", dtype)
        if not isinstance(self.stage_complete_core_graph_on_device, bool):
            raise PooledRelativeQKVTrainingError(
                "stage_complete_core_graph_on_device must be boolean."
            )
        geometry_dtype = str(self.staged_relative_geometry_dtype).lower()
        if geometry_dtype not in {"float16", "bfloat16", "float32"}:
            raise PooledRelativeQKVTrainingError(
                "staged_relative_geometry_dtype must be float16, bfloat16, or float32."
            )
        object.__setattr__(self, "staged_relative_geometry_dtype", geometry_dtype)


@dataclass(frozen=True)
class PooledRelativeQKVMaskViewRecord:
    view_index: int
    initial_mask_seed: int
    effective_mask_seed: int
    zero_total_mask_resamples: int
    mask_checksum_sha256: str
    n_masked_entries: int
    masked_count_min: int
    masked_count_mean: float
    masked_count_median: float
    masked_count_max: int
    zero_mask_cells: int
    full_mask_cells: int
    masked_huber_loss: float


@dataclass(frozen=True)
class PooledRelativeQKVCoreStepRecord:
    global_epoch: int
    completed_global_epoch: int
    step_in_epoch: int
    optimizer_step: int
    alias: str
    n_nodes: int
    n_edges: int
    mask_views: tuple[PooledRelativeQKVMaskViewRecord, ...]
    n_mask_views: int
    n_masked_entries_across_views: int
    model_step_seed: int
    masked_huber_loss: float
    gradient_norm: float


@dataclass(frozen=True)
class PooledRelativeQKVGlobalEpochRecord:
    global_epoch: int
    completed_global_epochs: int
    ordered_aliases: tuple[str, ...]
    optimizer_steps_this_epoch: int
    cumulative_optimizer_steps: int
    equal_core_mean_masked_huber: float
    aggregation: str = "equal_core_arithmetic_mean"


def _history_checksum(
    core_history: Sequence[PooledRelativeQKVCoreStepRecord],
    global_history: Sequence[PooledRelativeQKVGlobalEpochRecord],
) -> str:
    return _payload_sha256(
        {
            "schema": "pooled_relative_qkv_cumulative_history_v1",
            "core_history": [asdict(record) for record in core_history],
            "global_history": [asdict(record) for record in global_history],
        }
    )


@dataclass(frozen=True)
class PooledRelativeQKVEpochBoundaryResume:
    """Complete deterministic continuation state after a global epoch."""

    completed_global_epochs: int
    optimizer_steps_completed: int
    model_seed: int
    mask_base_seed: int
    core_order_seed: int
    mask_views_per_core_step: int
    model_state_dict: Mapping[str, Tensor] = field(repr=False)
    model_state_checksum: str
    optimizer_state_dict: Mapping[str, Any] = field(repr=False)
    optimizer_state_checksum: str
    scaler_state_dict: Mapping[str, Any] = field(repr=False)
    scaler_state_checksum: str
    core_history: tuple[PooledRelativeQKVCoreStepRecord, ...] = field(repr=False)
    global_history: tuple[PooledRelativeQKVGlobalEpochRecord, ...] = field(repr=False)
    history_checksum: str
    resume_checksum: str
    model_step_rng_derivation: str = MODEL_STEP_RNG_DERIVATION

    def __post_init__(self) -> None:
        completed = int(self.completed_global_epochs)
        if completed <= 0:
            raise PooledRelativeQKVTrainingError(
                "Resume state must follow at least one complete global epoch."
            )
        if int(self.optimizer_steps_completed) != completed * STEPS_PER_GLOBAL_EPOCH:
            raise PooledRelativeQKVTrainingError(
                "Resume optimizer-step count does not match completed epochs."
            )
        if len(self.global_history) != completed or len(self.core_history) != (
            completed * STEPS_PER_GLOBAL_EPOCH
        ):
            raise PooledRelativeQKVTrainingError(
                "Resume histories are not cumulative from global epoch zero."
            )
        if int(self.mask_views_per_core_step) != MASK_VIEWS_PER_CORE_STEP:
            raise PooledRelativeQKVTrainingError(
                "Resume mask-view count does not match the fixed ten-view protocol."
            )
        for record in self.core_history:
            if (
                record.n_mask_views != MASK_VIEWS_PER_CORE_STEP
                or len(record.mask_views) != MASK_VIEWS_PER_CORE_STEP
                or tuple(view.view_index for view in record.mask_views)
                != tuple(range(MASK_VIEWS_PER_CORE_STEP))
            ):
                raise PooledRelativeQKVTrainingError(
                    "Resume history lacks the exact ordered ten mask views."
                )
        if _tree_sha256(self.model_state_dict) != self.model_state_checksum:
            raise PooledRelativeQKVTrainingError("Resume model checksum mismatch.")
        if _tree_sha256(self.optimizer_state_dict) != self.optimizer_state_checksum:
            raise PooledRelativeQKVTrainingError("Resume optimizer checksum mismatch.")
        if _tree_sha256(self.scaler_state_dict) != self.scaler_state_checksum:
            raise PooledRelativeQKVTrainingError("Resume scaler checksum mismatch.")
        if _history_checksum(self.core_history, self.global_history) != (
            self.history_checksum
        ):
            raise PooledRelativeQKVTrainingError("Resume history checksum mismatch.")
        expected_resume_checksum = _payload_sha256(
            {
                "schema": "pooled_relative_qkv_epoch_boundary_resume_v1",
                "completed_global_epochs": completed,
                "optimizer_steps_completed": int(self.optimizer_steps_completed),
                "model_seed": int(self.model_seed),
                "mask_base_seed": int(self.mask_base_seed),
                "core_order_seed": int(self.core_order_seed),
                "mask_views_per_core_step": int(self.mask_views_per_core_step),
                "model_state_checksum": self.model_state_checksum,
                "optimizer_state_checksum": self.optimizer_state_checksum,
                "scaler_state_checksum": self.scaler_state_checksum,
                "history_checksum": self.history_checksum,
                "model_step_rng_derivation": self.model_step_rng_derivation,
            }
        )
        if expected_resume_checksum != self.resume_checksum:
            raise PooledRelativeQKVTrainingError("Resume payload checksum mismatch.")
        for checksum in (
            self.model_state_checksum,
            self.optimizer_state_checksum,
            self.scaler_state_checksum,
            self.history_checksum,
            self.resume_checksum,
        ):
            if not _is_sha256(checksum):
                raise PooledRelativeQKVTrainingError("Resume checksum is malformed.")


@dataclass(frozen=True)
class PooledRelativeQKVTrainingResult:
    core_history: tuple[PooledRelativeQKVCoreStepRecord, ...]
    global_history: tuple[PooledRelativeQKVGlobalEpochRecord, ...]
    segment_start_global_epoch: int
    segment_end_global_epoch: int
    completed_global_epochs: int
    optimizer_steps_completed: int
    optimizer_steps_this_segment: int
    mask_views_per_core_step: int
    final_train_loss: float
    final_state_dict: Mapping[str, Tensor] = field(repr=False)
    final_state_checksum: str
    final_optimizer_state_dict: Mapping[str, Any] = field(repr=False)
    final_scaler_state_dict: Mapping[str, Any] = field(repr=False)
    history_checksum: str
    resume: PooledRelativeQKVEpochBoundaryResume
    device: str
    maximum_simultaneously_staged_cores: int = 1
    optimizer: str = "AdamW"
    loss: str = "masked_huber_standardized_log1p"
    validation_or_test_partition_present: bool = False
    checkpoint_selection: str = "none_final_last_epoch_only"
    stopping_decision_owner: str = "external_training_plateau_coordinator"


def _build_resume_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    config: PooledRelativeQKVTrainingConfig,
    completed_global_epochs: int,
    core_history: Sequence[PooledRelativeQKVCoreStepRecord],
    global_history: Sequence[PooledRelativeQKVGlobalEpochRecord],
) -> PooledRelativeQKVEpochBoundaryResume:
    model_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    optimizer_state = _clone_tree_to_cpu(optimizer.state_dict())
    scaler_state = _clone_tree_to_cpu(scaler.state_dict())
    model_checksum = _tree_sha256(model_state)
    optimizer_checksum = _tree_sha256(optimizer_state)
    scaler_checksum = _tree_sha256(scaler_state)
    core_records = tuple(core_history)
    global_records = tuple(global_history)
    history_checksum = _history_checksum(core_records, global_records)
    payload = {
        "schema": "pooled_relative_qkv_epoch_boundary_resume_v1",
        "completed_global_epochs": int(completed_global_epochs),
        "optimizer_steps_completed": int(completed_global_epochs)
        * STEPS_PER_GLOBAL_EPOCH,
        "model_seed": config.model_seed,
        "mask_base_seed": config.mask_base_seed,
        "core_order_seed": config.core_order_seed,
        "mask_views_per_core_step": MASK_VIEWS_PER_CORE_STEP,
        "model_state_checksum": model_checksum,
        "optimizer_state_checksum": optimizer_checksum,
        "scaler_state_checksum": scaler_checksum,
        "history_checksum": history_checksum,
        "model_step_rng_derivation": MODEL_STEP_RNG_DERIVATION,
    }
    return PooledRelativeQKVEpochBoundaryResume(
        completed_global_epochs=int(completed_global_epochs),
        optimizer_steps_completed=int(completed_global_epochs)
        * STEPS_PER_GLOBAL_EPOCH,
        model_seed=config.model_seed,
        mask_base_seed=config.mask_base_seed,
        core_order_seed=config.core_order_seed,
        mask_views_per_core_step=MASK_VIEWS_PER_CORE_STEP,
        model_state_dict=model_state,
        model_state_checksum=model_checksum,
        optimizer_state_dict=optimizer_state,
        optimizer_state_checksum=optimizer_checksum,
        scaler_state_dict=scaler_state,
        scaler_state_checksum=scaler_checksum,
        core_history=core_records,
        global_history=global_records,
        history_checksum=history_checksum,
        resume_checksum=_payload_sha256(payload),
    )


def masked_huber_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    delta: float = 1.0,
) -> Tensor:
    """Return FP32 mean Huber loss over exactly the masked entries."""

    if prediction.shape != target.shape or mask.shape != target.shape:
        raise PooledRelativeQKVTrainingError(
            "Prediction, target, and mask shapes must match."
        )
    if mask.dtype != torch.bool:
        raise PooledRelativeQKVTrainingError("mask must be boolean.")
    n_masked = int(mask.sum().detach().cpu())
    if n_masked <= 0:
        raise PooledRelativeQKVTrainingError(
            "Masked Huber cannot evaluate a zero-total mask."
        )
    elementwise = F.huber_loss(
        prediction.float(), target.float(), reduction="none", delta=float(delta)
    )
    loss = elementwise.masked_select(mask).sum(dtype=torch.float32) / float(n_masked)
    if not bool(torch.isfinite(loss).detach().cpu()):
        raise FloatingPointError("Masked Huber loss is non-finite.")
    return loss


def _prediction_tensor(output: Any) -> Tensor:
    if torch.is_tensor(output):
        prediction = output
    elif hasattr(output, "prediction"):
        prediction = output.prediction
    elif isinstance(output, Mapping) and "prediction" in output:
        prediction = output["prediction"]
    else:
        raise PooledRelativeQKVTrainingError(
            "Model forward must return a tensor or an object with prediction."
        )
    if not torch.is_tensor(prediction):
        raise PooledRelativeQKVTrainingError("Model prediction must be a tensor.")
    return prediction


def _clear_graph_layout_caches(model: nn.Module) -> None:
    for module in model.modules():
        clear = getattr(module, "clear_edge_layout_cache", None)
        if callable(clear):
            clear()


def _assert_finite_gradients(model: nn.Module, *, epoch: int, alias: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all()
        ):
            raise FloatingPointError(
                f"Non-finite gradient at epoch {epoch}, {alias}, parameter {name}."
            )


CheckpointCallback = Callable[[PooledRelativeQKVEpochBoundaryResume], None]


def fit_pooled_relative_qkv_segment(
    model: nn.Module,
    core_batches: Sequence[PooledRelativeQKVCoreBatch],
    config: PooledRelativeQKVTrainingConfig,
    *,
    resume: PooledRelativeQKVEpochBoundaryResume | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> PooledRelativeQKVTrainingResult:
    """Fit one queued epoch segment using six staged complete-core steps/epoch."""

    batches_by_alias = _validate_core_batches(core_batches)
    if resume is None:
        if config.segment_start_global_epoch != 0:
            raise PooledRelativeQKVTrainingError(
                "A nonzero segment start requires an epoch-boundary resume payload."
            )
        core_history: list[PooledRelativeQKVCoreStepRecord] = []
        global_history: list[PooledRelativeQKVGlobalEpochRecord] = []
    else:
        if resume.completed_global_epochs != config.segment_start_global_epoch:
            raise PooledRelativeQKVTrainingError(
                "Resume epoch does not equal the queued segment start."
            )
        if (
            resume.model_seed != config.model_seed
            or resume.mask_base_seed != config.mask_base_seed
            or resume.core_order_seed != config.core_order_seed
        ):
            raise PooledRelativeQKVTrainingError(
                "Resume seeds do not match the queued continuation config."
            )
        core_history = list(resume.core_history)
        global_history = list(resume.global_history)

    set_deterministic_seed(
        config.model_seed,
        deterministic=config.deterministic,
        warn_only=config.deterministic_warn_only,
    )
    device = _resolve_device(model, config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = _make_grad_scaler(device, config.amp)
    if resume is not None:
        model.load_state_dict(resume.model_state_dict, strict=True)
        optimizer.load_state_dict(_clone_tree_to_cpu(resume.optimizer_state_dict))
        scaler.load_state_dict(dict(resume.scaler_state_dict))
        if _tree_sha256(model.state_dict()) != resume.model_state_checksum:
            raise PooledRelativeQKVTrainingError(
                "Loaded resume model checksum does not match."
            )

    aliases = tuple(batches_by_alias)
    staged_cores = 0
    maximum_staged_cores = 0
    _clear_graph_layout_caches(model)
    for global_epoch in range(
        config.segment_start_global_epoch, config.segment_end_global_epoch
    ):
        order = relative_qkv_core_order(
            global_epoch,
            aliases=aliases,
            core_order_seed=config.core_order_seed,
        )
        epoch_records: list[PooledRelativeQKVCoreStepRecord] = []
        for step_in_epoch, alias in enumerate(order):
            batch = batches_by_alias[alias]
            step_seed = relative_qkv_model_step_seed(
                config.model_seed, global_epoch, alias
            )
            torch.manual_seed(step_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(step_seed)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            _clear_graph_layout_caches(model)

            target = edge_index = relative_geometry = node_covariates = None
            target_nodes = input_expression = gene_mask = output = loss = None
            view_records: list[PooledRelativeQKVMaskViewRecord] = []
            staged_cores += 1
            maximum_staged_cores = max(maximum_staged_cores, staged_cores)
            if staged_cores != 1:
                raise RuntimeError("More than one complete core was staged at once.")
            try:
                dtype = _model_dtype(model)
                target = batch.target_expression.to(device=device, dtype=dtype)
                if config.stage_complete_core_graph_on_device:
                    # Preflight proved the largest graph fits.  One current
                    # core stays resident so ten views do not repeat the same
                    # PCIe transfers; exact receiver chunking is unchanged.
                    edge_index = batch.edge_index.to(
                        device=device,
                        dtype=torch.long,
                    )
                    geometry_dtype = {
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "float32": torch.float32,
                    }[config.staged_relative_geometry_dtype]
                    relative_geometry = batch.relative_geometry.to(
                        device=device,
                        dtype=geometry_dtype,
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
                for view_index in range(MASK_VIEWS_PER_CORE_STEP):
                    mask_realization = make_exact_uniform_training_mask(
                        batch,
                        global_epoch,
                        view_index=view_index,
                        mask_base_seed=config.mask_base_seed,
                    )
                    gene_mask = torch.from_numpy(
                        np.array(mask_realization.mask, copy=True)
                    ).to(device=device, dtype=torch.bool)
                    input_expression = target.masked_fill(gene_mask, 0.0)
                    with _autocast_context(
                        enabled=config.amp,
                        device=device,
                        dtype_name=config.amp_dtype,
                    ):
                        output = model(
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
                                "Model must decode all genes for every target node."
                            )
                        loss = masked_huber_reconstruction_loss(
                            prediction,
                            target,
                            gene_mask,
                            delta=config.huber_delta,
                        )
                    scaler.scale(loss / float(MASK_VIEWS_PER_CORE_STEP)).backward()
                    counts = mask_realization.masked_gene_counts
                    view_records.append(
                        PooledRelativeQKVMaskViewRecord(
                            view_index=view_index,
                            initial_mask_seed=mask_realization.initial_seed,
                            effective_mask_seed=mask_realization.effective_seed,
                            zero_total_mask_resamples=(
                                mask_realization.zero_total_resample_count
                            ),
                            mask_checksum_sha256=(
                                mask_realization.checksum_sha256
                            ),
                            n_masked_entries=(
                                mask_realization.n_masked_entries
                            ),
                            masked_count_min=int(counts.min()),
                            masked_count_mean=float(counts.mean()),
                            masked_count_median=float(np.median(counts)),
                            masked_count_max=int(counts.max()),
                            zero_mask_cells=int(np.sum(counts == 0)),
                            full_mask_cells=int(
                                np.sum(counts == batch.n_genes)
                            ),
                            masked_huber_loss=float(
                                loss.detach().float().cpu()
                            ),
                        )
                    )
                    del input_expression, gene_mask, output, loss
                    input_expression = gene_mask = output = loss = None
                if len(view_records) != MASK_VIEWS_PER_CORE_STEP:
                    raise RuntimeError(
                        "A core step did not complete exactly ten mask views."
                    )
                scaler.unscale_(optimizer)
                _assert_finite_gradients(model, epoch=global_epoch, alias=alias)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
                for name, parameter in model.named_parameters():
                    if not bool(torch.isfinite(parameter).all()):
                        raise FloatingPointError(
                            "Non-finite parameter after optimizer step at "
                            f"epoch {global_epoch}, {alias}, {name}."
                        )
                record = PooledRelativeQKVCoreStepRecord(
                    global_epoch=global_epoch,
                    completed_global_epoch=global_epoch + 1,
                    step_in_epoch=step_in_epoch,
                    optimizer_step=(
                        global_epoch * STEPS_PER_GLOBAL_EPOCH + step_in_epoch + 1
                    ),
                    alias=alias,
                    n_nodes=batch.n_nodes,
                    n_edges=batch.n_edges,
                    mask_views=tuple(view_records),
                    n_mask_views=MASK_VIEWS_PER_CORE_STEP,
                    n_masked_entries_across_views=sum(
                        view.n_masked_entries for view in view_records
                    ),
                    model_step_seed=step_seed,
                    masked_huber_loss=float(
                        np.mean(
                            np.asarray(
                                [
                                    view.masked_huber_loss
                                    for view in view_records
                                ],
                                dtype=np.float64,
                            )
                        )
                    ),
                    gradient_norm=float(gradient_norm.detach().float().cpu()),
                )
                core_history.append(record)
                epoch_records.append(record)
            finally:
                _clear_graph_layout_caches(model)
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

        if tuple(record.alias for record in epoch_records) != order:
            raise RuntimeError("Core visit order changed during a global epoch.")
        if len(epoch_records) != STEPS_PER_GLOBAL_EPOCH or set(order) != set(
            CAN_ALIASES
        ):
            raise RuntimeError(
                "A global epoch did not contain exactly six complete-core steps."
            )
        losses = np.asarray(
            [record.masked_huber_loss for record in epoch_records], dtype=np.float64
        )
        if not np.isfinite(losses).all():
            raise FloatingPointError("Global-epoch training losses are non-finite.")
        completed = global_epoch + 1
        global_history.append(
            PooledRelativeQKVGlobalEpochRecord(
                global_epoch=global_epoch,
                completed_global_epochs=completed,
                ordered_aliases=order,
                optimizer_steps_this_epoch=STEPS_PER_GLOBAL_EPOCH,
                cumulative_optimizer_steps=completed * STEPS_PER_GLOBAL_EPOCH,
                equal_core_mean_masked_huber=float(losses.mean()),
            )
        )
        if (
            checkpoint_callback is not None
            and completed % CHECKPOINT_INTERVAL_GLOBAL_EPOCHS == 0
        ):
            checkpoint_callback(
                _build_resume_payload(
                    model,
                    optimizer,
                    scaler,
                    config=config,
                    completed_global_epochs=completed,
                    core_history=core_history,
                    global_history=global_history,
                )
            )

    expected_history_length = (
        config.segment_end_global_epoch * STEPS_PER_GLOBAL_EPOCH
    )
    if len(core_history) != expected_history_length or len(global_history) != (
        config.segment_end_global_epoch
    ):
        raise RuntimeError("Cumulative histories are incomplete at segment end.")
    final_resume = _build_resume_payload(
        model,
        optimizer,
        scaler,
        config=config,
        completed_global_epochs=config.segment_end_global_epoch,
        core_history=core_history,
        global_history=global_history,
    )
    return PooledRelativeQKVTrainingResult(
        core_history=tuple(core_history),
        global_history=tuple(global_history),
        segment_start_global_epoch=config.segment_start_global_epoch,
        segment_end_global_epoch=config.segment_end_global_epoch,
        completed_global_epochs=config.segment_end_global_epoch,
        optimizer_steps_completed=(
            config.segment_end_global_epoch * STEPS_PER_GLOBAL_EPOCH
        ),
        optimizer_steps_this_segment=(
            (config.segment_end_global_epoch - config.segment_start_global_epoch)
            * STEPS_PER_GLOBAL_EPOCH
        ),
        mask_views_per_core_step=MASK_VIEWS_PER_CORE_STEP,
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


@dataclass(frozen=True)
class PlateauAudit:
    model_seed: int
    completed_global_epochs: int
    previous_25_mean: float | None
    last_25_mean: float | None
    relative_mean_improvement: float | None
    linear_slope_per_epoch: float | None
    window_mean: float | None
    normalized_absolute_slope_per_epoch: float | None
    all_window_losses_finite: bool
    conditions_passed: bool
    eligible_for_stopping: bool
    qualifying_passed: bool
    audit_metric: str = "equal_core_mean_training_masked_huber"
    validation_or_test_metric: bool = False


def audit_training_loss_plateau(
    losses: Sequence[float] | np.ndarray,
    *,
    model_seed: int,
    completed_global_epochs: int | None = None,
) -> PlateauAudit:
    """Audit one seed's last 50 equal-core training-loss epochs."""

    values = np.asarray(losses, dtype=np.float64)
    completed = len(values) if completed_global_epochs is None else int(
        completed_global_epochs
    )
    if values.ndim != 1 or completed != len(values):
        raise PooledRelativeQKVTrainingError(
            "Plateau loss history must be one value per completed global epoch."
        )
    if completed < PLATEAU_WINDOW_GLOBAL_EPOCHS or completed % (
        CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
    ) != 0:
        raise PooledRelativeQKVTrainingError(
            "Plateau audits require at least 50 epochs at a 25-epoch boundary."
        )
    window = values[-PLATEAU_WINDOW_GLOBAL_EPOCHS:]
    finite = bool(np.isfinite(window).all())
    eligible = completed >= PLATEAU_FIRST_ALLOWED_STOP_EPOCH
    if not finite:
        return PlateauAudit(
            model_seed=int(model_seed),
            completed_global_epochs=completed,
            previous_25_mean=None,
            last_25_mean=None,
            relative_mean_improvement=None,
            linear_slope_per_epoch=None,
            window_mean=None,
            normalized_absolute_slope_per_epoch=None,
            all_window_losses_finite=False,
            conditions_passed=False,
            eligible_for_stopping=eligible,
            qualifying_passed=False,
        )
    previous = float(window[:PLATEAU_HALF_WINDOW_GLOBAL_EPOCHS].mean())
    recent = float(window[PLATEAU_HALF_WINDOW_GLOBAL_EPOCHS:].mean())
    window_mean = float(window.mean())
    if previous == 0.0:
        relative_improvement = 0.0 if recent == 0.0 else float("-inf")
    else:
        relative_improvement = (previous - recent) / abs(previous)
    x = np.arange(PLATEAU_WINDOW_GLOBAL_EPOCHS, dtype=np.float64)
    x_centered = x - x.mean()
    slope = float(np.dot(x_centered, window - window_mean) / np.dot(x_centered, x_centered))
    normalized_slope = (
        abs(slope) / abs(window_mean)
        if window_mean != 0.0
        else (0.0 if slope == 0.0 else float("inf"))
    )
    conditions_passed = bool(
        relative_improvement <= PLATEAU_RELATIVE_IMPROVEMENT_MAX
        and normalized_slope <= PLATEAU_NORMALIZED_ABSOLUTE_SLOPE_MAX
    )
    return PlateauAudit(
        model_seed=int(model_seed),
        completed_global_epochs=completed,
        previous_25_mean=previous,
        last_25_mean=recent,
        relative_mean_improvement=float(relative_improvement),
        linear_slope_per_epoch=slope,
        window_mean=window_mean,
        normalized_absolute_slope_per_epoch=float(normalized_slope),
        all_window_losses_finite=True,
        conditions_passed=conditions_passed,
        eligible_for_stopping=eligible,
        qualifying_passed=bool(eligible and conditions_passed),
    )


@dataclass(frozen=True)
class SingleSeedPlateauDecision:
    """Two-consecutive-audit stopping decision for one production seed."""

    model_seed: int
    completed_global_epochs: int
    current_audit: PlateauAudit
    previous_audit: PlateauAudit
    consecutive_passing_audits: int
    should_stop: bool
    final_epoch: int | None
    continue_for_global_epochs: int
    stopping_metric_role: str = "training_loss_plateau_not_checkpoint_selection"


def single_seed_plateau_decision(
    losses: Sequence[float] | np.ndarray,
    *,
    model_seed: int = 0,
    completed_global_epochs: int,
) -> SingleSeedPlateauDecision:
    """Apply the active two-audit plateau rule to one aligned loss history."""

    completed = int(completed_global_epochs)
    if completed < PLATEAU_FIRST_ALLOWED_STOP_EPOCH or completed % (
        CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
    ) != 0:
        raise PooledRelativeQKVTrainingError(
            "Single-seed stopping decisions begin at epoch 150 on a "
            "25-epoch boundary."
        )
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or len(values) != completed:
        raise PooledRelativeQKVTrainingError(
            "The single-seed loss history must align to the completed epoch."
        )
    previous_completed = completed - CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
    current = audit_training_loss_plateau(
        values,
        model_seed=int(model_seed),
        completed_global_epochs=completed,
    )
    previous = audit_training_loss_plateau(
        values[:previous_completed],
        model_seed=int(model_seed),
        completed_global_epochs=previous_completed,
    )
    current_pass = current.qualifying_passed
    previous_pass = previous.qualifying_passed
    consecutive = 2 if current_pass and previous_pass else (1 if current_pass else 0)
    should_stop = bool(
        consecutive >= PLATEAU_CONSECUTIVE_PASSING_AUDITS
    )
    return SingleSeedPlateauDecision(
        model_seed=int(model_seed),
        completed_global_epochs=completed,
        current_audit=current,
        previous_audit=previous,
        consecutive_passing_audits=consecutive,
        should_stop=should_stop,
        final_epoch=completed if should_stop else None,
        continue_for_global_epochs=(
            0 if should_stop else CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
        ),
    )


@dataclass(frozen=True)
class JointFiveSeedPlateauDecision:
    completed_global_epochs: int
    current_audits: Mapping[int, PlateauAudit]
    previous_audits: Mapping[int, PlateauAudit]
    every_seed_currently_passes: bool
    every_seed_previously_passed: bool
    consecutive_joint_passing_audits: int
    should_stop_all_five: bool
    common_final_epoch: int | None
    continue_all_five_for_global_epochs: int
    uncertainty_label: str = "seed uncertainty"
    stopping_metric_role: str = "training_loss_plateau_not_checkpoint_selection"


def joint_five_seed_plateau_decision(
    losses_by_model_seed: Mapping[int, Sequence[float] | np.ndarray],
    *,
    completed_global_epochs: int,
) -> JointFiveSeedPlateauDecision:
    """Apply the common-epoch, two-audit plateau rule to seeds 0 through 4."""

    if set(losses_by_model_seed) != set(range(5)):
        raise PooledRelativeQKVTrainingError(
            "Joint plateau requires exactly model seeds 0, 1, 2, 3, and 4."
        )
    completed = int(completed_global_epochs)
    if completed < PLATEAU_FIRST_ALLOWED_STOP_EPOCH or completed % (
        CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
    ) != 0:
        raise PooledRelativeQKVTrainingError(
            "Joint plateau decisions begin at epoch 150 on 25-epoch boundaries."
        )
    previous_completed = completed - CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
    current_audits: dict[int, PlateauAudit] = {}
    previous_audits: dict[int, PlateauAudit] = {}
    for seed in range(5):
        values = np.asarray(losses_by_model_seed[seed], dtype=np.float64)
        if values.ndim != 1 or len(values) != completed:
            raise PooledRelativeQKVTrainingError(
                "All five loss histories must align to the common completed epoch."
            )
        current_audits[seed] = audit_training_loss_plateau(
            values,
            model_seed=seed,
            completed_global_epochs=completed,
        )
        previous_audits[seed] = audit_training_loss_plateau(
            values[:previous_completed],
            model_seed=seed,
            completed_global_epochs=previous_completed,
        )
    current_pass = all(
        audit.qualifying_passed for audit in current_audits.values()
    )
    previous_pass = all(
        audit.qualifying_passed for audit in previous_audits.values()
    )
    consecutive = 2 if current_pass and previous_pass else (1 if current_pass else 0)
    should_stop = bool(
        consecutive >= PLATEAU_CONSECUTIVE_PASSING_AUDITS
    )
    return JointFiveSeedPlateauDecision(
        completed_global_epochs=completed,
        current_audits=current_audits,
        previous_audits=previous_audits,
        every_seed_currently_passes=current_pass,
        every_seed_previously_passed=previous_pass,
        consecutive_joint_passing_audits=consecutive,
        should_stop_all_five=should_stop,
        common_final_epoch=completed if should_stop else None,
        continue_all_five_for_global_epochs=(
            0 if should_stop else CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
        ),
    )


__all__ = [
    "CAN_ALIASES",
    "CHECKPOINT_INTERVAL_GLOBAL_EPOCHS",
    "CORE_ORDER_SEED",
    "MASK_BASE_SEED",
    "MASK_VIEWS_PER_CORE_STEP",
    "MODEL_STEP_RNG_DERIVATION",
    "PLATEAU_CONSECUTIVE_PASSING_AUDITS",
    "PLATEAU_FIRST_ALLOWED_STOP_EPOCH",
    "PLATEAU_NORMALIZED_ABSOLUTE_SLOPE_MAX",
    "PLATEAU_RELATIVE_IMPROVEMENT_MAX",
    "PLATEAU_WINDOW_GLOBAL_EPOCHS",
    "STEPS_PER_GLOBAL_EPOCH",
    "ExactUniformTrainingMask",
    "JointFiveSeedPlateauDecision",
    "PlateauAudit",
    "PooledRelativeQKVCoreBatch",
    "PooledRelativeQKVCoreStepRecord",
    "PooledRelativeQKVMaskViewRecord",
    "PooledRelativeQKVEpochBoundaryResume",
    "PooledRelativeQKVGlobalEpochRecord",
    "PooledRelativeQKVTrainingConfig",
    "PooledRelativeQKVTrainingError",
    "PooledRelativeQKVTrainingResult",
    "SingleSeedPlateauDecision",
    "audit_training_loss_plateau",
    "fit_pooled_relative_qkv_segment",
    "joint_five_seed_plateau_decision",
    "make_exact_uniform_training_mask",
    "masked_huber_reconstruction_loss",
    "relative_qkv_core_order",
    "relative_qkv_mask_seed",
    "relative_qkv_model_step_seed",
    "single_seed_plateau_decision",
]
