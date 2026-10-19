"""Scalar-only full-model and per-block gradient direction observability.

The trackers are deliberately outside optimizer state and checkpoint payloads.
They read unscaled, finite gradients immediately before clipping, retain only a
bounded set of transient FP32 vectors in process memory, and expose scalar
summaries per completed epoch.  The durable writers atomically publish those
summaries without ever serializing a gradient tensor.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


GRADIENT_DIRECTION_METRICS_SCHEMA = "so2_full_gradient_direction_metrics_v1"
BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA = (
    "so2_block_gradient_direction_metrics_v1"
)
GRADIENT_DIRECTION_METRICS_COLUMNS = (
    "schema",
    "run_id",
    "model_seed",
    "global_epoch",
    "trainable_parameter_count",
    "optimizer_updates_observed",
    "gradient_norm_mean_before_clip",
    "gradient_norm_min_before_clip",
    "gradient_norm_max_before_clip",
    "consecutive_optimizer_step_cosine_mean",
    "consecutive_optimizer_step_cosine_median",
    "consecutive_optimizer_step_cosine_min",
    "consecutive_optimizer_step_cosine_max",
    "consecutive_optimizer_step_cosine_valid_pairs",
    "epoch_aggregate_gradient_cosine_to_previous_epoch",
    "resume_boundary_unavailable",
)
BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS = (
    "schema",
    "run_id",
    "model_seed",
    "global_epoch",
    "block_index",
    "block_name",
    "trainable_parameter_count",
    "optimizer_updates_observed",
    "gradient_norm_mean_before_clip",
    "gradient_norm_min_before_clip",
    "gradient_norm_max_before_clip",
    "consecutive_optimizer_step_cosine_mean",
    "consecutive_optimizer_step_cosine_median",
    "consecutive_optimizer_step_cosine_min",
    "consecutive_optimizer_step_cosine_max",
    "consecutive_optimizer_step_cosine_valid_pairs",
    "epoch_aggregate_gradient_cosine_to_previous_epoch",
    "resume_boundary_unavailable",
)


class GradientDirectionObservabilityError(RuntimeError):
    """Raised when gradient-direction diagnostics would be ambiguous."""


@dataclass(frozen=True)
class GradientDirectionUpdateContext:
    """Identity of one unscaled pre-clip optimizer-update gradient.

    ``global_epoch`` and ``optimizer_update_in_epoch`` are zero based, matching
    the cohort trainer's records.  Their corresponding completed/cumulative
    fields are one based.
    """

    global_epoch: int
    completed_global_epoch: int
    optimizer_update_in_epoch: int
    cumulative_optimizer_update: int
    aliases: tuple[str, str]

    def __post_init__(self) -> None:
        if int(self.global_epoch) < 0:
            raise GradientDirectionObservabilityError(
                "global_epoch must be non-negative."
            )
        if int(self.completed_global_epoch) != int(self.global_epoch) + 1:
            raise GradientDirectionObservabilityError(
                "completed_global_epoch must equal global_epoch + 1."
            )
        if int(self.optimizer_update_in_epoch) < 0:
            raise GradientDirectionObservabilityError(
                "optimizer_update_in_epoch must be non-negative."
            )
        if int(self.cumulative_optimizer_update) <= 0:
            raise GradientDirectionObservabilityError(
                "cumulative_optimizer_update must be positive."
            )
        aliases = tuple(str(alias).strip().upper() for alias in self.aliases)
        if len(aliases) != 2 or any(not alias for alias in aliases):
            raise GradientDirectionObservabilityError(
                "aliases must identify exactly two non-empty cores."
            )
        object.__setattr__(self, "global_epoch", int(self.global_epoch))
        object.__setattr__(
            self, "completed_global_epoch", int(self.completed_global_epoch)
        )
        object.__setattr__(
            self, "optimizer_update_in_epoch", int(self.optimizer_update_in_epoch)
        )
        object.__setattr__(
            self, "cumulative_optimizer_update", int(self.cumulative_optimizer_update)
        )
        object.__setattr__(self, "aliases", aliases)


def _optional_cosine(value: float | None, *, field: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < -1.0 or result > 1.0:
        raise GradientDirectionObservabilityError(
            f"{field} must be finite in [-1, 1] when present."
        )
    return result


@dataclass(frozen=True)
class GradientDirectionEpochSummary:
    """One scalar-only summary for a completed, one-based global epoch."""

    global_epoch: int
    trainable_parameter_count: int
    optimizer_updates_observed: int
    gradient_norm_mean_before_clip: float
    gradient_norm_min_before_clip: float
    gradient_norm_max_before_clip: float
    consecutive_optimizer_step_cosine_mean: float | None
    consecutive_optimizer_step_cosine_median: float | None
    consecutive_optimizer_step_cosine_min: float | None
    consecutive_optimizer_step_cosine_max: float | None
    consecutive_optimizer_step_cosine_valid_pairs: int
    epoch_aggregate_gradient_cosine_to_previous_epoch: float | None
    resume_boundary_unavailable: bool
    schema: str = GRADIENT_DIRECTION_METRICS_SCHEMA

    def __post_init__(self) -> None:
        epoch = int(self.global_epoch)
        parameters = int(self.trainable_parameter_count)
        updates = int(self.optimizer_updates_observed)
        valid_pairs = int(self.consecutive_optimizer_step_cosine_valid_pairs)
        if epoch <= 0 or parameters <= 0 or updates <= 0:
            raise GradientDirectionObservabilityError(
                "Epoch, parameter count, and update count must be positive."
            )
        if valid_pairs < 0 or valid_pairs > updates:
            raise GradientDirectionObservabilityError(
                "The valid consecutive-pair count is inconsistent."
            )
        norms = tuple(
            float(value)
            for value in (
                self.gradient_norm_mean_before_clip,
                self.gradient_norm_min_before_clip,
                self.gradient_norm_max_before_clip,
            )
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in norms):
            raise GradientDirectionObservabilityError(
                "Gradient norms must be finite and non-negative."
            )
        if not norms[1] <= norms[0] <= norms[2]:
            raise GradientDirectionObservabilityError(
                "Gradient norm min/mean/max ordering is invalid."
            )
        cosines = tuple(
            _optional_cosine(value, field=field)
            for value, field in (
                (
                    self.consecutive_optimizer_step_cosine_mean,
                    "consecutive_optimizer_step_cosine_mean",
                ),
                (
                    self.consecutive_optimizer_step_cosine_median,
                    "consecutive_optimizer_step_cosine_median",
                ),
                (
                    self.consecutive_optimizer_step_cosine_min,
                    "consecutive_optimizer_step_cosine_min",
                ),
                (
                    self.consecutive_optimizer_step_cosine_max,
                    "consecutive_optimizer_step_cosine_max",
                ),
            )
        )
        if valid_pairs == 0 and any(value is not None for value in cosines):
            raise GradientDirectionObservabilityError(
                "Cosine statistics must be absent when no pair is valid."
            )
        if valid_pairs > 0 and any(value is None for value in cosines):
            raise GradientDirectionObservabilityError(
                "Cosine statistics must be present when a pair is valid."
            )
        if valid_pairs > 0:
            assert all(value is not None for value in cosines)
            mean, median, minimum, maximum = (
                float(value) for value in cosines if value is not None
            )
            if not minimum <= mean <= maximum or not minimum <= median <= maximum:
                raise GradientDirectionObservabilityError(
                    "Consecutive cosine min/center/max ordering is invalid."
                )
        aggregate_cosine = _optional_cosine(
            self.epoch_aggregate_gradient_cosine_to_previous_epoch,
            field="epoch_aggregate_gradient_cosine_to_previous_epoch",
        )
        if not isinstance(self.resume_boundary_unavailable, bool):
            raise GradientDirectionObservabilityError(
                "resume_boundary_unavailable must be boolean."
            )
        if self.schema != GRADIENT_DIRECTION_METRICS_SCHEMA:
            raise GradientDirectionObservabilityError(
                "Gradient-direction summary schema mismatch."
            )
        object.__setattr__(self, "global_epoch", epoch)
        object.__setattr__(self, "trainable_parameter_count", parameters)
        object.__setattr__(self, "optimizer_updates_observed", updates)
        object.__setattr__(
            self, "consecutive_optimizer_step_cosine_valid_pairs", valid_pairs
        )
        object.__setattr__(self, "gradient_norm_mean_before_clip", norms[0])
        object.__setattr__(self, "gradient_norm_min_before_clip", norms[1])
        object.__setattr__(self, "gradient_norm_max_before_clip", norms[2])
        for field, value in zip(
            (
                "consecutive_optimizer_step_cosine_mean",
                "consecutive_optimizer_step_cosine_median",
                "consecutive_optimizer_step_cosine_min",
                "consecutive_optimizer_step_cosine_max",
            ),
            cosines,
            strict=True,
        ):
            object.__setattr__(self, field, value)
        object.__setattr__(
            self,
            "epoch_aggregate_gradient_cosine_to_previous_epoch",
            aggregate_cosine,
        )


@dataclass(frozen=True)
class BlockGradientDirectionEpochSummary:
    """One scalar-only gradient summary for one graph block and epoch."""

    global_epoch: int
    block_index: int
    block_name: str
    trainable_parameter_count: int
    optimizer_updates_observed: int
    gradient_norm_mean_before_clip: float
    gradient_norm_min_before_clip: float
    gradient_norm_max_before_clip: float
    consecutive_optimizer_step_cosine_mean: float | None
    consecutive_optimizer_step_cosine_median: float | None
    consecutive_optimizer_step_cosine_min: float | None
    consecutive_optimizer_step_cosine_max: float | None
    consecutive_optimizer_step_cosine_valid_pairs: int
    epoch_aggregate_gradient_cosine_to_previous_epoch: float | None
    resume_boundary_unavailable: bool
    schema: str = BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA

    def __post_init__(self) -> None:
        block_index = int(self.block_index)
        block_name = str(self.block_name).strip()
        if block_index < 0 or block_name != f"blocks.{block_index}":
            raise GradientDirectionObservabilityError(
                "Block summaries require a non-negative index and canonical "
                "blocks.<index> name."
            )
        if self.schema != BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA:
            raise GradientDirectionObservabilityError(
                "Block gradient-direction summary schema mismatch."
            )
        validated = GradientDirectionEpochSummary(
            global_epoch=self.global_epoch,
            trainable_parameter_count=self.trainable_parameter_count,
            optimizer_updates_observed=self.optimizer_updates_observed,
            gradient_norm_mean_before_clip=self.gradient_norm_mean_before_clip,
            gradient_norm_min_before_clip=self.gradient_norm_min_before_clip,
            gradient_norm_max_before_clip=self.gradient_norm_max_before_clip,
            consecutive_optimizer_step_cosine_mean=(
                self.consecutive_optimizer_step_cosine_mean
            ),
            consecutive_optimizer_step_cosine_median=(
                self.consecutive_optimizer_step_cosine_median
            ),
            consecutive_optimizer_step_cosine_min=(
                self.consecutive_optimizer_step_cosine_min
            ),
            consecutive_optimizer_step_cosine_max=(
                self.consecutive_optimizer_step_cosine_max
            ),
            consecutive_optimizer_step_cosine_valid_pairs=(
                self.consecutive_optimizer_step_cosine_valid_pairs
            ),
            epoch_aggregate_gradient_cosine_to_previous_epoch=(
                self.epoch_aggregate_gradient_cosine_to_previous_epoch
            ),
            resume_boundary_unavailable=self.resume_boundary_unavailable,
        )
        object.__setattr__(self, "block_index", block_index)
        object.__setattr__(self, "block_name", block_name)
        for field in (
            "global_epoch",
            "trainable_parameter_count",
            "optimizer_updates_observed",
            "gradient_norm_mean_before_clip",
            "gradient_norm_min_before_clip",
            "gradient_norm_max_before_clip",
            "consecutive_optimizer_step_cosine_mean",
            "consecutive_optimizer_step_cosine_median",
            "consecutive_optimizer_step_cosine_min",
            "consecutive_optimizer_step_cosine_max",
            "consecutive_optimizer_step_cosine_valid_pairs",
            "epoch_aggregate_gradient_cosine_to_previous_epoch",
            "resume_boundary_unavailable",
        ):
            object.__setattr__(self, field, getattr(validated, field))


def _parameter_signature(model: nn.Module) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        (name, tuple(int(size) for size in parameter.shape))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def full_trainable_gradient_vector(model: nn.Module) -> Tensor:
    """Copy the complete trainable gradient into one deterministic FP32 vector.

    Parameters are traversed in registered ``named_parameters`` order.  A
    trainable parameter with no gradient contributes an explicit zero slice, so
    vector coordinates remain stable.  The returned tensor is detached and
    never aliases a parameter's gradient storage.
    """

    pieces: list[Tensor] = []
    device: torch.device | None = None
    with torch.no_grad():
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if device is None:
                device = parameter.device
            elif parameter.device != device:
                raise GradientDirectionObservabilityError(
                    "All trainable parameters must share one device."
                )
            gradient = parameter.grad
            if gradient is None:
                piece = torch.zeros(
                    parameter.numel(), device=parameter.device, dtype=torch.float32
                )
            else:
                if gradient.is_sparse:
                    gradient = gradient.coalesce().to_dense()
                if gradient.device != parameter.device:
                    raise GradientDirectionObservabilityError(
                        "A parameter gradient is on the wrong device."
                    )
                piece = gradient.detach().to(dtype=torch.float32).reshape(-1)
            pieces.append(piece)
    if not pieces:
        raise GradientDirectionObservabilityError(
            "Gradient tracking requires at least one trainable parameter."
        )
    vector = torch.cat(pieces)
    if vector.dtype != torch.float32 or vector.requires_grad:
        raise RuntimeError("Full-gradient vector construction drifted.")
    if not bool(torch.isfinite(vector).all()):
        raise GradientDirectionObservabilityError(
            "Full-gradient vector contains a non-finite value."
        )
    return vector


def gradient_cosine_similarity(left: Tensor, right: Tensor) -> float | None:
    """Return a clamped cosine, treating either zero-norm vector as invalid."""

    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise GradientDirectionObservabilityError(
            "Gradient cosine inputs must be equal-length vectors."
        )
    if left.device != right.device:
        raise GradientDirectionObservabilityError(
            "Gradient cosine inputs must share one device."
        )
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise GradientDirectionObservabilityError(
            "Gradient cosine inputs must be finite."
        )
    with torch.no_grad():
        left_fp32 = left.detach().to(dtype=torch.float32)
        right_fp32 = right.detach().to(dtype=torch.float32)
        left_norm = torch.linalg.vector_norm(left_fp32)
        right_norm = torch.linalg.vector_norm(right_fp32)
        if not bool(torch.isfinite(left_norm)) or not bool(torch.isfinite(right_norm)):
            raise GradientDirectionObservabilityError(
                "Gradient cosine norm overflowed."
            )
        if float(left_norm) == 0.0 or float(right_norm) == 0.0:
            return None
        # Normalizing first avoids overflow in a raw dot/(norm product).
        cosine = torch.dot(left_fp32 / left_norm, right_fp32 / right_norm)
        if not bool(torch.isfinite(cosine)):
            raise GradientDirectionObservabilityError(
                "Gradient cosine calculation is non-finite."
            )
        return float(cosine.clamp(min=-1.0, max=1.0).cpu())


class FullGradientDirectionTracker:
    """Track step-to-step and epoch-to-epoch full-gradient alignment in memory."""

    def __init__(
        self,
        *,
        expected_optimizer_updates_per_epoch: int = 7,
        resume_boundary_unavailable: bool = False,
    ) -> None:
        expected = int(expected_optimizer_updates_per_epoch)
        if expected <= 0:
            raise GradientDirectionObservabilityError(
                "Expected optimizer updates per epoch must be positive."
            )
        if not isinstance(resume_boundary_unavailable, bool):
            raise GradientDirectionObservabilityError(
                "resume_boundary_unavailable must be boolean."
            )
        self.expected_optimizer_updates_per_epoch = expected
        self._resume_boundary_pending = resume_boundary_unavailable
        self._parameter_signature: tuple[tuple[str, tuple[int, ...]], ...] | None = None
        self._trainable_parameter_count: int | None = None
        self._current_epoch: int | None = None
        self._update_indices: list[int] = []
        self._gradient_norms: list[float] = []
        self._step_cosines: list[float] = []
        self._previous_step_gradient: Tensor | None = None
        self._current_epoch_gradient: Tensor | None = None
        self._previous_epoch_gradient: Tensor | None = None

    def reset_for_resume_boundary(self) -> None:
        """Forget transient vectors when continuing from a durable checkpoint."""

        if self._current_epoch is not None:
            raise GradientDirectionObservabilityError(
                "Cannot reset gradient tracking during an active epoch."
            )
        self._previous_step_gradient = None
        self._previous_epoch_gradient = None
        self._resume_boundary_pending = True

    def observe(self, model: nn.Module, context: GradientDirectionUpdateContext) -> None:
        """Read one unscaled, finite, pre-clip gradient without mutating it."""

        if not isinstance(context, GradientDirectionUpdateContext):
            raise GradientDirectionObservabilityError(
                "Gradient observation context has the wrong type."
            )
        if self._current_epoch is None:
            self._current_epoch = context.global_epoch
        elif context.global_epoch != self._current_epoch:
            raise GradientDirectionObservabilityError(
                "Complete the current gradient epoch before observing another."
            )
        expected_update = len(self._update_indices)
        if context.optimizer_update_in_epoch != expected_update:
            raise GradientDirectionObservabilityError(
                f"Expected optimizer update {expected_update}; received "
                f"{context.optimizer_update_in_epoch}."
            )
        if expected_update >= self.expected_optimizer_updates_per_epoch:
            raise GradientDirectionObservabilityError(
                "Observed too many optimizer updates in one epoch."
            )

        signature = _parameter_signature(model)
        if not signature:
            raise GradientDirectionObservabilityError(
                "Gradient tracking requires trainable parameters."
            )
        if self._parameter_signature is None:
            self._parameter_signature = signature
            self._trainable_parameter_count = sum(
                math.prod(shape) for _, shape in signature
            )
        elif signature != self._parameter_signature:
            raise GradientDirectionObservabilityError(
                "Trainable parameter order or shape changed during tracking."
            )

        current = full_trainable_gradient_vector(model)
        if current.numel() != self._trainable_parameter_count:
            raise RuntimeError("Full-gradient parameter count drifted.")
        norm_tensor = torch.linalg.vector_norm(current)
        if not bool(torch.isfinite(norm_tensor)):
            raise GradientDirectionObservabilityError(
                "Full-gradient norm overflowed."
            )
        norm = float(norm_tensor.cpu())
        if self._previous_step_gradient is not None:
            cosine = gradient_cosine_similarity(
                self._previous_step_gradient, current
            )
            if cosine is not None:
                self._step_cosines.append(cosine)
        if self._current_epoch_gradient is None:
            self._current_epoch_gradient = current.clone()
        else:
            self._current_epoch_gradient.add_(current)
            if not bool(torch.isfinite(self._current_epoch_gradient).all()):
                raise GradientDirectionObservabilityError(
                    "Epoch aggregate gradient overflowed."
                )
        self._previous_step_gradient = current
        self._gradient_norms.append(norm)
        self._update_indices.append(context.optimizer_update_in_epoch)

    def complete_epoch(self, global_epoch: int) -> GradientDirectionEpochSummary:
        """Finalize the zero-based epoch and return its one-based scalar row."""

        requested = int(global_epoch)
        if self._current_epoch is None or requested != self._current_epoch:
            raise GradientDirectionObservabilityError(
                "No matching active gradient epoch can be completed."
            )
        if self._update_indices != list(
            range(self.expected_optimizer_updates_per_epoch)
        ):
            raise GradientDirectionObservabilityError(
                "Gradient epoch lacks its exact ordered optimizer updates."
            )
        if (
            self._current_epoch_gradient is None
            or self._trainable_parameter_count is None
            or len(self._gradient_norms) != self.expected_optimizer_updates_per_epoch
        ):
            raise RuntimeError("Gradient tracker state is incomplete.")

        norms = self._gradient_norms
        if self._step_cosines:
            cosine_mean: float | None = statistics.fmean(self._step_cosines)
            cosine_median: float | None = statistics.median(self._step_cosines)
            cosine_min: float | None = min(self._step_cosines)
            cosine_max: float | None = max(self._step_cosines)
        else:
            cosine_mean = cosine_median = cosine_min = cosine_max = None
        epoch_cosine = (
            None
            if self._previous_epoch_gradient is None
            else gradient_cosine_similarity(
                self._previous_epoch_gradient, self._current_epoch_gradient
            )
        )
        summary = GradientDirectionEpochSummary(
            global_epoch=requested + 1,
            trainable_parameter_count=self._trainable_parameter_count,
            optimizer_updates_observed=len(norms),
            gradient_norm_mean_before_clip=statistics.fmean(norms),
            gradient_norm_min_before_clip=min(norms),
            gradient_norm_max_before_clip=max(norms),
            consecutive_optimizer_step_cosine_mean=cosine_mean,
            consecutive_optimizer_step_cosine_median=cosine_median,
            consecutive_optimizer_step_cosine_min=cosine_min,
            consecutive_optimizer_step_cosine_max=cosine_max,
            consecutive_optimizer_step_cosine_valid_pairs=len(self._step_cosines),
            epoch_aggregate_gradient_cosine_to_previous_epoch=epoch_cosine,
            resume_boundary_unavailable=self._resume_boundary_pending,
        )

        self._previous_epoch_gradient = self._current_epoch_gradient
        self._current_epoch_gradient = None
        self._current_epoch = None
        self._update_indices = []
        self._gradient_norms = []
        self._step_cosines = []
        self._resume_boundary_pending = False
        return summary


class BlockGradientDirectionTracker:
    """Track full-gradient direction independently for eight untied blocks.

    The public model must expose its graph blocks as ``model.blocks`` in an
    ordered :class:`~torch.nn.ModuleList`.  Blocks and their trainable
    parameters must be distinct so the resulting long-form rows have an
    unambiguous architectural interpretation.
    """

    def __init__(
        self,
        *,
        expected_blocks: int = 8,
        expected_optimizer_updates_per_epoch: int = 7,
        resume_boundary_unavailable: bool = False,
    ) -> None:
        blocks = int(expected_blocks)
        if blocks <= 0:
            raise GradientDirectionObservabilityError(
                "Expected graph-block count must be positive."
            )
        self.expected_blocks = blocks
        self.expected_optimizer_updates_per_epoch = int(
            expected_optimizer_updates_per_epoch
        )
        self._trackers = tuple(
            FullGradientDirectionTracker(
                expected_optimizer_updates_per_epoch=(
                    expected_optimizer_updates_per_epoch
                ),
                resume_boundary_unavailable=resume_boundary_unavailable,
            )
            for _ in range(blocks)
        )

    def _ordered_blocks(self, model: nn.Module) -> tuple[nn.Module, ...]:
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList):
            raise GradientDirectionObservabilityError(
                "Per-block gradient tracking requires model.blocks ModuleList."
            )
        ordered = tuple(blocks)
        if len(ordered) != self.expected_blocks:
            raise GradientDirectionObservabilityError(
                f"Expected exactly {self.expected_blocks} graph blocks; received "
                f"{len(ordered)}."
            )
        if len({id(block) for block in ordered}) != len(ordered):
            raise GradientDirectionObservabilityError(
                "Per-block diagnostics require distinct untied block modules."
            )
        parameter_ids = [
            id(parameter)
            for block in ordered
            for parameter in block.parameters()
            if parameter.requires_grad
        ]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise GradientDirectionObservabilityError(
                "Per-block diagnostics require disjoint trainable parameters."
            )
        if any(not _parameter_signature(block) for block in ordered):
            raise GradientDirectionObservabilityError(
                "Every tracked graph block must have trainable parameters."
            )
        return ordered

    def reset_for_resume_boundary(self) -> None:
        """Forget every block's transient vectors at a checkpoint boundary."""

        for tracker in self._trackers:
            tracker.reset_for_resume_boundary()

    def observe(self, model: nn.Module, context: GradientDirectionUpdateContext) -> None:
        """Read one unscaled, finite, pre-clip gradient for every block."""

        ordered = self._ordered_blocks(model)
        for block, tracker in zip(ordered, self._trackers, strict=True):
            tracker.observe(block, context)

    def complete_epoch(
        self, global_epoch: int
    ) -> tuple[BlockGradientDirectionEpochSummary, ...]:
        """Return eight summaries ordered by ``block_index`` for one epoch."""

        summaries: list[BlockGradientDirectionEpochSummary] = []
        for block_index, tracker in enumerate(self._trackers):
            full = tracker.complete_epoch(global_epoch)
            summaries.append(
                BlockGradientDirectionEpochSummary(
                    global_epoch=full.global_epoch,
                    block_index=block_index,
                    block_name=f"blocks.{block_index}",
                    trainable_parameter_count=full.trainable_parameter_count,
                    optimizer_updates_observed=full.optimizer_updates_observed,
                    gradient_norm_mean_before_clip=(
                        full.gradient_norm_mean_before_clip
                    ),
                    gradient_norm_min_before_clip=(
                        full.gradient_norm_min_before_clip
                    ),
                    gradient_norm_max_before_clip=(
                        full.gradient_norm_max_before_clip
                    ),
                    consecutive_optimizer_step_cosine_mean=(
                        full.consecutive_optimizer_step_cosine_mean
                    ),
                    consecutive_optimizer_step_cosine_median=(
                        full.consecutive_optimizer_step_cosine_median
                    ),
                    consecutive_optimizer_step_cosine_min=(
                        full.consecutive_optimizer_step_cosine_min
                    ),
                    consecutive_optimizer_step_cosine_max=(
                        full.consecutive_optimizer_step_cosine_max
                    ),
                    consecutive_optimizer_step_cosine_valid_pairs=(
                        full.consecutive_optimizer_step_cosine_valid_pairs
                    ),
                    epoch_aggregate_gradient_cosine_to_previous_epoch=(
                        full.epoch_aggregate_gradient_cosine_to_previous_epoch
                    ),
                    resume_boundary_unavailable=(
                        full.resume_boundary_unavailable
                    ),
                )
            )
        return tuple(summaries)


def _normalized_csv_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class DurableGradientDirectionCSV:
    """Atomically publish one scalar gradient-direction row per global epoch."""

    def __init__(
        self,
        run_root: str | Path,
        *,
        run_id: str,
        model_seed: int,
        rank: int = 0,
    ) -> None:
        if int(rank) != 0:
            raise GradientDirectionObservabilityError(
                "Only distributed rank zero may write gradient metrics."
            )
        self.run_root = Path(run_root).resolve(strict=True)
        self.run_id = str(run_id).strip()
        self.model_seed = int(model_seed)
        if not self.run_id:
            raise GradientDirectionObservabilityError("run_id must be non-empty.")
        if self.model_seed < 0:
            raise GradientDirectionObservabilityError(
                "model_seed must be non-negative."
            )
        self.path = self.run_root / "results" / "gradient_direction_metrics.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise GradientDirectionObservabilityError(
                "Gradient metrics path must be a regular file."
            )
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != GRADIENT_DIRECTION_METRICS_COLUMNS:
                raise GradientDirectionObservabilityError(
                    "Gradient metrics CSV schema mismatch."
                )
            rows = list(reader)
        if any(
            row.get("run_id") != self.run_id
            or row.get("model_seed") != str(self.model_seed)
            for row in rows
        ):
            raise GradientDirectionObservabilityError(
                "Gradient metrics CSV run identity mismatch."
            )
        try:
            observed = [int(row["global_epoch"]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise GradientDirectionObservabilityError(
                "Gradient metrics CSV contains an invalid epoch."
            ) from exc
        if observed != list(range(1, len(rows) + 1)):
            raise GradientDirectionObservabilityError(
                "Gradient metrics CSV requires unique contiguous epochs from one."
            )
        return rows

    @property
    def completed_epochs(self) -> int:
        return len(self._read_rows())

    def read_rows(self) -> tuple[dict[str, str], ...]:
        return tuple(self._read_rows())

    def reconcile(self, *, checkpoint_epoch: int) -> int:
        """Trim any scalar row newer than the durable rolling checkpoint."""

        durable_epoch = int(checkpoint_epoch)
        if durable_epoch < 0:
            raise GradientDirectionObservabilityError(
                "Checkpoint epoch cannot be negative."
            )
        rows = self._read_rows()
        if len(rows) < durable_epoch:
            raise GradientDirectionObservabilityError(
                "Gradient metrics end before the resume checkpoint."
            )
        if len(rows) == durable_epoch:
            return durable_epoch
        self._atomic_rewrite(rows[:durable_epoch])
        return durable_epoch

    def _atomic_rewrite(self, rows: Sequence[Mapping[str, object]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".gradient_direction_metrics.csv.",
            suffix=".writing",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=GRADIENT_DIRECTION_METRICS_COLUMNS
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            column: _normalized_csv_value(row[column])
                            for column in GRADIENT_DIRECTION_METRICS_COLUMNS
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def append(
        self, summary: GradientDirectionEpochSummary | Mapping[str, object]
    ) -> bool:
        """Atomically add one row; return false for an exact idempotent replay."""

        raw: Mapping[str, object]
        if isinstance(summary, GradientDirectionEpochSummary):
            raw = asdict(summary)
        else:
            raw = summary
        summary_columns = set(GRADIENT_DIRECTION_METRICS_COLUMNS) - {
            "run_id",
            "model_seed",
        }
        if set(raw) == set(GRADIENT_DIRECTION_METRICS_COLUMNS):
            if (
                str(raw["run_id"]) != self.run_id
                or int(raw["model_seed"]) != self.model_seed
            ):
                raise GradientDirectionObservabilityError(
                    "Gradient metrics row run identity mismatch."
                )
            row = raw
        elif set(raw) == summary_columns:
            row = {
                "schema": raw["schema"],
                "run_id": self.run_id,
                "model_seed": self.model_seed,
                **{key: value for key, value in raw.items() if key != "schema"},
            }
        else:
            raise GradientDirectionObservabilityError(
                "Gradient metrics row has the wrong columns."
            )
        rows = self._read_rows()
        try:
            epoch = int(row["global_epoch"])
        except (TypeError, ValueError) as exc:
            raise GradientDirectionObservabilityError(
                "Gradient metrics epoch is invalid."
            ) from exc
        normalized = {
            column: _normalized_csv_value(row[column])
            for column in GRADIENT_DIRECTION_METRICS_COLUMNS
        }
        if epoch == len(rows) and rows:
            if rows[-1] == normalized:
                return False
            raise GradientDirectionObservabilityError(
                "Refusing a divergent duplicate gradient metrics row."
            )
        expected = len(rows) + 1
        if epoch != expected:
            raise GradientDirectionObservabilityError(
                f"Expected gradient metrics epoch {expected}; received {epoch}."
            )
        self._atomic_rewrite((*rows, normalized))
        return True


def _csv_boolean(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise GradientDirectionObservabilityError(
        f"{field} must be a CSV boolean encoded as 0 or 1."
    )


def _optional_csv_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _block_summary_from_mapping(
    value: Mapping[str, object],
) -> BlockGradientDirectionEpochSummary:
    try:
        return BlockGradientDirectionEpochSummary(
            global_epoch=int(value["global_epoch"]),
            block_index=int(value["block_index"]),
            block_name=str(value["block_name"]),
            trainable_parameter_count=int(value["trainable_parameter_count"]),
            optimizer_updates_observed=int(value["optimizer_updates_observed"]),
            gradient_norm_mean_before_clip=float(
                value["gradient_norm_mean_before_clip"]
            ),
            gradient_norm_min_before_clip=float(
                value["gradient_norm_min_before_clip"]
            ),
            gradient_norm_max_before_clip=float(
                value["gradient_norm_max_before_clip"]
            ),
            consecutive_optimizer_step_cosine_mean=_optional_csv_float(
                value["consecutive_optimizer_step_cosine_mean"]
            ),
            consecutive_optimizer_step_cosine_median=_optional_csv_float(
                value["consecutive_optimizer_step_cosine_median"]
            ),
            consecutive_optimizer_step_cosine_min=_optional_csv_float(
                value["consecutive_optimizer_step_cosine_min"]
            ),
            consecutive_optimizer_step_cosine_max=_optional_csv_float(
                value["consecutive_optimizer_step_cosine_max"]
            ),
            consecutive_optimizer_step_cosine_valid_pairs=int(
                value["consecutive_optimizer_step_cosine_valid_pairs"]
            ),
            epoch_aggregate_gradient_cosine_to_previous_epoch=(
                _optional_csv_float(
                    value[
                        "epoch_aggregate_gradient_cosine_to_previous_epoch"
                    ]
                )
            ),
            resume_boundary_unavailable=_csv_boolean(
                value["resume_boundary_unavailable"],
                field="resume_boundary_unavailable",
            ),
            schema=str(value["schema"]),
        )
    except GradientDirectionObservabilityError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise GradientDirectionObservabilityError(
            "Block gradient metrics row is malformed."
        ) from exc


class DurableBlockGradientDirectionCSV:
    """Atomically publish eight long-form block summaries per global epoch."""

    def __init__(
        self,
        run_root: str | Path,
        *,
        run_id: str,
        model_seed: int,
        expected_blocks: int = 8,
        rank: int = 0,
    ) -> None:
        if int(rank) != 0:
            raise GradientDirectionObservabilityError(
                "Only distributed rank zero may write block gradient metrics."
            )
        self.run_root = Path(run_root).resolve(strict=True)
        self.run_id = str(run_id).strip()
        self.model_seed = int(model_seed)
        self.expected_blocks = int(expected_blocks)
        if not self.run_id:
            raise GradientDirectionObservabilityError("run_id must be non-empty.")
        if self.model_seed < 0:
            raise GradientDirectionObservabilityError(
                "model_seed must be non-negative."
            )
        if self.expected_blocks <= 0:
            raise GradientDirectionObservabilityError(
                "Expected graph-block count must be positive."
            )
        self.path = self.run_root / "results" / "gradient_direction_by_block.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _validate_rows(self, rows: Sequence[Mapping[str, str]]) -> None:
        if len(rows) % self.expected_blocks != 0:
            raise GradientDirectionObservabilityError(
                "Block gradient metrics require a complete block group per epoch."
            )
        for offset in range(0, len(rows), self.expected_blocks):
            epoch = offset // self.expected_blocks + 1
            group = rows[offset : offset + self.expected_blocks]
            for block_index, row in enumerate(group):
                if (
                    row.get("run_id") != self.run_id
                    or row.get("model_seed") != str(self.model_seed)
                ):
                    raise GradientDirectionObservabilityError(
                        "Block gradient metrics CSV run identity mismatch."
                    )
                summary = _block_summary_from_mapping(row)
                if (
                    summary.global_epoch != epoch
                    or summary.block_index != block_index
                    or summary.block_name != f"blocks.{block_index}"
                ):
                    raise GradientDirectionObservabilityError(
                        "Block gradient metrics must contain contiguous epochs and "
                        "ordered blocks."
                    )

    def _read_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise GradientDirectionObservabilityError(
                "Block gradient metrics path must be a regular file."
            )
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != (
                BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS
            ):
                raise GradientDirectionObservabilityError(
                    "Block gradient metrics CSV schema mismatch."
                )
            rows = list(reader)
        self._validate_rows(rows)
        return rows

    @property
    def completed_epochs(self) -> int:
        return len(self._read_rows()) // self.expected_blocks

    def read_rows(self) -> tuple[dict[str, str], ...]:
        """Return a validated long-form scalar snapshot."""

        return tuple(self._read_rows())

    def reconcile(self, *, checkpoint_epoch: int) -> int:
        """Trim complete epoch groups newer than the durable checkpoint."""

        durable_epoch = int(checkpoint_epoch)
        if durable_epoch < 0:
            raise GradientDirectionObservabilityError(
                "Checkpoint epoch cannot be negative."
            )
        rows = self._read_rows()
        completed = len(rows) // self.expected_blocks
        if completed < durable_epoch:
            raise GradientDirectionObservabilityError(
                "Block gradient metrics end before the resume checkpoint."
            )
        if completed == durable_epoch:
            return durable_epoch
        self._atomic_rewrite(rows[: durable_epoch * self.expected_blocks])
        return durable_epoch

    def _atomic_rewrite(self, rows: Sequence[Mapping[str, object]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".gradient_direction_by_block.csv.",
            suffix=".writing",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            column: _normalized_csv_value(row[column])
                            for column in BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _normalize_summary(
        self, summary: BlockGradientDirectionEpochSummary | Mapping[str, object]
    ) -> dict[str, str]:
        raw: Mapping[str, object]
        if isinstance(summary, BlockGradientDirectionEpochSummary):
            raw = asdict(summary)
        else:
            raw = summary
        summary_columns = set(BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS) - {
            "run_id",
            "model_seed",
        }
        if set(raw) == set(BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS):
            if (
                str(raw["run_id"]) != self.run_id
                or int(raw["model_seed"]) != self.model_seed
            ):
                raise GradientDirectionObservabilityError(
                    "Block gradient metrics row run identity mismatch."
                )
            payload = {
                key: value
                for key, value in raw.items()
                if key not in {"run_id", "model_seed"}
            }
        elif set(raw) == summary_columns:
            payload = dict(raw)
        else:
            raise GradientDirectionObservabilityError(
                "Block gradient metrics row has the wrong columns."
            )
        validated = _block_summary_from_mapping(payload)
        values = asdict(validated)
        row: dict[str, object] = {
            "schema": values.pop("schema"),
            "run_id": self.run_id,
            "model_seed": self.model_seed,
            **values,
        }
        return {
            column: _normalized_csv_value(row[column])
            for column in BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS
        }

    def append(
        self,
        summaries: Sequence[
            BlockGradientDirectionEpochSummary | Mapping[str, object]
        ],
    ) -> bool:
        """Atomically add one ordered eight-row epoch; replay is idempotent."""

        if isinstance(summaries, (str, bytes)):
            raise GradientDirectionObservabilityError(
                "Block gradient metrics append requires a summary sequence."
            )
        materialized = tuple(summaries)
        if len(materialized) != self.expected_blocks:
            raise GradientDirectionObservabilityError(
                f"Expected {self.expected_blocks} block summaries per epoch."
            )
        normalized = tuple(self._normalize_summary(item) for item in materialized)
        epochs = {int(row["global_epoch"]) for row in normalized}
        if len(epochs) != 1:
            raise GradientDirectionObservabilityError(
                "All block summaries in an append must share one epoch."
            )
        epoch = next(iter(epochs))
        if tuple(int(row["block_index"]) for row in normalized) != tuple(
            range(self.expected_blocks)
        ) or tuple(row["block_name"] for row in normalized) != tuple(
            f"blocks.{index}" for index in range(self.expected_blocks)
        ):
            raise GradientDirectionObservabilityError(
                "Block summaries must be ordered exactly by block index."
            )
        rows = self._read_rows()
        completed = len(rows) // self.expected_blocks
        if epoch == completed and rows:
            if tuple(rows[-self.expected_blocks :]) == normalized:
                return False
            raise GradientDirectionObservabilityError(
                "Refusing a divergent duplicate block gradient epoch."
            )
        expected_epoch = completed + 1
        if epoch != expected_epoch:
            raise GradientDirectionObservabilityError(
                f"Expected block gradient epoch {expected_epoch}; received {epoch}."
            )
        self._atomic_rewrite((*rows, *normalized))
        return True


__all__ = [
    "BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS",
    "BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA",
    "GRADIENT_DIRECTION_METRICS_COLUMNS",
    "GRADIENT_DIRECTION_METRICS_SCHEMA",
    "BlockGradientDirectionEpochSummary",
    "BlockGradientDirectionTracker",
    "DurableBlockGradientDirectionCSV",
    "DurableGradientDirectionCSV",
    "FullGradientDirectionTracker",
    "GradientDirectionEpochSummary",
    "GradientDirectionObservabilityError",
    "GradientDirectionUpdateContext",
    "full_trainable_gradient_vector",
    "gradient_cosine_similarity",
]
