from __future__ import annotations

import csv
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import nn

from spatial_benchmark.gradient_direction_observability import (
    BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
    BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
    GRADIENT_DIRECTION_METRICS_COLUMNS,
    BlockGradientDirectionEpochSummary,
    BlockGradientDirectionTracker,
    DurableBlockGradientDirectionCSV,
    DurableGradientDirectionCSV,
    FullGradientDirectionTracker,
    GradientDirectionEpochSummary,
    GradientDirectionUpdateContext,
    full_trainable_gradient_vector,
    gradient_cosine_similarity,
)
from spatial_benchmark.pooled_relative_qkv_training_v2 import (
    SO2_14CORE_ALIASES,
    CohortRelativeQKVCoreBatch,
    CohortRelativeQKVTrainingConfig,
    fit_cohort_relative_qkv_segment,
)


class _VectorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vector = nn.Parameter(torch.zeros(2))


def _context(epoch: int, update: int) -> GradientDirectionUpdateContext:
    return GradientDirectionUpdateContext(
        global_epoch=epoch,
        completed_global_epoch=epoch + 1,
        optimizer_update_in_epoch=update,
        cumulative_optimizer_update=epoch * 7 + update + 1,
        aliases=("SO2-C15", "SO2-C16"),
    )


def _observe_epoch(
    tracker: FullGradientDirectionTracker,
    model: _VectorModel,
    *,
    epoch: int,
    gradients: tuple[tuple[float, float], ...],
) -> GradientDirectionEpochSummary:
    assert len(gradients) == 7
    for update, gradient in enumerate(gradients):
        model.vector.grad = torch.tensor(gradient)
        tracker.observe(model, _context(epoch, update))
    return tracker.complete_epoch(epoch)


def test_full_vector_order_and_known_cosine_directions() -> None:
    class Parameters(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.left = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.frozen = nn.Parameter(torch.tensor([9.0]), requires_grad=False)
            self.right = nn.Parameter(torch.tensor([3.0]))

    model = Parameters()
    model.left.grad = torch.tensor([4.0, 5.0])
    model.right.grad = None
    vector = full_trainable_gradient_vector(model)
    assert vector.dtype == torch.float32
    assert vector.tolist() == [4.0, 5.0, 0.0]
    vector[0] = 99.0
    assert model.left.grad.tolist() == [4.0, 5.0]

    horizontal = torch.tensor([1.0, 0.0])
    vertical = torch.tensor([0.0, 2.0])
    assert gradient_cosine_similarity(horizontal, horizontal) == pytest.approx(1.0)
    assert gradient_cosine_similarity(horizontal, vertical) == pytest.approx(0.0)
    assert gradient_cosine_similarity(horizontal, -horizontal) == pytest.approx(-1.0)
    assert gradient_cosine_similarity(horizontal, torch.zeros(2)) is None


def test_tracker_epoch_boundaries_and_resume_reset_semantics() -> None:
    model = _VectorModel()
    tracker = FullGradientDirectionTracker()

    first = _observe_epoch(
        tracker,
        model,
        epoch=0,
        gradients=((1.0, 0.0),) * 7,
    )
    assert first.global_epoch == 1
    assert first.trainable_parameter_count == 2
    assert first.optimizer_updates_observed == 7
    assert first.gradient_norm_mean_before_clip == pytest.approx(1.0)
    assert first.consecutive_optimizer_step_cosine_valid_pairs == 6
    assert first.consecutive_optimizer_step_cosine_mean == pytest.approx(1.0)
    assert first.epoch_aggregate_gradient_cosine_to_previous_epoch is None
    assert first.resume_boundary_unavailable is False

    second = _observe_epoch(
        tracker,
        model,
        epoch=1,
        gradients=((0.0, 2.0),) * 7,
    )
    assert second.consecutive_optimizer_step_cosine_valid_pairs == 7
    assert second.consecutive_optimizer_step_cosine_mean == pytest.approx(6.0 / 7.0)
    assert second.consecutive_optimizer_step_cosine_min == pytest.approx(0.0)
    assert second.consecutive_optimizer_step_cosine_max == pytest.approx(1.0)
    assert second.epoch_aggregate_gradient_cosine_to_previous_epoch == pytest.approx(
        0.0
    )
    assert second.resume_boundary_unavailable is False

    resumed = FullGradientDirectionTracker(resume_boundary_unavailable=True)
    resumed_first = _observe_epoch(
        resumed,
        model,
        epoch=2,
        gradients=((-1.0, 0.0),) * 7,
    )
    assert resumed_first.consecutive_optimizer_step_cosine_valid_pairs == 6
    assert resumed_first.epoch_aggregate_gradient_cosine_to_previous_epoch is None
    assert resumed_first.resume_boundary_unavailable is True
    resumed_second = _observe_epoch(
        resumed,
        model,
        epoch=3,
        gradients=((1.0, 0.0),) * 7,
    )
    assert resumed_second.consecutive_optimizer_step_cosine_valid_pairs == 7
    assert resumed_second.consecutive_optimizer_step_cosine_min == pytest.approx(-1.0)
    assert (
        resumed_second.epoch_aggregate_gradient_cosine_to_previous_epoch
        == pytest.approx(-1.0)
    )
    assert resumed_second.resume_boundary_unavailable is False


class _EightBlockModel(nn.Module):
    def __init__(self, blocks: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_VectorModel() for _ in range(blocks))


def _observe_block_epoch(
    tracker: BlockGradientDirectionTracker,
    model: _EightBlockModel,
    *,
    epoch: int,
    gradients: tuple[tuple[float, float], ...],
) -> tuple[BlockGradientDirectionEpochSummary, ...]:
    assert len(gradients) == 8
    for update in range(7):
        for block, gradient in zip(model.blocks, gradients, strict=True):
            block.vector.grad = torch.tensor(gradient)  # type: ignore[attr-defined]
        tracker.observe(model, _context(epoch, update))
    return tracker.complete_epoch(epoch)


def test_block_tracker_emits_ordered_independent_eight_block_summaries() -> None:
    model = _EightBlockModel()
    tracker = BlockGradientDirectionTracker()
    horizontal = tuple((float(index + 1), 0.0) for index in range(8))

    first = _observe_block_epoch(
        tracker,
        model,
        epoch=0,
        gradients=horizontal,
    )

    assert len(first) == 8
    assert tuple(summary.block_index for summary in first) == tuple(range(8))
    assert tuple(summary.block_name for summary in first) == tuple(
        f"blocks.{index}" for index in range(8)
    )
    assert all(
        summary.schema == BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA
        for summary in first
    )
    assert all(summary.trainable_parameter_count == 2 for summary in first)
    assert all(summary.optimizer_updates_observed == 7 for summary in first)
    assert all(
        summary.consecutive_optimizer_step_cosine_valid_pairs == 6
        for summary in first
    )
    assert all(
        summary.epoch_aggregate_gradient_cosine_to_previous_epoch is None
        for summary in first
    )
    assert [summary.gradient_norm_mean_before_clip for summary in first] == (
        pytest.approx([float(index + 1) for index in range(8)])
    )

    second_gradients = tuple(
        (0.0, float(index + 1))
        if index % 2 == 0
        else (-float(index + 1), 0.0)
        for index in range(8)
    )
    second = _observe_block_epoch(
        tracker,
        model,
        epoch=1,
        gradients=second_gradients,
    )
    assert all(
        summary.consecutive_optimizer_step_cosine_valid_pairs == 7
        for summary in second
    )
    for summary in second:
        if summary.block_index % 2 == 0:
            assert summary.consecutive_optimizer_step_cosine_mean == pytest.approx(
                6.0 / 7.0
            )
            assert (
                summary.epoch_aggregate_gradient_cosine_to_previous_epoch
                == pytest.approx(0.0)
            )
        else:
            assert summary.consecutive_optimizer_step_cosine_mean == pytest.approx(
                5.0 / 7.0
            )
            assert (
                summary.epoch_aggregate_gradient_cosine_to_previous_epoch
                == pytest.approx(-1.0)
            )
    assert all(
        not isinstance(value, torch.Tensor)
        for summary in (*first, *second)
        for value in asdict(summary).values()
    )


def test_block_tracker_validates_untied_shape_and_resume_boundary() -> None:
    tracker = BlockGradientDirectionTracker(resume_boundary_unavailable=True)
    model = _EightBlockModel()
    resumed = _observe_block_epoch(
        tracker,
        model,
        epoch=4,
        gradients=((1.0, 0.0),) * 8,
    )
    assert all(summary.resume_boundary_unavailable for summary in resumed)
    assert all(
        summary.consecutive_optimizer_step_cosine_valid_pairs == 6
        for summary in resumed
    )
    assert all(
        summary.epoch_aggregate_gradient_cosine_to_previous_epoch is None
        for summary in resumed
    )

    with pytest.raises(RuntimeError, match="exactly 8 graph blocks"):
        BlockGradientDirectionTracker().observe(_EightBlockModel(7), _context(0, 0))

    shared = _VectorModel()
    aliased = _EightBlockModel(0)
    aliased.blocks = nn.ModuleList([shared] * 8)
    with pytest.raises(RuntimeError, match="distinct untied"):
        BlockGradientDirectionTracker().observe(aliased, _context(0, 0))


def _summary(epoch: int) -> GradientDirectionEpochSummary:
    return GradientDirectionEpochSummary(
        global_epoch=epoch,
        trainable_parameter_count=2_605_680,
        optimizer_updates_observed=7,
        gradient_norm_mean_before_clip=2.0,
        gradient_norm_min_before_clip=1.0,
        gradient_norm_max_before_clip=3.0,
        consecutive_optimizer_step_cosine_mean=0.5,
        consecutive_optimizer_step_cosine_median=0.6,
        consecutive_optimizer_step_cosine_min=-0.2,
        consecutive_optimizer_step_cosine_max=0.9,
        consecutive_optimizer_step_cosine_valid_pairs=6 if epoch == 1 else 7,
        epoch_aggregate_gradient_cosine_to_previous_epoch=(
            None if epoch == 1 else 0.7
        ),
        resume_boundary_unavailable=False,
    )


def _block_summaries(
    epoch: int,
) -> tuple[BlockGradientDirectionEpochSummary, ...]:
    return tuple(
        BlockGradientDirectionEpochSummary(
            global_epoch=epoch,
            block_index=block_index,
            block_name=f"blocks.{block_index}",
            trainable_parameter_count=799_112,
            optimizer_updates_observed=7,
            gradient_norm_mean_before_clip=2.0 + block_index,
            gradient_norm_min_before_clip=1.0 + block_index,
            gradient_norm_max_before_clip=3.0 + block_index,
            consecutive_optimizer_step_cosine_mean=0.5,
            consecutive_optimizer_step_cosine_median=0.6,
            consecutive_optimizer_step_cosine_min=-0.2,
            consecutive_optimizer_step_cosine_max=0.9,
            consecutive_optimizer_step_cosine_valid_pairs=(
                6 if epoch == 1 else 7
            ),
            epoch_aggregate_gradient_cosine_to_previous_epoch=(
                None if epoch == 1 else 0.7
            ),
            resume_boundary_unavailable=False,
        )
        for block_index in range(8)
    )


def test_scalar_csv_is_atomic_ordered_idempotent_and_resume_reconciled(
    tmp_path: Path,
) -> None:
    writer = DurableGradientDirectionCSV(tmp_path, run_id="r_test", model_seed=0)
    first = _summary(1)
    assert writer.append(first) is True
    assert writer.append(first) is False
    assert writer.append(_summary(2)) is True
    assert writer.append(_summary(3)) is True
    assert writer.completed_epochs == 3

    with writer.path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == GRADIENT_DIRECTION_METRICS_COLUMNS
    assert rows[0]["epoch_aggregate_gradient_cosine_to_previous_epoch"] == ""
    assert rows[0]["run_id"] == "r_test"
    assert rows[0]["model_seed"] == "0"
    assert rows[2]["global_epoch"] == "3"
    assert writer.reconcile(checkpoint_epoch=2) == 2
    assert writer.completed_epochs == 2
    assert writer.append(_summary(3)) is True
    assert not list(
        (tmp_path / "results").glob(".gradient_direction_metrics.csv.*.writing")
    )

    divergent = replace(_summary(3), gradient_norm_mean_before_clip=2.5)
    with pytest.raises(RuntimeError, match="divergent duplicate"):
        writer.append(divergent)


def test_block_csv_is_long_form_atomic_idempotent_and_resume_reconciled(
    tmp_path: Path,
) -> None:
    writer = DurableBlockGradientDirectionCSV(
        tmp_path,
        run_id="r_blocks",
        model_seed=0,
    )
    first = _block_summaries(1)
    second = _block_summaries(2)
    third = _block_summaries(3)
    assert writer.append(first) is True
    assert writer.append(first) is False
    assert writer.append(second) is True
    assert writer.append(third) is True
    assert writer.completed_epochs == 3

    with writer.path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 24
    assert tuple(rows[0]) == BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS
    assert [int(row["block_index"]) for row in rows[:8]] == list(range(8))
    assert {row["global_epoch"] for row in rows[:8]} == {"1"}
    assert rows[0]["block_name"] == "blocks.0"
    assert rows[7]["block_name"] == "blocks.7"
    assert rows[0]["epoch_aggregate_gradient_cosine_to_previous_epoch"] == ""
    assert all(row["run_id"] == "r_blocks" for row in rows)
    assert all(row["model_seed"] == "0" for row in rows)

    assert writer.reconcile(checkpoint_epoch=2) == 2
    assert writer.completed_epochs == 2
    assert len(writer.read_rows()) == 16
    assert writer.append(third) is True
    divergent = list(third)
    divergent[4] = replace(
        divergent[4],
        gradient_norm_mean_before_clip=6.5,
    )
    with pytest.raises(RuntimeError, match="divergent duplicate"):
        writer.append(tuple(divergent))
    with pytest.raises(RuntimeError, match="8 block summaries"):
        writer.append(third[:-1])
    assert not list(
        (tmp_path / "results").glob(".gradient_direction_by_block.csv.*.writing")
    )


def test_failed_block_csv_replace_preserves_previous_complete_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = DurableBlockGradientDirectionCSV(
        tmp_path,
        run_id="r_blocks",
        model_seed=0,
    )
    writer.append(_block_summaries(1))
    original = writer.path.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("injected block CSV replacement failure")

    monkeypatch.setattr(
        "spatial_benchmark.gradient_direction_observability.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError, match="injected block CSV"):
        writer.append(_block_summaries(2))

    assert writer.path.read_bytes() == original
    assert writer.completed_epochs == 1
    assert not list(
        (tmp_path / "results").glob(".gradient_direction_by_block.csv.*.writing")
    )


class _TinyTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.bias = nn.Parameter(torch.tensor([-0.2, 0.3]))

    def forward(
        self,
        *,
        input_expression: torch.Tensor,
        gene_mask: torch.Tensor,
        edge_index: torch.Tensor,
        relative_geometry: torch.Tensor,
        node_covariates: torch.Tensor,
        target_nodes: torch.Tensor,
    ) -> torch.Tensor:
        del gene_mask, edge_index, relative_geometry, node_covariates, target_nodes
        return self.bias + self.weight * input_expression.mean(dim=1, keepdim=True)


def _tiny_batches() -> tuple[CohortRelativeQKVCoreBatch, ...]:
    result = []
    for index, alias in enumerate(SO2_14CORE_ALIASES):
        result.append(
            CohortRelativeQKVCoreBatch(
                alias=alias,
                target_expression=torch.tensor(
                    [[1.0 + index * 0.01, 2.0 + index * 0.01]]
                ),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                relative_geometry=torch.empty((0, 70)),
                node_covariates=torch.ones((1, 1)),
            )
        )
    return tuple(result)


def test_training_hook_is_pre_clip_pre_step_and_numerically_read_only() -> None:
    config = CohortRelativeQKVTrainingConfig(
        model_seed=12,
        segment_end_global_epoch=1,
        learning_rate=0.01,
        gradient_clip_norm=1e-6,
        device="cpu",
    )
    control = _TinyTrainingModel()
    observed = _TinyTrainingModel()
    observed.load_state_dict(control.state_dict())
    batches = _tiny_batches()

    fit_cohort_relative_qkv_segment(control, batches, config)
    tracker = FullGradientDirectionTracker()
    parameter_snapshots: list[tuple[float, float]] = []
    unscaled_norms: list[float] = []

    def observe_gradient(
        model: nn.Module, context: GradientDirectionUpdateContext
    ) -> None:
        parameter_snapshots.append(
            (
                float(model.weight.detach()),  # type: ignore[attr-defined]
                float(model.bias.detach()[0]),  # type: ignore[attr-defined]
            )
        )
        unscaled_norms.append(
            float(torch.linalg.vector_norm(full_trainable_gradient_vector(model)))
        )
        tracker.observe(model, context)

    summaries: list[GradientDirectionEpochSummary] = []
    fit_cohort_relative_qkv_segment(
        observed,
        batches,
        config,
        gradient_observer=observe_gradient,
        epoch_callback=lambda epoch, *_args: summaries.append(
            tracker.complete_epoch(epoch.global_epoch)
        ),
    )

    assert len(parameter_snapshots) == 7
    assert parameter_snapshots[0] == pytest.approx((2.0, -0.2))
    assert all(norm > config.gradient_clip_norm for norm in unscaled_norms)
    assert len(summaries) == 1
    assert summaries[0].consecutive_optimizer_step_cosine_valid_pairs == 6
    for name, parameter in observed.state_dict().items():
        assert torch.equal(parameter, control.state_dict()[name])
