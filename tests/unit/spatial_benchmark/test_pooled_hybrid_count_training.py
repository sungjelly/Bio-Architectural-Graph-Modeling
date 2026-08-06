"""Focused synthetic contracts for pooled ten-graph hybrid training."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from spatial_benchmark.models import ModelOutput  # noqa: E402
from spatial_benchmark.pooled_hybrid_count_training import (  # noqa: E402
    POOLED_CORE_ALIASES,
    PooledCoreBatch,
    fit_pooled_hybrid_count_model,
    make_pooled_epoch_mask,
    pooled_core_order,
)
from spatial_benchmark.training import (  # noqa: E402
    GraphSplitView,
    TrainingConfig,
)


_RAW_STATES = torch.tensor(
    [0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0]
)


def _batch(
    alias: str,
    *,
    invalid_positive_only: bool = False,
) -> PooledCoreBatch:
    alias_index = int(alias[-2:])
    if invalid_positive_only:
        counts = torch.ones(8 + alias_index, dtype=torch.float32)
    else:
        # Every core contains every required loss stratum.  Increasing numbers
        # of high-count cells make cell-weighted and equal-core means differ.
        counts = torch.cat(
            (
                _RAW_STATES,
                torch.full(
                    (alias_index * 3,),
                    32.0,
                    dtype=torch.float32,
                ),
            )
        )
    expression = counts.unsqueeze(1).repeat(1, 2)
    n_nodes = int(expression.shape[0])
    source = torch.arange(n_nodes, dtype=torch.long)
    receiver = torch.roll(source, shifts=-1)
    coordinates = torch.stack(
        (
            torch.arange(n_nodes, dtype=torch.float64),
            torch.zeros(n_nodes, dtype=torch.float64),
        ),
        dim=1,
    )
    return PooledCoreBatch(
        alias,
        GraphSplitView(
            expression=expression,
            coordinates_um=coordinates,
            edge_index=torch.stack((source, receiver)),
            node_covariates=torch.zeros(n_nodes, 1),
            edge_attributes=torch.ones(n_nodes, 1),
            name="fit",
        ),
    )


def _batches(
    *,
    invalid_positive_only: bool = False,
) -> tuple[PooledCoreBatch, ...]:
    return tuple(
        _batch(alias, invalid_positive_only=invalid_positive_only)
        for alias in POOLED_CORE_ALIASES
    )


def _config(
    *,
    epochs: int = 1,
    learning_rate: float = 1e-2,
    model_seed: int = 0,
) -> TrainingConfig:
    return TrainingConfig(
        max_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        huber_delta=1.0,
        patience=0,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=1.0,
        node_rate=0.5,
        block_node_rate=0.5,
        mask_seed=314159,
        model_seed=model_seed,
        edge_dropout=0.0,
        amp=False,
        deterministic=True,
        deterministic_warn_only=False,
        device="cpu",
        restore_best=False,
    )


class _FiniteForwardBadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value.clone()

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor]:
        del ctx
        return (torch.full_like(gradient, float("nan")),)


class _TinyHybridModel(torch.nn.Module):
    def __init__(
        self,
        *,
        nonfinite_prediction: bool = False,
        nonfinite_gradient: bool = False,
    ) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2, 8))
        self.nonfinite_prediction = nonfinite_prediction
        self.nonfinite_gradient = nonfinite_gradient
        self.forward_node_counts: list[int] = []
        self.forward_parameter_norms: list[float] = []
        self.cache_clear_calls = 0

    def clear_edge_layout_cache(self) -> None:
        self.cache_clear_calls += 1

    def forward(
        self,
        input_expression: torch.Tensor,
        gene_mask: torch.Tensor,
        *,
        target_nodes: torch.Tensor | None = None,
        **_: object,
    ) -> ModelOutput:
        del gene_mask
        self.forward_node_counts.append(int(input_expression.shape[0]))
        self.forward_parameter_norms.append(
            float(self.logits.detach().norm())
        )
        n_targets = (
            input_expression.shape[0]
            if target_nodes is None
            else int(target_nodes.numel())
        )
        prediction = self.logits.unsqueeze(0).expand(
            n_targets, -1, -1
        )
        if self.nonfinite_prediction:
            prediction = prediction * prediction.new_tensor(float("nan"))
        if self.nonfinite_gradient:
            prediction = _FiniteForwardBadBackward.apply(prediction)
        return ModelOutput(
            prediction=prediction,
            node_embedding=prediction.new_zeros((n_targets, 1)),
        )


def _fit(
    model: _TinyHybridModel,
    batches: tuple[PooledCoreBatch, ...],
    config: TrainingConfig,
    **kwargs: object,
):
    return fit_pooled_hybrid_count_model(
        model,
        batches,
        config,
        expression_mean=np.zeros(2, dtype=np.float32),
        expression_scale=np.ones(2, dtype=np.float32),
        **kwargs,
    )


def test_epoch_visits_every_core_once_in_frozen_order_and_takes_ten_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = tuple(reversed(_batches()))
    model = _TinyHybridModel()
    initial = model.logits.detach().clone()
    calls = 0
    original_step = torch.optim.AdamW.step

    def counted_step(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counted_step)
    result = _fit(model, batches, _config())

    expected_order = pooled_core_order(POOLED_CORE_ALIASES, 0)
    expected_nodes = {
        batch.alias: batch.view.num_nodes for batch in batches
    }
    assert calls == 10
    assert len(result.core_history) == 10
    assert len(result.global_history) == 1
    assert result.global_history[0].ordered_aliases == expected_order
    assert result.global_history[0].cores_visited == 10
    assert result.global_history[0].optimizer_steps == 10
    assert tuple(record.alias for record in result.core_history) == (
        expected_order
    )
    assert [record.optimizer_step for record in result.core_history] == list(
        range(1, 11)
    )
    assert model.forward_node_counts == [
        expected_nodes[alias] for alias in expected_order
    ]
    assert len(set(model.forward_node_counts)) == 10
    assert not torch.equal(initial, model.logits.detach())
    assert len(set(model.forward_parameter_norms)) > 1
    assert model.cache_clear_calls >= 2 * 10
    assert all(
        batch.view.expression.device.type == "cpu" for batch in batches
    )


def test_core_order_is_deterministic_and_caller_order_independent() -> None:
    forward = pooled_core_order(POOLED_CORE_ALIASES, 3)
    reversed_input = pooled_core_order(
        tuple(reversed(POOLED_CORE_ALIASES)), 3
    )
    assert forward == reversed_input
    assert forward == pooled_core_order(POOLED_CORE_ALIASES, 3)
    assert set(forward) == set(POOLED_CORE_ALIASES)
    assert len(forward) == 10
    with pytest.raises(ValueError, match="271828"):
        pooled_core_order(POOLED_CORE_ALIASES, 0, seed=271829)


def test_masks_are_alias_specific_paired_and_independent_of_model_seed() -> None:
    first = _batch("ANC-01")
    second = _batch("ANC-02")
    seed_zero = _config(model_seed=0)
    seed_six = replace(seed_zero, model_seed=6)

    mask_a = make_pooled_epoch_mask(first, seed_zero, 7)
    mask_b = make_pooled_epoch_mask(first, seed_six, 7)
    other_core = make_pooled_epoch_mask(second, seed_zero, 7)

    assert mask_a.seed == mask_b.seed
    np.testing.assert_array_equal(mask_a.mask, mask_b.mask)
    assert mask_a.seed != other_core.seed


def test_global_history_is_equal_core_mean_not_cell_weighted_mean() -> None:
    result = _fit(_TinyHybridModel(), _batches(), _config(learning_rate=0.0))
    losses = np.asarray(
        [record.train_hybrid_loss for record in result.core_history],
        dtype=np.float64,
    )
    masked_entries = np.asarray(
        [record.n_masked_entries for record in result.core_history],
        dtype=np.float64,
    )
    equal_core = float(losses.mean())
    cell_weighted = float(np.average(losses, weights=masked_entries))

    assert result.global_history[0].aggregation == (
        "equal_core_arithmetic_mean"
    )
    assert result.global_history[0].mean_hybrid_loss == pytest.approx(
        equal_core
    )
    assert equal_core != pytest.approx(cell_weighted, abs=1e-7)


def test_epoch_boundary_resume_matches_uninterrupted_shared_training() -> None:
    batches = _batches()
    uninterrupted = _fit(
        _TinyHybridModel(), batches, _config(epochs=2)
    )
    first_epoch = _fit(
        _TinyHybridModel(), batches, _config(epochs=1)
    )
    resumed = _fit(
        _TinyHybridModel(),
        batches,
        _config(epochs=2),
        resume=first_epoch.epoch_boundary_resume(),
    )

    assert resumed.completed_global_epochs == 2
    assert resumed.final_epoch == 1
    assert len(resumed.global_history) == 1
    assert resumed.global_history[0].epoch == 1
    assert resumed.final_state_checksum == uninterrupted.final_state_checksum


def test_nonfinite_predictions_and_gradients_fail_closed() -> None:
    with pytest.raises(FloatingPointError, match="prediction"):
        _fit(
            _TinyHybridModel(nonfinite_prediction=True),
            _batches(),
            _config(),
        )

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        _fit(
            _TinyHybridModel(nonfinite_gradient=True),
            _batches(),
            _config(),
        )


def test_missing_within_core_loss_stratum_fails_closed() -> None:
    with pytest.raises(ValueError, match="zero and positive strata"):
        _fit(
            _TinyHybridModel(),
            _batches(invalid_positive_only=True),
            _config(),
        )


def test_api_rejects_non_anc_alias_and_incomplete_ten_core_set() -> None:
    with pytest.raises(ValueError, match="ANC-01 through ANC-10"):
        PooledCoreBatch("core-one", _batch("ANC-01").view)

    with pytest.raises(ValueError, match="exactly ten"):
        _fit(
            _TinyHybridModel(),
            _batches()[:-1],
            _config(),
        )
