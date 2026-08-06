from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from spatial_benchmark.self_hurdle import (
    SelfHurdleModel,
    SelfHurdleSplitView,
    evaluate_fixed_self_hurdle_mask,
    fit_full_core_self_hurdle_model,
    trainable_parameter_count,
)
from spatial_benchmark.training import TrainingConfig


def _fixture() -> tuple[SelfHurdleSplitView, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    expression = torch.randint(
        0, 5, (24, 12), generator=generator
    ).float()
    expression[0] = 0
    expression[1] = 1
    coordinates = torch.stack(
        (torch.arange(24, dtype=torch.float32), torch.zeros(24)), dim=1
    )
    covariates = torch.randn(24, 3, generator=generator)
    return (
        SelfHurdleSplitView(
            expression=expression,
            coordinates_um=coordinates,
            node_covariates=covariates,
            block_ids=np.repeat(np.arange(6), 4),
            name="fit",
        ),
        torch.zeros(12),
        torch.ones(12),
    )


def _model(mean: torch.Tensor, scale: torch.Tensor) -> SelfHurdleModel:
    return SelfHurdleModel(
        num_genes=12,
        expression_mean=mean,
        expression_scale=scale,
        node_covariate_dim=3,
        hidden_dim=24,
        decoder_dim=24,
        ffn_dim=48,
        residual_blocks=2,
        dropout=0.0,
    )


def _training() -> TrainingConfig:
    return TrainingConfig(
        max_epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        huber_delta=1.0,
        patience=0,
        curriculum="P+N+B",
        warmup_epochs=0,
        partial_gene_rate=0.25,
        node_rate=0.25,
        block_node_rate=0.25,
        mask_seed=101,
        model_seed=202,
        edge_dropout=0.0,
        amp=False,
        deterministic=True,
        deterministic_warn_only=False,
        device="cpu",
        restore_best=False,
    )


def test_model_surface_has_no_graph_or_edge_arguments() -> None:
    parameters = set(inspect.signature(SelfHurdleModel.forward).parameters)
    assert parameters == {
        "self",
        "input_expression",
        "gene_mask",
        "node_covariates",
        "target_nodes",
    }


def test_masked_values_cannot_change_selected_predictions() -> None:
    view, mean, scale = _fixture()
    model = _model(mean, scale).eval()
    mask = torch.zeros_like(view.expression, dtype=torch.bool)
    mask[2:7, ::2] = True
    altered = view.expression.clone()
    altered[mask] += 1000
    selected = torch.arange(2, 7)

    with torch.no_grad():
        first = model(
            view.expression,
            mask,
            node_covariates=view.node_covariates,
            target_nodes=selected,
        )
        second = model(
            altered,
            mask,
            node_covariates=view.node_covariates,
            target_nodes=selected,
        )

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_large_contract_parameter_count_is_stable() -> None:
    model = SelfHurdleModel(
        num_genes=1000,
        expression_mean=torch.zeros(1000),
        expression_scale=torch.ones(1000),
        node_covariate_dim=22,
        hidden_dim=768,
        decoder_dim=768,
        ffn_dim=1536,
        residual_blocks=3,
    )
    assert trainable_parameter_count(model) == 16_917_200


def test_fixed_budget_training_and_percentage_metrics() -> None:
    view, mean, scale = _fixture()
    model = _model(mean, scale)
    result = fit_full_core_self_hurdle_model(
        model,
        view,
        _training(),
        expression_mean=mean,
        expression_scale=scale,
        target_node_batch_size=64,
    )
    assert result.final_epoch == 1
    assert len(result.history) == 2
    assert result.graph_execution == "none_graph_inputs_prohibited"
    assert all(
        np.isfinite(record.train_hurdle_loss)
        and record.peak_cuda_memory_bytes == 0
        for record in result.history
    )

    mask = torch.zeros_like(view.expression, dtype=torch.bool)
    mask[:8] = True
    evaluated = evaluate_fixed_self_hurdle_mask(
        model,
        view,
        mask,
        expression_mean=mean,
        expression_scale=scale,
        target_node_batch_size=4,
        device="cpu",
    )
    metrics = evaluated.evaluation.metrics
    for name in (
        "detection_balanced_accuracy",
        "state8_exact_accuracy",
        "state8_balanced_accuracy",
        "positive_count_state_exact_accuracy",
        "positive_count_state_within_one_accuracy",
    ):
        assert 0.0 <= float(metrics[name]) <= 1.0


def test_view_rejects_non_count_expression() -> None:
    view, _, _ = _fixture()
    invalid = view.expression.clone()
    invalid[0, 0] = 0.5
    with pytest.raises(ValueError, match="integer"):
        SelfHurdleSplitView(
            expression=invalid,
            coordinates_um=view.coordinates_um,
            node_covariates=view.node_covariates,
            name="fit",
        )

