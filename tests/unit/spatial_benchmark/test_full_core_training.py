"""Tests for explicitly held-in fixed-budget training."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatial_benchmark.full_core_training import fit_full_core_model
from spatial_benchmark.models import ModelOutput
from spatial_benchmark.training import GraphSplitView, TrainingConfig, build_model


def _fit_view() -> GraphSplitView:
    generator = torch.Generator().manual_seed(91)
    expression = torch.randn(10, 5, generator=generator)
    coordinates = torch.stack(
        (
            torch.arange(10, dtype=torch.float32),
            torch.zeros(10, dtype=torch.float32),
        ),
        dim=1,
    )
    sources: list[int] = []
    receivers: list[int] = []
    for node in range(10):
        other = (node + 1) % 10
        sources.extend((node, other))
        receivers.extend((other, node))
    return GraphSplitView(
        expression=expression,
        coordinates_um=coordinates,
        edge_index=torch.tensor([sources, receivers]),
        node_covariates=torch.randn(10, 2, generator=generator),
        edge_attributes=torch.ones(20, 1),
        block_ids=np.repeat(np.arange(2), 5),
        name="fit",
    )


def _config(*, restore_best: bool = False) -> TrainingConfig:
    return TrainingConfig(
        max_epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=0,
        curriculum="P-only",
        warmup_epochs=0,
        partial_gene_rate=0.4,
        mask_seed=31,
        model_seed=7,
        edge_dropout=0.0,
        amp=False,
        device="cpu",
        restore_best=restore_best,
    )


def test_full_core_training_runs_fixed_budget_and_keeps_final_epoch() -> None:
    view = _fit_view()
    model = build_model(
        "b0",
        num_genes=view.num_genes,
        node_covariate_dim=view.node_covariate_dim,
        seed=7,
        hidden_dim=8,
        ffn_dim=12,
        decoder_dim=10,
        dropout=0.0,
    )
    diagnostic = torch.zeros_like(view.expression, dtype=torch.bool)
    diagnostic[:2] = True
    result = fit_full_core_model(
        model,
        view,
        _config(),
        diagnostic_mask=diagnostic,
        diagnostic_every=2,
    )

    assert len(result.history) == 3
    assert result.final_epoch == 2
    assert result.checkpoint_policy == "final_epoch_no_validation_selection"
    assert result.history[0].diagnostic_loss is not None
    assert result.history[1].diagnostic_loss is not None
    assert result.history[2].diagnostic_loss is not None
    assert result.final_state_checksum
    assert result.final_train_loss == result.history[-1].train_loss


def test_full_core_training_rejects_validation_selection_semantics() -> None:
    view = _fit_view()
    model = build_model(
        "b0",
        num_genes=view.num_genes,
        node_covariate_dim=view.node_covariate_dim,
        seed=7,
        hidden_dim=8,
        ffn_dim=12,
        decoder_dim=10,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="restore_best=False"):
        fit_full_core_model(model, view, _config(restore_best=True))

    wrong_role = GraphSplitView(
        expression=view.expression,
        coordinates_um=view.coordinates_um,
        edge_index=view.edge_index,
        node_covariates=view.node_covariates,
        edge_attributes=view.edge_attributes,
        block_ids=view.block_ids,
        name="validation",
    )
    with pytest.raises(ValueError, match="held-in fit role"):
        fit_full_core_model(model, wrong_role, _config())


def test_full_core_training_supports_explicit_token_cross_entropy() -> None:
    regression_view = _fit_view()
    tokens = torch.randint(
        0,
        4,
        regression_view.expression.shape,
        generator=torch.Generator().manual_seed(20260726),
    ).float()
    view = GraphSplitView(
        expression=tokens,
        coordinates_um=regression_view.coordinates_um,
        edge_index=regression_view.edge_index,
        node_covariates=regression_view.node_covariates,
        edge_attributes=regression_view.edge_attributes,
        block_ids=regression_view.block_ids,
        name="fit",
    )

    class _TokenModel(torch.nn.Module):
        def __init__(self, num_genes: int) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.randn(num_genes, 4))

        def forward(
            self,
            input_expression: torch.Tensor,
            gene_mask: torch.Tensor,
            *,
            target_nodes: torch.Tensor | None = None,
            **_: object,
        ) -> ModelOutput:
            del gene_mask
            n_targets = (
                input_expression.shape[0]
                if target_nodes is None
                else target_nodes.numel()
            )
            logits = self.logits.unsqueeze(0).expand(n_targets, -1, -1)
            return ModelOutput(
                prediction=logits,
                node_embedding=logits.new_zeros((n_targets, 1)),
            )

    result = fit_full_core_model(
        _TokenModel(view.num_genes),
        view,
        _config(),
        objective="masked_token_cross_entropy",
        num_expression_tokens=4,
    )
    assert len(result.history) == 3
    assert all(np.isfinite(row.train_loss) for row in result.history)
    assert result.training_protocol == (
        "held_in_full_core_fixed_budget_token_classification"
    )

    with pytest.raises(ValueError, match="num_expression_tokens"):
        fit_full_core_model(
            _TokenModel(view.num_genes),
            view,
            _config(),
            objective="masked_token_cross_entropy",
        )
