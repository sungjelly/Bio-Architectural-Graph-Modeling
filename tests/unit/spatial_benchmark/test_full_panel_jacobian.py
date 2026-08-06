from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from spatial_benchmark.full_panel_jacobian import (
    FullPanelJacobianError,
    summarize_symmetric_directed_degree,
    uniform_shift_jacobian,
)
from spatial_benchmark.myjju_gradient_audit import (
    source_style_all_unmasked_gradients,
)


class DeterministicLinearModel(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_attr: torch.Tensor | None = None,
        entry_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del edge_index, edge_attr
        assert entry_mask is not None
        visible = torch.where(entry_mask, torch.zeros_like(x), x)
        return visible @ self.weight.T, entry_mask


def _prepared_model() -> DeterministicLinearModel:
    model = DeterministicLinearModel(
        torch.tensor(
            [[2.0, -1.0, 0.5], [0.25, 3.0, -2.0], [1.5, 0.0, 4.0]],
            dtype=torch.float64,
        )
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@pytest.mark.parametrize("target_batch_size", [1, 2, 3])
def test_uniform_shift_matches_explicit_source_style_sum(
    target_batch_size: int,
) -> None:
    model = _prepared_model()
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [4.0, 0.0, -2.0]],
        dtype=torch.float64,
    )
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    expected = source_style_all_unmasked_gradients(
        model,
        x,
        edges,
        target_gene_indices=(2, 0),
        source_gene_indices=(1, 2),
    )
    actual = uniform_shift_jacobian(
        model,
        x,
        edges,
        target_gene_indices=(2, 0),
        source_gene_indices=(1, 2),
        target_batch_size=target_batch_size,
    )
    assert actual.signed_jacobian.shape == (2, 2)
    np.testing.assert_allclose(
        actual.signed_jacobian,
        expected.signed_total,
        rtol=0.0,
        atol=0.0,
    )


def test_uniform_shift_full_panel_matches_linear_weight() -> None:
    model = _prepared_model()
    x = torch.zeros((4, 3), dtype=torch.float64)
    edges = torch.empty((2, 0), dtype=torch.long)
    result = uniform_shift_jacobian(model, x, edges, target_batch_size=2)
    np.testing.assert_array_equal(
        result.signed_jacobian,
        model.weight.detach().cpu().numpy(),
    )


def test_uniform_shift_requires_frozen_evaluation_model() -> None:
    model = nn.Linear(3, 3).train()
    x = torch.zeros((2, 3))
    edges = torch.empty((2, 0), dtype=torch.long)
    with pytest.raises(FullPanelJacobianError, match="evaluation mode"):
        uniform_shift_jacobian(model, x, edges)


def test_symmetric_degree_summary_reports_union_degree() -> None:
    edges = np.asarray(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=np.int64
    )
    summary = summarize_symmetric_directed_degree(
        edges, node_count=3, required_minimum=2
    )
    assert summary.minimum == 2
    assert summary.maximum == 2
    assert summary.mean == 2.0


def test_symmetric_degree_summary_rejects_low_degree() -> None:
    edges = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    with pytest.raises(FullPanelJacobianError, match="below 2"):
        summarize_symmetric_directed_degree(
            edges, node_count=3, required_minimum=2
        )
