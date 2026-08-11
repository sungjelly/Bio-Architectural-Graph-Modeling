from __future__ import annotations

import numpy as np
import pytest
import torch

from spatial_benchmark.same_gene_nonlinear import (
    AdditiveNeighborMLP,
    SameGeneNonlinearError,
    exact_gelu_derivative,
    planted_nonlinear_jacobian_control,
)


def test_exact_gelu_derivative_matches_autograd() -> None:
    values = torch.linspace(-3, 3, 41, dtype=torch.float64, requires_grad=True)
    output = torch.nn.functional.gelu(values, approximate="none")
    expected = torch.autograd.grad(output.sum(), values)[0]
    assert torch.max(torch.abs(exact_gelu_derivative(values) - expected)).item() < 1e-12


def test_mean_neighbor_jacobian_matches_explicit_population_gradient() -> None:
    torch.manual_seed(3)
    model = AdditiveNeighborMLP(
        gene_count=5,
        morphology_count=2,
        hidden_count=4,
        use_neighbor=True,
    ).double()
    model.eval()
    morphology = torch.randn(9, 2, dtype=torch.float64)
    neighbor = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    observed = model.mean_neighbor_jacobian(neighbor.detach(), chunk_size=2)
    prediction = model(morphology, neighbor)
    expected = np.zeros((5, 5), dtype=np.float64)
    for target in range(5):
        gradient = torch.autograd.grad(
            prediction[:, target].mean(), neighbor, retain_graph=True
        )[0]
        expected[target] = gradient.sum(dim=0).detach().numpy()
    assert np.max(np.abs(observed - expected)) < 1e-11


def test_weighted_mean_neighbor_jacobian_matches_explicit_gradient() -> None:
    torch.manual_seed(13)
    model = AdditiveNeighborMLP(
        gene_count=4,
        morphology_count=3,
        hidden_count=5,
        use_neighbor=True,
    ).double()
    model.eval()
    morphology = torch.randn(7, 3, dtype=torch.float64)
    neighbor = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    weights = torch.arange(1, 8, dtype=torch.float64)
    weights = weights / weights.sum()
    observed = model.mean_neighbor_jacobian(
        neighbor.detach(), weights=weights, chunk_size=3
    )
    prediction = model(morphology, neighbor)
    expected = np.zeros((4, 4), dtype=np.float64)
    for target in range(4):
        objective = torch.sum(weights * prediction[:, target])
        gradient = torch.autograd.grad(objective, neighbor, retain_graph=True)[0]
        expected[target] = gradient.sum(dim=0).detach().numpy()
    assert np.max(np.abs(observed - expected)) < 1e-11


def test_mean_hidden_derivative_reconstructs_reported_nonlinear_jacobian() -> None:
    torch.manual_seed(29)
    model = AdditiveNeighborMLP(
        gene_count=6,
        morphology_count=2,
        hidden_count=4,
        use_neighbor=True,
    ).double()
    neighbor = torch.randn(11, 6, dtype=torch.float64)
    weights = torch.arange(1, 12, dtype=torch.float64)
    parts = model.mean_neighbor_jacobian_parts(
        neighbor, weights=weights, chunk_size=3
    )
    hidden = model.mean_hidden_derivative(
        neighbor, weights=weights, chunk_size=3
    )
    expected = (
        model.neighbor_out.weight.detach()
        * torch.from_numpy(hidden.copy()).unsqueeze(0)
    ) @ model.neighbor_in.weight.detach()

    assert np.array_equal(parts["mean_hidden_derivative"], hidden)
    assert np.max(np.abs(parts["nonlinear"] - expected.numpy())) < 1e-15


def test_morphology_only_model_rejects_neighbor_and_jacobian() -> None:
    model = AdditiveNeighborMLP(
        gene_count=3,
        morphology_count=2,
        hidden_count=2,
        use_neighbor=False,
    )
    morphology = torch.zeros(4, 2)
    assert model(morphology).shape == (4, 3)
    with pytest.raises(SameGeneNonlinearError, match="unused neighbor"):
        model(morphology, torch.zeros(4, 3))
    with pytest.raises(SameGeneNonlinearError, match="no neighbor Jacobian"):
        model.mean_neighbor_jacobian(torch.zeros(4, 3))


def test_planted_nonlinear_control_passes() -> None:
    result = planted_nonlinear_jacobian_control()
    assert result["passed"] is True
    assert result["maximum_autograd_error"] <= 1e-10
    assert result["maximum_finite_difference_error"] <= 1e-8
