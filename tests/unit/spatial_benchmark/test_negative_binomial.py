"""Focused contracts for the four-block geometry-modulated NB2 model."""

from __future__ import annotations

import inspect
from itertools import combinations

import pytest
import torch

from spatial_benchmark.negative_binomial import (
    GeometryModulatedNegativeBinomialGraphTransformer,
    NegativeBinomialModelOutput,
    ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
    masked_log1p_mae,
    masked_log1p_rmse,
    masked_negative_binomial_metrics,
    masked_negative_binomial_nll,
    masked_observed_zero_rate,
    masked_poisson_deviance,
    masked_predicted_zero_probability_mean,
    masked_raw_count_mae,
    masked_raw_count_rmse,
    masked_zero_brier_score,
    nb2_inverse_dispersion_from_raw,
    nb2_mean_from_logits,
    negative_binomial_nll_elementwise,
    negative_binomial_zero_probability,
)


def _small_model(
    cls: type[
        GeometryModulatedNegativeBinomialGraphTransformer
        | ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer
    ] = ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
) -> (
    GeometryModulatedNegativeBinomialGraphTransformer
    | ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer
):
    kwargs = dict(
        num_genes=5,
        node_covariate_dim=2,
        hidden_dim=12,
        attention_heads=3,
        attention_head_dim=4,
        graph_layers=4,
        ffn_dim=24,
        decoder_dim=16,
        geometry_hidden_dim=10,
        dropout=0.0,
        attention_dropout=0.0,
    )
    if cls is ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer:
        return cls(
            **kwargs,
            receiver_chunk_size=2,
            max_edges_per_chunk=3,
            activation_checkpointing=False,
        )
    return cls(**kwargs)


def _small_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(1103)
    expression = torch.randn(4, 5)
    mask = torch.tensor(
        [
            [True, False, False, True, False],
            [False, True, False, False, True],
            [False, False, True, True, False],
            [True, False, True, False, False],
        ]
    )
    covariates = torch.randn(4, 2)
    edge_index = torch.tensor(
        [[0, 2, 1, 3, 0, 2], [1, 1, 2, 2, 3, 3]],
        dtype=torch.long,
    )
    geometry = torch.randn(edge_index.shape[1], 70)
    return expression, mask, covariates, edge_index, geometry


def test_output_is_positive_mu_and_cell_shared_unit_initialized_theta() -> None:
    expression, mask, covariates, edge_index, geometry = _small_inputs()
    torch.manual_seed(1109)
    model = _small_model().eval()
    output = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        return_explanations=True,
    )

    assert isinstance(output, NegativeBinomialModelOutput)
    assert output.prediction is output.mu
    assert output.mu.shape == expression.shape
    assert output.mu.dtype == torch.float32
    assert bool(torch.isfinite(output.mu).all())
    assert bool((output.mu > 0).all())
    assert output.theta.shape == (expression.shape[1],)
    assert output.theta.dtype == torch.float32
    torch.testing.assert_close(output.theta, torch.ones(5), rtol=0, atol=1e-7)
    assert tuple(model.state_dict())[0] == "raw_theta"
    assert model.raw_theta.shape == (5,)
    assert output.attention_weights is not None
    assert output.content_logits is not None
    assert output.positional_bias is not None
    assert output.combined_logits is not None


def test_raw_target_is_not_a_forward_input_and_hidden_values_cannot_change_mu() -> None:
    expression, mask, covariates, edge_index, geometry = _small_inputs()
    torch.manual_seed(1117)
    model = _small_model().eval()
    changed_hidden_values = expression.clone()
    changed_hidden_values[mask] = torch.linspace(100.0, 800.0, int(mask.sum()))

    first = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
    )
    second = model(
        changed_hidden_values,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
    )

    assert "raw_target" not in inspect.signature(model.forward).parameters
    torch.testing.assert_close(first.mu, second.mu, rtol=0, atol=0)
    torch.testing.assert_close(first.theta, second.theta, rtol=0, atol=0)


def test_full_and_receiver_chunked_nb_outputs_match() -> None:
    expression, mask, covariates, edge_index, geometry = _small_inputs()
    torch.manual_seed(1123)
    full = _small_model(GeometryModulatedNegativeBinomialGraphTransformer).eval()
    chunked = _small_model().eval()
    chunked.load_state_dict(full.state_dict(), strict=True)

    full_output = full(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        return_explanations=True,
        explanation_layer=2,
    )
    chunked_output = chunked(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        return_explanations=True,
        explanation_layer=2,
    )

    torch.testing.assert_close(full_output.mu, chunked_output.mu, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(full_output.theta, chunked_output.theta, rtol=0, atol=0)
    torch.testing.assert_close(
        full_output.attention_weights,
        chunked_output.attention_weights,
        rtol=2e-6,
        atol=2e-6,
    )


def test_production_shape_has_exact_parameter_and_independent_block_counts() -> None:
    torch.manual_seed(1129)
    model = ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer(
        num_genes=1000,
        node_covariate_dim=22,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == 5_135_088
    assert model.raw_theta.numel() == 1_000
    assert len(model.blocks) == 4
    block_counts = [
        sum(parameter.numel() for parameter in block.parameters())
        for block in model.blocks
    ]
    assert block_counts == [831_880, 831_880, 831_880, 831_880]
    parameter_ids = [
        {id(parameter) for parameter in block.parameters()} for block in model.blocks
    ]
    for left, right in combinations(parameter_ids, 2):
        assert left.isdisjoint(right)


def test_nb2_formula_matches_torch_distribution_with_full_constants() -> None:
    mu = torch.tensor(
        [[0.1, 1.3, 7.0], [3.2, 0.5, 19.0]],
        dtype=torch.float32,
    )
    theta = torch.tensor([0.7, 1.0, 4.0], dtype=torch.float32)
    raw_target = torch.tensor(
        [[0, 1, 729], [2, 0, 17]],
        dtype=torch.int32,
    )
    mask = torch.tensor(
        [[True, False, True], [False, True, True]],
        dtype=torch.bool,
    )

    actual = negative_binomial_nll_elementwise(mu, theta, raw_target)
    distribution = torch.distributions.NegativeBinomial(
        total_count=theta.view(1, -1),
        logits=torch.log(mu) - torch.log(theta.view(1, -1)),
    )
    expected = -distribution.log_prob(raw_target.float())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
    torch.testing.assert_close(
        masked_negative_binomial_nll(mu, theta, raw_target, mask),
        expected.masked_select(mask).mean(),
        rtol=2e-6,
        atol=2e-5,
    )


def test_zero_and_maximum_observed_count_have_finite_nonzero_gradients() -> None:
    decoder_logits = torch.tensor(
        [[-4.0, 0.0], [1.5, 3.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_theta = torch.tensor(
        [-1.0, 0.7],
        dtype=torch.float32,
        requires_grad=True,
    )
    raw_target = torch.tensor([[0, 729], [7, 0]], dtype=torch.int32)
    mask = torch.ones_like(raw_target, dtype=torch.bool)
    mu = nb2_mean_from_logits(decoder_logits)
    theta = nb2_inverse_dispersion_from_raw(raw_theta)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = masked_negative_binomial_nll(mu, theta, raw_target, mask)
    assert loss.dtype == torch.float32
    assert bool(torch.isfinite(loss))
    loss.backward()

    for gradient in (decoder_logits.grad, raw_theta.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
        assert bool((gradient != 0).any())


def test_mask_is_the_only_likelihood_support() -> None:
    mu = torch.tensor([[0.5, 1.0, 2.0], [3.0, 4.0, 5.0]])
    theta = torch.tensor([0.8, 1.2, 2.0])
    raw_target = torch.tensor([[0, 1, 2], [3, 4, 5]])
    mask = torch.tensor(
        [[True, False, True], [False, True, False]],
        dtype=torch.bool,
    )
    changed_mu = mu.clone()
    changed_mu[~mask] = torch.tensor([100.0, 200.0, 300.0])
    changed_target = raw_target.clone()
    changed_target[~mask] = torch.tensor([729, 729, 729])

    first = masked_negative_binomial_nll(mu, theta, raw_target, mask)
    second = masked_negative_binomial_nll(
        changed_mu,
        theta,
        changed_target,
        mask,
    )
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(ValueError, match="at least one"):
        masked_negative_binomial_nll(
            mu,
            theta,
            raw_target,
            torch.zeros_like(mask),
        )


def test_raw_count_metrics_and_nb_zero_probability_match_direct_definitions() -> None:
    mu = torch.tensor([[0.5, 2.0], [4.0, 8.0]])
    theta = torch.tensor([1.0, 3.0])
    raw_target = torch.tensor([[0, 3], [4, 0]], dtype=torch.int64)
    mask = torch.tensor([[True, True], [False, True]])

    metrics = masked_negative_binomial_metrics(mu, theta, raw_target, mask)
    selected_mu = mu[mask]
    selected_target = raw_target.float()[mask]
    expected_p0 = torch.pow(
        theta.view(1, -1) / (theta.view(1, -1) + mu),
        theta.view(1, -1),
    )
    expected_poisson = 2.0 * (
        torch.xlogy(selected_target, selected_target / selected_mu)
        - selected_target
        + selected_mu
    )
    observed_zero = (selected_target == 0).float()

    assert metrics.n_masked_entries == 3
    torch.testing.assert_close(
        negative_binomial_zero_probability(mu, theta),
        expected_p0,
    )
    torch.testing.assert_close(
        metrics.negative_binomial_nll,
        masked_negative_binomial_nll(mu, theta, raw_target, mask),
    )
    torch.testing.assert_close(
        metrics.raw_count_mae,
        torch.abs(selected_mu - selected_target).mean(),
    )
    torch.testing.assert_close(
        metrics.raw_count_rmse,
        torch.square(selected_mu - selected_target).mean().sqrt(),
    )
    torch.testing.assert_close(
        metrics.log1p_mae,
        torch.abs(torch.log1p(selected_mu) - torch.log1p(selected_target)).mean(),
    )
    torch.testing.assert_close(
        metrics.log1p_rmse,
        torch.square(
            torch.log1p(selected_mu) - torch.log1p(selected_target)
        ).mean().sqrt(),
    )
    torch.testing.assert_close(metrics.poisson_deviance, expected_poisson.mean())
    torch.testing.assert_close(metrics.observed_zero_rate, observed_zero.mean())
    torch.testing.assert_close(
        metrics.predicted_zero_probability_mean,
        expected_p0[mask].mean(),
    )
    torch.testing.assert_close(
        metrics.zero_brier_score,
        torch.square(expected_p0[mask] - observed_zero).mean(),
    )

    torch.testing.assert_close(
        masked_raw_count_mae(mu, raw_target, mask), metrics.raw_count_mae
    )
    torch.testing.assert_close(
        masked_raw_count_rmse(mu, raw_target, mask), metrics.raw_count_rmse
    )
    torch.testing.assert_close(
        masked_log1p_mae(mu, raw_target, mask), metrics.log1p_mae
    )
    torch.testing.assert_close(
        masked_log1p_rmse(mu, raw_target, mask), metrics.log1p_rmse
    )
    torch.testing.assert_close(
        masked_poisson_deviance(mu, raw_target, mask), metrics.poisson_deviance
    )
    torch.testing.assert_close(
        masked_zero_brier_score(mu, theta, raw_target, mask),
        metrics.zero_brier_score,
    )
    torch.testing.assert_close(
        masked_observed_zero_rate(raw_target, mask),
        metrics.observed_zero_rate,
    )
    torch.testing.assert_close(
        masked_predicted_zero_probability_mean(mu, theta, mask),
        metrics.predicted_zero_probability_mean,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"graph_layers": 3},
        {"mean_epsilon": 1e-5},
        {"inverse_dispersion_epsilon": 1e-5},
        {"inverse_dispersion_initial_value": 2.0},
    ],
)
def test_frozen_architecture_and_nb_parameterization_reject_mutation(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError, match="four|frozen"):
        ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer(
            num_genes=5,
            node_covariate_dim=2,
            hidden_dim=12,
            attention_heads=3,
            attention_head_dim=4,
            ffn_dim=24,
            decoder_dim=16,
            geometry_hidden_dim=10,
            activation_checkpointing=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("raw_target", "message"),
    [
        (torch.tensor([[0.0, -1.0]]), "nonnegative"),
        (torch.tensor([[0.0, 1.5]]), "integer-valued"),
        (torch.tensor([[0.0, float("nan")]]), "finite"),
    ],
)
def test_likelihood_rejects_invalid_masked_counts(
    raw_target: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        masked_negative_binomial_nll(
            torch.ones(1, 2),
            torch.ones(2),
            raw_target,
            torch.ones(1, 2, dtype=torch.bool),
        )
