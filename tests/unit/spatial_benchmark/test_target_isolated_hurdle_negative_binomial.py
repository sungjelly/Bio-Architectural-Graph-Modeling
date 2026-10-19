"""Model-output and initialization contracts for target-isolated hurdle-NB2."""

from __future__ import annotations

import torch

from spatial_benchmark.hurdle_negative_binomial import (
    HURDLE_DETECTION_PROBABILITY_EPSILON,
    HurdleNegativeBinomialModelOutput,
    TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    zero_truncated_nb2_conditional_mean,
)
from spatial_benchmark.negative_binomial import (
    NegativeBinomialModelOutput,
)
from spatial_benchmark.target_isolated_negative_binomial import (
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
)


BASE_PRODUCTION_PARAMETER_COUNT = 5_135_088
HURDLE_PRODUCTION_PARAMETER_COUNT = 6_160_088


def _small_model(
    initial_probability: torch.Tensor | None = None,
) -> TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer:
    return TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer(
        num_genes=4,
        node_covariate_dim=2,
        hidden_dim=8,
        attention_heads=2,
        attention_head_dim=4,
        graph_layers=4,
        ffn_dim=16,
        decoder_dim=12,
        geometry_hidden_dim=6,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=1,
        max_edges_per_chunk=2,
        activation_checkpointing=False,
        detection_initial_nonzero_probability=initial_probability,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(8401)
    expression = torch.randn(4, 4, generator=generator)
    covariates = torch.randn(4, 2, generator=generator)
    edge_index = torch.tensor(
        [
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ],
        dtype=torch.long,
    )
    geometry = torch.randn(edge_index.shape[1], 70, generator=generator)
    target_mask = torch.tensor(
        [[True, False, False, False], [False, True, False, False]],
        dtype=torch.bool,
    )
    return expression, target_mask, covariates, edge_index, geometry


def test_detection_initialization_clamps_prior_without_mutating_input() -> None:
    supplied = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float32)
    original = supplied.clone()
    model = _small_model(supplied)
    expected = torch.tensor(
        [
            HURDLE_DETECTION_PROBABILITY_EPSILON,
            0.25,
            0.75,
            1.0 - HURDLE_DETECTION_PROBABILITY_EPSILON,
        ],
        dtype=torch.float32,
    )

    assert torch.equal(supplied, original)
    torch.testing.assert_close(
        model.detection_initial_nonzero_probability,
        expected,
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        model.decoder.detection_linear_out.weight,
        torch.zeros_like(model.decoder.detection_linear_out.weight),
    )
    torch.testing.assert_close(
        model.decoder.detection_linear_out.bias,
        torch.logit(expected),
        rtol=0,
        atol=1e-7,
    )


def test_output_is_distinct_and_prediction_is_the_hurdle_marginal_mean() -> None:
    initial_probability = torch.tensor([0.2, 0.4, 0.6, 0.8])
    torch.manual_seed(8411)
    model = _small_model(initial_probability).eval()
    expression, target_mask, covariates, edge_index, geometry = _inputs()

    output = model(
        input_expression=expression,
        target_input_expression=expression[:2].masked_fill(target_mask, 0.0),
        target_gene_mask=target_mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        target_start=0,
        target_stop=2,
        return_intermediate_embeddings=True,
        return_graph_step_embeddings=True,
    )

    assert isinstance(output, HurdleNegativeBinomialModelOutput)
    assert not isinstance(output, NegativeBinomialModelOutput)
    assert output.prediction is output.marginal_mean
    assert output.nb_mean.shape == (2, 4)
    assert output.theta.shape == (4,)
    assert output.nonzero_logits.shape == (2, 4)
    for value in (
        output.nb_mean,
        output.theta,
        output.nonzero_probability,
        output.predicted_zero_probability,
        output.positive_conditional_mean,
        output.marginal_mean,
    ):
        assert value.dtype == torch.float32
        assert bool(torch.isfinite(value).all())

    expected_probability = initial_probability.expand(2, -1)
    torch.testing.assert_close(
        output.nonzero_probability,
        expected_probability,
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        output.predicted_zero_probability,
        1.0 - expected_probability,
        rtol=0,
        atol=1e-7,
    )
    expected_positive_mean = zero_truncated_nb2_conditional_mean(
        output.nb_mean,
        output.theta,
    )
    torch.testing.assert_close(
        output.positive_conditional_mean,
        expected_positive_mean,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output.marginal_mean,
        expected_probability * expected_positive_mean,
        rtol=0,
        atol=1e-7,
    )
    torch.testing.assert_close(output.theta, torch.ones(4), rtol=0, atol=1e-7)
    assert output.graph_step_embeddings is not None
    assert len(output.graph_step_embeddings) == 4
    assert model.observation_model == (
        "bernoulli_hurdle_zero_truncated_negative_binomial_nb2"
    )


def test_production_parameter_count_is_base_plus_one_decoder_width_head() -> None:
    torch.manual_seed(8423)
    base = TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer(
        num_genes=1000,
        node_covariate_dim=22,
    )
    torch.manual_seed(8423)
    hurdle = TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer(
        num_genes=1000,
        node_covariate_dim=22,
    )

    base_parameters = dict(base.named_parameters())
    hurdle_parameters = dict(hurdle.named_parameters())
    added = set(hurdle_parameters) - set(base_parameters)
    removed = set(base_parameters) - set(hurdle_parameters)

    assert sum(value.numel() for value in base_parameters.values()) == (
        BASE_PRODUCTION_PARAMETER_COUNT
    )
    assert sum(value.numel() for value in hurdle_parameters.values()) == (
        HURDLE_PRODUCTION_PARAMETER_COUNT
    )
    assert added == {
        "decoder.detection_linear_out.weight",
        "decoder.detection_linear_out.bias",
    }
    assert removed == set()
    assert sum(hurdle_parameters[name].numel() for name in added) == 1_025_000
    assert hurdle.decoder.detection_linear_out.in_features == 1024
    assert hurdle.decoder.detection_linear_out.out_features == 1000
