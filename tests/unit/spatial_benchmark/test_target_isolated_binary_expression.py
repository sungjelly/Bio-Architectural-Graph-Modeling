"""Model and initialization contracts for binary expression detection."""

from __future__ import annotations

import torch

from spatial_benchmark.binary_expression import (
    BINARY_EXPRESSION_PROBABILITY_EPSILON,
    BinaryExpressionModelOutput,
    TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
)
from spatial_benchmark.negative_binomial import NegativeBinomialModelOutput


BINARY_EXPRESSION_PRODUCTION_PARAMETER_COUNT = 5_134_088


def _small_model(
    initial_probability: torch.Tensor | None = None,
) -> TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer:
    return TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer(
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
    generator = torch.Generator().manual_seed(9401)
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


def test_detection_initialization_clamps_prior_without_count_parameters() -> None:
    supplied = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float32)
    original = supplied.clone()
    model = _small_model(supplied)
    expected = torch.tensor(
        [
            BINARY_EXPRESSION_PROBABILITY_EPSILON,
            0.25,
            0.75,
            1.0 - BINARY_EXPRESSION_PROBABILITY_EPSILON,
        ],
        dtype=torch.float32,
    )

    assert torch.equal(supplied, original)
    assert "raw_theta" not in dict(model.named_parameters())
    assert "raw_theta" not in model.state_dict()
    torch.testing.assert_close(
        model.detection_initial_nonzero_probability,
        expected,
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        model.decoder.linear_out.weight,
        torch.zeros_like(model.decoder.linear_out.weight),
    )
    assert model.detection_head is model.decoder.linear_out
    torch.testing.assert_close(
        model.decoder.linear_out.bias,
        torch.logit(expected),
        rtol=0,
        atol=1e-7,
    )


def test_output_is_probability_only_and_preserves_graph_embeddings() -> None:
    initial_probability = torch.tensor([0.2, 0.4, 0.6, 0.8])
    torch.manual_seed(9411)
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

    assert isinstance(output, BinaryExpressionModelOutput)
    assert not isinstance(output, NegativeBinomialModelOutput)
    assert output.prediction is output.nonzero_probability
    assert output.nonzero_logits.shape == (2, 4)
    assert output.nonzero_probability.shape == (2, 4)
    assert output.predicted_zero_probability.shape == (2, 4)
    expected = initial_probability.expand(2, -1)
    torch.testing.assert_close(output.nonzero_probability, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(
        output.predicted_zero_probability,
        1.0 - expected,
        rtol=0,
        atol=1e-7,
    )
    assert output.graph_step_embeddings is not None
    assert len(output.graph_step_embeddings) == 4
    assert model.observation_model == "bernoulli_expression_presence"


def test_production_model_has_one_shared_trunk_and_one_detection_head() -> None:
    model = TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer(
        num_genes=1000,
        node_covariate_dim=22,
    )
    parameters = dict(model.named_parameters())

    assert sum(value.numel() for value in parameters.values()) == (
        BINARY_EXPRESSION_PRODUCTION_PARAMETER_COUNT
    )
    assert "raw_theta" not in parameters
    assert model.decoder.linear_in.in_features == 256
    assert model.decoder.linear_in.out_features == 1024
    assert model.decoder.linear_out.in_features == 1024
    assert model.decoder.linear_out.out_features == 1000
