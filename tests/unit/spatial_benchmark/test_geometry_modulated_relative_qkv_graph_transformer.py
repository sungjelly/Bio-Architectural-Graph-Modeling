from __future__ import annotations

from itertools import combinations

import torch
from torch.nn import functional as F

from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import (
    GeometryModulatedRelativeQKVGraphTransformer,
    GeometryModulatedRelativeQKVGraphTransformerBlock,
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
)


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(701)
    expression = torch.randn(6, 5)
    mask = torch.tensor(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    covariates = torch.randn(6, 2)
    # Intentionally not receiver-sorted.  Each represented receiver has two or
    # three incoming edges, which makes receiver-wise normalization observable.
    edge_index = torch.tensor(
        [
            [4, 1, 5, 2, 0, 3, 2, 4, 0, 1, 3],
            [2, 0, 4, 1, 3, 1, 0, 3, 2, 4, 0],
        ],
        dtype=torch.long,
    )
    relative_geometry = torch.randn(edge_index.shape[1], 70)
    return expression, mask, covariates, edge_index, relative_geometry


def _model(
    cls: type[
        GeometryModulatedRelativeQKVGraphTransformer
        | ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer
    ],
) -> (
    GeometryModulatedRelativeQKVGraphTransformer
    | ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer
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
    if issubclass(
        cls,
        ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
    ):
        return cls(
            **kwargs,
            receiver_chunk_size=2,
            max_edges_per_chunk=3,
            activation_checkpointing=False,
        )
    return cls(**kwargs)


def _normalized_queries_and_keys(
    model: GeometryModulatedRelativeQKVGraphTransformer,
    expression: torch.Tensor,
    mask: torch.Tensor,
    covariates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block = model.blocks[0]
    node_embedding = model.encoder(expression, mask, covariates)
    normalized = block.attention_normalization(node_embedding)
    shape = (
        node_embedding.shape[0],
        block.attention_heads,
        block.attention_head_dim,
    )
    queries = F.normalize(
        block.query_projection(normalized).view(shape).float(),
        dim=-1,
    )
    keys = F.normalize(
        block.key_projection(normalized).view(shape).float(),
        dim=-1,
    )
    return queries, keys


def test_zero_initialized_geometry_heads_start_with_unit_modulation_and_zero_bias() -> None:
    expression, mask, covariates, edge_index, geometry = _inputs()
    torch.manual_seed(709)
    model = _model(GeometryModulatedRelativeQKVGraphTransformer).eval()
    block = model.blocks[0]

    modulation, positional_bias = block.geometry_encoder(geometry)
    torch.testing.assert_close(
        modulation,
        torch.ones_like(modulation),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(positional_bias) == 0
    assert torch.count_nonzero(
        block.geometry_encoder.modulation_projection.weight
    ) == 0
    assert torch.count_nonzero(block.geometry_encoder.bias_projection.weight) == 0

    first = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        return_explanations=True,
        explanation_layer=0,
    )
    changed_geometry = geometry.flip(0).mul(7.0).add(3.0)
    second = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=changed_geometry,
        node_covariates=covariates,
        return_explanations=True,
        explanation_layer=0,
    )

    assert torch.count_nonzero(first.positional_bias) == 0
    torch.testing.assert_close(first.content_logits, second.content_logits, rtol=0, atol=0)
    torch.testing.assert_close(first.combined_logits, first.content_logits, rtol=0, atol=0)
    torch.testing.assert_close(second.combined_logits, first.content_logits, rtol=0, atol=0)
    torch.testing.assert_close(first.prediction, second.prediction, rtol=0, atol=0)

    queries, keys = _normalized_queries_and_keys(
        model,
        expression,
        mask,
        covariates,
    )
    cosine = (
        queries.index_select(0, edge_index[1])
        * keys.index_select(0, edge_index[0])
    ).sum(dim=-1)
    # At neutral geometry, each head's content score must be a positive scalar
    # multiple of cosine compatibility.  A non-unit edge modulation would make
    # this ratio edge-dependent.
    informative = cosine.abs() > 1e-5
    for head in range(model.blocks[0].attention_heads):
        ratios = first.content_logits[informative[:, head], head] / cosine[
            informative[:, head], head
        ]
        assert ratios.numel() >= 2
        assert torch.isfinite(ratios).all()
        assert bool((ratios > 0).all())
        torch.testing.assert_close(
            ratios,
            ratios[0].expand_as(ratios),
            rtol=2e-5,
            atol=2e-6,
        )


def test_four_blocks_are_independently_parameterized() -> None:
    model = _model(GeometryModulatedRelativeQKVGraphTransformer)

    assert model.graph_layers == 4
    assert len(model.blocks) == 4
    assert all(
        isinstance(block, GeometryModulatedRelativeQKVGraphTransformerBlock)
        for block in model.blocks
    )
    parameter_id_sets = [
        {id(parameter) for parameter in block.parameters()}
        for block in model.blocks
    ]
    assert all(parameter_ids for parameter_ids in parameter_id_sets)
    for left, right in combinations(parameter_id_sets, 2):
        assert left.isdisjoint(right)

    state_keys = set(model.state_dict())
    for layer_number, block in enumerate(model.blocks):
        assert {
            key for key in state_keys if key.startswith(f"blocks.{layer_number}.")
        } == {
            f"blocks.{layer_number}.{key}" for key in block.state_dict()
        }


def test_attention_is_receiver_normalized_and_backward_gradients_are_finite() -> None:
    expression, mask, covariates, edge_index, geometry = _inputs()
    torch.manual_seed(719)
    model = _model(GeometryModulatedRelativeQKVGraphTransformer)
    output = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        return_explanations=True,
    )

    assert torch.isfinite(output.prediction).all()
    assert torch.isfinite(output.attention_weights).all()
    assert torch.isfinite(output.content_logits).all()
    assert torch.isfinite(output.positional_bias).all()
    assert torch.isfinite(output.combined_logits).all()
    for receiver in edge_index[1].unique():
        selected = output.receiver_indices == receiver
        torch.testing.assert_close(
            output.attention_weights[selected].sum(dim=0),
            torch.ones(3),
            rtol=2e-6,
            atol=2e-6,
        )

    output.prediction.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    for block in model.blocks:
        assert any(
            bool(torch.count_nonzero(parameter.grad))
            for parameter in block.parameters()
            if parameter.grad is not None
        )


def test_full_and_receiver_chunked_outputs_diagnostics_and_gradients_match() -> None:
    expression, mask, covariates, edge_index, geometry = _inputs()
    torch.manual_seed(727)
    full = _model(GeometryModulatedRelativeQKVGraphTransformer)
    chunked = _model(ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer)
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

    assert torch.equal(chunked_output.edge_index, full_output.edge_index)
    for chunked_tensor, full_tensor in (
        (chunked_output.prediction, full_output.prediction),
        (chunked_output.node_embedding, full_output.node_embedding),
        (chunked_output.attention_weights, full_output.attention_weights),
        (chunked_output.content_logits, full_output.content_logits),
        (chunked_output.positional_bias, full_output.positional_bias),
        (chunked_output.combined_logits, full_output.combined_logits),
    ):
        torch.testing.assert_close(
            chunked_tensor,
            full_tensor,
            rtol=2e-5,
            atol=2e-6,
        )

    full_output.prediction.square().mean().backward()
    chunked_output.prediction.square().mean().backward()
    for (full_name, full_parameter), (chunked_name, chunked_parameter) in zip(
        full.named_parameters(),
        chunked.named_parameters(),
        strict=True,
    ):
        assert full_name == chunked_name
        assert full_parameter.grad is not None
        assert chunked_parameter.grad is not None
        torch.testing.assert_close(
            chunked_parameter.grad,
            full_parameter.grad,
            rtol=3e-5,
            atol=3e-6,
        )
