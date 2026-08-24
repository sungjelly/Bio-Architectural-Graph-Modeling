from __future__ import annotations

import inspect

import pytest
import torch

from spatial_benchmark.models import SharedEdgeEncoder
from spatial_benchmark.relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    RelativeGeometryQKVGraphTransformer,
    RelativePositionBiasEncoder,
)


def _inputs(device: torch.device | str = "cpu") -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    expression = torch.randn(5, 6, device=device)
    mask = torch.tensor(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
        device=device,
    )
    metadata = torch.randn(5, 3, device=device)
    edge_index = torch.tensor(
        [[1, 2, 0, 2, 3, 0, 4], [0, 0, 1, 1, 1, 2, 3]],
        dtype=torch.long,
    )
    geometry = torch.randn(edge_index.shape[1], 70)
    return expression, mask, metadata, edge_index, geometry


def _model(
    cls: type[RelativeGeometryQKVGraphTransformer],
    *,
    checkpointing: bool = False,
) -> RelativeGeometryQKVGraphTransformer:
    kwargs = dict(
        num_genes=6,
        node_covariate_dim=3,
        hidden_dim=16,
        attention_heads=4,
        attention_head_dim=4,
        graph_layers=2,
        ffn_dim=32,
        decoder_dim=24,
        positional_bias_hidden_dim=12,
        dropout=0.0,
        attention_dropout=0.0,
    )
    if issubclass(cls, ReceiverChunkedRelativeGeometryQKVGraphTransformer):
        return cls(
            **kwargs,
            receiver_chunk_size=2,
            max_edges_per_chunk=4,
            activation_checkpointing=checkpointing,
        )
    return cls(**kwargs)


def test_positional_bias_starts_neutral_and_architecture_has_no_edge_values() -> None:
    encoder = RelativePositionBiasEncoder(attention_heads=4, hidden_dim=12)
    assert torch.count_nonzero(encoder.output_projection.weight) == 0
    assert torch.count_nonzero(encoder.output_projection.bias) == 0

    model = _model(RelativeGeometryQKVGraphTransformer)
    block = model.blocks[0]
    assert block.query_projection is not block.key_projection
    assert block.query_projection is not block.value_projection
    assert block.key_projection is not block.value_projection
    assert not any(isinstance(module, SharedEdgeEncoder) for module in model.modules())
    names = set(dict(model.named_modules()))
    assert not any(
        marker in name
        for name in names
        for marker in ("edge_key", "edge_value", "value_gate")
    )
    parameters = inspect.signature(model.forward).parameters
    assert "coordinates" not in parameters
    assert "relative_geometry" in parameters


def test_full_and_exact_receiver_chunked_outputs_gradients_and_explanations_match() -> None:
    expression, mask, metadata, edge_index, geometry = _inputs()
    torch.manual_seed(11)
    full = _model(RelativeGeometryQKVGraphTransformer)
    chunked = _model(
        ReceiverChunkedRelativeGeometryQKVGraphTransformer,
        checkpointing=True,
    )
    chunked.load_state_dict(full.state_dict(), strict=True)
    assert set(full.state_dict()) == set(chunked.state_dict())

    full_output = full(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=metadata,
        return_explanations=True,
    )
    chunked_output = chunked(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=metadata,
        return_explanations=True,
    )
    torch.testing.assert_close(chunked_output.prediction, full_output.prediction)
    torch.testing.assert_close(chunked_output.node_embedding, full_output.node_embedding)
    torch.testing.assert_close(
        chunked_output.attention_weights, full_output.attention_weights
    )
    torch.testing.assert_close(chunked_output.content_logits, full_output.content_logits)
    torch.testing.assert_close(chunked_output.positional_bias, full_output.positional_bias)
    torch.testing.assert_close(chunked_output.combined_logits, full_output.combined_logits)
    torch.testing.assert_close(
        chunked_output.content_logits + chunked_output.positional_bias,
        chunked_output.combined_logits,
    )
    assert torch.equal(chunked_output.edge_index, edge_index)

    full.zero_grad(set_to_none=True)
    chunked.zero_grad(set_to_none=True)
    full_output.prediction.square().mean().backward()
    chunked_output.prediction.square().mean().backward()
    for (full_name, full_parameter), (chunk_name, chunk_parameter) in zip(
        full.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert full_name == chunk_name
        torch.testing.assert_close(
            chunk_parameter.grad,
            full_parameter.grad,
            rtol=2e-5,
            atol=2e-6,
        )


def test_attention_normalizes_per_receiver_and_explanation_filter_stays_aligned() -> None:
    expression, mask, metadata, edge_index, geometry = _inputs()
    model = _model(ReceiverChunkedRelativeGeometryQKVGraphTransformer)
    output = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=metadata,
        return_explanations=True,
        attention_receivers=torch.tensor([1, 3]),
        explanation_layer=0,
    )
    expected_ids = torch.tensor([2, 3, 4, 6])
    assert torch.equal(output.edge_index, edge_index[:, expected_ids])
    assert torch.equal(output.source_indices, output.edge_index[0])
    assert torch.equal(output.receiver_indices, output.edge_index[1])
    assert output.layer_number == 0
    for receiver in (1, 3):
        selected = output.edge_index[1] == receiver
        torch.testing.assert_close(
            output.attention_weights[selected].sum(dim=0),
            torch.ones(4),
        )


def test_geometry_changes_logits_only_after_bias_projection_learns() -> None:
    expression, mask, metadata, edge_index, geometry = _inputs()
    model = _model(RelativeGeometryQKVGraphTransformer)
    model.eval()
    block = model.blocks[0]
    with torch.no_grad():
        block.relative_position_bias_encoder.output_projection.weight.fill_(0.1)
    first = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=metadata,
        return_explanations=True,
        explanation_layer=0,
    )
    changed = geometry.clone()
    changed[:, 0] += torch.linspace(0.0, 3.0, len(changed))
    second = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=changed,
        node_covariates=metadata,
        return_explanations=True,
        explanation_layer=0,
    )
    # Q/K/V depend only on node state; geometry is consumed only by the bias MLP.
    encoded = model.encoder(expression, mask, metadata)
    qkv_first = block.project_nodes(encoded)
    qkv_second = block.project_nodes(encoded)
    for left, right in zip(qkv_first, qkv_second, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert not torch.equal(first.positional_bias, second.positional_bias)
    assert not torch.equal(first.combined_logits, second.combined_logits)


def test_self_loops_are_rejected() -> None:
    expression, mask, metadata, edge_index, geometry = _inputs()
    bad_edges = edge_index.clone()
    bad_edges[:, 0] = 0
    for cls in (
        RelativeGeometryQKVGraphTransformer,
        ReceiverChunkedRelativeGeometryQKVGraphTransformer,
    ):
        with pytest.raises(ValueError, match="self loop"):
            _model(cls)(
                expression,
                mask,
                edge_index=bad_edges,
                relative_geometry=geometry,
                node_covariates=metadata,
            )


def test_edge_permutation_restores_explanations_to_caller_order() -> None:
    expression, mask, metadata, edge_index, geometry = _inputs()
    model = _model(ReceiverChunkedRelativeGeometryQKVGraphTransformer)
    model.eval()
    reference = model(
        expression,
        mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=metadata,
        return_explanations=True,
    )
    permutation = torch.tensor([6, 2, 0, 5, 4, 1, 3])
    permuted = model(
        expression,
        mask,
        edge_index=edge_index[:, permutation],
        relative_geometry=geometry[permutation],
        node_covariates=metadata,
        return_explanations=True,
    )
    torch.testing.assert_close(permuted.prediction, reference.prediction)
    assert torch.equal(permuted.edge_index, edge_index[:, permutation])
    inverse = torch.argsort(permutation)
    torch.testing.assert_close(
        permuted.attention_weights[inverse], reference.attention_weights
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_chunked_cuda_keeps_complete_graph_on_cpu_and_accumulates_high_degree_fp32() -> None:
    n_sources = 2048
    n_nodes = n_sources + 1
    model = ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=3,
        node_covariate_dim=1,
        hidden_dim=16,
        attention_heads=4,
        attention_head_dim=4,
        graph_layers=1,
        ffn_dim=32,
        decoder_dim=16,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=8,
        max_edges_per_chunk=128,
        activation_checkpointing=True,
    ).cuda()
    expression = torch.randn(n_nodes, 3, device="cuda")
    mask = torch.zeros(n_nodes, 3, dtype=torch.bool, device="cuda")
    metadata = torch.randn(n_nodes, 1, device="cuda")
    edges = torch.stack(
        [torch.arange(1, n_nodes), torch.zeros(n_sources, dtype=torch.long)]
    )
    geometry = torch.randn(n_sources, 70)
    with torch.autocast("cuda", dtype=torch.float16):
        output = model(
            expression,
            mask,
            edge_index=edges,
            relative_geometry=geometry,
            node_covariates=metadata,
            return_explanations=True,
            attention_receivers=torch.tensor([0]),
        )
    assert edges.device.type == "cpu"
    assert geometry.device.type == "cpu"
    assert output.prediction.device.type == "cuda"
    assert output.attention_weights.dtype == torch.float32
    assert torch.isfinite(output.attention_weights).all()
    torch.testing.assert_close(
        output.attention_weights.sum(dim=0),
        torch.ones(4, device="cuda"),
        rtol=2e-5,
        atol=2e-5,
    )
