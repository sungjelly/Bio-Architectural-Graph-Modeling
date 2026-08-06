"""Equivalence and memory-contract tests for receiver-chunked G2."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from spatial_benchmark.dense_gat import (  # noqa: E402
    ReceiverChunkedEdgeConditionedGATv2,
)
from spatial_benchmark.models import EdgeConditionedGATv2  # noqa: E402


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(901)
    # Deliberately source-major rather than receiver-major, matching the
    # repository graph builder and exercising the cached stable permutation.
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 0, 4],
        ],
        dtype=torch.long,
    )
    return {
        "input_expression": torch.randn(6, 7, generator=generator),
        "gene_mask": torch.rand(6, 7, generator=generator) < 0.3,
        "node_covariates": torch.randn(6, 3, generator=generator),
        "edge_index": edge_index,
        "edge_attributes": torch.randn(
            edge_index.shape[1], 5, generator=generator
        ),
    }


def _model_kwargs() -> dict[str, int | float]:
    return {
        "num_genes": 7,
        "edge_attribute_dim": 5,
        "node_covariate_dim": 3,
        "hidden_dim": 16,
        "attention_heads": 4,
        "graph_layers": 2,
        "ffn_dim": 23,
        "decoder_dim": 19,
        "edge_hidden_dim": 11,
        "edge_embedding_dim": 6,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }


def _paired_models(
    *, activation_checkpointing: bool
) -> tuple[
    EdgeConditionedGATv2, ReceiverChunkedEdgeConditionedGATv2
]:
    torch.manual_seed(12345)
    reference = EdgeConditionedGATv2(**_model_kwargs())
    chunked = ReceiverChunkedEdgeConditionedGATv2(
        **_model_kwargs(),
        receiver_chunk_size=2,
        activation_checkpointing=activation_checkpointing,
    )
    chunked.load_state_dict(reference.state_dict())
    assert tuple(chunked.state_dict()) == tuple(reference.state_dict())
    return reference, chunked


def test_two_layer_forward_and_full_explanations_match_ordinary_g2() -> None:
    inputs = _inputs()
    reference, chunked = _paired_models(activation_checkpointing=False)
    reference.eval()
    chunked.eval()

    expected = reference(**inputs, return_explanations=True)
    actual = chunked(**inputs, return_explanations=True)

    torch.testing.assert_close(
        actual.prediction, expected.prediction, rtol=2e-6, atol=2e-7
    )
    torch.testing.assert_close(
        actual.node_embedding,
        expected.node_embedding,
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        actual.attention_weights,
        expected.attention_weights,
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        actual.edge_embedding,
        expected.edge_embedding,
        rtol=2e-6,
        atol=2e-7,
    )
    assert torch.equal(actual.edge_index, inputs["edge_index"])


def test_checkpointed_backward_matches_ordinary_g2_gradients() -> None:
    inputs = _inputs()
    reference, chunked = _paired_models(activation_checkpointing=True)
    reference.train()
    chunked.train()

    expected = reference(**inputs).prediction
    actual = chunked(**inputs).prediction
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)

    expected.square().mean().backward()
    actual.square().mean().backward()
    reference_parameters = dict(reference.named_parameters())
    chunked_parameters = dict(chunked.named_parameters())
    assert chunked_parameters.keys() == reference_parameters.keys()
    for name, expected_parameter in reference_parameters.items():
        expected_gradient = expected_parameter.grad
        actual_gradient = chunked_parameters[name].grad
        assert expected_gradient is not None, name
        assert actual_gradient is not None, name
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=2e-5,
            atol=2e-7,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_selected_receiver_attention_is_aligned_to_original_edges() -> None:
    inputs = _inputs()
    reference, chunked = _paired_models(activation_checkpointing=False)
    reference.eval()
    chunked.eval()

    expected = reference(**inputs, return_explanations=True)
    selected_receivers = torch.tensor([4, 1])
    actual = chunked(
        **inputs,
        return_explanations=True,
        attention_receivers=selected_receivers,
    )
    selected_edges = torch.isin(
        inputs["edge_index"][1], selected_receivers
    )

    torch.testing.assert_close(
        actual.prediction, expected.prediction, rtol=2e-6, atol=2e-7
    )
    assert torch.equal(
        actual.edge_index, inputs["edge_index"][:, selected_edges]
    )
    torch.testing.assert_close(
        actual.attention_weights,
        expected.attention_weights[selected_edges],
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        actual.edge_embedding,
        expected.edge_embedding[selected_edges],
        rtol=2e-6,
        atol=2e-7,
    )


def test_edge_encoder_never_receives_the_full_edge_set() -> None:
    inputs = _inputs()
    _, chunked = _paired_models(activation_checkpointing=False)
    chunked.eval()
    encoded_edge_counts: list[int] = []

    def record_edge_count(
        _module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
    ) -> None:
        encoded_edge_counts.append(arguments[0].shape[0])

    handle = chunked.edge_encoder.register_forward_pre_hook(
        record_edge_count
    )
    try:
        with torch.no_grad():
            chunked(**inputs)
    finally:
        handle.remove()

    assert encoded_edge_counts
    assert max(encoded_edge_counts) < inputs["edge_index"].shape[1]
    assert sum(encoded_edge_counts) == (
        len(chunked.blocks) * inputs["edge_index"].shape[1]
    )


def test_campaign_width_depth_and_edge_embedding_configuration_runs() -> None:
    model = ReceiverChunkedEdgeConditionedGATv2(
        num_genes=7,
        edge_attribute_dim=5,
        node_covariate_dim=3,
        hidden_dim=512,
        attention_heads=4,
        graph_layers=2,
        edge_hidden_dim=64,
        edge_embedding_dim=64,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=2,
        activation_checkpointing=True,
    ).eval()
    with torch.no_grad():
        output = model(**_inputs())
    assert output.prediction.shape == (6, 7)
    assert output.node_embedding.shape == (6, 512)


def test_attention_receiver_selection_requires_explanations() -> None:
    _, chunked = _paired_models(activation_checkpointing=False)
    with pytest.raises(
        ValueError, match="requires return_explanations=True"
    ):
        chunked(**_inputs(), attention_receivers=[1])
