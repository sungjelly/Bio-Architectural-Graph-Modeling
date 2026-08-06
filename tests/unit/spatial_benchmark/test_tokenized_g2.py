"""Tests for categorical count-token G2 execution."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from spatial_benchmark.tokenized_g2 import (  # noqa: E402
    CategoricalExpressionDecoder,
    CategoricalNodeEncoder,
    TokenizedReceiverChunkedEdgeConditionedGATv2,
)


def _inputs() -> dict[str, torch.Tensor]:
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 0, 4],
        ],
        dtype=torch.long,
    )
    return {
        "input_expression": torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2],
                [1, 2, 3, 0, 1, 2, 3],
                [2, 3, 0, 1, 2, 3, 0],
                [3, 0, 1, 2, 3, 0, 1],
                [0, 2, 1, 3, 2, 0, 3],
                [3, 1, 2, 0, 1, 3, 2],
            ],
            dtype=torch.float32,
        ),
        "gene_mask": torch.tensor(
            [
                [False, True, False, False, False, False, False],
                [False, False, False, True, False, False, False],
                [False, False, True, False, False, False, False],
                [True, False, False, False, False, False, False],
                [False, False, False, False, True, False, False],
                [False, False, False, False, False, True, False],
            ]
        ),
        "node_covariates": torch.arange(
            18, dtype=torch.float32
        ).reshape(6, 3)
        / 10,
        "edge_index": edge_index,
        "edge_attributes": torch.arange(
            edge_index.shape[1] * 5, dtype=torch.float32
        ).reshape(edge_index.shape[1], 5)
        / 10,
    }


def _model(
    *,
    receiver_chunk_size: int = 2,
    activation_checkpointing: bool = False,
) -> TokenizedReceiverChunkedEdgeConditionedGATv2:
    return TokenizedReceiverChunkedEdgeConditionedGATv2(
        num_genes=7,
        edge_attribute_dim=5,
        node_covariate_dim=3,
        hidden_dim=16,
        attention_heads=4,
        graph_layers=2,
        ffn_dim=23,
        decoder_dim=19,
        edge_hidden_dim=11,
        edge_embedding_dim=6,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=receiver_chunk_size,
        activation_checkpointing=activation_checkpointing,
    )


def test_encoder_and_decoder_shapes() -> None:
    inputs = _inputs()
    encoder = CategoricalNodeEncoder(
        num_genes=7,
        node_covariate_dim=3,
        hidden_dim=16,
        dropout=0.0,
    )
    decoder = CategoricalExpressionDecoder(
        hidden_dim=16,
        num_genes=7,
        decoder_dim=19,
        dropout=0.0,
    )

    embedding = encoder(
        inputs["input_expression"],
        inputs["gene_mask"],
        inputs["node_covariates"],
    )
    assert embedding.shape == (6, 16)
    assert decoder(embedding).shape == (6, 7, 4)

    output = _model()(**inputs, target_nodes=[4, 1])
    assert output.prediction.shape == (2, 7, 4)
    assert output.node_embedding.shape == (2, 16)


def test_explicit_mask_makes_hidden_token_values_invariant() -> None:
    inputs = _inputs()
    changed = inputs["input_expression"].clone()
    changed[inputs["gene_mask"]] = (
        changed[inputs["gene_mask"]] + 2
    ) % 4
    encoder = CategoricalNodeEncoder(
        num_genes=7,
        node_covariate_dim=3,
        hidden_dim=16,
        dropout=0.0,
    ).eval()

    expected = encoder(
        inputs["input_expression"],
        inputs["gene_mask"],
        inputs["node_covariates"],
    )
    actual = encoder(
        changed,
        inputs["gene_mask"],
        inputs["node_covariates"],
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (1.5, ValueError, "integer-valued"),
        (-1.0, ValueError, r"range \[0, 4\]"),
        (5.0, ValueError, r"range \[0, 4\]"),
        (float("nan"), ValueError, "finite"),
    ],
)
def test_invalid_token_values_are_rejected(
    value: float,
    error: type[Exception],
    message: str,
) -> None:
    inputs = _inputs()
    inputs["input_expression"][0, 0] = value
    with pytest.raises(error, match=message):
        _model()(**inputs)


def test_invalid_token_and_mask_types_are_rejected() -> None:
    inputs = _inputs()
    with pytest.raises(TypeError, match="float32 categorical"):
        _model()(
            **{
                **inputs,
                "input_expression": inputs["input_expression"].long(),
            }
        )
    with pytest.raises(TypeError, match="boolean tensor"):
        _model()(
            **{
                **inputs,
                "gene_mask": inputs["gene_mask"].float(),
            }
        )


def test_checkpointed_backward_reaches_all_model_components() -> None:
    model = _model(activation_checkpointing=True).train()
    logits = model(**_inputs()).prediction
    logits.square().mean().backward()

    assert model.decoder.linear_out.weight.grad is not None
    assert model.edge_encoder.network[0].weight.grad is not None
    for projection in model.encoder.token_projections:
        assert projection.weight.grad is not None


def test_receiver_chunk_size_does_not_change_exact_execution() -> None:
    torch.manual_seed(1234)
    fine = _model(receiver_chunk_size=1).eval()
    coarse = _model(receiver_chunk_size=6).eval()
    coarse.load_state_dict(fine.state_dict())
    inputs = _inputs()

    with torch.no_grad():
        expected = coarse(**inputs, return_explanations=True)
        actual = fine(**inputs, return_explanations=True)

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
