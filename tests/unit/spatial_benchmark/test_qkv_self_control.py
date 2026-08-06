"""Semantic and parameter-use tests for the QKV single-cell control."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from spatial_benchmark.qkv_graph_transformer import (  # noqa: E402
    ReceiverChunkedEdgeAwareQKVGraphTransformer,
)
from spatial_benchmark.qkv_self_control import (  # noqa: E402
    QKVParameterMatchedSelfControl,
)


def _kwargs(edge_conditioning_mode: str) -> dict[str, object]:
    # H*d_h deliberately differs from hidden_dim to exercise every learned
    # projection rather than relying on a square attention representation.
    return {
        "num_genes": 9,
        "edge_attribute_dim": 6,
        "node_covariate_dim": 4,
        "hidden_dim": 18,
        "attention_heads": 3,
        "attention_head_dim": 7,
        "graph_layers": 2,
        "ffn_dim": 31,
        "decoder_dim": 23,
        "edge_hidden_dim": 13,
        "edge_embedding_dim": 8,
        "edge_conditioning_mode": edge_conditioning_mode,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "receiver_chunk_size": 2,
        "max_edges_per_chunk": 11,
        "activation_checkpointing": True,
    }


def _cell_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(4107)
    expression = torch.randn(7, 9, generator=generator)
    mask = torch.rand(7, 9, generator=generator) < 0.3
    covariates = torch.randn(7, 4, generator=generator)
    return expression, mask, covariates


def _trainable_count(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_constructor_surface_matches_receiver_chunked_graph_model() -> None:
    graph_parameters = [
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(
            ReceiverChunkedEdgeAwareQKVGraphTransformer
        ).parameters.items()
    ]
    control_parameters = [
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(
            QKVParameterMatchedSelfControl
        ).parameters.items()
    ]
    assert control_parameters == graph_parameters


@pytest.mark.parametrize(
    "edge_conditioning_mode", ["bias_gate", "vector"]
)
def test_parameter_layout_and_count_exactly_match_qkv_graph_model(
    edge_conditioning_mode: str,
) -> None:
    kwargs = _kwargs(edge_conditioning_mode)
    graph_model = ReceiverChunkedEdgeAwareQKVGraphTransformer(**kwargs)
    self_control = QKVParameterMatchedSelfControl(**kwargs)

    assert _trainable_count(self_control) == _trainable_count(graph_model)
    graph_state = graph_model.state_dict()
    control_state = self_control.state_dict()
    assert tuple(control_state) == tuple(graph_state)
    assert {
        name: tuple(value.shape) for name, value in control_state.items()
    } == {
        name: tuple(value.shape) for name, value in graph_state.items()
    }


@pytest.mark.parametrize(
    "edge_conditioning_mode", ["bias_gate", "vector"]
)
def test_predictions_are_exactly_invariant_to_graph_and_edge_values(
    edge_conditioning_mode: str,
) -> None:
    torch.manual_seed(823)
    model = QKVParameterMatchedSelfControl(
        **_kwargs(edge_conditioning_mode)
    ).eval()
    expression, mask, covariates = _cell_inputs()

    first = model(
        expression,
        mask,
        edge_index=torch.tensor([[0, 2, 4], [1, 3, 5]]),
        edge_attributes=torch.randn(3, 6),
        node_covariates=covariates,
        return_explanations=True,
    )
    second = model(
        expression,
        mask,
        edge_index=torch.tensor(
            [[6, 5, 4, 3, 2, 1], [0, 0, 0, 0, 0, 0]]
        ),
        edge_attributes=torch.randn(6, 6) * 1.0e6,
        node_covariates=covariates,
        return_explanations=True,
        attention_receivers=[0],
    )

    torch.testing.assert_close(
        first.prediction, second.prediction, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        first.node_embedding,
        second.node_embedding,
        rtol=0.0,
        atol=0.0,
    )
    for output in (first, second):
        assert output.edge_index is not None
        assert output.attention_weights is not None
        assert output.edge_embedding is not None
        assert output.edge_index.shape == (2, 0)
        assert output.attention_weights.shape == (0, 3)
        assert output.edge_embedding.shape == (0, 8)


@pytest.mark.parametrize(
    "edge_conditioning_mode", ["bias_gate", "vector"]
)
def test_every_trainable_parameter_has_a_finite_nonzero_gradient(
    edge_conditioning_mode: str,
) -> None:
    torch.manual_seed(1403)
    model = QKVParameterMatchedSelfControl(
        **_kwargs(edge_conditioning_mode)
    ).train()
    expression, mask, covariates = _cell_inputs()
    output = model(
        expression,
        mask,
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attributes=torch.randn(2, 6),
        node_covariates=covariates,
    )

    generator = torch.Generator().manual_seed(991)
    prediction_weights = torch.randn(
        output.prediction.shape, generator=generator
    )
    embedding_weights = torch.randn(
        output.node_embedding.shape, generator=generator
    )
    loss = (
        (output.prediction * prediction_weights).sum()
        + 0.37 * (output.node_embedding * embedding_weights).sum()
    )
    loss.backward()

    missing: list[str] = []
    nonfinite: list[str] = []
    numerically_zero: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        if not bool((parameter.grad != 0).any()):
            numerically_zero.append(name)

    assert not missing
    assert not nonfinite
    assert not numerically_zero


def test_target_selection_and_cell_chunking_do_not_change_predictions() -> None:
    kwargs = {
        **_kwargs("vector"),
        "activation_checkpointing": False,
    }
    torch.manual_seed(729)
    chunked = QKVParameterMatchedSelfControl(**kwargs).eval()
    unchunked = QKVParameterMatchedSelfControl(
        **{**kwargs, "receiver_chunk_size": 100}
    ).eval()
    unchunked.load_state_dict(chunked.state_dict())
    expression, mask, covariates = _cell_inputs()

    full = unchunked(
        expression,
        mask,
        node_covariates=covariates,
    )
    target_nodes = torch.tensor([5, 1, 6])
    selected = chunked(
        expression,
        mask,
        node_covariates=covariates,
        target_nodes=target_nodes,
    )

    torch.testing.assert_close(
        selected.prediction,
        full.prediction.index_select(0, target_nodes),
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        selected.node_embedding,
        full.node_embedding.index_select(0, target_nodes),
        rtol=2e-6,
        atol=2e-7,
    )
