"""Focused contracts for the additive multiscale hurdle-count model."""

from __future__ import annotations

import copy

import pytest


torch = pytest.importorskip("torch")

from spatial_benchmark.multiscale_hybrid import (  # noqa: E402
    MultiscaleAdditiveHybridModel,
    trainable_parameter_count,
)


def _graphs() -> dict[str, torch.Tensor]:
    local_edge_index = torch.tensor(
        [
            [0, 2, 1, 3, 0, 4, 1, 5, 2, 4, 3, 5],
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0],
        ],
        dtype=torch.long,
    )
    # Deliberately use a different, source-major regional order.
    regional_edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            [2, 4, 3, 5, 0, 4, 1, 5, 0, 2, 1, 3],
        ],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(913)
    return {
        "local_edge_index": local_edge_index,
        "local_edge_attributes": torch.randn(
            local_edge_index.shape[1], 5, generator=generator
        ),
        "regional_edge_index": regional_edge_index,
        "regional_edge_attributes": torch.randn(
            regional_edge_index.shape[1], 7, generator=generator
        ),
    }


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(431)
    counts = torch.randint(
        0,
        12,
        (6, 8),
        generator=generator,
        dtype=torch.int64,
    ).float()
    return {
        "input_expression": counts,
        "gene_mask": torch.rand(6, 8, generator=generator) < 0.3,
        "node_covariates": torch.randn(6, 3, generator=generator),
    }


def _kwargs() -> dict[str, object]:
    return {
        "num_genes": 8,
        "local_edge_attribute_dim": 5,
        "regional_edge_attribute_dim": 7,
        "expression_mean": torch.linspace(0.1, 0.8, 8),
        "expression_scale": torch.linspace(0.7, 1.4, 8),
        "node_covariate_dim": 3,
        "hidden_dim": 24,
        "decoder_dim": 19,
        "ffn_dim": 31,
        "attention_heads": 3,
        "attention_head_dim": 5,
        "value_head_dim": 4,
        "message_dim": 11,
        "edge_hidden_dim": 13,
        "edge_embedding_dim": 9,
        "output_channels": 2,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "receiver_chunk_size": 2,
        "activation_checkpointing": True,
    }


def _model(
    regional_routing: str,
    local_routing: str,
) -> MultiscaleAdditiveHybridModel:
    return MultiscaleAdditiveHybridModel(
        **_kwargs(),
        regional_routing=regional_routing,
        local_routing=local_routing,
    )


def _forward(
    model: MultiscaleAdditiveHybridModel,
    *,
    target_nodes: list[int] | None = None,
    selected_edges: list[int] | None = None,
    selected_genes: list[int] | None = None,
):
    graph_inputs: dict[str, torch.Tensor] = {}
    graphs = _graphs()
    if model.regional_routing != "surrogate":
        graph_inputs.update(
            {
                "regional_edge_index": graphs["regional_edge_index"],
                "regional_edge_attributes": graphs[
                    "regional_edge_attributes"
                ],
            }
        )
    if model.local_routing != "surrogate":
        graph_inputs.update(
            {
                "local_edge_index": graphs["local_edge_index"],
                "local_edge_attributes": graphs["local_edge_attributes"],
            }
        )
    if model.local_routing == "permuted":
        graph_inputs["local_source_index_by_node"] = torch.tensor(
            [3, 4, 5, 0, 1, 2],
            dtype=torch.long,
        )
    return model(
        **_inputs(),
        **graph_inputs,
        target_nodes=target_nodes,
        local_contribution_edge_indices=selected_edges,
        local_contribution_gene_indices=selected_genes,
    )


def test_four_routing_arms_are_exactly_parameter_identical() -> None:
    arms = [
        _model("surrogate", "surrogate"),
        _model("true", "surrogate"),
        _model("true", "true"),
        _model("true", "permuted"),
    ]
    expected_names = tuple(arms[0].state_dict())
    expected_shapes = tuple(
        value.shape for value in arms[0].state_dict().values()
    )
    expected_count = trainable_parameter_count(arms[0])
    for arm in arms[1:]:
        assert tuple(arm.state_dict()) == expected_names
        assert tuple(
            value.shape for value in arm.state_dict().values()
        ) == expected_shapes
        assert trainable_parameter_count(arm) == expected_count


def test_zero_initialized_context_branches_make_initial_predictor_self_only() -> None:
    torch.manual_seed(1007)
    source = _model("surrogate", "surrogate").eval()
    state = copy.deepcopy(source.state_dict())
    predictions = []
    for regional, local in (
        ("surrogate", "surrogate"),
        ("true", "surrogate"),
        ("true", "true"),
        ("true", "permuted"),
    ):
        arm = _model(regional, local).eval()
        arm.load_state_dict(state)
        output = _forward(arm)
        assert output.prediction.shape == (6, 8, 2)
        torch.testing.assert_close(
            output.regional_prediction,
            torch.zeros_like(output.regional_prediction),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output.local_prediction,
            torch.zeros_like(output.local_prediction),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            output.prediction,
            output.self_prediction,
            rtol=0,
            atol=0,
        )
        predictions.append(output.prediction)
    for prediction in predictions[1:]:
        torch.testing.assert_close(
            prediction, predictions[0], rtol=0, atol=0
        )


def test_hybrid_mask_is_authoritative_for_every_prediction_component() -> None:
    torch.manual_seed(117)
    model = _model("true", "true").eval()
    with torch.no_grad():
        model.regional_output_projection.weight.normal_()
        model.local_output_projection.weight.normal_()
    inputs = _inputs()
    changed = dict(inputs)
    changed_expression = inputs["input_expression"].clone()
    changed_expression[inputs["gene_mask"]] += 100
    changed["input_expression"] = changed_expression
    graphs = _graphs()

    original = model(**inputs, **graphs)
    altered = model(**changed, **graphs)
    for name in (
        "prediction",
        "node_embedding",
        "self_prediction",
        "regional_prediction",
        "local_prediction",
    ):
        torch.testing.assert_close(
            getattr(original, name),
            getattr(altered, name),
            rtol=0,
            atol=0,
        )


def test_prediction_is_exact_component_sum_and_target_selected() -> None:
    torch.manual_seed(991)
    model = _model("true", "true").eval()
    with torch.no_grad():
        model.regional_output_projection.weight.normal_()
        model.local_output_projection.weight.normal_()
    output = _forward(model, target_nodes=[4, 1, 5])
    assert output.prediction.shape == (3, 8, 2)
    torch.testing.assert_close(
        output.prediction,
        (
            output.self_prediction
            + output.regional_prediction
            + output.local_prediction
        ),
        rtol=0,
        atol=0,
    )


def test_permuted_routing_changes_only_local_source_state_gather() -> None:
    torch.manual_seed(823)
    true_model = _model("true", "true").eval()
    permuted_model = _model("true", "permuted").eval()
    permuted_model.load_state_dict(true_model.state_dict())
    with torch.no_grad():
        true_model.regional_output_projection.weight.normal_()
        true_model.local_output_projection.weight.normal_()
        permuted_model.load_state_dict(true_model.state_dict())

    graphs = _graphs()
    original_index = graphs["local_edge_index"].clone()
    original_attributes = graphs["local_edge_attributes"].clone()
    inputs = _inputs()
    permutation = torch.tensor([3, 4, 5, 0, 1, 2])
    true_output = true_model(**inputs, **graphs)
    permuted_output = permuted_model(
        **inputs,
        **graphs,
        local_source_index_by_node=permutation,
        local_contribution_edge_indices=[0, 1, 2],
    )

    for name in (
        "self_prediction",
        "regional_prediction",
    ):
        torch.testing.assert_close(
            getattr(true_output, name),
            getattr(permuted_output, name),
            rtol=0,
            atol=0,
        )
    assert not torch.allclose(
        true_output.local_prediction,
        permuted_output.local_prediction,
    )
    torch.testing.assert_close(
        permuted_output.prediction,
        (
            permuted_output.self_prediction
            + permuted_output.regional_prediction
            + permuted_output.local_prediction
        ),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        graphs["local_edge_index"],
        original_index,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        graphs["local_edge_attributes"],
        original_attributes,
        rtol=0,
        atol=0,
    )
    assert permuted_output.local_contribution_edge_index is not None
    torch.testing.assert_close(
        permuted_output.local_contribution_edge_index,
        original_index[:, :3],
        rtol=0,
        atol=0,
    )
    assert (
        permuted_output.local_contribution_effective_source_index
        is not None
    )
    torch.testing.assert_close(
        permuted_output.local_contribution_effective_source_index,
        permutation[original_index[0, :3]],
        rtol=0,
        atol=0,
    )
    assert true_output.local_routing == "true"
    assert permuted_output.local_routing == "permuted"


def test_permuted_routing_requires_a_complete_node_permutation() -> None:
    model = _model("true", "permuted")
    graphs = _graphs()
    with pytest.raises(ValueError, match="requires source_index_by_node"):
        model(**_inputs(), **graphs)
    with pytest.raises(ValueError, match="complete node permutation"):
        model(
            **_inputs(),
            **graphs,
            local_source_index_by_node=torch.zeros(6, dtype=torch.long),
        )


@pytest.mark.parametrize("mask_mode", ["whole_node", "partial_gene"])
def test_permuted_effective_self_slot_cannot_read_masked_receiver_values(
    mask_mode: str,
) -> None:
    """Amendment003's collision slot is bit-exact mask noninterference."""

    torch.manual_seed(1873)
    model = _model("true", "permuted").eval()
    with torch.no_grad():
        model.regional_output_projection.weight.normal_()
        model.local_output_projection.weight.normal_()
    inputs = _inputs()
    receiver = 1
    mask = torch.zeros_like(inputs["gene_mask"])
    if mask_mode == "whole_node":
        mask[receiver, :] = True
    else:
        mask[receiver, [2, 5]] = True
    inputs["gene_mask"] = mask
    changed = dict(inputs)
    changed_expression = inputs["input_expression"].clone()
    changed_expression[mask] += 100.0
    changed["input_expression"] = changed_expression

    graphs = _graphs()
    # Edge slot 0 is 0 -> 1; this permutation makes its effective source 1,
    # deliberately creating the audited effective-source == receiver case.
    permutation = torch.tensor([1, 0, 3, 2, 5, 4], dtype=torch.long)
    assert (
        permutation[graphs["local_edge_index"][0, 0]]
        == graphs["local_edge_index"][1, 0]
    )
    original = model(
        **inputs,
        **graphs,
        local_source_index_by_node=permutation,
        local_contribution_edge_indices=[0],
    )
    altered = model(
        **changed,
        **graphs,
        local_source_index_by_node=permutation,
        local_contribution_edge_indices=[0],
    )

    assert original.local_contribution_effective_source_index is not None
    assert int(
        original.local_contribution_effective_source_index[0].item()
    ) == receiver
    for name in (
        "prediction",
        "node_embedding",
        "self_prediction",
        "regional_prediction",
        "local_prediction",
        "local_attention_weights",
        "local_edge_messages",
        "local_contributions",
    ):
        assert torch.equal(getattr(original, name), getattr(altered, name))


def test_receiver_chunk_size_does_not_change_exact_graph_result() -> None:
    torch.manual_seed(349)
    chunked = _model("true", "true").eval()
    kwargs = _kwargs()
    kwargs["receiver_chunk_size"] = 64
    unchunked = MultiscaleAdditiveHybridModel(
        **kwargs,
        regional_routing="true",
        local_routing="true",
    ).eval()
    unchunked.load_state_dict(chunked.state_dict())
    with torch.no_grad():
        chunked.regional_output_projection.weight.normal_()
        chunked.local_output_projection.weight.normal_()
        unchunked.load_state_dict(chunked.state_dict())

    expected = _forward(unchunked)
    actual = _forward(chunked)
    for name in (
        "prediction",
        "regional_prediction",
        "local_prediction",
    ):
        torch.testing.assert_close(
            getattr(actual, name),
            getattr(expected, name),
            # Linear kernels may choose a different accumulation tile for a
            # two-receiver versus all-receiver batch.  Routing and segment
            # membership remain exact; only final-bit FP32 order can differ.
            rtol=2e-5,
            atol=5e-7,
        )


def test_selected_signed_edge_contributions_sum_to_local_prediction() -> None:
    torch.manual_seed(2201)
    model = _model("true", "true").eval()
    with torch.no_grad():
        model.local_output_projection.weight.normal_()
    graphs = _graphs()
    selected = list(range(graphs["local_edge_index"].shape[1]))
    output = _forward(model, selected_edges=selected)

    assert output.local_contributions is not None
    assert output.local_edge_messages is not None
    assert output.local_attention_weights is not None
    assert output.local_contribution_edge_index is not None
    assert output.local_contributions.shape == (len(selected), 8, 2)
    assert output.local_edge_messages.shape == (
        len(selected),
        _kwargs()["message_dim"],
    )
    reconstructed = torch.zeros_like(output.local_prediction)
    reconstructed.index_add_(
        0,
        output.local_contribution_edge_index[1],
        output.local_contributions,
    )
    torch.testing.assert_close(
        reconstructed,
        output.local_prediction,
        rtol=2e-6,
        atol=2e-7,
    )
    assert bool((output.local_contributions > 0).any())
    assert bool((output.local_contributions < 0).any())


def test_selected_contributions_preserve_requested_edge_and_gene_order() -> None:
    torch.manual_seed(88)
    model = _model("true", "true").eval()
    with torch.no_grad():
        model.local_output_projection.weight.normal_()
    selected_edges = [9, 1, 7]
    selected_genes = [6, 0, 3]
    output = _forward(
        model,
        selected_edges=selected_edges,
        selected_genes=selected_genes,
    )
    assert torch.equal(
        output.local_contribution_edge_indices,
        torch.tensor(selected_edges),
    )
    assert torch.equal(
        output.local_contribution_gene_indices,
        torch.tensor(selected_genes),
    )
    assert output.local_contributions is not None
    assert output.local_contributions.shape == (3, 3, 2)
    assert output.local_contribution_edge_index is not None
    torch.testing.assert_close(
        output.local_contribution_edge_index,
        _graphs()["local_edge_index"][:, selected_edges],
        rtol=0,
        atol=0,
    )


def test_checkpointed_backward_reaches_every_parameter_with_finite_gradient() -> None:
    torch.manual_seed(773)
    model = _model("true", "true").train()
    output = _forward(model, target_nodes=[0, 2, 5])
    output.prediction.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


def test_routing_inputs_fail_closed() -> None:
    inputs = _inputs()
    graphs = _graphs()
    surrogate = _model("surrogate", "surrogate")
    with pytest.raises(ValueError, match="surrogate routing prohibits"):
        surrogate(
            **inputs,
            regional_edge_index=graphs["regional_edge_index"],
            regional_edge_attributes=graphs["regional_edge_attributes"],
        )

    true_model = _model("true", "true")
    with pytest.raises(ValueError, match="requires edge_index"):
        true_model(**inputs)
    with pytest.raises(ValueError, match="genes require selected local edges"):
        true_model(
            **inputs,
            **graphs,
            local_contribution_gene_indices=[0],
        )
    with pytest.raises(ValueError, match="duplicate"):
        true_model(
            **inputs,
            **graphs,
            local_contribution_edge_indices=[1, 1],
        )
