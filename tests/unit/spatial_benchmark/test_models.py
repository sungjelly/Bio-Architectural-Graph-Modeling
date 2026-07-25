"""Synthetic contract tests for the nested masked-expression model ladder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.models import (  # noqa: E402
    AdditiveEdgeMessageModel,
    BroadSpatialFieldControl,
    EdgeConditionedGATv2,
    MeanNeighborModel,
    ModelOutput,
    ParameterMatchedSelfControl,
    SelfOnlyMLP,
    TopologyGATv2,
    mean_incoming_neighbors,
)


NUM_NODES = 5
NUM_GENES = 7
NUM_COVARIATES = 3
EDGE_ATTRIBUTE_DIM = 5
HIDDEN_DIM = 16


@pytest.fixture
def synthetic_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(314159)
    expression = torch.randn(
        NUM_NODES, NUM_GENES, generator=generator
    )
    gene_mask = (
        torch.rand(NUM_NODES, NUM_GENES, generator=generator) < 0.35
    )
    covariates = torch.randn(
        NUM_NODES, NUM_COVARIATES, generator=generator
    )
    # Directed storage of an undirected five-node cycle.  Convention is
    # source in row zero and receiver in row one.
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
        ],
        dtype=torch.long,
    )
    edge_attributes = torch.randn(
        edge_index.shape[1], EDGE_ATTRIBUTE_DIM, generator=generator
    )
    return {
        "input_expression": expression,
        "gene_mask": gene_mask,
        "node_covariates": covariates,
        "edge_index": edge_index,
        "edge_attributes": edge_attributes,
    }


def _common_kwargs() -> dict[str, int | float]:
    return {
        "num_genes": NUM_GENES,
        "node_covariate_dim": NUM_COVARIATES,
        "hidden_dim": HIDDEN_DIM,
        "ffn_dim": 24,
        "decoder_dim": 20,
        "dropout": 0.0,
    }


def _make_models() -> list[torch.nn.Module]:
    common = _common_kwargs()
    return [
        SelfOnlyMLP(**common),
        ParameterMatchedSelfControl(
            **common,
            attention_heads=4,
            graph_layers=1,
            attention_dropout=0.0,
        ),
        MeanNeighborModel(**common),
        TopologyGATv2(
            **common,
            attention_heads=4,
            graph_layers=1,
            attention_dropout=0.0,
        ),
        EdgeConditionedGATv2(
            **common,
            edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
            edge_embedding_dim=6,
            edge_hidden_dim=9,
            attention_heads=4,
            graph_layers=1,
            attention_dropout=0.0,
        ),
        AdditiveEdgeMessageModel(
            **common,
            edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
            edge_embedding_dim=6,
            edge_hidden_dim=9,
            attention_heads=4,
            message_head_dim=3,
            message_dim=8,
            attention_dropout=0.0,
        ),
    ]


def _forward_kwargs(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    explanations: bool = True,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "input_expression": inputs["input_expression"],
        "gene_mask": inputs["gene_mask"],
        "node_covariates": inputs["node_covariates"],
        "return_explanations": explanations,
    }
    if not isinstance(
        model, (SelfOnlyMLP, ParameterMatchedSelfControl)
    ):
        kwargs["edge_index"] = inputs["edge_index"]
    if isinstance(
        model, (EdgeConditionedGATv2, AdditiveEdgeMessageModel)
    ):
        kwargs["edge_attributes"] = inputs["edge_attributes"]
    return kwargs


def test_all_models_share_shape_stable_forward_contract(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    for model in _make_models():
        model.eval()
        assert not any(
            isinstance(module, torch.nn.Embedding)
            for module in model.modules()
        )
        output = model(**_forward_kwargs(model, synthetic_inputs))
        assert isinstance(output, ModelOutput)
        assert output.prediction.shape == (NUM_NODES, NUM_GENES)
        assert output.node_embedding.shape == (NUM_NODES, HIDDEN_DIM)
        assert torch.isfinite(output.prediction).all()


def test_encoder_applies_mask_internally_and_keeps_metadata_visible(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    torch.manual_seed(2)
    model = SelfOnlyMLP(**_common_kwargs()).eval()
    original = synthetic_inputs["input_expression"].clone()
    changed_hidden_values = original.clone()
    mask = synthetic_inputs["gene_mask"]
    changed_hidden_values[mask] += 10_000.0

    first = model(
        original,
        mask,
        node_covariates=synthetic_inputs["node_covariates"],
    ).prediction
    second = model(
        changed_hidden_values,
        mask,
        node_covariates=synthetic_inputs["node_covariates"],
    ).prediction
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    nonfinite_hidden_values = original.clone()
    nonfinite_hidden_values[mask] = torch.nan
    with_nonfinite_hidden_values = model(
        nonfinite_hidden_values,
        mask,
        node_covariates=synthetic_inputs["node_covariates"],
    ).prediction
    torch.testing.assert_close(
        first, with_nonfinite_hidden_values, rtol=0.0, atol=0.0
    )

    zero_expression = torch.zeros_like(original)
    no_mask = torch.zeros_like(mask)
    one_explicit_mask = no_mask.clone()
    one_explicit_mask[:, 0] = True
    without_mask_channel = model(
        zero_expression,
        no_mask,
        node_covariates=synthetic_inputs["node_covariates"],
    ).prediction
    with_mask_channel = model(
        zero_expression,
        one_explicit_mask,
        node_covariates=synthetic_inputs["node_covariates"],
    ).prediction
    assert not torch.allclose(without_mask_channel, with_mask_channel)

    fully_masked = torch.ones_like(mask)
    shifted_metadata = synthetic_inputs["node_covariates"].clone()
    shifted_metadata[0] += 5.0
    with_original_metadata = model(
        original,
        fully_masked,
        node_covariates=synthetic_inputs["node_covariates"],
    ).prediction
    with_shifted_metadata = model(
        original,
        fully_masked,
        node_covariates=shifted_metadata,
    ).prediction
    assert not torch.allclose(
        with_original_metadata[0], with_shifted_metadata[0]
    )


def test_covariates_cannot_be_silently_omitted_or_ignored(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    model_requiring_metadata = SelfOnlyMLP(**_common_kwargs())
    with pytest.raises(ValueError, match="node_covariates are required"):
        model_requiring_metadata(
            synthetic_inputs["input_expression"],
            synthetic_inputs["gene_mask"],
        )

    model_without_metadata = SelfOnlyMLP(
        num_genes=NUM_GENES,
        node_covariate_dim=0,
        hidden_dim=HIDDEN_DIM,
    )
    with pytest.raises(ValueError, match="node_covariates were supplied"):
        model_without_metadata(
            synthetic_inputs["input_expression"],
            synthetic_inputs["gene_mask"],
            node_covariates=synthetic_inputs["node_covariates"],
        )


def test_broad_field_is_explicit_train_fitted_and_graph_independent(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    model = BroadSpatialFieldControl(**_common_kwargs()).eval()
    coordinates = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
        ]
    )
    with pytest.raises(RuntimeError, match="fit_coordinate_basis"):
        model(
            synthetic_inputs["input_expression"],
            synthetic_inputs["gene_mask"],
            node_covariates=synthetic_inputs["node_covariates"],
            coordinates_um=coordinates,
        )
    provenance = model.fit_coordinate_basis(coordinates)
    assert provenance["fit_scope"] == "training split coordinates only"
    assert provenance["contains_cell_or_region_ids"] is False
    assert provenance["contains_graph_or_neighbor_features"] is False
    assert not any(
        isinstance(module, torch.nn.Embedding) for module in model.modules()
    )

    original = model(
        synthetic_inputs["input_expression"],
        synthetic_inputs["gene_mask"],
        edge_index=synthetic_inputs["edge_index"],
        edge_attributes=synthetic_inputs["edge_attributes"],
        node_covariates=synthetic_inputs["node_covariates"],
        coordinates_um=coordinates,
    ).prediction
    changed_hidden = synthetic_inputs["input_expression"].clone()
    changed_hidden[synthetic_inputs["gene_mask"]] += 100_000.0
    reversed_edges = synthetic_inputs["edge_index"].flip(1)
    reversed_attributes = synthetic_inputs["edge_attributes"].flip(0)
    same_control_inputs = model(
        changed_hidden,
        synthetic_inputs["gene_mask"],
        edge_index=reversed_edges,
        edge_attributes=reversed_attributes,
        node_covariates=synthetic_inputs["node_covariates"],
        coordinates_um=coordinates,
    ).prediction
    torch.testing.assert_close(
        original, same_control_inputs, rtol=0.0, atol=0.0
    )

    moved_coordinates = coordinates.clone()
    moved_coordinates[0] += torch.tensor([7.0, -3.0])
    moved = model(
        synthetic_inputs["input_expression"],
        synthetic_inputs["gene_mask"],
        node_covariates=synthetic_inputs["node_covariates"],
        coordinates_um=moved_coordinates,
    ).prediction
    assert not torch.allclose(original[0], moved[0])

    ordinary = SelfOnlyMLP(**_common_kwargs()).eval()
    with pytest.raises(TypeError, match="coordinates_um"):
        ordinary(
            synthetic_inputs["input_expression"],
            synthetic_inputs["gene_mask"],
            node_covariates=synthetic_inputs["node_covariates"],
            coordinates_um=coordinates,
        )


def test_mean_neighbor_aggregation_is_exact_and_isolates_are_zero() -> None:
    embedding = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [7.0, 8.0], [11.0, 12.0]]
    )
    # Sources 0 and 2 point to receiver 1; source 1 points to receiver 2.
    edges = torch.tensor([[0, 2, 1], [1, 1, 2]])
    result = mean_incoming_neighbors(embedding, edges)
    expected = torch.tensor(
        [[0.0, 0.0], [4.0, 5.0], [3.0, 4.0], [0.0, 0.0]]
    )
    torch.testing.assert_close(result, expected, rtol=0.0, atol=0.0)


def test_parameter_matched_control_exactly_matches_g1_parameter_count() -> None:
    kwargs = {
        **_common_kwargs(),
        "attention_heads": 4,
        "graph_layers": 2,
        "attention_dropout": 0.0,
    }
    self_control = ParameterMatchedSelfControl(**kwargs)
    graph_model = TopologyGATv2(**kwargs)
    self_count = sum(parameter.numel() for parameter in self_control.parameters())
    graph_count = sum(parameter.numel() for parameter in graph_model.parameters())
    assert self_count == graph_count


@pytest.mark.parametrize("hidden_dim", [128, 256, 512])
def test_screened_hidden_dimensions_are_supported(hidden_dim: int) -> None:
    model = TopologyGATv2(
        num_genes=3,
        node_covariate_dim=2,
        hidden_dim=hidden_dim,
        attention_heads=4,
        dropout=0.0,
        attention_dropout=0.0,
    ).eval()
    output = model(
        torch.randn(3, 3),
        torch.zeros(3, 3, dtype=torch.bool),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        node_covariates=torch.randn(3, 2),
    )
    assert output.node_embedding.shape == (3, hidden_dim)


def test_explicit_head_dimension_can_differ_from_hidden_partition() -> None:
    kwargs = {
        "num_genes": 3,
        "node_covariate_dim": 2,
        "hidden_dim": 18,
        "attention_heads": 4,
        "attention_head_dim": 5,
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    graph_model = TopologyGATv2(**kwargs).eval()
    self_control = ParameterMatchedSelfControl(**kwargs).eval()
    output = graph_model(
        torch.randn(3, 3),
        torch.zeros(3, 3, dtype=torch.bool),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        node_covariates=torch.randn(3, 2),
    )
    assert output.node_embedding.shape == (3, 18)
    assert sum(
        parameter.numel() for parameter in graph_model.parameters()
    ) == sum(parameter.numel() for parameter in self_control.parameters())


def test_attention_and_edge_explanations_align_with_supplied_edges(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    models = [
        TopologyGATv2(
            **_common_kwargs(),
            attention_heads=4,
            attention_dropout=0.0,
        ),
        EdgeConditionedGATv2(
            **_common_kwargs(),
            edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
            edge_embedding_dim=6,
            attention_heads=4,
            attention_dropout=0.0,
        ),
        AdditiveEdgeMessageModel(
            **_common_kwargs(),
            edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
            edge_embedding_dim=6,
            attention_heads=4,
            message_head_dim=3,
            message_dim=8,
            attention_dropout=0.0,
        ),
    ]
    for model in models:
        model.eval()
        output = model(**_forward_kwargs(model, synthetic_inputs))
        assert torch.equal(output.edge_index, synthetic_inputs["edge_index"])
        assert output.attention_weights.shape == (
            synthetic_inputs["edge_index"].shape[1],
            4,
        )
        receiver = output.edge_index[1]
        for node in range(NUM_NODES):
            incoming = receiver == node
            if incoming.any():
                torch.testing.assert_close(
                    output.attention_weights[incoming].sum(dim=0),
                    torch.ones(4),
                    rtol=1e-5,
                    atol=1e-6,
                )


def test_graph_models_reject_self_loops(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    model = MeanNeighborModel(**_common_kwargs())
    self_loop = torch.tensor([[0], [0]])
    with pytest.raises(ValueError, match="self loops are prohibited"):
        model(
            synthetic_inputs["input_expression"],
            synthetic_inputs["gene_mask"],
            edge_index=self_loop,
            node_covariates=synthetic_inputs["node_covariates"],
        )


def test_no_edge_graphs_are_finite_with_zero_neighbor_branch(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    models = [
        MeanNeighborModel(**_common_kwargs()),
        TopologyGATv2(
            **_common_kwargs(),
            attention_heads=4,
            attention_dropout=0.0,
        ),
        EdgeConditionedGATv2(
            **_common_kwargs(),
            edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
            edge_embedding_dim=6,
            attention_heads=4,
            attention_dropout=0.0,
        ),
        AdditiveEdgeMessageModel(
            **_common_kwargs(),
            edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
            edge_embedding_dim=6,
            attention_heads=4,
            message_head_dim=3,
            message_dim=8,
            attention_dropout=0.0,
        ),
    ]
    for model in models:
        model.eval()
        kwargs: dict[str, object] = {
            "input_expression": synthetic_inputs["input_expression"],
            "gene_mask": synthetic_inputs["gene_mask"],
            "edge_index": empty_edges,
            "node_covariates": synthetic_inputs["node_covariates"],
            "return_explanations": True,
        }
        # G2 and G3 deliberately accept None for attributes on an empty graph.
        output = model(**kwargs)
        assert torch.isfinite(output.prediction).all()
        if output.attention_weights is not None:
            assert output.attention_weights.shape[0] == 0
        if isinstance(model, AdditiveEdgeMessageModel):
            assert torch.count_nonzero(output.neighbor_prediction) == 0
            assert output.edge_message.shape == (0, model.message_dim)
            torch.testing.assert_close(
                output.prediction,
                output.self_prediction,
                rtol=0.0,
                atol=0.0,
            )


def test_gradients_reach_expression_metadata_and_graph_parameters(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    for model in _make_models():
        model.train()
        output = model(
            **_forward_kwargs(model, synthetic_inputs, explanations=False)
        )
        loss = output.prediction.square().mean()
        loss.backward()

        expression_gradient = (
            model.encoder.expression_projection.weight.grad
        )
        metadata_gradient = model.encoder.covariate_projection.weight.grad
        assert expression_gradient is not None
        assert metadata_gradient is not None
        assert torch.isfinite(expression_gradient).all()
        assert torch.isfinite(metadata_gradient).all()
        assert expression_gradient.abs().sum() > 0
        assert metadata_gradient.abs().sum() > 0

        if isinstance(model, TopologyGATv2):
            graph_gradient = model.blocks[0].convolution.att.grad
            assert graph_gradient is not None
            assert torch.isfinite(graph_gradient).all()
            assert graph_gradient.abs().sum() > 0
        elif isinstance(model, EdgeConditionedGATv2):
            graph_gradient = model.edge_encoder.network[0].weight.grad
            assert graph_gradient is not None
            assert torch.isfinite(graph_gradient).all()
            assert graph_gradient.abs().sum() > 0
        elif isinstance(model, AdditiveEdgeMessageModel):
            graph_gradient = model.attention_vector.grad
            assert graph_gradient is not None
            assert torch.isfinite(graph_gradient).all()
            assert graph_gradient.abs().sum() > 0


def test_eval_forward_is_deterministic(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    torch.manual_seed(99)
    model = AdditiveEdgeMessageModel(
        **_common_kwargs(),
        edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
        edge_embedding_dim=6,
        attention_heads=4,
        message_head_dim=3,
        message_dim=8,
        attention_dropout=0.25,
    ).eval()
    kwargs = _forward_kwargs(model, synthetic_inputs)
    first = model(**kwargs)
    second = model(**kwargs)
    torch.testing.assert_close(
        first.prediction, second.prediction, rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        first.attention_weights,
        second.attention_weights,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        first.edge_message, second.edge_message, rtol=0.0, atol=0.0
    )


def test_target_nodes_match_full_graph_predictions(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    model = EdgeConditionedGATv2(
        **_common_kwargs(),
        edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
        edge_embedding_dim=6,
        attention_heads=4,
        attention_dropout=0.0,
    ).eval()
    kwargs = _forward_kwargs(model, synthetic_inputs)
    full = model(**kwargs)
    target_nodes = torch.tensor([4, 1, 3])
    selected = model(**kwargs, target_nodes=target_nodes)
    torch.testing.assert_close(
        selected.prediction,
        full.prediction.index_select(0, target_nodes),
        # The decoder sees a three-row GEMM instead of the full five-row GEMM,
        # so CPU backends may differ by one floating-point accumulation bit.
        rtol=1e-6,
        atol=1e-7,
    )
    assert selected.node_embedding.shape == (3, HIDDEN_DIM)


def test_g3_has_exact_additive_output_and_edge_decomposition(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    model = AdditiveEdgeMessageModel(
        **_common_kwargs(),
        edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
        edge_embedding_dim=6,
        attention_heads=4,
        message_head_dim=3,
        message_dim=8,
        attention_dropout=0.0,
    ).eval()
    output = model(**_forward_kwargs(model, synthetic_inputs))

    assert torch.equal(
        output.prediction,
        output.self_prediction + output.neighbor_prediction,
    )
    contributions = model.edge_gene_contributions(output.edge_message)
    assert contributions.shape == (
        synthetic_inputs["edge_index"].shape[1],
        NUM_GENES,
    )
    summed_contributions = torch.zeros_like(output.neighbor_prediction)
    summed_contributions.index_add_(
        0, output.edge_index[1], contributions
    )
    torch.testing.assert_close(
        summed_contributions,
        output.neighbor_prediction,
        rtol=1e-5,
        atol=1e-6,
    )

    selected = model.edge_gene_contributions(
        output.edge_message,
        edge_indices=[1, 3],
        gene_indices=[0, 5],
    )
    torch.testing.assert_close(
        selected,
        contributions.index_select(0, torch.tensor([1, 3])).index_select(
            1, torch.tensor([0, 5])
        ),
    )


def test_g3_can_copy_and_freeze_the_b0_self_branch(
    synthetic_inputs: dict[str, torch.Tensor],
) -> None:
    torch.manual_seed(123)
    b0 = SelfOnlyMLP(**_common_kwargs()).eval()
    g3 = AdditiveEdgeMessageModel(
        **_common_kwargs(),
        edge_attribute_dim=EDGE_ATTRIBUTE_DIM,
        edge_embedding_dim=6,
        attention_heads=4,
        message_head_dim=3,
        message_dim=8,
        attention_dropout=0.0,
    ).eval()
    g3.copy_self_branch_from(b0)
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    b0_prediction = b0(
        synthetic_inputs["input_expression"],
        synthetic_inputs["gene_mask"],
        node_covariates=synthetic_inputs["node_covariates"],
    ).prediction
    g3_prediction = g3(
        synthetic_inputs["input_expression"],
        synthetic_inputs["gene_mask"],
        edge_index=empty_edges,
        node_covariates=synthetic_inputs["node_covariates"],
    ).prediction
    torch.testing.assert_close(
        b0_prediction, g3_prediction, rtol=0.0, atol=0.0
    )

    g3.set_self_branch_trainable(False)
    assert all(
        not parameter.requires_grad
        for module in (g3.encoder, g3.self_block, g3.self_decoder)
        for parameter in module.parameters()
    )
    g3.set_self_branch_trainable(True)
    assert all(
        parameter.requires_grad
        for module in (g3.encoder, g3.self_block, g3.self_decoder)
        for parameter in module.parameters()
    )
