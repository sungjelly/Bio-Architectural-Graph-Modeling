"""Contracts for target-isolated, static-clean-source NB2 attention."""

from __future__ import annotations

from itertools import combinations

import pytest
import torch

from spatial_benchmark.negative_binomial import (
    NegativeBinomialModelOutput,
    ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
)
from spatial_benchmark.target_isolated_negative_binomial import (
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
)


def _model(
    *,
    activation_checkpointing: bool = False,
) -> TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer:
    return TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer(
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
        activation_checkpointing=activation_checkpointing,
    )


def _base_model() -> ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer:
    return ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer(
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
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(6201)
    expression = torch.randn(4, 4, generator=generator)
    covariates = torch.randn(4, 2, generator=generator)
    # Complete directed graph without self-loops, sorted by receiver.  In
    # particular 0 -> 1 -> 0 is a cycle between the two simultaneous targets.
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


def _forward(
    model: TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
    expression: torch.Tensor,
    target_mask: torch.Tensor,
    covariates: torch.Tensor,
    edge_index: torch.Tensor,
    geometry: torch.Tensor,
    target_input_expression: torch.Tensor | None = None,
    **kwargs: bool,
) -> NegativeBinomialModelOutput:
    if target_input_expression is None:
        target_input_expression = expression[:2].masked_fill(target_mask, 0.0)
    return model(
        input_expression=expression,
        target_input_expression=target_input_expression,
        target_gene_mask=target_mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        target_start=0,
        target_stop=2,
        **kwargs,
    )


def test_parameterization_is_identical_to_receiver_chunked_nb2_base() -> None:
    torch.manual_seed(6211)
    isolated = _model()
    torch.manual_seed(6211)
    base = _base_model()

    assert isinstance(
        isolated,
        ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
    )
    assert sum(p.numel() for p in isolated.parameters()) == sum(
        p.numel() for p in base.parameters()
    )
    assert tuple(isolated.state_dict()) == tuple(base.state_dict())
    assert {
        name: tuple(value.shape) for name, value in isolated.state_dict().items()
    } == {name: tuple(value.shape) for name, value in base.state_dict().items()}
    assert len(isolated.blocks) == 4
    parameter_ids = [
        {id(parameter) for parameter in block.parameters()}
        for block in isolated.blocks
    ]
    for left, right in combinations(parameter_ids, 2):
        assert left.isdisjoint(right)


def test_withheld_target_value_cannot_return_through_a_graph_cycle() -> None:
    expression, target_mask, covariates, edge_index, geometry = _inputs()
    torch.manual_seed(6217)
    model = _model().eval()
    baseline = _forward(
        model,
        expression,
        target_mask,
        covariates,
        edge_index,
        geometry,
    )

    changed = expression.clone()
    changed[0, 0] += 10_000.0  # Withheld by target 0, clean when 0 sources target 1.
    perturbed = _forward(
        model,
        changed,
        target_mask,
        covariates,
        edge_index,
        geometry,
    )

    # Even though 0 -> 1 and 1 -> 0 exist, source states never graph-evolve, so
    # target 0 has no path by which its clean withheld value can return.
    torch.testing.assert_close(baseline.mu[0], perturbed.mu[0], rtol=0, atol=0)
    # Target 1 sees target 0's clean source row, as required by the conditioning.
    assert not torch.allclose(baseline.mu[1], perturbed.mu[1])


def test_simultaneous_targets_use_each_other_as_clean_sources() -> None:
    expression, target_mask, covariates, edge_index, geometry = _inputs()
    torch.manual_seed(6221)
    model = _model().eval()
    baseline = _forward(
        model,
        expression,
        target_mask,
        covariates,
        edge_index,
        geometry,
    )

    changed = expression.clone()
    changed[1, 1] -= 8_000.0  # Withheld from target 1's query stream only.
    perturbed = _forward(
        model,
        changed,
        target_mask,
        covariates,
        edge_index,
        geometry,
    )

    torch.testing.assert_close(baseline.mu[1], perturbed.mu[1], rtol=0, atol=0)
    assert not torch.allclose(baseline.mu[0], perturbed.mu[0])


def test_nonzero_contiguous_range_matches_separate_target_visits() -> None:
    expression, _, covariates, edge_index, geometry = _inputs()
    masks = torch.tensor(
        [[False, True, False, False], [False, False, True, False]],
        dtype=torch.bool,
    )
    torch.manual_seed(6227)
    model = _model().eval()

    together = model(
        input_expression=expression,
        target_input_expression=expression[1:3].masked_fill(masks, 0.0),
        target_gene_mask=masks,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        target_start=1,
        target_stop=3,
    )
    first = model(
        input_expression=expression,
        target_input_expression=expression[1:2].masked_fill(masks[:1], 0.0),
        target_gene_mask=masks[:1],
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        target_start=1,
        target_stop=2,
    )
    second = model(
        input_expression=expression,
        target_input_expression=expression[2:3].masked_fill(masks[1:], 0.0),
        target_gene_mask=masks[1:],
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        target_start=2,
        target_stop=3,
    )

    # Batched and singleton GEMM kernels can differ by one FP32 rounding unit.
    torch.testing.assert_close(together.mu[:1], first.mu, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(together.mu[1:], second.mu, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(together.theta, first.theta, rtol=0, atol=0)
    torch.testing.assert_close(together.theta, second.theta, rtol=0, atol=0)


def test_visible_target_and_clean_neighbor_expression_can_change_output() -> None:
    expression, target_mask, covariates, edge_index, geometry = _inputs()
    torch.manual_seed(6229)
    model = _model().eval()
    baseline = _forward(
        model,
        expression,
        target_mask,
        covariates,
        edge_index,
        geometry,
    )

    visible_changed = expression.clone()
    visible_changed[0, 2] += 20.0
    visible_output = _forward(
        model,
        visible_changed,
        target_mask,
        covariates,
        edge_index,
        geometry,
    )
    assert not torch.allclose(baseline.mu[0], visible_output.mu[0])

    neighbor_changed = expression.clone()
    neighbor_changed[2, 3] -= 20.0
    neighbor_output = _forward(
        model,
        neighbor_changed,
        target_mask,
        covariates,
        edge_index,
        geometry,
    )
    assert not torch.allclose(baseline.mu[0], neighbor_output.mu[0])


def test_target_shapes_checkpointed_gradients_and_all_four_blocks() -> None:
    expression, target_mask, covariates, edge_index, geometry = _inputs()
    expression.requires_grad_()
    torch.manual_seed(6239)
    model = _model(activation_checkpointing=True).train()
    geometry_rows_by_block: list[list[int]] = [[] for _ in model.blocks]
    handles = []
    for index, block in enumerate(model.blocks):
        handles.append(
            block.geometry_encoder.register_forward_pre_hook(
                lambda _module, arguments, block_index=index: (
                    geometry_rows_by_block[block_index].append(
                        int(arguments[0].shape[0])
                    )
                )
            )
        )
    try:
        output = _forward(
            model,
            expression,
            target_mask,
            covariates,
            edge_index,
            geometry,
            return_intermediate_embeddings=True,
            return_graph_step_embeddings=True,
        )
        loss = output.mu.sum() + output.theta.sum()
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    assert isinstance(output, NegativeBinomialModelOutput)
    assert output.prediction is output.mu
    assert output.mu.shape == (2, 4)
    assert output.theta.shape == (4,)
    assert output.node_embedding.shape == (2, 8)
    assert output.full_node_embedding is not None
    assert output.full_node_embedding.shape == (2, 8)
    assert output.node_encoder_embedding is not None
    assert output.node_encoder_embedding.shape == (2, 8)
    assert output.graph_step_embeddings is not None
    assert [tuple(state.shape) for state in output.graph_step_embeddings] == [
        (2, 8),
        (2, 8),
        (2, 8),
        (2, 8),
    ]
    # Two target receivers have three incoming edges apiece. Checkpointing
    # recomputes those shards in backward, but never evaluates non-target edges.
    assert all(sum(rows) == 12 for rows in geometry_rows_by_block)

    assert model.raw_theta.grad is not None
    assert bool((model.raw_theta.grad != 0).any())
    for block in model.blocks:
        for parameter in (
            block.query_projection.weight,
            block.key_projection.weight,
            block.value_projection.weight,
            block.attention_output_projection.weight,
            block.ffn_input.weight,
            block.ffn_output.weight,
        ):
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
            assert bool((parameter.grad != 0).any())

    # For target 0's own objective, its withheld input is structurally absent,
    # while a visible target gene and a clean neighbor have gradient paths.
    model.zero_grad(set_to_none=True)
    expression.grad = None
    own_output = _forward(
        model.eval(),
        expression,
        target_mask,
        covariates,
        edge_index,
        geometry,
    )
    own_output.mu[0].sum().backward()
    assert expression.grad is not None
    assert expression.grad[0, 0].item() == 0.0
    assert expression.grad[0, 2].item() != 0.0
    assert expression.grad[2].abs().sum().item() != 0.0


def test_self_loops_invalid_ranges_and_non_receiver_major_edges_fail_closed() -> None:
    expression, target_mask, covariates, edge_index, geometry = _inputs()
    model = _model().eval()

    self_loop = edge_index.clone()
    self_loop[0, 0] = self_loop[1, 0]
    with pytest.raises(ValueError, match="self loops"):
        _forward(
            model,
            expression,
            target_mask,
            covariates,
            self_loop,
            geometry,
        )

    unsorted_order = torch.tensor([3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11])
    with pytest.raises(ValueError, match="receiver-major"):
        _forward(
            model,
            expression,
            target_mask,
            covariates,
            edge_index[:, unsorted_order],
            geometry[unsorted_order],
        )

    common = dict(
        input_expression=expression,
        target_input_expression=expression[:2].masked_fill(target_mask, 0.0),
        target_gene_mask=target_mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
    )
    for target_start, target_stop in ((-1, 2), (0, 0), (2, 1), (0, 5)):
        with pytest.raises(ValueError, match="target range"):
            model(
                **common,
                target_start=target_start,
                target_stop=target_stop,
            )
    with pytest.raises(TypeError, match="target_start"):
        model(**common, target_start=False, target_stop=2)


def test_target_mask_must_be_boolean_aligned_and_nonempty_per_target() -> None:
    expression, target_mask, covariates, edge_index, geometry = _inputs()
    model = _model().eval()
    common = dict(
        input_expression=expression,
        target_input_expression=expression[:2].masked_fill(target_mask, 0.0),
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=covariates,
        target_start=0,
        target_stop=2,
    )

    with pytest.raises(TypeError, match="boolean"):
        model(target_gene_mask=target_mask.float(), **common)
    with pytest.raises(ValueError, match="shape"):
        model(target_gene_mask=target_mask[:1], **common)
    empty_row = target_mask.clone()
    empty_row[0].fill_(False)
    with pytest.raises(ValueError, match="at least one"):
        model(target_gene_mask=empty_row, **common)

    nonzero_withheld = expression[:2].masked_fill(target_mask, 0.0)
    nonzero_withheld[0, 0] = 1.0
    with pytest.raises(ValueError, match="exactly zero"):
        model(
            target_input_expression=nonzero_withheld,
            target_gene_mask=target_mask,
            **{key: value for key, value in common.items() if key != "target_input_expression"},
        )

    visible_drift = expression[:2].masked_fill(target_mask, 0.0)
    visible_drift[0, 2] += 1.0
    with pytest.raises(ValueError, match="visible"):
        model(
            target_input_expression=visible_drift,
            target_gene_mask=target_mask,
            **{key: value for key, value in common.items() if key != "target_input_expression"},
        )
