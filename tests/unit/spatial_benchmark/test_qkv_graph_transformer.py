"""Reference, chunk-equivalence, and semantic tests for QKV graph attention."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import spatial_benchmark.qkv_graph_transformer as qkv_module  # noqa: E402
from spatial_benchmark.qkv_graph_transformer import (  # noqa: E402
    EdgeAwareQKVGraphTransformer,
    ReceiverChunkedEdgeAwareQKVGraphTransformer,
)


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7301)
    # Deliberately source-major, not receiver-major, to exercise the exact
    # stable receiver permutation and restoration to caller edge order.
    edge_index = torch.tensor(
        [
            [
                0,
                0,
                0,
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                4,
                5,
                5,
                5,
                6,
                6,
                6,
            ],
            [
                1,
                2,
                6,
                0,
                2,
                5,
                0,
                1,
                3,
                1,
                4,
                6,
                2,
                3,
                5,
                0,
                4,
                6,
                1,
                3,
                5,
            ],
        ],
        dtype=torch.long,
    )
    return {
        "input_expression": torch.randn(7, 9, generator=generator),
        "gene_mask": torch.rand(7, 9, generator=generator) < 0.35,
        "node_covariates": torch.randn(7, 4, generator=generator),
        "edge_index": edge_index,
        "edge_attributes": torch.randn(
            edge_index.shape[1], 6, generator=generator
        ),
    }


def _model_kwargs() -> dict[str, int | float | str]:
    # raw QKV width (3 * 7 = 21) deliberately differs from hidden width 18.
    return {
        "num_genes": 9,
        "edge_attribute_dim": 6,
        "node_covariate_dim": 4,
        "hidden_dim": 18,
        "attention_heads": 3,
        "attention_head_dim": 7,
        "graph_layers": 3,
        "ffn_dim": 31,
        "decoder_dim": 23,
        "edge_hidden_dim": 13,
        "edge_embedding_dim": 8,
        "edge_conditioning_mode": "vector",
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }


def _paired_models(
    *,
    activation_checkpointing: bool,
    receiver_chunk_size: int = 2,
) -> tuple[
    EdgeAwareQKVGraphTransformer,
    ReceiverChunkedEdgeAwareQKVGraphTransformer,
]:
    torch.manual_seed(8502)
    reference = EdgeAwareQKVGraphTransformer(**_model_kwargs())
    chunked = ReceiverChunkedEdgeAwareQKVGraphTransformer(
        **_model_kwargs(),
        receiver_chunk_size=receiver_chunk_size,
        activation_checkpointing=activation_checkpointing,
    )
    chunked.load_state_dict(reference.state_dict())
    assert tuple(chunked.state_dict()) == tuple(reference.state_dict())
    return reference, chunked


def test_explicit_qkv_edge_equation_matches_independent_loop_reference() -> None:
    """Prove the operator is dot-product QKV, not additive GAT scoring."""

    inputs = _inputs()
    kwargs = {
        **_model_kwargs(),
        "hidden_dim": 8,
        "attention_heads": 2,
        "attention_head_dim": 4,
        "graph_layers": 1,
        "ffn_dim": 15,
    }
    torch.manual_seed(105)
    model = EdgeAwareQKVGraphTransformer(**kwargs).eval()

    with torch.no_grad():
        output = model(**inputs, return_explanations=True)
        node_embedding = model.encoder(
            inputs["input_expression"],
            inputs["gene_mask"],
            inputs["node_covariates"],
        )
        block = model.blocks[0]
        normalized = block.attention_normalization(node_embedding)
        shape = (7, 2, 4)
        queries = block.query_projection(normalized).view(shape)
        keys = block.key_projection(normalized).view(shape)
        values = block.value_projection(normalized).view(shape)
        encoded_edges = model.edge_encoder(inputs["edge_attributes"])
        edge_keys = block.edge_key_projection(encoded_edges).view(-1, 2, 4)
        edge_values = block.edge_value_projection(encoded_edges).view(
            -1, 2, 4
        )
        edge_bias = block.edge_attention_bias(encoded_edges)

        source, receiver = inputs["edge_index"]
        manual_attention = torch.empty(source.shape[0], 2)
        aggregate = torch.zeros(7, 2, 4)
        for receiver_id in range(7):
            selected = receiver == receiver_id
            sender_ids = source[selected]
            scores = (
                (
                    queries[receiver_id].unsqueeze(0)
                    * (keys[sender_ids] + edge_keys[selected])
                ).sum(dim=-1)
                * block.attention_scale
                + edge_bias[selected]
            )
            weights = torch.softmax(scores, dim=0)
            manual_attention[selected] = weights
            aggregate[receiver_id] = (
                weights.unsqueeze(-1)
                * (values[sender_ids] + edge_values[selected])
            ).sum(dim=0)

        manual_embedding = block.finish_partition(
            node_embedding, aggregate
        )
        manual_prediction = model.decoder(manual_embedding)

    torch.testing.assert_close(
        output.attention_weights,
        manual_attention,
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        output.node_embedding,
        manual_embedding,
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        output.prediction,
        manual_prediction,
        rtol=2e-6,
        atol=2e-7,
    )


def test_three_layer_chunked_forward_and_explanations_match_reference() -> None:
    inputs = _inputs()
    reference, chunked = _paired_models(activation_checkpointing=False)
    reference.eval()
    chunked.eval()

    expected = reference(**inputs, return_explanations=True)
    actual = chunked(**inputs, return_explanations=True)

    torch.testing.assert_close(
        actual.prediction, expected.prediction, rtol=3e-6, atol=3e-7
    )
    torch.testing.assert_close(
        actual.node_embedding,
        expected.node_embedding,
        rtol=3e-6,
        atol=3e-7,
    )
    torch.testing.assert_close(
        actual.attention_weights,
        expected.attention_weights,
        rtol=3e-6,
        atol=3e-7,
    )
    torch.testing.assert_close(
        actual.edge_embedding,
        expected.edge_embedding,
        rtol=3e-6,
        atol=3e-7,
    )
    assert torch.equal(actual.edge_index, inputs["edge_index"])

    assert actual.attention_weights is not None
    receiver = actual.edge_index[1]
    for receiver_id in range(inputs["input_expression"].shape[0]):
        torch.testing.assert_close(
            actual.attention_weights[receiver == receiver_id].sum(dim=0),
            torch.ones(_model_kwargs()["attention_heads"]),
            rtol=2e-6,
            atol=2e-7,
        )


def test_checkpointed_backward_matches_reference_for_every_parameter() -> None:
    inputs = _inputs()
    reference, chunked = _paired_models(activation_checkpointing=True)
    reference.train()
    chunked.train()

    expected = reference(**inputs, target_nodes=[0, 2, 5]).prediction
    actual = chunked(**inputs, target_nodes=[0, 2, 5]).prediction
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-7)

    expected.square().mean().backward()
    actual.square().mean().backward()
    expected_parameters = dict(reference.named_parameters())
    actual_parameters = dict(chunked.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name, expected_parameter in expected_parameters.items():
        expected_gradient = expected_parameter.grad
        actual_gradient = actual_parameters[name].grad
        assert expected_gradient is not None, name
        assert actual_gradient is not None, name
        assert bool(torch.isfinite(actual_gradient).all()), name
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=4e-5,
            atol=4e-7,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_checkpoint_inputs_reuse_receiver_sorted_graph_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not save a fresh E-length local-receiver tensor per layer."""

    inputs = _inputs()
    order = torch.argsort(inputs["edge_index"][1], stable=True)
    inputs["edge_index"] = inputs["edge_index"].index_select(1, order)
    inputs["edge_attributes"] = inputs["edge_attributes"].index_select(
        0, order
    )
    reference, chunked = _paired_models(activation_checkpointing=True)
    reference.train()
    chunked.train()

    integer_storage_pointers: list[int] = []
    checkpoint_calls = 0

    def inspect_checkpoint(
        function: object,
        *arguments: object,
        **_kwargs: object,
    ) -> object:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        for argument in arguments:
            if (
                isinstance(argument, torch.Tensor)
                and argument.dtype == torch.long
            ):
                integer_storage_pointers.append(
                    argument.untyped_storage().data_ptr()
                )
        assert callable(function)
        return function(*arguments)

    monkeypatch.setattr(qkv_module, "checkpoint", inspect_checkpoint)
    expected = reference(**inputs).prediction
    actual = chunked(**inputs).prediction

    torch.testing.assert_close(
        actual, expected, rtol=3e-6, atol=3e-7
    )
    assert checkpoint_calls == 12  # four chunks x three graph layers
    assert len(integer_storage_pointers) == 2 * checkpoint_calls
    graph_storage_pointer = (
        inputs["edge_index"].untyped_storage().data_ptr()
    )
    assert set(integer_storage_pointers) == {graph_storage_pointer}


def test_bfloat16_autocast_matches_reference_and_remains_finite() -> None:
    inputs = _inputs()
    reference, chunked = _paired_models(activation_checkpointing=False)
    reference.eval()
    chunked.eval()

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference(**inputs)
        actual = chunked(**inputs)

    assert bool(torch.isfinite(actual.prediction).all())
    assert bool(torch.isfinite(actual.node_embedding).all())
    torch.testing.assert_close(
        actual.prediction, expected.prediction, rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        actual.node_embedding,
        expected.node_embedding,
        rtol=1e-2,
        atol=1e-2,
    )


def _uniform_high_degree_attention(
    *,
    incoming_degree: int,
    device: torch.device,
    low_precision_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return attention and aggregation for a uniform one-receiver graph."""

    model = EdgeAwareQKVGraphTransformer(
        num_genes=1,
        edge_attribute_dim=1,
        hidden_dim=4,
        attention_heads=2,
        graph_layers=1,
        ffn_dim=8,
        decoder_dim=4,
        edge_hidden_dim=2,
        edge_embedding_dim=2,
        edge_conditioning_mode="bias_gate",
        dropout=0.0,
        attention_dropout=0.0,
    ).to(device).eval()
    block = model.blocks[0]
    assert block.edge_value_gate is not None
    with torch.no_grad():
        block.edge_attention_bias.weight.zero_()
        block.edge_value_gate.weight.zero_()
        block.attention_output_projection.weight.copy_(
            torch.eye(4, device=device)
        )
        block.ffn_output.weight.zero_()
        block.ffn_output.bias.zero_()

    source = torch.arange(
        incoming_degree, dtype=torch.long, device=device
    )
    local_receiver = torch.zeros(
        incoming_degree, dtype=torch.long, device=device
    )
    queries = torch.zeros(
        1, 2, 2, dtype=low_precision_dtype, device=device
    )
    keys = torch.zeros(
        incoming_degree,
        2,
        2,
        dtype=low_precision_dtype,
        device=device,
    )
    values = torch.ones_like(keys)
    residual = torch.zeros(
        1, 4, dtype=low_precision_dtype, device=device
    )
    edge_attributes = torch.zeros(
        incoming_degree, 1, dtype=torch.float32, device=device
    )

    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=low_precision_dtype,
    ):
        output, attention, _ = model._attention_partition(
            block,
            queries,
            keys,
            values,
            residual,
            source,
            local_receiver,
            edge_attributes,
        )
    return attention.cpu(), output.cpu().float()


@pytest.mark.parametrize("incoming_degree", [1000, 5000])
def test_high_degree_amp_reduction_stays_fp32_and_sums_to_one(
    incoming_degree: int,
) -> None:
    """Regress the BF16/FP16 index_add saturation at high neighbor counts."""

    attention, output = _uniform_high_degree_attention(
        incoming_degree=incoming_degree,
        device=torch.device("cpu"),
        low_precision_dtype=torch.bfloat16,
    )

    assert attention.dtype == torch.float32
    torch.testing.assert_close(
        attention.sum(dim=0),
        torch.ones(2),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        output,
        torch.ones(1, 4),
        rtol=2e-4,
        atol=2e-4,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA FP16 regression requires a CUDA device",
)
def test_k5000_cuda_fp16_reduction_does_not_collapse_to_half() -> None:
    attention, output = _uniform_high_degree_attention(
        incoming_degree=5000,
        device=torch.device("cuda:0"),
        low_precision_dtype=torch.float16,
    )

    assert attention.dtype == torch.float32
    torch.testing.assert_close(
        attention.sum(dim=0),
        torch.ones(2),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        output,
        torch.ones(1, 4),
        rtol=2e-4,
        atol=2e-4,
    )


def test_selected_receivers_preserve_original_edge_alignment() -> None:
    inputs = _inputs()
    reference, chunked = _paired_models(activation_checkpointing=False)
    reference.eval()
    chunked.eval()

    expected = reference(**inputs, return_explanations=True)
    selected_receivers = torch.tensor([5, 1], dtype=torch.long)
    actual = chunked(
        **inputs,
        return_explanations=True,
        attention_receivers=selected_receivers,
    )
    selected_edges = torch.isin(
        inputs["edge_index"][1], selected_receivers
    )

    torch.testing.assert_close(
        actual.prediction, expected.prediction, rtol=3e-6, atol=3e-7
    )
    assert torch.equal(
        actual.edge_index, inputs["edge_index"][:, selected_edges]
    )
    torch.testing.assert_close(
        actual.attention_weights,
        expected.attention_weights[selected_edges],
        rtol=3e-6,
        atol=3e-7,
    )
    torch.testing.assert_close(
        actual.edge_embedding,
        expected.edge_embedding[selected_edges],
        rtol=3e-6,
        atol=3e-7,
    )


def test_high_degree_receiver_sorted_graph_is_exact_and_memory_bounded() -> None:
    """Exercise the production ordering with 40 complete incoming neighbors."""

    generator = torch.Generator().manual_seed(921)
    num_nodes = 41
    receiver_parts: list[torch.Tensor] = []
    source_parts: list[torch.Tensor] = []
    all_nodes = torch.arange(num_nodes)
    for receiver_id in range(num_nodes):
        sources = all_nodes[all_nodes != receiver_id]
        source_parts.append(sources)
        receiver_parts.append(torch.full_like(sources, receiver_id))
    edge_index = torch.stack(
        [torch.cat(source_parts), torch.cat(receiver_parts)]
    )
    inputs = {
        "input_expression": torch.randn(
            num_nodes, 5, generator=generator
        ),
        "gene_mask": torch.rand(
            num_nodes, 5, generator=generator
        )
        < 0.25,
        "node_covariates": torch.randn(
            num_nodes, 2, generator=generator
        ),
        "edge_index": edge_index,
        "edge_attributes": torch.randn(
            edge_index.shape[1], 3, generator=generator
        ),
    }
    kwargs: dict[str, int | float | str] = {
        "num_genes": 5,
        "edge_attribute_dim": 3,
        "node_covariate_dim": 2,
        "hidden_dim": 12,
        "attention_heads": 3,
        "graph_layers": 2,
        "ffn_dim": 19,
        "decoder_dim": 11,
        "edge_hidden_dim": 7,
        "edge_embedding_dim": 5,
        "edge_conditioning_mode": "bias_gate",
        "dropout": 0.0,
        "attention_dropout": 0.0,
    }
    torch.manual_seed(654)
    reference = EdgeAwareQKVGraphTransformer(**kwargs).eval()
    chunked = ReceiverChunkedEdgeAwareQKVGraphTransformer(
        **kwargs,
        receiver_chunk_size=4,
        max_edges_per_chunk=95,
        activation_checkpointing=False,
    ).eval()
    chunked.load_state_dict(reference.state_dict())

    encoded_counts: list[int] = []

    def record_count(
        _module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
    ) -> None:
        encoded_counts.append(arguments[0].shape[0])

    handle = chunked.edge_encoder.register_forward_pre_hook(record_count)
    try:
        with torch.no_grad():
            expected = reference(**inputs).prediction
            actual = chunked(**inputs).prediction
    finally:
        handle.remove()

    torch.testing.assert_close(
        actual, expected, rtol=4e-6, atol=4e-7
    )
    assert encoded_counts
    # The 95-edge soft limit fits two complete 40-edge receivers.  It never
    # splits a receiver merely to hit the limit exactly.
    assert max(encoded_counts) <= 2 * (num_nodes - 1)
    assert max(encoded_counts) < edge_index.shape[1]
    assert sum(encoded_counts) == (
        len(chunked.blocks) * edge_index.shape[1]
    )
    assert chunked._receiver_layout_cache is not None
    assert chunked._receiver_layout_cache.edge_order is None


def test_empty_selected_receiver_explanations_are_well_shaped() -> None:
    _, chunked = _paired_models(activation_checkpointing=False)
    chunked.eval()
    with torch.no_grad():
        output = chunked(
            **_inputs(),
            return_explanations=True,
            attention_receivers=[],
        )
    assert output.edge_index is not None
    assert output.attention_weights is not None
    assert output.edge_embedding is not None
    assert output.edge_index.shape == (2, 0)
    assert output.attention_weights.shape == (0, 3)
    assert output.edge_embedding.shape == (0, 8)


def test_attention_receiver_selection_requires_explanations() -> None:
    _, chunked = _paired_models(activation_checkpointing=False)
    with pytest.raises(
        ValueError, match="requires return_explanations=True"
    ):
        chunked(**_inputs(), attention_receivers=[1])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("receiver_chunk_size", 0, "positive"),
        ("max_edges_per_chunk", 0, "positive"),
        ("activation_checkpointing", 1, "boolean"),
        ("edge_conditioning_mode", "unknown", "bias_gate.*vector"),
    ],
)
def test_chunk_execution_options_are_validated(
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs = {
        **_model_kwargs(),
        "receiver_chunk_size": 2,
        "max_edges_per_chunk": None,
        "activation_checkpointing": True,
        field: value,
    }
    with pytest.raises((TypeError, ValueError), match=match):
        ReceiverChunkedEdgeAwareQKVGraphTransformer(**kwargs)
