"""Verify that the decoder-free hL extractor preserves geometry-model states."""

from __future__ import annotations

import pytest
import torch

from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import (
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
)
from spatial_benchmark.so2_recurrent_hl_clustering import (
    SO2RecurrentHLClusteringError,
    extract_full_recurrent_contextual_embedding,
)


def _fixture() -> tuple[
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
    tuple[torch.Tensor, ...],
]:
    torch.manual_seed(905)
    model = ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer(
        num_genes=6,
        node_covariate_dim=3,
        hidden_dim=16,
        attention_heads=4,
        attention_head_dim=4,
        graph_layers=4,
        ffn_dim=32,
        decoder_dim=24,
        geometry_hidden_dim=12,
        dropout=0.0,
        attention_dropout=0.0,
        receiver_chunk_size=2,
        max_edges_per_chunk=4,
        activation_checkpointing=False,
    ).eval()
    # Non-neutral geometry ensures the check exercises learned score modulation.
    with torch.no_grad():
        for block in model.blocks:
            block.geometry_encoder.modulation_projection.weight.normal_(0.0, 0.1)
            block.geometry_encoder.bias_projection.weight.normal_(0.0, 0.1)
    expression = torch.randn(5, 6)
    mask = torch.zeros_like(expression, dtype=torch.bool)
    edges = torch.tensor(
        [[1, 2, 0, 2, 3, 0, 4], [0, 0, 1, 1, 1, 2, 3]], dtype=torch.long
    )
    return model, (
        expression,
        mask,
        edges,
        torch.randn(edges.shape[1], 70),
        torch.randn(5, 3),
    )


def test_geometry_hl_extraction_matches_ordinary_forward_final_embedding() -> None:
    model, inputs = _fixture()
    expression, mask, edges, geometry, covariates = inputs
    decoder_rows: list[int] = []
    handle = model.decoder.register_forward_pre_hook(
        lambda _module, arguments: decoder_rows.append(len(arguments[0]))
    )
    try:
        with torch.inference_mode():
            observed = extract_full_recurrent_contextual_embedding(model, *inputs)
            ordinary = model(
                expression,
                mask,
                edge_index=edges,
                relative_geometry=geometry,
                node_covariates=covariates,
                return_intermediate_embeddings=True,
            )
    finally:
        handle.remove()
    assert decoder_rows == [0, 5]
    assert observed.shape == (5, 16)
    assert ordinary.final_graph_embedding is not None
    assert torch.equal(observed, ordinary.final_graph_embedding)


def test_geometry_hl_extraction_rejects_nonzero_mask() -> None:
    model, inputs = _fixture()
    inputs[1][0, 0] = True
    with torch.inference_mode(), pytest.raises(
        SO2RecurrentHLClusteringError, match="all-zero"
    ):
        extract_full_recurrent_contextual_embedding(model, *inputs)


def test_geometry_hl_extraction_requires_eval_and_disabled_gradients() -> None:
    model, inputs = _fixture()
    with torch.enable_grad(), pytest.raises(
        SO2RecurrentHLClusteringError, match="eval and inference mode"
    ):
        extract_full_recurrent_contextual_embedding(model, *inputs)
    model.train()
    with torch.inference_mode(), pytest.raises(
        SO2RecurrentHLClusteringError, match="eval and inference mode"
    ):
        extract_full_recurrent_contextual_embedding(model, *inputs)
