from __future__ import annotations

import pytest
import torch

from spatial_benchmark.cli import build_parser
from spatial_benchmark.relative_qkv_graph_transformer import (
    ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer,
)
import spatial_benchmark.so2_recurrent_hl_clustering as analysis


def _small_model() -> ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer:
    return ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer(
        num_genes=6,
        node_covariate_dim=3,
        hidden_dim=16,
        attention_heads=4,
        attention_head_dim=4,
        ffn_dim=32,
        decoder_dim=24,
        positional_bias_hidden_dim=12,
        dropout=0.0,
        attention_dropout=0.0,
        recurrent_unroll_steps=4,
        receiver_chunk_size=2,
        max_edges_per_chunk=4,
        activation_checkpointing=False,
    ).eval()


def test_parse_extract_devices_requires_unique_explicit_cuda_indices() -> None:
    assert analysis.parse_extract_devices("cuda:0, cuda:2") == (
        "cuda:0",
        "cuda:2",
    )
    for invalid in ("", "cpu", "cuda", "cuda:0,cuda:0"):
        with pytest.raises(analysis.SO2RecurrentHLClusteringError):
            analysis.parse_extract_devices(invalid)


def test_full_recurrent_extraction_returns_final_all_node_embedding() -> None:
    torch.manual_seed(17)
    model = _small_model()
    expression = torch.randn(5, 6)
    mask = torch.zeros_like(expression, dtype=torch.bool)
    covariates = torch.randn(5, 3)
    edges = torch.tensor(
        [[1, 2, 0, 2, 3, 0, 4], [0, 0, 1, 1, 1, 2, 3]],
        dtype=torch.long,
    )
    geometry = torch.randn(edges.shape[1], 70)
    with torch.inference_mode():
        observed = analysis.extract_full_recurrent_contextual_embedding(
            model, expression, mask, edges, geometry, covariates
        )
        ordinary = model(
            expression,
            mask,
            edge_index=edges,
            relative_geometry=geometry,
            node_covariates=covariates,
            return_intermediate_embeddings=True,
        )
    assert observed.shape == (5, 16)
    assert ordinary.final_graph_embedding is not None
    torch.testing.assert_close(observed, ordinary.final_graph_embedding)


def test_full_recurrent_extraction_rejects_nonzero_mask() -> None:
    model = _small_model()
    expression = torch.randn(2, 6)
    mask = torch.zeros_like(expression, dtype=torch.bool)
    mask[0, 0] = True
    with torch.inference_mode(), pytest.raises(
        analysis.SO2RecurrentHLClusteringError, match="all-zero"
    ):
        analysis.extract_full_recurrent_contextual_embedding(
            model,
            expression,
            mask,
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 70)),
            torch.randn(2, 3),
        )


def test_edge_balanced_schedule_is_deterministic_and_complete() -> None:
    records = [
        {"core_number": number, "directed_edges": edges}
        for number, edges in ((15, 10), (16, 8), (17, 6), (18, 4), (19, 2))
    ]
    observed = analysis._balanced_assignments(records, ("cuda:0", "cuda:1"))
    assert {
        int(record["core_number"])
        for device_records in observed.values()
        for record in device_records
    } == {15, 16, 17, 18, 19}
    assert observed == analysis._balanced_assignments(
        records, ("cuda:0", "cuda:1")
    )


def test_recurrent_hl_cli_defaults_to_locked_png_workflow() -> None:
    arguments = build_parser().parse_args(["analyze-so2-recurrent-hl-clusters"])
    assert arguments.n_neighbors == 30
    assert arguments.leiden_resolution == 1.0
    assert arguments.pca_components == 50
    assert arguments.random_seed == 20260825
    assert arguments.extract_devices == "cuda:0,cuda:1,cuda:2,cuda:3"
    assert arguments.cpu_threads_per_worker == 4
    assert arguments.dpi == 300
