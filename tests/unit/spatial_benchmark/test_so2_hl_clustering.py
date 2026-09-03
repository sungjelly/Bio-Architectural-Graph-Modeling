from __future__ import annotations

import numpy as np
import pytest
import torch

import spatial_benchmark.so2_hl_clustering as analysis
from spatial_benchmark.cli import build_parser
from spatial_benchmark.relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.so2_pooled_full_core import SO2_ALIASES, SO2_CORE_NUMBERS


def _model() -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    torch.manual_seed(20260825)
    return ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=6,
        node_covariate_dim=3,
        hidden_dim=16,
        attention_heads=4,
        attention_head_dim=4,
        graph_layers=2,
        ffn_dim=32,
        decoder_dim=24,
        positional_bias_hidden_dim=12,
        dropout=0.35,
        attention_dropout=0.25,
        receiver_chunk_size=2,
        max_edges_per_chunk=4,
        activation_checkpointing=True,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    expression = torch.randn(5, 6, generator=generator)
    mask = torch.zeros_like(expression, dtype=torch.bool)
    metadata = torch.randn(5, 3, generator=generator)
    edge_index = torch.tensor(
        [[1, 2, 0, 2, 3, 0, 4], [0, 0, 1, 1, 1, 2, 3]],
        dtype=torch.long,
    )
    geometry = torch.randn(edge_index.shape[1], 70, generator=generator)
    return expression, mask, metadata, edge_index, geometry


def _contextual_cores(
    *, cells_per_core: int = 2
) -> tuple[analysis.SO2ContextualCore, ...]:
    cores: list[analysis.SO2ContextualCore] = []
    for offset, (core_number, alias) in enumerate(
        zip(SO2_CORE_NUMBERS, SO2_ALIASES, strict=True)
    ):
        cell_index = np.arange(cells_per_core, dtype=np.int64)
        coordinates = np.column_stack(
            (
                cell_index.astype(np.float64) * 10.0 + offset,
                cell_index.astype(np.float64) * 5.0 + offset * 2.0,
            )
        )
        contextual = np.column_stack(
            (
                cell_index.astype(np.float32),
                np.full(cells_per_core, offset, dtype=np.float32),
                np.ones(cells_per_core, dtype=np.float32),
            )
        )
        cores.append(
            analysis.SO2ContextualCore(
                alias=alias,
                core_number=core_number,
                cell_index=cell_index,
                coordinates_um=coordinates,
                hL=contextual,
            )
        )
    return tuple(cores)


def test_extract_full_contextual_embedding_matches_model_intermediate_output() -> None:
    expression, mask, metadata, edge_index, geometry = _inputs()
    expression_before = expression.clone()
    metadata_before = metadata.clone()
    model = _model().eval()

    extracted = analysis.extract_full_contextual_embedding(
        model,
        input_expression=expression,
        gene_mask=mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=metadata,
    )
    with torch.inference_mode():
        expected = model(
            input_expression=expression,
            gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=metadata,
            return_intermediate_embeddings=True,
        ).final_graph_embedding

    assert model.training is False
    assert expected is not None
    torch.testing.assert_close(extracted, expected)
    assert extracted.shape == (len(expression), model.hidden_dim)
    assert extracted.device.type == "cpu"
    assert torch.isfinite(extracted).all()
    assert torch.equal(expression, expression_before)
    assert torch.equal(metadata, metadata_before)


def test_extract_full_contextual_embedding_requires_all_zero_mask() -> None:
    expression, mask, metadata, edge_index, geometry = _inputs()
    mask[0, 0] = True

    with pytest.raises(ValueError, match="zero"):
        analysis.extract_full_contextual_embedding(
            _model().eval(),
            input_expression=expression,
            gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=metadata,
        )


def test_cpu_device_validation_rejects_cuda() -> None:
    assert analysis.validate_cpu_device("cpu") == torch.device("cpu")
    assert analysis.validate_cpu_device(torch.device("cpu")) == torch.device("cpu")
    with pytest.raises(ValueError, match="CPU|cpu"):
        analysis.validate_cpu_device("cuda")
    with pytest.raises(ValueError, match="CPU|cpu"):
        analysis.validate_cpu_device(torch.device("cuda:0"))


def test_so2_hl_cli_defaults_to_cpu_and_resolution_one() -> None:
    arguments = build_parser().parse_args(["analyze-so2-hl-clusters"])

    assert arguments.device == "cpu"
    assert arguments.leiden_resolution == 1.0
    assert arguments.n_neighbors == 30
    assert arguments.pca_components == 50
    assert arguments.random_seed == 20260825


def test_panel_order_and_spatial_spec_keep_core_numbers_identifiable() -> None:
    palette = {"C0": "#0072B2", "C1": "#D55E00"}
    spec = analysis.spatial_plot_spec(palette)

    assert analysis.requested_panel_order() == tuple(range(15, 29))
    assert tuple(spec["panel_order"]) == tuple(range(15, 29))
    assert int(np.prod(spec["grid_shape"])) >= len(SO2_CORE_NUMBERS)
    assert spec["equal_aspect"] is True
    assert spec["core_numbers_identifiable"] is True
    assert spec["panel_title_template"].format(core_number=21) == "SO2 Core 21"
    assert spec["palette"] == palette
    assert spec["one_shared_joint_cluster_palette"] is True


def test_joint_cell_frame_and_cluster_summaries_cover_all_fourteen_cores() -> None:
    cores = _contextual_cores()
    frame = analysis.build_cell_index_frame(cores)
    labels = np.tile(np.arange(2, dtype=np.int64), len(SO2_CORE_NUMBERS))
    summary, composition, dominated = analysis.cluster_summary_tables(labels, frame)

    assert tuple(frame["core_number"].drop_duplicates()) == SO2_CORE_NUMBERS
    assert len(frame) == sum(core.n_cells for core in cores)
    assert np.array_equal(
        frame["global_cell_index"].to_numpy(),
        np.arange(len(frame), dtype=np.int64),
    )
    assert not frame["cell_key"].duplicated().any()
    assert summary["cluster"].tolist() == ["C0", "C1"]
    assert summary["size"].tolist() == [14, 14]
    assert int(summary["size"].sum()) == len(frame)
    assert set(composition["core_number"]) == set(SO2_CORE_NUMBERS)
    assert len(composition) == 2 * len(SO2_CORE_NUMBERS)
    assert dominated == []

    with pytest.raises(ValueError, match="align|label|cell"):
        analysis.cluster_summary_tables(labels[:-1], frame)
