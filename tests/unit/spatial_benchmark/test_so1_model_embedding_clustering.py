from __future__ import annotations

from pathlib import Path

import matplotlib.axes
import numpy as np
import pytest
import torch

import spatial_benchmark.so1_model_embedding_clustering as analysis
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.relative_qkv_embedding_clustering import KNNGraphResult
from spatial_benchmark.relative_qkv_graph_transformer import (
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)


def _model() -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    torch.manual_seed(17)
    model = ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=7,
        node_covariate_dim=analysis.EXPECTED_NODE_COVARIATE_DIM,
        hidden_dim=16,
        attention_heads=2,
        attention_head_dim=8,
        graph_layers=2,
        ffn_dim=24,
        decoder_dim=12,
        positional_bias_hidden_dim=8,
        dropout=0.25,
        attention_dropout=0.0,
        relative_geometry_dim=analysis.EXPECTED_RELATIVE_GEOMETRY_DIM,
        receiver_chunk_size=13,
        max_edges_per_chunk=64,
        activation_checkpointing=False,
    )
    return model.eval()


def _model_inputs(
    n_cells: int = 65,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    expression = torch.randn((n_cells, 7), generator=generator)
    mask = torch.zeros_like(expression, dtype=torch.bool)
    metadata = torch.randn(
        (n_cells, analysis.EXPECTED_NODE_COVARIATE_DIM), generator=generator
    )
    source = torch.arange(n_cells, dtype=torch.long).repeat_interleave(2)
    receiver = torch.column_stack(
        (
            (torch.arange(n_cells) + 1) % n_cells,
            (torch.arange(n_cells) - 1) % n_cells,
        )
    ).reshape(-1)
    edge_index = torch.stack((source, receiver))
    geometry = torch.randn(
        (edge_index.shape[1], analysis.EXPECTED_RELATIVE_GEOMETRY_DIM),
        generator=generator,
    )
    return expression, mask, metadata, edge_index, geometry


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=root,
        config_root=root / "configs",
        data_root=root / "data",
        artifact_root=root / "artifacts",
        state_root=root / "state",
        scratch_root=root / "scratch",
        cache_root=root / "cache",
        export_root=root / "exports",
        report_root=root / "reports",
        result_root=root / "results",
    )


def test_extracts_exact_all_node_h0_hl_and_verifies_prediction_invariance() -> None:
    model = _model()
    expression, mask, metadata, edge_index, geometry = _model_inputs()
    expression_before = expression.clone()
    metadata_before = metadata.clone()

    h0, h_l, receipt = analysis.extract_full_h0_hl(
        model,
        input_expression=expression,
        gene_mask=mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=metadata,
    )

    expected_h0 = model.encoder(
        input_expression=expression,
        gene_mask=mask,
        node_covariates=metadata,
    )
    extended = model(
        input_expression=expression,
        gene_mask=mask,
        edge_index=edge_index,
        relative_geometry=geometry,
        node_covariates=metadata,
        return_intermediate_embeddings=True,
    )
    assert h0.shape == h_l.shape == (65, 16)
    assert torch.equal(h0, expected_h0)
    assert torch.equal(h_l, extended.full_node_embedding)
    assert torch.equal(h0, extended.node_encoder_embedding)
    assert torch.equal(h_l, extended.final_graph_embedding)
    assert receipt["verified_on_completed_trained_model"] is True
    assert receipt["predictions_allclose"] is True
    assert receipt["probe_count"] == 64
    assert receipt["all_cell_expression_decode_performed"] is False
    assert torch.equal(expression, expression_before)
    assert torch.equal(metadata, metadata_before)
    assert not bool(mask.any())


def test_extraction_rejects_nonzero_mask_and_training_mode() -> None:
    model = _model()
    expression, mask, metadata, edge_index, geometry = _model_inputs(8)
    mask[0, 0] = True
    with pytest.raises(ValueError, match="all-zero"):
        analysis.extract_full_h0_hl(
            model,
            input_expression=expression,
            gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=metadata,
        )

    model.train()
    mask.zero_()
    with pytest.raises(ValueError, match=r"model\.eval"):
        analysis.extract_full_h0_hl(
            model,
            input_expression=expression,
            gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=metadata,
        )


def test_core_embedding_validation_locks_order_delta_and_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "EXPECTED_HIDDEN_DIM", 3)
    h0 = np.arange(12, dtype=np.float32).reshape(4, 3)
    h_l = h0 + np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    delta = h_l - h0
    arrays = {
        "cell_index": np.arange(4, dtype=np.int64),
        "core_number": np.asarray(1, dtype=np.int16),
        "coordinates_um": np.arange(8, dtype=np.float64).reshape(4, 2),
        "h0": h0,
        "hL": h_l,
        "delta_h": delta,
        "delta_h_l2": np.linalg.norm(delta.astype(np.float64), axis=1),
    }
    core = analysis._validate_core_embedding_arrays(
        alias="SO1-C01", core_number=1, arrays=arrays, expected_cells=4
    )
    assert np.array_equal(core.cell_index, np.arange(4))
    assert np.isfinite(core.delta_h_l2).all()
    assert np.all(core.delta_h_l2 >= 0.0)

    reordered = dict(arrays)
    reordered["cell_index"] = np.asarray([1, 0, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="coverage/order"):
        analysis._validate_core_embedding_arrays(
            alias="SO1-C01", core_number=1, arrays=reordered, expected_cells=4
        )


def test_direct_cosine_setup_has_no_centering_or_projection() -> None:
    values = np.asarray(
        [[2.0, 0.0, 1.0], [0.0, 3.0, 1.0], [2.0, 2.0, 4.0]],
        dtype=np.float32,
    )
    before = values.copy()
    normalized, receipt = analysis.l2_normalize_for_cosine(
        values, representation="contextual"
    )
    assert np.array_equal(values, before)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert receipt["representation_learning_or_reduction"] is False
    assert receipt["pca"] is False
    assert receipt["mean_center"] is False
    assert receipt["raw_embedding_mutated"] is False
    assert receipt["cell_by_cell_matrix_constructed"] is False


def test_representation_knn_graphs_are_separate_with_shared_seeded_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[np.ndarray] = []

    def fake_knn(
        values: np.ndarray, *, n_neighbors: int, random_seed: int
    ) -> KNNGraphResult:
        calls.append(values.copy())
        n_cells = len(values)
        directed = np.column_stack(
            [
                (np.arange(n_cells) + offset) % n_cells
                for offset in range(1, n_neighbors + 1)
            ]
        ).astype(np.int64)
        source = np.repeat(np.arange(n_cells), n_neighbors)
        target = directed.reshape(-1)
        pairs = np.unique(np.sort(np.column_stack((source, target)), axis=1), axis=0)
        return KNNGraphResult(
            edge_pairs=np.ascontiguousarray(pairs),
            directed_neighbors=np.ascontiguousarray(directed),
            receipt={"neighbors_sha256": "n", "undirected_edges_sha256": "e"},
        )

    monkeypatch.setattr(analysis, "build_faiss_cosine_knn_graph", fake_knn)
    values = np.eye(9, dtype=np.float32)
    intrinsic_first = analysis._seeded_permuted_knn(
        values, representation="intrinsic", n_neighbors=2, random_seed=20260825
    )
    intrinsic_second = analysis._seeded_permuted_knn(
        values, representation="intrinsic", n_neighbors=2, random_seed=20260825
    )
    contextual = analysis._seeded_permuted_knn(
        values, representation="contextual", n_neighbors=2, random_seed=20260825
    )
    assert np.array_equal(intrinsic_first.edge_pairs, intrinsic_second.edge_pairs)
    assert np.array_equal(
        intrinsic_first.directed_neighbors, intrinsic_second.directed_neighbors
    )
    assert intrinsic_first.receipt["insertion_permutation_sha256"] == (
        intrinsic_second.receipt["insertion_permutation_sha256"]
    )
    assert intrinsic_first.receipt["insertion_permutation_sha256"] == (
        contextual.receipt["insertion_permutation_sha256"]
    )
    assert intrinsic_first.edge_pairs is not contextual.edge_pairs
    assert len(calls) == 3
    assert np.array_equal(calls[0], calls[1])
    assert np.array_equal(calls[0], calls[2])
    assert contextual.receipt[
        "shared_insertion_seed_across_independent_representations"
    ] is True
    assert contextual.receipt["core_labels_consulted_for_insertion_order"] is False


def test_direct_knn_leiden_assignments_are_deterministic() -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("igraph")
    pytest.importorskip("leidenalg")
    values = np.random.default_rng(44).normal(size=(80, 8)).astype(np.float32)
    first = analysis.cluster_direct_representation(
        values,
        representation="intrinsic",
        n_neighbors=5,
        leiden_resolution=1.0,
        random_seed=20260825,
        minimum_mean_recall=0.0,
    )
    second = analysis.cluster_direct_representation(
        values,
        representation="intrinsic",
        n_neighbors=5,
        leiden_resolution=1.0,
        random_seed=20260825,
        minimum_mean_recall=0.0,
    )
    assert np.array_equal(first.edge_pairs, second.edge_pairs)
    assert np.array_equal(first.labels, second.labels)
    assert first.receipt["pca"] is False
    assert first.receipt["mean_center"] is False
    assert first.receipt["knn"]["cell_by_cell_matrix_constructed"] is False


def test_exact_recall_audit_is_batched_and_never_builds_dense_n_by_n() -> None:
    values = np.random.default_rng(5).normal(size=(17, 6)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    similarities = values @ values.T
    np.fill_diagonal(similarities, -np.inf)
    neighbors = np.empty((len(values), 3), dtype=np.int64)
    for row in range(len(values)):
        candidates = np.arange(len(values), dtype=np.int64)
        order = np.lexsort((candidates, -similarities[row]))
        neighbors[row] = order[:3]
    receipt = analysis.exact_neighbor_recall_audit(
        values,
        neighbors,
        n_neighbors=3,
        query_count=11,
        query_batch_size=4,
    )
    assert receipt["recall_at_k_mean"] == 1.0
    assert receipt["maximum_explicit_pairwise_array_shape"] == [4, 17]
    assert receipt["cell_by_cell_matrix_constructed"] is False
    assert receipt["exact_similarity_matrix_persisted"] is False


def test_cpu_and_completed_last_checkpoint_gate_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert analysis.validate_cpu_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="CPU-only"):
        analysis.validate_cpu_device("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert analysis.validate_cuda_hidden() == ""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        analysis.validate_cuda_hidden()

    class RunningRegistry:
        def resolve_run_id(self, value: str) -> str:
            return value

        def show_run(self, value: str) -> dict[str, object]:
            return {
                "run_id": value,
                "status": "running",
                "campaign_id": analysis.UPSTREAM_TRAINING_CAMPAIGN_ID,
                "seed": 0,
                "fold": 0,
                "end_time": None,
            }

    with pytest.raises(ValueError, match="not a completed"):
        analysis.resolve_so1_analysis_inputs(
            registry=RunningRegistry(),  # type: ignore[arg-type]
            paths=_paths(tmp_path),
        )
    with pytest.raises(ValueError, match=r"latest\.ckpt"):
        analysis.resolve_so1_analysis_inputs(
            registry=RunningRegistry(),  # type: ignore[arg-type]
            paths=_paths(tmp_path),
            checkpoint=tmp_path / "checkpoints" / "latest.ckpt",
        )


def test_training_and_analysis_campaign_provenance_are_distinct() -> None:
    assert analysis.UPSTREAM_TRAINING_CAMPAIGN_ID == (
        "cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150"
    )
    assert analysis.ANALYSIS_CAMPAIGN_ID == (
        "cmp_20260827_so1_14core_relative_qkv_embeddings_direct_knn_clustering"
    )
    assert analysis.UPSTREAM_TRAINING_CAMPAIGN_ID != analysis.ANALYSIS_CAMPAIGN_ID


def test_code_provenance_binds_inference_dependencies_and_fails_on_resume_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed = analysis.runtime_code_provenance(project_root=Path.cwd())
    assert {
        "src/spatial_benchmark/models.py",
        "src/spatial_benchmark/relative_qkv_graph_transformer.py",
        "src/spatial_benchmark/relative_qkv_embedding_clustering.py",
        "src/spatial_benchmark/relative_qkv_post_training.py",
        "src/spatial_benchmark/so1_model_embedding_clustering.py",
    }.issubset(observed["relevant_code"])

    output_root = tmp_path / "report"
    output_root.mkdir()
    first = {
        "schema": analysis.CODE_PROVENANCE_SCHEMA,
        "workflow": analysis.OUTPUT_NAMESPACE,
        "command": "analysis --first",
        "relevant_code": {"analysis.py": "a" * 64},
        "gpu_used": False,
    }
    second = {
        **first,
        "command": "analysis --second",
        "relevant_code": {"analysis.py": "b" * 64},
    }
    monkeypatch.setattr(analysis, "runtime_code_provenance", lambda **kwargs: first)
    path = analysis._ensure_code_provenance(
        output_root=output_root, project_root=tmp_path
    )
    assert path.is_file()
    (output_root / "embeddings").mkdir()
    (output_root / "embeddings" / "core_1_receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    monkeypatch.setattr(analysis, "runtime_code_provenance", lambda **kwargs: second)
    with pytest.raises(ValueError, match="mixed-code"):
        analysis._ensure_code_provenance(
            output_root=output_root, project_root=tmp_path
        )


def _synthetic_final_checkpoint_payload() -> dict[str, object]:
    return {
        "checkpoint_schema": analysis.CHECKPOINT_SCHEMA,
        "run_id": analysis.EXPECTED_RUN_ID,
        "campaign_id": analysis.UPSTREAM_TRAINING_CAMPAIGN_ID,
        "model_seed": analysis.EXPECTED_MODEL_SEED,
        "completed_global_epochs": 175,
        "plateau": {"should_stop": True, "final_epoch": 175},
        "completion": None,
    }


def test_final_checkpoint_semantics_accepts_canonical_none_completion() -> None:
    payload = _synthetic_final_checkpoint_payload()
    receipt = analysis._validate_final_checkpoint_semantics(
        payload, expected_run_id=analysis.EXPECTED_RUN_ID
    )
    assert payload["completion"] is None
    assert receipt["completed_global_epochs"] == 175
    assert receipt["plateau"] == {"should_stop": True, "final_epoch": 175}
    assert receipt["completion_field_is_not_a_finality_gate"] is True


@pytest.mark.parametrize(
    "plateau",
    [
        None,
        {},
        {"should_stop": False, "final_epoch": 175},
        {"should_stop": True, "final_epoch": 174},
        {"should_stop": True, "final_epoch": "malformed"},
    ],
)
def test_final_checkpoint_semantics_rejects_malformed_or_mismatched_plateau(
    plateau: object,
) -> None:
    payload = _synthetic_final_checkpoint_payload()
    payload["plateau"] = plateau
    with pytest.raises(ValueError, match="plateau"):
        analysis._validate_final_checkpoint_semantics(
            payload, expected_run_id=analysis.EXPECTED_RUN_ID
        )


def test_panel_order_palette_and_static_output_contract() -> None:
    assert analysis.requested_panel_order() == tuple(range(1, 15))
    specification = analysis.spatial_plot_spec()
    assert specification["grid_shape"] == [3, 5]
    assert specification["panel_order"] == list(range(1, 15))
    assert specification["core_numbers_identifiable"] is True
    assert specification["equal_aspect"] is True
    assert specification["invert_y_axis"] is True
    assert specification["coordinate_units"] == "micrometres"

    contextual = analysis._palette(7, representation="contextual")
    expected = analysis.deterministic_glasbey_palette(7, namespace="contextual")
    assert contextual == {
        f"S1C{index}": expected[f"C{index}"] for index in range(7)
    }
    required = analysis._required_output_files()
    assert (
        "figures/contextual_direct_hl_leiden_resolution_1p0_spatial_14cores.png"
        in required
    )
    assert (
        "figures/intrinsic_direct_h0_leiden_resolution_1p0_spatial_14cores.png"
        in required
    )
    assert "figures/delta_h_l2_spatial_14cores.png" in required
    assert not any(path.endswith(".html") for path in required)


def _synthetic_cores() -> tuple[analysis.SO1CoreEmbeddings, ...]:
    cores: list[analysis.SO1CoreEmbeddings] = []
    for core_number in range(1, 15):
        h0 = np.full((2, 3), float(core_number), dtype=np.float32)
        h_l = h0 + np.asarray(
            [[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]], dtype=np.float32
        )
        delta = h_l - h0
        cores.append(
            analysis.SO1CoreEmbeddings(
                alias=f"SO1-C{core_number:02d}",
                core_number=core_number,
                cell_index=np.arange(2, dtype=np.int64),
                coordinates_um=np.asarray(
                    [[core_number, 0.0], [core_number, 1.0]], dtype=np.float64
                ),
                h0=h0,
                hL=h_l,
                delta_h=delta,
                delta_h_l2=np.linalg.norm(delta.astype(np.float64), axis=1),
            )
        )
    return tuple(cores)


def test_joint_concatenation_requires_all_fourteen_ordered_aligned_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "EXPECTED_HIDDEN_DIM", 3)
    monkeypatch.setattr(analysis, "EXPECTED_TOTAL_CELLS", 28)
    monkeypatch.setattr(
        analysis,
        "EXPECTED_CELL_COUNTS_BY_CORE",
        {core_number: 2 for core_number in range(1, 15)},
    )
    cores = _synthetic_cores()
    h0 = analysis.concatenate_representation(cores, representation="intrinsic")
    h_l = analysis.concatenate_representation(cores, representation="contextual")
    assert h0.shape == h_l.shape == (28, 3)
    assert np.array_equal(h_l - h0, np.concatenate([core.delta_h for core in cores]))
    with pytest.raises(ValueError, match="all 14 SO1 cores"):
        analysis.concatenate_representation(
            tuple(reversed(cores)), representation="contextual"
        )
    with pytest.raises(ValueError, match="all 14 SO1 cores"):
        analysis.concatenate_representation(cores[:-1], representation="intrinsic")


def test_delta_statistics_use_one_global_raw_unclipped_p1_p99_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(analysis, "EXPECTED_TOTAL_CELLS", 28)
    cores = _synthetic_cores()
    raw = np.concatenate([core.delta_h_l2 for core in cores])
    statistics = analysis.delta_norm_statistics(cores)
    assert statistics["count"] == 28
    assert statistics["minimum"] >= 0.0
    assert statistics["p01"] == pytest.approx(float(np.quantile(raw, 0.01)))
    assert statistics["p99"] == pytest.approx(float(np.quantile(raw, 0.99)))
    limits = statistics["plotting_limits"]
    assert limits["vmin_global_p01"] == statistics["p01"]
    assert limits["vmax_global_p99"] == statistics["p99"]
    assert limits["shared_across_all_fourteen_cores"] is True
    assert limits["raw_values_clipped_in_saved_data"] is False
    assert statistics["biological_influence_or_causal_effect"] is False


def test_combined_delta_map_passes_one_common_color_scale_to_all_cores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = []
    for core_number in range(1, 15):
        rows.append(
            {
                "core_number": core_number,
                "x_um": float(core_number),
                "y_um": float(core_number * 2),
                "delta_h_l2": float(core_number) / 10.0,
            }
        )
    frame = analysis.pd.DataFrame(rows)
    observed_norms: list[object] = []
    original_scatter = matplotlib.axes.Axes.scatter

    def recorded_scatter(self: object, *args: object, **kwargs: object) -> object:
        observed_norms.append(kwargs["norm"])
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recorded_scatter)
    monkeypatch.setattr(analysis, "_style_spatial_axis", lambda *_args: None)
    monkeypatch.setattr(
        analysis, "_atomic_save_figure_pair", lambda *_args, **_kwargs: None
    )
    analysis._render_combined_delta_map(
        frame,
        vmin=0.25,
        vmax=1.25,
        png_path=tmp_path / "delta.png",
        pdf_path=tmp_path / "delta.pdf",
        dpi=72,
    )
    assert len(observed_norms) == 14
    assert len({id(value) for value in observed_norms}) == 1
    norm = observed_norms[0]
    assert getattr(norm, "vmin") == 0.25
    assert getattr(norm, "vmax") == 1.25


def test_stage_receipt_self_checksum_rejects_tampering() -> None:
    receipt = analysis._receipt_with_self_hash(
        {
            "schema": analysis.EXTRACTION_SCHEMA,
            "status": "complete",
            "core_order": list(range(1, 15)),
        }
    )
    analysis._verify_self_hash(receipt, label="synthetic extraction receipt")
    tampered = dict(receipt)
    tampered["status"] = "partial"
    with pytest.raises(ValueError, match="self-checksum"):
        analysis._verify_self_hash(
            tampered, label="synthetic extraction receipt"
        )
