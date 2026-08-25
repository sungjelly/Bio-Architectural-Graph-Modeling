from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import spatial_benchmark.relative_qkv_embedding_clustering as analysis
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES, CORE_NUMBERS
from spatial_benchmark.pooled_relative_qkv_training import PooledRelativeQKVCoreBatch


def _core_embeddings(*, cells_per_core: int = 8, dimension: int = 6) -> tuple[analysis.CoreEmbeddings, ...]:
    rng = np.random.default_rng(20260825)
    cores: list[analysis.CoreEmbeddings] = []
    for core_offset, (core_number, alias) in enumerate(
        zip(CORE_NUMBERS, CANCER_ALIASES, strict=True)
    ):
        cell_index = np.arange(cells_per_core, dtype=np.int64)
        coordinates = np.column_stack(
            (
                cell_index.astype(np.float64) * 10.0 + core_offset,
                cell_index.astype(np.float64) * 5.0 + core_offset * 2.0,
            )
        )
        h0 = rng.normal(size=(cells_per_core, dimension)).astype(np.float32)
        h0[:, 0] += (cell_index % 3) * 2.0
        contextual_shift = np.zeros_like(h0)
        contextual_shift[:, 1] = (cell_index % 2) * (core_offset + 1) * 0.4
        contextual_shift[:, 2] = np.sin(cell_index + core_offset)
        hL = np.ascontiguousarray(h0 + contextual_shift, dtype=np.float32)
        delta = np.ascontiguousarray(hL - h0, dtype=np.float32)
        norms = np.linalg.norm(delta.astype(np.float64), axis=1)
        cores.append(
            analysis.CoreEmbeddings(
                alias=alias,
                core_number=core_number,
                cell_index=cell_index,
                coordinates_um=coordinates,
                h0=h0,
                hL=hL,
                delta_h=delta,
                delta_h_l2=norms,
            )
        )
    return tuple(cores)


def test_extraction_staging_uses_complete_expression_zero_mask_and_unchanged_metadata() -> None:
    expression = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    metadata = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    edges = torch.tensor([[1, 2, 3, 0], [0, 0, 1, 2]], dtype=torch.long)
    geometry = torch.arange(280, dtype=torch.float32).reshape(4, 70)
    batch = PooledRelativeQKVCoreBatch(
        alias="CAN-01",
        target_expression=expression,
        edge_index=edges,
        relative_geometry=geometry,
        node_covariates=metadata,
    )
    expression_before = batch.target_expression.clone()
    metadata_before = batch.node_covariates.clone()

    staged = analysis._staged_core_tensors(batch, device=torch.device("cpu"))
    staged_expression, gene_mask, staged_metadata, staged_edges, staged_geometry = staged

    assert torch.equal(staged_expression, expression_before)
    assert gene_mask.dtype == torch.bool
    assert gene_mask.shape == expression.shape
    assert torch.count_nonzero(gene_mask).item() == 0
    assert torch.equal(staged_metadata, metadata_before)
    assert torch.equal(staged_edges, edges)
    assert torch.equal(staged_geometry, geometry)
    assert torch.equal(batch.target_expression, expression_before)
    assert torch.equal(batch.node_covariates, metadata_before)


def test_embedding_delta_and_cell_alignment_cover_all_six_cores() -> None:
    cores = _core_embeddings()
    frame = analysis.build_cell_index_frame(cores)
    h0 = analysis.concatenate_representation(cores, representation="intrinsic")
    hL = analysis.concatenate_representation(cores, representation="contextual")
    statistics = analysis.delta_norm_statistics(cores)

    assert tuple(frame["core_number"].drop_duplicates()) == CORE_NUMBERS
    assert len(frame) == sum(core.n_cells for core in cores)
    assert h0.shape == hL.shape == (len(frame), cores[0].h0.shape[1])
    assert np.array_equal(h0, np.concatenate([core.h0 for core in cores]))
    assert np.array_equal(hL, np.concatenate([core.hL for core in cores]))
    assert np.array_equal(
        frame["delta_h_l2"].to_numpy(),
        np.concatenate([core.delta_h_l2 for core in cores]),
    )
    assert np.isfinite(frame[["x_um", "y_um", "delta_h_l2"]]).all().all()
    assert (frame["delta_h_l2"] >= 0.0).all()
    assert statistics["count"] == len(frame)
    assert statistics["plotting_limits"]["shared_across_all_six_cores"] is True
    assert statistics["plotting_limits"]["raw_values_clipped_in_saved_data"] is False


def test_core_npz_schema_binds_core_number_order_and_exact_delta(tmp_path: Path) -> None:
    core = _core_embeddings(cells_per_core=4)[0]
    path = tmp_path / "core_1_embeddings.npz"
    analysis._write_deterministic_npz(
        path,
        {
            "cell_index": core.cell_index,
            "core_number": np.asarray(core.core_number, dtype=np.int16),
            "coordinates_um": core.coordinates_um,
            "h0": core.h0,
            "hL": core.hL,
            "delta_h": core.delta_h,
            "delta_h_l2": core.delta_h_l2,
        },
    )

    loaded = analysis.load_core_embedding_file(
        path,
        alias=core.alias,
        core_number=core.core_number,
        expected_cells=core.n_cells,
    )

    assert np.array_equal(loaded.cell_index, core.cell_index)
    assert np.array_equal(loaded.coordinates_um, core.coordinates_um)
    assert np.array_equal(loaded.h0, core.h0)
    assert np.array_equal(loaded.hL, core.hL)
    assert np.array_equal(loaded.delta_h, loaded.hL - loaded.h0)
    assert np.isfinite(loaded.delta_h_l2).all()
    assert (loaded.delta_h_l2 >= 0.0).all()


def test_prepared_input_verification_detects_resumed_source_drift(
    tmp_path: Path,
) -> None:
    cohort_dir = tmp_path / "cohort"
    graph_dir = tmp_path / "graph"
    bundle_dir = tmp_path / "bundle"
    cohort_path = cohort_dir / "cores" / "CAN-01.npz"
    graph_path = graph_dir / "cores" / "CAN-01" / "edge_index.npy"
    cohort_path.parent.mkdir(parents=True)
    graph_path.parent.mkdir(parents=True)
    bundle_dir.mkdir()
    cohort_path.write_bytes(b"locked cohort")
    graph_path.write_bytes(b"locked graph")
    checkpoint = bundle_dir / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    inputs = analysis.ResolvedAnalysisInputs(
        run_id=analysis.EXPECTED_RUN_ID,
        project_root=tmp_path,
        checkpoint_path=checkpoint,
        checkpoint_sha256=analysis.sha256_file(checkpoint),
        checkpoint_payload={},
        bundle_path=bundle_dir,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
        cohort_manifest={
            "files": {"cores/CAN-01.npz": analysis.sha256_file(cohort_path)}
        },
        graph_manifest={
            "cores": [
                {
                    "alias": "CAN-01",
                    "files": {"edge_index.npy": analysis.sha256_file(graph_path)},
                }
            ]
        },
        provenance={},
    )

    analysis._verify_prepared_input_files(inputs)
    graph_path.write_bytes(b"drifted graph")
    with pytest.raises(
        analysis.EmbeddingClusterAnalysisError, match="checksum changed"
    ):
        analysis._verify_prepared_input_files(inputs)


def test_joint_pipelines_build_separate_sparse_graphs_and_are_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cores = _core_embeddings()
    observed_inputs: list[np.ndarray] = []
    original = analysis.build_faiss_cosine_knn_graph
    total_cells = sum(core.n_cells for core in cores)

    for allocator_name in ("empty", "zeros", "ones", "full"):
        original_allocator = getattr(np, allocator_name)

        def guarded_allocator(
            shape: object,
            *args: object,
            _original: object = original_allocator,
            _name: str = allocator_name,
            **kwargs: object,
        ) -> np.ndarray:
            if isinstance(shape, tuple) and shape == (total_cells, total_cells):
                pytest.fail(f"dense N x N allocation attempted through numpy.{_name}")
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(np, allocator_name, guarded_allocator)

    def recorded_graph(
        normalized_scores: np.ndarray, *, n_neighbors: int, random_seed: int
    ) -> analysis.KNNGraphResult:
        observed_inputs.append(normalized_scores.copy())
        return original(
            normalized_scores,
            n_neighbors=n_neighbors,
            random_seed=random_seed,
        )

    monkeypatch.setattr(analysis, "build_faiss_cosine_knn_graph", recorded_graph)
    intrinsic_a, intrinsic_receipt_a = analysis.cluster_joint_representation(
        cores,
        representation="intrinsic",
        n_neighbors=5,
        leiden_resolution=1.0,
        pca_components=5,
        random_seed=20260825,
    )
    contextual, contextual_receipt = analysis.cluster_joint_representation(
        cores,
        representation="contextual",
        n_neighbors=5,
        leiden_resolution=1.0,
        pca_components=5,
        random_seed=20260825,
    )
    intrinsic_b, intrinsic_receipt_b = analysis.cluster_joint_representation(
        cores,
        representation="intrinsic",
        n_neighbors=5,
        leiden_resolution=1.0,
        pca_components=5,
        random_seed=20260825,
    )

    assert len(observed_inputs) == 3
    assert not np.shares_memory(observed_inputs[0], observed_inputs[1])
    assert not np.array_equal(observed_inputs[0], observed_inputs[1])
    assert intrinsic_receipt_a["joint_core_order"] == list(CORE_NUMBERS)
    assert contextual_receipt["joint_core_order"] == list(CORE_NUMBERS)
    assert intrinsic_receipt_a["joint_cell_count"] == total_cells
    assert contextual_receipt["joint_cell_count"] == total_cells
    assert intrinsic_receipt_a["pipeline_instance_sha256"] != contextual_receipt[
        "pipeline_instance_sha256"
    ]
    assert np.array_equal(intrinsic_a.labels, intrinsic_b.labels)
    assert intrinsic_receipt_a["knn"]["undirected_edges_sha256"] == (
        intrinsic_receipt_b["knn"]["undirected_edges_sha256"]
    )
    assert intrinsic_receipt_a["pca"]["cell_by_cell_matrix_constructed"] is False
    assert intrinsic_receipt_a["knn"]["cell_by_cell_matrix_constructed"] is False
    assert intrinsic_receipt_a["pca"]["feature_covariance_shape"] == [6, 6]
    assert intrinsic_receipt_a["knn"]["maximum_explicit_pairwise_array_shape"][1] < total_cells
    assert len(contextual.labels) == total_cells


def test_cluster_summary_flags_only_more_than_ninety_percent_from_one_core() -> None:
    cores = _core_embeddings(cells_per_core=20)
    frame = analysis.build_cell_index_frame(cores)
    labels = np.ones(len(frame), dtype=np.int64)
    labels[:19] = 0
    labels[20] = 0

    summary, composition, dominated = analysis.cluster_summary_tables(
        labels, frame, prefix="I"
    )

    cluster_zero = summary.loc[summary["cluster"] == "I0"].iloc[0]
    assert cluster_zero["dominant_core_proportion"] == pytest.approx(19 / 20)
    assert bool(cluster_zero["core_dominated_gt_90pct"]) is True
    assert dominated == ["I0"]
    assert int(summary["size"].sum()) == len(frame)
    assert len(composition) == len(summary) * len(CORE_NUMBERS)


def test_panel_order_palettes_and_delta_scale_are_locked() -> None:
    intrinsic = analysis.deterministic_glasbey_palette(25, namespace="intrinsic")
    contextual = analysis.deterministic_glasbey_palette(25, namespace="contextual")
    repeated = analysis.deterministic_glasbey_palette(25, namespace="intrinsic")
    specification = analysis.spatial_plot_spec(
        intrinsic_palette=intrinsic,
        contextual_palette=contextual,
        delta_vmin=1.25,
        delta_vmax=8.75,
    )

    assert analysis.requested_panel_order() == (1, 9, 13, 15, 21, 23)
    assert specification["grid_shape"] == [2, 3]
    assert specification["panel_order"] == [1, 9, 13, 15, 21, 23]
    assert specification["equal_aspect"] is True
    assert specification["invert_y_axis"] is True
    assert intrinsic == repeated
    assert list(intrinsic) == [f"I{index}" for index in range(25)]
    assert list(contextual) == [f"C{index}" for index in range(25)]
    assert len(set(intrinsic.values())) == 25
    assert len(set(contextual.values())) == 25
    assert specification["palettes_are_separate"] is True
    assert specification["delta_common_scale"] == {
        "vmin": 1.25,
        "vmax": 8.75,
        "shared_across_all_panels": True,
        "colormap": "viridis",
    }


def test_combined_delta_map_reuses_one_normalization_for_every_core(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    for core_number in CORE_NUMBERS:
        rows.append(
            {
                "core_number": core_number,
                "x_um": float(core_number),
                "y_um": float(core_number + 1),
                "delta_h_l2": float(core_number) / 10.0,
            }
        )
    frame = pd.DataFrame(rows)
    observed_norms: list[object] = []

    class Axis:
        def scatter(self, *_args: object, **kwargs: object) -> object:
            observed_norms.append(kwargs["norm"])
            return object()

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

    class Colorbar:
        def set_label(self, *_args: object, **_kwargs: object) -> None:
            return None

    class Figure:
        def colorbar(self, *_args: object, **_kwargs: object) -> Colorbar:
            return Colorbar()

        def suptitle(self, *_args: object, **_kwargs: object) -> None:
            return None

        def subplots_adjust(self, *_args: object, **_kwargs: object) -> None:
            return None

    axes = np.asarray([[Axis(), Axis(), Axis()], [Axis(), Axis(), Axis()]], dtype=object)
    monkeypatch.setattr(analysis, "_style_spatial_axis", lambda *_args: None)
    monkeypatch.setattr(analysis, "_atomic_save_figure_pair", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "matplotlib.pyplot.subplots", lambda *_args, **_kwargs: (Figure(), axes)
    )
    monkeypatch.setattr("matplotlib.pyplot.close", lambda *_args, **_kwargs: None)

    analysis._render_combined_delta_map(
        frame,
        vmin=0.5,
        vmax=2.5,
        png_path=tmp_path / "delta.png",
        pdf_path=tmp_path / "delta.pdf",
        dpi=72,
    )

    assert len(observed_norms) == len(CORE_NUMBERS)
    assert len({id(value) for value in observed_norms}) == 1
    assert observed_norms[0].vmin == 0.5
    assert observed_norms[0].vmax == 2.5


def test_final_manifest_verifies_all_required_checksums_and_detects_tampering(
    tmp_path: Path,
) -> None:
    required = analysis._required_analysis_files()
    for relative in sorted(required):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    checkpoint_sha = "a" * 64
    configuration = {
        "run_id": analysis.EXPECTED_RUN_ID,
        "checkpoint_sha256": checkpoint_sha,
        "extraction_manifest_sha256": "b" * 64,
        "core_order": list(CORE_NUMBERS),
        "joint_clustering": True,
        "n_neighbors": 30,
        "distance_metric": "cosine",
        "leiden_resolution": 1.0,
        "pca_components": 50,
        "random_seed": 20260825,
    }
    extraction = analysis._receipt_with_self_hash(
        {
            "schema": analysis.EXTRACTION_SCHEMA,
            "status": "complete",
            "run_id": analysis.EXPECTED_RUN_ID,
            "core_order": list(CORE_NUMBERS),
            "total_cells": analysis.EXPECTED_TOTAL_CELLS,
            "embedding_dimension": 256,
            "files": {
                f"embeddings/core_{core}_embeddings.npz": analysis._file_record(
                    tmp_path / "embeddings" / f"core_{core}_embeddings.npz"
                )
                for core in CORE_NUMBERS
            },
        }
    )
    extraction_path = tmp_path / "embeddings" / "extraction_manifest.json"
    analysis._atomic_write_json(extraction_path, extraction)
    clustering = analysis._receipt_with_self_hash(
        {
            "schema": analysis.CLUSTERING_SCHEMA,
            "status": "complete",
            "run_id": analysis.EXPECTED_RUN_ID,
            "configuration": configuration,
            "core_order": list(CORE_NUMBERS),
            "total_cells": analysis.EXPECTED_TOTAL_CELLS,
            "independent_representation_pipelines": True,
            "files": {},
        }
    )
    clustering_path = tmp_path / "clustering" / "clustering_manifest.json"
    analysis._atomic_write_json(clustering_path, clustering)
    files = analysis._file_manifest(tmp_path)
    figure_names = {
        *analysis._required_combined_figures(),
        *analysis._required_per_core_figures(),
    }
    manifest = analysis._receipt_with_self_hash(
        {
            "schema": analysis.ANALYSIS_SCHEMA,
            "status": "complete",
            "campaign_id": analysis.CAMPAIGN_ID,
            "run_id": analysis.EXPECTED_RUN_ID,
            "model_seed": analysis.EXPECTED_MODEL_SEED,
            "checkpoint_sha256": checkpoint_sha,
            "preprocessing_checksum": "c" * 64,
            "training_source_commit": "d" * 40,
            "analysis_source_commit": "e" * 40,
            "model_construction": {
                "num_genes": 1_000,
                "node_covariate_dim": 22,
                "hidden_dim": 256,
                "embedding_dim": 256,
                "graph_layers": 4,
                "relative_geometry_dim": 70,
            },
            "input_provenance": {
                "campaign_id": analysis.CAMPAIGN_ID,
                "run_id": analysis.EXPECTED_RUN_ID,
                "model_seed": analysis.EXPECTED_MODEL_SEED,
            },
            "source_core_artifacts": [
                {
                    "core_number": core,
                    "prepared_core_artifact_sha256": "f" * 64,
                    "graph_record_sha256": "0" * 64,
                }
                for core in CORE_NUMBERS
            ],
            "core_order": list(CORE_NUMBERS),
            "cell_counts": {
                str(core): count
                for core, count in analysis.EXPECTED_CELL_COUNTS.items()
            },
            "total_cells": analysis.EXPECTED_TOTAL_CELLS,
            "embedding_shapes": {
                str(core): {"h0": [count, 256], "hL": [count, 256]}
                for core, count in analysis.EXPECTED_CELL_COUNTS.items()
            },
            "clustering_configuration": configuration,
            "cluster_counts": {"intrinsic": 2, "contextual": 3},
            "cluster_size_ranges": {
                "intrinsic": [1, analysis.EXPECTED_TOTAL_CELLS - 1],
                "contextual": [1, analysis.EXPECTED_TOTAL_CELLS - 2],
            },
            "core_dominated_gt_90pct": {
                "intrinsic": ["I1"],
                "contextual": ["C2"],
            },
            "delta_h_l2": {
                "count": analysis.EXPECTED_TOTAL_CELLS,
                "minimum": 0.0,
                "maximum": 9.0,
                "mean": 3.0,
                "median": 2.5,
                "standard_deviation": 1.0,
                "p01": 0.5,
                "p99": 8.0,
                "plotting_limits": {
                    "vmin_global_p01": 0.5,
                    "vmax_global_p99": 8.0,
                    "shared_across_all_six_cores": True,
                    "raw_values_clipped_in_saved_data": False,
                },
            },
            "extraction_manifest_sha256": analysis.sha256_file(extraction_path),
            "clustering_manifest_sha256": analysis.sha256_file(clustering_path),
            "plotting": {
                "combined_figure_count": 6,
                "per_core_figure_count": 36,
                "one_dot_per_cell": True,
                "point_layer_rasterized_in_pdf": True,
                "files": {name: files[name] for name in figure_names},
            },
            "combined_figure_checksums": {
                name: files[name] for name in analysis._required_combined_figures()
            },
            "interpretation": {
                "establishes_cell_type": False,
                "establishes_signaling": False,
                "establishes_biological_influence": False,
                "establishes_causality": False,
                "marker_and_pathology_validation_separate": True,
            },
            "files": files,
        }
    )
    analysis._atomic_write_json(tmp_path / "manifest.json", manifest)

    verified = analysis.verify_embedding_cluster_analysis_bundle(tmp_path)
    assert verified["manifest_content_sha256"] == manifest["manifest_content_sha256"]

    omitted = tmp_path / "figures" / "per_core" / "delta_h_l2_spatial_core_23.pdf"
    omitted.unlink()
    incomplete = dict(manifest)
    incomplete.pop("manifest_content_sha256")
    incomplete_files = dict(incomplete["files"])
    incomplete_files.pop("figures/per_core/delta_h_l2_spatial_core_23.pdf")
    incomplete["files"] = incomplete_files
    incomplete = analysis._receipt_with_self_hash(incomplete)
    analysis._atomic_write_json(tmp_path / "manifest.json", incomplete)
    with pytest.raises(analysis.EmbeddingClusterAnalysisError, match="required files"):
        analysis.verify_embedding_cluster_analysis_bundle(tmp_path)

    omitted_relative = omitted.relative_to(tmp_path).as_posix()
    omitted.write_bytes(omitted_relative.encode("utf-8"))
    analysis._atomic_write_json(tmp_path / "manifest.json", manifest)
    (tmp_path / "README.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(analysis.EmbeddingClusterAnalysisError, match="checksums"):
        analysis.verify_embedding_cluster_analysis_bundle(tmp_path)
