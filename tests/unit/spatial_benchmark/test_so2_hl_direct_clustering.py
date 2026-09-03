from __future__ import annotations

import numpy as np
import pytest

import spatial_benchmark.so2_hl_direct_clustering as direct
from spatial_benchmark.cli import build_parser
from spatial_benchmark.relative_qkv_embedding_clustering import (
    KNNGraphResult,
    LeidenResult,
    _array_sha256,
    _file_record,
    _receipt_with_self_hash,
    deterministic_glasbey_palette,
)


def _exact_neighbors(values: np.ndarray, k: int) -> np.ndarray:
    similarities = values @ values.T
    np.fill_diagonal(similarities, -np.inf)
    indices = np.arange(len(values), dtype=np.int64)
    result = np.empty((len(values), k), dtype=np.int64)
    for row in range(len(values)):
        result[row] = np.lexsort((indices, -similarities[row]))[:k]
    return result


def test_cosine_setup_uses_fresh_buffer_without_centering_or_pca() -> None:
    raw = np.asarray(
        [
            [4.0, 1.0, 0.5],
            [2.0, 3.0, 0.25],
            [6.0, 2.0, 1.5],
            [1.0, 5.0, 2.0],
        ],
        dtype=np.float32,
    )
    before = raw.copy()
    before_hash = _array_sha256("joint_contextual_hL", raw)

    normalized, receipt = direct.l2_normalize_hl_for_cosine(raw)

    assert normalized is not raw
    assert not np.shares_memory(normalized, raw)
    np.testing.assert_array_equal(raw, before)
    assert _array_sha256("joint_contextual_hL", raw) == before_hash
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)
    np.testing.assert_allclose(
        normalized,
        raw / np.linalg.norm(raw, axis=1, keepdims=True),
        rtol=1e-6,
        atol=1e-7,
    )
    assert receipt["pca"] is False
    assert receipt["mean_center"] is False
    assert receipt["l2_normalize_for_cosine"] is True
    assert receipt["representation_learning_or_reduction"] is False
    assert receipt["raw_hL_mutated"] is False
    assert receipt["cell_by_cell_matrix_constructed"] is False


def test_cosine_setup_rejects_zero_norm_cells() -> None:
    raw = np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="zero-norm|undefined"):
        direct.l2_normalize_hl_for_cosine(raw)


def test_production_guard_requires_cuda_to_be_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        direct.validate_cuda_hidden()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        direct.validate_cuda_hidden()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert direct.validate_cuda_hidden() == ""


def test_sampled_exact_recall_audit_is_batched_and_never_n_by_n() -> None:
    generator = np.random.default_rng(20260825)
    raw = generator.normal(size=(41, 7)).astype(np.float32)
    normalized, _ = direct.l2_normalize_hl_for_cosine(raw)
    exact = _exact_neighbors(normalized, 4)

    receipt = direct.exact_neighbor_recall_audit(
        normalized,
        exact,
        n_neighbors=4,
        query_count=13,
        query_batch_size=3,
    )

    assert receipt["recall_at_k_mean"] == pytest.approx(1.0)
    assert receipt["recall_at_k_minimum"] == pytest.approx(1.0)
    assert receipt["query_count_evaluated"] == 13
    assert receipt["maximum_explicit_pairwise_array_shape"] == [3, 41]
    assert receipt["cell_by_cell_matrix_constructed"] is False
    assert receipt["exact_similarity_matrix_persisted"] is False


def test_direct_pipeline_passes_unit_hl_directions_directly_to_knn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.asarray(
        [
            [9.0, 1.0, 0.2],
            [7.0, 3.0, 0.5],
            [4.0, 8.0, 1.0],
            [1.0, 9.0, 2.0],
            [-3.0, 4.0, 7.0],
            [2.0, -5.0, 8.0],
        ],
        dtype=np.float32,
    )
    raw_before = raw.copy()
    observed: dict[str, np.ndarray] = {}

    def fake_knn(
        values: np.ndarray, *, n_neighbors: int, random_seed: int
    ) -> KNNGraphResult:
        observed["knn_input"] = values.copy()
        neighbors = _exact_neighbors(values, n_neighbors)
        source = np.repeat(np.arange(len(values)), n_neighbors)
        target = neighbors.reshape(-1)
        pairs = np.unique(
            np.sort(np.column_stack((source, target)), axis=1), axis=0
        ).astype(np.int64)
        return KNNGraphResult(
            edge_pairs=pairs,
            receipt={
                "n_neighbors": n_neighbors,
                "cell_by_cell_matrix_constructed": False,
            },
            directed_neighbors=neighbors,
        )

    def fake_leiden(
        graph: KNNGraphResult,
        *,
        n_cells: int,
        resolution: float,
        random_seed: int,
    ) -> LeidenResult:
        labels = np.arange(n_cells, dtype=np.int64) % 2
        return LeidenResult(
            labels=labels,
            receipt={
                "labels_sha256": _array_sha256("sorted_leiden_labels", labels),
                "resolution": resolution,
                "random_seed": random_seed,
            },
        )

    monkeypatch.setattr(direct, "build_faiss_cosine_knn_graph", fake_knn)
    monkeypatch.setattr(direct, "run_seeded_leiden", fake_leiden)
    result = direct.direct_cosine_knn_leiden(
        raw,
        n_neighbors=2,
        leiden_resolution=1.0,
        random_seed=20260825,
    )

    expected = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    np.testing.assert_allclose(observed["knn_input"], expected, atol=1e-7)
    np.testing.assert_array_equal(raw, raw_before)
    assert result.receipt["pipeline_kind"] == direct.PIPELINE_KIND
    assert result.receipt["pca"] is False
    assert result.receipt["mean_center"] is False
    assert result.receipt["l2_normalize_for_cosine"] is True
    assert result.receipt["raw_hL_mutated"] is False
    assert result.receipt["knn_exact_recall_audit"]["recall_at_k_mean"] == 1.0


def test_direct_configuration_and_palette_are_distinct_from_pca_contextual() -> None:
    configuration = direct._clustering_configuration(
        n_neighbors=30,
        leiden_resolution=1.0,
        random_seed=20260825,
    )
    palette = direct.deterministic_direct_palette(24)
    prior_palette = deterministic_glasbey_palette(24, namespace="contextual")

    assert configuration["pipeline_kind"] == "direct_hl_cosine_knn_leiden"
    assert configuration["pca"] is False
    assert configuration["mean_center"] is False
    assert configuration["l2_normalize_for_cosine"] is True
    assert "pca_components" not in configuration
    assert configuration["source_pca_scores_reused"] is False
    assert configuration["source_knn_graph_reused"] is False
    assert configuration["dense_cell_by_cell_matrix_constructed"] is False
    assert list(palette) == [f"D{index}" for index in range(24)]
    assert len(set(palette.values())) == len(palette)
    assert palette["D0"] != prior_palette["C0"]
    assert direct.OUTPUT_ANALYSIS_NAMESPACE != direct.SOURCE_ANALYSIS_NAMESPACE


def test_clustering_verifier_uses_the_leiden_label_hash_domain(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    monkeypatch.setattr(direct, "EXPECTED_TOTAL_CELLS", 4)
    monkeypatch.setattr(direct, "SO2_CORE_NUMBERS", (15, 16))
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    labels_path = tmp_path / direct.LABELS_RELATIVE_PATH
    table_path = tmp_path / direct.TABLE_RELATIVE_PATH
    labels_path.parent.mkdir(parents=True)
    table_path.parent.mkdir(parents=True)
    np.save(labels_path, labels, allow_pickle=False)
    pd.DataFrame(
        {
            "core_number": [15, 15, 16, 16],
            "contextual_direct_cluster_number": labels,
        }
    ).to_parquet(table_path, index=False)
    configuration = {"fixture": True}
    receipt = _receipt_with_self_hash(
        {
            "schema": direct.CLUSTERING_SCHEMA,
            "status": "complete",
            "run_id": "fixture-run",
            "source_extraction_manifest_sha256": "fixture-extraction",
            "configuration": configuration,
            "core_order": [15, 16],
            "total_cells": 4,
            "pipeline": {
                "pipeline_kind": direct.PIPELINE_KIND,
                "pca": False,
                "mean_center": False,
                "l2_normalize_for_cosine": True,
                "knn_exact_recall_audit": {
                    "cell_by_cell_matrix_constructed": False,
                    "mean_recall_acceptance_passed": True,
                },
                "leiden": {
                    "labels_sha256": _array_sha256(
                        "sorted_leiden_labels", labels
                    )
                },
            },
            "files": {
                direct.LABELS_RELATIVE_PATH.as_posix(): _file_record(labels_path),
                direct.TABLE_RELATIVE_PATH.as_posix(): _file_record(table_path),
            },
        }
    )

    direct._verify_clustering_manifest(
        output_root=tmp_path,
        receipt=receipt,
        run_id="fixture-run",
        source_extraction_sha256="fixture-extraction",
        configuration=configuration,
    )


def test_direct_hl_cli_exposes_only_knn_and_leiden_parameters() -> None:
    arguments = build_parser().parse_args(["analyze-so2-hl-direct-clusters"])

    assert arguments.device == "cpu"
    assert arguments.n_neighbors == 30
    assert arguments.leiden_resolution == 1.0
    assert arguments.random_seed == 20260825
    assert not hasattr(arguments, "pca_components")

    viewer = build_parser().parse_args(["render-so2-hl-direct-interactive"])
    assert viewer.command_name == "render-so2-hl-direct-interactive"
