from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

import spatial_benchmark.full_core as full_core
from spatial_benchmark.artifacts import ArtifactContractError
from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS
from spatial_benchmark.full_core import (
    EDGE_ATTRIBUTE_NAMES,
    FullCoreContractError,
    RadiusGuardError,
    build_exact_mutual_knn_graph,
    load_and_refit_full_core,
)
from spatial_benchmark.graphs import build_spatial_graph
from spatial_benchmark.splits import TrainOnlyPreprocessor


def _synthetic_prepared_payload() -> tuple[
    dict[str, object],
    dict[str, np.ndarray],
    np.ndarray,
]:
    counts = np.asarray(
        [
            [0, 2, 1],
            [1, 4, 0],
            [2, 1, 3],
            [4, 0, 2],
            [8, 3, 1],
            [16, 7, 5],
        ],
        dtype=np.int32,
    )
    measured_metadata = (
        np.arange(counts.shape[0] * len(ALLOWED_METADATA_COLUMNS), dtype=np.float64)
        .reshape(counts.shape[0], -1)
        / 7.0
        + 1.0
    )
    measured_metadata[1, 0] = np.nan
    measured_metadata[2, 5] = np.nan
    source_labels = np.asarray(["train"] * 3 + ["val", "test", "test"])
    source_preprocessor = TrainOnlyPreprocessor().fit(
        counts,
        measured_metadata,
        source_labels,
    )
    source_nodes = source_preprocessor.transform(counts, measured_metadata)
    gene_names = ("GeneA", "GeneB", "GeneC")
    manifest: dict[str, object] = {
        "artifact_kind": "normal_true_tissue_spatial_benchmark_preparation",
        "artifact_id": "prepared-synthetic",
        "selection": {"restricted_identifiers_emitted": False},
        "features": {
            "gene_names": list(gene_names),
            "measured_metadata_names": list(ALLOWED_METADATA_COLUMNS),
            "model_covariate_names": list(source_nodes.metadata_names),
        },
        "preprocessing": {
            "metadata_transform": "median imputation, log1p, standardization",
        },
        "files": {"prepared_data.npz": "a" * 64},
    }
    arrays = {
        "expression_counts": counts,
        "node_covariates": source_nodes.metadata,
        "coordinates_um": np.column_stack(
            [
                np.arange(len(counts), dtype=np.float64),
                np.arange(len(counts), dtype=np.float64) ** 2,
            ]
        ),
        "macroblock_ids": np.asarray(
            ["block-a", "block-a", "block-b", "block-b", "block-c", "block-c"]
        ),
        "expression_mean": source_preprocessor.expression_mean_,
        "expression_scale": source_preprocessor.expression_scale_,
        "metadata_median": source_preprocessor.metadata_median_,
        "metadata_mean": source_preprocessor.metadata_mean_,
        "metadata_scale": source_preprocessor.metadata_scale_,
        "metadata_missing_indicator_indices": (
            source_preprocessor.missing_indicator_indices_
        ),
        # These routing identifiers may be present in the verified source, but
        # the full-core result must not expose them.
        "cell_ID": np.arange(91, 97, dtype=np.int32),
        "fov": np.asarray([44, 44, 45, 45, 46, 46], dtype=np.int32),
    }
    return manifest, arrays, measured_metadata


def test_full_core_loader_refits_all_nodes_and_drops_routing_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, arrays, measured_metadata = _synthetic_prepared_payload()
    calls: list[tuple[Path, bool]] = []

    def fake_verified_loader(
        path: str | Path,
        *,
        load_arrays: bool,
    ) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
        calls.append((Path(path), load_arrays))
        return manifest, arrays, {}

    monkeypatch.setattr(full_core, "load_prepared_artifact", fake_verified_loader)
    result = load_and_refit_full_core("verified-prepared")
    repeated = load_and_refit_full_core("verified-prepared")

    expected = TrainOnlyPreprocessor().fit(
        arrays["expression_counts"],
        measured_metadata,
        np.full(len(measured_metadata), "train"),
    ).transform(arrays["expression_counts"], measured_metadata)
    np.testing.assert_allclose(
        result.target_expression,
        expected.expression,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        result.node_covariates,
        expected.metadata,
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        result.macroblock_ids,
        arrays["macroblock_ids"],
    )
    assert result.gene_names == ("GeneA", "GeneB", "GeneC")
    assert result.metadata_names[-2:] == (
        "Area__missing",
        "Mean.PanCK__missing",
    )
    assert result.preprocessing_qc.n_metadata_missing_values == 2
    assert result.preprocessing_qc.fit_scope == "all nodes (transductive)"
    assert result.preprocessing_qc.source_metadata_roundtrip_max_abs_error < 1e-7
    assert result.preprocessing_qc.protected_identifier_arrays_returned is False
    assert result.checksums == repeated.checksums
    assert len(result.checksums.preprocessing_sha256) == 64
    assert calls == [
        (Path("verified-prepared"), True),
        (Path("verified-prepared"), True),
    ]
    public_fields = {item.name for item in fields(result)}
    assert "cell_ID" not in public_fields
    assert "fov" not in public_fields
    assert "slide" not in public_fields


def test_full_core_loader_uses_verified_artifact_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected_loader(
        path: str | Path,
        *,
        load_arrays: bool,
    ) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
        raise ArtifactContractError("checksum mismatch")

    monkeypatch.setattr(full_core, "load_prepared_artifact", rejected_loader)
    with pytest.raises(ArtifactContractError, match="checksum mismatch"):
        load_and_refit_full_core("corrupt")

    manifest, arrays, _ = _synthetic_prepared_payload()
    manifest["features"] = {
        **manifest["features"],  # type: ignore[index]
        "model_covariate_names": list(ALLOWED_METADATA_COLUMNS),
    }
    monkeypatch.setattr(
        full_core,
        "load_prepared_artifact",
        lambda path, *, load_arrays: (manifest, arrays, {}),
    )
    with pytest.raises(FullCoreContractError, match="reversible statistics"):
        load_and_refit_full_core("invalid-schema")


def _brute_force_mutual_edges(coordinates: np.ndarray, k: int) -> np.ndarray:
    distance = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=2,
    )
    np.fill_diagonal(distance, np.inf)
    node_index = np.arange(len(coordinates))
    directed = np.zeros((len(coordinates), len(coordinates)), dtype=bool)
    for source in range(len(coordinates)):
        order = np.lexsort((node_index, distance[source]))
        directed[source, order[:k]] = True
    mutual = directed & directed.T
    source, receiver = np.nonzero(mutual)
    order = np.lexsort((source, receiver))
    return np.vstack([source[order], receiver[order]]).astype(np.int64)


def test_exact_mutual_graph_matches_brute_force_and_existing_geometry() -> None:
    # A regular grid deliberately creates kth-distance ties.  The expected
    # graph therefore also checks deterministic node-index tie resolution.
    coordinates = np.asarray(
        [(x, y) for y in range(5) for x in range(5)],
        dtype=np.float64,
    )
    graph = build_exact_mutual_knn_graph(
        coordinates,
        k=4,
        radius_guard_um=10.0,
        query_chunk_size=6,
        receiver_chunk_size=7,
        mutual_search_chunk_size=11,
    )
    edge_index, edge_attributes = graph.concatenate()
    expected_edges = _brute_force_mutual_edges(coordinates, k=4)
    np.testing.assert_array_equal(edge_index, expected_edges)

    reference = build_spatial_graph(
        coordinates,
        k=4,
        radius_um=10.0,
        symmetry="mutual",
        rbf_bins=8,
    )
    reference_order = np.lexsort(
        (reference.edge_index[0], reference.edge_index[1])
    )
    np.testing.assert_array_equal(
        edge_index,
        reference.edge_index[:, reference_order],
    )
    reference_raw = reference.edge_attr[reference_order]
    reconstructed_raw = (
        edge_attributes.astype(np.float64) * graph.edge_attribute_scale
        + graph.edge_attribute_mean
    )
    np.testing.assert_allclose(
        reconstructed_raw,
        reference_raw,
        rtol=2e-6,
        atol=2e-6,
    )
    expected_mean = reference_raw.astype(np.float64).mean(axis=0)
    expected_raw_scale = reference_raw.astype(np.float64).std(axis=0, ddof=0)
    expected_scale = np.where(expected_raw_scale > 1e-8, expected_raw_scale, 1.0)
    np.testing.assert_allclose(graph.edge_attribute_mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(graph.edge_attribute_scale, expected_scale, atol=1e-12)
    assert graph.edge_attribute_names == EDGE_ATTRIBUTE_NAMES
    assert edge_attributes.shape[1] == 17

    source, receiver = edge_index
    encoded = source * len(coordinates) + receiver
    reverse = receiver * len(coordinates) + source
    np.testing.assert_array_equal(np.sort(encoded), np.sort(reverse))
    assert not np.any(source == receiver)
    assert len(np.unique(encoded)) == len(encoded)
    assert graph.qc.directed_edge_pairs_are_symmetric is True
    assert graph.qc.receiver_sorted is True
    assert graph.qc.self_loops == 0
    assert graph.qc.duplicate_directed_edges == 0
    assert graph.qc.edge_attribute_count == 17
    assert graph.qc.edge_standardization_scope == "all retained directed edges"


def test_receiver_shards_and_checksums_are_partition_independent() -> None:
    coordinates = np.random.default_rng(20260725).uniform(
        0.0,
        100.0,
        size=(41, 2),
    )
    first = build_exact_mutual_knn_graph(
        coordinates,
        k=7,
        radius_guard_um=200.0,
        query_chunk_size=5,
        receiver_chunk_size=6,
        mutual_search_chunk_size=17,
    )
    second = build_exact_mutual_knn_graph(
        coordinates,
        k=7,
        radius_guard_um=200.0,
        query_chunk_size=13,
        receiver_chunk_size=11,
        mutual_search_chunk_size=31,
    )
    first_edges, first_attributes = first.concatenate()
    second_edges, second_attributes = second.concatenate()
    np.testing.assert_array_equal(first_edges, second_edges)
    np.testing.assert_array_equal(first_attributes, second_attributes)
    np.testing.assert_array_equal(
        first.edge_attribute_mean,
        second.edge_attribute_mean,
    )
    np.testing.assert_array_equal(
        first.edge_attribute_scale,
        second.edge_attribute_scale,
    )
    assert first.qc == second.qc
    assert first.checksums == second.checksums

    expected_start = 0
    for shard in first.iter_shards():
        assert shard.receiver_start == expected_start
        expected_start = shard.receiver_stop
        receiver = shard.edge_index[1]
        assert np.all(receiver[:-1] <= receiver[1:])
        assert np.all(
            (receiver >= shard.receiver_start)
            & (receiver < shard.receiver_stop)
        )
        assert len(shard.checksum_sha256) == 64
    assert expected_start == len(coordinates)
    assert all(len(value) == 64 for value in first.checksums.to_dict().values())


def test_radius_guard_validates_without_truncating_exact_candidates() -> None:
    coordinates = np.column_stack(
        [np.arange(8, dtype=np.float64) * 10.0, np.zeros(8)]
    )
    with pytest.raises(RadiusGuardError, match="would truncate exact kNN"):
        build_exact_mutual_knn_graph(
            coordinates,
            k=2,
            radius_guard_um=9.0,
            query_chunk_size=3,
            receiver_chunk_size=3,
        )

    graph = build_exact_mutual_knn_graph(
        coordinates,
        k=2,
        radius_guard_um=20.0,
        query_chunk_size=3,
        receiver_chunk_size=3,
    )
    assert graph.qc.candidate_kth_distance_max_um == 20.0
    assert graph.qc.radius_guard_margin_um == 0.0
    assert graph.qc.n_directed_candidates == len(coordinates) * 2

    with pytest.raises(FullCoreContractError, match=r"\[1, N - 1\]"):
        build_exact_mutual_knn_graph(
            coordinates,
            k=len(coordinates),
            radius_guard_um=20.0,
        )
