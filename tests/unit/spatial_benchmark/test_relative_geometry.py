from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark.relative_geometry import (
    DEFAULT_RBF_COUNT,
    RELATIVE_GEOMETRY_DIM,
    LocalOrientationTensors,
    RelativeGeometryContractError,
    checksum_invariant_relative_geometry,
    gaussian_radial_basis,
    invariant_relative_features,
    iter_invariant_relative_feature_shards,
    local_orientation_tensors,
    radial_stratified_knn,
    smooth_cutoff_envelope,
)


def _sorted_old_index_edges(
    edge_index: np.ndarray, new_to_old: np.ndarray
) -> np.ndarray:
    source = new_to_old[edge_index[0]]
    receiver = new_to_old[edge_index[1]]
    order = np.lexsort((source, receiver))
    return np.vstack([source[order], receiver[order]])


def _edge_feature_map(
    edge_index: np.ndarray, features: np.ndarray
) -> dict[tuple[int, int], np.ndarray]:
    return {
        (int(source), int(receiver)): features[index]
        for index, (source, receiver) in enumerate(edge_index.T.tolist())
    }


def test_radial_stratified_graph_is_deterministic_canonical_and_covered() -> None:
    rng = np.random.default_rng(908)
    angles = rng.uniform(0.0, 2.0 * np.pi, 280)
    radii = np.concatenate(
        [
            rng.uniform(5.0, 49.0, 70),
            rng.uniform(51.0, 149.0, 80),
            rng.uniform(151.0, 299.0, 65),
            rng.uniform(301.0, 499.0, 65),
        ]
    )
    coordinates = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii])

    small_chunks = radial_stratified_knn(
        coordinates, receiver_query_chunk_size=7
    )
    large_chunks = radial_stratified_knn(
        coordinates, receiver_query_chunk_size=1_000
    )

    np.testing.assert_array_equal(small_chunks.edge_index, large_chunks.edge_index)
    assert small_chunks.checksums == large_chunks.checksums
    assert small_chunks.qc.nominal_pre_symmetrization_degree == 200
    assert small_chunks.qc.pre_sym_in_degree_max <= 200
    assert small_chunks.qc.in_degree_max >= small_chunks.qc.pre_sym_in_degree_max
    assert small_chunks.qc.self_loops == 0
    assert small_chunks.qc.cross_group_edges == 0
    assert small_chunks.qc.directed_edge_pairs_are_symmetric
    assert small_chunks.qc.receiver_major_canonical_order
    assert small_chunks.qc.coverage_eligible_cells > 0
    assert small_chunks.qc.eligible_long_range_coverage_fraction == 1.0
    assert small_chunks.qc.long_range_coverage_gate_passed
    source, receiver = small_chunks.edge_index
    assert np.all(receiver[1:] >= receiver[:-1])
    same_receiver = receiver[1:] == receiver[:-1]
    assert np.all(source[1:][same_receiver] > source[:-1][same_receiver])
    assert set(zip(source.tolist(), receiver.tolist())) == set(
        zip(receiver.tolist(), source.tolist())
    )


def test_radial_shell_refill_uses_nearest_remaining_and_never_exceeds_range() -> None:
    # Fewer than 200 total candidates forces all reserved-shell capacity to be
    # refilled.  The out-of-range point must remain absent.
    coordinates = np.column_stack(
        [np.concatenate([[0.0], np.arange(1.0, 181.0), [501.0]]), np.zeros(182)]
    )
    graph = radial_stratified_knn(coordinates, receiver_query_chunk_size=11)
    incoming_to_zero = graph.edge_index[0, graph.edge_index[1] == 0]
    assert set(range(1, 181)).issubset(incoming_to_zero.tolist())
    assert 181 not in incoming_to_zero
    distance = np.linalg.norm(
        coordinates[graph.edge_index[0]] - coordinates[graph.edge_index[1]], axis=1
    )
    assert np.all(distance <= 500.0)


def test_group_labels_prevent_cross_group_edges_even_when_coordinates_overlap() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 0.0], [10.0, 0.0]], dtype=float
    )
    labels = np.asarray(["left", "left", "right", "right"])
    graph = radial_stratified_knn(coordinates, group_labels=labels)
    source, receiver = graph.edge_index
    assert np.all(labels[source] == labels[receiver])
    assert graph.qc.n_groups == 2
    assert graph.qc.cross_group_edges == 0


@pytest.mark.parametrize(
    "transform",
    [
        np.asarray([[0.0, -1.0], [1.0, 0.0]]),
        np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_relative_features_are_rotation_and_reflection_invariant(
    transform: np.ndarray,
) -> None:
    rng = np.random.default_rng(440)
    coordinates = rng.normal(size=(90, 2)) * 90.0
    graph = radial_stratified_knn(coordinates, receiver_query_chunk_size=9)
    orientation = local_orientation_tensors(
        coordinates, receiver_query_chunk_size=9
    )
    baseline = invariant_relative_features(
        coordinates, graph.edge_index, orientation, output_dtype=np.float64
    )

    transformed_coordinates = coordinates @ transform.T + np.asarray([1200.0, -730.0])
    transformed_graph = radial_stratified_knn(
        transformed_coordinates, receiver_query_chunk_size=13
    )
    transformed_orientation = local_orientation_tensors(
        transformed_coordinates, receiver_query_chunk_size=13
    )
    transformed_features = invariant_relative_features(
        transformed_coordinates,
        transformed_graph.edge_index,
        transformed_orientation,
        output_dtype=np.float64,
    )

    np.testing.assert_array_equal(graph.edge_index, transformed_graph.edge_index)
    expected_tensors = np.einsum(
        "ab,nbc,dc->nad", transform, orientation.tensors, transform
    )
    np.testing.assert_allclose(
        transformed_orientation.tensors, expected_tensors, rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(transformed_features, baseline, rtol=2e-11, atol=2e-11)


def test_translation_invariance_and_node_permutation_equivariance() -> None:
    rng = np.random.default_rng(90210)
    # More than 201 nodes makes the radial quotas active rather than reducing
    # permutation equivariance to the trivial all-neighbors case.
    coordinates = rng.uniform(-180.0, 180.0, size=(225, 2))
    graph = radial_stratified_knn(coordinates, receiver_query_chunk_size=8)
    orientation = local_orientation_tensors(coordinates, receiver_query_chunk_size=8)
    features = invariant_relative_features(
        coordinates, graph.edge_index, orientation, output_dtype=np.float64
    )

    translated = coordinates + np.asarray([912.25, -713.75])
    translated_orientation = local_orientation_tensors(translated)
    translated_features = invariant_relative_features(
        translated, graph.edge_index, translated_orientation, output_dtype=np.float64
    )
    np.testing.assert_allclose(
        translated_orientation.tensors, orientation.tensors, rtol=3e-12, atol=3e-12
    )
    np.testing.assert_allclose(translated_features, features, rtol=3e-12, atol=3e-12)

    new_to_old = rng.permutation(len(coordinates))
    permuted_coordinates = coordinates[new_to_old]
    permuted_graph = radial_stratified_knn(
        permuted_coordinates, receiver_query_chunk_size=5
    )
    permuted_orientation = local_orientation_tensors(
        permuted_coordinates, receiver_query_chunk_size=5
    )
    permuted_features = invariant_relative_features(
        permuted_coordinates,
        permuted_graph.edge_index,
        permuted_orientation,
        output_dtype=np.float64,
    )
    mapped_edges = _sorted_old_index_edges(permuted_graph.edge_index, new_to_old)
    np.testing.assert_array_equal(mapped_edges, graph.edge_index)
    old_feature_map = _edge_feature_map(graph.edge_index, features)
    unsorted_mapped_edges = np.vstack(
        [
            new_to_old[permuted_graph.edge_index[0]],
            new_to_old[permuted_graph.edge_index[1]],
        ]
    )
    for edge, value in _edge_feature_map(
        unsorted_mapped_edges, permuted_features
    ).items():
        np.testing.assert_allclose(value, old_feature_map[edge], rtol=2e-12, atol=2e-12)


def test_edge_permutation_preserves_exact_feature_alignment() -> None:
    rng = np.random.default_rng(70)
    coordinates = rng.normal(size=(35, 2)) * 40.0
    graph = radial_stratified_knn(coordinates)
    orientation = local_orientation_tensors(coordinates)
    baseline = invariant_relative_features(coordinates, graph.edge_index, orientation)
    permutation = rng.permutation(graph.edge_index.shape[1])
    permuted = invariant_relative_features(
        coordinates, graph.edge_index[:, permutation], orientation
    )
    np.testing.assert_array_equal(permuted, baseline[permutation])


def test_rbf_is_finite_over_range_and_cutoff_is_exact() -> None:
    distances = np.asarray([0.0, 1.0, 250.0, 499.0, 500.0])
    rbf = gaussian_radial_basis(distances)
    assert rbf.shape == (len(distances), DEFAULT_RBF_COUNT)
    assert np.isfinite(rbf).all()
    assert np.all(rbf[:-1].sum(axis=1) > 0.0)
    np.testing.assert_array_equal(rbf[-1], np.zeros(DEFAULT_RBF_COUNT))
    envelope = smooth_cutoff_envelope(distances)
    assert envelope[0] == 1.0
    assert envelope[-1] == 0.0
    with pytest.raises(RelativeGeometryContractError, match="clipping"):
        gaussian_radial_basis(np.asarray([500.000001]))


def test_isotropic_neighborhood_has_near_zero_orientation() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    ring = np.column_stack([80.0 * np.cos(angles), 80.0 * np.sin(angles)])
    coordinates = np.vstack([np.zeros((1, 2)), ring])
    orientation = local_orientation_tensors(coordinates)
    np.testing.assert_allclose(orientation.tensors[0], np.zeros((2, 2)), atol=2e-14)
    assert orientation.anisotropy[0] < 3e-14
    assert orientation.selected_neighbor_counts[0] == 64


def test_axial_director_sign_cannot_change_invariant_features() -> None:
    coordinates = np.asarray([[0.0, 0.0], [40.0, 30.0]])
    edge_index = np.asarray([[1, 0], [0, 1]], dtype=np.int64)
    director = np.asarray([0.6, 0.8])
    opposite = -director
    tensor = np.outer(director, director) - 0.5 * np.eye(2)
    opposite_tensor = np.outer(opposite, opposite) - 0.5 * np.eye(2)
    np.testing.assert_array_equal(tensor, opposite_tensor)

    def orientation_for(value: np.ndarray) -> LocalOrientationTensors:
        tensors = np.stack([value, value])
        anisotropy = np.sqrt(
            2.0 * np.einsum("nab,nba->n", tensors, tensors)
        )
        return LocalOrientationTensors(
            tensors=tensors,
            anisotropy=anisotropy,
            selected_neighbor_counts=np.asarray([2, 2], dtype=np.int64),
            radius_um=150.0,
            neighbor_cap=64,
            sigma_um=75.0,
            minimum_neighbors=2,
            insufficient_neighbor_cells=0,
            checksum_sha256="0" * 64,
        )

    positive = invariant_relative_features(
        coordinates, edge_index, orientation_for(tensor)
    )
    negative = invariant_relative_features(
        coordinates, edge_index, orientation_for(opposite_tensor)
    )
    np.testing.assert_array_equal(positive, negative)


def test_receiver_shards_never_split_receiver_and_checksum_is_chunk_invariant() -> None:
    rng = np.random.default_rng(130)
    coordinates = rng.uniform(0.0, 600.0, size=(115, 2))
    graph = radial_stratified_knn(coordinates)
    orientation = local_orientation_tensors(coordinates)
    shards = tuple(
        iter_invariant_relative_feature_shards(
            coordinates,
            graph.edge_index,
            orientation,
            receiver_chunk_size=9,
            max_edges_per_chunk=60,
        )
    )
    assert sum(shard.n_edges for shard in shards) == graph.edge_index.shape[1]
    assert all(shard.features.shape[1] == RELATIVE_GEOMETRY_DIM for shard in shards)
    receiver_to_shard: dict[int, int] = {}
    for shard_index, shard in enumerate(shards):
        for receiver in np.unique(shard.edge_index[1]).tolist():
            assert receiver not in receiver_to_shard
            receiver_to_shard[receiver] = shard_index

    fine = checksum_invariant_relative_geometry(
        coordinates,
        graph.edge_index,
        orientation,
        receiver_chunk_size=3,
        max_edges_per_chunk=25,
    )
    coarse = checksum_invariant_relative_geometry(
        coordinates,
        graph.edge_index,
        orientation,
        receiver_chunk_size=1000,
        max_edges_per_chunk=1_000_000,
    )
    assert fine == coarse


def test_insufficient_orientation_neighbors_are_recorded_as_zero() -> None:
    coordinates = np.asarray([[0.0, 0.0], [1000.0, 0.0], [2000.0, 0.0]])
    orientation = local_orientation_tensors(coordinates)
    assert orientation.insufficient_neighbor_cells == 3
    np.testing.assert_array_equal(orientation.tensors, np.zeros((3, 2, 2)))
    np.testing.assert_array_equal(orientation.anisotropy, np.zeros(3))
