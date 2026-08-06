from __future__ import annotations

import numpy as np
import pytest

import spatial_benchmark.multiscale_graphs as multiscale_graphs
from spatial_benchmark.graphs import build_spatial_graph
from spatial_benchmark.multiscale_graphs import (
    EDGE_ATTRIBUTE_NAMES,
    LOCAL_K_CAP,
    LOCAL_MAX_DISTANCE_UM,
    LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION,
    REGIONAL_K_CAP,
    REGIONAL_MAX_DISTANCE_UM,
    REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM,
    MultiscaleGraphContractError,
    _build_scale_graph,
    _directed_candidate_codes,
    _local_rewiring_feasibility,
    _receiver_codes_from_directed_edges,
    _rewire_local,
    build_true_multiscale_graphs,
    true_graph_receipt,
)


@pytest.fixture(scope="module")
def coordinates() -> np.ndarray:
    rng = np.random.default_rng(401)
    # This density makes k=64 a genuine cap.  In a complete radius graph no
    # absent <=75 um relation exists, so a topology-changing geometric 2-switch
    # is mathematically impossible rather than merely hard to sample.
    return rng.uniform(0.0, 200.0, size=(240, 2))


def _build(
    coordinates: np.ndarray,
    *,
    query_chunk_size: int = 41,
    receiver_chunk_size: int = 37,
    mutual_search_chunk_size: int = 113,
):
    return build_true_multiscale_graphs(
        coordinates,
        query_chunk_size=query_chunk_size,
        receiver_chunk_size=receiver_chunk_size,
        mutual_search_chunk_size=mutual_search_chunk_size,
        workers=1,
    )


@pytest.fixture(scope="module")
def graphs(coordinates: np.ndarray):
    return _build(coordinates)


def _codes(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    return edge_index[1] * n_nodes + edge_index[0]


def _assert_receiver_graph_invariants(graph) -> None:
    edge_index, edge_attributes = graph.concatenate()
    source, receiver = edge_index
    assert edge_index.shape == (2, graph.qc.n_directed_edges)
    assert edge_attributes.shape == (
        graph.qc.n_directed_edges,
        len(EDGE_ATTRIBUTE_NAMES),
    )
    assert np.isfinite(edge_attributes).all()
    assert not np.any(source == receiver)
    codes = _codes(edge_index, graph.n_nodes)
    assert np.all(codes[1:] > codes[:-1])
    assert len(np.unique(codes)) == len(codes)
    reverse = source * graph.n_nodes + receiver
    position = np.searchsorted(codes, reverse)
    assert np.all(position < len(codes))
    np.testing.assert_array_equal(codes[position], reverse)
    assert graph.qc.self_loops == 0
    assert graph.qc.duplicate_directed_edges == 0
    assert graph.qc.directed_edge_pairs_are_symmetric is True
    assert graph.qc.receiver_sorted is True


def test_derivation_is_byte_deterministic(
    coordinates: np.ndarray,
    graphs,
) -> None:
    repeated = _build(
        coordinates,
        query_chunk_size=97,
        receiver_chunk_size=53,
        mutual_search_chunk_size=257,
    )
    assert repeated.checksums == graphs.checksums
    assert true_graph_receipt(repeated) == true_graph_receipt(graphs)
    for first, second in (
        (graphs.local, repeated.local),
        (graphs.regional, repeated.regional),
    ):
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


def test_graphs_are_symmetric_receiver_sorted_and_simple(graphs) -> None:
    for graph in (graphs.local, graphs.regional):
        _assert_receiver_graph_invariants(graph)
        for checksum in graph.checksums.to_dict().values():
            if checksum is not None:
                assert len(checksum) == 64
    assert graphs.qc.all_graphs_symmetric is True
    assert graphs.qc.all_graphs_receiver_sorted is True
    assert graphs.qc.all_graphs_loop_and_duplicate_free is True


def test_fixed_scales_are_disjoint_and_obey_distance_intervals(
    coordinates: np.ndarray,
    graphs,
) -> None:
    local_edges, _ = graphs.local.concatenate()
    regional_edges, _ = graphs.regional.concatenate()
    local_distance = np.linalg.norm(
        coordinates[local_edges[1]] - coordinates[local_edges[0]],
        axis=1,
    )
    regional_distance = np.linalg.norm(
        coordinates[regional_edges[1]] - coordinates[regional_edges[0]],
        axis=1,
    )
    assert graphs.local.k_cap == LOCAL_K_CAP
    assert graphs.regional.k_cap == REGIONAL_K_CAP
    assert local_distance.max() <= LOCAL_MAX_DISTANCE_UM
    assert regional_distance.min() > REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
    assert regional_distance.max() <= REGIONAL_MAX_DISTANCE_UM
    local_codes = _codes(local_edges, graphs.local.n_nodes)
    regional_codes = _codes(regional_edges, graphs.local.n_nodes)
    assert np.intersect1d(local_codes, regional_codes).size == 0
    assert graphs.qc.local_regional_overlap_directed_edges == 0
    assert graphs.qc.local_regional_disjoint is True


def test_true_topologies_match_existing_capped_radius_builder(
    coordinates: np.ndarray,
    graphs,
) -> None:
    local_reference = build_spatial_graph(
        coordinates,
        k=LOCAL_K_CAP,
        radius_um=LOCAL_MAX_DISTANCE_UM,
        symmetry="mutual",
    )
    regional_reference = build_spatial_graph(
        coordinates,
        k=REGIONAL_K_CAP,
        radius_um=REGIONAL_MAX_DISTANCE_UM,
        min_distance_um=np.nextafter(
            REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM,
            np.inf,
        ),
        symmetry="mutual",
    )
    local_edges, _ = graphs.local.concatenate()
    regional_edges, _ = graphs.regional.concatenate()
    np.testing.assert_array_equal(
        np.sort(_codes(local_edges, len(coordinates))),
        np.sort(
            _codes(local_reference.edge_index, len(coordinates))
        ),
    )
    np.testing.assert_array_equal(
        np.sort(_codes(regional_edges, len(coordinates))),
        np.sort(
            _codes(regional_reference.edge_index, len(coordinates))
        ),
    )


def test_annulus_is_strict_at_75_and_inclusive_at_300() -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [75.0, 0.0],
            [75.000001, 0.0],
            [300.0, 0.0],
            [300.000001, 0.0],
        ],
        dtype=np.float64,
    )
    regional, _ = _directed_candidate_codes(
        coordinates,
        k_cap=REGIONAL_K_CAP,
        minimum_distance_exclusive_um=(
            REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
        ),
        maximum_distance_um=REGIONAL_MAX_DISTANCE_UM,
        query_chunk_size=2,
        workers=1,
    )
    source_zero_targets = set(
        (regional[regional // len(coordinates) == 0] % len(coordinates)).tolist()
    )
    assert 1 not in source_zero_targets
    assert 2 in source_zero_targets
    assert 3 in source_zero_targets
    assert 4 not in source_zero_targets


def _square_matching_local():
    coordinates: list[list[float]] = []
    pairs: list[list[int]] = []
    for square in range(20):
        x_offset = 100.0 * (square % 5)
        y_offset = 100.0 * (square // 5)
        first = 4 * square
        coordinates.extend(
            [
                [x_offset, y_offset],
                [x_offset + 10.0, y_offset],
                [x_offset, y_offset + 10.0],
                [x_offset + 10.0, y_offset + 10.0],
            ]
        )
        pairs.extend(
            [
                [first, first + 1],
                [first + 2, first + 3],
            ]
        )
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    true_pairs = np.asarray(pairs, dtype=np.int64)
    directed = np.concatenate(
        [true_pairs, true_pairs[:, ::-1]],
        axis=0,
    ).T
    receiver_codes = _receiver_codes_from_directed_edges(
        directed,
        n_nodes=len(coordinate_array),
    )
    local = _build_scale_graph(
        receiver_codes,
        name="local",
        topology_kind="synthetic_square_matching",
        coordinates=coordinate_array,
        k_cap=LOCAL_K_CAP,
        minimum_distance_um=0.0,
        minimum_distance_inclusive=True,
        maximum_distance_um=LOCAL_MAX_DISTANCE_UM,
        candidate_counts=None,
        directed_candidates_sha256=None,
        edge_attribute_mean=np.zeros(len(EDGE_ATTRIBUTE_NAMES)),
        edge_attribute_scale=np.ones(len(EDGE_ATTRIBUTE_NAMES)),
        edge_standardization_source="true_local",
        receiver_chunk_size=17,
    )
    return coordinate_array, true_pairs, receiver_codes, local


def test_rewiring_preserves_degree_distance_and_strong_final_change() -> None:
    coordinates, true_pairs, receiver_codes, local = (
        _square_matching_local()
    )
    rewired, qc = _rewire_local(
        local,
        receiver_codes,
        coordinates,
        receiver_chunk_size=19,
        seed=271828,
        swaps_per_edge=0.5,
        distance_bins=8,
        max_attempts_per_swap=100,
        maximum_relative_mean_distance_change=0.10,
    )
    repeated, repeated_qc = _rewire_local(
        local,
        receiver_codes,
        coordinates,
        receiver_chunk_size=23,
        seed=271828,
        swaps_per_edge=0.5,
        distance_bins=8,
        max_attempts_per_swap=100,
        maximum_relative_mean_distance_change=0.10,
    )
    assert repeated.checksums.graph_sha256 == rewired.checksums.graph_sha256
    assert repeated_qc == qc
    true_edges, _ = local.concatenate()
    rewired_edges, rewired_attributes = rewired.concatenate()
    repeated_edges, repeated_attributes = repeated.concatenate()
    np.testing.assert_array_equal(repeated_edges, rewired_edges)
    np.testing.assert_array_equal(repeated_attributes, rewired_attributes)
    true_degree = np.bincount(
        true_edges[1],
        minlength=local.n_nodes,
    )
    rewired_degree = np.bincount(
        rewired_edges[1],
        minlength=local.n_nodes,
    )
    np.testing.assert_array_equal(true_degree, rewired_degree)
    assert not np.array_equal(true_edges, rewired_edges)
    assert qc.successful_swaps > 0
    assert qc.topology_changed is True
    assert qc.degree_preserved_exactly is True
    assert qc.removed_true_undirected_edges > 0
    assert (
        qc.removed_true_undirected_edges
        == qc.added_rewired_undirected_edges
    )
    assert qc.reintroduced_original_undirected_edges == 0
    assert qc.no_original_relations_reintroduced is True
    assert qc.final_relation_replacement_fraction >= (
        LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION
    )
    assert qc.final_relation_replacement_passed is True
    assert qc.feasibility.passed is True
    assert qc.feasibility.degree_constraints_relaxed is True
    original_relation_set = {tuple(pair) for pair in true_pairs.tolist()}
    rewired_relation_set = {
        tuple(pair)
        for pair in rewired_edges[:, rewired_edges[0] < rewired_edges[1]].T.tolist()
    }
    assert len(original_relation_set - rewired_relation_set) == (
        qc.removed_true_undirected_edges
    )
    assert qc.true_distance_bin_counts == qc.rewired_distance_bin_counts
    assert qc.distance_bin_counts_preserved_exactly is True
    assert qc.relative_edge_distance_mean_change <= (
        qc.maximum_relative_mean_distance_change
    )
    assert qc.distance_tolerance_passed is True

    np.testing.assert_array_equal(
        rewired.edge_attribute_mean,
        local.edge_attribute_mean,
    )
    np.testing.assert_array_equal(
        rewired.edge_attribute_scale,
        local.edge_attribute_scale,
    )
    assert rewired.edge_standardization_source == "true_local"
    assert qc.raw_geometry_recomputed_from_coordinates is True

    # Independently reconstruct raw distance and direction from the standardized
    # columns to verify that rewired attributes came from their new endpoints.
    raw = (
        rewired_attributes.astype(np.float64)
        * local.edge_attribute_scale
        + local.edge_attribute_mean
    )
    delta = coordinates[rewired_edges[1]] - coordinates[rewired_edges[0]]
    distance = np.linalg.norm(delta, axis=1)
    np.testing.assert_allclose(raw[:, 0], distance, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        raw[:, 3],
        delta[:, 0] / LOCAL_MAX_DISTANCE_UM,
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        raw[:, 4],
        delta[:, 1] / LOCAL_MAX_DISTANCE_UM,
        rtol=2e-5,
        atol=2e-5,
    )


def test_infeasible_distance_bins_fail_before_swaps(
    coordinates: np.ndarray,
    graphs,
) -> None:
    true_edges, _ = graphs.local.concatenate()
    true_pairs = true_edges[
        :,
        true_edges[0] < true_edges[1],
    ].T
    feasibility = _local_rewiring_feasibility(
        true_pairs,
        coordinates,
        distance_bins=6,
        target_swaps=len(true_pairs),
        maximum_relative_mean_distance_change=0.10,
    )
    assert (
        feasibility.degree_relaxed_optimistic_final_replacement_upper_bound
        == pytest.approx(1080 / 6409)
    )
    assert feasibility.distance_bin_capacity_can_reach_threshold is False
    assert feasibility.degree_constraints_relaxed is True
    assert feasibility.passed is False

    with pytest.raises(
        MultiscaleGraphContractError,
        match="optimistically infeasible",
    ):
        multiscale_graphs.build_multiscale_graphs(
            coordinates,
            query_chunk_size=41,
            receiver_chunk_size=37,
            mutual_search_chunk_size=113,
            workers=1,
            rewiring_swaps_per_edge=1.0,
            rewiring_distance_bins=6,
        )


def test_feasibility_rejects_swap_budget_and_best_case_mean_drift() -> None:
    coordinates, true_pairs, _, _ = _square_matching_local()
    insufficient_budget = _local_rewiring_feasibility(
        true_pairs,
        coordinates,
        distance_bins=8,
        target_swaps=1,
        maximum_relative_mean_distance_change=0.10,
    )
    assert insufficient_budget.swap_budget_can_reach_threshold is False
    assert insufficient_budget.passed is False

    elongated = coordinates.copy()
    top_nodes = np.flatnonzero(np.arange(len(elongated)) % 4 >= 2)
    elongated[top_nodes, 1] += 10.0
    excessive_mean_drift = _local_rewiring_feasibility(
        true_pairs,
        elongated,
        distance_bins=8,
        target_swaps=len(true_pairs),
        maximum_relative_mean_distance_change=0.10,
    )
    assert (
        excessive_mean_drift.distance_bin_capacity_can_reach_threshold
        is True
    )
    optimistic_drift = (
        excessive_mean_drift
        .degree_relaxed_optimistic_minimum_relative_mean_distance_increase_at_threshold
    )
    assert (
        optimistic_drift == pytest.approx(0.5)
    )
    assert (
        excessive_mean_drift.mean_distance_tolerance_can_reach_threshold
        is False
    )
    assert excessive_mean_drift.passed is False


def test_many_reported_swaps_cannot_mask_weak_final_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinates, true_pairs, receiver_codes, local = (
        _square_matching_local()
    )
    weak_pairs = true_pairs.copy()
    weak_pairs[0] = [0, 2]
    weak_pairs[1] = [1, 3]

    def weak_rewire(*args, **kwargs):
        return weak_pairs, {
            "rewire_target_swaps": len(true_pairs),
            "rewire_successful_swaps": len(true_pairs),
            "rewire_attempts": len(true_pairs),
            "distance_bin_boundaries_um": [10.0],
            "rewire_reintroduced_original_undirected_edges": 0,
            "rewire_monotonic_removed_original_undirected_edges": 2,
        }

    monkeypatch.setattr(
        multiscale_graphs,
        "_degree_distance_rewire_pairs",
        weak_rewire,
    )
    with pytest.raises(
        MultiscaleGraphContractError,
        match=r"final relation-replacement gate: .*successful_swaps=40",
    ):
        _rewire_local(
            local,
            receiver_codes,
            coordinates,
            receiver_chunk_size=19,
            seed=271828,
            swaps_per_edge=1.0,
            distance_bins=8,
            max_attempts_per_swap=100,
            maximum_relative_mean_distance_change=0.10,
        )


def test_true_only_builder_never_invokes_rewiring(
    coordinates: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def prohibited(*args, **kwargs):
        raise AssertionError("rewiring must not be invoked")

    monkeypatch.setattr(multiscale_graphs, "_rewire_local", prohibited)
    graphs = build_true_multiscale_graphs(
        coordinates,
        query_chunk_size=41,
        receiver_chunk_size=37,
        mutual_search_chunk_size=113,
        workers=1,
    )
    receipt = true_graph_receipt(graphs)
    assert not hasattr(graphs, "rewired_local")
    assert receipt["fixed_contract"]["rewired_graph_constructed"] is False
    assert "rewired_local" not in receipt
    assert set(receipt["bundle_checksums"]) == {
        "local_graph_sha256",
        "regional_graph_sha256",
        "bundle_sha256",
    }
    assert receipt["bundle_qc"]["local_regional_disjoint"] is True
