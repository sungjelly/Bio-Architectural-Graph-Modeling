from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark.same_gene_jacobian import SameGeneJacobianError
from spatial_benchmark.same_gene_robustness import (
    _conflict_free_permutation,
    build_robust_neighbor_aggregates,
    panel_log_cp10k,
)


def _toy_expression(nodes: int) -> np.ndarray:
    values = np.arange(nodes * 3, dtype=np.float32).reshape(nodes, 3)
    return values / 10.0


def test_component_graph_adds_cross_fov_edges_but_never_crosses_group() -> None:
    expression = _toy_expression(18)
    coordinates = np.column_stack(
        (
            np.asarray(
                [0, 20, 40, 100, 120, 140, 42, 62, 82, 160, 180, 200]
                + [500, 520, 540, 600, 620, 640],
                dtype=np.float64,
            ),
            np.zeros(18, dtype=np.float64),
        )
    )
    fov = np.asarray([1] * 6 + [2] * 6 + [3] * 6)
    groups = np.asarray([101] * 12 + [102] * 6)

    within = build_robust_neighbor_aggregates(
        expression,
        coordinates,
        fov,
        groups,
        slide="SO_1",
        partition_mode="within_fov",
        minimum_matched_degree=1,
    )
    component = build_robust_neighbor_aggregates(
        expression,
        coordinates,
        fov,
        groups,
        slide="SO_1",
        partition_mode="within_geometry_component",
        minimum_matched_degree=1,
    )

    assert within.audit["near_cross_fov_edge_count"] == 0
    assert int(component.audit["near_cross_fov_edge_count"]) > 0
    # The third FOV belongs to a distinct component despite sharing one slide.
    assert int(component.audit["partition_count"]) == 2


def test_qc_induced_graph_excludes_inactive_receivers_and_sources() -> None:
    expression = _toy_expression(10)
    coordinates = np.column_stack(
        (np.arange(10, dtype=np.float64) * 20.0, np.zeros(10))
    )
    fov = np.ones(10, dtype=np.int16)
    groups = np.full(10, 101, dtype=np.int16)
    active = np.asarray([True, True, False, True, True, True, True, True, True, True])
    result = build_robust_neighbor_aggregates(
        expression,
        coordinates,
        fov,
        groups,
        slide="SO_1",
        partition_mode="within_fov",
        active_nodes=active,
        minimum_matched_degree=1,
    )

    assert result.near_degree[2] == 0
    assert not result.matched_eligible[2]
    assert np.all(result.near_mean[2] == 0)
    assert 2 not in set(result.source_permutation[active].tolist())


def test_permutation_is_deterministic_bijective_and_degree_preserving() -> None:
    nodes = 19
    expression = _toy_expression(nodes)
    coordinates = np.column_stack(
        (np.arange(nodes, dtype=np.float64) * 8.0, np.zeros(nodes))
    )
    fov = np.ones(nodes, dtype=np.int16)
    groups = np.full(nodes, 101, dtype=np.int16)
    first = build_robust_neighbor_aggregates(
        expression,
        coordinates,
        fov,
        groups,
        slide="SO_1",
        partition_mode="within_fov",
        minimum_matched_degree=1,
        permutation_seed=91,
    )
    replay = build_robust_neighbor_aggregates(
        expression,
        coordinates,
        fov,
        groups,
        slide="SO_1",
        partition_mode="within_fov",
        minimum_matched_degree=1,
        permutation_seed=91,
    )
    changed_seed = build_robust_neighbor_aggregates(
        expression,
        coordinates,
        fov,
        groups,
        slide="SO_1",
        partition_mode="within_fov",
        minimum_matched_degree=1,
        permutation_seed=92,
    )

    assert np.array_equal(first.source_permutation, replay.source_permutation)
    assert np.array_equal(first.permuted_near_mean, replay.permuted_near_mean)
    assert not np.array_equal(first.source_permutation, changed_seed.source_permutation)
    assert sorted(first.source_permutation.tolist()) == list(range(nodes))
    assert np.all(first.source_permutation != np.arange(nodes))
    assert np.array_equal(first.permuted_near_degree, first.near_degree)
    assert first.audit["permutation_receiver_collisions"] == 0


def test_conflict_free_null_uses_minimum_fixed_sources_when_derangement_impossible() -> None:
    expression = _toy_expression(2)
    coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    fov = np.ones(2, dtype=np.int16)
    groups = np.full(2, 101, dtype=np.int16)
    result = build_robust_neighbor_aggregates(
        expression,
        coordinates,
        fov,
        groups,
        slide="SO_1",
        partition_mode="within_fov",
        minimum_matched_degree=1,
    )

    assert np.array_equal(result.source_permutation, np.arange(2))
    assert result.audit["permutation_fixed_source_count"] == 2
    assert result.audit["permutation_fixed_sources_by_fov"] == {"1": 2}
    assert result.audit["permutation_receiver_collisions"] == 0
    assert np.array_equal(result.permuted_near_degree, result.near_degree)


def test_exact_permutation_fallback_is_maximum_change_and_fail_closed() -> None:
    # Sources 0 and 1 must exchange destinations, forcing source 2 to remain
    # fixed. No zero-fixed-point matching exists, but one fixed point is exact.
    forbidden = [{2}, {2}, set()]
    permutation = _conflict_free_permutation(3, forbidden, seed=17)

    assert sorted(permutation.tolist()) == [0, 1, 2]
    assert int(np.sum(permutation == np.arange(3))) == 1
    assert all(
        int(permutation[source]) not in forbidden[source]
        for source in range(3)
    )
    with pytest.raises(SameGeneJacobianError, match="no receiver-collision-free"):
        _conflict_free_permutation(2, [{0, 1}, set()], seed=17)


def test_panel_log_cp10k_matches_hand_calculation_and_zero_policy() -> None:
    counts = np.asarray([[1, 1, 2], [0, 0, 0], [5, 0, 5]], dtype=np.float32)
    transformed, audit = panel_log_cp10k(np.log1p(counts))
    expected = np.log1p(
        np.asarray(
            [[2500, 2500, 5000], [0, 0, 0], [5000, 0, 5000]],
            dtype=np.float64,
        )
    ).astype(np.float32)

    assert np.allclose(transformed, expected, rtol=0, atol=1e-6)
    assert audit["zero_panel_total_cells"] == 1
    assert float(audit["maximum_integer_roundtrip_error"]) < 1e-5
