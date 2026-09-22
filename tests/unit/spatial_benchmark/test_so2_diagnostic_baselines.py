"""Scientific contracts of the SO2 constant and neighbor references."""

import numpy as np
import pytest
from scipy import sparse

from spatial_benchmark.so2_diagnostic_baselines import (
    huber_constant,
    local_adjacency,
    observed_neighbor_mean,
)


def _risk(values, weights, constant, delta=1.0):
    error = np.abs(constant - np.asarray(values))
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return np.average(loss, weights=weights)


def test_sparse_distribution_huber_optimum_differs_from_squared_loss_mean():
    values, weights = np.array([0.0, 10.0]), np.array([0.9, 0.1])
    optimum = huber_constant(values, weights)
    assert optimum == pytest.approx(0.1 / 0.9, abs=1e-12)
    assert np.average(values, weights=weights) == 1
    assert abs(np.dot(weights, np.clip(optimum - values, -1, 1))) < 1e-12
    assert _risk(values, weights, optimum) < _risk(values, weights, 0)
    assert _risk(values, weights, optimum) < _risk(values, weights, 1)
    assert np.average((values - optimum)**2, weights=weights) > np.average((values - 1)**2, weights=weights)


def test_huber_root_balances_weighted_score_and_minimizes_risk():
    values = np.array([-3, -0.5, 0, 0.2, 0.8, 4, 12], dtype=float)
    weights = np.array([3, 8, 23, 11, 7, 4, 2], dtype=float)
    optimum = huber_constant(values, weights, delta=0.6)
    score = np.average(np.clip(optimum - values, -0.6, 0.6), weights=weights)
    assert abs(score) < 1e-12
    assert _risk(values, weights, optimum, 0.6) <= min(
        _risk(values, weights, c, 0.6) for c in [0, np.average(values, weights=weights), optimum - 0.1, optimum + 0.1]
    )
    assert huber_constant([8, 100], [1, 0]) == 8
    assert huber_constant([0, 10], [9e307, 1e307]) == pytest.approx(1 / 9)


def test_directional_means_include_observed_zero_and_ignore_edge_magnitude():
    # Receiver 0 sees source 1 (observed zero) and source 2 (observed four).
    graph = sparse.csr_matrix(([7.0, 1.0, 1.0], ([0, 0, 1], [1, 2, 2])), shape=(3, 3))
    values = np.array([[99.0], [0.0], [4.0]])
    prediction, fallback = observed_neighbor_mean(graph, values, np.array([[False], [True], [True]]))
    np.testing.assert_allclose(prediction[:, 0], [2, 4, 2])
    np.testing.assert_array_equal(fallback[:, 0], [False, False, True])


def test_hidden_values_cannot_change_neighbors_or_either_fallback():
    graph = sparse.csr_matrix(([1, 1], ([0, 1], [1, 2])), shape=(3, 3))
    values = np.array([[4.0, 90.0, 3.0], [5.0, 8.0, 4.0], [6.0, 80.0, 5.0]])
    observed = np.array([[False, False, False], [False, True, False], [True, False, False]])
    first, fallback = observed_neighbor_mean(graph, values, observed)
    changed = values.copy()
    changed[~observed] = np.linspace(-1e6, 1e6, np.count_nonzero(~observed))
    second, second_fallback = observed_neighbor_mean(graph, changed, observed)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(fallback, second_fallback)
    np.testing.assert_allclose(first, np.tile([6, 8, 0], (3, 1)))
    assert fallback[0, 0]  # No visible local values; observed-core fallback.
    assert fallback[:, 2].all()  # No visible values anywhere; zero fallback.


def test_self_edges_rejected_and_duplicate_neighbors_not_double_counted():
    with pytest.raises(ValueError, match="self edges"):
        observed_neighbor_mean(sparse.eye(2, format="csr"), np.ones((2, 1)), np.ones((2, 1), bool))
    graph = sparse.csr_matrix((np.ones(3), np.array([1, 1, 2]), np.array([0, 3, 3, 3])), shape=(3, 3))
    prediction, _ = observed_neighbor_mean(graph, [[0], [0], [6]], np.ones((3, 1), bool))
    assert prediction[0, 0] == 3


def test_local_graph_respects_cutoff_self_exclusion_direction_and_stable_ties():
    coords = np.array([[0.0, 0], [1, 0], [-1, 0], [2.5, 0], [100, 0]])
    graph = local_adjacency(coords, k=1, radius=1.5)
    assert graph[0, 1] == 1  # Equal distance tie chooses lower source index.
    assert graph[0, 2] == 0
    assert graph[3, 1] == 1  # Inclusive distance boundary.
    assert graph[1, 3] == 0  # Graph stays directed.
    assert graph[4].nnz == 0
    assert not graph.diagonal().any()
    np.testing.assert_array_equal(graph.toarray(), local_adjacency(coords, k=1, radius=1.5).toarray())


def test_local_graph_handles_colocated_ties_empty_graph_and_single_node():
    graph = local_adjacency(np.zeros((8, 2)), k=2, radius=0)
    assert graph[7].indices.tolist() == [0, 1]
    assert graph[0].indices.tolist() == [1, 2]
    assert np.all(graph.getnnz(axis=1) == 2)
    assert local_adjacency(np.empty((0, 2))).shape == (0, 0)
    assert local_adjacency([[0, 0]]).nnz == 0


@pytest.mark.parametrize("values,weights", [([], []), ([0], [0]), ([0], [-1]), ([np.nan], [1]), ([0], [np.inf]), ([[0]], [[1]])])
def test_invalid_huber_inputs_fail(values, weights):
    with pytest.raises(ValueError):
        huber_constant(values, weights)


def test_invalid_geometry_and_observation_inputs_fail():
    with pytest.raises(ValueError):
        local_adjacency([[0, np.nan]])
    with pytest.raises(ValueError):
        local_adjacency([[0, 1]], k=0)
    graph = sparse.csr_matrix((2, 2))
    with pytest.raises(ValueError, match="boolean"):
        observed_neighbor_mean(graph, [[1], [2]], [[1], [0]])
    with pytest.raises(ValueError, match="finite"):
        observed_neighbor_mean(graph, [[1], [np.inf]], np.ones((2, 1), bool))
