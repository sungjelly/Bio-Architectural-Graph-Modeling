import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from scripts.analysis.so12_hidden_clustering import (
    cpu_neighbor_audit,
    exact_cosine_knn,
    normalized_rows,
    undirected_union,
)
from spatial_benchmark.relative_qkv_embedding_clustering import KNNGraphResult, run_seeded_leiden


def test_exact_cosine_neighbors_exclude_self_and_match_independent_reference():
    raw = np.random.default_rng(43).normal(size=(71, 13)).astype(np.float32)
    before = raw.copy()
    z = normalized_rows(raw)
    nn, distances, receipt = exact_cosine_knn(z, k=7, device="cpu", query_chunk=8)
    audit = cpu_neighbor_audit(z, nn, distances, query_count=len(z))
    assert audit["passed"]
    assert audit["mean_recall_at_k"] == 1
    assert receipt["maximum_explicit_pairwise_array_shape"] == [8, 71]
    assert not receipt["cell_by_cell_matrix_constructed"]
    assert not np.any(nn == np.arange(len(z))[:, None])
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-6)


def test_boundary_ties_choose_global_row_index_and_query_subsets_are_aligned():
    z = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [0, -1]], dtype=np.float32)
    queries = np.array([3, 0, 4])
    nn, distances, receipt = exact_cosine_knn(z, k=2, device="cpu", query_chunk=2, query_indices=queries)
    np.testing.assert_array_equal(nn, [[0, 1], [1, 2], [0, 1]])
    np.testing.assert_array_equal(distances, [[0, 0], [0, 0], [1, 1]])
    assert receipt["boundary_tie_rows"] == 3
    assert cpu_neighbor_audit(z, nn, distances, queries)["passed"]


def test_union_is_unweighted_and_rejects_self_duplicate_or_invalid_neighbors():
    nn = np.array([[1, 2], [0, 2], [1, 3], [1, 2]], dtype=np.int32)
    np.testing.assert_array_equal(undirected_union(nn), [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    for invalid in (np.array([[0], [0]]), np.array([[1, 1], [0, 2], [0, 1]]), np.array([[1], [2]])):
        with pytest.raises(ValueError):
            undirected_union(invalid)
    with pytest.raises(ValueError, match="zero-norm"):
        normalized_rows(np.array([[0, 0], [1, 0]], dtype=np.float32))


def test_leiden_covers_all_vertices_recovers_disconnected_cliques_and_replays():
    # Two graph components with no connecting edge establish a meaningful
    # planted control without asserting a preferred cluster count on real data.
    group_size = 12
    expected = np.repeat([0, 1], group_size)
    nn = np.array([[j for j in range(len(expected)) if expected[j] == expected[i] and j != i]
                   for i in range(len(expected))], dtype=np.int32)
    edges = undirected_union(nn)
    graph = KNNGraphResult(edge_pairs=edges, receipt={})
    first = run_seeded_leiden(graph, n_cells=len(expected), resolution=1.0, random_seed=0)
    replay = run_seeded_leiden(graph, n_cells=len(expected), resolution=1.0, random_seed=0)
    assert first.labels.shape == expected.shape
    assert adjusted_rand_score(expected, first.labels) == 1
    np.testing.assert_array_equal(first.labels, replay.labels)
    assert first.receipt["n_iterations"] == -1
    assert first.receipt["number_of_graph_components"] == 2
