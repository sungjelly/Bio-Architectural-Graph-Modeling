"""Scientific boundaries and planted-signal controls for the SO12 encoders."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "analysis"))
from so12_embedding_methods import (
    kernel_fit_transform,
    knn_graph,
    leiden_labels,
    spatial_fit_transform,
    spatial_neighbors,
)
from so12_embedding_report import _read_primary_labels


def _fixture():
    rng = np.random.default_rng(140)
    cores = np.repeat(np.array(["a", "b", "c"]), 24)
    train = cores != "c"
    # Every core deliberately has overlapping coordinate extents.
    xy = rng.uniform(0, 20, (len(cores), 2))
    z = rng.normal(size=(len(cores), 8)).astype(np.float32)
    config = {"seed": 3, "n_components": 4, "kernel_landmarks": 24,
              "spatial_k": 5, "spatial_radius_um": 12.0, "spatial_weight": 0.3}
    return z, xy, cores, train, config


def test_spatial_neighbors_isolate_cores_and_keep_radius_and_self_contracts():
    # Duplicate positions must not let self IDs survive self removal; distinct
    # cores sharing exactly the same coordinates must remain disconnected.
    xy = np.array([[0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [100, 0]], dtype=float)
    cores = np.array(["a", "a", "a", "b", "b", "b", "a"])
    w, indices, distances, receipt = spatial_neighbors(
        xy, cores, {"spatial_k": 3, "spatial_radius_um": 2.0})
    valid = indices >= 0
    row = np.broadcast_to(np.arange(len(xy))[:, None], indices.shape)[valid]
    assert np.all(cores[row] == cores[indices[valid]])
    assert np.all(row != indices[valid])
    assert np.all(distances[valid] <= 2.0)
    assert receipt["isolated_cells_self_fallback"] == 1
    assert np.all(indices[-1] == -1)
    assert w[-1, -1] == 1.0
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), 1, atol=1e-6)


def test_spatial_geometry_is_invariant_to_rotation_and_translation():
    z, xy, cores, train, config = _fixture()
    angle = 0.71
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = xy @ rotation + np.array([800, -300])
    w1, idx1, _, _ = spatial_neighbors(xy, cores, config)
    w2, idx2, _, _ = spatial_neighbors(transformed, cores, config)
    np.testing.assert_array_equal(idx1, idx2)
    np.testing.assert_allclose(w1.toarray(), w2.toarray(), atol=2e-6)
    embedding1, _, _ = spatial_fit_transform(z, w1, train, config)
    embedding2, _, _ = spatial_fit_transform(z, w2, train, config)
    np.testing.assert_allclose(pdist(embedding1), pdist(embedding2), atol=1e-5, rtol=1e-5)


def test_validation_core_cannot_change_spatial_fitted_parameters():
    z, xy, cores, train, config = _fixture()
    w, _, _, _ = spatial_neighbors(xy, cores, config)
    baseline, _, parameters1 = spatial_fit_transform(z, w, train, config)
    changed = z.copy()
    changed[~train] = changed[~train] * 100 + 900
    altered, _, parameters2 = spatial_fit_transform(changed, w, train, config)
    for name in parameters1:
        np.testing.assert_array_equal(parameters1[name], parameters2[name], err_msg=name)
    np.testing.assert_array_equal(baseline[train], altered[train])
    assert not np.allclose(baseline[~train], altered[~train])


def test_kernel_landmarks_and_fit_ignore_validation_expression():
    z, _, cores, train, config = _fixture()
    baseline, details, parameters1 = kernel_fit_transform(z, train, cores, config)
    changed = z.copy()
    changed[~train] *= 100
    altered, _, parameters2 = kernel_fit_transform(changed, train, cores, config)
    assert details["landmarks_train_only"]
    assert train[parameters1["landmark_indices"]].all()
    for name in parameters1:
        np.testing.assert_array_equal(parameters1[name], parameters2[name], err_msg=name)
    np.testing.assert_array_equal(baseline[train], altered[train])
    assert not np.allclose(baseline[~train], altered[~train])
    assert np.isfinite(altered).all()


def test_kernel_transform_replays_held_out_cells_from_saved_parameters():
    z, _, cores, train, config = _fixture()
    embedding, _, parameters = kernel_fit_transform(z, train, cores, config)
    x = z[~train].astype(np.float64)
    squared_distance = ((x[:, None, :] - parameters["landmarks"][None, :, :]) ** 2).sum(axis=2)
    kernel = np.exp(-float(parameters["gamma"]) * squared_distance)
    centered = (kernel - kernel.mean(axis=1, keepdims=True)
                - parameters["kernel_column_mean"][None, :]
                + parameters["kernel_grand_mean"])
    replay = centered @ parameters["projection"]
    np.testing.assert_allclose(embedding[~train], replay, rtol=1e-5, atol=1e-6)


def test_knn_and_leiden_recover_four_planted_groups():
    rng = np.random.default_rng(12)
    truth = np.repeat(np.arange(4), 20)
    centers = np.eye(4, dtype=np.float32) * 20
    z = centers[truth] + rng.normal(scale=0.02, size=(len(truth), 4)).astype(np.float32)
    graph, neighbors, receipt = knn_graph(z, {"seed": 0, "threads": 1, "knn_k": 19})
    assert receipt["mean_knn_recall"] >= 0.99
    assert np.all(truth[neighbors] == truth[:, None])
    assert (graph != graph.T).nnz == 0
    assert not graph.diagonal().any()
    first, _ = leiden_labels(graph, 1.0, 0)
    second, _ = leiden_labels(graph, 1.0, 1)
    assert adjusted_rand_score(truth, first) == 1.0
    assert adjusted_rand_score(first, second) == 1.0


def test_report_refuses_misaligned_cluster_rows(tmp_path):
    path = tmp_path / "clusters.csv.gz"
    pd.DataFrame({"row_index": [2, 1, 0], "leiden_r1.0_s0": [0, 0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="align"):
        _read_primary_labels(path, np.arange(3))


def test_parallel_partition_grid_matches_serial_labels_exactly():
    from so12_embedding_methods import leiden_grid
    rng = np.random.default_rng(22)
    z = rng.normal(size=(100, 5)).astype('float32')
    graph, _, _ = knn_graph(z, {'threads': 1, 'knn_k': 10})
    config = {'leiden_resolutions': [.5, 1., 1.5], 'leiden_seeds': [0, 1]}
    sequential, metrics = leiden_grid(graph, {**config, 'runtime': {'leiden_parallel_workers': 1}})
    concurrent, concurrent_metrics = leiden_grid(graph, {**config, 'runtime': {'leiden_parallel_workers': 4}})
    pd.testing.assert_frame_equal(sequential, concurrent)
    assert metrics == concurrent_metrics
