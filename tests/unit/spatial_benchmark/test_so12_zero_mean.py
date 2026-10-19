import numpy as np
import pytest
import torch

from scripts.analysis.compare_so12_zero_mean import fit_constants, constant_statistics
from spatial_benchmark.so12_relative_qkv_training import aggregate_statistics, sufficient_statistics


def test_means_use_training_only_and_distinguish_core_weights():
    targets = {"A": np.array([[0., 2.]], dtype=np.float32),
               "B": np.array([[4., 8.], [4., 8.], [4., 8.]], dtype=np.float32),
               "held_out": np.full((8, 2), 1e6, dtype=np.float32)}
    values = fit_constants(targets, ["A", "B"])
    np.testing.assert_array_equal(values["mean_equal_core"], [2., 5.])
    np.testing.assert_array_equal(values["mean_equal_cell"], [3., 6.5])
    np.testing.assert_array_equal(values["zero"], [0., 0.])


def test_zero_scores_have_hand_calculated_losses_and_undefined_correlation():
    y = np.array([[-2.], [0.], [3.]], dtype=np.float32)
    s = constant_statistics(y, np.ones_like(y, dtype=bool), np.zeros(1), chunk_cells=1)
    result = aggregate_statistics([{"alias": "A", "view": 0, "statistics": s}])
    assert result["equal_core_huber"] == pytest.approx(4 / 3)
    assert result["pooled_mae"] == pytest.approx(5 / 3)
    assert result["pooled_mse"] == pytest.approx(13 / 3)
    assert result["pooled_pearson"] is None


def test_chunked_statistics_match_trainer_on_selected_entries():
    rng = np.random.default_rng(3)
    y = rng.normal(size=(17, 5)).astype(np.float32)
    m = rng.random(y.shape) < .4
    p = rng.normal(size=5).astype(np.float32)
    expected = sufficient_statistics(torch.tensor(np.broadcast_to(p, y.shape).copy()),
                                     torch.tensor(y), torch.tensor(m))
    actual = constant_statistics(y, m, p, chunk_cells=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    altered = y.copy(); altered[~m] = 1e6
    np.testing.assert_array_equal(actual, constant_statistics(altered, m, p, chunk_cells=4))
