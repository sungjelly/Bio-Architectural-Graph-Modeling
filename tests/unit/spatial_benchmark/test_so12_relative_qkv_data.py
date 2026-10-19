"""Scientific leakage and train-only preprocessing checks for joint-slide data."""

import numpy as np
import pytest
import torch

from spatial_benchmark.so12_relative_qkv_data import (
    SO12DataContractError,
    fit_training_metadata,
    validate_donor_partition,
    visible_count_pearson_input,
)


def _parameters():
    return dict(gene_p=np.array([.2, .3, .5], dtype=np.float32),
                residual_mean=np.array([.1, -.2, .3], dtype=np.float32),
                residual_sd=np.array([.8, 1.2, .5], dtype=np.float32))


def test_hidden_count_perturbations_cannot_change_any_input():
    counts = torch.tensor([[1., 7., 4.], [5., 2., 8.], [9., 2., 3.]])
    hidden = torch.tensor([[False, True, False], [True, False, False], [True, True, True]])
    expected = visible_count_pearson_input(counts, hidden, **_parameters())
    altered = counts.clone()
    altered[hidden] = torch.tensor([100000., 0., 998., 1234., 55.])
    actual = visible_count_pearson_input(altered, hidden, **_parameters())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual[hidden], torch.zeros_like(actual[hidden]))
    assert torch.isfinite(actual).all()


def test_visible_only_formula_matches_independent_numpy_calculation():
    counts = np.array([[1, 99, 4], [3, 2, 1], [0, 7, 0]], dtype=np.float32)
    hidden = np.array([[False, True, False], [False, False, False], [False, True, False]])
    p = _parameters()
    expected = np.zeros_like(counts)
    for row in range(len(counts)):
        visible = ~hidden[row]
        exposure = max(float(counts[row, visible].sum() / p['gene_p'][visible].sum()), 1.)
        mu = exposure * p['gene_p'][visible]
        residual = (counts[row, visible] - mu) / np.sqrt(1.01 * mu)
        expected[row, visible] = np.minimum(
            (residual - p['residual_mean'][visible]) / p['residual_sd'][visible], 10.)
    actual = visible_count_pearson_input(torch.from_numpy(counts), torch.from_numpy(hidden), **p)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-6)


def test_fully_masked_rows_are_finite_zero_and_hidden_gradient_is_zero():
    counts = torch.tensor([[1., 100., 4.], [7., 5., 2.]], requires_grad=True)
    hidden = torch.tensor([[False, True, False], [True, True, True]])
    value = visible_count_pearson_input(counts, hidden, **_parameters())
    value.sum().backward()
    assert torch.isfinite(value).all()
    assert torch.equal(value[1], torch.zeros(3))
    assert torch.equal(counts.grad[hidden], torch.zeros_like(counts.grad[hidden]))


def test_normalization_stays_float32_under_autocast():
    counts = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    hidden = torch.tensor([[False, True, False]])
    plain = visible_count_pearson_input(counts, hidden, **_parameters())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = visible_count_pearson_input(counts, hidden, **_parameters())
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, plain, rtol=0, atol=0)


@pytest.mark.parametrize('parameter,value', [('gene_p', [0., .5, .5]),
                                           ('gene_p', [.2, .3, .4]),
                                           ('residual_sd', [1., 0., 1.])])
def test_invalid_frozen_parameters_are_rejected(parameter, value):
    parameters = _parameters()
    parameters[parameter] = np.asarray(value, dtype=np.float32)
    with pytest.raises(SO12DataContractError):
        visible_count_pearson_input(torch.ones((1, 3)), torch.zeros((1, 3), dtype=torch.bool),
                                   **parameters)


def test_hidden_sentinel_is_not_consumed_but_negative_visible_count_is_rejected():
    counts = torch.tensor([[1., float('nan'), 2.]])
    hidden = torch.tensor([[False, True, False]])
    assert torch.isfinite(visible_count_pearson_input(counts, hidden, **_parameters())).all()
    counts[0, 0] = -1
    with pytest.raises(SO12DataContractError, match='Visible counts'):
        visible_count_pearson_input(counts, hidden, **_parameters())


def test_morphology_fitting_ignores_holdout_values():
    raw = np.arange(6 * 22, dtype=float).reshape(6, 22) + 1
    train = np.array([True, True, True, True, False, False])
    transformed, stats = fit_training_metadata(raw, train)
    changed = raw.copy()
    changed[~train] = 1e7
    altered, altered_stats = fit_training_metadata(changed, train)
    assert stats == altered_stats
    np.testing.assert_array_equal(transformed[train], altered[train])
    np.testing.assert_allclose(transformed[train].mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(transformed[train].std(axis=0), 1, atol=1e-7)
    assert stats['fit_scope'] == 'pooled_training_cells_only'


def test_new_missing_training_morphology_cannot_silently_expand_model():
    raw = np.ones((3, 22))
    raw[0, 0] = np.nan
    with pytest.raises(SO12DataContractError, match='22-feature'):
        fit_training_metadata(raw, np.array([True, True, False]))


def test_donor_partition_rejects_shared_group():
    validate_donor_partition(np.array([1, 1, 2, 2]), np.array([True, True, False, False]))
    with pytest.raises(SO12DataContractError, match='crosses'):
        validate_donor_partition(np.array([1, 1, 2, 2]), np.array([True, False, False, False]))
    with pytest.raises(SO12DataContractError, match='anonymous integer'):
        validate_donor_partition(np.array(['a', 'b']), np.array([True, False]))
