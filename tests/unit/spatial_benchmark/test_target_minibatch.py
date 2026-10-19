"""Coverage, mask and pooled-objective gates for target-cell minibatching."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import pytest
import torch

from spatial_benchmark.binary_expression import masked_binary_expression_bce_sums
from spatial_benchmark.target_cell_masking import (
    make_training_target_cell_masks,
    make_validation_target_cell_masks,
)
from spatial_benchmark.target_minibatch import (
    expected_updates_per_epoch,
    target_window_plan,
    validate_target_windows,
    window_plan_receipt,
)


def plan(n_cells: int, batch_size: int, **overrides):
    kwargs = dict(seed=0, core_alias="synthetic", global_epoch=1)
    kwargs.update(overrides)
    return target_window_plan(n_cells, batch_size, **kwargs)


@pytest.mark.parametrize("n_cells,batch_size", [
    (1, 1), (1, 1024), (12, 1), (12, 4), (13, 4), (1025, 1024),
    (38145, 512), (43462, 1024), (14856, 2048),
])
def test_plan_covers_every_target_once_with_balanced_nonempty_windows(n_cells, batch_size):
    windows = plan(n_cells, batch_size)
    validate_target_windows(windows, n_cells)
    sizes = [stop - start for start, stop in windows]
    assert len(windows) == (n_cells + batch_size - 1) // batch_size
    assert 1 <= min(sizes) <= max(sizes) <= batch_size
    assert max(sizes) - min(sizes) <= 1
    assert sorted(row for start, stop in windows for row in range(start, stop)) == list(range(n_cells))


def test_remainder_is_balanced_instead_of_a_tiny_last_update():
    assert sorted(plan(1025, 1024)) == [(0, 513), (513, 1025)]
    assert sorted(plan(13, 4)) == [(0, 4), (4, 7), (7, 10), (10, 13)]


@pytest.mark.parametrize("batch_size,expected", [(512, 413), (1024, 209), (2048, 108)])
def test_observed_so2_cohort_optimizer_update_counts(batch_size, expected):
    # Fixed structural counts for SO2-C15..C26, matching so2_pooled_full_core.
    counts = [38145, 16229, 9041, 11696, 19119, 12155,
              4897, 11443, 43462, 14506, 13147, 14856]
    assert sum(counts) == 208696
    assert expected_updates_per_epoch(counts, batch_size) == expected
    assert sum(len(plan(n, batch_size)) for n in counts) == expected


def test_order_is_reproducible_domain_specific_and_independent_of_global_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    original = plan(101, 7)
    assert original == plan(101, 7, core_alias=" SYNTHETIC ")
    assert original != plan(101, 7, seed=1)
    assert original != plan(101, 7, global_epoch=2)
    assert original != plan(101, 7, core_alias="other")
    assert original != tuple(sorted(original))
    assert random.getstate() == python_state
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]


@pytest.mark.parametrize("windows", [
    (), ((0, 0), (0, 5)), ((0, 3), (3, 3), (3, 5)),
    ((0, 2), (3, 5)), ((0, 3), (2, 5)), ((0, 3), (0, 3), (3, 5)),
    ((1, 5),), ((0, 4),), ((0, 6),), ((-1, 5),), ((3, 0), (0, 5)),
])
def test_invalid_coverage_is_rejected(windows):
    with pytest.raises(ValueError):
        validate_target_windows(windows, 5)


@pytest.mark.parametrize("windows", ["0,5", ((0.0, 5),), ((False, 5),), ((0, 3, 5),)])
def test_malformed_window_types_are_rejected(windows):
    with pytest.raises(TypeError):
        validate_target_windows(windows, 5)


@pytest.mark.parametrize("kwargs,error", [
    ({"n_cells": 0}, ValueError), ({"n_cells": -1}, ValueError),
    ({"batch_size": 0}, ValueError), ({"batch_size": 2.5}, TypeError),
    ({"n_cells": True}, TypeError), ({"seed": -1}, ValueError),
    ({"seed": 1.5}, TypeError), ({"global_epoch": -1}, ValueError),
    ({"global_epoch": False}, TypeError), ({"core_alias": " "}, ValueError),
    ({"core_alias": None}, TypeError),
])
def test_invalid_plan_inputs_fail_closed(kwargs, error):
    arguments = dict(n_cells=11, batch_size=4, seed=0, core_alias="synthetic", global_epoch=1)
    arguments.update(kwargs)
    with pytest.raises(error):
        target_window_plan(**arguments)


@pytest.mark.parametrize("counts,error", [([], ValueError), ([1, 0], ValueError), ([1, 2.5], TypeError), ("12", TypeError)])
def test_invalid_core_counts_are_rejected(counts, error):
    with pytest.raises(error):
        expected_updates_per_epoch(counts, 4)


def test_receipt_canonical_hash_binds_order_and_inputs():
    receipt = window_plan_receipt(101, 7, seed=2, core_alias=" test ", global_epoch=3)
    assert receipt == window_plan_receipt(101, 7, seed=2, core_alias="TEST", global_epoch=3)
    assert tuple(map(tuple, receipt["windows"])) == plan(101, 7, seed=2, core_alias="TEST", global_epoch=3)
    digest = receipt.pop("plan_sha256")
    def sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
    assert sha(receipt) == digest
    receipt["windows"].reverse()
    assert sha(receipt) != digest


@pytest.mark.parametrize("batch_size", [1, 7, 16, 100])
@pytest.mark.parametrize("role", ["training", "validation"])
def test_window_partition_and_execution_order_preserve_canonical_gene_masks(batch_size, role):
    common = dict(n_cells=31, n_genes=17, core_alias="synthetic", namespace="original_mask_namespace")
    if role == "training":
        make = make_training_target_cell_masks
        common.update(base_seed=2026092601, global_epoch=6)
    else:
        make = make_validation_target_cell_masks
    whole = make(**common)
    restored_mask = np.zeros_like(whole.mask)
    restored_counts = np.zeros_like(whole.masked_gene_counts)
    for start, stop in plan(31, batch_size, seed=789, global_epoch=6):
        part = make(**common, target_indices=np.arange(start, stop, dtype=np.int64))
        restored_mask[start:stop] = part.mask
        restored_counts[start:stop] = part.masked_gene_counts
    np.testing.assert_array_equal(restored_mask, whole.mask)
    np.testing.assert_array_equal(restored_counts, whole.masked_gene_counts)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_masked_entry_weighted_split_bce_and_gradient_equal_pooled_objective(batch_size):
    # Fixed weights and no optimizer/dropout: only split/aggregation is varied.
    # Unequal masks expose incorrect unweighted averages of per-window means.
    features = torch.linspace(-2, 3, 39).reshape(13, 3)
    raw_target = ((torch.arange(39).reshape(13, 3) % 7) == 0).float()
    mask = torch.from_numpy(make_training_target_cell_masks(
        13, 3, base_seed=42, namespace="synthetic", core_alias="test", global_epoch=1,
    ).mask.copy())
    reference_parameter = torch.tensor([.25, -.1, .4], requires_grad=True)
    reference = masked_binary_expression_bce_sums(features * reference_parameter, raw_target, mask)
    reference_loss = reference.detection_nll_sum / reference.masked_entries
    reference_gradient, = torch.autograd.grad(reference_loss, reference_parameter)

    split_parameter = reference_parameter.detach().clone().requires_grad_()
    sums = [masked_binary_expression_bce_sums(
        features[start:stop] * split_parameter, raw_target[start:stop], mask[start:stop],
    ) for start, stop in plan(13, batch_size)]
    split_loss = sum(item.detection_nll_sum for item in sums) / sum(item.masked_entries for item in sums)
    split_gradient, = torch.autograd.grad(split_loss, split_parameter)
    torch.testing.assert_close(split_loss, reference_loss)
    torch.testing.assert_close(split_gradient, reference_gradient)
    assert sum(item.positive_entries for item in sums) == reference.positive_entries
