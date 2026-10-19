"""Focused objective contracts for Bernoulli-only expression detection."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from spatial_benchmark.binary_expression import (
    masked_binary_expression_bce,
    masked_binary_expression_bce_sums,
)


def test_masked_bce_matches_independent_reference_and_ignores_count_magnitude() -> None:
    logits = torch.tensor(
        [[-1.2, 0.7, 1.1], [0.3, -0.9, 2.0]],
        dtype=torch.float32,
    )
    target = torch.tensor([[0, 2, 7], [1, 0, 12]], dtype=torch.int64)
    mask = torch.tensor(
        [[True, True, False], [True, True, True]],
        dtype=torch.bool,
    )

    sums = masked_binary_expression_bce_sums(logits, target, mask)
    selected_target = target[mask]
    expected = F.binary_cross_entropy_with_logits(
        logits[mask].double(),
        (selected_target > 0).double(),
        reduction="sum",
    )

    assert sums.masked_entries == 5
    assert sums.positive_entries == 3
    torch.testing.assert_close(
        sums.detection_nll_sum.double(), expected, rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        masked_binary_expression_bce(logits, target, mask),
        sums.detection_nll_sum / 5.0,
        rtol=0,
        atol=0,
    )

    changed_positive_counts = target.clone()
    changed_positive_counts[changed_positive_counts > 0] *= 100
    unchanged = masked_binary_expression_bce(
        logits,
        changed_positive_counts,
        mask,
    )
    torch.testing.assert_close(
        unchanged,
        sums.detection_nll_sum / 5.0,
        rtol=0,
        atol=0,
    )


def test_bce_gradient_is_sigmoid_minus_binary_target_on_masked_entries() -> None:
    logits = torch.tensor(
        [[-0.4, 0.8], [0.2, -1.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[0, 1], [7, 0]], dtype=torch.int64)
    mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)

    sums = masked_binary_expression_bce_sums(logits, target, mask)
    sums.detection_nll_sum.backward()

    expected = torch.zeros_like(logits)
    expected[mask] = torch.sigmoid(logits.detach()[mask]) - (target[mask] > 0).float()
    torch.testing.assert_close(logits.grad, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "target",
    [
        torch.tensor([[0.0, -1.0]]),
        torch.tensor([[0.0, 1.5]]),
        torch.tensor([[0.0, float("nan")]]),
    ],
)
def test_bce_rejects_invalid_masked_raw_counts(target: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        masked_binary_expression_bce(
            torch.zeros_like(target),
            target,
            torch.ones_like(target, dtype=torch.bool),
        )

