"""Focused probability and gradient contracts for hurdle-NB2."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from spatial_benchmark.hurdle_negative_binomial import (
    log1mexp,
    masked_hurdle_negative_binomial_nll,
    masked_hurdle_negative_binomial_nll_sums,
    zero_truncated_nb2_conditional_mean,
    zero_truncated_negative_binomial_nll_elementwise,
)


def _reference_zero_truncated_nll(
    mu: torch.Tensor,
    theta: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Independent FP64 reference through Torch's NB distribution."""

    work_mu = mu.detach().double()
    work_theta = theta.detach().double()
    work_target = target.detach().double()
    distribution = torch.distributions.NegativeBinomial(
        total_count=work_theta,
        logits=torch.log(work_mu) - torch.log(work_theta),
    )
    log_probability_zero = distribution.log_prob(torch.zeros_like(work_target))
    return -distribution.log_prob(work_target) + torch.log(
        -torch.expm1(log_probability_zero)
    )


def test_zero_truncated_nll_matches_independent_full_constant_reference() -> None:
    mu = torch.tensor([0.1, 1.3, 7.0, 3.2, 0.5, 19.0])
    theta = torch.tensor([0.7, 1.0, 4.0, 0.7, 1.0, 4.0])
    target = torch.tensor([1, 2, 729, 2, 1, 17], dtype=torch.int64)

    actual = zero_truncated_negative_binomial_nll_elementwise(
        mu,
        theta,
        target,
    )
    expected = _reference_zero_truncated_nll(mu, theta, target)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(
        actual.double(),
        expected,
        rtol=2e-6,
        atol=3e-5,
    )


def test_log1mexp_is_stable_near_zero_and_in_the_far_tail() -> None:
    log_probability = torch.tensor(
        [-1e-8, -1e-4, -math.log(2.0), -10.0, -100.0],
        dtype=torch.float32,
        requires_grad=True,
    )

    actual = log1mexp(log_probability)
    expected = torch.log(-torch.expm1(log_probability.detach().double()))

    assert actual.dtype == torch.float32
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-6)

    actual.sum().backward()
    assert log_probability.grad is not None
    assert bool(torch.isfinite(log_probability.grad).all())
    assert bool((log_probability.grad <= 0.0).all())
    assert float(log_probability.grad[0]) < -1e7


def test_zero_truncated_nll_remains_accurate_in_large_theta_poisson_limit() -> None:
    """Large inverse dispersion must not lose the log-gamma ratio."""

    mu = torch.tensor([1e-4, 1e-3, 0.1, 1.0], dtype=torch.float32)
    theta = torch.full_like(mu, 1e6)
    target = torch.tensor([1, 2, 10, 729], dtype=torch.int64)

    actual = zero_truncated_negative_binomial_nll_elementwise(
        mu,
        theta,
        target,
    )
    expected = _reference_zero_truncated_nll(mu, theta, target)

    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(
        actual.double(),
        expected,
        rtol=2e-5,
        atol=1e-3,
    )


def test_conditional_mean_matches_normalized_nb_mean_at_numerical_edges() -> None:
    mu = torch.tensor(
        [[1e-4, 0.1, 2.0], [1e-3, 7.0, 100.0]],
        dtype=torch.float32,
    )
    theta = torch.tensor([100.0, 1.0, 0.1], dtype=torch.float32)

    actual = zero_truncated_nb2_conditional_mean(mu, theta)
    work_mu = mu.double()
    work_theta = theta.double().view(1, -1)
    log_probability_zero = -work_theta * F.softplus(
        torch.log(work_mu) - torch.log(work_theta)
    )
    expected = work_mu / (-torch.expm1(log_probability_zero))

    assert actual.dtype == torch.float32
    assert bool(torch.isfinite(actual).all())
    assert bool((actual >= 1.0 - 1e-5).all())
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-5)


def test_masked_joint_likelihood_matches_detection_plus_positive_reference() -> None:
    nonzero_logits = torch.tensor(
        [[-1.2, 0.7, 1.1], [0.3, -0.9, 2.0]],
        dtype=torch.float32,
    )
    mu = torch.tensor(
        [[0.2, 1.5, 4.0], [2.3, 0.4, 8.0]],
        dtype=torch.float32,
    )
    theta = torch.tensor([0.7, 1.0, 3.0], dtype=torch.float32)
    raw_target = torch.tensor([[0, 2, 7], [1, 0, 12]], dtype=torch.int64)
    mask = torch.tensor(
        [[True, True, False], [True, True, True]],
        dtype=torch.bool,
    )

    sums = masked_hurdle_negative_binomial_nll_sums(
        nonzero_logits,
        mu,
        theta,
        raw_target,
        mask,
    )
    selected_logits = nonzero_logits[mask]
    selected_target = raw_target[mask]
    positive = selected_target > 0
    expected_detection = F.binary_cross_entropy_with_logits(
        selected_logits.double(),
        positive.double(),
        reduction="sum",
    )
    expanded_theta = theta.view(1, -1).expand_as(mu)
    expected_count = _reference_zero_truncated_nll(
        mu[mask][positive],
        expanded_theta[mask][positive],
        selected_target[positive],
    ).sum()

    assert sums.masked_entries == int(mask.sum())
    assert sums.positive_entries == int(positive.sum())
    torch.testing.assert_close(
        sums.detection_nll_sum.double(), expected_detection, rtol=2e-6, atol=2e-6
    )
    torch.testing.assert_close(
        sums.positive_count_nll_sum.double(),
        expected_count,
        rtol=2e-6,
        atol=2e-6,
    )
    torch.testing.assert_close(
        sums.joint_nll_sum,
        sums.detection_nll_sum + sums.positive_count_nll_sum,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        masked_hurdle_negative_binomial_nll(
            nonzero_logits,
            mu,
            theta,
            raw_target,
            mask,
        ),
        sums.joint_nll_sum / float(sums.masked_entries),
        rtol=0,
        atol=0,
    )


def test_zero_and_positive_observations_route_gradients_to_correct_components() -> None:
    nonzero_logits = torch.tensor(
        [[-0.4, 0.8], [0.2, -1.1]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mu = torch.tensor(
        [[0.25, 0.7], [3.5, 2.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    theta = torch.tensor([0.8, 2.0], dtype=torch.float32, requires_grad=True)
    raw_target = torch.tensor([[0, 1], [7, 0]], dtype=torch.int64)
    mask = torch.ones_like(raw_target, dtype=torch.bool)

    sums = masked_hurdle_negative_binomial_nll_sums(
        nonzero_logits,
        mu,
        theta,
        raw_target,
        mask,
    )
    sums.joint_nll_sum.backward()

    expected_logit_gradient = torch.sigmoid(nonzero_logits.detach()) - (
        raw_target > 0
    ).float()
    torch.testing.assert_close(
        nonzero_logits.grad,
        expected_logit_gradient,
        rtol=2e-6,
        atol=2e-6,
    )
    assert mu.grad is not None and theta.grad is not None
    assert bool(torch.isfinite(mu.grad).all())
    assert bool(torch.isfinite(theta.grad).all())
    assert torch.equal(mu.grad[raw_target == 0], torch.zeros(2))
    assert bool((mu.grad[raw_target > 0] != 0).all())
    assert bool((theta.grad != 0).all())


def test_all_zero_shard_materializes_exact_zero_count_gradients() -> None:
    nonzero_logits = torch.tensor(
        [[0.2, -0.3], [0.5, -0.7]],
        dtype=torch.float32,
        requires_grad=True,
    )
    mu = torch.tensor(
        [[0.2, 1.0], [2.0, 3.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    theta = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
    raw_target = torch.zeros((2, 2), dtype=torch.int64)
    mask = torch.ones_like(raw_target, dtype=torch.bool)

    sums = masked_hurdle_negative_binomial_nll_sums(
        nonzero_logits,
        mu,
        theta,
        raw_target,
        mask,
    )
    sums.joint_nll_sum.backward()

    assert sums.masked_entries == 4
    assert sums.positive_entries == 0
    assert float(sums.positive_count_nll_sum.detach()) == 0.0
    assert nonzero_logits.grad is not None
    assert bool((nonzero_logits.grad != 0).all())
    assert mu.grad is not None
    assert theta.grad is not None
    assert torch.equal(mu.grad, torch.zeros_like(mu))
    assert torch.equal(theta.grad, torch.zeros_like(theta))
