"""Leakage, gene alignment, and count-distribution contracts for the new MLP."""

from dataclasses import fields

import pytest
import torch

from spatial_benchmark.hurdle_negative_binomial import (
    masked_hurdle_negative_binomial_nll,
)
from spatial_benchmark.local_gene_residual import LocalGeneResidualCountModel


def _model(*, use_local: bool = True) -> LocalGeneResidualCountModel:
    torch.manual_seed(41)
    return LocalGeneResidualCountModel(
        5, covariate_dim=2, hidden_dim=16, dropout=0.0, use_local=use_local,
        initial_nonzero_probability=torch.tensor([0.02, 0.1, 0.3, 0.6, 0.9]),
        initial_positive_conditional_mean=torch.tensor([1.1, 1.5, 2.0, 4.0, 8.0]),
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5], [0.6, 0.7, -0.8, 0.9, 0.1]]),
        torch.tensor([[True, False, True, False, True], [False, True, False, True, True]]),
        torch.tensor([[0.2, -0.1], [0.5, 0.6]]),
        torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.4, 0.8, 0.2, 0.5, 0.3]]),
        torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7, 0.9]]),
    )


def _activate_paths(model: LocalGeneResidualCountModel) -> None:
    # Exercise trained input paths, rather than trivially testing zero heads.
    with torch.no_grad():
        model.detection_head.weight.normal_(std=0.1)
        model.count_head.weight.normal_(std=0.1)
        model.local_detection_weights.fill_(0.2)
        model.local_count_weights.fill_(0.3)


def test_initial_distribution_matches_training_gene_priors() -> None:
    model = _model()
    output = model(*_inputs())
    expected_probability = model.initial_nonzero_probability.expand(2, -1)
    expected_positive_mean = model.initial_positive_conditional_mean.expand(2, -1)
    torch.testing.assert_close(output.nonzero_probability, expected_probability)
    torch.testing.assert_close(output.theta, torch.ones(5))
    torch.testing.assert_close(output.positive_conditional_mean, expected_positive_mean)
    torch.testing.assert_close(output.marginal_mean, expected_probability * expected_positive_mean)
    assert all(getattr(output, field.name).dtype == torch.float32 for field in fields(output))


@pytest.mark.parametrize("replacement", [float("nan"), float("inf"), -float("inf"), 1e20])
def test_hidden_values_never_change_prediction_or_gradient(replacement: float) -> None:
    model = _model().eval()
    _activate_paths(model)
    expression, mask, covariates, log_mean, fraction = _inputs()
    baseline = model(expression, mask, covariates, log_mean, fraction)
    corrupted = expression.masked_fill(mask, replacement).requires_grad_()
    actual = model(corrupted, mask, covariates, log_mean, fraction)
    for field in fields(actual):
        torch.testing.assert_close(getattr(actual, field.name), getattr(baseline, field.name), rtol=0, atol=0)
    actual.marginal_mean.sum().backward()
    assert corrupted.grad is not None
    assert torch.isfinite(corrupted.grad).all()
    assert torch.count_nonzero(corrupted.grad[mask]) == 0
    assert torch.count_nonzero(corrupted.grad[~mask]) > 0
    changed = expression.clone()
    changed[~mask] += 1.0
    visible_change = model(changed, mask, covariates, log_mean, fraction)
    assert not torch.equal(visible_change.marginal_mean, actual.marginal_mean)


def test_neighborhood_changes_affect_only_the_corresponding_gene() -> None:
    model = _model().eval()
    _activate_paths(model)
    expression, mask, covariates, log_mean, fraction = _inputs()
    before = model(expression, mask, covariates, log_mean, fraction)
    log_mean[:, 3] += 0.7
    fraction[:, 3] += 0.1
    after = model(expression, mask, covariates, log_mean, fraction)
    other_genes = [0, 1, 2, 4]
    for name in ("nonzero_logits", "nb_mean", "positive_conditional_mean", "marginal_mean"):
        torch.testing.assert_close(getattr(before, name)[:, other_genes], getattr(after, name)[:, other_genes], rtol=0, atol=0)
        assert not torch.equal(getattr(before, name)[:, 3], getattr(after, name)[:, 3])
    torch.testing.assert_close(after.nonzero_logits[:, 3] - before.nonzero_logits[:, 3], torch.full((2,), 0.16))
    torch.testing.assert_close(after.theta, before.theta, rtol=0, atol=0)


def test_no_neighbor_and_self_only_are_identical_with_shared_target_weights() -> None:
    local = _model().eval()
    _activate_paths(local)
    self_only = _model(use_local=False).eval()
    self_only.load_state_dict(local.state_dict())
    expression, mask, covariates, log_mean, fraction = _inputs()
    expected = local(expression, mask, covariates, torch.zeros_like(log_mean), torch.zeros_like(fraction))
    actual = self_only(expression, mask, covariates)
    ignored = self_only(expression, mask, covariates, torch.full_like(log_mean, float("nan")), fraction)
    for field in fields(actual):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(actual, field.name), getattr(ignored, field.name), rtol=0, atol=0)
    assert not self_only.local_detection_weights.requires_grad
    assert not self_only.local_count_weights.requires_grad


@pytest.mark.parametrize("all_zero", [False, True])
def test_likelihood_and_all_parameter_gradients_are_finite(all_zero: bool) -> None:
    model = _model()
    inputs = _inputs()
    target = torch.tensor([[0, 1, 2, 3, 4], [2, 3, 0, 5, 1]])
    if all_zero:
        target.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # A second step exercises trunk gradients after the zero-initialized heads move.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(*inputs)
        loss = masked_hurdle_negative_binomial_nll(
            output.nonzero_logits, output.nb_mean, output.theta, target, inputs[1]
        )
        assert loss.isfinite()
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
    assert torch.count_nonzero(model.trunk[0].weight.grad) > 0


def test_fully_hidden_target_is_finite_at_prior_boundary_values() -> None:
    model = LocalGeneResidualCountModel(
        3, covariate_dim=0, hidden_dim=8, use_local=False,
        initial_nonzero_probability=torch.tensor([0.0, 0.1, 1.0]),
        initial_positive_conditional_mean=torch.tensor([1.0, 2.0, 10000.0]),
    )
    output = model(torch.full((2, 3), float("nan")), torch.ones(2, 3, dtype=torch.bool))
    assert torch.isfinite(output.marginal_mean).all()
    assert torch.all(output.positive_conditional_mean >= 1.0)
    torch.testing.assert_close(output.positive_conditional_mean[0, 0], torch.tensor(1.0002), rtol=1e-5, atol=1e-6)


def test_default_model_remains_below_three_million_parameters() -> None:
    model = LocalGeneResidualCountModel(1000)
    assert sum(parameter.numel() for parameter in model.parameters()) < 3_000_000


def test_rejects_visible_nonfinite_or_misaligned_local_inputs() -> None:
    model = _model()
    expression, mask, covariates, log_mean, fraction = _inputs()
    expression[0, 1] = float("nan")
    with pytest.raises(ValueError, match="visible input_expression must be finite"):
        model(expression, mask, covariates, log_mean, fraction)
    expression[0, 1] = 0.0
    with pytest.raises(ValueError, match="local_log_mean must have shape"):
        model(expression, mask, covariates, log_mean[:, :-1], fraction)
    with pytest.raises(ValueError, match="local_positive_fraction must be in"):
        model(expression, mask, covariates, log_mean, fraction + 1.0)

