"""CPU-only contracts for cumulative-improvement validation early stopping."""

from copy import deepcopy
import json

import pytest

from spatial_benchmark.local_gene_stopping import LocalGeneEarlyStopping


def test_tiny_strict_minima_accumulate_against_significant_anchor() -> None:
    policy = LocalGeneEarlyStopping(patience=3, min_epochs=1, max_epochs=20)
    first = policy.observe(epoch=1, count_nll=1.0)
    assert first.improved_best and first.improved_anchor
    second = policy.observe(epoch=2, count_nll=0.99996)
    assert second.improved_best and not second.improved_anchor
    assert second.best_epoch == 2 and second.anchor_epoch == 1
    assert second.epochs_without_improvement == 1
    third = policy.observe(epoch=3, count_nll=0.99992)
    assert third.improved_best and not third.improved_anchor
    assert third.epochs_without_improvement == 2
    fourth = policy.observe(epoch=4, count_nll=1.0 - 1e-4)
    assert fourth.improved_best and fourth.improved_anchor
    assert fourth.anchor_epoch == fourth.best_epoch == 4
    assert fourth.epochs_without_improvement == 0
    assert not fourth.should_stop


def test_tiny_final_improvement_is_saved_even_when_patience_stops() -> None:
    policy = LocalGeneEarlyStopping(patience=2, min_epochs=1, max_epochs=10)
    policy.observe(epoch=1, count_nll=0.4)
    policy.observe(epoch=2, count_nll=0.39999)
    stop = policy.observe(epoch=3, count_nll=0.39998)
    assert stop.improved_best and not stop.improved_anchor
    assert stop.best_epoch == 3 and stop.anchor_epoch == 1
    assert stop.should_stop and stop.stop_reason == "patience"
    assert not stop.max_epochs_reached


def test_default_minimum_gates_stop_but_not_patience_counting() -> None:
    policy = LocalGeneEarlyStopping()
    for epoch in range(1, 50):
        decision = policy.observe(epoch=epoch, count_nll=0.3)
        assert not decision.should_stop
        assert decision.epochs_without_improvement == epoch - 1
    stop = policy.observe(epoch=50, count_nll=0.3)
    assert stop.should_stop and stop.stop_reason == "patience"
    assert stop.epochs_without_improvement == 49
    assert stop.best_epoch == 1
    assert not stop.improved_best and not stop.improved_anchor


def test_improvement_on_minimum_epoch_requires_thirty_complete_later_epochs() -> None:
    policy = LocalGeneEarlyStopping()
    for epoch in range(1, 50):
        policy.observe(epoch=epoch, count_nll=0.4)
    reset = policy.observe(epoch=50, count_nll=0.39)
    assert reset.improved_anchor and not reset.should_stop
    for epoch in range(51, 80):
        assert not policy.observe(epoch=epoch, count_nll=0.395).should_stop
    stop = policy.observe(epoch=80, count_nll=0.395)
    assert stop.should_stop and stop.stop_reason == "patience"
    assert stop.epochs_without_improvement == 30
    assert stop.best_epoch == 50


def test_safety_cap_is_reported_separately_from_patience() -> None:
    policy = LocalGeneEarlyStopping()
    for epoch in range(1, 501):
        decision = policy.observe(epoch=epoch, count_nll=1.0 - 0.001 * epoch)
        assert decision.should_stop == (epoch == 500)
    assert decision.stop_reason == "maximum_epochs"
    assert decision.max_epochs_reached
    assert decision.improved_best and decision.improved_anchor
    assert decision.epochs_without_improvement == 0


def test_simultaneous_patience_and_cap_keep_both_facts_explicit() -> None:
    policy = LocalGeneEarlyStopping(patience=2, min_epochs=3, max_epochs=3)
    policy.observe(epoch=1, count_nll=0.3)
    policy.observe(epoch=2, count_nll=0.3)
    stop = policy.observe(epoch=3, count_nll=0.3)
    assert stop.stop_reason == "patience" and stop.max_epochs_reached
    with pytest.raises(RuntimeError, match="after stopping"):
        policy.observe(epoch=4, count_nll=0.2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.1, True, "0.3"])
def test_invalid_validation_never_mutates_state(value) -> None:
    policy = LocalGeneEarlyStopping()
    policy.observe(epoch=1, count_nll=0.3)
    before = policy.state_dict()
    with pytest.raises(ValueError):
        policy.observe(epoch=2, count_nll=value)
    assert policy.state_dict() == before


@pytest.mark.parametrize("epoch", [0, 2, True, 1.0])
def test_epochs_are_integer_contiguous_complete_validations(epoch) -> None:
    policy = LocalGeneEarlyStopping()
    with pytest.raises(ValueError):
        policy.observe(epoch=epoch, count_nll=0.3)
    assert policy.completed_epoch == 0


def test_state_roundtrip_matches_uninterrupted_decisions_through_stop() -> None:
    uninterrupted = LocalGeneEarlyStopping(patience=4, min_epochs=5, max_epochs=30)
    restarted = LocalGeneEarlyStopping.from_state_dict(
        json.loads(json.dumps(uninterrupted.state_dict(), allow_nan=False))
    )
    losses = [0.4, 0.39996, 0.39992, 0.3998, 0.39981, 0.39977, 0.39975, 0.39974]
    for epoch, loss in enumerate(losses, 1):
        expected = uninterrupted.observe(epoch=epoch, count_nll=loss)
        actual = restarted.observe(epoch=epoch, count_nll=loss)
        assert actual == expected
        encoded = json.dumps(restarted.state_dict(), allow_nan=False)
        restarted = LocalGeneEarlyStopping.from_state_dict(json.loads(encoded))
        assert restarted.completed_epoch == epoch
        assert restarted.should_stop == expected.should_stop
        assert restarted.stop_reason == expected.stop_reason
    assert expected.stop_reason == "patience"
    assert expected.best_epoch == 8 and expected.anchor_epoch == 4
    with pytest.raises(RuntimeError, match="after stopping"):
        restarted.observe(epoch=9, count_nll=0.3)


@pytest.mark.parametrize("field,value", [
    ("count_nll", float("nan")), ("best_count_nll", float("inf")),
    ("epochs_without_improvement", 7), ("best_epoch", 8),
    ("anchor_count_nll", 0.2), ("should_stop", True),
    ("stop_reason", "maximum_epochs"), ("improved_best", 1),
    ("max_epochs_reached", True),
])
def test_reload_rejects_corrupt_or_nonfinite_policy_state(field, value) -> None:
    policy = LocalGeneEarlyStopping()
    policy.observe(epoch=1, count_nll=0.4)
    policy.observe(epoch=2, count_nll=0.39999)
    payload = deepcopy(policy.state_dict())
    payload["state"][field] = value
    with pytest.raises(ValueError):
        LocalGeneEarlyStopping.from_state_dict(payload)


@pytest.mark.parametrize("kwargs", [
    {"patience": 0}, {"patience": True}, {"min_delta": 0},
    {"min_delta": float("nan")}, {"min_epochs": 0},
    {"min_epochs": 51, "max_epochs": 50}, {"max_epochs": 500.0},
])
def test_invalid_policy_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        LocalGeneEarlyStopping(**kwargs)


def test_state_dict_is_detached_and_large_finite_values_do_not_reset_on_ties() -> None:
    policy = LocalGeneEarlyStopping()
    policy.observe(epoch=1, count_nll=1e20)
    state = policy.state_dict()
    state["state"]["count_nll"] = 0.1
    state["config"]["patience"] = 99
    restored = LocalGeneEarlyStopping.from_state_dict(policy.state_dict())
    decision = restored.observe(epoch=2, count_nll=1e20)
    assert not decision.improved_best and not decision.improved_anchor
    assert restored.config.patience == 30

