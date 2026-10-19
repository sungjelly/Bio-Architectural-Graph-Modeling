"""Execute pilot epoch controls through their terminal transition.

The original pilot failed only after validation because minimum_epochs=2 and
maximum_epochs=2 violated the shared stopping helper. These tests exercise the
actual helper using builder-produced settings, rather than merely asserting
that the configuration contains two epochs.
"""

from __future__ import annotations

import pytest

from scripts.train import run_so2_target_isolated_nb as base
from spatial_benchmark.so2_binary_target_minibatch import build_configuration
from spatial_benchmark.so2_nb_training import EarlyStoppingState, update_early_stopping


@pytest.mark.parametrize("batch_size", [512, 1024, 2048])
@pytest.mark.parametrize("second_value", [0.2, 0.3, 0.4, 0.29995])
def test_pilot_executes_two_validations_then_stops_at_maximum(batch_size, second_value):
    source = {
        "campaign": {"campaign_id": "synthetic_source"},
        "metadata": {"gradient_diagnostics": {}},
        "evaluation": {},
        "trainer": {"early_stopping_patience": 25, "early_stopping_min_delta": 1e-4},
    }
    config = build_configuration(source, {}, gpu="0", profile="pilot",
                                 batch_size=batch_size, task_contract_sha256="a" * 64)
    trainer = config["trainer"]
    control = EarlyStoppingState()
    selection = EarlyStoppingState()
    observed_epochs = []
    for epoch, value in enumerate((0.3, second_value), start=1):
        assert not control.should_stop, "Pilot stopped before completing both epochs"
        control = update_early_stopping(
            control, value, completed_epoch=epoch,
            minimum_epochs=trainer["minimum_global_epochs"],
            maximum_epochs=trainer["max_epochs"],
            patience=trainer["early_stopping_patience"],
            min_delta=trainer["early_stopping_min_delta"],
        )
        selection = base._update_exact_checkpoint_selection(selection, value, completed_epoch=epoch)
        observed_epochs.append(control.completed_epoch)
        assert not selection.should_stop
        assert selection.bad_validations == 0
        if epoch == 1:
            assert not control.should_stop
            assert control.stop_reason is None

    assert observed_epochs == [1, 2]
    assert control.should_stop and control.stop_reason == "maximum_epochs"
    assert selection.best_value == min(0.3, second_value)
    assert selection.best_epoch == (2 if second_value < 0.3 else 1)
    if second_value == 0.29995:
        # Exact-best selection accepts the small improvement while patience
        # keeps its independent meaningful-improvement reference.
        assert control.best_epoch == 1
        assert selection.best_epoch == 2

