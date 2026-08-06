from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.train.run_self_hurdle_capacity import (
    SelfHurdleRunnerError,
    _validate_config,
)


def _config() -> dict:
    return {
        "campaign": {
            "campaign_id": "cmp_20260729_self_hurdle_full_core_capacity",
            "frozen_contract_sha256": (
                "26cf4f094d843c4fa9020c52e1d45988"
                "f8e4a0c43f20beb186e4742b2dacba00"
            ),
        },
        "experiment": {
            "biological_unit_alias": "ANC-03",
            "resource_pilot": True,
            "arm": "self-hurdle",
            "graph_arms_authorized": False,
        },
        "model": {
            "name": "self-hurdle-count",
            "family": "self_only_hurdle_count",
            "uses_graph_inputs": False,
            "uses_edge_inputs": False,
            "expected_trainable_parameter_count": 16917200,
        },
        "features": {"use_edge_features": False},
        "graph": {
            "kind": "disabled_self_only_schema_placeholder",
            "enabled": False,
            "construction_performed": False,
            "expected_directed_edges": 0,
            "expected_graph_sha256": None,
            "edge_dropout": 0.0,
        },
        "trainer": {
            "max_epochs": 2,
            "restore_best": False,
            "checkpoint_policy": "last_only",
            "primary_checkpoint_role": "last",
            "graph_execution": "none_graph_inputs_prohibited",
            "target_node_batch_size": 16384,
        },
        "evaluation": {
            "protocol": "held_in_full_core_fixed_budget",
            "task_family": "masked_expression_hurdle_count",
            "primary_metric": "fit/whole_node/hurdle_loss",
            "splits": ["fit"],
            "mask_replicates_per_mode": 1,
        },
        "metadata": {
            "execution_role": "resource_pilot",
            "no_graph_construction_or_input": True,
        },
        "dataset": {
            "biological_unit_alias": "ANC-03",
            "validation_or_test_partition_present": False,
            "task": "masked_expression_hurdle_count",
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }


def test_resource_contract_is_graphless() -> None:
    contract = _validate_config(_config())
    assert contract.alias == "ANC-03"
    assert contract.resource_pilot is True


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("graph", "enabled", True),
        ("graph", "construction_performed", True),
        ("model", "uses_graph_inputs", True),
        ("features", "use_edge_features", True),
    ],
)
def test_contract_rejects_any_graph_enablement(
    section: str, field: str, value: object
) -> None:
    config = deepcopy(_config())
    config[section][field] = value
    with pytest.raises(SelfHurdleRunnerError, match="graphless campaign"):
        _validate_config(config)

