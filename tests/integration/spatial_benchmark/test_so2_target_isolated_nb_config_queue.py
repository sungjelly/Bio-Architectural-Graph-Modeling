from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from typing import Any

import pytest

from spatial_benchmark.configuration import (
    ConfigurationError,
    compose_config,
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import command_for_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"
CONFIG_PATH = CONFIG_ROOT / (
    "experiment/"
    "so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0.yaml"
)
CAMPAIGN_ID = (
    "cmp_20260926_so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0"
)
PROTOCOL = "donor_grouped_so2_target_isolated_nb_once_per_cell_earlystop_v1"


def _config() -> dict[str, Any]:
    return compose_config(CONFIG_PATH, config_root=CONFIG_ROOT)


def _set_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = mapping
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


def test_target_isolated_nb_protocol_is_exactly_locked() -> None:
    config = _config()
    validate_experiment_config(config)

    assert config["campaign"]["campaign_id"] == CAMPAIGN_ID
    assert config["evaluation"]["protocol"] == PROTOCOL
    assert config["evaluation"]["splits"] == ["validation"]
    assert config["evaluation"]["test_split_present"] is False
    assert config["evaluation"]["primary_metric"] == (
        "val/target_only_masked_negative_binomial_nll"
    )
    assert config["dataset"]["training_core_aliases"] == [
        f"SO2-C{core}" for core in range(15, 27)
    ]
    assert config["dataset"]["validation_core_aliases"] == [
        "SO2-C27",
        "SO2-C28",
    ]
    assert config["dataset"]["test_core_aliases"] == []
    assert config["masking"]["count_min"] == 1
    assert config["masking"]["count_max"] == 1000
    assert config["masking"]["target_cell_only"] is True
    assert config["masking"]["neighbor_cells_artificially_masked"] is False
    assert config["trainer"]["target_shards_per_core"] == 4
    assert config["trainer"]["training_cells_per_global_epoch"] == 208_696
    assert config["trainer"]["optimizer_updates_per_global_epoch"] == 6
    assert config["trainer"]["stage_complete_core_graph_on_device"] is False
    assert config["trainer"]["staged_relative_geometry_dtype"] == "float32"
    assert config["model"]["expected_trainable_parameter_count"] == 5_135_088
    contract = PROJECT_ROOT / (
        "experiments/campaigns/"
        f"{CAMPAIGN_ID}/frozen_task_contract.yaml"
    )
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == config["metadata"][
        "frozen_task_contract_sha256"
    ]


def test_target_isolated_nb_queue_routes_to_dedicated_four_rank_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        _config(),
        paths=ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)}),
    )
    assert command == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=4",
        "--max-restarts=0",
        str(tmp_path / "scripts/train/run_so2_target_isolated_nb.py"),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_target_isolated_primary_metric_has_pooled_likelihood_semantics() -> None:
    registry = load_yaml_mapping(CONFIG_ROOT / "schema/metrics_v1.yaml")
    family = registry["task_families"][
        "target_isolated_masked_expression_negative_binomial"
    ]
    assert family["primary_metric"] == (
        "val/target_only_masked_negative_binomial_nll"
    )
    assert family["primary_direction"] == "minimize"
    assert family["likelihood_contract"]["aggregation"] == (
        "total_masked_entry_nll_divided_by_total_masked_entries"
    )
    assert family["likelihood_contract"]["test_partition_present"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dataset", "training_cells"), 208_695, "12-core"),
        (("evaluation", "splits"), ["validation", "test"], "no test"),
        (("masking", "count_min"), 0, "inclusive support"),
        (("masking", "target_cell_only"), False, "target-isolated"),
        (("trainer", "target_shards_per_core"), 2, "target_shards_per_core"),
        (("trainer", "optimizer_updates_per_global_epoch"), 12, "optimizer_updates"),
        (("model", "graph_layers"), 8, "graph_layers"),
        (
            ("model", "neighbor_key_value_state_evolves_across_blocks"),
            True,
            "neighbor_key_value_state_evolves_across_blocks",
        ),
        (("model", "target_self_context_edge"), True, "target_self_context_edge"),
        (("model", "output_distribution"), "poisson", "output_distribution"),
    ],
)
def test_target_isolated_nb_contract_drift_is_rejected(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    config = deepcopy(_config())
    _set_nested(config, path, value)
    with pytest.raises(ConfigurationError, match=message):
        validate_experiment_config(config)
