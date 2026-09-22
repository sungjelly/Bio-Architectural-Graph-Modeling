from __future__ import annotations

from copy import deepcopy
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
CONFIG_PATH = (
    CONFIG_ROOT / "experiment/so2_geometry_modulated_nb_train12_val2_seed0.yaml"
)
CAMPAIGN_ID = (
    "cmp_20260907_so2_geometry_modulated_relative_qkv_nb_train12_val2_seed0"
)
PROTOCOL = "donor_grouped_so2_geometry_modulated_nb_train12_val2_earlystop_v1"


def _config() -> dict[str, Any]:
    return compose_config(CONFIG_PATH, config_root=CONFIG_ROOT)


def _set_nested(
    mapping: dict[str, Any], path: tuple[str, ...], value: Any
) -> None:
    target = mapping
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


def test_so2_nb_config_is_train_validation_only_and_validation_selected() -> None:
    config = _config()
    validate_experiment_config(config)

    assert config["campaign"]["campaign_id"] == CAMPAIGN_ID
    assert config["evaluation"]["protocol"] == PROTOCOL
    assert config["evaluation"]["splits"] == ["validation"]
    assert config["evaluation"]["test_split_present"] is False
    assert config["evaluation"]["primary_metric"] == (
        "val/unseen_donor/masked_negative_binomial_nll"
    )
    assert config["dataset"]["training_core_aliases"] == [
        f"SO2-C{core}" for core in range(15, 27)
    ]
    assert config["dataset"]["validation_core_aliases"] == [
        "SO2-C27",
        "SO2-C28",
    ]
    assert config["dataset"]["test_core_aliases"] == []
    assert config["trainer"]["primary_checkpoint_role"] == "best"
    assert config["trainer"]["early_stopping"] is True
    assert config["trainer"]["minimum_global_epochs"] == 50
    assert config["trainer"]["early_stopping_patience"] == 25
    assert config["trainer"]["max_epochs"] == 300
    assert config["model"]["expected_trainable_parameter_count"] == 5_135_088


def test_so2_nb_value_stream_contract_names_shared_content_without_geometry() -> None:
    config = _config()
    model = config["model"]
    assert model["value_content_source"] == (
        "shared_node_content_state_without_direct_relative_geometry"
    )
    assert model["shared_node_content_state_initial_components"] == [
        "masked_train_standardized_log1p_expression",
        "explicit_gene_mask_channel",
        "train_normalized_allowed_morphology_metadata",
    ]
    assert model["shared_node_content_state_later_blocks"] == (
        "prior_node_content_messages_plus_residual_feed_forward_state"
    )
    assert model["qkv_projection_source"] == (
        "common_pre_attention_layer_normalized_shared_node_content_state_h"
    )
    assert model["relative_geometry_value_injection"] is False
    assert model["relative_geometry_direct_value_projection"] is False
    assert model["relative_geometry_value_gating"] is False

    campaign_root = PROJECT_ROOT / (
        "experiments/campaigns/"
        "cmp_20260907_so2_geometry_modulated_relative_qkv_nb_train12_val2_seed0"
    )
    frozen = load_yaml_mapping(campaign_root / "frozen_task_contract.yaml")
    attention = frozen["attention_score"]
    assert attention["value_content_source"] == model["value_content_source"]
    assert attention["shared_node_content_state_initial_components"] == model[
        "shared_node_content_state_initial_components"
    ]
    readme = (campaign_root / "README.md").read_text(encoding="utf-8")
    assert "shared_node_content_state_without_direct_relative_geometry" in readme
    assert "never injected into, projected into, or used to gate V" in readme


def test_so2_nb_queue_routes_to_dedicated_four_rank_runner(tmp_path: Path) -> None:
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
        str(tmp_path / "scripts/train/run_so2_geometry_modulated_nb.py"),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_so2_nb_primary_metric_is_registered_with_likelihood_semantics() -> None:
    registry = load_yaml_mapping(CONFIG_ROOT / "schema/metrics_v1.yaml")
    family = registry["task_families"]["masked_expression_negative_binomial"]
    assert family["primary_metric"] == (
        "val/unseen_donor/masked_negative_binomial_nll"
    )
    assert family["primary_direction"] == "minimize"
    assert family["likelihood_contract"]["full_likelihood_constants_included"] is True
    assert family["likelihood_contract"]["test_partition_present"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dataset", "training_core_aliases"), ["SO2-C15"], "12-core"),
        (("dataset", "validation_core_aliases"), ["SO2-C25"], "12-core"),
        (("dataset", "test_core_aliases"), ["SO2-C28"], "12-core"),
        (("evaluation", "splits"), ["validation", "test"], "no test"),
        (("trainer", "early_stopping"), False, "early_stopping"),
        (("trainer", "minimum_global_epochs"), 49, "minimum_global_epochs"),
        (("trainer", "early_stopping_patience"), 24, "early_stopping_patience"),
        (("trainer", "max_epochs"), 301, "max_epochs"),
        (("model", "graph_layers"), 8, "graph_layers"),
        (("model", "relative_geometry_value_injection"), True, "value_injection"),
        (
            ("model", "value_content_source"),
            "expression_derived_node_embedding_only",
            "value_content_source",
        ),
        (("model", "output_distribution"), "poisson", "output_distribution"),
    ],
)
def test_so2_nb_scientific_contract_drift_is_rejected(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    config = deepcopy(_config())
    _set_nested(config, path, value)
    with pytest.raises(ConfigurationError, match=message):
        validate_experiment_config(config)
