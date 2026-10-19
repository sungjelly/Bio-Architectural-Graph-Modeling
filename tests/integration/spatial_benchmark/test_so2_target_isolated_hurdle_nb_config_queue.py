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
    "so2_target_isolated_geometry_modulated_hurdle_nb_once_per_cell_seed0.yaml"
)
COMPARATOR_CONFIG_PATH = CONFIG_ROOT / (
    "experiment/"
    "so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0.yaml"
)
CAMPAIGN_ID = (
    "cmp_20260928_so2_target_isolated_geometry_modulated_hurdle_nb_"
    "once_per_cell_seed0"
)
PROTOCOL = (
    "donor_grouped_so2_target_isolated_hurdle_nb_once_per_cell_earlystop_v1"
)
PRIMARY_METRIC = "val/target_only_masked_hurdle_negative_binomial_nll"


def _config() -> dict[str, Any]:
    return compose_config(CONFIG_PATH, config_root=CONFIG_ROOT)


def _set_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = mapping
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


def test_target_isolated_hurdle_nb_protocol_is_exactly_locked() -> None:
    config = _config()
    validate_experiment_config(config)

    assert config["campaign"]["campaign_id"] == CAMPAIGN_ID
    assert config["evaluation"]["protocol"] == PROTOCOL
    assert config["evaluation"]["primary_metric"] == PRIMARY_METRIC
    assert config["evaluation"]["splits"] == ["validation"]
    assert config["evaluation"]["test_split_present"] is False
    assert config["dataset"]["training_core_aliases"] == [
        f"SO2-C{core}" for core in range(15, 27)
    ]
    assert config["dataset"]["validation_core_aliases"] == [
        "SO2-C27",
        "SO2-C28",
    ]
    assert config["dataset"]["test_core_aliases"] == []
    assert config["model"]["output_distribution"] == (
        "bernoulli_hurdle_zero_truncated_negative_binomial_nb2"
    )
    assert config["model"]["decoder_trunk"] == "shared_count_and_detection"
    assert config["model"]["count_mean_projection_output_dim"] == 1_000
    assert config["model"]["detection_logit_projection_output_dim"] == 1_000
    assert config["model"]["expected_trainable_parameter_count"] == 6_160_088
    assert config["trainer"]["target_shards_per_core"] == 4
    assert config["trainer"]["optimizer_updates_per_global_epoch"] == 6
    assert config["trainer"]["checkpoint_selection"] == (
        "strict_lowest_validation_primary_metric"
    )
    contract = PROJECT_ROOT / (
        "experiments/campaigns/"
        f"{CAMPAIGN_ID}/frozen_task_contract.yaml"
    )
    assert hashlib.sha256(contract.read_bytes()).hexdigest() == config["metadata"][
        "frozen_task_contract_sha256"
    ]


def test_hurdle_campaign_reuses_comparator_data_masks_and_execution_seeds() -> None:
    hurdle = _config()
    comparator = compose_config(COMPARATOR_CONFIG_PATH, config_root=CONFIG_ROOT)

    for field in (
        "dataset_fingerprint",
        "split_fingerprint",
        "preprocessing_fingerprint",
        "prepared_artifact_reference",
        "training_core_aliases",
        "validation_core_aliases",
    ):
        assert hurdle["dataset"][field] == comparator["dataset"][field]
    for field in (
        "mask_base_seed",
        "training_mask_seed_namespace",
        "training_mask_seed_derivation_fields",
    ):
        assert hurdle["masking"][field] == comparator["masking"][field]
    assert hurdle["masking"]["validation_masks"] == comparator["masking"][
        "validation_masks"
    ]
    for field in ("core_order_seed", "target_shard_seed"):
        assert hurdle["trainer"][field] == comparator["trainer"][field]


def test_target_isolated_hurdle_nb_queue_routes_to_dedicated_four_rank_runner(
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
        str(tmp_path / "scripts/train/run_so2_target_isolated_hurdle_nb.py"),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_target_isolated_hurdle_metric_schema_preserves_joint_objective() -> None:
    registry = load_yaml_mapping(CONFIG_ROOT / "schema/metrics_v1.yaml")
    family = registry["task_families"][
        "target_isolated_masked_expression_hurdle_negative_binomial"
    ]
    assert family["primary_metric"] == PRIMARY_METRIC
    assert family["primary_direction"] == "minimize"
    contract = family["likelihood_contract"]
    assert contract["family"] == (
        "bernoulli_hurdle_zero_truncated_negative_binomial_nb2"
    )
    assert contract["optimization_aggregation"] == (
        "total_joint_nll_divided_by_total_masked_entries"
    )
    assert contract["positive_stratum_reweighting"] is False
    assert contract["paired_ordinary_nb_masks_reused_exactly"] is True
    assert contract["test_partition_present"] is False
    metrics = family["metrics"]
    for name in (
        "val/unseen_donor/positive_raw_count_mae",
        "val/unseen_donor/positive_raw_count_rmse",
        "val/unseen_donor/positive_log1p_mae",
        "val/unseen_donor/positive_log1p_rmse",
        "val/unseen_donor/observed_positive_rate",
        "val/unseen_donor/predicted_positive_probability_mean",
        "val/unseen_donor/detection_brier_score",
        "val/unseen_donor/detection_true_positive",
        "val/unseen_donor/detection_true_negative",
        "val/unseen_donor/detection_false_positive",
        "val/unseen_donor/detection_false_negative",
        "val/unseen_donor/detection_recall",
        "val/unseen_donor/detection_specificity",
        "val/unseen_donor/detection_precision",
        "val/unseen_donor/detection_f1",
    ):
        assert name in metrics


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dataset", "training_cells"), 208_695, "training_cells"),
        (("evaluation", "splits"), ["validation", "test"], "no test"),
        (("masking", "mask_base_seed"), 2026092801, "masking"),
        (
            ("masking", "training_mask_seed_namespace"),
            "bagm.so2.hurdle.new_masks.v1",
            "masking",
        ),
        (("trainer", "target_shards_per_core"), 2, "target_shards_per_core"),
        (("trainer", "checkpoint_selection"), "min_delta", "checkpoint_selection"),
        (("model", "graph_layers"), 8, "graph_layers"),
        (("model", "expected_trainable_parameter_count"), 5_135_088, "parameter"),
        (
            ("model", "output_distribution"),
            "zero_inflated_negative_binomial",
            "output_distribution",
        ),
        (
            ("model", "detection_projection_weight_initialization"),
            "xavier_uniform",
            "detection_projection_weight_initialization",
        ),
    ],
)
def test_target_isolated_hurdle_nb_contract_drift_is_rejected(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    config = deepcopy(_config())
    _set_nested(config, path, value)
    with pytest.raises(ConfigurationError, match=message):
        validate_experiment_config(config)
