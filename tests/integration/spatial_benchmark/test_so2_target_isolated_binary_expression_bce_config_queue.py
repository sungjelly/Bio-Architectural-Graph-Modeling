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
    "so2_target_isolated_geometry_modulated_binary_expression_bce_"
    "once_per_cell_seed0.yaml"
)
COMPARATOR_CONFIG_PATH = CONFIG_ROOT / (
    "experiment/"
    "so2_target_isolated_geometry_modulated_hurdle_nb_once_per_cell_seed0.yaml"
)
CAMPAIGN_ID = (
    "cmp_20260930_so2_target_isolated_geometry_modulated_binary_expression_"
    "bce_once_per_cell_seed0"
)
PROTOCOL = (
    "donor_grouped_so2_target_isolated_binary_expression_bce_"
    "once_per_cell_earlystop_v1"
)
PRIMARY_METRIC = "val/target_only_masked_binary_cross_entropy"


def _config() -> dict[str, Any]:
    return compose_config(CONFIG_PATH, config_root=CONFIG_ROOT)


def _set_nested(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = mapping
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


def test_target_isolated_binary_expression_bce_protocol_is_exactly_locked() -> None:
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
    assert config["model"]["output_distribution"] == "bernoulli_binary_expression"
    assert config["model"]["decoder_trunk"] == "binary_expression_only"
    assert config["model"]["detection_logit_projection_output_dim"] == 1_000
    assert config["model"]["expected_trainable_parameter_count"] == 5_134_088
    for forbidden in (
        "count_mean_projection_output_dim",
        "inverse_dispersion",
        "inverse_dispersion_initial_value",
    ):
        assert forbidden not in config["model"]
    assert config["trainer"]["class_reweighting"] == (
        "none_natural_observed_frequency"
    )
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


def test_binary_expression_campaign_reuses_hurdle_data_masks_and_seeds() -> None:
    binary = _config()
    comparator = compose_config(COMPARATOR_CONFIG_PATH, config_root=CONFIG_ROOT)

    for field in (
        "dataset_fingerprint",
        "split_fingerprint",
        "preprocessing_fingerprint",
        "prepared_artifact_reference",
        "training_core_aliases",
        "validation_core_aliases",
    ):
        assert binary["dataset"][field] == comparator["dataset"][field]
    for field in (
        "mask_base_seed",
        "training_mask_seed_namespace",
        "training_mask_seed_derivation_fields",
    ):
        assert binary["masking"][field] == comparator["masking"][field]
    assert binary["masking"]["validation_masks"] == comparator["masking"][
        "validation_masks"
    ]
    for field in ("core_order_seed", "target_shard_seed"):
        assert binary["trainer"][field] == comparator["trainer"][field]


def test_binary_expression_queue_routes_to_dedicated_four_rank_runner(
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
        str(
            tmp_path
            / "scripts/train/run_so2_target_isolated_binary_expression_bce.py"
        ),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_binary_expression_metric_schema_records_probabilistic_and_class_metrics() -> None:
    registry = load_yaml_mapping(CONFIG_ROOT / "schema/metrics_v1.yaml")
    family = registry["task_families"][
        "target_isolated_masked_binary_expression_classification"
    ]
    assert family["primary_metric"] == PRIMARY_METRIC
    assert family["primary_direction"] == "minimize"
    contract = family["likelihood_contract"]
    assert contract["family"] == "bernoulli_binary_expression"
    assert contract["optimization_aggregation"] == (
        "total_binary_cross_entropy_divided_by_total_masked_entries"
    )
    assert contract["class_reweighting"] == "none_natural_observed_frequency"
    assert contract["paired_hurdle_masks_reused_exactly"] is True
    assert contract["test_partition_present"] is False
    metrics = family["metrics"]
    for name in (
        "val/unseen_donor/detection_brier_score",
        "val/unseen_donor/detection_expected_calibration_error_histogram_4096",
        "val/unseen_donor/detection_auroc_histogram_4096",
        "val/unseen_donor/detection_auprc_histogram_4096",
        "val/unseen_donor/detection_accuracy",
        "val/unseen_donor/detection_balanced_accuracy",
        "val/unseen_donor/detection_precision",
        "val/unseen_donor/detection_recall",
        "val/unseen_donor/detection_f1",
        "val/unseen_donor/detection_matthews_correlation_coefficient",
    ):
        assert name in metrics


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("dataset", "training_cells"), 208_695, "training_cells"),
        (("evaluation", "splits"), ["validation", "test"], "no test"),
        (("evaluation", "probability_histogram_bins"), 1_024, "histogram_bins"),
        (("masking", "mask_base_seed"), 2026093001, "masking"),
        (("trainer", "target_shards_per_core"), 2, "target_shards_per_core"),
        (("trainer", "positive_class_weight"), 10.0, "positive_class_weight"),
        (("trainer", "checkpoint_selection"), "min_delta", "checkpoint_selection"),
        (("model", "graph_layers"), 8, "graph_layers"),
        (("model", "expected_trainable_parameter_count"), 5_135_088, "parameter"),
        (("model", "output_distribution"), "negative_binomial_nb2", "output_distribution"),
        (("model", "detection_projection_weight_initialization"), "xavier_uniform", "detection_projection_weight_initialization"),
    ],
)
def test_binary_expression_contract_drift_is_rejected(
    path: tuple[str, ...], value: Any, message: str
) -> None:
    config = deepcopy(_config())
    _set_nested(config, path, value)
    with pytest.raises(ConfigurationError, match=message):
        validate_experiment_config(config)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "count_mean_projection_output_dim",
        "inverse_dispersion",
        "inverse_dispersion_initial_value",
    ],
)
def test_binary_expression_model_rejects_count_or_dispersion_fields(
    forbidden_field: str,
) -> None:
    config = deepcopy(_config())
    config["model"][forbidden_field] = 1_000
    with pytest.raises(ConfigurationError, match="prohibits count"):
        validate_experiment_config(config)
