from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from spatial_benchmark.configuration import (
    ConfigurationError,
    compose_config,
    validate_experiment_config,
)
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import command_for_config


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "so12_pearson_relative_qkv_train_validation_v1"


@pytest.fixture
def config():
    value = compose_config(
        ROOT / "configs/experiment/so2_14core_relative_qkv_seed0_batch2.yaml",
        validate=False,
    )
    value["campaign"]["campaign_id"] = "cmp_20261016_so12_pearson_relative_qkv_train_validation"
    value["classification"]["lifecycle_stage"] = "exploratory_screen"
    value["dataset"].update({
        "dataset_id": "so12_pearson_training_v1",
        "split_id": "so12_donor_train_validation_v1",
        "target_scale": "train_standardized_quasipoisson_pearson_phi1p01_clip10",
        "input_scale": "observed_only_quasipoisson_pearson_frozen_train_statistics",
        "training_core_aliases": [
            *[f"SO1-C{core:02d}" for core in range(1, 11)],
            *[f"SO2-C{core:02d}" for core in range(15, 25)],
        ],
        "validation_core_aliases": [
            *[f"SO1-C{core:02d}" for core in range(11, 15)],
            *[f"SO2-C{core:02d}" for core in range(25, 29)],
        ],
    })
    value["trainer"].update({
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "minimum_global_epochs": 50,
        "max_epochs": 300,
        "early_stopping": True,
        "early_stopping_patience": 25,
        "early_stopping_min_delta": 1e-4,
        "restore_best": True,
        "primary_checkpoint_role": "best",
        "checkpoint_policy": "best_and_latest",
    })
    value["evaluation"] = {
        "protocol": PROTOCOL,
        "canonical_prediction_split": "validation",
        "splits": ["validation"],
        "primary_metric": "val/masked_huber",
        "primary_direction": "minimize",
        "validation_mask_views": 3,
    }
    return value


def test_new_protocol_dispatches_four_rank_validation_runner(config, tmp_path):
    validate_experiment_config(config)
    command = command_for_config(
        config, paths=ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})
    )
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc-per-node=4" in command
    assert "--max-restarts=0" in command
    assert str(tmp_path / "scripts/train/run_so12_relative_qkv.py") in command
    assert command[-4:] == [
        "--config", "{run_scratch}/config.resolved.yaml",
        "--run-scratch", "{run_scratch}",
    ]


@pytest.mark.parametrize("section,field,value", [
    ("model", "graph_layers", 1),
    ("model", "edge_value_gates", True),
    ("dataset", "input_scale", "fully_observed_precomputed_pearson"),
    ("dataset", "target_scale", "signed_log_pearson"),
    ("trainer", "learning_rate", 3e-4),
    ("trainer", "scheduler", "cosine"),
    ("trainer", "primary_checkpoint_role", "last"),
    ("trainer", "restore_best", False),
    ("evaluation", "validation_mask_views", 10),
    ("evaluation", "canonical_prediction_split", "fit"),
    ("launcher", "process_count", 1),
])
def test_protocol_rejects_architecture_leakage_and_selection_drift(
    config, section, field, value,
):
    config[section][field] = value
    with pytest.raises(ConfigurationError):
        validate_experiment_config(config)
    with pytest.raises(ConfigurationError):
        command_for_config(config)


def test_protocol_rejects_training_validation_overlap(config):
    config["dataset"]["validation_core_aliases"][0] = config["dataset"]["training_core_aliases"][0]
    with pytest.raises(ConfigurationError, match="validation_core_aliases"):
        validate_experiment_config(config)


def test_original_held_in_protocol_remains_distinct():
    original = compose_config(
        ROOT / "configs/experiment/so2_14core_relative_qkv_seed0_batch2.yaml"
    )
    assert original["evaluation"]["canonical_prediction_split"] == "fit"
    assert any(part.endswith("run_so2_14core_relative_qkv.py") for part in command_for_config(original))


def test_runner_requires_worker_owned_production_scratch(config, tmp_path, monkeypatch):
    from scripts.train.run_so12_relative_qkv import main

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.delenv("BAGM_RUN_ID", raising=False)
    monkeypatch.delenv("BAGM_RUN_SCRATCH", raising=False)
    with pytest.raises(ValueError, match="queue-owned"):
        main(["--config", str(config_path), "--run-scratch", str(tmp_path / "arbitrary")])


def test_preflight_uses_the_same_validated_scientific_config(config, tmp_path, monkeypatch):
    from scripts.train.run_so12_relative_qkv import main

    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config))
    received = []
    monkeypatch.setitem(sys.modules, "spatial_benchmark.so12_relative_qkv_training", SimpleNamespace(
        run_training=lambda *args, **kwargs: received.append((args, kwargs)),
    ))
    monkeypatch.delenv("BAGM_RUN_ID", raising=False)
    monkeypatch.delenv("BAGM_RUN_SCRATCH", raising=False)
    output = tmp_path / "pilot"
    assert main(["--config", str(config_path), "--run-scratch", str(output), "--preflight"]) == 0
    assert received[0][0][0] == config
    assert received[0][0][1] == output
    assert received[0][1] == {"preflight": True}
