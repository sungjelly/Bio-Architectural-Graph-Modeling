from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from spatial_benchmark.configuration import (
    ConfigurationError,
    compose_config,
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.identifiers import canonical_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"
CANDIDATE_PATH = (
    CONFIG_ROOT
    / "experiment/"
    "so1_14core_geometry_modulated_relative_qkv_seed0_batch2_plateau_min150.yaml"
)
SO1_REFERENCE_PATH = (
    CONFIG_ROOT
    / "experiment/so1_14core_relative_qkv_seed0_batch2_plateau_min150.yaml"
)
SO2_ARCHITECTURE_PATH = (
    CONFIG_ROOT
    / "experiment/so2_14core_geometry_modulated_relative_qkv_seed0_batch2.yaml"
)
CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "experiments/campaigns/"
    "cmp_20260905_so1_14core_geometry_modulated_relative_qkv_"
    "seed0_batch2_plateau_min150"
)
CAMPAIGN_ID = (
    "cmp_20260905_so1_14core_geometry_modulated_relative_qkv_"
    "seed0_batch2_plateau_min150"
)
PROTOCOL = (
    "held_in_pooled_so1_14core_geometry_modulated_relative_qkv_plateau_min150"
)
ALIASES = tuple(f"SO1-C{core:02d}" for core in range(1, 15))
CELLS_BY_ALIAS = {
    "SO1-C01": 8924,
    "SO1-C02": 7450,
    "SO1-C03": 12190,
    "SO1-C04": 14657,
    "SO1-C05": 11399,
    "SO1-C06": 18212,
    "SO1-C07": 10722,
    "SO1-C08": 4972,
    "SO1-C09": 17223,
    "SO1-C10": 14756,
    "SO1-C11": 18145,
    "SO1-C12": 7816,
    "SO1-C13": 5345,
    "SO1-C14": 9785,
}
SECTION_HASHES = {
    "model": "f7efd848610b2e500c218fa18618385c4799cdeb8a7c3f70f4d4d1a63fbd2609",
    "dataset": "857a168e00fe5d840406e68a18ff371c3230b19c90a05cb74b6fab4ecbc24307",
    "features": "3fc95a458dce1e6fe1254241a36a0e53869d615122503106ae5b4ecbd3d22f50",
    "graph": "c9dc31acee7ed2860da818a5f5ab188d2b8bd9755f22ad0076b3685f1618a11a",
    "masking": "07f55d3adaf92d0db845b87d30cdb322289ccc8d33bec071fab55b7b705d0dc2",
    "trainer": "8c29a36d15416b16494da3580ef670a84ce64f45e716ba6cef587489d245ab03",
    "evaluation": "50fc836335738973ccaa2348a143fb1e27337a8ff73fb10cc5e5c95ebf1a1b1f",
    "launcher": "5d6e260aebc190373a7b050299e686e1634be912c5babe4d2d7ea0f59a5d9f32",
    "metadata": "9778ca8b9ac6c52c92d1dbbc95674fd30530d8cbddb10a3e18d2f00207a767d6",
    "classification": "0de8e08055425731a365a54fea8d1870142f73f7e705118a57dd152b3a3765eb",
    "experiment": "c105da0f147676297823df7765d2ee9d603ffe8091cff22725b68272a7e9cdc1",
    "campaign": "0c45ffba8386ceb46954167e15631fd80150806c289363f745be8eead0f6f1bc",
}
CONFIG_HASH = "2c5c561b988b2eb49d01229e0174f5806f74eea0853b85d581f19a78176bcece"
TASK_CONTRACT_HASH = (
    "e9e24ad010ff436d78b729ebf173eefb13cd15c3fe73bf60996e48a2d8f2a5ae"
)


def _compose(path: Path) -> dict[str, Any]:
    return compose_config(path, config_root=CONFIG_ROOT, validate=False)


def _candidate() -> dict[str, Any]:
    return _compose(CANDIDATE_PATH)


def _set_nested(
    mapping: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    target = mapping
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


def test_composed_config_has_registered_identity_and_frozen_section_hashes() -> None:
    candidate = _candidate()

    validate_experiment_config(candidate)
    assert candidate["campaign"]["campaign_id"] == CAMPAIGN_ID
    assert candidate["evaluation"]["protocol"] == PROTOCOL
    assert (candidate["seed"], candidate["fold"], candidate["attempt"]) == (0, 0, 1)
    assert canonical_sha256(candidate) == CONFIG_HASH
    assert {
        section: canonical_sha256(candidate[section]) for section in SECTION_HASHES
    } == SECTION_HASHES


def test_exact_so2_architecture_is_combined_with_literal_so1_protocol() -> None:
    candidate = _candidate()
    so1_reference = _compose(SO1_REFERENCE_PATH)
    so2_architecture = _compose(SO2_ARCHITECTURE_PATH)

    assert candidate["model"] == so2_architecture["model"]
    assert canonical_sha256(candidate["model"]) == (
        candidate["metadata"]["completed_so2_architecture_reference"][
            "resolved_model_mapping_sha256"
        ]
    )
    for section in ("dataset", "graph", "masking", "trainer"):
        assert candidate[section] == so1_reference[section]

    candidate_features = deepcopy(candidate["features"])
    reference_features = deepcopy(so1_reference["features"])
    candidate_role = candidate_features["relative_positional_encoding"].pop("role")
    reference_role = reference_features["relative_positional_encoding"].pop("role")
    assert candidate_features == reference_features
    assert reference_role == "attention_logit_bias_only"
    assert candidate_role == "attention_logit_modulation_and_bias_only"

    candidate_evaluation = deepcopy(candidate["evaluation"])
    reference_evaluation = deepcopy(so1_reference["evaluation"])
    assert candidate_evaluation.pop("protocol") == PROTOCOL
    assert reference_evaluation.pop("protocol") == (
        "held_in_pooled_so1_14core_relative_qkv_plateau_min150"
    )
    assert candidate_evaluation == reference_evaluation

    candidate_launcher = deepcopy(candidate["launcher"])
    reference_launcher = deepcopy(so1_reference["launcher"])
    assert candidate_launcher.pop("hardware_preflight_receipt") == (
        "state/preflight/so1_14core_geometry_modulated_relative_qkv_"
        "ddp4_plateau_min150.json"
    )
    assert reference_launcher.pop("hardware_preflight_receipt") == (
        "state/preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json"
    )
    assert candidate_launcher == reference_launcher

    architecture_reference = candidate["metadata"][
        "completed_so2_architecture_reference"
    ]
    assert architecture_reference["run_id"] == (
        "r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf"
    )
    assert architecture_reference["status"] == "success"
    assert architecture_reference["reuse_scope"] == (
        "exact_architecture_and_implementation_only"
    )
    assert architecture_reference["checkpoint_or_optimizer_state_reused"] is False
    assert candidate["metadata"]["initialization"] == "fresh_from_scratch_seed0"
    assert candidate["metadata"]["cross_campaign_resume_allowed"] is False


def test_locked_aliases_cell_counts_and_immutable_so1_hashes_are_exact() -> None:
    candidate = _candidate()
    contract = load_yaml_mapping(CAMPAIGN_ROOT / "frozen_task_contract.yaml")

    assert tuple(candidate["dataset"]["core_aliases"]) == ALIASES
    assert candidate["dataset"]["total_fit_cells"] == 161596
    assert contract["cohort"]["core_aliases"] == list(ALIASES)
    assert contract["cohort"]["cells_by_alias"] == CELLS_BY_ALIAS
    assert sum(CELLS_BY_ALIAS.values()) == 161596
    assert sorted(CELLS_BY_ALIAS, key=CELLS_BY_ALIAS.get, reverse=True)[:2] == [
        "SO1-C06",
        "SO1-C11",
    ]
    assert contract["preflight"]["selected_core_aliases"] == [
        "SO1-C06",
        "SO1-C11",
    ]
    assert contract["immutable_preparation"] == {
        "dataset_content_sha256": (
            "e006316e0f04afa645191544bcac8aa64f58c423e755f2f8d79bfd9db68a233d"
        ),
        "split_sha256": (
            "a0d2c008ff02471010585a023724f75d6f029d1a81f39f1f2dbba1531091b9a4"
        ),
        "cohort_manifest_file_sha256": (
            "15f9da492959c35d89020b3956047ec163a5f3537eaaefd5b8a011cb9279b440"
        ),
        "graph_content_sha256": (
            "5262453fc631c15a66f00f960de2a766a6142a4f1f7644b8b77a8784ec43d3b4"
        ),
        "graph_manifest_file_sha256": (
            "754e98fa1b2d8b488892c4effbf095cf26da6d99c927ee45e9ee01c2d64b978f"
        ),
        "completed_cohort_manifest_sha256": (
            "e078588668d9b2285db27da6cda8b7f1d144aa055065c3022e43048fd1951596"
        ),
    }


def test_strict_plateau_storage_and_gradient_contracts_are_preserved() -> None:
    candidate = _candidate()
    so1_reference = _compose(SO1_REFERENCE_PATH)
    so2_architecture = _compose(SO2_ARCHITECTURE_PATH)
    contract = load_yaml_mapping(CAMPAIGN_ROOT / "frozen_task_contract.yaml")

    assert candidate["trainer"] == so1_reference["trainer"]
    assert candidate["trainer"]["continuation_policy"] == (
        "strict_training_loss_plateau_25_epoch_blocks"
    )
    assert candidate["trainer"]["plateau_absolute_relative_half_window_change_max"] == (
        0.0005
    )
    assert candidate["trainer"]["plateau_normalized_absolute_slope_per_epoch_max"] == (
        0.000025
    )
    assert candidate["trainer"]["checkpoint_policy"] == (
        "atomic_latest_then_final_last_only"
    )
    assert candidate["trainer"]["epoch_metrics_csv"] == "results/epoch_metrics.csv"
    assert candidate["trainer"]["epoch_metrics_fsync"] is True
    assert contract["checkpointing"] == {
        "epoch_loss_recording": "every_completed_epoch_fsynced",
        "in_training_path": "checkpoints/latest.ckpt",
        "in_training_checkpoint_count": 1,
        "replacement": "atomic_every_completed_epoch",
        "epoch_archive_allowed": False,
        "best_checkpoint_selection_allowed": False,
        "successful_final_path": "checkpoints/last.ckpt",
        "successful_final_checkpoint_count": 1,
    }

    diagnostics = candidate["metadata"]["gradient_diagnostics"]
    assert diagnostics == so2_architecture["metadata"]["gradient_diagnostics"]
    assert diagnostics["schema"] == "so2_full_gradient_direction_metrics_v1"
    assert diagnostics["layerwise"]["schema"] == (
        "so2_block_gradient_direction_metrics_v1"
    )
    assert diagnostics["layerwise"]["block_names"] == [
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
    ]
    assert diagnostics["layerwise"]["rows_per_completed_global_epoch"] == 4
    for scope in (diagnostics, diagnostics["layerwise"]):
        assert scope["persist_gradient_tensors"] is False
        assert scope["persist_per_optimizer_step_files"] is False
        assert scope["persist_gradient_vectors_in_checkpoints"] is False
        assert scope["affects_optimization_or_plateau_stopping"] is False


def test_campaign_and_frozen_task_contract_hashes_are_self_consistent() -> None:
    campaign = load_yaml_mapping(CAMPAIGN_ROOT / "campaign.yaml")
    contract_path = CAMPAIGN_ROOT / "frozen_task_contract.yaml"
    contract = load_yaml_mapping(contract_path)
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    checksum_line = (CAMPAIGN_ROOT / "frozen_task_contract.sha256").read_text(
        encoding="utf-8"
    )

    assert campaign["campaign_id"] == CAMPAIGN_ID
    assert campaign["frozen_task_contract_sha256"] == TASK_CONTRACT_HASH
    assert contract_hash == TASK_CONTRACT_HASH
    assert checksum_line == f"{TASK_CONTRACT_HASH}  frozen_task_contract.yaml\n"
    assert campaign["variant"]["expected_trainable_parameter_count"] == 5134088
    assert campaign["automatic_enqueue"] is False
    assert contract["resolved_config_locks"] == {
        "canonicalization": "sorted_keys_compact_json_sha256",
        "complete_resolved_config_sha256": CONFIG_HASH,
        "sections": SECTION_HASHES,
    }


@pytest.mark.parametrize(
    ("section", "path", "value"),
    [
        ("model", ("model", "graph_layers"), 8),
        (
            "features",
            ("features", "relative_positional_encoding", "role"),
            "attention_logit_bias_only",
        ),
        ("dataset", ("dataset", "total_fit_cells"), 161595),
        ("graph", ("graph", "self_loops"), True),
        ("masking", ("masking", "model_seed_in_mask_derivation"), True),
        ("trainer", ("trainer", "learning_rate"), 0.0002),
        (
            "evaluation",
            ("evaluation", "canonical_prediction_split"),
            "validation",
        ),
        ("launcher", ("launcher", "process_count"), 2),
        (
            "metadata",
            ("metadata", "gradient_diagnostics", "output_path"),
            "results/other.csv",
        ),
    ],
)
def test_frozen_section_hashes_detect_every_scientific_or_runtime_mutation(
    section: str,
    path: tuple[str, ...],
    value: Any,
) -> None:
    drifted = deepcopy(_candidate())
    _set_nested(drifted, path, value)

    assert canonical_sha256(drifted[section]) != SECTION_HASHES[section]
    with pytest.raises(ConfigurationError):
        validate_experiment_config(drifted)
