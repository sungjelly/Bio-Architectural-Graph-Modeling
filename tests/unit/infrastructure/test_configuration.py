from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from spatial_benchmark.configuration import (
    ConfigurationError,
    compose_config,
    load_yaml_mapping,
    validate_experiment_config,
)


GROUPS = {
    "model": (
        "model:\n  name: g2\n  family: edge_conditioned_gatv2\n"
        "  embedding_dim: 64\n"
    ),
    "masking": "masking:\n  type: partial_gene\n  rate: 0.2\n",
    "dataset": (
        "dataset:\n  dataset_id: gastric_cosmx\n  version: v1\n"
        "  split_id: patient_holdout_v1\n"
    ),
    "features": (
        "features:\n  use_edge_features: true\n"
        "  edge_features: [distance_um, contact]\n"
    ),
    "graph": (
        "graph:\n  neighbor_k: 16\n  radius_um: 75.0\n"
        "  symmetry: mutual\n"
    ),
    "trainer": "trainer:\n  learning_rate: 0.001\n  batch_size: 8\n",
    "evaluation": (
        "evaluation:\n  primary_metric: val/masked_huber\n"
        "  primary_direction: minimize\n"
    ),
}


def _configuration_tree(root: Path) -> Path:
    defaults = []
    for group, content in GROUPS.items():
        directory = root / group
        directory.mkdir(parents=True)
        (directory / "default.yaml").write_text(content, encoding="utf-8")
        defaults.append(f"  - {group}: default")
    base = root / "base.yaml"
    base.write_text(
        "defaults:\n"
        + "\n".join(defaults)
        + "\nmodel:\n  embedding_dim: 96\n"
        "seed: 3\nfold: 1\nattempt: 1\n"
        "campaign: cmp_20260724_edge_feature_ablation\n",
        encoding="utf-8",
    )
    return base


def test_strict_composition_merges_namespaced_groups(tmp_path: Path) -> None:
    base = _configuration_tree(tmp_path)

    resolved = compose_config(base, config_root=tmp_path)

    assert "defaults" not in resolved
    assert resolved["model"] == {
        "name": "g2",
        "family": "edge_conditioned_gatv2",
        "embedding_dim": 96,
    }
    assert resolved["dataset"]["split_id"] == "patient_holdout_v1"
    assert resolved["evaluation"]["primary_metric"] == "val/masked_huber"


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "duplicate.yaml"
    config.write_text("seed: 1\nseed: 2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Duplicate"):
        load_yaml_mapping(config)


def test_group_traversal_and_wrong_parameter_ownership_are_rejected(
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text("defaults:\n  - model: ../secret\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsafe"):
        compose_config(unsafe, config_root=tmp_path, validate=False)

    base = _configuration_tree(tmp_path / "valid")
    resolved = compose_config(base, config_root=tmp_path / "valid")
    resolved["trainer"]["embedding_dim"] = 64
    with pytest.raises(ConfigurationError, match="belongs in model"):
        validate_experiment_config(resolved)


def test_held_in_full_core_semantics_are_explicit_and_consistent(
    tmp_path: Path,
) -> None:
    base = _configuration_tree(tmp_path)
    resolved = compose_config(base, config_root=tmp_path)
    resolved["evaluation"] = {
        "protocol": "held_in_full_core_fixed_budget",
        "canonical_prediction_split": "fit",
        "splits": ["fit"],
        "primary_metric": "fit/whole_node/masked_huber",
        "primary_direction": "minimize",
    }
    resolved["trainer"].update(
        {
            "restore_best": False,
            "primary_checkpoint_role": "last",
            "checkpoint_policy": "last_only",
        }
    )
    validate_experiment_config(resolved)

    for field, value in (
        ("canonical_prediction_split", "validation"),
        ("splits", ["validation"]),
        ("primary_metric", "val/masked_huber"),
    ):
        invalid = deepcopy(resolved)
        invalid["evaluation"][field] = value
        with pytest.raises(ConfigurationError, match="held_in_full_core"):
            validate_experiment_config(invalid)

    invalid = deepcopy(resolved)
    invalid["trainer"]["restore_best"] = True
    with pytest.raises(ConfigurationError, match="restore_best=false"):
        validate_experiment_config(invalid)


def test_qkv_gat_requires_canonical_family_and_edge_features(
    tmp_path: Path,
) -> None:
    base = _configuration_tree(tmp_path)
    resolved = compose_config(base, config_root=tmp_path)
    resolved["model"].update(
        {
            "name": "qkv-gat",
            "family": "edge_aware_qkv_graph_transformer",
        }
    )
    validate_experiment_config(resolved)

    wrong_family = deepcopy(resolved)
    wrong_family["model"]["family"] = "edge_conditioned_gatv2"
    with pytest.raises(
        ConfigurationError,
        match="model.family=edge_aware_qkv_graph_transformer",
    ):
        validate_experiment_config(wrong_family)

    no_edges = deepcopy(resolved)
    no_edges["features"]["use_edge_features"] = False
    with pytest.raises(ConfigurationError, match="requires edge features on"):
        validate_experiment_config(no_edges)


def test_qkv_gat_matched_self_requires_canonical_family_and_no_edges(
    tmp_path: Path,
) -> None:
    base = _configuration_tree(tmp_path)
    resolved = compose_config(base, config_root=tmp_path)
    resolved["model"].update(
        {
            "name": "qkv-gat-matched-self",
            "family": "qkv_parameter_matched_self_control",
        }
    )
    resolved["features"] = {"use_edge_features": False}
    validate_experiment_config(resolved)

    wrong_family = deepcopy(resolved)
    wrong_family["model"]["family"] = "edge_aware_qkv_graph_transformer"
    with pytest.raises(
        ConfigurationError,
        match="model.family=qkv_parameter_matched_self_control",
    ):
        validate_experiment_config(wrong_family)

    with_edges = deepcopy(resolved)
    with_edges["features"] = {
        "use_edge_features": True,
        "edge_features": ["distance_um"],
    }
    with pytest.raises(ConfigurationError, match="requires edge features off"):
        validate_experiment_config(with_edges)


def test_relative_qkv_requires_logit_only_geometry_and_uniform_integer_masks(
    tmp_path: Path,
) -> None:
    base = _configuration_tree(tmp_path)
    resolved = compose_config(base, config_root=tmp_path)
    resolved["model"].update(
        {
            "name": "relative-qkv-gat",
            "family": "relative_geometry_qkv_graph_transformer",
            "uses_edge_inputs": False,
        }
    )
    resolved["features"] = {
        "use_edge_features": False,
        "relative_positional_encoding": {
            "role": "attention_logit_bias_only",
        },
    }
    resolved["dataset"]["task"] = "masked_expression_regression"
    resolved["masking"] = {
        "type": "uniform_per_cell_integer_count",
        "count_min": 0,
        "count_max": 1000,
        "positions_without_replacement": True,
        "model_seed_in_mask_derivation": False,
        "independent_views_per_core_epoch": 10,
        "ratio_stratification_or_bins": False,
        "mask_seed_derivation_fields": [
            "base_mask_seed",
            "core_alias",
            "global_epoch",
            "mask_view_index",
        ],
    }
    resolved["evaluation"] = {
        "protocol": "held_in_pooled_6core_relative_qkv_fixed_budget",
        "canonical_prediction_split": "fit",
        "splits": ["fit"],
        "primary_metric": "fit/uniform_per_cell/masked_huber",
        "primary_direction": "minimize",
    }
    resolved["trainer"].update(
        {
            "restore_best": False,
            "primary_checkpoint_role": "last",
            "checkpoint_policy": "periodic_and_last",
        }
    )
    validate_experiment_config(resolved)

    joint_plateau = deepcopy(resolved)
    joint_plateau["evaluation"]["protocol"] = (
        "held_in_pooled_6core_relative_qkv_joint_plateau"
    )
    joint_plateau["trainer"].update(
        {
            "max_epochs": 150,
            "minimum_global_epochs": 150,
            "fixed_epoch_budget": False,
            "continuation_policy": "joint_all_seed_plateau_25_epoch_blocks",
            "continuation_block_global_epochs": 25,
            "plateau_first_audit_epoch": 150,
            "plateau_window_global_epochs": 50,
            "plateau_consecutive_passing_audits": 2,
            "plateau_requires_all_five_seeds": True,
            "plateau_requires_common_final_epoch": True,
            "mask_views_per_core_step": 10,
            "optimizer_zero_grad_per_core_step": 1,
            "optimizer_steps_per_core_step": 1,
            "early_stopping": False,
        }
    )
    validate_experiment_config(joint_plateau)

    independent_plateau = deepcopy(joint_plateau)
    independent_plateau["seed"] = 1
    independent_plateau["evaluation"].update(
        {
            "protocol": "held_in_pooled_6core_relative_qkv_seed_plateau",
            "active_model_seeds": [0, 1, 2, 3],
        }
    )
    independent_plateau["trainer"].update(
        {
            "continuation_policy": (
                "independent_seed_training_loss_plateau_25_epoch_blocks"
            ),
            "plateau_requires_all_five_seeds": False,
            "plateau_requires_common_final_epoch": False,
        }
    )
    validate_experiment_config(independent_plateau)

    missing_active_seed = deepcopy(independent_plateau)
    missing_active_seed["evaluation"]["active_model_seeds"] = [0, 2, 3]
    with pytest.raises(ConfigurationError, match="current model seed"):
        validate_experiment_config(missing_active_seed)

    invalid = deepcopy(resolved)
    invalid["features"]["relative_positional_encoding"]["role"] = (
        "edge_value_gate"
    )
    with pytest.raises(ConfigurationError, match="logits only"):
        validate_experiment_config(invalid)

    invalid = deepcopy(resolved)
    invalid["masking"]["model_seed_in_mask_derivation"] = True
    with pytest.raises(ConfigurationError, match="exclude model seed"):
        validate_experiment_config(invalid)

    invalid = deepcopy(resolved)
    invalid["trainer"]["checkpoint_policy"] = "last_only"
    with pytest.raises(ConfigurationError, match="periodic_and_last"):
        validate_experiment_config(invalid)


def test_mean_adjacency_sage_requires_canonical_family(
    tmp_path: Path,
) -> None:
    base = _configuration_tree(tmp_path)
    resolved = compose_config(base, config_root=tmp_path)
    resolved["model"].update(
        {
            "name": "mean-adjacency-sage",
            "family": "explicit_self_mean_adjacency_graphsage",
        }
    )
    resolved["features"] = {"use_edge_features": False}
    validate_experiment_config(resolved)

    wrong_family = deepcopy(resolved)
    wrong_family["model"]["family"] = "edge_conditioned_gatv2"
    with pytest.raises(
        ConfigurationError,
        match="model.family=explicit_self_mean_adjacency_graphsage",
    ):
        validate_experiment_config(wrong_family)


def _hybrid_count_configuration(tmp_path: Path) -> dict[str, object]:
    base = _configuration_tree(tmp_path)
    resolved = compose_config(base, config_root=tmp_path)
    resolved["model"].update(
        {
            "name": "hybrid-count-gat",
            "family": "hybrid_count_edge_conditioned_gatv2",
        }
    )
    resolved["dataset"].update(
        {
            "task": "masked_expression_hybrid_count",
            "target_scale": "raw_biological_probe_counts",
        }
    )
    resolved["masking"]["type"] = "partial_gene_expression_masking"
    resolved["evaluation"] = {
        "task_family": "masked_expression_hybrid_count",
        "protocol": "held_in_full_core_fixed_budget",
        "canonical_prediction_split": "fit",
        "splits": ["fit"],
        "primary_metric": "fit/whole_node/hybrid_loss",
        "primary_direction": "minimize",
    }
    resolved["trainer"].update(
        {
            "restore_best": False,
            "primary_checkpoint_role": "last",
            "checkpoint_policy": "last_only",
        }
    )
    return resolved


def test_hybrid_count_arms_require_canonical_families_and_edge_states(
    tmp_path: Path,
) -> None:
    gat = _hybrid_count_configuration(tmp_path)
    validate_experiment_config(gat)

    self_control = deepcopy(gat)
    self_control["model"].update(
        {
            "name": "hybrid-count-matched-self",
            "family": "hybrid_count_parameter_matched_self_control",
        }
    )
    self_control["features"] = {"use_edge_features": False}
    validate_experiment_config(self_control)

    wrong_gat_family = deepcopy(gat)
    wrong_gat_family["model"]["family"] = "edge_conditioned_gatv2"
    with pytest.raises(
        ConfigurationError,
        match="model.family=hybrid_count_edge_conditioned_gatv2",
    ):
        validate_experiment_config(wrong_gat_family)

    self_with_edges = deepcopy(self_control)
    self_with_edges["features"] = {
        "use_edge_features": True,
        "edge_features": ["distance_um"],
    }
    with pytest.raises(ConfigurationError, match="requires edge features off"):
        validate_experiment_config(self_with_edges)


def test_pooled_hybrid_protocol_has_explicit_fit_semantics(
    tmp_path: Path,
) -> None:
    resolved = _hybrid_count_configuration(tmp_path)
    resolved["evaluation"]["protocol"] = (
        "held_in_pooled_10core_fixed_budget"
    )
    validate_experiment_config(resolved)

    invalid = deepcopy(resolved)
    invalid["evaluation"]["canonical_prediction_split"] = "validation"
    with pytest.raises(
        ConfigurationError,
        match="held_in_pooled_10core_fixed_budget",
    ):
        validate_experiment_config(invalid)


def test_myjju_genemae_requires_canonical_family_edges_and_partial_metric(
    tmp_path: Path,
) -> None:
    base = _configuration_tree(tmp_path)
    resolved = compose_config(base, config_root=tmp_path)
    resolved["model"].update(
        {
            "name": "myjju-genemae",
            "family": "myjju_dual_path_genemae",
            "embedding_dim": 192,
        }
    )
    resolved["dataset"]["task"] = "masked_expression_regression"
    resolved["masking"]["type"] = "partial_gene_expression_masking"
    resolved["evaluation"] = {
        "protocol": "held_in_pooled_10core_fixed_budget",
        "canonical_prediction_split": "fit",
        "splits": ["fit"],
        "primary_metric": "fit/partial_gene/log1p_cp10k_masked_huber",
        "primary_direction": "minimize",
    }
    resolved["trainer"].update(
        {
            "restore_best": False,
            "primary_checkpoint_role": "last",
            "checkpoint_policy": "last_only",
        }
    )
    validate_experiment_config(resolved)

    wrong_family = deepcopy(resolved)
    wrong_family["model"]["family"] = "edge_conditioned_gatv2"
    with pytest.raises(
        ConfigurationError,
        match="model.family=myjju_dual_path_genemae",
    ):
        validate_experiment_config(wrong_family)

    no_edges = deepcopy(resolved)
    no_edges["features"] = {"use_edge_features": False}
    with pytest.raises(ConfigurationError, match="requires edge features on"):
        validate_experiment_config(no_edges)


def _tokenized_g2_configuration(tmp_path: Path) -> dict[str, object]:
    base = _configuration_tree(tmp_path)
    resolved = compose_config(base, config_root=tmp_path)
    resolved["model"].update(
        {
            "name": "g2-tokenized",
            "family": "tokenized_edge_conditioned_gatv2",
            "num_expression_tokens": 4,
            "tokenizer_schema": "raw_count_tokens_0_1_2_3plus_v1",
        }
    )
    resolved["dataset"].update(
        {
            "task": "masked_expression_token_classification",
            "target_scale": "raw_count_token_0_1_2_3plus",
            "tokenization": {
                "schema": "raw_count_tokens_0_1_2_3plus_v1",
                "source_scale": "raw_biological_probe_counts",
                "num_output_tokens": 4,
                "mask_token_id": 4,
                "mask_token_is_output": False,
                "fixed_vocabulary": True,
                "fit_required": False,
                "count_mapping": {
                    "0": 0,
                    "1": 1,
                    "2": 2,
                    "3+": 3,
                },
            },
        }
    )
    resolved["masking"]["type"] = "partial_gene_expression_masking"
    resolved["evaluation"] = {
        "task_family": "masked_expression_token_classification",
        "protocol": "held_in_full_core_fixed_budget",
        "canonical_prediction_split": "fit",
        "splits": ["fit"],
        "primary_metric": "fit/whole_node/masked_token_accuracy_percent",
        "primary_direction": "maximize",
    }
    resolved["trainer"].update(
        {
            "restore_best": False,
            "primary_checkpoint_role": "last",
            "checkpoint_policy": "last_only",
        }
    )
    return resolved


def test_tokenized_g2_contract_is_accepted_only_as_one_matching_contract(
    tmp_path: Path,
) -> None:
    resolved = _tokenized_g2_configuration(tmp_path)
    validate_experiment_config(resolved)

    mismatches = (
        ("model", "name", "g2"),
        ("model", "family", "edge_conditioned_gatv2"),
        ("model", "num_expression_tokens", 5),
        ("model", "tokenizer_schema", "quantile_tokens_v1"),
        ("dataset", "task", "masked_expression_regression"),
        ("dataset", "target_scale", "standardized_log1p"),
        ("evaluation", "task_family", "classification"),
        ("evaluation", "primary_metric", "fit/whole_node/masked_huber"),
    )
    for section, field, value in mismatches:
        invalid = deepcopy(resolved)
        invalid[section][field] = value
        with pytest.raises(ConfigurationError):
            validate_experiment_config(invalid)

    for field, value in (
        ("schema", "raw_count_tokens_quantile_v1"),
        ("num_output_tokens", 5),
        ("mask_token_id", 0),
        ("mask_token_is_output", True),
        ("fixed_vocabulary", False),
        ("fit_required", True),
        ("source_scale", "standardized_log1p"),
        ("count_mapping", {"0": 0, "1+": 1}),
    ):
        invalid = deepcopy(resolved)
        invalid["dataset"]["tokenization"][field] = value
        with pytest.raises(
            ConfigurationError,
            match="matching task/model/tokenizer/metric contract",
        ):
            validate_experiment_config(invalid)


def test_token_metric_registry_matches_configuration_contract() -> None:
    project_root = Path(__file__).resolve().parents[3]
    registry = load_yaml_mapping(
        project_root / "configs/schema/metrics_v1.yaml"
    )
    family = registry["task_families"][
        "masked_expression_token_classification"
    ]
    assert family["primary_metric"] == (
        "fit/whole_node/masked_token_accuracy_percent"
    )
    assert family["primary_direction"] == "maximize"
    metrics = family["metrics"]
    for mode in ("partial_gene", "whole_node", "spatial_block"):
        prefix = f"fit/{mode}"
        for metric in (
            "masked_token_cross_entropy",
            "masked_token_accuracy_percent",
            "masked_token_balanced_accuracy_percent",
            "masked_nonzero_token_accuracy_percent",
            "baseline_uniform_accuracy_percent",
            "baseline_empirical_frequency_accuracy_percent",
            "baseline_always_zero_accuracy_percent",
            "baseline_always_zero_balanced_accuracy_percent",
            "baseline_per_gene_modal_accuracy_percent",
            "baseline_per_gene_modal_balanced_accuracy_percent",
            "baseline_per_gene_modal_nonzero_accuracy_percent",
            "token_0_recall_percent",
            "token_1_recall_percent",
            "token_2_recall_percent",
            "token_3_recall_percent",
        ):
            assert f"{prefix}/{metric}" in metrics
