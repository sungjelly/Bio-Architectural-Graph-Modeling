from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest
import torch

from spatial_benchmark.configuration import (
    ConfigurationError,
    compose_config,
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import command_for_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "configs"
GEOMETRY_MODULATED_CONFIG = (
    CONFIG_ROOT
    / "experiment/so2_14core_geometry_modulated_relative_qkv_seed0_batch2.yaml"
)
AUGUST25_CONFIG = (
    CONFIG_ROOT / "experiment/so2_14core_relative_qkv_seed0_batch2.yaml"
)
CAMPAIGN_ROOT = (
    PROJECT_ROOT
    / "experiments/campaigns/"
    "cmp_20260903_so2_14core_geometry_modulated_relative_qkv_seed0_batch2"
)

_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "_so2_geometry_modulated_config_queue_runner",
    PROJECT_ROOT / "scripts/train/run_so2_14core_relative_qkv.py",
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)


def _geometry_modulated_config() -> dict[str, Any]:
    return compose_config(GEOMETRY_MODULATED_CONFIG, config_root=CONFIG_ROOT)


def _august25_config() -> dict[str, Any]:
    return compose_config(AUGUST25_CONFIG, config_root=CONFIG_ROOT)


def _set_nested(
    mapping: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    target = mapping
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


def test_registered_config_has_exact_four_block_score_identity_and_august25_parity() -> None:
    candidate = _geometry_modulated_config()
    reference = _august25_config()

    validate_experiment_config(candidate)
    assert canonical_sha256(candidate) == (
        "b9ac95057d56084ee592d70f27b949293c595b5c7ceb6483b649f8672506bfe9"
    )
    assert candidate["campaign"]["campaign_id"] == (
        "cmp_20260903_so2_14core_geometry_modulated_relative_qkv_seed0_batch2"
    )
    assert candidate["evaluation"]["protocol"] == (
        "held_in_pooled_14core_geometry_modulated_relative_qkv_seed_plateau"
    )
    assert (candidate["seed"], candidate["fold"], candidate["attempt"]) == (0, 0, 1)

    model = candidate["model"]
    expected_score_contract = {
        "name": "geometry-modulated-relative-qkv-gat",
        "family": "geometry_modulated_relative_qkv_graph_transformer",
        "graph_layers": 4,
        "unique_graph_blocks": 4,
        "effective_graph_depth": 4,
        "graph_block_weight_tying": "none",
        "attention_heads": 8,
        "attention_head_dim": 32,
        "relative_geometry_dim": 70,
        "geometry_hidden_dim": 128,
        "attention_score_mechanism": "geometry_modulated_cosine_qkv_v1",
        "qk_normalization": "per_head_l2",
        "qk_normalization_epsilon": 0.000001,
        "modulation_activation": "tanh",
        "modulation_amplitude": 0.5,
        "modulation_raw_range": [0.5, 1.5],
        "modulation_mean_normalization": True,
        "modulation_mean_clamp_min": 0.000001,
        "modulation_projection_bias": False,
        "modulation_final_zero_init": True,
        "geometry_bias_activation": "tanh",
        "geometry_bias_bound": 1.0,
        "geometry_bias_projection_bias": False,
        "geometry_bias_final_zero_init": True,
        "logit_scale_parameterization": "bounded_sigmoid",
        "logit_scale_minimum": 0.1,
        "logit_scale_initial": 1.8856180831641267,
        "logit_scale_maximum": 20.0,
        "relative_geometry_role": "attention_logit_modulation_and_bias_only",
        "relative_geometry_value_injection": False,
        "value_content_source": "expression_derived_node_embedding_only",
        "receiver_chunk_size": 128,
        "max_edges_per_chunk": 50000,
        "fp32_attention_scoring": True,
        "fp32_attention_accumulation": True,
        "edge_key_vectors": False,
        "edge_value_vectors": False,
        "edge_value_gates": False,
    }
    assert {field: model[field] for field in expected_score_contract} == (
        expected_score_contract
    )
    assert "recurrent_unroll_steps" not in model

    for unchanged_section in ("dataset", "graph", "masking", "trainer"):
        assert candidate[unchanged_section] == reference[unchanged_section]

    candidate_features = deepcopy(candidate["features"])
    reference_features = deepcopy(reference["features"])
    candidate_role = candidate_features["relative_positional_encoding"].pop("role")
    reference_role = reference_features["relative_positional_encoding"].pop("role")
    assert candidate_features == reference_features
    assert reference_role == "attention_logit_bias_only"
    assert candidate_role == "attention_logit_modulation_and_bias_only"
    assert candidate["features"]["use_edge_features"] is False


def test_checkpoint_loss_and_gradient_observability_contract_is_storage_bounded() -> None:
    candidate = _geometry_modulated_config()
    trainer = candidate["trainer"]
    diagnostics = candidate["metadata"]["gradient_diagnostics"]
    layerwise = diagnostics["layerwise"]
    frozen_contract = load_yaml_mapping(CAMPAIGN_ROOT / "frozen_task_contract.yaml")

    assert trainer["checkpoint_policy"] == "atomic_latest_then_final_last_only"
    assert trainer["checkpoint_every_global_epochs"] == 1
    assert trainer["checkpoint_at_final_epoch"] is True
    assert trainer["restore_best"] is False
    assert trainer["primary_checkpoint_role"] == "last"
    assert trainer["epoch_metrics_csv"] == "results/epoch_metrics.csv"
    assert trainer["epoch_metrics_fsync"] is True

    checkpointing = frozen_contract["checkpointing"]
    assert checkpointing == {
        "epoch_loss_recording": "every_completed_epoch_fsynced",
        "in_training_path": "checkpoints/latest.ckpt",
        "in_training_checkpoint_count": 1,
        "replacement": "atomic_every_completed_epoch",
        "epoch_archive_allowed": False,
        "best_checkpoint_selection_allowed": False,
        "successful_final_path": "checkpoints/last.ckpt",
        "successful_final_checkpoint_count": 1,
    }

    assert diagnostics["enabled"] is True
    assert diagnostics["output_path"] == "results/gradient_direction_metrics.csv"
    assert diagnostics["global_epoch_indexing"] == "one_based"
    assert diagnostics["vector_dtype"] == "float32"
    assert diagnostics["gradient_scope"] == (
        "full_ddp_averaged_all_trainable_parameters"
    )
    assert diagnostics["read_timing"] == (
        "after_amp_unscale_and_finite_check_before_gradient_clipping_or_optimizer_step"
    )
    assert diagnostics["consecutive_optimizer_step_cosine"] is True
    assert diagnostics["epoch_aggregate_gradient_cosine_to_previous_epoch"] is True
    assert diagnostics["persistence"] == "global_epoch_scalars_only"
    assert {
        diagnostics["persist_gradient_tensors"],
        diagnostics["persist_per_optimizer_step_files"],
        diagnostics["persist_gradient_vectors_in_checkpoints"],
        diagnostics["affects_optimization_or_plateau_stopping"],
    } == {False}

    assert layerwise["enabled"] is True
    assert layerwise["scope"] == "graph_blocks_only"
    assert layerwise["block_names"] == [
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
    ]
    assert layerwise["output_path"] == "results/gradient_direction_by_block.csv"
    assert layerwise["rows_per_completed_global_epoch"] == 4
    assert {
        layerwise["persist_gradient_tensors"],
        layerwise["persist_per_optimizer_step_files"],
        layerwise["persist_gradient_vectors_in_checkpoints"],
        layerwise["affects_optimization_or_plateau_stopping"],
    } == {False}


def test_queue_routes_registered_config_through_four_rank_tracked_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        _geometry_modulated_config(),
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
        str(tmp_path / "scripts/train/run_so2_14core_relative_qkv.py"),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model", "graph_layers"), 8, "model.graph_layers"),
        (
            ("model", "attention_score_mechanism"),
            "scaled_dot_product",
            "attention_score_mechanism",
        ),
        (
            ("model", "relative_geometry_value_injection"),
            True,
            "relative_geometry_value_injection",
        ),
        (
            ("features", "relative_positional_encoding", "role"),
            "attention_logit_bias_only",
            "relative geometry role",
        ),
        (("trainer", "learning_rate"), 0.0002, "trainer"),
        (("graph", "self_loops"), True, "graph"),
        (("masking", "model_seed_in_mask_derivation"), True, "mask"),
        (("dataset", "total_fit_cells"), 246062, "dataset"),
        (
            ("metadata", "gradient_diagnostics", "output_path"),
            "results/other.csv",
            "metadata",
        ),
        (("model", "recurrent_unroll_steps"), 4, "recurrent_unroll_steps"),
    ],
)
def test_scientific_or_observability_mutations_are_rejected(
    path: tuple[str, ...],
    value: Any,
    message: str,
) -> None:
    drifted = deepcopy(_geometry_modulated_config())
    _set_nested(drifted, path, value)

    with pytest.raises(ConfigurationError, match=message):
        validate_experiment_config(drifted)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("model", "name"), "relative-qkv-gat"),
        (("campaign", "campaign_id"), "cmp_wrong"),
        (("launcher", "requested_gpu"), "0,1"),
        (("launcher", "process_count"), 2),
        (("launcher", "elastic_max_restarts"), 1),
    ],
)
def test_queue_rejects_model_campaign_or_four_rank_launcher_mutation(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
) -> None:
    drifted = deepcopy(_geometry_modulated_config())
    _set_nested(drifted, path, value)

    with pytest.raises(ConfigurationError, match="geometry-modulated.*four-rank"):
        command_for_config(
            drifted,
            paths=ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)}),
        )


def test_runner_constructs_exact_parameter_count_and_four_untied_blocks() -> None:
    candidate = _geometry_modulated_config()
    _RUNNER._validate_contract(candidate)
    model = _RUNNER._model_from_config(
        candidate,
        num_genes=1000,
        node_covariate_dim=22,
    )
    topology = _RUNNER._geometry_modulated4_block_topology(model)
    construction = _RUNNER._model_construction(
        model,
        candidate,
        num_genes=1000,
        node_covariate_dim=22,
    )

    assert type(model).__name__ == (
        "ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer"
    )
    assert isinstance(model.blocks, torch.nn.ModuleList)
    assert len(model.blocks) == 4
    assert len({id(block) for block in model.blocks}) == 4
    parameter_sets = [
        {id(parameter) for parameter in block.parameters()} for block in model.blocks
    ]
    assert all(parameter_sets)
    assert all(
        parameter_sets[left].isdisjoint(parameter_sets[right])
        for left in range(4)
        for right in range(left + 1, 4)
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        _RUNNER.EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT
    )
    assert topology == {
        "verified": True,
        "graph_block_count": 4,
        "unique_graph_block_objects": 4,
        "unique_graph_block_parameter_sets": 4,
        "state_dict_block_indices": [0, 1, 2, 3],
        "graph_block_weight_tying": "none",
    }
    assert construction["class"] == type(model).__name__
    assert construction["attention_score_mechanism"] == (
        "geometry_modulated_cosine_qkv_v1"
    )
    assert construction["relative_geometry_value_injection"] is False
