#!/usr/bin/env python3
"""Train the SO2 cores 15--28 Relative-QKV model under four-rank DDP.

This entry point is launched only by the repository queue through
``torch.distributed.run``.  Torchrun remains the queue-owned child process and
uses ``--max-restarts=0``; recovery therefore starts explicitly from the last
atomic epoch-boundary checkpoint rather than restarting inside an epoch.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from spatial_benchmark.adjacency_ablation import sample_uniform_mask_numpy  # noqa: E402
from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.gradient_direction_observability import (  # noqa: E402
    BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
    BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
    GRADIENT_DIRECTION_METRICS_COLUMNS,
    GRADIENT_DIRECTION_METRICS_SCHEMA,
    BlockGradientDirectionTracker,
    DurableBlockGradientDirectionCSV,
    DurableGradientDirectionCSV,
    FullGradientDirectionTracker,
)
from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import (  # noqa: E402
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
)
from spatial_benchmark.masking import derive_mask_seed  # noqa: E402
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training import (  # noqa: E402
    _tree_sha256,
    SingleSeedPlateauDecision,
    single_seed_plateau_decision,
)
from spatial_benchmark.pooled_relative_qkv_training_v2 import (  # noqa: E402
    CohortRelativeQKVCoreRecord,
    CohortRelativeQKVEpochBoundaryResume,
    CohortRelativeQKVGlobalEpochRecord,
    CohortRelativeQKVOptimizerUpdateRecord,
    CohortRelativeQKVTrainingConfig,
    cohort_epoch_boundary_resume_from_checkpoint,
    fit_cohort_relative_qkv_segment,
)
from spatial_benchmark.relative_qkv_graph_transformer import (  # noqa: E402
    ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer,
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    deidentify_prediction_rows,
)
from spatial_benchmark.so2_pooled_full_core import (  # noqa: E402
    EXPECTED_TOTAL_CELLS,
    SO2_ALIASES,
)
from spatial_benchmark.so2_relative_graphs import (  # noqa: E402
    load_so2_relative_qkv_batches,
)
from spatial_benchmark.so2_training_observability import (  # noqa: E402
    AtomicLatestCheckpointStore,
    CHECKPOINT_SCHEMA,
    DurableEpochMetricsCSV,
    build_epoch_metrics_row,
    require_latest_only_checkpoint_layout,
    strict_plateau_monitor_fields,
)
from spatial_benchmark.so2_training_plots import (  # noqa: E402
    GRADIENT_FIGURE_RELATIVE_PATH,
    LOSS_FIGURE_RELATIVE_PATH,
    write_so2_training_plots,
)
from spatial_benchmark.training import _autocast_context, set_deterministic_seed  # noqa: E402


CAMPAIGN_ID = "cmp_20260825_so2_14core_relative_qkv_seed0_batch2"
PLATEAU_PROTOCOL = "held_in_pooled_14core_relative_qkv_seed_plateau"
RECURRENT_CAMPAIGN_ID = (
    "cmp_20260831_so2_14core_recurrent_relative_qkv_seed0_batch2"
)
RECURRENT_PLATEAU_PROTOCOL = (
    "held_in_pooled_14core_recurrent_relative_qkv_seed_plateau"
)
UNTIED8_CAMPAIGN_ID = (
    "cmp_20260903_so2_14core_untied8_relative_qkv_seed0_batch2"
)
UNTIED8_PLATEAU_PROTOCOL = (
    "held_in_pooled_14core_untied8_relative_qkv_seed_plateau"
)
GEOMETRY_MODULATED_CAMPAIGN_ID = (
    "cmp_20260903_so2_14core_geometry_modulated_relative_qkv_seed0_batch2"
)
GEOMETRY_MODULATED_PLATEAU_PROTOCOL = (
    "held_in_pooled_14core_geometry_modulated_relative_qkv_seed_plateau"
)
FIXED_CONTINUATION_PROTOCOL = (
    "held_in_pooled_14core_relative_qkv_fixed_continuation_epoch300"
)
WORLD_SIZE = 4
VISIBLE_DEVICES = "0,1,2,3"
MODEL_SEED = 0
HELD_IN_MASK_BASE_SEED = 2026082591
STDOUT_PREFIX = "[bagm-so2-training]"
FINAL_RELOAD_PREDICTION_ATOL = 1e-6
FINAL_RELOAD_PREDICTION_RTOL = 1e-6
FINAL_RELOAD_RECEIVER_COUNT = 8
FIXED_CONTINUATION_SOURCE_EPOCH = 175
FIXED_CONTINUATION_FINAL_EPOCH = 300
FIXED_CONTINUATION_SOURCE_RUN_ID = (
    "r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6"
)
FIXED_CONTINUATION_SOURCE_CHECKPOINT_SHA256 = (
    "2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d"
)
FIXED_CONTINUATION_PREFLIGHT = (
    "state/preflight/so2_14core_relative_qkv_ddp4_resume175_fixed300.json"
)
BASELINE_PREFLIGHT = "state/preflight/so2_14core_relative_qkv_ddp4.json"
RECURRENT_PREFLIGHT = (
    "state/preflight/so2_14core_recurrent_relative_qkv_ddp4.json"
)
UNTIED8_PREFLIGHT = (
    "state/preflight/so2_14core_untied8_relative_qkv_ddp4.json"
)
GEOMETRY_MODULATED_PREFLIGHT = (
    "state/preflight/so2_14core_geometry_modulated_relative_qkv_ddp4.json"
)
BASELINE_PREFLIGHT_SCHEMA = "so2_14core_relative_qkv_ddp4_preflight_v1"
RECURRENT_PREFLIGHT_SCHEMA = (
    "so2_14core_recurrent_relative_qkv_ddp4_preflight_v1"
)
UNTIED8_PREFLIGHT_SCHEMA = (
    "so2_14core_untied8_relative_qkv_ddp4_preflight_v1"
)
GEOMETRY_MODULATED_PREFLIGHT_SCHEMA = (
    "so2_14core_geometry_modulated_relative_qkv_ddp4_preflight_v1"
)
EXPECTED_RECURRENT_PARAMETER_COUNT = 2_605_680
EXPECTED_UNTIED8_PARAMETER_COUNT = 8_199_464
EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT = 5_134_088
UNTIED8_PREFLIGHT_MAX_VRAM_GIB = 22.0
UNTIED8_MINIMUM_VRAM_HEADROOM_GIB = 2.0
GEOMETRY_MODULATED_PREFLIGHT_MAX_VRAM_GIB = 22.0
GEOMETRY_MODULATED_MINIMUM_VRAM_HEADROOM_GIB = 2.0
UNTIED8_OPERATIONAL_REVIEW_EPOCH = 300


class SO214CoreRunnerError(RuntimeError):
    """Raised before a run can violate the four-rank production contract."""


class SO214CoreOperationalReviewRequired(SO214CoreRunnerError):
    """Raised after a durable untied-8 epoch-300 boundary needs human review."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise SO214CoreRunnerError(f"Resolved config requires {name!r} mapping.")
    return value


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise SO214CoreRunnerError(
            f"{field} must be {expected!r}; received {actual!r}."
        )


def _is_fixed_continuation(config: Mapping[str, Any]) -> bool:
    evaluation = config.get("evaluation")
    trainer = config.get("trainer")
    return (
        isinstance(evaluation, Mapping)
        and isinstance(trainer, Mapping)
        and evaluation.get("protocol") == FIXED_CONTINUATION_PROTOCOL
        and trainer.get("execution_mode") == "resume_fixed_final_epoch"
    )


def _protocol(config: Mapping[str, Any]) -> str:
    return str(_section(config, "evaluation").get("protocol", ""))


def _is_recurrent(config: Mapping[str, Any]) -> bool:
    return _protocol(config) == RECURRENT_PLATEAU_PROTOCOL


def _is_untied8(config: Mapping[str, Any]) -> bool:
    return _protocol(config) == UNTIED8_PLATEAU_PROTOCOL


def _is_geometry_modulated(config: Mapping[str, Any]) -> bool:
    return _protocol(config) == GEOMETRY_MODULATED_PLATEAU_PROTOCOL


def _block_gradient_count(config: Mapping[str, Any]) -> int:
    if _is_untied8(config):
        return 8
    if _is_geometry_modulated(config):
        return 4
    return 0


def _block_trainable_parameter_count(config: Mapping[str, Any]) -> int | None:
    if _is_untied8(config):
        return 799_112
    if _is_geometry_modulated(config):
        return 831_880
    return None


def _block_vram_limits(config: Mapping[str, Any]) -> tuple[float, float] | None:
    if _is_untied8(config):
        return (
            UNTIED8_PREFLIGHT_MAX_VRAM_GIB,
            UNTIED8_MINIMUM_VRAM_HEADROOM_GIB,
        )
    if _is_geometry_modulated(config):
        return (
            GEOMETRY_MODULATED_PREFLIGHT_MAX_VRAM_GIB,
            GEOMETRY_MODULATED_MINIMUM_VRAM_HEADROOM_GIB,
        )
    return None


def _gradient_direction_enabled(config: Mapping[str, Any]) -> bool:
    return _is_recurrent(config) or _block_gradient_count(config) > 0


def _expected_parameter_count(config: Mapping[str, Any]) -> int | None:
    if _is_recurrent(config):
        return EXPECTED_RECURRENT_PARAMETER_COUNT
    if _is_untied8(config):
        return EXPECTED_UNTIED8_PARAMETER_COUNT
    if _is_geometry_modulated(config):
        return EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT
    return None


def _campaign_id(config: Mapping[str, Any]) -> str:
    value = _section(config, "campaign").get("campaign_id")
    if not isinstance(value, str) or not value.strip():
        raise SO214CoreRunnerError(
            "Resolved config requires a non-empty campaign.campaign_id."
        )
    return value.strip()


def _expected_campaign_id(config: Mapping[str, Any]) -> str:
    protocol = _protocol(config)
    if protocol == RECURRENT_PLATEAU_PROTOCOL:
        return RECURRENT_CAMPAIGN_ID
    if protocol == UNTIED8_PLATEAU_PROTOCOL:
        return UNTIED8_CAMPAIGN_ID
    if protocol == GEOMETRY_MODULATED_PLATEAU_PROTOCOL:
        return GEOMETRY_MODULATED_CAMPAIGN_ID
    if protocol in {PLATEAU_PROTOCOL, FIXED_CONTINUATION_PROTOCOL}:
        return CAMPAIGN_ID
    raise SO214CoreRunnerError(
        "evaluation.protocol is not an SO2 Relative-QKV training protocol."
    )


def _preflight_schema(config: Mapping[str, Any]) -> str:
    if _is_recurrent(config):
        return RECURRENT_PREFLIGHT_SCHEMA
    if _is_untied8(config):
        return UNTIED8_PREFLIGHT_SCHEMA
    if _is_geometry_modulated(config):
        return GEOMETRY_MODULATED_PREFLIGHT_SCHEMA
    return BASELINE_PREFLIGHT_SCHEMA


def _preflight_path(config: Mapping[str, Any]) -> str:
    protocol = _protocol(config)
    if protocol == RECURRENT_PLATEAU_PROTOCOL:
        return RECURRENT_PREFLIGHT
    if protocol == UNTIED8_PLATEAU_PROTOCOL:
        return UNTIED8_PREFLIGHT
    if protocol == GEOMETRY_MODULATED_PLATEAU_PROTOCOL:
        return GEOMETRY_MODULATED_PREFLIGHT
    if protocol == FIXED_CONTINUATION_PROTOCOL:
        return FIXED_CONTINUATION_PREFLIGHT
    if protocol == PLATEAU_PROTOCOL:
        return BASELINE_PREFLIGHT
    raise SO214CoreRunnerError(
        "evaluation.protocol is not an SO2 Relative-QKV training protocol."
    )


def _runtime_path(raw: object, paths: ProjectPaths) -> Path:
    """Resolve absolute and repository-root-relative config paths with overrides."""

    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if not candidate.parts or ".." in candidate.parts:
        raise SO214CoreRunnerError(f"Unsafe configured runtime path: {raw!r}.")
    roots = {
        "data": paths.data_root,
        "artifacts": paths.artifact_root,
        "state": paths.state_root,
        "scratch": paths.scratch_root,
        "reports": paths.report_root,
        "configs": paths.config_root,
    }
    if candidate.parts[0] in roots:
        return roots[candidate.parts[0]].joinpath(*candidate.parts[1:]).resolve(
            strict=False
        )
    return (paths.project_root / candidate).resolve(strict=False)


def _validate_contract(config: Mapping[str, Any]) -> None:
    validate_experiment_config(config)
    _require_equal(config.get("seed"), MODEL_SEED, field="seed")
    _require_equal(
        _campaign_id(config),
        _expected_campaign_id(config),
        field="campaign.campaign_id",
    )
    protocol = _protocol(config)
    if protocol not in {
        PLATEAU_PROTOCOL,
        RECURRENT_PLATEAU_PROTOCOL,
        UNTIED8_PLATEAU_PROTOCOL,
        GEOMETRY_MODULATED_PLATEAU_PROTOCOL,
        FIXED_CONTINUATION_PROTOCOL,
    }:
        raise SO214CoreRunnerError(
            "evaluation.protocol is not an SO2 Relative-QKV training protocol."
        )
    dataset = _section(config, "dataset")
    _require_equal(tuple(dataset.get("core_aliases", ())), SO2_ALIASES, field="dataset.core_aliases")
    _require_equal(
        dataset.get("total_fit_cells"),
        EXPECTED_TOTAL_CELLS,
        field="dataset.total_fit_cells",
    )
    model = _section(config, "model")
    locked_model = {
        "hidden_dim": 256,
        "attention_heads": 8,
        "attention_head_dim": 32,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "relative_geometry_dim": 70,
        "attention_dropout": 0.0,
        "activation_checkpointing": True,
        "uses_edge_inputs": False,
    }
    if protocol == GEOMETRY_MODULATED_PLATEAU_PROTOCOL:
        locked_model.update(
            {
                "name": "geometry-modulated-relative-qkv-gat",
                "family": "geometry_modulated_relative_qkv_graph_transformer",
                "embedding_dim": 256,
                "graph_layers": 4,
                "unique_graph_blocks": 4,
                "effective_graph_depth": 4,
                "graph_block_weight_tying": "none",
                "geometry_hidden_dim": 128,
                "attention_score_mechanism": (
                    "geometry_modulated_cosine_qkv_v1"
                ),
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
                "relative_geometry_role": (
                    "attention_logit_modulation_and_bias_only"
                ),
                "relative_geometry_value_injection": False,
                "value_content_source": (
                    "expression_derived_node_embedding_only"
                ),
                "receiver_chunk_size": 128,
                "max_edges_per_chunk": 50000,
                "exact_receiver_partitioning": True,
                "fp32_attention_scoring": True,
                "fp32_attention_accumulation": True,
                "implicit_self_loops": False,
                "trainable_node_identifiers": False,
                "trainable_edge_identifiers": False,
                "uses_graph_inputs": True,
                "uses_relative_position": True,
                "edge_key_vectors": False,
                "edge_value_vectors": False,
                "edge_value_gates": False,
            }
        )
    elif protocol == RECURRENT_PLATEAU_PROTOCOL:
        locked_model.update(
            {
                "name": "recurrent-relative-qkv-gat",
                "family": "recurrent_relative_geometry_qkv_graph_transformer",
                "graph_layers": 1,
                "unique_graph_blocks": 1,
                "recurrent_unroll_steps": 4,
                "effective_graph_depth": 4,
                "graph_block_weight_tying": "all_steps",
            }
        )
    elif protocol == UNTIED8_PLATEAU_PROTOCOL:
        locked_model.update(
            {
                "name": "relative-qkv-gat",
                "family": "relative_geometry_qkv_graph_transformer",
                "graph_layers": 8,
                "unique_graph_blocks": 8,
                "effective_graph_depth": 8,
                "graph_block_weight_tying": "none",
            }
        )
    else:
        locked_model.update(
            {
                "name": "relative-qkv-gat",
                "family": "relative_geometry_qkv_graph_transformer",
                "graph_layers": 4,
            }
        )
    for field, expected in locked_model.items():
        _require_equal(model.get(field), expected, field=f"model.{field}")
    trainer = _section(config, "trainer")
    locked_trainer = {
        "batch_size": 2,
        "core_visits_per_global_epoch": 14,
        "cores_per_optimizer_update": 2,
        "optimizer_updates_per_global_epoch": 7,
        "mask_views_per_core_step": 10,
        "mask_views_per_rank_per_optimizer_update": 5,
        "distributed_backend": "nccl",
        "distributed_world_size": WORLD_SIZE,
        "rank_zero_only_artifact_writes": True,
        "early_stopping": False,
        "restore_best": False,
    }
    if protocol in {
        PLATEAU_PROTOCOL,
        RECURRENT_PLATEAU_PROTOCOL,
        UNTIED8_PLATEAU_PROTOCOL,
        GEOMETRY_MODULATED_PLATEAU_PROTOCOL,
    }:
        locked_trainer.update(
            {
                "minimum_global_epochs": 150,
                "continuation_block_global_epochs": 25,
                "checkpoint_policy": "atomic_latest_then_final_last_only",
                "checkpoint_every_global_epochs": 1,
            }
        )
    else:
        locked_trainer.update(
            {
                "execution_mode": "resume_fixed_final_epoch",
                "required_resume_completed_global_epochs": (
                    FIXED_CONTINUATION_SOURCE_EPOCH
                ),
                "required_source_run_id": FIXED_CONTINUATION_SOURCE_RUN_ID,
                "required_source_checkpoint_sha256": (
                    FIXED_CONTINUATION_SOURCE_CHECKPOINT_SHA256
                ),
                "fixed_final_global_epoch": FIXED_CONTINUATION_FINAL_EPOCH,
                "max_epochs": FIXED_CONTINUATION_FINAL_EPOCH,
                "minimum_global_epochs": FIXED_CONTINUATION_FINAL_EPOCH,
                "fixed_epoch_budget": True,
                "continuation_policy": (
                    "fixed_epoch_300_from_confirmed_epoch_175"
                ),
                "plateau_extension": False,
                "plateau_stopping_enabled": False,
                "maximum_scientific_epoch_cap": FIXED_CONTINUATION_FINAL_EPOCH,
                "checkpoint_policy": "final_last_only_no_intermediate",
                "checkpoint_every_global_epochs": None,
                "checkpoint_final_role": "final_epoch_300_last",
                "strict_plateau_diagnostic_only": True,
                "strict_plateau_audit_interval_global_epochs": 25,
                "strict_plateau_window_global_epochs": 50,
                "strict_plateau_absolute_relative_half_window_change_max": 0.0005,
                "strict_plateau_normalized_absolute_slope_per_epoch_max": 0.000025,
                "strict_plateau_consecutive_passing_audits": 2,
            }
        )
    for field, expected in locked_trainer.items():
        _require_equal(trainer.get(field), expected, field=f"trainer.{field}")
    launcher = _section(config, "launcher")
    locked_launcher = {
        "requested_gpu": VISIBLE_DEVICES,
        "requested_gpu_count": WORLD_SIZE,
        "require_exact_visible_devices": VISIBLE_DEVICES,
        "process_count": WORLD_SIZE,
        "elastic_max_restarts": 0,
        "hardware_preflight_receipt": _preflight_path(config),
    }
    for field, expected in locked_launcher.items():
        _require_equal(launcher.get(field), expected, field=f"launcher.{field}")
    block_gradient_count = _block_gradient_count(config)
    if block_gradient_count:
        metadata = _section(config, "metadata")
        gradient_diagnostics = metadata.get("gradient_diagnostics")
        if not isinstance(gradient_diagnostics, Mapping):
            raise SO214CoreRunnerError(
                "Block-observed training requires metadata.gradient_diagnostics."
            )
        for field, expected in {
            "enabled": True,
            "output_path": "results/gradient_direction_metrics.csv",
            "schema": GRADIENT_DIRECTION_METRICS_SCHEMA,
            "ordered_columns": list(GRADIENT_DIRECTION_METRICS_COLUMNS),
            "persist_gradient_tensors": False,
            "persist_per_optimizer_step_files": False,
            "persist_gradient_vectors_in_checkpoints": False,
            "affects_optimization_or_plateau_stopping": False,
        }.items():
            _require_equal(
                gradient_diagnostics.get(field),
                expected,
                field=f"metadata.gradient_diagnostics.{field}",
            )
        layerwise = gradient_diagnostics.get("layerwise")
        if not isinstance(layerwise, Mapping):
            raise SO214CoreRunnerError(
                "Block-observed training requires graph-block gradient diagnostics."
            )
        for field, expected in {
            "enabled": True,
            "scope": "graph_blocks_only",
            "block_names": [
                f"blocks.{index}" for index in range(block_gradient_count)
            ],
            "output_path": "results/gradient_direction_by_block.csv",
            "schema": BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
            "ordered_columns": list(BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS),
            "rows_per_completed_global_epoch": block_gradient_count,
            "persist_gradient_tensors": False,
            "persist_per_optimizer_step_files": False,
            "persist_gradient_vectors_in_checkpoints": False,
            "affects_optimization_or_plateau_stopping": False,
        }.items():
            _require_equal(
                layerwise.get(field),
                expected,
                field=f"metadata.gradient_diagnostics.layerwise.{field}",
            )
    if protocol == UNTIED8_PLATEAU_PROTOCOL:
        metadata = _section(config, "metadata")
        operational_review = metadata.get("operational_review")
        if not isinstance(operational_review, Mapping):
            raise SO214CoreRunnerError(
                "Untied-8 requires its epoch-300 operational review gate."
            )
        for field, expected in {
            "enabled": True,
            "trigger_completed_global_epoch": UNTIED8_OPERATIONAL_REVIEW_EPOCH,
            "trigger_only_if_plateau_unconfirmed": True,
            "action": (
                "deliberate_non_success_exit_after_durable_epoch300_latest_"
                "checkpoint"
            ),
            "queue_bundle_outcome": "failed_inconclusive",
            "resumable_only_after_explicit_campaign_amendment": True,
            "automatic_retry_allowed": False,
            "scientific_convergence_criterion": False,
            "scientific_maximum_epoch_cap": None,
            "modifies_literal_plateau_rule": False,
        }.items():
            _require_equal(
                operational_review.get(field),
                expected,
                field=f"metadata.operational_review.{field}",
            )
    if block_gradient_count:
        metadata = _section(config, "metadata")
        preflight_acceptance = metadata.get("preflight_acceptance")
        if not isinstance(preflight_acceptance, Mapping):
            raise SO214CoreRunnerError(
                "Block-observed training requires an explicit preflight VRAM "
                "acceptance gate."
            )
        for field, expected in {
            "peak_vram_gib_all_ranks_max": 22.0,
            "minimum_vram_headroom_gib_each_rank": 2.0,
            "gpu_memory_gib_each_rank": 24.0,
        }.items():
            _require_equal(
                preflight_acceptance.get(field),
                expected,
                field=f"metadata.preflight_acceptance.{field}",
            )
    if protocol == FIXED_CONTINUATION_PROTOCOL and not str(
        launcher.get("resume_checkpoint", "")
    ).endswith(
        "/r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6/"
        "checkpoints/last.ckpt"
    ):
        raise SO214CoreRunnerError(
            "Fixed continuation requires the locked epoch-175 last checkpoint."
        )
    if protocol == FIXED_CONTINUATION_PROTOCOL and not str(
        launcher.get("source_artifact_path", "")
    ).rstrip("/").endswith(
        "/r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6"
    ):
        raise SO214CoreRunnerError(
            "Fixed continuation requires the locked source artifact bundle."
        )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _preflight_bound_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove retry routing while binding every scientific/execution setting."""

    normalized = dict(config)
    normalized["attempt"] = 1
    launcher = normalized.get("launcher")
    if isinstance(launcher, Mapping):
        normalized_launcher = dict(launcher)
        normalized_launcher.pop("resume_checkpoint", None)
        normalized["launcher"] = normalized_launcher
    return normalized


def _validate_hardware_preflight(
    config: Mapping[str, Any],
    *,
    paths: ProjectPaths,
    cohort_dir: Path,
    graph_dir: Path,
) -> dict[str, Any]:
    receipt_path = _runtime_path(
        _section(config, "launcher")["hardware_preflight_receipt"], paths
    )
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SO214CoreRunnerError(
            f"A passing four-rank hardware preflight is required: {receipt_path}."
        ) from exc
    if not isinstance(value, dict):
        raise SO214CoreRunnerError("Hardware preflight receipt must be a mapping.")
    expected_checksum = value.get("receipt_content_sha256")
    content = dict(value)
    content.pop("receipt_content_sha256", None)
    if expected_checksum != _canonical_sha256(content):
        raise SO214CoreRunnerError("Hardware preflight receipt checksum mismatch.")
    required = {
        "schema": _preflight_schema(config),
        "status": "passed",
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "distributed_world_size": WORLD_SIZE,
        "distributed_backend": "nccl",
        "visible_devices": VISIBLE_DEVICES,
        "elastic_max_restarts": 0,
        "optimizer_updates": 1,
        "complete_graph_mask_views": 20,
        "checkpoint_reload_verified": True,
        "finite_loss_and_gradients": True,
        "so2_c23_prior_equivalence_verified": True,
    }
    if _gradient_direction_enabled(config):
        expected_parameter_count = _expected_parameter_count(config)
        assert expected_parameter_count is not None
        required.update(
            {
                "campaign_id": _campaign_id(config),
                "parameter_count": expected_parameter_count,
                "gradient_direction_observer_verified": True,
            }
        )
        block_count = _block_gradient_count(config)
        if block_count:
            limits = _block_vram_limits(config)
            assert limits is not None
            maximum_vram_gib, minimum_headroom_gib = limits
            required.update(
                {
                    "all_unique_graph_blocks_receive_gradients": True,
                    "block_gradient_direction_observer_verified": True,
                    "peak_vram_gib_all_ranks_max": maximum_vram_gib,
                    "minimum_vram_headroom_gib_each_rank": minimum_headroom_gib,
                    "vram_acceptance_passed": True,
                }
            )
    for field, expected in required.items():
        _require_equal(value.get(field), expected, field=f"preflight.{field}")
    if _is_geometry_modulated(config):
        numerical_gates = value.get("geometry_modulated_numerical_gates")
        if not isinstance(numerical_gates, Mapping):
            raise SO214CoreRunnerError(
                "Geometry-modulated preflight lacks its numerical gates."
            )
        _require_equal(
            numerical_gates.get("passed"),
            True,
            field="preflight.geometry_modulated_numerical_gates.passed",
        )
        for gate_name in (
            "full_chunk_exactness",
            "amp_fp32_equivalence",
            "learned_geometry_heads",
        ):
            gate = numerical_gates.get(gate_name)
            if not isinstance(gate, Mapping):
                raise SO214CoreRunnerError(
                    "Geometry-modulated preflight lacks numerical gate "
                    f"{gate_name!r}."
                )
            _require_equal(
                gate.get("passed"),
                True,
                field=(
                    "preflight.geometry_modulated_numerical_gates."
                    f"{gate_name}.passed"
                ),
            )
    if value.get("cohort_manifest_sha256") != sha256_file(
        cohort_dir / "manifest.json"
    ):
        raise SO214CoreRunnerError("Preflight cohort manifest has changed.")
    if value.get("graph_manifest_sha256") != sha256_file(
        graph_dir / "manifest.json"
    ):
        raise SO214CoreRunnerError("Preflight graph manifest has changed.")
    if value.get("resolved_config_sha256") != _canonical_sha256(
        _preflight_bound_config(config)
    ):
        raise SO214CoreRunnerError("Resolved production config changed after preflight.")
    if _gradient_direction_enabled(config):
        expected_parameter_count = _expected_parameter_count(config)
        assert expected_parameter_count is not None
        receipt_model = value.get("model")
        if not isinstance(receipt_model, Mapping):
            raise SO214CoreRunnerError(
                "Gradient-observed preflight lacks its exact model construction."
            )
        expected_model = {
            "class": (
                "ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer"
                if _is_recurrent(config)
                else (
                    "ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer"
                    if _is_geometry_modulated(config)
                    else "ReceiverChunkedRelativeGeometryQKVGraphTransformer"
                )
            ),
            "num_genes": 1000,
            "node_covariate_dim": 22,
            **dict(_section(config, "model")),
        }
        for field, expected in expected_model.items():
            _require_equal(
                receipt_model.get(field),
                expected,
                field=f"preflight.model.{field}",
            )
        gradient_summary = value.get("gradient_direction_preflight_summary")
        if not isinstance(gradient_summary, Mapping):
            raise SO214CoreRunnerError(
                "Gradient-observed preflight lacks its scalar gradient summary."
            )
        for field, expected in {
            "schema": GRADIENT_DIRECTION_METRICS_SCHEMA,
            "global_epoch": 1,
            "trainable_parameter_count": expected_parameter_count,
            "optimizer_updates_observed": 1,
            "consecutive_optimizer_step_cosine_valid_pairs": 0,
            "epoch_aggregate_gradient_cosine_to_previous_epoch": None,
            "resume_boundary_unavailable": False,
        }.items():
            _require_equal(
                gradient_summary.get(field),
                expected,
                field=f"preflight.gradient_direction.{field}",
            )
        observed_gradient_norm = gradient_summary.get(
            "gradient_norm_mean_before_clip"
        )
        trainer_gradient_norm = value.get("gradient_norm_before_clip")
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in (observed_gradient_norm, trainer_gradient_norm)
        ) or not math.isclose(
            float(observed_gradient_norm),
            float(trainer_gradient_norm),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise SO214CoreRunnerError(
                "Preflight gradient observer norm did not match the "
                "trainer's pre-clip norm."
            )
        block_count = _block_gradient_count(config)
        if block_count:
            block_parameter_count = _block_trainable_parameter_count(config)
            assert block_parameter_count is not None
            topology_field = (
                "geometry_modulated_graph_block_topology"
                if _is_geometry_modulated(config)
                else "untied_graph_block_topology"
            )
            topology = value.get(topology_field)
            expected_topology = {
                "verified": True,
                "graph_block_count": block_count,
                "unique_graph_block_objects": block_count,
                "unique_graph_block_parameter_sets": block_count,
                "state_dict_block_indices": list(range(block_count)),
                "graph_block_weight_tying": "none",
            }
            _require_equal(
                topology,
                expected_topology,
                field=f"preflight.{topology_field}",
            )
            block_gradients = value.get("graph_block_gradient_diagnostics")
            if not isinstance(block_gradients, Sequence) or isinstance(
                block_gradients, (str, bytes)
            ):
                raise SO214CoreRunnerError(
                    "Block-observed preflight lacks per-block gradient diagnostics."
                )
            if len(block_gradients) != block_count:
                raise SO214CoreRunnerError(
                    "Block-observed preflight reported an invalid block-gradient "
                    "count."
                )
            for index, record in enumerate(block_gradients):
                if not isinstance(record, Mapping):
                    raise SO214CoreRunnerError(
                        "Block gradient diagnostics must be mappings."
                    )
                for field, expected in {
                    "block_index": index,
                    "block_name": f"blocks.{index}",
                    "trainable_parameter_count": block_parameter_count,
                    "parameters_missing_gradient": 0,
                }.items():
                    _require_equal(
                        record.get(field),
                        expected,
                        field=f"preflight.block_gradient[{index}].{field}",
                    )
                norm = record.get("gradient_norm_before_clip")
                if (
                    isinstance(norm, bool)
                    or not isinstance(norm, (int, float))
                    or not math.isfinite(float(norm))
                    or float(norm) <= 0.0
                ):
                    raise SO214CoreRunnerError(
                        "Every observed graph block must receive a finite, "
                        "non-zero pre-clip gradient."
                    )
            block_summaries = value.get(
                "block_gradient_direction_preflight_summary"
            )
            if not isinstance(block_summaries, Sequence) or isinstance(
                block_summaries, (str, bytes)
            ) or len(block_summaries) != block_count:
                raise SO214CoreRunnerError(
                    "Block-observed preflight has an invalid number of "
                    "block-direction summaries."
                )
            for index, summary in enumerate(block_summaries):
                if not isinstance(summary, Mapping):
                    raise SO214CoreRunnerError(
                        "Block-direction summaries must be mappings."
                    )
                for field, expected in {
                    "schema": BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
                    "global_epoch": 1,
                    "block_index": index,
                    "block_name": f"blocks.{index}",
                    "trainable_parameter_count": block_parameter_count,
                    "optimizer_updates_observed": 1,
                    "consecutive_optimizer_step_cosine_valid_pairs": 0,
                    "epoch_aggregate_gradient_cosine_to_previous_epoch": None,
                    "resume_boundary_unavailable": False,
                }.items():
                    _require_equal(
                        summary.get(field),
                        expected,
                        field=f"preflight.block_direction[{index}].{field}",
                    )
                norm = summary.get("gradient_norm_mean_before_clip")
                if (
                    isinstance(norm, bool)
                    or not isinstance(norm, (int, float))
                    or not math.isfinite(float(norm))
                    or float(norm) <= 0.0
                ):
                    raise SO214CoreRunnerError(
                        "Block direction norm must be finite and positive."
                    )
    peak = value.get("peak_vram_gib_all_ranks")
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(
        float(peak)
    ) or float(peak) <= 0:
        raise SO214CoreRunnerError("Preflight peak VRAM measurement is invalid.")
    limits = _block_vram_limits(config)
    if limits is not None and float(peak) > limits[0]:
        raise SO214CoreRunnerError(
            "Architecture preflight exceeds the 22.0 GiB peak VRAM "
            "acceptance gate."
        )
    if limits is not None:
        _, minimum_headroom_gib = limits
        headroom = value.get("measured_vram_headroom_gib_each_rank")
        if (
            isinstance(headroom, bool)
            or not isinstance(headroom, (int, float))
            or not math.isfinite(float(headroom))
            or float(headroom) < minimum_headroom_gib
            or not math.isclose(
                float(headroom),
                24.0 - float(peak),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise SO214CoreRunnerError(
                "Architecture preflight VRAM headroom receipt is invalid."
            )
    return value


def _distributed_identity() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO214CoreRunnerError(
            "This runner must be invoked by torch.distributed.run."
        ) from exc
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE) or local_rank not in range(
        WORLD_SIZE
    ):
        raise SO214CoreRunnerError("Torchrun rank/world-size contract is invalid.")
    if rank != local_rank:
        raise SO214CoreRunnerError(
            "Single-node production requires global rank to equal LOCAL_RANK."
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != VISIBLE_DEVICES:
        raise SO214CoreRunnerError(
            f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE_DEVICES}."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise SO214CoreRunnerError(
            "Four logical CUDA devices are required after visibility filtering."
        )
    return rank, local_rank, world_size


def _worker_inputs(
    args: argparse.Namespace,
) -> tuple[str, Path, dict[str, Any], ProjectPaths]:
    run_id = os.environ.get("BAGM_RUN_ID", "").strip()
    environment_scratch = os.environ.get("BAGM_RUN_SCRATCH", "").strip()
    if not run_id or not environment_scratch:
        raise SO214CoreRunnerError(
            "BAGM_RUN_ID and BAGM_RUN_SCRATCH are required from the queue worker."
        )
    supplied_scratch = args.run_scratch.resolve(strict=False)
    if supplied_scratch != Path(environment_scratch).resolve(strict=False):
        raise SO214CoreRunnerError("--run-scratch does not match BAGM_RUN_SCRATCH.")
    expected_config = supplied_scratch / "config.resolved.yaml"
    if args.config.resolve(strict=False) != expected_config.resolve(strict=False):
        raise SO214CoreRunnerError(
            "--config must be the worker-owned resolved configuration."
        )
    environment_config = os.environ.get("BAGM_CONFIG_PATH", "").strip()
    if environment_config and Path(environment_config).resolve(
        strict=False
    ) != expected_config.resolve(strict=False):
        raise SO214CoreRunnerError("BAGM_CONFIG_PATH does not match --config.")
    paths = current_paths()
    if supplied_scratch != (paths.scratch_root / "active_runs" / run_id).resolve(
        strict=False
    ):
        raise SO214CoreRunnerError("Scratch path is not canonical for this run ID.")
    config = load_yaml_mapping(expected_config)
    _validate_contract(config)
    return run_id, supplied_scratch, config, paths


def _model_from_config(
    config: Mapping[str, Any],
    *,
    num_genes: int,
    node_covariate_dim: int,
) -> torch.nn.Module:
    model = _section(config, "model")
    trainer = _section(config, "trainer")
    set_deterministic_seed(
        MODEL_SEED,
        deterministic=bool(trainer["deterministic"]),
        warn_only=bool(trainer["deterministic_warn_only"]),
    )
    common = {
        "num_genes": int(num_genes),
        "node_covariate_dim": int(node_covariate_dim),
        "hidden_dim": int(model["hidden_dim"]),
        "attention_heads": int(model["attention_heads"]),
        "attention_head_dim": int(model["attention_head_dim"]),
        "ffn_dim": int(model["ffn_dim"]),
        "decoder_dim": int(model["decoder_dim"]),
        "dropout": float(model["dropout"]),
        "attention_dropout": float(model["attention_dropout"]),
        "relative_geometry_dim": int(model["relative_geometry_dim"]),
        "receiver_chunk_size": int(model["receiver_chunk_size"]),
        "max_edges_per_chunk": int(model["max_edges_per_chunk"]),
        "activation_checkpointing": bool(model["activation_checkpointing"]),
    }
    if _is_geometry_modulated(config):
        built = ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer(
            **common,
            graph_layers=int(model["graph_layers"]),
            geometry_hidden_dim=int(model["geometry_hidden_dim"]),
            qk_normalization_epsilon=float(
                model["qk_normalization_epsilon"]
            ),
            logit_scale_initial=float(model["logit_scale_initial"]),
            logit_scale_minimum=float(model["logit_scale_minimum"]),
            logit_scale_maximum=float(model["logit_scale_maximum"]),
            modulation_amplitude=float(model["modulation_amplitude"]),
            geometry_bias_bound=float(model["geometry_bias_bound"]),
        )
        _geometry_modulated4_block_topology(built)
        for field in (
            "unique_graph_blocks",
            "effective_graph_depth",
            "graph_block_weight_tying",
            "attention_score_mechanism",
            "geometry_hidden_dim",
            "qk_normalization_epsilon",
            "logit_scale_initial",
            "logit_scale_minimum",
            "logit_scale_maximum",
            "modulation_amplitude",
            "geometry_bias_bound",
        ):
            _require_equal(
                getattr(built, field, None),
                model[field],
                field=f"constructed_model.{field}",
            )
        _require_equal(
            getattr(built, "qk_l2_normalized", None),
            True,
            field="constructed_model.qk_l2_normalized",
        )
        _require_equal(
            getattr(built, "geometry_values_enabled", None),
            False,
            field="constructed_model.geometry_values_enabled",
        )
        if int(num_genes) == 1000 and int(node_covariate_dim) == 22:
            _require_equal(
                sum(parameter.numel() for parameter in built.parameters()),
                EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT,
                field="constructed_model.parameter_count",
            )
        return built
    if _is_recurrent(config):
        built: torch.nn.Module = (
            ReceiverChunkedRecurrentRelativeGeometryQKVGraphTransformer(
                **common,
                positional_bias_hidden_dim=int(
                    model["positional_bias_hidden_dim"]
                ),
                recurrent_unroll_steps=int(model["recurrent_unroll_steps"]),
            )
        )
        for field, expected in {
            "graph_layers": 1,
            "unique_graph_blocks": 1,
            "recurrent_unroll_steps": 4,
            "effective_graph_depth": 4,
            "graph_block_weight_tying": "all_steps",
        }.items():
            _require_equal(
                getattr(built, field, None),
                expected,
                field=f"constructed_model.{field}",
            )
        block_keys = tuple(
            name for name in built.state_dict() if name.startswith("blocks.")
        )
        if not block_keys or any(
            not name.startswith("blocks.0.") for name in block_keys
        ):
            raise SO214CoreRunnerError(
                "Recurrent model must register exactly one graph block at blocks.0."
            )
        if int(num_genes) == 1000 and int(node_covariate_dim) == 22:
            _require_equal(
                sum(parameter.numel() for parameter in built.parameters()),
                EXPECTED_RECURRENT_PARAMETER_COUNT,
                field="constructed_model.parameter_count",
            )
        return built
    built = ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        **common,
        positional_bias_hidden_dim=int(model["positional_bias_hidden_dim"]),
        graph_layers=int(model["graph_layers"]),
    )
    if _is_untied8(config):
        _untied8_block_topology(built)
        if int(num_genes) == 1000 and int(node_covariate_dim) == 22:
            _require_equal(
                sum(parameter.numel() for parameter in built.parameters()),
                EXPECTED_UNTIED8_PARAMETER_COUNT,
                field="constructed_model.parameter_count",
            )
    return built


def _untied8_block_topology(model: torch.nn.Module) -> dict[str, Any]:
    """Verify eight independently parameterized graph blocks without aliases."""

    if type(model) is not ReceiverChunkedRelativeGeometryQKVGraphTransformer:
        raise SO214CoreRunnerError(
            "Untied-8 requires the non-recurrent Relative-QKV model class."
        )
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != 8:
        raise SO214CoreRunnerError(
            "Untied-8 must register exactly eight graph blocks."
        )
    _require_equal(
        getattr(model, "graph_layers", None),
        8,
        field="constructed_model.graph_layers",
    )
    object_ids = tuple(id(block) for block in blocks)
    if len(set(object_ids)) != 8:
        raise SO214CoreRunnerError("Untied-8 graph block modules may not be shared.")
    parameter_ids = tuple(
        frozenset(id(parameter) for parameter in block.parameters())
        for block in blocks
    )
    if any(not identifiers for identifiers in parameter_ids):
        raise SO214CoreRunnerError("Every untied graph block must own parameters.")
    for left in range(len(parameter_ids)):
        for right in range(left + 1, len(parameter_ids)):
            if parameter_ids[left].intersection(parameter_ids[right]):
                raise SO214CoreRunnerError(
                    "Untied-8 graph block parameters may not be shared."
                )
    state_indices: set[int] = set()
    for name in model.state_dict():
        if not name.startswith("blocks."):
            continue
        try:
            state_indices.add(int(name.split(".", 2)[1]))
        except (IndexError, ValueError) as exc:
            raise SO214CoreRunnerError(
                "Untied-8 graph block state names are malformed."
            ) from exc
    if state_indices != set(range(8)):
        raise SO214CoreRunnerError(
            "Untied-8 state_dict must contain blocks.0 through blocks.7."
        )
    return {
        "verified": True,
        "graph_block_count": 8,
        "unique_graph_block_objects": 8,
        "unique_graph_block_parameter_sets": 8,
        "state_dict_block_indices": list(range(8)),
        "graph_block_weight_tying": "none",
    }


def _geometry_modulated4_block_topology(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Verify four independently parameterized geometry-modulated blocks."""

    if type(model) is not ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer:
        raise SO214CoreRunnerError(
            "Geometry-modulated training requires its receiver-chunked model class."
        )
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != 4:
        raise SO214CoreRunnerError(
            "Geometry-modulated training must register exactly four graph blocks."
        )
    _require_equal(
        getattr(model, "graph_layers", None),
        4,
        field="constructed_model.graph_layers",
    )
    object_ids = tuple(id(block) for block in blocks)
    if len(set(object_ids)) != 4:
        raise SO214CoreRunnerError(
            "Geometry-modulated graph block modules may not be shared."
        )
    parameter_ids = tuple(
        frozenset(id(parameter) for parameter in block.parameters())
        for block in blocks
    )
    if any(not identifiers for identifiers in parameter_ids):
        raise SO214CoreRunnerError(
            "Every geometry-modulated graph block must own parameters."
        )
    for left in range(len(parameter_ids)):
        for right in range(left + 1, len(parameter_ids)):
            if parameter_ids[left].intersection(parameter_ids[right]):
                raise SO214CoreRunnerError(
                    "Geometry-modulated graph block parameters may not be shared."
                )
    state_indices: set[int] = set()
    for name in model.state_dict():
        if not name.startswith("blocks."):
            continue
        try:
            state_indices.add(int(name.split(".", 2)[1]))
        except (IndexError, ValueError) as exc:
            raise SO214CoreRunnerError(
                "Geometry-modulated graph block state names are malformed."
            ) from exc
    if state_indices != set(range(4)):
        raise SO214CoreRunnerError(
            "Geometry-modulated state_dict must contain blocks.0 through blocks.3."
        )
    return {
        "verified": True,
        "graph_block_count": 4,
        "unique_graph_block_objects": 4,
        "unique_graph_block_parameter_sets": 4,
        "state_dict_block_indices": list(range(4)),
        "graph_block_weight_tying": "none",
    }


def _training_config(
    config: Mapping[str, Any],
    *,
    start_epoch: int,
    end_epoch: int,
    rank: int,
    local_rank: int,
) -> CohortRelativeQKVTrainingConfig:
    trainer = _section(config, "trainer")
    masking = _section(config, "masking")
    checkpoint_interval = trainer.get("checkpoint_every_global_epochs")
    return CohortRelativeQKVTrainingConfig(
        model_seed=MODEL_SEED,
        cohort_aliases=SO2_ALIASES,
        segment_start_global_epoch=int(start_epoch),
        segment_end_global_epoch=int(end_epoch),
        cores_per_optimizer_update=int(trainer["cores_per_optimizer_update"]),
        mask_views_per_core=int(trainer["mask_views_per_core_step"]),
        learning_rate=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
        gradient_clip_norm=float(trainer["gradient_clip_norm"]),
        huber_delta=float(trainer["huber_delta"]),
        mask_base_seed=int(masking["mask_base_seed"]),
        core_order_seed=int(trainer["core_order_seed"]),
        amp=bool(trainer["amp"]),
        amp_dtype=str(trainer["amp_dtype"]),
        deterministic=bool(trainer["deterministic"]),
        deterministic_warn_only=bool(trainer["deterministic_warn_only"]),
        device=f"cuda:{local_rank}",
        stage_complete_core_graph_on_device=bool(
            trainer["stage_complete_core_graph_on_device"]
        ),
        staged_relative_geometry_dtype=str(
            trainer["staged_relative_geometry_dtype"]
        ),
        # A positive value remains required by the reusable segment trainer.
        # Fixed continuation passes no checkpoint callback, so this interval
        # is inert and cannot create an intermediate file.
        checkpoint_interval_global_epochs=(
            1 if checkpoint_interval is None else int(checkpoint_interval)
        ),
        distributed_world_size=WORLD_SIZE,
        distributed_rank=int(rank),
    )


def _model_construction(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    *,
    num_genes: int,
    node_covariate_dim: int,
) -> dict[str, Any]:
    return {
        "class": type(model).__name__,
        "num_genes": int(num_genes),
        "node_covariate_dim": int(node_covariate_dim),
        **dict(_section(config, "model")),
    }


def _resume_compatible_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only state-evolution-relevant resume configuration.

    The fixed epoch-300 continuation intentionally changes provenance labels,
    duration, diagnostic, and retention policy.  It may not change anything
    that determines model state evolution: model/data/graph/masks, optimizer,
    AMP, DDP work partitioning, RNG derivation, or core ordering remain bound.
    """

    normalized = _preflight_bound_config(config)
    for section in ("campaign", "experiment", "classification", "evaluation"):
        normalized.pop(section, None)

    trainer = normalized.get("trainer")
    if isinstance(trainer, Mapping):
        normalized_trainer = dict(trainer)
        schedule_fields = {
            "execution_mode",
            "required_resume_completed_global_epochs",
            "required_source_run_id",
            "required_source_checkpoint_sha256",
            "fixed_final_global_epoch",
            "max_epochs",
            "initial_global_epoch_budget",
            "minimum_global_epochs",
            "fixed_epoch_budget",
            "continuation_policy",
            "continuation_block_global_epochs",
            "plateau_extension",
            "plateau_stopping_enabled",
            "maximum_scientific_epoch_cap",
            "checkpoint_policy",
            "checkpoint_every_global_epochs",
            "checkpoint_final_role",
            "strict_plateau_diagnostic_only",
            "strict_plateau_audit_interval_global_epochs",
            "strict_plateau_window_global_epochs",
            "strict_plateau_absolute_relative_half_window_change_max",
            "strict_plateau_normalized_absolute_slope_per_epoch_max",
            "strict_plateau_consecutive_passing_audits",
        }
        for field in schedule_fields:
            normalized_trainer.pop(field, None)
        normalized["trainer"] = normalized_trainer

    launcher = normalized.get("launcher")
    if isinstance(launcher, Mapping):
        normalized_launcher = dict(launcher)
        normalized_launcher.pop("resume_checkpoint", None)
        normalized_launcher.pop("source_artifact_path", None)
        normalized_launcher.pop("hardware_preflight_receipt", None)
        normalized["launcher"] = normalized_launcher
    return normalized


def _checkpoint_payload(
    *,
    run_id: str,
    config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    parameter_count: int,
    resume: CohortRelativeQKVEpochBoundaryResume,
    plateau: Mapping[str, Any] | None = None,
    completion: Mapping[str, Any] | None = None,
    source_plateau: Mapping[str, Any] | None = None,
    strict_plateau_diagnostic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "resume_schema": resume.resume_schema,
        "run_id": str(run_id),
        "campaign_id": _campaign_id(config),
        "model_seed": MODEL_SEED,
        "completed_global_epochs": resume.completed_global_epochs,
        "optimizer_updates_completed": resume.optimizer_updates_completed,
        "mask_base_seed": resume.mask_base_seed,
        "core_order_seed": resume.core_order_seed,
        "cohort_aliases": list(resume.cohort_aliases),
        "cores_per_optimizer_update": resume.cores_per_optimizer_update,
        "optimizer_updates_per_global_epoch": (
            resume.optimizer_updates_per_global_epoch
        ),
        "mask_views_per_core": resume.mask_views_per_core,
        "losses_per_optimizer_update": resume.losses_per_optimizer_update,
        "model_construction": dict(model_construction),
        "parameter_count": int(parameter_count),
        "model_state_dict": resume.model_state_dict,
        "model_state_checksum": resume.model_state_checksum,
        "optimizer_state_dict": resume.optimizer_state_dict,
        "optimizer_state_checksum": resume.optimizer_state_checksum,
        "amp_scaler_state_dict": resume.scaler_state_dict,
        "amp_scaler_state_checksum": resume.scaler_state_checksum,
        "core_history": [asdict(record) for record in resume.core_history],
        "optimizer_update_history": [
            asdict(record) for record in resume.optimizer_update_history
        ],
        "global_history": [asdict(record) for record in resume.global_history],
        "history_checksum": resume.history_checksum,
        "resume_checksum": resume.resume_checksum,
        "model_step_rng_derivation": resume.model_step_rng_derivation,
        "resolved_config": dict(config),
        "distributed_execution": {
            "world_size": WORLD_SIZE,
            "backend": "nccl",
            "visible_devices": VISIBLE_DEVICES,
            "rank_work": [
                "core_a_views_0_4",
                "core_a_views_5_9",
                "core_b_views_0_4",
                "core_b_views_5_9",
            ],
            "elastic_max_restarts": 0,
        },
        "plateau": None if plateau is None else dict(plateau),
        "completion": None if completion is None else dict(completion),
        "source_plateau": (
            None if source_plateau is None else dict(source_plateau)
        ),
        "strict_plateau_diagnostic": (
            None
            if strict_plateau_diagnostic is None
            else dict(strict_plateau_diagnostic)
        ),
    }


def _load_resume_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    require_fixed_source: bool = True,
) -> tuple[CohortRelativeQKVEpochBoundaryResume, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    file_sha256 = sha256_file(resolved)
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SO214CoreRunnerError(f"Cannot reload checkpoint: {resolved}.") from exc
    if not isinstance(value, Mapping):
        raise SO214CoreRunnerError("Resume checkpoint must contain a mapping.")
    payload = dict(value)
    for field, expected in {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "campaign_id": _campaign_id(config),
        "model_seed": MODEL_SEED,
        "cohort_aliases": list(SO2_ALIASES),
        "model_construction": dict(model_construction),
    }.items():
        _require_equal(payload.get(field), expected, field=f"resume.{field}")
    execution = payload.get("distributed_execution")
    if not isinstance(execution, Mapping) or any(
        (
            execution.get("world_size") != WORLD_SIZE,
            execution.get("backend") != "nccl",
            execution.get("visible_devices") != VISIBLE_DEVICES,
            execution.get("elastic_max_restarts") != 0,
        )
    ):
        raise SO214CoreRunnerError("Resume distributed execution contract drifted.")
    source_config = payload.get("resolved_config")
    if not isinstance(source_config, Mapping) or _resume_compatible_config(
        source_config
    ) != _resume_compatible_config(config):
        raise SO214CoreRunnerError("Resume resolved configuration is incompatible.")
    resume = cohort_epoch_boundary_resume_from_checkpoint(payload)
    if _is_fixed_continuation(config) and require_fixed_source:
        trainer = _section(config, "trainer")
        required_epoch = int(trainer["required_resume_completed_global_epochs"])
        required_run_id = str(trainer["required_source_run_id"])
        required_sha256 = str(trainer["required_source_checkpoint_sha256"])
        for field, actual, expected in (
            ("path.name", resolved.name, "last.ckpt"),
            ("run_id", payload.get("run_id"), required_run_id),
            ("checkpoint_sha256", file_sha256, required_sha256),
            ("completed_global_epochs", resume.completed_global_epochs, required_epoch),
            (
                "optimizer_updates_completed",
                resume.optimizer_updates_completed,
                required_epoch * resume.optimizer_updates_per_global_epoch,
            ),
        ):
            _require_equal(actual, expected, field=f"resume.fixed_source.{field}")
        saved_plateau = payload.get("plateau")
        if not isinstance(saved_plateau, Mapping):
            raise SO214CoreRunnerError(
                "Fixed continuation source lacks its plateau decision."
            )
        decision = _resume_plateau_decision(
            [
                float(record.equal_core_mean_masked_huber)
                for record in resume.global_history
            ],
            completed_global_epochs=required_epoch,
            first_audit_epoch=150,
            audit_interval=25,
        )
        if decision is None or not decision.should_stop or decision.final_epoch != 175:
            raise SO214CoreRunnerError(
                "Fixed continuation source is not the confirmed epoch-175 plateau."
            )
        expected_plateau = _plateau_payload(
            decision,
            audit_source=str(saved_plateau.get("audit_source", "")),
        )
        if _canonical_sha256(dict(saved_plateau)) != _canonical_sha256(
            expected_plateau
        ):
            raise SO214CoreRunnerError(
                "Fixed continuation source plateau payload is inconsistent."
            )
    return resume, {
        "source_checkpoint": str(resolved),
        "source_checkpoint_sha256": file_sha256,
        "source_run_id": payload.get("run_id"),
        "completed_global_epochs": resume.completed_global_epochs,
        "optimizer_updates_completed": resume.optimizer_updates_completed,
        "resume_schema": resume.resume_schema,
        "resume_checksum": resume.resume_checksum,
        "model_state_checksum": resume.model_state_checksum,
        "optimizer_state_checksum": resume.optimizer_state_checksum,
        "amp_scaler_state_checksum": resume.scaler_state_checksum,
        "history_checksum": resume.history_checksum,
        "full_resume_payload_validated": True,
        "fixed_source_contract_verified": bool(
            _is_fixed_continuation(config) and require_fixed_source
        ),
        "source_plateau": (
            dict(payload["plateau"])
            if isinstance(payload.get("plateau"), Mapping)
            else None
        ),
    }


def _configured_resume_path(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    paths: ProjectPaths,
) -> Path | None:
    configured = _section(config, "launcher").get("resume_checkpoint")
    if args.resume_checkpoint is not None and configured is not None:
        raise SO214CoreRunnerError(
            "Specify resume checkpoint in either CLI or launcher config, not both."
        )
    selected = args.resume_checkpoint if args.resume_checkpoint is not None else configured
    return None if selected is None else _runtime_path(selected, paths)


def _copy_resume_csv(
    *,
    source_checkpoint: Path,
    archive: RunArchive,
    completed_epoch: int,
) -> None:
    destination = archive.scratch_path / "results" / "epoch_metrics.csv"
    if destination.exists():
        return
    source = source_checkpoint.parent.parent / "results" / "epoch_metrics.csv"
    if not source.is_file() or source.is_symlink():
        raise SO214CoreRunnerError(
            "Resume checkpoint requires its sibling results/epoch_metrics.csv."
        )
    archive.copy_file(source, "results/epoch_metrics.csv")
    writer = DurableEpochMetricsCSV(archive.scratch_path)
    writer.reconcile(checkpoint_epoch=int(completed_epoch))


def _copy_resume_gradient_direction_csv(
    *,
    run_id: str,
    source_checkpoint: Path,
    archive: RunArchive,
    completed_epoch: int,
) -> dict[str, Any]:
    """Copy the scalar gradient history required for a directional resume.

    Full gradient vectors are intentionally transient and never serialized.
    The source scalar rows are required so a resumed run retains one contiguous
    per-epoch diagnostic series; the first new row explicitly marks that its
    cross-checkpoint cosine is unavailable.
    """

    destination = (
        archive.scratch_path / "results" / "gradient_direction_metrics.csv"
    )
    writer = DurableGradientDirectionCSV(
        archive.scratch_path,
        run_id=run_id,
        model_seed=MODEL_SEED,
    )
    completed = int(completed_epoch)
    if destination.exists():
        writer.reconcile(checkpoint_epoch=completed)
        return {
            "schema": "so2_gradient_direction_resume_import_v1",
            "source_csv": str(destination),
            "source_csv_sha256": sha256_file(destination),
            "source_run_id": run_id,
            "destination_run_id": run_id,
            "model_seed": MODEL_SEED,
            "imported_rows": completed,
            "run_id_rebound": False,
            "existing_destination_reconciled": True,
        }

    source = (
        source_checkpoint.parent.parent
        / "results"
        / "gradient_direction_metrics.csv"
    )
    if not source.is_file() or source.is_symlink():
        raise SO214CoreRunnerError(
            "Directional resume requires its sibling scalar-only "
            "results/gradient_direction_metrics.csv."
        )
    source_sha256 = sha256_file(source)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            raw_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise SO214CoreRunnerError(
            "Directional resume source CSV cannot be read."
        ) from exc
    if fieldnames != GRADIENT_DIRECTION_METRICS_COLUMNS or not raw_rows:
        raise SO214CoreRunnerError(
            "Directional resume source CSV schema is incompatible."
        )
    source_run_ids = {str(row.get("run_id", "")) for row in raw_rows}
    source_model_seeds = {str(row.get("model_seed", "")) for row in raw_rows}
    if len(source_run_ids) != 1 or "" in source_run_ids:
        raise SO214CoreRunnerError(
            "Directional resume source CSV requires one source run ID."
        )
    if source_model_seeds != {str(MODEL_SEED)}:
        raise SO214CoreRunnerError(
            "Directional resume source CSV model seed drifted."
        )
    source_run_id = next(iter(source_run_ids))
    source_writer = DurableGradientDirectionCSV(
        source_checkpoint.parent.parent,
        run_id=source_run_id,
        model_seed=MODEL_SEED,
    )
    source_rows = source_writer.read_rows()
    if len(source_rows) < completed:
        raise SO214CoreRunnerError(
            "Directional resume source CSV ends before its checkpoint."
        )
    for row in source_rows[:completed]:
        if row.get("schema") != GRADIENT_DIRECTION_METRICS_SCHEMA:
            raise SO214CoreRunnerError(
                "Directional resume source row schema drifted."
            )
        rebound = dict(row)
        rebound["run_id"] = run_id
        writer.append(rebound)
    writer.reconcile(checkpoint_epoch=completed)
    return {
        "schema": "so2_gradient_direction_resume_import_v1",
        "source_csv": str(source.resolve(strict=True)),
        "source_csv_sha256": source_sha256,
        "source_run_id": source_run_id,
        "destination_run_id": run_id,
        "model_seed": MODEL_SEED,
        "source_rows": len(source_rows),
        "imported_rows": completed,
        "run_id_rebound": source_run_id != run_id,
        "existing_destination_reconciled": False,
    }


def _copy_resume_block_gradient_direction_csv(
    *,
    run_id: str,
    source_checkpoint: Path,
    archive: RunArchive,
    completed_epoch: int,
    expected_blocks: int = 8,
) -> dict[str, Any]:
    """Rebind and import complete per-block gradient epoch groups."""

    destination = (
        archive.scratch_path / "results" / "gradient_direction_by_block.csv"
    )
    writer = DurableBlockGradientDirectionCSV(
        archive.scratch_path,
        run_id=run_id,
        model_seed=MODEL_SEED,
        expected_blocks=expected_blocks,
    )
    completed = int(completed_epoch)
    if destination.exists():
        writer.reconcile(checkpoint_epoch=completed)
        return {
            "schema": "so2_block_gradient_direction_resume_import_v1",
            "source_csv": str(destination),
            "source_csv_sha256": sha256_file(destination),
            "source_run_id": run_id,
            "destination_run_id": run_id,
            "model_seed": MODEL_SEED,
            "source_epoch_groups": writer.completed_epochs,
            "imported_epoch_groups": completed,
            "imported_rows": completed * expected_blocks,
            "run_id_rebound": False,
            "existing_destination_reconciled": True,
        }

    source = (
        source_checkpoint.parent.parent
        / "results"
        / "gradient_direction_by_block.csv"
    )
    if not source.is_file() or source.is_symlink():
        raise SO214CoreRunnerError(
            "Block-observed resume requires its sibling scalar-only "
            "results/gradient_direction_by_block.csv."
        )
    source_sha256 = sha256_file(source)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            raw_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise SO214CoreRunnerError(
            "Block-gradient resume source CSV cannot be read."
        ) from exc
    if fieldnames != BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS or not raw_rows:
        raise SO214CoreRunnerError(
            "Block-gradient resume source CSV schema is incompatible."
        )
    source_run_ids = {str(row.get("run_id", "")) for row in raw_rows}
    source_model_seeds = {str(row.get("model_seed", "")) for row in raw_rows}
    if len(source_run_ids) != 1 or "" in source_run_ids:
        raise SO214CoreRunnerError(
            "Block-gradient resume source CSV requires one source run ID."
        )
    if source_model_seeds != {str(MODEL_SEED)}:
        raise SO214CoreRunnerError(
            "Block-gradient resume source CSV model seed drifted."
        )
    source_run_id = next(iter(source_run_ids))
    source_writer = DurableBlockGradientDirectionCSV(
        source_checkpoint.parent.parent,
        run_id=source_run_id,
        model_seed=MODEL_SEED,
        expected_blocks=expected_blocks,
    )
    source_rows = source_writer.read_rows()
    if source_writer.completed_epochs < completed:
        raise SO214CoreRunnerError(
            "Block-gradient resume source CSV ends before its checkpoint."
        )
    for epoch_index in range(completed):
        offset = epoch_index * expected_blocks
        rebound_group: list[dict[str, str]] = []
        for row in source_rows[offset : offset + expected_blocks]:
            if row.get("schema") != BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA:
                raise SO214CoreRunnerError(
                    "Block-gradient resume source row schema drifted."
                )
            rebound = dict(row)
            rebound["run_id"] = run_id
            rebound_group.append(rebound)
        writer.append(rebound_group)
    writer.reconcile(checkpoint_epoch=completed)
    return {
        "schema": "so2_block_gradient_direction_resume_import_v1",
        "source_csv": str(source.resolve(strict=True)),
        "source_csv_sha256": source_sha256,
        "source_run_id": source_run_id,
        "destination_run_id": run_id,
        "model_seed": MODEL_SEED,
        "source_epoch_groups": source_writer.completed_epochs,
        "source_rows": len(source_rows),
        "imported_epoch_groups": completed,
        "imported_rows": completed * expected_blocks,
        "run_id_rebound": source_run_id != run_id,
        "existing_destination_reconciled": False,
    }


def _is_fixed_plateau_audit_boundary(
    completed_global_epochs: int,
    *,
    first_audit_epoch: int,
    audit_interval: int,
) -> bool:
    completed = int(completed_global_epochs)
    first = int(first_audit_epoch)
    interval = int(audit_interval)
    if completed < 0 or first <= 0 or interval <= 0:
        raise SO214CoreRunnerError("Plateau audit epochs must be positive.")
    return completed >= first and (completed - first) % interval == 0


def _next_fixed_plateau_audit_epoch(
    completed_global_epochs: int,
    *,
    first_audit_epoch: int,
    audit_interval: int,
) -> int:
    """Return the first prespecified audit boundary strictly after *completed*.

    A resume at epoch 163 therefore trains only through epoch 175, rather than
    advancing an arbitrary 25 epochs to an invalid epoch-188 audit.
    """

    completed = int(completed_global_epochs)
    first = int(first_audit_epoch)
    interval = int(audit_interval)
    if completed < 0 or first <= 0 or interval <= 0:
        raise SO214CoreRunnerError("Plateau audit epochs must be positive.")
    if completed < first:
        return first
    offset = (completed - first) % interval
    return completed + (interval if offset == 0 else interval - offset)


def _resume_plateau_decision(
    losses: Sequence[float],
    *,
    completed_global_epochs: int,
    first_audit_epoch: int,
    audit_interval: int,
) -> SingleSeedPlateauDecision | None:
    """Re-evaluate a durable audit-boundary checkpoint before further training."""

    completed = int(completed_global_epochs)
    if len(losses) != completed:
        raise SO214CoreRunnerError(
            "Resume loss history does not align to its completed epoch."
        )
    if not _is_fixed_plateau_audit_boundary(
        completed,
        first_audit_epoch=first_audit_epoch,
        audit_interval=audit_interval,
    ):
        return None
    return single_seed_plateau_decision(
        losses,
        model_seed=MODEL_SEED,
        completed_global_epochs=completed,
    )


def _plateau_payload(
    decision: SingleSeedPlateauDecision,
    *,
    audit_source: str,
) -> dict[str, Any]:
    return {
        **asdict(decision),
        "audit_source": str(audit_source),
        "validation_or_test_metric": False,
        "checkpoint_selection_metric": False,
    }


def _untied8_operational_review_required(
    config: Mapping[str, Any],
    decision: SingleSeedPlateauDecision,
) -> bool:
    """Stop an unconfirmed untied-8 run at epoch 300 for explicit review."""

    return bool(
        _is_untied8(config)
        and decision.completed_global_epochs == UNTIED8_OPERATIONAL_REVIEW_EPOCH
        and not decision.should_stop
    )


def _untied8_operational_review_payload(
    *,
    run_id: str,
    parameter_count: int,
    plateau: Mapping[str, Any],
    checkpoint_sha256: str,
    epoch_metrics_rows: int,
    gradient_direction_rows: int,
    block_gradient_rows: int,
) -> dict[str, Any]:
    """Describe an honest non-success boundary without claiming convergence."""

    _require_equal(
        parameter_count,
        EXPECTED_UNTIED8_PARAMETER_COUNT,
        field="operational_review.parameter_count",
    )
    _require_equal(
        plateau.get("completed_global_epochs"),
        UNTIED8_OPERATIONAL_REVIEW_EPOCH,
        field="operational_review.completed_global_epochs",
    )
    _require_equal(
        plateau.get("should_stop"),
        False,
        field="operational_review.plateau_should_stop",
    )
    for field, observed, expected in (
        ("epoch_metrics_rows", epoch_metrics_rows, UNTIED8_OPERATIONAL_REVIEW_EPOCH),
        (
            "gradient_direction_rows",
            gradient_direction_rows,
            UNTIED8_OPERATIONAL_REVIEW_EPOCH,
        ),
        (
            "block_gradient_rows",
            block_gradient_rows,
            UNTIED8_OPERATIONAL_REVIEW_EPOCH * 8,
        ),
    ):
        _require_equal(observed, expected, field=f"operational_review.{field}")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise SO214CoreRunnerError(
            "Operational review checkpoint requires a SHA-256 identity."
        )
    return {
        "schema": "so2_14core_untied8_operational_review_required_v1",
        "run_id": run_id,
        "status": "operational_review_required",
        "completed_global_epochs": UNTIED8_OPERATIONAL_REVIEW_EPOCH,
        "optimizer_updates_completed": UNTIED8_OPERATIONAL_REVIEW_EPOCH * 7,
        "parameter_count": EXPECTED_UNTIED8_PARAMETER_COUNT,
        "checkpoint": "checkpoints/latest.ckpt",
        "checkpoint_role": "resumable_epoch_boundary_not_final_model",
        "checkpoint_sha256": checkpoint_sha256,
        "epoch_metrics_rows": epoch_metrics_rows,
        "gradient_direction_rows": gradient_direction_rows,
        "block_gradient_rows": block_gradient_rows,
        "plateau": dict(plateau),
        "plateau_confirmed": False,
        "scientific_convergence_claim": False,
        "scientific_maximum_epoch_cap": None,
        "further_training_authorized": False,
        "resume_requires_explicit_contract_amendment_and_approval": True,
        "queue_terminal_representation": "failed_inconclusive",
        "reason": (
            "Literal plateau stopping had not passed by epoch 300; the durable "
            "rolling checkpoint is preserved for explicit operational review."
        ),
    }


def _write_and_verify_untied8_training_plots(
    run_root: Path,
    *,
    expected_blocks: int = 8,
) -> dict[str, Any]:
    """Create the block-observed protocols' scalar-derived PNG artifacts."""

    written = write_so2_training_plots(
        run_root,
        expected_blocks=expected_blocks,
    )
    expected = {
        "loss_vs_epoch": str(LOSS_FIGURE_RELATIVE_PATH),
        "gradient_direction_vs_epoch": str(GRADIENT_FIGURE_RELATIVE_PATH),
    }
    _require_equal(written, expected, field="training_plots.relative_paths")
    checksums: dict[str, str] = {}
    for name, relative in expected.items():
        path = run_root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise SO214CoreRunnerError(
                f"Required training plot is missing: {relative}."
            )
        checksums[name] = sha256_file(path)
    return {
        "enabled": True,
        "descriptive_only": True,
        "affects_optimization_or_stopping": False,
        "paths": expected,
        "sha256": checksums,
    }


def _validate_gradient_direction_scalar_files(
    run_root: Path,
    *,
    gradient_csv: Path,
    block_gradient_csv: Path | None,
) -> tuple[Path, ...]:
    """Require exactly the protocol's durable scalar direction CSV artifacts."""

    expected = [gradient_csv]
    if block_gradient_csv is not None:
        expected.append(block_gradient_csv)
    expected_files = tuple(sorted(expected, key=lambda path: path.name))
    observed_files = tuple(
        sorted(
            (
                path
                for path in (run_root / "results").glob("gradient_direction*")
                if path.is_file()
            ),
            key=lambda path: path.name,
        )
    )
    _require_equal(
        observed_files,
        expected_files,
        field="gradient_direction.scalar_only_files",
    )
    return observed_files


def _strict_plateau_payload(
    losses: Sequence[float],
    *,
    trainer: Mapping[str, Any],
) -> dict[str, Any]:
    fields = strict_plateau_monitor_fields(
        losses,
        model_seed=MODEL_SEED,
        audit_interval_global_epochs=int(
            trainer["strict_plateau_audit_interval_global_epochs"]
        ),
        window_global_epochs=int(trainer["strict_plateau_window_global_epochs"]),
        absolute_relative_half_window_change_max=float(
            trainer["strict_plateau_absolute_relative_half_window_change_max"]
        ),
        normalized_absolute_slope_per_epoch_max=float(
            trainer[
                "strict_plateau_normalized_absolute_slope_per_epoch_max"
            ]
        ),
        consecutive_passing_audits=int(
            trainer["strict_plateau_consecutive_passing_audits"]
        ),
    )
    required = int(trainer["strict_plateau_consecutive_passing_audits"])
    return {
        "schema": "so2_14core_strict_plateau_diagnostic_v1",
        "completed_global_epochs": len(losses),
        "role": "diagnostic_only_never_stopping",
        "absolute_relative_half_window_change_max": float(
            trainer["strict_plateau_absolute_relative_half_window_change_max"]
        ),
        "normalized_absolute_slope_per_epoch_max": float(
            trainer[
                "strict_plateau_normalized_absolute_slope_per_epoch_max"
            ]
        ),
        "required_consecutive_passing_audits": required,
        "diagnostic_confirmed": bool(
            fields["plateau_audit_performed"]
            and fields["plateau_conditions_passed"]
            and int(fields["plateau_consecutive_passing_audits"]) >= required
        ),
        "training_stop_applied": False,
        **fields,
    }


def _write_or_verify_strict_plateau_audit(
    archive: RunArchive,
    payload: Mapping[str, Any],
) -> Path:
    """Write one immutable audit, or verify an identical prior callback write.

    Epoch 300 is both an ordinary 25-epoch audit boundary and the fixed-budget
    terminal boundary.  The epoch callback therefore owns the durable write;
    terminal finalization may only verify that same payload.  This helper keeps
    the operation idempotent without weakening the run archive's exclusive-
    write policy or permitting a drifted diagnostic to be silently replaced.
    """

    try:
        completed = int(payload["completed_global_epochs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO214CoreRunnerError(
            "Strict plateau audit lacks a valid completed epoch."
        ) from exc
    relative = Path(
        f"diagnostics/strict_plateau_audit_epoch_{completed:04d}.json"
    )
    target = archive.scratch_path / relative
    if not target.exists():
        archive.write_json(relative, dict(payload))
        return target
    try:
        existing = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SO214CoreRunnerError(
            f"Existing strict plateau audit is unreadable: {relative}."
        ) from exc
    if not isinstance(existing, Mapping) or _canonical_sha256(
        dict(existing)
    ) != _canonical_sha256(dict(payload)):
        raise SO214CoreRunnerError(
            f"Existing strict plateau audit drifted: {relative}."
        )
    return target


def _fixed_completion_payload(
    resume: CohortRelativeQKVEpochBoundaryResume,
    *,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "so2_14core_fixed_continuation_completion_v1",
        "mode": "resume_fixed_final_epoch",
        "required_source_run_id": FIXED_CONTINUATION_SOURCE_RUN_ID,
        "required_source_checkpoint_sha256": (
            FIXED_CONTINUATION_SOURCE_CHECKPOINT_SHA256
        ),
        "observed_source_checkpoint_sha256": str(source_checkpoint_sha256),
        "required_source_global_epoch": FIXED_CONTINUATION_SOURCE_EPOCH,
        "fixed_final_global_epoch": FIXED_CONTINUATION_FINAL_EPOCH,
        "completed_global_epochs": resume.completed_global_epochs,
        "optimizer_updates_completed": resume.optimizer_updates_completed,
        "fixed_budget_completed": bool(
            resume.completed_global_epochs == FIXED_CONTINUATION_FINAL_EPOCH
            and resume.optimizer_updates_completed
            == FIXED_CONTINUATION_FINAL_EPOCH
            * resume.optimizer_updates_per_global_epoch
        ),
        "plateau_stopping_enabled": False,
        "early_stop_applied": False,
        "checkpoint_selection_metric": False,
        "checkpoint_role": "final_epoch_300_last",
    }


def _fixed_continuation_plan(
    config: Mapping[str, Any],
    resume: CohortRelativeQKVEpochBoundaryResume,
) -> dict[str, Any]:
    """Validate and expose the non-negotiable continuation execution plan."""

    if not _is_fixed_continuation(config):
        raise SO214CoreRunnerError("Configuration is not a fixed continuation.")
    trainer = _section(config, "trainer")
    start = int(trainer["required_resume_completed_global_epochs"])
    end = int(trainer["fixed_final_global_epoch"])
    _require_equal(
        resume.completed_global_epochs,
        start,
        field="fixed_continuation.resume_epoch",
    )
    if end <= start:
        raise SO214CoreRunnerError("Fixed continuation target must follow its source.")
    return {
        "segment_start_global_epoch": start,
        "segment_end_global_epoch": end,
        "checkpoint_callback_enabled": False,
        "plateau_stopping_enabled": False,
        "epochs_this_segment": end - start,
        "optimizer_updates_this_segment": (
            (end - start) * resume.optimizer_updates_per_global_epoch
        ),
    }


def _validate_fixed_final_metadata(
    completion: Mapping[str, Any],
    strict_diagnostic: Mapping[str, Any],
    *,
    expected_resume: CohortRelativeQKVEpochBoundaryResume | Any,
) -> None:
    """Validate fixed completion without requiring the diagnostic to pass."""

    required_completion = {
        "schema": "so2_14core_fixed_continuation_completion_v1",
        "mode": "resume_fixed_final_epoch",
        "required_source_run_id": FIXED_CONTINUATION_SOURCE_RUN_ID,
        "required_source_checkpoint_sha256": (
            FIXED_CONTINUATION_SOURCE_CHECKPOINT_SHA256
        ),
        "observed_source_checkpoint_sha256": (
            FIXED_CONTINUATION_SOURCE_CHECKPOINT_SHA256
        ),
        "required_source_global_epoch": FIXED_CONTINUATION_SOURCE_EPOCH,
        "fixed_final_global_epoch": FIXED_CONTINUATION_FINAL_EPOCH,
        "completed_global_epochs": FIXED_CONTINUATION_FINAL_EPOCH,
        "optimizer_updates_completed": FIXED_CONTINUATION_FINAL_EPOCH * 7,
        "fixed_budget_completed": True,
        "plateau_stopping_enabled": False,
        "early_stop_applied": False,
        "checkpoint_selection_metric": False,
        "checkpoint_role": "final_epoch_300_last",
    }
    for field, expected in required_completion.items():
        _require_equal(
            completion.get(field),
            expected,
            field=f"final_reload.completion.{field}",
        )
    _require_equal(
        expected_resume.completed_global_epochs,
        FIXED_CONTINUATION_FINAL_EPOCH,
        field="final_reload.fixed_completed_global_epochs",
    )
    _require_equal(
        expected_resume.optimizer_updates_completed,
        FIXED_CONTINUATION_FINAL_EPOCH * 7,
        field="final_reload.fixed_optimizer_updates_completed",
    )
    for field, expected in {
        "schema": "so2_14core_strict_plateau_diagnostic_v1",
        "completed_global_epochs": FIXED_CONTINUATION_FINAL_EPOCH,
        "role": "diagnostic_only_never_stopping",
        "training_stop_applied": False,
        "plateau_should_stop": False,
    }.items():
        _require_equal(
            strict_diagnostic.get(field),
            expected,
            field=f"final_reload.strict_plateau_diagnostic.{field}",
        )
    if not isinstance(strict_diagnostic.get("diagnostic_confirmed"), bool):
        raise SO214CoreRunnerError(
            "Final strict plateau diagnostic requires a boolean result."
        )


def _validate_final_checkpoint_payload(
    checkpoint: Path,
    *,
    config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    expected_resume: CohortRelativeQKVEpochBoundaryResume,
    expected_plateau: Mapping[str, Any] | None = None,
    expected_completion: Mapping[str, Any] | None = None,
    expected_source_plateau: Mapping[str, Any] | None = None,
    expected_strict_plateau_diagnostic: Mapping[str, Any] | None = None,
) -> tuple[CohortRelativeQKVEpochBoundaryResume, dict[str, Any]]:
    """Reload and validate the complete serialized continuation payload."""

    loaded_resume, load_receipt = _load_resume_checkpoint(
        checkpoint,
        config=config,
        model_construction=model_construction,
        require_fixed_source=False,
    )
    for field in (
        "completed_global_epochs",
        "optimizer_updates_completed",
        "model_state_checksum",
        "optimizer_state_checksum",
        "scaler_state_checksum",
        "history_checksum",
        "resume_checksum",
    ):
        _require_equal(
            getattr(loaded_resume, field),
            getattr(expected_resume, field),
            field=f"final_reload.{field}",
        )
    try:
        raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SO214CoreRunnerError("Final checkpoint cannot be reloaded.") from exc
    if not isinstance(raw_payload, Mapping):
        raise SO214CoreRunnerError("Final checkpoint payload must be a mapping.")
    fixed_completion = expected_completion is not None
    if fixed_completion:
        for field, expected in (
            ("completion", expected_completion),
            ("source_plateau", expected_source_plateau),
            ("strict_plateau_diagnostic", expected_strict_plateau_diagnostic),
        ):
            saved = raw_payload.get(field)
            if not isinstance(saved, Mapping) or not isinstance(expected, Mapping):
                raise SO214CoreRunnerError(
                    f"Final fixed continuation checkpoint lacks {field}."
                )
            if _canonical_sha256(dict(saved)) != _canonical_sha256(dict(expected)):
                raise SO214CoreRunnerError(
                    f"Final fixed continuation checkpoint {field} drifted."
                )
        _validate_fixed_final_metadata(
            raw_payload["completion"],
            raw_payload["strict_plateau_diagnostic"],
            expected_resume=expected_resume,
        )
    else:
        if expected_plateau is None:
            raise SO214CoreRunnerError("Final plateau expectation is missing.")
        saved_plateau = raw_payload.get("plateau")
        if not isinstance(saved_plateau, Mapping):
            raise SO214CoreRunnerError("Final checkpoint lacks its plateau decision.")
        if _canonical_sha256(dict(saved_plateau)) != _canonical_sha256(
            dict(expected_plateau)
        ):
            raise SO214CoreRunnerError("Final checkpoint plateau payload drifted.")
        if (
            saved_plateau.get("should_stop") is not True
            or int(saved_plateau.get("final_epoch", -1))
            != expected_resume.completed_global_epochs
        ):
            raise SO214CoreRunnerError(
                "Final checkpoint does not encode a confirmed plateau epoch."
            )
    return loaded_resume, {
        **load_receipt,
        "full_resume_payload_validated": True,
        "plateau_payload_validated": not fixed_completion,
        "fixed_completion_payload_validated": fixed_completion,
        "model_state_checksum": loaded_resume.model_state_checksum,
        "optimizer_state_checksum": loaded_resume.optimizer_state_checksum,
        "amp_scaler_state_checksum": loaded_resume.scaler_state_checksum,
        "history_checksum": loaded_resume.history_checksum,
        "resume_checksum": loaded_resume.resume_checksum,
    }


@torch.inference_mode()
def _fixed_checkpoint_prediction_replay(
    *,
    in_memory_model: torch.nn.Module,
    reloaded_model: torch.nn.Module,
    batch: Any,
    expected_model_state_checksum: str,
    device: torch.device,
    atol: float = FINAL_RELOAD_PREDICTION_ATOL,
    rtol: float = FINAL_RELOAD_PREDICTION_RTOL,
) -> dict[str, Any]:
    """Compare deterministic fixed inputs after a fresh strict model reload."""

    in_memory_model.to(device)
    reloaded_model.to(device)
    in_memory_model.eval()
    reloaded_model.eval()
    in_memory_checksum = _tree_sha256(in_memory_model.state_dict())
    reloaded_checksum = _tree_sha256(reloaded_model.state_dict())
    if (
        in_memory_checksum != expected_model_state_checksum
        or reloaded_checksum != expected_model_state_checksum
    ):
        raise SO214CoreRunnerError(
            "Final in-memory/reloaded model state checksum differs from checkpoint."
        )

    seed = derive_mask_seed(
        HELD_IN_MASK_BASE_SEED,
        "so2-relative-qkv-final-checkpoint-replay-v1",
        batch.alias,
    )
    realization = sample_uniform_mask_numpy(
        batch.n_nodes,
        batch.n_genes,
        seed=seed,
    )
    receiver_count = min(FINAL_RELOAD_RECEIVER_COUNT, int(batch.n_nodes))
    receivers = np.unique(
        np.linspace(
            0,
            int(batch.n_nodes) - 1,
            num=receiver_count,
            dtype=np.int64,
        )
    )
    target = batch.target_expression.to(device=device)
    covariates = batch.node_covariates.to(device=device)
    mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(
        device=device,
        dtype=torch.bool,
    )
    masked_input = target.masked_fill(mask, 0.0)

    def predict(selected_model: torch.nn.Module) -> torch.Tensor:
        output = selected_model(
            input_expression=masked_input,
            gene_mask=mask,
            edge_index=batch.edge_index,
            relative_geometry=batch.relative_geometry,
            node_covariates=covariates,
            target_nodes=receivers.tolist(),
        )
        prediction = output.prediction if hasattr(output, "prediction") else output
        if not torch.is_tensor(prediction):
            raise SO214CoreRunnerError("Checkpoint replay did not return predictions.")
        return prediction.detach().float().cpu().clone()

    in_memory_prediction = predict(in_memory_model)
    reloaded_prediction = predict(reloaded_model)
    if (
        in_memory_prediction.shape != reloaded_prediction.shape
        or not bool(torch.isfinite(in_memory_prediction).all())
        or not bool(torch.isfinite(reloaded_prediction).all())
    ):
        raise SO214CoreRunnerError(
            "Final checkpoint replay prediction shape/finite gate failed."
        )
    difference = (in_memory_prediction - reloaded_prediction).abs()
    within_tolerance = torch.allclose(
        in_memory_prediction,
        reloaded_prediction,
        atol=float(atol),
        rtol=float(rtol),
    )
    if not within_tolerance:
        raise SO214CoreRunnerError(
            "Fresh final checkpoint predictions differ from the in-memory model."
        )
    receipt = {
        "schema": "so2_14core_final_checkpoint_prediction_replay_v1",
        "verified": True,
        "core_alias": batch.alias,
        "receiver_indices": receivers.tolist(),
        "mask_seed": int(seed),
        "mask_checksum": realization.checksum,
        "model_state_checksum": expected_model_state_checksum,
        "in_memory_model_state_checksum": in_memory_checksum,
        "reloaded_model_state_checksum": reloaded_checksum,
        "in_memory_prediction_checksum": _tree_sha256(in_memory_prediction),
        "reloaded_prediction_checksum": _tree_sha256(reloaded_prediction),
        "prediction_shape": list(in_memory_prediction.shape),
        "maximum_absolute_difference": float(difference.max().item()),
        "mean_absolute_difference": float(difference.mean().item()),
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "execution_device": str(device),
        "amp_enabled": False,
    }
    del target, covariates, mask, masked_input
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt


def _verify_final_checkpoint_reload(
    checkpoint: Path,
    *,
    expected_file_sha256: str,
    config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    parameter_count: int,
    expected_resume: CohortRelativeQKVEpochBoundaryResume,
    expected_plateau: Mapping[str, Any] | None = None,
    expected_completion: Mapping[str, Any] | None = None,
    expected_source_plateau: Mapping[str, Any] | None = None,
    expected_strict_plateau_diagnostic: Mapping[str, Any] | None = None,
    in_memory_model: torch.nn.Module,
    replay_batch: Any,
    device: torch.device,
) -> dict[str, Any]:
    loaded_resume, payload_receipt = _validate_final_checkpoint_payload(
        checkpoint,
        config=config,
        model_construction=model_construction,
        expected_resume=expected_resume,
        expected_plateau=expected_plateau,
        expected_completion=expected_completion,
        expected_source_plateau=expected_source_plateau,
        expected_strict_plateau_diagnostic=(
            expected_strict_plateau_diagnostic
        ),
    )
    _require_equal(
        payload_receipt["source_checkpoint_sha256"],
        expected_file_sha256,
        field="final_reload.checkpoint_file_sha256",
    )
    verification_model = _model_from_config(
        config,
        num_genes=replay_batch.n_genes,
        node_covariate_dim=int(replay_batch.node_covariates.shape[1]),
    )
    _require_equal(
        sum(parameter.numel() for parameter in verification_model.parameters()),
        int(parameter_count),
        field="final_reload.parameter_count",
    )
    verification_model.load_state_dict(loaded_resume.model_state_dict, strict=True)
    replay = _fixed_checkpoint_prediction_replay(
        in_memory_model=in_memory_model,
        reloaded_model=verification_model,
        batch=replay_batch,
        expected_model_state_checksum=loaded_resume.model_state_checksum,
        device=device,
    )
    del verification_model
    return {
        "schema": "so2_14core_final_checkpoint_reload_verification_v1",
        "verified": True,
        "checkpoint_file_sha256": expected_file_sha256,
        "payload": payload_receipt,
        "fixed_prediction_replay": replay,
    }


def _held_in_diagnostics(
    model: torch.nn.Module,
    batches: Sequence[Any],
    *,
    amp: bool,
    amp_dtype: str,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    model.eval()
    core_metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in batches:
            seed = derive_mask_seed(
                HELD_IN_MASK_BASE_SEED,
                "so2-relative-qkv-held-in-fit-diagnostic",
                batch.alias,
            )
            realization = sample_uniform_mask_numpy(
                batch.n_nodes, batch.n_genes, seed=seed
            )
            retry = 0
            while int(realization.masked_gene_counts.sum()) == 0:
                retry += 1
                seed = derive_mask_seed(seed, "nonzero-held-in", retry)
                realization = sample_uniform_mask_numpy(
                    batch.n_nodes, batch.n_genes, seed=seed
                )
            target = batch.target_expression.to(device=device)
            covariates = batch.node_covariates.to(device=device)
            mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(
                device=device, dtype=torch.bool
            )
            with _autocast_context(enabled=amp, device=device, dtype_name=amp_dtype):
                output = model(
                    input_expression=target.masked_fill(mask, 0.0),
                    gene_mask=mask,
                    edge_index=batch.edge_index,
                    relative_geometry=batch.relative_geometry,
                    node_covariates=covariates,
                )
            prediction = output.prediction.float()
            truth = target.float()
            selected_prediction = prediction[mask]
            selected_truth = truth[mask]
            residual = selected_prediction - selected_truth
            huber = F.huber_loss(
                selected_prediction, selected_truth, delta=1.0, reduction="mean"
            )
            mae = residual.abs().mean()
            mse = residual.square().mean()
            centered = selected_truth - selected_truth.mean()
            r2 = 1.0 - residual.square().sum() / centered.square().sum().clamp_min(
                torch.finfo(torch.float32).tiny
            )
            values = {
                "alias": batch.alias,
                "mask_seed": int(seed),
                "mask_checksum": realization.checksum,
                "n_masked_entries": int(mask.sum().item()),
                "masked_huber": float(huber.cpu()),
                "masked_mae": float(mae.cpu()),
                "masked_mse": float(mse.cpu()),
                "masked_r2": float(r2.cpu()),
                "mean_y_true": float(selected_truth.mean().cpu()),
                "mean_y_pred": float(selected_prediction.mean().cpu()),
            }
            if not all(
                math.isfinite(values[name])
                for name in ("masked_huber", "masked_mae", "masked_mse", "masked_r2")
            ):
                raise FloatingPointError("Held-in fit diagnostics are non-finite.")
            core_metrics.append(values)
            prediction_rows.append(
                {
                    "core_alias": batch.alias,
                    "dataset_id": "cosmx_so2_14core_pooled_fit_v1",
                    "split": "fit",
                    "y_true": values["mean_y_true"],
                    "y_pred": values["mean_y_pred"],
                    "sample_loss": values["masked_huber"],
                    "node_count": batch.n_nodes,
                    "edge_count": batch.n_edges,
                }
            )
            del target, covariates, mask, output, prediction, truth
            torch.cuda.empty_cache()
    return (
        {
            "fit/uniform_per_cell/masked_huber": float(
                np.mean([row["masked_huber"] for row in core_metrics])
            ),
            "fit/uniform_per_cell/masked_mae": float(
                np.mean([row["masked_mae"] for row in core_metrics])
            ),
            "fit/uniform_per_cell/masked_mse": float(
                np.mean([row["masked_mse"] for row in core_metrics])
            ),
            "fit/uniform_per_cell/masked_r2": float(
                np.mean([row["masked_r2"] for row in core_metrics])
            ),
        },
        core_metrics,
        prediction_rows,
    )


def run_distributed(
    args: argparse.Namespace,
    *,
    rank: int,
    local_rank: int,
) -> dict[str, Any] | None:
    started = time.monotonic()
    run_id, scratch_path, config, paths = _worker_inputs(args)
    fixed_mode = _is_fixed_continuation(config)
    torch.cuda.set_device(local_rank)
    dataset = _section(config, "dataset")
    cohort_dir = _runtime_path(dataset["prepared_artifact"], paths)
    graph_dir = _runtime_path(dataset["prepared_graph_artifact"], paths)
    preflight = _validate_hardware_preflight(
        config,
        paths=paths,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    batches = load_so2_relative_qkv_batches(
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    if tuple(batch.alias for batch in batches) != SO2_ALIASES:
        raise SO214CoreRunnerError("Loaded batches do not match SO2-C15--SO2-C28.")
    model = _model_from_config(
        config,
        num_genes=batches[0].n_genes,
        node_covariate_dim=int(batches[0].node_covariates.shape[1]),
    )
    construction = _model_construction(
        model,
        config,
        num_genes=batches[0].n_genes,
        node_covariate_dim=int(batches[0].node_covariates.shape[1]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameter_count = _expected_parameter_count(config)
    if expected_parameter_count is not None:
        _require_equal(
            parameter_count,
            expected_parameter_count,
            field="parameter_count",
        )
    resume_path = _configured_resume_path(args, config, paths)
    resume: CohortRelativeQKVEpochBoundaryResume | None = None
    resume_receipt: dict[str, Any] | None = None
    if resume_path is not None:
        resume, resume_receipt = _load_resume_checkpoint(
            resume_path,
            config=config,
            model_construction=construction,
        )
        if (
            _is_untied8(config)
            and resume.completed_global_epochs > UNTIED8_OPERATIONAL_REVIEW_EPOCH
        ):
            raise SO214CoreRunnerError(
                "Untied-8 checkpoints past the epoch-300 operational review "
                "boundary require an explicit campaign amendment and approval."
            )
    if fixed_mode and (resume is None or resume_receipt is None):
        raise SO214CoreRunnerError(
            "Fixed continuation requires its locked epoch-175 resume checkpoint."
        )

    gradient_direction_enabled = _gradient_direction_enabled(config)
    block_gradient_count = _block_gradient_count(config)
    block_gradient_direction_enabled = block_gradient_count > 0
    block_parameter_count = _block_trainable_parameter_count(config)
    gradient_tracker: FullGradientDirectionTracker | None = None
    block_gradient_tracker: BlockGradientDirectionTracker | None = None
    if gradient_direction_enabled and rank == 0:
        gradient_tracker = FullGradientDirectionTracker(
            expected_optimizer_updates_per_epoch=7,
            resume_boundary_unavailable=resume is not None,
        )
    if block_gradient_direction_enabled and rank == 0:
        block_gradient_tracker = BlockGradientDirectionTracker(
            expected_blocks=block_gradient_count,
            expected_optimizer_updates_per_epoch=7,
            resume_boundary_unavailable=resume is not None,
        )

    trainer = _section(config, "trainer")
    archive: RunArchive | None = None
    metrics_writer: DurableEpochMetricsCSV | None = None
    gradient_writer: DurableGradientDirectionCSV | None = None
    block_gradient_writer: DurableBlockGradientDirectionCSV | None = None
    gradient_resume_lineage: dict[str, Any] | None = None
    block_gradient_resume_lineage: dict[str, Any] | None = None
    checkpoint_store: AtomicLatestCheckpointStore | None = None
    if rank == 0:
        archive = RunArchive.attach_active(
            run_id, paths=paths, scratch_path=scratch_path
        )
        if resume is not None:
            assert resume_path is not None
            _copy_resume_csv(
                source_checkpoint=resume_path,
                archive=archive,
                completed_epoch=resume.completed_global_epochs,
            )
            if gradient_direction_enabled:
                gradient_resume_lineage = _copy_resume_gradient_direction_csv(
                    run_id=run_id,
                    source_checkpoint=resume_path,
                    archive=archive,
                    completed_epoch=resume.completed_global_epochs,
                )
                archive.write_json(
                    "provenance/gradient_direction_resume_import.json",
                    gradient_resume_lineage,
                )
            if block_gradient_direction_enabled:
                block_gradient_resume_lineage = (
                    _copy_resume_block_gradient_direction_csv(
                        run_id=run_id,
                        source_checkpoint=resume_path,
                        archive=archive,
                        completed_epoch=resume.completed_global_epochs,
                        expected_blocks=block_gradient_count,
                    )
                )
                archive.write_json(
                    "provenance/block_gradient_direction_resume_import.json",
                    block_gradient_resume_lineage,
                )
            archive.write_json("provenance/resume_source.json", resume_receipt)
            if fixed_mode:
                assert resume_receipt is not None
                source_csv = resume_path.parent.parent / "results/epoch_metrics.csv"
                source_verification = {
                    "schema": "so2_14core_source_checkpoint_verification_v1",
                    **resume_receipt,
                    "source_epoch_metrics_csv": str(source_csv.resolve(strict=True)),
                    "source_epoch_metrics_csv_sha256": sha256_file(source_csv),
                    "source_epoch_metrics_rows": resume.completed_global_epochs,
                    "verified": True,
                }
                archive.write_json(
                    "provenance/source_checkpoint_verification.json",
                    source_verification,
                )
                archive.write_json(
                    "provenance/continuation_lineage.json",
                    {
                        "schema": "so2_14core_fixed_continuation_lineage_v1",
                        "source_run_id": resume_receipt["source_run_id"],
                        "source_checkpoint": resume_receipt["source_checkpoint"],
                        "source_checkpoint_sha256": resume_receipt[
                            "source_checkpoint_sha256"
                        ],
                        "source_completed_global_epochs": (
                            resume.completed_global_epochs
                        ),
                        "source_optimizer_updates_completed": (
                            resume.optimizer_updates_completed
                        ),
                        "first_new_global_epoch": (
                            resume.completed_global_epochs + 1
                        ),
                        "fixed_final_global_epoch": (
                            FIXED_CONTINUATION_FINAL_EPOCH
                        ),
                        "additional_global_epochs": (
                            FIXED_CONTINUATION_FINAL_EPOCH
                            - resume.completed_global_epochs
                        ),
                        "source_bundle_mutation_allowed": False,
                        "source_checkpoint_retained": True,
                        "checkpoint_policy": "final_last_only_no_intermediate",
                        "plateau_role": "post_hoc_exploratory_diagnostic_only",
                    },
                )
        metrics_writer = DurableEpochMetricsCSV(archive.scratch_path)
        metrics_writer.reconcile(
            checkpoint_epoch=0 if resume is None else resume.completed_global_epochs
        )
        if gradient_direction_enabled:
            gradient_writer = DurableGradientDirectionCSV(
                archive.scratch_path,
                run_id=run_id,
                model_seed=MODEL_SEED,
            )
            gradient_writer.reconcile(
                checkpoint_epoch=(
                    0 if resume is None else resume.completed_global_epochs
                )
            )
        if block_gradient_direction_enabled:
            block_gradient_writer = DurableBlockGradientDirectionCSV(
                archive.scratch_path,
                run_id=run_id,
                model_seed=MODEL_SEED,
                expected_blocks=block_gradient_count,
            )
            block_gradient_writer.reconcile(
                checkpoint_epoch=(
                    0 if resume is None else resume.completed_global_epochs
                )
            )
        checkpoint_store = AtomicLatestCheckpointStore(
            archive.scratch_path,
            completed_global_epochs=(
                0 if resume is None else resume.completed_global_epochs
            ),
        )
        archive.write_json(
            "metrics/training_monitor_contract.json",
            {
                "schema": "so2_14core_training_monitor_v1",
                "csv": "results/epoch_metrics.csv",
                "jsonl": "metrics/events.jsonl",
                "frequency": "after_every_completed_global_epoch",
                "fsync": True,
                "core_loss_columns": list(SO2_ALIASES),
                "gradient_norm_unit": "seven_paired_core_optimizer_updates",
                "gradient_direction_observability": (
                    {
                        "enabled": True,
                        "schema": GRADIENT_DIRECTION_METRICS_SCHEMA,
                        "csv": "results/gradient_direction_metrics.csv",
                        "columns": list(GRADIENT_DIRECTION_METRICS_COLUMNS),
                        "sampling_point": (
                            "rank0_full_ddp_averaged_fp32_trainable_gradient_"
                            "after_amp_unscale_and_finite_check_before_clipping"
                        ),
                        "optimizer_step_pairing": (
                            "consecutive_updates_across_epoch_boundaries_when_"
                            "uninterrupted"
                        ),
                        "epoch_aggregate": "sum_of_seven_preclip_update_gradients",
                        "serialized_values": "epoch_scalars_only",
                        "gradient_tensors_serialized": False,
                        "checkpoint_state_saved": False,
                        "resume_behavior": (
                            "first_new_epoch_marks_cross_boundary_cosines_"
                            "unavailable"
                        ),
                        "interpretation": (
                            "directional_diagnostics_must_be_interpreted_jointly_"
                            "with_training_loss_and_gradient_norm_not_as_a_"
                            "standalone_convergence_criterion"
                        ),
                        "block_gradient_direction_observability": (
                            {
                                "enabled": True,
                                "scope": "graph_blocks_only",
                                "schema": BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
                                "csv": "results/gradient_direction_by_block.csv",
                                "columns": list(
                                    BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS
                                ),
                                "rows_per_completed_global_epoch": (
                                    block_gradient_count
                                ),
                                "block_names": [
                                    f"blocks.{index}"
                                    for index in range(block_gradient_count)
                                ],
                                "serialized_values": "epoch_block_scalars_only",
                                "gradient_tensors_serialized": False,
                                "checkpoint_state_saved": False,
                            }
                            if block_gradient_direction_enabled
                            else {"enabled": False}
                        ),
                    }
                    if gradient_direction_enabled
                    else {"enabled": False, "legacy_protocol_preserved": True}
                ),
                "throughput_scope": "140_complete_graph_mask_views_per_epoch",
                "checkpoint_policy": (
                    "final_last_only_no_intermediate"
                    if fixed_mode
                    else "atomic_latest_then_final_last_only"
                ),
                "plateau_monitoring": (
                    "strict_diagnostic_only_never_stopping"
                    if fixed_mode
                    else "legacy_training_loss_stopping_rule"
                ),
                "fixed_final_global_epoch": (
                    FIXED_CONTINUATION_FINAL_EPOCH if fixed_mode else None
                ),
                "strict_plateau_absolute_relative_half_window_change_max": (
                    trainer[
                        "strict_plateau_absolute_relative_half_window_change_max"
                    ]
                    if fixed_mode
                    else None
                ),
                "strict_plateau_normalized_absolute_slope_per_epoch_max": (
                    trainer[
                        "strict_plateau_normalized_absolute_slope_per_epoch_max"
                    ]
                    if fixed_mode
                    else None
                ),
                "validation_or_test_metric": False,
            },
        )
        archive.write_json("diagnostics/hardware_preflight.json", preflight)

    observed_losses = (
        []
        if resume is None
        else [
            float(record.equal_core_mean_masked_huber)
            for record in resume.global_history
        ]
    )

    def epoch_callback(
        epoch: CohortRelativeQKVGlobalEpochRecord,
        core_records: tuple[CohortRelativeQKVCoreRecord, ...],
        update_records: tuple[CohortRelativeQKVOptimizerUpdateRecord, ...],
        duration_seconds: float,
        peak_vram_gib_all_ranks: float,
    ) -> None:
        assert rank == 0 and archive is not None and metrics_writer is not None
        if epoch.completed_global_epochs != len(observed_losses) + 1:
            raise SO214CoreRunnerError("Epoch callback is not contiguous.")
        observed_losses.append(float(epoch.equal_core_mean_masked_huber))
        row = build_epoch_metrics_row(
            run_id=run_id,
            model_seed=MODEL_SEED,
            global_epoch=epoch.completed_global_epochs,
            equal_core_mean_masked_huber=epoch.equal_core_mean_masked_huber,
            per_core_masked_huber={
                record.alias: record.masked_huber_loss for record in core_records
            },
            gradient_norms_before_clip=[
                record.gradient_norm for record in update_records
            ],
            learning_rate=float(_section(config, "trainer")["learning_rate"]),
            optimizer_updates_this_epoch=epoch.optimizer_updates_this_epoch,
            cumulative_optimizer_updates=epoch.cumulative_optimizer_updates,
            epoch_duration_seconds=duration_seconds,
            prior_epoch_durations=metrics_writer.prior_durations,
            complete_graph_mask_views=sum(
                int(record.n_mask_views) for record in core_records
            ),
            masked_entries_across_views=sum(
                int(record.n_masked_entries_across_views)
                for record in core_records
            ),
            peak_vram_gib_all_ranks=peak_vram_gib_all_ranks,
            loss_history=observed_losses,
            strict_plateau_diagnostic=fixed_mode,
            strict_plateau_audit_interval_global_epochs=int(
                trainer.get("strict_plateau_audit_interval_global_epochs", 25)
            ),
            strict_plateau_window_global_epochs=int(
                trainer.get("strict_plateau_window_global_epochs", 50)
            ),
            strict_plateau_absolute_relative_half_window_change_max=float(
                trainer.get(
                    "strict_plateau_absolute_relative_half_window_change_max",
                    0.0005,
                )
            ),
            strict_plateau_normalized_absolute_slope_per_epoch_max=float(
                trainer.get(
                    "strict_plateau_normalized_absolute_slope_per_epoch_max",
                    0.000025,
                )
            ),
            strict_plateau_consecutive_passing_audits=int(
                trainer.get("strict_plateau_consecutive_passing_audits", 2)
            ),
        )
        metrics_writer.append(row)
        gradient_summary = None
        block_gradient_summaries = ()
        if gradient_direction_enabled:
            assert gradient_tracker is not None and gradient_writer is not None
            gradient_summary = gradient_tracker.complete_epoch(epoch.global_epoch)
            _require_equal(
                gradient_summary.global_epoch,
                epoch.completed_global_epochs,
                field="gradient_direction.global_epoch",
            )
            _require_equal(
                gradient_summary.trainable_parameter_count,
                parameter_count,
                field="gradient_direction.trainable_parameter_count",
            )
            _require_equal(
                gradient_summary.optimizer_updates_observed,
                epoch.optimizer_updates_this_epoch,
                field="gradient_direction.optimizer_updates_observed",
            )
            gradient_writer.append(gradient_summary)
        if block_gradient_direction_enabled:
            assert (
                block_gradient_tracker is not None
                and block_gradient_writer is not None
                and block_parameter_count is not None
            )
            block_gradient_summaries = block_gradient_tracker.complete_epoch(
                epoch.global_epoch
            )
            _require_equal(
                len(block_gradient_summaries),
                block_gradient_count,
                field="block_gradient_direction.summary_count",
            )
            for index, summary in enumerate(block_gradient_summaries):
                for field, actual, expected in (
                    ("global_epoch", summary.global_epoch, epoch.completed_global_epochs),
                    ("block_index", summary.block_index, index),
                    ("block_name", summary.block_name, f"blocks.{index}"),
                    (
                        "trainable_parameter_count",
                        summary.trainable_parameter_count,
                        block_parameter_count,
                    ),
                    (
                        "optimizer_updates_observed",
                        summary.optimizer_updates_observed,
                        epoch.optimizer_updates_this_epoch,
                    ),
                ):
                    _require_equal(
                        actual,
                        expected,
                        field=f"block_gradient_direction[{index}].{field}",
                    )
            block_gradient_writer.append(block_gradient_summaries)
        if fixed_mode and bool(row["plateau_audit_performed"]):
            strict_payload = _strict_plateau_payload(
                observed_losses,
                trainer=trainer,
            )
            _write_or_verify_strict_plateau_audit(archive, strict_payload)
        archive.append_metric_event(
            {
                "name": "fit/training/equal_core_mean_masked_huber",
                "value": row["equal_core_mean_masked_huber"],
                "step": row["global_epoch"],
                "phase": "training_epoch",
                "monitor": row,
            }
        )

        print(
            STDOUT_PREFIX
            + " "
            + json.dumps(
                {
                    "run_id": run_id,
                    "global_epoch": row["global_epoch"],
                    "equal_core_mean_masked_huber": row[
                        "equal_core_mean_masked_huber"
                    ],
                    "gradient_norm_mean_before_clip": row[
                        "gradient_norm_mean_before_clip"
                    ],
                    "consecutive_optimizer_step_cosine_mean": (
                        None
                        if gradient_summary is None
                        else gradient_summary.consecutive_optimizer_step_cosine_mean
                    ),
                    "epoch_aggregate_gradient_cosine_to_previous_epoch": (
                        None
                        if gradient_summary is None
                        else getattr(
                            gradient_summary,
                            "epoch_aggregate_gradient_cosine_to_previous_epoch",
                        )
                    ),
                    "epoch_duration_seconds": row["epoch_duration_seconds"],
                    "eta_to_epoch_200_seconds": row["eta_to_epoch_200_seconds"],
                    "peak_vram_gib_all_ranks": row["peak_vram_gib_all_ranks"],
                    "plateau_should_stop": row["plateau_should_stop"],
                    "csv": str(metrics_writer.path),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )

    def observe_gradient_direction(
        model_with_gradients: torch.nn.Module,
        context: Any,
    ) -> None:
        if gradient_tracker is not None:
            gradient_tracker.observe(model_with_gradients, context)
        if block_gradient_tracker is not None:
            block_gradient_tracker.observe(model_with_gradients, context)

    def checkpoint_callback(state: CohortRelativeQKVEpochBoundaryResume) -> None:
        assert rank == 0 and checkpoint_store is not None
        checkpoint_store.save(
            _checkpoint_payload(
                run_id=run_id,
                config=config,
                model_construction=construction,
                parameter_count=parameter_count,
                resume=state,
            ),
            completed_global_epochs=state.completed_global_epochs,
        )

    def persist_untied8_operational_review(
        state: CohortRelativeQKVEpochBoundaryResume,
        plateau_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist a resumable, explicitly non-success epoch-300 boundary."""

        assert (
            rank == 0
            and archive is not None
            and checkpoint_store is not None
            and metrics_writer is not None
            and gradient_writer is not None
            and block_gradient_writer is not None
        )
        _require_equal(
            state.completed_global_epochs,
            UNTIED8_OPERATIONAL_REVIEW_EPOCH,
            field="operational_review.resume_epoch",
        )
        review_checkpoint_store = checkpoint_store
        if not checkpoint_store.latest_path.is_file():
            # A process launched directly from an epoch-300 source checkpoint
            # still materializes a current-run, run-ID-bound rolling checkpoint
            # before taking the deliberate non-success exit.
            review_checkpoint_store = AtomicLatestCheckpointStore(
                archive.scratch_path,
                completed_global_epochs=0,
            )
            review_checkpoint_store.save(
                _checkpoint_payload(
                    run_id=run_id,
                    config=config,
                    model_construction=construction,
                    parameter_count=parameter_count,
                    resume=state,
                ),
                completed_global_epochs=state.completed_global_epochs,
            )
        loaded_checkpoint = review_checkpoint_store.load_latest()
        _require_equal(
            int(loaded_checkpoint["completed_global_epochs"]),
            UNTIED8_OPERATIONAL_REVIEW_EPOCH,
            field="operational_review.checkpoint_epoch",
        )
        latest = require_latest_only_checkpoint_layout(scratch_path, final=False)
        reloaded_state, checkpoint_reload_verification = _load_resume_checkpoint(
            latest,
            config=config,
            model_construction=construction,
            require_fixed_source=False,
        )
        for field in (
            "completed_global_epochs",
            "optimizer_updates_completed",
            "model_state_checksum",
            "optimizer_state_checksum",
            "scaler_state_checksum",
            "history_checksum",
            "resume_checksum",
        ):
            _require_equal(
                getattr(reloaded_state, field),
                getattr(state, field),
                field=f"operational_review.checkpoint_reload.{field}",
            )
        model.load_state_dict(reloaded_state.model_state_dict, strict=True)
        checkpoint_reload_verification = {
            **checkpoint_reload_verification,
            "strict_model_state_reload_verified": True,
        }
        _require_equal(
            loaded_checkpoint.get("run_id"),
            run_id,
            field="operational_review.checkpoint_reload.run_id",
        )
        _require_equal(
            loaded_checkpoint.get("parameter_count"),
            parameter_count,
            field="operational_review.checkpoint_reload.parameter_count",
        )
        checkpoint_sha256 = sha256_file(latest)
        epoch_rows = metrics_writer.read_rows()
        gradient_rows = gradient_writer.read_rows()
        block_rows = block_gradient_writer.read_rows()
        review = _untied8_operational_review_payload(
            run_id=run_id,
            parameter_count=parameter_count,
            plateau=plateau_payload,
            checkpoint_sha256=checkpoint_sha256,
            epoch_metrics_rows=len(epoch_rows),
            gradient_direction_rows=len(gradient_rows),
            block_gradient_rows=len(block_rows),
        )
        _validate_gradient_direction_scalar_files(
            archive.scratch_path,
            gradient_csv=gradient_writer.path,
            block_gradient_csv=block_gradient_writer.path,
        )
        plots = _write_and_verify_untied8_training_plots(archive.scratch_path)
        review.update(
            {
                "campaign_id": _campaign_id(config),
                "model_name": str(_section(config, "model")["name"]),
                "peak_vram_gib": max(
                    float(row["peak_vram_gib_all_ranks"]) for row in epoch_rows
                ),
                "duration_seconds": time.monotonic() - started,
                "epoch_metrics_csv": "results/epoch_metrics.csv",
                "gradient_direction_observability": {
                    "schema": GRADIENT_DIRECTION_METRICS_SCHEMA,
                    "csv": "results/gradient_direction_metrics.csv",
                    "csv_sha256": sha256_file(gradient_writer.path),
                    "completed_epoch_rows": len(gradient_rows),
                    "final_epoch_summary": dict(gradient_rows[-1]),
                },
                "block_gradient_direction_observability": {
                    "schema": BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
                    "csv": "results/gradient_direction_by_block.csv",
                    "csv_sha256": sha256_file(block_gradient_writer.path),
                    "completed_epoch_rows": len(block_rows),
                    "completed_epochs": block_gradient_writer.completed_epochs,
                    "rows_per_completed_global_epoch": 8,
                    "final_epoch_summaries": [
                        dict(row) for row in block_rows[-8:]
                    ],
                },
                "gradient_direction_resume_lineage": gradient_resume_lineage,
                "block_gradient_direction_resume_lineage": (
                    block_gradient_resume_lineage
                ),
                "training_plots": plots,
                "checkpoint_reload_verified": True,
                "checkpoint_reload_verification": (
                    checkpoint_reload_verification
                ),
                "checkpoint_layout": "active_latest_only",
            }
        )
        archive.write_json(
            "diagnostics/operational_review_required_epoch_0300.json",
            review,
        )
        archive.write_json(
            "provenance/untied8_operational_review.json",
            review,
        )
        archive.append_metric_event(
            {
                "name": "fit/operational_review/plateau_confirmed",
                "value": 0.0,
                "step": UNTIED8_OPERATIONAL_REVIEW_EPOCH,
                "phase": "operational_review_required",
            }
        )
        archive.write_summary(review)
        print(
            STDOUT_PREFIX + " " + json.dumps(review, sort_keys=True, allow_nan=False),
            flush=True,
        )
        return review

    start_epoch = 0 if resume is None else resume.completed_global_epochs
    final_resume: CohortRelativeQKVEpochBoundaryResume | None = None
    plateau: dict[str, Any] | None = None
    completion: dict[str, Any] | None = None
    source_plateau: dict[str, Any] | None = None
    strict_plateau_diagnostic: dict[str, Any] | None = None

    if fixed_mode:
        assert resume is not None and resume_receipt is not None
        fixed_plan = _fixed_continuation_plan(config, resume)
        _require_equal(
            start_epoch,
            fixed_plan["segment_start_global_epoch"],
            field="fixed_continuation.start_epoch",
        )
        raw_source_plateau = resume_receipt.get("source_plateau")
        if not isinstance(raw_source_plateau, Mapping):
            raise SO214CoreRunnerError(
                "Fixed continuation lineage lacks source plateau provenance."
            )
        source_plateau = dict(raw_source_plateau)
        segment_result = fit_cohort_relative_qkv_segment(
            model,
            batches,
            _training_config(
                config,
                start_epoch=start_epoch,
                end_epoch=int(fixed_plan["segment_end_global_epoch"]),
                rank=rank,
                local_rank=local_rank,
            ),
            resume=resume,
            # Deliberately no recovery checkpoint is serialized from epochs
            # 176 through 299. A failed attempt restarts from the immutable
            # epoch-175 source checkpoint.
            checkpoint_callback=None,
            epoch_callback=epoch_callback if rank == 0 else None,
            gradient_observer=(
                observe_gradient_direction
                if rank == 0 and gradient_direction_enabled
                else None
            ),
        )
        final_resume = segment_result.resume
        _require_equal(
            final_resume.completed_global_epochs,
            FIXED_CONTINUATION_FINAL_EPOCH,
            field="fixed_continuation.final_epoch",
        )
        _require_equal(
            final_resume.optimizer_updates_completed,
            FIXED_CONTINUATION_FINAL_EPOCH * 7,
            field="fixed_continuation.optimizer_updates_completed",
        )
        losses = [
            float(record.equal_core_mean_masked_huber)
            for record in final_resume.global_history
        ]
        strict_plateau_diagnostic = _strict_plateau_payload(
            losses,
            trainer=trainer,
        )
        completion = _fixed_completion_payload(
            final_resume,
            source_checkpoint_sha256=str(
                resume_receipt["source_checkpoint_sha256"]
            ),
        )
        if rank == 0:
            assert archive is not None
            _write_or_verify_strict_plateau_audit(
                archive,
                strict_plateau_diagnostic,
            )
    else:
        first_audit_epoch = int(trainer["plateau_first_audit_epoch"])
        continuation = int(trainer["continuation_block_global_epochs"])

        # A crash may happen after the epoch checkpoint and plateau audit but
        # before finalization. Recompute any durable boundary decision before
        # taking even one additional optimizer step.
        resume_decision = (
            None
            if resume is None
            else _resume_plateau_decision(
                observed_losses,
                completed_global_epochs=start_epoch,
                first_audit_epoch=first_audit_epoch,
                audit_interval=continuation,
            )
        )
        if resume_decision is not None:
            plateau = _plateau_payload(
                resume_decision,
                audit_source="resume_checkpoint_re_evaluation",
            )
            if rank == 0:
                assert archive is not None
                archive.write_json(
                    f"diagnostics/plateau_audit_epoch_{start_epoch:04d}.json",
                    plateau,
                )
            resume_should_stop = torch.tensor(
                [1 if resume_decision.should_stop else 0],
                device=torch.device(f"cuda:{local_rank}"),
                dtype=torch.int32,
            )
            torch.distributed.broadcast(resume_should_stop, src=0)
            if int(resume_should_stop.item()) == 1:
                assert resume is not None
                model.load_state_dict(resume.model_state_dict, strict=True)
                model.to(torch.device(f"cuda:{local_rank}"))
                final_resume = resume
            elif _untied8_operational_review_required(config, resume_decision):
                assert resume is not None and plateau is not None
                if rank == 0:
                    persist_untied8_operational_review(resume, plateau)
                torch.distributed.barrier()
                raise SO214CoreOperationalReviewRequired(
                    "operational_review_required: untied-8 plateau was not "
                    "confirmed at the durable epoch-300 boundary; resumption "
                    "requires an explicit campaign amendment and approval."
                )

        end_epoch = _next_fixed_plateau_audit_epoch(
            start_epoch,
            first_audit_epoch=first_audit_epoch,
            audit_interval=continuation,
        )
        while final_resume is None:
            segment_result = fit_cohort_relative_qkv_segment(
                model,
                batches,
                _training_config(
                    config,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    rank=rank,
                    local_rank=local_rank,
                ),
                resume=resume,
                checkpoint_callback=checkpoint_callback if rank == 0 else None,
                epoch_callback=epoch_callback if rank == 0 else None,
                gradient_observer=(
                    observe_gradient_direction
                    if rank == 0 and gradient_direction_enabled
                    else None
                ),
            )
            losses = [
                record.equal_core_mean_masked_huber
                for record in segment_result.global_history
            ]
            decision = single_seed_plateau_decision(
                losses,
                model_seed=MODEL_SEED,
                completed_global_epochs=end_epoch,
            )
            plateau = _plateau_payload(
                decision,
                audit_source="completed_training_segment",
            )
            if rank == 0:
                assert archive is not None
                archive.write_json(
                    f"diagnostics/plateau_audit_epoch_{end_epoch:04d}.json",
                    plateau,
                )
            should_stop = torch.tensor(
                [1 if decision.should_stop else 0],
                device=torch.device(f"cuda:{local_rank}"),
                dtype=torch.int32,
            )
            torch.distributed.broadcast(should_stop, src=0)
            if int(should_stop.item()) == 1:
                final_resume = segment_result.resume
                break
            if _untied8_operational_review_required(config, decision):
                assert plateau is not None
                if rank == 0:
                    persist_untied8_operational_review(
                        segment_result.resume,
                        plateau,
                    )
                torch.distributed.barrier()
                raise SO214CoreOperationalReviewRequired(
                    "operational_review_required: untied-8 plateau was not "
                    "confirmed at the durable epoch-300 boundary; resumption "
                    "requires an explicit campaign amendment and approval."
                )
            resume = segment_result.resume
            start_epoch = end_epoch
            end_epoch = _next_fixed_plateau_audit_epoch(
                start_epoch,
                first_audit_epoch=first_audit_epoch,
                audit_interval=continuation,
            )

    assert final_resume is not None
    if not fixed_mode and plateau is None:
        raise SO214CoreRunnerError("Legacy training lacks its plateau decision.")
    if rank == 0:
        assert checkpoint_store is not None
        final_payload = _checkpoint_payload(
            run_id=run_id,
            config=config,
            model_construction=construction,
            parameter_count=parameter_count,
            resume=final_resume,
            plateau=plateau,
            completion=completion,
            source_plateau=source_plateau,
            strict_plateau_diagnostic=strict_plateau_diagnostic,
        )
        if fixed_mode:
            # The first and only checkpoint write in the continuation attempt
            # is the completed epoch-300 payload. It is immediately renamed
            # from the atomic staging name to the sole final last.ckpt.
            checkpoint_store.save(
                final_payload,
                completed_global_epochs=FIXED_CONTINUATION_FINAL_EPOCH,
            )
            final_receipt = checkpoint_store.finalize()
        else:
            final_receipt = checkpoint_store.finalize(
                final_payload,
                completed_global_epochs=final_resume.completed_global_epochs,
            )
        require_latest_only_checkpoint_layout(scratch_path, final=True)
    else:
        final_receipt = None

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if rank != 0:
        return None

    assert archive is not None and final_receipt is not None
    epoch_metric_rows = DurableEpochMetricsCSV(archive.scratch_path).read_rows()
    _require_equal(
        len(epoch_metric_rows),
        final_resume.completed_global_epochs,
        field="epoch_metrics.completed_rows",
    )
    gradient_direction_provenance: dict[str, Any] = {
        "enabled": False,
        "legacy_protocol_preserved": True,
    }
    block_gradient_direction_provenance: dict[str, Any] = {
        "enabled": False,
        "non_untied8_protocol_preserved": True,
    }
    if gradient_direction_enabled:
        assert gradient_writer is not None
        gradient_rows = gradient_writer.read_rows()
        _require_equal(
            len(gradient_rows),
            final_resume.completed_global_epochs,
            field="gradient_direction.completed_rows",
        )
        _require_equal(
            int(gradient_rows[-1]["global_epoch"]),
            final_resume.completed_global_epochs,
            field="gradient_direction.final_epoch",
        )
        gradient_direction_provenance = {
            "enabled": True,
            "schema": GRADIENT_DIRECTION_METRICS_SCHEMA,
            "csv": "results/gradient_direction_metrics.csv",
            "csv_sha256": sha256_file(gradient_writer.path),
            "completed_epoch_rows": len(gradient_rows),
            "final_epoch_summary": dict(gradient_rows[-1]),
            "sampling_point": (
                "rank0_full_ddp_averaged_fp32_trainable_gradient_after_amp_"
                "unscale_and_finite_check_before_clipping"
            ),
            "gradient_tensors_serialized": False,
            "checkpoint_state_saved": False,
            "interpretation": (
                "directional_diagnostics_are_interpreted_jointly_with_training_"
                "loss_and_gradient_norm_and_are_not_a_standalone_convergence_"
                "criterion"
            ),
        }
    if block_gradient_direction_enabled:
        assert block_gradient_writer is not None
        block_gradient_rows = block_gradient_writer.read_rows()
        _require_equal(
            len(block_gradient_rows),
            final_resume.completed_global_epochs * block_gradient_count,
            field="block_gradient_direction.completed_rows",
        )
        _require_equal(
            block_gradient_writer.completed_epochs,
            final_resume.completed_global_epochs,
            field="block_gradient_direction.completed_epochs",
        )
        final_block_rows = tuple(block_gradient_rows[-block_gradient_count:])
        _require_equal(
            tuple(int(row["block_index"]) for row in final_block_rows),
            tuple(range(block_gradient_count)),
            field="block_gradient_direction.final_block_indices",
        )
        _require_equal(
            tuple(row["block_name"] for row in final_block_rows),
            tuple(
                f"blocks.{index}" for index in range(block_gradient_count)
            ),
            field="block_gradient_direction.final_block_names",
        )
        _require_equal(
            {int(row["global_epoch"]) for row in final_block_rows},
            {final_resume.completed_global_epochs},
            field="block_gradient_direction.final_epoch",
        )
        block_gradient_direction_provenance = {
            "enabled": True,
            "scope": "graph_blocks_only",
            "schema": BLOCK_GRADIENT_DIRECTION_METRICS_SCHEMA,
            "csv": "results/gradient_direction_by_block.csv",
            "csv_sha256": sha256_file(block_gradient_writer.path),
            "completed_epoch_rows": len(block_gradient_rows),
            "completed_epochs": block_gradient_writer.completed_epochs,
            "rows_per_completed_global_epoch": block_gradient_count,
            "block_names": [
                f"blocks.{index}" for index in range(block_gradient_count)
            ],
            "final_epoch_summaries": [dict(row) for row in final_block_rows],
            "sampling_point": (
                "rank0_per_block_ddp_averaged_fp32_trainable_gradient_after_"
                "amp_unscale_and_finite_check_before_clipping"
            ),
            "gradient_tensors_serialized": False,
            "checkpoint_state_saved": False,
            "interpretation": (
                "block_direction_diagnostics_are_interpreted_jointly_with_"
                "training_loss_full_gradient_direction_and_gradient_norm"
            ),
        }
    if gradient_direction_enabled:
        assert gradient_writer is not None
        _validate_gradient_direction_scalar_files(
            archive.scratch_path,
            gradient_csv=gradient_writer.path,
            block_gradient_csv=(
                block_gradient_writer.path
                if block_gradient_writer is not None
                else None
            ),
        )
        if any(
            str(field).startswith(("gradient_direction", "block_gradient"))
            for field in final_payload
        ):
            raise SO214CoreRunnerError(
                "Gradient direction tensors may not enter the checkpoint payload."
            )
    training_plots: dict[str, Any] = {
        "enabled": False,
        "non_untied8_protocol_preserved": True,
    }
    if block_gradient_direction_enabled:
        training_plots = _write_and_verify_untied8_training_plots(
            archive.scratch_path,
            expected_blocks=block_gradient_count,
        )
    replay_batch = min(batches, key=lambda batch: (batch.n_nodes, batch.alias))
    checkpoint_verification = _verify_final_checkpoint_reload(
        final_receipt.path,
        expected_file_sha256=final_receipt.sha256,
        config=config,
        model_construction=construction,
        parameter_count=parameter_count,
        expected_resume=final_resume,
        expected_plateau=plateau,
        expected_completion=completion,
        expected_source_plateau=source_plateau,
        expected_strict_plateau_diagnostic=strict_plateau_diagnostic,
        in_memory_model=model,
        replay_batch=replay_batch,
        device=torch.device(f"cuda:{local_rank}"),
    )
    archive.write_json(
        "diagnostics/final_checkpoint_reload_verification.json",
        checkpoint_verification,
    )

    evaluation = _section(config, "evaluation")
    held_in, core_diagnostics, prediction_rows = _held_in_diagnostics(
        model,
        batches,
        amp=bool(trainer["amp"]),
        amp_dtype=str(trainer["amp_dtype"]),
    )
    salt = os.environ.get("BAGM_SAMPLE_KEY_SALT", "").strip()
    if len(salt.encode("utf-8")) < 16:
        salt = hashlib.sha256(
            f"{run_id}:{_campaign_id(config)}".encode("utf-8")
        ).hexdigest()
    deidentified = deidentify_prediction_rows(
        prediction_rows,
        identifier_fields=["core_alias"],
        salt=salt,
        namespace="so2-14core-held-in-alias",
    )
    for row in deidentified:
        row["run_id"] = run_id
    archive.write_predictions("fit", deidentified)
    archive.write_table(
        "metrics/history",
        [
            {"run_id": run_id, "split": "fit", **asdict(record)}
            for record in final_resume.global_history
        ],
    )
    archive.write_table(
        "metrics/core_steps",
        [
            {
                "run_id": run_id,
                "split": "fit",
                **{
                    key: value
                    for key, value in asdict(record).items()
                    if key != "mask_views"
                },
                "mask_view_checksums": [
                    view.mask_checksum_sha256 for view in record.mask_views
                ],
            }
            for record in final_resume.core_history
        ],
    )
    archive.write_table(
        "metrics/optimizer_updates",
        [
            {"run_id": run_id, "split": "fit", **asdict(record)}
            for record in final_resume.optimizer_update_history
        ],
    )
    archive.write_json("diagnostics/held_in_fit_metrics_by_core.json", core_diagnostics)
    final_metrics = {
        **held_in,
        "fit/training/final_equal_core_masked_huber": float(
            final_resume.global_history[-1].equal_core_mean_masked_huber
        ),
        "fit/training/final_global_epoch": float(
            final_resume.completed_global_epochs
        ),
    }
    archive.write_json("metrics/final.json", final_metrics)
    for name, value in final_metrics.items():
        archive.append_metric_event(
            {
                "name": name,
                "value": value,
                "step": final_resume.completed_global_epochs,
            }
        )
    archive.write_json(
        "provenance/so2_relative_qkv_training.json",
        {
            "campaign_id": _campaign_id(config),
            "model_seed": MODEL_SEED,
            "mask_base_seed": int(_section(config, "masking")["mask_base_seed"]),
            "held_in_mask_base_seed": HELD_IN_MASK_BASE_SEED,
            "parameter_count": parameter_count,
            "model_construction": construction,
            "completed_global_epochs": final_resume.completed_global_epochs,
            "optimizer_updates_completed": final_resume.optimizer_updates_completed,
            "cores_per_optimizer_update": final_resume.cores_per_optimizer_update,
            "mask_views_per_core": final_resume.mask_views_per_core,
            "state_dict_sha256": final_resume.model_state_checksum,
            "history_sha256": final_resume.history_checksum,
            "plateau": plateau,
            "source_plateau": source_plateau,
            "strict_plateau_diagnostic": strict_plateau_diagnostic,
            "completion": completion,
            "cohort_manifest_sha256": sha256_file(cohort_dir / "manifest.json"),
            "graph_manifest_sha256": sha256_file(graph_dir / "manifest.json"),
            "checkpoint_sha256": final_receipt.sha256,
            "checkpoint_reload_verified": bool(
                checkpoint_verification["verified"]
            ),
            "checkpoint_reload_verification": checkpoint_verification,
            "checkpoint_layout": "final_last_only",
            "distributed_world_size": WORLD_SIZE,
            "distributed_backend": "nccl",
            "physical_gpu_ids": [0, 1, 2, 3],
            "elastic_max_restarts": 0,
            "resume_source": resume_receipt,
            "execution_mode": (
                "resume_fixed_final_epoch" if fixed_mode else "plateau_continuation"
            ),
            "epoch_loss_recording": {
                "frequency": "every_completed_global_epoch",
                "csv": "results/epoch_metrics.csv",
                "completed_epoch_rows": len(epoch_metric_rows),
            },
            "gradient_direction_observability": gradient_direction_provenance,
            "gradient_direction_resume_lineage": gradient_resume_lineage,
            "block_gradient_direction_observability": (
                block_gradient_direction_provenance
            ),
            "block_gradient_direction_resume_lineage": (
                block_gradient_resume_lineage
            ),
            "training_plots": training_plots,
        },
    )
    summary = {
        "run_id": run_id,
        "status": "success",
        "campaign_id": _campaign_id(config),
        "model_name": str(_section(config, "model")["name"]),
        "model_seed": MODEL_SEED,
        "final_epoch": final_resume.completed_global_epochs,
        "optimizer_steps": final_resume.optimizer_updates_completed,
        "parameter_count": parameter_count,
        "primary_metric_name": evaluation["primary_metric"],
        "primary_metric_value": held_in[evaluation["primary_metric"]],
        "peak_vram_gib": max(
            float(row["peak_vram_gib_all_ranks"]) for row in epoch_metric_rows
        ),
        "duration_seconds": time.monotonic() - started,
        "checkpoint": "checkpoints/last.ckpt",
        "epoch_metrics_csv": "results/epoch_metrics.csv",
        "epoch_loss_recorded_every_global_epoch": True,
        "epoch_metrics_rows": len(epoch_metric_rows),
        "gradient_direction_observability": gradient_direction_provenance,
        "block_gradient_direction_observability": (
            block_gradient_direction_provenance
        ),
        "training_plots": training_plots,
        "plateau_confirmed": bool(not fixed_mode),
        "source_plateau_confirmed_epoch": (
            FIXED_CONTINUATION_SOURCE_EPOCH if fixed_mode else None
        ),
        "strict_plateau_diagnostic_confirmed": (
            bool(strict_plateau_diagnostic["diagnostic_confirmed"])
            if strict_plateau_diagnostic is not None
            else None
        ),
        "plateau_stopping_used": bool(not fixed_mode),
        "fixed_epoch_target_completed": bool(
            fixed_mode
            and final_resume.completed_global_epochs
            == FIXED_CONTINUATION_FINAL_EPOCH
        ),
        "checkpoint_reload_verified": bool(checkpoint_verification["verified"]),
        "world_size": WORLD_SIZE,
        "generalization_estimate": False,
    }
    archive.write_summary(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SO2 cores 15--28 with four-rank Relative-QKV DDP."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, local_rank, _ = _distributed_identity()
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=30),
    )
    try:
        summary = run_distributed(args, rank=rank, local_rank=local_rank)
        if rank == 0 and summary is not None:
            print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
        return 0
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
