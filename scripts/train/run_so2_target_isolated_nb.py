#!/usr/bin/env python3
"""Train the frozen SO2 target-isolated geometry-modulated NB2 campaign.

This entry point is intentionally executable only through a single-node,
four-rank ``torchrun`` launch.  Every rank processes a contiguous target-cell
shard of every core.  Two complete cores form one optimizer update: the first
backward executes under :meth:`DistributedDataParallel.no_sync`, the second
performs the one DDP reduction.  Each local summed NB2 likelihood is scaled by
``WORLD_SIZE / globally_reduced_masked_entry_count`` so the reduced gradient is
the exact masked-entry-pooled objective across both cores and all four ranks.

The clean full-core expression tensor is used only for the immutable neighbor
key/value bank.  A distinct local target tensor is copied and explicitly
zeroed at the withheld positions before every forward.  Edges and relative
geometry remain receiver-major CPU tensors; only expression, raw targets, and
allowed covariates are staged on the rank's GPU.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from scripts.train.run_so2_geometry_modulated_nb import (  # noqa: E402
    _RawThetaParameterView,
    _assert_finite_gradients,
    _broadcast_rank_zero_result,
    _canonical_sha256,
    _checkpoint_payload_from_state,
    _current_gpu_identity,
    _load_data,
    _load_json_mapping,
    _read_runner_table,
    _reconcile_checkpoint_transaction,
    _runtime_path,
    _section,
    _synchronized_rank_zero_phase,
    _theta_summary,
    _tree_sha256,
)
from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.gradient_direction_observability import (  # noqa: E402
    BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
    GRADIENT_DIRECTION_METRICS_COLUMNS,
    BlockGradientDirectionTracker,
    FullGradientDirectionTracker,
    GradientDirectionUpdateContext,
)
from spatial_benchmark.negative_binomial import (  # noqa: E402
    NegativeBinomialModelOutput,
    masked_negative_binomial_metrics,
    masked_negative_binomial_nll_sum,
)
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training import (  # noqa: E402
    _clear_graph_layout_caches,
    _clone_tree_to_cpu,
)
from spatial_benchmark.pooled_relative_qkv_training_v2 import (  # noqa: E402
    cohort_relative_qkv_core_order,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    validate_prediction_rows,
)
from spatial_benchmark.so2_nb_data import (  # noqa: E402
    SO2NBDataBundle,
    SO2NBCoreBatch,
    SO2_NB_TEST_ALIASES,
    SO2_NB_TRAINING_ALIASES,
    SO2_NB_VALIDATION_ALIASES,
)
from spatial_benchmark.so2_nb_training import (  # noqa: E402
    AtomicBestLatestCheckpointStore,
    DurableScalarCSV,
    EarlyStoppingState,
    SO2NBTrainingError,
    capture_rng_state,
    restore_rng_state,
    sha256_file,
    update_early_stopping,
)
from spatial_benchmark.target_cell_masking import (  # noqa: E402
    make_contiguous_target_partition,
    make_contiguous_target_partitions,
    make_training_target_cell_masks,
    make_validation_target_cell_masks,
    target_partition_coverage_receipt,
    verify_target_cell_receipt,
)
from spatial_benchmark.target_isolated_negative_binomial import (  # noqa: E402
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
)
from spatial_benchmark.training import (  # noqa: E402
    _autocast_context,
    _make_grad_scaler,
    set_deterministic_seed,
)


CAMPAIGN_ID = (
    "cmp_20260926_so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0"
)
MODEL_NAME = "target-isolated-geometry-modulated-relative-qkv-gat-nb2"
MODEL_FAMILY = (
    "target_isolated_geometry_modulated_relative_qkv_graph_transformer"
)
EVALUATION_PROTOCOL = (
    "donor_grouped_so2_target_isolated_nb_once_per_cell_earlystop_v1"
)
PROTOCOL = EVALUATION_PROTOCOL
PREFLIGHT_SCHEMA = "so2_target_isolated_nb_once_per_cell_ddp4_preflight_v1"
CHECKPOINT_SCHEMA = "so2_target_isolated_nb_once_per_cell_checkpoint_v1"
EPOCH_METRICS_SCHEMA = "so2_target_isolated_nb_once_per_cell_epoch_metrics_v1"
PER_CORE_METRICS_SCHEMA = (
    "so2_target_isolated_nb_once_per_cell_per_core_metrics_v1"
)

WORLD_SIZE = 4
VISIBLE_DEVICES = "0,1,2,3"
MODEL_SEED = 0
MASK_BASE_SEED = 2_026_092_601
CORE_ORDER_SEED = 2_026_092_602
EXPECTED_PARAMETER_COUNT = 5_135_088
MAXIMUM_EPOCHS = 300
MINIMUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-4
OPTIMIZER_UPDATES_PER_EPOCH = 6
MINIMUM_FREE_DISK_GIB = 20.0
FROZEN_TASK_CONTRACT_SHA256 = (
    "dfaf10f1e828dffc1369629ae41dce133a42d0686aeeedba30a9d97c5b296bbc"
)
SOURCE_EXPERIMENT_CONFIG = Path(
    "configs/experiment/"
    "so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0.yaml"
)
FROZEN_TASK_CONTRACT = Path(
    "experiments/campaigns/"
    "cmp_20260926_so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0/"
    "frozen_task_contract.yaml"
)
PREFLIGHT_CODE_HASH_SCOPE = (
    "target_isolated_model_masking_data_training_preflight_runner_and_"
    "configuration_launch_contract"
)
PREFLIGHT_CODE_RELATIVE_PATHS = (
    Path("scripts/diagnostics/preflight_so2_target_isolated_nb_ddp.py"),
    Path("scripts/train/run_so2_target_isolated_nb.py"),
    Path("scripts/train/run_so2_geometry_modulated_nb.py"),
    Path("src/spatial_benchmark/target_isolated_negative_binomial.py"),
    Path("src/spatial_benchmark/target_cell_masking.py"),
    Path("src/spatial_benchmark/target_cell_training.py"),
    Path("src/spatial_benchmark/negative_binomial.py"),
    Path(
        "src/spatial_benchmark/"
        "geometry_modulated_relative_qkv_graph_transformer.py"
    ),
    Path("src/spatial_benchmark/relative_qkv_graph_transformer.py"),
    Path("src/spatial_benchmark/models.py"),
    Path("src/spatial_benchmark/so2_nb_data.py"),
    Path("src/spatial_benchmark/so2_nb_training.py"),
    Path("src/spatial_benchmark/gradient_direction_observability.py"),
    Path("src/spatial_benchmark/pooled_relative_qkv_training.py"),
    Path("src/spatial_benchmark/pooled_relative_qkv_training_v2.py"),
    Path("src/spatial_benchmark/run_archive.py"),
    Path("src/spatial_benchmark/training.py"),
    Path("src/spatial_benchmark/queueing.py"),
    Path("src/spatial_benchmark/configuration.py"),
    Path("src/spatial_benchmark/paths.py"),
    Path("src/spatial_benchmark/identifiers.py"),
)

TRAINING_ALIASES = tuple(SO2_NB_TRAINING_ALIASES)
VALIDATION_ALIASES = tuple(SO2_NB_VALIDATION_ALIASES)
VALIDATION_MASK_NAMESPACE = (
    "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"
)
TRAINING_MASK_NAMESPACE = "bagm.so2.target_isolated_nb.training_masks.v1"
PRIMARY_METRIC = "val/target_only_masked_negative_binomial_nll"

PREFLIGHT_REQUIRED_GATES = frozenset(
    {
        "configuration_and_contract",
        "model_topology",
        "shared_encoder_two_stream",
        "synthetic_target_coverage",
        "target_mask_contract",
        "neighbor_context_unmasked",
        "incoming_nonself_routing",
        "masked_target_value_invariance",
        "raw_target_forward_isolation",
        "no_return_leakage",
        "unequal_count_ddp_weighting",
        "fixed_validation_mask_replay",
        "fixed_validation_metric_replay",
        "full_constant_fp32_nb2",
        "amp_fp32_equivalence",
        "real_train_shard_forward_backward",
        "preclip_gradient_flow",
        "bounded_nb2_loss_decrease",
        "checkpoint_roundtrip",
        "ddp_parameter_sync",
        "gpu_inventory",
        "peak_vram",
        "disk_space",
        "no_test_split_or_artifacts",
    }
)

EPOCH_COLUMNS = (
    "schema",
    "global_epoch",
    "train_pooled_masked_negative_binomial_nll",
    "train_masked_entries",
    "train_target_cells",
    "validation_pooled_masked_negative_binomial_nll",
    "validation_masked_entries",
    "validation_target_cells",
    "validation_masked_raw_count_mae",
    "validation_masked_raw_count_rmse",
    "validation_masked_log1p_mae",
    "validation_masked_log1p_rmse",
    "validation_masked_poisson_deviance",
    "validation_observed_zero_rate",
    "validation_predicted_zero_probability_mean",
    "validation_zero_brier_score",
    "inverse_dispersion_min",
    "inverse_dispersion_median",
    "inverse_dispersion_max",
    "learning_rate",
    "epoch_duration_seconds",
    "training_targets_per_second",
    "peak_vram_gib_all_ranks",
    "best_validation_metric",
    "best_epoch",
    "patience_control_best_validation_metric",
    "patience_control_best_epoch",
    "bad_validations",
    "improved",
    "patience_control_improved",
    "should_stop",
    "stop_reason",
    "core_order_json",
    "training_coverage_receipts_json",
    "validation_coverage_receipts_json",
    "global_gradient_json",
    "block_gradients_json",
    "dispersion_gradient_json",
)

PER_CORE_COLUMNS = (
    "schema",
    "global_epoch",
    "split",
    "core_alias",
    "target_cells",
    "masked_entries",
    "pooled_masked_negative_binomial_nll",
    "masked_raw_count_mae",
    "masked_raw_count_rmse",
    "masked_log1p_mae",
    "masked_log1p_rmse",
    "masked_poisson_deviance",
    "observed_zero_rate",
    "predicted_zero_probability_mean",
    "zero_brier_score",
    "coverage_receipt_sha256",
)

_VALIDATION_SUM_FIELDS = (
    "masked_entries",
    "negative_binomial_nll_sum",
    "raw_absolute_error_sum",
    "raw_squared_error_sum",
    "log1p_absolute_error_sum",
    "log1p_squared_error_sum",
    "poisson_deviance_sum",
    "observed_zero_sum",
    "predicted_zero_probability_sum",
    "zero_brier_sum",
    "observed_count_sum",
    "predicted_count_sum",
    "target_cells",
)


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise SO2NBTrainingError(
            f"{field} must be {expected!r}; received {actual!r}."
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_field(value: object) -> str:
    return _canonical_json(value)


def _update_exact_checkpoint_selection(
    state: EarlyStoppingState,
    validation_value: float,
    *,
    completed_epoch: int,
) -> EarlyStoppingState:
    """Track the literal lowest validation NLL independently of patience.

    ``EarlyStoppingState`` is reused because the generalized checkpoint store
    validates that structure.  Its bad-validation/stop fields are deliberately
    inert here; the separate patience-control state owns the 1e-4 threshold and
    stopping decision.
    """

    epoch = int(completed_epoch)
    value = float(validation_value)
    if epoch != state.completed_epoch + 1:
        raise SO2NBTrainingError("Checkpoint-selection epochs must be contiguous.")
    if not math.isfinite(value):
        raise SO2NBTrainingError("Checkpoint-selection metric must be finite.")
    improved = state.best_value is None or value < state.best_value
    return EarlyStoppingState(
        best_value=value if improved else state.best_value,
        best_epoch=epoch if improved else state.best_epoch,
        bad_validations=0,
        completed_epoch=epoch,
        improved=improved,
        should_stop=False,
        stop_reason=None,
    )


def _configuration_source_file_hashes(paths: ProjectPaths) -> Mapping[str, str]:
    config_path = (paths.project_root / SOURCE_EXPERIMENT_CONFIG).resolve(strict=True)
    root = load_yaml_mapping(config_path)
    defaults = root.get("defaults")
    if not isinstance(defaults, Sequence) or isinstance(defaults, (str, bytes)):
        raise SO2NBTrainingError("Source experiment defaults are malformed.")
    files = {config_path}
    for entry in defaults:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise SO2NBTrainingError("Source experiment default entry is malformed.")
        group, name = next(iter(entry.items()))
        files.add(
            (paths.config_root / str(group) / f"{name}.yaml").resolve(strict=True)
        )
    return {
        str(path.relative_to(paths.project_root)): sha256_file(path)
        for path in sorted(files)
    }


def _overlay_source_artifacts(bundle: SO2NBDataBundle) -> Mapping[str, Any]:
    try:
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SO2NBTrainingError("Overlay manifest became unreadable.") from exc
    if not isinstance(manifest, Mapping):
        raise SO2NBTrainingError("Overlay manifest must be a mapping.")
    unsigned = dict(manifest)
    stored = unsigned.pop("manifest_content_sha256", None)
    _require_equal(
        stored,
        _canonical_sha256(unsigned),
        field="overlay.manifest_content_sha256",
    )
    _require_equal(
        stored,
        bundle.manifest_content_sha256,
        field="overlay.bundle_manifest_content_sha256",
    )
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or not source_artifacts:
        raise SO2NBTrainingError("Overlay manifest lacks source-artifact evidence.")
    return dict(source_artifacts)


def validate_target_isolated_config(config: Mapping[str, Any]) -> None:
    """Reassert every runner-critical value after general composition checks."""

    validate_experiment_config(config)
    _require_equal(config.get("seed"), MODEL_SEED, field="seed")
    campaign = _section(config, "campaign")
    _require_equal(campaign.get("campaign_id"), CAMPAIGN_ID, field="campaign_id")
    metadata = _section(config, "metadata")
    _require_equal(
        metadata.get("frozen_task_contract_sha256"),
        FROZEN_TASK_CONTRACT_SHA256,
        field="metadata.frozen_task_contract_sha256",
    )

    model = _section(config, "model")
    expected_model = {
        "name": MODEL_NAME,
        "family": MODEL_FAMILY,
        "hidden_dim": 256,
        "graph_layers": 4,
        "unique_graph_blocks": 4,
        "attention_heads": 8,
        "attention_head_dim": 32,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "relative_geometry_dim": 70,
        "geometry_hidden_dim": 128,
        "routing_architecture": "two_stream_target_isolated_v1",
        "neighbor_key_value_state_evolves_across_blocks": False,
        "incoming_nonself_edges_only": True,
        "target_self_context_edge": False,
        "relative_geometry_value_injection": False,
        "output_distribution": "negative_binomial_nb2",
        "expected_trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
    }
    for field, expected in expected_model.items():
        _require_equal(model.get(field), expected, field=f"model.{field}")

    masking = _section(config, "masking")
    for field, expected in {
        "type": "target_only_uniform_per_cell_integer_count",
        "count_min": 1,
        "count_max": 1000,
        "training_mask_realizations_per_target_per_global_epoch": 1,
        "target_visit_count_per_global_epoch": 1,
        "target_cell_only": True,
        "neighbor_cells_artificially_masked": False,
        "neighbor_context_rows_fully_observed": True,
        "mask_base_seed": MASK_BASE_SEED,
        "training_mask_seed_namespace": TRAINING_MASK_NAMESPACE,
    }.items():
        _require_equal(masking.get(field), expected, field=f"masking.{field}")
    validation_masks = masking.get("validation_masks")
    if not isinstance(validation_masks, Mapping):
        raise SO2NBTrainingError("masking.validation_masks must be a mapping.")
    _require_equal(
        validation_masks.get("seed_namespace"),
        VALIDATION_MASK_NAMESPACE,
        field="masking.validation_masks.seed_namespace",
    )

    dataset = _section(config, "dataset")
    for field, expected in {
        "training_core_aliases": list(TRAINING_ALIASES),
        "validation_core_aliases": list(VALIDATION_ALIASES),
        "test_core_aliases": [],
        "training_cells": 208_696,
        "validation_cells": 37_367,
        "test_cells": 0,
        "biological_target_count": 1000,
        "node_covariate_count": 22,
        "test_partition_present": False,
    }.items():
        _require_equal(dataset.get(field), expected, field=f"dataset.{field}")

    evaluation = _section(config, "evaluation")
    _require_equal(
        evaluation.get("protocol"), EVALUATION_PROTOCOL, field="evaluation.protocol"
    )
    _require_equal(
        evaluation.get("primary_metric"), PRIMARY_METRIC, field="evaluation.primary"
    )
    _require_equal(evaluation.get("splits"), ["validation"], field="evaluation.splits")
    _require_equal(
        evaluation.get("test_split_present"), False, field="evaluation.test"
    )

    trainer = _section(config, "trainer")
    for field, expected in {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "gradient_clip_norm": 1.0,
        "cores_per_optimizer_update": 2,
        "optimizer_updates_per_global_epoch": OPTIMIZER_UPDATES_PER_EPOCH,
        "target_shards_per_core": WORLD_SIZE,
        "max_epochs": MAXIMUM_EPOCHS,
        "minimum_global_epochs": MINIMUM_EPOCHS,
        "scheduler": "reduce_lr_on_plateau",
        "scheduler_monitor": PRIMARY_METRIC,
        "scheduler_factor": 0.5,
        "scheduler_patience": 8,
        "scheduler_threshold_mode": "abs",
        "scheduler_threshold": 1e-4,
        "scheduler_min_learning_rate": 1e-6,
        "early_stopping": True,
        "early_stopping_monitor": PRIMARY_METRIC,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "validation_every": 1,
        "amp": True,
        "likelihood_compute_dtype": "float32",
        "likelihood_outside_autocast": True,
        "restore_best": True,
        "distributed_world_size": WORLD_SIZE,
        "cores_staged_sequentially_per_optimizer_update": True,
        "ddp_sync_policy": (
            "first_core_backward_under_no_sync_second_core_backward_synced"
        ),
        "stage_complete_core_graph_on_device": False,
        "staged_relative_geometry_dtype": "float32",
        "core_order_seed": CORE_ORDER_SEED,
    }.items():
        _require_equal(trainer.get(field), expected, field=f"trainer.{field}")

    launcher = _section(config, "launcher")
    for field, expected in {
        "process_count": WORLD_SIZE,
        "requested_gpu_count": WORLD_SIZE,
        "require_exact_visible_devices": VISIBLE_DEVICES,
        "distributed_backend": "nccl",
        "elastic_max_restarts": 0,
        "hardware_preflight_receipt": (
            "state/preflight/so2_target_isolated_nb_once_per_cell_ddp4.json"
        ),
        "disk_safety_min_free_gb": 20,
    }.items():
        _require_equal(launcher.get(field), expected, field=f"launcher.{field}")


def _distributed_identity() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO2NBTrainingError("This runner must be invoked by torchrun.") from exc
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE):
        raise SO2NBTrainingError("Exactly four torchrun ranks are required.")
    if local_rank != rank:
        raise SO2NBTrainingError("Single-node RANK and LOCAL_RANK must match.")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != VISIBLE_DEVICES:
        raise SO2NBTrainingError(
            f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE_DEVICES}."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise SO2NBTrainingError("Exactly four visible CUDA devices are required.")
    return rank, local_rank, world_size


def _load_worker_config(
    args: argparse.Namespace,
) -> tuple[str, Path, dict[str, Any], ProjectPaths]:
    paths = current_paths()
    run_id = os.environ.get("BAGM_RUN_ID", "").strip()
    environment_scratch = os.environ.get("BAGM_RUN_SCRATCH", "").strip()
    if not run_id or not environment_scratch or args.run_scratch is None:
        raise SO2NBTrainingError(
            "BAGM_RUN_ID, BAGM_RUN_SCRATCH, and --run-scratch are required."
        )
    scratch = args.run_scratch.resolve(strict=False)
    if scratch != Path(environment_scratch).resolve(strict=False):
        raise SO2NBTrainingError("--run-scratch does not match BAGM_RUN_SCRATCH.")
    expected_config = scratch / "config.resolved.yaml"
    if args.config.resolve(strict=False) != expected_config.resolve(strict=False):
        raise SO2NBTrainingError("--config must be the worker-resolved config.")
    expected_scratch = (paths.scratch_root / "active_runs" / run_id).resolve(False)
    if scratch != expected_scratch:
        raise SO2NBTrainingError("Run scratch is not canonical for BAGM_RUN_ID.")
    config = load_yaml_mapping(expected_config)
    validate_target_isolated_config(config)
    return run_id, scratch, config, paths


def _build_model(
    config: Mapping[str, Any], bundle: SO2NBDataBundle
) -> TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer:
    model_config = _section(config, "model")
    trainer = _section(config, "trainer")
    set_deterministic_seed(
        MODEL_SEED,
        deterministic=bool(trainer["deterministic"]),
        warn_only=bool(trainer["deterministic_warn_only"]),
    )
    first = bundle.training_batches[0]
    model = TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer(
        num_genes=first.n_genes,
        node_covariate_dim=int(first.node_covariates.shape[1]),
        hidden_dim=int(model_config["hidden_dim"]),
        attention_heads=int(model_config["attention_heads"]),
        attention_head_dim=int(model_config["attention_head_dim"]),
        graph_layers=int(model_config["graph_layers"]),
        ffn_dim=int(model_config["ffn_dim"]),
        decoder_dim=int(model_config["decoder_dim"]),
        geometry_hidden_dim=int(model_config["geometry_hidden_dim"]),
        dropout=float(model_config["dropout"]),
        attention_dropout=float(model_config["attention_dropout"]),
        relative_geometry_dim=int(model_config["relative_geometry_dim"]),
        qk_normalization_epsilon=float(model_config["qk_normalization_epsilon"]),
        logit_scale_initial=float(model_config["logit_scale_initial"]),
        logit_scale_minimum=float(model_config["logit_scale_minimum"]),
        logit_scale_maximum=float(model_config["logit_scale_maximum"]),
        modulation_amplitude=float(model_config["modulation_amplitude"]),
        geometry_bias_bound=float(model_config["geometry_bias_bound"]),
        receiver_chunk_size=int(model_config["receiver_chunk_size"]),
        max_edges_per_chunk=int(model_config["max_edges_per_chunk"]),
        activation_checkpointing=bool(model_config["activation_checkpointing"]),
        mean_epsilon=float(model_config["output_mean_epsilon"]),
        inverse_dispersion_epsilon=float(
            model_config["inverse_dispersion_epsilon"]
        ),
        inverse_dispersion_initial_value=float(
            model_config["inverse_dispersion_initial_value"]
        ),
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    _require_equal(count, EXPECTED_PARAMETER_COUNT, field="model.parameter_count")
    blocks = tuple(model.blocks)
    if len(blocks) != 4 or len({id(block) for block in blocks}) != 4:
        raise SO2NBTrainingError("Model must expose four unique graph blocks.")
    parameter_ids = [id(p) for block in blocks for p in block.parameters()]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise SO2NBTrainingError("Graph-block parameter sets must be disjoint.")
    _require_equal(model.raw_theta.numel(), 1000, field="model.raw_theta")
    return model


def _verify_hashed_file_mapping(
    mapping: object, *, paths: ProjectPaths, field: str
) -> Mapping[str, str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise SO2NBTrainingError(f"{field} must be a nonempty mapping.")
    normalized: dict[str, str] = {}
    for raw_path, raw_digest in mapping.items():
        relative = Path(str(raw_path))
        digest = str(raw_digest)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise SO2NBTrainingError(f"{field} contains an unsafe path.")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SO2NBTrainingError(f"{field} contains a malformed SHA-256.")
        source = (paths.project_root / relative).resolve(strict=True)
        if not source.is_relative_to(paths.project_root.resolve(strict=True)):
            raise SO2NBTrainingError(f"{field} path escapes the project root.")
        _require_equal(sha256_file(source), digest, field=f"{field}.{relative}")
        normalized[relative.as_posix()] = digest
    return dict(sorted(normalized.items()))


def _load_preflight_receipt(
    config: Mapping[str, Any],
    paths: ProjectPaths,
    bundle: SO2NBDataBundle,
    model: nn.Module,
) -> Mapping[str, Any]:
    launcher = _section(config, "launcher")
    receipt_path = _runtime_path(launcher["hardware_preflight_receipt"], paths)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise SO2NBTrainingError(
            "Checksum-bound target-isolated hardware preflight receipt is missing."
        )
    receipt = _load_json_mapping(receipt_path, description="hardware preflight")
    stored_sha = receipt.get("receipt_content_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_content_sha256", None)
    _require_equal(
        stored_sha,
        _canonical_sha256(unsigned),
        field="preflight.receipt_content_sha256",
    )
    config_sha = _canonical_sha256(config)
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "passed": True,
        "all_required_gates_passed": True,
        "completed_experiment": False,
        "diagnostic_only": True,
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "world_size": WORLD_SIZE,
        "distributed_backend": "nccl",
        "control_plane_backend": "gloo",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "configuration_sha256": config_sha,
        "resolved_config_sha256": config_sha,
        "configuration_file_sha256": sha256_file(
            (paths.project_root / SOURCE_EXPERIMENT_CONFIG).resolve(strict=True)
        ),
        "configuration_source_file_sha256": _configuration_source_file_hashes(
            paths
        ),
        "frozen_task_contract_sha256": FROZEN_TASK_CONTRACT_SHA256,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "split_fingerprint": bundle.split_fingerprint,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "source_artifacts": _overlay_source_artifacts(bundle),
    }
    for field, value in expected.items():
        _require_equal(receipt.get(field), value, field=f"preflight.{field}")
    _require_equal(
        sha256_file((paths.project_root / FROZEN_TASK_CONTRACT).resolve(strict=True)),
        FROZEN_TASK_CONTRACT_SHA256,
        field="frozen_task_contract.file_sha256",
    )
    code_file_sha256 = receipt.get("code_file_sha256")
    if not isinstance(code_file_sha256, Mapping):
        raise SO2NBTrainingError("Preflight code_file_sha256 is missing.")
    _require_equal(
        set(code_file_sha256),
        {path.as_posix() for path in PREFLIGHT_CODE_RELATIVE_PATHS},
        field="preflight.code_file_sha256.paths",
    )
    verified_code_hashes = _verify_hashed_file_mapping(
        code_file_sha256, paths=paths, field="preflight.code_file_sha256"
    )
    _require_equal(
        set(verified_code_hashes),
        {path.as_posix() for path in PREFLIGHT_CODE_RELATIVE_PATHS},
        field="preflight.verified_code_file_sha256.paths",
    )
    _require_equal(
        receipt.get("code_hash_scope"),
        PREFLIGHT_CODE_HASH_SCOPE,
        field="preflight.code_hash_scope",
    )

    gates = receipt.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != PREFLIGHT_REQUIRED_GATES:
        raise SO2NBTrainingError("Preflight required-gate set is incomplete.")
    if any(
        not isinstance(gate, Mapping) or gate.get("passed") is not True
        for gate in gates.values()
    ):
        raise SO2NBTrainingError("Preflight contains a failed or malformed gate.")

    dataset = receipt.get("dataset")
    if not isinstance(dataset, Mapping):
        raise SO2NBTrainingError("Preflight dataset evidence is missing.")
    expected_dataset = {
        "training_aliases": list(TRAINING_ALIASES),
        "validation_aliases": list(VALIDATION_ALIASES),
        "test_aliases": [],
        "training_core_count": 12,
        "validation_core_count": 2,
        "test_core_count": 0,
        "training_cells": 208_696,
        "validation_cells": 37_367,
        "test_cells": 0,
        "biological_target_count": 1000,
        "node_covariate_count": 22,
        "representative_training_alias": dataset.get(
            "representative_training_alias"
        ),
        "representative_validation_alias": dataset.get(
            "representative_validation_alias"
        ),
        "representative_training_is_largest_by_nodes_and_edges": True,
        "representative_validation_is_largest_by_nodes_and_edges": True,
        "per_core": dataset.get("per_core"),
    }
    _require_equal(set(dataset), set(expected_dataset), field="preflight.dataset.fields")
    for field, value in expected_dataset.items():
        _require_equal(dataset.get(field), value, field=f"preflight.dataset.{field}")
    _require_equal(
        dataset["representative_training_alias"],
        "SO2-C23",
        field="preflight.dataset.representative_training_alias",
    )
    _require_equal(
        dataset["representative_validation_alias"],
        "SO2-C27",
        field="preflight.dataset.representative_validation_alias",
    )
    per_core = dataset["per_core"]
    if not isinstance(per_core, Mapping) or set(per_core) != set(
        TRAINING_ALIASES + VALIDATION_ALIASES
    ):
        raise SO2NBTrainingError("Preflight per-core dataset evidence is incomplete.")
    for batch in bundle.all_batches:
        _require_equal(
            per_core.get(batch.alias),
            {
                "role": batch.role,
                "n_nodes": batch.n_nodes,
                "n_edges": batch.n_edges,
                "n_genes": batch.n_genes,
            },
            field=f"preflight.dataset.per_core.{batch.alias}",
        )

    model_evidence = receipt.get("model")
    model_fields = {
        "class",
        "parameter_count",
        "graph_block_count",
        "unique_graph_block_count",
        "per_block_parameter_count",
        "raw_theta_parameter_count",
        "state_dict_key_sha256",
        "target_input_is_separate",
        "static_clean_source_bank",
    }
    if not isinstance(model_evidence, Mapping) or set(model_evidence) != model_fields:
        raise SO2NBTrainingError("Preflight model evidence schema drifted.")
    for field, value in {
        "class": type(model).__name__,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "graph_block_count": 4,
        "unique_graph_block_count": 4,
        "raw_theta_parameter_count": 1000,
        "target_input_is_separate": True,
        "static_clean_source_bank": True,
    }.items():
        _require_equal(model_evidence.get(field), value, field=f"preflight.model.{field}")
    per_block = model_evidence.get("per_block_parameter_count")
    expected_per_block = [sum(p.numel() for p in block.parameters()) for block in model.blocks]
    _require_equal(
        per_block,
        expected_per_block,
        field="preflight.model.per_block_parameter_count",
    )
    state_keys_sha = _canonical_sha256(list(model.state_dict().keys()))
    _require_equal(
        model_evidence.get("state_dict_key_sha256"),
        state_keys_sha,
        field="preflight.model.state_dict_key_sha256",
    )

    identities = receipt.get("gpu_identities")
    identity_fields = {
        "rank",
        "local_rank",
        "name",
        "total_memory_bytes",
        "total_memory_gib",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
    }
    if (
        not isinstance(identities, Sequence)
        or isinstance(identities, (str, bytes))
        or len(identities) != WORLD_SIZE
        or any(
            not isinstance(item, Mapping) or set(item) != identity_fields
            for item in identities
        )
    ):
        raise SO2NBTrainingError("Preflight GPU identity evidence is malformed.")
    by_rank = {int(item["rank"]): item for item in identities}
    if set(by_rank) != set(range(WORLD_SIZE)):
        raise SO2NBTrainingError("Preflight GPU identities do not cover four ranks.")
    live = _current_gpu_identity()
    expected_live = by_rank[int(live["rank"])]
    for field in identity_fields - {"total_memory_gib"}:
        _require_equal(
            live.get(field),
            expected_live.get(field),
            field=f"preflight.live_gpu.{field}",
        )

    resources = receipt.get("resource_gates")
    resource_fields = {
        "minimum_free_disk_gib_required",
        "observed_free_disk_gib",
        "maximum_peak_vram_gib_allowed",
        "minimum_vram_headroom_gib_required",
        "peak_allocated_vram_gib_all_ranks",
        "peak_reserved_vram_gib_all_ranks",
        "minimum_allocated_headroom_gib_all_ranks",
        "minimum_reserved_headroom_gib_all_ranks",
        "per_rank",
    }
    if not isinstance(resources, Mapping) or set(resources) != resource_fields:
        raise SO2NBTrainingError("Preflight resource-gate evidence schema drifted.")
    _require_equal(
        float(resources["minimum_free_disk_gib_required"]),
        MINIMUM_FREE_DISK_GIB,
        field="preflight.resource_gates.minimum_free_disk_gib_required",
    )
    for field in (
        "observed_free_disk_gib",
        "maximum_peak_vram_gib_allowed",
        "minimum_vram_headroom_gib_required",
        "peak_allocated_vram_gib_all_ranks",
        "peak_reserved_vram_gib_all_ranks",
        "minimum_allocated_headroom_gib_all_ranks",
        "minimum_reserved_headroom_gib_all_ranks",
    ):
        value = float(resources[field])
        if not math.isfinite(value) or value < 0.0:
            raise SO2NBTrainingError(f"Preflight resource value {field} is invalid.")
    if float(resources["observed_free_disk_gib"]) < MINIMUM_FREE_DISK_GIB:
        raise SO2NBTrainingError("Preflight disk-space gate is below 20 GiB.")
    if float(resources["peak_allocated_vram_gib_all_ranks"]) > float(
        resources["maximum_peak_vram_gib_allowed"]
    ):
        raise SO2NBTrainingError("Preflight peak VRAM exceeds its frozen limit.")
    if min(
        float(resources["minimum_allocated_headroom_gib_all_ranks"]),
        float(resources["minimum_reserved_headroom_gib_all_ranks"]),
    ) < float(resources["minimum_vram_headroom_gib_required"]):
        raise SO2NBTrainingError("Preflight VRAM headroom is insufficient.")
    per_rank = resources["per_rank"]
    if not isinstance(per_rank, Sequence) or len(per_rank) != WORLD_SIZE:
        raise SO2NBTrainingError("Preflight resource evidence lacks four ranks.")

    free_gib = shutil.disk_usage(paths.project_root).free / float(1024**3)
    if free_gib < MINIMUM_FREE_DISK_GIB:
        raise SO2NBTrainingError(
            f"Live free disk is {free_gib:.3f} GiB; at least 20 GiB is required."
        )
    return receipt


def _validate_cpu_graph(batch: SO2NBCoreBatch) -> None:
    edges = batch.edge_index
    geometry = batch.relative_geometry
    if edges.device.type != "cpu" or geometry.device.type != "cpu":
        raise SO2NBTrainingError("Graph edges and relative geometry must stay on CPU.")
    if edges.dtype != torch.long or geometry.dtype != torch.float32:
        raise SO2NBTrainingError("CPU graph tensors must be long/float32.")
    if edges.shape[1] != geometry.shape[0]:
        raise SO2NBTrainingError("Graph edges and geometry are not aligned.")
    if edges.shape[1] and bool((edges[0] == edges[1]).any()):
        raise SO2NBTrainingError("Target-isolated routing prohibits self edges.")
    receivers = edges[1]
    if receivers.numel() > 1 and bool((receivers[1:] < receivers[:-1]).any()):
        raise SO2NBTrainingError("Graph edges must be receiver-major sorted.")


def _stage_core(
    batch: SO2NBCoreBatch, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_cpu_graph(batch)
    return (
        batch.input_expression.to(device=device, dtype=torch.float32),
        batch.raw_count_target.to(device=device, dtype=torch.int32),
        batch.node_covariates.to(device=device, dtype=torch.float32),
    )


def _model_step_seed(
    global_epoch: int, update_index: int, alias: str, rank: int
) -> int:
    payload = {
        "namespace": "bagm.so2.target_isolated_nb.model_step.v1",
        "model_seed": MODEL_SEED,
        "global_epoch": int(global_epoch),
        "optimizer_update_in_epoch": int(update_index),
        "core_alias": str(alias),
        "rank": int(rank),
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_json(payload).encode("utf-8")).digest()[:8],
        byteorder="little",
    ) & ((1 << 63) - 1)


def _seed_model_step(seed: int, device: torch.device) -> None:
    torch.default_generator.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))


def _all_gather_objects(local: Any, *, control_group: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather_object(gathered, local, group=control_group)
    return gathered


def _all_rank_rng_states(*, control_group: Any) -> tuple[Mapping[str, Any], ...]:
    return tuple(_all_gather_objects(capture_rng_state(), control_group=control_group))


def _gradient_row(summary: Any, *, run_id: str) -> Mapping[str, Any]:
    row = asdict(summary)
    row["run_id"] = run_id
    row["model_seed"] = MODEL_SEED
    return row


def _partition_and_mask(
    batch: SO2NBCoreBatch,
    *,
    rank: int,
    global_epoch: int | None,
) -> tuple[Any, Any]:
    partition = make_contiguous_target_partition(
        batch.n_nodes, world_size=WORLD_SIZE, rank=rank
    )
    if partition.n_targets <= 0:
        raise SO2NBTrainingError("Every rank must receive a nonempty target shard.")
    if global_epoch is None:
        realization = make_validation_target_cell_masks(
            batch.n_nodes,
            batch.n_genes,
            namespace=VALIDATION_MASK_NAMESPACE,
            core_alias=batch.alias,
            target_indices=partition.target_indices,
        )
    else:
        realization = make_training_target_cell_masks(
            batch.n_nodes,
            batch.n_genes,
            base_seed=MASK_BASE_SEED,
            namespace=TRAINING_MASK_NAMESPACE,
            core_alias=batch.alias,
            global_epoch=global_epoch,
            target_indices=partition.target_indices,
        )
    if realization.n_targets != partition.n_targets:
        raise SO2NBTrainingError("Mask and target partition sizes differ.")
    if realization.n_masked_entries <= 0 or not bool(
        np.all(realization.masked_gene_counts >= 1)
    ):
        raise SO2NBTrainingError("Every target mask must be nonempty.")
    if not np.array_equal(realization.target_indices, partition.target_indices):
        raise SO2NBTrainingError("Mask targets differ from the rank partition.")
    partition_receipt = partition.to_receipt()
    mask_receipt = realization.to_receipt()
    verify_target_cell_receipt(partition_receipt)
    verify_target_cell_receipt(mask_receipt)
    _require_equal(
        mask_receipt["target_indices_sha256"],
        partition_receipt["target_indices_sha256"],
        field="mask_partition.target_indices_sha256",
    )
    return partition, realization


def _coverage_receipt(
    *,
    batch: SO2NBCoreBatch,
    role: str,
    global_epoch: int | None,
    gathered: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if len(gathered) != WORLD_SIZE:
        raise SO2NBTrainingError("Coverage evidence must contain four rank shards.")
    by_rank = {int(item.get("rank", -1)): item for item in gathered}
    if set(by_rank) != set(range(WORLD_SIZE)):
        raise SO2NBTrainingError("Coverage evidence does not cover every rank once.")
    canonical_partitions = make_contiguous_target_partitions(batch.n_nodes, WORLD_SIZE)
    partition_coverage = target_partition_coverage_receipt(canonical_partitions)
    for rank, partition in enumerate(canonical_partitions):
        item = by_rank[rank]
        _require_equal(item.get("alias"), batch.alias, field="coverage.alias")
        _require_equal(item.get("role"), role, field="coverage.role")
        _require_equal(item.get("partition"), partition.to_receipt(), field="coverage.partition")
        mask_receipt = item.get("mask")
        if not isinstance(mask_receipt, Mapping):
            raise SO2NBTrainingError("Coverage mask receipt is malformed.")
        verify_target_cell_receipt(mask_receipt)
        _require_equal(
            mask_receipt.get("n_targets"), partition.n_targets, field="coverage.n_targets"
        )
        _require_equal(
            mask_receipt.get("target_indices_sha256"),
            partition.to_receipt()["target_indices_sha256"],
            field="coverage.target_indices_sha256",
        )
        _require_equal(mask_receipt.get("all_masks_nonempty"), True, field="coverage.nonempty")
        _require_equal(item.get("neighbor_masked_entry_count"), 0, field="coverage.neighbor_masks")
        if role == "training":
            _require_equal(mask_receipt.get("global_epoch"), global_epoch, field="coverage.epoch")
        else:
            _require_equal(mask_receipt.get("fixed_across_epochs"), True, field="coverage.fixed")
    masked_entries = sum(int(item["mask"]["masked_entry_count"]) for item in gathered)
    target_cells = sum(int(item["mask"]["n_targets"]) for item in gathered)
    if target_cells != batch.n_nodes or masked_entries < target_cells:
        raise SO2NBTrainingError("Target coverage or nonempty-mask coverage is incomplete.")
    payload: dict[str, Any] = {
        "schema": "target_isolated_core_target_coverage_v1",
        "role": role,
        "core_alias": batch.alias,
        "global_epoch": global_epoch,
        "world_size": WORLD_SIZE,
        "target_cells": target_cells,
        "target_visit_count_min": 1,
        "target_visit_count_max": 1,
        "masked_entries": masked_entries,
        "neighbor_masked_entry_count": 0,
        "partition_coverage_receipt": partition_coverage,
        "rank_shards": [by_rank[rank] for rank in range(WORLD_SIZE)],
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload


def _local_receipt_record(
    *, rank: int, batch: SO2NBCoreBatch, partition: Any, realization: Any
) -> Mapping[str, Any]:
    return {
        "rank": int(rank),
        "alias": batch.alias,
        "role": realization.role,
        "partition": partition.to_receipt(),
        "mask": realization.to_receipt(),
        "neighbor_masked_entry_count": 0,
    }


def _make_target_tensors(
    *,
    clean_input: torch.Tensor,
    realization: Any,
    target_start: int,
    target_stop: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(
        device=device, dtype=torch.bool
    )
    target_input = clean_input[target_start:target_stop].clone()
    target_input.masked_fill_(mask, 0.0)
    if not bool((target_input.masked_select(mask) == 0).all()):
        raise SO2NBTrainingError("Masked target values were not zeroed before forward.")
    visible = ~mask
    if not torch.equal(
        target_input.masked_select(visible),
        clean_input[target_start:target_stop].masked_select(visible),
    ):
        raise SO2NBTrainingError("Visible target input changed while masking.")
    return target_input, mask


def _forward_target_shard(
    *,
    training_model: nn.Module,
    batch: SO2NBCoreBatch,
    clean_input: torch.Tensor,
    covariates: torch.Tensor,
    target_input: torch.Tensor,
    mask: torch.Tensor,
    target_start: int,
    target_stop: int,
    trainer: Mapping[str, Any],
) -> NegativeBinomialModelOutput:
    with _autocast_context(
        enabled=bool(trainer["amp"]),
        device=clean_input.device,
        dtype_name=str(trainer["amp_dtype"]),
    ):
        output = training_model(
            input_expression=clean_input,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=batch.edge_index,
            relative_geometry=batch.relative_geometry,
            node_covariates=covariates,
            target_start=target_start,
            target_stop=target_stop,
        )
    if not isinstance(output, NegativeBinomialModelOutput):
        raise SO2NBTrainingError("Target-isolated model did not return NB2 output.")
    if output.mu.shape != mask.shape:
        raise SO2NBTrainingError("Target-only model output is not shard-aligned.")
    return output


def _paired_update(
    *,
    model: TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
    training_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    pair: tuple[str, str],
    update_index: int,
    global_epoch: int,
    rank: int,
    device: torch.device,
    batches: Mapping[str, SO2NBCoreBatch],
    trainer: Mapping[str, Any],
    control_group: Any,
    global_tracker: FullGradientDirectionTracker,
    block_tracker: BlockGradientDirectionTracker,
    theta_tracker: FullGradientDirectionTracker,
    theta_view: nn.Module,
) -> tuple[Mapping[str, tuple[float, int, int]], tuple[Mapping[str, Any], ...], float]:
    """Execute one exact two-core pooled update on every rank."""

    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    plans: list[tuple[SO2NBCoreBatch, Any, Any]] = []
    local_count = 0
    for alias in pair:
        batch = batches[alias]
        completed_epoch = global_epoch + 1
        partition, realization = _partition_and_mask(
            batch, rank=rank, global_epoch=completed_epoch
        )
        plans.append((batch, partition, realization))
        local_count += int(realization.n_masked_entries)
    global_count_tensor = torch.tensor(local_count, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(global_count_tensor, op=torch.distributed.ReduceOp.SUM)
    global_count = int(global_count_tensor.item())
    if global_count <= 0:
        raise SO2NBTrainingError("Paired update has no globally masked entries.")

    local_stats: dict[str, tuple[float, int, int]] = {}
    coverage: list[Mapping[str, Any]] = []
    try:
        for position, (batch, partition, realization) in enumerate(plans):
            _clear_graph_layout_caches(training_model)
            clean_input, raw_target, covariates = _stage_core(batch, device=device)
            target_input, mask = _make_target_tensors(
                clean_input=clean_input,
                realization=realization,
                target_start=partition.start,
                target_stop=partition.stop,
                device=device,
            )
            _seed_model_step(
                _model_step_seed(completed_epoch, update_index, batch.alias, rank),
                device,
            )
            sync_context = training_model.no_sync() if position == 0 else nullcontext()
            with sync_context:
                output = _forward_target_shard(
                    training_model=training_model,
                    batch=batch,
                    clean_input=clean_input,
                    covariates=covariates,
                    target_input=target_input,
                    mask=mask,
                    target_start=partition.start,
                    target_stop=partition.stop,
                    trainer=trainer,
                )
                raw_shard = raw_target[partition.start : partition.stop]
                local_nll_sum = masked_negative_binomial_nll_sum(
                    output.mu, output.theta, raw_shard, mask
                )
                if local_nll_sum.dtype != torch.float32 or not bool(
                    torch.isfinite(local_nll_sum)
                ):
                    raise FloatingPointError("Training summed NB2 NLL is non-finite.")
                scaled_loss = local_nll_sum * (WORLD_SIZE / float(global_count))
                scaler.scale(scaled_loss).backward()
            local_stats[batch.alias] = (
                float(local_nll_sum.detach().to(dtype=torch.float64).cpu()),
                int(realization.n_masked_entries),
                int(realization.n_targets),
            )
            gathered = _all_gather_objects(
                _local_receipt_record(
                    rank=rank,
                    batch=batch,
                    partition=partition,
                    realization=realization,
                ),
                control_group=control_group,
            )
            coverage.append(
                _coverage_receipt(
                    batch=batch,
                    role="training",
                    global_epoch=completed_epoch,
                    gathered=gathered,
                )
            )
            del output, scaled_loss, local_nll_sum, target_input, mask
            del clean_input, raw_target, covariates
            _clear_graph_layout_caches(training_model)
            torch.cuda.empty_cache()

        scaler.unscale_(optimizer)
        _assert_finite_gradients(model)
        context = GradientDirectionUpdateContext(
            global_epoch=global_epoch,
            completed_global_epoch=global_epoch + 1,
            optimizer_update_in_epoch=update_index,
            cumulative_optimizer_update=(
                global_epoch * OPTIMIZER_UPDATES_PER_EPOCH + update_index + 1
            ),
            aliases=pair,
        )
        # Run the same diagnostic kernels on every rank; persist scalar rows on
        # rank zero only.  This avoids rank-skew before the next NCCL collective.
        global_tracker.observe(model, context)
        block_tracker.observe(model, context)
        theta_tracker.observe(theta_view, context)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            training_model.parameters(),
            float(trainer["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        norm_range = torch.stack(
            [gradient_norm.detach().double(), gradient_norm.detach().double()]
        )
        torch.distributed.all_reduce(norm_range[0], op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(norm_range[1], op=torch.distributed.ReduceOp.MAX)
        if not torch.isclose(norm_range[0], norm_range[1], rtol=1e-5, atol=1e-7):
            raise SO2NBTrainingError("Post-DDP gradient norms disagree across ranks.")
        scaler.step(optimizer)
        scaler.update()
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError(f"Non-finite parameter after update: {name}.")
        return local_stats, tuple(coverage), float(gradient_norm.detach().cpu())
    finally:
        _clear_graph_layout_caches(training_model)


@dataclass(frozen=True)
class ValidationResult:
    pooled_negative_binomial_nll: float
    raw_count_mae: float
    raw_count_rmse: float
    log1p_mae: float
    log1p_rmse: float
    poisson_deviance: float
    observed_zero_rate: float
    predicted_zero_probability_mean: float
    zero_brier_score: float
    masked_entries: int
    target_cells: int
    observed_count_mean: float
    predicted_count_mean: float
    per_core: Mapping[str, Mapping[str, float | int]]
    coverage_receipts: Mapping[str, Mapping[str, Any]]


def _validation_sum_tensor(
    *,
    output: NegativeBinomialModelOutput,
    raw_target: torch.Tensor,
    mask: torch.Tensor,
    target_cells: int,
) -> torch.Tensor:
    """Build additive FP64 transport statistics from FP32 frozen metrics."""

    metrics = masked_negative_binomial_metrics(
        output.mu, output.theta, raw_target, mask
    )
    n = int(metrics.n_masked_entries)
    exact_nll_sum = masked_negative_binomial_nll_sum(
        output.mu, output.theta, raw_target, mask
    )
    if n != int(mask.sum().item()) or n <= 0:
        raise SO2NBTrainingError("Validation masked-entry count drifted.")
    observed_sum = raw_target.masked_select(mask).sum(dtype=torch.float64)
    predicted_sum = output.mu.masked_select(mask).sum(dtype=torch.float64)
    values = (
        float(n),
        float(exact_nll_sum.detach().double().cpu()),
        float(metrics.raw_count_mae.detach().double().cpu()) * n,
        float(metrics.raw_count_rmse.detach().double().cpu()) ** 2 * n,
        float(metrics.log1p_mae.detach().double().cpu()) * n,
        float(metrics.log1p_rmse.detach().double().cpu()) ** 2 * n,
        float(metrics.poisson_deviance.detach().double().cpu()) * n,
        float(metrics.observed_zero_rate.detach().double().cpu()) * n,
        float(
            metrics.predicted_zero_probability_mean.detach().double().cpu()
        )
        * n,
        float(metrics.zero_brier_score.detach().double().cpu()) * n,
        float(observed_sum.cpu()),
        float(predicted_sum.cpu()),
        float(target_cells),
    )
    tensor = torch.tensor(values, dtype=torch.float64, device=output.mu.device)
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
        raise FloatingPointError("Validation additive statistics are invalid.")
    return tensor


def _metrics_from_sums(values: torch.Tensor) -> Mapping[str, float | int]:
    if values.shape != (len(_VALIDATION_SUM_FIELDS),):
        raise SO2NBTrainingError("Validation sufficient-statistic shape drifted.")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise FloatingPointError("Reduced validation statistics are invalid.")
    row = {
        field: float(values[index].item())
        for index, field in enumerate(_VALIDATION_SUM_FIELDS)
    }
    n = int(round(row["masked_entries"]))
    targets = int(round(row["target_cells"]))
    if n <= 0 or targets <= 0:
        raise SO2NBTrainingError("Reduced validation support is empty.")
    if not math.isclose(row["masked_entries"], n, rel_tol=0.0, abs_tol=0.0):
        raise SO2NBTrainingError("Reduced masked-entry count is not integral.")
    if not math.isclose(row["target_cells"], targets, rel_tol=0.0, abs_tol=0.0):
        raise SO2NBTrainingError("Reduced target count is not integral.")
    return {
        "target_cells": targets,
        "masked_entries": n,
        "pooled_masked_negative_binomial_nll": (
            row["negative_binomial_nll_sum"] / n
        ),
        "masked_raw_count_mae": row["raw_absolute_error_sum"] / n,
        "masked_raw_count_rmse": math.sqrt(row["raw_squared_error_sum"] / n),
        "masked_log1p_mae": row["log1p_absolute_error_sum"] / n,
        "masked_log1p_rmse": math.sqrt(row["log1p_squared_error_sum"] / n),
        "masked_poisson_deviance": row["poisson_deviance_sum"] / n,
        "observed_zero_rate": row["observed_zero_sum"] / n,
        "predicted_zero_probability_mean": (
            row["predicted_zero_probability_sum"] / n
        ),
        "zero_brier_score": row["zero_brier_sum"] / n,
        "observed_count_mean": row["observed_count_sum"] / n,
        "predicted_count_mean": row["predicted_count_sum"] / n,
    }


def _evaluate_validation(
    *,
    bundle: SO2NBDataBundle,
    training_model: DistributedDataParallel,
    rank: int,
    device: torch.device,
    trainer: Mapping[str, Any],
    control_group: Any,
) -> ValidationResult:
    training_model.eval()
    per_core_tensors: list[torch.Tensor] = []
    per_core: dict[str, Mapping[str, float | int]] = {}
    receipts: dict[str, Mapping[str, Any]] = {}
    try:
        with torch.no_grad():
            for alias in VALIDATION_ALIASES:
                batch = bundle.batches_by_alias[alias]
                partition, realization = _partition_and_mask(
                    batch, rank=rank, global_epoch=None
                )
                _clear_graph_layout_caches(training_model)
                clean_input, raw_target, covariates = _stage_core(
                    batch, device=device
                )
                target_input, mask = _make_target_tensors(
                    clean_input=clean_input,
                    realization=realization,
                    target_start=partition.start,
                    target_stop=partition.stop,
                    device=device,
                )
                output = _forward_target_shard(
                    training_model=training_model,
                    batch=batch,
                    clean_input=clean_input,
                    covariates=covariates,
                    target_input=target_input,
                    mask=mask,
                    target_start=partition.start,
                    target_stop=partition.stop,
                    trainer=trainer,
                )
                local_sums = _validation_sum_tensor(
                    output=output,
                    raw_target=raw_target[partition.start : partition.stop],
                    mask=mask,
                    target_cells=partition.n_targets,
                )
                torch.distributed.all_reduce(
                    local_sums, op=torch.distributed.ReduceOp.SUM
                )
                global_sums = local_sums.detach().cpu()
                core_metrics = _metrics_from_sums(global_sums)
                _require_equal(
                    int(core_metrics["target_cells"]),
                    batch.n_nodes,
                    field=f"validation.{alias}.target_cells",
                )
                per_core[alias] = core_metrics
                per_core_tensors.append(global_sums)

                gathered = _all_gather_objects(
                    _local_receipt_record(
                        rank=rank,
                        batch=batch,
                        partition=partition,
                        realization=realization,
                    ),
                    control_group=control_group,
                )
                receipts[alias] = _coverage_receipt(
                    batch=batch,
                    role="validation",
                    global_epoch=None,
                    gathered=gathered,
                )
                del output, local_sums, target_input, mask
                del clean_input, raw_target, covariates
                _clear_graph_layout_caches(training_model)
                torch.cuda.empty_cache()
        pooled_sums = torch.stack(per_core_tensors, dim=0).sum(dim=0)
        pooled = _metrics_from_sums(pooled_sums)
        _require_equal(
            int(pooled["target_cells"]),
            37_367,
            field="validation.target_cells",
        )
        return ValidationResult(
            pooled_negative_binomial_nll=float(
                pooled["pooled_masked_negative_binomial_nll"]
            ),
            raw_count_mae=float(pooled["masked_raw_count_mae"]),
            raw_count_rmse=float(pooled["masked_raw_count_rmse"]),
            log1p_mae=float(pooled["masked_log1p_mae"]),
            log1p_rmse=float(pooled["masked_log1p_rmse"]),
            poisson_deviance=float(pooled["masked_poisson_deviance"]),
            observed_zero_rate=float(pooled["observed_zero_rate"]),
            predicted_zero_probability_mean=float(
                pooled["predicted_zero_probability_mean"]
            ),
            zero_brier_score=float(pooled["zero_brier_score"]),
            masked_entries=int(pooled["masked_entries"]),
            target_cells=int(pooled["target_cells"]),
            observed_count_mean=float(pooled["observed_count_mean"]),
            predicted_count_mean=float(pooled["predicted_count_mean"]),
            per_core=dict(per_core),
            coverage_receipts=dict(receipts),
        )
    finally:
        training_model.train()
        _clear_graph_layout_caches(training_model)


def _aggregate_training_epoch(
    local_by_alias: Mapping[str, tuple[float, int, int]],
    *,
    device: torch.device,
) -> tuple[float, int, int, Mapping[str, Mapping[str, float | int]]]:
    if set(local_by_alias) != set(TRAINING_ALIASES):
        raise SO2NBTrainingError("Training epoch did not visit every core exactly once.")
    transport = torch.zeros(
        (len(TRAINING_ALIASES), 3), dtype=torch.float64, device=device
    )
    for index, alias in enumerate(TRAINING_ALIASES):
        nll_sum, masked_entries, target_cells = local_by_alias[alias]
        transport[index] = torch.tensor(
            [nll_sum, masked_entries, target_cells],
            dtype=torch.float64,
            device=device,
        )
    torch.distributed.all_reduce(transport, op=torch.distributed.ReduceOp.SUM)
    values = transport.detach().cpu()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise FloatingPointError("Reduced training statistics are invalid.")
    per_core: dict[str, Mapping[str, float | int]] = {}
    for index, alias in enumerate(TRAINING_ALIASES):
        nll_sum = float(values[index, 0])
        count_value = float(values[index, 1])
        target_value = float(values[index, 2])
        count = int(round(count_value))
        targets = int(round(target_value))
        if not math.isclose(count_value, count, rel_tol=0.0, abs_tol=0.0):
            raise SO2NBTrainingError("Reduced masked-entry count is not integral.")
        if not math.isclose(target_value, targets, rel_tol=0.0, abs_tol=0.0):
            raise SO2NBTrainingError("Reduced target-cell count is not integral.")
        if count < targets or targets <= 0:
            raise SO2NBTrainingError(f"Training support is invalid for {alias}.")
        per_core[alias] = {
            "target_cells": targets,
            "masked_entries": count,
            "pooled_masked_negative_binomial_nll": nll_sum / count,
        }
    total_count = sum(int(row["masked_entries"]) for row in per_core.values())
    total_targets = sum(int(row["target_cells"]) for row in per_core.values())
    if total_targets != 208_696:
        raise SO2NBTrainingError("Training target coverage is not exactly 208,696.")
    total_nll_sum = float(values[:, 0].sum(dtype=torch.float64).item())
    return total_nll_sum / total_count, total_count, total_targets, per_core


def _validate_epoch_coverage(
    receipts: Sequence[Mapping[str, Any]],
    *,
    global_epoch: int,
    control_group: Any,
) -> Mapping[str, Mapping[str, Any]]:
    by_alias = {str(receipt.get("core_alias")): receipt for receipt in receipts}
    if len(receipts) != len(TRAINING_ALIASES) or set(by_alias) != set(
        TRAINING_ALIASES
    ):
        raise SO2NBTrainingError("Training coverage receipts omit or duplicate a core.")
    if any(
        receipt.get("role") != "training"
        or int(receipt.get("global_epoch", -1)) != global_epoch
        or int(receipt.get("target_visit_count_min", 0)) != 1
        or int(receipt.get("target_visit_count_max", 0)) != 1
        or int(receipt.get("neighbor_masked_entry_count", -1)) != 0
        for receipt in receipts
    ):
        raise SO2NBTrainingError("Training target-coverage contract failed.")
    ordered = {alias: by_alias[alias] for alias in TRAINING_ALIASES}
    digest = _canonical_sha256(ordered)
    rank_digests = _all_gather_objects(digest, control_group=control_group)
    if any(value != digest for value in rank_digests):
        raise SO2NBTrainingError("Rank-local target coverage receipts disagree.")
    if sum(int(value["target_cells"]) for value in ordered.values()) != 208_696:
        raise SO2NBTrainingError("Coverage receipts do not span every training cell.")
    return ordered


def _peak_vram_all_ranks(device: torch.device) -> float:
    value = torch.tensor(
        float(torch.cuda.max_memory_allocated(device)),
        dtype=torch.float64,
        device=device,
    )
    torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return float(value.item()) / float(1024**3)


def _checkpoint_state_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    completed_epoch: int,
    checkpoint_selection_state: EarlyStoppingState,
    patience_control_state: EarlyStoppingState,
    bundle: SO2NBDataBundle,
    rng_states: Sequence[Mapping[str, Any]],
    validation_coverage_receipts: Mapping[str, Mapping[str, Any]],
    configuration_sha256: str,
    preflight_receipt_sha256: str,
) -> Mapping[str, Any]:
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "campaign_id": CAMPAIGN_ID,
        "completed_epoch": int(completed_epoch),
        "optimizer_updates_completed": (
            int(completed_epoch) * OPTIMIZER_UPDATES_PER_EPOCH
        ),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_state_dict": _clone_tree_to_cpu(model.state_dict()),
        "optimizer_state_dict": _clone_tree_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": _clone_tree_to_cpu(scheduler.state_dict()),
        "amp_scaler_state_dict": _clone_tree_to_cpu(scaler.state_dict()),
        "rng_states": tuple(rng_states),
        # The generalized store uses this field to bind latest.ckpt to the
        # literal validation-best payload.  Patience control is persisted
        # separately because its 1e-4 meaningful-improvement threshold is not
        # a checkpoint-selection threshold.
        "early_stopping_state": asdict(checkpoint_selection_state),
        "patience_early_stopping_state": asdict(patience_control_state),
        "best_validation_metric": checkpoint_selection_state.best_value,
        "best_epoch": checkpoint_selection_state.best_epoch,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "configuration_sha256": configuration_sha256,
        "frozen_task_contract_sha256": FROZEN_TASK_CONTRACT_SHA256,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "validation_coverage_receipts": dict(validation_coverage_receipts),
        "gradient_vectors_persisted": False,
        "full_prediction_matrices_persisted": False,
        "test_artifacts_present": False,
    }


def _load_resume(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    bundle: SO2NBDataBundle,
    rank: int,
    configuration_sha256: str,
    preflight_receipt_sha256: str,
) -> tuple[
    EarlyStoppingState,
    EarlyStoppingState,
    Mapping[str, Mapping[str, Any]],
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise SO2NBTrainingError("Resume checkpoint is not a mapping.")
    expected = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "campaign_id": CAMPAIGN_ID,
        "checkpoint_role": "latest",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "configuration_sha256": configuration_sha256,
        "frozen_task_contract_sha256": FROZEN_TASK_CONTRACT_SHA256,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "gradient_vectors_persisted": False,
        "full_prediction_matrices_persisted": False,
        "test_artifacts_present": False,
    }
    for field, value in expected.items():
        _require_equal(payload.get(field), value, field=f"resume.{field}")
    selection = EarlyStoppingState(**dict(payload["early_stopping_state"]))
    control_payload = payload.get("patience_early_stopping_state")
    if not isinstance(control_payload, Mapping):
        raise SO2NBTrainingError("Resume checkpoint lacks patience-control state.")
    control = EarlyStoppingState(**dict(control_payload))
    _require_equal(
        payload.get("completed_epoch"),
        selection.completed_epoch,
        field="resume.selection_epoch",
    )
    _require_equal(
        payload.get("completed_epoch"),
        control.completed_epoch,
        field="resume.patience_epoch",
    )
    _require_equal(
        payload.get("best_validation_metric"),
        selection.best_value,
        field="resume.selected_best_value",
    )
    _require_equal(
        payload.get("best_epoch"),
        selection.best_epoch,
        field="resume.selected_best_epoch",
    )
    if selection.bad_validations != 0 or selection.should_stop:
        raise SO2NBTrainingError("Checkpoint-selection state contains stop controls.")
    receipts = payload.get("validation_coverage_receipts")
    if not isinstance(receipts, Mapping) or set(receipts) != set(VALIDATION_ALIASES):
        raise SO2NBTrainingError("Resume checkpoint lacks fixed validation receipts.")
    embedded = payload.get("embedded_best_checkpoint")
    if not isinstance(embedded, Mapping):
        raise SO2NBTrainingError("Resume checkpoint lacks embedded best recovery state.")
    _require_equal(
        _tree_sha256(embedded),
        payload.get("embedded_best_checkpoint_tree_sha256"),
        field="resume.embedded_best_checkpoint_tree_sha256",
    )
    sibling_best = path.parent / "best.ckpt"
    if (
        not sibling_best.is_file()
        or sha256_file(sibling_best) != payload.get("best_checkpoint_sha256")
    ):
        raise SO2NBTrainingError("Resume best/latest checkpoint transaction is incomplete.")
    sibling = torch.load(sibling_best, map_location="cpu", weights_only=False)
    if not isinstance(sibling, Mapping):
        raise SO2NBTrainingError("Resume best checkpoint is malformed.")
    _require_equal(
        _tree_sha256(sibling),
        payload.get("embedded_best_checkpoint_tree_sha256"),
        field="resume.best_checkpoint_tree_sha256",
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    scaler.load_state_dict(dict(payload["amp_scaler_state_dict"]))
    rng_states = payload.get("rng_states")
    if not isinstance(rng_states, Sequence) or len(rng_states) != WORLD_SIZE:
        raise SO2NBTrainingError("Resume checkpoint lacks four rank RNG states.")
    restore_rng_state(rng_states[rank])
    return control, selection, dict(receipts)


def _initialize_training(
    model: nn.Module,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[DistributedDataParallel, torch.optim.Optimizer, Any, Any]:
    trainer = _section(config, "trainer")
    model.to(device)
    training_model = DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        training_model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(trainer["scheduler_factor"]),
        patience=int(trainer["scheduler_patience"]),
        threshold=float(trainer["scheduler_threshold"]),
        threshold_mode=str(trainer["scheduler_threshold_mode"]),
        min_lr=float(trainer["scheduler_min_learning_rate"]),
    )
    scaler = _make_grad_scaler(device, bool(trainer["amp"]))
    return training_model, optimizer, scheduler, scaler


def _reconcile_metric_events(path: Path, *, checkpoint_epoch: int) -> None:
    rows: list[Mapping[str, Any]] = []
    if path.is_file():
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SO2NBTrainingError("Metric-event log is unreadable.") from exc
    if any(not isinstance(row, Mapping) for row in rows):
        raise SO2NBTrainingError("Metric-event log contains a malformed row.")
    epoch = int(checkpoint_epoch)
    retained = [row for row in rows if int(row.get("step", -1)) <= epoch]
    expected_names = {
        "train/pooled_masked_negative_binomial_nll",
        PRIMARY_METRIC,
    }
    for step in range(1, epoch + 1):
        names = [
            str(row.get("name"))
            for row in retained
            if int(row.get("step", -1)) == step
        ]
        if len(names) != 2 or set(names) != expected_names:
            raise SO2NBTrainingError(
                f"Metric events do not cover committed epoch {step}."
            )
    if len(retained) == len(rows):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ".events.jsonl.reconcile.writing"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in retained:
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_interrupted_temps(scratch: Path) -> None:
    patterns = {
        scratch / "checkpoints": (".*.writing",),
        scratch / "results": (".*.writing",),
        scratch / "metrics": (".*.writing",),
        scratch / "predictions": (".*.writing",),
    }
    for directory, globs in patterns.items():
        if not directory.is_dir():
            continue
        for pattern in globs:
            for path in directory.glob(pattern):
                if path.is_symlink() or not path.is_file():
                    raise SO2NBTrainingError(f"Unsafe interrupted temp path: {path}.")
                path.unlink()


def _remove_partial_finalization_outputs(scratch: Path) -> None:
    candidates = [
        scratch / "summary.json",
        scratch / "metrics/final.json",
        scratch / "diagnostics/final_checkpoint_reload_verification.json",
        scratch / "diagnostics/hardware_preflight.json",
        scratch / "provenance/so2_target_isolated_nb_training.json",
    ]
    for stem in (scratch / "metrics/history", scratch / "predictions/validation"):
        candidates.extend(
            stem.with_suffix(suffix) for suffix in (".parquet", ".jsonl", ".csv")
        )
    for path in candidates:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise SO2NBTrainingError(f"Unsafe partial finalization path: {path}.")
        path.unlink(missing_ok=True)


def _make_epoch_row(
    *,
    completed_epoch: int,
    train_nll: float,
    train_masked_entries: int,
    train_target_cells: int,
    validation: ValidationResult,
    theta: tuple[float, float, float],
    learning_rate: float,
    duration: float,
    peak_vram: float,
    checkpoint_selection: EarlyStoppingState,
    patience_control: EarlyStoppingState,
    core_order: Sequence[str],
    training_receipts: Mapping[str, Mapping[str, Any]],
    global_gradient: Mapping[str, Any],
    block_gradients: Sequence[Mapping[str, Any]],
    theta_gradient: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema": EPOCH_METRICS_SCHEMA,
        "global_epoch": completed_epoch,
        "train_pooled_masked_negative_binomial_nll": train_nll,
        "train_masked_entries": train_masked_entries,
        "train_target_cells": train_target_cells,
        "validation_pooled_masked_negative_binomial_nll": (
            validation.pooled_negative_binomial_nll
        ),
        "validation_masked_entries": validation.masked_entries,
        "validation_target_cells": validation.target_cells,
        "validation_masked_raw_count_mae": validation.raw_count_mae,
        "validation_masked_raw_count_rmse": validation.raw_count_rmse,
        "validation_masked_log1p_mae": validation.log1p_mae,
        "validation_masked_log1p_rmse": validation.log1p_rmse,
        "validation_masked_poisson_deviance": validation.poisson_deviance,
        "validation_observed_zero_rate": validation.observed_zero_rate,
        "validation_predicted_zero_probability_mean": (
            validation.predicted_zero_probability_mean
        ),
        "validation_zero_brier_score": validation.zero_brier_score,
        "inverse_dispersion_min": theta[0],
        "inverse_dispersion_median": theta[1],
        "inverse_dispersion_max": theta[2],
        "learning_rate": learning_rate,
        "epoch_duration_seconds": duration,
        "training_targets_per_second": train_target_cells / duration,
        "peak_vram_gib_all_ranks": peak_vram,
        "best_validation_metric": checkpoint_selection.best_value,
        "best_epoch": checkpoint_selection.best_epoch,
        "patience_control_best_validation_metric": patience_control.best_value,
        "patience_control_best_epoch": patience_control.best_epoch,
        "bad_validations": patience_control.bad_validations,
        "improved": checkpoint_selection.improved,
        "patience_control_improved": patience_control.improved,
        "should_stop": patience_control.should_stop,
        "stop_reason": patience_control.stop_reason or "",
        "core_order_json": _json_field(list(core_order)),
        "training_coverage_receipts_json": _json_field(training_receipts),
        "validation_coverage_receipts_json": _json_field(
            validation.coverage_receipts
        ),
        "global_gradient_json": _json_field(global_gradient),
        "block_gradients_json": _json_field(list(block_gradients)),
        "dispersion_gradient_json": _json_field(theta_gradient),
    }


def _per_core_rows(
    *,
    completed_epoch: int,
    training: Mapping[str, Mapping[str, float | int]],
    training_receipts: Mapping[str, Mapping[str, Any]],
    validation: ValidationResult,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for alias in TRAINING_ALIASES:
        metrics = training[alias]
        rows.append(
            {
                "schema": PER_CORE_METRICS_SCHEMA,
                "global_epoch": completed_epoch,
                "split": "training",
                "core_alias": alias,
                "target_cells": metrics["target_cells"],
                "masked_entries": metrics["masked_entries"],
                "pooled_masked_negative_binomial_nll": metrics[
                    "pooled_masked_negative_binomial_nll"
                ],
                "masked_raw_count_mae": "",
                "masked_raw_count_rmse": "",
                "masked_log1p_mae": "",
                "masked_log1p_rmse": "",
                "masked_poisson_deviance": "",
                "observed_zero_rate": "",
                "predicted_zero_probability_mean": "",
                "zero_brier_score": "",
                "coverage_receipt_sha256": training_receipts[alias][
                    "receipt_sha256"
                ],
            }
        )
    for alias in VALIDATION_ALIASES:
        metrics = validation.per_core[alias]
        rows.append(
            {
                "schema": PER_CORE_METRICS_SCHEMA,
                "global_epoch": completed_epoch,
                "split": "validation",
                "core_alias": alias,
                "target_cells": metrics["target_cells"],
                "masked_entries": metrics["masked_entries"],
                "pooled_masked_negative_binomial_nll": metrics[
                    "pooled_masked_negative_binomial_nll"
                ],
                "masked_raw_count_mae": metrics["masked_raw_count_mae"],
                "masked_raw_count_rmse": metrics["masked_raw_count_rmse"],
                "masked_log1p_mae": metrics["masked_log1p_mae"],
                "masked_log1p_rmse": metrics["masked_log1p_rmse"],
                "masked_poisson_deviance": metrics["masked_poisson_deviance"],
                "observed_zero_rate": metrics["observed_zero_rate"],
                "predicted_zero_probability_mean": metrics[
                    "predicted_zero_probability_mean"
                ],
                "zero_brier_score": metrics["zero_brier_score"],
                "coverage_receipt_sha256": validation.coverage_receipts[alias][
                    "receipt_sha256"
                ],
            }
        )
    return rows


def _archive_history_rows(
    run_id: str, rows: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    integer_fields = {
        "global_epoch",
        "train_masked_entries",
        "train_target_cells",
        "validation_masked_entries",
        "validation_target_cells",
        "best_epoch",
        "patience_control_best_epoch",
        "bad_validations",
    }
    boolean_fields = {"improved", "patience_control_improved", "should_stop"}
    string_fields = {"schema", "stop_reason"}
    result: list[Mapping[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"run_id": run_id, "split": "train_validation"}
        for field, raw in row.items():
            if field.endswith("_json"):
                continue
            if field in integer_fields:
                record[field] = int(raw)
            elif field in boolean_fields:
                record[field] = str(raw).strip().lower() in {"1", "true"}
            elif field in string_fields:
                record[field] = str(raw)
            else:
                record[field] = float(raw)
        result.append(record)
    return result


def _validation_prediction_rows(
    *,
    run_id: str,
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
    validation: ValidationResult,
) -> list[Mapping[str, Any]]:
    dataset_id = str(_section(config, "dataset")["dataset_id"])
    rows: list[Mapping[str, Any]] = []
    for alias in VALIDATION_ALIASES:
        metrics = validation.per_core[alias]
        batch = bundle.batches_by_alias[alias]
        sample_key = hashlib.sha256(
            f"bagm.so2.target-isolated.validation-summary.v1:{alias}".encode(
                "utf-8"
            )
        ).hexdigest()
        rows.append(
            {
                "run_id": run_id,
                "sample_key": sample_key,
                "dataset_id": dataset_id,
                "split": "validation",
                "y_true": float(metrics["observed_count_mean"]),
                "y_pred": float(metrics["predicted_count_mean"]),
                "sample_loss": float(
                    metrics["pooled_masked_negative_binomial_nll"]
                ),
                "effective_mask_rate": int(metrics["masked_entries"])
                / float(batch.n_nodes * batch.n_genes),
                "node_count": batch.n_nodes,
                "edge_count": batch.n_edges,
                "graph_id": alias,
                "masked_entry_count": int(metrics["masked_entries"]),
                "aggregation_unit": "core_masked_entry_mean",
            }
        )
    return validate_prediction_rows(
        rows, expected_run_id=run_id, expected_split="validation"
    )


def _require_disk_headroom(path: Path) -> float:
    free_gib = shutil.disk_usage(path).free / float(1024**3)
    if not math.isfinite(free_gib) or free_gib < MINIMUM_FREE_DISK_GIB:
        raise SO2NBTrainingError(
            f"Free disk is {free_gib:.3f} GiB; at least 20 GiB is required."
        )
    return free_gib


def _validate_final_outputs(
    scratch: Path,
    *,
    run_id: str,
    checkpoint_store: AtomicBestLatestCheckpointStore,
    expect_latest: bool,
    expected_summary: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    required = (
        "results/epoch_metrics.csv",
        "results/per_core_epoch_metrics.csv",
        "results/gradient_direction_metrics.csv",
        "results/gradient_direction_by_block.csv",
        "results/gradient_direction_dispersion.csv",
        "metrics/events.jsonl",
        "metrics/final.json",
        "diagnostics/final_checkpoint_reload_verification.json",
        "diagnostics/hardware_preflight.json",
        "provenance/so2_target_isolated_nb_training.json",
        "summary.json",
    )
    for relative in required:
        path = scratch / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise SO2NBTrainingError(
                f"Missing, empty, or unsafe runner output: {relative}."
            )
    summary = _load_json_mapping(scratch / "summary.json", description="summary")
    _require_equal(summary.get("run_id"), run_id, field="summary.run_id")
    _require_equal(summary.get("status"), "success", field="summary.status")
    _require_equal(summary.get("campaign_id"), CAMPAIGN_ID, field="summary.campaign")
    _require_equal(summary.get("primary_metric_name"), PRIMARY_METRIC, field="summary.primary")
    if expected_summary is not None:
        _require_equal(
            _canonical_sha256(summary),
            _canonical_sha256(expected_summary),
            field="summary.content_sha256",
        )
    final_epoch = int(summary.get("final_epoch", 0))
    best_epoch = int(summary.get("best_epoch", 0))
    if final_epoch < 1 or best_epoch not in range(1, final_epoch + 1):
        raise SO2NBTrainingError("Summary epoch fields are invalid.")
    selection_payload = summary.get("checkpoint_selection_state")
    control_payload = summary.get("patience_control_state")
    if not isinstance(selection_payload, Mapping) or not isinstance(
        control_payload, Mapping
    ):
        raise SO2NBTrainingError("Summary lacks both validation-control states.")
    selection = EarlyStoppingState(**dict(selection_payload))
    control = EarlyStoppingState(**dict(control_payload))
    _require_equal(
        selection.completed_epoch,
        final_epoch,
        field="summary.checkpoint_selection_state.completed_epoch",
    )
    _require_equal(
        control.completed_epoch,
        final_epoch,
        field="summary.patience_control_state.completed_epoch",
    )
    _require_equal(selection.best_epoch, best_epoch, field="summary.selected_best_epoch")
    if selection.bad_validations != 0 or selection.should_stop:
        raise SO2NBTrainingError("Final checkpoint-selection state owns stop controls.")

    best = checkpoint_store.load("best")
    _require_equal(best.get("completed_epoch"), best_epoch, field="best.epoch")
    best_selection_payload = best.get("early_stopping_state")
    best_control_payload = best.get("patience_early_stopping_state")
    if not isinstance(best_selection_payload, Mapping) or not isinstance(
        best_control_payload, Mapping
    ):
        raise SO2NBTrainingError("Best checkpoint lacks both validation controls.")
    best_selection = EarlyStoppingState(**dict(best_selection_payload))
    best_control = EarlyStoppingState(**dict(best_control_payload))
    _require_equal(
        best_selection.best_value,
        selection.best_value,
        field="best.selected_best_value",
    )
    _require_equal(
        best_selection.best_epoch,
        selection.best_epoch,
        field="best.selected_best_epoch",
    )
    _require_equal(
        best_control.completed_epoch,
        best_epoch,
        field="best.patience_control_state.completed_epoch",
    )
    best_sha = sha256_file(checkpoint_store.best_path)
    if expect_latest:
        latest = checkpoint_store.load("latest")
        _require_equal(latest.get("completed_epoch"), final_epoch, field="latest.epoch")
        _require_equal(latest.get("best_checkpoint_sha256"), best_sha, field="latest.best_sha")
        _require_equal(
            latest.get("early_stopping_state"),
            dict(selection_payload),
            field="latest.checkpoint_selection_state",
        )
        _require_equal(
            latest.get("patience_early_stopping_state"),
            dict(control_payload),
            field="latest.patience_control_state",
        )
        expected_checkpoints = {"best.ckpt", "latest.ckpt"}
    else:
        expected_checkpoints = {"best.ckpt"}
    actual_checkpoints = {
        path.name for path in checkpoint_store.directory.iterdir() if path.is_file()
    }
    _require_equal(actual_checkpoints, expected_checkpoints, field="checkpoint.layout")

    epoch_writer = DurableScalarCSV(
        scratch,
        "results/epoch_metrics.csv",
        EPOCH_COLUMNS,
        rows_per_epoch=1,
    )
    core_writer = DurableScalarCSV(
        scratch,
        "results/per_core_epoch_metrics.csv",
        PER_CORE_COLUMNS,
        rows_per_epoch=14,
    )
    epoch_rows = epoch_writer.rows()
    core_rows = core_writer.rows()
    _require_equal(len(epoch_rows), final_epoch, field="epoch_metrics.row_count")
    _require_equal(len(core_rows), final_epoch * 14, field="per_core.row_count")
    for epoch in range(1, final_epoch + 1):
        rows = [row for row in core_rows if int(row["global_epoch"]) == epoch]
        _require_equal(len(rows), 14, field=f"per_core.epoch_{epoch}.rows")
        _require_equal(
            {row["core_alias"] for row in rows if row["split"] == "training"},
            set(TRAINING_ALIASES),
            field=f"per_core.epoch_{epoch}.training_aliases",
        )
        _require_equal(
            {row["core_alias"] for row in rows if row["split"] == "validation"},
            set(VALIDATION_ALIASES),
            field=f"per_core.epoch_{epoch}.validation_aliases",
        )
    for relative, columns, rows_per_epoch in (
        (
            "results/gradient_direction_metrics.csv",
            GRADIENT_DIRECTION_METRICS_COLUMNS,
            1,
        ),
        (
            "results/gradient_direction_by_block.csv",
            BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
            4,
        ),
        (
            "results/gradient_direction_dispersion.csv",
            GRADIENT_DIRECTION_METRICS_COLUMNS,
            1,
        ),
    ):
        rows = DurableScalarCSV(
            scratch, relative, columns, rows_per_epoch=rows_per_epoch
        ).rows()
        _require_equal(
            len(rows), final_epoch * rows_per_epoch, field=f"{relative}.rows"
        )
        _require_equal(
            [int(row["global_epoch"]) for row in rows],
            [
                epoch
                for epoch in range(1, final_epoch + 1)
                for _ in range(rows_per_epoch)
            ],
            field=f"{relative}.epochs",
        )

    history = _read_runner_table(scratch, "metrics/history")
    _require_equal(len(history), final_epoch, field="metrics.history.rows")
    if any(row.get("run_id") != run_id for row in history):
        raise SO2NBTrainingError("History rows are not bound to the run.")
    predictions = _read_runner_table(scratch, "predictions/validation")
    validated_predictions = validate_prediction_rows(
        predictions, expected_run_id=run_id, expected_split="validation"
    )
    _require_equal(
        len(validated_predictions), len(VALIDATION_ALIASES), field="predictions.rows"
    )
    final_metrics = _load_json_mapping(
        scratch / "metrics/final.json", description="final metrics"
    )
    primary = float(summary.get("primary_metric_value", float("nan")))
    if (
        PRIMARY_METRIC not in final_metrics
        or not math.isfinite(primary)
        or not math.isclose(
            primary,
            float(final_metrics[PRIMARY_METRIC]),
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise SO2NBTrainingError("Summary and final primary metric disagree.")
    reload_receipt = _load_json_mapping(
        scratch / "diagnostics/final_checkpoint_reload_verification.json",
        description="best-checkpoint reload receipt",
    )
    _require_equal(reload_receipt.get("verified"), True, field="reload.verified")
    _require_equal(
        reload_receipt.get("checkpoint_sha256"), best_sha, field="reload.best_sha"
    )
    provenance = _load_json_mapping(
        scratch / "provenance/so2_target_isolated_nb_training.json",
        description="target-isolated training provenance",
    )
    for field, expected in {
        "protocol": PROTOCOL,
        "gradient_vectors_persisted": False,
        "full_prediction_matrices_persisted": False,
        "test_artifacts_present": False,
        "validation_prediction_summary_rows": 2,
    }.items():
        _require_equal(provenance.get(field), expected, field=f"provenance.{field}")
    _reconcile_metric_events(
        scratch / "metrics/events.jsonl", checkpoint_epoch=final_epoch
    )
    forbidden = [
        path
        for directory in (scratch / "metrics", scratch / "predictions")
        if directory.is_dir()
        for path in directory.glob("test*")
    ]
    if forbidden:
        raise SO2NBTrainingError("Test artifacts are forbidden for this protocol.")
    vector_files = [
        path
        for path in scratch.rglob("*")
        if path.is_file()
        and "gradient" in path.name.lower()
        and path.suffix.lower() not in {".csv"}
    ]
    if vector_files:
        raise SO2NBTrainingError("Gradient vectors or non-scalar gradient files exist.")
    return summary


def _remaining_training_epochs(state: EarlyStoppingState) -> range:
    start = int(state.completed_epoch)
    return range(start, start if state.should_stop else MAXIMUM_EPOCHS)


def run_distributed(
    args: argparse.Namespace,
    *,
    rank: int,
    local_rank: int,
    control_group: Any,
) -> Mapping[str, Any] | None:
    started = time.monotonic()
    run_id, scratch, config, paths = _load_worker_config(args)
    bundle = _load_data(config, paths)
    _require_equal(
        tuple(batch.alias for batch in bundle.training_batches),
        TRAINING_ALIASES,
        field="data.training_aliases",
    )
    _require_equal(
        tuple(batch.alias for batch in bundle.validation_batches),
        VALIDATION_ALIASES,
        field="data.validation_aliases",
    )
    _require_equal(SO2_NB_TEST_ALIASES, (), field="data.test_aliases")
    _require_equal(
        sum(batch.n_nodes for batch in bundle.training_batches),
        208_696,
        field="data.training_cells",
    )
    _require_equal(
        sum(batch.n_nodes for batch in bundle.validation_batches),
        37_367,
        field="data.validation_cells",
    )
    for batch in bundle.all_batches:
        _validate_cpu_graph(batch)

    model = _build_model(config, bundle)
    device = torch.device(f"cuda:{local_rank}")
    training_model, optimizer, scheduler, scaler = _initialize_training(
        model, config, device
    )
    preflight = _load_preflight_receipt(config, paths, bundle, model)
    preflight_sha = str(preflight["receipt_content_sha256"])
    trainer = _section(config, "trainer")
    config_sha = _canonical_sha256(config)

    archive: RunArchive | None = None
    checkpoint_store: AtomicBestLatestCheckpointStore | None = None
    epoch_writer: DurableScalarCSV | None = None
    core_writer: DurableScalarCSV | None = None
    global_writer: DurableScalarCSV | None = None
    block_writer: DurableScalarCSV | None = None
    theta_writer: DurableScalarCSV | None = None
    with _synchronized_rank_zero_phase(
        rank=rank,
        control_group=control_group,
        phase="target-isolated archive initialization",
    ):
        if rank == 0:
            archive = RunArchive.attach_active(
                run_id, paths=paths, scratch_path=scratch
            )
            _remove_interrupted_temps(scratch)
            _require_disk_headroom(scratch)
            checkpoint_store = AtomicBestLatestCheckpointStore(
                scratch,
                checkpoint_schema=CHECKPOINT_SCHEMA,
                protocol=PROTOCOL,
            )
            epoch_writer = DurableScalarCSV(
                scratch,
                "results/epoch_metrics.csv",
                EPOCH_COLUMNS,
                rows_per_epoch=1,
            )
            core_writer = DurableScalarCSV(
                scratch,
                "results/per_core_epoch_metrics.csv",
                PER_CORE_COLUMNS,
                rows_per_epoch=14,
            )
            global_writer = DurableScalarCSV(
                scratch,
                "results/gradient_direction_metrics.csv",
                GRADIENT_DIRECTION_METRICS_COLUMNS,
                rows_per_epoch=1,
            )
            block_writer = DurableScalarCSV(
                scratch,
                "results/gradient_direction_by_block.csv",
                BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
                rows_per_epoch=4,
            )
            theta_writer = DurableScalarCSV(
                scratch,
                "results/gradient_direction_dispersion.csv",
                GRADIENT_DIRECTION_METRICS_COLUMNS,
                rows_per_epoch=1,
            )

    finalized_summary: Mapping[str, Any] | None = None
    with _synchronized_rank_zero_phase(
        rank=rank,
        control_group=control_group,
        phase="target-isolated finalized-run reconciliation",
    ):
        if rank == 0:
            assert checkpoint_store is not None
            summary_exists = (scratch / "summary.json").is_file()
            best_exists = checkpoint_store.best_path.is_file()
            latest_exists = checkpoint_store.latest_path.is_file()
            if summary_exists:
                if not best_exists:
                    raise SO2NBTrainingError(
                        "Finalized summary exists without a best checkpoint."
                    )
                try:
                    finalized_summary = _validate_final_outputs(
                        scratch,
                        run_id=run_id,
                        checkpoint_store=checkpoint_store,
                        expect_latest=latest_exists,
                    )
                except SO2NBTrainingError:
                    if not latest_exists:
                        raise
                    # A latest checkpoint is still the transaction commit
                    # point, so conclusion outputs interrupted before their
                    # verification may be discarded and replayed safely.
                    _remove_partial_finalization_outputs(scratch)
                    finalized_summary = None
                else:
                    if latest_exists:
                        checkpoint_store.finalize_best_only()
                        finalized_summary = _validate_final_outputs(
                            scratch,
                            run_id=run_id,
                            checkpoint_store=checkpoint_store,
                            expect_latest=False,
                        )
            elif best_exists and not latest_exists:
                checkpoint_store.discard_best_without_latest()
    finalized_values: list[Any] = [finalized_summary if rank == 0 else None]
    torch.distributed.broadcast_object_list(
        finalized_values, src=0, group=control_group
    )
    if finalized_values[0] is not None:
        return dict(finalized_values[0]) if rank == 0 else None

    resume_path = args.resume_checkpoint
    if resume_path is None and (scratch / "checkpoints/latest.ckpt").is_file():
        resume_path = scratch / "checkpoints/latest.ckpt"
    patience_control_state = EarlyStoppingState()
    checkpoint_selection_state = EarlyStoppingState()
    fixed_receipts: Mapping[str, Mapping[str, Any]] | None = None
    start_epoch = 0
    if resume_path is not None:
        resume_path = resume_path.resolve(strict=True)
        canonical_latest = (scratch / "checkpoints/latest.ckpt").resolve(strict=True)
        if resume_path != canonical_latest:
            raise SO2NBTrainingError("Resume checkpoint must be this run's latest.ckpt.")
        with _synchronized_rank_zero_phase(
            rank=rank,
            control_group=control_group,
            phase="target-isolated checkpoint transaction reconciliation",
        ):
            if rank == 0:
                assert checkpoint_store is not None
                _reconcile_checkpoint_transaction(checkpoint_store)
        (
            patience_control_state,
            checkpoint_selection_state,
            fixed_receipts,
        ) = _load_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            bundle=bundle,
            rank=rank,
            configuration_sha256=config_sha,
            preflight_receipt_sha256=preflight_sha,
        )
        start_epoch = patience_control_state.completed_epoch
        _require_equal(
            checkpoint_selection_state.completed_epoch,
            start_epoch,
            field="resume.checkpoint_selection_epoch",
        )
        with _synchronized_rank_zero_phase(
            rank=rank,
            control_group=control_group,
            phase="target-isolated metric reconciliation",
        ):
            if rank == 0:
                assert epoch_writer and core_writer
                assert global_writer and block_writer and theta_writer
                for writer in (
                    epoch_writer,
                    core_writer,
                    global_writer,
                    block_writer,
                    theta_writer,
                ):
                    writer.reconcile(checkpoint_epoch=start_epoch)
                _reconcile_metric_events(
                    scratch / "metrics/events.jsonl", checkpoint_epoch=start_epoch
                )
                _remove_partial_finalization_outputs(scratch)
    else:
        with _synchronized_rank_zero_phase(
            rank=rank,
            control_group=control_group,
            phase="target-isolated fresh metric reconciliation",
        ):
            if rank == 0:
                assert epoch_writer and core_writer
                assert global_writer and block_writer and theta_writer
                for writer in (
                    epoch_writer,
                    core_writer,
                    global_writer,
                    block_writer,
                    theta_writer,
                ):
                    writer.reconcile(checkpoint_epoch=0)
                _reconcile_metric_events(
                    scratch / "metrics/events.jsonl", checkpoint_epoch=0
                )
                _remove_partial_finalization_outputs(scratch)

    resumed = start_epoch > 0
    global_tracker = FullGradientDirectionTracker(
        expected_optimizer_updates_per_epoch=OPTIMIZER_UPDATES_PER_EPOCH,
        resume_boundary_unavailable=resumed,
    )
    block_tracker = BlockGradientDirectionTracker(
        expected_blocks=4,
        expected_optimizer_updates_per_epoch=OPTIMIZER_UPDATES_PER_EPOCH,
        resume_boundary_unavailable=resumed,
    )
    theta_tracker = FullGradientDirectionTracker(
        expected_optimizer_updates_per_epoch=OPTIMIZER_UPDATES_PER_EPOCH,
        resume_boundary_unavailable=resumed,
    )
    theta_view = _RawThetaParameterView(model.raw_theta)

    for global_epoch in _remaining_training_epochs(patience_control_state):
        epoch_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        ordered = cohort_relative_qkv_core_order(
            global_epoch + 1,
            aliases=TRAINING_ALIASES,
            core_order_seed=CORE_ORDER_SEED,
        )
        if len(ordered) != 12 or set(ordered) != set(TRAINING_ALIASES):
            raise SO2NBTrainingError("Core order is not one complete permutation.")
        pairs = tuple(
            (ordered[index], ordered[index + 1])
            for index in range(0, len(ordered), 2)
        )
        if len(pairs) != OPTIMIZER_UPDATES_PER_EPOCH:
            raise SO2NBTrainingError("Each epoch must contain six paired updates.")

        local_training: dict[str, tuple[float, int, int]] = {}
        coverage_receipts: list[Mapping[str, Any]] = []
        for update_index, pair in enumerate(pairs):
            local, coverage, _ = _paired_update(
                model=model,
                training_model=training_model,
                optimizer=optimizer,
                scaler=scaler,
                pair=pair,
                update_index=update_index,
                global_epoch=global_epoch,
                rank=rank,
                device=device,
                batches=bundle.batches_by_alias,
                trainer=trainer,
                control_group=control_group,
                global_tracker=global_tracker,
                block_tracker=block_tracker,
                theta_tracker=theta_tracker,
                theta_view=theta_view,
            )
            if set(local_training).intersection(local):
                raise SO2NBTrainingError("A training core was visited twice in one epoch.")
            local_training.update(local)
            coverage_receipts.extend(coverage)

        train_nll, train_entries, train_targets, train_per_core = (
            _aggregate_training_epoch(local_training, device=device)
        )
        training_receipts = _validate_epoch_coverage(
            coverage_receipts,
            global_epoch=global_epoch + 1,
            control_group=control_group,
        )
        for alias in TRAINING_ALIASES:
            _require_equal(
                int(train_per_core[alias]["target_cells"]),
                int(training_receipts[alias]["target_cells"]),
                field=f"training.{alias}.target_cells",
            )
            _require_equal(
                int(train_per_core[alias]["masked_entries"]),
                int(training_receipts[alias]["masked_entries"]),
                field=f"training.{alias}.masked_entries",
            )

        global_summary = global_tracker.complete_epoch(global_epoch)
        block_summaries = block_tracker.complete_epoch(global_epoch)
        theta_summary = theta_tracker.complete_epoch(global_epoch)
        global_gradient = _gradient_row(global_summary, run_id=run_id)
        block_gradients = [
            _gradient_row(summary, run_id=run_id) for summary in block_summaries
        ]
        theta_gradient = _gradient_row(theta_summary, run_id=run_id)

        validation = _evaluate_validation(
            bundle=bundle,
            training_model=training_model,
            rank=rank,
            device=device,
            trainer=trainer,
            control_group=control_group,
        )
        if fixed_receipts is None:
            fixed_receipts = validation.coverage_receipts
        elif _canonical_sha256(fixed_receipts) != _canonical_sha256(
            validation.coverage_receipts
        ):
            raise SO2NBTrainingError("Fixed validation target masks changed.")

        scheduler.step(validation.pooled_negative_binomial_nll)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        def update_epoch_controls() -> Mapping[str, Any]:
            return {
                "patience_control": asdict(
                    update_early_stopping(
                        patience_control_state,
                        validation.pooled_negative_binomial_nll,
                        completed_epoch=global_epoch + 1,
                        minimum_epochs=MINIMUM_EPOCHS,
                        patience=EARLY_STOPPING_PATIENCE,
                        min_delta=EARLY_STOPPING_MIN_DELTA,
                        maximum_epochs=MAXIMUM_EPOCHS,
                    )
                ),
                "checkpoint_selection": asdict(
                    _update_exact_checkpoint_selection(
                        checkpoint_selection_state,
                        validation.pooled_negative_binomial_nll,
                        completed_epoch=global_epoch + 1,
                    )
                ),
            }

        control_value = _broadcast_rank_zero_result(
            rank=rank,
            control_group=control_group,
            phase=f"epoch-{global_epoch + 1} validation controls",
            operation=update_epoch_controls,
        )
        if not isinstance(control_value, Mapping):
            raise SO2NBTrainingError("Validation-control broadcast is malformed.")
        patience_value = control_value.get("patience_control")
        selection_value = control_value.get("checkpoint_selection")
        if not isinstance(patience_value, Mapping) or not isinstance(
            selection_value, Mapping
        ):
            raise SO2NBTrainingError("Validation-control states are malformed.")
        patience_control_state = EarlyStoppingState(**dict(patience_value))
        checkpoint_selection_state = EarlyStoppingState(**dict(selection_value))
        _require_equal(
            checkpoint_selection_state.completed_epoch,
            patience_control_state.completed_epoch,
            field="validation_controls.completed_epoch",
        )

        duration = time.monotonic() - epoch_started
        if not math.isfinite(duration) or duration <= 0.0:
            raise SO2NBTrainingError("Epoch duration is invalid.")
        peak_vram = _peak_vram_all_ranks(device)
        theta_values = _theta_summary(model)
        rng_states = _all_rank_rng_states(control_group=control_group)
        assert fixed_receipts is not None
        checkpoint_state = _checkpoint_state_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            completed_epoch=global_epoch + 1,
            checkpoint_selection_state=checkpoint_selection_state,
            patience_control_state=patience_control_state,
            bundle=bundle,
            rng_states=rng_states,
            validation_coverage_receipts=fixed_receipts,
            configuration_sha256=config_sha,
            preflight_receipt_sha256=preflight_sha,
        )
        state_digests = _all_gather_objects(
            _tree_sha256(checkpoint_state), control_group=control_group
        )
        if len(set(state_digests)) != 1:
            raise SO2NBTrainingError("Rank checkpoint states disagree at epoch boundary.")

        def persist_epoch() -> Mapping[str, Any]:
            assert archive and checkpoint_store and epoch_writer and core_writer
            assert global_writer and block_writer and theta_writer
            _require_disk_headroom(scratch)
            completed_epoch = global_epoch + 1
            epoch_writer.append_epoch(
                [
                    _make_epoch_row(
                        completed_epoch=completed_epoch,
                        train_nll=train_nll,
                        train_masked_entries=train_entries,
                        train_target_cells=train_targets,
                        validation=validation,
                        theta=theta_values,
                        learning_rate=learning_rate,
                        duration=duration,
                        peak_vram=peak_vram,
                        checkpoint_selection=checkpoint_selection_state,
                        patience_control=patience_control_state,
                        core_order=ordered,
                        training_receipts=training_receipts,
                        global_gradient=global_gradient,
                        block_gradients=block_gradients,
                        theta_gradient=theta_gradient,
                    )
                ],
                global_epoch=completed_epoch,
            )
            core_writer.append_epoch(
                _per_core_rows(
                    completed_epoch=completed_epoch,
                    training=train_per_core,
                    training_receipts=training_receipts,
                    validation=validation,
                ),
                global_epoch=completed_epoch,
            )
            global_writer.append_epoch(
                [global_gradient], global_epoch=completed_epoch
            )
            block_writer.append_epoch(
                block_gradients, global_epoch=completed_epoch
            )
            theta_writer.append_epoch(
                [theta_gradient], global_epoch=completed_epoch
            )
            archive.append_metric_event(
                {
                    "name": "train/pooled_masked_negative_binomial_nll",
                    "value": train_nll,
                    "step": completed_epoch,
                }
            )
            archive.append_metric_event(
                {
                    "name": PRIMARY_METRIC,
                    "value": validation.pooled_negative_binomial_nll,
                    "step": completed_epoch,
                }
            )
            if checkpoint_selection_state.improved:
                best_payload = _checkpoint_payload_from_state(
                    checkpoint_state,
                    role="best",
                    best_checkpoint_sha256=None,
                )
                best_receipt = checkpoint_store.save_best(best_payload)
                best_sha = best_receipt.sha256
            else:
                best_payload = checkpoint_store.load("best")
                best_sha = sha256_file(checkpoint_store.best_path)
                _require_equal(
                    best_payload.get("completed_epoch"),
                    checkpoint_selection_state.best_epoch,
                    field="best.completed_epoch",
                )
            latest_receipt = checkpoint_store.save_latest(
                _checkpoint_payload_from_state(
                    checkpoint_state,
                    role="latest",
                    best_checkpoint_sha256=best_sha,
                    embedded_best_checkpoint=best_payload,
                )
            )
            print(
                _canonical_json(
                    {
                        "epoch": completed_epoch,
                        "train_pooled_nb_nll": train_nll,
                        "validation_pooled_nb_nll": (
                            validation.pooled_negative_binomial_nll
                        ),
                        "training_target_cells": train_targets,
                        "validation_target_cells": validation.target_cells,
                        "best_epoch": checkpoint_selection_state.best_epoch,
                        "bad_validations": patience_control_state.bad_validations,
                        "learning_rate": learning_rate,
                        "peak_vram_gib": peak_vram,
                        "should_stop": patience_control_state.should_stop,
                    }
                ),
                flush=True,
            )
            return {
                "completed_epoch": completed_epoch,
                "latest_checkpoint_sha256": latest_receipt.sha256,
            }

        persistence = _broadcast_rank_zero_result(
            rank=rank,
            control_group=control_group,
            phase=f"epoch-{global_epoch + 1} persistence",
            operation=persist_epoch,
        )
        if not isinstance(persistence, Mapping):
            raise SO2NBTrainingError("Epoch-persistence result is malformed.")
        _require_equal(
            persistence.get("completed_epoch"),
            global_epoch + 1,
            field="persistence.completed_epoch",
        )
        del checkpoint_state
        if patience_control_state.should_stop:
            break

    if checkpoint_selection_state.best_epoch <= 0 or fixed_receipts is None:
        raise SO2NBTrainingError("Training ended without a validation best.")
    best_path = scratch / "checkpoints/best.ckpt"
    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    if not isinstance(best_payload, Mapping):
        raise SO2NBTrainingError("Best checkpoint is malformed before final replay.")
    for field, expected in {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "checkpoint_role": "best",
        "completed_epoch": checkpoint_selection_state.best_epoch,
        "configuration_sha256": config_sha,
    }.items():
        _require_equal(best_payload.get(field), expected, field=f"best.{field}")
    best_selection_payload = best_payload.get("early_stopping_state")
    best_control_payload = best_payload.get("patience_early_stopping_state")
    if not isinstance(best_selection_payload, Mapping) or not isinstance(
        best_control_payload, Mapping
    ):
        raise SO2NBTrainingError("Best checkpoint lacks both validation controls.")
    best_selection_state = EarlyStoppingState(**dict(best_selection_payload))
    best_control_state = EarlyStoppingState(**dict(best_control_payload))
    _require_equal(
        best_selection_state.completed_epoch,
        checkpoint_selection_state.best_epoch,
        field="best.checkpoint_selection_state.completed_epoch",
    )
    _require_equal(
        best_selection_state.best_value,
        checkpoint_selection_state.best_value,
        field="best.checkpoint_selection_state.best_value",
    )
    _require_equal(
        best_control_state.completed_epoch,
        checkpoint_selection_state.best_epoch,
        field="best.patience_control_state.completed_epoch",
    )
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    replay = _evaluate_validation(
        bundle=bundle,
        training_model=training_model,
        rank=rank,
        device=device,
        trainer=trainer,
        control_group=control_group,
    )
    if (
        not math.isclose(
            replay.pooled_negative_binomial_nll,
            float(checkpoint_selection_state.best_value),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or _canonical_sha256(replay.coverage_receipts)
        != _canonical_sha256(fixed_receipts)
    ):
        raise SO2NBTrainingError("Reloaded best checkpoint failed fixed-mask replay.")
    torch.distributed.barrier()
    if rank != 0:
        return None

    assert archive and checkpoint_store and epoch_writer
    final_checkpoint_sha = sha256_file(checkpoint_store.best_path)
    epoch_rows = epoch_writer.rows()
    archive.write_table(
        "metrics/history",
        _archive_history_rows(run_id, epoch_rows),
        fallback="jsonl",
    )
    prediction_rows = _validation_prediction_rows(
        run_id=run_id,
        config=config,
        bundle=bundle,
        validation=replay,
    )
    archive.write_predictions("validation", prediction_rows, fallback="jsonl")
    final_metrics = {
        PRIMARY_METRIC: replay.pooled_negative_binomial_nll,
        "val/unseen_donor/raw_count_mae": replay.raw_count_mae,
        "val/unseen_donor/raw_count_rmse": replay.raw_count_rmse,
        "val/unseen_donor/log1p_mae": replay.log1p_mae,
        "val/unseen_donor/log1p_rmse": replay.log1p_rmse,
        "val/unseen_donor/poisson_deviance": replay.poisson_deviance,
        "val/unseen_donor/observed_zero_rate": replay.observed_zero_rate,
        "val/unseen_donor/predicted_zero_probability_mean": (
            replay.predicted_zero_probability_mean
        ),
        "val/unseen_donor/zero_brier_score": replay.zero_brier_score,
        "val/masked_entries": replay.masked_entries,
        "val/target_cells": replay.target_cells,
    }
    archive.write_json("metrics/final.json", final_metrics)
    archive.write_json(
        "diagnostics/final_checkpoint_reload_verification.json",
        {
            "verified": True,
            "checkpoint": "checkpoints/best.ckpt",
            "checkpoint_sha256": final_checkpoint_sha,
            "best_epoch": checkpoint_selection_state.best_epoch,
            "best_validation_metric": checkpoint_selection_state.best_value,
            "replayed_validation_metric": replay.pooled_negative_binomial_nll,
            "validation_coverage_receipts_sha256": _canonical_sha256(
                fixed_receipts
            ),
        },
    )
    archive.write_json("diagnostics/hardware_preflight.json", dict(preflight))
    provenance = {
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "model_name": MODEL_NAME,
        "training_aliases": list(TRAINING_ALIASES),
        "validation_aliases": list(VALIDATION_ALIASES),
        "test_aliases": [],
        "training_cells": 208_696,
        "validation_cells": 37_367,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "frozen_task_contract_sha256": FROZEN_TASK_CONTRACT_SHA256,
        "preflight_receipt_sha256": preflight_sha,
        "validation_coverage_receipts_sha256": _canonical_sha256(fixed_receipts),
        "checkpoint_sha256": final_checkpoint_sha,
        "checkpoint_layout": "final_best_only_after_verification",
        "target_partitioning": "four_contiguous_rank_shards_per_core",
        "paired_update_sync": "first_core_no_sync_second_core_synced",
        "objective_aggregation": "exact_global_masked_entry_weighted",
        "gradient_vectors_persisted": False,
        "full_prediction_matrices_persisted": False,
        "validation_prediction_summary_rows": len(prediction_rows),
        "test_artifacts_present": False,
    }
    archive.write_json(
        "provenance/so2_target_isolated_nb_training.json", provenance
    )
    summary = {
        "run_id": run_id,
        "status": "success",
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "model_name": MODEL_NAME,
        "model_seed": MODEL_SEED,
        "final_epoch": patience_control_state.completed_epoch,
        "best_epoch": checkpoint_selection_state.best_epoch,
        "checkpoint_selection_state": asdict(checkpoint_selection_state),
        "patience_control_state": asdict(patience_control_state),
        "optimizer_steps": patience_control_state.completed_epoch
        * OPTIMIZER_UPDATES_PER_EPOCH,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_value": replay.pooled_negative_binomial_nll,
        "checkpoint": "checkpoints/best.ckpt",
        "checkpoint_reload_verified": True,
        "checkpoint_count": 1,
        "epoch_metrics_rows": len(epoch_rows),
        "loss_recorded_every_epoch": True,
        "gradient_direction_recorded_every_epoch": True,
        "target_coverage_verified_every_epoch": True,
        "stopped_early": patience_control_state.stop_reason == "patience",
        "stop_reason": patience_control_state.stop_reason,
        "peak_vram_gib": max(
            float(row["peak_vram_gib_all_ranks"]) for row in epoch_rows
        ),
        "duration_seconds": time.monotonic() - started,
        "world_size": WORLD_SIZE,
        "generalization_estimate": False,
        "unbiased_test_estimate": False,
        "test_artifacts_present": False,
    }
    archive.write_summary(summary)
    _require_equal(
        sha256_file(checkpoint_store.best_path),
        final_checkpoint_sha,
        field="final.checkpoint_sha256",
    )
    _validate_final_outputs(
        scratch,
        run_id=run_id,
        checkpoint_store=checkpoint_store,
        expect_latest=True,
        expected_summary=summary,
    )
    checkpoint_store.finalize_best_only()
    _validate_final_outputs(
        scratch,
        run_id=run_id,
        checkpoint_store=checkpoint_store,
        expect_latest=False,
        expected_summary=summary,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train target-isolated SO2 C15--C26 and validate C27--C28 with "
            "four-rank masked-entry-pooled NB2 DDP."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", type=Path)
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
    control_group = torch.distributed.new_group(
        ranks=list(range(WORLD_SIZE)),
        backend="gloo",
        timeout=timedelta(minutes=5),
    )
    try:
        summary = run_distributed(
            args,
            rank=rank,
            local_rank=local_rank,
            control_group=control_group,
        )
        if rank == 0 and summary is not None:
            print(_canonical_json(summary), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
