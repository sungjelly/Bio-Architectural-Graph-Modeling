#!/usr/bin/env python3
"""Train one configured six-core relative-QKV model to loss plateau."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
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


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from scripts.train.run_full_core_capacity import _worker_archive_and_config  # noqa: E402
from spatial_benchmark.adjacency_ablation import (  # noqa: E402
    sample_uniform_mask_numpy,
)
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES  # noqa: E402
from spatial_benchmark.cancer_relative_graphs import (  # noqa: E402
    load_cancer_relative_qkv_batches,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.masking import derive_mask_seed  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training import (  # noqa: E402
    MASK_BASE_SEED,
    PooledRelativeQKVCoreStepRecord,
    PooledRelativeQKVEpochBoundaryResume,
    PooledRelativeQKVGlobalEpochRecord,
    PooledRelativeQKVTrainingConfig,
    epoch_boundary_resume_from_checkpoint,
    fit_pooled_relative_qkv_segment,
    single_seed_plateau_decision,
)
from spatial_benchmark.relative_qkv_graph_transformer import (  # noqa: E402
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    deidentify_prediction_rows,
)
from spatial_benchmark.training import (  # noqa: E402
    _autocast_context,
    set_deterministic_seed,
)


CAMPAIGN_ID = "cmp_20260824_cancer_6core_relative_qkv_multiseed"
ACTIVE_PROTOCOL = "held_in_pooled_6core_relative_qkv_seed_plateau"
ACTIVE_MODEL_SEEDS = (0, 1, 2, 3)
DEFERRED_MODEL_SEEDS = (4,)
PREFLIGHT_REFERENCE_SEED = 0
ACTIVE_AMENDMENT_SHA256 = (
    "e07c4e8d9d66df8d2c6063d58e4424950b156002a91e811c474d3b9b077c9429"
)
ACTIVE_AMENDMENT = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "task_contract_amendment_006_four_seeds_resource_gated.yaml"
)
SEED0_FIRST_AMENDMENT_SHA256 = (
    "0ad3c6373edc45b1c61649217f043f486cfe3cfcf925bad042a2ed635032d174"
)
HELD_IN_MASK_BASE_SEED = 2026082491
HARDWARE_PREFLIGHT_RECEIPT = Path(
    "state/preflight/cancer_6core_relative_qkv_seed0.json"
)
HARDWARE_PREFLIGHT_SCHEMA = "cancer_6core_relative_qkv_hardware_preflight_v1"
PREFLIGHT_BASE_ATTEMPT = 1
TRAINING_MONITOR_SCHEMA = "relative_qkv_training_epoch_monitor_v1"


class RelativeQKVRunnerError(RuntimeError):
    """Raised before an active per-seed run can violate its contract."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise RelativeQKVRunnerError(f"Resolved config requires {name!r} mapping.")
    return value


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise RelativeQKVRunnerError(
            f"{field} must be {expected!r}; got {actual!r}."
        )


def _configured_model_seed(config: Mapping[str, Any]) -> int:
    value = config.get("seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise RelativeQKVRunnerError("seed must be an integer model seed.")
    seed = int(value)
    if seed not in ACTIVE_MODEL_SEEDS:
        raise RelativeQKVRunnerError(
            f"seed must be one of {ACTIVE_MODEL_SEEDS}; got {seed!r}."
        )
    return seed


def _validate_active_contract(config: Mapping[str, Any]) -> None:
    _configured_model_seed(config)
    campaign = _section(config, "campaign")
    _require_equal(campaign.get("campaign_id"), CAMPAIGN_ID, field="campaign_id")
    evaluation = _section(config, "evaluation")
    _require_equal(
        evaluation.get("protocol"), ACTIVE_PROTOCOL, field="evaluation.protocol"
    )
    model = _section(config, "model")
    locked_model = {
        "name": "relative-qkv-gat",
        "hidden_dim": 256,
        "graph_layers": 4,
        "attention_heads": 8,
        "attention_head_dim": 32,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "relative_geometry_dim": 70,
        "attention_dropout": 0.0,
        "activation_checkpointing": True,
        "uses_edge_inputs": False,
    }
    for field, expected in locked_model.items():
        _require_equal(model.get(field), expected, field=f"model.{field}")
    masking = _section(config, "masking")
    _require_equal(
        masking.get("independent_views_per_core_epoch"),
        10,
        field="masking.independent_views_per_core_epoch",
    )
    _require_equal(
        masking.get("model_seed_in_mask_derivation"),
        False,
        field="masking.model_seed_in_mask_derivation",
    )
    trainer = _section(config, "trainer")
    locked_trainer = {
        "minimum_global_epochs": 150,
        "continuation_block_global_epochs": 25,
        "mask_views_per_core_step": 10,
        "optimizer_steps_per_core_step": 1,
        "stage_complete_core_graph_on_device": True,
        "staged_relative_geometry_dtype": "float16",
        "early_stopping": False,
        "restore_best": False,
    }
    for field, expected in locked_trainer.items():
        _require_equal(trainer.get(field), expected, field=f"trainer.{field}")
    amendment = _PROJECT_ROOT / ACTIVE_AMENDMENT
    if not amendment.is_file() or sha256_file(amendment) != ACTIVE_AMENDMENT_SHA256:
        raise RelativeQKVRunnerError("Active task amendment checksum mismatch.")


def _validate_hardware_preflight(
    config: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Require a checksum-valid preflight bound to current data and execution limits."""

    path = (
        (_PROJECT_ROOT / HARDWARE_PREFLIGHT_RECEIPT).resolve()
        if receipt_path is None
        else receipt_path.resolve()
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RelativeQKVRunnerError(
            f"A passing hardware preflight receipt is required: {path}."
        ) from exc
    if not isinstance(value, dict):
        raise RelativeQKVRunnerError("Hardware preflight receipt must be a mapping.")
    checksum = value.get("receipt_content_sha256")
    content = dict(value)
    content.pop("receipt_content_sha256", None)
    observed = hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if checksum != observed:
        raise RelativeQKVRunnerError("Hardware preflight receipt checksum mismatch.")
    if (
        value.get("schema") != HARDWARE_PREFLIGHT_SCHEMA
        or value.get("status") != "passed"
        or value.get("all_required_gates_passed") is not True
        or value.get("completed_experiment") is not False
    ):
        raise RelativeQKVRunnerError("Hardware preflight did not pass every gate.")
    # A queue retry changes only the execution attempt.  The passing preflight
    # is bound to the root attempt's otherwise-identical resolved scientific
    # and execution contract, so retries do not require rerunning an expensive
    # largest-core hardware diagnostic.
    preflight_config = _preflight_bound_config(config)
    resolved_config_sha256 = hashlib.sha256(
        json.dumps(
            preflight_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if value.get("resolved_config_sha256") != resolved_config_sha256:
        raise RelativeQKVRunnerError("Resolved configuration changed after preflight.")
    dataset = _section(config, "dataset")
    cohort_dir = (_PROJECT_ROOT / str(dataset["prepared_artifact"])).resolve()
    graph_dir = (_PROJECT_ROOT / str(dataset["prepared_graph_artifact"])).resolve()
    if value.get("cohort_manifest_sha256") != sha256_file(
        cohort_dir / "manifest.json"
    ):
        raise RelativeQKVRunnerError("Preflight cohort manifest has changed.")
    if value.get("graph_manifest_sha256") != sha256_file(graph_dir / "manifest.json"):
        raise RelativeQKVRunnerError("Preflight graph manifest has changed.")
    model = _section(config, "model")
    receipt_model = value.get("model")
    if not isinstance(receipt_model, Mapping):
        raise RelativeQKVRunnerError("Preflight receipt lacks model execution settings.")
    for field in (
        "receiver_chunk_size",
        "max_edges_per_chunk",
        "activation_checkpointing",
    ):
        if receipt_model.get(field) != model.get(field):
            raise RelativeQKVRunnerError(
                f"Preflight execution setting {field} no longer matches the run."
            )
    if receipt_model.get("stage_complete_core_graph_on_device") != _section(
        config, "trainer"
    ).get("stage_complete_core_graph_on_device"):
        raise RelativeQKVRunnerError(
            "Preflight graph-staging strategy no longer matches the run."
        )
    if receipt_model.get("staged_relative_geometry_dtype") != _section(
        config, "trainer"
    ).get("staged_relative_geometry_dtype"):
        raise RelativeQKVRunnerError(
            "Preflight staged geometry dtype no longer matches the run."
        )
    return value


def _preflight_bound_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize seed/retry routing while binding every contract setting.

    The expensive hardware preflight is model-seed neutral: seeds 0 through 3
    share it when, and only when, all scientific, model, and execution fields
    are otherwise identical.  Retry attempt and resume path are execution
    routing rather than properties exercised by the preflight.
    """

    normalized = dict(config)
    normalized["seed"] = PREFLIGHT_REFERENCE_SEED
    normalized["attempt"] = PREFLIGHT_BASE_ATTEMPT
    trainer = normalized.get("trainer")
    if isinstance(trainer, Mapping):
        normalized_trainer = dict(trainer)
        normalized_trainer["continuation_policy"] = (
            "seed0_training_loss_plateau_25_epoch_blocks"
        )
        normalized["trainer"] = normalized_trainer
    evaluation = normalized.get("evaluation")
    if isinstance(evaluation, Mapping):
        normalized_evaluation = dict(evaluation)
        normalized_evaluation["active_model_seeds"] = [0]
        normalized_evaluation["deferred_model_seeds"] = [1, 2, 3, 4]
        normalized["evaluation"] = normalized_evaluation
    launcher = normalized.get("launcher")
    if isinstance(launcher, Mapping):
        normalized_launcher = dict(launcher)
        normalized_launcher["requested_gpu"] = str(PREFLIGHT_REFERENCE_SEED)
        normalized_launcher.pop("resume_checkpoint", None)
        normalized["launcher"] = normalized_launcher
    return normalized


def _model_from_config(
    config: Mapping[str, Any],
    *,
    num_genes: int,
    node_covariate_dim: int,
) -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    model = _section(config, "model")
    return ReceiverChunkedRelativeGeometryQKVGraphTransformer(
        num_genes=num_genes,
        node_covariate_dim=node_covariate_dim,
        hidden_dim=int(model["hidden_dim"]),
        attention_heads=int(model["attention_heads"]),
        attention_head_dim=int(model["attention_head_dim"]),
        graph_layers=int(model["graph_layers"]),
        ffn_dim=int(model["ffn_dim"]),
        decoder_dim=int(model["decoder_dim"]),
        positional_bias_hidden_dim=int(model["positional_bias_hidden_dim"]),
        dropout=float(model["dropout"]),
        attention_dropout=float(model["attention_dropout"]),
        relative_geometry_dim=int(model["relative_geometry_dim"]),
        receiver_chunk_size=int(model["receiver_chunk_size"]),
        max_edges_per_chunk=int(model["max_edges_per_chunk"]),
        activation_checkpointing=bool(model["activation_checkpointing"]),
    )


def _seeded_model_from_config(
    config: Mapping[str, Any],
    *,
    num_genes: int,
    node_covariate_dim: int,
) -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    """Construct the production model after installing its configured seed."""

    trainer = _section(config, "trainer")
    # Parameters are initialized inside module constructors.  Installing the
    # model seed only when the trainer starts would be too late.
    set_deterministic_seed(
        _configured_model_seed(config),
        deterministic=bool(trainer["deterministic"]),
        warn_only=bool(trainer["deterministic_warn_only"]),
    )
    return _model_from_config(
        config,
        num_genes=num_genes,
        node_covariate_dim=node_covariate_dim,
    )


def _training_config(
    config: Mapping[str, Any],
    *,
    start_epoch: int,
    end_epoch: int,
) -> PooledRelativeQKVTrainingConfig:
    trainer = _section(config, "trainer")
    masking = _section(config, "masking")
    return PooledRelativeQKVTrainingConfig(
        model_seed=_configured_model_seed(config),
        segment_start_global_epoch=start_epoch,
        segment_end_global_epoch=end_epoch,
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        stage_complete_core_graph_on_device=bool(
            trainer["stage_complete_core_graph_on_device"]
        ),
        staged_relative_geometry_dtype=str(
            trainer["staged_relative_geometry_dtype"]
        ),
        checkpoint_interval_global_epochs=int(
            trainer["checkpoint_every_global_epochs"]
        ),
    )


def _checkpoint_payload(
    *,
    archive: RunArchive,
    config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    parameter_count: int,
    resume: PooledRelativeQKVEpochBoundaryResume,
    plateau: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_seed = _configured_model_seed(config)
    _require_equal(resume.model_seed, model_seed, field="resume.model_seed")
    return {
        "checkpoint_schema": "cancer_6core_relative_qkv_resume_v1",
        "run_id": archive.run_id,
        "campaign_id": CAMPAIGN_ID,
        "model_seed": model_seed,
        "completed_global_epochs": resume.completed_global_epochs,
        "optimizer_steps_completed": resume.optimizer_steps_completed,
        "mask_base_seed": resume.mask_base_seed,
        "core_order_seed": resume.core_order_seed,
        "mask_views_per_core_step": resume.mask_views_per_core_step,
        "model_construction": dict(model_construction),
        "parameter_count": int(parameter_count),
        "model_state_dict": resume.model_state_dict,
        "model_state_checksum": resume.model_state_checksum,
        "optimizer_state_dict": resume.optimizer_state_dict,
        "optimizer_state_checksum": resume.optimizer_state_checksum,
        "amp_scaler_state_dict": resume.scaler_state_dict,
        "amp_scaler_state_checksum": resume.scaler_state_checksum,
        "core_history": [asdict(record) for record in resume.core_history],
        "global_history": [asdict(record) for record in resume.global_history],
        "history_checksum": resume.history_checksum,
        "resume_checksum": resume.resume_checksum,
        "resolved_config": dict(config),
        "active_amendment_sha256": ACTIVE_AMENDMENT_SHA256,
        "plateau": None if plateau is None else dict(plateau),
    }


def _checkpoint_bytes(payload: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    return stream.getvalue()


def _load_resume_checkpoint(
    path: Path,
    *,
    config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
) -> tuple[PooledRelativeQKVEpochBoundaryResume, dict[str, Any]]:
    """Load a checksum-valid periodic checkpoint for a queue recovery run."""

    resolved = path.expanduser().resolve(strict=True)
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RelativeQKVRunnerError(
            f"Cannot load epoch-boundary resume checkpoint: {resolved}."
        ) from exc
    if not isinstance(value, Mapping):
        raise RelativeQKVRunnerError("Resume checkpoint must contain a mapping.")
    payload = dict(value)
    _require_equal(
        payload.get("checkpoint_schema"),
        "cancer_6core_relative_qkv_resume_v1",
        field="resume.checkpoint_schema",
    )
    _require_equal(payload.get("campaign_id"), CAMPAIGN_ID, field="resume.campaign_id")
    _require_equal(
        payload.get("model_seed"),
        _configured_model_seed(config),
        field="resume.model_seed",
    )
    source_amendment = payload.get("active_amendment_sha256")
    compatible_amendments = {ACTIVE_AMENDMENT_SHA256}
    if _configured_model_seed(config) == 0:
        compatible_amendments.add(SEED0_FIRST_AMENDMENT_SHA256)
    if source_amendment not in compatible_amendments:
        raise RelativeQKVRunnerError(
            "resume.active_amendment_sha256 is incompatible with the active seed."
        )
    if payload.get("model_construction") != dict(model_construction):
        raise RelativeQKVRunnerError(
            "Resume checkpoint model construction differs from the active model."
        )
    source_config = payload.get("resolved_config")
    if not isinstance(source_config, Mapping) or _preflight_bound_config(
        source_config
    ) != _preflight_bound_config(config):
        raise RelativeQKVRunnerError(
            "Resume checkpoint resolved configuration differs from the active contract."
        )
    resume = epoch_boundary_resume_from_checkpoint(payload)
    if resume.completed_global_epochs % int(
        _section(config, "trainer")["checkpoint_every_global_epochs"]
    ):
        raise RelativeQKVRunnerError(
            "Resume checkpoint is not at a locked 25-epoch boundary."
        )
    plateau = payload.get("plateau")
    if isinstance(plateau, Mapping) and plateau.get("should_stop") is True:
        raise RelativeQKVRunnerError(
            "A plateau-confirmed final checkpoint must be used directly, not resumed."
        )
    return resume, {
        "source_checkpoint": str(resolved),
        "source_checkpoint_sha256": sha256_file(resolved),
        "source_run_id": payload.get("run_id"),
        "completed_global_epochs": resume.completed_global_epochs,
        "resume_checksum": resume.resume_checksum,
    }


def _seed_plateau_decision(
    losses: Sequence[float],
    *,
    model_seed: int = PREFLIGHT_REFERENCE_SEED,
) -> dict[str, Any]:
    decision = single_seed_plateau_decision(
        losses,
        model_seed=int(model_seed),
        completed_global_epochs=len(losses),
    )
    return {
        **asdict(decision),
        "validation_or_test_metric": False,
        "checkpoint_selection_metric": False,
    }


def _training_monitor_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the live metric semantics before the first optimizer step."""

    trainer = _section(config, "trainer")
    return {
        "schema": TRAINING_MONITOR_SCHEMA,
        "event_file": "metrics/events.jsonl",
        "stdout_prefix": "[bagm-training]",
        "event_frequency": "after_each_complete_global_epoch",
        "global_epoch_definition": {
            "complete_core_optimizer_steps": len(CANCER_ALIASES),
            "mask_views_per_core_step": int(trainer["mask_views_per_core_step"]),
            "optimizer_steps_per_core_step": int(
                trainer["optimizer_steps_per_core_step"]
            ),
        },
        "training_objective": {
            "name": "equal_core_mean_masked_huber",
            "event_name": "fit/training/equal_core_mean_masked_huber",
            "direction": "minimize",
            "elementwise_loss": "Huber",
            "huber_delta": float(trainer["huber_delta"]),
            "target_scale": "shared_standardized_log1p_raw_counts",
            "mask_scope": "masked_entries_only",
            "view_aggregation": "arithmetic_mean_across_10_mask_views",
            "core_aggregation": "equal_core_arithmetic_mean_across_6_cores",
        },
        "live_diagnostics": [
            "per_core_masked_huber",
            "mean_gradient_norm_before_clip",
            "max_gradient_norm_before_clip",
            "learning_rate",
            "epoch_duration_seconds",
            "process_mean_epoch_duration_seconds",
            "rolling_5_epoch_duration_seconds",
            "eta_to_current_audit_seconds",
            "masked_entries_per_second",
            "process_peak_cuda_memory_allocated_gib",
        ],
        "gradient_clip_norm": float(trainer["gradient_clip_norm"]),
        "duration_scope": (
            "six complete-core steps and 60 mask views; excludes monitoring "
            "writes and periodic checkpoint serialization"
        ),
        "timing_in_checkpoint_checksum": False,
        "validation_or_test_partition_present": False,
        "post_training_fixed_mask_metrics": [
            "masked_huber",
            "masked_mae",
            "masked_mse",
            "masked_r2",
        ],
    }


def _training_progress_record(
    *,
    run_id: str,
    model_seed: int,
    epoch: PooledRelativeQKVGlobalEpochRecord,
    core_steps: Sequence[PooledRelativeQKVCoreStepRecord],
    epoch_duration_seconds: float,
    observed_epoch_durations: Sequence[float],
    segment_end_global_epoch: int,
    learning_rate: float,
    gradient_clip_norm: float,
    process_peak_cuda_memory_allocated_gib: float,
) -> dict[str, Any]:
    """Build one finite, aligned, human- and machine-readable epoch record."""

    records = tuple(core_steps)
    if len(records) != len(CANCER_ALIASES) or tuple(
        record.alias for record in records
    ) != tuple(epoch.ordered_aliases):
        raise RelativeQKVRunnerError(
            "Training monitor requires six core steps aligned to epoch order."
        )
    durations = np.asarray(observed_epoch_durations, dtype=np.float64)
    if durations.size == 0 or not np.isfinite(durations).all() or bool(
        (durations < 0).any()
    ):
        raise RelativeQKVRunnerError(
            "Training monitor durations must be finite and non-negative."
        )
    duration = float(epoch_duration_seconds)
    if not math.isfinite(duration) or duration < 0 or duration != float(
        durations[-1]
    ):
        raise RelativeQKVRunnerError(
            "Current epoch duration must be the last observed duration."
        )
    losses = np.asarray(
        [record.masked_huber_loss for record in records], dtype=np.float64
    )
    gradients = np.asarray(
        [record.gradient_norm for record in records], dtype=np.float64
    )
    scalar_values = np.concatenate(
        [
            losses,
            gradients,
            np.asarray(
                [
                    epoch.equal_core_mean_masked_huber,
                    learning_rate,
                    gradient_clip_norm,
                    process_peak_cuda_memory_allocated_gib,
                ],
                dtype=np.float64,
            ),
        ]
    )
    if not np.isfinite(scalar_values).all() or bool((gradients < 0).any()):
        raise RelativeQKVRunnerError(
            "Training monitor loss, gradient, and resource values must be finite."
        )
    if learning_rate <= 0 or gradient_clip_norm <= 0:
        raise RelativeQKVRunnerError(
            "Training monitor optimizer settings must be positive."
        )
    if process_peak_cuda_memory_allocated_gib < 0:
        raise RelativeQKVRunnerError(
            "Training monitor CUDA allocation cannot be negative."
        )
    completed = int(epoch.completed_global_epochs)
    segment_end = int(segment_end_global_epoch)
    if segment_end < completed:
        raise RelativeQKVRunnerError(
            "Training monitor segment end precedes the completed epoch."
        )
    rolling_duration = float(durations[-5:].mean())
    remaining_epochs = segment_end - completed
    masked_entries = sum(
        int(record.n_masked_entries_across_views) for record in records
    )
    nodes_across_views = sum(
        int(record.n_nodes) * int(record.n_mask_views) for record in records
    )
    edges_across_views = sum(
        int(record.n_edges) * int(record.n_mask_views) for record in records
    )
    masked_entries_per_second = (
        0.0 if duration == 0.0 else masked_entries / duration
    )
    return {
        "schema": TRAINING_MONITOR_SCHEMA,
        "run_id": str(run_id),
        "model_seed": int(model_seed),
        "global_epoch": int(epoch.global_epoch),
        "completed_global_epochs": completed,
        "current_segment_end_global_epoch": segment_end,
        "remaining_epochs_to_current_audit": remaining_epochs,
        "optimizer_steps_this_epoch": int(epoch.optimizer_steps_this_epoch),
        "cumulative_optimizer_steps": int(epoch.cumulative_optimizer_steps),
        "loss_name": "equal_core_mean_masked_huber",
        "equal_core_mean_masked_huber": float(
            epoch.equal_core_mean_masked_huber
        ),
        "per_core_masked_huber": {
            record.alias: float(record.masked_huber_loss) for record in records
        },
        "mean_gradient_norm_before_clip": float(gradients.mean()),
        "max_gradient_norm_before_clip": float(gradients.max()),
        "gradient_clip_norm": float(gradient_clip_norm),
        "learning_rate": float(learning_rate),
        "epoch_duration_seconds": duration,
        "process_mean_epoch_duration_seconds": float(durations.mean()),
        "rolling_5_epoch_duration_seconds": rolling_duration,
        "eta_to_current_audit_seconds": float(
            remaining_epochs * rolling_duration
        ),
        "mask_views_completed": sum(
            int(record.n_mask_views) for record in records
        ),
        "masked_entries_across_views": masked_entries,
        "nodes_across_views": nodes_across_views,
        "edges_across_views": edges_across_views,
        "masked_entries_per_second": float(masked_entries_per_second),
        "process_peak_cuda_memory_allocated_gib": float(
            process_peak_cuda_memory_allocated_gib
        ),
        "validation_or_test_metric": False,
        "checkpoint_selection_metric": False,
    }


def _emit_training_progress(
    archive: RunArchive,
    progress: Mapping[str, Any],
) -> None:
    """Durably append and immediately print one completed-epoch observation."""

    archive.append_metric_event(
        {
            "name": "fit/training/equal_core_mean_masked_huber",
            "value": progress["equal_core_mean_masked_huber"],
            "step": progress["completed_global_epochs"],
            "phase": "training_epoch",
            "monitor": dict(progress),
        }
    )
    stdout_progress = {
        key: progress[key]
        for key in (
            "run_id",
            "model_seed",
            "completed_global_epochs",
            "current_segment_end_global_epoch",
            "equal_core_mean_masked_huber",
            "epoch_duration_seconds",
            "rolling_5_epoch_duration_seconds",
            "eta_to_current_audit_seconds",
            "mean_gradient_norm_before_clip",
            "max_gradient_norm_before_clip",
            "process_peak_cuda_memory_allocated_gib",
        )
    }
    print(
        "[bagm-training] "
        + json.dumps(stdout_progress, sort_keys=True, allow_nan=False),
        flush=True,
    )


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
                "relative-qkv-held-in-fit-diagnostic",
                batch.alias,
            )
            realization = sample_uniform_mask_numpy(
                batch.n_nodes,
                batch.n_genes,
                seed=seed,
            )
            retry = 0
            while int(realization.masked_gene_counts.sum()) == 0:
                retry += 1
                seed = derive_mask_seed(seed, "nonzero-held-in", retry)
                realization = sample_uniform_mask_numpy(
                    batch.n_nodes,
                    batch.n_genes,
                    seed=seed,
                )
            target = batch.target_expression.to(device=device)
            covariates = batch.node_covariates.to(device=device)
            mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(
                device=device, dtype=torch.bool
            )
            with _autocast_context(
                enabled=amp,
                device=device,
                dtype_name=amp_dtype,
            ):
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
                selected_prediction,
                selected_truth,
                delta=1.0,
                reduction="mean",
            )
            mae = residual.abs().mean()
            mse = residual.square().mean()
            centered = selected_truth - selected_truth.mean()
            denominator = centered.square().sum()
            r2 = 1.0 - residual.square().sum() / denominator.clamp_min(
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
                    "dataset_id": "cosmx_cancer_6core_pooled_fit_v1",
                    "split": "fit",
                    "y_true": values["mean_y_true"],
                    "y_pred": values["mean_y_pred"],
                    "sample_loss": values["masked_huber"],
                    "node_count": batch.n_nodes,
                    "edge_count": batch.n_edges,
                }
            )
            del target, covariates, mask, output, prediction, truth
            if device.type == "cuda":
                torch.cuda.empty_cache()
    metrics = {
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
    }
    return metrics, core_metrics, prediction_rows


def run_seed_to_plateau(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train the configured seed until two per-seed plateau audits pass."""

    started = time.monotonic()
    _validate_active_contract(config)
    model_seed = _configured_model_seed(config)
    preflight = _validate_hardware_preflight(config)
    dataset = _section(config, "dataset")
    cohort_dir = (_PROJECT_ROOT / str(dataset["prepared_artifact"])).resolve()
    graph_dir = (_PROJECT_ROOT / str(dataset["prepared_graph_artifact"])).resolve()
    batches = load_cancer_relative_qkv_batches(
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    if tuple(batch.alias for batch in batches) != CANCER_ALIASES:
        raise RelativeQKVRunnerError("Loaded batches do not match the six-core order.")
    archive.write_json("diagnostics/hardware_preflight.json", preflight)
    archive.write_json(
        "metrics/training_monitor_contract.json",
        _training_monitor_contract(config),
    )
    model = _seeded_model_from_config(
        config,
        num_genes=batches[0].n_genes,
        node_covariate_dim=int(batches[0].node_covariates.shape[1]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_config = _section(config, "model")
    model_construction = {
        "class": type(model).__name__,
        "num_genes": batches[0].n_genes,
        "node_covariate_dim": int(batches[0].node_covariates.shape[1]),
        **dict(model_config),
    }

    configured_resume = _section(config, "launcher").get("resume_checkpoint")
    if resume_checkpoint is not None and configured_resume is not None:
        raise RelativeQKVRunnerError(
            "Specify a resume checkpoint through either CLI or launcher config, not both."
        )
    resume_path = (
        resume_checkpoint
        if resume_checkpoint is not None
        else (None if configured_resume is None else Path(str(configured_resume)))
    )
    resume_receipt: dict[str, Any] | None = None
    resume: PooledRelativeQKVEpochBoundaryResume | None = None
    if resume_path is not None:
        if not resume_path.is_absolute():
            resume_path = _PROJECT_ROOT / resume_path
        resume, resume_receipt = _load_resume_checkpoint(
            resume_path,
            config=config,
            model_construction=model_construction,
        )
        archive.write_json("provenance/resume_source.json", resume_receipt)

    written_epochs: set[int] = set()

    def save_periodic(resume: PooledRelativeQKVEpochBoundaryResume) -> None:
        epoch = resume.completed_global_epochs
        if epoch in written_epochs:
            return
        archive.write_bytes(
            f"checkpoints/epoch_{epoch:04d}.ckpt",
            _checkpoint_bytes(
                _checkpoint_payload(
                    archive=archive,
                    config=config,
                    model_construction=model_construction,
                    parameter_count=parameter_count,
                    resume=resume,
                )
            ),
        )
        written_epochs.add(epoch)

    start_epoch = 0 if resume is None else resume.completed_global_epochs
    minimum_epochs = int(_section(config, "trainer")["minimum_global_epochs"])
    continuation_epochs = int(
        _section(config, "trainer")["continuation_block_global_epochs"]
    )
    end_epoch = (
        minimum_epochs
        if start_epoch < minimum_epochs
        else start_epoch + continuation_epochs
    )
    trainer = _section(config, "trainer")
    observed_epoch_durations: list[float] = []

    def emit_epoch_progress(
        epoch: PooledRelativeQKVGlobalEpochRecord,
        core_steps: tuple[PooledRelativeQKVCoreStepRecord, ...],
        epoch_duration_seconds: float,
    ) -> None:
        observed_epoch_durations.append(float(epoch_duration_seconds))
        peak_cuda_gib = (
            torch.cuda.max_memory_allocated() / float(1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        progress = _training_progress_record(
            run_id=archive.run_id,
            model_seed=model_seed,
            epoch=epoch,
            core_steps=core_steps,
            epoch_duration_seconds=epoch_duration_seconds,
            observed_epoch_durations=observed_epoch_durations,
            segment_end_global_epoch=end_epoch,
            learning_rate=float(trainer["learning_rate"]),
            gradient_clip_norm=float(trainer["gradient_clip_norm"]),
            process_peak_cuda_memory_allocated_gib=peak_cuda_gib,
        )
        _emit_training_progress(archive, progress)

    plateau: dict[str, Any] | None = None
    final_result = None
    while True:
        final_result = fit_pooled_relative_qkv_segment(
            model,
            batches,
            _training_config(
                config,
                start_epoch=start_epoch,
                end_epoch=end_epoch,
            ),
            resume=resume,
            checkpoint_callback=save_periodic,
            epoch_callback=emit_epoch_progress,
        )
        losses = [
            record.equal_core_mean_masked_huber
            for record in final_result.global_history
        ]
        plateau = _seed_plateau_decision(losses, model_seed=model_seed)
        archive.write_json(
            f"diagnostics/plateau_audit_epoch_{end_epoch:04d}.json",
            plateau,
        )
        if plateau["should_stop"]:
            break
        resume = final_result.resume
        start_epoch = end_epoch
        end_epoch += int(
            _section(config, "trainer")["continuation_block_global_epochs"]
        )

    assert final_result is not None and plateau is not None
    final_payload = _checkpoint_payload(
        archive=archive,
        config=config,
        model_construction=model_construction,
        parameter_count=parameter_count,
        resume=final_result.resume,
        plateau=plateau,
    )
    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt", _checkpoint_bytes(final_payload)
    )

    evaluation = _section(config, "evaluation")
    held_in, core_diagnostics, prediction_rows = _held_in_diagnostics(
        model,
        batches,
        amp=bool(_section(config, "trainer")["amp"]),
        amp_dtype=str(_section(config, "trainer")["amp_dtype"]),
    )
    salt = os.environ.get("BAGM_SAMPLE_KEY_SALT", "").strip()
    if len(salt.encode("utf-8")) < 16:
        salt = hashlib.sha256(
            f"{archive.run_id}:{ACTIVE_AMENDMENT_SHA256}".encode("utf-8")
        ).hexdigest()
    deidentified = deidentify_prediction_rows(
        prediction_rows,
        identifier_fields=["core_alias"],
        salt=salt,
        namespace="cancer-6core-held-in-alias",
    )
    for row in deidentified:
        row["run_id"] = archive.run_id
    archive.write_predictions("fit", deidentified)

    global_rows = [
        {
            "run_id": archive.run_id,
            "split": "fit",
            **asdict(record),
        }
        for record in final_result.global_history
    ]
    archive.write_table("metrics/history", global_rows)
    archive.write_table(
        "metrics/core_steps",
        [
            {
                "run_id": archive.run_id,
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
            for record in final_result.core_history
        ],
    )
    archive.write_json("diagnostics/held_in_fit_metrics_by_core.json", core_diagnostics)
    final_metrics = {
        **held_in,
        "fit/training/final_equal_core_masked_huber": float(
            final_result.final_train_loss
        ),
        "fit/training/final_global_epoch": float(
            final_result.completed_global_epochs
        ),
    }
    archive.write_json("metrics/final.json", final_metrics)
    for name, value in final_metrics.items():
        archive.append_metric_event(
            {
                "name": name,
                "value": value,
                "step": final_result.completed_global_epochs,
            }
        )
    archive.write_json(
        "provenance/relative_qkv_training.json",
        {
            "active_amendment": str(ACTIVE_AMENDMENT),
            "active_amendment_sha256": ACTIVE_AMENDMENT_SHA256,
            "model_seed": model_seed,
            "active_model_seeds": list(ACTIVE_MODEL_SEEDS),
            "deferred_model_seeds": list(DEFERRED_MODEL_SEEDS),
            "five_seed_campaign_complete": False,
            "mask_base_seed": MASK_BASE_SEED,
            "held_in_mask_base_seed": HELD_IN_MASK_BASE_SEED,
            "parameter_count": parameter_count,
            "model_construction": model_construction,
            "completed_global_epochs": final_result.completed_global_epochs,
            "optimizer_steps_completed": final_result.optimizer_steps_completed,
            "mask_views_per_core_step": final_result.mask_views_per_core_step,
            "state_dict_sha256": final_result.final_state_checksum,
            "history_sha256": final_result.history_checksum,
            "plateau": plateau,
            "cohort_manifest_sha256": sha256_file(cohort_dir / "manifest.json"),
            "graph_manifest_sha256": sha256_file(graph_dir / "manifest.json"),
            "hardware_preflight_receipt_sha256": preflight[
                "receipt_content_sha256"
            ],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "resume_source": resume_receipt,
        },
    )
    peak_vram_gib = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "campaign_id": CAMPAIGN_ID,
        "model_name": "relative-qkv-gat",
        "model_seed": model_seed,
        "final_epoch": final_result.completed_global_epochs,
        "optimizer_steps": final_result.optimizer_steps_completed,
        "parameter_count": parameter_count,
        "primary_metric_name": evaluation["primary_metric"],
        "primary_metric_value": held_in[evaluation["primary_metric"]],
        "peak_vram_gib": peak_vram_gib,
        "duration_seconds": time.monotonic() - started,
        "checkpoint": "checkpoints/last.ckpt",
        "plateau_confirmed": True,
        "five_seed_campaign_complete": False,
        "active_model_seeds": list(ACTIVE_MODEL_SEEDS),
        "deferred_model_seeds": list(DEFERRED_MODEL_SEEDS),
        "resume_source": resume_receipt,
        "generalization_estimate": False,
    }
    archive.write_summary(summary)
    return summary


def run_seed0_to_plateau(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible entry point for the now config-seeded runner."""

    return run_seed_to_plateau(
        config,
        archive,
        resume_checkpoint=resume_checkpoint,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a configured six-core relative-QKV model seed to its "
            "training-loss plateau."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    summary = run_seed_to_plateau(
        config,
        archive,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
