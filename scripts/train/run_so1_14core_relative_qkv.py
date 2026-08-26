#!/usr/bin/env python3
"""Train SO_1 cores 1--14 with Relative-QKV under four-rank DDP.

This entry point is launched only by the repository queue through
``torch.distributed.run``.  Torchrun remains the queue-owned child process and
uses ``--max-restarts=0``.  The run trains for at least 150 global epochs and
then continues in prespecified 25-epoch blocks until the strict held-in
training-loss plateau rule passes at two consecutive audits.
"""

from __future__ import annotations

import argparse
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
from spatial_benchmark.masking import derive_mask_seed  # noqa: E402
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training import (  # noqa: E402
    _tree_sha256,
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
    ReceiverChunkedRelativeGeometryQKVGraphTransformer,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    deidentify_prediction_rows,
)
from spatial_benchmark.so1_pooled_full_core import (  # noqa: E402
    EXPECTED_TOTAL_CELLS,
    SO1_ALIASES,
)
from spatial_benchmark.so1_relative_graphs import (  # noqa: E402
    load_so1_relative_qkv_batches,
)
from spatial_benchmark.so1_training_observability import (  # noqa: E402
    AtomicLatestCheckpointStore,
    CHECKPOINT_SCHEMA,
    DurableEpochMetricsCSV,
    build_epoch_metrics_row,
    require_latest_only_checkpoint_layout,
    strict_plateau_monitor_fields,
)
from spatial_benchmark.training import _autocast_context, set_deterministic_seed  # noqa: E402


CAMPAIGN_ID = "cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150"
PLATEAU_PROTOCOL = "held_in_pooled_so1_14core_relative_qkv_plateau_min150"
WORLD_SIZE = 4
VISIBLE_DEVICES = "0,1,2,3"
MODEL_SEED = 0
HELD_IN_MASK_BASE_SEED = 2026082591
STDOUT_PREFIX = "[bagm-so1-training]"
FINAL_RELOAD_PREDICTION_ATOL = 1e-6
FINAL_RELOAD_PREDICTION_RTOL = 1e-6
FINAL_RELOAD_RECEIVER_COUNT = 8
PLATEAU_FIRST_AUDIT_EPOCH = 150
PLATEAU_AUDIT_INTERVAL = 25
PLATEAU_WINDOW_EPOCHS = 50


class SO114CoreRunnerError(RuntimeError):
    """Raised before a run can violate the four-rank production contract."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise SO114CoreRunnerError(f"Resolved config requires {name!r} mapping.")
    return value


def _require_equal(actual: object, expected: object, *, field: str) -> None:
    if actual != expected:
        raise SO114CoreRunnerError(
            f"{field} must be {expected!r}; received {actual!r}."
        )


def _is_plateau_training(config: Mapping[str, Any]) -> bool:
    evaluation = config.get("evaluation")
    trainer = config.get("trainer")
    return (
        isinstance(evaluation, Mapping)
        and isinstance(trainer, Mapping)
        and evaluation.get("protocol") == PLATEAU_PROTOCOL
        and trainer.get("execution_mode") == "fresh_plateau_min150"
    )


def _runtime_path(raw: object, paths: ProjectPaths) -> Path:
    """Resolve absolute and repository-root-relative config paths with overrides."""

    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if not candidate.parts or ".." in candidate.parts:
        raise SO114CoreRunnerError(f"Unsafe configured runtime path: {raw!r}.")
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
        _section(config, "campaign").get("campaign_id"),
        CAMPAIGN_ID,
        field="campaign.campaign_id",
    )
    protocol = _section(config, "evaluation").get("protocol")
    _require_equal(protocol, PLATEAU_PROTOCOL, field="evaluation.protocol")
    dataset = _section(config, "dataset")
    _require_equal(tuple(dataset.get("core_aliases", ())), SO1_ALIASES, field="dataset.core_aliases")
    _require_equal(
        dataset.get("total_fit_cells"),
        EXPECTED_TOTAL_CELLS,
        field="dataset.total_fit_cells",
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
        "execution_mode": "fresh_plateau_min150",
        "max_epochs": None,
        "minimum_global_epochs": PLATEAU_FIRST_AUDIT_EPOCH,
        "initial_global_epoch_budget": PLATEAU_FIRST_AUDIT_EPOCH,
        "fixed_epoch_budget": False,
        "continuation_policy": "strict_training_loss_plateau_25_epoch_blocks",
        "continuation_block_global_epochs": PLATEAU_AUDIT_INTERVAL,
        "plateau_stopping_enabled": True,
        "plateau_diagnostic_only": False,
        "plateau_rule": (
            "strict_absolute_relative_half_window_change_and_"
            "normalized_absolute_slope"
        ),
        "plateau_audit_interval_global_epochs": PLATEAU_AUDIT_INTERVAL,
        "plateau_first_audit_epoch": PLATEAU_FIRST_AUDIT_EPOCH,
        "plateau_window_global_epochs": PLATEAU_WINDOW_EPOCHS,
        "plateau_consecutive_passing_audits": 2,
        "plateau_absolute_relative_half_window_change_max": 0.0005,
        "plateau_normalized_absolute_slope_per_epoch_max": 0.000025,
        "maximum_scientific_epoch_cap": None,
        "optimizer_zero_grad_per_paired_core_update": 1,
        "optimizer_steps_per_paired_core_update": 1,
        "checkpoint_policy": "atomic_latest_then_final_last_only",
        "checkpoint_every_global_epochs": 1,
        "checkpoint_final_role": "last_confirmed_strict_training_loss_plateau_epoch",
    }
    for field, expected in locked_trainer.items():
        _require_equal(trainer.get(field), expected, field=f"trainer.{field}")
    launcher = _section(config, "launcher")
    locked_launcher = {
        "requested_gpu": VISIBLE_DEVICES,
        "requested_gpu_count": WORLD_SIZE,
        "require_exact_visible_devices": VISIBLE_DEVICES,
        "process_count": WORLD_SIZE,
        "elastic_max_restarts": 0,
        "hardware_preflight_receipt": (
            "state/preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json"
        ),
    }
    for field, expected in locked_launcher.items():
        _require_equal(launcher.get(field), expected, field=f"launcher.{field}")
    if launcher.get("resume_checkpoint") is not None:
        raise SO114CoreRunnerError(
            "The locked fresh SO1 experiment may not configure a source checkpoint."
        )
    if launcher.get("source_artifact_path") is not None:
        raise SO114CoreRunnerError(
            "The locked fresh SO1 experiment may not configure a source bundle."
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
        raise SO114CoreRunnerError(
            f"A passing four-rank hardware preflight is required: {receipt_path}."
        ) from exc
    if not isinstance(value, dict):
        raise SO114CoreRunnerError("Hardware preflight receipt must be a mapping.")
    expected_checksum = value.get("receipt_content_sha256")
    content = dict(value)
    content.pop("receipt_content_sha256", None)
    if expected_checksum != _canonical_sha256(content):
        raise SO114CoreRunnerError("Hardware preflight receipt checksum mismatch.")
    required = {
        "schema": "so1_14core_relative_qkv_ddp4_preflight_v1",
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
        "prior_relative_qkv_equivalence_verified": True,
    }
    for field, expected in required.items():
        _require_equal(value.get(field), expected, field=f"preflight.{field}")
    if value.get("cohort_manifest_sha256") != sha256_file(
        cohort_dir / "manifest.json"
    ):
        raise SO114CoreRunnerError("Preflight cohort manifest has changed.")
    if value.get("graph_manifest_sha256") != sha256_file(
        graph_dir / "manifest.json"
    ):
        raise SO114CoreRunnerError("Preflight graph manifest has changed.")
    if value.get("resolved_config_sha256") != _canonical_sha256(
        _preflight_bound_config(config)
    ):
        raise SO114CoreRunnerError("Resolved production config changed after preflight.")
    peak = value.get("peak_vram_gib_all_ranks")
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(
        float(peak)
    ) or float(peak) <= 0:
        raise SO114CoreRunnerError("Preflight peak VRAM measurement is invalid.")
    return value


def _distributed_identity() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO114CoreRunnerError(
            "This runner must be invoked by torch.distributed.run."
        ) from exc
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE) or local_rank not in range(
        WORLD_SIZE
    ):
        raise SO114CoreRunnerError("Torchrun rank/world-size contract is invalid.")
    if rank != local_rank:
        raise SO114CoreRunnerError(
            "Single-node production requires global rank to equal LOCAL_RANK."
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != VISIBLE_DEVICES:
        raise SO114CoreRunnerError(
            f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE_DEVICES}."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise SO114CoreRunnerError(
            "Four logical CUDA devices are required after visibility filtering."
        )
    return rank, local_rank, world_size


def _worker_inputs(
    args: argparse.Namespace,
) -> tuple[str, Path, dict[str, Any], ProjectPaths]:
    run_id = os.environ.get("BAGM_RUN_ID", "").strip()
    environment_scratch = os.environ.get("BAGM_RUN_SCRATCH", "").strip()
    if not run_id or not environment_scratch:
        raise SO114CoreRunnerError(
            "BAGM_RUN_ID and BAGM_RUN_SCRATCH are required from the queue worker."
        )
    supplied_scratch = args.run_scratch.resolve(strict=False)
    if supplied_scratch != Path(environment_scratch).resolve(strict=False):
        raise SO114CoreRunnerError("--run-scratch does not match BAGM_RUN_SCRATCH.")
    expected_config = supplied_scratch / "config.resolved.yaml"
    if args.config.resolve(strict=False) != expected_config.resolve(strict=False):
        raise SO114CoreRunnerError(
            "--config must be the worker-owned resolved configuration."
        )
    environment_config = os.environ.get("BAGM_CONFIG_PATH", "").strip()
    if environment_config and Path(environment_config).resolve(
        strict=False
    ) != expected_config.resolve(strict=False):
        raise SO114CoreRunnerError("BAGM_CONFIG_PATH does not match --config.")
    paths = current_paths()
    if supplied_scratch != (paths.scratch_root / "active_runs" / run_id).resolve(
        strict=False
    ):
        raise SO114CoreRunnerError("Scratch path is not canonical for this run ID.")
    config = load_yaml_mapping(expected_config)
    _validate_contract(config)
    return run_id, supplied_scratch, config, paths


def _model_from_config(
    config: Mapping[str, Any],
    *,
    num_genes: int,
    node_covariate_dim: int,
) -> ReceiverChunkedRelativeGeometryQKVGraphTransformer:
    model = _section(config, "model")
    trainer = _section(config, "trainer")
    set_deterministic_seed(
        MODEL_SEED,
        deterministic=bool(trainer["deterministic"]),
        warn_only=bool(trainer["deterministic_warn_only"]),
    )
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
        cohort_aliases=SO1_ALIASES,
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
        # The reusable segment trainer requires a positive epoch interval; SO1
        # writes one rolling atomic latest checkpoint at every epoch boundary.
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
    """Bind recovery to the exact locked campaign configuration."""

    return _preflight_bound_config(config)


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
        "campaign_id": CAMPAIGN_ID,
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
) -> tuple[CohortRelativeQKVEpochBoundaryResume, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    file_sha256 = sha256_file(resolved)
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SO114CoreRunnerError(f"Cannot reload checkpoint: {resolved}.") from exc
    if not isinstance(value, Mapping):
        raise SO114CoreRunnerError("Resume checkpoint must contain a mapping.")
    payload = dict(value)
    for field, expected in {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "model_seed": MODEL_SEED,
        "cohort_aliases": list(SO1_ALIASES),
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
        raise SO114CoreRunnerError("Resume distributed execution contract drifted.")
    source_config = payload.get("resolved_config")
    if not isinstance(source_config, Mapping) or _resume_compatible_config(
        source_config
    ) != _resume_compatible_config(config):
        raise SO114CoreRunnerError("Resume resolved configuration is incompatible.")
    resume = cohort_epoch_boundary_resume_from_checkpoint(payload)
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
        "same_campaign_resume_contract_verified": True,
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
        raise SO114CoreRunnerError(
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
        raise SO114CoreRunnerError(
            "Resume checkpoint requires its sibling results/epoch_metrics.csv."
        )
    archive.copy_file(source, "results/epoch_metrics.csv")
    writer = DurableEpochMetricsCSV(archive.scratch_path)
    writer.reconcile(checkpoint_epoch=int(completed_epoch))


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
        raise SO114CoreRunnerError("Plateau audit epochs must be positive.")
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
        raise SO114CoreRunnerError("Plateau audit epochs must be positive.")
    if completed < first:
        return first
    offset = (completed - first) % interval
    return completed + (interval if offset == 0 else interval - offset)


def _strict_plateau_payload(
    losses: Sequence[float],
    *,
    trainer: Mapping[str, Any],
) -> dict[str, Any]:
    fields = strict_plateau_monitor_fields(
        losses,
        model_seed=MODEL_SEED,
        audit_interval_global_epochs=int(
            trainer["plateau_audit_interval_global_epochs"]
        ),
        window_global_epochs=int(trainer["plateau_window_global_epochs"]),
        absolute_relative_half_window_change_max=float(
            trainer["plateau_absolute_relative_half_window_change_max"]
        ),
        normalized_absolute_slope_per_epoch_max=float(
            trainer["plateau_normalized_absolute_slope_per_epoch_max"]
        ),
        consecutive_passing_audits=int(
            trainer["plateau_consecutive_passing_audits"]
        ),
    )
    required = int(trainer["plateau_consecutive_passing_audits"])
    should_stop = bool(fields["plateau_should_stop"])
    return {
        "schema": "so1_14core_strict_plateau_decision_v1",
        "completed_global_epochs": len(losses),
        "final_epoch": len(losses) if should_stop else None,
        "role": "held_in_training_loss_stopping_rule",
        "absolute_relative_half_window_change_max": float(
            trainer["plateau_absolute_relative_half_window_change_max"]
        ),
        "normalized_absolute_slope_per_epoch_max": float(
            trainer["plateau_normalized_absolute_slope_per_epoch_max"]
        ),
        "required_consecutive_passing_audits": required,
        "should_stop": should_stop,
        "training_stop_applied": should_stop,
        "validation_or_test_metric": False,
        "checkpoint_selection_metric": False,
        **fields,
    }


def _write_or_verify_strict_plateau_audit(
    archive: RunArchive,
    payload: Mapping[str, Any],
) -> Path:
    """Write one immutable audit, or verify an identical resume recomputation."""

    try:
        completed = int(payload["completed_global_epochs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SO114CoreRunnerError(
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
        raise SO114CoreRunnerError(
            f"Existing strict plateau audit is unreadable: {relative}."
        ) from exc
    if not isinstance(existing, Mapping) or _canonical_sha256(
        dict(existing)
    ) != _canonical_sha256(dict(payload)):
        raise SO114CoreRunnerError(
            f"Existing strict plateau audit drifted: {relative}."
        )
    return target


def _append_epoch_metric_event_once(
    archive: RunArchive,
    row: Mapping[str, Any],
) -> bool:
    """Append an epoch event idempotently after its checkpoint is durable."""

    event = {
        "name": "fit/training/equal_core_mean_masked_huber",
        "value": float(row["equal_core_mean_masked_huber"]),
        "step": int(row["global_epoch"]),
        "phase": "training_epoch",
        "monitor": dict(row),
    }
    target = archive.scratch_path / "metrics" / "events.jsonl"
    if target.exists():
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
            existing = [json.loads(line) for line in lines if line.strip()]
        except (OSError, ValueError) as exc:
            raise SO114CoreRunnerError(
                "Existing metric event log is unreadable."
            ) from exc
        matches = [
            value
            for value in existing
            if isinstance(value, Mapping)
            and value.get("name") == event["name"]
            and value.get("phase") == event["phase"]
            and int(value.get("step", -1)) == event["step"]
        ]
        if len(matches) > 1:
            raise SO114CoreRunnerError(
                f"Duplicate durable epoch events exist at epoch {event['step']}."
            )
        if matches:
            if not math.isclose(
                float(matches[0].get("value")),
                event["value"],
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise SO114CoreRunnerError(
                    f"Durable epoch event drifted at epoch {event['step']}."
                )
            return False
    archive.append_metric_event(event)
    return True


def _validate_final_checkpoint_payload(
    checkpoint: Path,
    *,
    config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    expected_resume: CohortRelativeQKVEpochBoundaryResume,
    expected_plateau: Mapping[str, Any],
) -> tuple[CohortRelativeQKVEpochBoundaryResume, dict[str, Any]]:
    """Reload and validate the complete strict-plateau checkpoint payload."""

    loaded_resume, load_receipt = _load_resume_checkpoint(
        checkpoint,
        config=config,
        model_construction=model_construction,
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
        raise SO114CoreRunnerError("Final checkpoint cannot be reloaded.") from exc
    if not isinstance(raw_payload, Mapping):
        raise SO114CoreRunnerError("Final checkpoint payload must be a mapping.")
    saved_plateau = raw_payload.get("plateau")
    if not isinstance(saved_plateau, Mapping):
        raise SO114CoreRunnerError("Final checkpoint lacks its plateau decision.")
    if _canonical_sha256(dict(saved_plateau)) != _canonical_sha256(
        dict(expected_plateau)
    ):
        raise SO114CoreRunnerError("Final checkpoint plateau payload drifted.")
    if (
        saved_plateau.get("should_stop") is not True
        or int(saved_plateau.get("final_epoch", -1))
        != expected_resume.completed_global_epochs
    ):
        raise SO114CoreRunnerError(
            "Final checkpoint does not encode a confirmed plateau epoch."
        )
    return loaded_resume, {
        **load_receipt,
        "full_resume_payload_validated": True,
        "plateau_payload_validated": True,
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
        raise SO114CoreRunnerError(
            "Final in-memory/reloaded model state checksum differs from checkpoint."
        )

    seed = derive_mask_seed(
        HELD_IN_MASK_BASE_SEED,
        "so1-relative-qkv-final-checkpoint-replay-v1",
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
            raise SO114CoreRunnerError("Checkpoint replay did not return predictions.")
        return prediction.detach().float().cpu().clone()

    in_memory_prediction = predict(in_memory_model)
    reloaded_prediction = predict(reloaded_model)
    if (
        in_memory_prediction.shape != reloaded_prediction.shape
        or not bool(torch.isfinite(in_memory_prediction).all())
        or not bool(torch.isfinite(reloaded_prediction).all())
    ):
        raise SO114CoreRunnerError(
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
        raise SO114CoreRunnerError(
            "Fresh final checkpoint predictions differ from the in-memory model."
        )
    receipt = {
        "schema": "so1_14core_final_checkpoint_prediction_replay_v1",
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
    expected_plateau: Mapping[str, Any],
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
        "schema": "so1_14core_final_checkpoint_reload_verification_v1",
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
                "so1-relative-qkv-held-in-fit-diagnostic",
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
                    "dataset_id": "cosmx_so1_14core_pooled_fit_v1",
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
    if not _is_plateau_training(config):
        raise SO114CoreRunnerError("Configuration is not the locked SO1 plateau run.")
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
    batches = load_so1_relative_qkv_batches(
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    if tuple(batch.alias for batch in batches) != SO1_ALIASES:
        raise SO114CoreRunnerError("Loaded batches do not match SO1-C01--SO1-C14.")
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
    resume_path = _configured_resume_path(args, config, paths)
    resume: CohortRelativeQKVEpochBoundaryResume | None = None
    resume_receipt: dict[str, Any] | None = None
    if resume_path is not None:
        resume, resume_receipt = _load_resume_checkpoint(
            resume_path,
            config=config,
            model_construction=construction,
        )

    trainer = _section(config, "trainer")
    archive: RunArchive | None = None
    metrics_writer: DurableEpochMetricsCSV | None = None
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
            archive.write_json("provenance/resume_source.json", resume_receipt)
        metrics_writer = DurableEpochMetricsCSV(archive.scratch_path)
        metrics_writer.reconcile(
            checkpoint_epoch=0 if resume is None else resume.completed_global_epochs
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
                "schema": "so1_14core_training_monitor_v1",
                "csv": "results/epoch_metrics.csv",
                "jsonl": "metrics/events.jsonl",
                "frequency": "after_every_completed_global_epoch",
                "fsync": True,
                "core_loss_columns": list(SO1_ALIASES),
                "gradient_norm_unit": "seven_paired_core_optimizer_updates",
                "throughput_scope": "140_complete_graph_mask_views_per_epoch",
                "checkpoint_policy": "atomic_latest_then_final_last_only",
                "plateau_monitoring": "strict_training_loss_stopping_rule",
                "minimum_global_epochs": PLATEAU_FIRST_AUDIT_EPOCH,
                "plateau_absolute_relative_half_window_change_max": trainer[
                    "plateau_absolute_relative_half_window_change_max"
                ],
                "plateau_normalized_absolute_slope_per_epoch_max": trainer[
                    "plateau_normalized_absolute_slope_per_epoch_max"
                ],
                "validation_or_test_metric": False,
            },
        )
        archive.write_json("diagnostics/hardware_preflight.json", preflight)
        durable_rows = metrics_writer.read_rows()
        if durable_rows:
            _append_epoch_metric_event_once(archive, durable_rows[-1])

    observed_losses = (
        []
        if resume is None
        else [
            float(record.equal_core_mean_masked_huber)
            for record in resume.global_history
        ]
    )
    pending_epoch_rows: dict[int, dict[str, Any]] = {}

    def epoch_callback(
        epoch: CohortRelativeQKVGlobalEpochRecord,
        core_records: tuple[CohortRelativeQKVCoreRecord, ...],
        update_records: tuple[CohortRelativeQKVOptimizerUpdateRecord, ...],
        duration_seconds: float,
        peak_vram_gib_all_ranks: float,
    ) -> None:
        assert rank == 0 and archive is not None and metrics_writer is not None
        if epoch.completed_global_epochs != len(observed_losses) + 1:
            raise SO114CoreRunnerError("Epoch callback is not contiguous.")
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
            strict_plateau_diagnostic=True,
            strict_plateau_audit_interval_global_epochs=int(
                trainer["plateau_audit_interval_global_epochs"]
            ),
            strict_plateau_window_global_epochs=int(
                trainer["plateau_window_global_epochs"]
            ),
            strict_plateau_absolute_relative_half_window_change_max=float(
                trainer["plateau_absolute_relative_half_window_change_max"]
            ),
            strict_plateau_normalized_absolute_slope_per_epoch_max=float(
                trainer["plateau_normalized_absolute_slope_per_epoch_max"]
            ),
            strict_plateau_consecutive_passing_audits=int(
                trainer["plateau_consecutive_passing_audits"]
            ),
        )
        metrics_writer.append(row)
        pending_epoch_rows[int(row["global_epoch"])] = row
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

    def checkpoint_callback(state: CohortRelativeQKVEpochBoundaryResume) -> None:
        assert rank == 0 and checkpoint_store is not None and archive is not None
        completed = int(state.completed_global_epochs)
        row = pending_epoch_rows.get(completed)
        if row is None:
            raise SO114CoreRunnerError(
                f"Checkpoint epoch {completed} lacks its pending metric row."
            )
        checkpoint_store.save(
            _checkpoint_payload(
                run_id=run_id,
                config=config,
                model_construction=construction,
                parameter_count=parameter_count,
                resume=state,
            ),
            completed_global_epochs=completed,
        )
        _append_epoch_metric_event_once(archive, row)
        del pending_epoch_rows[completed]

    start_epoch = 0 if resume is None else resume.completed_global_epochs
    final_resume: CohortRelativeQKVEpochBoundaryResume | None = None
    plateau: dict[str, Any] | None = None
    completion = None
    source_plateau = None
    strict_plateau_diagnostic = None
    first_audit_epoch = int(trainer["minimum_global_epochs"])
    continuation = int(trainer["plateau_audit_interval_global_epochs"])

    # A crash may happen after the epoch checkpoint and plateau audit but
    # before finalization. Recompute a boundary decision before another step.
    resume_decision = None
    if resume is not None and _is_fixed_plateau_audit_boundary(
        start_epoch,
        first_audit_epoch=first_audit_epoch,
        audit_interval=continuation,
    ):
        resume_decision = _strict_plateau_payload(observed_losses, trainer=trainer)
    if resume_decision is not None:
        plateau = {
            **resume_decision,
            "audit_source": "deterministic_loss_history_recomputation",
        }
        if rank == 0:
            assert archive is not None
            _write_or_verify_strict_plateau_audit(archive, plateau)
        resume_should_stop = torch.tensor(
            [1 if bool(resume_decision["should_stop"]) else 0],
            device=torch.device(f"cuda:{local_rank}"),
            dtype=torch.int32,
        )
        torch.distributed.broadcast(resume_should_stop, src=0)
        if int(resume_should_stop.item()) == 1:
            assert resume is not None
            model.load_state_dict(resume.model_state_dict, strict=True)
            model.to(torch.device(f"cuda:{local_rank}"))
            final_resume = resume

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
        )
        losses = [
            record.equal_core_mean_masked_huber
            for record in segment_result.global_history
        ]
        decision = _strict_plateau_payload(losses, trainer=trainer)
        plateau = {
            **decision,
            "audit_source": "deterministic_loss_history_recomputation",
        }
        if rank == 0:
            assert archive is not None
            _write_or_verify_strict_plateau_audit(archive, plateau)
        should_stop = torch.tensor(
            [1 if bool(decision["should_stop"]) else 0],
            device=torch.device(f"cuda:{local_rank}"),
            dtype=torch.int32,
        )
        torch.distributed.broadcast(should_stop, src=0)
        if int(should_stop.item()) == 1:
            final_resume = segment_result.resume
            break
        resume = segment_result.resume
        start_epoch = end_epoch
        end_epoch = _next_fixed_plateau_audit_epoch(
            start_epoch,
            first_audit_epoch=first_audit_epoch,
            audit_interval=continuation,
        )

    assert final_resume is not None
    if plateau is None or plateau.get("should_stop") is not True:
        raise SO114CoreRunnerError("Training ended without a confirmed strict plateau.")
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
    replay_batch = min(batches, key=lambda batch: (batch.n_nodes, batch.alias))
    checkpoint_verification = _verify_final_checkpoint_reload(
        final_receipt.path,
        expected_file_sha256=final_receipt.sha256,
        config=config,
        model_construction=construction,
        parameter_count=parameter_count,
        expected_resume=final_resume,
        expected_plateau=plateau,
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
        salt = hashlib.sha256(f"{run_id}:{CAMPAIGN_ID}".encode("utf-8")).hexdigest()
    deidentified = deidentify_prediction_rows(
        prediction_rows,
        identifier_fields=["core_alias"],
        salt=salt,
        namespace="so1-14core-held-in-alias",
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
        "provenance/so1_relative_qkv_training.json",
        {
            "campaign_id": CAMPAIGN_ID,
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
                "strict_plateau_min150_25_epoch_blocks"
            ),
        },
    )
    summary = {
        "run_id": run_id,
        "status": "success",
        "campaign_id": CAMPAIGN_ID,
        "model_name": "relative-qkv-gat",
        "model_seed": MODEL_SEED,
        "final_epoch": final_resume.completed_global_epochs,
        "optimizer_steps": final_resume.optimizer_updates_completed,
        "parameter_count": parameter_count,
        "primary_metric_name": evaluation["primary_metric"],
        "primary_metric_value": held_in[evaluation["primary_metric"]],
        "peak_vram_gib": max(
            float(row["peak_vram_gib_all_ranks"])
            for row in DurableEpochMetricsCSV(archive.scratch_path).read_rows()
        ),
        "duration_seconds": time.monotonic() - started,
        "checkpoint": "checkpoints/last.ckpt",
        "epoch_metrics_csv": "results/epoch_metrics.csv",
        "plateau_confirmed": bool(plateau and plateau.get("should_stop")),
        "plateau_rule": trainer["plateau_rule"],
        "plateau_stopping_used": True,
        "minimum_global_epochs": PLATEAU_FIRST_AUDIT_EPOCH,
        "fixed_epoch_target_completed": False,
        "checkpoint_reload_verified": bool(checkpoint_verification["verified"]),
        "world_size": WORLD_SIZE,
        "generalization_estimate": False,
    }
    archive.write_summary(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SO_1 cores 1--14 with four-rank Relative-QKV DDP."
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
