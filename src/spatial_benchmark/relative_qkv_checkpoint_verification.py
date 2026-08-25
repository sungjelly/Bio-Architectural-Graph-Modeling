"""Independent final-checkpoint verification for six-core relative-QKV runs.

The verifier consumes the standalone ``last.ckpt`` payload through the public
post-training loader.  It validates the complete epoch-boundary resume state,
recomputes the locked per-seed plateau decision, and replays the fixed held-in
fit mask twice from two independently reconstructed models.  Attention replay
is deliberately bounded to a small deterministic receiver set.

This is a technical integrity and held-in reconstruction check.  Attention is
computational routing, and none of the receipt fields support a causal claim.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .cancer_pooled_full_core import CANCER_ALIASES, EXPECTED_N_GENES
from .data import ALLOWED_METADATA_COLUMNS
from .fingerprints import sha256_file
from .pooled_relative_qkv_training import (
    CHECKPOINT_INTERVAL_GLOBAL_EPOCHS,
    CORE_ORDER_SEED,
    MASK_BASE_SEED,
    MASK_VIEWS_PER_CORE_STEP,
    PLATEAU_FIRST_ALLOWED_STOP_EPOCH,
    STEPS_PER_GLOBAL_EPOCH,
    PooledRelativeQKVCoreBatch,
    PooledRelativeQKVTrainingError,
    epoch_boundary_resume_from_checkpoint,
    relative_qkv_core_order,
    relative_qkv_mask_seed,
    relative_qkv_model_step_seed,
    single_seed_plateau_decision,
)
from .relative_qkv_post_training import (
    CAMPAIGN_ID,
    CHECKPOINT_SCHEMA,
    RelativeQKVPostTrainingError,
    fixed_inference_mask,
    load_relative_qkv_checkpoint,
)
from .training import _autocast_context, set_deterministic_seed


VERIFICATION_SCHEMA = "cancer_6core_relative_qkv_checkpoint_verification_v1"
ALLOWED_MODEL_SEEDS = (0, 1, 2, 3)
EARLIEST_CONFIRMED_PLATEAU_EPOCH = (
    PLATEAU_FIRST_ALLOWED_STOP_EPOCH + CHECKPOINT_INTERVAL_GLOBAL_EPOCHS
)
MAX_ATTENTION_RECEIVERS_PER_CORE = 64
DEFAULT_PREDICTION_ATOL = 1e-7
DEFAULT_PREDICTION_RTOL = 1e-6
DEFAULT_ATTENTION_ATOL = 1e-7
DEFAULT_ATTENTION_RTOL = 1e-6
DEFAULT_METRIC_ATOL = 1e-6
DEFAULT_METRIC_RTOL = 1e-6
EXPECTED_PARAMETER_COUNT = 5_003_016
EXPECTED_DATASET_FINGERPRINT = (
    "45fe649d1de0f3df3af0f4a1ae69d17249711359d55dde7d1acf45ed1881e77d"
)
EXPECTED_SPLIT_FINGERPRINT = (
    "956931f2cee4d48d33768fa7c6d034e43e4aaf2b40247b527f0755d860805ead"
)


class RelativeQKVCheckpointVerificationError(RelativeQKVPostTrainingError):
    """Raised when a final seed checkpoint fails an integrity gate."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray | Tensor) -> str:
    if isinstance(value, Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(b"relative-qkv-verification-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelativeQKVCheckpointVerificationError(
            f"{location} must be a mapping."
        )
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise RelativeQKVCheckpointVerificationError(
            f"{location} must be a sequence."
        )
    return value


def _load_json_mapping(path: str | Path, location: str) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RelativeQKVCheckpointVerificationError(
            f"Cannot read {location}: {source}."
        ) from exc
    return _mapping(value, location)


def _require_equal(observed: Any, expected: Any, *, location: str) -> None:
    if observed != expected:
        raise RelativeQKVCheckpointVerificationError(
            f"{location}={observed!r}, expected {expected!r}."
        )


def _require_sha256(value: Any, *, location: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RelativeQKVCheckpointVerificationError(
            f"{location} is not a lowercase SHA-256 digest."
        )
    return text


def _float_close(
    observed: float,
    expected: float,
    *,
    location: str,
    atol: float,
    rtol: float,
) -> None:
    if not math.isfinite(float(observed)) or not math.isfinite(float(expected)):
        raise RelativeQKVCheckpointVerificationError(
            f"{location} contains a non-finite value."
        )
    if not math.isclose(float(observed), float(expected), abs_tol=atol, rel_tol=rtol):
        raise RelativeQKVCheckpointVerificationError(
            f"{location} differs: observed={observed}, expected={expected}."
        )


def _production_model_contract(construction: Mapping[str, Any]) -> None:
    locked: Mapping[str, Any] = {
        "num_genes": EXPECTED_N_GENES,
        "node_covariate_dim": len(ALLOWED_METADATA_COLUMNS),
        "hidden_dim": 256,
        "attention_heads": 8,
        "attention_head_dim": 32,
        "graph_layers": 4,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "positional_bias_hidden_dim": 128,
        "relative_geometry_dim": 70,
        "receiver_chunk_size": 512,
        "max_edges_per_chunk": 200_000,
        "activation_checkpointing": True,
    }
    for name, expected in locked.items():
        _require_equal(
            construction.get(name), expected, location=f"model_construction.{name}"
        )
    _float_close(
        float(construction.get("dropout")),
        0.1,
        location="model_construction.dropout",
        atol=0.0,
        rtol=0.0,
    )
    _float_close(
        float(construction.get("attention_dropout")),
        0.0,
        location="model_construction.attention_dropout",
        atol=0.0,
        rtol=0.0,
    )


def _validate_manifest_contract(
    *,
    cohort_manifest_path: Path,
    graph_manifest_path: Path,
    resolved_config: Mapping[str, Any],
    enforce_production_contract: bool,
) -> dict[str, Any]:
    cohort_manifest = _load_json_mapping(cohort_manifest_path, "cohort manifest")
    graph_manifest = _load_json_mapping(graph_manifest_path, "graph manifest")
    cohort_sha256 = sha256_file(cohort_manifest_path)
    graph_sha256 = sha256_file(graph_manifest_path)
    dataset = _mapping(resolved_config.get("dataset"), "resolved_config.dataset")
    _require_equal(
        dataset.get("cohort_manifest_file_sha256"),
        cohort_sha256,
        location="resolved_config.dataset.cohort_manifest_file_sha256",
    )
    _require_equal(
        dataset.get("graph_manifest_file_sha256"),
        graph_sha256,
        location="resolved_config.dataset.graph_manifest_file_sha256",
    )
    _require_equal(
        tuple(_sequence(dataset.get("core_aliases"), "dataset.core_aliases")),
        CANCER_ALIASES,
        location="resolved_config.dataset.core_aliases",
    )
    _require_equal(
        dataset.get("validation_or_test_partition_present"),
        False,
        location="dataset.validation_or_test_partition_present",
    )
    _require_equal(
        dataset.get("generalization_claim_supported"),
        False,
        location="dataset.generalization_claim_supported",
    )
    if enforce_production_contract:
        _require_equal(
            dataset.get("dataset_fingerprint"),
            EXPECTED_DATASET_FINGERPRINT,
            location="dataset.dataset_fingerprint",
        )
        _require_equal(
            dataset.get("split_fingerprint"),
            EXPECTED_SPLIT_FINGERPRINT,
            location="dataset.split_fingerprint",
        )

    cohort = _mapping(cohort_manifest.get("cohort"), "cohort_manifest.cohort")
    features = _mapping(
        cohort_manifest.get("features"), "cohort_manifest.features"
    )
    _require_equal(
        tuple(_sequence(cohort.get("aliases"), "cohort_manifest.cohort.aliases")),
        CANCER_ALIASES,
        location="cohort_manifest.cohort.aliases",
    )
    _require_equal(
        cohort.get("validation_or_test_partition_present"),
        False,
        location="cohort_manifest.cohort.validation_or_test_partition_present",
    )
    _require_equal(
        tuple(_sequence(graph_manifest.get("aliases"), "graph_manifest.aliases")),
        CANCER_ALIASES,
        location="graph_manifest.aliases",
    )
    _require_equal(
        graph_manifest.get("cohort_manifest_sha256"),
        cohort_sha256,
        location="graph_manifest.cohort_manifest_sha256",
    )
    gene_names = tuple(
        str(value)
        for value in _sequence(features.get("gene_names"), "features.gene_names")
    )
    metadata_names = tuple(
        str(value)
        for value in _sequence(
            features.get("model_covariate_names"), "features.model_covariate_names"
        )
    )
    if len(set(gene_names)) != len(gene_names):
        raise RelativeQKVCheckpointVerificationError(
            "The prepared gene schema contains duplicates."
        )
    _require_equal(
        features.get("coordinates_are_model_covariates"),
        False,
        location="features.coordinates_are_model_covariates",
    )
    if enforce_production_contract:
        _require_equal(
            len(gene_names), EXPECTED_N_GENES, location="features gene count"
        )
        _require_equal(
            metadata_names,
            tuple(ALLOWED_METADATA_COLUMNS),
            location="features.model_covariate_names",
        )
    return {
        "cohort_manifest_path": str(cohort_manifest_path),
        "cohort_manifest_file_sha256": cohort_sha256,
        "cohort_manifest_content_sha256": cohort_manifest.get(
            "manifest_content_sha256"
        ),
        "graph_manifest_path": str(graph_manifest_path),
        "graph_manifest_file_sha256": graph_sha256,
        "graph_manifest_content_sha256": graph_manifest.get(
            "manifest_content_sha256"
        ),
        "core_aliases": list(CANCER_ALIASES),
        "gene_count": len(gene_names),
        "node_covariate_count": len(metadata_names),
        "raw_coordinates_are_model_inputs": False,
    }


def _validate_history_semantics(
    payload: Mapping[str, Any],
    batches: Sequence[PooledRelativeQKVCoreBatch],
    *,
    model_seed: int,
) -> tuple[Any, list[float]]:
    try:
        resume = epoch_boundary_resume_from_checkpoint(payload)
    except (PooledRelativeQKVTrainingError, KeyError, TypeError, ValueError) as exc:
        raise RelativeQKVCheckpointVerificationError(
            "Checkpoint epoch-boundary resume payload failed checksum validation."
        ) from exc
    completed = resume.completed_global_epochs
    if (
        completed < EARLIEST_CONFIRMED_PLATEAU_EPOCH
        or completed % CHECKPOINT_INTERVAL_GLOBAL_EPOCHS != 0
    ):
        raise RelativeQKVCheckpointVerificationError(
            "Final checkpoint must be at a >=175 global-epoch, 25-epoch boundary."
        )
    _require_equal(
        resume.model_seed, model_seed, location="checkpoint.model_seed"
    )
    _require_equal(
        resume.mask_base_seed, MASK_BASE_SEED, location="checkpoint.mask_base_seed"
    )
    _require_equal(
        resume.core_order_seed,
        CORE_ORDER_SEED,
        location="checkpoint.core_order_seed",
    )
    _require_equal(
        resume.mask_views_per_core_step,
        MASK_VIEWS_PER_CORE_STEP,
        location="checkpoint.mask_views_per_core_step",
    )
    _require_equal(
        resume.optimizer_steps_completed,
        completed * STEPS_PER_GLOBAL_EPOCH,
        location="checkpoint.optimizer_steps_completed",
    )
    batch_by_alias = {batch.alias: batch for batch in batches}
    if tuple(batch_by_alias) != CANCER_ALIASES:
        raise RelativeQKVCheckpointVerificationError(
            "Replay requires the exact ordered six Cancer-core batches."
        )

    for epoch, global_record in enumerate(resume.global_history):
        expected_order = relative_qkv_core_order(
            epoch, aliases=CANCER_ALIASES, core_order_seed=CORE_ORDER_SEED
        )
        _require_equal(
            global_record.global_epoch,
            epoch,
            location=f"global_history[{epoch}].global_epoch",
        )
        _require_equal(
            global_record.completed_global_epochs,
            epoch + 1,
            location=f"global_history[{epoch}].completed_global_epochs",
        )
        _require_equal(
            tuple(global_record.ordered_aliases),
            expected_order,
            location=f"global_history[{epoch}].ordered_aliases",
        )
        _require_equal(
            global_record.optimizer_steps_this_epoch,
            STEPS_PER_GLOBAL_EPOCH,
            location=f"global_history[{epoch}].optimizer_steps_this_epoch",
        )
        _require_equal(
            global_record.cumulative_optimizer_steps,
            (epoch + 1) * STEPS_PER_GLOBAL_EPOCH,
            location=f"global_history[{epoch}].cumulative_optimizer_steps",
        )
        _require_equal(
            global_record.aggregation,
            "equal_core_arithmetic_mean",
            location=f"global_history[{epoch}].aggregation",
        )
        offset = epoch * STEPS_PER_GLOBAL_EPOCH
        core_records = resume.core_history[offset : offset + STEPS_PER_GLOBAL_EPOCH]
        observed_losses: list[float] = []
        for step, (alias, record) in enumerate(zip(expected_order, core_records, strict=True)):
            batch = batch_by_alias[alias]
            _require_equal(record.alias, alias, location="core_history.alias")
            _require_equal(record.global_epoch, epoch, location="core_history.epoch")
            _require_equal(
                record.completed_global_epoch,
                epoch + 1,
                location="core_history.completed_global_epoch",
            )
            _require_equal(record.step_in_epoch, step, location="core_history.step")
            _require_equal(
                record.optimizer_step,
                offset + step + 1,
                location="core_history.optimizer_step",
            )
            _require_equal(record.n_nodes, batch.n_nodes, location="core_history.n_nodes")
            _require_equal(record.n_edges, batch.n_edges, location="core_history.n_edges")
            _require_equal(
                record.n_mask_views,
                MASK_VIEWS_PER_CORE_STEP,
                location="core_history.n_mask_views",
            )
            _require_equal(
                record.model_step_seed,
                relative_qkv_model_step_seed(model_seed, epoch, alias),
                location="core_history.model_step_seed",
            )
            if not math.isfinite(float(record.gradient_norm)) or float(
                record.gradient_norm
            ) < 0.0:
                raise RelativeQKVCheckpointVerificationError(
                    "A historical gradient norm is invalid."
                )
            view_losses: list[float] = []
            total_masked = 0
            for view_index, view in enumerate(record.mask_views):
                _require_equal(
                    view.view_index,
                    view_index,
                    location="core_history.mask_view.view_index",
                )
                _require_equal(
                    view.initial_mask_seed,
                    relative_qkv_mask_seed(
                        alias,
                        epoch,
                        view_index=view_index,
                        mask_base_seed=MASK_BASE_SEED,
                    ),
                    location="core_history.mask_view.initial_mask_seed",
                )
                if int(view.zero_total_mask_resamples) < 0:
                    raise RelativeQKVCheckpointVerificationError(
                        "A historical mask view has a negative resample count."
                    )
                _require_equal(
                    view.effective_mask_seed,
                    relative_qkv_mask_seed(
                        alias,
                        epoch,
                        view_index=view_index,
                        mask_base_seed=MASK_BASE_SEED,
                        resample_attempt=int(view.zero_total_mask_resamples),
                    ),
                    location="core_history.mask_view.effective_mask_seed",
                )
                _require_sha256(
                    view.mask_checksum_sha256,
                    location="core_history.mask_view.mask_checksum_sha256",
                )
                if not 0 < int(view.n_masked_entries) <= batch.n_nodes * batch.n_genes:
                    raise RelativeQKVCheckpointVerificationError(
                        "A historical mask view has an invalid masked-entry count."
                    )
                if not (
                    0
                    <= int(view.masked_count_min)
                    <= int(view.masked_count_max)
                    <= batch.n_genes
                ):
                    raise RelativeQKVCheckpointVerificationError(
                        "A historical mask view is outside the full 0..G support."
                    )
                if not (
                    0.0 <= float(view.masked_count_mean) <= batch.n_genes
                    and 0.0 <= float(view.masked_count_median) <= batch.n_genes
                    and 0 <= int(view.zero_mask_cells) <= batch.n_nodes
                    and 0 <= int(view.full_mask_cells) <= batch.n_nodes
                ):
                    raise RelativeQKVCheckpointVerificationError(
                        "A historical mask-view summary is invalid."
                    )
                if not math.isfinite(float(view.masked_huber_loss)):
                    raise RelativeQKVCheckpointVerificationError(
                        "A historical mask-view loss is non-finite."
                    )
                total_masked += int(view.n_masked_entries)
                view_losses.append(float(view.masked_huber_loss))
            _require_equal(
                record.n_masked_entries_across_views,
                total_masked,
                location="core_history.n_masked_entries_across_views",
            )
            _float_close(
                float(record.masked_huber_loss),
                float(np.mean(view_losses)),
                location="core_history.masked_huber_loss",
                atol=1e-12,
                rtol=1e-12,
            )
            observed_losses.append(float(record.masked_huber_loss))
        _float_close(
            float(global_record.equal_core_mean_masked_huber),
            float(np.mean(observed_losses)),
            location=f"global_history[{epoch}].equal_core_mean_masked_huber",
            atol=1e-12,
            rtol=1e-12,
        )
    losses = [
        float(record.equal_core_mean_masked_huber)
        for record in resume.global_history
    ]
    return resume, losses


def _validate_plateau(
    payload: Mapping[str, Any],
    losses: Sequence[float],
    *,
    model_seed: int,
) -> dict[str, Any]:
    completed = len(losses)
    recomputed = single_seed_plateau_decision(
        losses, model_seed=model_seed, completed_global_epochs=completed
    )
    expected = {
        **asdict(recomputed),
        "validation_or_test_metric": False,
        "checkpoint_selection_metric": False,
    }
    observed = dict(_mapping(payload.get("plateau"), "checkpoint.plateau"))
    if observed != expected:
        raise RelativeQKVCheckpointVerificationError(
            "Serialized plateau decision does not equal an independent recomputation."
        )
    if recomputed.should_stop is not True or recomputed.final_epoch != completed:
        raise RelativeQKVCheckpointVerificationError(
            "Final checkpoint does not satisfy two consecutive plateau audits."
        )
    return expected


def _validate_config_contract(
    payload: Mapping[str, Any],
    *,
    amendment_path: Path | None,
    model_seed: int,
    enforce_production_contract: bool,
) -> Mapping[str, Any]:
    config = _mapping(payload.get("resolved_config"), "checkpoint.resolved_config")
    campaign = _mapping(config.get("campaign"), "resolved_config.campaign")
    trainer = _mapping(config.get("trainer"), "resolved_config.trainer")
    masking = _mapping(config.get("masking"), "resolved_config.masking")
    _require_equal(campaign.get("campaign_id"), CAMPAIGN_ID, location="campaign_id")
    _require_equal(
        config.get("seed"), model_seed, location="resolved_config.seed"
    )
    locked_trainer: Mapping[str, Any] = {
        "minimum_global_epochs": 150,
        "continuation_block_global_epochs": CHECKPOINT_INTERVAL_GLOBAL_EPOCHS,
        "mask_views_per_core_step": MASK_VIEWS_PER_CORE_STEP,
        "optimizer_steps_per_core_step": 1,
        "steps_per_global_epoch": STEPS_PER_GLOBAL_EPOCH,
        "early_stopping": False,
        "restore_best": False,
    }
    for name, expected in locked_trainer.items():
        _require_equal(trainer.get(name), expected, location=f"trainer.{name}")
    deterministic = trainer.get("deterministic")
    deterministic_warn_only = trainer.get("deterministic_warn_only")
    if type(deterministic) is not bool:
        raise RelativeQKVCheckpointVerificationError(
            "trainer.deterministic must be boolean."
        )
    if type(deterministic_warn_only) is not bool:
        raise RelativeQKVCheckpointVerificationError(
            "trainer.deterministic_warn_only must be boolean."
        )
    if enforce_production_contract and (
        deterministic is not True or deterministic_warn_only is not False
    ):
        raise RelativeQKVCheckpointVerificationError(
            "Production checkpoint verification requires "
            "trainer.deterministic=true and "
            "trainer.deterministic_warn_only=false."
        )
    _require_equal(
        masking.get("mask_base_seed"), MASK_BASE_SEED, location="masking.mask_base_seed"
    )
    _require_equal(
        masking.get("independent_views_per_core_epoch"),
        MASK_VIEWS_PER_CORE_STEP,
        location="masking.independent_views_per_core_epoch",
    )
    _require_equal(
        masking.get("model_seed_in_mask_derivation"),
        False,
        location="masking.model_seed_in_mask_derivation",
    )
    amendment_sha256 = _require_sha256(
        payload.get("active_amendment_sha256"),
        location="checkpoint.active_amendment_sha256",
    )
    if amendment_path is not None:
        if amendment_path.is_dir():
            matches = [
                candidate
                for candidate in amendment_path.glob("task_contract_amendment_*.yaml")
                if sha256_file(candidate) == amendment_sha256
            ]
            if len(matches) != 1:
                raise RelativeQKVCheckpointVerificationError(
                    "Exactly one campaign amendment must match the checkpoint checksum."
                )
        else:
            observed = sha256_file(amendment_path)
            _require_equal(
                amendment_sha256,
                observed,
                location="active amendment file checksum",
            )
    return config


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _bounded_receivers(n_nodes: int, count: int) -> np.ndarray:
    if count <= 0 or count > MAX_ATTENTION_RECEIVERS_PER_CORE:
        raise RelativeQKVCheckpointVerificationError(
            f"attention receiver count must be 1..{MAX_ATTENTION_RECEIVERS_PER_CORE}."
        )
    return np.unique(
        np.linspace(0, n_nodes - 1, num=min(n_nodes, count), dtype=np.int64)
    )


def _runner_metrics(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> dict[str, float]:
    selected_prediction = prediction.float()[mask]
    selected_truth = target.float()[mask]
    residual = selected_prediction - selected_truth
    centered = selected_truth - selected_truth.mean()
    denominator = centered.square().sum()
    values = {
        "masked_huber": float(
            F.huber_loss(
                selected_prediction, selected_truth, delta=1.0, reduction="mean"
            ).cpu()
        ),
        "masked_mae": float(residual.abs().mean().cpu()),
        "masked_mse": float(residual.square().mean().cpu()),
        "masked_r2": float(
            (
                1.0
                - residual.square().sum()
                / denominator.clamp_min(torch.finfo(torch.float32).tiny)
            ).cpu()
        ),
        "mean_y_true": float(selected_truth.mean().cpu()),
        "mean_y_pred": float(selected_prediction.mean().cpu()),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise RelativeQKVCheckpointVerificationError(
            "Held-in replay produced a non-finite metric."
        )
    return values


def _extract_replay(
    model: torch.nn.Module,
    batch: PooledRelativeQKVCoreBatch,
    *,
    mask: np.ndarray,
    receivers: np.ndarray,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    stage_graph: bool,
    staged_geometry_dtype: str,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    model.eval()
    model.clear_edge_layout_cache()
    target = batch.target_expression.to(device=device, dtype=torch.float32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    device_mask = torch.from_numpy(np.array(mask, copy=True)).to(
        device=device, dtype=torch.bool
    )
    if stage_graph:
        edge_index = batch.edge_index.to(device=device, dtype=torch.long)
        geometry_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(staged_geometry_dtype)
        if geometry_dtype is None:
            raise RelativeQKVCheckpointVerificationError(
                "Unsupported staged relative-geometry dtype."
            )
        relative_geometry = batch.relative_geometry.to(
            device=device, dtype=geometry_dtype
        )
    else:
        edge_index = batch.edge_index
        relative_geometry = batch.relative_geometry
    selected_receivers = torch.as_tensor(
        receivers, dtype=torch.long, device=edge_index.device
    )
    with torch.no_grad(), _autocast_context(
        enabled=amp, device=device, dtype_name=amp_dtype
    ):
        output = model(
            input_expression=target.masked_fill(device_mask, 0.0),
            gene_mask=device_mask,
            edge_index=edge_index,
            relative_geometry=relative_geometry,
            node_covariates=covariates,
            return_explanations=True,
            attention_receivers=selected_receivers,
            explanation_layer=-1,
        )
    if (
        output.edge_index is None
        or output.attention_weights is None
        or output.content_logits is None
        or output.positional_bias is None
        or output.combined_logits is None
    ):
        raise RelativeQKVCheckpointVerificationError(
            "Model did not return the bounded attention replay."
        )
    metrics = _runner_metrics(output.prediction, target, device_mask)
    prediction = np.ascontiguousarray(
        output.prediction.detach().float().cpu().numpy()
    )
    explanations = {
        "edge_index": np.ascontiguousarray(output.edge_index.detach().cpu().numpy()),
        "attention": np.ascontiguousarray(
            output.attention_weights.detach().float().cpu().numpy()
        ),
        "content": np.ascontiguousarray(
            output.content_logits.detach().float().cpu().numpy()
        ),
        "positional_bias": np.ascontiguousarray(
            output.positional_bias.detach().float().cpu().numpy()
        ),
        "combined": np.ascontiguousarray(
            output.combined_logits.detach().float().cpu().numpy()
        ),
    }
    model.clear_edge_layout_cache()
    del output, target, covariates, device_mask, edge_index, relative_geometry
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, metrics, explanations


def _difference_receipt(
    first: np.ndarray,
    second: np.ndarray,
    *,
    atol: float,
    rtol: float,
    location: str,
) -> dict[str, Any]:
    if first.shape != second.shape or first.dtype != second.dtype:
        raise RelativeQKVCheckpointVerificationError(
            f"{location} replay shape or dtype differs."
        )
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise RelativeQKVCheckpointVerificationError(
            f"{location} replay contains non-finite values."
        )
    first64 = first.astype(np.float64, copy=False)
    second64 = second.astype(np.float64, copy=False)
    difference = np.abs(first64 - second64)
    allowed_difference = float(atol) + float(rtol) * np.abs(second64)
    failing = difference > allowed_difference
    failing_count = int(np.count_nonzero(failing))
    allclose = failing_count == 0
    maximum_difference = float(difference.max(initial=0.0))
    mean_difference = float(difference.mean()) if difference.size else 0.0

    worst: dict[str, Any] | None = None
    if difference.size:
        ratio = np.zeros_like(difference, dtype=np.float64)
        np.divide(
            difference,
            allowed_difference,
            out=ratio,
            where=allowed_difference > 0.0,
        )
        ratio[(allowed_difference <= 0.0) & (difference > 0.0)] = np.inf
        worst_flat_index = int(np.argmax(ratio if failing_count else difference))
        worst_index = tuple(
            int(value) for value in np.unravel_index(worst_flat_index, first.shape)
        )
        worst = {
            "index": list(worst_index),
            "first_value": float(first64[worst_index]),
            "second_value": float(second64[worst_index]),
            "absolute_difference": float(difference[worst_index]),
            "allowed_difference": float(allowed_difference[worst_index]),
            "tolerance_ratio": float(ratio[worst_index]),
        }
    if not allclose:
        assert worst is not None
        raise RelativeQKVCheckpointVerificationError(
            f"{location} reload replay exceeds atol={atol}, rtol={rtol}; "
            f"maximum_absolute_difference={maximum_difference:.9g}, "
            f"mean_absolute_difference={mean_difference:.9g}, "
            f"failing_elements={failing_count}/{difference.size}, "
            f"worst_index={worst['index']}, "
            f"first_value={worst['first_value']:.9g}, "
            f"second_value={worst['second_value']:.9g}, "
            f"worst_absolute_difference={worst['absolute_difference']:.9g}, "
            f"worst_allowed_difference={worst['allowed_difference']:.9g}, "
            f"worst_tolerance_ratio={worst['tolerance_ratio']:.9g}."
        )
    first_sha = _array_sha256(first)
    second_sha = _array_sha256(second)
    return {
        "shape": list(first.shape),
        "dtype": str(first.dtype),
        "first_sha256": first_sha,
        "second_sha256": second_sha,
        "byte_identical": first_sha == second_sha,
        "within_tolerance": allclose,
        "element_count": int(difference.size),
        "failing_element_count": failing_count,
        "maximum_absolute_difference": maximum_difference,
        "mean_absolute_difference": mean_difference,
        "worst_comparison": worst,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _validate_attention(
    values: Mapping[str, np.ndarray],
    *,
    selected_receivers: np.ndarray,
    atol: float,
) -> dict[str, Any]:
    edges = values["edge_index"]
    attention = values["attention"]
    content = values["content"]
    bias = values["positional_bias"]
    combined = values["combined"]
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise RelativeQKVCheckpointVerificationError(
            "Bounded explanation edge_index is invalid."
        )
    if any(
        array.ndim != 2 or array.shape[0] != edges.shape[1]
        for array in (attention, content, bias, combined)
    ):
        raise RelativeQKVCheckpointVerificationError(
            "Bounded explanation tensors are not aligned to edge_index."
        )
    observed_receivers = np.unique(edges[1])
    if not np.array_equal(observed_receivers, selected_receivers):
        raise RelativeQKVCheckpointVerificationError(
            "Bounded explanation returned an unexpected receiver set."
        )
    maximum_normalization_error = 0.0
    for receiver in selected_receivers:
        receiver_sum = attention[edges[1] == receiver].sum(axis=0)
        maximum_normalization_error = max(
            maximum_normalization_error,
            float(np.max(np.abs(receiver_sum - 1.0), initial=0.0)),
        )
    if maximum_normalization_error > max(atol * 10.0, 1e-6):
        raise RelativeQKVCheckpointVerificationError(
            "Selected receiver attention does not normalize to one per head."
        )
    composition_error = float(
        np.max(np.abs(combined - (content + bias)), initial=0.0)
    )
    if composition_error > max(atol * 10.0, 1e-5):
        raise RelativeQKVCheckpointVerificationError(
            "Combined logits do not equal content logits plus positional bias."
        )
    return {
        "selected_receivers": selected_receivers.tolist(),
        "selected_directed_edges": int(edges.shape[1]),
        "attention_heads": int(attention.shape[1]),
        "maximum_receiver_head_normalization_error": maximum_normalization_error,
        "maximum_logit_composition_error": composition_error,
    }


def _compare_metrics(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    location: str,
    atol: float,
    rtol: float,
) -> None:
    exact_names = ("alias", "mask_seed", "mask_checksum", "n_masked_entries")
    for name in exact_names:
        if name in expected:
            _require_equal(
                observed.get(name), expected.get(name), location=f"{location}.{name}"
            )
    for name in (
        "masked_huber",
        "masked_mae",
        "masked_mse",
        "masked_r2",
        "mean_y_true",
        "mean_y_pred",
    ):
        if name in expected:
            _float_close(
                float(observed[name]),
                float(expected[name]),
                location=f"{location}.{name}",
                atol=atol,
                rtol=rtol,
            )


def _artifact_cross_checks(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    payload: Mapping[str, Any],
    per_core: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, float],
    expected_core_metrics_path: Path | None,
    expected_final_metrics_path: Path | None,
    training_provenance_path: Path | None,
    metric_atol: float,
    metric_rtol: float,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    if expected_core_metrics_path is None:
        checks["held_in_fit_metrics_by_core"] = "not_supplied"
    else:
        expected_rows = _sequence(
            json.loads(expected_core_metrics_path.read_text(encoding="utf-8")),
            "expected held-in core metrics",
        )
        _require_equal(
            len(expected_rows), len(per_core), location="held-in core metric row count"
        )
        for observed, expected in zip(per_core, expected_rows, strict=True):
            _compare_metrics(
                observed,
                _mapping(expected, "expected core metric row"),
                location=f"expected core metric {observed['alias']}",
                atol=metric_atol,
                rtol=metric_rtol,
            )
        checks["held_in_fit_metrics_by_core"] = {
            "status": "passed",
            "path": str(expected_core_metrics_path),
            "file_sha256": sha256_file(expected_core_metrics_path),
        }
    if expected_final_metrics_path is None:
        checks["final_metrics"] = "not_supplied"
    else:
        expected = _load_json_mapping(expected_final_metrics_path, "final metrics")
        for name, value in aggregate.items():
            _float_close(
                float(value),
                float(expected[name]),
                location=f"expected final metric {name}",
                atol=metric_atol,
                rtol=metric_rtol,
            )
        checks["final_metrics"] = {
            "status": "passed",
            "path": str(expected_final_metrics_path),
            "file_sha256": sha256_file(expected_final_metrics_path),
        }
    if training_provenance_path is None:
        checks["training_provenance"] = "not_supplied"
    else:
        provenance = _load_json_mapping(
            training_provenance_path, "relative-QKV training provenance"
        )
        exact = {
            "checkpoint_sha256": checkpoint_sha256,
            "state_dict_sha256": payload["model_state_checksum"],
            "history_sha256": payload["history_checksum"],
            "completed_global_epochs": payload["completed_global_epochs"],
            "model_seed": payload["model_seed"],
            "mask_views_per_core_step": MASK_VIEWS_PER_CORE_STEP,
        }
        for name, value in exact.items():
            _require_equal(
                provenance.get(name), value, location=f"training provenance {name}"
            )
        _require_equal(
            provenance.get("plateau"),
            payload.get("plateau"),
            location="training provenance plateau",
        )
        checks["training_provenance"] = {
            "status": "passed",
            "path": str(training_provenance_path),
            "file_sha256": sha256_file(training_provenance_path),
        }
    checks["checkpoint_path"] = str(checkpoint_path)
    return checks


def verify_relative_qkv_checkpoint(
    checkpoint_path: str | Path,
    batches: Sequence[PooledRelativeQKVCoreBatch],
    *,
    cohort_manifest_path: str | Path,
    graph_manifest_path: str | Path,
    amendment_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    attention_receivers_per_core: int = 4,
    prediction_atol: float = DEFAULT_PREDICTION_ATOL,
    prediction_rtol: float = DEFAULT_PREDICTION_RTOL,
    attention_atol: float = DEFAULT_ATTENTION_ATOL,
    attention_rtol: float = DEFAULT_ATTENTION_RTOL,
    metric_atol: float = DEFAULT_METRIC_ATOL,
    metric_rtol: float = DEFAULT_METRIC_RTOL,
    expected_core_metrics_path: str | Path | None = None,
    expected_final_metrics_path: str | Path | None = None,
    training_provenance_path: str | Path | None = None,
    enforce_production_contract: bool = True,
) -> dict[str, Any]:
    """Verify one final seed 0/1/2/3 checkpoint and return a bound receipt."""

    materialized = tuple(batches)
    if tuple(batch.alias for batch in materialized) != CANCER_ALIASES:
        raise RelativeQKVCheckpointVerificationError(
            "Verification requires exactly the ordered six Cancer cores."
        )
    first_batch = materialized[0]
    dimensions = {
        (batch.n_genes, int(batch.node_covariates.shape[1]))
        for batch in materialized
    }
    if len(dimensions) != 1:
        raise RelativeQKVCheckpointVerificationError(
            "The six prepared batches do not share input dimensions."
        )
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    resolved_cohort_manifest = Path(cohort_manifest_path).expanduser().resolve(
        strict=True
    )
    resolved_graph_manifest = Path(graph_manifest_path).expanduser().resolve(
        strict=True
    )
    resolved_amendment = (
        None
        if amendment_path is None
        else Path(amendment_path).expanduser().resolve(strict=True)
    )
    resolved_device = torch.device(device)
    cuda_was_initialized = bool(
        resolved_device.type == "cuda" and torch.cuda.is_initialized()
    )
    if resolved_device.type == "cuda":
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config is None:
            if cuda_was_initialized:
                raise RelativeQKVCheckpointVerificationError(
                    "CUDA was initialized before deterministic verification could "
                    "set CUBLAS_WORKSPACE_CONFIG; launch a fresh verifier process "
                    "with CUBLAS_WORKSPACE_CONFIG=:4096:8."
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            workspace_config = ":4096:8"
        if enforce_production_contract and workspace_config not in {
            ":4096:8",
            ":16:8",
        }:
            raise RelativeQKVCheckpointVerificationError(
                "Production CUDA verification requires "
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8."
            )
        if not torch.cuda.is_available():
            raise RelativeQKVCheckpointVerificationError(
                "CUDA checkpoint verification was requested but CUDA is unavailable."
            )

    # The first reconstruction deliberately remains on CPU.  This exposes the
    # serialized trainer contract before any model allocation or replay on CUDA.
    loaded_first = load_relative_qkv_checkpoint(
        resolved_checkpoint,
        num_genes=first_batch.n_genes,
        node_covariate_dim=int(first_batch.node_covariates.shape[1]),
        device="cpu",
    )
    payload = loaded_first.payload
    _require_equal(payload.get("checkpoint_schema"), CHECKPOINT_SCHEMA, location="schema")
    _require_equal(payload.get("campaign_id"), CAMPAIGN_ID, location="campaign_id")
    try:
        model_seed = int(payload.get("model_seed"))
    except (TypeError, ValueError) as exc:
        raise RelativeQKVCheckpointVerificationError(
            "Checkpoint model_seed is invalid."
        ) from exc
    if model_seed not in ALLOWED_MODEL_SEEDS:
        raise RelativeQKVCheckpointVerificationError(
            f"Checkpoint model_seed must be one of {ALLOWED_MODEL_SEEDS}."
        )
    if not isinstance(payload.get("run_id"), str) or not str(payload["run_id"]).strip():
        raise RelativeQKVCheckpointVerificationError(
            "Checkpoint run_id must be a non-empty string."
        )
    config = _validate_config_contract(
        payload,
        amendment_path=resolved_amendment,
        model_seed=model_seed,
        enforce_production_contract=enforce_production_contract,
    )
    resume, losses = _validate_history_semantics(
        payload, materialized, model_seed=model_seed
    )
    plateau = _validate_plateau(payload, losses, model_seed=model_seed)
    construction = _mapping(
        payload.get("model_construction"), "checkpoint.model_construction"
    )
    if enforce_production_contract:
        _production_model_contract(construction)
    observed_parameters = _parameter_count(loaded_first.model)
    _require_equal(
        payload.get("parameter_count"),
        observed_parameters,
        location="checkpoint.parameter_count",
    )
    if enforce_production_contract:
        _require_equal(
            observed_parameters,
            EXPECTED_PARAMETER_COUNT,
            location="production parameter count",
        )
    input_receipt = _validate_manifest_contract(
        cohort_manifest_path=resolved_cohort_manifest,
        graph_manifest_path=resolved_graph_manifest,
        resolved_config=config,
        enforce_production_contract=enforce_production_contract,
    )

    trainer = _mapping(config.get("trainer"), "resolved_config.trainer")
    amp = bool(trainer.get("amp"))
    amp_dtype = str(trainer.get("amp_dtype", "auto"))
    stage_graph = bool(trainer.get("stage_complete_core_graph_on_device", False))
    staged_geometry_dtype = str(
        trainer.get("staged_relative_geometry_dtype", "float32")
    )
    deterministic = bool(trainer["deterministic"])
    deterministic_warn_only = bool(trainer["deterministic_warn_only"])
    if amp and resolved_device.type != "cuda" and enforce_production_contract:
        raise RelativeQKVCheckpointVerificationError(
            "The production AMP held-in replay requires a CUDA device."
        )
    set_deterministic_seed(
        model_seed,
        deterministic=deterministic,
        warn_only=deterministic_warn_only,
    )
    deterministic_algorithms_enabled = bool(
        torch.are_deterministic_algorithms_enabled()
    )
    deterministic_algorithms_warn_only_enabled = bool(
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    if enforce_production_contract and (
        deterministic_algorithms_enabled is not True
        or deterministic_algorithms_warn_only_enabled is not False
    ):
        raise RelativeQKVCheckpointVerificationError(
            "The production deterministic CUDA execution contract was not "
            "installed successfully."
        )
    loaded_first.model.to(resolved_device)
    loaded_first.model.eval()
    loaded_second = load_relative_qkv_checkpoint(
        resolved_checkpoint,
        num_genes=first_batch.n_genes,
        node_covariate_dim=int(first_batch.node_covariates.shape[1]),
        device=resolved_device,
    )
    _require_equal(
        loaded_first.checkpoint_sha256,
        loaded_second.checkpoint_sha256,
        location="independent reload checkpoint checksum",
    )

    per_core: list[dict[str, Any]] = []
    for batch in materialized:
        fixed_mask = fixed_inference_mask(batch)
        repeated_fixed_mask = fixed_inference_mask(batch)
        if (
            fixed_mask.seed != repeated_fixed_mask.seed
            or fixed_mask.checksum_sha256
            != repeated_fixed_mask.checksum_sha256
            or not np.array_equal(fixed_mask.mask, repeated_fixed_mask.mask)
        ):
            raise RelativeQKVCheckpointVerificationError(
                f"{batch.alias} fixed held-in mask did not reproduce."
            )
        receivers = _bounded_receivers(batch.n_nodes, attention_receivers_per_core)
        prediction_first, metrics_first, explanations_first = _extract_replay(
            loaded_first.model,
            batch,
            mask=fixed_mask.mask,
            receivers=receivers,
            device=resolved_device,
            amp=amp,
            amp_dtype=amp_dtype,
            stage_graph=stage_graph,
            staged_geometry_dtype=staged_geometry_dtype,
        )
        prediction_second, metrics_second, explanations_second = _extract_replay(
            loaded_second.model,
            batch,
            mask=repeated_fixed_mask.mask,
            receivers=receivers,
            device=resolved_device,
            amp=amp,
            amp_dtype=amp_dtype,
            stage_graph=stage_graph,
            staged_geometry_dtype=staged_geometry_dtype,
        )
        for name, value in metrics_first.items():
            _float_close(
                value,
                metrics_second[name],
                location=f"{batch.alias} repeated metric {name}",
                atol=metric_atol,
                rtol=metric_rtol,
            )
        prediction_replay = _difference_receipt(
            prediction_first,
            prediction_second,
            atol=prediction_atol,
            rtol=prediction_rtol,
            location=f"{batch.alias} fixed prediction",
        )
        attention_alignment = _validate_attention(
            explanations_first,
            selected_receivers=receivers,
            atol=attention_atol,
        )
        _validate_attention(
            explanations_second,
            selected_receivers=receivers,
            atol=attention_atol,
        )
        edge_first = explanations_first["edge_index"]
        edge_second = explanations_second["edge_index"]
        if not np.array_equal(edge_first, edge_second):
            raise RelativeQKVCheckpointVerificationError(
                f"{batch.alias} bounded explanation edge alignment changed on reload."
            )
        channel_replay = {
            name: _difference_receipt(
                explanations_first[name],
                explanations_second[name],
                atol=attention_atol,
                rtol=attention_rtol,
                location=f"{batch.alias} selected {name}",
            )
            for name in ("attention", "content", "positional_bias", "combined")
        }
        row: dict[str, Any] = {
            "alias": batch.alias,
            "n_nodes": batch.n_nodes,
            "n_edges": batch.n_edges,
            "mask_seed": fixed_mask.seed,
            "mask_checksum": fixed_mask.checksum_sha256,
            "n_masked_entries": fixed_mask.n_masked_entries,
            **metrics_first,
            "prediction_replay": prediction_replay,
            "selected_attention_replay": {
                **attention_alignment,
                "edge_index_sha256": _array_sha256(edge_first),
                "edge_index_reload_sha256": _array_sha256(edge_second),
                "channels": channel_replay,
            },
        }
        per_core.append(row)
        del prediction_first, prediction_second, explanations_first, explanations_second

    aggregate = {
        "fit/uniform_per_cell/masked_huber": float(
            np.mean([row["masked_huber"] for row in per_core])
        ),
        "fit/uniform_per_cell/masked_mae": float(
            np.mean([row["masked_mae"] for row in per_core])
        ),
        "fit/uniform_per_cell/masked_mse": float(
            np.mean([row["masked_mse"] for row in per_core])
        ),
        "fit/uniform_per_cell/masked_r2": float(
            np.mean([row["masked_r2"] for row in per_core])
        ),
    }
    cross_checks = _artifact_cross_checks(
        checkpoint_path=resolved_checkpoint,
        checkpoint_sha256=loaded_first.checkpoint_sha256,
        payload=payload,
        per_core=per_core,
        aggregate=aggregate,
        expected_core_metrics_path=(
            None
            if expected_core_metrics_path is None
            else Path(expected_core_metrics_path).expanduser().resolve(strict=True)
        ),
        expected_final_metrics_path=(
            None
            if expected_final_metrics_path is None
            else Path(expected_final_metrics_path).expanduser().resolve(strict=True)
        ),
        training_provenance_path=(
            None
            if training_provenance_path is None
            else Path(training_provenance_path).expanduser().resolve(strict=True)
        ),
        metric_atol=metric_atol,
        metric_rtol=metric_rtol,
    )
    receipt: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "campaign_id": CAMPAIGN_ID,
        "run_id": payload.get("run_id"),
        "model_seed": model_seed,
        "allowed_model_seeds": list(ALLOWED_MODEL_SEEDS),
        "scope": "one_checkpoint_replay_with_active_seeds_0_1_2_3",
        "checkpoint": {
            "path": str(resolved_checkpoint),
            "file_sha256": loaded_first.checkpoint_sha256,
            "checkpoint_schema": payload.get("checkpoint_schema"),
            "completed_global_epochs": resume.completed_global_epochs,
            "optimizer_steps_completed": resume.optimizer_steps_completed,
            "parameter_count": observed_parameters,
            "model_state_sha256": resume.model_state_checksum,
            "optimizer_state_sha256": resume.optimizer_state_checksum,
            "amp_scaler_state_sha256": resume.scaler_state_checksum,
            "history_sha256": resume.history_checksum,
            "resume_payload_sha256": resume.resume_checksum,
            "independent_reload_count": 2,
            "independently_reloadable": True,
        },
        "plateau_verification": {
            "status": "passed",
            "earliest_permitted_final_epoch": EARLIEST_CONFIRMED_PLATEAU_EPOCH,
            "checkpoint_boundary_global_epochs": CHECKPOINT_INTERVAL_GLOBAL_EPOCHS,
            "recomputed_decision": plateau,
        },
        "prepared_inputs": input_receipt,
        "execution": {
            "device": str(resolved_device),
            "amp": amp,
            "amp_dtype": amp_dtype,
            "deterministic": deterministic,
            "deterministic_warn_only": deterministic_warn_only,
            "deterministic_seed": model_seed,
            "deterministic_algorithms_enabled": (
                deterministic_algorithms_enabled
            ),
            "deterministic_algorithms_warn_only_enabled": (
                deterministic_algorithms_warn_only_enabled
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "cuda_initialized_before_verifier": cuda_was_initialized,
            "stage_complete_core_graph_on_device": stage_graph,
            "staged_relative_geometry_dtype": staged_geometry_dtype,
            "attention_receivers_per_core_maximum": attention_receivers_per_core,
        },
        "held_in_fit_replay": {
            "status": "passed",
            "role": "held_in_fit_diagnostic_not_validation_or_test",
            "fixed_masks_identical_across_reloads": True,
            "per_core": per_core,
            "equal_core_metrics": aggregate,
        },
        "artifact_cross_checks": cross_checks,
        "tolerances": {
            "prediction_atol": prediction_atol,
            "prediction_rtol": prediction_rtol,
            "attention_atol": attention_atol,
            "attention_rtol": attention_rtol,
            "metric_atol": metric_atol,
            "metric_rtol": metric_rtol,
        },
        "generalization_claim_supported": False,
        "causal_claim_supported": False,
        "interpretation_notice": (
            "Attention is model-derived computational routing and held-in replay is "
            "a technical fit diagnostic; neither establishes direct signaling, "
            "biological mechanism, or causality."
        ),
    }
    receipt["receipt_content_sha256"] = _canonical_sha256(receipt)
    return receipt


def write_verification_receipt(receipt: Mapping[str, Any], path: str | Path) -> Path:
    """Write a new immutable JSON receipt after verifying its content checksum."""

    destination = Path(path).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite verification receipt: {destination}")
    content = dict(receipt)
    checksum = content.pop("receipt_content_sha256", None)
    _require_equal(
        checksum,
        _canonical_sha256(content),
        location="verification receipt content checksum",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary receipt: {temporary}")
    temporary.write_text(
        json.dumps(dict(receipt), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


__all__ = [
    "ALLOWED_MODEL_SEEDS",
    "DEFAULT_ATTENTION_ATOL",
    "DEFAULT_ATTENTION_RTOL",
    "DEFAULT_METRIC_ATOL",
    "DEFAULT_METRIC_RTOL",
    "DEFAULT_PREDICTION_ATOL",
    "DEFAULT_PREDICTION_RTOL",
    "EARLIEST_CONFIRMED_PLATEAU_EPOCH",
    "MAX_ATTENTION_RECEIVERS_PER_CORE",
    "RelativeQKVCheckpointVerificationError",
    "VERIFICATION_SCHEMA",
    "verify_relative_qkv_checkpoint",
    "write_verification_receipt",
]
