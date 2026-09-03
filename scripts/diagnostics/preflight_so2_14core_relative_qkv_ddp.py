#!/usr/bin/env python3
"""Bounded four-rank hardware preflight for SO2 Relative-QKV training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import torch


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from scripts.train.run_so2_14core_relative_qkv import (  # noqa: E402
    BASELINE_PREFLIGHT_SCHEMA,
    GEOMETRY_MODULATED_MINIMUM_VRAM_HEADROOM_GIB,
    GEOMETRY_MODULATED_PREFLIGHT_MAX_VRAM_GIB,
    GEOMETRY_MODULATED_PREFLIGHT_SCHEMA,
    MODEL_SEED,
    RECURRENT_PREFLIGHT_SCHEMA,
    UNTIED8_MINIMUM_VRAM_HEADROOM_GIB,
    UNTIED8_PREFLIGHT_MAX_VRAM_GIB,
    UNTIED8_PREFLIGHT_SCHEMA,
    VISIBLE_DEVICES,
    WORLD_SIZE,
    _campaign_id,
    _block_gradient_count,
    _block_trainable_parameter_count,
    _canonical_sha256,
    _checkpoint_payload,
    _distributed_identity,
    _expected_parameter_count,
    _gradient_direction_enabled,
    _geometry_modulated4_block_topology,
    _is_geometry_modulated,
    _is_untied8,
    _model_construction,
    _model_from_config,
    _preflight_bound_config,
    _preflight_schema,
    _runtime_path,
    _section,
    _untied8_block_topology,
    _validate_contract,
)
from spatial_benchmark.configuration import compose_config  # noqa: E402
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.gradient_direction_observability import (  # noqa: E402
    BlockGradientDirectionTracker,
    FullGradientDirectionTracker,
)
from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import (  # noqa: E402
    GeometryModulatedRelativeQKVGraphTransformer,
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training_v2 import (  # noqa: E402
    CohortRelativeQKVTrainingConfig,
    fit_cohort_relative_qkv_segment,
    make_cohort_exact_uniform_training_mask,
)
from spatial_benchmark.so2_relative_graphs import (  # noqa: E402
    load_so2_relative_qkv_batches,
)
from spatial_benchmark.so2_training_observability import (  # noqa: E402
    AtomicLatestCheckpointStore,
)
from spatial_benchmark.training import _autocast_context  # noqa: E402


PREFLIGHT_SCHEMA = BASELINE_PREFLIGHT_SCHEMA
PREFLIGHT_ALIASES = ("SO2-C22", "SO2-C23")
GEOMETRY_COMPARISON_RECEIVERS = 128
FULL_CHUNK_MAX_ABS_TOLERANCE = 2e-5
FULL_CHUNK_MEAN_ABS_TOLERANCE = 2e-6
AMP_FP32_MAX_ABS_TOLERANCE = 7.5e-2
AMP_FP32_MEAN_ABS_TOLERANCE = 7.5e-3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".writing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _prior_c23_equivalence(
    paths: Any,
    graph_dir: Path,
    *,
    prior_receipt: Path,
) -> dict[str, Any]:
    prior_receipt = prior_receipt.expanduser().resolve(strict=True)
    try:
        prior = json.loads(prior_receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "The verified CAN-23 full/chunked and AMP/FP32 preflight is required."
        ) from exc
    gates = prior.get("gates") if isinstance(prior, dict) else None
    if (
        not isinstance(gates, dict)
        or prior.get("status") != "passed"
        or prior.get("all_required_gates_passed") is not True
        or not all(
            isinstance(gates.get(name), dict)
            and gates[name].get("passed") is True
            for name in ("amp_fp32_equivalence", "full_chunk_exactness")
        )
        or not isinstance(prior.get("largest_core"), dict)
        or prior["largest_core"].get("alias") != "CAN-23"
    ):
        raise RuntimeError("Prior CAN-23 preflight does not contain passing gates.")
    old_root = paths.data_root / (
        "processed/cancer_6core_relative_qkv_graphs_v1/cores/CAN-23"
    )
    new_root = graph_dir / "cores/SO2-C23"
    file_checks = {}
    for filename in ("edge_index.npy", "relative_geometry.npy"):
        old_sha = sha256_file(old_root / filename)
        new_sha = sha256_file(new_root / filename)
        file_checks[filename] = {
            "prior_sha256": old_sha,
            "so2_sha256": new_sha,
            "byte_identical": old_sha == new_sha,
        }
    if not all(record["byte_identical"] for record in file_checks.values()):
        raise RuntimeError("SO2-C23 graph/geometry is not byte-identical to CAN-23.")
    return {
        "prior_receipt": str(prior_receipt),
        "prior_receipt_sha256": sha256_file(prior_receipt),
        "prior_full_chunk_exactness": gates["full_chunk_exactness"],
        "prior_amp_fp32_equivalence": gates["amp_fp32_equivalence"],
        "files": file_checks,
        "verified": True,
    }


def _preflight_config(
    config: dict[str, Any],
    *,
    rank: int,
    local_rank: int,
) -> CohortRelativeQKVTrainingConfig:
    trainer = _section(config, "trainer")
    masking = _section(config, "masking")
    return CohortRelativeQKVTrainingConfig(
        model_seed=MODEL_SEED,
        cohort_aliases=PREFLIGHT_ALIASES,
        segment_start_global_epoch=0,
        segment_end_global_epoch=1,
        cores_per_optimizer_update=2,
        mask_views_per_core=10,
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
        stage_complete_core_graph_on_device=True,
        staged_relative_geometry_dtype=str(
            trainer["staged_relative_geometry_dtype"]
        ),
        checkpoint_interval_global_epochs=1,
        distributed_world_size=WORLD_SIZE,
        distributed_rank=rank,
    )


def _independent_block_topology(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if _is_geometry_modulated(config):
        if not isinstance(
            model,
            ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
        ):
            raise RuntimeError(
                "Geometry-modulated preflight requires the receiver-chunked "
                "geometry-modulated model class."
            )
        return _geometry_modulated4_block_topology(model)
    if _is_untied8(config):
        return _untied8_block_topology(model)
    return None


def _block_gradient_snapshot(
    model: torch.nn.Module,
    *,
    expected_blocks: int,
) -> list[dict[str, Any]]:
    """Read scalar-only per-block gradients at the pre-clip observer point."""

    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != expected_blocks:
        raise RuntimeError(
            f"Independent-block model must expose exactly {expected_blocks} blocks."
        )
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, block in enumerate(blocks):
            parameters = tuple(
                parameter for parameter in block.parameters() if parameter.requires_grad
            )
            missing = sum(parameter.grad is None for parameter in parameters)
            if missing:
                raise RuntimeError(
                    f"Independent graph block blocks.{index} has {missing} "
                    "missing gradients."
                )
            squared_norm = torch.zeros(
                (), device=parameters[0].device, dtype=torch.float64
            )
            for parameter in parameters:
                assert parameter.grad is not None
                gradient = parameter.grad.detach()
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError(
                        f"Independent graph block blocks.{index} has "
                        "non-finite gradients."
                    )
                squared_norm.add_(gradient.square().sum(dtype=torch.float64))
            norm = float(torch.sqrt(squared_norm).cpu())
            if not np.isfinite(norm) or norm <= 0.0:
                raise RuntimeError(
                    f"Independent graph block blocks.{index} did not receive a "
                    "finite, non-zero gradient."
                )
            records.append(
                {
                    "block_index": index,
                    "block_name": f"blocks.{index}",
                    "trainable_parameter_count": sum(
                        parameter.numel() for parameter in parameters
                    ),
                    "parameters_with_gradient": len(parameters) - missing,
                    "parameters_missing_gradient": missing,
                    "gradient_norm_before_clip": norm,
                }
            )
    return records


def _difference(
    reference: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, float]:
    if reference.shape != observed.shape:
        raise RuntimeError("Numerical comparison shapes differ.")
    delta = (reference.float() - observed.float()).abs()
    return {
        "maximum_absolute_difference": (
            float(delta.max().cpu()) if delta.numel() else 0.0
        ),
        "mean_absolute_difference": (
            float(delta.mean().cpu()) if delta.numel() else 0.0
        ),
    }


def _geometry_modulated_full_reference(
    config: dict[str, Any],
    *,
    num_genes: int,
    node_covariate_dim: int,
) -> GeometryModulatedRelativeQKVGraphTransformer:
    model = _section(config, "model")
    return GeometryModulatedRelativeQKVGraphTransformer(
        num_genes=num_genes,
        node_covariate_dim=node_covariate_dim,
        hidden_dim=int(model["hidden_dim"]),
        attention_heads=int(model["attention_heads"]),
        attention_head_dim=int(model["attention_head_dim"]),
        graph_layers=int(model["graph_layers"]),
        ffn_dim=int(model["ffn_dim"]),
        decoder_dim=int(model["decoder_dim"]),
        geometry_hidden_dim=int(model["geometry_hidden_dim"]),
        dropout=float(model["dropout"]),
        attention_dropout=float(model["attention_dropout"]),
        relative_geometry_dim=int(model["relative_geometry_dim"]),
        qk_normalization_epsilon=float(model["qk_normalization_epsilon"]),
        logit_scale_initial=float(model["logit_scale_initial"]),
        logit_scale_minimum=float(model["logit_scale_minimum"]),
        logit_scale_maximum=float(model["logit_scale_maximum"]),
        modulation_amplitude=float(model["modulation_amplitude"]),
        geometry_bias_bound=float(model["geometry_bias_bound"]),
    )


def _geometry_modulated_numerical_checks(
    model: torch.nn.Module,
    batch: Any,
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Exercise learned modulation in full/chunked and AMP/FP32 paths."""

    if not isinstance(
        model,
        ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
    ):
        raise RuntimeError("Geometry numerical checks received the wrong model.")
    reference = _geometry_modulated_full_reference(
        config,
        num_genes=batch.n_genes,
        node_covariate_dim=int(batch.node_covariates.shape[1]),
    ).to(device)
    reference.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    reference.eval()

    target = batch.target_expression.to(device=device, dtype=torch.float32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    realization = make_cohort_exact_uniform_training_mask(
        batch,
        0,
        view_index=0,
    )
    gene_mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(
        device=device,
        dtype=torch.bool,
    )
    input_expression = target.masked_fill(gene_mask, 0.0)
    receiver_count = min(GEOMETRY_COMPARISON_RECEIVERS, batch.n_nodes)
    edge_ids = torch.nonzero(
        batch.edge_index[1] < receiver_count,
        as_tuple=False,
    ).flatten()
    if not edge_ids.numel():
        raise RuntimeError("Geometry comparison receiver range has no edges.")
    comparison_edges = batch.edge_index.index_select(1, edge_ids)
    comparison_geometry = batch.relative_geometry.index_select(0, edge_ids)
    target_nodes = torch.arange(receiver_count, device=device, dtype=torch.long)

    with torch.no_grad():
        full_output = reference(
            input_expression,
            gene_mask,
            comparison_edges,
            comparison_geometry,
            covariates,
            target_nodes=target_nodes,
            return_explanations=True,
            attention_receivers=target_nodes,
        )
        chunked_output = model(
            input_expression,
            gene_mask,
            comparison_edges,
            comparison_geometry,
            covariates,
            target_nodes=target_nodes,
            return_explanations=True,
            attention_receivers=target_nodes.cpu(),
        )
    prediction_difference = _difference(
        full_output.prediction,
        chunked_output.prediction,
    )
    attention_difference = _difference(
        full_output.attention_weights,
        chunked_output.attention_weights,
    )
    full_chunk_passed = bool(
        prediction_difference["maximum_absolute_difference"]
        <= FULL_CHUNK_MAX_ABS_TOLERANCE
        and prediction_difference["mean_absolute_difference"]
        <= FULL_CHUNK_MEAN_ABS_TOLERANCE
        and attention_difference["maximum_absolute_difference"]
        <= FULL_CHUNK_MAX_ABS_TOLERANCE
        and attention_difference["mean_absolute_difference"]
        <= FULL_CHUNK_MEAN_ABS_TOLERANCE
        and torch.equal(full_output.edge_index.cpu(), chunked_output.edge_index.cpu())
    )

    trainer = _section(config, "trainer")
    with torch.no_grad(), _autocast_context(
        enabled=True,
        device=device,
        dtype_name=str(trainer["amp_dtype"]),
    ):
        amp_output = model(
            input_expression,
            gene_mask,
            comparison_edges,
            comparison_geometry,
            covariates,
            target_nodes=target_nodes,
        )
    amp_difference = _difference(
        chunked_output.prediction,
        amp_output.prediction,
    )
    amp_passed = bool(
        amp_difference["maximum_absolute_difference"]
        <= AMP_FP32_MAX_ABS_TOLERANCE
        and amp_difference["mean_absolute_difference"]
        <= AMP_FP32_MEAN_ABS_TOLERANCE
        and bool(torch.isfinite(amp_output.prediction).all())
    )

    geometry_sample = comparison_geometry.to(
        device=device,
        dtype=torch.float32,
    )
    head_checks: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, block in enumerate(model.blocks):
            modulation, bias = block.geometry_encoder(geometry_sample)
            scale = block.attention_logit_scale()
            record = {
                "block_index": index,
                "modulation_mean_max_abs_error": float(
                    (modulation.mean(dim=-1) - 1.0).abs().max().cpu()
                ),
                "modulation_minimum": float(modulation.min().cpu()),
                "modulation_maximum": float(modulation.max().cpu()),
                "modulation_projection_weight_norm": float(
                    block.geometry_encoder.modulation_projection.weight.norm().cpu()
                ),
                "bias_max_abs": float(bias.abs().max().cpu()),
                "bias_projection_weight_norm": float(
                    block.geometry_encoder.bias_projection.weight.norm().cpu()
                ),
                "logit_scale_minimum": float(scale.min().cpu()),
                "logit_scale_maximum": float(scale.max().cpu()),
            }
            record["passed"] = bool(
                record["modulation_mean_max_abs_error"] <= 2e-6
                and record["modulation_minimum"] > 0.0
                and record["modulation_projection_weight_norm"] > 0.0
                and record["bias_max_abs"] < 1.0
                and record["bias_projection_weight_norm"] > 0.0
                and block.logit_scale_minimum
                < record["logit_scale_minimum"]
                <= record["logit_scale_maximum"]
                < block.logit_scale_maximum
            )
            head_checks.append(record)
    learned_geometry_passed = all(record["passed"] for record in head_checks)
    passed = full_chunk_passed and amp_passed and learned_geometry_passed
    if not passed:
        raise RuntimeError(
            "Geometry-modulated full/chunk, AMP, or learned-head gate failed."
        )

    del reference, full_output, chunked_output, amp_output
    torch.cuda.empty_cache()
    return {
        "passed": passed,
        "comparison_receivers": receiver_count,
        "comparison_edges": int(edge_ids.numel()),
        "full_chunk_exactness": {
            "passed": full_chunk_passed,
            "prediction": prediction_difference,
            "attention": attention_difference,
            "maximum_absolute_tolerance": FULL_CHUNK_MAX_ABS_TOLERANCE,
            "mean_absolute_tolerance": FULL_CHUNK_MEAN_ABS_TOLERANCE,
        },
        "amp_fp32_equivalence": {
            "passed": amp_passed,
            **amp_difference,
            "maximum_absolute_tolerance": AMP_FP32_MAX_ABS_TOLERANCE,
            "mean_absolute_tolerance": AMP_FP32_MEAN_ABS_TOLERANCE,
        },
        "learned_geometry_heads": {
            "passed": learned_geometry_passed,
            "blocks": head_checks,
        },
    }


def run_preflight(
    *,
    config: dict[str, Any],
    output: Path,
    prior_c23_receipt: Path,
    rank: int,
    local_rank: int,
) -> dict[str, Any] | None:
    paths = current_paths()
    _validate_contract(config)
    dataset = _section(config, "dataset")
    cohort_dir = _runtime_path(dataset["prepared_artifact"], paths)
    graph_dir = _runtime_path(dataset["prepared_graph_artifact"], paths)
    all_batches = load_so2_relative_qkv_batches(
        cohort_dir=cohort_dir, graph_dir=graph_dir
    )
    selected = tuple(batch for batch in all_batches if batch.alias in PREFLIGHT_ALIASES)
    if {batch.alias for batch in selected} != set(PREFLIGHT_ALIASES):
        raise RuntimeError("Preflight could not load its exact paired cores.")
    model = _model_from_config(
        config,
        num_genes=selected[0].n_genes,
        node_covariate_dim=int(selected[0].node_covariates.shape[1]),
    )
    construction = _model_construction(
        model,
        config,
        num_genes=selected[0].n_genes,
        node_covariate_dim=int(selected[0].node_covariates.shape[1]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_parameter_count = _expected_parameter_count(config)
    if (
        expected_parameter_count is not None
        and parameter_count != expected_parameter_count
    ):
        raise RuntimeError(
            "Architecture-specific preflight model must contain exactly "
            f"{expected_parameter_count:,} parameters; observed "
            f"{parameter_count:,}."
    )
    block_count = _block_gradient_count(config)
    expected_block_parameter_count = _block_trainable_parameter_count(config)
    independent_block_topology = _independent_block_topology(model, config)
    observed_peak: list[float] = []
    gradient_tracker = (
        FullGradientDirectionTracker(
            expected_optimizer_updates_per_epoch=1,
            resume_boundary_unavailable=False,
        )
        if rank == 0 and _gradient_direction_enabled(config)
        else None
    )
    gradient_summaries: list[dict[str, Any]] = []
    block_gradient_tracker = (
        BlockGradientDirectionTracker(
            expected_blocks=block_count,
            expected_optimizer_updates_per_epoch=1,
            resume_boundary_unavailable=False,
        )
        if rank == 0 and block_count
        else None
    )
    block_gradient_summaries: list[list[dict[str, Any]]] = []
    block_gradient_snapshots: list[list[dict[str, Any]]] = []

    def observe_epoch(
        epoch: Any,
        cores: Any,
        updates: Any,
        duration: float,
        peak: float,
    ) -> None:
        del cores, updates, duration
        observed_peak.append(float(peak))
        if gradient_tracker is not None:
            gradient_summaries.append(
                asdict(gradient_tracker.complete_epoch(epoch.global_epoch))
            )
        if block_gradient_tracker is not None:
            block_gradient_summaries.append(
                [
                    asdict(summary)
                    for summary in block_gradient_tracker.complete_epoch(
                        epoch.global_epoch
                    )
                ]
            )

    def observe_gradient(model_with_gradients: torch.nn.Module, context: Any) -> None:
        if gradient_tracker is not None:
            gradient_tracker.observe(model_with_gradients, context)
        if block_gradient_tracker is not None:
            block_gradient_tracker.observe(model_with_gradients, context)
        if block_count:
            block_gradient_snapshots.append(
                _block_gradient_snapshot(
                    model_with_gradients,
                    expected_blocks=block_count,
                )
            )

    result = fit_cohort_relative_qkv_segment(
        model,
        selected,
        _preflight_config(config, rank=rank, local_rank=local_rank),
        epoch_callback=observe_epoch if rank == 0 else None,
        gradient_observer=(
            observe_gradient
            if rank == 0 and _gradient_direction_enabled(config)
            else None
        ),
    )
    losses = np.asarray(
        [record.masked_huber_loss for record in result.core_history],
        dtype=np.float64,
    )
    gradients = np.asarray(
        [record.gradient_norm for record in result.optimizer_update_history],
        dtype=np.float64,
    )
    finite = bool(
        losses.size == 2
        and gradients.size == 1
        and np.isfinite(losses).all()
        and np.isfinite(gradients).all()
    )
    if not finite:
        raise RuntimeError("Four-rank preflight produced non-finite loss/gradient.")

    geometry_modulated_numerical_gates = None
    if _is_geometry_modulated(config):
        if rank == 0:
            largest_batch = next(
                batch for batch in selected if batch.alias == "SO2-C23"
            )
            geometry_modulated_numerical_gates = (
                _geometry_modulated_numerical_checks(
                    model,
                    largest_batch,
                    config,
                    device=torch.device(f"cuda:{local_rank}"),
                )
            )
        torch.distributed.barrier()

    if rank == 0:
        if len(observed_peak) != 1 or observed_peak[0] <= 0:
            raise RuntimeError("Four-rank preflight did not record peak VRAM.")
        vram_limits = (
            (
                GEOMETRY_MODULATED_PREFLIGHT_MAX_VRAM_GIB,
                GEOMETRY_MODULATED_MINIMUM_VRAM_HEADROOM_GIB,
            )
            if _is_geometry_modulated(config)
            else (
                UNTIED8_PREFLIGHT_MAX_VRAM_GIB,
                UNTIED8_MINIMUM_VRAM_HEADROOM_GIB,
            )
            if _is_untied8(config)
            else None
        )
        maximum_vram_gib, minimum_vram_headroom_gib = (
            vram_limits if vram_limits is not None else (None, None)
        )
        vram_acceptance_passed = bool(
            maximum_vram_gib is None or observed_peak[0] <= maximum_vram_gib
        )
        if not vram_acceptance_passed:
            raise RuntimeError(
                "Independent-block preflight exceeded the "
                f"{maximum_vram_gib:.1f} GiB peak VRAM gate."
            )
        gradient_direction_verified = bool(
            not _gradient_direction_enabled(config)
            or (
                len(gradient_summaries) == 1
                and gradient_summaries[0]["global_epoch"] == 1
                and gradient_summaries[0]["trainable_parameter_count"]
                == expected_parameter_count
                and gradient_summaries[0]["optimizer_updates_observed"] == 1
                and np.isclose(
                    float(
                        gradient_summaries[0][
                            "gradient_norm_mean_before_clip"
                        ]
                    ),
                    float(gradients[0]),
                    rtol=1e-6,
                    atol=1e-6,
                )
                and gradient_summaries[0][
                    "consecutive_optimizer_step_cosine_valid_pairs"
                ]
                == 0
                and gradient_summaries[0][
                    "epoch_aggregate_gradient_cosine_to_previous_epoch"
                ]
                is None
            )
        )
        if not gradient_direction_verified:
            raise RuntimeError(
                "Architecture preflight did not verify scalar gradient observation."
            )
        block_gradient_diagnostics = (
            block_gradient_snapshots[0]
            if block_count and len(block_gradient_snapshots) == 1
            else None
        )
        block_direction_observer_verified = bool(
            not block_count
            or (
                len(block_gradient_summaries) == 1
                and len(block_gradient_summaries[0]) == block_count
                and all(
                    summary["global_epoch"] == 1
                    and summary["block_index"] == index
                    and summary["block_name"] == f"blocks.{index}"
                    and summary["trainable_parameter_count"]
                    == expected_block_parameter_count
                    and summary["optimizer_updates_observed"] == 1
                    and summary[
                        "consecutive_optimizer_step_cosine_valid_pairs"
                    ]
                    == 0
                    and summary[
                        "epoch_aggregate_gradient_cosine_to_previous_epoch"
                    ]
                    is None
                    and np.isfinite(summary["gradient_norm_mean_before_clip"])
                    and summary["gradient_norm_mean_before_clip"] > 0.0
                    for index, summary in enumerate(block_gradient_summaries[0])
                )
            )
        )
        if not block_direction_observer_verified:
            raise RuntimeError(
                "Independent-block preflight did not verify block-direction "
                "observation."
            )
        all_unique_graph_blocks_receive_gradients = bool(
            not block_count
            or (
                block_gradient_diagnostics is not None
                and len(block_gradient_diagnostics) == block_count
                and all(
                    record["block_index"] == index
                    and record["block_name"] == f"blocks.{index}"
                    and record["trainable_parameter_count"]
                    == expected_block_parameter_count
                    and record["parameters_missing_gradient"] == 0
                    and np.isfinite(record["gradient_norm_before_clip"])
                    and record["gradient_norm_before_clip"] > 0.0
                    for index, record in enumerate(block_gradient_diagnostics)
                )
            )
        )
        if not all_unique_graph_blocks_receive_gradients:
            raise RuntimeError(
                "Independent-block preflight did not verify gradient flow "
                "through all blocks."
            )
        paths.state_root.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".so2-ddp4-checkpoint-", dir=paths.state_root)
        )
        try:
            store = AtomicLatestCheckpointStore(temporary_root)
            receipt = store.save(
                _checkpoint_payload(
                    run_id="r_20260825T000000Z_00000000_s000_f00_a00_00000000",
                    config=config,
                    model_construction=construction,
                    parameter_count=parameter_count,
                    resume=result.resume,
                ),
                completed_global_epochs=1,
            )
            loaded = store.load_latest()
            clone = _model_from_config(
                config,
                num_genes=selected[0].n_genes,
                node_covariate_dim=int(selected[0].node_covariates.shape[1]),
            )
            clone.load_state_dict(loaded["model_state_dict"], strict=True)
            reload_verified = all(
                torch.equal(clone.state_dict()[name], tensor.detach().cpu())
                for name, tensor in model.state_dict().items()
            )
            if not reload_verified:
                raise RuntimeError("Four-rank preflight checkpoint reload drifted.")
            checkpoint_sha = receipt.sha256
        finally:
            shutil.rmtree(temporary_root)
        c23_equivalence = _prior_c23_equivalence(
            paths,
            graph_dir,
            prior_receipt=prior_c23_receipt,
        )
        content: dict[str, Any] = {
            "schema": _preflight_schema(config),
            "status": "passed",
            "all_required_gates_passed": True,
            "completed_experiment": False,
            "diagnostic_only": True,
            "campaign_id": _campaign_id(config),
            "created_at": _utc_now(),
            "resolved_config_sha256": _canonical_sha256(
                _preflight_bound_config(config)
            ),
            "cohort_manifest_sha256": sha256_file(cohort_dir / "manifest.json"),
            "graph_manifest_sha256": sha256_file(graph_dir / "manifest.json"),
            "selected_core_aliases": list(PREFLIGHT_ALIASES),
            "distributed_world_size": WORLD_SIZE,
            "distributed_backend": "nccl",
            "visible_devices": VISIBLE_DEVICES,
            "elastic_max_restarts": 0,
            "rank_assignments": [
                "core_a_views_0_4",
                "core_a_views_5_9",
                "core_b_views_0_4",
                "core_b_views_5_9",
            ],
            "optimizer_updates": result.optimizer_updates_completed,
            "complete_graph_mask_views": sum(
                record.n_mask_views for record in result.core_history
            ),
            "finite_loss_and_gradients": finite,
            "core_masked_huber": {
                record.alias: record.masked_huber_loss
                for record in result.core_history
            },
            "gradient_norm_before_clip": float(gradients[0]),
            "peak_vram_gib_all_ranks": observed_peak[0],
            "checkpoint_reload_verified": reload_verified,
            "temporary_checkpoint_sha256": checkpoint_sha,
            "parameter_count": parameter_count,
            "gradient_direction_observer_verified": (
                gradient_direction_verified
            ),
            "gradient_direction_preflight_summary": (
                gradient_summaries[0] if gradient_summaries else None
            ),
            "so2_c23_prior_equivalence_verified": c23_equivalence["verified"],
            "so2_c23_prior_equivalence": c23_equivalence,
            "model": construction,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        if _is_geometry_modulated(config):
            content["geometry_modulated_numerical_gates"] = (
                geometry_modulated_numerical_gates
            )
        if block_count:
            block_content: dict[str, Any] = {
                "all_unique_graph_blocks_receive_gradients": (
                    all_unique_graph_blocks_receive_gradients
                ),
                "graph_block_gradient_diagnostics": block_gradient_diagnostics,
                "block_gradient_direction_observer_verified": (
                    block_direction_observer_verified
                ),
                "block_gradient_direction_preflight_summary": (
                    block_gradient_summaries[0]
                ),
                "peak_vram_gib_all_ranks_max": maximum_vram_gib,
                "minimum_vram_headroom_gib_each_rank": (
                    minimum_vram_headroom_gib
                ),
                "measured_vram_headroom_gib_each_rank": (
                    24.0 - observed_peak[0]
                ),
                "vram_acceptance_passed": vram_acceptance_passed,
            }
            if _is_geometry_modulated(config):
                block_content["geometry_modulated_graph_block_topology"] = (
                    independent_block_topology
                )
            else:
                block_content["untied_graph_block_topology"] = (
                    independent_block_topology
                )
            content.update(block_content)
        content["receipt_content_sha256"] = _canonical_sha256(content)
        _atomic_json(output, content)
    else:
        content = None
    torch.distributed.barrier()
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prior-c23-receipt",
        type=Path,
        help=(
            "Verified prior CAN-23 full/chunked and AMP/FP32 preflight. "
            "Defaults to state/preflight/cancer_6core_relative_qkv_seed0.json."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, local_rank, _ = _distributed_identity()
    paths = current_paths()
    config = compose_config(args.config, config_root=paths.config_root)
    output = (
        _runtime_path(
            _section(config, "launcher")["hardware_preflight_receipt"], paths
        )
        if args.output is None
        else args.output.resolve()
    )
    prior_c23_receipt = (
        paths.state_root / "preflight/cancer_6core_relative_qkv_seed0.json"
        if args.prior_c23_receipt is None
        else args.prior_c23_receipt.resolve()
    )
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=30),
    )
    try:
        receipt = run_preflight(
            config=config,
            output=output,
            prior_c23_receipt=prior_c23_receipt,
            rank=rank,
            local_rank=local_rank,
        )
        if rank == 0:
            print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        return 0
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
