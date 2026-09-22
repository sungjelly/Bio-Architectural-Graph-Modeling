#!/usr/bin/env python3
"""Bounded four-rank hardware preflight for SO1 Relative-QKV training.

The baseline diagnostic retains its original one-update checks.  The
geometry-modulated variant additionally proves neutral initialization,
post-update full/chunk and AMP/FP32 behavior on both largest SO1 cores,
learned modulation/bias/temperature heads, full and per-block gradient
observation, disjoint four-block topology, and the architecture-specific VRAM
limit.  Prior SO2 evidence is bound only to the shared attention operator and
never treated as SO1 graph identity.
"""

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
from typing import Any, Mapping

import numpy as np
import torch


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from scripts.train.run_so1_14core_relative_qkv import (  # noqa: E402
    BASELINE_PREFLIGHT_SCHEMA,
    CAMPAIGN_ID,
    EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT,
    GEOMETRY_MODULATED_BLOCK_PARAMETER_COUNT,
    GEOMETRY_MODULATED_MINIMUM_VRAM_HEADROOM_GIB,
    GEOMETRY_MODULATED_PREFLIGHT_MAX_VRAM_GIB,
    MODEL_SEED,
    VISIBLE_DEVICES,
    WORLD_SIZE,
    _campaign_id,
    _canonical_sha256,
    _checkpoint_payload,
    _distributed_identity,
    _geometry_modulated4_block_topology,
    _is_geometry_modulated,
    _model_construction,
    _model_from_config,
    _preflight_bound_config,
    _preflight_schema,
    _runtime_path,
    _section,
    _validate_bound_preparation,
    _validate_contract,
)
from spatial_benchmark.configuration import compose_config  # noqa: E402
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.geometry_modulated_relative_qkv_graph_transformer import (  # noqa: E402
    GeometryModulatedRelativeQKVGraphTransformer,
    ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
)
from spatial_benchmark.gradient_direction_observability import (  # noqa: E402
    BlockGradientDirectionTracker,
    FullGradientDirectionTracker,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.pooled_relative_qkv_training_v2 import (  # noqa: E402
    CohortRelativeQKVTrainingConfig,
    fit_cohort_relative_qkv_segment,
    make_cohort_exact_uniform_training_mask,
)
from spatial_benchmark.so1_relative_graphs import (  # noqa: E402
    load_so1_relative_qkv_batches,
)
from spatial_benchmark.so1_training_observability import (  # noqa: E402
    AtomicLatestCheckpointStore,
)
from spatial_benchmark.training import _autocast_context  # noqa: E402


PREFLIGHT_SCHEMA = BASELINE_PREFLIGHT_SCHEMA
PREFLIGHT_ALIASES = ("SO1-C06", "SO1-C11")
DEFAULT_OUTPUT = Path(
    "state/preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json"
)
DEFAULT_PRIOR_EQUIVALENCE_RECEIPT = Path(
    "state/preflight/cancer_6core_relative_qkv_seed0.json"
)
DEFAULT_PRIOR_GEOMETRY_OPERATOR_RECEIPT = Path(
    "state/preflight/so2_14core_geometry_modulated_relative_qkv_ddp4.json"
)
EXPECTED_PRIOR_GEOMETRY_OPERATOR_RECEIPT_SHA256 = (
    "9790c74d2dfc0e87cd69d7e0bfcccf3bacc4030d045ee98226e77a5e9ef3a5ba"
)
PRIOR_GEOMETRY_OPERATOR_SCHEMA = (
    "so2_14core_geometry_modulated_relative_qkv_ddp4_preflight_v1"
)
PRIOR_GEOMETRY_OPERATOR_CAMPAIGN_ID = (
    "cmp_20260903_so2_14core_geometry_modulated_relative_qkv_seed0_batch2"
)
GEOMETRY_COMPARISON_RECEIVERS = 128
FULL_CHUNK_MAX_ABS_TOLERANCE = 2e-5
FULL_CHUNK_MEAN_ABS_TOLERANCE = 2e-6
AMP_FP32_MAX_ABS_TOLERANCE = 7.5e-2
AMP_FP32_MEAN_ABS_TOLERANCE = 7.5e-3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".writing", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True, allow_nan=False)
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


def _prior_relative_qkv_equivalence(
    *,
    prior_receipt: Path,
) -> dict[str, Any]:
    """Validate the prior implementation-level Relative-QKV equivalence gates.

    The SO_1 graphs are distinct data and therefore are not expected to be
    byte-identical to CAN-23.  The inherited claim is limited to the shared
    Relative-QKV implementation: exact full/receiver-chunked output and the
    established AMP/FP32 tolerance.  SO_1-specific execution is checked by the
    paired-core optimization below.
    """

    resolved = prior_receipt.expanduser().resolve(strict=True)
    try:
        prior = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "A verified prior Relative-QKV equivalence preflight is required."
        ) from exc
    gates = prior.get("gates") if isinstance(prior, dict) else None
    required_gate_names = ("amp_fp32_equivalence", "full_chunk_exactness")
    if (
        not isinstance(gates, dict)
        or prior.get("status") != "passed"
        or prior.get("all_required_gates_passed") is not True
        or not all(
            isinstance(gates.get(name), dict)
            and gates[name].get("passed") is True
            for name in required_gate_names
        )
        or not isinstance(prior.get("largest_core"), dict)
        or prior["largest_core"].get("alias") != "CAN-23"
    ):
        raise RuntimeError(
            "Prior Relative-QKV receipt lacks the passing CAN-23 equivalence gates."
        )
    return {
        "prior_receipt": str(resolved),
        "prior_receipt_sha256": sha256_file(resolved),
        "scope": "shared_relative_qkv_implementation_only",
        "so1_graph_identity_claimed": False,
        "prior_largest_core_alias": "CAN-23",
        "prior_full_chunk_exactness": dict(gates["full_chunk_exactness"]),
        "prior_amp_fp32_equivalence": dict(gates["amp_fp32_equivalence"]),
        "verified": True,
    }


def _prior_geometry_operator_evidence(
    *,
    prior_receipt: Path,
) -> dict[str, Any]:
    """Bind the proven operator implementation without claiming SO1 graph identity."""

    resolved = prior_receipt.expanduser().resolve(strict=True)
    file_sha256 = sha256_file(resolved)
    if file_sha256 != EXPECTED_PRIOR_GEOMETRY_OPERATOR_RECEIPT_SHA256:
        raise RuntimeError(
            "Prior geometry-modulated operator receipt file hash mismatch."
        )
    try:
        prior = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "A verified prior geometry-modulated operator receipt is required."
        ) from exc
    if not isinstance(prior, dict):
        raise RuntimeError("Prior geometry-modulated receipt must be a mapping.")
    expected_content_sha256 = prior.get("receipt_content_sha256")
    content = dict(prior)
    content.pop("receipt_content_sha256", None)
    if expected_content_sha256 != _canonical_sha256(content):
        raise RuntimeError(
            "Prior geometry-modulated operator receipt content checksum mismatch."
        )
    numerical = prior.get("geometry_modulated_numerical_gates")
    model = prior.get("model")
    topology = prior.get("geometry_modulated_graph_block_topology")
    required_numerical = (
        "full_chunk_exactness",
        "amp_fp32_equivalence",
        "learned_geometry_heads",
    )
    valid = bool(
        prior.get("schema") == PRIOR_GEOMETRY_OPERATOR_SCHEMA
        and prior.get("campaign_id") == PRIOR_GEOMETRY_OPERATOR_CAMPAIGN_ID
        and prior.get("status") == "passed"
        and prior.get("all_required_gates_passed") is True
        and prior.get("parameter_count")
        == EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT
        and prior.get("selected_core_aliases") == ["SO2-C22", "SO2-C23"]
        and isinstance(numerical, Mapping)
        and numerical.get("passed") is True
        and all(
            isinstance(numerical.get(name), Mapping)
            and numerical[name].get("passed") is True
            for name in required_numerical
        )
        and isinstance(model, Mapping)
        and model.get("class")
        == "ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer"
        and model.get("attention_score_mechanism")
        == "geometry_modulated_cosine_qkv_v1"
        and model.get("graph_layers") == 4
        and model.get("unique_graph_blocks") == 4
        and model.get("graph_block_weight_tying") == "none"
        and model.get("relative_geometry_value_injection") is False
        and model.get("relative_geometry_role")
        == "attention_logit_modulation_and_bias_only"
        and isinstance(topology, Mapping)
        and topology.get("verified") is True
        and topology.get("graph_block_count") == 4
        and topology.get("unique_graph_block_objects") == 4
        and topology.get("unique_graph_block_parameter_sets") == 4
        and topology.get("graph_block_weight_tying") == "none"
    )
    if not valid:
        raise RuntimeError(
            "Prior receipt lacks the passing geometry-modulated operator evidence."
        )
    assert isinstance(numerical, Mapping)
    return {
        "prior_receipt": str(resolved),
        "prior_receipt_sha256": file_sha256,
        "prior_receipt_content_sha256": expected_content_sha256,
        "scope": "shared_geometry_modulated_attention_implementation_only",
        "so1_graph_identity_claimed": False,
        "prior_core_aliases": ["SO2-C22", "SO2-C23"],
        "prior_cohort_manifest_sha256": prior.get("cohort_manifest_sha256"),
        "prior_graph_manifest_sha256": prior.get("graph_manifest_sha256"),
        "prior_full_chunk_exactness": dict(numerical["full_chunk_exactness"]),
        "prior_amp_fp32_equivalence": dict(numerical["amp_fp32_equivalence"]),
        "prior_learned_geometry_heads": dict(numerical["learned_geometry_heads"]),
        "verified": True,
    }


def _preflight_config(
    config: Mapping[str, Any],
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


def _block_gradient_snapshot(
    model: torch.nn.Module,
    *,
    expected_blocks: int,
) -> list[dict[str, Any]]:
    """Record scalar-only, pre-clip gradient evidence for every unique block."""

    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != expected_blocks:
        raise RuntimeError(
            f"Geometry preflight requires exactly {expected_blocks} blocks."
        )
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, block in enumerate(blocks):
            parameters = tuple(
                parameter for parameter in block.parameters() if parameter.requires_grad
            )
            if not parameters:
                raise RuntimeError(f"Graph block blocks.{index} has no parameters.")
            missing = sum(parameter.grad is None for parameter in parameters)
            if missing:
                raise RuntimeError(
                    f"Graph block blocks.{index} has {missing} missing gradients."
                )
            squared_norm = torch.zeros(
                (), device=parameters[0].device, dtype=torch.float64
            )
            for parameter in parameters:
                assert parameter.grad is not None
                gradient = parameter.grad.detach()
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError(
                        f"Graph block blocks.{index} has non-finite gradients."
                    )
                squared_norm.add_(gradient.square().sum(dtype=torch.float64))
            norm = float(torch.sqrt(squared_norm).cpu())
            if not np.isfinite(norm) or norm <= 0.0:
                raise RuntimeError(
                    f"Graph block blocks.{index} did not receive a finite, "
                    "non-zero gradient."
                )
            records.append(
                {
                    "block_index": index,
                    "block_name": f"blocks.{index}",
                    "trainable_parameter_count": sum(
                        parameter.numel() for parameter in parameters
                    ),
                    "parameters_with_gradient": len(parameters),
                    "parameters_missing_gradient": missing,
                    "gradient_norm_before_clip": norm,
                }
            )
    return records


def _geometry_modulated_initialization_checks(
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Prove the geometry heads start neutral before the diagnostic update."""

    if not isinstance(
        model,
        ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
    ):
        raise RuntimeError("Geometry initialization check received the wrong model.")
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, block in enumerate(model.blocks):
            modulation_weight = block.geometry_encoder.modulation_projection.weight
            bias_weight = block.geometry_encoder.bias_projection.weight
            scale = block.attention_logit_scale()
            record = {
                "block_index": index,
                "modulation_projection_exactly_zero": bool(
                    torch.count_nonzero(modulation_weight) == 0
                ),
                "bias_projection_exactly_zero": bool(
                    torch.count_nonzero(bias_weight) == 0
                ),
                "logit_scale_max_abs_error_from_initial": float(
                    (scale - block.logit_scale_initial).abs().max().cpu()
                ),
            }
            record["passed"] = bool(
                record["modulation_projection_exactly_zero"]
                and record["bias_projection_exactly_zero"]
                and record["logit_scale_max_abs_error_from_initial"] <= 1e-6
            )
            records.append(record)
    passed = len(records) == 4 and all(record["passed"] for record in records)
    if not passed:
        raise RuntimeError("Geometry-modulated model did not initialize neutrally.")
    return {"passed": True, "blocks": records}


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
    config: Mapping[str, Any],
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
    batches: tuple[Any, ...],
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Check both SO1 preflight cores with the actual post-update model state."""

    if not isinstance(
        model,
        ReceiverChunkedGeometryModulatedRelativeQKVGraphTransformer,
    ):
        raise RuntimeError("Geometry numerical checks received the wrong model.")
    if tuple(batch.alias for batch in batches) != PREFLIGHT_ALIASES:
        raise RuntimeError("Geometry numerical checks require SO1-C06 and SO1-C11.")
    reference = _geometry_modulated_full_reference(
        config,
        num_genes=batches[0].n_genes,
        node_covariate_dim=int(batches[0].node_covariates.shape[1]),
    ).to(device)
    reference.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    reference.eval()
    trainer = _section(config, "trainer")
    masking = _section(config, "masking")
    core_checks: list[dict[str, Any]] = []

    for batch in batches:
        target = batch.target_expression.to(device=device, dtype=torch.float32)
        covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
        realization = make_cohort_exact_uniform_training_mask(
            batch,
            0,
            view_index=0,
            mask_base_seed=int(masking["mask_base_seed"]),
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
        if edge_ids.numel() == 0:
            raise RuntimeError(
                f"Geometry comparison range for {batch.alias} has no edges."
            )
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
        edge_order_equal = bool(
            torch.equal(full_output.edge_index.cpu(), chunked_output.edge_index.cpu())
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
            and edge_order_equal
        )

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
                return_explanations=True,
                attention_receivers=target_nodes.cpu(),
            )
        amp_prediction_difference = _difference(
            chunked_output.prediction,
            amp_output.prediction,
        )
        amp_attention_difference = _difference(
            chunked_output.attention_weights,
            amp_output.attention_weights,
        )
        amp_edge_order_equal = bool(
            torch.equal(chunked_output.edge_index.cpu(), amp_output.edge_index.cpu())
        )
        amp_passed = bool(
            amp_prediction_difference["maximum_absolute_difference"]
            <= AMP_FP32_MAX_ABS_TOLERANCE
            and amp_prediction_difference["mean_absolute_difference"]
            <= AMP_FP32_MEAN_ABS_TOLERANCE
            and amp_attention_difference["maximum_absolute_difference"]
            <= AMP_FP32_MAX_ABS_TOLERANCE
            and amp_attention_difference["mean_absolute_difference"]
            <= AMP_FP32_MEAN_ABS_TOLERANCE
            and amp_edge_order_equal
            and bool(torch.isfinite(amp_output.prediction).all())
            and bool(torch.isfinite(amp_output.attention_weights).all())
        )

        geometry_sample = comparison_geometry.to(device=device, dtype=torch.float32)
        learned_blocks: list[dict[str, Any]] = []
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
                    "modulation_max_abs_deviation_from_one": float(
                        (modulation - 1.0).abs().max().cpu()
                    ),
                    "modulation_projection_weight_norm": float(
                        block.geometry_encoder.modulation_projection.weight.norm().cpu()
                    ),
                    "bias_max_abs": float(bias.abs().max().cpu()),
                    "bias_projection_weight_norm": float(
                        block.geometry_encoder.bias_projection.weight.norm().cpu()
                    ),
                    "logit_scale_minimum": float(scale.min().cpu()),
                    "logit_scale_maximum": float(scale.max().cpu()),
                    "logit_scale_max_abs_change_from_initial": float(
                        (scale - block.logit_scale_initial).abs().max().cpu()
                    ),
                }
                record["passed"] = bool(
                    record["modulation_mean_max_abs_error"] <= 2e-6
                    and record["modulation_minimum"] > 0.0
                    and record["modulation_projection_weight_norm"] > 0.0
                    and record["modulation_max_abs_deviation_from_one"] > 0.0
                    and 0.0 < record["bias_max_abs"] < block.geometry_encoder.bias_bound
                    and record["bias_projection_weight_norm"] > 0.0
                    and block.logit_scale_minimum
                    < record["logit_scale_minimum"]
                    <= record["logit_scale_maximum"]
                    < block.logit_scale_maximum
                    and record["logit_scale_max_abs_change_from_initial"] > 0.0
                )
                learned_blocks.append(record)
        learned_passed = len(learned_blocks) == 4 and all(
            record["passed"] for record in learned_blocks
        )
        core_checks.append(
            {
                "alias": batch.alias,
                "comparison_receivers": receiver_count,
                "comparison_edges": int(edge_ids.numel()),
                "full_chunk": {
                    "passed": full_chunk_passed,
                    "prediction": prediction_difference,
                    "attention": attention_difference,
                    "edge_order_equal": edge_order_equal,
                },
                "amp_fp32": {
                    "passed": amp_passed,
                    "prediction": amp_prediction_difference,
                    "attention": amp_attention_difference,
                    "edge_order_equal": amp_edge_order_equal,
                },
                "learned_geometry_heads": {
                    "passed": learned_passed,
                    "blocks": learned_blocks,
                },
            }
        )
        del (
            target,
            covariates,
            gene_mask,
            input_expression,
            target_nodes,
            full_output,
            chunked_output,
            amp_output,
            geometry_sample,
        )

    full_records = [record["full_chunk"] for record in core_checks]
    amp_records = [record["amp_fp32"] for record in core_checks]
    learned_records = [
        {"alias": record["alias"], **record["learned_geometry_heads"]}
        for record in core_checks
    ]
    full_chunk_passed = all(record["passed"] for record in full_records)
    amp_passed = all(record["passed"] for record in amp_records)
    learned_passed = all(record["passed"] for record in learned_records)
    passed = full_chunk_passed and amp_passed and learned_passed
    if not passed:
        raise RuntimeError(
            "SO1 geometry full/chunk, AMP/FP32, or learned-head gate failed."
        )
    result = {
        "passed": True,
        "comparison_core_aliases": list(PREFLIGHT_ALIASES),
        "core_checks": core_checks,
        "full_chunk_exactness": {
            "passed": True,
            "maximum_prediction_absolute_difference": max(
                float(record["prediction"]["maximum_absolute_difference"])
                for record in full_records
            ),
            "maximum_attention_absolute_difference": max(
                float(record["attention"]["maximum_absolute_difference"])
                for record in full_records
            ),
            "maximum_absolute_tolerance": FULL_CHUNK_MAX_ABS_TOLERANCE,
            "mean_absolute_tolerance": FULL_CHUNK_MEAN_ABS_TOLERANCE,
        },
        "amp_fp32_equivalence": {
            "passed": True,
            "maximum_prediction_absolute_difference": max(
                float(record["prediction"]["maximum_absolute_difference"])
                for record in amp_records
            ),
            "maximum_attention_absolute_difference": max(
                float(record["attention"]["maximum_absolute_difference"])
                for record in amp_records
            ),
            "maximum_absolute_tolerance": AMP_FP32_MAX_ABS_TOLERANCE,
            "mean_absolute_tolerance": AMP_FP32_MEAN_ABS_TOLERANCE,
        },
        "learned_geometry_heads": {
            "passed": True,
            "comparison_core_aliases": list(PREFLIGHT_ALIASES),
            "cores": learned_records,
        },
    }
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_preflight(
    *,
    config: dict[str, Any],
    output: Path,
    prior_equivalence_receipt: Path,
    rank: int,
    local_rank: int,
    prior_geometry_operator_receipt: Path = (
        DEFAULT_PRIOR_GEOMETRY_OPERATOR_RECEIPT
    ),
) -> dict[str, Any] | None:
    paths = current_paths()
    _validate_contract(config)
    dataset = _section(config, "dataset")
    cohort_dir = _runtime_path(dataset["prepared_artifact"], paths)
    graph_dir = _runtime_path(dataset["prepared_graph_artifact"], paths)
    preparation_hashes = _validate_bound_preparation(
        dataset,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    all_batches = load_so1_relative_qkv_batches(
        cohort_dir=cohort_dir, graph_dir=graph_dir
    )
    batches_by_alias = {batch.alias: batch for batch in all_batches}
    try:
        selected = tuple(batches_by_alias[alias] for alias in PREFLIGHT_ALIASES)
    except KeyError as exc:
        raise RuntimeError("Preflight could not load its exact paired cores.") from exc
    if tuple(batch.alias for batch in selected) != PREFLIGHT_ALIASES:
        raise RuntimeError("Preflight core ordering drifted.")

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
    geometry_modulated = _is_geometry_modulated(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        geometry_modulated
        and parameter_count != EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Geometry-modulated preflight model must contain exactly "
            f"{EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT:,} parameters; "
            f"observed {parameter_count:,}."
        )
    block_topology = (
        _geometry_modulated4_block_topology(model)
        if geometry_modulated
        else None
    )
    initialization_checks = (
        _geometry_modulated_initialization_checks(model)
        if geometry_modulated
        else None
    )
    observed_peak: list[float] = []
    gradient_tracker = (
        FullGradientDirectionTracker(
            expected_optimizer_updates_per_epoch=1,
            resume_boundary_unavailable=False,
        )
        if rank == 0 and geometry_modulated
        else None
    )
    gradient_summaries: list[dict[str, Any]] = []
    block_gradient_tracker = (
        BlockGradientDirectionTracker(
            expected_blocks=4,
            expected_optimizer_updates_per_epoch=1,
            resume_boundary_unavailable=False,
        )
        if rank == 0 and geometry_modulated
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
        if geometry_modulated:
            block_gradient_snapshots.append(
                _block_gradient_snapshot(model_with_gradients, expected_blocks=4)
            )

    result = fit_cohort_relative_qkv_segment(
        model,
        selected,
        _preflight_config(config, rank=rank, local_rank=local_rank),
        epoch_callback=observe_epoch if rank == 0 else None,
        gradient_observer=(
            observe_gradient if rank == 0 and geometry_modulated else None
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
    if geometry_modulated:
        if rank == 0:
            geometry_modulated_numerical_gates = (
                _geometry_modulated_numerical_checks(
                    model,
                    selected,
                    config,
                    device=torch.device(f"cuda:{local_rank}"),
                )
            )
        torch.distributed.barrier()

    if rank == 0:
        if len(observed_peak) != 1 or observed_peak[0] <= 0:
            raise RuntimeError("Four-rank preflight did not record peak VRAM.")
        vram_acceptance_passed = bool(
            not geometry_modulated
            or observed_peak[0] <= GEOMETRY_MODULATED_PREFLIGHT_MAX_VRAM_GIB
        )
        if not vram_acceptance_passed:
            raise RuntimeError(
                "Geometry-modulated preflight exceeded the 22.0 GiB peak VRAM gate."
            )
        gradient_direction_verified = bool(
            not geometry_modulated
            or (
                len(gradient_summaries) == 1
                and gradient_summaries[0]["global_epoch"] == 1
                and gradient_summaries[0]["trainable_parameter_count"]
                == EXPECTED_GEOMETRY_MODULATED_PARAMETER_COUNT
                and gradient_summaries[0]["optimizer_updates_observed"] == 1
                and gradient_summaries[0][
                    "consecutive_optimizer_step_cosine_valid_pairs"
                ]
                == 0
                and gradient_summaries[0][
                    "epoch_aggregate_gradient_cosine_to_previous_epoch"
                ]
                is None
                and gradient_summaries[0]["resume_boundary_unavailable"] is False
                and np.isclose(
                    float(
                        gradient_summaries[0]["gradient_norm_mean_before_clip"]
                    ),
                    float(gradients[0]),
                    rtol=1e-6,
                    atol=1e-6,
                )
            )
        )
        if not gradient_direction_verified:
            raise RuntimeError(
                "Geometry preflight did not verify full-gradient observation."
            )
        block_gradient_diagnostics = (
            block_gradient_snapshots[0]
            if geometry_modulated and len(block_gradient_snapshots) == 1
            else None
        )
        all_unique_graph_blocks_receive_gradients = bool(
            not geometry_modulated
            or (
                block_gradient_diagnostics is not None
                and len(block_gradient_diagnostics) == 4
                and all(
                    record["block_index"] == index
                    and record["block_name"] == f"blocks.{index}"
                    and record["trainable_parameter_count"]
                    == GEOMETRY_MODULATED_BLOCK_PARAMETER_COUNT
                    and record["parameters_missing_gradient"] == 0
                    and np.isfinite(record["gradient_norm_before_clip"])
                    and record["gradient_norm_before_clip"] > 0.0
                    for index, record in enumerate(block_gradient_diagnostics)
                )
            )
        )
        if not all_unique_graph_blocks_receive_gradients:
            raise RuntimeError(
                "Geometry preflight did not verify gradient flow through all blocks."
            )
        block_direction_observer_verified = bool(
            not geometry_modulated
            or (
                len(block_gradient_summaries) == 1
                and len(block_gradient_summaries[0]) == 4
                and all(
                    summary["global_epoch"] == 1
                    and summary["block_index"] == index
                    and summary["block_name"] == f"blocks.{index}"
                    and summary["trainable_parameter_count"]
                    == GEOMETRY_MODULATED_BLOCK_PARAMETER_COUNT
                    and summary["optimizer_updates_observed"] == 1
                    and summary[
                        "consecutive_optimizer_step_cosine_valid_pairs"
                    ]
                    == 0
                    and summary[
                        "epoch_aggregate_gradient_cosine_to_previous_epoch"
                    ]
                    is None
                    and summary["resume_boundary_unavailable"] is False
                    and np.isfinite(summary["gradient_norm_mean_before_clip"])
                    and summary["gradient_norm_mean_before_clip"] > 0.0
                    for index, summary in enumerate(block_gradient_summaries[0])
                )
            )
        )
        if not block_direction_observer_verified:
            raise RuntimeError(
                "Geometry preflight did not verify per-block gradient direction."
            )
        paths.state_root.mkdir(parents=True, exist_ok=True)
        temporary_root = Path(
            tempfile.mkdtemp(prefix=".so1-ddp4-checkpoint-", dir=paths.state_root)
        )
        try:
            store = AtomicLatestCheckpointStore(temporary_root)
            checkpoint_receipt = store.save(
                _checkpoint_payload(
                    run_id=(
                        "r_20260826T000000Z_00000000_s000_f00_a00_00000000"
                    ),
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
            checkpoint_sha = checkpoint_receipt.sha256
        finally:
            shutil.rmtree(temporary_root)

        prior_equivalence = (
            _prior_geometry_operator_evidence(
                prior_receipt=prior_geometry_operator_receipt,
            )
            if geometry_modulated
            else _prior_relative_qkv_equivalence(
                prior_receipt=prior_equivalence_receipt,
            )
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
            "immutable_preparation_hashes": preparation_hashes,
            "selected_core_aliases": list(PREFLIGHT_ALIASES),
            "selected_core_cell_counts": {
                batch.alias: int(batch.n_nodes) for batch in selected
            },
            "selected_core_edge_counts": {
                batch.alias: int(batch.n_edges) for batch in selected
            },
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
            "prior_relative_qkv_equivalence_verified": prior_equivalence[
                "verified"
            ],
            "prior_relative_qkv_equivalence": prior_equivalence,
            "model": construction,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
        if geometry_modulated:
            content.update(
                {
                    "parameter_count": parameter_count,
                    "geometry_modulated_initialization": initialization_checks,
                    "geometry_modulated_numerical_gates": (
                        geometry_modulated_numerical_gates
                    ),
                    "prior_geometry_operator_evidence_verified": (
                        prior_equivalence["verified"]
                    ),
                    "prior_geometry_operator_evidence": prior_equivalence,
                    "gradient_direction_observer_verified": (
                        gradient_direction_verified
                    ),
                    "gradient_direction_preflight_summary": (
                        gradient_summaries[0]
                    ),
                    "geometry_modulated_graph_block_topology": block_topology,
                    "all_unique_graph_blocks_receive_gradients": (
                        all_unique_graph_blocks_receive_gradients
                    ),
                    "graph_block_gradient_diagnostics": (
                        block_gradient_diagnostics
                    ),
                    "block_gradient_direction_observer_verified": (
                        block_direction_observer_verified
                    ),
                    "block_gradient_direction_preflight_summary": (
                        block_gradient_summaries[0]
                    ),
                    "peak_vram_gib_all_ranks_max": (
                        GEOMETRY_MODULATED_PREFLIGHT_MAX_VRAM_GIB
                    ),
                    "minimum_vram_headroom_gib_each_rank": (
                        GEOMETRY_MODULATED_MINIMUM_VRAM_HEADROOM_GIB
                    ),
                    "measured_vram_headroom_gib_each_rank": (
                        24.0 - observed_peak[0]
                    ),
                    "vram_acceptance_passed": vram_acceptance_passed,
                }
            )
        content["receipt_content_sha256"] = _canonical_sha256(content)
        _atomic_json(output, content)
    else:
        content = None
    torch.distributed.barrier()
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded SO_1 cores 1--14 four-rank preflight."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--prior-equivalence-receipt",
        type=Path,
        help=(
            "Prior passing Relative-QKV full/chunked and AMP/FP32 receipt. "
            "Defaults to state/preflight/cancer_6core_relative_qkv_seed0.json."
        ),
    )
    parser.add_argument(
        "--prior-geometry-operator-receipt",
        type=Path,
        help=(
            "Immutable passing SO2 geometry-modulated operator preflight. "
            "Used only for the geometry-modulated SO1 configuration."
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
    prior_equivalence_receipt = (
        _runtime_path(DEFAULT_PRIOR_EQUIVALENCE_RECEIPT, paths)
        if args.prior_equivalence_receipt is None
        else args.prior_equivalence_receipt.resolve()
    )
    prior_geometry_operator_receipt = (
        _runtime_path(DEFAULT_PRIOR_GEOMETRY_OPERATOR_RECEIPT, paths)
        if args.prior_geometry_operator_receipt is None
        else args.prior_geometry_operator_receipt.resolve()
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
            prior_equivalence_receipt=prior_equivalence_receipt,
            prior_geometry_operator_receipt=prior_geometry_operator_receipt,
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
