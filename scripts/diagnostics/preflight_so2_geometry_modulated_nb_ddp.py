#!/usr/bin/env python3
"""Bounded four-rank preflight for the frozen SO2 geometry-NB2 campaign.

This diagnostic intentionally uses the production model, paired-update, and
fixed-validation implementations.  It writes only one atomic JSON receipt;
both checkpoints used by the reload gate live in a temporary directory and
are removed before the receipt is published.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.train.run_so2_geometry_modulated_nb import (  # noqa: E402
    CAMPAIGN_ID,
    CODE_HASH_SCOPE as RUNNER_CODE_HASH_SCOPE,
    MODEL_SEED,
    PREFLIGHT_CODE_FILES as RUNNER_PREFLIGHT_CODE_FILES,
    PREFLIGHT_MAX_VRAM_GIB,
    PREFLIGHT_MINIMUM_HEADROOM_GIB,
    PREFLIGHT_REQUIRED_GATES as RUNNER_PREFLIGHT_REQUIRED_GATES,
    PREFLIGHT_SCHEMA,
    PREFLIGHT_TRAINING_PAIR,
    VALIDATION_ALIASES,
    VISIBLE_DEVICES,
    WORLD_SIZE,
    _RawThetaParameterView,
    _all_rank_rng_states,
    _broadcast_rank_zero_result,
    _build_model,
    _checkpoint_payload_from_state,
    _checkpoint_state_payload,
    _distributed_identity,
    _evaluate_validation,
    _initialize_training,
    _paired_update,
    _runtime_path,
    _section,
    _synchronized_rank_zero_phase,
    validate_nb_config,
)
from spatial_benchmark.configuration import (  # noqa: E402
    compose_config,
    load_yaml_mapping,
)
from spatial_benchmark.gradient_direction_observability import (  # noqa: E402
    BlockGradientDirectionTracker,
    FullGradientDirectionTracker,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.negative_binomial import (  # noqa: E402
    NegativeBinomialModelOutput,
    ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
    masked_negative_binomial_nll,
    nb2_inverse_dispersion_from_raw,
    nb2_mean_from_logits,
)
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.so2_nb_data import (  # noqa: E402
    FROZEN_TASK_CONTRACT_SHA256,
    SO2NBDataBundle,
    SO2NBCoreBatch,
    load_so2_nb_data,
)
from spatial_benchmark.so2_nb_training import (  # noqa: E402
    AtomicBestLatestCheckpointStore,
    EXPECTED_PARAMETER_COUNT,
    PROTOCOL,
    EarlyStoppingState,
    SO2NBTrainingError,
    sha256_file,
    update_early_stopping,
)
from spatial_benchmark.training import _autocast_context  # noqa: E402


EXPECTED_BLOCKS = 4
EXPECTED_GENES = 1_000
EXPECTED_BLOCK_PARAMETER_COUNT = 831_880
MINIMUM_DEVICE_MEMORY_GIB = 23.0
REAL_SUBSET_MAX_CELLS = 64
REAL_SUBSET_MAX_GENES = 128
LOSS_DECREASE_STEPS = 24
LOSS_DECREASE_LEARNING_RATE = 5e-2
AMP_FP32_MU_MAX_ABS_TOLERANCE = 7.5e-2
AMP_FP32_MU_MEAN_ABS_TOLERANCE = 7.5e-3
AMP_FP32_NLL_ABS_TOLERANCE = 2.5e-2
AMP_FP32_NLL_RELATIVE_TOLERANCE = 2.5e-3
FROZEN_CONTRACT_RELATIVE_PATH = Path(
    "experiments/campaigns/"
    "cmp_20260907_so2_geometry_modulated_relative_qkv_nb_train12_val2_seed0/"
    "frozen_task_contract.yaml"
)
SOURCE_EXPERIMENT_CONFIG_RELATIVE_PATH = Path(
    "configs/experiment/so2_geometry_modulated_nb_train12_val2_seed0.yaml"
)
# Bind campaign numerics, data integrity/path resolution, runner archiving, and
# queue launch/finalization behavior into every accepted preflight receipt.
CODE_HASH_SCOPE = (
    "campaign_numerical_and_data_integrity_resolution_plus_runner_archive_"
    "and_queue_launch_finalization"
)
CODE_RELATIVE_PATHS = (
    Path("scripts/diagnostics/preflight_so2_geometry_modulated_nb_ddp.py"),
    Path("scripts/train/run_so2_geometry_modulated_nb.py"),
    Path("src/spatial_benchmark/negative_binomial.py"),
    Path(
        "src/spatial_benchmark/"
        "geometry_modulated_relative_qkv_graph_transformer.py"
    ),
    Path("src/spatial_benchmark/so2_nb_data.py"),
    Path("src/spatial_benchmark/so2_nb_training.py"),
    Path("src/spatial_benchmark/configuration.py"),
    Path("src/spatial_benchmark/gradient_direction_observability.py"),
    Path("src/spatial_benchmark/pooled_relative_qkv_training_v2.py"),
    Path("src/spatial_benchmark/pooled_relative_qkv_training.py"),
    Path("src/spatial_benchmark/adjacency_ablation.py"),
    Path("src/spatial_benchmark/masking.py"),
    Path("src/spatial_benchmark/training.py"),
    Path("src/spatial_benchmark/relative_qkv_graph_transformer.py"),
    Path("src/spatial_benchmark/models.py"),
    Path("src/spatial_benchmark/paths.py"),
    Path("src/spatial_benchmark/run_archive.py"),
    Path("src/spatial_benchmark/identifiers.py"),
    Path("src/spatial_benchmark/fingerprints.py"),
    Path("src/spatial_benchmark/so2_pooled_full_core.py"),
    Path("src/spatial_benchmark/so2_relative_graphs.py"),
    Path("src/spatial_benchmark/queueing.py"),
)
REQUIRED_GATE_NAMES = frozenset(
    {
        "configuration_and_overlay",
        "model_topology",
        "neutral_geometry_initialization",
        "raw_target_forward_isolation",
        "full_constant_fp32_nb2",
        "amp_fp32_equivalence",
        "synthetic_nb2_loss_decrease",
        "real_subset_nb2_loss_decrease",
        "ddp_parameter_sync",
        "complete_pair_training_update",
        "preclip_gradient_flow",
        "fixed_validation_mask_regeneration",
        "fixed_validation_forward",
        "checkpoint_best_latest_reload",
        "gpu_inventory",
        "peak_vram",
        "no_test_split_or_artifacts",
    }
)
if CODE_RELATIVE_PATHS != tuple(RUNNER_PREFLIGHT_CODE_FILES):
    raise RuntimeError("Standalone and production preflight code-hash inventories drifted.")
if REQUIRED_GATE_NAMES != frozenset(RUNNER_PREFLIGHT_REQUIRED_GATES):
    raise RuntimeError("Standalone and production required preflight gates drifted.")
if CODE_HASH_SCOPE != RUNNER_CODE_HASH_SCOPE:
    raise RuntimeError("Standalone and production preflight code-hash scopes drifted.")


class SO2NBPreflightError(RuntimeError):
    """Raised when any required launch gate fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace *path* with strict JSON and leave no writing file."""

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
        temporary.unlink(missing_ok=True)


def _tensor_sha256(tensor: Tensor) -> str:
    work = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(work.dtype).encode("ascii"))
    digest.update(json.dumps(list(work.shape), separators=(",", ":")).encode("ascii"))
    digest.update(work.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash names, types, shapes, and bytes of a tensor state mapping."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise SO2NBPreflightError(f"State entry {name!r} is not a tensor.")
        work = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(work.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(work.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(work.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _all_gather_objects(local: Any) -> list[Any]:
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather_object(gathered, local)
    return gathered


def _synchronized_state_receipt(model: nn.Module, *, phase: str) -> dict[str, Any]:
    local = _state_dict_sha256(model.state_dict())
    hashes = [str(value) for value in _all_gather_objects(local)]
    passed = len(hashes) == WORLD_SIZE and len(set(hashes)) == 1
    if not passed:
        raise SO2NBPreflightError(f"DDP parameters diverged during {phase}.")
    return {"passed": True, "phase": phase, "per_rank_sha256": hashes}


def _model_topology_receipt(model: nn.Module) -> dict[str, Any]:
    count = sum(parameter.numel() for parameter in model.parameters())
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise SO2NBPreflightError("Model does not expose blocks as a ModuleList.")
    ordered = tuple(blocks)
    block_counts = [
        sum(parameter.numel() for parameter in block.parameters())
        for block in ordered
    ]
    block_parameter_ids = [
        [id(parameter) for parameter in block.parameters()] for block in ordered
    ]
    flattened_ids = [value for values in block_parameter_ids for value in values]
    raw_theta = getattr(model, "raw_theta", None)
    passed = bool(
        isinstance(
            model,
            ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
        )
        and count == EXPECTED_PARAMETER_COUNT
        and len(ordered) == EXPECTED_BLOCKS
        and len({id(block) for block in ordered}) == EXPECTED_BLOCKS
        and len(flattened_ids) == len(set(flattened_ids))
        and block_counts
        == [EXPECTED_BLOCK_PARAMETER_COUNT for _ in range(EXPECTED_BLOCKS)]
        and isinstance(raw_theta, nn.Parameter)
        and raw_theta.numel() == EXPECTED_GENES
        and dict(model.named_parameters()).get("raw_theta") is raw_theta
    )
    if not passed:
        raise SO2NBPreflightError("Frozen NB2 model topology or parameter count drifted.")
    return {
        "passed": True,
        "model_class": type(model).__name__,
        "parameter_count": count,
        "graph_block_count": len(ordered),
        "distinct_graph_block_objects": len({id(block) for block in ordered}),
        "disjoint_graph_block_parameter_sets": True,
        "per_block_trainable_parameter_count": block_counts,
        "raw_theta_parameter_count": raw_theta.numel(),
        "raw_theta_registered_at_model_root": True,
    }


def _neutral_geometry_receipt(model: nn.Module, geometry: Tensor) -> dict[str, Any]:
    geometry = geometry.detach().cpu().float()
    if geometry.ndim != 2 or geometry.shape[1] != 70 or not geometry.shape[0]:
        raise SO2NBPreflightError("Neutrality check requires nonempty [E,70] geometry.")
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, block in enumerate(model.blocks):
            encoder = block.geometry_encoder
            modulation, bias = encoder(geometry)
            modulation_weight = encoder.modulation_projection.weight
            bias_weight = encoder.bias_projection.weight
            record = {
                "block_index": index,
                "modulation_projection_exactly_zero": bool(
                    torch.count_nonzero(modulation_weight) == 0
                ),
                "bias_projection_exactly_zero": bool(
                    torch.count_nonzero(bias_weight) == 0
                ),
                "modulation_max_abs_from_one": float(
                    (modulation - 1.0).abs().max()
                ),
                "bias_max_abs": float(bias.abs().max()),
            }
            record["passed"] = bool(
                record["modulation_projection_exactly_zero"]
                and record["bias_projection_exactly_zero"]
                and record["modulation_max_abs_from_one"] == 0.0
                and record["bias_max_abs"] == 0.0
            )
            records.append(record)
    theta = model.theta.detach().cpu().float()
    theta_error = float((theta - 1.0).abs().max())
    passed = bool(
        len(records) == EXPECTED_BLOCKS
        and all(record["passed"] for record in records)
        and theta_error <= 2e-7
    )
    if not passed:
        raise SO2NBPreflightError("Geometry or inverse dispersion is not neutral.")
    return {
        "passed": True,
        "blocks": records,
        "inverse_dispersion_initial_max_abs_from_one": theta_error,
    }


def _small_induced_graph(
    batch: SO2NBCoreBatch,
    *,
    maximum_nodes: int = 16,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    nodes = min(int(maximum_nodes), batch.n_nodes)
    expression = batch.input_expression[:nodes].clone()
    covariates = batch.node_covariates[:nodes].clone()
    raw_target = batch.raw_count_target[:nodes].clone()
    keep = (batch.edge_index[0] < nodes) & (batch.edge_index[1] < nodes)
    edges = batch.edge_index[:, keep].clone()
    geometry = batch.relative_geometry[keep].clone().float()
    if edges.shape[1] == 0:
        indices = torch.arange(nodes, dtype=torch.long)
        edges = torch.stack((indices, indices.roll(1)), dim=0)
        geometry = torch.zeros((nodes, 70), dtype=torch.float32)
    return expression, raw_target, covariates, edges, geometry


def _raw_target_isolation_receipt(
    model: nn.Module,
    batch: SO2NBCoreBatch,
) -> dict[str, Any]:
    """Prove target perturbation cannot enter the architecture's forward path."""

    expression, raw_a, covariates, edges, geometry = _small_induced_graph(batch)
    mask = torch.zeros_like(expression, dtype=torch.bool)
    mask[:, ::7] = True
    masked_input = expression.masked_fill(mask, 0.0)
    raw_b = raw_a.clone()
    raw_b[mask] = torch.remainder(raw_b[mask].to(torch.int64) + 17, 730).to(
        torch.int32
    )
    forbidden = {
        "raw_target",
        "raw_count_target",
        "target_counts",
        "library_total",
        "size_factor",
        "offset",
    }
    parameter_names = set(inspect.signature(model.forward).parameters)
    model.eval()
    with torch.no_grad():
        first = model(
            input_expression=masked_input,
            gene_mask=mask,
            edge_index=edges,
            relative_geometry=geometry,
            node_covariates=covariates,
        )
        second = model(
            input_expression=masked_input,
            gene_mask=mask,
            edge_index=edges,
            relative_geometry=geometry,
            node_covariates=covariates,
        )
    if not isinstance(first, NegativeBinomialModelOutput) or not isinstance(
        second, NegativeBinomialModelOutput
    ):
        raise SO2NBPreflightError("Raw-target isolation got the wrong model output.")
    targets_differ = not torch.equal(raw_a, raw_b)
    means_equal = torch.equal(first.mu, second.mu)
    passed = bool(not (parameter_names & forbidden) and targets_differ and means_equal)
    if not passed:
        raise SO2NBPreflightError("Raw-count target can affect or enter model.forward.")
    return {
        "passed": True,
        "forward_parameter_names": sorted(parameter_names),
        "forbidden_forward_parameters_absent": True,
        "perturbed_raw_target_sha256": [_tensor_sha256(raw_a), _tensor_sha256(raw_b)],
        "forward_mu_sha256": [_tensor_sha256(first.mu), _tensor_sha256(second.mu)],
        "forward_mu_bit_identical": True,
        "hidden_target_total_or_offset_forwarded": False,
    }


def _small_real_edge_subgraph(
    batch: SO2NBCoreBatch,
    *,
    maximum_edges: int = 128,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Select a bounded, remapped subgraph while preserving real edge geometry."""

    edge_count = min(int(maximum_edges), batch.n_edges)
    if edge_count <= 0:
        raise SO2NBPreflightError("AMP equivalence requires real graph edges.")
    selected_edges = batch.edge_index[:, :edge_count].clone()
    selected_geometry = batch.relative_geometry[:edge_count].clone().float()
    node_ids = torch.unique(selected_edges.reshape(-1), sorted=True)
    remap = torch.full((batch.n_nodes,), -1, dtype=torch.long)
    remap[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long)
    local_edges = remap[selected_edges]
    if bool((local_edges < 0).any()) or bool(
        (local_edges[0] == local_edges[1]).any()
    ):
        raise SO2NBPreflightError("Bounded real subgraph remapping failed.")
    return (
        batch.input_expression.index_select(0, node_ids).clone(),
        batch.raw_count_target.index_select(0, node_ids).clone(),
        batch.node_covariates.index_select(0, node_ids).clone(),
        local_edges,
        selected_geometry,
    )


def _absolute_difference(reference: Tensor, observed: Tensor) -> dict[str, float]:
    if reference.shape != observed.shape:
        raise SO2NBPreflightError("AMP/FP32 comparison tensor shapes differ.")
    difference = (reference.detach().float() - observed.detach().float()).abs()
    return {
        "maximum_absolute_difference": float(difference.max().cpu()),
        "mean_absolute_difference": float(difference.mean().cpu()),
    }


def _amp_fp32_equivalence_receipt(
    model: nn.Module,
    batch: SO2NBCoreBatch,
    *,
    device: torch.device,
    amp_dtype: str,
    staged_geometry_dtype: str,
) -> dict[str, Any]:
    """Compare deterministic real-subgraph mean and FP32 NB2 loss under AMP."""

    expression, raw_target, covariates, edges, geometry = _small_real_edge_subgraph(
        batch
    )
    expression = expression.to(device=device, dtype=torch.float32)
    raw_target = raw_target.to(device=device, dtype=torch.int32)
    covariates = covariates.to(device=device, dtype=torch.float32)
    edges = edges.to(device=device, dtype=torch.long)
    geometry_fp32 = geometry.to(device=device, dtype=torch.float32)
    amp_geometry_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(str(staged_geometry_dtype).lower())
    if amp_geometry_dtype is None:
        raise SO2NBPreflightError("Unknown staged geometry dtype for AMP check.")
    geometry_amp = geometry_fp32.to(
        dtype=amp_geometry_dtype if device.type == "cuda" else torch.float32
    )
    rows = torch.arange(expression.shape[0], device=device).view(-1, 1)
    genes = torch.arange(expression.shape[1], device=device).view(1, -1)
    mask = ((rows * 17 + genes * 13) % 7) == 0
    masked_input = expression.masked_fill(mask, 0.0)
    target_nodes = torch.arange(expression.shape[0], device=device, dtype=torch.long)
    previous_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            fp32_output = model(
                input_expression=masked_input,
                gene_mask=mask,
                edge_index=edges,
                relative_geometry=geometry_fp32,
                node_covariates=covariates,
                target_nodes=target_nodes,
            )
            with _autocast_context(
                enabled=True,
                device=device,
                dtype_name=str(amp_dtype),
            ):
                amp_output = model(
                    input_expression=masked_input,
                    gene_mask=mask,
                    edge_index=edges,
                    relative_geometry=geometry_amp,
                    node_covariates=covariates,
                    target_nodes=target_nodes,
                )
            if not isinstance(fp32_output, NegativeBinomialModelOutput) or not isinstance(
                amp_output, NegativeBinomialModelOutput
            ):
                raise SO2NBPreflightError("AMP equivalence received wrong model output.")
            fp32_loss = masked_negative_binomial_nll(
                fp32_output.mu, fp32_output.theta, raw_target, mask
            )
            amp_loss = masked_negative_binomial_nll(
                amp_output.mu, amp_output.theta, raw_target, mask
            )
    finally:
        model.train(previous_training)
    mean_difference = _absolute_difference(fp32_output.mu, amp_output.mu)
    loss_absolute_difference = abs(float(fp32_loss) - float(amp_loss))
    loss_relative_difference = loss_absolute_difference / max(
        abs(float(fp32_loss)), 1e-12
    )
    loss_combined_tolerance = AMP_FP32_NLL_ABS_TOLERANCE + (
        AMP_FP32_NLL_RELATIVE_TOLERANCE * abs(float(fp32_loss))
    )
    passed = bool(
        fp32_output.mu.dtype == torch.float32
        and amp_output.mu.dtype == torch.float32
        and fp32_loss.dtype == torch.float32
        and amp_loss.dtype == torch.float32
        and torch.isfinite(fp32_output.mu).all()
        and torch.isfinite(amp_output.mu).all()
        and torch.isfinite(fp32_loss)
        and torch.isfinite(amp_loss)
        and mean_difference["maximum_absolute_difference"]
        <= AMP_FP32_MU_MAX_ABS_TOLERANCE
        and mean_difference["mean_absolute_difference"]
        <= AMP_FP32_MU_MEAN_ABS_TOLERANCE
        and loss_absolute_difference <= loss_combined_tolerance
    )
    if not passed:
        raise SO2NBPreflightError("AMP and FP32 mean/loss equivalence gate failed.")
    return {
        "passed": True,
        "source_alias": batch.alias,
        "comparison_nodes": int(expression.shape[0]),
        "comparison_edges": int(edges.shape[1]),
        "masked_entries": int(mask.sum()),
        "amp_dtype": str(amp_dtype),
        "staged_amp_geometry_dtype": str(staged_geometry_dtype),
        "output_mean_dtype": "float32",
        "likelihood_dtype": "float32_outside_autocast",
        "mean": {
            **mean_difference,
            "maximum_absolute_tolerance": AMP_FP32_MU_MAX_ABS_TOLERANCE,
            "mean_absolute_tolerance": AMP_FP32_MU_MEAN_ABS_TOLERANCE,
        },
        "masked_full_constant_nb2_nll": {
            "fp32": float(fp32_loss),
            "amp": float(amp_loss),
            "absolute_difference": loss_absolute_difference,
            "relative_difference": loss_relative_difference,
            "absolute_tolerance": AMP_FP32_NLL_ABS_TOLERANCE,
            "relative_tolerance": AMP_FP32_NLL_RELATIVE_TOLERANCE,
            "combined_absolute_tolerance": loss_combined_tolerance,
            "tolerance_rule": "abs_delta_le_atol_plus_rtol_times_abs_fp32",
        },
        "gradient_comparison_performed": False,
        "gradient_gate_separate": "preclip_gradient_flow",
    }


def _loss_decrease_check(
    raw_counts: Tensor,
    *,
    steps: int = LOSS_DECREASE_STEPS,
    learning_rate: float = LOSS_DECREASE_LEARNING_RATE,
) -> dict[str, Any]:
    """Fit bounded free NB2 means/dispersion as an implementation smoke test."""

    counts = raw_counts.detach().cpu().to(torch.int32)
    if counts.ndim != 2 or not all(counts.shape) or bool((counts < 0).any()):
        raise SO2NBPreflightError("Loss-decrease counts must be nonempty and nonnegative.")
    logits = nn.Parameter(torch.zeros(counts.shape, dtype=torch.float32))
    theta_initial = math.log(math.expm1(1.0 - 1e-4))
    raw_theta = nn.Parameter(
        torch.full((counts.shape[1],), theta_initial, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam((logits, raw_theta), lr=float(learning_rate))
    mask = torch.ones_like(counts, dtype=torch.bool)

    def objective() -> Tensor:
        mu = nb2_mean_from_logits(logits)
        theta = nb2_inverse_dispersion_from_raw(raw_theta)
        return masked_negative_binomial_nll(mu, theta, counts, mask)

    initial = float(objective().detach())
    gradients_finite_nonzero = True
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
            raise SO2NBPreflightError("NB2 likelihood is not finite FP32.")
        loss.backward()
        for parameter in (logits, raw_theta):
            gradients_finite_nonzero = bool(
                gradients_finite_nonzero
                and parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and torch.linalg.vector_norm(parameter.grad) > 0
            )
        optimizer.step()
    final = float(objective().detach())
    passed = bool(
        gradients_finite_nonzero
        and math.isfinite(initial)
        and math.isfinite(final)
        and final < initial
    )
    if not passed:
        raise SO2NBPreflightError("Bounded NB2 optimization did not decrease loss.")
    return {
        "passed": True,
        "shape": list(counts.shape),
        "minimum_count": int(counts.min()),
        "maximum_count": int(counts.max()),
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "initial_full_constant_masked_nb2_nll": initial,
        "final_full_constant_masked_nb2_nll": final,
        "loss_decrease": initial - final,
        "likelihood_dtype": "float32",
        "finite_nonzero_gradients": True,
    }


def _synthetic_loss_decrease_receipt() -> dict[str, Any]:
    counts = torch.tensor(
        [
            [0, 1, 2, 5, 729],
            [7, 0, 19, 3, 128],
            [1, 4, 0, 31, 512],
            [2, 8, 16, 0, 255],
        ],
        dtype=torch.int32,
    )
    result = _loss_decrease_check(counts)
    result["contains_zero_and_frozen_maximum_729"] = True
    return result


def _real_subset_loss_decrease_receipt(batch: SO2NBCoreBatch) -> dict[str, Any]:
    counts = batch.raw_count_target[
        : min(REAL_SUBSET_MAX_CELLS, batch.n_nodes),
        : min(REAL_SUBSET_MAX_GENES, batch.n_genes),
    ]
    result = _loss_decrease_check(counts)
    result.update({"source_alias": batch.alias, "raw_target_transform": "none"})
    return result


def _gradient_snapshot(module: nn.Module, *, label: str) -> dict[str, Any]:
    parameters = tuple(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    if not parameters:
        raise SO2NBPreflightError(f"{label} has no trainable parameters.")
    missing = [name for name, parameter in parameters if parameter.grad is None]
    if missing:
        raise SO2NBPreflightError(f"{label} has missing gradients: {missing}.")
    squared = torch.zeros((), dtype=torch.float64, device=parameters[0][1].device)
    with torch.no_grad():
        for name, parameter in parameters:
            assert parameter.grad is not None
            gradient = parameter.grad.detach()
            if not bool(torch.isfinite(gradient).all()):
                raise SO2NBPreflightError(f"{label}.{name} gradient is non-finite.")
            squared.add_(gradient.square().sum(dtype=torch.float64))
    norm = float(torch.sqrt(squared).cpu())
    if not math.isfinite(norm) or norm <= 0.0:
        raise SO2NBPreflightError(f"{label} gradient norm is not finite and positive.")
    return {
        "name": label,
        "trainable_parameter_count": sum(p.numel() for _, p in parameters),
        "trainable_parameter_tensors": len(parameters),
        "parameters_missing_gradient": 0,
        "gradient_norm_before_clip": norm,
        "finite_nonzero": True,
        "read_timing": "post_amp_unscale_pre_gradient_clip_pre_optimizer_step",
    }


class _CapturingBlockTracker(BlockGradientDirectionTracker):
    snapshots: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(
            expected_blocks=EXPECTED_BLOCKS,
            expected_optimizer_updates_per_epoch=1,
        )
        self.snapshots = []

    def observe(self, model: nn.Module, context: Any) -> None:
        self.snapshots = [
            _gradient_snapshot(block, label=f"blocks.{index}")
            for index, block in enumerate(model.blocks)
        ]
        super().observe(model, context)


class _CapturingThetaTracker(FullGradientDirectionTracker):
    snapshot: dict[str, Any] | None

    def __init__(self) -> None:
        super().__init__(expected_optimizer_updates_per_epoch=1)
        self.snapshot = None

    def observe(self, model: nn.Module, context: Any) -> None:
        self.snapshot = _gradient_snapshot(model, label="raw_theta")
        super().observe(model, context)


def _fixed_mask_regeneration_receipt(
    bundle: SO2NBDataBundle,
    *,
    rank: int,
) -> dict[str, Any]:
    alias = VALIDATION_ALIASES[0 if rank < 2 else 1]
    view_index = (rank % 2) * 5
    first = bundle.validation_mask(alias, view_index).to_receipt()
    second = bundle.validation_mask(alias, view_index).to_receipt()
    local = {
        "rank": rank,
        "alias": alias,
        "view_index": view_index,
        "receipt_sha256": first["receipt_sha256"],
        "mask_realization_sha256": first["mask_realization_sha256"],
        "passed": first == second,
    }
    records = _all_gather_objects(local)
    passed = len(records) == WORLD_SIZE and all(record["passed"] for record in records)
    if not passed:
        raise SO2NBPreflightError("Fixed validation-mask regeneration drifted.")
    return {"passed": True, "per_rank_regeneration_checks": records}


def _gpu_identity(device: torch.device, *, rank: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "rank": rank,
        "local_rank": int(device.index),
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "total_memory_gib": float(properties.total_memory) / float(1024**3),
        "compute_capability": [properties.major, properties.minor],
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def _gpu_inventory_from_identities(
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    names = {str(record["name"]) for record in identities}
    local_ranks = {int(record["local_rank"]) for record in identities}
    passed = bool(
        len(identities) == WORLD_SIZE
        and len(names) == 1
        and names == {"NVIDIA GeForce RTX 3090"}
        and local_ranks == set(range(WORLD_SIZE))
        and all(record.get("compute_capability") == [8, 6] for record in identities)
        and all(record.get("torch_version") == torch.__version__ for record in identities)
        and all(record.get("cuda_runtime") == torch.version.cuda for record in identities)
        and all(
            int(record["total_memory_bytes"]) > 0
            and math.isclose(
                float(record["total_memory_gib"]),
                int(record["total_memory_bytes"]) / float(1024**3),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and float(record["total_memory_gib"]) >= MINIMUM_DEVICE_MEMORY_GIB
            for record in identities
        )
    )
    if not passed:
        raise SO2NBPreflightError("GPU inventory differs from the frozen DDP4 contract.")
    return {"passed": True, "devices": list(identities)}


def _gpu_inventory_receipt(device: torch.device, *, rank: int) -> dict[str, Any]:
    identities = _all_gather_objects(_gpu_identity(device, rank=rank))
    return _gpu_inventory_from_identities(identities)


def _vram_receipt_from_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(records) != WORLD_SIZE:
        raise SO2NBPreflightError("VRAM evidence requires exactly four ranks.")
    normalized: list[dict[str, Any]] = []
    for record in records:
        total_bytes = int(record["total_memory_bytes"])
        total = total_bytes / float(1024**3)
        allocated = float(record["peak_allocated_vram_gib"])
        reserved = float(record["peak_reserved_vram_gib"])
        normalized.append(
            {
                "rank": int(record["rank"]),
                "local_rank": int(record["local_rank"]),
                "total_memory_bytes": total_bytes,
                "total_memory_gib": total,
                "peak_allocated_vram_gib": allocated,
                "peak_reserved_vram_gib": reserved,
                "allocated_vram_headroom_gib": total - allocated,
                "reserved_vram_headroom_gib": total - reserved,
            }
        )
    if {record["rank"] for record in normalized} != set(range(WORLD_SIZE)):
        raise SO2NBPreflightError("VRAM evidence rank coverage is incomplete.")
    peak = max(record["peak_allocated_vram_gib"] for record in normalized)
    peak_reserved = max(record["peak_reserved_vram_gib"] for record in normalized)
    minimum_allocated_headroom = min(
        record["allocated_vram_headroom_gib"] for record in normalized
    )
    minimum_reserved_headroom = min(
        record["reserved_vram_headroom_gib"] for record in normalized
    )
    passed = bool(
        math.isfinite(peak)
        and peak > 0.0
        and peak <= PREFLIGHT_MAX_VRAM_GIB
        and math.isfinite(peak_reserved)
        and peak_reserved >= peak
        and all(
            record["total_memory_gib"] >= MINIMUM_DEVICE_MEMORY_GIB
            and 0.0 < record["peak_allocated_vram_gib"]
            <= record["peak_reserved_vram_gib"]
            <= record["total_memory_gib"]
            for record in normalized
        )
        and minimum_allocated_headroom >= PREFLIGHT_MINIMUM_HEADROOM_GIB
        and minimum_reserved_headroom >= PREFLIGHT_MINIMUM_HEADROOM_GIB
    )
    if not passed:
        raise SO2NBPreflightError("Peak VRAM or minimum-headroom gate failed.")
    return {
        "passed": True,
        "per_rank": normalized,
        "peak_vram_gib_all_ranks": peak,
        "peak_reserved_vram_gib_all_ranks": peak_reserved,
        "minimum_vram_headroom_gib_each_rank": minimum_allocated_headroom,
        "minimum_reserved_vram_headroom_gib_each_rank": minimum_reserved_headroom,
        "maximum_peak_vram_gib": PREFLIGHT_MAX_VRAM_GIB,
        "minimum_required_headroom_gib": PREFLIGHT_MINIMUM_HEADROOM_GIB,
        "headroom_uses_observed_device_total_bytes": True,
    }


def _peak_vram_receipt(device: torch.device, *, rank: int) -> dict[str, Any]:
    divisor = float(1024**3)
    local_allocated = float(torch.cuda.max_memory_allocated(device)) / divisor
    local_reserved = float(torch.cuda.max_memory_reserved(device)) / divisor
    local_total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    records = _all_gather_objects(
        {
            "rank": rank,
            "local_rank": int(device.index),
            "total_memory_bytes": local_total_bytes,
            "peak_allocated_vram_gib": local_allocated,
            "peak_reserved_vram_gib": local_reserved,
        }
    )
    return _vram_receipt_from_records(records)


def _overlay_evidence(bundle: SO2NBDataBundle) -> dict[str, Any]:
    try:
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SO2NBPreflightError("Verified overlay manifest became unreadable.") from exc
    if not isinstance(manifest, Mapping):
        raise SO2NBPreflightError("Overlay manifest root is not a mapping.")
    unsigned = dict(manifest)
    content_hash = unsigned.pop("manifest_content_sha256", None)
    if content_hash != canonical_sha256(unsigned):
        raise SO2NBPreflightError("Overlay manifest content hash drifted.")
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise SO2NBPreflightError("Overlay lacks verified source-artifact hashes.")
    return {
        "manifest_path": str(bundle.manifest_path),
        "manifest_file_sha256": bundle.manifest_sha256,
        "manifest_content_sha256": content_hash,
        "split_fingerprint": bundle.split_fingerprint,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "source_artifacts": json.loads(json.dumps(source_artifacts)),
    }


def _configuration_file_hashes(config_path: Path, config_root: Path) -> dict[str, str]:
    """Hash the root config and all directly named default group files."""

    files = {config_path.resolve(strict=True)}
    root = load_yaml_mapping(config_path)
    defaults = root.get("defaults", ())
    if not isinstance(defaults, Sequence) or isinstance(defaults, (str, bytes)):
        raise SO2NBPreflightError("Experiment defaults must be a sequence.")
    for entry in defaults:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise SO2NBPreflightError("Experiment default entry has invalid shape.")
        group, name = next(iter(entry.items()))
        candidate = config_root / str(group) / f"{name}.yaml"
        files.add(candidate.resolve(strict=True))
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in sorted(files)
    }


def _validate_amp_configuration(config: Mapping[str, Any]) -> None:
    trainer = _section(config, "trainer")
    expected = {
        "amp": True,
        "likelihood_compute_dtype": "float32",
        "likelihood_outside_autocast": True,
        "amp_requires_fp32_equivalence_preflight": True,
    }
    for field, value in expected.items():
        if trainer.get(field) != value:
            raise SO2NBPreflightError(
                f"trainer.{field} must be {value!r}; got {trainer.get(field)!r}."
            )
    metadata = _section(config, "metadata")
    acceptance = metadata.get("preflight_acceptance")
    if not isinstance(acceptance, Mapping):
        raise SO2NBPreflightError("metadata.preflight_acceptance must be a mapping.")
    if (
        acceptance.get("minimum_observed_cuda_memory_gib_each_rank")
        != MINIMUM_DEVICE_MEMORY_GIB
        or "gpu_memory_gib_each_rank" in acceptance
    ):
        raise SO2NBPreflightError(
            "Preflight metadata must require >=23 observed CUDA GiB and must not "
            "claim an exact 24 GiB torch-visible capacity."
        )


def _code_file_hashes() -> dict[str, str]:
    return {
        str(relative): sha256_file(PROJECT_ROOT / relative)
        for relative in CODE_RELATIVE_PATHS
    }


def _checkpoint_reload_receipt(
    *,
    output_parent: Path,
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
    checkpoint_state: Mapping[str, Any],
) -> dict[str, Any]:
    def checkpoint_phase(name: str) -> None:
        print(
            json.dumps(
                {"preflight_checkpoint_phase": name, "timestamp": _utc_now()},
                sort_keys=True,
            ),
            flush=True,
        )

    with tempfile.TemporaryDirectory(
        prefix=".so2-nb-ddp4-preflight-checkpoint-", dir=output_parent
    ) as temporary:
        store = AtomicBestLatestCheckpointStore(temporary)
        best_payload_to_save = _checkpoint_payload_from_state(
            checkpoint_state,
            role="best",
            best_checkpoint_sha256=None,
        )
        best = store.save_best(best_payload_to_save)
        checkpoint_phase("best_saved_and_reloaded")
        latest_payload_to_save = _checkpoint_payload_from_state(
            checkpoint_state,
            role="latest",
            best_checkpoint_sha256=best.sha256,
            embedded_best_checkpoint=best_payload_to_save,
        )
        checkpoint_phase("latest_payload_built")
        latest = store.save_latest(
            latest_payload_to_save
        )
        checkpoint_phase("latest_saved_and_reloaded")
        active_before = sorted(path.name for path in store.directory.iterdir())
        best_payload = store.load("best")
        latest_payload = store.load("latest")
        clone = _build_model(config, bundle)
        clone.load_state_dict(best_payload["model_state_dict"], strict=True)
        expected_hash = _state_dict_sha256(checkpoint_state["model_state_dict"])
        best_hash = _state_dict_sha256(clone.state_dict())
        latest_hash = _state_dict_sha256(latest_payload["model_state_dict"])
        final = store.finalize_best_only()
        final_payload = store.load("best")
        final_hash = _state_dict_sha256(final_payload["model_state_dict"])
        checkpoint_phase("best_latest_final_hashes_verified")
        active_after = sorted(path.name for path in store.directory.iterdir())
        passed = bool(
            active_before == ["best.ckpt", "latest.ckpt"]
            and active_after == ["best.ckpt"]
            and expected_hash == best_hash == latest_hash == final_hash
            and final.sha256 == best.sha256
        )
        if not passed:
            raise SO2NBPreflightError("Best/latest checkpoint reload gate failed.")
        result = {
            "passed": True,
            "checkpoint_schema": best_payload["checkpoint_schema"],
            "active_checkpoint_count_before_finalization": len(active_before),
            "retained_checkpoint_count_after_finalization": len(active_after),
            "roles_reloaded": ["best", "latest", "final_best"],
            "best_checkpoint_sha256": best.sha256,
            "latest_checkpoint_sha256": latest.sha256,
            "model_state_sha256": expected_hash,
            "final_role": "validation_best",
            "temporary_checkpoints_removed_on_context_exit": True,
        }
    return result


def _resolve_required_paths(
    args: argparse.Namespace,
    *,
    paths: ProjectPaths,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = paths.project_root / config_path
    config_path = config_path.resolve(strict=True)
    expected_config_path = (
        paths.project_root / SOURCE_EXPERIMENT_CONFIG_RELATIVE_PATH
    ).resolve(strict=True)
    if config_path != expected_config_path:
        raise SO2NBPreflightError(
            f"--config must be the frozen source experiment: {expected_config_path}"
        )
    config = compose_config(config_path, config_root=paths.config_root)
    validate_nb_config(config)
    _validate_amp_configuration(config)
    dataset = _section(config, "dataset")
    launcher = _section(config, "launcher")
    expected_overlay = _runtime_path(dataset["prepared_artifact_reference"], paths)
    expected_output = _runtime_path(launcher["hardware_preflight_receipt"], paths)

    def resolve_cli(path: Path) -> Path:
        expanded = path.expanduser()
        return (
            expanded.resolve(strict=False)
            if expanded.is_absolute()
            else (paths.project_root / expanded).resolve(strict=False)
        )

    overlay = resolve_cli(args.overlay)
    output = resolve_cli(args.output)
    if overlay != expected_overlay:
        raise SO2NBPreflightError(
            f"--overlay must be the configured artifact: {expected_overlay}"
        )
    if output != expected_output:
        raise SO2NBPreflightError(
            f"--output must be the configured preflight receipt: {expected_output}"
        )
    contract = paths.project_root / FROZEN_CONTRACT_RELATIVE_PATH
    if sha256_file(contract) != FROZEN_TASK_CONTRACT_SHA256:
        raise SO2NBPreflightError("Frozen task-contract file checksum drifted.")
    return config_path, overlay, output, config


def _validate_receipt(receipt: Mapping[str, Any]) -> None:
    expected = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "passed",
        "passed": True,
        "all_required_gates_passed": True,
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "world_size": WORLD_SIZE,
        "control_plane_backend": "gloo",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "receiver_chunk_size": 128,
        "train_update_completed": True,
        "validation_completed": True,
        "checkpoint_reload_verified": True,
        "test_artifacts_present": False,
        "code_hash_scope": CODE_HASH_SCOPE,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise SO2NBPreflightError(
                f"Receipt {field} must be {value!r}; got {receipt.get(field)!r}."
            )
    hash_fields = (
        "configuration_sha256",
        "resolved_config_sha256",
        "configuration_file_sha256",
        "frozen_task_contract_sha256",
        "overlay_manifest_sha256",
        "overlay_manifest_content_sha256",
        "preprocessing_fingerprint",
        "split_fingerprint",
    )
    if any(
        not isinstance(receipt.get(field), str)
        or len(str(receipt[field])) != 64
        or any(character not in "0123456789abcdef" for character in str(receipt[field]))
        for field in hash_fields
    ):
        raise SO2NBPreflightError("Receipt is missing a required SHA-256 field.")
    if receipt["configuration_sha256"] != receipt["resolved_config_sha256"]:
        raise SO2NBPreflightError("Canonical resolved-configuration hashes disagree.")
    if receipt["frozen_task_contract_sha256"] != FROZEN_TASK_CONTRACT_SHA256:
        raise SO2NBPreflightError("Receipt names the wrong frozen task contract.")
    code_hashes = receipt.get("code_file_sha256")
    if (
        not isinstance(code_hashes, Mapping)
        or set(code_hashes) != {str(path) for path in CODE_RELATIVE_PATHS}
        or any(not isinstance(value, str) or len(value) != 64 for value in code_hashes.values())
    ):
        raise SO2NBPreflightError("Receipt code-file hashes are incomplete.")
    if dict(code_hashes) != _code_file_hashes():
        raise SO2NBPreflightError("Receipt code-file hashes drifted.")
    config_hashes = receipt.get("configuration_source_file_sha256")
    if not isinstance(config_hashes, Mapping) or not config_hashes or any(
        not isinstance(value, str) or len(value) != 64
        for value in config_hashes.values()
    ):
        raise SO2NBPreflightError("Receipt configuration-file hashes are incomplete.")
    expected_config_path = PROJECT_ROOT / SOURCE_EXPERIMENT_CONFIG_RELATIVE_PATH
    expected_config_hashes = _configuration_file_hashes(
        expected_config_path, PROJECT_ROOT / "configs"
    )
    if dict(config_hashes) != expected_config_hashes or receipt[
        "configuration_file_sha256"
    ] != sha256_file(expected_config_path):
        raise SO2NBPreflightError("Receipt configuration-file hashes drifted.")
    gates = receipt.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != REQUIRED_GATE_NAMES:
        raise SO2NBPreflightError("Receipt does not contain the exact required gates.")
    if any(
        not isinstance(gate, Mapping) or gate.get("passed") is not True
        for gate in gates.values()
    ):
        raise SO2NBPreflightError("Receipt contains a failed required gate.")
    gpu_identities = receipt.get("gpu_identities")
    gpu_gate_devices = gates["gpu_inventory"].get("devices")
    if (
        not isinstance(gpu_identities, Sequence)
        or isinstance(gpu_identities, (str, bytes))
        or len(gpu_identities) != WORLD_SIZE
        or list(gpu_identities) != gpu_gate_devices
    ):
        raise SO2NBPreflightError(
            "Receipt GPU identities must equal the four inventory-gate devices."
        )
    _gpu_inventory_from_identities(gpu_identities)
    peak_gate = gates["peak_vram"]
    per_rank_vram = peak_gate.get("per_rank")
    if not isinstance(per_rank_vram, Sequence) or isinstance(
        per_rank_vram, (str, bytes)
    ):
        raise SO2NBPreflightError("Receipt lacks per-rank VRAM evidence.")
    recomputed_vram = _vram_receipt_from_records(per_rank_vram)
    identity_by_rank = {
        int(record["rank"]): record for record in gpu_identities
    }
    vram_by_rank = {int(record["rank"]): record for record in per_rank_vram}
    if set(identity_by_rank) != set(range(WORLD_SIZE)) or set(vram_by_rank) != set(
        range(WORLD_SIZE)
    ):
        raise SO2NBPreflightError("GPU identity/VRAM rank mapping is incomplete.")
    for rank in range(WORLD_SIZE):
        identity = identity_by_rank[rank]
        vram = vram_by_rank[rank]
        if (
            int(identity["local_rank"]) != int(vram["local_rank"])
            or int(identity["total_memory_bytes"])
            != int(vram["total_memory_bytes"])
        ):
            raise SO2NBPreflightError(
                "GPU identity and VRAM device mappings disagree."
            )
    peak = float(receipt.get("peak_vram_gib_all_ranks", float("nan")))
    reserved = float(
        receipt.get("peak_reserved_vram_gib_all_ranks", float("nan"))
    )
    headroom = float(
        receipt.get("minimum_vram_headroom_gib_each_rank", float("nan"))
    )
    reserved_headroom = float(
        receipt.get("minimum_reserved_vram_headroom_gib_each_rank", float("nan"))
    )
    if (
        not math.isfinite(peak)
        or peak <= 0.0
        or peak > PREFLIGHT_MAX_VRAM_GIB
        or not math.isfinite(headroom)
        or headroom < PREFLIGHT_MINIMUM_HEADROOM_GIB
        or not math.isfinite(reserved)
        or reserved < peak
        or reserved > PREFLIGHT_MAX_VRAM_GIB
        or not math.isfinite(reserved_headroom)
        or reserved_headroom < PREFLIGHT_MINIMUM_HEADROOM_GIB
        or not math.isclose(
            peak,
            float(recomputed_vram["peak_vram_gib_all_ranks"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            reserved,
            float(recomputed_vram["peak_reserved_vram_gib_all_ranks"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            headroom,
            float(recomputed_vram["minimum_vram_headroom_gib_each_rank"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            reserved_headroom,
            float(
                recomputed_vram["minimum_reserved_vram_headroom_gib_each_rank"]
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SO2NBPreflightError("Receipt VRAM fields fail the frozen gate.")
    stored_hash = receipt.get("receipt_content_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_content_sha256", None)
    if not isinstance(stored_hash, str) or stored_hash != canonical_sha256(unsigned):
        raise SO2NBPreflightError("Receipt self-checksum is invalid.")


def run_preflight(
    *,
    config_path: Path,
    overlay: Path,
    output: Path,
    config: dict[str, Any],
    paths: ProjectPaths,
    rank: int,
    local_rank: int,
    control_group: Any,
) -> dict[str, Any] | None:
    def phase(name: str) -> None:
        if rank == 0:
            print(
                json.dumps(
                    {"preflight_phase": name, "timestamp": _utc_now()},
                    sort_keys=True,
                ),
                flush=True,
            )

    phase("loading_and_validating_data")
    _validate_amp_configuration(config)
    dataset = _section(config, "dataset")
    bundle = load_so2_nb_data(
        overlay,
        cohort_dir=_runtime_path(dataset["source_cohort_artifact"], paths),
        graph_dir=_runtime_path(dataset["source_graph_artifact"], paths),
    )
    overlay_receipt = _overlay_evidence(bundle)
    model = _build_model(config, bundle)
    topology = _model_topology_receipt(model)
    phase("rank_zero_cpu_gates")

    rank_zero_cpu_checks: dict[str, Any] | None = None
    with _synchronized_rank_zero_phase(
        rank=rank,
        control_group=control_group,
        phase="preflight rank-zero CPU gates",
        exception_type=SO2NBPreflightError,
    ):
        if rank == 0:
            sample_geometry = bundle.training_batches[0].relative_geometry[:257]
            rank_zero_cpu_checks = {
                "neutral_geometry_initialization": _neutral_geometry_receipt(
                    model, sample_geometry
                ),
                "raw_target_forward_isolation": _raw_target_isolation_receipt(
                    model, bundle.training_batches[0]
                ),
                "synthetic_nb2_loss_decrease": _synthetic_loss_decrease_receipt(),
                "real_subset_nb2_loss_decrease": _real_subset_loss_decrease_receipt(
                    bundle.training_batches[0]
                ),
            }
    payloads: list[Any] = [rank_zero_cpu_checks]
    torch.distributed.broadcast_object_list(payloads, src=0, group=control_group)
    cpu_checks = payloads[0]
    if not isinstance(cpu_checks, Mapping):
        raise SO2NBPreflightError("Rank-zero CPU preflight checks were not broadcast.")

    phase("initializing_ddp_model")
    device = torch.device(f"cuda:{local_rank}")
    training_model, optimizer, scheduler, scaler = _initialize_training(
        model, config, device
    )
    trainer = _section(config, "trainer")
    gpu_inventory = _gpu_inventory_receipt(device, rank=rank)
    initial_sync = _synchronized_state_receipt(model, phase="before_optimizer_update")
    torch.cuda.reset_peak_memory_stats(device)
    phase("amp_fp32_equivalence")
    amp_equivalence: dict[str, Any] | None = None
    with _synchronized_rank_zero_phase(
        rank=rank,
        control_group=control_group,
        phase="preflight AMP/FP32 equivalence gate",
        exception_type=SO2NBPreflightError,
    ):
        if rank == 0:
            amp_equivalence = _amp_fp32_equivalence_receipt(
                model,
                bundle.training_batches[0],
                device=device,
                amp_dtype=str(trainer["amp_dtype"]),
                staged_geometry_dtype=str(trainer["staged_relative_geometry_dtype"]),
            )
    amp_payload: list[Any] = [amp_equivalence]
    torch.distributed.broadcast_object_list(
        amp_payload, src=0, group=control_group
    )
    amp_equivalence = amp_payload[0]
    if not isinstance(amp_equivalence, Mapping):
        raise SO2NBPreflightError("AMP/FP32 evidence was not broadcast.")
    phase("complete_pair_update_with_all_rank_gradient_observation")
    # The production path observes gradients on every rank so CUDA work stays
    # aligned before subsequent NCCL collectives.  Mirror that behavior here;
    # rank zero alone serializes the scalar evidence.
    block_tracker = _CapturingBlockTracker()
    theta_tracker = _CapturingThetaTracker()
    global_tracker = FullGradientDirectionTracker(
        expected_optimizer_updates_per_epoch=1
    )
    theta_view = _RawThetaParameterView(model.raw_theta)
    training_records, global_gradient_norm = _paired_update(
        model=model,
        training_model=training_model,
        optimizer=optimizer,
        scaler=scaler,
        pair=PREFLIGHT_TRAINING_PAIR,
        update_index=0,
        global_epoch=0,
        rank=rank,
        device=device,
        batches=bundle.batches_by_alias,
        trainer=trainer,
        global_tracker=global_tracker,
        block_tracker=block_tracker,
        theta_tracker=theta_tracker,
        theta_view=theta_view,
    )
    post_update_sync = _synchronized_state_receipt(model, phase="after_optimizer_update")
    if initial_sync["per_rank_sha256"][0] == post_update_sync["per_rank_sha256"][0]:
        raise SO2NBPreflightError("Optimizer update did not change model state.")

    global_summary = asdict(global_tracker.complete_epoch(0))
    block_summaries = [asdict(value) for value in block_tracker.complete_epoch(0)]
    theta_summary = asdict(theta_tracker.complete_epoch(0))
    phase("gradient_observation_complete")
    gradient_evidence: dict[str, Any] | None = None
    with _synchronized_rank_zero_phase(
        rank=rank,
        control_group=control_group,
        phase="preflight gradient evidence gate",
        exception_type=SO2NBPreflightError,
    ):
        if rank == 0:
            if (
                len(block_tracker.snapshots) != EXPECTED_BLOCKS
                or theta_tracker.snapshot is None
                or any(
                    value["gradient_norm_mean_before_clip"] <= 0.0
                    for value in block_summaries
                )
                or theta_summary["gradient_norm_mean_before_clip"] <= 0.0
            ):
                raise SO2NBPreflightError(
                    "Preclip block/dispersion gradient gate failed."
                )
            gradient_evidence = {
                "passed": True,
                "gradient_observation_execution": "identical_on_all_four_ranks",
                "scalar_evidence_writer_rank": 0,
                "global_gradient_norm_before_clip": global_gradient_norm,
                "global_direction_summary": global_summary,
                "block_snapshots": block_tracker.snapshots,
                "block_direction_summaries": block_summaries,
                "raw_theta_snapshot": theta_tracker.snapshot,
                "raw_theta_direction_summary": theta_summary,
            }
    gradient_payload: list[Any] = [gradient_evidence]
    torch.distributed.broadcast_object_list(
        gradient_payload, src=0, group=control_group
    )
    gradient_evidence = gradient_payload[0]
    if not isinstance(gradient_evidence, Mapping):
        raise SO2NBPreflightError("Gradient evidence was not broadcast.")

    expected_training_keys = {
        (alias, view)
        for alias in PREFLIGHT_TRAINING_PAIR
        for view in range(10)
    }
    observed_training_keys = {
        (str(record["alias"]), int(record["view_index"]))
        for record in training_records
    }
    training_passed = bool(
        len(training_records) == 20
        and observed_training_keys == expected_training_keys
        and all(
            math.isfinite(float(record["negative_binomial_nll"]))
            for record in training_records
        )
    )
    if not training_passed:
        raise SO2NBPreflightError("Complete paired-core training update is incomplete.")

    phase("fixed_validation_forward")
    mask_regeneration = _fixed_mask_regeneration_receipt(bundle, rank=rank)
    validation, validation_receipts = _evaluate_validation(
        bundle=bundle,
        training_model=training_model,
        rank=rank,
        device=device,
        trainer=trainer,
    )
    if (
        not math.isfinite(validation.primary_equal_core_nll)
        or len(validation_receipts) != 20
    ):
        raise SO2NBPreflightError("Fixed validation forward is incomplete or non-finite.")
    scheduler.step(validation.primary_equal_core_nll)
    rng_states = _all_rank_rng_states()
    peak_vram = _peak_vram_receipt(device, rank=rank)

    phase("all_rank_checkpoint_state_snapshot")
    checkpoint_early = update_early_stopping(
        EarlyStoppingState(),
        validation.primary_equal_core_nll,
        completed_epoch=1,
    )
    checkpoint_state = _checkpoint_state_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        completed_epoch=1,
        early_state=checkpoint_early,
        bundle=bundle,
        rng_states=rng_states,
        validation_mask_receipts=validation_receipts,
        configuration_sha256=canonical_sha256(config),
    )
    torch.distributed.barrier(group=control_group)
    phase("checkpoint_reload_gate")

    def reload_checkpoint() -> dict[str, Any]:
        output.parent.mkdir(parents=True, exist_ok=True)
        return _checkpoint_reload_receipt(
            output_parent=output.parent,
            config=config,
            bundle=bundle,
            checkpoint_state=checkpoint_state,
        )

    checkpoint = _broadcast_rank_zero_result(
        rank=rank,
        control_group=control_group,
        phase="preflight checkpoint reload gate",
        operation=reload_checkpoint,
        exception_type=SO2NBPreflightError,
    )
    del checkpoint_state
    if not isinstance(checkpoint, Mapping) or checkpoint.get("passed") is not True:
        raise SO2NBPreflightError("Checkpoint evidence was not broadcast.")

    gates: dict[str, Mapping[str, Any]] = {
        "configuration_and_overlay": {
            "passed": True,
            "configuration_sha256": canonical_sha256(config),
            "overlay_manifest_sha256": bundle.manifest_sha256,
        },
        "model_topology": topology,
        "neutral_geometry_initialization": cpu_checks[
            "neutral_geometry_initialization"
        ],
        "raw_target_forward_isolation": cpu_checks["raw_target_forward_isolation"],
        "full_constant_fp32_nb2": {
            "passed": True,
            "likelihood": "negative_binomial_nb2_full_constant",
            "compute_dtype": "float32_outside_autocast",
            "finite_training_view_losses": len(training_records),
        },
        "amp_fp32_equivalence": amp_equivalence,
        "synthetic_nb2_loss_decrease": cpu_checks[
            "synthetic_nb2_loss_decrease"
        ],
        "real_subset_nb2_loss_decrease": cpu_checks[
            "real_subset_nb2_loss_decrease"
        ],
        "ddp_parameter_sync": {
            "passed": True,
            "before_update": initial_sync,
            "after_update": post_update_sync,
        },
        "complete_pair_training_update": {
            "passed": True,
            "aliases": list(PREFLIGHT_TRAINING_PAIR),
            "complete_core_cells": {
                alias: bundle.batches_by_alias[alias].n_nodes
                for alias in PREFLIGHT_TRAINING_PAIR
            },
            "complete_core_edges": {
                alias: bundle.batches_by_alias[alias].n_edges
                for alias in PREFLIGHT_TRAINING_PAIR
            },
            "views_per_core": 10,
            "views_per_rank": 5,
            "view_loss_count": len(training_records),
            "optimizer_updates": 1,
        },
        "preclip_gradient_flow": gradient_evidence,
        "fixed_validation_mask_regeneration": mask_regeneration,
        "fixed_validation_forward": {
            "passed": True,
            "aliases": list(VALIDATION_ALIASES),
            "fixed_views_per_core": 10,
            "mask_receipt_count": len(validation_receipts),
            "mask_receipts_sha256": canonical_sha256(validation_receipts),
            "equal_core_masked_nb2_nll": validation.primary_equal_core_nll,
            "pooled_masked_nb2_nll": validation.pooled_nll,
        },
        "checkpoint_best_latest_reload": checkpoint,
        "gpu_inventory": gpu_inventory,
        "peak_vram": peak_vram,
        "no_test_split_or_artifacts": {
            "passed": True,
            "test_aliases": [],
            "test_partition_present": False,
            "test_artifacts_written": False,
        },
    }
    if set(gates) != REQUIRED_GATE_NAMES or not all(
        gate.get("passed") is True for gate in gates.values()
    ):
        raise SO2NBPreflightError("One or more required gates did not pass.")

    phase("publishing_receipt")

    def publish_receipt() -> dict[str, Any]:
        contract_path = paths.project_root / FROZEN_CONTRACT_RELATIVE_PATH
        receipt: dict[str, Any] = {
            "schema": PREFLIGHT_SCHEMA,
            "status": "passed",
            "passed": True,
            "all_required_gates_passed": True,
            "completed_experiment": False,
            "diagnostic_only": True,
            "created_at": _utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "protocol": PROTOCOL,
            "world_size": WORLD_SIZE,
            "distributed_backend": "nccl",
            "control_plane_backend": "gloo",
            "visible_devices": VISIBLE_DEVICES,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "receiver_chunk_size": int(model.receiver_chunk_size),
            "train_pair": list(PREFLIGHT_TRAINING_PAIR),
            "train_update_completed": True,
            "validation_completed": True,
            "checkpoint_reload_verified": True,
            "configuration_sha256": canonical_sha256(config),
            "resolved_config_sha256": canonical_sha256(config),
            "configuration_file_sha256": sha256_file(config_path),
            "configuration_source_file_sha256": _configuration_file_hashes(
                config_path, paths.config_root
            ),
            "frozen_task_contract_sha256": sha256_file(contract_path),
            "overlay_manifest_sha256": bundle.manifest_sha256,
            "overlay_manifest_content_sha256": overlay_receipt[
                "manifest_content_sha256"
            ],
            "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
            "split_fingerprint": bundle.split_fingerprint,
            "overlay": overlay_receipt,
            "source_artifacts": overlay_receipt["source_artifacts"],
            "code_file_sha256": _code_file_hashes(),
            "code_hash_scope": CODE_HASH_SCOPE,
            "gpu_identities": gpu_inventory["devices"],
            "peak_vram_gib_all_ranks": peak_vram["peak_vram_gib_all_ranks"],
            "peak_reserved_vram_gib_all_ranks": peak_vram[
                "peak_reserved_vram_gib_all_ranks"
            ],
            "minimum_vram_headroom_gib_each_rank": peak_vram[
                "minimum_vram_headroom_gib_each_rank"
            ],
            "minimum_reserved_vram_headroom_gib_each_rank": peak_vram[
                "minimum_reserved_vram_headroom_gib_each_rank"
            ],
            "validation_equal_core_masked_nb_nll": (
                validation.primary_equal_core_nll
            ),
            "test_partition_present": False,
            "test_artifacts_present": False,
            "temporary_checkpoints_cleaned": True,
            "gates": gates,
        }
        receipt["receipt_content_sha256"] = canonical_sha256(receipt)
        _validate_receipt(receipt)
        _atomic_json(output, receipt)
        return receipt

    published_receipt = _broadcast_rank_zero_result(
        rank=rank,
        control_group=control_group,
        phase="preflight receipt publication",
        operation=publish_receipt,
        exception_type=SO2NBPreflightError,
    )
    if not isinstance(published_receipt, Mapping):
        raise SO2NBPreflightError("Published receipt result is not a mapping.")
    return dict(published_receipt) if rank == 0 else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen SO2 geometry-modulated NB2 DDP4 preflight."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rank, local_rank, _ = _distributed_identity()
    paths = current_paths(anchor=PROJECT_ROOT)
    config_path, overlay, output, config = _resolve_required_paths(args, paths=paths)
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
        receipt = run_preflight(
            config_path=config_path,
            overlay=overlay,
            output=output,
            config=config,
            paths=paths,
            rank=rank,
            local_rank=local_rank,
            control_group=control_group,
        )
        if rank == 0:
            print(json.dumps(receipt, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
