#!/usr/bin/env python3
"""Four-rank launch gate for target-isolated SO2 binary_expression BCE training.

Run this diagnostic through ``torchrun --nproc-per-node=4``.  It validates the
frozen binary_expression configuration, derives the detection-head initialization from
training counts only, exercises the exact distributed objective on the largest
training core and the fixed validation task on the largest validation core,
and atomically writes one checksum-bound receipt.  It never launches training
and retains no diagnostic checkpoint.

The module deliberately does not import the production binary_expression runner.  The
runner is checksum-bound as a required source file, so a missing or changing
runner fails the executed preflight without preventing static inspection or
``--help`` while the campaign is being assembled.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, ContextManager, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.diagnostics import (  # noqa: E402
    preflight_so2_target_isolated_nb_ddp as base_preflight,
)
from spatial_benchmark.configuration import compose_config  # noqa: E402
from spatial_benchmark.binary_expression import (  # noqa: E402
    BinaryExpressionModelOutput,
    TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    masked_binary_expression_bce_sums,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.so2_nb_data import (  # noqa: E402
    SO2NBDataBundle,
    SO2NBCoreBatch,
    SO2_NB_TRAINING_ALIASES,
    SO2_NB_VALIDATION_ALIASES,
    load_so2_nb_data,
)
from spatial_benchmark.so2_nb_training import (  # noqa: E402
    AtomicBestLatestCheckpointStore,
    EarlyStoppingState,
    sha256_file,
)
from spatial_benchmark.target_cell_masking import (  # noqa: E402
    make_contiguous_target_partition,
    make_contiguous_target_partitions,
    make_training_target_cell_masks,
    make_validation_target_cell_masks,
)


CAMPAIGN_ID = (
    "cmp_20260930_so2_target_isolated_geometry_modulated_binary_expression_"
    "bce_once_per_cell_seed0"
)
PROTOCOL = (
    "donor_grouped_so2_target_isolated_binary_expression_bce_once_per_cell_"
    "earlystop_v1"
)
PREFLIGHT_SCHEMA = (
    "so2_target_isolated_binary_expression_bce_once_per_cell_ddp4_preflight_v1"
)
CHECKPOINT_SCHEMA = (
    "so2_target_isolated_binary_expression_bce_once_per_cell_checkpoint_v1"
)
PRIMARY_METRIC = "val/target_only_masked_binary_cross_entropy"
WORLD_SIZE = 4
VISIBLE_DEVICES = "0,1,2,3"
EXPECTED_PARAMETER_COUNT = 5_134_088
EXPECTED_DETECTION_PARAMETER_COUNT = 1_025_000
EXPECTED_BLOCK_COUNT = 4
EXPECTED_BLOCK_PARAMETER_COUNT = 831_880
EXPECTED_GENES = 1_000
EXPECTED_COVARIATES = 22
EXPECTED_TRAINING_CELLS = 208_696
EXPECTED_VALIDATION_CELLS = 37_367
REPRESENTATIVE_TRAINING_ALIAS = "SO2-C23"
REPRESENTATIVE_VALIDATION_ALIAS = "SO2-C27"
MAXIMUM_PEAK_VRAM_GIB = 22.0
MINIMUM_VRAM_HEADROOM_GIB = 2.0
MINIMUM_FREE_DISK_GIB = 20.0
DETECTION_EPSILON = 1e-4
FROZEN_TASK_CONTRACT_SHA256 = (
    "4d1e030bb597c4a7dce64da8adf3c57743bf8b4a0cbd1e646d2986cca114335a"
)
AMP_OUTPUT_MAX_ABS_TOLERANCE = 8e-2
AMP_OUTPUT_MEAN_ABS_TOLERANCE = 1e-2
AMP_LOSS_ABS_TOLERANCE = 3e-2
AMP_LOSS_RELATIVE_TOLERANCE = 3e-3
SOURCE_EXPERIMENT_CONFIG = Path(
    "configs/experiment/"
    "so2_target_isolated_geometry_modulated_binary_expression_bce_once_per_"
    "cell_seed0.yaml"
)
FROZEN_TASK_CONTRACT = Path(
    "experiments/campaigns/"
    "cmp_20260930_so2_target_isolated_geometry_modulated_binary_expression_"
    "bce_once_per_cell_seed0/frozen_task_contract.yaml"
)
CODE_HASH_SCOPE = (
    "target_isolated_binary_expression_model_masking_data_training_preflight_"
    "runner_and_configuration_launch_contract"
)
CODE_RELATIVE_PATHS = (
    Path("scripts/diagnostics/preflight_so2_target_isolated_binary_expression_bce_ddp.py"),
    Path("scripts/diagnostics/preflight_so2_target_isolated_nb_ddp.py"),
    Path("scripts/train/run_so2_target_isolated_binary_expression_bce.py"),
    Path("scripts/train/run_so2_target_isolated_nb.py"),
    Path("scripts/train/run_so2_geometry_modulated_nb.py"),
    Path("src/spatial_benchmark/binary_expression.py"),
    Path("src/spatial_benchmark/target_isolated_negative_binomial.py"),
    Path("src/spatial_benchmark/target_cell_masking.py"),
    Path("src/spatial_benchmark/target_cell_training.py"),
    Path("src/spatial_benchmark/geometry_modulated_relative_qkv_graph_transformer.py"),
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

PREFLIGHT_REQUIRED_GATES = frozenset(
    {
        "configuration_and_contract",
        "model_topology",
        "binary_head_and_probability",
        "synthetic_target_coverage",
        "target_mask_contract",
        "neighbor_context_unmasked",
        "incoming_nonself_routing",
        "masked_target_value_invariance",
        "raw_target_forward_isolation",
        "no_return_leakage",
        "binary_bce_reference",
        "unequal_count_ddp_weighting",
        "fixed_validation_mask_replay",
        "fixed_validation_metric_replay",
        "histogram_metric_replay",
        "amp_fp32_equivalence",
        "real_train_shard_forward_backward",
        "preclip_gradient_flow",
        "bounded_binary_loss_decrease",
        "checkpoint_roundtrip",
        "ddp_parameter_sync",
        "gpu_inventory",
        "peak_vram",
        "disk_space",
        "no_test_split_or_artifacts",
    }
)


class BinaryExpressionBCEPreflightError(RuntimeError):
    """Raised when a binary_expression launch-safety gate fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise BinaryExpressionBCEPreflightError(
            f"Resolved configuration requires a {name!r} mapping."
        )
    return value


def _runtime_path(raw: object, paths: ProjectPaths) -> Path:
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if not candidate.parts or ".." in candidate.parts:
        raise BinaryExpressionBCEPreflightError(
            f"Unsafe configured runtime path: {raw!r}."
        )
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


def _argument_path(raw: Path | None, *, default: Path) -> Path:
    candidate = default if raw is None else raw.expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve(strict=False)


def _validate_binary_config(config: Mapping[str, Any]) -> None:
    campaign = _section(config, "campaign")
    evaluation = _section(config, "evaluation")
    model = _section(config, "model")
    trainer = _section(config, "trainer")
    launcher = _section(config, "launcher")
    dataset = _section(config, "dataset")
    masking = _section(config, "masking")
    expected_model = {
        "name": "target-isolated-geometry-modulated-relative-qkv-gat-binary-expression-bce",
        "family": (
            "target_isolated_geometry_modulated_relative_qkv_binary_expression_graph_transformer"
        ),
        "output_distribution": "bernoulli_binary_expression",
        "decoder_trunk": "binary_expression_only",
        "decoder_trunk_dim": 1024,
        "detection_logit_projection_output_dim": EXPECTED_GENES,
        "detection_probability_activation": "sigmoid",
        "detection_projection_weight_initialization": "zeros",
        "detection_projection_bias_initialization": (
            "logit_jeffreys_smoothed_training_nonzero_prevalence"
        ),
        "detection_prevalence_smoothing": "jeffreys_beta_half_half",
        "detection_prevalence_denominator": "training_cells_plus_one",
        "detection_initialization_fit_scope": "twelve_training_cores_only",
        "hidden_dim": 256,
        "graph_layers": 4,
        "unique_graph_blocks": 4,
        "attention_heads": 8,
        "attention_head_dim": 32,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "expected_trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
    }
    expected_trainer = {
        "objective": "pooled_unweighted_masked_binary_cross_entropy_with_logits",
        "objective_numerator": (
            "sum_binary_cross_entropy_with_logits_over_all_masked_target_entries"
        ),
        "objective_denominator": (
            "globally_reduced_masked_target_entry_count_per_update"
        ),
        "local_ddp_loss_scale": (
            "world_size_times_local_bce_sum_divided_by_global_masked_entry_count"
        ),
        "likelihood_compute_dtype": "float32",
        "likelihood_outside_autocast": True,
        "amp": True,
        "distributed": True,
        "distributed_backend": "nccl",
        "distributed_world_size": WORLD_SIZE,
        "target_shards_per_core": WORLD_SIZE,
        "stage_complete_core_graph_on_device": False,
        "staged_relative_geometry_dtype": "float32",
    }
    for field, expected in expected_model.items():
        if model.get(field) != expected:
            raise BinaryExpressionBCEPreflightError(
                f"model.{field} must be {expected!r}; got {model.get(field)!r}."
            )
    for field, expected in expected_trainer.items():
        if trainer.get(field) != expected:
            raise BinaryExpressionBCEPreflightError(
                f"trainer.{field} must be {expected!r}; got {trainer.get(field)!r}."
            )
    if campaign.get("campaign_id") != CAMPAIGN_ID or config.get("seed") != 0:
        raise BinaryExpressionBCEPreflightError(
            "Campaign identity or seed drifted."
        )
    if (
        evaluation.get("protocol") != PROTOCOL
        or evaluation.get("primary_metric") != PRIMARY_METRIC
        or evaluation.get("primary_aggregation")
        != "sum_masked_entry_binary_cross_entropy_divided_by_total_masked_entries"
        or evaluation.get("splits") != ["validation"]
        or evaluation.get("test_split_present") is not False
    ):
        raise BinaryExpressionBCEPreflightError(
            "Evaluation protocol or primary binary_expression metric drifted."
        )
    if (
        dataset.get("training_cells") != EXPECTED_TRAINING_CELLS
        or dataset.get("validation_cells") != EXPECTED_VALIDATION_CELLS
        or dataset.get("test_cells") != 0
        or dataset.get("training_core_aliases") != list(SO2_NB_TRAINING_ALIASES)
        or dataset.get("validation_core_aliases") != list(SO2_NB_VALIDATION_ALIASES)
        or dataset.get("test_core_aliases") != []
    ):
        raise BinaryExpressionBCEPreflightError("Dataset split contract drifted.")
    if (
        masking.get("count_min") != 1
        or masking.get("count_max") != EXPECTED_GENES
        or masking.get("target_cell_only") is not True
        or masking.get("neighbor_cells_artificially_masked") is not False
        or masking.get("neighbor_context_rows_fully_observed") is not True
        or masking.get("mask_base_seed") != 2026092601
        or masking.get("training_mask_seed_namespace")
        != "bagm.so2.target_isolated_nb.training_masks.v1"
        or not isinstance(masking.get("validation_masks"), Mapping)
        or masking["validation_masks"].get("seed_namespace")
        != "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"
    ):
        raise BinaryExpressionBCEPreflightError("Target-only masking drifted.")
    if (
        launcher.get("process_count") != WORLD_SIZE
        or launcher.get("require_exact_visible_devices") != VISIBLE_DEVICES
        or launcher.get("disk_safety_min_free_gb") != 20
        or launcher.get("hardware_preflight_receipt")
        != "state/preflight/so2_target_isolated_binary_expression_bce_once_per_cell_ddp4.json"
    ):
        raise BinaryExpressionBCEPreflightError("Launcher contract drifted.")


def _resolve_required_paths(
    args: argparse.Namespace,
) -> tuple[ProjectPaths, Path, dict[str, Any], Path, Path, Path, Path, Path, str]:
    paths = current_paths(anchor=PROJECT_ROOT)
    paths.validate()
    if paths.project_root.resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise BinaryExpressionBCEPreflightError(
            "BAGM_ROOT must identify the source tree containing this preflight."
        )
    expected_config = (PROJECT_ROOT / SOURCE_EXPERIMENT_CONFIG).resolve(strict=True)
    config_path = _argument_path(args.config, default=expected_config)
    if config_path != expected_config:
        raise BinaryExpressionBCEPreflightError(
            "--config must be the frozen binary_expression experiment source."
        )
    config = compose_config(config_path, config_root=paths.config_root, validate=True)
    _validate_binary_config(config)
    dataset = _section(config, "dataset")
    launcher = _section(config, "launcher")
    expected_overlay = _runtime_path(dataset["prepared_artifact_reference"], paths)
    expected_cohort = _runtime_path(dataset["source_cohort_artifact"], paths)
    expected_graph = _runtime_path(dataset["source_graph_artifact"], paths)
    expected_output = _runtime_path(launcher["hardware_preflight_receipt"], paths)
    resolved = (
        _argument_path(args.overlay_dir, default=expected_overlay),
        _argument_path(args.cohort_dir, default=expected_cohort),
        _argument_path(args.graph_dir, default=expected_graph),
        _argument_path(args.output, default=expected_output),
    )
    if resolved != (expected_overlay, expected_cohort, expected_graph, expected_output):
        raise BinaryExpressionBCEPreflightError(
            "Data/output overrides must resolve to paths frozen in config."
        )
    overlay, cohort, graph, output = resolved
    for value, label in ((overlay, "overlay"), (cohort, "cohort"), (graph, "graph")):
        if not value.is_dir():
            raise BinaryExpressionBCEPreflightError(
                f"Configured {label} directory is absent: {value}."
            )
    contract = (PROJECT_ROOT / FROZEN_TASK_CONTRACT).resolve(strict=True)
    contract_sha = sha256_file(contract)
    metadata = _section(config, "metadata")
    if (
        contract_sha != FROZEN_TASK_CONTRACT_SHA256
        or metadata.get("frozen_task_contract_sha256") != contract_sha
    ):
        raise BinaryExpressionBCEPreflightError(
            "Frozen binary_expression task-contract checksum drifted."
        )
    if output.is_symlink():
        raise BinaryExpressionBCEPreflightError(
            "Preflight receipt path may not be a symbolic link."
        )
    return (
        paths,
        config_path,
        config,
        overlay,
        cohort,
        graph,
        output,
        contract,
        contract_sha,
    )


def _training_nonzero_initialization(bundle: SO2NBDataBundle) -> dict[str, Any]:
    positive = torch.zeros(EXPECTED_GENES, dtype=torch.int64)
    total_cells = 0
    for batch in bundle.training_batches:
        if batch.alias not in SO2_NB_TRAINING_ALIASES:
            raise BinaryExpressionBCEPreflightError(
                "Detection initialization encountered a non-training core."
            )
        for start in range(0, batch.n_nodes, 2048):
            stop = min(start + 2048, batch.n_nodes)
            positive += (batch.raw_count_target[start:stop] > 0).sum(
                dim=0, dtype=torch.int64
            )
        total_cells += batch.n_nodes
    if total_cells != EXPECTED_TRAINING_CELLS:
        raise BinaryExpressionBCEPreflightError(
            "Detection initialization training support drifted."
        )
    probability = (
        positive.to(torch.float64) + 0.5
    ) / float(total_cells + 1)
    probability = probability.clamp(DETECTION_EPSILON, 1.0 - DETECTION_EPSILON)
    probability = probability.to(torch.float32)
    if (
        probability.shape != (EXPECTED_GENES,)
        or not bool(torch.isfinite(probability).all())
        or not bool(((probability > 0.0) & (probability < 1.0)).all())
    ):
        raise BinaryExpressionBCEPreflightError(
            "Detection initialization probabilities are invalid."
        )
    return {
        "probability": probability.tolist(),
        "passed": True,
        "formula": "(training_positive_count + 0.5) / (training_cells + 1)",
        "smoothing": "jeffreys_beta_half_half",
        "fit_aliases": list(SO2_NB_TRAINING_ALIASES),
        "validation_aliases_consumed": [],
        "training_cells": total_cells,
        "positive_count_sha256": base_preflight._tensor_sha256(positive),
        "probability_sha256": base_preflight._tensor_sha256(probability),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
        "clamp": [DETECTION_EPSILON, 1.0 - DETECTION_EPSILON],
    }


def _build_model(
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
    detection_probability: Tensor,
) -> TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer:
    model_config = _section(config, "model")
    reference = bundle.training_batches[0]
    torch.manual_seed(int(config["seed"]))
    return TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer(
        num_genes=reference.n_genes,
        node_covariate_dim=reference.node_covariates.shape[1],
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
        detection_initial_nonzero_probability=detection_probability,
        detection_probability_epsilon=float(
            model_config.get("detection_probability_epsilon", DETECTION_EPSILON)
        ),
    )


def _model_topology_receipt(
    model: TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    expected_probability: Tensor,
) -> dict[str, Any]:
    count = sum(parameter.numel() for parameter in model.parameters())
    blocks = model.blocks
    block_counts = [sum(parameter.numel() for parameter in block.parameters()) for block in blocks]
    parameter_sets = [{id(parameter) for parameter in block.parameters()} for block in blocks]
    disjoint = all(
        left.isdisjoint(right)
        for index, left in enumerate(parameter_sets)
        for right in parameter_sets[index + 1 :]
    )
    detection = getattr(model, "detection_head", None)
    if not isinstance(detection, nn.Linear):
        raise BinaryExpressionBCEPreflightError("Detection projection is absent.")
    detection_count = sum(parameter.numel() for parameter in detection.parameters())
    expected_bias = torch.logit(expected_probability.cpu())
    weight_zero = bool((detection.weight.detach().cpu() == 0.0).all())
    bias_exact = torch.equal(detection.bias.detach().cpu(), expected_bias)
    buffer_exact = torch.equal(
        model.detection_initial_nonzero_probability.detach().cpu(),
        expected_probability.cpu(),
    )
    forbidden_state_keys = sorted(
        key
        for key in model.state_dict()
        if "raw_theta" in key
        or "count_head" in key
        or "count_mean" in key
        or "detection_linear_out" in key
    )
    named_parameters = dict(model.named_parameters())
    passed = bool(
        count == EXPECTED_PARAMETER_COUNT
        and len(blocks) == EXPECTED_BLOCK_COUNT
        and len({id(block) for block in blocks}) == EXPECTED_BLOCK_COUNT
        and block_counts == [EXPECTED_BLOCK_PARAMETER_COUNT] * EXPECTED_BLOCK_COUNT
        and disjoint
        and detection_count == EXPECTED_DETECTION_PARAMETER_COUNT
        and "raw_theta" not in named_parameters
        and not forbidden_state_keys
        and detection is model.decoder.linear_out
        and weight_zero
        and bias_exact
        and buffer_exact
    )
    if not passed:
        raise BinaryExpressionBCEPreflightError(
            "BinaryExpression model topology or initialization drifted."
        )
    return {
        "passed": True,
        "class": type(model).__name__,
        "parameter_count": count,
        "detection_head_parameter_count": detection_count,
        "graph_block_count": len(blocks),
        "unique_graph_block_count": len({id(block) for block in blocks}),
        "per_block_parameter_count": block_counts,
        "raw_theta_present": False,
        "count_head_present": False,
        "forbidden_count_state_keys": forbidden_state_keys,
        "state_dict_key_sha256": canonical_sha256(list(model.state_dict().keys())),
        "detection_weight_zero_initialized": weight_zero,
        "detection_bias_matches_train_prevalence": bias_exact,
        "detection_probability_buffer_matches": buffer_exact,
        "target_input_is_separate": True,
        "static_clean_source_bank": True,
        "shared_encoder_module_count": 1,
    }


def _binary_bce_reference_receipt() -> dict[str, Any]:
    """Compare FP32 masked BCE with FP64 and ignore positive magnitudes."""

    logits = torch.tensor(
        [[-80.0, -8.0, -2.0, 0.0, 2.0, 8.0, 80.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[0, 1, 2, 0, 31, 128, 729]], dtype=torch.int32)
    mask = torch.ones_like(target, dtype=torch.bool)
    sums = masked_binary_expression_bce_sums(logits, target, mask)
    oracle = F.binary_cross_entropy_with_logits(
        logits.detach().double(), (target > 0).double(), reduction="sum"
    )
    error = abs(float(sums.detection_nll_sum.detach()) - float(oracle))
    tolerance = 2e-5

    sums.detection_nll_sum.backward()
    original_gradient = logits.grad.detach().clone()
    changed_target = torch.where(target > 0, target * 97 + 11, target)
    changed_logits = logits.detach().clone().requires_grad_(True)
    changed = masked_binary_expression_bce_sums(
        changed_logits, changed_target, mask
    )
    changed.detection_nll_sum.backward()
    magnitude_invariant = bool(
        torch.equal(sums.detection_nll_sum, changed.detection_nll_sum)
        and torch.equal(original_gradient, changed_logits.grad)
    )
    if (
        sums.masked_entries != target.numel()
        or sums.positive_entries != int((target > 0).sum())
        or sums.detection_nll_sum.dtype != torch.float32
        or not bool(torch.isfinite(sums.detection_nll_sum))
        or error > tolerance
        or not magnitude_invariant
    ):
        raise BinaryExpressionBCEPreflightError(
            "Binary BCE disagrees with its FP64 oracle or uses count magnitude."
        )
    return {
        "passed": True,
        "dtype": "torch.float32",
        "masked_entries": sums.masked_entries,
        "positive_entries": sums.positive_entries,
        "contains_zero_and_positive_targets": True,
        "contains_positive_count_729": True,
        "bce_sum": float(sums.detection_nll_sum),
        "maximum_absolute_reference_error": error,
        "absolute_tolerance": tolerance,
        "positive_count_magnitude_invariant_loss": True,
        "positive_count_magnitude_invariant_gradient": True,
        "normalization": "binary_cross_entropy_sum_divided_by_all_masked_entries",
        "class_reweighting": "none",
    }


class _SyntheticBinaryExpressionParameters(nn.Module):
    """Tiny Bernoulli model for the exact unequal-count DDP gate."""

    def __init__(self) -> None:
        super().__init__()
        self.detection_bias = nn.Parameter(torch.tensor(-0.35))
        self.detection_slope = nn.Parameter(torch.tensor(0.2))

    def forward(self, feature: Tensor) -> Tensor:
        return (self.detection_bias + self.detection_slope * feature).view(-1, 1)


def _synthetic_rank_values(
    rank: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    count = rank + 2
    feature = torch.linspace(-0.8, 1.1, count, device=device) + 0.07 * rank
    values = [0, 1, 0, 3, 2, 0]
    target = torch.tensor(values[:count], dtype=torch.int32, device=device).view(-1, 1)
    mask = torch.ones_like(target, dtype=torch.bool)
    return feature, target, mask


def _unequal_count_ddp_weighting_receipt(
    *, device: torch.device, rank: int
) -> dict[str, Any]:
    torch.manual_seed(8801)
    module = _SyntheticBinaryExpressionParameters().to(device)
    distributed = DistributedDataParallel(
        module,
        device_ids=None,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    feature, target, mask = _synthetic_rank_values(rank, device)
    logits = distributed(feature)
    sums = masked_binary_expression_bce_sums(logits, target, mask)
    local_count = torch.tensor(sums.masked_entries, dtype=torch.int64, device=device)
    global_count = local_count.clone()
    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
    scaled = sums.detection_nll_sum * (
        float(WORLD_SIZE) / global_count.to(torch.float32)
    )
    scaled.backward()
    ddp_gradient = torch.stack(
        [
            distributed.module.detection_bias.grad.detach(),
            distributed.module.detection_slope.grad.detach(),
        ]
    ).float()

    reference = _SyntheticBinaryExpressionParameters().to(device)
    reference.load_state_dict(distributed.module.state_dict())
    reference_sum = torch.zeros((), dtype=torch.float32, device=device)
    reference_count = 0
    local_counts: list[int] = []
    for reference_rank in range(WORLD_SIZE):
        ref_feature, ref_target, ref_mask = _synthetic_rank_values(
            reference_rank, device
        )
        ref_sums = masked_binary_expression_bce_sums(
            reference(ref_feature), ref_target, ref_mask
        )
        reference_sum = reference_sum + ref_sums.detection_nll_sum
        reference_count += ref_sums.masked_entries
        local_counts.append(ref_sums.masked_entries)
    (reference_sum / float(reference_count)).backward()
    reference_gradient = torch.stack(
        [reference.detection_bias.grad.detach(), reference.detection_slope.grad.detach()]
    ).float()
    difference = float(torch.max(torch.abs(ddp_gradient - reference_gradient)).cpu())
    tolerance = 2e-6
    gathered = base_preflight._all_gather_objects(
        {
            "rank": rank,
            "gradient": ddp_gradient.detach().cpu().tolist(),
            "local_masked_entries": sums.masked_entries,
        }
    )
    if (
        int(global_count) != reference_count
        or difference > tolerance
        or any(
            max(
                abs(float(left) - float(right))
                for left, right in zip(item["gradient"], reference_gradient.cpu())
            )
            > tolerance
            for item in gathered
        )
    ):
        raise BinaryExpressionBCEPreflightError(
            "Unequal-count DDP BCE weighting disagrees with the pooled reference."
        )
    return {
        "passed": True,
        "local_masked_entry_counts_by_rank": local_counts,
        "global_masked_entry_count": reference_count,
        "ddp_loss_scale_rule": (
            "world_size_times_local_bce_sum_divided_by_global_masked_entry_count"
        ),
        "gradient_order": ["detection_bias", "detection_slope"],
        "ddp_gradients_by_rank": sorted(gathered, key=lambda item: int(item["rank"])),
        "single_process_reference_gradient": reference_gradient.cpu().tolist(),
        "maximum_absolute_gradient_error": difference,
        "tolerance": tolerance,
    }


def _training_mask_for_rank(
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    *,
    rank: int,
) -> tuple[Any, Any]:
    partition = make_contiguous_target_partition(
        batch.n_nodes, world_size=WORLD_SIZE, rank=rank
    )
    masking = _section(config, "masking")
    realization = make_training_target_cell_masks(
        batch.n_nodes,
        batch.n_genes,
        base_seed=int(masking["mask_base_seed"]),
        namespace=str(masking["training_mask_seed_namespace"]),
        core_alias=batch.alias,
        global_epoch=1,
        target_indices=partition.target_indices,
    )
    return partition, realization


def _validation_mask_for_rank(
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    *,
    rank: int,
) -> tuple[Any, Any]:
    partition = make_contiguous_target_partition(
        batch.n_nodes, world_size=WORLD_SIZE, rank=rank
    )
    validation = _section(config, "masking")["validation_masks"]
    if not isinstance(validation, Mapping):
        raise BinaryExpressionBCEPreflightError(
            "validation_masks must be a mapping."
        )
    realization = make_validation_target_cell_masks(
        batch.n_nodes,
        batch.n_genes,
        namespace=str(validation["seed_namespace"]),
        core_alias=batch.alias,
        target_indices=partition.target_indices,
    )
    return partition, realization


def _make_target_inputs(
    clean_expression: Tensor,
    mask: Tensor,
    *,
    start: int,
    stop: int,
) -> Tensor:
    target = clean_expression[start:stop].clone()
    target.masked_fill_(mask, 0.0)
    if not bool((target.masked_select(mask) == 0.0).all()):
        raise BinaryExpressionBCEPreflightError("Target masking failed.")
    if not torch.equal(
        target.masked_select(~mask),
        clean_expression[start:stop].masked_select(~mask),
    ):
        raise BinaryExpressionBCEPreflightError(
            "Visible target input values drifted."
        )
    return target


def _synthetic_model_inputs(
    *, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(9917)
    clean = torch.randn(4, EXPECTED_GENES, generator=generator).to(device)
    covariates = torch.randn(
        4, EXPECTED_COVARIATES, generator=generator
    ).to(device)
    edge_index = torch.tensor(
        [
            [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ],
        dtype=torch.long,
    )
    geometry = torch.randn(edge_index.shape[1], 70, generator=generator)
    mask = torch.zeros(2, EXPECTED_GENES, dtype=torch.bool, device=device)
    mask[0, [0, 7, 13, 101]] = True
    mask[1, [1, 5, 17, 103]] = True
    target_input = _make_target_inputs(clean, mask, start=0, stop=2)
    raw_target = torch.zeros(2, EXPECTED_GENES, dtype=torch.int32, device=device)
    raw_target[0, [7, 101]] = torch.tensor([1, 8], dtype=torch.int32, device=device)
    raw_target[1, [1, 103]] = torch.tensor([2, 29], dtype=torch.int32, device=device)
    return clean, target_input, mask, covariates, edge_index, geometry, raw_target


def _output_fingerprint(output: BinaryExpressionModelOutput, row: int) -> Tensor:
    if output.selected_node_embedding is None:
        raise BinaryExpressionBCEPreflightError(
            "Selected target embedding is absent from model output."
        )
    return output.selected_node_embedding[row]


def _model_invariant_receipts(
    model: TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    *,
    device: torch.device,
) -> dict[str, Mapping[str, Any]]:
    clean, target_input, mask, covariates, edge_index, geometry, _ = (
        _synthetic_model_inputs(device=device)
    )
    previous_training = model.training
    model.eval()
    encoder_mask_counts: list[int] = []
    geometry_edge_counts: list[list[int]] = [[] for _ in model.blocks]

    def encoder_hook(
        _module: nn.Module,
        _args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        gene_mask = kwargs.get("gene_mask")
        if not isinstance(gene_mask, Tensor):
            raise BinaryExpressionBCEPreflightError(
                "Encoder mask hook saw no mask."
            )
        encoder_mask_counts.append(int(gene_mask.sum()))

    handles = [model.encoder.register_forward_pre_hook(encoder_hook, with_kwargs=True)]
    for index, block in enumerate(model.blocks):
        handles.append(
            block.geometry_encoder.register_forward_pre_hook(
                lambda _module, arguments, block_index=index: geometry_edge_counts[
                    block_index
                ].append(int(arguments[0].shape[0]))
            )
        )
    try:
        with torch.no_grad():
            baseline = model(
                input_expression=clean,
                target_input_expression=target_input,
                target_gene_mask=mask,
                edge_index=edge_index,
                relative_geometry=geometry,
                node_covariates=covariates,
                target_start=0,
                target_stop=2,
            )
    finally:
        for handle in handles:
            handle.remove()
    if not isinstance(baseline, BinaryExpressionModelOutput):
        raise BinaryExpressionBCEPreflightError("BinaryExpression output type drifted.")

    changed_hidden = clean.clone()
    changed_hidden[0, 0] += 10_000.0
    with torch.no_grad():
        hidden_output = model(
            input_expression=changed_hidden,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    own_equal = torch.equal(
        _output_fingerprint(baseline, 0), _output_fingerprint(hidden_output, 0)
    )
    cross_changed = not torch.equal(
        _output_fingerprint(baseline, 1), _output_fingerprint(hidden_output, 1)
    )

    changed_neighbor = clean.clone()
    changed_neighbor[2, 2] += 100.0
    with torch.no_grad():
        neighbor_output = model(
            input_expression=changed_neighbor,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    neighbor_changed = not torch.equal(
        _output_fingerprint(baseline, 0), _output_fingerprint(neighbor_output, 0)
    )

    changed_visible = clean.clone()
    changed_visible[0, 2] += 100.0
    changed_target_input = target_input.clone()
    changed_target_input[0, 2] = changed_visible[0, 2]
    with torch.no_grad():
        visible_output = model(
            input_expression=changed_visible,
            target_input_expression=changed_target_input,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    visible_changed = not torch.equal(
        _output_fingerprint(baseline, 0), _output_fingerprint(visible_output, 0)
    )

    self_loop_rejected = False
    loop_edges = edge_index.clone()
    loop_edges[0, 0] = loop_edges[1, 0]
    try:
        model(
            input_expression=clean,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=loop_edges,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    except ValueError as exc:
        self_loop_rejected = "self loop" in str(exc)

    masked_drift_rejected = False
    bad_masked = target_input.clone()
    bad_masked[0, 0] = 1.0
    try:
        model(
            input_expression=clean,
            target_input_expression=bad_masked,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    except ValueError as exc:
        masked_drift_rejected = "exactly zero" in str(exc)

    visible_drift_rejected = False
    bad_visible = target_input.clone()
    bad_visible[0, 2] += 1.0
    try:
        model(
            input_expression=clean,
            target_input_expression=bad_visible,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    except ValueError as exc:
        visible_drift_rejected = "visible" in str(exc)

    forward_parameters = set(inspect.signature(model.forward).parameters)
    forbidden_forward = {
        "raw_target",
        "raw_count_target",
        "target_counts",
        "library_total",
        "size_factor",
        "offset",
    }
    raw_target_absent = not bool(forward_parameters & forbidden_forward)
    invariant_values = (
        encoder_mask_counts == [0, int(mask.sum())],
        all(sum(counts) == 6 for counts in geometry_edge_counts),
        own_equal,
        cross_changed,
        neighbor_changed,
        visible_changed,
        self_loop_rejected,
        masked_drift_rejected,
        visible_drift_rejected,
        raw_target_absent,
        getattr(model, "source_bank_evolves", None) is False,
    )
    if not all(invariant_values):
        raise BinaryExpressionBCEPreflightError(
            "Target/context routing or leakage invariant failed."
        )
    model.train(previous_training)
    model.clear_edge_layout_cache()
    return {
        "target_mask_contract": {
            "passed": True,
            "every_target_mask_nonempty": True,
            "masked_target_input_exact_zero_enforced": masked_drift_rejected,
            "visible_target_input_exact_clean_match_enforced": visible_drift_rejected,
        },
        "neighbor_context_unmasked": {
            "passed": True,
            "clean_source_encoder_masked_entries": encoder_mask_counts[0],
            "simultaneous_target_clean_source_effect_observed": cross_changed,
        },
        "incoming_nonself_routing": {
            "passed": True,
            "synthetic_full_graph_edges": int(edge_index.shape[1]),
            "target_incoming_edges_per_block": [
                sum(counts) for counts in geometry_edge_counts
            ],
            "self_loop_rejected": self_loop_rejected,
        },
        "masked_target_value_invariance": {
            "passed": True,
            "own_target_embedding_bit_identical": own_equal,
            "clean_source_changed_other_target": cross_changed,
        },
        "raw_target_forward_isolation": {
            "passed": True,
            "forward_parameter_names": sorted(forward_parameters),
            "forbidden_raw_target_or_total_inputs_absent": raw_target_absent,
        },
        "no_return_leakage": {
            "passed": True,
            "cycle_edges_present": True,
            "source_bank_static_across_all_four_blocks": True,
            "withheld_value_cannot_return_to_target_embedding": own_equal,
            "visible_target_effect_observed": visible_changed,
            "clean_neighbor_effect_observed": neighbor_changed,
        },
    }


def _binary_head_and_probability_receipt(
    model: TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    *,
    expected_probability: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Prove the sole detection head and Bernoulli probability identities."""

    clean, target_input, mask, covariates, edge_index, geometry, _ = (
        _synthetic_model_inputs(device=device)
    )
    previous_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(
            input_expression=clean,
            target_input_expression=target_input,
            target_gene_mask=mask,
            edge_index=edge_index,
            relative_geometry=geometry,
            node_covariates=covariates,
            target_start=0,
            target_stop=2,
        )
    model.train(previous_training)
    model.clear_edge_layout_cache()
    if not isinstance(output, BinaryExpressionModelOutput):
        raise BinaryExpressionBCEPreflightError(
            "Detection-head gate received the wrong output type."
        )
    expected = expected_probability.to(device=device).view(1, -1).expand(2, -1)
    initial_probability_error = float(
        (output.nonzero_probability - expected).abs().max().cpu()
    )
    identities = bool(
        torch.equal(output.predicted_zero_probability, 1.0 - output.nonzero_probability)
        and torch.equal(output.prediction, output.nonzero_probability)
    )
    finite = all(
        value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        for value in (
            output.nonzero_logits,
            output.nonzero_probability,
            output.predicted_zero_probability,
        )
    )
    detection = model.detection_head
    named_parameters = dict(model.named_parameters())
    passed = bool(
        finite
        and identities
        and initial_probability_error <= 2e-7
        and bool((output.nonzero_probability > 0.0).all())
        and bool((output.nonzero_probability < 1.0).all())
        and isinstance(detection, nn.Linear)
        and detection.out_features == EXPECTED_GENES
        and detection is model.decoder.linear_out
        and "raw_theta" not in named_parameters
    )
    if not passed:
        raise BinaryExpressionBCEPreflightError(
            "Binary head or probability identity gate failed."
        )
    return {
        "passed": True,
        "detection_head": "decoder.linear_out",
        "head_output_dimension": detection.out_features,
        "count_head_absent": True,
        "raw_theta_absent": True,
        "probability_activation": "sigmoid",
        "initial_probability_maximum_absolute_error": initial_probability_error,
        "probability_tolerance": 2e-7,
        "predicted_zero_probability_identity": True,
        "prediction_is_nonzero_probability": True,
        "all_public_outputs_finite_fp32": True,
    }


def _autocast_context(
    *, enabled: bool, device: torch.device, dtype_name: str
) -> ContextManager[object]:
    return base_preflight._autocast_context(
        enabled=enabled,
        device=device,
        dtype_name=dtype_name,
    )


def _all_gather_objects(value: Any) -> list[Any]:
    return base_preflight._all_gather_objects(value)


def _gradient_state(module: nn.Module) -> dict[str, Any]:
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    present = [parameter.grad for parameter in parameters if parameter.grad is not None]
    complete = len(present) == len(parameters) and bool(parameters)
    finite = complete and all(bool(torch.isfinite(value).all()) for value in present)
    squared = sum(float((value.detach().float() ** 2).sum()) for value in present)
    return {
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "gradient_tensor_count": len(present),
        "all_parameter_gradients_present": complete,
        "all_finite": finite,
        "l2_norm": math.sqrt(squared),
        "nonzero": finite and squared > 0.0,
    }


def _real_training_shard_update(
    *,
    model: TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    device: torch.device,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exercise exact pooled BCE and the zero-head two-step gradient contract."""

    trainer = _section(config, "trainer")
    partition, realization = _training_mask_for_rank(batch, config, rank=rank)
    _, replay_realization = _training_mask_for_rank(batch, config, rank=rank)
    if (
        realization.to_receipt() != replay_realization.to_receipt()
        or not np.array_equal(realization.mask, replay_realization.mask)
    ):
        raise BinaryExpressionBCEPreflightError(
            "Training mask did not replay exactly for the matched seed tuple."
        )
    if partition.n_targets <= 0:
        raise BinaryExpressionBCEPreflightError("Representative training shard is empty.")
    if batch.edge_index.device.type != "cpu" or batch.relative_geometry.device.type != "cpu":
        raise BinaryExpressionBCEPreflightError(
            "Target-isolated graph and geometry must remain CPU-resident."
        )
    input_expression = batch.input_expression.to(device=device, dtype=torch.float32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(device=device)
    target_input = _make_target_inputs(
        input_expression, mask, start=partition.start, stop=partition.stop
    )
    raw_target = batch.raw_count_target[partition.start : partition.stop].to(
        device=device, dtype=torch.int32
    )
    local_count = torch.tensor(int(mask.sum()), device=device, dtype=torch.int64)
    local_positive = torch.tensor(
        int(((raw_target > 0) & mask).sum()), device=device, dtype=torch.int64
    )
    global_count = local_count.clone()
    global_positive = local_positive.clone()
    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(global_positive, op=torch.distributed.ReduceOp.SUM)

    def forward_backward() -> tuple[BinaryExpressionModelOutput, Any, Tensor]:
        optimizer.zero_grad(set_to_none=True)
        ddp_model.train()
        with _autocast_context(
            enabled=True, device=device, dtype_name=str(trainer["amp_dtype"])
        ):
            output = ddp_model(
                input_expression=input_expression,
                target_input_expression=target_input,
                target_gene_mask=mask,
                edge_index=batch.edge_index,
                relative_geometry=batch.relative_geometry,
                node_covariates=covariates,
                target_start=partition.start,
                target_stop=partition.stop,
            )
        if not isinstance(output, BinaryExpressionModelOutput):
            raise BinaryExpressionBCEPreflightError("Real training output type drifted.")
        sums = masked_binary_expression_bce_sums(
            output.nonzero_logits, raw_target, mask
        )
        if sums.masked_entries != int(local_count) or sums.positive_entries != int(local_positive):
            raise BinaryExpressionBCEPreflightError(
                "Real shard BCE counts do not match the target mask."
            )
        scaled_loss = sums.detection_nll_sum * (
            float(WORLD_SIZE) / global_count.to(torch.float32)
        )
        if scaled_loss.dtype != torch.float32 or not bool(torch.isfinite(scaled_loss)):
            raise BinaryExpressionBCEPreflightError("Real BCE is not finite FP32.")
        scaler.scale(scaled_loss).backward()
        scaler.unscale_(optimizer)
        return output, sums, scaled_loss

    first_output, first_sums, _ = forward_backward()
    first_head = _gradient_state(model.detection_head)
    first_trunk = _gradient_state(model.decoder.linear_in)
    first_blocks = [_gradient_state(block) for block in model.blocks]
    expected_first_step = bool(
        first_head["nonzero"]
        and first_trunk["all_finite"]
        and not first_trunk["nonzero"]
        and all(item["all_finite"] and not item["nonzero"] for item in first_blocks)
    )
    if not expected_first_step:
        raise BinaryExpressionBCEPreflightError(
            "Zero-initialized head did not produce the expected first-step gradient pattern."
        )
    first_clip = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(trainer["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    head_weight_norm_after_first = float(
        torch.linalg.vector_norm(model.detection_head.weight.detach().float()).cpu()
    )
    if head_weight_norm_after_first <= 0.0:
        raise BinaryExpressionBCEPreflightError(
            "The first optimizer step did not open a gradient path through the head."
        )

    second_output, second_sums, _ = forward_backward()
    block_gradients = [
        base_preflight._gradient_scope_receipt(block, label=f"graph_block_{index}")
        for index, block in enumerate(model.blocks)
    ]
    decoder_trunk = base_preflight._gradient_scope_receipt(
        model.decoder.linear_in, label="decoder_trunk"
    )
    detection_head = base_preflight._gradient_scope_receipt(
        model.detection_head, label="detection_logit_projection"
    )
    full_model = base_preflight._gradient_scope_receipt(model, label="full_model")
    second_clip = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(trainer["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    pooled = second_sums.detection_nll_sum.detach().clone()
    torch.distributed.all_reduce(pooled, op=torch.distributed.ReduceOp.SUM)
    pooled_bce = float((pooled / global_count.to(torch.float32)).cpu())
    local_records = _all_gather_objects(
        {
            "rank": rank,
            "target_start": partition.start,
            "target_stop": partition.stop,
            "target_count": partition.n_targets,
            "masked_entries": int(local_count),
            "positive_entries": int(local_positive),
            "local_bce_sum": float(second_sums.detection_nll_sum.detach().cpu()),
            "mask_receipt_sha256": realization.to_receipt()["receipt_sha256"],
        }
    )
    ordered = sorted(local_records, key=lambda item: int(item["rank"]))
    ranges = [(int(item["target_start"]), int(item["target_stop"])) for item in ordered]
    expected_ranges = [
        (item.start, item.stop)
        for item in make_contiguous_target_partitions(batch.n_nodes, WORLD_SIZE)
    ]
    if (
        ranges != expected_ranges
        or sum(int(item["target_count"]) for item in ordered) != batch.n_nodes
        or not math.isfinite(pooled_bce)
    ):
        raise BinaryExpressionBCEPreflightError(
            "Real training shards are incomplete or produced non-finite BCE."
        )
    training_receipt = {
        "passed": True,
        "alias": batch.alias,
        "full_core_nodes": batch.n_nodes,
        "full_core_edges": batch.n_edges,
        "graph_device": "cpu",
        "relative_geometry_device": "cpu",
        "rank_shards": ordered,
        "global_target_count": batch.n_nodes,
        "global_masked_entry_count": int(global_count),
        "global_positive_entry_count": int(global_positive),
        "pooled_masked_binary_cross_entropy": pooled_bce,
        "local_loss_scale_rule": (
            "world_size_times_local_bce_sum_divided_by_global_masked_entry_count"
        ),
        "matched_training_mask_seed_tuple_replays_exactly": True,
        "optimizer_steps": 2,
        "first_clip_norm_before_clip": float(first_clip.detach().cpu()),
        "second_clip_norm_before_clip": float(second_clip.detach().cpu()),
    }
    gradient_receipt = {
        "passed": True,
        "zero_head_initialization_expected_first_backward": {
            "head": first_head,
            "decoder_trunk": first_trunk,
            "blocks": first_blocks,
            "expected_pattern_observed": True,
        },
        "head_weight_l2_after_first_optimizer_step": head_weight_norm_after_first,
        "second_backward_full_model": full_model,
        "second_backward_decoder_trunk": decoder_trunk,
        "second_backward_blocks": block_gradients,
        "second_backward_detection_head": detection_head,
        "all_four_blocks_nonzero_after_head_update": True,
        "read_timing": "after_amp_unscale_before_gradient_clipping",
    }
    del first_output, second_output, target_input, mask, raw_target
    del input_expression, covariates, first_sums
    model.clear_edge_layout_cache()
    return training_receipt, gradient_receipt


def _histogram_auc(
    positive_histogram: Tensor,
    negative_histogram: Tensor,
    probability_sum_histogram: Tensor,
) -> tuple[float, float, float]:
    positive = positive_histogram.double().flip(0)
    negative = negative_histogram.double().flip(0)
    tp = positive.cumsum(0)
    fp = negative.cumsum(0)
    total_positive = float(positive.sum())
    total_negative = float(negative.sum())
    if total_positive <= 0.0 or total_negative <= 0.0:
        raise BinaryExpressionBCEPreflightError(
            "Histogram metric gate requires both binary classes."
        )
    recall = tp / total_positive
    false_positive_rate = fp / total_negative
    precision = tp / (tp + fp).clamp_min(1.0)
    zero = torch.zeros(1, dtype=torch.float64, device=recall.device)
    auroc = torch.trapz(
        torch.cat((zero, recall)), torch.cat((zero, false_positive_rate))
    )
    recall_delta = recall - torch.cat((zero, recall[:-1]))
    auprc = (recall_delta * precision).sum()
    count = positive_histogram.double() + negative_histogram.double()
    nonempty = count > 0.0
    empirical_positive = torch.zeros_like(count)
    mean_probability = torch.zeros_like(count)
    empirical_positive[nonempty] = positive_histogram.double()[nonempty] / count[nonempty]
    mean_probability[nonempty] = probability_sum_histogram.double()[nonempty] / count[nonempty]
    ece = (
        count[nonempty]
        * (empirical_positive[nonempty] - mean_probability[nonempty]).abs()
    ).sum() / count.sum()
    return float(auroc.cpu()), float(auprc.cpu()), float(ece.cpu())


def _real_validation_shard_replay(
    *,
    model: TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    ddp_model: DistributedDataParallel,
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    device: torch.device,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay the exact fixed C27 mask, BCE, and histogram metrics."""

    trainer = _section(config, "trainer")
    partition, first = _validation_mask_for_rank(batch, config, rank=rank)
    _, replay = _validation_mask_for_rank(batch, config, rank=rank)
    first_receipt = first.to_receipt()
    replay_receipt = replay.to_receipt()
    if first_receipt != replay_receipt or not np.array_equal(first.mask, replay.mask):
        raise BinaryExpressionBCEPreflightError(
            "Fixed validation target masks did not replay exactly."
        )
    input_expression = batch.input_expression.to(device=device, dtype=torch.float32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.array(first.mask, copy=True)).to(device=device)
    target_input = _make_target_inputs(
        input_expression, mask, start=partition.start, stop=partition.stop
    )
    raw_target = batch.raw_count_target[partition.start : partition.stop].to(
        device=device, dtype=torch.int32
    )
    local_counts = torch.tensor(
        [int(mask.sum()), int(((raw_target > 0) & mask).sum())],
        device=device,
        dtype=torch.int64,
    )
    global_counts = local_counts.clone()
    torch.distributed.all_reduce(global_counts, op=torch.distributed.ReduceOp.SUM)
    ddp_model.eval()
    histogram_bins = int(
        _section(config, "evaluation").get("probability_histogram_bins", 4096)
    )
    if histogram_bins != 4096:
        raise BinaryExpressionBCEPreflightError("Detection histogram bins must equal 4096.")

    def evaluate_once() -> tuple[tuple[float, float, float], Tensor, Tensor, Tensor]:
        with torch.no_grad(), _autocast_context(
            enabled=True, device=device, dtype_name=str(trainer["amp_dtype"])
        ):
            output = ddp_model(
                input_expression=input_expression,
                target_input_expression=target_input,
                target_gene_mask=mask,
                edge_index=batch.edge_index,
                relative_geometry=batch.relative_geometry,
                node_covariates=covariates,
                target_start=partition.start,
                target_stop=partition.stop,
            )
        if not isinstance(output, BinaryExpressionModelOutput):
            raise BinaryExpressionBCEPreflightError(
                "Validation binary-expression output type drifted."
            )
        sums = masked_binary_expression_bce_sums(
            output.nonzero_logits, raw_target, mask
        )
        probabilities = output.nonzero_probability.masked_select(mask).float()
        labels = raw_target.masked_select(mask) > 0
        brier_sum = ((probabilities - labels.float()) ** 2).sum()
        probability_sum = probabilities.sum()
        values = torch.stack(
            (sums.detection_nll_sum, brier_sum, probability_sum)
        ).double()
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        index = torch.clamp(
            (probabilities * histogram_bins).to(torch.int64),
            min=0,
            max=histogram_bins - 1,
        )
        positive_histogram = torch.bincount(
            index[labels], minlength=histogram_bins
        ).to(torch.int64)
        negative_histogram = torch.bincount(
            index[~labels], minlength=histogram_bins
        ).to(torch.int64)
        probability_sum_histogram = torch.zeros(
            histogram_bins, dtype=torch.float64, device=device
        )
        probability_sum_histogram.scatter_add_(
            0, index, probabilities.to(torch.float64)
        )
        torch.distributed.all_reduce(positive_histogram, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(negative_histogram, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            probability_sum_histogram, op=torch.distributed.ReduceOp.SUM
        )
        denominator = float(global_counts[0])
        metrics = tuple(float(value.cpu()) / denominator for value in values)
        if not all(math.isfinite(value) for value in metrics):
            raise BinaryExpressionBCEPreflightError(
                "Validation binary metrics are non-finite."
            )
        return (
            metrics,
            positive_histogram.cpu(),
            negative_histogram.cpu(),
            probability_sum_histogram.cpu(),
        )

    (
        first_metrics,
        first_positive_hist,
        first_negative_hist,
        first_probability_sum_hist,
    ) = evaluate_once()
    (
        replay_metrics,
        replay_positive_hist,
        replay_negative_hist,
        replay_probability_sum_hist,
    ) = evaluate_once()
    if (
        first_metrics != replay_metrics
        or not torch.equal(first_positive_hist, replay_positive_hist)
        or not torch.equal(first_negative_hist, replay_negative_hist)
        or not torch.equal(
            first_probability_sum_hist, replay_probability_sum_hist
        )
    ):
        raise BinaryExpressionBCEPreflightError(
            "Fixed validation BCE or histogram metrics failed exact replay."
        )
    auroc, auprc, ece = _histogram_auc(
        first_positive_hist,
        first_negative_hist,
        first_probability_sum_hist,
    )
    local_receipts = _all_gather_objects(
        {
            "rank": rank,
            "target_start": partition.start,
            "target_stop": partition.stop,
            "target_count": partition.n_targets,
            "masked_entries": int(local_counts[0]),
            "positive_entries": int(local_counts[1]),
            "receipt_sha256": first_receipt["receipt_sha256"],
            "mask_sha256": first_receipt["mask_sha256"],
        }
    )
    ordered = sorted(local_receipts, key=lambda item: int(item["rank"]))
    if sum(int(item["target_count"]) for item in ordered) != batch.n_nodes:
        raise BinaryExpressionBCEPreflightError(
            "Validation target shards are not exhaustive."
        )
    mask_gate = {
        "passed": True,
        "alias": batch.alias,
        "fixed_across_epochs": True,
        "rank_receipts": ordered,
        "rank_receipts_sha256": canonical_sha256(ordered),
        "global_target_count": batch.n_nodes,
        "global_masked_entry_count": int(global_counts[0]),
        "global_positive_entry_count": int(global_counts[1]),
        "all_masks_nonempty": True,
    }
    metric_gate = {
        "passed": True,
        "alias": batch.alias,
        "full_core_nodes": batch.n_nodes,
        "full_core_edges": batch.n_edges,
        "first_pooled_masked_binary_cross_entropy": first_metrics[0],
        "replayed_pooled_masked_binary_cross_entropy": replay_metrics[0],
        "detection_brier_score": first_metrics[1],
        "predicted_positive_probability_mean": first_metrics[2],
        "bit_exact_replay": True,
    }
    histogram_gate = {
        "passed": True,
        "bin_count": histogram_bins,
        "positive_histogram_total": int(first_positive_hist.sum()),
        "negative_histogram_total": int(first_negative_hist.sum()),
        "histogram_total_matches_masked_entries": (
            int(first_positive_hist.sum() + first_negative_hist.sum())
            == int(global_counts[0])
        ),
        "histogram_positive_total_matches_labels": (
            int(first_positive_hist.sum()) == int(global_counts[1])
        ),
        "auroc_histogram_4096": auroc,
        "auprc_histogram_4096": auprc,
        "expected_calibration_error_histogram_4096": ece,
        "bit_exact_replay": True,
    }
    if not (
        histogram_gate["histogram_total_matches_masked_entries"]
        and histogram_gate["histogram_positive_total_matches_labels"]
        and 0.0 <= auroc <= 1.0
        and 0.0 <= auprc <= 1.0
        and 0.0 <= ece <= 1.0
    ):
        raise BinaryExpressionBCEPreflightError("Histogram metric replay gate failed.")
    del target_input, mask, raw_target, input_expression, covariates
    model.clear_edge_layout_cache()
    return mask_gate, metric_gate, histogram_gate


def _amp_fp32_equivalence_receipt(
    model: TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    *,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, Any]:
    """Bound AMP drift while proving BCE arithmetic remains FP32."""

    clean, target_input, mask, covariates, edge_index, geometry, raw_target = (
        _synthetic_model_inputs(device=device)
    )
    previous_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            fp32 = model(
                input_expression=clean,
                target_input_expression=target_input,
                target_gene_mask=mask,
                edge_index=edge_index,
                relative_geometry=geometry,
                node_covariates=covariates,
                target_start=0,
                target_stop=2,
            )
            with _autocast_context(enabled=True, device=device, dtype_name=amp_dtype):
                amp = model(
                    input_expression=clean,
                    target_input_expression=target_input,
                    target_gene_mask=mask,
                    edge_index=edge_index,
                    relative_geometry=geometry,
                    node_covariates=covariates,
                    target_start=0,
                    target_stop=2,
                )
            fp32_sums = masked_binary_expression_bce_sums(
                fp32.nonzero_logits, raw_target, mask
            )
            amp_sums = masked_binary_expression_bce_sums(
                amp.nonzero_logits, raw_target, mask
            )
    finally:
        model.train(previous_training)
    fields = {
        "nonzero_logits": (fp32.nonzero_logits, amp.nonzero_logits),
        "nonzero_probability": (fp32.nonzero_probability, amp.nonzero_probability),
        "predicted_zero_probability": (
            fp32.predicted_zero_probability,
            amp.predicted_zero_probability,
        ),
    }
    differences = {
        name: {
            "maximum_absolute": float((left - right).abs().max().cpu()),
            "mean_absolute": float((left - right).abs().mean().cpu()),
        }
        for name, (left, right) in fields.items()
    }
    fp32_loss = fp32_sums.detection_nll_sum / float(fp32_sums.masked_entries)
    amp_loss = amp_sums.detection_nll_sum / float(amp_sums.masked_entries)
    loss_absolute = abs(float(fp32_loss) - float(amp_loss))
    loss_relative = loss_absolute / max(abs(float(fp32_loss)), 1e-12)
    loss_allowed = AMP_LOSS_ABS_TOLERANCE + (
        AMP_LOSS_RELATIVE_TOLERANCE * abs(float(fp32_loss))
    )
    public_values = (
        fp32.nonzero_logits,
        fp32.nonzero_probability,
        fp32.predicted_zero_probability,
        amp.nonzero_logits,
        amp.nonzero_probability,
        amp.predicted_zero_probability,
        fp32_sums.detection_nll_sum,
        amp_sums.detection_nll_sum,
    )
    if (
        any(value.dtype != torch.float32 for value in public_values)
        or any(
            value["maximum_absolute"] > AMP_OUTPUT_MAX_ABS_TOLERANCE
            or value["mean_absolute"] > AMP_OUTPUT_MEAN_ABS_TOLERANCE
            for value in differences.values()
        )
        or loss_absolute > loss_allowed
    ):
        raise BinaryExpressionBCEPreflightError("AMP/FP32 BCE equivalence gate failed.")
    model.clear_edge_layout_cache()
    return {
        "passed": True,
        "synthetic_nodes": int(clean.shape[0]),
        "synthetic_edges": int(edge_index.shape[1]),
        "target_count": int(mask.shape[0]),
        "masked_entries": int(mask.sum()),
        "amp_dtype": amp_dtype,
        "all_public_outputs_and_bce_sums_fp32": True,
        "output_differences": differences,
        "output_maximum_absolute_tolerance": AMP_OUTPUT_MAX_ABS_TOLERANCE,
        "output_mean_absolute_tolerance": AMP_OUTPUT_MEAN_ABS_TOLERANCE,
        "bce_absolute_difference": loss_absolute,
        "bce_relative_difference": loss_relative,
        "bce_combined_absolute_tolerance": loss_allowed,
    }


def _binary_loss_decrease_check(raw_counts: Tensor) -> dict[str, Any]:
    counts = raw_counts.detach().cpu().to(torch.int32)
    logits = nn.Parameter(torch.zeros(counts.shape, dtype=torch.float32))
    mask = torch.ones_like(counts, dtype=torch.bool)
    optimizer = torch.optim.Adam((logits,), lr=5e-2)

    def objective() -> tuple[Tensor, Any]:
        sums = masked_binary_expression_bce_sums(logits, counts, mask)
        return sums.detection_nll_sum / float(sums.masked_entries), sums

    initial_loss, initial_sums = objective()
    initial = float(initial_loss.detach())
    finite_nonzero_gradients = True
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective()
        loss.backward()
        finite_nonzero_gradients = bool(
            finite_nonzero_gradients
            and logits.grad is not None
            and bool(torch.isfinite(logits.grad).all())
            and float(torch.linalg.vector_norm(logits.grad)) > 0.0
        )
        optimizer.step()
    final_loss, final_sums = objective()
    final = float(final_loss.detach())
    if not finite_nonzero_gradients or not math.isfinite(final) or not final < initial:
        raise BinaryExpressionBCEPreflightError(
            "Bounded binary BCE did not decrease."
        )
    return {
        "initial_binary_cross_entropy": initial,
        "final_binary_cross_entropy": final,
        "decrease": initial - final,
        "steps": 16,
        "finite_nonzero_logit_gradients": finite_nonzero_gradients,
        "shape": list(counts.shape),
        "masked_entries": initial_sums.masked_entries,
        "positive_entries": initial_sums.positive_entries,
        "initial_bce_sum": float(initial_sums.detection_nll_sum.detach()),
        "final_bce_sum": float(final_sums.detection_nll_sum.detach()),
    }


def _bounded_binary_loss_decrease_receipt(batch: SO2NBCoreBatch) -> dict[str, Any]:
    synthetic = torch.tensor(
        [
            [0, 1, 2, 5, 729],
            [7, 0, 19, 3, 128],
            [1, 4, 0, 31, 512],
            [2, 8, 16, 0, 255],
        ],
        dtype=torch.int32,
    )
    real = batch.raw_count_target[:32, :64]
    return {
        "passed": True,
        "synthetic": _binary_loss_decrease_check(synthetic),
        "real_subset": {
            "source_alias": batch.alias,
            **_binary_loss_decrease_check(real),
        },
    }


def _checkpoint_roundtrip_receipt(
    *,
    output_parent: Path,
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
    model: TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    validation_metric: float,
    detection_probability: Tensor,
) -> dict[str, Any]:
    """Round-trip the custom binary_expression schema and retain no diagnostic checkpoint."""

    output_parent.mkdir(parents=True, exist_ok=True)
    early = EarlyStoppingState(
        best_value=float(validation_metric),
        best_epoch=1,
        bad_validations=0,
        completed_epoch=1,
        improved=True,
        should_stop=False,
        stop_reason=None,
    )
    state: dict[str, Any] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "campaign_id": CAMPAIGN_ID,
        "completed_epoch": 1,
        "optimizer_updates_completed": 2,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_state_dict": base_preflight._clone_to_cpu(model.state_dict()),
        "optimizer_state_dict": base_preflight._clone_to_cpu(optimizer.state_dict()),
        "amp_scaler_state_dict": base_preflight._clone_to_cpu(scaler.state_dict()),
        "early_stopping_state": asdict(early),
        "best_validation_metric": float(validation_metric),
        "best_epoch": 1,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "configuration_sha256": canonical_sha256(config),
        "detection_initialization_probability_sha256": (
            base_preflight._tensor_sha256(detection_probability)
        ),
        "gradient_vectors_persisted": False,
        "test_artifacts_present": False,
    }
    best_payload: dict[str, Any] = {
        **state,
        "checkpoint_role": "best",
        "best_checkpoint_sha256": None,
        "embedded_best_checkpoint": None,
        "embedded_best_checkpoint_tree_sha256": None,
    }
    prefix = ".target-isolated-binary-expression-bce-preflight-checkpoint-"
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix=prefix, dir=output_parent) as temporary:
        temporary_path = Path(temporary)
        store = AtomicBestLatestCheckpointStore(
            temporary,
            checkpoint_schema=CHECKPOINT_SCHEMA,
            protocol=PROTOCOL,
        )
        best = store.save_best(best_payload)
        latest_payload: dict[str, Any] = {
            **state,
            "checkpoint_role": "latest",
            "best_checkpoint_sha256": best.sha256,
            "embedded_best_checkpoint": base_preflight._clone_to_cpu(best_payload),
            "embedded_best_checkpoint_tree_sha256": base_preflight._tree_sha256(
                best_payload
            ),
        }
        latest = store.save_latest(latest_payload)
        inventory_before = sorted(path.name for path in store.directory.iterdir())
        loaded_best = store.load("best")
        loaded_latest = store.load("latest")
        memory_clone = _build_model(config, bundle, detection_probability).eval()
        memory_clone.load_state_dict(state["model_state_dict"], strict=True)
        clone = _build_model(config, bundle, detection_probability)
        clone.load_state_dict(loaded_best["model_state_dict"], strict=True)
        clone.eval()
        synthetic = _synthetic_model_inputs(device=torch.device("cpu"))
        clean, target_input, target_mask, covariates, edges, geometry, _ = synthetic
        with torch.no_grad():
            expected_output = memory_clone(
                input_expression=clean,
                target_input_expression=target_input,
                target_gene_mask=target_mask,
                edge_index=edges,
                relative_geometry=geometry,
                node_covariates=covariates,
                target_start=0,
                target_stop=2,
            )
            replayed_output = clone(
                input_expression=clean,
                target_input_expression=target_input,
                target_gene_mask=target_mask,
                edge_index=edges,
                relative_geometry=geometry,
                node_covariates=covariates,
                target_start=0,
                target_stop=2,
            )
        forward_replay_exact = bool(
            torch.equal(expected_output.nonzero_logits, replayed_output.nonzero_logits)
            and torch.equal(
                expected_output.nonzero_probability,
                replayed_output.nonzero_probability,
            )
        )
        expected_hash = base_preflight._state_dict_sha256(
            state["model_state_dict"]
        )
        best_hash = base_preflight._state_dict_sha256(clone.state_dict())
        latest_hash = base_preflight._state_dict_sha256(
            loaded_latest["model_state_dict"]
        )
        final = store.finalize_best_only()
        final_payload = store.load("best")
        final_hash = base_preflight._state_dict_sha256(
            final_payload["model_state_dict"]
        )
        inventory_after = sorted(path.name for path in store.directory.iterdir())
        if not (
            inventory_before == ["best.ckpt", "latest.ckpt"]
            and inventory_after == ["best.ckpt"]
            and expected_hash == best_hash == latest_hash == final_hash
            and final.sha256 == best.sha256
            and latest_payload["embedded_best_checkpoint_tree_sha256"]
            == base_preflight._tree_sha256(
                loaded_latest["embedded_best_checkpoint"]
            )
            and torch.equal(
                clone.detection_initial_nonzero_probability,
                detection_probability,
            )
            and forward_replay_exact
        ):
            raise BinaryExpressionBCEPreflightError(
                "Atomic binary_expression checkpoint roundtrip failed."
            )
        receipt = {
            "passed": True,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "protocol": PROTOCOL,
            "best_checkpoint_sha256": best.sha256,
            "latest_checkpoint_sha256": latest.sha256,
            "model_state_sha256": expected_hash,
            "detection_buffer_roundtrip_exact": True,
            "binary_forward_replay_bit_exact": forward_replay_exact,
            "sole_detection_head_roundtrip_exact": True,
            "active_inventory_before_finalization": inventory_before,
            "retained_inventory_after_finalization": inventory_after,
            "temporary_directory_removed_on_exit": True,
        }
    if temporary_path is None or temporary_path.exists():
        raise BinaryExpressionBCEPreflightError(
            "Diagnostic checkpoint directory was not removed."
        )
    return receipt


def _configuration_contract_receipt(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    contract: Path,
    contract_sha256: str,
    bundle: SO2NBDataBundle,
) -> dict[str, Any]:
    local = {
        "configuration_sha256": canonical_sha256(config),
        "configuration_file_sha256": sha256_file(config_path),
        "frozen_task_contract_sha256": contract_sha256,
        "frozen_task_contract_path": str(contract.relative_to(PROJECT_ROOT)),
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "split_fingerprint": bundle.split_fingerprint,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
    }
    identities = _all_gather_objects(local)
    if any(item != local for item in identities):
        raise BinaryExpressionBCEPreflightError(
            "Ranks resolved different binary_expression config, contract, or data identities."
        )
    return {
        "passed": True,
        **local,
        "rank_identity_count": len(identities),
        "all_ranks_identical": True,
    }


def _code_file_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in CODE_RELATIVE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise BinaryExpressionBCEPreflightError(
                f"Receipt-bound binary_expression source file is absent: {relative}."
            )
        hashes[relative.as_posix()] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _public_model_evidence(topology: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "class",
        "parameter_count",
        "graph_block_count",
        "unique_graph_block_count",
        "per_block_parameter_count",
        "detection_head_parameter_count",
        "state_dict_key_sha256",
        "target_input_is_separate",
        "static_clean_source_bank",
    )
    return {field: topology[field] for field in fields}


def validate_preflight_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the checksum-bound contract consumed by the binary_expression runner."""

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
        "code_hash_scope": CODE_HASH_SCOPE,
    }
    for field, expected_value in expected.items():
        if receipt.get(field) != expected_value:
            raise BinaryExpressionBCEPreflightError(
                f"Receipt field {field!r} drifted."
            )
    gates = receipt.get("gates")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != PREFLIGHT_REQUIRED_GATES
        or any(
            not isinstance(value, Mapping) or value.get("passed") is not True
            for value in gates.values()
        )
    ):
        raise BinaryExpressionBCEPreflightError(
            "Receipt required binary_expression-gate set is incomplete."
        )
    code_hashes = receipt.get("code_file_sha256")
    if not isinstance(code_hashes, Mapping) or set(code_hashes) != {
        path.as_posix() for path in CODE_RELATIVE_PATHS
    }:
        raise BinaryExpressionBCEPreflightError(
            "Receipt code-hash scope drifted."
        )
    model = receipt.get("model")
    model_fields = {
        "class",
        "parameter_count",
        "graph_block_count",
        "unique_graph_block_count",
        "per_block_parameter_count",
        "detection_head_parameter_count",
        "state_dict_key_sha256",
        "target_input_is_separate",
        "static_clean_source_bank",
    }
    if not isinstance(model, Mapping) or set(model) != model_fields:
        raise BinaryExpressionBCEPreflightError(
            "Receipt public model evidence schema drifted."
        )
    expected_model_values = {
        "class": (
            "TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer"
        ),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "graph_block_count": EXPECTED_BLOCK_COUNT,
        "unique_graph_block_count": EXPECTED_BLOCK_COUNT,
        "per_block_parameter_count": (
            [EXPECTED_BLOCK_PARAMETER_COUNT] * EXPECTED_BLOCK_COUNT
        ),
        "detection_head_parameter_count": EXPECTED_DETECTION_PARAMETER_COUNT,
        "target_input_is_separate": True,
        "static_clean_source_bank": True,
    }
    if any(model.get(field) != value for field, value in expected_model_values.items()):
        raise BinaryExpressionBCEPreflightError(
            "Receipt public model evidence values drifted."
        )
    dataset = receipt.get("dataset")
    dataset_fields = {
        "training_aliases",
        "validation_aliases",
        "test_aliases",
        "training_core_count",
        "validation_core_count",
        "test_core_count",
        "training_cells",
        "validation_cells",
        "test_cells",
        "biological_target_count",
        "node_covariate_count",
        "representative_training_alias",
        "representative_validation_alias",
        "representative_training_is_largest_by_nodes_and_edges",
        "representative_validation_is_largest_by_nodes_and_edges",
        "per_core",
    }
    if not isinstance(dataset, Mapping) or set(dataset) != dataset_fields:
        raise BinaryExpressionBCEPreflightError(
            "Receipt dataset evidence schema drifted."
        )
    stored = receipt.get("receipt_content_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_content_sha256", None)
    if (
        not isinstance(stored, str)
        or len(stored) != 64
        or any(character not in "0123456789abcdef" for character in stored)
        or stored != canonical_sha256(unsigned)
    ):
        raise BinaryExpressionBCEPreflightError(
            "Receipt self-checksum is invalid."
        )


def _distributed_identity() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BinaryExpressionBCEPreflightError(
            "This preflight must be invoked through torchrun."
        ) from exc
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE):
        raise BinaryExpressionBCEPreflightError(
            "Exactly four torchrun ranks are required."
        )
    if local_rank != rank:
        raise BinaryExpressionBCEPreflightError(
            "This single-node preflight requires RANK == LOCAL_RANK."
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != VISIBLE_DEVICES:
        raise BinaryExpressionBCEPreflightError(
            f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE_DEVICES}."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise BinaryExpressionBCEPreflightError(
            "Exactly four visible CUDA devices are required."
        )
    return rank, local_rank, world_size


def run_preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    """Execute every four-rank gate and atomically publish one receipt."""

    rank, local_rank, _ = _distributed_identity()
    if int(args.timeout_seconds) < 300:
        raise BinaryExpressionBCEPreflightError(
            "--timeout-seconds must be at least 300."
        )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=int(args.timeout_seconds)),
    )
    control_group = torch.distributed.new_group(
        backend="gloo",
        timeout=timedelta(seconds=int(args.timeout_seconds)),
    )
    # Generic helpers use their module's control-plane group.  Binding it here
    # keeps all rank-zero failures broadcast to peers rather than stranding an
    # NCCL collective.
    base_preflight._CONTROL_GROUP = control_group

    (
        paths,
        config_path,
        config,
        overlay_dir,
        cohort_dir,
        graph_dir,
        output,
        contract,
        contract_sha256,
    ) = _resolve_required_paths(args)
    base_preflight._synchronized_rank_zero_phase(
        "invalidate_stale_binary_expression_receipt",
        rank=rank,
        group=control_group,
        action=lambda: base_preflight._remove_stale_receipt(output),
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.cuda.manual_seed_all(seed)
    trainer = _section(config, "trainer")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(
        bool(trainer["deterministic"]),
        warn_only=bool(trainer["deterministic_warn_only"]),
    )

    bundle = load_so2_nb_data(
        overlay_dir,
        cohort_dir=cohort_dir,
        graph_dir=graph_dir,
    )
    base_preflight._validate_bundle_binding(config, bundle)
    configuration_gate = _configuration_contract_receipt(
        config=config,
        config_path=config_path,
        contract=contract,
        contract_sha256=contract_sha256,
        bundle=bundle,
    )
    dataset_evidence = base_preflight._dataset_receipt(bundle)
    overlay_evidence = base_preflight._overlay_evidence(bundle)
    training_batch = bundle.batches_by_alias[REPRESENTATIVE_TRAINING_ALIAS]
    validation_batch = bundle.batches_by_alias[REPRESENTATIVE_VALIDATION_ALIAS]

    initialization = base_preflight._synchronized_rank_zero_phase(
        "derive_train_only_detection_initialization",
        rank=rank,
        group=control_group,
        action=lambda: _training_nonzero_initialization(bundle),
    )
    if not isinstance(initialization, Mapping) or initialization.get("passed") is not True:
        raise BinaryExpressionBCEPreflightError(
            "Detection initialization evidence broadcast failed."
        )
    probability_values = initialization.get("probability")
    if not isinstance(probability_values, Sequence) or len(probability_values) != EXPECTED_GENES:
        raise BinaryExpressionBCEPreflightError(
            "Detection initialization probability vector is malformed."
        )
    detection_probability = torch.tensor(probability_values, dtype=torch.float32)

    model = _build_model(config, bundle, detection_probability)
    topology = _model_topology_receipt(model, detection_probability)
    model = model.to(device)
    detection_gate = _binary_head_and_probability_receipt(
        model,
        expected_probability=detection_probability,
        device=device,
    )
    invariant_gates = _model_invariant_receipts(model, device=device)
    ddp_model = base_preflight._wrap_target_model_ddp(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    scaler = base_preflight._make_grad_scaler()
    sync_before = base_preflight._synchronized_state_receipt(
        model, phase="after_binary_expression_ddp_initial_broadcast"
    )

    coverage_gate = base_preflight._synthetic_target_coverage_receipt(config)
    bce_reference_gate = _binary_bce_reference_receipt()
    unequal_count_gate = _unequal_count_ddp_weighting_receipt(
        device=device,
        rank=rank,
    )
    no_test_gate = base_preflight._no_test_artifacts_receipt(config)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    training_gate, gradient_gate = _real_training_shard_update(
        model=model,
        ddp_model=ddp_model,
        optimizer=optimizer,
        scaler=scaler,
        batch=training_batch,
        config=config,
        device=device,
        rank=rank,
    )
    sync_after = base_preflight._synchronized_state_receipt(
        model, phase="after_real_c23_binary_expression_optimizer_update"
    )
    (
        validation_mask_gate,
        validation_metric_gate,
        histogram_metric_gate,
    ) = _real_validation_shard_replay(
        model=model,
        ddp_model=ddp_model,
        batch=validation_batch,
        config=config,
        device=device,
        rank=rank,
    )
    amp_gate = _amp_fp32_equivalence_receipt(
        model,
        device=device,
        amp_dtype=str(trainer["amp_dtype"]),
    )
    torch.cuda.synchronize(device)
    vram_gate = base_preflight._peak_vram_receipt(
        device,
        rank=rank,
        local_rank=local_rank,
    )

    loss_decrease_gate = base_preflight._synchronized_rank_zero_phase(
        "bounded_binary_loss_decrease",
        rank=rank,
        group=control_group,
        action=lambda: _bounded_binary_loss_decrease_receipt(training_batch),
    )

    def rank_zero_artifact_evidence() -> dict[str, Any]:
        disk = base_preflight._disk_space_receipt(output)
        checkpoint = _checkpoint_roundtrip_receipt(
            output_parent=output.parent,
            config=config,
            bundle=bundle,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            validation_metric=float(
                validation_metric_gate[
                    "first_pooled_masked_binary_cross_entropy"
                ]
            ),
            detection_probability=detection_probability,
        )
        return {
            "disk": disk,
            "checkpoint": checkpoint,
            "configuration_sources": base_preflight._configuration_file_hashes(
                config_path,
                paths.config_root,
            ),
            "code_hashes": _code_file_hashes(),
        }

    artifact_evidence = base_preflight._synchronized_rank_zero_phase(
        "binary_expression_checkpoint_disk_and_hash_evidence",
        rank=rank,
        group=control_group,
        action=rank_zero_artifact_evidence,
    )
    if not isinstance(artifact_evidence, Mapping):
        raise BinaryExpressionBCEPreflightError(
            "Rank-zero binary_expression artifact evidence is malformed."
        )
    checkpoint_gate = artifact_evidence.get("checkpoint")
    disk_gate = artifact_evidence.get("disk")
    configuration_sources = artifact_evidence.get("configuration_sources")
    code_hashes = artifact_evidence.get("code_hashes")
    if not all(
        isinstance(value, Mapping)
        for value in (
            checkpoint_gate,
            disk_gate,
            configuration_sources,
            code_hashes,
            loss_decrease_gate,
        )
    ):
        raise BinaryExpressionBCEPreflightError(
            "Rank-zero binary_expression evidence broadcast failed."
        )

    gpu_gate = base_preflight._gpu_inventory_receipt(
        device,
        rank=rank,
        local_rank=local_rank,
    )
    ddp_sync_gate = {
        "passed": True,
        "before_optimizer_update": sync_before,
        "after_optimizer_update": sync_after,
    }
    gates: dict[str, Mapping[str, Any]] = {
        "configuration_and_contract": configuration_gate,
        "model_topology": topology,
        "binary_head_and_probability": detection_gate,
        "synthetic_target_coverage": coverage_gate,
        "binary_bce_reference": bce_reference_gate,
        "unequal_count_ddp_weighting": unequal_count_gate,
        "fixed_validation_mask_replay": validation_mask_gate,
        "fixed_validation_metric_replay": validation_metric_gate,
        "histogram_metric_replay": histogram_metric_gate,
        "amp_fp32_equivalence": amp_gate,
        "real_train_shard_forward_backward": training_gate,
        "preclip_gradient_flow": gradient_gate,
        "bounded_binary_loss_decrease": loss_decrease_gate,
        "checkpoint_roundtrip": checkpoint_gate,
        "ddp_parameter_sync": ddp_sync_gate,
        "gpu_inventory": gpu_gate,
        "peak_vram": vram_gate,
        "disk_space": disk_gate,
        "no_test_split_or_artifacts": no_test_gate,
        **invariant_gates,
    }
    if set(gates) != PREFLIGHT_REQUIRED_GATES or any(
        gate.get("passed") is not True for gate in gates.values()
    ):
        raise BinaryExpressionBCEPreflightError(
            "Internal required binary_expression-gate assembly is incomplete."
        )

    gpu_identities = list(gpu_gate["devices"])
    resources = base_preflight._resource_evidence(disk_gate, vram_gate)
    unsigned_receipt: dict[str, Any] = {
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
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "configuration_sha256": canonical_sha256(config),
        "resolved_config_sha256": canonical_sha256(config),
        "configuration_file_sha256": sha256_file(config_path),
        "configuration_source_file_sha256": dict(configuration_sources),
        "frozen_task_contract_sha256": contract_sha256,
        "overlay_manifest_sha256": bundle.manifest_sha256,
        "overlay_manifest_content_sha256": bundle.manifest_content_sha256,
        "split_fingerprint": bundle.split_fingerprint,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "source_artifacts": overlay_evidence["source_artifacts"],
        "source_artifacts_sha256": canonical_sha256(
            overlay_evidence["source_artifacts"]
        ),
        "code_file_sha256": dict(code_hashes),
        "code_hash_scope": CODE_HASH_SCOPE,
        "gpu_identities": gpu_identities,
        "gpu_identities_sha256": canonical_sha256(gpu_identities),
        "dataset": dataset_evidence,
        "dataset_evidence_sha256": canonical_sha256(dataset_evidence),
        "model": _public_model_evidence(topology),
        "detection_initialization": dict(initialization),
        "resource_gates": resources,
        "gates": gates,
    }
    unsigned_receipt["receipt_content_sha256"] = canonical_sha256(
        unsigned_receipt
    )

    def publish_receipt() -> Mapping[str, Any]:
        validate_preflight_receipt(unsigned_receipt)
        base_preflight._atomic_json(output, unsigned_receipt)
        written = json.loads(output.read_text(encoding="utf-8"))
        validate_preflight_receipt(written)
        return written

    receipt = base_preflight._synchronized_rank_zero_phase(
        "publish_checksum_bound_binary_expression_receipt",
        rank=rank,
        group=control_group,
        action=publish_receipt,
    )
    if not isinstance(receipt, Mapping):
        raise BinaryExpressionBCEPreflightError(
            "Published binary_expression receipt broadcast failed."
        )
    validate_preflight_receipt(receipt)
    torch.distributed.barrier(group=control_group)
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "receipt": str(output),
                    "receipt_content_sha256": receipt[
                        "receipt_content_sha256"
                    ],
                    "representative_training_alias": (
                        REPRESENTATIVE_TRAINING_ALIAS
                    ),
                    "representative_validation_alias": (
                        REPRESENTATIVE_VALIDATION_ALIAS
                    ),
                    "peak_reserved_vram_gib_all_ranks": resources[
                        "peak_reserved_vram_gib_all_ranks"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / SOURCE_EXPERIMENT_CONFIG,
        help="Frozen source binary_expression experiment YAML (overrides fail closed).",
    )
    parser.add_argument("--overlay-dir", type=Path, default=None)
    parser.add_argument("--cohort-dir", type=Path, default=None)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7_200,
        help="Per-collective timeout for the real full-core gate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_preflight(args)
        return 0
    except BaseException:
        rank = os.environ.get("RANK", "unknown")
        print(
            f"target-isolated binary_expression BCE preflight failed on rank {rank}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        return 1
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            finally:
                base_preflight._CONTROL_GROUP = None


if __name__ == "__main__":
    raise SystemExit(main())
