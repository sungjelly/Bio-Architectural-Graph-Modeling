#!/usr/bin/env python3
"""Four-rank launch gate for target-isolated SO2 hurdle-NB2 training.

Run this diagnostic through ``torchrun --nproc-per-node=4``.  It validates the
frozen hurdle configuration, derives the detection-head initialization from
training counts only, exercises the exact distributed objective on the largest
training core and the fixed validation task on the largest validation core,
and atomically writes one checksum-bound receipt.  It never launches training
and retains no diagnostic checkpoint.

The module deliberately does not import the production hurdle runner.  The
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
from spatial_benchmark.hurdle_negative_binomial import (  # noqa: E402
    HurdleNegativeBinomialModelOutput,
    TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    log1mexp,
    masked_hurdle_negative_binomial_nll_sums,
    zero_truncated_negative_binomial_nll_elementwise,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.negative_binomial import (  # noqa: E402
    NB2_INVERSE_DISPERSION_EPSILON,
    NB2_MEAN_EPSILON,
    nb2_inverse_dispersion_from_raw,
    nb2_mean_from_logits,
)
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
    "cmp_20260928_so2_target_isolated_geometry_modulated_hurdle_nb_"
    "once_per_cell_seed0"
)
PROTOCOL = (
    "donor_grouped_so2_target_isolated_hurdle_nb_once_per_cell_earlystop_v1"
)
PREFLIGHT_SCHEMA = (
    "so2_target_isolated_hurdle_nb_once_per_cell_ddp4_preflight_v1"
)
CHECKPOINT_SCHEMA = "so2_target_isolated_hurdle_nb_once_per_cell_checkpoint_v1"
PRIMARY_METRIC = "val/target_only_masked_hurdle_negative_binomial_nll"
WORLD_SIZE = 4
VISIBLE_DEVICES = "0,1,2,3"
EXPECTED_PARAMETER_COUNT = 6_160_088
EXPECTED_BASE_PARAMETER_COUNT = 5_135_088
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
AMP_OUTPUT_MAX_ABS_TOLERANCE = 8e-2
AMP_OUTPUT_MEAN_ABS_TOLERANCE = 1e-2
AMP_LOSS_ABS_TOLERANCE = 3e-2
AMP_LOSS_RELATIVE_TOLERANCE = 3e-3
SOURCE_EXPERIMENT_CONFIG = Path(
    "configs/experiment/"
    "so2_target_isolated_geometry_modulated_hurdle_nb_once_per_cell_seed0.yaml"
)
FROZEN_TASK_CONTRACT = Path(
    "experiments/campaigns/"
    "cmp_20260928_so2_target_isolated_geometry_modulated_hurdle_nb_"
    "once_per_cell_seed0/frozen_task_contract.yaml"
)
CODE_HASH_SCOPE = (
    "target_isolated_hurdle_model_likelihood_masking_data_training_preflight_"
    "runner_and_configuration_launch_contract"
)
CODE_RELATIVE_PATHS = (
    Path("scripts/diagnostics/preflight_so2_target_isolated_hurdle_nb_ddp.py"),
    Path("scripts/diagnostics/preflight_so2_target_isolated_nb_ddp.py"),
    Path("scripts/train/run_so2_target_isolated_hurdle_nb.py"),
    Path("scripts/train/run_so2_target_isolated_nb.py"),
    Path("scripts/train/run_so2_geometry_modulated_nb.py"),
    Path("src/spatial_benchmark/hurdle_negative_binomial.py"),
    Path("src/spatial_benchmark/target_isolated_negative_binomial.py"),
    Path("src/spatial_benchmark/target_cell_masking.py"),
    Path("src/spatial_benchmark/target_cell_training.py"),
    Path("src/spatial_benchmark/negative_binomial.py"),
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
        "shared_encoder_two_stream",
        "detection_head_and_probability",
        "synthetic_target_coverage",
        "target_mask_contract",
        "neighbor_context_unmasked",
        "incoming_nonself_routing",
        "masked_target_value_invariance",
        "raw_target_forward_isolation",
        "no_return_leakage",
        "hurdle_joint_likelihood_reference",
        "zero_truncated_nb2_fp32",
        "unequal_count_ddp_weighting",
        "fixed_validation_mask_replay",
        "fixed_validation_metric_replay",
        "amp_fp32_equivalence",
        "real_train_shard_forward_backward",
        "preclip_gradient_flow",
        "bounded_hurdle_loss_decrease",
        "checkpoint_roundtrip",
        "ddp_parameter_sync",
        "gpu_inventory",
        "peak_vram",
        "disk_space",
        "no_test_split_or_artifacts",
    }
)


class TargetIsolatedHurdleNBPreflightError(RuntimeError):
    """Raised when a hurdle launch-safety gate fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise TargetIsolatedHurdleNBPreflightError(
            f"Resolved configuration requires a {name!r} mapping."
        )
    return value


def _runtime_path(raw: object, paths: ProjectPaths) -> Path:
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if not candidate.parts or ".." in candidate.parts:
        raise TargetIsolatedHurdleNBPreflightError(
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


def _validate_hurdle_config(config: Mapping[str, Any]) -> None:
    campaign = _section(config, "campaign")
    evaluation = _section(config, "evaluation")
    model = _section(config, "model")
    trainer = _section(config, "trainer")
    launcher = _section(config, "launcher")
    dataset = _section(config, "dataset")
    masking = _section(config, "masking")
    expected_model = {
        "name": "target-isolated-geometry-modulated-relative-qkv-gat-hurdle-nb2",
        "family": (
            "target_isolated_geometry_modulated_relative_qkv_hurdle_graph_transformer"
        ),
        "output_distribution": (
            "bernoulli_hurdle_zero_truncated_negative_binomial_nb2"
        ),
        "decoder_trunk": "shared_count_and_detection",
        "decoder_trunk_dim": 1024,
        "count_mean_projection_output_dim": EXPECTED_GENES,
        "detection_logit_projection_output_dim": EXPECTED_GENES,
        "detection_probability_activation": "sigmoid",
        "detection_probability_epsilon": DETECTION_EPSILON,
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
        "objective": (
            "pooled_masked_entry_full_constant_bernoulli_hurdle_"
            "zero_truncated_negative_binomial_nb2_nll"
        ),
        "objective_numerator": (
            "sum_detection_nll_all_masked_plus_zero_truncated_nb2_nll_"
            "positive_masked"
        ),
        "objective_denominator": (
            "globally_reduced_masked_target_entry_count_per_update"
        ),
        "local_ddp_loss_scale": (
            "world_size_times_local_joint_nll_sum_divided_by_global_"
            "masked_entry_count"
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
            raise TargetIsolatedHurdleNBPreflightError(
                f"model.{field} must be {expected!r}; got {model.get(field)!r}."
            )
    for field, expected in expected_trainer.items():
        if trainer.get(field) != expected:
            raise TargetIsolatedHurdleNBPreflightError(
                f"trainer.{field} must be {expected!r}; got {trainer.get(field)!r}."
            )
    if campaign.get("campaign_id") != CAMPAIGN_ID or config.get("seed") != 0:
        raise TargetIsolatedHurdleNBPreflightError(
            "Campaign identity or seed drifted."
        )
    if (
        evaluation.get("protocol") != PROTOCOL
        or evaluation.get("primary_metric") != PRIMARY_METRIC
        or evaluation.get("primary_aggregation")
        != "sum_masked_entry_joint_nll_divided_by_total_masked_entries"
        or evaluation.get("splits") != ["validation"]
        or evaluation.get("test_split_present") is not False
    ):
        raise TargetIsolatedHurdleNBPreflightError(
            "Evaluation protocol or primary hurdle metric drifted."
        )
    if (
        dataset.get("training_cells") != EXPECTED_TRAINING_CELLS
        or dataset.get("validation_cells") != EXPECTED_VALIDATION_CELLS
        or dataset.get("test_cells") != 0
        or dataset.get("training_core_aliases") != list(SO2_NB_TRAINING_ALIASES)
        or dataset.get("validation_core_aliases") != list(SO2_NB_VALIDATION_ALIASES)
        or dataset.get("test_core_aliases") != []
    ):
        raise TargetIsolatedHurdleNBPreflightError("Dataset split contract drifted.")
    if (
        masking.get("count_min") != 1
        or masking.get("count_max") != EXPECTED_GENES
        or masking.get("target_cell_only") is not True
        or masking.get("neighbor_cells_artificially_masked") is not False
        or masking.get("neighbor_context_rows_fully_observed") is not True
    ):
        raise TargetIsolatedHurdleNBPreflightError("Target-only masking drifted.")
    if (
        launcher.get("process_count") != WORLD_SIZE
        or launcher.get("require_exact_visible_devices") != VISIBLE_DEVICES
        or launcher.get("disk_safety_min_free_gb") != 20
        or launcher.get("hardware_preflight_receipt")
        != "state/preflight/so2_target_isolated_hurdle_nb_once_per_cell_ddp4.json"
    ):
        raise TargetIsolatedHurdleNBPreflightError("Launcher contract drifted.")


def _resolve_required_paths(
    args: argparse.Namespace,
) -> tuple[ProjectPaths, Path, dict[str, Any], Path, Path, Path, Path, Path, str]:
    paths = current_paths(anchor=PROJECT_ROOT)
    paths.validate()
    if paths.project_root.resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise TargetIsolatedHurdleNBPreflightError(
            "BAGM_ROOT must identify the source tree containing this preflight."
        )
    expected_config = (PROJECT_ROOT / SOURCE_EXPERIMENT_CONFIG).resolve(strict=True)
    config_path = _argument_path(args.config, default=expected_config)
    if config_path != expected_config:
        raise TargetIsolatedHurdleNBPreflightError(
            "--config must be the frozen hurdle experiment source."
        )
    config = compose_config(config_path, config_root=paths.config_root, validate=True)
    _validate_hurdle_config(config)
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
        raise TargetIsolatedHurdleNBPreflightError(
            "Data/output overrides must resolve to paths frozen in config."
        )
    overlay, cohort, graph, output = resolved
    for value, label in ((overlay, "overlay"), (cohort, "cohort"), (graph, "graph")):
        if not value.is_dir():
            raise TargetIsolatedHurdleNBPreflightError(
                f"Configured {label} directory is absent: {value}."
            )
    contract = (PROJECT_ROOT / FROZEN_TASK_CONTRACT).resolve(strict=True)
    contract_sha = sha256_file(contract)
    metadata = _section(config, "metadata")
    if metadata.get("frozen_task_contract_sha256") != contract_sha:
        raise TargetIsolatedHurdleNBPreflightError(
            "Frozen hurdle task-contract checksum drifted."
        )
    if output.is_symlink():
        raise TargetIsolatedHurdleNBPreflightError(
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
            raise TargetIsolatedHurdleNBPreflightError(
                "Detection initialization encountered a non-training core."
            )
        for start in range(0, batch.n_nodes, 2048):
            stop = min(start + 2048, batch.n_nodes)
            positive += (batch.raw_count_target[start:stop] > 0).sum(
                dim=0, dtype=torch.int64
            )
        total_cells += batch.n_nodes
    if total_cells != EXPECTED_TRAINING_CELLS:
        raise TargetIsolatedHurdleNBPreflightError(
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
        raise TargetIsolatedHurdleNBPreflightError(
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
) -> TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer:
    model_config = _section(config, "model")
    reference = bundle.training_batches[0]
    torch.manual_seed(int(config["seed"]))
    return TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer(
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
        mean_epsilon=float(model_config["output_mean_epsilon"]),
        inverse_dispersion_epsilon=float(
            model_config["inverse_dispersion_epsilon"]
        ),
        inverse_dispersion_initial_value=float(
            model_config["inverse_dispersion_initial_value"]
        ),
        detection_initial_nonzero_probability=detection_probability,
        detection_probability_epsilon=float(
            model_config["detection_probability_epsilon"]
        ),
    )


def _model_topology_receipt(
    model: TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
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
    detection = getattr(model.decoder, "detection_linear_out", None)
    if not isinstance(detection, nn.Linear):
        raise TargetIsolatedHurdleNBPreflightError("Detection projection is absent.")
    detection_count = sum(parameter.numel() for parameter in detection.parameters())
    expected_bias = torch.logit(expected_probability.cpu())
    weight_zero = bool((detection.weight.detach().cpu() == 0.0).all())
    bias_exact = torch.equal(detection.bias.detach().cpu(), expected_bias)
    buffer_exact = torch.equal(
        model.detection_initial_nonzero_probability.detach().cpu(),
        expected_probability.cpu(),
    )
    passed = bool(
        count == EXPECTED_PARAMETER_COUNT
        and len(blocks) == EXPECTED_BLOCK_COUNT
        and len({id(block) for block in blocks}) == EXPECTED_BLOCK_COUNT
        and block_counts == [EXPECTED_BLOCK_PARAMETER_COUNT] * EXPECTED_BLOCK_COUNT
        and disjoint
        and detection_count == EXPECTED_DETECTION_PARAMETER_COUNT
        and model.raw_theta.numel() == EXPECTED_GENES
        and weight_zero
        and bias_exact
        and buffer_exact
    )
    if not passed:
        raise TargetIsolatedHurdleNBPreflightError(
            "Hurdle model topology or initialization drifted."
        )
    return {
        "passed": True,
        "class": type(model).__name__,
        "parameter_count": count,
        "base_nb_parameter_count": EXPECTED_BASE_PARAMETER_COUNT,
        "detection_head_parameter_count": detection_count,
        "graph_block_count": len(blocks),
        "unique_graph_block_count": len({id(block) for block in blocks}),
        "per_block_parameter_count": block_counts,
        "raw_theta_parameter_count": model.raw_theta.numel(),
        "state_dict_key_sha256": canonical_sha256(list(model.state_dict().keys())),
        "detection_weight_zero_initialized": weight_zero,
        "detection_bias_matches_train_prevalence": bias_exact,
        "detection_probability_buffer_matches": buffer_exact,
        "target_input_is_separate": True,
        "static_clean_source_bank": True,
        "shared_encoder_module_count": 1,
    }


def _independent_joint_values(
    logits: Tensor,
    mu: Tensor,
    theta: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Independent FP64 reference for the normalized hurdle likelihood."""

    work_logits = logits.double()
    work_mu = mu.double()
    work_theta = theta.double().view(1, -1).expand_as(work_mu)
    work_target = target.double()
    positive = work_target > 0.0
    detection = F.binary_cross_entropy_with_logits(
        work_logits, positive.double(), reduction="none"
    )
    log_probability = (
        torch.lgamma(work_target + work_theta)
        - torch.lgamma(work_theta)
        - torch.lgamma(work_target + 1.0)
        + work_theta * (torch.log(work_theta) - torch.log(work_theta + work_mu))
        + work_target * (torch.log(work_mu) - torch.log(work_theta + work_mu))
    )
    log_zero = work_theta * (
        torch.log(work_theta) - torch.log(work_theta + work_mu)
    )
    count = torch.zeros_like(detection)
    count[positive] = (-log_probability + torch.log(-torch.expm1(log_zero)))[
        positive
    ]
    return detection + count, detection, count


def _hurdle_reference_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    logits = torch.tensor([[-8.0, -1.0, 0.0, 2.0, 8.0, -12.0, 12.0]])
    mu = torch.tensor([[1e-4, 1e-3, 0.1, 1.0, 10.0, 100.0, 1_000.0]])
    theta = torch.tensor([1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1_000.0])
    target = torch.tensor([[0, 1, 2, 3, 8, 29, 729]], dtype=torch.int32)
    mask = torch.ones_like(target, dtype=torch.bool)
    sums = masked_hurdle_negative_binomial_nll_sums(
        logits, mu, theta, target, mask
    )
    joint_ref, detection_ref, count_ref = _independent_joint_values(
        logits, mu, theta, target
    )
    errors = {
        "joint": abs(float(sums.joint_nll_sum) - float(joint_ref.sum())),
        "detection": abs(
            float(sums.detection_nll_sum) - float(detection_ref.sum())
        ),
        "positive_count": abs(
            float(sums.positive_count_nll_sum) - float(count_ref.sum())
        ),
    }
    tolerance = 5e-4
    if max(errors.values()) > tolerance:
        raise TargetIsolatedHurdleNBPreflightError(
            f"Hurdle likelihood disagrees with independent FP64 reference: {errors}."
        )
    joint_gate = {
        "passed": True,
        "dtype": str(sums.joint_nll_sum.dtype),
        "masked_entries": sums.masked_entries,
        "positive_entries": sums.positive_entries,
        "zero_entries": sums.masked_entries - sums.positive_entries,
        "joint_nll_sum": float(sums.joint_nll_sum),
        "detection_nll_sum": float(sums.detection_nll_sum),
        "positive_count_nll_sum": float(sums.positive_count_nll_sum),
        "maximum_absolute_error_vs_independent_fp64": max(errors.values()),
        "absolute_tolerance": tolerance,
        "normalization": "joint_sum_divided_by_all_masked_entries",
    }

    extreme_log_probability = torch.tensor(
        [-1e-7, -1e-4, -0.1, -10.0, -80.0], dtype=torch.float32
    )
    stable_log_nonzero = log1mexp(extreme_log_probability)
    positive_target = target.float().masked_select(target > 0)
    selected_mu = mu.masked_select(target > 0)
    selected_theta = theta.view(1, -1).expand_as(mu).masked_select(target > 0)
    truncated = zero_truncated_negative_binomial_nll_elementwise(
        selected_mu, selected_theta, positive_target
    )
    poisson_limit_mu = torch.tensor([1e-4, 1.0, 10.0], dtype=torch.float32)
    poisson_limit_theta = torch.full((3,), 1e6, dtype=torch.float32)
    poisson_limit_target = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32)
    poisson_limit_values = zero_truncated_negative_binomial_nll_elementwise(
        poisson_limit_mu,
        poisson_limit_theta,
        poisson_limit_target,
    )
    reference_mu = poisson_limit_mu.double()
    reference_theta = poisson_limit_theta.double()
    reference_target = poisson_limit_target.double()
    reference_log_probability = (
        torch.lgamma(reference_target + reference_theta)
        - torch.lgamma(reference_theta)
        - torch.lgamma(reference_target + 1.0)
        + reference_theta
        * (
            torch.log(reference_theta)
            - torch.log(reference_theta + reference_mu)
        )
        + reference_target
        * (torch.log(reference_mu) - torch.log(reference_theta + reference_mu))
    )
    reference_log_zero = reference_theta * (
        torch.log(reference_theta) - torch.log(reference_theta + reference_mu)
    )
    poisson_limit_reference = -reference_log_probability + torch.log(
        -torch.expm1(reference_log_zero)
    )
    poisson_limit_error = float(
        (poisson_limit_values.double() - poisson_limit_reference).abs().max()
    )
    if (
        stable_log_nonzero.dtype != torch.float32
        or truncated.dtype != torch.float32
        or poisson_limit_values.dtype != torch.float32
        or not bool(torch.isfinite(stable_log_nonzero).all())
        or not bool(torch.isfinite(truncated).all())
        or not bool(torch.isfinite(poisson_limit_values).all())
        or bool((truncated < 0.0).any())
        or bool((poisson_limit_values < 0.0).any())
        or poisson_limit_error > 1e-3
    ):
        raise TargetIsolatedHurdleNBPreflightError(
            "FP32 zero-truncated NB2 stability gate failed."
        )
    stability_gate = {
        "passed": True,
        "dtype": "torch.float32",
        "contains_count_729": True,
        "mu_range": [float(mu.min()), float(mu.max())],
        "theta_range": [float(theta.min()), float(theta.max())],
        "log_nonzero_input_range": [
            float(extreme_log_probability.min()),
            float(extreme_log_probability.max()),
        ],
        "all_values_finite": True,
        "all_truncated_nll_nonnegative": True,
        "maximum_reference_error": errors["positive_count"],
        "absolute_tolerance": tolerance,
        "large_theta_poisson_limit": {
            "theta": 1e6,
            "target_counts": poisson_limit_target.tolist(),
            "maximum_absolute_error_vs_fp64": poisson_limit_error,
            "absolute_tolerance": 1e-3,
        },
    }
    return joint_gate, stability_gate


class _SyntheticHurdleParameters(nn.Module):
    """Tiny differentiable hurdle model for the exact DDP-weighting gate."""

    def __init__(self) -> None:
        super().__init__()
        self.detection_bias = nn.Parameter(torch.tensor(-0.35))
        self.count_logit = nn.Parameter(torch.tensor(0.2))
        theta_initial = math.log(math.expm1(1.0 - NB2_INVERSE_DISPERSION_EPSILON))
        self.raw_theta = nn.Parameter(torch.tensor(theta_initial))

    def forward(
        self, feature: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.detection_bias + 0.15 * feature
        mu = F.softplus(self.count_logit + 0.1 * feature).float() + NB2_MEAN_EPSILON
        theta = nb2_inverse_dispersion_from_raw(
            self.raw_theta.view(1), epsilon=NB2_INVERSE_DISPERSION_EPSILON
        )
        return logits.view(-1, 1), mu.view(-1, 1), theta


def _synthetic_rank_values(rank: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
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
    module = _SyntheticHurdleParameters().to(device)
    distributed = DistributedDataParallel(
        module,
        device_ids=None,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    feature, target, mask = _synthetic_rank_values(rank, device)
    logits, mu, theta = distributed(feature)
    sums = masked_hurdle_negative_binomial_nll_sums(
        logits, mu, theta, target, mask
    )
    local_count = torch.tensor(sums.masked_entries, dtype=torch.int64, device=device)
    global_count = local_count.clone()
    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
    scaled = sums.joint_nll_sum * (
        float(WORLD_SIZE) / global_count.to(torch.float32)
    )
    scaled.backward()
    ddp_gradient = torch.stack(
        [
            distributed.module.detection_bias.grad.detach(),
            distributed.module.count_logit.grad.detach(),
            distributed.module.raw_theta.grad.detach(),
        ]
    ).float()

    reference = _SyntheticHurdleParameters().to(device)
    reference.load_state_dict(distributed.module.state_dict())
    reference_sum = torch.zeros((), dtype=torch.float32, device=device)
    reference_count = 0
    local_counts: list[int] = []
    for reference_rank in range(WORLD_SIZE):
        ref_feature, ref_target, ref_mask = _synthetic_rank_values(
            reference_rank, device
        )
        ref_logits, ref_mu, ref_theta = reference(ref_feature)
        ref_sums = masked_hurdle_negative_binomial_nll_sums(
            ref_logits, ref_mu, ref_theta, ref_target, ref_mask
        )
        reference_sum = reference_sum + ref_sums.joint_nll_sum
        reference_count += ref_sums.masked_entries
        local_counts.append(ref_sums.masked_entries)
    (reference_sum / float(reference_count)).backward()
    reference_gradient = torch.stack(
        [
            reference.detection_bias.grad.detach(),
            reference.count_logit.grad.detach(),
            reference.raw_theta.grad.detach(),
        ]
    ).float()
    difference = float(torch.max(torch.abs(ddp_gradient - reference_gradient)).cpu())
    tolerance = 2e-5
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
        raise TargetIsolatedHurdleNBPreflightError(
            "Unequal-count DDP hurdle weighting disagrees with pooled reference."
        )
    return {
        "passed": True,
        "local_masked_entry_counts_by_rank": local_counts,
        "global_masked_entry_count": reference_count,
        "ddp_loss_scale_rule": (
            "world_size_times_local_joint_nll_sum_divided_by_global_masked_entry_count"
        ),
        "gradient_order": ["detection_bias", "count_logit", "raw_theta"],
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
        raise TargetIsolatedHurdleNBPreflightError(
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
        raise TargetIsolatedHurdleNBPreflightError("Target masking failed.")
    if not torch.equal(
        target.masked_select(~mask),
        clean_expression[start:stop].masked_select(~mask),
    ):
        raise TargetIsolatedHurdleNBPreflightError(
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


def _output_fingerprint(output: HurdleNegativeBinomialModelOutput, row: int) -> Tensor:
    return torch.cat(
        (
            output.nb_mean[row],
            output.nonzero_logits[row],
            output.marginal_mean[row],
        )
    )


def _model_invariant_receipts(
    model: TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
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
            raise TargetIsolatedHurdleNBPreflightError(
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
    if not isinstance(baseline, HurdleNegativeBinomialModelOutput):
        raise TargetIsolatedHurdleNBPreflightError("Hurdle output type drifted.")

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
        raise TargetIsolatedHurdleNBPreflightError(
            "Target/context routing or leakage invariant failed."
        )
    model.train(previous_training)
    model.clear_edge_layout_cache()
    return {
        "shared_encoder_two_stream": {
            "passed": True,
            "shared_encoder_object_count": 1,
            "encoder_masked_entry_counts_by_call": encoder_mask_counts,
            "clean_source_bank_static": True,
        },
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
            "own_output_bit_identical_for_both_heads": own_equal,
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
            "withheld_value_cannot_return_to_either_head": own_equal,
            "visible_target_effect_observed": visible_changed,
            "clean_neighbor_effect_observed": neighbor_changed,
        },
    }


def _detection_head_and_probability_receipt(
    model: TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    *,
    expected_probability: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Prove both decoder heads and the hurdle probability identities."""

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
    if not isinstance(output, HurdleNegativeBinomialModelOutput):
        raise TargetIsolatedHurdleNBPreflightError(
            "Detection-head gate received the wrong output type."
        )
    expected = expected_probability.to(device=device).view(1, -1).expand(2, -1)
    initial_probability_error = float(
        (output.nonzero_probability - expected).abs().max().cpu()
    )
    identities = bool(
        torch.equal(output.predicted_zero_probability, 1.0 - output.nonzero_probability)
        and torch.equal(
            output.marginal_mean,
            output.nonzero_probability * output.positive_conditional_mean,
        )
        and torch.equal(output.prediction, output.marginal_mean)
    )
    finite = all(
        value.dtype == torch.float32 and bool(torch.isfinite(value).all())
        for value in (
            output.nb_mean,
            output.theta,
            output.nonzero_logits,
            output.nonzero_probability,
            output.predicted_zero_probability,
            output.positive_conditional_mean,
            output.marginal_mean,
        )
    )
    detection = model.decoder.detection_linear_out
    count = model.decoder.linear_out
    passed = bool(
        finite
        and identities
        and initial_probability_error <= 2e-7
        and bool((output.nb_mean > 0.0).all())
        and bool((output.theta > 0.0).all())
        and bool((output.nonzero_probability > 0.0).all())
        and bool((output.nonzero_probability < 1.0).all())
        and bool((output.positive_conditional_mean >= 1.0 - 1e-5).all())
        and isinstance(detection, nn.Linear)
        and isinstance(count, nn.Linear)
        and detection is not count
        and detection.out_features == EXPECTED_GENES
        and count.out_features == EXPECTED_GENES
    )
    if not passed:
        raise TargetIsolatedHurdleNBPreflightError(
            "Detection/count head or hurdle probability identity gate failed."
        )
    return {
        "passed": True,
        "count_head": "decoder.linear_out",
        "detection_head": "decoder.detection_linear_out",
        "heads_are_distinct": True,
        "head_output_dimensions": [count.out_features, detection.out_features],
        "probability_activation": "sigmoid",
        "initial_probability_maximum_absolute_error": initial_probability_error,
        "probability_tolerance": 2e-7,
        "predicted_zero_probability_identity": True,
        "positive_conditional_mean_lower_bound": 1.0,
        "marginal_mean_identity": True,
        "prediction_is_marginal_mean": True,
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


def _real_training_shard_update(
    *,
    model: TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    device: torch.device,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exercise one exact C23 target shard and the production DDP scaling."""

    trainer = _section(config, "trainer")
    partition, realization = _training_mask_for_rank(batch, config, rank=rank)
    if partition.n_targets <= 0:
        raise TargetIsolatedHurdleNBPreflightError(
            "Representative training shard is empty."
        )
    if batch.edge_index.device.type != "cpu" or batch.relative_geometry.device.type != "cpu":
        raise TargetIsolatedHurdleNBPreflightError(
            "Target-isolated graph and geometry must remain CPU-resident."
        )
    input_expression = batch.input_expression.to(device=device, dtype=torch.float32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.array(realization.mask, copy=True)).to(device=device)
    target_input = _make_target_inputs(
        input_expression,
        mask,
        start=partition.start,
        stop=partition.stop,
    )
    raw_target = batch.raw_count_target[partition.start : partition.stop].to(
        device=device,
        dtype=torch.int32,
    )
    local_count = torch.tensor(int(mask.sum()), device=device, dtype=torch.int64)
    local_positive = torch.tensor(
        int(((raw_target > 0) & mask).sum()), device=device, dtype=torch.int64
    )
    global_count = local_count.clone()
    global_positive = local_positive.clone()
    torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(global_positive, op=torch.distributed.ReduceOp.SUM)
    optimizer.zero_grad(set_to_none=True)
    ddp_model.train()
    with _autocast_context(
        enabled=True,
        device=device,
        dtype_name=str(trainer["amp_dtype"]),
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
    if not isinstance(output, HurdleNegativeBinomialModelOutput):
        raise TargetIsolatedHurdleNBPreflightError(
            "Real training output type drifted."
        )
    sums = masked_hurdle_negative_binomial_nll_sums(
        output.nonzero_logits,
        output.nb_mean,
        output.theta,
        raw_target,
        mask,
    )
    if sums.masked_entries != int(local_count) or sums.positive_entries != int(
        local_positive
    ):
        raise TargetIsolatedHurdleNBPreflightError(
            "Real shard likelihood counts do not match the target mask."
        )
    scaled_loss = sums.joint_nll_sum * (
        float(WORLD_SIZE) / global_count.to(torch.float32)
    )
    if scaled_loss.dtype != torch.float32 or not bool(torch.isfinite(scaled_loss)):
        raise TargetIsolatedHurdleNBPreflightError(
            "Real hurdle training loss is not finite FP32."
        )
    scaler.scale(scaled_loss).backward()
    scaler.unscale_(optimizer)

    block_gradients = [
        base_preflight._gradient_scope_receipt(
            block, label=f"graph_block_{index}"
        )
        for index, block in enumerate(model.blocks)
    ]
    count_head = base_preflight._gradient_scope_receipt(
        model.decoder.linear_out, label="count_mean_projection"
    )
    detection_head = base_preflight._gradient_scope_receipt(
        model.decoder.detection_linear_out, label="detection_logit_projection"
    )
    theta_gradient = model.raw_theta.grad
    if theta_gradient is None or not bool(torch.isfinite(theta_gradient).all()):
        raise TargetIsolatedHurdleNBPreflightError(
            "raw_theta gradient is absent or non-finite."
        )
    theta_norm = float(torch.linalg.vector_norm(theta_gradient.float()).cpu())
    if theta_norm <= 0.0:
        raise TargetIsolatedHurdleNBPreflightError("raw_theta gradient is zero.")
    full_model = base_preflight._gradient_scope_receipt(model, label="full_model")
    clip_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=float(trainer["gradient_clip_norm"]),
        error_if_nonfinite=True,
    )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    pooled = torch.stack(
        (
            sums.joint_nll_sum.detach(),
            sums.detection_nll_sum.detach(),
            sums.positive_count_nll_sum.detach(),
        )
    )
    torch.distributed.all_reduce(pooled, op=torch.distributed.ReduceOp.SUM)
    pooled_values = (pooled / global_count.to(torch.float32)).cpu().tolist()
    local_records = _all_gather_objects(
        {
            "rank": rank,
            "target_start": partition.start,
            "target_stop": partition.stop,
            "target_count": partition.n_targets,
            "masked_entries": int(local_count),
            "positive_entries": int(local_positive),
            "local_joint_nll_sum": float(sums.joint_nll_sum.detach().cpu()),
            "mask_receipt_sha256": realization.to_receipt()["receipt_sha256"],
        }
    )
    ordered = sorted(local_records, key=lambda item: int(item["rank"]))
    ranges = [
        (int(item["target_start"]), int(item["target_stop"])) for item in ordered
    ]
    expected_ranges = [
        (item.start, item.stop)
        for item in make_contiguous_target_partitions(batch.n_nodes, WORLD_SIZE)
    ]
    if (
        ranges != expected_ranges
        or sum(int(item["target_count"]) for item in ordered) != batch.n_nodes
        or not all(math.isfinite(float(value)) for value in pooled_values)
    ):
        raise TargetIsolatedHurdleNBPreflightError(
            "Real C23 shards are incomplete or produced non-finite metrics."
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
        "pooled_masked_joint_hurdle_nll": float(pooled_values[0]),
        "pooled_masked_detection_nll": float(pooled_values[1]),
        "pooled_positive_count_nll_divided_by_masked_entries": float(
            pooled_values[2]
        ),
        "local_loss_scale_rule": (
            "world_size_times_local_joint_nll_sum_divided_by_global_"
            "masked_entry_count"
        ),
        "optimizer_steps": 1,
        "clip_norm_before_clip": float(clip_norm.detach().cpu()),
    }
    gradient_receipt = {
        "passed": True,
        "full_model": full_model,
        "blocks": block_gradients,
        "count_head": count_head,
        "detection_head": detection_head,
        "raw_theta": {
            "finite": True,
            "nonzero": True,
            "l2_norm_before_clip": theta_norm,
        },
        "read_timing": "after_amp_unscale_before_gradient_clipping",
    }
    del output, target_input, mask, raw_target, input_expression, covariates
    model.clear_edge_layout_cache()
    return training_receipt, gradient_receipt


def _real_validation_shard_replay(
    *,
    model: TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    ddp_model: DistributedDataParallel,
    batch: SO2NBCoreBatch,
    config: Mapping[str, Any],
    device: torch.device,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay the exact fixed C27 mask and pooled joint hurdle metric."""

    trainer = _section(config, "trainer")
    partition, first = _validation_mask_for_rank(batch, config, rank=rank)
    _, replay = _validation_mask_for_rank(batch, config, rank=rank)
    first_receipt = first.to_receipt()
    replay_receipt = replay.to_receipt()
    if first_receipt != replay_receipt or not np.array_equal(first.mask, replay.mask):
        raise TargetIsolatedHurdleNBPreflightError(
            "Fixed validation target masks did not replay exactly."
        )
    input_expression = batch.input_expression.to(device=device, dtype=torch.float32)
    covariates = batch.node_covariates.to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.array(first.mask, copy=True)).to(device=device)
    target_input = _make_target_inputs(
        input_expression,
        mask,
        start=partition.start,
        stop=partition.stop,
    )
    raw_target = batch.raw_count_target[partition.start : partition.stop].to(
        device=device,
        dtype=torch.int32,
    )
    local_counts = torch.tensor(
        [int(mask.sum()), int(((raw_target > 0) & mask).sum())],
        device=device,
        dtype=torch.int64,
    )
    global_counts = local_counts.clone()
    torch.distributed.all_reduce(global_counts, op=torch.distributed.ReduceOp.SUM)
    ddp_model.eval()

    def evaluate_once() -> tuple[float, float, float]:
        with torch.no_grad(), _autocast_context(
            enabled=True,
            device=device,
            dtype_name=str(trainer["amp_dtype"]),
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
        if not isinstance(output, HurdleNegativeBinomialModelOutput):
            raise TargetIsolatedHurdleNBPreflightError(
                "Validation hurdle output type drifted."
            )
        sums = masked_hurdle_negative_binomial_nll_sums(
            output.nonzero_logits,
            output.nb_mean,
            output.theta,
            raw_target,
            mask,
        )
        values = torch.stack(
            (
                sums.joint_nll_sum.detach(),
                sums.detection_nll_sum.detach(),
                sums.positive_count_nll_sum.detach(),
            )
        )
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        metrics = (values / global_counts[0].to(torch.float32)).cpu().tolist()
        if not all(math.isfinite(float(value)) for value in metrics):
            raise TargetIsolatedHurdleNBPreflightError(
                "Validation hurdle metrics are non-finite."
            )
        return tuple(float(value) for value in metrics)

    first_metrics = evaluate_once()
    replay_metrics = evaluate_once()
    if first_metrics != replay_metrics:
        raise TargetIsolatedHurdleNBPreflightError(
            "Fixed validation joint hurdle metric failed exact replay."
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
        raise TargetIsolatedHurdleNBPreflightError(
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
        "first_pooled_joint_hurdle_nll": first_metrics[0],
        "replayed_pooled_joint_hurdle_nll": replay_metrics[0],
        "first_pooled_detection_nll": first_metrics[1],
        "first_positive_count_nll_divided_by_masked_entries": first_metrics[2],
        "bit_exact_replay": True,
    }
    del target_input, mask, raw_target, input_expression, covariates
    model.clear_edge_layout_cache()
    return mask_gate, metric_gate


def _amp_fp32_equivalence_receipt(
    model: TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    *,
    device: torch.device,
    amp_dtype: str,
) -> dict[str, Any]:
    """Bound AMP drift while proving that output likelihood arithmetic is FP32."""

    (
        clean,
        target_input,
        mask,
        covariates,
        edge_index,
        geometry,
        raw_target,
    ) = _synthetic_model_inputs(device=device)
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
            with _autocast_context(
                enabled=True, device=device, dtype_name=amp_dtype
            ):
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
            fp32_sums = masked_hurdle_negative_binomial_nll_sums(
                fp32.nonzero_logits,
                fp32.nb_mean,
                fp32.theta,
                raw_target,
                mask,
            )
            amp_sums = masked_hurdle_negative_binomial_nll_sums(
                amp.nonzero_logits,
                amp.nb_mean,
                amp.theta,
                raw_target,
                mask,
            )
    finally:
        model.train(previous_training)
    fields = {
        "nb_mean": (fp32.nb_mean, amp.nb_mean),
        "nonzero_probability": (
            fp32.nonzero_probability,
            amp.nonzero_probability,
        ),
        "marginal_mean": (fp32.marginal_mean, amp.marginal_mean),
    }
    differences = {
        name: {
            "maximum_absolute": float((left - right).abs().max().cpu()),
            "mean_absolute": float((left - right).abs().mean().cpu()),
        }
        for name, (left, right) in fields.items()
    }
    fp32_loss = fp32_sums.joint_nll_sum / float(fp32_sums.masked_entries)
    amp_loss = amp_sums.joint_nll_sum / float(amp_sums.masked_entries)
    loss_absolute = abs(float(fp32_loss) - float(amp_loss))
    loss_relative = loss_absolute / max(abs(float(fp32_loss)), 1e-12)
    loss_allowed = AMP_LOSS_ABS_TOLERANCE + (
        AMP_LOSS_RELATIVE_TOLERANCE * abs(float(fp32_loss))
    )
    public_values = (
        fp32.nb_mean,
        fp32.theta,
        fp32.nonzero_logits,
        fp32.nonzero_probability,
        fp32.positive_conditional_mean,
        fp32.marginal_mean,
        amp.nb_mean,
        amp.theta,
        amp.nonzero_logits,
        amp.nonzero_probability,
        amp.positive_conditional_mean,
        amp.marginal_mean,
        fp32_sums.joint_nll_sum,
        amp_sums.joint_nll_sum,
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
        raise TargetIsolatedHurdleNBPreflightError(
            "AMP/FP32 hurdle equivalence gate failed."
        )
    model.clear_edge_layout_cache()
    return {
        "passed": True,
        "synthetic_nodes": int(clean.shape[0]),
        "synthetic_edges": int(edge_index.shape[1]),
        "target_count": int(mask.shape[0]),
        "masked_entries": int(mask.sum()),
        "amp_dtype": amp_dtype,
        "all_public_outputs_and_likelihood_sums_fp32": True,
        "output_differences": differences,
        "output_maximum_absolute_tolerance": AMP_OUTPUT_MAX_ABS_TOLERANCE,
        "output_mean_absolute_tolerance": AMP_OUTPUT_MEAN_ABS_TOLERANCE,
        "joint_nll_absolute_difference": loss_absolute,
        "joint_nll_relative_difference": loss_relative,
        "joint_nll_combined_absolute_tolerance": loss_allowed,
    }


def _hurdle_loss_decrease_check(raw_counts: Tensor) -> dict[str, Any]:
    counts = raw_counts.detach().cpu().to(torch.int32)
    detection_logits = nn.Parameter(torch.zeros(counts.shape, dtype=torch.float32))
    count_logits = nn.Parameter(torch.zeros(counts.shape, dtype=torch.float32))
    raw_theta = nn.Parameter(
        torch.full(
            (counts.shape[1],),
            math.log(math.expm1(1.0 - NB2_INVERSE_DISPERSION_EPSILON)),
            dtype=torch.float32,
        )
    )
    mask = torch.ones_like(counts, dtype=torch.bool)
    optimizer = torch.optim.Adam(
        (detection_logits, count_logits, raw_theta), lr=5e-2
    )

    def objective() -> tuple[Tensor, Any]:
        sums = masked_hurdle_negative_binomial_nll_sums(
            detection_logits,
            nb2_mean_from_logits(count_logits, epsilon=NB2_MEAN_EPSILON),
            nb2_inverse_dispersion_from_raw(
                raw_theta, epsilon=NB2_INVERSE_DISPERSION_EPSILON
            ),
            counts,
            mask,
        )
        return sums.joint_nll_sum / float(sums.masked_entries), sums

    initial_loss, initial_sums = objective()
    initial = float(initial_loss.detach())
    gradient_scopes = {
        "detection_logits": True,
        "count_logits": True,
        "raw_theta": True,
    }
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective()
        loss.backward()
        for name, parameter in (
            ("detection_logits", detection_logits),
            ("count_logits", count_logits),
            ("raw_theta", raw_theta),
        ):
            gradient_scopes[name] = bool(
                gradient_scopes[name]
                and parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(torch.linalg.vector_norm(parameter.grad)) > 0.0
            )
        optimizer.step()
    final_loss, final_sums = objective()
    final = float(final_loss.detach())
    if not all(gradient_scopes.values()) or not math.isfinite(final) or not final < initial:
        raise TargetIsolatedHurdleNBPreflightError(
            "Bounded joint hurdle likelihood did not decrease through both heads."
        )
    return {
        "initial_joint_nll": initial,
        "final_joint_nll": final,
        "decrease": initial - final,
        "steps": 16,
        "finite_nonzero_gradient_scopes": gradient_scopes,
        "shape": list(counts.shape),
        "masked_entries": initial_sums.masked_entries,
        "positive_entries": initial_sums.positive_entries,
        "initial_detection_nll_sum": float(
            initial_sums.detection_nll_sum.detach()
        ),
        "initial_positive_count_nll_sum": float(
            initial_sums.positive_count_nll_sum.detach()
        ),
        "final_detection_nll_sum": float(final_sums.detection_nll_sum.detach()),
        "final_positive_count_nll_sum": float(
            final_sums.positive_count_nll_sum.detach()
        ),
    }


def _bounded_hurdle_loss_decrease_receipt(
    batch: SO2NBCoreBatch,
) -> dict[str, Any]:
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
        "synthetic": _hurdle_loss_decrease_check(synthetic),
        "real_subset": {
            "source_alias": batch.alias,
            **_hurdle_loss_decrease_check(real),
        },
    }


def _checkpoint_roundtrip_receipt(
    *,
    output_parent: Path,
    config: Mapping[str, Any],
    bundle: SO2NBDataBundle,
    model: TargetIsolatedGeometryModulatedHurdleNegativeBinomialGraphTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    validation_metric: float,
    detection_probability: Tensor,
) -> dict[str, Any]:
    """Round-trip the custom hurdle schema and retain no diagnostic checkpoint."""

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
        "optimizer_updates_completed": 1,
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
    prefix = ".target-isolated-hurdle-nb-preflight-checkpoint-"
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
        clone = _build_model(config, bundle, detection_probability)
        clone.load_state_dict(loaded_best["model_state_dict"], strict=True)
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
        ):
            raise TargetIsolatedHurdleNBPreflightError(
                "Atomic hurdle checkpoint roundtrip failed."
            )
        receipt = {
            "passed": True,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "protocol": PROTOCOL,
            "best_checkpoint_sha256": best.sha256,
            "latest_checkpoint_sha256": latest.sha256,
            "model_state_sha256": expected_hash,
            "detection_buffer_roundtrip_exact": True,
            "both_decoder_heads_roundtrip_exact": True,
            "active_inventory_before_finalization": inventory_before,
            "retained_inventory_after_finalization": inventory_after,
            "temporary_directory_removed_on_exit": True,
        }
    if temporary_path is None or temporary_path.exists():
        raise TargetIsolatedHurdleNBPreflightError(
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
        raise TargetIsolatedHurdleNBPreflightError(
            "Ranks resolved different hurdle config, contract, or data identities."
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
            raise TargetIsolatedHurdleNBPreflightError(
                f"Receipt-bound hurdle source file is absent: {relative}."
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
        "raw_theta_parameter_count",
        "state_dict_key_sha256",
        "target_input_is_separate",
        "static_clean_source_bank",
    )
    return {field: topology[field] for field in fields}


def validate_preflight_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the checksum-bound contract consumed by the hurdle runner."""

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
            raise TargetIsolatedHurdleNBPreflightError(
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
        raise TargetIsolatedHurdleNBPreflightError(
            "Receipt required hurdle-gate set is incomplete."
        )
    code_hashes = receipt.get("code_file_sha256")
    if not isinstance(code_hashes, Mapping) or set(code_hashes) != {
        path.as_posix() for path in CODE_RELATIVE_PATHS
    }:
        raise TargetIsolatedHurdleNBPreflightError(
            "Receipt code-hash scope drifted."
        )
    model = receipt.get("model")
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
    if not isinstance(model, Mapping) or set(model) != model_fields:
        raise TargetIsolatedHurdleNBPreflightError(
            "Receipt public model evidence schema drifted."
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
        raise TargetIsolatedHurdleNBPreflightError(
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
        raise TargetIsolatedHurdleNBPreflightError(
            "Receipt self-checksum is invalid."
        )


def _distributed_identity() -> tuple[int, int, int]:
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TargetIsolatedHurdleNBPreflightError(
            "This preflight must be invoked through torchrun."
        ) from exc
    if world_size != WORLD_SIZE or rank not in range(WORLD_SIZE):
        raise TargetIsolatedHurdleNBPreflightError(
            "Exactly four torchrun ranks are required."
        )
    if local_rank != rank:
        raise TargetIsolatedHurdleNBPreflightError(
            "This single-node preflight requires RANK == LOCAL_RANK."
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != VISIBLE_DEVICES:
        raise TargetIsolatedHurdleNBPreflightError(
            f"CUDA_VISIBLE_DEVICES must be exactly {VISIBLE_DEVICES}."
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise TargetIsolatedHurdleNBPreflightError(
            "Exactly four visible CUDA devices are required."
        )
    return rank, local_rank, world_size


def run_preflight(args: argparse.Namespace) -> Mapping[str, Any]:
    """Execute every four-rank gate and atomically publish one receipt."""

    rank, local_rank, _ = _distributed_identity()
    if int(args.timeout_seconds) < 300:
        raise TargetIsolatedHurdleNBPreflightError(
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
        "invalidate_stale_hurdle_receipt",
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
        raise TargetIsolatedHurdleNBPreflightError(
            "Detection initialization evidence broadcast failed."
        )
    probability_values = initialization.get("probability")
    if not isinstance(probability_values, Sequence) or len(probability_values) != EXPECTED_GENES:
        raise TargetIsolatedHurdleNBPreflightError(
            "Detection initialization probability vector is malformed."
        )
    detection_probability = torch.tensor(probability_values, dtype=torch.float32)

    model = _build_model(config, bundle, detection_probability)
    topology = _model_topology_receipt(model, detection_probability)
    model = model.to(device)
    detection_gate = _detection_head_and_probability_receipt(
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
        model, phase="after_hurdle_ddp_initial_broadcast"
    )

    coverage_gate = base_preflight._synthetic_target_coverage_receipt(config)
    joint_reference_gate, truncated_fp32_gate = _hurdle_reference_receipts()
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
        model, phase="after_real_c23_hurdle_optimizer_update"
    )
    validation_mask_gate, validation_metric_gate = _real_validation_shard_replay(
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
        "bounded_joint_hurdle_loss_decrease",
        rank=rank,
        group=control_group,
        action=lambda: _bounded_hurdle_loss_decrease_receipt(training_batch),
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
                validation_metric_gate["first_pooled_joint_hurdle_nll"]
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
        "hurdle_checkpoint_disk_and_hash_evidence",
        rank=rank,
        group=control_group,
        action=rank_zero_artifact_evidence,
    )
    if not isinstance(artifact_evidence, Mapping):
        raise TargetIsolatedHurdleNBPreflightError(
            "Rank-zero hurdle artifact evidence is malformed."
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
        raise TargetIsolatedHurdleNBPreflightError(
            "Rank-zero hurdle evidence broadcast failed."
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
        "detection_head_and_probability": detection_gate,
        "synthetic_target_coverage": coverage_gate,
        "hurdle_joint_likelihood_reference": joint_reference_gate,
        "zero_truncated_nb2_fp32": truncated_fp32_gate,
        "unequal_count_ddp_weighting": unequal_count_gate,
        "fixed_validation_mask_replay": validation_mask_gate,
        "fixed_validation_metric_replay": validation_metric_gate,
        "amp_fp32_equivalence": amp_gate,
        "real_train_shard_forward_backward": training_gate,
        "preclip_gradient_flow": gradient_gate,
        "bounded_hurdle_loss_decrease": loss_decrease_gate,
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
        raise TargetIsolatedHurdleNBPreflightError(
            "Internal required hurdle-gate assembly is incomplete."
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
        "publish_checksum_bound_hurdle_receipt",
        rank=rank,
        group=control_group,
        action=publish_receipt,
    )
    if not isinstance(receipt, Mapping):
        raise TargetIsolatedHurdleNBPreflightError(
            "Published hurdle receipt broadcast failed."
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
        help="Frozen source hurdle experiment YAML (overrides fail closed).",
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
            f"target-isolated hurdle-NB2 preflight failed on rank {rank}",
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
