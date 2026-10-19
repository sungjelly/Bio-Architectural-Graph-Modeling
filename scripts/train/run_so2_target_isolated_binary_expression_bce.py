#!/usr/bin/env python3
"""Train the matched SO2 target-isolated Bernoulli-only expression model.

The graph encoder, target isolation, masks, four-rank execution, optimizer,
scheduler, stopping controls, checkpoint transaction, and scalar-only gradient
telemetry are inherited from the audited target-isolated NB2 runner.  This
runner replaces only the output contract and objective: one logit per gene and
unweighted masked BCE for ``1[raw_count > 0]``.  It has no count mean,
dispersion parameter, or count-reconstruction loss.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel


_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SOURCE_ROOT))
sys.path.insert(0, str(_SOURCE_ROOT / "src"))

from scripts.train import run_so2_target_isolated_nb as base  # noqa: E402
from spatial_benchmark.binary_expression import (  # noqa: E402
    BinaryExpressionModelOutput,
    TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
    masked_binary_expression_bce_sums,
)
from spatial_benchmark.so2_nb_data import SO2NBDataBundle, SO2NBCoreBatch  # noqa: E402
from spatial_benchmark.so2_nb_training import (  # noqa: E402
    EarlyStoppingState,
    SO2NBTrainingError,
)
from spatial_benchmark.training import _autocast_context, set_deterministic_seed  # noqa: E402


CAMPAIGN_ID = (
    "cmp_20260930_so2_target_isolated_geometry_modulated_binary_expression_"
    "bce_once_per_cell_seed0"
)
MODEL_NAME = (
    "target-isolated-geometry-modulated-relative-qkv-gat-binary-expression-bce"
)
MODEL_FAMILY = (
    "target_isolated_geometry_modulated_relative_qkv_binary_expression_"
    "graph_transformer"
)
EVALUATION_PROTOCOL = (
    "donor_grouped_so2_target_isolated_binary_expression_bce_once_per_cell_"
    "earlystop_v1"
)
PROTOCOL = EVALUATION_PROTOCOL
PREFLIGHT_SCHEMA = (
    "so2_target_isolated_binary_expression_bce_once_per_cell_ddp4_preflight_v1"
)
CHECKPOINT_SCHEMA = (
    "so2_target_isolated_binary_expression_bce_once_per_cell_checkpoint_v1"
)
EPOCH_METRICS_SCHEMA = (
    "so2_target_isolated_binary_expression_bce_once_per_cell_epoch_metrics_v1"
)
PER_CORE_METRICS_SCHEMA = (
    "so2_target_isolated_binary_expression_bce_once_per_cell_per_core_metrics_v1"
)
EXPECTED_PARAMETER_COUNT = 5_134_088
DETECTION_HEAD_PARAMETER_COUNT = 1_025_000
PRIMARY_METRIC = "val/target_only_masked_binary_cross_entropy"
TRAIN_PRIMARY_METRIC = "train/pooled_masked_binary_cross_entropy"
PROVENANCE_RELATIVE_PATH = (
    "provenance/so2_target_isolated_binary_expression_training.json"
)
OBJECTIVE_AGGREGATION = "exact_global_masked_entry_weighted_unweighted_binary_bce"
TRACKED_HEAD_GRADIENT_RELATIVE_PATH = (
    "results/gradient_direction_detection_head.csv"
)
DETECTION_CURVE_RELATIVE_PATH = "results/detection_curve_histogram_4096.json"
ADDITIONAL_REQUIRED_FINAL_OUTPUTS = (DETECTION_CURVE_RELATIVE_PATH,)
PREFLIGHT_TRACKED_HEAD_COUNT_FIELD = "detection_head_parameter_count"
PREFLIGHT_TRACKED_HEAD_PARAMETER_COUNT = DETECTION_HEAD_PARAMETER_COUNT
DETECTION_PROBABILITY_EPSILON = 1e-4
HISTOGRAM_BIN_COUNT = 4096
DETECTION_THRESHOLD = 0.5
OPERATING_THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)

FROZEN_TASK_CONTRACT_SHA256 = (
    "4d1e030bb597c4a7dce64da8adf3c57743bf8b4a0cbd1e646d2986cca114335a"
)
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
PREFLIGHT_CODE_HASH_SCOPE = (
    "target_isolated_binary_expression_model_masking_data_training_preflight_"
    "runner_and_configuration_launch_contract"
)
PREFLIGHT_CODE_RELATIVE_PATHS = (
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

EPOCH_COLUMNS = (
    "schema",
    "global_epoch",
    "train_pooled_masked_binary_cross_entropy",
    "train_masked_entries",
    "train_positive_entries",
    "train_observed_positive_rate",
    "train_target_cells",
    "validation_pooled_masked_binary_cross_entropy",
    "validation_masked_entries",
    "validation_positive_entries",
    "validation_target_cells",
    "validation_observed_positive_rate",
    "validation_predicted_positive_probability_mean",
    "validation_predicted_positive_class_rate",
    "validation_detection_brier_score",
    "validation_detection_threshold",
    "validation_detection_true_positive",
    "validation_detection_true_negative",
    "validation_detection_false_positive",
    "validation_detection_false_negative",
    "validation_detection_accuracy",
    "validation_detection_balanced_accuracy",
    "validation_detection_recall",
    "validation_detection_specificity",
    "validation_detection_precision",
    "validation_detection_f1",
    "validation_detection_mcc",
    "validation_detection_auroc_histogram_4096",
    "validation_detection_auprc_histogram_4096",
    "validation_detection_expected_calibration_error_histogram_4096",
    "validation_operating_threshold_metrics_json",
    "detection_head_bias_min",
    "detection_head_bias_median",
    "detection_head_bias_max",
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
    "detection_head_gradient_json",
)

PER_CORE_COLUMNS = (
    "schema",
    "global_epoch",
    "split",
    "core_alias",
    "target_cells",
    "masked_entries",
    "positive_entries",
    "pooled_masked_binary_cross_entropy",
    "observed_positive_rate",
    "predicted_positive_probability_mean",
    "predicted_positive_class_rate",
    "detection_brier_score",
    "detection_true_positive",
    "detection_true_negative",
    "detection_false_positive",
    "detection_false_negative",
    "detection_accuracy",
    "detection_balanced_accuracy",
    "detection_recall",
    "detection_specificity",
    "detection_precision",
    "detection_f1",
    "detection_mcc",
    "detection_auroc_histogram_4096",
    "detection_auprc_histogram_4096",
    "detection_expected_calibration_error_histogram_4096",
    "coverage_receipt_sha256",
)

_SCALAR_FIELDS = (
    "masked_entries",
    "positive_entries",
    "detection_nll_sum",
    "predicted_positive_probability_sum",
    "detection_brier_sum",
    "detection_true_positive",
    "detection_true_negative",
    "detection_false_positive",
    "detection_false_negative",
    "target_cells",
)
_THRESHOLD_COUNT_FIELDS = tuple(
    f"threshold_{index}_{field}"
    for index in range(len(OPERATING_THRESHOLDS))
    for field in ("tp", "tn", "fp", "fn")
)
_SCALAR_COUNT = len(_SCALAR_FIELDS) + len(_THRESHOLD_COUNT_FIELDS)
_POSITIVE_HISTOGRAM_SLICE = slice(_SCALAR_COUNT, _SCALAR_COUNT + HISTOGRAM_BIN_COUNT)
_NEGATIVE_HISTOGRAM_SLICE = slice(
    _POSITIVE_HISTOGRAM_SLICE.stop,
    _POSITIVE_HISTOGRAM_SLICE.stop + HISTOGRAM_BIN_COUNT,
)
_PROBABILITY_SUM_HISTOGRAM_SLICE = slice(
    _NEGATIVE_HISTOGRAM_SLICE.stop,
    _NEGATIVE_HISTOGRAM_SLICE.stop + HISTOGRAM_BIN_COUNT,
)
_VALIDATION_VECTOR_LENGTH = _PROBABILITY_SUM_HISTOGRAM_SLICE.stop

_DETECTION_INITIALIZATION_RECEIPT: dict[str, Any] = {}


def _require_config_values(config: Mapping[str, Any]) -> None:
    """Reassert values that materially define the BCE-only comparison."""

    base.validate_experiment_config(config)
    base._require_equal(config.get("seed"), 0, field="seed")
    expected_sections: Mapping[str, Mapping[str, Any]] = {
        "campaign": {"campaign_id": CAMPAIGN_ID},
        "metadata": {
            "frozen_task_contract_sha256": FROZEN_TASK_CONTRACT_SHA256,
            "no_test_split_or_artifacts": True,
        },
        "model": {
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
            "output_distribution": "bernoulli_binary_expression",
            "detection_logit_projection_output_dim": 1000,
            "detection_projection_weight_initialization": "zeros",
            "detection_projection_bias_initialization": (
                "logit_jeffreys_smoothed_training_nonzero_prevalence"
            ),
            "expected_trainable_parameter_count": EXPECTED_PARAMETER_COUNT,
        },
        "masking": {
            "type": "target_only_uniform_per_cell_integer_count",
            "count_min": 1,
            "count_max": 1000,
            "training_mask_realizations_per_target_per_global_epoch": 1,
            "target_visit_count_per_global_epoch": 1,
            "target_cell_only": True,
            "neighbor_cells_artificially_masked": False,
            "neighbor_context_rows_fully_observed": True,
            "mask_base_seed": base.MASK_BASE_SEED,
            "training_mask_seed_namespace": base.TRAINING_MASK_NAMESPACE,
        },
        "dataset": {
            "training_core_aliases": list(base.TRAINING_ALIASES),
            "validation_core_aliases": list(base.VALIDATION_ALIASES),
            "test_core_aliases": [],
            "training_cells": 208_696,
            "validation_cells": 37_367,
            "test_cells": 0,
            "biological_target_count": 1000,
            "node_covariate_count": 22,
            "test_partition_present": False,
        },
        "evaluation": {
            "protocol": EVALUATION_PROTOCOL,
            "primary_metric": PRIMARY_METRIC,
            "splits": ["validation"],
            "test_split_present": False,
            "decision_threshold": DETECTION_THRESHOLD,
            "probability_histogram_bins": HISTOGRAM_BIN_COUNT,
        },
        "trainer": {
            "optimizer": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "gradient_clip_norm": 1.0,
            "cores_per_optimizer_update": 2,
            "optimizer_updates_per_global_epoch": 6,
            "target_shards_per_core": 4,
            "objective": (
                "pooled_unweighted_masked_binary_cross_entropy_with_logits"
            ),
            "objective_denominator": (
                "globally_reduced_masked_target_entry_count_per_update"
            ),
            "max_epochs": 300,
            "minimum_global_epochs": 50,
            "scheduler": "reduce_lr_on_plateau",
            "scheduler_monitor": PRIMARY_METRIC,
            "scheduler_factor": 0.5,
            "scheduler_patience": 8,
            "scheduler_threshold_mode": "abs",
            "scheduler_threshold": 1e-4,
            "scheduler_min_learning_rate": 1e-6,
            "early_stopping": True,
            "early_stopping_monitor": PRIMARY_METRIC,
            "early_stopping_patience": 25,
            "early_stopping_min_delta": 1e-4,
            "validation_every": 1,
            "amp": True,
            "restore_best": True,
            "distributed_world_size": 4,
            "cores_staged_sequentially_per_optimizer_update": True,
            "ddp_sync_policy": (
                "first_core_backward_under_no_sync_second_core_backward_synced"
            ),
            "stage_complete_core_graph_on_device": False,
            "staged_relative_geometry_dtype": "float32",
            "core_order_seed": base.CORE_ORDER_SEED,
        },
        "launcher": {
            "process_count": 4,
            "requested_gpu_count": 4,
            "require_exact_visible_devices": "0,1,2,3",
            "distributed_backend": "nccl",
            "elastic_max_restarts": 0,
            "hardware_preflight_receipt": (
                "state/preflight/"
                "so2_target_isolated_binary_expression_bce_once_per_cell_ddp4.json"
            ),
            "disk_safety_min_free_gb": 20,
        },
    }
    for section_name, fields in expected_sections.items():
        section = base._section(config, section_name)
        for field, expected in fields.items():
            base._require_equal(
                section.get(field), expected, field=f"{section_name}.{field}"
            )
    validation_masks = base._section(config, "masking").get("validation_masks")
    if not isinstance(validation_masks, Mapping):
        raise SO2NBTrainingError("masking.validation_masks must be a mapping.")
    base._require_equal(
        validation_masks.get("seed_namespace"),
        base.VALIDATION_MASK_NAMESPACE,
        field="masking.validation_masks.seed_namespace",
    )


def _training_nonzero_prevalence(bundle: SO2NBDataBundle) -> torch.Tensor:
    positive = torch.zeros(1000, dtype=torch.int64)
    cells = 0
    for batch in bundle.training_batches:
        positive.add_((batch.raw_count_target > 0).sum(dim=0, dtype=torch.int64))
        cells += batch.n_nodes
    if cells != 208_696 or bool((positive < 0).any()) or bool((positive > cells).any()):
        raise SO2NBTrainingError("Training-only nonzero prevalence support drifted.")
    probability = (positive.to(torch.float64) + 0.5) / float(cells + 1)
    probability.clamp_(
        DETECTION_PROBABILITY_EPSILON,
        1.0 - DETECTION_PROBABILITY_EPSILON,
    )
    probability = probability.to(torch.float32)
    digest = hashlib.sha256(
        probability.contiguous().numpy().tobytes(order="C")
    ).hexdigest()
    global _DETECTION_INITIALIZATION_RECEIPT
    _DETECTION_INITIALIZATION_RECEIPT = {
        "method": "jeffreys_smoothed_training_only_per_gene_nonzero_prevalence",
        "formula": "(training_positive_count + 0.5) / (training_cells + 1)",
        "training_cells": cells,
        "gene_count": int(probability.numel()),
        "clamp_epsilon": DETECTION_PROBABILITY_EPSILON,
        "minimum": float(probability.min()),
        "median": float(probability.median()),
        "maximum": float(probability.max()),
        "mean": float(probability.mean()),
        "float32_values_sha256": digest,
        "validation_targets_used": False,
    }
    return probability


def _build_model(
    config: Mapping[str, Any], bundle: SO2NBDataBundle
) -> TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer:
    model_config = base._section(config, "model")
    trainer = base._section(config, "trainer")
    set_deterministic_seed(
        base.MODEL_SEED,
        deterministic=bool(trainer["deterministic"]),
        warn_only=bool(trainer["deterministic_warn_only"]),
    )
    first = bundle.training_batches[0]
    model = TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer(
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
        detection_initial_nonzero_probability=_training_nonzero_prevalence(bundle),
        detection_probability_epsilon=DETECTION_PROBABILITY_EPSILON,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    base._require_equal(count, EXPECTED_PARAMETER_COUNT, field="model.parameter_count")
    blocks = tuple(model.blocks)
    if len(blocks) != 4 or len({id(block) for block in blocks}) != 4:
        raise SO2NBTrainingError("Model must expose four unique graph blocks.")
    parameter_ids = [id(parameter) for block in blocks for parameter in block.parameters()]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise SO2NBTrainingError("Graph-block parameter sets must be disjoint.")
    if hasattr(model, "raw_theta") or any("theta" in key for key in model.state_dict()):
        raise SO2NBTrainingError("Binary-expression model retained count dispersion.")
    base._require_equal(
        sum(parameter.numel() for parameter in model.detection_head.parameters()),
        DETECTION_HEAD_PARAMETER_COUNT,
        field="model.detection_head_parameter_count",
    )
    return model


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
) -> BinaryExpressionModelOutput:
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
    if not isinstance(output, BinaryExpressionModelOutput):
        raise SO2NBTrainingError("Target-isolated model did not return binary output.")
    if output.nonzero_logits.shape != mask.shape:
        raise SO2NBTrainingError("Target-only binary output is not shard-aligned.")
    return output


def _paired_update(
    *,
    model: TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer,
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
    global_tracker: Any,
    block_tracker: Any,
    theta_tracker: Any,
    theta_view: nn.Module,
) -> tuple[Mapping[str, tuple[float, int, int, int]], tuple[Mapping[str, Any], ...], float]:
    """Execute one exactly pooled, two-core binary BCE update on every rank."""

    training_model.train()
    optimizer.zero_grad(set_to_none=True)
    plans: list[tuple[SO2NBCoreBatch, Any, Any]] = []
    local_count = 0
    for alias in pair:
        batch = batches[alias]
        completed_epoch = global_epoch + 1
        partition, realization = base._partition_and_mask(
            batch, rank=rank, global_epoch=completed_epoch
        )
        plans.append((batch, partition, realization))
        local_count += int(realization.n_masked_entries)
    global_count_tensor = torch.tensor(local_count, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(global_count_tensor, op=torch.distributed.ReduceOp.SUM)
    global_count = int(global_count_tensor.item())
    if global_count <= 0:
        raise SO2NBTrainingError("Paired update has no globally masked entries.")

    local_stats: dict[str, tuple[float, int, int, int]] = {}
    coverage: list[Mapping[str, Any]] = []
    try:
        for position, (batch, partition, realization) in enumerate(plans):
            base._clear_graph_layout_caches(training_model)
            clean_input, raw_target, covariates = base._stage_core(batch, device=device)
            target_input, mask = base._make_target_tensors(
                clean_input=clean_input,
                realization=realization,
                target_start=partition.start,
                target_stop=partition.stop,
                device=device,
            )
            base._seed_model_step(
                base._model_step_seed(completed_epoch, update_index, batch.alias, rank),
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
                sums = masked_binary_expression_bce_sums(
                    output.nonzero_logits,
                    raw_target[partition.start : partition.stop],
                    mask,
                )
                scaled_loss = sums.detection_nll_sum * (
                    base.WORLD_SIZE / float(global_count)
                )
                scaler.scale(scaled_loss).backward()
            local_stats[batch.alias] = (
                float(sums.detection_nll_sum.detach().double().cpu()),
                int(sums.masked_entries),
                int(sums.positive_entries),
                int(realization.n_targets),
            )
            gathered = base._all_gather_objects(
                base._local_receipt_record(
                    rank=rank,
                    batch=batch,
                    partition=partition,
                    realization=realization,
                ),
                control_group=control_group,
            )
            coverage.append(
                base._coverage_receipt(
                    batch=batch,
                    role="training",
                    global_epoch=completed_epoch,
                    gathered=gathered,
                )
            )
            del output, sums, scaled_loss, target_input, mask
            del clean_input, raw_target, covariates
            base._clear_graph_layout_caches(training_model)
            torch.cuda.empty_cache()

        scaler.unscale_(optimizer)
        base._assert_finite_gradients(model)
        context = base.GradientDirectionUpdateContext(
            global_epoch=global_epoch,
            completed_global_epoch=global_epoch + 1,
            optimizer_update_in_epoch=update_index,
            cumulative_optimizer_update=(
                global_epoch * base.OPTIMIZER_UPDATES_PER_EPOCH + update_index + 1
            ),
            aliases=pair,
        )
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
        base._clear_graph_layout_caches(training_model)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _confusion_metrics(
    *, tp: int, tn: int, fp: int, fn: int
) -> Mapping[str, float | int]:
    support = tp + tn + fp + fn
    if support <= 0 or min(tp, tn, fp, fn) < 0:
        raise SO2NBTrainingError("Detection confusion support is invalid.")
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = _safe_ratio(float(tp * tn - fp * fn), denominator)
    return {
        "detection_true_positive": tp,
        "detection_true_negative": tn,
        "detection_false_positive": fp,
        "detection_false_negative": fn,
        "detection_accuracy": (tp + tn) / support,
        "detection_balanced_accuracy": 0.5 * (recall + specificity),
        "detection_recall": recall,
        "detection_specificity": specificity,
        "detection_precision": precision,
        "detection_f1": f1,
        "detection_mcc": mcc,
        "predicted_positive_class_rate": (tp + fp) / support,
    }


def _histogram_auc_metrics(
    positive_histogram: torch.Tensor,
    negative_histogram: torch.Tensor,
    probability_sum_histogram: torch.Tensor,
) -> tuple[float, float, float]:
    """Return deterministic 4096-bin AUROC, AP, and calibration error."""

    expected = (HISTOGRAM_BIN_COUNT,)
    if (
        positive_histogram.shape != expected
        or negative_histogram.shape != expected
        or probability_sum_histogram.shape != expected
    ):
        raise SO2NBTrainingError("Detection histogram shape drifted.")
    positive = positive_histogram.detach().to(dtype=torch.float64, device="cpu")
    negative = negative_histogram.detach().to(dtype=torch.float64, device="cpu")
    probability_sum = probability_sum_histogram.detach().to(
        dtype=torch.float64, device="cpu"
    )
    if (
        not bool(torch.isfinite(positive).all())
        or not bool(torch.isfinite(negative).all())
        or bool((positive < 0).any())
        or bool((negative < 0).any())
        or not bool(torch.isfinite(probability_sum).all())
        or bool((probability_sum < 0).any())
    ):
        raise FloatingPointError("Detection histograms are invalid.")
    total_positive = float(positive.sum())
    total_negative = float(negative.sum())
    if total_positive <= 0.0 or total_negative <= 0.0:
        raise SO2NBTrainingError("AUROC/AUPRC require both binary classes.")
    true_positive = torch.flip(positive, dims=(0,)).cumsum(dim=0)
    false_positive = torch.flip(negative, dims=(0,)).cumsum(dim=0)
    recall = true_positive / total_positive
    false_positive_rate = false_positive / total_negative
    zero = torch.zeros(1, dtype=torch.float64)
    auroc = float(
        torch.trapezoid(
            torch.cat((zero, recall)),
            torch.cat((zero, false_positive_rate)),
        )
    )
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall_increment = torch.diff(torch.cat((zero, recall)))
    auprc = float((recall_increment * precision).sum())
    count = positive + negative
    nonempty = count > 0.0
    empirical_positive = torch.zeros_like(count)
    mean_probability = torch.zeros_like(count)
    empirical_positive[nonempty] = positive[nonempty] / count[nonempty]
    mean_probability[nonempty] = probability_sum[nonempty] / count[nonempty]
    ece = float(
        (
            count[nonempty]
            * (empirical_positive[nonempty] - mean_probability[nonempty]).abs()
        ).sum()
        / count.sum()
    )
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (auroc, auprc, ece)
    ):
        raise FloatingPointError("Histogram AUROC/AUPRC/ECE is outside [0, 1].")
    return auroc, auprc, ece


@dataclass(frozen=True)
class ValidationResult:
    pooled_binary_cross_entropy: float
    observed_positive_rate: float
    predicted_positive_probability_mean: float
    predicted_positive_class_rate: float
    detection_brier_score: float
    detection_true_positive: int
    detection_true_negative: int
    detection_false_positive: int
    detection_false_negative: int
    detection_accuracy: float
    detection_balanced_accuracy: float
    detection_recall: float
    detection_specificity: float
    detection_precision: float
    detection_f1: float
    detection_mcc: float
    detection_auroc_histogram_4096: float
    detection_auprc_histogram_4096: float
    detection_expected_calibration_error_histogram_4096: float
    operating_threshold_metrics: Mapping[str, Mapping[str, float | int]]
    masked_entries: int
    positive_entries: int
    target_cells: int
    positive_histogram_4096: tuple[int, ...]
    negative_histogram_4096: tuple[int, ...]
    probability_sum_histogram_4096: tuple[float, ...]
    per_core: Mapping[str, Mapping[str, float | int]]
    coverage_receipts: Mapping[str, Mapping[str, Any]]

    @property
    def pooled_negative_binomial_nll(self) -> float:
        """Compatibility name consumed only by the shared control-plane loop."""

        return self.pooled_binary_cross_entropy


def _validation_sum_tensor(
    *,
    output: BinaryExpressionModelOutput,
    raw_target: torch.Tensor,
    mask: torch.Tensor,
    target_cells: int,
) -> torch.Tensor:
    sums = masked_binary_expression_bce_sums(
        output.nonzero_logits,
        raw_target,
        mask,
    )
    n = int(sums.masked_entries)
    positives = int(sums.positive_entries)
    if n != int(mask.sum().item()) or n <= 0 or positives <= 0 or positives >= n:
        raise SO2NBTrainingError("Validation binary support drifted or lacks a class.")
    observed_positive = raw_target.masked_select(mask) > 0
    probability = output.nonzero_probability.masked_select(mask).float()
    binary = observed_positive.float()
    predicted_class = probability >= DETECTION_THRESHOLD
    scalar_values = [
        float(n),
        float(positives),
        float(sums.detection_nll_sum.detach().double().cpu()),
        float(probability.sum(dtype=torch.float64).cpu()),
        float((probability - binary).square().sum(dtype=torch.float64).cpu()),
        float((predicted_class & observed_positive).sum().cpu()),
        float(((~predicted_class) & (~observed_positive)).sum().cpu()),
        float((predicted_class & (~observed_positive)).sum().cpu()),
        float(((~predicted_class) & observed_positive).sum().cpu()),
        float(target_cells),
    ]
    for threshold in OPERATING_THRESHOLDS:
        predicted = probability >= threshold
        scalar_values.extend(
            (
                float((predicted & observed_positive).sum().cpu()),
                float(((~predicted) & (~observed_positive)).sum().cpu()),
                float((predicted & (~observed_positive)).sum().cpu()),
                float(((~predicted) & observed_positive).sum().cpu()),
            )
        )
    bin_index = torch.floor(probability * HISTOGRAM_BIN_COUNT).to(torch.long)
    bin_index.clamp_(min=0, max=HISTOGRAM_BIN_COUNT - 1)
    positive_histogram = torch.bincount(
        bin_index[observed_positive], minlength=HISTOGRAM_BIN_COUNT
    ).to(dtype=torch.float64)
    negative_histogram = torch.bincount(
        bin_index[~observed_positive], minlength=HISTOGRAM_BIN_COUNT
    ).to(dtype=torch.float64)
    probability_sum_histogram = torch.zeros(
        HISTOGRAM_BIN_COUNT,
        dtype=torch.float64,
        device=probability.device,
    )
    probability_sum_histogram.scatter_add_(
        0,
        bin_index,
        probability.to(dtype=torch.float64),
    )
    scalars = torch.tensor(scalar_values, dtype=torch.float64, device=probability.device)
    tensor = torch.cat(
        (
            scalars,
            positive_histogram,
            negative_histogram,
            probability_sum_histogram,
        )
    )
    if tensor.shape != (_VALIDATION_VECTOR_LENGTH,):
        raise RuntimeError("Validation sufficient-statistic length drifted.")
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()):
        raise FloatingPointError("Validation binary additive statistics are invalid.")
    return tensor


def _metrics_from_sums(values: torch.Tensor) -> Mapping[str, Any]:
    if values.shape != (_VALIDATION_VECTOR_LENGTH,):
        raise SO2NBTrainingError("Validation sufficient-statistic shape drifted.")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise FloatingPointError("Reduced validation statistics are invalid.")
    scalars = values[:_SCALAR_COUNT].detach().to(dtype=torch.float64, device="cpu")
    row = {
        field: float(scalars[index])
        for index, field in enumerate(_SCALAR_FIELDS + _THRESHOLD_COUNT_FIELDS)
    }
    integral_fields = (
        "masked_entries",
        "positive_entries",
        "detection_true_positive",
        "detection_true_negative",
        "detection_false_positive",
        "detection_false_negative",
        "target_cells",
        *_THRESHOLD_COUNT_FIELDS,
    )
    if any(
        not math.isclose(row[field], round(row[field]), rel_tol=0.0, abs_tol=0.0)
        for field in integral_fields
    ):
        raise SO2NBTrainingError("Reduced validation count is not integral.")
    n = int(round(row["masked_entries"]))
    positives = int(round(row["positive_entries"]))
    targets = int(round(row["target_cells"]))
    if n <= 0 or positives <= 0 or positives >= n or targets <= 0:
        raise SO2NBTrainingError("Reduced validation support is invalid.")
    tp = int(round(row["detection_true_positive"]))
    tn = int(round(row["detection_true_negative"]))
    fp = int(round(row["detection_false_positive"]))
    fn = int(round(row["detection_false_negative"]))
    confusion = _confusion_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    if tp + tn + fp + fn != n or tp + fn != positives:
        raise SO2NBTrainingError("Detection confusion counts do not match support.")
    positive_histogram = values[_POSITIVE_HISTOGRAM_SLICE]
    negative_histogram = values[_NEGATIVE_HISTOGRAM_SLICE]
    probability_sum_histogram = values[_PROBABILITY_SUM_HISTOGRAM_SLICE]
    if int(round(float(positive_histogram.sum()))) != positives:
        raise SO2NBTrainingError("Positive histogram support drifted.")
    if int(round(float(negative_histogram.sum()))) != n - positives:
        raise SO2NBTrainingError("Negative histogram support drifted.")
    histogram_count = positive_histogram + negative_histogram
    if bool((probability_sum_histogram > histogram_count + 1e-7).any()):
        raise SO2NBTrainingError("Probability histogram exceeds bin support.")
    auroc, auprc, ece = _histogram_auc_metrics(
        positive_histogram,
        negative_histogram,
        probability_sum_histogram,
    )
    threshold_metrics: dict[str, Mapping[str, float | int]] = {}
    cursor = len(_SCALAR_FIELDS)
    for threshold in OPERATING_THRESHOLDS:
        threshold_tp, threshold_tn, threshold_fp, threshold_fn = (
            int(round(float(scalars[cursor + offset]))) for offset in range(4)
        )
        cursor += 4
        metrics = dict(
            _confusion_metrics(
                tp=threshold_tp,
                tn=threshold_tn,
                fp=threshold_fp,
                fn=threshold_fn,
            )
        )
        metrics["threshold"] = threshold
        threshold_metrics[f"{threshold:.2f}"] = metrics
    return {
        "target_cells": targets,
        "masked_entries": n,
        "positive_entries": positives,
        "pooled_masked_binary_cross_entropy": row["detection_nll_sum"] / n,
        "observed_positive_rate": positives / n,
        "predicted_positive_probability_mean": (
            row["predicted_positive_probability_sum"] / n
        ),
        "detection_brier_score": row["detection_brier_sum"] / n,
        **confusion,
        "detection_auroc_histogram_4096": auroc,
        "detection_auprc_histogram_4096": auprc,
        "detection_expected_calibration_error_histogram_4096": ece,
        "operating_threshold_metrics": threshold_metrics,
        "positive_histogram_4096": tuple(
            int(round(float(value))) for value in positive_histogram
        ),
        "negative_histogram_4096": tuple(
            int(round(float(value))) for value in negative_histogram
        ),
        "probability_sum_histogram_4096": tuple(
            float(value) for value in probability_sum_histogram
        ),
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
            for alias in base.VALIDATION_ALIASES:
                batch = bundle.batches_by_alias[alias]
                partition, realization = base._partition_and_mask(
                    batch, rank=rank, global_epoch=None
                )
                base._clear_graph_layout_caches(training_model)
                clean_input, raw_target, covariates = base._stage_core(
                    batch, device=device
                )
                target_input, mask = base._make_target_tensors(
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
                base._require_equal(
                    int(core_metrics["target_cells"]),
                    batch.n_nodes,
                    field=f"validation.{alias}.target_cells",
                )
                per_core[alias] = {
                    key: value
                    for key, value in core_metrics.items()
                    if key
                    not in {
                        "operating_threshold_metrics",
                        "positive_histogram_4096",
                        "negative_histogram_4096",
                        "probability_sum_histogram_4096",
                    }
                }
                per_core_tensors.append(global_sums)
                gathered = base._all_gather_objects(
                    base._local_receipt_record(
                        rank=rank,
                        batch=batch,
                        partition=partition,
                        realization=realization,
                    ),
                    control_group=control_group,
                )
                receipts[alias] = base._coverage_receipt(
                    batch=batch,
                    role="validation",
                    global_epoch=None,
                    gathered=gathered,
                )
                del output, local_sums, target_input, mask
                del clean_input, raw_target, covariates
                base._clear_graph_layout_caches(training_model)
                torch.cuda.empty_cache()
        pooled = _metrics_from_sums(torch.stack(per_core_tensors, dim=0).sum(dim=0))
        base._require_equal(
            int(pooled["target_cells"]), 37_367, field="validation.target_cells"
        )
        return ValidationResult(
            pooled_binary_cross_entropy=float(
                pooled["pooled_masked_binary_cross_entropy"]
            ),
            observed_positive_rate=float(pooled["observed_positive_rate"]),
            predicted_positive_probability_mean=float(
                pooled["predicted_positive_probability_mean"]
            ),
            predicted_positive_class_rate=float(
                pooled["predicted_positive_class_rate"]
            ),
            detection_brier_score=float(pooled["detection_brier_score"]),
            detection_true_positive=int(pooled["detection_true_positive"]),
            detection_true_negative=int(pooled["detection_true_negative"]),
            detection_false_positive=int(pooled["detection_false_positive"]),
            detection_false_negative=int(pooled["detection_false_negative"]),
            detection_accuracy=float(pooled["detection_accuracy"]),
            detection_balanced_accuracy=float(
                pooled["detection_balanced_accuracy"]
            ),
            detection_recall=float(pooled["detection_recall"]),
            detection_specificity=float(pooled["detection_specificity"]),
            detection_precision=float(pooled["detection_precision"]),
            detection_f1=float(pooled["detection_f1"]),
            detection_mcc=float(pooled["detection_mcc"]),
            detection_auroc_histogram_4096=float(
                pooled["detection_auroc_histogram_4096"]
            ),
            detection_auprc_histogram_4096=float(
                pooled["detection_auprc_histogram_4096"]
            ),
            detection_expected_calibration_error_histogram_4096=float(
                pooled[
                    "detection_expected_calibration_error_histogram_4096"
                ]
            ),
            operating_threshold_metrics=dict(pooled["operating_threshold_metrics"]),
            masked_entries=int(pooled["masked_entries"]),
            positive_entries=int(pooled["positive_entries"]),
            target_cells=int(pooled["target_cells"]),
            positive_histogram_4096=tuple(pooled["positive_histogram_4096"]),
            negative_histogram_4096=tuple(pooled["negative_histogram_4096"]),
            probability_sum_histogram_4096=tuple(
                pooled["probability_sum_histogram_4096"]
            ),
            per_core=dict(per_core),
            coverage_receipts=dict(receipts),
        )
    finally:
        training_model.train()
        base._clear_graph_layout_caches(training_model)


def _aggregate_training_epoch(
    local_by_alias: Mapping[str, tuple[float, int, int, int]],
    *,
    device: torch.device,
) -> tuple[float, int, int, Mapping[str, Mapping[str, float | int]]]:
    if set(local_by_alias) != set(base.TRAINING_ALIASES):
        raise SO2NBTrainingError("Training epoch did not visit every core exactly once.")
    transport = torch.zeros(
        (len(base.TRAINING_ALIASES), 4), dtype=torch.float64, device=device
    )
    for index, alias in enumerate(base.TRAINING_ALIASES):
        transport[index] = torch.tensor(
            local_by_alias[alias], dtype=torch.float64, device=device
        )
    torch.distributed.all_reduce(transport, op=torch.distributed.ReduceOp.SUM)
    values = transport.detach().cpu()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise FloatingPointError("Reduced binary training statistics are invalid.")
    per_core: dict[str, Mapping[str, float | int]] = {}
    for index, alias in enumerate(base.TRAINING_ALIASES):
        loss_sum = float(values[index, 0])
        count, positives, targets = (
            int(round(float(values[index, column]))) for column in range(1, 4)
        )
        if count < targets or positives <= 0 or positives >= count or targets <= 0:
            raise SO2NBTrainingError(f"Training binary support is invalid for {alias}.")
        per_core[alias] = {
            "target_cells": targets,
            "masked_entries": count,
            "positive_entries": positives,
            "detection_nll_sum": loss_sum,
            "pooled_masked_binary_cross_entropy": loss_sum / count,
            "observed_positive_rate": positives / count,
        }
    total_count = sum(int(row["masked_entries"]) for row in per_core.values())
    total_targets = sum(int(row["target_cells"]) for row in per_core.values())
    if total_targets != 208_696:
        raise SO2NBTrainingError("Training target coverage is not exactly 208,696.")
    total_loss = sum(float(row["detection_nll_sum"]) for row in per_core.values())
    return total_loss / total_count, total_count, total_targets, per_core


def _make_epoch_row(
    *,
    completed_epoch: int,
    train_nll: float,
    train_masked_entries: int,
    train_target_cells: int,
    training: Mapping[str, Mapping[str, float | int]],
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
    train_positives = sum(int(row["positive_entries"]) for row in training.values())
    return {
        "schema": EPOCH_METRICS_SCHEMA,
        "global_epoch": completed_epoch,
        "train_pooled_masked_binary_cross_entropy": train_nll,
        "train_masked_entries": train_masked_entries,
        "train_positive_entries": train_positives,
        "train_observed_positive_rate": train_positives / train_masked_entries,
        "train_target_cells": train_target_cells,
        "validation_pooled_masked_binary_cross_entropy": (
            validation.pooled_binary_cross_entropy
        ),
        "validation_masked_entries": validation.masked_entries,
        "validation_positive_entries": validation.positive_entries,
        "validation_target_cells": validation.target_cells,
        "validation_observed_positive_rate": validation.observed_positive_rate,
        "validation_predicted_positive_probability_mean": (
            validation.predicted_positive_probability_mean
        ),
        "validation_predicted_positive_class_rate": (
            validation.predicted_positive_class_rate
        ),
        "validation_detection_brier_score": validation.detection_brier_score,
        "validation_detection_threshold": DETECTION_THRESHOLD,
        "validation_detection_true_positive": validation.detection_true_positive,
        "validation_detection_true_negative": validation.detection_true_negative,
        "validation_detection_false_positive": validation.detection_false_positive,
        "validation_detection_false_negative": validation.detection_false_negative,
        "validation_detection_accuracy": validation.detection_accuracy,
        "validation_detection_balanced_accuracy": (
            validation.detection_balanced_accuracy
        ),
        "validation_detection_recall": validation.detection_recall,
        "validation_detection_specificity": validation.detection_specificity,
        "validation_detection_precision": validation.detection_precision,
        "validation_detection_f1": validation.detection_f1,
        "validation_detection_mcc": validation.detection_mcc,
        "validation_detection_auroc_histogram_4096": (
            validation.detection_auroc_histogram_4096
        ),
        "validation_detection_auprc_histogram_4096": (
            validation.detection_auprc_histogram_4096
        ),
        "validation_detection_expected_calibration_error_histogram_4096": (
            validation.detection_expected_calibration_error_histogram_4096
        ),
        "validation_operating_threshold_metrics_json": base._json_field(
            validation.operating_threshold_metrics
        ),
        "detection_head_bias_min": theta[0],
        "detection_head_bias_median": theta[1],
        "detection_head_bias_max": theta[2],
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
        "core_order_json": base._json_field(list(core_order)),
        "training_coverage_receipts_json": base._json_field(training_receipts),
        "validation_coverage_receipts_json": base._json_field(
            validation.coverage_receipts
        ),
        "global_gradient_json": base._json_field(global_gradient),
        "block_gradients_json": base._json_field(list(block_gradients)),
        "detection_head_gradient_json": base._json_field(theta_gradient),
    }


def _per_core_rows(
    *,
    completed_epoch: int,
    training: Mapping[str, Mapping[str, float | int]],
    training_receipts: Mapping[str, Mapping[str, Any]],
    validation: ValidationResult,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    blank_validation_fields = {
        "predicted_positive_probability_mean": "",
        "predicted_positive_class_rate": "",
        "detection_brier_score": "",
        "detection_true_positive": "",
        "detection_true_negative": "",
        "detection_false_positive": "",
        "detection_false_negative": "",
        "detection_accuracy": "",
        "detection_balanced_accuracy": "",
        "detection_recall": "",
        "detection_specificity": "",
        "detection_precision": "",
        "detection_f1": "",
        "detection_mcc": "",
        "detection_auroc_histogram_4096": "",
        "detection_auprc_histogram_4096": "",
        "detection_expected_calibration_error_histogram_4096": "",
    }
    for alias in base.TRAINING_ALIASES:
        metrics = training[alias]
        rows.append(
            {
                "schema": PER_CORE_METRICS_SCHEMA,
                "global_epoch": completed_epoch,
                "split": "training",
                "core_alias": alias,
                "target_cells": metrics["target_cells"],
                "masked_entries": metrics["masked_entries"],
                "positive_entries": metrics["positive_entries"],
                "pooled_masked_binary_cross_entropy": metrics[
                    "pooled_masked_binary_cross_entropy"
                ],
                "observed_positive_rate": metrics["observed_positive_rate"],
                **blank_validation_fields,
                "coverage_receipt_sha256": training_receipts[alias][
                    "receipt_sha256"
                ],
            }
        )
    for alias in base.VALIDATION_ALIASES:
        metrics = validation.per_core[alias]
        rows.append(
            {
                "schema": PER_CORE_METRICS_SCHEMA,
                "global_epoch": completed_epoch,
                "split": "validation",
                "core_alias": alias,
                **{
                    field: metrics[field]
                    for field in PER_CORE_COLUMNS
                    if field in metrics
                },
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
        "train_positive_entries",
        "train_target_cells",
        "validation_masked_entries",
        "validation_positive_entries",
        "validation_target_cells",
        "validation_detection_true_positive",
        "validation_detection_true_negative",
        "validation_detection_false_positive",
        "validation_detection_false_negative",
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
    dataset_id = str(base._section(config, "dataset")["dataset_id"])
    rows: list[Mapping[str, Any]] = []
    for alias in base.VALIDATION_ALIASES:
        metrics = validation.per_core[alias]
        batch = bundle.batches_by_alias[alias]
        sample_key = hashlib.sha256(
            (
                "bagm.so2.target-isolated.binary-expression-bce."
                f"validation-summary.v1:{alias}"
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            {
                "run_id": run_id,
                "sample_key": sample_key,
                "dataset_id": dataset_id,
                "split": "validation",
                "y_true": float(metrics["observed_positive_rate"]),
                "y_pred": float(metrics["predicted_positive_probability_mean"]),
                "sample_loss": float(
                    metrics["pooled_masked_binary_cross_entropy"]
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
    return base.validate_prediction_rows(
        rows, expected_run_id=run_id, expected_split="validation"
    )


def _final_metrics(validation: ValidationResult) -> Mapping[str, float | int]:
    return {
        PRIMARY_METRIC: validation.pooled_binary_cross_entropy,
        "val/unseen_donor/observed_positive_rate": validation.observed_positive_rate,
        "val/unseen_donor/predicted_positive_probability_mean": (
            validation.predicted_positive_probability_mean
        ),
        "val/unseen_donor/predicted_positive_class_rate": (
            validation.predicted_positive_class_rate
        ),
        "val/unseen_donor/detection_brier_score": validation.detection_brier_score,
        "val/unseen_donor/detection_threshold": DETECTION_THRESHOLD,
        "val/unseen_donor/detection_true_positive": validation.detection_true_positive,
        "val/unseen_donor/detection_true_negative": validation.detection_true_negative,
        "val/unseen_donor/detection_false_positive": validation.detection_false_positive,
        "val/unseen_donor/detection_false_negative": validation.detection_false_negative,
        "val/unseen_donor/detection_accuracy": validation.detection_accuracy,
        "val/unseen_donor/detection_balanced_accuracy": (
            validation.detection_balanced_accuracy
        ),
        "val/unseen_donor/detection_recall": validation.detection_recall,
        "val/unseen_donor/detection_specificity": validation.detection_specificity,
        "val/unseen_donor/detection_precision": validation.detection_precision,
        "val/unseen_donor/detection_f1": validation.detection_f1,
        "val/unseen_donor/detection_matthews_correlation_coefficient": (
            validation.detection_mcc
        ),
        "val/unseen_donor/detection_auroc_histogram_4096": (
            validation.detection_auroc_histogram_4096
        ),
        "val/unseen_donor/detection_auprc_histogram_4096": (
            validation.detection_auprc_histogram_4096
        ),
        "val/unseen_donor/detection_expected_calibration_error_histogram_4096": (
            validation.detection_expected_calibration_error_histogram_4096
        ),
        "val/masked_entries": validation.masked_entries,
        "val/positive_entries": validation.positive_entries,
        "val/target_cells": validation.target_cells,
    }


def _additional_provenance(validation: ValidationResult) -> Mapping[str, Any]:
    prevalence = validation.observed_positive_rate
    return {
        "observation_model": "bernoulli_binary_expression",
        "objective": "unweighted_binary_cross_entropy_all_masked_entries",
        "binary_label": "one_if_raw_count_strictly_positive_else_zero",
        "count_mean_or_dispersion_learned": False,
        "visible_input_scale": "train_standardized_log1p_raw_counts",
        "detection_threshold": DETECTION_THRESHOLD,
        "operating_thresholds": list(OPERATING_THRESHOLDS),
        "operating_threshold_metrics": dict(
            validation.operating_threshold_metrics
        ),
        "histogram_curve_approximation": {
            "bins": HISTOGRAM_BIN_COUNT,
            "binning": "floor(probability * bins)_clamped",
            "auroc_tie_policy": "trapezoidal_within_bin",
            "auprc_definition": "descending_threshold_average_precision",
            "prediction_vectors_persisted": False,
            "final_numeric_artifact": DETECTION_CURVE_RELATIVE_PATH,
        },
        "constant_prevalence_baselines": {
            "bce": (
                -prevalence * math.log(prevalence)
                - (1.0 - prevalence) * math.log(1.0 - prevalence)
            ),
            "brier": prevalence * (1.0 - prevalence),
            "accuracy_all_negative": 1.0 - prevalence,
            "balanced_accuracy_all_negative": 0.5,
            "auroc": 0.5,
            "auprc": prevalence,
        },
        "detection_initialization": dict(_DETECTION_INITIALIZATION_RECEIPT),
        "validation_positive_entries": validation.positive_entries,
    }


def _write_additional_final_artifacts(
    archive: Any,
    validation: ValidationResult,
) -> None:
    """Persist the compact final histogram underlying AUROC/AUPRC/ECE."""

    archive.write_json(
        DETECTION_CURVE_RELATIVE_PATH,
        {
            "schema": "binary_expression_detection_curve_histogram_v1",
            "bin_count": HISTOGRAM_BIN_COUNT,
            "bin_width": 1.0 / HISTOGRAM_BIN_COUNT,
            "bin_index_definition": (
                "floor(nonzero_probability * bin_count), clamped to "
                "[0, bin_count - 1]"
            ),
            "positive_counts_by_ascending_probability_bin": list(
                validation.positive_histogram_4096
            ),
            "negative_counts_by_ascending_probability_bin": list(
                validation.negative_histogram_4096
            ),
            "probability_sums_by_ascending_probability_bin": list(
                validation.probability_sum_histogram_4096
            ),
            "auroc_histogram_4096": (
                validation.detection_auroc_histogram_4096
            ),
            "auprc_histogram_4096": (
                validation.detection_auprc_histogram_4096
            ),
            "expected_calibration_error_histogram_4096": (
                validation.detection_expected_calibration_error_histogram_4096
            ),
            "masked_entries": validation.masked_entries,
            "positive_entries": validation.positive_entries,
            "per_entry_predictions_persisted": False,
        },
    )


def _detection_head_summary(model: nn.Module) -> tuple[float, float, float]:
    head = getattr(model, "detection_head", None)
    if not isinstance(head, nn.Linear) or head.bias is None:
        raise SO2NBTrainingError("Binary model lacks its gene-wise detection head.")
    bias = head.bias.detach().float()
    if not bool(torch.isfinite(bias).all()):
        raise FloatingPointError("Detection-head bias is non-finite.")
    return (
        float(bias.min().cpu()),
        float(bias.median().cpu()),
        float(bias.max().cpu()),
    )


def _make_tracked_head_view(model: nn.Module) -> nn.Module:
    head = getattr(model, "detection_head", None)
    if not isinstance(head, nn.Linear):
        raise SO2NBTrainingError("Binary model lacks its tracked detection head.")
    return head


def _install_binary_expression_contract() -> None:
    constants = {
        "CAMPAIGN_ID": CAMPAIGN_ID,
        "MODEL_NAME": MODEL_NAME,
        "MODEL_FAMILY": MODEL_FAMILY,
        "EVALUATION_PROTOCOL": EVALUATION_PROTOCOL,
        "PROTOCOL": PROTOCOL,
        "PREFLIGHT_SCHEMA": PREFLIGHT_SCHEMA,
        "CHECKPOINT_SCHEMA": CHECKPOINT_SCHEMA,
        "EPOCH_METRICS_SCHEMA": EPOCH_METRICS_SCHEMA,
        "PER_CORE_METRICS_SCHEMA": PER_CORE_METRICS_SCHEMA,
        "EXPECTED_PARAMETER_COUNT": EXPECTED_PARAMETER_COUNT,
        "FROZEN_TASK_CONTRACT_SHA256": FROZEN_TASK_CONTRACT_SHA256,
        "SOURCE_EXPERIMENT_CONFIG": SOURCE_EXPERIMENT_CONFIG,
        "FROZEN_TASK_CONTRACT": FROZEN_TASK_CONTRACT,
        "PREFLIGHT_CODE_HASH_SCOPE": PREFLIGHT_CODE_HASH_SCOPE,
        "PREFLIGHT_CODE_RELATIVE_PATHS": PREFLIGHT_CODE_RELATIVE_PATHS,
        "PREFLIGHT_REQUIRED_GATES": PREFLIGHT_REQUIRED_GATES,
        "PREFLIGHT_TRACKED_HEAD_COUNT_FIELD": (
            PREFLIGHT_TRACKED_HEAD_COUNT_FIELD
        ),
        "PREFLIGHT_TRACKED_HEAD_PARAMETER_COUNT": (
            PREFLIGHT_TRACKED_HEAD_PARAMETER_COUNT
        ),
        "TRACKED_HEAD_GRADIENT_RELATIVE_PATH": (
            TRACKED_HEAD_GRADIENT_RELATIVE_PATH
        ),
        "PRIMARY_METRIC": PRIMARY_METRIC,
        "TRAIN_PRIMARY_METRIC": TRAIN_PRIMARY_METRIC,
        "PROVENANCE_RELATIVE_PATH": PROVENANCE_RELATIVE_PATH,
        "OBJECTIVE_AGGREGATION": OBJECTIVE_AGGREGATION,
        "ADDITIONAL_REQUIRED_FINAL_OUTPUTS": ADDITIONAL_REQUIRED_FINAL_OUTPUTS,
        "EPOCH_COLUMNS": EPOCH_COLUMNS,
        "PER_CORE_COLUMNS": PER_CORE_COLUMNS,
    }
    functions = {
        "validate_target_isolated_config": _require_config_values,
        "_build_model": _build_model,
        "_forward_target_shard": _forward_target_shard,
        "_paired_update": _paired_update,
        "ValidationResult": ValidationResult,
        "_validation_sum_tensor": _validation_sum_tensor,
        "_metrics_from_sums": _metrics_from_sums,
        "_evaluate_validation": _evaluate_validation,
        "_aggregate_training_epoch": _aggregate_training_epoch,
        "_make_epoch_row": _make_epoch_row,
        "_per_core_rows": _per_core_rows,
        "_archive_history_rows": _archive_history_rows,
        "_validation_prediction_rows": _validation_prediction_rows,
        "_final_metrics": _final_metrics,
        "_additional_provenance": _additional_provenance,
        "_write_additional_final_artifacts": _write_additional_final_artifacts,
        "_theta_summary": _detection_head_summary,
        "_make_tracked_head_view": _make_tracked_head_view,
    }
    for name, value in {**constants, **functions}.items():
        setattr(base, name, value)


_install_binary_expression_contract()


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
