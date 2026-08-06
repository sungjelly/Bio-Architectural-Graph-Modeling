#!/usr/bin/env python3
"""Run one shared-model ten-core hybrid-count campaign member.

The ten complete core graphs remain disconnected and CPU resident.  One graph
is staged on the selected device for each optimizer step.  This runner accepts
only worker-owned, checksum-bound configurations materialized for the frozen
pooled campaign.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT))
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from scripts.train.run_full_core_capacity import (  # noqa: E402
    CapacityRunResult,
    _build_graph,
    _fit_view,
    _mean_finite,
    _peak_host_memory_bytes,
    _section,
    _training_config,
    _verify_graph_identity,
    _worker_archive_and_config,
)
from scripts.train.run_hybrid_count_capacity import (  # noqa: E402
    _array_sha256,
    _finite_scalar,
    _paired_models,
    _reference_provenance,
    _replicate_metric_row,
)
from spatial_benchmark.full_core import (  # noqa: E402
    EDGE_ATTRIBUTE_NAMES,
    ReceiverSortedGraph,
)
from spatial_benchmark.hybrid_count import (  # noqa: E402
    tokenize_raw_counts,
    validate_raw_counts,
)
from spatial_benchmark.hybrid_count_metrics import (  # noqa: E402
    HybridCountReferences,
    fit_hybrid_count_references,
)
from spatial_benchmark.hybrid_count_training import (  # noqa: E402
    PrecisionEquivalenceResult,
    compare_fp32_amp_loss,
    evaluate_fixed_hybrid_count_mask,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.masking import (  # noqa: E402
    MaskSpec,
    create_fixed_mask_bundle,
)
from spatial_benchmark.pooled_full_core import (  # noqa: E402
    ANC_ALIASES,
    EXPECTED_N_GENES,
    EXPECTED_N_MODEL_COVARIATES,
    EXPECTED_TOTAL_NODES,
    PooledCoreData,
    PooledFullCoreCohort,
    load_pooled_full_core_cohort,
)
from spatial_benchmark.pooled_hybrid_count_training import (  # noqa: E402
    PooledCoreBatch,
    PooledHybridCountTrainingResult,
    fit_pooled_hybrid_count_model,
)
from spatial_benchmark.pooled_references import (  # noqa: E402
    fit_equal_core_hybrid_count_references,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    RunValidationError,
    deidentify_prediction_rows,
)
from spatial_benchmark.training import TrainingConfig  # noqa: E402


_CAMPAIGN_ID = (
    "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
)
_FROZEN_CONTRACT_SHA256 = (
    "c6af3dc756155ee502506f08304a7436ae99da36ad2b4ed8fae48672a312f6e2"
)
_FROZEN_CONTRACT_RELATIVE = (
    Path("experiments/campaigns")
    / _CAMPAIGN_ID
    / "frozen_task_contract.yaml"
)
_CAMPAIGN_RELATIVE = (
    Path("experiments/campaigns") / _CAMPAIGN_ID / "campaign.yaml"
)
_LOCKED_ROOT_RELATIVE = Path("scratch/locked_campaigns") / _CAMPAIGN_ID
_MATERIALIZATION_RECEIPT_RELATIVE = (
    _LOCKED_ROOT_RELATIVE / "locked_config_materialization.json"
)
_PILOT_RECEIPT_RELATIVE = _LOCKED_ROOT_RELATIVE / "pilot_gate_receipt.json"
_PILOT_RECEIPT_SCHEMA = "pooled_hybrid_count_pilot_gate_v1"
_PROTOCOL = "held_in_pooled_10core_fixed_budget"
_TASK_FAMILY = "masked_expression_hybrid_count"
_PRIMARY_METRIC = "fit/whole_node/hybrid_loss"
_REPRESENTATION_SCHEMA = (
    "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1"
)
_OBJECTIVE = (
    "equal_weight_balanced_detection_ordinal_positive_huber_within_core"
)
_EXPECTED_PARAMETER_COUNT = 11_674_880
_PUBLIC_VARIANTS = {
    "hybrid-count-gat": "pooled-hybrid-gat-k1000",
    "hybrid-count-matched-self": "pooled-hybrid-matched-self",
}
_MODEL_KEYS = {
    "hybrid-count-gat": "pooledhybridgatk1000",
    "hybrid-count-matched-self": "pooledhybridmatchedself",
}
_PUBLIC_MASK_NAMES = {
    "partial": "partial_gene",
    "node": "whole_node",
    "block": "spatial_block",
}
_REQUIRED_PUBLIC_MASKS = (
    "partial_gene",
    "whole_node",
    "spatial_block",
)
_EXPECTED_METADATA_FIELDS = (
    "Area",
    "Area.um2",
    "AspectRatio",
    "Width",
    "Height",
    "Mean.PanCK",
    "Max.PanCK",
    "Mean.G",
    "Max.G",
    "Mean.Membrane",
    "Max.Membrane",
    "Mean.CD45",
    "Max.CD45",
    "Mean.DAPI",
    "Max.DAPI",
    "SplitRatioToLocal",
    "NucArea",
    "NucAspectRatio",
    "Circularity",
    "Eccentricity",
    "Perimeter",
    "Solidity",
)
_FORBIDDEN_IDENTIFIER_KEYS = {
    "patient_id",
    "patient_identifier",
    "donor_id",
    "donor_identifier",
    "subject_id",
    "subject_identifier",
    "core_id",
    "core_identifier",
    "clinical_id",
    "clinical_identifier",
}
_ROW_METADATA_FIELDS = {
    "biological_unit_alias",
    "split",
    "mask_mode",
    "mask_replicate",
    "mask_entry_id",
    "mask_seed",
    "mask_checksum",
}
_PILOT_MAX_AMP_DISCREPANCY = 1e-3
_PILOT_MAX_VRAM_GIB = 20.5
_PILOT_MAX_HOST_GIB = 40.0
_PILOT_MAX_PROJECTED_HOURS = 6.0
_PILOT_MIN_PROJECTED_FREE_DISK_GIB = 27.5
_PROJECTED_CAMPAIGN_OUTPUT_BYTES = 8 * 1024**3
_PILOT_GATE_THRESHOLDS = {
    "fp32_amp_absolute_total_loss_discrepancy_each_core_maximum": (
        _PILOT_MAX_AMP_DISCREPANCY
    ),
    "peak_allocated_vram_gib_maximum": _PILOT_MAX_VRAM_GIB,
    "peak_host_memory_gib_per_process_maximum": _PILOT_MAX_HOST_GIB,
    "projected_200_epoch_gat_runtime_hours_maximum": (
        _PILOT_MAX_PROJECTED_HOURS
    ),
    "projected_final_free_disk_gib_minimum": (
        _PILOT_MIN_PROJECTED_FREE_DISK_GIB
    ),
}
_PILOT_JOB_REQUIRED_TRUE_FIELDS = (
    "verified_bundle",
    "checkpoint_verified",
    "finite_losses_and_gradients",
    "parameter_match",
    "paired_initialization_match",
    "precision_equivalence_passed",
    "peak_vram_passed",
    "peak_host_memory_passed",
    "projected_runtime_passed",
    "projected_disk_passed",
    "every_core_once_each_epoch",
    "all_20_optimizer_steps_completed",
    "runner_pilot_gate_passed",
)


class PooledHybridCountRunnerError(RuntimeError):
    """Raised before a pooled run can violate its frozen contract."""


@dataclass(frozen=True)
class PooledHybridContract:
    model_name: str
    model_key: str
    public_variant: str
    diagnostic_resource_pilot: bool
    uses_graph: bool
    seed: int


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise PooledHybridCountRunnerError(
            f"{field} must be {expected!r}; got {actual!r}"
        )


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _assert_alias_only_mapping(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_key(raw_key)
            child_path = f"{path}.{raw_key}"
            if key in _FORBIDDEN_IDENTIFIER_KEYS:
                raise PooledHybridCountRunnerError(
                    f"alias-only configuration prohibits {child_path}"
                )
            _assert_alias_only_mapping(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_alias_only_mapping(child, path=f"{path}[{index}]")


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PooledHybridCountRunnerError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _prepared_artifacts(dataset: Mapping[str, Any]) -> dict[str, Path]:
    records = dataset.get("prepared_artifacts")
    if not isinstance(records, Mapping) or tuple(records) != ANC_ALIASES:
        raise PooledHybridCountRunnerError(
            "dataset.prepared_artifacts must contain ANC-01 through ANC-10 "
            "in frozen order"
        )
    resolved: dict[str, Path] = {}
    for alias in ANC_ALIASES:
        expected = (
            "data/processed/adjacent_normal_10core_qkv_large_k_v1/"
            f"{alias.lower()}/prepared_v1"
        )
        _require_equal(
            records.get(alias),
            expected,
            field=f"dataset.prepared_artifacts.{alias}",
        )
        resolved[alias] = Path(expected)
    return resolved


def _graph_records(graph: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    records = graph.get("expected_core_graphs")
    if not isinstance(records, Mapping) or tuple(records) != ANC_ALIASES:
        raise PooledHybridCountRunnerError(
            "graph.expected_core_graphs must contain every ANC alias in "
            "frozen order"
        )
    for alias, record in records.items():
        if not isinstance(record, Mapping):
            raise PooledHybridCountRunnerError(
                f"graph.expected_core_graphs.{alias} must be a mapping"
            )
        checksum = record.get(
            "expected_materialized_graph_sha256",
            record.get("graph_sha256"),
        )
        edge_count = record.get(
            "expected_directed_edges",
            record.get("n_directed_edges"),
        )
        _require_sha256(
            checksum,
            field=f"graph.expected_core_graphs.{alias}.graph_sha256",
        )
        if (
            isinstance(edge_count, bool)
            or not isinstance(edge_count, int)
            or edge_count <= 0
        ):
            raise PooledHybridCountRunnerError(
                f"graph.expected_core_graphs.{alias}.directed_edges must "
                "be positive"
            )
    return records  # type: ignore[return-value]


def _prior_mask_records(
    evaluation: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    records = evaluation.get("prior_mask_sources")
    if not isinstance(records, Mapping) or tuple(records) != ANC_ALIASES:
        raise PooledHybridCountRunnerError(
            "evaluation.prior_mask_sources must contain every ANC "
            "alias in frozen order"
        )
    for alias, record in records.items():
        if not isinstance(record, Mapping):
            raise PooledHybridCountRunnerError(
                f"evaluation prior mask record for {alias} must be a mapping"
            )
        expected = record.get(
            "expected_mask_bundle_sha256",
            record.get("bundle_checksum"),
        )
        _require_sha256(
            expected,
            field=f"evaluation prior mask record {alias}.mask_bundle_sha256",
        )
        base_seed = record.get("base_seed")
        entries = record.get("entries")
        if (
            isinstance(base_seed, bool)
            or not isinstance(base_seed, int)
            or not isinstance(entries, list)
            or len(entries) != 9
            or not all(isinstance(entry, Mapping) for entry in entries)
        ):
            raise PooledHybridCountRunnerError(
                f"evaluation prior mask record {alias} lacks its fixed "
                "base seed and nine entry identities"
            )
    return records  # type: ignore[return-value]


def _validate_pooled_contract(
    config: Mapping[str, Any],
) -> PooledHybridContract:
    """Validate the immutable scientific and execution-critical settings."""

    _assert_alias_only_mapping(config)
    _require_equal(
        _section(config, "campaign").get("campaign_id"),
        _CAMPAIGN_ID,
        field="campaign.campaign_id",
    )
    campaign = _section(config, "campaign")
    _require_equal(
        campaign.get("exploratory"), True, field="campaign.exploratory"
    )
    _require_equal(
        campaign.get("frozen_contract_sha256"),
        _FROZEN_CONTRACT_SHA256,
        field="campaign.frozen_contract_sha256",
    )
    trainer = _section(config, "trainer")
    diagnostic = trainer.get("diagnostic_resource_pilot") is True
    seed = int(config.get("seed", -1))
    if (diagnostic and seed != 0) or (
        not diagnostic and seed not in tuple(range(7))
    ):
        raise PooledHybridCountRunnerError(
            "pilot requires seed 0 and production requires seed 0 through 6"
        )
    if int(config.get("fold", -1)) != 0 or int(
        config.get("attempt", -1)
    ) not in {1, 2}:
        raise PooledHybridCountRunnerError(
            "pooled runs require fold=0 and attempt in {1,2}"
        )

    model = _section(config, "model")
    model_name = str(model.get("name", "")).strip().lower()
    if model_name not in _MODEL_KEYS:
        raise PooledHybridCountRunnerError(
            "model.name must be hybrid-count-gat or "
            "hybrid-count-matched-self"
        )
    uses_graph = model_name == "hybrid-count-gat"
    experiment = _section(config, "experiment")
    expected_arm = _PUBLIC_VARIANTS[model_name]
    for field, expected in {
        "arm": expected_arm,
        "core_aliases": list(ANC_ALIASES),
        "one_shared_model_state": True,
        "cross_core_edges": False,
        "resource_pilot": diagnostic,
    }.items():
        _require_equal(
            experiment.get(field), expected, field=f"experiment.{field}"
        )
    fixed_model = {
        "family": (
            "hybrid_count_edge_conditioned_gatv2"
            if uses_graph
            else "hybrid_count_parameter_matched_self_control"
        ),
        "count_representation_schema": _REPRESENTATION_SCHEMA,
        "output_count_states": 8,
        "input_mask_token_id": 8,
        "detection_logits_per_gene": 1,
        "positive_ordinal_logits_per_gene": 6,
        "continuous_predictions_per_gene": 1,
        "embedding_dim": 512,
        "hidden_dim": 512,
        "graph_layers": 2,
        "attention_heads": 4,
        "ffn_dim": 512,
        "decoder_dim": 512,
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_checkpointing": True,
        "exact_receiver_partitioning": True,
        "implicit_self_loops": False,
        "trainable_node_identifiers": False,
        "trainable_edge_identifiers": False,
        "uses_graph_inputs": uses_graph,
        "uses_edge_inputs": uses_graph,
    }
    for field, expected in fixed_model.items():
        _require_equal(model.get(field), expected, field=f"model.{field}")
    chunk_size = model.get("receiver_chunk_size")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise PooledHybridCountRunnerError(
            "model.receiver_chunk_size must be positive"
        )

    dataset = _section(config, "dataset")
    _require_equal(dataset.get("task"), _TASK_FAMILY, field="dataset.task")
    _require_equal(
        dataset.get("tissue_context"),
        "pathology_confirmed_adjacent_normal",
        field="dataset.tissue_context",
    )
    _require_equal(
        dataset.get("fit_scope"),
        "all_117386_cells_across_ten_cores_transductive",
        field="dataset.fit_scope",
    )
    _require_equal(
        dataset.get("target_scale"),
        "raw_biological_probe_counts_with_shared_standardized_log1p",
        field="dataset.target_scale",
    )
    _require_equal(
        dataset.get("total_fit_cells"),
        EXPECTED_TOTAL_NODES,
        field="dataset.total_fit_cells",
    )
    _require_equal(
        dataset.get("biological_target_count"),
        EXPECTED_N_GENES,
        field="dataset.biological_target_count",
    )
    _require_equal(
        tuple(dataset.get("core_aliases", ())),
        ANC_ALIASES,
        field="dataset.core_aliases",
    )
    _require_equal(
        dataset.get("validation_or_test_partition_present"),
        False,
        field="dataset.validation_or_test_partition_present",
    )
    if dataset.get("patient_generalization_supported") not in {None, False}:
        raise PooledHybridCountRunnerError(
            "dataset.patient_generalization_supported must be false"
        )
    _prepared_artifacts(dataset)
    representation = dataset.get("count_representation")
    if not isinstance(representation, Mapping):
        raise PooledHybridCountRunnerError(
            "dataset.count_representation is required"
        )
    for field, expected in {
        "schema": _REPRESENTATION_SCHEMA,
        "source_scale": "raw_biological_probe_counts",
        "num_output_states": 8,
        "mask_token_id": 8,
        "mask_token_is_output": False,
        "fixed_boundaries": True,
        "fit_required": False,
    }.items():
        _require_equal(
            representation.get(field),
            expected,
            field=f"dataset.count_representation.{field}",
        )
    continuous_channel = representation.get("continuous_channel")
    if not isinstance(continuous_channel, Mapping):
        raise PooledHybridCountRunnerError(
            "dataset.count_representation.continuous_channel is required"
        )
    for field, expected in {
        "source": "raw_biological_probe_counts",
        "transform": "shared_equal_core_standardized_log1p",
        "preserves_exact_within_bin_value": True,
        "masked_value": 0.0,
    }.items():
        _require_equal(
            continuous_channel.get(field),
            expected,
            field=f"dataset.count_representation.continuous_channel.{field}",
        )
    expected_count_mapping = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4-7": 4,
        "8-15": 5,
        "16-31": 6,
        "32+": 7,
    }
    if not isinstance(
        representation.get("count_mapping"), Mapping
    ) or dict(representation["count_mapping"]) != expected_count_mapping:
        raise PooledHybridCountRunnerError(
            "dataset.count_representation.count_mapping changed"
        )

    features = _section(config, "features")
    _require_equal(
        features.get("use_edge_features"),
        uses_graph,
        field="features.use_edge_features",
    )
    _require_equal(
        features.get("fit_scope"),
        "all_nodes_transductive",
        field="features.fit_scope",
    )
    expected_expression = {
        "biological_targets": EXPECTED_N_GENES,
        "source_scale": "raw_biological_probe_counts",
        "discrete_transform": "fixed_hybrid_count_states",
        "continuous_transform": "shared_equal_core_standardized_log1p",
        "masked_discrete_value": "input_only_mask_token_8",
        "masked_continuous_value": 0.0,
        "explicit_mask_authoritative_inside_model": True,
    }
    if not isinstance(features.get("node_expression"), Mapping) or dict(
        features["node_expression"]
    ) != expected_expression:
        raise PooledHybridCountRunnerError(
            "features.node_expression differs from the hybrid pooled input"
        )
    node_metadata = features.get("node_metadata")
    if not isinstance(node_metadata, Mapping) or tuple(
        node_metadata.get("fields", ())
    ) != _EXPECTED_METADATA_FIELDS:
        raise PooledHybridCountRunnerError(
            "features.node_metadata.fields must be the frozen 22 covariates"
        )
    _require_equal(
        node_metadata.get("transformed_with"),
        "full_core_fitted_median_imputation_log1p_standardization",
        field="features.node_metadata.transformed_with",
    )
    prohibited = set(features.get("prohibited_node_inputs", ()))
    required_prohibited = {
        "direct_identifiers",
        "absolute_or_local_coordinates",
        "expression_derived_library_size",
        "rna_derived_qc",
        "vendor_cell_type_cluster_neighborhood_or_niche",
        "hidden_target_values",
    }
    if not required_prohibited.issubset(prohibited):
        raise PooledHybridCountRunnerError(
            "features.prohibited_node_inputs omits a frozen leakage guard"
        )
    edge_features = features.get("edge_features")
    if uses_graph:
        if not isinstance(edge_features, Mapping) or tuple(
            edge_features.get("fields", ())
        ) != tuple(EDGE_ATTRIBUTE_NAMES):
            raise PooledHybridCountRunnerError(
                "GAT edge features must be the frozen 17 geometry fields"
            )
        _require_equal(
            edge_features.get("fit_scope"),
            "all_retained_directed_edges_transductive",
            field="features.edge_features.fit_scope",
        )
        _require_equal(
            edge_features.get("standardization"),
            "full_core_edge_wise",
            field="features.edge_features.standardization",
        )
    elif edge_features not in (None, [], ()):
        raise PooledHybridCountRunnerError(
            "matched self must receive no edge features"
        )

    graph = _section(config, "graph")
    for field, expected in {
        "kind": "exact_spatial_knn_radius_guard",
        "neighbor_k": 1000,
        "k": 1000,
        "radius_um": 2000.0,
        "radius_guard_um": 2000.0,
        "symmetry": "mutual",
        "edge_dropout": 0.0,
        "self_loops": False,
        "cross_core_edges": False,
        "graph_batches": "ten_disconnected_complete_core_graphs",
        "coordinates_are_node_covariates": False,
    }.items():
        _require_equal(graph.get(field), expected, field=f"graph.{field}")
    if graph.get("neighbor_sampling") not in (None, False):
        raise PooledHybridCountRunnerError("neighbor sampling is prohibited")
    _graph_records(graph)

    masking = _section(config, "masking")
    for field, expected in {
        "curriculum": "P+N+B",
        "warmup_epochs": 10,
        "mask_seed": 314159,
        "block_shape": "disk",
    }.items():
        _require_equal(masking.get(field), expected, field=f"masking.{field}")
    if masking.get("block_width_um") is not None:
        raise PooledHybridCountRunnerError(
            "masking.block_width_um must remain null"
        )
    rates = masking.get("rates", masking.get("rate"))
    if not isinstance(rates, Mapping) or {
        name: float(rates.get(name, -1.0))
        for name in ("partial_gene", "whole_node", "spatial_block")
    } != {
        "partial_gene": 0.2,
        "whole_node": 0.1,
        "spatial_block": 0.1,
    }:
        raise PooledHybridCountRunnerError(
            "masking rates differ from the frozen schedule"
        )
    _require_equal(
        masking.get("post_warmup_probabilities"),
        {
            "partial_gene": 0.6,
            "whole_node": 0.3,
            "spatial_block": 0.1,
        },
        field="masking.post_warmup_probabilities",
    )
    _require_equal(
        masking.get("mask_expression_only"),
        True,
        field="masking.mask_expression_only",
    )
    _require_equal(
        masking.get("explicit_gene_mask_channel"),
        True,
        field="masking.explicit_gene_mask_channel",
    )

    evaluation = _section(config, "evaluation")
    for field, expected in {
        "task_family": _TASK_FAMILY,
        "protocol": _PROTOCOL,
        "canonical_prediction_split": "fit",
        "primary_metric": _PRIMARY_METRIC,
        "primary_direction": "minimize",
        "splits": ["fit"],
        "mask_modes": list(_REQUIRED_PUBLIC_MASKS),
        "mask_replicates_per_mode": 3,
        "fixed_mask_bundle": True,
        "mask_source": "exact_regeneration_of_prior_per_core_fixed_masks",
        "generalization_estimate": False,
        "validation_or_test_selection": False,
        "diagnostic_only": diagnostic,
        "conclusion_bearing": not diagnostic,
    }.items():
        _require_equal(
            evaluation.get(field), expected, field=f"evaluation.{field}"
        )
    _prior_mask_records(evaluation)

    expected_epochs = 2 if diagnostic else 200
    expected_steps = expected_epochs * len(ANC_ALIASES)
    for field, expected in {
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "huber_delta": 1.0,
        "max_epochs": expected_epochs,
        "optimizer_steps_per_epoch": 10,
        "batch_unit": "one_complete_core_graph",
        "total_optimizer_steps": expected_steps,
        "core_order": "deterministic_epoch_shuffle",
        "core_order_seed": 271828,
        "core_sampling": "exactly_once_per_epoch",
        "core_weighting": "equal",
        "neighbor_sampling": False,
        "fixed_epoch_budget": True,
        "early_stopping": False,
        "validation_every": None,
        "precision": "mixed",
        "amp": True,
        "deterministic": True,
        "deterministic_warn_only": False,
        "restore_best": False,
        "monitored_metric": None,
        "primary_checkpoint_role": "last",
        "checkpoint_policy": "last_only",
        "graph_execution": (
            "sequential_complete_core_exact_no_neighbor_sampling"
        ),
        "objective": _OBJECTIVE,
        "resume_boundary": "completed_global_epoch_only",
        "run_fp32_amp_equivalence": diagnostic,
    }.items():
        _require_equal(trainer.get(field), expected, field=f"trainer.{field}")
    authorization = trainer.get("amp_authorization")
    if not isinstance(authorization, Mapping):
        raise PooledHybridCountRunnerError(
            "trainer.amp_authorization is required"
        )
    for field, expected in {
        "mode": (
            "same_weight_fp32_amp_each_core_diagnostic"
            if diagnostic
            else "require_external_pilot_gate_receipt"
        ),
        "receipt_schema": _PILOT_RECEIPT_SCHEMA,
        "receipt_reference": _PILOT_RECEIPT_RELATIVE.as_posix(),
        "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
    }.items():
        _require_equal(
            authorization.get(field),
            expected,
            field=f"trainer.amp_authorization.{field}",
        )
    return PooledHybridContract(
        model_name=model_name,
        model_key=_MODEL_KEYS[model_name],
        public_variant=_PUBLIC_VARIANTS[model_name],
        diagnostic_resource_pilot=diagnostic,
        uses_graph=uses_graph,
        seed=seed,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_contract(project_root: Path) -> Mapping[str, Any]:
    contract_path = project_root / _FROZEN_CONTRACT_RELATIVE
    campaign_path = project_root / _CAMPAIGN_RELATIVE
    if (
        not contract_path.is_file()
        or contract_path.is_symlink()
        or not campaign_path.is_file()
        or campaign_path.is_symlink()
    ):
        raise PooledHybridCountRunnerError(
            "frozen contract or campaign declaration is missing or unsafe"
        )
    actual = _sha256_file(contract_path)
    if actual != _FROZEN_CONTRACT_SHA256:
        raise PooledHybridCountRunnerError(
            "frozen pooled task contract checksum changed"
        )
    import yaml

    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    if not isinstance(campaign, Mapping) or campaign.get(
        "frozen_contract_sha256"
    ) != actual:
        raise PooledHybridCountRunnerError(
            "campaign declaration disagrees with the frozen contract checksum"
        )
    return {
        "path": _FROZEN_CONTRACT_RELATIVE.as_posix(),
        "sha256": actual,
        "campaign_declaration": _CAMPAIGN_RELATIVE.as_posix(),
        "verified": True,
    }


def _read_signed_receipt(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise PooledHybridCountRunnerError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PooledHybridCountRunnerError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping):
        raise PooledHybridCountRunnerError(f"{label} must be a JSON mapping")
    core = dict(payload)
    checksum = core.pop("checksum", None)
    if not isinstance(checksum, str) or checksum != canonical_sha256(core):
        raise PooledHybridCountRunnerError(f"{label} checksum is invalid")
    return dict(payload), checksum


def _validate_materialization_receipt(
    project_root: Path,
    config: Mapping[str, Any],
    contract: PooledHybridContract,
) -> Mapping[str, Any]:
    configured = _section(config, "metadata").get(
        "locked_config_materialization_receipt"
    )
    _require_equal(
        configured,
        _MATERIALIZATION_RECEIPT_RELATIVE.as_posix(),
        field="metadata.locked_config_materialization_receipt",
    )
    payload, checksum = _read_signed_receipt(
        project_root / _MATERIALIZATION_RECEIPT_RELATIVE,
        label="locked config materialization receipt",
    )
    _require_equal(
        payload.get("campaign_id"),
        _CAMPAIGN_ID,
        field="materialization.campaign_id",
    )
    frozen = payload.get("frozen_contract")
    declared_frozen = (
        frozen.get("sha256") if isinstance(frozen, Mapping) else payload.get(
            "frozen_contract_sha256"
        )
    )
    _require_equal(
        declared_frozen,
        _FROZEN_CONTRACT_SHA256,
        field="materialization.frozen_contract_sha256",
    )
    # The materializer binds the root attempt.  A queue-owned retry changes
    # only the run attempt and must still verify against that immutable
    # scientific configuration.
    locked_config = deepcopy(dict(config))
    locked_config["attempt"] = 1
    config_sha = canonical_sha256(locked_config)
    jobs_field = (
        "pilot_jobs"
        if contract.diagnostic_resource_pilot
        else "production_jobs"
    )
    jobs = payload.get(jobs_field)
    if not isinstance(jobs, list) or not any(
        isinstance(job, Mapping)
        and job.get("config_sha256") == config_sha
        and (
            job.get("arm") in {
                contract.public_variant,
                contract.public_variant.removeprefix("pooled-"),
            }
        )
        and int(job.get("seed", contract.seed)) == contract.seed
        for job in jobs
    ):
        raise PooledHybridCountRunnerError(
            "resolved configuration is not checksum-bound to the "
            "materialization receipt"
        )
    return {
        "path": _MATERIALIZATION_RECEIPT_RELATIVE.as_posix(),
        "materialization_checksum": checksum,
        "config_sha256": config_sha,
        "runtime_config_sha256": canonical_sha256(config),
        "run_attempt": int(config.get("attempt", 1)),
        "config_list": jobs_field,
        "verified": True,
    }


def _validate_production_pilot_receipt(
    project_root: Path,
    config: Mapping[str, Any],
    contract: PooledHybridContract,
    *,
    materialization_checksum: str,
) -> Mapping[str, Any] | None:
    if contract.diagnostic_resource_pilot:
        return None
    authorization = _section(config, "trainer").get("amp_authorization")
    if not isinstance(authorization, Mapping):
        raise PooledHybridCountRunnerError(
            "production trainer.amp_authorization is required"
        )
    for field, expected in {
        "mode": "require_external_pilot_gate_receipt",
        "receipt_schema": _PILOT_RECEIPT_SCHEMA,
        "receipt_reference": _PILOT_RECEIPT_RELATIVE.as_posix(),
        "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
    }.items():
        _require_equal(
            authorization.get(field),
            expected,
            field=f"trainer.amp_authorization.{field}",
        )
    payload, checksum = _read_signed_receipt(
        project_root / _PILOT_RECEIPT_RELATIVE,
        label="pooled pilot gate receipt",
    )
    for field, expected in {
        "schema_version": 1,
        "receipt_kind": _PILOT_RECEIPT_SCHEMA,
        "campaign_id": _CAMPAIGN_ID,
        "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
        "materialization_checksum": materialization_checksum,
        "gate_passed": True,
        "production_authorized": True,
        "thresholds": _PILOT_GATE_THRESHOLDS,
        "failure_reasons": [],
        "same_frozen_precision_batches_all_cores": True,
        "same_evaluation_masks": True,
        "same_verified_graph_bundle": True,
        "paired_initialization_digests_match": True,
    }.items():
        _require_equal(
            payload.get(field), expected, field=f"pilot_receipt.{field}"
        )
    jobs = payload.get("jobs")
    if (
        not isinstance(jobs, list)
        or len(jobs) != 2
        or not all(isinstance(job, Mapping) for job in jobs)
    ):
        raise PooledHybridCountRunnerError(
            "pilot receipt must contain exactly two pooled arm jobs"
        )
    expected_slots = {
        (arm, 0) for arm in _PUBLIC_VARIANTS.values()
    }
    observed_slots = {
        (str(job.get("arm")), int(job.get("seed", -1))) for job in jobs
    }
    if observed_slots != expected_slots:
        raise PooledHybridCountRunnerError(
            "pilot receipt does not contain the exact two pooled arms"
        )
    for job in jobs:
        if (
            job.get("parameter_count") != _EXPECTED_PARAMETER_COUNT
            or job.get("checkpoint_epoch") != 1
            or job.get("checkpoint_role") != "last"
            or any(
                job.get(field) is not True
                for field in _PILOT_JOB_REQUIRED_TRUE_FIELDS
            )
        ):
            raise PooledHybridCountRunnerError(
                f"pilot receipt arm {job.get('arm')} is invalid"
            )
    return {
        "path": _PILOT_RECEIPT_RELATIVE.as_posix(),
        "checksum": checksum,
        "materialization_checksum": materialization_checksum,
        "passed": True,
    }


def _graph_config_for_alias(
    graph_config: Mapping[str, Any], alias: str
) -> dict[str, Any]:
    record = _graph_records(graph_config)[alias]
    checksum = record.get(
        "expected_materialized_graph_sha256",
        record.get("graph_sha256"),
    )
    edges = record.get(
        "expected_directed_edges", record.get("n_directed_edges")
    )
    return {
        **dict(graph_config),
        "expected_materialized_graph_sha256": checksum,
        "expected_directed_edges": edges,
    }


def _validate_graph(
    graph: ReceiverSortedGraph,
    core: PooledCoreData,
    config: Mapping[str, Any],
) -> None:
    if graph.n_nodes != core.n_nodes or graph.k != 1000:
        raise PooledHybridCountRunnerError(
            f"{core.alias} exact mutual-k1000 graph is not aligned"
        )
    if tuple(graph.edge_attribute_names) != tuple(EDGE_ATTRIBUTE_NAMES):
        raise PooledHybridCountRunnerError(
            f"{core.alias} graph edge schema changed"
        )
    if (
        not graph.qc.receiver_sorted
        or graph.qc.self_loops
        or graph.qc.duplicate_directed_edges
        or not graph.qc.directed_edge_pairs_are_symmetric
    ):
        raise PooledHybridCountRunnerError(
            f"{core.alias} graph violates exact mutual graph invariants"
        )
    _verify_graph_identity(graph, config)


def _prepare_core_batches(
    cohort: PooledFullCoreCohort,
    graph_config: Mapping[str, Any],
    *,
    uses_graph: bool,
    receiver_graphs: Mapping[str, ReceiverSortedGraph] | None,
) -> tuple[
    tuple[PooledCoreBatch, ...],
    tuple[dict[str, Any], ...],
    float,
]:
    if receiver_graphs is not None and set(receiver_graphs) != set(ANC_ALIASES):
        raise PooledHybridCountRunnerError(
            "injected receiver_graphs must contain every ANC alias exactly once"
        )
    started = time.monotonic()
    batches: list[PooledCoreBatch] = []
    records: list[dict[str, Any]] = []
    for core in cohort.cores:
        per_core_config = _graph_config_for_alias(graph_config, core.alias)
        graph = (
            receiver_graphs[core.alias]
            if receiver_graphs is not None
            else _build_graph(core, per_core_config)
        )
        _validate_graph(graph, core, per_core_config)
        view = _fit_view(
            core=core,
            graph=graph,
            uses_graph=uses_graph,
            expression=validate_raw_counts(
                core.expression_counts,
                name=f"{core.alias} expression_counts",
            ),
        )
        batches.append(PooledCoreBatch(alias=core.alias, view=view))
        records.append(
            {
                "alias": core.alias,
                "n_nodes": core.n_nodes,
                "n_directed_edges": graph.qc.n_directed_edges,
                "graph_sha256": graph.checksums.graph_sha256,
                "graph_qc": graph.qc.to_dict(),
                "edge_attribute_names": list(graph.edge_attribute_names),
                "supplied_to_model": uses_graph,
            }
        )
        del graph
        gc.collect()
    return tuple(batches), tuple(records), time.monotonic() - started


def _mask_bundle_for_core(
    *,
    config: Mapping[str, Any],
    core: PooledCoreData,
    training: TrainingConfig,
) -> Any:
    evaluation = _section(config, "evaluation")
    record = _prior_mask_records(evaluation)[core.alias]
    specs = [
        MaskSpec(
            mode="partial",
            partial_gene_rate=training.partial_gene_rate,
            node_rate=training.node_rate,
            block_node_rate=training.block_node_rate,
            block_width_um=training.block_width_um,
            block_shape=training.block_shape,
            label="partial_gene",
        ),
        MaskSpec(
            mode="node",
            partial_gene_rate=training.partial_gene_rate,
            node_rate=training.node_rate,
            block_node_rate=training.block_node_rate,
            block_width_um=training.block_width_um,
            block_shape=training.block_shape,
            label="whole_node",
        ),
        MaskSpec(
            mode="block",
            partial_gene_rate=training.partial_gene_rate,
            node_rate=training.node_rate,
            block_node_rate=training.block_node_rate,
            block_width_um=training.block_width_um,
            block_shape=training.block_shape,
            label="spatial_block",
        ),
    ]
    bundle = create_fixed_mask_bundle(
        {"fit": core.coordinates_um},
        core.n_genes,
        specs,
        replicates=int(evaluation["mask_replicates_per_mode"]),
        base_seed=int(record["base_seed"]),
    )
    expected = record.get(
        "expected_mask_bundle_sha256",
        record.get("bundle_checksum"),
    )
    if bundle.checksum != expected:
        raise PooledHybridCountRunnerError(
            f"{core.alias} regenerated evaluation mask bundle checksum changed"
        )
    observed_entries = [
        {
            "entry_id": str(entry["entry_id"]),
            "mode": str(entry["spec"]["label"]),
            "replicate": int(entry["replicate"]),
            "seed": int(entry["seed"]),
            "mask_checksum": str(entry["mask_checksum"]),
        }
        for entry in bundle.manifest["entries"]
    ]
    if list(record["entries"]) != observed_entries:
        raise PooledHybridCountRunnerError(
            f"{core.alias} evaluation mask entry identities changed"
        )
    return bundle


def _whole_node_replicate_zero_mask(bundle: Any) -> np.ndarray:
    selected = [
        entry
        for entry in bundle.manifest["entries"]
        if entry["spec"]["mode"] == "node"
        and int(entry["replicate"]) == 0
    ]
    if len(selected) != 1:
        raise PooledHybridCountRunnerError(
            "mask bundle lacks one whole-node replicate zero"
        )
    return bundle.masks[str(selected[0]["entry_id"])]


def _clear_graph_layout_caches(model: torch.nn.Module) -> None:
    for module in model.modules():
        clear = getattr(module, "clear_edge_layout_cache", None)
        if callable(clear):
            clear()


def _state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash an initial module state independently of serialization details."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = (
            torch.as_tensor(state_dict[name])
            .detach()
            .cpu()
            .contiguous()
        )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _precision_payload(
    results: Mapping[str, PrecisionEquivalenceResult],
    *,
    materialization_checksum: str,
    config_sha256: str,
) -> dict[str, Any]:
    records = {
        alias: {
            **asdict(results[alias]),
            "actual_amp_dtype": (
                "float16"
                if results[alias].amp_dtype == "auto"
                else results[alias].amp_dtype
            ),
        }
        for alias in ANC_ALIASES
    }
    maximum = max(
        result.absolute_total_loss_discrepancy for result in results.values()
    )
    return {
        "schema": "pooled_hybrid_count_fp32_amp_equivalence_v1",
        "aliases": list(ANC_ALIASES),
        "same_frozen_batch_per_core": True,
        "maximum_allowed_discrepancy": _PILOT_MAX_AMP_DISCREPANCY,
        "maximum_observed_discrepancy": maximum,
        "maximum_fp32_amp_absolute_total_loss_discrepancy": maximum,
        "all_cores_passed": all(result.passed for result in results.values()),
        "precision_equivalence_passed": all(
            result.passed for result in results.values()
        ),
        "per_core": records,
        "materialization_checksum": materialization_checksum,
        "config_sha256": config_sha256,
    }


def _balanced_bce(
    target: np.ndarray, probability: np.ndarray, *, label: str
) -> float:
    target = target.astype(bool, copy=False)
    if not target.any() or target.all():
        raise PooledHybridCountRunnerError(
            f"{label} balanced BCE is missing a stratum"
        )
    zero_loss = -np.log1p(-probability[~target]).mean()
    positive_loss = -np.log(probability[target]).mean()
    value = 0.5 * (zero_loss + positive_loss)
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"{label} balanced BCE is non-finite")
    return float(value)


def _reference_metrics(
    result: Any,
    references: HybridCountReferences,
    *,
    expression_mean: np.ndarray,
    expression_scale: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    """Evaluate reference heads on masked entries without dense head expansion."""

    target = result.target.numpy()
    mask = result.target_mask.numpy().astype(bool, copy=False)
    if target.shape != mask.shape:
        raise PooledHybridCountRunnerError(
            "reference target and mask are not aligned"
        )
    rows, genes = np.nonzero(mask)
    counts = target[rows, genes].astype(np.float64, copy=False)
    states = tokenize_raw_counts(counts.reshape(1, -1)).reshape(-1)
    positive = counts > 0
    detection_probability = references.detection_probability[genes]
    detection = _balanced_bce(
        positive,
        detection_probability,
        label=f"{prefix} detection",
    )

    positive_states = states[positive]
    positive_genes = genes[positive]
    threshold_losses: list[float] = []
    for threshold_index in range(6):
        above = positive_states > threshold_index + 1
        probability = references.positive_ordinal_probability[
            positive_genes, threshold_index
        ]
        threshold_losses.append(
            _balanced_bce(
                above,
                probability,
                label=f"{prefix} ordinal threshold {threshold_index + 1}",
            )
        )
    ordinal = float(np.mean(threshold_losses))
    standardized = (
        np.log1p(counts[positive]) - expression_mean[positive_genes]
    ) / expression_scale[positive_genes]
    continuous_prediction = references.positive_continuous_standardized[
        positive_genes
    ]
    continuous_error = continuous_prediction - standardized
    absolute = np.abs(continuous_error)
    huber = float(
        np.mean(np.where(absolute <= 1.0, 0.5 * absolute**2, absolute - 0.5))
    )
    decoded_detection = references.detected_state[genes]
    true_positive = int(np.count_nonzero(positive & decoded_detection))
    true_negative = int(np.count_nonzero(~positive & ~decoded_detection))
    sensitivity = true_positive / int(np.count_nonzero(positive))
    specificity = true_negative / int(np.count_nonzero(~positive))
    positive_state = references.positive_state[positive_genes]
    ordinal_error = np.abs(positive_state - positive_states)
    decoded_state = references.count_state[genes]
    detected_log = (
        references.positive_continuous_standardized[genes]
        * expression_scale[genes]
        + expression_mean[genes]
    )
    reconstructed_log = np.where(
        decoded_detection, np.maximum(detected_log, 0.0), 0.0
    )
    metrics = {
        f"{prefix}_hybrid_loss": (detection + ordinal + huber) / 3.0,
        f"{prefix}_detection_bce": detection,
        f"{prefix}_ordinal_bce": ordinal,
        f"{prefix}_positive_continuous_huber": huber,
        f"{prefix}_detection_balanced_accuracy": 0.5
        * (sensitivity + specificity),
        f"{prefix}_positive_ordinal_mae": float(np.mean(ordinal_error)),
        f"{prefix}_positive_continuous_mae": float(
            np.mean(np.abs(continuous_error))
        ),
        f"{prefix}_state8_exact_accuracy": float(
            np.mean(decoded_state == states)
        ),
        f"{prefix}_reconstructed_count_log1p_mae": float(
            np.mean(np.abs(reconstructed_log - np.log1p(counts)))
        ),
    }
    if not all(_finite_scalar(value) for value in metrics.values()):
        raise FloatingPointError(f"{prefix} reference metrics are non-finite")
    return metrics


def _required_metric_aliases(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Expose frozen report names alongside the established metric schema."""

    result = dict(metrics)
    aliases = {
        "total_hybrid_loss": "hybrid_loss",
        "eight_state_exact_accuracy": "state8_exact_accuracy",
        "eight_state_balanced_accuracy": "state8_balanced_accuracy",
        "eight_state_support": "state8_support",
        "eight_state_recall": "state8_recall",
        "positive_standardized_log1p_huber": (
            "positive_continuous_huber"
        ),
        "positive_standardized_log1p_mae": "positive_continuous_mae",
    }
    for public_name, established_name in aliases.items():
        if established_name not in result:
            raise PooledHybridCountRunnerError(
                f"required metric {established_name} is missing"
            )
        result[public_name] = result[established_name]
    return result


def _protected_prediction_rows(
    *,
    archive: RunArchive,
    dataset: Mapping[str, Any],
    alias: str,
    graph_id: str,
    edge_count: int,
    fold: int,
    entry: Mapping[str, Any],
    result: Any,
    sample_key_salt: str,
) -> Iterator[dict[str, Any]]:
    target = result.target.numpy()
    mask = result.target_mask.numpy().astype(bool, copy=False)
    reconstructed = result.evaluation.reconstructed_count.numpy()
    predicted_state = result.evaluation.count_state.numpy()
    predicted_detected = result.evaluation.detected.numpy()
    predicted_continuous = (
        result.evaluation.positive_continuous_standardized.numpy()
    )
    if not (
        target.shape
        == mask.shape
        == reconstructed.shape
        == predicted_state.shape
        == predicted_detected.shape
        == predicted_continuous.shape
    ):
        raise PooledHybridCountRunnerError(
            f"{alias} prediction arrays are not aligned"
        )
    namespace = (
        f"bagm:{dataset['dataset_id']}:{dataset['version']}:{alias}:"
        "pooled-full-core-fit"
    )
    target_nodes = result.target_nodes.numpy().astype(np.int64, copy=False)
    batch: list[dict[str, Any]] = []
    for local_index, node_index in enumerate(target_nodes):
        target_indices = np.flatnonzero(mask[local_index])
        truth = np.rint(target[local_index, target_indices]).astype(np.int32)
        truth_state = tokenize_raw_counts(truth.reshape(1, -1)).reshape(-1)
        batch.append(
            {
                "_protected_local_index": int(node_index),
                "run_id": archive.run_id,
                "biological_unit_alias": alias,
                "graph_id": graph_id,
                "dataset_id": str(dataset["dataset_id"]),
                "split": "fit",
                "fold": fold,
                "y_true": truth.astype(int).tolist(),
                "y_pred": reconstructed[
                    local_index, target_indices
                ].astype(float).tolist(),
                "target_indices": target_indices.astype(int).tolist(),
                "y_true_state": truth_state.astype(int).tolist(),
                "y_pred_state": predicted_state[
                    local_index, target_indices
                ].astype(int).tolist(),
                "y_pred_detected": predicted_detected[
                    local_index, target_indices
                ].astype(int).tolist(),
                "y_pred_standardized_log1p": predicted_continuous[
                    local_index, target_indices
                ].astype(float).tolist(),
                "node_count": int(result.full_mask.shape[0]),
                "edge_count": edge_count,
                "effective_mask_rate": float(
                    target_indices.size / result.full_mask.shape[1]
                ),
                "masking_type": "whole_node",
                "mask_replicate": int(entry["replicate"]),
            }
        )
        if len(batch) >= 128:
            yield from deidentify_prediction_rows(
                batch,
                identifier_fields=["_protected_local_index"],
                salt=sample_key_salt,
                namespace=namespace,
            )
            batch.clear()
    if batch:
        yield from deidentify_prediction_rows(
            batch,
            identifier_fields=["_protected_local_index"],
            salt=sample_key_salt,
            namespace=namespace,
        )


def _aggregate_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates_per_mode: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    candidate_fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in _ROW_METADATA_FIELDS
            and key != "n_masked"
            and (_finite_scalar(value) or value is None)
        }
    )
    for alias in ANC_ALIASES:
        for mode in _REQUIRED_PUBLIC_MASKS:
            selected = [
                row
                for row in rows
                if row["biological_unit_alias"] == alias
                and row["mask_mode"] == mode
            ]
            if len(selected) != replicates_per_mode:
                raise PooledHybridCountRunnerError(
                    f"expected {replicates_per_mode} rows for {alias} {mode}"
                )
            for field in candidate_fields:
                metrics[f"fit/{alias}/{mode}/{field}"] = _mean_finite(
                    selected, field
                )
    for mode in _REQUIRED_PUBLIC_MASKS:
        for field in candidate_fields:
            per_core = [
                metrics[f"fit/{alias}/{mode}/{field}"]
                for alias in ANC_ALIASES
            ]
            finite = [
                float(value)
                for value in per_core
                if value is not None and math.isfinite(float(value))
            ]
            metrics[f"fit/{mode}/{field}"] = (
                float(np.mean(finite)) if finite else None
            )
    return metrics


def _checkpoint_bytes(
    *,
    archive: RunArchive,
    contract: PooledHybridContract,
    model_config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    parameter_audit: Mapping[str, Any],
    training: PooledHybridCountTrainingResult,
    cohort: PooledFullCoreCohort,
    graph_records: Sequence[Mapping[str, Any]],
    mask_bundles: Mapping[str, Any],
    per_core_references: Mapping[str, HybridCountReferences],
    pooled_references: HybridCountReferences,
    effective_amp: bool,
    materialization_checksum: str,
    config_sha256: str,
) -> bytes:
    payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "checkpoint_role": "last",
        "checkpoint_policy": "final_global_epoch_no_validation_selection",
        "training_protocol": training.training_protocol,
        "task_family": _TASK_FAMILY,
        "model_name": contract.model_name,
        "public_variant": contract.public_variant,
        "model_seed": contract.seed,
        "aliases": list(ANC_ALIASES),
        "model_config": dict(model_config),
        "model_construction": dict(model_construction),
        "parameter_structure_audit": dict(parameter_audit),
        "epoch": training.final_epoch,
        "completed_global_epochs": training.completed_global_epochs,
        "optimizer_steps_completed": training.optimizer_steps_completed,
        "fixed_epoch_budget": training.fixed_epoch_budget,
        "model_state_dict": dict(training.final_state_dict),
        "optimizer_state_dict": dict(training.final_optimizer_state_dict),
        "scaler_state_dict": dict(training.final_scaler_state_dict),
        "state_dict_sha256": training.final_state_checksum,
        "pooled_cohort_fingerprint_sha256": cohort.fingerprint_sha256,
        "pooled_cohort_checksums": cohort.checksums.to_dict(),
        "per_core_preprocessing_sha256": {
            core.alias: core.checksums.preprocessing_sha256
            for core in cohort.cores
        },
        "per_core_raw_count_sha256": {
            core.alias: core.checksums.expression_counts_sha256
            for core in cohort.cores
        },
        "per_core_graph_sha256": {
            str(record["alias"]): str(record["graph_sha256"])
            for record in graph_records
        },
        "per_core_evaluation_mask_bundle_sha256": {
            alias: mask_bundles[alias].checksum for alias in ANC_ALIASES
        },
        "per_core_reference_sha256": {
            alias: per_core_references[alias].audit["reference_sha256"]
            for alias in ANC_ALIASES
        },
        "pooled_reference_sha256": pooled_references.audit[
            "reference_sha256"
        ],
        "count_representation_schema": _REPRESENTATION_SCHEMA,
        "objective": _OBJECTIVE,
        "effective_amp": effective_amp,
        "materialization_checksum": materialization_checksum,
        "config_sha256": config_sha256,
        "selection_policy": "last_epoch_without_validation_selection",
        "monitored_metric": None,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _cuda_peak_bytes(device: str) -> int:
    resolved = torch.device(device)
    return (
        int(torch.cuda.max_memory_allocated(resolved))
        if resolved.type == "cuda"
        else 0
    )


def _summary_peak_vram_fields(peak_vram_bytes: int) -> dict[str, float]:
    value_gib = float(peak_vram_bytes) / 1024**3
    return {
        "peak_vram_gib": value_gib,
        # Schema-v3 retains the historical ``gb`` column name, whose values
        # have consistently represented binary GiB.
        "peak_vram_gb": value_gib,
    }


def _resource_diagnostic(
    *,
    archive: RunArchive,
    contract: PooledHybridContract,
    configured_training: TrainingConfig,
    effective_training: TrainingConfig,
    precision_payload: Mapping[str, Any] | None,
    training: PooledHybridCountTrainingResult,
    total_duration: float,
    training_duration: float,
    evaluation_duration: float,
    peak_vram_bytes: int,
    parameter_audit: Mapping[str, Any],
    materialization_checksum: str,
    config_sha256: str,
) -> dict[str, Any]:
    host_bytes = _peak_host_memory_bytes()
    projected_hours = (
        float(
            np.mean(
                [record.duration_seconds for record in training.global_history]
            )
        )
        * 200.0
        / 3600.0
        if contract.diagnostic_resource_pilot
        else None
    )
    current_free = shutil.disk_usage(archive.paths.artifact_root).free
    projected_free = max(
        0, current_free - _PROJECTED_CAMPAIGN_OUTPUT_BYTES
    )
    every_core = all(
        record.cores_visited == 10
        and set(record.ordered_aliases) == set(ANC_ALIASES)
        and len(record.ordered_aliases) == len(set(record.ordered_aliases))
        for record in training.global_history
    )
    expected_steps = training.fixed_epoch_budget * len(ANC_ALIASES)
    all_steps = training.optimizer_steps_completed == expected_steps and (
        len(training.core_history) == expected_steps
    )
    precision_passed = (
        True
        if precision_payload is None
        else precision_payload.get("all_cores_passed") is True
    )
    vram_passed = peak_vram_bytes / 1024**3 <= _PILOT_MAX_VRAM_GIB
    host_passed = host_bytes / 1024**3 <= _PILOT_MAX_HOST_GIB
    runtime_passed = (
        True
        if not contract.uses_graph or projected_hours is None
        else projected_hours <= _PILOT_MAX_PROJECTED_HOURS
    )
    disk_passed = (
        projected_free / 1024**3
        >= _PILOT_MIN_PROJECTED_FREE_DISK_GIB
    )
    parameter_match = (
        parameter_audit.get("exact_trainable_parameter_match") is True
        and parameter_audit.get("trainable_parameter_count_graph")
        == _EXPECTED_PARAMETER_COUNT
        and parameter_audit.get("trainable_parameter_count_self")
        == _EXPECTED_PARAMETER_COUNT
    )
    initialization_match = (
        parameter_audit.get("encoder_initial_state_bit_identical") is True
        and parameter_audit.get("decoder_initial_state_bit_identical") is True
    )
    checks = {
        "finite_losses_and_gradients": True,
        "parameter_match_passed": parameter_match,
        "paired_initialization_passed": initialization_match,
        "precision_equivalence_passed": precision_passed,
        "peak_vram_passed": vram_passed,
        "peak_host_memory_passed": host_passed,
        "projected_runtime_passed": runtime_passed,
        "projected_disk_passed": disk_passed,
        "every_core_once_each_epoch": every_core,
        "all_20_optimizer_steps_completed": (
            all_steps
            and training.optimizer_steps_completed == 20
            if contract.diagnostic_resource_pilot
            else all_steps
        ),
    }
    pilot_passed = all(checks.values())
    return {
        "schema": "pooled_hybrid_count_resource_diagnostic_v1",
        "diagnostic_resource_pilot": contract.diagnostic_resource_pilot,
        "public_variant": contract.public_variant,
        "aliases": list(ANC_ALIASES),
        "device": training.device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(training.device))
            if torch.device(training.device).type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "configured_amp": configured_training.amp,
        "effective_training_amp": effective_training.amp,
        "amp_fallback_reason": (
            "one_or_more_core_precision_thresholds_failed"
            if configured_training.amp and not effective_training.amp
            else None
        ),
        "parameter_count": _EXPECTED_PARAMETER_COUNT,
        "finite_losses_and_gradients": True,
        "exact_parameter_match": parameter_match,
        "paired_initialization_match": initialization_match,
        "completed_global_epochs": training.completed_global_epochs,
        "optimizer_steps_completed": training.optimizer_steps_completed,
        "global_epoch_duration_seconds": [
            record.duration_seconds for record in training.global_history
        ],
        "projected_200_epoch_runtime_hours": projected_hours,
        "training_duration_seconds": training_duration,
        "evaluation_duration_seconds": evaluation_duration,
        "total_pilot_or_run_duration_seconds": total_duration,
        "peak_allocated_vram_bytes": peak_vram_bytes,
        "peak_allocated_vram_gib": peak_vram_bytes / 1024**3,
        "peak_host_memory_bytes": host_bytes,
        "peak_host_memory_gib": host_bytes / 1024**3,
        "current_free_disk_bytes": current_free,
        "projected_campaign_output_bytes": (
            _PROJECTED_CAMPAIGN_OUTPUT_BYTES
        ),
        "projected_final_free_disk_bytes": projected_free,
        "projected_final_free_disk_gib": projected_free / 1024**3,
        "pilot_thresholds": dict(_PILOT_GATE_THRESHOLDS),
        "pilot_checks": checks,
        "every_core_once_each_epoch": every_core,
        "all_20_optimizer_steps_completed": checks[
            "all_20_optimizer_steps_completed"
        ],
        "pilot_gate_passed": (
            pilot_passed if contract.diagnostic_resource_pilot else None
        ),
        "materialization_checksum": materialization_checksum,
        "config_sha256": config_sha256,
    }


def run_pooled_hybrid_count_capacity(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    sample_key_salt: str,
    pooled_cohort: PooledFullCoreCohort | None = None,
    receiver_graphs: Mapping[str, ReceiverSortedGraph] | None = None,
) -> CapacityRunResult:
    """Fit and evaluate one shared pooled GAT or matched-self member."""

    started = time.monotonic()
    if len(sample_key_salt.encode("utf-8")) < 16:
        raise RunValidationError(
            "BAGM_SAMPLE_KEY_SALT must contain at least 16 bytes"
        )
    contract = _validate_pooled_contract(config)
    project_root = archive.paths.project_root
    frozen_verification = _verify_frozen_contract(project_root)
    materialization = _validate_materialization_receipt(
        project_root, config, contract
    )
    pilot_receipt = _validate_production_pilot_receipt(
        project_root,
        config,
        contract,
        materialization_checksum=str(
            materialization["materialization_checksum"]
        ),
    )
    config_sha256 = str(materialization["config_sha256"])
    runtime_config_sha256 = str(materialization["runtime_config_sha256"])
    run_attempt = int(materialization["run_attempt"])
    materialization_checksum = str(
        materialization["materialization_checksum"]
    )
    dataset = _section(config, "dataset")
    graph_config = _section(config, "graph")
    model_config = _section(config, "model")
    evaluation_config = _section(config, "evaluation")

    data_started = time.monotonic()
    if pooled_cohort is None:
        prepared = {
            alias: (
                path
                if path.is_absolute()
                else project_root / path
            )
            for alias, path in _prepared_artifacts(dataset).items()
        }
        cohort = load_pooled_full_core_cohort(prepared)
    else:
        cohort = pooled_cohort
    if (
        cohort.aliases != ANC_ALIASES
        or cohort.total_nodes != EXPECTED_TOTAL_NODES
        or cohort.n_genes != EXPECTED_N_GENES
        or tuple(cohort.metadata_names) != _EXPECTED_METADATA_FIELDS
    ):
        raise PooledHybridCountRunnerError(
            "pooled cohort dimensions or ordered schemas changed"
        )
    configured_fingerprint = dataset.get("dataset_fingerprint")
    _require_equal(
        configured_fingerprint,
        cohort.fingerprint_sha256,
        field="dataset.dataset_fingerprint",
    )
    data_duration = time.monotonic() - data_started

    batches, graph_records, graph_duration = _prepare_core_batches(
        cohort,
        graph_config,
        uses_graph=contract.uses_graph,
        receiver_graphs=receiver_graphs,
    )
    graph_bundle_sha256 = canonical_sha256(
        [
            {
                "alias": record["alias"],
                "graph_sha256": record["graph_sha256"],
                "n_directed_edges": record["n_directed_edges"],
            }
            for record in graph_records
        ]
    )
    training_config = _training_config(config)
    mask_bundles = {
        core.alias: _mask_bundle_for_core(
            config=config, core=core, training=training_config
        )
        for core in cohort.cores
    }
    mask_bundle_sha256 = canonical_sha256(
        [
            {"alias": alias, "mask_bundle_sha256": mask_bundles[alias].checksum}
            for alias in ANC_ALIASES
        ]
    )
    per_core_references = {
        core.alias: fit_hybrid_count_references(
            core.expression_counts,
            expression_mean=cohort.expression_mean,
            expression_scale=cohort.expression_scale,
        )
        for core in cohort.cores
    }
    pooled_references = fit_equal_core_hybrid_count_references(
        {
            core.alias: core.expression_counts for core in cohort.cores
        },
        expression_mean=cohort.expression_mean,
        expression_scale=cohort.expression_scale,
        expected_aliases=ANC_ALIASES,
    )

    model, model_construction, parameter_audit = _paired_models(
        core=cohort.cores[0],
        graph=SimpleNamespace(edge_attribute_names=EDGE_ATTRIBUTE_NAMES),
        model_config=model_config,
        selected_model_name=contract.model_name,
        seed=training_config.model_seed,
    )
    parameter_audit = {
        **dict(parameter_audit),
        "encoder_initial_state_sha256": _state_dict_sha256(
            model.encoder.state_dict()
        ),
        "decoder_initial_state_sha256": _state_dict_sha256(
            model.decoder.state_dict()
        ),
        "materialization_checksum": materialization_checksum,
        "config_sha256": config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "run_attempt": run_attempt,
    }
    parameter_count = int(
        parameter_audit["trainable_parameter_count_graph"]
    )
    archive.write_json(
        "diagnostics/parameter_structure_audit.json", parameter_audit
    )
    archive.write_json(
        "diagnostics/pooled_preprocessing.json",
        {
            "schema": "pooled_full_core_preprocessing_v1",
            "aliases": list(ANC_ALIASES),
            "total_nodes": cohort.total_nodes,
            "n_genes": cohort.n_genes,
            "n_model_covariates": EXPECTED_N_MODEL_COVARIATES,
            "expression_fit_scope": "all_ten_cores_transductive",
            "expression_moment_weighting": "equal_core",
            "cohort_checksums": cohort.checksums.to_dict(),
            "per_core": {
                core.alias: {
                    "qc": core.preprocessing_qc.to_dict(),
                    "checksums": core.checksums.to_dict(),
                }
                for core in cohort.cores
            },
            "materialization_checksum": materialization_checksum,
            "config_sha256": config_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "run_attempt": run_attempt,
        },
    )
    archive.write_json(
        "diagnostics/graph_statistics.json",
        {
            "schema": "pooled_disconnected_graph_bundle_v1",
            "aliases": list(ANC_ALIASES),
            "cross_core_edges": False,
            "graph_concatenation_on_gpu": False,
            "graph_bundle_sha256": graph_bundle_sha256,
            "per_core": list(graph_records),
            "materialization_checksum": materialization_checksum,
            "config_sha256": config_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "run_attempt": run_attempt,
        },
    )
    archive.write_json(
        "diagnostics/mask_statistics.json",
        {
            "schema": "pooled_prior_mask_regeneration_v1",
            "aliases": list(ANC_ALIASES),
            "mask_bundle_sha256": mask_bundle_sha256,
            "per_core_manifests": {
                alias: mask_bundles[alias].manifest
                for alias in ANC_ALIASES
            },
            "model_seed_excluded_from_evaluation_mask_derivation": True,
            "technical_replicates_not_biological_replicates": True,
            "materialization_checksum": materialization_checksum,
            "config_sha256": config_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "run_attempt": run_attempt,
        },
    )
    archive.write_json(
        "diagnostics/raw_count_representation.json",
        {
            "schema": _REPRESENTATION_SCHEMA,
            "validation": "finite_nonnegative_integer",
            "total_nodes": cohort.total_nodes,
            "n_genes": cohort.n_genes,
            "shared_expression_mean_sha256": _array_sha256(
                "expression_mean", cohort.expression_mean
            ),
            "shared_expression_scale_sha256": _array_sha256(
                "expression_scale", cohort.expression_scale
            ),
            "mask_token_id": 8,
            "mask_token_is_output": False,
            "continuous_masked_value": 0.0,
            "thresholds_fitted": False,
            "materialization_checksum": materialization_checksum,
            "config_sha256": config_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "run_attempt": run_attempt,
        },
    )
    archive.write_json(
        "provenance/frozen_task_contract.json", frozen_verification
    )
    archive.write_json(
        "provenance/materialization.json", dict(materialization)
    )
    archive.write_json(
        "provenance/pooled_inputs.json",
        {
            "aliases": list(ANC_ALIASES),
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "fit_scope": "all_nodes_all_ten_cores_transductive",
            "generalization_estimate": False,
            "prepared_artifacts": dict(dataset["prepared_artifacts"]),
            "cohort_checksums": cohort.checksums.to_dict(),
            "graph_bundle_sha256": graph_bundle_sha256,
            "mask_bundle_sha256": mask_bundle_sha256,
            "graph_interpretation": (
                "broad_regional_context_not_direct_interaction"
            ),
            "data_preparation_duration_seconds": data_duration,
            "graph_construction_duration_seconds": graph_duration,
            "materialization_checksum": materialization_checksum,
            "config_sha256": config_sha256,
        },
    )
    archive.write_json(
        "provenance/hybrid_count_references.json",
        {
            "per_core": {
                alias: _reference_provenance(per_core_references[alias])
                for alias in ANC_ALIASES
            },
            "equal_core_pooled": _reference_provenance(pooled_references),
            "transductive_references": True,
        },
    )
    archive.write_json(
        "provenance/amp_authorization.json",
        {
            "configured": _section(config, "trainer").get(
                "amp_authorization"
            ),
            "production_pilot_receipt": pilot_receipt,
            "materialization_checksum": materialization_checksum,
            "config_sha256": config_sha256,
        },
    )

    precision_results: dict[str, PrecisionEquivalenceResult] = {}
    precision_payload: dict[str, Any] | None = None
    effective_training = training_config
    if contract.diagnostic_resource_pilot:
        for batch in batches:
            _clear_graph_layout_caches(model)
            precision_results[batch.alias] = compare_fp32_amp_loss(
                model,
                batch.view,
                _whole_node_replicate_zero_mask(
                    mask_bundles[batch.alias]
                ),
                expression_mean=cohort.expression_mean,
                expression_scale=cohort.expression_scale,
                device=training_config.device or "cuda",
                amp_dtype=training_config.amp_dtype,
                maximum_discrepancy=_PILOT_MAX_AMP_DISCREPANCY,
            )
        precision_payload = _precision_payload(
            precision_results,
            materialization_checksum=materialization_checksum,
            config_sha256=config_sha256,
        )
        archive.write_json(
            "diagnostics/fp32_amp_equivalence.json", precision_payload
        )
        if not precision_payload["all_cores_passed"]:
            effective_training = replace(training_config, amp=False)

    training_started = time.monotonic()
    training_result = fit_pooled_hybrid_count_model(
        model,
        batches,
        effective_training,
        expression_mean=cohort.expression_mean,
        expression_scale=cohort.expression_scale,
    )
    training_duration = time.monotonic() - training_started
    core_history_rows = [
        {
            "run_id": archive.run_id,
            "split": "fit",
            "training_protocol": training_result.training_protocol,
            **row,
        }
        for row in training_result.core_history_rows()
    ]
    # ``metrics/history`` is the archive-wide canonical training-history
    # contract.  The pooled-specific name is retained for explicit core-visit
    # auditing by the pilot verifier.
    archive.write_table(
        "metrics/history", core_history_rows, fallback="jsonl"
    )
    archive.write_table(
        "metrics/core_step_history",
        core_history_rows,
        fallback="jsonl",
    )
    archive.write_table(
        "metrics/global_epoch_history",
        [
            {
                "run_id": archive.run_id,
                "split": "fit",
                "training_protocol": training_result.training_protocol,
                **row,
            }
            for row in training_result.global_history_rows()
        ],
        fallback="jsonl",
    )

    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt",
        _checkpoint_bytes(
            archive=archive,
            contract=contract,
            model_config=model_config,
            model_construction=model_construction,
            parameter_audit=parameter_audit,
            training=training_result,
            cohort=cohort,
            graph_records=graph_records,
            mask_bundles=mask_bundles,
            per_core_references=per_core_references,
            pooled_references=pooled_references,
            effective_amp=effective_training.amp,
            materialization_checksum=materialization_checksum,
            config_sha256=config_sha256,
        ),
    )
    checkpoint_size = checkpoint_path.stat().st_size
    archive.write_json(
        "provenance/pooled_training.json",
        {
            "training_protocol": training_result.training_protocol,
            "graph_execution": training_result.graph_execution,
            "checkpoint_policy": training_result.checkpoint_policy,
            "final_epoch": training_result.final_epoch,
            "completed_global_epochs": (
                training_result.completed_global_epochs
            ),
            "fixed_epoch_budget": training_result.fixed_epoch_budget,
            "optimizer_steps_completed": (
                training_result.optimizer_steps_completed
            ),
            "state_dict_sha256": training_result.final_state_checksum,
            "model_seed": effective_training.model_seed,
            "epoch_mask_seed": effective_training.mask_seed,
            "core_order_seed": 271828,
            "parameter_count": parameter_count,
            "exact_parameter_match": True,
            "paired_initialization_match": True,
            "device": training_result.device,
            "model_construction": model_construction,
            "objective": _OBJECTIVE,
            "configured_amp": training_config.amp,
            "effective_amp": effective_training.amp,
            "aliases": list(ANC_ALIASES),
            "materialization_checksum": materialization_checksum,
            "config_sha256": config_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "run_attempt": run_attempt,
        },
    )

    graph_by_alias = {
        str(record["alias"]): record for record in graph_records
    }
    batch_by_alias = {batch.alias: batch for batch in batches}
    replicate_rows: list[dict[str, Any]] = []
    evaluation_started = time.monotonic()

    def prediction_rows() -> Iterator[dict[str, Any]]:
        for alias in ANC_ALIASES:
            batch = batch_by_alias[alias]
            bundle = mask_bundles[alias]
            for entry in bundle.manifest["entries"]:
                _clear_graph_layout_caches(model)
                result = evaluate_fixed_hybrid_count_mask(
                    model,
                    batch.view,
                    bundle.masks[str(entry["entry_id"])],
                    expression_mean=cohort.expression_mean,
                    expression_scale=cohort.expression_scale,
                    references=per_core_references[alias],
                    device=effective_training.device,
                    amp=effective_training.amp,
                    amp_dtype=effective_training.amp_dtype,
                )
                metrics = _required_metric_aliases(
                    result.evaluation.metrics
                )
                metrics.update(
                    _reference_metrics(
                        result,
                        pooled_references,
                        expression_mean=cohort.expression_mean,
                        expression_scale=cohort.expression_scale,
                        prefix="reference_equal_core",
                    )
                )
                row = {
                    "biological_unit_alias": alias,
                    **_replicate_metric_row(entry=entry, metrics=metrics),
                }
                replicate_rows.append(row)
                mode = str(row["mask_mode"])
                for name, value in row.items():
                    if name in _ROW_METADATA_FIELDS or not _finite_scalar(
                        value
                    ):
                        continue
                    archive.append_metric_event(
                        {
                            "name": f"fit/{alias}/{mode}/{name}",
                            "value": value,
                            "mask_replicate": int(entry["replicate"]),
                            "mask_seed": int(entry["seed"]),
                        }
                    )
                if mode == "whole_node" and int(entry["replicate"]) == 0:
                    graph_record = graph_by_alias[alias]
                    graph_id = (
                        "exact_mutual_k1000_"
                        f"{str(graph_record['graph_sha256'])[:16]}"
                        if contract.uses_graph
                        else "self_only_paired_graph_"
                        f"{str(graph_record['graph_sha256'])[:16]}"
                    )
                    yield from _protected_prediction_rows(
                        archive=archive,
                        dataset=dataset,
                        alias=alias,
                        graph_id=graph_id,
                        edge_count=(
                            int(graph_record["n_directed_edges"])
                            if contract.uses_graph
                            else 0
                        ),
                        fold=int(config.get("fold", 0)),
                        entry=entry,
                        result=result,
                        sample_key_salt=sample_key_salt,
                    )

    prediction_path = archive.write_prediction_jsonl_stream(
        "fit", prediction_rows()
    )
    evaluation_duration = time.monotonic() - evaluation_started
    archive.write_table(
        "metrics/evaluation_replicates",
        replicate_rows,
        fallback="jsonl",
    )
    final_metrics = _aggregate_metrics(
        replicate_rows,
        replicates_per_mode=int(
            evaluation_config["mask_replicates_per_mode"]
        ),
    )
    precision_peak = max(
        (
            result.peak_cuda_memory_bytes
            for result in precision_results.values()
        ),
        default=0,
    )
    training_peak = max(
        (
            record.peak_cuda_memory_bytes
            for record in training_result.core_history
        ),
        default=0,
    )
    peak_vram_bytes = max(
        precision_peak,
        training_peak,
        _cuda_peak_bytes(training_result.device),
    )
    total_duration = time.monotonic() - started
    final_metrics.update(
        {
            "resource/data_preparation_duration_seconds": data_duration,
            "resource/graph_construction_duration_seconds": graph_duration,
            "resource/training_duration_seconds": training_duration,
            "resource/inference_duration_seconds": evaluation_duration,
            "resource/total_duration_seconds": total_duration,
            "resource/parameter_count": parameter_count,
            "resource/checkpoint_size_bytes": checkpoint_size,
            "resource/peak_allocated_vram_bytes": peak_vram_bytes,
            "resource/peak_vram_gib": peak_vram_bytes / 1024**3,
            "training/final_epoch": training_result.final_epoch,
            "training/completed_global_epochs": (
                training_result.completed_global_epochs
            ),
            "training/optimizer_steps_completed": (
                training_result.optimizer_steps_completed
            ),
            "training/final_hybrid_loss": (
                training_result.final_train_loss
            ),
        }
    )
    primary_value = final_metrics.get(_PRIMARY_METRIC)
    if not _finite_scalar(primary_value):
        raise PooledHybridCountRunnerError(
            "equal-core primary hybrid loss is missing or non-finite"
        )
    for name, value in final_metrics.items():
        if _finite_scalar(value):
            archive.append_metric_event(
                {"name": name, "value": value, "phase": "final_aggregate"}
            )
    archive.write_json("metrics/final.json", final_metrics)

    global_losses = np.asarray(
        [
            record.mean_hybrid_loss
            for record in training_result.global_history
        ],
        dtype=np.float64,
    )
    tail = global_losses[-min(20, len(global_losses)) :]
    convergence_slope = (
        float(
            np.polyfit(
                np.arange(len(tail), dtype=np.float64), tail, 1
            )[0]
        )
        if len(tail) >= 2
        else None
    )
    convergence = {
        "schema": "pooled_hybrid_count_convergence_v1",
        "objective": _OBJECTIVE,
        "final_epoch": training_result.final_epoch,
        "completed_global_epochs": training_result.completed_global_epochs,
        "optimizer_steps_completed": (
            training_result.optimizer_steps_completed
        ),
        "final_equal_core_train_hybrid_loss": (
            training_result.final_train_loss
        ),
        "minimum_observed_global_train_hybrid_loss": float(
            global_losses.min()
        ),
        "last_20_global_epoch_loss_slope": convergence_slope,
        "all_global_epochs_completed": len(
            training_result.global_history
        )
        == training_result.fixed_epoch_budget,
        "all_epochs_completed": len(training_result.global_history)
        == training_result.fixed_epoch_budget,
        "all_losses_and_gradients_finite": True,
        "every_core_once_each_epoch": all(
            record.cores_visited == 10
            and set(record.ordered_aliases) == set(ANC_ALIASES)
            for record in training_result.global_history
        ),
        "materialization_checksum": materialization_checksum,
        "config_sha256": config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "run_attempt": run_attempt,
    }
    archive.write_json(
        "diagnostics/training_convergence.json", convergence
    )
    resource_diagnostic = _resource_diagnostic(
        archive=archive,
        contract=contract,
        configured_training=training_config,
        effective_training=effective_training,
        precision_payload=precision_payload,
        training=training_result,
        total_duration=total_duration,
        training_duration=training_duration,
        evaluation_duration=evaluation_duration,
        peak_vram_bytes=peak_vram_bytes,
        parameter_audit=parameter_audit,
        materialization_checksum=materialization_checksum,
        config_sha256=config_sha256,
    )
    archive.write_json(
        "diagnostics/resource_usage.json", resource_diagnostic
    )

    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "training_exit_status": "success",
        "campaign_id": _CAMPAIGN_ID,
        "aliases": list(ANC_ALIASES),
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "evaluation_protocol": _PROTOCOL,
        "task_family": _TASK_FAMILY,
        "canonical_prediction_split": "fit",
        "model_name": contract.model_name,
        "public_variant": contract.public_variant,
        "model_seed": contract.seed,
        "final_epoch": training_result.final_epoch,
        "completed_global_epochs": training_result.completed_global_epochs,
        "fixed_epoch_budget": training_result.fixed_epoch_budget,
        "optimizer_steps_completed": (
            training_result.optimizer_steps_completed
        ),
        "checkpoint_role": "last",
        "checkpoint": {
            "role": "last",
            "final_epoch": training_result.final_epoch,
            "policy": "final_global_epoch_no_validation_selection",
            "monitored_metric": None,
            "state_dict_sha256": training_result.final_state_checksum,
        },
        "primary_metric_name": _PRIMARY_METRIC,
        "primary_metric_value": float(primary_value),
        "metrics": final_metrics,
        "parameter_count": parameter_count,
        "exact_parameter_match": True,
        "paired_initialization_match": True,
        "duration_seconds": total_duration,
        **_summary_peak_vram_fields(peak_vram_bytes),
        "peak_host_memory_bytes": _peak_host_memory_bytes(),
        "graph_bundle_sha256": graph_bundle_sha256,
        "graph_supplied_to_model": contract.uses_graph,
        "graph_interpretation": (
            "broad_regional_context_not_direct_interaction"
        ),
        "evaluation_mask_bundle_sha256": mask_bundle_sha256,
        "evaluation_mask_replicates_per_mode": int(
            evaluation_config["mask_replicates_per_mode"]
        ),
        "evaluation_metrics_first_average_replicates_then_cores": True,
        "canonical_prediction_selection": {
            "split": "fit",
            "aliases": list(ANC_ALIASES),
            "mask_mode": "whole_node",
            "mask_replicate": 0,
            "selection_status": "prespecified",
        },
        "diagnostic_resource_pilot": (
            contract.diagnostic_resource_pilot
        ),
        "pilot_gate_passed": resource_diagnostic["pilot_gate_passed"],
        "conclusion_eligible": not contract.diagnostic_resource_pilot,
        "generalization_estimate": False,
        "materialization_checksum": materialization_checksum,
        "config_sha256": config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "run_attempt": run_attempt,
        "maximum_claim": (
            "diagnostic pooled runtime, memory, and precision feasibility only"
            if contract.diagnostic_resource_pilot
            else (
                "held-in masked-expression capacity of one shared model "
                "across ten adjacent-normal cores and descriptive "
                "broad-context graph gain"
            )
        ),
        "prohibited_claims": [
            "patient-held-out generalization",
            "core-held-out generalization",
            "true-Normal performance",
            "direct cellular interaction",
            "biological mechanism",
            "causality",
        ],
    }
    archive.write_summary(summary)
    return CapacityRunResult(
        run_id=archive.run_id,
        model_name=contract.model_name,
        primary_metric_name=_PRIMARY_METRIC,
        primary_metric_value=float(primary_value),
        final_epoch=training_result.final_epoch,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one worker-owned pooled ten-core hybrid-count member."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    result = run_pooled_hybrid_count_capacity(
        config,
        archive,
        sample_key_salt=os.environ.get("BAGM_SAMPLE_KEY_SALT", ""),
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "model_name": result.model_name,
                "primary_metric_name": result.primary_metric_name,
                "primary_metric_value": result.primary_metric_value,
                "final_epoch": result.final_epoch,
                "checkpoint": str(result.checkpoint_path),
                "predictions": str(result.prediction_path),
                "pilot_gate_passed": result.summary.get(
                    "pilot_gate_passed"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
