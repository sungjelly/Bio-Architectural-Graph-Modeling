#!/usr/bin/env python3
"""Run one worker-owned multiscale continuous-hurdle capacity experiment.

The runner is fail-closed: it accepts only checksum-bound configs materialized
for the frozen campaign, rebuilds and verifies every graph scale, audits all
four parameter-matched routing arms, and writes only held-in fit artifacts to
the queue worker's active :class:`RunArchive`.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
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
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT))
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from scripts.train.run_full_core_capacity import (  # noqa: E402
    CapacityRunResult,
    _evaluation_masks,
    _mean_finite,
    _peak_host_memory_bytes,
    _resolve_prepared_artifact,
    _training_config,
    _verify_materialized_identity,
    _worker_archive_and_config,
)
from spatial_benchmark.full_core import (  # noqa: E402
    ALLOWED_METADATA_COLUMNS,
    EDGE_ATTRIBUTE_NAMES,
    FullCoreData,
    load_and_refit_full_core,
)
from spatial_benchmark.hurdle_continuous import (  # noqa: E402
    evaluate_hurdle_continuous_output,
)
from spatial_benchmark.hybrid_count import (  # noqa: E402
    tokenize_raw_counts,
    validate_raw_counts,
)
from spatial_benchmark.hybrid_count_metrics import (  # noqa: E402
    json_safe_metrics,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.local_source_permutation import (  # noqa: E402
    LocalSourcePermutation,
    LocalSourcePermutationError,
    build_macroblock_spatial_antipode_permutation,
    verify_local_source_permutation_receipt,
)
from spatial_benchmark.multiscale_hurdle_contract import (  # noqa: E402
    ACTIVE_CONTRACT_AMENDMENT_RELATIVE,
    ACTIVE_CONTRACT_AMENDMENT_SHA256,
    ARMS,
    CAMPAIGN_ID,
    FROZEN_CONTRACT_RELATIVE,
    FROZEN_CONTRACT_SHA256,
    LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM,
    LOCAL_SOURCE_PERMUTATION_SCHEMA,
    REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE,
    REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
    ROUTING,
    SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE,
    SUPERSEDED_CONTRACT_AMENDMENT_SHA256,
)
from spatial_benchmark.multiscale_graphs import (  # noqa: E402
    LOCAL_K_CAP,
    LOCAL_MAX_DISTANCE_UM,
    REGIONAL_K_CAP,
    REGIONAL_MAX_DISTANCE_UM,
    REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM,
    ReceiverSortedScaleGraph,
    TrueMultiscaleGraphs,
    build_true_multiscale_graphs,
    true_graph_receipt,
)
from spatial_benchmark.multiscale_hurdle_training import (  # noqa: E402
    MultiscaleGraphSplitView,
    MultiscaleHurdleTrainingResult,
    MultiscalePrecisionEquivalenceResult,
    compare_multiscale_fp32_amp_loss,
    evaluate_fixed_multiscale_hurdle_mask,
    fit_full_core_multiscale_hurdle_model,
)
from spatial_benchmark.multiscale_hybrid import (  # noqa: E402
    MultiscaleAdditiveHybridModel,
    trainable_parameter_count,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    RunValidationError,
    deidentify_prediction_rows,
)
from spatial_benchmark.training import (  # noqa: E402
    set_deterministic_seed,
)


_CAMPAIGN_ID = CAMPAIGN_ID
_SOURCE_CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
_FROZEN_CONTRACT_SHA256 = FROZEN_CONTRACT_SHA256
_FROZEN_CONTRACT_RELATIVE = FROZEN_CONTRACT_RELATIVE
_FROZEN_CONTRACT_HASH_RELATIVE = _FROZEN_CONTRACT_RELATIVE.with_suffix(
    ".sha256"
)
_CONTRACT_AMENDMENT_RELATIVE = ACTIVE_CONTRACT_AMENDMENT_RELATIVE
_CONTRACT_AMENDMENT_SHA256 = ACTIVE_CONTRACT_AMENDMENT_SHA256
_CONTRACT_SUPPLEMENT_RELATIVE = REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE
_CONTRACT_SUPPLEMENT_SHA256 = REQUIRED_CONTRACT_SUPPLEMENT_SHA256
_LOCKED_ROOT_RELATIVE = Path("scratch/locked_campaigns") / _CAMPAIGN_ID
_MATERIALIZATION_RELATIVE = (
    _LOCKED_ROOT_RELATIVE / "locked_config_materialization.json"
)
_RESOURCE_GATE_RELATIVE = _LOCKED_ROOT_RELATIVE / "resource_gate_receipt.json"
_REPRESENTATION_GATE_RELATIVE = (
    _LOCKED_ROOT_RELATIVE / "representation_gate_receipt.json"
)
_MATERIALIZATION_KIND = "multiscale_hurdle_locked_config_materialization_v1"
_RESOURCE_GATE_KIND = "multiscale_hurdle_resource_gate_v1"
_REPRESENTATION_GATE_KIND = "multiscale_hurdle_representation_gate_v1"
_PROTOCOL = "held_in_full_core_fixed_budget"
_TASK_FAMILY = "masked_expression_hurdle_count"
_PRIMARY_METRIC = "fit/whole_node/hurdle_loss"
_REPRESENTATION_SCHEMA = (
    "hurdle_detection_plus_positive_standardized_log1p_v1"
)
_INPUT_STATE_SCHEMA = (
    "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1"
)
_OBJECTIVE = "equal_weight_balanced_detection_positive_huber"
_EXPECTED_PARAMETER_COUNT = 7_559_184
_TARGET_NODE_BATCH_SIZE = 16_384
_PILOT_MAX_VRAM_GIB = 12.0
_PILOT_MAX_AMP_DISCREPANCY = 1e-3
_PILOT_MAX_PROJECTED_HOURS = 0.5
_DISK_MAX_USED_DECIMAL_GB = 55.0
_RESOURCE_GATE_THRESHOLDS = {
    "stage1_peak_allocated_vram_gib_maximum": _PILOT_MAX_VRAM_GIB,
    "fp32_amp_absolute_loss_discrepancy_maximum": (
        _PILOT_MAX_AMP_DISCREPANCY
    ),
    "stage1_projected_gpu_hours_per_200_epoch_core_maximum": (
        _PILOT_MAX_PROJECTED_HOURS
    ),
    "filesystem_used_decimal_gb_hard_stop": _DISK_MAX_USED_DECIMAL_GB,
}
_REPRESENTATION_GATE_THRESHOLDS = {
    "minimum_relative_improvement": 0.02,
    "required_core_count": 2,
    "required_core_total": 2,
}
_SAFE_GPU_IDS = (0, 1, 2, 3, 5, 6, 7)
_PILOT_ALIASES = ("ANC-03", "ANC-05")
_SCIENCE_ALIASES = ("ANC-02", "ANC-03", "ANC-05", "ANC-06", "ANC-09")
_ARMS = ARMS
_ROUTING = ROUTING
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
_ROW_METADATA_FIELDS = {
    "split",
    "mask_mode",
    "mask_replicate",
    "mask_entry_id",
    "mask_seed",
    "mask_checksum",
}
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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ALIAS = re.compile(r"ANC-(?:0[1-9]|10)\Z")
_PERCENT_FIELDS = (
    "detection_balanced_accuracy",
    "detection_sensitivity",
    "detection_specificity",
    "detection_precision",
    "state8_exact_accuracy",
    "state8_balanced_accuracy",
    "positive_count_state_exact_accuracy",
    "positive_count_state_within_one_accuracy",
)


class MultiscaleHurdleRunnerError(RuntimeError):
    """Raised before a run can violate the frozen campaign contract."""


@dataclass(frozen=True)
class MultiscaleHurdleContract:
    arm: str
    biological_unit_alias: str
    resource_pilot: bool
    regional_routing: str
    local_routing: str

    @property
    def uses_permuted_local(self) -> bool:
        return self.local_routing == "permuted"

    @property
    def requires_representation_gate(self) -> bool:
        return not self.resource_pilot and self.arm != "self"


@dataclass(frozen=True)
class HurdlePerGeneReferences:
    detection_probability: np.ndarray
    positive_continuous_standardized: np.ndarray
    audit: Mapping[str, Any]


def _mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise MultiscaleHurdleRunnerError(
            f"resolved configuration requires a {name!r} mapping"
        )
    return value


def _require_equal(
    actual: Any,
    expected: Any,
    *,
    field: str,
) -> None:
    if actual != expected:
        raise MultiscaleHurdleRunnerError(
            f"{field} must be {expected!r}; got {actual!r}"
        )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _assert_alias_only_mapping(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _normalized_key(raw_key)
            child_path = f"{path}.{raw_key}"
            if key in _FORBIDDEN_IDENTIFIER_KEYS:
                raise MultiscaleHurdleRunnerError(
                    f"alias-only configuration prohibits {child_path}"
                )
            _assert_alias_only_mapping(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_alias_only_mapping(child, path=f"{path}[{index}]")


def _validate_count_representation(dataset: Mapping[str, Any]) -> None:
    value = dataset.get("count_representation")
    if not isinstance(value, Mapping):
        raise MultiscaleHurdleRunnerError(
            "dataset.count_representation is required"
        )
    expected = {
        "schema": _REPRESENTATION_SCHEMA,
        "source_scale": "raw_biological_probe_counts",
        "output_channels_per_gene": 2,
        "output_channels": [
            "detection_logit",
            "positive_standardized_log1p",
        ],
        "detection_threshold": 0.5,
        "count_rounding": "nonnegative_half_up_floor_x_plus_0_5",
        "fixed_count_states": [
            "0",
            "1",
            "2",
            "3",
            "4-7",
            "8-15",
            "16-31",
            "32+",
        ],
        "continuous_transform": "per_gene_all_fit_standardized_log1p",
        "fit_required": False,
    }
    for field, required in expected.items():
        _require_equal(
            value.get(field),
            required,
            field=f"dataset.count_representation.{field}",
        )
    input_states = value.get("input_states")
    if not isinstance(input_states, Mapping):
        raise MultiscaleHurdleRunnerError(
            "dataset.count_representation.input_states is required"
        )
    for field, required in {
        "schema": _INPUT_STATE_SCHEMA,
        "num_states": 8,
        "mask_token_id": 8,
        "mask_token_is_output": False,
    }.items():
        _require_equal(
            input_states.get(field),
            required,
            field=f"dataset.count_representation.input_states.{field}",
        )


def _validate_multiscale_contract(
    config: Mapping[str, Any],
) -> MultiscaleHurdleContract:
    """Validate the scientific and execution settings before any GPU mutation."""

    _assert_alias_only_mapping(config)
    campaign = _mapping(config, "campaign")
    experiment = _mapping(config, "experiment")
    model = _mapping(config, "model")
    dataset = _mapping(config, "dataset")
    features = _mapping(config, "features")
    graph = _mapping(config, "graph")
    masking = _mapping(config, "masking")
    trainer = _mapping(config, "trainer")
    evaluation = _mapping(config, "evaluation")
    launcher = _mapping(config, "launcher")
    metadata = _mapping(config, "metadata")

    for field, required in {
        "campaign_id": _CAMPAIGN_ID,
        "exploratory": True,
        "frozen_contract": _FROZEN_CONTRACT_RELATIVE.as_posix(),
        "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
        "contract_amendment": _CONTRACT_AMENDMENT_RELATIVE.as_posix(),
        "contract_amendment_sha256": _CONTRACT_AMENDMENT_SHA256,
        "contract_supplement": _CONTRACT_SUPPLEMENT_RELATIVE.as_posix(),
        "contract_supplement_sha256": _CONTRACT_SUPPLEMENT_SHA256,
        "superseded_contract_amendment": (
            SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.as_posix()
        ),
        "superseded_contract_amendment_sha256": (
            SUPERSEDED_CONTRACT_AMENDMENT_SHA256
        ),
        "superseded_amendment_retained_as_negative_record": True,
    }.items():
        _require_equal(campaign.get(field), required, field=f"campaign.{field}")
    if int(config.get("seed", -1)) != 0:
        raise MultiscaleHurdleRunnerError("the frozen campaign requires seed=0")
    if int(config.get("fold", -1)) != 0 or int(
        config.get("attempt", -1)
    ) not in {1, 2}:
        raise MultiscaleHurdleRunnerError(
            "the frozen run identity requires fold=0 and attempt in {1,2}"
        )

    arm = str(experiment.get("arm", ""))
    if arm not in _ARMS:
        raise MultiscaleHurdleRunnerError(f"unsupported frozen arm {arm!r}")
    resource_pilot = experiment.get("resource_pilot") is True
    alias = str(experiment.get("biological_unit_alias", ""))
    if not _ALIAS.fullmatch(alias):
        raise MultiscaleHurdleRunnerError(
            "experiment.biological_unit_alias must be an opaque ANC alias"
        )
    allowed_aliases = _PILOT_ALIASES if resource_pilot else _SCIENCE_ALIASES
    if alias not in allowed_aliases:
        raise MultiscaleHurdleRunnerError(
            f"{alias} is not prespecified for this execution stage"
        )
    expected_stage = 1 if resource_pilot or arm == "self" else 2
    expected_role = "resource_pilot" if resource_pilot else "science"
    expected_variant = (
        f"{alias.lower().replace('-', '')}_"
        f"{arm.replace('-', '_')}_{expected_role}"
    )
    for field, required in {
        "variant_label": expected_variant,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "estimand": "held_in_full_core_whole_node_masked_raw_count",
        "permitted_claim": (
            "exploratory_held_in_graph_specific_and_correctly_aligned_"
            "local_sender_state_predictive_dependency"
        ),
        "paired_within_core": True,
        "conclusion_eligible": not resource_pilot,
        "excluded_from_primary_comparison": resource_pilot,
        "stage": expected_stage,
    }.items():
        _require_equal(
            experiment.get(field), required, field=f"experiment.{field}"
        )

    regional_routing, local_routing = _ROUTING[arm]
    fixed_model = {
        "name": "multiscale-hurdle-count",
        "family": "additive_multiscale_hurdle_count",
        "count_representation_schema": _REPRESENTATION_SCHEMA,
        "output_channels_per_gene": 2,
        "embedding_dim": 512,
        "hidden_dim": 512,
        "decoder_dim": 512,
        "ffn_dim": 512,
        "attention_heads": 4,
        "attention_head_dim": 32,
        "value_head_dim": 16,
        "message_dim": 64,
        "local_edge_attribute_dim": len(EDGE_ATTRIBUTE_NAMES),
        "regional_edge_attribute_dim": len(EDGE_ATTRIBUTE_NAMES),
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 32,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "receiver_chunk_size": 128,
        "activation_checkpointing": True,
        "regional_routing": regional_routing,
        "local_routing": local_routing,
        "regional_output_zero_initialized": True,
        "local_output_zero_initialized": True,
        "exact_additive_decomposition": True,
        "trainable_node_identifiers": False,
        "trainable_edge_identifiers": False,
        "uses_graph_inputs": arm != "self",
        "uses_edge_inputs": arm != "self",
        "parameter_match_group": "multiscale_hurdle_v1",
    }
    for field, required in fixed_model.items():
        _require_equal(model.get(field), required, field=f"model.{field}")

    for field, required in {
        "task": _TASK_FAMILY,
        "target_scale": (
            "raw_biological_probe_counts_with_per_gene_all_fit_"
            "standardized_log1p"
        ),
        "biological_target_count": 1000,
        "biological_unit_alias": alias,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "preprocessing_fit_scope": "all_nodes_transductive",
        "validation_or_test_partition_present": False,
        "frozen_task_contract_sha256": _FROZEN_CONTRACT_SHA256,
    }.items():
        _require_equal(dataset.get(field), required, field=f"dataset.{field}")
    if dataset.get("patient_generalization_supported") not in {None, False}:
        raise MultiscaleHurdleRunnerError(
            "dataset.patient_generalization_supported must be false"
        )
    prepared = dataset.get("prepared_artifact_reference")
    if (
        not isinstance(prepared, str)
        or Path(prepared).is_absolute()
        or ".." in Path(prepared).parts
        or alias.lower() not in Path(prepared).parts
    ):
        raise MultiscaleHurdleRunnerError(
            "prepared artifact must be alias-safe and project-relative"
        )
    _validate_count_representation(dataset)

    _require_equal(
        features.get("fit_scope"),
        "all_nodes_transductive",
        field="features.fit_scope",
    )
    _require_equal(
        features.get("use_edge_features"),
        True,
        field="features.use_edge_features",
    )
    node_expression = features.get("node_expression")
    if not isinstance(node_expression, Mapping):
        raise MultiscaleHurdleRunnerError(
            "features.node_expression is required"
        )
    for field, required in {
        "biological_targets": 1000,
        "source_scale": "raw_biological_probe_counts",
        "discrete_transform": "fixed_hybrid_count_states",
        "continuous_transform": "per_gene_all_fit_standardized_log1p",
        "masked_discrete_value": "input_only_mask_token_8",
        "masked_continuous_value": 0.0,
        "explicit_mask_authoritative_inside_model": True,
        "prediction_schema": "detection_plus_positive_standardized_log1p",
    }.items():
        _require_equal(
            node_expression.get(field),
            required,
            field=f"features.node_expression.{field}",
        )
    node_metadata = features.get("node_metadata")
    if not isinstance(node_metadata, Mapping) or tuple(
        node_metadata.get("fields", ())
    ) != tuple(ALLOWED_METADATA_COLUMNS):
        raise MultiscaleHurdleRunnerError(
            "features.node_metadata.fields differ from the permitted schema"
        )
    edge_features = features.get("edge_features")
    if not isinstance(edge_features, Mapping) or tuple(
        edge_features.get("fields", ())
    ) != tuple(EDGE_ATTRIBUTE_NAMES):
        raise MultiscaleHurdleRunnerError(
            "features.edge_features must contain the frozen 17 fields"
        )
    _require_equal(
        edge_features.get("fit_scope"),
        "true_local_and_regional_edges_transductive",
        field="features.edge_features.fit_scope",
    )
    _require_equal(
        edge_features.get("standardization"),
        "per_scale_true_graph_edge_wise",
        field="features.edge_features.standardization",
    )
    prohibited = set(features.get("prohibited_node_inputs", ()))
    if not {
        "direct_identifiers",
        "absolute_or_local_coordinates",
        "expression_derived_library_size",
        "rna_derived_qc",
        "vendor_cell_type_cluster_neighborhood_or_niche",
        "hidden_target_values",
    }.issubset(prohibited):
        raise MultiscaleHurdleRunnerError(
            "features.prohibited_node_inputs omits a leakage guard"
        )

    fixed_graph = {
        "kind": "multiscale_exact_mutual_knn",
        "neighbor_k": 64,
        "k": 64,
        "radius_um": 75.0,
        "radius_guard_um": 75.0,
        "symmetry": "mutual",
        "min_distance_um": 0.0,
        "edge_dropout": 0.0,
        "self_loops": False,
        "full_core_graph": True,
        "coordinates_are_node_covariates": False,
    }
    for field, required in fixed_graph.items():
        _require_equal(graph.get(field), required, field=f"graph.{field}")
    for scale, required in {
        "local": {
            "neighbor_k_cap": LOCAL_K_CAP,
            "min_distance_um": 0.0,
            "max_distance_um": LOCAL_MAX_DISTANCE_UM,
            "symmetry": "mutual",
        },
        "regional": {
            "neighbor_k_cap": REGIONAL_K_CAP,
            "min_distance_um_exclusive": (
                REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
            ),
            "max_distance_um": REGIONAL_MAX_DISTANCE_UM,
            "symmetry": "mutual",
        },
    }.items():
        section = graph.get(scale)
        if not isinstance(section, Mapping):
            raise MultiscaleHurdleRunnerError(f"graph.{scale} is required")
        for field, expected in required.items():
            _require_equal(
                section.get(field),
                expected,
                field=f"graph.{scale}.{field}",
            )
        if not _is_sha256(section.get("expected_graph_sha256")):
            raise MultiscaleHurdleRunnerError(
                f"graph.{scale}.expected_graph_sha256 is required"
            )
        for field in (
            "expected_directed_edges",
            "expected_components",
            "expected_isolated_nodes",
        ):
            value = section.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise MultiscaleHurdleRunnerError(
                    f"graph.{scale}.{field} must be nonnegative integer"
                )
    if "rewired_local" in graph:
        raise MultiscaleHurdleRunnerError(
            "graph.rewired_local is prohibited by amendment002"
        )
    permutation = graph.get("local_source_permutation")
    if not isinstance(permutation, Mapping):
        raise MultiscaleHurdleRunnerError(
            "graph.local_source_permutation is required"
        )
    try:
        verify_local_source_permutation_receipt(permutation)
    except LocalSourcePermutationError as exc:
        raise MultiscaleHurdleRunnerError(
            "graph.local_source_permutation is invalid"
        ) from exc
    for field in (
        "expected_materialized_graph_sha256",
        "expected_graph_receipt_sha256",
    ):
        if not _is_sha256(graph.get(field)):
            raise MultiscaleHurdleRunnerError(f"graph.{field} is required")
    if not isinstance(graph.get("expected_bundle_qc"), Mapping):
        raise MultiscaleHurdleRunnerError(
            "graph.expected_bundle_qc is required"
        )

    for field, required in {
        "type": "mixed_expression_masking",
        "curriculum": "P+N+B",
        "warmup_epochs": 10,
        "block_shape": "disk",
        "block_width_um": None,
        "mask_seed": 314159,
        "validation_replicates": 0,
        "test_replicates": 0,
        "mask_expression_only": True,
        "explicit_gene_mask_channel": True,
    }.items():
        _require_equal(masking.get(field), required, field=f"masking.{field}")
    expected_rates = {
        "partial_gene": 0.2,
        "whole_node": 0.1,
        "spatial_block": 0.1,
    }
    for field in ("rate", "rates"):
        value = masking.get(field)
        if not isinstance(value, Mapping) or dict(value) != expected_rates:
            raise MultiscaleHurdleRunnerError(
                f"masking.{field} differs from the frozen rates"
            )
    _require_equal(
        masking.get("post_warmup_probabilities"),
        {"partial_gene": 0.6, "whole_node": 0.3, "spatial_block": 0.1},
        field="masking.post_warmup_probabilities",
    )

    expected_epochs = 2 if resource_pilot else 200
    expected_replicates = 1 if resource_pilot else 3
    for field, required in {
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "batch_size": 1,
        "batch_unit": "complete_full_core_graph",
        "target_node_batch_size": _TARGET_NODE_BATCH_SIZE,
        "neighbor_sampling": False,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "huber_delta": 1.0,
        "max_epochs": expected_epochs,
        "fixed_epoch_budget": True,
        "early_stopping": False,
        "early_stopping_patience": None,
        "validation_every": None,
        "precision": "mixed",
        "amp": True,
        "amp_dtype": "auto",
        "amp_requires_fp32_equivalence_smoke": True,
        "deterministic": True,
        "deterministic_warn_only": False,
        "restore_best": False,
        "monitored_metric": None,
        "primary_checkpoint_role": "last",
        "checkpoint_policy": "last_only",
        "graph_execution": "full_core_exact_receiver_chunked_no_sampling",
        "objective": _OBJECTIVE,
    }.items():
        _require_equal(trainer.get(field), required, field=f"trainer.{field}")
    authorization = trainer.get("amp_authorization")
    expected_authorization = (
        {
            "mode": "runner_internal_same_batch_equivalence",
            "maximum_absolute_loss_discrepancy": 0.001,
        }
        if resource_pilot
        else {
            "mode": "require_external_resource_gate_receipt",
            "receipt_schema": _RESOURCE_GATE_KIND,
            "receipt_reference": _RESOURCE_GATE_RELATIVE.as_posix(),
            "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
        }
    )
    if not isinstance(authorization, Mapping) or dict(
        authorization
    ) != expected_authorization:
        raise MultiscaleHurdleRunnerError(
            "trainer.amp_authorization differs from the frozen stage"
        )

    for field, required in {
        "task_family": _TASK_FAMILY,
        "protocol": _PROTOCOL,
        "canonical_prediction_split": "fit",
        "primary_metric": _PRIMARY_METRIC,
        "primary_direction": "minimize",
        "splits": ["fit"],
        "mask_modes": list(_REQUIRED_PUBLIC_MASKS),
        "mask_replicates_per_mode": expected_replicates,
        "fixed_mask_bundle": True,
        "generalization_estimate": False,
        "validation_or_test_selection": False,
        "conclusion_bearing": not resource_pilot,
        "diagnostic_only": resource_pilot,
        "save_per_mask_numeric_data": True,
    }.items():
        _require_equal(
            evaluation.get(field), required, field=f"evaluation.{field}"
        )
    _require_equal(
        masking.get("fit_replicates"),
        expected_replicates,
        field="masking.fit_replicates",
    )

    requested_gpu = str(launcher.get("requested_gpu", ""))
    if requested_gpu not in {str(value) for value in _SAFE_GPU_IDS}:
        raise MultiscaleHurdleRunnerError(
            "launcher.requested_gpu is not a safe campaign device"
        )
    for field, required in {
        "requested_gpu_count": 1,
        "set_cuda_visible_devices": True,
        "concurrency": 1,
        "disk_safety_max_used_decimal_gb": _DISK_MAX_USED_DECIMAL_GB,
    }.items():
        _require_equal(launcher.get(field), required, field=f"launcher.{field}")
    for field, required in {
        "locked_config_materialization_receipt": (
            _MATERIALIZATION_RELATIVE.as_posix()
        ),
        "frozen_scientific_contract": True,
        "sender_state_permutation_amendment_enforced": True,
        "mask_noninterference_supplement_enforced": True,
        "original_rewired_arm_authorized": False,
        "execution_role": expected_role,
        "production_requires_resource_gate": not resource_pilot,
        "stage2_requires_representation_gate": (
            not resource_pilot and arm != "self"
        ),
    }.items():
        _require_equal(metadata.get(field), required, field=f"metadata.{field}")
    return MultiscaleHurdleContract(
        arm=arm,
        biological_unit_alias=alias,
        resource_pilot=resource_pilot,
        regional_routing=regional_routing,
        local_routing=local_routing,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_json_receipt(
    path: Path,
    *,
    kind: str,
) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise MultiscaleHurdleRunnerError(
            f"required {kind} receipt is missing or unsafe"
        )
    def reject_constant(value: str) -> None:
        raise MultiscaleHurdleRunnerError(
            f"{kind} receipt contains non-finite JSON constant {value!r}"
        )

    def unique_object(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MultiscaleHurdleRunnerError(
                    f"{kind} receipt contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except MultiscaleHurdleRunnerError:
        raise
    except (OSError, ValueError) as error:
        raise MultiscaleHurdleRunnerError(
            f"{kind} receipt is unreadable"
        ) from error
    if not isinstance(payload, dict):
        raise MultiscaleHurdleRunnerError(
            f"{kind} receipt must be a JSON mapping"
        )
    core = dict(payload)
    checksum = core.pop("checksum", None)
    if not isinstance(checksum, str) or checksum != canonical_sha256(core):
        raise MultiscaleHurdleRunnerError(
            f"{kind} receipt checksum is invalid"
        )
    return payload, checksum


def _verify_frozen_contract(project_root: Path) -> Mapping[str, Any]:
    contract = project_root / _FROZEN_CONTRACT_RELATIVE
    sidecar = project_root / _FROZEN_CONTRACT_HASH_RELATIVE
    if (
        not contract.is_file()
        or contract.is_symlink()
        or not sidecar.is_file()
        or sidecar.is_symlink()
    ):
        raise MultiscaleHurdleRunnerError(
            "frozen task contract or checksum sidecar is missing or unsafe"
        )
    observed = _sha256_file(contract)
    declared = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if observed != _FROZEN_CONTRACT_SHA256 or declared != observed:
        raise MultiscaleHurdleRunnerError(
            "frozen task contract checksum changed or disagrees"
        )
    amendment = project_root / _CONTRACT_AMENDMENT_RELATIVE
    amendment_sidecar = amendment.with_suffix(".sha256")
    if (
        not amendment.is_file()
        or amendment.is_symlink()
        or not amendment_sidecar.is_file()
        or amendment_sidecar.is_symlink()
    ):
        raise MultiscaleHurdleRunnerError(
            "active sender-permutation amendment or sidecar is missing or "
            "unsafe"
        )
    amendment_observed = _sha256_file(amendment)
    amendment_declared = (
        amendment_sidecar.read_text(encoding="utf-8").strip().split()[0]
    )
    if (
        amendment_observed != _CONTRACT_AMENDMENT_SHA256
        or amendment_declared != amendment_observed
    ):
        raise MultiscaleHurdleRunnerError(
            "active sender-permutation amendment checksum changed"
        )
    supplement = project_root / _CONTRACT_SUPPLEMENT_RELATIVE
    supplement_sidecar = supplement.with_suffix(".sha256")
    if (
        not supplement.is_file()
        or supplement.is_symlink()
        or not supplement_sidecar.is_file()
        or supplement_sidecar.is_symlink()
        or _sha256_file(supplement) != _CONTRACT_SUPPLEMENT_SHA256
        or supplement_sidecar.read_text(
            encoding="utf-8"
        ).strip().split()[0]
        != _CONTRACT_SUPPLEMENT_SHA256
    ):
        raise MultiscaleHurdleRunnerError(
            "required amendment003 mask-safety supplement changed"
        )
    superseded = project_root / SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE
    superseded_sidecar = superseded.with_suffix(".sha256")
    if (
        not superseded.is_file()
        or superseded.is_symlink()
        or not superseded_sidecar.is_file()
        or superseded_sidecar.is_symlink()
        or _sha256_file(superseded)
        != SUPERSEDED_CONTRACT_AMENDMENT_SHA256
        or superseded_sidecar.read_text(encoding="utf-8").strip().split()[0]
        != SUPERSEDED_CONTRACT_AMENDMENT_SHA256
    ):
        raise MultiscaleHurdleRunnerError(
            "superseded rewiring amendment negative record changed"
        )
    return {
        "path": _FROZEN_CONTRACT_RELATIVE.as_posix(),
        "sha256": observed,
        "verified": True,
        "amendment": {
            "path": _CONTRACT_AMENDMENT_RELATIVE.as_posix(),
            "sha256": amendment_observed,
            "verified": True,
            "required_supplement": {
                "path": _CONTRACT_SUPPLEMENT_RELATIVE.as_posix(),
                "sha256": _CONTRACT_SUPPLEMENT_SHA256,
                "verified": True,
                "mask_noninterference_gate_required_before_gpu_training": (
                    True
                ),
            },
            "supersedes": {
                "path": (
                    SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.as_posix()
                ),
                "sha256": SUPERSEDED_CONTRACT_AMENDMENT_SHA256,
                "retained_as_negative_design_record": True,
            },
        },
    }


def _normalized_locked_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(config))
    value["attempt"] = 1
    return value


def _verify_locked_materialization(
    project_root: Path,
    contract: MultiscaleHurdleContract,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind this run to one immutable materialized job and source receipt."""

    path = project_root / _MATERIALIZATION_RELATIVE
    receipt, checksum = _verified_json_receipt(
        path, kind="locked materialization"
    )
    for field, expected in {
        "schema_version": 1,
        "receipt_kind": _MATERIALIZATION_KIND,
        "campaign_id": _CAMPAIGN_ID,
    }.items():
        _require_equal(
            receipt.get(field), expected, field=f"materialization.{field}"
        )
    for field, expected in {
        "counts": {
            "cores": 5,
            "pilot_configs": 2,
            "science_configs": 20,
        },
        "registry_mutation_performed": False,
        "queue_mutation_performed": False,
        "training_performed": False,
        "resource_limits": {
            "preferred_aggregate_gpu_hours": 12.0,
            "absolute_aggregate_gpu_hours": 24.0,
            "stage1_peak_allocated_vram_gib": 12.0,
            "per_device_peak_allocated_vram_gib": 20.5,
            "aggregate_observed_process_vram_gib": 50.0,
            "filesystem_used_decimal_gb_hard_stop": 55.0,
        },
        "fixed_graph_contract": {
            "local_k_cap": LOCAL_K_CAP,
            "local_maximum_distance_um": LOCAL_MAX_DISTANCE_UM,
            "regional_k_cap": REGIONAL_K_CAP,
            "regional_minimum_distance_exclusive_um": (
                REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
            ),
            "regional_maximum_distance_um": REGIONAL_MAX_DISTANCE_UM,
            "original_rewired_arm_authorized": False,
            "local_source_permutation": {
                "schema": LOCAL_SOURCE_PERMUTATION_SCHEMA,
                "uses_random_seed": False,
                "minimum_node_mapping_changed_fraction": (
                    LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM
                ),
                "displacement_threshold_um": (
                    LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM
                ),
                "minimum_node_displacement_above_threshold_fraction": (
                    LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM
                ),
                "minimum_local_edge_slot_sender_identity_changed_fraction": (
                    LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM
                ),
                "observed_local_topology_and_attributes_unchanged": True,
            },
        },
        "resource_gate_receipt_reference": (
            _RESOURCE_GATE_RELATIVE.as_posix()
        ),
        "representation_gate_receipt_reference": (
            _REPRESENTATION_GATE_RELATIVE.as_posix()
        ),
    }.items():
        _require_equal(
            receipt.get(field), expected, field=f"materialization.{field}"
        )
    frozen = receipt.get("frozen_contract")
    if not isinstance(frozen, Mapping):
        raise MultiscaleHurdleRunnerError(
            "materialization.frozen_contract is required"
        )
    _require_equal(
        frozen.get("reference"),
        _FROZEN_CONTRACT_RELATIVE.as_posix(),
        field="materialization.frozen_contract.reference",
    )
    _require_equal(
        frozen.get("sha256"),
        _FROZEN_CONTRACT_SHA256,
        field="materialization.frozen_contract.sha256",
    )
    amendment = receipt.get("contract_amendment")
    if not isinstance(amendment, Mapping):
        raise MultiscaleHurdleRunnerError(
            "materialization.contract_amendment is required"
        )
    _require_equal(
        amendment.get("reference"),
        _CONTRACT_AMENDMENT_RELATIVE.as_posix(),
        field="materialization.contract_amendment.reference",
    )
    _require_equal(
        amendment.get("sha256"),
        _CONTRACT_AMENDMENT_SHA256,
        field="materialization.contract_amendment.sha256",
    )
    supplement = amendment.get("required_supplement")
    if not isinstance(supplement, Mapping):
        raise MultiscaleHurdleRunnerError(
            "materialization.contract_amendment.required_supplement is "
            "required"
        )
    for field, expected in {
        "reference": _CONTRACT_SUPPLEMENT_RELATIVE.as_posix(),
        "sha256": _CONTRACT_SUPPLEMENT_SHA256,
        "mask_noninterference_gate_required_before_gpu_training": True,
    }.items():
        _require_equal(
            supplement.get(field),
            expected,
            field=(
                "materialization.contract_amendment."
                f"required_supplement.{field}"
            ),
        )
    superseded = amendment.get("supersedes")
    if not isinstance(superseded, Mapping):
        raise MultiscaleHurdleRunnerError(
            "materialization.contract_amendment.supersedes is required"
        )
    for field, expected in {
        "reference": SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.as_posix(),
        "sha256": SUPERSEDED_CONTRACT_AMENDMENT_SHA256,
        "retained_as_negative_design_record": True,
    }.items():
        _require_equal(
            superseded.get(field),
            expected,
            field=f"materialization.contract_amendment.supersedes.{field}",
        )
    parameter_audit = receipt.get("parameter_audit")
    if not isinstance(parameter_audit, Mapping):
        raise MultiscaleHurdleRunnerError(
            "materialization.parameter_audit is required"
        )
    _require_equal(
        parameter_audit.get("trainable_parameter_count"),
        _EXPECTED_PARAMETER_COUNT,
        field="materialization.parameter_audit.trainable_parameter_count",
    )
    if (
        not _is_sha256(parameter_audit.get("named_parameter_shapes_sha256"))
        or tuple(parameter_audit.get("matched_arms", ())) != _ARMS
    ):
        raise MultiscaleHurdleRunnerError(
            "materialization parameter-shape audit is invalid"
        )
    _require_equal(
        receipt.get("allowed_gpu_ids"),
        list(_SAFE_GPU_IDS),
        field="materialization.allowed_gpu_ids",
    )

    job_field = "pilot_jobs" if contract.resource_pilot else "science_jobs"
    jobs = receipt.get(job_field)
    if not isinstance(jobs, list):
        raise MultiscaleHurdleRunnerError(
            f"materialization.{job_field} must be a list"
        )
    matches = [
        item
        for item in jobs
        if isinstance(item, Mapping)
        and item.get("alias") == contract.biological_unit_alias
        and item.get("arm") == contract.arm
    ]
    if len(matches) != 1:
        raise MultiscaleHurdleRunnerError(
            "materialization does not contain exactly one matching job"
        )
    job = dict(matches[0])
    locked_config = _normalized_locked_config(config)
    config_sha = canonical_sha256(locked_config)
    _require_equal(
        job.get("config_sha256"),
        config_sha,
        field="materialization job config_sha256",
    )
    config_reference = job.get("config")
    if not isinstance(config_reference, str):
        raise MultiscaleHurdleRunnerError(
            "materialization job config reference is missing"
        )
    locked_config_path = (project_root / config_reference).resolve()
    try:
        locked_config_path.relative_to(project_root.resolve())
    except ValueError as error:
        raise MultiscaleHurdleRunnerError(
            "materialized config escapes the project root"
        ) from error
    if not locked_config_path.is_file() or locked_config_path.is_symlink():
        raise MultiscaleHurdleRunnerError(
            "materialized config is missing or unsafe"
        )
    try:
        import yaml

        file_config = yaml.safe_load(
            locked_config_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise MultiscaleHurdleRunnerError(
            "materialized config cannot be parsed"
        ) from error
    if (
        not isinstance(file_config, Mapping)
        or canonical_sha256(file_config) != config_sha
        or _sha256_file(locked_config_path) != job.get("file_sha256")
    ):
        raise MultiscaleHurdleRunnerError(
            "materialized config content identity does not verify"
        )
    _require_equal(
        str(job.get("requested_gpu")),
        str(_mapping(config, "launcher").get("requested_gpu")),
        field="materialization job requested_gpu",
    )

    cores = receipt.get("cores")
    if not isinstance(cores, list):
        raise MultiscaleHurdleRunnerError("materialization.cores is required")
    core_matches = [
        item
        for item in cores
        if isinstance(item, Mapping)
        and item.get("alias") == contract.biological_unit_alias
    ]
    if len(core_matches) != 1:
        raise MultiscaleHurdleRunnerError(
            "materialization core identity is missing or duplicated"
        )
    core = dict(core_matches[0])
    dataset = _mapping(config, "dataset")
    for field, expected in {
        "n_genes": 1000,
        "prepared_artifact": dataset.get("prepared_artifact_reference"),
        "preprocessing_sha256": dataset.get("dataset_fingerprint"),
        "split_fingerprint": dataset.get("split_fingerprint"),
    }.items():
        _require_equal(
            core.get(field), expected, field=f"materialization core {field}"
        )
    graph_record = core.get("graph_receipt")
    if not isinstance(graph_record, Mapping):
        raise MultiscaleHurdleRunnerError(
            "materialization core graph receipt is missing"
        )
    graph_record_sha = canonical_sha256(graph_record)
    _require_equal(
        core.get("graph_receipt_sha256"),
        graph_record_sha,
        field="materialization core graph_receipt_sha256",
    )
    graph_config = _mapping(config, "graph")
    _require_equal(
        graph_config.get("expected_graph_receipt_sha256"),
        graph_record_sha,
        field="graph.expected_graph_receipt_sha256",
    )
    permutation_record = graph_record.get("local_source_permutation")
    if not isinstance(permutation_record, Mapping):
        raise MultiscaleHurdleRunnerError(
            "materialization core source permutation receipt is missing"
        )
    try:
        verify_local_source_permutation_receipt(permutation_record)
    except LocalSourcePermutationError as exc:
        raise MultiscaleHurdleRunnerError(
            "materialization core source permutation receipt is invalid"
        ) from exc
    _require_equal(
        graph_config.get("local_source_permutation"),
        permutation_record,
        field="graph.local_source_permutation",
    )
    _require_equal(
        job.get("local_source_permutation_sha256"),
        permutation_record.get("checksum"),
        field="materialization job local_source_permutation_sha256",
    )
    bundle = graph_record.get("bundle_checksums")
    if not isinstance(bundle, Mapping):
        raise MultiscaleHurdleRunnerError(
            "materialized graph bundle checksums are missing"
        )
    for observed, field in (
        (job.get("graph_bundle_sha256"), "job graph_bundle_sha256"),
        (
            graph_config.get("expected_materialized_graph_sha256"),
            "graph.expected_materialized_graph_sha256",
        ),
    ):
        _require_equal(
            observed,
            bundle.get("bundle_sha256"),
            field=f"materialization {field}",
        )

    source = receipt.get("source")
    if not isinstance(source, Mapping):
        raise MultiscaleHurdleRunnerError("materialization.source is required")
    _require_equal(
        source.get("campaign_id"),
        _SOURCE_CAMPAIGN_ID,
        field="materialization.source.campaign_id",
    )
    source_reference = source.get("materialization_reference")
    if not isinstance(source_reference, str):
        raise MultiscaleHurdleRunnerError(
            "source materialization reference is missing"
        )
    source_path = (project_root / source_reference).resolve()
    try:
        source_path.relative_to(project_root.resolve())
    except ValueError as error:
        raise MultiscaleHurdleRunnerError(
            "source materialization escapes the project root"
        ) from error
    source_receipt, source_checksum = _verified_json_receipt(
        source_path, kind="source materialization"
    )
    _require_equal(
        source_receipt.get("campaign_id"),
        _SOURCE_CAMPAIGN_ID,
        field="source materialization campaign_id",
    )
    _require_equal(
        source.get("file_sha256"),
        _sha256_file(source_path),
        field="materialization.source.file_sha256",
    )
    _require_equal(
        source.get("canonical_checksum"),
        source_checksum,
        field="materialization.source.canonical_checksum",
    )
    return {
        "path": _MATERIALIZATION_RELATIVE.as_posix(),
        "checksum": checksum,
        "config_sha256": config_sha,
        "job": job,
        "core": core,
        "graph_receipt": dict(graph_record),
        "parameter_audit": dict(parameter_audit),
        "source": {
            "campaign_id": _SOURCE_CAMPAIGN_ID,
            "path": source_reference,
            "checksum": source_checksum,
            "file_sha256": source.get("file_sha256"),
        },
    }


def _validate_bound_success_job(
    job: Mapping[str, Any],
    *,
    project_root: Path,
    expected_alias: str,
    expected_config_sha256: str,
    require_h1: bool = False,
) -> None:
    for field, expected in {
        "alias": expected_alias,
        "arm": "self",
        "config_sha256": expected_config_sha256,
        "verified_bundle": True,
    }.items():
        _require_equal(job.get(field), expected, field=f"gate job {field}")
    run_id = job.get("run_id")
    bundle = job.get("bundle_reference")
    marker = job.get("success_marker_content_sha256")
    if (
        not isinstance(run_id, str)
        or not run_id.startswith("r_")
        or not isinstance(bundle, str)
        or Path(bundle).is_absolute()
        or ".." in Path(bundle).parts
        or not _is_sha256(marker)
    ):
        raise MultiscaleHurdleRunnerError(
            "gate job lacks immutable run/bundle/_SUCCESS identity"
        )
    bundle_path = (project_root / bundle).resolve()
    try:
        bundle_path.relative_to(project_root.resolve())
    except ValueError as error:
        raise MultiscaleHurdleRunnerError(
            "gate bundle reference escapes the project root"
        ) from error
    success_path = bundle_path / "_SUCCESS"
    if (
        bundle_path.name != run_id
        or not success_path.is_file()
        or success_path.is_symlink()
    ):
        raise MultiscaleHurdleRunnerError(
            "gate job does not reference its immutable successful bundle"
        )
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MultiscaleHurdleRunnerError(
            "gate job _SUCCESS marker is unreadable"
        ) from error
    if (
        not isinstance(success, Mapping)
        or success.get("run_id") != run_id
        or success.get("status") != "success"
        or success.get("content_sha256") != marker
    ):
        raise MultiscaleHurdleRunnerError(
            "gate job _SUCCESS marker identity does not verify"
        )
    if require_h1:
        for field in (
            "h1_gate_passed",
            "detection_balanced_accuracy_above_prevalence_reference",
        ):
            _require_equal(job.get(field), True, field=f"gate job {field}")
        for field in (
            "positive_continuous_huber_relative_improvement",
            "positive_count_state_mae_relative_improvement",
        ):
            value = job.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.02
            ):
                raise MultiscaleHurdleRunnerError(
                    f"gate job {field} does not pass the frozen threshold"
                )
        detection = job.get("detection_balanced_accuracy")
        prevalence = job.get("prevalence_reference_balanced_accuracy")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in (detection, prevalence)
        ) or float(detection) <= float(prevalence):
            raise MultiscaleHurdleRunnerError(
                "gate job detection accuracy does not exceed its reference"
            )
    else:
        for field in (
            "finite_losses_and_gradients",
            "parameter_match",
            "precision_equivalence_passed",
            "peak_vram_passed",
            "projected_runtime_passed",
            "disk_safety_passed",
            "runner_pilot_gate_passed",
        ):
            _require_equal(job.get(field), True, field=f"gate job {field}")
        _require_equal(
            job.get("parameter_count"),
            _EXPECTED_PARAMETER_COUNT,
            field="gate job parameter_count",
        )
        numeric_limits = {
            "fp32_amp_absolute_loss_discrepancy": (
                _PILOT_MAX_AMP_DISCREPANCY
            ),
            "peak_allocated_vram_gib": _PILOT_MAX_VRAM_GIB,
            "projected_gpu_hours_per_200_epochs": (
                _PILOT_MAX_PROJECTED_HOURS
            ),
            "filesystem_used_decimal_gb": _DISK_MAX_USED_DECIMAL_GB,
        }
        for field, maximum in numeric_limits.items():
            value = job.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or float(value) > maximum
                or (
                    field == "filesystem_used_decimal_gb"
                    and float(value) >= maximum
                )
            ):
                raise MultiscaleHurdleRunnerError(
                    f"gate job {field} exceeds its frozen threshold"
                )


def _materialized_self_hashes(project_root: Path) -> dict[str, str]:
    receipt, _ = _verified_json_receipt(
        project_root / _MATERIALIZATION_RELATIVE,
        kind="locked materialization",
    )
    jobs = receipt.get("science_jobs")
    if not isinstance(jobs, list):
        raise MultiscaleHurdleRunnerError(
            "materialization science jobs are missing"
        )
    result: dict[str, str] = {}
    for alias in _PILOT_ALIASES:
        matches = [
            item
            for item in jobs
            if isinstance(item, Mapping)
            and item.get("alias") == alias
            and item.get("arm") == "self"
        ]
        if len(matches) != 1 or not _is_sha256(
            matches[0].get("config_sha256")
        ):
            raise MultiscaleHurdleRunnerError(
                f"materialization lacks the unique {alias} science-self job"
            )
        result[alias] = str(matches[0]["config_sha256"])
    return result


def _validate_external_gates(
    project_root: Path,
    contract: MultiscaleHurdleContract,
    materialization: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Verify external resource and, for Stage 2, representation decisions."""

    if contract.resource_pilot:
        return None
    expected_hashes = _materialized_self_hashes(project_root)
    resource, resource_checksum = _verified_json_receipt(
        project_root / _RESOURCE_GATE_RELATIVE,
        kind="resource gate",
    )
    for field, expected in {
        "schema_version": 1,
        "receipt_kind": _RESOURCE_GATE_KIND,
        "campaign_id": _CAMPAIGN_ID,
        "stage": "resource",
        "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
        "materialization_checksum": materialization.get("checksum"),
        "thresholds": _RESOURCE_GATE_THRESHOLDS,
        "complete": True,
        "gate_passed": True,
        "production_authorized": True,
        "failure_reasons": [],
    }.items():
        _require_equal(resource.get(field), expected, field=f"resource_gate.{field}")
    jobs = resource.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2:
        raise MultiscaleHurdleRunnerError(
            "resource gate must contain exactly two pilot jobs"
        )
    by_alias = {
        str(job.get("alias")): job
        for job in jobs
        if isinstance(job, Mapping)
    }
    if set(by_alias) != set(_PILOT_ALIASES):
        raise MultiscaleHurdleRunnerError(
            "resource gate pilot job identities are not prespecified"
        )
    if len({str(job.get("run_id")) for job in jobs}) != 2:
        raise MultiscaleHurdleRunnerError(
            "resource gate repeats an immutable run ID"
        )
    for alias in _PILOT_ALIASES:
        _validate_bound_success_job(
            by_alias[alias],
            project_root=project_root,
            expected_alias=alias,
            expected_config_sha256=_pilot_config_hash(project_root, alias),
        )
    verification: dict[str, Any] = {
        "resource": {
            "path": _RESOURCE_GATE_RELATIVE.as_posix(),
            "checksum": resource_checksum,
            "passed": True,
        }
    }
    if not contract.requires_representation_gate:
        return verification

    representation, representation_checksum = _verified_json_receipt(
        project_root / _REPRESENTATION_GATE_RELATIVE,
        kind="representation gate",
    )
    for field, expected in {
        "schema_version": 1,
        "receipt_kind": _REPRESENTATION_GATE_KIND,
        "campaign_id": _CAMPAIGN_ID,
        "stage": "stage1",
        "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
        "materialization_checksum": materialization.get("checksum"),
        "resource_gate_checksum": resource_checksum,
        "thresholds": _REPRESENTATION_GATE_THRESHOLDS,
        "complete": True,
        "gate_passed": True,
        "both_h1_gates_passed": True,
        "stage2_authorized": True,
        "failure_reasons": [],
    }.items():
        _require_equal(
            representation.get(field),
            expected,
            field=f"representation_gate.{field}",
        )
    jobs = representation.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 2:
        raise MultiscaleHurdleRunnerError(
            "representation gate must contain exactly two science-self jobs"
        )
    by_alias = {
        str(job.get("alias")): job
        for job in jobs
        if isinstance(job, Mapping)
    }
    if set(by_alias) != set(_PILOT_ALIASES):
        raise MultiscaleHurdleRunnerError(
            "representation gate job identities are not prespecified"
        )
    if len({str(job.get("run_id")) for job in jobs}) != 2:
        raise MultiscaleHurdleRunnerError(
            "representation gate repeats an immutable run ID"
        )
    for alias in _PILOT_ALIASES:
        _validate_bound_success_job(
            by_alias[alias],
            project_root=project_root,
            expected_alias=alias,
            expected_config_sha256=expected_hashes[alias],
            require_h1=True,
        )
    verification["representation"] = {
        "path": _REPRESENTATION_GATE_RELATIVE.as_posix(),
        "checksum": representation_checksum,
        "passed": True,
    }
    return verification


def _pilot_config_hash(project_root: Path, alias: str) -> str:
    receipt, _ = _verified_json_receipt(
        project_root / _MATERIALIZATION_RELATIVE,
        kind="locked materialization",
    )
    jobs = receipt.get("pilot_jobs")
    if not isinstance(jobs, list):
        raise MultiscaleHurdleRunnerError(
            "materialization pilot jobs are missing"
        )
    matches = [
        item
        for item in jobs
        if isinstance(item, Mapping)
        and item.get("alias") == alias
        and item.get("arm") == "self"
    ]
    if len(matches) != 1 or not _is_sha256(matches[0].get("config_sha256")):
        raise MultiscaleHurdleRunnerError(
            f"materialization lacks the unique {alias} resource-pilot job"
        )
    return str(matches[0]["config_sha256"])


def _disk_usage_record(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    used_decimal_gb = float(usage.used / 1_000_000_000)
    return {
        "path": str(path.resolve()),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "used_decimal_gb": used_decimal_gb,
        "maximum_used_decimal_gb": _DISK_MAX_USED_DECIMAL_GB,
        "passed": used_decimal_gb <= _DISK_MAX_USED_DECIMAL_GB,
    }


def _enforce_disk_safety(path: Path) -> Mapping[str, Any]:
    record = _disk_usage_record(path)
    if record["passed"] is not True:
        raise MultiscaleHurdleRunnerError(
            "filesystem used space exceeds the 55.0 GB decimal hard stop"
        )
    return record


def _build_verified_graphs(
    core: FullCoreData,
    graph_config: Mapping[str, Any],
    expected_receipt: Mapping[str, Any],
) -> tuple[
    TrueMultiscaleGraphs,
    Mapping[str, Any],
    LocalSourcePermutation,
]:
    graphs = build_true_multiscale_graphs(
        core.coordinates_um,
        query_chunk_size=int(graph_config.get("query_chunk_size", 1024)),
        receiver_chunk_size=int(
            graph_config.get("receiver_shard_size", 128)
        ),
        mutual_search_chunk_size=int(
            graph_config.get("mutual_search_chunk_size", 4_000_000)
        ),
        workers=int(graph_config.get("construction_workers", 1)),
        epsilon=float(
            graph_config.get("edge_standardizer_epsilon", 1e-8)
        ),
    )
    local_index, local_attributes = graphs.local.concatenate()
    try:
        permutation = build_macroblock_spatial_antipode_permutation(
            core.coordinates_um,
            core.macroblock_ids,
            local_edge_index=local_index,
            local_edge_attributes=local_attributes,
        )
        verify_local_source_permutation_receipt(permutation.receipt)
    except LocalSourcePermutationError as exc:
        raise MultiscaleHurdleRunnerError(
            "rebuilt local source permutation failed pre-GPU QC"
        ) from exc
    observed = true_graph_receipt(graphs)
    observed["local_source_permutation"] = dict(permutation.receipt)
    if canonical_sha256(observed) != graph_config.get(
        "expected_graph_receipt_sha256"
    ):
        raise MultiscaleHurdleRunnerError(
            "rebuilt multiscale graph receipt checksum changed"
        )
    if dict(observed) != dict(expected_receipt):
        raise MultiscaleHurdleRunnerError(
            "rebuilt multiscale graph receipt differs from materialization"
        )
    for scale in ("local", "regional"):
        graph = getattr(graphs, scale)
        expected = graph_config.get(scale)
        if not isinstance(expected, Mapping):
            raise MultiscaleHurdleRunnerError(
                f"graph.{scale} expected identity is missing"
            )
        checks = {
            "expected_graph_sha256": graph.checksums.graph_sha256,
            "expected_directed_edges": graph.qc.n_directed_edges,
            "expected_components": graph.qc.n_components,
            "expected_isolated_nodes": graph.qc.n_isolated_nodes,
        }
        for field, actual in checks.items():
            _require_equal(
                expected.get(field), actual, field=f"graph.{scale}.{field}"
            )
    _require_equal(
        graph_config.get("expected_materialized_graph_sha256"),
        graphs.checksums.bundle_sha256,
        field="graph.expected_materialized_graph_sha256",
    )
    _require_equal(
        graph_config.get("expected_bundle_qc"),
        observed.get("bundle_qc"),
        field="graph.expected_bundle_qc",
    )
    return graphs, observed, permutation


def _model_arguments(
    core: FullCoreData,
    model_config: Mapping[str, Any],
    *,
    regional_routing: str,
    local_routing: str,
) -> dict[str, Any]:
    return {
        "num_genes": core.n_genes,
        "local_edge_attribute_dim": len(EDGE_ATTRIBUTE_NAMES),
        "regional_edge_attribute_dim": len(EDGE_ATTRIBUTE_NAMES),
        "expression_mean": core.expression_mean.astype(np.float32).tolist(),
        "expression_scale": core.expression_scale.astype(np.float32).tolist(),
        "node_covariate_dim": int(core.node_covariates.shape[1]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "decoder_dim": int(model_config["decoder_dim"]),
        "ffn_dim": int(model_config["ffn_dim"]),
        "attention_heads": int(model_config["attention_heads"]),
        "attention_head_dim": int(model_config["attention_head_dim"]),
        "value_head_dim": int(model_config["value_head_dim"]),
        "message_dim": int(model_config["message_dim"]),
        "edge_hidden_dim": int(model_config["edge_hidden_dim"]),
        "edge_embedding_dim": int(model_config["edge_embedding_dim"]),
        "output_channels": int(model_config["output_channels_per_gene"]),
        "dropout": float(model_config["dropout"]),
        "attention_dropout": float(model_config["attention_dropout"]),
        "receiver_chunk_size": int(model_config["receiver_chunk_size"]),
        "activation_checkpointing": bool(
            model_config["activation_checkpointing"]
        ),
        "regional_routing": regional_routing,
        "local_routing": local_routing,
    }


def _named_shapes(module: torch.nn.Module) -> dict[str, list[int]]:
    return {
        name: list(parameter.shape)
        for name, parameter in module.named_parameters()
    }


def _state_dicts_identical(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> bool:
    return list(first) == list(second) and all(
        first[name].dtype == second[name].dtype
        and first[name].shape == second[name].shape
        and torch.equal(first[name], second[name])
        for name in first
    )


def _paired_models(
    *,
    core: FullCoreData,
    model_config: Mapping[str, Any],
    selected_arm: str,
    seed: int,
    expected_shape_sha256: str,
) -> tuple[torch.nn.Module, Mapping[str, Any], Mapping[str, Any]]:
    """Audit identical shapes/initial states for every routing arm."""

    baseline_shapes: dict[str, list[int]] | None = None
    baseline_state: dict[str, torch.Tensor] | None = None
    selected: torch.nn.Module | None = None
    per_arm: dict[str, Any] = {}
    for arm in _ARMS:
        regional, local = _ROUTING[arm]
        arguments = _model_arguments(
            core,
            model_config,
            regional_routing=regional,
            local_routing=local,
        )
        set_deterministic_seed(seed, deterministic=True, warn_only=False)
        model = MultiscaleAdditiveHybridModel(**arguments)
        count = trainable_parameter_count(model)
        shapes = _named_shapes(model)
        state = {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        }
        if count != _EXPECTED_PARAMETER_COUNT:
            raise MultiscaleHurdleRunnerError(
                "frozen architecture parameter count changed: "
                f"expected {_EXPECTED_PARAMETER_COUNT}, got {count}"
            )
        if baseline_shapes is None:
            baseline_shapes = shapes
            baseline_state = {
                name: value.clone() for name, value in state.items()
            }
        elif shapes != baseline_shapes:
            raise MultiscaleHurdleRunnerError(
                "routing arms violate the exact parameter-shape contract"
            )
        elif baseline_state is None or not _state_dicts_identical(
            baseline_state, state
        ):
            raise MultiscaleHurdleRunnerError(
                "routing arms do not share bit-identical seeded initialization"
            )
        regional_zero = bool(
            torch.count_nonzero(
                model.regional_output_projection.weight
            ).item()
            == 0
        )
        local_zero = bool(
            torch.count_nonzero(model.local_output_projection.weight).item()
            == 0
        )
        if not regional_zero or not local_zero:
            raise MultiscaleHurdleRunnerError(
                "spatial output projections are not zero-initialized"
            )
        per_arm[arm] = {
            "regional_routing": regional,
            "local_routing": local,
            "trainable_parameter_count": count,
            "regional_output_zero_initialized": regional_zero,
            "local_output_zero_initialized": local_zero,
            "state_key_count": len(state),
        }
        if arm == selected_arm:
            selected = model
            selected_arguments = arguments
        else:
            del model
        gc.collect()
    assert baseline_shapes is not None and selected is not None
    shape_sha = canonical_sha256(baseline_shapes)
    if shape_sha != expected_shape_sha256:
        raise MultiscaleHurdleRunnerError(
            "named parameter shapes differ from materialization"
        )
    construction = {
        "canonical_model_key": "multiscaleadditivehybrid",
        "implementation_class": (
            f"{selected.__class__.__module__}."
            f"{selected.__class__.__qualname__}"
        ),
        "constructor_arguments": selected_arguments,
    }
    audit = {
        "schema": "multiscale_hurdle_parameter_structure_audit_v1",
        "trainable_parameter_count": _EXPECTED_PARAMETER_COUNT,
        "named_parameter_shapes_sha256": shape_sha,
        "all_arm_parameter_shapes_identical": True,
        "all_arm_initial_states_bit_identical": True,
        "matched_arms": list(_ARMS),
        "selected_arm": selected_arm,
        "arms": per_arm,
    }
    return selected, construction, audit


def _fit_multiscale_view(
    *,
    core: FullCoreData,
    counts: np.ndarray,
    graphs: TrueMultiscaleGraphs,
    permutation: LocalSourcePermutation,
) -> tuple[MultiscaleGraphSplitView, ReceiverSortedScaleGraph]:
    local_graph = graphs.local
    local_index, local_attributes = local_graph.concatenate()
    regional_index, regional_attributes = graphs.regional.concatenate()
    view = MultiscaleGraphSplitView(
        expression=torch.from_numpy(np.asarray(counts, dtype=np.float32)),
        coordinates_um=torch.from_numpy(
            np.asarray(core.coordinates_um, dtype=np.float64)
        ),
        local_edge_index=torch.from_numpy(
            np.asarray(local_index, dtype=np.int64)
        ),
        local_edge_attributes=torch.from_numpy(
            np.asarray(local_attributes, dtype=np.float32)
        ),
        regional_edge_index=torch.from_numpy(
            np.asarray(regional_index, dtype=np.int64)
        ),
        regional_edge_attributes=torch.from_numpy(
            np.asarray(regional_attributes, dtype=np.float32)
        ),
        local_source_index_by_node=torch.from_numpy(
            np.asarray(
                permutation.source_index_by_node,
                dtype=np.int64,
            ).copy()
        ),
        node_covariates=torch.from_numpy(
            np.asarray(core.node_covariates, dtype=np.float32)
        ),
        block_ids=np.asarray(core.macroblock_ids),
        name="fit",
    )
    return view, local_graph


def _array_sha256(name: str, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _fit_per_gene_references(
    raw_counts: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> HurdlePerGeneReferences:
    counts = validate_raw_counts(raw_counts, name="expression_counts")
    mean = np.asarray(expression_mean, dtype=np.float64)
    scale = np.asarray(expression_scale, dtype=np.float64)
    n_nodes, n_genes = counts.shape
    if mean.shape != (n_genes,) or scale.shape != (n_genes,):
        raise MultiscaleHurdleRunnerError(
            "per-gene reference standardization is misaligned"
        )
    positive = counts > 0
    support = positive.sum(axis=0, dtype=np.int64)
    if np.any(support == 0):
        raise MultiscaleHurdleRunnerError(
            "every gene requires positive support for its reference"
        )
    probability = (support.astype(np.float64) + 0.5) / (
        float(n_nodes) + 1.0
    )
    log1p = np.log1p(counts.astype(np.float64, copy=False))
    continuous = np.asarray(
        [
            (
                float(np.median(log1p[positive[:, gene], gene]))
                - mean[gene]
            )
            / scale[gene]
            for gene in range(n_genes)
        ],
        dtype=np.float64,
    )
    arrays = {
        "detection_probability": probability,
        "positive_continuous_standardized": continuous,
    }
    audit = {
        "schema": "multiscale_hurdle_all_fit_per_gene_references_v1",
        "fit_scope": "all_nodes_transductive",
        "n_nodes": n_nodes,
        "n_genes": n_genes,
        "smoothing": "Jeffreys_add_half",
        "positive_continuous_statistic": "per_gene_positive_median_log1p",
        "detection_threshold": 0.5,
        "thresholds_fitted": False,
        "positive_support_minimum": int(support.min()),
        "positive_support_maximum": int(support.max()),
        "reference_sha256": canonical_sha256(
            {
                name: _array_sha256(name, value)
                for name, value in arrays.items()
            }
        ),
    }
    return HurdlePerGeneReferences(
        detection_probability=probability,
        positive_continuous_standardized=continuous,
        audit=audit,
    )


def _reference_metrics(
    references: HurdlePerGeneReferences,
    result: Any,
    *,
    expression_mean: Any,
    expression_scale: Any,
) -> dict[str, Any]:
    probability = torch.from_numpy(
        references.detection_probability.astype(np.float32)
    ).clamp(1e-7, 1.0 - 1e-7)
    logits = torch.logit(probability)
    continuous = torch.from_numpy(
        references.positive_continuous_standardized.astype(np.float32)
    )
    one = torch.stack((logits, continuous), dim=-1)
    prediction = one.unsqueeze(0).expand(result.target.shape[0], -1, -1)
    evaluation = evaluate_hurdle_continuous_output(
        prediction,
        result.target,
        result.target_mask,
        expression_mean=expression_mean,
        expression_scale=expression_scale,
    )
    return {
        f"reference_per_gene_{name}": value
        for name, value in evaluation.metrics.items()
    }


def _whole_node_replicate_zero_mask(bundle: Any) -> np.ndarray:
    matches = [
        entry
        for entry in bundle.manifest["entries"]
        if str(entry["spec"]["mode"]) == "node"
        and int(entry["replicate"]) == 0
    ]
    if len(matches) != 1:
        raise MultiscaleHurdleRunnerError(
            "fixed masks lack unique whole-node replicate zero"
        )
    return bundle.masks[str(matches[0]["entry_id"])]


def _replicate_metric_row(
    *,
    entry: Mapping[str, Any],
    model_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = json_safe_metrics({**model_metrics, **reference_metrics})
    for field in (
        "hurdle_loss",
        "positive_continuous_huber",
        "positive_count_state_mae",
    ):
        reference = float(metrics[f"reference_per_gene_{field}"])
        observed = float(metrics[field])
        metrics[f"{field}_relative_improvement_over_per_gene_reference"] = (
            (reference - observed) / reference if reference > 0.0 else None
        )
    metrics[
        "detection_balanced_accuracy_gain_over_per_gene_reference"
    ] = float(metrics["detection_balanced_accuracy"]) - float(
        metrics["reference_per_gene_detection_balanced_accuracy"]
    )
    for field in _PERCENT_FIELDS:
        for prefix in ("", "reference_per_gene_"):
            name = f"{prefix}{field}"
            value = metrics.get(name)
            metrics[f"{name}_percent"] = (
                None if value is None else 100.0 * float(value)
            )
    support = metrics.get("state8_support")
    recall = metrics.get("state8_recall")
    if not isinstance(support, list) or len(support) != 8:
        raise MultiscaleHurdleRunnerError("state8 support vector is malformed")
    if not isinstance(recall, list) or len(recall) != 8:
        raise MultiscaleHurdleRunnerError("state8 recall vector is malformed")
    return {
        "split": "fit",
        "mask_mode": _PUBLIC_MASK_NAMES[str(entry["spec"]["mode"])],
        "mask_replicate": int(entry["replicate"]),
        "mask_entry_id": str(entry["entry_id"]),
        "mask_seed": int(entry["seed"]),
        "mask_checksum": str(entry["mask_checksum"]),
        **metrics,
    }


def _finite_scalar(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _final_metrics(
    *,
    replicate_rows: Sequence[Mapping[str, Any]],
    training: MultiscaleHurdleTrainingResult,
    replicates_per_mode: int,
    durations: Mapping[str, float],
    parameter_count: int,
    checkpoint_size: int,
    peak_vram_bytes: int,
    projected_runtime_hours: float | None,
    disk: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    fields = sorted(
        {
            key
            for row in replicate_rows
            for key, value in row.items()
            if key not in _ROW_METADATA_FIELDS
            and key not in {"n_masked", "n_zero", "n_positive"}
            and (_finite_scalar(value) or value is None)
        }
    )
    for mode in _REQUIRED_PUBLIC_MASKS:
        selected = [
            row for row in replicate_rows if row["mask_mode"] == mode
        ]
        if len(selected) != replicates_per_mode:
            raise MultiscaleHurdleRunnerError(
                f"expected {replicates_per_mode} rows for {mode}"
            )
        for field in fields:
            metrics[f"fit/{mode}/{field}"] = _mean_finite(selected, field)
    metrics.update(
        {
            "resource/data_preparation_duration_seconds": durations["data"],
            "resource/graph_construction_duration_seconds": durations["graph"],
            "resource/training_duration_seconds": durations["training"],
            "resource/inference_duration_seconds": durations["evaluation"],
            "resource/total_duration_seconds": durations["total"],
            "resource/parameter_count": parameter_count,
            "resource/checkpoint_size_bytes": checkpoint_size,
            "resource/peak_allocated_vram_bytes": peak_vram_bytes,
            "resource/peak_allocated_vram_gib": peak_vram_bytes / (1024**3),
            "resource/projected_gpu_hours_per_200_epochs": (
                projected_runtime_hours
            ),
            "resource/filesystem_used_decimal_gb": disk[
                "used_decimal_gb"
            ],
            "training/final_epoch": training.final_epoch,
            "training/final_hurdle_loss": training.final_train_loss,
        }
    )
    return metrics


def _precision_record(
    result: MultiscalePrecisionEquivalenceResult | None,
) -> dict[str, Any]:
    if result is None:
        return {
            "performed": False,
            "authorization": "external_verified_resource_gate_receipt",
        }
    return {
        "performed": True,
        **asdict(result),
        "maximum_allowed_discrepancy": _PILOT_MAX_AMP_DISCREPANCY,
        "actual_amp_dtype": (
            "float16" if result.amp_dtype == "auto" else result.amp_dtype
        ),
    }


def _cuda_peak_bytes(device: str) -> int:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(resolved))


def _resource_diagnostic(
    *,
    contract: MultiscaleHurdleContract,
    training: MultiscaleHurdleTrainingResult,
    precision: MultiscalePrecisionEquivalenceResult | None,
    durations: Mapping[str, float],
    peak_vram_bytes: int,
    projected_runtime_hours: float | None,
    disk: Mapping[str, Any],
    parameter_count: int,
) -> dict[str, Any]:
    discrepancy = (
        None
        if precision is None
        else precision.absolute_total_loss_discrepancy
    )
    precision_pass = precision is None or precision.passed
    vram_pass = peak_vram_bytes / (1024**3) <= _PILOT_MAX_VRAM_GIB
    runtime_pass = (
        projected_runtime_hours is None
        or projected_runtime_hours <= _PILOT_MAX_PROJECTED_HOURS
    )
    pilot_pass = (
        precision_pass
        and vram_pass
        and runtime_pass
        and disk["passed"] is True
        and parameter_count == _EXPECTED_PARAMETER_COUNT
    )
    device = torch.device(training.device)
    return {
        "schema": "multiscale_hurdle_resource_diagnostic_v1",
        "resource_pilot": contract.resource_pilot,
        "arm": contract.arm,
        "biological_unit_alias": contract.biological_unit_alias,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "precision_equivalence": _precision_record(precision),
        "fp32_amp_absolute_loss_discrepancy": discrepancy,
        "parameter_count": parameter_count,
        "parameter_match": parameter_count == _EXPECTED_PARAMETER_COUNT,
        "finite_losses_and_gradients": True,
        "epochs_completed": len(training.history),
        "epoch_duration_seconds": [
            record.duration_seconds for record in training.history
        ],
        "projected_gpu_hours_per_200_epochs": projected_runtime_hours,
        "peak_allocated_vram_bytes": peak_vram_bytes,
        "peak_allocated_vram_gib": peak_vram_bytes / (1024**3),
        "filesystem_used_decimal_gb": disk["used_decimal_gb"],
        "durations_seconds": dict(durations),
        "thresholds": dict(_RESOURCE_GATE_THRESHOLDS),
        "checks": {
            "precision_equivalence_passed": precision_pass,
            "peak_vram_passed": vram_pass,
            "projected_runtime_passed": runtime_pass,
            "disk_safety_passed": disk["passed"],
            "parameter_match_passed": (
                parameter_count == _EXPECTED_PARAMETER_COUNT
            ),
        },
        "runner_pilot_gate_passed": (
            pilot_pass if contract.resource_pilot else None
        ),
    }
def _checkpoint_bytes(
    *,
    archive: RunArchive,
    contract: MultiscaleHurdleContract,
    model_config: Mapping[str, Any],
    construction: Mapping[str, Any],
    parameter_audit: Mapping[str, Any],
    training: MultiscaleHurdleTrainingResult,
    core: FullCoreData,
    graph_receipt_value: Mapping[str, Any],
    selected_local: ReceiverSortedScaleGraph,
    masks: Any,
    references: HurdlePerGeneReferences,
) -> bytes:
    payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "checkpoint_role": "last",
        "checkpoint_policy": training.checkpoint_policy,
        "training_protocol": training.training_protocol,
        "task_family": _TASK_FAMILY,
        "arm": contract.arm,
        "biological_unit_alias": contract.biological_unit_alias,
        "model_config": dict(model_config),
        "model_construction": dict(construction),
        "parameter_structure_audit": dict(parameter_audit),
        "epoch": training.final_epoch,
        "fixed_epoch_budget": training.fixed_epoch_budget,
        "model_state_dict": dict(training.final_state_dict),
        "state_dict_sha256": training.final_state_checksum,
        "full_core_preprocessing_sha256": (
            core.checksums.preprocessing_sha256
        ),
        "raw_count_sha256": core.checksums.expression_counts_sha256,
        "graph_receipt": dict(graph_receipt_value),
        "selected_local_graph_sha256": (
            selected_local.checksums.graph_sha256
        ),
        "evaluation_mask_bundle_sha256": masks.checksum,
        "reference_sha256": references.audit["reference_sha256"],
        "count_representation_schema": _REPRESENTATION_SCHEMA,
        "objective": _OBJECTIVE,
        "effective_amp": True,
        "monitored_metric": None,
        "selection_policy": "last_epoch_without_validation_selection",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _protected_prediction_rows(
    *,
    archive: RunArchive,
    dataset: Mapping[str, Any],
    graph_id: str,
    edge_count: int,
    fold: int,
    entry: Mapping[str, Any],
    result: Any,
    sample_key_salt: str,
) -> Iterator[dict[str, Any]]:
    target = result.target.numpy()
    mask = result.target_mask.numpy().astype(bool, copy=False)
    evaluation = result.evaluation
    reconstructed = evaluation.reconstructed_count.numpy()
    state = evaluation.count_state.numpy()
    positive_state = evaluation.positive_count_state.numpy()
    detected = evaluation.detected.numpy()
    continuous = evaluation.positive_continuous_standardized.numpy()
    positive_count = evaluation.positive_reconstructed_count.numpy()
    shapes = {
        target.shape,
        mask.shape,
        reconstructed.shape,
        state.shape,
        positive_state.shape,
        detected.shape,
        continuous.shape,
        positive_count.shape,
    }
    if len(shapes) != 1:
        raise MultiscaleHurdleRunnerError(
            "protected hurdle prediction arrays are not aligned"
        )
    namespace = (
        f"bagm:{dataset['dataset_id']}:{dataset['version']}:full-core-fit"
    )
    target_nodes = result.target_nodes.numpy().astype(np.int64, copy=False)
    batch: list[dict[str, Any]] = []
    for local_index, node_index in enumerate(target_nodes):
        genes = np.flatnonzero(mask[local_index])
        truth = np.rint(target[local_index, genes]).astype(np.int32)
        truth_state = tokenize_raw_counts(truth.reshape(1, -1)).reshape(-1)
        batch.append(
            {
                "_protected_local_index": int(node_index),
                "run_id": archive.run_id,
                "graph_id": graph_id,
                "dataset_id": str(dataset["dataset_id"]),
                "split": "fit",
                "fold": fold,
                "y_true": truth.astype(int).tolist(),
                "y_pred": reconstructed[local_index, genes].astype(
                    float
                ).tolist(),
                "target_indices": genes.astype(int).tolist(),
                "y_true_state": truth_state.astype(int).tolist(),
                "y_pred_state": state[local_index, genes].astype(int).tolist(),
                "y_pred_positive_state": positive_state[
                    local_index, genes
                ].astype(int).tolist(),
                "y_pred_detected": detected[
                    local_index, genes
                ].astype(int).tolist(),
                "y_pred_positive_count": positive_count[
                    local_index, genes
                ].astype(float).tolist(),
                "y_pred_standardized_log1p": continuous[
                    local_index, genes
                ].astype(float).tolist(),
                "node_count": int(result.full_mask.shape[0]),
                "edge_count": edge_count,
                "effective_mask_rate": float(
                    genes.size / result.full_mask.shape[1]
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


def run_multiscale_hurdle_capacity(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    sample_key_salt: str,
    full_core_data: FullCoreData | None = None,
    multiscale_graphs: TrueMultiscaleGraphs | None = None,
) -> CapacityRunResult:
    """Execute one frozen arm and populate its worker-owned active archive."""

    started = time.monotonic()
    if len(sample_key_salt.encode("utf-8")) < 16:
        raise RunValidationError(
            "BAGM_SAMPLE_KEY_SALT must contain at least 16 bytes"
        )
    contract = _validate_multiscale_contract(config)
    project_root = archive.paths.project_root
    frozen = _verify_frozen_contract(project_root)
    locked = _verify_locked_materialization(project_root, contract, config)
    gates = _validate_external_gates(
        project_root, contract, locked
    )
    initial_disk = _enforce_disk_safety(project_root)
    dataset = _mapping(config, "dataset")
    model_config = _mapping(config, "model")
    graph_config = _mapping(config, "graph")
    evaluation_config = _mapping(config, "evaluation")

    data_started = time.monotonic()
    core = (
        full_core_data
        if full_core_data is not None
        else load_and_refit_full_core(
            _resolve_prepared_artifact(config, archive)
        )
    )
    counts = validate_raw_counts(
        core.expression_counts, name="expression_counts"
    )
    if (
        core.n_genes != 1000
        or core.node_covariates.shape[1] != len(ALLOWED_METADATA_COLUMNS)
        or tuple(core.metadata_names) != tuple(ALLOWED_METADATA_COLUMNS)
    ):
        raise MultiscaleHurdleRunnerError(
            "materialized core differs from the 1000-gene/22-covariate schema"
        )
    materialized_identity = _verify_materialized_identity(core, dataset)
    references = _fit_per_gene_references(
        counts,
        expression_mean=core.expression_mean,
        expression_scale=core.expression_scale,
    )
    data_duration = time.monotonic() - data_started

    graph_started = time.monotonic()
    if multiscale_graphs is None:
        graphs, rebuilt_receipt, permutation = _build_verified_graphs(
            core, graph_config, locked["graph_receipt"]
        )
    else:
        graphs = multiscale_graphs
        local_index, local_attributes = graphs.local.concatenate()
        try:
            permutation = build_macroblock_spatial_antipode_permutation(
                core.coordinates_um,
                core.macroblock_ids,
                local_edge_index=local_index,
                local_edge_attributes=local_attributes,
            )
        except LocalSourcePermutationError as exc:
            raise MultiscaleHurdleRunnerError(
                "supplied graph source permutation failed pre-GPU QC"
            ) from exc
        rebuilt_receipt = true_graph_receipt(graphs)
        rebuilt_receipt["local_source_permutation"] = dict(
            permutation.receipt
        )
        if dict(rebuilt_receipt) != dict(locked["graph_receipt"]):
            raise MultiscaleHurdleRunnerError(
                "supplied multiscale graph receipt differs from materialization"
            )
        _build_verified_graphs_identity_only(
            graphs, graph_config, rebuilt_receipt
        )
    graph_duration = time.monotonic() - graph_started
    _enforce_disk_safety(project_root)

    training_config = _training_config(config)
    view, selected_local = _fit_multiscale_view(
        core=core,
        counts=counts,
        graphs=graphs,
        permutation=permutation,
    )
    masks = _evaluation_masks(config, core, training_config)
    expected_shape_sha = str(
        _mapping(locked, "parameter_audit")[
            "named_parameter_shapes_sha256"
        ]
    )
    model, construction, parameter_audit = _paired_models(
        core=core,
        model_config=model_config,
        selected_arm=contract.arm,
        seed=training_config.model_seed,
        expected_shape_sha256=expected_shape_sha,
    )

    archive.write_json(
        "diagnostics/full_core_preprocessing.json",
        core.preprocessing_qc.to_dict(),
    )
    archive.write_json(
        "diagnostics/multiscale_graph_receipt.json", rebuilt_receipt
    )
    archive.write_json(
        "diagnostics/raw_count_representation.json",
        {
            "schema": _REPRESENTATION_SCHEMA,
            "input_state_schema": _INPUT_STATE_SCHEMA,
            "validation": "finite_nonnegative_integer",
            "n_nodes": core.n_nodes,
            "n_genes": core.n_genes,
            "raw_count_sha256": core.checksums.expression_counts_sha256,
            "continuous_mean_sha256": _array_sha256(
                "expression_mean", core.expression_mean
            ),
            "continuous_scale_sha256": _array_sha256(
                "expression_scale", core.expression_scale
            ),
            "prediction_channels": [
                "detection_logit",
                "positive_standardized_log1p",
            ],
            "reference_audit": dict(references.audit),
        },
    )
    archive.write_json(
        "diagnostics/parameter_structure_audit.json", parameter_audit
    )
    archive.write_json(
        "diagnostics/mask_statistics.json",
        {
            "role": "held_in_fit_technical_replicates",
            "biological_unit_alias": contract.biological_unit_alias,
            "independent_biological_replicates": 1,
            "manifest": masks.manifest,
        },
    )
    archive.write_json("diagnostics/disk_safety_start.json", initial_disk)
    archive.write_json("provenance/frozen_task_contract.json", frozen)
    archive.write_json(
        "provenance/locked_materialization_identity.json",
        {
            key: value
            for key, value in locked.items()
            if key != "graph_receipt"
        },
    )
    archive.write_json(
        "provenance/external_gate_authorization.json",
        {
            "resource_pilot": contract.resource_pilot,
            "required_representation_gate": (
                contract.requires_representation_gate
            ),
            "verified_gates": gates,
        },
    )
    archive.write_json(
        "provenance/full_core_inputs.json",
        {
            "biological_unit_alias": contract.biological_unit_alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "fit_scope": "all_nodes_transductive",
            "generalization_estimate": False,
            "prepared_artifact_reference": (
                dataset["prepared_artifact_reference"]
            ),
            "preprocessing_checksums": core.checksums.to_dict(),
            "materialized_identity_verification": materialized_identity,
            "graph_receipt_sha256": canonical_sha256(rebuilt_receipt),
            "selected_local_scale": selected_local.name,
            "selected_local_graph_sha256": (
                selected_local.checksums.graph_sha256
            ),
            "regional_graph_sha256": (
                graphs.regional.checksums.graph_sha256
            ),
            "regional_interpretation": (
                "regional_context_not_direct_interaction"
            ),
            "data_preparation_duration_seconds": data_duration,
            "graph_construction_duration_seconds": graph_duration,
        },
    )
    archive.write_json(
        "provenance/per_gene_references.json",
        {
            "audit": dict(references.audit),
            "detection_probability": (
                references.detection_probability.tolist()
            ),
            "positive_continuous_standardized": (
                references.positive_continuous_standardized.tolist()
            ),
        },
    )
    archive.write_json(
        "provenance/fixed_evaluation_masks.json",
        {
            "bundle_manifest": masks.manifest,
            "seed_namespace": "held-in-full-core-fixed-evaluation",
            "used_for_gradient_updates": False,
            "used_for_checkpoint_selection": False,
            "technical_replicates_not_biological_replicates": True,
        },
    )

    precision: MultiscalePrecisionEquivalenceResult | None = None
    if contract.resource_pilot:
        precision = compare_multiscale_fp32_amp_loss(
            model,
            view,
            _whole_node_replicate_zero_mask(masks),
            expression_mean=core.expression_mean,
            expression_scale=core.expression_scale,
            device=training_config.device or "cuda",
            target_node_batch_size=_TARGET_NODE_BATCH_SIZE,
            amp_dtype=training_config.amp_dtype,
            maximum_discrepancy=_PILOT_MAX_AMP_DISCREPANCY,
        )
        precision_record = _precision_record(precision)
        archive.write_json(
            "diagnostics/fp32_amp_equivalence.json", precision_record
        )
        if not precision.passed:
            raise MultiscaleHurdleRunnerError(
                "resource pilot FP32/AMP discrepancy exceeds 1e-3"
            )
        if (
            precision.peak_cuda_memory_bytes / (1024**3)
            > _PILOT_MAX_VRAM_GIB
        ):
            raise MultiscaleHurdleRunnerError(
                "resource pilot precision smoke exceeds 12 GiB"
            )

    training_started = time.monotonic()
    training_result = fit_full_core_multiscale_hurdle_model(
        model,
        view,
        training_config,
        expression_mean=core.expression_mean,
        expression_scale=core.expression_scale,
        target_node_batch_size=_TARGET_NODE_BATCH_SIZE,
    )
    training_duration = time.monotonic() - training_started
    archive.write_table(
        "metrics/history",
        [
            {
                "run_id": archive.run_id,
                "split": "fit",
                "training_protocol": training_result.training_protocol,
                **row,
            }
            for row in training_result.history_rows()
        ],
        fallback="jsonl",
    )
    history_peak = max(
        record.peak_cuda_memory_bytes for record in training_result.history
    )
    precision_peak = (
        0 if precision is None else precision.peak_cuda_memory_bytes
    )
    peak_vram_bytes = max(
        history_peak,
        precision_peak,
        _cuda_peak_bytes(training_result.device),
    )
    projected_runtime_hours = (
        float(
            np.mean(
                [
                    record.duration_seconds
                    for record in training_result.history
                ]
            )
        )
        * 200.0
        / 3600.0
        if contract.resource_pilot
        else None
    )
    _enforce_disk_safety(project_root)
    if contract.resource_pilot and (
        peak_vram_bytes / (1024**3) > _PILOT_MAX_VRAM_GIB
        or projected_runtime_hours is None
        or projected_runtime_hours > _PILOT_MAX_PROJECTED_HOURS
    ):
        archive.write_json(
            "diagnostics/resource_gate_failure.json",
            {
                "peak_allocated_vram_gib": (
                    peak_vram_bytes / (1024**3)
                ),
                "projected_gpu_hours_per_200_epochs": (
                    projected_runtime_hours
                ),
                "thresholds": _RESOURCE_GATE_THRESHOLDS,
            },
        )
        raise MultiscaleHurdleRunnerError(
            "resource pilot exceeds the frozen memory or runtime gate"
        )

    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt",
        _checkpoint_bytes(
            archive=archive,
            contract=contract,
            model_config=model_config,
            construction=construction,
            parameter_audit=parameter_audit,
            training=training_result,
            core=core,
            graph_receipt_value=rebuilt_receipt,
            selected_local=selected_local,
            masks=masks,
            references=references,
        ),
    )
    checkpoint_size = checkpoint_path.stat().st_size
    _enforce_disk_safety(project_root)
    archive.write_json(
        "provenance/full_core_training.json",
        {
            "training_protocol": training_result.training_protocol,
            "graph_execution": training_result.graph_execution,
            "checkpoint_policy": training_result.checkpoint_policy,
            "final_epoch": training_result.final_epoch,
            "fixed_epoch_budget": training_result.fixed_epoch_budget,
            "state_dict_sha256": training_result.final_state_checksum,
            "model_seed": training_config.model_seed,
            "epoch_mask_seed": training_config.mask_seed,
            "parameter_count": _EXPECTED_PARAMETER_COUNT,
            "device": training_result.device,
            "model_construction": construction,
            "parameter_structure_audit": parameter_audit,
            "objective": _OBJECTIVE,
            "effective_amp": training_config.amp,
            "target_node_batch_size": _TARGET_NODE_BATCH_SIZE,
            "regional_routing": training_result.regional_routing,
            "local_routing": training_result.local_routing,
        },
    )

    replicate_rows: list[dict[str, Any]] = []
    evaluation_started = time.monotonic()
    edge_count = (
        (graphs.regional.qc.n_directed_edges
         if contract.regional_routing != "surrogate" else 0)
        + (selected_local.qc.n_directed_edges
           if contract.local_routing != "surrogate" else 0)
    )
    graph_id = (
        f"multiscale_{contract.arm.replace('-', '_')}_"
        f"{graphs.checksums.bundle_sha256[:16]}"
    )

    def prediction_rows() -> Iterator[dict[str, Any]]:
        for entry in masks.manifest["entries"]:
            result = evaluate_fixed_multiscale_hurdle_mask(
                model,
                view,
                masks.masks[str(entry["entry_id"])],
                expression_mean=core.expression_mean,
                expression_scale=core.expression_scale,
                target_node_batch_size=_TARGET_NODE_BATCH_SIZE,
                device=training_config.device,
                amp=training_config.amp,
                amp_dtype=training_config.amp_dtype,
            )
            references_for_mask = _reference_metrics(
                references,
                result,
                expression_mean=core.expression_mean,
                expression_scale=core.expression_scale,
            )
            row = _replicate_metric_row(
                entry=entry,
                model_metrics=result.evaluation.metrics,
                reference_metrics=references_for_mask,
            )
            replicate_rows.append(row)
            mode = str(row["mask_mode"])
            for name, value in row.items():
                if name in _ROW_METADATA_FIELDS or not _finite_scalar(value):
                    continue
                archive.append_metric_event(
                    {
                        "name": f"fit/{mode}/{name}",
                        "value": value,
                        "mask_replicate": int(entry["replicate"]),
                        "mask_seed": int(entry["seed"]),
                    }
                )
            if mode == "whole_node" and int(entry["replicate"]) == 0:
                yield from _protected_prediction_rows(
                    archive=archive,
                    dataset=dataset,
                    graph_id=graph_id,
                    edge_count=edge_count,
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
    final_disk = _enforce_disk_safety(project_root)
    durations = {
        "data": data_duration,
        "graph": graph_duration,
        "training": training_duration,
        "evaluation": evaluation_duration,
        "total": time.monotonic() - started,
    }
    final_metrics = _final_metrics(
        replicate_rows=replicate_rows,
        training=training_result,
        replicates_per_mode=int(
            evaluation_config["mask_replicates_per_mode"]
        ),
        durations=durations,
        parameter_count=_EXPECTED_PARAMETER_COUNT,
        checkpoint_size=checkpoint_size,
        peak_vram_bytes=peak_vram_bytes,
        projected_runtime_hours=projected_runtime_hours,
        disk=final_disk,
    )
    primary_value = final_metrics.get(_PRIMARY_METRIC)
    if not _finite_scalar(primary_value):
        raise MultiscaleHurdleRunnerError(
            "primary whole-node hurdle loss is missing or non-finite"
        )
    for name, value in final_metrics.items():
        if _finite_scalar(value):
            archive.append_metric_event(
                {"name": name, "value": value, "phase": "final_aggregate"}
            )
    archive.write_json("metrics/final.json", final_metrics)

    losses = np.asarray(
        [
            record.train_hurdle_loss
            for record in training_result.history
        ],
        dtype=np.float64,
    )
    tail = losses[-min(20, len(losses)) :]
    archive.write_json(
        "diagnostics/training_convergence.json",
        {
            "objective": _OBJECTIVE,
            "final_epoch": training_result.final_epoch,
            "final_train_hurdle_loss": training_result.final_train_loss,
            "minimum_observed_train_hurdle_loss": float(losses.min()),
            "last_20_epoch_loss_slope": (
                float(
                    np.polyfit(
                        np.arange(len(tail), dtype=np.float64), tail, 1
                    )[0]
                )
                if len(tail) >= 2
                else None
            ),
            "all_epochs_completed": len(training_result.history)
            == training_result.fixed_epoch_budget,
            "all_losses_and_gradients_finite": True,
        },
    )
    resource = _resource_diagnostic(
        contract=contract,
        training=training_result,
        precision=precision,
        durations=durations,
        peak_vram_bytes=peak_vram_bytes,
        projected_runtime_hours=projected_runtime_hours,
        disk=final_disk,
        parameter_count=_EXPECTED_PARAMETER_COUNT,
    )
    archive.write_json("diagnostics/resource_usage.json", resource)
    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "training_exit_status": "success",
        "campaign_id": _CAMPAIGN_ID,
        "biological_unit_alias": contract.biological_unit_alias,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "evaluation_protocol": _PROTOCOL,
        "task_family": _TASK_FAMILY,
        "canonical_prediction_split": "fit",
        "model_name": "multiscale-hurdle-count",
        "arm": contract.arm,
        "regional_routing": contract.regional_routing,
        "local_routing": contract.local_routing,
        "model_seed": training_config.model_seed,
        "final_epoch": training_result.final_epoch,
        "fixed_epoch_budget": training_result.fixed_epoch_budget,
        "checkpoint_role": "last",
        "primary_metric_name": _PRIMARY_METRIC,
        "primary_metric_value": float(primary_value),
        "metrics": final_metrics,
        "parameter_count": _EXPECTED_PARAMETER_COUNT,
        "exact_parameter_shape_match": True,
        "duration_seconds": durations["total"],
        "peak_vram_gib": peak_vram_bytes / (1024**3),
        "peak_host_memory_bytes": _peak_host_memory_bytes(),
        "graph_bundle_sha256": graphs.checksums.bundle_sha256,
        "selected_local_graph_sha256": (
            selected_local.checksums.graph_sha256
        ),
        "evaluation_mask_bundle_sha256": masks.checksum,
        "canonical_prediction_selection": {
            "split": "fit",
            "mask_mode": "whole_node",
            "mask_replicate": 0,
            "selection_status": "prespecified",
        },
        "resource_pilot": contract.resource_pilot,
        "runner_pilot_gate_passed": resource[
            "runner_pilot_gate_passed"
        ],
        "fp32_amp_absolute_loss_discrepancy": resource[
            "fp32_amp_absolute_loss_discrepancy"
        ],
        "peak_allocated_vram_gib": resource[
            "peak_allocated_vram_gib"
        ],
        "projected_gpu_hours_per_200_epochs": resource[
            "projected_gpu_hours_per_200_epochs"
        ],
        "filesystem_used_decimal_gb": resource[
            "filesystem_used_decimal_gb"
        ],
        "conclusion_eligible": not contract.resource_pilot,
        "generalization_estimate": False,
        "maximum_claim": (
            "diagnostic precision, runtime, and memory feasibility only"
            if contract.resource_pilot
            else (
                "exploratory held-in adjacent-normal representation capacity "
                "and possible topology-specific predictive dependency"
            )
        ),
        "prohibited_claims": [
            "unseen-patient generalization",
            "direct cell-cell communication",
            "biological mechanism",
            "causal influence",
        ],
    }
    archive.write_summary(summary)
    return CapacityRunResult(
        run_id=archive.run_id,
        model_name="multiscale-hurdle-count",
        primary_metric_name=_PRIMARY_METRIC,
        primary_metric_value=float(primary_value),
        final_epoch=training_result.final_epoch,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        summary=summary,
    )


def _build_verified_graphs_identity_only(
    graphs: TrueMultiscaleGraphs,
    graph_config: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    if canonical_sha256(receipt) != graph_config.get(
        "expected_graph_receipt_sha256"
    ):
        raise MultiscaleHurdleRunnerError(
            "supplied graph receipt checksum differs from config"
        )
    _require_equal(
        graphs.checksums.bundle_sha256,
        graph_config.get("expected_materialized_graph_sha256"),
        field="supplied graph bundle checksum",
    )
    for scale in ("local", "regional"):
        expected = graph_config.get(scale)
        graph = getattr(graphs, scale)
        if not isinstance(expected, Mapping):
            raise MultiscaleHurdleRunnerError(
                f"graph.{scale} identity is missing"
            )
        _require_equal(
            graph.checksums.graph_sha256,
            expected.get("expected_graph_sha256"),
            field=f"supplied graph.{scale} checksum",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    result = run_multiscale_hurdle_capacity(
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
                "runner_pilot_gate_passed": result.summary.get(
                    "runner_pilot_gate_passed"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
