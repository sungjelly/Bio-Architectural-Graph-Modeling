#!/usr/bin/env python3
"""Run one worker-owned hybrid raw-count full-core capacity experiment.

This entry point is intentionally separate from the regression and categorical
full-core runner.  It enforces the frozen adjacent-normal campaign contract,
rebuilds the checksum-bound exact mutual-k1000 graph, audits both paired model
arms, and writes only scientific outputs into an archive already owned by the
queue worker.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
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
    _build_graph,
    _evaluation_masks,
    _fit_view,
    _mean_finite,
    _peak_host_memory_bytes,
    _resolve_prepared_artifact,
    _section,
    _training_config,
    _verify_graph_identity,
    _verify_materialized_identity,
    _worker_archive_and_config,
)
from spatial_benchmark.full_core import (  # noqa: E402
    EDGE_ATTRIBUTE_NAMES,
    FullCoreData,
    ReceiverSortedGraph,
    load_and_refit_full_core,
)
from spatial_benchmark.hybrid_count import (  # noqa: E402
    HybridEdgeParameterMatchedSelfControl,
    HybridReceiverChunkedEdgeConditionedGATv2,
    assert_exact_parameter_match,
    tokenize_raw_counts,
    validate_raw_counts,
)
from spatial_benchmark.hybrid_count_metrics import (  # noqa: E402
    HybridCountReferences,
    fit_hybrid_count_references,
    json_safe_metrics,
)
from spatial_benchmark.hybrid_count_training import (  # noqa: E402
    HybridCountTrainingResult,
    PrecisionEquivalenceResult,
    compare_fp32_amp_loss,
    evaluate_fixed_hybrid_count_mask,
    fit_full_core_hybrid_count_model,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    RunValidationError,
    deidentify_prediction_rows,
)
from spatial_benchmark.training import (  # noqa: E402
    TrainingConfig,
    set_deterministic_seed,
)


_CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
_FROZEN_CONTRACT_SHA256 = (
    "2c4db92868ab37b806b82274f20f31e402b737714ff0a946f4954386db9eca1c"
)
_FROZEN_CONTRACT_RELATIVE = Path(
    "experiments/campaigns"
) / _CAMPAIGN_ID / "frozen_task_contract.yaml"
_FROZEN_CONTRACT_HASH_RELATIVE = _FROZEN_CONTRACT_RELATIVE.with_suffix(
    ".sha256"
)
_PROTOCOL = "held_in_full_core_fixed_budget"
_TASK_FAMILY = "masked_expression_hybrid_count"
_PRIMARY_METRIC = "fit/whole_node/hybrid_loss"
_REPRESENTATION_SCHEMA = (
    "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1"
)
_OBJECTIVE = "equal_weight_balanced_detection_ordinal_positive_huber"
_PILOT_RECEIPT_SCHEMA = "hybrid_count_pilot_gate_v1"
_PILOT_RECEIPT_RELATIVE = Path(
    "scratch/locked_campaigns"
) / _CAMPAIGN_ID / "pilot_gate_receipt.json"
_MODEL_KEYS = {
    "hybrid-count-gat": "hybridgatk1000",
    "hybrid-count-matched-self": "hybridmatchedself",
}
_PUBLIC_VARIANTS = {
    "hybrid-count-gat": "hybrid-gat-k1000",
    "hybrid-count-matched-self": "hybrid-matched-self",
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
_ALIAS_PATTERN = re.compile(r"ANC-(?:0[1-9]|10)\Z")
_EXPECTED_PARAMETER_COUNT = 11_674_880
_PILOT_MAX_VRAM_GIB = 20.5
_PILOT_MAX_AMP_DISCREPANCY = 1e-3
_PILOT_MAX_PROJECTED_HOURS = 6.0
_PILOT_GATE_THRESHOLDS = {
    "peak_allocated_vram_gib_maximum": _PILOT_MAX_VRAM_GIB,
    "fp32_amp_absolute_total_loss_discrepancy_maximum": (
        _PILOT_MAX_AMP_DISCREPANCY
    ),
    "projected_gat_runtime_hours_per_core_maximum": (
        _PILOT_MAX_PROJECTED_HOURS
    ),
}
_PILOT_GATE_REQUIRED_TRUE_FIELDS = (
    "same_frozen_precision_batch",
    "same_evaluation_masks",
    "same_verified_graph",
)
_PILOT_JOB_REQUIRED_TRUE_FIELDS = (
    "verified_bundle",
    "finite_losses_and_gradients",
    "parameter_match",
    "precision_equivalence_passed",
    "peak_vram_passed",
    "projected_runtime_passed",
    "runner_pilot_gate_passed",
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
    "source_version",
}
_ROW_METADATA_FIELDS = {
    "split",
    "mask_mode",
    "mask_replicate",
    "mask_entry_id",
    "mask_seed",
    "mask_checksum",
}


class HybridCountRunnerError(RuntimeError):
    """Raised before a run can violate the frozen hybrid-count contract."""


@dataclass(frozen=True)
class HybridContract:
    model_name: str
    model_key: str
    public_variant: str
    biological_unit_alias: str
    diagnostic_resource_pilot: bool
    uses_graph: bool


def _require_equal(
    actual: Any,
    expected: Any,
    *,
    field: str,
) -> None:
    if actual != expected:
        raise HybridCountRunnerError(
            f"{field} must be {expected!r}; got {actual!r}"
        )


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _assert_alias_only_mapping(value: Any, *, path: str = "config") -> None:
    """Reject direct identifier fields before runner-owned provenance writes."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _normalized_key(raw_key)
            child = f"{path}.{raw_key}"
            if key in _FORBIDDEN_IDENTIFIER_KEYS:
                raise HybridCountRunnerError(
                    f"alias-only configuration prohibits {child}"
                )
            _assert_alias_only_mapping(item, path=child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_alias_only_mapping(item, path=f"{path}[{index}]")


def _validated_alias(config: Mapping[str, Any]) -> str:
    dataset = _section(config, "dataset")
    experiment = _section(config, "experiment")
    dataset_alias = str(dataset.get("biological_unit_alias", ""))
    experiment_alias = str(experiment.get("biological_unit_alias", ""))
    if not _ALIAS_PATTERN.fullmatch(dataset_alias):
        raise HybridCountRunnerError(
            "dataset.biological_unit_alias must be ANC-01 through ANC-10"
        )
    _require_equal(
        experiment_alias,
        dataset_alias,
        field="experiment.biological_unit_alias",
    )
    _require_equal(
        dataset.get("tissue_context"),
        "pathology_confirmed_adjacent_normal",
        field="dataset.tissue_context",
    )
    _require_equal(
        experiment.get("tissue_context"),
        "pathology_confirmed_adjacent_normal",
        field="experiment.tissue_context",
    )
    alias_slug = dataset_alias.lower()
    expected_reference = (
        "data/processed/adjacent_normal_10core_qkv_large_k_v1/"
        f"{alias_slug}/prepared_v1"
    )
    _require_equal(
        dataset.get("prepared_artifact_reference"),
        expected_reference,
        field="dataset.prepared_artifact_reference",
    )
    compact_alias = alias_slug.replace("-", "")
    dataset_id = str(dataset.get("dataset_id", "")).lower()
    if compact_alias not in dataset_id or not re.fullmatch(
        rf"[a-z0-9_]*{re.escape(compact_alias)}[a-z0-9_]*",
        dataset_id,
    ):
        raise HybridCountRunnerError(
            "dataset.dataset_id must use only its ANC alias"
        )
    return dataset_alias


def _validate_count_representation(dataset: Mapping[str, Any]) -> None:
    representation = dataset.get("count_representation")
    if not isinstance(representation, Mapping):
        raise HybridCountRunnerError(
            "dataset.count_representation must explicitly describe the "
            "frozen vocabulary"
        )
    expected = {
        "schema": _REPRESENTATION_SCHEMA,
        "source_scale": "raw_biological_probe_counts",
        "num_output_states": 8,
        "mask_token_id": 8,
        "mask_token_is_output": False,
        "fixed_boundaries": True,
        "fit_required": False,
    }
    for field, required in expected.items():
        _require_equal(
            representation.get(field),
            required,
            field=f"dataset.count_representation.{field}",
        )
    mapping = representation.get("count_mapping")
    label_to_token = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4-7": 4,
        "8-15": 5,
        "16-31": 6,
        "32+": 7,
    }
    if not isinstance(mapping, Mapping) or dict(mapping) != label_to_token:
        raise HybridCountRunnerError(
            "dataset.count_representation.count_mapping must encode exactly "
            "0,1,2,3,4-7,8-15,16-31,32+"
        )
    continuous = representation.get("continuous_channel")
    if not isinstance(continuous, Mapping):
        raise HybridCountRunnerError(
            "dataset.count_representation.continuous_channel is required"
        )
    normalized_values = " ".join(
        str(item).strip().lower()
        for item in continuous.values()
        if isinstance(item, (str, int, float))
    )
    if "log1p" not in normalized_values or "standard" not in normalized_values:
        raise HybridCountRunnerError(
            "continuous_channel must declare per-gene standardized log1p counts"
        )
    masked_value = continuous.get(
        "masked_value", continuous.get("masked_continuous_value")
    )
    if float(masked_value) != 0.0:
        raise HybridCountRunnerError(
            "continuous_channel masked value must be standardized zero"
        )


def _validate_hybrid_contract(config: Mapping[str, Any]) -> HybridContract:
    """Validate every scientific setting owned by the frozen campaign."""

    _assert_alias_only_mapping(config)
    campaign = _section(config, "campaign")
    _require_equal(
        campaign.get("campaign_id"), _CAMPAIGN_ID, field="campaign.campaign_id"
    )
    alias = _validated_alias(config)
    if int(config.get("seed", -1)) != 0:
        raise HybridCountRunnerError("the frozen campaign requires seed=0")
    if (
        int(config.get("fold", -1)) != 0
        or int(config.get("attempt", -1)) not in {1, 2}
    ):
        raise HybridCountRunnerError(
            "the frozen run identity requires fold=0 and attempt in {1,2}"
        )

    model = _section(config, "model")
    model_name = str(model.get("name", "")).strip().lower()
    if model_name not in _MODEL_KEYS:
        raise HybridCountRunnerError(
            "model.name must be hybrid-count-gat or hybrid-count-matched-self"
        )
    uses_graph = model_name == "hybrid-count-gat"
    expected_family = (
        "hybrid_count_edge_conditioned_gatv2"
        if uses_graph
        else "hybrid_count_parameter_matched_self_control"
    )
    fixed_model_values = {
        "family": expected_family,
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
    for field, required in fixed_model_values.items():
        _require_equal(model.get(field), required, field=f"model.{field}")
    head_dim = model.get("attention_head_dim")
    if head_dim is not None and int(head_dim) != 128:
        raise HybridCountRunnerError(
            "model.attention_head_dim must be omitted or equal hidden_dim/heads=128"
        )
    chunk_size = model.get("receiver_chunk_size")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise HybridCountRunnerError("model.receiver_chunk_size must be positive")
    if not uses_graph:
        _require_equal(
            model.get("parameter_match_reference"),
            "hybrid_count_edge_conditioned_gatv2",
            field="model.parameter_match_reference",
        )

    dataset = _section(config, "dataset")
    _require_equal(dataset.get("task"), _TASK_FAMILY, field="dataset.task")
    _require_equal(
        dataset.get("target_scale"),
        "raw_biological_probe_counts_with_per_gene_all_fit_standardized_log1p",
        field="dataset.target_scale",
    )
    _require_equal(
        dataset.get("biological_target_count"),
        1000,
        field="dataset.biological_target_count",
    )
    _require_equal(
        dataset.get("preprocessing_fit_scope"),
        "all_nodes_transductive",
        field="dataset.preprocessing_fit_scope",
    )
    _require_equal(
        dataset.get("validation_or_test_partition_present"),
        False,
        field="dataset.validation_or_test_partition_present",
    )
    if dataset.get("patient_generalization_supported") not in {None, False}:
        raise HybridCountRunnerError(
            "dataset.patient_generalization_supported must be false"
        )
    _validate_count_representation(dataset)

    features = _section(config, "features")
    _require_equal(
        features.get("fit_scope"),
        "all_nodes_transductive",
        field="features.fit_scope",
    )
    _require_equal(
        features.get("use_edge_features"),
        uses_graph,
        field="features.use_edge_features",
    )
    node_expression = features.get("node_expression")
    expected_node_expression = {
        "biological_targets": 1000,
        "source_scale": "raw_biological_probe_counts",
        "discrete_transform": "fixed_hybrid_count_states",
        "continuous_transform": "per_gene_all_fit_standardized_log1p",
        "masked_discrete_value": "input_only_mask_token_8",
        "masked_continuous_value": 0.0,
        "explicit_mask_authoritative_inside_model": True,
    }
    if not isinstance(node_expression, Mapping) or dict(
        node_expression
    ) != expected_node_expression:
        raise HybridCountRunnerError(
            "features.node_expression does not match the frozen hybrid input"
        )
    node_metadata = features.get("node_metadata")
    if not isinstance(node_metadata, Mapping) or tuple(
        node_metadata.get("fields", ())
    ) != _EXPECTED_METADATA_FIELDS:
        raise HybridCountRunnerError(
            "features.node_metadata.fields must be the frozen 22 "
            "morphology/imaging fields"
        )
    _require_equal(
        node_metadata.get("transformed_with"),
        "full_core_fitted_median_imputation_log1p_standardization",
        field="features.node_metadata.transformed_with",
    )
    edge_features = features.get("edge_features")
    if uses_graph:
        if not isinstance(edge_features, Mapping) or tuple(
            edge_features.get("fields", ())
        ) != tuple(EDGE_ATTRIBUTE_NAMES):
            raise HybridCountRunnerError(
                "the GAT requires the frozen 17 measured geometry edge fields"
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
    elif edge_features not in ([], (), None):
        raise HybridCountRunnerError(
            "matched self features.edge_features must be empty"
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
        raise HybridCountRunnerError(
            "features.prohibited_node_inputs omits a frozen leakage prohibition"
        )

    graph = _section(config, "graph")
    fixed_graph_values = {
        "kind": "exact_spatial_knn_radius_guard",
        "symmetry": "mutual",
        "k": 1000,
        "neighbor_k": 1000,
        "radius_um": 2000.0,
        "radius_guard_um": 2000.0,
        "full_core_graph": True,
        "edge_dropout": 0.0,
        "self_loops": False,
        "coordinates_are_node_covariates": False,
    }
    for field, required in fixed_graph_values.items():
        _require_equal(graph.get(field), required, field=f"graph.{field}")
    if graph.get("neighbor_sampling") not in {None, False}:
        raise HybridCountRunnerError("graph.neighbor_sampling is prohibited")
    checksum = graph.get("expected_materialized_graph_sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise HybridCountRunnerError(
            "graph.expected_materialized_graph_sha256 is required"
        )
    edge_count = graph.get("expected_directed_edges")
    if (
        isinstance(edge_count, bool)
        or not isinstance(edge_count, int)
        or edge_count <= 0
    ):
        raise HybridCountRunnerError("graph.expected_directed_edges must be positive")

    masking = _section(config, "masking")
    _require_equal(
        masking.get("type"),
        "mixed_expression_masking",
        field="masking.type",
    )
    _require_equal(masking.get("curriculum"), "P+N+B", field="masking.curriculum")
    _require_equal(masking.get("warmup_epochs"), 10, field="masking.warmup_epochs")
    _require_equal(masking.get("mask_seed"), 314159, field="masking.mask_seed")
    _require_equal(masking.get("block_shape"), "disk", field="masking.block_shape")
    if masking.get("block_width_um") is not None:
        raise HybridCountRunnerError("masking.block_width_um must remain null")
    expected_rates = {
        "partial_gene": 0.2,
        "whole_node": 0.1,
        "spatial_block": 0.1,
    }
    for rate_field in ("rate", "rates"):
        rates = masking.get(rate_field)
        if not isinstance(rates, Mapping) or {
            key: float(rates.get(key, -1.0)) for key in expected_rates
        } != expected_rates:
            raise HybridCountRunnerError(
                f"masking.{rate_field} does not match the frozen schedule"
            )
    probabilities = masking.get("post_warmup_probabilities")
    expected_probabilities = {
        "partial_gene": 0.6,
        "whole_node": 0.3,
        "spatial_block": 0.1,
    }
    if not isinstance(probabilities, Mapping) or {
        key: float(probabilities.get(key, -1.0)) for key in expected_probabilities
    } != expected_probabilities:
        raise HybridCountRunnerError(
            "masking post-warmup probabilities do not match P+N+B"
        )
    _require_equal(
        masking.get("mask_expression_only"), True, field="masking.mask_expression_only"
    )
    _require_equal(
        masking.get("explicit_gene_mask_channel"),
        True,
        field="masking.explicit_gene_mask_channel",
    )

    evaluation = _section(config, "evaluation")
    fixed_evaluation_values = {
        "task_family": _TASK_FAMILY,
        "protocol": _PROTOCOL,
        "canonical_prediction_split": "fit",
        "primary_metric": _PRIMARY_METRIC,
        "primary_direction": "minimize",
        "splits": ["fit"],
        "mask_modes": list(_REQUIRED_PUBLIC_MASKS),
        "fixed_mask_bundle": True,
        "generalization_estimate": False,
        "validation_or_test_selection": False,
    }
    for field, required in fixed_evaluation_values.items():
        _require_equal(evaluation.get(field), required, field=f"evaluation.{field}")

    trainer = _section(config, "trainer")
    diagnostic = trainer.get("diagnostic_resource_pilot") is True
    expected_epochs = 2 if diagnostic else 200
    expected_replicates = 1 if diagnostic else 3
    fixed_trainer_values = {
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "batch_size": 1,
        "neighbor_sampling": False,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "huber_delta": 1.0,
        "max_epochs": expected_epochs,
        "fixed_epoch_budget": True,
        "early_stopping": False,
        "validation_every": None,
        "precision": "mixed",
        "amp": True,
        "amp_requires_fp32_equivalence_smoke": True,
        "deterministic": True,
        "deterministic_warn_only": False,
        "restore_best": False,
        "monitored_metric": None,
        "primary_checkpoint_role": "last",
        "checkpoint_policy": "last_only",
        "objective": _OBJECTIVE,
    }
    for field, required in fixed_trainer_values.items():
        _require_equal(trainer.get(field), required, field=f"trainer.{field}")
    _require_equal(
        evaluation.get("mask_replicates_per_mode"),
        expected_replicates,
        field="evaluation.mask_replicates_per_mode",
    )
    _require_equal(
        evaluation.get("diagnostic_only"),
        diagnostic,
        field="evaluation.diagnostic_only",
    )
    _require_equal(
        evaluation.get("conclusion_bearing"),
        not diagnostic,
        field="evaluation.conclusion_bearing",
    )
    if diagnostic:
        _require_equal(alias, "ANC-01", field="pilot biological_unit_alias")
        _require_equal(
            trainer.get("run_fp32_amp_equivalence"),
            True,
            field="trainer.run_fp32_amp_equivalence",
        )
    elif trainer.get("run_fp32_amp_equivalence") not in {None, False}:
        raise HybridCountRunnerError(
            "production must use the external pilot receipt, not rerun the diagnostic"
        )

    authorization = trainer.get("amp_authorization")
    if not isinstance(authorization, Mapping):
        raise HybridCountRunnerError("trainer.amp_authorization is required")
    expected_mode = (
        "same_batch_fp32_amp_equivalence_diagnostic"
        if diagnostic
        else "require_external_pilot_gate_receipt"
    )
    auth_values = {
        "mode": expected_mode,
        "receipt_schema": _PILOT_RECEIPT_SCHEMA,
        "receipt_reference": _PILOT_RECEIPT_RELATIVE.as_posix(),
        "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
    }
    for field, required in auth_values.items():
        _require_equal(
            authorization.get(field),
            required,
            field=f"trainer.amp_authorization.{field}",
        )

    return HybridContract(
        model_name=model_name,
        model_key=_MODEL_KEYS[model_name],
        public_variant=_PUBLIC_VARIANTS[model_name],
        biological_unit_alias=alias,
        diagnostic_resource_pilot=diagnostic,
        uses_graph=uses_graph,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_contract(project_root: Path) -> Mapping[str, Any]:
    contract_path = project_root / _FROZEN_CONTRACT_RELATIVE
    sidecar_path = project_root / _FROZEN_CONTRACT_HASH_RELATIVE
    if not contract_path.is_file() or contract_path.is_symlink():
        raise HybridCountRunnerError("frozen task contract is missing or unsafe")
    actual = _sha256_file(contract_path)
    if actual != _FROZEN_CONTRACT_SHA256:
        raise HybridCountRunnerError("frozen task contract checksum changed")
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise HybridCountRunnerError("frozen task contract checksum sidecar is missing")
    declared = sidecar_path.read_text(encoding="utf-8").strip().split()[0]
    if declared != actual:
        raise HybridCountRunnerError("frozen task contract checksum sidecar disagrees")
    return {
        "path": _FROZEN_CONTRACT_RELATIVE.as_posix(),
        "sha256": actual,
        "verified": True,
    }


def _validate_production_pilot_receipt(
    project_root: Path,
    contract: HybridContract,
    config: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    if contract.diagnostic_resource_pilot:
        return None
    path = project_root / _PILOT_RECEIPT_RELATIVE
    if not path.is_file() or path.is_symlink():
        raise HybridCountRunnerError(
            "production AMP is blocked until the verified pilot gate receipt exists"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HybridCountRunnerError("pilot gate receipt is unreadable") from error
    if not isinstance(payload, Mapping):
        raise HybridCountRunnerError("pilot gate receipt must be a JSON mapping")
    receipt = dict(payload)
    checksum = receipt.pop("checksum", None)
    if not isinstance(checksum, str) or checksum != canonical_sha256(receipt):
        raise HybridCountRunnerError("pilot gate receipt checksum is invalid")
    metadata = _section(config, "metadata")
    materialization_reference = metadata.get(
        "locked_config_materialization_receipt"
    )
    if not isinstance(materialization_reference, str):
        raise HybridCountRunnerError(
            "metadata.locked_config_materialization_receipt is required"
        )
    materialization_path = project_root / materialization_reference
    if (
        materialization_path != path.parent / "locked_config_materialization.json"
        or not materialization_path.is_file()
        or materialization_path.is_symlink()
    ):
        raise HybridCountRunnerError(
            "locked config materialization receipt is missing or inconsistent"
        )
    try:
        materialization = json.loads(
            materialization_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise HybridCountRunnerError(
            "locked config materialization receipt is unreadable"
        ) from error
    if not isinstance(materialization, Mapping):
        raise HybridCountRunnerError(
            "locked config materialization receipt must be a mapping"
        )
    materialization_core = dict(materialization)
    materialization_checksum = materialization_core.pop("checksum", None)
    if (
        not isinstance(materialization_checksum, str)
        or canonical_sha256(materialization_core) != materialization_checksum
    ):
        raise HybridCountRunnerError(
            "locked config materialization receipt checksum is invalid"
        )
    required = {
        "schema_version": 1,
        "receipt_kind": _PILOT_RECEIPT_SCHEMA,
        "campaign_id": _CAMPAIGN_ID,
        "frozen_contract_sha256": _FROZEN_CONTRACT_SHA256,
        "materialization_checksum": materialization_checksum,
    }
    for field, expected in required.items():
        _require_equal(payload.get(field), expected, field=f"pilot_receipt.{field}")
    if payload.get("gate_passed") is not True:
        raise HybridCountRunnerError("pilot_receipt.gate_passed must be true")
    if payload.get("production_authorized") is not True:
        raise HybridCountRunnerError(
            "pilot_receipt.production_authorized must be true"
        )
    thresholds = payload.get("thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or dict(thresholds) != _PILOT_GATE_THRESHOLDS
    ):
        raise HybridCountRunnerError(
            "pilot receipt does not declare the exact frozen thresholds"
        )
    if any(
        payload.get(field) is not True
        for field in _PILOT_GATE_REQUIRED_TRUE_FIELDS
    ):
        raise HybridCountRunnerError(
            "pilot receipt does not verify the frozen batch, masks, and graph"
        )
    if payload.get("failure_reasons") != []:
        raise HybridCountRunnerError(
            "pilot receipt must contain no failure reasons"
        )
    jobs = payload.get("jobs")
    if (
        not isinstance(jobs, list)
        or len(jobs) != 2
        or not all(isinstance(job, Mapping) for job in jobs)
    ):
        raise HybridCountRunnerError(
            "pilot receipt must contain exactly two job mappings"
        )
    observed = {
        (str(job.get("alias")), str(job.get("arm"))) for job in jobs
    }
    expected_jobs = {
        ("ANC-01", "hybrid-gat-k1000"),
        ("ANC-01", "hybrid-matched-self"),
    }
    if observed != expected_jobs:
        raise HybridCountRunnerError(
            "pilot receipt does not contain the exact ANC-01 paired arms"
        )
    for job in jobs:
        arm = str(job.get("arm"))
        if job.get("parameter_count") != _EXPECTED_PARAMETER_COUNT or any(
            job.get(field) is not True
            for field in _PILOT_JOB_REQUIRED_TRUE_FIELDS
        ):
            raise HybridCountRunnerError(
                f"pilot receipt arm {arm} is unverified or invalid"
            )
    return {
        "path": _PILOT_RECEIPT_RELATIVE.as_posix(),
        "checksum": checksum,
        "materialization_checksum": materialization_checksum,
        "receipt_kind": _PILOT_RECEIPT_SCHEMA,
        "passed": True,
    }


def _model_arguments(
    core: FullCoreData,
    graph: ReceiverSortedGraph,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "num_genes": core.n_genes,
        "edge_attribute_dim": len(graph.edge_attribute_names),
        "expression_mean": core.expression_mean.astype(np.float32).tolist(),
        "expression_scale": core.expression_scale.astype(np.float32).tolist(),
        "node_covariate_dim": int(core.node_covariates.shape[1]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "attention_heads": int(model_config["attention_heads"]),
        "attention_head_dim": (
            None
            if model_config.get("attention_head_dim") is None
            else int(model_config["attention_head_dim"])
        ),
        "graph_layers": int(model_config["graph_layers"]),
        "ffn_dim": int(model_config["ffn_dim"]),
        "decoder_dim": int(model_config["decoder_dim"]),
        "edge_hidden_dim": int(model_config["edge_hidden_dim"]),
        "edge_embedding_dim": int(model_config["edge_embedding_dim"]),
        "dropout": float(model_config["dropout"]),
        "attention_dropout": float(model_config["attention_dropout"]),
    }


def _named_shapes(module: torch.nn.Module) -> dict[str, list[int]]:
    return {
        name: list(parameter.shape)
        for name, parameter in module.named_parameters()
    }


def _state_dicts_bit_identical(
    first: torch.nn.Module,
    second: torch.nn.Module,
) -> bool:
    first_state = first.state_dict()
    second_state = second.state_dict()
    return list(first_state) == list(second_state) and all(
        first_state[name].dtype == second_state[name].dtype
        and first_state[name].shape == second_state[name].shape
        and torch.equal(first_state[name], second_state[name])
        for name in first_state
    )


def _paired_models(
    *,
    core: FullCoreData,
    graph: ReceiverSortedGraph,
    model_config: Mapping[str, Any],
    selected_model_name: str,
    seed: int,
) -> tuple[torch.nn.Module, Mapping[str, Any], Mapping[str, Any]]:
    """Build both arms, enforce structural parity, and retain one arm."""

    common = _model_arguments(core, graph, model_config)
    set_deterministic_seed(seed, deterministic=True, warn_only=False)
    graph_model = HybridReceiverChunkedEdgeConditionedGATv2(
        **common,
        receiver_chunk_size=int(model_config["receiver_chunk_size"]),
        activation_checkpointing=bool(model_config["activation_checkpointing"]),
    )
    set_deterministic_seed(seed, deterministic=True, warn_only=False)
    self_model = HybridEdgeParameterMatchedSelfControl(**common)
    parameter_count = assert_exact_parameter_match(graph_model, self_model)
    if parameter_count != _EXPECTED_PARAMETER_COUNT:
        raise HybridCountRunnerError(
            "frozen hybrid architecture parameter count changed: "
            f"expected {_EXPECTED_PARAMETER_COUNT}, got {parameter_count}"
        )
    if graph_model.encoder.__class__ is not self_model.encoder.__class__:
        raise HybridCountRunnerError("paired hybrid encoders use different classes")
    if graph_model.decoder.__class__ is not self_model.decoder.__class__:
        raise HybridCountRunnerError("paired hybrid decoders use different classes")
    encoder_shapes = _named_shapes(graph_model.encoder)
    decoder_shapes = _named_shapes(graph_model.decoder)
    if encoder_shapes != _named_shapes(self_model.encoder):
        raise HybridCountRunnerError("paired hybrid encoder structures differ")
    if decoder_shapes != _named_shapes(self_model.decoder):
        raise HybridCountRunnerError("paired hybrid decoder structures differ")
    try:
        self_model.encoder.load_state_dict(
            graph_model.encoder.state_dict(), strict=True
        )
        self_model.decoder.load_state_dict(
            graph_model.decoder.state_dict(), strict=True
        )
    except RuntimeError as error:
        raise HybridCountRunnerError(
            "paired hybrid encoder/decoder initialization copy failed"
        ) from error
    encoder_initial_state_identical = _state_dicts_bit_identical(
        graph_model.encoder, self_model.encoder
    )
    decoder_initial_state_identical = _state_dicts_bit_identical(
        graph_model.decoder, self_model.decoder
    )
    if not encoder_initial_state_identical:
        raise HybridCountRunnerError(
            "paired hybrid encoder initial states differ"
        )
    if not decoder_initial_state_identical:
        raise HybridCountRunnerError(
            "paired hybrid decoder initial states differ"
        )
    graph_block_budgets = [
        sum(parameter.numel() for parameter in block.parameters())
        for block in graph_model.blocks
    ]
    self_block_budgets = [
        sum(parameter.numel() for parameter in block.parameters())
        for block in self_model.blocks
    ]
    if graph_block_budgets != self_block_budgets or len(graph_block_budgets) != 2:
        raise HybridCountRunnerError("paired graph/self layer parameter budgets differ")
    graph_keys = len(graph_model.state_dict())
    self_keys = len(self_model.state_dict())
    if graph_keys != self_keys:
        raise HybridCountRunnerError("paired graph/self state-key counts differ")

    audit = {
        "schema": "hybrid_count_parameter_structure_audit_v1",
        "graph_implementation": (
            f"{graph_model.__class__.__module__}.{graph_model.__class__.__qualname__}"
        ),
        "self_implementation": (
            f"{self_model.__class__.__module__}.{self_model.__class__.__qualname__}"
        ),
        "trainable_parameter_count_graph": parameter_count,
        "trainable_parameter_count_self": parameter_count,
        "exact_trainable_parameter_match": True,
        "encoder_class_identical": True,
        "decoder_class_identical": True,
        "encoder_parameter_shapes_identical": True,
        "decoder_parameter_shapes_identical": True,
        "shared_initialization_source_arm": "hybrid-gat-k1000",
        "shared_initialization_destination_arm": "hybrid-matched-self",
        "encoder_initial_state_bit_identical": (
            encoder_initial_state_identical
        ),
        "decoder_initial_state_bit_identical": (
            decoder_initial_state_identical
        ),
        "graph_layer_parameter_budgets": graph_block_budgets,
        "self_layer_parameter_budgets": self_block_budgets,
        "state_key_count_graph": graph_keys,
        "state_key_count_self": self_keys,
        "full_parameter_shape_multiset_required_identical": False,
        "reason": (
            "attention tensors are repurposed into within-cell routing tensors; "
            "total count and hybrid encoder/decoder structures are the "
            "matching contract"
        ),
    }
    if selected_model_name == "hybrid-count-gat":
        selected = graph_model
        del self_model
        constructor = {
            **common,
            "receiver_chunk_size": int(model_config["receiver_chunk_size"]),
            "activation_checkpointing": bool(model_config["activation_checkpointing"]),
        }
    else:
        selected = self_model
        del graph_model
        constructor = common
    gc.collect()
    construction = {
        "canonical_model_key": _MODEL_KEYS[selected_model_name],
        "implementation_class": (
            f"{selected.__class__.__module__}.{selected.__class__.__qualname__}"
        ),
        "constructor_arguments": constructor,
    }
    return selected, construction, audit


def _array_sha256(name: str, value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _reference_provenance(references: HybridCountReferences) -> dict[str, Any]:
    return {
        "audit": dict(references.audit),
        "detection_probability": references.detection_probability.tolist(),
        "positive_ordinal_probability": (
            references.positive_ordinal_probability.tolist()
        ),
        "positive_continuous_standardized": (
            references.positive_continuous_standardized.tolist()
        ),
        "decoded_detected_state": references.detected_state.astype(bool).tolist(),
        "decoded_positive_state": references.positive_state.astype(int).tolist(),
        "decoded_count_state": references.count_state.astype(int).tolist(),
    }


def _whole_node_replicate_zero_mask(bundle: Any) -> np.ndarray:
    matches = [
        entry
        for entry in bundle.manifest["entries"]
        if str(entry["spec"]["mode"]) == "node"
        and int(entry["replicate"]) == 0
    ]
    if len(matches) != 1:
        raise HybridCountRunnerError(
            "fixed masks do not contain exactly one whole-node replicate zero"
        )
    return bundle.masks[str(matches[0]["entry_id"])]


def _precision_record(result: PrecisionEquivalenceResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "performed": False,
            "authorization": "external_verified_pilot_gate_receipt",
        }
    return {
        "performed": True,
        **asdict(result),
        "maximum_allowed_discrepancy": _PILOT_MAX_AMP_DISCREPANCY,
        "actual_amp_dtype": (
            "float16" if result.amp_dtype == "auto" else result.amp_dtype
        ),
    }


def _checkpoint_bytes(
    *,
    archive: RunArchive,
    contract: HybridContract,
    model_config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    parameter_audit: Mapping[str, Any],
    training: HybridCountTrainingResult,
    graph: ReceiverSortedGraph,
    core: FullCoreData,
    masks: Any,
    references: HybridCountReferences,
    effective_amp: bool,
) -> bytes:
    payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "checkpoint_role": "last",
        "checkpoint_policy": "final_epoch_no_validation_selection",
        "training_protocol": training.training_protocol,
        "task_family": _TASK_FAMILY,
        "model_name": contract.model_name,
        "public_variant": contract.public_variant,
        "biological_unit_alias": contract.biological_unit_alias,
        "model_config": dict(model_config),
        "model_construction": dict(model_construction),
        "parameter_structure_audit": dict(parameter_audit),
        "epoch": training.final_epoch,
        "fixed_epoch_budget": training.fixed_epoch_budget,
        "model_state_dict": dict(training.final_state_dict),
        "state_dict_sha256": training.final_state_checksum,
        "full_core_preprocessing_sha256": core.checksums.preprocessing_sha256,
        "raw_count_sha256": core.checksums.expression_counts_sha256,
        "graph_sha256": graph.checksums.graph_sha256,
        "evaluation_mask_bundle_sha256": masks.checksum,
        "reference_sha256": references.audit["reference_sha256"],
        "count_representation_schema": _REPRESENTATION_SCHEMA,
        "objective": _OBJECTIVE,
        "effective_amp": effective_amp,
        "monitored_metric": None,
        "selection_policy": "last_epoch_without_validation_selection",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _replicate_metric_row(
    *,
    entry: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    safe = json_safe_metrics(metrics)
    row: dict[str, Any] = {
        "split": "fit",
        "mask_mode": _PUBLIC_MASK_NAMES[str(entry["spec"]["mode"])],
        "mask_replicate": int(entry["replicate"]),
        "mask_entry_id": str(entry["entry_id"]),
        "mask_seed": int(entry["seed"]),
        "mask_checksum": str(entry["mask_checksum"]),
        **safe,
    }
    # Keep the requested list-valued support/recall columns intact.  These
    # lists preserve explicit nulls for unsupported states.
    for prefix, states in (("state8", 8), ("collapsed4", 4)):
        support = row.get(f"{prefix}_support")
        recall = row.get(f"{prefix}_recall")
        if not isinstance(support, list) or len(support) != states:
            raise HybridCountRunnerError(f"{prefix} support vector is malformed")
        if not isinstance(recall, list) or len(recall) != states:
            raise HybridCountRunnerError(f"{prefix} recall vector is malformed")
    return row


def _finite_scalar(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _final_metrics(
    *,
    replicate_rows: Sequence[Mapping[str, Any]],
    training: HybridCountTrainingResult,
    training_duration: float,
    evaluation_duration: float,
    total_duration: float,
    parameter_count: int,
    checkpoint_size: int,
    replicates_per_mode: int,
    graph_duration: float,
    data_duration: float,
    peak_vram_bytes: int,
    projected_runtime_hours: float | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    candidate_fields = sorted(
        {
            key
            for row in replicate_rows
            for key, value in row.items()
            if key not in _ROW_METADATA_FIELDS
            and key != "n_masked"
            and (_finite_scalar(value) or value is None)
        }
    )
    for mode in _REQUIRED_PUBLIC_MASKS:
        selected = [row for row in replicate_rows if row["mask_mode"] == mode]
        if len(selected) != replicates_per_mode:
            raise HybridCountRunnerError(
                f"expected {replicates_per_mode} evaluation rows for {mode}"
            )
        for field in candidate_fields:
            value = _mean_finite(selected, field)
            # Requested metrics remain explicit even if every replicate is
            # undefined (notably detection precision).
            metrics[f"fit/{mode}/{field}"] = value
    metrics.update(
        {
            "resource/data_preparation_duration_seconds": data_duration,
            "resource/graph_construction_duration_seconds": graph_duration,
            "resource/training_duration_seconds": training_duration,
            "resource/inference_duration_seconds": evaluation_duration,
            "resource/total_duration_seconds": total_duration,
            "resource/parameter_count": parameter_count,
            "resource/checkpoint_size_bytes": checkpoint_size,
            "resource/peak_allocated_vram_bytes": peak_vram_bytes,
            "resource/peak_vram_gib": peak_vram_bytes / (1024**3),
            "resource/projected_200_epoch_runtime_hours": projected_runtime_hours,
            "training/final_epoch": training.final_epoch,
            "training/final_hybrid_loss": training.final_train_loss,
        }
    )
    return metrics


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
        raise HybridCountRunnerError("hybrid prediction arrays are not aligned")
    namespace = (
        f"bagm:{dataset['dataset_id']}:{dataset['version']}:full-core-fit"
    )
    public_mode = _PUBLIC_MASK_NAMES[str(entry["spec"]["mode"])]
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
                "graph_id": graph_id,
                "dataset_id": str(dataset["dataset_id"]),
                "split": "fit",
                "fold": fold,
                "y_true": truth.astype(int).tolist(),
                "y_pred": reconstructed[local_index, target_indices].astype(
                    float
                ).tolist(),
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
                "masking_type": public_mode,
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


def _cuda_peak_bytes(device: str) -> int:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(resolved))


def _resource_diagnostic(
    *,
    contract: HybridContract,
    configured_training: TrainingConfig,
    effective_training: TrainingConfig,
    precision: PrecisionEquivalenceResult | None,
    training: HybridCountTrainingResult,
    training_duration: float,
    evaluation_duration: float,
    total_duration: float,
    peak_vram_bytes: int,
    projected_runtime_hours: float | None,
    parameter_count: int,
) -> dict[str, Any]:
    device = torch.device(effective_training.device or training.device)
    epoch_durations = [record.duration_seconds for record in training.history]
    diagnostic_pass = True if precision is None else precision.passed
    vram_pass = peak_vram_bytes / (1024**3) <= _PILOT_MAX_VRAM_GIB
    runtime_pass = (
        True
        if not contract.uses_graph or projected_runtime_hours is None
        else projected_runtime_hours <= _PILOT_MAX_PROJECTED_HOURS
    )
    pilot_gate_passed = (
        diagnostic_pass
        and vram_pass
        and runtime_pass
        and parameter_count == _EXPECTED_PARAMETER_COUNT
    )
    return {
        "schema": "hybrid_count_resource_diagnostic_v1",
        "diagnostic_resource_pilot": contract.diagnostic_resource_pilot,
        "public_variant": contract.public_variant,
        "biological_unit_alias": contract.biological_unit_alias,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "configured_amp": configured_training.amp,
        "effective_training_amp": effective_training.amp,
        "amp_fallback_reason": (
            "fp32_amp_equivalence_threshold_failed"
            if configured_training.amp and not effective_training.amp
            else None
        ),
        "precision_equivalence": _precision_record(precision),
        "parameter_count": parameter_count,
        "finite_losses_and_gradients": True,
        "epochs_completed": len(training.history),
        "epoch_duration_seconds": epoch_durations,
        "mean_measured_epoch_duration_seconds": float(np.mean(epoch_durations)),
        "projected_200_epoch_runtime_hours": projected_runtime_hours,
        "projection_basis": (
            "mean_measured_two_epoch_partial_gene_warmup_training_duration_times_200"
            if contract.diagnostic_resource_pilot
            else None
        ),
        "training_duration_seconds": training_duration,
        "evaluation_duration_seconds": evaluation_duration,
        "total_pilot_or_run_duration_seconds": total_duration,
        "peak_allocated_vram_bytes": peak_vram_bytes,
        "peak_allocated_vram_gib": peak_vram_bytes / (1024**3),
        "pilot_thresholds": {
            "peak_allocated_vram_gib_maximum": _PILOT_MAX_VRAM_GIB,
            "fp32_amp_absolute_total_loss_discrepancy_maximum": (
                _PILOT_MAX_AMP_DISCREPANCY
            ),
            "projected_gat_runtime_hours_per_core_maximum": (
                _PILOT_MAX_PROJECTED_HOURS
            ),
        },
        "pilot_checks": {
            "precision_equivalence_passed": diagnostic_pass,
            "peak_vram_passed": vram_pass,
            "projected_runtime_passed": runtime_pass,
            "parameter_match_passed": parameter_count == _EXPECTED_PARAMETER_COUNT,
        },
        "pilot_gate_passed": (
            pilot_gate_passed if contract.diagnostic_resource_pilot else None
        ),
        "production_authorization_decision_owned_by": (
            "external_two_arm_pilot_gate_receipt"
        ),
    }


def run_hybrid_count_capacity(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    sample_key_salt: str,
    full_core_data: FullCoreData | None = None,
    receiver_graph: ReceiverSortedGraph | None = None,
) -> CapacityRunResult:
    """Execute one frozen hybrid-count arm in a worker-owned active archive."""

    started = time.monotonic()
    if len(sample_key_salt.encode("utf-8")) < 16:
        raise RunValidationError(
            "BAGM_SAMPLE_KEY_SALT must contain at least 16 bytes"
        )
    contract = _validate_hybrid_contract(config)
    frozen_verification = _verify_frozen_contract(archive.paths.project_root)
    pilot_receipt = _validate_production_pilot_receipt(
        archive.paths.project_root, contract, config
    )
    model_config = _section(config, "model")
    graph_config = _section(config, "graph")
    dataset = _section(config, "dataset")
    evaluation = _section(config, "evaluation")

    data_started = time.monotonic()
    core = (
        full_core_data
        if full_core_data is not None
        else load_and_refit_full_core(_resolve_prepared_artifact(config, archive))
    )
    counts = validate_raw_counts(core.expression_counts, name="expression_counts")
    if core.n_genes != 1000 or core.node_covariates.shape[1] != 22:
        raise HybridCountRunnerError(
            "materialized core does not contain 1000 genes and 22 permitted covariates"
        )
    if tuple(core.metadata_names) != _EXPECTED_METADATA_FIELDS:
        raise HybridCountRunnerError(
            "materialized morphology/imaging fields differ from the frozen schema"
        )
    data_duration = time.monotonic() - data_started
    materialized_identity = _verify_materialized_identity(core, dataset)
    references = fit_hybrid_count_references(
        counts,
        expression_mean=core.expression_mean,
        expression_scale=core.expression_scale,
    )

    graph_started = time.monotonic()
    graph = (
        receiver_graph
        if receiver_graph is not None
        else _build_graph(core, graph_config)
    )
    graph_duration = time.monotonic() - graph_started
    if graph.n_nodes != core.n_nodes or graph.k != 1000:
        raise HybridCountRunnerError("exact mutual-k1000 graph is not aligned")
    if graph.edge_attribute_names != EDGE_ATTRIBUTE_NAMES:
        raise HybridCountRunnerError(
            "graph does not contain the frozen 17 edge features"
        )
    if (
        not graph.qc.receiver_sorted
        or graph.qc.self_loops
        or graph.qc.duplicate_directed_edges
    ):
        raise HybridCountRunnerError(
            "graph violates receiver sorting, no-loop, or uniqueness invariants"
        )
    if not graph.qc.directed_edge_pairs_are_symmetric:
        raise HybridCountRunnerError("graph is not exact mutual symmetric")
    _verify_graph_identity(graph, graph_config)

    training_config = _training_config(config)
    view = _fit_view(
        core=core,
        graph=graph,
        uses_graph=contract.uses_graph,
        expression=counts,
    )
    masks = _evaluation_masks(config, core, training_config)
    model, model_construction, parameter_audit = _paired_models(
        core=core,
        graph=graph,
        model_config=model_config,
        selected_model_name=contract.model_name,
        seed=training_config.model_seed,
    )
    parameter_count = int(parameter_audit["trainable_parameter_count_graph"])

    archive.write_json(
        "diagnostics/full_core_preprocessing.json",
        core.preprocessing_qc.to_dict(),
    )
    archive.write_json("diagnostics/graph_statistics.json", graph.qc.to_dict())
    archive.write_json(
        "diagnostics/raw_count_representation.json",
        {
            "schema": _REPRESENTATION_SCHEMA,
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
            "mask_token_id": 8,
            "mask_token_is_output": False,
            "continuous_masked_value": 0.0,
            "thresholds_fitted": False,
            "reference_audit": dict(references.audit),
        },
    )
    archive.write_json("diagnostics/parameter_structure_audit.json", parameter_audit)
    archive.write_json(
        "diagnostics/mask_statistics.json",
        {
            "role": "held_in_fit_technical_replicates",
            "biological_unit_alias": contract.biological_unit_alias,
            "independent_biological_replicates": 1,
            "manifest": masks.manifest,
        },
    )
    archive.write_json("provenance/frozen_task_contract.json", frozen_verification)
    archive.write_json(
        "provenance/full_core_inputs.json",
        {
            "biological_unit_alias": contract.biological_unit_alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "fit_scope": "all_nodes_transductive",
            "generalization_estimate": False,
            "prepared_artifact_reference": dataset["prepared_artifact_reference"],
            "preprocessing_checksums": core.checksums.to_dict(),
            "materialized_identity_verification": materialized_identity,
            "graph_checksums": graph.checksums.to_dict(),
            "graph_config": dict(graph_config),
            "graph_interpretation": "broad_regional_context_not_direct_interaction",
            "task_family": _TASK_FAMILY,
            "expression_representation": _REPRESENTATION_SCHEMA,
            "data_preparation_duration_seconds": data_duration,
            "graph_construction_duration_seconds": graph_duration,
        },
    )
    archive.write_json(
        "provenance/hybrid_count_references.json",
        _reference_provenance(references),
    )
    archive.write_json(
        "provenance/fixed_evaluation_masks.json",
        {
            "bundle_manifest": masks.manifest,
            "seed_namespace": "held-in-full-core-fixed-evaluation",
            "seed_derivation_relationship": "separate_from_epoch_mask_seed_derivation",
            "entrywise_holdout": False,
            "entries_may_overlap_training_masks": True,
            "used_for_gradient_updates": False,
            "used_for_checkpoint_selection": False,
            "technical_replicates_not_biological_replicates": True,
        },
    )
    archive.write_json(
        "provenance/amp_authorization.json",
        {
            "configured": dict(_section(config, "trainer")["amp_authorization"]),
            "production_pilot_receipt": pilot_receipt,
        },
    )

    precision_result: PrecisionEquivalenceResult | None = None
    effective_training = training_config
    if contract.diagnostic_resource_pilot:
        precision_result = compare_fp32_amp_loss(
            model,
            view,
            _whole_node_replicate_zero_mask(masks),
            expression_mean=core.expression_mean,
            expression_scale=core.expression_scale,
            device=training_config.device or "cuda",
            amp_dtype=training_config.amp_dtype,
            maximum_discrepancy=_PILOT_MAX_AMP_DISCREPANCY,
        )
        archive.write_json(
            "diagnostics/fp32_amp_equivalence.json",
            _precision_record(precision_result),
        )
        if not precision_result.passed:
            # The failed threshold remains completed diagnostic evidence, but
            # mixed precision is not authorized for gradient updates.
            effective_training = replace(training_config, amp=False)

    training_started = time.monotonic()
    training_result = fit_full_core_hybrid_count_model(
        model,
        view,
        effective_training,
        expression_mean=core.expression_mean,
        expression_scale=core.expression_scale,
    )
    training_duration = time.monotonic() - training_started
    history_rows = [
        {
            "run_id": archive.run_id,
            "split": "fit",
            "training_protocol": training_result.training_protocol,
            **row,
        }
        for row in training_result.history_rows()
    ]
    archive.write_table("metrics/history", history_rows, fallback="jsonl")

    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt",
        _checkpoint_bytes(
            archive=archive,
            contract=contract,
            model_config=model_config,
            model_construction=model_construction,
            parameter_audit=parameter_audit,
            training=training_result,
            graph=graph,
            core=core,
            masks=masks,
            references=references,
            effective_amp=effective_training.amp,
        ),
    )
    checkpoint_size = checkpoint_path.stat().st_size
    archive.write_json(
        "provenance/full_core_training.json",
        {
            "training_protocol": training_result.training_protocol,
            "graph_execution": training_result.graph_execution,
            "checkpoint_policy": training_result.checkpoint_policy,
            "final_epoch": training_result.final_epoch,
            "fixed_epoch_budget": training_result.fixed_epoch_budget,
            "state_dict_sha256": training_result.final_state_checksum,
            "model_seed": effective_training.model_seed,
            "epoch_mask_seed": effective_training.mask_seed,
            "parameter_count": parameter_count,
            "device": training_result.device,
            "model_construction": model_construction,
            "parameter_structure_audit": parameter_audit,
            "objective": _OBJECTIVE,
            "configured_amp": training_config.amp,
            "effective_amp": effective_training.amp,
        },
    )

    replicate_rows: list[dict[str, Any]] = []
    fold = int(config.get("fold", 0))
    graph_id = (
        f"exact_mutual_k1000_{graph.checksums.graph_sha256[:16]}"
        if contract.uses_graph
        else f"self_only_paired_graph_{graph.checksums.graph_sha256[:16]}"
    )
    prediction_edge_count = graph.qc.n_directed_edges if contract.uses_graph else 0
    evaluation_started = time.monotonic()

    def prediction_rows() -> Iterator[dict[str, Any]]:
        for entry in masks.manifest["entries"]:
            result = evaluate_fixed_hybrid_count_mask(
                model,
                view,
                masks.masks[str(entry["entry_id"])],
                expression_mean=core.expression_mean,
                expression_scale=core.expression_scale,
                references=references,
                device=effective_training.device,
                amp=effective_training.amp,
                amp_dtype=effective_training.amp_dtype,
            )
            row = _replicate_metric_row(
                entry=entry, metrics=result.evaluation.metrics
            )
            replicate_rows.append(row)
            public_mode = str(row["mask_mode"])
            for name, value in row.items():
                if name in _ROW_METADATA_FIELDS or not _finite_scalar(value):
                    continue
                archive.append_metric_event(
                    {
                        "name": f"fit/{public_mode}/{name}",
                        "value": value,
                        "mask_replicate": int(entry["replicate"]),
                        "mask_seed": int(entry["seed"]),
                    }
                )
            if public_mode == "whole_node" and int(entry["replicate"]) == 0:
                yield from _protected_prediction_rows(
                    archive=archive,
                    dataset=dataset,
                    graph_id=graph_id,
                    edge_count=prediction_edge_count,
                    fold=fold,
                    entry=entry,
                    result=result,
                    sample_key_salt=sample_key_salt,
                )

    prediction_path = archive.write_prediction_jsonl_stream("fit", prediction_rows())
    evaluation_duration = time.monotonic() - evaluation_started
    archive.write_table(
        "metrics/evaluation_replicates", replicate_rows, fallback="jsonl"
    )
    end_cuda_peak = _cuda_peak_bytes(
        effective_training.device or training_result.device
    )
    precision_peak = (
        0
        if precision_result is None
        else precision_result.peak_cuda_memory_bytes
    )
    history_peak = max(
        record.peak_cuda_memory_bytes for record in training_result.history
    )
    peak_vram_bytes = max(precision_peak, history_peak, end_cuda_peak)
    projected_runtime_hours = (
        float(np.mean([record.duration_seconds for record in training_result.history]))
        * 200.0
        / 3600.0
        if contract.diagnostic_resource_pilot
        else None
    )
    total_duration = time.monotonic() - started
    final_metrics = _final_metrics(
        replicate_rows=replicate_rows,
        training=training_result,
        training_duration=training_duration,
        evaluation_duration=evaluation_duration,
        total_duration=total_duration,
        parameter_count=parameter_count,
        checkpoint_size=checkpoint_size,
        replicates_per_mode=int(evaluation["mask_replicates_per_mode"]),
        graph_duration=graph_duration,
        data_duration=data_duration,
        peak_vram_bytes=peak_vram_bytes,
        projected_runtime_hours=projected_runtime_hours,
    )
    primary_value = final_metrics[_PRIMARY_METRIC]
    if not _finite_scalar(primary_value):
        raise HybridCountRunnerError("primary hybrid loss is missing or non-finite")
    for name, value in final_metrics.items():
        if _finite_scalar(value):
            archive.append_metric_event(
                {"name": name, "value": value, "phase": "final_aggregate"}
            )
    archive.write_json("metrics/final.json", final_metrics)

    losses = np.asarray(
        [record.train_hybrid_loss for record in training_result.history],
        dtype=np.float64,
    )
    tail = losses[-min(20, len(losses)) :]
    convergence_slope = (
        float(np.polyfit(np.arange(len(tail), dtype=np.float64), tail, 1)[0])
        if len(tail) >= 2
        else None
    )
    archive.write_json(
        "diagnostics/training_convergence.json",
        {
            "objective": _OBJECTIVE,
            "final_epoch": training_result.final_epoch,
            "final_train_hybrid_loss": training_result.final_train_loss,
            "minimum_observed_train_hybrid_loss": float(losses.min()),
            "last_20_epoch_loss_slope": convergence_slope,
            "all_epochs_completed": len(training_result.history)
            == training_result.fixed_epoch_budget,
            "all_losses_and_gradients_finite": True,
            "final_components": {
                "detection_bce": training_result.history[-1].train_detection_bce,
                "ordinal_bce": training_result.history[-1].train_ordinal_bce,
                "positive_continuous_huber": training_result.history[
                    -1
                ].train_positive_continuous_huber,
            },
        },
    )
    resource_diagnostic = _resource_diagnostic(
        contract=contract,
        configured_training=training_config,
        effective_training=effective_training,
        precision=precision_result,
        training=training_result,
        training_duration=training_duration,
        evaluation_duration=evaluation_duration,
        total_duration=total_duration,
        peak_vram_bytes=peak_vram_bytes,
        projected_runtime_hours=projected_runtime_hours,
        parameter_count=parameter_count,
    )
    archive.write_json("diagnostics/resource_usage.json", resource_diagnostic)

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
        "model_name": contract.model_name,
        "public_variant": contract.public_variant,
        "model_seed": effective_training.model_seed,
        "final_epoch": training_result.final_epoch,
        "fixed_epoch_budget": training_result.fixed_epoch_budget,
        "checkpoint_role": "last",
        "checkpoint": {
            "role": "last",
            "final_epoch": training_result.final_epoch,
            "policy": "final_epoch_no_validation_selection",
            "monitored_metric": None,
            "state_dict_sha256": training_result.final_state_checksum,
        },
        "primary_metric_name": _PRIMARY_METRIC,
        "primary_metric_value": float(primary_value),
        "metrics": final_metrics,
        "parameter_count": parameter_count,
        "exact_parameter_match": True,
        "duration_seconds": total_duration,
        "peak_vram_gib": peak_vram_bytes / (1024**3),
        "peak_host_memory_bytes": _peak_host_memory_bytes(),
        "graph_sha256": graph.checksums.graph_sha256,
        "graph_directed_edges": graph.qc.n_directed_edges,
        "graph_supplied_to_model": contract.uses_graph,
        "graph_interpretation": "broad_regional_context_not_direct_interaction",
        "evaluation_mask_bundle_sha256": masks.checksum,
        "evaluation_mask_replicates_per_mode": int(
            evaluation["mask_replicates_per_mode"]
        ),
        "evaluation_metrics_include_all_configured_replicates_per_mode": True,
        "canonical_prediction_selection": {
            "split": "fit",
            "mask_mode": "whole_node",
            "mask_replicate": 0,
            "selection_status": "prespecified",
        },
        "diagnostic_resource_pilot": contract.diagnostic_resource_pilot,
        "pilot_gate_passed": resource_diagnostic["pilot_gate_passed"],
        "conclusion_eligible": not contract.diagnostic_resource_pilot,
        "generalization_estimate": False,
        "maximum_claim": (
            "diagnostic runtime, memory, and precision feasibility only"
            if contract.diagnostic_resource_pilot
            else (
                "held-in masked-expression representation capacity and possible "
                "broad-context graph gain in one adjacent-normal core"
            )
        ),
        "prohibited_claims": [
            "patient-held-out generalization",
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
            "Run one worker-owned frozen adjacent-normal hybrid-count capacity job."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    result = run_hybrid_count_capacity(
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
                "pilot_gate_passed": result.summary.get("pilot_gate_passed"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
