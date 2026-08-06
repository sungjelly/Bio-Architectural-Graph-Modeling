#!/usr/bin/env python3
"""Materialize the locked ten-core hybrid-count campaign configurations.

This utility is deliberately preparation-only.  It verifies the frozen task
contract, the prior alias-safe multicore materialization, every source
prepared artifact, and the hybrid parameter match before writing anything.
It then publishes two ANC-01 resource-pilot configs, twenty production
configs, and one checksum-bound receipt through a single directory rename.

It does not register, enqueue, train, or create the pilot-gate receipt.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.artifacts import load_prepared_artifact  # noqa: E402
from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.full_core import (  # noqa: E402
    ALLOWED_METADATA_COLUMNS,
    EDGE_ATTRIBUTE_NAMES,
    load_and_refit_full_core,
)
from spatial_benchmark.hybrid_count import (  # noqa: E402
    HybridEdgeParameterMatchedSelfControl,
    HybridReceiverChunkedEdgeConditionedGATv2,
    assert_exact_parameter_match,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402


CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
SOURCE_CAMPAIGN_ID = "cmp_20260728_adjacent_normal_10core_qkv_large_k"
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
ARMS = ("hybrid-gat-k1000", "hybrid-matched-self")
SAFE_GPU_IDS = (0, 1, 2, 3, 5, 6, 7)
EXPECTED_SELECTION_SHA256 = (
    "a460de90ba9998e6817cdf3fbbcd2416eec048ec3fe6c825db8677b70b8b297f"
)
EXPECTED_FROZEN_CONTRACT_SHA256 = (
    "2c4db92868ab37b806b82274f20f31e402b737714ff0a946f4954386db9eca1c"
)
RECEIPT_NAME = "locked_config_materialization.json"
PILOT_GATE_RECEIPT_NAME = "pilot_gate_receipt.json"
RECEIPT_KIND = "hybrid_count_locked_config_materialization_v1"
PILOT_GATE_RECEIPT_KIND = "hybrid_count_pilot_gate_v1"

_MODEL_COMPONENTS = {
    "hybrid-gat-k1000": "configs/model/hybrid_count_gat.yaml",
    "hybrid-matched-self": "configs/model/hybrid_count_matched_self.yaml",
}
_TRAINER_COMPONENTS = {
    "pilot": "configs/trainer/full_core_hybrid_resource_pilot_2.yaml",
    "production": "configs/trainer/full_core_hybrid_fixed_200.yaml",
}
_EVALUATION_COMPONENTS = {
    "pilot": "configs/evaluation/held_in_full_core_hybrid_count_pilot_v1.yaml",
    "production": "configs/evaluation/held_in_full_core_hybrid_count_v1.yaml",
}
_LAUNCHER_COMPONENT = "configs/launcher/local_single_gpu_3090.yaml"

_FROZEN_MASKING: dict[str, Any] = {
    "type": "mixed_expression_masking",
    "curriculum": "P+N+B",
    "rate": {
        "partial_gene": 0.2,
        "whole_node": 0.1,
        "spatial_block": 0.1,
    },
    "rates": {
        "partial_gene": 0.2,
        "whole_node": 0.1,
        "spatial_block": 0.1,
    },
    "post_warmup_probabilities": {
        "partial_gene": 0.6,
        "whole_node": 0.3,
        "spatial_block": 0.1,
    },
    "warmup_epochs": 10,
    "block_shape": "disk",
    "block_width_um": None,
    "mask_seed": 314159,
    "validation_replicates": 0,
    "test_replicates": 0,
    "mask_expression_only": True,
    "explicit_gene_mask_channel": True,
}

_COUNT_REPRESENTATION: dict[str, Any] = {
    "schema": "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1",
    "source_scale": "raw_biological_probe_counts",
    "num_output_states": 8,
    "mask_token_id": 8,
    "mask_token_is_output": False,
    "fixed_boundaries": True,
    "fit_required": False,
    "count_mapping": {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4-7": 4,
        "8-15": 5,
        "16-31": 6,
        "32+": 7,
    },
    "continuous_channel": {
        "source": "raw_biological_probe_counts",
        "transform": "per_gene_all_fit_standardized_log1p",
        "preserves_exact_within_bin_value": True,
        "masked_value": 0.0,
    },
}

_PROHIBITED_IDENTIFIER_KEYS = frozenset(
    {
        "cell_id",
        "core_id",
        "core_label",
        "donor_id",
        "fov",
        "patient_id",
        "slide",
    }
)


class HybridCountMaterializationError(ValueError):
    """Raised when a frozen input or output contract is violated."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HybridCountMaterializationError(f"{label} must be a mapping")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _yaml_bytes(value: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(value),
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")


def _project_reference(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise HybridCountMaterializationError(
            "Campaign path escapes the project root"
        ) from exc


def _resolve_project_reference(
    project_root: Path,
    value: Any,
    *,
    label: str,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HybridCountMaterializationError(
            f"{label} must be a non-empty project-relative path"
        )
    reference = Path(value)
    if reference.is_absolute():
        raise HybridCountMaterializationError(
            f"{label} must remain project-relative"
        )
    unresolved = project_root / reference
    if unresolved.is_symlink():
        raise HybridCountMaterializationError(f"{label} cannot be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise HybridCountMaterializationError(
            f"{label} escapes the project root"
        ) from exc
    if require_file and not resolved.is_file():
        raise HybridCountMaterializationError(f"{label} is not a file")
    if require_directory and not resolved.is_dir():
        raise HybridCountMaterializationError(f"{label} is not a directory")
    return resolved


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HybridCountMaterializationError(
            f"{label} is not readable JSON"
        ) from exc
    return dict(_mapping(payload, label))


def _verify_canonical_checksum(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> str:
    checksum = payload.get("checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise HybridCountMaterializationError(
            f"{label} lacks a SHA-256 checksum"
        )
    core = dict(payload)
    core.pop("checksum", None)
    observed = canonical_sha256(core)
    if observed != checksum:
        raise HybridCountMaterializationError(
            f"{label} canonical checksum does not verify"
        )
    return checksum


def _component_section(
    project_root: Path,
    reference: str,
    section: str,
) -> tuple[dict[str, Any], str]:
    path = _resolve_project_reference(
        project_root,
        reference,
        label=f"{section} component",
        require_file=True,
    )
    payload = load_yaml_mapping(path)
    value = payload.get(section)
    body = dict(_mapping(value, f"{section} component body"))
    if set(payload) != {section}:
        raise HybridCountMaterializationError(
            f"{reference} must contain only the {section!r} group"
        )
    return body, sha256_file(path)


def _validate_frozen_contract(contract_path: Path) -> tuple[dict[str, Any], str]:
    observed = sha256_file(contract_path)
    if observed != EXPECTED_FROZEN_CONTRACT_SHA256:
        raise HybridCountMaterializationError(
            "Frozen hybrid task contract differs from its locked SHA-256"
        )
    checksum_path = contract_path.with_suffix(".sha256")
    if not checksum_path.is_file():
        raise HybridCountMaterializationError(
            "Frozen task contract checksum sidecar is missing"
        )
    fields = checksum_path.read_text(encoding="utf-8").split()
    if not fields or fields[0] != observed:
        raise HybridCountMaterializationError(
            "Frozen task contract checksum sidecar does not match"
        )
    contract = load_yaml_mapping(contract_path)
    if contract.get("campaign_id") != CAMPAIGN_ID:
        raise HybridCountMaterializationError(
            "Frozen task contract campaign ID is wrong"
        )
    cohort = _mapping(contract.get("cohort"), "contract.cohort")
    if tuple(cohort.get("aliases", ())) != ALIASES:
        raise HybridCountMaterializationError(
            "Frozen task contract does not contain the exact ten aliases"
        )
    training = _mapping(contract.get("training"), "contract.training")
    expected_training = {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "max_epochs": 200,
        "seed": 0,
        "edge_dropout": 0.0,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise HybridCountMaterializationError(
                f"Frozen training field {key!r} drifted"
            )
    models = _mapping(contract.get("models"), "contract.models")
    graph_model = _mapping(
        models.get("hybrid-gat-k1000"),
        "contract.models.hybrid-gat-k1000",
    )
    expected_dimensions = {
        "hidden_width": 512,
        "attention_heads": 4,
        "graph_layers": 2,
        "ffn_width": 512,
        "decoder_width": 512,
        "edge_feature_count": 17,
    }
    for key, expected in expected_dimensions.items():
        if graph_model.get(key) != expected:
            raise HybridCountMaterializationError(
                f"Frozen model field {key!r} drifted"
            )
    pilot = _mapping(contract.get("pilot_gate"), "contract.pilot_gate")
    if pilot.get("alias") != "ANC-01" or pilot.get("epochs") != 2:
        raise HybridCountMaterializationError(
            "Frozen pilot identity or epoch budget drifted"
        )
    return contract, observed


def _validate_source_materialization(
    source_path: Path,
) -> tuple[dict[str, Any], str]:
    payload = _read_json(source_path, label="source materialization")
    checksum = _verify_canonical_checksum(
        payload, label="source materialization"
    )
    if payload.get("campaign_id") != SOURCE_CAMPAIGN_ID:
        raise HybridCountMaterializationError(
            "Source materialization belongs to the wrong campaign"
        )
    cohort = _mapping(payload.get("cohort"), "source cohort")
    if (
        cohort.get("core_count") != 10
        or cohort.get("true_normal_core_count") != 0
        or cohort.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or cohort.get("protected_selection_manifest_sha256")
        != EXPECTED_SELECTION_SHA256
    ):
        raise HybridCountMaterializationError(
            "Source materialization cohort contract is not the locked "
            "ten-core adjacent-normal cohort"
        )
    return payload, checksum


def _source_job_map(
    source: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    jobs = source.get("jobs")
    if not isinstance(jobs, list):
        raise HybridCountMaterializationError(
            "Source materialization jobs must be a list"
        )
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_job in jobs:
        job = _mapping(raw_job, "source job")
        key = (str(job.get("alias")), str(job.get("arm")))
        if key in result:
            raise HybridCountMaterializationError(
                "Source materialization contains a duplicate job"
            )
        result[key] = job
    return result


def _assert_no_direct_identifier_fields(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            canonical = str(key).strip().lower().replace("-", "_")
            if canonical in _PROHIBITED_IDENTIFIER_KEYS:
                raise HybridCountMaterializationError(
                    f"{label} contains prohibited direct-identifier field {key!r}"
                )
            _assert_no_direct_identifier_fields(child, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_no_direct_identifier_fields(child, label=label)


def _validate_source_core(
    *,
    project_root: Path,
    source_core: Mapping[str, Any],
    source_jobs: Mapping[tuple[str, str], Mapping[str, Any]],
    alias: str,
) -> dict[str, Any]:
    if source_core.get("alias") != alias:
        raise HybridCountMaterializationError(
            f"Source core order or alias is invalid for {alias}"
        )
    if source_core.get("n_genes") != 1000:
        raise HybridCountMaterializationError(
            f"{alias} source artifact does not have 1,000 genes"
        )
    n_nodes = source_core.get("n_nodes")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes <= 1000:
        raise HybridCountMaterializationError(
            f"{alias} does not contain enough nodes for exact k=1,000"
        )
    configs = _mapping(source_core.get("configs"), f"{alias} source configs")
    config_reference = configs.get("k1000")
    source_job = source_jobs.get((alias, "k1000"))
    if source_job is None or source_job.get("config") != config_reference:
        raise HybridCountMaterializationError(
            f"{alias} k=1,000 config is not bound to its source job"
        )
    config_path = _resolve_project_reference(
        project_root,
        config_reference,
        label=f"{alias} source k=1,000 config",
        require_file=True,
    )
    source_config = load_yaml_mapping(config_path)
    expected_config_sha = source_job.get("config_sha256")
    if (
        not isinstance(expected_config_sha, str)
        or canonical_sha256(source_config) != expected_config_sha
    ):
        raise HybridCountMaterializationError(
            f"{alias} source config checksum does not verify"
        )
    if _mapping(source_config.get("campaign"), "source campaign").get(
        "campaign_id"
    ) != SOURCE_CAMPAIGN_ID:
        raise HybridCountMaterializationError(
            f"{alias} source config belongs to the wrong campaign"
        )
    experiment = _mapping(source_config.get("experiment"), "source experiment")
    dataset = _mapping(source_config.get("dataset"), "source dataset")
    if (
        experiment.get("biological_unit_alias") != alias
        or dataset.get("biological_unit_alias") != alias
        or dataset.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
    ):
        raise HybridCountMaterializationError(
            f"{alias} source config alias or tissue context is inconsistent"
        )
    artifact_reference = source_core.get("prepared_artifact")
    if dataset.get("prepared_artifact_reference") != artifact_reference:
        raise HybridCountMaterializationError(
            f"{alias} prepared artifact routing is inconsistent"
        )
    artifact_path = _resolve_project_reference(
        project_root,
        artifact_reference,
        label=f"{alias} prepared artifact",
        require_directory=True,
    )
    manifest, _, _ = load_prepared_artifact(artifact_path, load_arrays=False)
    selection = _mapping(manifest.get("selection"), f"{alias} selection record")
    if (
        selection.get("opaque_alias") != alias
        or selection.get("restricted_identifiers_emitted") is not False
    ):
        raise HybridCountMaterializationError(
            f"{alias} prepared artifact is not alias-safe"
        )
    core = load_and_refit_full_core(artifact_path)
    if core.n_nodes != n_nodes or core.n_genes != 1000:
        raise HybridCountMaterializationError(
            f"{alias} materialized array shape differs from its receipt"
        )
    if (
        core.checksums.preprocessing_sha256
        != source_core.get("preprocessing_sha256")
        or dataset.get("dataset_fingerprint")
        != core.checksums.preprocessing_sha256
    ):
        raise HybridCountMaterializationError(
            f"{alias} full-core preprocessing checksum does not verify"
        )
    if (
        dataset.get("split_id") != source_core.get("split_id")
        or dataset.get("split_fingerprint")
        != source_core.get("split_fingerprint")
    ):
        raise HybridCountMaterializationError(
            f"{alias} split identity differs from its receipt"
        )
    features = _mapping(source_config.get("features"), "source features")
    node_metadata = _mapping(
        features.get("node_metadata"), "source node metadata"
    )
    if tuple(node_metadata.get("fields", ())) != ALLOWED_METADATA_COLUMNS:
        raise HybridCountMaterializationError(
            f"{alias} does not have the exact 22 permitted covariates"
        )
    edge_features = _mapping(
        features.get("edge_features"), "source edge features"
    )
    if tuple(edge_features.get("fields", ())) != EDGE_ATTRIBUTE_NAMES:
        raise HybridCountMaterializationError(
            f"{alias} does not have the exact 17 edge features"
        )
    graphs = _mapping(source_core.get("graphs"), f"{alias} source graphs")
    graph_record = _mapping(graphs.get("1000"), f"{alias} k=1,000 graph")
    graph = dict(_mapping(source_config.get("graph"), "source graph config"))
    if (
        graph.get("k") != 1000
        or graph.get("neighbor_k") != 1000
        or graph.get("symmetry") != "mutual"
        or graph.get("edge_dropout") != 0.0
        or graph.get("self_loops") is not False
        or graph.get("expected_materialized_graph_sha256")
        != graph_record.get("graph_sha256")
        or graph.get("expected_directed_edges")
        != graph_record.get("n_directed_edges")
    ):
        raise HybridCountMaterializationError(
            f"{alias} exact mutual k=1,000 graph identity is inconsistent"
        )
    if graph_record.get("n_components") != 1 or graph_record.get(
        "n_isolated_nodes"
    ) != 0:
        raise HybridCountMaterializationError(
            f"{alias} source graph is disconnected or has isolated nodes"
        )
    source_basis = _mapping(
        dataset.get("dataset_fingerprint_basis"),
        "source dataset fingerprint basis",
    )
    if source_basis.get("source_prepared_data_sha256") != (
        core.checksums.source_prepared_data_sha256
    ):
        raise HybridCountMaterializationError(
            f"{alias} prepared-data checksum differs from its dataset basis"
        )
    identity = {
        "alias": alias,
        "n_nodes": core.n_nodes,
        "n_genes": core.n_genes,
        "prepared_artifact": _project_reference(project_root, artifact_path),
        "source_artifact_id": core.checksums.source_artifact_id,
        "source_prepared_data_sha256": (
            core.checksums.source_prepared_data_sha256
        ),
        "preprocessing_sha256": core.checksums.preprocessing_sha256,
        "split_id": str(dataset["split_id"]),
        "split_fingerprint": str(dataset["split_fingerprint"]),
        "split_fingerprint_basis": deepcopy(
            dict(_mapping(dataset.get("split_fingerprint_basis"), "split basis"))
        ),
        "dataset_id": str(dataset["dataset_id"]),
        "dataset_version": str(dataset["version"]),
        "preprocessing_version": str(dataset["preprocessing_version"]),
        "experimental_unit": str(dataset["experimental_unit"]),
        "source_recipe_schema": str(
            source_basis.get("recipe_schema", "full_core_fit_v1")
        ),
        "source_metadata_transform": str(
            source_basis.get(
                "metadata_transform",
                "inverse_prepared_transform_then_median_imputation_"
                "log1p_standardization",
            )
        ),
        "graph": graph,
        "graph_sha256": str(graph_record["graph_sha256"]),
        "n_directed_edges": int(graph_record["n_directed_edges"]),
        "node_metadata": deepcopy(dict(node_metadata)),
        "edge_features": deepcopy(dict(edge_features)),
    }
    del core
    gc.collect()
    return identity


def _validate_model_components(
    model_sections: Mapping[str, Mapping[str, Any]],
) -> int:
    graph = model_sections["hybrid-gat-k1000"]
    self_only = model_sections["hybrid-matched-self"]
    expected_graph = {
        "name": "hybrid-count-gat",
        "family": "hybrid_count_edge_conditioned_gatv2",
        "count_representation_schema": _COUNT_REPRESENTATION["schema"],
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
        "uses_graph_inputs": True,
        "uses_edge_inputs": True,
    }
    for key, expected in expected_graph.items():
        if graph.get(key) != expected:
            raise HybridCountMaterializationError(
                f"Hybrid GAT model component field {key!r} drifted"
            )
    expected_self = {
        **expected_graph,
        "name": "hybrid-count-matched-self",
        "family": "hybrid_count_parameter_matched_self_control",
        "uses_graph_inputs": False,
        "uses_edge_inputs": False,
    }
    for key, expected in expected_self.items():
        if self_only.get(key) != expected:
            raise HybridCountMaterializationError(
                f"Hybrid self model component field {key!r} drifted"
            )
    paired_fields = (
        "embedding_dim",
        "hidden_dim",
        "graph_layers",
        "attention_heads",
        "ffn_dim",
        "decoder_dim",
        "edge_hidden_dim",
        "edge_embedding_dim",
        "dropout",
        "attention_dropout",
    )
    if any(graph.get(key) != self_only.get(key) for key in paired_fields):
        raise HybridCountMaterializationError(
            "Hybrid model components do not share all paired dimensions"
        )
    common = {
        "num_genes": 1000,
        "edge_attribute_dim": len(EDGE_ATTRIBUTE_NAMES),
        "expression_mean": np.zeros(1000, dtype=np.float32),
        "expression_scale": np.ones(1000, dtype=np.float32),
        "node_covariate_dim": len(ALLOWED_METADATA_COLUMNS),
        "hidden_dim": int(graph["hidden_dim"]),
        "attention_heads": int(graph["attention_heads"]),
        "attention_head_dim": graph.get("attention_head_dim"),
        "graph_layers": int(graph["graph_layers"]),
        "ffn_dim": int(graph["ffn_dim"]),
        "decoder_dim": int(graph["decoder_dim"]),
        "edge_hidden_dim": int(graph["edge_hidden_dim"]),
        "edge_embedding_dim": int(graph["edge_embedding_dim"]),
        "dropout": float(graph["dropout"]),
        "attention_dropout": float(graph["attention_dropout"]),
    }
    graph_model = HybridReceiverChunkedEdgeConditionedGATv2(
        **common,
        receiver_chunk_size=int(graph["receiver_chunk_size"]),
        activation_checkpointing=bool(graph["activation_checkpointing"]),
    )
    self_model = HybridEdgeParameterMatchedSelfControl(**common)
    count = assert_exact_parameter_match(graph_model, self_model)
    del graph_model, self_model
    gc.collect()
    return count


def _validate_trainer_component(
    trainer: Mapping[str, Any], *, pilot: bool
) -> None:
    expected = {
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "huber_delta": 1.0,
        "max_epochs": 2 if pilot else 200,
        "fixed_epoch_budget": True,
        "early_stopping": False,
        "neighbor_sampling": False,
        "amp": True,
        "deterministic": True,
        "restore_best": False,
        "primary_checkpoint_role": "last",
        "checkpoint_policy": "last_only",
        "objective": (
            "equal_weight_balanced_detection_ordinal_positive_huber"
        ),
    }
    for key, value in expected.items():
        if trainer.get(key) != value:
            raise HybridCountMaterializationError(
                f"Hybrid trainer component field {key!r} drifted"
            )
    if pilot and trainer.get("run_fp32_amp_equivalence") is not True:
        raise HybridCountMaterializationError(
            "Hybrid resource pilot must run FP32/AMP equivalence"
        )


def _validate_evaluation_component(
    evaluation: Mapping[str, Any], *, pilot: bool
) -> None:
    expected = {
        "task_family": "masked_expression_hybrid_count",
        "protocol": "held_in_full_core_fixed_budget",
        "canonical_prediction_split": "fit",
        "primary_metric": "fit/whole_node/hybrid_loss",
        "primary_direction": "minimize",
        "splits": ["fit"],
        "mask_modes": ["partial_gene", "whole_node", "spatial_block"],
        "mask_replicates_per_mode": 1 if pilot else 3,
        "fixed_mask_bundle": True,
        "generalization_estimate": False,
        "validation_or_test_selection": False,
        "conclusion_bearing": not pilot,
        "diagnostic_only": pilot,
    }
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise HybridCountMaterializationError(
                f"Hybrid evaluation component field {key!r} drifted"
            )


def _dataset_section(identity: Mapping[str, Any]) -> dict[str, Any]:
    n_nodes = int(identity["n_nodes"])
    return {
        "dataset_id": identity["dataset_id"],
        "version": identity["dataset_version"],
        "split_id": identity["split_id"],
        "dataset_fingerprint": identity["preprocessing_sha256"],
        "dataset_fingerprint_role": (
            "materialized_full_core_preprocessing_checksum"
        ),
        "dataset_fingerprint_basis": {
            "recipe_schema": identity["source_recipe_schema"],
            "source_artifact_id": identity["source_artifact_id"],
            "source_prepared_data_sha256": (
                identity["source_prepared_data_sha256"]
            ),
            "biological_targets": 1000,
            "technical_control_prefixes_excluded": [
                "Negative",
                "SystemControl",
            ],
            "expression_transform": (
                "raw_counts_plus_gene_wise_log1p_standardization"
            ),
            "metadata_transform": identity["source_metadata_transform"],
            "numerical_epsilon": 1e-8,
            "fit_scope": f"all_{n_nodes}_nodes_transductive",
            "materialized_preprocessing_sha256": (
                identity["preprocessing_sha256"]
            ),
        },
        "split_fingerprint": identity["split_fingerprint"],
        "split_fingerprint_status": (
            "verified_materialized_no_holdout_role_assignment"
        ),
        "split_fingerprint_basis": deepcopy(
            identity["split_fingerprint_basis"]
        ),
        "preprocessing_version": identity["preprocessing_version"],
        "prepared_artifact_reference": identity["prepared_artifact"],
        "task": "masked_expression_hybrid_count",
        "target_scale": (
            "raw_biological_probe_counts_with_per_gene_all_fit_"
            "standardized_log1p"
        ),
        "count_representation": deepcopy(_COUNT_REPRESENTATION),
        "biological_target_count": 1000,
        "technical_control_prefixes_excluded": [
            "Negative",
            "SystemControl",
        ],
        "preprocessing_fit_scope": "all_nodes_transductive",
        "experimental_unit": identity["experimental_unit"],
        "biological_unit_alias": identity["alias"],
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "generalization_scope": "held_in_reconstruction_only",
        "validation_or_test_partition_present": False,
        "patient_generalization_supported": False,
    }


def _features_section(
    identity: Mapping[str, Any], *, uses_graph: bool
) -> dict[str, Any]:
    return {
        "use_edge_features": uses_graph,
        "fit_scope": "all_nodes_transductive",
        "node_expression": {
            "biological_targets": 1000,
            "source_scale": "raw_biological_probe_counts",
            "discrete_transform": "fixed_hybrid_count_states",
            "continuous_transform": (
                "per_gene_all_fit_standardized_log1p"
            ),
            "masked_discrete_value": "input_only_mask_token_8",
            "masked_continuous_value": 0.0,
            "explicit_mask_authoritative_inside_model": True,
        },
        "node_metadata": deepcopy(identity["node_metadata"]),
        "edge_features": (
            deepcopy(identity["edge_features"]) if uses_graph else []
        ),
        "prohibited_node_inputs": [
            "direct_identifiers",
            "absolute_or_local_coordinates",
            "expression_derived_library_size",
            "rna_derived_qc",
            "vendor_cell_type_cluster_neighborhood_or_niche",
            "hidden_target_values",
        ],
    }


def _experiment_section(
    *, alias: str, arm: str, pilot: bool
) -> dict[str, Any]:
    alias_key = alias.lower().replace("-", "")
    arm_key = arm.replace("-", "_")
    if pilot:
        return {
            "variant_label": f"{alias_key}_{arm_key}_resource_pilot",
            "arm": arm,
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "estimand": "implementation_resource_feasibility",
            "permitted_claim": "diagnostic_runtime_memory_and_precision_only",
            "paired_within_core": True,
            "conclusion_eligible": False,
            "excluded_from_primary_comparison": True,
            "resource_pilot": True,
        }
    return {
        "variant_label": f"{alias_key}_{arm_key}_full_core",
        "arm": arm,
        "biological_unit_alias": alias,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "estimand": "held_in_full_core_whole_node_masked_expression",
        "permitted_claim": (
            "held_in_representation_capacity_and_possible_broad_context_"
            "graph_gain"
        ),
        "paired_within_core": True,
        "conclusion_eligible": True,
        "excluded_from_primary_comparison": False,
        "resource_pilot": False,
    }


def _classification_section(*, pilot: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "lifecycle_stage": "diagnostic" if pilot else "exploratory_screen",
        "study_axis": (
            "hybrid_count_resource_feasibility"
            if pilot
            else "adjacent_normal_10core_hybrid_count_graph_gain"
        ),
        "retention_class": (
            "retain_diagnostic_evidence"
            if pilot
            else "retain_exploratory_evidence"
        ),
        "classification_confidence": "high",
    }


def _build_config(
    *,
    identity: Mapping[str, Any],
    arm: str,
    pilot: bool,
    requested_gpu: int,
    output_root_reference: str,
    contract_reference: str,
    contract_sha256: str,
    model_sections: Mapping[str, Mapping[str, Any]],
    trainer_sections: Mapping[str, Mapping[str, Any]],
    evaluation_sections: Mapping[str, Mapping[str, Any]],
    launcher_section: Mapping[str, Any],
) -> dict[str, Any]:
    uses_graph = arm == "hybrid-gat-k1000"
    trainer = deepcopy(
        dict(trainer_sections["pilot" if pilot else "production"])
    )
    gate_reference = f"{output_root_reference}/{PILOT_GATE_RECEIPT_NAME}"
    trainer["amp_authorization"] = {
        "mode": (
            "same_batch_fp32_amp_equivalence_diagnostic"
            if pilot
            else "require_external_pilot_gate_receipt"
        ),
        "receipt_schema": PILOT_GATE_RECEIPT_KIND,
        "receipt_reference": gate_reference,
        "frozen_contract_sha256": contract_sha256,
    }
    masking = deepcopy(_FROZEN_MASKING)
    masking["fit_replicates"] = 1 if pilot else 3
    launcher = deepcopy(dict(launcher_section))
    launcher["requested_gpu"] = str(requested_gpu)
    config: dict[str, Any] = {
        "model": deepcopy(dict(model_sections[arm])),
        "masking": masking,
        "dataset": _dataset_section(identity),
        "features": _features_section(identity, uses_graph=uses_graph),
        "graph": deepcopy(dict(identity["graph"])),
        "trainer": trainer,
        "evaluation": deepcopy(
            dict(evaluation_sections["pilot" if pilot else "production"])
        ),
        "launcher": launcher,
        "version": 1,
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "display_name": "Ten-core adjacent-normal hybrid raw-count GAT",
            "exploratory": True,
            "frozen_contract": contract_reference,
            "frozen_contract_sha256": contract_sha256,
        },
        "experiment": _experiment_section(
            alias=str(identity["alias"]), arm=arm, pilot=pilot
        ),
        "classification": _classification_section(pilot=pilot),
        "metadata": {
            "locked_config_materialization_receipt": (
                f"{output_root_reference}/{RECEIPT_NAME}"
            ),
            "frozen_scientific_contract": True,
            "execution_role": "resource_pilot" if pilot else "production",
            "production_requires_pilot_gate": not pilot,
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }
    validate_experiment_config(config)
    _assert_no_direct_identifier_fields(config, label="resolved hybrid config")
    if requested_gpu not in SAFE_GPU_IDS:
        raise HybridCountMaterializationError(
            "Resolved config requests a prohibited GPU"
        )
    return config


def _estimated_work(identity: Mapping[str, Any], arm: str) -> float:
    shared_node_gene_work = float(identity["n_nodes"] * identity["n_genes"])
    if arm == "hybrid-gat-k1000":
        return shared_node_gene_work + float(identity["n_directed_edges"])
    return shared_node_gene_work


def _lpt_assign(
    jobs: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], int], dict[str, float], dict[str, int]]:
    loads = {gpu: 0.0 for gpu in SAFE_GPU_IDS}
    counts = {gpu: 0 for gpu in SAFE_GPU_IDS}
    assignments: dict[tuple[str, str], int] = {}
    ordered = sorted(
        jobs,
        key=lambda job: (
            -float(job["estimated_work"]),
            str(job["alias"]),
            str(job["arm"]),
        ),
    )
    for job in ordered:
        gpu = min(SAFE_GPU_IDS, key=lambda index: (loads[index], index))
        key = (str(job["alias"]), str(job["arm"]))
        assignments[key] = gpu
        loads[gpu] += float(job["estimated_work"])
        counts[gpu] += 1
    return (
        assignments,
        {str(key): value for key, value in loads.items()},
        {str(key): value for key, value in counts.items()},
    )


def _config_filename(alias: str, arm: str, *, pilot: bool) -> str:
    suffix = arm.replace("-", "_")
    pilot_suffix = "_resource_pilot" if pilot else ""
    return f"{alias.lower()}_{suffix}{pilot_suffix}.yaml"


def _existing_output_matches(
    output_dir: Path,
    expected_files: Mapping[str, bytes],
) -> bool:
    if not output_dir.is_dir():
        return False
    for relative, content in expected_files.items():
        path = output_dir / relative
        if not path.is_file() or path.read_bytes() != content:
            return False
    for directory in ("resource_pilot_configs", "production_configs"):
        observed = {
            path.relative_to(output_dir).as_posix()
            for path in (output_dir / directory).glob("*.yaml")
        }
        expected = {
            relative
            for relative in expected_files
            if relative.startswith(f"{directory}/")
        }
        if observed != expected:
            return False
    return True


def _publish_atomically(
    output_dir: Path,
    files: Mapping[str, bytes],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.stage-",
            dir=output_dir.parent,
        )
    )
    try:
        for relative, content in files.items():
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        for relative, content in files.items():
            if (stage / relative).read_bytes() != content:
                raise HybridCountMaterializationError(
                    "Staged config materialization failed byte verification"
                )
        if output_dir.exists():
            if _existing_output_matches(output_dir, files):
                return
            raise HybridCountMaterializationError(
                "Locked output directory already exists with different content"
            )
        os.replace(stage, output_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def materialize_campaign(
    *,
    project_root: Path,
    source_materialization: Path,
    contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and atomically publish the complete locked config set."""

    project_root = project_root.resolve()
    source_materialization = source_materialization.resolve()
    contract_path = contract_path.resolve()
    output_dir = output_dir.resolve()
    for path, label in (
        (source_materialization, "source materialization"),
        (contract_path, "frozen task contract"),
    ):
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise HybridCountMaterializationError(
                f"{label} must remain under the project root"
            ) from exc
    try:
        output_dir.relative_to(project_root)
    except ValueError as exc:
        raise HybridCountMaterializationError(
            "Output directory must remain under the project root"
        ) from exc
    if output_dir == project_root or output_dir.parent == project_root.parent:
        raise HybridCountMaterializationError(
            "Output directory is too broad for atomic publication"
        )

    contract, contract_sha = _validate_frozen_contract(contract_path)
    source, source_checksum = _validate_source_materialization(
        source_materialization
    )
    source_cores_raw = source.get("materialized_cores")
    if not isinstance(source_cores_raw, list) or len(source_cores_raw) != 10:
        raise HybridCountMaterializationError(
            "Source materialization must contain exactly ten cores"
        )
    if tuple(str(item.get("alias")) for item in source_cores_raw) != ALIASES:
        raise HybridCountMaterializationError(
            "Source materialization aliases are not exact or ordered"
        )
    source_jobs = _source_job_map(source)

    identities = [
        _validate_source_core(
            project_root=project_root,
            source_core=_mapping(raw, f"source core {alias}"),
            source_jobs=source_jobs,
            alias=alias,
        )
        for alias, raw in zip(ALIASES, source_cores_raw, strict=True)
    ]

    model_sections: dict[str, Mapping[str, Any]] = {}
    component_hashes: dict[str, str] = {}
    for arm in ARMS:
        section, checksum = _component_section(
            project_root, _MODEL_COMPONENTS[arm], "model"
        )
        model_sections[arm] = section
        component_hashes[_MODEL_COMPONENTS[arm]] = checksum
    trainer_sections: dict[str, Mapping[str, Any]] = {}
    evaluation_sections: dict[str, Mapping[str, Any]] = {}
    for role in ("pilot", "production"):
        trainer, trainer_sha = _component_section(
            project_root, _TRAINER_COMPONENTS[role], "trainer"
        )
        evaluation, evaluation_sha = _component_section(
            project_root, _EVALUATION_COMPONENTS[role], "evaluation"
        )
        _validate_trainer_component(trainer, pilot=role == "pilot")
        _validate_evaluation_component(evaluation, pilot=role == "pilot")
        trainer_sections[role] = trainer
        evaluation_sections[role] = evaluation
        component_hashes[_TRAINER_COMPONENTS[role]] = trainer_sha
        component_hashes[_EVALUATION_COMPONENTS[role]] = evaluation_sha
    launcher_section, launcher_sha = _component_section(
        project_root, _LAUNCHER_COMPONENT, "launcher"
    )
    component_hashes[_LAUNCHER_COMPONENT] = launcher_sha
    parameter_count = _validate_model_components(model_sections)

    identities_by_alias = {
        str(identity["alias"]): identity for identity in identities
    }
    production_plan = [
        {
            "alias": alias,
            "arm": arm,
            "estimated_work": _estimated_work(identities_by_alias[alias], arm),
        }
        for alias in ALIASES
        for arm in ARMS
    ]
    pilot_plan = [
        {
            "alias": "ANC-01",
            "arm": arm,
            "estimated_work": _estimated_work(identities_by_alias["ANC-01"], arm),
        }
        for arm in ARMS
    ]
    production_assignments, production_loads, production_counts = _lpt_assign(
        production_plan
    )
    pilot_assignments, pilot_loads, pilot_counts = _lpt_assign(pilot_plan)

    output_reference = _project_reference(project_root, output_dir)
    contract_reference = _project_reference(project_root, contract_path)
    generated: dict[str, bytes] = {}
    pilot_jobs: list[dict[str, Any]] = []
    production_jobs: list[dict[str, Any]] = []
    for pilot, plan, assignments, directory, records in (
        (
            True,
            pilot_plan,
            pilot_assignments,
            "resource_pilot_configs",
            pilot_jobs,
        ),
        (
            False,
            production_plan,
            production_assignments,
            "production_configs",
            production_jobs,
        ),
    ):
        for job in sorted(
            plan, key=lambda value: (str(value["alias"]), str(value["arm"]))
        ):
            alias = str(job["alias"])
            arm = str(job["arm"])
            requested_gpu = assignments[(alias, arm)]
            config = _build_config(
                identity=identities_by_alias[alias],
                arm=arm,
                pilot=pilot,
                requested_gpu=requested_gpu,
                output_root_reference=output_reference,
                contract_reference=contract_reference,
                contract_sha256=contract_sha,
                model_sections=model_sections,
                trainer_sections=trainer_sections,
                evaluation_sections=evaluation_sections,
                launcher_section=launcher_section,
            )
            relative = (
                f"{directory}/"
                f"{_config_filename(alias, arm, pilot=pilot)}"
            )
            content = _yaml_bytes(config)
            generated[relative] = content
            records.append(
                {
                    "alias": alias,
                    "arm": arm,
                    "config": f"{output_reference}/{relative}",
                    "config_sha256": canonical_sha256(config),
                    "file_sha256": _sha256_bytes(content),
                    "requested_gpu": requested_gpu,
                    "estimated_work": float(job["estimated_work"]),
                }
            )

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": RECEIPT_KIND,
        "campaign_id": CAMPAIGN_ID,
        "frozen_contract": {
            "reference": contract_reference,
            "sha256": contract_sha,
        },
        "source": {
            "campaign_id": SOURCE_CAMPAIGN_ID,
            "materialization_reference": _project_reference(
                project_root, source_materialization
            ),
            "file_sha256": sha256_file(source_materialization),
            "canonical_checksum": source_checksum,
            "selection_manifest_sha256": EXPECTED_SELECTION_SHA256,
        },
        "component_file_sha256": dict(sorted(component_hashes.items())),
        "feature_contract": {
            "node_metadata_fields": list(ALLOWED_METADATA_COLUMNS),
            "edge_attribute_fields": list(EDGE_ATTRIBUTE_NAMES),
            "sha256": canonical_sha256(
                {
                    "node_metadata_fields": list(ALLOWED_METADATA_COLUMNS),
                    "edge_attribute_fields": list(EDGE_ATTRIBUTE_NAMES),
                }
            ),
        },
        "parameter_count": parameter_count,
        "allowed_gpu_ids": list(SAFE_GPU_IDS),
        "assignment": {
            "algorithm": "longest_processing_time_first_v1",
            "workload_formula": {
                "hybrid-gat-k1000": "n_nodes*n_genes+n_directed_edges",
                "hybrid-matched-self": "n_nodes*n_genes",
            },
            "pilot_loads": pilot_loads,
            "pilot_job_counts": pilot_counts,
            "production_loads": production_loads,
            "production_job_counts": production_counts,
        },
        "cores": [
            {
                "alias": identity["alias"],
                "n_nodes": identity["n_nodes"],
                "n_genes": identity["n_genes"],
                "prepared_artifact": identity["prepared_artifact"],
                "source_prepared_data_sha256": identity[
                    "source_prepared_data_sha256"
                ],
                "preprocessing_sha256": identity["preprocessing_sha256"],
                "split_id": identity["split_id"],
                "split_fingerprint": identity["split_fingerprint"],
                "k1000_graph_sha256": identity["graph_sha256"],
                "k1000_directed_edges": identity["n_directed_edges"],
            }
            for identity in identities
        ],
        "pilot_jobs": pilot_jobs,
        "production_jobs": production_jobs,
        "pilot_gate_receipt_reference": (
            f"{output_reference}/{PILOT_GATE_RECEIPT_NAME}"
        ),
        "counts": {
            "aliases": len(identities),
            "pilot_configs": len(pilot_jobs),
            "production_configs": len(production_jobs),
        },
        "registry_mutation_performed": False,
        "queue_mutation_performed": False,
        "training_performed": False,
    }
    if receipt["counts"] != {
        "aliases": 10,
        "pilot_configs": 2,
        "production_configs": 20,
    }:
        raise HybridCountMaterializationError(
            "Materialized config count does not match the frozen campaign"
        )
    if any(
        int(job["requested_gpu"]) not in SAFE_GPU_IDS
        for job in (*pilot_jobs, *production_jobs)
    ):
        raise HybridCountMaterializationError(
            "Materialized job requested an unsafe GPU"
        )
    _assert_no_direct_identifier_fields(receipt, label="materialization receipt")
    receipt["checksum"] = canonical_sha256(receipt)
    generated[RECEIPT_NAME] = _json_bytes(receipt)

    # All source, scientific, configuration, alias-safety, parameter-match,
    # assignment, and receipt validations above complete before the first write.
    if output_dir.exists():
        if not _existing_output_matches(output_dir, generated):
            raise HybridCountMaterializationError(
                "Locked output directory already exists with different content"
            )
        return receipt
    _publish_atomically(output_dir, generated)
    if not _existing_output_matches(output_dir, generated):
        raise HybridCountMaterializationError(
            "Published materialization failed post-publication verification"
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-materialization",
        type=Path,
        default=(
            paths.scratch_root
            / "locked_campaigns"
            / SOURCE_CAMPAIGN_ID
            / "campaign_materialization.json"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            paths.project_root
            / "experiments"
            / "campaigns"
            / CAMPAIGN_ID
            / "frozen_task_contract.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = materialize_campaign(
        project_root=current_paths().project_root,
        source_materialization=args.source_materialization,
        contract_path=args.contract,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "campaign_id": receipt["campaign_id"],
                "checksum": receipt["checksum"],
                "counts": receipt["counts"],
                "output": _project_reference(
                    current_paths().project_root, args.output_dir
                ),
                "queue_mutation_performed": False,
                "registry_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
