#!/usr/bin/env python3
"""Materialize the frozen multiscale hurdle-count pilot.

The command is preparation-only.  It validates the frozen contract and the
checksum-bound adjacent-normal source configs, rebuilds the sparse observed
local and regional graphs, constructs the deterministic local sender-state
permutation, and atomically publishes two resource pilot configs plus twenty
unique science configs. It does not register, enqueue, or train runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

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
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.local_source_permutation import (  # noqa: E402
    LocalSourcePermutationError,
    build_macroblock_spatial_antipode_permutation,
    verify_local_source_permutation_receipt,
)
from spatial_benchmark.multiscale_hurdle_contract import (  # noqa: E402
    ACTIVE_CONTRACT_AMENDMENT_RELATIVE,
    ACTIVE_CONTRACT_AMENDMENT_SHA256,
    ARMS,
    CAMPAIGN_ID,
    FROZEN_CONTRACT_SHA256,
    LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM,
    LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM,
    LOCAL_SOURCE_PERMUTATION_SCHEMA,
    REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE,
    REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
    SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE,
    SUPERSEDED_CONTRACT_AMENDMENT_SHA256,
)
from spatial_benchmark.multiscale_graphs import (  # noqa: E402
    LOCAL_K_CAP,
    LOCAL_MAX_DISTANCE_UM,
    REGIONAL_K_CAP,
    REGIONAL_MAX_DISTANCE_UM,
    REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM,
    build_true_multiscale_graphs,
    true_graph_receipt,
)
from spatial_benchmark.paths import current_paths  # noqa: E402


SOURCE_CAMPAIGN_ID = "cmp_20260729_adjacent_normal_10core_hybrid_count_gat"
CONTRACT_AMENDMENT_RELATIVE = ACTIVE_CONTRACT_AMENDMENT_RELATIVE
CONTRACT_AMENDMENT_SHA256 = ACTIVE_CONTRACT_AMENDMENT_SHA256
STAGE1_ALIASES = ("ANC-03", "ANC-05")
STAGE2_ALIASES = ("ANC-02", "ANC-03", "ANC-05", "ANC-06", "ANC-09")
SAFE_GPU_IDS = (0, 1, 2, 3, 5, 6, 7)
RECEIPT_NAME = "locked_config_materialization.json"
RECEIPT_KIND = "multiscale_hurdle_locked_config_materialization_v1"

_MODEL_COMPONENTS = {
    "self": "configs/model/multiscale_hurdle_self.yaml",
    "self-regional": "configs/model/multiscale_hurdle_self_regional.yaml",
    "self-regional-local": (
        "configs/model/multiscale_hurdle_self_regional_local.yaml"
    ),
    "self-regional-local-permuted": (
        "configs/model/multiscale_hurdle_self_regional_local_permuted.yaml"
    ),
}
_GRAPH_COMPONENT = "configs/graph/multiscale_local64_regional256.yaml"
_TRAINER_COMPONENTS = {
    "pilot": "configs/trainer/multiscale_hurdle_resource_pilot_2.yaml",
    "science": "configs/trainer/multiscale_hurdle_fixed_200.yaml",
}
_EVALUATION_COMPONENTS = {
    "pilot": (
        "configs/evaluation/"
        "held_in_multiscale_hurdle_resource_pilot_v1.yaml"
    ),
    "science": "configs/evaluation/held_in_multiscale_hurdle_v1.yaml",
}
_LAUNCHER_COMPONENT = "configs/launcher/local_single_gpu_3090.yaml"

_COUNT_REPRESENTATION: dict[str, Any] = {
    "schema": "hurdle_detection_plus_positive_standardized_log1p_v1",
    "source_scale": "raw_biological_probe_counts",
    "input_states": {
        "schema": "hybrid_raw_count_0_1_2_3_4_7_8_15_16_31_32plus_v1",
        "num_states": 8,
        "mask_token_id": 8,
        "mask_token_is_output": False,
    },
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


class MultiscaleHurdleMaterializationError(ValueError):
    """Raised when a frozen materialization invariant is violated."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiscaleHurdleMaterializationError(f"{label} must be a mapping")
    return value


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
        allow_unicode=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _project_reference(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise MultiscaleHurdleMaterializationError(
            f"Path must remain under project root: {path}"
        ) from exc


def _resolve_reference(
    project_root: Path,
    reference: object,
    *,
    label: str,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise MultiscaleHurdleMaterializationError(
            f"{label} must be a nonempty project-relative path"
        )
    candidate = (project_root / reference).resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise MultiscaleHurdleMaterializationError(
            f"{label} escapes the project root"
        ) from exc
    if require_file and not candidate.is_file():
        raise MultiscaleHurdleMaterializationError(
            f"{label} is not a file: {candidate}"
        )
    if require_directory and not candidate.is_dir():
        raise MultiscaleHurdleMaterializationError(
            f"{label} is not a directory: {candidate}"
        )
    return candidate


def _assert_no_direct_identifiers(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _PROHIBITED_IDENTIFIER_KEYS:
                raise MultiscaleHurdleMaterializationError(
                    f"{label} contains prohibited identifier field {key!r}"
                )
            _assert_no_direct_identifiers(child, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _assert_no_direct_identifiers(child, label=label)


def _component(
    project_root: Path,
    reference: str,
) -> tuple[dict[str, Any], str]:
    path = _resolve_reference(
        project_root,
        reference,
        label=f"component {reference}",
        require_file=True,
    )
    return dict(load_yaml_mapping(path)), sha256_file(path)


def _verify_frozen_contract(
    project_root: Path,
    contract_path: Path,
) -> str:
    expected = (
        project_root
        / "experiments/campaigns"
        / CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    ).resolve()
    if contract_path.resolve() != expected:
        raise MultiscaleHurdleMaterializationError(
            "Frozen contract path is not the authoritative campaign contract"
        )
    observed = sha256_file(contract_path)
    if observed != FROZEN_CONTRACT_SHA256:
        raise MultiscaleHurdleMaterializationError(
            "Frozen task contract checksum drifted"
        )
    contract = load_yaml_mapping(contract_path)
    if (
        contract.get("campaign_id") != CAMPAIGN_ID
        or contract.get("frozen_before_new_training") is not True
        or contract.get("exploratory") is not True
    ):
        raise MultiscaleHurdleMaterializationError(
            "Frozen task contract metadata is invalid"
        )
    return observed


def _verify_contract_amendment(project_root: Path) -> dict[str, Any]:
    path = (project_root / CONTRACT_AMENDMENT_RELATIVE).resolve()
    supplement_path = (
        project_root / REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE
    ).resolve()
    superseded_path = (
        project_root / SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE
    ).resolve()
    if (
        not isinstance(CONTRACT_AMENDMENT_SHA256, str)
        or len(CONTRACT_AMENDMENT_SHA256) != 64
    ):
        raise MultiscaleHurdleMaterializationError(
            "Active contract amendment hash has not been finalized"
        )
    if (
        not path.is_file()
        or sha256_file(path) != CONTRACT_AMENDMENT_SHA256
        or not supplement_path.is_file()
        or sha256_file(supplement_path)
        != REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        or not superseded_path.is_file()
        or sha256_file(superseded_path)
        != SUPERSEDED_CONTRACT_AMENDMENT_SHA256
    ):
        raise MultiscaleHurdleMaterializationError(
            "Contract amendment checksum drifted"
        )
    amendment = dict(load_yaml_mapping(path))
    superseded = _mapping(
        amendment.get("supersedes_execution_design"),
        "superseded execution design",
    )
    replacement = _mapping(
        amendment.get("replacement_null"),
        "replacement null",
    )
    effects = _mapping(
        amendment.get("contract_effects"),
        "contract effects",
    )
    qc = _mapping(
        replacement.get("mandatory_pre_gpu_qc"),
        "sender permutation pre-GPU QC",
    )
    if (
        amendment.get("campaign_id") != CAMPAIGN_ID
        or amendment.get("status") != "frozen_before_any_gpu_training"
        or amendment.get("original_frozen_contract_sha256")
        != FROZEN_CONTRACT_SHA256
        or superseded.get("amendment_reference")
        != SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.as_posix()
        or superseded.get("amendment_sha256")
        != SUPERSEDED_CONTRACT_AMENDMENT_SHA256
        or superseded.get("retained_as_negative_design_record") is not True
        or replacement.get("arm_name")
        != "self_regional_local_permuted"
        or effects.get("original_rewired_arm_authorized") is not False
        or effects.get("old_materialization_authorized") is not False
        or qc.get("minimum_node_mapping_changed_fraction")
        != LOCAL_SOURCE_PERMUTATION_CHANGED_MINIMUM
        or qc.get("displacement_threshold_um")
        != LOCAL_SOURCE_PERMUTATION_DISPLACEMENT_THRESHOLD_UM
        or qc.get(
            "minimum_node_displacement_above_local_radius_fraction"
        )
        != LOCAL_SOURCE_PERMUTATION_DISPLACED_MINIMUM
        or qc.get(
            "minimum_local_edge_slot_sender_identity_changed_fraction"
        )
        != LOCAL_SOURCE_EDGE_SLOT_CHANGED_MINIMUM
    ):
        raise MultiscaleHurdleMaterializationError(
            "Contract amendment semantics are invalid"
        )
    supplement = dict(load_yaml_mapping(supplement_path))
    supplements = _mapping(
        supplement.get("supplements"),
        "contract supplement identity",
    )
    numeric = _mapping(
        supplement.get("deterministic_axis_numeric_definition"),
        "contract supplement deterministic axis definition",
    )
    mask_gate = _mapping(
        supplement.get("mandatory_mask_noninterference_gate"),
        "contract supplement mask noninterference gate",
    )
    tolerance = _mapping(
        mask_gate.get("tolerance"),
        "contract supplement mask tolerance",
    )
    contrasts = _mapping(
        supplement.get("contrast_estimands"),
        "contract supplement contrast estimands",
    )
    if (
        supplement.get("campaign_id") != CAMPAIGN_ID
        or supplement.get("status") != "frozen_before_any_gpu_training"
        or supplement.get("original_frozen_contract_sha256")
        != FROZEN_CONTRACT_SHA256
        or supplements.get("amendment_reference")
        != CONTRACT_AMENDMENT_RELATIVE.as_posix()
        or supplements.get("amendment_sha256")
        != CONTRACT_AMENDMENT_SHA256
        or supplements.get("execution_design_changed") is not False
        or numeric
        != {
            "covariance_precision": "IEEE_754_float64",
            "covariance_divisor": "macroblock_node_count",
            "eigenvalue_tie_discriminant": (
                "hypot(cov_xx_minus_cov_yy, 2_times_cov_xy)"
            ),
            "tie_tolerance": (
                "64_times_float64_epsilon_times_"
                "max_abs_covariance_entry_or_one"
            ),
            "tie_or_degenerate_axis": "positive_x",
            "leading_eigenvector_sign": (
                "largest_absolute_loading_positive_with_x_tie_break"
            ),
            "ordering_tie_break": "stable_node_row_index",
        }
        or mask_gate.get("required_before_gpu_training") is not True
        or mask_gate.get(
            "whole_node_prediction_invariant_to_all_hidden_"
            "receiver_expression_values"
        )
        is not True
        or mask_gate.get(
            "partial_gene_prediction_invariant_to_hidden_receiver_entries"
        )
        is not True
        or mask_gate.get(
            "test_fixture_must_include_effective_permuted_"
            "source_equals_receiver_slot"
        )
        is not True
        or tolerance != {"relative": 0.0, "absolute": 0.0}
        or contrasts.get("combined_maximum_claim")
        != (
            "exploratory held-in graph-specific and correctly aligned local "
            "sender-state predictive dependency"
        )
    ):
        raise MultiscaleHurdleMaterializationError(
            "Required contract supplement semantics are invalid"
        )
    return {
        "reference": CONTRACT_AMENDMENT_RELATIVE.as_posix(),
        "sha256": CONTRACT_AMENDMENT_SHA256,
        "required_supplement": {
            "reference": REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE.as_posix(),
            "sha256": REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
            "mask_noninterference_gate_required_before_gpu_training": True,
        },
        "supersedes": {
            "reference": (
                SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.as_posix()
            ),
            "sha256": SUPERSEDED_CONTRACT_AMENDMENT_SHA256,
            "retained_as_negative_design_record": True,
        },
    }


def _load_source_receipt(
    project_root: Path,
    source_receipt_path: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], Mapping[str, Any]]]:
    try:
        receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultiscaleHurdleMaterializationError(
            "Cannot read source materialization receipt"
        ) from exc
    if not isinstance(receipt, dict):
        raise MultiscaleHurdleMaterializationError(
            "Source materialization receipt must be a mapping"
        )
    checksum = receipt.pop("checksum", None)
    if (
        receipt.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or not isinstance(checksum, str)
        or canonical_sha256(receipt) != checksum
    ):
        raise MultiscaleHurdleMaterializationError(
            "Source materialization receipt identity does not verify"
        )
    receipt["checksum"] = checksum
    jobs: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in receipt.get("production_jobs", []):
        item = _mapping(raw, "source production job")
        key = (str(item.get("alias")), str(item.get("arm")))
        if key in jobs:
            raise MultiscaleHurdleMaterializationError(
                f"Duplicate source production job {key}"
            )
        jobs[key] = item
    for alias in STAGE2_ALIASES:
        if (alias, "hybrid-gat-k1000") not in jobs:
            raise MultiscaleHurdleMaterializationError(
                f"Source receipt lacks {alias} hybrid graph config"
            )
    return receipt, jobs


def _validate_source_identity(
    *,
    project_root: Path,
    alias: str,
    source_job: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    config_path = _resolve_reference(
        project_root,
        source_job.get("config"),
        label=f"{alias} source config",
        require_file=True,
    )
    source_config = dict(load_yaml_mapping(config_path))
    if canonical_sha256(source_config) != source_job.get("config_sha256"):
        raise MultiscaleHurdleMaterializationError(
            f"{alias} source config checksum does not verify"
        )
    campaign = _mapping(source_config.get("campaign"), "source campaign")
    experiment = _mapping(source_config.get("experiment"), "source experiment")
    dataset = _mapping(source_config.get("dataset"), "source dataset")
    if (
        campaign.get("campaign_id") != SOURCE_CAMPAIGN_ID
        or experiment.get("biological_unit_alias") != alias
        or dataset.get("biological_unit_alias") != alias
        or dataset.get("biological_target_count") != 1000
        or dataset.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
    ):
        raise MultiscaleHurdleMaterializationError(
            f"{alias} source config identity is inconsistent"
        )
    artifact_path = _resolve_reference(
        project_root,
        dataset.get("prepared_artifact_reference"),
        label=f"{alias} prepared artifact",
        require_directory=True,
    )
    manifest, _, _ = load_prepared_artifact(artifact_path, load_arrays=False)
    selection = _mapping(manifest.get("selection"), f"{alias} selection")
    if (
        selection.get("opaque_alias") != alias
        or selection.get("restricted_identifiers_emitted") is not False
    ):
        raise MultiscaleHurdleMaterializationError(
            f"{alias} prepared artifact is not alias-safe"
        )
    core = load_and_refit_full_core(artifact_path)
    if (
        core.n_genes != 1000
        or core.checksums.preprocessing_sha256
        != dataset.get("dataset_fingerprint")
    ):
        raise MultiscaleHurdleMaterializationError(
            f"{alias} preprocessing identity does not verify"
        )
    features = _mapping(source_config.get("features"), "source features")
    node_metadata = _mapping(
        features.get("node_metadata"), "source node metadata"
    )
    edge_features = _mapping(
        features.get("edge_features"), "source edge features"
    )
    if tuple(node_metadata.get("fields", ())) != ALLOWED_METADATA_COLUMNS:
        raise MultiscaleHurdleMaterializationError(
            f"{alias} node covariates differ from the permitted contract"
        )
    if tuple(edge_features.get("fields", ())) != EDGE_ATTRIBUTE_NAMES:
        raise MultiscaleHurdleMaterializationError(
            f"{alias} edge attributes differ from the fixed geometry schema"
        )
    return source_config, core


def _materialize_one_core(
    *,
    project_root: Path,
    alias: str,
    source_job: Mapping[str, Any],
    workers: int,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build and verify one independent core in a process-safe call."""

    source_config, core = _validate_source_identity(
        project_root=project_root,
        alias=alias,
        source_job=source_job,
    )
    graphs = build_true_multiscale_graphs(
        core.coordinates_um,
        query_chunk_size=1024,
        receiver_chunk_size=128,
        mutual_search_chunk_size=4_000_000,
        workers=workers,
        epsilon=1e-8,
    )
    record = true_graph_receipt(graphs)
    qc = _mapping(record.get("bundle_qc"), f"{alias} graph bundle qc")
    if (
        qc.get("local_regional_disjoint") is not True
        or qc.get("all_graphs_symmetric") is not True
        or qc.get("all_graphs_receiver_sorted") is not True
        or qc.get("all_graphs_loop_and_duplicate_free") is not True
    ):
        raise MultiscaleHurdleMaterializationError(
            f"{alias} multiscale graph failed a frozen invariant"
        )
    local_edge_index, local_edge_attributes = graphs.local.concatenate()
    try:
        permutation = build_macroblock_spatial_antipode_permutation(
            core.coordinates_um,
            core.macroblock_ids,
            local_edge_index=local_edge_index,
            local_edge_attributes=local_edge_attributes,
        )
        verify_local_source_permutation_receipt(permutation.receipt)
    except LocalSourcePermutationError as exc:
        raise MultiscaleHurdleMaterializationError(
            f"{alias} local source permutation failed pre-GPU QC"
        ) from exc
    record["local_source_permutation"] = deepcopy(
        dict(permutation.receipt)
    )
    dataset = _mapping(source_config.get("dataset"), "source dataset")
    identity = {
        "source_config": source_config,
        "n_nodes": core.n_nodes,
        "n_genes": core.n_genes,
        "prepared_artifact": dataset["prepared_artifact_reference"],
        "preprocessing_sha256": core.checksums.preprocessing_sha256,
        "source_prepared_data_sha256": (
            core.checksums.source_prepared_data_sha256
        ),
        "split_fingerprint": dataset["split_fingerprint"],
    }
    del graphs
    del core
    gc.collect()
    return alias, source_config, identity, record


def _dataset_section(
    source_config: Mapping[str, Any],
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    dataset = deepcopy(dict(_mapping(source_config.get("dataset"), "dataset")))
    dataset["task"] = "masked_expression_hurdle_count"
    dataset["target_scale"] = (
        "raw_biological_probe_counts_with_per_gene_all_fit_"
        "standardized_log1p"
    )
    dataset["count_representation"] = deepcopy(_COUNT_REPRESENTATION)
    dataset["frozen_task_contract_sha256"] = contract_sha256
    return dataset


def _features_section(source_config: Mapping[str, Any]) -> dict[str, Any]:
    source = deepcopy(
        dict(_mapping(source_config.get("features"), "source features"))
    )
    source["use_edge_features"] = True
    node_expression = dict(
        _mapping(source.get("node_expression"), "node expression")
    )
    node_expression.update(
        {
            "biological_targets": 1000,
            "source_scale": "raw_biological_probe_counts",
            "discrete_transform": "fixed_hybrid_count_states",
            "continuous_transform": (
                "per_gene_all_fit_standardized_log1p"
            ),
            "masked_discrete_value": "input_only_mask_token_8",
            "masked_continuous_value": 0.0,
            "explicit_mask_authoritative_inside_model": True,
            "prediction_schema": (
                "detection_plus_positive_standardized_log1p"
            ),
        }
    )
    source["node_expression"] = node_expression
    edge = dict(_mapping(source.get("edge_features"), "edge features"))
    edge["fit_scope"] = "true_local_and_regional_edges_transductive"
    edge["standardization"] = "per_scale_true_graph_edge_wise"
    source["edge_features"] = edge
    return source


def _graph_section(
    component: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    graph = deepcopy(dict(component))
    bundle_checksums = _mapping(
        receipt.get("bundle_checksums"), "bundle graph checksums"
    )
    graph["expected_materialized_graph_sha256"] = bundle_checksums[
        "bundle_sha256"
    ]
    graph["expected_graph_receipt_sha256"] = canonical_sha256(receipt)
    graph["expected_directed_edges"] = int(
        _mapping(
            _mapping(receipt.get("local"), "local graph").get("qc"),
            "local graph qc",
        )["n_directed_edges"]
    )
    for scale in ("local", "regional"):
        scale_record = _mapping(receipt.get(scale), f"{scale} graph")
        scale_qc = _mapping(scale_record.get("qc"), f"{scale} qc")
        scale_checksums = _mapping(
            scale_record.get("checksums"), f"{scale} checksums"
        )
        destination = dict(
            _mapping(graph.get(scale), f"graph.{scale}")
            if scale in graph
            else {}
        )
        destination.update(
            {
                "expected_graph_sha256": scale_checksums["graph_sha256"],
                "expected_directed_edges": int(
                    scale_qc["n_directed_edges"]
                ),
                "expected_components": int(scale_qc["n_components"]),
                "expected_isolated_nodes": int(
                    scale_qc["n_isolated_nodes"]
                ),
            }
        )
        graph[scale] = destination
    permutation = _mapping(
        receipt.get("local_source_permutation"),
        "local source permutation receipt",
    )
    verify_local_source_permutation_receipt(permutation)
    graph["local_source_permutation"] = deepcopy(dict(permutation))
    graph["expected_bundle_qc"] = deepcopy(
        dict(_mapping(receipt.get("bundle_qc"), "bundle graph qc"))
    )
    return graph


def _build_config(
    *,
    source_config: Mapping[str, Any],
    alias: str,
    arm: str,
    pilot: bool,
    requested_gpu: int,
    model_component: Mapping[str, Any],
    graph_component: Mapping[str, Any],
    graph_record: Mapping[str, Any],
    trainer_component: Mapping[str, Any],
    evaluation_component: Mapping[str, Any],
    launcher_component: Mapping[str, Any],
    contract_reference: str,
    contract_sha256: str,
    contract_amendment: Mapping[str, Any],
    output_reference: str,
) -> dict[str, Any]:
    role = "resource_pilot" if pilot else "science"
    model = deepcopy(dict(model_component))
    dataset = _dataset_section(
        source_config,
        contract_sha256=contract_sha256,
    )
    features = _features_section(source_config)
    graph = _graph_section(graph_component, graph_record)
    trainer = deepcopy(dict(trainer_component))
    trainer["amp_authorization"] = (
        {
            "mode": "runner_internal_same_batch_equivalence",
            "maximum_absolute_loss_discrepancy": 0.001,
        }
        if pilot
        else {
            "mode": "require_external_resource_gate_receipt",
            "receipt_schema": "multiscale_hurdle_resource_gate_v1",
            "receipt_reference": (
                f"{output_reference}/resource_gate_receipt.json"
            ),
            "frozen_contract_sha256": contract_sha256,
        }
    )
    masking = deepcopy(_FROZEN_MASKING)
    masking["fit_replicates"] = int(
        evaluation_component["mask_replicates_per_mode"]
    )
    evaluation = deepcopy(dict(evaluation_component))
    launcher = deepcopy(dict(launcher_component))
    launcher.update(
        {
            "requested_gpu": str(requested_gpu),
            "requested_gpu_count": 1,
            "set_cuda_visible_devices": True,
            "concurrency": 1,
            "disk_safety_max_used_decimal_gb": 55.0,
        }
    )
    config: dict[str, Any] = {
        "model": model,
        "masking": masking,
        "dataset": dataset,
        "features": features,
        "graph": graph,
        "trainer": trainer,
        "evaluation": evaluation,
        "launcher": launcher,
        "version": 1,
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "display_name": "Multiscale hurdle-count biological-signal pilot",
            "exploratory": True,
            "frozen_contract": contract_reference,
            "frozen_contract_sha256": contract_sha256,
            "contract_amendment": contract_amendment["reference"],
            "contract_amendment_sha256": contract_amendment["sha256"],
            "contract_supplement": (
                contract_amendment["required_supplement"]["reference"]
            ),
            "contract_supplement_sha256": (
                contract_amendment["required_supplement"]["sha256"]
            ),
            "superseded_contract_amendment": (
                contract_amendment["supersedes"]["reference"]
            ),
            "superseded_contract_amendment_sha256": (
                contract_amendment["supersedes"]["sha256"]
            ),
            "superseded_amendment_retained_as_negative_record": True,
        },
        "experiment": {
            "variant_label": (
                f"{alias.lower().replace('-', '')}_"
                f"{arm.replace('-', '_')}_{role}"
            ),
            "arm": arm,
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "estimand": (
                "held_in_full_core_whole_node_masked_raw_count"
            ),
            "permitted_claim": (
                "exploratory_held_in_graph_specific_and_correctly_aligned_"
                "local_sender_state_predictive_dependency"
            ),
            "paired_within_core": True,
            "conclusion_eligible": not pilot,
            "excluded_from_primary_comparison": pilot,
            "resource_pilot": pilot,
            "stage": 1 if pilot or arm == "self" else 2,
        },
        "classification": {
            "schema_version": 1,
            "lifecycle_stage": (
                "diagnostic" if pilot else "exploratory_screen"
            ),
            "study_axis": "multiscale_hurdle_count_signal_separation",
            "scientific_variant": arm,
            "retention_class": (
                "retain_diagnostic"
                if pilot
                else "retain_exploratory_evidence"
            ),
            "classification_confidence": "high",
        },
        "metadata": {
            "locked_config_materialization_receipt": (
                f"{output_reference}/{RECEIPT_NAME}"
            ),
            "frozen_scientific_contract": True,
            "sender_state_permutation_amendment_enforced": True,
            "mask_noninterference_supplement_enforced": True,
            "original_rewired_arm_authorized": False,
            "execution_role": role,
            "production_requires_resource_gate": not pilot,
            "stage2_requires_representation_gate": (
                not pilot and arm != "self"
            ),
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }
    validate_experiment_config(config)
    _assert_no_direct_identifiers(config, label="resolved config")
    return config


def _config_filename(alias: str, arm: str, *, pilot: bool) -> str:
    prefix = alias.lower()
    suffix = arm.replace("-", "_")
    role = "resource_pilot" if pilot else "science"
    return f"{prefix}_{suffix}_{role}.yaml"


def _atomic_publish(output_dir: Path, files: Mapping[str, bytes]) -> None:
    if output_dir.exists():
        for relative, expected in files.items():
            path = output_dir / relative
            if not path.is_file() or path.read_bytes() != expected:
                raise MultiscaleHurdleMaterializationError(
                    "Existing locked output differs from requested materialization"
                )
        unexpected = {
            path.relative_to(output_dir).as_posix()
            for path in output_dir.rglob("*")
            if path.is_file()
        } - set(files)
        if unexpected:
            raise MultiscaleHurdleMaterializationError(
                "Existing locked output contains unexpected files"
            )
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=output_dir.parent,
        )
    )
    try:
        for relative, content in files.items():
            destination = temporary_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        temporary_root.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _gpu_assignments(
    jobs: Sequence[Mapping[str, Any]],
    *,
    node_counts: Mapping[str, int],
) -> dict[tuple[str, str], int]:
    loads = {gpu: 0.0 for gpu in SAFE_GPU_IDS}
    assignments: dict[tuple[str, str], int] = {}
    ordered = sorted(
        jobs,
        key=lambda item: (
            -float(node_counts[str(item["alias"])]),
            str(item["alias"]),
            str(item["arm"]),
        ),
    )
    for item in ordered:
        gpu = min(SAFE_GPU_IDS, key=lambda value: (loads[value], value))
        key = (str(item["alias"]), str(item["arm"]))
        assignments[key] = gpu
        loads[gpu] += float(node_counts[key[0]])
    return assignments


def _parameter_audit(
    *,
    expression_mean: Any,
    expression_scale: Any,
    node_covariate_dim: int,
    model_components: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    import torch

    from spatial_benchmark.multiscale_hybrid import (
        MultiscaleAdditiveHybridModel,
    )

    baseline_shapes: dict[str, list[int]] | None = None
    baseline_count: int | None = None
    for arm in ARMS:
        section = model_components[arm]
        torch.manual_seed(0)
        model = MultiscaleAdditiveHybridModel(
            num_genes=1000,
            local_edge_attribute_dim=len(EDGE_ATTRIBUTE_NAMES),
            regional_edge_attribute_dim=len(EDGE_ATTRIBUTE_NAMES),
            expression_mean=expression_mean,
            expression_scale=expression_scale,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=int(section["hidden_dim"]),
            decoder_dim=int(section["decoder_dim"]),
            ffn_dim=int(section["ffn_dim"]),
            attention_heads=int(section["attention_heads"]),
            attention_head_dim=int(section["attention_head_dim"]),
            value_head_dim=int(section["value_head_dim"]),
            message_dim=int(section["message_dim"]),
            edge_hidden_dim=int(section["edge_hidden_dim"]),
            edge_embedding_dim=int(section["edge_embedding_dim"]),
            output_channels=int(section["output_channels_per_gene"]),
            dropout=float(section["dropout"]),
            attention_dropout=float(section["attention_dropout"]),
            receiver_chunk_size=int(section["receiver_chunk_size"]),
            activation_checkpointing=bool(
                section["activation_checkpointing"]
            ),
            regional_routing=str(section["regional_routing"]),
            local_routing=str(section["local_routing"]),
        )
        shapes = {
            name: list(parameter.shape)
            for name, parameter in model.named_parameters()
        }
        count = sum(parameter.numel() for parameter in model.parameters())
        if baseline_shapes is None:
            baseline_shapes = shapes
            baseline_count = count
        elif shapes != baseline_shapes or count != baseline_count:
            raise MultiscaleHurdleMaterializationError(
                "Routing arms do not have an exact parameter-shape match"
            )
        del model
        gc.collect()
    assert baseline_shapes is not None and baseline_count is not None
    return {
        "trainable_parameter_count": int(baseline_count),
        "named_parameter_shapes_sha256": canonical_sha256(baseline_shapes),
        "matched_arms": list(ARMS),
    }


def materialize_campaign(
    *,
    project_root: Path,
    source_receipt_path: Path,
    contract_path: Path,
    output_dir: Path,
    workers: int,
    parallel_cores: int = 1,
) -> dict[str, Any]:
    """Validate inputs, construct graph receipts, and publish locked configs."""

    project_root = project_root.resolve()
    source_receipt_path = source_receipt_path.resolve()
    contract_path = contract_path.resolve()
    output_dir = output_dir.resolve()
    for path, label in (
        (source_receipt_path, "source receipt"),
        (contract_path, "frozen contract"),
    ):
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise MultiscaleHurdleMaterializationError(
                f"{label} must remain under the project root"
            ) from exc
        if not path.is_file():
            raise MultiscaleHurdleMaterializationError(
                f"{label} does not exist: {path}"
            )
    try:
        output_dir.relative_to(project_root)
    except ValueError as exc:
        raise MultiscaleHurdleMaterializationError(
            "Output directory must remain under the project root"
        ) from exc
    if output_dir == project_root:
        raise MultiscaleHurdleMaterializationError(
            "Output directory cannot be the project root"
        )
    contract_sha = _verify_frozen_contract(project_root, contract_path)
    contract_amendment = _verify_contract_amendment(project_root)
    source_receipt, source_jobs = _load_source_receipt(
        project_root,
        source_receipt_path,
    )

    components: dict[str, str] = {}
    model_components: dict[str, Mapping[str, Any]] = {}
    for arm, reference in _MODEL_COMPONENTS.items():
        value, checksum = _component(project_root, reference)
        model_components[arm] = value
        components[reference] = checksum
    graph_component, checksum = _component(project_root, _GRAPH_COMPONENT)
    components[_GRAPH_COMPONENT] = checksum
    trainer_components: dict[str, Mapping[str, Any]] = {}
    evaluation_components: dict[str, Mapping[str, Any]] = {}
    for role in ("pilot", "science"):
        value, checksum = _component(
            project_root,
            _TRAINER_COMPONENTS[role],
        )
        trainer_components[role] = value
        components[_TRAINER_COMPONENTS[role]] = checksum
        value, checksum = _component(
            project_root,
            _EVALUATION_COMPONENTS[role],
        )
        evaluation_components[role] = value
        components[_EVALUATION_COMPONENTS[role]] = checksum
    launcher_component, checksum = _component(
        project_root,
        _LAUNCHER_COMPONENT,
    )
    components[_LAUNCHER_COMPONENT] = checksum

    if (
        isinstance(parallel_cores, bool)
        or not isinstance(parallel_cores, int)
        or parallel_cores <= 0
        or parallel_cores > len(STAGE2_ALIASES)
    ):
        raise MultiscaleHurdleMaterializationError(
            "parallel_cores must be between 1 and 5"
        )
    identities: dict[str, dict[str, Any]] = {}
    graph_records: dict[str, dict[str, Any]] = {}

    def accept(
        result: tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]
    ) -> None:
        alias, _, identity, record = result
        if alias in identities or alias not in STAGE2_ALIASES:
            raise MultiscaleHurdleMaterializationError(
                f"Unexpected or duplicate materialized core {alias}"
            )
        identities[alias] = identity
        graph_records[alias] = record

    if parallel_cores == 1:
        for alias in STAGE2_ALIASES:
            accept(
                _materialize_one_core(
                    project_root=project_root,
                    alias=alias,
                    source_job=source_jobs[
                        (alias, "hybrid-gat-k1000")
                    ],
                    workers=workers,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=parallel_cores) as executor:
            futures = {
                executor.submit(
                    _materialize_one_core,
                    project_root=project_root,
                    alias=alias,
                    source_job=dict(
                        source_jobs[(alias, "hybrid-gat-k1000")]
                    ),
                    workers=workers,
                ): alias
                for alias in STAGE2_ALIASES
            }
            for future in as_completed(futures):
                alias = futures[future]
                try:
                    accept(future.result())
                except Exception as exc:
                    raise MultiscaleHurdleMaterializationError(
                        f"{alias} parallel graph materialization failed"
                    ) from exc
    if set(identities) != set(STAGE2_ALIASES):
        raise MultiscaleHurdleMaterializationError(
            "Not all five core graph records were materialized"
        )
    parameter_audit = _parameter_audit(
        expression_mean=[0.0] * 1000,
        expression_scale=[1.0] * 1000,
        node_covariate_dim=len(ALLOWED_METADATA_COLUMNS),
        model_components=model_components,
    )

    pilot_jobs = [
        {"alias": alias, "arm": "self"} for alias in STAGE1_ALIASES
    ]
    science_jobs = [
        {"alias": alias, "arm": arm}
        for alias in STAGE2_ALIASES
        for arm in ARMS
    ]
    node_counts = {
        alias: int(identity["n_nodes"])
        for alias, identity in identities.items()
    }
    pilot_assignments = _gpu_assignments(
        pilot_jobs,
        node_counts=node_counts,
    )
    science_assignments = _gpu_assignments(
        science_jobs,
        node_counts=node_counts,
    )
    for label, jobs, assignments in (
        ("pilot", pilot_jobs, pilot_assignments),
        ("science", science_jobs, science_assignments),
    ):
        expected_keys = {
            (str(item["alias"]), str(item["arm"])) for item in jobs
        }
        if set(assignments) != expected_keys or any(
            value not in SAFE_GPU_IDS for value in assignments.values()
        ):
            raise MultiscaleHurdleMaterializationError(
                f"{label} GPU assignments are incomplete or unsafe"
            )
    output_reference = _project_reference(project_root, output_dir)
    contract_reference = _project_reference(project_root, contract_path)
    generated: dict[str, bytes] = {}
    pilot_records: list[dict[str, Any]] = []
    science_records: list[dict[str, Any]] = []
    for pilot, jobs, assignments, records, directory in (
        (
            True,
            pilot_jobs,
            pilot_assignments,
            pilot_records,
            "resource_pilot_configs",
        ),
        (
            False,
            science_jobs,
            science_assignments,
            science_records,
            "science_configs",
        ),
    ):
        role = "pilot" if pilot else "science"
        for item in jobs:
            alias = str(item["alias"])
            arm = str(item["arm"])
            config = _build_config(
                source_config=identities[alias]["source_config"],
                alias=alias,
                arm=arm,
                pilot=pilot,
                requested_gpu=assignments[(alias, arm)],
                model_component=model_components[arm],
                graph_component=graph_component,
                graph_record=graph_records[alias],
                trainer_component=trainer_components[role],
                evaluation_component=evaluation_components[role],
                launcher_component=launcher_component,
                contract_reference=contract_reference,
                contract_sha256=contract_sha,
                contract_amendment=contract_amendment,
                output_reference=output_reference,
            )
            filename = _config_filename(alias, arm, pilot=pilot)
            relative = f"{directory}/{filename}"
            content = _yaml_bytes(config)
            generated[relative] = content
            records.append(
                {
                    "alias": alias,
                    "arm": arm,
                    "stage": 1 if pilot or arm == "self" else 2,
                    "config": f"{output_reference}/{relative}",
                    "config_sha256": canonical_sha256(config),
                    "file_sha256": _sha256_bytes(content),
                    "requested_gpu": assignments[(alias, arm)],
                    "n_nodes": node_counts[alias],
                    "graph_bundle_sha256": _mapping(
                        graph_records[alias]["bundle_checksums"],
                        "bundle checksums",
                    )["bundle_sha256"],
                    "local_source_permutation_sha256": _mapping(
                        graph_records[alias][
                            "local_source_permutation"
                        ],
                        "local source permutation",
                    )["checksum"],
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
        "contract_amendment": dict(contract_amendment),
        "source": {
            "campaign_id": SOURCE_CAMPAIGN_ID,
            "materialization_reference": _project_reference(
                project_root,
                source_receipt_path,
            ),
            "file_sha256": sha256_file(source_receipt_path),
            "canonical_checksum": source_receipt["checksum"],
        },
        "component_file_sha256": dict(sorted(components.items())),
        "parameter_audit": parameter_audit,
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
        "cores": [
            {
                "alias": alias,
                "n_nodes": identities[alias]["n_nodes"],
                "n_genes": identities[alias]["n_genes"],
                "prepared_artifact": identities[alias][
                    "prepared_artifact"
                ],
                "source_prepared_data_sha256": identities[alias][
                    "source_prepared_data_sha256"
                ],
                "preprocessing_sha256": identities[alias][
                    "preprocessing_sha256"
                ],
                "split_fingerprint": identities[alias][
                    "split_fingerprint"
                ],
                "graph_receipt": graph_records[alias],
                "graph_receipt_sha256": canonical_sha256(
                    graph_records[alias]
                ),
            }
            for alias in STAGE2_ALIASES
        ],
        "allowed_gpu_ids": list(SAFE_GPU_IDS),
        "resource_limits": {
            "preferred_aggregate_gpu_hours": 12.0,
            "absolute_aggregate_gpu_hours": 24.0,
            "stage1_peak_allocated_vram_gib": 12.0,
            "per_device_peak_allocated_vram_gib": 20.5,
            "aggregate_observed_process_vram_gib": 50.0,
            "filesystem_used_decimal_gb_hard_stop": 55.0,
        },
        "pilot_jobs": pilot_records,
        "science_jobs": science_records,
        "resource_gate_receipt_reference": (
            f"{output_reference}/resource_gate_receipt.json"
        ),
        "representation_gate_receipt_reference": (
            f"{output_reference}/representation_gate_receipt.json"
        ),
        "counts": {
            "cores": len(identities),
            "pilot_configs": len(pilot_records),
            "science_configs": len(science_records),
        },
        "registry_mutation_performed": False,
        "queue_mutation_performed": False,
        "training_performed": False,
    }
    if receipt["counts"] != {
        "cores": 5,
        "pilot_configs": 2,
        "science_configs": 20,
    }:
        raise MultiscaleHurdleMaterializationError(
            "Materialized job counts violate the frozen campaign"
        )
    _assert_no_direct_identifiers(receipt, label="materialization receipt")
    receipt["checksum"] = canonical_sha256(receipt)
    generated[RECEIPT_NAME] = _json_bytes(receipt)
    _atomic_publish(output_dir, generated)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = current_paths()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=paths.project_root,
    )
    parser.add_argument(
        "--source-receipt",
        type=Path,
        default=(
            paths.project_root
            / "scratch/locked_campaigns"
            / SOURCE_CAMPAIGN_ID
            / RECEIPT_NAME
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            paths.project_root
            / "experiments/campaigns"
            / CAMPAIGN_ID
            / "frozen_task_contract.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            paths.project_root
            / "scratch/locked_campaigns"
            / CAMPAIGN_ID
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--parallel-cores",
        type=int,
        default=5,
        help="Independent core graph builds to run concurrently (1-5).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = materialize_campaign(
        project_root=args.project_root,
        source_receipt_path=args.source_receipt,
        contract_path=args.contract,
        output_dir=args.output_dir,
        workers=args.workers,
        parallel_cores=args.parallel_cores,
    )
    print(
        json.dumps(
            {
                "campaign_id": receipt["campaign_id"],
                "counts": receipt["counts"],
                "output_dir": str(args.output_dir.resolve()),
                "checksum": receipt["checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
