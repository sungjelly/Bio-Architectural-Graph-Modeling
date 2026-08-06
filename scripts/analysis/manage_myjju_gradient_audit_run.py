#!/usr/bin/env python3
"""Prepare and finalize the aggregate MyJJu gradient-audit run.

This is a deliberately campaign-specific two-phase wrapper.  ``prepare``
registers exactly one aggregate post-hoc run and creates its owned active
bundle at ``scratch/active_runs/<run_id>``.  The numerical workflow writes its
pilot, seed shards, and aggregate report beneath the returned work root.

``finalize`` accepts only the frozen report-manifest schema, verifies every
declared input and output checksum, removes verified cell-level temporary NPZ
files, constructs the canonical run-archive contract, publishes it without a
marker, commits registry/checkpoint metadata, and only then writes ``_SUCCESS``.
It is restartable across each durable finalization boundary and never updates a
completed bundle.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import socket
import sqlite3
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.checkpoint_catalog import (  # noqa: E402
    index_checkpoint_catalog,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import (  # noqa: E402
    canonical_json,
    canonical_sha256,
    create_run_id,
    repro_id,
    scientific_id,
)
from spatial_benchmark.paths import (  # noqa: E402
    ProjectPaths,
    current_paths,
)
from spatial_benchmark.queueing import _provenance  # noqa: E402
from spatial_benchmark.registry import Registry, utc_now  # noqa: E402
from spatial_benchmark.run_archive import (  # noqa: E402
    COMPLETION_MARKERS,
    RunArchive,
    RunValidationError,
    validate_prediction_rows,
    verify_run_bundle,
    verify_unmarked_run_bundle,
)


CAMPAIGN_ID = "cmp_20260731_myjju_genemae_gradient_audit_cpu_replay"
UPSTREAM_CAMPAIGN_ID = "cmp_20260730_myjju_genemae_10core_comparison"
FROZEN_CONTRACT_SHA256 = (
    "bcdd6d7223f9b97f26833f77935c75645110aaba1669d11c0a7180c957514cc1"
)
IMPLEMENTATION_PROTOCOL_SHA256 = (
    "b4328095bcde383633fb2862b3c35b6bc8203c98906aee7f2058443b2d16ddb8"
)
COHORT_FINGERPRINT_SHA256 = (
    "b3c06228ce7fda4d4c5da09c3281de46e67b8069dfadb14276ba6402af87767e"
)
GRAPH_BUNDLE_SHA256 = (
    "f860f84f96e9daf4c9c4e4ddd5fc40aeacd3f6362c9e97d07248ac2575b462d5"
)
FROZEN_CONTRACT_RELATIVE = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "frozen_task_contract.yaml"
)
IMPLEMENTATION_PROTOCOL_RELATIVE = (
    Path("experiments/campaigns")
    / CAMPAIGN_ID
    / "implementation_protocol.yaml"
)
CAMPAIGN_RELATIVE = Path("experiments/campaigns") / CAMPAIGN_ID / "campaign.yaml"
WORK_RELATIVE = Path("diagnostics/audit_work")
REPORT_MANIFEST_RELATIVE = WORK_RELATIVE / "aggregate/manifest.json"
RUN_CONTEXT_RELATIVE = WORK_RELATIVE / "run_context.json"
PREPARE_RECEIPT_RELATIVE = Path("diagnostics/aggregate_run_receipt.json")
TRANSIENT_CLEANUP_RECEIPT_RELATIVE = (
    Path("diagnostics/transient_cleanup_receipt.json")
)
REPORT_MANIFEST_KIND = "myjju_genemae_gradient_audit_report_manifest"
REPORT_MANIFEST_SCHEMA_VERSION = 1
PREPARE_RECEIPT_KIND = "myjju_gradient_audit_aggregate_run_v1"
TRANSIENT_RECEIPT_KIND = "myjju_gradient_audit_transient_cleanup_v1"
PRIMARY_METRIC = "audit/gate_row_pass_fraction_descriptive"
MODEL_PARAMETER_COUNT = 6_888_016
EXPECTED_SEEDS = tuple(range(7))
EXPECTED_CORES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
EXPECTED_MASK_REPLICATES = (0, 1, 2)
EXPECTED_TARGETS = ("KRT8", "COL1A1", "EPCAM", "OLFM4", "CEACAM6")
DATASET_ID = "cosmx_adjacent_normal_10core_pooled_fit_v1"
DATASET_VERSION = "adjacent_normal_10core_pooled_fit_v1"
SPLIT_ID = "held_in_all_fit_no_independent_split"
SPLIT_FINGERPRINT = canonical_sha256(
    {
        "schema": "held_in_all_fit_no_independent_split_v1",
        "dataset_fingerprint": COHORT_FINGERPRINT_SHA256,
        "core_aliases": EXPECTED_CORES,
        "role": "fit",
        "generalization_estimate": False,
    }
)
REQUIRED_REPORT_FILES = {
    "audit_report_json": "report.json",
    "gate_results_csv": "gate_results.csv",
    "audit_report_markdown": "report.md",
    "audit_report_html": "report.html",
    "canonical_fit_predictions_jsonl": "canonical_fit_predictions.jsonl",
}
REQUIRED_GATE_FAMILIES = frozenset(
    {
        "target_predictive_eligibility",
        "target_graph_use_eligibility",
        "seed_rank_stability",
        "mask_rank_stability",
        "signed_pair_stability",
        "bounded_faithfulness",
        "graph_gradient_structure_null",
        "parameter_randomization",
        "matched_pair_null",
    }
)
EXPECTED_LOCKED_PAIRS = frozenset(
    {
        ("KRT8", "KRT19"),
        ("KRT8", "KRT18"),
        ("KRT8", "KRT7"),
        ("COL1A1", "COL3A1"),
        ("COL1A1", "COL1A2"),
        ("COL1A1", "DCN"),
        ("EPCAM", "PSCA"),
        ("EPCAM", "OLFM4"),
        ("EPCAM", "KRT19"),
        ("EPCAM", "CLDN4"),
        ("EPCAM", "CDH1"),
        ("OLFM4", "PSCA"),
        ("OLFM4", "EPCAM"),
        ("OLFM4", "KRT19"),
        ("OLFM4", "CDH1"),
        ("CEACAM6", "KRT19"),
        ("CEACAM6", "KRT8"),
        ("CEACAM6", "CLDN4"),
        ("CEACAM6", "KRT18"),
    }
)
EXPECTED_GATE_COUNTS = {
    "target_predictive_eligibility": 5,
    "target_graph_use_eligibility": 5,
    "seed_rank_stability": 1,
    "mask_rank_stability": 5,
    "signed_pair_stability": 19,
    "bounded_faithfulness": 2,
    "graph_gradient_structure_null": 1,
    "parameter_randomization": 1,
    "matched_pair_null": 1,
}
MAXIMUM_DEFENSIBLE_CLAIMS = frozenset(
    {
        (
            "candidate_set_contains_stable_faithful_graph_dependent_"
            "null_calibrated_model_implied_predictive_sensitivities"
        ),
        "no_claim_beyond_reported_model_behavior",
    }
)
EXPECTED_TARGET_SCOPES = frozenset(
    EXPECTED_TARGETS
)
EXPECTED_GATE_SCOPES = {
    "target_predictive_eligibility": EXPECTED_TARGET_SCOPES,
    "target_graph_use_eligibility": EXPECTED_TARGET_SCOPES,
    "seed_rank_stability": frozenset({"all_cores"}),
    "mask_rank_stability": EXPECTED_TARGET_SCOPES,
    "signed_pair_stability": frozenset(
        f"{target}<-{source}" for target, source in EXPECTED_LOCKED_PAIRS
    ),
    "bounded_faithfulness": frozenset(
        {
            "scale_0.10_sd_offdiagonal",
            "scale_0.25_sd_offdiagonal",
        }
    ),
    "graph_gradient_structure_null": frozenset({"aggregate"}),
    "parameter_randomization": frozenset({"aggregate"}),
    "matched_pair_null": frozenset({"aggregate"}),
}
_DIGEST_CHARACTERS = frozenset("0123456789abcdef")
_FORBIDDEN_COMMAND_TOKENS = (
    "password",
    "passwd",
    "secret",
    "api-key",
    "api_key",
    "access-token",
    "access_token",
    "sample-key-salt",
)
_FORBIDDEN_NPZ_KEY_TOKENS = (
    "barcode",
    "cell_id",
    "cellid",
    "cell_index",
    "cell_indices",
    "donor",
    "patient",
    "person",
    "receiver_index",
    "receiver_indices",
    "row_index",
    "row_indices",
    "subject",
)
_FORBIDDEN_RETAINED_JSON_KEYS = frozenset(
    {
        "barcode",
        "cell_barcode",
        "cell_id",
        "cell_ids",
        "cell_index",
        "cell_indices",
        "donor_id",
        "patient_id",
        "person_id",
        "receiver_index",
        "receiver_indices",
        "row_id",
        "row_ids",
        "row_index",
        "row_indices",
        "subject_id",
    }
)


class GradientAuditRunError(RuntimeError):
    """Raised when aggregate run preparation/finalization is unsafe."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GradientAuditRunError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise GradientAuditRunError(f"{label} must be a list")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GradientAuditRunError(f"{label} must be an integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise GradientAuditRunError(f"{label} must be finite numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GradientAuditRunError(f"{label} must be finite numeric") from exc
    if not math.isfinite(number):
        raise GradientAuditRunError(f"{label} must be finite numeric")
    return number


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _DIGEST_CHARACTERS for character in value)
    ):
        raise GradientAuditRunError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _strict_json(path: Path, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise GradientAuditRunError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GradientAuditRunError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except GradientAuditRunError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GradientAuditRunError(f"{label} is not strict JSON") from exc


def _strict_json_line(value: str, *, label: str) -> Mapping[str, Any]:
    def reject_constant(constant: str) -> None:
        raise GradientAuditRunError(
            f"{label} contains non-finite JSON constant {constant!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise GradientAuditRunError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = item
        return result

    try:
        payload = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except GradientAuditRunError:
        raise
    except json.JSONDecodeError as exc:
        raise GradientAuditRunError(f"{label} is not strict JSON") from exc
    return _mapping(payload, label)


def _signed_payload(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    result = dict(payload)
    checksum = _digest(result.pop("checksum", None), f"{label}.checksum")
    if not hmac.compare_digest(canonical_sha256(result), checksum):
        raise GradientAuditRunError(f"{label} checksum does not verify")
    return {**result, "checksum": checksum}


def _safe_relative(reference: Any, *, label: str) -> Path:
    if not isinstance(reference, str):
        raise GradientAuditRunError(f"{label} path must be a string")
    pure = PurePosixPath(reference)
    if (
        not reference
        or pure.is_absolute()
        or reference != pure.as_posix()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise GradientAuditRunError(f"{label} has unsafe path {reference!r}")
    return Path(*pure.parts)


def _regular_child(base: Path, relative: Path, *, label: str) -> Path:
    root = base.resolve(strict=True)
    candidate = (base / relative).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise GradientAuditRunError(f"{label} escapes {root}") from exc
    lexical = base / relative
    if lexical.is_symlink() or not lexical.is_file():
        raise GradientAuditRunError(
            f"{label} is missing or is not a regular file: {lexical}"
        )
    if candidate != lexical.resolve(strict=True):
        raise GradientAuditRunError(f"{label} resolves unexpectedly: {lexical}")
    return lexical


def _verify_file_descriptor(
    raw: Any,
    *,
    base: Path,
    label: str,
    expected_path: str | None = None,
    expected_keys: set[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    descriptor = dict(_mapping(raw, label))
    required = expected_keys or {"path", "sha256", "size_bytes"}
    if set(descriptor) != required:
        raise GradientAuditRunError(
            f"{label} must have exact keys {sorted(required)}"
        )
    relative = _safe_relative(descriptor.get("path"), label=label)
    if expected_path is not None and relative.as_posix() != expected_path:
        raise GradientAuditRunError(
            f"{label} path must be {expected_path!r}"
        )
    digest = _digest(descriptor.get("sha256"), f"{label}.sha256")
    size = _integer(descriptor.get("size_bytes"), f"{label}.size_bytes")
    if size < 0:
        raise GradientAuditRunError(f"{label}.size_bytes may not be negative")
    path = _regular_child(base, relative, label=label)
    if path.stat().st_size != size or not hmac.compare_digest(
        sha256_file(path), digest
    ):
        raise GradientAuditRunError(f"{label} size or checksum mismatch")
    return descriptor, path


def _verify_sidecar(path: Path, descriptor: Mapping[str, Any]) -> Path:
    sidecar = path.with_name(f"{path.name}.sha256.json")
    payload = _strict_json(sidecar, label=f"{path.name} checksum sidecar")
    expected = {
        "path": path.name,
        "sha256": descriptor["sha256"],
        "size_bytes": descriptor["size_bytes"],
    }
    if payload != expected:
        raise GradientAuditRunError(
            f"checksum sidecar disagrees with {path.name}"
        )
    return sidecar


def _verify_frozen_inputs(paths: ProjectPaths) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = paths.project_root / FROZEN_CONTRACT_RELATIVE
    protocol_path = paths.project_root / IMPLEMENTATION_PROTOCOL_RELATIVE
    campaign_path = paths.project_root / CAMPAIGN_RELATIVE
    if sha256_file(contract_path) != FROZEN_CONTRACT_SHA256:
        raise GradientAuditRunError("frozen task contract checksum changed")
    if sha256_file(protocol_path) != IMPLEMENTATION_PROTOCOL_SHA256:
        raise GradientAuditRunError("implementation protocol checksum changed")
    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise GradientAuditRunError("campaign inputs are unreadable") from exc
    contract_map = dict(_mapping(contract, "frozen task contract"))
    protocol_map = dict(_mapping(protocol, "implementation protocol"))
    campaign_map = _mapping(campaign, "campaign declaration")
    if (
        contract_map.get("campaign_id") != CAMPAIGN_ID
        or protocol_map.get("campaign_id") != CAMPAIGN_ID
        or protocol_map.get("frozen_contract_sha256")
        != FROZEN_CONTRACT_SHA256
        or campaign_map.get("campaign_id") != CAMPAIGN_ID
        or campaign_map.get("frozen_contract_sha256")
        != FROZEN_CONTRACT_SHA256
        or campaign_map.get("exploratory") is not True
    ):
        raise GradientAuditRunError("campaign inputs disagree")
    verified = _mapping(protocol_map.get("verified_preflight"), "verified_preflight")
    if (
        verified.get("cohort_fingerprint_sha256")
        != COHORT_FINGERPRINT_SHA256
        or verified.get("graph_bundle_sha256") != GRAPH_BUNDLE_SHA256
        or verified.get("total_nodes") != 117_386
        or tuple(verified.get("tile_counts", {})) != EXPECTED_CORES
    ):
        raise GradientAuditRunError("implementation preflight identity drifted")
    return contract_map, protocol_map


def _resolved_configuration(
    contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    return {
        "campaign": {
            "campaign_id": CAMPAIGN_ID,
            "upstream_campaign_id": UPSTREAM_CAMPAIGN_ID,
            "exploratory": True,
        },
        "classification": {
            "schema_version": 1,
            "lifecycle_stage": "exploratory_screen",
            "study_axis": "gradient_attribution_audit",
            "retention_class": "retain_exploratory_evidence",
            "classification_confidence": "high",
            "source_batch": CAMPAIGN_ID,
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "version": DATASET_VERSION,
            "split_id": SPLIT_ID,
            "dataset_fingerprint": COHORT_FINGERPRINT_SHA256,
            "split_fingerprint": SPLIT_FINGERPRINT,
            "preprocessing_version": "full_cell_log1p_cp10k_v1",
            "fit_scope": "all_117386_cells_across_ten_cores_transductive",
            "core_aliases": list(EXPECTED_CORES),
            "experimental_unit": "adjacent_normal_spatial_core",
            "generalization_estimate": False,
        },
        "model": {
            "name": "myjju-genemae",
            "family": "myjju_dual_path_genemae",
            "embedding_dim": 192,
            "graph_layers": 4,
            "expected_trainable_parameters_per_upstream_model": (
                MODEL_PARAMETER_COUNT
            ),
            "upstream_model_seeds": list(EXPECTED_SEEDS),
        },
        "masking": {
            "type": "partial_gene_expression_gradient_audit",
            "rate": 0.2,
            "replicates": list(EXPECTED_MASK_REPLICATES),
        },
        "graph": {
            "graph_id": "myjju-k15-tiled-union",
            "kind": "symmetric_spatial_knn_tiled",
            "neighbor_k": 15,
            "symmetry": "union",
            "maximum_tile_nodes": 7000,
            "graph_bundle_sha256": GRAPH_BUNDLE_SHA256,
            "edge_control": "observed_and_node_label_permuted",
        },
        "features": {
            "use_edge_features": True,
            "edge_features": ["gaussian_distance_kernel"],
        },
        "trainer": {
            "checkpoint_policy": "analysis_state_only",
            "primary_checkpoint_role": "last",
            "restore_best": False,
            "upstream_checkpoints_frozen": True,
            "retraining": False,
        },
        "evaluation": {
            "protocol": "held_in_pooled_10core_fixed_budget",
            "canonical_prediction_split": "fit",
            "primary_metric": PRIMARY_METRIC,
            "primary_direction": "maximize",
            "biological_unit": "core",
            "weighting": "equal_core",
            "patient_generalization_supported": False,
        },
        "experiment": {
            "variant_label": "myjju_genemae_gradient_audit_aggregate",
            "posthoc": True,
            "exploratory": True,
            "conclusion_eligible": False,
            "mechanism_validation_available": False,
        },
        "audit": {
            "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
            "implementation_protocol_sha256": (
                IMPLEMENTATION_PROTOCOL_SHA256
            ),
            "marker_gene_count": 39,
            "locked_pair_count": 19,
            "target_count": 5,
            "checkpoint_count": 7,
            "core_count": 10,
            "mask_replicate_count": 3,
            "claim_ceiling": (
                "stable_faithful_graph_dependent_null_calibrated_"
                "model_implied_predictive_sensitivity"
            ),
            "prohibited_claim": "validated_biological_mechanism",
            "contract_schema_version": contract.get(
                "contract_schema_version"
            ),
            "protocol_schema_version": protocol.get(
                "protocol_schema_version"
            ),
        },
        # Aggregate run IDs require execution coordinates in the current
        # schema.  Registry knownness below explicitly marks seed/fold as
        # placeholders rather than observed aggregate coordinates.
        "seed": 0,
        "fold": 0,
        "attempt": attempt,
    }


def _capture_provenance(
    configuration: Mapping[str, Any],
    paths: ProjectPaths,
) -> dict[str, Any]:
    return _provenance(configuration, paths.project_root)


def _semantic_alias(run_id: str, scientific_identifier: str) -> str:
    variant = canonical_sha256(
        {
            "scientific_id": scientific_identifier,
            "dataset": DATASET_ID,
            "graph": GRAPH_BUNDLE_SHA256,
            "contract": FROZEN_CONTRACT_SHA256,
            "protocol": IMPLEMENTATION_PROTOCOL_SHA256,
        }
    )[:12]
    execution = canonical_sha256(
        {
            "run_id": run_id,
            "aggregate_model_seeds": EXPECTED_SEEDS,
            "aggregate_cores": EXPECTED_CORES,
            "attempt": 1,
        }
    )[:12]
    return (
        "run.exploratory-screen.gradient-attribution-audit."
        "myjju-genemae.myjju-k15-tiled-union."
        "partial-gene-expression-gradient-audit.enabled."
        f"d192.sna.fna.a01.v{variant}.x{execution}"
    )


def _run_semantics(
    run_id: str,
    scientific_identifier: str,
) -> dict[str, Any]:
    alias = _semantic_alias(run_id, scientific_identifier)
    return {
        "lifecycle_stage": "exploratory_screen",
        "study_axis": "gradient_attribution_audit",
        "source_batch": CAMPAIGN_ID,
        "variant_label": "myjju_genemae_gradient_audit_aggregate",
        "model_key": "myjju-genemae",
        "dataset_key": f"{DATASET_ID}@{DATASET_VERSION}",
        "masking_key": "partial_gene_expression_gradient_audit",
        "graph_key": "myjju-k15-tiled-union",
        "feature_key": "edge-enabled",
        "embedding_key": "d192",
        "seed_known": False,
        "fold_known": False,
        "attempt_known": True,
        "retention_class": "retain_exploratory_evidence",
        "category_key": (
            "exploratory_screen/gradient_attribution_audit/"
            f"{DATASET_ID}/myjju-genemae/aggregate/"
            "myjju-k15-tiled-union/"
            "partial_gene_expression_gradient_audit/edge-enabled/d192"
        ),
        "classification_confidence": "high",
        "timestamp_basis": "registry",
        "rules": {
            "schema_version": 1,
            "classification_source": "resolved_config",
            "aggregate_model_seeds": list(EXPECTED_SEEDS),
            "aggregate_cores": list(EXPECTED_CORES),
            "aggregate_mask_replicates": list(EXPECTED_MASK_REPLICATES),
            "seed_placeholder": 0,
            "fold_placeholder": 0,
            "seed_and_fold_are_not_observed_coordinates": True,
            "mechanism_validation_available": False,
        },
        "semantic_alias": alias,
        "preferred_alias_type": "semantic",
        "alias_metadata": {
            "semantic": {
                "schema_version": 1,
                "generated_from": "aggregate_gradient_audit_config",
                "primary_run_id": run_id,
                "aggregate_coordinates": True,
            }
        },
    }


@contextmanager
def _registry_lock(paths: ProjectPaths) -> Iterator[None]:
    lock_path = paths.state_root / "locks/myjju-gradient-audit-registry.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GradientAuditRunError(
                f"another gradient-audit registry operation owns {lock_path}"
            ) from exc
        yield


def _backup_registry(
    registry: Registry,
    paths: ProjectPaths,
    *,
    operation: str,
) -> Path:
    directory = paths.state_root / "backups/gradient_audit"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / f"{operation}-{stamp}.sqlite3"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=directory,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with registry.connect() as source, sqlite3.connect(temporary) as target:
            source.backup(target)
        if destination.exists() or destination.is_symlink():
            raise GradientAuditRunError(
                f"registry backup destination already exists: {destination}"
            )
        temporary.rename(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name in COMPLETION_MARKERS:
            continue
        relative = path.relative_to(root)
        records.append(
            {
                "kind": (
                    relative.parts[0] if len(relative.parts) > 1 else "metadata"
                ),
                "path": path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "status": "present",
            }
        )
    return records


def _write_prepare_provenance(
    archive: RunArchive,
    *,
    provenance: Mapping[str, Any],
    invocation: Sequence[str],
    configuration: Mapping[str, Any],
    scientific_identifier: str,
    reproduction_identifier: str,
) -> None:
    archive.prepare_log_files()
    archive.write_json(
        "provenance/git.json",
        {
            "commit": provenance["git_commit"],
            "dirty": provenance["dirty_fingerprint"] is not None,
            "dirty_fingerprint": provenance["dirty_fingerprint"],
            "untracked_file_count": len(provenance["untracked_files"]),
        },
    )
    archive.write_text(
        "provenance/uncommitted_changes.patch",
        str(provenance["uncommitted_changes_patch"]),
    )
    archive.write_json(
        "provenance/untracked_files.json",
        {
            "files": provenance["untracked_files"],
            "note": (
                "Hashes cover non-ignored untracked files. Protected data are "
                "not embedded in this bundle."
            ),
        },
    )
    archive.write_text(
        "provenance/environment.txt",
        str(provenance["environment_text"]),
    )
    archive.write_json(
        "provenance/hardware.json",
        {
            "host": socket.gethostname(),
            "aggregate_run": True,
            **dict(_mapping(provenance["hardware"], "hardware provenance")),
        },
    )
    archive.write_json(
        "provenance/data_fingerprints.json",
        {
            "dataset_id": DATASET_ID,
            "dataset_version": DATASET_VERSION,
            "dataset_fingerprint": COHORT_FINGERPRINT_SHA256,
            "graph_bundle_sha256": GRAPH_BUNDLE_SHA256,
            "preprocessing_version": configuration["dataset"][
                "preprocessing_version"
            ],
        },
    )
    archive.write_json(
        "provenance/split_fingerprint.json",
        {
            "split_id": SPLIT_ID,
            "split_fingerprint": SPLIT_FINGERPRINT,
            "role": "all cells are held-in fit observations",
            "generalization_estimate": False,
        },
    )
    archive.write_text(
        "provenance/command.txt",
        canonical_json(
            {
                "prepare_argv": list(invocation),
                "working_directory": archive.paths.project_root.as_posix(),
                "numerical_execution_commands": (
                    "bound by provenance/audit_execution_commands.json "
                    "during finalization"
                ),
            }
        )
        + "\n",
    )
    archive.write_json(
        "provenance/identifiers.json",
        {
            "run_id": archive.run_id,
            "scientific_id": scientific_identifier,
            "repro_id": reproduction_identifier,
        },
    )


def _prepare_receipt(
    *,
    run_id: str,
    scientific_identifier: str,
    reproduction_identifier: str,
    archive: RunArchive,
) -> dict[str, Any]:
    core = {
        "kind": PREPARE_RECEIPT_KIND,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "run_id": run_id,
        "scientific_id": scientific_identifier,
        "repro_id": reproduction_identifier,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
        "active_run_root": archive.scratch_path.as_posix(),
        "work_root": (archive.scratch_path / WORK_RELATIVE).as_posix(),
        "report_manifest": (
            archive.scratch_path / REPORT_MANIFEST_RELATIVE
        ).as_posix(),
        "artifact_path": archive.artifact_path.as_posix(),
    }
    return {**core, "checksum": canonical_sha256(core)}


def _existing_aggregate_runs(
    registry: Registry,
) -> list[dict[str, Any]]:
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT r.* FROM runs r
            LEFT JOIN run_categories c ON c.run_id = r.run_id
            WHERE r.campaign_id = ?
              AND (
                    c.study_axis = 'gradient_attribution_audit'
                    OR instr(
                        r.config_json,
                        '"variant_label":"myjju_genemae_gradient_audit_aggregate"'
                    ) > 0
                  )
            ORDER BY r.created_at, r.run_id
            """,
            (CAMPAIGN_ID,),
        ).fetchall()
    return [dict(row) for row in rows]


def _validate_registered_campaign(registry: Registry) -> None:
    campaign = registry.get_campaign(CAMPAIGN_ID)
    if campaign is None:
        raise GradientAuditRunError(
            f"campaign is not registered: {CAMPAIGN_ID}"
        )
    config = _mapping(campaign.get("config"), "registered campaign config")
    if (
        config.get("frozen_contract_sha256") != FROZEN_CONTRACT_SHA256
        or config.get("exploratory") is not True
        or campaign.get("status") not in {"planned", "running", "complete"}
    ):
        raise GradientAuditRunError(
            "registered campaign does not match the frozen exploratory audit"
        )


def prepare_run(
    *,
    registry: Registry,
    paths: ProjectPaths,
    attempt: int = 1,
    invocation: Sequence[str] = (),
) -> dict[str, Any]:
    """Register one aggregate run and create its owned active bundle."""

    if isinstance(attempt, bool) or attempt != 1:
        raise GradientAuditRunError(
            "the frozen aggregate audit permits exactly attempt 1"
        )
    paths.validate()
    contract, protocol = _verify_frozen_inputs(paths)
    _validate_registered_campaign(registry)
    configuration = _resolved_configuration(
        contract,
        protocol,
        attempt=attempt,
    )
    scientific_identifier = scientific_id(configuration)
    with _registry_lock(paths):
        existing = _existing_aggregate_runs(registry)
        if existing:
            if len(existing) != 1:
                raise GradientAuditRunError(
                    "multiple aggregate audit runs are registered"
                )
            run = existing[0]
            if str(run["scientific_id"]) != scientific_identifier:
                raise GradientAuditRunError(
                    "a gradient-audit aggregate run already exists under a "
                    "different frozen scientific identity"
                )
            if run["status"] == "failed":
                raise GradientAuditRunError(
                    "the sole frozen aggregate attempt failed; a new attempt "
                    "requires a revised documented contract"
                )
            run_id = str(run["run_id"])
            scratch = paths.scratch_root / "active_runs" / run_id
            artifact = RunArchive.artifact_path_for(run_id, paths)
            return {
                "created": False,
                "run_id": run_id,
                "status": str(run["status"]),
                "scientific_id": scientific_identifier,
                "repro_id": str(run["repro_id"]),
                "active_run_root": scratch.as_posix(),
                "work_root": (scratch / WORK_RELATIVE).as_posix(),
                "report_manifest": (
                    scratch / REPORT_MANIFEST_RELATIVE
                ).as_posix(),
                "artifact_path": artifact.as_posix(),
            }

        campaign = registry.get_campaign(CAMPAIGN_ID)
        assert campaign is not None
        if campaign.get("status") == "complete":
            raise GradientAuditRunError(
                "cannot create a new aggregate run for a complete campaign"
            )

        provenance = _capture_provenance(configuration, paths)
        reproduction_identifier = repro_id(
            configuration,
            git_commit=str(provenance["git_commit"]),
            dirty_fingerprint=provenance["dirty_fingerprint"],
            dataset_fingerprint=COHORT_FINGERPRINT_SHA256,
            split_fingerprint=SPLIT_FINGERPRINT,
            preprocessing_version=str(
                configuration["dataset"]["preprocessing_version"]
            ),
            environment_fingerprint=str(
                provenance["environment_fingerprint"]
            ),
        )
        run_id = create_run_id(
            seed=0,
            fold=0,
            attempt=attempt,
            scientific_id_value=scientific_identifier,
        )
        artifact_path = RunArchive.artifact_path_for(run_id, paths)
        backup = _backup_registry(
            registry,
            paths,
            operation="prepare",
        )
        archive: RunArchive | None = None
        run_registered = False
        try:
            registry.register_variant(
                scientific_identifier,
                campaign_id=CAMPAIGN_ID,
                configuration=configuration,
            )
            registry.create_run(
                run_id,
                campaign_id=CAMPAIGN_ID,
                scientific_id=scientific_identifier,
                repro_id=reproduction_identifier,
                seed=0,
                fold=0,
                attempt=attempt,
                configuration=configuration,
                status="pending",
                artifact_path=artifact_path,
                git_commit=provenance["git_commit"],
                dirty_status=provenance["dirty_fingerprint"] is not None,
                host=socket.gethostname(),
                gpu_model=provenance.get("gpu_model"),
            )
            run_registered = True
            registry.register_run_semantics(
                run_id,
                **_run_semantics(run_id, scientific_identifier),
            )
            archive = RunArchive.create(
                run_id,
                paths=paths,
                resolved_config=configuration,
            )
            _write_prepare_provenance(
                archive,
                provenance=provenance,
                invocation=invocation,
                configuration=configuration,
                scientific_identifier=scientific_identifier,
                reproduction_identifier=reproduction_identifier,
            )
            receipt = _prepare_receipt(
                run_id=run_id,
                scientific_identifier=scientific_identifier,
                reproduction_identifier=reproduction_identifier,
                archive=archive,
            )
            archive.write_json(PREPARE_RECEIPT_RELATIVE, receipt)
            archive.write_json(
                RUN_CONTEXT_RELATIVE,
                {
                    "run_id": run_id,
                    "campaign_id": CAMPAIGN_ID,
                    "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
                    "implementation_protocol_sha256": (
                        IMPLEMENTATION_PROTOCOL_SHA256
                    ),
                    "report_manifest_schema": {
                        "kind": REPORT_MANIFEST_KIND,
                        "schema_version": REPORT_MANIFEST_SCHEMA_VERSION,
                    },
                },
            )
            registry.transition_run(
                run_id,
                "running",
                start_time=utc_now(),
                artifact_path=artifact_path,
            )
        except BaseException as error:
            final_path: Path | None = None
            if archive is not None and archive.scratch_path.is_dir():
                try:
                    final_path = archive.finalize_failure(
                        error,
                        failure_category="artifact_finalization_failure",
                    )
                    if run_registered:
                        registry.record_artifacts(
                            run_id,
                            _artifact_records(final_path),
                        )
                except BaseException:
                    final_path = None
            if run_registered:
                current = registry.get_run(run_id)
                if current and current["status"] in {"pending", "running"}:
                    registry.transition_run(
                        run_id,
                        "failed",
                        end_time=utc_now(),
                        artifact_path=final_path or artifact_path,
                        failure_category="artifact_finalization_failure",
                    )
                registry.record_failure(
                    category="artifact_finalization_failure",
                    message=str(error),
                    run_id=run_id,
                    details={"operation": "prepare", "backup": backup.as_posix()},
                )
            raise
        return {
            "created": True,
            "run_id": run_id,
            "status": "running",
            "scientific_id": scientific_identifier,
            "repro_id": reproduction_identifier,
            "active_run_root": archive.scratch_path.as_posix(),
            "work_root": (archive.scratch_path / WORK_RELATIVE).as_posix(),
            "report_manifest": (
                archive.scratch_path / REPORT_MANIFEST_RELATIVE
            ).as_posix(),
            "artifact_path": archive.artifact_path.as_posix(),
            "registry_backup": backup.as_posix(),
        }


def _coverage_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    coverage = dict(_mapping(payload.get("coverage"), "manifest.coverage"))
    if set(coverage) != {"seeds", "cores", "mask_replicates"}:
        raise GradientAuditRunError("manifest.coverage has unexpected keys")
    if coverage != {
        "seeds": list(EXPECTED_SEEDS),
        "cores": list(EXPECTED_CORES),
        "mask_replicates": list(EXPECTED_MASK_REPLICATES),
    }:
        raise GradientAuditRunError(
            "manifest does not contain exact seed/core/mask coverage"
        )
    return coverage


def _validate_execution_commands(value: Any) -> list[list[str]]:
    commands = _list(value, "manifest.execution_commands")
    if len(commands) != 9:
        raise GradientAuditRunError(
            "manifest.execution_commands must contain pilot, seven seed "
            "shards, and aggregate"
        )
    normalized: list[list[str]] = []
    for index, raw in enumerate(commands):
        parts = _list(raw, f"manifest.execution_commands[{index}]")
        if not parts or any(not isinstance(item, str) or not item for item in parts):
            raise GradientAuditRunError(
                f"manifest.execution_commands[{index}] is not an argv list"
            )
        lowered = " ".join(parts).lower()
        if any(token in lowered for token in _FORBIDDEN_COMMAND_TOKENS):
            raise GradientAuditRunError(
                "manifest execution command may expose a secret-bearing option"
            )
        normalized.append([str(item) for item in parts])
    pilot = [command for command in normalized if "pilot" in command]
    aggregate = [command for command in normalized if "aggregate" in command]
    shard_seeds: list[int] = []
    for command in normalized:
        if "seed-shard" not in command:
            continue
        try:
            index = command.index("--seed")
            seed = int(command[index + 1])
        except (ValueError, IndexError) as exc:
            raise GradientAuditRunError(
                "seed-shard execution command omits a valid --seed"
            ) from exc
        shard_seeds.append(seed)
    if (
        len(pilot) != 1
        or len(aggregate) != 1
        or sorted(shard_seeds) != list(EXPECTED_SEEDS)
    ):
        raise GradientAuditRunError(
            "execution commands do not provide exact pilot/seed/aggregate coverage"
        )
    return normalized


def _validate_npz(path: Path) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            if not names:
                raise GradientAuditRunError(f"transient NPZ is empty: {path}")
            for name in names:
                normalized = name.lower().replace("-", "_")
                if any(token in normalized for token in _FORBIDDEN_NPZ_KEY_TOKENS):
                    raise GradientAuditRunError(
                        f"transient NPZ contains identifier-like key {name!r}"
                    )
                array = np.asarray(archive[name])
                if array.dtype.hasobject:
                    raise GradientAuditRunError(
                        f"transient NPZ contains object array {name!r}"
                    )
                if array.dtype.kind in {"S", "U", "V"}:
                    raise GradientAuditRunError(
                        f"transient NPZ contains string/opaque array {name!r}"
                    )
                if np.issubdtype(array.dtype, np.number) and not np.all(
                    np.isfinite(array)
                ):
                    raise GradientAuditRunError(
                        f"transient NPZ contains non-finite values in {name!r}"
                    )
    except GradientAuditRunError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise GradientAuditRunError(
            f"transient NPZ is unreadable: {path}"
        ) from exc


def _assert_no_direct_identifiers(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            if key in _FORBIDDEN_RETAINED_JSON_KEYS:
                raise GradientAuditRunError(
                    f"{label} contains prohibited direct-identifier key {raw_key!r}"
                )
            _assert_no_direct_identifiers(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_no_direct_identifiers(item, label=label)


def _nonempty_gate_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, Mapping)):
        return bool(value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _validate_gate_rows(value: Any) -> list[dict[str, Any]]:
    raw_rows = _list(value, "gradient audit report gate_rows")
    if not raw_rows:
        raise GradientAuditRunError("gradient audit gate_rows is empty")
    rows: list[dict[str, Any]] = []
    seen_scopes: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_rows):
        row = dict(_mapping(raw, f"gate_rows[{index}]"))
        gate = row.get("gate")
        scope = row.get("scope")
        passed = row.get("pass")
        if gate not in REQUIRED_GATE_FAMILIES:
            raise GradientAuditRunError(
                f"gate_rows[{index}] has unknown gate {gate!r}"
            )
        if not isinstance(scope, str) or not scope.strip():
            raise GradientAuditRunError(
                f"gate_rows[{index}].scope must be non-empty"
            )
        if not isinstance(passed, bool):
            raise GradientAuditRunError(
                f"gate_rows[{index}].pass must be boolean"
            )
        if not _nonempty_gate_value(row.get("threshold")):
            raise GradientAuditRunError(
                f"gate_rows[{index}].threshold must be non-empty"
            )
        if not _nonempty_gate_value(row.get("observed")):
            raise GradientAuditRunError(
                f"gate_rows[{index}].observed must be non-empty"
            )
        identity = (str(gate), scope)
        if identity in seen_scopes:
            raise GradientAuditRunError(
                "gradient audit has duplicate (gate, scope) rows"
            )
        seen_scopes.add(identity)
        rows.append(row)
    families = {str(row["gate"]) for row in rows}
    if families != REQUIRED_GATE_FAMILIES:
        missing = sorted(REQUIRED_GATE_FAMILIES - families)
        extra = sorted(families - REQUIRED_GATE_FAMILIES)
        raise GradientAuditRunError(
            f"gradient audit gate family mismatch; missing={missing}, extra={extra}"
        )
    counts = {
        gate: sum(int(row["gate"] == gate) for row in rows)
        for gate in REQUIRED_GATE_FAMILIES
    }
    if counts != EXPECTED_GATE_COUNTS or len(rows) != 40:
        raise GradientAuditRunError(
            f"gradient audit requires exact 40-row gate inventory; got {counts}"
        )
    scopes = {
        gate: frozenset(
            str(row["scope"]) for row in rows if row["gate"] == gate
        )
        for gate in REQUIRED_GATE_FAMILIES
    }
    if scopes != EXPECTED_GATE_SCOPES:
        raise GradientAuditRunError(
            "gradient audit gate scopes do not match the frozen 40-row inventory"
        )
    signed_pair_rows = [
        row for row in rows if row["gate"] == "signed_pair_stability"
    ]
    if all(
        isinstance(row.get("target"), str)
        and isinstance(row.get("source"), str)
        for row in signed_pair_rows
    ):
        signed_pairs = [
            (str(row["target"]), str(row["source"]))
            for row in signed_pair_rows
        ]
        if len(signed_pairs) != len(set(signed_pairs)):
            raise GradientAuditRunError(
                "signed_pair_stability contains duplicate directed pairs"
            )
        if set(signed_pairs) != EXPECTED_LOCKED_PAIRS:
            raise GradientAuditRunError(
                "signed_pair_stability does not cover the exact 19 locked pairs"
            )
    return rows


def _transient_core(
    *,
    coverage: Mapping[str, Any],
    pilot: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
        "coverage": dict(coverage),
        "pilot": dict(pilot),
        "shards": [dict(shard) for shard in shards],
    }


def _load_cleanup_receipt(active_root: Path) -> dict[str, Any] | None:
    path = active_root / TRANSIENT_CLEANUP_RECEIPT_RELATIVE
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise GradientAuditRunError("transient cleanup receipt is unsafe")
    payload = _signed_payload(
        _mapping(_strict_json(path, label="transient cleanup receipt"), "receipt"),
        label="transient cleanup receipt",
    )
    if (
        payload.get("kind") != TRANSIENT_RECEIPT_KIND
        or payload.get("schema_version") != 1
        or payload.get("campaign_id") != CAMPAIGN_ID
    ):
        raise GradientAuditRunError("transient cleanup receipt identity mismatch")
    return payload


def _validate_report_manifest(
    active_root: Path,
    *,
    allow_cleaned_transients: bool,
) -> dict[str, Any]:
    manifest_path = active_root / REPORT_MANIFEST_RELATIVE
    manifest = _signed_payload(
        _mapping(
            _strict_json(manifest_path, label="gradient audit report manifest"),
            "gradient audit report manifest",
        ),
        label="gradient audit report manifest",
    )
    expected_keys = {
        "kind",
        "schema_version",
        "campaign_id",
        "frozen_contract_sha256",
        "implementation_protocol_sha256",
        "analysis_input_sha256",
        "coverage",
        "pilot",
        "shards",
        "files",
        "execution_commands",
        "checksum",
    }
    if set(manifest) != expected_keys:
        raise GradientAuditRunError(
            "gradient audit report manifest has unexpected top-level keys"
        )
    if (
        manifest.get("kind") != REPORT_MANIFEST_KIND
        or manifest.get("schema_version") != REPORT_MANIFEST_SCHEMA_VERSION
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("frozen_contract_sha256")
        != FROZEN_CONTRACT_SHA256
        or manifest.get("implementation_protocol_sha256")
        != IMPLEMENTATION_PROTOCOL_SHA256
    ):
        raise GradientAuditRunError("report manifest frozen identity mismatch")
    _digest(manifest.get("analysis_input_sha256"), "analysis_input_sha256")
    coverage = _coverage_payload(manifest)
    _validate_execution_commands(manifest.get("execution_commands"))

    work_root = active_root / WORK_RELATIVE
    pilot, pilot_path = _verify_file_descriptor(
        manifest.get("pilot"),
        base=work_root,
        label="manifest.pilot",
        expected_path="pilot/resource_pilot.json",
    )
    pilot_payload = _strict_json(
        pilot_path,
        label="retained resource pilot",
    )
    _assert_no_direct_identifiers(
        pilot_payload,
        label="retained resource pilot",
    )
    pilot_scientific_input = _digest(
        _mapping(pilot_payload, "retained resource pilot").get(
            "analysis_input_sha256"
        ),
        "retained resource pilot analysis_input_sha256",
    )
    shards_raw = _list(manifest.get("shards"), "manifest.shards")
    if len(shards_raw) != len(EXPECTED_SEEDS):
        raise GradientAuditRunError("manifest requires exactly seven seed shards")
    cleanup = _load_cleanup_receipt(active_root) if allow_cleaned_transients else None
    shards: list[dict[str, Any]] = []
    transient_paths: list[Path] = []
    scientific_inputs = {pilot_scientific_input}
    for expected_seed, raw_shard in zip(
        EXPECTED_SEEDS,
        shards_raw,
        strict=True,
    ):
        shard = dict(_mapping(raw_shard, f"manifest shard {expected_seed}"))
        if set(shard) != {"seed", "metadata", "arrays"}:
            raise GradientAuditRunError(
                f"manifest shard {expected_seed} has unexpected keys"
            )
        if shard.get("seed") != expected_seed:
            raise GradientAuditRunError("manifest shard order/seed mismatch")
        directory = f"shards/seed-{expected_seed:02d}"
        metadata, metadata_path = _verify_file_descriptor(
            shard.get("metadata"),
            base=work_root,
            label=f"manifest shard {expected_seed} metadata",
            expected_path=f"{directory}/seed-{expected_seed:02d}.metadata.json",
        )
        metadata_payload = _strict_json(
            metadata_path,
            label=f"seed {expected_seed} retained metadata",
        )
        _assert_no_direct_identifiers(
            metadata_payload,
            label=f"seed {expected_seed} retained metadata",
        )
        scientific_inputs.add(
            _digest(
                _mapping(
                    metadata_payload,
                    f"seed {expected_seed} retained metadata",
                ).get("analysis_input_sha256"),
                f"seed {expected_seed} analysis_input_sha256",
            )
        )
        metadata_sidecar = _verify_sidecar(metadata_path, metadata)
        arrays_raw = dict(
            _mapping(
                shard.get("arrays"),
                f"manifest shard {expected_seed} arrays",
            )
        )
        arrays_expected = {
            "path",
            "sha256",
            "size_bytes",
            "retention",
        }
        if set(arrays_raw) != arrays_expected:
            raise GradientAuditRunError(
                f"manifest shard {expected_seed} arrays descriptor drifted"
            )
        if (
            arrays_raw.get("retention")
            != "transient_deleted_after_verified_aggregation"
        ):
            raise GradientAuditRunError(
                "cell-level shard arrays must be explicitly transient"
            )
        arrays_relative = (
            f"{directory}/seed-{expected_seed:02d}.arrays.npz"
        )
        arrays_path = work_root / arrays_relative
        arrays_sidecar = arrays_path.with_name(
            f"{arrays_path.name}.sha256.json"
        )
        if arrays_path.is_file() and not arrays_path.is_symlink():
            arrays, arrays_path = _verify_file_descriptor(
                arrays_raw,
                base=work_root,
                label=f"manifest shard {expected_seed} arrays",
                expected_path=arrays_relative,
                expected_keys=arrays_expected,
            )
            arrays_sidecar = _verify_sidecar(arrays_path, arrays)
            _validate_npz(arrays_path)
            transient_paths.extend([arrays_path, arrays_sidecar])
        elif cleanup is None:
            raise GradientAuditRunError(
                f"manifest shard {expected_seed} arrays are missing"
            )
        shards.append(
            {
                "seed": expected_seed,
                "metadata": metadata,
                "arrays": arrays_raw,
            }
        )
        # Sidecars and metadata remain as compact provenance.
        _ = metadata_sidecar

    if len(scientific_inputs) != 1:
        raise GradientAuditRunError(
            "pilot and seed shards disagree on scientific input identity"
        )
    scientific_input_sha256 = next(iter(scientific_inputs))

    transient_payload = _transient_core(
        coverage=coverage,
        pilot=pilot,
        shards=shards,
    )
    expected_input = canonical_sha256(transient_payload)
    if not hmac.compare_digest(
        expected_input,
        str(manifest["analysis_input_sha256"]),
    ):
        raise GradientAuditRunError(
            "analysis_input_sha256 does not bind pilot/shard inputs"
        )
    if cleanup is not None:
        if cleanup.get("analysis_input_sha256") != expected_input:
            raise GradientAuditRunError(
                "transient cleanup receipt does not bind analysis inputs"
            )
        declared = cleanup.get("deleted_files")
        expected_deleted = sorted(
            [
                f"diagnostics/audit_work/shards/seed-{seed:02d}/"
                f"seed-{seed:02d}.arrays.npz"
                for seed in EXPECTED_SEEDS
            ]
            + [
                f"diagnostics/audit_work/shards/seed-{seed:02d}/"
                f"seed-{seed:02d}.arrays.npz.sha256.json"
                for seed in EXPECTED_SEEDS
            ]
        )
        if declared != expected_deleted:
            raise GradientAuditRunError(
                "transient cleanup receipt inventory mismatch"
            )

    report_dir = manifest_path.parent
    files_raw = _list(manifest.get("files"), "manifest.files")
    if len(files_raw) != len(REQUIRED_REPORT_FILES):
        raise GradientAuditRunError("manifest report file inventory is incomplete")
    report_files: dict[str, Path] = {}
    seen_roles: set[str] = set()
    for index, raw in enumerate(files_raw):
        entry = dict(_mapping(raw, f"manifest.files[{index}]"))
        if set(entry) != {"role", "path", "sha256", "size_bytes"}:
            raise GradientAuditRunError("manifest report file entry drifted")
        role = entry.get("role")
        if not isinstance(role, str) or role in seen_roles:
            raise GradientAuditRunError("manifest report roles are invalid")
        expected_name = REQUIRED_REPORT_FILES.get(role)
        if expected_name is None:
            raise GradientAuditRunError(f"unexpected report role {role!r}")
        descriptor, path = _verify_file_descriptor(
            {key: value for key, value in entry.items() if key != "role"},
            base=report_dir,
            label=f"manifest report role {role}",
            expected_path=expected_name,
        )
        report_files[role] = path
        seen_roles.add(role)
        _ = descriptor
    if seen_roles != set(REQUIRED_REPORT_FILES):
        raise GradientAuditRunError("manifest report roles are incomplete")
    actual_report_files = {
        path.name
        for path in report_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    expected_report_files = {
        "manifest.json",
        *REQUIRED_REPORT_FILES.values(),
    }
    if actual_report_files != expected_report_files or any(
        path.is_symlink() for path in report_dir.iterdir()
    ):
        raise GradientAuditRunError(
            "aggregate report directory and manifest inventory differ"
        )
    report = _mapping(
        _strict_json(
            report_files["audit_report_json"],
            label="gradient audit report",
        ),
        "gradient audit report",
    )
    _assert_no_direct_identifiers(report, label="gradient audit report")
    if report.get("analysis_status") != "complete":
        raise GradientAuditRunError(
            "gradient audit report is not complete; pending gate reduction "
            "cannot be finalized"
        )
    if (
        report.get("campaign_id") != CAMPAIGN_ID
        or report.get("frozen_contract_sha256")
        != FROZEN_CONTRACT_SHA256
        or report.get("implementation_protocol_sha256")
        != IMPLEMENTATION_PROTOCOL_SHA256
        or report.get("analysis_input_sha256")
        != manifest["analysis_input_sha256"]
        or report.get("scientific_input_sha256")
        != scientific_input_sha256
        or report.get("coverage") != coverage
    ):
        raise GradientAuditRunError(
            "gradient audit report identity or coverage disagrees with inputs"
        )
    gate_results = _validate_gate_rows(report.get("gate_rows"))
    final_metrics = _mapping(
        report.get("final_metrics"),
        "gradient audit report final_metrics",
    )
    if PRIMARY_METRIC not in final_metrics:
        raise GradientAuditRunError(
            f"gradient audit report omits {PRIMARY_METRIC}"
        )
    normalized_metrics: dict[str, float] = {}
    for name, value in final_metrics.items():
        if not isinstance(name, str) or "/" not in name:
            raise GradientAuditRunError(
                "gradient audit final metrics must be namespaced"
            )
        normalized_metrics[name] = _finite(
            value,
            f"gradient audit metric {name}",
        )
    if not 0.0 <= normalized_metrics[PRIMARY_METRIC] <= 1.0:
        raise GradientAuditRunError(
            f"{PRIMARY_METRIC} must be in [0, 1]"
        )
    expected_gate_fraction = sum(
        int(bool(row["pass"])) for row in gate_results
    ) / len(gate_results)
    if not math.isclose(
        normalized_metrics[PRIMARY_METRIC],
        expected_gate_fraction,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise GradientAuditRunError(
            f"{PRIMARY_METRIC} must equal the mean boolean outcome over all "
            "emitted gate rows"
        )
    precursors_supported = report.get(
        "candidate_set_computational_precursors_supported"
    )
    if not isinstance(precursors_supported, bool):
        raise GradientAuditRunError(
            "candidate_set_computational_precursors_supported must be boolean"
        )
    if report.get("mechanism_validation_available") is not False:
        raise GradientAuditRunError(
            "report must state mechanism_validation_available=false"
        )
    if report.get("mechanism_claim_supported") is not False:
        raise GradientAuditRunError(
            "report must state mechanism_claim_supported=false"
        )
    maximum_claim = report.get("maximum_defensible_claim")
    if maximum_claim not in MAXIMUM_DEFENSIBLE_CLAIMS:
        raise GradientAuditRunError(
            "report maximum_defensible_claim is outside the frozen claim enum"
        )
    expected_maximum = (
        "candidate_set_contains_stable_faithful_graph_dependent_"
        "null_calibrated_model_implied_predictive_sensitivities"
        if precursors_supported
        else "no_claim_beyond_reported_model_behavior"
    )
    if maximum_claim != expected_maximum:
        raise GradientAuditRunError(
            "maximum_defensible_claim disagrees with computational precursor "
            "qualification"
        )
    verdict = report.get("claim_verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        raise GradientAuditRunError("report claim_verdict must be non-empty")
    normalized_verdict = verdict.strip().lower()
    if (
        "mechanism" not in normalized_verdict
        or not any(
            token in normalized_verdict
            for token in ("not_supported", "not_validated", "unavailable")
        )
    ):
        raise GradientAuditRunError(
            "claim_verdict must explicitly reject biological-mechanism support"
        )
    with report_files["gate_results_csv"].open(
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise GradientAuditRunError("gate_results.csv has no header")
        forbidden_headers = sorted(
            field
            for field in reader.fieldnames
            if field.strip().lower().replace("-", "_")
            in _FORBIDDEN_RETAINED_JSON_KEYS
        )
        if forbidden_headers:
            raise GradientAuditRunError(
                "gate_results.csv contains direct-identifier columns: "
                + ", ".join(forbidden_headers)
            )
        required_csv_columns = {
            "gate",
            "scope",
            "threshold",
            "observed",
            "pass",
        }
        if not required_csv_columns.issubset(reader.fieldnames):
            raise GradientAuditRunError(
                "gate_results.csv does not mirror the gate-row schema"
            )
        csv_rows = list(reader)
    if len(csv_rows) != len(gate_results):
        raise GradientAuditRunError(
            "gate_results.csv row count differs from report gate_rows"
        )

    def csv_value(value: str) -> Any:
        stripped = value.strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

    report_by_scope = {
        (str(row["gate"]), str(row["scope"])): row for row in gate_results
    }
    csv_scopes: set[tuple[str, str]] = set()
    for index, row in enumerate(csv_rows):
        identity = (str(row["gate"]), str(row["scope"]))
        if identity in csv_scopes or identity not in report_by_scope:
            raise GradientAuditRunError(
                f"gate_results.csv row {index} has unknown or duplicate scope"
            )
        csv_scopes.add(identity)
        reported = report_by_scope[identity]
        pass_text = str(row["pass"]).strip().lower()
        if pass_text not in {"true", "false"}:
            raise GradientAuditRunError(
                f"gate_results.csv row {index} pass is not boolean"
            )
        if (pass_text == "true") is not bool(reported["pass"]):
            raise GradientAuditRunError(
                f"gate_results.csv row {index} pass disagrees with report"
            )
        if canonical_json(csv_value(str(row["threshold"]))) != canonical_json(
            reported["threshold"]
        ) or canonical_json(csv_value(str(row["observed"]))) != canonical_json(
            reported["observed"]
        ):
            raise GradientAuditRunError(
                f"gate_results.csv row {index} values disagree with report"
            )
    if csv_scopes != set(report_by_scope):
        raise GradientAuditRunError(
            "gate_results.csv scope inventory differs from report gate_rows"
        )

    prediction_rows: list[dict[str, Any]] = []
    prediction_path = report_files["canonical_fit_predictions_jsonl"]
    with prediction_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = dict(
                _strict_json_line(
                    line,
                    label=f"canonical prediction line {line_number}",
                )
            )
            source_run_id = row.get("run_id")
            if source_run_id not in {active_root.name, "pending-registration"}:
                raise GradientAuditRunError(
                    "canonical prediction template has an unexpected run_id"
                )
            row["run_id"] = active_root.name
            prediction_rows.append(row)
    run_id = active_root.name
    try:
        validate_prediction_rows(
            prediction_rows,
            expected_run_id=run_id,
            expected_split="fit",
        )
    except RunValidationError as exc:
        raise GradientAuditRunError(
            "canonical fit prediction rows violate the run contract"
        ) from exc
    if any(row.get("dataset_id") != DATASET_ID for row in prediction_rows):
        raise GradientAuditRunError(
            "canonical predictions use an unexpected dataset_id"
        )
    expected_sample_keys = {
        f"{core}/{target}/mask-{replicate}"
        for core in EXPECTED_CORES
        for replicate in EXPECTED_MASK_REPLICATES
        for target in EXPECTED_TARGETS
    }
    observed_sample_keys = [
        str(row["sample_key"]) for row in prediction_rows
    ]
    if (
        len(observed_sample_keys) != len(expected_sample_keys)
        or len(set(observed_sample_keys)) != len(observed_sample_keys)
        or set(observed_sample_keys) != expected_sample_keys
    ):
        raise GradientAuditRunError(
            "canonical fit predictions do not contain exact "
            "core/target/mask aggregate coverage"
        )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "work_root": work_root,
        "coverage": coverage,
        "pilot": pilot,
        "shards": shards,
        "transient_paths": transient_paths,
        "report_files": report_files,
        "report": dict(report),
        "gate_results": gate_results,
        "final_metrics": normalized_metrics,
        "prediction_rows": prediction_rows,
    }


def _ensure_bytes(archive: RunArchive, relative: Path, content: bytes) -> Path:
    path = archive.scratch_path / relative
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise GradientAuditRunError(
                f"existing derived file differs: {relative.as_posix()}"
            )
        return path
    return archive.write_bytes(relative, content)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            json.loads(canonical_json(value)),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _ensure_json(archive: RunArchive, relative: Path, value: Any) -> Path:
    return _ensure_bytes(archive, relative, _json_bytes(value))


def _ensure_copy(
    archive: RunArchive,
    source: Path,
    relative: Path,
) -> Path:
    return _ensure_bytes(archive, relative, source.read_bytes())


def _cleanup_transients(
    archive: RunArchive,
    validated: Mapping[str, Any],
) -> None:
    active_root = archive.scratch_path
    existing = _load_cleanup_receipt(active_root)
    deleted_relative = sorted(
        [
            f"diagnostics/audit_work/shards/seed-{seed:02d}/"
            f"seed-{seed:02d}.arrays.npz"
            for seed in EXPECTED_SEEDS
        ]
        + [
            f"diagnostics/audit_work/shards/seed-{seed:02d}/"
            f"seed-{seed:02d}.arrays.npz.sha256.json"
            for seed in EXPECTED_SEEDS
        ]
    )
    if existing is None:
        core = {
            "kind": TRANSIENT_RECEIPT_KIND,
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "analysis_input_sha256": validated["manifest"][
                "analysis_input_sha256"
            ],
            "deleted_files": deleted_relative,
            "reason": (
                "verified cell-level ensemble inputs are excluded from the "
                "immutable report bundle by the frozen contract"
            ),
        }
        receipt = {**core, "checksum": canonical_sha256(core)}
        _ensure_json(archive, TRANSIENT_CLEANUP_RECEIPT_RELATIVE, receipt)
    else:
        if (
            existing.get("analysis_input_sha256")
            != validated["manifest"]["analysis_input_sha256"]
            or existing.get("deleted_files") != deleted_relative
        ):
            raise GradientAuditRunError("cleanup receipt does not match manifest")
    for relative_text in deleted_relative:
        path = active_root / Path(relative_text)
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise GradientAuditRunError(
                f"refusing to delete unsafe transient path: {path}"
            )
        path.unlink()
    remaining = [
        path.relative_to(active_root).as_posix()
        for path in (active_root / WORK_RELATIVE / "shards").rglob("*.npz")
    ]
    if remaining:
        raise GradientAuditRunError(
            f"transient NPZ files remain after cleanup: {remaining}"
        )


def _prepare_canonical_contents(
    archive: RunArchive,
    run: Mapping[str, Any],
    validated: Mapping[str, Any],
) -> None:
    _cleanup_transients(archive, validated)
    # Re-read the signed manifest after transient removal using its signed
    # cleanup receipt as the durable proof that all temporary arrays verified.
    validated_after = _validate_report_manifest(
        archive.scratch_path,
        allow_cleaned_transients=True,
    )
    report_files = _mapping(validated_after["report_files"], "report files")
    for role, filename in REQUIRED_REPORT_FILES.items():
        if role == "canonical_fit_predictions_jsonl":
            continue
        _ensure_copy(
            archive,
            Path(str(report_files[role])),
            Path("interpretation") / filename,
        )
    _ensure_copy(
        archive,
        Path(str(validated_after["manifest_path"])),
        Path("provenance/audit_report_manifest.json"),
    )
    _ensure_json(
        archive,
        Path("provenance/audit_execution_commands.json"),
        {
            "commands": validated_after["manifest"]["execution_commands"],
            "working_directory": archive.paths.project_root.as_posix(),
        },
    )
    predictions = [
        canonical_json(row)
        for row in validated_after["prediction_rows"]
    ]
    _ensure_bytes(
        archive,
        Path("predictions/fit.jsonl"),
        ("\n".join(predictions) + "\n").encode("utf-8"),
    )
    metrics = dict(validated_after["final_metrics"])
    _ensure_json(archive, Path("metrics/final.json"), metrics)
    event_lines = [
        canonical_json(
            {
                "name": name,
                "value": value,
                "step": 0,
                "split": "fit",
            }
        )
        for name, value in sorted(metrics.items())
    ]
    _ensure_bytes(
        archive,
        Path("metrics/events.jsonl"),
        ("\n".join(event_lines) + "\n").encode("utf-8"),
    )
    history_lines = [
        canonical_json(
            {
                "name": name,
                "value": value,
                "step": 0,
                "split": "fit",
            }
        )
        for name, value in sorted(metrics.items())
    ]
    _ensure_bytes(
        archive,
        Path("metrics/history.jsonl"),
        ("\n".join(history_lines) + "\n").encode("utf-8"),
    )
    state = {
        "kind": "myjju_gradient_audit_analysis_state",
        "schema_version": 1,
        "run_id": archive.run_id,
        "campaign_id": CAMPAIGN_ID,
        "model_weights": False,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": IMPLEMENTATION_PROTOCOL_SHA256,
        "report_manifest_checksum": validated_after["manifest"]["checksum"],
        "analysis_input_sha256": validated_after["manifest"][
            "analysis_input_sha256"
        ],
        "coverage": validated_after["coverage"],
        "final_metrics": metrics,
        "gate_row_pass_fraction_role": (
            "registry_bookkeeping_only_not_an_evidence_score_or_claim"
        ),
        "mechanism_validation_available": False,
    }
    _ensure_json(archive, Path("checkpoints/last.ckpt"), state)
    report = _mapping(validated_after["report"], "gradient audit report")
    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "campaign_id": CAMPAIGN_ID,
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_value": metrics[PRIMARY_METRIC],
        "primary_metric_role": (
            "registry_bookkeeping_only_not_an_evidence_score_or_claim"
        ),
        "metrics": metrics,
        "claim_verdict": report["claim_verdict"],
        "mechanism_validation_available": False,
        "experimental_unit": "adjacent_normal_spatial_core",
        "core_count": len(EXPECTED_CORES),
        "model_seed_count": len(EXPECTED_SEEDS),
        "mask_replicate_count": len(EXPECTED_MASK_REPLICATES),
        "held_in_transductive": True,
        "patient_generalization_supported": False,
        "parameter_count_per_upstream_model": MODEL_PARAMETER_COUNT,
        "analysis_state_checkpoint_is_model_weights": False,
    }
    _ensure_json(archive, Path("summary.json"), summary)
    manifest = {
        "run_id": archive.run_id,
        "status": "completed",
        "campaign_id": CAMPAIGN_ID,
        "scientific_id": run["scientific_id"],
        "repro_id": run["repro_id"],
        "schema_version": 1,
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_value": metrics[PRIMARY_METRIC],
        "primary_metric_role": (
            "registry_bookkeeping_only_not_an_evidence_score_or_claim"
        ),
        "lifecycle_status_source": "registry_and_completion_marker",
        "artifact_roles": {
            "primary_checkpoint": "last",
            "primary_checkpoint_semantics": (
                "analysis_state_only_not_model_weights"
            ),
            "canonical_predictions": "fit",
            "audit_report": "interpretation/report.json",
        },
        "claim_boundary": {
            "exploratory": True,
            "held_in": True,
            "mechanism_validation_available": False,
            "maximum_claim": (
                "model-implied predictive sensitivity subject to the "
                "reported computational gates"
            ),
        },
    }
    content = yaml.safe_dump(
        json.loads(canonical_json(manifest)),
        sort_keys=True,
        allow_unicode=False,
    ).encode("utf-8")
    _ensure_bytes(archive, Path("manifest.yaml"), content)


def _duration_seconds(run: Mapping[str, Any]) -> float:
    start = run.get("start_time")
    if not isinstance(start, str):
        return 0.0
    try:
        parsed = datetime.fromisoformat(start.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(
        0.0,
        (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds(),
    )


def _begin_registry_finalization(
    registry: Registry,
    *,
    run_id: str,
    artifact_path: Path,
    metrics: Mapping[str, float],
    duration_seconds: float,
    records: Sequence[Mapping[str, Any]],
) -> None:
    finished = utc_now()
    with registry.transaction(immediate=True) as connection:
        run = connection.execute(
            "SELECT status FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if run is None or str(run["status"]) != "running":
            raise GradientAuditRunError(
                f"run {run_id} is not running at finalization commit"
            )
        existing = connection.execute(
            "SELECT COUNT(*) AS count FROM artifacts WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if existing is None or int(existing["count"]) != 0:
            raise GradientAuditRunError(
                "running aggregate run already has registered artifacts"
            )
        connection.execute(
            """
            UPDATE runs
            SET status = 'finalizing', end_time = ?, duration_seconds = ?,
                primary_metric_name = ?, primary_metric_value = ?,
                parameter_count = ?, artifact_path = ?, updated_at = ?
            WHERE run_id = ? AND status = 'running'
            """,
            (
                finished,
                float(duration_seconds),
                PRIMARY_METRIC,
                float(metrics[PRIMARY_METRIC]),
                MODEL_PARAMETER_COUNT,
                artifact_path.as_posix(),
                finished,
                run_id,
            ),
        )
        connection.executemany(
            """
            INSERT INTO metrics(
                run_id, evaluation_id, name, value, step, split, recorded_at
            ) VALUES (?, NULL, ?, ?, 0, 'fit', ?)
            """,
            [
                (run_id, name, float(value), finished)
                for name, value in sorted(metrics.items())
            ],
        )
        connection.executemany(
            """
            INSERT INTO artifacts(
                run_id, evaluation_id, kind, path, sha256, size_bytes,
                status, created_at
            ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    str(record["kind"]),
                    str(record["path"]),
                    record["sha256"],
                    int(record["size_bytes"]),
                    str(record.get("status", "present")),
                    finished,
                )
                for record in records
            ],
        )


def _registered_artifacts_match(
    registry: Registry,
    *,
    run_id: str,
    root: Path,
) -> bool:
    expected = {
        str(record["path"]): (
            str(record["kind"]),
            str(record["sha256"]),
            int(record["size_bytes"]),
            str(record["status"]),
        )
        for record in _artifact_records(root)
    }
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT kind, path, sha256, size_bytes, status
            FROM artifacts WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    actual = {
        str(row["path"]): (
            str(row["kind"]),
            str(row["sha256"]),
            int(row["size_bytes"]),
            str(row["status"]),
        )
        for row in rows
    }
    return actual == expected


def _register_analysis_state_checkpoint(
    registry: Registry,
    *,
    run_id: str,
    artifact_path: Path,
    primary_value: float,
    update: bool = False,
) -> None:
    checkpoint_path = artifact_path / "checkpoints/last.ckpt"
    with registry.connect() as connection:
        rows = connection.execute(
            """
            SELECT artifact_id FROM artifacts
            WHERE run_id = ? AND path = ? AND kind = 'checkpoints'
            """,
            (run_id, checkpoint_path.as_posix()),
        ).fetchall()
    if len(rows) != 1:
        raise GradientAuditRunError(
            "analysis-state checkpoint artifact is missing or ambiguous"
        )
    metadata = {
        "schema_version": 1,
        "artifact_semantics": "analysis_state_only_not_model_weights",
        "model_weights": False,
        "monitored_metric_role": (
            "registry_bookkeeping_only_not_an_evidence_score_or_claim"
        ),
        "campaign_id": CAMPAIGN_ID,
        "frozen_contract_sha256": FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": (
            IMPLEMENTATION_PROTOCOL_SHA256
        ),
        "mechanism_validation_available": False,
    }
    if update:
        with registry.connect() as connection:
            existing = connection.execute(
                """
                SELECT metadata_json FROM checkpoint_catalog
                WHERE artifact_id = ?
                """,
                (int(rows[0]["artifact_id"]),),
            ).fetchone()
        if existing is not None:
            decoded = json.loads(str(existing["metadata_json"]))
            metadata = {**dict(_mapping(decoded, "checkpoint metadata")), **metadata}
    registry.register_checkpoint_metadata(
        int(rows[0]["artifact_id"]),
        run_id=run_id,
        role="last",
        best_epoch=None,
        monitored_metric=PRIMARY_METRIC,
        monitored_mode="max",
        monitored_value=primary_value,
        retention_class="retain_exploratory_evidence",
        verification_status="verified",
        metadata=metadata,
        update=update,
    )


def _complete_registry_run(
    registry: Registry,
    *,
    run_id: str,
    artifact_path: Path,
) -> None:
    verify_run_bundle(artifact_path)
    registry.transition_run(run_id, "completed")
    issues = registry.verify_artifacts(run_id=run_id)
    if issues:
        raise GradientAuditRunError(
            f"registered artifact verification failed: {issues[:5]}"
        )


def finalize_run(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_reference: str,
) -> dict[str, Any]:
    """Verify, publish, register, index, and complete one aggregate run."""

    paths.validate()
    _verify_frozen_inputs(paths)
    _validate_registered_campaign(registry)
    with _registry_lock(paths):
        run_id = registry.resolve_run_id(run_reference)
        run = registry.get_run(run_id)
        if run is None or run.get("campaign_id") != CAMPAIGN_ID:
            raise GradientAuditRunError(
                f"run does not belong to {CAMPAIGN_ID}: {run_id}"
            )
        category = registry.get_run_category(run_id)
        if (
            category is None
            or category.get("study_axis") != "gradient_attribution_audit"
            or category.get("lifecycle_stage") != "exploratory_screen"
            or category.get("seed_known") is not False
            or category.get("fold_known") is not False
            or category.get("attempt_known") is not True
        ):
            raise GradientAuditRunError(
                "registered aggregate run semantics are missing or incorrect"
            )
        artifact_path = RunArchive.artifact_path_for(run_id, paths)
        status = str(run["status"])
        if status == "completed":
            verification = verify_run_bundle(artifact_path)
            if registry.verify_artifacts(run_id=run_id):
                raise GradientAuditRunError(
                    "completed run has failing registered artifacts"
                )
            checkpoints = registry.list_checkpoint_catalog(
                run_id=run_id,
                role="last",
                verification_status="verified",
            )
            if len(checkpoints) != 1:
                raise GradientAuditRunError(
                    "completed run lacks one verified analysis-state checkpoint"
                )
            return {
                "completed": True,
                "idempotent": True,
                "run_id": run_id,
                "artifact_path": artifact_path.as_posix(),
                "verification": verification,
            }
        if status not in {"running", "finalizing"}:
            raise GradientAuditRunError(
                f"run {run_id} cannot finalize from status {status!r}"
            )

        active_path = paths.scratch_root / "active_runs" / run_id
        marker_names = [
            marker for marker in COMPLETION_MARKERS if (artifact_path / marker).exists()
        ] if artifact_path.is_dir() else []
        if status == "running" and active_path.is_dir():
            archive = RunArchive.attach_active(run_id, paths=paths)
            validated = _validate_report_manifest(
                active_path,
                allow_cleaned_transients=True,
            )
            _prepare_canonical_contents(archive, run, validated)
            validated = _validate_report_manifest(
                active_path,
                allow_cleaned_transients=True,
            )
            backup = _backup_registry(
                registry,
                paths,
                operation="finalize",
            )
            published = archive.publish_success_pending()
            verify_unmarked_run_bundle(published)
            records = _artifact_records(published)
            _begin_registry_finalization(
                registry,
                run_id=run_id,
                artifact_path=published,
                metrics=validated["final_metrics"],
                duration_seconds=_duration_seconds(run),
                records=records,
            )
            status = "finalizing"
            artifact_path = published
        elif status == "running" and artifact_path.is_dir() and not marker_names:
            # Crash after publish and before the registry transaction.
            archive = RunArchive.from_published(run_id, paths=paths)
            verify_unmarked_run_bundle(artifact_path)
            report = _mapping(
                _strict_json(
                    artifact_path / "interpretation/report.json",
                    label="published gradient audit report",
                ),
                "published gradient audit report",
            )
            metrics = {
                str(name): _finite(value, f"published metric {name}")
                for name, value in _mapping(
                    report.get("final_metrics"),
                    "published final metrics",
                ).items()
            }
            if PRIMARY_METRIC not in metrics:
                raise GradientAuditRunError("published report omits primary metric")
            backup = _backup_registry(
                registry,
                paths,
                operation="finalize-recovery",
            )
            _begin_registry_finalization(
                registry,
                run_id=run_id,
                artifact_path=artifact_path,
                metrics=metrics,
                duration_seconds=_duration_seconds(run),
                records=_artifact_records(artifact_path),
            )
            status = "finalizing"
        elif status == "running":
            raise GradientAuditRunError(
                "running aggregate has neither owned scratch nor recoverable "
                "published content"
            )
        else:
            backup = _backup_registry(
                registry,
                paths,
                operation="finalize-recovery",
            )
            archive = RunArchive.from_published(run_id, paths=paths)

        latest = registry.get_run(run_id)
        assert latest is not None
        if str(latest["status"]) != "finalizing":
            raise GradientAuditRunError(
                "aggregate run did not enter finalizing state"
            )
        if not _registered_artifacts_match(
            registry,
            run_id=run_id,
            root=artifact_path,
        ):
            raise GradientAuditRunError(
                "registered finalizing artifacts differ from published bundle"
            )
        summary = _mapping(
            _strict_json(artifact_path / "summary.json", label="run summary"),
            "run summary",
        )
        primary_value = _finite(
            summary.get("primary_metric_value"),
            "run summary primary metric",
        )
        _register_analysis_state_checkpoint(
            registry,
            run_id=run_id,
            artifact_path=artifact_path,
            primary_value=primary_value,
        )
        index_checkpoint_catalog(
            registry,
            paths,
            run_reference=run_id,
            verify=True,
        )
        # The generic index adds searchable duplicate/catalog metadata. Merge
        # back the campaign-specific analysis-state boundary afterwards so the
        # indexed ``.ckpt`` can never be mistaken for model weights.
        _register_analysis_state_checkpoint(
            registry,
            run_id=run_id,
            artifact_path=artifact_path,
            primary_value=primary_value,
            update=True,
        )
        markers = [
            marker for marker in COMPLETION_MARKERS if (artifact_path / marker).exists()
        ]
        if not markers:
            verify_unmarked_run_bundle(artifact_path)
            archive.mark_success()
        elif markers != ["_SUCCESS"]:
            raise GradientAuditRunError(
                f"finalizing bundle has incompatible markers: {markers}"
            )
        verification = verify_run_bundle(artifact_path)
        _complete_registry_run(
            registry,
            run_id=run_id,
            artifact_path=artifact_path,
        )
        final = registry.get_run(run_id)
        assert final is not None
        return {
            "completed": True,
            "idempotent": False,
            "run_id": run_id,
            "scientific_id": str(final["scientific_id"]),
            "repro_id": str(final["repro_id"]),
            "primary_metric_name": PRIMARY_METRIC,
            "primary_metric_value": float(final["primary_metric_value"]),
            "artifact_path": artifact_path.as_posix(),
            "registry_backup": backup.as_posix(),
            "verification": verification,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Registry database (default: BAGM state/tracking/bagm.sqlite3).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="Register the sole aggregate run and create owned active scratch.",
    )
    prepare.add_argument("--attempt", type=int, default=1)
    finalize = subparsers.add_parser(
        "finalize",
        help="Verify and atomically finalize the aggregate bundle.",
    )
    finalize.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    paths = current_paths()
    database = args.database or (
        paths.state_root / "tracking/bagm.sqlite3"
    )
    if not database.is_absolute():
        database = paths.project_root / database
    registry = Registry(database)
    if args.command == "prepare":
        result = prepare_run(
            registry=registry,
            paths=paths,
            attempt=int(args.attempt),
            invocation=[sys.executable, str(Path(__file__).resolve()), *arguments],
        )
    else:
        result = finalize_run(
            registry=registry,
            paths=paths,
            run_reference=str(args.run_id),
        )
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
