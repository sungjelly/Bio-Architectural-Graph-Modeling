#!/usr/bin/env python3
"""Fail-closed wrapper for one same-gene robustness variant/seed/fold.

This module deliberately reuses :mod:`run_same_gene_nonlinear` after verifying
all launch, source, contract, prepared-variant, and (for production) pilot
identities.  No numerical prepared array is loaded before those checks pass.

The launch manifest is strict JSON with exactly these top-level keys::

    {
      "campaign_id": "cmp_20260810_same_gene_robustness_multiverse_v1",
      "contract": {"path": "...", "sha": "<lowercase sha256>"},
      "sources": [{"path": "...", "size": 123, "sha": "<sha256>"}]
    }

``source_manifest_sha`` is the canonical SHA-256 of the source rows sorted by
path.  A production pilot receipt is strict JSON of the form::

    {"payload": {
       "campaign": "...", "contract": "<sha256>",
       "source_manifest_sha": "<sha256>", "variant": "V0",
       "prepared_manifest_sha": "<sha256>",
       "bundle": {"verified": true}, "controls": {...}},
     "receipt_sha256": "canonical sha256 of payload"}

The canonical digest is an integrity binding, not a cryptographic signature.
The receipt is trusted only because its file is itself included in the run's
immutable provenance snapshot.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = str(PROJECT_ROOT / "src")
if not sys.path or sys.path[0] != _SOURCE_ROOT:
    sys.path.insert(0, _SOURCE_ROOT)

from spatial_benchmark.identifiers import canonical_json, canonical_sha256
from spatial_benchmark.environment_lock import (
    EnvironmentLockError,
    verify_live_environment,
)
from spatial_benchmark.same_gene_phase_transforms import (
    TrainOnlyPhaseTransformBuilder,
)


CAMPAIGN_ID = "cmp_20260810_same_gene_robustness_multiverse_v1"
CAMPAIGN_DISPLAY_NAME = "Same-gene robustness multiverse v1"
CAMPAIGN_SCIENTIFIC_QUESTION = (
    "Are same-name magnitude enrichment, strict-exclusivity failure, and the "
    "12-epoch-budget conclusion robust?"
)
CONTRACT_RELATIVE_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/frozen_task_contract.yaml"
)
ENVIRONMENT_LOCK_RELATIVE_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/environment_lock.json"
)
ENVIRONMENT_VERIFIER_RELATIVE_PATH = "src/spatial_benchmark/environment_lock.py"
WRAPPER_RELATIVE_PATH = "scripts/train/run_same_gene_robustness.py"
BASE_RUNNER_RELATIVE_PATH = "scripts/train/run_same_gene_nonlinear.py"
PHASE_TRANSFORM_RELATIVE_PATH = (
    "src/spatial_benchmark/same_gene_phase_transforms.py"
)
RESIDUALIZATION_RELATIVE_PATH = (
    "src/spatial_benchmark/same_gene_residualization.py"
)
ALLOWED_MODEL_SEEDS = (20260810, 20261810, 20262810, 20263810, 20264810)
ALLOWED_VARIANTS = frozenset(f"V{index}" for index in range(7))
EPOCH_CANDIDATES = (12, 24, 48, 96, 192)
ANCHOR_EPOCH = 12
PILOT_FORCED_REFIT_EPOCH = 192
MAXIMUM_PILOT_VRAM_GB = 20.5
MAXIMUM_PILOT_PROJECTED_HOURS = 0.25
MAXIMUM_AUTOGRAD_ERROR = 1e-10
MAXIMUM_FINITE_DIFFERENCE_ERROR = 1e-8
MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR = 1e-7
DEFAULT_FEATURE_FILES = {
    "observed_near": "neighbor_near_mean.npy",
    "observed_annular": "neighbor_annular_mean.npy",
    "within_fov_permuted_near": "neighbor_permuted_near_mean.npy",
}
DEFAULT_ELIGIBILITY_FILE = "matched_eligible.npy"
FROZEN_GENE_ELIGIBILITY_FILE = "eligible_genes.npy"
FROZEN_GENE_ELIGIBILITY_SHA256 = (
    "2420a9d160894a78e6a1db1cc4ff2c52757211cf31dec859856f7c3cf231712d"
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_VARIANT = re.compile(r"^V[0-6]$")
_LAUNCH_KEYS = frozenset({"campaign_id", "contract", "sources"})
_JOB_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "variant_root",
        "model_seed",
        "fold",
        "profile",
        "attempt",
        "launch_manifest",
        "pilot_receipt",
        "job_success_marker",
    }
)
_VARIANT_KEYS = frozenset(
    {
        "variant_id",
        "preprocessing_version",
        "raw_snapshot",
        "processed_fingerprint",
        "split_fingerprint",
        "variant_spec",
        "base_prepared_manifest_sha256",
    }
)
_PREPARED_BINDING_KEYS = frozenset(
    {
        "status",
        "robustness_root_manifest_sha256",
        "robustness_root_processed_fingerprint",
        "variants",
    }
)
_PREPARED_VARIANT_KEYS = frozenset(
    {
        "manifest_sha256",
        "integrity_manifest_sha256",
        "processed_fingerprint",
        "variant_spec_sha256",
    }
)
_RECEIPT_KEYS = frozenset({"payload", "receipt_sha256"})
_RECEIPT_PAYLOAD_KEYS = frozenset(
    {
        "campaign",
        "contract",
        "source_manifest_sha",
        "variant",
        "prepared_manifest_sha",
        "selected_attempt",
        "attempt_history",
        "attempt_history_sha256",
        "bundle",
        "controls",
    }
)
_CONTROL_KEYS = frozenset(
    {
        "all_outputs_finite",
        "train_validation_test_component_overlap",
        "receiver_rna_or_derived_covariate_model_input",
        "identity_oracle_row_top1_fraction",
        "identity_oracle_actually_executed",
        "analytical_autograd_max_abs_error",
        "analytical_finite_difference_max_abs_error",
        "graph_specific_invariants",
        "checkpoint_gpu_replay_max_abs_metric_error",
        "checkpoint_gpu_replay_max_abs_prediction_error",
        "checkpoint_replay_device_type",
        "canonical_production_split_label",
        "source_config_data_hashes_verified",
        "outer_test_untouched",
        "peak_vram_gb",
        "projected_full_hours_per_fold",
        "gate_passed",
        "production_authorized",
        "environment_lock_verified",
        "environment_lock_sha256",
        "environment_verification_sha256",
        "environment_visibility_mode",
    }
)


class RobustnessRunError(RuntimeError):
    """Raised before numerical data loading when a launch binding is invalid."""


class UnclaimedAttemptError(RobustnessRunError):
    """Raised when no scientific attempt authority exists for this job."""


def _load_base_runner() -> Any:
    path = PROJECT_ROOT / BASE_RUNNER_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location(
        "bagm_same_gene_nonlinear_robustness_base", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_base_runner()
_BASE_CONFIGURATION = runner._configuration


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RobustnessRunError(
            f"{label} contains non-finite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RobustnessRunError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except RobustnessRunError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RobustnessRunError(
            f"{label} is not strict readable JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RobustnessRunError(f"{label} must contain one JSON object")
    return value


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _unique_yaml_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise RobustnessRunError("contract YAML mapping keys must be strings")
        if key in result:
            raise RobustnessRunError(f"contract contains duplicate YAML key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_yaml_mapping
)


def _strict_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except RobustnessRunError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RobustnessRunError(f"{label} is not strict readable YAML: {path}") from error
    if not isinstance(value, dict):
        raise RobustnessRunError(f"{label} must contain one YAML mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unknown = sorted(observed.difference(expected))
        raise RobustnessRunError(
            f"{label} keys differ; missing={missing}, unknown={unknown}"
        )


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise RobustnessRunError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RobustnessRunError(f"{label} must be a nonempty string")
    return value


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RobustnessRunError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RobustnessRunError(f"{label} must be finite and nonnegative")
    return result


def _require_frozen_at(payload: Mapping[str, Any]) -> str:
    value = payload.get("frozen_at")
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ) is None:
        raise RobustnessRunError(
            "task contract frozen_at must be a non-null canonical UTC timestamp"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise RobustnessRunError(
            "task contract frozen_at is not a valid canonical UTC timestamp"
        ) from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise RobustnessRunError("task contract frozen_at is not canonical UTC")
    return value


def _project_file(
    value: str | Path,
    *,
    project_root: Path,
    label: str,
    require_file: bool = True,
    require_directory: bool = False,
) -> Path:
    root = project_root.resolve(strict=True)
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    absolute = candidate.absolute()
    try:
        relative_unresolved = absolute.relative_to(root)
    except ValueError as error:
        raise RobustnessRunError(f"{label} escapes project root: {candidate}") from error
    cursor = root
    for part in relative_unresolved.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RobustnessRunError(f"{label} may not traverse a symlink: {cursor}")
    try:
        resolved = candidate.resolve(strict=require_file or require_directory)
    except OSError as error:
        raise RobustnessRunError(f"{label} does not exist: {candidate}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RobustnessRunError(f"{label} resolves outside project root: {candidate}") from error
    if require_file and not resolved.is_file():
        raise RobustnessRunError(f"{label} is not a regular file: {resolved}")
    if require_directory and not resolved.is_dir():
        raise RobustnessRunError(f"{label} is not a directory: {resolved}")
    return resolved


def _relative(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(
            project_root.resolve(strict=True)
        ).as_posix()
    except ValueError as error:
        raise RobustnessRunError(f"provenance path escapes project root: {path}") from error


@dataclass(frozen=True, slots=True)
class PreparedContractBinding:
    raw_fingerprint: str
    split_fingerprint: str
    base_prepared_manifest_sha256: str
    robustness_root_manifest_sha256: str
    robustness_root_processed_fingerprint: str
    variants: dict[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class VerifiedLaunch:
    path: Path
    sha256: str
    contract_path: Path
    contract_sha256: str
    sources: tuple[dict[str, Any], ...]
    source_manifest_sha: str
    prepared_binding: PreparedContractBinding


@dataclass(frozen=True, slots=True)
class VerifiedVariant:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    integrity_manifest_path: Path
    integrity_manifest_sha256: str
    robustness_root_manifest_path: Path
    robustness_root_manifest_sha256: str
    robustness_root_processed_fingerprint: str
    variant_id: str
    preprocessing_version: str
    raw_fingerprint: str
    processed_fingerprint: str
    split_fingerprint: str
    base_prepared_manifest_sha256: str
    variant_spec: dict[str, Any]
    feature_files: dict[str, str]
    eligibility_file: str


@dataclass(frozen=True, slots=True)
class VerifiedReceipt:
    path: Path
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedInputs:
    launch: VerifiedLaunch
    variant: VerifiedVariant
    receipt: VerifiedReceipt | None
    model_seed: int
    job_config_path: Path
    job_config_sha256: str
    job_success_marker: Path
    environment_lock_sha256: str
    environment_verification: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class MaterializedJob:
    config_path: Path
    config_sha256: str
    variant_root: Path
    model_seed: int
    fold: int
    profile: str
    attempt: int
    launch_manifest: Path
    pilot_receipt: Path | None
    job_success_marker: Path


def _source_manifest_sha(sources: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {"path": str(row["path"]), "size": int(row["size"]), "sha": str(row["sha"])}
        for row in sources
    ]
    normalized.sort(key=lambda row: row["path"])
    return canonical_sha256(normalized)


def _prepared_contract_binding(
    contract: Mapping[str, Any],
) -> PreparedContractBinding:
    """Read the exact pre-outcome prepared-data authority from a frozen contract."""

    dataset = contract.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RobustnessRunError("frozen contract dataset authority is missing")
    raw_fingerprint = _sha(
        dataset.get("raw_fingerprint"), label="contract raw fingerprint"
    )
    split_fingerprint = _sha(
        dataset.get("split_fingerprint"), label="contract split fingerprint"
    )
    base_manifest_sha = _sha(
        dataset.get("base_prepared_manifest_sha256"),
        label="contract base prepared manifest SHA",
    )
    prepared = dataset.get("prepared_variant_fingerprints")
    if not isinstance(prepared, Mapping):
        raise RobustnessRunError(
            "frozen contract prepared-variant authority is missing"
        )
    _exact_keys(
        prepared, _PREPARED_BINDING_KEYS, label="prepared-variant authority"
    )
    if prepared["status"] != "complete_preoutcome_identity_binding":
        raise RobustnessRunError(
            "prepared-variant authority is not complete_preoutcome_identity_binding"
        )
    root_manifest_sha = _sha(
        prepared["robustness_root_manifest_sha256"],
        label="contract robustness root manifest SHA",
    )
    root_processed = _sha(
        prepared["robustness_root_processed_fingerprint"],
        label="contract robustness root processed fingerprint",
    )
    raw_variants = prepared["variants"]
    if not isinstance(raw_variants, Mapping) or set(raw_variants) != set(
        ALLOWED_VARIANTS
    ):
        raise RobustnessRunError(
            "prepared-variant authority must bind exactly V0 through V6"
        )
    variants: dict[str, dict[str, str]] = {}
    for variant_id in sorted(ALLOWED_VARIANTS):
        authority = raw_variants[variant_id]
        if not isinstance(authority, Mapping):
            raise RobustnessRunError(
                f"contract prepared authority for {variant_id} is malformed"
            )
        _exact_keys(
            authority,
            _PREPARED_VARIANT_KEYS,
            label=f"contract prepared authority for {variant_id}",
        )
        variants[variant_id] = {
            key: _sha(
                authority[key], label=f"contract {variant_id} prepared {key}"
            )
            for key in sorted(_PREPARED_VARIANT_KEYS)
        }
    return PreparedContractBinding(
        raw_fingerprint=raw_fingerprint,
        split_fingerprint=split_fingerprint,
        base_prepared_manifest_sha256=base_manifest_sha,
        robustness_root_manifest_sha256=root_manifest_sha,
        robustness_root_processed_fingerprint=root_processed,
        variants=variants,
    )


def _verify_launch_manifest(path: Path, *, project_root: Path) -> VerifiedLaunch:
    manifest_path = _project_file(
        path, project_root=project_root, label="launch manifest"
    )
    payload = _strict_json(manifest_path, label="launch manifest")
    _exact_keys(payload, _LAUNCH_KEYS, label="launch manifest")
    if payload["campaign_id"] != CAMPAIGN_ID:
        raise RobustnessRunError("launch manifest campaign_id mismatch")

    contract = payload["contract"]
    if not isinstance(contract, Mapping):
        raise RobustnessRunError("launch manifest contract must be an object")
    _exact_keys(contract, frozenset({"path", "sha"}), label="launch contract")
    contract_path = _project_file(
        _nonempty_string(contract["path"], label="launch contract.path"),
        project_root=project_root,
        label="launch contract.path",
    )
    if _relative(contract_path, project_root=project_root) != CONTRACT_RELATIVE_PATH:
        raise RobustnessRunError("launch manifest points at the wrong campaign contract")
    contract_sha = _sha(contract["sha"], label="launch contract.sha")
    if _sha256_file(contract_path) != contract_sha:
        raise RobustnessRunError("launch contract SHA-256 mismatch")
    contract_payload = _strict_yaml(contract_path, label="frozen task contract")
    if contract_payload.get("campaign_id") != CAMPAIGN_ID:
        raise RobustnessRunError("frozen task contract campaign_id mismatch")
    if (
        contract_payload.get("status") != "frozen_preoutcome"
        or contract_payload.get("launch_authorized") is not True
    ):
        raise RobustnessRunError("task contract is not frozen_preoutcome and authorized")
    _require_frozen_at(contract_payload)
    prepared_binding = _prepared_contract_binding(contract_payload)

    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RobustnessRunError("launch manifest sources must be a nonempty list")
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_sources):
        label = f"launch sources[{index}]"
        if not isinstance(raw, Mapping):
            raise RobustnessRunError(f"{label} must be an object")
        _exact_keys(raw, frozenset({"path", "size", "sha"}), label=label)
        source_path = _project_file(
            _nonempty_string(raw["path"], label=f"{label}.path"),
            project_root=project_root,
            label=f"{label}.path",
        )
        relative = _relative(source_path, project_root=project_root)
        if relative in seen:
            raise RobustnessRunError(f"duplicate launch source path: {relative}")
        seen.add(relative)
        size = raw["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RobustnessRunError(f"{label}.size must be a nonnegative integer")
        digest = _sha(raw["sha"], label=f"{label}.sha")
        if source_path.stat().st_size != size or _sha256_file(source_path) != digest:
            raise RobustnessRunError(f"launch source identity mismatch: {relative}")
        sources.append({"path": relative, "size": size, "sha": digest})
    required = {
        ENVIRONMENT_LOCK_RELATIVE_PATH,
        ENVIRONMENT_VERIFIER_RELATIVE_PATH,
        WRAPPER_RELATIVE_PATH,
        BASE_RUNNER_RELATIVE_PATH,
        PHASE_TRANSFORM_RELATIVE_PATH,
        RESIDUALIZATION_RELATIVE_PATH,
    }
    if not required.issubset(seen):
        raise RobustnessRunError(
            f"launch sources omit required runners: {sorted(required.difference(seen))}"
        )
    normalized = tuple(sorted(sources, key=lambda row: str(row["path"])))
    return VerifiedLaunch(
        path=manifest_path,
        sha256=_sha256_file(manifest_path),
        contract_path=contract_path,
        contract_sha256=contract_sha,
        sources=normalized,
        source_manifest_sha=_source_manifest_sha(normalized),
        prepared_binding=prepared_binding,
    )


def _safe_filename(value: Any, *, label: str) -> str:
    filename = _nonempty_string(value, label=label)
    path = Path(filename)
    if path.name != filename or filename in {".", ".."} or "\\" in filename:
        raise RobustnessRunError(f"{label} is unsafe: {filename!r}")
    return filename


def _safe_feature_files(spec: Mapping[str, Any]) -> dict[str, str]:
    raw = spec.get("feature_files", DEFAULT_FEATURE_FILES)
    if not isinstance(raw, Mapping) or set(raw) != set(DEFAULT_FEATURE_FILES):
        raise RobustnessRunError(
            "variant_spec.feature_files must map exactly the three neighbor arms"
        )
    result: dict[str, str] = {}
    for arm in DEFAULT_FEATURE_FILES:
        result[arm] = _safe_filename(raw[arm], label=f"feature_files.{arm}")
    return result


def _verify_bundle_checksum_manifest(artifact: Path, manifest_path: Path) -> None:
    """Recheck every immutable file declared by a canonical run bundle."""

    manifest = _strict_json(manifest_path, label="pilot bundle checksum manifest")
    files = manifest.get("files")
    if manifest.get("version") != 1 or not isinstance(files, Mapping) or not files:
        raise RobustnessRunError("pilot bundle checksum manifest is malformed")
    for relative, identity in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(identity, Mapping)
            or set(identity) != {"type", "size", "sha256"}
            or identity.get("type") != "file"
        ):
            raise RobustnessRunError(
                f"pilot bundle checksum entry is unsafe: {relative!r}"
            )
        target = artifact / relative
        if target.is_symlink() or not target.is_file():
            raise RobustnessRunError(f"pilot bundle file is missing: {relative}")
        size = identity.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RobustnessRunError(f"pilot bundle size is invalid: {relative}")
        digest = _sha(identity.get("sha256"), label=f"pilot bundle {relative}")
        if target.stat().st_size != size or _sha256_file(target) != digest:
            raise RobustnessRunError(f"pilot bundle payload mismatch: {relative}")


def _verify_graph_specific_invariants(
    variant_root: Path,
    *,
    robustness_root: Path,
    variant_spec: Mapping[str, Any],
    eligibility_file: str,
) -> None:
    """Recompute the graph invariants used by the robustness data builder."""

    graph = variant_spec.get("graph")
    node_policy = variant_spec.get("node_policy")
    if not isinstance(graph, Mapping) or not isinstance(node_policy, Mapping):
        raise RobustnessRunError("graph/node-policy authority is malformed")
    partition = graph.get("partition")
    policy = node_policy.get("name")
    if partition not in {"within_fov", "within_geometry_component"}:
        raise RobustnessRunError("graph partition authority is unsupported")
    if policy not in {"all", "qc_induced"}:
        raise RobustnessRunError("graph node-policy authority is unsupported")

    for slide in ("SO_1", "SO_2"):
        slide_root = variant_root / slide
        arrays = {
            name: np.load(slide_root / f"{name}.npy", allow_pickle=False)
            for name in (
                "near_degree",
                "annular_degree",
                "permuted_near_degree",
                "near_indptr",
                "annular_indptr",
                "near_indices",
                "annular_indices",
                "source_permutation",
                "fov",
                "geometry_group",
                "fold",
                "matched_eligible",
                "qc_passed",
            )
        }
        primary = np.load(slide_root / eligibility_file, allow_pickle=False)
        groups = arrays["geometry_group"]
        folds = arrays["fold"]
        fov = arrays["fov"]
        qc = arrays["qc_passed"]
        native = arrays["matched_eligible"]
        source_permutation = arrays["source_permutation"]
        near_degree = arrays["near_degree"]
        annular_degree = arrays["annular_degree"]
        permuted_degree = arrays["permuted_near_degree"]
        near_indptr = arrays["near_indptr"]
        annular_indptr = arrays["annular_indptr"]
        near_indices = arrays["near_indices"]
        annular_indices = arrays["annular_indices"]
        node_count = len(groups)
        vectors = {
            "fold": folds,
            "fov": fov,
            "qc": qc,
            "native": native,
            "primary": primary,
            "source_permutation": source_permutation,
            "near_degree": near_degree,
            "annular_degree": annular_degree,
            "permuted_degree": permuted_degree,
        }
        if any(value.shape != (node_count,) for value in vectors.values()):
            raise RobustnessRunError(f"{slide} graph vectors are misaligned")
        for label, indptr, indices in (
            ("near", near_indptr, near_indices),
            ("annular", annular_indptr, annular_indices),
        ):
            if (
                indptr.shape != (node_count + 1,)
                or int(indptr[0]) != 0
                or int(indptr[-1]) != len(indices)
                or np.any(np.diff(indptr) < 0)
                or np.any(indices < 0)
                or np.any(indices >= node_count)
            ):
                raise RobustnessRunError(f"{slide} {label} CSR is malformed")
        if (
            not np.array_equal(np.diff(near_indptr), near_degree)
            or not np.array_equal(np.diff(annular_indptr), annular_degree)
            or not np.array_equal(near_degree, permuted_degree)
        ):
            raise RobustnessRunError(f"{slide} graph degree invariant failed")

        active = source_permutation >= 0
        expected_active = (
            np.ones(node_count, dtype=bool)
            if policy == "all"
            else np.asarray(qc, dtype=bool)
        )
        if not np.array_equal(active, expected_active):
            raise RobustnessRunError(
                f"{slide} source-permutation active mask changed"
            )
        for fov_value in np.unique(fov[active]):
            nodes = np.flatnonzero(active & (fov == fov_value))
            if (
                not np.array_equal(
                    np.sort(source_permutation[nodes].astype(np.int64)),
                    nodes.astype(np.int64),
                )
            ):
                raise RobustnessRunError(
                    f"{slide} source permutation is not a within-FOV bijection"
                )

        near_rows = np.repeat(
            np.arange(node_count, dtype=np.int64), near_degree
        )
        annular_rows = np.repeat(
            np.arange(node_count, dtype=np.int64), annular_degree
        )
        effective_source = source_permutation[near_indices]
        if np.any(effective_source < 0) or np.any(effective_source == near_rows):
            raise RobustnessRunError(
                f"{slide} permuted source collides with its receiver"
            )
        for label, rows, indices in (
            ("near", near_rows, near_indices),
            ("annular", annular_rows, annular_indices),
        ):
            if (
                np.any(~active[rows])
                or np.any(~active[indices])
                or np.any(groups[rows] != groups[indices])
                or np.any(folds[rows] != folds[indices])
                or np.any(rows == indices)
                or (
                    partition == "within_fov"
                    and np.any(fov[rows] != fov[indices])
                )
            ):
                raise RobustnessRunError(
                    f"{slide} {label} graph violates split/self-edge isolation"
                )

        expected_native = active & (near_degree >= 4) & (annular_degree >= 4)
        if not np.array_equal(np.asarray(native, dtype=bool), expected_native):
            raise RobustnessRunError(
                f"{slide} native eligibility differs from graph degrees"
            )
        base_primary_path = robustness_root / "shared" / slide / "base_matched_eligible.npy"
        base_primary = np.load(base_primary_path, allow_pickle=False).astype(
            bool, copy=False
        )
        if base_primary.shape != (node_count,):
            raise RobustnessRunError(
                f"{slide} frozen primary cohort is misaligned"
            )
        expected_primary = base_primary & (
            np.asarray(qc, dtype=bool) if policy == "qc_induced" else True
        )
        if not np.array_equal(np.asarray(primary, dtype=bool), expected_primary):
            raise RobustnessRunError(
                f"{slide} primary eligibility differs from frozen cohort"
            )


def _verify_variant_root(
    path: Path,
    *,
    project_root: Path,
    binding: PreparedContractBinding | None = None,
) -> VerifiedVariant:
    root = _project_file(
        path,
        project_root=project_root,
        label="variant root",
        require_file=False,
        require_directory=True,
    )
    manifest_path = _project_file(
        root / "manifest.json", project_root=project_root, label="variant manifest"
    )
    payload = _strict_json(manifest_path, label="variant manifest")
    _exact_keys(payload, _VARIANT_KEYS, label="variant manifest")
    variant_id = _nonempty_string(payload["variant_id"], label="variant_id")
    if _SAFE_VARIANT.fullmatch(variant_id) is None or variant_id not in ALLOWED_VARIANTS:
        raise RobustnessRunError(f"unsupported robustness variant: {variant_id!r}")
    preprocessing = _nonempty_string(
        payload["preprocessing_version"], label="preprocessing_version"
    )
    raw_snapshot = payload["raw_snapshot"]
    if not isinstance(raw_snapshot, Mapping):
        raise RobustnessRunError("raw_snapshot must be an object")
    _exact_keys(raw_snapshot, frozenset({"fingerprint"}), label="raw_snapshot")
    variant_spec = payload["variant_spec"]
    if not isinstance(variant_spec, dict):
        raise RobustnessRunError("variant_spec must be an object")
    required_spec = {"graph", "normalization", "node_policy", "permutation_seed"}
    missing = sorted(required_spec.difference(variant_spec))
    if missing:
        raise RobustnessRunError(f"variant_spec is missing keys: {missing}")
    for key in ("graph", "normalization", "node_policy"):
        if variant_spec[key] is None:
            raise RobustnessRunError(f"variant_spec.{key} may not be null")
        canonical_json(variant_spec[key])
    permutation_seed = variant_spec["permutation_seed"]
    if (
        isinstance(permutation_seed, bool)
        or not isinstance(permutation_seed, int)
        or permutation_seed < 0
    ):
        raise RobustnessRunError("variant_spec.permutation_seed must be nonnegative int")
    feature_files = _safe_feature_files(variant_spec)
    eligibility_file = _safe_filename(
        variant_spec.get("primary_eligibility_file", DEFAULT_ELIGIBILITY_FILE),
        label="variant_spec.primary_eligibility_file",
    )

    base_sha = _sha(
        payload["base_prepared_manifest_sha256"],
        label="base_prepared_manifest_sha256",
    )
    optional_base_path = variant_spec.get("base_prepared_manifest_path")
    if optional_base_path is not None:
        base_path = _project_file(
            _nonempty_string(
                optional_base_path, label="variant_spec.base_prepared_manifest_path"
            ),
            project_root=project_root,
            label="variant_spec.base_prepared_manifest_path",
        )
        if _sha256_file(base_path) != base_sha:
            raise RobustnessRunError("base prepared manifest SHA-256 mismatch")

    if root.parent.name != "variants":
        raise RobustnessRunError("variant root must be inside a variants directory")
    robustness_root = root.parent.parent
    root_manifest_path = _project_file(
        robustness_root / "manifest.json",
        project_root=project_root,
        label="robustness root manifest",
    )
    root_manifest = _strict_json(
        root_manifest_path, label="robustness root manifest"
    )
    if root_manifest.get("manifest_schema_version") != 1:
        raise RobustnessRunError("robustness root manifest schema is unsupported")
    root_variants = root_manifest.get("variants")
    if not isinstance(root_variants, Mapping):
        raise RobustnessRunError("robustness root variant inventory is missing")
    matches: list[Mapping[str, Any]] = []
    for record in root_variants.values():
        if not isinstance(record, Mapping):
            raise RobustnessRunError("robustness root variant record is malformed")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise RobustnessRunError("robustness root variant path is malformed")
        candidate = _project_file(
            robustness_root / relative,
            project_root=project_root,
            label="robustness root variant path",
            require_file=False,
            require_directory=True,
        )
        if candidate == root:
            matches.append(record)
    if len(matches) != 1:
        raise RobustnessRunError("variant root has no unique root-manifest authority")
    root_record = matches[0]
    expected_record_keys = {
        "path",
        "manifest_sha256",
        "integrity_manifest_sha256",
        "variant_fingerprint",
        "contract_variant_id",
    }
    if set(root_record) != expected_record_keys:
        raise RobustnessRunError("robustness root variant record keys differ")
    manifest_sha = _sha256_file(manifest_path)
    if (
        root_record.get("contract_variant_id") != variant_id
        or _sha(root_record.get("manifest_sha256"), label="root manifest variant SHA")
        != manifest_sha
        or _sha(root_record.get("variant_fingerprint"), label="root variant fingerprint")
        != payload["processed_fingerprint"]
    ):
        raise RobustnessRunError("root/runner variant identities differ")
    root_fingerprint_payload = dict(root_manifest)
    expected_root_fingerprint = _sha(
        root_fingerprint_payload.pop("processed_fingerprint", None),
        label="robustness root processed fingerprint",
    )
    root_fingerprint_payload.pop("created_at", None)
    if canonical_sha256(root_fingerprint_payload) != expected_root_fingerprint:
        raise RobustnessRunError("robustness root processed fingerprint changed")

    integrity_path = _project_file(
        root / "integrity_manifest.json",
        project_root=project_root,
        label="variant integrity manifest",
    )
    integrity_sha = _sha256_file(integrity_path)
    if integrity_sha != _sha(
        root_record.get("integrity_manifest_sha256"),
        label="root integrity-manifest SHA",
    ):
        raise RobustnessRunError("variant integrity-manifest SHA changed")
    integrity = _strict_json(integrity_path, label="variant integrity manifest")
    if integrity.get("contract_variant_id") != variant_id:
        raise RobustnessRunError("variant integrity contract identity differs")
    for integrity_key, runner_value in (
        ("preprocessing_version", preprocessing),
        ("raw_snapshot", raw_snapshot),
        ("split_fingerprint", payload["split_fingerprint"]),
        ("base_prepared_manifest_sha256", base_sha),
        ("variant_spec", variant_spec),
    ):
        if integrity.get(integrity_key) != runner_value:
            raise RobustnessRunError(
                f"variant integrity {integrity_key} differs from runner manifest"
            )
    integrity_payload = dict(integrity)
    integrity_fingerprint = _sha(
        integrity_payload.pop("variant_fingerprint", None),
        label="integrity variant fingerprint",
    )
    if (
        canonical_sha256(integrity_payload) != integrity_fingerprint
        or integrity_fingerprint != payload["processed_fingerprint"]
    ):
        raise RobustnessRunError("variant integrity fingerprint changed")
    verified = VerifiedVariant(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        integrity_manifest_path=integrity_path,
        integrity_manifest_sha256=integrity_sha,
        robustness_root_manifest_path=root_manifest_path,
        robustness_root_manifest_sha256=_sha256_file(root_manifest_path),
        robustness_root_processed_fingerprint=expected_root_fingerprint,
        variant_id=variant_id,
        preprocessing_version=preprocessing,
        raw_fingerprint=_sha(raw_snapshot["fingerprint"], label="raw fingerprint"),
        processed_fingerprint=_sha(
            payload["processed_fingerprint"], label="processed fingerprint"
        ),
        split_fingerprint=_sha(
            payload["split_fingerprint"], label="split fingerprint"
        ),
        base_prepared_manifest_sha256=base_sha,
        variant_spec=copy.deepcopy(variant_spec),
        feature_files=feature_files,
        eligibility_file=eligibility_file,
    )
    if binding is not None:
        _verify_variant_contract_binding_values(binding, verified)
    content = integrity.get("content")
    if not isinstance(content, Mapping) or not content:
        raise RobustnessRunError("variant integrity content inventory is missing")
    observed_content = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.name not in {"manifest.json", "integrity_manifest.json"}
    }
    if observed_content != set(content):
        raise RobustnessRunError(
            "variant tree contains undeclared or missing content"
        )
    required_content = {FROZEN_GENE_ELIGIBILITY_FILE}
    for slide in ("SO_1", "SO_2"):
        required_content.update(
            {
                f"{slide}/fold.npy",
                f"{slide}/geometry_group.npy",
                f"{slide}/{eligibility_file}",
                f"{slide}/expression_log1p.npy",
                f"{slide}/metadata.npy",
                *(f"{slide}/{name}" for name in feature_files.values()),
            }
        )
    if not required_content.issubset(content):
        raise RobustnessRunError(
            "variant integrity inventory omits runner-required inputs"
        )
    for relative, record in content.items():
        raw_relative = Path(relative) if isinstance(relative, str) else Path("/")
        if (
            not isinstance(relative, str)
            or raw_relative.is_absolute()
            or ".." in raw_relative.parts
            or not isinstance(record, Mapping)
            or record.get("path") != relative
            or not isinstance(record.get("size_bytes"), int)
            or int(record["size_bytes"]) < 0
        ):
            raise RobustnessRunError("variant integrity content record is malformed")
        _sha(record.get("sha256"), label=f"variant content {relative}")
        unresolved = root / raw_relative
        if unresolved.is_symlink():
            link_target = os.readlink(unresolved)
            if Path(link_target).is_absolute():
                raise RobustnessRunError(
                    f"variant content uses an absolute symlink: {relative}"
                )
            target = (unresolved.parent / link_target).resolve(strict=True)
            if record.get("storage") != "relative_symlink" or (
                record.get("link_target") != link_target
            ):
                raise RobustnessRunError(
                    f"variant content symlink authority changed: {relative}"
                )
        else:
            if record.get("storage") != "regular_file":
                raise RobustnessRunError(
                    f"variant content storage authority changed: {relative}"
                )
            target = unresolved.resolve(strict=True)
        if not target.is_file() or target.is_symlink():
            raise RobustnessRunError(
                f"variant content target is not a regular file: {relative}"
            )
        try:
            target.relative_to(robustness_root.resolve(strict=True))
        except ValueError as error:
            raise RobustnessRunError(
                f"variant content escapes robustness root: {relative}"
            ) from error
        if target.stat().st_size != int(record["size_bytes"]):
            raise RobustnessRunError(f"variant content size changed: {relative}")
        if _sha256_file(target) != record["sha256"]:
            raise RobustnessRunError(f"variant content hash changed: {relative}")
        if target.suffix == ".npy":
            try:
                array = np.load(target, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as error:
                raise RobustnessRunError(
                    f"variant array cannot be read: {relative}"
                ) from error
            if list(array.shape) != record.get("shape") or str(array.dtype) != record.get(
                "dtype"
            ):
                raise RobustnessRunError(
                    f"variant array shape/dtype changed: {relative}"
                )

    _verify_graph_specific_invariants(
        root,
        robustness_root=robustness_root,
        variant_spec=variant_spec,
        eligibility_file=eligibility_file,
    )

    return verified


def _verify_variant_contract_binding(
    launch: VerifiedLaunch, variant: VerifiedVariant
) -> None:
    """Fail closed if a prepared variant differs from its frozen authority."""

    _verify_variant_contract_binding_values(launch.prepared_binding, variant)


def _verify_variant_contract_binding_values(
    binding: PreparedContractBinding, variant: VerifiedVariant
) -> None:
    """Compare one fully described variant with the contract binding values."""

    observed_common = {
        "raw fingerprint": variant.raw_fingerprint,
        "split fingerprint": variant.split_fingerprint,
        "base prepared manifest SHA": variant.base_prepared_manifest_sha256,
        "robustness root manifest SHA": variant.robustness_root_manifest_sha256,
        "robustness root processed fingerprint": (
            variant.robustness_root_processed_fingerprint
        ),
    }
    expected_common = {
        "raw fingerprint": binding.raw_fingerprint,
        "split fingerprint": binding.split_fingerprint,
        "base prepared manifest SHA": binding.base_prepared_manifest_sha256,
        "robustness root manifest SHA": binding.robustness_root_manifest_sha256,
        "robustness root processed fingerprint": (
            binding.robustness_root_processed_fingerprint
        ),
    }
    for label, observed in observed_common.items():
        if observed != expected_common[label]:
            raise RobustnessRunError(
                f"{variant.variant_id} {label} differs from frozen contract"
            )
    observed_variant = {
        "manifest_sha256": variant.manifest_sha256,
        "integrity_manifest_sha256": variant.integrity_manifest_sha256,
        "processed_fingerprint": variant.processed_fingerprint,
        "variant_spec_sha256": canonical_sha256(variant.variant_spec),
    }
    expected_variant = binding.variants[variant.variant_id]
    for key, observed in observed_variant.items():
        if observed != expected_variant[key]:
            raise RobustnessRunError(
                f"{variant.variant_id} {key} differs from frozen contract"
            )


def _verify_pilot_receipt(
    path: Path,
    *,
    project_root: Path,
    launch: VerifiedLaunch,
    variant: VerifiedVariant,
) -> VerifiedReceipt:
    receipt_path = _project_file(
        path, project_root=project_root, label="pilot receipt"
    )
    receipt = _strict_json(receipt_path, label="pilot receipt")
    _exact_keys(receipt, _RECEIPT_KEYS, label="pilot receipt")
    payload = receipt["payload"]
    if not isinstance(payload, Mapping):
        raise RobustnessRunError("pilot receipt payload must be an object")
    _exact_keys(payload, _RECEIPT_PAYLOAD_KEYS, label="pilot receipt payload")
    receipt_sha = _sha(receipt["receipt_sha256"], label="pilot receipt_sha256")
    if canonical_sha256(payload) != receipt_sha:
        raise RobustnessRunError("pilot receipt canonical SHA-256 mismatch")
    expected = {
        "campaign": CAMPAIGN_ID,
        "contract": launch.contract_sha256,
        "source_manifest_sha": launch.source_manifest_sha,
        "variant": variant.variant_id,
        "prepared_manifest_sha": variant.manifest_sha256,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise RobustnessRunError(f"pilot receipt {key} binding mismatch")
    bundle = payload["bundle"]
    if not isinstance(bundle, Mapping):
        raise RobustnessRunError("pilot receipt bundle must be an object")
    _exact_keys(
        bundle,
        frozenset(
            {
                "verified",
                "run_id",
                "artifact_path",
                "success_sha256",
                "config_sha256",
                "bundle_manifest_sha256",
            }
        ),
        label="pilot receipt bundle",
    )
    if bundle["verified"] is not True:
        raise RobustnessRunError("pilot receipt bundle is not verified")
    _nonempty_string(bundle["run_id"], label="pilot receipt bundle.run_id")
    artifact = _project_file(
        _nonempty_string(
            bundle["artifact_path"], label="pilot receipt bundle.artifact_path"
        ),
        project_root=project_root,
        label="pilot receipt bundle.artifact_path",
        require_file=False,
        require_directory=True,
    )
    success_path = _project_file(
        artifact / "_SUCCESS",
        project_root=project_root,
        label="pilot receipt bundle _SUCCESS",
    )
    bundle_manifest_path = _project_file(
        artifact / "provenance/artifact_checksums.json",
        project_root=project_root,
        label="pilot receipt bundle manifest",
    )
    if _sha256_file(success_path) != _sha(
        bundle["success_sha256"], label="pilot receipt bundle.success_sha256"
    ):
        raise RobustnessRunError("pilot receipt _SUCCESS SHA-256 mismatch")
    if _sha256_file(bundle_manifest_path) != _sha(
        bundle["bundle_manifest_sha256"],
        label="pilot receipt bundle.bundle_manifest_sha256",
    ):
        raise RobustnessRunError("pilot receipt bundle-manifest SHA-256 mismatch")
    _sha(bundle["config_sha256"], label="pilot receipt bundle.config_sha256")
    _verify_bundle_checksum_manifest(artifact, bundle_manifest_path)

    selected_attempt = payload["selected_attempt"]
    history = payload["attempt_history"]
    if (
        isinstance(selected_attempt, bool)
        or not isinstance(selected_attempt, int)
        or selected_attempt < 1
        or not isinstance(history, list)
        or not history
    ):
        raise RobustnessRunError("pilot receipt attempt history is malformed")
    history_sha = _sha(
        payload["attempt_history_sha256"],
        label="pilot receipt attempt_history_sha256",
    )
    if canonical_sha256(history) != history_sha:
        raise RobustnessRunError("pilot receipt attempt-history SHA mismatch")
    expected_history_keys = frozenset(
        {
            "attempt",
            "job_id",
            "plan_sha256",
            "config_sha256",
            "selected",
            "registry_status",
            "run_id",
            "artifact_path",
            "artifact_status",
        }
    )
    attempts: list[int] = []
    selected_rows: list[Mapping[str, Any]] = []
    for index, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            raise RobustnessRunError("pilot receipt attempt entry is malformed")
        _exact_keys(entry, expected_history_keys, label=f"pilot attempt {index}")
        attempt = entry["attempt"]
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise RobustnessRunError("pilot receipt attempt number is invalid")
        attempts.append(attempt)
        _nonempty_string(entry["job_id"], label="pilot attempt job_id")
        _sha(entry["plan_sha256"], label="pilot attempt plan_sha256")
        _sha(entry["config_sha256"], label="pilot attempt config_sha256")
        _nonempty_string(entry["run_id"], label="pilot attempt run_id")
        attempt_artifact = _project_file(
            _nonempty_string(
                entry["artifact_path"], label="pilot attempt artifact_path"
            ),
            project_root=project_root,
            label="pilot attempt artifact",
            require_file=False,
            require_directory=True,
        )
        if entry["selected"] is True:
            selected_rows.append(entry)
            if (
                entry["registry_status"] != "completed"
                or entry["artifact_status"] != "success"
                or entry["run_id"] != bundle["run_id"]
                or entry["config_sha256"] != bundle["config_sha256"]
                or attempt_artifact != artifact
            ):
                raise RobustnessRunError("selected pilot attempt binding mismatch")
        elif entry["selected"] is False:
            if entry["registry_status"] not in {"failed", "cancelled", "pruned"}:
                raise RobustnessRunError(
                    "superseded pilot attempt is not terminal unsuccessful"
                )
            if entry["artifact_status"] not in {"failed", "pruned"}:
                raise RobustnessRunError(
                    "superseded pilot attempt artifact is not terminal"
                )
            try:
                runner.verify_run_bundle(
                    attempt_artifact, require_success_contract=False
                )
            except Exception as error:
                raise RobustnessRunError(
                    "superseded pilot attempt bundle verification failed"
                ) from error
        else:
            raise RobustnessRunError("pilot receipt selected flag must be boolean")
    if (
        attempts != list(range(1, selected_attempt + 1))
        or len(selected_rows) != 1
        or selected_rows[0]["attempt"] != selected_attempt
    ):
        raise RobustnessRunError("pilot receipt attempt lineage is noncontiguous")

    controls = payload["controls"]
    if not isinstance(controls, Mapping):
        raise RobustnessRunError("pilot receipt controls must be an object")
    _exact_keys(controls, _CONTROL_KEYS, label="pilot receipt controls")
    if controls["all_outputs_finite"] is not True:
        raise RobustnessRunError("pilot finite-output control failed")
    if controls["train_validation_test_component_overlap"] is not False:
        raise RobustnessRunError("pilot split-overlap control failed")
    if controls["receiver_rna_or_derived_covariate_model_input"] is not False:
        raise RobustnessRunError("pilot receiver-input control failed")
    if controls["identity_oracle_actually_executed"] is not True:
        raise RobustnessRunError("pilot identity-oracle execution control failed")
    if _number(
        controls["identity_oracle_row_top1_fraction"], label="pilot oracle"
    ) != 1.0:
        raise RobustnessRunError("pilot identity-oracle control failed")
    if _number(
        controls["analytical_autograd_max_abs_error"], label="pilot autograd"
    ) > MAXIMUM_AUTOGRAD_ERROR:
        raise RobustnessRunError("pilot analytical-autograd control failed")
    if _number(
        controls["analytical_finite_difference_max_abs_error"],
        label="pilot finite difference",
    ) > MAXIMUM_FINITE_DIFFERENCE_ERROR:
        raise RobustnessRunError("pilot finite-difference control failed")
    if controls["graph_specific_invariants"] is not True:
        raise RobustnessRunError("pilot graph-invariant control failed")
    if _number(
        controls["checkpoint_gpu_replay_max_abs_metric_error"],
        label="pilot checkpoint replay metric",
    ) > MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR:
        raise RobustnessRunError("pilot checkpoint metric-replay control failed")
    if _number(
        controls["checkpoint_gpu_replay_max_abs_prediction_error"],
        label="pilot checkpoint replay prediction",
    ) > MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR:
        raise RobustnessRunError("pilot checkpoint prediction-replay control failed")
    if controls["checkpoint_replay_device_type"] != "cuda":
        raise RobustnessRunError("pilot checkpoint replay was not executed on GPU")
    if controls["canonical_production_split_label"] != "test":
        raise RobustnessRunError("pilot canonical production-split control failed")
    if controls["source_config_data_hashes_verified"] is not True:
        raise RobustnessRunError("pilot source/config/data hash control failed")
    if controls["outer_test_untouched"] is not True:
        raise RobustnessRunError("pilot outer-test isolation control failed")
    if controls["environment_lock_verified"] is not True:
        raise RobustnessRunError("pilot environment-lock control failed")
    if controls["environment_visibility_mode"] != "job":
        raise RobustnessRunError("pilot environment visibility mode changed")
    if _sha(
        controls["environment_lock_sha256"],
        label="pilot environment-lock SHA",
    ) != _sha256_file(project_root / ENVIRONMENT_LOCK_RELATIVE_PATH):
        raise RobustnessRunError("pilot environment-lock identity changed")
    _sha(
        controls["environment_verification_sha256"],
        label="pilot environment verification SHA",
    )
    if _number(controls["peak_vram_gb"], label="pilot VRAM") > MAXIMUM_PILOT_VRAM_GB:
        raise RobustnessRunError("pilot VRAM gate failed")
    if _number(
        controls["projected_full_hours_per_fold"], label="pilot runtime"
    ) > MAXIMUM_PILOT_PROJECTED_HOURS:
        raise RobustnessRunError("pilot projected-runtime gate failed")
    if controls["gate_passed"] is not True:
        raise RobustnessRunError("pilot aggregate gate did not pass")
    if controls["production_authorized"] is not True:
        raise RobustnessRunError("pilot did not authorize production")
    return VerifiedReceipt(path=receipt_path, receipt_sha256=receipt_sha)


def _job_path(
    value: Any,
    *,
    project_root: Path,
    label: str,
    require_file: bool = True,
    require_directory: bool = False,
) -> Path:
    return _project_file(
        _nonempty_string(value, label=label),
        project_root=project_root,
        label=label,
        require_file=require_file,
        require_directory=require_directory,
    )


def _load_materialized_job(
    path: Path, *, project_root: Path = PROJECT_ROOT
) -> MaterializedJob:
    config_path = _project_file(
        path, project_root=project_root, label="materialized job config"
    )
    payload = _strict_json(config_path, label="materialized job config")
    _exact_keys(payload, _JOB_CONFIG_KEYS, label="materialized job config")
    if payload["schema_version"] != 1:
        raise RobustnessRunError("materialized job schema_version must equal 1")
    model_seed = payload["model_seed"]
    fold = payload["fold"]
    attempt = payload["attempt"]
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise RobustnessRunError("materialized model_seed must be an integer")
    if model_seed not in ALLOWED_MODEL_SEEDS:
        raise RobustnessRunError("materialized model_seed is outside the frozen set")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(4):
        raise RobustnessRunError("materialized fold must be 0, 1, 2, or 3")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise RobustnessRunError("materialized attempt must be a positive integer")
    profile = payload["profile"]
    if profile not in {"pilot", "full"}:
        raise RobustnessRunError("materialized profile must be pilot or full")
    variant_root = _job_path(
        payload["variant_root"],
        project_root=project_root,
        label="materialized variant_root",
        require_file=False,
        require_directory=True,
    )
    launch_manifest = _job_path(
        payload["launch_manifest"],
        project_root=project_root,
        label="materialized launch_manifest",
    )
    receipt_raw = payload["pilot_receipt"]
    pilot_receipt = (
        None
        if receipt_raw is None
        else _job_path(
            receipt_raw,
            project_root=project_root,
            label="materialized pilot_receipt",
        )
    )
    marker = _job_path(
        payload["job_success_marker"],
        project_root=project_root,
        label="materialized job_success_marker",
        require_file=False,
    )
    if marker.suffix != ".json":
        raise RobustnessRunError("job_success_marker must have a .json suffix")
    if marker == config_path or marker in {launch_manifest, pilot_receipt}:
        raise RobustnessRunError("job_success_marker collides with an input file")
    return MaterializedJob(
        config_path=config_path,
        config_sha256=_sha256_file(config_path),
        variant_root=variant_root,
        model_seed=model_seed,
        fold=fold,
        profile=str(profile),
        attempt=attempt,
        launch_manifest=launch_manifest,
        pilot_receipt=pilot_receipt,
        job_success_marker=marker,
    )


def _cli_path(
    value: Path, *, project_root: Path, label: str, directory: bool = False
) -> Path:
    return _project_file(
        value,
        project_root=project_root,
        label=label,
        require_file=not directory,
        require_directory=directory,
    )


def _materialize_cli_arguments(
    cli: argparse.Namespace, *, project_root: Path = PROJECT_ROOT
) -> argparse.Namespace:
    job = _load_materialized_job(cli.config, project_root=project_root)
    scalar_pairs = {
        "model_seed": (cli.model_seed, job.model_seed),
        "fold": (cli.fold, job.fold),
        "profile": (cli.profile, job.profile),
        "attempt": (cli.attempt, job.attempt),
    }
    for label, (observed, expected) in scalar_pairs.items():
        if observed is not None and observed != expected:
            raise RobustnessRunError(
                f"CLI --{label.replace('_', '-')} does not match materialized config"
            )
    path_pairs: tuple[tuple[str, Path | None, Path | None, bool], ...] = (
        ("variant-root", cli.variant_root, job.variant_root, True),
        ("launch-manifest", cli.launch_manifest, job.launch_manifest, False),
        ("pilot-receipt", cli.pilot_receipt, job.pilot_receipt, False),
    )
    for label, observed, expected, directory in path_pairs:
        if observed is None:
            continue
        resolved = _cli_path(
            observed,
            project_root=project_root,
            label=f"CLI --{label}",
            directory=directory,
        )
        if expected is None or resolved != expected:
            raise RobustnessRunError(
                f"CLI --{label} does not match materialized config"
            )
    return argparse.Namespace(
        config_path=job.config_path,
        config_sha256=job.config_sha256,
        job_success_marker=job.job_success_marker,
        variant_root=job.variant_root,
        model_seed=job.model_seed,
        fold=job.fold,
        profile=job.profile,
        attempt=job.attempt,
        launch_manifest=job.launch_manifest,
        pilot_receipt=job.pilot_receipt,
        verify_job_marker=bool(cli.verify_job_marker),
        abandon_incomplete_attempt=bool(
            getattr(cli, "abandon_incomplete_attempt", False)
        ),
    )


def _verify_inputs(
    arguments: argparse.Namespace,
    *,
    project_root: Path = PROJECT_ROOT,
    verify_live_job_environment: bool = True,
) -> VerifiedInputs:
    if arguments.model_seed not in ALLOWED_MODEL_SEEDS:
        raise RobustnessRunError("model seed is outside the frozen five-seed set")
    if arguments.fold not in range(4):
        raise RobustnessRunError("fold must be 0, 1, 2, or 3")
    if arguments.profile not in {"pilot", "full"}:
        raise RobustnessRunError("profile must be pilot or full")
    if isinstance(arguments.attempt, bool) or arguments.attempt < 1:
        raise RobustnessRunError("attempt must be a positive integer")
    launch = _verify_launch_manifest(arguments.launch_manifest, project_root=project_root)
    environment_lock_sha256 = _sha256_file(
        project_root / ENVIRONMENT_LOCK_RELATIVE_PATH
    )
    environment_verification: dict[str, Any] | None = None
    if verify_live_job_environment:
        try:
            environment_verification = verify_live_environment(
                project_root / ENVIRONMENT_LOCK_RELATIVE_PATH,
                visibility_mode="job",
            )
        except EnvironmentLockError as error:
            raise RobustnessRunError(
                "live environment differs from the frozen environment lock"
            ) from error
        if (
            environment_verification.get("environment_lock_sha256")
            != environment_lock_sha256
        ):
            raise RobustnessRunError(
                "live environment receipt is not bound to the frozen lock"
            )
    variant = _verify_variant_root(
        arguments.variant_root,
        project_root=project_root,
        binding=launch.prepared_binding,
    )
    _verify_variant_contract_binding(launch, variant)
    if arguments.profile == "full" and arguments.pilot_receipt is None:
        raise RobustnessRunError("--pilot-receipt is required for full production")
    receipt = (
        None
        if arguments.pilot_receipt is None
        else _verify_pilot_receipt(
            arguments.pilot_receipt,
            project_root=project_root,
            launch=launch,
            variant=variant,
        )
    )
    return VerifiedInputs(
        launch=launch,
        variant=variant,
        receipt=receipt,
        model_seed=int(arguments.model_seed),
        job_config_path=arguments.config_path,
        job_config_sha256=arguments.config_sha256,
        job_success_marker=arguments.job_success_marker,
        environment_lock_sha256=environment_lock_sha256,
        environment_verification=environment_verification,
    )


def _build_configuration(
    *,
    profile: str,
    fold: int,
    attempt: int,
    verified: VerifiedInputs,
) -> dict[str, Any]:
    configuration = _BASE_CONFIGURATION(
        profile=profile, fold=fold, attempt=attempt
    )
    variant = verified.variant
    configuration["campaign"].update(
        {
            "campaign_id": CAMPAIGN_ID,
            "frozen_contract_sha256": verified.launch.contract_sha256,
            "experiment_flavor": "robustness_multiverse_v1",
            "launch_manifest_sha256": verified.launch.sha256,
            "source_manifest_sha256": verified.launch.source_manifest_sha,
            "materialized_job_config_sha256": verified.job_config_sha256,
            "pilot_receipt_sha256": (
                None
                if verified.receipt is None
                else verified.receipt.receipt_sha256
            ),
            # Bind the immutable authority in the scientific configuration.
            # The process-local live receipt is separately archived in hardware
            # provenance and technical controls.
            "environment_lock_sha256": verified.environment_lock_sha256,
        }
    )
    configuration["dataset"].update(
        {
            "dataset_fingerprint": variant.raw_fingerprint,
            "processed_fingerprint": variant.processed_fingerprint,
            "split_fingerprint": variant.split_fingerprint,
            "preprocessing_version": variant.preprocessing_version,
            "variant_manifest_sha256": variant.manifest_sha256,
            "base_prepared_manifest_sha256": (
                variant.base_prepared_manifest_sha256
            ),
        }
    )
    graph = copy.deepcopy(variant.variant_spec["graph"])
    normalization = copy.deepcopy(variant.variant_spec["normalization"])
    node_policy = copy.deepcopy(variant.variant_spec["node_policy"])
    configuration["robustness_variant"] = {
        "variant_id": variant.variant_id,
        "graph": graph,
        "normalization": normalization,
        "node_policy": node_policy,
        "permutation_seed": int(variant.variant_spec["permutation_seed"]),
        "feature_files": dict(variant.feature_files),
        "primary_eligibility_file": variant.eligibility_file,
        "complete_variant_spec": copy.deepcopy(variant.variant_spec),
    }
    configuration["graph"]["robustness_graph"] = graph
    configuration["graph"]["permutation_seed"] = int(
        variant.variant_spec["permutation_seed"]
    )
    configuration["normalization"] = normalization
    configuration["node_policy"] = node_policy
    configuration["cohort"] = {
        "node_policy": node_policy,
        "primary_eligibility_file": variant.eligibility_file,
    }
    configuration["evaluation"]["primary_eligibility_file"] = (
        variant.eligibility_file
    )
    configuration["evaluation"].update(
        {
            "frozen_gene_eligibility_file": FROZEN_GENE_ELIGIBILITY_FILE,
            "frozen_gene_eligibility_sha256": (
                FROZEN_GENE_ELIGIBILITY_SHA256
            ),
            "frozen_gene_eligibility_count": 932,
        }
    )
    configuration["trainer"].update(
        {
            "epoch_candidates": list(EPOCH_CANDIDATES),
            "effective_epoch_candidates": list(EPOCH_CANDIDATES),
            "anchor_epoch": ANCHOR_EPOCH,
            "pilot_refit_epoch_override": (
                PILOT_FORCED_REFIT_EPOCH if profile == "pilot" else None
            ),
            "model_seed": verified.model_seed + fold,
        }
    )
    configuration["seed"] = verified.model_seed % 1_000_000
    configuration["classification"].update(
        {
            "variant_label": f"same_gene_robustness_{variant.variant_id}_{profile}",
            "experiment_flavor": "robustness_multiverse_v1",
            "seed": verified.model_seed % 1_000_000,
            "execution_seed": verified.model_seed + fold,
        }
    )
    return configuration


def _patch_base_runner(
    verified: VerifiedInputs,
    *,
    project_root: Path,
    require_live_job_environment: bool = True,
) -> None:
    if require_live_job_environment and verified.environment_verification is None:
        raise RobustnessRunError(
            "a live job environment verification is required to run training"
        )
    variant = verified.variant
    runner.CAMPAIGN_ID = CAMPAIGN_ID
    runner.CAMPAIGN_DISPLAY_NAME = CAMPAIGN_DISPLAY_NAME
    runner.CAMPAIGN_SCIENTIFIC_QUESTION = CAMPAIGN_SCIENTIFIC_QUESTION
    runner.EXPERIMENT_FLAVOR = "robustness_multiverse_v1"
    runner.FROZEN_CONTRACT_SHA256 = verified.launch.contract_sha256
    runner.PREPROCESSING_VERSION = variant.preprocessing_version
    runner.RAW_FINGERPRINT = variant.raw_fingerprint
    runner.PROCESSED_FINGERPRINT = variant.processed_fingerprint
    runner.SPLIT_FINGERPRINT = variant.split_fingerprint
    runner.FEATURE_FILE = dict(variant.feature_files)
    runner.ELIGIBILITY_FILE = variant.eligibility_file
    runner.FROZEN_GENE_ELIGIBILITY_FILE = FROZEN_GENE_ELIGIBILITY_FILE
    runner.FROZEN_GENE_ELIGIBILITY_SHA256 = (
        FROZEN_GENE_ELIGIBILITY_SHA256
    )
    residualization = variant.variant_spec.get("residualization")
    if residualization is None or residualization == {"kind": "none"}:
        runner.PHASE_INPUT_BUILDER = None
    elif isinstance(residualization, Mapping):
        runner.PHASE_INPUT_BUILDER = TrainOnlyPhaseTransformBuilder(
            variant.root,
            residualization,
        )
    else:
        raise RobustnessRunError(
            "variant_spec.residualization must be an object or null"
        )
    runner.EPOCH_CANDIDATES = EPOCH_CANDIDATES
    runner.PILOT_EPOCH_CANDIDATES = EPOCH_CANDIDATES
    runner.PILOT_ARMS = tuple(runner.FULL_ARMS)
    runner.PILOT_REFIT_EPOCH_OVERRIDE = PILOT_FORCED_REFIT_EPOCH
    runner.ANCHOR_EPOCH = ANCHOR_EPOCH
    runner.MAXIMUM_PROJECTED_HOURS_PER_FOLD = MAXIMUM_PILOT_PROJECTED_HOURS
    runner.MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR = (
        MAXIMUM_CHECKPOINT_REPLAY_ABS_ERROR
    )
    # Reaching this point means launch sources/config, every declared array
    # byte, and the authenticated builder manifests have all passed the
    # wrapper's fail-closed verification before numerical loading.
    runner.SOURCE_CONFIG_DATA_HASHES_VERIFIED = True
    runner.GRAPH_SPECIFIC_INVARIANTS_VERIFIED = True
    runner.ENVIRONMENT_LOCK_REQUIRED = require_live_job_environment
    runner.ENVIRONMENT_LOCK_REPORT = (
        copy.deepcopy(verified.environment_verification)
        if require_live_job_environment
        else None
    )
    runner.CANONICAL_PRODUCTION_SPLIT_LABEL = "test"
    runner.SEED_BASE = verified.model_seed
    runner.TRACKING_SEED = verified.model_seed % 1_000_000
    runner._configuration = lambda *, profile, fold, attempt: _build_configuration(
        profile=profile,
        fold=fold,
        attempt=attempt,
        verified=verified,
    )

    source_paths = {str(row["path"]) for row in verified.launch.sources}
    source_paths.update(
        {
            _relative(verified.launch.contract_path, project_root=project_root),
            _relative(verified.launch.path, project_root=project_root),
            _relative(verified.job_config_path, project_root=project_root),
            _relative(variant.manifest_path, project_root=project_root),
            _relative(variant.integrity_manifest_path, project_root=project_root),
            _relative(
                variant.robustness_root_manifest_path,
                project_root=project_root,
            ),
            WRAPPER_RELATIVE_PATH,
            BASE_RUNNER_RELATIVE_PATH,
        }
    )
    if verified.receipt is not None:
        source_paths.add(_relative(verified.receipt.path, project_root=project_root))
    runner.PROVENANCE_SOURCE_PATHS = tuple(sorted(source_paths))


def _verify_completed_run(
    *, run_id: str, artifact_path: Path, project_root: Path
) -> tuple[Path, str]:
    _nonempty_string(run_id, label="completed run_id")
    artifact = _project_file(
        artifact_path,
        project_root=project_root,
        label="completed run artifact",
        require_file=False,
        require_directory=True,
    )
    success = _project_file(
        artifact / "_SUCCESS",
        project_root=project_root,
        label="completed run _SUCCESS",
    )
    runner.verify_run_bundle(artifact)
    paths = runner.current_paths()
    registry = runner.Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    row = registry.get_run(run_id)
    if not row or row.get("status") != "completed":
        raise RobustnessRunError("run registry status is not completed")
    registered_artifact = row.get("artifact_path")
    if registered_artifact:
        registered = Path(str(registered_artifact)).resolve(strict=False)
        if registered != artifact:
            raise RobustnessRunError("registry artifact path differs from run result")
    issues = registry.verify_artifacts(run_id=run_id)
    if issues:
        raise RobustnessRunError(f"registry artifact verification failed: {issues}")
    return artifact, _sha256_file(success)


def _atomic_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RobustnessRunError(f"job success marker already exists: {path}")
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RobustnessRunError(
                f"job success marker was concurrently created: {path}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_job_marker(
    *,
    verified: VerifiedInputs,
    run_id: str,
    artifact: Path,
    success_sha256: str,
    project_root: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "config_sha256": verified.job_config_sha256,
        "run_id": run_id,
        "artifact_path": _relative(artifact, project_root=project_root),
        "artifact_success_sha256": success_sha256,
    }
    marker = {"payload": payload, "marker_sha256": canonical_sha256(payload)}
    _atomic_exclusive_json(verified.job_success_marker, marker)
    return marker


def verify_job_marker(
    arguments: argparse.Namespace,
    *,
    project_root: Path = PROJECT_ROOT,
    verify_live_job_environment: bool = True,
) -> dict[str, Any]:
    """Revalidate or crash-reconcile one coordinator marker without training.

    Launch and resume callers retain the live one-GPU environment gate. Offline
    analysis may skip that live gate because it separately verifies both the
    archived job-environment evidence and its live four-GPU analysis environment.
    Missing-marker reconciliation remains restricted to the live-gated mode.
    """

    verified = _verify_inputs(
        arguments,
        project_root=project_root,
        verify_live_job_environment=verify_live_job_environment,
    )
    if verified.job_success_marker.is_symlink():
        raise RobustnessRunError("job success marker may not be a symlink")
    reconciled = False
    if not verified.job_success_marker.exists():
        if not verify_live_job_environment:
            raise RobustnessRunError(
                "offline marker verification cannot reconcile a missing marker"
            )
        _patch_base_runner(verified, project_root=project_root)
        base_arguments = argparse.Namespace(
            fold=arguments.fold,
            profile=arguments.profile,
            attempt=arguments.attempt,
            prepared=verified.variant.root,
        )
        recovered = runner.reconcile_existing_attempt(base_arguments)
        if recovered is None:
            raise UnclaimedAttemptError(
                "job marker is absent and no published attempt can be reconciled"
            )
        run_id = _nonempty_string(
            recovered.get("run_id"), label="reconciled run_id"
        )
        artifact_value = recovered.get("artifact_path")
        if not isinstance(artifact_value, str) or not artifact_value:
            raise RobustnessRunError("reconciled attempt has no artifact path")
        artifact, success_sha = _verify_completed_run(
            run_id=run_id,
            artifact_path=Path(artifact_value),
            project_root=project_root,
        )
        _publish_job_marker(
            verified=verified,
            run_id=run_id,
            artifact=artifact,
            success_sha256=success_sha,
            project_root=project_root,
        )
        reconciled = True
    marker_path = _project_file(
        verified.job_success_marker,
        project_root=project_root,
        label="job success marker",
    )
    marker = _strict_json(marker_path, label="job success marker")
    _exact_keys(
        marker,
        frozenset({"payload", "marker_sha256"}),
        label="job success marker",
    )
    payload = marker["payload"]
    if not isinstance(payload, Mapping):
        raise RobustnessRunError("job success marker payload must be an object")
    expected_payload_keys = frozenset(
        {
            "schema_version",
            "campaign_id",
            "config_sha256",
            "run_id",
            "artifact_path",
            "artifact_success_sha256",
        }
    )
    _exact_keys(payload, expected_payload_keys, label="job success marker payload")
    marker_sha = _sha(marker["marker_sha256"], label="job marker_sha256")
    if canonical_sha256(payload) != marker_sha:
        raise RobustnessRunError("job success marker canonical SHA-256 mismatch")
    if payload["schema_version"] != 1 or payload["campaign_id"] != CAMPAIGN_ID:
        raise RobustnessRunError("job success marker campaign/schema mismatch")
    if payload["config_sha256"] != verified.job_config_sha256:
        raise RobustnessRunError("job success marker config SHA-256 mismatch")
    run_id = _nonempty_string(payload["run_id"], label="job marker run_id")
    artifact = _project_file(
        _nonempty_string(payload["artifact_path"], label="job marker artifact_path"),
        project_root=project_root,
        label="job marker artifact_path",
        require_file=False,
        require_directory=True,
    )
    verified_artifact, success_sha = _verify_completed_run(
        run_id=run_id, artifact_path=artifact, project_root=project_root
    )
    if payload["artifact_success_sha256"] != success_sha:
        raise RobustnessRunError("job marker _SUCCESS SHA-256 mismatch")
    return {
        "verified": True,
        "run_id": run_id,
        "artifact_path": str(verified_artifact),
        "config_sha256": verified.job_config_sha256,
        "job_success_marker": str(marker_path),
        "lifecycle_reconciled": reconciled,
    }


def run(
    arguments: argparse.Namespace, *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Verify every binding and only then delegate to the existing GPU runner."""

    verified = _verify_inputs(arguments, project_root=project_root)
    if verified.job_success_marker.exists() or verified.job_success_marker.is_symlink():
        raise RobustnessRunError(
            "job success marker already exists; use --verify-job-marker"
        )
    _patch_base_runner(verified, project_root=project_root)
    base_arguments = argparse.Namespace(
        fold=arguments.fold,
        profile=arguments.profile,
        attempt=arguments.attempt,
        prepared=verified.variant.root,
    )
    result = runner.reconcile_existing_attempt(base_arguments)
    if result is None:
        result = runner.run(base_arguments)
    run_id = _nonempty_string(result.get("run_id"), label="base runner run_id")
    artifact_value = result.get("artifact_path")
    if not isinstance(artifact_value, str) or not artifact_value:
        raise RobustnessRunError("base runner did not return artifact_path")
    artifact, success_sha = _verify_completed_run(
        run_id=run_id,
        artifact_path=Path(artifact_value),
        project_root=project_root,
    )
    _publish_job_marker(
        verified=verified,
        run_id=run_id,
        artifact=artifact,
        success_sha256=success_sha,
        project_root=project_root,
    )
    if arguments.profile == "pilot":
        allowed = {
            "run_id",
            "profile",
            "fold",
            "artifact_path",
            "duration_seconds",
            "peak_vram_gb",
            "projected_full_hours_per_fold",
        }
        result = {key: value for key, value in result.items() if key in allowed}
    return {
        **result,
        "config_sha256": verified.job_config_sha256,
        "job_success_marker": str(verified.job_success_marker),
    }


def abandon_incomplete_job_attempt(
    arguments: argparse.Namespace, *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Explicitly seal a dead partial attempt before an attempt-N+1 plan."""

    verified = _verify_inputs(arguments, project_root=project_root)
    if verified.job_success_marker.exists() or verified.job_success_marker.is_symlink():
        raise RobustnessRunError(
            "a job marker exists; successful attempts must be verified, not abandoned"
        )
    _patch_base_runner(verified, project_root=project_root)
    base_arguments = argparse.Namespace(
        fold=arguments.fold,
        profile=arguments.profile,
        attempt=arguments.attempt,
        prepared=verified.variant.root,
    )
    result = runner.abandon_incomplete_attempt(base_arguments)
    return {
        **result,
        "config_sha256": verified.job_config_sha256,
        "job_success_marker": str(verified.job_success_marker),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path)
    parser.add_argument(
        "--model-seed", type=int, choices=ALLOWED_MODEL_SEEDS
    )
    parser.add_argument("--fold", type=int, choices=range(4))
    parser.add_argument("--profile", choices=("pilot", "full"))
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--launch-manifest", type=Path)
    parser.add_argument("--pilot-receipt", type=Path)
    parser.add_argument("--verify-job-marker", action="store_true")
    parser.add_argument("--abandon-incomplete-attempt", action="store_true")
    config_occurrences = sum(
        value == "--config" or value.startswith("--config=") for value in raw_argv
    )
    parsed = parser.parse_args(raw_argv)
    if config_occurrences != 1:
        parser.error("exactly one --config argument is required")
    return parsed


def main() -> None:
    cli = parse_args()
    if cli.verify_job_marker and cli.abandon_incomplete_attempt:
        raise RobustnessRunError(
            "verification and explicit abandonment modes are mutually exclusive"
        )
    arguments = _materialize_cli_arguments(cli)
    try:
        if arguments.verify_job_marker:
            result = verify_job_marker(arguments)
        elif arguments.abandon_incomplete_attempt:
            result = abandon_incomplete_job_attempt(arguments)
        else:
            result = run(arguments)
    except UnclaimedAttemptError as error:
        print(
            canonical_json({"status": "unclaimed", "error": str(error)}),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(75) from error
    print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
