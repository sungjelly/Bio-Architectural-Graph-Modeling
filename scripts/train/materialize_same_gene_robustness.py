#!/usr/bin/env python3
"""Materialize immutable launch, job-plan, and pilot-receipt authorities.

This command never loads expression arrays or reports scientific arm effects.
It only binds source/config identities, constructs predictable coordinator
slots, and extracts prespecified technical controls after the wrapper's
``--verify-job-marker`` mode independently revalidates a completed pilot.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

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
from spatial_benchmark.run_archive import verify_run_bundle


CAMPAIGN_ID = "cmp_20260810_same_gene_robustness_multiverse_v1"
VARIANTS = tuple(f"V{index}" for index in range(7))
SEEDS = (20260810, 20261810, 20262810, 20263810, 20264810)
FOLDS = tuple(range(4))
PILOT_SEED = 20260810
PILOT_FOLD = 0
MINIMUM_FREE_DISK_GB = 40
MAXIMUM_PILOT_PROJECTED_HOURS = 0.25

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
DATA_BUILDER_RELATIVE_PATH = "scripts/data/prepare_same_gene_jacobian.py"
ROBUSTNESS_DATA_BUILDER_RELATIVE_PATH = (
    "scripts/data/prepare_same_gene_robustness_variants.py"
)
BASE_PREPARED_RELATIVE_PATH = (
    "data/processed/same_gene_cross_cell_jacobian_v1"
)
GRAPH_RELATIVE_PATH = "src/spatial_benchmark/same_gene_robustness.py"
BASE_GRAPH_RELATIVE_PATH = "src/spatial_benchmark/same_gene_jacobian.py"
MODEL_RELATIVE_PATH = "src/spatial_benchmark/same_gene_nonlinear.py"
PHASE_TRANSFORM_RELATIVE_PATH = (
    "src/spatial_benchmark/same_gene_phase_transforms.py"
)
ARCHIVE_RELATIVE_PATH = "src/spatial_benchmark/run_archive.py"
LAUNCHER_RELATIVE_PATH = "scripts/train/launch_same_gene_robustness.py"
MATERIALIZER_RELATIVE_PATH = "scripts/train/materialize_same_gene_robustness.py"
ANALYZER_RELATIVE_PATH = "scripts/analysis/analyze_same_gene_robustness.py"
BASE_ANALYZER_RELATIVE_PATH = "scripts/analysis/analyze_same_gene_nonlinear.py"
RESIDUALIZATION_RELATIVE_PATH = (
    "src/spatial_benchmark/same_gene_residualization.py"
)
REQUIRED_SOURCES = frozenset(
    {
        CONTRACT_RELATIVE_PATH,
        ENVIRONMENT_LOCK_RELATIVE_PATH,
        ENVIRONMENT_VERIFIER_RELATIVE_PATH,
        WRAPPER_RELATIVE_PATH,
        BASE_RUNNER_RELATIVE_PATH,
        DATA_BUILDER_RELATIVE_PATH,
        ROBUSTNESS_DATA_BUILDER_RELATIVE_PATH,
        GRAPH_RELATIVE_PATH,
        BASE_GRAPH_RELATIVE_PATH,
        MODEL_RELATIVE_PATH,
        PHASE_TRANSFORM_RELATIVE_PATH,
        ARCHIVE_RELATIVE_PATH,
        LAUNCHER_RELATIVE_PATH,
        MATERIALIZER_RELATIVE_PATH,
        ANALYZER_RELATIVE_PATH,
        BASE_ANALYZER_RELATIVE_PATH,
        RESIDUALIZATION_RELATIVE_PATH,
    }
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_VARIANT_ASSIGNMENT = re.compile(r"^(V[0-6])=(.+)$")
_SAFE_RETRY_SLOT = re.compile(r"^(V[0-6]):([0-9]+):([0-3])$")
_LAUNCH_KEYS = frozenset({"campaign_id", "contract", "sources"})
_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "plan_id",
        "working_directory",
        "source_manifest",
        "disk_path",
        "minimum_free_disk_gb",
        "projected_output_bytes",
        "data_preflight_argv",
        "environment_lock",
        "jobs",
    }
)
_PLAN_JOB_KEYS = frozenset(
    {
        "job_id",
        "argv",
        "gpu",
        "stdout_path",
        "stderr_path",
        "expected_success_marker",
        "expected_config_sha256",
        "verify_argv",
    }
)
_CONFIG_KEYS = frozenset(
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
_BUNDLE_KEYS = frozenset(
    {
        "verified",
        "run_id",
        "artifact_path",
        "success_sha256",
        "config_sha256",
        "bundle_manifest_sha256",
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


class SameGeneMaterializationError(RuntimeError):
    """Raised when an orchestration authority cannot be safely materialized."""


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _unique_yaml_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise SameGeneMaterializationError(
                "contract YAML keys must be unique strings"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_yaml_mapping
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_value(path: Path, *, label: str) -> Any:
    def reject_constant(value: str) -> None:
        raise SameGeneMaterializationError(
            f"{label} contains nonfinite JSON constant {value!r}"
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SameGeneMaterializationError(
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
    except SameGeneMaterializationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SameGeneMaterializationError(
            f"{label} is not strict readable JSON: {path}"
        ) from error


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    value = _strict_json_value(path, label=label)
    if not isinstance(value, dict):
        raise SameGeneMaterializationError(f"{label} must be a JSON object")
    return value


def _strict_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = yaml.load(
            path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader
        )
    except SameGeneMaterializationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SameGeneMaterializationError(
            f"{label} is not strict readable YAML"
        ) from error
    if not isinstance(value, dict):
        raise SameGeneMaterializationError(f"{label} must be a YAML mapping")
    return value


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    if set(value) != expected:
        raise SameGeneMaterializationError(
            f"{label} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise SameGeneMaterializationError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SameGeneMaterializationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise SameGeneMaterializationError(
            f"{label} must be finite and nonnegative"
        )
    return result


def _require_frozen_at(payload: Mapping[str, Any]) -> str:
    value = payload.get("frozen_at")
    if not isinstance(value, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ) is None:
        raise SameGeneMaterializationError(
            "contract frozen_at must be a non-null canonical UTC timestamp"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise SameGeneMaterializationError(
            "contract frozen_at is not a valid canonical UTC timestamp"
        ) from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise SameGeneMaterializationError(
            "contract frozen_at is not canonical UTC"
        )
    return value


def _project_path(
    value: str | Path,
    *,
    project_root: Path,
    label: str,
    require_file: bool = False,
    require_directory: bool = False,
) -> Path:
    root = project_root.resolve(strict=True)
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    absolute = candidate.absolute()
    try:
        unresolved = absolute.relative_to(root)
    except ValueError as error:
        raise SameGeneMaterializationError(
            f"{label} escapes project root: {candidate}"
        ) from error
    cursor = root
    for part in unresolved.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SameGeneMaterializationError(
                f"{label} may not traverse a symlink: {cursor}"
            )
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise SameGeneMaterializationError(
            f"{label} resolves outside project root"
        ) from error
    if require_file and not resolved.is_file():
        raise SameGeneMaterializationError(
            f"{label} is not a regular file: {resolved}"
        )
    if require_directory and not resolved.is_dir():
        raise SameGeneMaterializationError(
            f"{label} is not a directory: {resolved}"
        )
    return resolved


def _relative(path: Path, *, project_root: Path, must_exist: bool = True) -> str:
    resolved = path.resolve(strict=must_exist)
    try:
        return resolved.relative_to(project_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise SameGeneMaterializationError(
            f"path escapes project root: {path}"
        ) from error


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise SameGeneMaterializationError(f"output may not be a symlink: {path}")
    if path.exists():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise SameGeneMaterializationError(
            f"refusing to replace different materialized authority: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != encoded:
                raise SameGeneMaterializationError(
                    f"concurrent materialization differs at {path}"
                )
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _source_manifest_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {"path": str(row["path"]), "size": int(row["size"]), "sha": str(row["sha"])}
        for row in rows
    ]
    normalized.sort(key=lambda row: row["path"])
    return canonical_sha256(normalized)


def _module_for_source(relative: str) -> str | None:
    path = Path(relative)
    try:
        within = path.relative_to("src")
    except ValueError:
        return None
    if within.suffix != ".py":
        return None
    parts = list(within.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _source_for_module(module: str, *, project_root: Path) -> str | None:
    if module != "spatial_benchmark" and not module.startswith(
        "spatial_benchmark."
    ):
        return None
    relative = Path("src", *module.split("."))
    file_candidate = relative.with_suffix(".py")
    package_candidate = relative / "__init__.py"
    for candidate in (file_candidate, package_candidate):
        if (project_root / candidate).is_file():
            return candidate.as_posix()
    raise SameGeneMaterializationError(
        f"local imported module has no source file: {module}"
    )


def _transitive_local_sources(
    initial: Sequence[str], *, project_root: Path
) -> frozenset[str]:
    """Discover local Python imports so provenance cannot omit a dependency."""

    discovered = set(initial)
    queue = [relative for relative in initial if relative.endswith(".py")]
    visited: set[str] = set()
    while queue:
        relative = queue.pop()
        if relative in visited:
            continue
        visited.add(relative)
        path = project_root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeError, SyntaxError) as error:
            raise SameGeneMaterializationError(
                f"cannot inspect transitive source imports: {relative}"
            ) from error
        current_module = _module_for_source(relative)
        current_package = (
            (
                current_module.split(".")
                if Path(relative).name == "__init__.py"
                else current_module.split(".")[:-1]
            )
            if current_module is not None
            else []
        )
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if current_module is None or node.level > len(current_package) + 1:
                        continue
                    keep = len(current_package) - (node.level - 1)
                    prefix = current_package[:keep]
                    suffix = [] if node.module is None else node.module.split(".")
                    resolved = ".".join([*prefix, *suffix])
                    imported_modules.add(resolved)
                    if node.module is None:
                        imported_modules.update(
                            f"{resolved}.{alias.name}" for alias in node.names
                        )
                elif node.module is not None:
                    imported_modules.add(node.module)
                    if node.module == "spatial_benchmark":
                        imported_modules.update(
                            f"{node.module}.{alias.name}" for alias in node.names
                        )
        for module in imported_modules:
            source = _source_for_module(module, project_root=project_root)
            if source is not None and source not in discovered:
                discovered.add(source)
                queue.append(source)
    return frozenset(discovered)


@dataclass(frozen=True, slots=True)
class PreparedContractBinding:
    raw_fingerprint: str
    split_fingerprint: str
    base_prepared_manifest_sha256: str
    robustness_root_manifest_sha256: str
    robustness_root_processed_fingerprint: str
    variants: dict[str, dict[str, str]]


@dataclass(frozen=True, slots=True)
class LaunchIdentity:
    path: Path
    sha256: str
    contract_sha256: str
    source_manifest_sha: str
    payload: dict[str, Any]
    prepared_binding: PreparedContractBinding


@dataclass(frozen=True, slots=True)
class VariantIdentity:
    variant_id: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    integrity_manifest_sha256: str
    raw_fingerprint: str
    processed_fingerprint: str
    split_fingerprint: str
    base_prepared_manifest_sha256: str
    variant_spec_sha256: str
    robustness_root_manifest_sha256: str
    robustness_root_processed_fingerprint: str


def _prepared_contract_binding(
    contract: Mapping[str, Any],
) -> PreparedContractBinding:
    dataset = contract.get("dataset")
    if not isinstance(dataset, Mapping):
        raise SameGeneMaterializationError(
            "frozen contract dataset authority is missing"
        )
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
        raise SameGeneMaterializationError(
            "frozen contract prepared-variant authority is missing"
        )
    _exact_keys(
        prepared, _PREPARED_BINDING_KEYS, label="prepared-variant authority"
    )
    if prepared["status"] != "complete_preoutcome_identity_binding":
        raise SameGeneMaterializationError(
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
    if not isinstance(raw_variants, Mapping) or set(raw_variants) != set(VARIANTS):
        raise SameGeneMaterializationError(
            "prepared-variant authority must bind exactly V0 through V6"
        )
    variants: dict[str, dict[str, str]] = {}
    for variant_id in VARIANTS:
        authority = raw_variants[variant_id]
        if not isinstance(authority, Mapping):
            raise SameGeneMaterializationError(
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


def _contract_identity(
    path: Path, *, project_root: Path
) -> tuple[Path, str, PreparedContractBinding]:
    contract = _project_path(
        path,
        project_root=project_root,
        label="contract",
        require_file=True,
    )
    if _relative(contract, project_root=project_root) != CONTRACT_RELATIVE_PATH:
        raise SameGeneMaterializationError(
            "contract is not the canonical robustness contract"
        )
    payload = _strict_yaml(contract, label="frozen task contract")
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise SameGeneMaterializationError("contract campaign_id mismatch")
    if payload.get("status") != "frozen_preoutcome":
        raise SameGeneMaterializationError("contract is not frozen_preoutcome")
    if payload.get("launch_authorized") is not True:
        raise SameGeneMaterializationError("contract is not launch-authorized")
    _require_frozen_at(payload)
    pilots = payload.get("pilots")
    gates = pilots.get("fail_closed_gates") if isinstance(pilots, Mapping) else None
    if (
        not isinstance(gates, Mapping)
        or _finite_number(
            gates.get("maximum_projected_hours_per_full_fold"),
            label="contract pilot runtime gate",
        )
        != MAXIMUM_PILOT_PROJECTED_HOURS
    ):
        raise SameGeneMaterializationError(
            "contract pilot runtime gate is not the frozen 0.25-hour authority"
        )
    authorization = payload.get("authorization")
    if isinstance(authorization, Mapping):
        required = authorization.get("required_status_to_launch")
        if required not in {None, "frozen_preoutcome"}:
            raise SameGeneMaterializationError(
                "contract authorization status rule is inconsistent"
            )
    return contract, _sha256_file(contract), _prepared_contract_binding(payload)


def _load_source_paths(
    source_list: str | Path, *, project_root: Path
) -> tuple[str, ...]:
    source_list_path = _project_path(
        source_list,
        project_root=project_root,
        label="source list",
        require_file=True,
    )
    value = _strict_json_value(source_list_path, label="source list")
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise SameGeneMaterializationError(
            "source list must be a nonempty JSON list of relative file paths"
        )
    normalized: list[str] = []
    for index, item in enumerate(value):
        raw = Path(item)
        if (
            raw.is_absolute()
            or raw.as_posix() != item
            or item in {".", ".."}
            or ".." in raw.parts
            or "\\" in item
        ):
            raise SameGeneMaterializationError(
                f"source list entry {index} is not a normalized relative path"
            )
        path = _project_path(
            item,
            project_root=project_root,
            label=f"source list entry {index}",
            require_file=True,
        )
        normalized.append(_relative(path, project_root=project_root))
    if len(normalized) != len(set(normalized)):
        raise SameGeneMaterializationError("source list contains duplicate paths")
    required = _transitive_local_sources(
        tuple(REQUIRED_SOURCES), project_root=project_root
    )
    missing = required.difference(normalized)
    if missing:
        raise SameGeneMaterializationError(
            f"source list omits required authorities: {sorted(missing)}"
        )
    return tuple(sorted(normalized))


def _verify_frozen_environment(project_root: Path) -> dict[str, Any]:
    lock_path = _project_path(
        ENVIRONMENT_LOCK_RELATIVE_PATH,
        project_root=project_root,
        label="environment lock",
        require_file=True,
    )
    try:
        return verify_live_environment(lock_path, visibility_mode="launcher")
    except EnvironmentLockError as error:
        raise SameGeneMaterializationError(
            "live environment differs from the frozen environment lock"
        ) from error


def build_launch_manifest(
    *,
    contract: str | Path,
    source_list: str | Path,
    output: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    _verify_frozen_environment(root)
    contract_path, contract_sha, _ = _contract_identity(
        Path(contract), project_root=root
    )
    sources = _load_source_paths(source_list, project_root=root)
    if _relative(contract_path, project_root=root) not in sources:
        raise SameGeneMaterializationError("source list must include the contract")
    rows = [
        {
            "path": relative,
            "size": (root / relative).stat().st_size,
            "sha": _sha256_file(root / relative),
        }
        for relative in sources
    ]
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "contract": {
            "path": _relative(contract_path, project_root=root),
            "sha": contract_sha,
        },
        "sources": rows,
    }
    output_path = _project_path(
        output, project_root=root, label="launch output"
    )
    _atomic_json(output_path, payload)
    return payload


def _verify_launch(path: str | Path, *, project_root: Path) -> LaunchIdentity:
    launch_path = _project_path(
        path, project_root=project_root, label="launch manifest", require_file=True
    )
    payload = _strict_json(launch_path, label="launch manifest")
    _exact_keys(payload, _LAUNCH_KEYS, label="launch manifest")
    if payload["campaign_id"] != CAMPAIGN_ID:
        raise SameGeneMaterializationError("launch campaign_id mismatch")
    contract = payload["contract"]
    if not isinstance(contract, Mapping):
        raise SameGeneMaterializationError("launch contract must be an object")
    _exact_keys(contract, frozenset({"path", "sha"}), label="launch contract")
    contract_path, observed_contract_sha, prepared_binding = _contract_identity(
        _project_path(
            str(contract["path"]),
            project_root=project_root,
            label="launch contract path",
            require_file=True,
        ),
        project_root=project_root,
    )
    declared_contract_sha = _sha(contract["sha"], label="launch contract SHA")
    if declared_contract_sha != observed_contract_sha:
        raise SameGeneMaterializationError("launch contract SHA mismatch")
    sources = payload["sources"]
    if not isinstance(sources, list) or not sources:
        raise SameGeneMaterializationError("launch sources must be nonempty")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(sources):
        if not isinstance(row, Mapping):
            raise SameGeneMaterializationError("launch source row must be an object")
        _exact_keys(
            row, frozenset({"path", "size", "sha"}), label=f"launch source {index}"
        )
        relative = row["path"]
        if not isinstance(relative, str) or relative in seen:
            raise SameGeneMaterializationError("launch source paths are invalid")
        path = _project_path(
            relative,
            project_root=project_root,
            label=f"launch source {index}",
            require_file=True,
        )
        canonical_relative = _relative(path, project_root=project_root)
        if canonical_relative != relative:
            raise SameGeneMaterializationError("launch source path is not canonical")
        seen.add(relative)
        size = row["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SameGeneMaterializationError("launch source size is invalid")
        digest = _sha(row["sha"], label="launch source SHA")
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise SameGeneMaterializationError(
                f"launch source identity mismatch: {relative}"
            )
        normalized.append({"path": relative, "size": size, "sha": digest})
    required = _transitive_local_sources(
        tuple(REQUIRED_SOURCES), project_root=project_root
    )
    missing = required.difference(seen)
    if missing:
        raise SameGeneMaterializationError(
            f"launch manifest omits required authorities: {sorted(missing)}"
        )
    _verify_frozen_environment(project_root)
    return LaunchIdentity(
        path=launch_path,
        sha256=_sha256_file(launch_path),
        contract_sha256=observed_contract_sha,
        source_manifest_sha=_source_manifest_sha(normalized),
        payload=payload,
        prepared_binding=prepared_binding,
    )


def _variant_identity(
    variant_id: str,
    path: str | Path,
    *,
    project_root: Path,
    binding: PreparedContractBinding,
) -> VariantIdentity:
    if variant_id not in VARIANTS:
        raise SameGeneMaterializationError(f"unsupported variant: {variant_id}")
    root = _project_path(
        path,
        project_root=project_root,
        label=f"{variant_id} root",
        require_directory=True,
    )
    manifest = _project_path(
        root / "manifest.json",
        project_root=project_root,
        label=f"{variant_id} manifest",
        require_file=True,
    )
    payload = _strict_json(manifest, label=f"{variant_id} manifest")
    _exact_keys(payload, _VARIANT_KEYS, label=f"{variant_id} manifest")
    if payload["variant_id"] != variant_id:
        raise SameGeneMaterializationError(
            f"variant mapping key {variant_id} disagrees with manifest"
        )
    processed_fingerprint = _sha(
        payload["processed_fingerprint"],
        label=f"{variant_id} processed_fingerprint",
    )
    split_fingerprint = _sha(
        payload["split_fingerprint"], label=f"{variant_id} split_fingerprint"
    )
    base_manifest_sha = _sha(
        payload["base_prepared_manifest_sha256"],
        label=f"{variant_id} base_prepared_manifest_sha256",
    )
    raw_snapshot = payload["raw_snapshot"]
    if not isinstance(raw_snapshot, Mapping) or set(raw_snapshot) != {"fingerprint"}:
        raise SameGeneMaterializationError(
            f"{variant_id} raw_snapshot is malformed"
        )
    raw_fingerprint = _sha(
        raw_snapshot["fingerprint"], label=f"{variant_id} raw fingerprint"
    )
    if not isinstance(payload["variant_spec"], Mapping):
        raise SameGeneMaterializationError(f"{variant_id} variant_spec is malformed")
    variant_spec = payload["variant_spec"]
    required_spec = {"graph", "normalization", "node_policy", "permutation_seed"}
    if not required_spec.issubset(variant_spec):
        raise SameGeneMaterializationError(
            f"{variant_id} variant_spec lacks required fields"
        )
    for key in ("graph", "normalization", "node_policy"):
        if variant_spec[key] is None:
            raise SameGeneMaterializationError(
                f"{variant_id} variant_spec.{key} may not be null"
            )
        canonical_json(variant_spec[key])
    permutation_seed = variant_spec["permutation_seed"]
    if (
        isinstance(permutation_seed, bool)
        or not isinstance(permutation_seed, int)
        or permutation_seed < 0
    ):
        raise SameGeneMaterializationError(
            f"{variant_id} permutation_seed is invalid"
        )
    preprocessing = payload["preprocessing_version"]
    if not isinstance(preprocessing, str) or not preprocessing:
        raise SameGeneMaterializationError(
            f"{variant_id} preprocessing_version is invalid"
        )
    if root.parent.name != "variants":
        raise SameGeneMaterializationError(
            f"{variant_id} root must be inside a variants directory"
        )
    integrity_path = _project_path(
        root / "integrity_manifest.json",
        project_root=project_root,
        label=f"{variant_id} integrity manifest",
        require_file=True,
    )
    integrity_sha = _sha256_file(integrity_path)
    robustness_manifest = _project_path(
        root.parent.parent / "manifest.json",
        project_root=project_root,
        label="robustness root manifest",
        require_file=True,
    )
    robustness_payload = _strict_json(
        robustness_manifest, label="robustness root manifest"
    )
    robustness_processed = _sha(
        robustness_payload.get("processed_fingerprint"),
        label="robustness root processed fingerprint",
    )
    manifest_sha = _sha256_file(manifest)
    observed_common = {
        "raw fingerprint": raw_fingerprint,
        "split fingerprint": split_fingerprint,
        "base prepared manifest SHA": base_manifest_sha,
        "robustness root manifest SHA": _sha256_file(robustness_manifest),
        "robustness root processed fingerprint": robustness_processed,
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
            raise SameGeneMaterializationError(
                f"{variant_id} {label} differs from frozen contract"
            )
    variant_spec_sha = canonical_sha256(variant_spec)
    observed_variant = {
        "manifest_sha256": manifest_sha,
        "integrity_manifest_sha256": integrity_sha,
        "processed_fingerprint": processed_fingerprint,
        "variant_spec_sha256": variant_spec_sha,
    }
    for key, observed in observed_variant.items():
        if observed != binding.variants[variant_id][key]:
            raise SameGeneMaterializationError(
                f"{variant_id} {key} differs from frozen contract"
            )
    return VariantIdentity(
        variant_id=variant_id,
        root=root,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        integrity_manifest_sha256=integrity_sha,
        raw_fingerprint=raw_fingerprint,
        processed_fingerprint=processed_fingerprint,
        split_fingerprint=split_fingerprint,
        base_prepared_manifest_sha256=base_manifest_sha,
        variant_spec_sha256=variant_spec_sha,
        robustness_root_manifest_sha256=_sha256_file(robustness_manifest),
        robustness_root_processed_fingerprint=robustness_processed,
    )


def _validate_mapping(
    raw: Mapping[str, str | Path], *, label: str
) -> dict[str, str | Path]:
    if set(raw) != set(VARIANTS):
        raise SameGeneMaterializationError(
            f"{label} must map exactly {list(VARIANTS)}"
        )
    return dict(raw)


def _validate_receipt_identity(
    path: str | Path,
    *,
    launch: LaunchIdentity,
    variant: VariantIdentity,
    project_root: Path,
) -> Path:
    receipt_path = _project_path(
        path,
        project_root=project_root,
        label=f"{variant.variant_id} pilot receipt",
        require_file=True,
    )
    receipt = _strict_json(receipt_path, label="pilot receipt")
    _exact_keys(
        receipt, frozenset({"payload", "receipt_sha256"}), label="pilot receipt"
    )
    payload = receipt["payload"]
    if not isinstance(payload, Mapping):
        raise SameGeneMaterializationError("pilot receipt payload is malformed")
    _exact_keys(payload, _RECEIPT_PAYLOAD_KEYS, label="pilot receipt payload")
    if canonical_sha256(payload) != _sha(
        receipt["receipt_sha256"], label="pilot receipt SHA"
    ):
        raise SameGeneMaterializationError("pilot receipt digest mismatch")
    expected = {
        "campaign": CAMPAIGN_ID,
        "contract": launch.contract_sha256,
        "source_manifest_sha": launch.source_manifest_sha,
        "variant": variant.variant_id,
        "prepared_manifest_sha": variant.manifest_sha256,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise SameGeneMaterializationError(
                f"pilot receipt {key} binding mismatch"
            )
    selected_attempt = payload["selected_attempt"]
    history = payload["attempt_history"]
    if (
        isinstance(selected_attempt, bool)
        or not isinstance(selected_attempt, int)
        or selected_attempt < 1
        or not isinstance(history, list)
        or len(history) != selected_attempt
        or canonical_sha256(history)
        != _sha(
            payload["attempt_history_sha256"],
            label="pilot receipt attempt-history SHA",
        )
    ):
        raise SameGeneMaterializationError("pilot receipt attempt history is invalid")
    if [row.get("attempt") for row in history if isinstance(row, Mapping)] != list(
        range(1, selected_attempt + 1)
    ):
        raise SameGeneMaterializationError("pilot receipt attempt history is noncontiguous")
    bundle = payload["bundle"]
    controls = payload["controls"]
    if not isinstance(bundle, Mapping) or not isinstance(controls, Mapping):
        raise SameGeneMaterializationError("pilot receipt body is malformed")
    _exact_keys(bundle, _BUNDLE_KEYS, label="pilot receipt bundle")
    _exact_keys(controls, _CONTROL_KEYS, label="pilot receipt controls")
    if bundle["verified"] is not True:
        raise SameGeneMaterializationError("pilot receipt bundle is unverified")
    for key in ("run_id", "artifact_path"):
        if not isinstance(bundle[key], str) or not bundle[key]:
            raise SameGeneMaterializationError(
                f"pilot receipt bundle {key} is invalid"
            )
    for key in ("success_sha256", "config_sha256", "bundle_manifest_sha256"):
        _sha(bundle[key], label=f"pilot receipt bundle {key}")
    selected_rows = [
        row for row in history
        if isinstance(row, Mapping) and row.get("selected") is True
    ]
    if (
        len(selected_rows) != 1
        or selected_rows[0].get("attempt") != selected_attempt
        or selected_rows[0].get("registry_status") != "completed"
        or selected_rows[0].get("artifact_status") != "success"
        or selected_rows[0].get("run_id") != bundle["run_id"]
        or selected_rows[0].get("config_sha256") != bundle["config_sha256"]
        or selected_rows[0].get("artifact_path") != bundle["artifact_path"]
    ):
        raise SameGeneMaterializationError("pilot selected-attempt binding mismatch")
    for row in history[:-1]:
        if (
            not isinstance(row, Mapping)
            or row.get("selected") is not False
            or row.get("registry_status") not in {"failed", "cancelled", "pruned"}
            or row.get("artifact_status") not in {"failed", "pruned"}
        ):
            raise SameGeneMaterializationError(
                "pilot superseded attempt is not terminal unsuccessful"
            )
    if controls["all_outputs_finite"] is not True:
        raise SameGeneMaterializationError("pilot receipt finite-output gate failed")
    if controls["train_validation_test_component_overlap"] is not False:
        raise SameGeneMaterializationError("pilot receipt split-overlap gate failed")
    if controls["receiver_rna_or_derived_covariate_model_input"] is not False:
        raise SameGeneMaterializationError("pilot receipt receiver-input gate failed")
    if controls["identity_oracle_actually_executed"] is not True:
        raise SameGeneMaterializationError(
            "pilot receipt identity-oracle execution gate failed"
        )
    if _finite_number(
        controls["identity_oracle_row_top1_fraction"],
        label="pilot receipt identity oracle",
    ) != 1.0:
        raise SameGeneMaterializationError("pilot receipt identity gate failed")
    if _finite_number(
        controls["analytical_autograd_max_abs_error"],
        label="pilot receipt autograd error",
    ) > 1e-10:
        raise SameGeneMaterializationError("pilot receipt autograd gate failed")
    if _finite_number(
        controls["analytical_finite_difference_max_abs_error"],
        label="pilot receipt finite-difference error",
    ) > 1e-8:
        raise SameGeneMaterializationError(
            "pilot receipt finite-difference gate failed"
        )
    if controls["graph_specific_invariants"] is not True:
        raise SameGeneMaterializationError(
            "pilot receipt graph-invariant gate failed"
        )
    if _finite_number(
        controls["checkpoint_gpu_replay_max_abs_metric_error"],
        label="pilot receipt checkpoint metric replay",
    ) > 1e-7:
        raise SameGeneMaterializationError(
            "pilot receipt checkpoint metric-replay gate failed"
        )
    if _finite_number(
        controls["checkpoint_gpu_replay_max_abs_prediction_error"],
        label="pilot receipt checkpoint prediction replay",
    ) > 1e-7:
        raise SameGeneMaterializationError(
            "pilot receipt checkpoint prediction-replay gate failed"
        )
    if controls["checkpoint_replay_device_type"] != "cuda":
        raise SameGeneMaterializationError(
            "pilot receipt checkpoint replay was not executed on GPU"
        )
    if controls["canonical_production_split_label"] != "test":
        raise SameGeneMaterializationError(
            "pilot receipt canonical production-split gate failed"
        )
    if controls["source_config_data_hashes_verified"] is not True:
        raise SameGeneMaterializationError(
            "pilot receipt source/config/data hash gate failed"
        )
    if controls["outer_test_untouched"] is not True:
        raise SameGeneMaterializationError(
            "pilot receipt outer-test isolation gate failed"
        )
    if controls["environment_lock_verified"] is not True:
        raise SameGeneMaterializationError(
            "pilot receipt environment-lock gate failed"
        )
    if controls["environment_visibility_mode"] != "job":
        raise SameGeneMaterializationError(
            "pilot receipt environment visibility mode changed"
        )
    if _sha(
        controls["environment_lock_sha256"],
        label="pilot receipt environment-lock SHA",
    ) != _sha256_file(project_root / ENVIRONMENT_LOCK_RELATIVE_PATH):
        raise SameGeneMaterializationError(
            "pilot receipt environment-lock identity changed"
        )
    _sha(
        controls["environment_verification_sha256"],
        label="pilot receipt environment verification SHA",
    )
    if _finite_number(
        controls["peak_vram_gb"], label="pilot receipt peak VRAM"
    ) > 20.5:
        raise SameGeneMaterializationError("pilot receipt VRAM gate failed")
    if _finite_number(
        controls["projected_full_hours_per_fold"],
        label="pilot receipt projected runtime",
    ) > MAXIMUM_PILOT_PROJECTED_HOURS:
        raise SameGeneMaterializationError("pilot receipt runtime gate failed")
    if (
        controls["gate_passed"] is not True
        or controls["production_authorized"] is not True
    ):
        raise SameGeneMaterializationError("pilot receipt does not authorize production")
    return receipt_path


def _slot_iter(
    profile: str,
    selected: Sequence[tuple[str, int, int]] | None = None,
) -> Sequence[tuple[str, int, int]]:
    if profile == "pilot":
        universe = tuple((variant, PILOT_SEED, PILOT_FOLD) for variant in VARIANTS)
    elif profile == "full":
        universe = tuple(
            (variant, seed, fold)
            for variant in VARIANTS
            for seed in SEEDS
            for fold in FOLDS
        )
    else:
        raise SameGeneMaterializationError("profile must be pilot or full")
    if selected is None:
        return universe
    normalized = tuple((str(v), int(s), int(f)) for v, s, f in selected)
    if not normalized or len(set(normalized)) != len(normalized):
        raise SameGeneMaterializationError("retry slots must be nonempty and unique")
    extra = set(normalized).difference(universe)
    if extra:
        raise SameGeneMaterializationError(f"retry slots are outside the profile: {sorted(extra)}")
    wanted = set(normalized)
    return tuple(slot for slot in universe if slot in wanted)


def _pilot_bundle_size_bytes(receipt_path: Path, *, project_root: Path) -> int:
    receipt = _strict_json(receipt_path, label="pilot receipt for disk projection")
    payload = receipt.get("payload")
    bundle = payload.get("bundle") if isinstance(payload, Mapping) else None
    if not isinstance(bundle, Mapping):
        raise SameGeneMaterializationError("pilot bundle projection authority is malformed")
    artifact = _project_path(
        str(bundle.get("artifact_path", "")),
        project_root=project_root,
        label="pilot bundle projection artifact",
        require_directory=True,
    )
    manifest = artifact / "provenance/artifact_checksums.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise SameGeneMaterializationError("pilot bundle projection manifest is missing")
    if _sha256_file(manifest) != _sha(
        bundle.get("bundle_manifest_sha256"),
        label="pilot bundle projection manifest SHA",
    ):
        raise SameGeneMaterializationError("pilot bundle projection manifest changed")
    manifest_payload = _strict_json(manifest, label="pilot bundle projection manifest")
    files = manifest_payload.get("files")
    if not isinstance(files, Mapping):
        raise SameGeneMaterializationError("pilot bundle projection inventory is malformed")
    total = manifest.stat().st_size
    for relative, identity in files.items():
        if not isinstance(relative, str) or not isinstance(identity, Mapping):
            raise SameGeneMaterializationError("pilot bundle projection entry is malformed")
        if identity.get("type") == "file":
            size = identity.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise SameGeneMaterializationError("pilot bundle projection size is invalid")
            total += size
    success = artifact / "_SUCCESS"
    if success.is_file() and not success.is_symlink():
        total += success.stat().st_size
    return int(total)


def _verify_materialization_disk_projection(
    project_root: Path, *, projected_output_bytes: int
) -> None:
    if (
        isinstance(projected_output_bytes, bool)
        or not isinstance(projected_output_bytes, int)
        or projected_output_bytes < 0
    ):
        raise SameGeneMaterializationError("projected output size is invalid")
    free_bytes = int(shutil.disk_usage(project_root).free)
    floor_bytes = MINIMUM_FREE_DISK_GB * 1024**3
    ending_free = free_bytes - projected_output_bytes
    if ending_free < floor_bytes:
        raise SameGeneMaterializationError(
            "materializing this plan is blocked: current free disk minus projected "
            f"outputs leaves {ending_free / 1024**3:.2f} GiB, below the frozen "
            f"{MINIMUM_FREE_DISK_GB} GiB ending-free floor"
        )


def _job_id(
    profile: str, variant: str, seed: int, fold: int, attempt: int = 1
) -> str:
    return f"same-gene-robustness-{profile}-{variant}-s{seed}-f{fold}-a{attempt}"


def build_job_plan(
    *,
    profile: str,
    variant_roots: Mapping[str, str | Path],
    launch_manifest: str | Path,
    output_dir: str | Path,
    plan: str | Path,
    variant_receipts: Mapping[str, str | Path] | None = None,
    project_root: str | Path = PROJECT_ROOT,
    python_executable: str = sys.executable,
    attempt: int = 1,
    retry_slots: Sequence[tuple[str, int, int]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    if profile not in {"pilot", "full"}:
        raise SameGeneMaterializationError("profile must be pilot or full")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise SameGeneMaterializationError("attempt must be a positive integer")
    if attempt == 1 and retry_slots is not None:
        raise SameGeneMaterializationError("attempt 1 may not declare retry slots")
    if attempt > 1 and retry_slots is None:
        raise SameGeneMaterializationError(
            "attempt >1 requires explicit retry slots; broad implicit retry is forbidden"
        )
    launch = _verify_launch(launch_manifest, project_root=root)
    roots = _validate_mapping(variant_roots, label="variant roots")
    variants = {
        variant: _variant_identity(
            variant,
            roots[variant],
            project_root=root,
            binding=launch.prepared_binding,
        )
        for variant in VARIANTS
    }
    robustness_roots = {identity.root.parent.parent for identity in variants.values()}
    if len(robustness_roots) != 1:
        raise SameGeneMaterializationError(
            "all core variants must share one robustness prepared root"
        )
    robustness_root = next(iter(robustness_roots))
    receipt_paths: dict[str, Path] = {}
    if profile == "full":
        if variant_receipts is None:
            raise SameGeneMaterializationError(
                "full profile requires a receipt mapping for every variant"
            )
        receipts = _validate_mapping(
            variant_receipts, label="variant pilot receipts"
        )
        receipt_paths = {
            variant: _validate_receipt_identity(
                receipts[variant],
                launch=launch,
                variant=variants[variant],
                project_root=root,
            )
            for variant in VARIANTS
        }
    elif variant_receipts:
        raise SameGeneMaterializationError(
            "pilot plans may not consume production receipts"
        )

    materialized_root = _project_path(
        output_dir, project_root=root, label="output directory"
    )
    materialized_root.mkdir(parents=True, exist_ok=True)
    if not materialized_root.is_dir():
        raise SameGeneMaterializationError("output directory could not be created")
    plan_path = _project_path(plan, project_root=root, label="plan output")
    if not isinstance(python_executable, str) or not python_executable:
        raise SameGeneMaterializationError("python executable is invalid")
    if os.path.realpath(python_executable) != os.path.realpath(sys.executable):
        raise SameGeneMaterializationError(
            "python executable must match the materializer interpreter"
        )

    jobs: list[dict[str, Any]] = []
    launch_relative = _relative(launch.path, project_root=root)
    wrapper_relative = WRAPPER_RELATIVE_PATH
    planned_slots = tuple(_slot_iter(profile, retry_slots))
    projected_output_bytes = 0
    if profile == "full":
        pilot_sizes = {
            variant: _pilot_bundle_size_bytes(
                receipt_paths[variant], project_root=root
            )
            for variant in VARIANTS
        }
        raw_projection = sum(
            pilot_sizes[variant] for variant, _seed, _fold in planned_slots
        )
        # A fixed 5 GiB or 20% (whichever is larger) covers coordinator logs,
        # SQLite/WAL growth, filesystem allocation granularity, and bundle-size
        # variance while retaining the independent 40 GiB ending-free floor.
        safety_margin = max(5 * 1024**3, math.ceil(raw_projection * 0.20))
        projected_output_bytes = int(raw_projection + safety_margin)
    # Gate before writing any attempt-specific config or plan authority. The
    # launcher repeats the check and then scales it to remaining jobs.
    _verify_materialization_disk_projection(
        root, projected_output_bytes=projected_output_bytes
    )
    for variant, seed, fold in planned_slots:
        stem = f"{variant.lower()}-s{seed}-f{fold}-a{attempt}"
        config_path = materialized_root / "configs" / profile / f"{stem}.json"
        marker_path = materialized_root / "markers" / profile / f"{stem}.json"
        stdout_path = materialized_root / "logs" / profile / f"{stem}.stdout.log"
        stderr_path = materialized_root / "logs" / profile / f"{stem}.stderr.log"
        config_relative = _relative(
            config_path, project_root=root, must_exist=False
        )
        marker_relative = _relative(
            marker_path, project_root=root, must_exist=False
        )
        config = {
            "schema_version": 1,
            "variant_root": _relative(variants[variant].root, project_root=root),
            "model_seed": seed,
            "fold": fold,
            "profile": profile,
            "attempt": attempt,
            "launch_manifest": launch_relative,
            "pilot_receipt": (
                _relative(receipt_paths[variant], project_root=root)
                if profile == "full"
                else None
            ),
            "job_success_marker": marker_relative,
        }
        _atomic_json(config_path, config)
        config_sha = _sha256_file(config_path)
        argv = [python_executable, wrapper_relative, "--config", config_relative]
        jobs.append(
            {
                "job_id": _job_id(profile, variant, seed, fold, attempt),
                "argv": argv,
                "gpu": "auto",
                "stdout_path": _relative(
                    stdout_path, project_root=root, must_exist=False
                ),
                "stderr_path": _relative(
                    stderr_path, project_root=root, must_exist=False
                ),
                "expected_success_marker": marker_relative,
                "expected_config_sha256": config_sha,
                "verify_argv": [*argv, "--verify-job-marker"],
            }
        )
    payload = {
        "schema_version": 1,
        "plan_id": (
            f"{CAMPAIGN_ID}-{profile}-core-v1"
            if attempt == 1
            else f"{CAMPAIGN_ID}-{profile}-retry-a{attempt}-v1"
        ),
        "working_directory": ".",
        "source_manifest": {
            "path": launch_relative,
            "sha256": launch.sha256,
        },
        "environment_lock": {
            "path": ENVIRONMENT_LOCK_RELATIVE_PATH,
            "sha256": _sha256_file(root / ENVIRONMENT_LOCK_RELATIVE_PATH),
        },
        "disk_path": ".",
        "minimum_free_disk_gb": MINIMUM_FREE_DISK_GB,
        "projected_output_bytes": projected_output_bytes,
        "data_preflight_argv": [
            python_executable,
            ROBUSTNESS_DATA_BUILDER_RELATIVE_PATH,
            "--base",
            BASE_PREPARED_RELATIVE_PATH,
            "--output",
            _relative(robustness_root, project_root=root),
            "--verify-only",
        ],
        "jobs": jobs,
    }
    _atomic_json(plan_path, payload)
    return payload


def _load_job_config(path: Path, *, project_root: Path) -> dict[str, Any]:
    config_path = _project_path(
        path, project_root=project_root, label="job config", require_file=True
    )
    payload = _strict_json(config_path, label="job config")
    _exact_keys(payload, _CONFIG_KEYS, label="job config")
    attempt = payload["attempt"]
    if (
        payload["schema_version"] != 1
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
    ):
        raise SameGeneMaterializationError("job config schema/attempt mismatch")
    return payload


@dataclass(frozen=True, slots=True)
class PilotSlot:
    variant: VariantIdentity
    job_id: str
    config_path: Path
    config_sha256: str
    marker_path: Path
    verify_argv: tuple[str, ...]
    launch: LaunchIdentity
    attempt: int
    plan_path: Path


def _pilot_slots(plan_path: Path, *, project_root: Path) -> tuple[PilotSlot, ...]:
    path = _project_path(
        plan_path, project_root=project_root, label="pilot plan", require_file=True
    )
    payload = _strict_json(path, label="pilot plan")
    _exact_keys(payload, _PLAN_KEYS, label="pilot plan")
    minimum_disk = _finite_number(
        payload["minimum_free_disk_gb"], label="pilot plan disk floor"
    )
    plan_id = payload.get("plan_id")
    retry_match = (
        re.fullmatch(
            rf"{re.escape(CAMPAIGN_ID)}-pilot-retry-a([2-9][0-9]*)-v1",
            str(plan_id),
        )
        if plan_id is not None
        else None
    )
    plan_attempt = 1 if plan_id == f"{CAMPAIGN_ID}-pilot-core-v1" else (
        int(retry_match.group(1)) if retry_match is not None else 0
    )
    if (
        payload["schema_version"] != 1
        or plan_attempt < 1
        or payload["working_directory"] != "."
        or payload["disk_path"] != "."
        or payload["projected_output_bytes"] != 0
        or minimum_disk < MINIMUM_FREE_DISK_GB
    ):
        raise SameGeneMaterializationError("pilot plan top-level contract mismatch")
    source = payload["source_manifest"]
    if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
        raise SameGeneMaterializationError("pilot plan source manifest is malformed")
    launch = _verify_launch(str(source["path"]), project_root=project_root)
    if launch.sha256 != _sha(source["sha256"], label="plan source SHA"):
        raise SameGeneMaterializationError("pilot plan launch SHA mismatch")
    environment = payload["environment_lock"]
    if (
        not isinstance(environment, Mapping)
        or set(environment) != {"path", "sha256"}
        or environment["path"] != ENVIRONMENT_LOCK_RELATIVE_PATH
        or _sha256_file(project_root / ENVIRONMENT_LOCK_RELATIVE_PATH)
        != _sha(environment["sha256"], label="plan environment-lock SHA")
    ):
        raise SameGeneMaterializationError("pilot environment-lock authority changed")
    jobs = payload["jobs"]
    if (
        not isinstance(jobs, list)
        or not jobs
        or len(jobs) > len(VARIANTS)
        or (plan_attempt == 1 and len(jobs) != len(VARIANTS))
    ):
        raise SameGeneMaterializationError(
            "initial pilot plan needs seven jobs; retry plans need one to seven"
        )

    slots: list[PilotSlot] = []
    observed: set[tuple[str, int, int]] = set()
    seen_job_ids: set[str] = set()
    seen_markers: set[Path] = set()
    for index, raw_job in enumerate(jobs):
        if not isinstance(raw_job, Mapping):
            raise SameGeneMaterializationError("pilot plan job must be an object")
        _exact_keys(raw_job, _PLAN_JOB_KEYS, label=f"pilot job {index}")
        job_id = raw_job["job_id"]
        if not isinstance(job_id, str) or job_id in seen_job_ids:
            raise SameGeneMaterializationError("pilot plan has duplicate job IDs")
        seen_job_ids.add(job_id)
        argv = raw_job["argv"]
        verify_argv = raw_job["verify_argv"]
        if not isinstance(argv, list) or not all(isinstance(v, str) for v in argv):
            raise SameGeneMaterializationError("pilot argv is malformed")
        if (
            len(argv) != 4
            or os.path.realpath(argv[0]) != os.path.realpath(sys.executable)
            or argv[1] != WRAPPER_RELATIVE_PATH
            or argv[2] != "--config"
            or not argv[0]
        ):
            raise SameGeneMaterializationError(
                "pilot argv must be exactly python wrapper --config PATH"
            )
        expected_verify = [*argv, "--verify-job-marker"]
        if verify_argv != expected_verify:
            raise SameGeneMaterializationError("pilot verify argv was tampered")
        config_path = _project_path(
            argv[3],
            project_root=project_root,
            label="pilot config path",
            require_file=True,
        )
        config_sha = _sha256_file(config_path)
        if config_sha != _sha(
            raw_job["expected_config_sha256"], label="expected config SHA"
        ):
            raise SameGeneMaterializationError("pilot config SHA mismatch")
        config = _load_job_config(config_path, project_root=project_root)
        if (
            config["profile"] != "pilot"
            or config["model_seed"] != PILOT_SEED
            or config["fold"] != PILOT_FOLD
            or config["pilot_receipt"] is not None
            or config["attempt"] != plan_attempt
            or config["launch_manifest"] != _relative(
                launch.path, project_root=project_root
            )
        ):
            raise SameGeneMaterializationError("pilot config slot mismatch")
        variant_root = _project_path(
            str(config["variant_root"]),
            project_root=project_root,
            label="pilot variant root",
            require_directory=True,
        )
        manifest_payload = _strict_json(
            variant_root / "manifest.json", label="pilot variant manifest"
        )
        variant_id = manifest_payload.get("variant_id")
        if variant_id not in VARIANTS:
            raise SameGeneMaterializationError("pilot variant ID is invalid")
        variant = _variant_identity(
            str(variant_id),
            variant_root,
            project_root=project_root,
            binding=launch.prepared_binding,
        )
        slot = (variant.variant_id, config["model_seed"], config["fold"])
        if slot in observed:
            raise SameGeneMaterializationError("pilot plan has a duplicate slot")
        observed.add(slot)
        expected_job_id = _job_id(
            "pilot", variant.variant_id, PILOT_SEED, PILOT_FOLD, plan_attempt
        )
        if job_id != expected_job_id or raw_job["gpu"] != "auto":
            raise SameGeneMaterializationError("pilot job identity/GPU was tampered")
        stem = (
            f"{variant.variant_id.lower()}-s{PILOT_SEED}-f{PILOT_FOLD}"
            f"-a{plan_attempt}"
        )
        if config_path.name != f"{stem}.json" or config_path.parent.name != "pilot":
            raise SameGeneMaterializationError("pilot config path is not predictable")
        materialized_root = config_path.parents[2]
        expected_marker_path = materialized_root / "markers/pilot" / f"{stem}.json"
        expected_stdout = materialized_root / "logs/pilot" / f"{stem}.stdout.log"
        expected_stderr = materialized_root / "logs/pilot" / f"{stem}.stderr.log"
        marker_path = _project_path(
            str(raw_job["expected_success_marker"]),
            project_root=project_root,
            label="pilot marker",
        )
        if str(config["job_success_marker"]) != _relative(
            marker_path, project_root=project_root, must_exist=False
        ):
            raise SameGeneMaterializationError("pilot marker/config mismatch")
        if (
            marker_path != expected_marker_path
            or raw_job["stdout_path"]
            != _relative(expected_stdout, project_root=project_root, must_exist=False)
            or raw_job["stderr_path"]
            != _relative(expected_stderr, project_root=project_root, must_exist=False)
        ):
            raise SameGeneMaterializationError(
                "pilot marker/log paths are not the predictable slot paths"
            )
        if marker_path in seen_markers:
            raise SameGeneMaterializationError("pilot plan has duplicate markers")
        seen_markers.add(marker_path)
        slots.append(
            PilotSlot(
                variant=variant,
                job_id=job_id,
                config_path=config_path,
                config_sha256=config_sha,
                marker_path=marker_path,
                verify_argv=tuple(verify_argv),
                launch=launch,
                attempt=plan_attempt,
                plan_path=path,
            )
        )
    expected = {(variant, PILOT_SEED, PILOT_FOLD) for variant in VARIANTS}
    if plan_attempt == 1 and observed != expected:
        raise SameGeneMaterializationError("pilot plan has missing or extra slots")
    prepared_roots = {slot.variant.root.parent.parent for slot in slots}
    if len(prepared_roots) != 1:
        raise SameGeneMaterializationError("pilot variants do not share one data root")
    expected_data_preflight = [
        sys.executable,
        ROBUSTNESS_DATA_BUILDER_RELATIVE_PATH,
        "--base",
        BASE_PREPARED_RELATIVE_PATH,
        "--output",
        _relative(next(iter(prepared_roots)), project_root=project_root),
        "--verify-only",
    ]
    if payload["data_preflight_argv"] != expected_data_preflight:
        raise SameGeneMaterializationError("pilot data preflight was tampered")
    return tuple(sorted(slots, key=lambda slot: slot.variant.variant_id))


def _verify_checksum_manifest(artifact: Path) -> Path:
    manifest_path = artifact / "provenance/artifact_checksums.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise SameGeneMaterializationError("pilot bundle manifest is missing")
    payload = _strict_json(manifest_path, label="pilot bundle manifest")
    if set(payload) != {"version", "files"} or payload["version"] != 1:
        raise SameGeneMaterializationError("pilot bundle manifest is malformed")
    files = payload["files"]
    if not isinstance(files, Mapping) or "results.json" not in files:
        raise SameGeneMaterializationError(
            "pilot bundle manifest does not bind results.json"
        )
    for relative, identity in files.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(identity, Mapping)
            or set(identity) != {"type", "size", "sha256"}
            or identity["type"] != "file"
        ):
            raise SameGeneMaterializationError("pilot bundle entry is unsafe")
        target = artifact / relative
        size = identity["size"]
        if (
            target.is_symlink()
            or not target.is_file()
            or isinstance(size, bool)
            or not isinstance(size, int)
            or target.stat().st_size != size
            or _sha256_file(target)
            != _sha(identity["sha256"], label="pilot bundle file SHA")
        ):
            raise SameGeneMaterializationError("pilot bundle file identity mismatch")
    return manifest_path


def _marker_bundle(
    slot: PilotSlot, *, project_root: Path
) -> tuple[dict[str, Any], Path, str]:
    marker = _strict_json(slot.marker_path, label="pilot coordinator marker")
    _exact_keys(
        marker,
        frozenset({"payload", "marker_sha256"}),
        label="pilot coordinator marker",
    )
    payload = marker["payload"]
    if not isinstance(payload, Mapping):
        raise SameGeneMaterializationError("pilot marker payload is malformed")
    expected_keys = frozenset(
        {
            "schema_version",
            "campaign_id",
            "config_sha256",
            "run_id",
            "artifact_path",
            "artifact_success_sha256",
        }
    )
    _exact_keys(payload, expected_keys, label="pilot marker payload")
    if canonical_sha256(payload) != _sha(
        marker["marker_sha256"], label="pilot marker SHA"
    ):
        raise SameGeneMaterializationError("pilot marker digest mismatch")
    if (
        payload["schema_version"] != 1
        or payload["campaign_id"] != CAMPAIGN_ID
        or payload["config_sha256"] != slot.config_sha256
        or not isinstance(payload["run_id"], str)
        or not payload["run_id"]
    ):
        raise SameGeneMaterializationError("pilot marker binding mismatch")
    artifact = _project_path(
        str(payload["artifact_path"]),
        project_root=project_root,
        label="pilot artifact",
        require_directory=True,
    )
    success = _project_path(
        artifact / "_SUCCESS",
        project_root=project_root,
        label="pilot _SUCCESS",
        require_file=True,
    )
    success_sha = _sha256_file(success)
    if success_sha != _sha(
        payload["artifact_success_sha256"], label="pilot success SHA"
    ):
        raise SameGeneMaterializationError("pilot marker success SHA mismatch")
    return dict(payload), artifact, success_sha


def _technical_controls(
    results_path: Path, *, run_id: str, project_root: Path
) -> dict[str, Any]:
    results = _strict_json(results_path, label="pilot results")
    if (
        results.get("run_id") != run_id
        or results.get("status") != "completed"
        or results.get("profile") != "pilot"
        or results.get("statistical_evaluation_role") != "resource_validation"
    ):
        raise SameGeneMaterializationError("pilot results identity/role mismatch")
    raw = results.get("controls")
    if not isinstance(raw, Mapping):
        raise SameGeneMaterializationError("pilot technical controls are missing")
    analytical = raw.get("analytical_nonlinear_jacobian")
    if not isinstance(analytical, Mapping) or analytical.get("passed") is not True:
        raise SameGeneMaterializationError("pilot analytical control failed")
    controls = {
        "all_outputs_finite": raw.get("all_outputs_finite"),
        "train_validation_test_component_overlap": raw.get(
            "train_validation_test_component_overlap"
        ),
        "receiver_rna_or_derived_covariate_model_input": raw.get(
            "receiver_rna_or_derived_covariate_model_input"
        ),
        "identity_oracle_row_top1_fraction": raw.get(
            "identity_oracle_row_top1_fraction"
        ),
        "identity_oracle_actually_executed": raw.get(
            "identity_oracle_actually_executed"
        ),
        "analytical_autograd_max_abs_error": analytical.get(
            "maximum_autograd_error"
        ),
        "analytical_finite_difference_max_abs_error": analytical.get(
            "maximum_finite_difference_error"
        ),
        "graph_specific_invariants": raw.get("graph_specific_invariants"),
        "checkpoint_gpu_replay_max_abs_metric_error": raw.get(
            "checkpoint_gpu_replay_max_abs_metric_error"
        ),
        "checkpoint_gpu_replay_max_abs_prediction_error": raw.get(
            "checkpoint_gpu_replay_max_abs_prediction_error"
        ),
        "checkpoint_replay_device_type": raw.get(
            "checkpoint_replay_device_type"
        ),
        "canonical_production_split_label": raw.get(
            "canonical_production_split_label"
        ),
        "source_config_data_hashes_verified": raw.get(
            "source_config_data_hashes_verified"
        ),
        "outer_test_untouched": raw.get("outer_test_untouched"),
        "peak_vram_gb": raw.get("peak_vram_gb"),
        "projected_full_hours_per_fold": raw.get(
            "projected_full_hours_per_fold"
        ),
        "environment_lock_verified": raw.get("environment_lock_verified"),
        "environment_lock_sha256": raw.get("environment_lock_sha256"),
        "environment_verification_sha256": raw.get(
            "environment_verification_sha256"
        ),
        "environment_visibility_mode": raw.get("environment_visibility_mode"),
    }
    if controls["all_outputs_finite"] is not True:
        raise SameGeneMaterializationError("pilot finite-output control failed")
    if controls["train_validation_test_component_overlap"] is not False:
        raise SameGeneMaterializationError("pilot split-overlap control failed")
    if controls["receiver_rna_or_derived_covariate_model_input"] is not False:
        raise SameGeneMaterializationError("pilot receiver-input control failed")
    if controls["identity_oracle_actually_executed"] is not True:
        raise SameGeneMaterializationError(
            "pilot identity-oracle execution control failed"
        )
    if _finite_number(
        controls["identity_oracle_row_top1_fraction"], label="pilot identity oracle"
    ) != 1.0:
        raise SameGeneMaterializationError("pilot identity-oracle control failed")
    if _finite_number(
        controls["analytical_autograd_max_abs_error"], label="pilot autograd error"
    ) > 1e-10:
        raise SameGeneMaterializationError("pilot autograd control failed")
    if _finite_number(
        controls["analytical_finite_difference_max_abs_error"],
        label="pilot finite-difference error",
    ) > 1e-8:
        raise SameGeneMaterializationError("pilot finite-difference control failed")
    if controls["graph_specific_invariants"] is not True:
        raise SameGeneMaterializationError("pilot graph-invariant control failed")
    if _finite_number(
        controls["checkpoint_gpu_replay_max_abs_metric_error"],
        label="pilot checkpoint metric replay",
    ) > 1e-7:
        raise SameGeneMaterializationError(
            "pilot checkpoint metric-replay control failed"
        )
    if _finite_number(
        controls["checkpoint_gpu_replay_max_abs_prediction_error"],
        label="pilot checkpoint prediction replay",
    ) > 1e-7:
        raise SameGeneMaterializationError(
            "pilot checkpoint prediction-replay control failed"
        )
    if controls["checkpoint_replay_device_type"] != "cuda":
        raise SameGeneMaterializationError(
            "pilot checkpoint replay was not executed on GPU"
        )
    if controls["canonical_production_split_label"] != "test":
        raise SameGeneMaterializationError(
            "pilot canonical production-split control failed"
        )
    if controls["source_config_data_hashes_verified"] is not True:
        raise SameGeneMaterializationError(
            "pilot source/config/data hash control failed"
        )
    if controls["outer_test_untouched"] is not True:
        raise SameGeneMaterializationError(
            "pilot outer-test isolation control failed"
        )
    if controls["environment_lock_verified"] is not True:
        raise SameGeneMaterializationError(
            "pilot environment-lock control failed"
        )
    if controls["environment_visibility_mode"] != "job":
        raise SameGeneMaterializationError(
            "pilot environment visibility mode changed"
        )
    if _sha(
        controls["environment_lock_sha256"],
        label="pilot environment-lock SHA",
    ) != _sha256_file(project_root / ENVIRONMENT_LOCK_RELATIVE_PATH):
        raise SameGeneMaterializationError(
            "pilot environment-lock identity changed"
        )
    _sha(
        controls["environment_verification_sha256"],
        label="pilot environment verification SHA",
    )
    if _finite_number(controls["peak_vram_gb"], label="pilot peak VRAM") > 20.5:
        raise SameGeneMaterializationError("pilot VRAM control failed")
    if _finite_number(
        controls["projected_full_hours_per_fold"], label="pilot projected runtime"
    ) > MAXIMUM_PILOT_PROJECTED_HOURS:
        raise SameGeneMaterializationError("pilot runtime control failed")
    return {**controls, "gate_passed": True, "production_authorized": True}


def _terminal_unsuccessful_attempt(
    slot: PilotSlot, *, project_root: Path
) -> dict[str, Any]:
    """Verify one superseded pilot attempt is terminal and immutable."""

    registry_path = project_root / "state/tracking/bagm.sqlite3"
    if registry_path.is_symlink() or not registry_path.is_file():
        raise SameGeneMaterializationError(
            "pilot retry lineage requires the authoritative registry"
        )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{registry_path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT run_id, status, artifact_path, config_json
            FROM runs
            WHERE campaign_id = ? AND seed = ? AND fold = ? AND attempt = ?
            """,
            (
                CAMPAIGN_ID,
                PILOT_SEED % 1_000_000,
                PILOT_FOLD,
                slot.attempt,
            ),
        ).fetchall()
    except sqlite3.Error as error:
        raise SameGeneMaterializationError(
            "pilot retry lineage registry query failed"
        ) from error
    finally:
        if connection is not None:
            connection.close()
    matches: list[sqlite3.Row] = []
    for row in rows:
        try:
            configuration = json.loads(str(row["config_json"]))
        except (TypeError, ValueError) as error:
            raise SameGeneMaterializationError(
                "pilot retry lineage has malformed registry configuration"
            ) from error
        if not isinstance(configuration, Mapping):
            continue
        robustness = configuration.get("robustness_variant")
        classification = configuration.get("classification")
        if (
            isinstance(robustness, Mapping)
            and robustness.get("variant_id") == slot.variant.variant_id
            and isinstance(classification, Mapping)
            and classification.get("variant_label")
            == f"same_gene_robustness_{slot.variant.variant_id}_pilot"
        ):
            matches.append(row)
    if len(matches) != 1:
        raise SameGeneMaterializationError(
            f"pilot retry lineage needs exactly one registry row for "
            f"{slot.variant.variant_id} attempt {slot.attempt}"
        )
    row = matches[0]
    status = str(row["status"])
    if status not in {"failed", "cancelled", "pruned"}:
        raise SameGeneMaterializationError(
            f"superseded pilot attempt is not terminal unsuccessful: {status}"
        )
    artifact_raw = row["artifact_path"]
    if not isinstance(artifact_raw, str) or not artifact_raw:
        raise SameGeneMaterializationError(
            "superseded pilot attempt has no immutable artifact bundle"
        )
    artifact = _project_path(
        artifact_raw,
        project_root=project_root,
        label="superseded pilot artifact",
        require_directory=True,
    )
    if artifact.is_symlink() or not any(
        (artifact / marker).is_file() for marker in ("_FAILED", "_PRUNED")
    ):
        raise SameGeneMaterializationError(
            "superseded pilot attempt lacks a failure/pruned completion marker"
        )
    try:
        verified = verify_run_bundle(artifact, require_success_contract=False)
    except Exception as error:
        raise SameGeneMaterializationError(
            "superseded pilot artifact verification failed"
        ) from error
    return {
        "run_id": str(row["run_id"]),
        "registry_status": status,
        "artifact_path": _relative(artifact, project_root=project_root),
        "artifact_status": str(verified["status"]),
    }


def build_pilot_receipts(
    *,
    pilot_plan: str | Path | Sequence[str | Path],
    output_dir: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    verification_timeout_seconds: float = 300.0,
) -> dict[str, Path]:
    root = Path(project_root).resolve(strict=True)
    if (
        not math.isfinite(verification_timeout_seconds)
        or verification_timeout_seconds <= 0
    ):
        raise SameGeneMaterializationError("verification timeout must be positive")
    plan_values = (
        tuple(pilot_plan)
        if isinstance(pilot_plan, Sequence) and not isinstance(pilot_plan, (str, Path))
        else (pilot_plan,)
    )
    if not plan_values:
        raise SameGeneMaterializationError("at least one pilot plan is required")
    declared_slots = tuple(
        slot
        for value in plan_values
        for slot in _pilot_slots(Path(value), project_root=root)
    )
    launch_shas = {slot.launch.sha256 for slot in declared_slots}
    if len(launch_shas) != 1:
        raise SameGeneMaterializationError(
            "pilot attempt plans do not share one launch authority"
        )
    histories: dict[str, list[PilotSlot]] = {variant: [] for variant in VARIANTS}
    seen_declarations: set[tuple[str, int]] = set()
    for slot in declared_slots:
        key = (slot.variant.variant_id, slot.attempt)
        if key in seen_declarations:
            raise SameGeneMaterializationError(
                f"pilot attempt is declared more than once: {key}"
            )
        seen_declarations.add(key)
        histories[slot.variant.variant_id].append(slot)
    for variant, history in histories.items():
        attempts = sorted(slot.attempt for slot in history)
        if not attempts or attempts != list(range(1, max(attempts) + 1)):
            raise SameGeneMaterializationError(
                f"pilot attempt history is missing/noncontiguous for {variant}: {attempts}"
            )
    slots = tuple(
        max(histories[variant], key=lambda slot: slot.attempt)
        for variant in VARIANTS
    )
    receipt_root = _project_path(
        output_dir, project_root=root, label="receipt output directory"
    )
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, Path] = {}
    global_history: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    seen_artifacts: set[Path] = set()
    verification_environment = dict(os.environ)
    verification_environment.update(
        {
            # Marker reconciliation deliberately replays the wrapper's job-mode
            # environment gate on one real GPU.  Hiding CUDA here would make
            # every valid pilot receipt fail its required one-GPU smoke check.
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(root / "src"),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
        }
    )
    for slot in slots:
        if slot.marker_path.is_symlink() or not slot.marker_path.is_file():
            raise SameGeneMaterializationError(
                f"pilot marker is missing for {slot.variant.variant_id}"
            )
        try:
            verification = subprocess.run(
                list(slot.verify_argv),
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=verification_environment,
                shell=False,
                check=False,
                timeout=float(verification_timeout_seconds),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SameGeneMaterializationError(
                f"wrapper marker verification could not complete for "
                f"{slot.variant.variant_id}"
            ) from error
        if verification.returncode != 0:
            raise SameGeneMaterializationError(
                f"wrapper marker verification failed for {slot.variant.variant_id}"
            )
        marker, artifact, success_sha = _marker_bundle(
            slot, project_root=root
        )
        run_id = str(marker["run_id"])
        if run_id in seen_run_ids or artifact in seen_artifacts:
            raise SameGeneMaterializationError(
                "pilot slots reuse a run_id or artifact bundle"
            )
        seen_run_ids.add(run_id)
        seen_artifacts.add(artifact)
        bundle_manifest = _verify_checksum_manifest(artifact)
        controls = _technical_controls(
            artifact / "results.json", run_id=run_id, project_root=root
        )
        lineage: list[dict[str, Any]] = []
        for declared in sorted(
            histories[slot.variant.variant_id], key=lambda value: value.attempt
        ):
            common = {
                "attempt": declared.attempt,
                "job_id": declared.job_id,
                "plan_sha256": _sha256_file(declared.plan_path),
                "config_sha256": declared.config_sha256,
            }
            if declared.attempt == slot.attempt:
                entry = {
                    **common,
                    "selected": True,
                    "registry_status": "completed",
                    "run_id": run_id,
                    "artifact_path": _relative(artifact, project_root=root),
                    "artifact_status": "success",
                }
            else:
                terminal = _terminal_unsuccessful_attempt(
                    declared, project_root=root
                )
                entry = {**common, "selected": False, **terminal}
            lineage.append(entry)
            global_history.append(
                {"variant": slot.variant.variant_id, **entry}
            )
        lineage_sha = canonical_sha256(lineage)
        payload = {
            "campaign": CAMPAIGN_ID,
            "contract": slot.launch.contract_sha256,
            "source_manifest_sha": slot.launch.source_manifest_sha,
            "variant": slot.variant.variant_id,
            "prepared_manifest_sha": slot.variant.manifest_sha256,
            "selected_attempt": slot.attempt,
            "attempt_history": lineage,
            "attempt_history_sha256": lineage_sha,
            "bundle": {
                "verified": True,
                "run_id": run_id,
                "artifact_path": _relative(artifact, project_root=root),
                "success_sha256": success_sha,
                "config_sha256": slot.config_sha256,
                "bundle_manifest_sha256": _sha256_file(bundle_manifest),
            },
            "controls": controls,
        }
        receipt = {
            "payload": payload,
            "receipt_sha256": canonical_sha256(payload),
        }
        receipt_path = receipt_root / f"{slot.variant.variant_id}.pilot-receipt.json"
        _atomic_json(receipt_path, receipt)
        receipts[slot.variant.variant_id] = receipt_path
    if set(receipts) != set(VARIANTS):
        raise SameGeneMaterializationError("pilot receipt inventory is incomplete")
    attempt_history = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "selection_rule": "latest_declared_after_terminal_unsuccessful_predecessors",
        "attempts": sorted(
            global_history, key=lambda row: (str(row["variant"]), int(row["attempt"]))
        ),
    }
    history_sha = canonical_sha256(attempt_history)
    _atomic_json(
        receipt_root / f"pilot-attempt-history-{history_sha[:16]}.json",
        {"payload": attempt_history, "history_sha256": history_sha},
    )
    return receipts


def _assignment_mapping(values: Sequence[str], *, label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        match = _SAFE_VARIANT_ASSIGNMENT.fullmatch(value)
        if match is None:
            raise SameGeneMaterializationError(
                f"{label} must use repeated V0=path assignments"
            )
        variant, raw_path = match.groups()
        if variant in result:
            raise SameGeneMaterializationError(f"duplicate {label}: {variant}")
        result[variant] = Path(raw_path)
    return result


def _retry_slots(values: Sequence[str]) -> tuple[tuple[str, int, int], ...] | None:
    if not values:
        return None
    result: list[tuple[str, int, int]] = []
    for value in values:
        match = _SAFE_RETRY_SLOT.fullmatch(value)
        if match is None:
            raise SameGeneMaterializationError(
                "retry slots must use repeated V0:SEED:FOLD assignments"
            )
        variant, seed, fold = match.groups()
        result.append((variant, int(seed), int(fold)))
    if len(set(result)) != len(result):
        raise SameGeneMaterializationError("duplicate retry slot")
    return tuple(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("build-launch")
    launch.add_argument("--contract", type=Path, required=True)
    launch.add_argument("--source-list", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)

    plan = subparsers.add_parser("build-plan")
    plan.add_argument("--profile", choices=("pilot", "full"), required=True)
    plan.add_argument("--variant-root", action="append", default=[], required=True)
    plan.add_argument("--variant-receipt", action="append", default=[])
    plan.add_argument("--launch-manifest", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--plan", type=Path, required=True)
    plan.add_argument("--attempt", type=int, default=1)
    plan.add_argument("--retry-slot", action="append", default=[])

    receipts = subparsers.add_parser("build-receipts")
    receipts.add_argument(
        "--pilot-plan", type=Path, action="append", required=True
    )
    receipts.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    root = arguments.project_root
    if arguments.command == "build-launch":
        build_launch_manifest(
            contract=arguments.contract,
            source_list=arguments.source_list,
            output=arguments.output,
            project_root=root,
        )
        summary = {"command": "build-launch", "status": "materialized"}
    elif arguments.command == "build-plan":
        roots = _assignment_mapping(arguments.variant_root, label="variant root")
        receipts = _assignment_mapping(
            arguments.variant_receipt, label="variant receipt"
        )
        payload = build_job_plan(
            profile=arguments.profile,
            variant_roots=roots,
            variant_receipts=receipts or None,
            launch_manifest=arguments.launch_manifest,
            output_dir=arguments.output_dir,
            plan=arguments.plan,
            project_root=root,
            attempt=arguments.attempt,
            retry_slots=_retry_slots(arguments.retry_slot),
        )
        summary = {
            "command": "build-plan",
            "profile": arguments.profile,
            "attempt": arguments.attempt,
            "job_count": len(payload["jobs"]),
            "status": "materialized",
        }
    else:
        paths = build_pilot_receipts(
            pilot_plan=tuple(arguments.pilot_plan),
            output_dir=arguments.output_dir,
            project_root=root,
        )
        summary = {
            "command": "build-receipts",
            "receipt_count": len(paths),
            "status": "materialized",
        }
    print(canonical_json(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
