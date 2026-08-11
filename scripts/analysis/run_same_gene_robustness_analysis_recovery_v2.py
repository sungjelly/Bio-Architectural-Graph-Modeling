#!/usr/bin/env python3
"""Run the frozen robustness analyzer through two hash-bound recovery layers.

This entry point does not replace or modify the canonical analyzer or the v1
registry adapter.  It first verifies and loads the exact v1 recovery layer,
uses that layer's authority checks and registry patch, then installs one
additional analysis-only adapter for integer-key component-coverage equality.
Both publication and ``--verify-only`` use this same entry point and preflight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "cmp_20260810_same_gene_robustness_multiverse_v1"
AMENDMENT_ID = (
    "ta_cmp_20260810_same_gene_robustness_analysis_component_coverage_adapter_v2"
)
V1_AMENDMENT_ID = (
    "ta_cmp_20260810_same_gene_robustness_analysis_registry_config_adapter_v1"
)
DEFAULT_AMENDMENT = PROJECT_ROOT / (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "analysis_component_coverage_adapter_v2.json"
)
V2_SCHEMA_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "analysis_component_coverage_adapter_v2.schema.json"
)
V2_WRAPPER_PATH = (
    "scripts/analysis/run_same_gene_robustness_analysis_recovery_v2.py"
)
V2_TEST_PATH = (
    "tests/unit/spatial_benchmark/"
    "test_same_gene_robustness_analysis_recovery_v2.py"
)
ATTEMPT_TWO_EVIDENCE_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "evidence/analysis_attempt2_component_coverage.stderr.txt"
)
V1_AMENDMENT_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "analysis_registry_config_adapter_v1.json"
)
V1_SCHEMA_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "analysis_registry_config_adapter_v1.schema.json"
)
V1_WRAPPER_PATH = (
    "scripts/analysis/run_same_gene_robustness_registry_recovery.py"
)
V1_TEST_PATH = (
    "tests/unit/spatial_benchmark/"
    "test_same_gene_robustness_analysis_registry_recovery.py"
)

EXPECTED_V1_FILES = {
    "v1_amendment": {
        "path": V1_AMENDMENT_PATH,
        "size_bytes": 5261,
        "sha256": "7e4053886d402e50d6a417242cfa62edaa6a180f7fc0e3b0b5e56da96311e684",
    },
    "v1_schema": {
        "path": V1_SCHEMA_PATH,
        "size_bytes": 9224,
        "sha256": "a0694d74664298c6d8a9ddeac6c0461f0e1b70a236c577294ec5195e93c35321",
    },
    "v1_wrapper": {
        "path": V1_WRAPPER_PATH,
        "size_bytes": 39505,
        "sha256": "1bd72017bb94ac7e372822b1f23e35de871b26c544921f45d295557d2ae2b569",
    },
    "v1_tests": {
        "path": V1_TEST_PATH,
        "size_bytes": 16852,
        "sha256": "b4ee73649d9622e00e4a8df9c75e41c0bfb6f73beff0af4bdc7ff565090c54e9",
    },
}
EXPECTED_V2_FIXED_FILES = {
    "v2_amendment_schema": {
        "path": V2_SCHEMA_PATH,
        "size_bytes": 11399,
        "sha256": "628ff60d103597e1f45091e4790755450a47d24fa365b1c504e224510b59682f",
    },
    "attempt_two_stderr_evidence": {
        "path": ATTEMPT_TWO_EVIDENCE_PATH,
        "size_bytes": 1896,
        "sha256": "6e17a70269444a009bfb495ed79cafa02df13326427821a6814c3ad05546d2a5",
    },
}
EXPECTED_V2_PATHS = {
    "v2_amendment_schema": V2_SCHEMA_PATH,
    "v2_wrapper": V2_WRAPPER_PATH,
    "v2_tests": V2_TEST_PATH,
    "attempt_two_stderr_evidence": ATTEMPT_TWO_EVIDENCE_PATH,
}
EXPECTED_PARENT_COMMIT = "5508e7b1e67e04b1987ad18cb13c6fcecf87d1ff"
EXPECTED_CORRECTION = {
    "kind": "integer_item_tuple_component_coverage_equality_adapter",
    "newly_patched_symbols": ["_verify_component_coverage", "_source_provenance"],
    "arm_fold_equality": "tuple_sorted_integer_items",
    "arm_count_equality": "tuple_sorted_integer_items",
    "integer_key_semantics_changed": False,
    "model_or_training_changed": False,
    "scientific_computation_changed": False,
    "gate_or_threshold_changed": False,
    "randomization_changed": False,
    "scientific_payload_schema_changed": False,
    "provenance_extended": True,
}
EXPECTED_EXECUTION_CONTRACT = {
    "contract_path": (
        "experiments/campaigns/"
        "cmp_20260810_same_gene_robustness_multiverse_v1/frozen_task_contract.yaml"
    ),
    "launch_manifest_path": (
        "state/materialized/same_gene_robustness_v1/recovery_r1/"
        "launch_manifest.json"
    ),
    "full_plan_path": (
        "state/materialized/same_gene_robustness_v1/recovery_r1/"
        "full/attempt1/plan.json"
    ),
    "registry_database_path": "state/tracking/bagm.sqlite3",
    "output_path": "reports/analyses/same_gene_robustness_20260811",
    "required_plan_count": 1,
    "verify_only_supported": True,
}
EXPECTED_ATTEMPTS = [
    {
        "attempt": 1,
        "recovery_layer": "pre_v1",
        "exit_code": 2,
        "failure_message": "registry row omits config_json",
        "failure_stage": (
            "first_selected_run_registry_configuration_reconciliation"
        ),
        "automated_first_run_payload_loaded": True,
        "aggregate_computed": False,
        "output_published": False,
        "staging_retained": False,
        "evidence_path": V1_AMENDMENT_PATH,
        "evidence_sha256": EXPECTED_V1_FILES["v1_amendment"]["sha256"],
    },
    {
        "attempt": 2,
        "recovery_layer": "v1",
        "started_at": "2026-08-11T04:33:27Z",
        "ended_at": "2026-08-11T05:29:31Z",
        "exit_code": 1,
        "stderr_path": ATTEMPT_TWO_EVIDENCE_PATH,
        "stderr_size_bytes": 1896,
        "stderr_sha256": EXPECTED_V2_FIXED_FILES[
            "attempt_two_stderr_evidence"
        ]["sha256"],
        "verified_run_count": 140,
        "build_payload_reached": True,
        "component_coverage_reached": True,
        "gates_computed": False,
        "bootstrap_computed": False,
        "gene_label_null_computed": False,
        "publication_started": False,
        "output_published": False,
        "staging_retained": False,
        "scientific_effect_values_printed": False,
        "scientific_effect_values_human_inspected": False,
    },
]
EXPECTED_INVARIANTS = {
    "v1_layer_unchanged": True,
    "original_48_sources_unchanged": True,
    "contract_launch_plan_ledger_registry_unchanged": True,
    "run_bundles_and_prepared_data_unchanged": True,
    "seeds_folds_models_hyperparameters_unchanged": True,
    "component_rows_unchanged": True,
    "gates_statistics_bootstrap_and_nulls_unchanged": True,
    "same_v2_wrapper_required_for_publish_and_verify_only": True,
    "no_retraining_authorized": True,
}
TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "amendment_id",
        "campaign_id",
        "status",
        "frozen_at",
        "scope",
        "parent_layer",
        "correction",
        "execution_contract",
        "failed_analysis_attempts",
        "recovery_sources",
        "scientific_invariants",
    }
)
FILE_ROW_KEYS = frozenset({"role", "path", "size_bytes", "sha256"})


class AnalysisRecoveryV2Error(RuntimeError):
    """Raised before v2 may continue with a mismatched recovery authority."""


@dataclass(frozen=True)
class RecoveryV2Authority:
    amendment_path: Path
    amendment_size_bytes: int
    amendment_sha256: str
    payload: dict[str, Any]
    parent_files: tuple[dict[str, Any], ...]
    recovery_sources: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RecoveryV2Arguments:
    amendment_path: Path
    authority_check_only: bool
    analyzer_argv: tuple[str, ...]


@dataclass(frozen=True)
class AnalyzerInvocationV2:
    contract_path: str
    launch_manifest_path: str
    plan_path: str
    output_path: str
    verify_only: bool
    argv: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(content: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AnalysisRecoveryV2Error(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise AnalysisRecoveryV2Error(f"{label} is not an object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_snapshot(path: Path, *, label: str) -> tuple[bytes, int, str]:
    try:
        before = path.stat()
        with path.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            content = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = path.stat()
    except OSError as error:
        raise AnalysisRecoveryV2Error(f"cannot read {label}") from error
    identities = {
        _stat_identity(before),
        _stat_identity(opened_before),
        _stat_identity(opened_after),
        _stat_identity(after),
    }
    if len(identities) != 1 or len(content) != before.st_size:
        raise AnalysisRecoveryV2Error(f"{label} changed while reading")
    return content, len(content), hashlib.sha256(content).hexdigest()


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise AnalysisRecoveryV2Error(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _project_file(relative: Any, *, project_root: Path, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AnalysisRecoveryV2Error(f"{label} path is not a nonempty string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or "\\" in relative:
        raise AnalysisRecoveryV2Error(f"{label} path is not canonical: {relative!r}")
    candidate = project_root.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise AnalysisRecoveryV2Error(
            f"{label} is missing, non-file, or symlink: {relative}"
        )
    try:
        resolved = candidate.resolve(strict=True)
        canonical = resolved.relative_to(project_root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as error:
        raise AnalysisRecoveryV2Error(f"{label} escapes the project root") from error
    if canonical != relative:
        raise AnalysisRecoveryV2Error(f"{label} path is not canonical: {relative!r}")
    return resolved


def _validate_file_row(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisRecoveryV2Error(f"{label} is not an object")
    _exact_keys(value, FILE_ROW_KEYS, label=label)
    role = value["role"]
    path = value["path"]
    size = value["size_bytes"]
    digest = value["sha256"]
    if not isinstance(role, str) or not role:
        raise AnalysisRecoveryV2Error(f"{label}.role is invalid")
    if not isinstance(path, str) or not path:
        raise AnalysisRecoveryV2Error(f"{label}.path is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise AnalysisRecoveryV2Error(f"{label}.size_bytes is invalid")
    if not _valid_sha256(digest):
        raise AnalysisRecoveryV2Error(f"{label}.sha256 is invalid")
    return dict(value)


def _rows_by_role(
    values: Any,
    *,
    label: str,
    expected_roles: set[str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list) or len(values) != len(expected_roles):
        raise AnalysisRecoveryV2Error(f"{label} inventory differs")
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(values):
        row = _validate_file_row(raw, label=f"{label}[{index}]")
        role = str(row["role"])
        if role in result:
            raise AnalysisRecoveryV2Error(f"duplicate {label} role: {role}")
        result[role] = row
    if set(result) != expected_roles:
        raise AnalysisRecoveryV2Error(f"{label} roles differ")
    return result


def _validate_amendment(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AnalysisRecoveryV2Error("v2 recovery amendment is not an object")
    _exact_keys(payload, TOP_LEVEL_KEYS, label="v2 recovery amendment")
    scalar_expected = {
        "schema_version": 2,
        "amendment_id": AMENDMENT_ID,
        "campaign_id": CAMPAIGN_ID,
        "status": "frozen_layered_analysis_technical_recovery",
        "scope": "component_coverage_equality_adapter_only_no_scientific_change",
    }
    for key, expected in scalar_expected.items():
        if payload.get(key) != expected:
            raise AnalysisRecoveryV2Error(f"v2 recovery amendment {key} differs")
    frozen_at = payload.get("frozen_at")
    if not isinstance(frozen_at, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        frozen_at,
    ) is None:
        raise AnalysisRecoveryV2Error("v2 recovery frozen_at is invalid")
    exact_sections = {
        "correction": EXPECTED_CORRECTION,
        "execution_contract": EXPECTED_EXECUTION_CONTRACT,
        "failed_analysis_attempts": EXPECTED_ATTEMPTS,
        "scientific_invariants": EXPECTED_INVARIANTS,
    }
    for key, expected in exact_sections.items():
        if payload.get(key) != expected:
            raise AnalysisRecoveryV2Error(f"v2 recovery {key} differs")

    parent = payload.get("parent_layer")
    if not isinstance(parent, Mapping):
        raise AnalysisRecoveryV2Error("v2 parent layer is not an object")
    _exact_keys(
        parent,
        frozenset({"git_commit", "amendment_id", "files"}),
        label="v2 parent layer",
    )
    if parent.get("git_commit") != EXPECTED_PARENT_COMMIT:
        raise AnalysisRecoveryV2Error("v2 parent git commit differs")
    if parent.get("amendment_id") != V1_AMENDMENT_ID:
        raise AnalysisRecoveryV2Error("v2 parent amendment ID differs")
    parent_rows = _rows_by_role(
        parent.get("files"),
        label="v2 parent files",
        expected_roles=set(EXPECTED_V1_FILES),
    )
    for role, expected in EXPECTED_V1_FILES.items():
        if {key: parent_rows[role][key] for key in expected} != expected:
            raise AnalysisRecoveryV2Error(f"v2 parent identity differs: {role}")

    source_rows = _rows_by_role(
        payload.get("recovery_sources"),
        label="v2 recovery sources",
        expected_roles=set(EXPECTED_V2_PATHS),
    )
    for role, path in EXPECTED_V2_PATHS.items():
        if source_rows[role]["path"] != path:
            raise AnalysisRecoveryV2Error(f"v2 recovery source path differs: {role}")
    for role, expected in EXPECTED_V2_FIXED_FILES.items():
        if {key: source_rows[role][key] for key in expected} != expected:
            raise AnalysisRecoveryV2Error(f"v2 fixed source identity differs: {role}")
    return json.loads(_canonical_json(payload))


def _verify_file_row(
    row: Mapping[str, Any], *, project_root: Path, label: str
) -> Path:
    path = _project_file(row["path"], project_root=project_root, label=label)
    _content, size, digest = _read_snapshot(path, label=label)
    if size != int(row["size_bytes"]):
        raise AnalysisRecoveryV2Error(f"{label} size differs: {row['path']}")
    if digest != row["sha256"]:
        raise AnalysisRecoveryV2Error(f"{label} SHA-256 differs: {row['path']}")
    return path


def _verify_authority(
    amendment_path: Path = DEFAULT_AMENDMENT,
    *,
    project_root: Path = PROJECT_ROOT,
) -> RecoveryV2Authority:
    root = project_root.resolve(strict=True)
    candidate = amendment_path if amendment_path.is_absolute() else root / amendment_path
    if candidate.is_symlink():
        raise AnalysisRecoveryV2Error("v2 recovery amendment may not be a symlink")
    try:
        relative = candidate.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise AnalysisRecoveryV2Error("v2 recovery amendment escapes the project") from error
    expected_relative = DEFAULT_AMENDMENT.relative_to(PROJECT_ROOT).as_posix()
    if relative != expected_relative:
        raise AnalysisRecoveryV2Error("v2 recovery amendment path differs")
    amendment = _project_file(relative, project_root=root, label="v2 recovery amendment")
    content, size, digest = _read_snapshot(
        amendment, label="v2 recovery amendment"
    )
    payload = _validate_amendment(
        _strict_json_bytes(content, label="v2 recovery amendment")
    )
    parent_files = tuple(
        sorted(
            (dict(row) for row in payload["parent_layer"]["files"]),
            key=lambda row: str(row["role"]),
        )
    )
    recovery_sources = tuple(
        sorted(
            (dict(row) for row in payload["recovery_sources"]),
            key=lambda row: str(row["role"]),
        )
    )
    for row in parent_files:
        _verify_file_row(
            row, project_root=root, label=f"v2 parent source {row['role']}"
        )
    for row in recovery_sources:
        _verify_file_row(
            row, project_root=root, label=f"v2 recovery source {row['role']}"
        )
    authority = RecoveryV2Authority(
        amendment_path=amendment,
        amendment_size_bytes=size,
        amendment_sha256=digest,
        payload=payload,
        parent_files=parent_files,
        recovery_sources=recovery_sources,
    )
    _verify_authority_unchanged(authority, project_root=root)
    return authority


def _verify_authority_unchanged(
    authority: RecoveryV2Authority,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    root = project_root.resolve(strict=True)
    try:
        relative = authority.amendment_path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise AnalysisRecoveryV2Error("v2 recovery amendment moved or escaped") from error
    expected_relative = DEFAULT_AMENDMENT.relative_to(PROJECT_ROOT).as_posix()
    if relative != expected_relative:
        raise AnalysisRecoveryV2Error("v2 recovery amendment path changed")
    path = _project_file(
        relative, project_root=root, label="v2 recovery amendment recheck"
    )
    _content, size, digest = _read_snapshot(
        path, label="v2 recovery amendment recheck"
    )
    if path != authority.amendment_path:
        raise AnalysisRecoveryV2Error("v2 recovery amendment inode path changed")
    if size != authority.amendment_size_bytes:
        raise AnalysisRecoveryV2Error("v2 recovery amendment size changed")
    if digest != authority.amendment_sha256:
        raise AnalysisRecoveryV2Error("v2 recovery amendment SHA-256 changed")
    for row in (*authority.parent_files, *authority.recovery_sources):
        _verify_file_row(
            row,
            project_root=root,
            label=f"v2 bound source recheck {row['role']}",
        )


def _row_for_role(rows: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("role") == role]
    if len(selected) != 1:
        raise AnalysisRecoveryV2Error(f"v2 authority role is not unique: {role}")
    return dict(selected[0])


def _load_v1_module(
    authority: RecoveryV2Authority,
    *,
    project_root: Path = PROJECT_ROOT,
) -> ModuleType:
    row = _row_for_role(authority.parent_files, "v1_wrapper")
    path = _verify_file_row(row, project_root=project_root, label="v1 wrapper before import")
    module_name = "same_gene_robustness_analysis_recovery_v2_parent_v1"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AnalysisRecoveryV2Error("cannot construct exact v1 recovery import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _verify_file_row(row, project_root=project_root, label="v1 wrapper after import")
    return module


def _verify_v1_layer(
    v1: ModuleType,
    authority: RecoveryV2Authority,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Any:
    required = (
        "_verify_authority",
        "_verify_authority_unchanged",
        "_parse_analyzer_invocation",
        "_verify_invocation_authority",
        "_load_analyzer",
        "_install_patches",
        "_verify_effective_registry_path",
        "_verify_public_registry_lifecycle",
    )
    if any(not callable(getattr(v1, name, None)) for name in required):
        raise AnalysisRecoveryV2Error("exact v1 recovery API surface differs")
    try:
        v1_authority = v1._verify_authority(
            project_root=project_root,
            amendment_path=project_root / V1_AMENDMENT_PATH,
        )
    except Exception as error:
        raise AnalysisRecoveryV2Error("v1 recovery authority verification failed") from error
    expected_amendment = _row_for_role(authority.parent_files, "v1_amendment")
    if (
        v1_authority.amendment_size_bytes != expected_amendment["size_bytes"]
        or v1_authority.amendment_sha256 != expected_amendment["sha256"]
    ):
        raise AnalysisRecoveryV2Error("verified v1 amendment identity differs")
    if v1_authority.payload.get("amendment_id") != V1_AMENDMENT_ID:
        raise AnalysisRecoveryV2Error("verified v1 amendment ID differs")
    if v1_authority.payload.get("execution_contract") != EXPECTED_EXECUTION_CONTRACT:
        raise AnalysisRecoveryV2Error("v1 and v2 execution contracts differ")
    expected_parent_sources = {
        row["path"]: {key: row[key] for key in ("path", "size_bytes", "sha256")}
        for row in authority.parent_files
        if row["role"] in {"v1_schema", "v1_wrapper", "v1_tests"}
    }
    observed_parent_sources = {
        row["path"]: dict(row) for row in v1_authority.recovery_sources
    }
    if observed_parent_sources != expected_parent_sources:
        raise AnalysisRecoveryV2Error("verified v1 recovery source inventory differs")
    try:
        v1._verify_authority_unchanged(v1_authority, project_root=project_root)
    except Exception as error:
        raise AnalysisRecoveryV2Error("v1 recovery authority changed") from error
    _verify_authority_unchanged(authority, project_root=project_root)
    return v1_authority


def _arguments(argv: Sequence[str] | None) -> RecoveryV2Arguments:
    values = list(sys.argv[1:] if argv is None else argv)
    amendment = DEFAULT_AMENDMENT
    amendment_seen = False
    authority_check_only = False
    analyzer_argv: list[str] = []
    index = 0
    while index < len(values):
        token = values[index]
        if token == "--recovery-v2-amendment":
            if amendment_seen:
                raise AnalysisRecoveryV2Error(
                    "--recovery-v2-amendment may be supplied only once"
                )
            if index + 1 >= len(values) or values[index + 1].startswith("--"):
                raise AnalysisRecoveryV2Error(
                    "--recovery-v2-amendment requires one path"
                )
            amendment = Path(values[index + 1])
            amendment_seen = True
            index += 2
            continue
        if token == "--authority-check-only":
            if authority_check_only:
                raise AnalysisRecoveryV2Error(
                    "--authority-check-only may be supplied only once"
                )
            authority_check_only = True
            index += 1
            continue
        if token.startswith("--recovery-v2-amendment=") or token.startswith(
            "--authority-check-only="
        ):
            raise AnalysisRecoveryV2Error(f"noncanonical v2 recovery argument: {token}")
        analyzer_argv.append(token)
        index += 1
    if authority_check_only and analyzer_argv:
        raise AnalysisRecoveryV2Error(
            "--authority-check-only does not accept analyzer arguments"
        )
    return RecoveryV2Arguments(
        amendment_path=amendment,
        authority_check_only=authority_check_only,
        analyzer_argv=tuple(analyzer_argv),
    )


def _parse_analyzer_invocation(argv: Sequence[str]) -> AnalyzerInvocationV2:
    """Fail closed on analyzer CLI before loading either analysis layer."""

    value_options = {
        "--contract": "contract_path",
        "--launch-manifest": "launch_manifest_path",
        "--plan": "plan_path",
        "--output": "output_path",
    }
    seen: dict[str, str] = {}
    verify_only = False
    values = list(argv)
    index = 0
    while index < len(values):
        token = values[index]
        if token in value_options:
            if token in seen:
                raise AnalysisRecoveryV2Error(f"duplicate analyzer authority: {token}")
            if index + 1 >= len(values) or values[index + 1].startswith("--"):
                raise AnalysisRecoveryV2Error(
                    f"analyzer argument requires one path: {token}"
                )
            seen[token] = values[index + 1]
            index += 2
            continue
        if token == "--verify-only":
            if verify_only:
                raise AnalysisRecoveryV2Error("duplicate analyzer flag: --verify-only")
            verify_only = True
            index += 1
            continue
        if token.startswith("--"):
            raise AnalysisRecoveryV2Error(f"unrecognized analyzer argument: {token}")
        raise AnalysisRecoveryV2Error(
            f"unexpected analyzer positional argument: {token}"
        )
    missing = [option for option in value_options if option not in seen]
    if missing:
        raise AnalysisRecoveryV2Error(
            f"analyzer authority arguments are missing: {sorted(missing)}"
        )
    invocation = AnalyzerInvocationV2(
        contract_path=seen["--contract"],
        launch_manifest_path=seen["--launch-manifest"],
        plan_path=seen["--plan"],
        output_path=seen["--output"],
        verify_only=verify_only,
        argv=tuple(values),
    )
    observed = {
        "contract_path": invocation.contract_path,
        "launch_manifest_path": invocation.launch_manifest_path,
        "full_plan_path": invocation.plan_path,
        "output_path": invocation.output_path,
    }
    for key, value in observed.items():
        if value != EXPECTED_EXECUTION_CONTRACT[key]:
            raise AnalysisRecoveryV2Error(f"analyzer {key} differs from v2 amendment")
    if Path.cwd().resolve(strict=True) != PROJECT_ROOT.resolve(strict=True):
        raise AnalysisRecoveryV2Error("analyzer must run from the bound project root")
    return invocation


def _same_invocation(left: AnalyzerInvocationV2, right: Any) -> bool:
    return all(
        getattr(left, field) == getattr(right, field, object())
        for field in (
            "contract_path",
            "launch_manifest_path",
            "plan_path",
            "output_path",
            "verify_only",
            "argv",
        )
    )


def _integer_item_tuple(value: Mapping[int, int]) -> tuple[tuple[int, int], ...]:
    """Canonical equality view without coercing the already-integer mapping."""

    return tuple(sorted(value.items()))


def _component_coverage_adapter(analyzer: ModuleType) -> Any:
    arms = tuple(getattr(analyzer, "ARMS", ()))
    expected_components = getattr(analyzer, "EXPECTED_COMPONENTS", None)
    error_type = getattr(analyzer, "RobustnessAnalysisError", None)
    if not arms or isinstance(expected_components, bool) or not isinstance(
        expected_components, int
    ) or not isinstance(error_type, type):
        raise AnalysisRecoveryV2Error("canonical component coverage surface differs")

    def verify_component_coverage(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[int, ...]:
        expected: tuple[int, ...] | None = None
        expected_fold_by_group: dict[int, int] | None = None
        expected_counts_by_variant: dict[str, dict[int, int]] = {}
        arm_names = set(arms)
        variants = {str(row["variant_id"]) for row in rows}
        seeds = {int(row["model_seed"]) for row in rows}
        for variant in variants:
            for seed in seeds:
                arm_sets: dict[str, set[int]] = {}
                arm_folds: dict[str, dict[int, int]] = {}
                arm_counts: dict[str, dict[int, int]] = {}
                for arm in arm_names:
                    selected = [
                        row
                        for row in rows
                        if row["variant_id"] == variant
                        and int(row["model_seed"]) == seed
                        and row["arm"] == arm
                    ]
                    groups = [int(row["geometry_group"]) for row in selected]
                    if len(groups) != len(set(groups)):
                        raise error_type(
                            f"duplicate component across folds: {variant}/{seed}/{arm}"
                        )
                    if len(groups) != expected_components:
                        raise error_type(
                            f"component coverage is not 27: {variant}/{seed}/{arm}"
                        )
                    arm_sets[arm] = set(groups)
                    arm_folds[arm] = {
                        int(row["geometry_group"]): int(row["fold"])
                        for row in selected
                    }
                    arm_counts[arm] = {
                        int(row["geometry_group"]): int(row["cell_count"])
                        for row in selected
                    }
                if len({tuple(sorted(value)) for value in arm_sets.values()}) != 1:
                    raise error_type(f"arm component axes differ: {variant}/{seed}")
                if len({_integer_item_tuple(value) for value in arm_folds.values()}) != 1:
                    raise error_type(
                        f"arm component-to-fold assignments differ: {variant}/{seed}"
                    )
                if len({_integer_item_tuple(value) for value in arm_counts.values()}) != 1:
                    raise error_type(
                        f"arm component cell counts differ: {variant}/{seed}"
                    )
                axis = tuple(sorted(next(iter(arm_sets.values()))))
                reference_rows = [
                    row
                    for row in rows
                    if row["variant_id"] == variant
                    and int(row["model_seed"]) == seed
                    and row["arm"] == "observed_near"
                ]
                fold_by_group = {
                    int(row["geometry_group"]): int(row["fold"])
                    for row in reference_rows
                }
                if expected is None:
                    expected = axis
                    expected_fold_by_group = fold_by_group
                elif expected != axis:
                    raise error_type("component axis differs across variants/seeds")
                elif expected_fold_by_group != fold_by_group:
                    raise error_type("component-to-fold assignment differs")
                count_by_group = arm_counts["observed_near"]
                if variant not in expected_counts_by_variant:
                    expected_counts_by_variant[variant] = count_by_group
                elif expected_counts_by_variant[variant] != count_by_group:
                    raise error_type(
                        f"component cell counts differ across seeds: {variant}"
                    )
        if expected is None:
            raise error_type("no component rows")
        return expected

    return verify_component_coverage


def _source_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]


def _extend_v2_provenance(
    provenance: Mapping[str, Any],
    *,
    authority: RecoveryV2Authority,
    v1_authority: Any,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    _verify_authority_unchanged(authority, project_root=project_root)
    if not isinstance(provenance, Mapping):
        raise AnalysisRecoveryV2Error("v1 analyzer provenance is not an object")
    result = json.loads(_canonical_json(provenance))
    if "analysis_recovery_v2" in result:
        raise AnalysisRecoveryV2Error("provenance already has analysis_recovery_v2")
    parent_recovery = result.get("analysis_recovery")
    if not isinstance(parent_recovery, Mapping):
        raise AnalysisRecoveryV2Error("v1 analysis recovery provenance is absent")
    parent_row = _row_for_role(authority.parent_files, "v1_amendment")
    if (
        parent_recovery.get("amendment_id") != V1_AMENDMENT_ID
        or parent_recovery.get("amendment_size_bytes") != parent_row["size_bytes"]
        or parent_recovery.get("amendment_sha256") != parent_row["sha256"]
    ):
        raise AnalysisRecoveryV2Error("v1 analysis recovery provenance differs")
    raw_sources = result.get("sources")
    if not isinstance(raw_sources, list):
        raise AnalysisRecoveryV2Error("v1 provenance sources are not a list")
    sources: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise AnalysisRecoveryV2Error(f"v1 provenance source {index} is invalid")
        row = dict(raw)
        path = row.get("path")
        if not isinstance(path, str) or path in sources:
            raise AnalysisRecoveryV2Error("v1 provenance source paths are invalid")
        sources[path] = row
    amendment_record = {
        "path": authority.amendment_path.relative_to(
            project_root.resolve(strict=True)
        ).as_posix(),
        "size_bytes": authority.amendment_size_bytes,
        "sha256": authority.amendment_sha256,
    }
    additions = [amendment_record, *_source_records(authority.recovery_sources)]
    for row in additions:
        path = str(row["path"])
        existing = sources.get(path)
        if existing is not None and existing != row:
            raise AnalysisRecoveryV2Error(f"v2 recovery source collision: {path}")
        sources[path] = dict(row)
    result["sources"] = [sources[path] for path in sorted(sources)]
    result["analysis_recovery_v2"] = {
        "schema_version": 2,
        "amendment_id": authority.payload["amendment_id"],
        "amendment_path": amendment_record["path"],
        "amendment_size_bytes": amendment_record["size_bytes"],
        "amendment_sha256": amendment_record["sha256"],
        "entrypoint_path": V2_WRAPPER_PATH,
        "entrypoint_sha256": _row_for_role(
            authority.recovery_sources, "v2_wrapper"
        )["sha256"],
        "schema_path": V2_SCHEMA_PATH,
        "schema_sha256": EXPECTED_V2_FIXED_FILES["v2_amendment_schema"]["sha256"],
        "parent_amendment_id": v1_authority.payload["amendment_id"],
        "parent_amendment_sha256": v1_authority.amendment_sha256,
        "parent_layer": dict(authority.payload["parent_layer"]),
        "newly_patched_symbols": list(EXPECTED_CORRECTION["newly_patched_symbols"]),
        "arm_fold_equality": EXPECTED_CORRECTION["arm_fold_equality"],
        "arm_count_equality": EXPECTED_CORRECTION["arm_count_equality"],
        "failed_analysis_attempts": [
            dict(row) for row in authority.payload["failed_analysis_attempts"]
        ],
        "execution_contract": dict(authority.payload["execution_contract"]),
        "bound_recovery_sources": [
            dict(row)
            for row in sorted(
                authority.recovery_sources, key=lambda row: str(row["role"])
            )
        ],
    }
    _verify_authority_unchanged(authority, project_root=project_root)
    return result


def _install_v2_patches(
    analyzer: ModuleType,
    *,
    authority: RecoveryV2Authority,
    v1: ModuleType,
    v1_authority: Any,
    project_root: Path = PROJECT_ROOT,
) -> None:
    original_component = getattr(analyzer, "_verify_component_coverage", None)
    parent_provenance = getattr(analyzer, "_source_provenance", None)
    if not callable(original_component) or not callable(parent_provenance):
        raise AnalysisRecoveryV2Error("canonical v2 patch surface differs")
    component_adapter = _component_coverage_adapter(analyzer)

    def source_provenance(**kwargs: Any) -> dict[str, Any]:
        _verify_authority_unchanged(authority, project_root=project_root)
        try:
            v1._verify_authority_unchanged(v1_authority, project_root=project_root)
        except Exception as error:
            raise AnalysisRecoveryV2Error("v1 authority changed before provenance") from error
        base = parent_provenance(**kwargs)
        try:
            v1._verify_authority_unchanged(v1_authority, project_root=project_root)
        except Exception as error:
            raise AnalysisRecoveryV2Error("v1 authority changed after provenance") from error
        _verify_authority_unchanged(authority, project_root=project_root)
        return _extend_v2_provenance(
            base,
            authority=authority,
            v1_authority=v1_authority,
            project_root=project_root,
        )

    analyzer._verify_component_coverage = component_adapter
    analyzer._source_provenance = source_provenance


def _recheck_layers(
    v1: ModuleType,
    v1_authority: Any,
    authority: RecoveryV2Authority,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    _verify_authority_unchanged(authority, project_root=project_root)
    try:
        v1._verify_authority_unchanged(v1_authority, project_root=project_root)
    except Exception as error:
        raise AnalysisRecoveryV2Error("v1 recovery authority changed") from error
    _verify_authority_unchanged(authority, project_root=project_root)


def _is_v1_error(error: BaseException, v1: ModuleType | None) -> bool:
    error_type = getattr(v1, "AnalysisRecoveryError", None) if v1 is not None else None
    return isinstance(error_type, type) and isinstance(error, error_type)


def main(argv: Sequence[str] | None = None) -> int:
    v1: ModuleType | None = None
    try:
        recovery = _arguments(argv)
        preliminary_invocation = (
            None
            if recovery.authority_check_only
            else _parse_analyzer_invocation(recovery.analyzer_argv)
        )
        authority = _verify_authority(recovery.amendment_path)
        v1 = _load_v1_module(authority)
        v1_authority = _verify_v1_layer(v1, authority)
        if recovery.authority_check_only:
            _recheck_layers(v1, v1_authority, authority)
            print(
                _canonical_json(
                    {
                        "verified": True,
                        "amendment_id": authority.payload["amendment_id"],
                        "amendment_size_bytes": authority.amendment_size_bytes,
                        "amendment_sha256": authority.amendment_sha256,
                        "parent_amendment_id": v1_authority.payload["amendment_id"],
                        "parent_amendment_sha256": v1_authority.amendment_sha256,
                        "scientific_outcomes_read": False,
                    }
                )
            )
            return 0
        invocation = v1._parse_analyzer_invocation(recovery.analyzer_argv)
        if preliminary_invocation is None or not _same_invocation(
            preliminary_invocation, invocation
        ):
            raise AnalysisRecoveryV2Error("v1 and v2 analyzer CLI parsing differs")
        v1._verify_invocation_authority(invocation, v1_authority)
        _recheck_layers(v1, v1_authority, authority)
        analyzer = v1._load_analyzer(v1_authority)
        v1._install_patches(analyzer, authority=v1_authority)
        _recheck_layers(v1, v1_authority, authority)
        bound_database = v1._verify_effective_registry_path(analyzer, v1_authority)
        v1._verify_public_registry_lifecycle(
            analyzer,
            v1_authority,
            bound_database=bound_database,
        )
        _recheck_layers(v1, v1_authority, authority)
        _install_v2_patches(
            analyzer,
            authority=authority,
            v1=v1,
            v1_authority=v1_authority,
        )
        _recheck_layers(v1, v1_authority, authority)
    except BaseException as error:
        if isinstance(error, AnalysisRecoveryV2Error) or _is_v1_error(error, v1):
            print(f"ERROR: analysis recovery v2 authority invalid: {error}", file=sys.stderr)
            return 2
        raise
    try:
        result = int(analyzer.main(invocation.argv))
    except BaseException as error:
        try:
            _recheck_layers(v1, v1_authority, authority)
        except AnalysisRecoveryV2Error as recheck_error:
            print(
                f"ERROR: analysis recovery v2 authority invalid: {recheck_error}",
                file=sys.stderr,
            )
            return 2
        if isinstance(error, AnalysisRecoveryV2Error) or _is_v1_error(error, v1):
            print(f"ERROR: analysis recovery v2 authority invalid: {error}", file=sys.stderr)
            return 2
        raise
    try:
        _recheck_layers(v1, v1_authority, authority)
    except AnalysisRecoveryV2Error as error:
        print(f"ERROR: analysis recovery v2 authority invalid: {error}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
