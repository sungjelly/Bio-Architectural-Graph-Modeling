#!/usr/bin/env python3
"""Run the frozen robustness analyzer through one hash-bound API adapter.

The production launch binds the canonical analyzer and registry implementation
byte-for-byte.  This entry point leaves both files unchanged.  Before importing
the analyzer, it verifies a frozen analysis-only amendment and every authority
named by that amendment.  It then adapts the decoded ``Registry.get_run``
configuration key and extends the canonical analysis provenance with the
recovery authority.  Publish and ``--verify-only`` therefore use identical code.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
import sys
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ID = "cmp_20260810_same_gene_robustness_multiverse_v1"
AMENDMENT_ID = (
    "ta_cmp_20260810_same_gene_robustness_analysis_registry_config_adapter_v1"
)
DEFAULT_AMENDMENT = PROJECT_ROOT / (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "analysis_registry_config_adapter_v1.json"
)
SCHEMA_PATH = (
    "experiments/campaigns/"
    "cmp_20260810_same_gene_robustness_multiverse_v1/technical_amendments/"
    "analysis_registry_config_adapter_v1.schema.json"
)
WRAPPER_PATH = "scripts/analysis/run_same_gene_robustness_registry_recovery.py"
TEST_PATH = (
    "tests/unit/spatial_benchmark/"
    "test_same_gene_robustness_analysis_registry_recovery.py"
)

EXPECTED_SCHEMA_IDENTITY = {
    "path": SCHEMA_PATH,
    "size_bytes": 8070,
    "sha256": "27ff6935a6ab9fe63b156edb4d623f72d569e99a5f8fe0ae6c1f7be764e47b20",
}
EXPECTED_AUTHORITIES = {
    "contract": {
        "path": (
            "experiments/campaigns/"
            "cmp_20260810_same_gene_robustness_multiverse_v1/"
            "frozen_task_contract.yaml"
        ),
        "size_bytes": 28442,
        "sha256": "8761098cbafa91ae81b53a1c9cd0d8dcd293476d967f74cddde9be7dc5c990e9",
    },
    "launch_manifest": {
        "path": "state/materialized/same_gene_robustness_v1/recovery_r1/launch_manifest.json",
        "size_bytes": 7537,
        "sha256": "38fd482c1343f10efc8ec18507b05f5110f226898f236815a97b9671f7269223",
    },
    "full_plan": {
        "path": (
            "state/materialized/same_gene_robustness_v1/recovery_r1/"
            "full/attempt1/plan.json"
        ),
        "size_bytes": 139836,
        "sha256": "2c8b08c37ec6997c45417adbb4e74e0f394cb133845129e7a2c3fece66a0d0a1",
    },
    "final_ledger": {
        "path": (
            "state/materialized/same_gene_robustness_v1/recovery_r1/"
            "full/attempt1/ledger.json"
        ),
        "size_bytes": 109014,
        "sha256": "9e2a27d989e3e26e02af1bc667b999e6fced905c8575383aed94aff4ec77ef9e",
    },
    "canonical_analyzer": {
        "path": "scripts/analysis/analyze_same_gene_robustness.py",
        "size_bytes": 169429,
        "sha256": "0f77667e4e805b2d0b62a6ec75204c366badefb8b62d6bd1f4f1aea3ad11db85",
    },
    "registry_implementation": {
        "path": "src/spatial_benchmark/registry.py",
        "size_bytes": 119334,
        "sha256": "cdcedd6c9d6866576bdb3ad4b11c5a2edc020a588a4cbe2e35babe581b1a4bc3",
    },
    "registry_database": {
        "path": "state/tracking/bagm.sqlite3",
        "size_bytes": 12029952,
        "sha256": "3a5e5b895e74852fee79699345d3df7de4ada3799b806c670a6b641df0db6443",
    },
}
EXPECTED_RECOVERY_PATHS = {
    "recovery_wrapper": WRAPPER_PATH,
    "amendment_schema": SCHEMA_PATH,
    "dedicated_tests": TEST_PATH,
}
EXPECTED_DIAGNOSIS = {
    "failure_message": "registry row omits config_json",
    "failure_stage": "first_selected_run_registry_configuration_reconciliation",
    "campaign_registry_rows": 154,
    "full_registry_rows": 140,
    "full_rows_with_nonnull_raw_config_json": 140,
    "registry_query_projection": "SELECT_ALL",
    "registry_public_key": "config",
    "registry_public_value_type": "mapping",
    "root_cause": (
        "analyzer_expected_raw_json_after_registry_api_decoded_and_renamed_it"
    ),
}
EXPECTED_OUTCOME_ACCESS = {
    "automated_failed_analyzer_loaded_first_run_payload": True,
    "aggregate_computed": False,
    "analysis_output_published": False,
    "staging_output_retained": False,
    "scientific_effect_values_printed": False,
    "scientific_effect_values_human_inspected": False,
}
EXPECTED_CORRECTION = {
    "kind": "registry_api_decoded_config_adapter",
    "registry_configuration_contract": "decoded_config_mapping_only",
    "patched_symbols": ["_registry_configuration", "_source_provenance"],
    "raw_json_fallback_allowed": False,
    "model_or_training_changed": False,
    "scientific_computation_changed": False,
    "gate_or_threshold_changed": False,
    "randomization_changed": False,
    "scientific_payload_schema_changed": False,
    "provenance_extended": True,
}
EXPECTED_INVARIANTS = {
    "contract_unchanged": True,
    "launch_unchanged": True,
    "full_plan_unchanged": True,
    "final_ledger_unchanged": True,
    "canonical_analyzer_unchanged": True,
    "registry_implementation_unchanged": True,
    "run_bundles_unchanged": True,
    "prepared_data_unchanged": True,
    "seeds_folds_models_hyperparameters_unchanged": True,
    "gates_statistics_and_nulls_unchanged": True,
    "same_wrapper_required_for_publish_and_verify_only": True,
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
        "diagnosis",
        "outcome_access_record",
        "correction",
        "authorities",
        "recovery_sources",
        "scientific_invariants",
    }
)
FILE_ROW_KEYS = frozenset({"role", "path", "size_bytes", "sha256"})


class AnalysisRecoveryError(RuntimeError):
    """Raised before the canonical analyzer may read a scientific result."""


@dataclass(frozen=True)
class RecoveryAuthority:
    amendment_path: Path
    amendment_sha256: str
    payload: dict[str, Any]
    recovery_sources: tuple[dict[str, Any], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _strict_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AnalysisRecoveryError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisRecoveryError(f"{label} is not an object: {path}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], *, label: str) -> None:
    observed = frozenset(value)
    if observed != expected:
        raise AnalysisRecoveryError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _project_file(relative: Any, *, project_root: Path, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AnalysisRecoveryError(f"{label} path is not a nonempty string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts or "\\" in relative:
        raise AnalysisRecoveryError(f"{label} path is not canonical: {relative!r}")
    candidate = project_root.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise AnalysisRecoveryError(f"{label} is missing, non-file, or symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        canonical = resolved.relative_to(project_root.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise AnalysisRecoveryError(f"{label} escapes the project root") from error
    if canonical != relative:
        raise AnalysisRecoveryError(f"{label} path is not canonical: {relative!r}")
    return resolved


def _validate_file_row(row: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise AnalysisRecoveryError(f"{label} is not an object")
    _exact_keys(row, FILE_ROW_KEYS, label=label)
    role = row["role"]
    path = row["path"]
    size = row["size_bytes"]
    digest = row["sha256"]
    if not isinstance(role, str) or not role:
        raise AnalysisRecoveryError(f"{label}.role is invalid")
    if not isinstance(path, str) or not path:
        raise AnalysisRecoveryError(f"{label}.path is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise AnalysisRecoveryError(f"{label}.size_bytes is invalid")
    if not _valid_sha256(digest):
        raise AnalysisRecoveryError(f"{label}.sha256 is invalid")
    return dict(row)


def _validate_amendment(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise AnalysisRecoveryError("analysis recovery amendment is not an object")
    _exact_keys(payload, TOP_LEVEL_KEYS, label="analysis recovery amendment")
    scalar_expected = {
        "schema_version": 1,
        "amendment_id": AMENDMENT_ID,
        "campaign_id": CAMPAIGN_ID,
        "status": "frozen_analysis_technical_recovery",
        "scope": "analysis_adapter_only_no_scientific_change",
    }
    for key, expected in scalar_expected.items():
        if payload.get(key) != expected:
            raise AnalysisRecoveryError(f"analysis recovery amendment {key} differs")
    frozen_at = payload.get("frozen_at")
    if not isinstance(frozen_at, str) or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        frozen_at,
    ) is None:
        raise AnalysisRecoveryError("analysis recovery frozen_at is invalid")
    exact_sections = {
        "diagnosis": EXPECTED_DIAGNOSIS,
        "outcome_access_record": EXPECTED_OUTCOME_ACCESS,
        "correction": EXPECTED_CORRECTION,
        "scientific_invariants": EXPECTED_INVARIANTS,
    }
    for key, expected in exact_sections.items():
        if payload.get(key) != expected:
            raise AnalysisRecoveryError(f"analysis recovery {key} differs")

    raw_authorities = payload.get("authorities")
    if not isinstance(raw_authorities, list) or len(raw_authorities) != len(
        EXPECTED_AUTHORITIES
    ):
        raise AnalysisRecoveryError("analysis recovery authorities differ")
    authorities: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_authorities):
        row = _validate_file_row(raw, label=f"authorities[{index}]")
        role = str(row["role"])
        if role in authorities:
            raise AnalysisRecoveryError(f"duplicate authority role: {role}")
        authorities[role] = row
    if set(authorities) != set(EXPECTED_AUTHORITIES):
        raise AnalysisRecoveryError("analysis recovery authority roles differ")
    for role, expected in EXPECTED_AUTHORITIES.items():
        observed = {key: authorities[role][key] for key in expected}
        if observed != expected:
            raise AnalysisRecoveryError(f"analysis recovery {role} authority differs")

    raw_sources = payload.get("recovery_sources")
    if not isinstance(raw_sources, list) or len(raw_sources) != len(
        EXPECTED_RECOVERY_PATHS
    ):
        raise AnalysisRecoveryError("analysis recovery source inventory differs")
    sources: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_sources):
        row = _validate_file_row(raw, label=f"recovery_sources[{index}]")
        role = str(row["role"])
        if role in sources:
            raise AnalysisRecoveryError(f"duplicate recovery source role: {role}")
        sources[role] = row
    if set(sources) != set(EXPECTED_RECOVERY_PATHS):
        raise AnalysisRecoveryError("analysis recovery source roles differ")
    for role, expected_path in EXPECTED_RECOVERY_PATHS.items():
        if sources[role]["path"] != expected_path:
            raise AnalysisRecoveryError(f"analysis recovery source path differs: {role}")
    schema = sources["amendment_schema"]
    if {key: schema[key] for key in EXPECTED_SCHEMA_IDENTITY} != EXPECTED_SCHEMA_IDENTITY:
        raise AnalysisRecoveryError("analysis recovery schema identity differs")
    return json.loads(_canonical_json(payload))


def _verify_file_row(
    row: Mapping[str, Any], *, project_root: Path, label: str
) -> Path:
    path = _project_file(row["path"], project_root=project_root, label=label)
    if path.stat().st_size != int(row["size_bytes"]):
        raise AnalysisRecoveryError(f"{label} size differs: {row['path']}")
    if _sha256_file(path) != row["sha256"]:
        raise AnalysisRecoveryError(f"{label} SHA-256 differs: {row['path']}")
    return path


def _verify_registry_diagnosis(
    database: Path, *, expected: Mapping[str, Any] = EXPECTED_DIAGNOSIS
) -> None:
    """Recheck only schema/config metadata, never metrics or result payloads."""

    try:
        connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro&immutable=1", uri=True
        )
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT config_json FROM runs WHERE campaign_id = ? ORDER BY run_id",
            (CAMPAIGN_ID,),
        ).fetchall()
    except sqlite3.Error as error:
        raise AnalysisRecoveryError("cannot read the bound registry database") from error
    finally:
        if "connection" in locals():
            connection.close()
    if len(rows) != int(expected["campaign_registry_rows"]):
        raise AnalysisRecoveryError("campaign registry row count differs from diagnosis")
    full = 0
    full_nonnull = 0
    for index, row in enumerate(rows):
        raw = row["config_json"]
        if not isinstance(raw, str):
            raise AnalysisRecoveryError(f"registry config_json is absent at row {index}")
        try:
            configuration = json.loads(
                raw,
                object_pairs_hook=_strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise AnalysisRecoveryError(
                f"registry config_json is invalid at row {index}"
            ) from error
        if not isinstance(configuration, dict):
            raise AnalysisRecoveryError(
                f"registry config_json is not an object at row {index}"
            )
        evaluation = configuration.get("evaluation")
        protocol = evaluation.get("protocol") if isinstance(evaluation, Mapping) else None
        if protocol == "held_out_geometry_masked_reconstruction":
            full += 1
            full_nonnull += 1
        elif protocol != "resource_validation":
            raise AnalysisRecoveryError(
                f"registry contains an unexpected execution protocol at row {index}"
            )
    if full != int(expected["full_registry_rows"]):
        raise AnalysisRecoveryError("full registry row count differs from diagnosis")
    if full_nonnull != int(expected["full_rows_with_nonnull_raw_config_json"]):
        raise AnalysisRecoveryError("full raw config_json coverage differs from diagnosis")


def _verify_authority(
    amendment_path: Path = DEFAULT_AMENDMENT, *, project_root: Path = PROJECT_ROOT
) -> RecoveryAuthority:
    root = project_root.resolve(strict=True)
    try:
        relative_amendment = amendment_path.resolve(strict=True).relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise AnalysisRecoveryError("analysis recovery amendment escapes the project") from error
    amendment = _project_file(
        relative_amendment, project_root=root, label="analysis recovery amendment"
    )
    payload = _validate_amendment(
        _strict_json(amendment, label="analysis recovery amendment")
    )
    for row in payload["authorities"]:
        _verify_file_row(row, project_root=root, label=f"authority {row['role']}")
    database_row = next(
        row for row in payload["authorities"] if row["role"] == "registry_database"
    )
    _verify_registry_diagnosis(root / database_row["path"])
    recovery_sources = tuple(
        sorted(
            (
                {
                    "path": row["path"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
                for row in payload["recovery_sources"]
            ),
            key=lambda row: str(row["path"]),
        )
    )
    for row in recovery_sources:
        _verify_file_row(row, project_root=root, label="analysis recovery source")
    return RecoveryAuthority(
        amendment_path=amendment,
        amendment_sha256=_sha256_file(amendment),
        payload=payload,
        recovery_sources=recovery_sources,
    )


def _decoded_registry_configuration(
    row: Mapping[str, Any], *, error_type: type[Exception]
) -> dict[str, Any]:
    """Accept only the decoded public ``Registry.get_run`` configuration."""

    if not isinstance(row, Mapping):
        raise error_type("registry row is not a mapping")
    if "config_json" in row or "configuration_json" in row:
        raise error_type("registry row exposes a prohibited raw JSON configuration")
    value = row.get("config")
    if not isinstance(value, Mapping):
        raise error_type("registry row omits decoded config mapping")
    try:
        decoded = json.loads(
            _canonical_json(value),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise error_type("registry decoded config is not strict canonical JSON") from error
    if not isinstance(decoded, dict):
        raise error_type("registry decoded config is not an object")
    return decoded


def _source_record(path: Path, *, project_root: Path) -> dict[str, Any]:
    relative = path.resolve(strict=True).relative_to(project_root.resolve(strict=True))
    return {
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _extend_provenance(
    provenance: Mapping[str, Any],
    *,
    authority: RecoveryAuthority,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise AnalysisRecoveryError("canonical analyzer provenance is not an object")
    result = json.loads(_canonical_json(provenance))
    if "analysis_recovery" in result:
        raise AnalysisRecoveryError("canonical provenance already has analysis_recovery")
    raw_sources = result.get("sources")
    if not isinstance(raw_sources, list):
        raise AnalysisRecoveryError("canonical provenance sources are not a list")
    sources: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping) or set(raw) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise AnalysisRecoveryError(f"canonical provenance source {index} is invalid")
        row = dict(raw)
        path = row.get("path")
        if not isinstance(path, str) or path in sources:
            raise AnalysisRecoveryError("canonical provenance source paths are invalid")
        sources[path] = row
    amendment_record = _source_record(
        authority.amendment_path, project_root=project_root
    )
    authority_records = [
        {
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in authority.payload["authorities"]
    ]
    additions = [
        amendment_record,
        *authority_records,
        *authority.recovery_sources,
    ]
    for raw in additions:
        row = dict(raw)
        path = str(row["path"])
        existing = sources.get(path)
        if existing is not None and existing != row:
            raise AnalysisRecoveryError(f"analysis recovery source collision: {path}")
        sources[path] = row
    result["sources"] = [sources[path] for path in sorted(sources)]
    result["analysis_recovery"] = {
        "schema_version": 1,
        "amendment_id": authority.payload["amendment_id"],
        "amendment_path": amendment_record["path"],
        "amendment_sha256": amendment_record["sha256"],
        "entrypoint_path": WRAPPER_PATH,
        "entrypoint_sha256": next(
            row["sha256"]
            for row in authority.recovery_sources
            if row["path"] == WRAPPER_PATH
        ),
        "schema_path": SCHEMA_PATH,
        "schema_sha256": EXPECTED_SCHEMA_IDENTITY["sha256"],
        "patched_symbols": list(EXPECTED_CORRECTION["patched_symbols"]),
        "registry_configuration_contract": EXPECTED_CORRECTION[
            "registry_configuration_contract"
        ],
        "diagnosis": dict(authority.payload["diagnosis"]),
        "outcome_access_record": dict(authority.payload["outcome_access_record"]),
        "bound_authorities": [
            dict(row)
            for row in sorted(
                authority.payload["authorities"], key=lambda row: str(row["role"])
            )
        ],
    }
    return result


def _load_analyzer(
    authority: RecoveryAuthority, *, project_root: Path = PROJECT_ROOT
) -> ModuleType:
    row = next(
        value
        for value in authority.payload["authorities"]
        if value["role"] == "canonical_analyzer"
    )
    path = _verify_file_row(
        row, project_root=project_root, label="canonical analyzer before import"
    )
    module_name = "same_gene_robustness_analysis_registry_recovery_target"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AnalysisRecoveryError("cannot construct canonical analyzer import")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    _verify_file_row(
        row, project_root=project_root, label="canonical analyzer after import"
    )
    return module


def _install_patches(
    analyzer: ModuleType,
    *,
    authority: RecoveryAuthority,
    project_root: Path = PROJECT_ROOT,
) -> None:
    original_provenance = getattr(analyzer, "_source_provenance", None)
    error_type = getattr(analyzer, "RobustnessAnalysisError", None)
    if not callable(original_provenance) or not isinstance(error_type, type):
        raise AnalysisRecoveryError("canonical analyzer patch surface differs")

    def registry_configuration(row: Mapping[str, Any]) -> dict[str, Any]:
        return _decoded_registry_configuration(row, error_type=error_type)

    def source_provenance(**kwargs: Any) -> dict[str, Any]:
        base = original_provenance(**kwargs)
        return _extend_provenance(
            base, authority=authority, project_root=project_root
        )

    analyzer._registry_configuration = registry_configuration
    analyzer._source_provenance = source_provenance


def _arguments(
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    values = list(sys.argv[1:] if argv is None else argv)
    if values.count("--recovery-amendment") > 1:
        raise AnalysisRecoveryError("--recovery-amendment may be supplied only once")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--recovery-amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--authority-check-only", action="store_true")
    recovery, analyzer_argv = parser.parse_known_args(values)
    return recovery, analyzer_argv


def main(argv: Sequence[str] | None = None) -> int:
    try:
        recovery, analyzer_argv = _arguments(argv)
        authority = _verify_authority(recovery.recovery_amendment)
        if recovery.authority_check_only:
            print(
                _canonical_json(
                    {
                        "verified": True,
                        "amendment_id": authority.payload["amendment_id"],
                        "amendment_sha256": authority.amendment_sha256,
                        "scientific_outcomes_read": False,
                    }
                )
            )
            return 0
        analyzer = _load_analyzer(authority)
        _install_patches(analyzer, authority=authority)
    except AnalysisRecoveryError as error:
        print(f"ERROR: analysis recovery authority invalid: {error}", file=sys.stderr)
        return 2
    return int(analyzer.main(analyzer_argv))


if __name__ == "__main__":
    raise SystemExit(main())
