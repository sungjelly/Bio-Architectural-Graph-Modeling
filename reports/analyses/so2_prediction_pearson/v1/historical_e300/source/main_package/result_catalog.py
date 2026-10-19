"""Validated, human-facing catalog for conclusion-bearing BAGM results."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import yaml

from .configuration import load_yaml_mapping
from .identifiers import canonical_sha256
from .paths import ProjectPaths, current_paths


RESULT_SCHEMA_NAME = "bagm_result_record"
RESULT_SCHEMA_VERSION = 1
RESULT_MANIFEST_NAME = "result.yaml"
RESULT_README_NAME = "README.md"
CATALOG_JSON_NAME = "catalog.json"
CATALOG_MARKDOWN_NAME = "CATALOG.md"

EXPERIMENT_TYPES = (
    "data_qc",
    "synthetic_recovery",
    "predictive_benchmark",
    "ablation",
    "stability_audit",
    "faithfulness_audit",
    "null_calibration",
    "niche_discovery",
    "external_validation",
    "perturbation",
    "descriptive_analysis",
    "infrastructure_validation",
)
RESULT_STATUSES = ("draft", "verified", "superseded")
RESULT_OUTCOMES = ("supported", "negative", "inconclusive", "blocked")
EVIDENCE_DIMENSIONS = (
    "predictive_gain",
    "stability",
    "faithfulness",
    "null_calibration",
    "patient_replication",
    "external_support",
    "perturbation_support",
)
EVIDENCE_STATUSES = (
    "pending",
    "not_tested",
    "not_applicable",
    "failed",
    "mixed",
    "supported",
)
SOURCE_KINDS = (
    "run_artifact",
    "evaluation",
    "report",
    "campaign_record",
    "external_record",
    "other",
)
SOURCE_ROOTS = ("project", "artifact", "report", "result")
SOURCE_VERIFICATION_STATUSES = (
    "pending",
    "verified",
    "tombstoned",
    "unavailable",
)
FILE_ROLES = (
    "figure",
    "table",
    "numeric_source",
    "summary",
    "attachment",
    "other",
)

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_RESULT_ID = re.compile(r"^res_[a-z0-9][a-z0-9_]*$")
_CAMPAIGN_ID = re.compile(r"^cmp_[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_RUN_ID = re.compile(r"^(?:r|lr)_[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TODO = re.compile(r"\bTODO\b", re.IGNORECASE)
_PROHIBITED_KEYS = frozenset(
    {
        "patient_id",
        "patient_identifier",
        "donor_id",
        "donor_identifier",
        "original_cell_identifier",
        "source_patient_id",
    }
)


class ResultCatalogError(ValueError):
    """Raised when a curated result record or catalog violates its contract."""


@dataclass(frozen=True, slots=True)
class ValidatedResult:
    """One validated result manifest and its canonical directory."""

    manifest_path: Path
    result_directory: Path
    payload: dict[str, Any]


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultCatalogError(f"{location} must be a mapping.")
    return value


def _sequence(value: Any, location: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ResultCatalogError(f"{location} must be a list.")
    return value


def _text(value: Any, location: str, *, allow_todo: bool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResultCatalogError(f"{location} must be a nonempty string.")
    normalized = value.strip()
    if not allow_todo and _TODO.search(normalized):
        raise ResultCatalogError(f"{location} still contains a TODO placeholder.")
    return normalized


def _enum(
    value: Any,
    allowed: Sequence[str],
    location: str,
) -> str:
    normalized = _text(value, location, allow_todo=False)
    if normalized not in allowed:
        raise ResultCatalogError(
            f"{location} must be one of {list(allowed)}, got {normalized!r}."
        )
    return normalized


def _keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    location: str,
) -> None:
    required_set = set(required)
    missing = sorted(required_set.difference(value))
    if missing:
        raise ResultCatalogError(f"{location} is missing required keys: {missing}")
    unknown = sorted(set(value).difference(required_set, optional))
    if unknown:
        raise ResultCatalogError(f"{location} has unknown keys: {unknown}")


def _string_list(
    value: Any,
    location: str,
    *,
    allow_todo: bool,
) -> list[str]:
    items = _sequence(value, location)
    normalized = [
        _text(item, f"{location}[{index}]", allow_todo=allow_todo)
        for index, item in enumerate(items)
    ]
    if len(set(normalized)) != len(normalized):
        raise ResultCatalogError(f"{location} contains duplicate values.")
    return normalized


def _integer_list(value: Any, location: str) -> list[int]:
    items = _sequence(value, location)
    normalized: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ResultCatalogError(
                f"{location}[{index}] must be a non-negative integer."
            )
        normalized.append(item)
    if len(set(normalized)) != len(normalized):
        raise ResultCatalogError(f"{location} contains duplicate values.")
    return normalized


def _safe_relative_path(value: Any, location: str) -> str:
    text = _text(value, location, allow_todo=False)
    if "\\" in text:
        raise ResultCatalogError(f"{location} must use POSIX separators.")
    path = PurePosixPath(text)
    if path.is_absolute() or text != path.as_posix() or not path.parts:
        raise ResultCatalogError(f"{location} must be a normalized relative path.")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ResultCatalogError(f"{location} contains an unsafe path component.")
    return text


def _timestamp(value: Any, location: str, *, allow_todo: bool) -> str:
    text = _text(value, location, allow_todo=allow_todo)
    if allow_todo and _TODO.search(text):
        return text
    if not text.endswith("Z"):
        raise ResultCatalogError(f"{location} must be an explicit UTC timestamp.")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise ResultCatalogError(f"{location} is not a valid ISO timestamp.") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResultCatalogError(f"{location} must be UTC.")
    return text


def _scan_prohibited_keys(value: Any, location: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _PROHIBITED_KEYS:
                raise ResultCatalogError(
                    f"{location} contains prohibited identifier field {key!r}."
                )
            _scan_prohibited_keys(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_prohibited_keys(item, f"{location}[{index}]")


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under(root: Path, relative: str, location: str) -> Path:
    root = root.resolve(strict=False)
    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ResultCatalogError(f"{location} escapes its declared root.") from error
    return resolved


def _source_root(paths: ProjectPaths, root_name: str) -> Path:
    return {
        "project": paths.project_root,
        "artifact": paths.artifact_root,
        "report": paths.report_root,
        "result": paths.result_root,
    }[root_name]


def _validate_method(value: Any, *, allow_todo: bool) -> dict[str, Any]:
    method = dict(_mapping(value, "method"))
    required = (
        "name",
        "model_family",
        "graph_context",
        "masking_or_perturbation",
        "evaluation_design",
        "implementation_version",
    )
    _keys(method, required=required, location="method")
    for key in required:
        method[key] = _text(method[key], f"method.{key}", allow_todo=allow_todo)
    return method


def _validate_conclusion(value: Any, *, allow_todo: bool) -> dict[str, Any]:
    conclusion = dict(_mapping(value, "conclusion"))
    required = (
        "question",
        "estimand",
        "observed_result",
        "strongest_alternative_explanation",
        "controls",
        "remaining_uncertainty",
        "maximum_defensible_claim",
    )
    _keys(conclusion, required=required, location="conclusion")
    for key in required:
        if key == "controls":
            conclusion[key] = _string_list(
                conclusion[key], "conclusion.controls", allow_todo=allow_todo
            )
        else:
            conclusion[key] = _text(
                conclusion[key], f"conclusion.{key}", allow_todo=allow_todo
            )
    if not allow_todo and not conclusion["controls"]:
        raise ResultCatalogError("conclusion.controls must not be empty when verified.")
    return conclusion


def _validate_evidence(value: Any, *, allow_todo: bool) -> dict[str, Any]:
    evidence = dict(_mapping(value, "evidence"))
    _keys(evidence, required=EVIDENCE_DIMENSIONS, location="evidence")
    normalized: dict[str, Any] = {}
    for dimension in EVIDENCE_DIMENSIONS:
        entry = dict(_mapping(evidence[dimension], f"evidence.{dimension}"))
        _keys(
            entry,
            required=("status", "summary"),
            location=f"evidence.{dimension}",
        )
        status = _enum(
            entry["status"], EVIDENCE_STATUSES, f"evidence.{dimension}.status"
        )
        if not allow_todo and status == "pending":
            raise ResultCatalogError(
                f"evidence.{dimension}.status cannot be pending when verified."
            )
        normalized[dimension] = {
            "status": status,
            "summary": _text(
                entry["summary"],
                f"evidence.{dimension}.summary",
                allow_todo=allow_todo,
            ),
        }
    return normalized


def _validate_sources(
    value: Any,
    *,
    allow_todo: bool,
    verify_sources: bool,
    paths: ProjectPaths,
) -> list[dict[str, Any]]:
    sources = _sequence(value, "provenance.sources")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(sources):
        location = f"provenance.sources[{index}]"
        source = dict(_mapping(raw, location))
        _keys(
            source,
            required=("kind", "root", "path", "sha256", "verification_status"),
            optional=("run_id", "description"),
            location=location,
        )
        kind = _enum(source["kind"], SOURCE_KINDS, f"{location}.kind")
        root_name = _enum(source["root"], SOURCE_ROOTS, f"{location}.root")
        relative = _safe_relative_path(source["path"], f"{location}.path")
        verification_status = _enum(
            source["verification_status"],
            SOURCE_VERIFICATION_STATUSES,
            f"{location}.verification_status",
        )
        digest = _text(source["sha256"], f"{location}.sha256", allow_todo=allow_todo)
        if not (allow_todo and _TODO.search(digest)) and not _SHA256.fullmatch(digest):
            raise ResultCatalogError(f"{location}.sha256 must be lowercase SHA-256.")
        run_id = source.get("run_id")
        if run_id is not None:
            run_id = _text(run_id, f"{location}.run_id", allow_todo=allow_todo)
            if not (allow_todo and _TODO.search(run_id)) and not _RUN_ID.fullmatch(run_id):
                raise ResultCatalogError(f"{location}.run_id is invalid.")
        if kind == "run_artifact" and run_id is None:
            raise ResultCatalogError(f"{location}.run_id is required for run artifacts.")
        description = source.get("description")
        if description is not None:
            description = _text(
                description, f"{location}.description", allow_todo=allow_todo
            )
        identity = (root_name, relative, digest)
        if identity in identities:
            raise ResultCatalogError(f"{location} duplicates another source.")
        identities.add(identity)
        if not allow_todo and verification_status == "pending":
            raise ResultCatalogError(
                f"{location}.verification_status cannot be pending when verified."
            )
        if verify_sources and verification_status == "verified":
            source_path = _resolve_under(
                _source_root(paths, root_name), relative, f"{location}.path"
            )
            if not source_path.is_file():
                raise ResultCatalogError(f"Verified source is missing: {source_path}")
            observed = sha256_file(source_path)
            if observed != digest:
                raise ResultCatalogError(
                    f"Verified source checksum drifted for {source_path}: {observed}"
                )
        normalized_source: dict[str, Any] = {
            "kind": kind,
            "root": root_name,
            "path": relative,
            "sha256": digest,
            "verification_status": verification_status,
        }
        if run_id is not None:
            normalized_source["run_id"] = run_id
        if description is not None:
            normalized_source["description"] = description
        normalized.append(normalized_source)
    if not allow_todo and not normalized:
        raise ResultCatalogError("provenance.sources must not be empty when verified.")
    return normalized


def _validate_provenance(
    value: Any,
    *,
    allow_todo: bool,
    verify_sources: bool,
    paths: ProjectPaths,
) -> dict[str, Any]:
    provenance = dict(_mapping(value, "provenance"))
    required = (
        "created_at_utc",
        "updated_at_utc",
        "git_commit",
        "run_ids",
        "expected_seeds",
        "included_seeds",
        "expected_folds",
        "included_folds",
        "failed_or_excluded_runs",
        "sources",
    )
    _keys(provenance, required=required, location="provenance")
    created = _timestamp(
        provenance["created_at_utc"], "provenance.created_at_utc", allow_todo=allow_todo
    )
    updated = _timestamp(
        provenance["updated_at_utc"], "provenance.updated_at_utc", allow_todo=allow_todo
    )
    git_commit = _text(
        provenance["git_commit"], "provenance.git_commit", allow_todo=allow_todo
    )
    if not (allow_todo and _TODO.search(git_commit)) and not _GIT_COMMIT.fullmatch(
        git_commit
    ):
        raise ResultCatalogError("provenance.git_commit must be a full lowercase hash.")
    run_ids = _string_list(
        provenance["run_ids"], "provenance.run_ids", allow_todo=allow_todo
    )
    for run_id in run_ids:
        if not (allow_todo and _TODO.search(run_id)) and not _RUN_ID.fullmatch(run_id):
            raise ResultCatalogError(f"Invalid provenance run ID: {run_id!r}")
    expected_seeds = _integer_list(provenance["expected_seeds"], "provenance.expected_seeds")
    included_seeds = _integer_list(provenance["included_seeds"], "provenance.included_seeds")
    expected_folds = _integer_list(provenance["expected_folds"], "provenance.expected_folds")
    included_folds = _integer_list(provenance["included_folds"], "provenance.included_folds")
    if not set(included_seeds).issubset(expected_seeds):
        raise ResultCatalogError("included_seeds must be a subset of expected_seeds.")
    if not set(included_folds).issubset(expected_folds):
        raise ResultCatalogError("included_folds must be a subset of expected_folds.")
    exclusions_raw = _sequence(
        provenance["failed_or_excluded_runs"], "provenance.failed_or_excluded_runs"
    )
    exclusions: list[dict[str, str]] = []
    for index, raw in enumerate(exclusions_raw):
        location = f"provenance.failed_or_excluded_runs[{index}]"
        exclusion = dict(_mapping(raw, location))
        _keys(exclusion, required=("run_id", "reason"), location=location)
        excluded_run = _text(
            exclusion["run_id"], f"{location}.run_id", allow_todo=allow_todo
        )
        if not (allow_todo and _TODO.search(excluded_run)) and not _RUN_ID.fullmatch(
            excluded_run
        ):
            raise ResultCatalogError(f"{location}.run_id is invalid.")
        exclusions.append(
            {
                "run_id": excluded_run,
                "reason": _text(
                    exclusion["reason"], f"{location}.reason", allow_todo=allow_todo
                ),
            }
        )
    sources = _validate_sources(
        provenance["sources"],
        allow_todo=allow_todo,
        verify_sources=verify_sources,
        paths=paths,
    )
    source_run_ids = {
        str(source["run_id"]) for source in sources if "run_id" in source
    }
    if not source_run_ids.issubset(run_ids):
        raise ResultCatalogError(
            "Every source run_id must also appear in provenance.run_ids."
        )
    return {
        "created_at_utc": created,
        "updated_at_utc": updated,
        "git_commit": git_commit,
        "run_ids": run_ids,
        "expected_seeds": expected_seeds,
        "included_seeds": included_seeds,
        "expected_folds": expected_folds,
        "included_folds": included_folds,
        "failed_or_excluded_runs": exclusions,
        "sources": sources,
    }


def _validate_files(
    value: Any,
    *,
    allow_todo: bool,
    verify_payloads: bool,
    result_directory: Path,
) -> list[dict[str, Any]]:
    files = _sequence(value, "files")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(files):
        location = f"files[{index}]"
        record = dict(_mapping(raw, location))
        _keys(
            record,
            required=("path", "role", "sha256", "size_bytes"),
            optional=("description",),
            location=location,
        )
        relative = _safe_relative_path(record["path"], f"{location}.path")
        if relative in {RESULT_MANIFEST_NAME, RESULT_README_NAME}:
            raise ResultCatalogError(
                f"{location}.path must describe a curated payload, not record metadata."
            )
        if relative in seen:
            raise ResultCatalogError(f"Duplicate curated file path: {relative}")
        seen.add(relative)
        role = _enum(record["role"], FILE_ROLES, f"{location}.role")
        digest = _text(record["sha256"], f"{location}.sha256", allow_todo=allow_todo)
        if not (allow_todo and _TODO.search(digest)) and not _SHA256.fullmatch(digest):
            raise ResultCatalogError(f"{location}.sha256 must be lowercase SHA-256.")
        size = record["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ResultCatalogError(f"{location}.size_bytes must be non-negative.")
        description = record.get("description")
        if description is not None:
            description = _text(
                description, f"{location}.description", allow_todo=allow_todo
            )
        if verify_payloads:
            payload_path = _resolve_under(result_directory, relative, f"{location}.path")
            if not payload_path.is_file():
                raise ResultCatalogError(f"Curated payload is missing: {payload_path}")
            observed_size = payload_path.stat().st_size
            if observed_size != size:
                raise ResultCatalogError(
                    f"Curated payload size drifted for {payload_path}: {observed_size}"
                )
            observed_digest = sha256_file(payload_path)
            if observed_digest != digest:
                raise ResultCatalogError(
                    f"Curated payload checksum drifted for {payload_path}: "
                    f"{observed_digest}"
                )
        normalized_record: dict[str, Any] = {
            "path": relative,
            "role": role,
            "sha256": digest,
            "size_bytes": size,
        }
        if description is not None:
            normalized_record["description"] = description
        normalized.append(normalized_record)
    return normalized


def validate_result_manifest(
    manifest_path: str | Path,
    *,
    paths: ProjectPaths | None = None,
    verify_payloads: bool = False,
    verify_sources: bool = False,
) -> ValidatedResult:
    """Validate one result record and optionally verify referenced bytes."""

    resolved_paths = current_paths() if paths is None else paths
    manifest = Path(manifest_path).resolve(strict=False)
    if manifest.name != RESULT_MANIFEST_NAME or not manifest.is_file():
        raise ResultCatalogError(f"Result manifest does not exist: {manifest}")
    payload = load_yaml_mapping(manifest)
    _scan_prohibited_keys(payload)
    required = (
        "schema_name",
        "schema_version",
        "revision",
        "result_id",
        "title",
        "status",
        "outcome",
        "experiment_type",
        "method_family",
        "lifecycle_stage",
        "study_axis",
        "campaign_ids",
        "method",
        "conclusion",
        "evidence",
        "provenance",
        "files",
    )
    _keys(
        payload,
        required=required,
        optional=("supersedes", "superseded_by"),
        location="result",
    )
    if payload["schema_name"] != RESULT_SCHEMA_NAME:
        raise ResultCatalogError(f"Unsupported result schema: {payload['schema_name']!r}")
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultCatalogError(
            f"Unsupported result schema version: {payload['schema_version']!r}"
        )
    revision = payload["revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ResultCatalogError("revision must be a positive integer.")
    status = _enum(payload["status"], RESULT_STATUSES, "status")
    allow_todo = status == "draft"
    result_id = _text(payload["result_id"], "result_id", allow_todo=False)
    if not _RESULT_ID.fullmatch(result_id):
        raise ResultCatalogError("result_id must match res_<lowercase_ascii_slug>.")
    experiment_type = _enum(
        payload["experiment_type"], EXPERIMENT_TYPES, "experiment_type"
    )
    method_family = _text(payload["method_family"], "method_family", allow_todo=False)
    if not _SLUG.fullmatch(method_family):
        raise ResultCatalogError("method_family must be a lowercase ASCII slug.")
    result_root = resolved_paths.result_root.resolve(strict=False)
    result_directory = manifest.parent.resolve(strict=False)
    try:
        relative_directory = result_directory.relative_to(result_root)
    except ValueError as error:
        raise ResultCatalogError(
            f"Result directory is outside BAGM_RESULT_ROOT: {result_directory}"
        ) from error
    expected_parts = (experiment_type, method_family, result_id)
    if relative_directory.parts != expected_parts:
        raise ResultCatalogError(
            "Result directory must match experiment_type/method_family/result_id: "
            f"expected {Path(*expected_parts)}, got {relative_directory}"
        )
    campaign_ids = _string_list(
        payload["campaign_ids"], "campaign_ids", allow_todo=allow_todo
    )
    for campaign_id in campaign_ids:
        if not (allow_todo and _TODO.search(campaign_id)) and not _CAMPAIGN_ID.fullmatch(
            campaign_id
        ):
            raise ResultCatalogError(f"Invalid campaign ID: {campaign_id!r}")
    supersedes = payload.get("supersedes")
    superseded_by = payload.get("superseded_by")
    for key, value in (("supersedes", supersedes), ("superseded_by", superseded_by)):
        if value is not None and not _RESULT_ID.fullmatch(
            _text(value, key, allow_todo=False)
        ):
            raise ResultCatalogError(f"{key} must be a valid result ID.")
    if status == "superseded" and superseded_by is None:
        raise ResultCatalogError("A superseded result requires superseded_by.")
    readme = result_directory / RESULT_README_NAME
    if status != "draft":
        if not readme.is_file() or not readme.read_text(encoding="utf-8").strip():
            raise ResultCatalogError(f"Verified result README is missing: {readme}")
        if _TODO.search(readme.read_text(encoding="utf-8")):
            raise ResultCatalogError("Verified result README still contains TODO.")
    normalized: dict[str, Any] = {
        "schema_name": RESULT_SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "revision": revision,
        "result_id": result_id,
        "title": _text(payload["title"], "title", allow_todo=allow_todo),
        "status": status,
        "outcome": _enum(payload["outcome"], RESULT_OUTCOMES, "outcome"),
        "experiment_type": experiment_type,
        "method_family": method_family,
        "lifecycle_stage": _text(
            payload["lifecycle_stage"], "lifecycle_stage", allow_todo=allow_todo
        ),
        "study_axis": _text(
            payload["study_axis"], "study_axis", allow_todo=allow_todo
        ),
        "campaign_ids": campaign_ids,
        "method": _validate_method(payload["method"], allow_todo=allow_todo),
        "conclusion": _validate_conclusion(
            payload["conclusion"], allow_todo=allow_todo
        ),
        "evidence": _validate_evidence(payload["evidence"], allow_todo=allow_todo),
        "provenance": _validate_provenance(
            payload["provenance"],
            allow_todo=allow_todo,
            verify_sources=verify_sources,
            paths=resolved_paths,
        ),
        "files": _validate_files(
            payload["files"],
            allow_todo=allow_todo,
            verify_payloads=verify_payloads,
            result_directory=result_directory,
        ),
    }
    if supersedes is not None:
        normalized["supersedes"] = supersedes
    if superseded_by is not None:
        normalized["superseded_by"] = superseded_by
    return ValidatedResult(
        manifest_path=manifest,
        result_directory=result_directory,
        payload=normalized,
    )


def discover_result_manifests(result_root: str | Path) -> list[Path]:
    """Return every result manifest in stable path order."""

    root = Path(result_root).resolve(strict=False)
    if not root.exists():
        return []
    if not root.is_dir():
        raise ResultCatalogError(f"BAGM_RESULT_ROOT is not a directory: {root}")
    return sorted(path.resolve() for path in root.rglob(RESULT_MANIFEST_NAME))


def validate_result_tree(
    *,
    paths: ProjectPaths | None = None,
    verify_payloads: bool = False,
    verify_sources: bool = False,
) -> list[ValidatedResult]:
    """Validate all curated result records and enforce global ID uniqueness."""

    resolved_paths = current_paths() if paths is None else paths
    records = [
        validate_result_manifest(
            manifest,
            paths=resolved_paths,
            verify_payloads=verify_payloads,
            verify_sources=verify_sources,
        )
        for manifest in discover_result_manifests(resolved_paths.result_root)
    ]
    seen: dict[str, Path] = {}
    for record in records:
        result_id = str(record.payload["result_id"])
        previous = seen.get(result_id)
        if previous is not None:
            raise ResultCatalogError(
                f"Duplicate result_id {result_id!r}: {previous} and "
                f"{record.manifest_path}"
            )
        seen[result_id] = record.manifest_path
    return sorted(
        records,
        key=lambda record: (
            record.payload["experiment_type"],
            record.payload["method_family"],
            record.payload["result_id"],
        ),
    )


def build_catalog_payload(records: Sequence[ValidatedResult]) -> dict[str, Any]:
    """Build the deterministic machine-readable discovery view."""

    entries: list[dict[str, Any]] = []
    for record in records:
        payload = record.payload
        entries.append(
            {
                "result_id": payload["result_id"],
                "title": payload["title"],
                "status": payload["status"],
                "outcome": payload["outcome"],
                "experiment_type": payload["experiment_type"],
                "method_family": payload["method_family"],
                "lifecycle_stage": payload["lifecycle_stage"],
                "study_axis": payload["study_axis"],
                "campaign_ids": payload["campaign_ids"],
                "run_ids": payload["provenance"]["run_ids"],
                "evidence_status": {
                    dimension: payload["evidence"][dimension]["status"]
                    for dimension in EVIDENCE_DIMENSIONS
                },
                "directory": Path(
                    str(payload["experiment_type"]),
                    str(payload["method_family"]),
                    str(payload["result_id"]),
                ).as_posix(),
                "manifest_sha256": sha256_file(record.manifest_path),
            }
        )
    catalog: dict[str, Any] = {
        "schema_name": "bagm_result_catalog",
        "schema_version": 1,
        "result_count": len(entries),
        "results": entries,
    }
    catalog["catalog_content_sha256"] = canonical_sha256(catalog)
    return catalog


def catalog_json_bytes(catalog: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(catalog, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def catalog_markdown_bytes(catalog: Mapping[str, Any]) -> bytes:
    lines = [
        "# Curated Result Catalog",
        "",
        "Generated from validated `result.yaml` records. Do not edit this file",
        "by hand; run `scripts/results/manage_results.py catalog`.",
        "",
    ]
    results = list(catalog["results"])
    if not results:
        lines.append("No curated results are registered yet.")
    else:
        lines.extend(
            [
                "| Experiment type | Method | Result | Outcome | Status | Stage |",
                "|---|---|---|---|---|---|",
            ]
        )
        for entry in results:
            relative = Path(str(entry["directory"]))
            result_link = f"[{entry['result_id']}]({relative.as_posix()}/README.md)"
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        entry["experiment_type"],
                        entry["method_family"],
                        result_link,
                        entry["outcome"],
                        entry["status"],
                        entry["lifecycle_stage"],
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            f"Catalog content SHA-256: `{catalog['catalog_content_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_or_check_catalog(
    *,
    paths: ProjectPaths | None = None,
    check: bool = False,
    verify_payloads: bool = False,
    verify_sources: bool = False,
) -> dict[str, Any]:
    """Regenerate the result catalogs or fail if checked files have drifted."""

    resolved_paths = current_paths() if paths is None else paths
    records = validate_result_tree(
        paths=resolved_paths,
        verify_payloads=verify_payloads,
        verify_sources=verify_sources,
    )
    catalog = build_catalog_payload(records)
    expected = {
        resolved_paths.result_root / CATALOG_JSON_NAME: catalog_json_bytes(catalog),
        resolved_paths.result_root / CATALOG_MARKDOWN_NAME: catalog_markdown_bytes(catalog),
    }
    if check:
        drifted = [
            path
            for path, content in expected.items()
            if not path.is_file() or path.read_bytes() != content
        ]
        if drifted:
            raise ResultCatalogError(
                "Result catalog is missing or stale: "
                + ", ".join(path.as_posix() for path in drifted)
            )
    else:
        for path, content in expected.items():
            _atomic_write(path, content)
    return catalog


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def create_result_scaffold(
    *,
    result_id: str,
    title: str,
    experiment_type: str,
    method_family: str,
    lifecycle_stage: str,
    study_axis: str,
    campaign_ids: Sequence[str],
    paths: ProjectPaths | None = None,
    git_commit: str = "TODO",
    timestamp_utc: str | None = None,
) -> Path:
    """Create a fail-closed draft result directory with explicit placeholders."""

    resolved_paths = current_paths() if paths is None else paths
    if not _RESULT_ID.fullmatch(result_id):
        raise ResultCatalogError("result_id must match res_<lowercase_ascii_slug>.")
    if experiment_type not in EXPERIMENT_TYPES:
        raise ResultCatalogError(f"Unknown experiment type: {experiment_type!r}")
    if not _SLUG.fullmatch(method_family):
        raise ResultCatalogError("method_family must be a lowercase ASCII slug.")
    _text(title, "title", allow_todo=False)
    _text(lifecycle_stage, "lifecycle_stage", allow_todo=False)
    _text(study_axis, "study_axis", allow_todo=False)
    for campaign_id in campaign_ids:
        if not _CAMPAIGN_ID.fullmatch(campaign_id):
            raise ResultCatalogError(f"Invalid campaign ID: {campaign_id!r}")
    result_directory = (
        resolved_paths.result_root / experiment_type / method_family / result_id
    )
    if result_directory.exists():
        raise ResultCatalogError(f"Result directory already exists: {result_directory}")
    now = _utc_now() if timestamp_utc is None else timestamp_utc
    payload: dict[str, Any] = {
        "schema_name": RESULT_SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "revision": 1,
        "result_id": result_id,
        "title": title,
        "status": "draft",
        "outcome": "inconclusive",
        "experiment_type": experiment_type,
        "method_family": method_family,
        "lifecycle_stage": lifecycle_stage,
        "study_axis": study_axis,
        "campaign_ids": list(campaign_ids),
        "method": {
            "name": "TODO",
            "model_family": "TODO",
            "graph_context": "TODO",
            "masking_or_perturbation": "TODO",
            "evaluation_design": "TODO",
            "implementation_version": "TODO",
        },
        "conclusion": {
            "question": "TODO",
            "estimand": "TODO",
            "observed_result": "TODO",
            "strongest_alternative_explanation": "TODO",
            "controls": ["TODO"],
            "remaining_uncertainty": "TODO",
            "maximum_defensible_claim": "TODO",
        },
        "evidence": {
            dimension: {"status": "pending", "summary": "TODO"}
            for dimension in EVIDENCE_DIMENSIONS
        },
        "provenance": {
            "created_at_utc": now,
            "updated_at_utc": now,
            "git_commit": git_commit,
            "run_ids": [],
            "expected_seeds": [],
            "included_seeds": [],
            "expected_folds": [],
            "included_folds": [],
            "failed_or_excluded_runs": [],
            "sources": [],
        },
        "files": [],
    }
    result_directory.mkdir(parents=True)
    for child in ("figures", "tables", "attachments"):
        (result_directory / child).mkdir()
    manifest = result_directory / RESULT_MANIFEST_NAME
    manifest.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    (result_directory / RESULT_README_NAME).write_text(
        "\n".join(
            (
                f"# {title}",
                "",
                "Status: draft",
                "",
                "## Question",
                "",
                "TODO",
                "",
                "## Observed result",
                "",
                "TODO",
                "",
                "## Evidence and limitations",
                "",
                "TODO",
                "",
                "## Provenance and reproduction",
                "",
                "TODO",
                "",
            )
        ),
        encoding="utf-8",
    )
    validate_result_manifest(manifest, paths=resolved_paths)
    return result_directory


__all__ = [
    "CATALOG_JSON_NAME",
    "CATALOG_MARKDOWN_NAME",
    "EVIDENCE_DIMENSIONS",
    "EXPERIMENT_TYPES",
    "RESULT_MANIFEST_NAME",
    "RESULT_OUTCOMES",
    "RESULT_SCHEMA_NAME",
    "RESULT_SCHEMA_VERSION",
    "RESULT_STATUSES",
    "ResultCatalogError",
    "ValidatedResult",
    "build_catalog_payload",
    "create_result_scaffold",
    "discover_result_manifests",
    "sha256_file",
    "validate_result_manifest",
    "validate_result_tree",
    "write_or_check_catalog",
]
