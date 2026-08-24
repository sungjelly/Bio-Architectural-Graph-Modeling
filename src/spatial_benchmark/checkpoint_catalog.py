"""Semantic checkpoint catalog for BAGM.

Canonical run bundles remain organized by immutable execution identity under
``artifacts/runs/YYYY/MM/<run_id>/``.  This module adds an orthogonal semantic
view over checkpoint artifacts; it never moves, copies, loads, or rewrites a
checkpoint and never changes registry state.

Catalog construction, filtering, resolution, and export are read-only.
``index_checkpoint_catalog`` is the sole explicit write path: it persists only
derived aliases/categories/checkpoint metadata in the authoritative registry
and never changes checkpoint files or canonical run manifests.

Historical records are enriched only by an exact ``legacy_run_id`` match in an
audited ``index.jsonl``.  In particular, placeholder fold/attempt values in the
SQLite ``runs`` table are replaced with ``None`` when the historical index says
those dimensions were not recorded.  Optional schema-v3 alias, category, and
checkpoint metadata are consumed when present, while the catalog also works
against registries that predate those tables.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

import yaml

from . import identifiers as identifier_api
from .configuration import PRIMARY_METRIC_DIRECTIONS
from .identifiers import canonical_json, canonical_sha256
from .paths import ProjectPaths, current_paths
from .registry import (
    ARTIFACT_STATUS_DELETED_BY_RETENTION,
    ARTIFACT_STATUS_RETENTION_PENDING,
    Registry,
    RegistryConflictError,
    RegistryError,
)


CATALOG_SCHEMA_VERSION = 1

CHECKPOINT_ARTIFACT_KINDS = frozenset(
    {
        "legacy_checkpoint",
        "checkpoint",
        "checkpoint_best",
        "checkpoint_last",
        "model_checkpoint",
    }
)

CONTROLLED_STAGES = frozenset(
    {
        "diagnostic",
        "exploratory_screen",
        "validation_confirmation",
        "locked_final",
        "posthoc_evaluation",
        "unknown",
    }
)

_STAGE_ORDER = {
    "diagnostic": 0,
    "exploratory_screen": 1,
    "validation_confirmation": 2,
    "locked_final": 3,
    "posthoc_evaluation": 4,
    "unknown": 5,
}

_BATCH_CLASSIFICATION: dict[str, dict[str, Any]] = {
    "runtime_smoke_cpu_b0_v1": {
        "stage": "diagnostic",
        "study_axis": "runtime_smoke",
        "study_axes": ["runtime", "cpu", "smoke"],
    },
    "runtime_smoke_cpu_broad_field_v1": {
        "stage": "diagnostic",
        "study_axis": "runtime_smoke",
        "study_axes": ["runtime", "cpu", "smoke"],
    },
    "runtime_smoke_cpu_g1_v1": {
        "stage": "diagnostic",
        "study_axis": "runtime_smoke",
        "study_axes": ["runtime", "cpu", "smoke"],
    },
    "runtime_smoke_gpu_g1_amp_v1": {
        "stage": "diagnostic",
        "study_axis": "runtime_smoke",
        "study_axes": ["runtime", "gpu", "mixed_precision", "smoke"],
    },
    "smoke_gpu_v1": {
        "stage": "diagnostic",
        "study_axis": "gpu_smoke",
        "study_axes": ["runtime", "gpu", "smoke"],
    },
    "screen_graph_v1": {
        "stage": "exploratory_screen",
        "study_axis": "graph_screen",
        "study_axes": ["graph", "model_family"],
    },
    "screen_mask_v1": {
        "stage": "exploratory_screen",
        "study_axis": "masking_screen",
        "study_axes": ["masking", "model_family"],
    },
    "screen_hidden_v1": {
        "stage": "exploratory_screen",
        "study_axis": "hidden_dim_screen",
        "study_axes": ["model_hidden_dim"],
    },
    "screen_depth_v1": {
        "stage": "exploratory_screen",
        "study_axis": "depth_screen",
        "study_axes": ["model_depth"],
    },
    "screen_edge_embedding_v1": {
        "stage": "exploratory_screen",
        "study_axis": "edge_embedding_screen",
        "study_axes": ["edge_embedding_dim"],
    },
    "screen_graph_mask_interaction_v1": {
        "stage": "exploratory_screen",
        "study_axis": "graph_mask_interaction",
        "study_axes": ["graph", "masking", "interaction"],
    },
    "postlock_confirmation_v1": {
        "stage": "validation_confirmation",
        "study_axis": "postlock_confirmation",
        "study_axes": ["locked_graph", "paired_seed_confirmation"],
    },
    "final_locked_v1": {
        "stage": "locked_final",
        "study_axis": "locked_ladder",
        "study_axes": ["model_ladder", "mechanism_controls", "locked_seeds"],
    },
}

_STAGE_SEMANTICS = {
    "diagnostic": {
        "evidence_tier": "diagnostic_only",
        "interpretation_tier": "not_for_scientific_interpretation",
        "retention_tier": "retain_diagnostic_audit",
    },
    "exploratory_screen": {
        "evidence_tier": "validation_selection",
        "interpretation_tier": "selection_only",
        "retention_tier": "retain_exploratory_evidence",
    },
    "validation_confirmation": {
        "evidence_tier": "locked_validation_confirmation",
        "interpretation_tier": "validation_confirmation",
        "retention_tier": "retain_confirmation",
    },
    "locked_final": {
        "evidence_tier": "sealed_conclusion_bearing",
        "interpretation_tier": "conclusion_bearing_with_campaign_limits",
        "retention_tier": "retain_locked_final",
    },
    "posthoc_evaluation": {
        "evidence_tier": "versioned_posthoc",
        "interpretation_tier": "posthoc_only",
        "retention_tier": "retain_versioned_posthoc",
    },
    "unknown": {
        "evidence_tier": "unclassified",
        "interpretation_tier": "requires_review",
        "retention_tier": "retain_pending_review",
    },
}

_STAGE_SYNONYMS = {
    "smoke": "diagnostic",
    "runtime_smoke": "diagnostic",
    "diagnostic": "diagnostic",
    "screen": "exploratory_screen",
    "screening": "exploratory_screen",
    "exploratory": "exploratory_screen",
    "exploratory_screen": "exploratory_screen",
    "validation_confirmation": "validation_confirmation",
    "confirmation": "validation_confirmation",
    "postlock_confirmation": "validation_confirmation",
    "sealed_final": "locked_final",
    "final": "locked_final",
    "final_locked": "locked_final",
    "locked_final": "locked_final",
    "posthoc": "posthoc_evaluation",
    "posthoc_evaluation": "posthoc_evaluation",
    "unknown": "unknown",
}

_SAFE_COMPONENT = re.compile(r"[^a-z0-9]+")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CheckpointCatalogError(RuntimeError):
    """Base error for semantic checkpoint discovery and export."""


class CheckpointNotFoundError(CheckpointCatalogError):
    """Raised when a run has no matching checkpoint."""


class CheckpointVerificationError(CheckpointCatalogError):
    """Raised when a checkpoint does not match registered immutable metadata."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _safe_component(value: Any, *, fallback: str = "unknown") -> str:
    token = _SAFE_COMPONENT.sub("-", str(value or "").strip().lower()).strip("-")
    return token[:80].rstrip("-") or fallback


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return []


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _boolean(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _decode_row_json(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for field in (
        "config_json",
        "canonical_config_json",
        "campaign_config_json",
        "metadata_json",
        "rules_json",
    ):
        if field in result:
            result[field] = _mapping(result[field])
    return result


def _load_registry_rows(
    registry: Registry,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[int, dict[str, Any]],
]:
    """Bulk-read mandatory and optional registry metadata without mutations."""

    with registry.connect() as connection:
        tables = _table_names(connection)
        rows = [
            _decode_row_json(dict(row))
            for row in connection.execute(
                """
                SELECT r.*,
                       a.artifact_id AS checkpoint_artifact_id,
                       a.kind AS checkpoint_artifact_kind,
                       a.path AS checkpoint_path,
                       a.sha256 AS checkpoint_sha256,
                       a.size_bytes AS checkpoint_size_bytes,
                       a.status AS checkpoint_artifact_status,
                       a.created_at AS checkpoint_registered_at,
                       c.name AS campaign_name,
                       c.config_json AS campaign_config_json,
                       v.canonical_config_json AS canonical_config_json
                FROM runs r
                JOIN artifacts a ON a.run_id = r.run_id
                LEFT JOIN campaigns c ON c.campaign_id = r.campaign_id
                LEFT JOIN variants v ON v.scientific_id = r.scientific_id
                ORDER BY r.run_id, a.artifact_id
                """
            )
        ]

        categories: dict[str, dict[str, Any]] = {}
        if "run_categories" in tables:
            categories = {
                str(row["run_id"]): _decode_row_json(dict(row))
                for row in connection.execute(
                    "SELECT * FROM run_categories ORDER BY run_id"
                )
            }

        aliases: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if "run_aliases" in tables:
            for row in connection.execute(
                """
                SELECT * FROM run_aliases
                ORDER BY run_id, preferred DESC, alias_type, alias_id
                """
            ):
                aliases[str(row["run_id"])].append(_decode_row_json(dict(row)))

        checkpoint_metadata: dict[int, dict[str, Any]] = {}
        if "checkpoint_catalog" in tables:
            checkpoint_metadata = {
                int(row["artifact_id"]): _decode_row_json(dict(row))
                for row in connection.execute(
                    "SELECT * FROM checkpoint_catalog ORDER BY artifact_id"
                )
            }
    return rows, categories, dict(aliases), checkpoint_metadata


def _legacy_reference(
    reference: str | Path | None,
    *,
    index_path: Path,
    paths: ProjectPaths,
) -> Path | None:
    if reference is None or str(reference).strip() == "":
        return None
    candidate = Path(str(reference))
    candidates = (
        [candidate]
        if candidate.is_absolute()
        else [
            paths.project_root / candidate,
            index_path.parent / candidate,
            paths.artifact_root / candidate,
        ]
    )
    for item in candidates:
        if item.exists() or item.is_symlink():
            return item.resolve(strict=False)
    return candidates[0].resolve(strict=False)


def load_legacy_run_index(paths: ProjectPaths) -> dict[str, dict[str, Any]]:
    """Load audited legacy records keyed only by exact ``legacy_run_id``."""

    legacy_root = paths.artifact_root / "legacy_runs"
    if not legacy_root.is_dir():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for index_path in sorted(
        legacy_root.rglob("index.jsonl"), key=lambda item: item.as_posix()
    ):
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CheckpointCatalogError(
                        f"Invalid JSON in {index_path}:{line_number}: {error}"
                    ) from error
                if not isinstance(value, Mapping):
                    raise CheckpointCatalogError(
                        f"Legacy index row is not a mapping: "
                        f"{index_path}:{line_number}"
                    )
                run_id = value.get("legacy_run_id")
                if not isinstance(run_id, str) or not run_id.strip():
                    raise CheckpointCatalogError(
                        f"Legacy index row has no legacy_run_id: "
                        f"{index_path}:{line_number}"
                    )
                record = dict(value)
                record["_index_path"] = index_path.resolve(strict=False).as_posix()
                existing = records.get(run_id)
                if existing is not None and canonical_json(existing) != canonical_json(
                    record
                ):
                    raise CheckpointCatalogError(
                        f"Conflicting exact legacy_run_id records: {run_id}"
                    )
                records[run_id] = record
    return records


def _read_native_manifest(
    legacy: Mapping[str, Any],
    *,
    paths: ProjectPaths,
) -> tuple[dict[str, Any], Path | None]:
    index_path_value = legacy.get("_index_path")
    if not isinstance(index_path_value, str):
        return {}, None
    index_path = Path(index_path_value)
    manifest_path = _legacy_reference(
        legacy.get("manifest_path"), index_path=index_path, paths=paths
    )
    if manifest_path is None or not manifest_path.is_file():
        return {}, manifest_path
    try:
        if manifest_path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        else:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise CheckpointCatalogError(
            f"Cannot parse legacy run manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise CheckpointCatalogError(
            f"Legacy run manifest is not a mapping: {manifest_path}"
        )
    return dict(value), manifest_path


def _read_run_summary(
    row: Mapping[str, Any],
    *,
    paths: ProjectPaths,
) -> dict[str, Any]:
    artifact_value = row.get("artifact_path")
    if artifact_value is None or str(artifact_value).strip() == "":
        return {}
    artifact_root = Path(str(artifact_value))
    if not artifact_root.is_absolute():
        artifact_root = paths.project_root / artifact_root
    summary_path = artifact_root / "summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        return {}
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CheckpointCatalogError(
            f"Cannot parse canonical run summary {summary_path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise CheckpointCatalogError(
            f"Canonical run summary is not a mapping: {summary_path}"
        )
    return dict(value)


def _is_checkpoint_artifact(kind: Any, path: Any) -> bool:
    normalized = str(kind or "").strip().lower().replace("-", "_")
    name = Path(str(path or "")).name.lower()
    if normalized == "checkpoints":
        return name.endswith((".ckpt", ".pt", ".pth"))
    if normalized in CHECKPOINT_ARTIFACT_KINDS:
        return True
    if normalized.startswith("checkpoint_"):
        return True
    return False


def _checkpoint_role(
    *,
    kind: Any,
    path: Any,
    registered: Mapping[str, Any],
    native_manifest: Mapping[str, Any],
) -> str:
    explicit = registered.get("role")
    if isinstance(explicit, str) and explicit.strip():
        return _safe_component(explicit)
    normalized = str(kind or "").strip().lower().replace("-", "_")
    name = Path(str(path)).name.lower()
    if normalized == "checkpoint_best" or name == "best.ckpt":
        return "best"
    if normalized == "checkpoint_last" or name == "last.ckpt":
        return "last"
    if normalized == "legacy_checkpoint":
        native_config = _mapping(native_manifest.get("config"))
        training_config = _mapping(native_config.get("training"))
        training_result = _mapping(native_manifest.get("training"))
        if bool(training_config.get("restore_best")) or training_result.get(
            "best_epoch"
        ) is not None:
            return "best"
        return "legacy"
    if "best" in name:
        return "best"
    if "last" in name:
        return "last"
    return "checkpoint"


def _checkpoint_epoch(
    *,
    role: str,
    registered: Mapping[str, Any],
    run_summary: Mapping[str, Any],
    training: Mapping[str, Any],
) -> int | None:
    """Return the epoch represented by the checkpoint catalog row.

    Schema v1 retained the historical ``best_epoch`` column name.  For a
    canonical ``last`` checkpoint, that column instead records the final
    checkpoint epoch and must not fall back to a selected best epoch.
    """

    registered_metadata = _mapping(registered.get("metadata_json"))
    summary_checkpoint = _mapping(run_summary.get("checkpoint"))
    if role == "last":
        return _integer(
            _first(
                registered_metadata.get("final_epoch"),
                registered_metadata.get("checkpoint_epoch"),
                registered_metadata.get("epoch"),
                summary_checkpoint.get("final_epoch"),
                summary_checkpoint.get("checkpoint_epoch"),
                summary_checkpoint.get("epoch"),
                run_summary.get("final_epoch"),
                run_summary.get("checkpoint_epoch"),
                training.get("final_epoch"),
                registered.get("best_epoch"),
            )
        )
    return _integer(
        _first(
            registered.get("best_epoch"),
            registered_metadata.get("best_epoch"),
            registered_metadata.get("checkpoint_epoch"),
            registered_metadata.get("epoch"),
            summary_checkpoint.get("best_epoch"),
            summary_checkpoint.get("checkpoint_epoch"),
            summary_checkpoint.get("epoch"),
            run_summary.get("best_epoch"),
            training.get("best_epoch"),
        )
    )


def _normalize_stage(value: Any) -> str:
    token = _safe_component(value).replace("-", "_")
    normalized = _STAGE_SYNONYMS.get(token, token)
    return normalized if normalized in CONTROLLED_STAGES else "unknown"


def _classification(
    *,
    source_batch: str | None,
    registered: Mapping[str, Any],
    run_config: Mapping[str, Any],
    campaign_config: Mapping[str, Any],
) -> dict[str, Any]:
    if source_batch in _BATCH_CLASSIFICATION:
        base = dict(_BATCH_CLASSIFICATION[str(source_batch)])
    else:
        run_classification = _mapping(run_config.get("classification"))
        campaign_classification = _mapping(campaign_config.get("classification"))
        raw_stage = _first(
            registered.get("lifecycle_stage"),
            run_classification.get("lifecycle_stage"),
            run_classification.get("stage"),
            campaign_classification.get("lifecycle_stage"),
            campaign_classification.get("stage"),
        )
        stage = _normalize_stage(raw_stage)
        axes = _first(
            registered.get("study_axis"),
            run_classification.get("study_axis"),
            campaign_classification.get("study_axis"),
        )
        study_axis = _safe_component(axes).replace("-", "_")
        declared_axes = _first(
            run_classification.get("study_axes"),
            campaign_classification.get("study_axes"),
        )
        study_axes = [
            _safe_component(item).replace("-", "_")
            for item in _list(declared_axes)
            if str(item).strip()
        ]
        base = {
            "stage": stage,
            "study_axis": study_axis,
            "study_axes": study_axes or [study_axis],
        }
    semantics = _STAGE_SEMANTICS[base["stage"]]
    return {**base, **semantics}


def _source_batch(
    legacy: Mapping[str, Any],
    registered: Mapping[str, Any],
    run: Mapping[str, Any],
) -> str:
    return str(
        _first(
            legacy.get("batch"),
            registered.get("source_batch"),
            run.get("campaign_id"),
            "unknown",
        )
    )


def _known_from_legacy(
    legacy: Mapping[str, Any], field: str, value: Any
) -> bool:
    confidence = _mapping(legacy.get("inference_confidence")).get(field)
    return value is not None and str(confidence or "").lower() != "unknown"


def _graph_values(
    legacy: Mapping[str, Any],
    run_config: Mapping[str, Any],
    native_manifest: Mapping[str, Any],
    registered: Mapping[str, Any],
) -> dict[str, Any]:
    legacy_graph = _mapping(legacy.get("graph"))
    config_graph = _mapping(run_config.get("graph"))
    native_graph = _mapping(native_manifest.get("graph"))
    native_graph_config = _mapping(native_graph.get("config"))
    graph_id = _first(
        legacy_graph.get("graph_id"),
        native_graph.get("graph_id"),
        config_graph.get("graph_id"),
        registered.get("graph_key"),
    )
    neighbor_k = _integer(
        _first(
            legacy_graph.get("neighbor_k"),
            legacy_graph.get("k"),
            config_graph.get("neighbor_k"),
            config_graph.get("k"),
            native_graph_config.get("k"),
        )
    )
    radius = _finite_float(
        _first(
            legacy_graph.get("radius_um"),
            config_graph.get("radius_um"),
            native_graph_config.get("radius_um"),
        )
    )
    symmetry = _first(
        legacy_graph.get("symmetry"),
        config_graph.get("symmetry"),
        native_graph_config.get("symmetry"),
    )
    if graph_id is None:
        components = []
        if neighbor_k is not None:
            components.append(f"k{neighbor_k}")
        if radius is not None:
            components.append(f"r{radius:g}")
        if symmetry:
            components.append(str(symmetry))
        graph_id = "_".join(components) or "unknown"
    return {
        "graph_id": str(graph_id),
        "graph_key": str(_first(registered.get("graph_key"), graph_id)),
        "neighbor_k": neighbor_k,
        "radius_um": radius,
        "symmetry": str(symmetry) if symmetry is not None else None,
        "min_distance_um": _finite_float(
            _first(
                legacy_graph.get("min_distance_um"),
                config_graph.get("min_distance_um"),
                native_graph_config.get("min_distance_um"),
            )
        ),
        "rewired": _boolean(
            _first(
                legacy_graph.get("rewired"),
                _mapping(native_graph.get("rewire")).get("enabled"),
                _mapping(_mapping(native_manifest.get("config")).get("run")).get(
                    "rewired"
                ),
            )
        ),
        "edge_control": str(
            _first(
                legacy_graph.get("edge_control"),
                native_graph.get("edge_control"),
                "none",
            )
        ),
    }


def _condition(
    *,
    model_name: str,
    graph: Mapping[str, Any],
    native_manifest: Mapping[str, Any],
) -> tuple[str, str]:
    standards_lock = _mapping(native_manifest.get("standards_lock"))
    explicit = standards_lock.get("condition")
    if isinstance(explicit, str) and explicit.strip():
        condition = explicit.strip().lower().replace("-", "_")
    elif model_name == "b0-matched":
        condition = "b0_parameter_matched"
    elif model_name == "qkv-gat-matched-self":
        condition = "qkv_parameter_matched_self"
    elif model_name == "hybrid-count-matched-self":
        condition = "hybrid_count_parameter_matched_self"
    elif model_name == "broad-field":
        condition = "broad_field"
    elif bool(graph.get("rewired")):
        condition = f"{model_name}_rewired"
    elif str(graph.get("edge_control", "none")) != "none":
        condition = f"{model_name}_{graph['edge_control']}"
    elif model_name in {
        "g1",
        "g2",
        "g3",
        "hybrid-count-gat",
        "qkv-gat",
        "relative-qkv-gat",
    }:
        condition = f"{model_name}_true"
    else:
        condition = model_name

    if condition in {"b0", "b0_parameter_matched", "b1", "broad_field"}:
        role = "baseline"
    elif condition in {
        "hybrid_count_parameter_matched_self",
        "qkv_parameter_matched_self",
    }:
        role = "parameter_matched_self_control"
    elif condition.endswith("_rewired"):
        role = "mechanism_breaking_control"
    elif condition.endswith(("_zero", "_distance_only", "_permuted")):
        role = "edge_feature_control"
    elif condition.startswith(
        (
            "g1_",
            "g2_",
            "g3_",
            "hybrid-count-gat_",
            "hybrid_count_gat_",
            "qkv-gat_",
            "qkv_gat_",
            "relative-qkv-gat_",
            "relative_qkv_gat_",
        )
    ):
        role = "candidate_model"
    else:
        role = "unspecified"
    return condition, role


def _relative_to_project(path: Path, paths: ProjectPaths) -> str | None:
    try:
        return path.absolute().relative_to(
            paths.project_root.absolute()
        ).as_posix()
    except ValueError:
        return None


def _generated_semantic_alias(record: Mapping[str, Any]) -> str:
    categories = {
        "lifecycle_stage": str(record["lifecycle_stage"]),
        "study_axis": str(record["study_axis"]),
        "model": str(record["model_name"]),
        "graph": str(record["graph_id"]),
        "mask": str(record["masking_type"]),
        "edge_feature_state": (
            "enabled" if record.get("use_edge_features") else "disabled"
        ),
        "embedding_dim": int(record["embedding_dim"]),
        "seed": int(record["seed"]),
        "fold": record.get("fold") if record.get("fold_known") else None,
        "attempt": record.get("attempt") if record.get("attempt_known") else None,
        "variant_evidence": {
            "campaign_id": record.get("campaign_id"),
            "scientific_id": record.get("scientific_id"),
            "dataset_id": record.get("dataset_id"),
            "dataset_version": record.get("dataset_version"),
            "split_id": record.get("split_id"),
            "condition": record.get("condition"),
        },
    }
    function = getattr(identifier_api, "semantic_run_alias", None)
    if callable(function):
        try:
            generated = function(
                primary_run_id=str(record["run_id"]),
                categories=categories,
                historical=bool(record["historical"]),
            )
            return str(generated)
        except (TypeError, ValueError):
            # A registry created during an identifier-schema transition remains
            # browsable through the deterministic local fallback below.
            pass

    visible = ".".join(
        _safe_component(value)
        for value in (
            "hist" if record["historical"] else "run",
            record["lifecycle_stage"],
            record["study_axis"],
            record["model_name"],
            record["graph_id"],
            record["masking_type"],
            f"d{record['embedding_dim']}",
            f"s{int(record['seed']):03d}",
            (
                f"f{int(record['fold']):02d}"
                if record.get("fold_known")
                else "fna"
            ),
            (
                f"a{int(record['attempt']):02d}"
                if record.get("attempt_known")
                else "ana"
            ),
        )
    )
    digest = canonical_sha256(
        {"run_id": record["run_id"], "categories": categories}
    )[:12]
    return f"{visible}.x{digest}"


def _semantic_alias(
    *,
    record: Mapping[str, Any],
    registered_aliases: Sequence[Mapping[str, Any]],
) -> tuple[str, str, list[dict[str, Any]]]:
    aliases = [dict(item) for item in registered_aliases]
    preferred = next(
        (
            str(item["alias_id"])
            for item in aliases
            if bool(item.get("preferred")) and item.get("alias_id")
        ),
        None,
    )
    semantic_registered = next(
        (
            str(item["alias_id"])
            for item in aliases
            if item.get("alias_type") == "semantic" and item.get("alias_id")
        ),
        None,
    )
    if semantic_registered:
        return semantic_registered, preferred or semantic_registered, aliases
    generated = _generated_semantic_alias(record)
    return generated, preferred or generated, aliases


def _build_record(
    row: Mapping[str, Any],
    *,
    paths: ProjectPaths,
    legacy: Mapping[str, Any],
    registered_category: Mapping[str, Any],
    registered_aliases: Sequence[Mapping[str, Any]],
    registered_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    run_config = _mapping(row.get("config_json"))
    canonical_config = _mapping(row.get("canonical_config_json"))
    if canonical_config:
        run_config = canonical_config | run_config
    campaign_config = _mapping(row.get("campaign_config_json"))
    native_manifest, native_manifest_path = _read_native_manifest(
        legacy, paths=paths
    )
    run_summary = _read_run_summary(row, paths=paths)
    historical = bool(legacy) or str(row.get("checkpoint_artifact_kind")) == (
        "legacy_checkpoint"
    )

    source_batch = _source_batch(legacy, registered_category, row)
    classification = _classification(
        source_batch=source_batch,
        registered=registered_category,
        run_config=run_config,
        campaign_config=campaign_config,
    )
    graph = _graph_values(
        legacy, run_config, native_manifest, registered_category
    )

    model = _mapping(run_config.get("model"))
    hyperparameters = _mapping(legacy.get("hyperparameters"))
    model_name = str(
        _first(
            legacy.get("model_family"),
            model.get("name"),
            registered_category.get("model_key"),
            row.get("model_family"),
            "unknown",
        )
    )
    model_family = str(
        _first(
            legacy.get("model_family"),
            row.get("model_family"),
            model.get("family"),
            model_name,
        )
    )
    masking = _mapping(run_config.get("masking"))
    masking_type = str(
        _first(
            legacy.get("masking_type"),
            masking.get("type"),
            registered_category.get("masking_key"),
            row.get("masking_type"),
            "unknown",
        )
    )
    dataset = _mapping(run_config.get("dataset"))
    dataset_id = _first(
        legacy.get("dataset_id"), row.get("dataset_id"), dataset.get("dataset_id")
    )
    dataset_version = _first(
        legacy.get("dataset_version"), row.get("dataset_version"), dataset.get("version")
    )
    split_id = _first(
        legacy.get("split_id"), row.get("split_id"), dataset.get("split_id")
    )
    embedding_dim = _integer(
        _first(
            hyperparameters.get("embedding_dim"),
            row.get("embedding_dim"),
            model.get("embedding_dim"),
            1,
        )
    )
    assert embedding_dim is not None
    edge_embedding_dim = _integer(
        _first(
            hyperparameters.get("edge_embedding_dim"),
            model.get("edge_embedding_dim"),
        )
    )
    graph_layers = _integer(
        _first(hyperparameters.get("graph_layers"), model.get("graph_layers"))
    )
    features = _mapping(run_config.get("features"))
    use_edge_features = _boolean(
        _first(
            legacy.get("use_edge_features"),
            row.get("use_edge_features"),
            features.get("use_edge_features"),
        )
    )
    if use_edge_features is None:
        use_edge_features = model_name in {
            "g2",
            "g3",
            "hybrid-count-gat",
            "qkv-gat",
        }

    if historical:
        seed = _integer(legacy.get("seed"))
        fold = _integer(legacy.get("fold"))
        attempt = _integer(legacy.get("attempt"))
        seed_known = _known_from_legacy(legacy, "seed", seed)
        fold_known = _known_from_legacy(legacy, "fold", fold)
        attempt_known = _known_from_legacy(legacy, "attempt", attempt)
    else:
        seed = _integer(row.get("seed"))
        fold = _integer(row.get("fold"))
        attempt = _integer(row.get("attempt"))
        seed_known = bool(
            _first(registered_category.get("seed_known"), seed is not None)
        )
        fold_known = bool(
            _first(registered_category.get("fold_known"), fold is not None)
        )
        attempt_known = bool(
            _first(registered_category.get("attempt_known"), attempt is not None)
        )
    seed = seed if seed is not None else _integer(row.get("seed"))
    if seed is None:
        seed = 0
        seed_known = False
    if not fold_known:
        fold = None
    if not attempt_known:
        attempt = None

    condition, condition_role = _condition(
        model_name=model_name, graph=graph, native_manifest=native_manifest
    )
    timing = _mapping(native_manifest.get("timing"))
    training = _mapping(native_manifest.get("training"))
    native_artifacts = _mapping(native_manifest.get("artifacts"))
    checkpoint_role = _checkpoint_role(
        kind=row.get("checkpoint_artifact_kind"),
        path=row.get("checkpoint_path"),
        registered=registered_checkpoint,
        native_manifest=native_manifest,
    )
    checkpoint_path = Path(str(row["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = paths.project_root / checkpoint_path
    # Preserve the registered lexical path. Resolving here would erase a
    # direct symlink and defeat verify_checkpoint_record's source-symlink ban.
    checkpoint_path = checkpoint_path.absolute()
    checkpoint_sha = str(row.get("checkpoint_sha256") or "").lower() or None
    size_bytes = _integer(row.get("checkpoint_size_bytes"))
    best_epoch = _checkpoint_epoch(
        role=checkpoint_role,
        registered=registered_checkpoint,
        run_summary=run_summary,
        training=training,
    )
    monitored_metric = _first(
        registered_checkpoint.get("monitored_metric"),
        row.get("primary_metric_name"),
        run_summary.get("primary_metric_name"),
    )
    monitored_value = _finite_float(
        _first(
            registered_checkpoint.get("monitored_value"),
            row.get("primary_metric_value"),
            run_summary.get("primary_metric_value"),
            training.get("best_validation_loss"),
        )
    )
    evaluation = _mapping(run_config.get("evaluation"))
    configured_direction = str(
        evaluation.get("primary_direction", "")
    ).strip().lower()
    configured_mode = {
        "minimize": "min",
        "maximize": "max",
        "min": "min",
        "max": "max",
    }.get(configured_direction)
    registry_mode = {
        "minimize": "min",
        "maximize": "max",
    }.get(PRIMARY_METRIC_DIRECTIONS.get(str(monitored_metric), ""))
    monitored_mode = _first(
        registered_checkpoint.get("monitored_mode"),
        configured_mode,
        registry_mode,
        (
            "min"
            if monitored_metric
            and any(
                token in str(monitored_metric).lower()
                for token in ("loss", "error", "brier")
            )
            else None
        ),
    )
    index_checkpoint_files = [
        str(value) for value in _list(legacy.get("checkpoint_files"))
    ]
    declared_in_index = (
        any(checkpoint_path.as_posix().endswith(value) for value in index_checkpoint_files)
        if historical
        else None
    )

    record: dict[str, Any] = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "checkpoint_id": "ckpt_"
        + canonical_sha256(
            {
                "run_id": row["run_id"],
                "artifact_id": row.get("checkpoint_artifact_id"),
                "path": checkpoint_path.as_posix(),
                "role": checkpoint_role,
            }
        )[:20],
        "run_id": str(row["run_id"]),
        "native_run_id": _first(
            legacy.get("legacy_native_run_id"), native_manifest.get("run_id")
        ),
        "historical": historical,
        "campaign_id": row.get("campaign_id"),
        "campaign_name": row.get("campaign_name"),
        "scientific_id": row.get("scientific_id"),
        "repro_id": row.get("repro_id"),
        "status": row.get("status"),
        "started_at": _first(timing.get("started_at"), row.get("start_time")),
        "completed_at": _first(timing.get("completed_at"), row.get("end_time")),
        "timestamp_basis": _first(
            registered_category.get("timestamp_basis"),
            "native_manifest" if timing.get("started_at") else "registry",
        ),
        "lifecycle_stage": classification["stage"],
        "study_axis": classification["study_axis"],
        "study_axes": classification["study_axes"],
        "source_batch": source_batch,
        "evidence_tier": classification["evidence_tier"],
        "interpretation_tier": classification["interpretation_tier"],
        "retention_tier": _first(
            registered_checkpoint.get("retention_class"),
            registered_category.get("retention_class"),
            classification["retention_tier"],
        ),
        "retention_action": "retain",
        "condition": condition,
        "condition_role": condition_role,
        "category_key": _first(
            registered_category.get("category_key"),
            "/".join(
                _safe_component(value)
                for value in (
                    classification["stage"],
                    classification["study_axis"],
                    model_name,
                    condition,
                    graph["graph_id"],
                    masking_type,
                    f"d{embedding_dim}",
                )
            ),
        ),
        "variant_label": registered_category.get("variant_label"),
        "model_name": model_name,
        "model_family": model_family,
        "masking_type": masking_type,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "split_id": split_id,
        "use_edge_features": bool(use_edge_features),
        "edge_feature_state": "enabled" if use_edge_features else "disabled",
        "embedding_dim": embedding_dim,
        "edge_embedding_dim": edge_embedding_dim,
        "graph_layers": graph_layers,
        **graph,
        "seed": seed,
        "seed_known": seed_known,
        "fold": fold,
        "fold_known": fold_known,
        "attempt": attempt,
        "attempt_known": attempt_known,
        "classification_confidence": _first(
            registered_category.get("classification_confidence"),
            _mapping(legacy.get("inference_confidence")).get(
                "run_boundary", "high" if not historical else "unknown"
            ),
        ),
        "sealed_test_opened": _boolean(native_manifest.get("sealed_test_opened")),
        "promoted": row.get("promoted_at") is not None,
        "promoted_at": row.get("promoted_at"),
        "primary_metric_name": _first(
            row.get("primary_metric_name"),
            run_summary.get("primary_metric_name"),
        ),
        "primary_metric_value": _finite_float(
            _first(
                row.get("primary_metric_value"),
                run_summary.get("primary_metric_value"),
            )
        ),
        "checkpoint_role": checkpoint_role,
        "best_epoch": best_epoch,
        "monitored_metric": monitored_metric,
        "monitored_mode": monitored_mode,
        "monitored_value": monitored_value,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_relative_path": _relative_to_project(checkpoint_path, paths),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_size_bytes": size_bytes,
        "checkpoint_artifact_id": _integer(row.get("checkpoint_artifact_id")),
        "checkpoint_artifact_kind": row.get("checkpoint_artifact_kind"),
        "checkpoint_artifact_status": row.get("checkpoint_artifact_status"),
        "checkpoint_registered_at": row.get("checkpoint_registered_at"),
        "verification_status": _first(
            registered_checkpoint.get("verification_status"), "declared"
        ),
        "verification_checked": False,
        "legacy_index_path": legacy.get("_index_path"),
        "legacy_manifest_path": (
            native_manifest_path.as_posix()
            if native_manifest_path is not None
            else None
        ),
        "legacy_checkpoint_declared_in_index": declared_in_index,
        "native_checkpoint_name": native_artifacts.get("checkpoint"),
        "registry_category": dict(registered_category) or None,
        "registry_checkpoint_metadata": dict(registered_checkpoint) or None,
        "content_duplicate_group": None,
        "content_duplicate_count": 1,
        "duplicate_representative_checkpoint_id": None,
        "duplicate_representative_run_id": None,
        "duplicate_of_checkpoint_id": None,
        "duplicate_group_redundant_bytes": 0,
    }
    semantic_alias, preferred_alias, aliases = _semantic_alias(
        record=record, registered_aliases=registered_aliases
    )
    record["semantic_alias"] = semantic_alias
    record["preferred_alias"] = preferred_alias
    record["registered_aliases"] = aliases
    return record


def _annotate_duplicates(records: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        digest = record.get("checkpoint_sha256")
        if isinstance(digest, str) and _SHA256.fullmatch(digest):
            groups[digest].append(record)
    for digest, group in groups.items():
        if len(group) < 2:
            continue
        ordered = sorted(
            group,
            key=lambda item: (
                str(item["run_id"]),
                str(item["checkpoint_role"]),
                str(item["checkpoint_path"]),
                str(item["checkpoint_id"]),
            ),
        )
        representative = ordered[0]
        sizes = [
            int(item["checkpoint_size_bytes"])
            for item in ordered
            if item.get("checkpoint_size_bytes") is not None
        ]
        one_size = sizes[0] if sizes and len(set(sizes)) == 1 else 0
        redundant = max(0, len(ordered) - 1) * one_size
        group_id = f"ckdup_{digest[:20]}"
        for item in ordered:
            item["content_duplicate_group"] = group_id
            item["content_duplicate_count"] = len(ordered)
            item["duplicate_representative_checkpoint_id"] = representative[
                "checkpoint_id"
            ]
            item["duplicate_representative_run_id"] = representative["run_id"]
            item["duplicate_of_checkpoint_id"] = (
                None
                if item is representative
                else representative["checkpoint_id"]
            )
            item["duplicate_group_redundant_bytes"] = redundant


def _record_sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _STAGE_ORDER.get(str(record.get("lifecycle_stage")), 99),
        str(record.get("source_batch") or ""),
        str(record.get("study_axis") or ""),
        str(record.get("model_name") or ""),
        str(record.get("condition") or ""),
        str(record.get("scientific_id") or ""),
        int(record.get("seed") or 0),
        str(record.get("run_id") or ""),
        str(record.get("checkpoint_role") or ""),
    )


def build_checkpoint_catalog(
    registry: Registry,
    paths: ProjectPaths | None = None,
    *,
    run_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Build the complete read-only catalog from registry and manifest metadata."""

    selected_paths = paths or current_paths()
    rows, categories, aliases, checkpoint_metadata = _load_registry_rows(registry)
    selected_run_ids = (
        {str(run_id) for run_id in run_ids}
        if run_ids is not None
        else None
    )
    if selected_run_ids is not None:
        rows = [
            row for row in rows if str(row["run_id"]) in selected_run_ids
        ]
    legacy_index = (
        load_legacy_run_index(selected_paths)
        if any(
            str(row.get("checkpoint_artifact_kind")) == "legacy_checkpoint"
            for row in rows
        )
        else {}
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("checkpoint_artifact_status") in {
            ARTIFACT_STATUS_DELETED_BY_RETENTION,
            ARTIFACT_STATUS_RETENTION_PENDING,
        }:
            # Existing catalog rows retain the tombstoned checkpoint metadata.
            # Global re-indexing must not attempt to re-verify deleted bytes.
            continue
        if not _is_checkpoint_artifact(
            row.get("checkpoint_artifact_kind"), row.get("checkpoint_path")
        ):
            continue
        run_id = str(row["run_id"])
        record = _build_record(
            row,
            paths=selected_paths,
            legacy=legacy_index.get(run_id, {}),
            registered_category=categories.get(run_id, {}),
            registered_aliases=aliases.get(run_id, []),
            registered_checkpoint=checkpoint_metadata.get(
                int(row["checkpoint_artifact_id"]), {}
            ),
        )
        records.append(record)
    _annotate_duplicates(records)
    records.sort(key=_record_sort_key)
    return records


def run_semantics_from_configuration(
    *,
    primary_run_id: str,
    configuration: Mapping[str, Any],
    scientific_id_value: str | None = None,
) -> dict[str, Any]:
    """Return explicit registry semantics for one future execution.

    A configuration may declare ``classification`` with
    ``lifecycle_stage``, ``study_axis``, ``retention_class``,
    ``classification_confidence``, and optional ``source_batch``. Missing
    classification remains visibly unclassified; it is never guessed from a
    filename or timestamp.
    """

    classification = _mapping(configuration.get("classification"))
    lifecycle_stage = _normalize_stage(
        classification.get("lifecycle_stage", "unknown")
    )
    study_axis = _safe_component(
        classification.get("study_axis", "unclassified")
    ).replace("-", "_")
    source_batch = str(
        _first(
            classification.get("source_batch"),
            _mapping(configuration.get("campaign")).get("campaign_id"),
            configuration.get("campaign_id"),
            "unclassified",
        )
    )
    default_retention = _STAGE_SEMANTICS[lifecycle_stage]["retention_tier"]
    retention_class = str(
        classification.get("retention_class", default_retention)
    )
    confidence = str(
        classification.get("classification_confidence", "unknown")
    ).lower()
    if confidence not in {"high", "medium", "low", "unknown"}:
        raise ValueError(
            "classification.classification_confidence must be high, medium, "
            "low, or unknown."
        )

    model = _mapping(configuration.get("model"))
    masking = _mapping(configuration.get("masking"))
    dataset = _mapping(configuration.get("dataset"))
    graph = _mapping(configuration.get("graph"))
    features = _mapping(configuration.get("features"))
    experiment = _mapping(configuration.get("experiment"))
    model_name = str(_first(model.get("name"), model.get("family"), "unknown"))
    masking_type = str(_first(masking.get("type"), "unknown"))
    graph_id = _first(graph.get("graph_id"), graph.get("id"))
    if graph_id is None:
        graph_parts = [
            f"k{_integer(graph.get('neighbor_k')) or 'na'}",
            (
                f"r{_finite_float(graph.get('radius_um')):g}"
                if _finite_float(graph.get("radius_um")) is not None
                else "rna"
            ),
            _safe_component(graph.get("symmetry")),
        ]
        if _boolean(graph.get("rewired")):
            graph_parts.append("rewired")
        graph_id = "-".join(graph_parts)
    use_edge_features = bool(_boolean(features.get("use_edge_features")))
    embedding_dim = _integer(model.get("embedding_dim"))
    seed = _integer(configuration.get("seed"))
    fold = _integer(configuration.get("fold"))
    attempt = _integer(configuration.get("attempt"))
    if embedding_dim is None or embedding_dim < 1:
        raise ValueError("model.embedding_dim must be a positive integer.")
    if seed is None or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if fold is None or fold < 0:
        raise ValueError("fold must be a non-negative integer.")
    if attempt is None or attempt < 1:
        raise ValueError("attempt must be a positive integer.")
    condition, condition_role = _condition(
        model_name=model_name,
        graph={
            "rewired": _boolean(graph.get("rewired")),
            "edge_control": str(graph.get("edge_control", "none")),
        },
        native_manifest={},
    )
    scientific_value = scientific_id_value or identifier_api.scientific_id(
        configuration
    )
    alias = identifier_api.semantic_run_alias(
        primary_run_id=primary_run_id,
        categories={
            "lifecycle_stage": lifecycle_stage,
            "study_axis": study_axis,
            "model": model_name,
            "graph": str(graph_id),
            "mask": masking_type,
            "edge_feature_state": (
                "enabled" if use_edge_features else "disabled"
            ),
            "embedding_dim": embedding_dim,
            "seed": seed,
            "fold": fold,
            "attempt": attempt,
            "variant_evidence": {
                "scientific_id": scientific_value,
                "dataset_id": dataset.get("dataset_id"),
                "dataset_version": dataset.get("version"),
                "split_id": dataset.get("split_id"),
                "condition": condition,
            },
        },
        historical=False,
    )
    category_key = "/".join(
        _safe_component(value)
        for value in (
            lifecycle_stage,
            study_axis,
            dataset.get("dataset_id"),
            model_name,
            condition,
            graph_id,
            masking_type,
            "edge-enabled" if use_edge_features else "edge-disabled",
            f"d{embedding_dim}",
        )
    )
    return {
        "lifecycle_stage": lifecycle_stage,
        "study_axis": study_axis,
        "source_batch": source_batch,
        "variant_label": (
            str(experiment["variant_label"])
            if experiment.get("variant_label") is not None
            else None
        ),
        "model_key": model_name,
        "dataset_key": (
            f"{dataset.get('dataset_id', 'unknown')}@"
            f"{dataset.get('version', 'unknown')}"
        ),
        "masking_key": masking_type,
        "graph_key": str(graph_id),
        "feature_key": "edge-enabled" if use_edge_features else "edge-disabled",
        "embedding_key": f"d{embedding_dim}",
        "seed_known": True,
        "fold_known": True,
        "attempt_known": True,
        "retention_class": retention_class,
        "category_key": category_key,
        "classification_confidence": confidence,
        "timestamp_basis": "registry",
        "rules": {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "classification_source": (
                "resolved_config"
                if classification
                else "missing_explicit_classification"
            ),
            "scientific_id": scientific_value,
            "condition": condition,
            "condition_role": condition_role,
            "evidence_tier": _STAGE_SEMANTICS[lifecycle_stage][
                "evidence_tier"
            ],
            "interpretation_tier": _STAGE_SEMANTICS[lifecycle_stage][
                "interpretation_tier"
            ],
        },
        "semantic_alias": alias,
        "preferred_alias_type": "semantic",
        "alias_metadata": {
            "semantic": {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "generated_from": "resolved_config",
                "primary_run_id": primary_run_id,
            }
        },
    }


def index_checkpoint_catalog(
    registry: Registry,
    paths: ProjectPaths | None = None,
    *,
    run_reference: str | None = None,
    verify: bool = True,
    update: bool = False,
) -> dict[str, Any]:
    """Persist derived run/checkpoint semantics without touching artifacts.

    Verification is completed before the first registry write. Checkpoint
    metadata is a derived, monotonically reconciled index, including any
    exact-content peers. Changing an existing run category or semantic alias
    still requires the caller to opt in with ``update=True``.
    """

    selected_paths = paths or current_paths()
    requested_run_id: str | None = None
    if run_reference is not None:
        requested_run_id = _resolve_run_reference(registry, run_reference)
        records = build_checkpoint_catalog(
            registry,
            selected_paths,
            run_ids=[requested_run_id],
        )
        if not records:
            raise CheckpointNotFoundError(
                f"No checkpoint is registered for run {run_reference!r}."
            )
        digests = sorted(
            {
                str(row["checkpoint_sha256"]).lower()
                for row in records
                if isinstance(row.get("checkpoint_sha256"), str)
                and _SHA256.fullmatch(str(row["checkpoint_sha256"]).lower())
            }
        )
        if digests:
            placeholders = ", ".join("?" for _ in digests)
            with registry.connect() as connection:
                peer_run_ids = {
                    str(row["run_id"])
                    for row in connection.execute(
                        f"""
                        SELECT DISTINCT run_id FROM artifacts
                        WHERE lower(sha256) IN ({placeholders})
                          AND instr(lower(kind), 'checkpoint') > 0
                        """,
                        digests,
                    )
                }
            if peer_run_ids != {requested_run_id}:
                records = build_checkpoint_catalog(
                    registry,
                    selected_paths,
                    run_ids=peer_run_ids,
                )
    else:
        records = build_checkpoint_catalog(registry, selected_paths)

    prepared: list[dict[str, Any]] = []
    verification_errors: list[str] = []
    for record in records:
        if not verify:
            prepared.append(dict(record))
            continue
        try:
            prepared.append(
                verify_checkpoint_record(record, paths=selected_paths)
            )
        except CheckpointVerificationError as error:
            verification_errors.append(str(error))
    if verification_errors:
        preview = "; ".join(verification_errors[:5])
        raise CheckpointVerificationError(
            f"{len(verification_errors)} checkpoint(s) failed verification: "
            f"{preview}"
        )

    registered_runs: set[str] = set()
    for record in prepared:
        run_id = str(record["run_id"])
        if run_id not in registered_runs:
            existing_category = _mapping(record.get("registry_category"))
            if not existing_category:
                rules = {
                    "schema_version": CATALOG_SCHEMA_VERSION,
                    "historical": bool(record["historical"]),
                    "study_axes": record["study_axes"],
                    "evidence_tier": record["evidence_tier"],
                    "interpretation_tier": record["interpretation_tier"],
                    "condition": record["condition"],
                    "condition_role": record["condition_role"],
                    "sealed_test_opened": record["sealed_test_opened"],
                    "retention_action": record["retention_action"],
                    "legacy_index_path": record["legacy_index_path"],
                    "fold_source": (
                        "observed" if record["fold_known"] else "unknown"
                    ),
                    "attempt_source": (
                        "observed" if record["attempt_known"] else "unknown"
                    ),
                }
                category_values = {
                    "lifecycle_stage": str(record["lifecycle_stage"]),
                    "study_axis": str(record["study_axis"]),
                    "source_batch": str(record["source_batch"]),
                    "variant_label": record.get("variant_label"),
                    "model_key": str(record["model_name"]),
                    "dataset_key": (
                        f"{record.get('dataset_id', 'unknown')}@"
                        f"{record.get('dataset_version', 'unknown')}"
                    ),
                    "masking_key": str(record["masking_type"]),
                    "graph_key": str(record["graph_key"]),
                    "feature_key": str(record["edge_feature_state"]),
                    "embedding_key": f"d{int(record['embedding_dim'])}",
                    "seed_known": bool(record["seed_known"]),
                    "fold_known": bool(record["fold_known"]),
                    "attempt_known": bool(record["attempt_known"]),
                    "retention_class": str(record["retention_tier"]),
                    "category_key": str(record["category_key"]),
                    "classification_confidence": str(
                        record["classification_confidence"]
                    ),
                    "timestamp_basis": str(record["timestamp_basis"]),
                    "rules": rules,
                }
            else:
                category_values = {
                    "lifecycle_stage": str(
                        existing_category["lifecycle_stage"]
                    ),
                    "study_axis": str(existing_category["study_axis"]),
                    "source_batch": str(existing_category["source_batch"]),
                    "variant_label": existing_category.get("variant_label"),
                    "model_key": existing_category.get("model_key"),
                    "dataset_key": existing_category.get("dataset_key"),
                    "masking_key": existing_category.get("masking_key"),
                    "graph_key": existing_category.get("graph_key"),
                    "feature_key": existing_category.get("feature_key"),
                    "embedding_key": existing_category.get("embedding_key"),
                    "seed_known": bool(existing_category["seed_known"]),
                    "fold_known": bool(existing_category["fold_known"]),
                    "attempt_known": bool(existing_category["attempt_known"]),
                    "retention_class": str(
                        existing_category["retention_class"]
                    ),
                    "category_key": str(existing_category["category_key"]),
                    "classification_confidence": str(
                        existing_category["classification_confidence"]
                    ),
                    "timestamp_basis": str(
                        existing_category["timestamp_basis"]
                    ),
                    "rules": _mapping(existing_category.get("rules_json")),
                }

            if not existing_category or update:
                generated_alias = _generated_semantic_alias(record)
                existing_semantic_alias = next(
                    (
                        item
                        for item in _list(record.get("registered_aliases"))
                        if isinstance(item, Mapping)
                        and item.get("alias_type") == "semantic"
                    ),
                    {},
                )
                alias_metadata = _mapping(
                    _mapping(existing_semantic_alias).get("metadata_json")
                ) or {
                    "schema_version": CATALOG_SCHEMA_VERSION,
                    "generated_from": (
                        "audited_legacy_index"
                        if record["historical"]
                        else "resolved_config_and_registry"
                    ),
                    "primary_run_id": run_id,
                }
                registry.register_run_semantics(
                    run_id,
                    **category_values,
                    semantic_alias=generated_alias,
                    preferred_alias_type="semantic",
                    alias_metadata={"semantic": alias_metadata},
                    update=update,
                )
                record["semantic_alias"] = generated_alias
                record["preferred_alias"] = generated_alias
            registered_runs.add(run_id)

        artifact_id = _integer(record.get("checkpoint_artifact_id"))
        if artifact_id is None:
            raise CheckpointCatalogError(
                f"Checkpoint has no registry artifact ID: "
                f"{record.get('checkpoint_id')}"
            )
        registry.register_checkpoint_metadata(
            artifact_id,
            run_id=run_id,
            role=str(record["checkpoint_role"]),
            best_epoch=_integer(record.get("best_epoch")),
            monitored_metric=(
                str(record["monitored_metric"])
                if record.get("monitored_metric") is not None
                else None
            ),
            monitored_mode=(
                str(record["monitored_mode"])
                if record.get("monitored_mode") is not None
                else None
            ),
            monitored_value=_finite_float(record.get("monitored_value")),
            retention_class=str(record["retention_tier"]),
            duplicate_group=(
                str(record["content_duplicate_group"])
                if record.get("content_duplicate_group") is not None
                else None
            ),
            duplicate_count=int(record["content_duplicate_count"]),
            verification_status=str(record["verification_status"]),
            metadata={
                "schema_version": CATALOG_SCHEMA_VERSION,
                "checkpoint_id": record["checkpoint_id"],
                "semantic_alias": record["semantic_alias"],
                "native_run_id": record["native_run_id"],
                "duplicate_representative_checkpoint_id": record[
                    "duplicate_representative_checkpoint_id"
                ],
                "duplicate_representative_run_id": record[
                    "duplicate_representative_run_id"
                ],
                "duplicate_of_checkpoint_id": record[
                    "duplicate_of_checkpoint_id"
                ],
                "evidence_tier": record["evidence_tier"],
                "interpretation_tier": record["interpretation_tier"],
            },
            # Checkpoint rows are a derived index of immutable artifact
            # metadata. Reconcile them monotonically so verification upgrades
            # and exact-duplicate group growth remain rerunnable.
            update=True,
        )

    summary = checkpoint_catalog_summary(prepared)
    return {
        "indexed_runs": len(registered_runs),
        "indexed_checkpoints": len(prepared),
        "requested_run_id": requested_run_id,
        "duplicate_peer_runs_reconciled": (
            max(
                0,
                len({str(row["run_id"]) for row in prepared}) - 1,
            )
            if requested_run_id is not None
            else 0
        ),
        "verified": bool(verify),
        **summary,
    }


_FILTER_FIELDS = frozenset(
    {
        "run_id",
        "campaign_id",
        "scientific_id",
        "repro_id",
        "lifecycle_stage",
        "study_axis",
        "source_batch",
        "evidence_tier",
        "interpretation_tier",
        "retention_tier",
        "condition",
        "condition_role",
        "model_name",
        "model_family",
        "masking_type",
        "dataset_id",
        "dataset_version",
        "split_id",
        "graph_id",
        "neighbor_k",
        "use_edge_features",
        "embedding_dim",
        "edge_embedding_dim",
        "seed",
        "fold",
        "attempt",
        "status",
        "checkpoint_role",
    }
)


def filter_checkpoint_records(
    records: Iterable[Mapping[str, Any]],
    *,
    filters: Mapping[str, Any] | None = None,
    include_failed: bool = False,
    promoted_only: bool = False,
    duplicates_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter catalog rows by exact semantic fields in deterministic order."""

    selected = dict(filters or {})
    unknown = sorted(set(selected) - _FILTER_FIELDS)
    if unknown:
        raise ValueError("Unknown checkpoint filters: " + ", ".join(unknown))

    def matches(record: Mapping[str, Any]) -> bool:
        if not include_failed and record.get("status") != "completed":
            return False
        if promoted_only and not bool(record.get("promoted")):
            return False
        if duplicates_only and int(record.get("content_duplicate_count", 1)) < 2:
            return False
        for field, wanted in selected.items():
            if wanted is None:
                continue
            choices = (
                set(wanted)
                if isinstance(wanted, Sequence)
                and not isinstance(wanted, (str, bytes, bytearray))
                else {wanted}
            )
            if record.get(field) not in choices:
                return False
        return True

    result = [dict(record) for record in records if matches(record)]
    result.sort(key=_record_sort_key)
    return result


def list_checkpoints(
    registry: Registry,
    paths: ProjectPaths | None = None,
    *,
    filters: Mapping[str, Any] | None = None,
    include_failed: bool = False,
    promoted_only: bool = False,
    duplicates_only: bool = False,
) -> list[dict[str, Any]]:
    """Build and filter the checkpoint catalog."""

    return filter_checkpoint_records(
        build_checkpoint_catalog(registry, paths),
        filters=filters,
        include_failed=include_failed,
        promoted_only=promoted_only,
        duplicates_only=duplicates_only,
    )


def _resolve_run_reference(registry: Registry, run_reference: str) -> str:
    resolver = getattr(registry, "resolve_run_id", None)
    if callable(resolver):
        try:
            return str(resolver(run_reference))
        except sqlite3.OperationalError:
            # Schema-v2 registries have no aliases table.
            pass
    return str(run_reference)


def show_checkpoint(
    registry: Registry,
    run_reference: str,
    paths: ProjectPaths | None = None,
    *,
    role: str = "best",
    verify: bool = False,
) -> dict[str, Any]:
    """Return one checkpoint for a primary run ID or registered alias."""

    run_id = _resolve_run_reference(registry, run_reference)
    records = build_checkpoint_catalog(registry, paths, run_ids=[run_id])
    if not records:
        raise CheckpointNotFoundError(
            f"No checkpoint is registered for run {run_reference!r}."
        )
    matching = [record for record in records if record["checkpoint_role"] == role]
    if not matching:
        roles = sorted({str(record["checkpoint_role"]) for record in records})
        raise CheckpointNotFoundError(
            f"Run {run_reference!r} has no {role!r} checkpoint; roles={roles}."
        )
    if len(matching) != 1:
        raise CheckpointCatalogError(
            f"Run {run_reference!r} has multiple {role!r} checkpoints: "
            f"{[item['checkpoint_id'] for item in matching]}"
        )
    return (
        verify_checkpoint_record(matching[0], paths=paths)
        if verify
        else dict(matching[0])
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_record(
    record: Mapping[str, Any],
    *,
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Verify one source checkpoint against path, size, and SHA-256 metadata."""

    selected_paths = paths or current_paths()
    artifact_status = str(record.get("checkpoint_artifact_status") or "")
    if artifact_status == ARTIFACT_STATUS_DELETED_BY_RETENTION:
        raise CheckpointVerificationError(
            "Checkpoint bytes were intentionally deleted by an explicit "
            f"retention decision: {record.get('checkpoint_path')}"
        )
    if artifact_status == ARTIFACT_STATUS_RETENTION_PENDING:
        raise CheckpointVerificationError(
            "Checkpoint retention deletion is incomplete and requires "
            f"reconciliation: {record.get('checkpoint_path')}"
        )
    raw_path = Path(str(record.get("checkpoint_path") or ""))
    if not raw_path.is_absolute():
        raw_path = selected_paths.project_root / raw_path
    if raw_path.is_symlink():
        raise CheckpointVerificationError(
            f"Registered source checkpoint may not be a symlink: {raw_path}"
        )
    if not raw_path.is_file():
        raise CheckpointVerificationError(f"Checkpoint is missing: {raw_path}")
    resolved = raw_path.resolve(strict=True)
    artifact_root = selected_paths.artifact_root.resolve(strict=False)
    if not resolved.is_relative_to(artifact_root):
        raise CheckpointVerificationError(
            f"Checkpoint is outside the configured artifact root: {resolved}"
        )
    declared_size = _integer(record.get("checkpoint_size_bytes"))
    actual_size = resolved.stat().st_size
    if declared_size is None:
        raise CheckpointVerificationError(
            f"Checkpoint has no registered size: {record.get('checkpoint_id')}"
        )
    if actual_size != declared_size:
        raise CheckpointVerificationError(
            f"Checkpoint size mismatch for {resolved}: "
            f"declared={declared_size}, actual={actual_size}"
        )
    declared_sha = str(record.get("checkpoint_sha256") or "").lower()
    if not _SHA256.fullmatch(declared_sha):
        raise CheckpointVerificationError(
            f"Checkpoint has no valid registered SHA-256: "
            f"{record.get('checkpoint_id')}"
        )
    actual_sha = _sha256_file(resolved)
    if actual_sha != declared_sha:
        raise CheckpointVerificationError(
            f"Checkpoint SHA-256 mismatch for {resolved}: "
            f"declared={declared_sha}, actual={actual_sha}"
        )
    result = dict(record)
    result.update(
        {
            "verification_checked": True,
            "verification_status": "verified",
            "verified_at": _utc_now(),
            "verified_path": resolved.as_posix(),
        }
    )
    return result


def resolve_checkpoint(
    registry: Registry,
    run_reference: str,
    paths: ProjectPaths | None = None,
    *,
    role: str = "best",
) -> Path:
    """Resolve one checkpoint only after existence, size, and SHA verification."""

    verified = show_checkpoint(
        registry, run_reference, paths, role=role, verify=True
    )
    return Path(str(verified["verified_path"]))


def checkpoint_catalog_summary(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate a catalog without ranking individual seeds."""

    rows = [dict(record) for record in records]
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = row.get("content_duplicate_group")
        if group:
            duplicate_groups[str(group)].append(row)
    redundant = sum(
        int(group[0].get("duplicate_group_redundant_bytes") or 0)
        for group in duplicate_groups.values()
    )
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "checkpoint_count": len(rows),
        "run_count": len({str(row["run_id"]) for row in rows}),
        "total_checkpoint_bytes": sum(
            int(row.get("checkpoint_size_bytes") or 0) for row in rows
        ),
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_record_count": sum(
            len(group) for group in duplicate_groups.values()
        ),
        "exact_duplicate_redundant_bytes": redundant,
        "stage_counts": dict(
            sorted(Counter(str(row["lifecycle_stage"]) for row in rows).items())
        ),
        "source_batch_counts": dict(
            sorted(Counter(str(row["source_batch"]) for row in rows).items())
        ),
        "evidence_tier_counts": dict(
            sorted(Counter(str(row["evidence_tier"]) for row in rows).items())
        ),
        "interpretation_tier_counts": dict(
            sorted(
                Counter(str(row["interpretation_tier"]) for row in rows).items()
            )
        ),
        "retention_tier_counts": dict(
            sorted(Counter(str(row["retention_tier"]) for row in rows).items())
        ),
        "model_counts": dict(
            sorted(Counter(str(row["model_name"]) for row in rows).items())
        ),
        "condition_counts": dict(
            sorted(Counter(str(row["condition"]) for row in rows).items())
        ),
        "note": (
            "Per-run metrics are catalog metadata, not a seed-ranking policy. "
            "Interpret scientific results through prespecified variant aggregates."
        ),
    }


def _exclusive_text(path: Path, value: str) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise CheckpointCatalogError(
            f"Refusing to overwrite checkpoint catalog file: {path}"
        ) from error


def _csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return canonical_json(value)
    if value is None:
        return ""
    return value


def _create_symlink_view(
    output: Path,
    records: list[dict[str, Any]],
    *,
    paths: ProjectPaths,
) -> None:
    root = output / "by-stage"
    root.mkdir()
    for record in records:
        verified = verify_checkpoint_record(record, paths=paths)
        source = Path(str(verified["verified_path"]))
        directory = (
            root
            / _safe_component(record["lifecycle_stage"])
            / _safe_component(record["model_name"])
            / _safe_component(record["condition"])
        )
        directory.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix if source.suffix else ".ckpt"
        link_name = (
            f"{record['semantic_alias']}.{record['checkpoint_role']}."
            f"{str(record['checkpoint_id'])[-8:]}{suffix}"
        )
        link = directory / link_name
        if link.exists() or link.is_symlink():
            raise CheckpointCatalogError(
                f"Refusing to overwrite checkpoint view link: {link}"
            )
        relative_target = os.path.relpath(source, start=link.parent.resolve())
        os.symlink(relative_target, link)
        if link.resolve(strict=True) != source.resolve(strict=True):
            raise CheckpointCatalogError(
                f"Checkpoint view link resolves incorrectly: {link}"
            )
        record["catalog_link"] = link.relative_to(output).as_posix()


def export_checkpoint_catalog(
    registry: Registry,
    output: str | Path,
    paths: ProjectPaths | None = None,
    *,
    filters: Mapping[str, Any] | None = None,
    include_failed: bool = False,
    promoted_only: bool = False,
    duplicates_only: bool = False,
    symlink_view: bool = False,
) -> dict[str, Any]:
    """Exclusively export JSONL, CSV, summary, README, and optional link view."""

    selected_paths = paths or current_paths()
    target = Path(output)
    if not target.is_absolute():
        target = selected_paths.project_root / target
    if target.is_symlink():
        raise FileExistsError(
            f"Refusing checkpoint catalog symlink destination: {target}"
        )
    target = target.resolve(strict=False)
    export_root = selected_paths.export_root.resolve(strict=False)
    if not target.is_relative_to(export_root) or target == export_root:
        raise CheckpointCatalogError(
            "Checkpoint catalogs must be created in a new directory below "
            f"the configured export root: {export_root}"
        )
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite checkpoint catalog: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite checkpoint catalog: {target}"
        ) from error

    records = list_checkpoints(
        registry,
        selected_paths,
        filters=filters,
        include_failed=include_failed,
        promoted_only=promoted_only,
        duplicates_only=duplicates_only,
    )
    if symlink_view:
        _create_symlink_view(target, records, paths=selected_paths)

    jsonl = "".join(canonical_json(record) + "\n" for record in records)
    _exclusive_text(target / "catalog.jsonl", jsonl)

    fields = sorted({field for record in records for field in record})
    csv_path = target / "catalog.csv"
    try:
        with csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for record in records:
                writer.writerow({key: _csv_value(record.get(key)) for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise CheckpointCatalogError(
            f"Refusing to overwrite checkpoint catalog file: {csv_path}"
        ) from error

    summary = {
        **checkpoint_catalog_summary(records),
        "generated_at": _utc_now(),
        "catalog_path": target.as_posix(),
        "symlink_view": bool(symlink_view),
    }
    _exclusive_text(
        target / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _exclusive_text(
        target / "README.md",
        (
            "# BAGM Checkpoint Catalog\n\n"
            "This directory is a generated, non-authoritative semantic view. "
            "The SQLite registry and immutable run/native manifests remain the "
            "sources of truth. Canonical run storage stays date-partitioned; "
            "no checkpoint was copied, moved, loaded, deleted, or rewritten.\n\n"
            "- `catalog.jsonl`: complete machine-readable records\n"
            "- `catalog.csv`: flat interoperable view; nested fields are JSON\n"
            "- `summary.json`: counts, bytes, and exact-content duplicate audit\n"
            + (
                "- `by-stage/`: relative symlinks to verified source checkpoints\n"
                if symlink_view
                else ""
            )
            + "\nExact duplicates are annotations only. Retention actions remain "
            "explicit review decisions. Per-run metrics must not replace "
            "multi-seed/fold variant aggregation.\n"
        ),
    )
    return {
        "output": target.as_posix(),
        "records": len(records),
        "symlink_view": bool(symlink_view),
        "summary": summary,
    }


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CHECKPOINT_ARTIFACT_KINDS",
    "CONTROLLED_STAGES",
    "CheckpointCatalogError",
    "CheckpointNotFoundError",
    "CheckpointVerificationError",
    "build_checkpoint_catalog",
    "checkpoint_catalog_summary",
    "export_checkpoint_catalog",
    "filter_checkpoint_records",
    "index_checkpoint_catalog",
    "list_checkpoints",
    "load_legacy_run_index",
    "resolve_checkpoint",
    "run_semantics_from_configuration",
    "show_checkpoint",
    "verify_checkpoint_record",
]
