"""Canonical campaign, variant, reproduction, and execution identifiers.

``scientific_id`` hashes canonical sorted JSON after recursively removing only
organizational or execution fields.  The exact excluded keys are exposed as
``SCIENTIFIC_EXCLUDED_FIELDS`` and include seed, fold, attempt/retry fields,
campaign identifiers, timestamps, host/GPU/device fields, runtime status, and
generated output/log/state/scratch/cache paths.  Dataset identity, split
identity, preprocessing choices, model parameters, features, graph choices,
masking, trainer settings, and evaluation settings are not excluded.

``repro_id`` hashes the canonical scientific payload together with Git commit,
dirty-tree fingerprint (including explicit ``null`` for a clean tree), dataset
fingerprint, split fingerprint, preprocessing version, and environment and/or
container fingerprint.  Therefore a change to any of those inputs changes the
reproduction identifier.

``run_id`` is an execution identifier rather than a configuration hash.  It
contains a UTC second, eight characters from the scientific identifier, seed,
fold, attempt, and a random suffix.  Supplying ``timestamp`` and
``unique_suffix`` makes its construction deterministic for audits and tests.

``semantic_run_alias`` is an additive, date-free display identifier.  It never
infers categories from a primary run ID, path, or configuration: callers must
supply every visible category and explicitly state whether the record is
historical.  The alias ends in canonical-JSON variant and execution hashes, so
normalized display-token collisions do not silently alias different runs.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
from typing import Any, Mapping


SCIENTIFIC_EXCLUDED_FIELDS = frozenset(
    {
        "seed",
        "random_seed",
        "model_seed",
        "mask_seed",
        "fold",
        "fold_id",
        "attempt",
        "attempt_number",
        "retry",
        "retry_of",
        "campaign",
        "campaign_id",
        "classification",
        "variant_label",
        "display_name",
        "timestamp",
        "created_at",
        "updated_at",
        "start_time",
        "started_at",
        "end_time",
        "ended_at",
        "finished_at",
        "duration",
        "duration_seconds",
        "hostname",
        "host",
        "gpu",
        "gpu_id",
        "gpu_model",
        "requested_gpu",
        "device",
        "cuda_visible_devices",
        "worker_id",
        "job_id",
        "run_id",
        "status",
        "runtime",
        "launcher",
        "heartbeat_seconds",
        "stale_after_seconds",
        "disk_safety_min_free_gb",
        "capture_stdout",
        "capture_stderr",
        "output_dir",
        "output_path",
        "artifact_dir",
        "artifact_path",
        "log_dir",
        "log_path",
        "state_root",
        "scratch_root",
        "cache_root",
        "export_root",
    }
)

_SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_HEX = re.compile(r"[0-9a-fA-F]")
_SAFE_ALIAS = re.compile(r"^(?:hist|run)(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)+$")

SEMANTIC_RUN_ALIAS_FIELDS = (
    "lifecycle_stage",
    "study_axis",
    "model",
    "graph",
    "mask",
    "edge_feature_state",
    "embedding_dim",
    "seed",
    "fold",
    "attempt",
)

_ALIAS_TEXT_LIMITS = {
    "lifecycle_stage": 20,
    "study_axis": 20,
    "model": 16,
    "graph": 28,
    "mask": 12,
    "edge_feature_state": 16,
}
_ALIAS_DIGEST_CHARACTERS = 12
_MAX_SEMANTIC_ALIAS_LENGTH = 180


class IdentifierError(ValueError):
    """Raised when an identifier input cannot be canonicalized safely."""


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise IdentifierError("Naive datetimes are not canonical; supply UTC.")
        normalized = value.astimezone(timezone.utc)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentifierError("Canonical JSON mappings require string keys.")
            if key in normalized:
                raise IdentifierError(f"Duplicate canonical key: {key}")
            normalized[key] = _canonical_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(items, key=canonical_json)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentifierError("NaN and infinity are not valid canonical JSON.")
        return value
    raise IdentifierError(
        f"Unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Serialize a value as whitespace-free, key-sorted, strict JSON."""

    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Return the full SHA-256 hex digest of canonical JSON."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scientific_payload(
    configuration: Mapping[str, Any],
    *,
    excluded_fields: frozenset[str] = SCIENTIFIC_EXCLUDED_FIELDS,
) -> dict[str, Any]:
    """Return the scientific configuration after recursive declared exclusions."""

    def visit(value: Any) -> Any:
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                raise IdentifierError(
                    "Scientific configuration mappings require string keys."
                )
            return {
                key: visit(item)
                for key, item in value.items()
                if key not in excluded_fields
            }
        if isinstance(value, (list, tuple)):
            return [visit(item) for item in value]
        return value

    result = visit(configuration)
    if not isinstance(result, dict):
        raise IdentifierError("Scientific configuration must be a mapping.")
    return _canonical_value(result)


def scientific_id(
    configuration: Mapping[str, Any],
    *,
    digest_characters: int = 16,
) -> str:
    """Hash scientific fields, excluding only ``SCIENTIFIC_EXCLUDED_FIELDS``."""

    if not 8 <= digest_characters <= 64:
        raise IdentifierError("digest_characters must be between 8 and 64.")
    digest = canonical_sha256(scientific_payload(configuration))
    return f"sci_{digest[:digest_characters]}"


def repro_id(
    configuration: Mapping[str, Any],
    *,
    git_commit: str,
    dirty_fingerprint: str | None,
    dataset_fingerprint: str,
    split_fingerprint: str,
    preprocessing_version: str,
    environment_fingerprint: str | None = None,
    container_fingerprint: str | None = None,
    digest_characters: int = 20,
) -> str:
    """Hash scientific configuration and all declared reproducibility inputs.

    Included fields are: the post-exclusion scientific payload, Git commit,
    dirty-tree fingerprint, dataset fingerprint, split fingerprint,
    preprocessing version, environment fingerprint, and container fingerprint.
    At least one environment or container fingerprint is required.
    """

    required = {
        "git_commit": git_commit,
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "preprocessing_version": preprocessing_version,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise IdentifierError(
            "Missing repro_id inputs: " + ", ".join(sorted(missing))
        )
    if not environment_fingerprint and not container_fingerprint:
        raise IdentifierError(
            "repro_id requires an environment_fingerprint or container_fingerprint."
        )
    if not 8 <= digest_characters <= 64:
        raise IdentifierError("digest_characters must be between 8 and 64.")
    payload = {
        "scientific_configuration": scientific_payload(configuration),
        "git": {
            "commit": git_commit,
            "dirty_fingerprint": dirty_fingerprint,
        },
        "data": {
            "dataset_fingerprint": dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "preprocessing_version": preprocessing_version,
        },
        "environment": {
            "environment_fingerprint": environment_fingerprint,
            "container_fingerprint": container_fingerprint,
        },
    }
    return f"rep_{canonical_sha256(payload)[:digest_characters]}"


def create_run_id(
    *,
    seed: int,
    fold: int,
    attempt: int,
    scientific_id_value: str | None = None,
    timestamp: datetime | None = None,
    unique_suffix: str | None = None,
) -> str:
    """Create a unique readable execution ID; seed/fold/attempt are not variants."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise IdentifierError("seed must be a non-negative integer.")
    if isinstance(fold, bool) or not isinstance(fold, int) or fold < 0:
        raise IdentifierError("fold must be a non-negative integer.")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise IdentifierError("attempt must be a positive integer.")
    moment = timestamp or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        raise IdentifierError("run_id timestamp must include a timezone.")
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if scientific_id_value:
        identifier_body = scientific_id_value.lower().removeprefix("sci_")
        hexadecimal = "".join(_HEX.findall(identifier_body))
        scientific_token = (hexadecimal + "00000000")[:8]
    else:
        scientific_token = "unknown0"
    suffix = (unique_suffix or secrets.token_hex(4)).lower()
    if not _SAFE_TOKEN.fullmatch(suffix) or not 4 <= len(suffix) <= 32:
        raise IdentifierError(
            "unique_suffix must be 4-32 lowercase alphanumeric, '-' or '_' characters."
        )
    return (
        f"r_{stamp}_{scientific_token}_s{seed:03d}_f{fold:02d}"
        f"_a{attempt:02d}_{suffix}"
    )


def _semantic_alias_text(value: Any, *, field: str) -> str:
    if field == "edge_feature_state" and isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, str):
        text = value.strip().lower()
    else:
        raise IdentifierError(f"{field} must be a non-empty string.")
    token = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if not token:
        raise IdentifierError(f"{field} must contain ASCII letters or digits.")
    limit = _ALIAS_TEXT_LIMITS[field]
    return token[:limit].rstrip("-")


def _semantic_alias_integer(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IdentifierError(f"{field} must be an integer.")
    if not minimum <= value <= maximum:
        raise IdentifierError(
            f"{field} must be between {minimum} and {maximum}, inclusive."
        )
    return value


def semantic_run_alias(
    *,
    primary_run_id: str,
    categories: Mapping[str, Any],
    historical: bool,
) -> str:
    """Build a deterministic, date-free semantic alias for a primary run ID.

    ``categories`` must explicitly provide all fields in
    :data:`SEMANTIC_RUN_ALIAS_FIELDS`.  ``fold`` and ``attempt`` may be
    ``None`` when the source evidence does not establish them; the visible
    tokens are then ``fna`` and ``ana``.  ``variant_evidence`` is the sole
    optional field and, when supplied, must be a mapping of additional fixed
    scientific evidence (for example dataset or split identity).

    The function does not inspect or mutate a registry and does not infer
    semantics from ``primary_run_id``.  The ``v`` hash covers fixed variant
    fields and optional variant evidence.  The ``x`` hash additionally covers
    the primary ID, historical namespace, lifecycle/study categories, and
    execution coordinates.
    """

    if not isinstance(primary_run_id, str) or not primary_run_id.strip():
        raise IdentifierError("primary_run_id must be a non-empty string.")
    if not isinstance(historical, bool):
        raise IdentifierError("historical must be a boolean.")
    if not isinstance(categories, Mapping):
        raise IdentifierError("categories must be a mapping.")
    if not all(isinstance(key, str) for key in categories):
        raise IdentifierError("Semantic alias category keys must be strings.")

    expected = set(SEMANTIC_RUN_ALIAS_FIELDS)
    provided = set(categories)
    missing = sorted(expected - provided)
    unknown = sorted(provided - expected - {"variant_evidence"})
    if missing:
        raise IdentifierError(
            "Missing semantic alias categories: " + ", ".join(missing)
        )
    if unknown:
        raise IdentifierError(
            "Unknown semantic alias categories: " + ", ".join(unknown)
        )

    variant_evidence = categories.get("variant_evidence", {})
    if not isinstance(variant_evidence, Mapping):
        raise IdentifierError("variant_evidence must be a mapping when supplied.")

    display = {
        field: _semantic_alias_text(categories[field], field=field)
        for field in _ALIAS_TEXT_LIMITS
    }
    embedding_dim = _semantic_alias_integer(
        categories["embedding_dim"],
        field="embedding_dim",
        minimum=1,
        maximum=9_999_999,
    )
    seed = _semantic_alias_integer(
        categories["seed"], field="seed", minimum=0, maximum=999_999
    )

    fold_value = categories["fold"]
    if fold_value is None:
        fold_token = "fna"
    else:
        fold = _semantic_alias_integer(
            fold_value, field="fold", minimum=0, maximum=9_999
        )
        fold_token = f"f{fold:02d}"

    attempt_value = categories["attempt"]
    if attempt_value is None:
        attempt_token = "ana"
    else:
        attempt = _semantic_alias_integer(
            attempt_value, field="attempt", minimum=1, maximum=9_999
        )
        attempt_token = f"a{attempt:02d}"

    variant_payload = {
        field: categories[field]
        for field in (
            "model",
            "graph",
            "mask",
            "edge_feature_state",
            "embedding_dim",
        )
    }
    variant_payload["variant_evidence"] = variant_evidence
    variant_token = canonical_sha256(variant_payload)[:_ALIAS_DIGEST_CHARACTERS]

    execution_payload = {
        "primary_run_id": primary_run_id,
        "historical": historical,
        "categories": dict(categories),
    }
    execution_token = canonical_sha256(execution_payload)[
        :_ALIAS_DIGEST_CHARACTERS
    ]

    prefix = "hist" if historical else "run"
    alias = ".".join(
        (
            prefix,
            display["lifecycle_stage"],
            display["study_axis"],
            display["model"],
            display["graph"],
            display["mask"],
            display["edge_feature_state"],
            f"d{embedding_dim}",
            f"s{seed:03d}",
            fold_token,
            attempt_token,
            f"v{variant_token}",
            f"x{execution_token}",
        )
    )
    if len(alias) > _MAX_SEMANTIC_ALIAS_LENGTH:
        raise IdentifierError(
            f"Semantic alias exceeds {_MAX_SEMANTIC_ALIAS_LENGTH} characters."
        )
    if not _SAFE_ALIAS.fullmatch(alias):
        raise IdentifierError("Semantic alias contains an unsafe token.")
    return alias


def create_campaign_id(
    name: str,
    *,
    campaign_date: date | None = None,
) -> str:
    """Create a human-readable campaign ID without registering it."""

    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise IdentifierError("Campaign name must contain letters or digits.")
    day = campaign_date or datetime.now(timezone.utc).date()
    return f"cmp_{day:%Y%m%d}_{slug}"


__all__ = [
    "SCIENTIFIC_EXCLUDED_FIELDS",
    "SEMANTIC_RUN_ALIAS_FIELDS",
    "IdentifierError",
    "canonical_json",
    "canonical_sha256",
    "create_campaign_id",
    "create_run_id",
    "repro_id",
    "semantic_run_alias",
    "scientific_id",
    "scientific_payload",
]
