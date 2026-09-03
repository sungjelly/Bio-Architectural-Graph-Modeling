"""Canonical scratch-to-artifact lifecycle for BAGM runs.

Active output is written beneath ``scratch/active_runs/<run_id>``.  A successful,
failed, or pruned bundle is moved to ``artifacts/runs/YYYY/MM/<run_id>`` and its
completion marker is created *after* content verification.  Same-filesystem
finalization uses an atomic directory rename.  Cross-filesystem finalization
copies to a private staging directory, verifies every checksum, publishes by
rename, and only then removes the owned scratch copy.

The API never overwrites a file.  A finalized bundle is immutable through this
API and its checksum manifest makes later changes detectable.  Failed bundles
retain partial metrics, logs, configuration, and traceback with ``_FAILED``.
Table writes prefer Parquet when PyArrow is available and otherwise retain the
same logical API with a JSONL or CSV fallback.
"""

from __future__ import annotations

import csv
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import hmac
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import traceback
from typing import Any, Collection, Iterable, Mapping, Sequence

import yaml

from .fingerprints import sha256_file
from .identifiers import canonical_json
from .paths import ProjectPaths, current_paths


RUN_DIRECTORIES = (
    "metrics",
    "predictions",
    "checkpoints",
    "diagnostics",
    "interpretation",
    "provenance",
    "logs",
)
COMPLETION_MARKERS = ("_SUCCESS", "_FAILED", "_PRUNED")
SUCCESS_REQUIRED = (
    "manifest.yaml",
    "config.resolved.yaml",
    "summary.json",
    "metrics/events.jsonl",
    "metrics/final.json",
    "provenance/git.json",
    "provenance/uncommitted_changes.patch",
    "provenance/environment.txt",
    "provenance/hardware.json",
    "provenance/data_fingerprints.json",
    "provenance/split_fingerprint.json",
    "provenance/command.txt",
    "logs/stdout.log",
    "logs/stderr.log",
)
PREDICTION_REQUIRED_COLUMNS = frozenset(
    {"run_id", "sample_key", "dataset_id", "split", "y_true", "y_pred"}
)
PREDICTION_OPTIONAL_COLUMNS = frozenset(
    {
        "graph_id",
        "fold",
        "sample_loss",
        "confidence",
        "prediction_entropy",
        "correct",
        "node_count",
        "edge_count",
        "effective_neighbor_count",
        "effective_mask_rate",
    }
)
FORBIDDEN_IDENTIFIER_COLUMNS = frozenset(
    {
        "patient",
        "patient_id",
        "patientid",
        "subject",
        "subject_id",
        "subjectid",
        "donor",
        "donor_id",
        "donorid",
        "person_id",
        "medical_record_number",
        "mrn",
        "full_name",
        "first_name",
        "last_name",
        "date_of_birth",
        "dob",
        "email",
        "phone",
        "address",
        "cell_id",
        "cellid",
        "cell_barcode",
        "barcode",
        "sample_id",
        "specimen_id",
    }
)

_RUN_ID = re.compile(
    r"^r_(?P<year>\d{4})(?P<month>\d{2})\d{2}T\d{6}Z_"
    r"[a-z0-9]+_s\d+_f\d+_a\d+_[a-z0-9_-]+$"
)
_SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class RunArchiveError(RuntimeError):
    """Base class for run lifecycle failures."""


class RunImmutableError(RunArchiveError):
    """Raised when code attempts to change a finalized bundle."""


class RunValidationError(RunArchiveError):
    """Raised when a run or prediction table violates its contract."""


def _json_value(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _normalized_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _is_forbidden_identifier(name: str) -> bool:
    normalized = _normalized_column(name)
    if normalized in FORBIDDEN_IDENTIFIER_COLUMNS:
        return True
    return any(
        normalized.endswith(suffix)
        for suffix in (
            "_patient_id",
            "_subject_id",
            "_donor_id",
            "_medical_record_number",
            "_mrn",
        )
    )


def _finite_numeric(value: Any, field: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise RunValidationError(f"{field} contains NaN or infinity.")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _finite_numeric(item, field)
        return
    raise RunValidationError(f"{field} must contain only finite numeric values.")


def validate_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_run_id: str | None = None,
    expected_split: str | None = None,
) -> list[dict[str, Any]]:
    """Validate regression-first predictions with optional classification fields.

    ``y_true`` and ``y_pred`` may be finite scalars or vectors, supporting
    masked-expression regression directly.  Probability, correctness,
    confidence, and entropy columns are optional classification extensions.
    Direct identifier column names are rejected; use a stable, salted
    ``sample_key`` and non-identifying ``subgroup_*`` labels.
    """

    materialized = [dict(row) for row in rows]
    if not materialized:
        raise RunValidationError("Prediction table must contain at least one row.")
    columns = set(materialized[0])
    missing = PREDICTION_REQUIRED_COLUMNS - columns
    if missing:
        raise RunValidationError(
            "Prediction table is missing required columns: "
            + ", ".join(sorted(missing))
        )
    forbidden = sorted(name for name in columns if _is_forbidden_identifier(name))
    if forbidden:
        raise RunValidationError(
            "Prediction table contains direct identifier columns: "
            + ", ".join(forbidden)
        )
    for index, row in enumerate(materialized):
        if set(row) != columns:
            raise RunValidationError(
                f"Prediction row {index} has a different column schema."
            )
        sample_key = row["sample_key"]
        if (
            not isinstance(sample_key, str)
            or not sample_key.strip()
            or len(sample_key) > 160
        ):
            raise RunValidationError(
                f"Prediction row {index} has an invalid sample_key."
            )
        if expected_run_id is not None and row["run_id"] != expected_run_id:
            raise RunValidationError(
                f"Prediction row {index} does not match run {expected_run_id}."
            )
        if expected_split is not None and row["split"] != expected_split:
            raise RunValidationError(
                f"Prediction row {index} does not match split {expected_split}."
            )
        for field in ("y_true", "y_pred", "sample_loss"):
            if field in row and row[field] is not None:
                _finite_numeric(row[field], field)
        for field in ("confidence", "prediction_entropy", "effective_mask_rate"):
            if field in row and row[field] is not None:
                _finite_numeric(row[field], field)
                value = float(row[field])
                if value < 0 or (field != "prediction_entropy" and value > 1):
                    raise RunValidationError(
                        f"Prediction row {index} has invalid {field}."
                    )
        for field in ("node_count", "edge_count", "effective_neighbor_count", "fold"):
            if field in row and row[field] is not None:
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0
                ):
                    raise RunValidationError(
                        f"Prediction row {index} has invalid {field}."
                    )
        if "correct" in row and row["correct"] is not None and not isinstance(
            row["correct"], bool
        ):
            raise RunValidationError("Prediction column correct must be boolean.")
        for field, value in row.items():
            normalized = _normalized_column(field)
            if normalized == "probability" or normalized.startswith("probability_"):
                _finite_numeric(value, field)
                values = value if isinstance(value, (list, tuple)) else [value]
                if any(float(item) < 0 or float(item) > 1 for item in values):
                    raise RunValidationError(
                        f"Prediction row {index} has probability outside [0, 1]."
                    )
    return materialized


def deidentify_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    identifier_fields: Sequence[str],
    salt: str | bytes,
    namespace: str = "bagm",
) -> list[dict[str, Any]]:
    """Replace explicit identifiers with a salted, one-way stable sample key.

    The salt must be at least 16 bytes and must be supplied from protected local
    configuration; this function never stores it.  Input rows are copied rather
    than mutated.  Every named identifier field is removed.
    """

    secret = salt.encode("utf-8") if isinstance(salt, str) else bytes(salt)
    if len(secret) < 16:
        raise RunValidationError("Deidentification salt must be at least 16 bytes.")
    if not identifier_fields:
        raise RunValidationError("At least one identifier field is required.")
    output: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = dict(original)
        if "sample_key" in row:
            raise RunValidationError(
                f"Row {index} already has sample_key; refusing ambiguous replacement."
            )
        missing = [field for field in identifier_fields if field not in row]
        if missing:
            raise RunValidationError(
                f"Row {index} is missing identifier fields: {', '.join(missing)}"
            )
        identity = {
            "namespace": namespace,
            "values": [row[field] for field in identifier_fields],
        }
        digest = hmac.new(
            secret,
            canonical_json(identity).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        for field in identifier_fields:
            row.pop(field)
        row["sample_key"] = f"sk_{digest[:32]}"
        output.append(row)
    return output


def _safe_relative(relative: str | Path) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise RunArchiveError(f"Unsafe run-relative path: {relative}")
    return candidate


def _bundle_checksums(root: Path) -> dict[str, dict[str, Any]]:
    checksums: dict[str, dict[str, Any]] = {}
    excluded = {
        "provenance/artifact_checksums.json",
        *COMPLETION_MARKERS,
    }
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or path.is_dir():
            continue
        if path.is_symlink():
            target = os.readlink(path)
            checksums[relative] = {
                "type": "symlink",
                "target_sha256": hashlib.sha256(
                    target.encode("utf-8", errors="surrogateescape")
                ).hexdigest(),
                "target_is_absolute": Path(target).is_absolute(),
            }
        elif path.is_file():
            checksums[relative] = {
                "type": "file",
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            checksums[relative] = {"type": "special"}
    return checksums


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise RunArchiveError(f"Refusing to overwrite run file: {path}") from error


def _canonical_prediction_split(root: Path) -> str:
    """Return the explicitly configured canonical prediction data role.

    Historical and ordinary predictive runs default to ``validation``.
    Transductive capacity studies must opt into ``fit`` in their resolved
    evaluation configuration so held-in predictions cannot be mistaken for
    validation evidence.
    """

    config_path = root / "config.resolved.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise RunValidationError(
            "config.resolved.yaml cannot be parsed for prediction semantics."
        ) from error
    if not isinstance(config, Mapping):
        raise RunValidationError("config.resolved.yaml must contain a mapping.")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise RunValidationError("evaluation configuration must be a mapping.")
    split = str(
        evaluation.get("canonical_prediction_split", "validation")
    ).strip().lower()
    artifact_contract = str(evaluation.get("artifact_contract", "predictive"))
    protocol = str(evaluation.get("protocol", "")).strip().lower()
    if artifact_contract == "analysis_only":
        if (
            protocol != "posthoc_attention_routing_niche_v1"
            or split != "analysis"
        ):
            raise RunValidationError(
                "analysis_only bundles require the registered post-hoc "
                "attention-routing protocol and canonical role 'analysis'."
            )
        return "analysis"
    if artifact_contract != "predictive":
        raise RunValidationError(
            "evaluation.artifact_contract must be predictive or analysis_only."
        )
    if split not in {"validation", "fit"}:
        raise RunValidationError(
            "evaluation.canonical_prediction_split must be validation or fit."
        )
    if split == "fit":
        held_in_protocols = {
            "held_in_full_core_fixed_budget",
            "held_in_pooled_10core_fixed_budget",
            "held_in_pooled_14core_relative_qkv_fixed_continuation_epoch300",
            "held_in_pooled_14core_geometry_modulated_relative_qkv_seed_plateau",
            "held_in_pooled_14core_recurrent_relative_qkv_seed_plateau",
            "held_in_pooled_14core_relative_qkv_seed_plateau",
            "held_in_pooled_14core_untied8_relative_qkv_seed_plateau",
            "held_in_pooled_so1_14core_relative_qkv_plateau_min150",
            "held_in_pooled_6core_relative_qkv_fixed_budget",
            "held_in_pooled_6core_relative_qkv_joint_plateau",
            "held_in_pooled_6core_relative_qkv_seed_plateau",
        }
        if protocol not in held_in_protocols:
            raise RunValidationError(
                "canonical fit predictions require an explicitly supported "
                "held-in full-core or pooled-core evaluation protocol; "
                f"unsupported evaluation.protocol={protocol!r}."
            )
    return split


def _canonical_checkpoint_path(root: Path) -> Path:
    """Resolve the declared primary checkpoint without inventing selection."""

    config_path = root / "config.resolved.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise RunValidationError(
            "config.resolved.yaml cannot be parsed for checkpoint semantics."
        ) from error
    if not isinstance(config, Mapping):
        raise RunValidationError("config.resolved.yaml must contain a mapping.")
    trainer = config.get("trainer", {})
    if not isinstance(trainer, Mapping):
        raise RunValidationError("trainer configuration must be a mapping.")
    role = str(trainer.get("primary_checkpoint_role", "best")).strip().lower()
    if role not in {"best", "last"}:
        raise RunValidationError(
            "trainer.primary_checkpoint_role must be best or last."
        )
    if role == "last":
        restore_best = trainer.get("restore_best")
        if restore_best not in {False, None}:
            raise RunValidationError(
                "a last-checkpoint protocol cannot restore a selected best state."
            )
    return root / "checkpoints" / f"{role}.ckpt"


def _validate_success_contract_at(
    root: Path,
    run_id: str,
    *,
    tombstoned_paths: Collection[str] = (),
) -> None:
    missing = [relative for relative in SUCCESS_REQUIRED if not (root / relative).is_file()]
    if missing:
        raise RunValidationError(
            "Successful run is missing required files: " + ", ".join(missing)
        )
    prediction_split = _canonical_prediction_split(root)
    analysis_only = prediction_split == "analysis"
    if not analysis_only:
        checkpoint = _canonical_checkpoint_path(root)
        if not checkpoint.is_file():
            checkpoint_relative = checkpoint.relative_to(root).as_posix()
            if checkpoint_relative not in tombstoned_paths:
                raise RunValidationError(
                    "Successful run is missing its declared primary checkpoint: "
                    f"{checkpoint_relative}"
                )
        elif checkpoint.stat().st_size == 0:
            raise RunValidationError(
                "Successful run declared primary checkpoint is empty."
            )
    history_paths = [
        root / f"metrics/history{suffix}"
        for suffix in (".parquet", ".jsonl", ".csv")
        if (root / f"metrics/history{suffix}").is_file()
    ]
    if not history_paths or all(path.stat().st_size == 0 for path in history_paths):
        raise RunValidationError(
            "Successful run is missing non-empty metrics/history in a supported format."
        )
    if not analysis_only:
        prediction_paths = [
            root / f"predictions/{prediction_split}{suffix}"
            for suffix in (".parquet", ".jsonl", ".csv")
            if (root / f"predictions/{prediction_split}{suffix}").is_file()
        ]
        tombstoned_prediction = any(
            f"predictions/{prediction_split}{suffix}" in tombstoned_paths
            for suffix in (".parquet", ".jsonl", ".csv")
        )
        if (
            not tombstoned_prediction
            and (
                not prediction_paths
                or all(path.stat().st_size == 0 for path in prediction_paths)
            )
        ):
            raise RunValidationError(
                "Successful run requires non-empty canonical "
                f"{prediction_split} predictions."
            )
    resolved_config: Mapping[str, Any] | None = None
    for relative in ("manifest.yaml", "config.resolved.yaml"):
        value = yaml.safe_load((root / relative).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise RunValidationError(f"{relative} must contain a mapping.")
        if relative == "manifest.yaml":
            if value.get("run_id") != run_id:
                raise RunValidationError("Manifest run_id does not match run directory.")
            if value.get("status") == "running":
                raise RunValidationError(
                    "A finalized manifest may not retain status='running'."
                )
        else:
            resolved_config = value
    assert resolved_config is not None
    if analysis_only:
        evaluation = resolved_config.get("evaluation", {})
        metadata = resolved_config.get("metadata", {})
        if (
            not isinstance(evaluation, Mapping)
            or evaluation.get("artifact_contract") != "analysis_only"
            or not isinstance(metadata, Mapping)
        ):
            raise RunValidationError(
                "Analysis-only bundle configuration is malformed."
            )
        required_analysis = metadata.get("required_analysis_outputs")
        if (
            not isinstance(required_analysis, Sequence)
            or isinstance(required_analysis, (str, bytes))
            or not required_analysis
        ):
            raise RunValidationError(
                "Analysis-only bundle must declare required_analysis_outputs."
            )
        seen_analysis: set[str] = set()
        for raw_relative in required_analysis:
            relative = _safe_relative(str(raw_relative)).as_posix()
            if relative in seen_analysis:
                raise RunValidationError(
                    "required_analysis_outputs contains duplicates."
                )
            seen_analysis.add(relative)
            output = root / relative
            if not output.is_file() or output.stat().st_size == 0:
                raise RunValidationError(
                    f"Analysis-only bundle is missing required output: {relative}"
                )
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping) or summary.get("run_id") != run_id:
        raise RunValidationError(
            "summary.json must be a mapping bound to the run_id."
        )
    final_metrics = json.loads(
        (root / "metrics/final.json").read_text(encoding="utf-8")
    )
    if not isinstance(final_metrics, Mapping) or not final_metrics:
        raise RunValidationError("metrics/final.json must contain metrics.")
    if prediction_split == "fit":
        forbidden_metric_prefixes = ("val/", "test/", "external/")
        forbidden_metrics = sorted(
            str(name)
            for name in final_metrics
            if str(name).startswith(forbidden_metric_prefixes)
        )
        if forbidden_metrics:
            raise RunValidationError(
                "Held-in fit runs may not contain held-out final metrics: "
                + ", ".join(forbidden_metrics)
            )
        forbidden_prediction_files = sorted(
            path.relative_to(root).as_posix()
            for split in ("validation", "test", "external")
            for path in (root / "predictions").glob(f"{split}.*")
        )
        if forbidden_prediction_files:
            raise RunValidationError(
                "Held-in fit runs may not contain held-out prediction artifacts: "
                + ", ".join(forbidden_prediction_files)
            )
        misleading_best = root / "checkpoints/best.ckpt"
        if misleading_best.exists() or misleading_best.is_symlink():
            raise RunValidationError(
                "Held-in fixed-budget runs may not contain best.ckpt because "
                "no validation checkpoint selection occurred."
            )
    evaluation = resolved_config.get("evaluation", {})
    configured_primary = (
        evaluation.get("primary_metric")
        if isinstance(evaluation, Mapping)
        else None
    )
    numeric_metrics = 0
    for name, value in final_metrics.items():
        if not isinstance(name, str) or "/" not in name:
            raise RunValidationError(
                "Final metric names must be explicit and namespaced."
            )
        if value is None:
            if name == configured_primary:
                raise RunValidationError(
                    "The configured primary final metric must be finite numeric."
                )
            # Some prespecified metrics are mathematically undefined for a
            # replicate/core (for example, precision with no predicted
            # positives).  Preserve that distinction as an explicit null;
            # append-only metric events remain finite-numeric only.
            continue
        _finite_numeric(value, f"final metric {name}")
        numeric_metrics += 1
    if numeric_metrics == 0:
        raise RunValidationError("metrics/final.json has no numeric metrics.")
    declared_primary = summary.get("primary_metric_name")
    if (
        declared_primary is not None
        and configured_primary is not None
        and declared_primary != configured_primary
    ):
        raise RunValidationError(
            "summary primary_metric_name does not match the resolved "
            "evaluation.primary_metric."
        )
    if prediction_split in {"fit", "analysis"} and configured_primary is not None:
        if declared_primary != configured_primary:
            raise RunValidationError(
                "Summary must declare the configured primary metric."
            )
        if configured_primary not in final_metrics:
            raise RunValidationError(
                "Final metrics omit the configured primary metric."
            )
        declared_value = summary.get("primary_metric_value")
        if isinstance(declared_value, bool) or not isinstance(
            declared_value, (int, float)
        ):
            raise RunValidationError(
                "Summary must declare a numeric primary_metric_value."
            )
        _finite_numeric(declared_value, "summary primary metric")
        if not math.isclose(
            float(declared_value),
            float(final_metrics[configured_primary]),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise RunValidationError(
                "Summary primary metric value does not match "
                "metrics/final.json."
            )
    event_lines = [
        line
        for line in (root / "metrics/events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if not event_lines:
        raise RunValidationError("metrics/events.jsonl must not be empty.")
    for line in event_lines:
        event = json.loads(line)
        if (
            not isinstance(event, Mapping)
            or not isinstance(event.get("name"), str)
            or "/" not in str(event["name"])
            or "value" not in event
        ):
            raise RunValidationError("Metric event is malformed or ambiguous.")
        _finite_numeric(event["value"], f"metric event {event['name']}")
    jsonl_prediction = root / f"predictions/{prediction_split}.jsonl"
    if jsonl_prediction.is_file():
        expected_columns: set[str] | None = None
        row_count = 0
        with jsonl_prediction.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                validated = validate_prediction_rows(
                    [row],
                    expected_run_id=run_id,
                    expected_split=prediction_split,
                )[0]
                columns = set(validated)
                if expected_columns is None:
                    expected_columns = columns
                elif columns != expected_columns:
                    raise RunValidationError(
                        f"Canonical {prediction_split} prediction rows change schema."
                    )
                row_count += 1
        if row_count == 0:
            raise RunValidationError(
                f"Canonical {prediction_split} predictions are empty."
            )


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing even an empty destination."""

    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RunArchiveError(
            "This platform lacks renameat2; refusing a non-atomic artifact publish."
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise RunArchiveError(
            f"Refusing to overwrite artifact run: {destination}"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination.as_posix(),
    )


class RunArchive:
    """One mutable scratch bundle that may be finalized exactly once."""

    def __init__(
        self,
        run_id: str,
        *,
        paths: ProjectPaths,
        scratch_path: Path,
        artifact_path: Path,
    ) -> None:
        self.run_id = run_id
        self.paths = paths
        self.scratch_path = scratch_path
        self.artifact_path = artifact_path

    @staticmethod
    def artifact_path_for(run_id: str, paths: ProjectPaths) -> Path:
        match = _RUN_ID.fullmatch(run_id)
        if match is None:
            raise RunArchiveError(f"Invalid canonical run_id: {run_id}")
        return (
            paths.artifact_root
            / "runs"
            / match.group("year")
            / match.group("month")
            / run_id
        )

    @classmethod
    def create(
        cls,
        run_id: str,
        *,
        paths: ProjectPaths | None = None,
        manifest: Mapping[str, Any] | None = None,
        resolved_config: Mapping[str, Any] | None = None,
    ) -> "RunArchive":
        """Create only the run root; contract subdirectories are created lazily."""

        selected_paths = paths or current_paths()
        scratch_path = selected_paths.scratch_root / "active_runs" / run_id
        artifact_path = cls.artifact_path_for(run_id, selected_paths)
        if scratch_path.exists() or scratch_path.is_symlink():
            raise RunArchiveError(f"Scratch run already exists: {scratch_path}")
        if artifact_path.exists() or artifact_path.is_symlink():
            raise RunArchiveError(f"Artifact run already exists: {artifact_path}")
        scratch_path.parent.mkdir(parents=True, exist_ok=True)
        scratch_path.mkdir()
        archive = cls(
            run_id,
            paths=selected_paths,
            scratch_path=scratch_path,
            artifact_path=artifact_path,
        )
        archive.write_json(
            ".bagm-run-owner.json",
            {"run_id": run_id, "created_at": _utc_now(), "format_version": 1},
        )
        if manifest is not None:
            archive.write_manifest(manifest)
        if resolved_config is not None:
            archive.write_resolved_config(resolved_config)
        return archive

    @classmethod
    def attach_active(
        cls,
        run_id: str,
        *,
        paths: ProjectPaths | None = None,
        scratch_path: str | Path | None = None,
    ) -> "RunArchive":
        """Attach to the exact worker-owned active bundle for ``run_id``.

        Scientific subprocesses use this entry point to add outputs to a
        scratch bundle that the queue worker already created.  The canonical
        location and ownership marker are both checked so a caller cannot
        redirect writes into another run or an arbitrary directory.
        """

        selected_paths = paths or current_paths()
        expected = selected_paths.scratch_root / "active_runs" / run_id
        if _RUN_ID.fullmatch(run_id) is None:
            raise RunArchiveError(f"Invalid canonical run_id: {run_id}")
        if scratch_path is not None:
            supplied = Path(scratch_path)
            if supplied.resolve(strict=False) != expected.resolve(strict=False):
                raise RunArchiveError(
                    "Active scratch path does not match the canonical run path."
                )
        if expected.is_symlink() or not expected.is_dir():
            raise RunArchiveError(
                f"Active scratch run does not exist as a directory: {expected}"
            )
        owner_path = expected / ".bagm-run-owner.json"
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RunArchiveError(
                "Active scratch run has no readable ownership marker."
            ) from error
        if (
            not isinstance(owner, Mapping)
            or owner.get("run_id") != run_id
            or owner.get("format_version") != 1
        ):
            raise RunArchiveError(
                "Active scratch ownership marker does not match the run."
            )
        if any((expected / marker).exists() for marker in COMPLETION_MARKERS):
            raise RunImmutableError(f"Run {run_id} is already finalized.")
        artifact_path = cls.artifact_path_for(run_id, selected_paths)
        if artifact_path.exists() or artifact_path.is_symlink():
            raise RunArchiveError(
                "Published artifact already exists for an active scratch run."
            )
        return cls(
            run_id,
            paths=selected_paths,
            scratch_path=expected,
            artifact_path=artifact_path,
        )

    @classmethod
    def from_published(
        cls, run_id: str, *, paths: ProjectPaths | None = None
    ) -> "RunArchive":
        """Reattach to an owned published bundle for crash reconciliation."""

        selected_paths = paths or current_paths()
        artifact_path = cls.artifact_path_for(run_id, selected_paths)
        if not artifact_path.is_dir():
            raise RunArchiveError(
                f"Published artifact run does not exist: {artifact_path}"
            )
        return cls(
            run_id,
            paths=selected_paths,
            scratch_path=selected_paths.scratch_root / "active_runs" / run_id,
            artifact_path=artifact_path,
        )

    def _assert_mutable(self) -> None:
        if not self.scratch_path.is_dir():
            raise RunImmutableError(
                f"Run {self.run_id} is not an active scratch bundle."
            )
        if any((self.scratch_path / marker).exists() for marker in COMPLETION_MARKERS):
            raise RunImmutableError(f"Run {self.run_id} is already finalized.")

    def _target(self, relative: str | Path) -> Path:
        self._assert_mutable()
        safe = _safe_relative(relative)
        target = self.scratch_path / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = target.parent.resolve(strict=False)
        resolved_root = self.scratch_path.resolve(strict=True)
        if not resolved_parent.is_relative_to(resolved_root):
            raise RunArchiveError(f"Run path escapes scratch bundle: {relative}")
        return target

    def write_bytes(self, relative: str | Path, value: bytes) -> Path:
        target = self._target(relative)
        _write_exclusive(target, value)
        return target

    def write_text(
        self,
        relative: str | Path,
        value: str,
        *,
        encoding: str = "utf-8",
    ) -> Path:
        return self.write_bytes(relative, value.encode(encoding))

    def write_json(self, relative: str | Path, value: Any) -> Path:
        content = json.dumps(
            _json_value(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        return self.write_text(relative, content + "\n")

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        value = dict(manifest)
        if "run_id" in value and value["run_id"] != self.run_id:
            raise RunValidationError("Manifest run_id does not match archive run_id.")
        value.setdefault("run_id", self.run_id)
        content = yaml.safe_dump(
            _json_value(value),
            sort_keys=True,
            allow_unicode=False,
        )
        return self.write_text("manifest.yaml", content)

    def write_resolved_config(self, config: Mapping[str, Any]) -> Path:
        content = yaml.safe_dump(
            _json_value(config),
            sort_keys=True,
            allow_unicode=False,
        )
        return self.write_text("config.resolved.yaml", content)

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        value = dict(summary)
        if "run_id" in value and value["run_id"] != self.run_id:
            raise RunValidationError("Summary run_id does not match archive run_id.")
        value.setdefault("run_id", self.run_id)
        return self.write_json("summary.json", value)

    def prepare_log_files(self) -> tuple[Path, Path]:
        stdout = self.write_text("logs/stdout.log", "")
        stderr = self.write_text("logs/stderr.log", "")
        return stdout, stderr

    def append_metric_event(self, event: Mapping[str, Any]) -> Path:
        """Append one namespaced metric event as a durable JSONL record."""

        self._assert_mutable()
        value = dict(event)
        name = value.get("name")
        if not isinstance(name, str) or "/" not in name:
            raise RunValidationError(
                "Metric event name must be explicit and namespaced, e.g. val/loss."
            )
        if "value" not in value:
            raise RunValidationError("Metric event requires value.")
        _finite_numeric(value["value"], "metric value")
        value.setdefault("timestamp", _utc_now())
        target = self._target("metrics/events.jsonl")
        with target.open("ab") as handle:
            handle.write(canonical_json(value).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return target

    def copy_file(self, source: str | Path, relative: str | Path) -> Path:
        """Copy one regular source without changing or deleting it."""

        source_path = Path(source)
        if not source_path.is_file() or source_path.is_symlink():
            raise RunArchiveError(f"Run copy source must be a regular file: {source}")
        target = self._target(relative)
        if target.exists() or target.is_symlink():
            raise RunArchiveError(f"Refusing to overwrite run file: {target}")
        shutil.copy2(source_path, target)
        if sha256_file(source_path) != sha256_file(target):
            raise RunArchiveError(f"Run file copy checksum mismatch: {source}")
        return target

    def write_table(
        self,
        relative_stem: str | Path,
        rows: Iterable[Mapping[str, Any]],
        *,
        fallback: str = "jsonl",
    ) -> Path:
        """Write one logical table as Parquet, or JSONL/CSV when unavailable."""

        materialized = [dict(row) for row in rows]
        if not materialized:
            raise RunValidationError("Cannot write an empty table.")
        if fallback not in {"jsonl", "csv"}:
            raise RunArchiveError("Table fallback must be 'jsonl' or 'csv'.")
        stem = _safe_relative(relative_stem)
        if stem.suffix in {".parquet", ".jsonl", ".csv"}:
            stem = stem.with_suffix("")

        if importlib.util.find_spec("pyarrow") is not None:
            target = self._target(stem.with_suffix(".parquet"))
            if target.exists():
                raise RunArchiveError(f"Refusing to overwrite run file: {target}")
            temporary = target.with_name(f".{target.name}.writing")
            if temporary.exists():
                raise RunArchiveError(f"Temporary table path already exists: {temporary}")
            try:
                import pyarrow as pa
                import pyarrow.parquet as parquet

                parquet.write_table(pa.Table.from_pylist(materialized), temporary)
                if target.exists():
                    raise RunArchiveError(f"Refusing to overwrite run file: {target}")
                temporary.rename(target)
                return target
            except (ImportError, ModuleNotFoundError):
                if temporary.exists():
                    temporary.unlink()
            except Exception as error:
                if temporary.exists():
                    temporary.unlink()
                raise RunArchiveError(
                    f"Parquet table serialization failed for {relative_stem}: {error}"
                ) from error

        target = self._target(stem.with_suffix(f".{fallback}"))
        if fallback == "jsonl":
            lines = [canonical_json(row) for row in materialized]
            _write_exclusive(target, ("\n".join(lines) + "\n").encode("utf-8"))
            return target

        columns = list(materialized[0])
        if any(set(row) != set(columns) for row in materialized):
            raise RunValidationError("CSV table rows must share one column schema.")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for row in materialized:
                    writer.writerow(
                        {
                            key: (
                                canonical_json(value)
                                if isinstance(value, (dict, list, tuple))
                                else value
                            )
                            for key, value in row.items()
                        }
                    )
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise RunArchiveError(f"Refusing to overwrite run file: {target}") from error
        return target

    def write_predictions(
        self,
        split: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        fallback: str = "jsonl",
    ) -> Path:
        normalized_split = split.strip().lower()
        if not _SAFE_COMPONENT.fullmatch(normalized_split):
            raise RunValidationError(f"Unsafe prediction split name: {split!r}")
        validated = validate_prediction_rows(
            rows,
            expected_run_id=self.run_id,
            expected_split=normalized_split,
        )
        return self.write_table(
            Path("predictions") / normalized_split,
            validated,
            fallback=fallback,
        )

    def write_prediction_jsonl_stream(
        self,
        split: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> Path:
        """Validate and stream a large canonical prediction table as JSONL."""

        normalized_split = split.strip().lower()
        if not _SAFE_COMPONENT.fullmatch(normalized_split):
            raise RunValidationError(f"Unsafe prediction split name: {split!r}")
        target = self._target(
            Path("predictions") / f"{normalized_split}.jsonl"
        )
        expected_columns: set[str] | None = None
        count = 0
        try:
            with target.open("xb") as handle:
                for raw_row in rows:
                    row = validate_prediction_rows(
                        [raw_row],
                        expected_run_id=self.run_id,
                        expected_split=normalized_split,
                    )[0]
                    columns = set(row)
                    if expected_columns is None:
                        expected_columns = columns
                    elif columns != expected_columns:
                        raise RunValidationError(
                            "Prediction stream changes column schema."
                        )
                    handle.write(canonical_json(row).encode("utf-8") + b"\n")
                    count += 1
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise RunArchiveError(
                f"Refusing to overwrite run file: {target}"
            ) from error
        if count == 0:
            raise RunValidationError("Prediction stream must contain at least one row.")
        return target

    def _write_checksums(self) -> dict[str, dict[str, Any]]:
        checksums = _bundle_checksums(self.scratch_path)
        existing_path = self.scratch_path / "provenance/artifact_checksums.json"
        if existing_path.is_file():
            existing_payload = json.loads(existing_path.read_text(encoding="utf-8"))
            existing = (
                existing_payload.get("files")
                if isinstance(existing_payload, Mapping)
                else None
            )
            if existing != checksums:
                raise RunArchiveError(
                    "Existing artifact checksum manifest no longer matches scratch."
                )
            return checksums
        self.write_json(
            "provenance/artifact_checksums.json",
            {"version": 1, "files": checksums},
        )
        return checksums

    def _validate_success_ready(self) -> None:
        _validate_success_contract_at(self.scratch_path, self.run_id)

    def _publish_unmarked(self) -> Path:
        self._assert_mutable()
        if self.artifact_path.exists() or self.artifact_path.is_symlink():
            raise RunArchiveError(
                f"Refusing to overwrite artifact run: {self.artifact_path}"
            )
        checksums = self._write_checksums()
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)

        same_filesystem = (
            self.scratch_path.stat().st_dev == self.artifact_path.parent.stat().st_dev
        )
        if same_filesystem:
            _rename_no_replace(self.scratch_path, self.artifact_path)
        else:
            staging = self.artifact_path.parent / (
                f".{self.run_id}.finalizing-{os.getpid()}"
            )
            if staging.exists() or staging.is_symlink():
                raise RunArchiveError(f"Finalization staging path exists: {staging}")
            shutil.copytree(
                self.scratch_path,
                staging,
                symlinks=True,
                copy_function=shutil.copy2,
            )
            actual = _bundle_checksums(staging)
            if actual != checksums:
                raise RunArchiveError(
                    "Cross-filesystem run copy failed checksum verification."
                )
            if self.artifact_path.exists() or self.artifact_path.is_symlink():
                raise RunArchiveError(
                    f"Refusing to overwrite artifact run: {self.artifact_path}"
                )
            _rename_no_replace(staging, self.artifact_path)
            owner = self.scratch_path / ".bagm-run-owner.json"
            if not owner.is_file():
                raise RunArchiveError(
                    "Owned scratch marker missing; refusing cross-filesystem cleanup."
                )
            shutil.rmtree(self.scratch_path)

        final_checksums = _bundle_checksums(self.artifact_path)
        if final_checksums != checksums:
            raise RunArchiveError("Final artifact differs from verified scratch bundle.")
        return self.artifact_path

    def _mark_published(self, marker: str) -> Path:
        if marker not in COMPLETION_MARKERS:
            raise RunArchiveError(f"Unknown completion marker: {marker}")
        if not self.artifact_path.is_dir():
            raise RunArchiveError(
                f"Published artifact run does not exist: {self.artifact_path}"
            )
        existing = [
            name for name in COMPLETION_MARKERS
            if (self.artifact_path / name).exists()
        ]
        if existing:
            raise RunArchiveError(
                f"Published run already has a completion marker: {existing}"
            )
        checksum_path = self.artifact_path / "provenance/artifact_checksums.json"
        try:
            payload = json.loads(checksum_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RunArchiveError(
                "Published run has no readable checksum manifest."
            ) from error
        checksums = payload.get("files") if isinstance(payload, Mapping) else None
        if not isinstance(checksums, Mapping) or _bundle_checksums(
            self.artifact_path
        ) != checksums:
            raise RunArchiveError(
                "Published run changed before its completion marker was written."
            )
        expected_digest = hashlib.sha256(
            canonical_json(checksums).encode("utf-8")
        ).hexdigest()
        _write_exclusive(
            self.artifact_path / marker,
            (
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "status": marker.removeprefix("_").lower(),
                        "finalized_at": _utc_now(),
                        "content_sha256": expected_digest,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        return self.artifact_path

    def publish_success_pending(self) -> Path:
        """Publish verified content without a marker for registry commit."""

        self._assert_mutable()
        self._validate_success_ready()
        return self._publish_unmarked()

    def mark_success(self) -> Path:
        """Write ``_SUCCESS`` after the registry transaction has committed."""

        return self._mark_published("_SUCCESS")

    def mark_published_failure(self) -> Path:
        """Mark an unmarked published bundle failed after commit/finalization error."""

        return self._mark_published("_FAILED")

    def finalize_success(self) -> Path:
        """Validate required outputs, publish atomically, then write ``_SUCCESS``."""

        self._assert_mutable()
        self._validate_success_ready()
        self._publish_unmarked()
        return self._mark_published("_SUCCESS")

    def finalize_failure(
        self,
        error: BaseException | str,
        *,
        failure_category: str = "nonzero_exit",
        traceback_text: str | None = None,
    ) -> Path:
        """Preserve a partial bundle and traceback, then write ``_FAILED``."""

        self._assert_mutable()
        message = str(error)
        if traceback_text is None and isinstance(error, BaseException):
            traceback_text = "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
        if traceback_text is None:
            traceback_text = message
        summary_path = self.scratch_path / "summary.json"
        if summary_path.is_file():
            try:
                existing_summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                existing_summary = None
            if (
                isinstance(existing_summary, Mapping)
                and existing_summary.get("status")
                in {"success", "completed", "running"}
            ):
                preserved = self.scratch_path / "diagnostics/pre_failure_summary.json"
                preserved.parent.mkdir(parents=True, exist_ok=True)
                _rename_no_replace(summary_path, preserved)
                self.write_summary(
                    {
                        "status": "failed",
                        "failure_category": failure_category,
                        "failure_message": message,
                        "pre_failure_summary": (
                            "diagnostics/pre_failure_summary.json"
                        ),
                    }
                )
        if not (self.scratch_path / "logs/exception.txt").exists():
            self.write_text("logs/exception.txt", traceback_text.rstrip() + "\n")
        if not (self.scratch_path / "manifest.yaml").exists():
            self.write_manifest(
                {
                    "run_id": self.run_id,
                    "status": "failed",
                    "failure_category": failure_category,
                }
            )
        if not (self.scratch_path / "config.resolved.yaml").exists():
            self.write_resolved_config({})
        if not (self.scratch_path / "summary.json").exists():
            self.write_summary(
                {
                    "status": "failed",
                    "failure_category": failure_category,
                    "failure_message": message,
                }
            )
        self._publish_unmarked()
        return self._mark_published("_FAILED")

    def finalize_pruned(self, *, reason: str) -> Path:
        """Preserve a deliberately stopped run and write ``_PRUNED``."""

        self._assert_mutable()
        if not reason.strip():
            raise RunValidationError("Pruned run requires a reason.")
        if not (self.scratch_path / "manifest.yaml").exists():
            self.write_manifest({"status": "pruned", "reason": reason})
        if not (self.scratch_path / "config.resolved.yaml").exists():
            self.write_resolved_config({})
        if not (self.scratch_path / "summary.json").exists():
            self.write_summary({"status": "pruned", "reason": reason})
        self._publish_unmarked()
        return self._mark_published("_PRUNED")


def _verified_retention_tombstones(
    root: Path,
    expected: Mapping[str, Any],
    tombstoned_artifacts: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Validate registry-supplied tombstones against the immutable manifest."""

    verified: dict[str, dict[str, Any]] = {}
    failed_or_pruned = (root / "_FAILED").is_file() or (root / "_PRUNED").is_file()
    protected_failed_evidence = {
        ".bagm-run-owner.json",
        "config.resolved.yaml",
        "manifest.yaml",
        "summary.json",
        "_FAILED",
        "_PRUNED",
        "_SUCCESS",
    }
    allowed_failed_derived_payloads = {
        "directed_attention_edges.parquet",
        "mutual_attention_edges.parquet",
    }
    for raw_relative, raw_metadata in (tombstoned_artifacts or {}).items():
        relative = _safe_relative(raw_relative).as_posix()
        if relative in verified:
            raise RunValidationError(
                f"Duplicate normalized retention tombstone: {relative}"
            )
        parts = Path(relative).parts
        top_level = parts[0]
        ordinary_payload = top_level in {"checkpoints", "predictions"}
        if not ordinary_payload:
            if not failed_or_pruned:
                raise RunValidationError(
                    "Successful-run retention tombstones may cover only checkpoint "
                    f"or prediction payloads, not {relative}."
                )
            if (
                relative in protected_failed_evidence
                or top_level in {"logs", "metrics", "provenance"}
                or Path(relative).name not in allowed_failed_derived_payloads
            ):
                raise RunValidationError(
                    "Failed-run retention tombstones may cover only explicitly "
                    "approved oversized derived edge tables, not compact audit "
                    f"evidence or other outputs: {relative}."
                )
        target = root / relative
        if target.exists() or target.is_symlink():
            raise RunValidationError(
                f"Retention tombstone path still exists: {relative}"
            )
        metadata = dict(raw_metadata)
        if relative not in expected or expected[relative] != metadata:
            raise RunValidationError(
                "Retention tombstone does not match the immutable checksum "
                f"manifest: {relative}"
            )
        verified[relative] = metadata
    return verified


def verify_run_bundle(
    run_path: str | Path,
    *,
    require_success_contract: bool = True,
    tombstoned_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify marker exclusivity, required files, symlinks, and all checksums.

    ``tombstoned_artifacts`` is accepted only from an audited external
    registry. Successful bundles permit checkpoint or prediction tombstones.
    Failed and pruned bundles may additionally tombstone explicitly approved
    oversized derived attention-edge tables, but never compact outputs,
    configuration, log, metric, provenance, summary, or completion-marker
    evidence. Every absence must match the original immutable bundle manifest
    exactly; changed or unrecorded missing content is never accepted.
    """

    root = Path(run_path)
    if not root.is_dir():
        raise RunValidationError(f"Run bundle is not a directory: {root}")
    markers = [marker for marker in COMPLETION_MARKERS if (root / marker).is_file()]
    if len(markers) != 1:
        raise RunValidationError(
            f"Run bundle requires exactly one completion marker; found {markers}."
        )
    broken_links = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink() and not path.exists()
    ]
    if broken_links:
        raise RunValidationError(
            "Run bundle contains broken symlinks: " + ", ".join(broken_links)
        )
    checksum_path = root / "provenance/artifact_checksums.json"
    if not checksum_path.is_file():
        raise RunValidationError("Run bundle has no artifact checksum manifest.")
    payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    expected = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(expected, Mapping):
        raise RunValidationError("Artifact checksum manifest is malformed.")
    actual = _bundle_checksums(root)
    verified_tombstones = _verified_retention_tombstones(
        root,
        expected,
        tombstoned_artifacts,
    )
    effective_actual = {**actual, **verified_tombstones}
    if effective_actual != expected:
        missing = sorted(set(expected) - set(effective_actual))
        extra = sorted(set(effective_actual) - set(expected))
        changed = sorted(
            key
            for key in set(expected).intersection(effective_actual)
            if expected[key] != effective_actual[key]
        )
        raise RunValidationError(
            f"Artifact checksum mismatch; missing={missing}, extra={extra}, "
            f"changed={changed}."
        )
    marker_path = root / markers[0]
    try:
        marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RunValidationError("Completion marker is not readable JSON.") from error
    if not isinstance(marker_payload, Mapping):
        raise RunValidationError("Completion marker must contain a JSON mapping.")
    expected_status = markers[0].removeprefix("_").lower()
    expected_digest = hashlib.sha256(
        canonical_json(expected).encode("utf-8")
    ).hexdigest()
    if marker_payload.get("status") != expected_status:
        raise RunValidationError(
            "Completion marker status does not match its filename."
        )
    if marker_payload.get("run_id") != root.name:
        raise RunValidationError(
            "Completion marker run_id does not match the run directory."
        )
    if not hmac.compare_digest(
        str(marker_payload.get("content_sha256", "")), expected_digest
    ):
        raise RunValidationError(
            "Completion marker content_sha256 does not bind the checksum manifest."
        )
    try:
        manifest = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise RunValidationError(
            "Run manifest or summary cannot be parsed for identity validation."
        ) from error
    for name, document in (("manifest.yaml", manifest), ("summary.json", summary)):
        if not isinstance(document, Mapping) or document.get("run_id") != root.name:
            raise RunValidationError(
                f"{name} run_id does not match the run directory."
            )
    if markers[0] == "_SUCCESS" and require_success_contract:
        _validate_success_contract_at(
            root,
            root.name,
            tombstoned_paths=verified_tombstones,
        )
    return {
        "valid": True,
        "status": expected_status,
        "file_count": len(expected),
        "present_file_count": len(actual),
        "tombstoned_file_count": len(verified_tombstones),
        "run_path": root.as_posix(),
    }


def verify_unmarked_run_bundle(run_path: str | Path) -> dict[str, Any]:
    """Verify published success content before/recovering marker creation."""

    root = Path(run_path)
    if not root.is_dir():
        raise RunValidationError(f"Run bundle is not a directory: {root}")
    markers = [marker for marker in COMPLETION_MARKERS if (root / marker).exists()]
    if markers:
        raise RunValidationError(
            f"Expected an unmarked finalizing bundle; found {markers}."
        )
    broken_links = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink() and not path.exists()
    ]
    if broken_links:
        raise RunValidationError(
            "Run bundle contains broken symlinks: " + ", ".join(broken_links)
        )
    checksum_path = root / "provenance/artifact_checksums.json"
    if not checksum_path.is_file():
        raise RunValidationError("Run bundle has no artifact checksum manifest.")
    payload = json.loads(checksum_path.read_text(encoding="utf-8"))
    expected = payload.get("files") if isinstance(payload, Mapping) else None
    if not isinstance(expected, Mapping) or _bundle_checksums(root) != expected:
        raise RunValidationError(
            "Unmarked finalizing bundle does not match its checksum manifest."
        )
    _validate_success_contract_at(root, root.name)
    return {
        "valid": True,
        "status": "finalizing",
        "file_count": len(expected),
        "run_path": root.as_posix(),
    }


__all__ = [
    "COMPLETION_MARKERS",
    "FORBIDDEN_IDENTIFIER_COLUMNS",
    "PREDICTION_OPTIONAL_COLUMNS",
    "PREDICTION_REQUIRED_COLUMNS",
    "RUN_DIRECTORIES",
    "RunArchive",
    "RunArchiveError",
    "RunImmutableError",
    "RunValidationError",
    "deidentify_prediction_rows",
    "validate_prediction_rows",
    "verify_unmarked_run_bundle",
    "verify_run_bundle",
]
