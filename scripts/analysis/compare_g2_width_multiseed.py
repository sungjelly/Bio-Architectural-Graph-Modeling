#!/usr/bin/env python3
"""Audit and compare the locked six-run full-core G2 width campaign.

The comparison is deliberately campaign-specific.  It accepts exactly three
successful current-width and three successful wider-G2 runs, verifies their
complete protocol and execution evidence, aggregates fixed whole-node metrics
and resource diagnostics, and applies the prespecified H1 and Jacobian gates.

``masked_percent_variance_explained`` is 100 times masked R-squared, not
classification accuracy.  This utility never computes Jacobians.  When the
Jacobian gate opens, the result remains ``jacobian_required`` until a separate
workflow supplies that evidence.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import yaml


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.run_archive import (  # noqa: E402
    RunValidationError,
    verify_run_bundle,
)


_CAMPAIGN_ID = "cmp_20260726_full_core_g2_larger_multiseed"
_CAMPAIGN_DISPLAY_NAME = "Full-core G2 width scaling across seeds"
_PROTOCOL = "held_in_full_core_fixed_budget"
_BASELINE = "g2_width512_exact_k1000_full_core"
_WIDER = "g2_width1024_exact_k1000_full_core"
_RESOURCE_PILOT = "g2_width1024_exact_k1000_resource_pilot"
_CONCLUSION_VARIANTS = (_BASELINE, _WIDER)
_ALL_VARIANTS = (*_CONCLUSION_VARIANTS, _RESOURCE_PILOT)
_EXPECTED_SEEDS = (0, 1, 2)
_EXPECTED_EPOCHS = 200
_PILOT_EPOCHS = 2
_EXPECTED_REPLICATES = 3
_PILOT_REPLICATES = 1
_EXPECTED_PARAMETER_COUNTS = {
    _BASELINE: 3_987_880,
    _WIDER: 12_687_784,
    _RESOURCE_PILOT: 12_687_784,
}
_EXPECTED_SCIENTIFIC_IDS = {
    _BASELINE: (
        "e0ce78224f85f6654d49f86cf5328de41a16e905b12d984e8105392431c1d999"
    ),
    _WIDER: (
        "04a51acf353f8073b288187d8a64bc959b63424ad61fb84ac6df275aaf1da790"
    ),
    _RESOURCE_PILOT: (
        "1334b662fbdf517205612bf671a5d808cd04d5c3ddf966a052449fbdb192b7f5"
    ),
}
_EXPECTED_GRAPH_SHA256 = (
    "2469064e2fe14b48f642fca09a546d9e420d9fd851b9668996da62ee8246d060"
)
_EXPECTED_GRAPH_DIRECTED_EDGES = 21_029_944
_DATA_IDENTITY_FIELDS = (
    "dataset_id",
    "version",
    "dataset_fingerprint",
    "preprocessing_version",
    "split_id",
    "split_fingerprint",
)
_EXPECTED_EXPERIMENT = {
    _BASELINE: {
        "variant_label": _BASELINE,
        "estimand": "held_in_full_core_whole_node_masked_reconstruction",
        "permitted_claim": "one_core_transductive_width_capacity",
        "paired_variant": "full_core_g2_multiseed_width1024",
    },
    _WIDER: {
        "variant_label": _WIDER,
        "estimand": "held_in_full_core_whole_node_masked_reconstruction",
        "permitted_claim": "one_core_transductive_width_capacity",
        "paired_variant": "full_core_g2_multiseed_baseline",
    },
    _RESOURCE_PILOT: {
        "variant_label": _RESOURCE_PILOT,
        "estimand": "implementation_resource_feasibility_on_held_in_full_core",
        "permitted_claim": "diagnostic_runtime_and_memory_feasibility_only",
        "conclusion_eligible": False,
        "excluded_from_capacity_comparison": True,
    },
}
_EXPECTED_CLASSIFICATION = {
    _BASELINE: {
        "schema_version": 1,
        "lifecycle_stage": "exploratory_screen",
        "study_axis": "full_core_g2_width_capacity",
        "retention_class": "retain_exploratory_evidence",
        "classification_confidence": "high",
    },
    _WIDER: {
        "schema_version": 1,
        "lifecycle_stage": "exploratory_screen",
        "study_axis": "full_core_g2_width_capacity",
        "retention_class": "retain_exploratory_evidence",
        "classification_confidence": "high",
    },
    _RESOURCE_PILOT: {
        "schema_version": 1,
        "lifecycle_stage": "diagnostic",
        "study_axis": "g2_width_resource_feasibility",
        "retention_class": "retain_diagnostic_evidence",
        "classification_confidence": "high",
    },
}
_EXPECTED_MODEL_SHAPE = {
    _BASELINE: {
        "name": "g2",
        "family": "edge_conditioned_gatv2",
        "embedding_dim": 512,
        "hidden_dim": 512,
        "graph_layers": 2,
        "attention_heads": 4,
        "ffn_dim": 512,
        "decoder_dim": 512,
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "implicit_self_loops": False,
        "trainable_node_identifiers": False,
        "trainable_edge_identifiers": False,
        "receiver_chunk_size": 512,
        "activation_checkpointing": True,
        "exact_receiver_partitioning": True,
    },
    _WIDER: {
        "name": "g2",
        "family": "edge_conditioned_gatv2",
        "embedding_dim": 1024,
        "hidden_dim": 1024,
        "graph_layers": 2,
        "attention_heads": 8,
        "ffn_dim": 1024,
        "decoder_dim": 1024,
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "implicit_self_loops": False,
        "trainable_node_identifiers": False,
        "trainable_edge_identifiers": False,
        "receiver_chunk_size": 256,
        "activation_checkpointing": True,
        "exact_receiver_partitioning": True,
    },
}
_EXPECTED_MODEL_SHAPE[_RESOURCE_PILOT] = _EXPECTED_MODEL_SHAPE[_WIDER]

# SHA-256 of canonical JSON for the fully composed locked sections.  These
# reject missing, altered, and extra fields rather than checking a favorable
# subset.  Values are derived from the three campaign experiment definitions.
_SHARED_SECTION_DIGESTS = {
    "dataset": "bb6e5d7f59a465762094aef8ab2607a702b62e7be032ee2fc114b8e5eaa47932",
    "graph": "e72135fd7295bd4598b85b3c2f8ef41ec3850fc3ef462a3db1c9877454bbf0f2",
    "features": "294dcf4b06fae2e121d59687a3aecf9ab3aafaf203db7c0fd08068b77c8f3316",
}
_CONCLUSION_SECTION_DIGESTS = {
    **_SHARED_SECTION_DIGESTS,
    "masking": "f9f0871ded533a118943f4a11e585336156f58fa3eff367801256e81c63fd454",
    "trainer": "538206efa117768e2c6fd3a638724f173bc1953e60d66998bcf168391959ec07",
    "evaluation": "168ed4aea4e504111e8f91acee44dc7781794b70f76fbb0b0ab25c84765dabc8",
}
_PILOT_SECTION_DIGESTS = {
    **_SHARED_SECTION_DIGESTS,
    "masking": "3116310866e53b051384891ec280af1b3b9e56e506d3545f813897bf1dbcc7dc",
    "trainer": "70ffe2980b540c9bbdbdd5ec2abf696fd7f14371c351cca47b219d094032d241",
    "evaluation": "2542a68de2a76d688ffe713a40b217c5e9d28564c8002344b3b5b36fcf932761",
}
_COMPLETION_MARKERS = ("_SUCCESS", "_FAILED", "_PRUNED")
_TABLE_SUFFIXES = (".jsonl", ".parquet")
_SHA256_LENGTH = 64
_NUMBER_REL_TOL = 1e-9
_NUMBER_ABS_TOL = 1e-9
_RUN_ID_RE = re.compile(
    r"^r_\d{8}T\d{6}Z_(?P<scientific>[a-z0-9]+)_s(?P<seed>\d{3})_"
    r"f(?P<fold>\d{2})_a(?P<attempt>\d{2})_[a-z0-9_-]+$"
)
_QUEUE_ID_RE = re.compile(r"^q_[a-z0-9][a-z0-9_-]{2,63}$")
_FAILURE_CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_OWNER_FILE = ".g2-width-comparison-owner.json"
_OUTPUT_FILES = frozenset({_OWNER_FILE, "comparison.json", "report.md"})
_OWNER_PAYLOAD = {
    "schema_version": 1,
    "owner": "compare_g2_width_multiseed.py",
    "artifact_kind": "g2_width_multiseed_comparison",
    "campaign_id": _CAMPAIGN_ID,
}
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2


class G2WidthComparisonError(RuntimeError):
    """Raised when evidence violates the locked comparison contract."""


@dataclass(frozen=True)
class ExecutionLineage:
    seed: int
    fold: int
    attempt: int
    job_id: str
    retry_of: str | None


@dataclass(frozen=True)
class HistoryEvidence:
    table_format: str
    rows: int
    summed_epoch_duration_seconds: float
    final_train_loss: float
    minimum_train_loss: float
    last_20_epoch_loss_slope: float
    peak_vram_gb: float
    schedule: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class RunEvidence:
    root: Path
    run_id: str
    variant: str
    lineage: ExecutionLineage
    parameter_count: int
    graph_sha256: str
    graph_directed_edges: int
    mask_bundle_sha256: str
    evaluation_mask_identity: tuple[tuple[int, int, str, int], ...]
    data_identity: tuple[Any, ...]
    mean_huber: float
    mean_r2: float
    mean_pve: float
    evaluation_table_format: str
    history: HistoryEvidence
    total_duration_seconds: float
    recorded_training_duration_seconds: float


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise G2WidthComparisonError(f"{label} must be a mapping")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise G2WidthComparisonError(
            f"required JSON artifact is unreadable: {path.name}"
        ) from error
    return _mapping(value, path.name)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise G2WidthComparisonError(
            f"required YAML artifact is unreadable: {path.name}"
        ) from error
    return _mapping(value, path.name)


def _logical_table_path(root: Path, stem: str) -> Path:
    matches = [
        root / f"{stem}{suffix}"
        for suffix in _TABLE_SUFFIXES
        if (root / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise G2WidthComparisonError(
            f"{root.name} requires exactly one {stem} table in JSONL or "
            f"Parquet; found {[path.name for path in matches]}"
        )
    return matches[0]


def _load_table(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    if path.suffix == ".jsonl":
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except ValueError as error:
                        raise G2WidthComparisonError(
                            f"{path.name}:{line_number} is not valid JSON"
                        ) from error
                    rows.append(_mapping(value, f"{path.name}:{line_number}"))
        except OSError as error:
            raise G2WidthComparisonError(
                f"required JSONL table is unreadable: {path.name}"
            ) from error
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except (ImportError, ModuleNotFoundError) as error:
            raise G2WidthComparisonError(
                f"reading Parquet requires pyarrow: {path.name}"
            ) from error
        try:
            rows = [
                _mapping(row, f"row in {path.name}")
                for row in parquet.read_table(path).to_pylist()
            ]
        except Exception as error:
            raise G2WidthComparisonError(
                f"required Parquet table is unreadable: {path.name}"
            ) from error
    else:
        raise G2WidthComparisonError(f"unsupported table format: {path.name}")
    if not rows:
        raise G2WidthComparisonError(f"required table is empty: {path.name}")
    return tuple(rows)


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise G2WidthComparisonError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise G2WidthComparisonError(f"{label} must be an integer") from error
    if isinstance(value, float) and not value.is_integer():
        raise G2WidthComparisonError(f"{label} must be an integer")
    if isinstance(value, str) and str(converted) != value.strip():
        raise G2WidthComparisonError(f"{label} must be an integer")
    return converted


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise G2WidthComparisonError(f"{label} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise G2WidthComparisonError(
            f"{label} must be a finite number"
        ) from error
    if not math.isfinite(converted):
        raise G2WidthComparisonError(f"{label} must be a finite number")
    return converted


def _sha256(value: Any, label: str) -> str:
    checksum = str(value)
    if len(checksum) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise G2WidthComparisonError(f"{label} must be a lowercase SHA-256")
    return checksum


def _same_number(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_NUMBER_REL_TOL,
        abs_tol=_NUMBER_ABS_TOL,
    )


def _canonical_digest(value: Mapping[str, Any]) -> str:
    try:
        content = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise G2WidthComparisonError(
            "locked configuration section is not canonical JSON"
        ) from error
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _verify_marker(root: Path, expected_marker: str) -> None:
    markers = [
        marker for marker in _COMPLETION_MARKERS if (root / marker).is_file()
    ]
    if markers != [expected_marker]:
        raise G2WidthComparisonError(
            f"{root.name} requires exactly one {expected_marker} marker"
        )
    try:
        verification = verify_run_bundle(
            root,
            require_success_contract=expected_marker == "_SUCCESS",
        )
    except (RunValidationError, OSError, ValueError) as error:
        raise G2WidthComparisonError(
            f"{root.name} failed finalized-bundle verification"
        ) from error
    expected_status = expected_marker.removeprefix("_").lower()
    if verification.get("status") != expected_status:
        raise G2WidthComparisonError(
            f"{root.name} marker status is not {expected_status}"
        )


def _resolve_run_root(path: str | Path) -> Path:
    supplied = Path(path)
    try:
        root = supplied.resolve(strict=True)
    except OSError as error:
        raise G2WidthComparisonError(
            f"run directory does not exist: {supplied}"
        ) from error
    if not root.is_dir():
        raise G2WidthComparisonError(f"run input is not a directory: {supplied}")
    return root


def _validate_locked_config(
    config: Mapping[str, Any],
    *,
    variant: str,
    run_id: str,
) -> None:
    if config.get("version") != 1:
        raise G2WidthComparisonError(f"{run_id} config version is not locked v1")
    campaign = _mapping(config.get("campaign"), f"{run_id} config.campaign")
    if campaign != {
        "campaign_id": _CAMPAIGN_ID,
        "display_name": _CAMPAIGN_DISPLAY_NAME,
    }:
        raise G2WidthComparisonError(
            f"{run_id} campaign metadata is not the locked campaign"
        )
    experiment = _mapping(
        config.get("experiment"), f"{run_id} config.experiment"
    )
    if experiment != _EXPECTED_EXPERIMENT[variant]:
        raise G2WidthComparisonError(
            f"{run_id} experiment settings drifted for {variant}"
        )
    classification = _mapping(
        config.get("classification"), f"{run_id} config.classification"
    )
    if classification != _EXPECTED_CLASSIFICATION[variant]:
        raise G2WidthComparisonError(
            f"{run_id} classification settings drifted for {variant}"
        )

    expected_digests = (
        _PILOT_SECTION_DIGESTS
        if variant == _RESOURCE_PILOT
        else _CONCLUSION_SECTION_DIGESTS
    )
    for section, expected_digest in expected_digests.items():
        value = _mapping(config.get(section), f"{run_id} config.{section}")
        if _canonical_digest(value) != expected_digest:
            raise G2WidthComparisonError(
                f"{run_id} locked config.{section} settings drifted"
            )
    model = _mapping(config.get("model"), f"{run_id} config.model")
    if model != _EXPECTED_MODEL_SHAPE[variant]:
        raise G2WidthComparisonError(
            f"{run_id} exact model shape or receiver chunk drifted for {variant}"
        )
    if _as_int(config.get("fold"), f"{run_id} config.fold") != 0:
        raise G2WidthComparisonError(f"{run_id} fold must be 0")


def _execution_lineage(
    root: Path,
    *,
    config: Mapping[str, Any],
    variant: str,
) -> ExecutionLineage:
    run_id = root.name
    match = _RUN_ID_RE.fullmatch(run_id)
    if match is None:
        raise G2WidthComparisonError(
            f"{run_id} is not a canonical execution run identifier"
        )
    expected_scientific_token = _EXPECTED_SCIENTIFIC_IDS[variant][:8]
    if match.group("scientific") != expected_scientific_token:
        raise G2WidthComparisonError(
            f"{run_id} scientific identity does not match {variant}"
        )
    queue = _load_json(root / "provenance/queue.json")
    job_id = str(queue.get("job_id", ""))
    if _QUEUE_ID_RE.fullmatch(job_id) is None:
        raise G2WidthComparisonError(f"{run_id} queue job_id is not sanitized")
    retry_value = queue.get("retry_of")
    retry_of: str | None
    if retry_value is None:
        retry_of = None
    else:
        retry_of = str(retry_value)
        if _QUEUE_ID_RE.fullmatch(retry_of) is None:
            raise G2WidthComparisonError(
                f"{run_id} queue retry_of is not a sanitized job identifier"
            )
    seed = _as_int(config.get("seed"), f"{run_id} config.seed")
    fold = _as_int(config.get("fold"), f"{run_id} config.fold")
    attempt = _as_int(queue.get("attempt"), f"{run_id} queue.attempt")
    config_attempt = _as_int(config.get("attempt"), f"{run_id} config.attempt")
    encoded = {
        "seed": int(match.group("seed")),
        "fold": int(match.group("fold")),
        "attempt": int(match.group("attempt")),
    }
    observed = {"seed": seed, "fold": fold, "attempt": attempt}
    if observed != encoded:
        raise G2WidthComparisonError(
            f"{run_id} run identifier and execution lineage disagree"
        )
    if config_attempt != attempt:
        raise G2WidthComparisonError(
            f"{run_id} resolved config attempt and queue attempt disagree"
        )
    if attempt < 1:
        raise G2WidthComparisonError(f"{run_id} attempt must be positive")
    return ExecutionLineage(
        seed=seed,
        fold=fold,
        attempt=attempt,
        job_id=job_id,
        retry_of=retry_of,
    )


def _data_identity(config: Mapping[str, Any], run_id: str) -> tuple[Any, ...]:
    dataset = _mapping(config.get("dataset"), f"{run_id} config.dataset")
    values = tuple(dataset.get(field) for field in _DATA_IDENTITY_FIELDS)
    if any(
        value is None or (isinstance(value, str) and not value.strip())
        for value in values
    ):
        raise G2WidthComparisonError(f"{run_id} dataset identity is incomplete")
    _sha256(values[2], f"{run_id} dataset_fingerprint")
    _sha256(values[5], f"{run_id} split_fingerprint")
    return values


def _linear_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise G2WidthComparisonError("at least two losses are required for slope")
    x_mean = (len(values) - 1) / 2.0
    y_mean = statistics.fmean(values)
    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )
    denominator = sum(
        (index - x_mean) ** 2 for index in range(len(values))
    )
    return numerator / denominator


def _validate_history(
    root: Path,
    *,
    expected_epochs: int,
    graph_directed_edges: int,
    summary: Mapping[str, Any],
    convergence: Mapping[str, Any],
) -> HistoryEvidence:
    run_id = root.name
    path = _logical_table_path(root, "metrics/history")
    rows = _load_table(path)
    if len(rows) != expected_epochs:
        raise G2WidthComparisonError(
            f"{run_id} requires exactly {expected_epochs} complete history rows"
        )

    normalized: dict[int, tuple[float, float, int, tuple[Any, ...]]] = {}
    for row_number, row in enumerate(rows, start=1):
        if row.get("run_id") != run_id:
            raise G2WidthComparisonError(
                f"{run_id} history row {row_number} has the wrong run_id"
            )
        if row.get("split") != "fit" or row.get("training_protocol") != _PROTOCOL:
            raise G2WidthComparisonError(
                f"{run_id} history row {row_number} is outside the fit protocol"
            )
        epoch = _as_int(
            row.get("epoch"), f"{run_id} history row {row_number} epoch"
        )
        if epoch in normalized:
            raise G2WidthComparisonError(
                f"{run_id} repeats history epoch {epoch}"
            )
        mask_mode = str(row.get("mask_mode", ""))
        if mask_mode not in {"partial", "node", "block"}:
            raise G2WidthComparisonError(
                f"{run_id} history row {row_number} has an unknown mask mode"
            )
        mask_seed = _as_int(
            row.get("mask_seed"),
            f"{run_id} history row {row_number} mask_seed",
        )
        mask_checksum = _sha256(
            row.get("mask_checksum"),
            f"{run_id} history row {row_number} mask_checksum",
        )
        edge_dropout_seed = _as_int(
            row.get("edge_dropout_seed"),
            f"{run_id} history row {row_number} edge_dropout_seed",
        )
        edge_checksum = _sha256(
            row.get("edge_checksum"),
            f"{run_id} history row {row_number} edge_checksum",
        )
        n_masked_entries = _as_int(
            row.get("n_masked_entries"),
            f"{run_id} history row {row_number} n_masked_entries",
        )
        n_target_nodes = _as_int(
            row.get("n_target_nodes"),
            f"{run_id} history row {row_number} n_target_nodes",
        )
        n_edges_used = _as_int(
            row.get("n_edges_used"),
            f"{run_id} history row {row_number} n_edges_used",
        )
        if n_masked_entries <= 0 or n_target_nodes <= 0:
            raise G2WidthComparisonError(
                f"{run_id} history row {row_number} has an empty training mask"
            )
        if n_edges_used != graph_directed_edges:
            raise G2WidthComparisonError(
                f"{run_id} history row {row_number} did not use the full graph"
            )
        train_loss = _finite_float(
            row.get("train_loss"),
            f"{run_id} history row {row_number} train_loss",
        )
        diagnostic_loss = row.get("diagnostic_loss")
        if diagnostic_loss is not None:
            _finite_float(
                diagnostic_loss,
                f"{run_id} history row {row_number} diagnostic_loss",
            )
        gradient_norm = _finite_float(
            row.get("gradient_norm"),
            f"{run_id} history row {row_number} gradient_norm",
        )
        duration = _finite_float(
            row.get("duration_seconds"),
            f"{run_id} history row {row_number} duration_seconds",
        )
        peak_bytes = _as_int(
            row.get("peak_cuda_memory_bytes"),
            f"{run_id} history row {row_number} peak_cuda_memory_bytes",
        )
        if train_loss < 0.0 or gradient_norm < 0.0 or duration < 0.0:
            raise G2WidthComparisonError(
                f"{run_id} history row {row_number} has a negative diagnostic"
            )
        if peak_bytes < 0:
            raise G2WidthComparisonError(
                f"{run_id} history row {row_number} has negative peak VRAM"
            )
        schedule = (
            epoch,
            mask_mode,
            mask_seed,
            mask_checksum,
            edge_dropout_seed,
            edge_checksum,
            n_masked_entries,
            n_target_nodes,
            n_edges_used,
        )
        normalized[epoch] = (train_loss, duration, peak_bytes, schedule)

    if tuple(sorted(normalized)) != tuple(range(expected_epochs)):
        raise G2WidthComparisonError(
            f"{run_id} history epochs are not exactly 0 through "
            f"{expected_epochs - 1}"
        )
    ordered = [normalized[index] for index in range(expected_epochs)]
    losses = [item[0] for item in ordered]
    durations = [item[1] for item in ordered]
    peak_vram_gb = max(item[2] for item in ordered) / (1024**3)
    final_loss = losses[-1]
    minimum_loss = min(losses)
    slope = _linear_slope(losses[-min(20, len(losses)) :])

    if (
        _as_int(
            summary.get("fixed_epoch_budget"),
            f"{run_id} summary.fixed_epoch_budget",
        )
        != expected_epochs
        or _as_int(summary.get("final_epoch"), f"{run_id} summary.final_epoch")
        != expected_epochs - 1
    ):
        raise G2WidthComparisonError(
            f"{run_id} summary does not confirm all fixed epochs"
        )
    if (
        _as_int(
            convergence.get("final_epoch"),
            f"{run_id} convergence.final_epoch",
        )
        != expected_epochs - 1
        or convergence.get("all_epochs_completed") is not True
        or convergence.get("all_losses_and_gradients_finite") is not True
    ):
        raise G2WidthComparisonError(
            f"{run_id} convergence does not confirm complete finite training"
        )
    convergence_final = _finite_float(
        convergence.get("final_train_loss"),
        f"{run_id} convergence.final_train_loss",
    )
    convergence_minimum = _finite_float(
        convergence.get("minimum_observed_train_loss"),
        f"{run_id} convergence.minimum_observed_train_loss",
    )
    convergence_slope = _finite_float(
        convergence.get("last_20_epoch_loss_slope"),
        f"{run_id} convergence.last_20_epoch_loss_slope",
    )
    if not _same_number(final_loss, convergence_final):
        raise G2WidthComparisonError(
            f"{run_id} final history loss disagrees with convergence"
        )
    if not _same_number(minimum_loss, convergence_minimum):
        raise G2WidthComparisonError(
            f"{run_id} minimum history loss disagrees with convergence"
        )
    if not _same_number(slope, convergence_slope):
        raise G2WidthComparisonError(
            f"{run_id} last-20 loss slope disagrees with convergence"
        )
    return HistoryEvidence(
        table_format=path.suffix.removeprefix("."),
        rows=len(rows),
        summed_epoch_duration_seconds=sum(durations),
        final_train_loss=final_loss,
        minimum_train_loss=minimum_loss,
        last_20_epoch_loss_slope=slope,
        peak_vram_gb=peak_vram_gb,
        schedule=tuple(item[3] for item in ordered),
    )


def _validate_evaluation(
    root: Path,
    *,
    expected_replicates: int,
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
) -> tuple[
    float,
    float,
    float,
    tuple[tuple[int, int, str, int], ...],
    str,
]:
    run_id = root.name
    summary_metrics = _mapping(
        summary.get("metrics"), f"{run_id} summary.metrics"
    )
    if dict(summary_metrics) != dict(final_metrics):
        raise G2WidthComparisonError(
            f"{run_id} metrics/final.json disagrees with summary.metrics"
        )
    path = _logical_table_path(root, "metrics/evaluation_replicates")
    rows = _load_table(path)
    selected: dict[int, tuple[float, float, float, int, str, int]] = {}
    for row_number, row in enumerate(rows, start=1):
        if row.get("mask_mode") != "whole_node":
            continue
        if row.get("split") != "fit":
            raise G2WidthComparisonError(
                f"{run_id} whole-node row {row_number} is not fit"
            )
        replicate = _as_int(
            row.get("mask_replicate"),
            f"{run_id} whole-node row {row_number} mask_replicate",
        )
        if replicate in selected:
            raise G2WidthComparisonError(
                f"{run_id} repeats whole-node mask replicate {replicate}"
            )
        huber = _finite_float(
            row.get("masked_huber"),
            f"{run_id} whole-node row {row_number} masked_huber",
        )
        r2 = _finite_float(
            row.get("masked_r2"),
            f"{run_id} whole-node row {row_number} masked_r2",
        )
        pve = _finite_float(
            row.get("masked_percent_variance_explained"),
            f"{run_id} whole-node row {row_number} PVE",
        )
        if huber < 0.0:
            raise G2WidthComparisonError(
                f"{run_id} whole-node Huber cannot be negative"
            )
        if r2 > 1.0 and not _same_number(r2, 1.0):
            raise G2WidthComparisonError(
                f"{run_id} whole-node row {row_number} R2 cannot exceed 1"
            )
        if not _same_number(pve, 100.0 * r2):
            raise G2WidthComparisonError(
                f"{run_id} whole-node row {row_number} PVE is not 100 * R2"
            )
        mask_seed = _as_int(
            row.get("mask_seed"),
            f"{run_id} whole-node row {row_number} mask_seed",
        )
        mask_checksum = _sha256(
            row.get("mask_checksum"),
            f"{run_id} whole-node row {row_number} mask_checksum",
        )
        n_masked = _as_int(
            row.get("n_masked"),
            f"{run_id} whole-node row {row_number} n_masked",
        )
        if n_masked <= 0:
            raise G2WidthComparisonError(
                f"{run_id} whole-node row {row_number} has no masked entries"
            )
        selected[replicate] = (
            huber,
            r2,
            pve,
            mask_seed,
            mask_checksum,
            n_masked,
        )

    expected_indices = tuple(range(expected_replicates))
    if tuple(sorted(selected)) != expected_indices:
        raise G2WidthComparisonError(
            f"{run_id} requires whole-node mask replicates "
            f"{list(expected_indices)}"
        )
    ordered = [selected[index] for index in expected_indices]
    mean_huber = statistics.fmean(item[0] for item in ordered)
    mean_r2 = statistics.fmean(item[1] for item in ordered)
    mean_pve = statistics.fmean(item[2] for item in ordered)
    mask_identity = tuple(
        (index, item[3], item[4], item[5])
        for index, item in enumerate(ordered)
    )
    final_huber = _finite_float(
        final_metrics.get("fit/whole_node/masked_huber"),
        f"{run_id} final whole-node Huber",
    )
    final_r2 = _finite_float(
        final_metrics.get("fit/whole_node/masked_r2"),
        f"{run_id} final whole-node R2",
    )
    final_pve = _finite_float(
        final_metrics.get(
            "fit/whole_node/masked_percent_variance_explained"
        ),
        f"{run_id} final whole-node PVE",
    )
    if not _same_number(mean_huber, final_huber):
        raise G2WidthComparisonError(
            f"{run_id} whole-node Huber replicate mean disagrees with final metrics"
        )
    if not _same_number(mean_r2, final_r2):
        raise G2WidthComparisonError(
            f"{run_id} whole-node R2 replicate mean disagrees with final metrics"
        )
    if not _same_number(mean_pve, final_pve):
        raise G2WidthComparisonError(
            f"{run_id} whole-node PVE replicate mean disagrees with final metrics"
        )
    if not _same_number(final_pve, 100.0 * final_r2):
        raise G2WidthComparisonError(
            f"{run_id} final PVE is not 100 * final R2"
        )
    if final_r2 > 1.0 and not _same_number(final_r2, 1.0):
        raise G2WidthComparisonError(
            f"{run_id} final R2 cannot exceed 1"
        )
    if (
        summary.get("primary_metric_name")
        != "fit/whole_node/masked_huber"
        or not _same_number(
            _finite_float(
                summary.get("primary_metric_value"),
                f"{run_id} summary.primary_metric_value",
            ),
            mean_huber,
        )
    ):
        raise G2WidthComparisonError(
            f"{run_id} primary metric does not reconcile to whole-node Huber"
        )
    return (
        mean_huber,
        mean_r2,
        mean_pve,
        mask_identity,
        path.suffix.removeprefix("."),
    )


def _load_success_run(
    path: str | Path,
    *,
    expected_kind: str,
) -> RunEvidence:
    root = _resolve_run_root(path)
    _verify_marker(root, "_SUCCESS")
    config = _load_yaml(root / "config.resolved.yaml")
    summary = _load_json(root / "summary.json")
    final_metrics = _load_json(root / "metrics/final.json")
    convergence = _load_json(root / "diagnostics/training_convergence.json")
    run_id = root.name
    if summary.get("run_id") != run_id:
        raise G2WidthComparisonError(
            f"{run_id} summary run_id does not match its directory"
        )
    experiment = _mapping(
        config.get("experiment"), f"{run_id} config.experiment"
    )
    variant = str(experiment.get("variant_label", ""))
    allowed = (
        {_RESOURCE_PILOT}
        if expected_kind == "resource_pilot"
        else set(_CONCLUSION_VARIANTS)
    )
    if variant not in allowed:
        raise G2WidthComparisonError(
            f"{run_id} is not an allowed {expected_kind} variant"
        )
    _validate_locked_config(config, variant=variant, run_id=run_id)
    lineage = _execution_lineage(root, config=config, variant=variant)
    if variant == _RESOURCE_PILOT:
        if lineage.seed != 0:
            raise G2WidthComparisonError(f"{run_id} resource pilot seed must be 0")
        expected_epochs = _PILOT_EPOCHS
        expected_replicates = _PILOT_REPLICATES
        if (
            summary.get("diagnostic_resource_pilot") is not True
            or summary.get("conclusion_eligible") is not False
        ):
            raise G2WidthComparisonError(
                f"{run_id} is not explicitly a diagnostic resource pilot"
            )
    else:
        if lineage.seed not in _EXPECTED_SEEDS:
            raise G2WidthComparisonError(
                f"{run_id} model seed is outside 0, 1, and 2"
            )
        expected_epochs = _EXPECTED_EPOCHS
        expected_replicates = _EXPECTED_REPLICATES
        if (
            summary.get("diagnostic_resource_pilot") is not False
            or summary.get("conclusion_eligible") is not True
        ):
            raise G2WidthComparisonError(
                f"{run_id} is not conclusion-eligible"
            )
    if (
        summary.get("status") != "success"
        or summary.get("training_exit_status") != "success"
        or summary.get("evaluation_protocol") != _PROTOCOL
        or summary.get("generalization_estimate") is not False
        or summary.get("model_name") != "g2"
        or _as_int(summary.get("model_seed"), f"{run_id} summary.model_seed")
        != lineage.seed
    ):
        raise G2WidthComparisonError(
            f"{run_id} summary is outside the locked successful protocol"
        )
    if (
        _as_int(
            summary.get("evaluation_mask_replicates_per_mode"),
            f"{run_id} summary evaluation replicate count",
        )
        != expected_replicates
        or summary.get(
            "evaluation_metrics_include_all_configured_replicates_per_mode"
        )
        is not True
    ):
        raise G2WidthComparisonError(
            f"{run_id} summary does not confirm every evaluation replicate"
        )

    graph_sha256 = _sha256(
        summary.get("graph_sha256"), f"{run_id} summary.graph_sha256"
    )
    graph_directed_edges = _as_int(
        summary.get("graph_directed_edges"),
        f"{run_id} summary.graph_directed_edges",
    )
    if (
        graph_sha256 != _EXPECTED_GRAPH_SHA256
        or graph_directed_edges != _EXPECTED_GRAPH_DIRECTED_EDGES
    ):
        raise G2WidthComparisonError(
            f"{run_id} materialized graph is not the locked campaign graph"
        )
    mask_bundle_sha256 = _sha256(
        summary.get("evaluation_mask_bundle_sha256"),
        f"{run_id} summary evaluation mask bundle",
    )
    (
        mean_huber,
        mean_r2,
        mean_pve,
        evaluation_mask_identity,
        evaluation_table_format,
    ) = _validate_evaluation(
        root,
        expected_replicates=expected_replicates,
        summary=summary,
        final_metrics=final_metrics,
    )
    history = _validate_history(
        root,
        expected_epochs=expected_epochs,
        graph_directed_edges=graph_directed_edges,
        summary=summary,
        convergence=convergence,
    )

    parameter_sources = {
        "summary": _as_int(
            summary.get("parameter_count"), f"{run_id} summary.parameter_count"
        ),
        "final": _as_int(
            final_metrics.get("resource/parameter_count"),
            f"{run_id} final parameter_count",
        ),
    }
    if len(set(parameter_sources.values())) != 1:
        raise G2WidthComparisonError(
            f"{run_id} parameter-count records disagree"
        )
    parameter_count = parameter_sources["summary"]
    if parameter_count != _EXPECTED_PARAMETER_COUNTS[variant]:
        raise G2WidthComparisonError(
            f"{run_id} parameter count is {parameter_count}, expected "
            f"{_EXPECTED_PARAMETER_COUNTS[variant]}"
        )
    total_duration = _finite_float(
        final_metrics.get("resource/total_duration_seconds"),
        f"{run_id} final total duration",
    )
    summary_duration = _finite_float(
        summary.get("duration_seconds"), f"{run_id} summary duration"
    )
    recorded_training_duration = _finite_float(
        final_metrics.get("resource/training_duration_seconds"),
        f"{run_id} final training duration",
    )
    final_peak_vram = _finite_float(
        final_metrics.get("resource/peak_vram_gb"),
        f"{run_id} final peak VRAM",
    )
    summary_peak_vram = _finite_float(
        summary.get("peak_vram_gb"), f"{run_id} summary peak VRAM"
    )
    if total_duration < 0.0 or recorded_training_duration < 0.0:
        raise G2WidthComparisonError(f"{run_id} durations cannot be negative")
    if not _same_number(total_duration, summary_duration):
        raise G2WidthComparisonError(
            f"{run_id} total duration disagrees between final metrics and summary"
        )
    if recorded_training_duration > total_duration + _NUMBER_ABS_TOL:
        raise G2WidthComparisonError(
            f"{run_id} training duration exceeds total duration"
        )
    if (
        history.summed_epoch_duration_seconds
        > recorded_training_duration + _NUMBER_ABS_TOL
    ):
        raise G2WidthComparisonError(
            f"{run_id} summed epoch durations exceed recorded training duration"
        )
    if not (
        _same_number(history.peak_vram_gb, final_peak_vram)
        and _same_number(history.peak_vram_gb, summary_peak_vram)
    ):
        raise G2WidthComparisonError(
            f"{run_id} peak VRAM disagrees across history, final metrics, and summary"
        )
    return RunEvidence(
        root=root,
        run_id=run_id,
        variant=variant,
        lineage=lineage,
        parameter_count=parameter_count,
        graph_sha256=graph_sha256,
        graph_directed_edges=graph_directed_edges,
        mask_bundle_sha256=mask_bundle_sha256,
        evaluation_mask_identity=evaluation_mask_identity,
        data_identity=_data_identity(config, run_id),
        mean_huber=mean_huber,
        mean_r2=mean_r2,
        mean_pve=mean_pve,
        evaluation_table_format=evaluation_table_format,
        history=history,
        total_duration_seconds=total_duration,
        recorded_training_duration_seconds=recorded_training_duration,
    )


def _load_failed_run(path: str | Path) -> dict[str, Any]:
    root = _resolve_run_root(path)
    _verify_marker(root, "_FAILED")
    config = _load_yaml(root / "config.resolved.yaml")
    summary = _load_json(root / "summary.json")
    run_id = root.name
    if summary.get("run_id") != run_id or summary.get("status") != "failed":
        raise G2WidthComparisonError(
            f"{run_id} failed summary does not match its marker"
        )
    experiment = _mapping(
        config.get("experiment"), f"{run_id} config.experiment"
    )
    variant = str(experiment.get("variant_label", ""))
    if variant not in _ALL_VARIANTS:
        raise G2WidthComparisonError(
            f"{run_id} failed run has an unsupported campaign variant"
        )
    _validate_locked_config(config, variant=variant, run_id=run_id)
    lineage = _execution_lineage(root, config=config, variant=variant)
    expected_seeds = (0,) if variant == _RESOURCE_PILOT else _EXPECTED_SEEDS
    if lineage.seed not in expected_seeds:
        raise G2WidthComparisonError(
            f"{run_id} failed run seed is outside its campaign variant"
        )
    category = str(summary.get("failure_category", ""))
    if _FAILURE_CATEGORY_RE.fullmatch(category) is None:
        raise G2WidthComparisonError(
            f"{run_id} failure category is absent or not privacy-safe"
        )
    return {
        "run_id": run_id,
        "variant_label": variant,
        "seed": lineage.seed,
        "fold": lineage.fold,
        "attempt": lineage.attempt,
        "job_id": lineage.job_id,
        "retry_of": lineage.retry_of,
        "failure_category": category,
    }


def _statistics(values: Sequence[float]) -> dict[str, float | int]:
    if len(values) != len(_EXPECTED_SEEDS):
        raise G2WidthComparisonError("seed aggregate requires exactly three values")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _verify_six_run_design(
    runs: Sequence[RunEvidence],
) -> dict[str, dict[int, RunEvidence]]:
    if len(runs) != 6:
        raise G2WidthComparisonError(
            f"exactly six finalized success runs are required; got {len(runs)}"
        )
    roots = [run.root for run in runs]
    if len(set(roots)) != len(roots):
        raise G2WidthComparisonError("successful run directories must be distinct")
    grouped: dict[str, dict[int, RunEvidence]] = {
        _BASELINE: {},
        _WIDER: {},
    }
    for run in runs:
        if run.lineage.seed in grouped[run.variant]:
            raise G2WidthComparisonError(
                f"{run.variant} repeats model seed {run.lineage.seed}"
            )
        grouped[run.variant][run.lineage.seed] = run
    for variant, by_seed in grouped.items():
        if tuple(sorted(by_seed)) != _EXPECTED_SEEDS:
            raise G2WidthComparisonError(
                f"{variant} requires model seeds 0, 1, and 2 exactly once"
            )

    identities = {
        "materialized graph": {
            (run.graph_sha256, run.graph_directed_edges) for run in runs
        },
        "evaluation mask bundle": {
            (run.mask_bundle_sha256, run.evaluation_mask_identity)
            for run in runs
        },
        "dataset and split": {run.data_identity for run in runs},
    }
    for label, observed in identities.items():
        if len(observed) != 1:
            raise G2WidthComparisonError(
                f"the six runs do not share one {label} identity"
            )
    for seed in _EXPECTED_SEEDS:
        if grouped[_BASELINE][seed].history.schedule != grouped[_WIDER][
            seed
        ].history.schedule:
            raise G2WidthComparisonError(
                f"seed {seed} training mask or edge-dropout schedules are not paired"
            )
    return grouped


def _lineage_json(lineage: ExecutionLineage) -> dict[str, Any]:
    return {
        "seed": lineage.seed,
        "fold": lineage.fold,
        "attempt": lineage.attempt,
        "job_id": lineage.job_id,
        "retry_of": lineage.retry_of,
    }


def _validate_retry_lineage(
    grouped: Mapping[str, Mapping[int, RunEvidence]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> None:
    failures_by_job: dict[str, Mapping[str, Any]] = {}
    for row in failure_rows:
        job_id = str(row["job_id"])
        if job_id in failures_by_job:
            raise G2WidthComparisonError(
                f"failed-run ledger repeats queue job {job_id}"
            )
        failures_by_job[job_id] = row

    for variant in _CONCLUSION_VARIANTS:
        for seed in _EXPECTED_SEEDS:
            run = grouped[variant][seed]
            lineage = run.lineage
            if lineage.attempt == 1:
                if lineage.retry_of is not None:
                    raise G2WidthComparisonError(
                        f"{run.run_id} attempt 1 cannot declare retry_of"
                    )
                continue
            if lineage.retry_of is None:
                raise G2WidthComparisonError(
                    f"{run.run_id} attempt {lineage.attempt} lacks retry_of"
                )
            prior = failures_by_job.get(lineage.retry_of)
            if prior is None:
                raise G2WidthComparisonError(
                    f"{run.run_id} retry predecessor is absent from the "
                    "supplied failed-run ledger"
                )
            expected = {
                "variant_label": variant,
                "seed": seed,
                "fold": lineage.fold,
                "attempt": lineage.attempt - 1,
            }
            observed = {key: prior.get(key) for key in expected}
            if observed != expected:
                raise G2WidthComparisonError(
                    f"{run.run_id} retry predecessor does not match the "
                    "same variant, seed, fold, and preceding attempt"
                )


def _resource_pilot_result(run: RunEvidence | None) -> dict[str, Any]:
    if run is None:
        return {"supplied": False, "status": "not_supplied"}
    projected_training_seconds = (
        run.recorded_training_duration_seconds / _PILOT_EPOCHS
    ) * _EXPECTED_EPOCHS
    criteria = {
        "peak_vram_at_most_20_5_gib": run.history.peak_vram_gb <= 20.5,
        "projected_200_epoch_training_at_most_6_hours": (
            projected_training_seconds <= 6.0 * 3600.0
        ),
    }
    return {
        "supplied": True,
        "status": "passed" if all(criteria.values()) else "failed",
        "run_id": run.run_id,
        "variant_label": run.variant,
        **_lineage_json(run.lineage),
        "parameter_count": run.parameter_count,
        "total_duration_seconds": run.total_duration_seconds,
        "recorded_training_duration_seconds": (
            run.recorded_training_duration_seconds
        ),
        "summed_epoch_duration_seconds": (
            run.history.summed_epoch_duration_seconds
        ),
        "summed_training_duration_seconds": (
            run.history.summed_epoch_duration_seconds
        ),
        "peak_vram_gb": run.history.peak_vram_gb,
        "projected_200_epoch_training_seconds": projected_training_seconds,
        "projection_formula": (
            "recorded 2-epoch training duration / 2 * 200"
        ),
        "thresholds": {
            "maximum_peak_vram_gib": 20.5,
            "maximum_projected_200_epoch_training_seconds": 21_600.0,
        },
        "criteria": criteria,
    }


def compare_g2_width_multiseed(
    run_directories: Sequence[str | Path],
    *,
    failed_run_directories: Sequence[str | Path] = (),
    resource_pilot: str | Path | None = None,
    expected_failed_run_count: int | None = None,
) -> dict[str, Any]:
    """Verify campaign evidence and return the locked comparison record."""

    if len(run_directories) != 6:
        raise G2WidthComparisonError(
            "exactly six finalized success run directories are required"
        )
    runs = tuple(
        _load_success_run(path, expected_kind="conclusion")
        for path in run_directories
    )
    grouped = _verify_six_run_design(runs)
    pilot = (
        _load_success_run(resource_pilot, expected_kind="resource_pilot")
        if resource_pilot is not None
        else None
    )
    failure_rows = [_load_failed_run(path) for path in failed_run_directories]
    all_roots = [run.root for run in runs]
    if pilot is not None:
        all_roots.append(pilot.root)
    failed_roots = [_resolve_run_root(path) for path in failed_run_directories]
    if len(set(failed_roots)) != len(failed_roots):
        raise G2WidthComparisonError("failed run directories must be distinct")
    if set(all_roots).intersection(failed_roots):
        raise G2WidthComparisonError(
            "a run directory cannot be both success and failed evidence"
        )
    failure_rows.sort(
        key=lambda row: (
            str(row["variant_label"]),
            int(row["seed"]),
            int(row["attempt"]),
            str(row["run_id"]),
        )
    )
    if expected_failed_run_count is not None:
        if (
            isinstance(expected_failed_run_count, bool)
            or not isinstance(expected_failed_run_count, int)
            or expected_failed_run_count < 0
        ):
            raise G2WidthComparisonError(
                "expected_failed_run_count must be a non-negative integer"
            )
        if len(failure_rows) != expected_failed_run_count:
            raise G2WidthComparisonError(
                "supplied failed-run ledger count does not match "
                f"expected_failed_run_count={expected_failed_run_count}"
            )
    _validate_retry_lineage(grouped, failure_rows)
    pilot_result = _resource_pilot_result(pilot)

    per_run: list[dict[str, Any]] = []
    variant_aggregates: dict[str, Any] = {}
    for variant in _CONCLUSION_VARIANTS:
        variant_runs = [grouped[variant][seed] for seed in _EXPECTED_SEEDS]
        for run in variant_runs:
            per_run.append(
                {
                    "run_id": run.run_id,
                    "variant_label": run.variant,
                    **_lineage_json(run.lineage),
                    "parameter_count": run.parameter_count,
                    "whole_node_mask_replicates": _EXPECTED_REPLICATES,
                    "mean_whole_node_masked_huber": run.mean_huber,
                    "mean_whole_node_masked_r2": run.mean_r2,
                    (
                        "mean_whole_node_"
                        "masked_percent_variance_explained"
                    ): run.mean_pve,
                    "total_duration_seconds": run.total_duration_seconds,
                    "recorded_training_duration_seconds": (
                        run.recorded_training_duration_seconds
                    ),
                    "summed_epoch_duration_seconds": (
                        run.history.summed_epoch_duration_seconds
                    ),
                    "summed_training_duration_seconds": (
                        run.history.summed_epoch_duration_seconds
                    ),
                    "peak_vram_gb": run.history.peak_vram_gb,
                    "final_train_loss": run.history.final_train_loss,
                    "minimum_train_loss": run.history.minimum_train_loss,
                    "last_20_epoch_loss_slope": (
                        run.history.last_20_epoch_loss_slope
                    ),
                    "history_table_format": run.history.table_format,
                    "evaluation_table_format": run.evaluation_table_format,
                }
            )
        variant_aggregates[variant] = {
            "model_seeds": list(_EXPECTED_SEEDS),
            "parameter_count": _EXPECTED_PARAMETER_COUNTS[variant],
            "whole_node_masked_huber": _statistics(
                [run.mean_huber for run in variant_runs]
            ),
            "whole_node_masked_percent_variance_explained": _statistics(
                [run.mean_pve for run in variant_runs]
            ),
            "resources": {
                "total_duration_seconds": _statistics(
                    [run.total_duration_seconds for run in variant_runs]
                ),
                "recorded_training_duration_seconds": _statistics(
                    [
                        run.recorded_training_duration_seconds
                        for run in variant_runs
                    ]
                ),
                "summed_epoch_duration_seconds": _statistics(
                    [
                        run.history.summed_epoch_duration_seconds
                        for run in variant_runs
                    ]
                ),
                "summed_training_duration_seconds": _statistics(
                    [
                        run.history.summed_epoch_duration_seconds
                        for run in variant_runs
                    ]
                ),
                "peak_vram_gb": _statistics(
                    [run.history.peak_vram_gb for run in variant_runs]
                ),
            },
            "convergence": {
                "final_train_loss": _statistics(
                    [run.history.final_train_loss for run in variant_runs]
                ),
                "minimum_train_loss": _statistics(
                    [run.history.minimum_train_loss for run in variant_runs]
                ),
                "last_20_epoch_loss_slope": _statistics(
                    [
                        run.history.last_20_epoch_loss_slope
                        for run in variant_runs
                    ]
                ),
            },
        }

    paired: list[dict[str, Any]] = []
    for seed in _EXPECTED_SEEDS:
        current = grouped[_BASELINE][seed]
        large = grouped[_WIDER][seed]
        if current.mean_huber <= 0.0:
            raise G2WidthComparisonError(
                f"baseline seed {seed} Huber must be positive for relative change"
            )
        huber_reduction = current.mean_huber - large.mean_huber
        relative_reduction = huber_reduction / current.mean_huber
        pve_increase = large.mean_pve - current.mean_pve
        paired.append(
            {
                "seed": seed,
                "baseline_mean_huber": current.mean_huber,
                "wider_mean_huber": large.mean_huber,
                "baseline_minus_wider_huber": huber_reduction,
                "relative_huber_reduction": relative_reduction,
                "baseline_mean_percent_variance_explained": current.mean_pve,
                "wider_mean_percent_variance_explained": large.mean_pve,
                "wider_minus_baseline_percent_variance_explained": pve_increase,
                "wider_favored_on_huber": large.mean_huber < current.mean_huber,
                "wider_favored_on_percent_variance_explained": (
                    large.mean_pve > current.mean_pve
                ),
            }
        )

    baseline_mean_huber = float(
        variant_aggregates[_BASELINE]["whole_node_masked_huber"]["mean"]
    )
    wider_mean_huber = float(
        variant_aggregates[_WIDER]["whole_node_masked_huber"]["mean"]
    )
    if baseline_mean_huber <= 0.0:
        raise G2WidthComparisonError(
            "baseline aggregate mean Huber must be positive"
        )
    relative_mean_huber_reduction = (
        baseline_mean_huber - wider_mean_huber
    ) / baseline_mean_huber
    baseline_mean_pve = float(
        variant_aggregates[_BASELINE][
            "whole_node_masked_percent_variance_explained"
        ]["mean"]
    )
    wider_mean_pve = float(
        variant_aggregates[_WIDER][
            "whole_node_masked_percent_variance_explained"
        ]["mean"]
    )
    criteria = {
        "relative_mean_huber_reduction_at_least_2_percent": (
            relative_mean_huber_reduction >= 0.02
        ),
        "wider_mean_percent_variance_explained_is_higher": (
            wider_mean_pve > baseline_mean_pve
        ),
        "all_seeds_favor_wider_on_huber": all(
            bool(row["wider_favored_on_huber"]) for row in paired
        ),
        "all_seeds_favor_wider_on_percent_variance_explained": all(
            bool(row["wider_favored_on_percent_variance_explained"])
            for row in paired
        ),
    }
    h1_passes = all(criteria.values())
    wider_seed_pves = {
        str(seed): grouped[_WIDER][seed].mean_pve for seed in _EXPECTED_SEEDS
    }
    jacobian_eligible = all(value > 95.0 for value in wider_seed_pves.values())
    jacobian_status = "eligible" if jacobian_eligible else "skipped"
    protocol_issues: list[str] = []
    if pilot_result["status"] == "not_supplied":
        protocol_issues.append("required_resource_pilot_not_supplied")
    elif pilot_result["status"] != "passed":
        protocol_issues.append("required_resource_pilot_failed")
    if expected_failed_run_count is None:
        protocol_issues.append("failed_run_ledger_count_not_reconciled")
    if protocol_issues:
        top_level_status = "protocol_incomplete"
    else:
        top_level_status = (
            "jacobian_required" if jacobian_eligible else "complete"
        )
    jacobian_reason = (
        "Every wider-G2 seed has mean whole-node masked percent variance "
        "explained strictly above 95%; the separate Jacobian comparison is "
        "required before this campaign analysis is complete."
        if jacobian_eligible
        else (
            "At least one wider-G2 seed is not strictly above 95% mean "
            "whole-node masked percent variance explained; Jacobians must "
            "not be compared."
        )
    )

    first = grouped[_BASELINE][0]
    data_identity = dict(
        zip(_DATA_IDENTITY_FIELDS, first.data_identity, strict=True)
    )
    return {
        "schema_version": 2,
        "artifact_kind": "g2_width_multiseed_comparison",
        "status": top_level_status,
        "campaign_id": _CAMPAIGN_ID,
        "scope": {
            "estimand": (
                "held-in full-core whole-node masked-expression reconstruction"
            ),
            "generalization_estimate": False,
            "percentage_metric_interpretation": (
                "100 times masked R-squared; not classification accuracy"
            ),
            "model_seeds_are_independent_biological_replicates": False,
            "technical_masks_are_independent_biological_replicates": False,
        },
        "privacy": {
            "contains_source_rows_or_predictions": False,
            "contains_direct_identifiers": False,
            "contains_failure_messages": False,
            "run_paths_exported": False,
        },
        "verified_identity": {
            "all_six_runs_match": True,
            "locked_config_sections_verified": [
                "dataset",
                "graph",
                "features",
                "masking",
                "trainer",
                "evaluation",
                "model",
            ],
            "paired_training_mask_and_edge_dropout_schedules": True,
            "graph_sha256": first.graph_sha256,
            "graph_directed_edges": first.graph_directed_edges,
            "evaluation_mask_bundle_sha256": first.mask_bundle_sha256,
            "dataset_id": data_identity["dataset_id"],
            "dataset_version": data_identity["version"],
            "dataset_fingerprint": data_identity["dataset_fingerprint"],
            "preprocessing_version": data_identity["preprocessing_version"],
            "split_id": data_identity["split_id"],
            "split_fingerprint": data_identity["split_fingerprint"],
            "fixed_epochs_per_conclusion_run": _EXPECTED_EPOCHS,
            "fold": 0,
        },
        "aggregation": {
            "model_seed_count_per_variant": len(_EXPECTED_SEEDS),
            "technical_mask_count_per_run": _EXPECTED_REPLICATES,
            "seed_standard_deviation_ddof": 1,
            "technical_mask_aggregation": "unweighted arithmetic mean per run",
        },
        "protocol_completion": {
            "complete": not protocol_issues,
            "issues": protocol_issues,
            "resource_pilot_required": True,
            "failure_ledger_expected_count_required": True,
        },
        "runs": per_run,
        "variant_aggregates": variant_aggregates,
        "paired_seed_deltas": {
            "direction_convention": {
                "huber": "baseline minus wider; positive favors wider",
                "relative_huber": (
                    "(baseline minus wider) / baseline; positive favors wider"
                ),
                "percent_variance_explained": (
                    "wider minus baseline; positive favors wider"
                ),
            },
            "per_seed": paired,
            "aggregates": {
                "baseline_minus_wider_huber": _statistics(
                    [
                        float(row["baseline_minus_wider_huber"])
                        for row in paired
                    ]
                ),
                "relative_huber_reduction": _statistics(
                    [
                        float(row["relative_huber_reduction"])
                        for row in paired
                    ]
                ),
                "wider_minus_baseline_percent_variance_explained": _statistics(
                    [
                        float(
                            row[
                                "wider_minus_baseline_"
                                "percent_variance_explained"
                            ]
                        )
                        for row in paired
                    ]
                ),
            },
        },
        "h1_width_gate": {
            "thresholds": {
                "minimum_relative_mean_huber_reduction": 0.02,
                "mean_percent_variance_explained_direction": "strictly_higher",
                "required_favorable_seeds_on_both_metrics": 3,
            },
            "statistic_definition": (
                "(mean baseline Huber across seeds - mean wider Huber across "
                "seeds) / mean baseline Huber across seeds; this is the ratio "
                "of aggregate means, not the mean of per-seed ratios"
            ),
            "observed": {
                "baseline_aggregate_mean_huber": baseline_mean_huber,
                "wider_aggregate_mean_huber": wider_mean_huber,
                "relative_mean_huber_reduction": (
                    relative_mean_huber_reduction
                ),
                "baseline_mean_percent_variance_explained": baseline_mean_pve,
                "wider_mean_percent_variance_explained": wider_mean_pve,
            },
            "criteria": criteria,
            "passes": h1_passes,
        },
        "resource_pilot": pilot_result,
        "failed_runs": {
            "count": len(failure_rows),
            "expected_count": expected_failed_run_count,
            "count_reconciled": expected_failed_run_count is not None,
            "runs": failure_rows,
            "failure_messages_exported": False,
        },
        "jacobian_gate": {
            "metric": (
                "per-seed mean whole-node "
                "masked_percent_variance_explained"
            ),
            "operator": "strictly_greater_than",
            "threshold": 95.0,
            "wider_seed_values": wider_seed_pves,
            "eligible": jacobian_eligible,
            "status": jacobian_status,
            "jacobians_computed": False,
            "reason": jacobian_reason,
        },
    }


def _format_number(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _markdown_report(result: Mapping[str, Any]) -> str:
    aggregates = _mapping(
        result["variant_aggregates"], "result.variant_aggregates"
    )
    h1 = _mapping(result["h1_width_gate"], "result.h1_width_gate")
    jacobian = _mapping(result["jacobian_gate"], "result.jacobian_gate")
    pilot = _mapping(result["resource_pilot"], "result.resource_pilot")
    failures = _mapping(result["failed_runs"], "result.failed_runs")
    protocol = _mapping(
        result["protocol_completion"], "result.protocol_completion"
    )
    lines = [
        "# Full-core G2 width comparison",
        "",
        (
            "Six checksum-verified, fixed-200-epoch held-in runs were compared "
            "(two widths × seeds 0, 1, 2). Percent variance explained is "
            "`100 × masked R²`, not classification accuracy or a "
            "generalization estimate."
        ),
        "",
        (
            "Protocol completion: "
            f"**{'COMPLETE' if protocol['complete'] else 'INCOMPLETE'}**"
            + (
                "."
                if protocol["complete"]
                else f" ({', '.join(str(item) for item in protocol['issues'])})."
            )
        ),
        "",
        "## Per-seed evidence",
        "",
        (
            "| Variant | Seed | Attempt | Mean Huber | Mean PVE (%) | "
            "Total (s) | Summed epochs (s) | Peak VRAM (GiB) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in result["runs"]:
        lines.append(
            "| {variant} | {seed} | {attempt} | {huber} | {pve} | "
            "{total} | {training} | {vram} |".format(
                variant=run["variant_label"],
                seed=run["seed"],
                attempt=run["attempt"],
                huber=_format_number(
                    float(run["mean_whole_node_masked_huber"])
                ),
                pve=_format_number(
                    float(
                        run[
                            "mean_whole_node_"
                            "masked_percent_variance_explained"
                        ]
                    ),
                    3,
                ),
                total=_format_number(float(run["total_duration_seconds"]), 2),
                training=_format_number(
                    float(run["summed_epoch_duration_seconds"]), 2
                ),
                vram=_format_number(float(run["peak_vram_gb"]), 3),
            )
        )
    lines.extend(
        [
            "",
            "## Variant aggregates",
            "",
            (
                "| Variant | Huber mean ± SD | PVE mean ± SD (%) | "
                "Total mean ± SD (s) | Recorded training mean ± SD (s) | "
                "Summed epochs mean ± SD (s) | Peak VRAM mean ± SD (GiB) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in _CONCLUSION_VARIANTS:
        huber = aggregates[variant]["whole_node_masked_huber"]
        pve = aggregates[variant][
            "whole_node_masked_percent_variance_explained"
        ]
        total = aggregates[variant]["resources"]["total_duration_seconds"]
        recorded = aggregates[variant]["resources"][
            "recorded_training_duration_seconds"
        ]
        summed = aggregates[variant]["resources"][
            "summed_training_duration_seconds"
        ]
        vram = aggregates[variant]["resources"]["peak_vram_gb"]
        lines.append(
            "| {variant} | {hm} ± {hs} | {pm} ± {ps} | "
            "{tm} ± {ts} | {rm} ± {rs} | {em} ± {es} | "
            "{vm} ± {vs} |".format(
                variant=variant,
                hm=_format_number(float(huber["mean"])),
                hs=_format_number(float(huber["std"])),
                pm=_format_number(float(pve["mean"]), 3),
                ps=_format_number(float(pve["std"]), 3),
                tm=_format_number(float(total["mean"]), 2),
                ts=_format_number(float(total["std"]), 2),
                rm=_format_number(float(recorded["mean"]), 2),
                rs=_format_number(float(recorded["std"]), 2),
                em=_format_number(float(summed["mean"]), 2),
                es=_format_number(float(summed["std"]), 2),
                vm=_format_number(float(vram["mean"]), 3),
                vs=_format_number(float(vram["std"]), 3),
            )
        )
    lines.extend(
        [
            "",
            "### Per-run convergence",
            "",
            (
                "| Variant | Seed | Final train loss | Minimum train loss | "
                "Last-20 slope |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for run in result["runs"]:
        lines.append(
            "| {variant} | {seed} | {final} | {minimum} | {slope} |".format(
                variant=run["variant_label"],
                seed=run["seed"],
                final=_format_number(float(run["final_train_loss"])),
                minimum=_format_number(float(run["minimum_train_loss"])),
                slope=_format_number(
                    float(run["last_20_epoch_loss_slope"]), 9
                ),
            )
        )
    lines.extend(
        [
            "",
            (
                "| Variant | Final train loss mean ± SD | "
                "Minimum loss mean ± SD | Last-20 slope mean ± SD |"
            ),
            "|---|---:|---:|---:|",
        ]
    )
    for variant in _CONCLUSION_VARIANTS:
        convergence = aggregates[variant]["convergence"]
        final_loss = convergence["final_train_loss"]
        minimum = convergence["minimum_train_loss"]
        slope = convergence["last_20_epoch_loss_slope"]
        lines.append(
            "| {variant} | {fm} ± {fs} | {mm} ± {ms} | "
            "{sm} ± {ss} |".format(
                variant=variant,
                fm=_format_number(float(final_loss["mean"])),
                fs=_format_number(float(final_loss["std"])),
                mm=_format_number(float(minimum["mean"])),
                ms=_format_number(float(minimum["std"])),
                sm=_format_number(float(slope["mean"]), 9),
                ss=_format_number(float(slope["std"]), 9),
            )
        )
    lines.extend(
        [
            "",
            "## Paired seed deltas",
            "",
            (
                "| Seed | Baseline − wider Huber | Relative reduction | "
                "Wider − baseline PVE (pp) | Wider favored on both |"
            ),
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["paired_seed_deltas"]["per_seed"]:
        both = bool(
            row["wider_favored_on_huber"]
            and row["wider_favored_on_percent_variance_explained"]
        )
        lines.append(
            "| {seed} | {huber} | {relative:.3f}% | {pve:.3f} | "
            "{both} |".format(
                seed=row["seed"],
                huber=_format_number(
                    float(row["baseline_minus_wider_huber"])
                ),
                relative=100.0 * float(row["relative_huber_reduction"]),
                pve=float(
                    row[
                        "wider_minus_baseline_"
                        "percent_variance_explained"
                    ]
                ),
                both="yes" if both else "no",
            )
        )
    delta_aggregates = result["paired_seed_deltas"]["aggregates"]
    huber_delta = delta_aggregates["baseline_minus_wider_huber"]
    pve_delta = delta_aggregates[
        "wider_minus_baseline_percent_variance_explained"
    ]
    lines.extend(
        [
            "",
            (
                "Across seeds, baseline-minus-wider Huber was "
                f"{float(huber_delta['mean']):.6f} ± "
                f"{float(huber_delta['std']):.6f} "
                f"(min {float(huber_delta['min']):.6f}, "
                f"max {float(huber_delta['max']):.6f}); "
                "wider-minus-baseline PVE was "
                f"{float(pve_delta['mean']):.3f} ± "
                f"{float(pve_delta['std']):.3f} percentage points "
                f"(min {float(pve_delta['min']):.3f}, "
                f"max {float(pve_delta['max']):.3f})."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## H1-width gate",
            "",
            (
                "The locked Huber statistic is `(mean baseline Huber across "
                "seeds − mean wider Huber across seeds) / mean baseline Huber "
                "across seeds`. It is the ratio of aggregate means, not the "
                "mean of the three per-seed ratios."
            ),
            "",
            (
                f"- Observed relative reduction: "
                f"{100.0 * float(h1['observed']['relative_mean_huber_reduction']):.3f}%."
            ),
            (
                f"- H1-width: **{'PASS' if h1['passes'] else 'FAIL'}**. "
                "Passing also requires higher aggregate mean PVE and every "
                "seed favoring wider G2 on both metrics."
            ),
            "",
            "## Resource pilot",
            "",
        ]
    )
    if pilot.get("supplied") is True:
        lines.extend(
            [
                (
                    f"- Run `{pilot['run_id']}` (attempt {pilot['attempt']}): "
                    f"{int(pilot['parameter_count']):,} parameters, "
                    f"{float(pilot['peak_vram_gb']):.3f} GiB peak VRAM, "
                    f"{float(pilot['recorded_training_duration_seconds']):.2f}s "
                    "recorded two-epoch training "
                    f"({float(pilot['summed_epoch_duration_seconds']):.2f}s "
                    "summed epoch compute), "
                    f"{float(pilot['total_duration_seconds']):.2f}s total."
                ),
                (
                    "- Projected 200-epoch training: "
                    f"{float(pilot['projected_200_epoch_training_seconds']) / 3600.0:.3f}h; "
                    f"resource gate **{str(pilot['status']).upper()}**."
                ),
            ]
        )
    else:
        lines.append("- No resource-pilot bundle was supplied.")
    lines.extend(["", "## Failed-attempt ledger", ""])
    if int(failures["count"]) == 0:
        lines.append("- No failed finalized bundles were supplied.")
    else:
        lines.extend(
            [
                "| Run | Variant | Seed | Fold | Attempt | Retry of | Category |",
                "|---|---|---:|---:|---:|---|---|",
            ]
        )
        for row in failures["runs"]:
            lines.append(
                "| `{run}` | {variant} | {seed} | {fold} | {attempt} | "
                "{retry} | {category} |".format(
                    run=row["run_id"],
                    variant=row["variant_label"],
                    seed=row["seed"],
                    fold=row["fold"],
                    attempt=row["attempt"],
                    retry=row["retry_of"] or "none",
                    category=row["failure_category"],
                )
            )
    lines.append(
        "- Failure count reconciliation: "
        + (
            f"confirmed at {failures['expected_count']}."
            if failures["count_reconciled"]
            else "not confirmed against an expected registry count."
        )
    )
    lines.extend(
        [
            "",
            "## Jacobian branch",
            "",
            (
                f"- Status: **{str(jacobian['status']).upper()}**. "
                f"{jacobian['reason']} Jacobians were not computed here."
            ),
            f"- Comparison artifact status: `{result['status']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_and_sync(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _renameat2(source: Path, destination: Path, flag: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise G2WidthComparisonError(
            "this platform lacks renameat2; refusing non-atomic output publish"
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
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        flag,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if flag == _RENAME_NOREPLACE and error_number in {
        errno.EEXIST,
        errno.ENOTEMPTY,
    }:
        raise G2WidthComparisonError(
            f"output already exists: {destination}"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination.as_posix(),
    )


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assert_not_protected_output(destination: Path) -> None:
    project = _BOOTSTRAP_ROOT.resolve()
    home = Path.home().resolve()
    filesystem_root = Path(destination.anchor)
    if destination == filesystem_root:
        raise G2WidthComparisonError(
            f"refusing protected filesystem root output: {destination}"
        )
    for protected in (project, home):
        if destination == protected or protected.is_relative_to(destination):
            raise G2WidthComparisonError(
                f"refusing protected root or ancestor output: {destination}"
            )


def _validate_owned_output(destination: Path) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise G2WidthComparisonError(
            f"existing output is not an owned directory: {destination}"
        )
    entries = {path.name: path for path in destination.iterdir()}
    if set(entries) != set(_OUTPUT_FILES):
        raise G2WidthComparisonError(
            "overwrite requires an owned directory containing exactly "
            f"{sorted(_OUTPUT_FILES)}"
        )
    for path in entries.values():
        if path.is_symlink() or not path.is_file():
            raise G2WidthComparisonError(
                f"owned output contains a non-regular entry: {path.name}"
            )
    owner = _load_json(destination / _OWNER_FILE)
    if dict(owner) != _OWNER_PAYLOAD:
        raise G2WidthComparisonError(
            f"output ownership marker does not match this utility: {destination}"
        )
    comparison = _load_json(destination / "comparison.json")
    if (
        comparison.get("artifact_kind")
        != "g2_width_multiseed_comparison"
        or comparison.get("campaign_id") != _CAMPAIGN_ID
    ):
        raise G2WidthComparisonError(
            f"existing comparison is not owned by this campaign: {destination}"
        )


def _remove_exact_directory(
    path: Path,
    *,
    require_complete: bool,
) -> None:
    entries = {entry.name: entry for entry in path.iterdir()}
    names = set(entries)
    if require_complete:
        allowed = names == set(_OUTPUT_FILES)
    else:
        allowed = names.issubset(_OUTPUT_FILES)
    if not allowed:
        raise G2WidthComparisonError(
            f"refusing cleanup of directory with unexpected entries: {path}"
        )
    for entry in entries.values():
        if entry.is_symlink() or not entry.is_file():
            raise G2WidthComparisonError(
                f"refusing cleanup of non-regular entry: {entry}"
            )
    for name in sorted(entries):
        entries[name].unlink()
    path.rmdir()


def write_comparison(
    result: Mapping[str, Any],
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish the owned three-file comparison directory."""

    requested = Path(output_directory)
    if requested.is_symlink():
        raise G2WidthComparisonError(
            f"refusing a symlink output directory: {requested}"
        )
    destination = requested.resolve(strict=False)
    _assert_not_protected_output(destination)
    destination_exists = destination.exists()
    if destination_exists and not overwrite:
        raise G2WidthComparisonError(f"output already exists: {destination}")
    if destination_exists:
        _validate_owned_output(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.writing-",
            dir=destination.parent,
        )
    )
    committed = False
    try:
        json_content = json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
        owner_content = json.dumps(
            _OWNER_PAYLOAD,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"
        _write_and_sync(staging / "comparison.json", json_content)
        _write_and_sync(staging / "report.md", _markdown_report(result))
        _write_and_sync(staging / _OWNER_FILE, owner_content)
        _sync_directory(staging)

        if destination_exists:
            _validate_owned_output(destination)
            _renameat2(staging, destination, _RENAME_EXCHANGE)
        else:
            _renameat2(staging, destination, _RENAME_NOREPLACE)
        committed = True
        try:
            # The parent fsync follows the namespace commit before any old
            # directory cleanup, so the new directory is durably published.
            _sync_directory(destination.parent)
        except OSError as error:
            warnings.warn(
                f"comparison was committed but parent fsync failed: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
    except BaseException as error:
        if not committed and staging.exists():
            try:
                _remove_exact_directory(staging, require_complete=False)
            except BaseException as cleanup_error:
                warnings.warn(
                    f"could not clean private staging directory: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if isinstance(error, G2WidthComparisonError):
            raise
        raise G2WidthComparisonError(
            f"comparison output could not be published atomically: {destination}"
        ) from error

    if destination_exists:
        try:
            # After RENAME_EXCHANGE, staging is the already-validated old
            # owned output.  A cleanup failure must not turn a committed new
            # result into a false publication failure.
            _remove_exact_directory(staging, require_complete=True)
        except BaseException as cleanup_error:
            warnings.warn(
                "comparison was committed; old owned output remains at "
                f"{staging} because cleanup failed: {cleanup_error}",
                RuntimeWarning,
                stacklevel=2,
            )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and compare exactly six finalized G2 width-campaign "
            "success runs. Repeat --run six times."
        )
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        type=Path,
        help="Conclusion-eligible finalized run; provide exactly six.",
    )
    parser.add_argument(
        "--failed-run",
        dest="failed_runs",
        action="append",
        default=[],
        type=Path,
        help="Optional finalized failed attempt; repeat as needed.",
    )
    parser.add_argument(
        "--resource-pilot",
        required=True,
        type=Path,
        help="Required successful two-epoch wider-G2 diagnostic bundle.",
    )
    parser.add_argument(
        "--expected-failed-run-count",
        required=True,
        type=int,
        help="Registry-audited count of finalized failed campaign runs.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output owned by this utility.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if len(arguments.runs) != 6:
        parser.error("--run must be supplied exactly six times")
    result = compare_g2_width_multiseed(
        arguments.runs,
        failed_run_directories=arguments.failed_runs,
        resource_pilot=arguments.resource_pilot,
        expected_failed_run_count=arguments.expected_failed_run_count,
    )
    output = write_comparison(
        result,
        arguments.output,
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "status": result["status"],
                "h1_width_passes": result["h1_width_gate"]["passes"],
                "jacobian_status": result["jacobian_gate"]["status"],
                "failed_runs": result["failed_runs"]["count"],
                "resource_pilot": result["resource_pilot"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
