#!/usr/bin/env python3
"""Compare the three locked full-core QKV-GAT capacity runs.

The command accepts exactly one k=1,000 graph run, one k=5,000 graph run,
and one parameter-matched cell-only run from
``cmp_20260726_full_core_qkv_large_k``.  It verifies the immutable run
bundles, identical source/environment fingerprints, and the locked held-in
protocol before calculating any comparison.

Masked percent variance explained is ``100 * masked R2``.  It is a
regression metric, not classification accuracy or a generalization estimate.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from dataclasses import dataclass
import errno
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.run_archive import (  # noqa: E402
    RunValidationError,
    verify_run_bundle,
)


_CAMPAIGN_ID = "cmp_20260726_full_core_qkv_large_k"
_PROTOCOL = "held_in_full_core_fixed_budget"
_GRAPH_EXECUTION = "full_core_exact_no_neighbor_sampling"
_CHECKPOINT_POLICY = "final_epoch_no_validation_selection"
_PRIMARY_METRIC = "fit/whole_node/masked_huber"
_EXPECTED_EPOCHS = 300
_EXPECTED_REPLICATES = 3
_EXPECTED_SEED = 0
_EXPECTED_FOLD = 0
_REPRESENTATION_GAIN_THRESHOLD_PERCENT = 2.0
_K_GAIN_THRESHOLD_PERCENT = 2.0
_DATASET_ID = "cosmx_normal_core_full_core_fit_v1"
_DATASET_VERSION = "full_core_fit_v1"
_PREPROCESSING_VERSION = "full_core_fit_v1"
_DATASET_SHA256 = (
    "a112fbb2bdf929197759c82913fc379c4df797f75676457bcd10419fe7a969d5"
)
_SPLIT_SHA256 = (
    "2c8c59fb659401cc202126cb14154f064d2e3380819aff647f06a30db354d84c"
)
_EXPECTED_GRAPHS = {
    "k1000": {
        "k": 1000,
        "sha256": (
            "23d9b09af45fca21e4f30ef04a6921765b19399eff3e3ef78e371f6b019004e6"
        ),
        "directed_edges": 21_029_944,
    },
    "k5000": {
        "k": 5000,
        "sha256": (
            "50f293972c443011a80abfbd81bf7cc7f44a35ccb39794e900b67dc4d2ca8d85"
        ),
        "directed_edges": 101_237_016,
    },
    # The self-only control constructs and audits the paired k=5,000 graph,
    # but transfers no topology or edge attributes into the model.
    "matched_self": {
        "k": 5000,
        "sha256": (
            "50f293972c443011a80abfbd81bf7cc7f44a35ccb39794e900b67dc4d2ca8d85"
        ),
        "directed_edges": 101_237_016,
    },
}
_ROLE_MODEL_CONTRACT = {
    "k1000": {
        "model_key": "qkvgat",
        "family": "edge_aware_qkv_graph_transformer",
        "uses_graph": True,
        "uses_edges": True,
        "implementation": "ReceiverChunkedEdgeAwareQKVGraphTransformer",
    },
    "k5000": {
        "model_key": "qkvgat",
        "family": "edge_aware_qkv_graph_transformer",
        "uses_graph": True,
        "uses_edges": True,
        "implementation": "ReceiverChunkedEdgeAwareQKVGraphTransformer",
    },
    "matched_self": {
        "model_key": "qkvgatmatchedself",
        "family": "qkv_parameter_matched_self_control",
        "uses_graph": False,
        "uses_edges": False,
        "implementation": "QKVParameterMatchedSelfControl",
    },
}
_PUBLIC_MASK_MODES = ("partial_gene", "whole_node", "spatial_block")
_MASK_MODE_ALIASES = {
    "partial": "partial_gene",
    "partial_gene": "partial_gene",
    "node": "whole_node",
    "whole_node": "whole_node",
    "block": "spatial_block",
    "spatial_block": "spatial_block",
}
_TABLE_SUFFIXES = (".parquet", ".jsonl", ".csv")
_COMPLETION_MARKERS = ("_SUCCESS", "_FAILED", "_PRUNED")
_FORBIDDEN_METRIC_PREFIXES = ("val/", "validation/", "test/", "external/")
_DATA_IDENTITY_FIELDS = (
    "dataset_id",
    "version",
    "dataset_fingerprint",
    "preprocessing_version",
    "split_id",
    "split_fingerprint",
)
_NODE_FEATURE_FIELDS = (
    "fit_scope",
    "node_expression",
    "node_metadata",
    "prohibited_node_inputs",
)
_MODEL_ARCHITECTURE_FIELDS = (
    "embedding_dim",
    "hidden_dim",
    "graph_layers",
    "attention_heads",
    "attention_head_dim",
    "ffn_dim",
    "decoder_dim",
    "edge_hidden_dim",
    "edge_embedding_dim",
    "edge_conditioning_mode",
    "dropout",
    "attention_dropout",
    "receiver_chunk_size",
    "max_edges_per_chunk",
    "activation_checkpointing",
    "exact_receiver_partitioning",
    "implicit_self_loops",
    "trainable_node_identifiers",
    "trainable_edge_identifiers",
)
_CONSTRUCTOR_COMMON_FIELDS = (
    "num_genes",
    "node_covariate_dim",
    "edge_attribute_dim",
    "hidden_dim",
    "attention_heads",
    "attention_head_dim",
    "graph_layers",
    "ffn_dim",
    "decoder_dim",
    "edge_hidden_dim",
    "edge_embedding_dim",
    "dropout",
    "attention_dropout",
    "edge_conditioning_mode",
)
_NUMBER_REL_TOL = 1e-9
_NUMBER_ABS_TOL = 1e-9
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class QKVLargeKComparisonError(RuntimeError):
    """Raised when an input violates the locked comparison contract."""


@dataclass(frozen=True)
class WholeNodeMetric:
    replicate: int
    entry_id: str
    mask_seed: int
    mask_checksum: str
    n_masked: int
    huber: float
    r2: float
    pve_percent: float


@dataclass(frozen=True)
class GitProvenance:
    commit: str
    dirty: bool
    dirty_fingerprint: str | None


@dataclass(frozen=True)
class EnvironmentProvenance:
    software_fingerprint: str
    reproduction_fingerprint: str


@dataclass(frozen=True)
class RunEvidence:
    role: str
    root: Path
    run_id: str
    model_name: str
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    final_metrics: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    full_core_inputs: Mapping[str, Any]
    fixed_masks: Mapping[str, Any]
    data_provenance: Mapping[str, Any]
    split_provenance: Mapping[str, Any]
    git_provenance: GitProvenance
    environment_provenance: EnvironmentProvenance
    history: tuple[Mapping[str, Any], ...]
    evaluation_rows: tuple[Mapping[str, Any], ...]
    mask_manifest: Mapping[str, Any]
    mask_identity: Mapping[tuple[str, int], tuple[Any, ...]]
    whole_node: tuple[WholeNodeMetric, ...]
    parameter_count: int
    graph_k: int
    graph_sha256: str
    graph_directed_edges: int

    @property
    def mean_huber(self) -> float:
        return statistics.fmean(item.huber for item in self.whole_node)

    @property
    def mean_r2(self) -> float:
        return statistics.fmean(item.r2 for item in self.whole_node)

    @property
    def mean_pve_percent(self) -> float:
        return statistics.fmean(item.pve_percent for item in self.whole_node)


def _model_key(value: object) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QKVLargeKComparisonError(f"{label} must be a mapping")
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise QKVLargeKComparisonError(
            f"required JSON artifact is unreadable: {path}"
        ) from error
    return _mapping(value, path.as_posix())


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise QKVLargeKComparisonError(
            f"required YAML artifact is unreadable: {path}"
        ) from error
    return _mapping(value, path.as_posix())


def _logical_table_path(root: Path, stem: str) -> Path:
    matches = [
        root / f"{stem}{suffix}"
        for suffix in _TABLE_SUFFIXES
        if (root / f"{stem}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise QKVLargeKComparisonError(
            f"{root.name} requires exactly one {stem} table; found "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def _load_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    rows.append(
                        dict(_mapping(value, f"{path}:{line_number}"))
                    )
        except (OSError, ValueError) as error:
            raise QKVLargeKComparisonError(
                f"required JSONL table is unreadable: {path}"
            ) from error
    elif path.suffix == ".csv":
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except (OSError, csv.Error) as error:
            raise QKVLargeKComparisonError(
                f"required CSV table is unreadable: {path}"
            ) from error
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except (ImportError, ModuleNotFoundError) as error:
            raise QKVLargeKComparisonError(
                f"reading Parquet requires pyarrow: {path}"
            ) from error
        try:
            rows = [
                dict(_mapping(row, f"row in {path}"))
                for row in parquet.read_table(path).to_pylist()
            ]
        except Exception as error:
            raise QKVLargeKComparisonError(
                f"required Parquet table is unreadable: {path}"
            ) from error
    else:
        raise QKVLargeKComparisonError(
            f"unsupported table format: {path}"
        )
    if not rows:
        raise QKVLargeKComparisonError(f"required table is empty: {path}")
    return rows


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise QKVLargeKComparisonError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise QKVLargeKComparisonError(
            f"{label} must be an integer"
        ) from error
    if isinstance(value, float) and not value.is_integer():
        raise QKVLargeKComparisonError(f"{label} must be an integer")
    if isinstance(value, str) and str(converted) != value.strip():
        raise QKVLargeKComparisonError(f"{label} must be an integer")
    return converted


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise QKVLargeKComparisonError(f"{label} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise QKVLargeKComparisonError(f"{label} must be finite") from error
    if not math.isfinite(converted):
        raise QKVLargeKComparisonError(f"{label} must be finite")
    return converted


def _sha256(value: Any, label: str) -> str:
    checksum = str(value)
    if len(checksum) != 64 or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise QKVLargeKComparisonError(
            f"{label} must be a lowercase SHA-256"
        )
    return checksum


def _git_commit(value: Any, label: str) -> str:
    commit = str(value)
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise QKVLargeKComparisonError(
            f"{label} must be a lowercase Git object ID"
        )
    return commit


def _load_git_provenance(root: Path) -> GitProvenance:
    record = _load_json(root / "provenance/git.json")
    commit = _git_commit(record.get("commit"), f"{root.name} git commit")
    dirty = record.get("dirty")
    if not isinstance(dirty, bool):
        raise QKVLargeKComparisonError(
            f"{root.name} git dirty flag must be boolean"
        )
    raw_fingerprint = record.get("dirty_fingerprint")
    if dirty:
        fingerprint = _sha256(
            raw_fingerprint, f"{root.name} dirty-tree fingerprint"
        )
    else:
        if raw_fingerprint is not None:
            raise QKVLargeKComparisonError(
                f"{root.name} clean git provenance has a dirty fingerprint"
            )
        fingerprint = None
    return GitProvenance(
        commit=commit,
        dirty=dirty,
        dirty_fingerprint=fingerprint,
    )


def _load_environment_provenance(root: Path) -> EnvironmentProvenance:
    path = root / "provenance/environment.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise QKVLargeKComparisonError(
            f"required environment provenance is unreadable: {path}"
        ) from error

    section = ""
    software: list[str] = []
    reproduction: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            continue
        key, separator, value = stripped.partition("=")
        if separator != "=" or key.strip() != "environment_fingerprint":
            continue
        if section == "":
            software.append(value.strip())
        elif section == "bagm_repro":
            reproduction.append(value.strip())

    if len(software) != 1 or len(reproduction) != 1:
        raise QKVLargeKComparisonError(
            f"{root.name} environment provenance must contain exactly one "
            "software and one bagm_repro fingerprint"
        )
    return EnvironmentProvenance(
        software_fingerprint=_sha256(
            software[0], f"{root.name} software environment fingerprint"
        ),
        reproduction_fingerprint=_sha256(
            reproduction[0],
            f"{root.name} reproduction environment fingerprint",
        ),
    )


def _same_number(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_NUMBER_REL_TOL,
        abs_tol=_NUMBER_ABS_TOL,
    )


def _canonical_subset(
    value: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    return {field: value.get(field) for field in fields}


def _success_verification(root: Path) -> None:
    markers = [
        marker for marker in _COMPLETION_MARKERS if (root / marker).is_file()
    ]
    if markers != ["_SUCCESS"]:
        raise QKVLargeKComparisonError(
            f"{root.name} requires exactly one _SUCCESS marker"
        )
    try:
        verification = verify_run_bundle(root)
    except (RunValidationError, OSError, ValueError) as error:
        raise QKVLargeKComparisonError(
            f"{root.name} failed immutable-bundle verification"
        ) from error
    if verification.get("status") != "success":
        raise QKVLargeKComparisonError(
            f"{root.name} is not a successful immutable bundle"
        )


def _assert_no_held_out_artifacts(
    root: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> None:
    evaluation = _mapping(
        config.get("evaluation"), f"{root.name} config.evaluation"
    )
    masking = _mapping(
        config.get("masking"), f"{root.name} config.masking"
    )
    dataset = _mapping(
        config.get("dataset"), f"{root.name} config.dataset"
    )
    if list(evaluation.get("splits", ())) != ["fit"]:
        raise QKVLargeKComparisonError(
            f"{root.name} does not declare fit as its only role"
        )
    if (
        evaluation.get("canonical_prediction_split") != "fit"
        or summary.get("canonical_prediction_split") != "fit"
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} canonical prediction role is not fit"
        )
    if (
        evaluation.get("generalization_estimate") is not False
        or summary.get("generalization_estimate") is not False
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} must declare generalization_estimate=false"
        )
    if evaluation.get("validation_or_test_selection") is not False:
        raise QKVLargeKComparisonError(
            f"{root.name} permits validation/test selection"
        )
    if dataset.get("validation_or_test_partition_present") is not False:
        raise QKVLargeKComparisonError(
            f"{root.name} does not exclude validation/test partitions"
        )
    for role in ("validation", "test"):
        if _as_int(
            masking.get(f"{role}_replicates"),
            f"{root.name} masking.{role}_replicates",
        ) != 0:
            raise QKVLargeKComparisonError(
                f"{root.name} configures {role} mask replicates"
            )

    prediction_files = [
        path.relative_to(root / "predictions").as_posix()
        for path in (root / "predictions").rglob("*")
        if path.is_file()
    ]
    non_fit_predictions = sorted(
        path for path in prediction_files if Path(path).stem != "fit"
    )
    if non_fit_predictions:
        raise QKVLargeKComparisonError(
            f"{root.name} contains held-out prediction artifacts: "
            f"{non_fit_predictions}"
        )

    metric_mappings: list[tuple[str, Mapping[str, Any]]] = [
        ("metrics/final.json", final_metrics)
    ]
    summary_metrics = summary.get("metrics")
    if summary_metrics is not None:
        metric_mappings.append(
            (
                "summary.metrics",
                _mapping(summary_metrics, f"{root.name} summary.metrics"),
            )
        )
    for source, values in metric_mappings:
        forbidden = sorted(
            str(name)
            for name in values
            if str(name).startswith(_FORBIDDEN_METRIC_PREFIXES)
        )
        if forbidden:
            raise QKVLargeKComparisonError(
                f"{root.name} contains held-out metrics in {source}: "
                f"{forbidden}"
            )

    events_path = root / "metrics/events.jsonl"
    try:
        event_rows = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError) as error:
        raise QKVLargeKComparisonError(
            f"cannot inspect metric events: {events_path}"
        ) from error
    forbidden_events = sorted(
        str(event.get("name"))
        for event in event_rows
        if str(event.get("name", "")).startswith(
            _FORBIDDEN_METRIC_PREFIXES
        )
    )
    if forbidden_events:
        raise QKVLargeKComparisonError(
            f"{root.name} contains held-out metric events: "
            f"{forbidden_events}"
        )
    for table_name, rows in (
        ("history", history),
        ("evaluation replicates", evaluation_rows),
    ):
        if any(str(row.get("split")) != "fit" for row in rows):
            raise QKVLargeKComparisonError(
                f"{root.name} {table_name} contains a non-fit role"
            )
        forbidden_columns = sorted(
            {
                str(column)
                for row in rows
                for column in row
                if str(column).startswith(_FORBIDDEN_METRIC_PREFIXES)
            }
        )
        if forbidden_columns:
            raise QKVLargeKComparisonError(
                f"{root.name} {table_name} contains held-out columns: "
                f"{forbidden_columns}"
            )


def _validate_epoch_contract(
    root: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    convergence: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
) -> None:
    trainer = _mapping(
        config.get("trainer"), f"{root.name} config.trainer"
    )
    expected_budget_values = {
        "trainer.max_epochs": trainer.get("max_epochs"),
        "summary.fixed_epoch_budget": summary.get("fixed_epoch_budget"),
        "training provenance fixed_epoch_budget": training_provenance.get(
            "fixed_epoch_budget"
        ),
    }
    for label, value in expected_budget_values.items():
        if _as_int(value, f"{root.name} {label}") != _EXPECTED_EPOCHS:
            raise QKVLargeKComparisonError(
                f"{root.name} does not confirm the fixed 300-epoch budget"
            )
    expected_final = _EXPECTED_EPOCHS - 1
    for label, value in (
        ("summary.final_epoch", summary.get("final_epoch")),
        (
            "training provenance final_epoch",
            training_provenance.get("final_epoch"),
        ),
        ("convergence.final_epoch", convergence.get("final_epoch")),
    ):
        if _as_int(value, f"{root.name} {label}") != expected_final:
            raise QKVLargeKComparisonError(
                f"{root.name} does not confirm all 300 epochs"
            )
    if (
        trainer.get("fixed_epoch_budget") is not True
        or trainer.get("early_stopping") is not False
        or trainer.get("restore_best") is not False
        or trainer.get("primary_checkpoint_role") != "last"
        or trainer.get("checkpoint_policy") != "last_only"
        or trainer.get("neighbor_sampling") is not False
        or trainer.get("graph_execution") != _GRAPH_EXECUTION
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} violates the fixed final-epoch training contract"
        )
    if (
        summary.get("checkpoint_role") != "last"
        or training_provenance.get("training_protocol") != _PROTOCOL
        or training_provenance.get("graph_execution") != _GRAPH_EXECUTION
        or training_provenance.get("checkpoint_policy")
        != _CHECKPOINT_POLICY
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} violates final-checkpoint provenance"
        )
    if (root / "checkpoints/best.ckpt").exists():
        raise QKVLargeKComparisonError(
            f"{root.name} contains a selected best checkpoint"
        )
    if not (root / "checkpoints/last.ckpt").is_file():
        raise QKVLargeKComparisonError(
            f"{root.name} has no final-epoch checkpoint"
        )

    epochs = [
        _as_int(row.get("epoch"), f"{root.name} history epoch")
        for row in history
    ]
    if epochs != list(range(_EXPECTED_EPOCHS)):
        raise QKVLargeKComparisonError(
            f"{root.name} history must contain epochs 0..299 once in order"
        )
    for row_number, row in enumerate(history):
        if row.get("training_protocol") != _PROTOCOL:
            raise QKVLargeKComparisonError(
                f"{root.name} history row {row_number} uses the wrong protocol"
            )
        for field in ("train_loss", "gradient_norm", "duration_seconds"):
            value = _finite_float(
                row.get(field),
                f"{root.name} history row {row_number} {field}",
            )
            if value < 0.0:
                raise QKVLargeKComparisonError(
                    f"{root.name} history row {row_number} {field} is negative"
                )
        diagnostic = row.get("diagnostic_loss")
        if diagnostic not in (None, ""):
            _finite_float(
                diagnostic,
                f"{root.name} history row {row_number} diagnostic_loss",
            )
        peak = _as_int(
            row.get("peak_cuda_memory_bytes"),
            f"{root.name} history row {row_number} peak CUDA memory",
        )
        if peak < 0:
            raise QKVLargeKComparisonError(
                f"{root.name} history row {row_number} has negative memory"
            )
    if (
        convergence.get("all_epochs_completed") is not True
        or convergence.get("all_losses_and_gradients_finite") is not True
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} convergence does not confirm 300 finite epochs"
        )


def _public_mask_mode(entry: Mapping[str, Any], label: str) -> str:
    spec = _mapping(entry.get("spec"), f"{label}.spec")
    raw = spec.get("mode", spec.get("label"))
    mode = _MASK_MODE_ALIASES.get(str(raw))
    if mode is None:
        raise QKVLargeKComparisonError(
            f"{label} has an unsupported mask mode {raw!r}"
        )
    return mode


def _validate_evaluation_metrics(
    root: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    fixed_masks: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    Mapping[str, Any],
    Mapping[tuple[str, int], tuple[Any, ...]],
    tuple[WholeNodeMetric, ...],
]:
    manifest = _mapping(
        fixed_masks.get("bundle_manifest"),
        f"{root.name} fixed-mask bundle manifest",
    )
    bundle_checksum = _sha256(
        manifest.get("bundle_checksum"),
        f"{root.name} fixed-mask bundle checksum",
    )
    if (
        _sha256(
            summary.get("evaluation_mask_bundle_sha256"),
            f"{root.name} summary mask bundle checksum",
        )
        != bundle_checksum
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} summary and fixed-mask bundle identities disagree"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise QKVLargeKComparisonError(
            f"{root.name} fixed-mask manifest entries must be a list"
        )
    expected_count = len(_PUBLIC_MASK_MODES) * _EXPECTED_REPLICATES
    if len(entries) != expected_count or len(rows) != expected_count:
        raise QKVLargeKComparisonError(
            f"{root.name} must contain exactly nine fixed evaluation rows"
        )

    manifest_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for index, raw_entry in enumerate(entries):
        entry = _mapping(
            raw_entry, f"{root.name} mask manifest entry {index}"
        )
        if entry.get("split") != "fit":
            raise QKVLargeKComparisonError(
                f"{root.name} mask manifest contains a non-fit entry"
            )
        mode = _public_mask_mode(
            entry, f"{root.name} mask manifest entry {index}"
        )
        replicate = _as_int(
            entry.get("replicate"),
            f"{root.name} mask manifest entry {index} replicate",
        )
        key = (mode, replicate)
        if key in manifest_by_key:
            raise QKVLargeKComparisonError(
                f"{root.name} repeats fixed mask {key}"
            )
        manifest_by_key[key] = entry
    expected_keys = {
        (mode, replicate)
        for mode in _PUBLIC_MASK_MODES
        for replicate in range(_EXPECTED_REPLICATES)
    }
    if set(manifest_by_key) != expected_keys:
        raise QKVLargeKComparisonError(
            f"{root.name} fixed-mask manifest lacks the locked 3x3 masks"
        )

    identity: dict[tuple[str, int], tuple[Any, ...]] = {}
    whole: dict[int, WholeNodeMetric] = {}
    seen: set[tuple[str, int]] = set()
    for row_number, row in enumerate(rows, start=1):
        mode = str(row.get("mask_mode"))
        if mode not in _PUBLIC_MASK_MODES:
            raise QKVLargeKComparisonError(
                f"{root.name} evaluation row {row_number} has mode {mode!r}"
            )
        replicate = _as_int(
            row.get("mask_replicate"),
            f"{root.name} evaluation row {row_number} replicate",
        )
        key = (mode, replicate)
        if key in seen or key not in expected_keys:
            raise QKVLargeKComparisonError(
                f"{root.name} repeats or adds evaluation mask {key}"
            )
        seen.add(key)
        manifest_entry = manifest_by_key[key]
        entry_id = str(row.get("mask_entry_id", ""))
        seed = _as_int(
            row.get("mask_seed"),
            f"{root.name} evaluation row {row_number} mask_seed",
        )
        checksum = _sha256(
            row.get("mask_checksum"),
            f"{root.name} evaluation row {row_number} mask checksum",
        )
        n_masked = _as_int(
            row.get("n_masked"),
            f"{root.name} evaluation row {row_number} n_masked",
        )
        summary_record = _mapping(
            manifest_entry.get("summary"),
            f"{root.name} manifest entry {key} summary",
        )
        manifest_identity = (
            str(manifest_entry.get("entry_id", "")),
            _as_int(
                manifest_entry.get("seed"),
                f"{root.name} manifest entry {key} seed",
            ),
            _sha256(
                manifest_entry.get("mask_checksum"),
                f"{root.name} manifest entry {key} checksum",
            ),
            _as_int(
                summary_record.get("n_masked_entries"),
                f"{root.name} manifest entry {key} n_masked",
            ),
        )
        row_identity = (entry_id, seed, checksum, n_masked)
        if row_identity != manifest_identity:
            raise QKVLargeKComparisonError(
                f"{root.name} evaluation row {key} disagrees with its "
                "fixed-mask manifest"
            )
        identity[key] = row_identity

        huber = _finite_float(
            row.get("masked_huber"),
            f"{root.name} evaluation row {row_number} masked_huber",
        )
        mse = _finite_float(
            row.get("masked_mse"),
            f"{root.name} evaluation row {row_number} masked_mse",
        )
        mae = _finite_float(
            row.get("masked_mae"),
            f"{root.name} evaluation row {row_number} masked_mae",
        )
        r2 = _finite_float(
            row.get("masked_r2"),
            f"{root.name} evaluation row {row_number} masked_r2",
        )
        pve = _finite_float(
            row.get("masked_percent_variance_explained"),
            f"{root.name} evaluation row {row_number} masked PVE",
        )
        if huber < 0.0 or mse < 0.0 or mae < 0.0:
            raise QKVLargeKComparisonError(
                f"{root.name} evaluation row {row_number} has negative loss"
            )
        if not _same_number(pve, 100.0 * r2):
            raise QKVLargeKComparisonError(
                f"{root.name} evaluation row {row_number} PVE is not 100*R2"
            )
        if mode == "whole_node":
            whole[replicate] = WholeNodeMetric(
                replicate=replicate,
                entry_id=entry_id,
                mask_seed=seed,
                mask_checksum=checksum,
                n_masked=n_masked,
                huber=huber,
                r2=r2,
                pve_percent=pve,
            )
    if seen != expected_keys or set(whole) != set(range(_EXPECTED_REPLICATES)):
        raise QKVLargeKComparisonError(
            f"{root.name} evaluation rows lack the locked 3x3 masks"
        )

    ordered = tuple(whole[index] for index in range(_EXPECTED_REPLICATES))
    mean_huber = statistics.fmean(item.huber for item in ordered)
    mean_r2 = statistics.fmean(item.r2 for item in ordered)
    mean_pve = statistics.fmean(item.pve_percent for item in ordered)
    final_huber = _finite_float(
        final_metrics.get(_PRIMARY_METRIC),
        f"{root.name} final whole-node Huber",
    )
    final_r2 = _finite_float(
        final_metrics.get("fit/whole_node/masked_r2"),
        f"{root.name} final whole-node R2",
    )
    final_pve = _finite_float(
        final_metrics.get(
            "fit/whole_node/masked_percent_variance_explained"
        ),
        f"{root.name} final whole-node PVE",
    )
    if (
        not _same_number(final_huber, mean_huber)
        or not _same_number(final_r2, mean_r2)
        or not _same_number(final_pve, mean_pve)
        or not _same_number(final_pve, 100.0 * final_r2)
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} final metrics disagree with replicate means"
        )
    if (
        summary.get("primary_metric_name") != _PRIMARY_METRIC
        or not _same_number(
            _finite_float(
                summary.get("primary_metric_value"),
                f"{root.name} summary primary metric",
            ),
            mean_huber,
        )
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} summary primary metric disagrees with replicates"
        )
    evaluation = _mapping(
        config.get("evaluation"), f"{root.name} config.evaluation"
    )
    if (
        _as_int(
            evaluation.get("mask_replicates_per_mode"),
            f"{root.name} configured replicate count",
        )
        != _EXPECTED_REPLICATES
        or _as_int(
            summary.get("evaluation_mask_replicates_per_mode"),
            f"{root.name} summary replicate count",
        )
        != _EXPECTED_REPLICATES
        or summary.get(
            "evaluation_metrics_include_all_configured_replicates_per_mode"
        )
        is not True
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} does not confirm all three evaluation replicates"
        )
    return manifest, identity, ordered


def _validate_run_contract(
    role: str,
    root: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
    full_core_inputs: Mapping[str, Any],
    data_provenance: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
) -> tuple[str, int, str, int]:
    role_contract = _ROLE_MODEL_CONTRACT[role]
    graph_contract = _EXPECTED_GRAPHS[role]
    model = _mapping(config.get("model"), f"{root.name} config.model")
    model_name = str(model.get("name", ""))
    if _model_key(model_name) != role_contract["model_key"]:
        raise QKVLargeKComparisonError(
            f"{root.name} is not the required {role} model"
        )
    if (
        model.get("family") != role_contract["family"]
        or model.get("uses_graph_inputs") is not role_contract["uses_graph"]
        or model.get("uses_edge_inputs") is not role_contract["uses_edges"]
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} violates the {role} model-family contract"
        )
    if summary.get("model_name") != model_name:
        raise QKVLargeKComparisonError(
            f"{root.name} summary model identity drifted"
        )
    if role == "matched_self":
        if (
            model.get("parameter_match_reference")
            != "edge_aware_qkv_graph_transformer"
            or model.get("parameter_matching_method")
            != "repurpose_qkv_and_edge_parameters_as_within_cell_routing"
        ):
            raise QKVLargeKComparisonError(
                f"{root.name} is not the locked QKV parameter-matched control"
            )

    campaign = _mapping(
        config.get("campaign"), f"{root.name} config.campaign"
    )
    experiment = _mapping(
        config.get("experiment"), f"{root.name} config.experiment"
    )
    if campaign.get("campaign_id") != _CAMPAIGN_ID:
        raise QKVLargeKComparisonError(
            f"{root.name} belongs to the wrong campaign"
        )
    if (
        experiment.get("estimand")
        != "held_in_full_core_whole_node_masked_reconstruction"
        or experiment.get("permitted_claim")
        != "one_core_transductive_representation_capacity"
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} declares the wrong estimand or permitted claim"
        )
    seed = _as_int(config.get("seed"), f"{root.name} config.seed")
    fold = _as_int(config.get("fold"), f"{root.name} config.fold")
    if seed != _EXPECTED_SEED or fold != _EXPECTED_FOLD:
        raise QKVLargeKComparisonError(
            f"{root.name} must use locked seed 0 and fold 0"
        )
    if (
        _as_int(summary.get("model_seed"), f"{root.name} summary seed")
        != seed
        or _as_int(
            training_provenance.get("model_seed"),
            f"{root.name} training provenance seed",
        )
        != seed
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} model seed drifted"
        )

    dataset = _mapping(
        config.get("dataset"), f"{root.name} config.dataset"
    )
    for field in _DATA_IDENTITY_FIELDS:
        value = dataset.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise QKVLargeKComparisonError(
                f"{root.name} is missing config.dataset.{field}"
            )
    if (
        dataset.get("dataset_id") != _DATASET_ID
        or dataset.get("version") != _DATASET_VERSION
        or dataset.get("preprocessing_version") != _PREPROCESSING_VERSION
        or dataset.get("dataset_fingerprint") != _DATASET_SHA256
        or dataset.get("split_fingerprint") != _SPLIT_SHA256
        or dataset.get("experimental_unit") != "single_spatial_core"
        or dataset.get("preprocessing_fit_scope")
        != "all_nodes_transductive"
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} does not use the locked full-core preprocessing"
        )
    if (
        data_provenance.get("dataset_fingerprint") != _DATASET_SHA256
        or data_provenance.get("preprocessing_version")
        != _PREPROCESSING_VERSION
        or split_provenance.get("split_fingerprint") != _SPLIT_SHA256
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} preprocessing provenance drifted"
        )
    preprocessing = _mapping(
        full_core_inputs.get("preprocessing_checksums"),
        f"{root.name} preprocessing checksums",
    )
    materialized = _mapping(
        full_core_inputs.get("materialized_identity_verification"),
        f"{root.name} materialized identity",
    )
    if (
        preprocessing.get("preprocessing_sha256") != _DATASET_SHA256
        or materialized.get("dataset_fingerprint") != _DATASET_SHA256
        or materialized.get("split_fingerprint") != _SPLIT_SHA256
        or full_core_inputs.get("fit_scope") != "all_nodes_transductive"
        or full_core_inputs.get("generalization_estimate") is not False
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} materialized preprocessing identity drifted"
        )

    features = _mapping(
        config.get("features"), f"{root.name} config.features"
    )
    if features.get("use_edge_features") is not role_contract["uses_edges"]:
        raise QKVLargeKComparisonError(
            f"{root.name} feature and model edge-input contracts disagree"
        )
    if role == "matched_self":
        if features.get("edge_features") not in ([], ()):
            raise QKVLargeKComparisonError(
                f"{root.name} self-only control exposes edge features"
            )
    else:
        edge_features = _mapping(
            features.get("edge_features"),
            f"{root.name} config.features.edge_features",
        )
        edge_fields = edge_features.get("fields")
        if not isinstance(edge_fields, list) or len(edge_fields) != 17:
            raise QKVLargeKComparisonError(
                f"{root.name} does not use the locked 17 edge features"
            )

    graph = _mapping(config.get("graph"), f"{root.name} config.graph")
    expected_k = int(graph_contract["k"])
    expected_sha256 = str(graph_contract["sha256"])
    expected_edges = int(graph_contract["directed_edges"])
    if (
        graph.get("kind") != "exact_spatial_knn_radius_guard"
        or _as_int(graph.get("k"), f"{root.name} graph.k") != expected_k
        or _as_int(
            graph.get("neighbor_k"), f"{root.name} graph.neighbor_k"
        )
        != expected_k
        or graph.get("symmetry") != "mutual"
        or graph.get("full_core_graph") is not True
        or graph.get("self_loops") is not False
        or _finite_float(
            graph.get("edge_dropout"), f"{root.name} graph.edge_dropout"
        )
        != 0.0
        or not _same_number(
            _finite_float(
                graph.get("radius_guard_um"),
                f"{root.name} graph.radius_guard_um",
            ),
            1200.0,
        )
        or _sha256(
            graph.get("expected_materialized_graph_sha256"),
            f"{root.name} expected graph checksum",
        )
        != expected_sha256
        or _as_int(
            graph.get("expected_directed_edges"),
            f"{root.name} expected directed edges",
        )
        != expected_edges
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} does not use the locked {role} graph contract"
        )
    graph_config_provenance = _mapping(
        full_core_inputs.get("graph_config"),
        f"{root.name} graph config provenance",
    )
    if graph_config_provenance != graph:
        raise QKVLargeKComparisonError(
            f"{root.name} graph config provenance drifted"
        )
    graph_checksums = _mapping(
        full_core_inputs.get("graph_checksums"),
        f"{root.name} materialized graph checksums",
    )
    observed_sha256 = _sha256(
        graph_checksums.get("graph_sha256"),
        f"{root.name} materialized graph checksum",
    )
    summary_sha256 = _sha256(
        summary.get("graph_sha256"), f"{root.name} summary graph checksum"
    )
    summary_edges = _as_int(
        summary.get("graph_directed_edges"),
        f"{root.name} summary graph edge count",
    )
    if (
        observed_sha256 != expected_sha256
        or summary_sha256 != expected_sha256
        or summary_edges != expected_edges
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} materialized graph identity is not the locked "
            f"k={expected_k} graph"
        )

    construction = _mapping(
        training_provenance.get("model_construction"),
        f"{root.name} model construction provenance",
    )
    if (
        construction.get("canonical_model_key")
        != role_contract["model_key"]
        or not str(construction.get("implementation_class", "")).endswith(
            "." + str(role_contract["implementation"])
        )
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} model implementation provenance drifted"
        )
    constructor = _mapping(
        construction.get("constructor_arguments"),
        f"{root.name} model constructor arguments",
    )
    for field in _CONSTRUCTOR_COMMON_FIELDS:
        if field not in constructor:
            raise QKVLargeKComparisonError(
                f"{root.name} constructor is missing {field}"
            )
    return model_name, expected_k, expected_sha256, expected_edges


def _parameter_count(
    root: Path,
    *,
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
) -> int:
    sources = {
        "summary": summary.get("parameter_count"),
        "metrics/final": final_metrics.get("resource/parameter_count"),
        "training provenance": training_provenance.get("parameter_count"),
    }
    values = {
        source: _as_int(value, f"{root.name} {source} parameter_count")
        for source, value in sources.items()
    }
    if len(set(values.values())) != 1:
        raise QKVLargeKComparisonError(
            f"{root.name} parameter-count records disagree: {values}"
        )
    result = next(iter(values.values()))
    if result <= 0:
        raise QKVLargeKComparisonError(
            f"{root.name} parameter count must be positive"
        )
    return result


def _load_run(path: str | Path, *, role: str) -> RunEvidence:
    root = Path(path).resolve(strict=False)
    _success_verification(root)
    config = _load_yaml(root / "config.resolved.yaml")
    summary = _load_json(root / "summary.json")
    final_metrics = _load_json(root / "metrics/final.json")
    convergence = _load_json(root / "diagnostics/training_convergence.json")
    training_provenance = _load_json(
        root / "provenance/full_core_training.json"
    )
    full_core_inputs = _load_json(
        root / "provenance/full_core_inputs.json"
    )
    fixed_masks = _load_json(
        root / "provenance/fixed_evaluation_masks.json"
    )
    data_provenance = _load_json(
        root / "provenance/data_fingerprints.json"
    )
    split_provenance = _load_json(
        root / "provenance/split_fingerprint.json"
    )
    git_provenance = _load_git_provenance(root)
    environment_provenance = _load_environment_provenance(root)
    history = tuple(
        _load_table(_logical_table_path(root, "metrics/history"))
    )
    evaluation_rows = tuple(
        _load_table(
            _logical_table_path(root, "metrics/evaluation_replicates")
        )
    )

    if (
        summary.get("status") != "success"
        or summary.get("training_exit_status") != "success"
        or summary.get("evaluation_protocol") != _PROTOCOL
        or summary.get("diagnostic_resource_pilot") is not False
        or summary.get("conclusion_eligible") is not True
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} is not a conclusion-bearing successful run"
        )
    run_id = str(summary.get("run_id", ""))
    if run_id != root.name:
        raise QKVLargeKComparisonError(
            f"{root.name} summary run_id does not match its path"
        )
    evaluation = _mapping(
        config.get("evaluation"), f"{root.name} config.evaluation"
    )
    if evaluation.get("protocol") != _PROTOCOL:
        raise QKVLargeKComparisonError(
            f"{root.name} uses the wrong evaluation protocol"
        )
    canonical = _mapping(
        summary.get("canonical_prediction_selection"),
        f"{root.name} canonical prediction selection",
    )
    if (
        canonical.get("split") != "fit"
        or canonical.get("mask_mode") != "whole_node"
        or _as_int(
            canonical.get("mask_replicate"),
            f"{root.name} canonical prediction replicate",
        )
        != 0
    ):
        raise QKVLargeKComparisonError(
            f"{root.name} canonical prediction is not fit/whole_node/0"
        )

    _assert_no_held_out_artifacts(
        root,
        config=config,
        summary=summary,
        final_metrics=final_metrics,
        history=history,
        evaluation_rows=evaluation_rows,
    )
    _validate_epoch_contract(
        root,
        config=config,
        summary=summary,
        history=history,
        convergence=convergence,
        training_provenance=training_provenance,
    )
    model_name, graph_k, graph_sha256, graph_directed_edges = (
        _validate_run_contract(
            role,
            root,
            config=config,
            summary=summary,
            training_provenance=training_provenance,
            full_core_inputs=full_core_inputs,
            data_provenance=data_provenance,
            split_provenance=split_provenance,
        )
    )
    mask_manifest, mask_identity, whole_node = (
        _validate_evaluation_metrics(
            root,
            config=config,
            summary=summary,
            final_metrics=final_metrics,
            fixed_masks=fixed_masks,
            rows=evaluation_rows,
        )
    )
    return RunEvidence(
        role=role,
        root=root,
        run_id=run_id,
        model_name=model_name,
        config=config,
        summary=summary,
        final_metrics=final_metrics,
        training_provenance=training_provenance,
        full_core_inputs=full_core_inputs,
        fixed_masks=fixed_masks,
        data_provenance=data_provenance,
        split_provenance=split_provenance,
        git_provenance=git_provenance,
        environment_provenance=environment_provenance,
        history=history,
        evaluation_rows=evaluation_rows,
        mask_manifest=mask_manifest,
        mask_identity=mask_identity,
        whole_node=whole_node,
        parameter_count=_parameter_count(
            root,
            summary=summary,
            final_metrics=final_metrics,
            training_provenance=training_provenance,
        ),
        graph_k=graph_k,
        graph_sha256=graph_sha256,
        graph_directed_edges=graph_directed_edges,
    )


def _epoch_mask_identity(
    run: RunEvidence, row: Mapping[str, Any], index: int
) -> tuple[Any, ...]:
    return (
        str(row.get("mask_mode", "")),
        _as_int(
            row.get("mask_seed"),
            f"{run.run_id} history row {index} mask_seed",
        ),
        _sha256(
            row.get("mask_checksum"),
            f"{run.run_id} history row {index} mask checksum",
        ),
        _as_int(
            row.get("edge_dropout_seed"),
            f"{run.run_id} history row {index} edge_dropout_seed",
        ),
        _as_int(
            row.get("n_masked_entries"),
            f"{run.run_id} history row {index} n_masked_entries",
        ),
        _as_int(
            row.get("n_target_nodes"),
            f"{run.run_id} history row {index} n_target_nodes",
        ),
    )


def _validate_cross_run_compatibility(
    runs: Mapping[str, RunEvidence],
) -> None:
    ordered = [runs["k1000"], runs["k5000"], runs["matched_self"]]
    if len({run.run_id for run in ordered}) != 3:
        raise QKVLargeKComparisonError(
            "the three roles must reference distinct run bundles"
        )
    reference = ordered[0]
    for run in ordered[1:]:
        if run.git_provenance.commit != reference.git_provenance.commit:
            raise QKVLargeKComparisonError(
                "runs differ in git commit provenance"
            )
        if (
            run.git_provenance.dirty
            != reference.git_provenance.dirty
            or run.git_provenance.dirty_fingerprint
            != reference.git_provenance.dirty_fingerprint
        ):
            raise QKVLargeKComparisonError(
                "runs differ in dirty-tree fingerprint provenance"
            )
        if (
            run.environment_provenance.software_fingerprint
            != reference.environment_provenance.software_fingerprint
        ):
            raise QKVLargeKComparisonError(
                "runs differ in software environment fingerprint provenance"
            )
        if (
            run.environment_provenance.reproduction_fingerprint
            != reference.environment_provenance.reproduction_fingerprint
        ):
            raise QKVLargeKComparisonError(
                "runs differ in reproduction environment fingerprint "
                "provenance"
            )
        for section in ("dataset", "masking", "trainer", "evaluation"):
            if run.config.get(section) != reference.config.get(section):
                raise QKVLargeKComparisonError(
                    f"runs differ in exact config.{section}"
                )
        if run.config.get("classification") != reference.config.get(
            "classification"
        ):
            raise QKVLargeKComparisonError(
                "runs differ in classification metadata"
            )
        left_features = _mapping(
            reference.config.get("features"),
            f"{reference.run_id} features",
        )
        right_features = _mapping(
            run.config.get("features"), f"{run.run_id} features"
        )
        if _canonical_subset(
            left_features, _NODE_FEATURE_FIELDS
        ) != _canonical_subset(right_features, _NODE_FEATURE_FIELDS):
            raise QKVLargeKComparisonError(
                "runs differ in their exact node-feature contract"
            )
        if run.data_provenance != reference.data_provenance:
            raise QKVLargeKComparisonError(
                "runs differ in data preprocessing provenance"
            )
        if run.split_provenance != reference.split_provenance:
            raise QKVLargeKComparisonError(
                "runs differ in split provenance"
            )
        if run.full_core_inputs.get(
            "preprocessing_checksums"
        ) != reference.full_core_inputs.get("preprocessing_checksums"):
            raise QKVLargeKComparisonError(
                "runs differ in materialized preprocessing checksums"
            )
        if run.full_core_inputs.get(
            "materialized_identity_verification"
        ) != reference.full_core_inputs.get(
            "materialized_identity_verification"
        ):
            raise QKVLargeKComparisonError(
                "runs differ in materialized data identity"
            )
        if run.mask_manifest != reference.mask_manifest:
            raise QKVLargeKComparisonError(
                "runs differ in fixed evaluation mask bundle identity"
            )
        if run.mask_identity != reference.mask_identity:
            raise QKVLargeKComparisonError(
                "runs differ in fixed evaluation row identity"
            )
        for index, (left_row, right_row) in enumerate(
            zip(reference.history, run.history, strict=True)
        ):
            if _epoch_mask_identity(
                reference, left_row, index
            ) != _epoch_mask_identity(run, right_row, index):
                raise QKVLargeKComparisonError(
                    f"runs differ in epoch-mask identity at epoch {index}"
                )

    k1000_model = _mapping(
        runs["k1000"].config.get("model"), "k1000 config.model"
    )
    k5000_model = _mapping(
        runs["k5000"].config.get("model"), "k5000 config.model"
    )
    if k1000_model != k5000_model:
        raise QKVLargeKComparisonError(
            "k1000 and k5000 do not use the exact same QKV model config"
        )
    for run in ordered:
        model = _mapping(
            run.config.get("model"), f"{run.run_id} config.model"
        )
        for field in _MODEL_ARCHITECTURE_FIELDS:
            if field not in model:
                raise QKVLargeKComparisonError(
                    f"{run.run_id} model config is missing {field}"
                )
        if _canonical_subset(
            model, _MODEL_ARCHITECTURE_FIELDS
        ) != _canonical_subset(k1000_model, _MODEL_ARCHITECTURE_FIELDS):
            raise QKVLargeKComparisonError(
                "graph and matched-self runs differ in QKV architecture"
            )
        construction = _mapping(
            run.training_provenance.get("model_construction"),
            f"{run.run_id} model construction",
        )
        arguments = _mapping(
            construction.get("constructor_arguments"),
            f"{run.run_id} constructor arguments",
        )
        reference_construction = _mapping(
            reference.training_provenance.get("model_construction"),
            f"{reference.run_id} model construction",
        )
        reference_arguments = _mapping(
            reference_construction.get("constructor_arguments"),
            f"{reference.run_id} constructor arguments",
        )
        if _canonical_subset(
            arguments, _CONSTRUCTOR_COMMON_FIELDS
        ) != _canonical_subset(
            reference_arguments, _CONSTRUCTOR_COMMON_FIELDS
        ):
            raise QKVLargeKComparisonError(
                "runs differ in materialized QKV constructor dimensions"
            )

    counts = {run.role: run.parameter_count for run in ordered}
    if len(set(counts.values())) != 1:
        raise QKVLargeKComparisonError(
            f"parameter counts differ across QKV runs: {counts}"
        )


def _paired_comparison(
    candidate: RunEvidence,
    reference: RunEvidence,
    *,
    name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replicate_rows: list[dict[str, Any]] = []
    for candidate_item, reference_item in zip(
        candidate.whole_node, reference.whole_node, strict=True
    ):
        if (
            candidate_item.replicate != reference_item.replicate
            or candidate_item.entry_id != reference_item.entry_id
            or candidate_item.mask_seed != reference_item.mask_seed
            or candidate_item.mask_checksum != reference_item.mask_checksum
            or candidate_item.n_masked != reference_item.n_masked
        ):
            raise QKVLargeKComparisonError(
                f"{name} whole-node evaluation masks are not paired"
            )
        if reference_item.huber <= 0.0:
            raise QKVLargeKComparisonError(
                f"{name} reference Huber must be positive for relative gain"
            )
        huber_difference = reference_item.huber - candidate_item.huber
        pve_difference = (
            candidate_item.pve_percent - reference_item.pve_percent
        )
        replicate_rows.append(
            {
                "comparison": name,
                "replicate": candidate_item.replicate,
                "mask_entry_id": candidate_item.entry_id,
                "mask_seed": candidate_item.mask_seed,
                "mask_checksum": candidate_item.mask_checksum,
                "n_masked": candidate_item.n_masked,
                "reference_role": reference.role,
                "reference_run_id": reference.run_id,
                "candidate_role": candidate.role,
                "candidate_run_id": candidate.run_id,
                "reference_huber": reference_item.huber,
                "candidate_huber": candidate_item.huber,
                "reference_minus_candidate_huber": huber_difference,
                "relative_huber_gain_percent": (
                    100.0 * huber_difference / reference_item.huber
                ),
                "reference_r2": reference_item.r2,
                "candidate_r2": candidate_item.r2,
                "reference_pve_percent": reference_item.pve_percent,
                "candidate_pve_percent": candidate_item.pve_percent,
                "candidate_minus_reference_pve_points": pve_difference,
                "huber_favors_candidate": (
                    candidate_item.huber < reference_item.huber
                ),
                "pve_favors_candidate": (
                    candidate_item.pve_percent
                    > reference_item.pve_percent
                ),
            }
        )
    if reference.mean_huber <= 0.0:
        raise QKVLargeKComparisonError(
            f"{name} reference mean Huber must be positive"
        )
    mean_huber_difference = reference.mean_huber - candidate.mean_huber
    aggregate = {
        "comparison": name,
        "reference_role": reference.role,
        "reference_run_id": reference.run_id,
        "candidate_role": candidate.role,
        "candidate_run_id": candidate.run_id,
        "reference_mean_huber": reference.mean_huber,
        "candidate_mean_huber": candidate.mean_huber,
        "mean_huber_difference_reference_minus_candidate": (
            mean_huber_difference
        ),
        "relative_mean_huber_gain_percent": (
            100.0 * mean_huber_difference / reference.mean_huber
        ),
        "reference_mean_r2": reference.mean_r2,
        "candidate_mean_r2": candidate.mean_r2,
        "reference_mean_pve_percent": reference.mean_pve_percent,
        "candidate_mean_pve_percent": candidate.mean_pve_percent,
        "mean_pve_difference_candidate_minus_reference_points": (
            candidate.mean_pve_percent - reference.mean_pve_percent
        ),
        "all_three_huber_replicates_favor_candidate": all(
            bool(row["huber_favors_candidate"]) for row in replicate_rows
        ),
        "all_three_pve_replicates_favor_candidate": all(
            bool(row["pve_favors_candidate"]) for row in replicate_rows
        ),
    }
    return aggregate, replicate_rows


def compare_full_core_qkv_large_k(
    k1000_run: str | Path,
    k5000_run: str | Path,
    matched_self_run: str | Path,
) -> dict[str, Any]:
    """Verify and compare the three locked conclusion-bearing bundles."""

    runs = {
        "k1000": _load_run(k1000_run, role="k1000"),
        "k5000": _load_run(k5000_run, role="k5000"),
        "matched_self": _load_run(
            matched_self_run, role="matched_self"
        ),
    }
    _validate_cross_run_compatibility(runs)

    comparisons: dict[str, dict[str, Any]] = {}
    paired_rows: list[dict[str, Any]] = []
    for name, candidate_role, reference_role in (
        ("k1000_vs_matched_self", "k1000", "matched_self"),
        ("k5000_vs_matched_self", "k5000", "matched_self"),
        ("k5000_vs_k1000", "k5000", "k1000"),
    ):
        aggregate, rows = _paired_comparison(
            runs[candidate_role],
            runs[reference_role],
            name=name,
        )
        comparisons[name] = aggregate
        paired_rows.extend(rows)

    candidate_gates: dict[str, dict[str, Any]] = {}
    eligible_graph_run_ids: list[str] = []
    for role, comparison_name in (
        ("k1000", "k1000_vs_matched_self"),
        ("k5000", "k5000_vs_matched_self"),
    ):
        observed = comparisons[comparison_name]
        criteria = {
            "relative_mean_huber_gain_at_least_2_percent": (
                observed["relative_mean_huber_gain_percent"]
                >= _REPRESENTATION_GAIN_THRESHOLD_PERCENT
            ),
            "all_three_masks_favor_graph_on_huber": observed[
                "all_three_huber_replicates_favor_candidate"
            ],
            "graph_mean_masked_r2_is_positive": (
                observed["candidate_mean_r2"] > 0.0
            ),
            "all_runs_completed_300_finite_epochs": True,
        }
        passes = all(criteria.values())
        candidate_gates[role] = {
            "passes": passes,
            "graph_run_id": runs[role].run_id,
            "reference_run_id": runs["matched_self"].run_id,
            "criteria": criteria,
            "observed": {
                "relative_mean_huber_gain_percent": observed[
                    "relative_mean_huber_gain_percent"
                ],
                "all_three_huber_replicates_favor_graph": observed[
                    "all_three_huber_replicates_favor_candidate"
                ],
                "graph_mean_masked_r2": observed["candidate_mean_r2"],
                "graph_mean_pve_percent": observed[
                    "candidate_mean_pve_percent"
                ],
                "graph_minus_self_pve_points": observed[
                    "mean_pve_difference_candidate_minus_reference_points"
                ],
            },
        }
        if passes:
            eligible_graph_run_ids.append(runs[role].run_id)

    representation_passes = bool(eligible_graph_run_ids)
    representation_gate = {
        "passes": representation_passes,
        "threshold_relative_huber_gain_percent": (
            _REPRESENTATION_GAIN_THRESHOLD_PERCENT
        ),
        "eligible_graph_run_ids": eligible_graph_run_ids,
        "evaluated_graph_run_ids": [
            runs["k1000"].run_id,
            runs["k5000"].run_id,
        ],
        "matched_self_run_id": runs["matched_self"].run_id,
        "criteria": {
            "at_least_one_graph_candidate_passes_locked_gate": (
                representation_passes
            ),
            "all_three_runs_completed_300_finite_epochs": True,
        },
        "candidate_gates": candidate_gates,
    }

    k_observed = comparisons["k5000_vs_k1000"]
    k_criteria = {
        "relative_mean_huber_gain_at_least_2_percent": (
            k_observed["relative_mean_huber_gain_percent"]
            >= _K_GAIN_THRESHOLD_PERCENT
        ),
        "all_three_masks_favor_k5000_on_huber": k_observed[
            "all_three_huber_replicates_favor_candidate"
        ],
        "k5000_mean_masked_r2_is_higher": (
            k_observed["candidate_mean_r2"]
            > k_observed["reference_mean_r2"]
        ),
        "both_graph_runs_completed_300_finite_epochs": True,
    }
    k_gate = {
        "passes": all(k_criteria.values()),
        "threshold_relative_huber_gain_percent": _K_GAIN_THRESHOLD_PERCENT,
        "candidate_run_id": runs["k5000"].run_id,
        "reference_run_id": runs["k1000"].run_id,
        "criteria": k_criteria,
        "observed": {
            "relative_mean_huber_gain_percent": k_observed[
                "relative_mean_huber_gain_percent"
            ],
            "all_three_huber_replicates_favor_k5000": k_observed[
                "all_three_huber_replicates_favor_candidate"
            ],
            "k5000_minus_k1000_mean_r2": (
                k_observed["candidate_mean_r2"]
                - k_observed["reference_mean_r2"]
            ),
            "k5000_minus_k1000_mean_pve_points": k_observed[
                "mean_pve_difference_candidate_minus_reference_points"
            ],
        },
    }

    parameter_count = runs["k1000"].parameter_count
    return {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": _CAMPAIGN_ID,
        "analysis": (
            "locked_three_run_full_core_qkv_large_k_capacity_comparison"
        ),
        "evaluated_graph_run_ids": [
            runs["k1000"].run_id,
            runs["k5000"].run_id,
        ],
        "metric_interpretation": {
            "masked_r2": "coefficient of determination on masked entries",
            "masked_percent_variance_explained": "100 * masked_r2",
            "is_classification_accuracy": False,
            "relative_huber_gain_percent": (
                "100 * (reference_huber - candidate_huber) / "
                "reference_huber"
            ),
        },
        "compatibility": {
            "model_seed": _EXPECTED_SEED,
            "fold": _EXPECTED_FOLD,
            "fixed_epoch_budget": _EXPECTED_EPOCHS,
            "technical_mask_replicates": _EXPECTED_REPLICATES,
            "dataset_fingerprint": _DATASET_SHA256,
            "split_fingerprint": _SPLIT_SHA256,
            "preprocessing_version": _PREPROCESSING_VERSION,
            "evaluation_mask_bundle_sha256": runs[
                "k1000"
            ].mask_manifest["bundle_checksum"],
            "equal_parameter_count": True,
            "parameter_count": parameter_count,
            "no_validation_or_test_artifacts": True,
            "identical_source_and_environment_provenance": True,
            "git_commit": runs["k1000"].git_provenance.commit,
            "git_dirty": runs["k1000"].git_provenance.dirty,
            "dirty_tree_fingerprint": runs[
                "k1000"
            ].git_provenance.dirty_fingerprint,
            "software_environment_fingerprint": runs[
                "k1000"
            ].environment_provenance.software_fingerprint,
            "reproduction_environment_fingerprint": runs[
                "k1000"
            ].environment_provenance.reproduction_fingerprint,
        },
        "runs": {
            role: {
                "run_id": run.run_id,
                "model_name": run.model_name,
                "graph_k": run.graph_k,
                "graph_sha256": run.graph_sha256,
                "graph_directed_edges": run.graph_directed_edges,
                "parameter_count": run.parameter_count,
                "final_epoch": _EXPECTED_EPOCHS - 1,
                "whole_node_mean_masked_huber": run.mean_huber,
                "whole_node_mean_masked_r2": run.mean_r2,
                "whole_node_mean_masked_percent_variance_explained": (
                    run.mean_pve_percent
                ),
            }
            for role, run in runs.items()
        },
        "aggregate_comparisons": comparisons,
        "paired_whole_node_replicates": paired_rows,
        "representation_gate": representation_gate,
        "k_gate": k_gate,
        "limitations": [
            (
                "All cells and preprocessing statistics were fitted; this is "
                "held-in transductive reconstruction, not generalization."
            ),
            (
                "The three masks are technical repeats in one spatial core, "
                "not independent biological replicates."
            ),
            (
                "No validation, test, patient-held-out, or independent-cohort "
                "result is included."
            ),
            (
                "The k=5,000 graph represents broad regional context and "
                "cannot by itself establish direct cell-cell interaction."
            ),
            (
                "Predictive capacity and attention weights do not establish "
                "biological mechanism or causality."
            ),
        ],
        "maximum_defensible_claim": (
            "one-core transductive held-in masked-expression representation "
            "capacity under the locked QKV architecture and 300-epoch budget"
        ),
    }


def _format_number(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_report(result: Mapping[str, Any]) -> str:
    runs = _mapping(result.get("runs"), "result.runs")
    comparisons = _mapping(
        result.get("aggregate_comparisons"),
        "result.aggregate_comparisons",
    )
    representation = _mapping(
        result.get("representation_gate"), "result.representation_gate"
    )
    k_gate = _mapping(result.get("k_gate"), "result.k_gate")
    compatibility = _mapping(
        result.get("compatibility"), "result.compatibility"
    )
    lines = [
        "# Full-core QKV-GAT large-k comparison",
        "",
        (
            "This is a locked, held-in regression-capacity comparison. "
            "Percent variance explained is `100 * masked R²`; it is not "
            "classification accuracy. This is not an estimate of "
            "generalization."
        ),
        "",
        "## Runs",
        "",
        "| Role | Run ID | k | Parameters | Huber | R² | PVE (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for role in ("k1000", "k5000", "matched_self"):
        run = _mapping(runs.get(role), f"result.runs.{role}")
        lines.append(
            "| {role} | `{run_id}` | {k} | {parameters:,} | {huber} | "
            "{r2} | {pve} |".format(
                role=role,
                run_id=run["run_id"],
                k=run["graph_k"],
                parameters=int(run["parameter_count"]),
                huber=_format_number(
                    run["whole_node_mean_masked_huber"]
                ),
                r2=_format_number(run["whole_node_mean_masked_r2"]),
                pve=_format_number(
                    run[
                        "whole_node_mean_masked_percent_variance_explained"
                    ]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            (
                "| Candidate vs reference | Relative Huber gain (%) | "
                "PVE difference (points) | All 3 Huber masks favor candidate |"
            ),
            "|---|---:|---:|:---:|",
        ]
    )
    for name in (
        "k1000_vs_matched_self",
        "k5000_vs_matched_self",
        "k5000_vs_k1000",
    ):
        value = _mapping(comparisons.get(name), f"comparison {name}")
        lines.append(
            "| {name} | {gain} | {pve} | {direction} |".format(
                name=name,
                gain=_format_number(
                    value["relative_mean_huber_gain_percent"]
                ),
                pve=_format_number(
                    value[
                        "mean_pve_difference_candidate_minus_reference_points"
                    ]
                ),
                direction=(
                    "yes"
                    if value[
                        "all_three_huber_replicates_favor_candidate"
                    ]
                    else "no"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Locked gates",
            "",
            (
                "- Evaluated graph run IDs: {}."
            ).format(
                ", ".join(
                    f"`{run_id}`"
                    for run_id in result["evaluated_graph_run_ids"]
                )
            ),
            (
                "- Source/environment provenance: identical; Git commit "
                "`{}`."
            ).format(compatibility["git_commit"]),
            (
                "- Representation gate: **{}**; eligible graph run IDs: {}."
            ).format(
                "PASS" if representation["passes"] else "FAIL",
                (
                    ", ".join(
                        f"`{run_id}`"
                        for run_id in representation[
                            "eligible_graph_run_ids"
                        ]
                    )
                    or "none"
                ),
            ),
            "- k=5,000 vs k=1,000 gate: **{}**.".format(
                "PASS" if k_gate["passes"] else "FAIL"
            ),
            "",
            "A failed gate is a valid negative result; thresholds were not "
            "changed after observing these metrics.",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            (
                "Maximum defensible claim: "
                f"{result['maximum_defensible_claim']}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise QKVLargeKComparisonError(
            "paired comparison CSV requires at least one row"
        )
    columns = list(rows[0])
    if any(set(row) != set(columns) for row in rows):
        raise QKVLargeKComparisonError(
            "paired comparison rows do not share one schema"
        )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _rename_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise QKVLargeKComparisonError(
            "platform lacks renameat2; refusing a non-atomic report publish"
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
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise QKVLargeKComparisonError(
            f"output directory already exists: {destination}"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination.as_posix(),
    )


def write_comparison(
    result: Mapping[str, Any], output_dir: str | Path
) -> Path:
    """Atomically write JSON, paired CSV, and Markdown into a new directory."""

    destination = Path(output_dir).resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise QKVLargeKComparisonError(
            f"output directory already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.writing-",
            dir=destination.parent,
        )
    )
    try:
        (temporary / "comparison.json").write_text(
            json.dumps(dict(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows = result.get("paired_whole_node_replicates")
        if not isinstance(rows, list):
            raise QKVLargeKComparisonError(
                "result lacks paired whole-node replicate rows"
            )
        _write_csv(temporary / "paired_whole_node.csv", rows)
        (temporary / "report.md").write_text(
            _markdown_report(result), encoding="utf-8"
        )
        _rename_no_replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and compare the three locked full-core QKV-GAT runs."
        )
    )
    parser.add_argument("--k1000-run", required=True, type=Path)
    parser.add_argument("--k5000-run", required=True, type=Path)
    parser.add_argument("--matched-self-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare_full_core_qkv_large_k(
            args.k1000_run,
            args.k5000_run,
            args.matched_self_run,
        )
        output = write_comparison(result, args.output_dir)
    except QKVLargeKComparisonError as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
