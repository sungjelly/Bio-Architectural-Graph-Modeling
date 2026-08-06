#!/usr/bin/env python3
"""Audit and compare the locked ten-core QKV large-k campaign.

The input manifest maps the opaque aliases ``ANC-01`` through ``ANC-10`` to
three immutable run bundles: ``k1000``, ``k5000``, and ``matched_self``.
Technical whole-node mask repeats are averaged within a run.  Statistical
inference is then performed on the ten paired core-level effects, never on the
30 technical mask repeats.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from dataclasses import dataclass
import errno
import itertools
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


_CAMPAIGN_ID = "cmp_20260728_adjacent_normal_10core_qkv_large_k"
_CORE_ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
_ROLES = ("k1000", "k5000", "matched_self")
_EXPECTED_K = {"k1000": 1000, "k5000": 5000, "matched_self": 5000}
_EXPECTED_SEED = 0
_EXPECTED_EPOCHS = 300
_EXPECTED_MASK_REPLICATES = 3
_PROTOCOL = "held_in_full_core_fixed_budget"
_GRAPH_EXECUTION = "full_core_exact_no_neighbor_sampling"
_CHECKPOINT_POLICY = "final_epoch_no_validation_selection"
_PRIMARY_METRIC = "fit/whole_node/masked_huber"
_GAIN_THRESHOLD_PERCENT = 2.0
_ALPHA = 0.05
_TABLE_SUFFIXES = (".parquet", ".jsonl", ".csv")
_COMPLETION_MARKERS = ("_SUCCESS", "_FAILED", "_PRUNED")
_FORBIDDEN_METRIC_PREFIXES = ("val/", "validation/", "test/", "external/")
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
_NODE_FEATURE_FIELDS = (
    "fit_scope",
    "node_expression",
    "node_metadata",
    "prohibited_node_inputs",
)
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class MultiCoreQKVComparisonError(RuntimeError):
    """Raised when a bundle or campaign violates the locked contract."""


@dataclass(frozen=True)
class WholeNodeMetric:
    replicate: int
    entry_id: str
    seed: int
    checksum: str
    n_masked: int
    huber: float
    pve_percent: float


@dataclass(frozen=True)
class RunEvidence:
    alias: str
    role: str
    root: Path
    run_id: str
    config: Mapping[str, Any]
    summary: Mapping[str, Any]
    final_metrics: Mapping[str, Any]
    training_provenance: Mapping[str, Any]
    full_core_inputs: Mapping[str, Any]
    data_provenance: Mapping[str, Any]
    split_provenance: Mapping[str, Any]
    fixed_masks: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    whole_node: tuple[WholeNodeMetric, ...]
    parameter_count: int
    graph_sha256: str
    graph_directed_edges: int
    git_identity: tuple[str, bool, str | None]
    software_environment_fingerprint: str

    @property
    def mean_huber(self) -> float:
        return statistics.fmean(metric.huber for metric in self.whole_node)

    @property
    def mean_pve_percent(self) -> float:
        return statistics.fmean(
            metric.pve_percent for metric in self.whole_node
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MultiCoreQKVComparisonError(f"{label} must be a mapping")
    return value


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise MultiCoreQKVComparisonError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as error:
        raise MultiCoreQKVComparisonError(
            f"{label} must be an integer"
        ) from error
    if isinstance(value, float) and not value.is_integer():
        raise MultiCoreQKVComparisonError(f"{label} must be an integer")
    if isinstance(value, str) and str(converted) != value.strip():
        raise MultiCoreQKVComparisonError(f"{label} must be an integer")
    return converted


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise MultiCoreQKVComparisonError(f"{label} must be finite")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise MultiCoreQKVComparisonError(
            f"{label} must be finite"
        ) from error
    if not math.isfinite(converted):
        raise MultiCoreQKVComparisonError(f"{label} must be finite")
    return converted


def _sha256(value: Any, label: str) -> str:
    checksum = str(value)
    if len(checksum) != 64 or any(
        character not in "0123456789abcdef" for character in checksum
    ):
        raise MultiCoreQKVComparisonError(
            f"{label} must be a lowercase SHA-256"
        )
    return checksum


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise MultiCoreQKVComparisonError(
            f"required JSON artifact is unreadable: {path}"
        ) from error
    return _mapping(value, path.as_posix())


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise MultiCoreQKVComparisonError(
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
        raise MultiCoreQKVComparisonError(
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
                    if line.strip():
                        rows.append(
                            dict(
                                _mapping(
                                    json.loads(line),
                                    f"{path}:{line_number}",
                                )
                            )
                        )
        except (OSError, ValueError) as error:
            raise MultiCoreQKVComparisonError(
                f"required JSONL table is unreadable: {path}"
            ) from error
    elif path.suffix == ".csv":
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                rows = [dict(row) for row in csv.DictReader(handle)]
        except (OSError, csv.Error) as error:
            raise MultiCoreQKVComparisonError(
                f"required CSV table is unreadable: {path}"
            ) from error
    elif path.suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except (ImportError, ModuleNotFoundError) as error:
            raise MultiCoreQKVComparisonError(
                f"reading Parquet requires pyarrow: {path}"
            ) from error
        try:
            rows = [
                dict(_mapping(row, f"row in {path}"))
                for row in parquet.read_table(path).to_pylist()
            ]
        except Exception as error:
            raise MultiCoreQKVComparisonError(
                f"required Parquet table is unreadable: {path}"
            ) from error
    else:
        raise MultiCoreQKVComparisonError(
            f"unsupported table format: {path}"
        )
    if not rows:
        raise MultiCoreQKVComparisonError(f"required table is empty: {path}")
    return rows


def _canonical_subset(
    value: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    return {field: value.get(field) for field in fields}


def _model_key(value: Any) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )


def _software_fingerprint(root: Path) -> str:
    path = root / "provenance/environment.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise MultiCoreQKVComparisonError(
            f"required environment provenance is unreadable: {path}"
        ) from error
    section = ""
    values: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().lower()
            continue
        key, separator, value = stripped.partition("=")
        if (
            section == ""
            and separator == "="
            and key.strip() == "environment_fingerprint"
        ):
            values.append(value.strip())
    if len(values) != 1:
        raise MultiCoreQKVComparisonError(
            f"{root.name} must record exactly one software fingerprint"
        )
    return _sha256(values[0], f"{root.name} software fingerprint")


def _git_identity(root: Path) -> tuple[str, bool, str | None]:
    record = _load_json(root / "provenance/git.json")
    commit = str(record.get("commit", ""))
    if len(commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise MultiCoreQKVComparisonError(
            f"{root.name} has an invalid Git commit"
        )
    dirty = record.get("dirty")
    if not isinstance(dirty, bool):
        raise MultiCoreQKVComparisonError(
            f"{root.name} Git dirty flag must be boolean"
        )
    fingerprint = record.get("dirty_fingerprint")
    if dirty:
        parsed = _sha256(
            fingerprint, f"{root.name} dirty-tree fingerprint"
        )
    elif fingerprint is None:
        parsed = None
    else:
        raise MultiCoreQKVComparisonError(
            f"{root.name} clean Git record has a dirty fingerprint"
        )
    return commit, dirty, parsed


def _success_verification(root: Path) -> None:
    markers = [
        marker for marker in _COMPLETION_MARKERS if (root / marker).is_file()
    ]
    if markers != ["_SUCCESS"]:
        raise MultiCoreQKVComparisonError(
            f"{root.name} requires exactly one _SUCCESS marker"
        )
    try:
        result = verify_run_bundle(root)
    except (RunValidationError, OSError, ValueError) as error:
        raise MultiCoreQKVComparisonError(
            f"{root.name} failed immutable-bundle verification"
        ) from error
    if result.get("status") != "success":
        raise MultiCoreQKVComparisonError(
            f"{root.name} is not a successful immutable bundle"
        )


def _assert_held_in_only(
    root: Path,
    *,
    config: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> None:
    evaluation = _mapping(
        config.get("evaluation"), f"{root.name} config.evaluation"
    )
    dataset = _mapping(
        config.get("dataset"), f"{root.name} config.dataset"
    )
    masking = _mapping(
        config.get("masking"), f"{root.name} config.masking"
    )
    if (
        evaluation.get("protocol") != _PROTOCOL
        or evaluation.get("canonical_prediction_split") != "fit"
        or list(evaluation.get("splits", ())) != ["fit"]
        or evaluation.get("generalization_estimate") is not False
        or evaluation.get("validation_or_test_selection") is not False
        or dataset.get("validation_or_test_partition_present") is not False
    ):
        raise MultiCoreQKVComparisonError(
            f"{root.name} is not a held-in-only transductive run"
        )
    if (
        _as_int(
            evaluation.get("mask_replicates_per_mode"),
            f"{root.name} evaluation mask repeats",
        )
        != _EXPECTED_MASK_REPLICATES
        or _as_int(
            masking.get("fit_replicates"),
            f"{root.name} fit mask repeats",
        )
        != _EXPECTED_MASK_REPLICATES
        or len(evaluation_rows) != 3 * _EXPECTED_MASK_REPLICATES
    ):
        raise MultiCoreQKVComparisonError(
            f"{root.name} does not contain the locked 3x3 fit masks"
        )
    for split in ("validation", "test"):
        if _as_int(
            masking.get(f"{split}_replicates"),
            f"{root.name} masking.{split}_replicates",
        ) != 0:
            raise MultiCoreQKVComparisonError(
                f"{root.name} configures {split} mask repeats"
            )
    if any(str(row.get("split")) != "fit" for row in evaluation_rows):
        raise MultiCoreQKVComparisonError(
            f"{root.name} contains non-fit evaluation rows"
        )
    forbidden = [
        str(name)
        for name in final_metrics
        if str(name).startswith(_FORBIDDEN_METRIC_PREFIXES)
    ]
    prediction_files = [
        path.relative_to(root / "predictions").as_posix()
        for path in (root / "predictions").rglob("*")
        if path.is_file() and path.stem != "fit"
    ]
    if forbidden or prediction_files:
        raise MultiCoreQKVComparisonError(
            f"{root.name} contains validation/test artifacts"
        )


def _validate_completion(
    root: Path,
    *,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    training: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> None:
    trainer = _mapping(
        config.get("trainer"), f"{root.name} config.trainer"
    )
    convergence = _load_json(root / "diagnostics/training_convergence.json")
    for label, value in (
        ("trainer.max_epochs", trainer.get("max_epochs")),
        ("summary.fixed_epoch_budget", summary.get("fixed_epoch_budget")),
        (
            "training.fixed_epoch_budget",
            training.get("fixed_epoch_budget"),
        ),
    ):
        if _as_int(value, f"{root.name} {label}") != _EXPECTED_EPOCHS:
            raise MultiCoreQKVComparisonError(
                f"{root.name} did not use the locked 300-epoch budget"
            )
    for label, value in (
        ("summary.final_epoch", summary.get("final_epoch")),
        ("training.final_epoch", training.get("final_epoch")),
        ("convergence.final_epoch", convergence.get("final_epoch")),
    ):
        if _as_int(value, f"{root.name} {label}") != _EXPECTED_EPOCHS - 1:
            raise MultiCoreQKVComparisonError(
                f"{root.name} did not complete epoch 299"
            )
    if (
        trainer.get("fixed_epoch_budget") is not True
        or trainer.get("early_stopping") is not False
        or trainer.get("restore_best") is not False
        or trainer.get("primary_checkpoint_role") != "last"
        or trainer.get("checkpoint_policy") != "last_only"
        or trainer.get("neighbor_sampling") is not False
        or trainer.get("graph_execution") != _GRAPH_EXECUTION
        or training.get("training_protocol") != _PROTOCOL
        or training.get("graph_execution") != _GRAPH_EXECUTION
        or training.get("checkpoint_policy") != _CHECKPOINT_POLICY
        or summary.get("checkpoint_role") != "last"
        or convergence.get("all_epochs_completed") is not True
        or convergence.get("all_losses_and_gradients_finite") is not True
    ):
        raise MultiCoreQKVComparisonError(
            f"{root.name} violates the finite final-epoch contract"
        )
    if (
        not (root / "checkpoints/last.ckpt").is_file()
        or (root / "checkpoints/best.ckpt").exists()
    ):
        raise MultiCoreQKVComparisonError(
            f"{root.name} violates the last-only checkpoint contract"
        )
    epochs = [
        _as_int(row.get("epoch"), f"{root.name} history epoch")
        for row in history
    ]
    if epochs != list(range(_EXPECTED_EPOCHS)):
        raise MultiCoreQKVComparisonError(
            f"{root.name} history must contain epochs 0..299 exactly once"
        )
    for index, row in enumerate(history):
        for field in ("train_loss", "gradient_norm", "duration_seconds"):
            value = _finite_float(
                row.get(field), f"{root.name} history[{index}].{field}"
            )
            if value < 0.0:
                raise MultiCoreQKVComparisonError(
                    f"{root.name} history[{index}].{field} is negative"
                )
        diagnostic = row.get("diagnostic_loss")
        if diagnostic not in (None, ""):
            _finite_float(
                diagnostic,
                f"{root.name} history[{index}].diagnostic_loss",
            )


def _whole_node_metrics(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    fixed_masks: Mapping[str, Any],
) -> tuple[WholeNodeMetric, ...]:
    manifest = _mapping(
        fixed_masks.get("bundle_manifest"),
        f"{root.name} fixed mask manifest",
    )
    _sha256(
        manifest.get("bundle_checksum"),
        f"{root.name} fixed mask bundle checksum",
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 9:
        raise MultiCoreQKVComparisonError(
            f"{root.name} fixed mask manifest must contain nine entries"
        )
    selected = [
        row
        for row in rows
        if row.get("split") == "fit"
        and row.get("mask_mode") == "whole_node"
    ]
    if len(selected) != _EXPECTED_MASK_REPLICATES:
        raise MultiCoreQKVComparisonError(
            f"{root.name} requires exactly three whole-node mask repeats"
        )
    metrics: dict[int, WholeNodeMetric] = {}
    for row in selected:
        replicate = _as_int(
            row.get("mask_replicate"),
            f"{root.name} whole-node mask replicate",
        )
        if replicate in metrics or replicate not in range(3):
            raise MultiCoreQKVComparisonError(
                f"{root.name} has invalid whole-node replicate indices"
            )
        huber = _finite_float(
            row.get("masked_huber"),
            f"{root.name} whole-node masked_huber",
        )
        pve = _finite_float(
            row.get("masked_percent_variance_explained"),
            f"{root.name} whole-node PVE",
        )
        r2 = _finite_float(
            row.get("masked_r2"), f"{root.name} whole-node R2"
        )
        if huber < 0.0 or not math.isclose(
            pve, 100.0 * r2, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise MultiCoreQKVComparisonError(
                f"{root.name} has inconsistent whole-node metrics"
            )
        metrics[replicate] = WholeNodeMetric(
            replicate=replicate,
            entry_id=str(row.get("mask_entry_id", "")),
            seed=_as_int(
                row.get("mask_seed"), f"{root.name} whole-node mask seed"
            ),
            checksum=_sha256(
                row.get("mask_checksum"),
                f"{root.name} whole-node mask checksum",
            ),
            n_masked=_as_int(
                row.get("n_masked"), f"{root.name} whole-node n_masked"
            ),
            huber=huber,
            pve_percent=pve,
        )
        if (
            not metrics[replicate].entry_id
            or metrics[replicate].n_masked <= 0
        ):
            raise MultiCoreQKVComparisonError(
                f"{root.name} has an invalid whole-node mask identity"
            )
    return tuple(metrics[index] for index in range(3))


def _parameter_count(
    root: Path,
    *,
    summary: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    training: Mapping[str, Any],
) -> int:
    values = [
        _as_int(summary.get("parameter_count"), f"{root.name} summary params"),
        _as_int(
            final_metrics.get("resource/parameter_count"),
            f"{root.name} metric params",
        ),
        _as_int(
            training.get("parameter_count"), f"{root.name} training params"
        ),
    ]
    if len(set(values)) != 1 or values[0] <= 0:
        raise MultiCoreQKVComparisonError(
            f"{root.name} has inconsistent parameter counts"
        )
    return values[0]


def _validate_role_contract(
    run: RunEvidence,
) -> None:
    model = _mapping(
        run.config.get("model"), f"{run.run_id} config.model"
    )
    graph = _mapping(
        run.config.get("graph"), f"{run.run_id} config.graph"
    )
    expected_k = _EXPECTED_K[run.role]
    if (
        _as_int(graph.get("k"), f"{run.run_id} graph.k") != expected_k
        or _as_int(
            graph.get("neighbor_k"), f"{run.run_id} graph.neighbor_k"
        )
        != expected_k
        or graph.get("kind") != "exact_spatial_knn_radius_guard"
        or graph.get("full_core_graph") is not True
        or graph.get("self_loops") is not False
    ):
        raise MultiCoreQKVComparisonError(
            f"{run.run_id} violates the locked k={expected_k} graph contract"
        )
    construction = _mapping(
        run.training_provenance.get("model_construction"),
        f"{run.run_id} model construction",
    )
    model_key = _model_key(construction.get("canonical_model_key"))
    implementation = str(construction.get("implementation_class", ""))
    is_graph = run.role != "matched_self"
    expected_family = (
        "edge_aware_qkv_graph_transformer"
        if is_graph
        else "qkv_parameter_matched_self_control"
    )
    if (
        model.get("family") != expected_family
        or model.get("uses_graph_inputs") is not is_graph
        or model.get("uses_edge_inputs") is not is_graph
        or model_key
        != ("qkvgat" if is_graph else "qkvgatmatchedself")
        or not implementation.endswith(
            (
                ".ReceiverChunkedEdgeAwareQKVGraphTransformer"
                if is_graph
                else ".QKVParameterMatchedSelfControl"
            )
        )
    ):
        raise MultiCoreQKVComparisonError(
            f"{run.run_id} does not use the required QKV role"
        )
    for field in _MODEL_ARCHITECTURE_FIELDS:
        if field not in model:
            raise MultiCoreQKVComparisonError(
                f"{run.run_id} model config lacks {field}"
            )
    constructor = _mapping(
        construction.get("constructor_arguments"),
        f"{run.run_id} constructor arguments",
    )
    for field in _CONSTRUCTOR_COMMON_FIELDS:
        if field not in constructor:
            raise MultiCoreQKVComparisonError(
                f"{run.run_id} constructor lacks {field}"
            )


def _load_run(path: Path, *, alias: str, role: str) -> RunEvidence:
    root = path.resolve(strict=False)
    _success_verification(root)
    config = _load_yaml(root / "config.resolved.yaml")
    summary = _load_json(root / "summary.json")
    final_metrics = _load_json(root / "metrics/final.json")
    training = _load_json(root / "provenance/full_core_training.json")
    full_core_inputs = _load_json(
        root / "provenance/full_core_inputs.json"
    )
    data_provenance = _load_json(
        root / "provenance/data_fingerprints.json"
    )
    split_provenance = _load_json(
        root / "provenance/split_fingerprint.json"
    )
    fixed_masks = _load_json(
        root / "provenance/fixed_evaluation_masks.json"
    )
    history = tuple(
        _load_table(_logical_table_path(root, "metrics/history"))
    )
    evaluation_rows = tuple(
        _load_table(
            _logical_table_path(root, "metrics/evaluation_replicates")
        )
    )
    run_id = str(summary.get("run_id", ""))
    dataset = _mapping(
        config.get("dataset"), f"{root.name} config.dataset"
    )
    campaign = _mapping(
        config.get("campaign"), f"{root.name} config.campaign"
    )
    experiment = _mapping(
        config.get("experiment"), f"{root.name} config.experiment"
    )
    expected_prefix = alias.lower().replace("-", "") + "_"
    if (
        run_id != root.name
        or campaign.get("campaign_id") != _CAMPAIGN_ID
        or dataset.get("biological_unit_alias") != alias
        or not str(experiment.get("variant_label", "")).startswith(
            expected_prefix
        )
        or _as_int(config.get("seed"), f"{root.name} seed") != _EXPECTED_SEED
        or _as_int(
            training.get("model_seed"), f"{root.name} training seed"
        )
        != _EXPECTED_SEED
        or summary.get("status") != "success"
        or summary.get("training_exit_status") != "success"
        or summary.get("conclusion_eligible") is not True
        or summary.get("diagnostic_resource_pilot") is not False
        or summary.get("evaluation_protocol") != _PROTOCOL
        or summary.get("primary_metric_name") != _PRIMARY_METRIC
    ):
        raise MultiCoreQKVComparisonError(
            f"{root.name} violates campaign, alias, seed, or success metadata"
        )
    _assert_held_in_only(
        root,
        config=config,
        final_metrics=final_metrics,
        evaluation_rows=evaluation_rows,
    )
    _validate_completion(
        root,
        config=config,
        summary=summary,
        training=training,
        history=history,
    )
    whole = _whole_node_metrics(
        root, evaluation_rows, fixed_masks
    )
    mean_huber = statistics.fmean(metric.huber for metric in whole)
    mean_pve = statistics.fmean(metric.pve_percent for metric in whole)
    if (
        not math.isclose(
            _finite_float(
                final_metrics.get(_PRIMARY_METRIC),
                f"{root.name} final Huber",
            ),
            mean_huber,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _finite_float(
                final_metrics.get(
                    "fit/whole_node/masked_percent_variance_explained"
                ),
                f"{root.name} final PVE",
            ),
            mean_pve,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise MultiCoreQKVComparisonError(
            f"{root.name} final metrics do not equal the three-mask means"
        )
    graph_checksums = _mapping(
        full_core_inputs.get("graph_checksums"),
        f"{root.name} graph checksums",
    )
    graph_sha256 = _sha256(
        graph_checksums.get("graph_sha256"),
        f"{root.name} graph checksum",
    )
    summary_graph_sha256 = _sha256(
        summary.get("graph_sha256"), f"{root.name} summary graph checksum"
    )
    graph_edges = _as_int(
        summary.get("graph_directed_edges"),
        f"{root.name} directed edge count",
    )
    if graph_sha256 != summary_graph_sha256 or graph_edges <= 0:
        raise MultiCoreQKVComparisonError(
            f"{root.name} has inconsistent materialized graph provenance"
        )
    run = RunEvidence(
        alias=alias,
        role=role,
        root=root,
        run_id=run_id,
        config=config,
        summary=summary,
        final_metrics=final_metrics,
        training_provenance=training,
        full_core_inputs=full_core_inputs,
        data_provenance=data_provenance,
        split_provenance=split_provenance,
        fixed_masks=fixed_masks,
        history=history,
        whole_node=whole,
        parameter_count=_parameter_count(
            root,
            summary=summary,
            final_metrics=final_metrics,
            training=training,
        ),
        graph_sha256=graph_sha256,
        graph_directed_edges=graph_edges,
        git_identity=_git_identity(root),
        software_environment_fingerprint=_software_fingerprint(root),
    )
    _validate_role_contract(run)
    return run


def _epoch_mask_identity(
    run: RunEvidence, row: Mapping[str, Any], index: int
) -> tuple[Any, ...]:
    return (
        str(row.get("mask_mode", "")),
        _as_int(
            row.get("mask_seed"),
            f"{run.run_id} history[{index}] mask_seed",
        ),
        _sha256(
            row.get("mask_checksum"),
            f"{run.run_id} history[{index}] mask checksum",
        ),
        _as_int(
            row.get("edge_dropout_seed"),
            f"{run.run_id} history[{index}] edge dropout seed",
        ),
        _as_int(
            row.get("n_masked_entries"),
            f"{run.run_id} history[{index}] n_masked_entries",
        ),
        _as_int(
            row.get("n_target_nodes"),
            f"{run.run_id} history[{index}] n_target_nodes",
        ),
    )


def _whole_identity(
    metrics: Sequence[WholeNodeMetric],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            metric.replicate,
            metric.entry_id,
            metric.seed,
            metric.checksum,
            metric.n_masked,
        )
        for metric in metrics
    )


def _constructor_subset(run: RunEvidence) -> Mapping[str, Any]:
    construction = _mapping(
        run.training_provenance.get("model_construction"),
        f"{run.run_id} model construction",
    )
    arguments = _mapping(
        construction.get("constructor_arguments"),
        f"{run.run_id} constructor arguments",
    )
    return _canonical_subset(arguments, _CONSTRUCTOR_COMMON_FIELDS)


def _validate_core_pairing(runs: Mapping[str, RunEvidence]) -> None:
    ordered = [runs[role] for role in _ROLES]
    if len({run.run_id for run in ordered}) != 3:
        raise MultiCoreQKVComparisonError(
            f"{ordered[0].alias} roles do not reference distinct run IDs"
        )
    reference = runs["k1000"]
    for run in ordered[1:]:
        for section in ("dataset", "masking", "trainer", "evaluation"):
            if run.config.get(section) != reference.config.get(section):
                raise MultiCoreQKVComparisonError(
                    f"{run.alias} arms differ in config.{section}"
                )
        if run.config.get("classification") != reference.config.get(
            "classification"
        ):
            raise MultiCoreQKVComparisonError(
                f"{run.alias} arms differ in classification"
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
            raise MultiCoreQKVComparisonError(
                f"{run.alias} arms differ in node preprocessing"
            )
        if (
            run.data_provenance != reference.data_provenance
            or run.split_provenance != reference.split_provenance
            or run.full_core_inputs.get("preprocessing_checksums")
            != reference.full_core_inputs.get("preprocessing_checksums")
            or run.full_core_inputs.get(
                "materialized_identity_verification"
            )
            != reference.full_core_inputs.get(
                "materialized_identity_verification"
            )
        ):
            raise MultiCoreQKVComparisonError(
                f"{run.alias} arms differ in preprocessing identity"
            )
        if (
            run.fixed_masks.get("bundle_manifest")
            != reference.fixed_masks.get("bundle_manifest")
            or _whole_identity(run.whole_node)
            != _whole_identity(reference.whole_node)
        ):
            raise MultiCoreQKVComparisonError(
                f"{run.alias} arms do not use identical fixed masks"
            )
        for index, (left, right) in enumerate(
            zip(reference.history, run.history, strict=True)
        ):
            if _epoch_mask_identity(
                reference, left, index
            ) != _epoch_mask_identity(run, right, index):
                raise MultiCoreQKVComparisonError(
                    f"{run.alias} arms differ in epoch mask {index}"
                )
        reference_model = _mapping(
            reference.config.get("model"),
            f"{reference.run_id} config.model",
        )
        model = _mapping(
            run.config.get("model"), f"{run.run_id} config.model"
        )
        if _canonical_subset(
            model, _MODEL_ARCHITECTURE_FIELDS
        ) != _canonical_subset(
            reference_model, _MODEL_ARCHITECTURE_FIELDS
        ):
            raise MultiCoreQKVComparisonError(
                f"{run.alias} arms differ in QKV architecture"
            )
        if _constructor_subset(run) != _constructor_subset(reference):
            raise MultiCoreQKVComparisonError(
                f"{run.alias} arms differ in QKV constructor dimensions"
            )
    if runs["k1000"].config.get("model") != runs["k5000"].config.get(
        "model"
    ):
        raise MultiCoreQKVComparisonError(
            f"{reference.alias} graph arms use different model configs"
        )
    counts = {run.role: run.parameter_count for run in ordered}
    if len(set(counts.values())) != 1:
        raise MultiCoreQKVComparisonError(
            f"{reference.alias} arms are not parameter matched: {counts}"
        )
    if (
        runs["k5000"].graph_sha256
        != runs["matched_self"].graph_sha256
        or runs["k5000"].graph_directed_edges
        != runs["matched_self"].graph_directed_edges
    ):
        raise MultiCoreQKVComparisonError(
            f"{reference.alias} self control did not audit the paired k5000 graph"
        )


def _load_manifest(
    manifest_path: str | Path,
) -> tuple[Mapping[str, Mapping[str, Path]], Mapping[str, Any]]:
    path = Path(manifest_path).resolve(strict=False)
    manifest = _load_yaml(path)
    if (
        _as_int(
            manifest.get("schema_version"), "manifest.schema_version"
        )
        != 1
        or manifest.get("campaign_id") != _CAMPAIGN_ID
    ):
        raise MultiCoreQKVComparisonError(
            "manifest must use schema_version 1 and campaign_id "
            f"{_CAMPAIGN_ID}"
        )
    cores = _mapping(manifest.get("cores"), "manifest.cores")
    if set(cores) != set(_CORE_ALIASES):
        raise MultiCoreQKVComparisonError(
            "manifest must contain exactly the opaque aliases "
            + ", ".join(_CORE_ALIASES)
        )
    resolved: dict[str, dict[str, Path]] = {}
    all_paths: list[Path] = []
    for alias in _CORE_ALIASES:
        roles = _mapping(cores.get(alias), f"manifest.cores.{alias}")
        if set(roles) != set(_ROLES):
            raise MultiCoreQKVComparisonError(
                f"{alias} must map exactly {', '.join(_ROLES)}"
            )
        resolved[alias] = {}
        for role in _ROLES:
            raw = roles[role]
            if isinstance(raw, Mapping):
                raw = raw.get("run_dir")
            if not isinstance(raw, (str, os.PathLike)):
                raise MultiCoreQKVComparisonError(
                    f"{alias}.{role} must be a run path"
                )
            run_path = Path(raw)
            if not run_path.is_absolute():
                run_path = path.parent / run_path
            run_path = run_path.resolve(strict=False)
            resolved[alias][role] = run_path
            all_paths.append(run_path)
    if len(all_paths) != 30 or len(set(all_paths)) != 30:
        raise MultiCoreQKVComparisonError(
            "manifest must reference 30 distinct run directories"
        )
    return resolved, manifest


def _validate_across_cores(
    runs: Mapping[str, Mapping[str, RunEvidence]],
) -> None:
    flattened = [
        runs[alias][role] for alias in _CORE_ALIASES for role in _ROLES
    ]
    if len({run.run_id for run in flattened}) != 30:
        raise MultiCoreQKVComparisonError(
            "the campaign must contain 30 distinct immutable run IDs"
        )
    reference = runs[_CORE_ALIASES[0]]["k1000"]
    architecture = _canonical_subset(
        _mapping(reference.config.get("model"), "reference model"),
        _MODEL_ARCHITECTURE_FIELDS,
    )
    constructor = _constructor_subset(reference)
    parameter_count = reference.parameter_count
    for run in flattened:
        model = _mapping(
            run.config.get("model"), f"{run.run_id} config.model"
        )
        if (
            _canonical_subset(model, _MODEL_ARCHITECTURE_FIELDS)
            != architecture
            or _constructor_subset(run) != constructor
            or run.parameter_count != parameter_count
        ):
            raise MultiCoreQKVComparisonError(
                "the 30 runs do not share one parameter-matched QKV architecture"
            )
        if run.git_identity != reference.git_identity:
            raise MultiCoreQKVComparisonError(
                "the 30 runs differ in source-tree provenance"
            )
        if (
            run.software_environment_fingerprint
            != reference.software_environment_fingerprint
        ):
            raise MultiCoreQKVComparisonError(
                "the 30 runs differ in software environment provenance"
            )


def _core_contrast(
    candidate: RunEvidence,
    reference: RunEvidence,
    *,
    name: str,
) -> dict[str, Any]:
    if reference.mean_huber <= 0.0:
        raise MultiCoreQKVComparisonError(
            f"{candidate.alias} {name} reference Huber must be positive"
        )
    gain = (
        100.0
        * (reference.mean_huber - candidate.mean_huber)
        / reference.mean_huber
    )
    return {
        "core_alias": candidate.alias,
        "comparison": name,
        "candidate_role": candidate.role,
        "candidate_run_id": candidate.run_id,
        "reference_role": reference.role,
        "reference_run_id": reference.run_id,
        "candidate_mean_huber": candidate.mean_huber,
        "reference_mean_huber": reference.mean_huber,
        "relative_huber_gain_percent": gain,
        "candidate_mean_pve_percent": candidate.mean_pve_percent,
        "reference_mean_pve_percent": reference.mean_pve_percent,
        "candidate_minus_reference_pve_points": (
            candidate.mean_pve_percent - reference.mean_pve_percent
        ),
        "huber_favors_candidate": gain > 0.0,
        "pve_favors_candidate": (
            candidate.mean_pve_percent > reference.mean_pve_percent
        ),
    }


def _exact_sign_flip_p(values: Sequence[float]) -> float:
    if not values:
        raise MultiCoreQKVComparisonError(
            "sign-flip test requires paired core effects"
        )
    observed = statistics.fmean(values)
    extreme = 0
    total = 0
    tolerance = 1e-12 * max(1.0, abs(observed))
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = statistics.fmean(
            sign * value for sign, value in zip(signs, values, strict=True)
        )
        total += 1
        if statistic >= observed - tolerance:
            extreme += 1
    return extreme / total


def _paired_t_ci(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    if n != 10:
        raise MultiCoreQKVComparisonError(
            "paired t interval requires the locked ten core effects"
        )
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    standard_error = sd / math.sqrt(n)
    try:
        from scipy.stats import t

        critical = float(t.ppf(0.975, df=n - 1))
    except (ImportError, ModuleNotFoundError):
        # Exact 0.975 quantile of Student t with 9 degrees of freedom.
        critical = 2.2621571627409915
    return {
        "mean": mean,
        "standard_error": standard_error,
        "degrees_of_freedom": float(n - 1),
        "critical_value": critical,
        "lower": mean - critical * standard_error,
        "upper": mean + critical * standard_error,
    }


def _wilcoxon_sensitivity(values: Sequence[float]) -> dict[str, Any]:
    try:
        from scipy.stats import wilcoxon
    except (ImportError, ModuleNotFoundError):
        return {
            "available": False,
            "alternative": "greater",
            "reason": "scipy is not installed",
        }
    if all(value == 0.0 for value in values):
        return {
            "available": True,
            "alternative": "greater",
            "statistic": 0.0,
            "p_value": 1.0,
            "note": "all paired differences are zero",
        }
    try:
        result = wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
            method="auto",
        )
    except ValueError as error:
        return {
            "available": True,
            "alternative": "greater",
            "error": str(error),
        }
    return {
        "available": True,
        "alternative": "greater",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _aggregate_contrast(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_huber = [
        _finite_float(
            row.get("candidate_mean_huber"), "candidate core Huber"
        )
        for row in rows
    ]
    reference_huber = [
        _finite_float(
            row.get("reference_mean_huber"), "reference core Huber"
        )
        for row in rows
    ]
    gains = [
        _finite_float(
            row.get("relative_huber_gain_percent"),
            "core relative Huber gain",
        )
        for row in rows
    ]
    pve_differences = [
        _finite_float(
            row.get("candidate_minus_reference_pve_points"),
            "core PVE difference",
        )
        for row in rows
    ]
    candidate_pve = [
        _finite_float(
            row.get("candidate_mean_pve_percent"), "candidate core PVE"
        )
        for row in rows
    ]
    reference_pve = [
        _finite_float(
            row.get("reference_mean_pve_percent"), "reference core PVE"
        )
        for row in rows
    ]
    if len(gains) != 10:
        raise MultiCoreQKVComparisonError(
            "aggregate contrast requires exactly ten core rows"
        )
    leave_one_out = [
        {
            "omitted_core_alias": str(rows[index]["core_alias"]),
            "mean_relative_huber_gain_percent": statistics.fmean(
                value for position, value in enumerate(gains)
                if position != index
            ),
        }
        for index in range(len(gains))
    ]
    loo_values = [
        float(row["mean_relative_huber_gain_percent"])
        for row in leave_one_out
    ]
    return {
        "comparison": str(rows[0]["comparison"]),
        "n_independent_core_units": 10,
        "technical_masks_averaged_per_core": 3,
        "core_level_values": [dict(row) for row in rows],
        "whole_node_masked_huber": {
            "candidate_core_mean": statistics.fmean(candidate_huber),
            "candidate_core_median": statistics.median(candidate_huber),
            "candidate_core_sample_sd": statistics.stdev(candidate_huber),
            "reference_core_mean": statistics.fmean(reference_huber),
            "reference_core_median": statistics.median(reference_huber),
            "reference_core_sample_sd": statistics.stdev(reference_huber),
        },
        "relative_huber_gain_percent": {
            "mean": statistics.fmean(gains),
            "median": statistics.median(gains),
            "sample_sd": statistics.stdev(gains),
            "minimum": min(gains),
            "maximum": max(gains),
            "positive_core_count": sum(value > 0.0 for value in gains),
            "paired_t_95_percent_ci": _paired_t_ci(gains),
            "exact_sign_flip_one_sided_p": _exact_sign_flip_p(gains),
            "wilcoxon_one_sided_sensitivity": _wilcoxon_sensitivity(gains),
            "leave_one_core_out_means": leave_one_out,
            "leave_one_core_out_mean_range": {
                "minimum": min(loo_values),
                "maximum": max(loo_values),
            },
        },
        "pve_percent": {
            "candidate_core_mean": statistics.fmean(candidate_pve),
            "candidate_core_median": statistics.median(candidate_pve),
            "candidate_core_sample_sd": statistics.stdev(candidate_pve),
            "reference_core_mean": statistics.fmean(reference_pve),
            "reference_core_median": statistics.median(reference_pve),
            "reference_core_sample_sd": statistics.stdev(reference_pve),
            "candidate_minus_reference_mean_points": statistics.fmean(
                pve_differences
            ),
            "candidate_minus_reference_median_points": statistics.median(
                pve_differences
            ),
            "candidate_minus_reference_sample_sd_points": statistics.stdev(
                pve_differences
            ),
            "positive_difference_core_count": sum(
                value > 0.0 for value in pve_differences
            ),
        },
    }


def _holm_adjust(
    raw_p_values: Mapping[str, float],
) -> dict[str, float]:
    ordered = sorted(raw_p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def compare_multicore_qkv_large_k(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify 30 immutable bundles and compute locked core-level contrasts."""

    paths, _manifest = _load_manifest(manifest_path)
    runs: dict[str, dict[str, RunEvidence]] = {}
    for alias in _CORE_ALIASES:
        runs[alias] = {
            role: _load_run(paths[alias][role], alias=alias, role=role)
            for role in _ROLES
        }
        _validate_core_pairing(runs[alias])
    _validate_across_cores(runs)

    core_results: dict[str, Any] = {}
    contrast_rows: dict[str, list[dict[str, Any]]] = {
        "k5000_vs_matched_self": [],
        "k5000_vs_k1000": [],
    }
    for alias in _CORE_ALIASES:
        per_core = runs[alias]
        representation = _core_contrast(
            per_core["k5000"],
            per_core["matched_self"],
            name="k5000_vs_matched_self",
        )
        large_k = _core_contrast(
            per_core["k5000"],
            per_core["k1000"],
            name="k5000_vs_k1000",
        )
        contrast_rows["k5000_vs_matched_self"].append(representation)
        contrast_rows["k5000_vs_k1000"].append(large_k)
        core_results[alias] = {
            "runs": {
                role: {
                    "run_id": run.run_id,
                    "graph_k": _EXPECTED_K[role],
                    "graph_sha256": run.graph_sha256,
                    "graph_directed_edges": run.graph_directed_edges,
                    "parameter_count": run.parameter_count,
                    "whole_node_mean_masked_huber": run.mean_huber,
                    "whole_node_mean_masked_percent_variance_explained": (
                        run.mean_pve_percent
                    ),
                }
                for role, run in per_core.items()
            },
            "contrasts": {
                "k5000_vs_matched_self": representation,
                "k5000_vs_k1000": large_k,
            },
        }

    aggregates = {
        name: _aggregate_contrast(rows)
        for name, rows in contrast_rows.items()
    }
    raw_p = {
        name: float(
            aggregate["relative_huber_gain_percent"][
                "exact_sign_flip_one_sided_p"
            ]
        )
        for name, aggregate in aggregates.items()
    }
    adjusted_p = _holm_adjust(raw_p)
    for name, aggregate in aggregates.items():
        aggregate["confirmatory_inference"] = {
            "raw_exact_sign_flip_one_sided_p": raw_p[name],
            "holm_adjusted_exact_sign_flip_one_sided_p": adjusted_p[name],
            "family": [
                "k5000_vs_matched_self",
                "k5000_vs_k1000",
            ],
            "familywise_alpha": _ALPHA,
        }

    representation_aggregate = aggregates["k5000_vs_matched_self"]
    large_k_aggregate = aggregates["k5000_vs_k1000"]

    def gate(
        aggregate: Mapping[str, Any],
        *,
        include_representation_pve: bool,
    ) -> dict[str, Any]:
        gain = _mapping(
            aggregate.get("relative_huber_gain_percent"),
            "aggregate relative gain",
        )
        inference = _mapping(
            aggregate.get("confirmatory_inference"),
            "aggregate confirmatory inference",
        )
        ci = _mapping(
            gain.get("paired_t_95_percent_ci"), "paired t interval"
        )
        criteria: dict[str, bool] = {
            "mean_relative_huber_gain_at_least_2_percent": (
                float(gain["mean"]) >= _GAIN_THRESHOLD_PERCENT
            ),
            "paired_t_95_percent_ci_lower_above_zero": (
                float(ci["lower"]) > 0.0
            ),
            "holm_adjusted_one_sided_exact_p_below_0_05": (
                float(
                    inference[
                        "holm_adjusted_exact_sign_flip_one_sided_p"
                    ]
                )
                < _ALPHA
            ),
            "at_least_9_of_10_cores_have_positive_gain": (
                int(gain["positive_core_count"]) >= 9
            ),
            "all_30_runs_completed_300_finite_epochs": True,
        }
        if include_representation_pve:
            pve = _mapping(
                aggregate.get("pve_percent"), "aggregate PVE"
            )
            criteria.update(
                {
                    "k5000_core_mean_pve_is_positive": (
                        float(pve["candidate_core_mean"]) > 0.0
                    ),
                    "k5000_minus_self_mean_pve_points_is_positive": (
                        float(
                            pve[
                                "candidate_minus_reference_mean_points"
                            ]
                        )
                        > 0.0
                    ),
                }
            )
        return {
            "passes": all(criteria.values()),
            "threshold_mean_relative_huber_gain_percent": (
                _GAIN_THRESHOLD_PERCENT
            ),
            "criteria": criteria,
        }

    reference_run = runs[_CORE_ALIASES[0]]["k1000"]
    return {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": _CAMPAIGN_ID,
        "analysis": "locked_ten_core_qkv_large_k_capacity_comparison",
        "metric_interpretation": {
            "primary": "whole-node masked Huber averaged over 3 masks per core",
            "secondary": "percent variance explained = 100 * masked R2",
            "is_classification_accuracy": False,
            "relative_huber_gain_percent": (
                "100 * (reference_core_mean_huber - "
                "candidate_core_mean_huber) / reference_core_mean_huber"
            ),
            "inferential_unit": "one opaque adjacent-normal core alias",
            "technical_masks_are_independent_units": False,
        },
        "compatibility": {
            "core_aliases": list(_CORE_ALIASES),
            "independent_core_unit_count": 10,
            "run_count": 30,
            "arms_per_core": list(_ROLES),
            "model_seed": _EXPECTED_SEED,
            "fixed_epoch_budget": _EXPECTED_EPOCHS,
            "technical_mask_replicates_per_run": _EXPECTED_MASK_REPLICATES,
            "equal_parameter_count": True,
            "parameter_count": reference_run.parameter_count,
            "identical_model_architecture_across_30_runs": True,
            "paired_preprocessing_and_masks_within_core": True,
            "no_validation_or_test_artifacts": True,
            "git_commit": reference_run.git_identity[0],
            "git_dirty": reference_run.git_identity[1],
            "dirty_tree_fingerprint": reference_run.git_identity[2],
            "software_environment_fingerprint": (
                reference_run.software_environment_fingerprint
            ),
        },
        "cores": core_results,
        "aggregate_contrasts": aggregates,
        "representation_gate": gate(
            representation_aggregate,
            include_representation_pve=True,
        ),
        "large_k_gate": gate(
            large_k_aggregate,
            include_representation_pve=False,
        ),
        "limitations": [
            (
                "ANC aliases denote pathology-adjacent normal tissue, not "
                "ten healthy or independently documented true-Normal cores."
            ),
            (
                "All cells and preprocessing statistics are fitted in each "
                "core; this is held-in transductive reconstruction, not "
                "patient-held-out or prospective generalization."
            ),
            (
                "The three masks are technical repeats averaged within each "
                "core; they are not biological replicates and do not make "
                "the inferential sample size 30."
            ),
            (
                "Large-k graphs mix broad regional context with local "
                "neighborhood information and cannot identify direct "
                "cell-cell interactions."
            ),
            (
                "Predictive gain, PVE, and attention routing do not establish "
                "a biological mechanism or causal influence."
            ),
        ],
        "maximum_defensible_claim": (
            "paired ten-core adjacent-normal, held-in transductive "
            "masked-expression capacity under the locked QKV architecture"
        ),
    }


def _format(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_report(result: Mapping[str, Any]) -> str:
    aggregates = _mapping(
        result.get("aggregate_contrasts"), "aggregate contrasts"
    )
    representation = _mapping(
        aggregates.get("k5000_vs_matched_self"),
        "representation contrast",
    )
    large_k = _mapping(
        aggregates.get("k5000_vs_k1000"), "large-k contrast"
    )
    lines = [
        "# Ten-core QKV large-k comparison",
        "",
        (
            "Whole-node masked Huber is primary. PVE (%) is `100 * masked "
            "R²`, not classification accuracy. Three technical masks are "
            "averaged within each core; inference uses 10 core-level pairs."
        ),
        "",
        "## Core-level results",
        "",
        (
            "| Alias | k5000 Huber | self Huber | k1000 Huber | "
            "k5000 vs self gain (%) | k5000 vs k1000 gain (%) | "
            "k5000 PVE (%) | self PVE (%) | k1000 PVE (%) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cores = _mapping(result.get("cores"), "cores")
    for alias in _CORE_ALIASES:
        core = _mapping(cores.get(alias), f"cores.{alias}")
        runs = _mapping(core.get("runs"), f"cores.{alias}.runs")
        contrasts = _mapping(
            core.get("contrasts"), f"cores.{alias}.contrasts"
        )
        lines.append(
            "| {alias} | {k5} | {self_h} | {k1} | {rep} | {large} | "
            "{k5_pve} | {self_pve} | {k1_pve} |".format(
                alias=alias,
                k5=_format(
                    runs["k5000"]["whole_node_mean_masked_huber"], 6
                ),
                self_h=_format(
                    runs["matched_self"][
                        "whole_node_mean_masked_huber"
                    ],
                    6,
                ),
                k1=_format(
                    runs["k1000"]["whole_node_mean_masked_huber"], 6
                ),
                rep=_format(
                    contrasts["k5000_vs_matched_self"][
                        "relative_huber_gain_percent"
                    ]
                ),
                large=_format(
                    contrasts["k5000_vs_k1000"][
                        "relative_huber_gain_percent"
                    ]
                ),
                k5_pve=_format(
                    runs["k5000"][
                        "whole_node_mean_masked_percent_variance_explained"
                    ]
                ),
                self_pve=_format(
                    runs["matched_self"][
                        "whole_node_mean_masked_percent_variance_explained"
                    ]
                ),
                k1_pve=_format(
                    runs["k1000"][
                        "whole_node_mean_masked_percent_variance_explained"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Locked core-level inference",
            "",
            (
                "| Contrast | Mean gain (%) | Median | SD | 95% t CI | "
                "Positive cores | Mean PVE Δ (points) | Exact one-sided p | "
                "Holm p | Gate |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for name, aggregate, gate_name in (
        (
            "k5000 vs matched self",
            representation,
            "representation_gate",
        ),
        ("k5000 vs k1000", large_k, "large_k_gate"),
    ):
        gain = _mapping(
            aggregate.get("relative_huber_gain_percent"),
            f"{name} gain",
        )
        ci = _mapping(gain.get("paired_t_95_percent_ci"), f"{name} CI")
        inference = _mapping(
            aggregate.get("confirmatory_inference"),
            f"{name} inference",
        )
        pve = _mapping(aggregate.get("pve_percent"), f"{name} PVE")
        gate_result = _mapping(result.get(gate_name), gate_name)
        lines.append(
            "| {name} | {mean} | {median} | {sd} | [{lower}, {upper}] | "
            "{positive}/10 | {pve_difference} | {raw_p} | {holm_p} | "
            "{gate} |".format(
                name=name,
                mean=_format(gain["mean"]),
                median=_format(gain["median"]),
                sd=_format(gain["sample_sd"]),
                lower=_format(ci["lower"]),
                upper=_format(ci["upper"]),
                positive=int(gain["positive_core_count"]),
                pve_difference=_format(
                    pve["candidate_minus_reference_mean_points"]
                ),
                raw_p=_format(
                    inference["raw_exact_sign_flip_one_sided_p"], 6
                ),
                holm_p=_format(
                    inference[
                        "holm_adjusted_exact_sign_flip_one_sided_p"
                    ],
                    6,
                ),
                gate="PASS" if gate_result["passes"] else "FAIL",
            )
        )

    lines.extend(
        [
            "",
            (
                "The two exact one-sided sign-flip tests form one "
                "Holm-corrected confirmatory family. Wilcoxon and "
                "leave-one-core-out results are sensitivity analyses in "
                "`comparison.json`."
            ),
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
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


def _rename_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise MultiCoreQKVComparisonError(
            "platform lacks renameat2; refusing non-atomic publication"
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
        raise MultiCoreQKVComparisonError(
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
    """Atomically publish comparison JSON and Markdown without overwrite."""

    destination = Path(output_dir).resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise MultiCoreQKVComparisonError(
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
            "Verify and compare the locked ten-core, three-arm QKV campaign."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compare_multicore_qkv_large_k(args.manifest)
        output = write_comparison(result, args.output_dir)
    except MultiCoreQKVComparisonError as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
