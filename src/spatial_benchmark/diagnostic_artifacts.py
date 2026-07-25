"""Immutable artifacts for the benchmark's non-learned diagnostic controls.

The default route evaluates every fixed validation mask and leaves the sealed
test split unopened.  Test targets are scored only when ``open_test=True`` is
passed explicitly.  Both controls fit their fallback mean on training nodes
only, and nearest-neighbor candidates are restricted to the split being
evaluated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import numpy as np

from .artifacts import load_prepared_artifact, sha256_file
from .diagnostics import (
    NearestSpatialNeighborCopyPredictor,
    TrainGlobalMeanPredictor,
)
from .masking import FixedMaskBundle
from .metrics import evaluate_masked_predictions


DIAGNOSTIC_ARTIFACT_FORMAT_VERSION = 1
DIAGNOSTIC_ARTIFACT_KIND = (
    "normal_true_tissue_spatial_benchmark_diagnostic_controls"
)
_PREPARED_ARTIFACT_KIND = (
    "normal_true_tissue_spatial_benchmark_preparation"
)
_MANIFEST_FILENAME = "manifest.json"
_METRICS_FILENAME = "metrics.json"
_PREDICTIONS_FILENAME = "predictions.npz"
_CHECKSUM_FILENAME = "checksums.sha256"
_CONTROL_NAMES = (
    "train_global_gene_mean",
    "nearest_spatial_neighbor_copy",
)
_FIT_SCOPE = "train nodes only"
_HEX_16 = re.compile(r"^[0-9a-f]{16}$")
_HEX_20 = re.compile(r"^[0-9a-f]{20}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class DiagnosticArtifactError(ValueError):
    """Raised when a diagnostic artifact violates its immutable contract."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    core = deepcopy(dict(manifest))
    core.pop("manifest_content_sha256", None)
    return _canonical_hash(core)


def _array_content_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _fixed_mask_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"shape": list(array.shape), "dtype": array.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_checksum_file(root: Path) -> None:
    records = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == _CHECKSUM_FILENAME:
            continue
        if path.is_symlink() or not path.is_file():
            raise DiagnosticArtifactError(
                "Diagnostic artifacts may contain regular files only."
            )
        records.append((path.name, sha256_file(path)))
    (root / _CHECKSUM_FILENAME).write_text(
        "".join(
            f"{checksum}  {name}\n"
            for name, checksum in records
        ),
        encoding="ascii",
    )


def _read_checksum_file(root: Path) -> dict[str, str]:
    path = root / _CHECKSUM_FILENAME
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            "Diagnostic artifact lacks a regular checksums.sha256 file."
        )
    records: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        try:
            checksum, name = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise DiagnosticArtifactError(
                "Invalid diagnostic checksum line."
            ) from exc
        if (
            _HEX_64.fullmatch(checksum) is None
            or _SAFE_FILENAME.fullmatch(name) is None
            or name == _CHECKSUM_FILENAME
            or name in records
        ):
            raise DiagnosticArtifactError(
                "Invalid diagnostic checksum record."
            )
        records[name] = checksum
    return records


def _verify_artifact_files(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(
            f"Diagnostic artifact directory was not found: {root}"
        )
    entries = list(root.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise DiagnosticArtifactError(
            "Diagnostic artifact contains a directory, symlink, or special file."
        )
    actual = {
        path.name: path
        for path in entries
        if path.name != _CHECKSUM_FILENAME
    }
    expected = _read_checksum_file(root)
    if set(actual) != set(expected):
        raise DiagnosticArtifactError(
            "Diagnostic artifact file set differs from checksums."
        )
    for name, path in actual.items():
        if sha256_file(path) != expected[name]:
            raise DiagnosticArtifactError(
                f"Diagnostic artifact checksum mismatch for {name}."
            )


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise DiagnosticArtifactError(f"{name} must be boolean.")
    return bool(value)


def _validated_minimum(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DiagnosticArtifactError(
            "min_distance_um must be finite and non-negative."
        )
    try:
        minimum = float(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosticArtifactError(
            "min_distance_um must be finite and non-negative."
        ) from exc
    if not math.isfinite(minimum) or minimum < 0:
        raise DiagnosticArtifactError(
            "min_distance_um must be finite and non-negative."
        )
    return minimum


def _validated_block_size(value: object | None) -> int | None:
    if value is None:
        return None
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) <= 0
    ):
        raise DiagnosticArtifactError(
            "distance_block_size must be a positive integer or null."
        )
    return int(value)


def _validated_command(command: Sequence[str] | None) -> list[str]:
    if command is None:
        return []
    if isinstance(command, (str, bytes)) or any(
        not isinstance(item, str) for item in command
    ):
        raise DiagnosticArtifactError("command must be a sequence of strings.")
    return list(command)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _prepared_views(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    bundles: Mapping[str, FixedMaskBundle],
    *,
    evaluated_splits: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if manifest.get("artifact_kind") != _PREPARED_ARTIFACT_KIND:
        raise DiagnosticArtifactError(
            "Input is not a normal-tissue benchmark preparation artifact."
        )
    required_arrays = {
        "target_expression",
        "coordinates_um",
        "macroblock_ids",
        "split_labels",
        "train_node_index",
        "validation_node_index",
        "test_node_index",
    }
    if not required_arrays.issubset(arrays):
        missing = sorted(required_arrays.difference(arrays))
        raise DiagnosticArtifactError(
            f"Prepared artifact lacks required arrays: {', '.join(missing)}."
        )
    expression = np.asarray(arrays["target_expression"])
    coordinates = np.asarray(arrays["coordinates_um"])
    blocks = np.asarray(arrays["macroblock_ids"])
    labels = np.asarray(arrays["split_labels"]).astype(str)
    if (
        expression.ndim != 2
        or expression.shape[0] == 0
        or expression.shape[1] == 0
        or not (
            np.issubdtype(expression.dtype, np.integer)
            or np.issubdtype(expression.dtype, np.floating)
        )
    ):
        raise DiagnosticArtifactError(
            "Prepared target_expression must be numeric [nodes, genes]."
        )
    n_nodes, n_genes = expression.shape
    if coordinates.shape != (n_nodes, 2):
        raise DiagnosticArtifactError(
            "Prepared coordinates_um shape does not match target_expression."
        )
    if blocks.shape != (n_nodes,) or labels.shape != (n_nodes,):
        raise DiagnosticArtifactError(
            "Prepared block IDs or split labels are misaligned."
        )

    expected_labels = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    indices: dict[str, np.ndarray] = {}
    for split, expected_label in expected_labels.items():
        raw_index = np.asarray(arrays[f"{split}_node_index"])
        if (
            raw_index.ndim != 1
            or not np.issubdtype(raw_index.dtype, np.integer)
            or raw_index.size == 0
        ):
            raise DiagnosticArtifactError(
                f"Prepared {split}_node_index must be a nonempty integer vector."
            )
        index = np.asarray(raw_index, dtype=np.int64)
        expected_index = np.flatnonzero(labels == expected_label).astype(
            np.int64
        )
        if (
            np.any(index < 0)
            or np.any(index >= n_nodes)
            or len(np.unique(index)) != len(index)
            or not np.array_equal(index, expected_index)
        ):
            raise DiagnosticArtifactError(
                f"Prepared {split} node indices do not match split labels."
            )
        indices[split] = index
    combined = np.concatenate(
        [indices["train"], indices["validation"], indices["test"]]
    )
    if not np.array_equal(np.sort(combined), np.arange(n_nodes)):
        raise DiagnosticArtifactError(
            "Prepared train, validation, and test nodes do not form a partition."
        )

    # The sealed route deliberately does not form or inspect a test target view.
    for split in ("train", *evaluated_splits):
        index = indices[split]
        if (
            not np.isfinite(
                np.asarray(expression[index], dtype=np.float64)
            ).all()
            or not np.isfinite(
                np.asarray(coordinates[index], dtype=np.float64)
            ).all()
        ):
            raise DiagnosticArtifactError(
                f"Prepared {split} expression or coordinates are non-finite."
            )

    prepared_bundles = manifest.get("fixed_masks", {}).get("bundles", {})
    for split in evaluated_splits:
        if split not in bundles or split not in prepared_bundles:
            raise DiagnosticArtifactError(
                f"Prepared artifact lacks the {split} fixed-mask bundle."
            )
        bundle = bundles[split]
        record = prepared_bundles[split]
        if (
            bundle.bundle_id != record.get("bundle_id")
            or bundle.checksum != record.get("bundle_checksum")
            or int(bundle.manifest.get("n_genes", -1)) != n_genes
        ):
            raise DiagnosticArtifactError(
                f"Prepared {split} fixed-mask bundle declaration differs."
            )
        split_record = bundle.manifest.get("splits", {}).get(split, {})
        if int(split_record.get("n_nodes", -1)) != len(indices[split]):
            raise DiagnosticArtifactError(
                f"Prepared {split} fixed masks have the wrong node count."
            )
        entries = bundle.manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise DiagnosticArtifactError(
                f"Prepared {split} fixed-mask bundle has no entries."
            )
        for entry in entries:
            entry_id = str(entry.get("entry_id", ""))
            mask = bundle.masks.get(entry_id)
            if (
                entry.get("split") != split
                or mask is None
                or mask.shape != (len(indices[split]), n_genes)
            ):
                raise DiagnosticArtifactError(
                    f"Prepared fixed mask {entry_id!r} is not {split}-local."
                )
    return expression, indices


def _mask_record(
    split: str,
    bundle: FixedMaskBundle,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    spec = entry.get("spec")
    if not isinstance(spec, Mapping):
        raise DiagnosticArtifactError("Fixed-mask entry lacks its specification.")
    return {
        "split": split,
        "bundle_id": bundle.bundle_id,
        "bundle_checksum": bundle.checksum,
        "entry_id": str(entry["entry_id"]),
        "spec_id": str(entry["spec_id"]),
        "mode": str(spec["mode"]),
        "replicate": int(entry["replicate"]),
        "mask_checksum": str(entry["mask_checksum"]),
        "n_masked_entries": int(entry["summary"]["n_masked_entries"]),
    }


def _prediction_schema_record(
    array: np.ndarray,
    *,
    role: str,
    evaluation_id: str | None,
) -> dict[str, Any]:
    return {
        "role": role,
        "evaluation_id": evaluation_id,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def _evaluation_record(
    *,
    split: str,
    mask_record: Mapping[str, Any],
    n_split_nodes: int,
    prediction: Any,
    target_expression: np.ndarray,
    block_ids: np.ndarray,
) -> dict[str, Any]:
    metrics = evaluate_masked_predictions(
        **prediction.metrics_inputs(target_expression),
        block_ids=block_ids,
    )
    return {
        "split": split,
        "control": prediction.control,
        "mask_bundle_id": mask_record["bundle_id"],
        "mask_bundle_checksum": mask_record["bundle_checksum"],
        "mask_entry_id": mask_record["entry_id"],
        "mask_spec_id": mask_record["spec_id"],
        "mask_mode": mask_record["mode"],
        "mask_replicate": mask_record["replicate"],
        "mask_checksum": mask_record["mask_checksum"],
        "n_split_nodes": int(n_split_nodes),
        "n_evaluated_entries": prediction.n_evaluated_entries,
        "n_copied_entries": prediction.n_copied_entries,
        "n_fallback_mean_entries": prediction.n_fallback_entries,
        "source_copy_rate": prediction.copy_rate,
        "min_distance_um": prediction.min_distance_um,
        "fit_scope": _FIT_SCOPE,
        "n_training_nodes": prediction.n_training_cells,
        "candidate_scope": f"{split} nodes only",
        "source_node_index_space": "split-local",
        "metrics": metrics,
    }


def _run_evaluations(
    *,
    arrays: Mapping[str, np.ndarray],
    expression: np.ndarray,
    indices: Mapping[str, np.ndarray],
    bundles: Mapping[str, FixedMaskBundle],
    evaluated_splits: Sequence[str],
    mean_predictor: TrainGlobalMeanPredictor,
    nearest_predictor: NearestSpatialNeighborCopyPredictor,
    min_distance_um: float,
    distance_block_size: int | None,
    save_predictions: bool,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    metric_records: dict[str, list[dict[str, Any]]] = {
        "validation": [],
        "test": [],
    }
    declarations: list[dict[str, Any]] = []
    prediction_arrays: dict[str, np.ndarray] = {}
    prediction_schema: dict[str, dict[str, Any]] = {}
    mask_records: dict[str, list[dict[str, Any]]] = {
        split: [] for split in evaluated_splits
    }

    for split in evaluated_splits:
        split_index = indices[split]
        split_expression = np.asarray(
            expression[split_index], dtype=np.float64
        )
        split_coordinates = np.asarray(
            arrays["coordinates_um"][split_index], dtype=np.float64
        )
        split_blocks = np.asarray(arrays["macroblock_ids"][split_index])
        bundle = bundles[split]
        for mask_position, entry in enumerate(bundle.manifest["entries"]):
            mask_meta = _mask_record(split, bundle, entry)
            mask_records[split].append(mask_meta)
            mask = np.asarray(
                bundle.masks[mask_meta["entry_id"]], dtype=bool
            )
            if int(mask.sum()) != int(mask_meta["n_masked_entries"]):
                raise DiagnosticArtifactError(
                    f"Fixed mask {mask_meta['entry_id']} count changed."
                )
            mask_key: str | None = None
            if save_predictions:
                mask_key = f"{split}__m{mask_position:03d}__mask"
                stored_mask = np.asarray(mask, dtype=bool)
                prediction_arrays[mask_key] = stored_mask
                prediction_schema[mask_key] = _prediction_schema_record(
                    stored_mask,
                    role="evaluation_mask",
                    evaluation_id=None,
                )

            predictions = (
                mean_predictor.predict(mask, split_name=split),
                nearest_predictor.predict(
                    split_coordinates,
                    split_expression,
                    mask,
                    split_name=split,
                    min_distance_um=min_distance_um,
                    distance_block_size=distance_block_size,
                ),
            )
            for prediction in predictions:
                evaluation_id = (
                    f"{split}::{mask_meta['entry_id']}::{prediction.control}"
                )
                record = _evaluation_record(
                    split=split,
                    mask_record=mask_meta,
                    n_split_nodes=len(split_index),
                    prediction=prediction,
                    target_expression=split_expression,
                    block_ids=split_blocks,
                )
                metric_index = len(metric_records[split])
                metric_records[split].append(record)
                array_declaration: dict[str, str] | None = None
                if save_predictions:
                    assert mask_key is not None
                    prefix = (
                        f"{split}__m{mask_position:03d}__"
                        f"{prediction.control}"
                    )
                    prediction_key = f"{prefix}__prediction"
                    stored_prediction = np.asarray(
                        prediction.predictions, dtype=np.float32
                    )
                    prediction_arrays[prediction_key] = stored_prediction
                    prediction_schema[
                        prediction_key
                    ] = _prediction_schema_record(
                        stored_prediction,
                        role="prediction",
                        evaluation_id=evaluation_id,
                    )
                    array_declaration = {
                        "mask": mask_key,
                        "prediction": prediction_key,
                    }
                    if prediction.control == _CONTROL_NAMES[1]:
                        source_key = f"{prefix}__source_node_index"
                        distance_key = f"{prefix}__source_distance_um"
                        stored_source = np.asarray(
                            prediction.source_node_index, dtype=np.int32
                        )
                        stored_distance = np.asarray(
                            prediction.source_distance_um, dtype=np.float32
                        )
                        prediction_arrays[source_key] = stored_source
                        prediction_arrays[distance_key] = stored_distance
                        prediction_schema[
                            source_key
                        ] = _prediction_schema_record(
                            stored_source,
                            role="source_node_index",
                            evaluation_id=evaluation_id,
                        )
                        prediction_schema[
                            distance_key
                        ] = _prediction_schema_record(
                            stored_distance,
                            role="source_distance_um",
                            evaluation_id=evaluation_id,
                        )
                        array_declaration.update(
                            {
                                "source_node_index": source_key,
                                "source_distance_um": distance_key,
                            }
                        )
                declarations.append(
                    {
                        "evaluation_id": evaluation_id,
                        "split": split,
                        "control": prediction.control,
                        "mask_entry_id": mask_meta["entry_id"],
                        "mask_checksum": mask_meta["mask_checksum"],
                        "metric_record_index": metric_index,
                        "prediction_arrays": array_declaration,
                    }
                )
    return (
        metric_records,
        declarations,
        prediction_arrays,
        prediction_schema,
        mask_records,
    )


def _validate_metric_record(
    record: Mapping[str, Any],
    *,
    split: str,
    control: str,
    mask: Mapping[str, Any],
    n_training_nodes: int,
    n_split_nodes: int,
    min_distance_um: float,
) -> None:
    if (
        record.get("split") != split
        or record.get("control") != control
        or record.get("mask_bundle_id") != mask.get("bundle_id")
        or record.get("mask_bundle_checksum") != mask.get("bundle_checksum")
        or record.get("mask_entry_id") != mask.get("entry_id")
        or record.get("mask_spec_id") != mask.get("spec_id")
        or record.get("mask_mode") != mask.get("mode")
        or record.get("mask_replicate") != mask.get("replicate")
        or record.get("mask_checksum") != mask.get("mask_checksum")
    ):
        raise DiagnosticArtifactError(
            "Diagnostic metric record differs from its mask declaration."
        )
    if (
        record.get("fit_scope") != _FIT_SCOPE
        or record.get("n_training_nodes") != n_training_nodes
        or record.get("n_split_nodes") != n_split_nodes
        or record.get("candidate_scope") != f"{split} nodes only"
        or record.get("source_node_index_space") != "split-local"
    ):
        raise DiagnosticArtifactError(
            "Diagnostic metric record has invalid split or fit provenance."
        )
    n_evaluated = record.get("n_evaluated_entries")
    n_copied = record.get("n_copied_entries")
    n_fallback = record.get("n_fallback_mean_entries")
    if (
        not isinstance(n_evaluated, int)
        or isinstance(n_evaluated, bool)
        or n_evaluated < 0
        or not isinstance(n_copied, int)
        or isinstance(n_copied, bool)
        or n_copied < 0
        or not isinstance(n_fallback, int)
        or isinstance(n_fallback, bool)
        or n_fallback < 0
        or n_copied + n_fallback != n_evaluated
        or n_evaluated != mask.get("n_masked_entries")
    ):
        raise DiagnosticArtifactError(
            "Diagnostic source-copy counts are inconsistent."
        )
    rate = record.get("source_copy_rate")
    if (
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(float(rate))
        or not math.isclose(
            float(rate),
            n_copied / n_evaluated if n_evaluated else 0.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise DiagnosticArtifactError(
            "Diagnostic source-copy rate is inconsistent."
        )
    if control == _CONTROL_NAMES[0]:
        if (
            n_copied != 0
            or n_fallback != n_evaluated
            or record.get("min_distance_um") is not None
        ):
            raise DiagnosticArtifactError(
                "Global-mean diagnostic has invalid source-copy provenance."
            )
    elif (
        not isinstance(record.get("min_distance_um"), (int, float))
        or isinstance(record.get("min_distance_um"), bool)
        or not math.isclose(
            float(record["min_distance_um"]),
            min_distance_um,
            rel_tol=0,
            abs_tol=0,
        )
    ):
        raise DiagnosticArtifactError(
            "Nearest-copy diagnostic minimum distance differs."
        )
    metric = record.get("metrics")
    if not isinstance(metric, Mapping):
        raise DiagnosticArtifactError("Diagnostic record lacks benchmark metrics.")
    if (
        metric.get("n_masked") != n_evaluated
        or metric.get("n_prediction_seeds") != 1
        or any(name not in metric for name in ("huber", "mse", "mae"))
    ):
        raise DiagnosticArtifactError(
            "Diagnostic benchmark metric totals are inconsistent."
        )
    block_metrics = metric.get("blocks")
    if not isinstance(block_metrics, list) or not block_metrics:
        raise DiagnosticArtifactError(
            "Diagnostic record lacks per-block benchmark metrics."
        )
    if any(
        not isinstance(block, Mapping)
        or not isinstance(block.get("n_masked"), int)
        or isinstance(block.get("n_masked"), bool)
        or block["n_masked"] < 0
        for block in block_metrics
    ):
        raise DiagnosticArtifactError(
            "Per-block diagnostic metrics have invalid masked counts."
        )
    if sum(block["n_masked"] for block in block_metrics) != n_evaluated:
        raise DiagnosticArtifactError(
            "Per-block diagnostic metrics do not cover the evaluation mask."
        )
    for name in ("huber", "mse", "mae"):
        value = metric[name]
        if n_evaluated:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise DiagnosticArtifactError(
                    f"Diagnostic benchmark metric {name} is not finite."
                )
        elif value is not None:
            raise DiagnosticArtifactError(
                f"Empty diagnostic benchmark metric {name} must be null."
            )


def _validate_metrics_and_declarations(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> None:
    if set(metrics) != {
        "validation",
        "test",
        "test_targets_evaluated",
    }:
        raise DiagnosticArtifactError(
            "Diagnostic metrics root has an unexpected schema."
        )
    contract = manifest.get("run_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("version") != 1
        or contract.get("controls") != list(_CONTROL_NAMES)
        or contract.get("fit_scope") != _FIT_SCOPE
        or not isinstance(contract.get("split_id"), str)
        or not contract["split_id"]
    ):
        raise DiagnosticArtifactError(
            "Diagnostic run contract has invalid control or fit provenance."
        )
    _validated_block_size(contract.get("distance_block_size"))
    opened = _strict_bool(
        manifest.get("sealed_test_opened"), name="sealed_test_opened"
    )
    if (
        _strict_bool(
            metrics.get("test_targets_evaluated"),
            name="test_targets_evaluated",
        )
        != opened
        or _strict_bool(
            contract.get("open_test"),
            name="run_contract.open_test",
        )
        != opened
    ):
        raise DiagnosticArtifactError(
            "Sealed-test state differs across diagnostic declarations."
        )
    evaluated_splits = manifest.get("evaluated_splits")
    expected_splits = ["validation", "test"] if opened else ["validation"]
    if evaluated_splits != expected_splits:
        raise DiagnosticArtifactError(
            "Diagnostic evaluated_splits differs from sealed-test state."
        )
    evaluated_bundles = contract.get("evaluated_mask_bundles")
    prepared = manifest.get("prepared_artifact")
    fit = manifest.get("fit")
    if (
        not isinstance(evaluated_bundles, Mapping)
        or set(evaluated_bundles) != set(expected_splits)
        or not isinstance(prepared, Mapping)
        or _HEX_16.fullmatch(str(prepared.get("artifact_id", ""))) is None
        or prepared.get("artifact_id") != contract.get("prepared_artifact_id")
        or _HEX_64.fullmatch(
            str(prepared.get("manifest_sha256", ""))
        )
        is None
        or prepared.get("manifest_sha256")
        != contract.get("prepared_manifest_sha256")
        or prepared.get("split_id") != contract.get("split_id")
        or prepared.get("evaluated_mask_bundles") != evaluated_bundles
        or not isinstance(prepared.get("path"), str)
        or not prepared["path"]
        or not isinstance(fit, Mapping)
        or fit.get("scope") != _FIT_SCOPE
        or fit.get("shared_by_both_controls") is not True
        or _HEX_64.fullmatch(
            str(fit.get("train_global_gene_mean_sha256", ""))
        )
        is None
    ):
        raise DiagnosticArtifactError(
            "Diagnostic prepared-artifact or train-fit provenance is invalid."
        )
    if not isinstance(metrics.get("validation"), list) or not metrics["validation"]:
        raise DiagnosticArtifactError(
            "Diagnostic artifact has no validation metrics."
        )
    if opened:
        if not isinstance(metrics.get("test"), list) or not metrics["test"]:
            raise DiagnosticArtifactError(
                "Opened diagnostic artifact has no test metrics."
            )
    elif metrics.get("test") != []:
        raise DiagnosticArtifactError(
            "Sealed diagnostic artifact contains test outcomes."
        )

    split_counts = manifest.get("split_node_counts")
    if not isinstance(split_counts, Mapping):
        raise DiagnosticArtifactError(
            "Diagnostic manifest lacks split node counts."
        )
    n_training_nodes = split_counts.get("train")
    validation_nodes = split_counts.get("validation")
    test_nodes = split_counts.get("test")
    if (
        any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in (
                n_training_nodes,
                validation_nodes,
                *((test_nodes,) if opened else ()),
            )
        )
        or (not opened and test_nodes is not None)
        or fit.get("n_training_nodes") != n_training_nodes
        or not isinstance(fit.get("n_genes"), int)
        or isinstance(fit.get("n_genes"), bool)
        or fit["n_genes"] <= 0
    ):
        raise DiagnosticArtifactError(
            "Diagnostic split node counts violate sealed-test routing."
        )
    minimum = _validated_minimum(contract.get("min_distance_um"))
    mask_entries = manifest.get("evaluated_mask_entries")
    if (
        not isinstance(mask_entries, Mapping)
        or set(mask_entries) != set(expected_splits)
    ):
        raise DiagnosticArtifactError(
            "Diagnostic mask-entry declarations differ from evaluated splits."
        )
    expected_records: dict[
        tuple[str, str, str], tuple[Mapping[str, Any], int]
    ] = {}
    for split in expected_splits:
        entries = mask_entries.get(split)
        if not isinstance(entries, list) or not entries:
            raise DiagnosticArtifactError(
                f"Diagnostic manifest declares no {split} masks."
            )
        entry_ids: set[str] = set()
        bundle_declaration = evaluated_bundles[split]
        if (
            not isinstance(bundle_declaration, Mapping)
            or _HEX_16.fullmatch(
                str(bundle_declaration.get("bundle_id", ""))
            )
            is None
            or _HEX_64.fullmatch(
                str(bundle_declaration.get("bundle_checksum", ""))
            )
            is None
        ):
            raise DiagnosticArtifactError(
                f"Diagnostic {split} bundle declaration is invalid."
            )
        for mask in entries:
            if not isinstance(mask, Mapping):
                raise DiagnosticArtifactError(
                    "Diagnostic mask declaration must be a mapping."
                )
            entry_id = mask.get("entry_id")
            if (
                not isinstance(entry_id, str)
                or not entry_id
                or entry_id in entry_ids
                or mask.get("split") != split
                or mask.get("bundle_id")
                != bundle_declaration.get("bundle_id")
                or mask.get("bundle_checksum")
                != bundle_declaration.get("bundle_checksum")
                or _HEX_64.fullmatch(str(mask.get("mask_checksum", ""))) is None
                or _HEX_64.fullmatch(
                    str(mask.get("bundle_checksum", ""))
                )
                is None
            ):
                raise DiagnosticArtifactError(
                    "Diagnostic mask declaration is invalid or duplicated."
                )
            entry_ids.add(entry_id)
            for control in _CONTROL_NAMES:
                expected_records[(split, entry_id, control)] = (
                    mask,
                    int(split_counts[split]),
                )

    records_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for split in expected_splits:
        records = metrics.get(split)
        if not isinstance(records, list):
            raise DiagnosticArtifactError(
                f"Diagnostic {split} metrics must be a list."
            )
        for record in records:
            if not isinstance(record, Mapping):
                raise DiagnosticArtifactError(
                    "Diagnostic metric record must be a mapping."
                )
            key = (
                str(record.get("split")),
                str(record.get("mask_entry_id")),
                str(record.get("control")),
            )
            if key in records_by_key:
                raise DiagnosticArtifactError(
                    "Diagnostic metric record is duplicated."
                )
            records_by_key[key] = record
    if set(records_by_key) != set(expected_records):
        raise DiagnosticArtifactError(
            "Diagnostic metrics do not cover every mask/control pair exactly once."
        )
    for key, (mask, n_split_nodes) in expected_records.items():
        _validate_metric_record(
            records_by_key[key],
            split=key[0],
            control=key[2],
            mask=mask,
            n_training_nodes=int(n_training_nodes),
            n_split_nodes=n_split_nodes,
            min_distance_um=minimum,
        )

    declarations = manifest.get("evaluations")
    if not isinstance(declarations, list):
        raise DiagnosticArtifactError(
            "Diagnostic manifest evaluations must be a list."
        )
    declared_keys: set[tuple[str, str, str]] = set()
    evaluation_ids: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise DiagnosticArtifactError(
                "Diagnostic evaluation declaration must be a mapping."
            )
        key = (
            str(declaration.get("split")),
            str(declaration.get("mask_entry_id")),
            str(declaration.get("control")),
        )
        evaluation_id = declaration.get("evaluation_id")
        if (
            key not in expected_records
            or key in declared_keys
            or not isinstance(evaluation_id, str)
            or not evaluation_id
            or evaluation_id in evaluation_ids
            or declaration.get("mask_checksum")
            != expected_records[key][0].get("mask_checksum")
        ):
            raise DiagnosticArtifactError(
                "Diagnostic evaluation declaration is invalid or duplicated."
            )
        metric_index = declaration.get("metric_record_index")
        split_records = metrics[key[0]]
        if (
            not isinstance(metric_index, int)
            or isinstance(metric_index, bool)
            or metric_index < 0
            or metric_index >= len(split_records)
            or split_records[metric_index] is not records_by_key[key]
        ):
            raise DiagnosticArtifactError(
                "Diagnostic metric index does not identify its declared record."
            )
        declared_keys.add(key)
        evaluation_ids.add(evaluation_id)
    if declared_keys != set(expected_records):
        raise DiagnosticArtifactError(
            "Diagnostic evaluations do not declare every metric record."
        )


def _load_prediction_arrays(
    root: Path,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    materialize: bool,
) -> dict[str, np.ndarray] | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DiagnosticArtifactError(
            "Diagnostic manifest lacks artifact declarations."
        )
    prediction_name = artifacts.get("predictions")
    save_predictions = _strict_bool(
        manifest.get("run_contract", {}).get("save_predictions"),
        name="run_contract.save_predictions",
    )
    schema = manifest.get("prediction_arrays")
    if not isinstance(schema, Mapping):
        raise DiagnosticArtifactError(
            "Diagnostic prediction schema must be a mapping."
        )
    declarations = manifest.get("evaluations", [])
    declared_keys: set[str] = set()
    expected_schema_provenance: dict[
        str, tuple[str, str | None, tuple[int, int], str]
    ] = {}
    n_genes = int(manifest["fit"]["n_genes"])
    split_counts = manifest["split_node_counts"]
    for declaration in declarations:
        arrays = declaration.get("prediction_arrays")
        if save_predictions:
            control = declaration.get("control")
            split = str(declaration["split"])
            expected_roles = {"mask", "prediction"}
            if control == _CONTROL_NAMES[1]:
                expected_roles.update(
                    {"source_node_index", "source_distance_um"}
                )
            if (
                not isinstance(arrays, Mapping)
                or set(arrays) != expected_roles
                or any(
                    not isinstance(value, str) or not value
                    for value in arrays.values()
                )
            ):
                raise DiagnosticArtifactError(
                    "Diagnostic prediction declaration is invalid."
                )
            evaluation_id = str(declaration["evaluation_id"])
            for role, key in arrays.items():
                if not key.startswith(f"{split}__"):
                    raise DiagnosticArtifactError(
                        "Diagnostic prediction key is not split-prefixed."
                    )
                declared_keys.add(key)
                stored_role = "evaluation_mask" if role == "mask" else role
                expected_dtype = {
                    "evaluation_mask": np.dtype(bool).str,
                    "prediction": np.dtype(np.float32).str,
                    "source_node_index": np.dtype(np.int32).str,
                    "source_distance_um": np.dtype(np.float32).str,
                }[stored_role]
                provenance = (
                    stored_role,
                    None if role == "mask" else evaluation_id,
                    (int(split_counts[split]), n_genes),
                    expected_dtype,
                )
                if (
                    key in expected_schema_provenance
                    and expected_schema_provenance[key] != provenance
                ):
                    raise DiagnosticArtifactError(
                        "A diagnostic prediction key has conflicting provenance."
                    )
                expected_schema_provenance[key] = provenance
        elif arrays is not None:
            raise DiagnosticArtifactError(
                "Prediction-disabled diagnostic declares prediction arrays."
            )
    if not save_predictions:
        if prediction_name is not None or schema or declared_keys:
            raise DiagnosticArtifactError(
                "Prediction-disabled diagnostic contains prediction metadata."
            )
        return None
    if prediction_name != _PREDICTIONS_FILENAME:
        raise DiagnosticArtifactError(
            "Prediction-enabled diagnostic has an invalid NPZ declaration."
        )
    prediction_path = root / prediction_name
    if prediction_path.is_symlink() or not prediction_path.is_file():
        raise FileNotFoundError("Diagnostic predictions.npz was not found.")
    if set(schema) != declared_keys:
        raise DiagnosticArtifactError(
            "Diagnostic prediction declarations do not cover the NPZ schema."
        )

    opened = bool(manifest["sealed_test_opened"])
    loaded: dict[str, np.ndarray] | None = {} if materialize else None
    with np.load(prediction_path, allow_pickle=False) as archive:
        if set(archive.files) != set(schema):
            raise DiagnosticArtifactError(
                "Diagnostic prediction NPZ keys differ from its manifest."
            )
        if not opened and any(key.startswith("test__") for key in archive.files):
            raise DiagnosticArtifactError(
                "Sealed diagnostic predictions contain test arrays."
            )
        for key in archive.files:
            record = schema[key]
            if not isinstance(record, Mapping):
                raise DiagnosticArtifactError(
                    "Diagnostic prediction-array declaration is invalid."
                )
            (
                expected_role,
                expected_evaluation_id,
                expected_shape,
                expected_dtype,
            ) = (
                expected_schema_provenance[key]
            )
            array = np.asarray(archive[key])
            if (
                record.get("role") != expected_role
                or record.get("evaluation_id") != expected_evaluation_id
                or tuple(array.shape) != expected_shape
                or array.dtype.str != expected_dtype
                or list(array.shape) != record.get("shape")
                or array.dtype.str != record.get("dtype")
            ):
                raise DiagnosticArtifactError(
                    f"Diagnostic prediction-array schema mismatch for {key}."
                )
            if loaded is not None:
                loaded[key] = np.array(array, copy=True)
        minimum = float(manifest["run_contract"]["min_distance_um"])
        for declaration in declarations:
            keys = declaration["prediction_arrays"]

            def get_array(role: str) -> np.ndarray:
                key = keys[role]
                if loaded is not None:
                    return loaded[key]
                return np.asarray(archive[key])

            split = declaration["split"]
            metric_record = metrics[split][
                declaration["metric_record_index"]
            ]
            mask = get_array("mask")
            prediction = get_array("prediction")
            if (
                mask.dtype != np.bool_
                or int(mask.sum())
                != int(metric_record["n_evaluated_entries"])
                or _fixed_mask_hash(mask)
                != metric_record["mask_checksum"]
                or not np.isfinite(prediction).all()
            ):
                raise DiagnosticArtifactError(
                    "Saved diagnostic mask or prediction content is invalid."
                )
            if declaration["control"] == _CONTROL_NAMES[0]:
                continue
            source = get_array("source_node_index")
            distance = get_array("source_distance_um")
            copied = mask & (source >= 0)
            if (
                np.any(source[~mask] != -1)
                or np.any(source[mask] < -1)
                or np.any(source[copied] >= mask.shape[0])
                or np.any(~np.isfinite(distance[copied]))
                or np.any(distance[copied] + 1e-6 < minimum)
                or np.any(~np.isnan(distance[~copied]))
                or int(copied.sum())
                != int(metric_record["n_copied_entries"])
            ):
                raise DiagnosticArtifactError(
                    "Saved nearest-source arrays violate source provenance."
                )
            copied_rows, copied_genes = np.nonzero(copied)
            copied_sources = source[copied_rows, copied_genes]
            if (
                np.any(copied_sources == copied_rows)
                or np.any(mask[copied_sources, copied_genes])
            ):
                raise DiagnosticArtifactError(
                    "Saved nearest sources are self-sources or hidden."
                )
    return loaded


def load_diagnostic_artifact(
    path: str | Path,
    *,
    load_predictions: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray] | None,
]:
    """Verify and load one immutable diagnostic-control artifact."""

    root = Path(path)
    _verify_artifact_files(root)
    manifest_path = root / _MANIFEST_FILENAME
    metrics_path = root / _METRICS_FILENAME
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or metrics_path.is_symlink()
        or not metrics_path.is_file()
    ):
        raise FileNotFoundError(
            "Diagnostic artifact lacks its regular manifest or metrics file."
        )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                DiagnosticArtifactError(
                    f"Invalid JSON constant in manifest: {value}."
                )
            ),
        )
        metrics = json.loads(
            metrics_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                DiagnosticArtifactError(
                    f"Invalid JSON constant in metrics: {value}."
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise DiagnosticArtifactError(
            "Diagnostic artifact contains invalid JSON."
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(metrics, dict):
        raise DiagnosticArtifactError(
            "Diagnostic manifest and metrics roots must be mappings."
        )
    if (
        manifest.get("format_version") != DIAGNOSTIC_ARTIFACT_FORMAT_VERSION
        or manifest.get("artifact_kind") != DIAGNOSTIC_ARTIFACT_KIND
        or manifest.get("status") != "complete"
    ):
        raise DiagnosticArtifactError(
            "Unsupported or incomplete diagnostic artifact."
        )
    content_hash = _manifest_content_hash(manifest)
    if manifest.get("manifest_content_sha256") != content_hash:
        raise DiagnosticArtifactError(
            "Diagnostic manifest content checksum mismatch."
        )
    run_contract = manifest.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise DiagnosticArtifactError(
            "Diagnostic manifest lacks its run contract."
        )
    expected_id = _canonical_hash(run_contract)[:20]
    if (
        _HEX_20.fullmatch(str(manifest.get("diagnostic_id", ""))) is None
        or manifest.get("diagnostic_id") != expected_id
    ):
        raise DiagnosticArtifactError(
            "Diagnostic ID does not match its run contract."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise DiagnosticArtifactError(
            "Diagnostic manifest lacks file checksums."
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DiagnosticArtifactError(
            "Diagnostic manifest lacks artifact declarations."
        )
    expected_payload = {_METRICS_FILENAME}
    if artifacts.get("predictions") is not None:
        expected_payload.add(_PREDICTIONS_FILENAME)
    if set(files) != expected_payload:
        raise DiagnosticArtifactError(
            "Diagnostic manifest payload declaration is invalid."
        )
    actual_payload = {
        path.name
        for path in root.iterdir()
        if path.name not in {_MANIFEST_FILENAME, _CHECKSUM_FILENAME}
    }
    if actual_payload != expected_payload:
        raise DiagnosticArtifactError(
            "Diagnostic payload file set differs from its manifest."
        )
    for name, checksum in files.items():
        if (
            _SAFE_FILENAME.fullmatch(str(name)) is None
            or _HEX_64.fullmatch(str(checksum)) is None
            or sha256_file(root / str(name)) != checksum
        ):
            raise DiagnosticArtifactError(
                f"Diagnostic manifest file checksum mismatch for {name}."
            )
    if manifest.get("metrics_file") != _METRICS_FILENAME:
        raise DiagnosticArtifactError(
            "Diagnostic metrics filename declaration is invalid."
        )
    _validate_metrics_and_declarations(manifest, metrics)
    predictions = _load_prediction_arrays(
        root,
        manifest,
        metrics,
        materialize=bool(load_predictions),
    )
    return manifest, metrics, predictions


def run_diagnostic_artifact(
    prepared_path: str | Path,
    output_path: str | Path,
    *,
    min_distance_um: float = 0.0,
    open_test: bool = False,
    save_predictions: bool = False,
    distance_block_size: int | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    """Evaluate fixed masks and atomically publish a diagnostic artifact."""

    minimum = _validated_minimum(min_distance_um)
    block_size = _validated_block_size(distance_block_size)
    opened = _strict_bool(open_test, name="open_test")
    save = _strict_bool(save_predictions, name="save_predictions")
    command_record = _validated_command(command)

    prepared_root = Path(prepared_path).resolve()
    requested_destination = Path(output_path)
    if requested_destination.exists() or requested_destination.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite immutable diagnostic artifact: "
            f"{requested_destination}"
        )
    destination = requested_destination.resolve()
    if _is_within(destination, prepared_root):
        raise DiagnosticArtifactError(
            "Diagnostic output may not be placed inside its prepared artifact."
        )
    manifest, arrays, bundles = load_prepared_artifact(
        prepared_root, load_arrays=True
    )
    if arrays is None:
        raise DiagnosticArtifactError(
            "Prepared artifact arrays were not loaded."
        )
    evaluated_splits = ["validation", "test"] if opened else ["validation"]
    expression, indices = _prepared_views(
        manifest,
        arrays,
        bundles,
        evaluated_splits=evaluated_splits,
    )
    prepared_manifest_sha256 = sha256_file(
        prepared_root / _MANIFEST_FILENAME
    )
    prepared_artifact_id = str(manifest["artifact_id"])

    train_expression = np.asarray(
        expression[indices["train"]], dtype=np.float64
    )
    mean_predictor = TrainGlobalMeanPredictor.fit(train_expression)
    nearest_predictor = NearestSpatialNeighborCopyPredictor(mean_predictor)
    del train_expression
    run_contract = {
        "version": 1,
        "prepared_artifact_id": prepared_artifact_id,
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "split_id": str(manifest["split"]["split_id"]),
        "evaluated_mask_bundles": {
            split: {
                "bundle_id": bundles[split].bundle_id,
                "bundle_checksum": bundles[split].checksum,
            }
            for split in evaluated_splits
        },
        "controls": list(_CONTROL_NAMES),
        "fit_scope": _FIT_SCOPE,
        "min_distance_um": minimum,
        "distance_block_size": block_size,
        "open_test": opened,
        "save_predictions": save,
        "metric_protocol": "masked benchmark metrics with spatial-block summaries",
        "nearest_source_protocol": (
            "nearest split-local node where the same gene is visible; "
            "deterministic row-index ties; train-mean fallback"
        ),
    }
    diagnostic_id = _canonical_hash(run_contract)[:20]
    (
        metric_records,
        declarations,
        prediction_arrays,
        prediction_schema,
        mask_records,
    ) = _run_evaluations(
        arrays=arrays,
        expression=expression,
        indices=indices,
        bundles=bundles,
        evaluated_splits=evaluated_splits,
        mean_predictor=mean_predictor,
        nearest_predictor=nearest_predictor,
        min_distance_um=minimum,
        distance_block_size=block_size,
        save_predictions=save,
    )
    metrics = {
        "validation": metric_records["validation"],
        "test": metric_records["test"] if opened else [],
        "test_targets_evaluated": opened,
    }
    diagnostic_manifest: dict[str, Any] = {
        "format_version": DIAGNOSTIC_ARTIFACT_FORMAT_VERSION,
        "artifact_kind": DIAGNOSTIC_ARTIFACT_KIND,
        "diagnostic_id": diagnostic_id,
        "status": "complete",
        "run_contract": run_contract,
        "prepared_artifact": {
            "path": str(prepared_root),
            "artifact_id": prepared_artifact_id,
            "manifest_sha256": prepared_manifest_sha256,
            "split_id": str(manifest["split"]["split_id"]),
            "evaluated_mask_bundles": run_contract[
                "evaluated_mask_bundles"
            ],
        },
        "fit": {
            "scope": _FIT_SCOPE,
            "n_training_nodes": mean_predictor.n_training_cells,
            "n_genes": mean_predictor.n_genes,
            "train_global_gene_mean_sha256": _array_content_hash(
                mean_predictor.gene_mean
            ),
            "shared_by_both_controls": True,
        },
        "evaluated_splits": evaluated_splits,
        "split_node_counts": {
            "train": int(len(indices["train"])),
            "validation": int(len(indices["validation"])),
            "test": int(len(indices["test"])) if opened else None,
        },
        "sealed_test_opened": opened,
        "evaluated_mask_entries": mask_records,
        "evaluations": declarations,
        "metrics_file": _METRICS_FILENAME,
        "artifacts": {
            "predictions": _PREDICTIONS_FILENAME if save else None,
        },
        "prediction_arrays": prediction_schema,
        "provenance": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command_record,
        },
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        _write_json(temporary / _METRICS_FILENAME, metrics)
        if save:
            np.savez_compressed(
                temporary / _PREDICTIONS_FILENAME,
                **{
                    key: prediction_arrays[key]
                    for key in sorted(prediction_arrays)
                },
            )
        diagnostic_manifest["files"] = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
            if path.is_file()
        }
        diagnostic_manifest[
            "manifest_content_sha256"
        ] = _manifest_content_hash(diagnostic_manifest)
        _write_json(temporary / _MANIFEST_FILENAME, diagnostic_manifest)
        _write_checksum_file(temporary)
        load_diagnostic_artifact(temporary, load_predictions=False)

        current_manifest, _, _ = load_prepared_artifact(
            prepared_root, load_arrays=False
        )
        if (
            str(current_manifest["artifact_id"]) != prepared_artifact_id
            or sha256_file(prepared_root / _MANIFEST_FILENAME)
            != prepared_manifest_sha256
        ):
            raise DiagnosticArtifactError(
                "Prepared artifact changed during diagnostic evaluation."
            )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                "Diagnostic output appeared before atomic publication: "
                f"{destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "DIAGNOSTIC_ARTIFACT_FORMAT_VERSION",
    "DIAGNOSTIC_ARTIFACT_KIND",
    "DiagnosticArtifactError",
    "load_diagnostic_artifact",
    "run_diagnostic_artifact",
]
