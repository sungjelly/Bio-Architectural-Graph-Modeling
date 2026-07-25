"""Run-artifact aggregation for the within-core predictive benchmark.

The analysis contract is intentionally narrow:

* each training seed lives in an isolated run directory;
* predictions are averaged across seeds before scoring;
* fixed mask replicates are averaged within spatial blocks;
* uncertainty resamples spatial blocks, never cells, entries, masks, or seeds;
* the result is descriptive for one core and cannot establish patient-level
  generalisation, biological communication, mechanism, or causality.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactContractError, load_prepared_artifact
from .masking import MaskSpec
from .metrics import (
    block_bootstrap_ci,
    ensemble_predictions,
    paired_sign_flip_test,
    paired_spatial_gain,
)
from .standards_lock import (
    REQUIRED_FINAL_CONDITIONS,
    StandardsLockError,
    canonical_job_hash,
    condition_name,
    expand_matrix,
    load_standards_lock,
)


ANALYSIS_FORMAT_VERSION = 1
EXPERIMENT_RUN_ARTIFACT_KIND = "normal_true_tissue_spatial_benchmark_run"
CONCLUSION_SPLIT = "test"
FINAL_MASK_MODES = ("partial", "node", "block")
FINAL_CONDITION_LABELS = {
    "b0": "B0 self-only",
    "b0_parameter_matched": "B0 parameter-matched",
    "broad_field": "Broad-Field",
    "b1": "B1 mean-neighbor",
    "g1_true": "G1 true",
    "g1_rewired": "G1 rewired",
    "g2_true": "G2 true",
    "g2_zero": "G2 zero-edge",
    "g2_distance_only": "G2 distance-only",
    "g2_permuted": "G2 permuted-edge",
}
WITHIN_CORE_SCOPE = (
    "Exploratory, descriptive within-core masked-expression prediction. "
    "Cells, mask replicates, and model seeds are technical units, not "
    "independent biological replicates. This analysis does not establish "
    "patient-level generalisation, communication, mechanism, or causality."
)


class AnalysisContractError(ValueError):
    """Raised when run artifacts cannot support a paired analysis."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_content_sha256(array: np.ndarray) -> str:
    """Match the immutable fixed-mask bundle's shape/dtype-aware checksum."""

    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {"shape": list(contiguous.shape), "dtype": contiguous.dtype.str},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _manifest_content_sha256(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("manifest_content_sha256", None)
    try:
        payload = json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnalysisContractError(
            "Run manifest cannot be represented as canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _set_read_only(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.flags.writeable = False
    return result


def _normalise_model(value: Any, graph_kind: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    aliases = {
        "b0": "b0",
        "self": "b0",
        "selfonly": "b0",
        "selfonlymlp": "b0",
        "b0matched": "b0_matched",
        "matchedself": "b0_matched",
        "parametermatchedself": "b0_matched",
        "broadfield": "broad_field",
        "broadspatialfield": "broad_field",
        "spatialfield": "broad_field",
        "b1": "b1",
        "meanneighbor": "b1",
        "meanneighbormodel": "b1",
        "g1": "g1",
        "g1rewired": "g1",
        "topologygat": "g1",
        "topologygatv2": "g1",
        "g2": "g2",
        "edgegat": "g2",
        "edgeconditionedgatv2": "g2",
        "g3": "g3",
        "additivebiogat": "g3",
        "additiveedgemessagemodel": "g3",
    }
    model = aliases.get(text, text)
    if model == "g1" and graph_kind == "rewired":
        return "g1_rewired"
    return model


def _normalise_graph_kind(value: Any) -> str:
    text = re.sub(r"[^a-z]+", "", str(value or "true").lower())
    if "rewir" in text or text in {"null", "permuted"}:
        return "rewired"
    if text in {"self", "none", "nograph"}:
        return "self"
    if text in {"broadfield", "broadspatialfield", "spatialfield"}:
        return "broad_field"
    return "true"


def _normalise_edge_control(value: Any) -> str:
    text = re.sub(r"[^a-z]+", "_", str(value or "none").strip().lower()).strip("_")
    aliases = {
        "full": "none",
        "none": "none",
        "zero": "zero",
        "distance": "distance_only",
        "distance_only": "distance_only",
        "permuted": "permuted",
        "permutation": "permuted",
    }
    if text not in aliases:
        raise AnalysisContractError(f"Unknown edge control {value!r}")
    return aliases[text]


def _canonical_condition(
    model: str,
    graph_kind: str,
    edge_control: str,
) -> str:
    if model == "b0":
        return "b0"
    if model == "b0_matched":
        return "b0_parameter_matched"
    if model == "broad_field":
        return "broad_field"
    if model == "b1":
        return "b1"
    if model in {"g1", "g1_rewired"}:
        return "g1_rewired" if graph_kind == "rewired" else "g1_true"
    if model == "g2":
        return {
            "none": "g2_true",
            "zero": "g2_zero",
            "distance_only": "g2_distance_only",
            "permuted": "g2_permuted",
        }[edge_control]
    if model == "g3":
        return "g3_conditional"
    raise AnalysisContractError(
        f"Model {model!r} cannot be mapped to a final benchmark condition"
    )


def _normalise_mode(value: Any) -> str:
    try:
        return MaskSpec(str(value)).mode
    except ValueError as exc:
        raise AnalysisContractError(f"Unknown evaluation mask mode {value!r}") from exc


def _nested_get(mapping: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = mapping
        found = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None:
            return current
    return None


def _mapping_name(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("name") or value.get("id") or value.get("model")
    return value


def _graph_metadata(manifest: Mapping[str, Any], model_value: Any) -> dict[str, Any]:
    graph = manifest.get("graph")
    graph_mapping = graph if isinstance(graph, Mapping) else {}
    config_graph = _nested_get(manifest, "config.graph")
    if not isinstance(config_graph, Mapping):
        config_graph = {}
    embedded_config = graph_mapping.get("config")
    if not isinstance(embedded_config, Mapping):
        embedded_config = {}
    kind_value = (
        manifest.get("graph_kind")
        or graph_mapping.get("kind")
        or config_graph.get("kind")
    )
    model_text = str(model_value).lower()
    if kind_value is None and "rewir" in model_text:
        kind_value = "rewired"
    elif kind_value is None and re.sub(r"[^a-z0-9]+", "", model_text) in {
        "b0",
        "self",
        "selfonly",
        "selfonlymlp",
    }:
        kind_value = "self"
    kind = _normalise_graph_kind(kind_value)
    graph_id = (
        manifest.get("graph_id")
        or graph_mapping.get("graph_id")
        or graph_mapping.get("id")
    )
    base_graph_id = (
        manifest.get("base_graph_id")
        or manifest.get("source_graph_id")
        or graph_mapping.get("base_graph_id")
        or graph_mapping.get("source_graph_id")
    )
    combined_config = {**dict(config_graph), **dict(embedded_config)}
    edge_control_value = (
        graph_mapping.get("edge_control")
        or _nested_get(manifest, "config.run.edge_control")
        or _nested_get(manifest, "run_contract.edge_control.name")
        or "none"
    )
    if isinstance(edge_control_value, Mapping):
        edge_control_value = edge_control_value.get("name", "none")
    edge_control = _normalise_edge_control(edge_control_value)
    if graph_id is None:
        if kind == "self":
            graph_id = "self"
        else:
            k = combined_config.get("k", "na")
            radius = combined_config.get("radius_um", "na")
            symmetry = combined_config.get("symmetry", "na")
            graph_id = f"k{k}_r{radius}_{symmetry}"
            if kind == "rewired":
                graph_id += "__rewired"
    return {
        "graph_id": str(graph_id),
        "base_graph_id": None if base_graph_id is None else str(base_graph_id),
        "graph_kind": kind,
        "graph_config": combined_config,
        "edge_control": edge_control,
    }


@dataclass(frozen=True)
class PredictionRecord:
    """One seed's prediction for one split/mask/replicate."""

    run_id: str
    condition: str
    model: str
    model_seed: int
    graph_id: str
    graph_kind: str
    base_graph_id: str | None
    graph_config: Mapping[str, Any]
    edge_control: str
    split: str
    mask_mode: str
    mask_replicate: int
    y_true: np.ndarray
    prediction: np.ndarray
    mask: np.ndarray
    block_ids: np.ndarray
    cell_ids: np.ndarray | None
    manifest_path: str
    manifest_checksum: str

    def __post_init__(self) -> None:
        target = np.asarray(self.y_true)
        prediction = np.asarray(self.prediction)
        mask = np.asarray(self.mask)
        blocks = np.asarray(self.block_ids)
        if target.ndim != 2 or prediction.shape != target.shape:
            raise AnalysisContractError(
                "y_true and prediction must share [cells, genes] shape"
            )
        if mask.shape != target.shape:
            raise AnalysisContractError("mask must match prediction shape")
        if mask.dtype != np.bool_:
            if np.issubdtype(mask.dtype, np.integer) and np.all(
                (mask == 0) | (mask == 1)
            ):
                mask = mask.astype(bool)
            else:
                raise AnalysisContractError("mask must be boolean or binary integer")
        if blocks.shape != (target.shape[0],):
            raise AnalysisContractError("block_ids must align to prediction rows")
        if prediction.ndim != 2:
            raise AnalysisContractError(
                "each isolated run must contain one seed, not a seed ensemble"
            )
        if self.cell_ids is not None:
            cells = np.asarray(self.cell_ids)
            if cells.ndim < 1 or cells.shape[0] != target.shape[0]:
                raise AnalysisContractError("cell_ids must align to prediction rows")
            object.__setattr__(self, "cell_ids", _set_read_only(cells))
        object.__setattr__(self, "y_true", _set_read_only(target))
        object.__setattr__(self, "prediction", _set_read_only(prediction))
        object.__setattr__(self, "mask", _set_read_only(mask))
        object.__setattr__(self, "block_ids", _set_read_only(blocks))
        object.__setattr__(self, "model_seed", int(self.model_seed))
        object.__setattr__(self, "mask_replicate", int(self.mask_replicate))

    @property
    def evaluation_key(self) -> tuple[str, str, int]:
        return (self.split, self.mask_mode, self.mask_replicate)


@dataclass(frozen=True)
class RunCollection:
    records: tuple[PredictionRecord, ...]
    graph_qc: tuple[Mapping[str, Any], ...]
    failed_runs: tuple[Mapping[str, Any], ...]
    excluded_runs: tuple[Mapping[str, Any], ...]
    manifest_checksums: Mapping[str, str]
    prediction_checksums: Mapping[str, str]
    run_audits: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class PreparedEvaluationContract:
    """Canonical prepared rows and fixed masks used to audit prediction files."""

    summary: Mapping[str, Any]
    expected_evaluations: Mapping[
        str,
        frozenset[tuple[str, str, int]],
    ]
    masks: Mapping[tuple[str, str, int], np.ndarray]
    mask_checksums: Mapping[tuple[str, str, int], str]
    y_true_by_split: Mapping[str, np.ndarray]
    block_ids_by_split: Mapping[str, np.ndarray]
    cell_ids_by_split: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class EnsembleEvaluation:
    """Seed stack for one model/graph/evaluation record."""

    condition: str
    model: str
    graph_id: str
    graph_kind: str
    base_graph_id: str | None
    graph_config: Mapping[str, Any]
    edge_control: str
    split: str
    mask_mode: str
    mask_replicate: int
    seeds: tuple[int, ...]
    run_ids: tuple[str, ...]
    y_true: np.ndarray
    predictions: np.ndarray
    mask: np.ndarray
    block_ids: np.ndarray
    cell_ids: np.ndarray | None
    expected_seeds: int

    @property
    def seed_complete(self) -> bool:
        return len(self.seeds) == self.expected_seeds

    @property
    def ensemble_prediction(self) -> np.ndarray:
        return np.asarray(ensemble_predictions(self.predictions))

    @property
    def evaluation_key(self) -> tuple[str, str, int]:
        return (self.split, self.mask_mode, self.mask_replicate)


def _manifest_model_and_seed(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> tuple[str, int, dict[str, Any]]:
    model_value = _nested_get(
        manifest,
        "model_name",
        "model",
        "config.model.name",
    )
    model_value = _mapping_name(model_value)
    if model_value is None:
        raise AnalysisContractError(
            f"Run manifest lacks a model name: {manifest_path}"
        )
    graph = _graph_metadata(manifest, model_value)
    model = _normalise_model(model_value, graph["graph_kind"])
    graph["condition"] = _canonical_condition(
        model,
        str(graph["graph_kind"]),
        str(graph["edge_control"]),
    )
    seed_value = _nested_get(
        manifest,
        "model_seed",
        "seed",
        "config.run.model_seed",
        "run.model_seed",
    )
    if seed_value is None:
        raise AnalysisContractError(
            f"Run manifest lacks a model seed: {manifest_path}"
        )
    return model, int(seed_value), graph


def _safe_relative_path(base: Path, value: Any) -> Path:
    candidate = (base / str(value)).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise AnalysisContractError(
            "prediction paths must stay within their isolated run directory"
        ) from exc
    return candidate


def _prediction_files(
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> list[Path]:
    declared = _nested_get(
        manifest,
        "prediction_file",
        "artifacts.predictions",
        "artifacts.prediction_file",
    )
    if declared is not None:
        values = declared if isinstance(declared, Sequence) and not isinstance(
            declared, (str, bytes)
        ) else [declared]
        files = [_safe_relative_path(run_dir, value) for value in values]
    elif (run_dir / "predictions.npz").is_file():
        files = [run_dir / "predictions.npz"]
    else:
        files = sorted(run_dir.glob("*prediction*.npz"))
    return [path for path in files if path.is_file()]


def _expected_prediction_checksum(
    manifest: Mapping[str, Any],
    path: Path,
    *,
    n_prediction_files: int,
) -> str | None:
    candidates: list[str] = []
    value = _nested_get(
        manifest,
        "prediction_checksum",
        "prediction_sha256",
        "artifacts.prediction_checksum",
        "artifacts.prediction_sha256",
        "artifacts.predictions_sha256",
    )
    if isinstance(value, Mapping):
        candidate = (
            value.get(path.name)
            or value.get(str(path))
            or value.get(path.as_posix())
        )
        if candidate is not None:
            candidates.append(str(candidate))
    elif value is not None and n_prediction_files == 1:
        candidates.append(str(value))
    files = manifest.get("files")
    if isinstance(files, Mapping) and files.get(path.name) is not None:
        candidates.append(str(files[path.name]))
    if not candidates:
        return None
    if any(re.fullmatch(r"[0-9a-f]{64}", item) is None for item in candidates):
        raise AnalysisContractError(
            f"Invalid declared checksum for prediction artifact {path.name}"
        )
    if len(set(candidates)) != 1:
        raise AnalysisContractError(
            f"Conflicting declared checksums for prediction artifact {path.name}"
        )
    return candidates[0]


def _npz_scalar(archive: Any, key: str) -> Any:
    if key not in archive.files:
        return None
    value = np.asarray(archive[key])
    if value.size != 1:
        raise AnalysisContractError(f"NPZ metadata key {key!r} must be scalar")
    return value.reshape(-1)[0].item()


def _first_npz_key(archive: Any, candidates: Iterable[str]) -> str | None:
    available = set(archive.files)
    return next((candidate for candidate in candidates if candidate in available), None)


def _evaluation_items(
    manifest: Mapping[str, Any],
    archive: Any,
) -> list[dict[str, Any]]:
    declared = manifest.get("evaluations")
    if declared is not None:
        if not isinstance(declared, list) or not all(
            isinstance(item, Mapping) for item in declared
        ):
            raise AnalysisContractError(
                "manifest evaluations must be a list of mappings"
            )
        return [dict(item) for item in declared]

    items: list[dict[str, Any]] = []
    for key in archive.files:
        match = re.fullmatch(
            r"(.+)__(partial|partial_gene|node|whole_node|block|spatial_block)"
            r"__r(\d+)__(?:prediction|predictions|y_pred)",
            key,
        )
        if match:
            split, mode, replicate = match.groups()
            prefix = key.rsplit("__", 1)[0]
            items.append(
                {
                    "split": split,
                    "mask_mode": mode,
                    "mask_replicate": int(replicate),
                    "prefix": prefix,
                }
            )
    if items:
        unique: dict[str, dict[str, Any]] = {}
        for item in items:
            unique[str(item["prefix"])] = item
        return [unique[key] for key in sorted(unique)]
    return [{}]


def _field_key(
    archive: Any,
    item: Mapping[str, Any],
    field: str,
    aliases: Sequence[str],
    prefix: str | None,
) -> str | None:
    explicit = item.get(f"{field}_key")
    if explicit is None:
        explicit = next(
            (
                item[f"{alias}_key"]
                for alias in aliases
                if f"{alias}_key" in item
            ),
            None,
        )
    keys_mapping = item.get("keys")
    if explicit is None and isinstance(keys_mapping, Mapping):
        explicit = keys_mapping.get(field)
        if explicit is None:
            explicit = next(
                (
                    keys_mapping[alias]
                    for alias in aliases
                    if alias in keys_mapping
                ),
                None,
            )
    if explicit is not None:
        if str(explicit) not in archive.files:
            raise AnalysisContractError(
                f"Declared NPZ key {explicit!r} for {field} is missing"
            )
        return str(explicit)
    candidates: list[str] = []
    if prefix:
        for alias in aliases:
            candidates.extend([f"{prefix}__{alias}", f"{prefix}/{alias}"])
    candidates.extend(aliases)
    return _first_npz_key(archive, candidates)


def _item_value(
    item: Mapping[str, Any],
    archive: Any,
    manifest: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: Any = None,
) -> Any:
    for name in names:
        if item.get(name) is not None:
            return item[name]
    for name in names:
        value = _npz_scalar(archive, name)
        if value is not None:
            return value
    for name in names:
        value = _nested_get(manifest, name, f"evaluation.{name}")
        if value is not None:
            return value
    return default


def _load_prediction_file(
    path: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    run_id: str,
    model: str,
    model_seed: int,
    graph: Mapping[str, Any],
    manifest_checksum: str,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    try:
        archive_context = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise AnalysisContractError(f"Could not open prediction NPZ {path}") from exc
    with archive_context as archive:
        for item in _evaluation_items(manifest, archive):
            prefix_value = (
                item.get("prefix")
                or item.get("key_prefix")
                or item.get("key")
            )
            prefix = None if prefix_value is None else str(prefix_value)
            prediction_key = _field_key(
                archive,
                item,
                "prediction",
                ("prediction", "predictions", "y_pred"),
                prefix,
            )
            target_key = _field_key(
                archive,
                item,
                "y_true",
                ("y_true", "target", "target_expression"),
                prefix,
            )
            mask_key = _field_key(
                archive,
                item,
                "mask",
                ("mask", "gene_mask"),
                prefix,
            )
            block_key = _field_key(
                archive,
                item,
                "block_ids",
                ("block_ids", "macroblock_ids", "spatial_block_ids"),
                prefix,
            )
            missing = [
                name
                for name, key in (
                    ("prediction", prediction_key),
                    ("y_true", target_key),
                    ("mask", mask_key),
                    ("block_ids", block_key),
                )
                if key is None
            ]
            if missing:
                raise AnalysisContractError(
                    f"{path} evaluation {prefix!r} lacks {', '.join(missing)}"
                )
            mode_value = _item_value(
                item,
                archive,
                manifest,
                ("mask_mode", "mode"),
            )
            split_value = _item_value(
                item,
                archive,
                manifest,
                ("split",),
                default="test",
            )
            replicate_value = _item_value(
                item,
                archive,
                manifest,
                ("mask_replicate", "replicate"),
                default=0,
            )
            if mode_value is None and prefix:
                prefix_match = re.fullmatch(
                    r"(.+)__(partial|partial_gene|node|whole_node|block|"
                    r"spatial_block)__r(\d+)",
                    prefix,
                )
                if prefix_match:
                    split_value, mode_value, replicate_value = prefix_match.groups()
            if mode_value is None:
                raise AnalysisContractError(
                    f"Mask mode is missing for prediction evaluation in {path}"
                )
            cell_key = _field_key(
                archive,
                item,
                "cell_ids",
                ("cell_ids", "cell_keys"),
                prefix,
            )
            try:
                records.append(
                    PredictionRecord(
                        run_id=run_id,
                        condition=str(graph["condition"]),
                        model=model,
                        model_seed=model_seed,
                        graph_id=str(graph["graph_id"]),
                        graph_kind=str(graph["graph_kind"]),
                        base_graph_id=graph.get("base_graph_id"),
                        graph_config=dict(graph.get("graph_config", {})),
                        edge_control=str(graph["edge_control"]),
                        split=str(split_value),
                        mask_mode=_normalise_mode(mode_value),
                        mask_replicate=int(replicate_value),
                        y_true=np.asarray(archive[target_key]),
                        prediction=np.asarray(archive[prediction_key]),
                        mask=np.asarray(archive[mask_key]),
                        block_ids=np.asarray(archive[block_key]),
                        cell_ids=(
                            None
                            if cell_key is None
                            else np.asarray(archive[cell_key])
                        ),
                        manifest_path=str(manifest_path),
                        manifest_checksum=manifest_checksum,
                    )
                )
            except ValueError as exc:
                raise AnalysisContractError(
                    f"Invalid prediction evaluation {prefix!r} in {path}: {exc}"
                ) from exc
    return records


def _qc_record(
    value: Mapping[str, Any],
    *,
    fallback_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    qc_value = value.get("qc") if isinstance(value.get("qc"), Mapping) else value
    if not isinstance(qc_value, Mapping):
        return None
    known = {
        "n_nodes",
        "n_directed_edges",
        "n_undirected_edges",
        "n_components",
        "n_isolated_nodes",
        "mean_degree",
        "median_degree",
        "p95_degree",
        "max_degree",
        "edge_distance_mean_um",
        "edge_distance_p50_um",
        "edge_distance_p95_um",
        "edge_distance_max_um",
        "zero_distance_edges",
        "cap_hit_rate",
        "fov_seam_edges",
        "fov_seam_fraction",
    }
    if not any(name in qc_value for name in known):
        return None
    fallback_graph = fallback_graph or {}
    config = value.get("config")
    if not isinstance(config, Mapping):
        config = fallback_graph.get("graph_config", {})
    graph_id = (
        value.get("graph_id")
        or value.get("id")
        or fallback_graph.get("graph_id")
    )
    if graph_id is None:
        return None
    kind = _normalise_graph_kind(
        value.get("graph_kind")
        or value.get("kind")
        or fallback_graph.get("graph_kind")
    )
    record: dict[str, Any] = {
        "graph_id": str(graph_id),
        "graph_kind": kind,
        "base_graph_id": (
            value.get("base_graph_id")
            or fallback_graph.get("base_graph_id")
        ),
    }
    for name in ("k", "radius_um", "symmetry", "min_distance_um"):
        record[name] = value.get(name, config.get(name))
    for name in sorted(known):
        record[name] = qc_value.get(name)
    return record


def _graph_qc_from_manifest(
    manifest: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    if isinstance(manifest.get("graph_qc"), Mapping):
        candidates.append(manifest["graph_qc"])
    graph_value = manifest.get("graph")
    if isinstance(graph_value, Mapping) and isinstance(
        graph_value.get("qc"), Mapping
    ):
        candidates.append(
            {
                **dict(graph_value),
                "qc": graph_value["qc"],
            }
        )
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        record = _qc_record(candidate, fallback_graph=graph)
        if record is not None:
            records.append(record)
    return records


def _external_qc_candidates(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    if isinstance(value.get("graphs"), list):
        return [
            item for item in value["graphs"] if isinstance(item, Mapping)
        ]
    if "graph_id" in value or "qc" in value or "n_nodes" in value:
        return [value]
    candidates: list[Mapping[str, Any]] = []
    for graph_id, record in value.items():
        if isinstance(record, Mapping):
            candidates.append({"graph_id": graph_id, **dict(record)})
    return candidates


def _deduplicate_qc(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_record in records:
        record = dict(raw_record)
        key = (str(record["graph_id"]), str(record["graph_kind"]))
        if key in deduplicated:
            previous = deduplicated[key]
            for field, value in record.items():
                if value is None:
                    continue
                if previous.get(field) is not None and previous[field] != value:
                    raise AnalysisContractError(
                        f"Inconsistent graph QC for {key[0]!r}: field {field}"
                    )
                previous[field] = value
        else:
            deduplicated[key] = record
    return tuple(
        deduplicated[key] for key in sorted(deduplicated)
    )


def load_run_collection(
    runs_dir: str | os.PathLike[str],
    *,
    graph_qc_dir: str | os.PathLike[str] | None = None,
) -> RunCollection:
    """Load isolated run manifests and all prediction records beneath a root."""

    root = Path(runs_dir)
    if not root.exists():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    if (root / "manifest.json").is_file():
        manifest_paths = [root / "manifest.json"]
    else:
        manifest_paths = sorted(root.rglob("manifest.json"))
    records: list[PredictionRecord] = []
    graph_qc: list[Mapping[str, Any]] = []
    failed_runs: list[Mapping[str, Any]] = []
    excluded_runs: list[Mapping[str, Any]] = []
    checksums: dict[str, str] = {}
    prediction_checksums: dict[str, str] = {}
    run_audits: dict[str, dict[str, Any]] = {}

    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisContractError(
                f"Invalid run manifest {manifest_path}"
            ) from exc
        if not isinstance(manifest, Mapping):
            continue
        model_hint = _nested_get(
            manifest,
            "model_name",
            "model",
            "config.model.name",
        )
        if model_hint is None:
            # This excludes fixed-mask and other non-run manifests when the
            # caller points analysis at the workflow-wide results directory.
            continue
        run_dir = manifest_path.parent
        run_id = str(manifest.get("run_id") or run_dir.name)
        status_value = manifest.get("status")
        if not isinstance(status_value, str) or not status_value.strip():
            raise AnalysisContractError(
                f"Run manifest lacks an explicit status: {manifest_path}"
            )
        status = status_value.strip().lower()
        checksum = _file_sha256(manifest_path)
        if run_id in checksums and checksums[run_id] != checksum:
            raise AnalysisContractError(
                f"Run ID {run_id!r} is reused by different manifests"
            )
        checksums[run_id] = checksum
        if status in {
            "failed",
            "failure",
            "error",
            "aborted",
            "cancelled",
        }:
            failed_runs.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "manifest_checksum": checksum,
                }
            )
            continue
        if status != "complete":
            excluded_runs.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "reason": "run_not_complete",
                    "conclusion_eligible": False,
                    "manifest_checksum": checksum,
                }
            )
            continue
        artifact_kind = manifest.get("artifact_kind")
        if artifact_kind is not None and (
            artifact_kind != EXPERIMENT_RUN_ARTIFACT_KIND
        ):
            raise AnalysisContractError(
                f"Unsupported run artifact kind in {manifest_path}: "
                f"{artifact_kind!r}"
            )
        if artifact_kind == EXPERIMENT_RUN_ARTIFACT_KIND:
            if manifest.get("format_version") != 1:
                raise AnalysisContractError(
                    f"Unsupported experiment manifest version in {manifest_path}"
                )
            declared_content_checksum = manifest.get(
                "manifest_content_sha256"
            )
            if (
                not isinstance(declared_content_checksum, str)
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    declared_content_checksum,
                )
                is None
                or declared_content_checksum
                != _manifest_content_sha256(manifest)
            ):
                raise AnalysisContractError(
                    f"Run manifest content checksum mismatch: {manifest_path}"
                )
            evaluations = manifest.get("evaluations")
            required_evaluation_keys = {
                "split",
                "mask_mode",
                "mask_replicate",
                "prefix",
                "prediction_key",
                "y_true_key",
                "mask_key",
                "block_ids_key",
                "cell_ids_key",
            }
            if not isinstance(evaluations, list) or any(
                not isinstance(item, Mapping)
                or not required_evaluation_keys.issubset(item)
                for item in evaluations
            ):
                raise AnalysisContractError(
                    f"Run evaluation declaration schema is invalid: "
                    f"{manifest_path}"
                )
            declared_splits = {
                str(item["split"])
                for item in evaluations
            }
            if manifest["sealed_test_opened"] and (
                CONCLUSION_SPLIT not in declared_splits
            ):
                raise AnalysisContractError(
                    f"Opened-test run has no declared test predictions: "
                    f"{manifest_path}"
                )
            if (
                not manifest["sealed_test_opened"]
                and CONCLUSION_SPLIT in declared_splits
            ):
                raise AnalysisContractError(
                    f"Sealed screening run declares test predictions: "
                    f"{manifest_path}"
                )
            artifacts = manifest.get("artifacts")
            prediction_name = (
                artifacts.get("predictions")
                if isinstance(artifacts, Mapping)
                else None
            )
            files = manifest.get("files")
            if manifest["sealed_test_opened"] and (
                not isinstance(prediction_name, str)
                or not isinstance(files, Mapping)
                or prediction_name not in files
            ):
                raise AnalysisContractError(
                    f"Opened-test run lacks a checksummed prediction artifact: "
                    f"{manifest_path}"
                )
        sealed_test_opened = manifest.get("sealed_test_opened")
        if not isinstance(sealed_test_opened, bool):
            raise AnalysisContractError(
                f"Complete run lacks a boolean sealed-test declaration: "
                f"{manifest_path}"
            )
        if sealed_test_opened is False:
            excluded_runs.append(
                {
                    "run_id": run_id,
                    "status": status,
                    "reason": "sealed_test_not_opened",
                    "phase": "screening_or_validation",
                    "conclusion_eligible": False,
                    "manifest_checksum": checksum,
                }
            )
            continue
        prediction_files = _prediction_files(run_dir, manifest)
        if not prediction_files:
            raise AnalysisContractError(
                f"Conclusion-bearing run {run_id!r} has no prediction NPZ"
            )
        model, model_seed, graph = _manifest_model_and_seed(
            manifest,
            manifest_path,
        )
        declared_evaluations = manifest.get("evaluations")
        evaluation_keys = []
        if isinstance(declared_evaluations, list):
            evaluation_keys = [
                (
                    str(item.get("split")),
                    _normalise_mode(item.get("mask_mode")),
                    int(item.get("mask_replicate")),
                )
                for item in declared_evaluations
                if isinstance(item, Mapping)
            ]
        if run_id in run_audits:
            raise AnalysisContractError(f"Run ID {run_id!r} is duplicated")
        run_audits[run_id] = {
            "run_id": run_id,
            "condition": graph["condition"],
            "model": model,
            "model_seed": model_seed,
            "edge_control": graph["edge_control"],
            "graph_id": graph["graph_id"],
            "graph_kind": graph["graph_kind"],
            "base_graph_id": graph["base_graph_id"],
            "artifact_kind": artifact_kind,
            "manifest_path": str(manifest_path),
            "manifest_checksum": checksum,
            "standards_lock": manifest.get("standards_lock"),
            "prepared_artifact": manifest.get("prepared_artifact"),
            "config": manifest.get("config"),
            "graph": manifest.get("graph"),
            "evaluation_keys": tuple(evaluation_keys),
        }
        graph_qc.extend(_graph_qc_from_manifest(manifest, graph))
        for prediction_file in prediction_files:
            prediction_checksum = _file_sha256(prediction_file)
            expected_prediction_checksum = _expected_prediction_checksum(
                manifest,
                prediction_file,
                n_prediction_files=len(prediction_files),
            )
            if (
                expected_prediction_checksum is not None
                and prediction_checksum != expected_prediction_checksum
            ):
                raise AnalysisContractError(
                    f"Prediction checksum mismatch for {prediction_file}"
                )
            prediction_key = f"{run_id}/{prediction_file.name}"
            if (
                prediction_key in prediction_checksums
                and prediction_checksums[prediction_key] != prediction_checksum
            ):
                raise AnalysisContractError(
                    f"Prediction artifact key is reused: {prediction_key}"
                )
            prediction_checksums[prediction_key] = prediction_checksum
            records.extend(
                _load_prediction_file(
                    prediction_file,
                    manifest,
                    manifest_path,
                    run_id=run_id,
                    model=model,
                    model_seed=model_seed,
                    graph=graph,
                    manifest_checksum=checksum,
                )
            )

    if graph_qc_dir is not None:
        qc_root = Path(graph_qc_dir)
        paths = [qc_root] if qc_root.is_file() else sorted(qc_root.rglob("*.json"))
        for path in paths:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AnalysisContractError(f"Invalid graph-QC JSON {path}") from exc
            for candidate in _external_qc_candidates(value):
                candidate_with_id = dict(candidate)
                candidate_with_id.setdefault("graph_id", path.stem)
                record = _qc_record(candidate_with_id)
                if record is not None:
                    graph_qc.append(record)
    if not records:
        raise AnalysisContractError(
            "No conclusion-eligible sealed-test prediction records were discovered"
        )
    return RunCollection(
        records=tuple(records),
        graph_qc=_deduplicate_qc(graph_qc),
        failed_runs=tuple(failed_runs),
        excluded_runs=tuple(excluded_runs),
        manifest_checksums=checksums,
        prediction_checksums=prediction_checksums,
        run_audits=run_audits,
    )


def _arrays_equal(first: np.ndarray, second: np.ndarray) -> bool:
    if first.shape != second.shape or first.dtype.kind != second.dtype.kind:
        return False
    if np.issubdtype(first.dtype, np.number):
        return bool(np.array_equal(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _assert_aligned(
    reference: PredictionRecord | EnsembleEvaluation,
    candidate: PredictionRecord | EnsembleEvaluation,
    *,
    context: str,
) -> None:
    for name in ("y_true", "mask", "block_ids"):
        if not _arrays_equal(
            np.asarray(getattr(reference, name)),
            np.asarray(getattr(candidate, name)),
        ):
            raise AnalysisContractError(
                f"Unpaired {name} in {context}; fixed masks/row order differ"
            )
    first_cells = reference.cell_ids
    second_cells = candidate.cell_ids
    if (first_cells is None) != (second_cells is None):
        raise AnalysisContractError(f"Only one side supplies cell_ids in {context}")
    if first_cells is not None and not _arrays_equal(
        np.asarray(first_cells),
        np.asarray(second_cells),
    ):
        raise AnalysisContractError(f"cell_ids differ in {context}")


def ensemble_run_records(
    records: Sequence[PredictionRecord],
    *,
    expected_seeds: int = 5,
) -> tuple[EnsembleEvaluation, ...]:
    """Group isolated seeds and validate paired targets/masks before averaging."""

    expected_seeds = int(expected_seeds)
    if expected_seeds <= 0:
        raise ValueError("expected_seeds must be positive")
    groups: dict[tuple[Any, ...], list[PredictionRecord]] = defaultdict(list)
    for record in records:
        key = (
            record.condition,
            record.model,
            record.graph_id,
            record.graph_kind,
            record.base_graph_id,
            record.edge_control,
            record.split,
            record.mask_mode,
            record.mask_replicate,
        )
        groups[key].append(record)

    ensembles: list[EnsembleEvaluation] = []
    for key in sorted(groups, key=lambda value: tuple(str(item) for item in value)):
        group = sorted(groups[key], key=lambda item: (item.model_seed, item.run_id))
        seeds = [item.model_seed for item in group]
        if len(seeds) != len(set(seeds)):
            raise AnalysisContractError(
                f"Duplicate model seed in ensemble group {key}"
            )
        reference = group[0]
        for candidate in group[1:]:
            _assert_aligned(
                reference,
                candidate,
                context=(
                    f"{reference.model}/{reference.graph_id}/"
                    f"{reference.evaluation_key}"
                ),
            )
        ensembles.append(
            EnsembleEvaluation(
                condition=reference.condition,
                model=reference.model,
                graph_id=reference.graph_id,
                graph_kind=reference.graph_kind,
                base_graph_id=reference.base_graph_id,
                graph_config=dict(reference.graph_config),
                edge_control=reference.edge_control,
                split=reference.split,
                mask_mode=reference.mask_mode,
                mask_replicate=reference.mask_replicate,
                seeds=tuple(seeds),
                run_ids=tuple(item.run_id for item in group),
                y_true=reference.y_true,
                predictions=np.stack(
                    [item.prediction for item in group],
                    axis=0,
                ),
                mask=reference.mask,
                block_ids=reference.block_ids,
                cell_ids=reference.cell_ids,
                expected_seeds=expected_seeds,
            )
        )
    return tuple(ensembles)


def _block_label_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return "__nan__"
    return f"{type(value).__name__}:{value!r}"


def _block_label_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _huber_by_block(evaluation: EnsembleEvaluation) -> list[dict[str, Any]]:
    prediction = evaluation.ensemble_prediction
    target = np.asarray(evaluation.y_true, dtype=np.float64)
    mask = np.asarray(evaluation.mask, dtype=bool)
    blocks = np.asarray(evaluation.block_ids)
    order: list[str] = []
    rows_by_block: dict[str, list[int]] = {}
    labels: dict[str, Any] = {}
    for index, raw_label in enumerate(blocks):
        key = _block_label_key(raw_label)
        if key not in rows_by_block:
            order.append(key)
            rows_by_block[key] = []
            labels[key] = _block_label_value(raw_label)
        rows_by_block[key].append(index)
    output: list[dict[str, Any]] = []
    for key in order:
        rows = np.asarray(rows_by_block[key], dtype=np.int64)
        valid = (
            mask[rows]
            & np.isfinite(target[rows])
            & np.isfinite(prediction[rows])
        )
        if not np.any(valid):
            continue
        difference = prediction[rows][valid] - target[rows][valid]
        absolute = np.abs(difference)
        loss = np.where(
            absolute <= 1.0,
            0.5 * difference**2,
            absolute - 0.5,
        )
        output.append(
            {
                "block_id": labels[key],
                "block_key": key,
                "loss": float(loss.mean()),
                "n_masked": int(valid.sum()),
            }
        )
    return output


def summarise_model_losses(
    ensembles: Sequence[EnsembleEvaluation],
    *,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
) -> list[dict[str, Any]]:
    """Average mask replicates within blocks, then summarise across blocks."""

    groups: dict[tuple[str, ...], list[EnsembleEvaluation]] = defaultdict(list)
    for item in ensembles:
        groups[
            (
                item.condition,
                item.model,
                item.graph_id,
                item.graph_kind,
                item.edge_control,
                item.split,
                item.mask_mode,
            )
        ].append(item)
    rows: list[dict[str, Any]] = []
    for group_number, key in enumerate(sorted(groups)):
        evaluations = groups[key]
        block_values: dict[str, list[float]] = defaultdict(list)
        block_labels: dict[str, Any] = {}
        total_masked = 0
        for evaluation in evaluations:
            for block in _huber_by_block(evaluation):
                block_values[block["block_key"]].append(block["loss"])
                block_labels[block["block_key"]] = block["block_id"]
                total_masked += block["n_masked"]
        averaged = np.asarray(
            [
                np.mean(block_values[block_key])
                for block_key in sorted(block_values)
            ],
            dtype=np.float64,
        )
        if averaged.size == 0:
            continue
        ci = block_bootstrap_ci(
            averaged,
            confidence_level=confidence_level,
            n_resamples=n_bootstrap,
            seed=int(bootstrap_seed) + group_number,
        )
        seed_counts = sorted({len(item.seeds) for item in evaluations})
        rows.append(
            {
                "scope": "within_core_descriptive",
                "condition": key[0],
                "condition_label": FINAL_CONDITION_LABELS.get(key[0], key[0]),
                "model": key[1],
                "graph_id": key[2],
                "graph_kind": key[3],
                "edge_control": key[4],
                "base_graph_id": evaluations[0].base_graph_id,
                "graph_k": evaluations[0].graph_config.get("k"),
                "graph_radius_um": evaluations[0].graph_config.get("radius_um"),
                "graph_symmetry": evaluations[0].graph_config.get("symmetry"),
                "split": key[5],
                "mask_mode": key[6],
                "mask_replicates": len(evaluations),
                "model_seed_count_min": min(seed_counts),
                "model_seed_count_max": max(seed_counts),
                "seed_complete": all(item.seed_complete for item in evaluations),
                "n_spatial_blocks": int(averaged.size),
                "n_masked_across_replicates": total_masked,
                "huber_loss": float(averaged.mean()),
                "huber_ci_lower": ci["lower"],
                "huber_ci_upper": ci["upper"],
                "block_loss_sd": (
                    float(averaged.std(ddof=1))
                    if averaged.size > 1
                    else 0.0
                ),
                "aggregation": (
                    "seed mean -> within-block loss -> mask-replicate mean "
                    "-> equal-weight block mean"
                ),
            }
        )
    return rows


def _select_one(
    ensembles: Sequence[EnsembleEvaluation],
    *,
    condition: str,
    split: str,
    mode: str,
    replicate: int,
    graph_id: str | None = None,
    base_graph_id: str | None = None,
) -> EnsembleEvaluation | None:
    matches = [
        item
        for item in ensembles
        if item.condition == condition
        and item.split == split
        and item.mask_mode == mode
        and item.mask_replicate == replicate
        and (graph_id is None or item.graph_id == graph_id)
        and (
            base_graph_id is None
            or item.base_graph_id == base_graph_id
            or base_graph_id in item.graph_id
        )
    ]
    if not matches and condition == "g1_rewired" and base_graph_id is not None:
        # A single rewired condition is unambiguous even if an older manifest
        # omitted the source graph ID.
        fallback = [
            item
            for item in ensembles
            if item.condition == condition
            and item.split == split
            and item.mask_mode == mode
            and item.mask_replicate == replicate
        ]
        if len(fallback) == 1:
            matches = fallback
    if len(matches) > 1:
        raise AnalysisContractError(
            f"Ambiguous {condition}/{split}/{mode}/r{replicate} evaluation"
        )
    return matches[0] if matches else None


def _paired_ratio_ci(
    baseline: np.ndarray,
    spatial: np.ndarray,
    *,
    confidence_level: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if baseline.size == 1:
        if abs(baseline[0]) <= np.finfo(np.float64).eps:
            return float("nan"), float("nan")
        value = (baseline[0] - spatial[0]) / baseline[0]
        return float(value), float(value)
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(n_bootstrap), dtype=np.float64)
    for start in range(0, int(n_bootstrap), 4096):
        stop = min(start + 4096, int(n_bootstrap))
        indices = rng.integers(
            0,
            baseline.size,
            size=(stop - start, baseline.size),
        )
        baseline_mean = baseline[indices].mean(axis=1)
        spatial_mean = spatial[indices].mean(axis=1)
        values[start:stop] = np.divide(
            baseline_mean - spatial_mean,
            baseline_mean,
            out=np.full(stop - start, np.nan),
            where=np.abs(baseline_mean) > np.finfo(np.float64).eps,
        )
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def aggregate_paired_gain(
    ensembles: Sequence[EnsembleEvaluation],
    *,
    comparison: str,
    baseline_condition: str,
    spatial_condition: str,
    split: str,
    mask_mode: str,
    locked_graph_id: str,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any] | None:
    """Compute a paired gain after technical seed/mask aggregation."""

    mode = _normalise_mode(mask_mode)
    replicates = sorted(
        {
            item.mask_replicate
            for item in ensembles
            if item.split == split and item.mask_mode == mode
        }
    )
    block_losses: dict[str, dict[str, Any]] = {}
    used_replicates: list[int] = []
    seed_complete = True
    baseline_seed_counts: set[int] = set()
    spatial_seed_counts: set[int] = set()
    replicate_pair_complete = True
    for replicate in replicates:
        baseline = _select_one(
            ensembles,
            condition=baseline_condition,
            split=split,
            mode=mode,
            replicate=replicate,
            graph_id=(
                locked_graph_id
                if baseline_condition in {"g1_true", "g2_true"}
                else None
            ),
            base_graph_id=(
                locked_graph_id
                if baseline_condition == "g1_rewired"
                else None
            ),
        )
        spatial = _select_one(
            ensembles,
            condition=spatial_condition,
            split=split,
            mode=mode,
            replicate=replicate,
            graph_id=(
                locked_graph_id
                if spatial_condition in {"g1_true", "g2_true"}
                else None
            ),
            base_graph_id=(
                locked_graph_id
                if spatial_condition == "g1_rewired"
                else None
            ),
        )
        if (baseline is None) != (spatial is None):
            replicate_pair_complete = False
        if baseline is None or spatial is None:
            continue
        _assert_aligned(
            baseline,
            spatial,
            context=f"{comparison}/{split}/{mode}/r{replicate}",
        )
        if baseline.seeds != spatial.seeds:
            raise AnalysisContractError(
                f"Model-seed sets differ in {comparison}/{split}/{mode}/"
                f"r{replicate}: {baseline.seeds} versus {spatial.seeds}"
            )
        seed_complete = (
            seed_complete and baseline.seed_complete and spatial.seed_complete
        )
        baseline_seed_counts.add(len(baseline.seeds))
        spatial_seed_counts.add(len(spatial.seeds))
        # This produces paired per-block losses from seed-ensemble predictions.
        replicate_gain = paired_spatial_gain(
            baseline.y_true,
            baseline.predictions,
            spatial.predictions,
            baseline.mask,
            baseline.block_ids,
            loss="huber",
            confidence_level=confidence_level,
            n_bootstrap=min(200, int(n_bootstrap)),
            bootstrap_seed=int(bootstrap_seed) + replicate,
            n_sign_flips=1_000,
            sign_flip_seed=int(bootstrap_seed) + 10_000 + replicate,
        )
        used_replicates.append(replicate)
        for block in replicate_gain["blocks"]:
            if not (
                math.isfinite(block["baseline_loss"])
                and math.isfinite(block["spatial_loss"])
            ):
                continue
            key = _block_label_key(block["block_id"])
            entry = block_losses.setdefault(
                key,
                {
                    "block_id": block["block_id"],
                    "baseline": [],
                    "spatial": [],
                    "n_masked": [],
                },
            )
            entry["baseline"].append(block["baseline_loss"])
            entry["spatial"].append(block["spatial_loss"])
            entry["n_masked"].append(block["n_masked"])
    if not used_replicates:
        return None
    block_records: list[dict[str, Any]] = []
    for key in sorted(block_losses):
        values = block_losses[key]
        baseline_loss = float(np.mean(values["baseline"]))
        spatial_loss = float(np.mean(values["spatial"]))
        delta = baseline_loss - spatial_loss
        block_records.append(
            {
                "block_id": values["block_id"],
                "mask_replicates_observed": len(values["baseline"]),
                "mean_masked_entries": float(np.mean(values["n_masked"])),
                "baseline_loss": baseline_loss,
                "spatial_loss": spatial_loss,
                "delta": delta,
                "relative_gain": (
                    delta / baseline_loss
                    if abs(baseline_loss) > np.finfo(np.float64).eps
                    else float("nan")
                ),
            }
        )
    if not block_records:
        return None
    baseline_values = np.asarray(
        [item["baseline_loss"] for item in block_records],
        dtype=np.float64,
    )
    spatial_values = np.asarray(
        [item["spatial_loss"] for item in block_records],
        dtype=np.float64,
    )
    differences = baseline_values - spatial_values
    baseline_mean = float(baseline_values.mean())
    spatial_mean = float(spatial_values.mean())
    delta = float(differences.mean())
    relative = (
        delta / baseline_mean
        if abs(baseline_mean) > np.finfo(np.float64).eps
        else float("nan")
    )
    delta_ci = block_bootstrap_ci(
        differences,
        confidence_level=confidence_level,
        n_resamples=n_bootstrap,
        seed=bootstrap_seed,
    )
    relative_lower, relative_upper = _paired_ratio_ci(
        baseline_values,
        spatial_values,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    sign_flip = paired_sign_flip_test(
        differences,
        alternative="greater",
        seed=int(bootstrap_seed) + 1,
    )
    return {
        "scope": "within_core_descriptive",
        "comparison": comparison,
        "baseline_condition": baseline_condition,
        "spatial_condition": spatial_condition,
        "baseline_model": baseline.model,
        "spatial_model": spatial.model,
        "positive_favors": spatial_condition,
        "locked_graph_id": locked_graph_id,
        "split": split,
        "mask_mode": mode,
        "mask_replicates": used_replicates,
        "n_mask_replicates": len(used_replicates),
        "baseline_seed_counts": sorted(baseline_seed_counts),
        "spatial_seed_counts": sorted(spatial_seed_counts),
        "seed_complete": seed_complete,
        "replicate_pair_complete": replicate_pair_complete,
        "n_spatial_blocks": len(block_records),
        "baseline_loss": baseline_mean,
        "spatial_loss": spatial_mean,
        "delta": delta,
        "relative_gain": float(relative),
        "relative_gain_percent": float(relative * 100),
        "delta_ci_lower": delta_ci["lower"],
        "delta_ci_upper": delta_ci["upper"],
        "relative_gain_ci_lower": relative_lower,
        "relative_gain_ci_upper": relative_upper,
        "sign_flip_p_value": sign_flip["p_value"],
        "sign_flip_method": sign_flip["method"],
        "aggregation": (
            "model-seed mean before scoring; mask-replicate mean within each "
            "spatial block; equal-weight spatial-block mean/inference"
        ),
        "inference_unit": "spatial_block",
        "blocks": block_records,
    }


def _find_gain(
    gains: Sequence[Mapping[str, Any]],
    comparison: str,
    mode: str,
) -> Mapping[str, Any] | None:
    return next(
        (
            gain
            for gain in gains
            if gain["comparison"] == comparison
            and gain["mask_mode"] == mode
        ),
        None,
    )


def evaluate_acceptance_gate(
    gains: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: int = 5,
    minimum_relative_gain: float = 0.02,
    required_split: str = CONCLUSION_SPLIT,
) -> dict[str, Any]:
    """Apply the prespecified four-part within-core H1 gate."""

    if required_split != CONCLUSION_SPLIT:
        raise AnalysisContractError(
            "The acceptance gate cannot be retargeted from the sealed "
            f"{CONCLUSION_SPLIT!r} split."
        )
    primary = _find_gain(gains, "B0_minus_G1", "node")
    block = _find_gain(gains, "B0_minus_G1", "block")
    rewired = _find_gain(gains, "rewired_G1_minus_true_G1", "node")
    # The rewired block-mask contrast is a prespecified diagnostic, not a fifth
    # condition of the four-part H1 gate.
    required = [primary, block, rewired]
    complete_inputs = all(item is not None for item in required)
    conclusion_split_complete = complete_inputs and all(
        item.get("split") == required_split
        for item in required
        if item is not None
    )
    complete_seeds = complete_inputs and all(
        bool(item["seed_complete"])
        and item["baseline_seed_counts"] == [int(expected_seeds)]
        and item["spatial_seed_counts"] == [int(expected_seeds)]
        for item in required
        if item is not None
    )
    complete_mask_pairs = complete_inputs and all(
        bool(item["replicate_pair_complete"])
        for item in required
        if item is not None
    )
    criteria = [
        {
            "criterion": "whole_node_relative_gain_at_least_2pct",
            "passed": bool(
                conclusion_split_complete
                and primary is not None
                and math.isfinite(primary["relative_gain"])
                and primary["relative_gain"] >= minimum_relative_gain
            ),
            "observed": (
                None if primary is None else primary["relative_gain"]
            ),
            "threshold": minimum_relative_gain,
        },
        {
            "criterion": "whole_node_delta_ci_lower_above_zero",
            "passed": bool(
                conclusion_split_complete
                and primary is not None
                and math.isfinite(primary["delta_ci_lower"])
                and primary["delta_ci_lower"] > 0
            ),
            "observed": (
                None if primary is None else primary["delta_ci_lower"]
            ),
            "threshold": 0.0,
        },
        {
            "criterion": "true_g1_beats_rewired_g1_whole_node",
            "passed": bool(
                conclusion_split_complete
                and rewired is not None
                and math.isfinite(rewired["delta"])
                and rewired["delta"] > 0
            ),
            "observed": None if rewired is None else rewired["delta"],
            "threshold": 0.0,
        },
        {
            "criterion": "block_mask_gain_positive",
            "passed": bool(
                conclusion_split_complete
                and block is not None
                and math.isfinite(block["delta"])
                and block["delta"] > 0
            ),
            "observed": None if block is None else block["delta"],
            "threshold": 0.0,
        },
    ]
    gate_passed = all(item["passed"] for item in criteria)
    supported = bool(
        complete_inputs
        and conclusion_split_complete
        and complete_seeds
        and complete_mask_pairs
        and gate_passed
    )
    if (
        not complete_inputs
        or not conclusion_split_complete
        or not complete_seeds
        or not complete_mask_pairs
    ):
        outcome = "incomplete"
    elif supported:
        outcome = "supported_within_core"
    else:
        outcome = "not_supported_by_prespecified_gate"
    return {
        "outcome": outcome,
        "supported": supported,
        "input_complete": bool(complete_inputs),
        "conclusion_split_complete": bool(conclusion_split_complete),
        "required_conclusion_split": str(required_split),
        "five_seed_complete": bool(complete_seeds),
        "paired_mask_replicates_complete": bool(complete_mask_pairs),
        "expected_seeds_per_model": int(expected_seeds),
        "minimum_relative_gain": float(minimum_relative_gain),
        "criteria": criteria,
        "scope": WITHIN_CORE_SCOPE,
    }


def _infer_locked_graph_id(
    ensembles: Sequence[EnsembleEvaluation],
    locked_graph_id: str | None,
) -> str:
    candidates = sorted(
        {
            item.graph_id
            for item in ensembles
            if item.condition == "g1_true" and item.graph_kind == "true"
        }
    )
    if locked_graph_id is not None:
        if locked_graph_id not in candidates:
            raise AnalysisContractError(
                f"locked graph {locked_graph_id!r} is absent; candidates={candidates}"
            )
        return str(locked_graph_id)
    if len(candidates) != 1:
        raise AnalysisContractError(
            "Multiple/no true G1 graph candidates exist; pass a prespecified "
            "--locked-graph-id rather than selecting after viewing results"
        )
    return candidates[0]


def _job_manifest_value(
    audit: Mapping[str, Any],
    key: str,
) -> Any:
    config = audit.get("config")
    config = config if isinstance(config, Mapping) else {}
    model_config = config.get("model")
    model_config = model_config if isinstance(model_config, Mapping) else {}
    training_config = config.get("training")
    training_config = (
        training_config if isinstance(training_config, Mapping) else {}
    )
    run_config = config.get("run")
    run_config = run_config if isinstance(run_config, Mapping) else {}
    graph = audit.get("graph")
    graph = graph if isinstance(graph, Mapping) else {}
    graph_config = graph.get("config")
    graph_config = graph_config if isinstance(graph_config, Mapping) else {}
    if key == "model":
        return model_config.get("name")
    if key == "seed":
        return audit.get("model_seed")
    if key in {"hidden_dim", "graph_layers", "edge_embedding_dim"}:
        return model_config.get(key)
    if key in {"curriculum", "max_epochs", "patience", "amp"}:
        return training_config.get(key)
    if key in {"k", "radius_um", "symmetry", "min_distance_um"}:
        return graph_config.get(key)
    if key == "edge_control":
        return graph.get("edge_control", run_config.get("edge_control"))
    if key == "rewired":
        return run_config.get("rewired", graph.get("kind") == "rewired")
    if key in {"rewire_seed", "swaps_per_edge"}:
        rewire = graph.get("rewire")
        rewire = rewire if isinstance(rewire, Mapping) else {}
        field = "seed" if key == "rewire_seed" else "swaps_per_edge"
        if key == "rewire_seed" and not rewire:
            return graph.get("edge_control_seed")
        return rewire.get(field)
    return None


def _normalised_job_value(key: str, value: Any) -> Any:
    if key == "model":
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())
    if key == "edge_control":
        return _normalise_edge_control(value)
    if key in {"radius_um", "min_distance_um", "swaps_per_edge"}:
        return float(value)
    if key in {
        "seed",
        "hidden_dim",
        "graph_layers",
        "edge_embedding_dim",
        "max_epochs",
        "patience",
        "k",
        "rewire_seed",
    }:
        return int(value)
    if key in {"amp", "rewired"}:
        return bool(value)
    return value


def _assert_manifest_matches_job(
    audit: Mapping[str, Any],
    job: Mapping[str, Any],
) -> None:
    for key, expected in job.items():
        actual = _job_manifest_value(audit, str(key))
        if actual is None:
            raise AnalysisContractError(
                f"Run {audit['run_id']} lacks locked job field {key!r}"
            )
        try:
            actual_value = _normalised_job_value(str(key), actual)
            expected_value = _normalised_job_value(str(key), expected)
        except (TypeError, ValueError) as exc:
            raise AnalysisContractError(
                f"Run {audit['run_id']} has invalid locked field {key!r}"
            ) from exc
        if actual_value != expected_value:
            raise AnalysisContractError(
                f"Run {audit['run_id']} differs from its locked job at "
                f"{key}: {actual_value!r} != {expected_value!r}"
            )


def _load_prepared_evaluation_contract(
    prepared: Mapping[str, Any],
) -> PreparedEvaluationContract:
    required = {
        "path",
        "artifact_id",
        "manifest_sha256",
        "split_id",
        "validation_mask_bundle_id",
        "test_mask_bundle_id",
    }
    if not required.issubset(prepared):
        raise AnalysisContractError(
            "Run prepared-artifact provenance is incomplete"
        )
    root = Path(str(prepared["path"])).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise AnalysisContractError(
            f"Prepared-artifact manifest was not found: {manifest_path}"
        )
    try:
        manifest, arrays, loaded_bundles = load_prepared_artifact(
            root,
            load_arrays=True,
        )
    except (
        ArtifactContractError,
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        raise AnalysisContractError(
            f"Prepared artifact could not be verified: {exc}"
        ) from exc
    if arrays is None:
        raise AnalysisContractError("Prepared canonical arrays were not loaded")
    if _file_sha256(manifest_path) != prepared["manifest_sha256"]:
        raise AnalysisContractError(
            "Prepared-artifact manifest checksum differs from run provenance"
        )
    fixed_masks = manifest.get("fixed_masks")
    bundles = (
        fixed_masks.get("bundles")
        if isinstance(fixed_masks, Mapping)
        else None
    )
    split_record = manifest.get("split")
    if (
        manifest.get("artifact_kind")
        != "normal_true_tissue_spatial_benchmark_preparation"
        or manifest.get("format_version") != 1
        or manifest.get("artifact_id") != prepared["artifact_id"]
        or not isinstance(split_record, Mapping)
        or split_record.get("split_id") != prepared["split_id"]
        or not isinstance(bundles, Mapping)
    ):
        raise AnalysisContractError(
            "Prepared artifact, split, or fixed-mask provenance disagrees"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise AnalysisContractError(
            "Prepared manifest lacks file checksums"
        )
    required_arrays = {
        "target_expression",
        "macroblock_ids",
        "split_labels",
        "validation_node_index",
        "test_node_index",
    }
    missing_arrays = sorted(required_arrays.difference(arrays))
    if missing_arrays:
        raise AnalysisContractError(
            "Prepared artifact lacks canonical evaluation arrays: "
            + ", ".join(missing_arrays)
        )
    target_expression = np.asarray(arrays["target_expression"])
    macroblock_ids = np.asarray(arrays["macroblock_ids"])
    split_labels = np.asarray(arrays["split_labels"])
    if (
        target_expression.ndim != 2
        or macroblock_ids.shape != (target_expression.shape[0],)
        or split_labels.shape != (target_expression.shape[0],)
    ):
        raise AnalysisContractError(
            "Prepared target, block, and split arrays are not row-aligned"
        )

    expected: dict[str, frozenset[tuple[str, str, int]]] = {}
    canonical_masks: dict[tuple[str, str, int], np.ndarray] = {}
    canonical_mask_checksums: dict[tuple[str, str, int], str] = {}
    y_true_by_split: dict[str, np.ndarray] = {}
    block_ids_by_split: dict[str, np.ndarray] = {}
    cell_ids_by_split: dict[str, np.ndarray] = {}
    bundle_summary: dict[str, Any] = {}
    split_labels_by_name = {
        "validation": "val",
        CONCLUSION_SPLIT: "test",
    }
    for split, split_label in split_labels_by_name.items():
        bundle = bundles.get(split)
        if not isinstance(bundle, Mapping):
            raise AnalysisContractError(
                f"Prepared artifact lacks its {split} fixed-mask bundle"
            )
        expected_bundle_id = prepared[f"{split}_mask_bundle_id"]
        if bundle.get("bundle_id") != expected_bundle_id:
            raise AnalysisContractError(
                f"Run {split} mask-bundle ID differs from prepared artifact"
            )
        loaded_bundle = loaded_bundles.get(split)
        if loaded_bundle is None:
            raise AnalysisContractError(
                f"Verified prepared artifact lacks its {split} mask bundle"
            )
        if (
            loaded_bundle.bundle_id != expected_bundle_id
            or loaded_bundle.checksum != bundle.get("bundle_checksum")
        ):
            raise AnalysisContractError(
                f"Prepared {split} mask-bundle content disagrees"
            )
        directory = (root / str(bundle.get("directory"))).resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise AnalysisContractError(
                "Fixed-mask bundle path escapes the prepared artifact"
            ) from exc
        bundle_manifest_path = directory / "manifest.json"
        masks_path = directory / "masks.npz"
        for canonical_path in (bundle_manifest_path, masks_path):
            relative_name = canonical_path.relative_to(root).as_posix()
            declared_checksum = files.get(relative_name)
            if (
                not isinstance(declared_checksum, str)
                or _file_sha256(canonical_path) != declared_checksum
            ):
                raise AnalysisContractError(
                    f"Prepared {split} mask file checksum mismatch: "
                    f"{canonical_path.name}"
                )
        bundle_manifest = dict(loaded_bundle.manifest)
        entries = bundle_manifest.get("entries")
        if (
            bundle_manifest.get("bundle_id") != expected_bundle_id
            or bundle_manifest.get("bundle_checksum")
            != bundle.get("bundle_checksum")
            or not isinstance(entries, list)
        ):
            raise AnalysisContractError(
                f"Prepared {split} mask bundle metadata disagrees"
            )
        keys: list[tuple[str, str, int]] = []
        entry_summary: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise AnalysisContractError(
                    f"Prepared {split} mask entry is invalid"
                )
            spec = entry.get("spec")
            if not isinstance(spec, Mapping):
                raise AnalysisContractError(
                    f"Prepared {split} mask entry lacks a spec"
                )
            if entry.get("split") != split:
                raise AnalysisContractError(
                    f"Prepared {split} mask entry names another split"
                )
            key = (
                split,
                _normalise_mode(spec.get("mode")),
                int(entry.get("replicate")),
            )
            entry_id = entry.get("entry_id")
            checksum = entry.get("mask_checksum")
            if (
                not isinstance(entry_id, str)
                or not entry_id
                or not isinstance(checksum, str)
                or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
                or entry_id not in loaded_bundle.masks
            ):
                raise AnalysisContractError(
                    f"Prepared {split} mask entry provenance is invalid"
                )
            canonical_mask = np.asarray(
                loaded_bundle.masks[entry_id],
                dtype=bool,
            )
            if _array_content_sha256(canonical_mask) != checksum:
                raise AnalysisContractError(
                    f"Prepared {split} canonical mask checksum mismatch"
                )
            keys.append(key)
            canonical_masks[key] = _set_read_only(canonical_mask)
            canonical_mask_checksums[key] = checksum
            entry_summary.append(
                {
                    "split": split,
                    "mask_mode": key[1],
                    "mask_replicate": key[2],
                    "entry_id": entry_id,
                    "mask_checksum": checksum,
                }
            )
        if len(keys) != len(set(keys)):
            raise AnalysisContractError(
                f"Prepared {split} mask bundle has duplicate evaluation keys"
            )
        expected[split] = frozenset(keys)

        node_index = np.asarray(arrays[f"{split}_node_index"])
        if (
            node_index.ndim != 1
            or not np.issubdtype(node_index.dtype, np.integer)
            or np.any(node_index < 0)
            or np.any(node_index >= target_expression.shape[0])
            or len(np.unique(node_index)) != len(node_index)
        ):
            raise AnalysisContractError(
                f"Prepared {split} node index is invalid"
            )
        node_index = node_index.astype(np.int64, copy=False)
        expected_node_index = np.flatnonzero(split_labels == split_label)
        if not np.array_equal(node_index, expected_node_index):
            raise AnalysisContractError(
                f"Prepared {split} node index disagrees with split labels"
            )
        canonical_y_true = _set_read_only(target_expression[node_index])
        canonical_blocks = _set_read_only(macroblock_ids[node_index])
        canonical_cell_ids = _set_read_only(
            np.arange(len(node_index), dtype=np.int64)
        )
        if any(
            mask.shape != canonical_y_true.shape
            for key, mask in canonical_masks.items()
            if key[0] == split
        ):
            raise AnalysisContractError(
                f"Prepared {split} masks do not align with canonical target rows"
            )
        y_true_by_split[split] = canonical_y_true
        block_ids_by_split[split] = canonical_blocks
        cell_ids_by_split[split] = canonical_cell_ids
        bundle_summary[split] = {
            "bundle_id": expected_bundle_id,
            "bundle_checksum": bundle.get("bundle_checksum"),
            "n_evaluations": len(keys),
            "evaluation_keys": sorted(keys),
            "entries": sorted(
                entry_summary,
                key=lambda item: (
                    item["mask_mode"],
                    item["mask_replicate"],
                ),
            ),
            "n_canonical_rows": len(node_index),
        }
    return PreparedEvaluationContract(
        summary={
            "path": str(root),
            "artifact_id": prepared["artifact_id"],
            "manifest_sha256": prepared["manifest_sha256"],
            "split_id": prepared["split_id"],
            "mask_bundles": bundle_summary,
            "verified_file_checksums": dict(files),
            "canonical_prediction_cell_ids": (
                "zero-based split-local row indices"
            ),
        },
        expected_evaluations=expected,
        masks=canonical_masks,
        mask_checksums=canonical_mask_checksums,
        y_true_by_split=y_true_by_split,
        block_ids_by_split=block_ids_by_split,
        cell_ids_by_split=cell_ids_by_split,
    )


def _assert_records_match_prepared_contract(
    records: Sequence[PredictionRecord],
    contract: PreparedEvaluationContract,
) -> None:
    """Require every prediction evaluation to reproduce canonical prepared rows."""

    for record in records:
        key = record.evaluation_key
        canonical_mask = contract.masks.get(key)
        expected_mask_checksum = contract.mask_checksums.get(key)
        if canonical_mask is None or expected_mask_checksum is None:
            raise AnalysisContractError(
                f"Run {record.run_id} has an evaluation outside the prepared "
                f"fixed-mask contract: {key}"
            )
        observed_mask_checksum = _array_content_sha256(
            np.asarray(record.mask, dtype=bool)
        )
        if (
            observed_mask_checksum != expected_mask_checksum
            or not _arrays_equal(np.asarray(record.mask), canonical_mask)
        ):
            raise AnalysisContractError(
                f"Run {record.run_id} {key} mask does not match its exact "
                "canonical prepared fixed mask"
            )
        canonical_y_true = contract.y_true_by_split.get(record.split)
        canonical_blocks = contract.block_ids_by_split.get(record.split)
        canonical_cells = contract.cell_ids_by_split.get(record.split)
        if (
            canonical_y_true is None
            or canonical_blocks is None
            or canonical_cells is None
        ):
            raise AnalysisContractError(
                f"Run {record.run_id} uses an unsupported prepared split "
                f"{record.split!r}"
            )
        if not _arrays_equal(record.y_true, canonical_y_true):
            raise AnalysisContractError(
                f"Run {record.run_id} {key} y_true differs from canonical "
                "prepared target rows"
            )
        if not _arrays_equal(record.block_ids, canonical_blocks):
            raise AnalysisContractError(
                f"Run {record.run_id} {key} block_ids differ from canonical "
                "prepared split rows"
            )
        if record.cell_ids is None or not _arrays_equal(
            record.cell_ids,
            canonical_cells,
        ):
            raise AnalysisContractError(
                f"Run {record.run_id} {key} cell_ids differ from canonical "
                "prepared split-local row order"
            )


def validate_locked_final_execution(
    collection: RunCollection,
    standards_lock_dir: str | os.PathLike[str],
    *,
    expected_seeds: int | None = None,
) -> dict[str, Any]:
    """Require the exact checksum-authorized final ladder and mask coverage."""

    lock_root = Path(standards_lock_dir).resolve()
    try:
        lock_manifest, lock, matrices = load_standards_lock(lock_root)
    except (FileNotFoundError, StandardsLockError) as exc:
        raise AnalysisContractError(
            f"Standards lock could not be verified: {exc}"
        ) from exc
    final_execution = lock.get("final_execution")
    if not isinstance(final_execution, Mapping):
        raise AnalysisContractError("Standards lock lacks final execution")
    matrix_name = final_execution.get("matrix_file")
    matrix = matrices.get(str(matrix_name))
    if not isinstance(matrix, Mapping):
        raise AnalysisContractError("Locked final matrix is unavailable")
    locked_seeds = tuple(int(seed) for seed in final_execution.get("seeds", []))
    if (
        len(locked_seeds) != 5
        or len(set(locked_seeds)) != 5
        or (
            expected_seeds is not None
            and int(expected_seeds) != len(locked_seeds)
        )
    ):
        raise AnalysisContractError(
            "Analysis seed expectation differs from the five locked identities"
        )
    expected_jobs: dict[tuple[str, int], dict[str, Any]] = {}
    for job in expand_matrix(matrix):
        try:
            condition = condition_name(job)
            seed = int(job["seed"])
        except (StandardsLockError, KeyError, TypeError, ValueError) as exc:
            raise AnalysisContractError("Locked final job is invalid") from exc
        key = (condition, seed)
        if key in expected_jobs:
            raise AnalysisContractError(f"Locked final job is duplicated: {key}")
        expected_jobs[key] = dict(job)
    exact_keys = {
        (condition, seed)
        for condition in REQUIRED_FINAL_CONDITIONS
        for seed in locked_seeds
    }
    if set(expected_jobs) != exact_keys:
        raise AnalysisContractError(
            "Standards lock does not contain the exact 10-condition five-seed ladder"
        )

    lock_manifest_sha256 = _file_sha256(lock_root / "manifest.json")
    matrix_sha256 = lock_manifest["files"].get(str(matrix_name))
    observed: dict[tuple[str, int], Mapping[str, Any]] = {}
    prepared_signature: tuple[Any, ...] | None = None
    prepared_contract: PreparedEvaluationContract | None = None
    for audit in collection.run_audits.values():
        if audit.get("artifact_kind") != EXPERIMENT_RUN_ARTIFACT_KIND:
            raise AnalysisContractError(
                "Conclusion runs must use the strict experiment artifact schema"
            )
        key = (str(audit["condition"]), int(audit["model_seed"]))
        if key not in expected_jobs:
            raise AnalysisContractError(
                f"Unexpected conclusion-bearing condition/seed: {key}"
            )
        if key in observed:
            raise AnalysisContractError(
                f"Duplicate conclusion-bearing condition/seed: {key}"
            )
        authorization = audit.get("standards_lock")
        if not isinstance(authorization, Mapping):
            raise AnalysisContractError(
                f"Run {audit['run_id']} lacks standards-lock authorization"
            )
        expected_authorization = {
            "authorization_version": 1,
            "lock_id": lock["lock_id"],
            "artifact_id": lock_manifest["artifact_id"],
            "lock_manifest_sha256": lock_manifest_sha256,
            "matrix_role": "final_execution",
            "final_matrix_file": matrix_name,
            "final_matrix_sha256": matrix_sha256,
            "condition": key[0],
            "canonical_job_hash": canonical_job_hash(expected_jobs[key]),
        }
        for field, expected in expected_authorization.items():
            if authorization.get(field) != expected:
                raise AnalysisContractError(
                    f"Run {audit['run_id']} has mismatched lock field {field}"
                )
        canonical_job = authorization.get("canonical_job")
        if (
            not isinstance(canonical_job, Mapping)
            or dict(canonical_job) != expected_jobs[key]
            or canonical_job_hash(canonical_job)
            != authorization["canonical_job_hash"]
        ):
            raise AnalysisContractError(
                f"Run {audit['run_id']} is not authorized for its exact locked job"
            )
        _assert_manifest_matches_job(audit, expected_jobs[key])

        prepared = audit.get("prepared_artifact")
        if not isinstance(prepared, Mapping):
            raise AnalysisContractError(
                f"Run {audit['run_id']} lacks prepared-artifact provenance"
            )
        signature = tuple(
            prepared.get(field)
            for field in (
                "path",
                "artifact_id",
                "manifest_sha256",
                "split_id",
                "validation_mask_bundle_id",
                "test_mask_bundle_id",
            )
        )
        if prepared_signature is None:
            prepared_signature = signature
            prepared_contract = _load_prepared_evaluation_contract(prepared)
        elif signature != prepared_signature:
            raise AnalysisContractError(
                "Final runs do not share one prepared artifact/split/mask bundle"
            )
        assert prepared_contract is not None
        declared = tuple(audit.get("evaluation_keys", ()))
        declared_set = set(declared)
        expected_set = set().union(
            *prepared_contract.expected_evaluations.values()
        )
        if len(declared) != len(declared_set) or declared_set != expected_set:
            raise AnalysisContractError(
                f"Run {audit['run_id']} lacks the exact fixed-mask evaluations"
            )
        observed[key] = audit
    if set(observed) != exact_keys:
        missing = sorted(exact_keys.difference(observed))
        raise AnalysisContractError(
            "Final execution is incomplete; missing condition/seeds: "
            + repr(missing)
        )
    assert prepared_contract is not None
    _assert_records_match_prepared_contract(
        collection.records,
        prepared_contract,
    )

    condition_coverage = [
        {
            "condition": condition,
            "condition_label": FINAL_CONDITION_LABELS[condition],
            "expected_seeds": list(locked_seeds),
            "observed_seeds": sorted(
                seed for observed_condition, seed in observed
                if observed_condition == condition
            ),
            "complete": True,
        }
        for condition in REQUIRED_FINAL_CONDITIONS
    ]
    return {
        "status": "complete",
        "complete": True,
        "required_conditions": list(REQUIRED_FINAL_CONDITIONS),
        "locked_seed_identities": list(locked_seeds),
        "expected_run_count": len(exact_keys),
        "observed_run_count": len(observed),
        "condition_coverage": condition_coverage,
        "standards_lock": {
            "path": str(lock_root),
            "lock_id": lock["lock_id"],
            "artifact_id": lock_manifest["artifact_id"],
            "manifest_sha256": lock_manifest_sha256,
            "final_matrix_file": matrix_name,
            "final_matrix_sha256": matrix_sha256,
        },
        "prepared_artifact": dict(prepared_contract.summary),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _csv_cell(value: Any) -> Any:
    value = _json_safe(value)
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _csv_cell(row.get(field)) for field in fieldnames}
                )
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _plotting_module() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to create benchmark analysis figures"
        ) from exc
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 130,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save_figure(fig: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.png")
    fig.savefig(
        temporary,
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
    )
    os.replace(temporary, path)


def _plot_loss_by_model_mode(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    split: str,
    locked_graph_id: str,
) -> None:
    plt = _plotting_module()
    desired = [row for row in rows if row["split"] == split]
    modes = [
        mode for mode in FINAL_MASK_MODES
        if any(row["mask_mode"] == mode for row in desired)
    ]
    conditions = [
        condition for condition in REQUIRED_FINAL_CONDITIONS
        if any(row["condition"] == condition for row in desired)
    ]
    values = np.full((len(conditions), len(modes)), np.nan, dtype=float)
    for row_number, condition in enumerate(conditions):
        for column_number, mode in enumerate(modes):
            matches = [
                row for row in desired
                if row["condition"] == condition and row["mask_mode"] == mode
            ]
            if len(matches) != 1:
                raise AnalysisContractError(
                    f"Expected one loss row for {condition}/{split}/{mode}"
                )
            values[row_number, column_number] = matches[0]["huber_loss"]
    fig, axis = plt.subplots(figsize=(7.4, 6.0))
    image = axis.imshow(values, aspect="auto", cmap="viridis_r")
    axis.set_xticks(
        np.arange(len(modes)),
        [mode.replace("_", " ").title() for mode in modes],
    )
    axis.set_yticks(
        np.arange(len(conditions)),
        [FINAL_CONDITION_LABELS[condition] for condition in conditions],
    )
    for row_number in range(values.shape[0]):
        for column_number in range(values.shape[1]):
            value = values[row_number, column_number]
            axis.text(
                column_number,
                row_number,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=(
                    "white"
                    if value > np.nanmedian(values)
                    else "#111827"
                ),
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Masked Huber loss (lower is better)")
    axis.set_title(
        f"Complete locked-ladder seed-ensemble loss — {split} split"
    )
    axis.text(
        0.0,
        -0.10,
        "Each cell is the equal-weight spatial-block mean after seed and "
        "fixed-mask-replicate aggregation.",
        transform=axis.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    _save_figure(fig, path)
    plt.close(fig)


def _plot_graph_candidates(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    split: str,
    locked_graph_id: str,
) -> None:
    plt = _plotting_module()
    candidates = [
        row
        for row in rows
        if row["condition"] == "g1_true"
        and row["graph_kind"] == "true"
        and row["split"] == split
        and row["mask_mode"] == "node"
    ]
    candidates.sort(key=lambda row: row["graph_id"])
    if not candidates:
        raise AnalysisContractError(
            "No true G1 whole-node graph candidates exist for plotting"
        )
    fig, axis = plt.subplots(figsize=(max(6.0, 0.72 * len(candidates)), 4.2))
    x = np.arange(len(candidates))
    values = np.asarray([row["huber_loss"] for row in candidates])
    errors = np.asarray(
        [
            [
                max(0.0, row["huber_loss"] - row["huber_ci_lower"])
                for row in candidates
            ],
            [
                max(0.0, row["huber_ci_upper"] - row["huber_loss"])
                for row in candidates
            ],
        ]
    )
    colors = [
        "#1D4ED8" if row["graph_id"] == locked_graph_id else "#94A3B8"
        for row in candidates
    ]
    axis.bar(x, values, color=colors, width=0.72, yerr=errors, capsize=3)
    axis.set_xticks(
        x,
        [row["graph_id"] for row in candidates],
        rotation=35,
        ha="right",
    )
    axis.set_ylabel("Whole-node masked Huber loss")
    axis.set_title("Within-core prespecified graph candidates")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    axis.text(
        0.0,
        -0.30,
        "Blue denotes the externally locked graph, not a post-hoc biological "
        "selection.",
        transform=axis.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    _save_figure(fig, path)
    plt.close(fig)


def _plot_spatial_gains(
    gains: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    split: str,
) -> None:
    plt = _plotting_module()
    selected = [gain for gain in gains if gain["split"] == split]
    if not selected:
        raise AnalysisContractError("No paired spatial gains exist for plotting")
    ordering = {
        ("B0_minus_G1", "node"): 0,
        ("B0_minus_G1", "block"): 1,
        ("rewired_G1_minus_true_G1", "node"): 2,
        ("rewired_G1_minus_true_G1", "block"): 3,
    }
    selected.sort(
        key=lambda gain: ordering.get(
            (gain["comparison"], gain["mask_mode"]),
            99,
        )
    )
    labels = []
    for gain in selected:
        comparison = (
            "B0 − G1"
            if gain["comparison"] == "B0_minus_G1"
            else "rewired G1 − true G1"
        )
        labels.append(f"{comparison} | {gain['mask_mode'].title()}")
    y = np.arange(len(selected))[::-1]
    estimates = np.asarray([gain["delta"] for gain in selected])
    errors = np.asarray(
        [
            [
                max(0.0, gain["delta"] - gain["delta_ci_lower"])
                for gain in selected
            ],
            [
                max(0.0, gain["delta_ci_upper"] - gain["delta"])
                for gain in selected
            ],
        ]
    )
    fig, axis = plt.subplots(figsize=(7.2, 1.2 + 0.75 * len(selected)))
    axis.axvline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    axis.errorbar(
        estimates,
        y,
        xerr=errors,
        fmt="o",
        color="#2563EB",
        ecolor="#64748B",
        capsize=4,
        markersize=6,
    )
    axis.set_yticks(y, labels)
    axis.set_xlabel("Paired masked-Huber gain (positive favors true G1)")
    axis.set_title("Within-core spatial-block paired predictive gain")
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    axis.text(
        0.0,
        -0.25,
        "Intervals resample spatial blocks; cells and model seeds are not "
        "replicates.",
        transform=axis.transAxes,
        fontsize=8,
        color="#4B5563",
    )
    _save_figure(fig, path)
    plt.close(fig)


def _plot_graph_qc(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    plt = _plotting_module()
    true_rows = [row for row in rows if row["graph_kind"] == "true"]
    true_rows.sort(key=lambda row: row["graph_id"])
    if not true_rows:
        raise AnalysisContractError("No true-graph QC records exist for plotting")
    labels = [row["graph_id"] for row in true_rows]
    mean_degree = np.asarray(
        [
            np.nan if row.get("mean_degree") is None else row["mean_degree"]
            for row in true_rows
        ],
        dtype=float,
    )
    p95_distance = np.asarray(
        [
            (
                np.nan
                if row.get("edge_distance_p95_um") is None
                else row["edge_distance_p95_um"]
            )
            for row in true_rows
        ],
        dtype=float,
    )
    isolated_rate = np.asarray(
        [
            (
                np.nan
                if not row.get("n_nodes")
                or row.get("n_isolated_nodes") is None
                else row["n_isolated_nodes"] / row["n_nodes"]
            )
            for row in true_rows
        ],
        dtype=float,
    )
    x = np.arange(len(true_rows))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(10.0, 0.70 * len(true_rows)), 4.1),
        constrained_layout=True,
    )
    axes[0].bar(x, mean_degree, color="#2563EB", width=0.70)
    axes[0].set_ylabel("Mean undirected degree")
    axes[0].set_title("Graph connectivity")
    twin = axes[0].twinx()
    twin.plot(
        x,
        isolated_rate * 100,
        color="#DC2626",
        marker="o",
        linewidth=1.5,
    )
    twin.set_ylabel("Isolated cells (%)", color="#DC2626")
    twin.spines["right"].set_visible(True)
    axes[1].bar(x, p95_distance, color="#059669", width=0.70)
    axes[1].set_ylabel("95th percentile edge distance (µm)")
    axes[1].set_title("Physical edge scale")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.7)
    _save_figure(fig, path)
    plt.close(fig)


def _flat_gain_row(gain: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in gain.items()
        if key != "blocks"
    }


def analyze_run_directory(
    runs_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    standards_lock_dir: str | os.PathLike[str],
    locked_graph_id: str | None = None,
    graph_qc_dir: str | os.PathLike[str] | None = None,
    split: str = "test",
    graph_candidate_split: str = "validation",
    expected_seeds: int | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    """Aggregate runs, apply the locked H1 gate, and write tables/figures."""

    if split != CONCLUSION_SPLIT:
        raise AnalysisContractError(
            "The prespecified acceptance gate is restricted to the sealed "
            f"{CONCLUSION_SPLIT!r} split; screening/validation splits are not "
            "conclusion-bearing."
        )
    collection = load_run_collection(runs_dir, graph_qc_dir=graph_qc_dir)
    execution = validate_locked_final_execution(
        collection,
        standards_lock_dir,
        expected_seeds=expected_seeds,
    )
    locked_seed_count = len(execution["locked_seed_identities"])
    ensembles = ensemble_run_records(
        collection.records,
        expected_seeds=locked_seed_count,
    )
    locked = _infer_locked_graph_id(ensembles, locked_graph_id)
    loss_rows = summarise_model_losses(
        ensembles,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    gains: list[dict[str, Any]] = []
    for mode_number, mode in enumerate(("node", "block")):
        b0_gain = aggregate_paired_gain(
            ensembles,
            comparison="B0_minus_G1",
            baseline_condition="b0",
            spatial_condition="g1_true",
            split=split,
            mask_mode=mode,
            locked_graph_id=locked,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed + 100 * mode_number,
        )
        if b0_gain is not None:
            gains.append(b0_gain)
        rewired_gain = aggregate_paired_gain(
            ensembles,
            comparison="rewired_G1_minus_true_G1",
            baseline_condition="g1_rewired",
            spatial_condition="g1_true",
            split=split,
            mask_mode=mode,
            locked_graph_id=locked,
            confidence_level=confidence_level,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed + 100 * mode_number + 50,
        )
        if rewired_gain is not None:
            gains.append(rewired_gain)
    secondary_specs = (
        ("B0_matched_minus_G1", "b0_parameter_matched", "g1_true"),
        ("broad_field_minus_G1", "broad_field", "g1_true"),
        ("B1_minus_G1", "b1", "g1_true"),
        ("G1_minus_G2", "g1_true", "g2_true"),
        ("G2_zero_minus_G2_true", "g2_zero", "g2_true"),
        (
            "G2_distance_only_minus_G2_true",
            "g2_distance_only",
            "g2_true",
        ),
        ("G2_permuted_minus_G2_true", "g2_permuted", "g2_true"),
    )
    secondary_gains: list[dict[str, Any]] = []
    for comparison_number, (
        comparison,
        baseline_condition,
        spatial_condition,
    ) in enumerate(secondary_specs):
        for mode_number, mode in enumerate(FINAL_MASK_MODES):
            gain = aggregate_paired_gain(
                ensembles,
                comparison=comparison,
                baseline_condition=baseline_condition,
                spatial_condition=spatial_condition,
                split=split,
                mask_mode=mode,
                locked_graph_id=locked,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=(
                    bootstrap_seed
                    + 1_000
                    + 300 * comparison_number
                    + 50 * mode_number
                ),
            )
            if gain is None:
                raise AnalysisContractError(
                    f"Locked secondary comparison is missing: "
                    f"{comparison}/{mode}"
                )
            secondary_gains.append(gain)
    gate = evaluate_acceptance_gate(
        gains,
        expected_seeds=locked_seed_count,
        required_split=CONCLUSION_SPLIT,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    graph_qc_rows = [dict(row) for row in collection.graph_qc]
    if not graph_qc_rows:
        raise AnalysisContractError(
            "Graph-QC records are required for the conclusion-bearing report"
        )
    available_candidate_splits = {
        row["split"]
        for row in loss_rows
        if row["condition"] == "g1_true"
        and row["graph_kind"] == "true"
        and row["mask_mode"] == "node"
        and row["graph_id"] == locked
    }
    candidate_split_used = (
        graph_candidate_split
        if graph_candidate_split in available_candidate_splits
        else split
    )

    artifacts = {
        "summary_json": "summary.json",
        "loss_csv": "model_mode_losses.csv",
        "gain_csv": "spatial_gains.csv",
        "gate_csv": "acceptance_gate.csv",
        "graph_qc_csv": "graph_qc.csv",
        "condition_coverage_csv": "condition_coverage.csv",
        "loss_figure": "figures/loss_by_model_mode.png",
        "candidate_figure": "figures/graph_candidate_loss.png",
        "gain_figure": "figures/spatial_gain_ci.png",
        "graph_qc_figure": "figures/graph_qc_degree_distance.png",
    }
    conclusion_run_ids = sorted(
        {record.run_id for record in collection.records}
    )
    summary: dict[str, Any] = {
        "format_version": ANALYSIS_FORMAT_VERSION,
        "scope": WITHIN_CORE_SCOPE,
        "analysis_split": split,
        "graph_candidate_split": candidate_split_used,
        "locked_graph_id": locked,
        "expected_model_seeds": locked_seed_count,
        "confidence_level": float(confidence_level),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(bootstrap_seed),
        "conclusion_input_policy": {
            "required_run_status": "complete",
            "required_sealed_test_opened": True,
            "required_evaluation_split": CONCLUSION_SPLIT,
            "screening_and_incomplete_runs_excluded": True,
            "required_verified_standards_lock": True,
            "required_exact_final_matrix": True,
        },
        "execution_completeness": execution,
        "n_prediction_records": len(collection.records),
        "n_ensemble_evaluations": len(ensembles),
        "input_run_ids": conclusion_run_ids,
        "input_manifest_checksums": {
            run_id: collection.manifest_checksums[run_id]
            for run_id in conclusion_run_ids
        },
        "discovered_manifest_checksums": dict(
            collection.manifest_checksums
        ),
        "input_prediction_checksums": dict(collection.prediction_checksums),
        "failed_runs": list(collection.failed_runs),
        "excluded_nonconclusion_runs": list(collection.excluded_runs),
        "loss_summary": loss_rows,
        "paired_spatial_gains": gains,
        "secondary_paired_gains": secondary_gains,
        "acceptance_gate": gate,
        "graph_qc": graph_qc_rows,
        "artifacts": artifacts,
        "interpretation": (
            "Positive paired deltas indicate a within-core predictive advantage "
            "for the named spatial model under fixed masks. They are not "
            "biological effect sizes."
        ),
    }
    _write_csv(output / artifacts["loss_csv"], loss_rows)
    _write_csv(
        output / artifacts["gain_csv"],
        [
            {"comparison_family": "h1", **_flat_gain_row(gain)}
            for gain in gains
        ]
        + [
            {"comparison_family": "secondary", **_flat_gain_row(gain)}
            for gain in secondary_gains
        ],
    )
    gate_rows = [
        {
            "scope": "within_core_descriptive",
            "category": "input_validity",
            "criterion": "sealed_test_split_complete",
            "passed": gate["conclusion_split_complete"],
            "observed": gate["conclusion_split_complete"],
            "threshold": CONCLUSION_SPLIT,
        },
        {
            "scope": "within_core_descriptive",
            "category": "input_validity",
            "criterion": "five_seed_complete",
            "passed": gate["five_seed_complete"],
            "observed": gate["five_seed_complete"],
            "threshold": f"{locked_seed_count} locked seeds/model",
        },
        {
            "scope": "within_core_descriptive",
            "category": "input_validity",
            "criterion": "paired_mask_replicates_complete",
            "passed": gate["paired_mask_replicates_complete"],
            "observed": gate["paired_mask_replicates_complete"],
            "threshold": "same fixed replicates on both models",
        },
        *[
            {
                "scope": "within_core_descriptive",
                "category": "hypothesis_gate",
                **criterion,
            }
            for criterion in gate["criteria"]
        ],
    ]
    _write_csv(output / artifacts["gate_csv"], gate_rows)
    _write_csv(output / artifacts["graph_qc_csv"], graph_qc_rows)
    _write_csv(
        output / artifacts["condition_coverage_csv"],
        execution["condition_coverage"],
    )
    _plot_loss_by_model_mode(
        loss_rows,
        output / artifacts["loss_figure"],
        split=split,
        locked_graph_id=locked,
    )
    _plot_graph_candidates(
        loss_rows,
        output / artifacts["candidate_figure"],
        split=candidate_split_used,
        locked_graph_id=locked,
    )
    _plot_spatial_gains(
        gains,
        output / artifacts["gain_figure"],
        split=split,
    )
    _plot_graph_qc(
        graph_qc_rows,
        output / artifacts["graph_qc_figure"],
    )
    summary["artifact_checksums"] = {
        name: _file_sha256(output / relative_path)
        for name, relative_path in artifacts.items()
        if name != "summary_json"
    }
    safe_summary = _json_safe(summary)
    _atomic_text(
        output / artifacts["summary_json"],
        json.dumps(
            safe_summary,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )
    return safe_summary
