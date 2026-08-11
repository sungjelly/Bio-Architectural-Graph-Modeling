#!/usr/bin/env python3
"""Materialize checksum-bound same-gene robustness prepared-data overlays.

The published tree is immutable and self-contained.  Large inputs are stored
once under ``shared/``; every runner-facing variant uses relative symlinks that
resolve inside the published output root.  Graphs and source-state nulls are
constructed without consulting train/test outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = str(PROJECT_ROOT / "src")
if not sys.path or sys.path[0] != _SOURCE_ROOT:
    sys.path.insert(0, _SOURCE_ROOT)

from spatial_benchmark.data import (
    ALLOWED_METADATA_COLUMNS,
    CoreSelection,
    discover_slide_raw_path,
    load_selected_core,
)
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.identifiers import canonical_json, canonical_sha256
from spatial_benchmark.paths import current_paths
from spatial_benchmark.same_gene_jacobian import SameGeneJacobianError
from spatial_benchmark import same_gene_robustness as robustness


PREPROCESSING_VERSION = "same_gene_robustness_v1"
PERMUTATION_SEED = 20260810
SLIDES = ("SO_1", "SO_2")
CELL_TYPE_FIELD = "RNA_Gastric_testrun_Cell.Typing.InSituType.1_1_clusters"
EXPECTED_CELL_TYPE_LEVELS = 12
ROUNDTRIP_TOLERANCE = 2e-3
FROZEN_ELIGIBLE_GENE_SHA256 = (
    "2420a9d160894a78e6a1db1cc4ff2c52757211cf31dec859856f7c3cf231712d"
)
FROZEN_ELIGIBLE_GENE_COUNT = 932
DEFAULT_ELIGIBLE_GENE_RELATIVE_PATH = Path(
    "analyses/same_gene_nonlinear_convergence_20260810/aggregate_jacobians.npz"
)

COMMON_ARRAY_FILES = (
    "expression_log1p.npy",
    "metadata.npy",
    "coordinates_um.npy",
    "fov.npy",
    "geometry_group.npy",
    "fold.npy",
    "qc_passed.npy",
    "cell_type_code.npy",
    "panel_log_total.npy",
)
GRAPH_ARRAY_FILES = (
    "neighbor_near_mean.npy",
    "neighbor_annular_mean.npy",
    "neighbor_permuted_near_mean.npy",
    "near_degree.npy",
    "annular_degree.npy",
    "permuted_near_degree.npy",
    "matched_eligible.npy",
    "eligible_primary.npy",
    "source_permutation.npy",
    "near_indptr.npy",
    "near_indices.npy",
    "annular_indptr.npy",
    "annular_indices.npy",
    "neighbor_near_panel_log_total_mean.npy",
    "neighbor_annular_panel_log_total_mean.npy",
    "neighbor_permuted_near_panel_log_total_mean.npy",
)
CELL_TYPE_PROPORTION_FILES = (
    "neighbor_near_cell_type_proportions.npy",
    "neighbor_annular_cell_type_proportions.npy",
    "neighbor_permuted_near_cell_type_proportions.npy",
)
RUNNER_REQUIRED_FILES = (
    "expression_log1p.npy",
    "metadata.npy",
    "coordinates_um.npy",
    "fov.npy",
    "geometry_group.npy",
    "fold.npy",
    "qc_passed.npy",
    "neighbor_near_mean.npy",
    "neighbor_annular_mean.npy",
    "neighbor_permuted_near_mean.npy",
    "near_degree.npy",
    "annular_degree.npy",
    "permuted_near_degree.npy",
    "matched_eligible.npy",
    "eligible_primary.npy",
    "source_permutation.npy",
)


@dataclass(frozen=True, slots=True)
class VariantSpec:
    variant_id: str
    contract_variant_id: str
    normalization: str
    partition_mode: str
    node_policy: str
    residualization: Mapping[str, Any]
    graph_source_variant: str | None = None

    @property
    def manifest_variant_id(self) -> str:
        # A0 is a factorial secondary outside the core runner's V0--V6 enum.
        return self.variant_id if self.contract_variant_id == "A0" else self.contract_variant_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "graph": {
                "partition": self.partition_mode,
                "k": 12,
                "near_um": [0.0, 25.0],
                "annular_um": [25.0, 50.0],
                "self_edges": False,
                "zero_distance_edges": False,
            },
            "normalization": {"name": self.normalization},
            "node_policy": {"name": self.node_policy},
            "permutation_seed": PERMUTATION_SEED,
            "primary_eligibility_file": "eligible_primary.npy",
            "native_eligibility_file": "matched_eligible.npy",
            "eligible_gene_file": "eligible_genes.npy",
            "feature_files": {
                "observed_near": "neighbor_near_mean.npy",
                "observed_annular": "neighbor_annular_mean.npy",
                "within_fov_permuted_near": "neighbor_permuted_near_mean.npy",
            },
            "residualization": dict(self.residualization),
            "graph_source_variant": self.graph_source_variant,
        }


_PANEL_AUX = {
    "panel_log_total_file": "panel_log_total.npy",
    "neighbor_panel_log_total_files": {
        "observed_near": "neighbor_near_panel_log_total_mean.npy",
        "observed_annular": "neighbor_annular_panel_log_total_mean.npy",
        "within_fov_permuted_near": (
            "neighbor_permuted_near_panel_log_total_mean.npy"
        ),
    },
}
_CELL_TYPE_AUX = {
    "cell_type_code_file": "cell_type_code.npy",
    "cell_type_levels_file": "cell_type_levels.json",
    "neighbor_cell_type_proportion_files": {
        "observed_near": "neighbor_near_cell_type_proportions.npy",
        "observed_annular": "neighbor_annular_cell_type_proportions.npy",
        "within_fov_permuted_near": (
            "neighbor_permuted_near_cell_type_proportions.npy"
        ),
    },
}

VARIANT_SPECS = (
    VariantSpec(
        "v0_within_fov_log1p_all",
        "V0",
        "log1p_raw_count",
        "within_fov",
        "all",
        {"kind": "none"},
    ),
    VariantSpec(
        "v1_component_log1p_all",
        "V1",
        "log1p_raw_count",
        "within_geometry_component",
        "all",
        {"kind": "none"},
    ),
    VariantSpec(
        "v2_within_fov_log1p_qc_induced",
        "V2",
        "log1p_raw_count",
        "within_fov",
        "qc_induced",
        {"kind": "none"},
    ),
    VariantSpec(
        "v3_within_fov_cp10k_all",
        "V3",
        "panel_log_cp10k",
        "within_fov",
        "all",
        {"kind": "none"},
    ),
    VariantSpec(
        "a0_component_cp10k_all",
        "A0",
        "panel_log_cp10k",
        "within_geometry_component",
        "all",
        {"kind": "none"},
    ),
    VariantSpec(
        "v5_component_cp10k_qc_induced",
        "V5",
        "panel_log_cp10k",
        "within_geometry_component",
        "qc_induced",
        {"kind": "none"},
    ),
    VariantSpec(
        "v4_train_only_library_residual",
        "V4",
        "log1p_raw_count",
        "within_fov",
        "all",
        {"kind": "library", **_PANEL_AUX},
        graph_source_variant="v0_within_fov_log1p_all",
    ),
    VariantSpec(
        "v6_train_only_cell_type_library_residual",
        "V6",
        "log1p_raw_count",
        "within_fov",
        "all",
        {"kind": "cell_type_library", **_PANEL_AUX, **_CELL_TYPE_AUX},
        graph_source_variant="v0_within_fov_log1p_all",
    ),
)
VARIANT_BY_ID = {item.variant_id: item for item in VARIANT_SPECS}
STATIC_VARIANTS = tuple(item for item in VARIANT_SPECS if item.graph_source_variant is None)


@dataclass(slots=True)
class SlideSource:
    expression_log1p: np.ndarray
    expression_cp10k: np.ndarray
    raw_counts: np.ndarray
    metadata: np.ndarray
    coordinates_um: np.ndarray
    fov: np.ndarray
    geometry_group: np.ndarray
    fold: np.ndarray
    qc_passed: np.ndarray
    base_matched_eligible: np.ndarray
    cell_type_labels: np.ndarray
    panel_log_total: np.ndarray
    audit: dict[str, Any]


class RobustPreparedError(RuntimeError):
    """Raised when preparation or verification fails closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise RobustPreparedError(f"non-finite JSON constant {value!r}: {path}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RobustPreparedError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique,
        )
    except RobustPreparedError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RobustPreparedError(f"invalid JSON: {path}") from error


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safe_child(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise RobustPreparedError(f"unsafe relative path: {relative!r}")
    candidate = root / raw
    try:
        candidate.parent.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise RobustPreparedError(f"path escapes prepared root: {relative!r}") from error
    return candidate


def _content_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if not path.exists():
        raise RobustPreparedError(f"missing content file: {path}")
    record: dict[str, Any] = {
        "path": path.relative_to(relative_to).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path.resolve(strict=True) if path.is_symlink() else path),
    }
    if path.is_symlink():
        target = os.readlink(path)
        if Path(target).is_absolute():
            raise RobustPreparedError(f"absolute symlink forbidden: {path}")
        record["storage"] = "relative_symlink"
        record["link_target"] = target
    else:
        record["storage"] = "regular_file"
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record["shape"] = list(array.shape)
        record["dtype"] = str(array.dtype)
    return record


def _save_array(path: Path, value: np.ndarray) -> None:
    np.save(path, np.asarray(value), allow_pickle=False)


def _relative_symlink(link: Path, target: Path, *, output_root: Path) -> None:
    output = output_root.resolve()
    target_resolved = target.resolve(strict=True)
    try:
        target_resolved.relative_to(output)
    except ValueError as error:
        raise RobustPreparedError("symlink target must remain within output root") from error
    relative = os.path.relpath(target_resolved, start=link.parent.resolve())
    if Path(relative).is_absolute():
        raise RobustPreparedError("relative symlink construction failed")
    link.symlink_to(relative)


def _verify_record(container: Path, record: Mapping[str, Any], output_root: Path) -> None:
    required = {"path", "size_bytes", "sha256", "storage"}
    if not required.issubset(record):
        raise RobustPreparedError("content record is incomplete")
    path = _safe_child(container, str(record["path"]))
    storage = record["storage"]
    if storage == "relative_symlink":
        if not path.is_symlink():
            raise RobustPreparedError(f"expected relative symlink: {path}")
        target = os.readlink(path)
        if target != record.get("link_target") or Path(target).is_absolute():
            raise RobustPreparedError(f"symlink target changed: {path}")
        resolved = (path.parent / target).resolve(strict=True)
        try:
            resolved.relative_to(output_root.resolve())
        except ValueError as error:
            raise RobustPreparedError(f"symlink escaped output root: {path}") from error
        if resolved.is_symlink() or not resolved.is_file():
            raise RobustPreparedError(f"symlink target is not a regular file: {path}")
    elif storage == "regular_file":
        if path.is_symlink() or not path.is_file():
            raise RobustPreparedError(f"expected regular file: {path}")
    else:
        raise RobustPreparedError(f"unsupported storage mode: {storage!r}")
    if path.stat().st_size != int(record["size_bytes"]):
        raise RobustPreparedError(f"file size changed: {path}")
    content_path = path.resolve(strict=True) if path.is_symlink() else path
    if sha256_file(content_path) != record["sha256"]:
        raise RobustPreparedError(f"content hash changed: {path}")
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record.get("shape") or str(array.dtype) != record.get(
            "dtype"
        ):
            raise RobustPreparedError(f"array contract changed: {path}")


def _verify_base(base_root: Path) -> tuple[dict[str, Any], str, list[str], list[str]]:
    if base_root.is_symlink() or not base_root.is_dir():
        raise RobustPreparedError(f"base prepared root is invalid: {base_root}")
    manifest_path = base_root / "manifest.json"
    manifest = _strict_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("manifest_schema_version") != 1:
        raise RobustPreparedError("base prepared manifest schema is unsupported")
    slides = manifest.get("slides")
    if not isinstance(slides, dict) or set(slides) != set(SLIDES):
        raise RobustPreparedError("base prepared slide inventory is not exact")
    for slide in SLIDES:
        arrays = slides[slide].get("arrays")
        if not isinstance(arrays, dict):
            raise RobustPreparedError(f"base {slide} array inventory is missing")
        for item in arrays.values():
            path = base_root / slide / str(item["path"])
            if path.is_symlink() or not path.is_file():
                raise RobustPreparedError(f"base content must be a regular file: {path}")
            if path.stat().st_size != int(item["size_bytes"]):
                raise RobustPreparedError(f"base size mismatch: {path}")
            if sha256_file(path) != item["sha256"]:
                raise RobustPreparedError(f"base hash mismatch: {path}")
            if path.suffix == ".npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(array.shape) != item["shape"] or str(array.dtype) != item["dtype"]:
                    raise RobustPreparedError(f"base array contract mismatch: {path}")
    processed = canonical_sha256(
        {slide: slides[slide]["arrays"] for slide in SLIDES}
    )
    if processed != manifest.get("processed_fingerprint"):
        raise RobustPreparedError("base processed fingerprint changed")
    genes = _strict_json(base_root / "genes.json")
    metadata_names = _strict_json(base_root / "metadata_names.json")
    if (
        not isinstance(genes, list)
        or not all(isinstance(item, str) and item for item in genes)
        or len(set(genes)) != len(genes)
        or canonical_sha256(genes) != manifest.get("gene_order_sha256")
    ):
        raise RobustPreparedError("base gene order is invalid")
    if metadata_names != manifest.get("metadata_names"):
        raise RobustPreparedError("base metadata name order is invalid")
    return manifest, sha256_file(manifest_path), genes, metadata_names


def _load_cell_type_labels(
    raw_root: Path,
    base_root: Path,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    unknown = frozenset({"", "unknown", "unassigned", "na", "n/a", "none", "nan"})
    for slide in SLIDES:
        base_fov = np.load(base_root / slide / "fov.npy", allow_pickle=False)
        path = discover_slide_raw_path(raw_root, slide, "metadata")
        try:
            frame = pd.read_csv(
                path,
                usecols=["fov", "cell_ID", CELL_TYPE_FIELD],
                dtype={"fov": "int32", "cell_ID": "int32", CELL_TYPE_FIELD: "string"},
            )
        except (OSError, ValueError) as error:
            raise RobustPreparedError(f"could not read enriched cell type field: {path}") from error
        frame = frame.loc[frame["fov"].isin(np.unique(base_fov))].copy()
        if frame.duplicated(["fov", "cell_ID"]).any():
            raise RobustPreparedError(f"duplicate enriched cell key on {slide}")
        frame = frame.sort_values(["fov", "cell_ID"], kind="mergesort").reset_index(drop=True)
        if len(frame) != len(base_fov) or not np.array_equal(
            frame["fov"].to_numpy(dtype=np.int32), base_fov.astype(np.int32, copy=False)
        ):
            raise RobustPreparedError(f"enriched cell type coverage/order differs on {slide}")
        values = frame[CELL_TYPE_FIELD]
        if values.isna().any():
            raise RobustPreparedError(f"missing enriched cell type on {slide}")
        labels = values.astype(str).str.strip().to_numpy(dtype=np.str_)
        if any(value.casefold() in unknown for value in labels):
            raise RobustPreparedError(f"unknown enriched cell type on {slide}")
        result[slide] = labels
    levels = sorted({str(value) for slide in SLIDES for value in result[slide]})
    if len(levels) != EXPECTED_CELL_TYPE_LEVELS:
        raise RobustPreparedError(
            f"expected {EXPECTED_CELL_TYPE_LEVELS} cell types, observed {len(levels)}"
        )
    return result


def _eligible_gene_hash(mask: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _load_frozen_eligible_genes(
    path: Path,
    genes: list[str],
    *,
    enforce_production_identity: bool,
) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if "eligible" not in payload or "genes" not in payload:
                raise RobustPreparedError("eligible-gene NPZ lacks eligible/genes")
            mask = np.asarray(payload["eligible"], dtype=bool)
            observed_genes = [str(value) for value in payload["genes"].tolist()]
    except (OSError, ValueError) as error:
        raise RobustPreparedError(f"could not read frozen eligible-gene mask: {path}") from error
    if mask.shape != (len(genes),) or observed_genes != genes:
        raise RobustPreparedError("frozen eligible-gene mask/order does not match prepared genes")
    digest = _eligible_gene_hash(mask)
    if enforce_production_identity and (
        int(np.sum(mask)) != FROZEN_ELIGIBLE_GENE_COUNT
        or digest != FROZEN_ELIGIBLE_GENE_SHA256
    ):
        raise RobustPreparedError("frozen 932-gene mask identity changed")
    return mask


def _load_slide_source(
    raw_root: Path,
    base_root: Path,
    slide: str,
    genes: list[str],
    metadata_names: list[str],
    cell_type_labels: np.ndarray,
    *,
    expected_gene_count: int,
) -> SlideSource:
    base_fov = np.load(base_root / slide / "fov.npy", allow_pickle=False)
    selection = CoreSelection(
        slide=slide, fovs=tuple(int(value) for value in np.unique(base_fov))
    )
    dataset = load_selected_core(
        raw_root,
        selection,
        chunksize=32768,
        expected_biological_probes=expected_gene_count,
        qc_policy="all",
    )
    if list(dataset.gene_names) != genes or list(dataset.metadata_names) != metadata_names:
        raise RobustPreparedError(f"raw panel/metadata order changed on {slide}")
    if not np.array_equal(
        dataset.keys["fov"].to_numpy(dtype=np.int32),
        base_fov.astype(np.int32, copy=False),
    ):
        raise RobustPreparedError(f"raw cell key order differs from base on {slide}")
    # Re-read the enriched field directly and join on the full composite key.
    # This makes the label alignment independent of CSV row order and proves
    # strict one-to-one coverage against the exact cells used by the base loader.
    enriched_path = discover_slide_raw_path(raw_root, slide, "metadata")
    enriched = pd.read_csv(
        enriched_path,
        usecols=["fov", "cell_ID", CELL_TYPE_FIELD],
        dtype={"fov": "int32", "cell_ID": "int32", CELL_TYPE_FIELD: "string"},
    )
    enriched = enriched.loc[enriched["fov"].isin(selection.fovs)].copy()
    if enriched.duplicated(["fov", "cell_ID"]).any():
        raise RobustPreparedError(f"duplicate enriched cell key on {slide}")
    aligned = dataset.keys[["fov", "cell_ID"]].merge(
        enriched,
        how="left",
        on=["fov", "cell_ID"],
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if len(enriched) != len(aligned) or not (aligned["_merge"] == "both").all():
        raise RobustPreparedError(f"missing/extra enriched cell type key on {slide}")
    aligned_labels = aligned[CELL_TYPE_FIELD]
    if aligned_labels.isna().any():
        raise RobustPreparedError(f"missing enriched cell type on {slide}")
    aligned_values = aligned_labels.astype(str).str.strip().to_numpy(dtype=np.str_)
    if not np.array_equal(aligned_values, np.asarray(cell_type_labels, dtype=np.str_)):
        raise RobustPreparedError(f"enriched cell type order changed on {slide}")
    expression = np.load(base_root / slide / "expression_log1p.npy", allow_pickle=False)
    if expression.shape != dataset.expression.shape:
        raise RobustPreparedError(f"raw/base expression shape mismatch on {slide}")
    recovered = np.expm1(expression.astype(np.float64))
    raw_counts = dataset.expression.astype(np.int32, copy=False)
    maximum_raw_error = float(np.max(np.abs(recovered - raw_counts)))
    if maximum_raw_error > ROUNDTRIP_TOLERANCE or not np.array_equal(
        np.rint(recovered).astype(np.int64), raw_counts.astype(np.int64)
    ):
        raise RobustPreparedError(f"base log1p count round-trip failed on {slide}")
    cp10k, cp10k_audit = robustness.panel_log_cp10k(expression)

    base_metadata = np.load(base_root / slide / "metadata.npy", allow_pickle=False)
    if not np.array_equal(base_metadata, dataset.metadata, equal_nan=True):
        raise RobustPreparedError(f"base/raw morphology differs on {slide}")
    base_qc = np.load(base_root / slide / "qc_passed.npy", allow_pickle=False)
    if not np.array_equal(base_qc.astype(bool), dataset.qc_passed.astype(bool)):
        raise RobustPreparedError(f"base/raw QC differs on {slide}")
    base_coordinates = np.load(base_root / slide / "coordinates_um.npy", allow_pickle=False)
    coordinate_error = float(
        np.max(np.abs(base_coordinates.astype(np.float64) - dataset.coordinates_um))
    )
    panel_total = raw_counts.sum(axis=1, dtype=np.float64)
    return SlideSource(
        expression_log1p=expression.astype(np.float32, copy=False),
        expression_cp10k=cp10k,
        raw_counts=raw_counts,
        metadata=dataset.metadata.astype(np.float32, copy=False),
        coordinates_um=dataset.coordinates_um.astype(np.float64, copy=False),
        fov=base_fov.astype(np.int32, copy=False),
        geometry_group=np.load(
            base_root / slide / "geometry_group.npy", allow_pickle=False
        ).astype(np.int16, copy=False),
        fold=np.load(base_root / slide / "fold.npy", allow_pickle=False).astype(
            np.int8, copy=False
        ),
        qc_passed=base_qc.astype(bool, copy=False),
        base_matched_eligible=np.load(
            base_root / slide / "matched_eligible.npy", allow_pickle=False
        ).astype(bool, copy=False),
        cell_type_labels=np.asarray(cell_type_labels, dtype=np.str_),
        panel_log_total=np.log1p(panel_total).astype(np.float32),
        audit={
            "maximum_raw_count_roundtrip_error": maximum_raw_error,
            "roundtrip_tolerance": ROUNDTRIP_TOLERANCE,
            "base_float32_to_raw_float64_coordinate_max_abs_error_um": coordinate_error,
            "cp10k": cp10k_audit,
        },
    )


def _build_csr(
    coordinates: np.ndarray,
    fov: np.ndarray,
    groups: np.ndarray,
    folds: np.ndarray,
    active: np.ndarray,
    partition_mode: str,
) -> tuple[csr_matrix, csr_matrix]:
    partition = fov if partition_mode == "within_fov" else groups
    near_rows: list[np.ndarray] = []
    near_sources: list[np.ndarray] = []
    annular_rows: list[np.ndarray] = []
    annular_sources: list[np.ndarray] = []
    for value in sorted(int(item) for item in np.unique(partition[active])):
        indices = np.flatnonzero(active & (partition == value))
        if len(indices) < 2:
            continue
        nr, ns, ar, ass, _ = robustness._partition_edges(  # noqa: SLF001
            coordinates[indices], k=12, near_max_um=25.0, annular_max_um=50.0
        )
        near_rows.append(indices[nr])
        near_sources.append(indices[ns])
        annular_rows.append(indices[ar])
        annular_sources.append(indices[ass])
    nodes = len(coordinates)

    def matrix(rows_parts: list[np.ndarray], source_parts: list[np.ndarray]) -> csr_matrix:
        rows = np.concatenate(rows_parts) if rows_parts else np.empty(0, dtype=np.int64)
        sources = (
            np.concatenate(source_parts) if source_parts else np.empty(0, dtype=np.int64)
        )
        if len(rows):
            if np.any(groups[rows] != groups[sources]) or np.any(folds[rows] != folds[sources]):
                raise RobustPreparedError("graph edge crossed a geometry component/fold")
            if np.any(rows == sources):
                raise RobustPreparedError("graph contains a self edge")
        result = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, sources)),
            shape=(nodes, nodes),
        ).tocsr()
        result.sum_duplicates()
        if len(result.data) and not np.all(result.data == 1):
            raise RobustPreparedError("graph contains duplicate receiver/source edges")
        return result

    return matrix(near_rows, near_sources), matrix(annular_rows, annular_sources)


def _weighted_adjacency(topology: csr_matrix) -> csr_matrix:
    degree = np.diff(topology.indptr)
    rows = np.repeat(np.arange(topology.shape[0], dtype=np.int64), degree)
    values = np.reciprocal(degree[rows].astype(np.float32))
    return csr_matrix(
        (values, topology.indices.copy(), topology.indptr.copy()),
        shape=topology.shape,
    )


def _scalar_means(
    near: csr_matrix,
    annular: csr_matrix,
    source_permutation: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    near_weighted = _weighted_adjacency(near)
    annular_weighted = _weighted_adjacency(annular)
    near_mean = np.asarray(near_weighted @ values, dtype=np.float32)
    annular_mean = np.asarray(annular_weighted @ values, dtype=np.float32)
    rows = np.repeat(np.arange(near.shape[0], dtype=np.int64), np.diff(near.indptr))
    effective = source_permutation[near.indices]
    if np.any(effective < 0) or np.any(effective == rows):
        raise RobustPreparedError("permuted CSR reintroduced receiver/missing source")
    permuted = csr_matrix(
        (near_weighted.data.copy(), effective, near.indptr.copy()), shape=near.shape
    )
    permuted_mean = np.asarray(permuted @ values, dtype=np.float32)
    return near_mean, annular_mean, permuted_mean


def _cell_type_proportions(
    near: csr_matrix,
    annular: csr_matrix,
    source_permutation: np.ndarray,
    codes: np.ndarray,
    level_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    one_hot = np.eye(level_count, dtype=np.float32)[codes]
    return _scalar_means(near, annular, source_permutation, one_hot)


def _check_csr_against_result(
    expression: np.ndarray,
    near: csr_matrix,
    annular: csr_matrix,
    result: robustness.RobustNeighborAggregates,
) -> None:
    if not np.array_equal(np.diff(near.indptr), result.near_degree) or not np.array_equal(
        np.diff(annular.indptr), result.annular_degree
    ):
        raise RobustPreparedError("saved CSR degrees disagree with robustness API")
    probes = np.unique(
        np.linspace(
            0, expression.shape[1] - 1, min(16, expression.shape[1])
        ).astype(int)
    )
    observed_near = np.asarray(_weighted_adjacency(near) @ expression[:, probes])
    observed_annular = np.asarray(_weighted_adjacency(annular) @ expression[:, probes])
    rows = np.repeat(np.arange(near.shape[0], dtype=np.int64), np.diff(near.indptr))
    effective = result.source_permutation[near.indices]
    if np.any(effective == rows):
        raise RobustPreparedError("saved mapping collides with a receiver")
    permuted = csr_matrix(
        (_weighted_adjacency(near).data.copy(), effective, near.indptr.copy()),
        shape=near.shape,
    )
    observed_permuted = np.asarray(permuted @ expression[:, probes])
    for observed, expected in (
        (observed_near, result.near_mean[:, probes]),
        (observed_annular, result.annular_mean[:, probes]),
        (observed_permuted, result.permuted_near_mean[:, probes]),
    ):
        if not np.allclose(observed, expected, rtol=0.0, atol=2e-6):
            raise RobustPreparedError("saved CSR does not replay a dense neighbor aggregate")


def _link_common_files(
    variant_root: Path,
    shared_root: Path,
    output_root: Path,
    expression_filename: str,
) -> None:
    _relative_symlink(
        variant_root / "genes.json", shared_root / "genes.json", output_root=output_root
    )
    _relative_symlink(
        variant_root / "metadata_names.json",
        shared_root / "metadata_names.json",
        output_root=output_root,
    )
    _relative_symlink(
        variant_root / "cell_type_levels.json",
        shared_root / "cell_type_levels.json",
        output_root=output_root,
    )
    _relative_symlink(
        variant_root / "eligible_genes.npy",
        shared_root / "eligible_genes.npy",
        output_root=output_root,
    )
    for slide in SLIDES:
        destination = variant_root / slide
        destination.mkdir(parents=True, exist_ok=False)
        for filename in COMMON_ARRAY_FILES:
            source_name = expression_filename if filename == "expression_log1p.npy" else filename
            _relative_symlink(
                destination / filename,
                shared_root / slide / source_name,
                output_root=output_root,
            )


def _write_graph_arrays(
    slide_root: Path,
    expression: np.ndarray,
    panel_log_total: np.ndarray,
    cell_type_codes: np.ndarray,
    level_count: int,
    near: csr_matrix,
    annular: csr_matrix,
    result: robustness.RobustNeighborAggregates,
    eligible_primary: np.ndarray,
) -> None:
    values = {
        "neighbor_near_mean.npy": result.near_mean,
        "neighbor_annular_mean.npy": result.annular_mean,
        "neighbor_permuted_near_mean.npy": result.permuted_near_mean,
        "near_degree.npy": result.near_degree,
        "annular_degree.npy": result.annular_degree,
        "permuted_near_degree.npy": result.permuted_near_degree,
        "matched_eligible.npy": result.matched_eligible,
        "eligible_primary.npy": eligible_primary,
        "source_permutation.npy": result.source_permutation,
        "near_indptr.npy": near.indptr.astype(np.int64, copy=False),
        "near_indices.npy": near.indices.astype(np.int32, copy=False),
        "annular_indptr.npy": annular.indptr.astype(np.int64, copy=False),
        "annular_indices.npy": annular.indices.astype(np.int32, copy=False),
    }
    panel_means = _scalar_means(near, annular, result.source_permutation, panel_log_total)
    for filename, value in zip(
        (
            "neighbor_near_panel_log_total_mean.npy",
            "neighbor_annular_panel_log_total_mean.npy",
            "neighbor_permuted_near_panel_log_total_mean.npy",
        ),
        panel_means,
        strict=True,
    ):
        values[filename] = value
    type_means = _cell_type_proportions(
        near, annular, result.source_permutation, cell_type_codes, level_count
    )
    for filename, value in zip(CELL_TYPE_PROPORTION_FILES, type_means, strict=True):
        values[filename] = value
    for filename, value in values.items():
        _save_array(slide_root / filename, value)
    _check_csr_against_result(expression, near, annular, result)


def _variant_integrity_payload(
    variant_root: Path,
    spec: VariantSpec,
    *,
    base_manifest: Mapping[str, Any],
    base_manifest_sha256: str,
    slide_audits: Mapping[str, Any],
) -> dict[str, Any]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in variant_root.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        record = _content_record(path, relative_to=variant_root)
        records[record["path"]] = record
    payload: dict[str, Any] = {
        "manifest_schema_version": 1,
        "variant_path_id": spec.variant_id,
        "contract_variant_id": spec.contract_variant_id,
        "preprocessing_version": PREPROCESSING_VERSION,
        "raw_snapshot": {"fingerprint": base_manifest["raw_snapshot"]["fingerprint"]},
        "split_fingerprint": base_manifest["split_fingerprint"],
        "base_prepared_manifest_sha256": base_manifest_sha256,
        "base_processed_fingerprint": base_manifest["processed_fingerprint"],
        "variant_spec": spec.as_dict(),
        "runner_required_files": list(RUNNER_REQUIRED_FILES),
        "slide_graph_audits": dict(slide_audits),
        "content": records,
    }
    fingerprint = canonical_sha256(payload)
    payload["variant_fingerprint"] = fingerprint
    return payload


def _runner_manifest(
    spec: VariantSpec,
    *,
    base_manifest: Mapping[str, Any],
    base_manifest_sha256: str,
    processed_fingerprint: str,
) -> dict[str, Any]:
    """Return the exact seven-key manifest consumed by the frozen runner."""

    return {
        "variant_id": spec.manifest_variant_id,
        "preprocessing_version": PREPROCESSING_VERSION,
        "raw_snapshot": {"fingerprint": base_manifest["raw_snapshot"]["fingerprint"]},
        "processed_fingerprint": processed_fingerprint,
        "split_fingerprint": base_manifest["split_fingerprint"],
        "variant_spec": spec.as_dict(),
        "base_prepared_manifest_sha256": base_manifest_sha256,
    }


def _verify_variant(root: Path, variant_root: Path, expected: VariantSpec) -> dict[str, Any]:
    manifest = _strict_json(variant_root / "manifest.json")
    expected_runner_keys = {
        "variant_id",
        "preprocessing_version",
        "raw_snapshot",
        "processed_fingerprint",
        "split_fingerprint",
        "variant_spec",
        "base_prepared_manifest_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_runner_keys:
        raise RobustPreparedError("runner-facing variant manifest keys are not exact")
    if manifest.get("variant_id") != expected.manifest_variant_id:
        raise RobustPreparedError(f"variant identity changed: {expected.variant_id}")
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise RobustPreparedError("variant preprocessing version changed")
    if manifest.get("variant_spec") != expected.as_dict():
        raise RobustPreparedError(f"variant spec changed: {expected.variant_id}")
    integrity = _strict_json(variant_root / "integrity_manifest.json")
    if (
        not isinstance(integrity, dict)
        or integrity.get("variant_path_id") != expected.variant_id
        or integrity.get("contract_variant_id") != expected.contract_variant_id
    ):
        raise RobustPreparedError("variant integrity identity changed")
    slide_graph_audits = integrity.get("slide_graph_audits")
    if not isinstance(slide_graph_audits, dict) or set(slide_graph_audits) != set(SLIDES):
        raise RobustPreparedError("variant slide graph audits are missing")
    content = integrity.get("content")
    if not isinstance(content, dict) or set(content) != {
        str(item["path"]) for item in content.values()
    }:
        raise RobustPreparedError("variant content inventory is invalid")
    observed_paths = {
        path.relative_to(variant_root).as_posix()
        for path in variant_root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "integrity_manifest.json"}
    }
    if observed_paths != set(content):
        raise RobustPreparedError("variant contains undeclared or missing content")
    for record in content.values():
        _verify_record(variant_root, record, root)
    if "eligible_genes.npy" not in content:
        raise RobustPreparedError("variant lacks the frozen eligible-gene mask")
    for slide in SLIDES:
        for filename in RUNNER_REQUIRED_FILES:
            if f"{slide}/{filename}" not in content:
                raise RobustPreparedError(
                    f"runner-required file missing: {expected.variant_id}/{slide}/{filename}"
                )
        near_degree = np.load(variant_root / slide / "near_degree.npy", allow_pickle=False)
        annular_degree = np.load(
            variant_root / slide / "annular_degree.npy", allow_pickle=False
        )
        permuted_degree = np.load(
            variant_root / slide / "permuted_near_degree.npy", allow_pickle=False
        )
        near_indptr = np.load(variant_root / slide / "near_indptr.npy", allow_pickle=False)
        annular_indptr = np.load(
            variant_root / slide / "annular_indptr.npy", allow_pickle=False
        )
        if not np.array_equal(np.diff(near_indptr), near_degree) or not np.array_equal(
            np.diff(annular_indptr), annular_degree
        ):
            raise RobustPreparedError("variant CSR/degree mismatch")
        if not np.array_equal(near_degree, permuted_degree):
            raise RobustPreparedError("variant permutation does not preserve degree")
        native = np.load(
            variant_root / slide / "matched_eligible.npy", allow_pickle=False
        ).astype(bool)
        primary = np.load(
            variant_root / slide / "eligible_primary.npy", allow_pickle=False
        ).astype(bool)
        qc = np.load(variant_root / slide / "qc_passed.npy", allow_pickle=False).astype(bool)
        source_permutation = np.load(
            variant_root / slide / "source_permutation.npy", allow_pickle=False
        )
        fov = np.load(variant_root / slide / "fov.npy", allow_pickle=False)
        groups = np.load(
            variant_root / slide / "geometry_group.npy", allow_pickle=False
        )
        folds = np.load(variant_root / slide / "fold.npy", allow_pickle=False)
        near_indices = np.load(
            variant_root / slide / "near_indices.npy", allow_pickle=False
        )
        annular_indices = np.load(
            variant_root / slide / "annular_indices.npy", allow_pickle=False
        )
        near_rows = np.repeat(np.arange(len(near_degree), dtype=np.int64), near_degree)
        annular_rows = np.repeat(
            np.arange(len(annular_degree), dtype=np.int64), annular_degree
        )
        active = source_permutation >= 0
        expected_active = np.ones(len(qc), dtype=bool) if expected.node_policy == "all" else qc
        if not np.array_equal(active, expected_active):
            raise RobustPreparedError("source-permutation active-node mask changed")
        observed_fixed_by_fov: dict[str, int] = {}
        for fov_value in np.unique(fov[active]):
            nodes = np.flatnonzero(active & (fov == fov_value))
            if not np.array_equal(
                np.sort(source_permutation[nodes].astype(np.int64)), nodes.astype(np.int64)
            ):
                raise RobustPreparedError("source permutation is not a within-FOV bijection")
            fixed_count = int(np.sum(source_permutation[nodes] == nodes))
            if fixed_count:
                observed_fixed_by_fov[str(int(fov_value))] = fixed_count
        effective = source_permutation[near_indices]
        if np.any(effective < 0) or np.any(effective == near_rows):
            raise RobustPreparedError("source permutation collides with a receiver")
        slide_audit = slide_graph_audits.get(slide)
        if not isinstance(slide_audit, dict):
            raise RobustPreparedError("variant slide graph audit is invalid")
        expected_fixed_by_fov = slide_audit.get("permutation_fixed_sources_by_fov")
        expected_fixed_count = int(sum(observed_fixed_by_fov.values()))
        if (
            expected_fixed_by_fov != observed_fixed_by_fov
            or slide_audit.get("permutation_fixed_source_count") != expected_fixed_count
            or slide_audit.get("permutation_fov_count_with_fixed_sources")
            != len(observed_fixed_by_fov)
            or slide_audit.get("permutation_mapping_policy")
            != "receiver_collision_free_fov_bijection_maximizing_changed_sources"
        ):
            raise RobustPreparedError("permutation fixed-source audit changed")
        expected_changed_fraction = (int(np.sum(active)) - expected_fixed_count) / int(
            np.sum(active)
        )
        if (
            abs(
                float(slide_audit.get("permutation_source_mapping_changed_fraction", -1.0))
                - expected_changed_fraction
            )
            > 1e-15
        ):
            raise RobustPreparedError("permutation changed-source fraction changed")
        for rows, indices in ((near_rows, near_indices), (annular_rows, annular_indices)):
            if (
                np.any(groups[rows] != groups[indices])
                or np.any(folds[rows] != folds[indices])
                or np.any(rows == indices)
            ):
                raise RobustPreparedError("saved graph violates split/self-edge isolation")
            if expected.partition_mode == "within_fov" and np.any(fov[rows] != fov[indices]):
                raise RobustPreparedError("within-FOV variant contains a cross-FOV edge")
        expected_native = active & (near_degree >= 4) & (annular_degree >= 4)
        if not np.array_equal(native, expected_native):
            raise RobustPreparedError("native eligibility does not match graph degrees")
        base_primary = np.load(
            root / "shared" / slide / "base_matched_eligible.npy", allow_pickle=False
        ).astype(bool)
        expected_primary = base_primary & (qc if expected.node_policy == "qc_induced" else True)
        if not np.array_equal(primary, expected_primary):
            raise RobustPreparedError("primary eligibility does not match the fixed-v1 cohort")
        if expected.node_policy == "qc_induced" and (
            np.any(native & ~qc) or np.any(primary & ~qc)
        ):
            raise RobustPreparedError("QC-induced variant contains an ineligible QC-fail node")
    payload = dict(integrity)
    variant_fingerprint = payload.pop("variant_fingerprint", None)
    observed = canonical_sha256(payload)
    if observed != variant_fingerprint or observed != manifest["processed_fingerprint"]:
        raise RobustPreparedError(f"variant fingerprint changed: {expected.variant_id}")
    return {"variant_id": expected.variant_id, "variant_fingerprint": observed}


def verify_prepared(output_root: Path, *, base_root: Path | None = None) -> dict[str, Any]:
    candidate_root = Path(output_root)
    if candidate_root.is_symlink():
        raise RobustPreparedError("prepared output root may not be a symlink")
    root = candidate_root.resolve(strict=True)
    if not root.is_dir():
        raise RobustPreparedError("prepared output root must be a regular directory")
    manifest = _strict_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("manifest_schema_version") != 1:
        raise RobustPreparedError("robustness root manifest schema is unsupported")
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise RobustPreparedError("robustness root preprocessing version changed")
    if set(manifest.get("variants", {})) != set(VARIANT_BY_ID):
        raise RobustPreparedError("robustness variant inventory is not exact")
    if base_root is not None:
        base_manifest, base_sha, _, _ = _verify_base(base_root.resolve(strict=True))
        if base_sha != manifest.get("base_prepared_manifest_sha256"):
            raise RobustPreparedError("source base prepared manifest changed")
        if base_manifest["processed_fingerprint"] != manifest.get(
            "base_processed_fingerprint"
        ):
            raise RobustPreparedError("source base processed fingerprint changed")
    shared = manifest.get("shared_content")
    if not isinstance(shared, dict):
        raise RobustPreparedError("shared content inventory is missing")
    observed_shared = {
        path.relative_to(root).as_posix()
        for path in (root / "shared").rglob("*")
        if path.is_file()
    }
    if observed_shared != set(shared):
        raise RobustPreparedError("shared tree contains undeclared or missing content")
    for record in shared.values():
        _verify_record(root, record, root)
    eligible_genes = np.load(root / "shared/eligible_genes.npy", allow_pickle=False)
    eligible_digest = _eligible_gene_hash(eligible_genes)
    if (
        eligible_digest != manifest.get("eligible_gene_mask_sha256")
        or int(np.sum(eligible_genes)) != int(manifest.get("eligible_gene_count", -1))
    ):
        raise RobustPreparedError("root frozen eligible-gene identity changed")
    genes = _strict_json(root / "shared/genes.json")
    if len(genes) == 1000 and (
        eligible_digest != FROZEN_ELIGIBLE_GENE_SHA256
        or int(np.sum(eligible_genes)) != FROZEN_ELIGIBLE_GENE_COUNT
    ):
        raise RobustPreparedError("production frozen 932-gene identity changed")
    variants = []
    for spec in VARIANT_SPECS:
        record = manifest["variants"][spec.variant_id]
        variant_root = _safe_child(root, str(record["path"]))
        if sha256_file(variant_root / "manifest.json") != record["manifest_sha256"]:
            raise RobustPreparedError(f"variant manifest hash changed: {spec.variant_id}")
        if sha256_file(variant_root / "integrity_manifest.json") != record[
            "integrity_manifest_sha256"
        ]:
            raise RobustPreparedError(
                f"variant integrity manifest hash changed: {spec.variant_id}"
            )
        observed = _verify_variant(root, variant_root, spec)
        if observed["variant_fingerprint"] != record["variant_fingerprint"]:
            raise RobustPreparedError(f"root/variant fingerprint mismatch: {spec.variant_id}")
        variants.append(observed)
    payload = dict(manifest)
    expected_processed = payload.pop("processed_fingerprint", None)
    payload.pop("created_at", None)
    observed_processed = canonical_sha256(payload)
    if observed_processed != expected_processed:
        raise RobustPreparedError("robustness processed fingerprint changed")
    return {
        "verified": True,
        "output_root": str(root),
        "processed_fingerprint": observed_processed,
        "variant_count": len(variants),
        "variants": variants,
    }


def _prepare_static_variant_slide(
    variant_slide_root: Path,
    spec: VariantSpec,
    source: SlideSource,
    cell_type_codes: np.ndarray,
    level_count: int,
) -> dict[str, Any]:
    expression = (
        source.expression_log1p
        if spec.normalization == "log1p_raw_count"
        else source.expression_cp10k
    )
    active = (
        np.ones(len(expression), dtype=bool)
        if spec.node_policy == "all"
        else source.qc_passed
    )
    result = robustness.build_robust_neighbor_aggregates(
        expression,
        source.coordinates_um,
        source.fov,
        source.geometry_group,
        slide=str(source.audit["slide"]),
        partition_mode=spec.partition_mode,
        active_nodes=active,
        permutation_seed=PERMUTATION_SEED,
    )
    near, annular = _build_csr(
        source.coordinates_um,
        source.fov,
        source.geometry_group,
        source.fold,
        active,
        spec.partition_mode,
    )
    eligible_primary = source.base_matched_eligible & (
        source.qc_passed if spec.node_policy == "qc_induced" else True
    )
    _write_graph_arrays(
        variant_slide_root,
        expression,
        source.panel_log_total,
        cell_type_codes,
        level_count,
        near,
        annular,
        result,
        eligible_primary,
    )
    return dict(result.audit)


def prepare(
    base_root: Path,
    raw_root: Path,
    output_root: Path,
    *,
    expected_gene_count: int = 1000,
    eligible_genes_path: Path | None = None,
    frozen_eligible_genes: np.ndarray | None = None,
    source_loader: Callable[..., SlideSource] = _load_slide_source,
    cell_type_loader: Callable[[Path, Path], dict[str, np.ndarray]] = _load_cell_type_labels,
) -> dict[str, Any]:
    base = base_root.resolve(strict=True)
    raw = raw_root.resolve(strict=True)
    output = output_root.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"robustness prepared output already exists: {output}")
    base_manifest, base_manifest_sha, genes, metadata_names = _verify_base(base)
    if len(genes) != expected_gene_count:
        raise RobustPreparedError(
            f"expected {expected_gene_count} genes, base contains {len(genes)}"
        )
    if frozen_eligible_genes is None:
        mask_path = (
            eligible_genes_path.resolve(strict=True)
            if eligible_genes_path is not None
            else (
                current_paths().report_root / DEFAULT_ELIGIBLE_GENE_RELATIVE_PATH
            ).resolve(strict=True)
        )
        eligible_genes = _load_frozen_eligible_genes(
            mask_path,
            genes,
            enforce_production_identity=expected_gene_count == 1000,
        )
    else:
        eligible_genes = np.asarray(frozen_eligible_genes, dtype=bool)
        if eligible_genes.shape != (expected_gene_count,):
            raise RobustPreparedError("supplied eligible-gene mask has the wrong shape")
        if expected_gene_count == 1000 and (
            int(np.sum(eligible_genes)) != FROZEN_ELIGIBLE_GENE_COUNT
            or _eligible_gene_hash(eligible_genes) != FROZEN_ELIGIBLE_GENE_SHA256
        ):
            raise RobustPreparedError("supplied production eligible-gene identity changed")
    cell_type_labels = cell_type_loader(raw, base)
    levels = sorted({str(value) for slide in SLIDES for value in cell_type_labels[slide]})
    if len(levels) != EXPECTED_CELL_TYPE_LEVELS:
        raise RobustPreparedError("cell type level count changed after alignment")
    level_lookup = {value: index for index, value in enumerate(levels)}

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shared_root = staging / "shared"
        variants_root = staging / "variants"
        shared_root.mkdir()
        variants_root.mkdir()
        _write_json(shared_root / "genes.json", genes)
        _write_json(shared_root / "metadata_names.json", metadata_names)
        _write_json(shared_root / "cell_type_levels.json", levels)
        _save_array(shared_root / "eligible_genes.npy", eligible_genes)

        slide_sources: dict[str, SlideSource] = {}
        slide_codes: dict[str, np.ndarray] = {}
        source_audits: dict[str, Any] = {}
        for slide in SLIDES:
            source = source_loader(
                raw,
                base,
                slide,
                genes,
                metadata_names,
                cell_type_labels[slide],
                expected_gene_count=expected_gene_count,
            )
            source.audit["slide"] = slide
            codes = np.asarray(
                [level_lookup[str(value)] for value in source.cell_type_labels],
                dtype=np.int16,
            )
            if np.any(codes < 0) or np.any(codes >= len(levels)):
                raise RobustPreparedError(f"invalid cell type code on {slide}")
            slide_sources[slide] = source
            slide_codes[slide] = codes
            source_audits[slide] = source.audit
            slide_shared = shared_root / slide
            slide_shared.mkdir()
            shared_arrays = {
                "expression_log1p.npy": source.expression_log1p,
                "expression_cp10k_log1p.npy": source.expression_cp10k,
                "metadata.npy": source.metadata,
                "coordinates_um.npy": source.coordinates_um,
                "fov.npy": source.fov,
                "geometry_group.npy": source.geometry_group,
                "fold.npy": source.fold,
                "qc_passed.npy": source.qc_passed,
                "cell_type_code.npy": codes,
                "panel_log_total.npy": source.panel_log_total,
                "base_matched_eligible.npy": source.base_matched_eligible,
            }
            for filename, value in shared_arrays.items():
                _save_array(slide_shared / filename, value)

        variant_audits: dict[str, dict[str, Any]] = {}
        # Materialize independent topology/normalization variants first.
        for spec in STATIC_VARIANTS:
            variant_root = variants_root / spec.variant_id
            variant_root.mkdir()
            expression_filename = (
                "expression_log1p.npy"
                if spec.normalization == "log1p_raw_count"
                else "expression_cp10k_log1p.npy"
            )
            _link_common_files(variant_root, shared_root, staging, expression_filename)
            audits: dict[str, Any] = {}
            for slide in SLIDES:
                audits[slide] = _prepare_static_variant_slide(
                    variant_root / slide,
                    spec,
                    slide_sources[slide],
                    slide_codes[slide],
                    len(levels),
                )
            variant_audits[spec.variant_id] = audits
            integrity = _variant_integrity_payload(
                variant_root,
                spec,
                base_manifest=base_manifest,
                base_manifest_sha256=base_manifest_sha,
                slide_audits=audits,
            )
            _write_json(variant_root / "integrity_manifest.json", integrity)
            manifest = _runner_manifest(
                spec,
                base_manifest=base_manifest,
                base_manifest_sha256=base_manifest_sha,
                processed_fingerprint=integrity["variant_fingerprint"],
            )
            _write_json(variant_root / "manifest.json", manifest)

        # V4/V6 are lightweight overlays over the exact V0 graph/null arrays.
        v0_root = variants_root / "v0_within_fov_log1p_all"
        for spec in (item for item in VARIANT_SPECS if item.graph_source_variant):
            variant_root = variants_root / spec.variant_id
            variant_root.mkdir()
            _link_common_files(
                variant_root, shared_root, staging, "expression_log1p.npy"
            )
            for slide in SLIDES:
                destination = variant_root / slide
                source_slide = v0_root / slide
                for filename in (*GRAPH_ARRAY_FILES, *CELL_TYPE_PROPORTION_FILES):
                    _relative_symlink(
                        destination / filename,
                        source_slide / filename,
                        output_root=staging,
                    )
            audits = variant_audits[spec.graph_source_variant]
            variant_audits[spec.variant_id] = audits
            integrity = _variant_integrity_payload(
                variant_root,
                spec,
                base_manifest=base_manifest,
                base_manifest_sha256=base_manifest_sha,
                slide_audits=audits,
            )
            _write_json(variant_root / "integrity_manifest.json", integrity)
            manifest = _runner_manifest(
                spec,
                base_manifest=base_manifest,
                base_manifest_sha256=base_manifest_sha,
                processed_fingerprint=integrity["variant_fingerprint"],
            )
            _write_json(variant_root / "manifest.json", manifest)

        shared_content: dict[str, Any] = {}
        for path in sorted(item for item in shared_root.rglob("*") if item.is_file()):
            record = _content_record(path, relative_to=staging)
            shared_content[record["path"]] = record
        variants: dict[str, Any] = {}
        for spec in VARIANT_SPECS:
            path = variants_root / spec.variant_id / "manifest.json"
            value = _strict_json(path)
            integrity_path = variants_root / spec.variant_id / "integrity_manifest.json"
            variants[spec.variant_id] = {
                "path": f"variants/{spec.variant_id}",
                "manifest_sha256": sha256_file(path),
                "integrity_manifest_sha256": sha256_file(integrity_path),
                "variant_fingerprint": value["processed_fingerprint"],
                "contract_variant_id": spec.contract_variant_id,
            }
        root_payload: dict[str, Any] = {
            "manifest_schema_version": 1,
            "preprocessing_version": PREPROCESSING_VERSION,
            "created_at": _utc_now(),
            "raw_snapshot": {"fingerprint": base_manifest["raw_snapshot"]["fingerprint"]},
            "split_fingerprint": base_manifest["split_fingerprint"],
            "base_prepared_manifest_sha256": base_manifest_sha,
            "base_processed_fingerprint": base_manifest["processed_fingerprint"],
            "permutation_seed": PERMUTATION_SEED,
            "cell_type_field": CELL_TYPE_FIELD,
            "cell_type_level_count": len(levels),
            "eligible_gene_count": int(np.sum(eligible_genes)),
            "eligible_gene_mask_sha256": _eligible_gene_hash(eligible_genes),
            "source_audits": source_audits,
            "shared_content": shared_content,
            "variants": variants,
        }
        root_fingerprint_payload = dict(root_payload)
        root_fingerprint_payload.pop("created_at")
        root_payload["processed_fingerprint"] = canonical_sha256(
            root_fingerprint_payload
        )
        _write_json(staging / "manifest.json", root_payload)
        verification = verify_prepared(staging, base_root=base)
        portable_verification = dict(verification)
        portable_verification["output_root"] = "."
        _write_json(staging / "verification.json", portable_verification)
        staging.rename(output)
        return {
            **verification,
            "output_root": str(output),
            "base_prepared_manifest_sha256": base_manifest_sha,
            "maximum_raw_count_roundtrip_error": max(
                float(source_audits[slide]["maximum_raw_count_roundtrip_error"])
                for slide in SLIDES
            ),
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args() -> argparse.Namespace:
    paths = current_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=paths.data_root / "processed/same_gene_cross_cell_jacobian_v1",
    )
    parser.add_argument(
        "--eligible-genes-npz",
        type=Path,
        default=paths.report_root / DEFAULT_ELIGIBLE_GENE_RELATIVE_PATH,
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=paths.data_root / "raw/Gastric_Cancer_Analysis",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.data_root / "processed/same_gene_robustness_v1",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = (
        verify_prepared(arguments.output, base_root=arguments.base)
        if arguments.verify_only
        else prepare(
            arguments.base,
            arguments.raw_root,
            arguments.output,
            eligible_genes_path=arguments.eligible_genes_npz,
        )
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
