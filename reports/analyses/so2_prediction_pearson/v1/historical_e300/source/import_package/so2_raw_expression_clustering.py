"""Classical raw-expression clustering for SO2 cores 15 through 28.

This workflow is intentionally independent of every trained model and spatial
graph.  It starts from checksum-verified integer expression counts, applies a
CosMx-aware total-count normalization, computes a conventional scaled PCA,
then constructs a sparse expression-space kNN graph for seeded Leiden
clustering.  Core identity and coordinates are joined only after labels have
been frozen, for audit tables and tissue-coordinate maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import hmac
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .fingerprints import sha256_file
from .paths import ProjectPaths
from .relative_qkv_embedding_clustering import (
    KNNGraphResult,
    _array_sha256,
    _atomic_save_figure_pair,
    _atomic_write_csv,
    _atomic_write_json,
    _atomic_write_npy,
    _atomic_write_parquet,
    _atomic_write_text,
    _canonical_sha256,
    _file_manifest,
    _file_record,
    _read_json,
    _receipt_with_self_hash,
    _style_spatial_axis,
    _verify_self_hash,
    _write_deterministic_npz,
    build_faiss_cosine_knn_graph,
    deterministic_glasbey_palette,
    run_seeded_leiden,
)
from .so2_hl_clustering import cluster_summary_tables
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_N_GENES,
    EXPECTED_TOTAL_CELLS,
    SO2_ALIASES,
    SO2_CORE_NUMBERS,
    _array_sha256 as _source_array_sha256,
)
from .so2_relative_graphs import _verify_so2_cohort_manifest


ANALYSIS_SCHEMA = "so2_14core_raw_expression_clustering_v1"
PREPROCESSING_SCHEMA = "so2_14core_raw_expression_preprocessing_v1"
CLUSTERING_SCHEMA = "so2_14core_raw_expression_leiden_v1"
FIGURE_SCHEMA = "so2_14core_raw_expression_spatial_figure_v1"
EXPECTED_COHORT_MANIFEST_FILE_SHA256 = (
    "69efdf5e3d1d816ab98e52e2734c51434e3dd95968021c0867c11f4d804bc158"
)
EXPECTED_COHORT_MANIFEST_CONTENT_SHA256 = (
    "c7abbeddd8ed018d09163bb290224d3a92fd38ab3f3696d012b731f2f05f9bf0"
)
EXPECTED_GENE_ORDER_SHA256 = (
    "046eb86c7ea8f1fe6977598a0190132340400fc61802fcde63ab5ac0e9502b03"
)
DEFAULT_NORMALIZATION_TARGET = 197.0
DEFAULT_LIBRARY_SIZE_FLOOR = 20.0
DEFAULT_SCALE_CLIP = 10.0
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_N_NEIGHBORS = 30
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_DPI = 300
LABEL_PREFIX = "E"
ANALYSIS_ID = "rawexpr_pca50_cosinek30_leiden_r1p0_s20260825"


class SO2RawExpressionClusteringError(ValueError):
    """Raised when the independent raw-expression contract is violated."""


@dataclass(frozen=True, slots=True)
class RawExpressionInputs:
    cohort_dir: Path
    manifest_path: Path
    manifest: Mapping[str, Any] = field(repr=False)
    gene_names: tuple[str, ...]
    source_records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ExpressionPCAResult:
    normalized_scores: np.ndarray = field(repr=False)
    components: np.ndarray = field(repr=False)
    gene_mean: np.ndarray = field(repr=False)
    gene_scale: np.ndarray = field(repr=False)
    retained_mask: np.ndarray = field(repr=False)
    receipt: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_cpu_device(device: str) -> str:
    """Fail closed rather than allowing a baseline analysis to reserve a GPU."""

    if str(device).strip().lower() != "cpu":
        raise SO2RawExpressionClusteringError(
            "SO2 raw-expression clustering is CPU-only; --device must be 'cpu'."
        )
    return "cpu"


def requested_panel_order() -> tuple[int, ...]:
    return SO2_CORE_NUMBERS


def spatial_plot_spec(palette: Mapping[str, str]) -> dict[str, Any]:
    return {
        "panel_order": list(SO2_CORE_NUMBERS),
        "grid_shape": [3, 5],
        "legend_panel": [2, 4],
        "panel_title_template": "SO2 Core {core_number}",
        "core_numbers_identifiable": True,
        "equal_aspect": True,
        "invert_y_axis": True,
        "coordinate_units": "micrometres",
        "coordinates_used_for_clustering": False,
        "one_shared_joint_cluster_palette": True,
        "palette": dict(palette),
    }


def _validate_locked_parameters(
    *,
    normalization_target: float,
    library_size_floor: float,
    scale_clip: float,
    pca_components: int,
    n_neighbors: int,
    leiden_resolution: float,
    random_seed: int,
) -> dict[str, Any]:
    observed = {
        "normalization_target": float(normalization_target),
        "library_size_floor": float(library_size_floor),
        "scale_clip": float(scale_clip),
        "pca_components": int(pca_components),
        "n_neighbors": int(n_neighbors),
        "leiden_resolution": float(leiden_resolution),
        "random_seed": int(random_seed),
    }
    expected = {
        "normalization_target": DEFAULT_NORMALIZATION_TARGET,
        "library_size_floor": DEFAULT_LIBRARY_SIZE_FLOOR,
        "scale_clip": DEFAULT_SCALE_CLIP,
        "pca_components": DEFAULT_PCA_COMPONENTS,
        "n_neighbors": DEFAULT_N_NEIGHBORS,
        "leiden_resolution": DEFAULT_LEIDEN_RESOLUTION,
        "random_seed": DEFAULT_RANDOM_SEED,
    }
    if observed != expected:
        raise SO2RawExpressionClusteringError(
            f"Primary raw-expression analysis parameters are locked: {expected}."
        )
    return {
        **observed,
        "input_representation": "raw_nonnegative_integer_biological_probe_counts",
        "normalization": "counts * target / max(cell_total, library_size_floor)",
        "nonlinear_transform": "log1p",
        "gene_selection": "all_nonconstant_genes_from_fixed_targeted_panel",
        "gene_scaling": "joint_mean_center_unit_sample_variance_then_clip",
        "pca": "exact_clipped_scaled_feature_covariance_eigendecomposition",
        "pca_component_sign_rule": "largest_absolute_loading_positive",
        "pca_scores_l2_normalized": True,
        "distance_metric": "cosine",
        "knn_implementation": "faiss.IndexHNSWFlat",
        "knn_symmetrization": "undirected_union_unweighted",
        "leiden_partition": "leidenalg.RBConfigurationVertexPartition",
        "cluster_sort": "descending_size_then_minimum_global_cell_index_then_raw_id",
        "joint_core_order": list(SO2_CORE_NUMBERS),
        "joint_cell_count": EXPECTED_TOTAL_CELLS,
        "label_prefix": LABEL_PREFIX,
        "model_checkpoint_used": False,
        "model_embedding_used": False,
        "metadata_used_for_clustering": False,
        "core_identity_used_for_clustering": False,
        "coordinates_used_for_clustering": False,
        "spatial_graph_used_for_clustering": False,
        "batch_correction_applied": False,
        "dense_cell_by_cell_matrix_constructed": False,
        "device": "cpu",
    }


def resolve_raw_expression_inputs(
    *,
    paths: ProjectPaths,
    cohort_dir: str | Path | None = None,
) -> RawExpressionInputs:
    """Resolve and checksum-verify the immutable prepared raw-count cohort."""

    if cohort_dir is None:
        root = paths.data_root / "processed" / "so2_14core_relative_qkv_v1"
    else:
        root = Path(cohort_dir).expanduser()
        if not root.is_absolute():
            root = paths.project_root / root
        root = root.resolve(strict=False)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SO2RawExpressionClusteringError(
            f"SO2 prepared cohort manifest is missing: {manifest_path}"
        )
    if not hmac.compare_digest(
        sha256_file(manifest_path), EXPECTED_COHORT_MANIFEST_FILE_SHA256
    ):
        raise SO2RawExpressionClusteringError("SO2 cohort manifest file changed.")
    try:
        manifest = _verify_so2_cohort_manifest(root)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SO2RawExpressionClusteringError(
            "SO2 prepared cohort verification failed."
        ) from exc
    _verify_self_hash(manifest, label="SO2 prepared cohort manifest")
    if manifest.get("manifest_content_sha256") != (
        EXPECTED_COHORT_MANIFEST_CONTENT_SHA256
    ):
        raise SO2RawExpressionClusteringError("SO2 cohort fingerprint changed.")
    cohort = manifest.get("cohort")
    features = manifest.get("features")
    preprocessing = manifest.get("preprocessing")
    assurances = manifest.get("assurances")
    if not all(isinstance(value, Mapping) for value in (cohort, features, preprocessing, assurances)):
        raise SO2RawExpressionClusteringError("SO2 cohort manifest schema is incomplete.")
    gene_names = tuple(str(value) for value in features.get("gene_names", ()))
    if any(
        (
            tuple(cohort.get("aliases", ())) != SO2_ALIASES,
            tuple(cohort.get("original_core_numbers", ())) != SO2_CORE_NUMBERS,
            int(cohort.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            len(gene_names) != EXPECTED_N_GENES,
            len(set(gene_names)) != EXPECTED_N_GENES,
            _canonical_sha256(list(gene_names)) != EXPECTED_GENE_ORDER_SHA256,
            preprocessing.get("expression_source") != "raw_counts",
            assurances.get("unmapped_fov_246_excluded") is not True,
        )
    ):
        raise SO2RawExpressionClusteringError("SO2 raw-count cohort identity changed.")
    if any(name.startswith(("Negative", "SystemControl")) for name in gene_names):
        raise SO2RawExpressionClusteringError(
            "Technical controls entered the biological expression panel."
        )
    core_records = manifest.get("cores")
    if not isinstance(core_records, list) or len(core_records) != len(SO2_ALIASES):
        raise SO2RawExpressionClusteringError("SO2 core records are incomplete.")
    records: list[Mapping[str, Any]] = []
    for alias, core_number, record in zip(
        SO2_ALIASES, SO2_CORE_NUMBERS, core_records, strict=True
    ):
        if not isinstance(record, Mapping) or any(
            (
                record.get("alias") != alias,
                int(record.get("original_core_number", -1)) != core_number,
                int(record.get("cell_count", -1))
                != EXPECTED_CELL_COUNTS_BY_CORE[core_number],
            )
        ):
            raise SO2RawExpressionClusteringError(
                f"SO2 source record changed for {alias}."
            )
        records.append(dict(record))
    return RawExpressionInputs(
        cohort_dir=root,
        manifest_path=manifest_path,
        manifest=manifest,
        gene_names=gene_names,
        source_records=tuple(records),
    )


def log_normalize_counts(
    counts: np.ndarray | sparse.spmatrix,
    *,
    normalization_target: float,
    library_size_floor: float,
) -> tuple[sparse.csr_matrix, dict[str, np.ndarray], Mapping[str, Any]]:
    """Apply the locked CosMx floor normalization and log1p without dropping rows."""

    if not math.isfinite(float(normalization_target)) or normalization_target <= 0:
        raise SO2RawExpressionClusteringError(
            "normalization_target must be finite and positive."
        )
    if not math.isfinite(float(library_size_floor)) or library_size_floor <= 0:
        raise SO2RawExpressionClusteringError(
            "library_size_floor must be finite and positive."
        )
    if sparse.issparse(counts):
        matrix = sparse.csr_matrix(counts, copy=True)
        if matrix.ndim != 2:
            raise SO2RawExpressionClusteringError(
                "Counts must have shape [cells, genes]."
            )
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        raw_data = np.asarray(matrix.data)
        if raw_data.size and (
            not np.isfinite(raw_data).all()
            or np.any(raw_data < 0)
            or not np.equal(raw_data, np.floor(raw_data)).all()
        ):
            raise SO2RawExpressionClusteringError(
                "Raw counts must be finite, nonnegative integers."
            )
        raw_library_size = np.asarray(matrix.sum(axis=1)).reshape(-1).astype(np.float64)
        detected_genes = np.diff(matrix.indptr).astype(np.int32, copy=False)
    else:
        values = np.asarray(counts)
        if values.ndim != 2:
            raise SO2RawExpressionClusteringError(
                "Counts must have shape [cells, genes]."
            )
        if not np.isfinite(values).all() or np.any(values < 0) or not np.equal(
            values, np.floor(values)
        ).all():
            raise SO2RawExpressionClusteringError(
                "Raw counts must be finite, nonnegative integers."
            )
        raw_library_size = values.sum(axis=1, dtype=np.float64)
        detected_genes = np.count_nonzero(values, axis=1).astype(np.int32)
        matrix = sparse.csr_matrix(values)
        matrix.eliminate_zeros()
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise SO2RawExpressionClusteringError(
            "Raw-count matrix must contain at least two cells and one gene."
        )
    if np.any(raw_library_size <= 0.0):
        raise SO2RawExpressionClusteringError(
            "Zero-library cells are not permitted in the locked cohort."
        )

    denominator = np.maximum(raw_library_size, float(library_size_floor))
    factors = float(normalization_target) / denominator
    normalized = matrix.astype(np.float32, copy=True)
    repeated = np.repeat(factors.astype(np.float32), np.diff(normalized.indptr))
    normalized.data *= repeated
    normalized.data = np.log1p(normalized.data).astype(np.float32, copy=False)
    normalized.eliminate_zeros()
    if not np.isfinite(normalized.data).all() or normalized.shape != matrix.shape:
        raise SO2RawExpressionClusteringError(
            "Raw-count normalization produced invalid values."
        )
    below_floor = raw_library_size < float(library_size_floor)
    qc = {
        "raw_library_size": raw_library_size.astype(np.int64),
        "detected_genes": detected_genes,
        "normalization_denominator": denominator.astype(np.float64),
        "below_library_size_floor": below_floor.astype(bool),
    }
    receipt = {
        "method": "cosmx_total_count_floor_then_log1p",
        "normalization_target": float(normalization_target),
        "library_size_floor": float(library_size_floor),
        "cells_retained": int(matrix.shape[0]),
        "cells_dropped": 0,
        "genes": int(matrix.shape[1]),
        "raw_nonzero_entries": int(matrix.nnz),
        "normalized_nonzero_entries": int(normalized.nnz),
        "sparse_output": True,
        "below_library_size_floor_cells": int(np.count_nonzero(below_floor)),
        "raw_library_size_minimum": int(raw_library_size.min()),
        "raw_library_size_median": float(np.median(raw_library_size)),
        "raw_library_size_maximum": int(raw_library_size.max()),
        "row_order_unchanged": True,
        "gene_order_unchanged": True,
    }
    return normalized, qc, receipt


def exact_scaled_sparse_pca(
    log_matrix: sparse.spmatrix,
    *,
    n_components: int,
    clip_value: float = DEFAULT_SCALE_CLIP,
) -> ExpressionPCAResult:
    """Compute exact centered PCA of gene-z-scored, clipped sparse log counts.

    A dense cell-by-gene or cell-by-cell matrix is never constructed.  The
    clipped scaled matrix is represented as a per-gene baseline plus sparse
    deviations, permitting an exact 1,000-by-1,000 covariance calculation.
    """

    matrix = sparse.csr_matrix(log_matrix, dtype=np.float64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    n_cells, n_genes = matrix.shape
    if n_cells < 2 or n_genes < 1 or not np.isfinite(matrix.data).all():
        raise SO2RawExpressionClusteringError(
            "PCA input must be a finite sparse [cells, genes] matrix."
        )
    if isinstance(n_components, bool) or int(n_components) <= 0:
        raise SO2RawExpressionClusteringError("PCA components must be positive.")
    if not math.isfinite(float(clip_value)) or clip_value <= 0:
        raise SO2RawExpressionClusteringError("Scale clip must be positive.")

    gene_sum = np.asarray(matrix.sum(axis=0)).reshape(-1)
    gene_sum_squares = np.asarray(matrix.multiply(matrix).sum(axis=0)).reshape(-1)
    gene_mean = gene_sum / float(n_cells)
    centered_ss = gene_sum_squares - float(n_cells) * np.square(gene_mean)
    centered_ss = np.maximum(centered_ss, 0.0)
    gene_variance = centered_ss / float(n_cells - 1)
    gene_scale = np.sqrt(gene_variance)
    tolerance = np.finfo(np.float64).eps * max(1.0, float(gene_variance.max())) * 32.0
    retained_mask = gene_variance > tolerance
    retained_indices = np.flatnonzero(retained_mask)
    if retained_indices.size < 2:
        raise SO2RawExpressionClusteringError(
            "Fewer than two nonconstant genes remain for PCA."
        )
    retained = matrix[:, retained_indices].tocsr(copy=True)
    retained_mean = gene_mean[retained_indices]
    retained_scale = gene_scale[retained_indices]

    baseline = np.clip(
        -retained_mean / retained_scale,
        -float(clip_value),
        float(clip_value),
    )
    columns = retained.indices
    transformed_nonzero = np.clip(
        (retained.data - retained_mean[columns]) / retained_scale[columns],
        -float(clip_value),
        float(clip_value),
    )
    retained.data = transformed_nonzero - baseline[columns]
    retained.eliminate_zeros()
    deviation_sum = np.asarray(retained.sum(axis=0)).reshape(-1)
    clipped_mean = baseline + deviation_sum / float(n_cells)

    cross = (retained.T @ retained).toarray()
    ztz = (
        cross
        + float(n_cells) * np.outer(baseline, baseline)
        + np.outer(baseline, deviation_sum)
        + np.outer(deviation_sum, baseline)
    )
    covariance = (ztz - float(n_cells) * np.outer(clipped_mean, clipped_mean)) / float(
        n_cells - 1
    )
    covariance = (covariance + covariance.T) * 0.5
    if covariance.shape != (retained_indices.size, retained_indices.size):
        raise SO2RawExpressionClusteringError("Feature covariance shape is invalid.")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues, kind="stable")
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    retained_components = min(
        int(n_components), n_cells - 1, int(retained_indices.size)
    )
    components_retained = np.ascontiguousarray(
        eigenvectors[:, order[:retained_components]], dtype=np.float64
    )
    for column in range(components_retained.shape[1]):
        pivot = int(np.argmax(np.abs(components_retained[:, column])))
        if components_retained[pivot, column] < 0.0:
            components_retained[:, column] *= -1.0

    scores = np.asarray(retained @ components_retained, dtype=np.float64)
    scores += (baseline - clipped_mean) @ components_retained
    row_norms = np.linalg.norm(scores, axis=1)
    if np.any(~np.isfinite(row_norms)) or np.any(row_norms <= 0.0):
        raise SO2RawExpressionClusteringError(
            "PCA scores contain a non-finite or zero-norm cell."
        )
    scores /= row_norms[:, None]
    normalized_scores = np.ascontiguousarray(scores, dtype=np.float32)
    components = np.zeros((n_genes, retained_components), dtype=np.float64)
    components[retained_indices] = components_retained
    total_variance = float(eigenvalues.sum())
    explained = eigenvalues[:retained_components]
    ratios = explained / total_variance
    receipt = {
        "method": "exact_sparse_affine_feature_covariance_eigendecomposition",
        "input_shape": [int(n_cells), int(n_genes)],
        "input_sparse_nonzero_entries": int(matrix.nnz),
        "requested_components": int(n_components),
        "retained_components": int(retained_components),
        "input_genes": int(n_genes),
        "nonconstant_genes": int(retained_indices.size),
        "zero_variance_genes_excluded": int(n_genes - retained_indices.size),
        "gene_centering": True,
        "gene_scaling": "sample_standard_deviation_ddof_1",
        "scale_clip_absolute_value": float(clip_value),
        "feature_covariance_shape": list(covariance.shape),
        "maximum_dense_matrix_shape": list(covariance.shape),
        "dense_cell_by_gene_matrix_constructed": False,
        "cell_by_cell_matrix_constructed": False,
        "component_sign_rule": "largest_absolute_loading_positive",
        "l2_normalized_after_pca": True,
        "explained_variance": explained.tolist(),
        "explained_variance_ratio": ratios.tolist(),
        "total_explained_variance_ratio": float(ratios.sum()),
        "gene_mean_sha256": _array_sha256("log_gene_mean", gene_mean),
        "gene_scale_sha256": _array_sha256("log_gene_scale", gene_scale),
        "retained_mask_sha256": _array_sha256("retained_gene_mask", retained_mask),
        "components_sha256": _array_sha256("pca_components", components),
        "normalized_scores_sha256": _array_sha256(
            "normalized_expression_pca_scores", normalized_scores
        ),
    }
    return ExpressionPCAResult(
        normalized_scores=normalized_scores,
        components=components,
        gene_mean=np.ascontiguousarray(gene_mean),
        gene_scale=np.ascontiguousarray(gene_scale),
        retained_mask=np.ascontiguousarray(retained_mask),
        receipt=receipt,
    )


def cluster_expression_scores(
    scores: np.ndarray,
    *,
    n_neighbors: int,
    leiden_resolution: float,
    random_seed: int,
) -> tuple[np.ndarray, Mapping[str, Any], KNNGraphResult]:
    """Cluster expression PCA scores; no spatial or sample fields are accepted."""

    values = np.ascontiguousarray(np.asarray(scores), dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SO2RawExpressionClusteringError("Expression PCA scores are invalid.")
    knn = build_faiss_cosine_knn_graph(
        values, n_neighbors=n_neighbors, random_seed=random_seed
    )
    leiden = run_seeded_leiden(
        knn,
        n_cells=len(values),
        resolution=leiden_resolution,
        random_seed=random_seed,
    )
    labels = np.ascontiguousarray(leiden.labels, dtype=np.int64)
    return labels, {"knn": knn.receipt, "leiden": leiden.receipt}, knn


def deterministic_expression_palette(cluster_count: int) -> dict[str, str]:
    """Return a stable categorical palette with expression-only ``E`` labels.

    The shared palette generator intentionally accepts only the model's two
    representation namespaces.  Reuse its deterministic CIELAB color
    selection, rotate the sequence so this independent baseline does not start
    on the intrinsic anchor, and assign the expression-specific prefix here.
    """

    base = deterministic_glasbey_palette(cluster_count, namespace="intrinsic")
    colors = list(base.values())
    if len(colors) > 1:
        colors = colors[1:] + colors[:1]
    return {f"{LABEL_PREFIX}{index}": color for index, color in enumerate(colors)}


def _verify_stage_files(
    output_root: Path,
    files: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if not files:
        raise SO2RawExpressionClusteringError(f"{label} has no file checksums.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2RawExpressionClusteringError(
                f"{label} output is missing: {relative}"
            )
        if _file_record(path) != dict(record):
            raise SO2RawExpressionClusteringError(
                f"{label} output checksum changed: {relative}"
            )


def _computational_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove plotting-only settings from reusable computational stages."""

    return {
        str(key): value
        for key, value in configuration.items()
        if str(key) != "figure_dpi"
    }


def _source_file_records(inputs: RawExpressionInputs) -> dict[str, Any]:
    result: dict[str, Any] = {
        "cohort_manifest": {
            "sha256": sha256_file(inputs.manifest_path),
            "content_sha256": inputs.manifest["manifest_content_sha256"],
            "size_bytes": int(inputs.manifest_path.stat().st_size),
        },
        "gene_order_sha256": EXPECTED_GENE_ORDER_SHA256,
        "cores": {},
    }
    files = inputs.manifest["files"]
    for alias, core_number, record in zip(
        SO2_ALIASES, SO2_CORE_NUMBERS, inputs.source_records, strict=True
    ):
        relative = f"cores/{alias}.npz"
        result["cores"][alias] = {
            "core_number": int(core_number),
            "cell_count": EXPECTED_CELL_COUNTS_BY_CORE[core_number],
            "file_sha256": str(files[relative]),
            "expression_counts_component_sha256": str(
                record["component_checksums"]["expression_counts"]
            ),
            "coordinates_component_sha256": str(
                record["component_checksums"]["coordinates_um"]
            ),
        }
    return result


def _load_and_normalize_raw_counts(
    *,
    inputs: RawExpressionInputs,
    normalization_target: float,
    library_size_floor: float,
) -> tuple[sparse.csr_matrix, pd.DataFrame, Mapping[str, Any]]:
    """Load expression counts only and preserve canonical core/row order."""

    normalized_shards: list[sparse.csr_matrix] = []
    qc_frames: list[pd.DataFrame] = []
    core_receipts: list[dict[str, Any]] = []
    global_start = 0
    for alias, core_number, source_record in zip(
        SO2_ALIASES, SO2_CORE_NUMBERS, inputs.source_records, strict=True
    ):
        path = inputs.cohort_dir / "cores" / f"{alias}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {
                    "expression_counts",
                    "target_expression",
                    "node_covariates",
                    "coordinates_um",
                }:
                    raise SO2RawExpressionClusteringError(
                        f"Prepared source schema changed for {alias}."
                    )
                # Deliberately access no other archive array at this stage.
                counts = np.array(archive["expression_counts"], copy=True)
        except (OSError, ValueError, KeyError) as exc:
            raise SO2RawExpressionClusteringError(
                f"Cannot load raw expression counts for {alias}."
            ) from exc
        expected_cells = EXPECTED_CELL_COUNTS_BY_CORE[core_number]
        if counts.shape != (expected_cells, EXPECTED_N_GENES) or counts.dtype != np.int32:
            raise SO2RawExpressionClusteringError(
                f"Raw expression shape or dtype changed for {alias}."
            )
        observed_component = _source_array_sha256("expression_counts", counts)
        expected_component = str(
            source_record["component_checksums"]["expression_counts"]
        )
        if not hmac.compare_digest(observed_component, expected_component):
            raise SO2RawExpressionClusteringError(
                f"Raw expression component checksum changed for {alias}."
            )
        normalized, qc, normalization_receipt = log_normalize_counts(
            counts,
            normalization_target=normalization_target,
            library_size_floor=library_size_floor,
        )
        del counts
        normalized_shards.append(normalized)
        local_index = np.arange(expected_cells, dtype=np.int64)
        qc_frames.append(
            pd.DataFrame(
                {
                    "global_cell_index": np.arange(
                        global_start,
                        global_start + expected_cells,
                        dtype=np.int64,
                    ),
                    "cell_index": local_index,
                    "cell_key": [
                        f"{alias}:{int(index):08d}" for index in local_index
                    ],
                    "core_alias": alias,
                    "core_number": np.full(
                        expected_cells, core_number, dtype=np.int16
                    ),
                    "raw_library_size": qc["raw_library_size"],
                    "detected_genes": qc["detected_genes"],
                    "normalization_denominator": qc[
                        "normalization_denominator"
                    ],
                    "below_library_size_floor": qc[
                        "below_library_size_floor"
                    ],
                }
            )
        )
        global_start += expected_cells
        core_receipts.append(
            {
                "alias": alias,
                "core_number": int(core_number),
                "cell_count": int(expected_cells),
                "prepared_row_order_preserved": True,
                "source_file_sha256": sha256_file(path),
                "source_expression_component_sha256": observed_component,
                "normalization": dict(normalization_receipt),
                "normalized_sparse_data_sha256": _array_sha256(
                    f"{alias}_normalized_log1p_data", normalized.data
                ),
                "normalized_sparse_indices_sha256": _array_sha256(
                    f"{alias}_normalized_log1p_indices", normalized.indices
                ),
                "normalized_sparse_indptr_sha256": _array_sha256(
                    f"{alias}_normalized_log1p_indptr", normalized.indptr
                ),
            }
        )

    matrix = sparse.vstack(normalized_shards, format="csr", dtype=np.float32)
    qc_frame = pd.concat(qc_frames, ignore_index=True)
    del normalized_shards, qc_frames
    if any(
        (
            matrix.shape != (EXPECTED_TOTAL_CELLS, EXPECTED_N_GENES),
            len(qc_frame) != EXPECTED_TOTAL_CELLS,
            tuple(qc_frame["core_number"].drop_duplicates().tolist())
            != SO2_CORE_NUMBERS,
            not qc_frame["global_cell_index"].is_unique,
            not qc_frame["cell_key"].is_unique,
            not np.array_equal(
                qc_frame["global_cell_index"].to_numpy(dtype=np.int64),
                np.arange(EXPECTED_TOTAL_CELLS, dtype=np.int64),
            ),
        )
    ):
        raise SO2RawExpressionClusteringError(
            "Joint raw-expression row ordering or coverage changed."
        )
    totals = qc_frame["raw_library_size"].to_numpy(dtype=np.int64)
    eligible = totals[totals >= int(library_size_floor)]
    observed_target = float(np.median(eligible))
    if observed_target != float(normalization_target):
        raise SO2RawExpressionClusteringError(
            "Locked normalization target no longer equals the eligible-cell median "
            f"({observed_target} != {normalization_target})."
        )
    receipt = {
        "source_array_access": ["expression_counts"],
        "prohibited_arrays_accessed": False,
        "joint_core_order": list(SO2_CORE_NUMBERS),
        "joint_shape": list(matrix.shape),
        "joint_sparse_nonzero_entries": int(matrix.nnz),
        "cells_retained": EXPECTED_TOTAL_CELLS,
        "cells_dropped": 0,
        "below_library_size_floor_cells": int(
            qc_frame["below_library_size_floor"].sum()
        ),
        "eligible_cell_median_library_size": observed_target,
        "core_receipts": core_receipts,
    }
    return matrix, qc_frame, receipt


def _preprocessing_paths(output_root: Path) -> dict[str, Path]:
    root = output_root / "preprocessing"
    return {
        "manifest": root / "preprocessing_manifest.json",
        "scores": root / "expression_pca_l2_normalized.npy",
        "gene_statistics": root / "expression_gene_statistics.npz",
        "parameters": root / "expression_preprocessing_parameters.json",
        "gene_names": root / "gene_names.json",
        "cell_qc": root / "cell_source_qc.parquet",
    }


def _verify_preprocessing_receipt(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    inputs: RawExpressionInputs,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="SO2 raw-expression preprocessing manifest")
    if any(
        (
            receipt.get("schema") != PREPROCESSING_SCHEMA,
            receipt.get("status") != "complete",
            _computational_configuration(receipt.get("configuration", {}))
            != _computational_configuration(configuration),
            receipt.get("source_cohort_manifest_sha256")
            != sha256_file(inputs.manifest_path),
            tuple(receipt.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(receipt.get("gene_count", -1)) != EXPECTED_N_GENES,
        )
    ):
        raise SO2RawExpressionClusteringError(
            "Raw-expression preprocessing receipt identity is invalid."
        )
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise SO2RawExpressionClusteringError(
            "Raw-expression preprocessing receipt lacks files."
        )
    _verify_stage_files(output_root, files, label="raw-expression preprocessing")
    scores = np.load(_preprocessing_paths(output_root)["scores"], allow_pickle=False)
    retained_components = int(receipt["pca"]["retained_components"])
    if scores.shape != (EXPECTED_TOTAL_CELLS, retained_components):
        raise SO2RawExpressionClusteringError("Stored expression PCA shape changed.")
    if not np.isfinite(scores).all() or not np.allclose(
        np.linalg.norm(scores.astype(np.float64), axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise SO2RawExpressionClusteringError(
            "Stored expression PCA scores are invalid."
        )
    qc = pd.read_parquet(_preprocessing_paths(output_root)["cell_qc"])
    if len(qc) != EXPECTED_TOTAL_CELLS or tuple(
        qc["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise SO2RawExpressionClusteringError("Stored source/QC rows are invalid.")


def prepare_expression_representation(
    *,
    inputs: RawExpressionInputs,
    output_root: Path,
    configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Create or verify the resumable expression preprocessing/PCA stage."""

    paths = _preprocessing_paths(output_root)
    if paths["manifest"].is_file():
        receipt = _read_json(
            paths["manifest"], label="SO2 raw-expression preprocessing manifest"
        )
        _verify_preprocessing_receipt(
            output_root=output_root,
            receipt=receipt,
            inputs=inputs,
            configuration=configuration,
        )
        return receipt
    stage_root = paths["manifest"].parent
    if stage_root.exists() and any(stage_root.iterdir()):
        raise SO2RawExpressionClusteringError(
            "Partial preprocessing outputs exist without a complete receipt."
        )
    stage_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    matrix, qc_frame, loading = _load_and_normalize_raw_counts(
        inputs=inputs,
        normalization_target=float(configuration["normalization_target"]),
        library_size_floor=float(configuration["library_size_floor"]),
    )
    pca = exact_scaled_sparse_pca(
        matrix,
        n_components=int(configuration["pca_components"]),
        clip_value=float(configuration["scale_clip"]),
    )
    del matrix
    gc.collect()

    _atomic_write_npy(paths["scores"], pca.normalized_scores)
    _write_deterministic_npz(
        paths["gene_statistics"],
        {
            "gene_mean_log_normalized": pca.gene_mean,
            "gene_scale_log_normalized": pca.gene_scale,
            "gene_retained_mask": pca.retained_mask,
            "pca_components": pca.components,
        },
    )
    _atomic_write_json(
        paths["gene_names"],
        {
            "gene_names": list(inputs.gene_names),
            "gene_order_sha256": EXPECTED_GENE_ORDER_SHA256,
            "technical_controls_included": False,
        },
    )
    _atomic_write_json(
        paths["parameters"],
        {
            "configuration": _computational_configuration(configuration),
            "loading_and_normalization": loading,
            "pca": pca.receipt,
        },
    )
    _atomic_write_parquet(paths["cell_qc"], qc_frame)
    stage_files = {
        path.relative_to(output_root).as_posix(): _file_record(path)
        for name, path in paths.items()
        if name != "manifest"
    }
    receipt = _receipt_with_self_hash(
        {
            "schema": PREPROCESSING_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "configuration": _computational_configuration(configuration),
            "source_cohort_manifest_sha256": sha256_file(inputs.manifest_path),
            "source_cohort_content_sha256": inputs.manifest[
                "manifest_content_sha256"
            ],
            "source_artifacts": _source_file_records(inputs),
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "gene_count": EXPECTED_N_GENES,
            "loading_and_normalization": loading,
            "pca": pca.receipt,
            "elapsed_seconds": time.monotonic() - started,
            "clustering_inputs": ["normalized_log1p_scaled_raw_expression_pca"],
            "prohibited_inputs": {
                "checkpoint": False,
                "trained_model": False,
                "h0": False,
                "hL": False,
                "delta_h": False,
                "node_covariates": False,
                "core_identity": False,
                "coordinates": False,
                "spatial_graph": False,
                "relative_geometry": False,
                "clinical_or_vendor_labels": False,
            },
            "files": stage_files,
        }
    )
    _atomic_write_json(paths["manifest"], receipt)
    _verify_preprocessing_receipt(
        output_root=output_root,
        receipt=receipt,
        inputs=inputs,
        configuration=configuration,
    )
    return receipt


def _load_plotting_coordinates(
    *,
    inputs: RawExpressionInputs,
) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...]]:
    """Load coordinates only after expression labels have been frozen."""

    coordinates_by_core: list[np.ndarray] = []
    receipts: list[Mapping[str, Any]] = []
    for alias, core_number, source_record in zip(
        SO2_ALIASES, SO2_CORE_NUMBERS, inputs.source_records, strict=True
    ):
        path = inputs.cohort_dir / "cores" / f"{alias}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                coordinates = np.array(archive["coordinates_um"], copy=True)
        except (OSError, ValueError, KeyError) as exc:
            raise SO2RawExpressionClusteringError(
                f"Cannot load plotting coordinates for {alias}."
            ) from exc
        expected_cells = EXPECTED_CELL_COUNTS_BY_CORE[core_number]
        if coordinates.shape != (expected_cells, 2) or coordinates.dtype != np.float64:
            raise SO2RawExpressionClusteringError(
                f"Plotting-coordinate schema changed for {alias}."
            )
        if not np.isfinite(coordinates).all():
            raise SO2RawExpressionClusteringError(
                f"Plotting coordinates are non-finite for {alias}."
            )
        observed = _source_array_sha256("coordinates_um", coordinates)
        expected = str(source_record["component_checksums"]["coordinates_um"])
        if not hmac.compare_digest(observed, expected):
            raise SO2RawExpressionClusteringError(
                f"Plotting-coordinate checksum changed for {alias}."
            )
        coordinates_by_core.append(coordinates)
        receipts.append(
            {
                "alias": alias,
                "core_number": int(core_number),
                "cell_count": int(expected_cells),
                "coordinates_component_sha256": observed,
                "role": "post_clustering_spatial_plotting_only",
            }
        )
    combined = np.ascontiguousarray(np.concatenate(coordinates_by_core, axis=0))
    if combined.shape != (EXPECTED_TOTAL_CELLS, 2):
        raise SO2RawExpressionClusteringError(
            "Combined plotting coordinates changed shape."
        )
    return combined, tuple(receipts)


def _clustering_paths(output_root: Path) -> dict[str, Path]:
    clustering = output_root / "clustering"
    tables = output_root / "tables"
    return {
        "manifest": clustering / "clustering_manifest.json",
        "labels": clustering / "expression_labels.npy",
        "scores": _preprocessing_paths(output_root)["scores"],
        "edges": clustering / "expression_knn_undirected_edges.npy",
        "parameters": clustering / "expression_clustering_parameters.json",
        "palette": clustering / "expression_palette.json",
        "cell_table": tables / "cell_expression_clusters.parquet",
        "summary": tables / "expression_cluster_summary.csv",
        "composition": tables / "expression_cluster_core_composition.csv",
        "qc_summary": tables / "expression_cluster_qc_summary.csv",
    }


def _verify_clustering_receipt(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    preprocessing_manifest_sha256: str,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="SO2 raw-expression clustering manifest")
    if any(
        (
            receipt.get("schema") != CLUSTERING_SCHEMA,
            receipt.get("status") != "complete",
            _computational_configuration(receipt.get("configuration", {}))
            != _computational_configuration(configuration),
            receipt.get("preprocessing_manifest_sha256")
            != preprocessing_manifest_sha256,
            tuple(receipt.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO2RawExpressionClusteringError(
            "Raw-expression clustering receipt identity is invalid."
        )
    files = receipt.get("files")
    if not isinstance(files, Mapping):
        raise SO2RawExpressionClusteringError(
            "Raw-expression clustering receipt lacks files."
        )
    _verify_stage_files(output_root, files, label="raw-expression clustering")
    paths = _clustering_paths(output_root)
    labels = np.load(paths["labels"], allow_pickle=False)
    expected_hash = receipt.get("pipeline", {}).get("leiden", {}).get(
        "labels_sha256"
    )
    if (
        labels.shape != (EXPECTED_TOTAL_CELLS,)
        or np.any(labels < 0)
        or _array_sha256("sorted_leiden_labels", labels) != expected_hash
    ):
        raise SO2RawExpressionClusteringError(
            "Stored raw-expression labels are invalid."
        )
    table = pd.read_parquet(paths["cell_table"])
    if any(
        (
            len(table) != EXPECTED_TOTAL_CELLS,
            tuple(table["core_number"].drop_duplicates().tolist())
            != SO2_CORE_NUMBERS,
            not np.array_equal(
                table["expression_cluster_number"].to_numpy(dtype=np.int64),
                labels,
            ),
            not np.isfinite(table[["x_um", "y_um"]].to_numpy()).all(),
        )
    ):
        raise SO2RawExpressionClusteringError(
            "Stored raw-expression cell table is incomplete or misaligned."
        )


def _expression_qc_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = sorted(
        frame["expression_cluster"].unique(), key=lambda value: int(str(value)[1:])
    )
    for label in labels:
        selected = frame.loc[frame["expression_cluster"] == label]
        library = selected["raw_library_size"].to_numpy(dtype=np.float64)
        detected = selected["detected_genes"].to_numpy(dtype=np.float64)
        low_count = selected["below_library_size_floor"].to_numpy(dtype=bool)
        rows.append(
            {
                "cluster": str(label),
                "cluster_number": int(str(label)[1:]),
                "size": int(len(selected)),
                "raw_library_size_min": int(library.min()),
                "raw_library_size_median": float(np.median(library)),
                "raw_library_size_mean": float(library.mean()),
                "raw_library_size_max": int(library.max()),
                "detected_genes_min": int(detected.min()),
                "detected_genes_median": float(np.median(detected)),
                "detected_genes_mean": float(detected.mean()),
                "detected_genes_max": int(detected.max()),
                "below_library_size_floor_count": int(low_count.sum()),
                "below_library_size_floor_proportion": float(low_count.mean()),
            }
        )
    result = pd.DataFrame(rows)
    if int(result["size"].sum()) != len(frame):
        raise SO2RawExpressionClusteringError("Cluster QC summary dropped cells.")
    return result


def cluster_joint_raw_expression(
    *,
    inputs: RawExpressionInputs,
    output_root: Path,
    preprocessing_receipt: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Create or verify the joint sparse kNN/Leiden clustering stage."""

    paths = _clustering_paths(output_root)
    preprocessing_manifest = _preprocessing_paths(output_root)["manifest"]
    preprocessing_sha = sha256_file(preprocessing_manifest)
    if paths["manifest"].is_file():
        receipt = _read_json(
            paths["manifest"], label="SO2 raw-expression clustering manifest"
        )
        _verify_clustering_receipt(
            output_root=output_root,
            receipt=receipt,
            preprocessing_manifest_sha256=preprocessing_sha,
            configuration=configuration,
        )
        return receipt
    clustering_root = paths["manifest"].parent
    tables_root = paths["cell_table"].parent
    if (clustering_root.exists() and any(clustering_root.iterdir())) or (
        tables_root.exists() and any(tables_root.iterdir())
    ):
        raise SO2RawExpressionClusteringError(
            "Partial clustering outputs exist without a complete receipt."
        )
    clustering_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    scores = np.load(paths["scores"], allow_pickle=False)
    expected_components = int(preprocessing_receipt["pca"]["retained_components"])
    if scores.shape != (EXPECTED_TOTAL_CELLS, expected_components):
        raise SO2RawExpressionClusteringError(
            "Expression PCA scores do not cover all ordered cells."
        )
    labels, pipeline, knn = cluster_expression_scores(
        scores,
        n_neighbors=int(configuration["n_neighbors"]),
        leiden_resolution=float(configuration["leiden_resolution"]),
        random_seed=int(configuration["random_seed"]),
    )
    # Labels are complete before coordinates or core fields are loaded into this
    # stage.  Those fields cannot affect PCA, kNN, Leiden, or label sorting.
    frozen_labels_hash = _array_sha256("sorted_leiden_labels", labels)
    qc_frame = pd.read_parquet(_preprocessing_paths(output_root)["cell_qc"])
    coordinates, coordinate_receipts = _load_plotting_coordinates(inputs=inputs)
    if len(qc_frame) != len(labels) or len(coordinates) != len(labels):
        raise SO2RawExpressionClusteringError(
            "Post-clustering label/QC/coordinate row counts differ."
        )
    cell_frame = qc_frame.copy()
    cell_frame["x_um"] = coordinates[:, 0]
    cell_frame["y_um"] = coordinates[:, 1]
    cell_frame["expression_cluster_number"] = labels.astype(np.int32)
    cell_frame["expression_cluster"] = [f"{LABEL_PREFIX}{int(value)}" for value in labels]
    if _array_sha256("sorted_leiden_labels", labels) != frozen_labels_hash:
        raise SO2RawExpressionClusteringError(
            "Joining plotting fields changed frozen expression labels."
        )
    summary, composition, dominated = cluster_summary_tables(
        labels, cell_frame, prefix=LABEL_PREFIX
    )
    qc_summary = _expression_qc_summary(cell_frame)
    palette = deterministic_expression_palette(len(summary))

    _atomic_write_npy(paths["labels"], labels)
    _atomic_write_npy(paths["edges"], knn.edge_pairs)
    complete_pipeline = {
        "representation": "normalized_log1p_scaled_raw_expression_pca",
        "pca_scores_sha256": _array_sha256(
            "normalized_expression_pca_scores", scores
        ),
        "pca_score_shape": list(scores.shape),
        **dict(pipeline),
    }
    _atomic_write_json(paths["parameters"], complete_pipeline)
    _atomic_write_json(
        paths["palette"],
        {
            "representation": "raw_expression",
            "label_prefix": LABEL_PREFIX,
            "method": "deterministic_greedy_farthest_point_CIELAB",
            "colors": palette,
        },
    )
    _atomic_write_parquet(
        paths["cell_table"],
        cell_frame.loc[
            :,
            [
                "global_cell_index",
                "cell_index",
                "cell_key",
                "core_alias",
                "core_number",
                "x_um",
                "y_um",
                "raw_library_size",
                "detected_genes",
                "normalization_denominator",
                "below_library_size_floor",
                "expression_cluster_number",
                "expression_cluster",
            ],
        ],
    )
    _atomic_write_csv(paths["summary"], summary)
    _atomic_write_csv(paths["composition"], composition)
    _atomic_write_csv(paths["qc_summary"], qc_summary)
    stage_files = {
        path.relative_to(output_root).as_posix(): _file_record(path)
        for name, path in paths.items()
        if name not in {"manifest", "scores"}
    }
    receipt = _receipt_with_self_hash(
        {
            "schema": CLUSTERING_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "configuration": _computational_configuration(configuration),
            "preprocessing_manifest_sha256": preprocessing_sha,
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "cluster_count": int(len(summary)),
            "cluster_size_range": [
                int(summary["size"].min()),
                int(summary["size"].max()),
            ],
            "core_dominated_gt_90pct": dominated,
            "pipeline": complete_pipeline,
            "palette": palette,
            "frozen_labels_sha256_before_plotting_fields": frozen_labels_hash,
            "coordinates_loaded_after_labels_frozen": True,
            "coordinate_receipts": list(coordinate_receipts),
            "clustering_input_fields": [
                "normalized_log1p_scaled_raw_expression_pca"
            ],
            "coordinates_used_for_clustering": False,
            "core_identity_used_for_clustering": False,
            "metadata_used_for_clustering": False,
            "spatial_graph_used_for_clustering": False,
            "model_or_embedding_used_for_clustering": False,
            "elapsed_seconds": time.monotonic() - started,
            "files": stage_files,
        }
    )
    _atomic_write_json(paths["manifest"], receipt)
    _verify_clustering_receipt(
        output_root=output_root,
        receipt=receipt,
        preprocessing_manifest_sha256=preprocessing_sha,
        configuration=configuration,
    )
    return receipt


def _cluster_legend_handles(palette: Mapping[str, str]) -> list[Any]:
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in sorted(
            palette.items(), key=lambda item: int(str(item[0])[1:])
        )
    ]


def _atomic_save_png(figure: Any, *, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="png",
            dpi=int(dpi),
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "spatial_benchmark.so2_raw_expression_clustering"},
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _render_spatial_figures(
    frame: pd.DataFrame,
    *,
    palette: Mapping[str, str],
    resolution: float,
    output_root: Path,
    dpi: int,
) -> tuple[Path, Path, tuple[Path, ...]]:
    import matplotlib.pyplot as plt

    figure_dir = output_root / "figures"
    per_core_dir = figure_dir / "per_core"
    figure_dir.mkdir(parents=True, exist_ok=True)
    per_core_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "raw_expression_leiden_resolution_1p0_spatial_14cores.png"
    pdf_path = figure_dir / "raw_expression_leiden_resolution_1p0_spatial_14cores.pdf"

    figure, axes = plt.subplots(3, 5, figsize=(25.0, 15.0))
    for axis, core_number in zip(
        axes.ravel()[: len(SO2_CORE_NUMBERS)], SO2_CORE_NUMBERS, strict=True
    ):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected["expression_cluster"].map(palette)
        if len(selected) != EXPECTED_CELL_COUNTS_BY_CORE[core_number]:
            raise SO2RawExpressionClusteringError(
                f"Spatial panel is incomplete for SO2 core {core_number}."
            )
        if colors.isna().any():
            raise SO2RawExpressionClusteringError("Spatial palette is incomplete.")
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=0.52,
            c=colors.tolist(),
            marker="o",
            linewidths=0,
            edgecolors="none",
            alpha=0.92,
            rasterized=True,
        )
        axis.set_title(
            f"SO2 Core {core_number}\n$n$ = {len(selected):,}",
            fontsize=12,
            weight="bold",
            pad=7,
        )
        axis.text(
            0.025,
            0.97,
            f"CORE {core_number}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            weight="bold",
            color="#111827",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#CBD5E1",
                "alpha": 0.92,
            },
            zorder=6,
        )
        _style_spatial_axis(axis, coordinates)
    legend_axis = axes.ravel()[-1]
    legend_axis.axis("off")
    handles = _cluster_legend_handles(palette)
    legend_axis.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=3 if len(handles) > 24 else 2 if len(handles) > 14 else 1,
        title="Joint expression-only\ncluster",
        fontsize=8,
        title_fontsize=10,
        borderaxespad=0.0,
    )
    figure.suptitle(
        "SO2 raw-count-derived expression Leiden clusters — joint 14-core clustering\n"
        f"CosMx floor-normalized, log1p, scaled PCA; resolution {resolution:g}",
        fontsize=18,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "Classical expression-space clusters; no cell-type annotation is implied.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.045,
        top=0.925,
        wspace=0.27,
        hspace=0.34,
    )
    _atomic_save_figure_pair(
        figure, png_path=png_path, pdf_path=pdf_path, dpi=int(dpi)
    )
    plt.close(figure)

    per_core_paths: list[Path] = []
    for core_number in SO2_CORE_NUMBERS:
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected["expression_cluster"].map(palette)
        individual, axis = plt.subplots(figsize=(10.0, 9.0))
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=1.15,
            c=colors.tolist(),
            marker="o",
            linewidths=0,
            edgecolors="none",
            alpha=0.94,
            rasterized=True,
        )
        axis.set_title(
            f"SO2 Core {core_number} — expression-only Leiden clusters\n"
            f"resolution {resolution:g}; $n$ = {len(selected):,}",
            fontsize=15,
            weight="bold",
            pad=10,
        )
        _style_spatial_axis(axis, coordinates)
        handles = _cluster_legend_handles(palette)
        axis.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            frameon=False,
            ncol=2 if len(handles) > 20 else 1,
            fontsize=8,
            title="Expression cluster",
        )
        individual.subplots_adjust(right=0.80)
        path = per_core_dir / (
            f"core_{core_number}_raw_expression_leiden_resolution_1p0.png"
        )
        _atomic_save_png(individual, path=path, dpi=int(dpi))
        plt.close(individual)
        per_core_paths.append(path)
    return png_path, pdf_path, tuple(per_core_paths)


def _verify_figure_receipt(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    clustering_manifest_sha256: str,
) -> None:
    _verify_self_hash(receipt, label="SO2 raw-expression figure manifest")
    if any(
        (
            receipt.get("schema") != FIGURE_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("clustering_manifest_sha256")
            != clustering_manifest_sha256,
            tuple(receipt.get("plot_specification", {}).get("panel_order", ()))
            != SO2_CORE_NUMBERS,
            receipt.get("plot_specification", {}).get("grid_shape") != [3, 5],
            int(receipt.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO2RawExpressionClusteringError(
            "Raw-expression figure receipt identity is invalid."
        )
    files = receipt.get("files")
    if not isinstance(files, Mapping) or len(files) != 16:
        raise SO2RawExpressionClusteringError(
            "Raw-expression combined/per-core figures are incomplete."
        )
    _verify_stage_files(output_root, files, label="raw-expression figures")


def render_expression_spatial_maps(
    *,
    output_root: Path,
    clustering_receipt: Mapping[str, Any],
    resolution: float,
    dpi: int,
) -> Mapping[str, Any]:
    """Render or verify combined and per-core tissue-coordinate cluster maps."""

    if isinstance(dpi, bool) or int(dpi) < 72:
        raise SO2RawExpressionClusteringError("Figure DPI must be at least 72.")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": int(dpi),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    clustering_path = _clustering_paths(output_root)["manifest"]
    clustering_sha = sha256_file(clustering_path)
    receipt_path = output_root / "figures" / "figure_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(
            receipt_path, label="SO2 raw-expression figure manifest"
        )
        _verify_figure_receipt(
            output_root=output_root,
            receipt=receipt,
            clustering_manifest_sha256=clustering_sha,
        )
        return receipt
    figure_dir = receipt_path.parent
    if figure_dir.exists() and any(figure_dir.iterdir()):
        recoverable = {
            "raw_expression_leiden_resolution_1p0_spatial_14cores.png",
            "raw_expression_leiden_resolution_1p0_spatial_14cores.pdf",
            *(
                "per_core/"
                f"core_{core}_raw_expression_leiden_resolution_1p0.png"
                for core in SO2_CORE_NUMBERS
            ),
        }
        unexpected: list[str] = []
        for existing in figure_dir.rglob("*"):
            if not existing.is_file():
                continue
            relative = existing.relative_to(figure_dir).as_posix()
            if relative in recoverable:
                continue
            if ".tmp-" in existing.name:
                # Atomic writers can leave only their workflow-owned temporary
                # file after an abrupt process termination.  It is never a
                # valid deliverable and is safe to discard before retrying.
                existing.unlink(missing_ok=True)
                continue
            unexpected.append(relative)
        if unexpected:
            raise SO2RawExpressionClusteringError(
                "Unexpected partial figure outputs prevent a safe retry: "
                f"{sorted(unexpected)}"
            )
    frame = pd.read_parquet(_clustering_paths(output_root)["cell_table"])
    if len(frame) != EXPECTED_TOTAL_CELLS or tuple(
        frame["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise SO2RawExpressionClusteringError(
            "Figure source table lacks all 14 ordered cores."
        )
    palette_value = clustering_receipt.get("palette")
    if not isinstance(palette_value, Mapping):
        raise SO2RawExpressionClusteringError(
            "Clustering receipt lacks the expression palette."
        )
    palette = {str(key): str(value) for key, value in palette_value.items()}
    png_path, pdf_path, per_core_paths = _render_spatial_figures(
        frame,
        palette=palette,
        resolution=resolution,
        output_root=output_root,
        dpi=int(dpi),
    )
    figure_paths = (png_path, pdf_path, *per_core_paths)
    receipt = _receipt_with_self_hash(
        {
            "schema": FIGURE_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "clustering_manifest_sha256": clustering_sha,
            "leiden_resolution": float(resolution),
            "dpi": int(dpi),
            "one_dot_per_cell": True,
            "point_count": EXPECTED_TOTAL_CELLS,
            "point_layer_rasterized_in_pdf": True,
            "marker_borders": False,
            "lines_between_cells": False,
            "combined_figure_count": 2,
            "individual_core_figure_count": len(per_core_paths),
            "plot_specification": spatial_plot_spec(palette),
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in figure_paths
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_figure_receipt(
        output_root=output_root,
        receipt=receipt,
        clustering_manifest_sha256=clustering_sha,
    )
    return receipt


def _render_readme(
    *,
    output_root: Path,
    preprocessing: Mapping[str, Any],
    clustering: Mapping[str, Any],
) -> str:
    core_counts = "\n".join(
        f"- SO2 Core {number}: {EXPECTED_CELL_COUNTS_BY_CORE[number]:,} cells"
        for number in SO2_CORE_NUMBERS
    )
    dominated = clustering.get("core_dominated_gt_90pct", [])
    dominated_text = ", ".join(str(value) for value in dominated) or "None"
    low_count = int(
        preprocessing["loading_and_normalization"][
            "below_library_size_floor_cells"
        ]
    )
    pca = preprocessing["pca"]
    qc_summary = pd.read_csv(_clustering_paths(output_root)["qc_summary"])
    low_depth_row = qc_summary.sort_values(
        ["below_library_size_floor_count", "cluster_number"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    low_depth_cluster = str(low_depth_row["cluster"])
    low_depth_cluster_count = int(low_depth_row["below_library_size_floor_count"])
    low_depth_capture = low_depth_cluster_count / low_count
    low_depth_cluster_median = float(low_depth_row["raw_library_size_median"])
    return f"""# SO2 14-core classical raw-expression clustering

This report is an expression-only baseline for all {EXPECTED_TOTAL_CELLS:,}
cells from SO2 cores 15 through 28. It does not load or use a checkpoint, trained
model, h0/hL representation, graph, relative geometry, metadata, core identity,
or spatial coordinate during clustering. Coordinates are joined only after the
Leiden labels are frozen, solely for the maps.

## Method

The source is the checksum-verified nonnegative integer `expression_counts`
array for the fixed, ordered 1,000-gene biological CosMx panel. Technical probes
were excluded upstream. Each profile was normalized as

```text
counts * 197 / max(cell total, 20)
```

and transformed with `log1p`. The 20-count denominator floor follows CosMx 1K
guidance and prevents extreme up-scaling of very low-count cells. No cell was
filtered: {low_count:,} cells below 20 transcripts are retained and explicitly
flagged in the cell and cluster-QC tables.

All nonconstant panel genes were jointly mean-centered, scaled to unit sample
variance, and clipped at +/-10. Exact feature-covariance PCA retained
{int(pca['retained_components'])} components ({float(pca['total_explained_variance_ratio']):.2%}
of clipped scaled variance). L2-normalized PCA scores formed a deterministic
sparse cosine {int(clustering['configuration']['n_neighbors'])}-nearest-neighbor
graph. Seeded Leiden used resolution
{float(clustering['configuration']['leiden_resolution']):g} and seed
{int(clustering['configuration']['random_seed'])}. The expression-space graph is
not the model's spatial graph, and no dense cell-by-cell matrix was constructed.

The result has {int(clustering['cluster_count'])} joint expression clusters,
named `E0`, `E1`, ... by descending size. The size range is
{int(clustering['cluster_size_range'][0]):,} to
{int(clustering['cluster_size_range'][1]):,} cells. Clusters with more than 90%
of cells from one core: {dominated_text}.

`E3` and an hL label such as `C3` are unrelated identifiers; matching numbers do
not imply matching populations.

The prespecified depth audit found that `{low_depth_cluster}` contains
{low_depth_cluster_count:,} of the {low_count:,} cells below 20 transcripts
({low_depth_capture:.1%}), and its median library size is
{low_depth_cluster_median:g}. This is strong evidence that count depth contributes
to that partition; `{low_depth_cluster}` must not be treated as a cell type
without a QC sensitivity analysis and independent marker/pathology validation.

## Core coverage

{core_counts}

## Interpretation limits

This PCA + kNN + Leiden workflow is a conventional exploratory single-cell
clustering approach, adapted to targeted CosMx counts. The numerical resolution
1.0 is a prespecified comparison setting, not a universal cell-type resolution.
Clusters may reflect biology, count depth, technical effects, or targeted-panel
composition. They do not independently establish cell type, signaling,
biological influence, or causality. Marker-based and pathological validation
will be conducted separately; no biological cluster names are assigned here.

Method references:

- Bruker Spatial Biology, CosMx RNA QC and normalization:
  https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/normalization/
- Scanpy preprocessing and clustering:
  https://scanpy.readthedocs.io/en/latest/tutorials/basics/clustering.html
- Traag, Waltman & van Eck (2019), Leiden community detection:
  https://www.nature.com/articles/s41598-019-41695-z

## Reproduction

From the repository root, with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\
  analyze-so2-raw-expression-clusters \\
  --normalization-target 197 \\
  --library-size-floor 20 \\
  --scale-clip 10 \\
  --pca-components 50 \\
  --n-neighbors 30 \\
  --leiden-resolution 1.0 \\
  --random-seed 20260825 \\
  --device cpu
```

Preprocessing, clustering, and plotting are separate checksum-verified stages,
so a plotting retry does not repeat PCA or Leiden.
"""


def _verify_final_manifest(
    *,
    output_root: Path,
    manifest: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(manifest, label="SO2 raw-expression final manifest")
    if any(
        (
            manifest.get("schema") != ANALYSIS_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("analysis_id") != ANALYSIS_ID,
            manifest.get("configuration") != dict(configuration),
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(manifest.get("gene_count", -1)) != EXPECTED_N_GENES,
        )
    ):
        raise SO2RawExpressionClusteringError(
            "Raw-expression final manifest identity is invalid."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SO2RawExpressionClusteringError(
            "Raw-expression final manifest lacks file checksums."
        )
    _verify_stage_files(output_root, files, label="raw-expression final analysis")
    required = {
        "README.md",
        "preprocessing/preprocessing_manifest.json",
        "preprocessing/expression_pca_l2_normalized.npy",
        "preprocessing/expression_gene_statistics.npz",
        "preprocessing/expression_preprocessing_parameters.json",
        "preprocessing/gene_names.json",
        "preprocessing/cell_source_qc.parquet",
        "clustering/clustering_manifest.json",
        "clustering/expression_labels.npy",
        "clustering/expression_knn_undirected_edges.npy",
        "clustering/expression_clustering_parameters.json",
        "clustering/expression_palette.json",
        "tables/cell_expression_clusters.parquet",
        "tables/expression_cluster_summary.csv",
        "tables/expression_cluster_core_composition.csv",
        "tables/expression_cluster_qc_summary.csv",
        "figures/figure_manifest.json",
        "figures/raw_expression_leiden_resolution_1p0_spatial_14cores.png",
        "figures/raw_expression_leiden_resolution_1p0_spatial_14cores.pdf",
    }
    required.update(
        "figures/per_core/"
        f"core_{core}_raw_expression_leiden_resolution_1p0.png"
        for core in SO2_CORE_NUMBERS
    )
    if not required.issubset(files):
        raise SO2RawExpressionClusteringError(
            f"Required raw-expression outputs are absent: {sorted(required.difference(files))}"
        )
    if manifest.get("input_exclusion_audit") != {
        "checkpoint_used": False,
        "trained_model_used": False,
        "h0_used": False,
        "hL_used": False,
        "delta_h_used": False,
        "metadata_used": False,
        "spatial_graph_used": False,
        "relative_geometry_used": False,
        "coordinates_used_before_labels_frozen": False,
        "core_identity_used_before_labels_frozen": False,
        "clinical_or_vendor_labels_used": False,
    }:
        raise SO2RawExpressionClusteringError(
            "Raw-expression prohibited-input audit changed."
        )


def _finalize_analysis(
    *,
    inputs: RawExpressionInputs,
    output_root: Path,
    preprocessing: Mapping[str, Any],
    clustering: Mapping[str, Any],
    figures: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> Mapping[str, Any]:
    readme_path = output_root / "README.md"
    _atomic_write_text(
        readme_path,
        _render_readme(
            output_root=output_root,
            preprocessing=preprocessing,
            clustering=clustering,
        ),
    )
    manifest = _receipt_with_self_hash(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "analysis_id": ANALYSIS_ID,
            "analysis_scope": "classical_raw_expression_only_joint_clustering",
            "exploratory": True,
            "configuration": dict(configuration),
            "core_order": list(SO2_CORE_NUMBERS),
            "core_cell_counts": {
                str(core): EXPECTED_CELL_COUNTS_BY_CORE[core]
                for core in SO2_CORE_NUMBERS
            },
            "total_cells": EXPECTED_TOTAL_CELLS,
            "gene_count": EXPECTED_N_GENES,
            "source_artifacts": _source_file_records(inputs),
            "stage_manifests": {
                "preprocessing": _file_record(
                    _preprocessing_paths(output_root)["manifest"]
                ),
                "clustering": _file_record(
                    _clustering_paths(output_root)["manifest"]
                ),
                "figures": _file_record(
                    output_root / "figures" / "figure_manifest.json"
                ),
            },
            "cluster_count": int(clustering["cluster_count"]),
            "cluster_size_range": list(clustering["cluster_size_range"]),
            "core_dominated_gt_90pct": list(
                clustering["core_dominated_gt_90pct"]
            ),
            "below_library_size_floor_cells": int(
                preprocessing["loading_and_normalization"][
                    "below_library_size_floor_cells"
                ]
            ),
            "execution": {
                "device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_used": False,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "model_inference": False,
                "model_training": False,
            },
            "input_exclusion_audit": {
                "checkpoint_used": False,
                "trained_model_used": False,
                "h0_used": False,
                "hL_used": False,
                "delta_h_used": False,
                "metadata_used": False,
                "spatial_graph_used": False,
                "relative_geometry_used": False,
                "coordinates_used_before_labels_frozen": False,
                "core_identity_used_before_labels_frozen": False,
                "clinical_or_vendor_labels_used": False,
            },
            "interpretation": {
                "clusters_are_expression_derived": True,
                "cell_types_established": False,
                "signaling_established": False,
                "biological_influence_established": False,
                "causality_established": False,
                "marker_and_pathology_validation_separate": True,
            },
            "figure_paths": sorted(figures["files"]),
            "files": _file_manifest(output_root),
        }
    )
    manifest_path = output_root / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    _verify_final_manifest(
        output_root=output_root,
        manifest=manifest,
        configuration=configuration,
    )
    return manifest


def run_so2_raw_expression_clustering(
    *,
    paths: ProjectPaths,
    cohort_dir: str | Path | None = None,
    normalization_target: float = DEFAULT_NORMALIZATION_TARGET,
    library_size_floor: float = DEFAULT_LIBRARY_SIZE_FLOOR,
    scale_clip: float = DEFAULT_SCALE_CLIP,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
    device: str = "cpu",
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the locked, resumable CPU-only raw-expression analysis."""

    validate_cpu_device(device)
    configuration = _validate_locked_parameters(
        normalization_target=normalization_target,
        library_size_floor=library_size_floor,
        scale_clip=scale_clip,
        pca_components=pca_components,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_seed=random_seed,
    )
    if isinstance(dpi, bool) or int(dpi) < 72:
        raise SO2RawExpressionClusteringError("Figure DPI must be at least 72.")
    configuration = {**configuration, "figure_dpi": int(dpi)}
    inputs = resolve_raw_expression_inputs(paths=paths, cohort_dir=cohort_dir)
    if output_dir is None:
        output_root = (
            paths.report_root
            / "analyses"
            / "so2_14core_raw_expression_clustering"
            / ANALYSIS_ID
        )
    else:
        output_root = Path(output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = paths.project_root / output_root
        output_root = output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(
            manifest_path, label="SO2 raw-expression final manifest"
        )
        _verify_final_manifest(
            output_root=output_root,
            manifest=manifest,
            configuration=configuration,
        )
    else:
        preprocessing = prepare_expression_representation(
            inputs=inputs,
            output_root=output_root,
            configuration=configuration,
        )
        clustering = cluster_joint_raw_expression(
            inputs=inputs,
            output_root=output_root,
            preprocessing_receipt=preprocessing,
            configuration=configuration,
        )
        figures = render_expression_spatial_maps(
            output_root=output_root,
            clustering_receipt=clustering,
            resolution=leiden_resolution,
            dpi=int(dpi),
        )
        manifest = _finalize_analysis(
            inputs=inputs,
            output_root=output_root,
            preprocessing=preprocessing,
            clustering=clustering,
            figures=figures,
            configuration=configuration,
        )
    combined_png = (
        output_root
        / "figures"
        / "raw_expression_leiden_resolution_1p0_spatial_14cores.png"
    )
    combined_pdf = combined_png.with_suffix(".pdf")
    return {
        "status": "complete",
        "analysis_id": ANALYSIS_ID,
        "device": "cpu",
        "output_root": output_root.as_posix(),
        "total_cells": int(manifest["total_cells"]),
        "gene_count": int(manifest["gene_count"]),
        "cluster_count": int(manifest["cluster_count"]),
        "cluster_size_range": list(manifest["cluster_size_range"]),
        "core_dominated_gt_90pct": list(manifest["core_dominated_gt_90pct"]),
        "below_library_size_floor_cells": int(
            manifest["below_library_size_floor_cells"]
        ),
        "combined_png": combined_png.as_posix(),
        "combined_pdf": combined_pdf.as_posix(),
        "cell_table": _clustering_paths(output_root)["cell_table"].as_posix(),
        "manifest": manifest_path.as_posix(),
    }


__all__ = [
    "ExpressionPCAResult",
    "RawExpressionInputs",
    "SO2RawExpressionClusteringError",
    "cluster_expression_scores",
    "cluster_joint_raw_expression",
    "deterministic_expression_palette",
    "exact_scaled_sparse_pca",
    "log_normalize_counts",
    "prepare_expression_representation",
    "render_expression_spatial_maps",
    "requested_panel_order",
    "resolve_raw_expression_inputs",
    "run_so2_raw_expression_clustering",
    "spatial_plot_spec",
    "validate_cpu_device",
]
