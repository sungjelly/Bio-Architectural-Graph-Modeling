"""CPU-only direct-hL cosine-kNN/Leiden clustering for the SO2 cohort.

This analysis intentionally removes both dimensionality reduction and
mean-centering from the completed SO2 contextual-embedding workflow.  The only
row transformation is L2 normalization used to implement cosine distance; it
is distance arithmetic, not a learned representation or reduction step.

The input hL arrays are reused from the checksum-verified epoch-175 extraction
bundle.  The existing PCA analysis remains immutable and its PCA scores,
neighbor graph, and labels are never consumed by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hmac
import math
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .fingerprints import sha256_file
from .paths import ProjectPaths
from .registry import Registry
from .relative_qkv_embedding_clustering import (
    KNNGraphResult,
    LeidenResult,
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
    build_faiss_cosine_knn_graph,
    deterministic_glasbey_palette,
    run_seeded_leiden,
)
from .so2_hl_clustering import (
    CAMPAIGN_ID,
    EXPECTED_MODEL_SEED,
    EXPECTED_RUN_ID,
    SO2ContextualCore,
    SO2ResolvedInputs,
    _source_artifact_records,
    _verify_extraction_manifest,
    _verify_final_manifest as _verify_source_analysis_manifest,
    build_cell_index_frame,
    cluster_summary_tables,
    load_all_contextual_cores,
    resolve_so2_analysis_inputs,
    spatial_plot_spec,
    validate_cpu_device,
)
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_TOTAL_CELLS,
    SO2_CORE_NUMBERS,
)


ANALYSIS_SCHEMA = "so2_14core_contextual_embedding_direct_knn_clustering_v1"
CLUSTERING_SCHEMA = "so2_14core_joint_direct_hl_clustering_v1"
FIGURE_SCHEMA = "so2_14core_direct_hl_spatial_figure_v1"
PIPELINE_KIND = "direct_hl_cosine_knn_leiden"
SOURCE_ANALYSIS_NAMESPACE = "so2_14core_contextual_embedding_clustering"
OUTPUT_ANALYSIS_NAMESPACE = (
    "so2_14core_contextual_embedding_direct_knn_clustering"
)
EXPECTED_COMPLETED_EPOCHS = 175
DEFAULT_N_NEIGHBORS = 30
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_CPU_THREADS = 40
DEFAULT_RECALL_QUERY_COUNT = 128
DEFAULT_RECALL_QUERY_BATCH_SIZE = 8
MINIMUM_MEAN_RECALL_AT_30 = 0.90
DIRECT_LABEL_PREFIX = "D"
DIRECT_PALETTE_NAMESPACE = "contextual_direct_hl_no_pca"

TABLE_RELATIVE_PATH = Path("tables/cell_contextual_direct_clusters.parquet")
SUMMARY_RELATIVE_PATH = Path("tables/contextual_direct_cluster_summary.csv")
COMPOSITION_RELATIVE_PATH = Path(
    "tables/contextual_direct_cluster_core_composition.csv"
)
LABELS_RELATIVE_PATH = Path("clustering/contextual_direct_labels.npy")
EDGES_RELATIVE_PATH = Path(
    "clustering/contextual_direct_knn_undirected_edges.npy"
)
PARAMETERS_RELATIVE_PATH = Path(
    "clustering/contextual_direct_clustering_parameters.json"
)
PALETTE_RELATIVE_PATH = Path("clustering/contextual_direct_palette.json")
COMBINED_PNG_RELATIVE_PATH = Path(
    "figures/contextual_direct_hl_leiden_resolution_1p0_spatial_14cores.png"
)
COMBINED_PDF_RELATIVE_PATH = Path(
    "figures/contextual_direct_hl_leiden_resolution_1p0_spatial_14cores.pdf"
)


class SO2HLDirectClusteringError(ValueError):
    """Raised when the locked direct-hL analysis contract is violated."""


@dataclass(frozen=True, slots=True)
class VerifiedHLSource:
    """Checksum-verified source extraction from the completed PCA report.

    Only the extraction artifacts are used downstream.  The source report's
    PCA, kNN, Leiden, and figure outputs are verified as part of its immutable
    final manifest but are not loaded into this analysis.
    """

    inputs: SO2ResolvedInputs = field(repr=False)
    source_root: Path
    source_manifest: Mapping[str, Any] = field(repr=False)
    source_manifest_sha256: str
    extraction_receipt: Mapping[str, Any] = field(repr=False)
    extraction_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DirectCosineClusteringResult:
    labels: np.ndarray = field(repr=False)
    edge_pairs: np.ndarray = field(repr=False)
    receipt: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_cuda_hidden() -> str:
    """Require the production command to hide GPUs reserved for training."""

    observed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if observed not in {"", "-1"}:
        rendered = "unset" if observed is None else repr(observed)
        raise SO2HLDirectClusteringError(
            "Direct-hL analysis requires CUDA_VISIBLE_DEVICES='' (or '-1'); "
            f"observed {rendered}."
        )
    return str(observed)


def _source_root(paths: ProjectPaths, run_id: str) -> Path:
    return (
        paths.report_root
        / "analyses"
        / SOURCE_ANALYSIS_NAMESPACE
        / str(run_id)
    )


def _output_root(paths: ProjectPaths, run_id: str) -> Path:
    return (
        paths.report_root
        / "analyses"
        / OUTPUT_ANALYSIS_NAMESPACE
        / str(run_id)
    )


def _resolve_path(
    value: str | Path | None, *, default: Path, project_root: Path
) -> Path:
    if value is None:
        return default.resolve(strict=False)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=False)


def resolve_verified_hl_source(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None = None,
    checkpoint: str | Path | None = None,
    source_output_dir: str | Path | None = None,
) -> VerifiedHLSource:
    """Resolve epoch 175 and verify the existing per-core hL extraction."""

    inputs = resolve_so2_analysis_inputs(
        registry=registry,
        paths=paths,
        run_id=run_id,
        checkpoint=checkpoint,
    )
    completed_epochs = int(inputs.checkpoint_payload.get("completed_global_epochs", -1))
    if completed_epochs != EXPECTED_COMPLETED_EPOCHS:
        raise SO2HLDirectClusteringError(
            "Direct hL clustering is locked to the verified epoch-175 checkpoint."
        )
    source_root = _resolve_path(
        source_output_dir,
        default=_source_root(paths, inputs.run_id),
        project_root=paths.project_root,
    )
    manifest_path = source_root / "manifest.json"
    extraction_path = source_root / "embeddings" / "extraction_manifest.json"
    if not manifest_path.is_file() or not extraction_path.is_file():
        raise SO2HLDirectClusteringError(
            "The checksum-verified source hL extraction report is incomplete."
        )
    source_manifest = _read_json(
        manifest_path, label="source SO2 hL analysis manifest"
    )
    _verify_source_analysis_manifest(source_root, source_manifest)
    if any(
        (
            source_manifest.get("run_id") != inputs.run_id,
            int(source_manifest.get("checkpoint", {}).get(
                "completed_global_epochs", -1
            ))
            != EXPECTED_COMPLETED_EPOCHS,
            source_manifest.get("checkpoint", {}).get("sha256")
            != inputs.checkpoint_sha256,
            source_manifest.get("hL_shape")
            != [
                EXPECTED_TOTAL_CELLS,
                int(inputs.checkpoint_payload["model_construction"]["hidden_dim"]),
            ],
        )
    ):
        raise SO2HLDirectClusteringError(
            "Source hL report does not match the epoch-175 checkpoint contract."
        )
    extraction = _read_json(extraction_path, label="source hL extraction manifest")
    _verify_extraction_manifest(
        output_root=source_root,
        receipt=extraction,
        inputs=inputs,
    )
    return VerifiedHLSource(
        inputs=inputs,
        source_root=source_root,
        source_manifest=source_manifest,
        source_manifest_sha256=sha256_file(manifest_path),
        extraction_receipt=extraction,
        extraction_manifest_sha256=sha256_file(extraction_path),
    )


def concatenate_direct_hl(cores: Sequence[SO2ContextualCore]) -> np.ndarray:
    """Concatenate raw hL arrays without centering or projection."""

    if tuple(core.core_number for core in cores) != SO2_CORE_NUMBERS:
        raise SO2HLDirectClusteringError(
            "Direct hL clustering requires all 14 cores in locked order."
        )
    widths = {int(core.hL.shape[1]) for core in cores}
    if len(widths) != 1:
        raise SO2HLDirectClusteringError("hL widths differ across SO2 cores.")
    combined = np.ascontiguousarray(
        np.concatenate([core.hL for core in cores], axis=0), dtype=np.float32
    )
    if combined.shape[0] != EXPECTED_TOTAL_CELLS:
        raise SO2HLDirectClusteringError("Joint direct hL array dropped cells.")
    if not np.isfinite(combined).all():
        raise SO2HLDirectClusteringError("Joint direct hL contains non-finite values.")
    if not np.any(np.var(combined, axis=0, dtype=np.float64) > 0.0):
        raise SO2HLDirectClusteringError("Joint direct hL has zero variance.")
    return combined


def l2_normalize_hl_for_cosine(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Return unit rows solely for cosine inner-product distance evaluation.

    There is deliberately no centering, PCA, encoder, or other representation
    transformation here.  The input is checksum-checked before and after the
    operation to catch accidental in-place mutation.
    """

    values = np.asarray(embeddings)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 1
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise SO2HLDirectClusteringError(
            "Cosine input must be a finite floating [cells, hL dimensions] array."
        )
    before = _array_sha256("joint_contextual_hL", values)
    squared_norms = np.einsum(
        "ij,ij->i", values, values, dtype=np.float64, optimize=True
    )
    norms = np.sqrt(squared_norms, dtype=np.float64)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise SO2HLDirectClusteringError(
            "Direct hL contains a zero-norm or non-finite cell; cosine is undefined."
        )
    normalized = np.empty(values.shape, dtype=np.float32, order="C")
    np.divide(
        values,
        norms.astype(np.float32, copy=False)[:, None],
        out=normalized,
        casting="unsafe",
    )
    output_norms = np.sqrt(
        np.einsum(
            "ij,ij->i", normalized, normalized, dtype=np.float64, optimize=True
        ),
        dtype=np.float64,
    )
    if not np.allclose(output_norms, 1.0, rtol=1e-5, atol=1e-6):
        raise SO2HLDirectClusteringError("Cosine L2 normalization is inaccurate.")
    after = _array_sha256("joint_contextual_hL", values)
    if not hmac.compare_digest(before, after):
        raise SO2HLDirectClusteringError("Raw hL was mutated during cosine setup.")
    receipt = {
        "operation": "row_l2_normalization_for_cosine_distance_only",
        "representation_learning_or_reduction": False,
        "pca": False,
        "mean_center": False,
        "l2_normalize_for_cosine": True,
        "input_shape": list(values.shape),
        "output_shape": list(normalized.shape),
        "input_dtype": str(values.dtype),
        "output_dtype": str(normalized.dtype),
        "minimum_input_l2_norm": float(norms.min()),
        "maximum_input_l2_norm": float(norms.max()),
        "input_hL_sha256": before,
        "normalized_cosine_search_array_sha256": _array_sha256(
            "direct_hL_l2_for_cosine", normalized
        ),
        "raw_hL_mutated": False,
        "normalized_array_persisted": False,
        "cell_by_cell_matrix_constructed": False,
    }
    return normalized, receipt


def exact_neighbor_recall_audit(
    normalized_embeddings: np.ndarray,
    directed_neighbors: np.ndarray,
    *,
    n_neighbors: int,
    query_count: int = DEFAULT_RECALL_QUERY_COUNT,
    query_batch_size: int = DEFAULT_RECALL_QUERY_BATCH_SIZE,
) -> Mapping[str, Any]:
    """Audit HNSW recall with stable sampled exact cosine searches.

    Exact query-to-all similarities are allocated only as
    ``[query_batch_size, n_cells]``.  A dense ``[n_cells, n_cells]`` matrix is
    neither allocated nor persisted.
    """

    values = np.ascontiguousarray(normalized_embeddings, dtype=np.float32)
    neighbors = np.asarray(directed_neighbors, dtype=np.int64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SO2HLDirectClusteringError("Recall audit embeddings are invalid.")
    n_cells = int(values.shape[0])
    if neighbors.shape != (n_cells, int(n_neighbors)):
        raise SO2HLDirectClusteringError("Recall audit neighbor rows are invalid.")
    if (
        isinstance(query_count, bool)
        or int(query_count) <= 0
        or isinstance(query_batch_size, bool)
        or int(query_batch_size) <= 0
    ):
        raise SO2HLDirectClusteringError(
            "Recall query count and batch size must be positive."
        )
    sampled_count = min(int(query_count), n_cells)
    if sampled_count == n_cells:
        query_indices = np.arange(n_cells, dtype=np.int64)
    else:
        query_indices = np.rint(
            np.linspace(0, n_cells - 1, sampled_count, dtype=np.float64)
        ).astype(np.int64)
    if len(np.unique(query_indices)) != sampled_count:
        raise SO2HLDirectClusteringError("Recall query sampling is not unique.")

    exact_rows = np.empty((sampled_count, int(n_neighbors)), dtype=np.int64)
    recalls = np.empty(sampled_count, dtype=np.float64)
    maximum_batch_rows = min(int(query_batch_size), sampled_count)
    global_indices = np.arange(n_cells, dtype=np.int64)
    for start in range(0, sampled_count, int(query_batch_size)):
        stop = min(start + int(query_batch_size), sampled_count)
        batch_indices = query_indices[start:stop]
        pairwise = np.matmul(values[batch_indices], values.T)
        if pairwise.shape != (len(batch_indices), n_cells):
            raise SO2HLDirectClusteringError("Exact recall batch shape is invalid.")
        for local_row, query_index in enumerate(batch_indices):
            scores = pairwise[local_row]
            scores[int(query_index)] = -np.inf
            kth_offset = n_cells - int(n_neighbors)
            threshold = float(np.partition(scores, kth_offset)[kth_offset])
            above = np.flatnonzero(scores > threshold).astype(np.int64, copy=False)
            needed = int(n_neighbors) - len(above)
            tied = np.flatnonzero(scores == threshold).astype(np.int64, copy=False)
            if needed < 0 or len(tied) < needed:
                raise SO2HLDirectClusteringError(
                    "Exact recall tie resolution could not select k neighbors."
                )
            candidates = np.concatenate((above, tied[:needed]))
            order = np.lexsort((global_indices[candidates], -scores[candidates]))
            exact = candidates[order][: int(n_neighbors)]
            if len(exact) != int(n_neighbors) or len(np.unique(exact)) != len(exact):
                raise SO2HLDirectClusteringError(
                    "Exact recall search returned invalid neighbors."
                )
            exact_rows[start + local_row] = exact
            recalls[start + local_row] = (
                len(np.intersect1d(neighbors[int(query_index)], exact))
                / float(n_neighbors)
            )
        del pairwise
    return {
        "method": "deterministic_evenly_spaced_queries_exact_cosine",
        "purpose": "approximate_HNSW_neighbor_recall_quality_control",
        "query_count_requested": int(query_count),
        "query_count_evaluated": int(sampled_count),
        "query_batch_size": int(query_batch_size),
        "n_neighbors": int(n_neighbors),
        "recall_at_k_mean": float(recalls.mean()),
        "recall_at_k_median": float(np.median(recalls)),
        "recall_at_k_minimum": float(recalls.min()),
        "recall_at_k_maximum": float(recalls.max()),
        "query_indices_sha256": _array_sha256(
            "exact_recall_query_indices", query_indices
        ),
        "exact_neighbors_sha256": _array_sha256(
            "sampled_exact_cosine_neighbors", exact_rows
        ),
        "per_query_recall_sha256": _array_sha256(
            "sampled_exact_recall_at_k", recalls
        ),
        "maximum_explicit_pairwise_array_shape": [
            int(maximum_batch_rows),
            n_cells,
        ],
        "maximum_explicit_pairwise_array_elements": int(
            maximum_batch_rows * n_cells
        ),
        "cell_by_cell_matrix_constructed": False,
        "exact_similarity_matrix_persisted": False,
    }


def direct_cosine_knn_leiden(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> DirectCosineClusteringResult:
    """Run the exact direct-hL -> cosine-kNN -> Leiden computation."""

    raw_values = np.asarray(embeddings)
    raw_hash_before = _array_sha256("joint_contextual_hL", raw_values)
    normalized, normalization = l2_normalize_hl_for_cosine(raw_values)
    knn = build_faiss_cosine_knn_graph(
        normalized,
        n_neighbors=n_neighbors,
        random_seed=random_seed,
    )
    if knn.directed_neighbors is None:
        raise SO2HLDirectClusteringError(
            "FAISS result lacks directed rows required for exact-recall QC."
        )
    recall_audit = dict(exact_neighbor_recall_audit(
        normalized,
        knn.directed_neighbors,
        n_neighbors=n_neighbors,
    ))
    recall_audit["minimum_accepted_mean_recall_at_k"] = (
        MINIMUM_MEAN_RECALL_AT_30
    )
    recall_audit["mean_recall_acceptance_passed"] = bool(
        recall_audit["recall_at_k_mean"] >= MINIMUM_MEAN_RECALL_AT_30
    )
    if not recall_audit["mean_recall_acceptance_passed"]:
        raise SO2HLDirectClusteringError(
            "FAISS HNSW sampled exact mean recall@30 is below the predefined "
            f"{MINIMUM_MEAN_RECALL_AT_30:.2f} acceptance threshold."
        )
    leiden = run_seeded_leiden(
        knn,
        n_cells=len(normalized),
        resolution=leiden_resolution,
        random_seed=random_seed,
    )
    labels = np.ascontiguousarray(leiden.labels, dtype=np.int64)
    if labels.shape != (len(normalized),):
        raise SO2HLDirectClusteringError("Leiden labels do not align with direct hL.")
    raw_hash_after = _array_sha256("joint_contextual_hL", raw_values)
    if not hmac.compare_digest(raw_hash_before, raw_hash_after):
        raise SO2HLDirectClusteringError(
            "Raw hL changed during direct cosine kNN or Leiden clustering."
        )
    receipt = {
        "pipeline_kind": PIPELINE_KIND,
        "method": PIPELINE_KIND,
        "representation": "hL_final_graph_pre_decoder",
        "embedding_array": "hL",
        "joint_embedding_shape": list(np.asarray(embeddings).shape),
        "joint_embedding_sha256": normalization["input_hL_sha256"],
        "pca": False,
        "mean_center": False,
        "l2_normalize_for_cosine": True,
        "cosine_distance_setup": normalization,
        "knn": dict(knn.receipt),
        "knn_exact_recall_audit": recall_audit,
        "leiden": dict(leiden.receipt),
        "raw_hL_sha256_after_complete_pipeline": raw_hash_after,
        "raw_hL_mutated": False,
    }
    return DirectCosineClusteringResult(
        labels=labels,
        edge_pairs=np.ascontiguousarray(knn.edge_pairs, dtype=np.int64),
        receipt=receipt,
    )


def deterministic_direct_palette(cluster_count: int) -> dict[str, str]:
    """Return D-prefixed colors independent of the prior contextual C palette."""

    # The established intrinsic color sequence has a different anchor and
    # candidate rotation than the established contextual sequence.  Re-keying
    # it under the explicit direct-hL namespace keeps palette generation stable
    # while avoiding a false C-label/color correspondence with the PCA result.
    base = deterministic_glasbey_palette(cluster_count, namespace="intrinsic")
    result = {
        f"{DIRECT_LABEL_PREFIX}{index}": base[f"I{index}"]
        for index in range(int(cluster_count))
    }
    if len(result) != int(cluster_count) or len(set(result.values())) != len(result):
        raise SO2HLDirectClusteringError("Direct hL palette is invalid.")
    return result


def _clustering_configuration(
    *, n_neighbors: int, leiden_resolution: float, random_seed: int
) -> dict[str, Any]:
    if (
        isinstance(n_neighbors, bool)
        or int(n_neighbors) <= 0
        or int(n_neighbors) >= EXPECTED_TOTAL_CELLS
    ):
        raise SO2HLDirectClusteringError(
            "n_neighbors must be in [1, total cells - 1]."
        )
    if not math.isfinite(float(leiden_resolution)) or float(leiden_resolution) <= 0:
        raise SO2HLDirectClusteringError("Leiden resolution must be positive.")
    return {
        "pipeline_kind": PIPELINE_KIND,
        "method": PIPELINE_KIND,
        "representation": "hL_final_graph_pre_decoder",
        "joint_core_order": list(SO2_CORE_NUMBERS),
        "joint_cell_count": EXPECTED_TOTAL_CELLS,
        "embedding_dimension": 256,
        "pca": False,
        "mean_center": False,
        "l2_normalize_for_cosine": True,
        "l2_normalization_role": "cosine_distance_computation_only",
        "n_neighbors": int(n_neighbors),
        "distance_metric": "cosine",
        "knn_implementation": "faiss.IndexHNSWFlat",
        "knn_symmetrization": "undirected_union",
        "knn_exact_recall_audit_query_count": DEFAULT_RECALL_QUERY_COUNT,
        "knn_exact_recall_audit_query_batch_size": (
            DEFAULT_RECALL_QUERY_BATCH_SIZE
        ),
        "knn_exact_recall_audit_scope": "deterministic_sampled_query_vs_all",
        "knn_minimum_accepted_mean_recall_at_30": MINIMUM_MEAN_RECALL_AT_30,
        "leiden_resolution": float(leiden_resolution),
        "random_seed": int(random_seed),
        "label_prefix": DIRECT_LABEL_PREFIX,
        "palette_namespace": DIRECT_PALETTE_NAMESPACE,
        "cluster_sort": (
            "descending_size_then_minimum_global_cell_index_then_raw_id"
        ),
        "source_pca_scores_reused": False,
        "source_knn_graph_reused": False,
        "source_cluster_labels_reused": False,
        "spatial_training_graph_reused_for_clustering": False,
        "cross_core_embedding_neighbors_permitted": True,
        "dense_cell_by_cell_matrix_constructed": False,
        "device": "cpu",
    }


def _verify_clustering_manifest(
    *,
    output_root: Path,
    receipt: Mapping[str, Any],
    run_id: str,
    source_extraction_sha256: str,
    configuration: Mapping[str, Any],
) -> None:
    _verify_self_hash(receipt, label="SO2 direct-hL clustering manifest")
    if any(
        (
            receipt.get("schema") != CLUSTERING_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != run_id,
            receipt.get("source_extraction_manifest_sha256")
            != source_extraction_sha256,
            receipt.get("configuration") != dict(configuration),
            receipt.get("pipeline", {}).get("pipeline_kind") != PIPELINE_KIND,
            receipt.get("pipeline", {}).get("pca") is not False,
            receipt.get("pipeline", {}).get("mean_center") is not False,
            receipt.get("pipeline", {}).get("l2_normalize_for_cosine") is not True,
            receipt.get("pipeline", {}).get("knn_exact_recall_audit", {}).get(
                "cell_by_cell_matrix_constructed"
            )
            is not False,
            receipt.get("pipeline", {}).get("knn_exact_recall_audit", {}).get(
                "mean_recall_acceptance_passed"
            )
            is not True,
            tuple(receipt.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(receipt.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO2HLDirectClusteringError(
            "SO2 direct-hL clustering manifest identity is invalid."
        )
    files = receipt.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SO2HLDirectClusteringError("Direct-hL clustering files are absent.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HLDirectClusteringError(
                f"Direct-hL clustering output is missing: {relative}"
            )
        if _file_record(path) != dict(record):
            raise SO2HLDirectClusteringError(
                f"Direct-hL clustering checksum changed: {relative}"
            )
    labels = np.load(output_root / LABELS_RELATIVE_PATH, allow_pickle=False)
    if labels.shape != (EXPECTED_TOTAL_CELLS,) or _array_sha256(
        "sorted_leiden_labels", labels
    ) != receipt.get("pipeline", {}).get("leiden", {}).get("labels_sha256"):
        raise SO2HLDirectClusteringError("Stored direct-hL labels are invalid.")
    table = pd.read_parquet(output_root / TABLE_RELATIVE_PATH)
    if len(table) != EXPECTED_TOTAL_CELLS or tuple(
        table["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise SO2HLDirectClusteringError("Direct-hL cell table is incomplete.")
    if not np.array_equal(
        table["contextual_direct_cluster_number"].to_numpy(dtype=np.int64), labels
    ):
        raise SO2HLDirectClusteringError("Direct-hL table and labels are misaligned.")


def cluster_joint_direct_hl(
    *,
    source: VerifiedHLSource,
    output_root: Path,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Mapping[str, Any]:
    """Run or verify joint direct-hL clustering from source embedding files."""

    configuration = _clustering_configuration(
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_seed=random_seed,
    )
    receipt_path = output_root / "clustering" / "clustering_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="SO2 direct-hL clustering manifest")
        _verify_clustering_manifest(
            output_root=output_root,
            receipt=receipt,
            run_id=source.inputs.run_id,
            source_extraction_sha256=source.extraction_manifest_sha256,
            configuration=configuration,
        )
        return receipt
    clustering_dir = output_root / "clustering"
    tables_dir = output_root / "tables"
    if (clustering_dir.exists() and any(clustering_dir.iterdir())) or (
        tables_dir.exists() and any(tables_dir.iterdir())
    ):
        raise SO2HLDirectClusteringError(
            "Partial direct-hL clustering outputs exist without a complete receipt."
        )
    clustering_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cores = load_all_contextual_cores(
        output_root=source.source_root,
        extraction_receipt=source.extraction_receipt,
    )
    cell_frame = build_cell_index_frame(cores)
    combined = concatenate_direct_hl(cores)
    result = direct_cosine_knn_leiden(
        combined,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_seed=random_seed,
    )
    labels = result.labels
    cell_frame["contextual_direct_cluster_number"] = labels.astype(np.int32)
    cell_frame["contextual_direct_cluster"] = [
        f"{DIRECT_LABEL_PREFIX}{int(value)}" for value in labels
    ]
    summary, composition, dominated = cluster_summary_tables(
        labels, cell_frame, prefix=DIRECT_LABEL_PREFIX
    )
    palette = deterministic_direct_palette(len(summary))

    _atomic_write_npy(output_root / LABELS_RELATIVE_PATH, labels)
    _atomic_write_npy(output_root / EDGES_RELATIVE_PATH, result.edge_pairs)
    _atomic_write_json(output_root / PARAMETERS_RELATIVE_PATH, result.receipt)
    _atomic_write_json(
        output_root / PALETTE_RELATIVE_PATH,
        {
            "representation": "contextual_direct_hL",
            "pipeline_kind": PIPELINE_KIND,
            "label_prefix": DIRECT_LABEL_PREFIX,
            "palette_namespace": DIRECT_PALETTE_NAMESPACE,
            "semantic_palette_namespace": DIRECT_PALETTE_NAMESPACE,
            "base_generator_namespace": "intrinsic_rekeyed_I_to_D",
            "method": "deterministic_greedy_farthest_point_CIELAB",
            "independent_from_prior_contextual_palette": True,
            "colors": palette,
        },
    )
    table_columns = [
        "global_cell_index",
        "cell_index",
        "cell_key",
        "core_alias",
        "core_number",
        "x_um",
        "y_um",
        "contextual_direct_cluster_number",
        "contextual_direct_cluster",
    ]
    _atomic_write_parquet(
        output_root / TABLE_RELATIVE_PATH,
        cell_frame.loc[:, table_columns],
    )
    _atomic_write_csv(output_root / SUMMARY_RELATIVE_PATH, summary)
    _atomic_write_csv(output_root / COMPOSITION_RELATIVE_PATH, composition)
    stage_paths = (
        output_root / LABELS_RELATIVE_PATH,
        output_root / EDGES_RELATIVE_PATH,
        output_root / PARAMETERS_RELATIVE_PATH,
        output_root / PALETTE_RELATIVE_PATH,
        output_root / TABLE_RELATIVE_PATH,
        output_root / SUMMARY_RELATIVE_PATH,
        output_root / COMPOSITION_RELATIVE_PATH,
    )
    receipt = _receipt_with_self_hash(
        {
            "schema": CLUSTERING_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "run_id": source.inputs.run_id,
            "source_analysis_manifest_sha256": source.source_manifest_sha256,
            "source_extraction_manifest_sha256": (
                source.extraction_manifest_sha256
            ),
            "configuration": configuration,
            "core_order": list(SO2_CORE_NUMBERS),
            "total_cells": EXPECTED_TOTAL_CELLS,
            "cluster_count": int(len(summary)),
            "cluster_size_range": [
                int(summary["size"].min()),
                int(summary["size"].max()),
            ],
            "core_dominated_gt_90pct": dominated,
            "pipeline": dict(result.receipt),
            "palette": palette,
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in stage_paths
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_clustering_manifest(
        output_root=output_root,
        receipt=receipt,
        run_id=source.inputs.run_id,
        source_extraction_sha256=source.extraction_manifest_sha256,
        configuration=configuration,
    )
    return receipt


def _cluster_legend_handles(palette: Mapping[str, str]) -> list[Any]:
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in sorted(
            palette.items(), key=lambda item: int(item[0][1:])
        )
    ]


def _render_combined_map(
    frame: pd.DataFrame,
    *,
    palette: Mapping[str, str],
    resolution: float,
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 5, figsize=(25.0, 15.0))
    tissue_axes = axes.ravel()[: len(SO2_CORE_NUMBERS)]
    for axis, core_number in zip(tissue_axes, SO2_CORE_NUMBERS, strict=True):
        selected = frame.loc[frame["core_number"] == core_number]
        coordinates = selected[["x_um", "y_um"]].to_numpy(dtype=np.float64)
        colors = selected["contextual_direct_cluster"].map(palette)
        if len(selected) != EXPECTED_CELL_COUNTS_BY_CORE[core_number]:
            raise SO2HLDirectClusteringError(
                f"Direct-hL spatial panel is incomplete for core {core_number}."
            )
        if colors.isna().any():
            raise SO2HLDirectClusteringError("Direct-hL map palette is incomplete.")
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
        ncol=2 if len(handles) > 14 else 1,
        title="Joint direct-hL\nmodel-derived cluster",
        fontsize=8,
        title_fontsize=10,
        borderaxespad=0.0,
    )
    figure.suptitle(
        "SO2 direct contextual hL Leiden clusters — no PCA or centering\n"
        f"cosine kNN; resolution {resolution:g}",
        fontsize=18,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "Model-derived clusters; no cell-type or biological annotation is implied.",
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
        figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi
    )
    plt.close(figure)


def _verify_figure_manifest(
    *, output_root: Path, receipt: Mapping[str, Any], clustering_sha256: str
) -> None:
    _verify_self_hash(receipt, label="SO2 direct-hL figure manifest")
    if any(
        (
            receipt.get("schema") != FIGURE_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("pipeline_kind") != PIPELINE_KIND,
            receipt.get("clustering_manifest_sha256") != clustering_sha256,
            tuple(receipt.get("plot_specification", {}).get("panel_order", ()))
            != SO2_CORE_NUMBERS,
            receipt.get("plot_specification", {}).get("grid_shape") != [3, 5],
        )
    ):
        raise SO2HLDirectClusteringError("Direct-hL figure manifest is invalid.")
    files = receipt.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        COMBINED_PNG_RELATIVE_PATH.as_posix(),
        COMBINED_PDF_RELATIVE_PATH.as_posix(),
    }:
        raise SO2HLDirectClusteringError("Direct-hL figure pair is incomplete.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HLDirectClusteringError(f"Figure is missing: {relative}")
        if _file_record(path) != dict(record):
            raise SO2HLDirectClusteringError(f"Figure checksum changed: {relative}")


def render_direct_spatial_map(
    *,
    output_root: Path,
    clustering_receipt: Mapping[str, Any],
    resolution: float,
    dpi: int = 300,
) -> Mapping[str, Any]:
    """Render or verify the combined 14-core direct-hL spatial map."""

    if isinstance(dpi, bool) or int(dpi) < 72:
        raise SO2HLDirectClusteringError("Figure DPI must be at least 72.")
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
    clustering_path = output_root / "clustering" / "clustering_manifest.json"
    clustering_sha = sha256_file(clustering_path)
    receipt_path = output_root / "figures" / "figure_manifest.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path, label="SO2 direct-hL figure manifest")
        _verify_figure_manifest(
            output_root=output_root,
            receipt=receipt,
            clustering_sha256=clustering_sha,
        )
        return receipt
    frame = pd.read_parquet(output_root / TABLE_RELATIVE_PATH)
    if len(frame) != EXPECTED_TOTAL_CELLS or tuple(
        frame["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise SO2HLDirectClusteringError("Figure table lacks all ordered SO2 cores.")
    palette = clustering_receipt.get("palette")
    if not isinstance(palette, Mapping):
        raise SO2HLDirectClusteringError("Clustering receipt lacks a direct palette.")
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    png_path = output_root / COMBINED_PNG_RELATIVE_PATH
    pdf_path = output_root / COMBINED_PDF_RELATIVE_PATH
    _render_combined_map(
        frame,
        palette=dict(palette),
        resolution=resolution,
        png_path=png_path,
        pdf_path=pdf_path,
        dpi=int(dpi),
    )
    specification = spatial_plot_spec(dict(palette))
    receipt = _receipt_with_self_hash(
        {
            "schema": FIGURE_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "pipeline_kind": PIPELINE_KIND,
            "clustering_manifest_sha256": clustering_sha,
            "leiden_resolution": float(resolution),
            "dpi": int(dpi),
            "one_dot_per_cell": True,
            "point_count": EXPECTED_TOTAL_CELLS,
            "point_layer_rasterized_in_pdf": True,
            "marker_borders": False,
            "lines_between_cells": False,
            "plot_specification": specification,
            "files": {
                path.relative_to(output_root).as_posix(): _file_record(path)
                for path in (png_path, pdf_path)
            },
        }
    )
    _atomic_write_json(receipt_path, receipt)
    _verify_figure_manifest(
        output_root=output_root,
        receipt=receipt,
        clustering_sha256=clustering_sha,
    )
    return receipt


def _source_embedding_records(source: VerifiedHLSource) -> dict[str, Any]:
    records: dict[str, Any] = {}
    cores = source.extraction_receipt.get("cores")
    if not isinstance(cores, list):
        raise SO2HLDirectClusteringError("Source extraction lacks core records.")
    for core in cores:
        if not isinstance(core, Mapping):
            raise SO2HLDirectClusteringError("Source extraction core is malformed.")
        path = source.source_root / str(core["embedding_file"])
        records[str(core["alias"])] = {
            "core_number": int(core["core_number"]),
            "cell_count": int(core["cell_count"]),
            "path": path.as_posix(),
            "file": _file_record(path),
            "hL_array_sha256": str(core["hL_array_sha256"]),
        }
    if tuple(value["core_number"] for value in records.values()) != SO2_CORE_NUMBERS:
        raise SO2HLDirectClusteringError("Source embedding records are incomplete.")
    return records


def _render_readme(
    *, source: VerifiedHLSource, clustering: Mapping[str, Any]
) -> str:
    counts = "\n".join(
        f"- SO2 Core {number}: {EXPECTED_CELL_COUNTS_BY_CORE[number]:,} cells"
        for number in SO2_CORE_NUMBERS
    )
    dominated = clustering.get("core_dominated_gt_90pct", [])
    dominated_text = ", ".join(str(value) for value in dominated) or "None"
    cluster_size_min, cluster_size_max = clustering["cluster_size_range"]
    recall = clustering["pipeline"]["knn_exact_recall_audit"]
    command = (
        'CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python '
        "-m spatial_benchmark analyze-so2-hl-direct-clusters "
        f"--run-id {source.inputs.run_id} --n-neighbors 30 "
        "--leiden-resolution 1.0 --random-seed 20260825 "
        "--device cpu --cpu-threads 40"
    )
    return f"""# SO2 14-core direct contextual-hL clustering

This CPU-only exploratory analysis reuses the checksum-verified epoch-175 hL
embeddings from `{source.inputs.run_id}`. It performs no model inference or
training and uses no GPU. The existing PCA-based analysis remains unchanged.

## Exact clustering method

All {EXPECTED_TOTAL_CELLS:,} raw 256-dimensional hL rows were concatenated in
locked core/cell order and clustered jointly. There was **no PCA**, **no
mean-centering**, no new encoder, and no batch correction. Each raw hL row was
L2-normalized only to calculate cosine similarity, then a sparse FAISS cosine
30-nearest-neighbor graph was constructed and Leiden was run at resolution 1.0
with seed 20260825. No dense cell-by-cell matrix was constructed. D-prefixed
labels and a separate palette distinguish this partition from the prior
PCA-based C clusters.

This direct-hL pipeline is a requested sensitivity analysis, not evidence that
removing PCA is generally preferable. Direct cosine geometry retains all 256
model dimensions, including any noisy, redundant, or anisotropic directions.

## Results

- Checkpoint: `{source.inputs.checkpoint_path}`
- Checkpoint SHA-256: `{source.inputs.checkpoint_sha256}`
- Completed epochs / model seed: {EXPECTED_COMPLETED_EPOCHS} / {EXPECTED_MODEL_SEED}
- hL shape: `[{EXPECTED_TOTAL_CELLS}, 256]`
- Joint direct-hL cluster count: `{clustering['cluster_count']}`
- Cluster-size range: `{cluster_size_min:,}` to `{cluster_size_max:,}` cells
- Clusters with >90% of cells from one core: {dominated_text}
- Sampled exact-neighbor recall@30 for FAISS HNSW (128 deterministic
  queries): mean `{recall['recall_at_k_mean']:.6f}`, median
  `{recall['recall_at_k_median']:.6f}`, minimum
  `{recall['recall_at_k_minimum']:.6f}`; predefined mean-recall threshold
  `{MINIMUM_MEAN_RECALL_AT_30:.2f}` passed

## Core coverage

{counts}

## Interpretation limits

These are model-derived contextual clusters, not established cell types,
signaling states, biological influence, or causal effects. Marker-based and
pathological validation remain separate. The cohort is fit-only/transductive,
and one trained seed does not establish representation stability.

## Reproduction

Run from the repository root:

```bash
{command}
```

The workflow is resumable after checksum-verified clustering, so plotting can
be retried without rebuilding the direct neighbor graph.
"""


def _verify_external_source_records(manifest: Mapping[str, Any]) -> None:
    inputs = manifest.get("input_artifacts")
    if not isinstance(inputs, Mapping):
        raise SO2HLDirectClusteringError("Direct-hL manifest lacks input artifacts.")
    for name in ("source_analysis_manifest", "source_extraction_manifest"):
        record = inputs.get(name)
        if not isinstance(record, Mapping):
            raise SO2HLDirectClusteringError(f"Missing input record: {name}")
        path = Path(str(record.get("path", "")))
        expected = record.get("file")
        if not path.is_file() or not isinstance(expected, Mapping):
            raise SO2HLDirectClusteringError(f"Input artifact is missing: {name}")
        if _file_record(path) != dict(expected):
            raise SO2HLDirectClusteringError(f"Input checksum changed: {name}")
    embeddings = inputs.get("source_hL_embeddings")
    if not isinstance(embeddings, Mapping) or len(embeddings) != len(SO2_CORE_NUMBERS):
        raise SO2HLDirectClusteringError("Source hL embedding records are incomplete.")
    observed_cores: list[int] = []
    for alias, record in embeddings.items():
        if not isinstance(record, Mapping):
            raise SO2HLDirectClusteringError(f"Malformed hL record: {alias}")
        observed_cores.append(int(record.get("core_number", -1)))
        path = Path(str(record.get("path", "")))
        expected = record.get("file")
        if not path.is_file() or not isinstance(expected, Mapping):
            raise SO2HLDirectClusteringError(f"Source hL is missing: {alias}")
        if _file_record(path) != dict(expected):
            raise SO2HLDirectClusteringError(f"Source hL checksum changed: {alias}")
    if tuple(observed_cores) != SO2_CORE_NUMBERS:
        raise SO2HLDirectClusteringError("Source hL core order changed.")


def verify_direct_hl_analysis(
    output_root: Path, manifest: Mapping[str, Any]
) -> None:
    """Fail-closed verification used by the CLI and interactive viewer."""

    _verify_self_hash(manifest, label="SO2 direct-hL final analysis manifest")
    if any(
        (
            manifest.get("schema") != ANALYSIS_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("run_id") != EXPECTED_RUN_ID,
            manifest.get("pipeline_kind") != PIPELINE_KIND,
            manifest.get("method") != PIPELINE_KIND,
            manifest.get("pca") is not False,
            manifest.get("mean_center") is not False,
            manifest.get("l2_normalize_for_cosine") is not True,
            float(manifest.get("leiden_resolution", math.nan))
            != DEFAULT_LEIDEN_RESOLUTION,
            int(manifest.get("n_neighbors", -1)) != DEFAULT_N_NEIGHBORS,
            int(manifest.get("random_seed", -1)) != DEFAULT_RANDOM_SEED,
            int(manifest.get("embedding_dimension", -1)) != 256,
            manifest.get("knn_exact_recall_audit", {}).get(
                "cell_by_cell_matrix_constructed"
            )
            is not False,
            manifest.get("knn_exact_recall_audit", {}).get(
                "mean_recall_acceptance_passed"
            )
            is not True,
            int(manifest.get("checkpoint", {}).get(
                "completed_global_epochs", -1
            ))
            != EXPECTED_COMPLETED_EPOCHS,
            manifest.get("cpu_only_execution", {}).get("cuda_visible_devices")
            not in {"", "-1"},
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO2HLDirectClusteringError("Direct-hL final manifest is invalid.")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise SO2HLDirectClusteringError("Direct-hL final file manifest is absent.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HLDirectClusteringError(f"Final output is missing: {relative}")
        if _file_record(path) != dict(record):
            raise SO2HLDirectClusteringError(
                f"Final output checksum changed: {relative}"
            )
    required = {
        "README.md",
        "clustering/clustering_manifest.json",
        LABELS_RELATIVE_PATH.as_posix(),
        EDGES_RELATIVE_PATH.as_posix(),
        PARAMETERS_RELATIVE_PATH.as_posix(),
        PALETTE_RELATIVE_PATH.as_posix(),
        TABLE_RELATIVE_PATH.as_posix(),
        SUMMARY_RELATIVE_PATH.as_posix(),
        COMPOSITION_RELATIVE_PATH.as_posix(),
        "figures/figure_manifest.json",
        COMBINED_PNG_RELATIVE_PATH.as_posix(),
        COMBINED_PDF_RELATIVE_PATH.as_posix(),
    }
    if not required.issubset(files):
        missing = sorted(required.difference(files))
        raise SO2HLDirectClusteringError(
            f"Required direct-hL outputs are absent: {missing}"
        )
    _verify_external_source_records(manifest)
    table = pd.read_parquet(output_root / TABLE_RELATIVE_PATH)
    if len(table) != EXPECTED_TOTAL_CELLS or tuple(
        table["core_number"].drop_duplicates().tolist()
    ) != SO2_CORE_NUMBERS:
        raise SO2HLDirectClusteringError("Final direct-hL cell table is incomplete.")


def load_verified_direct_hl_analysis(output_root: Path) -> Mapping[str, Any]:
    """Load and fully verify a completed direct-hL analysis manifest."""

    root = Path(output_root).expanduser().resolve(strict=True)
    manifest = _read_json(root / "manifest.json", label="direct-hL final manifest")
    verify_direct_hl_analysis(root, manifest)
    return manifest


def _finalize_analysis(
    *,
    source: VerifiedHLSource,
    output_root: Path,
    clustering: Mapping[str, Any],
    figures: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    _atomic_write_text(
        output_root / "README.md",
        _render_readme(source=source, clustering=clustering),
    )
    source_manifest_path = source.source_root / "manifest.json"
    source_extraction_path = (
        source.source_root / "embeddings" / "extraction_manifest.json"
    )
    input_artifacts = {
        "checkpoint_and_prepared_inputs": _source_artifact_records(source.inputs),
        "source_analysis_manifest": {
            "path": source_manifest_path.as_posix(),
            "file": _file_record(source_manifest_path),
        },
        "source_extraction_manifest": {
            "path": source_extraction_path.as_posix(),
            "file": _file_record(source_extraction_path),
        },
        "source_hL_embeddings": _source_embedding_records(source),
    }
    manifest = _receipt_with_self_hash(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": "complete",
            "created_at": _utc_now(),
            "analysis_scope": "contextual_hL_direct_cosine_only",
            "pipeline_kind": PIPELINE_KIND,
            "method": PIPELINE_KIND,
            "pca": False,
            "mean_center": False,
            "l2_normalize_for_cosine": True,
            "leiden_resolution": float(parameters["leiden_resolution"]),
            "n_neighbors": int(parameters["n_neighbors"]),
            "random_seed": int(parameters["random_seed"]),
            "embedding_dimension": 256,
            "run_id": source.inputs.run_id,
            "campaign_id": CAMPAIGN_ID,
            "model_seed": EXPECTED_MODEL_SEED,
            "checkpoint": source.inputs.provenance["checkpoint"],
            "model_construction": source.inputs.provenance["model_construction"],
            "core_order": list(SO2_CORE_NUMBERS),
            "core_cell_counts": {
                str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                for number in SO2_CORE_NUMBERS
            },
            "total_cells": EXPECTED_TOTAL_CELLS,
            "hL_shape": [EXPECTED_TOTAL_CELLS, 256],
            "analysis_parameters": dict(parameters),
            "cpu_only_execution": {
                "enforced_device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cpu_threads": int(parameters["cpu_threads"]),
                "gpu_tensor_or_model_created": False,
                "model_inference_performed": False,
            },
            "input_artifacts": input_artifacts,
            "stage_manifests": {
                "clustering": _file_record(
                    output_root / "clustering" / "clustering_manifest.json"
                ),
                "figures": _file_record(
                    output_root / "figures" / "figure_manifest.json"
                ),
            },
            "cluster_count": int(clustering["cluster_count"]),
            "cluster_size_range": clustering["cluster_size_range"],
            "core_dominated_gt_90pct": clustering["core_dominated_gt_90pct"],
            "knn_exact_recall_audit": clustering["pipeline"][
                "knn_exact_recall_audit"
            ],
            "figure_paths": sorted(figures["files"]),
            "interpretation": {
                "clusters_are_model_derived": True,
                "cell_types_established": False,
                "biological_influence_established": False,
                "causality_established": False,
                "marker_and_pathology_validation_separate": True,
            },
            "files": _file_manifest(output_root),
        }
    )
    _atomic_write_json(output_root / "manifest.json", manifest)
    verify_direct_hl_analysis(output_root, manifest)
    return manifest


def run_so2_hl_direct_clustering(
    *,
    registry: Registry,
    paths: ProjectPaths,
    run_id: str | None = None,
    checkpoint: str | Path | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = DEFAULT_RANDOM_SEED,
    device: str | torch.device = "cpu",
    cpu_threads: int = DEFAULT_CPU_THREADS,
    dpi: int = 300,
    source_output_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute the resumable, CPU-only direct-hL sensitivity analysis."""

    validate_cpu_device(device)
    validate_cuda_hidden()
    if not math.isclose(
        float(leiden_resolution), DEFAULT_LEIDEN_RESOLUTION, rel_tol=0.0, abs_tol=0.0
    ):
        raise SO2HLDirectClusteringError(
            "The requested direct-hL map is locked to Leiden resolution 1.0."
        )
    if int(n_neighbors) != DEFAULT_N_NEIGHBORS:
        raise SO2HLDirectClusteringError(
            "The requested direct-hL map is locked to 30 cosine neighbors."
        )
    if int(random_seed) != DEFAULT_RANDOM_SEED:
        raise SO2HLDirectClusteringError(
            "The requested direct-hL map is locked to random seed 20260825."
        )
    if isinstance(cpu_threads, bool) or int(cpu_threads) <= 0:
        raise SO2HLDirectClusteringError("cpu_threads must be positive.")
    if isinstance(dpi, bool) or int(dpi) < 72:
        raise SO2HLDirectClusteringError("Figure DPI must be at least 72.")
    torch.set_num_threads(int(cpu_threads))
    source = resolve_verified_hl_source(
        registry=registry,
        paths=paths,
        run_id=run_id,
        checkpoint=checkpoint,
        source_output_dir=source_output_dir,
    )
    output_root = _resolve_path(
        output_dir,
        default=_output_root(paths, source.inputs.run_id),
        project_root=paths.project_root,
    )
    if output_root == source.source_root:
        raise SO2HLDirectClusteringError(
            "Direct-hL outputs must not overwrite the PCA-based source report."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    parameters = _clustering_configuration(
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        random_seed=random_seed,
    )
    parameters.update({"cpu_threads": int(cpu_threads), "figure_dpi": int(dpi)})
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = load_verified_direct_hl_analysis(output_root)
        if manifest.get("analysis_parameters") != parameters:
            raise SO2HLDirectClusteringError(
                "Existing direct-hL analysis used different parameters."
            )
    else:
        clustering = cluster_joint_direct_hl(
            source=source,
            output_root=output_root,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
            random_seed=random_seed,
        )
        figures = render_direct_spatial_map(
            output_root=output_root,
            clustering_receipt=clustering,
            resolution=leiden_resolution,
            dpi=dpi,
        )
        manifest = _finalize_analysis(
            source=source,
            output_root=output_root,
            clustering=clustering,
            figures=figures,
            parameters=parameters,
        )
    return {
        "status": "complete",
        "run_id": source.inputs.run_id,
        "device": "cpu",
        "pipeline_kind": PIPELINE_KIND,
        "pca": False,
        "mean_center": False,
        "l2_normalize_for_cosine": True,
        "output_root": output_root.as_posix(),
        "total_cells": int(manifest["total_cells"]),
        "hL_shape": manifest["hL_shape"],
        "cluster_count": int(manifest["cluster_count"]),
        "cluster_size_range": manifest["cluster_size_range"],
        "core_dominated_gt_90pct": manifest["core_dominated_gt_90pct"],
        "knn_recall_at_30": manifest["knn_exact_recall_audit"][
            "recall_at_k_mean"
        ],
        "combined_png": (output_root / COMBINED_PNG_RELATIVE_PATH).as_posix(),
        "combined_pdf": (output_root / COMBINED_PDF_RELATIVE_PATH).as_posix(),
        "manifest": manifest_path.as_posix(),
    }


__all__ = [
    "ANALYSIS_SCHEMA",
    "COMBINED_PDF_RELATIVE_PATH",
    "COMBINED_PNG_RELATIVE_PATH",
    "DIRECT_LABEL_PREFIX",
    "OUTPUT_ANALYSIS_NAMESPACE",
    "PALETTE_RELATIVE_PATH",
    "PIPELINE_KIND",
    "SOURCE_ANALYSIS_NAMESPACE",
    "SO2HLDirectClusteringError",
    "TABLE_RELATIVE_PATH",
    "VerifiedHLSource",
    "cluster_joint_direct_hl",
    "concatenate_direct_hl",
    "deterministic_direct_palette",
    "direct_cosine_knn_leiden",
    "l2_normalize_hl_for_cosine",
    "load_verified_direct_hl_analysis",
    "render_direct_spatial_map",
    "resolve_verified_hl_source",
    "run_so2_hl_direct_clustering",
    "validate_cuda_hidden",
    "verify_direct_hl_analysis",
]
