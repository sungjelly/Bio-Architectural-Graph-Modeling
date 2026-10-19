"""Deterministic CPU-only UMAP of the locked SO2 contextual hL clusters.

This module is a visualization extension of the completed joint Leiden report.
It reconstructs the exact directed cosine kNN inputs from the persisted PCA
scores, verifies them against the source receipt, and overlays the immutable
resolution-1.0 cluster labels.  UMAP never defines or changes a cluster here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hmac
import math
import os
from pathlib import Path
import platform
import random
import re
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .paths import ProjectPaths
from .relative_qkv_embedding_clustering import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    _array_sha256,
    _atomic_save_figure_pair,
    _atomic_write_json,
    _atomic_write_npy,
    _atomic_write_parquet,
    _atomic_write_text,
    _file_manifest,
    _file_record,
    _package_version,
    _read_json,
    _receipt_with_self_hash,
    _verify_self_hash,
)
from .so2_hl_clustering import (
    ANALYSIS_SCHEMA as SOURCE_ANALYSIS_SCHEMA,
    DEFAULT_LEIDEN_RESOLUTION,
    EXPECTED_RUN_ID,
    SO2HLClusteringError,
    _verify_final_manifest as _verify_source_manifest,
)
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_TOTAL_CELLS,
    SO2_CORE_NUMBERS,
)


UMAP_SCHEMA = "so2_14core_contextual_hl_umap_v1"
UMAP_RECEIPT_SCHEMA = "so2_14core_contextual_hl_umap_receipt_v1"
DEFAULT_SOURCE_REPORT = "so2_14core_contextual_embedding_clustering"
DEFAULT_OUTPUT_REPORT = "so2_14core_contextual_embedding_umap"
DEFAULT_N_NEIGHBORS = 30
DEFAULT_MIN_DIST = 0.3
DEFAULT_EPOCHS = 200
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_DPI = 300
EXPECTED_CLUSTER_COUNT = 19
EXPECTED_CLUSTER_LABELS = tuple(
    f"C{index}" for index in range(EXPECTED_CLUSTER_COUNT)
)
COORDINATE_FILENAME = "contextual_hl_umap_coordinates.npy"
TABLE_FILENAME = "cell_contextual_hl_umap.parquet"
FIGURE_STEM = "contextual_leiden_resolution_1p0_umap"


class SO2HLUMAPError(SO2HLClusteringError):
    """Raised when the locked SO2 contextual-UMAP contract is violated."""


@dataclass(frozen=True, slots=True)
class UMAPSource:
    root: Path
    manifest_path: Path
    clustering_manifest_path: Path
    pca_path: Path
    labels_path: Path
    palette_path: Path
    table_path: Path
    manifest: Mapping[str, Any] = field(repr=False)
    clustering_manifest: Mapping[str, Any] = field(repr=False)
    scores: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)
    frame: pd.DataFrame = field(repr=False)
    palette: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class DirectedKNN:
    neighbors: np.ndarray = field(repr=False)
    similarities: np.ndarray = field(repr=False)
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class NativeUMAPResult:
    coordinates: np.ndarray = field(repr=False)
    receipt: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_umap_parameters(
    *,
    n_neighbors: int,
    min_dist: float,
    epochs: int,
    random_seed: int,
    device: str,
    dpi: int = DEFAULT_DPI,
) -> None:
    """Require the prespecified visualization parameters and CPU device."""

    if str(device) != "cpu":
        raise SO2HLUMAPError("The SO2 hL UMAP is CPU-only; --device must be 'cpu'.")
    if isinstance(n_neighbors, bool) or int(n_neighbors) != DEFAULT_N_NEIGHBORS:
        raise SO2HLUMAPError(
            f"The locked SO2 hL UMAP requires n_neighbors={DEFAULT_N_NEIGHBORS}."
        )
    if (
        not math.isfinite(float(min_dist))
        or float(min_dist) != DEFAULT_MIN_DIST
    ):
        raise SO2HLUMAPError(
            f"The locked SO2 hL UMAP requires min_dist={DEFAULT_MIN_DIST}."
        )
    if isinstance(epochs, bool) or int(epochs) != DEFAULT_EPOCHS:
        raise SO2HLUMAPError(
            f"The locked SO2 hL UMAP requires epochs={DEFAULT_EPOCHS}."
        )
    if isinstance(random_seed, bool) or int(random_seed) != DEFAULT_RANDOM_SEED:
        raise SO2HLUMAPError(
            f"The locked SO2 hL UMAP requires random_seed={DEFAULT_RANDOM_SEED}."
        )
    if isinstance(dpi, bool) or int(dpi) != DEFAULT_DPI:
        raise SO2HLUMAPError(
            f"The locked SO2 hL UMAP requires dpi={DEFAULT_DPI}."
        )


def _require_cuda_hidden() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value not in {"", "-1"}:
        raise SO2HLUMAPError(
            "Set CUDA_VISIBLE_DEVICES='' (or -1) for the CPU-only SO2 hL UMAP."
        )
    return str(value)


def _ordered_cluster_labels(labels: np.ndarray) -> list[str]:
    unique = np.unique(np.asarray(labels, dtype=np.int64))
    expected = np.arange(len(unique), dtype=np.int64)
    if not np.array_equal(unique, expected):
        raise SO2HLUMAPError("Contextual cluster numbers must be contiguous from zero.")
    return [f"C{int(value)}" for value in unique]


def _validate_palette(
    palette: Mapping[str, object], *, labels: list[str]
) -> dict[str, str]:
    if set(str(key) for key in palette) != set(labels):
        raise SO2HLUMAPError("Source palette does not cover the fixed cluster labels.")
    validated: dict[str, str] = {}
    for label in labels:
        color = str(palette[label])
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None:
            raise SO2HLUMAPError(f"Invalid hexadecimal color for {label}: {color!r}.")
        validated[label] = color.upper()
    if len({value.lower() for value in validated.values()}) != len(validated):
        raise SO2HLUMAPError("Contextual cluster palette colors must be unique.")
    return validated


def _source_knn_receipt(source: UMAPSource) -> Mapping[str, Any]:
    pipeline = source.clustering_manifest.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise SO2HLUMAPError("Source clustering manifest lacks pipeline provenance.")
    receipt = pipeline.get("knn")
    if not isinstance(receipt, Mapping):
        raise SO2HLUMAPError("Source clustering manifest lacks the kNN receipt.")
    return receipt


def _source_file_records(source: UMAPSource) -> dict[str, dict[str, Any]]:
    return {
        "analysis_manifest": _file_record(source.manifest_path),
        "clustering_manifest": _file_record(source.clustering_manifest_path),
        "pca": _file_record(source.pca_path),
        "labels": _file_record(source.labels_path),
        "palette": _file_record(source.palette_path),
        "cluster_table": _file_record(source.table_path),
    }


def load_locked_source(*, paths: ProjectPaths, run_id: str | None) -> UMAPSource:
    """Load and checksum-verify the completed resolution-1.0 source report."""

    selected_run = EXPECTED_RUN_ID if run_id is None else str(run_id)
    if selected_run != EXPECTED_RUN_ID:
        raise SO2HLUMAPError("The UMAP is locked to the completed SO2 14-core run.")
    root = paths.report_root / "analyses" / DEFAULT_SOURCE_REPORT / selected_run
    manifest_path = root / "manifest.json"
    clustering_manifest_path = root / "clustering" / "clustering_manifest.json"
    pca_path = root / "clustering" / "contextual_pca_l2_normalized.npy"
    labels_path = root / "clustering" / "contextual_labels.npy"
    palette_path = root / "clustering" / "contextual_palette.json"
    table_path = root / "tables" / "cell_contextual_clusters.parquet"

    manifest = _read_json(manifest_path, label="SO2 contextual-clustering manifest")
    _verify_source_manifest(root, manifest)
    if any(
        (
            manifest.get("schema") != SOURCE_ANALYSIS_SCHEMA,
            manifest.get("analysis_scope") != "contextual_hL_only",
            manifest.get("run_id") != EXPECTED_RUN_ID,
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(manifest.get("cluster_count", -1)) != EXPECTED_CLUSTER_COUNT,
            float(
                manifest.get("analysis_parameters", {}).get(
                    "leiden_resolution", math.nan
                )
            )
            != DEFAULT_LEIDEN_RESOLUTION,
        )
    ):
        raise SO2HLUMAPError(
            "Source report is not the locked SO2 hL resolution-1.0 analysis."
        )

    clustering_manifest = _read_json(
        clustering_manifest_path, label="SO2 hL clustering-stage manifest"
    )
    _verify_self_hash(clustering_manifest, label="SO2 hL clustering-stage manifest")
    configuration = clustering_manifest.get("configuration")
    if not isinstance(configuration, Mapping) or any(
        (
            int(configuration.get("n_neighbors", -1)) != DEFAULT_N_NEIGHBORS,
            float(configuration.get("leiden_resolution", math.nan))
            != DEFAULT_LEIDEN_RESOLUTION,
            int(configuration.get("pca_components_requested", -1)) != 50,
            int(configuration.get("random_seed", -1)) != DEFAULT_RANDOM_SEED,
            configuration.get("distance_metric") != "cosine",
        )
    ):
        raise SO2HLUMAPError("Source clustering parameters drifted from the lock.")

    scores = np.load(pca_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    if scores.shape != (EXPECTED_TOTAL_CELLS, 50) or scores.dtype != np.float32:
        raise SO2HLUMAPError("Source contextual PCA array shape or dtype is invalid.")
    if not np.isfinite(scores).all():
        raise SO2HLUMAPError("Source contextual PCA array contains non-finite values.")
    norms = np.linalg.norm(scores.astype(np.float64, copy=False), axis=1)
    if not np.allclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise SO2HLUMAPError("Source contextual PCA rows are not L2-normalized.")
    pipeline = clustering_manifest.get("pipeline")
    pca_receipt = pipeline.get("pca") if isinstance(pipeline, Mapping) else None
    if not isinstance(pca_receipt, Mapping) or not hmac.compare_digest(
        _array_sha256("normalized_pca_scores", scores),
        str(pca_receipt.get("normalized_scores_sha256", "")),
    ):
        raise SO2HLUMAPError("Source contextual PCA logical checksum changed.")

    if labels.shape != (EXPECTED_TOTAL_CELLS,) or labels.dtype != np.int64:
        raise SO2HLUMAPError("Source contextual label array shape or dtype is invalid.")
    ordered_labels = _ordered_cluster_labels(labels)
    if ordered_labels != [f"C{index}" for index in range(EXPECTED_CLUSTER_COUNT)]:
        raise SO2HLUMAPError("Source contextual label set is not C0 through C18.")
    leiden_receipt = pipeline.get("leiden") if isinstance(pipeline, Mapping) else None
    if not isinstance(leiden_receipt, Mapping) or not hmac.compare_digest(
        _array_sha256("sorted_leiden_labels", labels),
        str(leiden_receipt.get("labels_sha256", "")),
    ):
        raise SO2HLUMAPError("Source contextual label logical checksum changed.")

    palette_document = _read_json(palette_path, label="SO2 contextual palette")
    palette_value = palette_document.get("colors")
    if not isinstance(palette_value, Mapping):
        raise SO2HLUMAPError("Source contextual palette lacks a colors mapping.")
    palette = _validate_palette(palette_value, labels=ordered_labels)

    frame = pd.read_parquet(
        table_path,
        columns=[
            "global_cell_index",
            "core_number",
            "contextual_cluster_number",
            "contextual_cluster",
        ],
    )
    if len(frame) != EXPECTED_TOTAL_CELLS:
        raise SO2HLUMAPError("Source contextual cluster table has the wrong row count.")
    if not np.array_equal(
        frame["global_cell_index"].to_numpy(dtype=np.int64),
        np.arange(EXPECTED_TOTAL_CELLS, dtype=np.int64),
    ):
        raise SO2HLUMAPError("Source contextual cluster table row order drifted.")
    if not np.array_equal(
        frame["contextual_cluster_number"].to_numpy(dtype=np.int64), labels
    ):
        raise SO2HLUMAPError("Source table and contextual label array are misaligned.")
    expected_names = np.asarray([f"C{int(value)}" for value in labels], dtype=object)
    if not np.array_equal(
        frame["contextual_cluster"].astype(str).to_numpy(), expected_names
    ):
        raise SO2HLUMAPError("Source contextual cluster names are misaligned.")
    observed_core_order = tuple(
        int(value) for value in frame["core_number"].drop_duplicates().tolist()
    )
    if observed_core_order != SO2_CORE_NUMBERS:
        raise SO2HLUMAPError("Source contextual table core order drifted.")
    observed_counts = {
        int(key): int(value)
        for key, value in frame.groupby("core_number", sort=False).size().items()
    }
    if observed_counts != EXPECTED_CELL_COUNTS_BY_CORE:
        raise SO2HLUMAPError("Source contextual table per-core counts drifted.")

    return UMAPSource(
        root=root,
        manifest_path=manifest_path,
        clustering_manifest_path=clustering_manifest_path,
        pca_path=pca_path,
        labels_path=labels_path,
        palette_path=palette_path,
        table_path=table_path,
        manifest=manifest,
        clustering_manifest=clustering_manifest,
        scores=np.ascontiguousarray(scores),
        labels=np.ascontiguousarray(labels),
        frame=frame,
        palette=palette,
    )


def reconstruct_directed_faiss_knn(
    normalized_scores: np.ndarray, *, n_neighbors: int
) -> DirectedKNN:
    """Rebuild the directed FAISS-HNSW neighbors used by source clustering."""

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise SO2HLUMAPError(
            "FAISS is required; install the embedding-analysis dependencies."
        ) from exc

    values = np.ascontiguousarray(normalized_scores, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise SO2HLUMAPError("UMAP kNN input must be a finite [cells, PCA] array.")
    if (
        isinstance(n_neighbors, bool)
        or int(n_neighbors) <= 0
        or int(n_neighbors) >= len(values)
    ):
        raise SO2HLUMAPError("UMAP n_neighbors must be in [1, cells-1].")
    norms = np.linalg.norm(values.astype(np.float64, copy=False), axis=1)
    if not np.allclose(norms, 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise SO2HLUMAPError("UMAP cosine input is not L2-normalized.")

    n_cells, dimension = values.shape
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(dimension, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = max(HNSW_EF_SEARCH, int(n_neighbors) + 16)
    index.add(values)
    search_width = min(n_cells, int(n_neighbors) + 8)
    similarities, indices = index.search(values, search_width)
    selected = np.empty((n_cells, int(n_neighbors)), dtype=np.int64)
    selected_similarity = np.empty(
        (n_cells, int(n_neighbors)), dtype=np.float32
    )
    for row in range(n_cells):
        candidates = indices[row]
        scores = similarities[row]
        valid = (candidates >= 0) & (candidates != row)
        candidates = candidates[valid]
        scores = scores[valid]
        if len(candidates) < int(n_neighbors):
            raise SO2HLUMAPError(
                f"FAISS returned too few non-self neighbors for cell {row}."
            )
        order = np.lexsort((candidates, -scores))[: int(n_neighbors)]
        chosen = candidates[order]
        if len(np.unique(chosen)) != int(n_neighbors):
            raise SO2HLUMAPError("FAISS returned duplicate directed neighbors.")
        selected[row] = chosen
        selected_similarity[row] = scores[order]

    receipt = {
        "implementation": "faiss.IndexHNSWFlat",
        "faiss_version": getattr(faiss, "__version__", _package_version("faiss-cpu")),
        "distance_metric": "cosine_via_l2_normalized_inner_product",
        "single_threaded_index_and_search": True,
        "input_order": "locked_global_core_then_cell_index",
        "hnsw_m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "ef_search": int(index.hnsw.efSearch),
        "n_neighbors": int(n_neighbors),
        "directed_neighbor_entries": int(selected.size),
        "neighbor_storage_shape": list(selected.shape),
        "cell_by_cell_matrix_constructed": False,
        "maximum_explicit_pairwise_array_shape": list(indices.shape),
        "neighbors_sha256": _array_sha256("knn_neighbors", selected),
        "neighbor_cosine_sha256": _array_sha256(
            "knn_neighbor_cosine", selected_similarity
        ),
    }
    return DirectedKNN(
        neighbors=np.ascontiguousarray(selected),
        similarities=np.ascontiguousarray(selected_similarity),
        receipt=receipt,
    )


def verify_knn_against_source(
    result: DirectedKNN, source_receipt: Mapping[str, Any]
) -> None:
    """Fail if the reconstructed directed graph differs from the source graph."""

    for key in (
        "implementation",
        "faiss_version",
        "distance_metric",
        "hnsw_m",
        "ef_construction",
        "ef_search",
        "n_neighbors",
        "directed_neighbor_entries",
        "neighbor_storage_shape",
        "neighbors_sha256",
        "neighbor_cosine_sha256",
    ):
        if result.receipt.get(key) != source_receipt.get(key):
            raise SO2HLUMAPError(
                f"Reconstructed directed kNN graph disagrees with source field {key}."
            )


def deterministic_pca_initialization(scores: np.ndarray) -> np.ndarray:
    """Return a fixed two-dimensional PCA initialization scaled to [-10, 10]."""

    values = np.asarray(scores)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise SO2HLUMAPError("UMAP initialization requires two finite PCA columns.")
    initial = values[:, :2].astype(np.float64, copy=True)
    initial -= initial.mean(axis=0, dtype=np.float64)
    maximum = float(np.max(np.abs(initial)))
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise SO2HLUMAPError("UMAP PCA initialization has zero extent.")
    initial *= 10.0 / maximum
    return np.ascontiguousarray(initial)


def _one_native_umap_layout(
    *,
    graph: Any,
    weights: np.ndarray,
    initial: np.ndarray,
    min_dist: float,
    epochs: int,
    random_seed: int,
) -> np.ndarray:
    import igraph as ig

    ig.set_random_number_generator(random.Random(int(random_seed)))
    try:
        layout = graph.layout_umap(
            weights=weights,
            dim=2,
            seed=initial.copy(),
            min_dist=float(min_dist),
            epochs=int(epochs),
        )
    finally:
        ig.set_random_number_generator(None)
    coordinates = np.ascontiguousarray(np.asarray(layout.coords), dtype=np.float64)
    if coordinates.shape != (graph.vcount(), 2) or not np.isfinite(coordinates).all():
        raise SO2HLUMAPError("Native UMAP returned invalid coordinates.")
    if np.any(np.ptp(coordinates, axis=0) <= 0.0):
        raise SO2HLUMAPError("Native UMAP returned a zero-extent coordinate axis.")
    return coordinates


def run_native_umap(
    scores: np.ndarray,
    directed_knn: DirectedKNN,
    *,
    min_dist: float,
    epochs: int,
    random_seed: int,
    verify_determinism: bool = True,
    expected_undirected_edges: int | None = None,
    expected_undirected_edges_sha256: str | None = None,
) -> NativeUMAPResult:
    """Compute a genuine native-igraph UMAP from a directed cosine kNN graph."""

    try:
        import igraph as ig
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise SO2HLUMAPError(
            "python-igraph is required for native UMAP visualization."
        ) from exc

    values = np.asarray(scores)
    neighbors = np.asarray(directed_knn.neighbors, dtype=np.int64)
    similarities = np.asarray(directed_knn.similarities, dtype=np.float32)
    if (
        values.ndim != 2
        or neighbors.ndim != 2
        or similarities.shape != neighbors.shape
        or len(neighbors) != len(values)
    ):
        raise SO2HLUMAPError("UMAP scores and directed-neighbor arrays are misaligned.")
    if np.any(neighbors < 0) or np.any(neighbors >= len(values)):
        raise SO2HLUMAPError("UMAP directed-neighbor indices are out of range.")
    if not np.isfinite(similarities).all():
        raise SO2HLUMAPError("UMAP neighbor similarities contain non-finite values.")

    n_cells, n_neighbors = neighbors.shape
    if np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
        raise SO2HLUMAPError("UMAP directed graph has duplicate neighbors per source.")
    source = np.repeat(np.arange(n_cells, dtype=np.int32), n_neighbors)
    target = np.ascontiguousarray(neighbors.reshape(-1), dtype=np.int32)
    if np.any(source == target):
        raise SO2HLUMAPError("UMAP directed graph contains self edges.")
    edges = np.empty((len(source), 2), dtype=np.int32)
    edges[:, 0] = source
    edges[:, 1] = target
    distances = np.clip(
        1.0 - similarities.reshape(-1), 0.0, 2.0
    ).astype(np.float32, copy=False)
    if not np.isfinite(distances).all():
        raise SO2HLUMAPError("UMAP cosine distances contain non-finite values.")

    started = time.monotonic()
    graph = ig.Graph(n=n_cells, edges=edges, directed=True)
    if graph.vcount() != n_cells or graph.ecount() != len(edges):
        raise SO2HLUMAPError("igraph changed the directed UMAP graph topology.")
    weight_values = ig.umap_compute_weights(graph, distances)
    fuzzy_weights = np.asarray(weight_values, dtype=np.float64)
    del weight_values
    if (
        fuzzy_weights.shape != distances.shape
        or not np.isfinite(fuzzy_weights).all()
        or np.any(fuzzy_weights < 0.0)
        or np.any(fuzzy_weights > 1.0 + 1.0e-12)
    ):
        raise SO2HLUMAPError("igraph returned invalid UMAP fuzzy weights.")
    positive_weight_count = int(np.count_nonzero(fuzzy_weights > 0.0))
    if positive_weight_count <= 0:
        raise SO2HLUMAPError("The UMAP fuzzy graph has no positive edges.")

    positive = fuzzy_weights > 0.0
    positive_edges = edges[positive].astype(np.int64, copy=False)
    low = np.minimum(positive_edges[:, 0], positive_edges[:, 1])
    high = np.maximum(positive_edges[:, 0], positive_edges[:, 1])
    codes = np.sort(low * np.int64(n_cells) + high)
    if len(codes) != len(np.unique(codes)):
        raise SO2HLUMAPError("Positive UMAP fuzzy edges do not form a simple union.")
    union_edges = np.column_stack(
        (codes // np.int64(n_cells), codes % np.int64(n_cells))
    ).astype(np.int64, copy=False)
    union_sha = _array_sha256("knn_undirected_edges", union_edges)
    directed_edges_sha = _array_sha256("umap_directed_edges", edges)
    distances_sha = _array_sha256("umap_cosine_distances", distances)
    del positive, positive_edges, low, high, codes, union_edges

    if (
        expected_undirected_edges is not None
        and positive_weight_count != int(expected_undirected_edges)
    ):
        raise SO2HLUMAPError(
            "UMAP fuzzy graph does not match the expected undirected edge count."
        )
    if (
        expected_undirected_edges_sha256 is not None
        and not hmac.compare_digest(union_sha, str(expected_undirected_edges_sha256))
    ):
        raise SO2HLUMAPError(
            "UMAP positive fuzzy-edge union checksum differs from the expected graph."
        )

    initial = deterministic_pca_initialization(values)
    coordinates = _one_native_umap_layout(
        graph=graph,
        weights=fuzzy_weights,
        initial=initial,
        min_dist=min_dist,
        epochs=epochs,
        random_seed=random_seed,
    )
    coordinate_sha = _array_sha256("umap_coordinates", coordinates)
    repeated_sha: str | None = None
    if verify_determinism:
        repeated = _one_native_umap_layout(
            graph=graph,
            weights=fuzzy_weights,
            initial=initial,
            min_dist=min_dist,
            epochs=epochs,
            random_seed=random_seed,
        )
        repeated_sha = _array_sha256("umap_coordinates", repeated)
        if not hmac.compare_digest(coordinate_sha, repeated_sha) or not np.array_equal(
            coordinates, repeated
        ):
            raise SO2HLUMAPError("Immediate fixed-seed UMAP repeat was not deterministic.")

    receipt = {
        "implementation": "igraph.Graph.layout_umap",
        "weight_implementation": "igraph.umap_compute_weights",
        "igraph_version": getattr(ig, "__version__", _package_version("igraph")),
        "representation": "contextual_hL_exact_PCA50_L2_normalized",
        "metric": "cosine",
        "n_cells": int(n_cells),
        "n_neighbors": int(n_neighbors),
        "directed_edge_count": int(graph.ecount()),
        "positive_fuzzy_weight_count": positive_weight_count,
        "positive_fuzzy_undirected_edges_sha256": union_sha,
        "dense_cell_by_cell_matrix_constructed": False,
        "directed_edges_dtype": str(edges.dtype),
        "directed_edges_sha256": directed_edges_sha,
        "distance_array_shape": list(distances.shape),
        "distance_dtype": str(distances.dtype),
        "distance_sha256": distances_sha,
        "fuzzy_weight_array_shape": list(fuzzy_weights.shape),
        "fuzzy_weight_dtype": str(fuzzy_weights.dtype),
        "fuzzy_weight_min": float(fuzzy_weights.min()),
        "fuzzy_weight_max": float(fuzzy_weights.max()),
        "fuzzy_weights_sha256": _array_sha256(
            "umap_fuzzy_weights", fuzzy_weights
        ),
        "initialization": "first_two_locked_PCA_scores_centered_and_max_abs_scaled_to_10",
        "initialization_sha256": _array_sha256("umap_initialization", initial),
        "rng": "igraph_process_global_random.Random_reset_before_each_layout",
        "rng_restored_to_igraph_default_after_each_layout": True,
        "min_dist": float(min_dist),
        "epochs": int(epochs),
        "random_seed": int(random_seed),
        "coordinate_shape": list(coordinates.shape),
        "coordinate_dtype": str(coordinates.dtype),
        "coordinate_bounds": {
            "umap_1_min": float(coordinates[:, 0].min()),
            "umap_1_max": float(coordinates[:, 0].max()),
            "umap_2_min": float(coordinates[:, 1].min()),
            "umap_2_max": float(coordinates[:, 1].max()),
        },
        "coordinate_sha256": coordinate_sha,
        "determinism_repeat_performed": bool(verify_determinism),
        "determinism_repeat_coordinate_sha256": repeated_sha,
        "elapsed_seconds_including_repeat": float(time.monotonic() - started),
    }
    return NativeUMAPResult(coordinates=coordinates, receipt=receipt)


def build_privacy_minimized_table(
    frame: pd.DataFrame, coordinates: np.ndarray
) -> pd.DataFrame:
    """Align UMAP coordinates without exporting cell keys or expression values."""

    required = {
        "global_cell_index",
        "core_number",
        "contextual_cluster_number",
        "contextual_cluster",
    }
    if not required.issubset(frame.columns):
        raise SO2HLUMAPError("Source frame lacks columns required for UMAP alignment.")
    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (len(frame), 2) or not np.isfinite(values).all():
        raise SO2HLUMAPError("UMAP coordinate table input is invalid.")
    output = frame[
        [
            "global_cell_index",
            "core_number",
            "contextual_cluster_number",
            "contextual_cluster",
        ]
    ].copy()
    output["umap_1"] = values[:, 0]
    output["umap_2"] = values[:, 1]
    return output


def core_palette() -> dict[int, str]:
    """Return a fixed categorical palette for the fourteen numeric cores."""

    colors = (
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
        "#393B79",
        "#637939",
        "#8C6D31",
        "#843C39",
    )
    return dict(zip(SO2_CORE_NUMBERS, colors, strict=True))


def render_umap_figure(
    *,
    coordinates: np.ndarray,
    frame: pd.DataFrame,
    cluster_palette: Mapping[str, str],
    png_path: Path,
    pdf_path: Path,
    random_seed: int,
    dpi: int,
) -> Mapping[str, Any]:
    """Render identical UMAP coordinates by fixed cluster and by core."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    values = np.asarray(coordinates, dtype=np.float64)
    if values.shape != (len(frame), 2) or not np.isfinite(values).all():
        raise SO2HLUMAPError("Cannot plot invalid UMAP coordinates.")
    if isinstance(dpi, bool) or int(dpi) <= 0:
        raise SO2HLUMAPError("Figure DPI must be positive.")
    cluster_labels = [f"C{index}" for index in range(EXPECTED_CLUSTER_COUNT)]
    if set(cluster_palette) != set(cluster_labels):
        raise SO2HLUMAPError("Figure palette does not match C0 through C18.")
    core_colors = core_palette()
    cluster_values = frame["contextual_cluster"].astype(str).to_numpy()
    core_values = frame["core_number"].to_numpy(dtype=np.int64)
    order = np.random.default_rng(int(random_seed)).permutation(len(frame))

    figure, axes = plt.subplots(1, 2, figsize=(18.0, 9.2), constrained_layout=True)
    for axis in axes:
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("UMAP 1", fontsize=10)
        axis.set_ylabel("UMAP 2", fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)

    cluster_colors = np.asarray(
        [cluster_palette[label] for label in cluster_values], dtype=object
    )
    axes[0].scatter(
        values[order, 0],
        values[order, 1],
        c=cluster_colors[order],
        s=0.45,
        alpha=0.62,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )
    axes[0].set_title(
        "Fixed joint Leiden clusters (resolution 1.0)", fontsize=13, weight="bold"
    )
    for label in cluster_labels:
        selected = cluster_values == label
        center = np.median(values[selected], axis=0)
        annotation = axes[0].text(
            float(center[0]),
            float(center[1]),
            label,
            color="#111827",
            fontsize=8,
            weight="bold",
            ha="center",
            va="center",
            zorder=4,
        )
        annotation.set_path_effects(
            [path_effects.withStroke(linewidth=2.5, foreground="white")]
        )
    cluster_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor=cluster_palette[label],
            markeredgecolor="none",
            label=label,
        )
        for label in cluster_labels
    ]
    axes[0].legend(
        handles=cluster_handles,
        title="Contextual cluster",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=7,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    core_point_colors = np.asarray(
        [core_colors[int(core)] for core in core_values], dtype=object
    )
    axes[1].scatter(
        values[order, 0],
        values[order, 1],
        c=core_point_colors[order],
        s=0.45,
        alpha=0.62,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )
    axes[1].set_title("Same coordinates colored by tissue core", fontsize=13, weight="bold")
    core_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5,
            markerfacecolor=core_colors[number],
            markeredgecolor="none",
            label=f"Core {number}",
        )
        for number in SO2_CORE_NUMBERS
    ]
    axes[1].legend(
        handles=core_handles,
        title="SO2 tissue core",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=5,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    figure.suptitle(
        "SO2 14-core contextual hL UMAP — 246,063 cells",
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.5,
        0.002,
        "Model-derived exploratory visualization; UMAP geometry does not establish cell type, mechanism, or causality.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    figure.set_dpi(int(dpi))
    with matplotlib.rc_context({"savefig.dpi": int(dpi)}):
        _atomic_save_figure_pair(
            figure, png_path=png_path, pdf_path=pdf_path, dpi=int(dpi)
        )
    plt.close(figure)
    return {
        "plot_type": "two_panel_identical_coordinates",
        "left_color": "fixed_joint_contextual_leiden_resolution_1p0",
        "right_color": "numeric_tissue_core_confounding_control",
        "point_count_per_panel": int(len(frame)),
        "point_size": 0.45,
        "point_alpha": 0.62,
        "borderless_points": True,
        "rasterized_point_layer": True,
        "draw_order": "fixed_seed_random_permutation",
        "draw_order_seed": int(random_seed),
        "cluster_palette": dict(cluster_palette),
        "core_palette": {str(key): value for key, value in core_colors.items()},
        "cluster_labels_annotated_at_coordinate_median": True,
        "png_dpi": int(dpi),
        "pdf_raster_dpi": int(dpi),
    }


def _render_readme(*, run_id: str, receipt: Mapping[str, Any]) -> str:
    coordinate_hash = receipt["umap"]["coordinate_sha256"]
    layout_seconds = float(receipt["umap"]["elapsed_seconds_including_repeat"])
    return f"""# SO2 14-core contextual hL UMAP

This is a CPU-only two-dimensional UMAP of all {EXPECTED_TOTAL_CELLS:,} cells
from the completed joint contextual hL clustering for run `{run_id}`. The left
panel colors the immutable Leiden resolution-1.0 labels C0 through C18; the
right panel colors the exact same coordinates by numeric tissue core to expose
possible core-driven structure.

## Method

The workflow reused the checksum-verified 50-component, L2-normalized PCA
representation of contextual `hL`. It reconstructed the original directed
cosine {DEFAULT_N_NEIGHBORS}-nearest-neighbor graph with single-threaded FAISS
HNSW, verified the directed-neighbor and similarity hashes against the source
clustering receipt, computed UMAP fuzzy weights with python-igraph, and ran
native UMAP for {DEFAULT_EPOCHS} epochs with `min_dist={DEFAULT_MIN_DIST}` and
seed {DEFAULT_RANDOM_SEED}. An immediate second layout produced the identical
coordinate hash `{coordinate_hash}`. No model inference, retraining, or
reclustering occurred. The two native layouts took {layout_seconds:.1f} CPU
seconds in total; the PNG and rasterized PDF point layers were both rendered
at {DEFAULT_DPI} DPI.

## Observed display

The fixed cluster labels occupy visually coherent local regions, but the most
prominent detached islands also track tissue-core identity. In the source
cluster table, C4 and C6 are respectively 99.76% and 99.87% core 15; C1 and C8
are 98.44% and 99.45% core 23; and C12 is 96.63% core 25. The central manifold
contains visibly more mixed core colors. This supports the narrow claim that
the fixed clusters have coherent displayed geometry while also flagging core
structure as a strong alternative explanation for several islands.

## Interpretation limits

UMAP preserves selected local graph relationships imperfectly. Its axes,
orientation, island area, and distances between separated islands are
visualization coordinates, not biological measurements. Apparent cluster
separation does not validate cell identities or mechanisms. Five source
clusters (C1, C4, C6, C8, and C12) contain more than 90% of their cells from one
core, so the core-colored panel is essential context. This single-seed,
transductive, all-fit analysis does not establish patient generalization,
signaling, biological influence, or causality.

The aligned Parquet table omits cell keys, tissue coordinates, expression,
learned vectors, clinical fields, and donor identifiers. `global_cell_index`
is only the zero-based locked source-report row position.

## Reproduction

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\
  render-so2-hl-umap \\
  --run-id {run_id} \\
  --n-neighbors {DEFAULT_N_NEIGHBORS} \\
  --min-dist {DEFAULT_MIN_DIST} \\
  --epochs {DEFAULT_EPOCHS} \\
  --random-seed {DEFAULT_RANDOM_SEED} \\
  --device cpu \\
  --dpi {DEFAULT_DPI}
```
"""


def _required_outputs() -> set[str]:
    return {
        "README.md",
        f"umap/{COORDINATE_FILENAME}",
        "umap/umap_receipt.json",
        f"tables/{TABLE_FILENAME}",
        f"figures/{FIGURE_STEM}.png",
        f"figures/{FIGURE_STEM}.pdf",
    }


def verify_umap_manifest(output_root: Path, manifest: Mapping[str, Any]) -> None:
    """Verify the complete additive UMAP report and its privacy contract."""

    _verify_self_hash(manifest, label="SO2 hL UMAP manifest")
    expected_core_counts = {
        str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
        for number in SO2_CORE_NUMBERS
    }
    expected_parameters = {
        "n_neighbors": DEFAULT_N_NEIGHBORS,
        "metric": "cosine",
        "min_dist": DEFAULT_MIN_DIST,
        "epochs": DEFAULT_EPOCHS,
        "random_seed": DEFAULT_RANDOM_SEED,
        "dpi": DEFAULT_DPI,
        "leiden_labels_recomputed": False,
    }
    if any(
        (
            manifest.get("schema") != UMAP_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("run_id") != EXPECTED_RUN_ID,
            manifest.get("analysis_scope")
            != "contextual_hL_resolution_1p0_umap_visualization_only",
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            manifest.get("core_cell_counts") != expected_core_counts,
            int(manifest.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
            int(manifest.get("cluster_count", -1)) != EXPECTED_CLUSTER_COUNT,
            float(manifest.get("leiden_resolution", math.nan))
            != DEFAULT_LEIDEN_RESOLUTION,
            manifest.get("parameters") != expected_parameters,
        )
    ):
        raise SO2HLUMAPError("SO2 hL UMAP manifest identity is invalid.")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != _required_outputs():
        raise SO2HLUMAPError("SO2 hL UMAP output file set is incomplete or changed.")
    for relative, record in files.items():
        path = output_root / str(relative)
        if not isinstance(record, Mapping) or not path.is_file():
            raise SO2HLUMAPError(f"SO2 hL UMAP output is missing: {relative}")
        if _file_record(path) != dict(record):
            raise SO2HLUMAPError(f"SO2 hL UMAP output checksum changed: {relative}")

    receipt = _read_json(
        output_root / "umap" / "umap_receipt.json", label="SO2 hL UMAP receipt"
    )
    _verify_self_hash(receipt, label="SO2 hL UMAP receipt")
    if any(
        (
            receipt.get("schema") != UMAP_RECEIPT_SCHEMA,
            receipt.get("status") != "complete",
            receipt.get("run_id") != EXPECTED_RUN_ID,
            manifest.get("source_artifacts") != receipt.get("source"),
            manifest.get("umap_receipt")
            != _file_record(output_root / "umap" / "umap_receipt.json"),
        )
    ):
        raise SO2HLUMAPError("SO2 hL UMAP receipt schema is invalid.")
    umap_receipt = receipt.get("umap")
    if not isinstance(umap_receipt, Mapping) or any(
        (
            int(umap_receipt.get("n_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(umap_receipt.get("n_neighbors", -1)) != DEFAULT_N_NEIGHBORS,
            float(umap_receipt.get("min_dist", math.nan)) != DEFAULT_MIN_DIST,
            int(umap_receipt.get("epochs", -1)) != DEFAULT_EPOCHS,
            int(umap_receipt.get("random_seed", -1)) != DEFAULT_RANDOM_SEED,
            not bool(umap_receipt.get("determinism_repeat_performed")),
            umap_receipt.get("coordinate_sha256")
            != umap_receipt.get("determinism_repeat_coordinate_sha256"),
        )
    ):
        raise SO2HLUMAPError("SO2 hL UMAP receipt parameters are invalid.")
    if manifest.get("coordinate_sha256") != umap_receipt.get("coordinate_sha256"):
        raise SO2HLUMAPError("Manifest and UMAP coordinate checksums disagree.")

    directed_knn = receipt.get("directed_knn")
    source_knn = receipt.get("source_knn_hashes")
    if not isinstance(directed_knn, Mapping) or not isinstance(source_knn, Mapping):
        raise SO2HLUMAPError("SO2 hL UMAP kNN provenance is incomplete.")
    if any(
        (
            int(directed_knn.get("n_neighbors", -1)) != DEFAULT_N_NEIGHBORS,
            int(directed_knn.get("directed_neighbor_entries", -1))
            != EXPECTED_TOTAL_CELLS * DEFAULT_N_NEIGHBORS,
            directed_knn.get("neighbors_sha256")
            != source_knn.get("neighbors_sha256"),
            directed_knn.get("neighbor_cosine_sha256")
            != source_knn.get("neighbor_cosine_sha256"),
            int(source_knn.get("undirected_edges", -1))
            != int(umap_receipt.get("positive_fuzzy_weight_count", -2)),
            source_knn.get("undirected_edges_sha256")
            != umap_receipt.get("positive_fuzzy_undirected_edges_sha256"),
        )
    ):
        raise SO2HLUMAPError("SO2 hL UMAP graph receipts disagree.")

    figure = receipt.get("figure")
    if not isinstance(figure, Mapping):
        raise SO2HLUMAPError("SO2 hL UMAP figure provenance is incomplete.")
    raw_palette = figure.get("cluster_palette")
    if not isinstance(raw_palette, Mapping):
        raise SO2HLUMAPError("SO2 hL UMAP cluster palette is missing.")
    figure_palette = _validate_palette(
        raw_palette, labels=list(EXPECTED_CLUSTER_LABELS)
    )
    expected_core_palette = {
        str(key): value for key, value in core_palette().items()
    }
    if any(
        (
            list(figure_palette) != list(EXPECTED_CLUSTER_LABELS),
            figure.get("core_palette") != expected_core_palette,
            int(figure.get("point_count_per_panel", -1)) != EXPECTED_TOTAL_CELLS,
            int(figure.get("draw_order_seed", -1)) != DEFAULT_RANDOM_SEED,
            int(figure.get("png_dpi", -1)) != DEFAULT_DPI,
            int(figure.get("pdf_raster_dpi", -1)) != DEFAULT_DPI,
            figure.get("plot_type") != "two_panel_identical_coordinates",
            figure.get("left_color")
            != "fixed_joint_contextual_leiden_resolution_1p0",
            figure.get("right_color")
            != "numeric_tissue_core_confounding_control",
        )
    ):
        raise SO2HLUMAPError("SO2 hL UMAP figure contract changed.")

    expected_output_records = {
        "coordinates": _file_record(output_root / "umap" / COORDINATE_FILENAME),
        "table": _file_record(output_root / "tables" / TABLE_FILENAME),
        "png": _file_record(output_root / "figures" / f"{FIGURE_STEM}.png"),
        "pdf": _file_record(output_root / "figures" / f"{FIGURE_STEM}.pdf"),
    }
    if receipt.get("outputs") != expected_output_records:
        raise SO2HLUMAPError("SO2 hL UMAP receipt output checksums changed.")

    coordinates = np.load(
        output_root / "umap" / COORDINATE_FILENAME, allow_pickle=False
    )
    if (
        coordinates.shape != (EXPECTED_TOTAL_CELLS, 2)
        or coordinates.dtype != np.float64
        or not np.isfinite(coordinates).all()
        or _array_sha256("umap_coordinates", coordinates)
        != umap_receipt.get("coordinate_sha256")
    ):
        raise SO2HLUMAPError("Persisted SO2 hL UMAP coordinates are invalid.")
    table = pd.read_parquet(output_root / "tables" / TABLE_FILENAME)
    expected_columns = [
        "global_cell_index",
        "core_number",
        "contextual_cluster_number",
        "contextual_cluster",
        "umap_1",
        "umap_2",
    ]
    if list(table.columns) != expected_columns or len(table) != EXPECTED_TOTAL_CELLS:
        raise SO2HLUMAPError("Persisted SO2 hL UMAP table schema is invalid.")
    if not np.array_equal(
        table["global_cell_index"].to_numpy(dtype=np.int64),
        np.arange(EXPECTED_TOTAL_CELLS, dtype=np.int64),
    ):
        raise SO2HLUMAPError("Persisted SO2 hL UMAP table order is invalid.")
    if not np.array_equal(
        table[["umap_1", "umap_2"]].to_numpy(dtype=np.float64), coordinates
    ):
        raise SO2HLUMAPError("Persisted SO2 hL UMAP table coordinates are misaligned.")
    if not all(
        pd.api.types.is_integer_dtype(table[column].dtype)
        for column in (
            "global_cell_index",
            "core_number",
            "contextual_cluster_number",
        )
    ):
        raise SO2HLUMAPError("Persisted SO2 hL UMAP identifiers must be integers.")
    observed_core_order = tuple(
        int(value) for value in table["core_number"].drop_duplicates().tolist()
    )
    observed_core_counts = {
        int(key): int(value)
        for key, value in table.groupby("core_number", sort=False).size().items()
    }
    if (
        observed_core_order != SO2_CORE_NUMBERS
        or observed_core_counts != EXPECTED_CELL_COUNTS_BY_CORE
    ):
        raise SO2HLUMAPError("Persisted SO2 hL UMAP core order or counts changed.")
    cluster_numbers = table["contextual_cluster_number"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        np.unique(cluster_numbers), np.arange(EXPECTED_CLUSTER_COUNT, dtype=np.int64)
    ):
        raise SO2HLUMAPError("Persisted SO2 hL UMAP clusters are not C0 through C18.")
    expected_names = np.asarray(
        [f"C{int(value)}" for value in cluster_numbers], dtype=object
    )
    if not np.array_equal(
        table["contextual_cluster"].astype(str).to_numpy(), expected_names
    ):
        raise SO2HLUMAPError("Persisted SO2 hL UMAP cluster names are misaligned.")
    prohibited = {
        "cell_key",
        "cell_index",
        "x_um",
        "y_um",
        "expression",
        "hL",
        "donor",
        "patient",
    }
    if prohibited.intersection(table.columns):
        raise SO2HLUMAPError("Persisted SO2 hL UMAP table violates privacy scope.")


def verify_umap_source_binding(output_root: Path, source: UMAPSource) -> None:
    """Bind a completed UMAP bundle to the currently verified source report."""

    receipt = _read_json(
        output_root / "umap" / "umap_receipt.json", label="SO2 hL UMAP receipt"
    )
    _verify_self_hash(receipt, label="SO2 hL UMAP receipt")
    if receipt.get("source") != _source_file_records(source):
        raise SO2HLUMAPError("UMAP source artifact receipts differ from current files.")

    source_knn = _source_knn_receipt(source)
    expected_knn_hashes = {
        "neighbors_sha256": source_knn["neighbors_sha256"],
        "neighbor_cosine_sha256": source_knn["neighbor_cosine_sha256"],
        "undirected_edges": source_knn["undirected_edges"],
        "undirected_edges_sha256": source_knn["undirected_edges_sha256"],
    }
    if receipt.get("source_knn_hashes") != expected_knn_hashes:
        raise SO2HLUMAPError("UMAP kNN receipts differ from the current source report.")

    figure = receipt.get("figure")
    if not isinstance(figure, Mapping) or figure.get("cluster_palette") != dict(
        source.palette
    ):
        raise SO2HLUMAPError("UMAP cluster palette differs from the current source.")

    table = pd.read_parquet(
        output_root / "tables" / TABLE_FILENAME,
        columns=["core_number", "contextual_cluster_number"],
    )
    if not np.array_equal(
        table["contextual_cluster_number"].to_numpy(dtype=np.int64), source.labels
    ) or not np.array_equal(
        table["core_number"].to_numpy(dtype=np.int64),
        source.frame["core_number"].to_numpy(dtype=np.int64),
    ):
        raise SO2HLUMAPError("UMAP table is not row-bound to the current source report.")


def _resolve_output_root(
    *, paths: ProjectPaths, run_id: str, output_dir: str | Path | None
) -> Path:
    allowed_root = (paths.report_root / "analyses").resolve(strict=False)
    if output_dir is None:
        output_root = allowed_root / DEFAULT_OUTPUT_REPORT / run_id
    else:
        output_root = Path(output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = paths.project_root / output_root
        output_root = output_root.resolve(strict=False)
    if output_root == allowed_root or not output_root.is_relative_to(allowed_root):
        raise SO2HLUMAPError(
            "SO2 hL UMAP output must be a child of the configured reports/analyses root."
        )
    return output_root


def run_so2_hl_umap(
    *,
    paths: ProjectPaths,
    run_id: str | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    epochs: int = DEFAULT_EPOCHS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    device: str = "cpu",
    dpi: int = DEFAULT_DPI,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the complete locked SO2 contextual hL UMAP report."""

    validate_umap_parameters(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        epochs=epochs,
        random_seed=random_seed,
        device=device,
        dpi=dpi,
    )
    cuda_visibility = _require_cuda_hidden()
    selected_run = EXPECTED_RUN_ID if run_id is None else str(run_id)
    source = load_locked_source(paths=paths, run_id=selected_run)
    output_root = _resolve_output_root(
        paths=paths, run_id=selected_run, output_dir=output_dir
    )
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, label="SO2 hL UMAP manifest")
        verify_umap_manifest(output_root, manifest)
        verify_umap_source_binding(output_root, source)
    else:
        if output_root.exists():
            raise SO2HLUMAPError(
                "SO2 hL UMAP output path exists without a complete manifest."
            )
        final_output_root = output_root
        active_parent = (
            paths.scratch_root
            / "active_runs"
            / selected_run
            / "posthoc_reports"
        )
        active_parent.mkdir(parents=True, exist_ok=True)
        final_output_root.parent.mkdir(parents=True, exist_ok=True)
        if os.stat(active_parent).st_dev != os.stat(final_output_root.parent).st_dev:
            raise SO2HLUMAPError(
                "Scratch and report roots are on different filesystems; atomic "
                "UMAP report publication is unavailable."
            )
        staging_context = tempfile.TemporaryDirectory(
            prefix=f".{DEFAULT_OUTPUT_REPORT}.tmp-",
            dir=active_parent,
            ignore_cleanup_errors=True,
        )
        output_root = Path(staging_context.name)
        manifest_path = output_root / "manifest.json"
        source_records = _source_file_records(source)
        directed_knn = reconstruct_directed_faiss_knn(
            source.scores, n_neighbors=n_neighbors
        )
        source_knn = _source_knn_receipt(source)
        verify_knn_against_source(directed_knn, source_knn)
        umap = run_native_umap(
            source.scores,
            directed_knn,
            min_dist=min_dist,
            epochs=epochs,
            random_seed=random_seed,
            verify_determinism=True,
            expected_undirected_edges=int(source_knn.get("undirected_edges", -1)),
            expected_undirected_edges_sha256=str(
                source_knn.get("undirected_edges_sha256", "")
            ),
        )
        expected_positive = int(source_knn.get("undirected_edges", -1))
        if int(umap.receipt["positive_fuzzy_weight_count"]) != expected_positive:
            raise SO2HLUMAPError(
                "UMAP fuzzy graph does not match the source undirected kNN union."
            )
        if umap.receipt["positive_fuzzy_undirected_edges_sha256"] != source_knn.get(
            "undirected_edges_sha256"
        ):
            raise SO2HLUMAPError(
                "UMAP positive fuzzy-edge union checksum differs from the source graph."
            )

        coordinate_path = output_root / "umap" / COORDINATE_FILENAME
        table_path = output_root / "tables" / TABLE_FILENAME
        png_path = output_root / "figures" / f"{FIGURE_STEM}.png"
        pdf_path = output_root / "figures" / f"{FIGURE_STEM}.pdf"
        _atomic_write_npy(coordinate_path, umap.coordinates)
        table = build_privacy_minimized_table(source.frame, umap.coordinates)
        _atomic_write_parquet(table_path, table)
        figure_receipt = render_umap_figure(
            coordinates=umap.coordinates,
            frame=table,
            cluster_palette=source.palette,
            png_path=png_path,
            pdf_path=pdf_path,
            random_seed=random_seed,
            dpi=dpi,
        )
        if _source_file_records(source) != source_records:
            raise SO2HLUMAPError("Source report artifacts changed during UMAP generation.")

        receipt = _receipt_with_self_hash(
            {
                "schema": UMAP_RECEIPT_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "run_id": selected_run,
                "source": source_records,
                "directed_knn": dict(directed_knn.receipt),
                "source_knn_hashes": {
                    "neighbors_sha256": source_knn["neighbors_sha256"],
                    "neighbor_cosine_sha256": source_knn[
                        "neighbor_cosine_sha256"
                    ],
                    "undirected_edges": source_knn["undirected_edges"],
                    "undirected_edges_sha256": source_knn[
                        "undirected_edges_sha256"
                    ],
                },
                "umap": dict(umap.receipt),
                "figure": dict(figure_receipt),
                "outputs": {
                    "coordinates": _file_record(coordinate_path),
                    "table": _file_record(table_path),
                    "png": _file_record(png_path),
                    "pdf": _file_record(pdf_path),
                },
            }
        )
        receipt_path = output_root / "umap" / "umap_receipt.json"
        _atomic_write_json(receipt_path, receipt)
        _atomic_write_text(
            output_root / "README.md",
            _render_readme(run_id=selected_run, receipt=receipt),
        )
        manifest = _receipt_with_self_hash(
            {
                "schema": UMAP_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "run_id": selected_run,
                "analysis_scope": "contextual_hL_resolution_1p0_umap_visualization_only",
                "leiden_resolution": DEFAULT_LEIDEN_RESOLUTION,
                "core_order": list(SO2_CORE_NUMBERS),
                "core_cell_counts": {
                    str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                    for number in SO2_CORE_NUMBERS
                },
                "point_count": EXPECTED_TOTAL_CELLS,
                "cluster_count": EXPECTED_CLUSTER_COUNT,
                "parameters": {
                    "n_neighbors": int(n_neighbors),
                    "metric": "cosine",
                    "min_dist": float(min_dist),
                    "epochs": int(epochs),
                    "random_seed": int(random_seed),
                    "dpi": int(dpi),
                    "leiden_labels_recomputed": False,
                },
                "execution": {
                    "cpu_only": True,
                    "device": "cpu",
                    "cuda_visible_devices": cuda_visibility,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "faiss": directed_knn.receipt["faiss_version"],
                    "igraph": umap.receipt["igraph_version"],
                    "pandas": pd.__version__,
                    "pyarrow": _package_version("pyarrow"),
                    "matplotlib": _package_version("matplotlib"),
                    "layout_elapsed_seconds_including_repeat": umap.receipt[
                        "elapsed_seconds_including_repeat"
                    ],
                    "model_inference": False,
                    "retraining": False,
                    "reclustering": False,
                    "dense_cell_by_cell_matrix_constructed": False,
                },
                "source_artifacts": receipt["source"],
                "umap_receipt": _file_record(receipt_path),
                "coordinate_sha256": umap.receipt["coordinate_sha256"],
                "interpretation": {
                    "exploratory_visualization": True,
                    "fixed_model_derived_clusters": True,
                    "cell_types_established": False,
                    "patient_generalization_established": False,
                    "biological_influence_established": False,
                    "causality_established": False,
                    "core_colored_confounding_control_included": True,
                    "observed_visual_result": (
                        "Fixed labels are locally coherent, but prominent detached "
                        "islands track cores 15, 23, and 25; the central manifold is "
                        "more core-mixed."
                    ),
                    "strongest_alternative_explanation": (
                        "Source-core structure contributes materially to the visible "
                        "separation of several clusters."
                    ),
                },
                "privacy": {
                    "stable_cell_key_included": False,
                    "tissue_coordinates_included": False,
                    "expression_included": False,
                    "embedding_vectors_included": False,
                    "clinical_or_donor_fields_included": False,
                    "global_cell_index_is_source_row_position_only": True,
                },
                "files": _file_manifest(output_root),
            }
        )
        _atomic_write_json(manifest_path, manifest)
        verify_umap_manifest(output_root, manifest)
        verify_umap_source_binding(output_root, source)
        if final_output_root.exists():
            raise SO2HLUMAPError("SO2 hL UMAP destination appeared during generation.")
        os.replace(output_root, final_output_root)
        staging_context.cleanup()
        output_root = final_output_root
        manifest_path = output_root / "manifest.json"
        manifest = _read_json(manifest_path, label="SO2 hL UMAP manifest")
        verify_umap_manifest(output_root, manifest)
        verify_umap_source_binding(output_root, source)

    return {
        "status": "complete",
        "run_id": selected_run,
        "point_count": int(manifest["point_count"]),
        "cluster_count": int(manifest["cluster_count"]),
        "leiden_resolution": float(manifest["leiden_resolution"]),
        "coordinate_sha256": str(manifest["coordinate_sha256"]),
        "output_dir": output_root.as_posix(),
        "png": (output_root / "figures" / f"{FIGURE_STEM}.png").as_posix(),
        "pdf": (output_root / "figures" / f"{FIGURE_STEM}.pdf").as_posix(),
        "manifest": manifest_path.as_posix(),
    }


__all__ = [
    "COORDINATE_FILENAME",
    "DEFAULT_DPI",
    "DEFAULT_EPOCHS",
    "DEFAULT_MIN_DIST",
    "DEFAULT_N_NEIGHBORS",
    "DEFAULT_RANDOM_SEED",
    "DirectedKNN",
    "FIGURE_STEM",
    "NativeUMAPResult",
    "SO2HLUMAPError",
    "TABLE_FILENAME",
    "UMAP_SCHEMA",
    "build_privacy_minimized_table",
    "core_palette",
    "deterministic_pca_initialization",
    "load_locked_source",
    "reconstruct_directed_faiss_knn",
    "render_umap_figure",
    "run_native_umap",
    "run_so2_hl_umap",
    "validate_umap_parameters",
    "verify_knn_against_source",
    "verify_umap_manifest",
    "verify_umap_source_binding",
]
