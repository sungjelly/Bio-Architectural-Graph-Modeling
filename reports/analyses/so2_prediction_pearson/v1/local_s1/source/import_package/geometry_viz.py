"""Geometry-only visualization of one prepared spatial graph.

Only coordinates, split labels, macroblock identifiers, topology, and physical
edge distances are loaded. Expression matrices and biological labels are never
read or emitted.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


FORMAT_VERSION = 1
ARTIFACT_KIND = "geometry_only_spatial_graph_visualization"
PREPARED_ARTIFACT_KIND = "normal_true_tissue_spatial_benchmark_preparation"
GRAPH_GRID_ARTIFACT_KIND = "geometry_only_graph_grid"
LOADED_PREPARED_ARRAY_KEYS = (
    "coordinates_um",
    "split_labels",
    "macroblock_ids",
)
LOADED_GRAPH_ARRAY_KEYS = (
    "edge_index",
    "edge_attributes_raw",
    "edge_attribute_names",
)


class GeometryVisualizationError(ValueError):
    """Raised when geometry artifacts cannot support a safe visualization."""


@dataclass(frozen=True)
class DenseWindow:
    """A deterministically selected axis-aligned physical window."""

    x_min_um: float
    x_max_um: float
    y_min_um: float
    y_max_um: float
    center_x_um: float
    center_y_um: float
    width_um: float
    stride_um: float
    n_nodes: int
    candidate_count: int
    node_indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.ascontiguousarray(self.node_indices, dtype=np.int64)
        indices.flags.writeable = False
        object.__setattr__(self, "node_indices", indices)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "selection": "maximum cell count on fixed half-window-stride grid",
            "tie_break": "lowest y center, then lowest x center",
            "width_um": self.width_um,
            "stride_um": self.stride_um,
            "candidate_count": self.candidate_count,
            "n_nodes": self.n_nodes,
            "center_um": [self.center_x_um, self.center_y_um],
            "bounds_um": {
                "x": [self.x_min_um, self.x_max_um],
                "y": [self.y_min_um, self.y_max_um],
            },
        }


@dataclass(frozen=True)
class GeometryData:
    coordinates_um: np.ndarray
    split_labels: np.ndarray
    macroblock_codes: np.ndarray
    n_macroblocks: int
    edge_index: np.ndarray
    edge_distances_um: np.ndarray
    undirected_edge_mask: np.ndarray
    degree: np.ndarray
    component_labels: np.ndarray
    component_sizes: np.ndarray
    qc: Mapping[str, Any]
    provenance: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeometryVisualizationError(
            f"Could not read artifact manifest {path}"
        ) from exc
    if not isinstance(value, Mapping):
        raise GeometryVisualizationError(
            f"Artifact manifest must contain a JSON object: {path}"
        )
    return dict(value)


def _verify_declared_file(
    path: Path,
    manifest: Mapping[str, Any],
) -> str:
    files = manifest.get("files")
    expected = files.get(path.name) if isinstance(files, Mapping) else None
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise GeometryVisualizationError(
            f"Manifest does not declare a valid checksum for {path.name}"
        )
    observed = _sha256_file(path)
    if observed != expected:
        raise GeometryVisualizationError(
            f"Input checksum mismatch for {path.name}"
        )
    return observed


def _load_input_manifests(
    prepared_path: Path,
    graph_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    prepared_manifest_path = prepared_path.parent / "manifest.json"
    graph_manifest_path = graph_path.parent / "manifest.json"
    if not prepared_manifest_path.is_file():
        raise GeometryVisualizationError(
            "prepared_data.npz must have a sibling manifest.json"
        )
    if not graph_manifest_path.is_file():
        raise GeometryVisualizationError(
            "graph-grid NPZ must have a sibling manifest.json"
        )
    prepared_manifest = _read_json(prepared_manifest_path)
    graph_manifest = _read_json(graph_manifest_path)
    if prepared_manifest.get("artifact_kind") != PREPARED_ARTIFACT_KIND:
        raise GeometryVisualizationError(
            "Unsupported prepared artifact kind"
        )
    if graph_manifest.get("artifact_kind") != GRAPH_GRID_ARTIFACT_KIND:
        raise GeometryVisualizationError(
            "Unsupported graph-grid artifact kind"
        )
    if graph_manifest.get("test_expression_targets_evaluated") is not False:
        raise GeometryVisualizationError(
            "Graph-grid manifest does not certify geometry-only construction"
        )
    prepared_id = prepared_manifest.get("artifact_id")
    if (
        not isinstance(prepared_id, str)
        or graph_manifest.get("prepared_artifact_id") != prepared_id
    ):
        raise GeometryVisualizationError(
            "Prepared and graph-grid artifact IDs do not match"
        )
    prepared_split = prepared_manifest.get("split")
    split_id = (
        prepared_split.get("split_id")
        if isinstance(prepared_split, Mapping)
        else None
    )
    if (
        not isinstance(split_id, str)
        or graph_manifest.get("split_id") != split_id
    ):
        raise GeometryVisualizationError(
            "Prepared and graph-grid split IDs do not match"
        )
    checksums = {
        "prepared_data.npz": _verify_declared_file(
            prepared_path,
            prepared_manifest,
        ),
        graph_path.name: _verify_declared_file(
            graph_path,
            graph_manifest,
        ),
        "prepared_manifest.json": _sha256_file(prepared_manifest_path),
        "graph_grid_manifest.json": _sha256_file(graph_manifest_path),
    }
    return prepared_manifest, graph_manifest, checksums


def _normalise_splits(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise GeometryVisualizationError("split_labels must be one-dimensional")
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    normalised: list[str] = []
    for value in labels.tolist():
        key = str(value).strip().lower()
        if key not in aliases:
            raise GeometryVisualizationError(
                f"Unsupported spatial split label {value!r}"
            )
        normalised.append(aliases[key])
    result = np.asarray(normalised, dtype="U10")
    if set(np.unique(result)) != {"train", "validation", "test"}:
        raise GeometryVisualizationError(
            "Expected train, validation, and test spatial splits"
        )
    return result


def _encode_macroblocks(values: np.ndarray, n_nodes: int) -> tuple[np.ndarray, int]:
    labels = np.asarray(values)
    if labels.shape != (n_nodes,):
        raise GeometryVisualizationError(
            "macroblock_ids must align with coordinates"
        )
    anonymous = np.asarray([str(value) for value in labels.tolist()], dtype="U256")
    _, codes = np.unique(anonymous, return_inverse=True)
    return codes.astype(np.int64, copy=False), int(codes.max(initial=-1) + 1)


def _load_selected_arrays(
    prepared_path: Path,
    graph_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    try:
        with np.load(prepared_path, allow_pickle=False) as archive:
            missing = set(LOADED_PREPARED_ARRAY_KEYS).difference(archive.files)
            if missing:
                raise GeometryVisualizationError(
                    "Prepared NPZ lacks geometry arrays: "
                    + ", ".join(sorted(missing))
                )
            coordinates = np.asarray(archive["coordinates_um"], dtype=np.float64)
            split_labels = np.asarray(archive["split_labels"])
            macroblock_ids = np.asarray(archive["macroblock_ids"])
    except GeometryVisualizationError:
        raise
    except (OSError, ValueError) as exc:
        raise GeometryVisualizationError(
            f"Could not load prepared geometry from {prepared_path}"
        ) from exc
    try:
        with np.load(graph_path, allow_pickle=False) as archive:
            missing = set(LOADED_GRAPH_ARRAY_KEYS).difference(archive.files)
            if missing:
                raise GeometryVisualizationError(
                    "Graph NPZ lacks required arrays: "
                    + ", ".join(sorted(missing))
                )
            edge_index = np.asarray(archive["edge_index"], dtype=np.int64)
            edge_attributes = np.asarray(
                archive["edge_attributes_raw"],
                dtype=np.float64,
            )
            edge_names = np.asarray(archive["edge_attribute_names"]).astype(str)
    except GeometryVisualizationError:
        raise
    except (OSError, ValueError) as exc:
        raise GeometryVisualizationError(
            f"Could not load graph geometry from {graph_path}"
        ) from exc
    return (
        coordinates,
        split_labels,
        macroblock_ids,
        edge_index,
        edge_attributes,
        edge_names,
    )


def _graph_id_and_config(graph_path: Path) -> tuple[str, dict[str, Any]]:
    graph_id = graph_path.stem
    match = re.fullmatch(
        r"k(?P<k>\d+)_r(?P<radius>\d+(?:\.\d+)?)_"
        r"(?P<symmetry>union|mutual)",
        graph_id,
    )
    if match is None:
        return graph_id, {}
    return graph_id, {
        "k": int(match.group("k")),
        "radius_um": float(match.group("radius")),
        "symmetry": match.group("symmetry"),
    }


def _validate_and_summarise(
    coordinates: np.ndarray,
    split_labels: np.ndarray,
    edge_index: np.ndarray,
    edge_attributes: np.ndarray,
    edge_names: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) == 0
        or not np.isfinite(coordinates).all()
    ):
        raise GeometryVisualizationError(
            "coordinates_um must be finite with shape [nodes, 2]"
        )
    n_nodes = len(coordinates)
    splits = _normalise_splits(split_labels)
    if splits.shape != (n_nodes,):
        raise GeometryVisualizationError(
            "split_labels must align with coordinates"
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise GeometryVisualizationError(
            "edge_index must have shape [2, directed_edges]"
        )
    if edge_attributes.ndim != 2 or edge_attributes.shape[0] != edge_index.shape[1]:
        raise GeometryVisualizationError(
            "edge_attributes_raw must align with edge_index"
        )
    if edge_names.shape != (edge_attributes.shape[1],):
        raise GeometryVisualizationError(
            "edge_attribute_names do not match edge attributes"
        )
    if edge_index.size and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= n_nodes
    ):
        raise GeometryVisualizationError("edge_index contains invalid node indices")
    source, target = edge_index
    if edge_index.shape[1] == 0:
        raise GeometryVisualizationError(
            "Geometry graph must contain at least one edge"
        )
    if np.any(source == target):
        raise GeometryVisualizationError("Geometry graph contains self loops")
    cross_split_edges = int(np.count_nonzero(splits[source] != splits[target]))
    if cross_split_edges:
        raise GeometryVisualizationError(
            "Geometry graph contains cross-split edges"
        )
    directed_codes = source * np.int64(n_nodes) + target
    reverse_codes = target * np.int64(n_nodes) + source
    if not np.array_equal(np.sort(directed_codes), np.sort(reverse_codes)):
        raise GeometryVisualizationError(
            "Geometry graph is not directed-pair symmetric"
        )
    duplicate_directed_edges = int(
        len(directed_codes) - len(np.unique(directed_codes))
    )
    if duplicate_directed_edges:
        raise GeometryVisualizationError(
            "Geometry graph contains duplicate directed edges"
        )
    distance_matches = np.flatnonzero(edge_names == "distance_um")
    if distance_matches.size != 1:
        raise GeometryVisualizationError(
            "Graph must declare exactly one distance_um edge attribute"
        )
    distances = edge_attributes[:, int(distance_matches[0])]
    if distances.shape != (edge_index.shape[1],) or (
        not np.isfinite(distances).all()
    ):
        raise GeometryVisualizationError("Physical edge distances are invalid")
    computed_distances = np.linalg.norm(
        coordinates[source] - coordinates[target],
        axis=1,
    )
    maximum_distance_error = float(
        np.max(np.abs(distances - computed_distances), initial=0.0)
    )
    tolerance = max(
        1e-3,
        1e-5 * float(np.max(computed_distances, initial=0.0)),
    )
    if maximum_distance_error > tolerance:
        raise GeometryVisualizationError(
            "distance_um attributes do not match prepared coordinates"
        )
    undirected_mask = source < target
    undirected_source = source[undirected_mask]
    undirected_target = target[undirected_mask]
    undirected_distances = distances[undirected_mask]
    degree = np.bincount(source, minlength=n_nodes).astype(np.int64, copy=False)
    adjacency = coo_matrix(
        (
            np.ones(len(undirected_source), dtype=np.uint8),
            (undirected_source, undirected_target),
        ),
        shape=(n_nodes, n_nodes),
    )
    n_components, component_labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(
        component_labels,
        minlength=n_components,
    ).astype(np.int64, copy=False)
    qc = {
        "n_nodes": n_nodes,
        "n_directed_edges": int(edge_index.shape[1]),
        "n_undirected_edges": int(undirected_mask.sum()),
        "directed_edge_pairs_are_symmetric": True,
        "duplicate_directed_edges": duplicate_directed_edges,
        "self_loops": 0,
        "cross_split_edges": cross_split_edges,
        "n_isolated_nodes": int(np.count_nonzero(degree == 0)),
        "mean_degree": float(degree.mean()),
        "median_degree": float(np.median(degree)),
        "p95_degree": float(np.quantile(degree, 0.95)),
        "max_degree": int(degree.max(initial=0)),
        "edge_distance_mean_um": float(undirected_distances.mean()),
        "edge_distance_p50_um": float(np.median(undirected_distances)),
        "edge_distance_p95_um": float(
            np.quantile(undirected_distances, 0.95)
        ),
        "edge_distance_max_um": float(
            undirected_distances.max(initial=0.0)
        ),
        "edge_distance_attribute_max_error_um": maximum_distance_error,
        "n_components": int(n_components),
        "largest_component_nodes": int(component_sizes.max(initial=0)),
        "largest_component_fraction": float(component_sizes.max() / n_nodes),
    }
    return (
        splits,
        edge_index,
        distances,
        undirected_mask,
        degree,
        component_labels,
        component_sizes,
        qc,
    )


def load_geometry_data(
    prepared_data_path: str | os.PathLike[str],
    graph_path: str | os.PathLike[str],
) -> GeometryData:
    """Load and validate only geometry arrays from paired immutable artifacts."""

    prepared = Path(prepared_data_path).resolve()
    graph = Path(graph_path).resolve()
    if not prepared.is_file():
        raise FileNotFoundError(f"Prepared data NPZ was not found: {prepared}")
    if not graph.is_file():
        raise FileNotFoundError(f"Graph-grid NPZ was not found: {graph}")
    prepared_manifest, graph_manifest, checksums = _load_input_manifests(
        prepared,
        graph,
    )
    (
        coordinates,
        raw_splits,
        raw_macroblocks,
        edge_index,
        edge_attributes,
        edge_names,
    ) = _load_selected_arrays(prepared, graph)
    (
        splits,
        edge_index,
        distances,
        undirected_mask,
        degree,
        component_labels,
        component_sizes,
        qc,
    ) = _validate_and_summarise(
        coordinates,
        raw_splits,
        edge_index,
        edge_attributes,
        edge_names,
    )
    macroblock_codes, n_macroblocks = _encode_macroblocks(
        raw_macroblocks,
        len(coordinates),
    )
    declared_n_cells = graph_manifest.get("n_cells")
    if (
        not isinstance(declared_n_cells, int)
        or declared_n_cells != len(coordinates)
    ):
        raise GeometryVisualizationError(
            "Graph-grid node count does not match prepared coordinates"
        )
    graph_id, graph_config = _graph_id_and_config(graph)
    split_counts = {
        name: int(np.count_nonzero(splits == name))
        for name in ("train", "validation", "test")
    }
    provenance = {
        "prepared_artifact_id": prepared_manifest["artifact_id"],
        "graph_grid_artifact_id": graph_manifest.get("artifact_id"),
        "split_id": graph_manifest["split_id"],
        "graph_id": graph_id,
        "graph_config": graph_config,
        "input_files": {
            "prepared_data": {
                "name": prepared.name,
                "sha256": checksums["prepared_data.npz"],
            },
            "graph": {
                "name": graph.name,
                "sha256": checksums[graph.name],
            },
            "prepared_manifest": {
                "name": "manifest.json",
                "sha256": checksums["prepared_manifest.json"],
            },
            "graph_grid_manifest": {
                "name": "manifest.json",
                "sha256": checksums["graph_grid_manifest.json"],
            },
        },
        "loaded_prepared_array_keys": list(LOADED_PREPARED_ARRAY_KEYS),
        "loaded_graph_array_keys": list(LOADED_GRAPH_ARRAY_KEYS),
        "expression_arrays_accessed": False,
        "biological_or_sample_labels_emitted": False,
        "split_counts": split_counts,
    }
    return GeometryData(
        coordinates_um=np.ascontiguousarray(coordinates),
        split_labels=np.ascontiguousarray(splits),
        macroblock_codes=np.ascontiguousarray(macroblock_codes),
        n_macroblocks=n_macroblocks,
        edge_index=np.ascontiguousarray(edge_index),
        edge_distances_um=np.ascontiguousarray(distances),
        undirected_edge_mask=np.ascontiguousarray(undirected_mask),
        degree=np.ascontiguousarray(degree),
        component_labels=np.ascontiguousarray(component_labels),
        component_sizes=np.ascontiguousarray(component_sizes),
        qc=qc,
        provenance=provenance,
    )


def _candidate_axis(
    minimum: float,
    maximum: float,
    width: float,
    stride: float,
) -> np.ndarray:
    half = width / 2.0
    if maximum - minimum <= width:
        return np.asarray([(minimum + maximum) / 2.0], dtype=np.float64)
    start = minimum + half
    stop = maximum - half
    values = np.arange(start, stop + stride * 0.25, stride, dtype=np.float64)
    values = values[values <= stop + 1e-9]
    if values.size == 0 or values[-1] < stop - 1e-9:
        values = np.append(values, stop)
    return np.unique(values)


def select_dense_window(
    coordinates_um: np.ndarray,
    *,
    window_size_um: float = 200.0,
) -> DenseWindow:
    """Select the densest fixed window on a deterministic half-stride grid."""

    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) == 0
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError("coordinates_um must be finite with shape [nodes, 2]")
    width = float(window_size_um)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("window_size_um must be finite and positive")
    stride = width / 2.0
    x_values = _candidate_axis(
        float(coordinates[:, 0].min()),
        float(coordinates[:, 0].max()),
        width,
        stride,
    )
    y_values = _candidate_axis(
        float(coordinates[:, 1].min()),
        float(coordinates[:, 1].max()),
        width,
        stride,
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    centers = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    tree = cKDTree(coordinates)
    counts = np.asarray(
        tree.query_ball_point(
            centers,
            width / 2.0,
            p=np.inf,
            return_length=True,
        ),
        dtype=np.int64,
    )
    order = np.lexsort((centers[:, 0], centers[:, 1], -counts))
    center_x, center_y = centers[int(order[0])]
    half = width / 2.0
    x_min = float(center_x - half)
    x_max = float(center_x + half)
    y_min = float(center_y - half)
    y_max = float(center_y + half)
    node_indices = np.flatnonzero(
        (coordinates[:, 0] >= x_min)
        & (coordinates[:, 0] <= x_max)
        & (coordinates[:, 1] >= y_min)
        & (coordinates[:, 1] <= y_max)
    )
    return DenseWindow(
        x_min_um=x_min,
        x_max_um=x_max,
        y_min_um=y_min,
        y_max_um=y_max,
        center_x_um=float(center_x),
        center_y_um=float(center_y),
        width_um=width,
        stride_um=stride,
        n_nodes=int(len(node_indices)),
        candidate_count=int(len(centers)),
        node_indices=node_indices,
    )


def _macroblock_palette(n_blocks: int) -> np.ndarray:
    colors = [
        colorsys.hsv_to_rgb(
            (index * 0.6180339887498949) % 1.0,
            0.52,
            0.82,
        )
        for index in range(n_blocks)
    ]
    return np.asarray(colors, dtype=np.float64)


def _style_spatial_axis(axis: Any) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.set_xlabel("global x (mm)")
    axis.set_ylabel("global y (mm)")
    axis.set_facecolor("#F8FAFC")
    axis.grid(False)


def _render_figure(
    data: GeometryData,
    window: DenseWindow,
    png_path: Path,
    pdf_path: Path,
    *,
    dpi: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for geometry visualization"
        ) from exc
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    coordinates_mm = data.coordinates_um / 1000.0
    split_colors = {
        "train": "#0072B2",
        "validation": "#E69F00",
        "test": "#009E73",
    }
    split_labels = {
        "train": "Train",
        "validation": "Validation",
        "test": "Test",
    }
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 10.2))

    split_axis = axes[0, 0]
    for split in ("train", "validation", "test"):
        selected = data.split_labels == split
        split_axis.scatter(
            coordinates_mm[selected, 0],
            coordinates_mm[selected, 1],
            s=2.2,
            c=split_colors[split],
            label=f"{split_labels[split]} (n={int(selected.sum()):,})",
            linewidths=0,
            alpha=0.82,
            rasterized=True,
        )
    split_axis.set_title("A  Spatial split allocation", loc="left", weight="bold")
    split_axis.legend(
        frameon=False,
        markerscale=3,
        loc="upper right",
        handletextpad=0.3,
    )
    _style_spatial_axis(split_axis)

    block_axis = axes[0, 1]
    palette = _macroblock_palette(data.n_macroblocks)
    block_axis.scatter(
        coordinates_mm[:, 0],
        coordinates_mm[:, 1],
        s=2.2,
        c=palette[data.macroblock_codes],
        linewidths=0,
        alpha=0.88,
        rasterized=True,
    )
    block_axis.set_title(
        "B  Macroblock tiling",
        loc="left",
        weight="bold",
    )
    block_axis.text(
        0.02,
        0.98,
        f"{data.n_macroblocks:,} spatial blocks; identifiers suppressed",
        transform=block_axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
    )
    _style_spatial_axis(block_axis)

    degree_axis = axes[0, 2]
    degree_cap = max(1.0, float(np.quantile(data.degree, 0.99)))
    degree_scatter = degree_axis.scatter(
        coordinates_mm[:, 0],
        coordinates_mm[:, 1],
        s=2.4,
        c=data.degree,
        cmap="viridis",
        vmin=0,
        vmax=degree_cap,
        linewidths=0,
        rasterized=True,
    )
    degree_axis.set_title(
        "C  Local graph degree",
        loc="left",
        weight="bold",
    )
    colorbar = figure.colorbar(
        degree_scatter,
        ax=degree_axis,
        fraction=0.046,
        pad=0.02,
    )
    colorbar.set_label("undirected degree (color capped at p99)")
    _style_spatial_axis(degree_axis)

    local_axis = axes[1, 0]
    local_nodes = np.zeros(len(data.coordinates_um), dtype=bool)
    local_nodes[window.node_indices] = True
    source, target = data.edge_index
    local_edge_mask = (
        data.undirected_edge_mask
        & local_nodes[source]
        & local_nodes[target]
    )
    local_source = source[local_edge_mask]
    local_target = target[local_edge_mask]
    origin = np.asarray([window.x_min_um, window.y_min_um])
    local_coordinates = data.coordinates_um - origin
    segments = np.stack(
        (
            local_coordinates[local_source],
            local_coordinates[local_target],
        ),
        axis=1,
    )
    collection = LineCollection(
        segments,
        colors="#64748B",
        linewidths=0.36,
        alpha=0.20,
        rasterized=True,
        zorder=1,
    )
    local_axis.add_collection(collection)
    for split in ("train", "validation", "test"):
        selected = local_nodes & (data.split_labels == split)
        if np.any(selected):
            local_axis.scatter(
                local_coordinates[selected, 0],
                local_coordinates[selected, 1],
                s=8,
                c=split_colors[split],
                label=split_labels[split],
                linewidths=0,
                alpha=0.90,
                rasterized=True,
                zorder=2,
            )
    local_axis.set_xlim(0, window.width_um)
    local_axis.set_ylim(0, window.width_um)
    local_axis.set_aspect("equal", adjustable="box")
    local_axis.invert_yaxis()
    local_axis.set_xlabel("local x (µm)")
    local_axis.set_ylabel("local y (µm)")
    local_axis.set_facecolor("#F8FAFC")
    local_axis.set_title(
        f"D  Densest fixed {window.width_um:g} µm window",
        loc="left",
        weight="bold",
    )
    local_axis.text(
        0.02,
        0.98,
        (
            f"{window.n_nodes:,} nodes; "
            f"{int(local_edge_mask.sum()):,} undirected edges"
        ),
        transform=local_axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80},
        zorder=3,
    )
    local_axis.legend(
        frameon=False,
        loc="lower right",
        markerscale=1.4,
    )
    scale_length = min(50.0, window.width_um / 3.0)
    scale_x = window.width_um * 0.06
    scale_y = window.width_um * 0.92
    local_axis.plot(
        [scale_x, scale_x + scale_length],
        [scale_y, scale_y],
        color="#111827",
        linewidth=2.2,
        solid_capstyle="butt",
        zorder=4,
    )
    local_axis.text(
        scale_x + scale_length / 2.0,
        scale_y - window.width_um * 0.025,
        f"{scale_length:g} µm",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=4,
    )

    degree_hist_axis = axes[1, 1]
    maximum_degree = int(data.degree.max(initial=0))
    degree_bins = np.arange(-0.5, maximum_degree + 1.5)
    degree_hist_axis.hist(
        data.degree,
        bins=degree_bins,
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.25,
    )
    degree_hist_axis.axvline(
        data.qc["median_degree"],
        color="#D55E00",
        linewidth=1.6,
        label=f"median {data.qc['median_degree']:.1f}",
    )
    degree_hist_axis.axvline(
        data.qc["p95_degree"],
        color="#009E73",
        linewidth=1.6,
        linestyle="--",
        label=f"p95 {data.qc['p95_degree']:.1f}",
    )
    degree_hist_axis.set_title(
        "E  Degree distribution",
        loc="left",
        weight="bold",
    )
    degree_hist_axis.set_xlabel("undirected degree")
    degree_hist_axis.set_ylabel("nodes")
    degree_hist_axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.6)
    degree_hist_axis.legend(frameon=False)
    degree_hist_axis.text(
        0.98,
        0.95,
        (
            f"mean {data.qc['mean_degree']:.1f}\n"
            f"max {data.qc['max_degree']:,}\n"
            f"isolated {data.qc['n_isolated_nodes']:,}"
        ),
        transform=degree_hist_axis.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        color="#374151",
    )

    distance_axis = axes[1, 2]
    undirected_distances = data.edge_distances_um[data.undirected_edge_mask]
    weights = np.full(
        len(undirected_distances),
        100.0 / len(undirected_distances),
    )
    distance_axis.hist(
        undirected_distances,
        bins=40,
        weights=weights,
        color="#59A14F",
        edgecolor="white",
        linewidth=0.25,
    )
    distance_axis.axvline(
        data.qc["edge_distance_p50_um"],
        color="#D55E00",
        linewidth=1.6,
        label=f"median {data.qc['edge_distance_p50_um']:.1f} µm",
    )
    distance_axis.axvline(
        data.qc["edge_distance_p95_um"],
        color="#0072B2",
        linewidth=1.6,
        linestyle="--",
        label=f"p95 {data.qc['edge_distance_p95_um']:.1f} µm",
    )
    distance_axis.set_title(
        "F  Edge scale and components",
        loc="left",
        weight="bold",
    )
    distance_axis.set_xlabel("physical edge distance (µm)")
    distance_axis.set_ylabel("undirected edges (%)")
    distance_axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.6)
    distance_axis.legend(frameon=False, loc="upper left")
    component_axis = distance_axis.inset_axes([0.58, 0.52, 0.38, 0.38])
    sorted_sizes = np.sort(data.component_sizes)[::-1]
    shown_sizes = sorted_sizes[: min(8, len(sorted_sizes))]
    component_axis.bar(
        np.arange(len(shown_sizes)),
        shown_sizes / len(data.coordinates_um) * 100.0,
        color="#B279A2",
        width=0.75,
    )
    component_axis.set_title("Largest components", fontsize=8)
    component_axis.set_ylabel("nodes (%)", fontsize=7)
    component_axis.set_xticks([])
    component_axis.tick_params(axis="y", labelsize=7)
    component_axis.spines["top"].set_visible(False)
    component_axis.spines["right"].set_visible(False)
    distance_axis.text(
        0.98,
        0.05,
        (
            f"{data.qc['n_components']:,} components; "
            f"largest {data.qc['largest_component_fraction'] * 100:.1f}%\n"
            f"cross-split edges {data.qc['cross_split_edges']}"
        ),
        transform=distance_axis.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
        color="#374151",
    )

    graph_id = data.provenance["graph_id"]
    figure.suptitle(
        f"Geometry-only spatial graph audit — {graph_id}",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        (
            "Coordinates, split/macroblock geometry, topology, and physical "
            "edge distances only; no expression values or outcome labels loaded."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.98,
        bottom=0.075,
        top=0.935,
        wspace=0.27,
        hspace=0.28,
    )
    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "spatial_benchmark.geometry_viz"},
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "spatial_benchmark.geometry_viz",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def _canonical_artifact_id(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def render_geometry_report(
    prepared_data_path: str | os.PathLike[str],
    graph_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    window_size_um: float = 200.0,
    dpi: int = 300,
) -> Path:
    """Atomically render a geometry-only PNG/PDF and checksum manifest."""

    destination = Path(output_dir).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite geometry visualization: {destination}"
        )
    if int(dpi) < 150:
        raise ValueError("dpi must be at least 150")
    data = load_geometry_data(prepared_data_path, graph_path)
    window = select_dense_window(
        data.coordinates_um,
        window_size_um=window_size_um,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        png_name = "geometry_panels.png"
        pdf_name = "geometry_panels.pdf"
        _render_figure(
            data,
            window,
            temporary / png_name,
            temporary / pdf_name,
            dpi=int(dpi),
        )
        output_files = {
            png_name: _sha256_file(temporary / png_name),
            pdf_name: _sha256_file(temporary / pdf_name),
        }
        manifest: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "artifact_kind": ARTIFACT_KIND,
            "geometry_only": True,
            "scope": (
                "Descriptive geometry and graph-QC visualization; no expression "
                "values, outcome labels, or biological/sample identifiers."
            ),
            "provenance": dict(data.provenance),
            "dense_window": window.to_manifest(),
            "geometry_summary": {
                "n_macroblocks": data.n_macroblocks,
                **dict(data.qc),
            },
            "rendering": {
                "dpi": int(dpi),
                "panels": [
                    "spatial_split",
                    "macroblock_tiling",
                    "node_degree_map",
                    "dense_window_edge_overlay",
                    "degree_distribution",
                    "edge_distance_and_components",
                ],
            },
            "files": output_files,
        }
        manifest["artifact_id"] = _canonical_artifact_id(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(
                manifest,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "ARTIFACT_KIND",
    "DenseWindow",
    "GeometryData",
    "GeometryVisualizationError",
    "load_geometry_data",
    "render_geometry_report",
    "select_dense_window",
]
