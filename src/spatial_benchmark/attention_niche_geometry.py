"""Deterministic spatial geometry for attention-routing niche analyses.

The utilities in this module are deliberately independent of model inference.
They operate on prepared-node order and physical coordinates in micrometres,
and never refit or otherwise change a trained model.  Raw CosMx polygon rows
are streamed and joined through the complete ``(slide, fov, cell_ID)`` key;
FOV or row order alone is never treated as cell identity.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import colorsys
import csv
import hashlib
import heapq
import math
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import Delaunay, QhullError, cKDTree
from shapely import make_valid, normalize
from shapely.geometry import (
    GeometryCollection,
    MultiPolygon,
    Point,
    Polygon,
    mapping as geometry_mapping,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from spatial_benchmark.data import (
    CELL_KEY_COLUMNS,
    DEFAULT_PIXEL_SIZE_UM,
    DataContractError,
    normalize_slide,
)


DEFAULT_LOCAL_MAX_GAP_UM = 75.0
DEFAULT_LOCAL_K = 6
DEFAULT_POLYGON_ADJACENCY_COVERAGE = 0.90
DEFAULT_MICRO_NICHE_THRESHOLD = 20
NEUTRAL_NICHE_COLOR = "#9a9a9a"

CellKey: TypeAlias = tuple[str, int, int]


class AttentionNicheGeometryError(ValueError):
    """Raised when spatial inputs violate the niche-analysis contract."""


@dataclass(frozen=True)
class LocalContiguityGraph:
    """One canonical, unweighted, undirected local spatial graph.

    ``adjacency`` is a symmetric CSR matrix with a zero diagonal.  The graph
    may legitimately contain multiple components or isolated cells: a local
    maximum gap must not be relaxed merely to bridge empty tissue.
    """

    adjacency: sparse.csr_matrix
    method: str
    max_gap_um: float
    edge_count: int
    nonisolated_fraction: float
    polygon_nonisolated_fraction: float | None
    polygon_edge_count: int | None
    fallback_reason: str | None

    @property
    def n_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def edge_pairs(self) -> np.ndarray:
        """Return lexicographically sorted unique ``i < j`` node pairs."""

        rows, columns = sparse.triu(self.adjacency, k=1).nonzero()
        if len(rows) == 0:
            return np.empty((0, 2), dtype=np.int64)
        pairs = np.column_stack((rows, columns)).astype(np.int64, copy=False)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        return pairs[order]


@dataclass(frozen=True)
class NicheSplitResult:
    """Deterministic connected-component split of preliminary communities."""

    core_number: int
    final_niche_ids: np.ndarray
    micro_niche: np.ndarray
    niche_members: Mapping[str, tuple[int, ...]]
    preliminary_label_by_niche: Mapping[str, Any]
    micro_niche_threshold: int

    @property
    def niche_count(self) -> int:
        return len(self.niche_members)


@dataclass(frozen=True)
class CentroidAlignmentReport:
    """Polygon/key alignment audit against prepared cell coordinates.

    A vendor cell coordinate is not necessarily the geometric centroid of an
    irregular segmentation polygon.  Alignment therefore passes when the
    coordinate is within the declared centroid tolerance *or* the keyed
    polygon covers the coordinate.  The raw centroid-distance exceptions are
    retained explicitly so containment does not hide the audit evidence.
    """

    distances_um: np.ndarray
    tolerance_um: float
    mismatch_indices: tuple[int, ...]
    median_distance_um: float
    maximum_distance_um: float
    centroid_tolerance_exceeded_indices: tuple[int, ...] = ()
    containment_accepted_indices: tuple[int, ...] = ()
    coordinate_outside_polygon_indices: tuple[int, ...] = ()
    alignment_rule: str = "centroid_tolerance_or_polygon_covers_coordinate"

    @property
    def aligned(self) -> bool:
        return not self.mismatch_indices


@dataclass(frozen=True)
class PolygonAlignment:
    """Raw segmentation polygons aligned exactly to prepared node order."""

    keys: tuple[CellKey, ...]
    polygons_um: tuple[BaseGeometry, ...]
    vertex_counts: np.ndarray
    repaired: np.ndarray
    source_rows_scanned: int
    selected_vertex_rows: int
    pixel_size_um: float
    centroid_alignment: CentroidAlignmentReport | None

    @property
    def n_cells(self) -> int:
        return len(self.keys)


@dataclass(frozen=True)
class DissolvedNicheRegion:
    """Dissolved polygon geometry for one core-scoped niche."""

    niche_id: str
    geometry: BaseGeometry
    cell_count: int
    area_um2: float


def _validate_coordinates_um(coordinates_um: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise AttentionNicheGeometryError(
            "Physical coordinates must have shape [n_cells, 2]."
        )
    if not np.isfinite(coordinates).all():
        raise AttentionNicheGeometryError(
            "Physical coordinates in micrometres must be finite."
        )
    return coordinates


def validate_undirected_adjacency(
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    n_nodes: int | None = None,
) -> sparse.csr_matrix:
    """Validate and return a canonical boolean undirected CSR adjacency.

    The function does not silently symmetrise a directed graph.  Such a repair
    could incorrectly convert a one-directional relation into local spatial
    adjacency and would hide an upstream alignment error.
    """

    matrix = sparse.csr_matrix(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise AttentionNicheGeometryError("Adjacency must be a square matrix.")
    if n_nodes is not None and matrix.shape != (int(n_nodes), int(n_nodes)):
        raise AttentionNicheGeometryError(
            "Adjacency shape does not match the prepared node count."
        )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or np.any(matrix.data <= 0)
    ):
        raise AttentionNicheGeometryError(
            "Adjacency entries must be finite and strictly positive."
        )
    if np.any(matrix.diagonal() != 0):
        raise AttentionNicheGeometryError("Local adjacency cannot contain self-edges.")
    binary = matrix.astype(bool).astype(np.uint8).tocsr()
    if (binary != binary.T).nnz:
        raise AttentionNicheGeometryError("Local adjacency must be undirected/symmetric.")
    binary.sort_indices()
    return binary


def _adjacency_from_pairs(n_nodes: int, pairs: Iterable[tuple[int, int]]) -> sparse.csr_matrix:
    canonical: set[tuple[int, int]] = set()
    for raw_i, raw_j in pairs:
        i = int(raw_i)
        j = int(raw_j)
        if i == j:
            continue
        if i < 0 or j < 0 or i >= n_nodes or j >= n_nodes:
            raise AttentionNicheGeometryError("An adjacency edge has an invalid node index.")
        canonical.add((min(i, j), max(i, j)))
    if not canonical:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.uint8)
    ordered = np.asarray(sorted(canonical), dtype=np.int64)
    rows = np.concatenate((ordered[:, 0], ordered[:, 1]))
    columns = np.concatenate((ordered[:, 1], ordered[:, 0]))
    values = np.ones(len(rows), dtype=np.uint8)
    matrix = sparse.csr_matrix((values, (rows, columns)), shape=(n_nodes, n_nodes))
    return validate_undirected_adjacency(matrix, n_nodes=n_nodes)


def _query_tree_indices(
    tree: STRtree,
    query_geometry: BaseGeometry,
    geometries: Sequence[BaseGeometry],
) -> tuple[int, ...]:
    """Return STRtree query indices on Shapely 2 and legacy Shapely 1."""

    result = tree.query(query_geometry)
    values = np.asarray(result)
    if values.size == 0:
        return ()
    if np.issubdtype(values.dtype, np.integer):
        return tuple(int(value) for value in values.tolist())
    by_identity: dict[int, list[int]] = defaultdict(list)
    for index, geometry in enumerate(geometries):
        by_identity[id(geometry)].append(index)
    indices: list[int] = []
    for geometry in result:
        candidates = by_identity.get(id(geometry), [])
        if not candidates:
            raise AttentionNicheGeometryError(
                "The spatial index returned an unaligned polygon object."
            )
        indices.extend(candidates)
    return tuple(indices)


def _validate_polygonal_geometry(
    geometry: BaseGeometry | None,
    *,
    position: int,
) -> BaseGeometry:
    if geometry is None or geometry.is_empty:
        raise AttentionNicheGeometryError(
            f"Cell polygon {position} is missing or empty."
        )
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise AttentionNicheGeometryError(
            f"Cell polygon {position} is not Polygon/MultiPolygon geometry."
        )
    if not geometry.is_valid:
        raise AttentionNicheGeometryError(f"Cell polygon {position} is invalid.")
    return geometry


def segmentation_polygon_adjacency(
    polygons_um: Sequence[BaseGeometry | None],
    *,
    touch_tolerance_um: float = 0.0,
) -> sparse.csr_matrix:
    """Construct exact/tolerance-capped segmentation-polygon adjacency.

    Polygon pairs are adjacent when their physical separation is at most the
    declared tolerance.  A zero tolerance therefore accepts only touching or
    intersecting segmentation geometries and cannot bridge an empty gap.
    """

    tolerance = float(touch_tolerance_um)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise AttentionNicheGeometryError(
            "Polygon touch tolerance must be finite and nonnegative."
        )
    geometries = tuple(
        _validate_polygonal_geometry(geometry, position=index)
        for index, geometry in enumerate(polygons_um)
    )
    n_nodes = len(geometries)
    if n_nodes <= 1:
        return _adjacency_from_pairs(n_nodes, ())
    tree = STRtree(geometries)
    pairs: list[tuple[int, int]] = []
    for i, geometry in enumerate(geometries):
        query_geometry = geometry if tolerance == 0 else geometry.buffer(tolerance)
        for j in sorted(set(_query_tree_indices(tree, query_geometry, geometries))):
            if j <= i:
                continue
            if float(geometry.distance(geometries[j])) <= tolerance:
                pairs.append((i, j))
    return _adjacency_from_pairs(n_nodes, pairs)


def _delaunay_adjacency(coordinates_um: np.ndarray, max_gap_um: float) -> sparse.csr_matrix:
    n_nodes = len(coordinates_um)
    if n_nodes < 3:
        raise QhullError("Delaunay requires at least three points.")
    if len(np.unique(coordinates_um, axis=0)) != n_nodes:
        raise QhullError("Delaunay is ambiguous for duplicate cell centroids.")
    if np.linalg.matrix_rank(coordinates_um - coordinates_um[0]) < 2:
        raise QhullError("Delaunay is undefined for collinear cell centroids.")
    triangulation = Delaunay(coordinates_um)
    candidate_pairs: set[tuple[int, int]] = set()
    for simplex in np.asarray(triangulation.simplices, dtype=np.int64):
        ordered = sorted(int(value) for value in simplex.tolist())
        for offset, i in enumerate(ordered):
            for j in ordered[offset + 1 :]:
                candidate_pairs.add((i, j))
    retained = []
    for i, j in sorted(candidate_pairs):
        distance = float(np.linalg.norm(coordinates_um[i] - coordinates_um[j]))
        if distance <= max_gap_um:
            retained.append((i, j))
    return _adjacency_from_pairs(n_nodes, retained)


def _knn_radius_adjacency(
    coordinates_um: np.ndarray,
    *,
    k: int,
    max_radius_um: float,
) -> sparse.csr_matrix:
    n_nodes = len(coordinates_um)
    if n_nodes <= 1:
        return _adjacency_from_pairs(n_nodes, ())
    tree = cKDTree(coordinates_um)
    pairs: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        candidates = tree.query_ball_point(coordinates_um[i], r=max_radius_um)
        ranked = sorted(
            (
                (float(np.linalg.norm(coordinates_um[i] - coordinates_um[j])), int(j))
                for j in candidates
                if int(j) != i
            ),
            key=lambda item: (item[0], item[1]),
        )
        for _, j in ranked[:k]:
            pairs.add((min(i, j), max(i, j)))
    return _adjacency_from_pairs(n_nodes, sorted(pairs))


def _nonisolated_fraction(adjacency: sparse.csr_matrix) -> float:
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return 1.0
    if n_nodes == 1:
        return 1.0
    degree = np.diff(adjacency.indptr)
    return float(np.mean(degree > 0))


def construct_local_contiguity_graph(
    coordinates_um: np.ndarray | Sequence[Sequence[float]],
    *,
    polygons_um: Sequence[BaseGeometry | None] | None = None,
    polygon_touch_tolerance_um: float = 0.0,
    minimum_polygon_nonisolated_fraction: float = DEFAULT_POLYGON_ADJACENCY_COVERAGE,
    max_gap_um: float = DEFAULT_LOCAL_MAX_GAP_UM,
    fallback_k: int = DEFAULT_LOCAL_K,
) -> LocalContiguityGraph:
    """Build the locked polygon -> Delaunay -> local-kNN contiguity graph.

    Polygon adjacency is selected only when every polygon is usable and the
    declared fraction of cells has at least one polygon-adjacent neighbour.
    Otherwise Delaunay adjacency is attempted and every Delaunay edge is capped
    at ``max_gap_um``.  Local ``k``-nearest-neighbour adjacency with the same
    radius is used only when Delaunay is geometrically undefined or empty.
    """

    coordinates = _validate_coordinates_um(coordinates_um)
    n_nodes = len(coordinates)
    gap = float(max_gap_um)
    min_coverage = float(minimum_polygon_nonisolated_fraction)
    if not np.isfinite(gap) or gap <= 0:
        raise AttentionNicheGeometryError("Local maximum gap must be positive and finite.")
    if not 0 <= min_coverage <= 1:
        raise AttentionNicheGeometryError(
            "Minimum polygon nonisolated fraction must lie in [0, 1]."
        )
    if int(fallback_k) != fallback_k or int(fallback_k) <= 0:
        raise AttentionNicheGeometryError("Fallback k must be a positive integer.")

    polygon_fraction: float | None = None
    polygon_edge_count: int | None = None
    fallback_reason: str | None = None
    if polygons_um is not None:
        if len(polygons_um) != n_nodes:
            raise AttentionNicheGeometryError(
                "Polygon count does not match the prepared node count."
            )
        try:
            polygon_adjacency = segmentation_polygon_adjacency(
                polygons_um, touch_tolerance_um=polygon_touch_tolerance_um
            )
        except AttentionNicheGeometryError as exc:
            fallback_reason = f"segmentation polygons unusable: {exc}"
        else:
            polygon_fraction = _nonisolated_fraction(polygon_adjacency)
            polygon_edge_count = int(polygon_adjacency.nnz // 2)
            usable = n_nodes <= 1 or (
                polygon_edge_count > 0 and polygon_fraction >= min_coverage
            )
            if usable:
                return LocalContiguityGraph(
                    adjacency=polygon_adjacency,
                    method="segmentation_polygon_adjacency",
                    max_gap_um=gap,
                    edge_count=polygon_edge_count,
                    nonisolated_fraction=polygon_fraction,
                    polygon_nonisolated_fraction=polygon_fraction,
                    polygon_edge_count=polygon_edge_count,
                    fallback_reason=None,
                )
            fallback_reason = (
                "segmentation polygon adjacency coverage below the declared "
                f"threshold ({polygon_fraction:.6f} < {min_coverage:.6f})"
            )
    else:
        fallback_reason = "segmentation polygons unavailable"

    try:
        delaunay = _delaunay_adjacency(coordinates, gap)
        if n_nodes > 1 and delaunay.nnz == 0:
            raise QhullError("No Delaunay edge survived the local maximum gap.")
    except QhullError as exc:
        reason_prefix = fallback_reason or "segmentation polygon adjacency not selected"
        fallback_reason = f"{reason_prefix}; Delaunay unavailable: {exc}"
        adjacency = _knn_radius_adjacency(
            coordinates, k=int(fallback_k), max_radius_um=gap
        )
        method = f"knn_k{int(fallback_k)}_radius"
    else:
        adjacency = delaunay
        method = "delaunay_max_gap"

    return LocalContiguityGraph(
        adjacency=adjacency,
        method=method,
        max_gap_um=gap,
        edge_count=int(adjacency.nnz // 2),
        nonisolated_fraction=_nonisolated_fraction(adjacency),
        polygon_nonisolated_fraction=polygon_fraction,
        polygon_edge_count=polygon_edge_count,
        fallback_reason=fallback_reason,
    )


def _label_is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.ndim(missing) == 0 else False


def split_preliminary_niches(
    core_number: int,
    preliminary_labels: Sequence[Any] | np.ndarray,
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    micro_niche_threshold: int = DEFAULT_MICRO_NICHE_THRESHOLD,
) -> NicheSplitResult:
    """Split each preliminary community into local connected components.

    Final IDs are ordered by each component's smallest prepared-node index,
    making the IDs independent of arbitrary Leiden label numbering.  Components
    smaller than the threshold remain separate and are marked as micro-niches.
    """

    core = int(core_number)
    if core <= 0 or core > 99 or core != core_number:
        raise AttentionNicheGeometryError(
            "Core number must be an integer from 1 through 99."
        )
    threshold = int(micro_niche_threshold)
    if threshold <= 0 or threshold != micro_niche_threshold:
        raise AttentionNicheGeometryError(
            "Micro-niche threshold must be a positive integer."
        )
    labels = np.asarray(preliminary_labels, dtype=object)
    if labels.ndim != 1:
        raise AttentionNicheGeometryError("Preliminary labels must be one-dimensional.")
    graph = validate_undirected_adjacency(adjacency, n_nodes=len(labels))
    if any(_label_is_missing(value) for value in labels.tolist()):
        raise AttentionNicheGeometryError("Preliminary labels cannot be missing.")

    members_by_label: dict[Any, list[int]] = defaultdict(list)
    try:
        for index, label in enumerate(labels.tolist()):
            members_by_label[label].append(index)
    except TypeError as exc:
        raise AttentionNicheGeometryError(
            "Preliminary community labels must be scalar/hashable values."
        ) from exc

    components: list[tuple[tuple[int, ...], Any]] = []
    for label, indices_list in members_by_label.items():
        indices = np.asarray(sorted(indices_list), dtype=np.int64)
        induced = graph[indices][:, indices]
        n_components, component_labels = csgraph.connected_components(
            induced, directed=False, return_labels=True
        )
        for component in range(int(n_components)):
            selected = indices[component_labels == component]
            members = tuple(int(value) for value in np.sort(selected).tolist())
            components.append((members, label))
    components.sort(key=lambda item: (item[0][0], item[0]))

    final_ids = np.empty(len(labels), dtype=object)
    micro = np.zeros(len(labels), dtype=bool)
    niche_members: dict[str, tuple[int, ...]] = {}
    preliminary_by_niche: dict[str, Any] = {}
    for offset, (members, label) in enumerate(components, start=1):
        niche_id = f"C{core:02d}-N{offset:03d}"
        niche_members[niche_id] = members
        preliminary_by_niche[niche_id] = label
        is_micro = len(members) < threshold
        indices = np.asarray(members, dtype=np.int64)
        final_ids[indices] = niche_id
        micro[indices] = is_micro

    result = NicheSplitResult(
        core_number=core,
        final_niche_ids=final_ids,
        micro_niche=micro,
        niche_members=niche_members,
        preliminary_label_by_niche=preliminary_by_niche,
        micro_niche_threshold=threshold,
    )
    verify_niche_connectedness(result.final_niche_ids, graph)
    return result


def niche_connected_component_counts(
    niche_ids: Sequence[object] | np.ndarray,
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    ignored_labels: Iterable[object] = (None, "", "unassigned"),
) -> dict[str, int]:
    """Count induced local connected components for every assigned niche."""

    values = np.asarray(niche_ids, dtype=object)
    if values.ndim != 1:
        raise AttentionNicheGeometryError("Niche IDs must be one-dimensional.")
    graph = validate_undirected_adjacency(adjacency, n_nodes=len(values))
    ignored = set(ignored_labels)
    members: dict[str, list[int]] = defaultdict(list)
    for index, raw_label in enumerate(values.tolist()):
        if raw_label in ignored or _label_is_missing(raw_label):
            continue
        label = str(raw_label)
        if not label:
            continue
        members[label].append(index)
    counts: dict[str, int] = {}
    for label in sorted(members):
        indices = np.asarray(members[label], dtype=np.int64)
        induced = graph[indices][:, indices]
        count = csgraph.connected_components(
            induced, directed=False, return_labels=False
        )
        counts[label] = int(count)
    return counts


def verify_niche_connectedness(
    niche_ids: Sequence[object] | np.ndarray,
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    ignored_labels: Iterable[object] = (None, "", "unassigned"),
) -> dict[str, int]:
    """Raise unless every final niche induces exactly one local component."""

    counts = niche_connected_component_counts(
        niche_ids, adjacency, ignored_labels=ignored_labels
    )
    disconnected = {label: count for label, count in counts.items() if count != 1}
    if disconnected:
        details = ", ".join(
            f"{label}={count}" for label, count in sorted(disconnected.items())
        )
        raise AttentionNicheGeometryError(
            f"Final niche IDs span disconnected spatial components: {details}."
        )
    return counts


def _canonical_integer_series(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
        raise AttentionNicheGeometryError(f"{name} values must be finite integers.")
    return numeric.astype(np.int64)


def _prepared_cell_keys(prepared_keys: pd.DataFrame) -> tuple[CellKey, ...]:
    missing = set(CELL_KEY_COLUMNS).difference(prepared_keys.columns)
    if missing:
        raise AttentionNicheGeometryError(
            "Prepared keys must contain slide, fov, and cell_ID."
        )
    slides: list[str] = []
    try:
        slides = [normalize_slide(value) for value in prepared_keys["slide"].tolist()]
    except DataContractError as exc:
        raise AttentionNicheGeometryError("Prepared keys contain an invalid slide.") from exc
    fovs = _canonical_integer_series(prepared_keys["fov"], name="fov")
    cell_ids = _canonical_integer_series(prepared_keys["cell_ID"], name="cell_ID")
    keys = tuple(
        (slide, int(fov), int(cell_id))
        for slide, fov, cell_id in zip(slides, fovs, cell_ids, strict=True)
    )
    if len(set(keys)) != len(keys):
        raise AttentionNicheGeometryError(
            "Prepared node order contains duplicate slide-qualified cell keys."
        )
    return keys


def _normalise_polygon_sources(
    polygon_csvs: Mapping[object, str | Path] | Sequence[str | Path],
) -> dict[str, Path]:
    if isinstance(polygon_csvs, Mapping):
        raw_items = list(polygon_csvs.items())
    else:
        raw_items = []
        for raw_path in polygon_csvs:
            path = Path(raw_path)
            try:
                slide = normalize_slide(path.name)
            except DataContractError as exc:
                raise AttentionNicheGeometryError(
                    "Every polygon CSV filename must identify its source slide."
                ) from exc
            raw_items.append((slide, path))
    sources: dict[str, Path] = {}
    for raw_slide, raw_path in raw_items:
        try:
            slide = normalize_slide(raw_slide)
        except DataContractError as exc:
            raise AttentionNicheGeometryError(
                "Polygon source mapping contains an invalid slide."
            ) from exc
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Polygon CSV does not exist: {path}")
        try:
            filename_slide = normalize_slide(path.name)
        except DataContractError as exc:
            raise AttentionNicheGeometryError(
                "Polygon CSV filename does not contain an explicit slide token."
            ) from exc
        if filename_slide != slide:
            raise AttentionNicheGeometryError(
                "Polygon source mapping disagrees with the CSV filename slide."
            )
        if slide in sources:
            raise AttentionNicheGeometryError(
                f"More than one polygon CSV was supplied for {slide}."
            )
        sources[slide] = path
    if not sources:
        raise AttentionNicheGeometryError("At least one polygon CSV is required.")
    return sources


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        try:
            return next(csv.reader(handle))
        except StopIteration as exc:
            raise AttentionNicheGeometryError("A polygon CSV is empty.") from exc


def _polygon_columns(path: Path) -> tuple[str, str, str, str]:
    header = _csv_header(path)
    by_lower: dict[str, list[str]] = defaultdict(list)
    for column in header:
        by_lower[column.strip().lower()].append(column)

    def unique(aliases: tuple[str, ...], description: str) -> str:
        matches: list[str] = []
        for alias in aliases:
            matches.extend(by_lower.get(alias, []))
        matches = list(dict.fromkeys(matches))
        if len(matches) != 1:
            raise AttentionNicheGeometryError(
                f"Polygon CSV must contain exactly one {description} column."
            )
        return matches[0]

    return (
        unique(("fov",), "fov"),
        unique(("cellid", "cell_id"), "cell ID"),
        unique(("x_global_px",), "global-x"),
        unique(("y_global_px",), "global-y"),
    )


def _polygonal_part(geometry: BaseGeometry) -> BaseGeometry:
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygonal = [
            part
            for part in geometry.geoms
            if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
        ]
        if polygonal:
            return unary_union(polygonal)
    raise AttentionNicheGeometryError(
        "Repairing a raw cell boundary did not produce polygonal geometry."
    )


def _polygon_from_vertices(vertices_um: Sequence[tuple[float, float]]) -> tuple[BaseGeometry, bool]:
    coordinates = np.asarray(vertices_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise AttentionNicheGeometryError("Raw polygon vertices have invalid shape.")
    if len(coordinates) < 3 or len(np.unique(coordinates, axis=0)) < 3:
        raise AttentionNicheGeometryError(
            "A raw cell polygon has fewer than three distinct vertices."
        )
    if not np.isfinite(coordinates).all():
        raise AttentionNicheGeometryError("Raw polygon vertices must be finite.")
    geometry: BaseGeometry = Polygon(coordinates)
    repaired = False
    if not geometry.is_valid:
        geometry = _polygonal_part(make_valid(geometry))
        repaired = True
    geometry = normalize(geometry)
    if geometry.is_empty or not geometry.is_valid or not isinstance(
        geometry, (Polygon, MultiPolygon)
    ):
        raise AttentionNicheGeometryError(
            "A raw cell boundary could not be made into valid polygonal geometry."
        )
    return geometry, repaired


def validate_polygon_centroid_alignment(
    polygons_um: Sequence[BaseGeometry],
    coordinates_um: np.ndarray | Sequence[Sequence[float]],
    *,
    tolerance_um: float = 5.0,
    raise_on_mismatch: bool = True,
) -> CentroidAlignmentReport:
    """Audit keyed polygon alignment against prepared coordinates in µm.

    A coordinate that lies inside (or on the boundary of) its keyed polygon is
    accepted even when an elongated or irregular polygon's geometric centroid
    lies farther than ``tolerance_um`` away.  A row fails only when the
    coordinate is both outside its polygon and beyond the centroid tolerance.
    """

    coordinates = _validate_coordinates_um(coordinates_um)
    tolerance = float(tolerance_um)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise AttentionNicheGeometryError(
            "Centroid-alignment tolerance must be finite and nonnegative."
        )
    if len(polygons_um) != len(coordinates):
        raise AttentionNicheGeometryError(
            "Polygon and prepared-coordinate counts do not match."
        )
    polygon_centroids = np.empty_like(coordinates)
    coordinate_covered = np.empty(len(coordinates), dtype=np.bool_)
    for index, raw_geometry in enumerate(polygons_um):
        geometry = _validate_polygonal_geometry(raw_geometry, position=index)
        centroid = geometry.centroid
        polygon_centroids[index] = (float(centroid.x), float(centroid.y))
        coordinate_covered[index] = bool(
            geometry.covers(
                Point(float(coordinates[index, 0]), float(coordinates[index, 1]))
            )
        )
    distances = np.linalg.norm(polygon_centroids - coordinates, axis=1)
    tolerance_exceeded = distances > tolerance
    centroid_tolerance_exceeded_indices = tuple(
        int(value) for value in np.flatnonzero(tolerance_exceeded)
    )
    containment_accepted_indices = tuple(
        int(value)
        for value in np.flatnonzero(tolerance_exceeded & coordinate_covered)
    )
    coordinate_outside_polygon_indices = tuple(
        int(value) for value in np.flatnonzero(~coordinate_covered)
    )
    mismatch_indices = tuple(
        int(value)
        for value in np.flatnonzero(tolerance_exceeded & ~coordinate_covered)
    )
    report = CentroidAlignmentReport(
        distances_um=distances,
        tolerance_um=tolerance,
        mismatch_indices=mismatch_indices,
        median_distance_um=float(np.median(distances)) if len(distances) else 0.0,
        maximum_distance_um=float(np.max(distances)) if len(distances) else 0.0,
        centroid_tolerance_exceeded_indices=centroid_tolerance_exceeded_indices,
        containment_accepted_indices=containment_accepted_indices,
        coordinate_outside_polygon_indices=coordinate_outside_polygon_indices,
    )
    if raise_on_mismatch and mismatch_indices:
        raise AttentionNicheGeometryError(
            "Keyed polygons are not aligned to prepared coordinates in micrometres: "
            "each mismatch is outside its polygon and beyond the centroid tolerance "
            f"({len(mismatch_indices)} mismatches; max={report.maximum_distance_um:.6f} um; "
            f"tolerance={tolerance:.6f} um)."
        )
    return report


def load_aligned_cosmx_polygons(
    polygon_csvs: Mapping[object, str | Path] | Sequence[str | Path],
    prepared_keys: pd.DataFrame,
    *,
    prepared_coordinates_um: np.ndarray | Sequence[Sequence[float]] | None = None,
    pixel_size_um: float = DEFAULT_PIXEL_SIZE_UM,
    chunksize: int = 250_000,
    centroid_tolerance_um: float = 5.0,
) -> PolygonAlignment:
    """Stream raw CosMx vertices and align polygons to prepared node order.

    Only rows whose complete slide-qualified keys occur in ``prepared_keys``
    are retained.  The function reads each CSV in chunks and preserves source
    vertex order within every cell, including across chunk boundaries.
    """

    keys = _prepared_cell_keys(prepared_keys)
    sources = _normalise_polygon_sources(polygon_csvs)
    pixel_size = float(pixel_size_um)
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise AttentionNicheGeometryError("Pixel size must be positive and finite.")
    if int(chunksize) != chunksize or int(chunksize) <= 0:
        raise AttentionNicheGeometryError("Polygon CSV chunksize must be positive.")
    required_slides = {key[0] for key in keys}
    missing_sources = sorted(required_slides.difference(sources))
    if missing_sources:
        raise AttentionNicheGeometryError(
            "No polygon CSV was supplied for prepared slide(s): "
            + ", ".join(missing_sources)
        )

    index_by_key = {key: index for index, key in enumerate(keys)}
    vertices_by_index: dict[int, list[tuple[float, float]]] = defaultdict(list)
    rows_scanned = 0
    selected_rows = 0
    for slide in sorted(required_slides):
        path = sources[slide]
        fov_column, cell_column, x_column, y_column = _polygon_columns(path)
        wanted_fovs = {key[1] for key in keys if key[0] == slide}
        dtype: dict[str, object] = {
            fov_column: "int64",
            cell_column: "int64",
            x_column: "float64",
            y_column: "float64",
        }
        reader = pd.read_csv(
            path,
            usecols=[fov_column, cell_column, x_column, y_column],
            dtype=dtype,
            chunksize=int(chunksize),
        )
        for chunk in reader:
            rows_scanned += len(chunk)
            selected = chunk.loc[chunk[fov_column].isin(wanted_fovs)]
            values = selected[
                [fov_column, cell_column, x_column, y_column]
            ].to_numpy(copy=False)
            for raw_fov, raw_cell, raw_x, raw_y in values:
                key = (slide, int(raw_fov), int(raw_cell))
                index = index_by_key.get(key)
                if index is None:
                    continue
                x = float(raw_x) * pixel_size
                y = float(raw_y) * pixel_size
                if not math.isfinite(x) or not math.isfinite(y):
                    raise AttentionNicheGeometryError(
                        "A selected raw polygon vertex is non-finite."
                    )
                vertices_by_index[index].append((x, y))
                selected_rows += 1

    missing_indices = [index for index in range(len(keys)) if index not in vertices_by_index]
    if missing_indices:
        raise AttentionNicheGeometryError(
            "Raw polygon coverage is incomplete for prepared node order "
            f"({len(missing_indices)} missing cells)."
        )
    polygons: list[BaseGeometry] = []
    vertex_counts = np.empty(len(keys), dtype=np.int32)
    repaired = np.zeros(len(keys), dtype=bool)
    for index in range(len(keys)):
        vertices = vertices_by_index[index]
        geometry, was_repaired = _polygon_from_vertices(vertices)
        polygons.append(geometry)
        vertex_counts[index] = len(vertices)
        repaired[index] = was_repaired

    centroid_report = None
    if prepared_coordinates_um is not None:
        centroid_report = validate_polygon_centroid_alignment(
            polygons,
            prepared_coordinates_um,
            tolerance_um=centroid_tolerance_um,
            raise_on_mismatch=True,
        )
    return PolygonAlignment(
        keys=keys,
        polygons_um=tuple(polygons),
        vertex_counts=vertex_counts,
        repaired=repaired,
        source_rows_scanned=rows_scanned,
        selected_vertex_rows=selected_rows,
        pixel_size_um=pixel_size,
        centroid_alignment=centroid_report,
    )


def dissolve_niche_regions(
    polygons_um: Sequence[BaseGeometry],
    niche_ids: Sequence[object] | np.ndarray,
) -> tuple[DissolvedNicheRegion, ...]:
    """Dissolve cell polygons by final niche while retaining full topology."""

    labels = np.asarray(niche_ids, dtype=object)
    if labels.ndim != 1 or len(labels) != len(polygons_um):
        raise AttentionNicheGeometryError(
            "Polygon and final-niche arrays must be aligned one-to-one."
        )
    grouped: dict[str, list[BaseGeometry]] = defaultdict(list)
    for index, (raw_geometry, raw_label) in enumerate(
        zip(polygons_um, labels.tolist(), strict=True)
    ):
        if _label_is_missing(raw_label) or str(raw_label).strip() in {"", "unassigned"}:
            raise AttentionNicheGeometryError(
                "Every polygon must have one assigned final niche ID before dissolve."
            )
        geometry = _validate_polygonal_geometry(raw_geometry, position=index)
        grouped[str(raw_label)].append(geometry)

    regions: list[DissolvedNicheRegion] = []
    for niche_id in sorted(grouped):
        geometry = _polygonal_part(unary_union(grouped[niche_id]))
        if not geometry.is_valid:
            geometry = _polygonal_part(make_valid(geometry))
        geometry = normalize(geometry)
        regions.append(
            DissolvedNicheRegion(
                niche_id=niche_id,
                geometry=geometry,
                cell_count=len(grouped[niche_id]),
                area_um2=float(geometry.area),
            )
        )
    return tuple(regions)


def _rounded_geojson_value(value: Any, precision: int | None) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _rounded_geojson_value(item, precision)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_rounded_geojson_value(item, precision) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if precision is None else round(numeric, precision)
    return value


def dissolved_regions_geojson(
    regions: Sequence[DissolvedNicheRegion],
    *,
    core_number: int,
    colors: Mapping[str, str] | None = None,
    properties_by_niche: Mapping[str, Mapping[str, Any]] | None = None,
    coordinate_precision: int | None = 6,
) -> dict[str, Any]:
    """Emit a deterministic GeoJSON FeatureCollection in micrometres."""

    core = int(core_number)
    if core <= 0 or core > 99 or core != core_number:
        raise AttentionNicheGeometryError(
            "Core number must be an integer from 1 through 99."
        )
    if coordinate_precision is not None and coordinate_precision < 0:
        raise AttentionNicheGeometryError(
            "GeoJSON coordinate precision must be nonnegative or None."
        )
    extras = properties_by_niche or {}
    reserved = {
        "core_number",
        "final_niche_id",
        "cell_count",
        "area_um2",
        "niche_color",
        "coordinate_unit",
    }
    features: list[dict[str, Any]] = []
    observed: set[str] = set()
    for region in sorted(regions, key=lambda value: value.niche_id):
        if region.niche_id in observed:
            raise AttentionNicheGeometryError("Dissolved niche IDs must be unique.")
        observed.add(region.niche_id)
        extra = dict(extras.get(region.niche_id, {}))
        collision = reserved.intersection(extra)
        if collision:
            raise AttentionNicheGeometryError(
                "Extra GeoJSON properties cannot override reserved fields: "
                + ", ".join(sorted(collision))
            )
        properties: dict[str, Any] = {
            "core_number": core,
            "final_niche_id": region.niche_id,
            "cell_count": int(region.cell_count),
            "area_um2": float(region.area_um2),
            "coordinate_unit": "um",
        }
        if colors is not None:
            if region.niche_id not in colors:
                raise AttentionNicheGeometryError(
                    f"No deterministic color was supplied for {region.niche_id}."
                )
            properties["niche_color"] = str(colors[region.niche_id])
        for key in sorted(extra):
            properties[key] = extra[key]
        geometry = _rounded_geojson_value(
            geometry_mapping(normalize(region.geometry)), coordinate_precision
        )
        features.append(
            {
                "type": "Feature",
                "id": region.niche_id,
                "properties": properties,
                "geometry": geometry,
            }
        )
    return {
        "type": "FeatureCollection",
        "coordinate_unit": "um",
        "features": features,
    }


def niche_adjacency_from_cells(
    niche_ids: Sequence[object] | np.ndarray,
    cell_adjacency: sparse.spmatrix | np.ndarray,
) -> dict[str, tuple[str, ...]]:
    """Collapse a cell-level contiguity graph into deterministic niche adjacency."""

    labels = np.asarray(niche_ids, dtype=object)
    if labels.ndim != 1:
        raise AttentionNicheGeometryError("Niche IDs must be one-dimensional.")
    graph = validate_undirected_adjacency(cell_adjacency, n_nodes=len(labels))
    canonical = [str(value) for value in labels.tolist()]
    if any(_label_is_missing(value) or not label for value, label in zip(labels, canonical)):
        raise AttentionNicheGeometryError(
            "Every cell must have a nonempty niche ID for niche adjacency."
        )
    neighbours: dict[str, set[str]] = {
        label: set() for label in sorted(set(canonical))
    }
    rows, columns = sparse.triu(graph, k=1).nonzero()
    for i, j in zip(rows.tolist(), columns.tolist(), strict=True):
        first = canonical[int(i)]
        second = canonical[int(j)]
        if first == second:
            continue
        neighbours[first].add(second)
        neighbours[second].add(first)
    return {
        label: tuple(sorted(values)) for label, values in sorted(neighbours.items())
    }


def region_adjacency(
    regions: Sequence[DissolvedNicheRegion],
    *,
    touch_tolerance_um: float = 0.0,
) -> dict[str, tuple[str, ...]]:
    """Construct adjacency directly from dissolved region geometry."""

    tolerance = float(touch_tolerance_um)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise AttentionNicheGeometryError(
            "Region touch tolerance must be finite and nonnegative."
        )
    ordered = sorted(regions, key=lambda value: value.niche_id)
    if len({region.niche_id for region in ordered}) != len(ordered):
        raise AttentionNicheGeometryError("Dissolved niche IDs must be unique.")
    geometries = tuple(
        _validate_polygonal_geometry(region.geometry, position=index)
        for index, region in enumerate(ordered)
    )
    neighbours: dict[str, set[str]] = {
        region.niche_id: set() for region in ordered
    }
    if not geometries:
        return {}
    tree = STRtree(geometries)
    for i, geometry in enumerate(geometries):
        query_geometry = geometry if tolerance == 0 else geometry.buffer(tolerance)
        for j in sorted(set(_query_tree_indices(tree, query_geometry, geometries))):
            if j <= i:
                continue
            if float(geometry.distance(geometries[j])) <= tolerance:
                first = ordered[i].niche_id
                second = ordered[j].niche_id
                neighbours[first].add(second)
                neighbours[second].add(first)
    return {
        label: tuple(sorted(values)) for label, values in sorted(neighbours.items())
    }


def _validate_niche_adjacency(
    adjacency: Mapping[str, Iterable[str]],
) -> dict[str, set[str]]:
    nodes = {str(node) for node in adjacency}
    for values in adjacency.values():
        nodes.update(str(value) for value in values)
    neighbours = {node: set() for node in nodes}
    for raw_node, raw_values in adjacency.items():
        node = str(raw_node)
        for raw_neighbour in raw_values:
            neighbour = str(raw_neighbour)
            if neighbour == node:
                raise AttentionNicheGeometryError(
                    "Niche adjacency cannot contain a self-edge."
                )
            neighbours[node].add(neighbour)
    for node in sorted(neighbours):
        for neighbour in neighbours[node]:
            if node not in neighbours.get(neighbour, set()):
                raise AttentionNicheGeometryError(
                    "Niche adjacency must be symmetric before graph coloring."
                )
    return neighbours


def _dsatur_color_classes(neighbours: Mapping[str, set[str]]) -> dict[str, int]:
    assigned: dict[str, int] = {}
    saturation_colors: dict[str, set[int]] = {
        node: set() for node in neighbours
    }
    heap: list[tuple[int, int, str]] = [
        (0, -len(neighbours[node]), node) for node in neighbours
    ]
    heapq.heapify(heap)
    while heap:
        negative_saturation, negative_degree, node = heapq.heappop(heap)
        if node in assigned:
            continue
        expected = (
            -len(saturation_colors[node]),
            -len(neighbours[node]),
            node,
        )
        if (negative_saturation, negative_degree, node) != expected:
            continue
        used = saturation_colors[node]
        color_class = 0
        while color_class in used:
            color_class += 1
        assigned[node] = color_class
        for other in neighbours[node]:
            if other in assigned or color_class in saturation_colors[other]:
                continue
            saturation_colors[other].add(color_class)
            heapq.heappush(
                heap,
                (
                    -len(saturation_colors[other]),
                    -len(neighbours[other]),
                    other,
                ),
            )
    if len(assigned) != len(neighbours):
        raise AssertionError("DSATUR did not assign every niche a color class.")
    return assigned


def _categorical_hex(core_number: int, color_class: int, color_seed: int) -> str:
    # The first graph-color classes are deliberately separated around the hue
    # wheel.  Tissue-region adjacency is planar in the usual case and therefore
    # consumes only the first few, maximally separated entries.
    hue_offsets = (0.0, 210.0, 105.0, 30.0, 270.0, 150.0, 330.0, 75.0, 240.0, 15.0, 180.0, 300.0)
    cycle, offset = divmod(int(color_class), len(hue_offsets))
    seed_digest = hashlib.sha256(str(int(color_seed)).encode("utf-8")).digest()
    seed_rotation = int.from_bytes(seed_digest[:2], "big") % 360
    # 29 is coprime with 360, giving each supported core a distinct rotation
    # before graph-color-class offsets are applied.
    hue = (seed_rotation + 29.0 * core_number + hue_offsets[offset]) % 360.0
    saturation = max(0.52, 0.72 - 0.05 * (cycle % 3))
    lightness = min(0.62, 0.47 + 0.075 * ((cycle // 3) % 3))
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    channels_list = [int(round(255 * value)) for value in (red, green, blue)]
    # Preserve the high blue bit from the categorical palette and encode the
    # 1..99 core number in the remaining seven bits.  Thus no exact hex color
    # can recur across supported cores, even if hue rotations happen to round
    # to the same RGB value.  Red/green and the high blue bit retain the
    # adjacency-aware categorical contrast.
    channels_list[2] = (channels_list[2] & 0x80) | core_number
    channels = tuple(channels_list)
    return "#{:02x}{:02x}{:02x}".format(*channels)


def deterministic_niche_colors(
    core_number: int,
    niche_adjacency: Mapping[str, Iterable[str]],
    *,
    color_seed: int = 2026082503,
) -> dict[str, str]:
    """Assign deterministic, core-scoped, adjacency-aware categorical colors."""

    core = int(core_number)
    if core <= 0 or core > 99 or core != core_number:
        raise AttentionNicheGeometryError(
            "Core number must be an integer from 1 through 99."
        )
    neighbours = _validate_niche_adjacency(niche_adjacency)
    expected_prefix = f"C{core:02d}-N"
    unexpected = sorted(node for node in neighbours if not node.startswith(expected_prefix))
    if unexpected:
        raise AttentionNicheGeometryError(
            "Niche IDs must be scoped to the requested core before coloring."
        )
    classes = _dsatur_color_classes(neighbours)
    colors = {
        node: _categorical_hex(core, classes[node], int(color_seed))
        for node in sorted(classes)
    }
    for node, adjacent in neighbours.items():
        for other in adjacent:
            if colors[node] == colors[other]:
                raise AssertionError("Adjacent graph-color classes received one color.")
    return colors


__all__ = [
    "AttentionNicheGeometryError",
    "CentroidAlignmentReport",
    "DEFAULT_LOCAL_K",
    "DEFAULT_LOCAL_MAX_GAP_UM",
    "DEFAULT_MICRO_NICHE_THRESHOLD",
    "DEFAULT_POLYGON_ADJACENCY_COVERAGE",
    "DissolvedNicheRegion",
    "LocalContiguityGraph",
    "NEUTRAL_NICHE_COLOR",
    "NicheSplitResult",
    "PolygonAlignment",
    "construct_local_contiguity_graph",
    "deterministic_niche_colors",
    "dissolve_niche_regions",
    "dissolved_regions_geojson",
    "load_aligned_cosmx_polygons",
    "niche_adjacency_from_cells",
    "niche_connected_component_counts",
    "region_adjacency",
    "segmentation_polygon_adjacency",
    "split_preliminary_niches",
    "validate_polygon_centroid_alignment",
    "validate_undirected_adjacency",
    "verify_niche_connectedness",
]
