"""Sparse physical graph construction, QC, and mechanism-breaking rewiring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GraphQC:
    n_nodes: int
    n_directed_edges: int
    n_undirected_edges: int
    n_components: int
    n_isolated_nodes: int
    mean_degree: float
    median_degree: float
    p95_degree: float
    max_degree: int
    edge_distance_mean_um: float | None
    edge_distance_p50_um: float | None
    edge_distance_p95_um: float | None
    edge_distance_max_um: float | None
    zero_distance_edges: int
    cap_hit_rate: float | None
    self_loops: int
    duplicate_directed_edges: int
    cross_group_edges: int
    fov_seam_edges: int | None
    fov_seam_fraction: float | None
    directed_edge_pairs_are_symmetric: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SpatialGraph:
    """Directed representation of a symmetric physical candidate graph."""

    edge_index: np.ndarray
    edge_attr: np.ndarray
    edge_attr_names: tuple[str, ...]
    n_nodes: int
    qc: GraphQC
    config: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        edges = np.asarray(self.edge_index)
        attributes = np.asarray(self.edge_attr)
        if edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_directed_edges].")
        if attributes.ndim != 2 or attributes.shape[0] != edges.shape[1]:
            raise ValueError("edge_attr must align to directed edges.")
        if attributes.shape[1] != len(self.edge_attr_names):
            raise ValueError("edge_attr names do not match its columns.")
        if edges.size:
            if edges.min() < 0 or edges.max() >= self.n_nodes:
                raise ValueError("edge_index contains an out-of-range node.")
            if np.any(edges[0] == edges[1]):
                raise ValueError("Spatial graphs cannot contain self-loops.")
        if not np.isfinite(attributes).all():
            raise ValueError("All edge attributes must be finite.")

    @property
    def n_directed_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def n_undirected_edges(self) -> int:
        return self.edge_index.shape[1] // 2

    def to_torch(self, *, device: object | None = None) -> tuple[object, object]:
        """Return torch tensors without making torch an import-time dependency."""

        import torch

        edge_index = torch.as_tensor(self.edge_index, dtype=torch.long, device=device)
        edge_attr = torch.as_tensor(self.edge_attr, dtype=torch.float32, device=device)
        return edge_index, edge_attr


def _validate_coordinates(coordinates_um: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_um must have shape [n_nodes, 2].")
    if len(coordinates) == 0 or not np.isfinite(coordinates).all():
        raise ValueError("Graph coordinates must be non-empty and finite.")
    return coordinates


def _canonical_labels(
    labels: Sequence[object] | np.ndarray | None,
    n_nodes: int,
    *,
    name: str,
) -> np.ndarray:
    if labels is None:
        return np.full(n_nodes, "__all__", dtype="U16")
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != n_nodes:
        raise ValueError(f"{name} labels must align to graph nodes.")
    if any(value is None for value in values.tolist()):
        raise ValueError(f"{name} labels cannot be missing.")
    return np.asarray([str(value) for value in values.tolist()], dtype="U128")


def _directed_knn_candidates(
    coordinates: np.ndarray,
    *,
    k: int,
    radius_um: float,
    min_distance_um: float,
    group_labels: np.ndarray,
) -> tuple[set[tuple[int, int]], np.ndarray]:
    candidates: set[tuple[int, int]] = set()
    candidate_counts = np.zeros(len(coordinates), dtype=np.int64)
    for group in np.unique(group_labels):
        global_indices = np.flatnonzero(group_labels == group)
        local_coordinates = coordinates[global_indices]
        tree = cKDTree(local_coordinates)
        neighborhoods = tree.query_ball_point(local_coordinates, r=radius_um)
        for local_source, local_neighbors in enumerate(neighborhoods):
            source = int(global_indices[local_source])
            ranked: list[tuple[float, int]] = []
            for local_target in local_neighbors:
                target = int(global_indices[int(local_target)])
                if target == source:
                    continue
                distance = float(np.linalg.norm(coordinates[target] - coordinates[source]))
                if distance < min_distance_um or distance > radius_um:
                    continue
                ranked.append((distance, target))
            ranked.sort(key=lambda item: (item[0], item[1]))
            selected = ranked[:k]
            candidate_counts[source] = len(selected)
            candidates.update((source, target) for _, target in selected)
    return candidates, candidate_counts


def _symmetrize_candidates(
    candidates: set[tuple[int, int]],
    symmetry: str,
) -> np.ndarray:
    if symmetry not in {"union", "mutual"}:
        raise ValueError("symmetry must be 'union' or 'mutual'.")
    undirected: set[tuple[int, int]] = set()
    for source, target in candidates:
        pair = (min(source, target), max(source, target))
        if symmetry == "union" or (target, source) in candidates:
            undirected.add(pair)
    if not undirected:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(undirected), dtype=np.int64)


def _rbf_parameters(radius_um: float, rbf_bins: int) -> tuple[np.ndarray, float]:
    if rbf_bins <= 0:
        raise ValueError("rbf_bins must be positive.")
    centers = np.linspace(0.0, radius_um, rbf_bins, dtype=np.float64)
    width = radius_um if rbf_bins == 1 else float(centers[1] - centers[0])
    return centers, max(width, np.finfo(np.float64).eps)


def _edge_attribute_names(rbf_bins: int) -> tuple[str, ...]:
    return (
        "distance_um",
        "distance_over_radius",
        "log1p_distance_um",
        "delta_x_over_radius",
        "delta_y_over_radius",
        "cos_theta",
        "sin_theta",
        "cos_2theta",
        "sin_2theta",
        *(f"distance_rbf_{index}" for index in range(rbf_bins)),
    )


def _directed_edges_and_attributes(
    undirected_pairs: np.ndarray,
    coordinates: np.ndarray,
    *,
    radius_um: float,
    rbf_bins: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    names = _edge_attribute_names(rbf_bins)
    if len(undirected_pairs) == 0:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, len(names)), dtype=np.float32),
            names,
        )
    forward = np.asarray(undirected_pairs, dtype=np.int64)
    reverse = forward[:, ::-1]
    directed_pairs = np.concatenate([forward, reverse], axis=0)
    order = np.lexsort((directed_pairs[:, 1], directed_pairs[:, 0]))
    directed_pairs = directed_pairs[order]
    source = directed_pairs[:, 0]
    target = directed_pairs[:, 1]
    delta = coordinates[target] - coordinates[source]
    distance = np.linalg.norm(delta, axis=1)
    unit = np.divide(
        delta,
        distance[:, None],
        out=np.zeros_like(delta),
        where=distance[:, None] > 0,
    )
    cos_theta = unit[:, 0]
    sin_theta = unit[:, 1]
    centers, width = _rbf_parameters(radius_um, rbf_bins)
    rbf = np.exp(-0.5 * ((distance[:, None] - centers[None, :]) / width) ** 2)
    attributes = np.column_stack(
        [
            distance,
            distance / radius_um,
            np.log1p(distance),
            delta[:, 0] / radius_um,
            delta[:, 1] / radius_um,
            cos_theta,
            sin_theta,
            cos_theta**2 - sin_theta**2,
            2.0 * cos_theta * sin_theta,
            rbf,
        ]
    )
    return (
        directed_pairs.T.astype(np.int64, copy=False),
        attributes.astype(np.float32, copy=False),
        names,
    )


def _graph_qc(
    n_nodes: int,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    candidate_counts: np.ndarray | None,
    k: int | None,
    group_labels: np.ndarray,
    fov_labels: np.ndarray | None,
) -> GraphQC:
    if edge_index.shape[1]:
        encoded = edge_index[0].astype(np.int64) * n_nodes + edge_index[1]
        duplicate_count = int(len(encoded) - len(np.unique(encoded)))
        self_loops = int(np.sum(edge_index[0] == edge_index[1]))
        edge_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        symmetric = all((target, source) in edge_set for source, target in edge_set)
        unique_mask = edge_index[0] < edge_index[1]
        unique_source = edge_index[0, unique_mask]
        unique_target = edge_index[1, unique_mask]
        distance = edge_attr[unique_mask, 0].astype(np.float64)
        rows = np.concatenate([unique_source, unique_target])
        cols = np.concatenate([unique_target, unique_source])
        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n_nodes, n_nodes),
        ).tocsr()
        n_components = int(
            connected_components(adjacency, directed=False, return_labels=False)
        )
        cross_group = int(
            np.sum(group_labels[unique_source] != group_labels[unique_target])
        )
        if fov_labels is not None:
            seam_edges = int(np.sum(fov_labels[unique_source] != fov_labels[unique_target]))
            seam_fraction: float | None = seam_edges / len(unique_source)
        else:
            seam_edges = None
            seam_fraction = None
    else:
        duplicate_count = 0
        self_loops = 0
        symmetric = True
        distance = np.empty(0, dtype=np.float64)
        n_components = n_nodes
        cross_group = 0
        seam_edges = 0 if fov_labels is not None else None
        seam_fraction = 0.0 if fov_labels is not None else None

    degree = np.bincount(edge_index[0], minlength=n_nodes).astype(np.int64)
    if candidate_counts is not None and k is not None:
        cap_hit_rate: float | None = float(np.mean(candidate_counts >= k))
    else:
        cap_hit_rate = None
    if len(distance):
        distance_mean: float | None = float(distance.mean())
        distance_p50: float | None = float(np.quantile(distance, 0.50))
        distance_p95: float | None = float(np.quantile(distance, 0.95))
        distance_max: float | None = float(distance.max())
        zero_distance = int(np.sum(distance == 0))
    else:
        distance_mean = distance_p50 = distance_p95 = distance_max = None
        zero_distance = 0
    return GraphQC(
        n_nodes=int(n_nodes),
        n_directed_edges=int(edge_index.shape[1]),
        n_undirected_edges=int(edge_index.shape[1] // 2),
        n_components=n_components,
        n_isolated_nodes=int(np.sum(degree == 0)),
        mean_degree=float(degree.mean()),
        median_degree=float(np.median(degree)),
        p95_degree=float(np.quantile(degree, 0.95)),
        max_degree=int(degree.max(initial=0)),
        edge_distance_mean_um=distance_mean,
        edge_distance_p50_um=distance_p50,
        edge_distance_p95_um=distance_p95,
        edge_distance_max_um=distance_max,
        zero_distance_edges=zero_distance,
        cap_hit_rate=cap_hit_rate,
        self_loops=self_loops,
        duplicate_directed_edges=duplicate_count,
        cross_group_edges=cross_group,
        fov_seam_edges=seam_edges,
        fov_seam_fraction=seam_fraction,
        directed_edge_pairs_are_symmetric=bool(symmetric),
    )


def _assemble_graph(
    undirected_pairs: np.ndarray,
    coordinates: np.ndarray,
    *,
    radius_um: float,
    rbf_bins: int,
    group_labels: np.ndarray,
    fov_labels: np.ndarray | None,
    candidate_counts: np.ndarray | None,
    k_for_qc: int | None,
    config: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> SpatialGraph:
    edge_index, edge_attr, names = _directed_edges_and_attributes(
        undirected_pairs,
        coordinates,
        radius_um=radius_um,
        rbf_bins=rbf_bins,
    )
    qc = _graph_qc(
        len(coordinates),
        edge_index,
        edge_attr,
        candidate_counts=candidate_counts,
        k=k_for_qc,
        group_labels=group_labels,
        fov_labels=fov_labels,
    )
    if qc.cross_group_edges:
        raise RuntimeError("Internal error: a graph edge crosses group boundaries.")
    return SpatialGraph(
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_attr_names=names,
        n_nodes=len(coordinates),
        qc=qc,
        config=dict(config),
        metadata={} if metadata is None else dict(metadata),
    )


def build_spatial_graph(
    coordinates_um: np.ndarray,
    *,
    k: int = 12,
    radius_um: float = 50.0,
    symmetry: str = "union",
    group_labels: Sequence[object] | np.ndarray | None = None,
    fov: Sequence[object] | np.ndarray | None = None,
    min_distance_um: float = 0.0,
    rbf_bins: int = 8,
) -> SpatialGraph:
    """Build a deterministic capped kNN/radius graph.

    Each node first selects at most ``k`` neighbors inside ``radius_um``.
    ``union`` retains a relation selected in either direction, whereas
    ``mutual`` requires both directions.  Every retained undirected relation is
    then represented by two directed sender-to-receiver edges.  Supplying split
    labels as ``group_labels`` constructs each split independently and makes
    cross-split edges impossible.
    """

    coordinates = _validate_coordinates(coordinates_um)
    if not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("radius_um must be positive and finite.")
    if (
        not np.isfinite(min_distance_um)
        or min_distance_um < 0
        or min_distance_um >= radius_um
    ):
        raise ValueError("min_distance_um must be in [0, radius_um).")
    groups = _canonical_labels(group_labels, len(coordinates), name="group")
    fov_labels = (
        None if fov is None else _canonical_labels(fov, len(coordinates), name="FOV")
    )
    candidates, candidate_counts = _directed_knn_candidates(
        coordinates,
        k=int(k),
        radius_um=float(radius_um),
        min_distance_um=float(min_distance_um),
        group_labels=groups,
    )
    undirected = _symmetrize_candidates(candidates, symmetry)
    config = {
        "kind": "spatial_knn_radius",
        "k": int(k),
        "radius_um": float(radius_um),
        "min_distance_um": float(min_distance_um),
        "symmetry": symmetry,
        "rbf_bins": int(rbf_bins),
        "group_restricted": group_labels is not None,
    }
    return _assemble_graph(
        undirected,
        coordinates,
        radius_um=float(radius_um),
        rbf_bins=int(rbf_bins),
        group_labels=groups,
        fov_labels=fov_labels,
        candidate_counts=candidate_counts,
        k_for_qc=int(k),
        config=config,
    )


def _unique_undirected_pairs(graph: SpatialGraph) -> np.ndarray:
    mask = graph.edge_index[0] < graph.edge_index[1]
    pairs = graph.edge_index[:, mask].T.astype(np.int64, copy=True)
    if len(pairs) * 2 != graph.edge_index.shape[1]:
        raise ValueError("Rewiring requires two directed edges per undirected relation.")
    edge_set = set(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    if not all((target, source) in edge_set for source, target in edge_set):
        raise ValueError("Rewiring requires a symmetric directed graph.")
    return pairs


def _distance_bin_boundaries(
    distances: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("distance_bins must be positive.")
    if len(distances) == 0:
        return np.empty(0, dtype=np.float64)
    quantiles = np.quantile(distances, np.linspace(0.0, 1.0, n_bins + 1))
    return np.unique(quantiles[1:-1])


def _bin_distance(value: float, boundaries: np.ndarray) -> int:
    return int(np.searchsorted(boundaries, value, side="right"))


def rewire_spatial_graph(
    graph: SpatialGraph,
    coordinates_um: np.ndarray,
    *,
    seed: int = 0,
    swaps_per_edge: float = 3.0,
    distance_bins: int = 8,
    max_attempts_per_swap: int = 100,
    group_labels: Sequence[object] | np.ndarray | None = None,
    fov: Sequence[object] | np.ndarray | None = None,
) -> SpatialGraph:
    """Degree-preserving double-edge swaps constrained by distance quantiles.

    Every accepted swap preserves node degrees exactly, remains inside its
    supplied split/group, obeys the original radius/minimum-distance limits,
    and preserves each swapped edge's original distance-quantile bin.  Sparse
    or highly regular graphs may admit few such swaps; the achieved fraction is
    reported rather than silently relaxing the null.
    """

    coordinates = _validate_coordinates(coordinates_um)
    if len(coordinates) != graph.n_nodes:
        raise ValueError("Coordinates must align to graph nodes.")
    if swaps_per_edge < 0 or not np.isfinite(swaps_per_edge):
        raise ValueError("swaps_per_edge must be finite and nonnegative.")
    if max_attempts_per_swap <= 0:
        raise ValueError("max_attempts_per_swap must be positive.")
    radius_um = float(graph.config.get("radius_um", np.inf))
    min_distance_um = float(graph.config.get("min_distance_um", 0.0))
    rbf_bins = int(graph.config.get("rbf_bins", 8))
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("The source graph must record a finite positive radius.")
    if bool(graph.config.get("group_restricted", False)) and group_labels is None:
        raise ValueError(
            "A group-restricted source graph requires the same group labels for rewiring."
        )

    groups = _canonical_labels(group_labels, graph.n_nodes, name="group")
    fov_labels = (
        None if fov is None else _canonical_labels(fov, graph.n_nodes, name="FOV")
    )
    pairs = _unique_undirected_pairs(graph)
    n_edges = len(pairs)
    target_swaps = int(round(float(swaps_per_edge) * n_edges))
    if n_edges < 2 or target_swaps == 0:
        return _assemble_graph(
            pairs,
            coordinates,
            radius_um=radius_um,
            rbf_bins=rbf_bins,
            group_labels=groups,
            fov_labels=fov_labels,
            candidate_counts=None,
            k_for_qc=None,
            config={**dict(graph.config), "kind": "degree_distance_rewired"},
            metadata={
                "rewire_seed": int(seed),
                "rewire_target_swaps": target_swaps,
                "rewire_successful_swaps": 0,
                "rewire_success_fraction": 0.0 if target_swaps else 1.0,
            },
        )

    original_pairs = pairs.copy()
    original_distances = np.linalg.norm(
        coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]], axis=1
    )
    boundaries = _distance_bin_boundaries(original_distances, distance_bins)
    slot_bins = np.asarray(
        [_bin_distance(value, boundaries) for value in original_distances],
        dtype=np.int64,
    )
    edge_groups = groups[pairs[:, 0]]
    if np.any(edge_groups != groups[pairs[:, 1]]):
        raise ValueError("The source graph already crosses supplied group boundaries.")

    slots_by_group_bin: dict[tuple[str, int], np.ndarray] = {}
    midpoint_trees: dict[tuple[str, int], cKDTree] = {}
    for group in np.unique(edge_groups):
        for bin_index in np.unique(slot_bins[edge_groups == group]):
            slots = np.flatnonzero(
                (edge_groups == group) & (slot_bins == bin_index)
            )
            key = (str(group), int(bin_index))
            slots_by_group_bin[key] = slots
            midpoints = coordinates[pairs[slots]].mean(axis=1)
            midpoint_trees[key] = cKDTree(midpoints)

    edge_set = {tuple(pair) for pair in pairs.tolist()}
    rng = np.random.default_rng(int(seed))
    successful = 0
    attempts = 0
    max_attempts = max(1, target_swaps * int(max_attempts_per_swap))
    while successful < target_swaps and attempts < max_attempts:
        attempts += 1
        first_slot = int(rng.integers(n_edges))
        key = (str(edge_groups[first_slot]), int(slot_bins[first_slot]))
        candidate_slots = slots_by_group_bin[key]
        if len(candidate_slots) < 2:
            continue

        # Query genuinely local edges rather than drawing a global random pool:
        # local quadrilaterals admit distance-matched double swaps far more
        # often, while the seed still controls which nearby proposal is tried.
        first_midpoint = coordinates[pairs[first_slot]].mean(axis=0)
        neighbor_count = min(64, len(candidate_slots))
        _, local_neighbor_indices = midpoint_trees[key].query(
            first_midpoint, k=neighbor_count
        )
        local_neighbor_indices = np.atleast_1d(local_neighbor_indices).astype(
            np.int64, copy=False
        )
        nearby_slots = candidate_slots[local_neighbor_indices]
        nearby_slots = nearby_slots[nearby_slots != first_slot]
        if len(nearby_slots) == 0:
            continue
        choice_count = min(24, len(nearby_slots))
        second_slot = int(nearby_slots[int(rng.integers(choice_count))])

        a, b = (int(value) for value in pairs[first_slot])
        c, d = (int(value) for value in pairs[second_slot])
        if len({a, b, c, d}) < 4:
            continue
        proposals = [
            ((min(a, c), max(a, c)), (min(b, d), max(b, d))),
            ((min(a, d), max(a, d)), (min(b, c), max(b, c))),
        ]
        if int(rng.integers(2)):
            proposals.reverse()
        accepted: tuple[tuple[int, int], tuple[int, int]] | None = None
        old_first = tuple(pairs[first_slot].tolist())
        old_second = tuple(pairs[second_slot].tolist())
        old_pair_set = {old_first, old_second}
        for new_first, new_second in proposals:
            if (
                new_first == new_second
                or new_first[0] == new_first[1]
                or new_second[0] == new_second[1]
            ):
                continue
            if (
                (new_first in edge_set and new_first not in old_pair_set)
                or (new_second in edge_set and new_second not in old_pair_set)
            ):
                continue
            if groups[new_first[0]] != groups[new_first[1]]:
                continue
            if groups[new_second[0]] != groups[new_second[1]]:
                continue
            first_distance = float(
                np.linalg.norm(coordinates[new_first[1]] - coordinates[new_first[0]])
            )
            second_distance = float(
                np.linalg.norm(coordinates[new_second[1]] - coordinates[new_second[0]])
            )
            if not (
                min_distance_um <= first_distance <= radius_um
                and min_distance_um <= second_distance <= radius_um
            ):
                continue
            first_bin = _bin_distance(first_distance, boundaries)
            second_bin = _bin_distance(second_distance, boundaries)
            if first_bin == slot_bins[first_slot] and second_bin == slot_bins[second_slot]:
                accepted = (new_first, new_second)
                break
            if first_bin == slot_bins[second_slot] and second_bin == slot_bins[first_slot]:
                accepted = (new_second, new_first)
                break
        if accepted is None:
            continue
        edge_set.remove(old_first)
        edge_set.remove(old_second)
        edge_set.add(accepted[0])
        edge_set.add(accepted[1])
        pairs[first_slot] = accepted[0]
        pairs[second_slot] = accepted[1]
        successful += 1

    pairs = np.asarray(sorted(edge_set), dtype=np.int64)
    old_degree = np.bincount(original_pairs.ravel(), minlength=graph.n_nodes)
    new_degree = np.bincount(pairs.ravel(), minlength=graph.n_nodes)
    if not np.array_equal(old_degree, new_degree):
        raise RuntimeError("Internal error: rewiring changed node degrees.")
    new_distances = np.linalg.norm(
        coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]], axis=1
    )
    metadata = {
        "rewire_seed": int(seed),
        "rewire_target_swaps": int(target_swaps),
        "rewire_successful_swaps": int(successful),
        "rewire_attempts": int(attempts),
        "rewire_success_fraction": float(successful / target_swaps),
        "degree_preserved_exactly": True,
        "original_edge_distance_mean_um": float(original_distances.mean()),
        "rewired_edge_distance_mean_um": float(new_distances.mean()),
        "relative_edge_distance_mean_change": float(
            abs(new_distances.mean() - original_distances.mean())
            / max(original_distances.mean(), np.finfo(np.float64).eps)
        ),
        "distance_bin_boundaries_um": boundaries.tolist(),
    }
    return _assemble_graph(
        pairs,
        coordinates,
        radius_um=radius_um,
        rbf_bins=rbf_bins,
        group_labels=groups,
        fov_labels=fov_labels,
        candidate_counts=None,
        k_for_qc=None,
        config={**dict(graph.config), "kind": "degree_distance_rewired"},
        metadata=metadata,
    )


class EdgeAttributeStandardizer:
    """Optional train-graph-only standardization for raw edge attributes."""

    def __init__(self, *, epsilon: float = 1e-8) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.epsilon = float(epsilon)
        self.is_fitted_ = False

    def fit(self, training_edge_attr: np.ndarray) -> "EdgeAttributeStandardizer":
        attributes = np.asarray(training_edge_attr, dtype=np.float64)
        if attributes.ndim != 2 or len(attributes) == 0:
            raise ValueError("Training edge attributes must be a non-empty 2D array.")
        if not np.isfinite(attributes).all():
            raise ValueError("Training edge attributes must be finite.")
        self.mean_ = attributes.mean(axis=0)
        scale = attributes.std(axis=0, ddof=0)
        self.scale_ = np.where(scale > self.epsilon, scale, 1.0)
        self.n_features_ = attributes.shape[1]
        self.is_fitted_ = True
        return self

    def transform(self, edge_attr: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("EdgeAttributeStandardizer must be fitted first.")
        attributes = np.asarray(edge_attr, dtype=np.float64)
        if attributes.ndim != 2 or attributes.shape[1] != self.n_features_:
            raise ValueError("Edge attributes differ from the fitted schema.")
        if not np.isfinite(attributes).all():
            raise ValueError("Edge attributes must be finite.")
        return ((attributes - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, training_edge_attr: np.ndarray) -> np.ndarray:
        return self.fit(training_edge_attr).transform(training_edge_attr)
