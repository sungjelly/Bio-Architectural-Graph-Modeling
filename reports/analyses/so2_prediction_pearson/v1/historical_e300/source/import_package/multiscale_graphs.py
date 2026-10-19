"""Deterministic multiscale graphs and a geometry-matched local null.

The campaign graph separates a sparse local scale from regional context:

* local: exact capped kNN with ``k=64`` and distance at most ``75 um``;
* regional: exact capped kNN with ``k=256`` in the annulus ``(75, 300] um``.

Both candidate graphs require mutual selection and are stored as symmetric,
receiver-sorted directed edges.  The local null uses the established
degree/distance-stratified double-edge-swap implementation.  Its raw geometry
is recomputed from measured coordinates and transformed with the standardizer
fitted on the *true* local graph, so the null cannot silently refit its edge
representation.  Rewiring fails closed when optimistic feasibility or final
topology-change gates fail.  ``build_true_multiscale_graphs`` is the explicit
no-null path for designs that retain only the measured local/regional graphs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterator, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .full_core import (
    EDGE_ATTRIBUTE_NAMES,
    FullCoreContractError,
    ReceiverEdgeShard,
    _array_sha256,
    _build_standardized_shards,
    _canonical_json,
    _component_count,
    _decode_receiver_codes,
    _fit_edge_standardizer,
    _mutual_edge_codes,
    _raw_geometry_attributes,
    _receiver_ranges,
    _select_exact_neighbors,
    _update_raw_array_bytes,
)
LOCAL_K_CAP = 64
LOCAL_MAX_DISTANCE_UM = 75.0
REGIONAL_K_CAP = 256
REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM = 75.0
REGIONAL_MAX_DISTANCE_UM = 300.0
LOCAL_REWIRE_SEED = 271828
LOCAL_REWIRE_DISTANCE_BINS = 8
LOCAL_REWIRE_SWAPS_PER_EDGE = 1.0
LOCAL_REWIRE_MAX_RELATIVE_MEAN_DISTANCE_CHANGE = 0.10
LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION = 0.50
_RBF_BINS = len(EDGE_ATTRIBUTE_NAMES) - 9
_GEOMETRY_HASH_CHUNK_EDGES = 1_000_000


class MultiscaleGraphContractError(ValueError):
    """Raised when a multiscale graph violates the frozen graph contract."""


def _payload_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ScaleGraphQC:
    """Auditable topology and geometry summary for one graph scale."""

    scale: str
    topology_kind: str
    n_nodes: int
    k_cap: int
    minimum_distance_um: float
    minimum_distance_inclusive: bool
    maximum_distance_um: float
    n_directed_candidates: int | None
    candidate_cap_hit_fraction: float | None
    n_directed_edges: int
    n_undirected_edges: int
    n_components: int
    n_isolated_nodes: int
    mean_degree: float
    median_degree: float
    p95_degree: float
    max_degree: int
    edge_distance_min_um: float
    edge_distance_mean_um: float
    edge_distance_p50_um: float
    edge_distance_p95_um: float
    edge_distance_max_um: float
    zero_distance_undirected_edges: int
    self_loops: int
    duplicate_directed_edges: int
    directed_edge_pairs_are_symmetric: bool
    receiver_sorted: bool
    edge_attribute_count: int
    edge_standardization_source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleGraphChecksums:
    """Content checksums for one receiver-sorted scale graph."""

    directed_candidates_sha256: str | None
    receiver_sorted_edge_codes_sha256: str
    edge_index_sha256: str
    raw_geometry_attributes_sha256: str
    standardized_edge_attributes_sha256: str
    edge_attribute_mean_sha256: str
    edge_attribute_scale_sha256: str
    graph_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and not _is_sha256(value):
                raise MultiscaleGraphContractError(
                    f"{name} is not a valid SHA-256 checksum."
                )

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True)
class ReceiverSortedScaleGraph:
    """One standardized graph scale stored in receiver-aligned shards."""

    name: str
    n_nodes: int
    k_cap: int
    minimum_distance_um: float
    minimum_distance_inclusive: bool
    maximum_distance_um: float
    edge_attribute_names: tuple[str, ...]
    edge_attribute_mean: np.ndarray = field(repr=False)
    edge_attribute_scale: np.ndarray = field(repr=False)
    edge_standardization_source: str
    shards: tuple[ReceiverEdgeShard, ...] = field(repr=False)
    qc: ScaleGraphQC
    checksums: ScaleGraphChecksums

    def __post_init__(self) -> None:
        if self.edge_attribute_names != EDGE_ATTRIBUTE_NAMES:
            raise MultiscaleGraphContractError(
                "Multiscale graphs require the locked 17-feature geometry schema."
            )
        if self.n_nodes < 2 or self.k_cap <= 0:
            raise MultiscaleGraphContractError(
                "A scale graph requires at least two nodes and a positive k cap."
            )
        if (
            not np.isfinite(self.minimum_distance_um)
            or not np.isfinite(self.maximum_distance_um)
            or self.minimum_distance_um < 0
            or self.maximum_distance_um <= self.minimum_distance_um
        ):
            raise MultiscaleGraphContractError(
                "Scale distance bounds must be finite and ordered."
            )
        mean = np.asarray(self.edge_attribute_mean)
        scale = np.asarray(self.edge_attribute_scale)
        expected = (len(EDGE_ATTRIBUTE_NAMES),)
        if mean.shape != expected or scale.shape != expected:
            raise MultiscaleGraphContractError(
                "Edge transform statistics have the wrong shape."
            )
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise MultiscaleGraphContractError(
                "Edge transform statistics must be finite with positive scale."
            )
        expected_start = 0
        edge_count = 0
        for shard in self.shards:
            if shard.receiver_start != expected_start:
                raise MultiscaleGraphContractError(
                    "Receiver shards must be contiguous and ordered."
                )
            expected_start = shard.receiver_stop
            edge_count += shard.n_edges
        if expected_start != self.n_nodes:
            raise MultiscaleGraphContractError(
                "Receiver shards do not cover all nodes."
            )
        if edge_count != self.qc.n_directed_edges:
            raise MultiscaleGraphContractError(
                "Receiver shards do not cover every edge exactly once."
            )
        if (
            self.qc.scale != self.name
            or self.qc.n_nodes != self.n_nodes
            or self.qc.k_cap != self.k_cap
            or self.qc.edge_standardization_source
            != self.edge_standardization_source
        ):
            raise MultiscaleGraphContractError(
                "Scale graph metadata and QC disagree."
            )

    @property
    def k(self) -> int:
        """Compatibility alias for graph consumers that name the cap ``k``."""

        return self.k_cap

    @property
    def radius_guard_um(self) -> float:
        """Compatibility alias for the feature-normalization radius."""

        return self.maximum_distance_um

    def iter_shards(self) -> Iterator[ReceiverEdgeShard]:
        return iter(self.shards)

    def concatenate(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialize receiver-sorted edges and attributes on explicit request."""

        if len(self.shards) == 1:
            return self.shards[0].edge_index, self.shards[0].edge_attributes
        return (
            np.concatenate([shard.edge_index for shard in self.shards], axis=1),
            np.concatenate(
                [shard.edge_attributes for shard in self.shards],
                axis=0,
            ),
        )


@dataclass(frozen=True)
class LocalRewiringFeasibilityQC:
    """Necessary, degree-relaxed feasibility checks for the local null."""

    required_final_replaced_relations: int
    maximum_replacements_from_swap_budget: int
    true_distance_bin_counts: tuple[int, ...]
    available_nonoriginal_distance_bin_counts: tuple[int, ...]
    optimistic_replacement_capacity_by_distance_bin: tuple[int, ...]
    degree_relaxed_optimistic_final_replacement_upper_bound: float
    degree_relaxed_optimistic_minimum_final_mean_um_at_threshold: (
        float | None
    )
    degree_relaxed_optimistic_minimum_relative_mean_distance_increase_at_threshold: (
        float | None
    )
    swap_budget_can_reach_threshold: bool
    distance_bin_capacity_can_reach_threshold: bool
    mean_distance_tolerance_can_reach_threshold: bool
    degree_constraints_relaxed: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LocalRewiringQC:
    """Checks specific to the degree/distance-preserving local null."""

    seed: int
    target_swaps: int
    successful_swaps: int
    attempts: int
    success_fraction: float
    topology_changed: bool
    removed_true_undirected_edges: int
    added_rewired_undirected_edges: int
    reintroduced_original_undirected_edges: int
    no_original_relations_reintroduced: bool
    minimum_final_relation_replacement_fraction: float
    final_relation_replacement_fraction: float
    final_relation_replacement_passed: bool
    degree_preserved_exactly: bool
    distance_bin_boundaries_um: tuple[float, ...]
    true_distance_bin_counts: tuple[int, ...]
    rewired_distance_bin_counts: tuple[int, ...]
    distance_bin_counts_preserved_exactly: bool
    true_edge_distance_mean_um: float
    rewired_edge_distance_mean_um: float
    relative_edge_distance_mean_change: float
    maximum_relative_mean_distance_change: float
    distance_tolerance_passed: bool
    raw_geometry_recomputed_from_coordinates: bool
    edge_standardization_source: str
    feasibility: LocalRewiringFeasibilityQC

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultiscaleGraphQC:
    """Cross-scale invariants and local-null QC."""

    local_regional_overlap_directed_edges: int
    local_regional_disjoint: bool
    all_graphs_symmetric: bool
    all_graphs_receiver_sorted: bool
    all_graphs_loop_and_duplicate_free: bool
    rewiring: LocalRewiringQC

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultiscaleGraphChecksums:
    local_graph_sha256: str
    regional_graph_sha256: str
    rewired_local_graph_sha256: str
    bundle_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not _is_sha256(value):
                raise MultiscaleGraphContractError(
                    f"{name} is not a valid SHA-256 checksum."
                )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TrueMultiscaleGraphQC:
    """Cross-scale invariants for a bundle without any null graph."""

    local_regional_overlap_directed_edges: int
    local_regional_disjoint: bool
    all_graphs_symmetric: bool
    all_graphs_receiver_sorted: bool
    all_graphs_loop_and_duplicate_free: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrueMultiscaleGraphChecksums:
    """Checksums for a true-local plus true-regional graph bundle."""

    local_graph_sha256: str
    regional_graph_sha256: str
    bundle_sha256: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not _is_sha256(value):
                raise MultiscaleGraphContractError(
                    f"{name} is not a valid SHA-256 checksum."
                )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TrueMultiscaleGraphs:
    """True local and regional graphs, explicitly excluding any null graph."""

    local: ReceiverSortedScaleGraph
    regional: ReceiverSortedScaleGraph
    qc: TrueMultiscaleGraphQC
    checksums: TrueMultiscaleGraphChecksums

    def __post_init__(self) -> None:
        if self.local.n_nodes != self.regional.n_nodes:
            raise MultiscaleGraphContractError(
                "True local and regional graphs must use the same nodes."
            )
        if not self.qc.local_regional_disjoint:
            raise MultiscaleGraphContractError(
                "True local and regional graph scales must be disjoint."
            )
        if self.qc.local_regional_overlap_directed_edges != 0:
            raise MultiscaleGraphContractError(
                "True graph QC reports a nonzero cross-scale overlap."
            )
        if not (
            self.qc.all_graphs_symmetric
            and self.qc.all_graphs_receiver_sorted
            and self.qc.all_graphs_loop_and_duplicate_free
        ):
            raise MultiscaleGraphContractError(
                "True local/regional graph structural QC failed."
            )
        if (
            self.checksums.local_graph_sha256
            != self.local.checksums.graph_sha256
            or self.checksums.regional_graph_sha256
            != self.regional.checksums.graph_sha256
        ):
            raise MultiscaleGraphContractError(
                "True graph bundle and scale checksums disagree."
            )


@dataclass(frozen=True)
class MultiscaleGraphs:
    """True local, true regional, and rewired-local graph bundle."""

    local: ReceiverSortedScaleGraph
    regional: ReceiverSortedScaleGraph
    rewired_local: ReceiverSortedScaleGraph
    qc: MultiscaleGraphQC
    checksums: MultiscaleGraphChecksums

    def __post_init__(self) -> None:
        node_counts = {
            self.local.n_nodes,
            self.regional.n_nodes,
            self.rewired_local.n_nodes,
        }
        if len(node_counts) != 1:
            raise MultiscaleGraphContractError(
                "All multiscale graphs must use the same nodes."
            )
        if not self.qc.local_regional_disjoint:
            raise MultiscaleGraphContractError(
                "Local and regional graph scales must be disjoint."
            )
        if not self.qc.rewiring.degree_preserved_exactly:
            raise MultiscaleGraphContractError(
                "The rewired local graph must preserve node degree exactly."
            )
        if not self.qc.rewiring.distance_tolerance_passed:
            raise MultiscaleGraphContractError(
                "The rewired local graph failed its distance tolerance."
            )
        if not self.qc.rewiring.no_original_relations_reintroduced:
            raise MultiscaleGraphContractError(
                "The rewired local graph reintroduced an original relation."
            )
        if not self.qc.rewiring.final_relation_replacement_passed:
            raise MultiscaleGraphContractError(
                "The rewired local graph failed its final replacement gate."
            )


def _validate_true_inputs(
    coordinates_um: np.ndarray,
    *,
    query_chunk_size: int,
    receiver_chunk_size: int,
    workers: int,
    epsilon: float,
) -> np.ndarray:
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or len(coordinates) < 2
        or not np.isfinite(coordinates).all()
    ):
        raise MultiscaleGraphContractError(
            "coordinates_um must be finite with shape [N, 2] and N >= 2."
        )
    for name, value in (
        ("query_chunk_size", query_chunk_size),
        ("receiver_chunk_size", receiver_chunk_size),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise MultiscaleGraphContractError(f"{name} must be an integer.")
        if int(value) <= 0:
            raise MultiscaleGraphContractError(f"{name} must be positive.")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, (int, np.integer))
        or int(workers) == 0
        or int(workers) < -1
    ):
        raise MultiscaleGraphContractError(
            "workers must be -1 or a positive integer."
        )
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise MultiscaleGraphContractError(
            "epsilon must be finite and positive."
        )
    if len(coordinates) > int(np.sqrt(np.iinfo(np.int64).max)):
        raise MultiscaleGraphContractError(
            "Node count cannot be encoded safely in int64."
        )
    return coordinates


def _validate_inputs(
    coordinates_um: np.ndarray,
    *,
    query_chunk_size: int,
    receiver_chunk_size: int,
    workers: int,
    epsilon: float,
    rewiring_swaps_per_edge: float,
    rewiring_distance_bins: int,
    rewiring_max_attempts_per_swap: int,
    maximum_relative_mean_distance_change: float,
) -> np.ndarray:
    coordinates = _validate_true_inputs(
        coordinates_um,
        query_chunk_size=query_chunk_size,
        receiver_chunk_size=receiver_chunk_size,
        workers=workers,
        epsilon=epsilon,
    )
    for name, value in (
        ("rewiring_distance_bins", rewiring_distance_bins),
        ("rewiring_max_attempts_per_swap", rewiring_max_attempts_per_swap),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise MultiscaleGraphContractError(f"{name} must be an integer.")
        if int(value) <= 0:
            raise MultiscaleGraphContractError(f"{name} must be positive.")
    if (
        not np.isfinite(rewiring_swaps_per_edge)
        or rewiring_swaps_per_edge <= 0
    ):
        raise MultiscaleGraphContractError(
            "rewiring_swaps_per_edge must be finite and positive."
        )
    if (
        not np.isfinite(maximum_relative_mean_distance_change)
        or maximum_relative_mean_distance_change < 0
    ):
        raise MultiscaleGraphContractError(
            "maximum_relative_mean_distance_change must be finite and "
            "nonnegative."
        )
    return coordinates


def _distance_counts_within(
    tree: cKDTree,
    coordinates: np.ndarray,
    row_indices: np.ndarray,
    *,
    radius_um: float,
    workers: int,
) -> np.ndarray:
    counts = np.asarray(
        tree.query_ball_point(
            coordinates[row_indices],
            r=float(radius_um),
            eps=0.0,
            workers=int(workers),
            return_length=True,
        ),
        dtype=np.int64,
    )
    # Every query contains its own row exactly once.  Other zero-distance nodes
    # remain legitimate local candidates.
    counts -= 1
    if counts.shape != (len(row_indices),) or np.any(counts < 0):
        raise MultiscaleGraphContractError(
            "Exact radius counts returned an invalid result."
        )
    return counts


def _directed_candidate_codes(
    coordinates: np.ndarray,
    *,
    k_cap: int,
    minimum_distance_exclusive_um: float | None,
    maximum_distance_um: float,
    query_chunk_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic source-major capped-radius candidate codes."""

    n_nodes = len(coordinates)
    tree = cKDTree(coordinates)
    chunks: list[np.ndarray] = []
    candidate_counts = np.zeros(n_nodes, dtype=np.int64)
    for start in range(0, n_nodes, int(query_chunk_size)):
        stop = min(start + int(query_chunk_size), n_nodes)
        rows = np.arange(start, stop, dtype=np.int64)
        if minimum_distance_exclusive_um is None:
            query_k = min(int(k_cap), n_nodes - 1)
        else:
            excluded_counts = _distance_counts_within(
                tree,
                coordinates,
                rows,
                radius_um=float(minimum_distance_exclusive_um),
                workers=int(workers),
            )
            query_k = min(
                n_nodes - 1,
                int(excluded_counts.max(initial=0)) + int(k_cap),
            )
        neighbors, distances = _select_exact_neighbors(
            tree,
            coordinates,
            rows,
            k=int(query_k),
            workers=int(workers),
        )
        for local_row, source in enumerate(rows.tolist()):
            keep = distances[local_row] <= float(maximum_distance_um)
            if minimum_distance_exclusive_um is not None:
                keep &= (
                    distances[local_row]
                    > float(minimum_distance_exclusive_um)
                )
            selected = neighbors[local_row, keep][: int(k_cap)]
            if np.any(selected == source):
                raise MultiscaleGraphContractError(
                    "Candidate selection retained a self-loop."
                )
            candidate_counts[source] = len(selected)
            if len(selected):
                chunks.append(
                    int(source) * n_nodes
                    + selected.astype(np.int64, copy=False)
                )
    if not chunks:
        raise MultiscaleGraphContractError(
            "The requested graph scale contains no directed candidates."
        )
    codes = np.concatenate(chunks).astype(np.int64, copy=False)
    codes.sort()
    if np.any(codes[1:] == codes[:-1]):
        raise MultiscaleGraphContractError(
            "Candidate selection produced duplicate directed edges."
        )
    source = codes // n_nodes
    target = codes - source * n_nodes
    if np.any(source == target):
        raise MultiscaleGraphContractError(
            "Candidate selection produced a self-loop."
        )
    return codes, candidate_counts


def _mutual_receiver_codes(
    directed_candidates: np.ndarray,
    *,
    n_nodes: int,
    mutual_search_chunk_size: int,
) -> np.ndarray:
    try:
        codes = _mutual_edge_codes(
            directed_candidates,
            n_nodes=int(n_nodes),
            search_chunk_size=int(mutual_search_chunk_size),
        )
    except FullCoreContractError as exc:
        raise MultiscaleGraphContractError(str(exc)) from exc
    if np.any(codes[1:] <= codes[:-1]):
        raise MultiscaleGraphContractError(
            "Mutual receiver codes must be strictly sorted and unique."
        )
    return codes


def _raw_geometry_checksum(
    receiver_codes: np.ndarray,
    *,
    n_nodes: int,
    coordinates: np.ndarray,
    feature_radius_um: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "name": "raw_geometry_attributes",
                "shape": [len(receiver_codes), len(EDGE_ATTRIBUTE_NAMES)],
                "dtype": np.dtype(np.float32).str,
            }
        )
    )
    for start in range(0, len(receiver_codes), _GEOMETRY_HASH_CHUNK_EDGES):
        stop = min(start + _GEOMETRY_HASH_CHUNK_EDGES, len(receiver_codes))
        raw = _raw_geometry_attributes(
            receiver_codes[start:stop],
            n_nodes=int(n_nodes),
            coordinates=coordinates,
            radius_guard_um=float(feature_radius_um),
        )
        _update_raw_array_bytes(digest, raw)
    return digest.hexdigest()


def _receiver_code_invariants(
    receiver_codes: np.ndarray,
    *,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(receiver_codes) == 0 or len(receiver_codes) % 2:
        raise MultiscaleGraphContractError(
            "A symmetric graph requires a non-empty even directed-edge count."
        )
    if np.any(receiver_codes[1:] <= receiver_codes[:-1]):
        raise MultiscaleGraphContractError(
            "Receiver edge codes must be strictly sorted and unique."
        )
    source, receiver = _decode_receiver_codes(
        receiver_codes,
        n_nodes=int(n_nodes),
    )
    if np.any(source == receiver):
        raise MultiscaleGraphContractError(
            "Receiver edge codes contain a self-loop."
        )
    reverse = source * int(n_nodes) + receiver
    positions = np.searchsorted(receiver_codes, reverse)
    found = positions < len(receiver_codes)
    if np.any(found):
        found[found] = (
            receiver_codes[positions[found]] == reverse[found]
        )
    if not np.all(found):
        raise MultiscaleGraphContractError(
            "Every directed edge must have its reverse edge."
        )
    return source, receiver, source < receiver


def _build_scale_graph(
    receiver_codes: np.ndarray,
    *,
    name: str,
    topology_kind: str,
    coordinates: np.ndarray,
    k_cap: int,
    minimum_distance_um: float,
    minimum_distance_inclusive: bool,
    maximum_distance_um: float,
    candidate_counts: np.ndarray | None,
    directed_candidates_sha256: str | None,
    edge_attribute_mean: np.ndarray,
    edge_attribute_scale: np.ndarray,
    edge_standardization_source: str,
    receiver_chunk_size: int,
) -> ReceiverSortedScaleGraph:
    n_nodes = len(coordinates)
    source, receiver, unique_relation = _receiver_code_invariants(
        receiver_codes,
        n_nodes=n_nodes,
    )
    distances = np.linalg.norm(
        coordinates[receiver[unique_relation]]
        - coordinates[source[unique_relation]],
        axis=1,
    )
    lower_ok = (
        distances >= float(minimum_distance_um)
        if minimum_distance_inclusive
        else distances > float(minimum_distance_um)
    )
    if not np.all(lower_ok) or np.any(
        distances > float(maximum_distance_um)
    ):
        raise MultiscaleGraphContractError(
            f"{name} edges violate the declared distance interval."
        )
    receiver_boundaries = np.searchsorted(
        receiver_codes,
        np.arange(n_nodes + 1, dtype=np.int64) * n_nodes,
    ).astype(np.int64, copy=False)
    degree = np.diff(receiver_boundaries)
    n_components = _component_count(
        receiver_codes,
        n_nodes=n_nodes,
        receiver_boundaries=receiver_boundaries,
    )
    qc = ScaleGraphQC(
        scale=name,
        topology_kind=topology_kind,
        n_nodes=n_nodes,
        k_cap=int(k_cap),
        minimum_distance_um=float(minimum_distance_um),
        minimum_distance_inclusive=bool(minimum_distance_inclusive),
        maximum_distance_um=float(maximum_distance_um),
        n_directed_candidates=(
            None
            if candidate_counts is None
            else int(candidate_counts.sum())
        ),
        candidate_cap_hit_fraction=(
            None
            if candidate_counts is None
            else float(np.mean(candidate_counts >= int(k_cap)))
        ),
        n_directed_edges=len(receiver_codes),
        n_undirected_edges=len(receiver_codes) // 2,
        n_components=n_components,
        n_isolated_nodes=int(np.sum(degree == 0)),
        mean_degree=float(degree.mean()),
        median_degree=float(np.median(degree)),
        p95_degree=float(np.quantile(degree, 0.95)),
        max_degree=int(degree.max(initial=0)),
        edge_distance_min_um=float(distances.min()),
        edge_distance_mean_um=float(distances.mean()),
        edge_distance_p50_um=float(np.quantile(distances, 0.50)),
        edge_distance_p95_um=float(np.quantile(distances, 0.95)),
        edge_distance_max_um=float(distances.max()),
        zero_distance_undirected_edges=int(np.sum(distances == 0.0)),
        self_loops=0,
        duplicate_directed_edges=0,
        directed_edge_pairs_are_symmetric=True,
        receiver_sorted=True,
        edge_attribute_count=len(EDGE_ATTRIBUTE_NAMES),
        edge_standardization_source=edge_standardization_source,
    )
    ranges = _receiver_ranges(
        receiver_codes,
        n_nodes=n_nodes,
        receiver_chunk_size=int(receiver_chunk_size),
    )
    shards, edge_index_checksum, standardized_checksum = (
        _build_standardized_shards(
            receiver_codes,
            ranges,
            n_nodes=n_nodes,
            coordinates=coordinates,
            radius_guard_um=float(maximum_distance_um),
            mean=np.asarray(edge_attribute_mean, dtype=np.float64),
            scale=np.asarray(edge_attribute_scale, dtype=np.float64),
        )
    )
    receiver_codes_checksum = _array_sha256(
        "receiver_sorted_edge_codes",
        receiver_codes,
    )
    raw_geometry_checksum = _raw_geometry_checksum(
        receiver_codes,
        n_nodes=n_nodes,
        coordinates=coordinates,
        feature_radius_um=float(maximum_distance_um),
    )
    mean_checksum = _array_sha256(
        "edge_attribute_mean",
        np.asarray(edge_attribute_mean, dtype=np.float64),
    )
    scale_checksum = _array_sha256(
        "edge_attribute_scale",
        np.asarray(edge_attribute_scale, dtype=np.float64),
    )
    graph_payload: dict[str, object] = {
        "schema": "multiscale_receiver_graph_v1",
        "name": name,
        "topology_kind": topology_kind,
        "config": {
            "k_cap": int(k_cap),
            "minimum_distance_um": float(minimum_distance_um),
            "minimum_distance_inclusive": bool(
                minimum_distance_inclusive
            ),
            "maximum_distance_um": float(maximum_distance_um),
            "symmetry": "mutual",
            "self_loops": False,
            "rbf_bins": _RBF_BINS,
            "edge_standardization_source": edge_standardization_source,
        },
        "edge_attribute_names": list(EDGE_ATTRIBUTE_NAMES),
        "qc": qc.to_dict(),
        "component_checksums": {
            "directed_candidates": directed_candidates_sha256,
            "receiver_sorted_edge_codes": receiver_codes_checksum,
            "edge_index": edge_index_checksum,
            "raw_geometry_attributes": raw_geometry_checksum,
            "standardized_edge_attributes": standardized_checksum,
            "edge_attribute_mean": mean_checksum,
            "edge_attribute_scale": scale_checksum,
        },
    }
    checksums = ScaleGraphChecksums(
        directed_candidates_sha256=directed_candidates_sha256,
        receiver_sorted_edge_codes_sha256=receiver_codes_checksum,
        edge_index_sha256=edge_index_checksum,
        raw_geometry_attributes_sha256=raw_geometry_checksum,
        standardized_edge_attributes_sha256=standardized_checksum,
        edge_attribute_mean_sha256=mean_checksum,
        edge_attribute_scale_sha256=scale_checksum,
        graph_sha256=_payload_sha256(graph_payload),
    )
    return ReceiverSortedScaleGraph(
        name=name,
        n_nodes=n_nodes,
        k_cap=int(k_cap),
        minimum_distance_um=float(minimum_distance_um),
        minimum_distance_inclusive=bool(minimum_distance_inclusive),
        maximum_distance_um=float(maximum_distance_um),
        edge_attribute_names=EDGE_ATTRIBUTE_NAMES,
        edge_attribute_mean=np.asarray(
            edge_attribute_mean,
            dtype=np.float64,
        ).copy(),
        edge_attribute_scale=np.asarray(
            edge_attribute_scale,
            dtype=np.float64,
        ).copy(),
        edge_standardization_source=edge_standardization_source,
        shards=shards,
        qc=qc,
        checksums=checksums,
    )


def _true_scale(
    coordinates: np.ndarray,
    *,
    name: str,
    k_cap: int,
    minimum_distance_exclusive_um: float | None,
    maximum_distance_um: float,
    query_chunk_size: int,
    receiver_chunk_size: int,
    mutual_search_chunk_size: int,
    workers: int,
    epsilon: float,
) -> tuple[ReceiverSortedScaleGraph, np.ndarray]:
    candidates, candidate_counts = _directed_candidate_codes(
        coordinates,
        k_cap=int(k_cap),
        minimum_distance_exclusive_um=minimum_distance_exclusive_um,
        maximum_distance_um=float(maximum_distance_um),
        query_chunk_size=int(query_chunk_size),
        workers=int(workers),
    )
    candidate_checksum = _array_sha256(
        "sorted_directed_candidate_codes",
        candidates,
    )
    receiver_codes = _mutual_receiver_codes(
        candidates,
        n_nodes=len(coordinates),
        mutual_search_chunk_size=int(mutual_search_chunk_size),
    )
    mean, scale, _ = _fit_edge_standardizer(
        receiver_codes,
        n_nodes=len(coordinates),
        coordinates=coordinates,
        radius_guard_um=float(maximum_distance_um),
        epsilon=float(epsilon),
    )
    graph = _build_scale_graph(
        receiver_codes,
        name=name,
        topology_kind="true_exact_mutual_capped_knn",
        coordinates=coordinates,
        k_cap=int(k_cap),
        minimum_distance_um=(
            0.0
            if minimum_distance_exclusive_um is None
            else float(minimum_distance_exclusive_um)
        ),
        minimum_distance_inclusive=minimum_distance_exclusive_um is None,
        maximum_distance_um=float(maximum_distance_um),
        candidate_counts=candidate_counts,
        directed_candidates_sha256=candidate_checksum,
        edge_attribute_mean=mean,
        edge_attribute_scale=scale,
        edge_standardization_source=f"true_{name}",
        receiver_chunk_size=int(receiver_chunk_size),
    )
    del candidates
    return graph, receiver_codes


def _unique_undirected_pairs(
    receiver_codes: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    source, receiver, unique = _receiver_code_invariants(
        receiver_codes,
        n_nodes=n_nodes,
    )
    return np.column_stack([source[unique], receiver[unique]]).astype(
        np.int64,
        copy=False,
    )


def _receiver_codes_from_directed_edges(
    edge_index: np.ndarray,
    *,
    n_nodes: int,
) -> np.ndarray:
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise MultiscaleGraphContractError(
            "edge_index must have shape [2, E]."
        )
    codes = (
        edges[1].astype(np.int64, copy=False) * int(n_nodes)
        + edges[0].astype(np.int64, copy=False)
    )
    codes.sort()
    _receiver_code_invariants(codes, n_nodes=int(n_nodes))
    return codes


def _distance_bin_counts(
    distances: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    bins = np.searchsorted(boundaries, distances, side="right")
    return np.bincount(bins, minlength=len(boundaries) + 1).astype(
        np.int64,
        copy=False,
    )


def _distance_bin_boundaries(
    distances: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    if len(distances) == 0:
        raise MultiscaleGraphContractError(
            "Distance-stratified rewiring requires at least one relation."
        )
    quantiles = np.quantile(
        distances,
        np.linspace(0.0, 1.0, int(n_bins) + 1),
    )
    return np.unique(quantiles[1:-1]).astype(np.float64, copy=False)


def _local_rewiring_feasibility(
    true_pairs: np.ndarray,
    coordinates: np.ndarray,
    *,
    distance_bins: int,
    target_swaps: int,
    maximum_relative_mean_distance_change: float,
) -> LocalRewiringFeasibilityQC:
    """Compute necessary optimistic bounds before attempting any swaps.

    These bounds deliberately relax node-degree constraints.  Failure is
    therefore conclusive: an exact-degree construction cannot rescue a graph
    that already fails candidate capacity, swap budget, or the best possible
    mean-distance bound in this relaxed problem.  Passing is not evidence that
    a degree-preserving realization exists.
    """

    pairs = np.asarray(true_pairs, dtype=np.int64)
    n_nodes = len(coordinates)
    n_relations = len(pairs)
    distances = np.linalg.norm(
        coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]],
        axis=1,
    )
    boundaries = _distance_bin_boundaries(distances, int(distance_bins))
    true_bins = np.searchsorted(
        boundaries,
        distances,
        side="right",
    )
    true_counts = np.bincount(
        true_bins,
        minlength=len(boundaries) + 1,
    ).astype(np.int64, copy=False)

    geometric_pairs = np.asarray(
        cKDTree(coordinates).query_pairs(
            LOCAL_MAX_DISTANCE_UM,
            eps=0.0,
            output_type="ndarray",
        ),
        dtype=np.int64,
    ).reshape(-1, 2)
    geometric_codes = (
        geometric_pairs[:, 0] * n_nodes + geometric_pairs[:, 1]
    )
    true_codes = pairs[:, 0] * n_nodes + pairs[:, 1]
    nonoriginal = ~np.isin(
        geometric_codes,
        true_codes,
        assume_unique=False,
    )
    nonoriginal_pairs = geometric_pairs[nonoriginal]
    nonoriginal_distances = np.linalg.norm(
        coordinates[nonoriginal_pairs[:, 1]]
        - coordinates[nonoriginal_pairs[:, 0]],
        axis=1,
    )
    nonoriginal_bins = np.searchsorted(
        boundaries,
        nonoriginal_distances,
        side="right",
    )
    available_counts = np.bincount(
        nonoriginal_bins,
        minlength=len(boundaries) + 1,
    ).astype(np.int64, copy=False)
    capacity_by_bin = np.minimum(true_counts, available_counts)

    required = int(
        np.ceil(
            LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION
            * n_relations
        )
    )
    swap_budget = min(n_relations, 2 * int(target_swaps))
    capacity = int(capacity_by_bin.sum())
    optimistic_replacements = min(swap_budget, capacity)
    optimistic_fraction = float(
        optimistic_replacements / n_relations
    )
    swap_budget_feasible = swap_budget >= required
    distance_capacity_feasible = capacity >= required

    optimistic_mean: float | None = None
    optimistic_relative_increase: float | None = None
    mean_feasible = False
    if distance_capacity_feasible:
        marginal_distance_changes: list[np.ndarray] = []
        for bin_index, bin_capacity in enumerate(
            capacity_by_bin.tolist()
        ):
            if int(bin_capacity) == 0:
                continue
            original_bin_distances = np.sort(
                distances[true_bins == bin_index]
            )
            candidate_bin_distances = np.sort(
                nonoriginal_distances[
                    nonoriginal_bins == bin_index
                ]
            )
            count = int(bin_capacity)
            marginal_distance_changes.append(
                candidate_bin_distances[:count]
                - original_bin_distances[::-1][:count]
            )
        marginal = np.concatenate(marginal_distance_changes)
        smallest = np.partition(marginal, required - 1)[:required]
        optimistic_total = float(distances.sum() + smallest.sum())
        optimistic_mean = optimistic_total / n_relations
        true_mean = float(distances.mean())
        optimistic_relative_increase = float(
            max(0.0, optimistic_mean - true_mean)
            / max(true_mean, np.finfo(np.float64).eps)
        )
        mean_feasible = bool(
            optimistic_relative_increase
            <= float(maximum_relative_mean_distance_change)
        )

    passed = bool(
        swap_budget_feasible
        and distance_capacity_feasible
        and mean_feasible
    )
    return LocalRewiringFeasibilityQC(
        required_final_replaced_relations=required,
        maximum_replacements_from_swap_budget=swap_budget,
        true_distance_bin_counts=tuple(
            int(value) for value in true_counts.tolist()
        ),
        available_nonoriginal_distance_bin_counts=tuple(
            int(value) for value in available_counts.tolist()
        ),
        optimistic_replacement_capacity_by_distance_bin=tuple(
            int(value) for value in capacity_by_bin.tolist()
        ),
        degree_relaxed_optimistic_final_replacement_upper_bound=(
            optimistic_fraction
        ),
        degree_relaxed_optimistic_minimum_final_mean_um_at_threshold=(
            optimistic_mean
        ),
        degree_relaxed_optimistic_minimum_relative_mean_distance_increase_at_threshold=(
            optimistic_relative_increase
        ),
        swap_budget_can_reach_threshold=swap_budget_feasible,
        distance_bin_capacity_can_reach_threshold=(
            distance_capacity_feasible
        ),
        mean_distance_tolerance_can_reach_threshold=mean_feasible,
        degree_constraints_relaxed=True,
        passed=passed,
    )


def _canonical_pair(first: int, second: int) -> tuple[int, int]:
    return (
        (int(first), int(second))
        if first < second
        else (int(second), int(first))
    )


def _degree_distance_rewire_pairs(
    true_pairs: np.ndarray,
    coordinates: np.ndarray,
    *,
    seed: int,
    swaps_per_edge: float,
    distance_bins: int,
    max_attempts_per_swap: int,
    maximum_relative_mean_distance_change: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply deterministic, original-edge-excluding geometric 2-switches."""

    pairs = np.asarray(true_pairs, dtype=np.int64).copy()
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) < 2:
        raise MultiscaleGraphContractError(
            "Local rewiring requires at least two undirected pairs."
        )
    n_nodes = len(coordinates)
    original_pairs = pairs.copy()
    original_distances = np.linalg.norm(
        coordinates[pairs[:, 1]] - coordinates[pairs[:, 0]],
        axis=1,
    )
    boundaries = _distance_bin_boundaries(
        original_distances,
        int(distance_bins),
    )
    slot_bins = np.searchsorted(
        boundaries,
        original_distances,
        side="right",
    ).astype(np.int64, copy=False)
    edge_set = {tuple(pair) for pair in pairs.tolist()}
    original_edge_set = frozenset(edge_set)
    geometric_pairs = np.asarray(
        cKDTree(coordinates).query_pairs(
            LOCAL_MAX_DISTANCE_UM,
            eps=0.0,
            output_type="ndarray",
        ),
        dtype=np.int64,
    )
    geometric_pair_codes = (
        geometric_pairs[:, 0] * n_nodes + geometric_pairs[:, 1]
    )
    original_pair_codes = (
        original_pairs[:, 0] * n_nodes + original_pairs[:, 1]
    )
    nonoriginal_mask = ~np.isin(
        geometric_pair_codes,
        original_pair_codes,
        assume_unique=False,
    )
    nonoriginal_pairs = geometric_pairs[nonoriginal_mask]
    nonoriginal_distances = np.linalg.norm(
        coordinates[nonoriginal_pairs[:, 1]]
        - coordinates[nonoriginal_pairs[:, 0]],
        axis=1,
    )
    nonoriginal_bins = np.searchsorted(
        boundaries,
        nonoriginal_distances,
        side="right",
    ).astype(np.int64, copy=False)
    node_bin_slots: dict[tuple[int, int], set[int]] = {}
    for slot, (first, second) in enumerate(pairs.tolist()):
        bin_index = int(slot_bins[slot])
        node_bin_slots.setdefault((int(first), bin_index), set()).add(slot)
        node_bin_slots.setdefault((int(second), bin_index), set()).add(slot)

    target_swaps = int(round(float(swaps_per_edge) * len(pairs)))
    minimum_removed_relations = int(
        np.ceil(
            LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION
            * len(pairs)
        )
    )
    slot_holds_original_relation = np.ones(len(pairs), dtype=bool)
    removed_original_relations = 0
    maximum_attempts = max(
        1,
        target_swaps * int(max_attempts_per_swap),
    )
    rng = np.random.default_rng(int(seed))
    successful = 0
    attempts = 0
    original_distance_sum = float(original_distances.sum())
    current_distance_sum = original_distance_sum
    maximum_distance_sum_delta = (
        float(maximum_relative_mean_distance_change)
        * original_distance_sum
    )
    while successful < target_swaps and attempts < maximum_attempts:
        attempts += 1
        if len(nonoriginal_pairs) == 0:
            break
        candidate_index = int(rng.integers(len(nonoriginal_pairs)))
        new_first = tuple(
            int(value) for value in nonoriginal_pairs[candidate_index]
        )
        if new_first in edge_set:
            continue
        bin_index = int(nonoriginal_bins[candidate_index])
        a, c = new_first
        first_slots = sorted(node_bin_slots.get((a, bin_index), ()))
        second_slots = sorted(node_bin_slots.get((c, bin_index), ()))
        if not first_slots or not second_slots:
            continue
        first_start = int(rng.integers(len(first_slots)))
        second_start = int(rng.integers(len(second_slots)))
        accepted: (
            tuple[
                int,
                int,
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
                float,
            ]
            | None
        ) = None
        for first_offset in range(min(24, len(first_slots))):
            first_slot = int(
                first_slots[
                    (first_start + first_offset) % len(first_slots)
                ]
            )
            old_first = tuple(int(value) for value in pairs[first_slot])
            b = old_first[1] if old_first[0] == a else old_first[0]
            for second_offset in range(min(24, len(second_slots))):
                second_slot = int(
                    second_slots[
                        (second_start + second_offset)
                        % len(second_slots)
                    ]
                )
                if second_slot == first_slot:
                    continue
                old_second = tuple(
                    int(value) for value in pairs[second_slot]
                )
                d = (
                    old_second[1]
                    if old_second[0] == c
                    else old_second[0]
                )
                if len({a, b, c, d}) < 4:
                    continue
                if (
                    removed_original_relations
                    < minimum_removed_relations
                    and not (
                        slot_holds_original_relation[first_slot]
                        or slot_holds_original_relation[second_slot]
                    )
                ):
                    continue
                new_second = _canonical_pair(b, d)
                if (
                    new_first == new_second
                    or new_second in edge_set
                    or new_second in original_edge_set
                ):
                    continue
                second_distance = float(
                    np.linalg.norm(
                        coordinates[new_second[1]]
                        - coordinates[new_second[0]]
                    )
                )
                if (
                    second_distance > LOCAL_MAX_DISTANCE_UM
                    or int(
                        np.searchsorted(
                            boundaries,
                            second_distance,
                            side="right",
                        )
                    )
                    != bin_index
                ):
                    continue
                old_distance_sum = float(
                    np.linalg.norm(
                        coordinates[old_first[1]]
                        - coordinates[old_first[0]]
                    )
                    + np.linalg.norm(
                        coordinates[old_second[1]]
                        - coordinates[old_second[0]]
                    )
                )
                new_distance_sum = float(
                    nonoriginal_distances[candidate_index]
                    + second_distance
                )
                proposed_distance_sum = (
                    current_distance_sum
                    - old_distance_sum
                    + new_distance_sum
                )
                if (
                    abs(proposed_distance_sum - original_distance_sum)
                    > maximum_distance_sum_delta
                ):
                    continue
                accepted = (
                    first_slot,
                    second_slot,
                    old_first,
                    old_second,
                    new_first,
                    new_second,
                    proposed_distance_sum,
                )
                break
            if accepted is not None:
                break
        if accepted is None:
            continue
        (
            first_slot,
            second_slot,
            old_first,
            old_second,
            new_first,
            new_second,
            proposed_distance_sum,
        ) = accepted
        edge_set.remove(old_first)
        edge_set.remove(old_second)
        edge_set.add(new_first)
        edge_set.add(new_second)
        for node in old_first:
            node_bin_slots[(node, bin_index)].remove(first_slot)
        for node in old_second:
            node_bin_slots[(node, bin_index)].remove(second_slot)
        pairs[first_slot] = new_first
        pairs[second_slot] = new_second
        for node in new_first:
            node_bin_slots.setdefault((node, bin_index), set()).add(
                first_slot
            )
        for node in new_second:
            node_bin_slots.setdefault((node, bin_index), set()).add(
                second_slot
            )
        current_distance_sum = proposed_distance_sum
        removed_original_relations += int(
            slot_holds_original_relation[first_slot]
        )
        removed_original_relations += int(
            slot_holds_original_relation[second_slot]
        )
        slot_holds_original_relation[first_slot] = False
        slot_holds_original_relation[second_slot] = False
        successful += 1

    rewired_pairs = np.asarray(sorted(edge_set), dtype=np.int64)
    original_degree = np.bincount(
        original_pairs.ravel(),
        minlength=n_nodes,
    )
    rewired_degree = np.bincount(
        rewired_pairs.ravel(),
        minlength=n_nodes,
    )
    if not np.array_equal(original_degree, rewired_degree):
        raise MultiscaleGraphContractError(
            "Internal error: geometric 2-switch changed node degree."
        )
    rewired_distances = np.linalg.norm(
        coordinates[rewired_pairs[:, 1]]
        - coordinates[rewired_pairs[:, 0]],
        axis=1,
    )
    metadata: dict[str, object] = {
        "rewire_seed": int(seed),
        "rewire_target_swaps": int(target_swaps),
        "rewire_successful_swaps": int(successful),
        "rewire_attempts": int(attempts),
        "rewire_success_fraction": (
            float(successful / target_swaps)
            if target_swaps
            else 0.0
        ),
        "degree_preserved_exactly": True,
        "original_edge_distance_mean_um": float(
            original_distances.mean()
        ),
        "rewired_edge_distance_mean_um": float(
            rewired_distances.mean()
        ),
        "relative_edge_distance_mean_change": float(
            abs(
                rewired_distances.mean()
                - original_distances.mean()
            )
            / max(
                original_distances.mean(),
                np.finfo(np.float64).eps,
            )
        ),
        "distance_bin_boundaries_um": boundaries.tolist(),
        "rewire_reintroduced_original_undirected_edges": 0,
        "rewire_monotonic_removed_original_undirected_edges": int(
            removed_original_relations
        ),
        "proposal": (
            "candidate_driven_monotonic_original_edge_excluding_"
            "stratified_2switch_v3"
        ),
    }
    return rewired_pairs, metadata


def _rewire_local(
    local: ReceiverSortedScaleGraph,
    local_receiver_codes: np.ndarray,
    coordinates: np.ndarray,
    *,
    receiver_chunk_size: int,
    seed: int,
    swaps_per_edge: float,
    distance_bins: int,
    max_attempts_per_swap: int,
    maximum_relative_mean_distance_change: float,
) -> tuple[ReceiverSortedScaleGraph, LocalRewiringQC]:
    n_nodes = len(coordinates)
    true_pairs = _unique_undirected_pairs(
        local_receiver_codes,
        n_nodes=n_nodes,
    )
    target_swaps = int(round(float(swaps_per_edge) * len(true_pairs)))
    feasibility = _local_rewiring_feasibility(
        true_pairs,
        coordinates,
        distance_bins=int(distance_bins),
        target_swaps=target_swaps,
        maximum_relative_mean_distance_change=float(
            maximum_relative_mean_distance_change
        ),
    )
    if not feasibility.passed:
        raise MultiscaleGraphContractError(
            "The local rewiring contract is optimistically infeasible "
            "before exact-degree swaps; degree constraints were relaxed "
            "for this conclusive preflight: "
            + json.dumps(
                feasibility.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    rewired_pairs, metadata = _degree_distance_rewire_pairs(
        true_pairs,
        coordinates,
        seed=int(seed),
        swaps_per_edge=float(swaps_per_edge),
        distance_bins=int(distance_bins),
        max_attempts_per_swap=int(max_attempts_per_swap),
        maximum_relative_mean_distance_change=float(
            maximum_relative_mean_distance_change
        ),
    )
    rewired_directed = np.concatenate(
        [rewired_pairs, rewired_pairs[:, ::-1]],
        axis=0,
    ).T
    rewired_codes = _receiver_codes_from_directed_edges(
        rewired_directed,
        n_nodes=n_nodes,
    )
    if len(rewired_codes) != len(local_receiver_codes):
        raise MultiscaleGraphContractError(
            "Local rewiring changed the number of directed edges."
        )
    true_source, true_receiver, true_unique = _receiver_code_invariants(
        local_receiver_codes,
        n_nodes=n_nodes,
    )
    new_source, new_receiver, new_unique = _receiver_code_invariants(
        rewired_codes,
        n_nodes=n_nodes,
    )
    true_degree = np.bincount(true_receiver, minlength=n_nodes)
    rewired_degree = np.bincount(new_receiver, minlength=n_nodes)
    degree_preserved = bool(np.array_equal(true_degree, rewired_degree))
    if not degree_preserved:
        raise MultiscaleGraphContractError(
            "Local rewiring changed node degree."
        )
    true_distances = np.linalg.norm(
        coordinates[true_receiver[true_unique]]
        - coordinates[true_source[true_unique]],
        axis=1,
    )
    rewired_distances = np.linalg.norm(
        coordinates[new_receiver[new_unique]]
        - coordinates[new_source[new_unique]],
        axis=1,
    )
    boundaries = np.asarray(
        metadata.get("distance_bin_boundaries_um", ()),
        dtype=np.float64,
    )
    true_bin_counts = _distance_bin_counts(true_distances, boundaries)
    rewired_bin_counts = _distance_bin_counts(
        rewired_distances,
        boundaries,
    )
    bins_preserved = bool(
        np.array_equal(true_bin_counts, rewired_bin_counts)
    )
    relative_mean_change = float(
        abs(rewired_distances.mean() - true_distances.mean())
        / max(true_distances.mean(), np.finfo(np.float64).eps)
    )
    distance_tolerance_passed = bool(
        bins_preserved
        and relative_mean_change
        <= float(maximum_relative_mean_distance_change)
    )
    true_pair_codes = (
        true_pairs[:, 0] * n_nodes + true_pairs[:, 1]
    ).astype(np.int64, copy=False)
    rewired_pairs = _unique_undirected_pairs(
        rewired_codes,
        n_nodes=n_nodes,
    )
    rewired_pair_codes = (
        rewired_pairs[:, 0] * n_nodes + rewired_pairs[:, 1]
    ).astype(np.int64, copy=False)
    removed = int(
        len(
            np.setdiff1d(
                true_pair_codes,
                rewired_pair_codes,
                assume_unique=True,
            )
        )
    )
    added = int(
        len(
            np.setdiff1d(
                rewired_pair_codes,
                true_pair_codes,
                assume_unique=True,
            )
        )
    )
    topology_changed = removed > 0 and added > 0
    successful_swaps = int(metadata.get("rewire_successful_swaps", 0))
    reintroduced_original = int(
        metadata.get(
            "rewire_reintroduced_original_undirected_edges",
            -1,
        )
    )
    no_original_reintroduced = reintroduced_original == 0
    monotonic_removed = int(
        metadata.get(
            "rewire_monotonic_removed_original_undirected_edges",
            -1,
        )
    )
    if monotonic_removed != removed:
        raise MultiscaleGraphContractError(
            "Local rewiring history and final removed-relation count disagree."
        )
    if not no_original_reintroduced:
        raise MultiscaleGraphContractError(
            "Local rewiring reintroduced an original relation."
        )
    replacement_fraction = float(removed / len(true_pairs))
    replacement_passed = bool(
        replacement_fraction
        >= LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION
    )
    if successful_swaps <= 0 or not topology_changed:
        raise MultiscaleGraphContractError(
            "Local rewiring produced no topology-changing swap."
        )
    if not replacement_passed:
        raise MultiscaleGraphContractError(
            "Local rewiring failed the final relation-replacement gate: "
            f"observed={replacement_fraction:.12f}, "
            "required="
            f"{LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION:.12f}, "
            f"successful_swaps={successful_swaps}."
        )
    if not distance_tolerance_passed:
        raise MultiscaleGraphContractError(
            "Local rewiring did not preserve its distance strata/tolerance."
        )
    # _build_scale_graph regenerates all 17 raw geometry features from the new
    # endpoint coordinates and applies the true-local transform supplied here.
    rewired = _build_scale_graph(
        rewired_codes,
        name="rewired_local",
        topology_kind="degree_distance_rewired_local",
        coordinates=coordinates,
        k_cap=LOCAL_K_CAP,
        minimum_distance_um=0.0,
        minimum_distance_inclusive=True,
        maximum_distance_um=LOCAL_MAX_DISTANCE_UM,
        candidate_counts=None,
        directed_candidates_sha256=None,
        edge_attribute_mean=local.edge_attribute_mean,
        edge_attribute_scale=local.edge_attribute_scale,
        edge_standardization_source="true_local",
        receiver_chunk_size=int(receiver_chunk_size),
    )
    target_swaps = int(metadata.get("rewire_target_swaps", 0))
    attempts = int(metadata.get("rewire_attempts", 0))
    rewiring_qc = LocalRewiringQC(
        seed=int(seed),
        target_swaps=target_swaps,
        successful_swaps=successful_swaps,
        attempts=attempts,
        success_fraction=(
            float(successful_swaps / target_swaps)
            if target_swaps
            else 0.0
        ),
        topology_changed=topology_changed,
        removed_true_undirected_edges=removed,
        added_rewired_undirected_edges=added,
        reintroduced_original_undirected_edges=reintroduced_original,
        no_original_relations_reintroduced=no_original_reintroduced,
        minimum_final_relation_replacement_fraction=(
            LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION
        ),
        final_relation_replacement_fraction=replacement_fraction,
        final_relation_replacement_passed=replacement_passed,
        degree_preserved_exactly=degree_preserved,
        distance_bin_boundaries_um=tuple(
            float(value) for value in boundaries.tolist()
        ),
        true_distance_bin_counts=tuple(
            int(value) for value in true_bin_counts.tolist()
        ),
        rewired_distance_bin_counts=tuple(
            int(value) for value in rewired_bin_counts.tolist()
        ),
        distance_bin_counts_preserved_exactly=bins_preserved,
        true_edge_distance_mean_um=float(true_distances.mean()),
        rewired_edge_distance_mean_um=float(rewired_distances.mean()),
        relative_edge_distance_mean_change=relative_mean_change,
        maximum_relative_mean_distance_change=float(
            maximum_relative_mean_distance_change
        ),
        distance_tolerance_passed=distance_tolerance_passed,
        raw_geometry_recomputed_from_coordinates=True,
        edge_standardization_source="true_local",
        feasibility=feasibility,
    )
    return rewired, rewiring_qc


def _overlap_count(
    first: np.ndarray,
    second: np.ndarray,
) -> int:
    if len(first) > len(second):
        first, second = second, first
    positions = np.searchsorted(second, first)
    found = positions < len(second)
    if np.any(found):
        found[found] = second[positions[found]] == first[found]
    return int(np.sum(found))


def build_true_multiscale_graphs(
    coordinates_um: np.ndarray,
    *,
    query_chunk_size: int = 1024,
    receiver_chunk_size: int = 512,
    mutual_search_chunk_size: int = 4_000_000,
    workers: int = 1,
    epsilon: float = 1e-8,
) -> TrueMultiscaleGraphs:
    """Build only the true local and regional scales.

    This entry point has no rewiring parameters and never constructs,
    evaluates, or serializes a rewired-local graph.  It exists for campaigns
    whose null acts on sender states while retaining the measured topology.
    """

    if (
        isinstance(mutual_search_chunk_size, bool)
        or not isinstance(mutual_search_chunk_size, (int, np.integer))
        or int(mutual_search_chunk_size) <= 0
    ):
        raise MultiscaleGraphContractError(
            "mutual_search_chunk_size must be a positive integer."
        )
    coordinates = _validate_true_inputs(
        coordinates_um,
        query_chunk_size=query_chunk_size,
        receiver_chunk_size=receiver_chunk_size,
        workers=workers,
        epsilon=epsilon,
    )
    local, local_codes = _true_scale(
        coordinates,
        name="local",
        k_cap=LOCAL_K_CAP,
        minimum_distance_exclusive_um=None,
        maximum_distance_um=LOCAL_MAX_DISTANCE_UM,
        query_chunk_size=int(query_chunk_size),
        receiver_chunk_size=int(receiver_chunk_size),
        mutual_search_chunk_size=int(mutual_search_chunk_size),
        workers=int(workers),
        epsilon=float(epsilon),
    )
    regional, regional_codes = _true_scale(
        coordinates,
        name="regional",
        k_cap=REGIONAL_K_CAP,
        minimum_distance_exclusive_um=(
            REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
        ),
        maximum_distance_um=REGIONAL_MAX_DISTANCE_UM,
        query_chunk_size=int(query_chunk_size),
        receiver_chunk_size=int(receiver_chunk_size),
        mutual_search_chunk_size=int(mutual_search_chunk_size),
        workers=int(workers),
        epsilon=float(epsilon),
    )
    overlap = _overlap_count(local_codes, regional_codes)
    if overlap:
        raise MultiscaleGraphContractError(
            "Local and regional edge sets overlap."
        )
    graphs = (local, regional)
    qc = TrueMultiscaleGraphQC(
        local_regional_overlap_directed_edges=overlap,
        local_regional_disjoint=overlap == 0,
        all_graphs_symmetric=all(
            graph.qc.directed_edge_pairs_are_symmetric
            for graph in graphs
        ),
        all_graphs_receiver_sorted=all(
            graph.qc.receiver_sorted for graph in graphs
        ),
        all_graphs_loop_and_duplicate_free=all(
            graph.qc.self_loops == 0
            and graph.qc.duplicate_directed_edges == 0
            for graph in graphs
        ),
    )
    checksum_payload: dict[str, object] = {
        "schema": "true_multiscale_graph_bundle_v1",
        "fixed_scales": {
            "local": {
                "k_cap": LOCAL_K_CAP,
                "maximum_distance_um": LOCAL_MAX_DISTANCE_UM,
            },
            "regional": {
                "k_cap": REGIONAL_K_CAP,
                "minimum_distance_exclusive_um": (
                    REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
                ),
                "maximum_distance_um": REGIONAL_MAX_DISTANCE_UM,
            },
        },
        "graphs": {
            "local": local.checksums.graph_sha256,
            "regional": regional.checksums.graph_sha256,
        },
        "qc": qc.to_dict(),
    }
    checksums = TrueMultiscaleGraphChecksums(
        local_graph_sha256=local.checksums.graph_sha256,
        regional_graph_sha256=regional.checksums.graph_sha256,
        bundle_sha256=_payload_sha256(checksum_payload),
    )
    return TrueMultiscaleGraphs(
        local=local,
        regional=regional,
        qc=qc,
        checksums=checksums,
    )


def build_multiscale_graphs(
    coordinates_um: np.ndarray,
    *,
    query_chunk_size: int = 1024,
    receiver_chunk_size: int = 512,
    mutual_search_chunk_size: int = 4_000_000,
    workers: int = 1,
    epsilon: float = 1e-8,
    rewiring_seed: int = LOCAL_REWIRE_SEED,
    rewiring_swaps_per_edge: float = LOCAL_REWIRE_SWAPS_PER_EDGE,
    rewiring_distance_bins: int = LOCAL_REWIRE_DISTANCE_BINS,
    rewiring_max_attempts_per_swap: int = 100,
    maximum_relative_mean_distance_change: float = (
        LOCAL_REWIRE_MAX_RELATIVE_MEAN_DISTANCE_CHANGE
    ),
) -> MultiscaleGraphs:
    """Build the frozen local/regional graphs and rewired local control.

    Neighbor caps and distance scales are intentionally not arguments: the
    campaign contract fixes them.  Runtime-only chunk sizes and the declared
    rewiring diagnostics remain explicit and are captured by returned QC.
    """

    if (
        isinstance(mutual_search_chunk_size, bool)
        or not isinstance(mutual_search_chunk_size, (int, np.integer))
        or int(mutual_search_chunk_size) <= 0
    ):
        raise MultiscaleGraphContractError(
            "mutual_search_chunk_size must be a positive integer."
        )
    if isinstance(rewiring_seed, bool) or not isinstance(
        rewiring_seed,
        (int, np.integer),
    ):
        raise MultiscaleGraphContractError(
            "rewiring_seed must be an integer."
        )
    if int(rewiring_seed) != LOCAL_REWIRE_SEED:
        raise MultiscaleGraphContractError(
            f"This campaign fixes rewiring_seed={LOCAL_REWIRE_SEED}."
        )
    coordinates = _validate_inputs(
        coordinates_um,
        query_chunk_size=query_chunk_size,
        receiver_chunk_size=receiver_chunk_size,
        workers=workers,
        epsilon=epsilon,
        rewiring_swaps_per_edge=rewiring_swaps_per_edge,
        rewiring_distance_bins=rewiring_distance_bins,
        rewiring_max_attempts_per_swap=rewiring_max_attempts_per_swap,
        maximum_relative_mean_distance_change=(
            maximum_relative_mean_distance_change
        ),
    )
    local, local_codes = _true_scale(
        coordinates,
        name="local",
        k_cap=LOCAL_K_CAP,
        minimum_distance_exclusive_um=None,
        maximum_distance_um=LOCAL_MAX_DISTANCE_UM,
        query_chunk_size=int(query_chunk_size),
        receiver_chunk_size=int(receiver_chunk_size),
        mutual_search_chunk_size=int(mutual_search_chunk_size),
        workers=int(workers),
        epsilon=float(epsilon),
    )
    regional, regional_codes = _true_scale(
        coordinates,
        name="regional",
        k_cap=REGIONAL_K_CAP,
        minimum_distance_exclusive_um=(
            REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
        ),
        maximum_distance_um=REGIONAL_MAX_DISTANCE_UM,
        query_chunk_size=int(query_chunk_size),
        receiver_chunk_size=int(receiver_chunk_size),
        mutual_search_chunk_size=int(mutual_search_chunk_size),
        workers=int(workers),
        epsilon=float(epsilon),
    )
    overlap = _overlap_count(local_codes, regional_codes)
    if overlap:
        raise MultiscaleGraphContractError(
            "Local and regional edge sets overlap."
        )
    rewired_local, rewiring_qc = _rewire_local(
        local,
        local_codes,
        coordinates,
        receiver_chunk_size=int(receiver_chunk_size),
        seed=int(rewiring_seed),
        swaps_per_edge=float(rewiring_swaps_per_edge),
        distance_bins=int(rewiring_distance_bins),
        max_attempts_per_swap=int(rewiring_max_attempts_per_swap),
        maximum_relative_mean_distance_change=float(
            maximum_relative_mean_distance_change
        ),
    )
    graphs = (local, regional, rewired_local)
    qc = MultiscaleGraphQC(
        local_regional_overlap_directed_edges=overlap,
        local_regional_disjoint=overlap == 0,
        all_graphs_symmetric=all(
            graph.qc.directed_edge_pairs_are_symmetric
            for graph in graphs
        ),
        all_graphs_receiver_sorted=all(
            graph.qc.receiver_sorted for graph in graphs
        ),
        all_graphs_loop_and_duplicate_free=all(
            graph.qc.self_loops == 0
            and graph.qc.duplicate_directed_edges == 0
            for graph in graphs
        ),
        rewiring=rewiring_qc,
    )
    checksum_payload: dict[str, object] = {
        "schema": "multiscale_graph_bundle_v1",
        "fixed_scales": {
            "local": {
                "k_cap": LOCAL_K_CAP,
                "maximum_distance_um": LOCAL_MAX_DISTANCE_UM,
            },
            "regional": {
                "k_cap": REGIONAL_K_CAP,
                "minimum_distance_exclusive_um": (
                    REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
                ),
                "maximum_distance_um": REGIONAL_MAX_DISTANCE_UM,
            },
        },
        "graphs": {
            "local": local.checksums.graph_sha256,
            "regional": regional.checksums.graph_sha256,
            "rewired_local": rewired_local.checksums.graph_sha256,
        },
        "qc": qc.to_dict(),
    }
    checksums = MultiscaleGraphChecksums(
        local_graph_sha256=local.checksums.graph_sha256,
        regional_graph_sha256=regional.checksums.graph_sha256,
        rewired_local_graph_sha256=(
            rewired_local.checksums.graph_sha256
        ),
        bundle_sha256=_payload_sha256(checksum_payload),
    )
    return MultiscaleGraphs(
        local=local,
        regional=regional,
        rewired_local=rewired_local,
        qc=qc,
        checksums=checksums,
    )


def graph_receipt(graphs: MultiscaleGraphs) -> dict[str, Any]:
    """Return a JSON-safe graph receipt for campaign materialization."""

    return json.loads(
        json.dumps(
            {
                "schema": "multiscale_graph_receipt_v1",
                "fixed_contract": {
                    "local_k_cap": LOCAL_K_CAP,
                    "local_maximum_distance_um": LOCAL_MAX_DISTANCE_UM,
                    "regional_k_cap": REGIONAL_K_CAP,
                    "regional_minimum_distance_exclusive_um": (
                        REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
                    ),
                    "regional_maximum_distance_um": (
                        REGIONAL_MAX_DISTANCE_UM
                    ),
                    "default_rewiring_seed": LOCAL_REWIRE_SEED,
                    "minimum_final_relation_replacement_fraction": (
                        LOCAL_REWIRE_MIN_FINAL_RELATION_REPLACEMENT_FRACTION
                    ),
                    "original_relations_may_be_reintroduced": False,
                },
                "local": {
                    "qc": graphs.local.qc.to_dict(),
                    "checksums": graphs.local.checksums.to_dict(),
                },
                "regional": {
                    "qc": graphs.regional.qc.to_dict(),
                    "checksums": graphs.regional.checksums.to_dict(),
                },
                "rewired_local": {
                    "qc": graphs.rewired_local.qc.to_dict(),
                    "checksums": graphs.rewired_local.checksums.to_dict(),
                },
                "bundle_qc": graphs.qc.to_dict(),
                "bundle_checksums": graphs.checksums.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def true_graph_receipt(
    graphs: TrueMultiscaleGraphs,
) -> dict[str, Any]:
    """Return a JSON-safe receipt that cannot contain a rewired graph."""

    return json.loads(
        json.dumps(
            {
                "schema": "true_multiscale_graph_receipt_v1",
                "fixed_contract": {
                    "local_k_cap": LOCAL_K_CAP,
                    "local_maximum_distance_um": (
                        LOCAL_MAX_DISTANCE_UM
                    ),
                    "regional_k_cap": REGIONAL_K_CAP,
                    "regional_minimum_distance_exclusive_um": (
                        REGIONAL_MIN_DISTANCE_EXCLUSIVE_UM
                    ),
                    "regional_maximum_distance_um": (
                        REGIONAL_MAX_DISTANCE_UM
                    ),
                    "rewired_graph_constructed": False,
                },
                "local": {
                    "qc": graphs.local.qc.to_dict(),
                    "checksums": graphs.local.checksums.to_dict(),
                },
                "regional": {
                    "qc": graphs.regional.qc.to_dict(),
                    "checksums": graphs.regional.checksums.to_dict(),
                },
                "bundle_qc": graphs.qc.to_dict(),
                "bundle_checksums": graphs.checksums.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
