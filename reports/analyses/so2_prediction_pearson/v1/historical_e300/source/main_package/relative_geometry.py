"""Deterministic radial graphs and invariant relative positional geometry.

This module implements the coordinate-only geometry contract used by the
relative-geometric QKV model.  Coordinates are used only while constructing
the graph, local axial orientation tensors, and edge-relative positional
features.  They are not node covariates.

Edges follow the repository convention ``edge_index[0] = source`` and
``edge_index[1] = receiver``.  Stored graph edges are sorted receiver-major,
then source-major.  The 64 radial basis functions use centers linearly spaced
from 0 through 500 micrometres and a width equal to one center spacing.  Thus
adjacent bases intersect at ``exp(-1/2)`` before multiplication by the smooth
cosine cutoff envelope.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree


DEFAULT_RADIAL_SHELL_BOUNDS_UM = (50.0, 150.0, 300.0, 500.0)
DEFAULT_RADIAL_SHELL_QUOTAS = (48, 64, 48, 40)
DEFAULT_NOMINAL_K = sum(DEFAULT_RADIAL_SHELL_QUOTAS)
DEFAULT_MAX_DISTANCE_UM = DEFAULT_RADIAL_SHELL_BOUNDS_UM[-1]
DEFAULT_ORIENTATION_RADIUS_UM = 150.0
DEFAULT_ORIENTATION_NEIGHBOR_CAP = 64
DEFAULT_ORIENTATION_SIGMA_UM = 75.0
DEFAULT_ORIENTATION_MIN_NEIGHBORS = 2
DEFAULT_RBF_COUNT = 64
DEFAULT_RBF_WIDTH_UM = DEFAULT_MAX_DISTANCE_UM / (DEFAULT_RBF_COUNT - 1)
RELATIVE_GEOMETRY_FEATURE_NAMES = (
    *(f"distance_rbf_{index:02d}" for index in range(DEFAULT_RBF_COUNT)),
    "receiver_axial_alignment",
    "source_axial_alignment",
    "orientation_tensor_alignment",
    "receiver_anisotropy",
    "source_anisotropy",
    "distance_over_cutoff",
)
RELATIVE_GEOMETRY_DIM = len(RELATIVE_GEOMETRY_FEATURE_NAMES)
_HASH_CHUNK_BYTES = 64 * 1024 * 1024


class RelativeGeometryContractError(ValueError):
    """Raised when topology or geometry violates the relative-position contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _update_raw_bytes(digest: Any, values: np.ndarray) -> None:
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    view = memoryview(array).cast("B")
    for start in range(0, view.nbytes, _HASH_CHUNK_BYTES):
        digest.update(view[start : start + _HASH_CHUNK_BYTES])


def _array_sha256(name: str, values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {"name": name, "shape": list(array.shape), "dtype": array.dtype.str}
        )
    )
    _update_raw_bytes(digest, array)
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_coordinates(
    coordinates_um: np.ndarray,
    *,
    minimum_nodes: int = 1,
) -> np.ndarray:
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise RelativeGeometryContractError(
            "coordinates_um must have shape [n_nodes, 2]."
        )
    if len(coordinates) < minimum_nodes or not np.isfinite(coordinates).all():
        raise RelativeGeometryContractError(
            f"coordinates_um must contain at least {minimum_nodes} finite nodes."
        )
    if len(coordinates) > int(np.sqrt(np.iinfo(np.int64).max)):
        raise RelativeGeometryContractError(
            "The node count cannot be encoded safely in signed int64."
        )
    return coordinates


def _validate_positive_integer(value: int, *, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise RelativeGeometryContractError(f"{name} must be a positive integer.")
    return int(value)


def _canonical_group_labels(
    labels: Sequence[object] | np.ndarray | None,
    n_nodes: int,
) -> np.ndarray:
    if labels is None:
        return np.zeros(n_nodes, dtype=np.int64)
    raw = np.asarray(labels, dtype=object)
    if raw.ndim != 1 or len(raw) != n_nodes:
        raise RelativeGeometryContractError(
            "group_labels must be one-dimensional and aligned to coordinates."
        )
    if any(value is None for value in raw.tolist()):
        raise RelativeGeometryContractError(
            "group_labels cannot contain missing values."
        )
    # Sorting repr strings makes group traversal deterministic even for mixed
    # scalar types, while all topology tie breaks still use global node index.
    canonical = np.asarray([str(value) for value in raw.tolist()], dtype="U256")
    return canonical


def _validate_shell_contract(
    bounds_um: Sequence[float],
    quotas: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(tuple(bounds_um), dtype=np.float64)
    quota = np.asarray(tuple(quotas), dtype=np.int64)
    if (
        bounds.ndim != 1
        or quota.ndim != 1
        or len(bounds) == 0
        or len(bounds) != len(quota)
        or not np.isfinite(bounds).all()
        or np.any(bounds <= 0)
        or np.any(bounds[1:] <= bounds[:-1])
        or np.any(quota < 0)
        or int(quota.sum()) <= 0
    ):
        raise RelativeGeometryContractError(
            "Radial shell bounds and quotas must be finite, ordered, aligned, "
            "and have a positive total quota."
        )
    return bounds, quota


@dataclass(frozen=True)
class RadialStratifiedGraphQC:
    """Topology, physical-range, and long-range-coverage audit."""

    n_nodes: int
    n_groups: int
    nominal_pre_symmetrization_degree: int
    n_directed_candidates: int
    pre_sym_in_degree_min: int
    pre_sym_in_degree_mean: float
    pre_sym_in_degree_median: float
    pre_sym_in_degree_max: int
    n_directed_edges: int
    in_degree_min: int
    in_degree_mean: float
    in_degree_median: float
    in_degree_max: int
    edge_distance_p01_um: float | None
    edge_distance_p05_um: float | None
    edge_distance_p25_um: float | None
    edge_distance_p50_um: float | None
    edge_distance_p75_um: float | None
    edge_distance_p95_um: float | None
    edge_distance_p99_um: float | None
    edge_distance_max_um: float | None
    duplicate_directed_edges_removed: int
    self_candidates_removed: int
    zero_distance_nonself_candidates_excluded: int
    self_loops: int
    cross_group_edges: int
    directed_edge_pairs_are_symmetric: bool
    receiver_major_canonical_order: bool
    coverage_eligible_cells: int
    coverage_eligible_cells_with_300_500_um_edge: int
    eligible_long_range_coverage_fraction: float | None
    long_range_coverage_gate_min_fraction: float
    long_range_coverage_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RadialStratifiedGraphChecksums:
    coordinates_um_sha256: str
    directed_candidates_sha256: str
    receiver_major_edge_index_sha256: str
    graph_parameters_sha256: str
    graph_sha256: str

    def __post_init__(self) -> None:
        if not all(_is_sha256(value) for value in asdict(self).values()):
            raise RelativeGeometryContractError(
                "Graph checksums must be SHA-256 values."
            )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RadialStratifiedGraph:
    """A bidirectional-union graph with receiver-major canonical edges."""

    n_nodes: int
    edge_index: np.ndarray = field(repr=False)
    shell_bounds_um: tuple[float, ...]
    shell_quotas: tuple[int, ...]
    qc: RadialStratifiedGraphQC
    checksums: RadialStratifiedGraphChecksums

    def __post_init__(self) -> None:
        edges = np.asarray(self.edge_index)
        if edges.shape != (2, self.qc.n_directed_edges):
            raise RelativeGeometryContractError(
                "edge_index must have shape [2, n_directed_edges]."
            )
        if edges.dtype.kind not in "iu":
            raise RelativeGeometryContractError("edge_index must be integral.")
        if edges.size:
            source, receiver = edges
            if source.min() < 0 or receiver.min() < 0:
                raise RelativeGeometryContractError("edge_index cannot be negative.")
            if source.max() >= self.n_nodes or receiver.max() >= self.n_nodes:
                raise RelativeGeometryContractError(
                    "edge_index contains an invalid node."
                )
            if np.any(source == receiver):
                raise RelativeGeometryContractError("Self-loops are prohibited.")
            if np.any(receiver[1:] < receiver[:-1]):
                raise RelativeGeometryContractError(
                    "edge_index is not receiver-major sorted."
                )
            same_receiver = receiver[1:] == receiver[:-1]
            if np.any(source[1:][same_receiver] <= source[:-1][same_receiver]):
                raise RelativeGeometryContractError(
                    "Sources must be strictly increasing within each receiver."
                )

    @property
    def nominal_k(self) -> int:
        return int(sum(self.shell_quotas))


def _select_receiver_sources(
    *,
    receiver: int,
    receiver_coordinate: np.ndarray,
    group_global_indices: np.ndarray,
    tree: cKDTree,
    coordinates: np.ndarray,
    shell_bounds: np.ndarray,
    shell_quotas: np.ndarray,
) -> tuple[np.ndarray, bool, int, int]:
    """Select one receiver's sources without retaining cohort-wide radius lists."""

    local_candidates = np.asarray(
        tree.query_ball_point(
            receiver_coordinate,
            r=float(shell_bounds[-1]),
            eps=0.0,
            workers=1,
        ),
        dtype=np.int64,
    )
    sources = group_global_indices[local_candidates]
    self_count = int(np.sum(sources == receiver))
    sources = sources[sources != receiver]
    if not len(sources):
        return np.empty(0, dtype=np.int64), False, self_count, 0

    delta = coordinates[sources] - receiver_coordinate
    distances = np.linalg.norm(delta, axis=1)
    zero_distance = int(np.sum(distances == 0.0))
    valid = (distances > 0.0) & (distances <= shell_bounds[-1])
    sources = sources[valid]
    distances = distances[valid]
    if not len(sources):
        return np.empty(0, dtype=np.int64), False, self_count, zero_distance

    order = np.lexsort((sources, distances))
    sources = sources[order]
    distances = distances[order]
    shell_index = np.searchsorted(shell_bounds, distances, side="left")
    selected = np.zeros(len(sources), dtype=bool)
    for index, quota in enumerate(shell_quotas.tolist()):
        if quota == 0:
            continue
        positions = np.flatnonzero(shell_index == index)
        selected[positions[:quota]] = True

    nominal_k = int(shell_quotas.sum())
    refill = nominal_k - int(selected.sum())
    if refill > 0:
        remaining = np.flatnonzero(~selected)
        selected[remaining[:refill]] = True

    selected_sources = sources[selected]
    coverage_eligible = bool(
        np.any(
            (distances > DEFAULT_RADIAL_SHELL_BOUNDS_UM[-2])
            & (distances <= shell_bounds[-1])
        )
    )
    # Canonical pre-symmetrization encoding uses ascending source indices within
    # a receiver.  Distance/source ordering has already determined membership.
    return (
        np.sort(selected_sources),
        coverage_eligible,
        self_count,
        zero_distance,
    )


def radial_stratified_knn(
    coordinates_um: np.ndarray,
    *,
    shell_bounds_um: Sequence[float] = DEFAULT_RADIAL_SHELL_BOUNDS_UM,
    shell_quotas: Sequence[int] = DEFAULT_RADIAL_SHELL_QUOTAS,
    receiver_query_chunk_size: int = 512,
    group_labels: Sequence[object] | np.ndarray | None = None,
    long_range_coverage_gate_min_fraction: float = 1.0,
    fail_on_coverage_gate: bool = True,
) -> RadialStratifiedGraph:
    """Build an exact deterministic radial-stratified bidirectional graph.

    Radius searches are issued one receiver at a time inside bounded receiver
    chunks.  This intentionally avoids ``query_ball_point`` over the complete
    cohort, whose Python list-of-lists representation can dominate memory.
    Missing shell capacity is refilled by globally nearest remaining candidates
    after every non-empty shell has received its reserved quota.
    """

    coordinates = _validate_coordinates(coordinates_um, minimum_nodes=2)
    bounds, quotas = _validate_shell_contract(shell_bounds_um, shell_quotas)
    receiver_query_chunk_size = _validate_positive_integer(
        receiver_query_chunk_size, name="receiver_query_chunk_size"
    )
    if not np.isfinite(long_range_coverage_gate_min_fraction) or not (
        0.0 <= long_range_coverage_gate_min_fraction <= 1.0
    ):
        raise RelativeGeometryContractError(
            "long_range_coverage_gate_min_fraction must be in [0, 1]."
        )
    if not np.isclose(bounds[-1], DEFAULT_MAX_DISTANCE_UM, rtol=0.0, atol=0.0):
        raise RelativeGeometryContractError(
            "The relative-geometric graph contract requires a 500 um outer range."
        )
    if len(bounds) != 4 or not np.array_equal(
        bounds, np.asarray(DEFAULT_RADIAL_SHELL_BOUNDS_UM)
    ):
        raise RelativeGeometryContractError(
            "The graph contract requires radial shells (0,50], (50,150], "
            "(150,300], and (300,500] um."
        )
    if len(quotas) != 4 or not np.array_equal(
        quotas, np.asarray(DEFAULT_RADIAL_SHELL_QUOTAS)
    ):
        raise RelativeGeometryContractError(
            "The graph contract requires radial quotas 48, 64, 48, and 40."
        )

    n_nodes = len(coordinates)
    groups = _canonical_group_labels(group_labels, n_nodes)
    candidate_code_chunks: list[np.ndarray] = []
    candidate_counts = np.zeros(n_nodes, dtype=np.int64)
    coverage_eligible = np.zeros(n_nodes, dtype=bool)
    self_candidates_removed = 0
    zero_distance_candidates = 0

    for group in np.unique(groups):
        group_indices = np.flatnonzero(groups == group).astype(np.int64, copy=False)
        tree = cKDTree(coordinates[group_indices])
        for chunk_start in range(0, len(group_indices), receiver_query_chunk_size):
            chunk_stop = min(
                chunk_start + receiver_query_chunk_size, len(group_indices)
            )
            chunk_codes: list[np.ndarray] = []
            for receiver in group_indices[chunk_start:chunk_stop].tolist():
                sources, eligible, self_count, zero_count = _select_receiver_sources(
                    receiver=int(receiver),
                    receiver_coordinate=coordinates[receiver],
                    group_global_indices=group_indices,
                    tree=tree,
                    coordinates=coordinates,
                    shell_bounds=bounds,
                    shell_quotas=quotas,
                )
                candidate_counts[receiver] = len(sources)
                coverage_eligible[receiver] = eligible
                self_candidates_removed += self_count
                zero_distance_candidates += zero_count
                if len(sources):
                    chunk_codes.append(receiver * n_nodes + sources)
            if chunk_codes:
                candidate_code_chunks.append(np.concatenate(chunk_codes))

    if candidate_code_chunks:
        candidate_codes = np.sort(np.concatenate(candidate_code_chunks))
    else:
        candidate_codes = np.empty(0, dtype=np.int64)
    if len(candidate_codes) != len(np.unique(candidate_codes)):
        raise RelativeGeometryContractError(
            "Pre-symmetrization selection unexpectedly contains duplicate edges."
        )

    candidate_receiver = candidate_codes // n_nodes
    candidate_source = candidate_codes - candidate_receiver * n_nodes
    reverse_codes = candidate_source * n_nodes + candidate_receiver
    union_input = np.concatenate([candidate_codes, reverse_codes])
    edge_codes = np.unique(union_input)
    duplicate_edges_removed = int(len(union_input) - len(edge_codes))
    receiver = edge_codes // n_nodes
    source = edge_codes - receiver * n_nodes
    edge_index = np.vstack([source, receiver]).astype(np.int64, copy=False)

    if len(edge_codes):
        edge_distances = np.linalg.norm(
            coordinates[source] - coordinates[receiver], axis=1
        )
        if np.any(edge_distances <= 0.0) or np.any(edge_distances > bounds[-1]):
            raise RelativeGeometryContractError(
                "The final graph contains a zero-length or out-of-range edge."
            )
        cross_group_edges = int(np.sum(groups[source] != groups[receiver]))
        reverse_edge_codes = source * n_nodes + receiver
        symmetric = bool(np.array_equal(np.sort(reverse_edge_codes), edge_codes))
        final_long_range = np.bincount(
            receiver[(edge_distances > bounds[-2]) & (edge_distances <= bounds[-1])],
            minlength=n_nodes,
        ) > 0
        quantiles = np.quantile(
            edge_distances, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        distance_max: float | None = float(edge_distances.max())
    else:
        edge_distances = np.empty(0, dtype=np.float64)
        cross_group_edges = 0
        symmetric = True
        final_long_range = np.zeros(n_nodes, dtype=bool)
        quantiles = np.full(7, np.nan, dtype=np.float64)
        distance_max = None
    if cross_group_edges:
        raise RelativeGeometryContractError(
            "Graph construction created cross-group edges."
        )

    eligible_count = int(coverage_eligible.sum())
    covered_eligible_count = int(np.sum(coverage_eligible & final_long_range))
    coverage_fraction = (
        covered_eligible_count / eligible_count if eligible_count else None
    )
    coverage_passed = bool(
        coverage_fraction is None
        or coverage_fraction + np.finfo(np.float64).eps
        >= long_range_coverage_gate_min_fraction
    )
    if fail_on_coverage_gate and not coverage_passed:
        raise RelativeGeometryContractError(
            "Eligible 300-500 um coverage failed: "
            f"{covered_eligible_count}/{eligible_count} eligible receivers have "
            "a selected long-range edge."
        )

    degree = np.bincount(receiver, minlength=n_nodes).astype(np.int64)
    canonical_order = bool(
        not len(edge_codes)
        or (
            np.all(receiver[1:] >= receiver[:-1])
            and np.all(edge_codes[1:] > edge_codes[:-1])
        )
    )
    qc = RadialStratifiedGraphQC(
        n_nodes=n_nodes,
        n_groups=int(len(np.unique(groups))),
        nominal_pre_symmetrization_degree=int(quotas.sum()),
        n_directed_candidates=int(len(candidate_codes)),
        pre_sym_in_degree_min=int(candidate_counts.min()),
        pre_sym_in_degree_mean=float(candidate_counts.mean()),
        pre_sym_in_degree_median=float(np.median(candidate_counts)),
        pre_sym_in_degree_max=int(candidate_counts.max(initial=0)),
        n_directed_edges=int(len(edge_codes)),
        in_degree_min=int(degree.min()),
        in_degree_mean=float(degree.mean()),
        in_degree_median=float(np.median(degree)),
        in_degree_max=int(degree.max(initial=0)),
        edge_distance_p01_um=(float(quantiles[0]) if len(edge_distances) else None),
        edge_distance_p05_um=(float(quantiles[1]) if len(edge_distances) else None),
        edge_distance_p25_um=(float(quantiles[2]) if len(edge_distances) else None),
        edge_distance_p50_um=(float(quantiles[3]) if len(edge_distances) else None),
        edge_distance_p75_um=(float(quantiles[4]) if len(edge_distances) else None),
        edge_distance_p95_um=(float(quantiles[5]) if len(edge_distances) else None),
        edge_distance_p99_um=(float(quantiles[6]) if len(edge_distances) else None),
        edge_distance_max_um=distance_max,
        duplicate_directed_edges_removed=duplicate_edges_removed,
        self_candidates_removed=int(self_candidates_removed),
        zero_distance_nonself_candidates_excluded=int(zero_distance_candidates),
        self_loops=int(np.sum(source == receiver)),
        cross_group_edges=cross_group_edges,
        directed_edge_pairs_are_symmetric=symmetric,
        receiver_major_canonical_order=canonical_order,
        coverage_eligible_cells=eligible_count,
        coverage_eligible_cells_with_300_500_um_edge=covered_eligible_count,
        eligible_long_range_coverage_fraction=coverage_fraction,
        long_range_coverage_gate_min_fraction=float(
            long_range_coverage_gate_min_fraction
        ),
        long_range_coverage_gate_passed=coverage_passed,
    )
    if qc.self_loops or not qc.directed_edge_pairs_are_symmetric or not canonical_order:
        raise RelativeGeometryContractError(
            "The final graph violates loop, symmetry, or canonical-order invariants."
        )

    parameter_payload: dict[str, object] = {
        "policy": "radial_stratified_knn",
        "shell_intervals": ["(0,50]", "(50,150]", "(150,300]", "(300,500]"],
        "shell_bounds_um": bounds.tolist(),
        "shell_quotas": quotas.tolist(),
        "nominal_k": int(quotas.sum()),
        "refill": "nearest remaining by exact distance then global source index",
        "tie_break": "exact_distance_then_global_source_index",
        "symmetry": "bidirectional_union",
        "self_loops": False,
        "edge_order": "receiver_major_then_source_major",
        "coverage_gate_min_fraction": float(
            long_range_coverage_gate_min_fraction
        ),
    }
    coordinates_checksum = _array_sha256("coordinates_um", coordinates)
    candidate_checksum = _array_sha256(
        "receiver_major_directed_candidate_codes", candidate_codes
    )
    edge_checksum = _array_sha256("receiver_major_edge_index", edge_index)
    parameters_checksum = _payload_sha256(parameter_payload)
    graph_checksum = _payload_sha256(
        {
            "coordinates_um_sha256": coordinates_checksum,
            "directed_candidates_sha256": candidate_checksum,
            "receiver_major_edge_index_sha256": edge_checksum,
            "graph_parameters_sha256": parameters_checksum,
        }
    )
    return RadialStratifiedGraph(
        n_nodes=n_nodes,
        edge_index=edge_index,
        shell_bounds_um=tuple(float(value) for value in bounds.tolist()),
        shell_quotas=tuple(int(value) for value in quotas.tolist()),
        qc=qc,
        checksums=RadialStratifiedGraphChecksums(
            coordinates_um_sha256=coordinates_checksum,
            directed_candidates_sha256=candidate_checksum,
            receiver_major_edge_index_sha256=edge_checksum,
            graph_parameters_sha256=parameters_checksum,
            graph_sha256=graph_checksum,
        ),
    )


@dataclass(frozen=True)
class LocalOrientationTensors:
    """Normalized traceless axial tensors derived only from local coordinates."""

    tensors: np.ndarray = field(repr=False)
    anisotropy: np.ndarray = field(repr=False)
    selected_neighbor_counts: np.ndarray = field(repr=False)
    radius_um: float
    neighbor_cap: int
    sigma_um: float
    minimum_neighbors: int
    insufficient_neighbor_cells: int
    checksum_sha256: str

    def __post_init__(self) -> None:
        tensors = np.asarray(self.tensors)
        anisotropy = np.asarray(self.anisotropy)
        counts = np.asarray(self.selected_neighbor_counts)
        if tensors.ndim != 3 or tensors.shape[1:] != (2, 2):
            raise RelativeGeometryContractError(
                "Orientation tensors must have shape [n_nodes, 2, 2]."
            )
        if anisotropy.shape != (len(tensors),) or counts.shape != (len(tensors),):
            raise RelativeGeometryContractError(
                "Orientation anisotropy and counts must align to nodes."
            )
        if (
            not np.isfinite(tensors).all()
            or not np.isfinite(anisotropy).all()
            or np.any(anisotropy < 0)
            or counts.dtype.kind not in "iu"
        ):
            raise RelativeGeometryContractError("Orientation outputs are invalid.")
        if not _is_sha256(self.checksum_sha256):
            raise RelativeGeometryContractError("Orientation checksum is invalid.")


def local_orientation_tensors(
    coordinates_um: np.ndarray,
    *,
    radius_um: float = DEFAULT_ORIENTATION_RADIUS_UM,
    neighbor_cap: int = DEFAULT_ORIENTATION_NEIGHBOR_CAP,
    sigma_um: float = DEFAULT_ORIENTATION_SIGMA_UM,
    minimum_neighbors: int = DEFAULT_ORIENTATION_MIN_NEIGHBORS,
    receiver_query_chunk_size: int = 512,
    group_labels: Sequence[object] | np.ndarray | None = None,
    epsilon: float = 1e-12,
) -> LocalOrientationTensors:
    """Derive local axial orientation tensors with deterministic capped queries."""

    coordinates = _validate_coordinates(coordinates_um)
    neighbor_cap = _validate_positive_integer(neighbor_cap, name="neighbor_cap")
    minimum_neighbors = _validate_positive_integer(
        minimum_neighbors, name="minimum_neighbors"
    )
    receiver_query_chunk_size = _validate_positive_integer(
        receiver_query_chunk_size, name="receiver_query_chunk_size"
    )
    if minimum_neighbors > neighbor_cap:
        raise RelativeGeometryContractError(
            "minimum_neighbors cannot exceed neighbor_cap."
        )
    for value, name in (
        (radius_um, "radius_um"),
        (sigma_um, "sigma_um"),
        (epsilon, "epsilon"),
    ):
        if not np.isfinite(value) or value <= 0:
            raise RelativeGeometryContractError(f"{name} must be positive and finite.")

    n_nodes = len(coordinates)
    groups = _canonical_group_labels(group_labels, n_nodes)
    tensors = np.zeros((n_nodes, 2, 2), dtype=np.float64)
    anisotropy = np.zeros(n_nodes, dtype=np.float64)
    selected_counts = np.zeros(n_nodes, dtype=np.int64)
    identity = np.eye(2, dtype=np.float64)

    for group in np.unique(groups):
        group_indices = np.flatnonzero(groups == group).astype(np.int64, copy=False)
        tree = cKDTree(coordinates[group_indices])
        for chunk_start in range(0, len(group_indices), receiver_query_chunk_size):
            chunk_stop = min(
                chunk_start + receiver_query_chunk_size, len(group_indices)
            )
            for receiver in group_indices[chunk_start:chunk_stop].tolist():
                local_candidates = np.asarray(
                    tree.query_ball_point(
                        coordinates[receiver], r=radius_um, eps=0.0, workers=1
                    ),
                    dtype=np.int64,
                )
                sources = group_indices[local_candidates]
                sources = sources[sources != receiver]
                if len(sources):
                    delta = coordinates[sources] - coordinates[receiver]
                    distance = np.linalg.norm(delta, axis=1)
                    valid = (distance > 0.0) & (distance <= radius_um)
                    sources = sources[valid]
                    delta = delta[valid]
                    distance = distance[valid]
                    order = np.lexsort((sources, distance))
                    take = order[:neighbor_cap]
                    delta = delta[take]
                    distance = distance[take]
                else:
                    delta = np.empty((0, 2), dtype=np.float64)
                    distance = np.empty(0, dtype=np.float64)
                selected_counts[receiver] = len(distance)
                if len(distance) < minimum_neighbors:
                    continue
                weight = np.exp(-0.5 * np.square(distance / sigma_um))
                covariance = np.einsum(
                    "n,na,nb->ab", weight, delta, delta, optimize=True
                ) / (float(weight.sum()) + epsilon)
                trace = float(np.trace(covariance))
                if not np.isfinite(trace) or trace <= epsilon:
                    continue
                tensor = covariance / (trace + epsilon) - 0.5 * identity
                tensor = 0.5 * (tensor + tensor.T)
                tensors[receiver] = tensor
                anisotropy[receiver] = np.sqrt(
                    max(0.0, 2.0 * float(np.trace(tensor @ tensor)))
                )

    insufficient = int(np.sum(selected_counts < minimum_neighbors))
    parameters = {
        "method": "coordinate_local_covariance_axial_tensor",
        "radius_um": float(radius_um),
        "neighbor_cap": neighbor_cap,
        "sigma_um": float(sigma_um),
        "minimum_neighbors": minimum_neighbors,
        "epsilon": float(epsilon),
        "tie_break": "exact_distance_then_global_source_index",
    }
    checksum = _payload_sha256(
        {
            "parameters_sha256": _payload_sha256(parameters),
            "tensors_sha256": _array_sha256("orientation_tensors", tensors),
            "anisotropy_sha256": _array_sha256("orientation_anisotropy", anisotropy),
            "selected_neighbor_counts_sha256": _array_sha256(
                "orientation_selected_neighbor_counts", selected_counts
            ),
        }
    )
    return LocalOrientationTensors(
        tensors=tensors,
        anisotropy=anisotropy,
        selected_neighbor_counts=selected_counts,
        radius_um=float(radius_um),
        neighbor_cap=neighbor_cap,
        sigma_um=float(sigma_um),
        minimum_neighbors=minimum_neighbors,
        insufficient_neighbor_cells=insufficient,
        checksum_sha256=checksum,
    )


def smooth_cutoff_envelope(
    distances_um: np.ndarray,
    *,
    maximum_distance_um: float = DEFAULT_MAX_DISTANCE_UM,
) -> np.ndarray:
    """Return a C1 cosine envelope that is exactly zero at the range limit."""

    distances = np.asarray(distances_um, dtype=np.float64)
    if not np.isfinite(distances).all() or np.any(distances < 0.0):
        raise RelativeGeometryContractError(
            "RBF distances must be finite and non-negative."
        )
    if not np.isfinite(maximum_distance_um) or maximum_distance_um <= 0:
        raise RelativeGeometryContractError(
            "maximum_distance_um must be positive and finite."
        )
    if np.any(distances > maximum_distance_um):
        raise RelativeGeometryContractError(
            "An edge distance exceeds the relative-geometry range; clipping is "
            "prohibited."
        )
    return 0.5 * (np.cos(np.pi * distances / maximum_distance_um) + 1.0)


def gaussian_radial_basis(
    distances_um: np.ndarray,
    *,
    count: int = DEFAULT_RBF_COUNT,
    maximum_distance_um: float = DEFAULT_MAX_DISTANCE_UM,
    width_um: float | None = None,
) -> np.ndarray:
    """Encode distances with overlapping Gaussian bases and a smooth cutoff."""

    count = _validate_positive_integer(count, name="count")
    if count != DEFAULT_RBF_COUNT or maximum_distance_um != DEFAULT_MAX_DISTANCE_UM:
        raise RelativeGeometryContractError(
            "The production relative-geometry contract requires 64 RBFs over 0-500 um."
        )
    distances = np.asarray(distances_um, dtype=np.float64)
    envelope = smooth_cutoff_envelope(
        distances, maximum_distance_um=maximum_distance_um
    )
    centers = np.linspace(0.0, maximum_distance_um, count, dtype=np.float64)
    if width_um is None:
        width_um = float(centers[1] - centers[0])
    if not np.isfinite(width_um) or width_um <= 0:
        raise RelativeGeometryContractError("width_um must be positive and finite.")
    rbf = np.exp(
        -0.5 * np.square((distances[..., None] - centers) / float(width_um))
    )
    return rbf * envelope[..., None]


def _validate_edge_index(edge_index: np.ndarray, *, n_nodes: int) -> np.ndarray:
    raw = np.asarray(edge_index)
    if raw.ndim != 2 or raw.shape[0] != 2 or raw.dtype.kind not in "iu":
        raise RelativeGeometryContractError(
            "edge_index must be an integral array with shape [2, n_edges]."
        )
    edges = raw.astype(np.int64, copy=False)
    if edges.size:
        if edges.min() < 0 or edges.max() >= n_nodes:
            raise RelativeGeometryContractError("edge_index contains an invalid node.")
        if np.any(edges[0] == edges[1]):
            raise RelativeGeometryContractError(
                "Relative geometry prohibits self-loops."
            )
    return edges


def invariant_relative_features(
    coordinates_um: np.ndarray,
    edge_index: np.ndarray,
    orientation: LocalOrientationTensors,
    *,
    epsilon: float = 1e-12,
    output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Materialize the 70 invariant features for an explicitly supplied edge shard."""

    coordinates = _validate_coordinates(coordinates_um)
    edges = _validate_edge_index(edge_index, n_nodes=len(coordinates))
    if len(orientation.tensors) != len(coordinates):
        raise RelativeGeometryContractError(
            "Orientation tensors must align to the coordinate nodes."
        )
    if not np.issubdtype(np.dtype(output_dtype), np.floating):
        raise RelativeGeometryContractError("output_dtype must be floating point.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise RelativeGeometryContractError("epsilon must be positive and finite.")
    source, receiver = edges
    delta = coordinates[source] - coordinates[receiver]
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance <= 0.0):
        raise RelativeGeometryContractError(
            "Relative geometry requires strictly positive edge distances."
        )
    if np.any(distance > DEFAULT_MAX_DISTANCE_UM):
        raise RelativeGeometryContractError(
            "An edge exceeds 500 um; silently clipping it is prohibited."
        )
    unit = delta / (distance[:, None] + epsilon)
    receiver_tensor = orientation.tensors[receiver]
    source_tensor = orientation.tensors[source]
    receiver_alignment = np.einsum(
        "ea,eab,eb->e", unit, receiver_tensor, unit, optimize=True
    )
    source_alignment = np.einsum(
        "ea,eab,eb->e", unit, source_tensor, unit, optimize=True
    )
    tensor_alignment = np.einsum(
        "eab,eba->e", receiver_tensor, source_tensor, optimize=True
    )
    features = np.column_stack(
        [
            gaussian_radial_basis(distance),
            receiver_alignment,
            source_alignment,
            tensor_alignment,
            orientation.anisotropy[receiver],
            orientation.anisotropy[source],
            distance / DEFAULT_MAX_DISTANCE_UM,
        ]
    )
    if features.shape != (edges.shape[1], RELATIVE_GEOMETRY_DIM):
        raise RelativeGeometryContractError(
            "Invariant relative geometry did not produce the locked 70 features."
        )
    if not np.isfinite(features).all():
        raise RelativeGeometryContractError(
            "Invariant relative geometry contains non-finite values."
        )
    return features.astype(output_dtype, copy=False)


@dataclass(frozen=True)
class RelativeGeometryShard:
    """One complete set of incoming edges for a contiguous receiver interval."""

    receiver_start: int
    receiver_stop: int
    edge_index: np.ndarray = field(repr=False)
    features: np.ndarray = field(repr=False)
    checksum_sha256: str

    def __post_init__(self) -> None:
        edges = np.asarray(self.edge_index)
        features = np.asarray(self.features)
        if edges.shape != (2, len(features)) or features.shape[1:] != (
            RELATIVE_GEOMETRY_DIM,
        ):
            raise RelativeGeometryContractError(
                "Relative-geometry shard edges and features do not align."
            )
        if edges.shape[1]:
            receiver = edges[1]
            if (
                receiver.min() < self.receiver_start
                or receiver.max() >= self.receiver_stop
            ):
                raise RelativeGeometryContractError(
                    "Relative-geometry shard contains an out-of-range receiver."
                )
        if self.receiver_start < 0 or self.receiver_stop <= self.receiver_start:
            raise RelativeGeometryContractError("Shard receiver range is invalid.")
        if not _is_sha256(self.checksum_sha256):
            raise RelativeGeometryContractError("Shard checksum is invalid.")

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def _assert_receiver_major_edges(edges: np.ndarray) -> None:
    if not edges.shape[1]:
        return
    source, receiver = edges
    if np.any(receiver[1:] < receiver[:-1]):
        raise RelativeGeometryContractError(
            "Receiver-sharded encoding requires receiver-major edge order."
        )
    same_receiver = receiver[1:] == receiver[:-1]
    if np.any(source[1:][same_receiver] <= source[:-1][same_receiver]):
        raise RelativeGeometryContractError(
            "Receiver-sharded encoding requires unique source-major ties."
        )


def iter_invariant_relative_feature_shards(
    coordinates_um: np.ndarray,
    edge_index: np.ndarray,
    orientation: LocalOrientationTensors,
    *,
    receiver_chunk_size: int = 512,
    max_edges_per_chunk: int = 200_000,
    output_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float32,
) -> Iterator[RelativeGeometryShard]:
    """Yield exact receiver-aligned features without cohort-wide ``rho`` storage.

    ``max_edges_per_chunk`` is a soft execution cap: one high-degree receiver is
    never split merely to satisfy it.
    """

    coordinates = _validate_coordinates(coordinates_um)
    edges = _validate_edge_index(edge_index, n_nodes=len(coordinates))
    _assert_receiver_major_edges(edges)
    receiver_chunk_size = _validate_positive_integer(
        receiver_chunk_size, name="receiver_chunk_size"
    )
    max_edges_per_chunk = _validate_positive_integer(
        max_edges_per_chunk, name="max_edges_per_chunk"
    )
    n_nodes = len(coordinates)
    degree = np.bincount(edges[1], minlength=n_nodes).astype(np.int64)
    prefix = np.empty(n_nodes + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(degree, out=prefix[1:])

    receiver_start = 0
    while receiver_start < n_nodes:
        nominal_stop = min(receiver_start + receiver_chunk_size, n_nodes)
        if prefix[nominal_stop] - prefix[receiver_start] <= max_edges_per_chunk:
            receiver_stop = nominal_stop
        else:
            limit = int(prefix[receiver_start] + max_edges_per_chunk)
            receiver_stop = int(np.searchsorted(prefix, limit, side="right") - 1)
            receiver_stop = min(receiver_stop, nominal_stop)
            if receiver_stop <= receiver_start:
                receiver_stop = receiver_start + 1
        edge_start = int(prefix[receiver_start])
        edge_stop = int(prefix[receiver_stop])
        shard_edges = edges[:, edge_start:edge_stop]
        features = invariant_relative_features(
            coordinates,
            shard_edges,
            orientation,
            output_dtype=output_dtype,
        )
        checksum = _payload_sha256(
            {
                "receiver_start": receiver_start,
                "receiver_stop": receiver_stop,
                "edge_index_sha256": _array_sha256("edge_index", shard_edges),
                "features_sha256": _array_sha256("relative_features", features),
            }
        )
        yield RelativeGeometryShard(
            receiver_start=receiver_start,
            receiver_stop=receiver_stop,
            edge_index=shard_edges,
            features=features,
            checksum_sha256=checksum,
        )
        receiver_start = receiver_stop


@dataclass(frozen=True)
class RelativeGeometryChecksums:
    orientation_sha256: str
    edge_index_sha256: str
    parameters_sha256: str
    invariant_features_sha256: str
    relative_geometry_sha256: str

    def __post_init__(self) -> None:
        if not all(_is_sha256(value) for value in asdict(self).values()):
            raise RelativeGeometryContractError(
                "Relative-geometry checksums must be SHA-256 values."
            )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def checksum_invariant_relative_geometry(
    coordinates_um: np.ndarray,
    edge_index: np.ndarray,
    orientation: LocalOrientationTensors,
    *,
    receiver_chunk_size: int = 512,
    max_edges_per_chunk: int = 200_000,
) -> RelativeGeometryChecksums:
    """Checksum the invariant encoding while streaming receiver-aligned shards."""

    coordinates = _validate_coordinates(coordinates_um)
    edges = _validate_edge_index(edge_index, n_nodes=len(coordinates))
    feature_digest = hashlib.sha256()
    feature_digest.update(
        _canonical_json(
            {
                "name": "invariant_relative_features",
                "shape": [int(edges.shape[1]), RELATIVE_GEOMETRY_DIM],
                "dtype": np.dtype(np.float32).str,
            }
        )
    )
    observed_edges = 0
    for shard in iter_invariant_relative_feature_shards(
        coordinates,
        edges,
        orientation,
        receiver_chunk_size=receiver_chunk_size,
        max_edges_per_chunk=max_edges_per_chunk,
        output_dtype=np.float32,
    ):
        _update_raw_bytes(feature_digest, shard.features)
        observed_edges += shard.n_edges
    if observed_edges != edges.shape[1]:
        raise RelativeGeometryContractError(
            "Receiver shards did not checksum every edge exactly once."
        )
    parameters = {
        "schema": "relative_geometry_invariant_v1",
        "feature_names": list(RELATIVE_GEOMETRY_FEATURE_NAMES),
        "rbf_count": DEFAULT_RBF_COUNT,
        "rbf_centers_um": np.linspace(
            0.0, DEFAULT_MAX_DISTANCE_UM, DEFAULT_RBF_COUNT
        ).tolist(),
        "rbf_width_um": DEFAULT_RBF_WIDTH_UM,
        "cutoff": "0.5*(cos(pi*r/500)+1)",
        "maximum_distance_um": DEFAULT_MAX_DISTANCE_UM,
        "directed_displacement": "source_minus_receiver",
        "orientation_kind": "normalized_traceless_axial_tensor",
    }
    orientation_checksum = orientation.checksum_sha256
    edge_checksum = _array_sha256("receiver_major_edge_index", edges)
    parameter_checksum = _payload_sha256(parameters)
    feature_checksum = feature_digest.hexdigest()
    combined = _payload_sha256(
        {
            "orientation_sha256": orientation_checksum,
            "edge_index_sha256": edge_checksum,
            "parameters_sha256": parameter_checksum,
            "invariant_features_sha256": feature_checksum,
        }
    )
    return RelativeGeometryChecksums(
        orientation_sha256=orientation_checksum,
        edge_index_sha256=edge_checksum,
        parameters_sha256=parameter_checksum,
        invariant_features_sha256=feature_checksum,
        relative_geometry_sha256=combined,
    )


__all__ = [
    "DEFAULT_MAX_DISTANCE_UM",
    "DEFAULT_NOMINAL_K",
    "DEFAULT_ORIENTATION_MIN_NEIGHBORS",
    "DEFAULT_ORIENTATION_NEIGHBOR_CAP",
    "DEFAULT_ORIENTATION_RADIUS_UM",
    "DEFAULT_ORIENTATION_SIGMA_UM",
    "DEFAULT_RADIAL_SHELL_BOUNDS_UM",
    "DEFAULT_RADIAL_SHELL_QUOTAS",
    "DEFAULT_RBF_COUNT",
    "DEFAULT_RBF_WIDTH_UM",
    "LocalOrientationTensors",
    "RELATIVE_GEOMETRY_DIM",
    "RELATIVE_GEOMETRY_FEATURE_NAMES",
    "RadialStratifiedGraph",
    "RadialStratifiedGraphChecksums",
    "RadialStratifiedGraphQC",
    "RelativeGeometryChecksums",
    "RelativeGeometryContractError",
    "RelativeGeometryShard",
    "checksum_invariant_relative_geometry",
    "gaussian_radial_basis",
    "invariant_relative_features",
    "iter_invariant_relative_feature_shards",
    "local_orientation_tensors",
    "radial_stratified_knn",
    "smooth_cutoff_envelope",
]
