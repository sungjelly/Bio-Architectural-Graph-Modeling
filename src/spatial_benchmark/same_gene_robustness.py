"""Fail-closed graph and normalization primitives for same-gene robustness runs.

The functions in this module deliberately operate on geometry partitions rather
than train/test labels.  They may therefore be run once before model fitting
without using expression outcomes to choose folds or graph hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from spatial_benchmark.same_gene_jacobian import (
    ANNULAR_MAX_UM,
    MINIMUM_MATCHED_DEGREE,
    NEAR_MAX_UM,
    NEIGHBOR_K,
    SameGeneJacobianError,
)


PartitionMode = Literal["within_fov", "within_geometry_component"]


@dataclass(frozen=True, slots=True)
class RobustNeighborAggregates:
    """Dense graph aggregates plus exact edge and permutation audit fields."""

    near_mean: np.ndarray
    annular_mean: np.ndarray
    permuted_near_mean: np.ndarray
    near_degree: np.ndarray
    annular_degree: np.ndarray
    permuted_near_degree: np.ndarray
    matched_eligible: np.ndarray
    source_permutation: np.ndarray
    audit: dict[str, object]


def panel_log_cp10k(expression_log1p: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Recover integer counts and apply panel-total log-CP10k deterministically."""

    expression = np.asarray(expression_log1p, dtype=np.float32)
    if expression.ndim != 2 or expression.shape[1] < 2:
        raise SameGeneJacobianError("expression_log1p must have shape [N,G]")
    if not bool(np.isfinite(expression).all()) or bool(np.any(expression < 0)):
        raise SameGeneJacobianError("expression_log1p must be finite and nonnegative")
    recovered = np.expm1(expression.astype(np.float64))
    counts = np.rint(recovered)
    maximum_rounding_error = float(np.max(np.abs(recovered - counts)))
    if maximum_rounding_error > 2e-3 or bool(np.any(counts < 0)):
        raise SameGeneJacobianError(
            "stored log1p values do not round-trip to nonnegative integer counts"
        )
    totals = counts.sum(axis=1, dtype=np.float64)
    denominator = np.maximum(totals, 1.0)
    transformed = np.log1p(counts * (10000.0 / denominator[:, None]))
    transformed = transformed.astype(np.float32)
    if not bool(np.isfinite(transformed).all()):
        raise SameGeneJacobianError("panel log-CP10k transform is nonfinite")
    return transformed, {
        "cell_count": int(expression.shape[0]),
        "gene_count": int(expression.shape[1]),
        "zero_panel_total_cells": int(np.sum(totals == 0)),
        "maximum_integer_roundtrip_error": maximum_rounding_error,
        "median_panel_total": float(np.median(totals)),
        "mean_panel_total": float(np.mean(totals)),
    }


def _seed(slide: str, fov: int, permutation_seed: int) -> int:
    payload = f"{slide}:{int(fov)}:{int(permutation_seed)}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "little", signed=False)


def _conflict_free_permutation(
    size: int,
    forbidden_destinations: list[set[int]],
    *,
    seed: int,
) -> np.ndarray:
    """Find a deterministic maximum-change collision-free bijection.

    The fast path seeks a full derangement. A small QC-filtered FOV can violate
    Hall's condition once fixed sources are forbidden even though a
    receiver-collision-free bijection exists. The exact fallback therefore
    minimizes fixed sources while retaining every edge slot and source state.
    """

    if size < 2 or len(forbidden_destinations) != size:
        raise SameGeneJacobianError("a conflict-free permutation needs >=2 aligned cells")
    receiver_forbidden = [set(values) for values in forbidden_destinations]
    for source, forbidden in enumerate(receiver_forbidden):
        if any(destination < 0 or destination >= size for destination in forbidden):
            raise SameGeneJacobianError("a forbidden permutation destination is invalid")
        if len(forbidden) >= size:
            raise SameGeneJacobianError(
                f"source {source} has no receiver-collision-free destination"
            )
    derangement_forbidden = [
        forbidden | {source}
        for source, forbidden in enumerate(receiver_forbidden)
    ]
    rng = np.random.default_rng(int(seed))
    identity = np.arange(size, dtype=np.int64)
    if all(len(forbidden) < size for forbidden in derangement_forbidden):
        for _restart in range(1024):
            destinations = rng.permutation(identity)
            for _pass in range(8):
                bad = [
                    source
                    for source in range(size)
                    if int(destinations[source]) in derangement_forbidden[source]
                ]
                if not bad:
                    if len(np.unique(destinations)) != size:
                        raise SameGeneJacobianError("permutation lost bijectivity")
                    return destinations
                bad_set = set(bad)
                good = np.asarray(
                    [source for source in range(size) if source not in bad_set],
                    dtype=np.int64,
                )
                if len(good):
                    good = good[rng.permutation(len(good))]
                progress = False
                for source in bad:
                    source_destination = int(destinations[source])
                    for partner_value in good:
                        partner = int(partner_value)
                        partner_destination = int(destinations[partner])
                        if (
                            partner_destination not in derangement_forbidden[source]
                            and source_destination not in derangement_forbidden[partner]
                        ):
                            destinations[source], destinations[partner] = (
                                destinations[partner],
                                destinations[source],
                            )
                            progress = True
                            break
                if not progress:
                    break

    # A prohibited edge costs more than all possible fixed points together;
    # hence any feasible optimum is collision-free and minimizes fixed sources.
    prohibited_cost = size + 1
    cost = np.zeros((size, size), dtype=np.int32)
    for source, forbidden in enumerate(receiver_forbidden):
        if forbidden:
            blocked = np.fromiter(forbidden, dtype=np.int64)
            cost[source, blocked] = prohibited_cost
        if source not in forbidden:
            cost[source, source] = 1
    row_order = rng.permutation(identity)
    column_order = rng.permutation(identity)
    assigned_rows, assigned_columns = linear_sum_assignment(
        cost[np.ix_(row_order, column_order)]
    )
    destinations = np.full(size, -1, dtype=np.int64)
    destinations[row_order[assigned_rows]] = column_order[assigned_columns]
    if (
        np.any(destinations < 0)
        or len(np.unique(destinations)) != size
        or any(
            int(destinations[source]) in receiver_forbidden[source]
            for source in range(size)
        )
    ):
        raise SameGeneJacobianError(
            "no receiver-collision-free degree-preserving source permutation exists"
        )
    return destinations


def _mean_from_local_edges(
    expression: np.ndarray,
    rows: np.ndarray,
    sources: np.ndarray,
    degree: np.ndarray,
) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros_like(expression, dtype=np.float32)
    weights = np.reciprocal(degree[rows].astype(np.float32))
    matrix = coo_matrix(
        (weights, (rows, sources)),
        shape=(len(expression), len(expression)),
        dtype=np.float32,
    ).tocsr()
    return np.asarray(matrix @ expression, dtype=np.float32)


def _partition_edges(
    coordinates: np.ndarray,
    *,
    k: int,
    near_max_um: float,
    annular_max_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    neighborhoods = cKDTree(coordinates).query_ball_point(
        coordinates, r=float(annular_max_um), workers=-1
    )
    near_rows: list[int] = []
    near_sources: list[int] = []
    annular_rows: list[int] = []
    annular_sources: list[int] = []
    zero_distance = 0
    for receiver, candidate_list in enumerate(neighborhoods):
        candidates = np.asarray(
            [candidate for candidate in candidate_list if candidate != receiver],
            dtype=np.int64,
        )
        if not len(candidates):
            continue
        delta = coordinates[candidates] - coordinates[receiver]
        distance = np.sqrt(np.sum(delta * delta, axis=1))
        zero_distance += int(np.sum(distance == 0.0))
        order = np.lexsort((candidates, distance))
        candidates = candidates[order]
        distance = distance[order]
        near = candidates[(distance > 0.0) & (distance <= near_max_um)][:k]
        annular = candidates[
            (distance > near_max_um) & (distance <= annular_max_um)
        ][:k]
        near_rows.extend([receiver] * len(near))
        near_sources.extend(int(value) for value in near)
        annular_rows.extend([receiver] * len(annular))
        annular_sources.extend(int(value) for value in annular)
    return (
        np.asarray(near_rows, dtype=np.int64),
        np.asarray(near_sources, dtype=np.int64),
        np.asarray(annular_rows, dtype=np.int64),
        np.asarray(annular_sources, dtype=np.int64),
        zero_distance,
    )


def build_robust_neighbor_aggregates(
    expression_values: np.ndarray,
    coordinates_um: np.ndarray,
    fovs: np.ndarray,
    geometry_groups: np.ndarray,
    *,
    slide: str,
    partition_mode: PartitionMode,
    active_nodes: np.ndarray | None = None,
    k: int = NEIGHBOR_K,
    near_max_um: float = NEAR_MAX_UM,
    annular_max_um: float = ANNULAR_MAX_UM,
    minimum_matched_degree: int = MINIMUM_MATCHED_DEGREE,
    permutation_seed: int = 20260810,
) -> RobustNeighborAggregates:
    """Build observed and degree-preserving null aggregates within partitions."""

    expression = np.asarray(expression_values, dtype=np.float32)
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    fov_array = np.asarray(fovs, dtype=np.int64)
    groups = np.asarray(geometry_groups, dtype=np.int64)
    nodes = len(expression)
    active = (
        np.ones(nodes, dtype=bool)
        if active_nodes is None
        else np.asarray(active_nodes, dtype=bool)
    )
    if expression.ndim != 2 or expression.shape[1] < 2:
        raise SameGeneJacobianError("expression_values must have shape [N,G]")
    if (
        coordinates.shape != (nodes, 2)
        or fov_array.shape != (nodes,)
        or groups.shape != (nodes,)
        or active.shape != (nodes,)
    ):
        raise SameGeneJacobianError("geometry and active mask must align with expression")
    if not np.any(active) or not bool(np.isfinite(expression).all()):
        raise SameGeneJacobianError("active graph nodes and finite expression are required")
    if not bool(np.isfinite(coordinates).all()):
        raise SameGeneJacobianError("coordinates must be finite")
    if partition_mode not in {"within_fov", "within_geometry_component"}:
        raise SameGeneJacobianError(f"unsupported partition mode: {partition_mode}")
    if not 0 < near_max_um < annular_max_um or k < 1:
        raise SameGeneJacobianError("invalid distance bands or k")

    partition = fov_array if partition_mode == "within_fov" else groups
    near_output = np.zeros_like(expression, dtype=np.float32)
    annular_output = np.zeros_like(expression, dtype=np.float32)
    permuted_output = np.zeros_like(expression, dtype=np.float32)
    near_degree = np.zeros(nodes, dtype=np.int16)
    annular_degree = np.zeros(nodes, dtype=np.int16)
    permuted_degree = np.zeros(nodes, dtype=np.int16)
    source_permutation = np.full(nodes, -1, dtype=np.int32)

    partition_payloads: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    total_zero_distance = 0
    total_near = 0
    total_annular = 0
    near_cross_fov = 0
    annular_cross_fov = 0
    for value in sorted(int(item) for item in np.unique(partition[active])):
        indices = np.flatnonzero(active & (partition == value))
        if len(indices) < 2:
            continue
        near_rows, near_sources, annular_rows, annular_sources, zero_count = (
            _partition_edges(
                coordinates[indices],
                k=k,
                near_max_um=near_max_um,
                annular_max_um=annular_max_um,
            )
        )
        partition_payloads.append(
            (indices, near_rows, near_sources, annular_rows, annular_sources)
        )
        total_zero_distance += zero_count
        total_near += len(near_rows)
        total_annular += len(annular_rows)
        near_cross_fov += int(
            np.sum(fov_array[indices[near_rows]] != fov_array[indices[near_sources]])
        )
        annular_cross_fov += int(
            np.sum(
                fov_array[indices[annular_rows]]
                != fov_array[indices[annular_sources]]
            )
        )

    # The source-state null is a global bijection within every original FOV.
    # For same-FOV edge slots it forbids remapping a source to the receiver, so
    # no slot is dropped. Fixed sources are minimized exactly when a perfect
    # derangement is mathematically impossible after QC filtering.
    same_fov_incoming: dict[int, list[tuple[int, int]]] = {}
    for indices, near_rows, near_sources, _, _ in partition_payloads:
        global_rows = indices[near_rows]
        global_sources = indices[near_sources]
        same = fov_array[global_rows] == fov_array[global_sources]
        for receiver, source in zip(global_rows[same], global_sources[same], strict=True):
            same_fov_incoming.setdefault(int(fov_array[source]), []).append(
                (int(receiver), int(source))
            )
    changed = 0
    fixed_sources_by_fov: dict[str, int] = {}
    for fov in sorted(int(value) for value in np.unique(fov_array[active])):
        indices = np.flatnonzero(active & (fov_array == fov))
        if len(indices) < 2:
            raise SameGeneJacobianError(
                f"active FOV {fov} on {slide} has fewer than two cells"
            )
        inverse = {int(global_index): local for local, global_index in enumerate(indices)}
        forbidden = [set() for _ in range(len(indices))]
        for receiver, source in same_fov_incoming.get(fov, []):
            forbidden[inverse[source]].add(inverse[receiver])
        try:
            local_permutation = _conflict_free_permutation(
                len(indices), forbidden, seed=_seed(slide, fov, permutation_seed)
            )
        except SameGeneJacobianError as error:
            forbidden_sizes = np.asarray(
                [len(destinations | {source}) for source, destinations in enumerate(forbidden)],
                dtype=np.int64,
            )
            raise SameGeneJacobianError(
                "conflict-free permutation failed for "
                f"slide={slide}, fov={fov}, active_cells={len(indices)}, "
                f"forbidden_min={int(forbidden_sizes.min())}, "
                f"forbidden_mean={float(forbidden_sizes.mean()):.6f}, "
                f"forbidden_max={int(forbidden_sizes.max())}: {error}"
            ) from error
        source_permutation[indices] = indices[local_permutation]
        local_fixed = int(
            np.sum(local_permutation == np.arange(len(indices), dtype=np.int64))
        )
        changed += len(indices) - local_fixed
        if local_fixed:
            fixed_sources_by_fov[str(fov)] = local_fixed
    if np.any(source_permutation[active] < 0):
        raise SameGeneJacobianError("an active source is missing a permutation mapping")

    for indices, near_rows, near_sources, annular_rows, annular_sources in partition_payloads:
        local_near_degree = np.bincount(near_rows, minlength=len(indices)).astype(np.int16)
        local_annular_degree = np.bincount(
            annular_rows, minlength=len(indices)
        ).astype(np.int16)
        near_output[indices] = _mean_from_local_edges(
            expression[indices], near_rows, near_sources, local_near_degree
        )
        annular_output[indices] = _mean_from_local_edges(
            expression[indices], annular_rows, annular_sources, local_annular_degree
        )
        permuted_global = source_permutation[indices[near_sources]]
        permuted_sources = np.searchsorted(indices, permuted_global).astype(np.int64)
        if (
            np.any(permuted_sources >= len(indices))
            or not np.array_equal(indices[permuted_sources], permuted_global)
        ):
            raise SameGeneJacobianError("permuted source escaped its geometry partition")
        if np.any(permuted_sources == near_rows):
            raise SameGeneJacobianError("permuted source reintroduced the receiver target")
        local_permuted_degree = np.bincount(
            near_rows, minlength=len(indices)
        ).astype(np.int16)
        permuted_output[indices] = _mean_from_local_edges(
            expression[indices], near_rows, permuted_sources, local_permuted_degree
        )
        near_degree[indices] = local_near_degree
        annular_degree[indices] = local_annular_degree
        permuted_degree[indices] = local_permuted_degree

    if not np.array_equal(permuted_degree, near_degree):
        raise SameGeneJacobianError("permutation failed to preserve every receiver degree")
    eligible = (
        active
        & (near_degree >= minimum_matched_degree)
        & (annular_degree >= minimum_matched_degree)
        & (permuted_degree >= minimum_matched_degree)
    )
    for value in (near_output, annular_output, permuted_output):
        if not bool(np.isfinite(value).all()):
            raise SameGeneJacobianError("a robust neighbor aggregate is nonfinite")
    audit: dict[str, object] = {
        "slide": str(slide),
        "partition_mode": partition_mode,
        "node_count": nodes,
        "active_node_count": int(np.sum(active)),
        "inactive_node_count": int(np.sum(~active)),
        "gene_count": int(expression.shape[1]),
        "partition_count": int(len(np.unique(partition[active]))),
        "fov_count": int(len(np.unique(fov_array[active]))),
        "near_directed_edge_count": total_near,
        "annular_directed_edge_count": total_annular,
        "permuted_directed_edge_count": total_near,
        "near_cross_fov_edge_count": near_cross_fov,
        "annular_cross_fov_edge_count": annular_cross_fov,
        "near_cross_fov_edge_fraction": near_cross_fov / total_near if total_near else 0.0,
        "annular_cross_fov_edge_fraction": (
            annular_cross_fov / total_annular if total_annular else 0.0
        ),
        "zero_distance_candidate_pair_count": total_zero_distance,
        "permutation_seed": int(permutation_seed),
        "permutation_degree_preserved": True,
        "permutation_mapping_policy": (
            "receiver_collision_free_fov_bijection_maximizing_changed_sources"
        ),
        "permutation_fixed_source_count": int(sum(fixed_sources_by_fov.values())),
        "permutation_fov_count_with_fixed_sources": len(fixed_sources_by_fov),
        "permutation_fixed_sources_by_fov": fixed_sources_by_fov,
        "permutation_source_mapping_changed_fraction": changed / int(np.sum(active)),
        "permutation_receiver_collisions": 0,
        "near_degree_mean_active": float(np.mean(near_degree[active])),
        "annular_degree_mean_active": float(np.mean(annular_degree[active])),
        "matched_eligible_count": int(np.sum(eligible)),
        "matched_eligible_fraction_active": float(np.mean(eligible[active])),
    }
    return RobustNeighborAggregates(
        near_mean=near_output,
        annular_mean=annular_output,
        permuted_near_mean=permuted_output,
        near_degree=near_degree,
        annular_degree=annular_degree,
        permuted_near_degree=permuted_degree,
        matched_eligible=eligible,
        source_permutation=source_permutation,
        audit=audit,
    )


__all__ = [
    "PartitionMode",
    "RobustNeighborAggregates",
    "build_robust_neighbor_aggregates",
    "panel_log_cp10k",
]
