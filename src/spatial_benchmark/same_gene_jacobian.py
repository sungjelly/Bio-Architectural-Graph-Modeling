"""Label-free same-gene cross-cell Jacobian experiment primitives.

The primary model in this workflow is deliberately additive and linear in a
degree-normalized neighbor-expression feature.  Its cross-cell Jacobian is
therefore exact and auditable: the fitted neighbor coefficient divided by the
receiver degree is the derivative for one included sender.  Nothing in this
module assigns clinical, patient, core, cell-type, or mechanistic meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.stats import rankdata


COMPONENT_THRESHOLD_MM = 0.75
PIXEL_SIZE_UM = 0.120281
NEIGHBOR_K = 12
NEAR_MAX_UM = 25.0
ANNULAR_MAX_UM = 50.0
MINIMUM_MATCHED_DEGREE = 4
PERMUTATION_SEED = 20260810

# Ordinals are defined by minimum FOV ascending within each slide.  The map was
# frozen from geometry and aggregate cell/FOV counts before expression outcomes
# were inspected.  It balances 407,999 cells over four folds while keeping
# complete geometry components intact.
FOLD_COMPONENT_ORDINALS: Mapping[int, Mapping[str, tuple[int, ...]]] = {
    0: {"SO_1": (1, 8, 11, 13), "SO_2": (7, 13, 14)},
    1: {"SO_1": (6, 7, 9), "SO_2": (4, 10, 12)},
    2: {"SO_1": (2, 4, 5), "SO_2": (5, 6, 8, 9)},
    3: {"SO_1": (3, 10, 12), "SO_2": (1, 2, 3, 11)},
}


class SameGeneJacobianError(ValueError):
    """Raised when the experiment's data or numerical contract is violated."""


@dataclass(frozen=True, slots=True)
class GeometryComponents:
    """Opaque FOV-origin connected components for one explicit slide."""

    slide: str
    fovs: np.ndarray
    component_ordinals: np.ndarray
    folds: np.ndarray
    component_count: int
    minimum_between_component_distance_mm: float


@dataclass(frozen=True, slots=True)
class NeighborAggregates:
    """Expression means and geometry-only audit fields for three graph arms."""

    near_mean: np.ndarray
    annular_mean: np.ndarray
    permuted_near_mean: np.ndarray
    near_degree: np.ndarray
    annular_degree: np.ndarray
    permuted_near_degree: np.ndarray
    matched_eligible: np.ndarray
    audit: dict[str, float | int]


@dataclass(frozen=True, slots=True)
class DiagonalSummary:
    """Scale-aware descriptive summary of a target-by-source coefficient map."""

    eligible_gene_count: int
    median_absolute_diagonal: float
    median_absolute_offdiagonal: float
    diagonal_offdiagonal_ratio: float
    row_top1_fraction: float
    row_top10_fraction: float
    row_top1_percent_fraction: float
    median_diagonal_absolute_rank: float
    positive_diagonal_fraction: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "eligible_gene_count": self.eligible_gene_count,
            "median_absolute_diagonal": self.median_absolute_diagonal,
            "median_absolute_offdiagonal": self.median_absolute_offdiagonal,
            "diagonal_offdiagonal_ratio": self.diagonal_offdiagonal_ratio,
            "row_top1_fraction": self.row_top1_fraction,
            "row_top10_fraction": self.row_top10_fraction,
            "row_top1_percent_fraction": self.row_top1_percent_fraction,
            "median_diagonal_absolute_rank": self.median_diagonal_absolute_rank,
            "positive_diagonal_fraction": self.positive_diagonal_fraction,
        }


def _canonical_slide(slide: str) -> str:
    value = str(slide).strip().upper().replace("-", "_")
    if value not in {"SO_1", "SO_2"}:
        raise SameGeneJacobianError("slide must be SO_1 or SO_2")
    return value


def geometry_components(
    slide: str,
    fovs: Sequence[int] | np.ndarray,
    origins_mm: np.ndarray,
    *,
    threshold_mm: float = COMPONENT_THRESHOLD_MM,
) -> GeometryComponents:
    """Build deterministic same-slide FOV-origin components and frozen folds."""

    canonical_slide = _canonical_slide(slide)
    fov_array = np.asarray(fovs, dtype=np.int64)
    origins = np.asarray(origins_mm, dtype=np.float64)
    if fov_array.ndim != 1 or origins.shape != (len(fov_array), 2):
        raise SameGeneJacobianError("FOVs and origins must have shapes [F] and [F,2]")
    if len(fov_array) < 1 or len(np.unique(fov_array)) != len(fov_array):
        raise SameGeneJacobianError("FOV identifiers must be non-empty and unique")
    if np.any(fov_array <= 0) or not bool(np.isfinite(origins).all()):
        raise SameGeneJacobianError("FOV identifiers and origins must be valid")
    if not np.isfinite(threshold_mm) or threshold_mm <= 0:
        raise SameGeneJacobianError("threshold_mm must be positive and finite")

    pairs = cKDTree(origins).query_pairs(float(threshold_mm), output_type="ndarray")
    if len(pairs):
        rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
        columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, columns)),
            shape=(len(fov_array), len(fov_array)),
        ).tocsr()
    else:
        adjacency = coo_matrix((len(fov_array), len(fov_array))).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    ordered_labels = sorted(
        range(count), key=lambda label: int(fov_array[labels == label].min())
    )
    ordinal_by_label = {label: ordinal for ordinal, label in enumerate(ordered_labels, 1)}
    ordinals = np.asarray([ordinal_by_label[int(label)] for label in labels], dtype=np.int16)

    expected_ordinals = {
        ordinal
        for fold_map in FOLD_COMPONENT_ORDINALS.values()
        for ordinal in fold_map[canonical_slide]
    }
    observed_ordinals = set(int(value) for value in np.unique(ordinals))
    if observed_ordinals != expected_ordinals:
        raise SameGeneJacobianError(
            f"{canonical_slide} geometry components changed: expected "
            f"{sorted(expected_ordinals)}, observed {sorted(observed_ordinals)}"
        )
    ordinal_to_fold: dict[int, int] = {}
    for fold, slide_map in FOLD_COMPONENT_ORDINALS.items():
        for ordinal in slide_map[canonical_slide]:
            if ordinal in ordinal_to_fold:
                raise SameGeneJacobianError("Frozen component map contains a duplicate")
            ordinal_to_fold[ordinal] = fold
    folds = np.asarray([ordinal_to_fold[int(value)] for value in ordinals], dtype=np.int8)

    minimum_between = np.inf
    for first in range(len(origins)):
        mask = ordinals != ordinals[first]
        if np.any(mask):
            distance = np.sqrt(np.sum((origins[mask] - origins[first]) ** 2, axis=1))
            minimum_between = min(minimum_between, float(distance.min()))
    if not np.isfinite(minimum_between):
        raise SameGeneJacobianError("At least two geometry components are required")
    return GeometryComponents(
        slide=canonical_slide,
        fovs=fov_array.copy(),
        component_ordinals=ordinals,
        folds=folds,
        component_count=count,
        minimum_between_component_distance_mm=minimum_between,
    )


def _seed_for_fov(slide: str, fov: int, seed: int) -> int:
    payload = f"{_canonical_slide(slide)}:{int(fov)}:{int(seed)}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "little", signed=False)


def deterministic_derangement(size: int, *, seed: int) -> np.ndarray:
    """Return a reproducible permutation with no fixed points."""

    if isinstance(size, bool) or int(size) != size or size < 2:
        raise SameGeneJacobianError("A derangement requires at least two items")
    rng = np.random.default_rng(int(seed))
    identity = np.arange(int(size), dtype=np.int64)
    for _ in range(256):
        candidate = rng.permutation(identity)
        if bool(np.all(candidate != identity)):
            return candidate
    raise SameGeneJacobianError("Could not construct a deterministic derangement")


def _mean_from_edges(
    expression: np.ndarray,
    rows: np.ndarray,
    sources: np.ndarray,
    degree: np.ndarray,
) -> np.ndarray:
    nodes = expression.shape[0]
    if len(rows) != len(sources):
        raise SameGeneJacobianError("edge rows and sources are not aligned")
    if len(rows) == 0:
        return np.zeros_like(expression, dtype=np.float32)
    weights = np.reciprocal(degree[rows].astype(np.float32))
    matrix = coo_matrix(
        (weights, (rows, sources)), shape=(nodes, nodes), dtype=np.float32
    ).tocsr()
    result = matrix @ expression
    return np.asarray(result, dtype=np.float32)


def build_neighbor_aggregates(
    expression_log1p: np.ndarray,
    coordinates_um: np.ndarray,
    fovs: Sequence[int] | np.ndarray,
    *,
    slide: str,
    k: int = NEIGHBOR_K,
    near_max_um: float = NEAR_MAX_UM,
    annular_max_um: float = ANNULAR_MAX_UM,
    minimum_matched_degree: int = MINIMUM_MATCHED_DEGREE,
    permutation_seed: int = PERMUTATION_SEED,
) -> NeighborAggregates:
    """Build near, annular, and source-state-permuted neighbor means.

    All edges remain within an explicit FOV.  The permutation changes source
    states in fixed near-edge slots.  A remapped source equal to the receiver
    is dropped so a whole-node target never re-enters through the null arm.
    """

    canonical_slide = _canonical_slide(slide)
    expression = np.asarray(expression_log1p, dtype=np.float32)
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    fov_array = np.asarray(fovs, dtype=np.int64)
    nodes = expression.shape[0]
    if expression.ndim != 2 or expression.shape[1] < 2:
        raise SameGeneJacobianError("expression_log1p must have shape [N,G]")
    if coordinates.shape != (nodes, 2) or fov_array.shape != (nodes,):
        raise SameGeneJacobianError("coordinates/FOVs do not align with expression")
    if not bool(np.isfinite(expression).all()) or not bool(np.isfinite(coordinates).all()):
        raise SameGeneJacobianError("neighbor inputs must be finite")
    if k < 1 or minimum_matched_degree < 1 or minimum_matched_degree > k:
        raise SameGeneJacobianError("invalid k or minimum matched degree")
    if not 0 < near_max_um < annular_max_um:
        raise SameGeneJacobianError("distance bands must satisfy 0 < near < annular")

    near_output = np.zeros_like(expression, dtype=np.float32)
    annular_output = np.zeros_like(expression, dtype=np.float32)
    permuted_output = np.zeros_like(expression, dtype=np.float32)
    near_degree = np.zeros(nodes, dtype=np.int16)
    annular_degree = np.zeros(nodes, dtype=np.int16)
    permuted_degree = np.zeros(nodes, dtype=np.int16)

    total_near_edges = 0
    total_annular_edges = 0
    total_permuted_edges = 0
    target_self_collisions = 0
    zero_distance_pairs = 0
    displacement_values: list[np.ndarray] = []

    for fov in sorted(int(value) for value in np.unique(fov_array)):
        global_indices = np.flatnonzero(fov_array == fov)
        local_coordinates = coordinates[global_indices]
        local_expression = expression[global_indices]
        if len(global_indices) < 2:
            raise SameGeneJacobianError("Every FOV must contain at least two cells")
        neighborhoods = cKDTree(local_coordinates).query_ball_point(
            local_coordinates, r=float(annular_max_um), workers=-1
        )
        near_rows: list[int] = []
        near_sources: list[int] = []
        annular_rows: list[int] = []
        annular_sources: list[int] = []
        for receiver, candidates_list in enumerate(neighborhoods):
            candidates = np.asarray(
                [value for value in candidates_list if value != receiver],
                dtype=np.int64,
            )
            if candidates.size == 0:
                continue
            delta = local_coordinates[candidates] - local_coordinates[receiver]
            distance = np.sqrt(np.sum(delta * delta, axis=1))
            zero_distance_pairs += int(np.sum(distance == 0.0))
            # Distance first and local row second give a deterministic tie break.
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

        near_rows_array = np.asarray(near_rows, dtype=np.int64)
        near_sources_array = np.asarray(near_sources, dtype=np.int64)
        annular_rows_array = np.asarray(annular_rows, dtype=np.int64)
        annular_sources_array = np.asarray(annular_sources, dtype=np.int64)
        local_near_degree = np.bincount(
            near_rows_array, minlength=len(global_indices)
        ).astype(np.int16)
        local_annular_degree = np.bincount(
            annular_rows_array, minlength=len(global_indices)
        ).astype(np.int16)
        near_output[global_indices] = _mean_from_edges(
            local_expression,
            near_rows_array,
            near_sources_array,
            local_near_degree,
        )
        annular_output[global_indices] = _mean_from_edges(
            local_expression,
            annular_rows_array,
            annular_sources_array,
            local_annular_degree,
        )
        near_degree[global_indices] = local_near_degree
        annular_degree[global_indices] = local_annular_degree

        permutation = deterministic_derangement(
            len(global_indices),
            seed=_seed_for_fov(canonical_slide, fov, permutation_seed),
        )
        displacement_values.append(
            np.sqrt(
                np.sum(
                    (local_coordinates[permutation] - local_coordinates) ** 2,
                    axis=1,
                )
            )
        )
        permuted_sources = permutation[near_sources_array]
        keep = permuted_sources != near_rows_array
        target_self_collisions += int(np.sum(~keep))
        permuted_rows = near_rows_array[keep]
        permuted_sources = permuted_sources[keep]
        local_permuted_degree = np.bincount(
            permuted_rows, minlength=len(global_indices)
        ).astype(np.int16)
        permuted_output[global_indices] = _mean_from_edges(
            local_expression,
            permuted_rows,
            permuted_sources,
            local_permuted_degree,
        )
        permuted_degree[global_indices] = local_permuted_degree

        total_near_edges += len(near_rows_array)
        total_annular_edges += len(annular_rows_array)
        total_permuted_edges += len(permuted_rows)

    matched = (
        (near_degree >= minimum_matched_degree)
        & (annular_degree >= minimum_matched_degree)
        & (permuted_degree >= minimum_matched_degree)
    )
    displacement = np.concatenate(displacement_values)
    audit: dict[str, float | int] = {
        "node_count": nodes,
        "gene_count": expression.shape[1],
        "fov_count": int(len(np.unique(fov_array))),
        "near_directed_edge_count": total_near_edges,
        "annular_directed_edge_count": total_annular_edges,
        "permuted_directed_edge_count": total_permuted_edges,
        "permutation_source_mapping_changed_fraction": 1.0,
        "permutation_target_self_collisions_removed": target_self_collisions,
        "permutation_target_self_collision_fraction": (
            target_self_collisions / total_near_edges if total_near_edges else 0.0
        ),
        "permutation_median_displacement_um": float(np.median(displacement)),
        "permutation_displacement_over_75um_fraction": float(
            np.mean(displacement > 75.0)
        ),
        "zero_distance_candidate_pair_count": zero_distance_pairs,
        "near_degree_mean": float(np.mean(near_degree)),
        "annular_degree_mean": float(np.mean(annular_degree)),
        "matched_eligible_count": int(np.sum(matched)),
        "matched_eligible_fraction": float(np.mean(matched)),
    }
    if not all(
        bool(np.isfinite(value).all())
        for value in (near_output, annular_output, permuted_output)
    ):
        raise SameGeneJacobianError("A neighbor aggregate contains nonfinite values")
    return NeighborAggregates(
        near_mean=near_output,
        annular_mean=annular_output,
        permuted_near_mean=permuted_output,
        near_degree=near_degree,
        annular_degree=annular_degree,
        permuted_near_degree=permuted_degree,
        matched_eligible=matched,
        audit=audit,
    )


def component_equal_weights(groups: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Return weights summing to one with equal mass per selected component."""

    group_array = np.asarray(groups)
    mask = np.asarray(selected, dtype=bool)
    if group_array.ndim != 1 or mask.shape != group_array.shape or not np.any(mask):
        raise SameGeneJacobianError("groups and a non-empty selected mask are required")
    result = np.zeros(len(groups), dtype=np.float64)
    unique = np.unique(group_array[mask])
    for group in unique:
        positions = mask & (group_array == group)
        result[positions] = 1.0 / (len(unique) * int(np.sum(positions)))
    if not np.isclose(result.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise SameGeneJacobianError("component-equal weights do not sum to one")
    return result


def solve_weighted_ridge_numpy(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    penalty: float,
    unpenalized_columns: Sequence[int] = (0,),
) -> np.ndarray:
    """Small CPU reference solver used by analytical controls and unit tests."""

    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise SameGeneJacobianError("design and targets must align")
    if w.shape != (x.shape[0],) or np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise SameGeneJacobianError("weights must be nonnegative and sum to one")
    if not np.isfinite(penalty) or penalty < 0:
        raise SameGeneJacobianError("penalty must be finite and nonnegative")
    xtx = x.T @ (w[:, None] * x)
    xty = x.T @ (w[:, None] * y)
    regularizer = np.eye(x.shape[1], dtype=np.float64) * float(penalty)
    for column in unpenalized_columns:
        regularizer[int(column), int(column)] = 0.0
    coefficients = np.linalg.solve(xtx + regularizer, xty)
    if not bool(np.isfinite(coefficients).all()):
        raise SameGeneJacobianError("ridge coefficients are nonfinite")
    return coefficients


def planted_recovery_control(seed: int = 20260810) -> dict[str, float | bool]:
    """Recover a planted signed cross-gene map and verify finite differences."""

    rng = np.random.default_rng(seed)
    nodes, genes, morphology_features = 4096, 7, 3
    morphology = rng.normal(size=(nodes, morphology_features))
    neighbors = rng.normal(size=(nodes, genes))
    planted = np.diag(np.linspace(0.35, 0.95, genes))
    planted[0, 1] = -0.27
    planted[3, 5] = 0.19
    morphology_coefficients = rng.normal(scale=0.05, size=(morphology_features, genes))
    targets = morphology @ morphology_coefficients + neighbors @ planted.T
    design = np.column_stack((np.ones(nodes), morphology, neighbors))
    coefficients = solve_weighted_ridge_numpy(
        design,
        targets,
        np.full(nodes, 1.0 / nodes),
        penalty=1e-10,
    )
    recovered = coefficients[1 + morphology_features :, :].T
    maximum_error = float(np.max(np.abs(recovered - planted)))

    receiver = 17
    source_gene = 4
    delta = 1e-4
    original = design[receiver] @ coefficients
    perturbed_design = design[receiver].copy()
    perturbed_design[1 + morphology_features + source_gene] += delta
    perturbed = perturbed_design @ coefficients
    finite_change = (perturbed - original) / delta
    finite_difference_error = float(
        np.max(np.abs(finite_change - recovered[:, source_gene]))
    )
    return {
        "maximum_planted_coefficient_error": maximum_error,
        "maximum_finite_difference_error": finite_difference_error,
        "passed": maximum_error <= 1e-6 and finite_difference_error <= 1e-8,
    }


def diagonal_summary(
    target_by_source: np.ndarray,
    eligible: np.ndarray | Sequence[bool] | None = None,
) -> DiagonalSummary:
    """Summarize whether same-name coefficients dominate matched off-diagonals."""

    matrix = np.asarray(target_by_source, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SameGeneJacobianError("Jacobian must be a square target-by-source matrix")
    if not bool(np.isfinite(matrix).all()):
        raise SameGeneJacobianError("Jacobian contains nonfinite values")
    genes = matrix.shape[0]
    eligibility = (
        np.ones(genes, dtype=bool)
        if eligible is None
        else np.asarray(eligible, dtype=bool)
    )
    if eligibility.shape != (genes,) or int(np.sum(eligibility)) < 2:
        raise SameGeneJacobianError("At least two eligible genes are required")
    indices = np.flatnonzero(eligibility)
    selected = matrix[np.ix_(indices, indices)]
    absolute = np.abs(selected)
    diagonal = np.diag(selected)
    absolute_diagonal = np.abs(diagonal)
    offdiagonal = absolute[~np.eye(len(indices), dtype=bool)]
    median_offdiagonal = float(np.median(offdiagonal))
    median_diagonal = float(np.median(absolute_diagonal))
    ratio = (
        median_diagonal / median_offdiagonal
        if median_offdiagonal > 0
        else (float("inf") if median_diagonal > 0 else 1.0)
    )
    ranks = np.empty(len(indices), dtype=np.float64)
    for row in range(len(indices)):
        # scipy rank 1 is smallest, so reverse the absolute row for rank 1=largest.
        ranks[row] = rankdata(-absolute[row], method="min")[row]
    top_one_percent = max(1, int(np.ceil(0.01 * len(indices))))
    return DiagonalSummary(
        eligible_gene_count=len(indices),
        median_absolute_diagonal=median_diagonal,
        median_absolute_offdiagonal=median_offdiagonal,
        diagonal_offdiagonal_ratio=float(ratio),
        row_top1_fraction=float(np.mean(ranks <= 1)),
        row_top10_fraction=float(np.mean(ranks <= min(10, len(indices)))),
        row_top1_percent_fraction=float(np.mean(ranks <= top_one_percent)),
        median_diagonal_absolute_rank=float(np.median(ranks)),
        positive_diagonal_fraction=float(np.mean(diagonal > 0)),
    )


__all__ = [
    "ANNULAR_MAX_UM",
    "COMPONENT_THRESHOLD_MM",
    "DiagonalSummary",
    "FOLD_COMPONENT_ORDINALS",
    "GeometryComponents",
    "MINIMUM_MATCHED_DEGREE",
    "NEAR_MAX_UM",
    "NEIGHBOR_K",
    "NeighborAggregates",
    "PERMUTATION_SEED",
    "PIXEL_SIZE_UM",
    "SameGeneJacobianError",
    "build_neighbor_aggregates",
    "component_equal_weights",
    "deterministic_derangement",
    "diagonal_summary",
    "geometry_components",
    "planted_recovery_control",
    "solve_weighted_ridge_numpy",
]
