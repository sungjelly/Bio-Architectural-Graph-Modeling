"""Constant and observed-neighbor references for the SO2 diagnostic audit.

These helpers do not select masks, fit neural models, or retain row-level data.
Callers own the split, all-fit labeling, provenance, and output contracts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def huber_constant(values: Any, weights: Any, delta: float = 1.0) -> float:
    """Minimize weighted Huber risk for a finite one-dimensional distribution.

    The returned constant solves the monotone score equation
    ``sum(normalized_weights * clip(constant - values, -delta, delta)) = 0``.
    Zero-weight values are ignored, including when bounding the root. When
    there are multiple minimizers, one root in their interval is returned.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    delta = float(delta)
    if values.ndim != 1 or values.shape != weights.shape or not values.size:
        raise ValueError("values and weights must be aligned nonempty vectors")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("values and weights must be finite")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("weights must be nonnegative with positive total weight")
    selected = weights > 0
    values = values[selected]
    weights = weights[selected]
    # Rescale before summing to avoid overflow from valid large input weights.
    weights = weights / weights.max()
    weights /= weights.sum(dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if low == high:
        return low
    constant = low / 2.0 + high / 2.0
    for _ in range(256):
        constant = low / 2.0 + high / 2.0
        with np.errstate(over="ignore"):
            score = float(np.dot(weights, np.clip(constant - values, -delta, delta)))
        if abs(score) <= 1e-13 * min(delta, 1.0):
            return constant
        if constant == low or constant == high:
            break
        if score > 0:
            high = constant
        else:
            low = constant
    if abs(score) > 1e-9:
        raise ArithmeticError("Huber root could not meet the score residual tolerance")
    return constant


def observed_neighbor_mean(
    adjacency_csr: sparse.csr_matrix,
    values: Any,
    observed_bool: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform mean of observed neighbors, including observed zero values.

    Rows denote receivers; columns denote sources. Duplicate neighbor entries
    count once, and positive edge magnitudes do not change uniform weights.
    Self edges are rejected. Missing neighborhoods fall back to the observed
    core gene mean, or zero when that gene has no observed core values.
    ``fallback_mask`` identifies entries with no observed graph neighbors,
    regardless of which fallback applies. Inputs must already share alignment.
    """
    if not sparse.issparse(adjacency_csr) or adjacency_csr.format != "csr":
        raise ValueError("adjacency_csr must be a CSR sparse matrix")
    values = np.asarray(values)
    observed = np.asarray(observed_bool)
    if values.ndim != 2 or values.shape != observed.shape:
        raise ValueError("values and observed_bool must have aligned [nodes, genes] shapes")
    if observed.dtype != np.bool_:
        raise ValueError("observed_bool must be boolean")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("values must be finite numeric values")
    n_nodes, n_genes = values.shape
    if adjacency_csr.shape != (n_nodes, n_nodes):
        raise ValueError("adjacency shape must match the number of nodes")
    if not np.isfinite(adjacency_csr.data).all() or np.any(adjacency_csr.data < 0):
        raise ValueError("adjacency entries must be finite and nonnegative")
    adjacency = adjacency_csr.astype(np.float64, copy=True)
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    if np.any(adjacency.diagonal() != 0):
        raise ValueError("self edges are prohibited")
    adjacency.data.fill(1.0)
    prediction = np.empty((n_nodes, n_genes), dtype=np.float64)
    fallback_mask = np.empty((n_nodes, n_genes), dtype=bool)
    for start in range(0, n_genes, 128):
        stop = min(start + 128, n_genes)
        visible = observed[:, start:stop]
        observed_values = np.where(visible, values[:, start:stop], 0.0).astype(np.float64)
        denominator = adjacency @ visible.astype(np.float64)
        numerator = adjacency @ observed_values
        missing = denominator == 0
        core_n = visible.sum(axis=0, dtype=np.int64)
        core_sum = observed_values.sum(axis=0, dtype=np.float64)
        core_mean = np.divide(core_sum, core_n, out=np.zeros(stop - start), where=core_n > 0)
        block = np.broadcast_to(core_mean, denominator.shape).copy()
        np.divide(numerator, denominator, out=block, where=~missing)
        if not np.isfinite(block).all():
            raise ArithmeticError("observed-neighbor means must remain finite")
        prediction[:, start:stop] = block
        fallback_mask[:, start:stop] = missing
    return prediction, fallback_mask


def local_adjacency(coords: Any, k: int = 16, radius: float = 75.0) -> sparse.csr_matrix:
    """Directed nearest-neighbor adjacency using only finite coordinates.

    Each receiver selects at most ``k`` other nodes within the inclusive radius.
    Sort order is distance then source index, including ties at the kth distance.
    Co-located distinct cells may be neighbors. The graph is not symmetrized.
    """
    coords = np.asarray(coords, dtype=np.float64)
    radius = float(radius)
    if coords.ndim != 2 or coords.shape[1] < 1 or not np.isfinite(coords).all():
        raise ValueError("coords must be a finite [nodes, dimensions] matrix")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    n_nodes = len(coords)
    if n_nodes < 2:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float64)
    tree = cKDTree(coords)
    # Find a small search radius first; then include all ties at that distance.
    distances, candidates = tree.query(coords, k=min(int(k) + 1, n_nodes), workers=1)
    indptr = [0]
    indices: list[int] = []
    for receiver in range(n_nodes):
        nonself = candidates[receiver] != receiver
        ordered_distances = distances[receiver][nonself]
        cutoff = min(radius, float(ordered_distances[min(int(k), len(ordered_distances)) - 1]))
        possible = np.asarray(
            tree.query_ball_point(coords[receiver], np.nextafter(cutoff, np.inf)),
            dtype=np.int64,
        )
        possible = possible[possible != receiver]
        distance = np.linalg.norm(coords[possible] - coords[receiver], axis=1)
        keep = distance <= radius
        possible, distance = possible[keep], distance[keep]
        order = np.lexsort((possible, distance))[:k]
        indices.extend(possible[order].tolist())
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float64), np.asarray(indices), np.asarray(indptr)),
        shape=(n_nodes, n_nodes),
    )
