"""Dependency-light five-seed stability primitives for relative QKV models.

The functions in this module operate only on caller-supplied, fixed-inference
NumPy arrays.  They do not load datasets or models and never construct an
exhaustive Jacobian.  Cross-seed APIs fail closed unless they receive exactly
five distinct seeds with identical fixed-input identifiers.

Reported variation is labelled *ensemble spread* or *seed uncertainty*.  It is
not a calibrated biological confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Sequence

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


EXPECTED_SEED_COUNT = 5
ENSEMBLE_SPREAD_LABEL = "ensemble spread"
SEED_UNCERTAINTY_LABEL = "seed uncertainty"


class StabilityContractError(ValueError):
    """Raised when fixed-input or five-seed stability contracts are violated."""


def _as_finite_float_array(
    values: np.ndarray,
    *,
    name: str,
    minimum_ndim: int = 1,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < minimum_ndim or any(size == 0 for size in array.shape):
        raise StabilityContractError(
            f"{name} must have at least {minimum_ndim} dimensions and no empty axis."
        )
    if not np.isfinite(array).all():
        raise StabilityContractError(f"{name} must contain only finite values.")
    return array


def _canonical_input_ids(values: Sequence[object]) -> tuple[object, ...]:
    identifiers = tuple(values)
    if not identifiers:
        raise StabilityContractError("fixed_input_ids cannot be empty.")
    try:
        unique_count = len(set(identifiers))
    except TypeError as exc:
        raise StabilityContractError(
            "fixed_input_ids must contain hashable identifiers."
        ) from exc
    if unique_count != len(identifiers):
        raise StabilityContractError("fixed_input_ids must be unique.")
    return identifiers


@dataclass(frozen=True)
class FixedSeedArray:
    """One seed's values aligned to explicit fixed-inference identifiers."""

    seed: int
    fixed_input_ids: tuple[object, ...]
    values: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise StabilityContractError("seed must be an integer.")
        identifiers = _canonical_input_ids(self.fixed_input_ids)
        values = _as_finite_float_array(self.values, name="values")
        if values.shape[0] != len(identifiers):
            raise StabilityContractError(
                "values axis zero must align exactly to fixed_input_ids."
            )
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "fixed_input_ids", identifiers)
        object.__setattr__(self, "values", values)


def _validate_five_seed_arrays(
    records: Sequence[FixedSeedArray],
    *,
    name: str,
    require_same_shape: bool,
) -> tuple[FixedSeedArray, ...]:
    ordered = tuple(sorted(records, key=lambda record: int(record.seed)))
    if len(ordered) != EXPECTED_SEED_COUNT:
        raise StabilityContractError(
            f"{name} requires exactly five seed records; received {len(ordered)}."
        )
    seeds = tuple(int(record.seed) for record in ordered)
    if len(set(seeds)) != EXPECTED_SEED_COUNT:
        raise StabilityContractError(f"{name} requires five distinct seeds.")
    reference_ids = ordered[0].fixed_input_ids
    reference_shape = np.asarray(ordered[0].values).shape
    for record in ordered[1:]:
        if record.fixed_input_ids != reference_ids:
            raise StabilityContractError(
                f"{name} fixed inputs are not identically aligned across seeds."
            )
        shape = np.asarray(record.values).shape
        if shape[0] != reference_shape[0] or (
            require_same_shape and shape != reference_shape
        ):
            raise StabilityContractError(
                f"{name} value shapes are not aligned across seeds."
            )
    return ordered


def _seed_index(records: Sequence[FixedSeedArray], seed: int) -> int:
    for index, record in enumerate(records):
        if int(record.seed) == seed:
            return index
    raise StabilityContractError(f"reference seed {seed} is absent.")


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    """Return linear centered-kernel alignment without materializing Gram matrices."""

    x = _as_finite_float_array(first, name="first", minimum_ndim=2)
    y = _as_finite_float_array(second, name="second", minimum_ndim=2)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise StabilityContractError(
            "Linear CKA inputs must be two-dimensional with aligned rows."
        )
    if x.shape[0] < 2:
        raise StabilityContractError("Linear CKA requires at least two observations.")
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    cross = x_centered.T @ y_centered
    x_covariance = x_centered.T @ x_centered
    y_covariance = y_centered.T @ y_centered
    numerator = float(np.square(cross).sum())
    denominator = float(
        np.sqrt(np.square(x_covariance).sum() * np.square(y_covariance).sum())
    )
    if denominator <= np.finfo(np.float64).tiny:
        raise StabilityContractError(
            "Linear CKA is undefined for a constant or zero-energy representation."
        )
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def orthogonal_procrustes_embedding_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return centered cosine similarity after optimal orthogonal alignment."""

    x = _as_finite_float_array(first, name="first", minimum_ndim=2)
    y = _as_finite_float_array(second, name="second", minimum_ndim=2)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise StabilityContractError(
            "Orthogonal Procrustes inputs must have the same two-dimensional shape."
        )
    if x.shape[0] < 2:
        raise StabilityContractError(
            "Orthogonal Procrustes similarity requires at least two observations."
        )
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    x_norm = float(np.linalg.norm(x_centered))
    y_norm = float(np.linalg.norm(y_centered))
    if min(x_norm, y_norm) <= np.finfo(np.float64).tiny:
        raise StabilityContractError(
            "Orthogonal Procrustes similarity is undefined for a constant embedding."
        )
    rotation, _ = orthogonal_procrustes(y_centered, x_centered)
    aligned = y_centered @ rotation
    similarity = float(np.sum(x_centered * aligned) / (x_norm * y_norm))
    return float(np.clip(similarity, -1.0, 1.0))


@dataclass(frozen=True)
class EmbeddingStability:
    seeds: tuple[int, ...]
    linear_cka: np.ndarray = field(repr=False)
    orthogonal_procrustes_similarity: np.ndarray | None = field(
        default=None, repr=False
    )
    spread_label: str = ENSEMBLE_SPREAD_LABEL
    uncertainty_label: str = SEED_UNCERTAINTY_LABEL


def embedding_stability(
    records: Sequence[FixedSeedArray],
    *,
    include_orthogonal_procrustes: bool = True,
) -> EmbeddingStability:
    """Calculate all pairwise embedding similarities for exactly five seeds."""

    ordered = _validate_five_seed_arrays(
        records, name="embedding stability", require_same_shape=False
    )
    arrays = [
        _as_finite_float_array(record.values, name="embedding", minimum_ndim=2)
        for record in ordered
    ]
    if any(array.ndim != 2 for array in arrays):
        raise StabilityContractError("Each embedding must be two-dimensional.")
    cka = np.eye(EXPECTED_SEED_COUNT, dtype=np.float64)
    procrustes = (
        np.eye(EXPECTED_SEED_COUNT, dtype=np.float64)
        if include_orthogonal_procrustes
        else None
    )
    for left in range(EXPECTED_SEED_COUNT):
        for right in range(left + 1, EXPECTED_SEED_COUNT):
            cka[left, right] = cka[right, left] = linear_cka(
                arrays[left], arrays[right]
            )
            if procrustes is not None:
                procrustes[left, right] = procrustes[right, left] = (
                    orthogonal_procrustes_embedding_similarity(
                        arrays[left], arrays[right]
                    )
                )
    return EmbeddingStability(
        seeds=tuple(int(record.seed) for record in ordered),
        linear_cka=cka,
        orthogonal_procrustes_similarity=procrustes,
    )


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Return deterministic average-rank Spearman correlation.

    As in the standard definition, the result is ``nan`` when either input is
    constant.  Head matching maps such undefined cells to the lowest matching
    score while retaining ``nan`` in the reported signature matrix.
    """

    x = _as_finite_float_array(first, name="first").reshape(-1)
    y = _as_finite_float_array(second, name="second").reshape(-1)
    if x.shape != y.shape or len(x) < 2:
        raise StabilityContractError(
            "Spearman inputs must be aligned vectors with at least two values."
        )
    x_rank = np.asarray(rankdata(x, method="average"), dtype=np.float64)
    y_rank = np.asarray(rankdata(y, method="average"), dtype=np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    if denominator <= np.finfo(np.float64).tiny:
        return float("nan")
    return float(np.clip(np.dot(x_rank, y_rank) / denominator, -1.0, 1.0))


def attention_head_signature(
    attention: np.ndarray,
    *,
    content_logits: np.ndarray | None = None,
    positional_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Build a reproducible fixed-data signature for every attention head.

    Each supplied channel must have shape ``[fixed_edges, heads]``.  Channels
    are independently centered and scaled per head before being concatenated,
    preventing an arbitrary logit scale from dominating matching.
    """

    attention_array = _as_finite_float_array(
        attention, name="attention", minimum_ndim=2
    )
    if attention_array.ndim != 2:
        raise StabilityContractError("attention must have shape [items, heads].")
    channels = [attention_array]
    for values, name in (
        (content_logits, "content_logits"),
        (positional_bias, "positional_bias"),
    ):
        if values is None:
            continue
        array = _as_finite_float_array(values, name=name, minimum_ndim=2)
        if array.shape != attention_array.shape:
            raise StabilityContractError(
                f"{name} must align exactly to attention items and heads."
            )
        channels.append(array)
    normalized: list[np.ndarray] = []
    for channel in channels:
        centered = channel - channel.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0, ddof=0, keepdims=True)
        safe_scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
        normalized.append(centered / safe_scale)
    return np.concatenate(normalized, axis=0)


def _column_spearman_matrix(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape != candidate.shape:
        raise StabilityContractError(
            "Head signatures must have the same [signature_items, heads] shape."
        )
    heads = reference.shape[1]
    similarity = np.empty((heads, heads), dtype=np.float64)
    for reference_head in range(heads):
        for candidate_head in range(heads):
            similarity[reference_head, candidate_head] = spearman_correlation(
                reference[:, reference_head], candidate[:, candidate_head]
            )
    return similarity


def _deterministic_hungarian_permutation(similarity: np.ndarray) -> np.ndarray:
    heads = similarity.shape[0]
    safe_similarity = np.where(np.isnan(similarity), -1.0, similarity)
    # A base-(H+1) fractional code resolves exact assignment ties
    # lexicographically by candidate head, beginning with reference head zero.
    row = np.arange(heads, dtype=np.float64)[:, None]
    column = np.arange(heads, dtype=np.float64)[None, :]
    tie_code = column / np.power(float(heads + 1), row + 1.0)
    cost = -safe_similarity + 1e-12 * tie_code
    row_index, column_index = linear_sum_assignment(cost)
    permutation = np.empty(heads, dtype=np.int64)
    permutation[row_index] = column_index
    return permutation


@dataclass(frozen=True)
class HeadAlignment:
    seeds: tuple[int, ...]
    reference_seed: int
    reference_to_seed_head: np.ndarray = field(repr=False)
    signature_spearman: np.ndarray = field(repr=False)
    matched_signature_spearman: np.ndarray = field(repr=False)
    matching_method: str = "Hungarian maximum fixed-signature Spearman"
    spread_label: str = ENSEMBLE_SPREAD_LABEL
    uncertainty_label: str = SEED_UNCERTAINTY_LABEL

    @property
    def n_heads(self) -> int:
        return int(self.reference_to_seed_head.shape[1])


def match_attention_heads(
    signature_records: Sequence[FixedSeedArray],
    *,
    reference_seed: int = 0,
) -> HeadAlignment:
    """Align every seed's heads to a reference using fixed-data signatures."""

    ordered = _validate_five_seed_arrays(
        signature_records, name="attention-head matching", require_same_shape=True
    )
    signatures = [
        _as_finite_float_array(record.values, name="head signature", minimum_ndim=2)
        for record in ordered
    ]
    if any(array.ndim != 2 for array in signatures):
        raise StabilityContractError(
            "Each attention-head signature must be two-dimensional."
        )
    reference_index = _seed_index(ordered, int(reference_seed))
    heads = signatures[reference_index].shape[1]
    if heads < 1:
        raise StabilityContractError("At least one attention head is required.")
    permutations = np.empty((EXPECTED_SEED_COUNT, heads), dtype=np.int64)
    matrices = np.empty(
        (EXPECTED_SEED_COUNT, heads, heads), dtype=np.float64
    )
    matched = np.empty((EXPECTED_SEED_COUNT, heads), dtype=np.float64)
    reference = signatures[reference_index]
    for seed_index, signature in enumerate(signatures):
        matrix = _column_spearman_matrix(reference, signature)
        matrices[seed_index] = matrix
        if seed_index == reference_index:
            permutation = np.arange(heads, dtype=np.int64)
        else:
            permutation = _deterministic_hungarian_permutation(matrix)
        permutations[seed_index] = permutation
        matched[seed_index] = matrix[np.arange(heads), permutation]
    matched[reference_index] = 1.0
    return HeadAlignment(
        seeds=tuple(int(record.seed) for record in ordered),
        reference_seed=int(reference_seed),
        reference_to_seed_head=permutations,
        signature_spearman=matrices,
        matched_signature_spearman=matched,
    )


def _validate_alignment(
    records: Sequence[FixedSeedArray],
    alignment: HeadAlignment,
    *,
    name: str,
) -> tuple[FixedSeedArray, ...]:
    ordered = _validate_five_seed_arrays(
        records, name=name, require_same_shape=True
    )
    seeds = tuple(int(record.seed) for record in ordered)
    if seeds != alignment.seeds:
        raise StabilityContractError(f"{name} seeds do not match the head alignment.")
    shape = np.asarray(ordered[0].values).shape
    if len(shape) != 2 or shape[1] != alignment.n_heads:
        raise StabilityContractError(
            f"{name} values must have shape [fixed_items, aligned_heads]."
        )
    return ordered


def _top_indices(values: np.ndarray, top_k: int) -> np.ndarray:
    indices = np.arange(len(values), dtype=np.int64)
    order = np.lexsort((indices, -values))
    return np.sort(order[:top_k])


def _resolve_top_k(n_items: int, *, top_k: int | None, top_fraction: float) -> int:
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, Integral):
            raise StabilityContractError("top_k must be an integer when supplied.")
        resolved = int(top_k)
    else:
        if not np.isfinite(top_fraction) or not (0.0 < top_fraction <= 1.0):
            raise StabilityContractError("top_fraction must be in (0, 1].")
        resolved = max(1, int(np.ceil(n_items * top_fraction)))
    if resolved <= 0 or resolved > n_items:
        raise StabilityContractError("top_k must be in [1, number of fixed items].")
    return resolved


@dataclass(frozen=True)
class MatchedHeadAttentionStability:
    seeds: tuple[int, ...]
    reference_seed: int
    top_k: int
    matched_head_spearman: np.ndarray = field(repr=False)
    matched_head_top_edge_jaccard: np.ndarray = field(repr=False)
    spread_label: str = ENSEMBLE_SPREAD_LABEL
    uncertainty_label: str = SEED_UNCERTAINTY_LABEL


def matched_head_attention_stability(
    attention_records: Sequence[FixedSeedArray],
    alignment: HeadAlignment,
    *,
    top_k: int | None = None,
    top_fraction: float = 0.05,
) -> MatchedHeadAttentionStability:
    """Compare aligned head attention profiles and deterministic top-edge sets."""

    ordered = _validate_alignment(
        attention_records, alignment, name="matched-head attention stability"
    )
    arrays = [np.asarray(record.values, dtype=np.float64) for record in ordered]
    reference_index = _seed_index(ordered, alignment.reference_seed)
    reference = arrays[reference_index]
    resolved_top_k = _resolve_top_k(
        len(reference), top_k=top_k, top_fraction=top_fraction
    )
    spearman = np.empty((EXPECTED_SEED_COUNT, alignment.n_heads), dtype=np.float64)
    jaccard = np.empty_like(spearman)
    for seed_index, candidate in enumerate(arrays):
        for reference_head in range(alignment.n_heads):
            candidate_head = alignment.reference_to_seed_head[
                seed_index, reference_head
            ]
            if seed_index == reference_index:
                spearman[seed_index, reference_head] = 1.0
                jaccard[seed_index, reference_head] = 1.0
                continue
            spearman[seed_index, reference_head] = spearman_correlation(
                reference[:, reference_head], candidate[:, candidate_head]
            )
            reference_top = _top_indices(
                reference[:, reference_head], resolved_top_k
            )
            candidate_top = _top_indices(
                candidate[:, candidate_head], resolved_top_k
            )
            intersection = len(np.intersect1d(reference_top, candidate_top))
            union = len(np.union1d(reference_top, candidate_top))
            jaccard[seed_index, reference_head] = intersection / union
    return MatchedHeadAttentionStability(
        seeds=alignment.seeds,
        reference_seed=alignment.reference_seed,
        top_k=resolved_top_k,
        matched_head_spearman=spearman,
        matched_head_top_edge_jaccard=jaccard,
    )


@dataclass(frozen=True)
class PositionalBiasStability:
    seeds: tuple[int, ...]
    reference_seed: int
    matched_head_spearman: np.ndarray = field(repr=False)
    spread_label: str = ENSEMBLE_SPREAD_LABEL
    uncertainty_label: str = SEED_UNCERTAINTY_LABEL


def positional_bias_response_stability(
    positional_bias_records: Sequence[FixedSeedArray],
    alignment: HeadAlignment,
) -> PositionalBiasStability:
    """Correlate fixed-probe positional-bias responses after head alignment."""

    ordered = _validate_alignment(
        positional_bias_records, alignment, name="positional-bias stability"
    )
    arrays = [np.asarray(record.values, dtype=np.float64) for record in ordered]
    reference_index = _seed_index(ordered, alignment.reference_seed)
    reference = arrays[reference_index]
    correlations = np.empty(
        (EXPECTED_SEED_COUNT, alignment.n_heads), dtype=np.float64
    )
    for seed_index, candidate in enumerate(arrays):
        for reference_head in range(alignment.n_heads):
            if seed_index == reference_index:
                correlations[seed_index, reference_head] = 1.0
            else:
                candidate_head = alignment.reference_to_seed_head[
                    seed_index, reference_head
                ]
                correlations[seed_index, reference_head] = spearman_correlation(
                    reference[:, reference_head], candidate[:, candidate_head]
                )
    return PositionalBiasStability(
        seeds=alignment.seeds,
        reference_seed=alignment.reference_seed,
        matched_head_spearman=correlations,
    )


@dataclass(frozen=True)
class ContentPositionAgreement:
    seeds: tuple[int, ...]
    reference_seed: int
    within_seed_spearman: np.ndarray = field(repr=False)
    same_sign_fraction: np.ndarray = field(repr=False)
    content_absolute_fraction: np.ndarray = field(repr=False)
    spread_label: str = ENSEMBLE_SPREAD_LABEL
    uncertainty_label: str = SEED_UNCERTAINTY_LABEL


def content_position_contribution_agreement(
    content_logit_records: Sequence[FixedSeedArray],
    positional_bias_records: Sequence[FixedSeedArray],
    alignment: HeadAlignment,
    *,
    epsilon: float = 1e-12,
) -> ContentPositionAgreement:
    """Describe content-versus-position contributions on aligned fixed edges."""

    content = _validate_alignment(
        content_logit_records, alignment, name="content-logit contribution"
    )
    position = _validate_alignment(
        positional_bias_records, alignment, name="positional-bias contribution"
    )
    if content[0].fixed_input_ids != position[0].fixed_input_ids:
        raise StabilityContractError(
            "Content logits and positional biases must use identical fixed inputs."
        )
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise StabilityContractError("epsilon must be positive and finite.")
    spearman = np.empty((EXPECTED_SEED_COUNT, alignment.n_heads), dtype=np.float64)
    same_sign = np.empty_like(spearman)
    content_fraction = np.empty_like(spearman)
    for seed_index in range(EXPECTED_SEED_COUNT):
        content_values = np.asarray(content[seed_index].values, dtype=np.float64)
        position_values = np.asarray(position[seed_index].values, dtype=np.float64)
        for reference_head in range(alignment.n_heads):
            head = alignment.reference_to_seed_head[seed_index, reference_head]
            content_head = content_values[:, head]
            position_head = position_values[:, head]
            spearman[seed_index, reference_head] = spearman_correlation(
                content_head, position_head
            )
            same_sign[seed_index, reference_head] = float(
                np.mean(np.sign(content_head) == np.sign(position_head))
            )
            content_magnitude = float(np.mean(np.abs(content_head)))
            position_magnitude = float(np.mean(np.abs(position_head)))
            content_fraction[seed_index, reference_head] = content_magnitude / (
                content_magnitude + position_magnitude + epsilon
            )
    return ContentPositionAgreement(
        seeds=alignment.seeds,
        reference_seed=alignment.reference_seed,
        within_seed_spearman=spearman,
        same_sign_fraction=same_sign,
        content_absolute_fraction=content_fraction,
    )


@dataclass(frozen=True)
class RelationshipEnsembleSummary:
    relationship_ids: tuple[object, ...]
    seeds: tuple[int, ...]
    mean: np.ndarray = field(repr=False)
    standard_deviation: np.ndarray = field(repr=False)
    median: np.ndarray = field(repr=False)
    minimum: np.ndarray = field(repr=False)
    maximum: np.ndarray = field(repr=False)
    quantile_levels: tuple[float, ...]
    quantiles: np.ndarray = field(repr=False)
    support_count: np.ndarray = field(repr=False)
    standard_deviation_ddof: int = 1
    spread_label: str = ENSEMBLE_SPREAD_LABEL
    uncertainty_label: str = SEED_UNCERTAINTY_LABEL
    calibrated_confidence_interval: bool = False

    def to_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, relationship_id in enumerate(self.relationship_ids):
            row: dict[str, object] = {
                "relationship_id": relationship_id,
                "ensemble_mean": float(self.mean[index]),
                "ensemble_standard_deviation": float(
                    self.standard_deviation[index]
                ),
                "median": float(self.median[index]),
                "minimum": float(self.minimum[index]),
                "maximum": float(self.maximum[index]),
                "seed_support_count": int(self.support_count[index]),
                "spread_label": self.spread_label,
                "uncertainty_label": self.uncertainty_label,
            }
            for quantile_index, level in enumerate(self.quantile_levels):
                row[f"empirical_quantile_{level:g}"] = float(
                    self.quantiles[index, quantile_index]
                )
            rows.append(row)
        return rows


def summarize_relationship_ensemble(
    records: Sequence[FixedSeedArray],
    *,
    support_records: Sequence[FixedSeedArray] | None = None,
    support_magnitude_threshold: float = 0.0,
    quantile_levels: Sequence[float] = (0.05, 0.25, 0.75, 0.95),
) -> RelationshipEnsembleSummary:
    """Summarize scalar relationship scores across exactly five aligned seeds.

    If explicit binary ``support_records`` are omitted, a seed supports a
    relationship when its absolute score is strictly greater than
    ``support_magnitude_threshold``.
    """

    ordered = _validate_five_seed_arrays(
        records, name="relationship ensemble", require_same_shape=True
    )
    arrays = [np.asarray(record.values, dtype=np.float64) for record in ordered]
    if any(array.ndim != 1 for array in arrays):
        raise StabilityContractError(
            "Relationship ensemble values must be one scalar per fixed identifier."
        )
    values = np.stack(arrays, axis=0)
    if not np.isfinite(support_magnitude_threshold) or support_magnitude_threshold < 0:
        raise StabilityContractError(
            "support_magnitude_threshold must be finite and non-negative."
        )
    levels = np.asarray(tuple(quantile_levels), dtype=np.float64)
    if (
        levels.ndim != 1
        or len(levels) == 0
        or not np.isfinite(levels).all()
        or np.any(levels < 0.0)
        or np.any(levels > 1.0)
        or np.any(levels[1:] <= levels[:-1])
    ):
        raise StabilityContractError(
            "quantile_levels must be a strictly increasing sequence in [0, 1]."
        )
    if support_records is None:
        support = np.abs(values) > support_magnitude_threshold
    else:
        support_ordered = _validate_five_seed_arrays(
            support_records,
            name="relationship support",
            require_same_shape=True,
        )
        if tuple(int(record.seed) for record in support_ordered) != tuple(
            int(record.seed) for record in ordered
        ) or support_ordered[0].fixed_input_ids != ordered[0].fixed_input_ids:
            raise StabilityContractError(
                "Relationship support records must align to scores and seeds."
            )
        support_values = np.stack(
            [np.asarray(record.values) for record in support_ordered], axis=0
        )
        if support_values.shape != values.shape:
            raise StabilityContractError(
                "Relationship support records must be scalar and aligned."
            )
        if not np.isin(support_values, (0.0, 1.0)).all():
            raise StabilityContractError(
                "Relationship support records must contain only binary masks."
            )
        support = support_values.astype(bool)
    return RelationshipEnsembleSummary(
        relationship_ids=ordered[0].fixed_input_ids,
        seeds=tuple(int(record.seed) for record in ordered),
        mean=values.mean(axis=0),
        standard_deviation=values.std(axis=0, ddof=1),
        median=np.median(values, axis=0),
        minimum=values.min(axis=0),
        maximum=values.max(axis=0),
        quantile_levels=tuple(float(level) for level in levels.tolist()),
        quantiles=np.quantile(values, levels, axis=0).T,
        support_count=support.sum(axis=0).astype(np.int64),
    )


@dataclass(frozen=True)
class MutualRoutingScores:
    pair_ids: tuple[tuple[int, int], ...]
    scores: np.ndarray = field(repr=False)


def mutual_routing_scores(
    edge_index: np.ndarray,
    attention: np.ndarray,
) -> MutualRoutingScores:
    """Calculate degree-adjusted mutual routing for reciprocal directed edges.

    For ``source -> receiver``, ``R = receiver_indegree * mean_head_attention``.
    The undirected pair score is the minimum of its two reciprocal ``R`` values.
    This is a descriptive model-routing score, not causal influence.
    """

    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype.kind not in "iu":
        raise StabilityContractError(
            "edge_index must be integral with shape [2, directed_edges]."
        )
    if edges.shape[1] == 0:
        raise StabilityContractError("Mutual routing requires at least one edge.")
    source = edges[0].astype(np.int64, copy=False)
    receiver = edges[1].astype(np.int64, copy=False)
    if edges.min() < 0 or np.any(source == receiver):
        raise StabilityContractError(
            "Mutual routing edges must be non-negative and loop-free."
        )
    weights = _as_finite_float_array(attention, name="attention")
    if weights.ndim == 1:
        mean_attention = weights
    elif weights.ndim == 2:
        mean_attention = weights.mean(axis=1)
    else:
        raise StabilityContractError(
            "attention must have shape [directed_edges] or [directed_edges, heads]."
        )
    if len(mean_attention) != edges.shape[1] or np.any(mean_attention < 0.0):
        raise StabilityContractError(
            "attention must align to edges and be non-negative."
        )
    n_nodes = int(edges.max()) + 1
    codes = source * n_nodes + receiver
    if len(np.unique(codes)) != len(codes):
        raise StabilityContractError("Mutual routing requires unique directed edges.")
    degree = np.bincount(receiver, minlength=n_nodes).astype(np.float64)
    routing = degree[receiver] * mean_attention
    routing_by_code = {
        int(code): float(value) for code, value in zip(codes.tolist(), routing.tolist())
    }
    pairs: list[tuple[int, int]] = []
    scores: list[float] = []
    for edge_source, edge_receiver in zip(source.tolist(), receiver.tolist()):
        low, high = sorted((int(edge_source), int(edge_receiver)))
        if edge_source != low:
            continue
        reverse_code = high * n_nodes + low
        forward_code = low * n_nodes + high
        if reverse_code not in routing_by_code:
            continue
        pairs.append((low, high))
        scores.append(
            min(routing_by_code[forward_code], routing_by_code[reverse_code])
        )
    if not pairs:
        raise StabilityContractError(
            "Mutual routing requires at least one reciprocal edge pair."
        )
    order = np.lexsort(
        (
            np.asarray([pair[1] for pair in pairs]),
            np.asarray([pair[0] for pair in pairs]),
        )
    )
    return MutualRoutingScores(
        pair_ids=tuple(pairs[index] for index in order.tolist()),
        scores=np.asarray(scores, dtype=np.float64)[order],
    )


def mutual_routing_pair_stability(
    records: Sequence[FixedSeedArray],
    *,
    top_k: int | None = None,
    top_fraction: float = 0.05,
    quantile_levels: Sequence[float] = (0.05, 0.25, 0.75, 0.95),
) -> RelationshipEnsembleSummary:
    """Summarize mutual-routing pairs and count top-pair support across seeds."""

    ordered = _validate_five_seed_arrays(
        records, name="mutual-routing stability", require_same_shape=True
    )
    arrays = [np.asarray(record.values, dtype=np.float64) for record in ordered]
    if any(array.ndim != 1 for array in arrays):
        raise StabilityContractError(
            "Mutual-routing records must contain one scalar per pair."
        )
    resolved_top_k = _resolve_top_k(
        len(arrays[0]), top_k=top_k, top_fraction=top_fraction
    )
    support_records = []
    for record, values in zip(ordered, arrays):
        support = np.zeros(len(values), dtype=np.float64)
        support[_top_indices(values, resolved_top_k)] = 1.0
        support_records.append(
            FixedSeedArray(
                seed=int(record.seed),
                fixed_input_ids=record.fixed_input_ids,
                values=support,
            )
        )
    return summarize_relationship_ensemble(
        ordered,
        support_records=support_records,
        quantile_levels=quantile_levels,
    )


@dataclass(frozen=True)
class SelectedGradientStability:
    summary: RelationshipEnsembleSummary
    ensemble_median_sign: np.ndarray = field(repr=False)
    consistent_sign_support_count: np.ndarray = field(repr=False)
    consistent_sign_fraction: np.ndarray = field(repr=False)
    pairwise_seed_spearman: np.ndarray = field(repr=False)
    analysis_scope: str = "selected relationships only; no exhaustive Jacobian"
    spread_label: str = ENSEMBLE_SPREAD_LABEL
    uncertainty_label: str = SEED_UNCERTAINTY_LABEL


def selected_gradient_stability(
    records: Sequence[FixedSeedArray],
    *,
    magnitude_threshold: float = 0.0,
    quantile_levels: Sequence[float] = (0.05, 0.25, 0.75, 0.95),
) -> SelectedGradientStability:
    """Summarize caller-selected source-gene to target-gene gradients only."""

    ordered = _validate_five_seed_arrays(
        records, name="selected-gradient stability", require_same_shape=True
    )
    arrays = [np.asarray(record.values, dtype=np.float64) for record in ordered]
    if any(array.ndim != 1 for array in arrays):
        raise StabilityContractError(
            "Selected gradients must contain one scalar per selected relationship."
        )
    if not np.isfinite(magnitude_threshold) or magnitude_threshold < 0:
        raise StabilityContractError(
            "magnitude_threshold must be finite and non-negative."
        )
    values = np.stack(arrays, axis=0)
    supported = np.abs(values) > magnitude_threshold
    support_records = tuple(
        FixedSeedArray(
            seed=int(record.seed),
            fixed_input_ids=record.fixed_input_ids,
            values=supported[index].astype(np.float64),
        )
        for index, record in enumerate(ordered)
    )
    summary = summarize_relationship_ensemble(
        ordered,
        support_records=support_records,
        quantile_levels=quantile_levels,
    )
    median_sign = np.sign(summary.median)
    sign_match = (np.sign(values) == median_sign[None, :]) & supported
    sign_count = sign_match.sum(axis=0).astype(np.int64)
    sign_fraction = np.divide(
        sign_count,
        summary.support_count,
        out=np.zeros_like(sign_count, dtype=np.float64),
        where=summary.support_count > 0,
    )
    pairwise = np.eye(EXPECTED_SEED_COUNT, dtype=np.float64)
    for left in range(EXPECTED_SEED_COUNT):
        for right in range(left + 1, EXPECTED_SEED_COUNT):
            value = (
                spearman_correlation(values[left], values[right])
                if values.shape[1] >= 2
                else float("nan")
            )
            pairwise[left, right] = pairwise[right, left] = value
    return SelectedGradientStability(
        summary=summary,
        ensemble_median_sign=median_sign,
        consistent_sign_support_count=sign_count,
        consistent_sign_fraction=sign_fraction,
        pairwise_seed_spearman=pairwise,
    )


__all__ = [
    "ENSEMBLE_SPREAD_LABEL",
    "EXPECTED_SEED_COUNT",
    "SEED_UNCERTAINTY_LABEL",
    "ContentPositionAgreement",
    "EmbeddingStability",
    "FixedSeedArray",
    "HeadAlignment",
    "MatchedHeadAttentionStability",
    "MutualRoutingScores",
    "PositionalBiasStability",
    "RelationshipEnsembleSummary",
    "SelectedGradientStability",
    "StabilityContractError",
    "attention_head_signature",
    "content_position_contribution_agreement",
    "embedding_stability",
    "linear_cka",
    "match_attention_heads",
    "matched_head_attention_stability",
    "mutual_routing_pair_stability",
    "mutual_routing_scores",
    "orthogonal_procrustes_embedding_similarity",
    "positional_bias_response_stability",
    "selected_gradient_stability",
    "spearman_correlation",
    "summarize_relationship_ensemble",
]
