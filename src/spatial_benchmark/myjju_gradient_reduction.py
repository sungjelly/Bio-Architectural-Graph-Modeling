"""Pure reducers for the frozen MyJJu GeneMAE gradient audit.

This module consumes already verified seven-seed shard arrays.  It never loads
patient data, checkpoints, or graphs.  Signed gradients are averaged across
seeds before magnitudes or ranks are computed, and cores always receive equal
weight.

The outputs describe model behaviour.  They do not validate a biological
mechanism or causal effect.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .myjju_gradient_audit import (
    FROZEN_LOCKED_DIRECTED_PAIRS,
    FROZEN_MARKER_GENES,
    FROZEN_NONNEGLIGIBLE_ABSOLUTE_EFFECT,
    FROZEN_SOURCE_SELECTED_PAIRS,
    FROZEN_TARGET_GENES,
    GradientAuditError,
    deterministic_spearman,
    faithfulness_metrics,
)


SEEDS = tuple(range(7))
CORE_COUNT = 10
MASK_REPLICATES = (0, 1, 2)
PERTURBATION_SCALES = (0.10, 0.25)
GENE_MEAN_GAIN_THRESHOLD = 0.02
GRAPH_PREDICTION_GAIN_THRESHOLD = 0.02
FAVORING_CORE_THRESHOLD = 8
RANK_STABILITY_THRESHOLD = 0.70
SIGNED_SEED_PREVALENCE_THRESHOLD = 6
SIGNED_CORE_PREVALENCE_THRESHOLD = 8
FAITHFULNESS_SIGN_THRESHOLD = 0.80
FAITHFULNESS_ERROR_RATIO_THRESHOLD = 0.50
FAITHFULNESS_TESTABLE_FRACTION_THRESHOLD = 0.90
GRAPH_STRUCTURE_GAIN_THRESHOLD = 0.25
MATCHED_NULL_DRAWS = 10_000
MATCHED_NULL_SEED = 20_260_731
DENOMINATOR_FLOOR = 1e-12


class GradientReductionError(RuntimeError):
    """Raised when complete, aligned audit evidence cannot be reduced."""


def _finite_array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if shape is not None and array.shape != shape:
        raise GradientReductionError(
            f"{label} has shape {array.shape}, expected {shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise GradientReductionError(f"{label} must be a real numeric array")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise GradientReductionError(f"{label} contains nonfinite values")
    return result


def _bool_array(
    value: Any,
    *,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if shape is not None and array.shape != shape:
        raise GradientReductionError(
            f"{label} has shape {array.shape}, expected {shape}"
        )
    if array.dtype != np.bool_:
        raise GradientReductionError(f"{label} must have boolean dtype")
    return array


def _identical(
    seed_arrays: Sequence[Mapping[str, np.ndarray]],
    name: str,
) -> np.ndarray:
    reference = np.asarray(seed_arrays[0][name])
    for seed, arrays in enumerate(seed_arrays[1:], start=1):
        candidate = np.asarray(arrays[name])
        if candidate.dtype != reference.dtype or not np.array_equal(
            candidate, reference
        ):
            raise GradientReductionError(
                f"{name} is not exactly invariant across seeds 0 and {seed}"
            )
    return reference


def mean_huber(
    truth: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    *,
    delta: float = 1.0,
) -> float:
    """Return the mean element-wise Huber loss."""

    target = _finite_array(truth, label="Huber truth")
    estimate = _finite_array(prediction, label="Huber prediction")
    if target.shape != estimate.shape or target.size < 1:
        raise GradientReductionError(
            "Huber truth and prediction must have one nonempty common shape"
        )
    if not math.isfinite(delta) or delta <= 0:
        raise GradientReductionError("Huber delta must be positive and finite")
    difference = np.abs(estimate - target)
    loss = np.where(
        difference <= delta,
        0.5 * difference**2,
        delta * (difference - 0.5 * delta),
    )
    return float(np.mean(loss, dtype=np.float64))


def relative_improvement(reference: float, candidate: float) -> float:
    """Return ``(reference - candidate) / reference`` fail-closed."""

    reference_value = float(reference)
    candidate_value = float(candidate)
    if (
        not math.isfinite(reference_value)
        or not math.isfinite(candidate_value)
        or reference_value <= 0
    ):
        raise GradientReductionError(
            "relative-improvement reference must be positive and finite"
        )
    return (reference_value - candidate_value) / reference_value


def _offdiagonal_values(matrix: np.ndarray) -> np.ndarray:
    array = _finite_array(matrix, label="gradient matrix")
    if array.shape != (len(FROZEN_MARKER_GENES),) * 2:
        raise GradientReductionError("gradient matrix must have shape [39, 39]")
    keep = ~np.eye(array.shape[0], dtype=bool)
    return np.abs(array[keep])


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    try:
        return deterministic_spearman(first, second)
    except GradientAuditError:
        return None


def _median_pairwise_spearman(vectors: Sequence[np.ndarray]) -> float | None:
    values: list[float] = []
    for first_index, second_index in combinations(range(len(vectors)), 2):
        correlation = _safe_spearman(
            np.asarray(vectors[first_index], dtype=np.float64),
            np.asarray(vectors[second_index], dtype=np.float64),
        )
        if correlation is None:
            return None
        values.append(correlation)
    if not values:
        return None
    return float(np.median(values))


def structure_statistic(matrix: np.ndarray) -> dict[str, float]:
    """Compute the frozen locked-pair enrichment statistic ``S``."""

    value = _finite_array(
        matrix,
        label="structure matrix",
        shape=(len(FROZEN_MARKER_GENES), len(FROZEN_MARKER_GENES)),
    )
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    locked: list[float] = []
    nonlocked: list[float] = []
    for target in FROZEN_TARGET_GENES:
        target_position = positions[target]
        locked_sources = set(FROZEN_SOURCE_SELECTED_PAIRS[target])
        for source in FROZEN_MARKER_GENES:
            if source == target:
                continue
            magnitude = abs(float(value[target_position, positions[source]]))
            if source in locked_sources:
                locked.append(magnitude)
            else:
                nonlocked.append(magnitude)
    if len(locked) != len(FROZEN_LOCKED_DIRECTED_PAIRS) or not nonlocked:
        raise GradientReductionError("locked/nonlocked pair inventory changed")
    numerator = float(np.median(locked))
    denominator = float(np.median(nonlocked))
    return {
        "locked_median_absolute_gradient": numerator,
        "nonlocked_median_absolute_gradient": denominator,
        "S": numerator / max(denominator, DENOMINATOR_FLOOR),
    }


def _target_eligibility(
    *,
    truth: np.ndarray,
    mask: np.ndarray,
    observed_prediction: np.ndarray,
    permuted_prediction: np.ndarray,
    offsets: np.ndarray,
    gene_mean: np.ndarray,
) -> dict[str, Any]:
    core_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for core_index in range(CORE_COUNT):
        selection = slice(int(offsets[core_index]), int(offsets[core_index + 1]))
        for target_index, target in enumerate(FROZEN_TARGET_GENES):
            observed_losses: list[float] = []
            permuted_losses: list[float] = []
            baseline_losses: list[float] = []
            for replicate in MASK_REPLICATES:
                selected = mask[selection, replicate, target_index]
                if not bool(selected.any()):
                    raise GradientReductionError(
                        f"core {core_index} {target} mask {replicate} is empty"
                    )
                target_values = truth[selection, target_index][selected]
                observed_losses.append(
                    mean_huber(
                        target_values,
                        observed_prediction[
                            selection, replicate, target_index
                        ][selected],
                    )
                )
                permuted_losses.append(
                    mean_huber(
                        target_values,
                        permuted_prediction[
                            selection, replicate, target_index
                        ][selected],
                    )
                )
                baseline_losses.append(
                    mean_huber(
                        target_values,
                        np.full(
                            target_values.shape,
                            gene_mean[core_index, target_index],
                            dtype=np.float64,
                        ),
                    )
                )
            observed = float(np.mean(observed_losses))
            permuted = float(np.mean(permuted_losses))
            baseline = float(np.mean(baseline_losses))
            core_rows.append(
                {
                    "core_index": core_index,
                    "target": target,
                    "observed_huber": observed,
                    "permuted_huber": permuted,
                    "gene_mean_huber": baseline,
                    "gene_mean_gain": relative_improvement(baseline, observed),
                    "graph_gain": relative_improvement(permuted, observed),
                    "model_beats_gene_mean": observed < baseline,
                    "observed_beats_permuted": observed < permuted,
                }
            )

    for target in FROZEN_TARGET_GENES:
        selected_rows = [row for row in core_rows if row["target"] == target]
        observed = float(np.mean([row["observed_huber"] for row in selected_rows]))
        permuted = float(np.mean([row["permuted_huber"] for row in selected_rows]))
        baseline = float(np.mean([row["gene_mean_huber"] for row in selected_rows]))
        baseline_gain = relative_improvement(baseline, observed)
        graph_gain = relative_improvement(permuted, observed)
        baseline_count = sum(
            bool(row["model_beats_gene_mean"]) for row in selected_rows
        )
        graph_count = sum(
            bool(row["observed_beats_permuted"]) for row in selected_rows
        )
        target_rows.append(
            {
                "target": target,
                "equal_core_observed_huber": observed,
                "equal_core_permuted_huber": permuted,
                "equal_core_gene_mean_huber": baseline,
                "gene_mean_relative_improvement": baseline_gain,
                "gene_mean_favoring_core_count": baseline_count,
                "graph_relative_improvement": graph_gain,
                "graph_favoring_core_count": graph_count,
                "predictive_gate_pass": (
                    baseline_gain >= GENE_MEAN_GAIN_THRESHOLD
                    and baseline_count >= FAVORING_CORE_THRESHOLD
                ),
                "graph_use_gate_pass": (
                    graph_gain >= GRAPH_PREDICTION_GAIN_THRESHOLD
                    and graph_count >= FAVORING_CORE_THRESHOLD
                ),
                "eligible": (
                    baseline_gain >= GENE_MEAN_GAIN_THRESHOLD
                    and baseline_count >= FAVORING_CORE_THRESHOLD
                    and graph_gain >= GRAPH_PREDICTION_GAIN_THRESHOLD
                    and graph_count >= FAVORING_CORE_THRESHOLD
                ),
            }
        )
    return {"core_rows": core_rows, "target_rows": target_rows}


def _seed_rank_stability(masked: np.ndarray) -> dict[str, Any]:
    core_rows: list[dict[str, Any]] = []
    for core_index in range(CORE_COUNT):
        vectors = [
            _offdiagonal_values(masked[seed, core_index])
            for seed in range(len(SEEDS))
        ]
        median = _median_pairwise_spearman(vectors)
        core_rows.append(
            {
                "core_index": core_index,
                "median_pairwise_spearman": median,
                "pass": (
                    median is not None
                    and median >= RANK_STABILITY_THRESHOLD
                ),
            }
        )
    pass_count = sum(bool(row["pass"]) for row in core_rows)
    return {
        "core_rows": core_rows,
        "passing_core_count": pass_count,
        "pass": pass_count >= FAVORING_CORE_THRESHOLD,
    }


def _mask_rank_stability(masked_locked: np.ndarray) -> dict[str, Any]:
    ensemble = np.mean(masked_locked, axis=0, dtype=np.float64)
    marker_position = {
        gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)
    }
    core_target_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(FROZEN_TARGET_GENES):
        source_keep = np.ones(len(FROZEN_MARKER_GENES), dtype=bool)
        source_keep[marker_position[target]] = False
        for core_index in range(CORE_COUNT):
            vectors = [
                np.abs(
                    ensemble[
                        core_index, replicate, target_index, source_keep
                    ]
                )
                for replicate in MASK_REPLICATES
            ]
            median = _median_pairwise_spearman(vectors)
            core_target_rows.append(
                {
                    "core_index": core_index,
                    "target": target,
                    "median_pairwise_spearman": median,
                    "pass": (
                        median is not None
                        and median >= RANK_STABILITY_THRESHOLD
                    ),
                }
            )
        rows = [
            row for row in core_target_rows if row["target"] == target
        ]
        pass_count = sum(bool(row["pass"]) for row in rows)
        target_rows.append(
            {
                "target": target,
                "passing_core_count": pass_count,
                "pass": pass_count >= FAVORING_CORE_THRESHOLD,
            }
        )
    return {
        "core_target_rows": core_target_rows,
        "target_rows": target_rows,
        "pass": all(bool(row["pass"]) for row in target_rows),
    }


def _signed_pair_stability(masked: np.ndarray) -> dict[str, Any]:
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    rows: list[dict[str, Any]] = []
    for target, source in FROZEN_LOCKED_DIRECTED_PAIRS:
        values = masked[:, :, positions[target], positions[source]]
        seed_values = np.mean(values, axis=1, dtype=np.float64)
        core_values = np.mean(values, axis=0, dtype=np.float64)
        seed_signs = np.sign(seed_values)
        core_signs = np.sign(core_values)
        sign_counts = {
            "positive_seed_count": int(np.sum(seed_signs == 1)),
            "negative_seed_count": int(np.sum(seed_signs == -1)),
            "zero_seed_count": int(np.sum(seed_signs == 0)),
            "positive_core_count": int(np.sum(core_signs == 1)),
            "negative_core_count": int(np.sum(core_signs == -1)),
            "zero_core_count": int(np.sum(core_signs == 0)),
        }
        selected_direction = 0
        seed_count = 0
        core_count = 0
        for direction in (1, -1):
            label = "positive" if direction == 1 else "negative"
            candidate_seed_count = int(
                sign_counts[f"{label}_seed_count"]
            )
            candidate_core_count = int(
                sign_counts[f"{label}_core_count"]
            )
            if (
                candidate_seed_count >= SIGNED_SEED_PREVALENCE_THRESHOLD
                and candidate_core_count >= SIGNED_CORE_PREVALENCE_THRESHOLD
            ):
                selected_direction = direction
                seed_count = candidate_seed_count
                core_count = candidate_core_count
                break
        rows.append(
            {
                "target": target,
                "source": source,
                "direction": selected_direction,
                "seed_direction_count": seed_count,
                "core_direction_count": core_count,
                **sign_counts,
                "pass": selected_direction != 0,
                "equal_core_equal_seed_signed_gradient": float(
                    np.mean(values, dtype=np.float64)
                ),
            }
        )
    target_rows = []
    for target in FROZEN_TARGET_GENES:
        selected = [row for row in rows if row["target"] == target]
        target_rows.append(
            {
                "target": target,
                "passing_pair_count": sum(bool(row["pass"]) for row in selected),
                "pair_count": len(selected),
                "any_pair_pass": any(bool(row["pass"]) for row in selected),
                "all_pairs_pass": all(bool(row["pass"]) for row in selected),
            }
        )
    return {
        "pair_rows": rows,
        "target_rows": target_rows,
        "passing_pair_count": sum(bool(row["pass"]) for row in rows),
    }


def _faithfulness_case_mask(*, include_diagonal: bool) -> np.ndarray:
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    keep = np.ones((CORE_COUNT, len(FROZEN_TARGET_GENES), 39), dtype=bool)
    if not include_diagonal:
        for target_index, target in enumerate(FROZEN_TARGET_GENES):
            keep[:, target_index, positions[target]] = False
    return keep


def _locked_faithfulness_case_mask() -> np.ndarray:
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    keep = np.zeros((CORE_COUNT, len(FROZEN_TARGET_GENES), 39), dtype=bool)
    target_positions = {
        gene: index for index, gene in enumerate(FROZEN_TARGET_GENES)
    }
    for target, source in FROZEN_LOCKED_DIRECTED_PAIRS:
        keep[:, target_positions[target], positions[source]] = True
    return keep


def _one_faithfulness_summary(
    predicted: np.ndarray,
    actual: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    first = np.asarray(predicted[selected], dtype=np.float64)
    second = np.asarray(actual[selected], dtype=np.float64)
    testable = np.abs(second) > FROZEN_NONNEGLIGIBLE_ABSOLUTE_EFFECT
    testable_fraction = float(np.mean(testable)) if second.size else 0.0
    try:
        metrics = faithfulness_metrics(
            first,
            second,
            minimum_absolute_actual_change=(
                FROZEN_NONNEGLIGIBLE_ABSOLUTE_EFFECT
            ),
        )
        result = {
            "valid": True,
            "case_count": int(second.size),
            "testable_case_count": int(testable.sum()),
            "testable_case_fraction": testable_fraction,
            "spearman": metrics.spearman,
            "sign_agreement": metrics.sign_agreement,
            "median_absolute_error": metrics.median_absolute_error,
            "median_absolute_actual_change": (
                metrics.median_absolute_actual_change
            ),
            "median_absolute_error_ratio": (
                metrics.median_absolute_error_ratio
            ),
        }
    except GradientAuditError as exc:
        result = {
            "valid": False,
            "case_count": int(second.size),
            "testable_case_count": int(testable.sum()),
            "testable_case_fraction": testable_fraction,
            "error": str(exc),
        }
    result["pass"] = bool(
        result["valid"]
        and result["testable_case_fraction"]
        >= FAITHFULNESS_TESTABLE_FRACTION_THRESHOLD
        and result["spearman"] >= RANK_STABILITY_THRESHOLD
        and result["sign_agreement"] >= FAITHFULNESS_SIGN_THRESHOLD
        and result["median_absolute_error_ratio"]
        <= FAITHFULNESS_ERROR_RATIO_THRESHOLD
    )
    return result


def _faithfulness(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> dict[str, Any]:
    ensemble_prediction = np.mean(predicted, axis=0, dtype=np.float64)
    ensemble_actual = np.mean(actual, axis=0, dtype=np.float64)
    primary_keep = _faithfulness_case_mask(include_diagonal=False)
    all_keep = _faithfulness_case_mask(include_diagonal=True)
    locked_keep = _locked_faithfulness_case_mask()
    rows: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    for scale_index, scale in enumerate(PERTURBATION_SCALES):
        primary = _one_faithfulness_summary(
            ensemble_prediction[:, scale_index],
            ensemble_actual[:, scale_index],
            primary_keep,
        )
        rows.append({"scale": scale, **primary})
        secondary.append(
            {
                "scale": scale,
                "scope": "all_including_diagonal",
                **_one_faithfulness_summary(
                    ensemble_prediction[:, scale_index],
                    ensemble_actual[:, scale_index],
                    all_keep,
                ),
            }
        )
        secondary.append(
            {
                "scale": scale,
                "scope": "locked_pairs",
                **_one_faithfulness_summary(
                    ensemble_prediction[:, scale_index],
                    ensemble_actual[:, scale_index],
                    locked_keep,
                ),
            }
        )
    return {
        "primary_offdiagonal_rows": rows,
        "secondary_rows": secondary,
        "pass": all(bool(row["pass"]) for row in rows),
    }


def _graph_structure_null(
    observed: np.ndarray,
    permuted: np.ndarray,
) -> dict[str, Any]:
    core_rows: list[dict[str, Any]] = []
    for core_index in range(CORE_COUNT):
        observed_statistic = structure_statistic(observed[core_index])
        permuted_statistic = structure_statistic(permuted[core_index])
        denominator = max(permuted_statistic["S"], DENOMINATOR_FLOOR)
        gain = (observed_statistic["S"] - permuted_statistic["S"]) / denominator
        core_rows.append(
            {
                "core_index": core_index,
                "observed": observed_statistic,
                "permuted": permuted_statistic,
                "relative_S_improvement": gain,
                "pass": gain >= GRAPH_STRUCTURE_GAIN_THRESHOLD,
            }
        )
    pass_count = sum(bool(row["pass"]) for row in core_rows)
    return {
        "core_rows": core_rows,
        "passing_core_count": pass_count,
        "pass": pass_count >= FAVORING_CORE_THRESHOLD,
    }


def _parameter_randomization(
    trained_reference_core: np.ndarray,
    randomized: np.ndarray,
) -> dict[str, Any]:
    trained = structure_statistic(trained_reference_core)
    controls = []
    for seed_index, random_seed in enumerate(range(9100, 9107)):
        statistic = structure_statistic(randomized[seed_index])
        controls.append(
            {
                "random_model_seed": random_seed,
                **statistic,
                "trained_strictly_greater": trained["S"] > statistic["S"],
            }
        )
    return {
        "trained": trained,
        "controls": controls,
        "pass": all(bool(row["trained_strictly_greater"]) for row in controls),
    }


def _equal_core_pair_magnitudes(ensemble: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(ensemble), axis=0, dtype=np.float64)


def _standardized_target_features(
    prevalence: np.ndarray,
    expression: np.ndarray,
    correlation: np.ndarray,
    *,
    target_marker_position: int,
    target_index: int,
) -> np.ndarray:
    features = np.stack(
        (
            prevalence,
            expression,
            correlation[target_index],
        ),
        axis=1,
    )
    eligible = np.ones(len(FROZEN_MARKER_GENES), dtype=bool)
    eligible[target_marker_position] = False
    selected = features[eligible]
    means = np.mean(selected, axis=0, dtype=np.float64)
    scales = np.std(selected, axis=0, ddof=0, dtype=np.float64)
    standardized = np.zeros_like(features, dtype=np.float64)
    nonconstant = scales > 0
    standardized[:, nonconstant] = (
        features[:, nonconstant] - means[nonconstant]
    ) / scales[nonconstant]
    return standardized


def matched_pair_null(
    *,
    ensemble_observed: np.ndarray,
    source_nonzero_prevalence: np.ndarray,
    source_mean_expression: np.ndarray,
    absolute_raw_pearson: np.ndarray,
    draws: int = MATCHED_NULL_DRAWS,
    random_seed: int = MATCHED_NULL_SEED,
) -> dict[str, Any]:
    """Run the frozen target-preserving expression/co-expression matched null."""

    matrix = _finite_array(
        ensemble_observed,
        label="ensemble observed gradient",
        shape=(CORE_COUNT, 39, 39),
    )
    prevalence = np.mean(
        _finite_array(
            source_nonzero_prevalence,
            label="source prevalence",
            shape=(CORE_COUNT, 39),
        ),
        axis=0,
        dtype=np.float64,
    )
    expression = np.mean(
        _finite_array(
            source_mean_expression,
            label="source mean expression",
            shape=(CORE_COUNT, 39),
        ),
        axis=0,
        dtype=np.float64,
    )
    correlation = np.mean(
        _finite_array(
            absolute_raw_pearson,
            label="absolute raw Pearson",
            shape=(CORE_COUNT, 5, 39),
        ),
        axis=0,
        dtype=np.float64,
    )
    if np.any((prevalence < 0.0) | (prevalence > 1.0)):
        raise GradientReductionError("source prevalence must lie in [0, 1]")
    if np.any(expression < 0.0):
        raise GradientReductionError("source mean expression must be nonnegative")
    if np.any((correlation < 0.0) | (correlation > 1.0 + 1e-7)):
        raise GradientReductionError(
            "absolute raw Pearson correlation must lie in [0, 1]"
        )
    if isinstance(draws, bool) or int(draws) != draws or draws < 1:
        raise GradientReductionError("matched-null draws must be positive integer")
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    target_positions = {
        gene: index for index, gene in enumerate(FROZEN_TARGET_GENES)
    }
    pair_magnitude = _equal_core_pair_magnitudes(matrix)
    observed_values: list[float] = []
    candidate_pools: list[np.ndarray] = []
    candidate_records: list[dict[str, Any]] = []
    for target, source in FROZEN_LOCKED_DIRECTED_PAIRS:
        target_marker = positions[target]
        source_marker = positions[source]
        target_index = target_positions[target]
        standardized = _standardized_target_features(
            prevalence,
            expression,
            correlation,
            target_marker_position=target_marker,
            target_index=target_index,
        )
        excluded = {target_marker}
        excluded.update(
            positions[item] for item in FROZEN_SOURCE_SELECTED_PAIRS[target]
        )
        candidates = np.asarray(
            [index for index in range(39) if index not in excluded],
            dtype=np.int64,
        )
        if candidates.size < 5:
            raise GradientReductionError(
                f"matched-null candidate pool is too small for {target}"
            )
        distances = np.linalg.norm(
            standardized[candidates] - standardized[source_marker],
            axis=1,
        )
        order = np.lexsort((candidates, distances))
        nearest = candidates[order[:5]]
        candidate_pools.append(nearest)
        observed_values.append(float(pair_magnitude[target_marker, source_marker]))
        candidate_records.append(
            {
                "target": target,
                "source": source,
                "candidate_sources": [
                    FROZEN_MARKER_GENES[int(index)] for index in nearest
                ],
                "candidate_distances": [
                    float(distances[int(position)]) for position in order[:5]
                ],
            }
        )
    observed_statistic = float(np.median(observed_values))
    generator = np.random.default_rng(int(random_seed))
    null = np.empty(int(draws), dtype=np.float64)
    for draw in range(int(draws)):
        values = [
            pair_magnitude[
                positions[target],
                int(pool[generator.integers(0, len(pool))]),
            ]
            for (target, _), pool in zip(
                FROZEN_LOCKED_DIRECTED_PAIRS,
                candidate_pools,
                strict=True,
            )
        ]
        null[draw] = np.median(values)
    exceedance = int(np.sum(null >= observed_statistic))
    p_value = (1 + exceedance) / (int(draws) + 1)
    return {
        "observed_statistic": observed_statistic,
        "null_draws": int(draws),
        "null_seed": int(random_seed),
        "null_median": float(np.median(null)),
        "null_95th_percentile": float(
            np.quantile(null, 0.95, method="linear")
        ),
        "upper_tail_p": p_value,
        "exceedance_count": exceedance,
        "candidate_pools": candidate_records,
        "pass": p_value <= 0.05,
    }


def _source_style_reproduction(
    source_unmasked: np.ndarray,
    masked_observed: np.ndarray,
    absolute_raw_pearson: np.ndarray,
) -> dict[str, Any]:
    source_matrix = np.mean(source_unmasked, axis=(0, 1), dtype=np.float64)
    published_symmetric_matrix = 0.5 * (
        np.abs(source_matrix) + np.abs(source_matrix.T)
    )
    masked_matrix = np.mean(masked_observed, axis=(0, 1), dtype=np.float64)
    concordance = _safe_spearman(
        _offdiagonal_values(source_matrix),
        _offdiagonal_values(masked_matrix),
    )
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    target_positions = {
        gene: index for index, gene in enumerate(FROZEN_TARGET_GENES)
    }
    raw_matrix = np.mean(
        _finite_array(
            absolute_raw_pearson,
            label="source-reproduction absolute raw Pearson",
            shape=(CORE_COUNT, 5, 39),
        ),
        axis=0,
        dtype=np.float64,
    )
    masked_target_values: list[float] = []
    raw_target_values: list[float] = []
    for target in FROZEN_TARGET_GENES:
        keep = np.ones(39, dtype=bool)
        keep[positions[target]] = False
        masked_target_values.extend(
            np.abs(masked_matrix[positions[target], keep]).tolist()
        )
        raw_target_values.extend(
            raw_matrix[target_positions[target], keep].tolist()
        )
    raw_concordance = _safe_spearman(
        np.asarray(masked_target_values, dtype=np.float64),
        np.asarray(raw_target_values, dtype=np.float64),
    )
    pair_rows = []
    for target, source in FROZEN_LOCKED_DIRECTED_PAIRS:
        target_position = positions[target]
        source_position = positions[source]
        directed_row = np.abs(source_matrix[target_position]).copy()
        directed_row[target_position] = -np.inf
        directed_order = np.argsort(-directed_row, kind="stable")
        directed_rank = (
            int(np.flatnonzero(directed_order == source_position)[0]) + 1
        )
        published_row = published_symmetric_matrix[target_position].copy()
        published_row[target_position] = -np.inf
        published_order = np.argsort(-published_row, kind="stable")
        published_rank = (
            int(np.flatnonzero(published_order == source_position)[0]) + 1
        )
        pair_rows.append(
            {
                "target": target,
                "source": source,
                "source_style_signed_gradient": float(
                    source_matrix[target_position, source_position]
                ),
                "source_style_directed_absolute_rank_within_38": (
                    directed_rank
                ),
                "published_symmetric_absolute_gradient": float(
                    published_symmetric_matrix[
                        target_position, source_position
                    ]
                ),
                "published_symmetric_absolute_rank_within_38": (
                    published_rank
                ),
                "masked_signed_gradient": float(
                    masked_matrix[target_position, source_position]
                ),
                "absolute_raw_pearson": float(
                    raw_matrix[target_positions[target], source_position]
                ),
            }
        )
    return {
        "marker_gene_order": list(FROZEN_MARKER_GENES),
        "equal_core_seven_seed_signed_directed_matrix": (
            source_matrix.tolist()
        ),
        "equal_core_seven_seed_published_symmetric_absolute_matrix": (
            published_symmetric_matrix.tolist()
        ),
        "published_transform": "0.5 * (abs(J) + abs(J.T))",
        "adaptation_note": (
            "J is the equal-core mean of seven fixed-seed source-style "
            "unmasked summed-output gradients on the ten-core adjacent-normal "
            "cohort; it is not a replay of MyJJu's unavailable historical "
            "slide/checkpoint output."
        ),
        "source_style_vs_masked_offdiagonal_absolute_spearman": concordance,
        "masked_gradient_vs_raw_coexpression_absolute_spearman": (
            raw_concordance
        ),
        "locked_pair_rows": pair_rows,
        "role": (
            "descriptive_circular_source_procedure_and_coexpression_control_"
            "not_independent_validation"
        ),
    }


def _decomposition_summary(
    same_signed: np.ndarray,
    other_signed: np.ndarray,
    same_l1: np.ndarray,
    other_l1: np.ndarray,
) -> dict[str, Any]:
    ensemble_same = np.mean(same_signed, axis=0, dtype=np.float64)
    ensemble_other = np.mean(other_signed, axis=0, dtype=np.float64)
    ensemble_same_l1 = np.mean(same_l1, axis=0, dtype=np.float64)
    ensemble_other_l1 = np.mean(other_l1, axis=0, dtype=np.float64)
    positions = {gene: index for index, gene in enumerate(FROZEN_MARKER_GENES)}
    targets = {gene: index for index, gene in enumerate(FROZEN_TARGET_GENES)}
    rows = []
    for target, source in FROZEN_LOCKED_DIRECTED_PAIRS:
        target_position = targets[target]
        source_position = positions[source]
        same_value = float(
            np.mean(ensemble_same[:, target_position, source_position])
        )
        other_value = float(
            np.mean(ensemble_other[:, target_position, source_position])
        )
        same_mass = float(
            np.mean(ensemble_same_l1[:, target_position, source_position])
        )
        other_mass = float(
            np.mean(ensemble_other_l1[:, target_position, source_position])
        )
        total_mass = same_mass + other_mass
        rows.append(
            {
                "target": target,
                "source": source,
                "same_cell_signed": same_value,
                "other_cell_signed": other_value,
                "same_cell_l1": same_mass,
                "other_cell_l1": other_mass,
                "other_cell_l1_fraction": (
                    other_mass / total_mass if total_mass > 0 else None
                ),
            }
        )
    all_same = float(np.sum(ensemble_same_l1, dtype=np.float64))
    all_other = float(np.sum(ensemble_other_l1, dtype=np.float64))
    return {
        "locked_pair_rows": rows,
        "overall_other_cell_l1_fraction": (
            all_other / (all_same + all_other)
            if all_same + all_other > 0
            else None
        ),
        "exactness": (
            "exact_per_sampled_receiver_population_weighted_stratified_estimate"
        ),
    }


def _gate_row(
    name: str,
    *,
    scope: str,
    threshold: str,
    observed: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "gate": name,
        "scope": scope,
        "threshold": threshold,
        "observed": observed,
        "pass": bool(passed),
    }


def reduce_gradient_audit(
    seed_arrays: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    """Reduce exact seven-seed arrays into all frozen computational gates."""

    if len(seed_arrays) != len(SEEDS):
        raise GradientReductionError("exactly seven seed shards are required")
    truth = _identical(seed_arrays, "selected_truth")
    mask = _identical(seed_arrays, "selected_mask")
    offsets_raw = _identical(seed_arrays, "core_offsets")
    gene_mean = _identical(seed_arrays, "per_core_gene_mean")
    prevalence = _identical(seed_arrays, "source_nonzero_prevalence")
    expression = _identical(seed_arrays, "source_mean_expression")
    raw_pearson = _identical(seed_arrays, "abs_target_source_raw_pearson")
    source_population_sd = _identical(seed_arrays, "source_population_sd")
    source_q01 = _identical(seed_arrays, "source_q01")
    source_q99 = _identical(seed_arrays, "source_q99")
    node_count = int(truth.shape[0])
    truth = _finite_array(
        truth, label="selected truth", shape=(node_count, 5)
    )
    mask = _bool_array(
        mask, label="selected mask", shape=(node_count, 3, 5)
    )
    offsets = np.asarray(offsets_raw)
    if (
        offsets.dtype.kind not in {"i", "u"}
        or offsets.shape != (CORE_COUNT + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != node_count
        or np.any(np.diff(offsets) <= 0)
    ):
        raise GradientReductionError("core offsets are invalid")
    gene_mean = _finite_array(
        gene_mean, label="per-core gene mean", shape=(CORE_COUNT, 5)
    )
    source_population_sd = _finite_array(
        source_population_sd,
        label="source population SD",
        shape=(CORE_COUNT, 39),
    )
    source_q01 = _finite_array(
        source_q01, label="source q01", shape=(CORE_COUNT, 39)
    )
    source_q99 = _finite_array(
        source_q99, label="source q99", shape=(CORE_COUNT, 39)
    )
    if np.any(source_population_sd <= 0.0) or np.any(source_q01 >= source_q99):
        raise GradientReductionError(
            "full-core perturbation SD or quantile bounds are invalid"
        )

    observed_prediction = np.mean(
        np.stack(
            [
                _finite_array(
                    arrays["selected_prediction_observed"],
                    label="observed prediction",
                    shape=(node_count, 3, 5),
                )
                for arrays in seed_arrays
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.float64,
    )
    permuted_prediction = np.mean(
        np.stack(
            [
                _finite_array(
                    arrays["selected_prediction_permuted"],
                    label="permuted prediction",
                    shape=(node_count, 3, 5),
                )
                for arrays in seed_arrays
            ],
            axis=0,
        ),
        axis=0,
        dtype=np.float64,
    )
    masked = np.stack(
        [
            _finite_array(
                arrays["masked_rep0_observed_signed"],
                label="observed masked gradient",
                shape=(CORE_COUNT, 39, 39),
            )
            for arrays in seed_arrays
        ],
        axis=0,
    )
    permuted = np.stack(
        [
            _finite_array(
                arrays["masked_rep0_permuted_signed"],
                label="permuted masked gradient",
                shape=(CORE_COUNT, 39, 39),
            )
            for arrays in seed_arrays
        ],
        axis=0,
    )
    masked_locked = np.stack(
        [
            _finite_array(
                arrays["masked_locked_observed_signed"],
                label="masked locked rows",
                shape=(CORE_COUNT, 3, 5, 39),
            )
            for arrays in seed_arrays
        ],
        axis=0,
    )
    source_unmasked = np.stack(
        [
            _finite_array(
                arrays["source_unmasked_signed"],
                label="source-style gradient",
                shape=(CORE_COUNT, 39, 39),
            )
            for arrays in seed_arrays
        ],
        axis=0,
    )
    faithfulness_predicted = np.stack(
        [
            _finite_array(
                arrays["faithfulness_predicted"],
                label="faithfulness prediction",
                shape=(CORE_COUNT, 2, 5, 39),
            )
            for arrays in seed_arrays
        ],
        axis=0,
    )
    faithfulness_actual = np.stack(
        [
            _finite_array(
                arrays["faithfulness_actual"],
                label="faithfulness actual",
                shape=(CORE_COUNT, 2, 5, 39),
            )
            for arrays in seed_arrays
        ],
        axis=0,
    )
    randomized = np.stack(
        [
            _finite_array(
                arrays["randomized_rep0_observed_signed"],
                label="randomized gradient",
                shape=(39, 39),
            )
            for arrays in seed_arrays
        ],
        axis=0,
    )

    eligibility = _target_eligibility(
        truth=truth,
        mask=mask,
        observed_prediction=observed_prediction,
        permuted_prediction=permuted_prediction,
        offsets=offsets,
        gene_mean=gene_mean,
    )
    seed_stability = _seed_rank_stability(masked)
    mask_stability = _mask_rank_stability(masked_locked)
    sign_stability = _signed_pair_stability(masked)
    faithfulness = _faithfulness(
        faithfulness_predicted, faithfulness_actual
    )
    ensemble_observed = np.mean(masked, axis=0, dtype=np.float64)
    ensemble_permuted = np.mean(permuted, axis=0, dtype=np.float64)
    graph_null = _graph_structure_null(
        ensemble_observed, ensemble_permuted
    )
    randomization = _parameter_randomization(
        ensemble_observed[0], randomized
    )
    matched_null_result = matched_pair_null(
        ensemble_observed=ensemble_observed,
        source_nonzero_prevalence=prevalence,
        source_mean_expression=expression,
        absolute_raw_pearson=raw_pearson,
    )
    source_reproduction = _source_style_reproduction(
        source_unmasked, masked, raw_pearson
    )
    decomposition = _decomposition_summary(
        np.stack(
            [
                _finite_array(
                    arrays["decomposition_same_signed"],
                    label="same signed decomposition",
                    shape=(CORE_COUNT, 5, 39),
                )
                for arrays in seed_arrays
            ]
        ),
        np.stack(
            [
                _finite_array(
                    arrays["decomposition_other_signed"],
                    label="other signed decomposition",
                    shape=(CORE_COUNT, 5, 39),
                )
                for arrays in seed_arrays
            ]
        ),
        np.stack(
            [
                _finite_array(
                    arrays["decomposition_same_l1"],
                    label="same L1 decomposition",
                    shape=(CORE_COUNT, 5, 39),
                )
                for arrays in seed_arrays
            ]
        ),
        np.stack(
            [
                _finite_array(
                    arrays["decomposition_other_l1"],
                    label="other L1 decomposition",
                    shape=(CORE_COUNT, 5, 39),
                )
                for arrays in seed_arrays
            ]
        ),
    )

    mask_gate_by_target = {
        row["target"]: bool(row["pass"])
        for row in mask_stability["target_rows"]
    }
    sign_gate_by_target = {
        row["target"]: bool(row["any_pair_pass"])
        for row in sign_stability["target_rows"]
    }
    eligible_targets = [
        str(row["target"])
        for row in eligibility["target_rows"]
        if bool(row["eligible"])
    ]
    eligible_targets_with_stable_pair = []
    for row in eligibility["target_rows"]:
        target = str(row["target"])
        if (
            row["eligible"]
            and mask_gate_by_target[target]
            and sign_gate_by_target[target]
        ):
            eligible_targets_with_stable_pair.append(target)
    candidate_set_qualified = bool(
        eligible_targets_with_stable_pair
        and seed_stability["pass"]
        and faithfulness["pass"]
        and graph_null["pass"]
        and randomization["pass"]
        and matched_null_result["pass"]
    )

    gate_rows: list[dict[str, Any]] = []
    for row in eligibility["target_rows"]:
        target = str(row["target"])
        gate_rows.append(
            _gate_row(
                "target_predictive_eligibility",
                scope=target,
                threshold=">=2% vs gene mean and favors >=8/10 cores",
                observed={
                    "relative_improvement": row[
                        "gene_mean_relative_improvement"
                    ],
                    "favoring_cores": row["gene_mean_favoring_core_count"],
                },
                passed=bool(row["predictive_gate_pass"]),
            )
        )
        gate_rows.append(
            _gate_row(
                "target_graph_use_eligibility",
                scope=target,
                threshold=">=2% vs permuted graph and favors >=8/10 cores",
                observed={
                    "relative_improvement": row["graph_relative_improvement"],
                    "favoring_cores": row["graph_favoring_core_count"],
                },
                passed=bool(row["graph_use_gate_pass"]),
            )
        )
    gate_rows.append(
        _gate_row(
            "seed_rank_stability",
            scope="all_cores",
            threshold="median pairwise Spearman >=0.70 in >=8/10 cores",
            observed={
                "passing_cores": seed_stability["passing_core_count"]
            },
            passed=bool(seed_stability["pass"]),
        )
    )
    for row in mask_stability["target_rows"]:
        gate_rows.append(
            _gate_row(
                "mask_rank_stability",
                scope=str(row["target"]),
                threshold="median pairwise Spearman >=0.70 in >=8/10 cores",
                observed={"passing_cores": row["passing_core_count"]},
                passed=bool(row["pass"]),
            )
        )
    for row in sign_stability["pair_rows"]:
        gate_rows.append(
            _gate_row(
                "signed_pair_stability",
                scope=f"{row['target']}<-{row['source']}",
                threshold="one sign in >=6/7 seeds and >=8/10 cores",
                observed={
                    "direction": row["direction"],
                    "seed_count": row["seed_direction_count"],
                    "core_count": row["core_direction_count"],
                    "positive_seed_count": row["positive_seed_count"],
                    "negative_seed_count": row["negative_seed_count"],
                    "zero_seed_count": row["zero_seed_count"],
                    "positive_core_count": row["positive_core_count"],
                    "negative_core_count": row["negative_core_count"],
                    "zero_core_count": row["zero_core_count"],
                },
                passed=bool(row["pass"]),
            )
        )
    for row in faithfulness["primary_offdiagonal_rows"]:
        gate_rows.append(
            _gate_row(
                "bounded_faithfulness",
                scope=f"scale_{row['scale']:.2f}_sd_offdiagonal",
                threshold=(
                    "testable>=90%, Spearman>=0.70, sign>=80%, "
                    "median-error-ratio<=0.50"
                ),
                observed=row,
                passed=bool(row["pass"]),
            )
        )
    gate_rows.extend(
        (
            _gate_row(
                "graph_gradient_structure_null",
                scope="aggregate",
                threshold=">=25% improvement in >=8/10 cores",
                observed={"passing_cores": graph_null["passing_core_count"]},
                passed=bool(graph_null["pass"]),
            ),
            _gate_row(
                "parameter_randomization",
                scope="aggregate",
                threshold="trained ensemble exceeds all seven randomized models",
                observed={
                    "trained_S": randomization["trained"]["S"],
                    "maximum_randomized_S": max(
                        row["S"] for row in randomization["controls"]
                    ),
                },
                passed=bool(randomization["pass"]),
            ),
            _gate_row(
                "matched_pair_null",
                scope="aggregate",
                threshold="empirical upper-tail p<=0.05, 10,000 draws",
                observed={
                    "p": matched_null_result["upper_tail_p"],
                    "observed": matched_null_result["observed_statistic"],
                    "null_95th": matched_null_result[
                        "null_95th_percentile"
                    ],
                },
                passed=bool(matched_null_result["pass"]),
            ),
        )
    )
    return {
        "eligibility": eligibility,
        "seed_rank_stability": seed_stability,
        "mask_rank_stability": mask_stability,
        "signed_pair_stability": sign_stability,
        "bounded_faithfulness": faithfulness,
        "graph_gradient_null": graph_null,
        "parameter_randomization": randomization,
        "matched_pair_null": matched_null_result,
        "source_style_reproduction": source_reproduction,
        "sampled_receiver_decomposition": decomposition,
        "gate_rows": gate_rows,
        "eligible_targets": eligible_targets,
        "eligible_targets_with_stable_pair": (
            eligible_targets_with_stable_pair
        ),
        "candidate_set_computational_precursors_supported": (
            candidate_set_qualified
        ),
        "mechanism_validation_available": False,
        "mechanism_claim_supported": False,
        "maximum_defensible_claim": (
            "candidate_set_contains_stable_faithful_graph_dependent_"
            "null_calibrated_model_implied_predictive_sensitivities"
            if candidate_set_qualified
            else "no_claim_beyond_reported_model_behavior"
        ),
    }


__all__ = [
    "GradientReductionError",
    "matched_pair_null",
    "mean_huber",
    "reduce_gradient_audit",
    "relative_improvement",
    "structure_statistic",
]
