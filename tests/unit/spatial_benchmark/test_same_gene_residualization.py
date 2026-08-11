from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark.same_gene_residualization import (
    SameGeneResidualizationError,
    apply_neighbor_mean_residual,
    apply_target_residual,
    component_equal_train_weights,
    fit_cell_type_library_wls,
    fit_component_equal_library_wls,
)


def _component_layout() -> tuple[np.ndarray, np.ndarray]:
    components = np.repeat(np.arange(4), [37, 61, 43, 29])
    train_mask = components != 3
    return components, train_mask


def test_library_wls_recovers_planted_coefficients_and_neighbor_means() -> None:
    rng = np.random.default_rng(20260810)
    components, train_mask = _component_layout()
    rows = len(components)
    genes = 7
    log_library = rng.normal(loc=8.0, scale=0.7, size=rows)
    intercept = rng.normal(scale=0.4, size=genes)
    slope = rng.normal(scale=0.2, size=genes)
    expression = intercept + log_library[:, None] * slope

    fit = fit_component_equal_library_wls(
        expression,
        log_library,
        components,
        train_mask=train_mask,
        row_chunk_size=11,
        gene_chunk_size=3,
    )

    assert np.allclose(fit.intercept, intercept, rtol=0, atol=2e-13)
    assert np.allclose(fit.library_slope, slope, rtol=0, atol=3e-14)
    target_residual = apply_target_residual(
        fit,
        expression,
        log_library,
        row_chunk_size=13,
        gene_chunk_size=2,
        output_dtype=np.float64,
    )
    assert np.max(np.abs(target_residual)) < 3e-13

    neighbor_mean_library = rng.normal(loc=8.2, scale=0.3, size=19)
    neighbor_mean_expression = (
        intercept + neighbor_mean_library[:, None] * slope
    )
    neighbor_residual = apply_neighbor_mean_residual(
        fit,
        neighbor_mean_expression,
        neighbor_mean_library,
        row_chunk_size=5,
        gene_chunk_size=3,
        output_dtype=np.float64,
    )
    assert np.max(np.abs(neighbor_residual)) < 3e-13


def test_heldout_outcome_mutation_cannot_change_library_coefficients() -> None:
    rng = np.random.default_rng(81)
    components, train_mask = _component_layout()
    log_library = rng.normal(size=len(components))
    expression = rng.normal(size=(len(components), 9))

    first = fit_component_equal_library_wls(
        expression,
        log_library,
        components,
        train_mask=train_mask,
        row_chunk_size=17,
        gene_chunk_size=4,
    )
    changed = expression.copy()
    changed[~train_mask] = rng.normal(
        loc=1e7, scale=1e5, size=changed[~train_mask].shape
    )
    replay = fit_component_equal_library_wls(
        changed,
        log_library,
        components,
        train_mask=train_mask,
        row_chunk_size=17,
        gene_chunk_size=4,
    )

    assert np.array_equal(first.intercept, replay.intercept)
    assert np.array_equal(first.library_slope, replay.library_slope)


def test_cell_type_wls_recovers_frozen_intercepts_shared_slope_and_unseen_rule() -> None:
    rng = np.random.default_rng(19)
    components, train_mask = _component_layout()
    levels = ("A", "B", "C")
    cell_types = np.asarray(
        [levels[index % len(levels)] for index in range(len(components))],
        dtype=str,
    )
    heldout_positions = np.flatnonzero(~train_mask)
    cell_types[heldout_positions[::2]] = "D"
    weights = component_equal_train_weights(components, train_mask)
    type_mass = np.asarray(
        [weights[train_mask & (cell_types == level)].sum() for level in levels]
    )
    genes = 6
    planted_intercepts = rng.normal(scale=0.3, size=(len(levels), genes))
    planted_slope = rng.normal(scale=0.15, size=genes)
    planted_global = type_mass @ planted_intercepts
    log_library = rng.normal(loc=7.5, scale=0.9, size=len(components))
    mapping = {level: index for index, level in enumerate(levels)}
    expression = np.empty((len(components), genes), dtype=np.float64)
    for row, label in enumerate(cell_types):
        base = (
            planted_intercepts[mapping[label]]
            if label in mapping
            else planted_global
        )
        expression[row] = base + planted_slope * log_library[row]

    fit = fit_cell_type_library_wls(
        expression,
        log_library,
        cell_types,
        components,
        levels=levels,
        train_mask=train_mask,
        row_chunk_size=9,
        gene_chunk_size=2,
    )

    assert fit.levels == levels
    assert np.allclose(fit.type_intercepts, planted_intercepts, atol=4e-13)
    assert np.allclose(fit.library_slope, planted_slope, atol=5e-14)
    assert np.allclose(fit.global_intercept, planted_global, atol=4e-13)
    residual = apply_target_residual(
        fit,
        expression,
        log_library,
        cell_types=cell_types,
        row_chunk_size=7,
        gene_chunk_size=2,
        output_dtype=np.float64,
    )
    assert np.max(np.abs(residual)) < 5e-13

    proportions = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.2, 0.3, 0.5],
            [0.2, 0.3, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    neighbor_library = np.asarray([7.0, 8.0, 8.5, 6.5])
    mass = proportions.sum(axis=1)
    neighbor_expression = (
        proportions @ planted_intercepts
        + (1.0 - mass)[:, None] * planted_global
        + neighbor_library[:, None] * planted_slope
    )
    frozen_copy = proportions.copy()
    neighbor_residual = apply_neighbor_mean_residual(
        fit,
        neighbor_expression,
        neighbor_library,
        type_proportions=proportions,
        type_proportion_levels=levels,
        row_chunk_size=2,
        gene_chunk_size=1,
        output_dtype=np.float64,
    )
    assert np.max(np.abs(neighbor_residual)) < 5e-13
    assert np.array_equal(proportions, frozen_copy)


def test_heldout_outcome_mutation_cannot_change_cell_type_coefficients() -> None:
    rng = np.random.default_rng(57)
    components, train_mask = _component_layout()
    levels = ("epithelial", "immune")
    cell_types = np.asarray(
        [levels[index % 2] for index in range(len(components))], dtype=str
    )
    cell_types[~train_mask] = "heldout_unseen"
    log_library = rng.normal(size=len(components))
    expression = rng.normal(size=(len(components), 5))

    first = fit_cell_type_library_wls(
        expression,
        log_library,
        cell_types,
        components,
        levels=levels,
        train_mask=train_mask,
        row_chunk_size=13,
        gene_chunk_size=2,
    )
    changed = expression.copy()
    changed[~train_mask] += 1e9
    replay = fit_cell_type_library_wls(
        changed,
        log_library,
        cell_types,
        components,
        levels=levels,
        train_mask=train_mask,
        row_chunk_size=13,
        gene_chunk_size=2,
    )

    assert np.array_equal(first.type_intercepts, replay.type_intercepts)
    assert np.array_equal(first.library_slope, replay.library_slope)
    assert np.array_equal(first.global_intercept, replay.global_intercept)


def test_component_equal_weights_are_train_only_and_cover_each_component() -> None:
    components = np.asarray([1, 1, 1, 2, 2, 3])
    train_mask = np.asarray([True, True, True, True, True, False])
    weights = component_equal_train_weights(components, train_mask)

    assert weights[~train_mask].sum() == 0
    assert weights[components == 1].sum() == pytest.approx(0.5)
    assert weights[components == 2].sum() == pytest.approx(0.5)
    assert np.all(weights[train_mask] > 0)

    leaking = weights.copy()
    leaking.setflags(write=True)
    leaking[-1] = 0.1
    with pytest.raises(SameGeneResidualizationError, match="outside train_mask"):
        component_equal_train_weights(
            components, train_mask, weights=leaking
        )

    uncovered = weights.copy()
    uncovered.setflags(write=True)
    uncovered[components == 2] = 0
    with pytest.raises(SameGeneResidualizationError, match="positive weight"):
        component_equal_train_weights(
            components, train_mask, weights=uncovered
        )


def test_library_fit_rejects_nonboolean_mask_nonfinite_input_and_rank_failure() -> None:
    components = np.asarray([0, 0, 1, 1])
    library = np.asarray([2.0, 2.0, 2.0, 3.0])
    expression = np.arange(12, dtype=np.float64).reshape(4, 3)

    with pytest.raises(SameGeneResidualizationError, match="explicit boolean"):
        fit_component_equal_library_wls(
            expression,
            library,
            components,
            train_mask=np.ones(4, dtype=np.int8),
        )

    nonfinite = expression.copy()
    nonfinite[-1, 0] = np.nan
    with pytest.raises(SameGeneResidualizationError, match="nonfinite"):
        fit_component_equal_library_wls(
            nonfinite,
            library,
            components,
            train_mask=np.asarray([True, True, False, False]),
        )

    with pytest.raises(SameGeneResidualizationError, match="rank deficient"):
        fit_component_equal_library_wls(
            expression,
            library,
            components,
            train_mask=np.asarray([True, True, True, False]),
        )


def test_cell_type_fit_requires_frozen_training_coverage_and_full_rank() -> None:
    components = np.asarray([0, 0, 1, 1, 2, 2])
    expression = np.arange(18, dtype=np.float64).reshape(6, 3)
    train_mask = np.ones(6, dtype=bool)

    with pytest.raises(SameGeneResidualizationError, match="lack training coverage"):
        fit_cell_type_library_wls(
            expression,
            np.arange(6, dtype=np.float64),
            np.asarray(["A", "A", "B", "B", "A", "B"]),
            components,
            levels=("A", "B", "C"),
            train_mask=train_mask,
        )

    with pytest.raises(SameGeneResidualizationError, match="outside frozen"):
        fit_cell_type_library_wls(
            expression,
            np.arange(6, dtype=np.float64),
            np.asarray(["A", "A", "B", "B", "A", "unseen"]),
            components,
            levels=("A", "B"),
            train_mask=train_mask,
        )

    types = np.asarray(["A", "A", "B", "B", "A", "B"])
    library_collinear_with_type = np.asarray(
        [0.0 if value == "A" else 1.0 for value in types]
    )
    with pytest.raises(SameGeneResidualizationError, match="rank deficient"):
        fit_cell_type_library_wls(
            expression,
            library_collinear_with_type,
            types,
            components,
            levels=("A", "B"),
            train_mask=train_mask,
        )


def test_neighbor_type_proportions_are_shape_level_and_mass_checked() -> None:
    components = np.asarray([0, 0, 1, 1, 2, 2])
    train_mask = np.ones(6, dtype=bool)
    types = np.asarray(["A", "B", "A", "B", "A", "B"])
    library = np.asarray([0.0, 0.2, 0.7, 1.1, 1.5, 2.0])
    expression = np.column_stack((library, 2.0 * library + 1.0))
    fit = fit_cell_type_library_wls(
        expression,
        library,
        types,
        components,
        levels=("A", "B"),
        train_mask=train_mask,
    )
    neighbor_expression = np.ones((2, 2))
    neighbor_library = np.ones(2)

    with pytest.raises(SameGeneResidualizationError, match="do not match"):
        apply_neighbor_mean_residual(
            fit,
            neighbor_expression,
            neighbor_library,
            type_proportions=np.full((2, 2), 0.5),
            type_proportion_levels=("B", "A"),
        )
    with pytest.raises(SameGeneResidualizationError, match="nonnegative"):
        apply_neighbor_mean_residual(
            fit,
            neighbor_expression,
            neighbor_library,
            type_proportions=np.asarray([[1.0, 0.0], [-0.1, 0.5]]),
            type_proportion_levels=("A", "B"),
        )
    with pytest.raises(SameGeneResidualizationError, match="exceed one"):
        apply_neighbor_mean_residual(
            fit,
            neighbor_expression,
            neighbor_library,
            type_proportions=np.asarray([[0.8, 0.3], [0.5, 0.5]]),
            type_proportion_levels=("A", "B"),
        )
