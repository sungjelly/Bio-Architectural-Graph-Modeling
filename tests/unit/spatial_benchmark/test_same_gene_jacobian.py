from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark.same_gene_jacobian import (
    FOLD_COMPONENT_ORDINALS,
    SameGeneJacobianError,
    build_neighbor_aggregates,
    component_equal_weights,
    deterministic_derangement,
    diagonal_summary,
    geometry_components,
    planted_recovery_control,
    solve_weighted_ridge_numpy,
)


def test_geometry_components_use_minimum_fov_ordinals_and_frozen_folds() -> None:
    fovs = np.arange(1, 14, dtype=np.int64)
    origins = np.column_stack((np.arange(13) * 2.0, np.zeros(13)))
    result = geometry_components("SO_1", fovs, origins)

    assert result.component_count == 13
    assert np.array_equal(result.component_ordinals, fovs)
    expected = {
        ordinal: fold
        for fold, mapping in FOLD_COMPONENT_ORDINALS.items()
        for ordinal in mapping["SO_1"]
    }
    assert result.folds.tolist() == [expected[value] for value in fovs]
    assert result.minimum_between_component_distance_mm == pytest.approx(2.0)


def test_geometry_components_fail_closed_if_snapshot_partition_changes() -> None:
    with pytest.raises(SameGeneJacobianError, match="geometry components changed"):
        geometry_components(
            "SO_1",
            np.arange(1, 13),
            np.column_stack((np.arange(12) * 2.0, np.zeros(12))),
        )


def test_derangement_is_reproducible_and_has_no_fixed_points() -> None:
    first = deterministic_derangement(101, seed=7)
    second = deterministic_derangement(101, seed=7)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.arange(101))
    assert not np.any(first == np.arange(101))


def test_neighbor_aggregates_are_cross_cell_banded_and_finite() -> None:
    # Four close pairs.  Pair partners are within 10 um; adjacent pairs are
    # separated by 30 um and therefore fall in the 25--50 um annulus.
    coordinates = np.asarray(
        [[0, 0], [10, 0], [40, 0], [50, 0], [80, 0], [90, 0], [120, 0], [130, 0]],
        dtype=np.float64,
    )
    expression = np.column_stack(
        (np.arange(1, 9), np.arange(11, 19), np.arange(21, 29))
    ).astype(np.float32)
    result = build_neighbor_aggregates(
        expression,
        coordinates,
        np.ones(8, dtype=np.int64),
        slide="SO_1",
        k=2,
        minimum_matched_degree=1,
        permutation_seed=9,
    )

    assert np.all(result.near_degree == 1)
    assert np.array_equal(result.near_mean[0], expression[1])
    assert np.array_equal(result.near_mean[1], expression[0])
    assert result.audit["permutation_source_mapping_changed_fraction"] == 1.0
    assert np.isfinite(result.permuted_near_mean).all()
    assert not np.any(np.all(result.permuted_near_mean == expression, axis=1))


def test_component_equal_weights_do_not_treat_cells_as_equal_groups() -> None:
    groups = np.asarray([1, 1, 1, 2])
    selected = np.ones(4, dtype=bool)
    weights = component_equal_weights(groups, selected)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[:3].sum() == pytest.approx(0.5)
    assert weights[3] == pytest.approx(0.5)


def test_weighted_ridge_and_diagonal_summary_recover_known_map() -> None:
    rng = np.random.default_rng(4)
    design = np.column_stack((np.ones(2000), rng.normal(size=(2000, 4))))
    target_by_source = np.diag([0.2, -0.4, 0.6, 0.8])
    targets = design[:, 1:] @ target_by_source.T
    coefficients = solve_weighted_ridge_numpy(
        design,
        targets,
        np.full(2000, 1 / 2000),
        penalty=1e-10,
    )
    recovered = coefficients[1:].T
    summary = diagonal_summary(recovered)

    assert np.max(np.abs(recovered - target_by_source)) < 1e-7
    assert summary.row_top1_fraction == 1.0
    assert summary.diagonal_offdiagonal_ratio > 1e9


def test_planted_recovery_and_finite_difference_control_pass() -> None:
    control = planted_recovery_control()
    assert control["passed"] is True
    assert control["maximum_planted_coefficient_error"] <= 1e-6
    assert control["maximum_finite_difference_error"] <= 1e-8
