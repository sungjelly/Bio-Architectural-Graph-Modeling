"""Dependency-light tests for the broad spatial-field leakage contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.spatial_field import (  # noqa: E402
    BROAD_SPATIAL_BASIS_NAME,
    BROAD_SPATIAL_FEATURE_NAMES,
    BroadSpatialFieldBasis,
)


def test_global_quadratic_basis_has_exact_locked_terms_and_provenance() -> None:
    training = np.asarray(
        [[0.0, 10.0], [2.0, 10.0], [4.0, 14.0], [6.0, 14.0]]
    )
    basis = BroadSpatialFieldBasis.fit(training)
    query = np.asarray([[5.0, 13.0]])
    standardized = (query[0] - basis.center_um) / basis.scale_um
    x, y = standardized
    np.testing.assert_allclose(
        basis.transform(query)[0],
        np.asarray([x, y, x * x, x * y, y * y]),
        rtol=1e-6,
        atol=1e-7,
    )

    provenance = basis.provenance()
    assert provenance["control_name"] == "broad_spatial_field"
    assert provenance["basis"] == BROAD_SPATIAL_BASIS_NAME
    assert provenance["feature_names"] == list(BROAD_SPATIAL_FEATURE_NAMES)
    assert provenance["maximum_polynomial_degree"] == 2
    assert provenance["fit_scope"] == "training split coordinates only"
    assert provenance["held_out_refit"] is False
    assert provenance["contains_cell_or_region_ids"] is False
    assert provenance["contains_periodic_or_fourier_features"] is False
    assert provenance["contains_knots_or_spatial_lookup_embeddings"] is False
    assert provenance["contains_graph_or_neighbor_features"] is False
    assert len(provenance["basis_fit_checksum"]) == 64


def test_basis_fit_is_train_only_and_held_out_coordinates_cannot_change_it() -> None:
    training = np.asarray(
        [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0], [6.0, 12.0]]
    )
    validation = np.asarray([[100.0, -50.0], [200.0, -100.0]])
    basis = BroadSpatialFieldBasis.fit(training)
    original_provenance = basis.provenance()
    original_training_features = basis.transform(training)

    altered_validation = validation * 1_000_000.0
    basis.transform(altered_validation)
    assert basis.provenance() == original_provenance
    np.testing.assert_array_equal(
        basis.transform(training),
        original_training_features,
    )
    np.testing.assert_allclose(basis.center_um, training.mean(axis=0))
    np.testing.assert_allclose(basis.scale_um, training.std(axis=0, ddof=0))

    translated = training + np.asarray([500.0, -300.0])
    translated_basis = BroadSpatialFieldBasis.fit(translated)
    np.testing.assert_allclose(
        translated_basis.transform(translated),
        original_training_features,
        rtol=1e-6,
        atol=1e-6,
    )


def test_constant_axis_is_zero_and_invalid_coordinates_are_rejected() -> None:
    training = np.asarray([[2.0, 1.0], [2.0, 3.0], [2.0, 5.0]])
    basis = BroadSpatialFieldBasis.fit(training)
    assert basis.constant_axes == (True, False)
    assert basis.scale_um[0] == 1.0
    features = basis.transform(training)
    assert not features[:, [0, 2, 3]].any()
    held_out_features = basis.transform(np.asarray([[102.0, 3.0]]))
    assert not held_out_features[:, [0, 2, 3]].any()

    with pytest.raises(ValueError, match="shape"):
        BroadSpatialFieldBasis.fit(np.ones((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        basis.transform(np.asarray([[np.nan, 0.0]]))
    with pytest.raises(ValueError, match="at least one"):
        BroadSpatialFieldBasis.fit(np.empty((0, 2)))
