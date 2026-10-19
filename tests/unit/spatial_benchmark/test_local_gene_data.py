from __future__ import annotations

import numpy as np
import pytest

import spatial_benchmark.local_gene_data as local_data
from spatial_benchmark.local_gene_data import (
    LocalGeneCore,
    MASK_BASE_SEED,
    TRAINING_MASK_NAMESPACE,
    VALIDATION_MASK_NAMESPACE,
    local_gene_context,
    make_mask,
    training_gene_initialization,
)
from spatial_benchmark.target_cell_masking import (
    make_training_target_cell_masks,
    make_validation_target_cell_masks,
)


def _core(counts, *, alias="SO2-C15", role="train"):
    counts = np.asarray(counts, dtype=np.int32)
    return LocalGeneCore(
        alias=alias, role=role, counts=counts,
        expression=np.log1p(counts.astype(np.float32)),
        covariates=np.zeros((len(counts), 22), dtype=np.float32),
        local_log_mean=np.zeros(counts.shape, dtype=np.float32),
        local_positive_fraction=np.zeros(counts.shape, dtype=np.float32),
        density=np.zeros((len(counts), 1), dtype=np.float32),
    )


def test_local_context_is_log_of_raw_mean_and_has_physical_density():
    counts = np.array([[100, 100], [2, 0], [0, 6], [20, 30]], dtype=np.int32)
    coordinates = np.array([[0, 0], [1, 0], [2, 0], [100, 0]], dtype=np.float64)
    means, fractions, density = local_gene_context(counts, coordinates, neighbors=2, radius_um=3)
    np.testing.assert_allclose(means[0], np.log1p([1, 3]), rtol=1e-7)
    np.testing.assert_array_equal(fractions[0], [0.5, 0.5])
    np.testing.assert_array_equal(density[:, 0], [2, 2, 2, 0])
    np.testing.assert_array_equal(means[3], [0, 0])
    np.testing.assert_array_equal(fractions[3], [0, 0])
    for array in (means, fractions, density):
        assert array.dtype == np.float32
        assert np.isfinite(array).all()


def test_hidden_target_expression_cannot_change_its_context():
    counts = np.array([[1, 0], [2, 3], [0, 5]], dtype=np.int32)
    coordinates = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float64)
    original = local_gene_context(counts, coordinates)
    changed_counts = counts.copy()
    changed_counts[1] = [2_000_000_000, 999]
    changed = local_gene_context(changed_counts, coordinates)
    for before, after in zip(original, changed):
        np.testing.assert_array_equal(before[1], after[1])
    assert not np.array_equal(original[0][0], changed[0][0])
    np.testing.assert_array_equal(counts, [[1, 0], [2, 3], [0, 5]])


def test_density_counts_more_neighbors_than_the_pooled_limit():
    counts = np.array([[100], [2], [4], [80]], dtype=np.int32)
    coordinates = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float64)
    means, fractions, density = local_gene_context(counts, coordinates, neighbors=2, radius_um=3)
    np.testing.assert_allclose(means[0], np.log1p([3]), rtol=1e-7)
    np.testing.assert_array_equal(fractions[0], [1])
    assert density[0, 0] == 3


def test_radius_is_inclusive_and_coordinates_are_physical_micrometers():
    counts = np.array([[1], [3], [9]], dtype=np.int32)
    coordinates = np.array([[0, 0], [75, 0], [75.001, 0]])
    means, fractions, density = local_gene_context(counts, coordinates)
    np.testing.assert_allclose(means[0], np.log1p([3]), rtol=1e-7)
    assert fractions[0, 0] == 1
    assert density[0, 0] == 1


def test_single_node_has_zero_finite_context():
    means, fractions, density = local_gene_context(
        np.array([[2, 8]], dtype=np.int32), np.zeros((1, 2)),
    )
    for array in (means, fractions, density):
        assert np.count_nonzero(array) == 0
        assert np.isfinite(array).all()


def test_coincident_coordinates_still_exclude_self_and_cap_sources():
    counts = np.arange(1, 26, dtype=np.int32)[:, None]
    coordinates = np.zeros((25, 2))
    before = local_gene_context(counts, coordinates, neighbors=4)
    for target in range(len(counts)):
        changed = counts.copy()
        changed[target] = 100_000
        after = local_gene_context(changed, coordinates, neighbors=4)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a[target], b[target])
    np.testing.assert_array_equal(before[2][:, 0], np.full(25, 24))


def test_chunk_size_does_not_change_context():
    rng = np.random.default_rng(812)
    counts = rng.integers(0, 5, (29, 7), dtype=np.int32)
    coordinates = rng.normal(size=(29, 2))
    first = local_gene_context(counts, coordinates, row_chunk_size=3)
    second = local_gene_context(counts, coordinates, row_chunk_size=29)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_training_initialization_is_cell_weighted_and_excludes_validation():
    small = _core([[4, 0]], alias="SO2-C15")
    large = _core([[2, 0], [0, 0], [0, 0]], alias="SO2-C16")
    prevalence, mean = training_gene_initialization([small, large])
    np.testing.assert_array_equal(prevalence, [0.5, 0])
    np.testing.assert_array_equal(mean, [3, 1])
    with pytest.raises(ValueError, match="training cores only"):
        training_gene_initialization([small, _core([[900, 5]], role="validation")])


def test_masks_exactly_match_inherited_sampling_and_namespaces():
    assert MASK_BASE_SEED == 2_026_092_601
    assert TRAINING_MASK_NAMESPACE == "bagm.so2.target_isolated_nb.training_masks.v1"
    assert VALIDATION_MASK_NAMESPACE == "bagm.so2.target_isolated_nb.fixed_validation_masks.v1"
    train = _core(np.zeros((50, 1000), dtype=np.int32))
    observed = make_mask(train, epoch=7)
    expected = make_training_target_cell_masks(
        50, 1000, base_seed=MASK_BASE_SEED, namespace=TRAINING_MASK_NAMESPACE,
        core_alias="SO2-C15", global_epoch=7,
    )
    np.testing.assert_array_equal(observed.mask, expected.mask)
    assert observed.to_receipt() == expected.to_receipt()
    assert observed.to_receipt()["covers_all_cells"]
    assert np.all(observed.masked_gene_counts >= 1)
    assert not np.array_equal(observed.mask, make_mask(train, epoch=8).mask)
    validation = _core(np.zeros((50, 1000), dtype=np.int32), alias="SO2-C27", role="validation")
    fixed = make_mask(validation)
    expected_fixed = make_validation_target_cell_masks(
        50, 1000, namespace=VALIDATION_MASK_NAMESPACE, core_alias="SO2-C27",
    )
    assert fixed.to_receipt() == expected_fixed.to_receipt()
    assert fixed.mask_sha256 == make_mask(validation).mask_sha256
    with pytest.raises(ValueError, match="global epoch"):
        make_mask(train)
    with pytest.raises(ValueError, match="epoch must be None"):
        make_mask(validation, epoch=7)


def test_core_inputs_are_read_only_and_not_aliased_to_count_targets():
    core = _core([[0, 3], [5, 0]])
    assert core.n_cells == 2 and core.n_genes == 2
    assert not np.shares_memory(core.counts, core.expression)
    with pytest.raises(ValueError, match="read-only"):
        core.counts[0, 0] = 50
    with pytest.raises(ValueError, match="read-only"):
        core.expression[0, 0] = 50


@pytest.mark.parametrize("kwargs", [{"neighbors": 0}, {"neighbors": True}, {"radius_um": 0}, {"radius_um": np.nan}, {"row_chunk_size": 0}])
def test_invalid_neighborhood_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        local_gene_context(np.zeros((2, 3), dtype=np.int32), np.zeros((2, 2)), **kwargs)


def test_changed_overlay_is_rejected_before_any_source_arrays_are_read(tmp_path):
    root = tmp_path / "overlay"
    root.mkdir(mode=0o700)
    manifest = root / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    manifest.chmod(0o600)
    with pytest.raises(ValueError, match="frozen baseline artifact"):
        local_data._verify_overlay(root)
