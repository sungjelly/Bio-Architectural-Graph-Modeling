"""Focused contract tests for the grouped adjacency ablation core."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spatial_benchmark.adjacency_ablation import (
    CORE_ALIASES,
    POSITION_PERMUTED_NULL_ARM,
    AdjacencyBundle,
    ExplicitSelfMeanGraphSAGE,
    FoldSplit,
    add_exact_self_adjacency,
    assert_valid_five_fold_splits,
    build_five_fold_splits,
    build_seeded_explicit_self_model,
    compute_true_neighbor_availability,
    derive_evaluation_mask_seed,
    derive_training_mask_seed,
    evaluate_fixed_mask_strata,
    explicit_self_incoming_mean,
    fit_equal_core_log1p_standardizer,
    inverse_standardized_log1p,
    materialize_fixed_adjacencies,
    ndarray_sha256,
    sample_uniform_mask_numpy,
    sample_uniform_mask_torch,
    state_dict_sha256,
    trainable_parameter_count,
    transform_log1p_counts,
    true_neighbor_observed_fraction,
)


def test_frozen_folds_are_group_disjoint_and_test_every_alias_once() -> None:
    folds = build_five_fold_splits()
    assert [fold.test_aliases for fold in folds] == [
        ("ANC-01", "ANC-06"),
        ("ANC-02", "ANC-07"),
        ("ANC-03", "ANC-08"),
        ("ANC-04", "ANC-09"),
        ("ANC-05", "ANC-10"),
    ]
    assert [fold.validation_aliases for fold in folds] == [
        ("ANC-02",),
        ("ANC-08",),
        ("ANC-04",),
        ("ANC-10",),
        ("ANC-01",),
    ]
    tested = [alias for fold in folds for alias in fold.test_aliases]
    assert sorted(tested) == sorted(CORE_ALIASES)
    for fold in folds:
        assert not (set(fold.train_aliases) & set(fold.validation_aliases))
        assert not (set(fold.train_aliases) & set(fold.test_aliases))
        assert not (set(fold.validation_aliases) & set(fold.test_aliases))

    invalid = list(folds)
    invalid[0] = FoldSplit(
        fold_index=0,
        train_aliases=invalid[0].train_aliases,
        validation_aliases=invalid[0].validation_aliases,
        test_aliases=("ANC-01", "ANC-07"),
    )
    with pytest.raises(ValueError):
        assert_valid_five_fold_splits(invalid)


def test_equal_core_log1p_moments_do_not_weight_a_duplicated_core_more() -> None:
    core_a = np.asarray([[0, 3], [8, 1]], dtype=np.int64)
    core_b = np.asarray([[2, 5], [4, 7], [6, 9]], dtype=np.int64)
    validation_sentinel = np.full((20, 2), 10**8, dtype=np.int64)
    original = fit_equal_core_log1p_standardizer(
        {"A": core_a, "B": core_b, "VAL": validation_sentinel},
        train_aliases=("A", "B"),
    )
    duplicated = fit_equal_core_log1p_standardizer(
        {"A": np.repeat(core_a, 5, axis=0), "B": core_b, "VAL": validation_sentinel},
        train_aliases=("A", "B"),
    )
    np.testing.assert_allclose(original.mean, duplicated.mean, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(original.variance, duplicated.variance, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(original.scale, duplicated.scale, rtol=0.0, atol=1e-15)

    expected_mean = np.mean(
        [np.log1p(core_a).mean(axis=0), np.log1p(core_b).mean(axis=0)], axis=0
    )
    np.testing.assert_allclose(original.mean, expected_mean)
    standardized = transform_log1p_counts(core_a, original, dtype=np.float64)
    restored = inverse_standardized_log1p(standardized, original)
    np.testing.assert_allclose(restored, np.log1p(core_a))


def test_exact_uniform_mask_has_exact_row_counts_endpoints_and_torch_identity() -> None:
    first = sample_uniform_mask_numpy(256, 8, seed=91, chunk_cells=1)
    repeated = sample_uniform_mask_numpy(256, 8, seed=91, chunk_cells=37)
    different = sample_uniform_mask_numpy(256, 8, seed=92)

    np.testing.assert_array_equal(first.mask, repeated.mask)
    np.testing.assert_array_equal(first.masked_gene_counts, first.mask.sum(axis=1))
    assert first.checksum == repeated.checksum
    assert first.checksum != different.checksum
    assert np.any(first.masked_gene_counts == 0)
    assert np.any(first.masked_gene_counts == 8)
    assert not first.mask.flags.writeable
    assert not first.masked_gene_counts.flags.writeable

    torch_mask, torch_counts, torch_checksum = sample_uniform_mask_torch(
        256, 8, seed=91, chunk_cells=19
    )
    np.testing.assert_array_equal(torch_mask.cpu().numpy(), first.mask)
    np.testing.assert_array_equal(torch_counts.cpu().numpy(), first.masked_gene_counts)
    assert torch_checksum == first.checksum


def test_mask_seed_derivation_is_paired_but_evaluation_is_model_seed_independent() -> None:
    training = derive_training_mask_seed(
        fold_index=2, model_seed=4, epoch=17, core_alias="ANC-03"
    )
    assert training == derive_training_mask_seed(
        fold_index=2, model_seed=4, epoch=17, core_alias="ANC-03"
    )
    assert training != derive_training_mask_seed(
        fold_index=2, model_seed=4, epoch=18, core_alias="ANC-03"
    )
    evaluation = derive_evaluation_mask_seed(core_alias="ANC-03", replicate_index=1)
    assert evaluation == derive_evaluation_mask_seed(
        core_alias="ANC-03", replicate_index=1
    )


def _overlapping_fov_geometry() -> tuple[np.ndarray, np.ndarray]:
    # The two FOVs deliberately occupy identical coordinate ranges.  Any
    # failure to use FOV as the construction group would create seam edges.
    x = np.arange(15, dtype=np.float64) * 2.0
    one_fov = np.column_stack([x, np.zeros_like(x)])
    coordinates = np.concatenate([one_fov, one_fov], axis=0)
    fov = np.asarray(["fov-a"] * 15 + ["fov-b"] * 15)
    return coordinates, fov


def _assert_no_cross_fov_edges(edge_index: np.ndarray, fov: np.ndarray) -> None:
    assert np.all(fov[edge_index[0]] == fov[edge_index[1]])


def test_fixed_graph_identity_and_position_null_are_fov_safe_and_audited() -> None:
    coordinates, fov = _overlapping_fov_geometry()
    bundle = materialize_fixed_adjacencies(coordinates, fov, scope="ANC-test")
    repeated = materialize_fixed_adjacencies(coordinates, fov, scope="ANC-test")
    changed_scope = materialize_fixed_adjacencies(coordinates, fov, scope="ANC-other")
    assert isinstance(bundle, AdjacencyBundle)
    assert bundle.checksum == repeated.checksum
    assert bundle.position_assignment_checksum == ndarray_sha256(
        bundle.position_assignment
    )
    assert bundle.position_assignment_checksum != changed_scope.position_assignment_checksum

    for name in ("spatial", "isolated", POSITION_PERMUTED_NULL_ARM):
        arm = bundle.arm(name)
        _assert_no_cross_fov_edges(arm.edge_index, fov)
        loops = arm.edge_index[:, arm.edge_index[0] == arm.edge_index[1]]
        assert loops.shape[1] == coordinates.shape[0]
        np.testing.assert_array_equal(np.sort(loops[0]), np.arange(coordinates.shape[0]))
        assert arm.qc.exact_one_self_loop_per_node
        assert arm.qc.duplicate_directed_edges == 0
        assert arm.qc.cross_fov_edges == 0

    identity = bundle.isolated.edge_index
    assert identity.shape == (2, coordinates.shape[0])
    np.testing.assert_array_equal(identity[0], identity[1])
    assert bundle.isolated.qc.minimum_in_degree == 1
    assert bundle.isolated.qc.maximum_in_degree == 1
    assert (
        bundle.spatial.qc.in_degree_multiset_checksum
        == bundle.position_permuted_null.qc.in_degree_multiset_checksum
    )
    assert (
        bundle.spatial.qc.n_off_diagonal_edges
        == bundle.position_permuted_null.qc.n_off_diagonal_edges
    )
    _assert_no_cross_fov_edges(bundle.true_off_diagonal_edge_index, fov)
    assert np.all(fov[bundle.position_assignment] == fov)


def test_explicit_self_mean_rejects_missing_self_and_identity_is_literal() -> None:
    embedding = torch.tensor([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    identity = torch.arange(3, dtype=torch.long).repeat(2, 1)
    torch.testing.assert_close(
        explicit_self_incoming_mean(embedding, identity), embedding, rtol=0.0, atol=0.0
    )
    missing_self = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    with pytest.raises(ValueError, match="exactly one explicit self"):
        explicit_self_incoming_mean(embedding, missing_self)


def _small_model(seed: int = 7) -> ExplicitSelfMeanGraphSAGE:
    return build_seeded_explicit_self_model(
        6,
        seed=seed,
        hidden_dim=8,
        ffn_dim=12,
        decoder_dim=10,
        dropout=0.0,
    ).eval()


def test_identity_adjacency_prevents_all_cross_cell_passage_but_uses_aggregate_path() -> None:
    model = _small_model()
    expression = torch.randn(3, 6, generator=torch.Generator().manual_seed(22))
    mask = torch.zeros_like(expression, dtype=torch.bool)
    identity = torch.arange(3, dtype=torch.long).repeat(2, 1)

    aggregate_calls: list[torch.Tensor] = []
    handle = model.aggregate_projection.register_forward_hook(
        lambda _module, _inputs, output: aggregate_calls.append(output.detach())
    )
    original = model(expression, mask, identity).prediction
    handle.remove()
    changed = expression.clone()
    changed[1] += 1000.0
    after_change = model(changed, mask, identity).prediction
    torch.testing.assert_close(original[0], after_change[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(original[2], after_change[2], rtol=0.0, atol=0.0)
    assert aggregate_calls and aggregate_calls[0].shape == (3, 8)

    off_diagonal = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    spatial = torch.from_numpy(add_exact_self_adjacency(off_diagonal, n_nodes=3))
    spatial_original = model(expression, mask, spatial).prediction
    spatial_changed = model(changed, mask, spatial).prediction
    assert not torch.allclose(spatial_original[0], spatial_changed[0])


def test_paired_initialization_parameter_count_and_internal_remasking() -> None:
    spatial_model = _small_model(seed=3)
    isolated_model = _small_model(seed=3)
    other_seed_model = _small_model(seed=4)
    assert state_dict_sha256(spatial_model) == state_dict_sha256(isolated_model)
    assert state_dict_sha256(spatial_model) != state_dict_sha256(other_seed_model)
    assert trainable_parameter_count(spatial_model) == trainable_parameter_count(isolated_model)

    frozen = build_seeded_explicit_self_model(1000, seed=0)
    assert trainable_parameter_count(frozen) == 645_736

    expression = torch.randn(4, 6, generator=torch.Generator().manual_seed(2))
    mask = torch.tensor(
        [
            [True, False, True, False, False, True],
            [False, True, False, True, False, False],
            [True, True, False, False, True, False],
            [False, False, True, True, False, True],
        ]
    )
    changed_hidden = expression.clone()
    changed_hidden[mask] += 100_000.0
    off_diagonal = np.asarray(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=np.int64
    )
    adjacency = torch.from_numpy(add_exact_self_adjacency(off_diagonal, n_nodes=4))
    first = spatial_model(expression, mask, adjacency).prediction
    second = spatial_model(changed_hidden, mask, adjacency).prediction
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    zeros = torch.zeros_like(expression)
    no_mask = spatial_model(zeros, torch.zeros_like(mask), adjacency).prediction
    some_mask = spatial_model(zeros, mask, adjacency).prediction
    assert not torch.allclose(no_mask, some_mask)


def test_fixed_target_mask_bins_use_frozen_nonoverlapping_endpoints() -> None:
    genes = 8
    counts = [0, 2, 3, 4, 5, 6, 7, 8]
    mask = np.zeros((len(counts), genes), dtype=bool)
    for row, count in enumerate(counts):
        mask[row, :count] = True
    target = np.arange(mask.size, dtype=np.float64).reshape(mask.shape)
    prediction = target + 2.0
    nodes = np.arange(len(counts), dtype=np.int64)
    # A symmetric cycle gives every receiver true off-diagonal neighbors.
    next_nodes = np.roll(nodes, -1)
    edges = np.concatenate(
        [np.stack([nodes, next_nodes]), np.stack([next_nodes, nodes])], axis=1
    )
    result = evaluate_fixed_mask_strata(target, prediction, mask, edges)
    assert result["zero_mask_cells"] == 1
    assert [
        result["target_mask_bins"][name]["n_target_cells"]
        for name in ("0_to_25", "25_to_50", "50_to_75", "75_to_100", "exactly_100")
    ] == [1, 2, 2, 1, 1]
    for summary in result["target_mask_bins"].values():
        if summary["n_masked"]:
            assert summary["mae"] == pytest.approx(2.0)
            assert summary["mse"] == pytest.approx(4.0)
            assert summary["rmse"] == pytest.approx(2.0)
            assert summary["huber"] == pytest.approx(1.5)


def test_neighbor_availability_is_per_target_gene_on_true_off_diagonal_edges() -> None:
    mask = np.asarray(
        [
            [False, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    # Nodes 0, 1, and 2 are the three true incoming neighbors of node 3.
    edges = np.asarray([[0, 1, 2], [3, 3, 3]], dtype=np.int64)
    availability = true_neighbor_observed_fraction(mask, edges)
    np.testing.assert_allclose(availability[3, :3], [1 / 3, 2 / 3, 1.0])
    np.testing.assert_array_equal(availability[:3], 0.0)

    target = np.arange(16, dtype=np.float64).reshape(4, 4)
    prediction = target + 1.0
    result = evaluate_fixed_mask_strata(target, prediction, mask, edges)
    neighbor = result["neighbor_observed_bins"]
    assert neighbor["0_to_25"]["n_target_entries"] == 3
    assert neighbor["25_to_50"]["n_target_entries"] == 1
    assert neighbor["50_to_75"]["n_target_entries"] == 1
    assert neighbor["75_to_100"]["n_target_entries"] == 1
    assert result["masked_targets_without_true_neighbors"] == 3
    assert result["neighbor_zero_degree_policy"] == "included_as_zero_in_0_to_25_bin"


def test_sparse_chunked_neighbor_availability_matches_bounded_naive_reference() -> None:
    rng = np.random.default_rng(918)
    n_cells, n_genes = 11, 13
    mask = rng.random((n_cells, n_genes)) < 0.45
    # Symmetric graph on nodes 0..8; nodes 9 and 10 intentionally have degree zero.
    pairs = np.asarray(
        [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8]],
        dtype=np.int64,
    )
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    computed = compute_true_neighbor_availability(
        mask, directed, gene_chunk_size=3
    )

    observed_counts = np.zeros((n_cells, n_genes), dtype=np.int64)
    degree = np.zeros(n_cells, dtype=np.int64)
    for source, receiver in directed.T:
        observed_counts[receiver] += ~mask[source]
        degree[receiver] += 1
    expected = np.zeros((n_cells, n_genes), dtype=np.float32)
    np.divide(
        observed_counts,
        degree[:, None],
        out=expected,
        where=degree[:, None] > 0,
    )
    np.testing.assert_allclose(computed.fraction, expected, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(computed.off_diagonal_degree, degree)
    np.testing.assert_array_equal(computed.fraction[degree == 0], 0.0)

    target = rng.normal(size=(n_cells, n_genes))
    prediction = target + rng.normal(scale=0.1, size=target.shape)
    cached = evaluate_fixed_mask_strata(
        target,
        prediction,
        mask,
        directed,
        neighbor_availability=computed,
    )
    uncached = evaluate_fixed_mask_strata(target, prediction, mask, directed)
    assert cached["neighbor_availability_checksum"] == uncached["neighbor_availability_checksum"]
    for name in cached["neighbor_observed_bins"]:
        left = cached["neighbor_observed_bins"][name]
        right = uncached["neighbor_observed_bins"][name]
        assert left["n_target_entries"] == right["n_target_entries"]
        assert left["n_masked"] == right["n_masked"]
        np.testing.assert_allclose(left["huber"], right["huber"], equal_nan=True)


def test_checksum_changes_with_array_dtype_shape_or_value() -> None:
    base = np.asarray([[1, 2], [3, 4]], dtype=np.int64)
    assert ndarray_sha256(base) != ndarray_sha256(base.astype(np.int32))
    assert ndarray_sha256(base) != ndarray_sha256(base.reshape(4))
    changed = base.copy()
    changed[0, 0] = 9
    assert ndarray_sha256(base) != ndarray_sha256(changed)
