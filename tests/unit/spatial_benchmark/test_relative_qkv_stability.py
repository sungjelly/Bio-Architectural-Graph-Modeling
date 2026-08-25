from __future__ import annotations

import numpy as np
import pytest

from spatial_benchmark.relative_qkv_stability import (
    ENSEMBLE_SPREAD_LABEL,
    SEED_UNCERTAINTY_LABEL,
    FixedSeedArray,
    StabilityContractError,
    attention_head_signature,
    content_position_contribution_agreement,
    embedding_stability,
    linear_cka,
    match_attention_heads,
    matched_head_attention_stability,
    mutual_routing_pair_stability,
    mutual_routing_scores,
    orthogonal_procrustes_embedding_similarity,
    positional_bias_response_stability,
    selected_gradient_stability,
    spearman_correlation,
    summarize_relationship_ensemble,
)


def _records(
    arrays: list[np.ndarray],
    *,
    identifiers: tuple[object, ...] | None = None,
    seed_order: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> tuple[FixedSeedArray, ...]:
    if identifiers is None:
        identifiers = tuple(f"item-{index}" for index in range(len(arrays[0])))
    return tuple(
        FixedSeedArray(
            seed=seed,
            fixed_input_ids=identifiers,
            values=arrays[index],
        )
        for index, seed in enumerate(seed_order)
    )


def _permuted_head_fixture() -> tuple[
    tuple[FixedSeedArray, ...], tuple[np.ndarray, ...]
]:
    rng = np.random.default_rng(711)
    reference = rng.normal(size=(120, 4))
    candidate_column_orders = (
        np.asarray([0, 1, 2, 3]),
        np.asarray([2, 0, 3, 1]),
        np.asarray([3, 2, 1, 0]),
        np.asarray([1, 3, 0, 2]),
        np.asarray([0, 2, 1, 3]),
    )
    arrays = [reference[:, order] for order in candidate_column_orders]
    records = _records(arrays)
    expected_reference_to_seed = tuple(
        np.argsort(order) for order in candidate_column_orders
    )
    return records, expected_reference_to_seed


def test_linear_cka_and_procrustes_recover_rotated_embedding() -> None:
    rng = np.random.default_rng(44)
    embedding = rng.normal(size=(80, 6))
    orthogonal, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    transformed = 3.5 * embedding @ orthogonal + 19.0

    assert linear_cka(embedding, transformed) == pytest.approx(1.0, abs=1e-12)
    assert orthogonal_procrustes_embedding_similarity(
        embedding, transformed
    ) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(StabilityContractError, match="constant"):
        linear_cka(np.ones((8, 2)), np.ones((8, 3)))


def test_embedding_stability_requires_five_aligned_seeds() -> None:
    rng = np.random.default_rng(91)
    base = rng.normal(size=(30, 5))
    records = _records([base + seed * 0.001 for seed in range(5)])
    report = embedding_stability(records)
    assert report.seeds == (0, 1, 2, 3, 4)
    np.testing.assert_allclose(np.diag(report.linear_cka), np.ones(5))
    assert report.orthogonal_procrustes_similarity is not None
    assert report.spread_label == ENSEMBLE_SPREAD_LABEL
    assert report.uncertainty_label == SEED_UNCERTAINTY_LABEL

    with pytest.raises(StabilityContractError, match="exactly five"):
        embedding_stability(records[:4])
    misaligned = list(records)
    misaligned[-1] = FixedSeedArray(
        seed=4,
        fixed_input_ids=tuple(reversed(records[-1].fixed_input_ids)),
        values=records[-1].values,
    )
    with pytest.raises(StabilityContractError, match="not identically aligned"):
        embedding_stability(misaligned)


def test_explicit_four_seed_amendment_uses_dynamic_ensemble_dimensions() -> None:
    rng = np.random.default_rng(20260824)
    seed_order = (0, 1, 2, 3)
    base_embedding = rng.normal(size=(40, 6))
    embedding_records = _records(
        [base_embedding + seed * 0.001 for seed in seed_order],
        seed_order=seed_order,
    )
    embedding_report = embedding_stability(
        embedding_records,
        expected_seed_count=4,
    )
    assert embedding_report.seeds == seed_order
    assert embedding_report.linear_cka.shape == (4, 4)
    assert embedding_report.orthogonal_procrustes_similarity is not None
    assert embedding_report.orthogonal_procrustes_similarity.shape == (4, 4)

    reference = rng.normal(size=(80, 3))
    column_orders = (
        np.asarray([0, 1, 2]),
        np.asarray([2, 0, 1]),
        np.asarray([1, 2, 0]),
        np.asarray([0, 2, 1]),
    )
    attention_records = _records(
        [reference[:, order] for order in column_orders],
        seed_order=seed_order,
    )
    alignment = match_attention_heads(
        attention_records,
        expected_seed_count=4,
    )
    assert alignment.seeds == seed_order
    assert alignment.reference_to_seed_head.shape == (4, 3)
    attention_report = matched_head_attention_stability(
        attention_records,
        alignment,
        top_k=8,
    )
    position_report = positional_bias_response_stability(
        attention_records,
        alignment,
    )
    contribution_report = content_position_contribution_agreement(
        attention_records,
        attention_records,
        alignment,
    )
    assert attention_report.matched_head_spearman.shape == (4, 3)
    assert attention_report.matched_head_top_edge_jaccard.shape == (4, 3)
    assert position_report.matched_head_spearman.shape == (4, 3)
    assert contribution_report.within_seed_spearman.shape == (4, 3)

    relationship_ids = ("relation-a", "relation-b")
    relationship_records = _records(
        [
            np.asarray([float(seed + 1), -float(seed + 1)])
            for seed in seed_order
        ],
        identifiers=relationship_ids,
        seed_order=seed_order,
    )
    relationship_report = summarize_relationship_ensemble(
        relationship_records,
        expected_seed_count=4,
    )
    np.testing.assert_array_equal(relationship_report.support_count, [4, 4])
    assert relationship_report.seeds == seed_order
    assert relationship_report.calibrated_confidence_interval is False

    mutual_report = mutual_routing_pair_stability(
        relationship_records,
        top_k=1,
        expected_seed_count=4,
    )
    gradient_report = selected_gradient_stability(
        relationship_records,
        expected_seed_count=4,
    )
    assert int(mutual_report.support_count.max()) <= 4
    assert gradient_report.pairwise_seed_spearman.shape == (4, 4)


def test_four_seed_records_still_require_explicit_amendment_count() -> None:
    seed_order = (0, 1, 2, 3)
    records = _records(
        [np.arange(8, dtype=float)[:, None] + seed for seed in seed_order],
        seed_order=seed_order,
    )
    with pytest.raises(StabilityContractError, match="exactly five"):
        embedding_stability(records)
    with pytest.raises(StabilityContractError, match="greater than or equal to two"):
        embedding_stability(records, expected_seed_count=1)


def test_five_seeds_must_be_distinct() -> None:
    values = [np.arange(10, dtype=float)[:, None] for _ in range(5)]
    duplicate_seed_records = _records(
        values, seed_order=(0, 1, 2, 3, 3)
    )
    with pytest.raises(StabilityContractError, match="distinct seeds"):
        embedding_stability(duplicate_seed_records)


def test_attention_signature_combines_standardized_fixed_channels() -> None:
    attention = np.asarray([[1.0, 4.0], [2.0, 6.0], [3.0, 8.0]])
    content = attention * 10.0
    position = -attention
    signature = attention_head_signature(
        attention, content_logits=content, positional_bias=position
    )
    assert signature.shape == (9, 2)
    for block in np.split(signature, 3):
        np.testing.assert_allclose(block.mean(axis=0), np.zeros(2), atol=1e-15)
        np.testing.assert_allclose(block.std(axis=0), np.ones(2), atol=1e-15)


def test_hungarian_head_matching_recovers_permutations_reproducibly() -> None:
    records, expected = _permuted_head_fixture()
    shuffled_records = (records[3], records[0], records[4], records[2], records[1])
    first = match_attention_heads(shuffled_records, reference_seed=0)
    second = match_attention_heads(shuffled_records, reference_seed=0)
    assert first.seeds == (0, 1, 2, 3, 4)
    for seed in range(5):
        np.testing.assert_array_equal(
            first.reference_to_seed_head[seed], expected[seed]
        )
    np.testing.assert_array_equal(
        first.reference_to_seed_head, second.reference_to_seed_head
    )
    np.testing.assert_allclose(first.matched_signature_spearman, np.ones((5, 4)))


def test_hungarian_matching_has_deterministic_constant_signature_ties() -> None:
    records = _records([np.ones((20, 3)) for _ in range(5)])
    alignment = match_attention_heads(records)
    expected = np.tile(np.arange(3), (5, 1))
    np.testing.assert_array_equal(alignment.reference_to_seed_head, expected)
    assert np.isnan(alignment.signature_spearman[1:]).all()


def test_matched_attention_and_positional_bias_are_compared_after_alignment() -> None:
    attention_records, _ = _permuted_head_fixture()
    alignment = match_attention_heads(attention_records)
    attention_report = matched_head_attention_stability(
        attention_records, alignment, top_k=12
    )
    np.testing.assert_allclose(
        attention_report.matched_head_spearman, np.ones((5, 4))
    )
    np.testing.assert_allclose(
        attention_report.matched_head_top_edge_jaccard, np.ones((5, 4))
    )
    assert attention_report.top_k == 12

    position_records = _records(
        [np.asarray(record.values) * 0.25 for record in attention_records]
    )
    position_report = positional_bias_response_stability(
        position_records, alignment
    )
    np.testing.assert_allclose(
        position_report.matched_head_spearman, np.ones((5, 4))
    )


def test_content_position_agreement_reports_sign_scale_and_rank() -> None:
    content_records, _ = _permuted_head_fixture()
    alignment = match_attention_heads(content_records)
    position_records = _records(
        [np.asarray(record.values) * 2.0 for record in content_records]
    )
    report = content_position_contribution_agreement(
        content_records, position_records, alignment
    )
    np.testing.assert_allclose(report.within_seed_spearman, np.ones((5, 4)))
    np.testing.assert_allclose(report.same_sign_fraction, np.ones((5, 4)))
    np.testing.assert_allclose(
        report.content_absolute_fraction,
        np.full((5, 4), 1.0 / 3.0),
        rtol=1e-10,
    )


def test_relationship_summary_reports_spread_quantiles_and_support() -> None:
    identifiers = ("relation-a", "relation-b")
    records = _records(
        [np.asarray([float(seed), -1.0]) for seed in range(5)],
        identifiers=identifiers,
    )
    report = summarize_relationship_ensemble(
        records, quantile_levels=(0.25, 0.5, 0.75)
    )
    np.testing.assert_allclose(report.mean, [2.0, -1.0])
    np.testing.assert_allclose(report.standard_deviation, [np.sqrt(2.5), 0.0])
    np.testing.assert_allclose(report.median, [2.0, -1.0])
    np.testing.assert_allclose(report.minimum, [0.0, -1.0])
    np.testing.assert_allclose(report.maximum, [4.0, -1.0])
    np.testing.assert_array_equal(report.support_count, [4, 5])
    np.testing.assert_allclose(report.quantiles[0], [1.0, 2.0, 3.0])
    row = report.to_rows()[0]
    assert row["spread_label"] == "ensemble spread"
    assert row["uncertainty_label"] == "seed uncertainty"
    assert report.calibrated_confidence_interval is False

    invalid_support = _records(
        [np.asarray([2.0, 0.0]) for _ in range(5)], identifiers=identifiers
    )
    with pytest.raises(StabilityContractError, match="binary masks"):
        summarize_relationship_ensemble(records, support_records=invalid_support)


def test_mutual_routing_uses_receiver_degree_and_reciprocal_minimum() -> None:
    edge_index = np.asarray(
        [[0, 1, 2, 1], [1, 0, 1, 2]], dtype=np.int64
    )
    attention = np.asarray(
        [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [0.25, 0.25]]
    )
    result = mutual_routing_scores(edge_index, attention)
    assert result.pair_ids == ((0, 1), (1, 2))
    np.testing.assert_allclose(result.scores, [0.5, 0.25])


def test_mutual_routing_stability_counts_top_pair_seed_support() -> None:
    pair_ids = ((0, 1), (1, 2), (2, 3))
    records = _records(
        [
            np.asarray([3.0, 2.0, 1.0]),
            np.asarray([4.0, 3.0, 1.0]),
            np.asarray([1.0, 4.0, 2.0]),
            np.asarray([5.0, 2.0, 1.0]),
            np.asarray([1.0, 3.0, 2.0]),
        ],
        identifiers=pair_ids,
    )
    report = mutual_routing_pair_stability(records, top_k=1)
    np.testing.assert_array_equal(report.support_count, [3, 2, 0])
    assert report.relationship_ids == pair_ids


def test_selected_gradient_stability_is_bounded_to_selected_relationships() -> None:
    selected_ids = (
        ("source-a", "target-a"),
        ("source-b", "target-b"),
        ("source-c", "target-c"),
    )
    records = _records(
        [
            np.asarray([1.0, -2.0, 0.0]),
            np.asarray([2.0, -1.0, 0.0]),
            np.asarray([3.0, -3.0, 0.0]),
            np.asarray([4.0, 2.0, 0.0]),
            np.asarray([5.0, -4.0, 0.0]),
        ],
        identifiers=selected_ids,
    )
    report = selected_gradient_stability(records, magnitude_threshold=0.1)
    np.testing.assert_array_equal(report.summary.support_count, [5, 5, 0])
    np.testing.assert_array_equal(report.ensemble_median_sign, [1.0, -1.0, 0.0])
    np.testing.assert_array_equal(report.consistent_sign_support_count, [5, 4, 0])
    np.testing.assert_allclose(report.consistent_sign_fraction, [1.0, 0.8, 0.0])
    assert report.pairwise_seed_spearman.shape == (5, 5)
    assert "selected relationships only" in report.analysis_scope
    assert "exhaustive Jacobian" in report.analysis_scope


def test_selected_single_gradient_does_not_imply_rank_stability() -> None:
    records = _records(
        [np.asarray([float(seed + 1)]) for seed in range(5)],
        identifiers=(("source", "target"),),
    )
    report = selected_gradient_stability(records)
    np.testing.assert_array_equal(report.summary.support_count, [5])
    assert np.isnan(report.pairwise_seed_spearman[0, 1])
    np.testing.assert_array_equal(
        np.diag(report.pairwise_seed_spearman), np.ones(5)
    )


def test_spearman_uses_average_tie_ranks_and_constants_are_undefined() -> None:
    first = np.asarray([1.0, 1.0, 2.0, 3.0])
    second = np.asarray([4.0, 4.0, 8.0, 12.0])
    assert spearman_correlation(first, second) == pytest.approx(1.0)
    assert np.isnan(spearman_correlation(np.ones(4), np.arange(4.0)))
