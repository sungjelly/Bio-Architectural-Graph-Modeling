from __future__ import annotations

import inspect

import numpy as np
import pytest

from spatial_benchmark.attention_routing_niches import (
    ANALYSIS_MASK_VIEW_COUNT,
    AttentionRoutingNicheError,
    audit_attention_normalization,
    build_consensus_routing_partition,
    build_reciprocal_pair_index,
    build_seed_specific_partitions,
    degree_adjusted_routing,
    derive_analysis_mask_seed,
    make_analysis_mask_view,
    make_analysis_mask_views,
    match_seed_partitions_to_consensus,
    mutual_routing_hub_scores,
    partition_similarity,
    retain_top_k_mutual_edges,
    summarize_mutual_routing_consensus,
    summarize_parameter_sensitivity,
    validate_attention_shard,
    weighted_leiden_partition,
)


def _receiver_softmax(edge_index: np.ndarray, logits: np.ndarray) -> np.ndarray:
    result = np.empty_like(logits, dtype=np.float64)
    for receiver in np.unique(edge_index[1]).tolist():
        selected = edge_index[1] == receiver
        values = logits[selected]
        shifted = values - values.max(axis=0, keepdims=True)
        exponentials = np.exp(shifted)
        result[selected] = exponentials / exponentials.sum(axis=0, keepdims=True)
    return result


def _reciprocal_edges(pairs: np.ndarray) -> np.ndarray:
    directed = []
    for first, second in np.asarray(pairs, dtype=np.int64).tolist():
        directed.extend(((first, second), (second, first)))
    return np.asarray(directed, dtype=np.int64).T


def test_analysis_masks_are_exact_paired_and_exclude_model_seed() -> None:
    assert "model_seed" not in inspect.signature(derive_analysis_mask_seed).parameters
    assert "model_seed" not in inspect.signature(make_analysis_mask_view).parameters

    first = make_analysis_mask_views(
        127,
        11,
        core_alias="CAN-01",
        analysis_mask_seed=813,
        chunk_cells=13,
    )
    repeated_for_a_different_conceptual_model_seed = make_analysis_mask_views(
        127,
        11,
        core_alias="can-01",
        analysis_mask_seed=813,
        chunk_cells=64,
    )
    assert len(first) == ANALYSIS_MASK_VIEW_COUNT == 10
    assert [view.mask_view_index for view in first] == list(range(10))
    for left, right in zip(
        first, repeated_for_a_different_conceptual_model_seed, strict=True
    ):
        np.testing.assert_array_equal(left.mask, right.mask)
        np.testing.assert_array_equal(
            left.masked_gene_counts,
            left.mask.sum(axis=1, dtype=np.int64),
        )
        assert left.derived_seed == right.derived_seed
        assert left.receipt_sha256 == right.receipt_sha256
        assert left.to_receipt()["seed_derivation"].endswith(
            "model_seed_excluded"
        )
        assert np.all((left.masked_gene_counts >= 0) & (left.masked_gene_counts <= 11))
    assert len({view.derived_seed for view in first}) == 10
    assert make_analysis_mask_view(
        127,
        11,
        core_alias="CAN-09",
        mask_view_index=0,
        analysis_mask_seed=813,
    ).derived_seed != first[0].derived_seed


def test_attention_shard_validates_alignment_decomposition_and_normalization() -> None:
    edge_index = np.asarray(
        [[1, 2, 0, 2], [0, 0, 1, 1]],
        dtype=np.int64,
    )
    content = np.asarray(
        [[0.0, 0.5], [1.0, -0.5], [0.25, 0.0], [-0.25, 1.0]],
        dtype=np.float64,
    )
    bias = np.asarray(
        [[0.2, 0.0], [-0.1, 0.2], [0.1, -0.3], [0.4, 0.0]],
        dtype=np.float64,
    )
    combined = content + bias
    attention = _receiver_softmax(edge_index, combined)
    expected_degree = np.asarray([2, 2, 0], dtype=np.int64)

    result = validate_attention_shard(
        edge_index,
        attention,
        content,
        bias,
        combined,
        n_nodes=3,
        source_indices=edge_index[0],
        receiver_indices=edge_index[1],
        expected_edge_index=edge_index,
        expected_edge_ids=np.arange(edge_index.shape[1]),
        expected_receiver_in_degrees=expected_degree,
    )
    assert result.edge_count == 4
    assert result.head_count == 2
    assert result.normalization.passed
    np.testing.assert_allclose(result.normalization.per_head_sums, 1.0)

    bad_combined = combined.copy()
    bad_combined[0, 0] += 0.1
    with pytest.raises(AttentionRoutingNicheError, match="content_logits"):
        validate_attention_shard(
            edge_index,
            attention,
            content,
            bias,
            bad_combined,
            n_nodes=3,
        )

    with pytest.raises(AttentionRoutingNicheError, match="truncates"):
        validate_attention_shard(
            edge_index[:, :1],
            np.ones((1, 2)),
            content[:1],
            bias[:1],
            combined[:1],
            n_nodes=3,
            expected_edge_index=edge_index,
            expected_edge_ids=np.asarray([0]),
        )


def test_attention_normalization_audit_rejects_nonunit_receiver_head_sum() -> None:
    edge_index = np.asarray([[0, 2, 1], [1, 1, 0]], dtype=np.int64)
    attention = np.asarray([[0.25, 0.5], [0.75, 0.4], [1.0, 1.0]])
    with pytest.raises(AttentionRoutingNicheError, match="does not sum to one"):
        audit_attention_normalization(
            edge_index,
            attention,
            n_nodes=3,
        )


def test_degree_adjustment_uses_receiver_indegree_not_sender_degree() -> None:
    # Receiver 1 has degree two; receivers 0 and 2 each have degree one.
    edge_index = np.asarray(
        [[0, 2, 1, 1], [1, 1, 0, 2]],
        dtype=np.int64,
    )
    attention = np.asarray(
        [[0.25, 0.25], [0.75, 0.75], [1.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    result = degree_adjusted_routing(edge_index, attention, n_nodes=3)
    np.testing.assert_array_equal(result.receiver_in_degree, [2, 2, 1, 1])
    np.testing.assert_allclose(result.enrichment, [0.5, 1.5, 1.0, 1.0])

    pairs = build_reciprocal_pair_index(
        edge_index,
        n_nodes=3,
        require_all_edges_reciprocal=True,
    )
    np.testing.assert_array_equal(pairs.pair_cells, [[0, 1], [1, 2]])
    mutual = np.minimum(
        result.enrichment[pairs.i_to_j_edge_ids],
        result.enrichment[pairs.j_to_i_edge_ids],
    )
    np.testing.assert_allclose(mutual, [0.5, 1.0])


def test_one_directional_edge_is_never_treated_as_mutual() -> None:
    edge_index = np.asarray(
        [[0, 1, 2], [1, 0, 1]],
        dtype=np.int64,
    )
    pairs = build_reciprocal_pair_index(edge_index, n_nodes=3)
    np.testing.assert_array_equal(pairs.pair_cells, [[0, 1]])
    np.testing.assert_array_equal(pairs.one_way_edge_ids, [2])
    assert pairs.directed_pair_positions[2] == -1
    assert pairs.reverse_edge_ids[2] == -1
    with pytest.raises(AttentionRoutingNicheError, match="one-directional"):
        build_reciprocal_pair_index(
            edge_index,
            n_nodes=3,
            require_all_edges_reciprocal=True,
        )


def test_consensus_uses_reciprocal_minimum_strict_support_and_axis_spreads() -> None:
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    # The last axis is [0->1, 1->0].
    routing = np.asarray(
        [
            [[2.0, 4.0], [0.5, 5.0]],
            [[3.0, 1.5], [4.0, 0.8]],
        ],
        dtype=np.float64,
    )
    result = summarize_mutual_routing_consensus(
        edge_index,
        routing,
        n_nodes=2,
        seed_ids=(11, 17),
        mask_view_ids=(3, 9),
    )
    chunked_one_pair_at_a_time = summarize_mutual_routing_consensus(
        edge_index,
        routing.astype(np.float32),
        n_nodes=2,
        seed_ids=(11, 17),
        mask_view_ids=(3, 9),
        pair_chunk_size=1,
    )
    np.testing.assert_allclose(result.mutual_samples[:, :, 0], [[2.0, 0.5], [1.5, 0.8]])
    np.testing.assert_allclose(result.Mij, [1.15])
    # Strict > 1 excludes neither equality here; exactly two of four pass.
    np.testing.assert_allclose(result.Pij, [0.5])
    np.testing.assert_allclose(result.i_to_j_directional_median, [2.5])
    np.testing.assert_allclose(result.j_to_i_directional_median, [2.75])
    np.testing.assert_allclose(result.per_seed_view_median[:, 0], [1.25, 1.15])
    np.testing.assert_allclose(result.seed_mean_after_view_median, [1.2])
    np.testing.assert_allclose(
        result.seed_standard_deviation_after_view_median,
        [np.std([1.25, 1.15], ddof=1)],
    )
    np.testing.assert_allclose(result.per_mask_view_seed_median[:, 0], [1.75, 0.65])
    np.testing.assert_allclose(result.mask_view_mean_after_seed_median, [1.2])
    np.testing.assert_allclose(
        result.mask_view_standard_deviation_after_seed_median,
        [np.std([1.75, 0.65], ddof=1)],
    )
    np.testing.assert_allclose(
        chunked_one_pair_at_a_time.Mij,
        result.Mij,
        atol=1.0e-7,
    )

    strict = summarize_mutual_routing_consensus(
        edge_index,
        np.ones((1, 2, 2), dtype=np.float64),
        n_nodes=2,
    )
    np.testing.assert_allclose(strict.Pij, [0.0])
    assert strict.seed_standard_deviation_after_view_median is None


def test_thresholds_and_per_cell_top_k_use_deterministic_undirected_union() -> None:
    pair_cells = np.asarray(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=np.int64,
    )
    scores = np.asarray([5.0, 4.0, 4.0, 6.0, 6.0, 2.0])
    support = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.59])
    retained = retain_top_k_mutual_edges(
        pair_cells,
        scores,
        support,
        n_nodes=4,
        top_k=1,
        score_threshold=1.0,
        support_threshold=0.60,
    )
    # Cell 1's score-six tie is resolved by smaller neighbor 2.  Cell 3 still
    # independently selects 1--3, so the undirected union retains both.
    np.testing.assert_array_equal(
        retained.edge_pairs,
        [[0, 1], [1, 2], [1, 3]],
    )
    np.testing.assert_allclose(retained.weights, [5.0, 6.0, 6.0])
    np.testing.assert_array_equal(retained.endpoint_selection_count, [1, 2, 1])
    assert retained.eligible_pair_count == 5
    np.testing.assert_allclose(
        mutual_routing_hub_scores(4, retained.edge_pairs, retained.weights),
        [5.0, 17.0, 6.0, 6.0],
    )

    # Eligibility is strict Mij > 1, not >= 1.
    strict = retain_top_k_mutual_edges(
        np.asarray([[0, 1]], dtype=np.int64),
        np.asarray([1.0]),
        np.asarray([1.0]),
        n_nodes=2,
        top_k=1,
    )
    assert strict.edge_count == 0


def test_weighted_leiden_is_deterministic_per_core_and_keeps_isolates() -> None:
    # Two disconnected weighted triangles plus isolated cell 6.
    edges = np.asarray(
        [[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]],
        dtype=np.int64,
    )
    weights = np.asarray([3.0, 2.0, 4.0, 3.0, 2.0, 4.0])
    first = weighted_leiden_partition(
        7,
        edges,
        weights,
        resolution=1.0,
        random_seed=41,
        core_alias="CAN-01",
    )
    second = weighted_leiden_partition(
        7,
        edges[::-1],
        weights[::-1],
        resolution=1.0,
        random_seed=41,
        core_alias="CAN-01",
    )
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.community_count == 3
    assert len(set(first.labels[:3].tolist())) == 1
    assert len(set(first.labels[3:6].tolist())) == 1
    assert first.labels[0] != first.labels[3]
    assert first.labels[6] not in {first.labels[0], first.labels[3]}
    assert first.receipt["weighted"] is True
    assert first.receipt["isolated_vertices_retained"] is True


def test_seed_partitions_and_maximum_jaccard_agreement() -> None:
    undirected = np.asarray(
        [[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5]],
        dtype=np.int64,
    )
    edge_index = _reciprocal_edges(undirected)
    routing = np.full((2, 10, edge_index.shape[1]), 2.0, dtype=np.float64)
    consensus = summarize_mutual_routing_consensus(
        edge_index,
        routing,
        n_nodes=7,
        seed_ids=(3, 8),
        require_all_edges_reciprocal=True,
    )
    primary = build_consensus_routing_partition(
        consensus,
        n_nodes=7,
        top_k=8,
        core_alias="CAN-01",
    )
    seeds = build_seed_specific_partitions(
        consensus,
        n_nodes=7,
        top_k=8,
        core_alias="CAN-01",
    )
    assert [partition.seed_id for partition in seeds] == [3, 8]
    for partition in seeds:
        np.testing.assert_array_equal(
            partition.graph_partition.leiden.labels,
            primary.leiden.labels,
        )

    consensus_labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    seed_labels = np.asarray(
        [
            [1, 1, 1, 0, 0, 0],  # exact partition with swapped IDs
            [0, 0, 1, 2, 2, 2],  # one consensus niche is split
        ],
        dtype=np.int64,
    )
    agreement = match_seed_partitions_to_consensus(
        consensus_labels,
        seed_labels,
        seed_ids=(3, 8),
    )
    np.testing.assert_allclose(
        agreement.cell_assignment_agreement,
        [1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        agreement.niche_assignment_agreement,
        [5.0 / 6.0, 1.0],
    )
    np.testing.assert_allclose(
        agreement.niche_mean_matched_jaccard,
        [(1.0 + 2.0 / 3.0) / 2.0, 1.0],
    )


def test_partition_sensitivity_helpers_are_label_invariant_and_primary_locked() -> None:
    primary = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    relabelled = np.asarray([9, 9, 4, 4, 7, 7], dtype=np.int64)
    changed = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    identical = partition_similarity(primary, relabelled)
    assert identical.adjusted_rand_index == pytest.approx(1.0)
    assert identical.coassignment_jaccard == pytest.approx(1.0)
    assert identical.maximum_jaccard_cell_agreement == pytest.approx(1.0)

    rows = summarize_parameter_sensitivity(
        {
            (5, 1.0): changed,
            (8, 1.0): primary,
            (10, 1.0): relabelled,
            (8, 0.5): changed,
            (8, 1.5): changed,
        }
    )
    assert [(row.top_k, row.resolution) for row in rows] == [
        (5, 1.0),
        (8, 0.5),
        (8, 1.0),
        (8, 1.5),
        (10, 1.0),
    ]
    primary_row = next(row for row in rows if row.is_primary)
    assert primary_row.adjusted_rand_index_to_primary == pytest.approx(1.0)
    assert next(row for row in rows if row.top_k == 10).adjusted_rand_index_to_primary == pytest.approx(1.0)
