from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from spatial_benchmark.expression_tokens import (
    DEFAULT_COUNT_TOKEN_SPEC,
    CountTokenizationSpec,
    audit_expression_tokens,
    evaluate_masked_token_predictions,
    fit_per_gene_modal_tokens,
    tokenize_expression_counts,
)


def test_locked_count_tokenization_and_immutable_spec() -> None:
    counts = np.array(
        [[0, 1, 2, 3, 4], [10, 2, 1, 0, 65535]],
        dtype=np.uint16,
    )
    expected = np.array(
        [[0, 1, 2, 3, 3], [3, 2, 1, 0, 3]],
        dtype=np.int64,
    )
    actual = tokenize_expression_counts(counts)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int64
    assert DEFAULT_COUNT_TOKEN_SPEC.output_token_ids == (0, 1, 2, 3)
    assert DEFAULT_COUNT_TOKEN_SPEC.num_output_tokens == 4
    assert DEFAULT_COUNT_TOKEN_SPEC.mask_token_id == 4
    assert DEFAULT_COUNT_TOKEN_SPEC.vocabulary_size == 5

    with pytest.raises(FrozenInstanceError):
        DEFAULT_COUNT_TOKEN_SPEC.mask_token_id = 9  # type: ignore[misc]
    with pytest.raises(ValueError, match="mask_token_id must be 4"):
        CountTokenizationSpec(mask_token_id=3)


def test_deterministic_audit_covers_tokens_genes_modes_and_checksums() -> None:
    tokens = np.array(
        [
            [0, 0, 1],
            [3, 3, 2],
            [0, 1, 3],
            [3, 2, 0],
        ],
        dtype=np.int64,
    )
    expected_modes = np.array([0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(
        fit_per_gene_modal_tokens(tokens),
        expected_modes,
    )

    first = audit_expression_tokens(tokens)
    repeated = audit_expression_tokens(np.asfortranarray(tokens))
    assert first == repeated
    assert json.loads(json.dumps(first, allow_nan=False)) == first
    assert first["shape"] == [4, 3]
    assert first["token_counts"] == [4, 2, 2, 4]
    assert first["token_prevalence_percent"] == pytest.approx(
        [100 / 3, 100 / 6, 100 / 6, 100 / 3]
    )
    assert first["per_gene_modal_tokens"] == [0, 0, 0]
    assert first["all_output_tokens_present"] is True
    coverage = first["gene_class_coverage"]
    assert coverage["classes_present_per_gene"] == [2, 4, 4]
    assert coverage["genes_with_all_output_tokens"] == 2
    assert coverage["all_genes_have_all_output_tokens"] is False
    assert coverage["genes_containing_each_output_token"] == [3, 2, 2, 3]
    assert len(first["token_checksum_sha256"]) == 64
    assert len(first["per_gene_modal_tokens_checksum_sha256"]) == 64

    changed = tokens.copy()
    changed[0, 0] = 1
    changed_audit = audit_expression_tokens(changed)
    assert (
        changed_audit["token_checksum_sha256"]
        != first["token_checksum_sha256"]
    )


def test_masked_metrics_known_confusion_and_baselines() -> None:
    target = np.array(
        [[0, 1, 2, 3], [0, 1, 2, 3]],
        dtype=np.int64,
    )
    predicted = np.array(
        [[0, 2, 2, 3], [1, 1, 0, 3]],
        dtype=np.int64,
    )
    mask = np.ones_like(target, dtype=bool)
    modes = fit_per_gene_modal_tokens(target)

    result = evaluate_masked_token_predictions(
        target,
        predicted,
        mask,
        cross_entropy=1.25,
        per_gene_modal_tokens=modes,
    )
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["n_masked"] == 8
    assert result["cross_entropy"] == 1.25
    assert result["accuracy_percent"] == pytest.approx(62.5)
    assert result["balanced_accuracy_percent"] == pytest.approx(62.5)
    assert result["nonzero_accuracy_percent"] == pytest.approx(100 * 4 / 6)
    assert result["per_token_support"] == [2, 2, 2, 2]
    assert result["per_token_recall_percent"] == pytest.approx(
        [50.0, 50.0, 50.0, 100.0]
    )
    assert result["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 2],
    ]
    assert result["uniform_chance_accuracy_percent"] == 25.0
    assert result["empirical_frequency_random_accuracy_percent"] == 25.0
    assert result["always_zero_accuracy_percent"] == 25.0
    assert result["always_zero_balanced_accuracy_percent"] == 25.0
    assert result["per_gene_modal_accuracy_percent"] == 100.0
    assert result["per_gene_modal_balanced_accuracy_percent"] == 100.0
    assert result["per_gene_modal_nonzero_accuracy_percent"] == 100.0


def test_sparse_targets_expose_accuracy_imbalance_and_undefined_values() -> None:
    target = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int64)
    predicted = np.zeros_like(target)
    mask = np.ones_like(target, dtype=bool)
    result = evaluate_masked_token_predictions(target, predicted, mask)

    assert result["accuracy_percent"] == pytest.approx(100 * 5 / 6)
    assert result["balanced_accuracy_percent"] == 50.0
    assert result["nonzero_accuracy_percent"] == 0.0
    assert result["per_token_support"] == [5, 1, 0, 0]
    assert result["per_token_recall_percent"] == [100.0, 0.0, None, None]
    assert result["empirical_frequency_random_accuracy_percent"] == pytest.approx(
        100 * (26 / 36)
    )
    assert result["always_zero_accuracy_percent"] == pytest.approx(100 * 5 / 6)
    assert result["always_zero_balanced_accuracy_percent"] == 50.0
    assert result["per_gene_modal_accuracy_percent"] is None
    assert result["per_gene_modal_balanced_accuracy_percent"] is None
    assert result["per_gene_modal_nonzero_accuracy_percent"] is None

    all_zero_target = np.zeros((1, 3), dtype=np.int64)
    no_nonzero = evaluate_masked_token_predictions(
        all_zero_target,
        all_zero_target.copy(),
        np.ones_like(all_zero_target, dtype=bool),
    )
    assert no_nonzero["nonzero_accuracy_percent"] is None


@pytest.mark.parametrize(
    ("counts", "error", "match"),
    [
        (np.array([0, 1], dtype=np.int64), ValueError, "two-dimensional"),
        (np.empty((0, 2), dtype=np.int64), ValueError, "at least one"),
        (np.array([[0.0, 1.0]]), TypeError, "integer dtype"),
        (np.array([[False, True]]), TypeError, "integer dtype"),
        (np.array([[0, -1]], dtype=np.int64), ValueError, "nonnegative"),
    ],
)
def test_tokenizer_rejects_invalid_counts(
    counts: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        tokenize_expression_counts(counts)


def test_audit_and_metrics_reject_invalid_tokens_masks_and_scalars() -> None:
    valid = np.array([[0, 1], [2, 3]], dtype=np.int64)
    bool_mask = np.ones_like(valid, dtype=bool)

    with pytest.raises(ValueError, match="IDs 0 through 3"):
        audit_expression_tokens(np.array([[0, 4]], dtype=np.int64))
    with pytest.raises(TypeError, match="integer dtype"):
        fit_per_gene_modal_tokens(valid.astype(np.float32))
    with pytest.raises(ValueError, match="num_tokens must be 4"):
        fit_per_gene_modal_tokens(valid, num_tokens=3)
    with pytest.raises(ValueError, match="same shape"):
        evaluate_masked_token_predictions(valid, valid[:, :1], bool_mask)
    with pytest.raises(TypeError, match="boolean dtype"):
        evaluate_masked_token_predictions(
            valid,
            valid,
            bool_mask.astype(np.int8),
        )
    with pytest.raises(ValueError, match="selects no"):
        evaluate_masked_token_predictions(
            valid,
            valid,
            np.zeros_like(valid, dtype=bool),
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_masked_token_predictions(
            valid,
            valid,
            bool_mask,
            cross_entropy=np.nan,
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_masked_token_predictions(
            valid,
            valid,
            bool_mask,
            cross_entropy=-0.1,
        )
    with pytest.raises(ValueError, match="one token per target gene"):
        evaluate_masked_token_predictions(
            valid,
            valid,
            bool_mask,
            per_gene_modal_tokens=np.array([0], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="IDs 0 through 3"):
        evaluate_masked_token_predictions(
            valid,
            valid,
            bool_mask,
            per_gene_modal_tokens=np.array([0, 4], dtype=np.int64),
        )
