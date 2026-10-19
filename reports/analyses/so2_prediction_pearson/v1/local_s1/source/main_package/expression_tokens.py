"""Locked raw-count tokenization and masked classification metrics.

The output vocabulary is deliberately small and fixed:

``0`` = count 0, ``1`` = count 1, ``2`` = count 2, and
``3`` = count 3 or greater.  Token ``4`` is reserved for masked model inputs
and is never a valid prediction target.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


_LOCKED_SPEC_NAME = "raw_count_tokens_0_1_2_3plus_v1"
_LOCKED_OUTPUT_TOKEN_LABELS = ("count0", "count1", "count2", "count>=3")
_LOCKED_MASK_TOKEN_ID = 4
_LOCKED_MASK_TOKEN_LABEL = "masked"
_NUM_OUTPUT_TOKENS = 4


@dataclass(frozen=True, slots=True)
class CountTokenizationSpec:
    """Immutable description of the locked raw-count vocabulary.

    Constructor arguments exist so the specification serializes naturally as
    data, but values other than the locked contract are rejected.
    """

    name: str = _LOCKED_SPEC_NAME
    output_token_labels: tuple[str, ...] = _LOCKED_OUTPUT_TOKEN_LABELS
    mask_token_id: int = _LOCKED_MASK_TOKEN_ID
    mask_token_label: str = _LOCKED_MASK_TOKEN_LABEL

    def __post_init__(self) -> None:
        if self.name != _LOCKED_SPEC_NAME:
            raise ValueError(f"name must be {_LOCKED_SPEC_NAME!r}")
        if self.output_token_labels != _LOCKED_OUTPUT_TOKEN_LABELS:
            raise ValueError(
                "output_token_labels must preserve the locked count vocabulary"
            )
        if self.mask_token_id != _LOCKED_MASK_TOKEN_ID:
            raise ValueError("mask_token_id must be 4")
        if self.mask_token_label != _LOCKED_MASK_TOKEN_LABEL:
            raise ValueError("mask_token_label must be 'masked'")

    @property
    def output_token_ids(self) -> tuple[int, ...]:
        """Valid target and prediction token IDs."""

        return tuple(range(_NUM_OUTPUT_TOKENS))

    @property
    def num_output_tokens(self) -> int:
        """Number of target classes, excluding the input-only mask token."""

        return _NUM_OUTPUT_TOKENS

    @property
    def vocabulary_size(self) -> int:
        """Total input vocabulary size, including the mask token."""

        return self.mask_token_id + 1

    def to_json_dict(self) -> dict[str, Any]:
        """Return the specification using only JSON-native values."""

        return {
            "name": self.name,
            "output_tokens": [
                {"id": token_id, "label": label}
                for token_id, label in enumerate(self.output_token_labels)
            ],
            "mask_input_token": {
                "id": self.mask_token_id,
                "label": self.mask_token_label,
            },
            "num_output_tokens": self.num_output_tokens,
            "vocabulary_size": self.vocabulary_size,
        }


DEFAULT_COUNT_TOKEN_SPEC = CountTokenizationSpec()


def _require_locked_spec(spec: CountTokenizationSpec) -> CountTokenizationSpec:
    if not isinstance(spec, CountTokenizationSpec):
        raise TypeError("spec must be a CountTokenizationSpec")
    # A frozen dataclass is sufficient for normal use.  Rechecking its fields
    # also catches instances altered through low-level object mutation.
    CountTokenizationSpec(
        name=spec.name,
        output_token_labels=spec.output_token_labels,
        mask_token_id=spec.mask_token_id,
        mask_token_label=spec.mask_token_label,
    )
    return spec


def _require_two_dimensional_array(value: Any, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional [cells, genes] array")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one cell and one gene")
    return value


def _require_num_tokens(num_tokens: int) -> int:
    if isinstance(num_tokens, (bool, np.bool_)) or not isinstance(
        num_tokens, (int, np.integer)
    ):
        raise TypeError("num_tokens must be an integer")
    if int(num_tokens) != _NUM_OUTPUT_TOKENS:
        raise ValueError("num_tokens must be 4 for the locked output vocabulary")
    return _NUM_OUTPUT_TOKENS


def _validate_output_tokens(
    value: Any,
    *,
    name: str,
    num_tokens: int = _NUM_OUTPUT_TOKENS,
) -> np.ndarray:
    tokens = _require_two_dimensional_array(value, name=name)
    if tokens.dtype.kind not in "iu" or tokens.dtype.kind == "b":
        raise TypeError(f"{name} must have an integer dtype")
    minimum = int(tokens.min())
    maximum = int(tokens.max())
    if minimum < 0 or maximum >= num_tokens:
        raise ValueError(
            f"{name} must contain only output token IDs 0 through {num_tokens - 1}"
        )
    return tokens


def _canonical_int64_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value, dtype="<i8")
    header = json.dumps(
        {"dtype": "int64-le", "shape": [int(size) for size in canonical.shape]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def tokenize_expression_counts(
    counts: np.ndarray,
    spec: CountTokenizationSpec = DEFAULT_COUNT_TOKEN_SPEC,
) -> np.ndarray:
    """Map nonnegative integer counts to locked output token IDs.

    The input must be a nonempty ``[cells, genes]`` NumPy integer array.
    Returned IDs always have dtype ``int64`` and never include the input-only
    mask token.
    """

    _require_locked_spec(spec)
    count_array = _require_two_dimensional_array(counts, name="counts")
    if count_array.dtype.kind not in "iu" or count_array.dtype.kind == "b":
        raise TypeError("counts must have an integer dtype")
    if count_array.dtype.kind == "i" and int(count_array.min()) < 0:
        raise ValueError("counts must be nonnegative")
    return np.minimum(count_array, 3).astype(np.int64, copy=False)


def _per_gene_token_counts(
    tokens: np.ndarray,
    *,
    num_tokens: int,
) -> np.ndarray:
    counts = np.empty((num_tokens, tokens.shape[1]), dtype=np.int64)
    for token_id in range(num_tokens):
        counts[token_id] = np.count_nonzero(tokens == token_id, axis=0)
    return counts


def fit_per_gene_modal_tokens(
    tokens: np.ndarray,
    num_tokens: int = _NUM_OUTPUT_TOKENS,
) -> np.ndarray:
    """Fit one deterministic modal output token per gene.

    Ties are resolved toward the lowest token ID through ``argmax``'s
    first-occurrence rule.
    """

    num_tokens = _require_num_tokens(num_tokens)
    token_array = _validate_output_tokens(
        tokens,
        name="tokens",
        num_tokens=num_tokens,
    )
    per_gene_counts = _per_gene_token_counts(
        token_array,
        num_tokens=num_tokens,
    )
    return np.argmax(per_gene_counts, axis=0).astype(np.int64, copy=False)


def audit_expression_tokens(
    tokens: np.ndarray,
    spec: CountTokenizationSpec = DEFAULT_COUNT_TOKEN_SPEC,
) -> dict[str, Any]:
    """Return a deterministic, JSON-safe audit of a token matrix."""

    spec = _require_locked_spec(spec)
    token_array = _validate_output_tokens(
        tokens,
        name="tokens",
        num_tokens=spec.num_output_tokens,
    )
    n_cells, n_genes = (int(size) for size in token_array.shape)
    n_entries = int(token_array.size)
    per_gene_counts = _per_gene_token_counts(
        token_array,
        num_tokens=spec.num_output_tokens,
    )
    token_counts = per_gene_counts.sum(axis=1, dtype=np.int64)
    token_prevalence = token_counts.astype(np.float64) / n_entries
    class_presence_by_gene = per_gene_counts > 0
    classes_present_per_gene = class_presence_by_gene.sum(axis=0)
    genes_with_all_classes = int(
        np.count_nonzero(classes_present_per_gene == spec.num_output_tokens)
    )
    per_gene_modes = np.argmax(per_gene_counts, axis=0).astype(
        np.int64,
        copy=False,
    )

    return {
        "spec": spec.to_json_dict(),
        "shape": [n_cells, n_genes],
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_entries": n_entries,
        "token_counts": [int(value) for value in token_counts],
        "token_prevalence": [float(value) for value in token_prevalence],
        "token_prevalence_percent": [
            float(100.0 * value) for value in token_prevalence
        ],
        "all_output_tokens_present": bool(np.all(token_counts > 0)),
        "gene_class_coverage": {
            "classes_present_per_gene": [
                int(value) for value in classes_present_per_gene
            ],
            "genes_with_all_output_tokens": genes_with_all_classes,
            "genes_with_all_output_tokens_percent": float(
                100.0 * genes_with_all_classes / n_genes
            ),
            "all_genes_have_all_output_tokens": bool(
                genes_with_all_classes == n_genes
            ),
            "genes_containing_each_output_token": [
                int(value)
                for value in class_presence_by_gene.sum(axis=1, dtype=np.int64)
            ],
            "gene_coverage_percent_by_output_token": [
                float(100.0 * value / n_genes)
                for value in class_presence_by_gene.sum(
                    axis=1,
                    dtype=np.int64,
                )
            ],
        },
        "token_checksum_sha256": _canonical_int64_sha256(token_array),
        "per_gene_modal_tokens": [int(value) for value in per_gene_modes],
        "per_gene_modal_tokens_checksum_sha256": _canonical_int64_sha256(
            per_gene_modes
        ),
    }


def _validate_cross_entropy(cross_entropy: Any) -> float | None:
    if cross_entropy is None:
        return None
    if isinstance(cross_entropy, (bool, np.bool_)):
        raise TypeError("cross_entropy must be a finite nonnegative scalar")
    try:
        value = float(cross_entropy)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "cross_entropy must be a finite nonnegative scalar"
        ) from error
    if not math.isfinite(value) or value < 0:
        raise ValueError("cross_entropy must be finite and nonnegative")
    return value


def _classification_percentages(
    target: np.ndarray,
    predicted: np.ndarray,
    *,
    num_tokens: int,
) -> tuple[float, float, float | None, np.ndarray, np.ndarray, list[float | None]]:
    encoded_pairs = target * num_tokens + predicted
    confusion = np.bincount(
        encoded_pairs,
        minlength=num_tokens * num_tokens,
    ).reshape(num_tokens, num_tokens)
    support = confusion.sum(axis=1, dtype=np.int64)
    present = support > 0
    recall = np.full(num_tokens, np.nan, dtype=np.float64)
    recall[present] = (
        np.diag(confusion)[present].astype(np.float64) / support[present]
    )
    accuracy = float(100.0 * np.mean(target == predicted))
    balanced = float(100.0 * np.mean(recall[present]))
    nonzero = target != 0
    nonzero_accuracy = (
        float(100.0 * np.mean(target[nonzero] == predicted[nonzero]))
        if np.any(nonzero)
        else None
    )
    recall_percent: list[float | None] = [
        float(100.0 * value) if math.isfinite(float(value)) else None
        for value in recall
    ]
    return (
        accuracy,
        balanced,
        nonzero_accuracy,
        confusion,
        support,
        recall_percent,
    )


def evaluate_masked_token_predictions(
    target_tokens: np.ndarray,
    predicted_tokens: np.ndarray,
    mask: np.ndarray,
    *,
    cross_entropy: float | None = None,
    per_gene_modal_tokens: np.ndarray | None = None,
    num_tokens: int = _NUM_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Evaluate categorical predictions at selected expression entries.

    Confusion-matrix rows are true tokens and columns are predicted tokens.
    Balanced accuracy is macro recall over target classes represented in the
    selected entries.  Undefined nonzero or absent-class values are returned
    as JSON ``null`` (Python ``None``), never NaN.
    """

    num_tokens = _require_num_tokens(num_tokens)
    target = _validate_output_tokens(
        target_tokens,
        name="target_tokens",
        num_tokens=num_tokens,
    )
    predicted = _validate_output_tokens(
        predicted_tokens,
        name="predicted_tokens",
        num_tokens=num_tokens,
    )
    if predicted.shape != target.shape:
        raise ValueError("target_tokens and predicted_tokens must have the same shape")
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask must be a NumPy boolean array")
    if mask.dtype.kind != "b":
        raise TypeError("mask must have boolean dtype")
    if mask.shape != target.shape:
        raise ValueError(
            "target_tokens, predicted_tokens, and mask must have the same shape"
        )
    flat_mask = mask.ravel()
    n_masked = int(np.count_nonzero(flat_mask))
    if n_masked == 0:
        raise ValueError("mask selects no token predictions")
    selected_target = target.ravel()[flat_mask].astype(np.int64, copy=False)
    selected_predicted = predicted.ravel()[flat_mask].astype(
        np.int64,
        copy=False,
    )

    (
        accuracy,
        balanced,
        nonzero_accuracy,
        confusion,
        support,
        recall_percent,
    ) = _classification_percentages(
        selected_target,
        selected_predicted,
        num_tokens=num_tokens,
    )

    target_probabilities = support.astype(np.float64) / n_masked
    empirical_random_accuracy = float(
        100.0 * np.dot(target_probabilities, target_probabilities)
    )
    always_zero = np.zeros(n_masked, dtype=np.int64)
    always_zero_accuracy, always_zero_balanced, _, _, _, _ = (
        _classification_percentages(
            selected_target,
            always_zero,
            num_tokens=num_tokens,
        )
    )

    modal_accuracy: float | None = None
    modal_balanced: float | None = None
    modal_nonzero: float | None = None
    if per_gene_modal_tokens is not None:
        if not isinstance(per_gene_modal_tokens, np.ndarray):
            raise TypeError("per_gene_modal_tokens must be a NumPy array")
        if per_gene_modal_tokens.ndim != 1:
            raise ValueError("per_gene_modal_tokens must be one-dimensional")
        if per_gene_modal_tokens.shape[0] != target.shape[1]:
            raise ValueError(
                "per_gene_modal_tokens must contain one token per target gene"
            )
        if (
            per_gene_modal_tokens.dtype.kind not in "iu"
            or per_gene_modal_tokens.dtype.kind == "b"
        ):
            raise TypeError("per_gene_modal_tokens must have an integer dtype")
        if (
            int(per_gene_modal_tokens.min()) < 0
            or int(per_gene_modal_tokens.max()) >= num_tokens
        ):
            raise ValueError(
                "per_gene_modal_tokens must contain only output token IDs 0 through 3"
            )
        selected_gene_indices = np.flatnonzero(flat_mask) % target.shape[1]
        modal_predictions = per_gene_modal_tokens[selected_gene_indices].astype(
            np.int64,
            copy=False,
        )
        (
            modal_accuracy,
            modal_balanced,
            modal_nonzero,
            _,
            _,
            _,
        ) = _classification_percentages(
            selected_target,
            modal_predictions,
            num_tokens=num_tokens,
        )

    return {
        "n_masked": n_masked,
        "cross_entropy": _validate_cross_entropy(cross_entropy),
        "accuracy_percent": accuracy,
        "balanced_accuracy_percent": balanced,
        "nonzero_accuracy_percent": nonzero_accuracy,
        "per_token_support": [int(value) for value in support],
        "per_token_recall_percent": recall_percent,
        "confusion_matrix": [
            [int(value) for value in row] for row in confusion
        ],
        "uniform_chance_accuracy_percent": float(100.0 / num_tokens),
        "empirical_frequency_random_accuracy_percent": empirical_random_accuracy,
        "always_zero_accuracy_percent": always_zero_accuracy,
        "always_zero_balanced_accuracy_percent": always_zero_balanced,
        "per_gene_modal_accuracy_percent": modal_accuracy,
        "per_gene_modal_balanced_accuracy_percent": modal_balanced,
        "per_gene_modal_nonzero_accuracy_percent": modal_nonzero,
    }
