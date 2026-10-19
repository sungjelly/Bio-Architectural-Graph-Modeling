"""Deterministic routing and clustering primitives for attention-defined niches.

This module contains the dependency-light numerical core of the locked
six-core post-training analysis.  It never loads a model, touches a checkpoint,
constructs tissue geometry, or assigns biological meaning to a partition.
The directed-edge convention is always ``edge_index[0] = source`` and
``edge_index[1] = receiver``.

Attention is treated as normalized computational routing.  The routines below
make edge identity, receiver-degree adjustment, reciprocal-pair construction,
cross-seed/mask aggregation, and deterministic graph retention explicit so a
caller cannot silently turn a one-directional edge into a mutual relation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .adjacency_ablation import (
    MaskRealization,
    ndarray_sha256,
    sample_uniform_mask_numpy,
)
from .masking import derive_mask_seed


CORE_ALIASES = ("CAN-01", "CAN-09", "CAN-13", "CAN-15", "CAN-21", "CAN-23")
ANALYSIS_MASK_BASE_SEED = 2026082501
ANALYSIS_MASK_VIEW_COUNT = 10
LEIDEN_SEED = 2026082502
DEFAULT_MUTUAL_SCORE_THRESHOLD = 1.0
DEFAULT_SUPPORT_THRESHOLD = 0.60
DEFAULT_TOP_K = 8
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_PAIR_CHUNK_SIZE = 131_072
SENSITIVITY_TOP_K = (5, 8, 10)
SENSITIVITY_RESOLUTIONS = (0.5, 1.0, 1.5)


class AttentionRoutingNicheError(ValueError):
    """Raised when an attention-routing analysis contract is violated."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _readonly_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype)).copy()
    array.setflags(write=False)
    return array


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise AttentionRoutingNicheError(f"{name} must be a positive integer.")
    return int(value)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise AttentionRoutingNicheError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "" if minimum is None else f" and at least {minimum:g}"
        raise AttentionRoutingNicheError(f"{name} must be finite{suffix}.")
    return result


def _canonical_core_alias(core_alias: object) -> str:
    alias = str(core_alias).strip().upper()
    if alias not in CORE_ALIASES:
        raise AttentionRoutingNicheError(
            "core_alias must be one of the six locked cancer-core aliases."
        )
    return alias


def _validate_edge_index(edge_index: Any, *, n_nodes: int) -> np.ndarray:
    nodes = _positive_integer(n_nodes, name="n_nodes")
    raw = np.asarray(edge_index)
    if (
        raw.ndim != 2
        or raw.shape[0] != 2
        or raw.dtype.kind not in "iu"
        or raw.dtype == np.dtype(np.bool_)
    ):
        raise AttentionRoutingNicheError(
            "edge_index must be an integral array with shape [2, directed_edges]."
        )
    edges = np.ascontiguousarray(raw, dtype=np.int64)
    if edges.size and (np.any(edges < 0) or np.any(edges >= nodes)):
        raise AttentionRoutingNicheError("edge_index contains an out-of-range node.")
    if edges.shape[1] and np.any(edges[0] == edges[1]):
        raise AttentionRoutingNicheError(
            "Self edges are prohibited in the attention-routing graph."
        )
    if edges.shape[1]:
        codes = edges[0] * nodes + edges[1]
        if len(np.unique(codes)) != len(codes):
            raise AttentionRoutingNicheError(
                "edge_index contains a duplicate directed edge."
            )
    return edges


def _finite_array(
    value: Any,
    *,
    name: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.dtype == np.dtype(np.bool_):
        raise AttentionRoutingNicheError(f"{name} must be numeric.")
    array = np.asarray(raw, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise AttentionRoutingNicheError(f"{name} must have {ndim} dimensions.")
    if shape is not None and array.shape != shape:
        raise AttentionRoutingNicheError(
            f"{name} must have shape {shape}; received {array.shape}."
        )
    if not np.isfinite(array).all():
        raise AttentionRoutingNicheError(f"{name} contains a non-finite value.")
    return np.ascontiguousarray(array)


def _finite_floating_array_preserve_precision(
    value: Any,
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    """Validate a floating array without promoting a large float32 payload."""

    raw = np.asarray(value)
    if raw.dtype.kind != "f" or raw.ndim != ndim:
        raise AttentionRoutingNicheError(
            f"{name} must be a floating array with {ndim} dimensions."
        )
    # Promote half precision because NumPy reductions over it are unnecessarily
    # fragile, but retain float32 inputs from the model to avoid doubling a
    # seed-by-view-by-edge analysis tensor in host memory.
    dtype = np.float64 if raw.dtype.itemsize > 4 else np.float32
    array = np.ascontiguousarray(raw, dtype=dtype)
    if not np.isfinite(array).all():
        raise AttentionRoutingNicheError(f"{name} contains a non-finite value.")
    return array


def _readonly_contiguous_view(
    value: Any,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    """Return a read-only contiguous view without duplicating owned results."""

    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    view = array.view()
    view.setflags(write=False)
    return view


# ---------------------------------------------------------------------------
# Exact deterministic analysis masks
# ---------------------------------------------------------------------------


def derive_analysis_mask_seed(
    analysis_mask_seed: int,
    core_alias: str,
    mask_view_index: int,
) -> int:
    """Derive one mask seed from only base seed, core alias, and view index.

    The deliberately narrow signature has no model-seed argument.  This makes
    it impossible for callers to accidentally use different masks for
    different trained-model seeds.
    """

    base_seed = _nonnegative_integer(analysis_mask_seed, name="analysis_mask_seed")
    alias = _canonical_core_alias(core_alias)
    view_index = _nonnegative_integer(mask_view_index, name="mask_view_index")
    if view_index >= ANALYSIS_MASK_VIEW_COUNT:
        raise AttentionRoutingNicheError(
            f"mask_view_index must be in 0..{ANALYSIS_MASK_VIEW_COUNT - 1}."
        )
    return derive_mask_seed(
        base_seed,
        "attention-routing-analysis-mask",
        alias,
        view_index,
    )


@dataclass(frozen=True)
class AnalysisMaskView:
    """One exact Uniform{0,...,G} per-cell analysis mask and its receipt."""

    core_alias: str
    mask_view_index: int
    analysis_mask_seed: int
    realization: MaskRealization = field(repr=False)
    receipt_sha256: str

    def __post_init__(self) -> None:
        alias = _canonical_core_alias(self.core_alias)
        view_index = _nonnegative_integer(
            self.mask_view_index, name="mask_view_index"
        )
        if view_index >= ANALYSIS_MASK_VIEW_COUNT:
            raise AttentionRoutingNicheError(
                f"mask_view_index must be in 0..{ANALYSIS_MASK_VIEW_COUNT - 1}."
            )
        base_seed = _nonnegative_integer(
            self.analysis_mask_seed, name="analysis_mask_seed"
        )
        if not isinstance(self.realization, MaskRealization):
            raise AttentionRoutingNicheError(
                "realization must use the repository's audited MaskRealization."
            )
        expected_seed = derive_analysis_mask_seed(base_seed, alias, view_index)
        if int(self.realization.seed) != expected_seed:
            raise AttentionRoutingNicheError(
                "Mask realization seed does not match the locked derivation."
            )
        expected_receipt = _payload_sha256(self._receipt_payload())
        if str(self.receipt_sha256) != expected_receipt:
            raise AttentionRoutingNicheError("Analysis-mask receipt checksum mismatch.")
        object.__setattr__(self, "core_alias", alias)
        object.__setattr__(self, "mask_view_index", view_index)
        object.__setattr__(self, "analysis_mask_seed", base_seed)

    @property
    def mask(self) -> np.ndarray:
        return self.realization.mask

    @property
    def masked_gene_counts(self) -> np.ndarray:
        return self.realization.masked_gene_counts

    @property
    def derived_seed(self) -> int:
        return int(self.realization.seed)

    def _receipt_payload(self) -> dict[str, object]:
        counts = self.realization.masked_gene_counts
        return {
            "schema": "attention_routing_analysis_mask_v1",
            "core_alias": str(self.core_alias).strip().upper(),
            "mask_view_index": int(self.mask_view_index),
            "analysis_mask_seed": int(self.analysis_mask_seed),
            "derived_seed": int(self.realization.seed),
            "seed_derivation": (
                "analysis_mask_seed+core_alias+mask_view_index;model_seed_excluded"
            ),
            "sampling": (
                "per-cell exact count Uniform{0,...,G}; positions uniform "
                "without replacement"
            ),
            "n_cells": int(self.realization.n_cells),
            "n_genes": int(self.realization.num_genes),
            "masked_entry_count": int(counts.sum()),
            "zero_mask_rows": int(np.count_nonzero(counts == 0)),
            "full_mask_rows": int(
                np.count_nonzero(counts == self.realization.num_genes)
            ),
            "masked_gene_counts_sha256": ndarray_sha256(counts),
            "mask_realization_sha256": self.realization.checksum,
        }

    def to_receipt(self) -> dict[str, object]:
        receipt = self._receipt_payload()
        receipt["receipt_sha256"] = self.receipt_sha256
        return receipt


def make_analysis_mask_view(
    n_cells: int,
    n_genes: int,
    *,
    core_alias: str,
    mask_view_index: int,
    analysis_mask_seed: int = ANALYSIS_MASK_BASE_SEED,
    chunk_cells: int = 2048,
) -> AnalysisMaskView:
    """Create one newly sampled, exact, model-seed-independent analysis mask."""

    n_cells = _positive_integer(n_cells, name="n_cells")
    n_genes = _positive_integer(n_genes, name="n_genes")
    chunk_cells = _positive_integer(chunk_cells, name="chunk_cells")
    alias = _canonical_core_alias(core_alias)
    view_index = _nonnegative_integer(mask_view_index, name="mask_view_index")
    derived_seed = derive_analysis_mask_seed(
        analysis_mask_seed,
        alias,
        view_index,
    )
    realization = sample_uniform_mask_numpy(
        n_cells,
        n_genes,
        seed=derived_seed,
        chunk_cells=chunk_cells,
    )
    provisional = object.__new__(AnalysisMaskView)
    object.__setattr__(provisional, "core_alias", alias)
    object.__setattr__(provisional, "mask_view_index", view_index)
    object.__setattr__(provisional, "analysis_mask_seed", int(analysis_mask_seed))
    object.__setattr__(provisional, "realization", realization)
    receipt_sha256 = _payload_sha256(provisional._receipt_payload())
    return AnalysisMaskView(
        core_alias=alias,
        mask_view_index=view_index,
        analysis_mask_seed=int(analysis_mask_seed),
        realization=realization,
        receipt_sha256=receipt_sha256,
    )


def make_analysis_mask_views(
    n_cells: int,
    n_genes: int,
    *,
    core_alias: str,
    analysis_mask_seed: int = ANALYSIS_MASK_BASE_SEED,
    chunk_cells: int = 2048,
) -> tuple[AnalysisMaskView, ...]:
    """Create exactly the ten locked analysis views in index order."""

    return tuple(
        make_analysis_mask_view(
            n_cells,
            n_genes,
            core_alias=core_alias,
            mask_view_index=view_index,
            analysis_mask_seed=analysis_mask_seed,
            chunk_cells=chunk_cells,
        )
        for view_index in range(ANALYSIS_MASK_VIEW_COUNT)
    )


# ---------------------------------------------------------------------------
# Edge-aligned attention validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionNormalizationAudit:
    """Per-receiver/head normalization receipt for a complete edge shard."""

    receiver_indices: np.ndarray = field(repr=False)
    observed_in_degrees: np.ndarray = field(repr=False)
    per_head_sums: np.ndarray = field(repr=False)
    maximum_absolute_deviation: float
    tolerance: float
    checksum_sha256: str

    def __post_init__(self) -> None:
        receivers = _readonly_array(self.receiver_indices, dtype=np.int64)
        degrees = _readonly_array(self.observed_in_degrees, dtype=np.int64)
        sums = _readonly_array(self.per_head_sums, dtype=np.float64)
        if (
            receivers.ndim != 1
            or degrees.shape != receivers.shape
            or sums.ndim != 2
            or sums.shape[0] != len(receivers)
            or np.any(degrees <= 0)
            or (len(receivers) > 1 and np.any(receivers[1:] <= receivers[:-1]))
        ):
            raise AttentionRoutingNicheError(
                "Attention normalization receipt arrays are not aligned."
            )
        tolerance = _finite_float(self.tolerance, name="tolerance", minimum=0.0)
        deviation = _finite_float(
            self.maximum_absolute_deviation,
            name="maximum_absolute_deviation",
            minimum=0.0,
        )
        if sums.shape[1] == 0 or not np.isfinite(sums).all():
            raise AttentionRoutingNicheError(
                "Attention normalization sums must contain finite heads."
            )
        expected = float(np.max(np.abs(sums - 1.0), initial=0.0))
        if not math.isclose(deviation, expected, rel_tol=0.0, abs_tol=1e-15):
            raise AttentionRoutingNicheError(
                "Attention normalization maximum deviation is inconsistent."
            )
        payload = {
            "schema": "attention_receiver_normalization_audit_v1",
            "receiver_indices_sha256": ndarray_sha256(receivers),
            "observed_in_degrees_sha256": ndarray_sha256(degrees),
            "per_head_sums_sha256": ndarray_sha256(sums),
            "maximum_absolute_deviation": deviation,
            "tolerance": tolerance,
        }
        if self.checksum_sha256 != _payload_sha256(payload):
            raise AttentionRoutingNicheError(
                "Attention normalization audit checksum mismatch."
            )
        object.__setattr__(self, "receiver_indices", receivers)
        object.__setattr__(self, "observed_in_degrees", degrees)
        object.__setattr__(self, "per_head_sums", sums)
        object.__setattr__(self, "maximum_absolute_deviation", deviation)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def passed(self) -> bool:
        return self.maximum_absolute_deviation <= self.tolerance


def audit_attention_normalization(
    edge_index: Any,
    attention_weights: Any,
    *,
    n_nodes: int,
    expected_receiver_in_degrees: Any | None = None,
    tolerance: float = 1.0e-6,
    raise_on_failure: bool = True,
) -> AttentionNormalizationAudit:
    """Audit that every included receiver/head has total incoming weight one.

    When ``expected_receiver_in_degrees`` is supplied, every receiver present
    in the shard must include its entire global incoming neighborhood.
    """

    edges = _validate_edge_index(edge_index, n_nodes=n_nodes)
    edge_count = int(edges.shape[1])
    attention = _finite_array(attention_weights, name="attention_weights", ndim=2)
    if attention.shape[0] != edge_count or attention.shape[1] == 0:
        raise AttentionRoutingNicheError(
            "attention_weights must be edge-aligned with at least one head."
        )
    tolerance = _finite_float(tolerance, name="tolerance", minimum=0.0)
    if np.any(attention < -tolerance) or np.any(attention > 1.0 + tolerance):
        raise AttentionRoutingNicheError(
            "attention_weights contain values outside the probability range."
        )
    if edge_count == 0:
        raise AttentionRoutingNicheError(
            "An attention shard must contain at least one complete receiver."
        )
    receivers, inverse = np.unique(edges[1], return_inverse=True)
    observed_degrees = np.bincount(inverse, minlength=len(receivers)).astype(
        np.int64, copy=False
    )
    if expected_receiver_in_degrees is not None:
        raw_degrees = np.asarray(expected_receiver_in_degrees)
        if (
            raw_degrees.shape != (int(n_nodes),)
            or raw_degrees.dtype.kind not in "iu"
            or raw_degrees.dtype == np.dtype(np.bool_)
        ):
            raise AttentionRoutingNicheError(
                "expected_receiver_in_degrees must be integral [n_nodes]."
            )
        expected = np.asarray(raw_degrees, dtype=np.int64)
        if np.any(expected < 0):
            raise AttentionRoutingNicheError(
                "expected_receiver_in_degrees cannot be negative."
            )
        if not np.array_equal(observed_degrees, expected[receivers]):
            raise AttentionRoutingNicheError(
                "Attention shard truncates at least one receiver neighborhood."
            )
    sums = np.zeros((len(receivers), attention.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, attention)
    maximum_deviation = float(np.max(np.abs(sums - 1.0), initial=0.0))
    payload = {
        "schema": "attention_receiver_normalization_audit_v1",
        "receiver_indices_sha256": ndarray_sha256(receivers.astype(np.int64)),
        "observed_in_degrees_sha256": ndarray_sha256(observed_degrees),
        "per_head_sums_sha256": ndarray_sha256(sums),
        "maximum_absolute_deviation": maximum_deviation,
        "tolerance": tolerance,
    }
    result = AttentionNormalizationAudit(
        receiver_indices=receivers,
        observed_in_degrees=observed_degrees,
        per_head_sums=sums,
        maximum_absolute_deviation=maximum_deviation,
        tolerance=tolerance,
        checksum_sha256=_payload_sha256(payload),
    )
    if raise_on_failure and not result.passed:
        raise AttentionRoutingNicheError(
            "Incoming attention does not sum to one for every receiver and head; "
            f"maximum deviation is {maximum_deviation:.3g}."
        )
    return result


@dataclass(frozen=True)
class AttentionShardAudit:
    """Successful edge-alignment, logit, softmax, and normalization audit."""

    edge_count: int
    head_count: int
    edge_index_sha256: str
    attention_weights_sha256: str
    content_logits_sha256: str
    positional_bias_sha256: str
    combined_logits_sha256: str
    maximum_logit_decomposition_deviation: float
    maximum_softmax_deviation: float
    normalization: AttentionNormalizationAudit
    receipt_sha256: str

    def __post_init__(self) -> None:
        edge_count = _positive_integer(self.edge_count, name="edge_count")
        head_count = _positive_integer(self.head_count, name="head_count")
        if not isinstance(self.normalization, AttentionNormalizationAudit):
            raise AttentionRoutingNicheError(
                "normalization must be an AttentionNormalizationAudit."
            )
        for name in (
            "edge_index_sha256",
            "attention_weights_sha256",
            "content_logits_sha256",
            "positional_bias_sha256",
            "combined_logits_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise AttentionRoutingNicheError(f"{name} is not a SHA-256 digest.")
        logit_deviation = _finite_float(
            self.maximum_logit_decomposition_deviation,
            name="maximum_logit_decomposition_deviation",
            minimum=0.0,
        )
        softmax_deviation = _finite_float(
            self.maximum_softmax_deviation,
            name="maximum_softmax_deviation",
            minimum=0.0,
        )
        payload = {
            "schema": "attention_shard_audit_v1",
            "edge_count": edge_count,
            "head_count": head_count,
            "edge_index_sha256": self.edge_index_sha256,
            "attention_weights_sha256": self.attention_weights_sha256,
            "content_logits_sha256": self.content_logits_sha256,
            "positional_bias_sha256": self.positional_bias_sha256,
            "combined_logits_sha256": self.combined_logits_sha256,
            "maximum_logit_decomposition_deviation": logit_deviation,
            "maximum_softmax_deviation": softmax_deviation,
            "normalization_checksum_sha256": self.normalization.checksum_sha256,
        }
        if self.receipt_sha256 != _payload_sha256(payload):
            raise AttentionRoutingNicheError("Attention-shard receipt checksum mismatch.")
        object.__setattr__(self, "edge_count", edge_count)
        object.__setattr__(self, "head_count", head_count)


def validate_attention_shard(
    edge_index: Any,
    attention_weights: Any,
    content_logits: Any,
    positional_bias: Any,
    combined_logits: Any,
    *,
    n_nodes: int,
    source_indices: Any | None = None,
    receiver_indices: Any | None = None,
    expected_edge_index: Any | None = None,
    expected_edge_ids: Any | None = None,
    expected_receiver_in_degrees: Any | None = None,
    normalization_tolerance: float = 1.0e-6,
    decomposition_atol: float = 1.0e-6,
    decomposition_rtol: float = 1.0e-6,
    softmax_atol: float = 1.0e-6,
    softmax_rtol: float = 1.0e-5,
) -> AttentionShardAudit:
    """Fail closed unless one exact receiver-complete diagnostic shard aligns.

    Besides checking ``combined = content + bias``, this reconstructs the
    receiver-wise softmax from the combined logits.  Consequently a row
    permutation in any diagnostic tensor cannot pass merely because all shapes
    happen to agree.
    """

    edges = _validate_edge_index(edge_index, n_nodes=n_nodes)
    edge_count = int(edges.shape[1])
    if edge_count == 0:
        raise AttentionRoutingNicheError("Attention shards cannot be empty.")
    arrays: dict[str, np.ndarray] = {}
    for name, value in (
        ("attention_weights", attention_weights),
        ("content_logits", content_logits),
        ("positional_bias", positional_bias),
        ("combined_logits", combined_logits),
    ):
        array = _finite_array(value, name=name, ndim=2)
        if array.shape[0] != edge_count or array.shape[1] == 0:
            raise AttentionRoutingNicheError(
                f"{name} must be [directed_edges, attention_heads]."
            )
        arrays[name] = array
    shape = arrays["attention_weights"].shape
    if any(array.shape != shape for array in arrays.values()):
        raise AttentionRoutingNicheError(
            "Attention, content, bias, and combined logits are not aligned."
        )
    if source_indices is not None:
        sources = np.asarray(source_indices)
        if sources.shape != (edge_count,) or sources.dtype.kind not in "iu":
            raise AttentionRoutingNicheError(
                "source_indices must be integral and edge aligned."
            )
        if not np.array_equal(np.asarray(sources, dtype=np.int64), edges[0]):
            raise AttentionRoutingNicheError(
                "source_indices do not align with edge_index[0]."
            )
    if receiver_indices is not None:
        receivers = np.asarray(receiver_indices)
        if receivers.shape != (edge_count,) or receivers.dtype.kind not in "iu":
            raise AttentionRoutingNicheError(
                "receiver_indices must be integral and edge aligned."
            )
        if not np.array_equal(np.asarray(receivers, dtype=np.int64), edges[1]):
            raise AttentionRoutingNicheError(
                "receiver_indices do not align with edge_index[1]."
            )
    if expected_edge_index is not None:
        expected_edges = _validate_edge_index(expected_edge_index, n_nodes=n_nodes)
        if expected_edge_ids is None:
            expected_shard = expected_edges
        else:
            raw_ids = np.asarray(expected_edge_ids)
            if (
                raw_ids.shape != (edge_count,)
                or raw_ids.dtype.kind not in "iu"
                or raw_ids.dtype == np.dtype(np.bool_)
            ):
                raise AttentionRoutingNicheError(
                    "expected_edge_ids must be integral and edge aligned."
                )
            edge_ids = np.asarray(raw_ids, dtype=np.int64)
            if (
                np.any(edge_ids < 0)
                or np.any(edge_ids >= expected_edges.shape[1])
                or len(np.unique(edge_ids)) != edge_count
            ):
                raise AttentionRoutingNicheError(
                    "expected_edge_ids are duplicated or out of range."
                )
            expected_shard = expected_edges[:, edge_ids]
        if not np.array_equal(edges, expected_shard):
            raise AttentionRoutingNicheError(
                "Extracted edge_index does not match the expected edge identities."
            )
        full_degrees = np.bincount(
            expected_edges[1], minlength=int(n_nodes)
        ).astype(np.int64, copy=False)
        if expected_receiver_in_degrees is None:
            expected_receiver_in_degrees = full_degrees
        elif not np.array_equal(
            np.asarray(expected_receiver_in_degrees, dtype=np.int64), full_degrees
        ):
            raise AttentionRoutingNicheError(
                "Expected receiver degrees disagree with expected_edge_index."
            )
    elif expected_edge_ids is not None:
        raise AttentionRoutingNicheError(
            "expected_edge_ids require expected_edge_index."
        )

    decomposition_atol = _finite_float(
        decomposition_atol, name="decomposition_atol", minimum=0.0
    )
    decomposition_rtol = _finite_float(
        decomposition_rtol, name="decomposition_rtol", minimum=0.0
    )
    expected_combined = arrays["content_logits"] + arrays["positional_bias"]
    logit_residual = np.abs(arrays["combined_logits"] - expected_combined)
    maximum_logit_deviation = float(logit_residual.max(initial=0.0))
    if not np.allclose(
        arrays["combined_logits"],
        expected_combined,
        atol=decomposition_atol,
        rtol=decomposition_rtol,
    ):
        raise AttentionRoutingNicheError(
            "combined_logits do not equal content_logits + positional_bias."
        )
    normalization = audit_attention_normalization(
        edges,
        arrays["attention_weights"],
        n_nodes=n_nodes,
        expected_receiver_in_degrees=expected_receiver_in_degrees,
        tolerance=normalization_tolerance,
        raise_on_failure=True,
    )
    reconstructed = np.empty_like(arrays["combined_logits"])
    receiver_values = edges[1]
    if len(receiver_values) <= 1 or np.all(
        receiver_values[1:] >= receiver_values[:-1]
    ):
        order = np.arange(edge_count, dtype=np.int64)
        ordered_receivers = receiver_values
    else:
        order = np.argsort(receiver_values, kind="stable")
        ordered_receivers = receiver_values[order]
    starts = np.r_[
        0,
        np.flatnonzero(ordered_receivers[1:] != ordered_receivers[:-1]) + 1,
    ]
    stops = np.r_[starts[1:], edge_count]
    combined = arrays["combined_logits"]
    for start, stop in zip(starts, stops, strict=True):
        ids = order[start:stop]
        receiver_logits = combined[ids]
        shifted = receiver_logits - receiver_logits.max(axis=0, keepdims=True)
        exponentials = np.exp(shifted)
        reconstructed[ids] = exponentials / exponentials.sum(
            axis=0, keepdims=True
        )
    softmax_atol = _finite_float(softmax_atol, name="softmax_atol", minimum=0.0)
    softmax_rtol = _finite_float(softmax_rtol, name="softmax_rtol", minimum=0.0)
    softmax_residual = np.abs(arrays["attention_weights"] - reconstructed)
    maximum_softmax_deviation = float(softmax_residual.max(initial=0.0))
    if not np.allclose(
        arrays["attention_weights"],
        reconstructed,
        atol=softmax_atol,
        rtol=softmax_rtol,
    ):
        raise AttentionRoutingNicheError(
            "attention_weights are not the receiver-wise softmax of combined_logits."
        )
    payload = {
        "schema": "attention_shard_audit_v1",
        "edge_count": edge_count,
        "head_count": int(shape[1]),
        "edge_index_sha256": ndarray_sha256(edges),
        "attention_weights_sha256": ndarray_sha256(arrays["attention_weights"]),
        "content_logits_sha256": ndarray_sha256(arrays["content_logits"]),
        "positional_bias_sha256": ndarray_sha256(arrays["positional_bias"]),
        "combined_logits_sha256": ndarray_sha256(arrays["combined_logits"]),
        "maximum_logit_decomposition_deviation": maximum_logit_deviation,
        "maximum_softmax_deviation": maximum_softmax_deviation,
        "normalization_checksum_sha256": normalization.checksum_sha256,
    }
    return AttentionShardAudit(
        edge_count=edge_count,
        head_count=int(shape[1]),
        edge_index_sha256=str(payload["edge_index_sha256"]),
        attention_weights_sha256=str(payload["attention_weights_sha256"]),
        content_logits_sha256=str(payload["content_logits_sha256"]),
        positional_bias_sha256=str(payload["positional_bias_sha256"]),
        combined_logits_sha256=str(payload["combined_logits_sha256"]),
        maximum_logit_decomposition_deviation=maximum_logit_deviation,
        maximum_softmax_deviation=maximum_softmax_deviation,
        normalization=normalization,
        receipt_sha256=_payload_sha256(payload),
    )


# ---------------------------------------------------------------------------
# Directed routing and canonical reciprocal pairs
# ---------------------------------------------------------------------------


def receiver_in_degrees(edge_index: Any, *, n_nodes: int) -> np.ndarray:
    """Return the complete directed in-degree vector using receiver row one."""

    edges = _validate_edge_index(edge_index, n_nodes=n_nodes)
    result = np.bincount(edges[1], minlength=int(n_nodes)).astype(
        np.int64, copy=False
    )
    return _readonly_array(result)


@dataclass(frozen=True)
class DirectionalRouting:
    """Head-mean attention and receiver-degree-adjusted routing by edge."""

    receiver_in_degree: np.ndarray = field(repr=False)
    head_mean_attention: np.ndarray = field(repr=False)
    enrichment: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        degree = _readonly_array(self.receiver_in_degree, dtype=np.int64)
        mean_attention = _finite_floating_array_preserve_precision(
            self.head_mean_attention,
            name="head_mean_attention",
            ndim=1,
        )
        enrichment = _finite_floating_array_preserve_precision(
            self.enrichment,
            name="enrichment",
            ndim=1,
        )
        mean_attention = _readonly_contiguous_view(mean_attention)
        enrichment = _readonly_contiguous_view(enrichment)
        if (
            degree.ndim != 1
            or mean_attention.shape != degree.shape
            or enrichment.shape != degree.shape
            or np.any(degree <= 0)
            or not np.isfinite(mean_attention).all()
            or not np.isfinite(enrichment).all()
            or np.any(mean_attention < 0.0)
            or np.any(enrichment < 0.0)
        ):
            raise AttentionRoutingNicheError(
                "Directional routing arrays must be finite, non-negative, and aligned."
            )
        calculation_tolerance = (
            1.0e-6 if mean_attention.dtype.itemsize <= 4 else 1.0e-12
        )
        if not np.allclose(
            enrichment,
            degree.astype(np.float64) * mean_attention,
            atol=calculation_tolerance,
            rtol=calculation_tolerance,
        ):
            raise AttentionRoutingNicheError(
                "Directional enrichment does not use receiver in-degree."
            )
        object.__setattr__(self, "receiver_in_degree", degree)
        object.__setattr__(self, "head_mean_attention", mean_attention)
        object.__setattr__(self, "enrichment", enrichment)


def degree_adjusted_routing(
    edge_index: Any,
    attention_weights: Any,
    *,
    n_nodes: int,
    verify_normalization: bool = True,
    normalization_tolerance: float = 1.0e-6,
) -> DirectionalRouting:
    """Calculate ``receiver_in_degree * mean_head_attention`` per edge."""

    edges = _validate_edge_index(edge_index, n_nodes=n_nodes)
    attention = _finite_floating_array_preserve_precision(
        attention_weights,
        name="attention_weights",
        ndim=np.asarray(attention_weights).ndim,
    )
    if attention.ndim == 1:
        if attention.shape != (edges.shape[1],):
            raise AttentionRoutingNicheError(
                "One-dimensional attention must align to directed edges."
            )
        attention_2d = attention[:, None]
    elif attention.ndim == 2 and attention.shape[0] == edges.shape[1] and attention.shape[1]:
        attention_2d = attention
    else:
        raise AttentionRoutingNicheError(
            "attention_weights must be [edges] or [edges, heads]."
        )
    if verify_normalization:
        audit_attention_normalization(
            edges,
            attention_2d,
            n_nodes=n_nodes,
            tolerance=normalization_tolerance,
            raise_on_failure=True,
        )
    degrees_by_node = receiver_in_degrees(edges, n_nodes=n_nodes)
    degrees_by_edge = np.asarray(degrees_by_node[edges[1]], dtype=np.int64)
    result_dtype = np.float64 if attention_2d.dtype.itemsize > 4 else np.float32
    mean_attention = attention_2d.mean(axis=1, dtype=result_dtype)
    enrichment = degrees_by_edge.astype(result_dtype) * mean_attention
    return DirectionalRouting(
        receiver_in_degree=degrees_by_edge,
        head_mean_attention=mean_attention,
        enrichment=enrichment,
    )


def degree_adjusted_routing_samples(
    edge_index: Any,
    attention_weight_samples: Any,
    *,
    n_nodes: int,
    verify_normalization: bool = True,
    normalization_tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Vectorize degree adjustment for ``[seed, view, edge, head]`` attention."""

    edges = _validate_edge_index(edge_index, n_nodes=n_nodes)
    attention = _finite_floating_array_preserve_precision(
        attention_weight_samples,
        name="attention_weight_samples",
        ndim=4,
    )
    if attention.shape[2] != edges.shape[1] or attention.shape[3] == 0:
        raise AttentionRoutingNicheError(
            "attention_weight_samples must be [seed, view, edge, head]."
        )
    if attention.shape[0] == 0 or attention.shape[1] == 0:
        raise AttentionRoutingNicheError("Seed and mask-view axes cannot be empty.")
    if verify_normalization:
        for seed_index in range(attention.shape[0]):
            for view_index in range(attention.shape[1]):
                audit_attention_normalization(
                    edges,
                    attention[seed_index, view_index],
                    n_nodes=n_nodes,
                    tolerance=normalization_tolerance,
                    raise_on_failure=True,
                )
    result_dtype = np.float64 if attention.dtype.itemsize > 4 else np.float32
    degrees = receiver_in_degrees(edges, n_nodes=n_nodes)[edges[1]].astype(
        result_dtype
    )
    result = attention.mean(axis=3, dtype=result_dtype) * degrees[None, None, :]
    return _readonly_contiguous_view(result)


@dataclass(frozen=True)
class ReciprocalPairIndex:
    """Canonical ``i < j`` pairs and exact directed-edge identities."""

    n_nodes: int
    pair_cells: np.ndarray = field(repr=False)
    i_to_j_edge_ids: np.ndarray = field(repr=False)
    j_to_i_edge_ids: np.ndarray = field(repr=False)
    one_way_edge_ids: np.ndarray = field(repr=False)
    reverse_edge_ids: np.ndarray = field(repr=False)
    directed_pair_positions: np.ndarray = field(repr=False)
    receipt_sha256: str

    def __post_init__(self) -> None:
        n_nodes = _positive_integer(self.n_nodes, name="n_nodes")
        pairs = _readonly_array(self.pair_cells, dtype=np.int64)
        forward = _readonly_array(self.i_to_j_edge_ids, dtype=np.int64)
        reverse = _readonly_array(self.j_to_i_edge_ids, dtype=np.int64)
        one_way = _readonly_array(self.one_way_edge_ids, dtype=np.int64)
        reverse_ids = _readonly_array(self.reverse_edge_ids, dtype=np.int64)
        positions = _readonly_array(self.directed_pair_positions, dtype=np.int64)
        pair_count = len(pairs)
        if (
            pairs.shape != (pair_count, 2)
            or forward.shape != (pair_count,)
            or reverse.shape != (pair_count,)
            or one_way.ndim != 1
            or reverse_ids.ndim != 1
            or positions.shape != reverse_ids.shape
            or np.any(pairs < 0)
            or np.any(pairs >= n_nodes)
            or np.any(pairs[:, 0] >= pairs[:, 1])
        ):
            raise AttentionRoutingNicheError(
                "Reciprocal-pair index arrays are malformed."
            )
        if pair_count > 1:
            pair_codes = pairs[:, 0] * n_nodes + pairs[:, 1]
            if np.any(pair_codes[1:] <= pair_codes[:-1]):
                raise AttentionRoutingNicheError(
                    "Canonical reciprocal pairs must be unique and sorted."
                )
        edge_count = len(reverse_ids)
        if (
            np.any(forward < 0)
            or np.any(reverse < 0)
            or np.any(forward >= edge_count)
            or np.any(reverse >= edge_count)
            or np.any(one_way < 0)
            or np.any(one_way >= edge_count)
        ):
            raise AttentionRoutingNicheError(
                "Reciprocal-pair edge identities are out of range."
            )
        mutual_ids = np.concatenate((forward, reverse))
        if len(np.unique(mutual_ids)) != len(mutual_ids):
            raise AttentionRoutingNicheError(
                "A directed edge was assigned to more than one reciprocal pair."
            )
        if len(np.unique(one_way)) != len(one_way) or np.intersect1d(
            mutual_ids, one_way
        ).size:
            raise AttentionRoutingNicheError(
                "One-way and reciprocal edge identities overlap."
            )
        if len(mutual_ids) + len(one_way) != edge_count:
            raise AttentionRoutingNicheError(
                "Every directed edge must be reciprocal or explicitly one-way."
            )
        if pair_count:
            if not np.array_equal(reverse_ids[forward], reverse) or not np.array_equal(
                reverse_ids[reverse], forward
            ):
                raise AttentionRoutingNicheError(
                    "Reverse-edge identities are not involutive."
                )
            if not np.array_equal(positions[forward], np.arange(pair_count)) or not np.array_equal(
                positions[reverse], np.arange(pair_count)
            ):
                raise AttentionRoutingNicheError(
                    "Directed edge to pair-position mapping is inconsistent."
                )
        if one_way.size and (
            np.any(reverse_ids[one_way] != -1) or np.any(positions[one_way] != -1)
        ):
            raise AttentionRoutingNicheError(
                "One-way edges must not have reverse or mutual-pair identities."
            )
        payload = {
            "schema": "attention_reciprocal_pair_index_v1",
            "n_nodes": n_nodes,
            "pair_cells_sha256": ndarray_sha256(pairs),
            "i_to_j_edge_ids_sha256": ndarray_sha256(forward),
            "j_to_i_edge_ids_sha256": ndarray_sha256(reverse),
            "one_way_edge_ids_sha256": ndarray_sha256(one_way),
            "reverse_edge_ids_sha256": ndarray_sha256(reverse_ids),
            "directed_pair_positions_sha256": ndarray_sha256(positions),
        }
        if self.receipt_sha256 != _payload_sha256(payload):
            raise AttentionRoutingNicheError(
                "Reciprocal-pair index receipt checksum mismatch."
            )
        object.__setattr__(self, "n_nodes", n_nodes)
        object.__setattr__(self, "pair_cells", pairs)
        object.__setattr__(self, "i_to_j_edge_ids", forward)
        object.__setattr__(self, "j_to_i_edge_ids", reverse)
        object.__setattr__(self, "one_way_edge_ids", one_way)
        object.__setattr__(self, "reverse_edge_ids", reverse_ids)
        object.__setattr__(self, "directed_pair_positions", positions)

    @property
    def pair_count(self) -> int:
        return int(self.pair_cells.shape[0])

    @property
    def edge_count(self) -> int:
        return int(self.reverse_edge_ids.shape[0])


def build_reciprocal_pair_index(
    edge_index: Any,
    *,
    n_nodes: int,
    require_all_edges_reciprocal: bool = False,
) -> ReciprocalPairIndex:
    """Build canonical pairs while excluding every one-directional edge.

    Set ``require_all_edges_reciprocal=True`` for the production six-core QC.
    With the default, one-way edges are retained only in ``one_way_edge_ids``
    and can never contribute a mutual score.
    """

    edges = _validate_edge_index(edge_index, n_nodes=n_nodes)
    edge_count = int(edges.shape[1])
    codes = edges[0] * int(n_nodes) + edges[1]
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    reverse_codes = edges[1] * int(n_nodes) + edges[0]
    positions = np.searchsorted(sorted_codes, reverse_codes)
    reciprocal = positions < edge_count
    if edge_count:
        safe_positions = np.minimum(positions, edge_count - 1)
        reciprocal &= sorted_codes[safe_positions] == reverse_codes
    reverse_edge_ids = np.full(edge_count, -1, dtype=np.int64)
    reverse_edge_ids[reciprocal] = order[positions[reciprocal]]
    one_way = np.flatnonzero(~reciprocal).astype(np.int64, copy=False)
    if require_all_edges_reciprocal and one_way.size:
        raise AttentionRoutingNicheError(
            f"Graph contains {len(one_way)} one-directional edges."
        )
    canonical_ids = np.flatnonzero(reciprocal & (edges[0] < edges[1])).astype(
        np.int64, copy=False
    )
    pairs = edges[:, canonical_ids].T.astype(np.int64, copy=False)
    if len(pairs):
        pair_order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[pair_order]
        canonical_ids = canonical_ids[pair_order]
    opposite_ids = reverse_edge_ids[canonical_ids]
    directed_pair_positions = np.full(edge_count, -1, dtype=np.int64)
    pair_positions = np.arange(len(pairs), dtype=np.int64)
    directed_pair_positions[canonical_ids] = pair_positions
    directed_pair_positions[opposite_ids] = pair_positions
    payload = {
        "schema": "attention_reciprocal_pair_index_v1",
        "n_nodes": int(n_nodes),
        "pair_cells_sha256": ndarray_sha256(pairs),
        "i_to_j_edge_ids_sha256": ndarray_sha256(canonical_ids),
        "j_to_i_edge_ids_sha256": ndarray_sha256(opposite_ids),
        "one_way_edge_ids_sha256": ndarray_sha256(one_way),
        "reverse_edge_ids_sha256": ndarray_sha256(reverse_edge_ids),
        "directed_pair_positions_sha256": ndarray_sha256(
            directed_pair_positions
        ),
    }
    return ReciprocalPairIndex(
        n_nodes=int(n_nodes),
        pair_cells=pairs,
        i_to_j_edge_ids=canonical_ids,
        j_to_i_edge_ids=opposite_ids,
        one_way_edge_ids=one_way,
        reverse_edge_ids=reverse_edge_ids,
        directed_pair_positions=directed_pair_positions,
        receipt_sha256=_payload_sha256(payload),
    )


# ---------------------------------------------------------------------------
# Cross-seed and cross-mask reciprocal consensus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingConsensus:
    """Reciprocal routing statistics aligned to canonical undirected pairs."""

    pairs: ReciprocalPairIndex
    seed_ids: tuple[int, ...]
    mask_view_ids: tuple[int, ...]
    mutual_samples: np.ndarray = field(repr=False)
    consensus_mutual_score: np.ndarray = field(repr=False)
    support_fraction: np.ndarray = field(repr=False)
    i_to_j_directional_median: np.ndarray = field(repr=False)
    j_to_i_directional_median: np.ndarray = field(repr=False)
    per_seed_view_median: np.ndarray = field(repr=False)
    seed_mean_after_view_median: np.ndarray = field(repr=False)
    seed_standard_deviation_after_view_median: np.ndarray | None = field(
        repr=False
    )
    per_mask_view_seed_median: np.ndarray = field(repr=False)
    mask_view_mean_after_seed_median: np.ndarray = field(repr=False)
    mask_view_standard_deviation_after_seed_median: np.ndarray | None = field(
        repr=False
    )
    uniform_threshold: float
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.pairs, ReciprocalPairIndex):
            raise AttentionRoutingNicheError(
                "pairs must be a canonical ReciprocalPairIndex."
            )
        seed_ids = tuple(int(value) for value in self.seed_ids)
        view_ids = tuple(int(value) for value in self.mask_view_ids)
        if (
            not seed_ids
            or len(set(seed_ids)) != len(seed_ids)
            or not view_ids
            or len(set(view_ids)) != len(view_ids)
        ):
            raise AttentionRoutingNicheError(
                "Seed and mask-view identifiers must be non-empty and unique."
            )
        pair_count = self.pairs.pair_count
        samples = _finite_floating_array_preserve_precision(
            self.mutual_samples,
            name="mutual_samples",
            ndim=3,
        )
        samples = _readonly_contiguous_view(samples)
        score = _readonly_array(self.consensus_mutual_score, dtype=np.float64)
        support = _readonly_array(self.support_fraction, dtype=np.float64)
        i_to_j = _readonly_array(self.i_to_j_directional_median, dtype=np.float64)
        j_to_i = _readonly_array(self.j_to_i_directional_median, dtype=np.float64)
        per_seed = _readonly_array(self.per_seed_view_median, dtype=np.float64)
        seed_mean = _readonly_array(
            self.seed_mean_after_view_median, dtype=np.float64
        )
        per_view = _readonly_array(self.per_mask_view_seed_median, dtype=np.float64)
        view_mean = _readonly_array(
            self.mask_view_mean_after_seed_median, dtype=np.float64
        )
        if (
            samples.shape != (len(seed_ids), len(view_ids), pair_count)
            or score.shape != (pair_count,)
            or support.shape != (pair_count,)
            or i_to_j.shape != (pair_count,)
            or j_to_i.shape != (pair_count,)
            or per_seed.shape != (len(seed_ids), pair_count)
            or seed_mean.shape != (pair_count,)
            or per_view.shape != (len(view_ids), pair_count)
            or view_mean.shape != (pair_count,)
        ):
            raise AttentionRoutingNicheError(
                "Routing-consensus arrays are not pair/seed/view aligned."
            )
        arrays = (samples, score, support, i_to_j, j_to_i, per_seed, seed_mean, per_view, view_mean)
        if any(not np.isfinite(array).all() for array in arrays) or any(
            np.any(array < 0.0) for array in arrays
        ):
            raise AttentionRoutingNicheError(
                "Routing consensus contains a non-finite or negative value."
            )
        if np.any(support > 1.0):
            raise AttentionRoutingNicheError("Support fractions must be in [0,1].")
        seed_sd: np.ndarray | None
        if self.seed_standard_deviation_after_view_median is None:
            if len(seed_ids) != 1:
                raise AttentionRoutingNicheError(
                    "Seed spread may be absent only for a single-model analysis."
                )
            seed_sd = None
        else:
            seed_sd = _readonly_array(
                self.seed_standard_deviation_after_view_median, dtype=np.float64
            )
            if seed_sd.shape != (pair_count,) or not np.isfinite(seed_sd).all() or np.any(seed_sd < 0.0):
                raise AttentionRoutingNicheError("Seed spread is malformed.")
        view_sd: np.ndarray | None
        if self.mask_view_standard_deviation_after_seed_median is None:
            if len(view_ids) != 1:
                raise AttentionRoutingNicheError(
                    "Mask-view spread may be absent only for one view."
                )
            view_sd = None
        else:
            view_sd = _readonly_array(
                self.mask_view_standard_deviation_after_seed_median,
                dtype=np.float64,
            )
            if view_sd.shape != (pair_count,) or not np.isfinite(view_sd).all() or np.any(view_sd < 0.0):
                raise AttentionRoutingNicheError("Mask-view spread is malformed.")
        threshold = _finite_float(
            self.uniform_threshold, name="uniform_threshold", minimum=0.0
        )
        payload = {
            "schema": "attention_mutual_routing_consensus_v1",
            "pair_index_receipt_sha256": self.pairs.receipt_sha256,
            "seed_ids": list(seed_ids),
            "mask_view_ids": list(view_ids),
            "mutual_samples_sha256": ndarray_sha256(samples),
            "consensus_mutual_score_sha256": ndarray_sha256(score),
            "support_fraction_sha256": ndarray_sha256(support),
            "i_to_j_directional_median_sha256": ndarray_sha256(i_to_j),
            "j_to_i_directional_median_sha256": ndarray_sha256(j_to_i),
            "per_seed_view_median_sha256": ndarray_sha256(per_seed),
            "seed_mean_after_view_median_sha256": ndarray_sha256(seed_mean),
            "seed_standard_deviation_after_view_median_sha256": (
                None if seed_sd is None else ndarray_sha256(seed_sd)
            ),
            "per_mask_view_seed_median_sha256": ndarray_sha256(per_view),
            "mask_view_mean_after_seed_median_sha256": ndarray_sha256(view_mean),
            "mask_view_standard_deviation_after_seed_median_sha256": (
                None if view_sd is None else ndarray_sha256(view_sd)
            ),
            "uniform_threshold": threshold,
            "consensus_aggregation": "median_over_seed_and_mask_view",
            "axis_spread_aggregation": "median_within_axis_then_sample_sd_across_axis",
        }
        if self.receipt_sha256 != _payload_sha256(payload):
            raise AttentionRoutingNicheError(
                "Routing-consensus receipt checksum mismatch."
            )
        object.__setattr__(self, "seed_ids", seed_ids)
        object.__setattr__(self, "mask_view_ids", view_ids)
        object.__setattr__(self, "mutual_samples", samples)
        object.__setattr__(self, "consensus_mutual_score", score)
        object.__setattr__(self, "support_fraction", support)
        object.__setattr__(self, "i_to_j_directional_median", i_to_j)
        object.__setattr__(self, "j_to_i_directional_median", j_to_i)
        object.__setattr__(self, "per_seed_view_median", per_seed)
        object.__setattr__(self, "seed_mean_after_view_median", seed_mean)
        object.__setattr__(
            self, "seed_standard_deviation_after_view_median", seed_sd
        )
        object.__setattr__(self, "per_mask_view_seed_median", per_view)
        object.__setattr__(self, "mask_view_mean_after_seed_median", view_mean)
        object.__setattr__(
            self, "mask_view_standard_deviation_after_seed_median", view_sd
        )
        object.__setattr__(self, "uniform_threshold", threshold)

    @property
    def Mij(self) -> np.ndarray:
        """Alias for the locked consensus mutual score."""

        return self.consensus_mutual_score

    @property
    def Pij(self) -> np.ndarray:
        """Alias for the locked above-uniform support fraction."""

        return self.support_fraction

    @property
    def has_seed_spread(self) -> bool:
        return self.seed_standard_deviation_after_view_median is not None


def summarize_mutual_routing_consensus(
    edge_index: Any,
    directional_routing_samples: Any,
    *,
    n_nodes: int,
    seed_ids: Sequence[int] | None = None,
    mask_view_ids: Sequence[int] | None = None,
    uniform_threshold: float = DEFAULT_MUTUAL_SCORE_THRESHOLD,
    require_all_edges_reciprocal: bool = False,
    pair_chunk_size: int = DEFAULT_PAIR_CHUNK_SIZE,
) -> RoutingConsensus:
    """Aggregate ``[seed, view, directed_edge]`` routing into mutual pairs."""

    edges = _validate_edge_index(edge_index, n_nodes=n_nodes)
    routing = _finite_floating_array_preserve_precision(
        directional_routing_samples,
        name="directional_routing_samples",
        ndim=3,
    )
    if (
        routing.shape[0] == 0
        or routing.shape[1] == 0
        or routing.shape[2] != edges.shape[1]
        or np.any(routing < 0.0)
    ):
        raise AttentionRoutingNicheError(
            "directional_routing_samples must be non-negative "
            "[seed, view, directed_edge]."
        )
    seeds = (
        tuple(range(routing.shape[0]))
        if seed_ids is None
        else tuple(int(value) for value in seed_ids)
    )
    views = (
        tuple(range(routing.shape[1]))
        if mask_view_ids is None
        else tuple(int(value) for value in mask_view_ids)
    )
    if (
        len(seeds) != routing.shape[0]
        or len(set(seeds)) != len(seeds)
        or len(views) != routing.shape[1]
        or len(set(views)) != len(views)
    ):
        raise AttentionRoutingNicheError(
            "seed_ids and mask_view_ids must uniquely align to sample axes."
        )
    threshold = _finite_float(
        uniform_threshold, name="uniform_threshold", minimum=0.0
    )
    pair_chunk_size = _positive_integer(
        pair_chunk_size, name="pair_chunk_size"
    )
    pairs = build_reciprocal_pair_index(
        edges,
        n_nodes=n_nodes,
        require_all_edges_reciprocal=require_all_edges_reciprocal,
    )
    pair_count = pairs.pair_count
    if pair_count:
        sample_shape = (len(seeds), len(views), pair_count)
        mutual = np.empty(sample_shape, dtype=routing.dtype)
        consensus_score = np.empty(pair_count, dtype=np.float64)
        support = np.empty(pair_count, dtype=np.float64)
        i_to_j_median = np.empty(pair_count, dtype=np.float64)
        j_to_i_median = np.empty(pair_count, dtype=np.float64)
        per_seed = np.empty((len(seeds), pair_count), dtype=np.float64)
        per_view = np.empty((len(views), pair_count), dtype=np.float64)
        for start in range(0, pair_count, pair_chunk_size):
            stop = min(start + pair_chunk_size, pair_count)
            forward = routing[..., pairs.i_to_j_edge_ids[start:stop]]
            backward = routing[..., pairs.j_to_i_edge_ids[start:stop]]
            mutual_chunk = np.minimum(forward, backward)
            mutual[..., start:stop] = mutual_chunk
            consensus_score[start:stop] = np.median(
                mutual_chunk, axis=(0, 1)
            )
            support[start:stop] = np.mean(
                mutual_chunk > threshold,
                axis=(0, 1),
                dtype=np.float64,
            )
            i_to_j_median[start:stop] = np.median(
                forward, axis=(0, 1)
            )
            j_to_i_median[start:stop] = np.median(
                backward, axis=(0, 1)
            )
            per_seed[:, start:stop] = np.median(mutual_chunk, axis=1)
            per_view[:, start:stop] = np.median(mutual_chunk, axis=0)
        seed_mean = per_seed.mean(axis=0, dtype=np.float64)
        seed_sd = (
            per_seed.std(axis=0, ddof=1, dtype=np.float64)
            if len(seeds) > 1
            else None
        )
        view_mean = per_view.mean(axis=0, dtype=np.float64)
        view_sd = (
            per_view.std(axis=0, ddof=1, dtype=np.float64)
            if len(views) > 1
            else None
        )
    else:
        consensus_score = np.empty(0, dtype=np.float64)
        support = np.empty(0, dtype=np.float64)
        i_to_j_median = np.empty(0, dtype=np.float64)
        j_to_i_median = np.empty(0, dtype=np.float64)
        per_seed = np.empty((len(seeds), 0), dtype=np.float64)
        seed_mean = np.empty(0, dtype=np.float64)
        seed_sd = np.empty(0, dtype=np.float64) if len(seeds) > 1 else None
        per_view = np.empty((len(views), 0), dtype=np.float64)
        view_mean = np.empty(0, dtype=np.float64)
        view_sd = np.empty(0, dtype=np.float64) if len(views) > 1 else None
    payload = {
        "schema": "attention_mutual_routing_consensus_v1",
        "pair_index_receipt_sha256": pairs.receipt_sha256,
        "seed_ids": list(seeds),
        "mask_view_ids": list(views),
        "mutual_samples_sha256": ndarray_sha256(mutual),
        "consensus_mutual_score_sha256": ndarray_sha256(consensus_score),
        "support_fraction_sha256": ndarray_sha256(support),
        "i_to_j_directional_median_sha256": ndarray_sha256(i_to_j_median),
        "j_to_i_directional_median_sha256": ndarray_sha256(j_to_i_median),
        "per_seed_view_median_sha256": ndarray_sha256(per_seed),
        "seed_mean_after_view_median_sha256": ndarray_sha256(seed_mean),
        "seed_standard_deviation_after_view_median_sha256": (
            None if seed_sd is None else ndarray_sha256(seed_sd)
        ),
        "per_mask_view_seed_median_sha256": ndarray_sha256(per_view),
        "mask_view_mean_after_seed_median_sha256": ndarray_sha256(view_mean),
        "mask_view_standard_deviation_after_seed_median_sha256": (
            None if view_sd is None else ndarray_sha256(view_sd)
        ),
        "uniform_threshold": threshold,
        "consensus_aggregation": "median_over_seed_and_mask_view",
        "axis_spread_aggregation": "median_within_axis_then_sample_sd_across_axis",
    }
    return RoutingConsensus(
        pairs=pairs,
        seed_ids=seeds,
        mask_view_ids=views,
        mutual_samples=mutual,
        consensus_mutual_score=consensus_score,
        support_fraction=support,
        i_to_j_directional_median=i_to_j_median,
        j_to_i_directional_median=j_to_i_median,
        per_seed_view_median=per_seed,
        seed_mean_after_view_median=seed_mean,
        seed_standard_deviation_after_view_median=seed_sd,
        per_mask_view_seed_median=per_view,
        mask_view_mean_after_seed_median=view_mean,
        mask_view_standard_deviation_after_seed_median=view_sd,
        uniform_threshold=threshold,
        receipt_sha256=_payload_sha256(payload),
    )


# ---------------------------------------------------------------------------
# Locked per-cell top-k union and weighted Leiden
# ---------------------------------------------------------------------------


def _validate_pair_values(
    pair_cells: Any,
    values: Any,
    support: Any,
    *,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.asarray(pair_cells)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise AttentionRoutingNicheError(
            "pair_cells must be integral canonical [pairs, 2]."
        )
    pairs = np.ascontiguousarray(pairs, dtype=np.int64)
    pair_count = len(pairs)
    if (
        np.any(pairs < 0)
        or np.any(pairs >= int(n_nodes))
        or np.any(pairs[:, 0] >= pairs[:, 1])
    ):
        raise AttentionRoutingNicheError(
            "pair_cells must contain in-range canonical i < j pairs."
        )
    if pair_count:
        codes = pairs[:, 0] * int(n_nodes) + pairs[:, 1]
        if len(np.unique(codes)) != pair_count:
            raise AttentionRoutingNicheError("pair_cells contains duplicates.")
    scores = _finite_array(values, name="consensus_mutual_score", ndim=1)
    fractions = _finite_array(support, name="support_fraction", ndim=1)
    if scores.shape != (pair_count,) or fractions.shape != (pair_count,):
        raise AttentionRoutingNicheError(
            "Pair scores and support must align one-for-one with pair_cells."
        )
    if np.any(scores < 0.0) or np.any(fractions < 0.0) or np.any(fractions > 1.0):
        raise AttentionRoutingNicheError(
            "Pair scores must be non-negative and support must be in [0,1]."
        )
    return pairs, scores, fractions


@dataclass(frozen=True)
class RetainedMutualGraph:
    """Deterministic undirected union of each cell's top-k eligible pairs."""

    n_nodes: int
    pair_positions: np.ndarray = field(repr=False)
    edge_pairs: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    support_fraction: np.ndarray = field(repr=False)
    endpoint_selection_count: np.ndarray = field(repr=False)
    eligible_pair_count: int
    top_k: int
    score_threshold: float
    support_threshold: float
    receipt_sha256: str

    def __post_init__(self) -> None:
        n_nodes = _positive_integer(self.n_nodes, name="n_nodes")
        positions = _readonly_array(self.pair_positions, dtype=np.int64)
        pairs = _readonly_array(self.edge_pairs, dtype=np.int64)
        weights = _readonly_array(self.weights, dtype=np.float64)
        support = _readonly_array(self.support_fraction, dtype=np.float64)
        endpoint_count = _readonly_array(
            self.endpoint_selection_count, dtype=np.int64
        )
        retained_count = len(positions)
        if (
            positions.shape != (retained_count,)
            or pairs.shape != (retained_count, 2)
            or weights.shape != (retained_count,)
            or support.shape != (retained_count,)
            or endpoint_count.shape != (retained_count,)
            or np.any(positions < 0)
            or np.any(pairs < 0)
            or np.any(pairs >= n_nodes)
            or np.any(pairs[:, 0] >= pairs[:, 1])
            or np.any(weights < 0.0)
            or np.any(support < 0.0)
            or np.any(support > 1.0)
            or np.any((endpoint_count < 1) | (endpoint_count > 2))
        ):
            raise AttentionRoutingNicheError("Retained mutual graph is malformed.")
        if retained_count > 1:
            codes = pairs[:, 0] * n_nodes + pairs[:, 1]
            if np.any(codes[1:] <= codes[:-1]):
                raise AttentionRoutingNicheError(
                    "Retained mutual pairs must be canonical and sorted."
                )
        eligible_count = _nonnegative_integer(
            self.eligible_pair_count, name="eligible_pair_count"
        )
        if retained_count > eligible_count:
            raise AttentionRoutingNicheError(
                "Retained edge count exceeds the eligible pair count."
            )
        top_k = _positive_integer(self.top_k, name="top_k")
        score_threshold = _finite_float(
            self.score_threshold, name="score_threshold", minimum=0.0
        )
        support_threshold = _finite_float(
            self.support_threshold, name="support_threshold", minimum=0.0
        )
        if support_threshold > 1.0:
            raise AttentionRoutingNicheError("support_threshold must be in [0,1].")
        payload = {
            "schema": "attention_retained_mutual_graph_v1",
            "n_nodes": n_nodes,
            "pair_positions_sha256": ndarray_sha256(positions),
            "edge_pairs_sha256": ndarray_sha256(pairs),
            "weights_sha256": ndarray_sha256(weights),
            "support_fraction_sha256": ndarray_sha256(support),
            "endpoint_selection_count_sha256": ndarray_sha256(endpoint_count),
            "eligible_pair_count": eligible_count,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "support_threshold": support_threshold,
            "selection": "per-cell top-k then undirected union and deduplication",
            "tie_break": "descending_score_then_neighbor_index_then_pair_position",
        }
        if self.receipt_sha256 != _payload_sha256(payload):
            raise AttentionRoutingNicheError(
                "Retained-mutual-graph receipt checksum mismatch."
            )
        object.__setattr__(self, "n_nodes", n_nodes)
        object.__setattr__(self, "pair_positions", positions)
        object.__setattr__(self, "edge_pairs", pairs)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "support_fraction", support)
        object.__setattr__(self, "endpoint_selection_count", endpoint_count)
        object.__setattr__(self, "eligible_pair_count", eligible_count)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "score_threshold", score_threshold)
        object.__setattr__(self, "support_threshold", support_threshold)

    @property
    def edge_count(self) -> int:
        return int(self.edge_pairs.shape[0])


def retain_top_k_mutual_edges(
    pair_cells: Any,
    consensus_mutual_score: Any,
    support_fraction: Any,
    *,
    n_nodes: int,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_MUTUAL_SCORE_THRESHOLD,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> RetainedMutualGraph:
    """Apply strict eligibility, per-cell top-k, undirected union, and dedup."""

    n_nodes = _positive_integer(n_nodes, name="n_nodes")
    top_k = _positive_integer(top_k, name="top_k")
    score_threshold = _finite_float(
        score_threshold, name="score_threshold", minimum=0.0
    )
    support_threshold = _finite_float(
        support_threshold, name="support_threshold", minimum=0.0
    )
    if support_threshold > 1.0:
        raise AttentionRoutingNicheError("support_threshold must be in [0,1].")
    pairs, scores, support = _validate_pair_values(
        pair_cells,
        consensus_mutual_score,
        support_fraction,
        n_nodes=n_nodes,
    )
    eligible = (scores > score_threshold) & (support >= support_threshold)
    eligible_positions = np.flatnonzero(eligible).astype(np.int64, copy=False)
    selected_count = np.zeros(len(pairs), dtype=np.int64)
    if len(eligible_positions):
        eligible_pairs = pairs[eligible_positions]
        endpoints = np.concatenate(
            (eligible_pairs[:, 0], eligible_pairs[:, 1])
        )
        neighbors = np.concatenate(
            (eligible_pairs[:, 1], eligible_pairs[:, 0])
        )
        pair_positions = np.concatenate(
            (eligible_positions, eligible_positions)
        )
        endpoint_scores = scores[pair_positions]
        # This is exactly the per-node ordering specified above, evaluated in
        # one vectorized endpoint grouping rather than rescanning every pair
        # for every node.  The latter is O(nodes * eligible_pairs) and is not
        # viable for the complete six-core graphs.
        order = np.lexsort(
            (pair_positions, neighbors, -endpoint_scores, endpoints)
        )
        ordered_endpoints = endpoints[order]
        group_starts = np.r_[
            0,
            np.flatnonzero(ordered_endpoints[1:] != ordered_endpoints[:-1]) + 1,
        ]
        group_lengths = np.diff(np.r_[group_starts, len(order)])
        within_group_rank = np.arange(len(order), dtype=np.int64) - np.repeat(
            group_starts, group_lengths
        )
        selected = pair_positions[order[within_group_rank < top_k]]
        np.add.at(selected_count, selected, 1)
    retained_positions = np.flatnonzero(selected_count > 0).astype(
        np.int64, copy=False
    )
    if len(retained_positions):
        retained_pairs = pairs[retained_positions]
        canonical_order = np.lexsort(
            (retained_pairs[:, 1], retained_pairs[:, 0])
        )
        retained_positions = retained_positions[canonical_order]
        retained_pairs = retained_pairs[canonical_order]
    else:
        retained_pairs = np.empty((0, 2), dtype=np.int64)
    retained_weights = scores[retained_positions]
    retained_support = support[retained_positions]
    retained_endpoint_count = selected_count[retained_positions]
    payload = {
        "schema": "attention_retained_mutual_graph_v1",
        "n_nodes": n_nodes,
        "pair_positions_sha256": ndarray_sha256(retained_positions),
        "edge_pairs_sha256": ndarray_sha256(retained_pairs),
        "weights_sha256": ndarray_sha256(retained_weights),
        "support_fraction_sha256": ndarray_sha256(retained_support),
        "endpoint_selection_count_sha256": ndarray_sha256(
            retained_endpoint_count
        ),
        "eligible_pair_count": int(eligible.sum()),
        "top_k": top_k,
        "score_threshold": score_threshold,
        "support_threshold": support_threshold,
        "selection": "per-cell top-k then undirected union and deduplication",
        "tie_break": "descending_score_then_neighbor_index_then_pair_position",
    }
    return RetainedMutualGraph(
        n_nodes=n_nodes,
        pair_positions=retained_positions,
        edge_pairs=retained_pairs,
        weights=retained_weights,
        support_fraction=retained_support,
        endpoint_selection_count=retained_endpoint_count,
        eligible_pair_count=int(eligible.sum()),
        top_k=top_k,
        score_threshold=score_threshold,
        support_threshold=support_threshold,
        receipt_sha256=_payload_sha256(payload),
    )


def mutual_routing_hub_scores(
    n_nodes: int,
    edge_pairs: Any,
    mutual_scores: Any,
) -> np.ndarray:
    """Return ``S_i`` as the sum of retained mutual scores incident to a cell."""

    n_nodes = _positive_integer(n_nodes, name="n_nodes")
    pairs = np.asarray(edge_pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise AttentionRoutingNicheError(
            "edge_pairs must be integral with shape [retained_edges, 2]."
        )
    pairs = np.asarray(pairs, dtype=np.int64)
    values = _finite_array(mutual_scores, name="mutual_scores", ndim=1)
    if values.shape != (len(pairs),) or np.any(values < 0.0):
        raise AttentionRoutingNicheError(
            "mutual_scores must be non-negative and edge aligned."
        )
    if pairs.size and (
        np.any(pairs < 0)
        or np.any(pairs >= n_nodes)
        or np.any(pairs[:, 0] >= pairs[:, 1])
    ):
        raise AttentionRoutingNicheError(
            "edge_pairs must be canonical in-range i < j pairs."
        )
    scores = np.zeros(n_nodes, dtype=np.float64)
    if len(pairs):
        np.add.at(scores, pairs[:, 0], values)
        np.add.at(scores, pairs[:, 1], values)
    return _readonly_array(scores)


@dataclass(frozen=True)
class LeidenPartition:
    """Deterministically relabelled weighted Leiden result for one core."""

    labels: np.ndarray = field(repr=False)
    community_sizes: np.ndarray = field(repr=False)
    resolution: float
    random_seed: int
    quality: float | None
    modularity: float | None
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        labels = _readonly_array(self.labels, dtype=np.int64)
        sizes = _readonly_array(self.community_sizes, dtype=np.int64)
        if labels.ndim != 1 or len(labels) == 0 or np.any(labels < 0):
            raise AttentionRoutingNicheError(
                "Leiden labels must be non-negative [n_nodes]."
            )
        if sizes.ndim != 1 or len(sizes) == 0 or np.any(sizes <= 0):
            raise AttentionRoutingNicheError("Leiden community sizes are invalid.")
        if not np.array_equal(np.unique(labels), np.arange(len(sizes))) or not np.array_equal(
            np.bincount(labels), sizes
        ):
            raise AttentionRoutingNicheError(
                "Leiden labels must be contiguous and align to community sizes."
            )
        resolution = _finite_float(
            self.resolution, name="resolution", minimum=np.finfo(float).tiny
        )
        seed = _nonnegative_integer(self.random_seed, name="random_seed")
        for name in ("quality", "modularity"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise AttentionRoutingNicheError(
                    f"Leiden {name} must be finite when available."
                )
        receipt = dict(self.receipt)
        if receipt.get("labels_sha256") != ndarray_sha256(labels):
            raise AttentionRoutingNicheError("Leiden receipt label checksum mismatch.")
        if receipt.get("community_sizes_sha256") != ndarray_sha256(sizes):
            raise AttentionRoutingNicheError(
                "Leiden receipt community-size checksum mismatch."
            )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "community_sizes", sizes)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "random_seed", seed)
        object.__setattr__(self, "receipt", receipt)

    @property
    def community_count(self) -> int:
        return int(len(self.community_sizes))


def _canonicalize_weighted_edges(
    n_nodes: int,
    edge_pairs: Any,
    edge_weights: Any,
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = _positive_integer(n_nodes, name="n_nodes")
    raw_pairs = np.asarray(edge_pairs)
    if (
        raw_pairs.ndim != 2
        or raw_pairs.shape[1] != 2
        or raw_pairs.dtype.kind not in "iu"
        or raw_pairs.dtype == np.dtype(np.bool_)
    ):
        raise AttentionRoutingNicheError(
            "Leiden edge_pairs must be integral [edges, 2]."
        )
    pairs = np.asarray(raw_pairs, dtype=np.int64)
    weights = _finite_array(edge_weights, name="edge_weights", ndim=1)
    if weights.shape != (len(pairs),) or np.any(weights <= 0.0):
        raise AttentionRoutingNicheError(
            "Leiden edge weights must be finite, positive, and edge aligned."
        )
    if pairs.size and (
        np.any(pairs < 0)
        or np.any(pairs >= n_nodes)
        or np.any(pairs[:, 0] == pairs[:, 1])
    ):
        raise AttentionRoutingNicheError(
            "Leiden edges must be in range and loop free."
        )
    canonical_pairs = np.sort(pairs, axis=1)
    if len(canonical_pairs):
        codes = canonical_pairs[:, 0] * n_nodes + canonical_pairs[:, 1]
        if len(np.unique(codes)) != len(codes):
            raise AttentionRoutingNicheError(
                "Leiden graph contains a duplicate undirected edge."
            )
        order = np.lexsort((canonical_pairs[:, 1], canonical_pairs[:, 0]))
        canonical_pairs = canonical_pairs[order]
        weights = weights[order]
    return np.ascontiguousarray(canonical_pairs), np.ascontiguousarray(weights)


def weighted_leiden_partition(
    n_nodes: int,
    edge_pairs: Any,
    edge_weights: Any,
    *,
    resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = LEIDEN_SEED,
    core_alias: str | None = None,
) -> LeidenPartition:
    """Run weighted seeded Leiden within one caller-supplied core graph.

    Isolated cells are retained as vertices and therefore receive singleton
    preliminary communities rather than disappearing from the partition.
    """

    n_nodes = _positive_integer(n_nodes, name="n_nodes")
    resolution = _finite_float(
        resolution, name="resolution", minimum=np.finfo(float).tiny
    )
    random_seed = _nonnegative_integer(random_seed, name="random_seed")
    alias = None if core_alias is None else _canonical_core_alias(core_alias)
    pairs, weights = _canonicalize_weighted_edges(
        n_nodes,
        edge_pairs,
        edge_weights,
    )
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise AttentionRoutingNicheError(
            "igraph and leidenalg are required for weighted Leiden clustering."
        ) from exc

    graph = ig.Graph(n=n_nodes, edges=pairs.tolist(), directed=False)
    if graph.vcount() != n_nodes or graph.ecount() != len(pairs):
        raise AttentionRoutingNicheError(
            "igraph changed the supplied sparse graph identity."
        )
    if len(pairs):
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights.tolist(),
            resolution_parameter=resolution,
            n_iterations=-1,
            seed=random_seed,
        )
        raw_labels = np.asarray(partition.membership, dtype=np.int64)
        quality: float | None = float(partition.quality())
        modularity: float | None = float(
            graph.modularity(raw_labels.tolist(), weights=weights.tolist())
        )
    else:
        raw_labels = np.arange(n_nodes, dtype=np.int64)
        quality = None
        modularity = None
    if raw_labels.shape != (n_nodes,) or np.any(raw_labels < 0):
        raise AttentionRoutingNicheError("Leiden returned invalid memberships.")
    raw_ids, counts = np.unique(raw_labels, return_counts=True)
    ordering = sorted(
        range(len(raw_ids)),
        key=lambda index: (
            -int(counts[index]),
            int(np.flatnonzero(raw_labels == raw_ids[index])[0]),
            int(raw_ids[index]),
        ),
    )
    mapping = {
        int(raw_ids[old_position]): new_position
        for new_position, old_position in enumerate(ordering)
    }
    labels = np.asarray([mapping[int(value)] for value in raw_labels], dtype=np.int64)
    community_sizes = np.bincount(labels)
    receipt: dict[str, object] = {
        "schema": "attention_weighted_leiden_partition_v1",
        "implementation": "leidenalg.RBConfigurationVertexPartition",
        "igraph_version": getattr(ig, "__version__", "unknown"),
        "leidenalg_version": getattr(leidenalg, "__version__", "unknown"),
        "core_alias": alias,
        "n_nodes": n_nodes,
        "edge_count": int(len(pairs)),
        "edge_pairs_sha256": ndarray_sha256(pairs),
        "edge_weights_sha256": ndarray_sha256(weights),
        "resolution": resolution,
        "random_seed": random_seed,
        "n_iterations": -1,
        "weighted": True,
        "isolated_vertices_retained": True,
        "quality": quality,
        "modularity": modularity,
        "community_count": int(len(community_sizes)),
        "community_sizes_sha256": ndarray_sha256(community_sizes),
        "labels_sha256": ndarray_sha256(labels),
        "cluster_sort": (
            "descending_size_then_minimum_cell_index_then_raw_community_id"
        ),
    }
    return LeidenPartition(
        labels=labels,
        community_sizes=community_sizes,
        resolution=resolution,
        random_seed=random_seed,
        quality=quality,
        modularity=modularity,
        receipt=receipt,
    )


@dataclass(frozen=True)
class RoutingGraphPartition:
    """A retained mutual-routing graph and its preliminary Leiden partition."""

    retained_graph: RetainedMutualGraph
    leiden: LeidenPartition

    def __post_init__(self) -> None:
        if not isinstance(self.retained_graph, RetainedMutualGraph) or not isinstance(
            self.leiden, LeidenPartition
        ):
            raise AttentionRoutingNicheError(
                "RoutingGraphPartition requires retained graph and Leiden result."
            )
        if len(self.leiden.labels) != self.retained_graph.n_nodes:
            raise AttentionRoutingNicheError(
                "Leiden labels do not align to retained-graph nodes."
            )


def build_consensus_routing_partition(
    consensus: RoutingConsensus,
    *,
    n_nodes: int,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_MUTUAL_SCORE_THRESHOLD,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = LEIDEN_SEED,
    core_alias: str | None = None,
) -> RoutingGraphPartition:
    """Apply the locked graph retention and weighted Leiden to consensus scores."""

    if not isinstance(consensus, RoutingConsensus):
        raise AttentionRoutingNicheError("consensus must be a RoutingConsensus.")
    if int(n_nodes) != consensus.pairs.n_nodes:
        raise AttentionRoutingNicheError("n_nodes disagrees with consensus pairs.")
    retained = retain_top_k_mutual_edges(
        consensus.pairs.pair_cells,
        consensus.consensus_mutual_score,
        consensus.support_fraction,
        n_nodes=n_nodes,
        top_k=top_k,
        score_threshold=score_threshold,
        support_threshold=support_threshold,
    )
    leiden = weighted_leiden_partition(
        n_nodes,
        retained.edge_pairs,
        retained.weights,
        resolution=resolution,
        random_seed=random_seed,
        core_alias=core_alias,
    )
    return RoutingGraphPartition(retained_graph=retained, leiden=leiden)


@dataclass(frozen=True)
class SeedRoutingPartition:
    """One model seed's ten-view routing graph and preliminary partition."""

    seed_id: int
    mutual_score_after_view_median: np.ndarray = field(repr=False)
    support_fraction_across_views: np.ndarray = field(repr=False)
    graph_partition: RoutingGraphPartition

    def __post_init__(self) -> None:
        seed = _nonnegative_integer(self.seed_id, name="seed_id")
        score = _readonly_array(
            self.mutual_score_after_view_median, dtype=np.float64
        )
        support = _readonly_array(
            self.support_fraction_across_views, dtype=np.float64
        )
        if (
            score.ndim != 1
            or support.shape != score.shape
            or not np.isfinite(score).all()
            or not np.isfinite(support).all()
            or np.any(score < 0.0)
            or np.any(support < 0.0)
            or np.any(support > 1.0)
            or not isinstance(self.graph_partition, RoutingGraphPartition)
        ):
            raise AttentionRoutingNicheError(
                "Seed-specific routing partition is malformed."
            )
        object.__setattr__(self, "seed_id", seed)
        object.__setattr__(self, "mutual_score_after_view_median", score)
        object.__setattr__(self, "support_fraction_across_views", support)


def build_seed_specific_partitions(
    consensus: RoutingConsensus,
    *,
    n_nodes: int,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_MUTUAL_SCORE_THRESHOLD,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    resolution: float = DEFAULT_LEIDEN_RESOLUTION,
    random_seed: int = LEIDEN_SEED,
    core_alias: str | None = None,
    require_ten_mask_views: bool = True,
) -> tuple[SeedRoutingPartition, ...]:
    """Aggregate views within each model seed and partition with locked settings."""

    if not isinstance(consensus, RoutingConsensus):
        raise AttentionRoutingNicheError("consensus must be a RoutingConsensus.")
    if int(n_nodes) != consensus.pairs.n_nodes:
        raise AttentionRoutingNicheError("n_nodes disagrees with consensus pairs.")
    if require_ten_mask_views and len(consensus.mask_view_ids) != ANALYSIS_MASK_VIEW_COUNT:
        raise AttentionRoutingNicheError(
            "Seed-specific production partitions require exactly ten mask views."
        )
    results: list[SeedRoutingPartition] = []
    for seed_position, seed_id in enumerate(consensus.seed_ids):
        samples = consensus.mutual_samples[seed_position]
        if samples.shape[1]:
            score = np.median(samples, axis=0)
            support = np.mean(
                samples > consensus.uniform_threshold,
                axis=0,
                dtype=np.float64,
            )
        else:
            score = np.empty(0, dtype=np.float64)
            support = np.empty(0, dtype=np.float64)
        retained = retain_top_k_mutual_edges(
            consensus.pairs.pair_cells,
            score,
            support,
            n_nodes=n_nodes,
            top_k=top_k,
            score_threshold=score_threshold,
            support_threshold=support_threshold,
        )
        leiden = weighted_leiden_partition(
            n_nodes,
            retained.edge_pairs,
            retained.weights,
            resolution=resolution,
            random_seed=random_seed,
            core_alias=core_alias,
        )
        results.append(
            SeedRoutingPartition(
                seed_id=seed_id,
                mutual_score_after_view_median=score,
                support_fraction_across_views=support,
                graph_partition=RoutingGraphPartition(
                    retained_graph=retained,
                    leiden=leiden,
                ),
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# Maximum-Jaccard seed agreement and parameter sensitivity
# ---------------------------------------------------------------------------


def _partition_labels(value: Any, *, name: str, n_nodes: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.dtype.kind not in "iu"
        or raw.dtype == np.dtype(np.bool_)
        or len(raw) == 0
    ):
        raise AttentionRoutingNicheError(
            f"{name} must be a non-empty integral label vector."
        )
    labels = np.asarray(raw, dtype=np.int64)
    if np.any(labels < 0) or (n_nodes is not None and len(labels) != n_nodes):
        raise AttentionRoutingNicheError(
            f"{name} labels must be non-negative and node aligned."
        )
    return labels


@dataclass(frozen=True)
class NicheJaccardMatch:
    seed_id: int
    seed_community: int
    consensus_community: int | None
    intersection_size: int
    union_size: int
    jaccard: float


def _maximum_jaccard_alignment(
    reference_labels: np.ndarray,
    candidate_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int | None, int, int, float], ...]]:
    reference_ids, reference_inverse = np.unique(
        reference_labels, return_inverse=True
    )
    candidate_ids, candidate_inverse = np.unique(
        candidate_labels, return_inverse=True
    )
    contingency = np.zeros(
        (len(candidate_ids), len(reference_ids)), dtype=np.int64
    )
    np.add.at(contingency, (candidate_inverse, reference_inverse), 1)
    candidate_sizes = contingency.sum(axis=1)
    reference_sizes = contingency.sum(axis=0)
    unions = (
        candidate_sizes[:, None]
        + reference_sizes[None, :]
        - contingency
    )
    jaccard = contingency / unions
    row_ids, column_ids = linear_sum_assignment(jaccard, maximize=True)
    candidate_to_reference = np.full(len(candidate_ids), -1, dtype=np.int64)
    matched_jaccard = np.zeros(len(candidate_ids), dtype=np.float64)
    match_rows: list[tuple[int, int | None, int, int, float]] = []
    assignments = {int(row): int(column) for row, column in zip(row_ids, column_ids, strict=True)}
    for candidate_position, candidate_id in enumerate(candidate_ids.tolist()):
        reference_position = assignments.get(candidate_position)
        if reference_position is None:
            match_rows.append((int(candidate_id), None, 0, int(candidate_sizes[candidate_position]), 0.0))
            continue
        reference_id = int(reference_ids[reference_position])
        candidate_to_reference[candidate_position] = reference_id
        value = float(jaccard[candidate_position, reference_position])
        matched_jaccard[candidate_position] = value
        match_rows.append(
            (
                int(candidate_id),
                reference_id,
                int(contingency[candidate_position, reference_position]),
                int(unions[candidate_position, reference_position]),
                value,
            )
        )
    mapped = candidate_to_reference[candidate_inverse]
    return mapped, matched_jaccard, tuple(match_rows)


@dataclass(frozen=True)
class SeedPartitionAgreement:
    """Maximum-Jaccard alignment of seed partitions to the consensus partition."""

    seed_ids: tuple[int, ...]
    consensus_community_ids: np.ndarray = field(repr=False)
    mapped_consensus_labels_by_seed: np.ndarray = field(repr=False)
    cell_assignment_agreement: np.ndarray = field(repr=False)
    niche_assignment_agreement: np.ndarray = field(repr=False)
    niche_mean_matched_jaccard: np.ndarray = field(repr=False)
    matches: tuple[NicheJaccardMatch, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        seeds = tuple(int(value) for value in self.seed_ids)
        communities = _readonly_array(
            self.consensus_community_ids, dtype=np.int64
        )
        mapped = _readonly_array(
            self.mapped_consensus_labels_by_seed, dtype=np.int64
        )
        cell = _readonly_array(self.cell_assignment_agreement, dtype=np.float64)
        niche = _readonly_array(self.niche_assignment_agreement, dtype=np.float64)
        jaccard = _readonly_array(self.niche_mean_matched_jaccard, dtype=np.float64)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or communities.ndim != 1
            or len(communities) == 0
            or mapped.ndim != 2
            or mapped.shape[0] != len(seeds)
            or cell.shape != (mapped.shape[1],)
            or niche.shape != communities.shape
            or jaccard.shape != communities.shape
            or np.any(cell < 0.0)
            or np.any(cell > 1.0)
            or np.any(niche < 0.0)
            or np.any(niche > 1.0)
            or np.any(jaccard < 0.0)
            or np.any(jaccard > 1.0)
        ):
            raise AttentionRoutingNicheError(
                "Seed-partition agreement arrays are malformed."
            )
        payload = {
            "schema": "attention_seed_partition_agreement_v1",
            "seed_ids": list(seeds),
            "consensus_community_ids_sha256": ndarray_sha256(communities),
            "mapped_consensus_labels_by_seed_sha256": ndarray_sha256(mapped),
            "cell_assignment_agreement_sha256": ndarray_sha256(cell),
            "niche_assignment_agreement_sha256": ndarray_sha256(niche),
            "niche_mean_matched_jaccard_sha256": ndarray_sha256(jaccard),
            "matches": [
                {
                    "seed_id": match.seed_id,
                    "seed_community": match.seed_community,
                    "consensus_community": match.consensus_community,
                    "intersection_size": match.intersection_size,
                    "union_size": match.union_size,
                    "jaccard": match.jaccard,
                }
                for match in self.matches
            ],
            "matching": "maximum_total_cell_set_jaccard_bipartite",
        }
        if self.receipt_sha256 != _payload_sha256(payload):
            raise AttentionRoutingNicheError(
                "Seed-partition agreement receipt checksum mismatch."
            )
        object.__setattr__(self, "seed_ids", seeds)
        object.__setattr__(self, "consensus_community_ids", communities)
        object.__setattr__(self, "mapped_consensus_labels_by_seed", mapped)
        object.__setattr__(self, "cell_assignment_agreement", cell)
        object.__setattr__(self, "niche_assignment_agreement", niche)
        object.__setattr__(self, "niche_mean_matched_jaccard", jaccard)


def match_seed_partitions_to_consensus(
    consensus_labels: Any,
    seed_labels: Any,
    *,
    seed_ids: Sequence[int] | None = None,
) -> SeedPartitionAgreement:
    """Match seed communities to consensus by maximum total cell-set Jaccard."""

    consensus = _partition_labels(consensus_labels, name="consensus_labels")
    raw_seed_labels = np.asarray(seed_labels)
    if raw_seed_labels.ndim != 2 or raw_seed_labels.dtype.kind not in "iu":
        raise AttentionRoutingNicheError(
            "seed_labels must be integral [seed, node]."
        )
    seeds_array = np.asarray(raw_seed_labels, dtype=np.int64)
    if seeds_array.shape[0] == 0 or seeds_array.shape[1] != len(consensus) or np.any(seeds_array < 0):
        raise AttentionRoutingNicheError(
            "seed_labels must contain non-negative, node-aligned partitions."
        )
    identifiers = (
        tuple(range(seeds_array.shape[0]))
        if seed_ids is None
        else tuple(int(value) for value in seed_ids)
    )
    if len(identifiers) != seeds_array.shape[0] or len(set(identifiers)) != len(identifiers):
        raise AttentionRoutingNicheError(
            "seed_ids must uniquely align to seed_labels."
        )
    consensus_ids = np.unique(consensus)
    consensus_position = {
        int(community): position
        for position, community in enumerate(consensus_ids.tolist())
    }
    mapped_rows: list[np.ndarray] = []
    jaccard_by_seed_consensus = np.zeros(
        (len(identifiers), len(consensus_ids)), dtype=np.float64
    )
    matches: list[NicheJaccardMatch] = []
    for seed_position, seed_id in enumerate(identifiers):
        mapped, _, match_rows = _maximum_jaccard_alignment(
            consensus,
            seeds_array[seed_position],
        )
        mapped_rows.append(mapped)
        for seed_community, consensus_community, intersection, union, jaccard in match_rows:
            matches.append(
                NicheJaccardMatch(
                    seed_id=seed_id,
                    seed_community=seed_community,
                    consensus_community=consensus_community,
                    intersection_size=intersection,
                    union_size=union,
                    jaccard=jaccard,
                )
            )
            if consensus_community is not None:
                jaccard_by_seed_consensus[
                    seed_position, consensus_position[consensus_community]
                ] = jaccard
    mapped_array = np.stack(mapped_rows, axis=0)
    agreement_matrix = mapped_array == consensus[None, :]
    cell_agreement = agreement_matrix.mean(axis=0, dtype=np.float64)
    niche_agreement = np.asarray(
        [
            agreement_matrix[:, consensus == community].mean(dtype=np.float64)
            for community in consensus_ids
        ],
        dtype=np.float64,
    )
    niche_jaccard = jaccard_by_seed_consensus.mean(axis=0, dtype=np.float64)
    payload = {
        "schema": "attention_seed_partition_agreement_v1",
        "seed_ids": list(identifiers),
        "consensus_community_ids_sha256": ndarray_sha256(consensus_ids),
        "mapped_consensus_labels_by_seed_sha256": ndarray_sha256(mapped_array),
        "cell_assignment_agreement_sha256": ndarray_sha256(cell_agreement),
        "niche_assignment_agreement_sha256": ndarray_sha256(niche_agreement),
        "niche_mean_matched_jaccard_sha256": ndarray_sha256(niche_jaccard),
        "matches": [
            {
                "seed_id": match.seed_id,
                "seed_community": match.seed_community,
                "consensus_community": match.consensus_community,
                "intersection_size": match.intersection_size,
                "union_size": match.union_size,
                "jaccard": match.jaccard,
            }
            for match in matches
        ],
        "matching": "maximum_total_cell_set_jaccard_bipartite",
    }
    return SeedPartitionAgreement(
        seed_ids=identifiers,
        consensus_community_ids=consensus_ids,
        mapped_consensus_labels_by_seed=mapped_array,
        cell_assignment_agreement=cell_agreement,
        niche_assignment_agreement=niche_agreement,
        niche_mean_matched_jaccard=niche_jaccard,
        matches=tuple(matches),
        receipt_sha256=_payload_sha256(payload),
    )


def _combination_two(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values * (values - 1.0) / 2.0


@dataclass(frozen=True)
class PartitionSimilarity:
    """Label-invariant comparison of two partitions over the same cells."""

    adjusted_rand_index: float
    coassignment_jaccard: float
    maximum_jaccard_cell_agreement: float
    mean_matched_niche_jaccard: float


def partition_similarity(reference_labels: Any, candidate_labels: Any) -> PartitionSimilarity:
    """Return deterministic label-invariant partition agreement statistics."""

    reference = _partition_labels(reference_labels, name="reference_labels")
    candidate = _partition_labels(
        candidate_labels,
        name="candidate_labels",
        n_nodes=len(reference),
    )
    _, reference_inverse = np.unique(reference, return_inverse=True)
    _, candidate_inverse = np.unique(candidate, return_inverse=True)
    contingency = np.zeros(
        (candidate_inverse.max() + 1, reference_inverse.max() + 1),
        dtype=np.int64,
    )
    np.add.at(contingency, (candidate_inverse, reference_inverse), 1)
    intersection_pairs = float(_combination_two(contingency).sum())
    candidate_pairs = float(_combination_two(contingency.sum(axis=1)).sum())
    reference_pairs = float(_combination_two(contingency.sum(axis=0)).sum())
    total_pairs = float(len(reference) * (len(reference) - 1) / 2)
    union_pairs = candidate_pairs + reference_pairs - intersection_pairs
    coassignment_jaccard = (
        intersection_pairs / union_pairs if union_pairs > 0.0 else 1.0
    )
    if total_pairs == 0.0:
        adjusted_rand = 1.0
    else:
        expected = candidate_pairs * reference_pairs / total_pairs
        maximum = 0.5 * (candidate_pairs + reference_pairs)
        denominator = maximum - expected
        if abs(denominator) <= np.finfo(float).eps:
            # Identical coassignment follows when every within-candidate pair
            # is also within-reference and vice versa.  This scalar test avoids
            # constructing an O(n^2) cell-by-cell coassignment matrix for the
            # all-singleton or all-one-community degeneracies.
            adjusted_rand = (
                1.0
                if intersection_pairs == candidate_pairs == reference_pairs
                else 0.0
            )
        else:
            adjusted_rand = (intersection_pairs - expected) / denominator
    mapped, matched_jaccard, _ = _maximum_jaccard_alignment(
        reference,
        candidate,
    )
    cell_agreement = float(np.mean(mapped == reference))
    # Treat unmatched communities on either side as zero-overlap rather than
    # silently averaging only favorable matches.
    match_denominator = max(
        len(np.unique(reference)),
        len(np.unique(candidate)),
    )
    mean_jaccard = float(matched_jaccard.sum() / match_denominator)
    return PartitionSimilarity(
        adjusted_rand_index=float(adjusted_rand),
        coassignment_jaccard=float(coassignment_jaccard),
        maximum_jaccard_cell_agreement=cell_agreement,
        mean_matched_niche_jaccard=mean_jaccard,
    )


@dataclass(frozen=True)
class SensitivitySummary:
    top_k: int
    resolution: float
    community_count: int
    is_primary: bool
    adjusted_rand_index_to_primary: float
    coassignment_jaccard_to_primary: float
    maximum_jaccard_cell_agreement_to_primary: float
    mean_matched_niche_jaccard_to_primary: float
    labels_sha256: str


def summarize_parameter_sensitivity(
    partitions: Mapping[tuple[int, float], Any],
    *,
    primary_top_k: int = DEFAULT_TOP_K,
    primary_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
) -> tuple[SensitivitySummary, ...]:
    """Compare top-k/resolution sensitivity partitions to the locked primary."""

    if not partitions:
        raise AttentionRoutingNicheError("partitions cannot be empty.")
    primary_key = (int(primary_top_k), float(primary_resolution))
    canonical: dict[tuple[int, float], np.ndarray] = {}
    n_nodes: int | None = None
    for key, labels_value in partitions.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise AttentionRoutingNicheError(
                "Sensitivity keys must be (top_k, resolution) tuples."
            )
        top_k = _positive_integer(key[0], name="sensitivity top_k")
        resolution = _finite_float(
            key[1],
            name="sensitivity resolution",
            minimum=np.finfo(float).tiny,
        )
        labels = _partition_labels(
            labels_value,
            name=f"partition[{top_k},{resolution}]",
            n_nodes=n_nodes,
        )
        if n_nodes is None:
            n_nodes = len(labels)
        canonical_key = (top_k, resolution)
        if canonical_key in canonical:
            raise AttentionRoutingNicheError(
                "Sensitivity configuration keys are duplicated after canonicalization."
            )
        canonical[canonical_key] = labels
    if primary_key not in canonical:
        raise AttentionRoutingNicheError(
            "The locked primary top-k/resolution partition is absent."
        )
    primary = canonical[primary_key]
    rows: list[SensitivitySummary] = []
    for key in sorted(canonical, key=lambda value: (value[0], value[1])):
        labels = canonical[key]
        similarity = partition_similarity(primary, labels)
        rows.append(
            SensitivitySummary(
                top_k=key[0],
                resolution=key[1],
                community_count=int(len(np.unique(labels))),
                is_primary=key == primary_key,
                adjusted_rand_index_to_primary=similarity.adjusted_rand_index,
                coassignment_jaccard_to_primary=similarity.coassignment_jaccard,
                maximum_jaccard_cell_agreement_to_primary=(
                    similarity.maximum_jaccard_cell_agreement
                ),
                mean_matched_niche_jaccard_to_primary=(
                    similarity.mean_matched_niche_jaccard
                ),
                labels_sha256=ndarray_sha256(labels),
            )
        )
    return tuple(rows)


__all__ = [
    "ANALYSIS_MASK_BASE_SEED",
    "ANALYSIS_MASK_VIEW_COUNT",
    "CORE_ALIASES",
    "DEFAULT_LEIDEN_RESOLUTION",
    "DEFAULT_MUTUAL_SCORE_THRESHOLD",
    "DEFAULT_PAIR_CHUNK_SIZE",
    "DEFAULT_SUPPORT_THRESHOLD",
    "DEFAULT_TOP_K",
    "LEIDEN_SEED",
    "SENSITIVITY_RESOLUTIONS",
    "SENSITIVITY_TOP_K",
    "AnalysisMaskView",
    "AttentionNormalizationAudit",
    "AttentionRoutingNicheError",
    "AttentionShardAudit",
    "DirectionalRouting",
    "LeidenPartition",
    "NicheJaccardMatch",
    "PartitionSimilarity",
    "ReciprocalPairIndex",
    "RetainedMutualGraph",
    "RoutingConsensus",
    "RoutingGraphPartition",
    "SeedPartitionAgreement",
    "SeedRoutingPartition",
    "SensitivitySummary",
    "audit_attention_normalization",
    "build_consensus_routing_partition",
    "build_reciprocal_pair_index",
    "build_seed_specific_partitions",
    "degree_adjusted_routing",
    "degree_adjusted_routing_samples",
    "derive_analysis_mask_seed",
    "make_analysis_mask_view",
    "make_analysis_mask_views",
    "match_seed_partitions_to_consensus",
    "mutual_routing_hub_scores",
    "partition_similarity",
    "receiver_in_degrees",
    "retain_top_k_mutual_edges",
    "summarize_mutual_routing_consensus",
    "summarize_parameter_sensitivity",
    "validate_attention_shard",
    "weighted_leiden_partition",
]
