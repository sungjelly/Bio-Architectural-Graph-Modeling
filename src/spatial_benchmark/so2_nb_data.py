"""Leakage-safe SO2 data overlay for donor-grouped NB2 reconstruction.

The immutable SO2 count and relative-geometry artifacts are deliberately not
copied.  This module writes and verifies a small overlay containing only:

* the fixed train/validation role assignment;
* train-core-only expression and covariate normalization statistics;
* deterministic validation-mask seeds and checksums; and
* checksum-bound references to the existing cohort and graph caches.

The existing SO2 node covariates are an affine standardization of complete,
measured morphology/imaging columns.  Recentring and rescaling those values
with moments fitted on the training cores cancels the original all-cohort
affine transform exactly in real arithmetic::

    z = (x - global_mean) / global_scale
    z_train = (z - train_mean(z)) / train_scale(z)
            = (x - train_mean(x)) / train_scale(x)

The preparation receipt records and verifies the numerical cancellation.  Raw
integer counts are retained as a distinct likelihood target; the model input
is a separately allocated, train-standardized ``log1p`` array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import yaml

from .adjacency_ablation import MaskRealization, sample_uniform_mask_numpy
from .fingerprints import sha256_file
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_N_GENES,
    EXPECTED_TOTAL_CELLS,
    SO2_ALIASES,
)
from .so2_relative_graphs import (
    _verify_so2_cohort_manifest,
    _verify_so2_graph_collection,
)


OVERLAY_SCHEMA = "so2_nb_train_validation_overlay_v1"
PREPROCESSING_SCHEMA = "so2_nb_train_only_preprocessing_v1"
VALIDATION_MASK_SCHEMA = "so2_nb_fixed_validation_mask_v1"
CAMPAIGN_ID = (
    "cmp_20260907_so2_geometry_modulated_relative_qkv_nb_train12_val2_seed0"
)
SO2_NB_TRAINING_ALIASES = tuple(f"SO2-C{core:02d}" for core in range(15, 27))
SO2_NB_VALIDATION_ALIASES = ("SO2-C27", "SO2-C28")
SO2_NB_TEST_ALIASES: tuple[str, ...] = ()
SO2_NB_TRAINING_CELLS = 208_696
SO2_NB_VALIDATION_CELLS = 37_367
SO2_NB_TEST_CELLS = 0
EXPECTED_NODE_COVARIATES = 22
VALIDATION_MASK_NAMESPACE = "bagm.so2.nb.fixed_validation_masks.v1"
VALIDATION_MASK_VIEW_COUNT = 10
VALIDATION_MASK_CHUNK_CELLS = 2_048
SELECTION_NAMESPACE = "bagm.so2.nb.train_validation.seed0.v1"
SELECTED_VALIDATION_PAIR_DIGEST = (
    "0b8d6d58e4c87953425c4132b0fd8334b368e4d96dc920cdf66a0b6a2360f05d"
)
PROTECTED_GROUPING_SOURCE_SHA256 = (
    "30a1c3de1fee0d0045ac6bbfa849f9bb47de437c1b11428e6e16ac48d06fd5e0"
)
FROZEN_TASK_CONTRACT_SHA256 = (
    "13ff231bf842ecddfe7bfe000660d108d04b829d9706978972e0cbd06b39d74f"
)
EXPECTED_DONOR_GROUP_PAIRS = tuple(
    (f"SO2-C{core:02d}", f"SO2-C{core + 1:02d}")
    for core in range(15, 29, 2)
)
DIRECTORY_MODE = 0o700
FILE_MODE = 0o600
_STATISTIC_ARRAY_NAMES = (
    "expression_log1p_mean",
    "expression_log1p_scale",
    "covariate_recenter_mean",
    "covariate_rescale",
    "covariate_train_log1p_mean",
    "covariate_train_log1p_scale",
)


class SO2NBDataContractError(ValueError):
    """Raised when the fixed SO2 NB data contract is violated."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _array_sha256(name: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(_canonical_json(list(array.shape)))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _readonly(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    result.setflags(write=False)
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise SO2NBDataContractError(
            f"Overlay JSON contains non-finite constant {value!r}."
        )

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SO2NBDataContractError(
                    f"Overlay JSON contains duplicate key {key!r}."
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except SO2NBDataContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SO2NBDataContractError("Cannot read the SO2 NB overlay manifest.") from exc
    if not isinstance(value, dict):
        raise SO2NBDataContractError("SO2 NB overlay manifest must be a mapping.")
    return value


def validation_pair_digest(
    aliases: Sequence[str], *, namespace: str = SELECTION_NAMESPACE
) -> str:
    """Hash one sorted, identifier-free core pair using the frozen encoding."""

    canonical = tuple(sorted(str(alias).strip().upper() for alias in aliases))
    if len(canonical) != 2 or len(set(canonical)) != 2:
        raise SO2NBDataContractError(
            "Validation selection requires exactly two distinct core aliases."
        )
    payload = f"{namespace}|{canonical[0]}+{canonical[1]}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ProtectedGroupingVerification:
    """Identifier-free result of inspecting the protected grouping source."""

    source_sha256: str
    donor_group_count: int
    cores_per_group: int
    core_pairs: tuple[tuple[str, str], ...]
    selected_validation_pair_digest: str

    def to_receipt(self) -> dict[str, Any]:
        return {
            "source_sha256": self.source_sha256,
            "donor_group_count": self.donor_group_count,
            "cores_per_group": self.cores_per_group,
            "core_pairs": [list(pair) for pair in self.core_pairs],
            "equivalence_classes_match_expected_core_pairs": True,
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_rule": (
                "minimum_sha256_of_namespace_plus_sorted_core_pair_aliases"
            ),
            "selected_validation_pair_digest": self.selected_validation_pair_digest,
            "donor_identifier_values_persisted_or_reported": False,
        }


def verify_protected_so2_grouping(
    source: str | Path,
    *,
    expected_source_sha256: str = PROTECTED_GROUPING_SOURCE_SHA256,
) -> ProtectedGroupingVerification:
    """Verify donor equality classes without returning or serializing identifiers.

    Only columns zero (core number) and one (protected grouping value) are read.
    Error messages intentionally contain neither cell values nor dataframe rows.
    """

    path = Path(source)
    observed_sha256 = sha256_file(path)
    if observed_sha256 != expected_source_sha256:
        raise SO2NBDataContractError("Protected grouping source checksum changed.")
    try:
        frame = pd.read_excel(path, header=None, usecols=[0, 1])
    except Exception:  # pandas/openpyxl errors can contain protected cell values
        raise SO2NBDataContractError(
            "Protected grouping source cannot be parsed with the frozen schema."
        ) from None
    numbers = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    selected = frame.loc[numbers.isin(range(15, 29)), [frame.columns[0], frame.columns[1]]]
    selected_numbers = pd.to_numeric(selected.iloc[:, 0], errors="coerce").astype(int)
    if (
        len(selected) != len(SO2_ALIASES)
        or selected_numbers.duplicated().any()
        or set(selected_numbers.tolist()) != set(range(15, 29))
        or selected.iloc[:, 1].isna().any()
    ):
        raise SO2NBDataContractError(
            "Protected grouping source does not cover each SO2 core exactly once."
        )
    ordered = selected.assign(_core_number=selected_numbers).sort_values(
        "_core_number"
    )
    protected = ordered.iloc[:, 1]
    try:
        codes, uniques = pd.factorize(protected, sort=False)
    except Exception:
        raise SO2NBDataContractError(
            "Protected grouping equality comparison failed."
        ) from None
    counts = np.bincount(codes, minlength=len(uniques))
    if len(uniques) != 7 or not np.array_equal(np.sort(counts), np.full(7, 2)):
        raise SO2NBDataContractError(
            "Protected grouping must contain seven equality classes of two cores."
        )
    observed_pairs: list[tuple[str, str]] = []
    core_numbers = ordered["_core_number"].to_numpy(dtype=np.int64)
    for code in range(len(uniques)):
        members = sorted(int(value) for value in core_numbers[codes == code])
        if len(members) != 2:
            raise SO2NBDataContractError(
                "Protected grouping contains an invalid equality-class size."
            )
        observed_pairs.append(tuple(f"SO2-C{value:02d}" for value in members))
    canonical_pairs = tuple(sorted(observed_pairs))
    if canonical_pairs != EXPECTED_DONOR_GROUP_PAIRS:
        raise SO2NBDataContractError(
            "Protected grouping equality classes differ from the frozen core pairs."
        )
    pair_digests = {
        pair: validation_pair_digest(pair) for pair in canonical_pairs
    }
    selected_pair, selected_digest = min(
        pair_digests.items(), key=lambda item: (item[1], item[0])
    )
    if (
        selected_pair != SO2_NB_VALIDATION_ALIASES
        or selected_digest != SELECTED_VALIDATION_PAIR_DIGEST
    ):
        raise SO2NBDataContractError(
            "Frozen identifier-free validation-pair selection no longer verifies."
        )
    # Do not retain the protected values or factorization labels.
    del protected, codes, uniques, frame, selected, ordered
    return ProtectedGroupingVerification(
        source_sha256=observed_sha256,
        donor_group_count=7,
        cores_per_group=2,
        core_pairs=canonical_pairs,
        selected_validation_pair_digest=selected_digest,
    )


def _validate_training_aliases(aliases: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple(str(alias).strip().upper() for alias in aliases)
    if not canonical or len(set(canonical)) != len(canonical):
        raise SO2NBDataContractError("Training aliases must be nonempty and unique.")
    return canonical


def _equal_core_mean_scale(
    arrays: Mapping[str, np.ndarray],
    aliases: Sequence[str],
    *,
    log1p: bool,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    aliases = _validate_training_aliases(aliases)
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise SO2NBDataContractError("Preprocessing epsilon must be positive.")
    means: list[np.ndarray] = []
    seconds: list[np.ndarray] = []
    width: int | None = None
    for alias in aliases:
        if alias not in arrays:
            raise SO2NBDataContractError("A training core array is unavailable.")
        raw = np.asarray(arrays[alias])
        if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
            raise SO2NBDataContractError("Core preprocessing arrays must be nonempty 2D.")
        if width is None:
            width = int(raw.shape[1])
        elif raw.shape[1] != width:
            raise SO2NBDataContractError("Core preprocessing widths are inconsistent.")
        if log1p:
            if raw.dtype != np.int32 or np.any(raw < 0):
                raise SO2NBDataContractError(
                    "Expression inputs must originate from nonnegative int32 counts."
                )
            values = np.log1p(raw.astype(np.float64, copy=False))
        else:
            if not np.issubdtype(raw.dtype, np.floating) or not np.isfinite(raw).all():
                raise SO2NBDataContractError(
                    "Existing standardized covariates must be finite floating values."
                )
            values = raw.astype(np.float64, copy=False)
        means.append(values.mean(axis=0, dtype=np.float64))
        seconds.append(np.square(values).mean(axis=0, dtype=np.float64))
    mean = np.stack(means, axis=0).mean(axis=0, dtype=np.float64)
    second = np.stack(seconds, axis=0).mean(axis=0, dtype=np.float64)
    variance = np.maximum(second - np.square(mean), 0.0)
    raw_scale = np.sqrt(variance)
    scale = np.where(raw_scale > float(epsilon), raw_scale, 1.0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise SO2NBDataContractError("Train-only preprocessing statistics are invalid.")
    return _readonly(mean, dtype=np.float64), _readonly(scale, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SO2NBPreprocessingStatistics:
    """Train-fitted statistics for expression inputs and node covariates."""

    training_aliases: tuple[str, ...]
    expression_log1p_mean: np.ndarray = field(repr=False)
    expression_log1p_scale: np.ndarray = field(repr=False)
    covariate_recenter_mean: np.ndarray = field(repr=False)
    covariate_rescale: np.ndarray = field(repr=False)
    covariate_train_log1p_mean: np.ndarray = field(repr=False)
    covariate_train_log1p_scale: np.ndarray = field(repr=False)
    affine_cancellation_max_abs_error: float

    def __post_init__(self) -> None:
        aliases = _validate_training_aliases(self.training_aliases)
        expression_mean = _readonly(self.expression_log1p_mean, dtype=np.float64)
        expression_scale = _readonly(self.expression_log1p_scale, dtype=np.float64)
        covariate_mean = _readonly(self.covariate_recenter_mean, dtype=np.float64)
        covariate_scale = _readonly(self.covariate_rescale, dtype=np.float64)
        train_log_mean = _readonly(self.covariate_train_log1p_mean, dtype=np.float64)
        train_log_scale = _readonly(self.covariate_train_log1p_scale, dtype=np.float64)
        if expression_mean.shape != expression_scale.shape or expression_mean.ndim != 1:
            raise SO2NBDataContractError("Expression statistics must be aligned vectors.")
        if covariate_mean.shape != covariate_scale.shape or covariate_mean.ndim != 1:
            raise SO2NBDataContractError("Covariate statistics must be aligned vectors.")
        if train_log_mean.shape != covariate_mean.shape or train_log_scale.shape != covariate_mean.shape:
            raise SO2NBDataContractError("Affine-cancellation statistics are misaligned.")
        for scale in (expression_scale, covariate_scale, train_log_scale):
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise SO2NBDataContractError("Preprocessing scales must be finite and positive.")
        for mean in (expression_mean, covariate_mean, train_log_mean):
            if not np.isfinite(mean).all():
                raise SO2NBDataContractError("Preprocessing means must be finite.")
        error = float(self.affine_cancellation_max_abs_error)
        if not math.isfinite(error) or error < 0 or error > 1e-10:
            raise SO2NBDataContractError("Covariate affine cancellation did not verify.")
        object.__setattr__(self, "training_aliases", aliases)
        object.__setattr__(self, "expression_log1p_mean", expression_mean)
        object.__setattr__(self, "expression_log1p_scale", expression_scale)
        object.__setattr__(self, "covariate_recenter_mean", covariate_mean)
        object.__setattr__(self, "covariate_rescale", covariate_scale)
        object.__setattr__(self, "covariate_train_log1p_mean", train_log_mean)
        object.__setattr__(self, "covariate_train_log1p_scale", train_log_scale)
        object.__setattr__(self, "affine_cancellation_max_abs_error", error)

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(
            {
                "schema": PREPROCESSING_SCHEMA,
                "fit_aliases": list(self.training_aliases),
                "expression_transform": "log1p_then_gene_wise_standardize",
                "expression_moment_weighting": "equal_core",
                "covariate_transform": (
                    "equal_core_recenter_rescale_of_existing_standardized_covariates"
                ),
                "covariate_affine_cancellation_verified": True,
                "arrays": {
                    name: _array_sha256(name, getattr(self, name))
                    for name in _STATISTIC_ARRAY_NAMES
                },
            }
        )

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in _STATISTIC_ARRAY_NAMES}


def fit_so2_nb_preprocessing(
    expression_counts_by_alias: Mapping[str, np.ndarray],
    standardized_covariates_by_alias: Mapping[str, np.ndarray],
    *,
    training_aliases: Sequence[str] = SO2_NB_TRAINING_ALIASES,
    source_global_covariate_mean: np.ndarray | None = None,
    source_global_covariate_scale: np.ndarray | None = None,
    epsilon: float = 1e-8,
) -> SO2NBPreprocessingStatistics:
    """Fit both transforms while reading only explicitly named training cores."""

    aliases = _validate_training_aliases(training_aliases)
    expression_mean, expression_scale = _equal_core_mean_scale(
        expression_counts_by_alias, aliases, log1p=True, epsilon=epsilon
    )
    covariate_mean, covariate_scale = _equal_core_mean_scale(
        standardized_covariates_by_alias,
        aliases,
        log1p=False,
        epsilon=epsilon,
    )
    width = len(covariate_mean)
    global_mean = np.zeros(width, dtype=np.float64)
    global_scale = np.ones(width, dtype=np.float64)
    if source_global_covariate_mean is not None:
        global_mean = np.asarray(source_global_covariate_mean, dtype=np.float64)
    if source_global_covariate_scale is not None:
        global_scale = np.asarray(source_global_covariate_scale, dtype=np.float64)
    if (
        global_mean.shape != (width,)
        or global_scale.shape != (width,)
        or not np.isfinite(global_mean).all()
        or not np.isfinite(global_scale).all()
        or np.any(global_scale <= 0)
    ):
        raise SO2NBDataContractError("Source global covariate statistics are invalid.")
    train_log_mean = global_mean + global_scale * covariate_mean
    train_log_scale = global_scale * covariate_scale
    maximum_error = 0.0
    for alias in aliases:
        z_values = np.asarray(
            standardized_covariates_by_alias[alias], dtype=np.float64
        )
        recentered = (z_values - covariate_mean) / covariate_scale
        reconstructed_log = z_values * global_scale + global_mean
        direct = (reconstructed_log - train_log_mean) / train_log_scale
        maximum_error = max(
            maximum_error,
            float(np.max(np.abs(recentered - direct), initial=0.0)),
        )
    return SO2NBPreprocessingStatistics(
        training_aliases=aliases,
        expression_log1p_mean=expression_mean,
        expression_log1p_scale=expression_scale,
        covariate_recenter_mean=covariate_mean,
        covariate_rescale=covariate_scale,
        covariate_train_log1p_mean=train_log_mean,
        covariate_train_log1p_scale=train_log_scale,
        affine_cancellation_max_abs_error=maximum_error,
    )


def standardize_so2_nb_expression_input(
    raw_counts: np.ndarray,
    statistics: SO2NBPreprocessingStatistics,
    *,
    row_chunk_size: int = 2_048,
) -> np.ndarray:
    """Create a separate float32 standardized-log input from int32 counts."""

    counts = np.asarray(raw_counts)
    if (
        counts.dtype != np.int32
        or counts.ndim != 2
        or counts.shape[1] != len(statistics.expression_log1p_mean)
        or np.any(counts < 0)
    ):
        raise SO2NBDataContractError(
            "Raw likelihood targets must be aligned nonnegative int32 counts."
        )
    if not isinstance(row_chunk_size, int) or row_chunk_size <= 0:
        raise SO2NBDataContractError("Expression row chunk size must be positive.")
    output = np.empty(counts.shape, dtype=np.float32)
    for start in range(0, len(counts), row_chunk_size):
        stop = min(start + row_chunk_size, len(counts))
        values = np.log1p(counts[start:stop].astype(np.float64, copy=False))
        values = (
            values - statistics.expression_log1p_mean
        ) / statistics.expression_log1p_scale
        output[start:stop] = values
    if not np.isfinite(output).all() or np.shares_memory(output, counts):
        raise SO2NBDataContractError("Standardized expression input is invalid.")
    return output


def recenter_so2_nb_covariates(
    standardized_covariates: np.ndarray,
    statistics: SO2NBPreprocessingStatistics,
) -> np.ndarray:
    values = np.asarray(standardized_covariates)
    if (
        values.ndim != 2
        or values.shape[1] != len(statistics.covariate_recenter_mean)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise SO2NBDataContractError("Source node covariates are invalid.")
    output = (
        values.astype(np.float64, copy=False)
        - statistics.covariate_recenter_mean
    ) / statistics.covariate_rescale
    output = np.asarray(output, dtype=np.float32)
    if not np.isfinite(output).all():
        raise SO2NBDataContractError("Train-recentered node covariates are invalid.")
    return output


def derive_so2_nb_validation_mask_seed(alias: str, view_index: int) -> int:
    canonical = str(alias).strip().upper()
    if canonical not in SO2_NB_VALIDATION_ALIASES:
        raise SO2NBDataContractError("Fixed validation masks require a validation alias.")
    if isinstance(view_index, bool) or not isinstance(view_index, (int, np.integer)):
        raise SO2NBDataContractError("Validation mask view index must be an integer.")
    if int(view_index) not in range(VALIDATION_MASK_VIEW_COUNT):
        raise SO2NBDataContractError("Validation mask view index must be 0 through 9.")
    payload = f"{VALIDATION_MASK_NAMESPACE}|{canonical}|{int(view_index)}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


@dataclass(frozen=True, slots=True)
class SO2NBValidationMask:
    alias: str
    view_index: int
    seed: int
    realization: MaskRealization = field(repr=False)
    receipt_sha256: str

    def __post_init__(self) -> None:
        alias = str(self.alias).strip().upper()
        expected_seed = derive_so2_nb_validation_mask_seed(alias, self.view_index)
        if int(self.seed) != expected_seed or int(self.realization.seed) != expected_seed:
            raise SO2NBDataContractError("Validation mask seed derivation changed.")
        expected_receipt = _canonical_sha256(self._receipt_payload())
        if self.receipt_sha256 != expected_receipt:
            raise SO2NBDataContractError("Validation mask receipt checksum changed.")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "view_index", int(self.view_index))
        object.__setattr__(self, "seed", expected_seed)

    @property
    def mask(self) -> np.ndarray:
        return self.realization.mask

    @property
    def masked_gene_counts(self) -> np.ndarray:
        return self.realization.masked_gene_counts

    @property
    def checksum_sha256(self) -> str:
        return self.realization.checksum

    @property
    def n_masked_entries(self) -> int:
        return int(self.masked_gene_counts.sum(dtype=np.int64))

    def _receipt_payload(self) -> dict[str, Any]:
        counts = self.realization.masked_gene_counts
        return {
            "schema": VALIDATION_MASK_SCHEMA,
            "namespace": VALIDATION_MASK_NAMESPACE,
            "alias": str(self.alias).strip().upper(),
            "view_index": int(self.view_index),
            "seed": int(self.seed),
            "n_cells": int(self.realization.n_cells),
            "n_genes": int(self.realization.num_genes),
            "masked_entry_count": int(counts.sum(dtype=np.int64)),
            "zero_mask_cells": int(np.count_nonzero(counts == 0)),
            "full_mask_cells": int(
                np.count_nonzero(counts == self.realization.num_genes)
            ),
            "masked_gene_counts_sha256": _array_sha256(
                "masked_gene_counts", counts
            ),
            "mask_realization_sha256": self.realization.checksum,
            "sampling": (
                "per_cell_integer_count_uniform_0_through_G_inclusive_then_"
                "positions_without_replacement"
            ),
            "generation_chunk_cells": VALIDATION_MASK_CHUNK_CELLS,
        }

    def to_receipt(self) -> dict[str, Any]:
        payload = self._receipt_payload()
        payload["receipt_sha256"] = self.receipt_sha256
        return payload


def make_so2_nb_validation_mask(
    n_cells: int,
    n_genes: int,
    *,
    alias: str,
    view_index: int,
    chunk_cells: int = VALIDATION_MASK_CHUNK_CELLS,
) -> SO2NBValidationMask:
    if int(chunk_cells) != VALIDATION_MASK_CHUNK_CELLS:
        raise SO2NBDataContractError(
            "Validation-mask generation chunk size is frozen at 2048 cells."
        )
    seed = derive_so2_nb_validation_mask_seed(alias, view_index)
    realization = sample_uniform_mask_numpy(
        n_cells, n_genes, seed=seed, chunk_cells=chunk_cells
    )
    provisional = object.__new__(SO2NBValidationMask)
    object.__setattr__(provisional, "alias", str(alias).strip().upper())
    object.__setattr__(provisional, "view_index", int(view_index))
    object.__setattr__(provisional, "seed", seed)
    object.__setattr__(provisional, "realization", realization)
    receipt_sha256 = _canonical_sha256(provisional._receipt_payload())
    return SO2NBValidationMask(
        alias=alias,
        view_index=view_index,
        seed=seed,
        realization=realization,
        receipt_sha256=receipt_sha256,
    )


@dataclass(frozen=True, slots=True)
class SO2NBCoreBatch:
    """One complete core with expression input and raw target kept distinct."""

    alias: str
    role: Literal["train", "validation"]
    input_expression: Tensor = field(repr=False)
    raw_count_target: Tensor = field(repr=False)
    node_covariates: Tensor = field(repr=False)
    edge_index: Tensor = field(repr=False)
    relative_geometry: Tensor = field(repr=False)

    def __post_init__(self) -> None:
        alias = str(self.alias).strip().upper()
        expected_role = (
            "train" if alias in SO2_NB_TRAINING_ALIASES else "validation"
            if alias in SO2_NB_VALIDATION_ALIASES
            else None
        )
        if expected_role is None or self.role != expected_role:
            raise SO2NBDataContractError("Core alias and train/validation role disagree.")
        expression = torch.as_tensor(self.input_expression)
        target = torch.as_tensor(self.raw_count_target)
        covariates = torch.as_tensor(self.node_covariates)
        edges = torch.as_tensor(self.edge_index)
        geometry = torch.as_tensor(self.relative_geometry)
        if any(
            tensor.device.type != "cpu"
            for tensor in (expression, target, covariates, edges, geometry)
        ):
            raise SO2NBDataContractError("Prepared core tensors must be CPU-resident.")
        if (
            expression.dtype != torch.float32
            or expression.ndim != 2
            or not bool(torch.isfinite(expression).all())
        ):
            raise SO2NBDataContractError("Input expression must be finite float32 [N,G].")
        if (
            target.dtype != torch.int32
            or target.shape != expression.shape
            or bool((target < 0).any())
        ):
            raise SO2NBDataContractError(
                "Raw likelihood target must be aligned nonnegative int32 [N,G]."
            )
        if (
            covariates.dtype != torch.float32
            or covariates.shape != (expression.shape[0], EXPECTED_NODE_COVARIATES)
            or not bool(torch.isfinite(covariates).all())
        ):
            raise SO2NBDataContractError("Node covariates must be finite float32 [N,22].")
        if (
            edges.dtype != torch.long
            or edges.ndim != 2
            or edges.shape[0] != 2
            or geometry.ndim != 2
            or geometry.shape[0] != edges.shape[1]
            or geometry.shape[1] != 70
            or not geometry.is_floating_point()
            or not bool(torch.isfinite(geometry).all())
        ):
            raise SO2NBDataContractError("Core graph tensors are invalid or misaligned.")
        if edges.numel() and (
            bool((edges < 0).any()) or bool((edges >= expression.shape[0]).any())
        ):
            raise SO2NBDataContractError("Core graph contains an out-of-range endpoint.")
        if expression.untyped_storage().data_ptr() == target.untyped_storage().data_ptr():
            raise SO2NBDataContractError("Raw target storage cannot alias model input storage.")
        object.__setattr__(self, "alias", alias)
        object.__setattr__(self, "input_expression", expression)
        object.__setattr__(self, "raw_count_target", target)
        object.__setattr__(self, "node_covariates", covariates)
        object.__setattr__(self, "edge_index", edges)
        object.__setattr__(self, "relative_geometry", geometry)

    @property
    def n_nodes(self) -> int:
        return int(self.input_expression.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.input_expression.shape[1])

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def target_counts(self) -> Tensor:
        return self.raw_count_target


@dataclass(frozen=True, slots=True)
class SO2NBDataBundle:
    manifest_path: Path
    manifest_sha256: str
    manifest_content_sha256: str
    split_fingerprint: str
    preprocessing_fingerprint: str
    statistics: SO2NBPreprocessingStatistics = field(repr=False)
    training_batches: tuple[SO2NBCoreBatch, ...] = field(repr=False)
    validation_batches: tuple[SO2NBCoreBatch, ...] = field(repr=False)
    validation_mask_receipts: Mapping[str, Mapping[str, Any]] = field(repr=False)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("manifest_sha256", self.manifest_sha256),
            ("manifest_content_sha256", self.manifest_content_sha256),
            ("split_fingerprint", self.split_fingerprint),
            ("preprocessing_fingerprint", self.preprocessing_fingerprint),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise SO2NBDataContractError(
                    f"Bundle {field_name} must be a lowercase SHA-256 digest."
                )
        if tuple(batch.alias for batch in self.training_batches) != SO2_NB_TRAINING_ALIASES:
            raise SO2NBDataContractError("Training batches do not match the frozen aliases.")
        if tuple(batch.alias for batch in self.validation_batches) != SO2_NB_VALIDATION_ALIASES:
            raise SO2NBDataContractError("Validation batches do not match the frozen aliases.")
        if self.statistics.training_aliases != SO2_NB_TRAINING_ALIASES:
            raise SO2NBDataContractError(
                "Bundle preprocessing was not fitted on the frozen training aliases."
            )
        object.__setattr__(
            self,
            "validation_mask_receipts",
            MappingProxyType(dict(self.validation_mask_receipts)),
        )

    @property
    def all_batches(self) -> tuple[SO2NBCoreBatch, ...]:
        return self.training_batches + self.validation_batches

    @property
    def batches_by_alias(self) -> Mapping[str, SO2NBCoreBatch]:
        return MappingProxyType({batch.alias: batch for batch in self.all_batches})

    def validation_mask(self, alias: str, view_index: int) -> SO2NBValidationMask:
        mask = make_so2_nb_validation_mask(
            self.batches_by_alias[str(alias).strip().upper()].n_nodes,
            self.batches_by_alias[str(alias).strip().upper()].n_genes,
            alias=alias,
            view_index=view_index,
        )
        key = f"{mask.alias}:{mask.view_index}"
        expected = self.validation_mask_receipts.get(key)
        if not isinstance(expected, Mapping) or dict(expected) != mask.to_receipt():
            raise SO2NBDataContractError("Regenerated validation mask receipt changed.")
        return mask


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    contract_sha256 = sha256_file(path)
    if contract_sha256 != FROZEN_TASK_CONTRACT_SHA256:
        raise SO2NBDataContractError("Frozen SO2 NB task-contract checksum changed.")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SO2NBDataContractError("Cannot read the frozen SO2 NB contract.") from exc
    if not isinstance(value, dict):
        raise SO2NBDataContractError("Frozen SO2 NB contract must be a mapping.")
    split = value.get("split")
    cohort = value.get("cohort")
    preprocessing = value.get("preprocessing")
    masking = value.get("masking")
    if not all(isinstance(item, Mapping) for item in (split, cohort, preprocessing, masking)):
        raise SO2NBDataContractError("Frozen SO2 NB contract sections are incomplete.")
    assert isinstance(split, Mapping)
    assert isinstance(cohort, Mapping)
    assert isinstance(preprocessing, Mapping)
    assert isinstance(masking, Mapping)
    validation_masking = masking.get("validation")
    if not isinstance(validation_masking, Mapping):
        raise SO2NBDataContractError("Frozen validation masking contract is incomplete.")
    expected = {
        "campaign_id": CAMPAIGN_ID,
        "cohort.aliases": list(SO2_ALIASES),
        "cohort.total_cells": EXPECTED_TOTAL_CELLS,
        "cohort.biological_target_count": EXPECTED_N_GENES,
        "cohort.node_covariate_count": EXPECTED_NODE_COVARIATES,
        "split.training_aliases": list(SO2_NB_TRAINING_ALIASES),
        "split.validation_aliases": list(SO2_NB_VALIDATION_ALIASES),
        "split.test_aliases": [],
        "split.training_cells": SO2_NB_TRAINING_CELLS,
        "split.validation_cells": SO2_NB_VALIDATION_CELLS,
        "split.test_cells": 0,
        "split.selection_namespace": SELECTION_NAMESPACE,
        "split.selected_validation_pair_digest": SELECTED_VALIDATION_PAIR_DIGEST,
        "split.protected_grouping_source_sha256": PROTECTED_GROUPING_SOURCE_SHA256,
        "preprocessing.fit_scope": "training_aliases_only",
        "validation.fixed_views_per_core": VALIDATION_MASK_VIEW_COUNT,
        "validation.base_seed_namespace": VALIDATION_MASK_NAMESPACE,
    }
    observed = {
        "campaign_id": value.get("campaign_id"),
        "cohort.aliases": cohort.get("aliases"),
        "cohort.total_cells": cohort.get("total_cells"),
        "cohort.biological_target_count": cohort.get("biological_target_count"),
        "cohort.node_covariate_count": cohort.get("node_covariate_count"),
        "split.training_aliases": split.get("training_aliases"),
        "split.validation_aliases": split.get("validation_aliases"),
        "split.test_aliases": split.get("test_aliases"),
        "split.training_cells": split.get("training_cells"),
        "split.validation_cells": split.get("validation_cells"),
        "split.test_cells": split.get("test_cells"),
        "split.selection_namespace": split.get("selection_namespace"),
        "split.selected_validation_pair_digest": split.get(
            "selected_validation_pair_digest"
        ),
        "split.protected_grouping_source_sha256": split.get(
            "protected_grouping_source_sha256"
        ),
        "preprocessing.fit_scope": preprocessing.get("fit_scope"),
        "validation.fixed_views_per_core": validation_masking.get(
            "fixed_views_per_core"
        ),
        "validation.base_seed_namespace": validation_masking.get(
            "base_seed_namespace"
        ),
    }
    if observed != expected:
        raise SO2NBDataContractError("Frozen SO2 NB data contract changed.")
    return value, contract_sha256


def _source_reference(source: Path, *, overlay_parent: Path) -> str:
    return Path(os.path.relpath(source.resolve(), start=overlay_parent.resolve())).as_posix()


def _split_payload() -> dict[str, Any]:
    return {
        "schema": "so2_nb_donor_grouped_train_validation_split_v1",
        "training_aliases": list(SO2_NB_TRAINING_ALIASES),
        "validation_aliases": list(SO2_NB_VALIDATION_ALIASES),
        "test_aliases": [],
        "training_cells": SO2_NB_TRAINING_CELLS,
        "validation_cells": SO2_NB_VALIDATION_CELLS,
        "test_cells": 0,
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_rule": (
            "minimum_sha256_of_namespace_plus_sorted_core_pair_aliases"
        ),
        "selected_validation_pair_digest": SELECTED_VALIDATION_PAIR_DIGEST,
        "expression_outcomes_used_for_selection": False,
        "pathology_labels_used_for_selection": False,
        "model_results_used_for_selection": False,
        "no_test_partition_or_artifacts": True,
    }


def _inspect_source_arrays_and_fit(
    cohort_root: Path,
    cohort_manifest: Mapping[str, Any],
) -> tuple[SO2NBPreprocessingStatistics, dict[str, Any]]:
    stats_path = cohort_root / "cohort_statistics.npz"
    with np.load(stats_path, allow_pickle=False) as source_stats:
        if "metadata_missing_indicator_indices" not in source_stats.files:
            raise SO2NBDataContractError("Source metadata missingness receipt is absent.")
        missing_indices = np.asarray(source_stats["metadata_missing_indicator_indices"])
        if missing_indices.size != 0:
            raise SO2NBDataContractError("SO2 node covariates unexpectedly contain missingness indicators.")
        global_mean = np.asarray(source_stats["metadata_mean"], dtype=np.float64)
        global_scale = np.asarray(source_stats["metadata_scale"], dtype=np.float64)
    counts_by_alias: dict[str, np.ndarray] = {}
    covariates_by_alias: dict[str, np.ndarray] = {}
    core_records_by_alias = {
        str(record["alias"]): record for record in cohort_manifest["cores"]
    }
    audit: dict[str, Any] = {}
    try:
        for alias in SO2_ALIASES:
            path = cohort_root / "cores" / f"{alias}.npz"
            with np.load(path, allow_pickle=False) as payload:
                required = {
                    "expression_counts",
                    "target_expression",
                    "node_covariates",
                    "coordinates_um",
                }
                if set(payload.files) != required:
                    raise SO2NBDataContractError(
                        f"Source core {alias} has an unexpected array schema."
                    )
                counts = np.array(payload["expression_counts"], copy=True)
                covariates = np.array(payload["node_covariates"], copy=True)
                coordinates = np.asarray(payload["coordinates_um"])
            core_number = int(alias[-2:])
            expected_nodes = EXPECTED_CELL_COUNTS_BY_CORE[core_number]
            if (
                counts.dtype != np.int32
                or counts.shape != (expected_nodes, EXPECTED_N_GENES)
                or np.any(counts < 0)
                or covariates.dtype != np.float32
                or covariates.shape != (expected_nodes, EXPECTED_NODE_COVARIATES)
                or not np.isfinite(covariates).all()
                or coordinates.shape != (expected_nodes, 2)
                or not np.isfinite(coordinates).all()
            ):
                raise SO2NBDataContractError(
                    f"Source core {alias} raw-count/covariate alignment is invalid."
                )
            if alias in SO2_NB_TRAINING_ALIASES:
                counts_by_alias[alias] = counts
                covariates_by_alias[alias] = covariates
            audit[alias] = {
                "role": "train" if alias in SO2_NB_TRAINING_ALIASES else "validation",
                "n_nodes": expected_nodes,
                "n_genes": EXPECTED_N_GENES,
                "n_node_covariates": EXPECTED_NODE_COVARIATES,
                "source_core_file_sha256": cohort_manifest["files"][
                    f"cores/{alias}.npz"
                ],
                "expression_counts_component_sha256": core_records_by_alias[alias][
                    "component_checksums"
                ]["expression_counts"],
                "node_covariates_component_sha256": core_records_by_alias[alias][
                    "component_checksums"
                ]["node_covariates"],
            }
        statistics = fit_so2_nb_preprocessing(
            counts_by_alias,
            covariates_by_alias,
            source_global_covariate_mean=global_mean,
            source_global_covariate_scale=global_scale,
        )
    finally:
        counts_by_alias.clear()
        covariates_by_alias.clear()
    return statistics, audit


def _write_npz(path: Path, statistics: SO2NBPreprocessingStatistics) -> None:
    np.savez(path, **statistics.arrays())
    os.chmod(path, FILE_MODE)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def prepare_so2_nb_train_validation_overlay(
    *,
    cohort_dir: str | Path,
    graph_dir: str | Path,
    protected_grouping_source: str | Path,
    frozen_contract_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Materialize a two-file, checksum-bound overlay without copying source arrays."""

    cohort_root = Path(cohort_dir).resolve()
    graph_root = Path(graph_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("SO2 NB overlay is immutable and will not be overwritten.")
    _, contract_sha256 = _load_contract(Path(frozen_contract_path))
    grouping = verify_protected_so2_grouping(protected_grouping_source)
    cohort_manifest = _verify_so2_cohort_manifest(cohort_root)
    graph_manifest = _verify_so2_graph_collection(
        graph_root, cohort_manifest_path=cohort_root / "manifest.json"
    )
    statistics, core_audit = _inspect_source_arrays_and_fit(
        cohort_root, cohort_manifest
    )
    graph_records = {str(record["alias"]): record for record in graph_manifest["cores"]}
    for alias in SO2_ALIASES:
        graph_qc = graph_records[alias]["graph"]["qc"]
        if int(graph_qc["n_nodes"]) != core_audit[alias]["n_nodes"]:
            raise SO2NBDataContractError(f"Graph node alignment changed for {alias}.")
        core_audit[alias]["n_edges"] = int(graph_qc["n_directed_edges"])
        core_audit[alias]["graph_files"] = dict(graph_records[alias]["files"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    os.chmod(staging, DIRECTORY_MODE)
    try:
        stats_path = staging / "preprocessing.npz"
        _write_npz(stats_path, statistics)
        mask_receipts: dict[str, dict[str, Any]] = {}
        for alias in SO2_NB_VALIDATION_ALIASES:
            n_cells = EXPECTED_CELL_COUNTS_BY_CORE[int(alias[-2:])]
            for view_index in range(VALIDATION_MASK_VIEW_COUNT):
                mask = make_so2_nb_validation_mask(
                    n_cells,
                    EXPECTED_N_GENES,
                    alias=alias,
                    view_index=view_index,
                )
                mask_receipts[f"{alias}:{view_index}"] = mask.to_receipt()
                del mask
        split = _split_payload()
        split_fingerprint = _canonical_sha256(split)
        preprocessing_arrays = {
            name: {
                "shape": list(getattr(statistics, name).shape),
                "dtype": np.asarray(getattr(statistics, name)).dtype.str,
                "sha256": _array_sha256(name, getattr(statistics, name)),
            }
            for name in _STATISTIC_ARRAY_NAMES
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": OVERLAY_SCHEMA,
            "campaign_id": CAMPAIGN_ID,
            "frozen_task_contract_sha256": contract_sha256,
            "split": {**split, "split_fingerprint": split_fingerprint},
            "protected_grouping_verification": grouping.to_receipt(),
            "preprocessing": {
                "schema": PREPROCESSING_SCHEMA,
                "fit_aliases": list(SO2_NB_TRAINING_ALIASES),
                "fit_scope": "training_aliases_only",
                "expression_source": "raw_nonnegative_int32_biological_counts",
                "expression_transform": "log1p_then_gene_wise_standardize",
                "expression_moment_weighting": "equal_core",
                "covariate_source": "existing_standardized_measured_covariates",
                "covariate_transform": "train_equal_core_recenter_and_rescale",
                "covariate_moment_weighting": "equal_core",
                "source_missing_values_present": False,
                "global_affine_transform_cancels": True,
                "affine_cancellation_max_abs_error": (
                    statistics.affine_cancellation_max_abs_error
                ),
                "raw_count_target_transform": "none",
                "raw_count_target_allowed_as_direct_model_input": False,
                "target_derived_library_total_or_offset_computed": False,
                "statistics_reference": "preprocessing.npz",
                "statistics_file_sha256": sha256_file(stats_path),
                "statistics_arrays": preprocessing_arrays,
                "preprocessing_fingerprint": statistics.fingerprint,
            },
            "validation_masks": {
                "schema": VALIDATION_MASK_SCHEMA,
                "namespace": VALIDATION_MASK_NAMESPACE,
                "views_per_core": VALIDATION_MASK_VIEW_COUNT,
                "fixed_across_epochs": True,
                "mask_arrays_persisted": False,
                "receipts": mask_receipts,
            },
            "source_artifacts": {
                "cohort": {
                    "reference": _source_reference(
                        cohort_root, overlay_parent=destination.parent
                    ),
                    "manifest_file_sha256": sha256_file(
                        cohort_root / "manifest.json"
                    ),
                    "manifest_content_sha256": cohort_manifest[
                        "manifest_content_sha256"
                    ],
                },
                "graph": {
                    "reference": _source_reference(
                        graph_root, overlay_parent=destination.parent
                    ),
                    "manifest_file_sha256": sha256_file(graph_root / "manifest.json"),
                    "manifest_content_sha256": graph_manifest[
                        "manifest_content_sha256"
                    ],
                    "reuse_immutable_core_local_geometry": True,
                    "duplicate_graph_artifact": False,
                },
            },
            "cores": core_audit,
            "files": {
                "preprocessing.npz": {
                    "sha256": sha256_file(stats_path),
                    "mode": "0600",
                }
            },
            "permissions": {
                "directory_mode": "0700",
                "file_mode": "0600",
            },
            "storage": {
                "output_files": ["manifest.json", "preprocessing.npz"],
                "source_count_arrays_duplicated": False,
                "source_graph_arrays_duplicated": False,
                "validation_mask_arrays_persisted": False,
            },
            "privacy": {
                "protected_source_used_only_for_group_equality_verification": True,
                "donor_identifier_values_persisted_or_reported": False,
                "direct_cell_identifiers_persisted": False,
            },
        }
        manifest["manifest_content_sha256"] = _canonical_sha256(manifest)
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(
                json.dumps(
                    manifest, sort_keys=True, indent=2, allow_nan=False
                ).encode("utf-8")
            )
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(manifest_path, FILE_MODE)
        if {path.name for path in staging.iterdir()} != {
            "manifest.json",
            "preprocessing.npz",
        }:
            raise SO2NBDataContractError("SO2 NB overlay output inventory changed.")
        staging.rename(destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return manifest
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


def _statistics_from_overlay(
    root: Path, manifest: Mapping[str, Any]
) -> SO2NBPreprocessingStatistics:
    preprocessing = manifest.get("preprocessing")
    files = manifest.get("files")
    if not isinstance(preprocessing, Mapping) or not isinstance(files, Mapping):
        raise SO2NBDataContractError("Overlay preprocessing receipt is incomplete.")
    expected_semantics = {
        "schema": PREPROCESSING_SCHEMA,
        "fit_aliases": list(SO2_NB_TRAINING_ALIASES),
        "fit_scope": "training_aliases_only",
        "expression_source": "raw_nonnegative_int32_biological_counts",
        "expression_transform": "log1p_then_gene_wise_standardize",
        "expression_moment_weighting": "equal_core",
        "covariate_source": "existing_standardized_measured_covariates",
        "covariate_transform": "train_equal_core_recenter_and_rescale",
        "covariate_moment_weighting": "equal_core",
        "source_missing_values_present": False,
        "global_affine_transform_cancels": True,
        "raw_count_target_transform": "none",
        "raw_count_target_allowed_as_direct_model_input": False,
        "target_derived_library_total_or_offset_computed": False,
    }
    if any(preprocessing.get(key) != value for key, value in expected_semantics.items()):
        raise SO2NBDataContractError("Overlay preprocessing semantics changed.")
    relative = str(preprocessing.get("statistics_reference", ""))
    if relative != "preprocessing.npz":
        raise SO2NBDataContractError("Overlay statistics reference changed.")
    path = root / relative
    file_record = files.get(relative)
    if not isinstance(file_record, Mapping) or sha256_file(path) != file_record.get("sha256"):
        raise SO2NBDataContractError("Overlay statistics file checksum changed.")
    if stat.S_IMODE(path.stat().st_mode) != FILE_MODE:
        raise SO2NBDataContractError("Overlay statistics file permissions changed.")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != set(_STATISTIC_ARRAY_NAMES):
            raise SO2NBDataContractError("Overlay statistics array schema changed.")
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    result = SO2NBPreprocessingStatistics(
        training_aliases=tuple(preprocessing.get("fit_aliases", ())),
        affine_cancellation_max_abs_error=float(
            preprocessing.get("affine_cancellation_max_abs_error", float("nan"))
        ),
        **arrays,
    )
    if result.fingerprint != preprocessing.get("preprocessing_fingerprint"):
        raise SO2NBDataContractError("Overlay preprocessing fingerprint changed.")
    array_records = preprocessing.get("statistics_arrays")
    if not isinstance(array_records, Mapping):
        raise SO2NBDataContractError("Overlay statistic-array receipts are absent.")
    for name in _STATISTIC_ARRAY_NAMES:
        record = array_records.get(name)
        values = getattr(result, name)
        if not isinstance(record, Mapping) or dict(record) != {
            "shape": list(values.shape),
            "dtype": np.asarray(values).dtype.str,
            "sha256": _array_sha256(name, values),
        }:
            raise SO2NBDataContractError("Overlay statistic-array receipt changed.")
    return result


def _validate_overlay_semantics(root: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("frozen_task_contract_sha256") != FROZEN_TASK_CONTRACT_SHA256:
        raise SO2NBDataContractError("Overlay is bound to a different frozen contract.")
    if {path.name for path in root.iterdir()} != {
        "manifest.json",
        "preprocessing.npz",
    }:
        raise SO2NBDataContractError(
            "Overlay contains files outside the reference-only inventory."
        )
    permissions = manifest.get("permissions")
    storage = manifest.get("storage")
    privacy = manifest.get("privacy")
    grouping = manifest.get("protected_grouping_verification")
    masks = manifest.get("validation_masks")
    if not all(
        isinstance(item, Mapping)
        for item in (permissions, storage, privacy, grouping, masks)
    ):
        raise SO2NBDataContractError("Overlay safety receipts are incomplete.")
    assert isinstance(permissions, Mapping)
    assert isinstance(storage, Mapping)
    assert isinstance(privacy, Mapping)
    assert isinstance(grouping, Mapping)
    assert isinstance(masks, Mapping)
    if dict(permissions) != {"directory_mode": "0700", "file_mode": "0600"}:
        raise SO2NBDataContractError("Overlay permission contract changed.")
    if dict(storage) != {
        "output_files": ["manifest.json", "preprocessing.npz"],
        "source_count_arrays_duplicated": False,
        "source_graph_arrays_duplicated": False,
        "validation_mask_arrays_persisted": False,
    }:
        raise SO2NBDataContractError("Overlay reference-only storage contract changed.")
    if dict(privacy) != {
        "protected_source_used_only_for_group_equality_verification": True,
        "donor_identifier_values_persisted_or_reported": False,
        "direct_cell_identifiers_persisted": False,
    }:
        raise SO2NBDataContractError("Overlay privacy contract changed.")
    expected_grouping = {
        "source_sha256": PROTECTED_GROUPING_SOURCE_SHA256,
        "donor_group_count": 7,
        "cores_per_group": 2,
        "core_pairs": [list(pair) for pair in EXPECTED_DONOR_GROUP_PAIRS],
        "equivalence_classes_match_expected_core_pairs": True,
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_rule": (
            "minimum_sha256_of_namespace_plus_sorted_core_pair_aliases"
        ),
        "selected_validation_pair_digest": SELECTED_VALIDATION_PAIR_DIGEST,
        "donor_identifier_values_persisted_or_reported": False,
    }
    if dict(grouping) != expected_grouping:
        raise SO2NBDataContractError("Overlay protected-grouping receipt changed.")
    receipts = masks.get("receipts")
    expected_keys = {
        f"{alias}:{view_index}"
        for alias in SO2_NB_VALIDATION_ALIASES
        for view_index in range(VALIDATION_MASK_VIEW_COUNT)
    }
    if (
        masks.get("schema") != VALIDATION_MASK_SCHEMA
        or masks.get("namespace") != VALIDATION_MASK_NAMESPACE
        or masks.get("views_per_core") != VALIDATION_MASK_VIEW_COUNT
        or masks.get("fixed_across_epochs") is not True
        or masks.get("mask_arrays_persisted") is not False
        or not isinstance(receipts, Mapping)
        or set(receipts) != expected_keys
    ):
        raise SO2NBDataContractError("Overlay validation-mask contract changed.")
    assert isinstance(receipts, Mapping)
    for key in sorted(expected_keys):
        alias, raw_view = key.split(":", 1)
        receipt = receipts[key]
        if not isinstance(receipt, Mapping):
            raise SO2NBDataContractError("Overlay validation-mask receipt is malformed.")
        receipt_payload = dict(receipt)
        receipt_sha256 = receipt_payload.pop("receipt_sha256", None)
        if (
            receipt.get("schema") != VALIDATION_MASK_SCHEMA
            or receipt.get("namespace") != VALIDATION_MASK_NAMESPACE
            or receipt.get("alias") != alias
            or receipt.get("view_index") != int(raw_view)
            or receipt.get("seed")
            != derive_so2_nb_validation_mask_seed(alias, int(raw_view))
            or receipt.get("n_cells")
            != EXPECTED_CELL_COUNTS_BY_CORE[int(alias[-2:])]
            or receipt.get("n_genes") != EXPECTED_N_GENES
            or receipt.get("generation_chunk_cells")
            != VALIDATION_MASK_CHUNK_CELLS
            or not isinstance(receipt_sha256, str)
            or receipt_sha256 != _canonical_sha256(receipt_payload)
        ):
            raise SO2NBDataContractError("Overlay validation-mask receipt changed.")


def _resolve_source_root(
    overlay_root: Path,
    record: Mapping[str, Any],
    explicit: str | Path | None,
) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    reference = record.get("reference")
    if not isinstance(reference, str) or not reference:
        raise SO2NBDataContractError("Overlay source reference is absent.")
    return (overlay_root.parent / reference).resolve()


def _load_core_batch(
    alias: str,
    *,
    role: Literal["train", "validation"],
    cohort_root: Path,
    graph_root: Path,
    statistics: SO2NBPreprocessingStatistics,
) -> SO2NBCoreBatch:
    with np.load(cohort_root / "cores" / f"{alias}.npz", allow_pickle=False) as payload:
        counts = np.array(payload["expression_counts"], dtype=np.int32, copy=True)
        if payload["expression_counts"].dtype != np.int32:
            raise SO2NBDataContractError(f"Raw count dtype changed for {alias}.")
        source_covariates = np.array(
            payload["node_covariates"], dtype=np.float32, copy=True
        )
    input_expression = standardize_so2_nb_expression_input(counts, statistics)
    node_covariates = recenter_so2_nb_covariates(source_covariates, statistics)
    edge_map = np.load(
        graph_root / "cores" / alias / "edge_index.npy", mmap_mode="r"
    )
    geometry_map = np.load(
        graph_root / "cores" / alias / "relative_geometry.npy", mmap_mode="r"
    )
    # The two mappings are immutable source caches.  PyTorch retains the mmap
    # storage without copying the 16 GiB graph collection.
    edge_tensor = torch.from_numpy(edge_map)
    geometry_tensor = torch.from_numpy(geometry_map)
    return SO2NBCoreBatch(
        alias=alias,
        role=role,
        input_expression=torch.from_numpy(input_expression),
        raw_count_target=torch.from_numpy(counts),
        node_covariates=torch.from_numpy(node_covariates),
        edge_index=edge_tensor,
        relative_geometry=geometry_tensor,
    )


def load_so2_nb_data(
    overlay_dir: str | Path,
    *,
    cohort_dir: str | Path | None = None,
    graph_dir: str | Path | None = None,
) -> SO2NBDataBundle:
    """Load train/validation batches from a verified reference-only overlay."""

    root = Path(overlay_dir).resolve()
    manifest_path = root / "manifest.json"
    if stat.S_IMODE(root.stat().st_mode) != DIRECTORY_MODE:
        raise SO2NBDataContractError("SO2 NB overlay directory permissions changed.")
    if stat.S_IMODE(manifest_path.stat().st_mode) != FILE_MODE:
        raise SO2NBDataContractError("SO2 NB overlay manifest permissions changed.")
    manifest = _strict_json(manifest_path)
    content_sha = manifest.get("manifest_content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_content_sha256", None)
    if content_sha != _canonical_sha256(unsigned):
        raise SO2NBDataContractError("SO2 NB overlay manifest checksum changed.")
    if manifest.get("artifact_kind") != OVERLAY_SCHEMA or manifest.get("campaign_id") != CAMPAIGN_ID:
        raise SO2NBDataContractError("SO2 NB overlay identity changed.")
    _validate_overlay_semantics(root, manifest)
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise SO2NBDataContractError("SO2 NB overlay split receipt is absent.")
    expected_split = _split_payload()
    observed_split = {key: split.get(key) for key in expected_split}
    if observed_split != expected_split or split.get("split_fingerprint") != _canonical_sha256(expected_split):
        raise SO2NBDataContractError("SO2 NB overlay split changed.")
    source_records = manifest.get("source_artifacts")
    if not isinstance(source_records, Mapping):
        raise SO2NBDataContractError("SO2 NB source references are absent.")
    cohort_record = source_records.get("cohort")
    graph_record = source_records.get("graph")
    if not isinstance(cohort_record, Mapping) or not isinstance(graph_record, Mapping):
        raise SO2NBDataContractError("SO2 NB source reference schema changed.")
    cohort_root = _resolve_source_root(root, cohort_record, cohort_dir)
    graph_root = _resolve_source_root(root, graph_record, graph_dir)
    if sha256_file(cohort_root / "manifest.json") != cohort_record.get("manifest_file_sha256"):
        raise SO2NBDataContractError("Referenced SO2 cohort manifest changed.")
    if sha256_file(graph_root / "manifest.json") != graph_record.get("manifest_file_sha256"):
        raise SO2NBDataContractError("Referenced SO2 graph manifest changed.")
    cohort_manifest = _verify_so2_cohort_manifest(cohort_root)
    graph_manifest = _verify_so2_graph_collection(
        graph_root, cohort_manifest_path=cohort_root / "manifest.json"
    )
    if (
        cohort_manifest.get("manifest_content_sha256")
        != cohort_record.get("manifest_content_sha256")
        or graph_manifest.get("manifest_content_sha256")
        != graph_record.get("manifest_content_sha256")
    ):
        raise SO2NBDataContractError("Referenced source content fingerprint changed.")
    statistics = _statistics_from_overlay(root, manifest)
    training = tuple(
        _load_core_batch(
            alias,
            role="train",
            cohort_root=cohort_root,
            graph_root=graph_root,
            statistics=statistics,
        )
        for alias in SO2_NB_TRAINING_ALIASES
    )
    validation = tuple(
        _load_core_batch(
            alias,
            role="validation",
            cohort_root=cohort_root,
            graph_root=graph_root,
            statistics=statistics,
        )
        for alias in SO2_NB_VALIDATION_ALIASES
    )
    validation_masks = manifest.get("validation_masks")
    if not isinstance(validation_masks, Mapping) or not isinstance(
        validation_masks.get("receipts"), Mapping
    ):
        raise SO2NBDataContractError("Validation mask receipts are absent.")
    return SO2NBDataBundle(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        manifest_content_sha256=str(content_sha),
        split_fingerprint=str(split["split_fingerprint"]),
        preprocessing_fingerprint=statistics.fingerprint,
        statistics=statistics,
        training_batches=training,
        validation_batches=validation,
        validation_mask_receipts=dict(validation_masks["receipts"]),
    )


__all__ = [
    "CAMPAIGN_ID",
    "EXPECTED_DONOR_GROUP_PAIRS",
    "FROZEN_TASK_CONTRACT_SHA256",
    "OVERLAY_SCHEMA",
    "PREPROCESSING_SCHEMA",
    "PROTECTED_GROUPING_SOURCE_SHA256",
    "SELECTED_VALIDATION_PAIR_DIGEST",
    "SELECTION_NAMESPACE",
    "SO2NBCoreBatch",
    "SO2NBDataBundle",
    "SO2NBDataContractError",
    "SO2NBPreprocessingStatistics",
    "SO2NBValidationMask",
    "SO2_NB_TEST_ALIASES",
    "SO2_NB_TRAINING_ALIASES",
    "SO2_NB_VALIDATION_ALIASES",
    "VALIDATION_MASK_NAMESPACE",
    "VALIDATION_MASK_CHUNK_CELLS",
    "VALIDATION_MASK_VIEW_COUNT",
    "derive_so2_nb_validation_mask_seed",
    "fit_so2_nb_preprocessing",
    "load_so2_nb_data",
    "make_so2_nb_validation_mask",
    "prepare_so2_nb_train_validation_overlay",
    "recenter_so2_nb_covariates",
    "standardize_so2_nb_expression_input",
    "validation_pair_digest",
    "verify_protected_so2_grouping",
]
