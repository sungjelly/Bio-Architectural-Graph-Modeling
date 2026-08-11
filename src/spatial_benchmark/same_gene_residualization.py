"""Train-only nuisance residualization for same-gene robustness analyses.

The fits in this module are deliberately small weighted least-squares models.
Expression matrices remain in their original dtype; only bounded row/gene
chunks are promoted to float64 while sufficient statistics are accumulated.
No fit may inspect held-out outcomes because every outcome cross-product is
indexed by the explicit boolean training mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np


class SameGeneResidualizationError(RuntimeError):
    """Raised when a residualization contract is unsafe or rank deficient."""


Label: TypeAlias = str | int | float


def _positive_chunk_size(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SameGeneResidualizationError(f"{label} must be a positive integer")
    result = int(value)
    if result < 1:
        raise SameGeneResidualizationError(f"{label} must be a positive integer")
    return result


def _numeric_matrix(
    values: np.ndarray,
    *,
    label: str,
    row_chunk_size: int,
) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise SameGeneResidualizationError(f"{label} must have shape [N,G]")
    if matrix.dtype.kind not in "iuf":
        raise SameGeneResidualizationError(f"{label} must be a real numeric matrix")
    for start in range(0, matrix.shape[0], row_chunk_size):
        if not bool(np.isfinite(matrix[start : start + row_chunk_size]).all()):
            raise SameGeneResidualizationError(f"{label} contains nonfinite values")
    return matrix


def _numeric_vector(values: np.ndarray, *, rows: int, label: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (rows,) or raw.dtype.kind not in "iuf":
        raise SameGeneResidualizationError(f"{label} must have shape [N]")
    result = raw.astype(np.float64, copy=False)
    if not bool(np.isfinite(result).all()):
        raise SameGeneResidualizationError(f"{label} contains nonfinite values")
    return result


def _boolean_train_mask(values: np.ndarray, *, rows: int) -> np.ndarray:
    mask = np.asarray(values)
    if mask.shape != (rows,) or mask.dtype != np.bool_:
        raise SameGeneResidualizationError(
            "train_mask must be an explicit boolean vector with shape [N]"
        )
    if not bool(mask.any()):
        raise SameGeneResidualizationError("train_mask must select at least one row")
    return mask


def _component_vector(values: np.ndarray, *, rows: int) -> np.ndarray:
    components = np.asarray(values)
    if components.shape != (rows,):
        raise SameGeneResidualizationError("components must have shape [N]")
    if components.dtype.kind in "iu":
        return components
    if components.dtype.kind == "f":
        if not bool(np.isfinite(components).all()):
            raise SameGeneResidualizationError("components contain nonfinite labels")
        return components
    if components.dtype.kind in "US":
        converted = components.astype(str, copy=False)
        if bool(np.any(np.char.str_len(converted) == 0)):
            raise SameGeneResidualizationError("components contain empty labels")
        return converted
    raise SameGeneResidualizationError(
        "components must contain finite numeric or nonempty string labels"
    )


def _cell_type_vector(values: np.ndarray, *, rows: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (rows,):
        raise SameGeneResidualizationError("cell_types must have shape [N]")
    if raw.dtype.kind == "U":
        result = raw
    elif raw.dtype.kind == "S":
        result = raw.astype(str)
    elif raw.dtype.kind == "O" and all(isinstance(value, str) for value in raw):
        result = raw.astype(str)
    else:
        raise SameGeneResidualizationError(
            "cell_types must contain only nonempty string labels"
        )
    if bool(np.any(np.char.str_len(result) == 0)):
        raise SameGeneResidualizationError("cell_types contain empty labels")
    return result


def _frozen_levels(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SameGeneResidualizationError("levels must be an explicit sequence")
    levels = tuple(values)
    if not levels or any(not isinstance(value, str) or not value for value in levels):
        raise SameGeneResidualizationError(
            "levels must contain nonempty strings"
        )
    if len(set(levels)) != len(levels):
        raise SameGeneResidualizationError("levels must be unique and ordered")
    return levels


def component_equal_train_weights(
    components: np.ndarray,
    train_mask: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return normalized train-only weights with equal mass per train component.

    Supplied weights may vary within a component, but they must be positive on
    every training row, exactly zero outside the training mask, and assign the
    same total mass to every component represented in training.
    """

    mask_raw = np.asarray(train_mask)
    if mask_raw.ndim != 1:
        raise SameGeneResidualizationError("train_mask must have shape [N]")
    mask = _boolean_train_mask(mask_raw, rows=len(mask_raw))
    group = _component_vector(components, rows=len(mask))
    levels = np.unique(group[mask])
    if len(levels) < 1:
        raise SameGeneResidualizationError("training has no component coverage")

    if weights is None:
        result = np.zeros(len(mask), dtype=np.float64)
        for level in levels:
            selected = mask & (group == level)
            count = int(selected.sum())
            if count < 1:
                raise SameGeneResidualizationError(
                    "a training component has no selected rows"
                )
            result[selected] = 1.0 / (len(levels) * count)
    else:
        result = _numeric_vector(weights, rows=len(mask), label="weights").copy()
        if bool(np.any(result < 0)):
            raise SameGeneResidualizationError("weights must be nonnegative")
        if bool(np.any(result[~mask] != 0)):
            raise SameGeneResidualizationError(
                "weights must be exactly zero outside train_mask"
            )
        if bool(np.any(result[mask] <= 0)):
            raise SameGeneResidualizationError(
                "every training row must have positive weight"
            )
        total = float(result.sum(dtype=np.float64))
        if not np.isfinite(total) or total <= 0:
            raise SameGeneResidualizationError("training weights have no mass")
        result /= total

    masses = np.asarray(
        [result[mask & (group == level)].sum(dtype=np.float64) for level in levels],
        dtype=np.float64,
    )
    expected = np.full(len(levels), 1.0 / len(levels), dtype=np.float64)
    if not np.allclose(masses, expected, rtol=1e-10, atol=1e-12):
        raise SameGeneResidualizationError(
            "training weights are not component-equal or lack component coverage"
        )
    if (
        bool(np.any(result[~mask] != 0))
        or bool(np.any(result[mask] <= 0))
        or not np.isclose(result.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise SameGeneResidualizationError("train-only weight assertion failed")
    result.setflags(write=False)
    return result


def _assert_full_rank(matrix: np.ndarray, *, columns: int, label: str) -> None:
    if matrix.shape != (columns, columns) or not bool(np.isfinite(matrix).all()):
        raise SameGeneResidualizationError(f"{label} normal matrix is invalid")
    rank = int(np.linalg.matrix_rank(matrix))
    if rank != columns:
        raise SameGeneResidualizationError(
            f"{label} design is rank deficient: rank {rank}, expected {columns}"
        )


def _readonly_float64(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float64)
    if not bool(np.isfinite(result).all()):
        raise SameGeneResidualizationError("fitted coefficients are nonfinite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class LibraryWLSFit:
    """Per-gene intercept and log-library slope from component-equal WLS."""

    intercept: np.ndarray
    library_slope: np.ndarray
    training_row_count: int
    training_component_count: int
    weighted_library_mean: float

    @property
    def gene_count(self) -> int:
        return int(self.intercept.shape[0])


@dataclass(frozen=True, slots=True)
class CellTypeLibraryWLSFit:
    """Frozen type intercepts plus a per-gene shared log-library slope."""

    levels: tuple[str, ...]
    type_intercepts: np.ndarray
    global_intercept: np.ndarray
    library_slope: np.ndarray
    type_weight_mass: np.ndarray
    training_row_count: int
    training_component_count: int
    weighted_library_mean: float

    @property
    def gene_count(self) -> int:
        return int(self.global_intercept.shape[0])


ResidualizationFit: TypeAlias = LibraryWLSFit | CellTypeLibraryWLSFit


def _fit_dimensions(
    expression: np.ndarray,
    log_library: np.ndarray,
    components: np.ndarray,
    train_mask: np.ndarray,
    *,
    weights: np.ndarray | None,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_chunk = _positive_chunk_size(row_chunk_size, label="row_chunk_size")
    outcome = _numeric_matrix(
        expression, label="expression", row_chunk_size=row_chunk
    )
    rows = outcome.shape[0]
    library = _numeric_vector(log_library, rows=rows, label="log_library")
    mask = _boolean_train_mask(train_mask, rows=rows)
    group = _component_vector(components, rows=rows)
    fit_weights = component_equal_train_weights(group, mask, weights=weights)
    return outcome, library, group, mask, fit_weights


def fit_component_equal_library_wls(
    expression: np.ndarray,
    log_library: np.ndarray,
    components: np.ndarray,
    *,
    train_mask: np.ndarray,
    weights: np.ndarray | None = None,
    row_chunk_size: int = 4096,
    gene_chunk_size: int = 128,
) -> LibraryWLSFit:
    """Fit ``expression_g ~ intercept_g + slope_g * log_library`` on train rows."""

    row_chunk = _positive_chunk_size(row_chunk_size, label="row_chunk_size")
    gene_chunk = _positive_chunk_size(gene_chunk_size, label="gene_chunk_size")
    outcome, library, group, mask, fit_weights = _fit_dimensions(
        expression,
        log_library,
        components,
        train_mask,
        weights=weights,
        row_chunk_size=row_chunk,
    )
    positions = np.flatnonzero(mask)
    selected_weights = fit_weights[positions]
    selected_library = library[positions]
    weighted_library_mean = float(
        np.dot(selected_weights, selected_library)
    )
    normal = np.asarray(
        [
            [1.0, weighted_library_mean],
            [
                weighted_library_mean,
                float(np.dot(selected_weights, selected_library * selected_library)),
            ],
        ],
        dtype=np.float64,
    )
    _assert_full_rank(normal, columns=2, label="library WLS")

    coefficients = np.empty((2, outcome.shape[1]), dtype=np.float64)
    for gene_start in range(0, outcome.shape[1], gene_chunk):
        gene_stop = min(gene_start + gene_chunk, outcome.shape[1])
        rhs = np.zeros((2, gene_stop - gene_start), dtype=np.float64)
        for row_start in range(0, len(positions), row_chunk):
            index = positions[row_start : row_start + row_chunk]
            batch = np.asarray(
                outcome[index, gene_start:gene_stop], dtype=np.float64
            )
            batch_weight = fit_weights[index]
            batch_library = library[index]
            rhs[0] += np.einsum("i,ij->j", batch_weight, batch, optimize=True)
            rhs[1] += np.einsum(
                "i,ij->j", batch_weight * batch_library, batch, optimize=True
            )
        coefficients[:, gene_start:gene_stop] = np.linalg.solve(normal, rhs)

    component_count = int(len(np.unique(group[mask])))
    return LibraryWLSFit(
        intercept=_readonly_float64(coefficients[0]),
        library_slope=_readonly_float64(coefficients[1]),
        training_row_count=int(mask.sum()),
        training_component_count=component_count,
        weighted_library_mean=weighted_library_mean,
    )


def fit_cell_type_library_wls(
    expression: np.ndarray,
    log_library: np.ndarray,
    cell_types: np.ndarray,
    components: np.ndarray,
    *,
    levels: Sequence[str],
    train_mask: np.ndarray,
    weights: np.ndarray | None = None,
    row_chunk_size: int = 4096,
    gene_chunk_size: int = 128,
) -> CellTypeLibraryWLSFit:
    """Fit K frozen type intercepts and one shared per-gene library slope.

    Every frozen level must occur in training and every training label must be
    frozen.  Unseen labels are allowed only when applying the fitted model.
    """

    row_chunk = _positive_chunk_size(row_chunk_size, label="row_chunk_size")
    gene_chunk = _positive_chunk_size(gene_chunk_size, label="gene_chunk_size")
    frozen = _frozen_levels(levels)
    outcome, library, group, mask, fit_weights = _fit_dimensions(
        expression,
        log_library,
        components,
        train_mask,
        weights=weights,
        row_chunk_size=row_chunk,
    )
    types = _cell_type_vector(cell_types, rows=outcome.shape[0])
    level_index = {level: index for index, level in enumerate(frozen)}
    encoded = np.fromiter(
        (level_index.get(value, -1) for value in types),
        dtype=np.int64,
        count=len(types),
    )
    if bool(np.any(encoded[mask] < 0)):
        unseen = sorted(set(types[mask][encoded[mask] < 0].tolist()))
        raise SameGeneResidualizationError(
            f"training contains cell types outside frozen levels: {unseen}"
        )
    type_mass = np.bincount(
        encoded[mask], weights=fit_weights[mask], minlength=len(frozen)
    ).astype(np.float64, copy=False)
    if type_mass.shape != (len(frozen),) or bool(np.any(type_mass <= 0)):
        missing = [
            level for level, mass in zip(frozen, type_mass, strict=True) if mass <= 0
        ]
        raise SameGeneResidualizationError(
            f"frozen cell-type levels lack training coverage: {missing}"
        )

    positions = np.flatnonzero(mask)
    selected_weights = fit_weights[positions]
    selected_library = library[positions]
    selected_types = encoded[positions]
    cross = np.bincount(
        selected_types,
        weights=selected_weights * selected_library,
        minlength=len(frozen),
    ).astype(np.float64, copy=False)
    normal = np.zeros((len(frozen) + 1, len(frozen) + 1), dtype=np.float64)
    normal[np.arange(len(frozen)), np.arange(len(frozen))] = type_mass
    normal[:-1, -1] = cross
    normal[-1, :-1] = cross
    normal[-1, -1] = float(
        np.dot(selected_weights, selected_library * selected_library)
    )
    _assert_full_rank(
        normal, columns=len(frozen) + 1, label="cell-type + library WLS"
    )

    coefficients = np.empty(
        (len(frozen) + 1, outcome.shape[1]), dtype=np.float64
    )
    for gene_start in range(0, outcome.shape[1], gene_chunk):
        gene_stop = min(gene_start + gene_chunk, outcome.shape[1])
        rhs = np.zeros(
            (len(frozen) + 1, gene_stop - gene_start), dtype=np.float64
        )
        for row_start in range(0, len(positions), row_chunk):
            index = positions[row_start : row_start + row_chunk]
            batch = np.asarray(
                outcome[index, gene_start:gene_stop], dtype=np.float64
            )
            batch_weight = fit_weights[index]
            batch_library = library[index]
            batch_types = encoded[index]
            for type_index in np.unique(batch_types):
                selected = batch_types == type_index
                rhs[int(type_index)] += np.einsum(
                    "i,ij->j",
                    batch_weight[selected],
                    batch[selected],
                    optimize=True,
                )
            rhs[-1] += np.einsum(
                "i,ij->j", batch_weight * batch_library, batch, optimize=True
            )
        coefficients[:, gene_start:gene_stop] = np.linalg.solve(normal, rhs)

    type_intercepts = coefficients[:-1]
    slope = coefficients[-1]
    global_intercept = np.einsum(
        "k,kg->g", type_mass, type_intercepts, optimize=True
    )
    weighted_library_mean = float(np.dot(selected_weights, selected_library))
    component_count = int(len(np.unique(group[mask])))
    return CellTypeLibraryWLSFit(
        levels=frozen,
        type_intercepts=_readonly_float64(type_intercepts),
        global_intercept=_readonly_float64(global_intercept),
        library_slope=_readonly_float64(slope),
        type_weight_mass=_readonly_float64(type_mass),
        training_row_count=int(mask.sum()),
        training_component_count=component_count,
        weighted_library_mean=weighted_library_mean,
    )


fit_component_equal_cell_type_library_wls = fit_cell_type_library_wls


def _validate_fit(fit: ResidualizationFit) -> int:
    if isinstance(fit, LibraryWLSFit):
        arrays = (fit.intercept, fit.library_slope)
        if fit.intercept.ndim != 1 or fit.library_slope.shape != fit.intercept.shape:
            raise SameGeneResidualizationError("library WLS fit has invalid shapes")
        genes = len(fit.intercept)
    elif isinstance(fit, CellTypeLibraryWLSFit):
        genes = len(fit.global_intercept)
        arrays = (
            fit.type_intercepts,
            fit.global_intercept,
            fit.library_slope,
            fit.type_weight_mass,
        )
        if (
            not fit.levels
            or fit.type_intercepts.shape != (len(fit.levels), genes)
            or fit.library_slope.shape != (genes,)
            or fit.type_weight_mass.shape != (len(fit.levels),)
            or bool(np.any(fit.type_weight_mass <= 0))
            or not np.isclose(fit.type_weight_mass.sum(), 1.0, atol=1e-12)
        ):
            raise SameGeneResidualizationError("cell-type WLS fit has invalid shapes")
    else:
        raise SameGeneResidualizationError("unsupported residualization fit")
    if genes < 1 or not all(bool(np.isfinite(value).all()) for value in arrays):
        raise SameGeneResidualizationError("residualization fit is nonfinite")
    return genes


def _output_dtype(value: np.dtype | type[np.floating]) -> np.dtype:
    dtype = np.dtype(value)
    if dtype.kind != "f" or dtype.itemsize < 4:
        raise SameGeneResidualizationError(
            "output_dtype must be float32 or a wider floating dtype"
        )
    return dtype


def _residual_inputs(
    expression: np.ndarray,
    log_library: np.ndarray,
    fit: ResidualizationFit,
    *,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    row_chunk = _positive_chunk_size(row_chunk_size, label="row_chunk_size")
    outcome = _numeric_matrix(
        expression, label="expression", row_chunk_size=row_chunk
    )
    genes = _validate_fit(fit)
    if outcome.shape[1] != genes:
        raise SameGeneResidualizationError(
            "expression gene dimension does not match fitted coefficients"
        )
    library = _numeric_vector(
        log_library, rows=outcome.shape[0], label="log_library"
    )
    return outcome, library, row_chunk


def _store_residual(
    output: np.ndarray,
    row_slice: slice,
    gene_slice: slice,
    values: np.ndarray,
) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        converted = values.astype(output.dtype, copy=False)
    if not bool(np.isfinite(converted).all()):
        raise SameGeneResidualizationError("residual output is nonfinite")
    output[row_slice, gene_slice] = converted


def apply_target_residual(
    fit: ResidualizationFit,
    expression: np.ndarray,
    log_library: np.ndarray,
    *,
    cell_types: np.ndarray | None = None,
    row_chunk_size: int = 4096,
    gene_chunk_size: int = 128,
    output_dtype: np.dtype | type[np.floating] = np.float32,
) -> np.ndarray:
    """Apply a frozen fit to per-cell targets without refitting any coefficient."""

    gene_chunk = _positive_chunk_size(gene_chunk_size, label="gene_chunk_size")
    outcome, library, row_chunk = _residual_inputs(
        expression, log_library, fit, row_chunk_size=row_chunk_size
    )
    dtype = _output_dtype(output_dtype)
    output = np.empty(outcome.shape, dtype=dtype)

    encoded: np.ndarray | None = None
    if isinstance(fit, CellTypeLibraryWLSFit):
        if cell_types is None:
            raise SameGeneResidualizationError(
                "cell_types are required for a cell-type WLS fit"
            )
        types = _cell_type_vector(cell_types, rows=outcome.shape[0])
        mapping = {level: index for index, level in enumerate(fit.levels)}
        encoded = np.fromiter(
            (mapping.get(value, -1) for value in types),
            dtype=np.int64,
            count=len(types),
        )
    elif cell_types is not None:
        raise SameGeneResidualizationError(
            "cell_types must be omitted for a library-only WLS fit"
        )

    for row_start in range(0, outcome.shape[0], row_chunk):
        row_stop = min(row_start + row_chunk, outcome.shape[0])
        row_slice = slice(row_start, row_stop)
        batch_library = library[row_slice]
        for gene_start in range(0, outcome.shape[1], gene_chunk):
            gene_stop = min(gene_start + gene_chunk, outcome.shape[1])
            gene_slice = slice(gene_start, gene_stop)
            batch = np.asarray(outcome[row_slice, gene_slice], dtype=np.float64)
            if isinstance(fit, LibraryWLSFit):
                prediction = fit.intercept[gene_slice] + (
                    batch_library[:, None] * fit.library_slope[gene_slice]
                )
            else:
                assert encoded is not None
                batch_types = encoded[row_slice]
                intercept = np.empty(batch.shape, dtype=np.float64)
                known = batch_types >= 0
                if bool(known.any()):
                    intercept[known] = fit.type_intercepts[
                        batch_types[known], gene_slice
                    ]
                if bool((~known).any()):
                    intercept[~known] = fit.global_intercept[gene_slice]
                prediction = intercept + (
                    batch_library[:, None] * fit.library_slope[gene_slice]
                )
            _store_residual(
                output, row_slice, gene_slice, batch - prediction
            )
    return output


def apply_neighbor_mean_residual(
    fit: ResidualizationFit,
    neighbor_mean_expression: np.ndarray,
    neighbor_mean_log_library: np.ndarray,
    *,
    type_proportions: np.ndarray | None = None,
    type_proportion_levels: Sequence[str] | None = None,
    row_chunk_size: int = 4096,
    gene_chunk_size: int = 128,
    output_dtype: np.dtype | type[np.floating] = np.float32,
) -> np.ndarray:
    """Residualize neighbor means using mean covariates and frozen coefficients.

    For a cell-type fit, columns of ``type_proportions`` must be explicitly
    bound to ``fit.levels``.  A row sum below one denotes unseen-type neighbor
    mass, which receives the global weighted intercept.  Tiny floating excess
    above one (at most 1e-6) is normalized; larger excess fails closed.
    """

    gene_chunk = _positive_chunk_size(gene_chunk_size, label="gene_chunk_size")
    outcome, library, row_chunk = _residual_inputs(
        neighbor_mean_expression,
        neighbor_mean_log_library,
        fit,
        row_chunk_size=row_chunk_size,
    )
    dtype = _output_dtype(output_dtype)
    output = np.empty(outcome.shape, dtype=dtype)

    proportions: np.ndarray | None = None
    row_mass: np.ndarray | None = None
    if isinstance(fit, CellTypeLibraryWLSFit):
        if type_proportions is None or type_proportion_levels is None:
            raise SameGeneResidualizationError(
                "cell-type neighbor residuals require proportions and ordered levels"
            )
        supplied_levels = _frozen_levels(type_proportion_levels)
        if supplied_levels != fit.levels:
            raise SameGeneResidualizationError(
                "type-proportion columns do not match frozen fitted levels"
            )
        proportions = _numeric_matrix(
            type_proportions,
            label="type_proportions",
            row_chunk_size=row_chunk,
        )
        if proportions.shape != (outcome.shape[0], len(fit.levels)):
            raise SameGeneResidualizationError(
                "type_proportions must have shape [N,K]"
            )
        if bool(np.any(proportions < 0)):
            raise SameGeneResidualizationError(
                "type_proportions must be nonnegative"
            )
        row_mass = proportions.sum(axis=1, dtype=np.float64)
        if bool(np.any(row_mass > 1.0 + 1e-6)):
            raise SameGeneResidualizationError(
                "type-proportion row mass may not exceed one"
            )
    elif type_proportions is not None or type_proportion_levels is not None:
        raise SameGeneResidualizationError(
            "type proportions must be omitted for a library-only WLS fit"
        )

    for row_start in range(0, outcome.shape[0], row_chunk):
        row_stop = min(row_start + row_chunk, outcome.shape[0])
        row_slice = slice(row_start, row_stop)
        batch_library = library[row_slice]
        for gene_start in range(0, outcome.shape[1], gene_chunk):
            gene_stop = min(gene_start + gene_chunk, outcome.shape[1])
            gene_slice = slice(gene_start, gene_stop)
            batch = np.asarray(outcome[row_slice, gene_slice], dtype=np.float64)
            if isinstance(fit, LibraryWLSFit):
                prediction = fit.intercept[gene_slice] + (
                    batch_library[:, None] * fit.library_slope[gene_slice]
                )
            else:
                assert proportions is not None and row_mass is not None
                batch_proportions = np.asarray(
                    proportions[row_slice], dtype=np.float64
                ).copy()
                batch_mass = row_mass[row_slice]
                excess = batch_mass > 1.0
                if bool(excess.any()):
                    batch_proportions[excess] /= batch_mass[excess, None]
                bounded_mass = np.minimum(batch_mass, 1.0)
                intercept = (
                    batch_proportions @ fit.type_intercepts[:, gene_slice]
                ) + (
                    (1.0 - bounded_mass)[:, None]
                    * fit.global_intercept[gene_slice]
                )
                prediction = intercept + (
                    batch_library[:, None] * fit.library_slope[gene_slice]
                )
            _store_residual(
                output, row_slice, gene_slice, batch - prediction
            )
    return output


__all__ = [
    "CellTypeLibraryWLSFit",
    "LibraryWLSFit",
    "SameGeneResidualizationError",
    "apply_neighbor_mean_residual",
    "apply_target_residual",
    "component_equal_train_weights",
    "fit_cell_type_library_wls",
    "fit_component_equal_cell_type_library_wls",
    "fit_component_equal_library_wls",
]
