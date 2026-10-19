"""Pure-NumPy non-learned controls for masked-expression evaluation.

The predictors in this module are deliberately fitted and applied in two
separate steps.  Only training rows may be supplied to ``fit``.  Prediction is
then performed on one split-local coordinate/expression/mask triple at a time,
so a validation or test call has no candidate nodes from another split.

Masks use the benchmark convention: ``True`` means that an expression entry is
hidden.  The nearest-neighbor control reads only candidate entries whose mask
is ``False``.  Values stored under masked entries can therefore be changed
without changing a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


class DiagnosticContractError(ValueError):
    """Raised when a diagnostic would violate its fit/prediction contract."""


def _readonly_copy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.flags.writeable = False
    return array


def _expression_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    is_real_numeric = np.issubdtype(
        array.dtype, np.floating
    ) or np.issubdtype(array.dtype, np.integer)
    if array.ndim != 2 or not is_real_numeric:
        raise DiagnosticContractError(
            f"{name} must be a numeric [cells, genes] array."
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DiagnosticContractError(f"{name} dimensions must be non-empty.")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise DiagnosticContractError(f"{name} must contain only finite values.")
    return result


def _split_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise DiagnosticContractError(
            "split_mask must be boolean and match [cells, genes]."
        )
    return mask


def _coordinates(value: Any, n_cells: int) -> np.ndarray:
    coordinates = np.asarray(value, dtype=np.float64)
    if coordinates.shape != (n_cells, 2):
        raise DiagnosticContractError(
            "coordinates_um must have shape [cells, 2]."
        )
    if not np.isfinite(coordinates).all():
        raise DiagnosticContractError(
            "coordinates_um must contain only finite values."
        )
    return coordinates


def _split_name(value: object) -> str:
    name = str(value).strip()
    if not name:
        raise DiagnosticContractError("split_name must be non-empty.")
    return name


@dataclass(frozen=True)
class DiagnosticPrediction:
    """One metrics-ready split-local diagnostic prediction.

    ``source_node_index`` is split-local and is populated only at evaluated
    (masked) entries copied from a visible neighbor.  ``-1`` means the
    train-global mean was used.  Entries outside ``evaluation_mask`` also use
    the train mean and intentionally carry no source.
    """

    control: str
    split_name: str
    predictions: np.ndarray
    evaluation_mask: np.ndarray
    source_node_index: np.ndarray
    source_distance_um: np.ndarray
    train_gene_mean: np.ndarray
    n_training_cells: int
    min_distance_um: float | None

    def __post_init__(self) -> None:
        control = str(self.control).strip()
        split_name = _split_name(self.split_name)
        predictions = np.asarray(self.predictions, dtype=np.float64)
        mask = np.asarray(self.evaluation_mask)
        sources = np.asarray(self.source_node_index)
        distances = np.asarray(self.source_distance_um, dtype=np.float64)
        means = np.asarray(self.train_gene_mean, dtype=np.float64)
        if not control:
            raise DiagnosticContractError("control must be non-empty.")
        if predictions.ndim != 2 or not np.isfinite(predictions).all():
            raise DiagnosticContractError(
                "predictions must be finite [cells, genes]."
            )
        if mask.dtype != np.bool_ or mask.shape != predictions.shape:
            raise DiagnosticContractError(
                "evaluation_mask must be boolean and match predictions."
            )
        if (
            sources.shape != predictions.shape
            or not np.issubdtype(sources.dtype, np.integer)
        ):
            raise DiagnosticContractError(
                "source_node_index must be integer and match predictions."
            )
        if distances.shape != predictions.shape:
            raise DiagnosticContractError(
                "source_distance_um must match predictions."
            )
        if means.shape != (predictions.shape[1],) or not np.isfinite(means).all():
            raise DiagnosticContractError(
                "train_gene_mean must be finite and match the gene dimension."
            )
        if (
            not isinstance(self.n_training_cells, (int, np.integer))
            or isinstance(self.n_training_cells, (bool, np.bool_))
            or int(self.n_training_cells) <= 0
        ):
            raise DiagnosticContractError("n_training_cells must be positive.")
        copied = mask & (sources >= 0)
        if np.any(sources[~mask] != -1) or np.any(sources[mask] < -1):
            raise DiagnosticContractError(
                "sources may be recorded only for evaluated entries."
            )
        if np.any(sources[copied] >= predictions.shape[0]):
            raise DiagnosticContractError(
                "source_node_index contains an out-of-range split-local node."
            )
        if (
            np.any(~np.isfinite(distances[copied]))
            or np.any(distances[copied] < 0)
            or np.any(~np.isnan(distances[~copied]))
        ):
            raise DiagnosticContractError(
                "source distances must be finite for copies and NaN otherwise."
            )
        minimum = self.min_distance_um
        if minimum is not None:
            if isinstance(minimum, (bool, np.bool_)):
                raise DiagnosticContractError(
                    "min_distance_um must be finite and non-negative."
                )
            minimum = float(minimum)
            if not math.isfinite(minimum) or minimum < 0:
                raise DiagnosticContractError(
                    "min_distance_um must be finite and non-negative."
                )
            if np.any(distances[copied] + 1e-12 < minimum):
                raise DiagnosticContractError(
                    "a copied source violates min_distance_um."
                )

        object.__setattr__(self, "control", control)
        object.__setattr__(self, "split_name", split_name)
        object.__setattr__(
            self, "predictions", _readonly_copy(predictions, dtype=np.float64)
        )
        object.__setattr__(
            self, "evaluation_mask", _readonly_copy(mask, dtype=bool)
        )
        object.__setattr__(
            self, "source_node_index", _readonly_copy(sources, dtype=np.int64)
        )
        object.__setattr__(
            self,
            "source_distance_um",
            _readonly_copy(distances, dtype=np.float64),
        )
        object.__setattr__(
            self, "train_gene_mean", _readonly_copy(means, dtype=np.float64)
        )
        object.__setattr__(
            self, "n_training_cells", int(self.n_training_cells)
        )
        object.__setattr__(self, "min_distance_um", minimum)

    @property
    def fit_scope(self) -> str:
        return "training cells only"

    @property
    def fallback_mask(self) -> np.ndarray:
        result = self.evaluation_mask & (self.source_node_index < 0)
        result.flags.writeable = False
        return result

    @property
    def copied_mask(self) -> np.ndarray:
        result = self.evaluation_mask & (self.source_node_index >= 0)
        result.flags.writeable = False
        return result

    @property
    def n_evaluated_entries(self) -> int:
        return int(self.evaluation_mask.sum())

    @property
    def n_copied_entries(self) -> int:
        return int(self.copied_mask.sum())

    @property
    def n_fallback_entries(self) -> int:
        return int(self.fallback_mask.sum())

    @property
    def copy_rate(self) -> float:
        if self.n_evaluated_entries == 0:
            return 0.0
        return self.n_copied_entries / self.n_evaluated_entries

    def metrics_inputs(self, target_expression: Any) -> dict[str, np.ndarray]:
        """Return keyword arguments for ``evaluate_masked_predictions``."""

        target = np.asarray(target_expression)
        if target.shape != self.predictions.shape:
            raise DiagnosticContractError(
                "target_expression must match prediction shape."
            )
        if not np.isfinite(np.asarray(target, dtype=np.float64)[self.evaluation_mask]).all():
            raise DiagnosticContractError(
                "evaluated target_expression entries must be finite."
            )
        return {
            "y_true": target,
            "predictions": self.predictions,
            "mask": self.evaluation_mask,
        }


@dataclass(frozen=True)
class TrainGlobalMeanPredictor:
    """Per-gene mean fitted exclusively from transformed training rows."""

    gene_mean: np.ndarray
    n_training_cells: int

    def __post_init__(self) -> None:
        means = np.asarray(self.gene_mean, dtype=np.float64)
        if means.ndim != 1 or len(means) == 0 or not np.isfinite(means).all():
            raise DiagnosticContractError(
                "gene_mean must be a non-empty finite vector."
            )
        if (
            not isinstance(self.n_training_cells, (int, np.integer))
            or isinstance(self.n_training_cells, (bool, np.bool_))
            or int(self.n_training_cells) <= 0
        ):
            raise DiagnosticContractError("n_training_cells must be positive.")
        object.__setattr__(
            self, "gene_mean", _readonly_copy(means, dtype=np.float64)
        )
        object.__setattr__(
            self, "n_training_cells", int(self.n_training_cells)
        )

    @classmethod
    def fit(cls, train_expression: Any) -> "TrainGlobalMeanPredictor":
        training = _expression_matrix(
            train_expression, name="train_expression"
        )
        return cls(
            gene_mean=training.mean(axis=0, dtype=np.float64),
            n_training_cells=training.shape[0],
        )

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_mean))

    def predict(
        self,
        split_mask: Any,
        *,
        split_name: str = "split",
    ) -> DiagnosticPrediction:
        mask = np.asarray(split_mask)
        if (
            mask.ndim != 2
            or mask.dtype != np.bool_
            or mask.shape[1] != self.n_genes
            or mask.shape[0] == 0
        ):
            raise DiagnosticContractError(
                "split_mask must be boolean [split cells, fitted genes]."
            )
        predictions = np.broadcast_to(
            self.gene_mean, mask.shape
        ).astype(np.float64, copy=True)
        return DiagnosticPrediction(
            control="train_global_gene_mean",
            split_name=split_name,
            predictions=predictions,
            evaluation_mask=mask,
            source_node_index=np.full(mask.shape, -1, dtype=np.int64),
            source_distance_um=np.full(mask.shape, np.nan, dtype=np.float64),
            train_gene_mean=self.gene_mean,
            n_training_cells=self.n_training_cells,
            min_distance_um=None,
        )


def _automatic_distance_block_size(n_cells: int) -> int:
    # Delta uses two float64 coordinates and the squared-distance matrix uses
    # one more float64 per pair.  Keep those arrays near 64 MiB.
    bytes_per_pair = 3 * np.dtype(np.float64).itemsize
    target_bytes = 64 * 1024 * 1024
    return max(1, min(n_cells, target_bytes // (bytes_per_pair * n_cells)))


@dataclass(frozen=True)
class NearestSpatialNeighborCopyPredictor:
    """Nearest visible-gene copy control with train-mean fallback."""

    train_mean_predictor: TrainGlobalMeanPredictor

    def __post_init__(self) -> None:
        if not isinstance(
            self.train_mean_predictor, TrainGlobalMeanPredictor
        ):
            raise DiagnosticContractError(
                "train_mean_predictor must be fitted on training expression."
            )

    @classmethod
    def fit(
        cls, train_expression: Any
    ) -> "NearestSpatialNeighborCopyPredictor":
        return cls(TrainGlobalMeanPredictor.fit(train_expression))

    def predict(
        self,
        coordinates_um: Any,
        transformed_expression: Any,
        split_mask: Any,
        *,
        split_name: str = "split",
        min_distance_um: float = 0.0,
        distance_block_size: int | None = None,
    ) -> DiagnosticPrediction:
        """Predict masked entries from their nearest split-local visible source.

        The nearest source is selected independently for each masked
        cell/gene entry because candidate visibility is gene-specific.  Ties
        are resolved by the smallest split-local row index.  If no other node
        has a visible copy of that gene at or beyond ``min_distance_um``, the
        fitted training mean is retained.
        """

        expression = _expression_matrix(
            transformed_expression, name="transformed_expression"
        )
        if expression.shape[1] != self.train_mean_predictor.n_genes:
            raise DiagnosticContractError(
                "transformed_expression gene dimension differs from training."
            )
        mask = _split_mask(split_mask, expression.shape)
        coordinates = _coordinates(coordinates_um, expression.shape[0])
        if isinstance(min_distance_um, (bool, np.bool_)):
            raise DiagnosticContractError(
                "min_distance_um must be finite and non-negative."
            )
        minimum = float(min_distance_um)
        if not math.isfinite(minimum) or minimum < 0:
            raise DiagnosticContractError(
                "min_distance_um must be finite and non-negative."
            )
        if distance_block_size is None:
            block_size = _automatic_distance_block_size(len(expression))
        else:
            if (
                not isinstance(distance_block_size, (int, np.integer))
                or isinstance(distance_block_size, (bool, np.bool_))
                or int(distance_block_size) <= 0
            ):
                raise DiagnosticContractError(
                    "distance_block_size must be a positive integer."
                )
            block_size = min(int(distance_block_size), len(expression))

        base = self.train_mean_predictor.predict(
            mask, split_name=_split_name(split_name)
        )
        predictions = np.array(base.predictions, copy=True)
        sources = np.full(mask.shape, -1, dtype=np.int64)
        source_distances = np.full(mask.shape, np.nan, dtype=np.float64)
        globally_visible = (~mask).any(axis=0)
        minimum_squared = minimum * minimum

        target_rows = np.flatnonzero(mask.any(axis=1)).astype(np.int64)
        for start in range(0, len(target_rows), block_size):
            rows = target_rows[start : start + block_size]
            delta = (
                coordinates[rows, None, :]
                - coordinates[None, :, :]
            )
            squared_distance = np.einsum(
                "bij,bij->bi", delta, delta, optimize=True
            )
            squared_distance[
                np.arange(len(rows), dtype=np.int64), rows
            ] = np.inf
            if minimum > 0:
                squared_distance[squared_distance < minimum_squared] = np.inf

            unresolved = mask[rows].copy()
            unresolved &= globally_visible[None, :]
            while np.any(unresolved):
                active_rows = np.flatnonzero(unresolved.any(axis=1))
                nearest = np.argmin(
                    squared_distance[active_rows], axis=1
                ).astype(np.int64, copy=False)
                nearest_squared = squared_distance[
                    active_rows, nearest
                ]
                has_candidate = np.isfinite(nearest_squared)
                if not np.any(has_candidate):
                    break

                local_rows = active_rows[has_candidate]
                candidate_nodes = nearest[has_candidate]
                candidate_squared = nearest_squared[has_candidate]
                can_copy = (
                    unresolved[local_rows]
                    & ~mask[candidate_nodes]
                )
                copy_rows, copy_genes = np.nonzero(can_copy)
                if len(copy_rows):
                    selected_local_rows = local_rows[copy_rows]
                    selected_sources = candidate_nodes[copy_rows]
                    # Index only entries already certified visible by can_copy;
                    # hidden candidate values are never read.
                    predictions[
                        rows[selected_local_rows], copy_genes
                    ] = expression[selected_sources, copy_genes]
                    sources[
                        rows[selected_local_rows], copy_genes
                    ] = selected_sources
                    source_distances[
                        rows[selected_local_rows], copy_genes
                    ] = np.sqrt(candidate_squared[copy_rows])
                    unresolved[selected_local_rows, copy_genes] = False

                squared_distance[
                    local_rows, candidate_nodes
                ] = np.inf
                no_candidate_rows = active_rows[~has_candidate]
                if len(no_candidate_rows):
                    unresolved[no_candidate_rows] = False

        return DiagnosticPrediction(
            control="nearest_spatial_neighbor_copy",
            split_name=split_name,
            predictions=predictions,
            evaluation_mask=mask,
            source_node_index=sources,
            source_distance_um=source_distances,
            train_gene_mean=self.train_mean_predictor.gene_mean,
            n_training_cells=self.train_mean_predictor.n_training_cells,
            min_distance_um=minimum,
        )


def fit_train_global_mean(
    train_expression: Any,
) -> TrainGlobalMeanPredictor:
    """Fit the reusable train-only per-gene mean control."""

    return TrainGlobalMeanPredictor.fit(train_expression)


def train_global_mean_prediction(
    train_expression: Any,
    split_mask: Any,
    *,
    split_name: str = "split",
) -> DiagnosticPrediction:
    """Fit on training rows and return one split's global-mean prediction."""

    return TrainGlobalMeanPredictor.fit(train_expression).predict(
        split_mask, split_name=split_name
    )


def nearest_spatial_neighbor_copy_prediction(
    train_expression: Any,
    coordinates_um: Any,
    transformed_expression: Any,
    split_mask: Any,
    *,
    split_name: str = "split",
    min_distance_um: float = 0.0,
    distance_block_size: int | None = None,
) -> DiagnosticPrediction:
    """Fit the fallback on train and run the split-local nearest-copy control."""

    return NearestSpatialNeighborCopyPredictor.fit(
        train_expression
    ).predict(
        coordinates_um,
        transformed_expression,
        split_mask,
        split_name=split_name,
        min_distance_um=min_distance_um,
        distance_block_size=distance_block_size,
    )


__all__ = [
    "DiagnosticContractError",
    "DiagnosticPrediction",
    "NearestSpatialNeighborCopyPredictor",
    "TrainGlobalMeanPredictor",
    "fit_train_global_mean",
    "nearest_spatial_neighbor_copy_prediction",
    "train_global_mean_prediction",
]
