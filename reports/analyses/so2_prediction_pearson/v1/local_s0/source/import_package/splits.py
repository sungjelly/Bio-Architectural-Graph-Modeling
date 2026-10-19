"""Deterministic spatial splits and train-only node preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

from .data import ALLOWED_METADATA_COLUMNS, CoreDataset, DataContractError


_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SpatialSplit:
    """Aligned split labels and spatial macroblock identifiers."""

    labels: np.ndarray
    macroblock_ids: np.ndarray
    seed: int
    block_size_um: float
    fov_aware: bool

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        blocks = np.asarray(self.macroblock_ids)
        if labels.ndim != 1 or blocks.ndim != 1 or labels.shape != blocks.shape:
            raise ValueError("Split labels and macroblock IDs must be aligned 1D arrays.")
        unknown = set(labels.astype(str).tolist()).difference(_SPLIT_NAMES)
        if unknown:
            raise ValueError("Split labels contain unknown values.")
        if any(not np.any(labels == name) for name in _SPLIT_NAMES):
            raise ValueError("Train, validation, and test must all be non-empty.")
        # A spatial block is an uncertainty unit and cannot span splits.
        for block in np.unique(blocks):
            if np.unique(labels[blocks == block]).size != 1:
                raise ValueError("A spatial macroblock crosses split boundaries.")

    @property
    def train_mask(self) -> np.ndarray:
        return self.labels == "train"

    @property
    def val_mask(self) -> np.ndarray:
        return self.labels == "val"

    @property
    def test_mask(self) -> np.ndarray:
        return self.labels == "test"

    @property
    def split_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.labels, dtype="U5").tobytes())
        digest.update(np.asarray(self.macroblock_ids, dtype="U64").tobytes())
        digest.update(str(self.seed).encode("ascii"))
        digest.update(repr(float(self.block_size_um)).encode("ascii"))
        return digest.hexdigest()[:16]

    def assert_fovs_disjoint(self, fov: Sequence[object]) -> None:
        values = np.asarray(fov)
        if values.shape != self.labels.shape:
            raise ValueError("FOV values must align to split labels.")
        for value in np.unique(values):
            if np.unique(self.labels[values == value]).size != 1:
                raise ValueError("An FOV crosses split boundaries.")


def _validate_spatial_inputs(
    coordinates_um: np.ndarray,
    fov: Sequence[object] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_um must have shape [n_cells, 2].")
    if coordinates.shape[0] < 3 or not np.isfinite(coordinates).all():
        raise ValueError("At least three cells with finite coordinates are required.")
    fov_values: np.ndarray | None = None
    if fov is not None:
        fov_values = np.asarray(fov)
        if fov_values.ndim != 1 or len(fov_values) != len(coordinates):
            raise ValueError("FOV values must be a 1D array aligned to cells.")
        if any(value is None for value in fov_values.tolist()):
            raise ValueError("FOV values cannot be missing.")
    return coordinates, fov_values


def _macroblock_ids(
    coordinates: np.ndarray,
    block_size_um: float,
    *,
    fov: np.ndarray | None,
    qualify_by_fov: bool,
) -> np.ndarray:
    grid = np.floor(coordinates / block_size_um).astype(np.int64)
    if qualify_by_fov and fov is not None:
        return np.asarray(
            [
                f"fov={value}|x={x_value}|y={y_value}"
                for value, (x_value, y_value) in zip(fov.tolist(), grid.tolist())
            ],
            dtype="U96",
        )
    return np.asarray(
        [f"x={x_value}|y={y_value}" for x_value, y_value in grid.tolist()],
        dtype="U64",
    )


def _summarize_units(
    coordinates: np.ndarray,
    unit_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unit_text = np.asarray([str(value) for value in unit_values], dtype="U128")
    unique_units = np.unique(unit_text)
    centroids = np.empty((len(unique_units), 2), dtype=np.float64)
    sizes = np.empty(len(unique_units), dtype=np.int64)
    for idx, unit in enumerate(unique_units):
        mask = unit_text == unit
        centroids[idx] = coordinates[mask].mean(axis=0)
        sizes[idx] = int(mask.sum())
    return unique_units, centroids, sizes


def _corner_anchor(centroids: np.ndarray, seed: int) -> int:
    directions = np.asarray(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    )
    direction = directions[int(seed) % len(directions)]
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    scores = centered @ direction
    # np.argmax returns the first index; units were lexically sorted.
    return int(np.argmax(scores))


def _grow_contiguous_region(
    centroids: np.ndarray,
    sizes: np.ndarray,
    candidates: np.ndarray,
    *,
    anchor: int,
    target_cells: int,
) -> np.ndarray:
    selected = [int(anchor)]
    remaining = set(int(value) for value in candidates.tolist())
    remaining.discard(int(anchor))
    total = int(sizes[anchor])
    while remaining and total < target_cells:
        selected_centroids = centroids[np.asarray(selected)]
        ranked: list[tuple[float, int]] = []
        for candidate in remaining:
            distance = np.linalg.norm(
                selected_centroids - centroids[candidate], axis=1
            ).min()
            ranked.append((float(distance), candidate))
        _, next_unit = min(ranked, key=lambda item: (item[0], item[1]))
        selected.append(next_unit)
        remaining.remove(next_unit)
        total += int(sizes[next_unit])
    return np.asarray(sorted(selected), dtype=np.int64)


def make_spatial_split(
    coordinates_um: np.ndarray,
    *,
    fov: Sequence[object] | None = None,
    block_size_um: float = 300.0,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    fov_aware: bool = True,
) -> SpatialSplit:
    """Create deterministic, contiguous held-out regions before preprocessing.

    With FOV values and ``fov_aware=True`` (the default), entire FOVs are the
    assignment units and therefore cannot cross splits.  Macroblocks remain the
    reporting/resampling units and are FOV-qualified at seams.  Without FOV
    grouping, entire global-coordinate macroblocks are assigned together.
    """

    coordinates, fov_values = _validate_spatial_inputs(coordinates_um, fov)
    if not np.isfinite(block_size_um) or block_size_um <= 0:
        raise ValueError("block_size_um must be positive and finite.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("Validation and test fractions must be between zero and one.")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave training cells.")
    if fov_aware and fov_values is None:
        raise ValueError("fov_aware=True requires explicit FOV values.")

    blocks = _macroblock_ids(
        coordinates,
        float(block_size_um),
        fov=fov_values,
        qualify_by_fov=fov_aware,
    )
    units = fov_values if fov_aware else blocks
    assert units is not None
    unique_units, centroids, sizes = _summarize_units(coordinates, units)
    if len(unique_units) < 3:
        raise ValueError("At least three spatial assignment units are required.")

    all_indices = np.arange(len(unique_units), dtype=np.int64)
    n_cells = len(coordinates)
    test_target = max(1, int(round(test_fraction * n_cells)))
    val_target = max(1, int(round(val_fraction * n_cells)))

    test_anchor = _corner_anchor(centroids, seed)
    test_indices = _grow_contiguous_region(
        centroids,
        sizes,
        all_indices,
        anchor=test_anchor,
        target_cells=test_target,
    )
    remaining_after_test = np.setdiff1d(all_indices, test_indices, assume_unique=True)
    if len(remaining_after_test) < 2:
        raise ValueError("Test grouping leaves too few units for validation and training.")

    test_centroid = np.average(
        centroids[test_indices], axis=0, weights=sizes[test_indices]
    )
    distance_from_test = np.linalg.norm(
        centroids[remaining_after_test] - test_centroid, axis=1
    )
    # Farthest remaining unit keeps the two held-out regions distinct.
    val_anchor = int(
        remaining_after_test[
            np.argmax(distance_from_test + np.arange(len(distance_from_test)) * 0.0)
        ]
    )
    val_indices = _grow_contiguous_region(
        centroids,
        sizes,
        remaining_after_test,
        anchor=val_anchor,
        target_cells=val_target,
    )
    train_indices = np.setdiff1d(
        remaining_after_test, val_indices, assume_unique=True
    )
    if len(train_indices) == 0:
        raise ValueError("Spatial grouping leaves no training unit.")

    assignment = np.full(len(unique_units), "train", dtype="U5")
    assignment[val_indices] = "val"
    assignment[test_indices] = "test"
    unit_to_split = {
        unit: assignment[idx] for idx, unit in enumerate(unique_units.tolist())
    }
    unit_text = np.asarray([str(value) for value in units], dtype="U128")
    labels = np.asarray([unit_to_split[value] for value in unit_text], dtype="U5")
    result = SpatialSplit(
        labels=labels,
        macroblock_ids=blocks,
        seed=int(seed),
        block_size_um=float(block_size_um),
        fov_aware=bool(fov_aware),
    )
    if fov_aware:
        result.assert_fovs_disjoint(fov_values)
    return result


@dataclass(frozen=True)
class PreprocessedNodes:
    expression: np.ndarray
    metadata: np.ndarray
    metadata_names: tuple[str, ...]


class TrainOnlyPreprocessor:
    """Train-fitted log/scale transforms for expression and allowed metadata."""

    def __init__(
        self,
        *,
        log1p_metadata: bool = True,
        epsilon: float = 1e-8,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        self.log1p_metadata = bool(log1p_metadata)
        self.epsilon = float(epsilon)
        self.is_fitted_ = False

    @staticmethod
    def _train_mask(
        split: SpatialSplit | Sequence[object] | np.ndarray,
        n_rows: int,
    ) -> np.ndarray:
        if isinstance(split, SpatialSplit):
            labels = split.labels
        else:
            labels = np.asarray(split)
        if labels.ndim != 1 or len(labels) != n_rows:
            raise ValueError("Split labels must align to node rows.")
        mask = labels.astype(str) == "train"
        if not mask.any():
            raise ValueError("At least one training row is required.")
        return mask

    def fit(
        self,
        expression: np.ndarray,
        metadata: np.ndarray,
        split: SpatialSplit | Sequence[object] | np.ndarray,
        *,
        metadata_names: Sequence[str] = ALLOWED_METADATA_COLUMNS,
    ) -> "TrainOnlyPreprocessor":
        """Fit every learned statistic using training rows only."""

        counts = np.asarray(expression)
        covariates = np.asarray(metadata, dtype=np.float64)
        if counts.ndim != 2 or covariates.ndim != 2:
            raise ValueError("Expression and metadata must be 2D.")
        if counts.shape[0] != covariates.shape[0]:
            raise ValueError("Expression and metadata row counts must match.")
        if covariates.shape[1] != len(metadata_names):
            raise ValueError("Metadata names do not match the metadata matrix.")
        if tuple(metadata_names) != ALLOWED_METADATA_COLUMNS:
            raise DataContractError("Only the exact metadata allow-list may be fitted.")
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Expression must contain finite nonnegative counts.")

        train_mask = self._train_mask(split, len(counts))
        train_expression = np.log1p(counts[train_mask].astype(np.float64, copy=False))
        self.expression_mean_ = train_expression.mean(axis=0)
        expression_scale = train_expression.std(axis=0, ddof=0)
        self.expression_scale_ = np.where(
            expression_scale > self.epsilon, expression_scale, 1.0
        )

        train_metadata = covariates[train_mask]
        finite_or_nan = np.isfinite(train_metadata) | np.isnan(train_metadata)
        if not finite_or_nan.all():
            raise ValueError("Metadata can contain NaN but not infinite values.")
        with np.errstate(all="ignore"):
            medians = np.nanmedian(train_metadata, axis=0)
        if np.isnan(medians).any():
            raise ValueError("A metadata column is entirely missing in training data.")
        self.metadata_median_ = medians
        self.missing_indicator_indices_ = np.flatnonzero(
            np.isnan(train_metadata).any(axis=0)
        )
        imputed = np.where(np.isnan(train_metadata), medians, train_metadata)
        if self.log1p_metadata:
            if np.any(imputed < 0):
                raise ValueError("log1p metadata transform requires nonnegative values.")
            imputed = np.log1p(imputed)
        self.metadata_mean_ = imputed.mean(axis=0)
        metadata_scale = imputed.std(axis=0, ddof=0)
        self.metadata_scale_ = np.where(
            metadata_scale > self.epsilon, metadata_scale, 1.0
        )
        self.metadata_names_ = tuple(metadata_names) + tuple(
            f"{metadata_names[index]}__missing"
            for index in self.missing_indicator_indices_.tolist()
        )
        self.n_train_ = int(train_mask.sum())
        self.n_expression_features_ = counts.shape[1]
        self.is_fitted_ = True
        return self

    def fit_dataset(
        self,
        dataset: CoreDataset,
        split: SpatialSplit | Sequence[object] | np.ndarray,
    ) -> "TrainOnlyPreprocessor":
        return self.fit(
            dataset.expression,
            dataset.metadata,
            split,
            metadata_names=dataset.metadata_names,
        )

    def transform(
        self,
        expression: np.ndarray,
        metadata: np.ndarray,
    ) -> PreprocessedNodes:
        if not self.is_fitted_:
            raise RuntimeError("TrainOnlyPreprocessor must be fitted before transform.")
        counts = np.asarray(expression)
        covariates = np.asarray(metadata, dtype=np.float64)
        if counts.ndim != 2 or counts.shape[1] != self.n_expression_features_:
            raise ValueError("Expression feature count differs from fitted data.")
        if covariates.shape != (len(counts), len(ALLOWED_METADATA_COLUMNS)):
            raise ValueError("Metadata shape differs from the fixed fitted schema.")
        if not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError("Expression must contain finite nonnegative counts.")
        finite_or_nan = np.isfinite(covariates) | np.isnan(covariates)
        if not finite_or_nan.all():
            raise ValueError("Metadata can contain NaN but not infinite values.")

        transformed_expression = (
            np.log1p(counts.astype(np.float64, copy=False)) - self.expression_mean_
        ) / self.expression_scale_
        missing = np.isnan(covariates)
        imputed = np.where(missing, self.metadata_median_, covariates)
        if self.log1p_metadata:
            if np.any(imputed < 0):
                raise ValueError("log1p metadata transform requires nonnegative values.")
            imputed = np.log1p(imputed)
        transformed_metadata = (
            imputed - self.metadata_mean_
        ) / self.metadata_scale_
        if len(self.missing_indicator_indices_):
            indicators = missing[:, self.missing_indicator_indices_].astype(np.float64)
            transformed_metadata = np.concatenate(
                [transformed_metadata, indicators], axis=1
            )
        return PreprocessedNodes(
            expression=transformed_expression.astype(np.float32, copy=False),
            metadata=transformed_metadata.astype(np.float32, copy=False),
            metadata_names=self.metadata_names_,
        )

    def fit_transform(
        self,
        expression: np.ndarray,
        metadata: np.ndarray,
        split: SpatialSplit | Sequence[object] | np.ndarray,
        *,
        metadata_names: Sequence[str] = ALLOWED_METADATA_COLUMNS,
    ) -> PreprocessedNodes:
        return self.fit(
            expression, metadata, split, metadata_names=metadata_names
        ).transform(expression, metadata)
