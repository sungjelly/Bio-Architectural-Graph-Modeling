"""Train-fitted, deliberately low-frequency spatial-field features.

This module is NumPy-only so the leakage contract can be tested even before
the optional Torch model dependencies are installed.  The basis is global and
quadratic: it contains no cell IDs, learned spatial lookup table, Fourier
frequencies, knots, graph adjacency, or nearest-neighbor-derived quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


BROAD_SPATIAL_BASIS_NAME = "global_standardized_polynomial_degree_2"
BROAD_SPATIAL_FEATURE_NAMES = (
    "x_standardized",
    "y_standardized",
    "x_standardized_squared",
    "x_y_standardized_product",
    "y_standardized_squared",
)


def _coordinates(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    coordinates = np.asarray(value, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape [num_nodes, 2]")
    if coordinates.shape[0] == 0:
        raise ValueError("coordinates must contain at least one node")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must contain only finite values")
    return coordinates


def _checksum(value: dict[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BroadSpatialFieldBasis:
    """A global quadratic basis normalized using training coordinates only."""

    center_um: np.ndarray
    scale_um: np.ndarray
    constant_axes: tuple[bool, bool]

    def __post_init__(self) -> None:
        center = np.asarray(self.center_um, dtype=np.float64)
        scale = np.asarray(self.scale_um, dtype=np.float64)
        constant_axes = tuple(bool(value) for value in self.constant_axes)
        if center.shape != (2,) or scale.shape != (2,):
            raise ValueError("center_um and scale_um must both have shape [2]")
        if not np.all(np.isfinite(center)):
            raise ValueError("center_um must be finite")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("scale_um must be finite and positive")
        if len(constant_axes) != 2:
            raise ValueError("constant_axes must contain two flags")
        center = np.array(center, copy=True)
        scale = np.array(scale, copy=True)
        center.flags.writeable = False
        scale.flags.writeable = False
        object.__setattr__(self, "center_um", center)
        object.__setattr__(self, "scale_um", scale)
        object.__setattr__(self, "constant_axes", constant_axes)

    @classmethod
    def fit(cls, training_coordinates_um: Any) -> BroadSpatialFieldBasis:
        """Fit center and population standard deviation on training nodes."""

        coordinates = _coordinates(training_coordinates_um)
        center = coordinates.mean(axis=0)
        scale = coordinates.std(axis=0, ddof=0)
        constant = tuple(bool(value <= np.finfo(np.float64).eps) for value in scale)
        # A constant axis contains no spatial information.  The one-micron
        # fallback keeps its standardized feature exactly zero and the
        # transform numerically defined without learning from held-out nodes.
        scale = np.where(np.asarray(constant), 1.0, scale)
        return cls(
            center_um=center,
            scale_um=scale,
            constant_axes=constant,
        )

    @property
    def n_features(self) -> int:
        return len(BROAD_SPATIAL_FEATURE_NAMES)

    def transform(self, coordinates_um: Any) -> np.ndarray:
        """Apply the fixed global quadratic basis without refitting."""

        coordinates = _coordinates(coordinates_um)
        standardized = (coordinates - self.center_um) / self.scale_um
        # A training-constant axis has no estimable spatial coefficient.
        # Keep all of its held-out basis columns identically zero instead of
        # activating untrained weights when a later split lies off that axis.
        standardized[:, np.asarray(self.constant_axes, dtype=np.bool_)] = 0.0
        x = standardized[:, 0]
        y = standardized[:, 1]
        return np.column_stack((x, y, x * x, x * y, y * y)).astype(
            np.float32,
            copy=False,
        )

    def provenance(self) -> dict[str, Any]:
        """Return a JSON-safe declaration of basis, scale, and exclusions."""

        record: dict[str, Any] = {
            "control_name": "broad_spatial_field",
            "basis": BROAD_SPATIAL_BASIS_NAME,
            "maximum_polynomial_degree": 2,
            "feature_names": list(BROAD_SPATIAL_FEATURE_NAMES),
            "n_features": self.n_features,
            "coordinate_units": "micrometers",
            "center_um": self.center_um.tolist(),
            "scale_um": self.scale_um.tolist(),
            "scale_estimator": "training_population_standard_deviation",
            "constant_axes": list(self.constant_axes),
            "constant_axis_fallback_scale_um": 1.0,
            "fit_scope": "training split coordinates only",
            "held_out_refit": False,
            "contains_cell_or_region_ids": False,
            "contains_periodic_or_fourier_features": False,
            "contains_knots_or_spatial_lookup_embeddings": False,
            "contains_graph_or_neighbor_features": False,
        }
        record["basis_fit_checksum"] = _checksum(record)
        return record


__all__ = [
    "BROAD_SPATIAL_BASIS_NAME",
    "BROAD_SPATIAL_FEATURE_NAMES",
    "BroadSpatialFieldBasis",
]
