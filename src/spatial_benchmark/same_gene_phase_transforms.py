"""Train-population-fitted phase transforms for robustness runner variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from spatial_benchmark.same_gene_residualization import (
    CellTypeLibraryWLSFit,
    LibraryWLSFit,
    fit_cell_type_library_wls,
    fit_component_equal_library_wls,
)


ARMS = (
    "observed_near",
    "observed_annular",
    "within_fov_permuted_near",
)


class SameGenePhaseTransformError(RuntimeError):
    """Raised when a residual robustness variant is not replayable."""


def _safe_filename(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise SameGenePhaseTransformError(f"{label} must be one safe filename")
    return value


def _file_map(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(ARMS):
        raise SameGenePhaseTransformError(f"{label} must map exactly {ARMS}")
    return {
        arm: _safe_filename(value[arm], label=f"{label}.{arm}") for arm in ARMS
    }


def _load_vector(root: Path, slides: tuple[str, ...], filename: str) -> np.ndarray:
    arrays = [np.load(root / slide / filename, allow_pickle=False) for slide in slides]
    if any(array.ndim != 1 for array in arrays):
        raise SameGenePhaseTransformError(f"{filename} must be a vector per slide")
    result = np.concatenate(arrays)
    if not bool(np.isfinite(result).all()):
        raise SameGenePhaseTransformError(f"{filename} contains nonfinite values")
    return result


def _load_matrix(root: Path, slides: tuple[str, ...], filename: str) -> np.ndarray:
    arrays = [np.load(root / slide / filename, allow_pickle=False) for slide in slides]
    if any(array.ndim != 2 for array in arrays):
        raise SameGenePhaseTransformError(f"{filename} must be a matrix per slide")
    result = np.concatenate(arrays, axis=0)
    if not bool(np.isfinite(result).all()):
        raise SameGenePhaseTransformError(f"{filename} contains nonfinite values")
    return result


def _array_identity(value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256_c_order": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _fit_payload(fit: LibraryWLSFit | CellTypeLibraryWLSFit) -> dict[str, Any]:
    common = {
        "training_row_count": int(fit.training_row_count),
        "training_component_count": int(fit.training_component_count),
        "weighted_library_mean": float(fit.weighted_library_mean),
        "library_slope": np.asarray(fit.library_slope, dtype=np.float64).tolist(),
        "library_slope_identity": _array_identity(fit.library_slope),
    }
    if isinstance(fit, LibraryWLSFit):
        return {
            "kind": "library",
            **common,
            "intercept": np.asarray(fit.intercept, dtype=np.float64).tolist(),
            "intercept_identity": _array_identity(fit.intercept),
        }
    return {
        "kind": "cell_type_library",
        **common,
        "levels": list(fit.levels),
        "type_intercepts": np.asarray(
            fit.type_intercepts, dtype=np.float64
        ).tolist(),
        "type_intercepts_identity": _array_identity(fit.type_intercepts),
        "global_intercept": np.asarray(
            fit.global_intercept, dtype=np.float64
        ).tolist(),
        "global_intercept_identity": _array_identity(fit.global_intercept),
        "type_weight_mass": np.asarray(
            fit.type_weight_mass, dtype=np.float64
        ).tolist(),
    }


class TrainOnlyPhaseTransformBuilder:
    """Build tuning/final residual tensors from their respective train masks.

    The outcome-independent auxiliary graph means are precomputed.  WLS fits
    are cached once per process and the two target tensors are retained, while
    arm-specific feature tensors are constructed and released one arm at a
    time by the base runner.
    """

    def __init__(
        self,
        variant_root: Path,
        residualization_spec: Mapping[str, Any],
        *,
        slides: tuple[str, ...] = ("SO_1", "SO_2"),
    ) -> None:
        self.root = Path(variant_root).resolve(strict=True)
        self.slides = tuple(slides)
        self.spec = dict(residualization_spec)
        kind = self.spec.get("kind")
        if kind not in {"library", "cell_type_library"}:
            raise SameGenePhaseTransformError(f"unsupported residualization: {kind}")
        self.kind = str(kind)
        self.panel_file = _safe_filename(
            self.spec.get("panel_log_total_file"), label="panel_log_total_file"
        )
        self.neighbor_library_files = _file_map(
            self.spec.get("neighbor_panel_log_total_files"),
            label="neighbor_panel_log_total_files",
        )
        self.cell_type_code_file: str | None = None
        self.cell_type_levels_file: str | None = None
        self.neighbor_type_files: dict[str, str] | None = None
        if self.kind == "cell_type_library":
            self.cell_type_code_file = _safe_filename(
                self.spec.get("cell_type_code_file"), label="cell_type_code_file"
            )
            self.cell_type_levels_file = _safe_filename(
                self.spec.get("cell_type_levels_file"), label="cell_type_levels_file"
            )
            self.neighbor_type_files = _file_map(
                self.spec.get("neighbor_cell_type_proportion_files"),
                label="neighbor_cell_type_proportion_files",
            )
        self._initialized = False
        self._fits: dict[str, LibraryWLSFit | CellTypeLibraryWLSFit] = {}
        self._target_tensors: dict[str, torch.Tensor] = {}
        self._metadata: dict[str, Any] = {}
        self._context: tuple[Any, ...] | None = None

    def _initialize(
        self,
        *,
        target: torch.Tensor,
        masks: Mapping[str, np.ndarray],
        groups: np.ndarray,
        profile: str,
        device: torch.device,
    ) -> None:
        if self._initialized:
            return
        expression = _load_matrix(self.root, self.slides, "expression_log1p.npy")
        library = _load_vector(self.root, self.slides, self.panel_file).astype(
            np.float64, copy=False
        )
        if expression.shape != tuple(target.shape) or len(library) != len(expression):
            raise SameGenePhaseTransformError("residual auxiliary arrays are misaligned")
        group_array = np.asarray(groups)
        if group_array.shape != (len(expression),):
            raise SameGenePhaseTransformError("geometry groups are misaligned")
        fit_mask_name = "tuning_train" if profile == "pilot" else "final_train"
        fit_masks = {
            "tuning": np.asarray(masks["tuning_train"], dtype=bool),
            "final": np.asarray(masks[fit_mask_name], dtype=bool),
        }
        codes: np.ndarray | None = None
        levels: tuple[str, ...] | None = None
        labels: np.ndarray | None = None
        if self.kind == "cell_type_library":
            assert self.cell_type_code_file is not None
            assert self.cell_type_levels_file is not None
            codes = _load_vector(self.root, self.slides, self.cell_type_code_file).astype(
                np.int16, copy=False
            )
            levels_path = self.root / self.cell_type_levels_file
            raw_levels = json.loads(levels_path.read_text(encoding="utf-8"))
            if (
                not isinstance(raw_levels, list)
                or not raw_levels
                or any(not isinstance(value, str) or not value for value in raw_levels)
                or len(set(raw_levels)) != len(raw_levels)
            ):
                raise SameGenePhaseTransformError("cell-type levels are invalid")
            levels = tuple(raw_levels)
            if np.any(codes < 0) or np.any(codes >= len(levels)):
                raise SameGenePhaseTransformError("cell-type code is outside frozen levels")
            labels = np.asarray(levels, dtype=str)[codes]
        for phase, mask in fit_masks.items():
            if self.kind == "library":
                fit: LibraryWLSFit | CellTypeLibraryWLSFit = (
                    fit_component_equal_library_wls(
                        expression,
                        library,
                        group_array,
                        train_mask=mask,
                    )
                )
            else:
                assert labels is not None and levels is not None
                fit = fit_cell_type_library_wls(
                    expression,
                    library,
                    labels,
                    group_array,
                    levels=levels,
                    train_mask=mask,
                )
            self._fits[phase] = fit
            self._target_tensors[phase] = self._target_residual_tensor(
                target, library, fit, codes=codes, device=device
            )
        self._metadata = {
            "kind": self.kind,
            "tuning_fit_mask": "tuning_train",
            "final_fit_mask": fit_mask_name,
            "tuning": _fit_payload(self._fits["tuning"]),
            "final": _fit_payload(self._fits["final"]),
            "panel_log_total_file": self.panel_file,
            "tuning_fit_mask_identity": _array_identity(
                fit_masks["tuning"].astype(np.uint8, copy=False)
            ),
            "final_fit_mask_identity": _array_identity(
                fit_masks["final"].astype(np.uint8, copy=False)
            ),
        }
        self._initialized = True

    @staticmethod
    def _target_residual_tensor(
        target: torch.Tensor,
        library: np.ndarray,
        fit: LibraryWLSFit | CellTypeLibraryWLSFit,
        *,
        codes: np.ndarray | None,
        device: torch.device,
    ) -> torch.Tensor:
        slope = torch.from_numpy(np.asarray(fit.library_slope, dtype=np.float32)).to(
            device
        )
        library_tensor = torch.from_numpy(library.astype(np.float32, copy=False)).to(
            device
        )
        if isinstance(fit, LibraryWLSFit):
            intercept = torch.from_numpy(
                np.asarray(fit.intercept, dtype=np.float32)
            ).to(device)
            residual = target - intercept - library_tensor[:, None] * slope
        else:
            if codes is None:
                raise SameGenePhaseTransformError("cell-type codes are required")
            intercepts = torch.from_numpy(
                np.asarray(fit.type_intercepts, dtype=np.float32)
            ).to(device)
            code_tensor = torch.from_numpy(codes.astype(np.int64, copy=False)).to(device)
            residual = (
                target
                - intercepts.index_select(0, code_tensor)
                - library_tensor[:, None] * slope
            )
        if not bool(torch.isfinite(residual).all().item()):
            raise SameGenePhaseTransformError("target residual is nonfinite")
        return residual

    def _feature_residual_tensor(
        self,
        feature: torch.Tensor,
        *,
        arm: str,
        phase: str,
        device: torch.device,
    ) -> torch.Tensor:
        fit = self._fits[phase]
        mean_library = _load_vector(
            self.root, self.slides, self.neighbor_library_files[arm]
        ).astype(np.float32, copy=False)
        library_tensor = torch.from_numpy(mean_library).to(device)
        slope = torch.from_numpy(np.asarray(fit.library_slope, dtype=np.float32)).to(
            device
        )
        if isinstance(fit, LibraryWLSFit):
            intercept = torch.from_numpy(
                np.asarray(fit.intercept, dtype=np.float32)
            ).to(device)
            result = feature - intercept - library_tensor[:, None] * slope
        else:
            assert self.neighbor_type_files is not None
            proportions = _load_matrix(
                self.root, self.slides, self.neighbor_type_files[arm]
            ).astype(np.float32, copy=False)
            if proportions.shape != (feature.shape[0], len(fit.levels)):
                raise SameGenePhaseTransformError(
                    "neighbor cell-type proportions are misaligned"
                )
            if np.any(proportions < 0) or np.any(proportions.sum(axis=1) > 1 + 1e-6):
                raise SameGenePhaseTransformError(
                    "neighbor cell-type proportions are invalid"
                )
            proportion_tensor = torch.from_numpy(proportions).to(device)
            type_intercepts = torch.from_numpy(
                np.asarray(fit.type_intercepts, dtype=np.float32)
            ).to(device)
            global_intercept = torch.from_numpy(
                np.asarray(fit.global_intercept, dtype=np.float32)
            ).to(device)
            mass = proportion_tensor.sum(dim=1)
            mean_intercept = proportion_tensor @ type_intercepts
            mean_intercept += (1.0 - mass)[:, None] * global_intercept
            result = feature - mean_intercept - library_tensor[:, None] * slope
        if not bool(torch.isfinite(result).all().item()):
            raise SameGenePhaseTransformError("neighbor residual is nonfinite")
        return result

    def __call__(
        self,
        *,
        arm: str,
        target: torch.Tensor,
        feature: torch.Tensor | None,
        prepared: Path,
        masks: Mapping[str, np.ndarray],
        groups: np.ndarray,
        profile: str,
        fold: int,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor | None], dict[str, Any]]:
        if Path(prepared).resolve(strict=True) != self.root:
            raise SameGenePhaseTransformError("phase builder prepared root changed")
        if target.ndim != 2 or target.shape[1] < 1:
            raise SameGenePhaseTransformError("target must have shape [N,G]")
        if feature is not None and feature.shape != target.shape:
            raise SameGenePhaseTransformError(
                "neighbor feature must preserve target shape"
            )
        mask_identity = tuple(
            (
                name,
                _array_identity(np.asarray(masks[name], dtype=np.uint8))[
                    "sha256_c_order"
                ],
            )
            for name in ("tuning_train", "final_train", "validation", "test")
        )
        context = (
            str(profile),
            int(fold),
            str(device),
            tuple(int(value) for value in target.shape),
            mask_identity,
        )
        if self._context is None:
            self._context = context
        elif self._context != context:
            raise SameGenePhaseTransformError(
                "phase builder cannot be reused across fold/profile/mask contexts"
            )
        self._initialize(
            target=target,
            masks=masks,
            groups=groups,
            profile=profile,
            device=device,
        )
        tuning_feature = None
        final_feature = None
        if feature is not None:
            if arm not in ARMS:
                raise SameGenePhaseTransformError(f"unsupported feature arm: {arm}")
            tuning_feature = self._feature_residual_tensor(
                feature, arm=arm, phase="tuning", device=device
            )
            final_feature = self._feature_residual_tensor(
                feature, arm=arm, phase="final", device=device
            )
        return (
            {
                "tuning_target": self._target_tensors["tuning"],
                "final_target": self._target_tensors["final"],
                "tuning_feature": tuning_feature,
                "final_feature": final_feature,
            },
            {
                **self._metadata,
                "arm": arm,
                "neighbor_panel_log_total_file": (
                    None if feature is None else self.neighbor_library_files[arm]
                ),
                "neighbor_cell_type_proportion_file": (
                    None
                    if feature is None or self.neighbor_type_files is None
                    else self.neighbor_type_files[arm]
                ),
            },
        )


__all__ = [
    "SameGenePhaseTransformError",
    "TrainOnlyPhaseTransformBuilder",
]
