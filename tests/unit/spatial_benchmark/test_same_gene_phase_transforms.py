from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spatial_benchmark.same_gene_phase_transforms import (
    ARMS,
    SameGenePhaseTransformError,
    TrainOnlyPhaseTransformBuilder,
)


def _write_slide(root: Path, slide: str, arrays: dict[str, np.ndarray]) -> None:
    directory = root / slide
    directory.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(directory / name, value, allow_pickle=False)


def _masks(components: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "tuning_train": components < 2,
        "validation": components == 2,
        "final_train": components < 3,
        "test": components == 3,
    }


def test_library_phase_builder_uses_separate_train_fits_and_residualizes_neighbors(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    rows, genes = 48, 5
    groups = np.repeat(np.arange(4), rows // 4)
    library = rng.normal(7.0, 0.5, size=rows).astype(np.float32)
    intercept = rng.normal(size=genes).astype(np.float32)
    slope = rng.normal(scale=0.1, size=genes).astype(np.float32)
    expression = intercept + library[:, None] * slope
    mean_library = library + 0.25
    mean_expression = intercept + mean_library[:, None] * slope
    arrays: dict[str, np.ndarray] = {
        "expression_log1p.npy": expression,
        "panel_log_total.npy": library,
    }
    for arm in ARMS:
        arrays[f"{arm}_library.npy"] = mean_library
        arrays[f"{arm}_feature.npy"] = mean_expression
    _write_slide(tmp_path, "SO_1", {name: value[:24] for name, value in arrays.items()})
    _write_slide(tmp_path, "SO_2", {name: value[24:] for name, value in arrays.items()})
    spec = {
        "kind": "library",
        "panel_log_total_file": "panel_log_total.npy",
        "neighbor_panel_log_total_files": {
            arm: f"{arm}_library.npy" for arm in ARMS
        },
    }
    builder = TrainOnlyPhaseTransformBuilder(tmp_path, spec)
    phase, metadata = builder(
        arm="observed_near",
        target=torch.from_numpy(expression),
        feature=torch.from_numpy(mean_expression),
        prepared=tmp_path,
        masks=_masks(groups),
        groups=groups,
        profile="full",
        fold=0,
        device=torch.device("cpu"),
    )

    assert torch.max(torch.abs(phase["tuning_target"])).item() < 2e-6
    assert torch.max(torch.abs(phase["final_target"])).item() < 2e-6
    assert torch.max(torch.abs(phase["tuning_feature"])).item() < 2e-6
    assert torch.max(torch.abs(phase["final_feature"])).item() < 2e-6
    assert metadata["tuning_fit_mask"] == "tuning_train"
    assert metadata["final_fit_mask"] == "final_train"
    assert metadata["tuning"]["training_component_count"] == 2
    assert metadata["final"]["training_component_count"] == 3


def test_cell_type_phase_builder_applies_neighbor_type_mixture(tmp_path: Path) -> None:
    rng = np.random.default_rng(31)
    rows, genes, type_count = 48, 4, 3
    groups = np.repeat(np.arange(4), rows // 4)
    codes = np.arange(rows, dtype=np.int16) % type_count
    levels = ["A", "B", "C"]
    library = rng.normal(7.0, 0.5, size=rows).astype(np.float32)
    type_intercepts = rng.normal(size=(type_count, genes)).astype(np.float32)
    slope = rng.normal(scale=0.1, size=genes).astype(np.float32)
    expression = type_intercepts[codes] + library[:, None] * slope
    proportions = np.eye(type_count, dtype=np.float32)[codes]
    mean_library = library + 0.4
    mean_expression = proportions @ type_intercepts + mean_library[:, None] * slope
    arrays: dict[str, np.ndarray] = {
        "expression_log1p.npy": expression,
        "panel_log_total.npy": library,
        "cell_type_code.npy": codes,
    }
    for arm in ARMS:
        arrays[f"{arm}_library.npy"] = mean_library
        arrays[f"{arm}_types.npy"] = proportions
    _write_slide(tmp_path, "SO_1", {name: value[:24] for name, value in arrays.items()})
    _write_slide(tmp_path, "SO_2", {name: value[24:] for name, value in arrays.items()})
    (tmp_path / "cell_type_levels.json").write_text(
        json.dumps(levels), encoding="utf-8"
    )
    spec = {
        "kind": "cell_type_library",
        "panel_log_total_file": "panel_log_total.npy",
        "neighbor_panel_log_total_files": {
            arm: f"{arm}_library.npy" for arm in ARMS
        },
        "cell_type_code_file": "cell_type_code.npy",
        "cell_type_levels_file": "cell_type_levels.json",
        "neighbor_cell_type_proportion_files": {
            arm: f"{arm}_types.npy" for arm in ARMS
        },
    }
    builder = TrainOnlyPhaseTransformBuilder(tmp_path, spec)
    phase, metadata = builder(
        arm="observed_annular",
        target=torch.from_numpy(expression),
        feature=torch.from_numpy(mean_expression),
        prepared=tmp_path,
        masks=_masks(groups),
        groups=groups,
        profile="full",
        fold=0,
        device=torch.device("cpu"),
    )

    assert torch.max(torch.abs(phase["tuning_target"])).item() < 3e-5
    assert torch.max(torch.abs(phase["final_target"])).item() < 3e-5
    assert torch.max(torch.abs(phase["tuning_feature"])).item() < 3e-5
    assert torch.max(torch.abs(phase["final_feature"])).item() < 3e-5
    assert metadata["kind"] == "cell_type_library"
    assert metadata["tuning"]["levels"] == levels


def test_phase_builder_rejects_cross_fold_reuse(tmp_path: Path) -> None:
    rows, genes = 16, 3
    groups = np.repeat(np.arange(4), 4)
    library = np.linspace(1.0, 3.0, rows, dtype=np.float32)
    expression = np.column_stack(
        [library * (index + 1) + index for index in range(genes)]
    ).astype(np.float32)
    arrays = {
        "expression_log1p.npy": expression,
        "panel_log_total.npy": library,
        "neighbor_library.npy": library,
    }
    _write_slide(tmp_path, "SO_1", arrays)
    builder = TrainOnlyPhaseTransformBuilder(
        tmp_path,
        {
            "kind": "library",
            "panel_log_total_file": "panel_log_total.npy",
            "neighbor_panel_log_total_files": {
                arm: "neighbor_library.npy" for arm in ARMS
            },
        },
        slides=("SO_1",),
    )
    kwargs = {
        "arm": "observed_near",
        "target": torch.from_numpy(expression),
        "feature": torch.from_numpy(expression),
        "prepared": tmp_path,
        "masks": _masks(groups),
        "groups": groups,
        "profile": "full",
        "device": torch.device("cpu"),
    }
    _, metadata = builder(fold=0, **kwargs)
    assert metadata["tuning_fit_mask_identity"]["dtype"] == "uint8"
    with pytest.raises(SameGenePhaseTransformError, match="cannot be reused"):
        builder(fold=1, **kwargs)
