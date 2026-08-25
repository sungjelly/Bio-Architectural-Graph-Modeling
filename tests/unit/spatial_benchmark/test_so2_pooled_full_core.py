from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spatial_benchmark.so2_pooled_full_core as cohort_module
from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS, CoreDataset
from spatial_benchmark.so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_TOTAL_CELLS,
    SO2_ALIASES,
    SO2_CORE_NUMBERS,
    SO2CohortContractError,
    prepare_so2_14core_cohort,
    resolve_so2_core_routes,
)


def _write_core_map(path: Path, *, assign_fov246: bool = False) -> None:
    rows = [
        {"slide": "SO_1", "core_label": 1, "fov": 101},
        *[
            {"slide": "SO_2", "core_label": core, "fov": index + 1}
            for index, core in enumerate(SO2_CORE_NUMBERS)
        ],
    ]
    if assign_fov246:
        rows.append({"slide": "SO_2", "core_label": 28, "fov": 246})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_locked_so2_cell_counts_are_exact() -> None:
    assert tuple(EXPECTED_CELL_COUNTS_BY_CORE) == SO2_CORE_NUMBERS
    assert sum(EXPECTED_CELL_COUNTS_BY_CORE.values()) == EXPECTED_TOTAL_CELLS
    assert EXPECTED_TOTAL_CELLS == 246_063


def test_routes_are_neutral_ordered_and_exclude_unmapped_fov246(
    tmp_path: Path,
) -> None:
    core_map = tmp_path / "core_map.csv"
    _write_core_map(core_map)
    routes = resolve_so2_core_routes(core_map)
    assert tuple(route.alias for route in routes) == SO2_ALIASES
    assert tuple(route.core_number for route in routes) == SO2_CORE_NUMBERS
    assert {route.slide for route in routes} == {"SO_2"}
    assert 246 not in {fov for route in routes for fov in route.fovs}

    _write_core_map(core_map, assign_fov246=True)
    with pytest.raises(SO2CohortContractError, match="FOV246"):
        resolve_so2_core_routes(core_map)


def _synthetic_loaded_slide() -> CoreDataset:
    n_cells = len(SO2_CORE_NUMBERS)
    counts = np.repeat(
        np.arange(n_cells, dtype=np.int32)[:, None], 1_000, axis=1
    )
    metadata = np.ones(
        (n_cells, len(ALLOWED_METADATA_COLUMNS)), dtype=np.float32
    )
    coordinates = np.column_stack(
        [np.arange(n_cells, dtype=np.float64) * 10.0, np.zeros(n_cells)]
    )
    keys = pd.DataFrame(
        {
            "slide": pd.Series(["SO_2"] * n_cells, dtype="string"),
            "fov": np.arange(1, n_cells + 1, dtype=np.int32),
            "cell_ID": np.ones(n_cells, dtype=np.int32),
        }
    )
    return CoreDataset(
        expression=counts,
        metadata=metadata,
        coordinates_px=coordinates,
        coordinates_um=coordinates,
        keys=keys,
        qc_passed=np.ones(n_cells, dtype=bool),
        gene_names=tuple(f"Gene{index:04d}" for index in range(1_000)),
    )


def test_preparation_refits_equal_core_statistics_and_records_fov246_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_map = tmp_path / "core_map.csv"
    _write_core_map(core_map)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    expression_path = raw_dir / "SO_2-expression.csv"
    expression_path.write_text("synthetic expression source\n", encoding="utf-8")
    metadata_path = raw_dir / "SO_2-metadata.csv"
    pd.DataFrame(
        {"fov": [*range(1, len(SO2_CORE_NUMBERS) + 1), 246, 246]}
    ).to_csv(metadata_path, index=False)
    loaded = _synthetic_loaded_slide()

    monkeypatch.setattr(
        cohort_module,
        "discover_slide_raw_path",
        lambda _raw, _slide, kind: (
            expression_path if kind == "expression" else metadata_path
        ),
    )
    monkeypatch.setattr(
        cohort_module,
        "load_selected_core",
        lambda *_args, **_kwargs: loaded,
    )
    output = tmp_path / "prepared"
    expected_counts = {core: 1 for core in SO2_CORE_NUMBERS}
    manifest = prepare_so2_14core_cohort(
        raw_dir=raw_dir,
        core_map_csv=core_map,
        output_dir=output,
        expected_cell_counts_by_core=expected_counts,
        expected_total_cells=len(SO2_CORE_NUMBERS),
        expected_excluded_fov_cells=2,
    )

    expected_mean = np.mean(
        np.log1p(np.arange(len(SO2_CORE_NUMBERS), dtype=np.float64))
    )
    with np.load(output / "cohort_statistics.npz", allow_pickle=False) as stats:
        np.testing.assert_allclose(stats["expression_mean"], expected_mean)
    assert manifest["cohort"]["aliases"] == list(SO2_ALIASES)
    assert manifest["cohort"]["total_cells"] == len(SO2_CORE_NUMBERS)
    assert manifest["preprocessing"]["expression_moment_weighting"] == "equal_core"
    assert manifest["routing_audit"]["explicitly_excluded_unmapped_fov"] == 246
    assert manifest["routing_audit"]["excluded_unmapped_fov_cell_count"] == 2
    assert manifest["assurances"]["unmapped_fov_246_excluded"] is True
    for alias in SO2_ALIASES:
        with np.load(output / "cores" / f"{alias}.npz", allow_pickle=False) as data:
            assert set(data.files) == {
                "expression_counts",
                "target_expression",
                "node_covariates",
                "coordinates_um",
            }
            assert not any(
                "id" in name.lower() or "fov" in name.lower()
                for name in data.files
            )
            assert np.isfinite(data["target_expression"]).all()
