from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import spatial_benchmark.so1_pooled_full_core as cohort_module
from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS, CoreDataset
from spatial_benchmark.so1_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_MAPPED_FOV_COUNT,
    EXPECTED_TOTAL_CELLS,
    FIT_SCOPE,
    SO1_ALIASES,
    SO1_CORE_NUMBERS,
    SO1CohortContractError,
    audit_so1_raw_fov_partition,
    prepare_so1_14core_cohort,
    resolve_so1_core_routes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _write_core_map(path: Path, *, wrong_slide_core: int | None = None) -> None:
    rows = [
        {
            "slide": "SO_2" if core == wrong_slide_core else "SO_1",
            "core_label": core,
            "fov": core,
        }
        for core in SO1_CORE_NUMBERS
    ]
    rows.append({"slide": "SO_2", "core_label": 15, "fov": 101})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_locked_so1_counts_aliases_and_fit_scope_are_exact() -> None:
    assert tuple(EXPECTED_CELL_COUNTS_BY_CORE) == SO1_CORE_NUMBERS
    assert sum(EXPECTED_CELL_COUNTS_BY_CORE.values()) == EXPECTED_TOTAL_CELLS
    assert EXPECTED_TOTAL_CELLS == 161_596
    assert EXPECTED_MAPPED_FOV_COUNT == 205
    assert SO1_ALIASES == tuple(f"SO1-C{core:02d}" for core in range(1, 15))
    assert FIT_SCOPE == "all_cells_so1_cores_1_through_14_transductive"


def test_routes_are_neutral_ordered_and_slide_qualified(tmp_path: Path) -> None:
    core_map = tmp_path / "core_map.csv"
    _write_core_map(core_map)
    routes = resolve_so1_core_routes(core_map)
    assert tuple(route.alias for route in routes) == SO1_ALIASES
    assert tuple(route.core_number for route in routes) == SO1_CORE_NUMBERS
    assert {route.slide for route in routes} == {"SO_1"}
    assert tuple(route.fovs for route in routes) == tuple(
        (core,) for core in SO1_CORE_NUMBERS
    )
    assert len({route.route_sha256 for route in routes}) == len(routes)

    _write_core_map(core_map, wrong_slide_core=7)
    with pytest.raises(SO1CohortContractError, match="exclusively to SO_1"):
        resolve_so1_core_routes(core_map)


def test_routing_audit_rejects_any_unmapped_raw_fov(tmp_path: Path) -> None:
    core_map = tmp_path / "core_map.csv"
    _write_core_map(core_map)
    routes = resolve_so1_core_routes(core_map)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"fov": [*SO1_CORE_NUMBERS, 999]}).to_csv(metadata, index=False)
    with pytest.raises(SO1CohortContractError, match="Every raw SO_1 FOV"):
        audit_so1_raw_fov_partition(
            metadata,
            routes,
            expected_cell_counts_by_core={core: 1 for core in SO1_CORE_NUMBERS},
            expected_total_cells=len(SO1_CORE_NUMBERS),
            expected_mapped_fov_count=len(SO1_CORE_NUMBERS),
        )


def _synthetic_loaded_slide() -> CoreDataset:
    n_cells = len(SO1_CORE_NUMBERS)
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
            "slide": pd.Series(["SO_1"] * n_cells, dtype="string"),
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


def test_preparation_refits_equal_core_statistics_and_records_zero_unmapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_map = tmp_path / "core_map.csv"
    _write_core_map(core_map)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    expression_path = raw_dir / "SO_1-expression.csv"
    expression_path.write_text("synthetic expression source\n", encoding="utf-8")
    metadata_path = raw_dir / "SO_1-metadata.csv"
    pd.DataFrame({"fov": list(SO1_CORE_NUMBERS)}).to_csv(
        metadata_path, index=False
    )
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
    expected_counts = {core: 1 for core in SO1_CORE_NUMBERS}
    manifest = prepare_so1_14core_cohort(
        raw_dir=raw_dir,
        core_map_csv=core_map,
        output_dir=output,
        expected_cell_counts_by_core=expected_counts,
        expected_total_cells=len(SO1_CORE_NUMBERS),
        expected_mapped_fov_count=len(SO1_CORE_NUMBERS),
    )

    expected_mean = np.mean(
        np.log1p(np.arange(len(SO1_CORE_NUMBERS), dtype=np.float64))
    )
    with np.load(output / "cohort_statistics.npz", allow_pickle=False) as stats:
        np.testing.assert_allclose(stats["expression_mean"], expected_mean)
    assert manifest["artifact_kind"] == "so1_14core_pooled_full_core_preparation"
    assert manifest["cohort"]["aliases"] == list(SO1_ALIASES)
    assert manifest["cohort"]["total_cells"] == len(SO1_CORE_NUMBERS)
    assert manifest["cohort"]["tissue_context"] == (
        "not_asserted_neutral_core_number_cohort"
    )
    assert manifest["preprocessing"]["expression_moment_weighting"] == "equal_core"
    assert manifest["routing_audit"] == {
        "source_slide": "SO_1",
        "mapped_core_numbers": list(SO1_CORE_NUMBERS),
        "mapped_fov_count": len(SO1_CORE_NUMBERS),
        "selected_cell_count": len(SO1_CORE_NUMBERS),
        "per_core_cell_counts": {
            str(core): 1 for core in SO1_CORE_NUMBERS
        },
        "raw_slide_cell_count": len(SO1_CORE_NUMBERS),
        "selection_is_slide_qualified": True,
        "all_raw_fovs_mapped": True,
        "unmapped_fovs": [],
        "unmapped_fov_count": 0,
        "unmapped_fov_cell_count": 0,
        "unmapped_fov_entered_prepared_arrays": False,
    }
    assert manifest["assurances"]["all_raw_fovs_mapped"] is True
    assert manifest["assurances"]["unmapped_fov_exclusion_required"] is False
    for alias in SO1_ALIASES:
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
            assert data["expression_counts"].shape == (1, 1_000)
            assert data["node_covariates"].shape == (
                1,
                len(ALLOWED_METADATA_COLUMNS),
            )
            assert np.isfinite(data["target_expression"]).all()


def test_preparation_never_overwrites_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    output.mkdir()
    with pytest.raises(FileExistsError, match="immutable artifacts"):
        prepare_so1_14core_cohort(
            raw_dir=tmp_path / "raw",
            core_map_csv=tmp_path / "core_map.csv",
            output_dir=output,
        )


def test_preparation_cli_exposes_so1_zero_unmapped_contract() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/data/prepare_so1_14core_relative_qkv.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    assert "SO_1 cores 1 through 14" in completed.stdout
    assert "complete raw-FOV coverage" in completed.stdout
    assert "--core-map" in completed.stdout
    assert "--output-dir" in completed.stdout
