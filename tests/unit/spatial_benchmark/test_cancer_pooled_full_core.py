from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import spatial_benchmark.cancer_pooled_full_core as cohort_module
from spatial_benchmark.cancer_pooled_full_core import (
    CANCER_ALIASES,
    CORE_NUMBERS,
    CancerCohortContractError,
    prepare_cancer_6core_cohort,
    resolve_cancer_core_routes,
    validate_cancer_reconciliation,
)
from spatial_benchmark.data import ALLOWED_METADATA_COLUMNS, CoreDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY = (
    PROJECT_ROOT
    / "experiments/campaigns"
    / "cmp_20260824_cancer_6core_relative_qkv_multiseed"
    / "clinical_reconciliation_policy.yaml"
)


def test_authoritative_map_resolves_exact_six_slide_qualified_routes() -> None:
    receipt = validate_cancer_reconciliation(POLICY)
    routes = resolve_cancer_core_routes(
        PROJECT_ROOT / "data/clinical/fov_core_map.csv", POLICY
    )
    assert receipt.resolved_tissue_context == "Cancer"
    assert tuple(route.alias for route in routes) == CANCER_ALIASES
    assert tuple(route.core_number for route in routes) == CORE_NUMBERS
    assert tuple(route.slide for route in routes) == (
        "SO_1",
        "SO_1",
        "SO_1",
        "SO_2",
        "SO_2",
        "SO_2",
    )
    qualified = [
        (route.slide, fov) for route in routes for fov in route.fovs
    ]
    assert len(qualified) == len(set(qualified))


def test_reconciliation_fails_closed_without_all_core_cancer_attestation() -> None:
    import yaml

    policy = yaml.safe_load(POLICY.read_text())
    policy["user_attestation"]["resolved_tissue_context"] = "normal"
    with pytest.raises(CancerCohortContractError, match="resolve.*Cancer"):
        validate_cancer_reconciliation(policy)


def _synthetic_core(route_index: int, slide: str, fov: int) -> CoreDataset:
    n_cells = route_index + 2
    counts = np.full((n_cells, 1000), route_index, dtype=np.int32)
    counts[:, route_index] += np.arange(n_cells, dtype=np.int32)
    metadata = np.full(
        (n_cells, len(ALLOWED_METADATA_COLUMNS)),
        float(route_index + 1),
        dtype=np.float32,
    )
    if route_index == 0:
        metadata[0, 0] = np.nan
    coordinates = np.column_stack(
        [np.arange(n_cells) * 10.0, np.arange(n_cells) * 5.0]
    )
    keys = pd.DataFrame(
        {
            "slide": pd.Series([slide] * n_cells, dtype="string"),
            "fov": np.full(n_cells, fov, dtype=np.int32),
            "cell_ID": np.arange(1, n_cells + 1, dtype=np.int32),
        }
    )
    return CoreDataset(
        expression=counts,
        metadata=metadata,
        coordinates_px=coordinates,
        coordinates_um=coordinates,
        keys=keys,
        qc_passed=np.ones(n_cells, dtype=bool),
        gene_names=tuple(f"Gene{index:04d}" for index in range(1000)),
    )


def test_preparation_uses_equal_core_expression_moments_and_writes_no_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_map = tmp_path / "core_map.csv"
    rows = []
    slide_by_index = []
    for index, core_number in enumerate(CORE_NUMBERS):
        slide = "SO_1" if index < 3 else "SO_2"
        slide_by_index.append(slide)
        rows.append({"slide": slide, "core_label": core_number, "fov": index + 1})
    pd.DataFrame(rows).to_csv(core_map, index=False)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source_files = {}
    for slide in ("SO_1", "SO_2"):
        for kind in ("expression", "metadata"):
            path = raw_dir / f"{slide}-{kind}.csv"
            path.write_text(f"{slide},{kind}\n")
            source_files[(slide, kind)] = path

    cores = {
        (slide_by_index[index], (index + 1,)): _synthetic_core(
            index, slide_by_index[index], index + 1
        )
        for index in range(6)
    }

    def fake_load(_raw: Path, selection: object, **_: object) -> CoreDataset:
        return cores[(selection.slide, selection.fovs)]  # type: ignore[attr-defined]

    monkeypatch.setattr(cohort_module, "load_selected_core", fake_load)
    monkeypatch.setattr(
        cohort_module,
        "discover_slide_raw_path",
        lambda _raw, slide, kind: source_files[(slide, kind)],
    )
    output = tmp_path / "prepared"
    manifest = prepare_cancer_6core_cohort(
        raw_dir=raw_dir,
        core_map_csv=core_map,
        reconciliation_yaml=POLICY,
        output_dir=output,
    )

    expected_mean = np.mean(
        [np.log1p(core.expression).mean(axis=0) for core in cores.values()],
        axis=0,
    )
    with np.load(output / "cohort_statistics.npz", allow_pickle=False) as stats:
        np.testing.assert_allclose(stats["expression_mean"], expected_mean)
    assert manifest["cohort"]["aliases"] == list(CANCER_ALIASES)
    assert manifest["cohort"]["validation_or_test_partition_present"] is False
    assert manifest["preprocessing"]["expression_moment_weighting"] == "equal_core"
    for alias in CANCER_ALIASES:
        with np.load(output / "cores" / f"{alias}.npz", allow_pickle=False) as data:
            assert set(data.files) == {
                "expression_counts",
                "target_expression",
                "node_covariates",
                "coordinates_um",
            }
            assert not any("id" in name.lower() or "fov" in name.lower() for name in data.files)
            assert data["node_covariates"].shape[1] == 23
            assert np.isfinite(data["target_expression"]).all()
