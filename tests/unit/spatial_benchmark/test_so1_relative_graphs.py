from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import spatial_benchmark.so1_relative_graphs as graph_module
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES
from spatial_benchmark.cancer_relative_graphs import (
    prepare_cancer_6core_relative_graphs,
)
from spatial_benchmark.relative_geometry import RELATIVE_GEOMETRY_DIM
from spatial_benchmark.so1_pooled_full_core import SO1_ALIASES, SO1_CORE_NUMBERS
from spatial_benchmark.so1_relative_graphs import (
    SO1RelativeGraphContractError,
    materialize_so1_core_relative_graph,
    prepare_so1_14core_relative_graphs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _coordinates() -> np.ndarray:
    x, y = np.meshgrid(np.arange(4) * 25.0, np.arange(3) * 25.0)
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cohort(
    root: Path,
    aliases: tuple[str, ...],
    *,
    so1: bool,
    coordinate_overrides: dict[str, np.ndarray] | None = None,
) -> None:
    (root / "cores").mkdir(parents=True)
    coordinate_overrides = coordinate_overrides or {}
    files = {}
    rows = []
    total = 0
    for alias in aliases:
        coordinates = np.asarray(
            coordinate_overrides.get(alias, _coordinates()), dtype=np.float64
        )
        total += len(coordinates)
        path = root / "cores" / f"{alias}.npz"
        np.savez_compressed(
            path,
            target_expression=np.ones(
                (len(coordinates), 1_000), dtype=np.float32
            ),
            node_covariates=np.zeros((len(coordinates), 2), dtype=np.float32),
            coordinates_um=coordinates,
        )
        files[f"cores/{alias}.npz"] = _sha256(path)
        rows.append(
            {
                "alias": alias,
                "cell_count": len(coordinates),
                "graph_checksum": None,
            }
        )
    cohort: dict[str, object] = {
        "aliases": list(aliases),
        "validation_or_test_partition_present": False,
    }
    manifest: dict[str, object] = {"cohort": cohort, "cores": rows, "files": files}
    if so1:
        cohort.update(
            {
                "total_cells": total,
                "source_slide": "SO_1",
            }
        )
        manifest["routing_audit"] = {
            "source_slide": "SO_1",
            "mapped_core_numbers": list(SO1_CORE_NUMBERS),
            "mapped_fov_count": 205,
            "selected_cell_count": total,
            "per_core_cell_counts": {
                str(core): len(coordinate_overrides.get(alias, _coordinates()))
                for core, alias in zip(SO1_CORE_NUMBERS, aliases, strict=True)
            },
            "raw_slide_cell_count": total,
            "selection_is_slide_qualified": True,
            "all_raw_fovs_mapped": True,
            "unmapped_fovs": [],
            "unmapped_fov_count": 0,
            "unmapped_fov_cell_count": 0,
            "unmapped_fov_entered_prepared_arrays": False,
        }
        manifest["features"] = {
            "n_biological_probes": 1_000,
            "coordinates_are_model_covariates": False,
            "routing_keys_are_model_covariates": False,
            "core_alias_is_model_covariate": False,
            "slide_identity_is_model_covariate": False,
            "library_size_is_model_covariate": False,
        }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _prepare_cancer_sources(tmp_path: Path) -> tuple[Path, Path]:
    cancer_cohort = tmp_path / "cancer_cohort"
    _write_cohort(cancer_cohort, CANCER_ALIASES, so1=False)
    cancer_graphs = tmp_path / "cancer_graphs"
    prepare_cancer_6core_relative_graphs(
        cohort_dir=cancer_cohort,
        output_dir=cancer_graphs,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
    )
    return cancer_cohort, cancer_graphs


def _lock_synthetic_so1_counts(monkeypatch: pytest.MonkeyPatch) -> int:
    per_core = {core: len(_coordinates()) for core in SO1_CORE_NUMBERS}
    total = sum(per_core.values())
    monkeypatch.setattr(graph_module, "EXPECTED_CELL_COUNTS_BY_CORE", per_core)
    monkeypatch.setattr(graph_module, "EXPECTED_TOTAL_CELLS", total)
    return total


def test_so1_core_graph_uses_locked_radial_relative_geometry(tmp_path: Path) -> None:
    output = tmp_path / "SO1-C01"
    record = materialize_so1_core_relative_graph(
        alias="SO1-C01",
        coordinates_um=_coordinates(),
        output_dir=output,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
    )
    edge_index = np.load(output / "edge_index.npy", mmap_mode="r")
    geometry = np.load(output / "relative_geometry.npy", mmap_mode="r")
    assert geometry.shape == (edge_index.shape[1], RELATIVE_GEOMETRY_DIM)
    assert record["artifact_kind"] == "so1_core_radial_relative_geometry"
    assert record["graph"]["shell_quotas"] == [48, 64, 48, 40]
    assert record["graph"]["nominal_pre_symmetrization_degree"] == 200
    assert record["graph"]["maximum_distance_um"] == 500.0
    assert record["graph"]["qc"]["self_loops"] == 0
    assert record["graph"]["qc"]["cross_group_edges"] == 0

    with pytest.raises(ValueError, match="Unknown core alias"):
        materialize_so1_core_relative_graph(
            alias="CAN-01",
            coordinates_um=_coordinates(),
            output_dir=tmp_path / "bad",
        )


def test_collection_reuses_only_verified_identical_cancer_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancer_cohort, cancer_graphs = _prepare_cancer_sources(tmp_path)
    so1_cohort = tmp_path / "so1_cohort"
    _write_cohort(so1_cohort, SO1_ALIASES, so1=True)
    expected_total = _lock_synthetic_so1_counts(monkeypatch)
    so1_graphs = tmp_path / "so1_graphs"
    manifest = prepare_so1_14core_relative_graphs(
        cohort_dir=so1_cohort,
        output_dir=so1_graphs,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
        reuse_cancer_cohort_dir=cancer_cohort,
        reuse_cancer_graph_dir=cancer_graphs,
    )

    reused = {
        row["alias"]: row
        for row in manifest["reuse_audit"]["results"]
        if row["status"] == "reused"
    }
    assert set(reused) == {"SO1-C01", "SO1-C09", "SO1-C13"}
    for so1_alias, cancer_alias in {
        "SO1-C01": "CAN-01",
        "SO1-C09": "CAN-09",
        "SO1-C13": "CAN-13",
    }.items():
        assert (
            (cancer_graphs / "cores" / cancer_alias / "relative_geometry.npy")
            .stat()
            .st_ino
            == (so1_graphs / "cores" / so1_alias / "relative_geometry.npy")
            .stat()
            .st_ino
        )
    record = json.loads(
        (so1_graphs / "cores/SO1-C09/manifest.json").read_text()
    )
    assert record["alias"] == "SO1-C09"
    assert record["reuse_provenance"]["source_alias"] == "CAN-09"
    assert record["reuse_provenance"]["reuse_method"] == (
        "checksum_verified_hard_link"
    )
    assert manifest["artifact_kind"] == "so1_14core_relative_graph_collection"

    code = """
import json
import sys
import spatial_benchmark.so1_relative_graphs as module
module.EXPECTED_TOTAL_CELLS = int(sys.argv[3])
module.EXPECTED_CELL_COUNTS_BY_CORE = {
    core: int(sys.argv[4]) for core in module.SO1_CORE_NUMBERS
}
batches = module.load_so1_relative_qkv_batches(
    cohort_dir=sys.argv[1], graph_dir=sys.argv[2]
)
print(json.dumps({
    'aliases': [batch.alias for batch in batches],
    'devices': [batch.relative_geometry.device.type for batch in batches],
    'dimensions': [batch.relative_geometry.shape[1] for batch in batches],
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(so1_cohort),
            str(so1_graphs),
            str(expected_total),
            str(len(_coordinates())),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    loaded = json.loads(completed.stdout)
    assert tuple(loaded["aliases"]) == SO1_ALIASES
    assert set(loaded["devices"]) == {"cpu"}
    assert set(loaded["dimensions"]) == {RELATIVE_GEOMETRY_DIM}


def test_coordinate_mismatch_rejects_one_reuse_and_regenerates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancer_cohort, cancer_graphs = _prepare_cancer_sources(tmp_path)
    shifted = _coordinates().copy()
    shifted[0, 0] += 1.0
    so1_cohort = tmp_path / "so1_cohort"
    _write_cohort(
        so1_cohort,
        SO1_ALIASES,
        so1=True,
        coordinate_overrides={"SO1-C09": shifted},
    )
    expected_total = _lock_synthetic_so1_counts(monkeypatch)
    so1_graphs = tmp_path / "so1_graphs"
    manifest = prepare_so1_14core_relative_graphs(
        cohort_dir=so1_cohort,
        output_dir=so1_graphs,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
        reuse_cancer_cohort_dir=cancer_cohort,
        reuse_cancer_graph_dir=cancer_graphs,
    )

    by_alias = {row["alias"]: row for row in manifest["reuse_audit"]["results"]}
    assert by_alias["SO1-C01"]["status"] == "reused"
    assert by_alias["SO1-C09"]["status"] == "rejected_regenerated"
    assert "Ordered coordinates differ" in by_alias["SO1-C09"]["reason"]
    assert by_alias["SO1-C13"]["status"] == "reused"
    assert (
        (cancer_graphs / "cores/CAN-09/relative_geometry.npy").stat().st_ino
        != (so1_graphs / "cores/SO1-C09/relative_geometry.npy").stat().st_ino
    )


def test_tampered_source_cache_rejects_entire_reuse_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancer_cohort, cancer_graphs = _prepare_cancer_sources(tmp_path)
    source = cancer_graphs / "cores/CAN-01/edge_index.npy"
    content = bytearray(source.read_bytes())
    content[-1] ^= 1
    source.write_bytes(content)

    so1_cohort = tmp_path / "so1_cohort"
    _write_cohort(so1_cohort, SO1_ALIASES, so1=True)
    expected_total = _lock_synthetic_so1_counts(monkeypatch)
    so1_graphs = tmp_path / "so1_graphs"
    manifest = prepare_so1_14core_relative_graphs(
        cohort_dir=so1_cohort,
        output_dir=so1_graphs,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
        reuse_cancer_cohort_dir=cancer_cohort,
        reuse_cancer_graph_dir=cancer_graphs,
    )

    audit = manifest["reuse_audit"]
    assert audit["source_collection_status"] == "rejected_regenerate_all"
    assert {row["status"] for row in audit["results"]} == {
        "not_requested_or_source_rejected_regenerated"
    }


def test_parameter_mismatch_rejects_reuse_and_regenerates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancer_cohort, cancer_graphs = _prepare_cancer_sources(tmp_path)
    so1_cohort = tmp_path / "so1_cohort"
    _write_cohort(so1_cohort, SO1_ALIASES, so1=True)
    _lock_synthetic_so1_counts(monkeypatch)
    original_parameters = graph_module.relative_geometry_parameter_record()
    monkeypatch.setattr(
        graph_module,
        "relative_geometry_parameter_record",
        lambda: {**original_parameters, "intentional_test_mismatch": True},
    )
    so1_graphs = tmp_path / "so1_graphs"
    manifest = prepare_so1_14core_relative_graphs(
        cohort_dir=so1_cohort,
        output_dir=so1_graphs,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
        reuse_cancer_cohort_dir=cancer_cohort,
        reuse_cancer_graph_dir=cancer_graphs,
    )

    by_alias = {row["alias"]: row for row in manifest["reuse_audit"]["results"]}
    for alias in ("SO1-C01", "SO1-C09", "SO1-C13"):
        assert by_alias[alias]["status"] == "rejected_regenerated"
        assert "Relative-geometry parameters changed" in by_alias[alias]["reason"]


def test_cohort_verifier_requires_zero_unmapped_fovs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    so1_cohort = tmp_path / "so1_cohort"
    _write_cohort(so1_cohort, SO1_ALIASES, so1=True)
    manifest_path = so1_cohort / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["routing_audit"]["unmapped_fov_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _lock_synthetic_so1_counts(monkeypatch)

    with pytest.raises(
        SO1RelativeGraphContractError,
        match="complete FOV-routing audit",
    ):
        prepare_so1_14core_relative_graphs(
            cohort_dir=so1_cohort,
            output_dir=tmp_path / "graphs",
            receiver_chunk_size=4,
            max_edges_per_chunk=80,
        )


def test_materialization_cli_exposes_so1_defaults() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts/data/materialize_so1_14core_relative_graphs.py"
            ),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    assert "CAN-01" in completed.stdout
    assert "CAN-09" in completed.stdout
    assert "CAN-13" in completed.stdout
    assert "--no-reuse-cancer-graphs" in completed.stdout
