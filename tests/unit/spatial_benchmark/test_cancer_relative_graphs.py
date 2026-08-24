from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from spatial_benchmark.cancer_relative_graphs import (
    load_cancer_relative_qkv_batches,
    materialize_cancer_core_relative_graph,
    prepare_cancer_6core_relative_graphs,
)
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES
from spatial_benchmark.relative_geometry import RELATIVE_GEOMETRY_DIM


def _coordinates() -> np.ndarray:
    x, y = np.meshgrid(np.arange(6) * 25.0, np.arange(5) * 25.0)
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)


def test_one_shared_relative_geometry_cache_is_complete_and_checksum_bound(
    tmp_path: Path,
) -> None:
    output = tmp_path / "CAN-01"
    record = materialize_cancer_core_relative_graph(
        alias="CAN-01",
        coordinates_um=_coordinates(),
        output_dir=output,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
    )

    edge_index = np.load(output / "edge_index.npy", mmap_mode="r")
    geometry = np.load(output / "relative_geometry.npy", mmap_mode="r")
    assert edge_index.shape[0] == 2
    assert geometry.shape == (edge_index.shape[1], RELATIVE_GEOMETRY_DIM)
    assert geometry.dtype == np.float32
    assert np.isfinite(geometry).all()
    assert not np.any(edge_index[0] == edge_index[1])
    assert record["graph"]["qc"]["cross_group_edges"] == 0
    assert record["graph"]["qc"]["self_loops"] == 0
    assert record["relative_geometry"]["cache_policy"] == (
        "one_shared_read_only_memory_map_per_core"
    )
    assert record["relative_geometry"]["copied_into_per_seed_run_bundles"] is False
    persisted = json.loads((output / "manifest.json").read_text())
    assert persisted["record_sha256"] == record["record_sha256"]
    assert set(record["files"]) == {
        "edge_index.npy",
        "orientation.npz",
        "relative_geometry.npy",
    }


def test_graph_cache_is_immutable_and_alias_checked(tmp_path: Path) -> None:
    output = tmp_path / "CAN-01"
    materialize_cancer_core_relative_graph(
        alias="CAN-01",
        coordinates_um=_coordinates(),
        output_dir=output,
    )
    with pytest.raises(FileExistsError):
        materialize_cancer_core_relative_graph(
            alias="CAN-01",
            coordinates_um=_coordinates(),
            output_dir=output,
        )
    with pytest.raises(ValueError, match="Unknown Cancer core alias"):
        materialize_cancer_core_relative_graph(
            alias="CAN-99",
            coordinates_um=_coordinates(),
            output_dir=tmp_path / "bad",
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_six_core_collection_completes_manifest_and_loads_shared_mmaps(
    tmp_path: Path,
) -> None:
    cohort = tmp_path / "cohort"
    (cohort / "cores").mkdir(parents=True)
    files = {}
    core_rows = []
    coordinates = _coordinates()[:12]
    for alias in CANCER_ALIASES:
        path = cohort / "cores" / f"{alias}.npz"
        np.savez_compressed(
            path,
            target_expression=np.ones((len(coordinates), 1_000), dtype=np.float32),
            node_covariates=np.zeros((len(coordinates), 2), dtype=np.float32),
            coordinates_um=coordinates,
        )
        files[f"cores/{alias}.npz"] = _sha256(path)
        core_rows.append(
            {
                "alias": alias,
                "cell_count": len(coordinates),
                "graph_checksum": None,
            }
        )
    (cohort / "manifest.json").write_text(
        json.dumps(
            {
                "cohort": {
                    "aliases": list(CANCER_ALIASES),
                    "validation_or_test_partition_present": False,
                },
                "cores": core_rows,
                "files": files,
            }
        )
    )

    graph_root = tmp_path / "graphs"
    manifest = prepare_cancer_6core_relative_graphs(
        cohort_dir=cohort,
        output_dir=graph_root,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
    )
    assert tuple(manifest["aliases"]) == CANCER_ALIASES
    completed = json.loads(
        (graph_root / "cohort_manifest_with_graphs.json").read_text()
    )
    assert all(row["graph_checksum"] for row in completed["cores"])

    with pytest.warns(UserWarning, match="not writable"):
        batches = load_cancer_relative_qkv_batches(
            cohort_dir=cohort,
            graph_dir=graph_root,
        )
    assert tuple(batch.alias for batch in batches) == CANCER_ALIASES
    assert all(batch.relative_geometry.device.type == "cpu" for batch in batches)
    assert all(batch.relative_geometry.shape[1] == RELATIVE_GEOMETRY_DIM for batch in batches)
