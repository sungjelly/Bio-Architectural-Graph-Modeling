from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import spatial_benchmark.so2_relative_graphs as graph_module
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES
from spatial_benchmark.cancer_relative_graphs import (
    prepare_cancer_6core_relative_graphs,
)
from spatial_benchmark.relative_geometry import RELATIVE_GEOMETRY_DIM
from spatial_benchmark.so2_pooled_full_core import SO2_ALIASES
from spatial_benchmark.so2_relative_graphs import (
    materialize_so2_core_relative_graph,
    prepare_so2_14core_relative_graphs,
)


def _coordinates() -> np.ndarray:
    x, y = np.meshgrid(np.arange(4) * 25.0, np.arange(3) * 25.0)
    return np.column_stack([x.ravel(), y.ravel()]).astype(np.float64)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_cohort(
    root: Path,
    aliases: tuple[str, ...],
    *,
    so2: bool,
) -> None:
    (root / "cores").mkdir(parents=True)
    coordinates = _coordinates()
    files = {}
    rows = []
    for alias in aliases:
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
    cohort = {
        "aliases": list(aliases),
        "validation_or_test_partition_present": False,
    }
    manifest = {"cohort": cohort, "cores": rows, "files": files}
    if so2:
        total = len(aliases) * len(coordinates)
        cohort["total_cells"] = total
        manifest["routing_audit"] = {
            "explicitly_excluded_unmapped_fov": 246,
            "unmapped_fov_entered_prepared_arrays": False,
            "selected_cell_count": total,
        }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_so2_core_graph_uses_locked_radial_relative_geometry(tmp_path: Path) -> None:
    output = tmp_path / "SO2-C15"
    record = materialize_so2_core_relative_graph(
        alias="SO2-C15",
        coordinates_um=_coordinates(),
        output_dir=output,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
    )
    edge_index = np.load(output / "edge_index.npy", mmap_mode="r")
    geometry = np.load(output / "relative_geometry.npy", mmap_mode="r")
    assert geometry.shape == (edge_index.shape[1], RELATIVE_GEOMETRY_DIM)
    assert record["artifact_kind"] == "so2_core_radial_relative_geometry"
    assert record["graph"]["shell_quotas"] == [48, 64, 48, 40]
    assert record["graph"]["nominal_pre_symmetrization_degree"] == 200
    assert record["graph"]["maximum_distance_um"] == 500.0
    assert record["graph"]["qc"]["self_loops"] == 0
    assert record["graph"]["qc"]["cross_group_edges"] == 0

    with pytest.raises(ValueError, match="Unknown core alias"):
        materialize_so2_core_relative_graph(
            alias="CAN-15",
            coordinates_um=_coordinates(),
            output_dir=tmp_path / "bad",
        )


def test_collection_reuses_only_verified_identical_cancer_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancer_cohort = tmp_path / "cancer_cohort"
    _write_cohort(cancer_cohort, CANCER_ALIASES, so2=False)
    cancer_graphs = tmp_path / "cancer_graphs"
    prepare_cancer_6core_relative_graphs(
        cohort_dir=cancer_cohort,
        output_dir=cancer_graphs,
        receiver_chunk_size=4,
        max_edges_per_chunk=80,
    )

    so2_cohort = tmp_path / "so2_cohort"
    _write_cohort(so2_cohort, SO2_ALIASES, so2=True)
    expected_total = len(SO2_ALIASES) * len(_coordinates())
    monkeypatch.setattr(graph_module, "EXPECTED_TOTAL_CELLS", expected_total)
    so2_graphs = tmp_path / "so2_graphs"
    manifest = prepare_so2_14core_relative_graphs(
        cohort_dir=so2_cohort,
        output_dir=so2_graphs,
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
    assert set(reused) == {"SO2-C15", "SO2-C21", "SO2-C23"}
    assert (
        (cancer_graphs / "cores/CAN-15/relative_geometry.npy").stat().st_ino
        == (so2_graphs / "cores/SO2-C15/relative_geometry.npy").stat().st_ino
    )
    record = json.loads(
        (so2_graphs / "cores/SO2-C15/manifest.json").read_text()
    )
    assert record["alias"] == "SO2-C15"
    assert record["reuse_provenance"]["source_alias"] == "CAN-15"
    assert record["reuse_provenance"]["reuse_method"] == (
        "checksum_verified_hard_link"
    )

    # Exercise the read-only mmap loader in a child process so PyTorch's one-time
    # read-only-array warning cannot consume the legacy loader regression's warning.
    code = """
import json
import sys
import spatial_benchmark.so2_relative_graphs as module
module.EXPECTED_TOTAL_CELLS = int(sys.argv[3])
batches = module.load_so2_relative_qkv_batches(
    cohort_dir=sys.argv[1], graph_dir=sys.argv[2]
)
print(json.dumps({
    'aliases': [batch.alias for batch in batches],
    'devices': [batch.relative_geometry.device.type for batch in batches],
    'dimensions': [batch.relative_geometry.shape[1] for batch in batches],
}))
"""
    environment = dict(os.environ)
    source_root = Path(__file__).resolve().parents[3] / "src"
    environment["PYTHONPATH"] = str(source_root)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(so2_cohort),
            str(so2_graphs),
            str(expected_total),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    loaded = json.loads(completed.stdout)
    assert tuple(loaded["aliases"]) == SO2_ALIASES
    assert set(loaded["devices"]) == {"cpu"}
    assert set(loaded["dimensions"]) == {RELATIVE_GEOMETRY_DIM}
