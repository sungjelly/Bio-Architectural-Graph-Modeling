from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.geometry_viz import (  # noqa: E402
    GeometryVisualizationError,
    load_geometry_data,
    render_geometry_report,
    select_dense_window,
)


RESTRICTED_LABEL = "restricted-donor-core-label"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_geometry_artifacts(
    root: Path,
    *,
    cross_split_edge: bool = False,
) -> tuple[Path, Path]:
    prepared_root = root / "prepared"
    graph_root = root / "graph_grid"
    prepared_root.mkdir(parents=True)
    graph_root.mkdir(parents=True)
    x_grid, y_grid = np.meshgrid(
        np.arange(6, dtype=np.float64) * 10.0,
        np.arange(6, dtype=np.float64) * 10.0,
        indexing="xy",
    )
    coordinates = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    splits = np.asarray(
        ["train"] * 12 + ["val"] * 12 + ["test"] * 12,
        dtype="U5",
    )
    macroblocks = np.asarray(
        [
            f"{RESTRICTED_LABEL}-{row // 2}-{column // 2}"
            for row in range(6)
            for column in range(6)
        ],
        dtype="U96",
    )
    prepared_path = prepared_root / "prepared_data.npz"
    np.savez_compressed(
        prepared_path,
        coordinates_um=coordinates,
        split_labels=splits,
        macroblock_ids=macroblocks,
        target_expression=np.asarray([object()], dtype=object),
    )
    prepared_id = "a1b2c3d4e5f60718"
    split_id = "1029384756abcdef"
    prepared_manifest = {
        "format_version": 1,
        "artifact_kind": (
            "normal_true_tissue_spatial_benchmark_preparation"
        ),
        "artifact_id": prepared_id,
        "split": {"split_id": split_id},
        "files": {"prepared_data.npz": _sha256(prepared_path)},
    }
    (prepared_root / "manifest.json").write_text(
        json.dumps(prepared_manifest, sort_keys=True),
        encoding="utf-8",
    )

    undirected_edges: list[tuple[int, int]] = []
    for row in range(6):
        for column in range(6):
            node = row * 6 + column
            if column + 1 < 6:
                undirected_edges.append((node, node + 1))
            if row + 1 < 6 and splits[node] == splits[node + 6]:
                undirected_edges.append((node, node + 6))
    if cross_split_edge:
        undirected_edges.append((11, 12))
    directed_edges = [
        edge
        for source, target in undirected_edges
        for edge in ((source, target), (target, source))
    ]
    edge_index = np.asarray(directed_edges, dtype=np.int64).T
    source, target = edge_index
    distances = np.linalg.norm(
        coordinates[source] - coordinates[target],
        axis=1,
    ).astype(np.float32)
    graph_path = graph_root / "k4_r15_union.npz"
    np.savez_compressed(
        graph_path,
        edge_index=edge_index,
        edge_attributes_raw=distances[:, None],
        edge_attribute_names=np.asarray(["distance_um"], dtype="U64"),
    )
    graph_manifest = {
        "format_version": 1,
        "artifact_kind": "geometry_only_graph_grid",
        "artifact_id": "1122334455667788",
        "prepared_artifact_id": prepared_id,
        "split_id": split_id,
        "n_cells": len(coordinates),
        "test_expression_targets_evaluated": False,
        "files": {graph_path.name: _sha256(graph_path)},
    }
    (graph_root / "manifest.json").write_text(
        json.dumps(graph_manifest, sort_keys=True),
        encoding="utf-8",
    )
    return prepared_path, graph_path


def test_geometry_report_is_atomic_auditable_and_identifier_safe(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    prepared, graph = _write_geometry_artifacts(tmp_path)
    data = load_geometry_data(prepared, graph)
    assert data.qc["n_nodes"] == 36
    assert data.qc["n_components"] == 3
    assert data.qc["cross_split_edges"] == 0
    assert data.qc["edge_distance_p95_um"] == pytest.approx(10.0)
    assert data.n_macroblocks == 9
    assert set(data.provenance["loaded_prepared_array_keys"]) == {
        "coordinates_um",
        "split_labels",
        "macroblock_ids",
    }
    assert data.provenance["expression_arrays_accessed"] is False

    first_window = select_dense_window(
        data.coordinates_um,
        window_size_um=30.0,
    )
    second_window = select_dense_window(
        data.coordinates_um,
        window_size_um=30.0,
    )
    assert first_window.to_manifest() == second_window.to_manifest()
    np.testing.assert_array_equal(
        first_window.node_indices,
        second_window.node_indices,
    )

    output = tmp_path / "geometry_report"
    rendered = render_geometry_report(
        prepared,
        graph,
        output,
        window_size_um=30.0,
        dpi=180,
    )
    assert rendered == output.resolve()
    png = output / "geometry_panels.png"
    pdf = output / "geometry_panels.pdf"
    manifest_path = output / "manifest.json"
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.stat().st_size > 20_000
    assert pdf.stat().st_size > 10_000
    assert manifest_path.stat().st_size < 20_000

    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert RESTRICTED_LABEL not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["geometry_only"] is True
    assert manifest["provenance"]["biological_or_sample_labels_emitted"] is False
    assert manifest["provenance"]["expression_arrays_accessed"] is False
    assert manifest["geometry_summary"]["n_components"] == 3
    assert manifest["dense_window"]["selection"].startswith(
        "maximum cell count"
    )
    assert manifest["files"] == {
        "geometry_panels.png": _sha256(png),
        "geometry_panels.pdf": _sha256(pdf),
    }
    assert len(manifest["artifact_id"]) == 16
    assert not list(tmp_path.glob(".geometry_report.tmp-*"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        render_geometry_report(prepared, graph, output)


def test_invalid_topology_fails_without_publishing_output(
    tmp_path: Path,
) -> None:
    prepared, graph = _write_geometry_artifacts(
        tmp_path,
        cross_split_edge=True,
    )
    output = tmp_path / "invalid_report"
    with pytest.raises(
        GeometryVisualizationError,
        match="cross-split edges",
    ):
        render_geometry_report(prepared, graph, output, dpi=150)
    assert not output.exists()
    assert not list(tmp_path.glob(".invalid_report.tmp-*"))


def test_geometry_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "analysis" / "visualize_geometry.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--prepared-data" in result.stdout
    assert "--window-size-um" in result.stdout
    assert "without reading expression outcomes" in result.stdout
