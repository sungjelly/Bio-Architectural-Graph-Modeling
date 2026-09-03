from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spatial_benchmark.cli import build_parser
import spatial_benchmark.so2_hl_umap as umap
from spatial_benchmark.so2_pooled_full_core import SO2_CORE_NUMBERS


def _normalized_fixture(*, rows: int = 72, dimensions: int = 8) -> np.ndarray:
    values = np.random.default_rng(3817).normal(size=(rows, dimensions)).astype(
        np.float32
    )
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values)


def _palette() -> dict[str, str]:
    return {
        f"C{index}": f"#{(0x10203 + index * 0x0B1929) % 0x1000000:06X}"
        for index in range(umap.EXPECTED_CLUSTER_COUNT)
    }


def test_cli_defaults_match_locked_umap_contract() -> None:
    arguments = build_parser().parse_args(["render-so2-hl-umap"])

    assert arguments.command_name == "render-so2-hl-umap"
    assert arguments.n_neighbors == 30
    assert arguments.min_dist == 0.3
    assert arguments.epochs == 200
    assert arguments.random_seed == 20260825
    assert arguments.device == "cpu"
    assert arguments.dpi == 300


def test_parameter_validation_rejects_contract_drift() -> None:
    locked = {
        "n_neighbors": 30,
        "min_dist": 0.3,
        "epochs": 200,
        "random_seed": 20260825,
        "device": "cpu",
        "dpi": 300,
    }
    umap.validate_umap_parameters(**locked)

    for field, changed in (
        ("n_neighbors", 15),
        ("min_dist", 0.1),
        ("epochs", 500),
        ("random_seed", 1),
        ("device", "cuda:0"),
        ("dpi", 600),
    ):
        parameters = dict(locked)
        parameters[field] = changed
        with pytest.raises(ValueError, match="locked|CPU-only"):
            umap.validate_umap_parameters(**parameters)


def test_directed_knn_replay_is_finite_unique_and_checksum_verifiable() -> None:
    scores = _normalized_fixture()
    result = umap.reconstruct_directed_faiss_knn(scores, n_neighbors=7)

    assert result.neighbors.shape == (len(scores), 7)
    assert result.similarities.shape == (len(scores), 7)
    assert np.isfinite(result.similarities).all()
    for row, neighbors in enumerate(result.neighbors):
        assert row not in neighbors
        assert len(np.unique(neighbors)) == 7
    umap.verify_knn_against_source(result, result.receipt)

    changed = dict(result.receipt)
    changed["neighbors_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="neighbors_sha256"):
        umap.verify_knn_against_source(result, changed)


def test_native_igraph_umap_is_exactly_repeatable_without_dense_matrix() -> None:
    scores = _normalized_fixture(rows=60, dimensions=6)
    directed = umap.reconstruct_directed_faiss_knn(scores, n_neighbors=6)
    result = umap.run_native_umap(
        scores,
        directed,
        min_dist=0.3,
        epochs=30,
        random_seed=20260825,
        verify_determinism=True,
    )

    assert result.coordinates.shape == (len(scores), 2)
    assert result.coordinates.dtype == np.float64
    assert np.isfinite(result.coordinates).all()
    assert np.all(np.ptp(result.coordinates, axis=0) > 0.0)
    assert result.receipt["dense_cell_by_cell_matrix_constructed"] is False
    assert result.receipt["distance_array_shape"] == [len(scores) * 6]
    assert result.receipt["coordinate_sha256"] == result.receipt[
        "determinism_repeat_coordinate_sha256"
    ]

    with pytest.raises(ValueError, match="undirected edge count"):
        umap.run_native_umap(
            scores,
            directed,
            min_dist=0.3,
            epochs=30,
            random_seed=20260825,
            verify_determinism=False,
            expected_undirected_edges=int(
                result.receipt["positive_fuzzy_weight_count"]
            )
            + 1,
        )


def test_pca_initialization_is_centered_scaled_and_deterministic() -> None:
    scores = _normalized_fixture(rows=20, dimensions=4)
    first = umap.deterministic_pca_initialization(scores)
    second = umap.deterministic_pca_initialization(scores.copy())

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.mean(axis=0), 0.0, atol=1.0e-12)
    assert np.max(np.abs(first)) == pytest.approx(10.0)


def test_privacy_minimized_table_preserves_only_alignment_fields() -> None:
    frame = pd.DataFrame(
        {
            "global_cell_index": [0, 1, 2],
            "cell_index": [10, 11, 12],
            "cell_key": ["private-a", "private-b", "private-c"],
            "core_number": [15, 15, 16],
            "x_um": [1.0, 2.0, 3.0],
            "y_um": [4.0, 5.0, 6.0],
            "contextual_cluster_number": [0, 1, 0],
            "contextual_cluster": ["C0", "C1", "C0"],
        }
    )
    coordinates = np.asarray([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]], dtype=np.float32)

    output = umap.build_privacy_minimized_table(frame, coordinates)

    assert list(output.columns) == [
        "global_cell_index",
        "core_number",
        "contextual_cluster_number",
        "contextual_cluster",
        "umap_1",
        "umap_2",
    ]
    assert "cell_key" not in output
    assert "cell_index" not in output
    assert "x_um" not in output
    assert "y_um" not in output
    np.testing.assert_array_equal(
        output[["umap_1", "umap_2"]].to_numpy(dtype=np.float64),
        coordinates.astype(np.float64),
    )


def test_static_figure_uses_identical_coordinates_for_cluster_and_core(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    coordinates: list[list[float]] = []
    index = 0
    for core_offset, core in enumerate(SO2_CORE_NUMBERS):
        for cluster in range(umap.EXPECTED_CLUSTER_COUNT):
            rows.append(
                {
                    "global_cell_index": index,
                    "core_number": core,
                    "contextual_cluster_number": cluster,
                    "contextual_cluster": f"C{cluster}",
                }
            )
            coordinates.append(
                [float(cluster + core_offset / 20), float(core_offset - cluster / 10)]
            )
            index += 1
    frame = pd.DataFrame(rows)
    values = np.asarray(coordinates, dtype=np.float32)
    png = tmp_path / "umap.png"
    pdf = tmp_path / "umap.pdf"

    receipt = umap.render_umap_figure(
        coordinates=values,
        frame=frame,
        cluster_palette=_palette(),
        png_path=png,
        pdf_path=pdf,
        random_seed=20260825,
        dpi=72,
    )

    assert png.is_file() and png.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert receipt["plot_type"] == "two_panel_identical_coordinates"
    assert receipt["point_count_per_panel"] == len(frame)
    assert receipt["left_color"].endswith("resolution_1p0")
    assert receipt["right_color"] == "numeric_tissue_core_confounding_control"
    assert receipt["cluster_palette"] == _palette()
    assert list(receipt["core_palette"]) == [str(value) for value in SO2_CORE_NUMBERS]
    assert receipt["png_dpi"] == 72
    assert receipt["pdf_raster_dpi"] == 72


def test_output_override_is_constrained_to_report_analyses(tmp_path: Path) -> None:
    paths = SimpleNamespace(
        project_root=tmp_path,
        report_root=tmp_path / "reports",
    )
    default = umap._resolve_output_root(
        paths=paths, run_id="locked-run", output_dir=None
    )
    assert default == (
        tmp_path
        / "reports"
        / "analyses"
        / umap.DEFAULT_OUTPUT_REPORT
        / "locked-run"
    ).resolve()

    allowed = umap._resolve_output_root(
        paths=paths,
        run_id="locked-run",
        output_dir="reports/analyses/custom/locked-run",
    )
    assert allowed == (tmp_path / "reports/analyses/custom/locked-run").resolve()

    for forbidden in (tmp_path / "data/raw", tmp_path / "reports/analyses"):
        with pytest.raises(ValueError, match="reports/analyses"):
            umap._resolve_output_root(
                paths=paths, run_id="locked-run", output_dir=forbidden
            )


def test_source_binding_detects_persisted_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "bundle"
    (output_root / "umap").mkdir(parents=True)
    (output_root / "tables").mkdir()
    frame = pd.DataFrame(
        {
            "global_cell_index": [0, 1],
            "core_number": np.asarray([15, 16], dtype=np.int16),
            "contextual_cluster_number": np.asarray([0, 1], dtype=np.int32),
            "contextual_cluster": ["C0", "C1"],
        }
    )
    frame.to_parquet(output_root / "tables" / umap.TABLE_FILENAME, index=False)
    source = umap.UMAPSource(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        clustering_manifest_path=tmp_path / "clustering.json",
        pca_path=tmp_path / "scores.npy",
        labels_path=tmp_path / "labels.npy",
        palette_path=tmp_path / "palette.json",
        table_path=tmp_path / "table.parquet",
        manifest={},
        clustering_manifest={},
        scores=np.zeros((2, 2), dtype=np.float32),
        labels=np.asarray([0, 1], dtype=np.int64),
        frame=frame,
        palette=_palette(),
    )
    source_records = {"analysis_manifest": {"sha256": "a", "size_bytes": 1}}
    source_knn = {
        "neighbors_sha256": "neighbors",
        "neighbor_cosine_sha256": "cosines",
        "undirected_edges": 2,
        "undirected_edges_sha256": "union",
    }
    monkeypatch.setattr(umap, "_source_file_records", lambda _source: source_records)
    monkeypatch.setattr(umap, "_source_knn_receipt", lambda _source: source_knn)

    receipt = umap._receipt_with_self_hash(
        {
            "source": source_records,
            "source_knn_hashes": source_knn,
            "figure": {"cluster_palette": _palette()},
        }
    )
    receipt_path = output_root / "umap" / "umap_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    umap.verify_umap_source_binding(output_root, source)

    drifted = dict(receipt)
    drifted.pop("manifest_content_sha256")
    drifted["source"] = {
        "analysis_manifest": {"sha256": "changed", "size_bytes": 1}
    }
    receipt_path.write_text(
        json.dumps(umap._receipt_with_self_hash(drifted)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source artifact"):
        umap.verify_umap_source_binding(output_root, source)
