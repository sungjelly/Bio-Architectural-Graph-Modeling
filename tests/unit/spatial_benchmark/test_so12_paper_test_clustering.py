import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from scripts.analysis import so12_paper_test_clustering as analysis
from spatial_benchmark.data import DEFAULT_PIXEL_SIZE_UM


@pytest.fixture
def extracted(tmp_path, monkeypatch):
    """Two planted groups, each represented equally in both test cores."""
    monkeypatch.setenv("BAGM_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setattr(analysis, "EXPECTED_CORE_COUNTS", {15: 32, 23: 32})
    output = tmp_path / "scratch/active_runs/synthetic_test"
    (output / "embeddings").mkdir(parents=True)
    cores = np.repeat([15, 23], 32)
    cells = pd.DataFrame({"row_index": np.arange(64), "slide": 2, "fov": cores,
        "cell_ID": np.tile(np.arange(32), 2), "core_number": cores,
        "alias": [f"SO2-C{core}" for core in cores], "split": "test",
        "x_um": np.tile(np.arange(32) % 8, 2) * 15.0,
        "y_um": np.tile(np.arange(32) // 8, 2) * 15.0 + np.repeat([0., 100.], 32)})
    cells.to_csv(output / "cells.csv.gz", index=False)
    planted = np.tile(np.repeat([0, 1], 16), 2)
    values = np.random.default_rng(42).normal(0, .03, (64, 256)).astype(np.float32)
    values[np.arange(64), planted] += 3
    np.save(output / "embeddings/hL.npy", values)
    core_receipts = []
    for position, core in enumerate((15, 23)):
        path = output / f"embeddings/core_{core}_receipt.json"
        path.write_text(json.dumps({"core_number": core, "cells": 32}))
        core_receipts.append({"core_number": core, "n_cells": 32, "row_start": position * 32,
                              "row_stop": (position + 1) * 32, "receipt_path": str(path.relative_to(output)),
                              "receipt_sha256": analysis.sha(path)})
    manifest = {"status": "complete", "passed": True, "verified": True,
        "source_run_id": analysis.SOURCE_RUN_ID, "n_cells": 64, "dimensions": 256,
        "test_core_numbers": [15, 23], "all_observed": True, "checkpoint_sha256": "a" * 64,
        "preprocessing_fingerprint": "b" * 64, "fitted_parameters_sha256": "c" * 64,
        "cells_sha256": analysis.sha(output / "cells.csv.gz"),
        "embedding_sha256": analysis.sha(output / "embeddings/hL.npy"), "cores": core_receipts}
    analysis.dump(output / "extraction_manifest.json", manifest)
    return output, cells, values, planted


def test_cell_contract_rejects_reordering_duplicates_wrong_split_and_core(extracted):
    output, original, _, _ = extracted
    cells = original.rename(columns={"alias": "core_alias"})
    analysis.validate_test_cells(cells)
    bad = cells.iloc[::-1]
    with pytest.raises(ValueError, match="canonical row order"):
        analysis.validate_test_cells(bad)
    for column, value, message in (("core_number", 5, "test cores"), ("slide", 1, "wrong-slide"),
                                   ("split", "validation", "Non-test"),
                                   ("core_alias", "SO2-C5", "alias mismatch"),
                                   ("x_um", np.nan, "Missing")):
        bad = cells.copy()
        bad.loc[0, column] = value
        with pytest.raises(ValueError, match=message):
            analysis.validate_test_cells(bad)
    bad = cells.copy()
    bad.loc[1, ["slide", "fov", "cell_ID"]] = bad.loc[0, ["slide", "fov", "cell_ID"]].to_numpy()
    with pytest.raises(ValueError, match="Duplicate slide-qualified"):
        analysis.validate_test_cells(bad)


def test_extraction_checksums_and_nonzero_embeddings_are_required(extracted):
    output, cells, values, _ = extracted
    loaded_cells, loaded, bindings, _ = analysis.load_inputs(output)
    assert loaded_cells.core_alias.iloc[0] == "SO2-C15"
    np.testing.assert_array_equal(loaded, values)
    assert "extraction_manifest.json" in bindings["files"]
    values[0] = 0
    np.save(output / "embeddings/hL.npy", values)
    with pytest.raises(ValueError, match="Embedding checksum"):
        analysis.load_inputs(output)
    manifest = json.loads((output / "extraction_manifest.json").read_text())
    manifest["embedding_sha256"] = analysis.sha(output / "embeddings/hL.npy")
    analysis.dump(output / "extraction_manifest.json", manifest)
    with pytest.raises(ValueError, match="Zero-norm"):
        analysis.load_inputs(output)


def test_real_clustering_recovers_planted_groups_across_cores_and_replays(extracted):
    output, cells, _, planted = extracted
    result = analysis.cluster(output, device="cpu", query_chunk=13, threads=1)
    assert result["pilot_cpu_exact_audit"]["passed"]
    assert result["cpu_exact_audit"]["passed"]
    assert result["search"]["maximum_explicit_pairwise_array_shape"] == [13, 64]
    assert result["metrics"]["seed0_seed1_ari"] == 1
    assert result["metrics"]["cross_core_undirected_edge_fraction"] > .4
    assert result["metrics"]["clusters_over_90_percent_one_core"] == []
    labels = np.load(output / "clustering/hL_labels_seed0.npy")
    assert adjusted_rand_score(planted, labels) == 1
    table = pd.read_csv(output / "clustering/clusters.csv.gz")
    np.testing.assert_array_equal(table[analysis.KEYS], cells[analysis.KEYS])
    neighbors = np.load(output / "clustering/hL_knn_indices.npy")
    assert neighbors.shape == (64, 30)
    assert not np.any(neighbors == np.arange(64)[:, None])
    edges = np.load(output / "clustering/hL_undirected_edges.npy")
    assert np.all(edges[:, 0] < edges[:, 1])
    np.testing.assert_array_equal(planted[edges[:, 0]], planted[edges[:, 1]])
    replay = analysis.cluster(output, device="cpu", query_chunk=13, threads=1)
    assert replay == result
    summary = json.loads((output / "clustering_summary.json").read_text())
    assert summary["passed"] and summary["n_cells"] == 64
    # A completed stage must not silently accept changed output labels.
    labels[0] = 1 - labels[0]
    np.save(output / "clustering/hL_labels_seed0.npy", labels)
    with pytest.raises(ValueError, match="Artifact checksum changed"):
        analysis.cluster(output, device="cpu", threads=1)


def test_failed_cpu_pilot_stops_before_full_search(extracted, monkeypatch):
    output = extracted[0]
    calls = []
    exact = analysis.exact_cosine_knn

    def counted(*args, **kwargs):
        calls.append(kwargs.get("query_indices"))
        return exact(*args, **kwargs)

    monkeypatch.setattr(analysis, "exact_cosine_knn", counted)
    monkeypatch.setattr(analysis, "cpu_neighbor_audit", lambda *args, **kwargs: {"passed": False})
    with pytest.raises(ValueError, match="full search was not started"):
        analysis.cluster(output, device="cpu", threads=1)
    assert len(calls) == 1 and calls[0] is not None
    assert not (output / "clustering/hL_knn_indices.npy").exists()
    assert not (output / "clustering_summary.json").exists()


def test_partition_summary_preserves_weak_stability_and_core_segregation(extracted):
    cells = extracted[1].rename(columns={"alias": "core_alias"})
    primary = np.repeat([0, 1], 32)
    secondary = np.tile([0, 1], 32)
    table, sizes, composition, metrics = analysis.partition_tables(cells, primary, secondary)
    assert metrics["seed0_seed1_ari"] < 0
    assert metrics["clusters_over_90_percent_one_core"] == [0, 1]
    assert metrics["seed0_cluster_vs_core_ari_descriptive"] == 1
    assert sizes.cells.sum() == len(cells)
    assert composition.cells.sum() == len(cells)
    np.testing.assert_array_equal(table["leiden_r1.0_s0"], primary)
    with pytest.raises(ValueError, match="coverage"):
        analysis.partition_tables(cells, primary[:-1], secondary)


def test_render_uses_all_actual_polygons_shared_palette_and_verified_exports(extracted, tmp_path):
    output, cells, _, _ = extracted
    analysis.cluster(output, device="cpu", query_chunk=13, threads=1)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    vertices = []
    for row in cells.itertuples():
        for dx, dy in ((-2, -2), (2, -2), (2, 2), (-2, 2)):
            vertices.append({"fov": row.fov, "cellID": row.cell_ID,
                "x_global_px": (row.x_um + dx) / DEFAULT_PIXEL_SIZE_UM,
                "y_global_px": (row.y_um + dy) / DEFAULT_PIXEL_SIZE_UM})
    source = raw / "26040302SO_2-polygons.csv"
    pd.DataFrame(vertices).to_csv(source, index=False)
    manifest = analysis.render(output, paths=SimpleNamespace(data_root=raw.parent))
    assert manifest["qc"]["cells"] == 64
    assert manifest["qc"]["covered_cells_per_core"] == {"15": 32, "23": 32}
    assert manifest["qc"]["shared_palette"] and not manifest["qc"]["polygons_sampled"]
    assert manifest["qc"]["slide_qualified_key_order_verified"]
    assert manifest["qc"]["centroid_mismatch_count"] == 0
    assert manifest["qc"]["manual_visual_review"] == "not_performed"
    for check in manifest["qc"]["export_checks"].values():
        assert check["verified"] and min(check["png_dimensions"]) >= 100
    document = (output / "report/index.html").read_text()
    assert "data:image/png;base64," in document
    assert "transductive" in document and "not validated cell types" in document
    summary = json.loads((output / "rendering_summary.json").read_text())
    assert summary["passed"] and summary["cells_per_core"] == {"15": 32, "23": 32}
    # Raw segmentation holes/missing cells must never be replaced by centroids.
    pd.DataFrame(vertices[:-4]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        analysis.render(output, paths=SimpleNamespace(data_root=raw.parent))
