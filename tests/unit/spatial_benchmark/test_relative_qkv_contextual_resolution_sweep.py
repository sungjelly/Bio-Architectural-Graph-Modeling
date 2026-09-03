from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

import spatial_benchmark.relative_qkv_contextual_resolution_sweep as sweep
from spatial_benchmark.cancer_pooled_full_core import CANCER_ALIASES, CORE_NUMBERS
from spatial_benchmark.relative_qkv_embedding_clustering import (
    KNNGraphResult,
    _array_sha256,
)


def _small_cell_frame(*, cells_per_core: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    global_index = 0
    for core_number, alias in zip(CORE_NUMBERS, CANCER_ALIASES, strict=True):
        for cell_index in range(cells_per_core):
            rows.append(
                {
                    "global_cell_index": global_index,
                    "cell_index": cell_index,
                    "cell_key": f"{alias}:{cell_index:08d}",
                    "core_alias": alias,
                    "core_number": core_number,
                    "x_um": float(core_number * 10 + cell_index),
                    "y_um": float(core_number * 5 + cell_index),
                }
            )
            global_index += 1
    return pd.DataFrame(rows)


def _partition_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, pd.DataFrame, np.ndarray, KNNGraphResult, dict[str, object]]:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    (source_root / "clustering").mkdir(parents=True)
    (output_root / "clustering").mkdir(parents=True)
    frame = _small_cell_frame()
    baseline = np.asarray([0, 0, 1, 1] * 3, dtype=np.int64)
    np.save(source_root / "clustering" / "contextual_labels.npy", baseline)
    (source_root / "clustering" / "contextual_palette.json").write_text(
        json.dumps({"colors": {"C0": "#112233", "C1": "#abcdef"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sweep, "EXPECTED_TOTAL_CELLS", len(frame))
    monkeypatch.setattr(sweep, "_source_cell_frame", lambda _root: frame.copy())
    edge_pairs = np.asarray(
        [[index, index + 1] for index in range(len(frame) - 1)], dtype=np.int64
    )
    graph = KNNGraphResult(
        edge_pairs=edge_pairs,
        receipt={"undirected_edges_sha256": "shared-edges"},
    )
    graph_receipt: dict[str, object] = {
        "manifest_content_sha256": "graph-manifest",
        "knn": {"undirected_edges_sha256": "shared-edges"},
    }
    return source_root, output_root, frame, baseline, graph, graph_receipt


def _source_clustering_with_labels(labels: np.ndarray) -> dict[str, object]:
    return {
        "pipelines": {
            "contextual": {
                "leiden": {
                    "labels_sha256": _array_sha256(
                        "sorted_leiden_labels", labels
                    )
                }
            }
        }
    }


def test_resolution_normalization_sorts_and_tokens_are_stable() -> None:
    assert sweep._normalize_resolutions([2.0, 0.25, 1.0, 0.5]) == (
        0.25,
        0.5,
        1.0,
        2.0,
    )
    assert [
        sweep._resolution_token(value) for value in (0.25, 1.0, 1.25, 2.0)
    ] == ["0p25", "1p0", "1p25", "2p0"]
    assert sweep._resolution_prefix(1.25) == "R1p25_C"


@pytest.mark.parametrize(
    "values",
    (
        [],
        [0.0, 1.0],
        [-0.5, 1.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
        [0.5, 0.5, 1.0],
        [0.5, 1.5],
        [1.0, 1.0000001, 1.0000002],
    ),
)
def test_resolution_normalization_rejects_invalid_or_colliding_values(
    values: list[float],
) -> None:
    with pytest.raises(sweep.ContextualResolutionSweepError, match="Resolution"):
        sweep._normalize_resolutions(values)


def test_contextual_loader_requires_only_hl_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counts = {core_number: 2 for core_number in CORE_NUMBERS}
    monkeypatch.setattr(sweep, "EXPECTED_CELL_COUNTS", counts)
    monkeypatch.setattr(sweep, "EXPECTED_TOTAL_CELLS", 2 * len(CORE_NUMBERS))
    manifest = {
        "embedding_shapes": {
            str(core_number): {"hL": [2, 256]}
            for core_number in CORE_NUMBERS
        }
    }
    for core_number in CORE_NUMBERS:
        path = tmp_path / "embeddings" / f"core_{core_number}_embeddings.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            cell_index=np.arange(2, dtype=np.int64),
            core_number=np.asarray(core_number, dtype=np.int16),
            coordinates_um=np.asarray(
                [[core_number, 0.0], [core_number, 1.0]], dtype=np.float64
            ),
            hL=np.full((2, 256), float(core_number), dtype=np.float32),
        )

    cores = sweep._load_contextual_cores(tmp_path, manifest)

    assert tuple(core.core_number for core in cores) == CORE_NUMBERS
    assert sum(core.n_cells for core in cores) == 2 * len(CORE_NUMBERS)
    assert all(core.hL.shape == (2, 256) for core in cores)
    assert all(not hasattr(core, "h0") for core in cores)
    assert all(not hasattr(core, "delta_h") for core in cores)


def test_compute_partitions_normalizes_order_and_passes_one_graph_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_cells = 12
    monkeypatch.setattr(sweep, "EXPECTED_TOTAL_CELLS", n_cells)
    graph = KNNGraphResult(
        edge_pairs=np.asarray([[index, index + 1] for index in range(11)]),
        receipt={"undirected_edges_sha256": "shared"},
    )
    baseline = np.asarray([0, 0, 1, 1] * 3, dtype=np.int64)
    alternate = np.asarray([0, 1] * 6, dtype=np.int64)
    observed: list[tuple[int, float, int]] = []

    def fake_leiden(
        observed_graph: KNNGraphResult,
        *,
        n_cells: int,
        resolution: float,
        random_seed: int,
    ) -> SimpleNamespace:
        observed.append((id(observed_graph), resolution, n_cells))
        labels = baseline.copy() if resolution == 1.0 else alternate.copy()
        return SimpleNamespace(labels=labels, receipt={"cluster_count": 2})

    monkeypatch.setattr(sweep, "run_seeded_leiden", fake_leiden)
    results = sweep.compute_resolution_partitions(
        graph,
        resolutions=(1.5, 1.0, 0.5),
        random_seed=17,
        baseline_labels=baseline,
    )

    assert observed == [
        (id(graph), 0.5, n_cells),
        (id(graph), 1.0, n_cells),
        (id(graph), 1.5, n_cells),
    ]
    assert list(results) == ["0p5", "1p0", "1p5"]
    assert all(not np.shares_memory(result.labels, baseline) for result in results.values())


def test_partition_stage_reuses_one_graph_for_independent_leiden_runs_and_is_hl_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, output_root, frame, baseline, graph, graph_receipt = (
        _partition_fixture(tmp_path, monkeypatch)
    )
    calls: list[tuple[int, float, int]] = []
    alternate = np.asarray([0, 1] * (len(frame) // 2), dtype=np.int64)

    def fake_leiden(
        observed_graph: KNNGraphResult,
        *,
        n_cells: int,
        resolution: float,
        random_seed: int,
    ) -> SimpleNamespace:
        calls.append((id(observed_graph), resolution, random_seed))
        labels = baseline.copy() if resolution == 1.0 else alternate.copy()
        return SimpleNamespace(
            labels=labels,
            receipt={
                "cluster_count": int(len(np.unique(labels))),
                "labels_sha256": _array_sha256("sorted_leiden_labels", labels),
                "resolution": resolution,
            },
        )

    monkeypatch.setattr(sweep, "run_seeded_leiden", fake_leiden)
    receipt = sweep.build_or_load_resolution_partitions(
        source_root=source_root,
        source_clustering=_source_clustering_with_labels(baseline),
        output_root=output_root,
        graph=graph,
        graph_receipt=graph_receipt,
        resolutions=(0.5, 1.0),
        random_seed=20260825,
    )

    assert calls == [
        (id(graph), 0.5, 20260825),
        (id(graph), 1.0, 20260825),
    ]
    assert receipt["shared_graph_object_reused_for_all_resolutions"] is True
    assert receipt["independent_leiden_partition_per_resolution"] is True
    assert receipt["baseline_labels_exact_match"] is True
    assert receipt["h0_or_delta_used"] is False
    assert receipt["configuration"]["representation"] == "contextual_hL_only"
    assert receipt["configuration"]["resolutions"] == [0.5, 1.0]
    assert set(receipt["partitions"]) == {"0p5", "1p0"}
    assert all(
        partition["leiden"]["resolution"] == expected
        for partition, expected in zip(
            receipt["partitions"].values(), (0.5, 1.0), strict=True
        )
    )

    forbidden = ("intrinsic", "delta", "h0")
    assert not any(
        term in relative.lower()
        for relative in receipt["files"]
        for term in forbidden
    )
    table = pd.read_parquet(
        output_root / "tables" / "cell_contextual_resolution_clusters.parquet"
    )
    assert len(table) == len(frame)
    assert tuple(table["core_number"].drop_duplicates()) == CORE_NUMBERS
    assert not any(
        term in column.lower() for column in table.columns for term in forbidden
    )
    assert np.array_equal(
        table["contextual_cluster_number_r1p0"].to_numpy(), baseline
    )
    assert set(table["contextual_cluster_r0p5"]) == {"R0p5_C0", "R0p5_C1"}
    assert set(table["contextual_cluster_r1p0"]) == {"R1p0_C0", "R1p0_C1"}


def test_partition_stage_rejects_resolution_one_label_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, output_root, frame, baseline, graph, graph_receipt = (
        _partition_fixture(tmp_path, monkeypatch)
    )
    mismatched = np.asarray([0, 1] * (len(frame) // 2), dtype=np.int64)

    monkeypatch.setattr(
        sweep,
        "run_seeded_leiden",
        lambda *_args, **_kwargs: SimpleNamespace(
            labels=mismatched,
            receipt={
                "cluster_count": 2,
                "labels_sha256": _array_sha256(
                    "sorted_leiden_labels", mismatched
                ),
            },
        ),
    )

    with pytest.raises(
        sweep.ContextualResolutionSweepError,
        match="Resolution-1.0 labels do not match",
    ):
        sweep.build_or_load_resolution_partitions(
                source_root=source_root,
                source_clustering=_source_clustering_with_labels(baseline),
            output_root=output_root,
            graph=graph,
            graph_receipt=graph_receipt,
            resolutions=(1.0,),
            random_seed=20260825,
        )

    assert not (
        output_root / "clustering" / "resolution_sweep_manifest.json"
    ).exists()
    assert not any((output_root / "tables").iterdir())


def test_completed_partition_stage_resumes_without_running_leiden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, output_root, frame, baseline, graph, graph_receipt = (
        _partition_fixture(tmp_path, monkeypatch)
    )

    def baseline_leiden(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            labels=baseline.copy(),
            receipt={
                "cluster_count": 2,
                "labels_sha256": _array_sha256(
                    "sorted_leiden_labels", baseline
                ),
            },
        )

    monkeypatch.setattr(sweep, "run_seeded_leiden", baseline_leiden)
    first = sweep.build_or_load_resolution_partitions(
        source_root=source_root,
        source_clustering=_source_clustering_with_labels(baseline),
        output_root=output_root,
        graph=graph,
        graph_receipt=graph_receipt,
        resolutions=(1.0,),
        random_seed=20260825,
    )
    monkeypatch.setattr(
        sweep,
        "run_seeded_leiden",
        lambda *_args, **_kwargs: pytest.fail("Leiden reran on a valid receipt"),
    )
    resumed = sweep.build_or_load_resolution_partitions(
        source_root=source_root,
        source_clustering=_source_clustering_with_labels(baseline),
        output_root=output_root,
        graph=graph,
        graph_receipt=graph_receipt,
        resolutions=(1.0,),
        random_seed=20260825,
    )

    assert resumed == first


def test_partition_receipt_detects_label_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root, output_root, _frame, baseline, graph, graph_receipt = (
        _partition_fixture(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(
        sweep,
        "run_seeded_leiden",
        lambda *_args, **_kwargs: SimpleNamespace(
            labels=baseline.copy(),
            receipt={
                "cluster_count": 2,
                "labels_sha256": _array_sha256(
                    "sorted_leiden_labels", baseline
                ),
            },
        ),
    )
    sweep.build_or_load_resolution_partitions(
        source_root=source_root,
        source_clustering=_source_clustering_with_labels(baseline),
        output_root=output_root,
        graph=graph,
        graph_receipt=graph_receipt,
        resolutions=(1.0,),
        random_seed=20260825,
    )
    label_path = output_root / "clustering" / "labels_resolution_1p0.npy"
    label_path.write_bytes(b"tampered")

    with pytest.raises(
        sweep.ContextualResolutionSweepError, match="checksum changed"
    ):
        sweep.build_or_load_resolution_partitions(
            source_root=source_root,
            source_clustering=_source_clustering_with_labels(baseline),
            output_root=output_root,
            graph=graph,
            graph_receipt=graph_receipt,
            resolutions=(1.0,),
            random_seed=20260825,
        )


def test_contextual_map_uses_exact_six_core_order_and_one_palette(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _small_cell_frame()
    label_column = "contextual_cluster_r0p5"
    frame[label_column] = ["R0p5_C0", "R0p5_C1"] * len(CORE_NUMBERS)
    palette = {"R0p5_C0": "#123456", "R0p5_C1": "#fedcba"}
    monkeypatch.setattr(
        sweep,
        "EXPECTED_CELL_COUNTS",
        {core_number: 2 for core_number in CORE_NUMBERS},
    )
    scatters: list[tuple[np.ndarray, list[str], dict[str, object]]] = []
    styled: list[np.ndarray] = []

    class Axis:
        def scatter(
            self,
            x: np.ndarray,
            _y: np.ndarray,
            **kwargs: object,
        ) -> object:
            scatters.append(
                (
                    np.asarray(x),
                    list(kwargs["c"]),
                    dict(kwargs),
                )
            )
            return object()

        def set_title(self, *_args: object, **_kwargs: object) -> None:
            return None

    class Figure:
        def suptitle(self, *_args: object, **_kwargs: object) -> None:
            return None

        def legend(self, *_args: object, **_kwargs: object) -> None:
            return None

        def subplots_adjust(self, *_args: object, **_kwargs: object) -> None:
            return None

    axes = np.asarray(
        [[Axis(), Axis(), Axis()], [Axis(), Axis(), Axis()]], dtype=object
    )
    monkeypatch.setattr(
        "matplotlib.pyplot.subplots", lambda *_args, **_kwargs: (Figure(), axes)
    )
    monkeypatch.setattr("matplotlib.pyplot.close", lambda *_args: None)
    monkeypatch.setattr(
        sweep,
        "_style_spatial_axis",
        lambda _axis, coordinates: styled.append(np.asarray(coordinates)),
    )
    monkeypatch.setattr(
        sweep, "_atomic_save_figure_pair", lambda *_args, **_kwargs: None
    )

    sweep._render_contextual_resolution_map(
        frame,
        resolution=0.5,
        palette=palette,
        png_path=tmp_path / "map.png",
        pdf_path=tmp_path / "map.pdf",
        dpi=72,
    )

    assert len(scatters) == len(styled) == len(CORE_NUMBERS)
    assert [int(values[0][0] // 10) for values in scatters] == list(CORE_NUMBERS)
    assert all(colors == ["#123456", "#fedcba"] for _, colors, _ in scatters)
    assert all(kwargs["linewidths"] == 0 for _, _, kwargs in scatters)
    assert all(kwargs["edgecolors"] == "none" for _, _, kwargs in scatters)
    assert all(kwargs["rasterized"] is True for _, _, kwargs in scatters)
    specification = sweep.contextual_resolution_plot_spec(
        resolution=0.5, palette=palette
    )
    assert sweep.requested_contextual_panel_order() == CORE_NUMBERS
    assert specification["panel_order"] == list(CORE_NUMBERS)
    assert specification["grid_shape"] == [2, 3]
    assert specification["equal_aspect"] is True
    assert specification["invert_y_axis"] is True
    assert specification["representation"] == "contextual_hL_only"


def test_contextual_plotting_receipt_checksums_figures_and_excludes_h0_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _small_cell_frame()
    frame["contextual_cluster_r1p0"] = ["R1p0_C0", "R1p0_C1"] * len(
        CORE_NUMBERS
    )
    output_root = tmp_path / "sweep"
    table_path = (
        output_root / "tables" / "cell_contextual_resolution_clusters.parquet"
    )
    table_path.parent.mkdir(parents=True)
    frame.to_parquet(table_path, index=False)
    monkeypatch.setattr(sweep, "EXPECTED_TOTAL_CELLS", len(frame))
    monkeypatch.setattr(
        sweep,
        "EXPECTED_CELL_COUNTS",
        {core_number: 2 for core_number in CORE_NUMBERS},
    )

    def fake_render(
        _frame: pd.DataFrame,
        *,
        resolution: float,
        palette: Mapping[str, str],
        png_path: Path,
        pdf_path: Path,
        dpi: int,
    ) -> None:
        assert resolution == 1.0
        assert palette == {"R1p0_C0": "#123456", "R1p0_C1": "#fedcba"}
        assert dpi == 72
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(b"png")
        pdf_path.write_bytes(b"pdf")

    monkeypatch.setattr(sweep, "_render_contextual_resolution_map", fake_render)
    partition_receipt = {
        "manifest_content_sha256": "partition-manifest",
        "palettes": {
            "1p0": {"R1p0_C0": "#123456", "R1p0_C1": "#fedcba"}
        },
        "partitions": {"1p0": {"cluster_count": 2}},
    }
    receipt = sweep.render_contextual_resolution_figures(
        output_root=output_root,
        partition_receipt=partition_receipt,
        resolutions=(1.0,),
        dpi=72,
    )

    assert receipt["figure_count"] == 2
    assert receipt["h0_figures_created"] is False
    assert receipt["delta_h_figures_created"] is False
    assert set(receipt["files"]) == {
        "figures/contextual_leiden_resolution_1p0_spatial_6cores.png",
        "figures/contextual_leiden_resolution_1p0_spatial_6cores.pdf",
    }
    figure = output_root / next(iter(receipt["files"]))
    figure.write_bytes(b"tampered")
    with pytest.raises(
        sweep.ContextualResolutionSweepError, match="checksum changed"
    ):
        sweep.render_contextual_resolution_figures(
            output_root=output_root,
            partition_receipt=partition_receipt,
            resolutions=(1.0,),
            dpi=72,
        )


def _write_synthetic_final_sweep_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    n_cells = 2 * len(CORE_NUMBERS)
    monkeypatch.setattr(sweep, "EXPECTED_TOTAL_CELLS", n_cells)
    source_root = tmp_path / "source" / sweep.EXPECTED_RUN_ID
    output_root = tmp_path / "sweep"
    (source_root / "clustering").mkdir(parents=True)
    (output_root / "clustering").mkdir(parents=True)
    (output_root / "tables").mkdir(parents=True)
    (output_root / "figures").mkdir(parents=True)
    source_manifest_path = source_root / "manifest.json"
    source_manifest_path.write_text("{}\n", encoding="utf-8")

    labels = np.asarray([0, 0, 1, 1] * 3, dtype=np.int64)
    np.save(source_root / "clustering" / "contextual_labels.npy", labels)
    edge_pairs = np.asarray(
        [[index, index + 1] for index in range(n_cells - 1)], dtype=np.int64
    )
    edge_sha = _array_sha256("knn_undirected_edges", edge_pairs)
    pca_sha = "2" * 64
    embedding_sha = "3" * 64
    source_clustering = {
        "configuration": {
            "pca_components": 50,
            "n_neighbors": 30,
            "distance_metric": "cosine",
            "random_seed": 20260825,
        },
        "pipelines": {
            "contextual": {
                "joint_embedding_sha256": embedding_sha,
                "pca": {"normalized_scores_sha256": pca_sha},
                "knn": {"undirected_edges_sha256": edge_sha},
            }
        },
    }
    sweep._atomic_write_json(
        source_root / "clustering" / "clustering_manifest.json",
        source_clustering,
    )
    source_manifest: dict[str, object] = {
        "run_id": sweep.EXPECTED_RUN_ID,
        "model_seed": sweep.EXPECTED_MODEL_SEED,
        "checkpoint_sha256": "a" * 64,
        "manifest_content_sha256": "b" * 64,
        "cell_counts": {str(core_number): 2 for core_number in CORE_NUMBERS},
    }
    monkeypatch.setattr(
        sweep,
        "verify_embedding_cluster_analysis_bundle",
        lambda _root: source_manifest,
    )

    graph_path = output_root / "clustering" / "contextual_analysis_graph.npz"
    sweep._write_deterministic_npz(graph_path, {"edge_pairs": edge_pairs})
    graph_configuration = sweep._graph_configuration(
        source_manifest,
        source_clustering,
        source_manifest_path=source_manifest_path,
    )
    graph_receipt = sweep._receipt_with_self_hash(
        {
            "schema": sweep.GRAPH_STAGE_SCHEMA,
            "status": "complete",
            "configuration": graph_configuration,
            "cell_count": n_cells,
            "core_order": list(CORE_NUMBERS),
            "h0_or_delta_used": False,
            "pca": {"normalized_scores_sha256": pca_sha},
            "knn": {
                "undirected_edges_sha256": edge_sha,
                "cell_by_cell_matrix_constructed": False,
            },
            "files": {
                "clustering/contextual_analysis_graph.npz": sweep._file_record(
                    graph_path
                )
            },
        }
    )
    graph_manifest_path = (
        output_root / "clustering" / "contextual_graph_manifest.json"
    )
    sweep._atomic_write_json(graph_manifest_path, graph_receipt)

    token = "1p0"
    resolution = 1.0
    label_sha = _array_sha256("sorted_leiden_labels", labels)
    leiden = {"cluster_count": 2, "labels_sha256": label_sha}
    label_path = output_root / "clustering" / f"labels_resolution_{token}.npy"
    parameter_path = (
        output_root / "clustering" / f"parameters_resolution_{token}.json"
    )
    palette_path = (
        output_root / "clustering" / f"palette_resolution_{token}.json"
    )
    sweep._atomic_write_npy(label_path, labels)
    sweep._atomic_write_json(
        parameter_path,
        {
            "representation": "contextual_hL_only",
            "leiden_resolution": resolution,
            "resolution_token": token,
            "shared_knn_edges_sha256": edge_sha,
            "leiden": leiden,
        },
    )
    palette = {"R1p0_C0": "#123456", "R1p0_C1": "#fedcba"}
    sweep._atomic_write_json(
        palette_path,
        {
            "representation": "contextual_hL_only",
            "leiden_resolution": resolution,
            "resolution_qualified_prefix": "R1p0_C",
            "cross_resolution_cluster_identity_implied": False,
            "colors": palette,
        },
    )
    frame = _small_cell_frame()
    frame["contextual_cluster_number_r1p0"] = labels.astype(np.int32)
    frame["contextual_cluster_r1p0"] = [f"R1p0_C{value}" for value in labels]
    table_path = (
        output_root / "tables" / "cell_contextual_resolution_clusters.parquet"
    )
    frame.to_parquet(table_path, index=False)
    summary_path = (
        output_root / "tables" / "contextual_resolution_cluster_summary.csv"
    )
    composition_path = (
        output_root / "tables" / "contextual_resolution_core_composition.csv"
    )
    summary_path.write_text("cluster,size\nR1p0_C0,6\nR1p0_C1,6\n", encoding="utf-8")
    composition_path.write_text("cluster,core_number,count\n", encoding="utf-8")
    partition_files = {
        path.relative_to(output_root).as_posix(): sweep._file_record(path)
        for path in (
            label_path,
            parameter_path,
            palette_path,
            table_path,
            summary_path,
            composition_path,
        )
    }
    partition_configuration = sweep._partition_configuration(
        graph_receipt=graph_receipt,
        resolutions=(resolution,),
        random_seed=20260825,
    )
    partition_receipt = sweep._receipt_with_self_hash(
        {
            "schema": sweep.PARTITION_STAGE_SCHEMA,
            "status": "complete",
            "configuration": partition_configuration,
            "core_order": list(CORE_NUMBERS),
            "total_cells": n_cells,
            "shared_graph_object_reused_for_all_resolutions": True,
            "independent_leiden_partition_per_resolution": True,
            "baseline_labels_exact_match": True,
            "source_contextual_labels_sha256": label_sha,
            "h0_or_delta_used": False,
            "partitions": {
                token: {
                    "leiden_resolution": resolution,
                    "label_prefix": "R1p0_C",
                    "cluster_count": 2,
                    "cluster_size_range": [6, 6],
                    "core_dominated_gt_90pct": [],
                    "labels_sha256": label_sha,
                    "leiden": leiden,
                }
            },
            "palettes": {token: palette},
            "cell_table_columns": list(frame.columns),
            "files": partition_files,
        }
    )
    partition_manifest_path = (
        output_root / "clustering" / "resolution_sweep_manifest.json"
    )
    sweep._atomic_write_json(partition_manifest_path, partition_receipt)

    png_path = (
        output_root
        / "figures"
        / "contextual_leiden_resolution_1p0_spatial_6cores.png"
    )
    pdf_path = png_path.with_suffix(".pdf")
    png_path.write_bytes(b"png")
    pdf_path.write_bytes(b"pdf")
    plotting_configuration = sweep._plotting_configuration(
        partition_receipt=partition_receipt,
        resolutions=(resolution,),
        dpi=72,
    )
    plotting_receipt = sweep._receipt_with_self_hash(
        {
            "schema": sweep.PLOTTING_STAGE_SCHEMA,
            "status": "complete",
            "configuration": plotting_configuration,
            "figure_count": 2,
            "combined_six_core_figure_pairs": 1,
            "one_dot_per_cell": True,
            "point_layer_rasterized_in_pdf": True,
            "h0_figures_created": False,
            "delta_h_figures_created": False,
            "plot_specifications": {},
            "files": {
                path.relative_to(output_root).as_posix(): sweep._file_record(path)
                for path in (png_path, pdf_path)
            },
        }
    )
    plotting_manifest_path = output_root / "figures" / "plotting_manifest.json"
    sweep._atomic_write_json(plotting_manifest_path, plotting_receipt)
    (output_root / "README.md").write_text("contextual hL only\n", encoding="utf-8")
    files = sweep._file_manifest(output_root)
    manifest = sweep._receipt_with_self_hash(
        {
            "schema": sweep.SWEEP_SCHEMA,
            "status": "complete",
            "run_id": sweep.EXPECTED_RUN_ID,
            "model_seed": sweep.EXPECTED_MODEL_SEED,
            "checkpoint_sha256": source_manifest["checkpoint_sha256"],
            "representation": "contextual_hL_only",
            "h0_or_delta_used": False,
            "source_analysis_bundle": source_root.as_posix(),
            "source_analysis_manifest_file_sha256": sweep.sha256_file(
                source_manifest_path
            ),
            "source_analysis_manifest_content_sha256": source_manifest[
                "manifest_content_sha256"
            ],
            "core_order": list(CORE_NUMBERS),
            "cell_counts": source_manifest["cell_counts"],
            "total_cells": n_cells,
            "resolutions": [resolution],
            "random_seed": 20260825,
            "cluster_counts": {token: 2},
            "cluster_size_ranges": {token: [6, 6]},
            "core_dominated_gt_90pct": {token: []},
            "figure_dpi": 72,
            "graph_manifest_sha256": sweep.sha256_file(graph_manifest_path),
            "partition_manifest_sha256": sweep.sha256_file(
                partition_manifest_path
            ),
            "plotting_manifest_sha256": sweep.sha256_file(
                plotting_manifest_path
            ),
            "interpretation": {
                "establishes_cell_type": False,
                "establishes_signaling": False,
                "establishes_biological_influence": False,
                "establishes_causality": False,
                "marker_and_pathology_validation_separate": True,
            },
            "files": files,
        }
    )
    sweep._atomic_write_json(output_root / "manifest.json", manifest)
    return output_root, label_path


def test_final_sweep_manifest_verifies_exact_inventory_and_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root, label_path = _write_synthetic_final_sweep_bundle(
        tmp_path, monkeypatch
    )

    verified = sweep.verify_contextual_resolution_sweep_bundle(output_root)
    assert verified["representation"] == "contextual_hL_only"
    assert verified["h0_or_delta_used"] is False
    assert set(verified["files"]) == sweep._required_sweep_files((1.0,))

    prohibited = output_root / "figures" / "delta_h_not_allowed.png"
    prohibited.write_bytes(b"not allowed")
    with pytest.raises(
        sweep.ContextualResolutionSweepError, match="file inventory changed"
    ):
        sweep.verify_contextual_resolution_sweep_bundle(output_root)
    prohibited.unlink()
    sweep.verify_contextual_resolution_sweep_bundle(output_root)

    label_path.write_bytes(b"tampered")
    with pytest.raises(
        sweep.ContextualResolutionSweepError,
        match="file inventory changed|checksum changed",
    ):
        sweep.verify_contextual_resolution_sweep_bundle(output_root)
