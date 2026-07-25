from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark import (  # noqa: E402
    ALLOWED_METADATA_COLUMNS,
    CoreSelection,
    DataContractError,
    TrainOnlyPreprocessor,
    biological_probe_columns,
    build_spatial_graph,
    load_selected_core,
    make_spatial_split,
    rewire_spatial_graph,
    select_unique_legacy_true_normal_core,
)


def _write_nested_csv(raw_dir: Path, basename: str, frame: pd.DataFrame) -> Path:
    directory = raw_dir / basename
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / basename
    frame.to_csv(path, index=False)
    return path


def _synthetic_metadata(rows: list[tuple[int, int, float, float, object]]) -> pd.DataFrame:
    data: dict[str, object] = {
        "fov": [row[0] for row in rows],
        "cell_ID": [row[1] for row in rows],
        "slide_ID": [2] * len(rows),
        "CenterX_global_px": [row[2] for row in rows],
        "CenterY_global_px": [row[3] for row in rows],
        "qcCellsPassed": [row[4] for row in rows],
    }
    for index, column in enumerate(ALLOWED_METADATA_COLUMNS):
        data[column] = np.arange(1, len(rows) + 1, dtype=float) + index
    return pd.DataFrame(data)


def test_safe_legacy_selector_does_not_return_restricted_ids(tmp_path: Path) -> None:
    legacy_path = tmp_path / "legacy.xlsx"
    review_path = tmp_path / "review.xlsx"
    map_path = tmp_path / "map.csv"
    restricted_donor = "restricted-donor-value"
    private_core = 17
    pd.DataFrame(
        [
            ["heading", None, None],
            [private_core, restricted_donor, "정상"],
            [18, "another-restricted-value", "정상인접"],
        ]
    ).to_excel(legacy_path, header=False, index=False)
    pd.DataFrame(
        {
            "슬라이드번호": [private_core, 18],
            "진단명": ["diagnosis", "diagnosis"],
            "결과": ["reviewed", "reviewed"],
            "비고 (수정진단)": [np.nan, np.nan],
        }
    ).to_excel(review_path, index=False)
    pd.DataFrame(
        {
            "slide": ["SO_2", "SO_2", "SO_2"],
            "core_label": [private_core, private_core, 18],
            "fov": [7, 8, 9],
        }
    ).to_csv(map_path, index=False)

    selection = select_unique_legacy_true_normal_core(
        legacy_path,
        map_path,
        pathology_review_workbook=review_path,
    )

    assert selection == CoreSelection(slide="SO_2", fovs=(7, 8))
    public_fields = asdict(selection)
    assert not any("donor" in name.lower() or "core" in name.lower() for name in public_fields)
    assert restricted_donor not in repr(selection)
    assert str(private_core) not in repr(selection)


def test_selector_stops_on_ambiguous_or_corrected_label_without_ids(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy.xlsx"
    map_path = tmp_path / "map.csv"
    pd.DataFrame([[11, "private-a", "정상"], [12, "private-b", "정상"]]).to_excel(
        legacy_path, header=False, index=False
    )
    pd.DataFrame(
        {"slide": ["SO_1", "SO_2"], "core_label": [11, 12], "fov": [1, 1]}
    ).to_csv(map_path, index=False)
    with pytest.raises(DataContractError) as error:
        select_unique_legacy_true_normal_core(legacy_path, map_path)
    assert "private-a" not in str(error.value)
    assert "11" not in str(error.value)


def test_streamed_selected_core_load_enforces_probe_and_metadata_policy(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    expression_name = "26040302SO_2_exprMat_file.csv"
    metadata_name = "26040302SO_2_metadata_file.csv"
    expression = pd.DataFrame(
        {
            "fov": [7, 7, 8, 99],
            "cell_ID": [1, 2, 1, 1],
            "GeneA": [1, 2, 3, 999],
            "Negative1": [9, 9, 9, 9],
            "FCGR3A/B": [4, 5, 6, 999],
            "SystemControl1": [8, 8, 8, 8],
        }
    )
    metadata = _synthetic_metadata(
        [
            (8, 1, 30.0, 40.0, "passed"),
            (7, 2, 20.0, 30.0, "false"),
            (99, 1, 999.0, 999.0, "passed"),
            (7, 1, 10.0, 20.0, "true"),
        ]
    )
    expression_path = _write_nested_csv(raw_dir, expression_name, expression)
    _write_nested_csv(raw_dir, metadata_name, metadata)

    assert biological_probe_columns(expression_path, expected_count=2) == (
        "GeneA",
        "FCGR3A/B",
    )
    dataset = load_selected_core(
        raw_dir,
        CoreSelection("SO_2", (7, 8)),
        chunksize=1,
        expected_biological_probes=2,
        pixel_size_um=0.25,
    )

    assert dataset.n_cells == 3
    assert dataset.gene_names == ("GeneA", "FCGR3A/B")
    assert dataset.metadata_names == ALLOWED_METADATA_COLUMNS
    assert dataset.metadata.shape == (3, 22)
    assert dataset.keys.to_records(index=False).tolist() == [
        ("SO_2", 7, 1),
        ("SO_2", 7, 2),
        ("SO_2", 8, 1),
    ]
    np.testing.assert_array_equal(dataset.expression, [[1, 4], [2, 5], [3, 6]])
    np.testing.assert_allclose(
        dataset.coordinates_um,
        np.asarray([[2.5, 5.0], [5.0, 7.5], [7.5, 10.0]]),
    )
    np.testing.assert_array_equal(dataset.qc_passed, [True, False, True])
    assert all(name not in dataset.metadata_names for name in ("fov", "slide_ID"))
    assert "qcCellsPassed" not in dataset.metadata_names

    passed_only = load_selected_core(
        raw_dir,
        CoreSelection("SO_2", (7, 8)),
        chunksize=2,
        expected_biological_probes=2,
        qc_policy="passed",
    )
    assert passed_only.n_cells == 2
    assert passed_only.qc_passed.all()


def test_loader_rejects_wrong_slide_metadata_before_join(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    expression = pd.DataFrame({"fov": [7], "cell_ID": [1], "GeneA": [1]})
    metadata = _synthetic_metadata([(7, 1, 10.0, 20.0, True)])
    metadata["slide_ID"] = "26040302SO_1"
    _write_nested_csv(raw_dir, "26040302SO_2_exprMat_file.csv", expression)
    _write_nested_csv(raw_dir, "26040302SO_2_metadata_file.csv", metadata)
    with pytest.raises(DataContractError, match="slide_ID"):
        load_selected_core(
            raw_dir,
            CoreSelection("SO_2", (7,)),
            expected_biological_probes=1,
        )


def test_fov_aware_spatial_split_is_deterministic_and_group_disjoint() -> None:
    fov_centers = np.asarray(
        [
            [0.0, 0.0],
            [100.0, 0.0],
            [200.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0],
            [200.0, 100.0],
            [0.0, 200.0],
            [100.0, 200.0],
            [200.0, 200.0],
        ]
    )
    coordinates = np.repeat(fov_centers, 3, axis=0)
    coordinates += np.tile(np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]), (9, 1))
    fov = np.repeat(np.arange(1, 10), 3)

    first = make_spatial_split(
        coordinates,
        fov=fov,
        block_size_um=75.0,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=5,
    )
    second = make_spatial_split(
        coordinates,
        fov=fov,
        block_size_um=75.0,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=5,
    )

    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.macroblock_ids, second.macroblock_ids)
    assert first.split_id == second.split_id
    first.assert_fovs_disjoint(fov)
    for block in np.unique(first.macroblock_ids):
        assert np.unique(first.labels[first.macroblock_ids == block]).size == 1
    assert set(first.labels) == {"train", "val", "test"}


def test_preprocessing_statistics_are_train_only_and_metadata_is_unmasked() -> None:
    expression = np.asarray(
        [[0, 1], [3, 2], [8, 3], [1000, 1000], [5000, 5000]], dtype=float
    )
    metadata = np.tile(np.arange(1, 23, dtype=float), (5, 1))
    metadata[:3] += np.arange(3)[:, None]
    metadata[1, 0] = np.nan
    metadata[3:] = 1e9
    labels = np.asarray(["train", "train", "train", "val", "test"])

    fitted = TrainOnlyPreprocessor().fit(expression, metadata, labels)
    altered_expression = expression.copy()
    altered_metadata = metadata.copy()
    altered_expression[3:] *= 100
    altered_metadata[3:] *= 100
    refitted = TrainOnlyPreprocessor().fit(
        altered_expression, altered_metadata, labels
    )

    np.testing.assert_allclose(fitted.expression_mean_, refitted.expression_mean_)
    np.testing.assert_allclose(fitted.expression_scale_, refitted.expression_scale_)
    np.testing.assert_allclose(fitted.metadata_mean_, refitted.metadata_mean_)
    np.testing.assert_allclose(fitted.metadata_scale_, refitted.metadata_scale_)
    transformed = fitted.transform(expression, metadata)
    np.testing.assert_allclose(
        transformed.expression[:3].mean(axis=0), np.zeros(2), atol=1e-6
    )
    assert transformed.metadata.shape[1] == 23
    assert transformed.metadata_names[-1] == "Area__missing"
    np.testing.assert_array_equal(transformed.metadata[:, -1], [0, 1, 0, 0, 0])


def _edge_set(edge_index: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def test_union_mutual_graphs_are_directed_deterministic_and_split_safe() -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.01, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )
    split = np.asarray(["train"] * 3 + ["val"] * 3)
    fov = np.asarray([1, 1, 2, 2, 3, 3])
    union = build_spatial_graph(
        coordinates,
        k=1,
        radius_um=2.0,
        symmetry="union",
        group_labels=split,
        fov=fov,
        rbf_bins=4,
    )
    repeated = build_spatial_graph(
        coordinates,
        k=1,
        radius_um=2.0,
        symmetry="union",
        group_labels=split,
        fov=fov,
        rbf_bins=4,
    )
    mutual = build_spatial_graph(
        coordinates,
        k=1,
        radius_um=2.0,
        symmetry="mutual",
        group_labels=split,
        fov=fov,
        rbf_bins=4,
    )

    np.testing.assert_array_equal(union.edge_index, repeated.edge_index)
    np.testing.assert_allclose(union.edge_attr, repeated.edge_attr)
    assert union.qc.to_dict() == repeated.qc.to_dict()
    assert union.edge_index.shape[0] == 2
    assert union.edge_attr.shape[1] == 9 + 4
    assert union.qc.cross_group_edges == 0
    assert not any(split[source] != split[target] for source, target in _edge_set(union.edge_index))
    assert _edge_set(mutual.edge_index).issubset(_edge_set(union.edge_index))
    for source, target in _edge_set(union.edge_index):
        assert (target, source) in _edge_set(union.edge_index)
    # The reverse directed edge retains distance/RBF values and flips orientation.
    source, target = next(iter(_edge_set(union.edge_index)))
    forward_idx = np.flatnonzero(
        (union.edge_index[0] == source) & (union.edge_index[1] == target)
    )[0]
    reverse_idx = np.flatnonzero(
        (union.edge_index[0] == target) & (union.edge_index[1] == source)
    )[0]
    np.testing.assert_allclose(
        union.edge_attr[forward_idx, [0, 1, 2]],
        union.edge_attr[reverse_idx, [0, 1, 2]],
    )
    np.testing.assert_allclose(
        union.edge_attr[forward_idx, [3, 4, 5, 6]],
        -union.edge_attr[reverse_idx, [3, 4, 5, 6]],
        atol=1e-7,
    )


def test_degree_distance_rewiring_is_deterministic_and_preserves_degree() -> None:
    rng = np.random.default_rng(44)
    coordinates = rng.uniform(0.0, 100.0, size=(120, 2))
    groups = np.where(coordinates[:, 0] < 50.0, "left", "right")
    graph = build_spatial_graph(
        coordinates,
        k=8,
        radius_um=25.0,
        symmetry="union",
        group_labels=groups,
        rbf_bins=6,
    )
    with pytest.raises(ValueError, match="same group labels"):
        rewire_spatial_graph(graph, coordinates, seed=19, swaps_per_edge=0.1)
    rewired = rewire_spatial_graph(
        graph,
        coordinates,
        seed=19,
        swaps_per_edge=1.0,
        distance_bins=5,
        group_labels=groups,
    )
    repeated = rewire_spatial_graph(
        graph,
        coordinates,
        seed=19,
        swaps_per_edge=1.0,
        distance_bins=5,
        group_labels=groups,
    )

    old_degree = np.bincount(graph.edge_index[0], minlength=graph.n_nodes)
    new_degree = np.bincount(rewired.edge_index[0], minlength=graph.n_nodes)
    np.testing.assert_array_equal(old_degree, new_degree)
    np.testing.assert_array_equal(rewired.edge_index, repeated.edge_index)
    np.testing.assert_allclose(rewired.edge_attr, repeated.edge_attr)
    assert rewired.metadata["rewire_successful_swaps"] > 0
    assert rewired.metadata["degree_preserved_exactly"] is True
    assert rewired.metadata["relative_edge_distance_mean_change"] < 0.2
    assert rewired.qc.cross_group_edges == 0
    assert _edge_set(rewired.edge_index) != _edge_set(graph.edge_index)
