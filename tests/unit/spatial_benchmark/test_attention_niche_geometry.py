from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from shapely.geometry import MultiPolygon, Polygon, shape

import spatial_benchmark.attention_niche_geometry as geometry
from spatial_benchmark.attention_niche_geometry import (
    AttentionNicheGeometryError,
    construct_local_contiguity_graph,
    deterministic_niche_colors,
    dissolve_niche_regions,
    dissolved_regions_geojson,
    load_aligned_cosmx_polygons,
    niche_adjacency_from_cells,
    region_adjacency,
    split_preliminary_niches,
    validate_and_repair_regions_geojson,
    validate_polygon_centroid_alignment,
    validate_undirected_adjacency,
    verify_niche_connectedness,
)


def _adjacency(n_nodes: int, pairs: list[tuple[int, int]]) -> sparse.csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    for i, j in pairs:
        rows.extend((i, j))
        columns.extend((j, i))
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(n_nodes, n_nodes),
    )


def _edge_pairs(matrix: sparse.csr_matrix) -> set[tuple[int, int]]:
    rows, columns = sparse.triu(matrix, k=1).nonzero()
    return set(zip(rows.tolist(), columns.tolist(), strict=True))


def _square(x: float, y: float, width: float = 1.0) -> Polygon:
    return Polygon(
        [
            (x, y),
            (x + width, y),
            (x + width, y + width),
            (x, y + width),
        ]
    )


def test_local_contiguity_prefers_usable_segmentation_polygon_adjacency() -> None:
    coordinates = np.asarray([[0.5, 0.5], [1.5, 0.5], [9.5, 0.5]])
    polygons = (_square(0, 0), _square(1, 0), _square(9, 0))

    result = construct_local_contiguity_graph(
        coordinates,
        polygons_um=polygons,
        minimum_polygon_nonisolated_fraction=2 / 3,
        max_gap_um=75.0,
    )

    assert result.method == "segmentation_polygon_adjacency"
    assert result.edge_count == 1
    assert result.polygon_nonisolated_fraction == pytest.approx(2 / 3)
    assert _edge_pairs(result.adjacency) == {(0, 1)}
    assert np.array_equal(result.edge_pairs, np.asarray([[0, 1]]))


def test_local_contiguity_uses_deterministic_delaunay_with_locked_gap() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [200.0, 200.0]]
    )
    # Valid polygons are deliberately separated, making polygon adjacency
    # unusable under the declared complete-coverage audit.
    polygons = tuple(_square(x - 0.1, y - 0.1, 0.2) for x, y in coordinates)

    first = construct_local_contiguity_graph(
        coordinates,
        polygons_um=polygons,
        minimum_polygon_nonisolated_fraction=1.0,
        max_gap_um=75.0,
    )
    second = construct_local_contiguity_graph(
        coordinates,
        polygons_um=polygons,
        minimum_polygon_nonisolated_fraction=1.0,
        max_gap_um=75.0,
    )

    assert first.method == "delaunay_max_gap"
    assert first.fallback_reason is not None
    assert (first.adjacency != second.adjacency).nnz == 0
    for i, j in first.edge_pairs:
        assert np.linalg.norm(coordinates[i] - coordinates[j]) <= 75.0
    assert np.diff(first.adjacency.indptr)[4] == 0


def test_collinear_geometry_falls_back_to_tie_broken_knn_radius() -> None:
    coordinates = np.asarray([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])

    result = construct_local_contiguity_graph(
        coordinates,
        max_gap_um=15.0,
        fallback_k=1,
    )

    assert result.method == "knn_k1_radius"
    assert _edge_pairs(result.adjacency) == {(0, 1), (1, 2)}
    assert "collinear" in str(result.fallback_reason)


def test_undirected_validation_never_silently_symmetrizes() -> None:
    directed = sparse.csr_matrix(([1], ([0], [1])), shape=(2, 2))
    with pytest.raises(AttentionNicheGeometryError, match="undirected/symmetric"):
        validate_undirected_adjacency(directed)

    self_loop = sparse.eye(2, format="csr")
    with pytest.raises(AttentionNicheGeometryError, match="self-edges"):
        validate_undirected_adjacency(self_loop)


def test_preliminary_niches_are_split_into_deterministic_connected_core_ids() -> None:
    graph = _adjacency(7, [(0, 1), (1, 2), (3, 4), (5, 6)])
    preliminary = np.asarray([17, 17, 17, 17, 17, 2, 2])

    result = split_preliminary_niches(
        9, preliminary, graph, micro_niche_threshold=3
    )

    assert result.final_niche_ids.tolist() == [
        "C09-N001",
        "C09-N001",
        "C09-N001",
        "C09-N002",
        "C09-N002",
        "C09-N003",
        "C09-N003",
    ]
    assert result.micro_niche.tolist() == [False, False, False, True, True, True, True]
    assert result.preliminary_label_by_niche == {
        "C09-N001": 17,
        "C09-N002": 17,
        "C09-N003": 2,
    }
    assert verify_niche_connectedness(result.final_niche_ids, graph) == {
        "C09-N001": 1,
        "C09-N002": 1,
        "C09-N003": 1,
    }

    with pytest.raises(AttentionNicheGeometryError, match="disconnected"):
        verify_niche_connectedness(["C09-N001"] * 7, graph)


def _write_polygon_csv(path: Path, rows: list[tuple[int, int, float, float]]) -> None:
    pd.DataFrame(
        rows,
        columns=["fov", "cellID", "x_global_px", "y_global_px"],
    ).to_csv(path, index=False)


def test_raw_cosmx_polygons_stream_and_align_by_complete_slide_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slide_one = tmp_path / "26040302SO_1-polygons.csv"
    slide_two = tmp_path / "26040302SO_2-polygons.csv"
    _write_polygon_csv(
        slide_one,
        [
            (1, 5, 0.0, 0.0),
            (1, 5, 2.0, 0.0),
            (1, 5, 2.0, 2.0),
            (1, 5, 0.0, 2.0),
            (99, 7, 999.0, 999.0),
        ],
    )
    _write_polygon_csv(
        slide_two,
        [
            (1, 5, 10.0, 10.0),
            (1, 5, 12.0, 10.0),
            (1, 5, 12.0, 12.0),
            (1, 5, 10.0, 12.0),
            (99, 7, 999.0, 999.0),
        ],
    )
    prepared_keys = pd.DataFrame(
        {
            "slide": ["SO_2", "SO_1"],
            "fov": [1, 1],
            "cell_ID": [5, 5],
        }
    )
    prepared_coordinates = np.asarray([[11.0, 11.0], [1.0, 1.0]])
    real_read_csv = pd.read_csv
    observed_chunksizes: list[int | None] = []

    def recording_read_csv(*args: object, **kwargs: object) -> object:
        observed_chunksizes.append(kwargs.get("chunksize"))  # type: ignore[arg-type]
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(geometry.pd, "read_csv", recording_read_csv)
    alignment = load_aligned_cosmx_polygons(
        {"SO_1": slide_one, "SO_2": slide_two},
        prepared_keys,
        prepared_coordinates_um=prepared_coordinates,
        pixel_size_um=1.0,
        chunksize=2,
        centroid_tolerance_um=1e-9,
    )

    assert alignment.keys == (("SO_2", 1, 5), ("SO_1", 1, 5))
    assert [polygon.centroid.x for polygon in alignment.polygons_um] == [11.0, 1.0]
    assert alignment.vertex_counts.tolist() == [4, 4]
    assert alignment.source_rows_scanned == 10
    assert alignment.selected_vertex_rows == 8
    assert alignment.centroid_alignment is not None
    assert alignment.centroid_alignment.aligned
    assert observed_chunksizes == [2, 2]


def test_raw_polygon_loader_rejects_missing_coverage_and_slide_mismatch(
    tmp_path: Path,
) -> None:
    slide_one = tmp_path / "26040302SO_1-polygons.csv"
    _write_polygon_csv(
        slide_one,
        [(1, 5, 0, 0), (1, 5, 1, 0), (1, 5, 1, 1), (1, 5, 0, 1)],
    )
    keys = pd.DataFrame(
        {"slide": ["SO_1", "SO_1"], "fov": [1, 1], "cell_ID": [5, 6]}
    )
    with pytest.raises(AttentionNicheGeometryError, match="coverage is incomplete"):
        load_aligned_cosmx_polygons(
            {"SO_1": slide_one}, keys, pixel_size_um=1.0, chunksize=2
        )
    with pytest.raises(AttentionNicheGeometryError, match="disagrees"):
        load_aligned_cosmx_polygons(
            {"SO_2": slide_one},
            keys.iloc[:1],
            pixel_size_um=1.0,
            chunksize=2,
        )


def test_polygon_centroid_alignment_reports_um_distance_and_can_fail_closed() -> None:
    polygons = (_square(0, 0, 2), _square(10, 10, 2))
    coordinates = np.asarray([[1.0, 1.0], [13.0, 11.0]])

    report = validate_polygon_centroid_alignment(
        polygons, coordinates, tolerance_um=1.5, raise_on_mismatch=False
    )
    assert report.distances_um.tolist() == [0.0, 2.0]
    assert report.mismatch_indices == (1,)
    with pytest.raises(AttentionNicheGeometryError, match="micrometres"):
        validate_polygon_centroid_alignment(
            polygons, coordinates, tolerance_um=1.5, raise_on_mismatch=True
        )


def test_polygon_alignment_accepts_contained_coordinate_far_from_centroid() -> None:
    elongated = Polygon([(0, 0), (20, 0), (20, 100), (0, 100)])
    coordinates = np.asarray([[10.0, 2.0]])

    report = validate_polygon_centroid_alignment(
        (elongated,), coordinates, tolerance_um=5.0, raise_on_mismatch=True
    )

    assert report.aligned
    assert report.distances_um.tolist() == [48.0]
    assert report.centroid_tolerance_exceeded_indices == (0,)
    assert report.containment_accepted_indices == (0,)
    assert report.coordinate_outside_polygon_indices == ()
    assert report.mismatch_indices == ()


def test_dissolve_preserves_holes_and_multipart_geometry_and_geojson_is_stable() -> None:
    donut = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (2, 1), (2, 2), (1, 2)]],
    )
    separate = _square(10, 0)
    touching_other_niche = _square(11, 0)
    regions = dissolve_niche_regions(
        [donut, separate, touching_other_niche],
        ["C01-N001", "C01-N001", "C01-N002"],
    )

    assert [region.niche_id for region in regions] == ["C01-N001", "C01-N002"]
    first = regions[0]
    assert isinstance(first.geometry, MultiPolygon)
    assert sum(len(part.interiors) for part in first.geometry.geoms) == 1
    adjacency = region_adjacency(regions)
    assert adjacency == {
        "C01-N001": ("C01-N002",),
        "C01-N002": ("C01-N001",),
    }
    colors = deterministic_niche_colors(1, adjacency)
    first_geojson = dissolved_regions_geojson(
        regions, core_number=1, colors=colors, coordinate_precision=4
    )
    second_geojson = dissolved_regions_geojson(
        tuple(reversed(regions)), core_number=1, colors=colors, coordinate_precision=4
    )
    assert json.dumps(first_geojson, sort_keys=True) == json.dumps(
        second_geojson, sort_keys=True
    )
    assert [feature["id"] for feature in first_geojson["features"]] == [
        "C01-N001",
        "C01-N002",
    ]
    assert first_geojson["features"][0]["properties"] == {
        "core_number": 1,
        "final_niche_id": "C01-N001",
        "cell_count": 2,
        "area_um2": pytest.approx(first.area_um2),
        "coordinate_unit": "um",
        "niche_color": colors["C01-N001"],
    }


def test_serialized_region_validation_repairs_polygonal_self_intersection() -> None:
    source = {
        "type": "FeatureCollection",
        "coordinate_unit": "um",
        "features": [
            {
                "type": "Feature",
                "id": "C01-N001",
                "properties": {
                    "core_number": 1,
                    "final_niche_id": "C01-N001",
                    "cell_count": 2,
                    "area_um2": 0.5,
                    "coordinate_unit": "um",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
                    ],
                },
            }
        ],
    }

    repaired, audit = validate_and_repair_regions_geojson(source)

    assert audit["invalid_before_repair"] == 1
    assert audit["invalid_after_repair"] == 0
    assert audit["feature_count"] == 1
    assert shape(repaired["features"][0]["geometry"]).is_valid
    assert repaired["features"][0]["properties"]["area_um2"] == 0.5


def test_cell_niche_adjacency_and_colors_are_stable_distinct_and_core_scoped() -> None:
    cell_graph = _adjacency(4, [(0, 1), (1, 2), (2, 3)])
    labels = ["C01-N001", "C01-N001", "C01-N002", "C01-N003"]
    adjacency = niche_adjacency_from_cells(labels, cell_graph)
    assert adjacency == {
        "C01-N001": ("C01-N002",),
        "C01-N002": ("C01-N001", "C01-N003"),
        "C01-N003": ("C01-N002",),
    }

    colors = deterministic_niche_colors(1, adjacency, color_seed=41)
    reordered = deterministic_niche_colors(
        1,
        {
            "C01-N003": ["C01-N002"],
            "C01-N002": ["C01-N003", "C01-N001"],
            "C01-N001": ["C01-N002"],
        },
        color_seed=41,
    )
    assert colors == reordered
    assert colors["C01-N001"] != colors["C01-N002"]
    assert colors["C01-N002"] != colors["C01-N003"]

    core_nine = deterministic_niche_colors(
        9,
        {
            "C09-N001": ["C09-N002"],
            "C09-N002": ["C09-N001", "C09-N003"],
            "C09-N003": ["C09-N002"],
        },
        color_seed=41,
    )
    assert set(colors.values()).isdisjoint(core_nine.values())


def test_color_scope_and_adjacency_contracts_fail_closed() -> None:
    with pytest.raises(AttentionNicheGeometryError, match="scoped"):
        deterministic_niche_colors(1, {"C09-N001": []})
    with pytest.raises(AttentionNicheGeometryError, match="symmetric"):
        deterministic_niche_colors(
            1, {"C01-N001": ["C01-N002"], "C01-N002": []}
        )
