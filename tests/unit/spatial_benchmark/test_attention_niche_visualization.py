from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MatplotlibPath
import numpy as np
import pandas as pd

from spatial_benchmark import attention_niche_visualization as visualization


_COLORS = {
    1: "#E41A1C",
    9: "#377EB8",
    13: "#4DAF4A",
    15: "#984EA3",
    21: "#FF7F00",
    23: "#A65628",
}


def _closed_box(x_min: float, y_min: float, x_max: float, y_max: float) -> list[list[float]]:
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min],
    ]


def _synthetic_inputs() -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    assignments: list[dict[str, object]] = []
    features: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    points = ((0.0, 0.0), (20.0, 0.0), (0.0, 20.0), (20.0, 20.0))
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for core_position, core_number in enumerate(visualization.CORE_ORDER):
        niche_id = f"C{core_number:02d}-N001"
        for local_index, (x_um, y_um) in enumerate(points):
            assignments.append(
                {
                    "core_number": core_number,
                    "core_alias": f"CAN-{core_number:02d}",
                    "cell_index": core_number * 100 + local_index,
                    "x_um": x_um,
                    "y_um": y_um,
                    "final_niche_id": niche_id,
                    "niche_color": _COLORS[core_number],
                    "mutual_routing_hub_score": float(local_index + 1),
                    "assignment_confidence": 0.40 if local_index == 0 else 0.95,
                    "map_label": "4-model ensemble-consensus map",
                }
            )
        if core_number == 1:
            geometry: dict[str, object] = {
                "type": "Polygon",
                "coordinates": [
                    _closed_box(-2.0, -2.0, 22.0, 22.0),
                    _closed_box(8.0, 8.0, 12.0, 12.0),
                ],
            }
        elif core_number == 9:
            geometry = {
                "type": "MultiPolygon",
                "coordinates": [
                    [_closed_box(-2.0, -2.0, 10.0, 22.0)],
                    [_closed_box(10.0, -2.0, 22.0, 22.0)],
                ],
            }
        else:
            geometry = {
                "type": "Polygon",
                "coordinates": [_closed_box(-2.0, -2.0, 22.0, 22.0)],
            }
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "core_number": core_number,
                    "final_niche_id": niche_id,
                    "niche_color": _COLORS[core_number],
                    "cell_count": 4,
                    "area_um2": 576.0,
                    "coordinate_unit": "um",
                },
                "geometry": geometry,
            }
        )
        for pair_position, (left, right) in enumerate(pairs):
            edges.append(
                {
                    "core_number": core_number,
                    "cell_i_index": core_number * 100 + left,
                    "cell_j_index": core_number * 100 + right,
                    "M_ij": (
                        999.0
                        if pair_position == len(pairs) - 1
                        else 10.0 - pair_position + core_position / 100.0
                    ),
                    "support_P_ij": 0.95 - pair_position / 100.0,
                    "retained_primary": pair_position != len(pairs) - 1,
                }
            )
    return (
        pd.DataFrame(assignments),
        {"type": "FeatureCollection", "features": features},
        pd.DataFrame(edges),
    )


def test_polygon_and_multipolygon_paths_preserve_holes() -> None:
    polygon = {
        "niche_id": "C01-N001",
        "geometry_type": "Polygon",
        "coordinates": [
            _closed_box(0.0, 0.0, 10.0, 10.0),
            # Intentionally use the same input orientation as the exterior;
            # the renderer corrects it for a true non-zero-fill hole.
            _closed_box(2.0, 2.0, 4.0, 4.0),
        ],
    }
    polygon_paths = visualization._geometry_paths(polygon)
    assert len(polygon_paths) == 1
    assert np.count_nonzero(polygon_paths[0].codes == MatplotlibPath.MOVETO) == 2
    assert np.count_nonzero(polygon_paths[0].codes == MatplotlibPath.CLOSEPOLY) == 2

    multipolygon = {
        "niche_id": "C09-N001",
        "geometry_type": "MultiPolygon",
        "coordinates": [
            [_closed_box(0.0, 0.0, 5.0, 5.0)],
            [_closed_box(5.0, 0.0, 10.0, 5.0)],
        ],
    }
    assert len(visualization._geometry_paths(multipolygon)) == 2


def test_combined_figure_locks_panel_order_labels_aspect_and_scale_bars() -> None:
    assignments, regions, _edges = _synthetic_inputs()
    figure = visualization.create_combined_attention_niche_figure(
        assignments, regions
    )
    try:
        assert len(figure.axes) == 6
        assert "4-model ensemble-consensus map" in figure._suptitle.get_text()
        assert any("Gray cell outlines" in text.get_text() for text in figure.texts)
        for axis, core_number in zip(
            figure.axes, visualization.CORE_ORDER, strict=True
        ):
            assert axis.get_title().startswith(f"Core {core_number}  |")
            assert f"{4:,} cells" in axis.get_title()
            assert "1 niches" in axis.get_title()
            assert axis.get_aspect() == 1.0
            assert axis.get_ylim()[0] > axis.get_ylim()[1]
            assert len(axis.get_xticks()) == 0
            assert len(axis.get_yticks()) == 0
            assert any(
                text.get_text() == f"Core {core_number}"
                and text.get_gid() == f"in-panel-core-label-{core_number}"
                for text in axis.texts
            )
            assert any(
                line.get_gid() == f"scale-bar-core-{core_number}"
                for line in axis.lines
            )
            assert any(
                text.get_gid() == f"scale-bar-label-core-{core_number}"
                and text.get_text().endswith(" µm")
                for text in axis.texts
            )
            cell_collections = [
                collection
                for collection in axis.collections
                if collection.get_gid() == "all-eligible-cells"
            ]
            assert len(cell_collections) == 1
            assert len(cell_collections[0].get_offsets()) == 4

        first_axis = figure.axes[0]
        region_patch = next(
            patch
            for patch in first_axis.patches
            if patch.get_gid() == "region-C01-N001"
        )
        cell_collection = next(
            collection
            for collection in first_axis.collections
            if collection.get_gid() == "all-eligible-cells"
        )
        np.testing.assert_allclose(
            region_patch.get_facecolor()[:3],
            cell_collection.get_facecolors()[0, :3],
        )
    finally:
        plt.close(figure)


def test_strongest_overlay_edges_obey_both_caps_and_retained_flag() -> None:
    _assignments, _regions, edges = _synthetic_inputs()
    selected = visualization.select_strongest_mutual_edges(
        edges,
        max_edges_per_core=2,
        max_edges_total=7,
    )
    assert len(selected) == 7
    assert selected.groupby("core_number").size().max() <= 2
    # The unretained edge has an intentionally dominant score and must still be
    # absent from the display subset.
    assert selected["Mij"].max() < 999.0

    repeated = visualization.select_strongest_mutual_edges(
        edges.sample(frac=1.0, random_state=8),
        max_edges_per_core=2,
        max_edges_total=7,
    )
    pd.testing.assert_frame_equal(selected, repeated)


def test_render_all_outputs_atomically_with_qc_receipt(tmp_path: Path) -> None:
    assignments, regions, edges = _synthetic_inputs()
    artifacts = visualization.render_attention_niche_visualizations(
        assignments,
        regions,
        edges,
        tmp_path,
        dpi=35,
        individual_dpi=35,
        max_edges_per_core=2,
        max_edges_total=7,
    )

    expected_names = {
        "six_core_attention_niche_map.png",
        "six_core_attention_niche_map.pdf",
        "six_core_attention_niche_map.svg",
        "six_core_mutual_attention_network_overlay.png",
        "six_core_mutual_attention_network_overlay.pdf",
        *(f"core_{core:02d}_attention_niche_map.png" for core in visualization.CORE_ORDER),
    }
    assert {path.name for path in artifacts.all_paths} == expected_names
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.all_paths)
    assert not list(tmp_path.glob(".*.tmp-*"))

    receipt = artifacts.receipt
    assert receipt["status"] == "complete"
    assert receipt["map_label"] == "4-model ensemble-consensus map"
    assert receipt["core_order"] == [1, 9, 13, 15, 21, 23]
    assert receipt["grid_shape"] == [2, 3]
    assert receipt["equal_physical_aspect"] is True
    assert receipt["invert_imaging_y_axis"] is True
    assert receipt["coordinate_unit"] == "micrometres"
    assert receipt["total_cell_count"] == 24
    assert receipt["all_eligible_cells_rendered_without_sampling"] is True
    assert receipt["in_panel_core_labels"] == [
        "Core 1",
        "Core 9",
        "Core 13",
        "Core 15",
        "Core 21",
        "Core 23",
    ]
    assert all(
        value["present"] and value["unit"] == "µm"
        for value in receipt["scale_bars"].values()
    )
    assert receipt["overlay"]["displayed_edge_count"] == 7
    assert set(receipt["output_paths"]) == {
        str(path) for path in artifacts.all_paths
    }
