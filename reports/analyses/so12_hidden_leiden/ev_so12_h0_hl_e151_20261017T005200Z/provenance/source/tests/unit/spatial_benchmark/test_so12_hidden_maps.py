import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon


_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/analysis/so12_hidden_maps.py"
_SPEC = importlib.util.spec_from_file_location("so12_hidden_maps", _SCRIPT)
maps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(maps)


def _cells():
    # Numeric FOV and cell IDs deliberately collide across slides.
    return pd.DataFrame({"row_index": [0, 1, 2], "slide": [1, 2, 1], "fov": [1, 1, 2],
        "cell_ID": [7, 7, 9], "x_um": [10., 20., 30.], "y_um": [1., 2., 3.]})


def test_key_join_keeps_slide_identity_and_target_order():
    source = _cells()
    target = source.iloc[[1, 2, 0]].copy()
    target["row_index"] = [0, 1, 2]
    labels = pd.DataFrame({"row_index": [2, 0, 1], maps.PRIMARY_COLUMN: [1, 0, 2]})
    actual = maps.align_source_labels(target, source, labels, expected_cells=3)
    np.testing.assert_array_equal(actual, [2, 1, 0])


def test_key_join_rejects_missing_duplicate_and_geometry_mismatch():
    source = _cells()
    labels = pd.DataFrame({"row_index": [0, 1, 2], maps.PRIMARY_COLUMN: [0, 1, 2]})
    wrong = source.copy()
    wrong.loc[0, "cell_ID"] = 8
    with pytest.raises(ValueError, match="key coverage"):
        maps.align_source_labels(wrong, source, labels, expected_cells=3)
    duplicate = source.copy()
    duplicate.loc[1, ["slide", "fov", "cell_ID"]] = [1, 1, 7]
    with pytest.raises(ValueError, match="duplicate slide-qualified"):
        maps.align_source_labels(duplicate, source, labels, expected_cells=3)
    wrong = source.copy()
    wrong.loc[0, "x_um"] += .01
    with pytest.raises(ValueError, match="coordinates differ"):
        maps.align_source_labels(wrong, source, labels, expected_cells=3)


def test_palette_preserves_prior_colors_and_records_inconsistent_fallback():
    labels = np.asarray([0, 2, 1, 2], dtype=np.int32)
    expected, _ = maps.color_keys(labels, "pca")
    prior = pd.DataFrame({"method": ["pca"]*3, "cluster_id": list(expected), "color_hex": list(expected.values())})
    colors, provenance = maps.color_keys(labels, "pca", prior)
    assert colors == expected
    assert provenance == "existing_color_keys_verified_against_original_palette"
    incomplete = prior.iloc[:-1]
    colors, provenance = maps.color_keys(labels, "pca", incomplete)
    assert colors == expected
    assert "inconsistent" in provenance
    with pytest.raises(ValueError, match="contiguous"):
        maps.color_keys(np.array([0, 2]), "h0")


def test_polygon_converter_preserves_exterior_and_rejects_hole_loss():
    exterior = [(0, 0), (3, 0), (3, 3), (0, 3)]
    polygon = Polygon(exterior)
    np.testing.assert_array_equal(maps.polygon_vertices([polygon])[0], polygon.exterior.coords)
    holed = Polygon(exterior, [[(.5, .5), (1, .5), (1, 1), (.5, 1)]])
    with pytest.raises(ValueError, match="cannot discard holes"):
        maps.polygon_vertices([holed])


def test_polygon_render_smoke_preserves_bounds_and_exports_png_pdf(tmp_path):
    vertices = maps.polygon_vertices([Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])])
    figure, axis = maps.plt.subplots(figsize=(2, 2))
    bounds = (-10., 120., -10., 120.)
    maps._draw(axis, vertices, np.array([0]), {0: "#336699"}, bounds)
    assert axis.get_xlim() == (-10., 120.)
    assert axis.get_ylim() == (120., -10.)
    maps._save(figure, tmp_path, "pilot", 72)
    assert (tmp_path / "pilot.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "pilot.pdf").read_bytes().startswith(b"%PDF")


def test_partition_comparison_is_label_permutation_invariant_and_retains_subset_labels():
    labels = {"h0": np.array([0, 0, 1, 1]), "hL": np.array([1, 1, 0, 0]),
              "pca": np.array([0, 1, 0, 1]), "kernel_pca": np.array([0, 0, 1, 1]),
              "spatial_pca": np.array([0, 1, 0, 1])}
    comparison = maps.compare_partitions(labels, np.array([1, 1, 9, 9]), cores=(1, 9))
    all_cells = comparison["populations"]["all_cells"]
    assert all_cells["pairwise_ari"][0][1] == 1.0
    assert all_cells["pairwise_ari"][0][2] == -.5
    assert comparison["populations"]["core_01"]["observed_cluster_counts"]["hL"] == 1
    assert comparison["populations"]["core_01"]["observed_cluster_counts"]["pca"] == 2
    assert comparison["populations"]["requested_cores"]["cells"] == 4
    text = maps._comparison_html(comparison)
    assert "Core 1" in text and "All mapped cells" in text and "-0.500" in text
