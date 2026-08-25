"""Static visualizations for the locked six-core attention-routing analysis.

This module deliberately contains no model, clustering, or biological annotation
logic.  It renders already-audited cell assignments, dissolved GeoJSON regions,
and retained reciprocal routing edges in physical micrometre coordinates.  The
fixed panel order and terminology are part of the analysis contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


CORE_ORDER: tuple[int, ...] = (1, 9, 13, 15, 21, 23)
"""The locked row-major order: 1, 9, 13 / 15, 21, 23."""

COMBINED_MAP_STEM = "six_core_attention_niche_map"
OVERLAY_MAP_STEM = "six_core_mutual_attention_network_overlay"
NEUTRAL_COLOR = "#9CA3AF"
COORDINATE_UNIT = "um"


class AttentionNicheVisualizationError(ValueError):
    """Raised when an input cannot support an auditable spatial map."""


@dataclass(frozen=True)
class AttentionNicheVisualizationArtifacts:
    """Paths written by :func:`render_attention_niche_visualizations`."""

    combined_png: Path
    combined_pdf: Path
    combined_svg: Path
    individual_pngs: tuple[Path, ...]
    overlay_png: Path | None
    overlay_pdf: Path | None
    receipt: Mapping[str, Any]

    @property
    def all_paths(self) -> tuple[Path, ...]:
        required = (
            self.combined_png,
            self.combined_pdf,
            self.combined_svg,
            *self.individual_pngs,
        )
        optional = tuple(
            path for path in (self.overlay_png, self.overlay_pdf) if path is not None
        )
        return required + optional


_ASSIGNMENT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "core_number": ("core_number", "core", "core number"),
    "core_alias": ("core_alias", "core alias"),
    "cell_index": (
        "cell_index",
        "prepared_cell_index",
        "global_cell_index",
        "cell index",
    ),
    "x_um": ("x_um", "x_coordinate", "x coordinate", "x"),
    "y_um": ("y_um", "y_coordinate", "y coordinate", "y"),
    "final_niche_id": (
        "final_niche_id",
        "final_connected_niche_id",
        "niche_id",
        "final connected niche ID",
    ),
    "niche_color": ("niche_color", "niche colour", "niche color"),
    "assignment_confidence": (
        "assignment_confidence",
        "confidence",
        "assignment confidence",
    ),
    "mutual_routing_hub_score": (
        "mutual_routing_hub_score",
        "hub_score",
        "S_i",
        "Si",
        "mutual-routing hub score Si",
    ),
}

_EDGE_ALIASES: Mapping[str, tuple[str, ...]] = {
    "core_number": ("core_number", "core", "core number"),
    "cell_i": ("cell_i", "cell i", "i", "cell_i_index"),
    "cell_j": ("cell_j", "cell j", "j", "cell_j_index"),
    "Mij": (
        "Mij",
        "M_ij",
        "consensus_mutual_score",
        "consensus_score",
        "mutual_score",
    ),
    "support_Pij": (
        "support_Pij",
        "support_P_ij",
        "Pij",
        "support_fraction",
        "support",
        "support Pij",
    ),
}


def _resolve_column(
    frame: pd.DataFrame,
    canonical: str,
    aliases: Mapping[str, tuple[str, ...]],
    *,
    required: bool,
) -> str | None:
    matches = [name for name in aliases[canonical] if name in frame.columns]
    if len(matches) > 1:
        first = frame[matches[0]]
        for other_name in matches[1:]:
            if not first.equals(frame[other_name]):
                raise AttentionNicheVisualizationError(
                    f"Conflicting columns for {canonical!r}: {matches}."
                )
    if matches:
        return matches[0]
    if required:
        raise AttentionNicheVisualizationError(
            f"Missing {canonical!r}; accepted columns are {aliases[canonical]}."
        )
    return None


def _integer_core_values(series: pd.Series, *, name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise AttentionNicheVisualizationError(f"{name} must contain finite integers.")
    return pd.Series(values.astype(np.int64), index=series.index, name=name)


def _prepare_assignments(
    assignments: pd.DataFrame,
    *,
    require_exact_cores: bool = True,
    allow_cross_core_color_reuse: bool = False,
) -> pd.DataFrame:
    if not isinstance(assignments, pd.DataFrame) or assignments.empty:
        raise AttentionNicheVisualizationError(
            "Cell assignments must be a non-empty pandas DataFrame."
        )

    required = (
        "core_number",
        "core_alias",
        "cell_index",
        "x_um",
        "y_um",
        "final_niche_id",
        "niche_color",
    )
    columns = {
        name: _resolve_column(assignments, name, _ASSIGNMENT_ALIASES, required=True)
        for name in required
    }
    for optional in ("assignment_confidence", "mutual_routing_hub_score"):
        columns[optional] = _resolve_column(
            assignments, optional, _ASSIGNMENT_ALIASES, required=False
        )

    prepared = pd.DataFrame(index=np.arange(len(assignments)))
    prepared["core_number"] = _integer_core_values(
        assignments[columns["core_number"]].reset_index(drop=True),
        name="core_number",
    )
    prepared["core_alias"] = (
        assignments[columns["core_alias"]].reset_index(drop=True).astype(str)
    )
    if prepared["core_alias"].str.strip().eq("").any():
        raise AttentionNicheVisualizationError("Core aliases cannot be blank.")
    prepared["cell_index"] = assignments[columns["cell_index"]].reset_index(drop=True)
    prepared["x_um"] = pd.to_numeric(
        assignments[columns["x_um"]].reset_index(drop=True), errors="coerce"
    )
    prepared["y_um"] = pd.to_numeric(
        assignments[columns["y_um"]].reset_index(drop=True), errors="coerce"
    )
    coordinates = prepared[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise AttentionNicheVisualizationError(
            "Physical cell coordinates must be finite micrometre values."
        )

    raw_niches = assignments[columns["final_niche_id"]].reset_index(drop=True)
    prepared["final_niche_id"] = raw_niches.astype("object")
    blank_niche = raw_niches.isna() | raw_niches.astype(str).str.strip().eq("")
    prepared.loc[blank_niche, "final_niche_id"] = None

    raw_colors = assignments[columns["niche_color"]].reset_index(drop=True)
    prepared["niche_color"] = raw_colors.astype("object")
    prepared.loc[blank_niche, "niche_color"] = NEUTRAL_COLOR
    assigned_missing_color = (~blank_niche) & (
        raw_colors.isna() | raw_colors.astype(str).str.strip().eq("")
    )
    if assigned_missing_color.any():
        raise AttentionNicheVisualizationError(
            "Every assigned niche requires a categorical niche_color."
        )

    from matplotlib.colors import is_color_like, to_hex

    normalized_colors: list[str] = []
    for value in prepared["niche_color"].tolist():
        if not is_color_like(value):
            raise AttentionNicheVisualizationError(
                f"Invalid Matplotlib niche color: {value!r}."
            )
        normalized_colors.append(to_hex(value, keep_alpha=False).upper())
    prepared["niche_color"] = normalized_colors

    confidence_column = columns["assignment_confidence"]
    if confidence_column is None:
        prepared["assignment_confidence"] = 1.0
    else:
        prepared["assignment_confidence"] = pd.to_numeric(
            assignments[confidence_column].reset_index(drop=True), errors="coerce"
        )
        confidence = prepared["assignment_confidence"].to_numpy(dtype=np.float64)
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise AttentionNicheVisualizationError(
                "Assignment confidence must be finite and lie in [0, 1]."
            )

    hub_column = columns["mutual_routing_hub_score"]
    if hub_column is None:
        prepared["mutual_routing_hub_score"] = 0.0
    else:
        prepared["mutual_routing_hub_score"] = pd.to_numeric(
            assignments[hub_column].reset_index(drop=True), errors="coerce"
        )
        hubs = prepared["mutual_routing_hub_score"].to_numpy(dtype=np.float64)
        if not np.isfinite(hubs).all() or np.any(hubs < 0):
            raise AttentionNicheVisualizationError(
                "Mutual-routing hub scores must be finite and non-negative."
            )

    observed_cores = tuple(sorted(set(prepared["core_number"].tolist())))
    if require_exact_cores and observed_cores != tuple(sorted(CORE_ORDER)):
        raise AttentionNicheVisualizationError(
            f"Expected exactly cores {CORE_ORDER}; observed {observed_cores}."
        )
    unexpected = set(observed_cores).difference(CORE_ORDER)
    if unexpected:
        raise AttentionNicheVisualizationError(
            f"Unexpected cores in the six-core map: {sorted(unexpected)}."
        )

    if prepared.duplicated(["core_number", "cell_index"]).any():
        raise AttentionNicheVisualizationError(
            "Each (core_number, cell_index) must appear exactly once."
        )
    alias_counts = prepared.groupby("core_number")["core_alias"].nunique()
    if (alias_counts != 1).any():
        raise AttentionNicheVisualizationError(
            "Every core number must map to exactly one core alias."
        )
    alias_core_counts = prepared.groupby("core_alias")["core_number"].nunique()
    if (alias_core_counts != 1).any():
        raise AttentionNicheVisualizationError(
            "A core alias cannot refer to more than one core number."
        )

    assigned = prepared.loc[prepared["final_niche_id"].notna()]
    color_counts = assigned.groupby(
        ["core_number", "final_niche_id"], sort=False, dropna=False
    )["niche_color"].nunique()
    if (color_counts != 1).any():
        raise AttentionNicheVisualizationError(
            "A niche has more than one cell color; dots and fills would disagree."
        )
    # Non-adjacent niches may reuse a graph-color class.  The geometry module
    # has the niche-adjacency graph and is therefore the authoritative place
    # to reject an adjacent color collision; requiring globally unique colors
    # here would defeat deterministic categorical graph coloring on maps with
    # many connected components.
    if not allow_cross_core_color_reuse:
        palette = assigned[
            ["core_number", "final_niche_id", "niche_color"]
        ].drop_duplicates()
        cross_core_counts = palette.groupby("niche_color")["core_number"].nunique()
        reused = cross_core_counts[cross_core_counts > 1]
        if not reused.empty:
            raise AttentionNicheVisualizationError(
                "Niche colors are core-scoped and must not be reused across cores; "
                f"reused colors: {sorted(reused.index.tolist())}."
            )
    return prepared


def _feature_collection_features(
    regions: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    if not isinstance(regions, Mapping) or regions.get("type") != "FeatureCollection":
        raise AttentionNicheVisualizationError(
            "Dissolved regions must be a GeoJSON FeatureCollection mapping."
        )
    raw_features = regions.get("features")
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        raise AttentionNicheVisualizationError(
            "GeoJSON FeatureCollection.features must be a sequence."
        )
    features: list[Mapping[str, Any]] = []
    for position, feature in enumerate(raw_features):
        if not isinstance(feature, Mapping) or feature.get("type") != "Feature":
            raise AttentionNicheVisualizationError(
                f"Region feature {position} is not a GeoJSON Feature."
            )
        features.append(feature)
    return features


def _feature_property(
    properties: Mapping[str, Any], aliases: Iterable[str], *, required: bool
) -> Any:
    present = [name for name in aliases if name in properties]
    if len(present) > 1:
        first = properties[present[0]]
        if any(properties[name] != first for name in present[1:]):
            raise AttentionNicheVisualizationError(
                f"Conflicting GeoJSON properties: {present}."
            )
    if present:
        return properties[present[0]]
    if required:
        raise AttentionNicheVisualizationError(
            f"Missing GeoJSON property; accepted names are {tuple(aliases)}."
        )
    return None


def _geometry_type_and_coordinates(
    geometry: Any, *, feature_position: int
) -> tuple[str, Sequence[Any]]:
    if not isinstance(geometry, Mapping):
        raise AttentionNicheVisualizationError(
            f"Region feature {feature_position} has no geometry mapping."
        )
    geometry_type = geometry.get("type")
    if geometry_type not in {"Polygon", "MultiPolygon"}:
        raise AttentionNicheVisualizationError(
            f"Region feature {feature_position} must be Polygon or MultiPolygon, "
            f"not {geometry_type!r}."
        )
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, Sequence) or isinstance(coordinates, (str, bytes)):
        raise AttentionNicheVisualizationError(
            f"Region feature {feature_position} has invalid coordinates."
        )
    return str(geometry_type), coordinates


def _prepare_regions(
    regions: Mapping[str, Any],
    assignments: pd.DataFrame,
    *,
    require_complete_regions: bool = True,
) -> dict[int, list[dict[str, Any]]]:
    from matplotlib.colors import is_color_like, to_hex

    palette = {
        (int(row.core_number), str(row.final_niche_id)): str(row.niche_color)
        for row in assignments.loc[assignments["final_niche_id"].notna()].itertuples()
    }
    by_core: dict[int, list[dict[str, Any]]] = {core: [] for core in CORE_ORDER}
    covered: set[tuple[int, str]] = set()
    features = _feature_collection_features(regions)
    for position, feature in enumerate(features):
        properties = feature.get("properties")
        if not isinstance(properties, Mapping):
            raise AttentionNicheVisualizationError(
                f"Region feature {position} has no properties mapping."
            )
        raw_core = _feature_property(
            properties, ("core_number", "core", "core number"), required=True
        )
        try:
            numeric_core = float(raw_core)
        except (TypeError, ValueError) as error:
            raise AttentionNicheVisualizationError(
                f"Region feature {position} has invalid core {raw_core!r}."
            ) from error
        if not math.isfinite(numeric_core) or numeric_core != math.floor(numeric_core):
            raise AttentionNicheVisualizationError(
                f"Region feature {position} has non-integer core {raw_core!r}."
            )
        core = int(numeric_core)
        if core not in CORE_ORDER:
            raise AttentionNicheVisualizationError(
                f"Region feature {position} has unexpected core {core}."
            )
        niche = str(
            _feature_property(
                properties,
                ("final_niche_id", "final_connected_niche_id", "niche_id"),
                required=True,
            )
        )
        key = (core, niche)
        if key not in palette:
            raise AttentionNicheVisualizationError(
                f"Region {niche!r} in core {core} has no assigned cells."
            )
        raw_color = _feature_property(
            properties, ("niche_color", "niche color"), required=False
        )
        if raw_color is not None:
            if not is_color_like(raw_color):
                raise AttentionNicheVisualizationError(
                    f"Region {niche!r} has invalid niche_color {raw_color!r}."
                )
            region_color = to_hex(raw_color, keep_alpha=False).upper()
            if region_color != palette[key]:
                raise AttentionNicheVisualizationError(
                    f"Region and cell colors disagree for {niche!r} in core {core}."
                )
        raw_unit = _feature_property(
            properties,
            ("coordinate_unit", "coordinate_units", "coordinate unit"),
            required=False,
        )
        if raw_unit is not None and str(raw_unit).strip().lower() not in {
            "um",
            "µm",
            "micrometre",
            "micrometres",
            "micrometer",
            "micrometers",
        }:
            raise AttentionNicheVisualizationError(
                f"Region {niche!r} is not recorded in micrometres: {raw_unit!r}."
            )
        geometry_type, coordinates = _geometry_type_and_coordinates(
            feature.get("geometry"), feature_position=position
        )
        by_core[core].append(
            {
                "niche_id": niche,
                "color": palette[key],
                "geometry_type": geometry_type,
                "coordinates": coordinates,
                "position": position,
            }
        )
        covered.add(key)

    if require_complete_regions:
        missing = sorted(set(palette).difference(covered))
        if missing:
            display = ", ".join(f"core {core} {niche}" for core, niche in missing[:8])
            suffix = " ..." if len(missing) > 8 else ""
            raise AttentionNicheVisualizationError(
                "Dissolved geometry is missing for assigned niches: " + display + suffix
            )
    for core in CORE_ORDER:
        by_core[core].sort(key=lambda value: (value["niche_id"], value["position"]))
    return by_core


def _ring_array(
    ring: Any, *, context: str, allow_zero_area: bool = False
) -> np.ndarray | None:
    try:
        values = np.asarray(ring, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AttentionNicheVisualizationError(
            f"{context} contains non-numeric coordinates."
        ) from error
    if values.ndim != 2 or values.shape[1] < 2 or len(values) < 4:
        raise AttentionNicheVisualizationError(
            f"{context} must contain at least four coordinate pairs."
        )
    values = values[:, :2]
    if not np.isfinite(values).all():
        raise AttentionNicheVisualizationError(f"{context} contains non-finite coordinates.")
    if not np.allclose(values[0], values[-1], rtol=0.0, atol=1e-10):
        raise AttentionNicheVisualizationError(f"{context} is not a closed GeoJSON ring.")
    unique = values[:-1]
    if len(unique) < 3:
        raise AttentionNicheVisualizationError(f"{context} has fewer than three vertices.")
    area_twice = _stable_signed_area_twice(unique)
    if not math.isfinite(area_twice):
        raise AttentionNicheVisualizationError(f"{context} has zero signed area.")
    if area_twice == 0.0:
        if allow_zero_area:
            return None
        raise AttentionNicheVisualizationError(f"{context} has zero signed area.")
    return unique


def _stable_signed_area_twice(vertices: np.ndarray) -> float:
    """Return shoelace area after translation to avoid cancellation.

    Tissue coordinates can be orders of magnitude larger than tiny valid
    polygon slivers.  Subtracting one vertex preserves signed area while
    preventing the two large shoelace sums from rounding to the same float.
    """

    translated = np.asarray(vertices, dtype=np.float64) - vertices[0]
    return float(
        np.dot(translated[:, 0], np.roll(translated[:, 1], -1))
        - np.dot(translated[:, 1], np.roll(translated[:, 0], -1))
    )


def _polygon_path(rings: Any, *, context: str) -> Any:
    from matplotlib.path import Path as MatplotlibPath

    if not isinstance(rings, Sequence) or isinstance(rings, (str, bytes)) or not rings:
        raise AttentionNicheVisualizationError(f"{context} has no rings.")
    vertices: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    for ring_index, raw_ring in enumerate(rings):
        ring = _ring_array(
            raw_ring,
            context=f"{context} ring {ring_index}",
            # GEOS permits a zero-area interior ring in an otherwise valid
            # polygon.  It has no fill effect, so omit it from the plotting
            # path while continuing to reject a degenerate exterior.
            allow_zero_area=ring_index > 0,
        )
        if ring is None:
            continue
        area_twice = _stable_signed_area_twice(ring)
        want_counter_clockwise = ring_index == 0
        if (area_twice > 0) != want_counter_clockwise:
            ring = ring[::-1]
        closed = np.vstack((ring, ring[0]))
        ring_codes = np.full(len(closed), MatplotlibPath.LINETO, dtype=np.uint8)
        ring_codes[0] = MatplotlibPath.MOVETO
        ring_codes[-1] = MatplotlibPath.CLOSEPOLY
        vertices.append(closed)
        codes.append(ring_codes)
    return MatplotlibPath(np.vstack(vertices), np.concatenate(codes))


def _geometry_paths(region: Mapping[str, Any]) -> list[Any]:
    geometry_type = region["geometry_type"]
    coordinates = region["coordinates"]
    context = f"core region {region['niche_id']!r}"
    if geometry_type == "Polygon":
        return [_polygon_path(coordinates, context=context)]
    if not coordinates:
        raise AttentionNicheVisualizationError(f"{context} has no polygons.")
    return [
        _polygon_path(polygon, context=f"{context} polygon {position}")
        for position, polygon in enumerate(coordinates)
    ]


def _omitted_zero_area_interior_ring_receipt(
    regions_by_core: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Describe degenerate interior rings omitted only from Matplotlib paths."""

    identifiers: list[str] = []
    for core_number in sorted(regions_by_core):
        for region in regions_by_core[core_number]:
            polygons = (
                [region["coordinates"]]
                if region["geometry_type"] == "Polygon"
                else region["coordinates"]
            )
            for polygon_index, rings in enumerate(polygons):
                for ring_index, raw_ring in enumerate(rings[1:], start=1):
                    ring = _ring_array(
                        raw_ring,
                        context=(
                            f"core region {region['niche_id']!r} polygon "
                            f"{polygon_index} ring {ring_index}"
                        ),
                        allow_zero_area=True,
                    )
                    if ring is None:
                        identifiers.append(
                            f"C{int(core_number):02d}|{region['niche_id']}|"
                            f"feature={int(region['position'])}|"
                            f"polygon={polygon_index}|ring={ring_index}"
                        )
    identifiers.sort()
    encoded = "\n".join(identifiers).encode("utf-8")
    return {
        "count": len(identifiers),
        "identifiers": identifiers,
        "identifier_list_sha256": hashlib.sha256(encoded).hexdigest(),
        "signed_area_um2": 0.0,
        "scientific_geometry_modified": False,
    }


def _all_region_points(regions: Sequence[Mapping[str, Any]]) -> np.ndarray:
    point_arrays: list[np.ndarray] = []
    for region in regions:
        for path in _geometry_paths(region):
            point_arrays.append(np.asarray(path.vertices, dtype=np.float64))
    if not point_arrays:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(point_arrays)


def _nice_scale_bar_length(physical_extent: float) -> float:
    if not math.isfinite(physical_extent) or physical_extent <= 0:
        raise AttentionNicheVisualizationError(
            "Cannot derive a physical micrometre scale bar from a zero-size panel."
        )
    target = physical_extent * 0.20
    exponent = 10.0 ** math.floor(math.log10(target))
    candidates = (exponent, 2.0 * exponent, 5.0 * exponent, 10.0 * exponent)
    return max(value for value in candidates if value <= target)


def _core_bounds(
    cells: pd.DataFrame, regions: Sequence[Mapping[str, Any]]
) -> tuple[float, float, float, float]:
    cell_points = cells[["x_um", "y_um"]].to_numpy(dtype=np.float64)
    region_points = _all_region_points(regions)
    points = cell_points if len(region_points) == 0 else np.vstack((cell_points, region_points))
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_span = float(x_max - x_min)
    y_span = float(y_max - y_min)
    reference = max(x_span, y_span)
    if not math.isfinite(reference) or reference <= 0:
        reference = 1.0
    x_pad = max(0.035 * x_span, 0.015 * reference)
    y_pad = max(0.035 * y_span, 0.015 * reference)
    return (
        float(x_min - x_pad),
        float(x_max + x_pad),
        float(y_min - y_pad),
        float(y_max + y_pad),
    )


def _add_regions(
    axis: Any,
    regions: Sequence[Mapping[str, Any]],
    *,
    fill_alpha: float,
    boundary_alpha: float,
    boundary_width: float,
) -> None:
    from matplotlib.colors import to_rgba
    from matplotlib.patches import PathPatch

    for region in regions:
        color = str(region["color"])
        for path in _geometry_paths(region):
            patch = PathPatch(
                path,
                facecolor=to_rgba(color, fill_alpha),
                edgecolor=to_rgba(color, boundary_alpha),
                linewidth=boundary_width,
                antialiased=True,
                joinstyle="round",
                zorder=1,
            )
            patch.set_gid(f"region-{region['niche_id']}")
            axis.add_patch(patch)


def _add_scale_bar(
    axis: Any,
    bounds: tuple[float, float, float, float],
    *,
    core_number: int,
) -> None:
    x_min, x_max, y_min, y_max = bounds
    x_span = x_max - x_min
    y_span = y_max - y_min
    length = _nice_scale_bar_length(x_span)
    x_start = x_min + 0.055 * x_span
    y_value = y_max - 0.055 * y_span
    (line,) = axis.plot(
        [x_start, x_start + length],
        [y_value, y_value],
        color="#111827",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=8,
    )
    line.set_gid(f"scale-bar-core-{core_number}")
    label = axis.text(
        x_start + length / 2.0,
        y_value - 0.018 * y_span,
        f"{length:g} µm",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#111827",
        zorder=8,
    )
    label.set_gid(f"scale-bar-label-core-{core_number}")


def _style_axis(
    axis: Any,
    bounds: tuple[float, float, float, float],
    *,
    core_number: int,
    invert_y: bool,
) -> None:
    x_min, x_max, y_min, y_max = bounds
    axis.set_xlim(x_min, x_max)
    axis.set_ylim((y_max, y_min) if invert_y else (y_min, y_max))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.tick_params(which="both", length=0)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_facecolor("#F8FAFC")
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.7)
    _add_scale_bar(axis, bounds, core_number=core_number)


def _add_core_label(axis: Any, core_number: int) -> None:
    label = axis.text(
        0.025,
        0.972,
        f"Core {core_number}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": (1.0, 1.0, 1.0, 0.82),
            "edgecolor": "#CBD5E1",
            "linewidth": 0.5,
        },
        zorder=9,
    )
    label.set_gid(f"in-panel-core-label-{core_number}")


def _draw_cell_points(
    axis: Any,
    cells: pd.DataFrame,
    *,
    point_size: float,
    low_confidence_threshold: float | None,
    node_sizes: np.ndarray | None = None,
) -> Any:
    colors = cells["niche_color"].tolist()
    if low_confidence_threshold is None:
        low_confidence = np.zeros(len(cells), dtype=bool)
    else:
        if not 0 <= low_confidence_threshold <= 1:
            raise AttentionNicheVisualizationError(
                "low_confidence_threshold must lie in [0, 1] or be None."
            )
        low_confidence = (
            cells["assignment_confidence"].to_numpy(dtype=np.float64)
            < low_confidence_threshold
        )
    edge_colors = np.where(low_confidence, "#4B5563", "none").tolist()
    line_widths = np.where(low_confidence, 0.35, 0.0)
    sizes: float | np.ndarray = point_size if node_sizes is None else node_sizes
    scatter = axis.scatter(
        cells["x_um"].to_numpy(dtype=np.float64),
        cells["y_um"].to_numpy(dtype=np.float64),
        s=sizes,
        c=colors,
        marker="o",
        edgecolors=edge_colors,
        linewidths=line_widths,
        alpha=0.96,
        zorder=5,
    )
    scatter.set_gid("all-eligible-cells")
    return scatter


def _draw_niche_panel(
    axis: Any,
    cells: pd.DataFrame,
    regions: Sequence[Mapping[str, Any]],
    *,
    core_number: int,
    invert_y: bool,
    point_size: float,
    low_confidence_threshold: float | None,
) -> None:
    _add_regions(
        axis,
        regions,
        fill_alpha=0.24,
        boundary_alpha=0.95,
        boundary_width=0.65,
    )
    _draw_cell_points(
        axis,
        cells,
        point_size=point_size,
        low_confidence_threshold=low_confidence_threshold,
    )
    niche_count = int(cells["final_niche_id"].dropna().nunique())
    axis.set_title(
        f"Core {core_number}  |  {len(cells):,} cells  |  {niche_count:,} niches",
        fontsize=11.0,
        fontweight="semibold",
        pad=7.0,
    )
    _add_core_label(axis, core_number)
    bounds = _core_bounds(cells, regions)
    _style_axis(axis, bounds, core_number=core_number, invert_y=invert_y)


def _visualization_context() -> Any:
    import matplotlib as mpl

    return mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlecolor": "#111827",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "bagm-six-core-attention-routing-niches-v1",
        }
    )


def _optional_map_label(assignments: pd.DataFrame) -> str | None:
    if "map_label" not in assignments.columns:
        return None
    labels = sorted(
        {
            str(value).strip()
            for value in assignments["map_label"].dropna().tolist()
            if str(value).strip()
        }
    )
    if len(labels) > 1:
        raise AttentionNicheVisualizationError(
            "Cell assignments contain more than one map_label."
        )
    return labels[0] if labels else None


def _add_confidence_note(
    figure: Any,
    *,
    low_confidence_threshold: float | None,
) -> None:
    if low_confidence_threshold is None:
        return
    figure.text(
        0.5,
        0.008,
        (
            "Gray cell outlines mark assignment confidence "
            f"< {low_confidence_threshold:.2f}; colors remain the assigned niche."
        ),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#374151",
    )


def create_combined_attention_niche_figure(
    assignments: pd.DataFrame,
    regions: Mapping[str, Any],
    *,
    invert_y: bool = True,
    low_confidence_threshold: float | None = 0.60,
    point_size: float = 3.0,
    allow_cross_core_color_reuse: bool = False,
) -> Any:
    """Create the locked 2x3 six-core niche figure without saving it."""

    import matplotlib.pyplot as plt

    map_label = _optional_map_label(assignments)
    prepared = _prepare_assignments(
        assignments,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    prepared_regions = _prepare_regions(regions, prepared)
    with _visualization_context():
        figure, axes = plt.subplots(2, 3, figsize=(18.0, 11.5))
        for axis, core_number in zip(axes.ravel(), CORE_ORDER, strict=True):
            cells = prepared.loc[prepared["core_number"] == core_number]
            _draw_niche_panel(
                axis,
                cells,
                prepared_regions[core_number],
                core_number=core_number,
                invert_y=invert_y,
                point_size=point_size,
                low_confidence_threshold=low_confidence_threshold,
            )
        title = "Model-defined attention-routing niches — six cancer cores"
        if map_label is not None:
            title += f"\n{map_label}"
        figure.suptitle(
            title,
            fontsize=16.0,
            fontweight="bold",
            y=0.985,
        )
        figure.subplots_adjust(
            left=0.035,
            right=0.985,
            bottom=0.035,
            top=0.935,
            wspace=0.12,
            hspace=0.17,
        )
        _add_confidence_note(
            figure,
            low_confidence_threshold=low_confidence_threshold,
        )
    return figure


def _create_individual_attention_niche_figure(
    prepared: pd.DataFrame,
    prepared_regions: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    core_number: int,
    invert_y: bool,
    low_confidence_threshold: float | None,
    point_size: float,
    map_label: str | None,
) -> Any:
    import matplotlib.pyplot as plt

    cells = prepared.loc[prepared["core_number"] == core_number]
    with _visualization_context():
        figure, axis = plt.subplots(figsize=(11.0, 9.5))
        _draw_niche_panel(
            axis,
            cells,
            prepared_regions[core_number],
            core_number=core_number,
            invert_y=invert_y,
            point_size=point_size,
            low_confidence_threshold=low_confidence_threshold,
        )
        title = f"Core {core_number} model-defined attention-routing niches"
        if map_label is not None:
            title += f"\n{map_label}"
        figure.suptitle(
            title,
            fontsize=15.0,
            fontweight="bold",
            y=0.985,
        )
        figure.subplots_adjust(left=0.035, right=0.985, bottom=0.035, top=0.91)
        _add_confidence_note(
            figure,
            low_confidence_threshold=low_confidence_threshold,
        )
    return figure


def _format_metadata(format_name: str) -> dict[str, Any]:
    creator = "spatial_benchmark.attention_niche_visualization"
    if format_name == "png":
        return {"Software": creator}
    if format_name == "pdf":
        return {
            "Creator": creator,
            "Producer": creator,
            "CreationDate": None,
            "ModDate": None,
        }
    if format_name == "svg":
        return {"Creator": creator, "Date": None}
    raise AttentionNicheVisualizationError(
        f"Unsupported atomic figure format {format_name!r}."
    )


def _atomic_save_figure_formats(
    figure: Any,
    targets: Mapping[str, Path],
    *,
    dpi: int,
) -> tuple[Path, ...]:
    if not isinstance(dpi, int) or dpi <= 0:
        raise AttentionNicheVisualizationError("Figure DPI must be a positive integer.")
    temporary_paths: list[tuple[Path, Path]] = []
    try:
        for format_name, raw_target in targets.items():
            if format_name not in {"png", "pdf", "svg"}:
                raise AttentionNicheVisualizationError(
                    f"Unsupported figure format {format_name!r}."
                )
            target = Path(raw_target)
            if target.suffix.lower() != f".{format_name}":
                raise AttentionNicheVisualizationError(
                    f"Target {target} does not match format {format_name}."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.stem}.tmp-",
                suffix=f".{format_name}",
                dir=target.parent,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            figure.savefig(
                temporary,
                format=format_name,
                dpi=dpi if format_name == "png" else None,
                bbox_inches="tight",
                pad_inches=0.05,
                facecolor="white",
                metadata=_format_metadata(format_name),
            )
            temporary_paths.append((temporary, target))
        for temporary, target in temporary_paths:
            os.replace(temporary, target)
        return tuple(Path(target) for target in targets.values())
    finally:
        for temporary, _target in temporary_paths:
            temporary.unlink(missing_ok=True)


def render_combined_attention_niche_map(
    assignments: pd.DataFrame,
    regions: Mapping[str, Any],
    output_dir: str | Path,
    *,
    dpi: int = 300,
    invert_y: bool = True,
    low_confidence_threshold: float | None = 0.60,
    point_size: float = 3.0,
    allow_cross_core_color_reuse: bool = False,
) -> tuple[Path, Path, Path]:
    """Atomically save the required combined PNG, PDF, and SVG."""

    import matplotlib.pyplot as plt

    output_root = Path(output_dir)
    figure = create_combined_attention_niche_figure(
        assignments,
        regions,
        invert_y=invert_y,
        low_confidence_threshold=low_confidence_threshold,
        point_size=point_size,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    try:
        paths = _atomic_save_figure_formats(
            figure,
            {
                "png": output_root / f"{COMBINED_MAP_STEM}.png",
                "pdf": output_root / f"{COMBINED_MAP_STEM}.pdf",
                "svg": output_root / f"{COMBINED_MAP_STEM}.svg",
            },
            dpi=dpi,
        )
        return paths  # type: ignore[return-value]
    finally:
        plt.close(figure)


def render_individual_attention_niche_maps(
    assignments: pd.DataFrame,
    regions: Mapping[str, Any],
    output_dir: str | Path,
    *,
    dpi: int = 450,
    invert_y: bool = True,
    low_confidence_threshold: float | None = 0.60,
    point_size: float = 5.0,
    allow_cross_core_color_reuse: bool = False,
) -> tuple[Path, ...]:
    """Atomically save one high-resolution PNG for every locked core."""

    import matplotlib.pyplot as plt

    prepared = _prepare_assignments(
        assignments,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    map_label = _optional_map_label(assignments)
    prepared_regions = _prepare_regions(regions, prepared)
    output_root = Path(output_dir)
    outputs: list[Path] = []
    for core_number in CORE_ORDER:
        figure = _create_individual_attention_niche_figure(
            prepared,
            prepared_regions,
            core_number=core_number,
            invert_y=invert_y,
            low_confidence_threshold=low_confidence_threshold,
            point_size=point_size,
            map_label=map_label,
        )
        target = output_root / f"core_{core_number:02d}_attention_niche_map.png"
        try:
            (saved,) = _atomic_save_figure_formats(
                figure, {"png": target}, dpi=dpi
            )
            outputs.append(saved)
        finally:
            plt.close(figure)
    return tuple(outputs)


def _prepare_mutual_edges(mutual_edges: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(mutual_edges, pd.DataFrame):
        raise AttentionNicheVisualizationError(
            "Mutual edges must be supplied as a pandas DataFrame."
        )
    columns = {
        name: _resolve_column(mutual_edges, name, _EDGE_ALIASES, required=True)
        for name in _EDGE_ALIASES
    }
    retained_column = next(
        (
            name
            for name in ("retained_primary", "retained", "is_retained_primary")
            if name in mutual_edges.columns
        ),
        None,
    )
    source = mutual_edges.reset_index(drop=True)
    if retained_column is not None:
        raw_retained = source[retained_column]
        if pd.api.types.is_bool_dtype(raw_retained.dtype):
            retained_mask = raw_retained.astype(bool)
        elif pd.api.types.is_numeric_dtype(raw_retained.dtype):
            numeric_retained = pd.to_numeric(raw_retained, errors="coerce")
            if numeric_retained.isna().any() or not numeric_retained.isin([0, 1]).all():
                raise AttentionNicheVisualizationError(
                    "retained_primary must contain only boolean or 0/1 values."
                )
            retained_mask = numeric_retained.astype(bool)
        else:
            normalized = raw_retained.astype(str).str.strip().str.lower()
            if not normalized.isin({"true", "false"}).all():
                raise AttentionNicheVisualizationError(
                    "retained_primary must contain only boolean values."
                )
            retained_mask = normalized.eq("true")
        source = source.loc[retained_mask].reset_index(drop=True)
    prepared = pd.DataFrame(index=np.arange(len(source)))
    prepared["core_number"] = _integer_core_values(
        source[columns["core_number"]].reset_index(drop=True),
        name="core_number",
    )
    prepared["cell_i"] = source[columns["cell_i"]].reset_index(drop=True)
    prepared["cell_j"] = source[columns["cell_j"]].reset_index(drop=True)
    prepared["Mij"] = pd.to_numeric(
        source[columns["Mij"]].reset_index(drop=True), errors="coerce"
    )
    prepared["support_Pij"] = pd.to_numeric(
        source[columns["support_Pij"]].reset_index(drop=True), errors="coerce"
    )
    prepared["_source_position"] = np.arange(len(prepared), dtype=np.int64)
    scores = prepared["Mij"].to_numpy(dtype=np.float64)
    support = prepared["support_Pij"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise AttentionNicheVisualizationError(
            "Consensus mutual scores must be finite and non-negative."
        )
    if not np.isfinite(support).all() or np.any((support < 0) | (support > 1)):
        raise AttentionNicheVisualizationError(
            "Mutual-edge support fractions must lie in [0, 1]."
        )
    if not set(prepared["core_number"]).issubset(CORE_ORDER):
        raise AttentionNicheVisualizationError("Mutual edges contain an unexpected core.")
    if (prepared["cell_i"] == prepared["cell_j"]).any():
        raise AttentionNicheVisualizationError(
            "A retained mutual edge cannot be a self edge."
        )
    pair_keys = prepared.apply(
        lambda row: (
            int(row["core_number"]),
            min(str(row["cell_i"]), str(row["cell_j"])),
            max(str(row["cell_i"]), str(row["cell_j"])),
        ),
        axis=1,
    )
    if pair_keys.duplicated().any():
        raise AttentionNicheVisualizationError(
            "The retained mutual-edge table contains duplicate undirected pairs."
        )
    return prepared


def _validate_edge_cap(value: int | None, *, name: str) -> None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
        raise AttentionNicheVisualizationError(
            f"{name} must be a non-negative integer or None."
        )


def select_strongest_mutual_edges(
    mutual_edges: pd.DataFrame,
    *,
    max_edges_per_core: int | None = 1_000,
    max_edges_total: int | None = 4_000,
) -> pd.DataFrame:
    """Select display edges deterministically without changing analysis edges.

    The per-core cap is applied first.  The optional total cap is then applied
    globally by ``Mij`` with core number and endpoint strings as stable tie
    breakers.  The returned table has canonical visualization columns.
    """

    _validate_edge_cap(max_edges_per_core, name="max_edges_per_core")
    _validate_edge_cap(max_edges_total, name="max_edges_total")
    prepared = _prepare_mutual_edges(mutual_edges)
    selected: list[pd.DataFrame] = []
    for core_number in CORE_ORDER:
        core_edges = prepared.loc[prepared["core_number"] == core_number].copy()
        core_edges["_cell_i_tie"] = core_edges["cell_i"].map(str)
        core_edges["_cell_j_tie"] = core_edges["cell_j"].map(str)
        core_edges = core_edges.sort_values(
            ["Mij", "support_Pij", "_cell_i_tie", "_cell_j_tie", "_source_position"],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
        if max_edges_per_core is not None:
            core_edges = core_edges.head(max_edges_per_core)
        selected.append(core_edges)
    if selected:
        result = pd.concat(selected, ignore_index=True)
    else:  # pragma: no cover - CORE_ORDER is never empty
        result = prepared.iloc[0:0].copy()
    result = result.sort_values(
        ["Mij", "support_Pij", "core_number", "_cell_i_tie", "_cell_j_tie"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    if max_edges_total is not None:
        result = result.head(max_edges_total)
    result = result.sort_values(
        ["core_number", "Mij", "support_Pij", "_cell_i_tie", "_cell_j_tie"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    )
    return result[
        ["core_number", "cell_i", "cell_j", "Mij", "support_Pij"]
    ].reset_index(drop=True)


def _cell_lookup(cells: pd.DataFrame) -> dict[Any, tuple[float, float]]:
    return {
        row.cell_index: (float(row.x_um), float(row.y_um))
        for row in cells.itertuples()
    }


def _lookup_endpoint(
    lookup: Mapping[Any, tuple[float, float]],
    endpoint: Any,
    *,
    core_number: int,
) -> tuple[float, float]:
    if endpoint in lookup:
        return lookup[endpoint]
    string_lookup = {str(key): value for key, value in lookup.items()}
    if str(endpoint) in string_lookup:
        return string_lookup[str(endpoint)]
    raise AttentionNicheVisualizationError(
        f"Mutual edge endpoint {endpoint!r} is absent from core {core_number} assignments."
    )


def _draw_overlay_panel(
    axis: Any,
    cells: pd.DataFrame,
    regions: Sequence[Mapping[str, Any]],
    edges: pd.DataFrame,
    *,
    core_number: int,
    invert_y: bool,
    low_confidence_threshold: float | None,
) -> None:
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    _add_regions(
        axis,
        regions,
        fill_alpha=0.12,
        boundary_alpha=0.70,
        boundary_width=0.45,
    )
    lookup = _cell_lookup(cells)
    segments: list[list[tuple[float, float]]] = []
    for row in edges.itertuples(index=False):
        start = _lookup_endpoint(lookup, row.cell_i, core_number=core_number)
        end = _lookup_endpoint(lookup, row.cell_j, core_number=core_number)
        segments.append([start, end])
    if segments:
        scores = edges["Mij"].to_numpy(dtype=np.float64)
        maximum_score = float(np.max(scores))
        widths = np.clip(2.2 * scores / maximum_score, 0.20, 2.2)
        support = edges["support_Pij"].to_numpy(dtype=np.float64)
        colors = [to_rgba("#334155", float(value)) for value in support]
        collection = LineCollection(
            segments,
            colors=colors,
            linewidths=widths,
            capstyle="round",
            zorder=3,
        )
        collection.set_gid(f"displayed-mutual-edges-core-{core_number}")
        axis.add_collection(collection)

    hubs = cells["mutual_routing_hub_score"].to_numpy(dtype=np.float64)
    maximum_hub = float(np.max(hubs)) if len(hubs) else 0.0
    if maximum_hub > 0:
        node_sizes = 2.0 + 16.0 * hubs / maximum_hub
    else:
        node_sizes = np.full(len(cells), 2.0, dtype=np.float64)
    _draw_cell_points(
        axis,
        cells,
        point_size=2.0,
        low_confidence_threshold=low_confidence_threshold,
        node_sizes=node_sizes,
    )
    niche_count = int(cells["final_niche_id"].dropna().nunique())
    axis.set_title(
        f"Core {core_number}  |  {len(cells):,} cells  |  {niche_count:,} niches"
        f"  |  {len(edges):,} edges shown",
        fontsize=10.5,
        fontweight="semibold",
        pad=7.0,
    )
    _add_core_label(axis, core_number)
    bounds = _core_bounds(cells, regions)
    _style_axis(axis, bounds, core_number=core_number, invert_y=invert_y)


def create_mutual_attention_network_overlay_figure(
    assignments: pd.DataFrame,
    regions: Mapping[str, Any],
    mutual_edges: pd.DataFrame,
    *,
    max_edges_per_core: int | None = 1_000,
    max_edges_total: int | None = 4_000,
    invert_y: bool = True,
    low_confidence_threshold: float | None = 0.60,
    allow_cross_core_color_reuse: bool = False,
) -> Any:
    """Create the optional strongest-mutual-edge overlay without saving it."""

    import matplotlib.pyplot as plt

    map_label = _optional_map_label(assignments)
    prepared = _prepare_assignments(
        assignments,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    prepared_regions = _prepare_regions(regions, prepared)
    selected_edges = select_strongest_mutual_edges(
        mutual_edges,
        max_edges_per_core=max_edges_per_core,
        max_edges_total=max_edges_total,
    )
    with _visualization_context():
        figure, axes = plt.subplots(2, 3, figsize=(18.0, 11.5))
        for axis, core_number in zip(axes.ravel(), CORE_ORDER, strict=True):
            cells = prepared.loc[prepared["core_number"] == core_number]
            edges = selected_edges.loc[selected_edges["core_number"] == core_number]
            _draw_overlay_panel(
                axis,
                cells,
                prepared_regions[core_number],
                edges,
                core_number=core_number,
                invert_y=invert_y,
                low_confidence_threshold=low_confidence_threshold,
            )
        title = "Strongest retained mutual attention-routing edges — display subset"
        if map_label is not None:
            title += f"\n{map_label}"
        figure.suptitle(
            title,
            fontsize=16.0,
            fontweight="bold",
            y=0.985,
        )
        figure.subplots_adjust(
            left=0.035,
            right=0.985,
            bottom=0.035,
            top=0.935,
            wspace=0.12,
            hspace=0.17,
        )
        _add_confidence_note(
            figure,
            low_confidence_threshold=low_confidence_threshold,
        )
    return figure


def render_mutual_attention_network_overlay(
    assignments: pd.DataFrame,
    regions: Mapping[str, Any],
    mutual_edges: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    max_edges_per_core: int | None = 1_000,
    max_edges_total: int | None = 4_000,
    invert_y: bool = True,
    low_confidence_threshold: float | None = 0.60,
    allow_cross_core_color_reuse: bool = False,
) -> tuple[Path, Path]:
    """Atomically save the optional overlay as PNG and PDF."""

    import matplotlib.pyplot as plt

    figure = create_mutual_attention_network_overlay_figure(
        assignments,
        regions,
        mutual_edges,
        max_edges_per_core=max_edges_per_core,
        max_edges_total=max_edges_total,
        invert_y=invert_y,
        low_confidence_threshold=low_confidence_threshold,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    output_root = Path(output_dir)
    try:
        paths = _atomic_save_figure_formats(
            figure,
            {
                "png": output_root / f"{OVERLAY_MAP_STEM}.png",
                "pdf": output_root / f"{OVERLAY_MAP_STEM}.pdf",
            },
            dpi=dpi,
        )
        return paths  # type: ignore[return-value]
    finally:
        plt.close(figure)


def _visualization_receipt(
    assignments: pd.DataFrame,
    regions: Mapping[str, Any],
    mutual_edges: pd.DataFrame,
    *,
    output_paths: Sequence[Path],
    include_overlay: bool,
    max_edges_per_core: int | None,
    max_edges_total: int | None,
    invert_y: bool,
    low_confidence_threshold: float | None,
    allow_cross_core_color_reuse: bool,
) -> dict[str, Any]:
    prepared = _prepare_assignments(
        assignments,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    prepared_regions = _prepare_regions(regions, prepared)
    omitted_zero_area_rings = _omitted_zero_area_interior_ring_receipt(
        prepared_regions
    )
    if include_overlay:
        displayed_edges = select_strongest_mutual_edges(
            mutual_edges,
            max_edges_per_core=max_edges_per_core,
            max_edges_total=max_edges_total,
        )
    else:
        displayed_edges = pd.DataFrame(
            columns=["core_number", "cell_i", "cell_j", "Mij", "support_Pij"]
        )
    missing_outputs = [str(path) for path in output_paths if not Path(path).is_file()]
    if missing_outputs:
        raise AttentionNicheVisualizationError(
            f"Visualization outputs are missing after atomic save: {missing_outputs}."
        )

    cells_per_core: dict[str, int] = {}
    niches_per_core: dict[str, int] = {}
    displayed_edges_per_core: dict[str, int] = {}
    scale_bars: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    low_confidence_cells: dict[str, int] = {}
    for core_number in CORE_ORDER:
        cells = prepared.loc[prepared["core_number"] == core_number]
        key = str(core_number)
        cells_per_core[key] = int(len(cells))
        niches_per_core[key] = int(cells["final_niche_id"].dropna().nunique())
        displayed_edges_per_core[key] = int(
            (displayed_edges["core_number"] == core_number).sum()
        )
        aliases[key] = str(cells["core_alias"].iloc[0])
        bounds = _core_bounds(cells, prepared_regions[core_number])
        scale_bars[key] = {
            "present": True,
            "length": float(_nice_scale_bar_length(bounds[1] - bounds[0])),
            "unit": "µm",
        }
        if low_confidence_threshold is None:
            low_confidence_cells[key] = 0
        else:
            low_confidence_cells[key] = int(
                (cells["assignment_confidence"] < low_confidence_threshold).sum()
            )
    return {
        "schema": "attention_niche_visualization_receipt_v2",
        "status": "complete",
        "terminology": "model-defined attention-routing niches",
        "map_label": _optional_map_label(assignments),
        "core_order": list(CORE_ORDER),
        "core_aliases": aliases,
        "grid_shape": [2, 3],
        "panel_titles_include_core_number": True,
        "in_panel_core_labels": [f"Core {core}" for core in CORE_ORDER],
        "equal_physical_aspect": True,
        "invert_imaging_y_axis": bool(invert_y),
        "orientation_basis": (
            "repository CosMx global-image plotting convention; y increases "
            "opposite the displayed image vertical direction"
        ),
        "coordinate_unit": "micrometres",
        "scale_bars": scale_bars,
        "unnecessary_axis_ticks_hidden": True,
        "all_eligible_cells_rendered_without_sampling": True,
        "total_cell_count": int(len(prepared)),
        "cells_per_core": cells_per_core,
        "niches_per_core": niches_per_core,
        "low_confidence_outline_threshold": low_confidence_threshold,
        "low_confidence_outline_explained_on_figure": (
            low_confidence_threshold is not None
        ),
        "low_confidence_cells_per_core": low_confidence_cells,
        "region_geometry_types": ["Polygon", "MultiPolygon"],
        "region_holes_preserved": "all_nondegenerate",
        "nondegenerate_region_holes_preserved": True,
        "zero_area_interior_rings_omitted_from_render": omitted_zero_area_rings,
        "cell_facecolor_matches_niche_fill_color": True,
        "overlay": {
            "included": bool(include_overlay),
            "max_edges_per_core": max_edges_per_core,
            "max_edges_total": max_edges_total,
            "displayed_edges_per_core": displayed_edges_per_core,
            "displayed_edge_count": int(len(displayed_edges)),
            "edge_width_encodes": "M_ij",
            "edge_opacity_encodes": "support_P_ij",
            "node_size_encodes": "mutual_routing_hub_score",
            "width_and_node_size_normalization": "independent_within_core",
        },
        "output_paths": [str(path) for path in output_paths],
        "atomic_file_save": True,
    }


def render_attention_niche_visualizations(
    assignments: pd.DataFrame,
    regions: Mapping[str, Any],
    mutual_edges: pd.DataFrame,
    output_dir: str | Path,
    *,
    dpi: int = 300,
    individual_dpi: int = 450,
    include_overlay: bool = True,
    max_edges_per_core: int | None = 1_000,
    max_edges_total: int | None = 4_000,
    invert_y: bool = True,
    low_confidence_threshold: float | None = 0.60,
    allow_cross_core_color_reuse: bool = False,
) -> AttentionNicheVisualizationArtifacts:
    """Render every required static niche-map artifact.

    This orchestration helper only renders supplied analysis products.  It does
    not alter the assignment, region, or retained-edge tables.
    """

    combined_png, combined_pdf, combined_svg = render_combined_attention_niche_map(
        assignments,
        regions,
        output_dir,
        dpi=dpi,
        invert_y=invert_y,
        low_confidence_threshold=low_confidence_threshold,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    individual_pngs = render_individual_attention_niche_maps(
        assignments,
        regions,
        output_dir,
        dpi=individual_dpi,
        invert_y=invert_y,
        low_confidence_threshold=low_confidence_threshold,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    overlay_png: Path | None = None
    overlay_pdf: Path | None = None
    if include_overlay:
        overlay_png, overlay_pdf = render_mutual_attention_network_overlay(
            assignments,
            regions,
            mutual_edges,
            output_dir,
            dpi=dpi,
            max_edges_per_core=max_edges_per_core,
            max_edges_total=max_edges_total,
            invert_y=invert_y,
            low_confidence_threshold=low_confidence_threshold,
            allow_cross_core_color_reuse=allow_cross_core_color_reuse,
        )
    output_paths = (
        combined_png,
        combined_pdf,
        combined_svg,
        *individual_pngs,
        *((overlay_png, overlay_pdf) if include_overlay else ()),
    )
    receipt = _visualization_receipt(
        assignments,
        regions,
        mutual_edges,
        output_paths=output_paths,
        include_overlay=include_overlay,
        max_edges_per_core=max_edges_per_core,
        max_edges_total=max_edges_total,
        invert_y=invert_y,
        low_confidence_threshold=low_confidence_threshold,
        allow_cross_core_color_reuse=allow_cross_core_color_reuse,
    )
    return AttentionNicheVisualizationArtifacts(
        combined_png=combined_png,
        combined_pdf=combined_pdf,
        combined_svg=combined_svg,
        individual_pngs=individual_pngs,
        overlay_png=overlay_png,
        overlay_pdf=overlay_pdf,
        receipt=receipt,
    )


__all__ = [
    "AttentionNicheVisualizationArtifacts",
    "AttentionNicheVisualizationError",
    "COMBINED_MAP_STEM",
    "CORE_ORDER",
    "NEUTRAL_COLOR",
    "OVERLAY_MAP_STEM",
    "create_combined_attention_niche_figure",
    "create_mutual_attention_network_overlay_figure",
    "render_attention_niche_visualizations",
    "render_combined_attention_niche_map",
    "render_individual_attention_niche_maps",
    "render_mutual_attention_network_overlay",
    "select_strongest_mutual_edges",
]
