"""Static figures for the SO2 four-block hidden-state progression analysis.

This module deliberately owns only validation, partition-agreement summaries,
and PNG rendering.  It does not discover runs, load checkpoints, cluster
embeddings, or write manifests.  Callers must provide cell-aligned labels in
the same row order as the spatial frame.

Each hidden state is clustered independently.  Accordingly, this module gives
every displayed cluster a layer-qualified name and never treats an equal
numeric cluster identifier in two layers as a shared identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SO2_CORE_NUMBERS: tuple[int, ...] = tuple(range(15, 29))
SPATIAL_LAYERS: tuple[str, ...] = ("h0", "h1", "h2", "h3")
PROGRESSION_LAYER_ORDER: tuple[str, ...] = ("h0", "h1", "h2", "h3", "hL")

_DISPLAY_PREFIX = {
    "h0": "H0C",
    "h1": "H1C",
    "h2": "H2C",
    "h3": "H3C",
    "hL": "HLC",
}
_REQUIRED_SPATIAL_COLUMNS = ("core_number", "x_um", "y_um")
_CLUSTER_SUFFIX = re.compile(r"^(?:[A-Za-z][A-Za-z0-9_-]*)?(-?\d+)$")
_PRODUCER = "spatial_benchmark.so2_hidden_layer_progression_figures"


class HiddenLayerProgressionFigureError(ValueError):
    """Raised when a requested figure would be incomplete or misleading."""


@dataclass(frozen=True)
class SpatialFigureSummary:
    """Compact, orchestration-neutral description of a rendered spatial PNG."""

    layer: str
    output_path: Path
    point_count: int
    cluster_labels: tuple[str, ...]
    core_numbers: tuple[int, ...]
    resolution: float
    dpi: int


@dataclass(frozen=True)
class TransitionAgreement:
    """Cell-aligned agreement and conditional overlaps for one layer pair.

    ``split_fractions`` contains ``P(target cluster | source cluster)`` and is
    row-normalized.  ``merge_fractions`` contains
    ``P(source cluster | target cluster)`` and is column-normalized.
    """

    source_layer: str
    target_layer: str
    source_labels: tuple[str, ...]
    target_labels: tuple[str, ...]
    adjusted_rand_index: float
    normalized_mutual_information: float
    counts: np.ndarray = field(repr=False)
    split_fractions: np.ndarray = field(repr=False)
    merge_fractions: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class TransitionFigureSummary:
    """Description of the rendered h0-to-hL transition PNG."""

    output_path: Path
    point_count: int
    layer_order: tuple[str, ...]
    transitions: tuple[TransitionAgreement, ...]
    dpi: int


@dataclass(frozen=True)
class _PreparedLabels:
    layer: str
    values: np.ndarray = field(repr=False)
    categories: tuple[Any, ...]
    codes: np.ndarray = field(repr=False)
    display_labels: tuple[str, ...]


def _validate_layer(layer: str, *, spatial_only: bool = False) -> str:
    allowed = SPATIAL_LAYERS if spatial_only else PROGRESSION_LAYER_ORDER
    if layer not in allowed:
        raise HiddenLayerProgressionFigureError(
            f"Layer must be one of {allowed}; received {layer!r}."
        )
    return layer


def _as_one_dimensional_labels(
    values: Sequence[Any] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise HiddenLayerProgressionFigureError(
            f"{name} labels must be a non-empty one-dimensional array."
        )
    try:
        missing = np.asarray(pd.isna(array), dtype=bool)
    except (TypeError, ValueError) as exc:
        raise HiddenLayerProgressionFigureError(
            f"{name} labels cannot be checked for missing values."
        ) from exc
    if missing.shape != array.shape or bool(missing.any()):
        raise HiddenLayerProgressionFigureError(
            f"{name} labels contain missing values."
        )
    return np.ascontiguousarray(array)


def _category_sort_key(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return (0, int(value), str(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return (0, float(value), str(value))
    text = str(value)
    match = _CLUSTER_SUFFIX.fullmatch(text)
    if match is not None:
        return (1, int(match.group(1)), text)
    return (2, type(value).__name__, text)


def _category_number(value: Any, *, fallback: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise HiddenLayerProgressionFigureError(
            f"Boolean cluster identifiers are invalid; received {value!r}."
        )
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise HiddenLayerProgressionFigureError(
                "Numeric cluster identifiers must be finite integers; "
                f"received {value!r}."
            )
        number = int(numeric)
    else:
        match = _CLUSTER_SUFFIX.fullmatch(str(value))
        number = int(match.group(1)) if match is not None else int(fallback)
    if number < 0:
        raise HiddenLayerProgressionFigureError(
            f"Cluster identifiers must be non-negative; received {value!r}."
        )
    return number


def _prepare_labels(
    values: Sequence[Any] | np.ndarray,
    *,
    layer: str,
) -> _PreparedLabels:
    _validate_layer(layer)
    array = _as_one_dimensional_labels(values, name=layer)
    raw_categories: list[Any] = []
    seen: dict[Any, None] = {}
    for value in array.tolist():
        try:
            if value not in seen:
                seen[value] = None
                raw_categories.append(value)
        except TypeError as exc:
            raise HiddenLayerProgressionFigureError(
                f"{layer} cluster labels must be scalar, hashable values."
            ) from exc
    categories = tuple(sorted(raw_categories, key=_category_sort_key))
    category_to_code = {value: index for index, value in enumerate(categories)}
    codes = np.fromiter(
        (category_to_code[value] for value in array.tolist()),
        dtype=np.int64,
        count=len(array),
    )
    display_labels = tuple(
        f"{_DISPLAY_PREFIX[layer]}{_category_number(value, fallback=index)}"
        for index, value in enumerate(categories)
    )
    if len(set(display_labels)) != len(display_labels):
        raise HiddenLayerProgressionFigureError(
            f"{layer} cluster labels do not map to unique layer-qualified names."
        )
    return _PreparedLabels(
        layer=layer,
        values=array,
        categories=categories,
        codes=np.ascontiguousarray(codes),
        display_labels=display_labels,
    )


def _palette_for_labels(
    palette: Mapping[Any, str],
    *,
    prepared: _PreparedLabels,
) -> dict[str, str]:
    if not isinstance(palette, Mapping) or not palette:
        raise HiddenLayerProgressionFigureError("A non-empty cluster palette is required.")

    from matplotlib.colors import to_hex, to_rgba

    resolved: dict[str, str] = {}
    canonical_colors: list[tuple[float, float, float, float]] = []
    for index, (category, display) in enumerate(
        zip(prepared.categories, prepared.display_labels, strict=True)
    ):
        cluster_number = _category_number(category, fallback=index)
        candidate_keys = (
            category,
            str(category),
            display,
            cluster_number,
            str(cluster_number),
            f"C{cluster_number}",
        )
        color: Any = None
        found = False
        for key in candidate_keys:
            try:
                if key in palette:
                    color = palette[key]
                    found = True
                    break
            except TypeError:
                continue
        if not found:
            raise HiddenLayerProgressionFigureError(
                f"Palette is missing a color for {prepared.layer} cluster {category!r}."
            )
        try:
            rgba = tuple(float(component) for component in to_rgba(color))
        except ValueError as exc:
            raise HiddenLayerProgressionFigureError(
                f"Palette color for {display} is invalid: {color!r}."
            ) from exc
        if not math.isclose(rgba[3], 1.0):
            raise HiddenLayerProgressionFigureError(
                f"Palette color for {display} must be opaque."
            )
        canonical_colors.append(rgba)
        resolved[display] = str(to_hex(rgba, keep_alpha=False)).upper()
    if len(set(canonical_colors)) != len(canonical_colors):
        raise HiddenLayerProgressionFigureError(
            f"{prepared.layer} palette must assign a distinct color to every cluster."
        )
    return resolved


def _validate_png_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise HiddenLayerProgressionFigureError(
            f"Static progression figures must use a .png path, not {path.name!r}."
        )
    return path


def _validate_dpi(dpi: int) -> int:
    try:
        integer_dpi = int(dpi)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HiddenLayerProgressionFigureError(
            "DPI must be a positive integer."
        ) from exc
    if isinstance(dpi, bool) or integer_dpi != dpi or integer_dpi <= 0:
        raise HiddenLayerProgressionFigureError("DPI must be a positive integer.")
    return integer_dpi


def _atomic_save_png(figure: Any, *, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.tmp-",
        suffix=".png",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(
            temporary_path,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": _PRODUCER},
        )
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _nice_scale_bar_length(coordinates: np.ndarray) -> float:
    x_range = float(np.ptp(coordinates[:, 0]))
    if not math.isfinite(x_range) or x_range <= 0.0:
        raise HiddenLayerProgressionFigureError(
            "Every core needs a positive x-coordinate range for its scale bar."
        )
    target = x_range * 0.20
    exponent = 10.0 ** math.floor(math.log10(target))
    choices = (exponent, 2.0 * exponent, 5.0 * exponent, 10.0 * exponent)
    return max(value for value in choices if value <= target)


def _style_spatial_axis(axis: Any, coordinates: np.ndarray) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.set_xlabel("x (µm)")
    axis.set_ylabel("y (µm)")
    axis.set_facecolor("#F8FAFC")
    axis.grid(False)

    length = _nice_scale_bar_length(coordinates)
    x_min, x_max = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
    y_min, y_max = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])
    x_range = float(x_max - x_min)
    y_range = float(y_max - y_min)
    if not math.isfinite(y_range) or y_range <= 0.0:
        raise HiddenLayerProgressionFigureError(
            "Every core needs a positive y-coordinate range."
        )
    x_start = float(x_min + 0.06 * x_range)
    y_value = float(y_max - 0.06 * y_range)
    axis.plot(
        [x_start, x_start + length],
        [y_value, y_value],
        color="#111827",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=4,
    )
    axis.text(
        x_start + length / 2.0,
        y_value - 0.025 * y_range,
        f"{length:g} µm",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#111827",
        zorder=4,
    )


def _validate_spatial_frame(
    frame: pd.DataFrame,
    *,
    label_count: int,
    expected_counts_by_core: Mapping[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(frame, pd.DataFrame):
        raise HiddenLayerProgressionFigureError(
            "Spatial input must be a pandas DataFrame."
        )
    missing_columns = sorted(set(_REQUIRED_SPATIAL_COLUMNS).difference(frame.columns))
    if missing_columns:
        raise HiddenLayerProgressionFigureError(
            f"Spatial frame is missing required columns: {missing_columns}."
        )
    if len(frame) != label_count:
        raise HiddenLayerProgressionFigureError(
            "Spatial rows and cluster labels are not cell-aligned."
        )
    try:
        core_values_float = frame["core_number"].to_numpy(
            dtype=np.float64, copy=True
        )
    except (TypeError, ValueError) as exc:
        raise HiddenLayerProgressionFigureError(
            "Core numbers must be finite integers."
        ) from exc
    if not np.isfinite(core_values_float).all() or not np.equal(
        core_values_float, np.floor(core_values_float)
    ).all():
        raise HiddenLayerProgressionFigureError("Core numbers must be finite integers.")
    core_values = core_values_float.astype(np.int64)
    observed_cores = tuple(sorted(int(value) for value in np.unique(core_values)))
    if observed_cores != SO2_CORE_NUMBERS:
        raise HiddenLayerProgressionFigureError(
            f"Spatial frame must contain exactly SO2 cores 15--28; found {observed_cores}."
        )
    try:
        coordinates = frame[["x_um", "y_um"]].to_numpy(
            dtype=np.float64, copy=True
        )
    except (TypeError, ValueError) as exc:
        raise HiddenLayerProgressionFigureError(
            "Spatial coordinates must be finite numeric values."
        ) from exc
    if coordinates.shape != (len(frame), 2) or not np.isfinite(coordinates).all():
        raise HiddenLayerProgressionFigureError(
            "Spatial coordinates must be a finite two-column array."
        )
    if expected_counts_by_core is not None:
        if set(int(core) for core in expected_counts_by_core) != set(SO2_CORE_NUMBERS):
            raise HiddenLayerProgressionFigureError(
                "Expected cell counts must cover exactly SO2 cores 15--28."
            )
        for core_number in SO2_CORE_NUMBERS:
            observed = int(np.count_nonzero(core_values == core_number))
            expected = int(expected_counts_by_core[core_number])
            if observed != expected:
                raise HiddenLayerProgressionFigureError(
                    f"SO2 core {core_number} has {observed} rows; expected {expected}."
                )
    return core_values, coordinates


def _legend_handles(palette: Mapping[str, str]) -> list[Any]:
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in palette.items()
    ]


def render_layer_spatial_png(
    frame: pd.DataFrame,
    labels: Sequence[Any] | np.ndarray,
    palette: Mapping[Any, str],
    output_path: str | Path,
    *,
    layer: str,
    resolution: float = 1.0,
    dpi: int = 300,
    expected_counts_by_core: Mapping[int, int] | None = None,
) -> SpatialFigureSummary:
    """Render one 3x5 SO2 tissue-coordinate cluster map as a PNG.

    The input label array is aligned by position to ``frame``.  Raw labels may
    be integer Leiden IDs or strings such as ``C0``; displayed labels are
    always qualified by ``layer`` (for example, ``H1C0``).
    """

    layer = _validate_layer(layer, spatial_only=True)
    path = _validate_png_path(output_path)
    dpi = _validate_dpi(dpi)
    if not math.isfinite(float(resolution)) or float(resolution) <= 0.0:
        raise HiddenLayerProgressionFigureError(
            "Leiden resolution must be finite and positive."
        )
    prepared = _prepare_labels(labels, layer=layer)
    display_palette = _palette_for_labels(palette, prepared=prepared)
    core_values, coordinates = _validate_spatial_frame(
        frame,
        label_count=len(prepared.codes),
        expected_counts_by_core=expected_counts_by_core,
    )
    display_by_cell = np.asarray(prepared.display_labels, dtype=object)[prepared.codes]

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 5, figsize=(25.0, 15.0))
    try:
        for axis, core_number in zip(
            axes.ravel()[: len(SO2_CORE_NUMBERS)],
            SO2_CORE_NUMBERS,
            strict=True,
        ):
            selected = core_values == core_number
            core_coordinates = coordinates[selected]
            if core_coordinates.size == 0:
                raise HiddenLayerProgressionFigureError(
                    f"SO2 core {core_number} contains no spatial rows."
                )
            core_labels = display_by_cell[selected]
            colors = [display_palette[str(label)] for label in core_labels]
            axis.scatter(
                core_coordinates[:, 0],
                core_coordinates[:, 1],
                s=0.52,
                c=colors,
                marker="o",
                linewidths=0,
                edgecolors="none",
                alpha=0.92,
                rasterized=True,
            )
            axis.set_title(
                f"SO2 Core {core_number}\n$n$ = {len(core_coordinates):,}",
                fontsize=12,
                weight="bold",
                pad=7,
            )
            axis.text(
                0.025,
                0.97,
                f"CORE {core_number}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                weight="bold",
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#CBD5E1",
                    "alpha": 0.92,
                },
                zorder=6,
            )
            _style_spatial_axis(axis, core_coordinates)

        legend_axis = axes.ravel()[-1]
        legend_axis.axis("off")
        legend_axis.legend(
            handles=_legend_handles(display_palette),
            loc="center",
            frameon=False,
            ncol=2 if len(display_palette) > 14 else 1,
            title=f"Joint model-derived\n{layer} cluster",
            fontsize=8,
            title_fontsize=10,
            borderaxespad=0.0,
        )
        if layer == "h0":
            title = (
                "SO2 intrinsic h0 Leiden clusters — node encoder before graph blocks\n"
                f"joint 14-core clustering, resolution {float(resolution):g}"
            )
        else:
            block_number = int(layer[1:])
            title = (
                f"SO2 contextual {layer} Leiden clusters — after graph block "
                f"{block_number}\njoint 14-core clustering, resolution "
                f"{float(resolution):g}"
            )
        figure.suptitle(
            title,
            fontsize=18,
            weight="bold",
            y=0.995,
        )
        figure.text(
            0.5,
            0.006,
            "Layer-specific model-derived clusters; cluster IDs and colors are not "
            "identities across layers.",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#475569",
        )
        figure.subplots_adjust(
            left=0.045,
            right=0.985,
            bottom=0.045,
            top=0.925,
            wspace=0.27,
            hspace=0.34,
        )
        _atomic_save_png(figure, output_path=path, dpi=dpi)
    finally:
        plt.close(figure)

    return SpatialFigureSummary(
        layer=layer,
        output_path=path,
        point_count=len(frame),
        cluster_labels=prepared.display_labels,
        core_numbers=SO2_CORE_NUMBERS,
        resolution=float(resolution),
        dpi=dpi,
    )


def _combination_two_sum(values: np.ndarray) -> float:
    values_float = values.astype(np.float64, copy=False)
    return float(np.sum(values_float * (values_float - 1.0) / 2.0))


def _adjusted_rand_index(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total < 2:
        return 1.0
    index = _combination_two_sum(counts)
    rows = _combination_two_sum(counts.sum(axis=1))
    columns = _combination_two_sum(counts.sum(axis=0))
    total_pairs = float(total * (total - 1) / 2)
    expected = rows * columns / total_pairs
    maximum = 0.5 * (rows + columns)
    denominator = maximum - expected
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return 1.0
    value = (index - expected) / denominator
    return float(max(-1.0, min(1.0, value)))


def _normalized_mutual_information(counts: np.ndarray) -> float:
    total = float(counts.sum())
    probabilities = counts.astype(np.float64, copy=False) / total
    row_probabilities = probabilities.sum(axis=1)
    column_probabilities = probabilities.sum(axis=0)
    row_entropy = -float(
        np.sum(
            row_probabilities[row_probabilities > 0.0]
            * np.log(row_probabilities[row_probabilities > 0.0])
        )
    )
    column_entropy = -float(
        np.sum(
            column_probabilities[column_probabilities > 0.0]
            * np.log(column_probabilities[column_probabilities > 0.0])
        )
    )
    nonzero_rows, nonzero_columns = np.nonzero(counts)
    joint = probabilities[nonzero_rows, nonzero_columns]
    mutual_information = float(
        np.sum(
            joint
            * np.log(
                joint
                / (
                    row_probabilities[nonzero_rows]
                    * column_probabilities[nonzero_columns]
                )
            )
        )
    )
    denominator = 0.5 * (row_entropy + column_entropy)
    if math.isclose(denominator, 0.0, abs_tol=1e-15):
        return 1.0
    return float(max(0.0, min(1.0, mutual_information / denominator)))


def compute_adjacent_partition_transitions(
    labels_by_layer: Mapping[str, Sequence[Any] | np.ndarray],
    *,
    layer_order: Sequence[str] = PROGRESSION_LAYER_ORDER,
) -> tuple[TransitionAgreement, ...]:
    """Compute cell-aligned split/merge matrices, ARI, and arithmetic NMI."""

    order = tuple(layer_order)
    if order != PROGRESSION_LAYER_ORDER:
        raise HiddenLayerProgressionFigureError(
            f"Progression order must be exactly {PROGRESSION_LAYER_ORDER}."
        )
    missing_layers = [layer for layer in order if layer not in labels_by_layer]
    if missing_layers:
        raise HiddenLayerProgressionFigureError(
            f"Progression labels are missing layers: {missing_layers}."
        )
    prepared = {
        layer: _prepare_labels(labels_by_layer[layer], layer=layer) for layer in order
    }
    lengths = {len(value.codes) for value in prepared.values()}
    if len(lengths) != 1:
        raise HiddenLayerProgressionFigureError(
            "All progression label arrays must have the same cell-aligned length."
        )

    transitions: list[TransitionAgreement] = []
    for source_layer, target_layer in zip(order[:-1], order[1:], strict=True):
        source = prepared[source_layer]
        target = prepared[target_layer]
        counts = np.zeros(
            (len(source.categories), len(target.categories)), dtype=np.int64
        )
        np.add.at(counts, (source.codes, target.codes), 1)
        row_totals = counts.sum(axis=1, keepdims=True)
        column_totals = counts.sum(axis=0, keepdims=True)
        split_fractions = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts, dtype=np.float64),
            where=row_totals > 0,
        )
        merge_fractions = np.divide(
            counts,
            column_totals,
            out=np.zeros_like(counts, dtype=np.float64),
            where=column_totals > 0,
        )
        transitions.append(
            TransitionAgreement(
                source_layer=source_layer,
                target_layer=target_layer,
                source_labels=source.display_labels,
                target_labels=target.display_labels,
                adjusted_rand_index=_adjusted_rand_index(counts),
                normalized_mutual_information=_normalized_mutual_information(counts),
                counts=np.ascontiguousarray(counts),
                split_fractions=np.ascontiguousarray(split_fractions),
                merge_fractions=np.ascontiguousarray(merge_fractions),
            )
        )
    return tuple(transitions)


def _configure_heatmap_axis(
    axis: Any,
    values: np.ndarray,
    *,
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    annotation_threshold: float,
) -> Any:
    image = axis.imshow(
        values,
        origin="upper",
        interpolation="nearest",
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xticks(np.arange(len(target_labels)))
    axis.set_xticklabels(target_labels, rotation=90, fontsize=6)
    axis.set_yticks(np.arange(len(source_labels)))
    axis.set_yticklabels(source_labels, fontsize=6)
    axis.set_xlabel("Target-layer cluster", fontsize=8)
    axis.set_ylabel("Source-layer cluster", fontsize=8)
    axis.set_xticks(np.arange(len(target_labels) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(source_labels) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color="white", linewidth=0.25, alpha=0.45)
    axis.tick_params(which="minor", bottom=False, left=False)
    if annotation_threshold <= 1.0:
        rows, columns = np.nonzero(values >= annotation_threshold)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            value = float(values[row, column])
            axis.text(
                column,
                row,
                f"{100.0 * value:.0f}%",
                ha="center",
                va="center",
                fontsize=5,
                color="white" if value >= 0.52 else "#111827",
            )
    return image


def render_transition_heatmap_png(
    labels_by_layer: Mapping[str, Sequence[Any] | np.ndarray],
    output_path: str | Path,
    *,
    dpi: int = 300,
    annotation_threshold: float = 0.15,
) -> TransitionFigureSummary:
    """Render split and merge heatmaps for h0 -> h1 -> h2 -> h3 -> hL.

    The top row is source-normalized and therefore emphasizes splits.  The
    bottom row is target-normalized and therefore emphasizes merges.  Cluster
    names are layer-qualified, and no palette correspondence is used or
    implied across independently fitted partitions.
    """

    path = _validate_png_path(output_path)
    dpi = _validate_dpi(dpi)
    if not math.isfinite(float(annotation_threshold)) or not (
        0.0 <= float(annotation_threshold) <= 1.0
    ):
        raise HiddenLayerProgressionFigureError(
            "Heatmap annotation threshold must be between zero and one."
        )
    transitions = compute_adjacent_partition_transitions(labels_by_layer)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(28.0, 12.0), squeeze=False)
    image = None
    try:
        for column, transition in enumerate(transitions):
            split_axis = axes[0, column]
            merge_axis = axes[1, column]
            image = _configure_heatmap_axis(
                split_axis,
                transition.split_fractions,
                source_labels=transition.source_labels,
                target_labels=transition.target_labels,
                annotation_threshold=float(annotation_threshold),
            )
            _configure_heatmap_axis(
                merge_axis,
                transition.merge_fractions,
                source_labels=transition.source_labels,
                target_labels=transition.target_labels,
                annotation_threshold=float(annotation_threshold),
            )
            split_axis.set_title(
                f"{transition.source_layer} → {transition.target_layer}\n"
                f"ARI {transition.adjusted_rand_index:.3f} · "
                f"NMI {transition.normalized_mutual_information:.3f}",
                fontsize=11,
                weight="bold",
            )
            merge_axis.set_title(
                f"{transition.source_layer} → {transition.target_layer}",
                fontsize=10,
                weight="bold",
            )

        figure.suptitle(
            "SO2 four-block hidden-state Leiden partition progression\n"
            "cell-aligned overlaps between independently clustered layers",
            fontsize=18,
            weight="bold",
            y=0.985,
        )
        figure.text(
            0.012,
            0.70,
            "SPLITS\nP(target | source)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="#334155",
        )
        figure.text(
            0.012,
            0.28,
            "MERGES\nP(source | target)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="#334155",
        )
        figure.text(
            0.5,
            0.012,
            "Cluster numbers and colors are layer-specific and do not imply a "
            "shared identity or biological lineage.",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#475569",
        )
        figure.subplots_adjust(
            left=0.065,
            right=0.94,
            bottom=0.10,
            top=0.88,
            wspace=0.34,
            hspace=0.38,
        )
        if image is None:  # pragma: no cover - fixed five-layer order guards this
            raise HiddenLayerProgressionFigureError(
                "No layer transitions were supplied."
            )
        colorbar_axis = figure.add_axes((0.955, 0.18, 0.012, 0.62))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Conditional share of cells", fontsize=9)
        _atomic_save_png(figure, output_path=path, dpi=dpi)
    finally:
        plt.close(figure)

    return TransitionFigureSummary(
        output_path=path,
        point_count=int(transitions[0].counts.sum()),
        layer_order=PROGRESSION_LAYER_ORDER,
        transitions=transitions,
        dpi=dpi,
    )


__all__ = [
    "HiddenLayerProgressionFigureError",
    "PROGRESSION_LAYER_ORDER",
    "SO2_CORE_NUMBERS",
    "SPATIAL_LAYERS",
    "SpatialFigureSummary",
    "TransitionAgreement",
    "TransitionFigureSummary",
    "compute_adjacent_partition_transitions",
    "render_layer_spatial_png",
    "render_transition_heatmap_png",
]
