"""Offline interactive viewer for SO1 classical raw-expression clusters.

The renderer is visualization-only.  It first verifies the completed SO1
clustering bundle, then reads only numeric core labels, tissue coordinates, and
the independent ``S1E`` cluster labels.  Counts, PCA scores, neighbor edges,
model data, stable cell identifiers, and clinical fields never enter the
browser payload.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import hashlib
import hmac
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from .fingerprints import sha256_file
from .paths import ProjectPaths
from .relative_qkv_embedding_clustering import (
    _atomic_write_bytes,
    _atomic_write_json,
    _atomic_write_text,
    _file_manifest,
    _file_record,
    _read_json,
    _receipt_with_self_hash,
    _verify_self_hash,
)
from .so2_hl_interactive import _interactive_css as _shared_interactive_css
from .so2_raw_expression_interactive import (
    interactive_javascript as _shared_expression_javascript,
)


INTERACTIVE_SCHEMA = "so1_14core_raw_expression_interactive_v1"
SOURCE_ANALYSIS_ID = "rawexpr_pca50_cosinek30_leiden_r1p0_s20260825"
DEFAULT_SOURCE_REPORT = "so1_14core_raw_expression_clustering"
DEFAULT_OUTPUT_REPORT = "so1_14core_raw_expression_interactive"
HTML_FILENAME = (
    "so1_raw_expression_leiden_resolution_1p0_spatial_14cores_interactive.html"
)
ZIP_FILENAME = (
    "so1_raw_expression_leiden_resolution_1p0_spatial_14cores_interactive.zip"
)
SOURCE_TABLE_RELATIVE_PATH = Path("tables/cell_expression_clusters.parquet")
SOURCE_PALETTE_RELATIVE_PATH = Path("clustering/expression_palette.json")
SOURCE_FIGURE_RELATIVE_PATH = Path(
    "figures/so1_raw_expression_leiden_resolution_1p0_spatial_14cores.png"
)
SO1_CORE_NUMBERS = tuple(range(1, 15))
EXPECTED_CELL_COUNTS_BY_CORE = {
    1: 8_924,
    2: 7_450,
    3: 12_190,
    4: 14_657,
    5: 11_399,
    6: 18_212,
    7: 10_722,
    8: 4_972,
    9: 17_223,
    10: 14_756,
    11: 18_145,
    12: 7_816,
    13: 5_345,
    14: 9_785,
}
EXPECTED_TOTAL_CELLS = 161_596
DEFAULT_LEIDEN_RESOLUTION = 1.0
LABEL_PREFIX = "S1E"


class SO1RawExpressionInteractiveError(ValueError):
    """Raised when the SO1 interactive-viewer contract is violated."""


@dataclass(frozen=True, slots=True)
class LowCountClusterWarning:
    """Validated source-derived low-count composition for one cluster."""

    cluster: str
    below_threshold_cells: int
    cluster_cells: int
    fraction_below_threshold: float


@dataclass(frozen=True, slots=True)
class LowCountDepthWarning:
    """Validated source-derived low-count audit used for display only."""

    threshold_transcripts: int
    cohort_below_threshold_cells: int
    flag_rule: str
    flagged_clusters: tuple[LowCountClusterWarning, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cluster_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"S1E([0-9]+)", label)
    if match is None:
        raise SO1RawExpressionInteractiveError(
            f"SO1 expression cluster label must have the form S1E<number>: {label!r}"
        )
    return int(match.group(1)), label


def _validate_hex_color(value: object, *, label: str) -> str:
    color = str(value)
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None:
        raise SO1RawExpressionInteractiveError(
            f"Palette color for {label} must be a six-digit hexadecimal color."
        )
    return color


def validate_interactive_frame(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> pd.DataFrame:
    """Validate exact SO1 coverage and a dynamic contiguous ``S1E`` label set."""

    required = {"core_number", "x_um", "y_um", "expression_cluster"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SO1RawExpressionInteractiveError(
            f"Interactive source table lacks required columns: {missing}"
        )
    if frame.empty:
        raise SO1RawExpressionInteractiveError("Interactive source table is empty.")
    if frame.loc[:, sorted(required)].isna().any().any():
        raise SO1RawExpressionInteractiveError(
            "Interactive source fields contain missing values."
        )

    validated = frame.copy(deep=False)
    observed_core_order = tuple(
        int(value) for value in validated["core_number"].drop_duplicates().tolist()
    )
    if observed_core_order != SO1_CORE_NUMBERS:
        raise SO1RawExpressionInteractiveError(
            "Interactive map requires all 14 SO1 cores in the locked order "
            f"{SO1_CORE_NUMBERS}; observed {observed_core_order}."
        )
    core_values = validated["core_number"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(core_values).all() or not np.array_equal(
        core_values, core_values.astype(np.int64)
    ):
        raise SO1RawExpressionInteractiveError(
            "Core numbers must be finite integers."
        )
    for coordinate in ("x_um", "y_um"):
        values = validated[coordinate].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise SO1RawExpressionInteractiveError(
                f"Interactive coordinates contain non-finite {coordinate} values."
            )

    labels = validated["expression_cluster"].astype(str)
    observed_labels = sorted(set(labels.tolist()), key=_cluster_sort_key)
    expected_labels = [f"S1E{index}" for index in range(len(observed_labels))]
    if observed_labels != expected_labels:
        raise SO1RawExpressionInteractiveError(
            "SO1 expression cluster labels must be contiguous from S1E0."
        )
    palette_keys = sorted((str(key) for key in palette), key=_cluster_sort_key)
    if palette_keys != observed_labels:
        missing_palette = sorted(set(observed_labels).difference(palette_keys))
        extra_palette = sorted(set(palette_keys).difference(observed_labels))
        raise SO1RawExpressionInteractiveError(
            "Palette must cover the complete SO1 expression cluster set "
            f"(missing={missing_palette}, extra={extra_palette})."
        )
    colors = [_validate_hex_color(palette[label], label=label) for label in palette_keys]
    if len({color.lower() for color in colors}) != len(colors):
        raise SO1RawExpressionInteractiveError(
            "SO1 expression cluster palette colors must be unique."
        )

    if "global_cell_index" in validated:
        indices = validated["global_cell_index"]
        if indices.isna().any() or not indices.is_unique:
            raise SO1RawExpressionInteractiveError(
                "Global source row indices must be unique."
            )
    if "cell_key" in validated:
        keys = validated["cell_key"]
        if keys.isna().any() or not keys.is_unique:
            raise SO1RawExpressionInteractiveError(
                "Stable source keys must be unique."
            )
    return validated


def parse_low_count_depth_warning(
    source_manifest: Mapping[str, Any],
    *,
    valid_clusters: Sequence[str],
) -> LowCountDepthWarning:
    """Parse structured numeric QC fields without trusting arbitrary HTML text."""

    value = source_manifest.get("low_count_depth_warning")
    if not isinstance(value, Mapping):
        raise SO1RawExpressionInteractiveError(
            "Source manifest lacks the structured low-count-depth warning audit."
        )
    threshold = value.get("threshold_transcripts")
    if isinstance(threshold, bool):
        raise SO1RawExpressionInteractiveError(
            "Low-count transcript threshold must be a positive integer."
        )
    try:
        threshold_number = int(threshold)
    except (TypeError, ValueError) as exc:
        raise SO1RawExpressionInteractiveError(
            "Low-count transcript threshold must be a positive integer."
        ) from exc
    if threshold_number <= 0 or float(threshold_number) != float(threshold):
        raise SO1RawExpressionInteractiveError(
            "Low-count transcript threshold must be a positive integer."
        )
    entries = value.get("flagged_clusters")
    if not isinstance(entries, list):
        raise SO1RawExpressionInteractiveError(
            "Low-count warning flagged_clusters must be a list."
        )
    cohort_below = value.get("cohort_below_threshold_cells")
    if isinstance(cohort_below, bool):
        raise SO1RawExpressionInteractiveError(
            "Cohort low-count cell total must be a nonnegative integer."
        )
    try:
        cohort_below_number = int(cohort_below)
    except (TypeError, ValueError) as exc:
        raise SO1RawExpressionInteractiveError(
            "Cohort low-count cell total must be a nonnegative integer."
        ) from exc
    if (
        cohort_below_number < 0
        or cohort_below_number > EXPECTED_TOTAL_CELLS
        or float(cohort_below_number) != float(cohort_below)
    ):
        raise SO1RawExpressionInteractiveError(
            "Cohort low-count cell total must be a nonnegative integer."
        )
    flag_rule = str(value.get("flag_rule", ""))
    expected_rule = (
        "cluster_contains_at_least_50pct_of_all_cohort_cells_below_threshold"
    )
    if flag_rule != expected_rule:
        raise SO1RawExpressionInteractiveError(
            "Low-count warning uses an unknown flag rule."
        )
    allowed = set(valid_clusters)
    warnings: list[LowCountClusterWarning] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SO1RawExpressionInteractiveError(
                "Each low-count cluster warning must be a mapping."
            )
        cluster = str(entry.get("cluster", ""))
        _cluster_sort_key(cluster)
        if cluster not in allowed or cluster in seen:
            raise SO1RawExpressionInteractiveError(
                f"Low-count warning references an invalid or duplicate cluster: {cluster!r}."
            )
        seen.add(cluster)
        try:
            below = int(entry["below_threshold_cells"])
            cluster_cells = int(entry["cluster_cells"])
            fraction = float(entry["fraction_below_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SO1RawExpressionInteractiveError(
                f"Low-count warning values are invalid for {cluster}."
            ) from exc
        if (
            below < 0
            or cluster_cells <= 0
            or below > cluster_cells
            or not math.isfinite(fraction)
            or fraction < 0.0
            or fraction > 1.0
            or not math.isclose(
                fraction,
                below / cluster_cells,
                rel_tol=0.0,
                abs_tol=5e-7,
            )
        ):
            raise SO1RawExpressionInteractiveError(
                f"Low-count warning counts/fraction are inconsistent for {cluster}."
            )
        if cohort_below_number <= 0 or below / cohort_below_number < 0.5:
            raise SO1RawExpressionInteractiveError(
                f"Low-count warning does not satisfy its flag rule for {cluster}."
            )
        warnings.append(
            LowCountClusterWarning(
                cluster=cluster,
                below_threshold_cells=below,
                cluster_cells=cluster_cells,
                fraction_below_threshold=fraction,
            )
        )
    warnings.sort(key=lambda item: _cluster_sort_key(item.cluster))
    if sum(item.below_threshold_cells for item in warnings) > cohort_below_number:
        raise SO1RawExpressionInteractiveError(
            "Flagged low-count cluster totals exceed the cohort low-count total."
        )
    return LowCountDepthWarning(
        threshold_transcripts=threshold_number,
        cohort_below_threshold_cells=cohort_below_number,
        flag_rule=flag_rule,
        flagged_clusters=tuple(warnings),
    )


def _encode_array(value: np.ndarray, *, dtype: str | np.dtype[Any]) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return base64.b64encode(memoryview(array).cast("B")).decode("ascii")


def build_interactive_payload(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> dict[str, Any]:
    """Encode only core number, coordinates, and SO1 expression-cluster codes."""

    validated = validate_interactive_frame(frame, palette)
    clusters = sorted(
        set(validated["expression_cluster"].astype(str)), key=_cluster_sort_key
    )
    if len(clusters) > np.iinfo(np.uint8).max + 1:
        raise SO1RawExpressionInteractiveError(
            "Too many SO1 expression clusters for Uint8 browser codes."
        )
    canonical_palette = {label: str(palette[label]) for label in clusters}
    cluster_to_code = {label: index for index, label in enumerate(clusters)}

    core_records: list[dict[str, Any]] = []
    observed_points = 0
    for core_number in SO1_CORE_NUMBERS:
        core = validated.loc[validated["core_number"] == core_number]
        if core.empty:
            raise SO1RawExpressionInteractiveError(
                f"SO1 core {core_number} has no interactive points."
            )
        x = np.ascontiguousarray(core["x_um"].to_numpy(dtype="<f8", copy=True))
        y = np.ascontiguousarray(core["y_um"].to_numpy(dtype="<f8", copy=True))
        codes = np.ascontiguousarray(
            core["expression_cluster"]
            .astype(str)
            .map(cluster_to_code)
            .to_numpy(dtype=np.uint8, copy=True)
        )
        if not (len(x) == len(y) == len(codes) == len(core)):
            raise SO1RawExpressionInteractiveError(
                f"Coordinate/label ordering mismatch for SO1 core {core_number}."
            )
        x_b64 = _encode_array(x, dtype="<f8")
        y_b64 = _encode_array(y, dtype="<f8")
        codes_b64 = _encode_array(codes, dtype=np.uint8)
        if not (
            np.array_equal(
                np.frombuffer(base64.b64decode(x_b64, validate=True), dtype="<f8"),
                x,
            )
            and np.array_equal(
                np.frombuffer(base64.b64decode(y_b64, validate=True), dtype="<f8"),
                y,
            )
            and np.array_equal(
                np.frombuffer(
                    base64.b64decode(codes_b64, validate=True), dtype=np.uint8
                ),
                codes,
            )
        ):
            raise SO1RawExpressionInteractiveError(
                f"Browser payload failed round-trip validation for SO1 core {core_number}."
            )
        counts = np.bincount(codes, minlength=len(clusters)).astype(np.int64)
        core_records.append(
            {
                "core_number": int(core_number),
                "point_count": int(len(core)),
                "bounds": {
                    "x_min": float(x.min()),
                    "x_max": float(x.max()),
                    "y_min": float(y.min()),
                    "y_max": float(y.max()),
                },
                "cluster_counts": counts.tolist(),
                "x_b64": x_b64,
                "y_b64": y_b64,
                "cluster_codes_b64": codes_b64,
            }
        )
        observed_points += len(core)
    if observed_points != len(validated):
        raise SO1RawExpressionInteractiveError(
            "Not every source row was encoded into the interactive payload."
        )
    return {
        "schema": INTERACTIVE_SCHEMA,
        "core_order": list(SO1_CORE_NUMBERS),
        "clusters": clusters,
        "palette": canonical_palette,
        "point_count": int(len(validated)),
        "coordinate_units": "micrometres",
        "coordinate_orientation": "low_y_at_top",
        "cores": core_records,
    }


def interactive_javascript() -> str:
    """Return the dependency-free viewer with SO1 titles/export naming."""

    return (
        _shared_expression_javascript()
        .replace("SO2 Core", "SO1 Core")
        .replace("so2_raw_expression_clusters_", "so1_raw_expression_clusters_")
        .replace("BAGM_SO2_VIEWER", "BAGM_SO1_VIEWER")
    )


def _cluster_buttons(frame: pd.DataFrame, palette: Mapping[str, str]) -> str:
    counts = frame["expression_cluster"].astype(str).value_counts()
    labels = sorted(palette, key=_cluster_sort_key)
    return "\n".join(
        (
            f'<button type="button" class="cluster-chip" data-cluster="{escape(label)}" '
            f'aria-pressed="false" style="--cluster-color:{escape(str(palette[label]))}">'
            '<span class="swatch" aria-hidden="true"></span>'
            f'<span>{escape(label)}</span><span class="cluster-count">'
            f"{int(counts[label]):,}</span></button>"
        )
        for label in labels
    )


def _core_panels(frame: pd.DataFrame) -> str:
    counts = frame.groupby("core_number", sort=False).size()
    panels: list[str] = []
    for core_number in SO1_CORE_NUMBERS:
        panels.append(
            f'<article class="core-panel" data-panel-core="{core_number}">'
            f"<h2>SO1 Core {core_number}</h2>"
            f'<div class="point-count">{int(counts.loc[core_number]):,} cells</div>'
            '<div class="canvas-wrap">'
            f'<canvas data-core="{core_number}" role="img" '
            f'aria-label="Interactive expression cluster map for SO1 Core {core_number}"></canvas>'
            '<div class="hover-card" hidden></div>'
            "</div></article>"
        )
    panels.append(
        '<aside class="legend-panel"><div><strong>Shared joint SO1 clusters</strong><br>'
        "Cluster colors and IDs are identical across all 14 SO1 panels.<br>"
        "SO1 S1E labels do not map to SO2 E labels.</div></aside>"
    )
    return "\n".join(panels)


def _low_count_warning_html(warning: LowCountDepthWarning) -> str:
    if warning.flagged_clusters:
        details = "; ".join(
            (
                f"{escape(item.cluster)}: {item.below_threshold_cells:,} of "
                f"{item.cluster_cells:,} cells "
                f"({100.0 * item.fraction_below_threshold:.1f}%)"
            )
            for item in warning.flagged_clusters
        )
        audit_text = (
            f"Of {warning.cohort_below_threshold_cells:,} source cells below "
            f"{warning.threshold_transcripts:,} transcripts, the completed audit "
            f"flagged clusters containing at least half of that low-count set — {details}. "
        )
    else:
        audit_text = (
            f"The completed source audit found "
            f"{warning.cohort_below_threshold_cells:,} cells below "
            f"{warning.threshold_transcripts:,} transcripts, but no cluster "
            "contained at least half of that low-count set. "
        )
    return (
        '<p class="privacy-note"><strong>QC and sharing notice.</strong> '
        f"{audit_text}Count depth can be associated with expression-cluster separation. "
        f"All {EXPECTED_TOTAL_CELLS:,} source cells are displayed, including low-depth cells and "
        "cells that failed vendor QC; vendor-QC fields are not in the browser payload. "
        "These clusters require marker/pathology validation. This offline file "
        "contains exact cell-level tissue coordinates; share it only with "
        "authorized collaborators.</p>"
    )


def _render_html(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
    *,
    low_count_warning: LowCountDepthWarning,
) -> str:
    payload = build_interactive_payload(frame, palette)
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>SO1 14-core raw-expression clusters</title>
<style>{_shared_interactive_css()}</style>
</head>
<body>
<header>
  <h1>SO1 14-core raw-expression clusters</h1>
  <p>Joint classical raw-expression Leiden clustering at resolution 1.0. SO1 labels use the independent S1E namespace and do not correspond to SO2 E labels.</p>
  <p>Click a cluster to keep it bright while fading the others. Scroll to zoom, drag to pan, double-click a panel to reset it, and hover for core, cluster, and rounded tissue coordinates.</p>
</header>
<main class="workspace">
  <aside class="controls" aria-label="Cluster controls">
    <h2>SO1 expression clusters</h2>
    <p class="instructions">These are expression-derived groups, not validated cell types. Counts are across all 14 SO1 cores.</p>
    <div class="cluster-list">{_cluster_buttons(frame, palette)}</div>
    <div class="action-row">
      <button type="button" class="action" id="show-all">Show all clusters</button>
      <button type="button" class="action" id="reset-views">Reset all spatial views</button>
      <button type="button" class="action primary" id="export-png">Export current view as PNG</button>
    </div>
    <p id="selection-status" aria-live="polite">Showing all expression clusters.</p>
    {_low_count_warning_html(low_count_warning)}
  </aside>
  <section class="core-grid" aria-label="SO1 spatial core maps">{_core_panels(frame)}</section>
</main>
<script id="bagm-data" type="application/json">{payload_json}</script>
<script>{interactive_javascript()}</script>
</body>
</html>
"""


def _validate_rendered_html(html: str, *, frame: pd.DataFrame) -> None:
    prohibited = (
        "cell_key",
        "cell_index",
        "global_cell_index",
        "core_alias",
        "expression_cluster_number",
        "expression_counts",
        "raw_library_size",
        "detected_genes",
        "pca_score",
        "knn_edge",
        "embedding_values",
        "/workspace",
        "http://",
        "https://",
        "fetch(",
        "xmlhttprequest",
        "url(http",
    )
    lowered = html.lower()
    found = [token for token in prohibited if token.lower() in lowered]
    if found:
        raise SO1RawExpressionInteractiveError(
            f"Shareable HTML contains prohibited source or network fields: {found}"
        )
    if "cell_key" in frame and len(frame):
        first_key = str(frame["cell_key"].iloc[0])
        if first_key and first_key in html:
            raise SO1RawExpressionInteractiveError(
                "Shareable HTML contains a stable source key."
            )
    if "connect-src 'none'" not in html:
        raise SO1RawExpressionInteractiveError(
            "Shareable HTML lacks the offline network CSP."
        )
    if len(re.findall(r"<canvas\s+[^>]*data-core=", html)) != len(SO1_CORE_NUMBERS):
        raise SO1RawExpressionInteractiveError(
            "Shareable HTML lacks exactly 14 SO1 core canvases."
        )
    if re.search(
        r"<(?:script|img)\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=",
        html,
        flags=re.IGNORECASE,
    ) is not None:
        raise SO1RawExpressionInteractiveError(
            "Shareable HTML contains an external DOM asset."
        )


def _warning_record(warning: LowCountDepthWarning) -> dict[str, Any]:
    return {
        "threshold_transcripts": warning.threshold_transcripts,
        "cohort_below_threshold_cells": warning.cohort_below_threshold_cells,
        "flag_rule": warning.flag_rule,
        "flagged_clusters": [
            {
                "cluster": item.cluster,
                "below_threshold_cells": item.below_threshold_cells,
                "cluster_cells": item.cluster_cells,
                "fraction_below_threshold": item.fraction_below_threshold,
            }
            for item in warning.flagged_clusters
        ],
        "source_derived": True,
    }


def _browser_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Return a canonical digest for the exact browser-visible payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_interactive_html(
    *,
    frame: pd.DataFrame,
    palette: Mapping[str, str],
    low_count_warning: LowCountDepthWarning,
    output_path: str | Path,
    source_table_path: str | Path,
    source_manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Write the self-contained HTML and return its checksummed receipt."""

    output = Path(output_path)
    source_table = Path(source_table_path)
    source_manifest = Path(source_manifest_path)
    if not source_table.is_file() or not source_manifest.is_file():
        raise SO1RawExpressionInteractiveError(
            "Interactive source artifacts are missing."
        )
    validated = validate_interactive_frame(frame, palette)
    valid_clusters = sorted(palette, key=_cluster_sort_key)
    validated_warning = parse_low_count_depth_warning(
        {"low_count_depth_warning": _warning_record(low_count_warning)},
        valid_clusters=valid_clusters,
    )
    cluster_sizes = validated["expression_cluster"].astype(str).value_counts()
    for item in validated_warning.flagged_clusters:
        if int(cluster_sizes[item.cluster]) != item.cluster_cells:
            raise SO1RawExpressionInteractiveError(
                f"Low-count warning cluster size disagrees with the source table for {item.cluster}."
            )
    browser_payload = build_interactive_payload(validated, palette)
    html = _render_html(
        validated,
        palette,
        low_count_warning=validated_warning,
    )
    _validate_rendered_html(html, frame=validated)
    _atomic_write_text(output, html)
    if output.read_text(encoding="utf-8") != html:
        raise SO1RawExpressionInteractiveError(
            "Interactive HTML failed its write/read check."
        )
    return _receipt_with_self_hash(
        {
            "schema": INTERACTIVE_SCHEMA,
            "status": "complete",
            "self_contained": True,
            "offline_network_policy": "connect-src-none",
            "point_count": int(len(validated)),
            "core_order": list(SO1_CORE_NUMBERS),
            "cluster_count": int(len(palette)),
            "cluster_labels": valid_clusters,
            "low_count_depth_warning": _warning_record(validated_warning),
            "html_sha256": sha256_file(output),
            "html_size_bytes": int(output.stat().st_size),
            "source_table_sha256": sha256_file(source_table),
            "source_manifest_sha256": sha256_file(source_manifest),
            "browser_payload_sha256": _browser_payload_sha256(browser_payload),
            "browser_payload_fields": [
                "numeric_core_number",
                "x_um",
                "y_um",
                "so1_expression_cluster_code",
            ],
            "stable_identifiers_included": False,
            "expression_values_included": False,
            "pca_scores_included": False,
            "neighbor_edges_included": False,
            "clinical_fields_included": False,
            "model_data_included": False,
            "vendor_qc_fields_included": False,
        }
    )


def _write_transfer_zip(*, html_path: Path, zip_path: Path) -> Mapping[str, Any]:
    """Create and verify a deterministic, one-member transfer archive."""

    info = zipfile.ZipInfo(html_path.name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(info, html_path.read_bytes())
    _atomic_write_bytes(zip_path, buffer.getvalue())
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        members = archive.infolist()
        if archive.testzip() is not None or len(members) != 1:
            raise SO1RawExpressionInteractiveError(
                "Transfer ZIP integrity check failed."
            )
        member = members[0]
        member_path = Path(member.filename)
        if (
            member.filename != html_path.name
            or member_path.name != member.filename
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise SO1RawExpressionInteractiveError(
                "Transfer ZIP member name is unsafe."
            )
        archived = archive.read(member)
    if not hmac.compare_digest(
        hashlib.sha256(archived).hexdigest(), sha256_file(html_path)
    ):
        raise SO1RawExpressionInteractiveError(
            "Transfer ZIP member differs from the HTML file."
        )
    return {
        "member": html_path.name,
        "member_sha256": sha256_file(html_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "deterministic_timestamp": "1980-01-01T00:00:00",
    }


def _render_readme(
    *,
    html_name: str,
    zip_name: str,
    cluster_labels: Sequence[str],
    low_count_warning: LowCountDepthWarning,
) -> str:
    labels = f"`{cluster_labels[0]}` through `{cluster_labels[-1]}`"
    warning_lines = (
        "\n".join(
            f"- `{item.cluster}`: {item.below_threshold_cells:,} of "
            f"{item.cluster_cells:,} cells below "
            f"{low_count_warning.threshold_transcripts:,} transcripts "
            f"({100.0 * item.fraction_below_threshold:.1f}%)."
            for item in low_count_warning.flagged_clusters
        )
        if low_count_warning.flagged_clusters
        else (
            f"- The completed source audit found "
            f"{low_count_warning.cohort_below_threshold_cells:,} cells below "
            f"{low_count_warning.threshold_transcripts:,} transcripts, but no "
            "cluster contained at least half of that low-count set."
        )
    )
    return f"""# Interactive SO1 raw-expression cluster map

Open `{html_name}` in a current desktop browser. It is self-contained and needs
no server or internet connection. `{zip_name}` contains the identical HTML as
its only archive member.

## Interaction and label scope

- Click {labels} to keep one joint SO1 cluster bright while fading the others.
- Click the selected cluster again, choose **Show all clusters**, or press Escape
  to restore all colors.
- Scroll to zoom, drag to pan, double-click to reset a panel, and hover for only
  numeric core, cluster, and rounded tissue coordinates.
- **Export current view as PNG** saves the current combined 3 x 5 view.

The `S1E` namespace is specific to this independently clustered SO1 analysis.
For example, `S1E3` must not be interpreted as the same group as SO2 `E3`.
The viewer performs no normalization, PCA, neighbor construction, clustering,
model inference, training, or GPU work.

## Source-derived count-depth audit

{warning_lines}

Count depth can be associated with expression-cluster separation. These clusters are
expression-derived groups, not validated cell types, and they do not establish
signaling, biological influence, or causality. Marker-based and pathological
validation remains separate.

All {EXPECTED_TOTAL_CELLS:,} source cells are displayed. The source analysis retained low-depth
cells and cells that failed vendor QC; no vendor-QC field is included in the
browser payload.

The HTML omits stable cell identifiers, expression values, PCA scores, neighbor
edges, model data, clinical fields, and source filesystem paths. It contains
exact cell-level tissue coordinates, so share it only with authorized
collaborators.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\
  render-so1-raw-expression-interactive
```
"""


def _verify_interactive_manifest(
    output_root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_payload: Mapping[str, Any] | None = None,
) -> None:
    _verify_self_hash(manifest, label="SO1 raw-expression interactive manifest")
    cluster_count = int(manifest.get("cluster_count", -1))
    expected_labels = [f"S1E{index}" for index in range(max(0, cluster_count))]
    if any(
        (
            manifest.get("schema") != INTERACTIVE_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("source_analysis_id") != SOURCE_ANALYSIS_ID,
            tuple(manifest.get("core_order", ())) != SO1_CORE_NUMBERS,
            manifest.get("core_cell_counts")
            != {
                str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                for number in SO1_CORE_NUMBERS
            },
            int(manifest.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
            cluster_count <= 0,
            manifest.get("cluster_labels") != expected_labels,
            not math.isclose(
                float(manifest.get("leiden_resolution", math.nan)),
                DEFAULT_LEIDEN_RESOLUTION,
            ),
        )
    ):
        raise SO1RawExpressionInteractiveError(
            "SO1 raw-expression interactive manifest identity is invalid."
        )
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping) or any(
        (
            execution.get("visualization_only") is not True,
            execution.get("normalization_or_pca") is not False,
            execution.get("neighbor_construction") is not False,
            execution.get("reclustering") is not False,
            execution.get("model_inference") is not False,
            execution.get("model_training") is not False,
            execution.get("gpu_used") is not False,
        )
    ):
        raise SO1RawExpressionInteractiveError(
            "Interactive execution boundary changed."
        )
    code_provenance = execution.get("code_provenance")
    if not isinstance(code_provenance, Mapping) or any(
        (
            code_provenance.get("schema")
            != "so1_raw_expression_analysis_code_provenance_v1",
            code_provenance.get("workflow")
            != "so1_14core_raw_expression_interactive",
            not isinstance(code_provenance.get("relevant_code"), Mapping),
            not code_provenance.get("relevant_code"),
            code_provenance.get("gpu_used") is not False,
        )
    ):
        raise SO1RawExpressionInteractiveError(
            "Interactive code provenance is incomplete."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SO1RawExpressionInteractiveError(
            "SO1 interactive manifest lacks output checksums."
        )
    required = {"README.md", HTML_FILENAME, ZIP_FILENAME}
    if set(files) != required:
        raise SO1RawExpressionInteractiveError(
            "Interactive output set changed: "
            f"{sorted(set(files).symmetric_difference(required))}"
        )
    on_disk = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.relative_to(output_root).as_posix() != "manifest.json"
    }
    if on_disk != required:
        raise SO1RawExpressionInteractiveError(
            "Interactive directory contains unrecorded or missing files: "
            f"{sorted(on_disk.symmetric_difference(required))}"
        )
    for relative, record in files.items():
        path = output_root / str(relative)
        if not path.is_file() or _file_record(path) != dict(record):
            raise SO1RawExpressionInteractiveError(
                f"Interactive output checksum changed: {relative}"
            )
    render_receipt = manifest.get("render_receipt")
    zip_receipt = manifest.get("zip_receipt")
    if not isinstance(render_receipt, Mapping) or not isinstance(zip_receipt, Mapping):
        raise SO1RawExpressionInteractiveError(
            "Interactive manifest lacks render or ZIP checksum receipts."
        )
    _verify_self_hash(render_receipt, label="SO1 raw-expression HTML receipt")
    html_path = output_root / HTML_FILENAME
    zip_path = output_root / ZIP_FILENAME
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise SO1RawExpressionInteractiveError(
            "Interactive manifest lacks source-artifact checksums."
        )
    cluster_table_source = source_artifacts.get("cluster_table")
    analysis_manifest_source = source_artifacts.get("analysis_manifest")
    if not isinstance(cluster_table_source, Mapping) or not isinstance(
        analysis_manifest_source, Mapping
    ):
        raise SO1RawExpressionInteractiveError(
            "Interactive source-artifact records are incomplete."
        )
    expected_payload_fields = [
        "numeric_core_number",
        "x_um",
        "y_um",
        "so1_expression_cluster_code",
    ]
    if any(
        (
            render_receipt.get("schema") != INTERACTIVE_SCHEMA,
            render_receipt.get("status") != "complete",
            render_receipt.get("self_contained") is not True,
            render_receipt.get("offline_network_policy") != "connect-src-none",
            int(render_receipt.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
            tuple(render_receipt.get("core_order", ())) != SO1_CORE_NUMBERS,
            int(render_receipt.get("cluster_count", -1)) != cluster_count,
            render_receipt.get("cluster_labels") != expected_labels,
            render_receipt.get("low_count_depth_warning")
            != manifest.get("low_count_depth_warning"),
            render_receipt.get("browser_payload_fields") != expected_payload_fields,
            render_receipt.get("source_table_sha256")
            != cluster_table_source.get("sha256"),
            render_receipt.get("source_manifest_sha256")
            != analysis_manifest_source.get("sha256"),
            render_receipt.get("html_sha256") != sha256_file(html_path),
            int(render_receipt.get("html_size_bytes", -1))
            != int(html_path.stat().st_size),
            zip_receipt.get("member") != HTML_FILENAME,
            zip_receipt.get("member_sha256") != sha256_file(html_path),
            zip_receipt.get("zip_sha256") != sha256_file(zip_path),
            int(zip_receipt.get("zip_size_bytes", -1)) != int(zip_path.stat().st_size),
        )
    ):
        raise SO1RawExpressionInteractiveError(
            "Interactive render or ZIP receipt does not match the output files."
        )
    for flag in (
        "stable_identifiers_included",
        "expression_values_included",
        "pca_scores_included",
        "neighbor_edges_included",
        "clinical_fields_included",
        "model_data_included",
        "vendor_qc_fields_included",
    ):
        if render_receipt.get(flag) is not False:
            raise SO1RawExpressionInteractiveError(
                f"Interactive render privacy flag changed: {flag}."
            )
    html = html_path.read_text(encoding="utf-8")
    _validate_rendered_html(html, frame=pd.DataFrame())
    payload_match = re.search(
        r'<script id="bagm-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if payload_match is None:
        raise SO1RawExpressionInteractiveError(
            "Interactive HTML lacks its embedded data payload."
        )
    try:
        payload = json.loads(payload_match.group(1))
    except (TypeError, ValueError) as exc:
        raise SO1RawExpressionInteractiveError(
            "Interactive HTML payload is not valid JSON."
        ) from exc
    if render_receipt.get("browser_payload_sha256") != _browser_payload_sha256(
        payload
    ):
        raise SO1RawExpressionInteractiveError(
            "Interactive browser payload checksum changed."
        )
    if expected_payload is not None and payload != dict(expected_payload):
        raise SO1RawExpressionInteractiveError(
            "Interactive browser payload differs from the verified source table."
        )
    if any(
        (
            payload.get("schema") != INTERACTIVE_SCHEMA,
            payload.get("point_count") != EXPECTED_TOTAL_CELLS,
            tuple(payload.get("core_order", ())) != SO1_CORE_NUMBERS,
            payload.get("clusters") != expected_labels,
        )
    ):
        raise SO1RawExpressionInteractiveError(
            "Interactive HTML payload identity changed."
        )
    core_payloads = payload.get("cores")
    if not isinstance(core_payloads, list) or len(core_payloads) != len(
        SO1_CORE_NUMBERS
    ):
        raise SO1RawExpressionInteractiveError(
            "Interactive HTML payload lacks all 14 core arrays."
        )
    decoded_points = 0
    for core_number, core_payload in zip(
        SO1_CORE_NUMBERS, core_payloads, strict=True
    ):
        if not isinstance(core_payload, Mapping):
            raise SO1RawExpressionInteractiveError(
                "Interactive HTML core payload is malformed."
            )
        point_count = EXPECTED_CELL_COUNTS_BY_CORE[core_number]
        try:
            x = np.frombuffer(
                base64.b64decode(str(core_payload["x_b64"]), validate=True),
                dtype="<f8",
            )
            y = np.frombuffer(
                base64.b64decode(str(core_payload["y_b64"]), validate=True),
                dtype="<f8",
            )
            codes = np.frombuffer(
                base64.b64decode(
                    str(core_payload["cluster_codes_b64"]), validate=True
                ),
                dtype=np.uint8,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SO1RawExpressionInteractiveError(
                f"Interactive HTML arrays are invalid for SO1 core {core_number}."
            ) from exc
        if any(
            (
                int(core_payload.get("core_number", -1)) != core_number,
                int(core_payload.get("point_count", -1)) != point_count,
                len(x) != point_count,
                len(y) != point_count,
                len(codes) != point_count,
                not np.isfinite(x).all(),
                not np.isfinite(y).all(),
                bool(len(codes) and int(codes.max()) >= cluster_count),
            )
        ):
            raise SO1RawExpressionInteractiveError(
                f"Interactive HTML point arrays changed for SO1 core {core_number}."
            )
        decoded_points += point_count
    if decoded_points != EXPECTED_TOTAL_CELLS:
        raise SO1RawExpressionInteractiveError(
            "Interactive HTML payload point count does not reconcile."
        )
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        if archive.testzip() is not None or archive.namelist() != [HTML_FILENAME]:
            raise SO1RawExpressionInteractiveError(
                "Interactive transfer ZIP is invalid."
            )
        archived = archive.read(HTML_FILENAME)
    if not hmac.compare_digest(
        hashlib.sha256(archived).hexdigest(), sha256_file(html_path)
    ):
        raise SO1RawExpressionInteractiveError(
            "Archived and standalone HTML checksums differ."
        )


def _resolve_directory(value: str | Path, *, paths: ProjectPaths) -> Path:
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = paths.project_root / resolved
    return resolved.resolve(strict=False)


def run_so1_raw_expression_interactive(
    *,
    paths: ProjectPaths,
    source_analysis_id: str | None = None,
    source_report_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the completed offline SO1 expression viewer."""

    selected_analysis = (
        SOURCE_ANALYSIS_ID if source_analysis_id is None else str(source_analysis_id)
    )
    if selected_analysis != SOURCE_ANALYSIS_ID:
        raise SO1RawExpressionInteractiveError(
            "The viewer requires the completed primary SO1 raw-expression analysis."
        )
    source_root = (
        paths.report_root
        / "analyses"
        / DEFAULT_SOURCE_REPORT
        / selected_analysis
        if source_report_dir is None
        else _resolve_directory(source_report_dir, paths=paths)
    )
    source_manifest_path = source_root / "manifest.json"
    source_table_path = source_root / SOURCE_TABLE_RELATIVE_PATH
    source_palette_path = source_root / SOURCE_PALETTE_RELATIVE_PATH
    source_figure_path = source_root / SOURCE_FIGURE_RELATIVE_PATH
    source_manifest = _read_json(
        source_manifest_path, label="SO1 raw-expression clustering manifest"
    )

    from .so1_raw_expression_clustering import (
        ANALYSIS_ID as CLUSTERING_ANALYSIS_ID,
        DEFAULT_LEIDEN_RESOLUTION as CLUSTERING_LEIDEN_RESOLUTION,
        EXPECTED_CELL_COUNTS_BY_CORE as CLUSTERING_CELL_COUNTS_BY_CORE,
        EXPECTED_TOTAL_CELLS as CLUSTERING_TOTAL_CELLS,
        LABEL_PREFIX as CLUSTERING_LABEL_PREFIX,
        SO1_CORE_NUMBERS as CLUSTERING_CORE_NUMBERS,
        runtime_code_provenance,
        verify_so1_raw_expression_clustering_bundle,
    )

    if any(
        (
            CLUSTERING_ANALYSIS_ID != SOURCE_ANALYSIS_ID,
            CLUSTERING_LEIDEN_RESOLUTION != DEFAULT_LEIDEN_RESOLUTION,
            CLUSTERING_TOTAL_CELLS != EXPECTED_TOTAL_CELLS,
            dict(CLUSTERING_CELL_COUNTS_BY_CORE) != EXPECTED_CELL_COUNTS_BY_CORE,
            tuple(CLUSTERING_CORE_NUMBERS) != SO1_CORE_NUMBERS,
            CLUSTERING_LABEL_PREFIX != LABEL_PREFIX,
        )
    ):
        raise SO1RawExpressionInteractiveError(
            "SO1 viewer and clustering source contracts have drifted."
        )

    verify_so1_raw_expression_clustering_bundle(
        output_root=source_root,
        manifest=source_manifest,
        paths=paths,
    )
    configuration = source_manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise SO1RawExpressionInteractiveError(
            "SO1 source clustering manifest lacks its configuration."
        )
    if any(
        (
            source_manifest.get("analysis_id") != SOURCE_ANALYSIS_ID,
            source_manifest.get("analysis_scope")
            != "classical_raw_expression_only_joint_clustering",
            tuple(source_manifest.get("core_order", ())) != SO1_CORE_NUMBERS,
            int(source_manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            not math.isclose(
                float(configuration.get("leiden_resolution", math.nan)),
                DEFAULT_LEIDEN_RESOLUTION,
            ),
        )
    ):
        raise SO1RawExpressionInteractiveError(
            "Source report is not the completed SO1 raw-expression resolution-1.0 analysis."
        )
    palette_document = _read_json(
        source_palette_path, label="SO1 raw-expression palette"
    )
    palette_value = palette_document.get("colors")
    if not isinstance(palette_value, Mapping):
        raise SO1RawExpressionInteractiveError(
            "SO1 expression palette lacks a colors mapping."
        )
    palette = {str(key): str(value) for key, value in palette_value.items()}
    warning = parse_low_count_depth_warning(
        source_manifest,
        valid_clusters=sorted(palette, key=_cluster_sort_key),
    )
    if any(
        (
            int(source_manifest.get("below_library_size_floor_cells", -1))
            != warning.cohort_below_threshold_cells,
            not math.isclose(
                float(configuration.get("library_size_floor", math.nan)),
                float(warning.threshold_transcripts),
            ),
        )
    ):
        raise SO1RawExpressionInteractiveError(
            "Source low-count warning does not match its preprocessing audit."
        )

    # Deliberately read only the four fields permitted in the browser payload.
    frame = pd.read_parquet(
        source_table_path,
        columns=["core_number", "x_um", "y_um", "expression_cluster"],
    )
    validated = validate_interactive_frame(frame, palette)
    expected_payload = build_interactive_payload(validated, palette)
    if len(validated) != EXPECTED_TOTAL_CELLS:
        raise SO1RawExpressionInteractiveError(
            f"Expected {EXPECTED_TOTAL_CELLS} SO1 cells, found {len(validated)}."
        )
    observed_counts = validated.groupby("core_number", sort=False).size().to_dict()
    if observed_counts != EXPECTED_CELL_COUNTS_BY_CORE:
        raise SO1RawExpressionInteractiveError(
            "Interactive source per-core counts differ from the completed SO1 cohort."
        )
    if int(source_manifest.get("cluster_count", -1)) != len(palette):
        raise SO1RawExpressionInteractiveError(
            "Source manifest, table, and palette disagree on cluster count."
        )

    output_root = (
        paths.report_root
        / "analyses"
        / DEFAULT_OUTPUT_REPORT
        / selected_analysis
        if output_dir is None
        else _resolve_directory(output_dir, paths=paths)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    expected_sources = {
        "analysis_manifest": _file_record(source_manifest_path),
        "cluster_table": _file_record(source_table_path),
        "palette": _file_record(source_palette_path),
        "static_png": _file_record(source_figure_path),
    }
    if manifest_path.is_file():
        manifest = _read_json(
            manifest_path, label="SO1 raw-expression interactive manifest"
        )
        _verify_interactive_manifest(
            output_root,
            manifest,
            expected_payload=expected_payload,
        )
        if manifest.get("source_artifacts") != expected_sources:
            raise SO1RawExpressionInteractiveError(
                "Interactive viewer source checksums changed."
            )
        if manifest.get("low_count_depth_warning") != _warning_record(warning):
            raise SO1RawExpressionInteractiveError(
                "Interactive viewer source-derived low-count warning changed."
            )
    else:
        allowed_partial = {HTML_FILENAME, ZIP_FILENAME, "README.md"}
        unexpected = sorted(
            path.name
            for path in output_root.iterdir()
            if path.name not in allowed_partial
        )
        if unexpected:
            raise SO1RawExpressionInteractiveError(
                f"Unexpected partial interactive outputs: {unexpected}"
            )
        html_path = output_root / HTML_FILENAME
        zip_path = output_root / ZIP_FILENAME
        render_receipt = render_interactive_html(
            frame=validated,
            palette=palette,
            low_count_warning=warning,
            output_path=html_path,
            source_table_path=source_table_path,
            source_manifest_path=source_manifest_path,
        )
        zip_receipt = _write_transfer_zip(html_path=html_path, zip_path=zip_path)
        cluster_labels = sorted(palette, key=_cluster_sort_key)
        _atomic_write_text(
            output_root / "README.md",
            _render_readme(
                html_name=HTML_FILENAME,
                zip_name=ZIP_FILENAME,
                cluster_labels=cluster_labels,
                low_count_warning=warning,
            ),
        )
        manifest = _receipt_with_self_hash(
            {
                "schema": INTERACTIVE_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "source_analysis_id": selected_analysis,
                "analysis_scope": "so1_raw_expression_resolution_1p0_visualization_only",
                "leiden_resolution": DEFAULT_LEIDEN_RESOLUTION,
                "core_order": list(SO1_CORE_NUMBERS),
                "core_cell_counts": {
                    str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                    for number in SO1_CORE_NUMBERS
                },
                "point_count": EXPECTED_TOTAL_CELLS,
                "cluster_count": len(palette),
                "cluster_labels": cluster_labels,
                "low_count_depth_warning": _warning_record(warning),
                "interaction": {
                    "cluster_click_highlight": True,
                    "nonselected_opacity": 0.09,
                    "selected_draw_order": "last",
                    "show_all_and_escape_reset": True,
                    "zoom": "wheel",
                    "pan": "pointer_drag",
                    "hover_fields": [
                        "numeric_core_number",
                        "so1_expression_cluster",
                        "rounded_x_um",
                        "rounded_y_um",
                    ],
                    "combined_png_export": True,
                },
                "execution": {
                    "visualization_only": True,
                    "normalization_or_pca": False,
                    "neighbor_construction": False,
                    "reclustering": False,
                    "model_inference": False,
                    "model_training": False,
                    "gpu_used": False,
                    "code_provenance": runtime_code_provenance(
                        project_root=Path(__file__).resolve().parents[2],
                        relevant_paths=(
                            Path(__file__),
                            Path(__file__).resolve().with_name("cli.py"),
                            Path(__file__).resolve().with_name(
                                "so1_raw_expression_clustering.py"
                            ),
                            Path(__file__).resolve().with_name(
                                "relative_qkv_embedding_clustering.py"
                            ),
                            Path(__file__).resolve().with_name(
                                "so2_raw_expression_interactive.py"
                            ),
                            Path(__file__).resolve().with_name(
                                "so2_hl_interactive.py"
                            ),
                            Path(__file__).resolve().parents[2]
                            / "experiments"
                            / "campaigns"
                            / "cmp_20260826_so1_14core_classical_raw_expression"
                            / "INTERACTIVE_MAP.md",
                        ),
                        workflow="so1_14core_raw_expression_interactive",
                    ),
                },
                "sharing": {
                    "self_contained_offline_html": True,
                    "authorized_collaborators_only": True,
                    "exact_tissue_coordinates_included": True,
                    "stable_cell_identifiers_included": False,
                    "expression_values_included": False,
                    "pca_scores_included": False,
                    "neighbor_edges_included": False,
                    "clinical_fields_included": False,
                    "model_data_included": False,
                    "vendor_qc_fields_included": False,
                },
                "interpretation": {
                    "expression_derived_clusters": True,
                    "so1_labels_map_to_so2_labels": False,
                    "cell_types_established": False,
                    "signaling_established": False,
                    "biological_influence_established": False,
                    "causality_established": False,
                    "low_count_warning_source_derived": True,
                },
                "source_artifacts": expected_sources,
                "render_receipt": dict(render_receipt),
                "zip_receipt": dict(zip_receipt),
                "files": _file_manifest(output_root),
            }
        )
        _atomic_write_json(manifest_path, manifest)
        _verify_interactive_manifest(
            output_root,
            manifest,
            expected_payload=expected_payload,
        )

    return {
        "status": "complete",
        "source_analysis_id": selected_analysis,
        "device": "none (visualization-only CPU workflow)",
        "point_count": int(manifest["point_count"]),
        "cluster_count": int(manifest["cluster_count"]),
        "cluster_labels": list(manifest["cluster_labels"]),
        "leiden_resolution": float(manifest["leiden_resolution"]),
        "core_order": list(manifest["core_order"]),
        "static_png": source_figure_path.as_posix(),
        "output_root": output_root.as_posix(),
        "html": (output_root / HTML_FILENAME).as_posix(),
        "transfer_zip": (output_root / ZIP_FILENAME).as_posix(),
        "manifest": manifest_path.as_posix(),
    }


__all__ = [
    "LowCountClusterWarning",
    "LowCountDepthWarning",
    "SO1RawExpressionInteractiveError",
    "build_interactive_payload",
    "interactive_javascript",
    "parse_low_count_depth_warning",
    "render_interactive_html",
    "run_so1_raw_expression_interactive",
    "validate_interactive_frame",
]
