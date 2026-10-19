"""Offline interactive viewer for direct-hL SO1 contextual clusters.

This visualization-only workflow consumes the completed SO1 direct-hL
clustering report.  It performs no inference, embedding extraction, neighbor
construction, clustering, training, registry access, or GPU work.  The browser
payload contains only numeric core labels, tissue coordinates, compact ``S1C``
cluster codes, and row-independent display summaries.
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
from typing import Any, Mapping
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
    deterministic_glasbey_palette,
)
from .so2_hl_interactive import (
    _interactive_css as _shared_interactive_css,
    interactive_javascript as _shared_interactive_javascript,
)


INTERACTIVE_SCHEMA = "so1_14core_hl_direct_knn_interactive_v1"
SOURCE_SCHEMA = "so1_14core_relative_qkv_embeddings_direct_knn_clustering_v1"
SOURCE_PIPELINE_KIND = "direct_embedding_l2_cosine_knn_leiden"
SOURCE_REPRESENTATION = "hL_final_graph_pre_decoder"
DEFAULT_SOURCE_REPORT = "so1_14core_model_embedding_direct_knn_clustering"
DEFAULT_OUTPUT_REPORT = "so1_14core_hl_direct_knn_interactive"
HTML_FILENAME = (
    "contextual_direct_hl_leiden_resolution_1p0_spatial_14cores_interactive.html"
)
ZIP_FILENAME = (
    "contextual_direct_hl_leiden_resolution_1p0_spatial_14cores_interactive.zip"
)
SOURCE_TABLE_RELATIVE_PATH = Path("tables/cell_embedding_clusters.parquet")
SOURCE_PALETTE_RELATIVE_PATH = Path("clustering/contextual_palette.json")
SOURCE_FIGURE_RELATIVE_PATH = Path(
    "figures/contextual_direct_hl_leiden_resolution_1p0_spatial_14cores.png"
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
EXPECTED_EMBEDDING_DIMENSION = 256
DEFAULT_N_NEIGHBORS = 30
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_RANDOM_SEED = 20260825
LABEL_PREFIX = "S1C"
COORDINATE_ORIENTATION = "low_y_at_top"

_SOURCE_TABLE_COLUMNS = (
    "cell_index",
    "core_number",
    "x_um",
    "y_um",
    "contextual_cluster",
)
_PAYLOAD_TOP_LEVEL_FIELDS = {
    "schema",
    "method",
    "core_order",
    "clusters",
    "palette",
    "point_count",
    "coordinate_units",
    "coordinate_orientation",
    "cores",
}
_PAYLOAD_CORE_FIELDS = {
    "core_number",
    "point_count",
    "bounds",
    "cluster_counts",
    "x_b64",
    "y_b64",
    "cluster_codes_b64",
}
_PAYLOAD_BOUND_FIELDS = {"x_min", "x_max", "y_min", "y_max"}


class SO1HLDirectInteractiveError(ValueError):
    """Raised when the SO1 direct-hL viewer contract is violated."""


@dataclass(frozen=True, slots=True)
class VerifiedSO1DirectHLSource:
    """Checksum- and contract-verified static direct-hL source."""

    root: Path
    manifest: Mapping[str, Any]
    frame: pd.DataFrame
    palette: Mapping[str, str]
    cluster_labels: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cluster_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"S1C([0-9]+)", label)
    if match is None:
        raise SO1HLDirectInteractiveError(
            f"SO1 contextual cluster label must have the form S1C<number>: {label!r}"
        )
    return int(match.group(1)), label


def _validate_hex_color(value: object, *, label: str) -> str:
    color = str(value)
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None:
        raise SO1HLDirectInteractiveError(
            f"Palette color for {label} must be a six-digit hexadecimal color."
        )
    return color.upper()


def deterministic_contextual_palette(cluster_count: int) -> dict[str, str]:
    """Return the repository's deterministic contextual colors under ``S1C``."""

    if isinstance(cluster_count, bool) or int(cluster_count) <= 0:
        raise SO1HLDirectInteractiveError("Cluster count must be positive.")
    base = deterministic_glasbey_palette(int(cluster_count), namespace="contextual")
    palette = {
        f"{LABEL_PREFIX}{index}": str(base[f"C{index}"]).upper()
        for index in range(int(cluster_count))
    }
    if len(palette) != int(cluster_count) or len(set(palette.values())) != len(
        palette
    ):
        raise SO1HLDirectInteractiveError("Deterministic contextual palette is invalid.")
    return palette


def validate_interactive_frame(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> pd.DataFrame:
    """Validate exact cells, order, coordinates, and contiguous ``S1C`` labels."""

    missing = sorted(set(_SOURCE_TABLE_COLUMNS).difference(frame.columns))
    if missing:
        raise SO1HLDirectInteractiveError(
            f"Interactive source table lacks required columns: {missing}"
        )
    if frame.empty:
        raise SO1HLDirectInteractiveError("Interactive source table is empty.")
    if frame.loc[:, list(_SOURCE_TABLE_COLUMNS)].isna().any().any():
        raise SO1HLDirectInteractiveError(
            "Interactive source fields contain missing values."
        )

    validated = frame.copy(deep=False)
    observed_core_order = tuple(
        int(value) for value in validated["core_number"].drop_duplicates().tolist()
    )
    if observed_core_order != SO1_CORE_NUMBERS:
        raise SO1HLDirectInteractiveError(
            "Interactive map requires all 14 SO1 cores in the locked order "
            f"{SO1_CORE_NUMBERS}; observed {observed_core_order}."
        )
    core_values = validated["core_number"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(core_values).all() or not np.array_equal(
        core_values, core_values.astype(np.int64)
    ):
        raise SO1HLDirectInteractiveError("Core numbers must be finite integers.")

    indices = validated["cell_index"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(indices).all() or not np.array_equal(
        indices, indices.astype(np.int64)
    ):
        raise SO1HLDirectInteractiveError("Cell indices must be finite integers.")
    for core_number in SO1_CORE_NUMBERS:
        core_indices = validated.loc[
            validated["core_number"] == core_number, "cell_index"
        ].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(core_indices, np.arange(len(core_indices), dtype=np.int64)):
            raise SO1HLDirectInteractiveError(
                f"SO1 core {core_number} cell indices must be ordered contiguously from zero."
            )

    for coordinate in ("x_um", "y_um"):
        values = validated[coordinate].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise SO1HLDirectInteractiveError(
                f"Interactive coordinates contain non-finite {coordinate} values."
            )

    labels = validated["contextual_cluster"].astype(str)
    observed_labels = sorted(set(labels.tolist()), key=_cluster_sort_key)
    expected_labels = [f"{LABEL_PREFIX}{index}" for index in range(len(observed_labels))]
    if observed_labels != expected_labels:
        raise SO1HLDirectInteractiveError(
            "SO1 contextual cluster labels must be contiguous from S1C0."
        )
    palette_keys = sorted((str(key) for key in palette), key=_cluster_sort_key)
    if palette_keys != observed_labels:
        raise SO1HLDirectInteractiveError(
            "Palette must cover the exact SO1 contextual cluster label set."
        )
    canonical_palette = {
        label: _validate_hex_color(palette[label], label=label)
        for label in palette_keys
    }
    if len(set(canonical_palette.values())) != len(canonical_palette):
        raise SO1HLDirectInteractiveError("Contextual cluster colors must be unique.")
    if canonical_palette != deterministic_contextual_palette(len(observed_labels)):
        raise SO1HLDirectInteractiveError(
            "Source palette differs from the locked deterministic contextual palette."
        )
    return validated


def _encode_array(value: np.ndarray, *, dtype: str | np.dtype[Any]) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return base64.b64encode(memoryview(array).cast("B")).decode("ascii")


def build_interactive_payload(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> dict[str, Any]:
    """Encode only numeric core, coordinates, and contextual cluster codes."""

    validated = validate_interactive_frame(frame, palette)
    clusters = sorted(
        set(validated["contextual_cluster"].astype(str)), key=_cluster_sort_key
    )
    if len(clusters) > np.iinfo(np.uint8).max + 1:
        raise SO1HLDirectInteractiveError(
            "Too many SO1 contextual clusters for Uint8 browser codes."
        )
    canonical_palette = {
        label: _validate_hex_color(palette[label], label=label) for label in clusters
    }
    cluster_to_code = {label: index for index, label in enumerate(clusters)}

    core_records: list[dict[str, Any]] = []
    encoded_points = 0
    for core_number in SO1_CORE_NUMBERS:
        core = validated.loc[validated["core_number"] == core_number]
        if core.empty:
            raise SO1HLDirectInteractiveError(
                f"SO1 core {core_number} has no interactive points."
            )
        x = np.ascontiguousarray(core["x_um"].to_numpy(dtype="<f8", copy=True))
        y = np.ascontiguousarray(core["y_um"].to_numpy(dtype="<f8", copy=True))
        codes = np.ascontiguousarray(
            core["contextual_cluster"]
            .astype(str)
            .map(cluster_to_code)
            .to_numpy(dtype=np.uint8, copy=True)
        )
        if not (len(x) == len(y) == len(codes) == len(core)):
            raise SO1HLDirectInteractiveError(
                f"Coordinate/label ordering mismatch for SO1 core {core_number}."
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
                "x_b64": _encode_array(x, dtype="<f8"),
                "y_b64": _encode_array(y, dtype="<f8"),
                "cluster_codes_b64": _encode_array(codes, dtype=np.uint8),
            }
        )
        encoded_points += len(core)
    if encoded_points != len(validated):
        raise SO1HLDirectInteractiveError(
            "Not every source row was encoded into the interactive payload."
        )
    payload = {
        "schema": INTERACTIVE_SCHEMA,
        "method": SOURCE_PIPELINE_KIND,
        "core_order": list(SO1_CORE_NUMBERS),
        "clusters": clusters,
        "palette": canonical_palette,
        "point_count": int(len(validated)),
        "coordinate_units": "micrometres",
        "coordinate_orientation": COORDINATE_ORIENTATION,
        "cores": core_records,
    }
    _verify_payload_matches_source(payload, validated, canonical_palette)
    return payload


def _decode_array(value: object, *, dtype: str | np.dtype[Any], label: str) -> np.ndarray:
    try:
        raw = base64.b64decode(str(value), validate=True)
        array = np.frombuffer(raw, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise SO1HLDirectInteractiveError(
            f"Interactive payload has an invalid {label} array."
        ) from exc
    return array


def _verify_payload_matches_source(
    payload: Mapping[str, Any],
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> None:
    """Decode the browser payload and prove row-for-row source equality."""

    if set(payload) != _PAYLOAD_TOP_LEVEL_FIELDS:
        raise SO1HLDirectInteractiveError(
            "Browser payload top-level fields changed."
        )
    validated = validate_interactive_frame(frame, palette)
    labels = sorted(palette, key=_cluster_sort_key)
    if any(
        (
            payload.get("schema") != INTERACTIVE_SCHEMA,
            payload.get("method") != SOURCE_PIPELINE_KIND,
            tuple(payload.get("core_order", ())) != SO1_CORE_NUMBERS,
            payload.get("clusters") != labels,
            payload.get("palette")
            != {
                label: _validate_hex_color(palette[label], label=label)
                for label in labels
            },
            int(payload.get("point_count", -1)) != len(validated),
            payload.get("coordinate_units") != "micrometres",
            payload.get("coordinate_orientation") != COORDINATE_ORIENTATION,
        )
    ):
        raise SO1HLDirectInteractiveError("Browser payload identity changed.")
    core_payloads = payload.get("cores")
    if not isinstance(core_payloads, list) or len(core_payloads) != len(
        SO1_CORE_NUMBERS
    ):
        raise SO1HLDirectInteractiveError(
            "Browser payload lacks exactly fourteen core arrays."
        )
    cluster_to_code = {label: index for index, label in enumerate(labels)}
    decoded_points = 0
    for core_number, record in zip(SO1_CORE_NUMBERS, core_payloads, strict=True):
        if not isinstance(record, Mapping) or set(record) != _PAYLOAD_CORE_FIELDS:
            raise SO1HLDirectInteractiveError(
                f"Browser payload fields changed for SO1 core {core_number}."
            )
        bounds = record.get("bounds")
        if not isinstance(bounds, Mapping) or set(bounds) != _PAYLOAD_BOUND_FIELDS:
            raise SO1HLDirectInteractiveError(
                f"Browser payload bounds changed for SO1 core {core_number}."
            )
        core = validated.loc[validated["core_number"] == core_number]
        expected_x = core["x_um"].to_numpy(dtype="<f8", copy=True)
        expected_y = core["y_um"].to_numpy(dtype="<f8", copy=True)
        expected_codes = (
            core["contextual_cluster"]
            .astype(str)
            .map(cluster_to_code)
            .to_numpy(dtype=np.uint8, copy=True)
        )
        x = _decode_array(record.get("x_b64"), dtype="<f8", label="x")
        y = _decode_array(record.get("y_b64"), dtype="<f8", label="y")
        codes = _decode_array(
            record.get("cluster_codes_b64"), dtype=np.uint8, label="cluster-code"
        )
        expected_counts = np.bincount(
            expected_codes, minlength=len(labels)
        ).astype(np.int64)
        expected_bounds = {
            "x_min": float(expected_x.min()),
            "x_max": float(expected_x.max()),
            "y_min": float(expected_y.min()),
            "y_max": float(expected_y.max()),
        }
        if any(
            (
                int(record.get("core_number", -1)) != core_number,
                int(record.get("point_count", -1)) != len(core),
                not np.array_equal(x, expected_x),
                not np.array_equal(y, expected_y),
                not np.array_equal(codes, expected_codes),
                record.get("cluster_counts") != expected_counts.tolist(),
                dict(bounds) != expected_bounds,
            )
        ):
            raise SO1HLDirectInteractiveError(
                f"Decoded browser payload differs from the source table for SO1 core {core_number}."
            )
        decoded_points += len(core)
    if decoded_points != len(validated):
        raise SO1HLDirectInteractiveError(
            "Decoded browser payload does not reconcile to the source table."
        )


def _browser_payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def interactive_javascript() -> str:
    """Return the established dependency-free viewer with SO1 wording."""

    return (
        _shared_interactive_javascript()
        .replace("BAGM_SO2_VIEWER", "BAGM_SO1_HL_VIEWER")
        .replace("SO2 Core", "SO1 Core")
        .replace("Showing all contextual clusters.", "Showing all direct hL clusters.")
        .replace("All contextual clusters", "All direct hL clusters")
        .replace("so2_hL_contextual_clusters_", "so1_direct_hL_clusters_")
    )


def _cluster_buttons(frame: pd.DataFrame, palette: Mapping[str, str]) -> str:
    counts = frame["contextual_cluster"].astype(str).value_counts()
    return "\n".join(
        (
            f'<button type="button" class="cluster-chip" data-cluster="{escape(label)}" '
            f'aria-pressed="false" style="--cluster-color:{escape(str(palette[label]))}">'
            '<span class="swatch" aria-hidden="true"></span>'
            f'<span>{escape(label)}</span><span class="cluster-count">'
            f"{int(counts[label]):,}</span></button>"
        )
        for label in sorted(palette, key=_cluster_sort_key)
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
            f'aria-label="Interactive direct hL cluster map for SO1 Core {core_number}"></canvas>'
            '<div class="hover-card" hidden></div>'
            "</div></article>"
        )
    panels.append(
        '<aside class="legend-panel"><div><strong>Shared joint SO1 direct-hL clusters</strong><br>'
        "S1C labels and colors are identical across all fourteen tissue panels.<br>"
        "Select a cluster to highlight it across every core.</div></aside>"
    )
    return "\n".join(panels)


def _render_html(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
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
<title>SO1 14-core direct hL clusters</title>
<style>{_shared_interactive_css()}</style>
</head>
<body>
<header>
  <h1>SO1 14-core direct hL clusters</h1>
  <p>Direct {EXPECTED_EMBEDDING_DIMENSION}-dimensional final contextual representation (hL) &rarr; L2 normalization &rarr; cosine {DEFAULT_N_NEIGHBORS}-nearest-neighbor graph &rarr; Leiden clustering at resolution {DEFAULT_LEIDEN_RESOLUTION:g}. PCA, mean-centering, and feature projection were not used.</p>
  <p>Click an S1C cluster to keep it bright while fading the others. Scroll to zoom, drag to pan, double-click a panel to reset it, and hover for core, cluster, and rounded tissue coordinates.</p>
</header>
<main class="workspace">
  <aside class="controls" aria-label="Cluster controls">
    <h2>SO1 direct-hL clusters</h2>
    <p class="instructions">These are model-derived contextual groups, not validated cell types. Counts are joint across all fourteen SO1 cores.</p>
    <div class="cluster-list">{_cluster_buttons(frame, palette)}</div>
    <div class="action-row">
      <button type="button" class="action" id="show-all">Show all clusters</button>
      <button type="button" class="action" id="reset-views">Reset all spatial views</button>
      <button type="button" class="action primary" id="export-png">Export current view as PNG</button>
    </div>
    <p id="selection-status" aria-live="polite">Showing all direct hL clusters.</p>
    <p class="privacy-note"><strong>Authorized sharing only.</strong> This offline file contains exact cell-level tissue coordinates. It contains no cell identifiers, hL vectors, expression values, metadata fields, model checkpoint data, neighbor edges, source paths, or external network resources.</p>
  </aside>
  <section class="core-grid" aria-label="SO1 direct hL spatial cluster maps">{_core_panels(frame)}</section>
</main>
<script id="bagm-data" type="application/json">{payload_json}</script>
<script>{interactive_javascript()}</script>
</body>
</html>
"""


def _extract_browser_payload(html: str) -> Mapping[str, Any]:
    match = re.search(
        r'<script id="bagm-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if match is None:
        raise SO1HLDirectInteractiveError(
            "Interactive HTML lacks its embedded browser payload."
        )
    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError) as exc:
        raise SO1HLDirectInteractiveError(
            "Interactive HTML browser payload is invalid JSON."
        ) from exc
    if not isinstance(payload, Mapping):
        raise SO1HLDirectInteractiveError(
            "Interactive HTML browser payload must be a mapping."
        )
    return payload


def _validate_rendered_html(
    html: str,
    *,
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> Mapping[str, Any]:
    prohibited = (
        "cell_key",
        "cell_index",
        "global_cell_index",
        "core_alias",
        "checkpoint_path",
        "checkpoint_sha256",
        ".ckpt",
        "source_artifacts",
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
        raise SO1HLDirectInteractiveError(
            f"Shareable HTML contains prohibited source/network fields: {found}"
        )
    if "connect-src 'none'" not in html:
        raise SO1HLDirectInteractiveError(
            "Shareable HTML lacks the offline network CSP."
        )
    if len(re.findall(r"<canvas\s+[^>]*data-core=", html)) != len(
        SO1_CORE_NUMBERS
    ):
        raise SO1HLDirectInteractiveError(
            "Shareable HTML lacks exactly fourteen SO1 core canvases."
        )
    if "BAGM_SO2_VIEWER" in html or "SO2 Core" in html or any(
        f"SO1 Core {number}" not in html for number in SO1_CORE_NUMBERS
    ):
        raise SO1HLDirectInteractiveError(
            "Shareable HTML does not identify the exact SO1 core panels."
        )
    if re.search(
        r"<(?:script|img)\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=",
        html,
        flags=re.IGNORECASE,
    ) is not None:
        raise SO1HLDirectInteractiveError(
            "Shareable HTML contains an external DOM asset."
        )
    payload = _extract_browser_payload(html)
    _verify_payload_matches_source(payload, frame, palette)
    return payload


def render_interactive_html(
    *,
    frame: pd.DataFrame,
    palette: Mapping[str, str],
    output_path: str | Path,
    source_table_path: str | Path,
    source_manifest_path: str | Path,
    source_palette_path: str | Path,
    source_figure_path: str | Path,
) -> Mapping[str, Any]:
    """Write one self-contained HTML viewer and its checksum receipt."""

    source_paths = tuple(
        Path(value)
        for value in (
            source_table_path,
            source_manifest_path,
            source_palette_path,
            source_figure_path,
        )
    )
    if not all(path.is_file() for path in source_paths):
        raise SO1HLDirectInteractiveError("Interactive source artifacts are missing.")
    validated = validate_interactive_frame(frame, palette)
    html = _render_html(validated, palette)
    payload = _validate_rendered_html(html, frame=validated, palette=palette)
    output = Path(output_path)
    _atomic_write_text(output, html)
    persisted = output.read_text(encoding="utf-8")
    if persisted != html:
        raise SO1HLDirectInteractiveError(
            "Interactive HTML failed its write/read equality check."
        )
    persisted_payload = _validate_rendered_html(
        persisted, frame=validated, palette=palette
    )
    if persisted_payload != payload:
        raise SO1HLDirectInteractiveError(
            "Persisted browser payload changed after rendering."
        )
    return _receipt_with_self_hash(
        {
            "schema": INTERACTIVE_SCHEMA,
            "status": "complete",
            "pipeline_kind": SOURCE_PIPELINE_KIND,
            "representation": SOURCE_REPRESENTATION,
            "pca": False,
            "mean_center": False,
            "feature_projection": False,
            "l2_normalize_for_cosine": True,
            "self_contained": True,
            "offline_network_policy": "connect-src-none",
            "point_count": int(len(validated)),
            "core_order": list(SO1_CORE_NUMBERS),
            "cluster_count": int(len(palette)),
            "cluster_labels": sorted(palette, key=_cluster_sort_key),
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "n_neighbors": DEFAULT_N_NEIGHBORS,
            "leiden_resolution": DEFAULT_LEIDEN_RESOLUTION,
            "coordinate_orientation": COORDINATE_ORIENTATION,
            "html_sha256": sha256_file(output),
            "html_size_bytes": int(output.stat().st_size),
            "source_table_sha256": sha256_file(source_paths[0]),
            "source_manifest_sha256": sha256_file(source_paths[1]),
            "source_palette_sha256": sha256_file(source_paths[2]),
            "source_static_png_sha256": sha256_file(source_paths[3]),
            "browser_payload_sha256": _browser_payload_sha256(payload),
            "browser_payload_fields": [
                "numeric_core_number",
                "x_um",
                "y_um",
                "so1_contextual_cluster_code",
            ],
            "stable_identifiers_included": False,
            "cell_index_included": False,
            "expression_values_included": False,
            "embedding_values_included": False,
            "metadata_fields_included": False,
            "checkpoint_data_included": False,
            "neighbor_edges_included": False,
            "source_paths_included": False,
            "clinical_fields_included": False,
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
            raise SO1HLDirectInteractiveError("Transfer ZIP integrity check failed.")
        member = members[0]
        member_path = Path(member.filename)
        if any(
            (
                member.filename != html_path.name,
                member_path.name != member.filename,
                member_path.is_absolute(),
                ".." in member_path.parts,
                member.date_time != (1980, 1, 1, 0, 0, 0),
                member.compress_type != zipfile.ZIP_DEFLATED,
            )
        ):
            raise SO1HLDirectInteractiveError(
                "Transfer ZIP member contract is invalid."
            )
        archived = archive.read(member)
    if not hmac.compare_digest(
        hashlib.sha256(archived).hexdigest(), sha256_file(html_path)
    ):
        raise SO1HLDirectInteractiveError(
            "Transfer ZIP member differs from the standalone HTML."
        )
    return {
        "member": html_path.name,
        "member_sha256": sha256_file(html_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "deterministic_timestamp": "1980-01-01T00:00:00",
        "member_count": 1,
    }


def _render_readme(
    *,
    run_id: str,
    cluster_labels: list[str],
) -> str:
    return f"""# Interactive SO1 direct-hL cluster map

Open `{HTML_FILENAME}` in a current desktop browser. It is a self-contained
offline file. `{ZIP_FILENAME}` contains byte-identical HTML as its only member.

## Method and interaction

The joint analysis used the direct {EXPECTED_EMBEDDING_DIMENSION}-dimensional
final contextual representation (`hL`, after the final graph layer and before
the decoder), L2 normalization, a sparse cosine {DEFAULT_N_NEIGHBORS}-nearest-
neighbor graph, and Leiden resolution {DEFAULT_LEIDEN_RESOLUTION:g}. It used no
PCA, mean-centering, or feature projection. The cluster namespace is
`{cluster_labels[0]}` through `{cluster_labels[-1]}`.

- Click a cluster to keep it bright and fade the others in every core.
- Click it again, choose **Show all clusters**, or press Escape to show all.
- Scroll to zoom, drag to pan, double-click to reset a panel, and hover for only
  numeric core, cluster, and rounded tissue coordinates.
- **Export current view as PNG** saves the combined 3 x 5 view.

This viewer reuses completed assignments for run `{run_id}`. It performs no
embedding extraction, graph construction, clustering, inference, training, or
GPU work.

## Interpretation and sharing limits

These are model-derived contextual clusters, not validated cell types. They do
not independently establish cell type, signaling, biological influence, or
causality. Marker-based and pathological validation remains separate.

The HTML omits cell indices and stable identifiers, expression values, hL
vectors, metadata, checkpoint data, neighbor edges, clinical fields, and source
filesystem paths. It contains exact cell-level tissue coordinates, so share it
only with authorized collaborators.

## Reproduction

From the repository root, after the static direct-hL report is complete:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -c \
  'from spatial_benchmark.paths import current_paths; from spatial_benchmark.so1_hl_direct_interactive import run_so1_hl_direct_interactive; print(run_so1_hl_direct_interactive(paths=current_paths(), run_id="{run_id}"))'
```
"""


def _verify_declared_source_file(
    source_root: Path,
    files: Mapping[str, Any],
    relative: Path,
) -> Path:
    relative_name = relative.as_posix()
    record = files.get(relative_name)
    path = source_root / relative
    if (
        not isinstance(record, Mapping)
        or not path.is_file()
        or path.is_symlink()
        or _file_record(path) != dict(record)
    ):
        raise SO1HLDirectInteractiveError(
            f"Static source checksum is invalid for {relative_name}."
        )
    return path


def _source_plotting_contract(
    source_root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_plot_specification: Mapping[str, Any],
    figure_schema: str,
) -> Mapping[str, Any]:
    """Verify the producer-declared figure stage before using its orientation."""

    stage_manifests = manifest.get("stage_manifests")
    if not isinstance(stage_manifests, Mapping):
        raise SO1HLDirectInteractiveError(
            "Static source manifest lacks stage-manifest checksums."
        )
    figure_manifest_path = source_root / "figures" / "figure_manifest.json"
    if stage_manifests.get("figures") != _file_record(figure_manifest_path):
        raise SO1HLDirectInteractiveError(
            "Static source figure-stage manifest checksum changed."
        )
    figures = _read_json(
        figure_manifest_path, label="SO1 direct-hL figure-stage manifest"
    )
    try:
        _verify_self_hash(figures, label="SO1 direct-hL figure-stage manifest")
    except ValueError as exc:
        raise SO1HLDirectInteractiveError(
            "Static source figure-stage manifest self-checksum is invalid."
        ) from exc
    specification = figures.get("plot_specification")
    if any(
        (
            figures.get("schema") != figure_schema,
            figures.get("status") != "complete",
            specification != expected_plot_specification,
            not isinstance(figures.get("files"), Mapping),
        )
    ):
        raise SO1HLDirectInteractiveError(
            "Static source figure-stage contract is invalid."
        )
    assert isinstance(specification, Mapping)
    if any(
        (
            list(specification.get("grid_shape", ())) != [3, 5],
            specification.get("core_numbers_identifiable") is not True,
            specification.get("equal_aspect") is not True,
            specification.get("invert_y_axis") is not True,
            specification.get("coordinate_units") != "micrometres",
        )
    ):
        raise SO1HLDirectInteractiveError(
            "Static and interactive tissue-orientation contracts do not agree."
        )
    combined = manifest.get("combined_figures")
    expected_figure = SOURCE_FIGURE_RELATIVE_PATH.as_posix()
    if not isinstance(combined, Mapping) or combined.get("contextual_png") != (
        expected_figure
    ):
        raise SO1HLDirectInteractiveError(
            "Static source contextual PNG path is not the locked figure path."
        )
    figure_files = figures.get("files")
    assert isinstance(figure_files, Mapping)
    contextual_path = source_root / SOURCE_FIGURE_RELATIVE_PATH
    if figure_files.get(expected_figure) != _file_record(contextual_path):
        raise SO1HLDirectInteractiveError(
            "Static contextual PNG is not checksum-bound by the figure stage."
        )
    return specification


def load_verified_so1_direct_hl_source(
    source_root: str | Path,
    *,
    run_id: str,
) -> VerifiedSO1DirectHLSource:
    """Load and fail-closed verify the completed static direct-hL report."""

    root = Path(source_root).expanduser().resolve(strict=False)
    # The producer verifier is the authority for the complete extraction,
    # clustering, figure, provenance, and external-source bundle contract.  Do
    # not replace it with a partial consumer-side reconstruction.
    from .so1_model_embedding_clustering import (
        FIGURE_SCHEMA,
        spatial_plot_spec,
        verify_so1_model_embedding_clustering,
    )

    try:
        manifest = verify_so1_model_embedding_clustering(root)
    except ValueError as exc:
        raise SO1HLDirectInteractiveError(
            "Static source bundle failed authoritative producer verification/checksum "
            "validation."
        ) from exc
    if not isinstance(manifest, Mapping):
        raise SO1HLDirectInteractiveError(
            "Static source verifier did not return its final manifest."
        )
    if any(
        (
            manifest.get("schema") != SOURCE_SCHEMA,
            manifest.get("status") != "complete",
            str(manifest.get("run_id")) != run_id,
            tuple(manifest.get("core_order", ())) != SO1_CORE_NUMBERS,
            int(manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            manifest.get("core_cell_counts")
            != {
                str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                for number in SO1_CORE_NUMBERS
            },
        )
    ):
        raise SO1HLDirectInteractiveError(
            "Static source manifest identity or SO1 cohort coverage is invalid."
        )

    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        raise SO1HLDirectInteractiveError(
            "Static source manifest lacks its direct-clustering configuration."
        )
    if any(
        (
            configuration.get("pipeline") != SOURCE_PIPELINE_KIND,
            configuration.get("representations") != ["h0", "hL"],
            configuration.get("pca") is not False,
            configuration.get("mean_center") is not False,
            configuration.get("l2_normalize_for_cosine") is not True,
            configuration.get("distance_metric") != "cosine",
            int(configuration.get("n_neighbors", -1)) != DEFAULT_N_NEIGHBORS,
            not math.isclose(
                float(configuration.get("leiden_resolution", math.nan)),
                DEFAULT_LEIDEN_RESOLUTION,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            int(configuration.get("random_seed", -1)) != DEFAULT_RANDOM_SEED,
            configuration.get("contextual_label_prefix") != LABEL_PREFIX,
            tuple(configuration.get("joint_core_order", ())) != SO1_CORE_NUMBERS,
            int(configuration.get("joint_cell_count", -1))
            != EXPECTED_TOTAL_CELLS,
            configuration.get("independent_representation_graphs") is not True,
            configuration.get("spatial_training_graph_reused_for_clustering")
            is not False,
            configuration.get("dense_cell_by_cell_matrix_constructed") is not False,
            configuration.get("device") != "cpu",
        )
    ):
        raise SO1HLDirectInteractiveError(
            "Static source is not the locked direct-hL, no-PCA resolution-1.0 analysis."
        )
    _source_plotting_contract(
        root,
        manifest,
        expected_plot_specification=spatial_plot_spec(),
        figure_schema=FIGURE_SCHEMA,
    )

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SO1HLDirectInteractiveError(
            "Static source manifest lacks file checksums."
        )
    table_path = _verify_declared_source_file(
        root, files, SOURCE_TABLE_RELATIVE_PATH
    )
    palette_path = _verify_declared_source_file(
        root, files, SOURCE_PALETTE_RELATIVE_PATH
    )
    _verify_declared_source_file(root, files, SOURCE_FIGURE_RELATIVE_PATH)

    palette_document = _read_json(
        palette_path, label="SO1 direct-hL contextual palette"
    )
    colors = palette_document.get("colors")
    if not isinstance(colors, Mapping):
        raise SO1HLDirectInteractiveError(
            "Static contextual palette lacks a colors mapping."
        )
    palette = {str(key): str(value).upper() for key, value in colors.items()}
    frame = pd.read_parquet(table_path, columns=list(_SOURCE_TABLE_COLUMNS))
    validated = validate_interactive_frame(frame, palette)
    if len(validated) != EXPECTED_TOTAL_CELLS:
        raise SO1HLDirectInteractiveError(
            f"Expected {EXPECTED_TOTAL_CELLS} SO1 cells, found {len(validated)}."
        )
    observed_counts = validated.groupby("core_number", sort=False).size().to_dict()
    if observed_counts != EXPECTED_CELL_COUNTS_BY_CORE:
        raise SO1HLDirectInteractiveError(
            "Static table per-core counts differ from the locked SO1 cohort."
        )
    cluster_labels = tuple(sorted(palette, key=_cluster_sort_key))
    embedding_shapes = manifest.get("embedding_shapes")
    h_l_shape = (
        embedding_shapes.get("hL")
        if isinstance(embedding_shapes, Mapping)
        else None
    )
    if list(h_l_shape) != [EXPECTED_TOTAL_CELLS, EXPECTED_EMBEDDING_DIMENSION]:
        raise SO1HLDirectInteractiveError("Static source hL shape is invalid.")
    cluster_counts = manifest.get("cluster_counts")
    contextual_cluster_count = (
        cluster_counts.get("contextual")
        if isinstance(cluster_counts, Mapping)
        else None
    )
    if int(contextual_cluster_count or -1) != len(cluster_labels):
        raise SO1HLDirectInteractiveError(
            "Static final manifest cluster count differs from the contextual table."
        )
    size_range = manifest.get("cluster_size_ranges")
    contextual_range = (
        size_range.get("contextual") if isinstance(size_range, Mapping) else None
    )
    observed_sizes = validated["contextual_cluster"].value_counts()
    if list(contextual_range or ()) != [
        int(observed_sizes.min()),
        int(observed_sizes.max()),
    ]:
        raise SO1HLDirectInteractiveError(
            "Static contextual cluster-size range differs from the table."
        )
    return VerifiedSO1DirectHLSource(
        root=root,
        manifest=manifest,
        frame=validated,
        palette=palette,
        cluster_labels=cluster_labels,
    )


def _verify_interactive_manifest(
    output_root: Path,
    manifest: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_payload: Mapping[str, Any],
    source_frame: pd.DataFrame,
    source_palette: Mapping[str, str],
) -> None:
    try:
        _verify_self_hash(manifest, label="SO1 direct-hL interactive manifest")
    except ValueError as exc:
        raise SO1HLDirectInteractiveError(
            "Interactive manifest self-checksum is invalid."
        ) from exc
    cluster_count = int(manifest.get("cluster_count", -1))
    expected_labels = [f"{LABEL_PREFIX}{index}" for index in range(max(cluster_count, 0))]
    if any(
        (
            manifest.get("schema") != INTERACTIVE_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("run_id") != expected_run_id,
            manifest.get("pipeline_kind") != SOURCE_PIPELINE_KIND,
            manifest.get("representation") != SOURCE_REPRESENTATION,
            manifest.get("pca") is not False,
            manifest.get("mean_center") is not False,
            manifest.get("feature_projection") is not False,
            manifest.get("l2_normalize_for_cosine") is not True,
            tuple(manifest.get("core_order", ())) != SO1_CORE_NUMBERS,
            manifest.get("core_cell_counts")
            != {
                str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                for number in SO1_CORE_NUMBERS
            },
            int(manifest.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
            cluster_count <= 0,
            manifest.get("cluster_labels") != expected_labels,
            int(manifest.get("embedding_dimension", -1))
            != EXPECTED_EMBEDDING_DIMENSION,
            int(manifest.get("n_neighbors", -1)) != DEFAULT_N_NEIGHBORS,
            not math.isclose(
                float(manifest.get("leiden_resolution", math.nan)),
                DEFAULT_LEIDEN_RESOLUTION,
                rel_tol=0.0,
                abs_tol=0.0,
            ),
            manifest.get("coordinate_orientation") != COORDINATE_ORIENTATION,
        )
    ):
        raise SO1HLDirectInteractiveError(
            "SO1 direct-hL interactive manifest identity is invalid."
        )
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping) or any(
        (
            execution.get("visualization_only") is not True,
            execution.get("embedding_extraction") is not False,
            execution.get("neighbor_construction") is not False,
            execution.get("reclustering") is not False,
            execution.get("model_inference") is not False,
            execution.get("model_training") is not False,
            execution.get("registry_access") is not False,
            execution.get("gpu_used") is not False,
        )
    ):
        raise SO1HLDirectInteractiveError(
            "Interactive execution boundary changed."
        )
    files = manifest.get("files")
    required = {"README.md", HTML_FILENAME, ZIP_FILENAME}
    if not isinstance(files, Mapping) or set(files) != required:
        raise SO1HLDirectInteractiveError(
            "Interactive output checksum set changed."
        )
    on_disk = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.relative_to(output_root).as_posix() != "manifest.json"
    }
    if on_disk != required:
        raise SO1HLDirectInteractiveError(
            "Interactive directory contains unrecorded or missing files."
        )
    for relative, record in files.items():
        path = output_root / str(relative)
        if path.is_symlink() or not path.is_file() or _file_record(path) != dict(record):
            raise SO1HLDirectInteractiveError(
                f"Interactive output checksum changed: {relative}"
            )

    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise SO1HLDirectInteractiveError(
            "Interactive manifest lacks exact source checksums."
        )
    required_sources = {
        "analysis_manifest",
        "cluster_table",
        "palette",
        "static_png",
    }
    if set(source_artifacts) != required_sources or any(
        not isinstance(source_artifacts[name], Mapping) for name in required_sources
    ):
        raise SO1HLDirectInteractiveError(
            "Interactive source-checksum set changed."
        )

    render_receipt = manifest.get("render_receipt")
    zip_receipt = manifest.get("zip_receipt")
    if not isinstance(render_receipt, Mapping) or not isinstance(zip_receipt, Mapping):
        raise SO1HLDirectInteractiveError(
            "Interactive manifest lacks render or ZIP receipts."
        )
    try:
        _verify_self_hash(render_receipt, label="SO1 direct-hL HTML receipt")
    except ValueError as exc:
        raise SO1HLDirectInteractiveError(
            "Interactive HTML receipt self-checksum is invalid."
        ) from exc
    html_path = output_root / HTML_FILENAME
    zip_path = output_root / ZIP_FILENAME
    expected_payload_fields = [
        "numeric_core_number",
        "x_um",
        "y_um",
        "so1_contextual_cluster_code",
    ]
    if any(
        (
            render_receipt.get("browser_payload_fields") != expected_payload_fields,
            render_receipt.get("source_manifest_sha256")
            != source_artifacts["analysis_manifest"].get("sha256"),
            render_receipt.get("source_table_sha256")
            != source_artifacts["cluster_table"].get("sha256"),
            render_receipt.get("source_palette_sha256")
            != source_artifacts["palette"].get("sha256"),
            render_receipt.get("source_static_png_sha256")
            != source_artifacts["static_png"].get("sha256"),
            render_receipt.get("html_sha256") != sha256_file(html_path),
            int(render_receipt.get("html_size_bytes", -1))
            != int(html_path.stat().st_size),
            zip_receipt.get("member") != HTML_FILENAME,
            zip_receipt.get("member_sha256") != sha256_file(html_path),
            zip_receipt.get("zip_sha256") != sha256_file(zip_path),
            int(zip_receipt.get("zip_size_bytes", -1)) != int(zip_path.stat().st_size),
            int(zip_receipt.get("member_count", -1)) != 1,
        )
    ):
        raise SO1HLDirectInteractiveError(
            "Interactive render or ZIP receipt differs from its files/sources."
        )
    for flag in (
        "stable_identifiers_included",
        "cell_index_included",
        "expression_values_included",
        "embedding_values_included",
        "metadata_fields_included",
        "checkpoint_data_included",
        "neighbor_edges_included",
        "source_paths_included",
        "clinical_fields_included",
    ):
        if render_receipt.get(flag) is not False:
            raise SO1HLDirectInteractiveError(
                f"Interactive render privacy flag changed: {flag}."
            )

    html = html_path.read_text(encoding="utf-8")
    payload = _validate_rendered_html(
        html, frame=source_frame, palette=source_palette
    )
    if payload != dict(expected_payload):
        raise SO1HLDirectInteractiveError(
            "Browser payload differs from the exact verified source table."
        )
    if render_receipt.get("browser_payload_sha256") != _browser_payload_sha256(
        payload
    ):
        raise SO1HLDirectInteractiveError(
            "Interactive browser payload checksum changed."
        )
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        members = archive.infolist()
        if (
            archive.testzip() is not None
            or len(members) != 1
            or members[0].filename != HTML_FILENAME
            or members[0].date_time != (1980, 1, 1, 0, 0, 0)
        ):
            raise SO1HLDirectInteractiveError(
                "Interactive transfer ZIP is not deterministic and single-member."
            )
        archived = archive.read(HTML_FILENAME)
    if not hmac.compare_digest(
        hashlib.sha256(archived).hexdigest(), sha256_file(html_path)
    ):
        raise SO1HLDirectInteractiveError(
            "Archived and standalone HTML checksums differ."
        )


def _resolve_directory(value: str | Path, *, paths: ProjectPaths) -> Path:
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = paths.project_root / resolved
    return resolved.resolve(strict=False)


def run_so1_hl_direct_interactive(
    *,
    paths: ProjectPaths,
    run_id: str,
    source_report_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create or checksum-verify the completed offline SO1 direct-hL viewer."""

    selected_run = str(run_id).strip()
    if not selected_run:
        raise SO1HLDirectInteractiveError("run_id must not be empty.")
    source_root = (
        paths.report_root / "analyses" / DEFAULT_SOURCE_REPORT / selected_run
        if source_report_dir is None
        else _resolve_directory(source_report_dir, paths=paths)
    )
    source = load_verified_so1_direct_hl_source(source_root, run_id=selected_run)
    expected_payload = build_interactive_payload(source.frame, source.palette)
    source_manifest_path = source.root / "manifest.json"
    source_table_path = source.root / SOURCE_TABLE_RELATIVE_PATH
    source_palette_path = source.root / SOURCE_PALETTE_RELATIVE_PATH
    source_figure_path = source.root / SOURCE_FIGURE_RELATIVE_PATH
    expected_sources = {
        "analysis_manifest": _file_record(source_manifest_path),
        "cluster_table": _file_record(source_table_path),
        "palette": _file_record(source_palette_path),
        "static_png": _file_record(source_figure_path),
    }

    output_root = (
        paths.report_root / "analyses" / DEFAULT_OUTPUT_REPORT / selected_run
        if output_dir is None
        else _resolve_directory(output_dir, paths=paths)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(
            manifest_path, label="SO1 direct-hL interactive manifest"
        )
        _verify_interactive_manifest(
            output_root,
            manifest,
            expected_run_id=selected_run,
            expected_payload=expected_payload,
            source_frame=source.frame,
            source_palette=source.palette,
        )
        if manifest.get("source_artifacts") != expected_sources:
            raise SO1HLDirectInteractiveError(
                "Interactive viewer source checksums changed."
            )
    else:
        allowed_partial = {HTML_FILENAME, ZIP_FILENAME, "README.md"}
        unexpected = sorted(
            path.name
            for path in output_root.iterdir()
            if path.name not in allowed_partial
        )
        if unexpected:
            raise SO1HLDirectInteractiveError(
                f"Unexpected partial interactive outputs: {unexpected}"
            )
        html_path = output_root / HTML_FILENAME
        zip_path = output_root / ZIP_FILENAME
        render_receipt = render_interactive_html(
            frame=source.frame,
            palette=source.palette,
            output_path=html_path,
            source_table_path=source_table_path,
            source_manifest_path=source_manifest_path,
            source_palette_path=source_palette_path,
            source_figure_path=source_figure_path,
        )
        zip_receipt = _write_transfer_zip(html_path=html_path, zip_path=zip_path)
        _atomic_write_text(
            output_root / "README.md",
            _render_readme(
                run_id=selected_run,
                cluster_labels=list(source.cluster_labels),
            ),
        )
        manifest = _receipt_with_self_hash(
            {
                "schema": INTERACTIVE_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "run_id": selected_run,
                "analysis_scope": "so1_direct_contextual_hL_resolution_1p0_visualization_only",
                "pipeline_kind": SOURCE_PIPELINE_KIND,
                "representation": SOURCE_REPRESENTATION,
                "pca": False,
                "mean_center": False,
                "feature_projection": False,
                "l2_normalize_for_cosine": True,
                "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
                "n_neighbors": DEFAULT_N_NEIGHBORS,
                "leiden_resolution": DEFAULT_LEIDEN_RESOLUTION,
                "core_order": list(SO1_CORE_NUMBERS),
                "core_cell_counts": {
                    str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                    for number in SO1_CORE_NUMBERS
                },
                "point_count": EXPECTED_TOTAL_CELLS,
                "cluster_count": len(source.cluster_labels),
                "cluster_labels": list(source.cluster_labels),
                "coordinate_orientation": COORDINATE_ORIENTATION,
                "static_orientation": {
                    "grid_shape": [3, 5],
                    "equal_aspect": True,
                    "invert_y_axis": True,
                    "coordinate_units": "micrometres",
                },
                "interaction": {
                    "cluster_click_highlight": True,
                    "nonselected_opacity": 0.09,
                    "selected_draw_order": "last",
                    "show_all_and_escape_reset": True,
                    "zoom": "wheel",
                    "pan": "pointer_drag",
                    "hover_fields": [
                        "numeric_core_number",
                        "so1_contextual_cluster",
                        "rounded_x_um",
                        "rounded_y_um",
                    ],
                    "combined_png_export": True,
                    "combined_grid_shape": [3, 5],
                },
                "execution": {
                    "visualization_only": True,
                    "embedding_extraction": False,
                    "neighbor_construction": False,
                    "reclustering": False,
                    "model_inference": False,
                    "model_training": False,
                    "registry_access": False,
                    "gpu_used": False,
                },
                "sharing": {
                    "self_contained_offline_html": True,
                    "authorized_collaborators_only": True,
                    "exact_tissue_coordinates_included": True,
                    "stable_cell_identifiers_included": False,
                    "cell_index_included": False,
                    "expression_values_included": False,
                    "embedding_values_included": False,
                    "metadata_fields_included": False,
                    "checkpoint_data_included": False,
                    "neighbor_edges_included": False,
                    "source_paths_included": False,
                    "clinical_fields_included": False,
                },
                "interpretation": {
                    "model_derived_contextual_clusters": True,
                    "cell_types_established": False,
                    "signaling_established": False,
                    "biological_influence_established": False,
                    "causality_established": False,
                    "marker_and_pathology_validation_separate": True,
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
            expected_run_id=selected_run,
            expected_payload=expected_payload,
            source_frame=source.frame,
            source_palette=source.palette,
        )

    return {
        "status": "complete",
        "run_id": selected_run,
        "device": "none (visualization-only CPU workflow)",
        "pipeline_kind": SOURCE_PIPELINE_KIND,
        "pca": False,
        "mean_center": False,
        "feature_projection": False,
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
    "SO1HLDirectInteractiveError",
    "VerifiedSO1DirectHLSource",
    "build_interactive_payload",
    "deterministic_contextual_palette",
    "interactive_javascript",
    "load_verified_so1_direct_hl_source",
    "render_interactive_html",
    "run_so1_hl_direct_interactive",
    "validate_interactive_frame",
]
