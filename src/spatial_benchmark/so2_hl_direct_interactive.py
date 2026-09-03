"""Offline interactive viewer for direct-hL SO2 contextual clusters.

This visualization-only workflow consumes the finalized, checksum-verified
direct-hL clustering report.  It does not load embeddings, a checkpoint, the
training graph, or a GPU runtime.  The browser payload contains only numeric
core labels, tissue coordinates, and compact ``D`` cluster codes.
"""

from __future__ import annotations

import base64
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
)
from .so2_hl_interactive import (
    _interactive_css as _shared_interactive_css,
    interactive_javascript as _shared_interactive_javascript,
)
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_TOTAL_CELLS,
    SO2_CORE_NUMBERS,
)


INTERACTIVE_SCHEMA = "so2_14core_hl_direct_knn_interactive_v1"
SOURCE_PIPELINE_KIND = "direct_hl_cosine_knn_leiden"
DEFAULT_SOURCE_REPORT = "so2_14core_contextual_embedding_direct_knn_clustering"
DEFAULT_OUTPUT_REPORT = "so2_14core_hl_direct_knn_interactive"
HTML_FILENAME = (
    "contextual_direct_hl_leiden_resolution_1p0_spatial_14cores_interactive.html"
)
ZIP_FILENAME = (
    "contextual_direct_hl_leiden_resolution_1p0_spatial_14cores_interactive.zip"
)
SOURCE_TABLE_RELATIVE_PATH = Path("tables/cell_contextual_direct_clusters.parquet")
SOURCE_PALETTE_RELATIVE_PATH = Path("clustering/contextual_direct_palette.json")
SOURCE_FIGURE_RELATIVE_PATH = Path(
    "figures/contextual_direct_hl_leiden_resolution_1p0_spatial_14cores.png"
)
DEFAULT_N_NEIGHBORS = 30
DEFAULT_LEIDEN_RESOLUTION = 1.0
EXPECTED_EMBEDDING_DIMENSION = 256
LABEL_PREFIX = "D"


class SO2HLDirectInteractiveError(ValueError):
    """Raised when the direct-hL interactive-viewer contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cluster_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"D([0-9]+)", label)
    if match is None:
        raise SO2HLDirectInteractiveError(
            f"Direct-hL cluster label must have the form D<number>: {label!r}"
        )
    return int(match.group(1)), label


def _validate_hex_color(value: object, *, label: str) -> str:
    color = str(value)
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None:
        raise SO2HLDirectInteractiveError(
            f"Palette color for {label} must be a six-digit hexadecimal color."
        )
    return color


def validate_interactive_frame(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> pd.DataFrame:
    """Validate ordered core coverage and a dynamic contiguous ``D`` label set."""

    required = {
        "core_number",
        "x_um",
        "y_um",
        "contextual_direct_cluster",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SO2HLDirectInteractiveError(
            f"Interactive source table lacks required columns: {missing}"
        )
    if frame.empty:
        raise SO2HLDirectInteractiveError("Interactive source table is empty.")
    if frame.loc[:, sorted(required)].isna().any().any():
        raise SO2HLDirectInteractiveError(
            "Interactive source fields contain missing values."
        )

    validated = frame.copy(deep=False)
    observed_core_order = tuple(
        int(value) for value in validated["core_number"].drop_duplicates().tolist()
    )
    if observed_core_order != SO2_CORE_NUMBERS:
        raise SO2HLDirectInteractiveError(
            "Interactive map requires all 14 SO2 cores in the locked order "
            f"{SO2_CORE_NUMBERS}; observed {observed_core_order}."
        )
    core_values = validated["core_number"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(core_values).all():
        raise SO2HLDirectInteractiveError("Core numbers contain non-finite values.")
    if not np.array_equal(core_values, core_values.astype(np.int64)):
        raise SO2HLDirectInteractiveError("Core numbers must be integers.")

    for coordinate in ("x_um", "y_um"):
        values = validated[coordinate].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise SO2HLDirectInteractiveError(
                f"Interactive coordinates contain non-finite {coordinate} values."
            )

    labels = validated["contextual_direct_cluster"].astype(str)
    observed_labels = sorted(set(labels.tolist()), key=_cluster_sort_key)
    expected_labels = [f"D{index}" for index in range(len(observed_labels))]
    if observed_labels != expected_labels:
        raise SO2HLDirectInteractiveError(
            "Direct-hL cluster labels must be contiguous from D0."
        )
    palette_keys = sorted((str(key) for key in palette), key=_cluster_sort_key)
    if palette_keys != observed_labels:
        missing_palette = sorted(set(observed_labels).difference(palette_keys))
        extra_palette = sorted(set(palette_keys).difference(observed_labels))
        raise SO2HLDirectInteractiveError(
            "Palette must cover the complete direct-hL cluster set "
            f"(missing={missing_palette}, extra={extra_palette})."
        )
    colors = [_validate_hex_color(palette[label], label=label) for label in palette_keys]
    if len({color.lower() for color in colors}) != len(colors):
        raise SO2HLDirectInteractiveError(
            "Direct-hL cluster palette colors must be unique."
        )

    if "global_cell_index" in validated:
        indices = validated["global_cell_index"]
        if indices.isna().any() or not indices.is_unique:
            raise SO2HLDirectInteractiveError(
                "Global source row indices must be unique."
            )
    if "cell_key" in validated:
        keys = validated["cell_key"]
        if keys.isna().any() or not keys.is_unique:
            raise SO2HLDirectInteractiveError("Stable source keys must be unique.")
    return validated


def _encode_array(value: np.ndarray, *, dtype: str | np.dtype[Any]) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return base64.b64encode(memoryview(array).cast("B")).decode("ascii")


def build_interactive_payload(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> dict[str, Any]:
    """Encode only coordinates and direct-hL cluster codes for the browser."""

    validated = validate_interactive_frame(frame, palette)
    clusters = sorted(
        set(validated["contextual_direct_cluster"].astype(str)),
        key=_cluster_sort_key,
    )
    if len(clusters) > np.iinfo(np.uint8).max + 1:
        raise SO2HLDirectInteractiveError(
            "Too many direct-hL clusters for Uint8 browser codes."
        )
    canonical_palette = {label: str(palette[label]) for label in clusters}
    cluster_to_code = {label: index for index, label in enumerate(clusters)}

    core_records: list[dict[str, Any]] = []
    encoded_points = 0
    for core_number in SO2_CORE_NUMBERS:
        core = validated.loc[validated["core_number"] == core_number]
        if core.empty:
            raise SO2HLDirectInteractiveError(
                f"SO2 core {core_number} has no interactive points."
            )
        x = np.ascontiguousarray(core["x_um"].to_numpy(dtype="<f8", copy=True))
        y = np.ascontiguousarray(core["y_um"].to_numpy(dtype="<f8", copy=True))
        codes = np.ascontiguousarray(
            core["contextual_direct_cluster"]
            .astype(str)
            .map(cluster_to_code)
            .to_numpy(dtype=np.uint8, copy=True)
        )
        if not (len(x) == len(y) == len(codes) == len(core)):
            raise SO2HLDirectInteractiveError(
                f"Coordinate/label ordering mismatch for SO2 core {core_number}."
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
            raise SO2HLDirectInteractiveError(
                f"Browser payload failed round-trip validation for core {core_number}."
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
        encoded_points += len(core)
    if encoded_points != len(validated):
        raise SO2HLDirectInteractiveError(
            "Not every source row was encoded into the interactive payload."
        )
    return {
        "schema": INTERACTIVE_SCHEMA,
        "method": SOURCE_PIPELINE_KIND,
        "core_order": list(SO2_CORE_NUMBERS),
        "clusters": clusters,
        "palette": canonical_palette,
        "point_count": int(len(validated)),
        "coordinate_units": "micrometres",
        "coordinate_orientation": "low_y_at_top",
        "cores": core_records,
    }


def interactive_javascript() -> str:
    """Return the established dependency-free viewer with direct-hL wording."""

    return (
        _shared_interactive_javascript()
        .replace("Showing all contextual clusters.", "Showing all direct hL clusters.")
        .replace("All contextual clusters", "All direct hL clusters")
        .replace("so2_hL_contextual_clusters_", "so2_direct_hL_clusters_")
    )


def _cluster_buttons(frame: pd.DataFrame, palette: Mapping[str, str]) -> str:
    counts = frame["contextual_direct_cluster"].astype(str).value_counts()
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
    panels = []
    for core_number in SO2_CORE_NUMBERS:
        panels.append(
            f'<article class="core-panel" data-panel-core="{core_number}">'
            f"<h2>SO2 Core {core_number}</h2>"
            f'<div class="point-count">{int(counts.loc[core_number]):,} cells</div>'
            '<div class="canvas-wrap">'
            f'<canvas data-core="{core_number}" role="img" '
            f'aria-label="Interactive direct hL cluster map for SO2 Core {core_number}"></canvas>'
            '<div class="hover-card" hidden></div>'
            "</div></article>"
        )
    panels.append(
        '<aside class="legend-panel"><div><strong>Shared joint direct-hL clusters</strong><br>'
        "Cluster colors and D labels are identical across all 14 tissue panels.<br>"
        "Use the controls to highlight a cluster.</div></aside>"
    )
    return "\n".join(panels)


def _render_html(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
    *,
    embedding_dimension: int,
    n_neighbors: int,
    leiden_resolution: float,
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
<title>SO2 14-core direct hL clusters</title>
<style>{_shared_interactive_css()}</style>
</head>
<body>
<header>
  <h1>SO2 14-core direct hL clusters</h1>
  <p>Direct {embedding_dimension}-dimensional hL &rarr; L2 normalization &rarr; cosine {n_neighbors}-nearest-neighbor graph &rarr; Leiden clustering at resolution {leiden_resolution:g}. PCA and mean-centering were not used.</p>
  <p>Click a D cluster to keep it bright while fading the others. Scroll to zoom, drag to pan, double-click a panel to reset it, and hover for core, cluster, and rounded tissue coordinates.</p>
</header>
<main class="workspace">
  <aside class="controls" aria-label="Cluster controls">
    <h2>Direct-hL clusters</h2>
    <p class="instructions">These are model-derived groups, not validated cell types. Counts are across all 14 cores. D labels are distinct from the earlier PCA-derived C clusters.</p>
    <div class="cluster-list">{_cluster_buttons(frame, palette)}</div>
    <div class="action-row">
      <button type="button" class="action" id="show-all">Show all clusters</button>
      <button type="button" class="action" id="reset-views">Reset all spatial views</button>
      <button type="button" class="action primary" id="export-png">Export current view as PNG</button>
    </div>
    <p id="selection-status" aria-live="polite">Showing all direct hL clusters.</p>
    <p class="privacy-note"><strong>Authorized sharing only.</strong> This offline file contains exact cell-level tissue coordinates. It contains no stable cell identifiers, expression values, hL vectors, clinical labels, neighbor edges, or external network resources.</p>
  </aside>
  <section class="core-grid" aria-label="SO2 direct hL spatial cluster maps">{_core_panels(frame)}</section>
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
        "contextual_direct_cluster_number",
        "expression_values",
        "embedding_values",
        "neighbor_indices",
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
        raise SO2HLDirectInteractiveError(
            f"Shareable HTML contains prohibited source or network fields: {found}"
        )
    if "cell_key" in frame and len(frame):
        first_key = str(frame["cell_key"].iloc[0])
        if first_key and first_key in html:
            raise SO2HLDirectInteractiveError(
                "Shareable HTML contains a stable source key."
            )
    if "connect-src 'none'" not in html:
        raise SO2HLDirectInteractiveError(
            "Shareable HTML lacks the offline network CSP."
        )
    if len(re.findall(r"<canvas\s+[^>]*data-core=", html)) != len(SO2_CORE_NUMBERS):
        raise SO2HLDirectInteractiveError(
            "Shareable HTML lacks exactly 14 core canvases."
        )
    if re.search(
        r"<(?:script|img)\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=",
        html,
        flags=re.IGNORECASE,
    ) is not None:
        raise SO2HLDirectInteractiveError(
            "Shareable HTML contains an external DOM asset."
        )


def render_interactive_html(
    *,
    frame: pd.DataFrame,
    palette: Mapping[str, str],
    output_path: str | Path,
    source_table_path: str | Path,
    source_manifest_path: str | Path,
    embedding_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    leiden_resolution: float = DEFAULT_LEIDEN_RESOLUTION,
) -> Mapping[str, Any]:
    """Write one self-contained HTML viewer and return its checksum receipt."""

    if embedding_dimension <= 0 or n_neighbors <= 0 or leiden_resolution <= 0:
        raise SO2HLDirectInteractiveError(
            "Embedding dimension, kNN neighbors, and Leiden resolution must be positive."
        )
    output = Path(output_path)
    source_table = Path(source_table_path)
    source_manifest = Path(source_manifest_path)
    if not source_table.is_file() or not source_manifest.is_file():
        raise SO2HLDirectInteractiveError("Interactive source artifacts are missing.")
    validated = validate_interactive_frame(frame, palette)
    html = _render_html(
        validated,
        palette,
        embedding_dimension=int(embedding_dimension),
        n_neighbors=int(n_neighbors),
        leiden_resolution=float(leiden_resolution),
    )
    _validate_rendered_html(html, frame=validated)
    _atomic_write_text(output, html)
    if output.read_text(encoding="utf-8") != html:
        raise SO2HLDirectInteractiveError(
            "Interactive HTML failed its write/read check."
        )
    return _receipt_with_self_hash(
        {
            "schema": INTERACTIVE_SCHEMA,
            "status": "complete",
            "pipeline_kind": SOURCE_PIPELINE_KIND,
            "pca": False,
            "mean_center": False,
            "l2_normalize_for_cosine": True,
            "self_contained": True,
            "offline_network_policy": "connect-src-none",
            "point_count": int(len(validated)),
            "core_order": list(SO2_CORE_NUMBERS),
            "cluster_count": int(len(palette)),
            "embedding_dimension": int(embedding_dimension),
            "n_neighbors": int(n_neighbors),
            "leiden_resolution": float(leiden_resolution),
            "html_sha256": sha256_file(output),
            "html_size_bytes": int(output.stat().st_size),
            "source_table_sha256": sha256_file(source_table),
            "source_manifest_sha256": sha256_file(source_manifest),
            "browser_payload_fields": [
                "numeric_core_number",
                "x_um",
                "y_um",
                "contextual_direct_cluster_code",
            ],
            "stable_identifiers_included": False,
            "expression_values_included": False,
            "embedding_values_included": False,
            "neighbor_edges_included": False,
            "clinical_fields_included": False,
        }
    )


def _write_transfer_zip(*, html_path: Path, zip_path: Path) -> Mapping[str, Any]:
    """Create and verify a deterministic single-member transfer archive."""

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
            raise SO2HLDirectInteractiveError("Transfer ZIP integrity check failed.")
        member = members[0]
        member_path = Path(member.filename)
        if (
            member.filename != html_path.name
            or member_path.name != member.filename
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise SO2HLDirectInteractiveError("Transfer ZIP member name is unsafe.")
        archived = archive.read(member)
    if not hmac.compare_digest(
        hashlib.sha256(archived).hexdigest(), sha256_file(html_path)
    ):
        raise SO2HLDirectInteractiveError(
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
    run_id: str,
    html_name: str,
    zip_name: str,
    cluster_count: int,
    embedding_dimension: int,
    n_neighbors: int,
    leiden_resolution: float,
) -> str:
    final_label = f"D{cluster_count - 1}"
    return f"""# Interactive SO2 direct-hL cluster map

Open `{html_name}` in a current desktop browser. It is a self-contained offline
file; `{zip_name}` contains the identical HTML as its only archive member.

## Method and interaction

The joint analysis used the direct {embedding_dimension}-dimensional final
contextual representation (`hL`), L2 normalization, a sparse cosine
{n_neighbors}-nearest-neighbor graph, and Leiden resolution
{leiden_resolution:g}. It used **no PCA and no mean-centering**. `D0` through
`{final_label}` are a separate label space and must not be equated with the
earlier PCA-derived `C` clusters.

- Click a D cluster to keep it bright and fade all others across every core.
- Click it again, choose **Show all clusters**, or press Escape to reset.
- Scroll to zoom, drag to pan, double-click to reset a panel, and hover for only
  numeric core, D cluster, and rounded tissue coordinates.
- **Export current view as PNG** saves the current combined 3 x 5 view.

This viewer reuses completed assignments for run `{run_id}`. It performs no
embedding extraction, neighbor construction, clustering, inference, training,
or GPU work.

## Interpretation and sharing limits

These are model-derived contextual groups, not validated cell types. They do
not independently establish cell type, signaling, biological influence, or
causality. Marker-based and pathological validation remains separate.

The HTML omits stable cell identifiers, expression values, hL vectors, neighbor
edges, clinical fields, and source filesystem paths. It contains exact
cell-level tissue coordinates, so share it only with authorized collaborators.

## Reproduction

From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\
  render-so2-hl-direct-interactive \\
  --run-id {run_id}
```
"""


def _verify_interactive_manifest(
    output_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    _verify_self_hash(manifest, label="SO2 direct-hL interactive manifest")
    cluster_count = int(manifest.get("cluster_count", -1))
    expected_labels = [f"D{index}" for index in range(max(0, cluster_count))]
    observed_core_counts = manifest.get("core_cell_counts")
    if any(
        (
            manifest.get("schema") != INTERACTIVE_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("pipeline_kind") != SOURCE_PIPELINE_KIND,
            manifest.get("pca") is not False,
            manifest.get("mean_center") is not False,
            manifest.get("l2_normalize_for_cosine") is not True,
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
            cluster_count <= 0,
            manifest.get("cluster_labels") != expected_labels,
            observed_core_counts
            != {
                str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                for number in SO2_CORE_NUMBERS
            },
            int(manifest.get("embedding_dimension", -1))
            != EXPECTED_EMBEDDING_DIMENSION,
            int(manifest.get("n_neighbors", -1)) != DEFAULT_N_NEIGHBORS,
            not math.isclose(
                float(manifest.get("leiden_resolution", math.nan)),
                DEFAULT_LEIDEN_RESOLUTION,
            ),
        )
    ):
        raise SO2HLDirectInteractiveError(
            "Direct-hL interactive manifest identity is invalid."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SO2HLDirectInteractiveError(
            "Direct-hL interactive manifest lacks output checksums."
        )
    required = {"README.md", HTML_FILENAME, ZIP_FILENAME}
    if set(files) != required:
        raise SO2HLDirectInteractiveError(
            "Interactive output set changed: "
            f"{sorted(set(files).symmetric_difference(required))}"
        )
    for relative, record in files.items():
        path = output_root / str(relative)
        if not path.is_file() or _file_record(path) != dict(record):
            raise SO2HLDirectInteractiveError(
                f"Interactive output checksum changed: {relative}"
            )
    render_receipt = manifest.get("render_receipt")
    zip_receipt = manifest.get("zip_receipt")
    if not isinstance(render_receipt, Mapping) or not isinstance(zip_receipt, Mapping):
        raise SO2HLDirectInteractiveError(
            "Interactive manifest lacks render or ZIP checksum receipts."
        )
    _verify_self_hash(render_receipt, label="SO2 direct-hL HTML receipt")
    html_path = output_root / HTML_FILENAME
    zip_path = output_root / ZIP_FILENAME
    if any(
        (
            render_receipt.get("html_sha256") != sha256_file(html_path),
            int(render_receipt.get("html_size_bytes", -1))
            != int(html_path.stat().st_size),
            zip_receipt.get("member") != HTML_FILENAME,
            zip_receipt.get("member_sha256") != sha256_file(html_path),
            zip_receipt.get("zip_sha256") != sha256_file(zip_path),
            int(zip_receipt.get("zip_size_bytes", -1)) != int(zip_path.stat().st_size),
        )
    ):
        raise SO2HLDirectInteractiveError(
            "Interactive render or ZIP receipt does not match the output files."
        )
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        if archive.testzip() is not None or archive.namelist() != [HTML_FILENAME]:
            raise SO2HLDirectInteractiveError("Interactive transfer ZIP is invalid.")
        archived = archive.read(HTML_FILENAME)
    if not hmac.compare_digest(
        hashlib.sha256(archived).hexdigest(), sha256_file(html_path)
    ):
        raise SO2HLDirectInteractiveError(
            "Archived and standalone HTML checksums differ."
        )


def _source_parameter(manifest: Mapping[str, Any], key: str) -> object:
    if key in manifest:
        return manifest[key]
    for section_name in ("analysis_parameters", "clustering_parameters", "configuration"):
        section = manifest.get(section_name)
        if isinstance(section, Mapping) and key in section:
            return section[key]
    raise SO2HLDirectInteractiveError(
        f"Source direct-hL manifest lacks required parameter {key!r}."
    )


def _resolve_directory(
    value: str | Path,
    *,
    paths: ProjectPaths,
) -> Path:
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = paths.project_root / resolved
    return resolved.resolve(strict=False)


def run_so2_hl_direct_interactive(
    *,
    paths: ProjectPaths,
    run_id: str,
    source_report_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the direct-hL offline viewer bundle."""

    selected_run = str(run_id).strip()
    if not selected_run:
        raise SO2HLDirectInteractiveError("run_id must not be empty.")
    source_root = (
        paths.report_root / "analyses" / DEFAULT_SOURCE_REPORT / selected_run
        if source_report_dir is None
        else _resolve_directory(source_report_dir, paths=paths)
    )

    # Import lazily so low-level HTML tests remain independent of model and
    # clustering dependencies.  The source verifier checks every source file.
    from .so2_hl_direct_clustering import load_verified_direct_hl_analysis

    source_manifest = load_verified_direct_hl_analysis(source_root)
    if not isinstance(source_manifest, Mapping):
        raise SO2HLDirectInteractiveError(
            "Verified direct-hL source loader did not return a manifest."
        )
    source_manifest_path = source_root / "manifest.json"
    source_table_path = source_root / SOURCE_TABLE_RELATIVE_PATH
    source_palette_path = source_root / SOURCE_PALETTE_RELATIVE_PATH
    source_figure_path = source_root / SOURCE_FIGURE_RELATIVE_PATH

    if str(source_manifest.get("run_id")) != selected_run:
        raise SO2HLDirectInteractiveError("Source report run ID does not match.")
    if _source_parameter(source_manifest, "pipeline_kind") != SOURCE_PIPELINE_KIND:
        raise SO2HLDirectInteractiveError(
            "Source report is not direct-hL cosine-kNN Leiden."
        )
    if any(
        (
            _source_parameter(source_manifest, "pca") is not False,
            _source_parameter(source_manifest, "mean_center") is not False,
            _source_parameter(source_manifest, "l2_normalize_for_cosine") is not True,
        )
    ):
        raise SO2HLDirectInteractiveError(
            "Source method flags do not prove direct hL with no PCA/mean-centering."
        )
    embedding_dimension = int(
        _source_parameter(source_manifest, "embedding_dimension")
    )
    n_neighbors = int(_source_parameter(source_manifest, "n_neighbors"))
    leiden_resolution = float(
        _source_parameter(source_manifest, "leiden_resolution")
    )
    if any(
        (
            embedding_dimension != EXPECTED_EMBEDDING_DIMENSION,
            n_neighbors != DEFAULT_N_NEIGHBORS,
            not math.isclose(leiden_resolution, DEFAULT_LEIDEN_RESOLUTION),
            tuple(source_manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(source_manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
        )
    ):
        raise SO2HLDirectInteractiveError(
            "Source report is not the requested 256D, k=30, resolution-1.0 SO2 analysis."
        )

    palette_document = _read_json(
        source_palette_path, label="SO2 direct-hL palette"
    )
    palette_value = palette_document.get("colors")
    if not isinstance(palette_value, Mapping):
        raise SO2HLDirectInteractiveError(
            "Direct-hL palette lacks a colors mapping."
        )
    palette = {str(key): str(value) for key, value in palette_value.items()}
    frame = pd.read_parquet(
        source_table_path,
        columns=["core_number", "x_um", "y_um", "contextual_direct_cluster"],
    )
    validated = validate_interactive_frame(frame, palette)
    if len(validated) != EXPECTED_TOTAL_CELLS:
        raise SO2HLDirectInteractiveError(
            f"Expected {EXPECTED_TOTAL_CELLS} cells, found {len(validated)}."
        )
    observed_counts = validated.groupby("core_number", sort=False).size().to_dict()
    if observed_counts != EXPECTED_CELL_COUNTS_BY_CORE:
        raise SO2HLDirectInteractiveError(
            "Interactive source per-core counts differ from the completed SO2 cohort."
        )
    if int(source_manifest.get("cluster_count", -1)) != len(palette):
        raise SO2HLDirectInteractiveError(
            "Source manifest, table, and palette disagree on cluster count."
        )

    output_root = (
        paths.report_root / "analyses" / DEFAULT_OUTPUT_REPORT / selected_run
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
            manifest_path, label="SO2 direct-hL interactive manifest"
        )
        _verify_interactive_manifest(output_root, manifest)
        if manifest.get("run_id") != selected_run:
            raise SO2HLDirectInteractiveError(
                "Existing interactive output belongs to a different run."
            )
        if manifest.get("source_artifacts") != expected_sources:
            raise SO2HLDirectInteractiveError(
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
            raise SO2HLDirectInteractiveError(
                f"Unexpected partial interactive outputs: {unexpected}"
            )
        html_path = output_root / HTML_FILENAME
        zip_path = output_root / ZIP_FILENAME
        receipt = render_interactive_html(
            frame=validated,
            palette=palette,
            output_path=html_path,
            source_table_path=source_table_path,
            source_manifest_path=source_manifest_path,
            embedding_dimension=embedding_dimension,
            n_neighbors=n_neighbors,
            leiden_resolution=leiden_resolution,
        )
        zip_receipt = _write_transfer_zip(html_path=html_path, zip_path=zip_path)
        _atomic_write_text(
            output_root / "README.md",
            _render_readme(
                run_id=selected_run,
                html_name=HTML_FILENAME,
                zip_name=ZIP_FILENAME,
                cluster_count=len(palette),
                embedding_dimension=embedding_dimension,
                n_neighbors=n_neighbors,
                leiden_resolution=leiden_resolution,
            ),
        )
        manifest = _receipt_with_self_hash(
            {
                "schema": INTERACTIVE_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "run_id": selected_run,
                "analysis_scope": "direct_contextual_hL_resolution_1p0_visualization_only",
                "pipeline_kind": SOURCE_PIPELINE_KIND,
                "pca": False,
                "mean_center": False,
                "l2_normalize_for_cosine": True,
                "embedding_dimension": embedding_dimension,
                "n_neighbors": n_neighbors,
                "leiden_resolution": leiden_resolution,
                "core_order": list(SO2_CORE_NUMBERS),
                "core_cell_counts": {
                    str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                    for number in SO2_CORE_NUMBERS
                },
                "point_count": EXPECTED_TOTAL_CELLS,
                "cluster_count": len(palette),
                "cluster_labels": sorted(palette, key=_cluster_sort_key),
                "interaction": {
                    "cluster_click_highlight": True,
                    "nonselected_opacity": 0.09,
                    "selected_draw_order": "last",
                    "show_all_and_escape_reset": True,
                    "zoom": "wheel",
                    "pan": "pointer_drag",
                    "hover_fields": [
                        "numeric_core_number",
                        "contextual_direct_cluster",
                        "rounded_x_um",
                        "rounded_y_um",
                    ],
                    "combined_png_export": True,
                },
                "execution": {
                    "visualization_only": True,
                    "embedding_extraction": False,
                    "neighbor_construction": False,
                    "reclustering": False,
                    "model_inference": False,
                    "model_training": False,
                    "gpu_used": False,
                },
                "sharing": {
                    "self_contained_offline_html": True,
                    "authorized_collaborators_only": True,
                    "exact_tissue_coordinates_included": True,
                    "stable_cell_identifiers_included": False,
                    "expression_values_included": False,
                    "embedding_values_included": False,
                    "neighbor_edges_included": False,
                    "clinical_fields_included": False,
                },
                "interpretation": {
                    "model_derived_contextual_clusters": True,
                    "cell_types_established": False,
                    "signaling_established": False,
                    "biological_influence_established": False,
                    "causality_established": False,
                },
                "source_artifacts": expected_sources,
                "render_receipt": dict(receipt),
                "zip_receipt": dict(zip_receipt),
                "files": _file_manifest(output_root),
            }
        )
        _atomic_write_json(manifest_path, manifest)
        _verify_interactive_manifest(output_root, manifest)

    return {
        "status": "complete",
        "run_id": selected_run,
        "device": "none (visualization-only CPU workflow)",
        "pipeline_kind": SOURCE_PIPELINE_KIND,
        "pca": False,
        "mean_center": False,
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
    "SO2HLDirectInteractiveError",
    "build_interactive_payload",
    "interactive_javascript",
    "render_interactive_html",
    "run_so2_hl_direct_interactive",
    "validate_interactive_frame",
]
