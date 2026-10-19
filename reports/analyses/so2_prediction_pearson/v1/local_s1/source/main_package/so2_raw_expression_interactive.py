"""Offline interactive viewer for the completed SO2 raw-expression clusters.

The viewer is visualization-only.  It reads the checksum-verified expression
cluster table and categorical palette, then embeds only numeric core labels,
tissue coordinates, and cluster codes in one dependency-free HTML file.  It
never loads counts, PCA scores, neighbor edges, a model, or a GPU runtime.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
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
from .so2_raw_expression_clustering import (
    ANALYSIS_ID as SOURCE_ANALYSIS_ID,
    ANALYSIS_SCHEMA as SOURCE_ANALYSIS_SCHEMA,
    DEFAULT_LEIDEN_RESOLUTION,
    SO2RawExpressionClusteringError,
    _verify_final_manifest as _verify_source_manifest,
)


INTERACTIVE_SCHEMA = "so2_14core_raw_expression_interactive_v1"
DEFAULT_SOURCE_REPORT = "so2_14core_raw_expression_clustering"
DEFAULT_OUTPUT_REPORT = "so2_14core_raw_expression_interactive"
HTML_FILENAME = "raw_expression_leiden_resolution_1p0_spatial_14cores_interactive.html"
ZIP_FILENAME = "raw_expression_leiden_resolution_1p0_spatial_14cores_interactive.zip"
EXPECTED_CLUSTER_COUNT = 12
LABEL_PREFIX = "E"


class SO2RawExpressionInteractiveError(SO2RawExpressionClusteringError):
    """Raised when the raw-expression interactive-viewer contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cluster_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"E([0-9]+)", label)
    if match is None:
        raise SO2RawExpressionInteractiveError(
            f"Expression cluster label must have the form E<number>: {label!r}"
        )
    return int(match.group(1)), label


def _validate_hex_color(value: object, *, label: str) -> str:
    color = str(value)
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None:
        raise SO2RawExpressionInteractiveError(
            f"Palette color for {label} must be a six-digit hexadecimal color."
        )
    return color


def validate_interactive_frame(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> pd.DataFrame:
    """Validate ordered 14-core coverage and a contiguous expression label set."""

    required = {"core_number", "x_um", "y_um", "expression_cluster"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SO2RawExpressionInteractiveError(
            f"Interactive source table lacks required columns: {missing}"
        )
    if frame.empty:
        raise SO2RawExpressionInteractiveError("Interactive source table is empty.")
    if frame.loc[:, list(required)].isna().any().any():
        raise SO2RawExpressionInteractiveError(
            "Interactive source fields contain missing values."
        )

    validated = frame.copy(deep=False)
    observed_core_order = tuple(
        int(value) for value in validated["core_number"].drop_duplicates().tolist()
    )
    if observed_core_order != SO2_CORE_NUMBERS:
        raise SO2RawExpressionInteractiveError(
            "Interactive map requires all 14 SO2 cores in the locked order "
            f"{SO2_CORE_NUMBERS}; observed {observed_core_order}."
        )
    core_values = validated["core_number"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(core_values).all():
        raise SO2RawExpressionInteractiveError("Core numbers are non-finite.")
    for coordinate in ("x_um", "y_um"):
        values = validated[coordinate].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise SO2RawExpressionInteractiveError(
                f"Interactive coordinates contain non-finite {coordinate} values."
            )

    labels = validated["expression_cluster"].astype(str)
    observed_labels = sorted(set(labels.tolist()), key=_cluster_sort_key)
    expected_labels = [f"E{index}" for index in range(len(observed_labels))]
    if observed_labels != expected_labels:
        raise SO2RawExpressionInteractiveError(
            "Expression cluster labels must be contiguous from E0."
        )
    palette_keys = sorted((str(key) for key in palette), key=_cluster_sort_key)
    if palette_keys != observed_labels:
        missing_palette = sorted(set(observed_labels).difference(palette_keys))
        extra_palette = sorted(set(palette_keys).difference(observed_labels))
        raise SO2RawExpressionInteractiveError(
            "Palette must cover the complete expression cluster set "
            f"(missing={missing_palette}, extra={extra_palette})."
        )
    colors = [_validate_hex_color(palette[label], label=label) for label in palette_keys]
    if len({color.lower() for color in colors}) != len(colors):
        raise SO2RawExpressionInteractiveError(
            "Expression cluster palette colors must be unique."
        )

    if "global_cell_index" in validated:
        indices = validated["global_cell_index"]
        if indices.isna().any() or not indices.is_unique:
            raise SO2RawExpressionInteractiveError(
                "Global source row indices must be unique."
            )
    if "cell_key" in validated:
        keys = validated["cell_key"]
        if keys.isna().any() or not keys.is_unique:
            raise SO2RawExpressionInteractiveError(
                "Stable source keys must be unique."
            )
    return validated


def _encode_array(value: np.ndarray, *, dtype: str | np.dtype[Any]) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return base64.b64encode(memoryview(array).cast("B")).decode("ascii")


def build_interactive_payload(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> dict[str, Any]:
    """Encode only coordinates and expression-cluster codes for the browser."""

    validated = validate_interactive_frame(frame, palette)
    clusters = sorted(
        set(validated["expression_cluster"].astype(str)), key=_cluster_sort_key
    )
    canonical_palette = {label: str(palette[label]) for label in clusters}
    cluster_to_code = {label: index for index, label in enumerate(clusters)}
    if len(clusters) > np.iinfo(np.uint8).max + 1:
        raise SO2RawExpressionInteractiveError(
            "Too many expression clusters for Uint8 codes."
        )

    core_records: list[dict[str, Any]] = []
    observed_points = 0
    for core_number in SO2_CORE_NUMBERS:
        core = validated.loc[validated["core_number"] == core_number]
        x = np.ascontiguousarray(core["x_um"].to_numpy(dtype="<f8", copy=True))
        y = np.ascontiguousarray(core["y_um"].to_numpy(dtype="<f8", copy=True))
        codes = np.ascontiguousarray(
            core["expression_cluster"]
            .astype(str)
            .map(cluster_to_code)
            .to_numpy(dtype=np.uint8, copy=True)
        )
        if not (len(x) == len(y) == len(codes) == len(core)) or not len(core):
            raise SO2RawExpressionInteractiveError(
                f"Coordinate/label ordering mismatch for SO2 core {core_number}."
            )
        x_b64 = _encode_array(x, dtype="<f8")
        y_b64 = _encode_array(y, dtype="<f8")
        codes_b64 = _encode_array(codes, dtype=np.uint8)
        x_round_trip = np.frombuffer(
            base64.b64decode(x_b64, validate=True), dtype="<f8"
        )
        y_round_trip = np.frombuffer(
            base64.b64decode(y_b64, validate=True), dtype="<f8"
        )
        codes_round_trip = np.frombuffer(
            base64.b64decode(codes_b64, validate=True), dtype=np.uint8
        )
        if not (
            np.array_equal(x_round_trip, x)
            and np.array_equal(y_round_trip, y)
            and np.array_equal(codes_round_trip, codes)
        ):
            raise SO2RawExpressionInteractiveError(
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
        observed_points += len(core)
    if observed_points != len(validated):
        raise SO2RawExpressionInteractiveError(
            "Not every source row was encoded into the interactive payload."
        )
    return {
        "schema": INTERACTIVE_SCHEMA,
        "core_order": list(SO2_CORE_NUMBERS),
        "clusters": clusters,
        "palette": canonical_palette,
        "point_count": int(len(validated)),
        "coordinate_units": "micrometres",
        "coordinate_orientation": "low_y_at_top",
        "cores": core_records,
    }


def interactive_javascript() -> str:
    """Return the established dependency-free viewer with expression wording."""

    script = _shared_interactive_javascript()
    script = script.replace("contextual", "expression").replace(
        "Contextual", "Expression"
    )
    script = script.replace(
        "so2_hL_expression_clusters_", "so2_raw_expression_clusters_"
    )
    return script


def _cluster_buttons(frame: pd.DataFrame, palette: Mapping[str, str]) -> str:
    counts = frame["expression_cluster"].astype(str).value_counts()
    labels = sorted(palette, key=_cluster_sort_key)
    return "\n".join(
        (
            f'<button type="button" class="cluster-chip" data-cluster="{label}" '
            f'aria-pressed="false" style="--cluster-color:{palette[label]}">'
            '<span class="swatch" aria-hidden="true"></span>'
            f'<span>{label}</span><span class="cluster-count">{int(counts[label]):,}</span>'
            "</button>"
        )
        for label in labels
    )


def _core_panels(frame: pd.DataFrame) -> str:
    counts = frame.groupby("core_number", sort=False).size()
    panels: list[str] = []
    for core_number in SO2_CORE_NUMBERS:
        panels.append(
            f'<article class="core-panel" data-panel-core="{core_number}">'
            f"<h2>SO2 Core {core_number}</h2>"
            f'<div class="point-count">{int(counts.loc[core_number]):,} cells</div>'
            '<div class="canvas-wrap">'
            f'<canvas data-core="{core_number}" role="img" '
            f'aria-label="Interactive expression cluster map for SO2 Core {core_number}"></canvas>'
            '<div class="hover-card" hidden></div>'
            "</div></article>"
        )
    panels.append(
        '<aside class="legend-panel"><div><strong>Shared joint clusters</strong><br>'
        "Cluster colors and IDs are identical across all 14 tissue panels.<br>"
        "Use the controls to highlight a cluster.</div></aside>"
    )
    return "\n".join(panels)


def _render_html(frame: pd.DataFrame, palette: Mapping[str, str]) -> str:
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
<title>SO2 14-core raw-expression clusters</title>
<style>{_shared_interactive_css()}</style>
</head>
<body>
<header>
  <h1>SO2 14-core raw-expression clusters</h1>
  <p>Joint classical Leiden clustering at resolution 1.0. Click a cluster to keep it bright while fading the others; click it again, choose Show all, or press Escape to reset.</p>
  <p>Scroll to zoom, drag to pan, double-click a panel to reset it, and hover for core, cluster, and rounded tissue coordinates.</p>
</header>
<main class="workspace">
  <aside class="controls" aria-label="Cluster controls">
    <h2>Expression clusters</h2>
    <p class="instructions">These are expression-derived groups, not validated cell types. Counts are across all 14 cores.</p>
    <div class="cluster-list">{_cluster_buttons(frame, palette)}</div>
    <div class="action-row">
      <button type="button" class="action" id="show-all">Show all clusters</button>
      <button type="button" class="action" id="reset-views">Reset all spatial views</button>
      <button type="button" class="action primary" id="export-png">Export current view as PNG</button>
    </div>
    <p id="selection-status" aria-live="polite">Showing all expression clusters.</p>
    <p class="privacy-note"><strong>QC and sharing notice.</strong> E1 contains 94.9% of cells below 20 transcripts and is strongly count-depth-associated. These clusters require marker/pathology validation. This offline file contains exact cell-level tissue coordinates; share it only with authorized collaborators.</p>
  </aside>
  <section class="core-grid" aria-label="SO2 spatial core maps">{_core_panels(frame)}</section>
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
        "/workspace",
        "fetch(",
        "xmlhttprequest",
        "url(http",
    )
    lowered = html.lower()
    found = [token for token in prohibited if token.lower() in lowered]
    if found:
        raise SO2RawExpressionInteractiveError(
            f"Shareable HTML contains prohibited source or network fields: {found}"
        )
    if "cell_key" in frame and len(frame):
        first_key = str(frame["cell_key"].iloc[0])
        if first_key and first_key in html:
            raise SO2RawExpressionInteractiveError(
                "Shareable HTML contains a stable source key."
            )
    if "connect-src 'none'" not in html:
        raise SO2RawExpressionInteractiveError(
            "Shareable HTML lacks the offline network CSP."
        )
    if len(re.findall(r"<canvas\s+[^>]*data-core=", html)) != len(SO2_CORE_NUMBERS):
        raise SO2RawExpressionInteractiveError(
            "Shareable HTML lacks exactly 14 core canvases."
        )
    external_asset = re.search(
        r"<(?:script|img)\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=",
        html,
        flags=re.IGNORECASE,
    )
    if external_asset is not None:
        raise SO2RawExpressionInteractiveError(
            "Shareable HTML contains an external DOM asset."
        )


def render_interactive_html(
    *,
    frame: pd.DataFrame,
    palette: Mapping[str, str],
    output_path: str | Path,
    source_table_path: str | Path,
    source_manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Write the self-contained HTML and return its checksummed receipt."""

    output = Path(output_path)
    source_table = Path(source_table_path)
    source_manifest = Path(source_manifest_path)
    if not source_table.is_file() or not source_manifest.is_file():
        raise SO2RawExpressionInteractiveError(
            "Interactive source artifacts are missing."
        )
    validated = validate_interactive_frame(frame, palette)
    html = _render_html(validated, palette)
    _validate_rendered_html(html, frame=validated)
    _atomic_write_text(output, html)
    if output.read_text(encoding="utf-8") != html:
        raise SO2RawExpressionInteractiveError(
            "Interactive HTML failed its write/read check."
        )
    return _receipt_with_self_hash(
        {
            "schema": INTERACTIVE_SCHEMA,
            "status": "complete",
            "self_contained": True,
            "offline_network_policy": "connect-src-none",
            "point_count": int(len(validated)),
            "core_order": list(SO2_CORE_NUMBERS),
            "cluster_count": int(len(palette)),
            "html_sha256": sha256_file(output),
            "html_size_bytes": int(output.stat().st_size),
            "source_table_sha256": sha256_file(source_table),
            "source_manifest_sha256": sha256_file(source_manifest),
            "browser_payload_fields": [
                "numeric_core_number",
                "x_um",
                "y_um",
                "expression_cluster_code",
            ],
            "stable_identifiers_included": False,
            "expression_values_included": False,
            "pca_scores_included": False,
            "clinical_fields_included": False,
            "model_embeddings_included": False,
        }
    )


def _write_transfer_zip(*, html_path: Path, zip_path: Path) -> Mapping[str, Any]:
    """Create a deterministic, one-member transfer archive."""

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
            raise SO2RawExpressionInteractiveError(
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
            raise SO2RawExpressionInteractiveError(
                "Transfer ZIP member name is unsafe."
            )
        extracted = archive.read(member)
    if not hmac.compare_digest(
        hashlib.sha256(extracted).hexdigest(), sha256_file(html_path)
    ):
        raise SO2RawExpressionInteractiveError(
            "Transfer ZIP member differs from the HTML file."
        )
    return {
        "member": html_path.name,
        "member_sha256": sha256_file(html_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "deterministic_timestamp": "1980-01-01T00:00:00",
    }


def _render_readme(*, html_name: str, zip_name: str) -> str:
    return f"""# Interactive SO2 raw-expression cluster map

Open `{html_name}` in a current desktop browser. The HTML is self-contained and
needs no server or internet connection. `{zip_name}` contains the identical HTML
as its only member and is the convenient file to transfer.

## Interaction

- Click `E0` through `E11` to keep that joint expression cluster bright and fade
  every other cluster across all 14 cores.
- Click the selected cluster again, click **Show all clusters**, or press Escape
  to restore all colors.
- Scroll to zoom, drag to pan, double-click to reset a panel, and hover for only
  numeric core, cluster, and rounded tissue coordinates.
- **Export current view as PNG** saves the current combined 3 x 5 view.

This viewer reuses the completed classical raw-expression Leiden labels at
resolution 1.0. It performs no normalization, PCA, neighbor construction,
clustering, model inference, or training and does not use a GPU.

## Interpretation and sharing limits

The labels are expression-derived groups, not validated cell types. In
particular, `E1` contains 7,742 of the 8,157 cells below 20 transcripts (94.9%)
and is strongly associated with count depth. None of these clusters establishes
cell type, signaling, biological influence, or causality; marker-based and
pathological validation remains separate.

The HTML omits stable cell identifiers, expression values, PCA scores, neighbor
edges, model embeddings, clinical fields, and source filesystem paths. It does
contain exact cell-level tissue coordinates, so share it only with authorized
collaborators rather than publishing it openly.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\
  render-so2-raw-expression-interactive
```
"""


def _verify_interactive_manifest(
    output_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    _verify_self_hash(manifest, label="SO2 raw-expression interactive manifest")
    if any(
        (
            manifest.get("schema") != INTERACTIVE_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("source_analysis_id") != SOURCE_ANALYSIS_ID,
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
            int(manifest.get("cluster_count", -1)) != EXPECTED_CLUSTER_COUNT,
            float(manifest.get("leiden_resolution", math.nan))
            != DEFAULT_LEIDEN_RESOLUTION,
        )
    ):
        raise SO2RawExpressionInteractiveError(
            "Raw-expression interactive manifest identity is invalid."
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SO2RawExpressionInteractiveError(
            "Raw-expression interactive manifest lacks output checksums."
        )
    required = {"README.md", HTML_FILENAME, ZIP_FILENAME}
    if set(files) != required:
        raise SO2RawExpressionInteractiveError(
            "Interactive output set changed: "
            f"{sorted(set(files).symmetric_difference(required))}"
        )
    for relative, record in files.items():
        path = output_root / str(relative)
        if not path.is_file() or _file_record(path) != dict(record):
            raise SO2RawExpressionInteractiveError(
                f"Interactive output checksum changed: {relative}"
            )
    html_path = output_root / HTML_FILENAME
    zip_path = output_root / ZIP_FILENAME
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        if archive.testzip() is not None or archive.namelist() != [HTML_FILENAME]:
            raise SO2RawExpressionInteractiveError(
                "Interactive transfer ZIP is invalid."
            )
        archived = archive.read(HTML_FILENAME)
    if not hmac.compare_digest(
        hashlib.sha256(archived).hexdigest(), sha256_file(html_path)
    ):
        raise SO2RawExpressionInteractiveError(
            "Archived and standalone HTML checksums differ."
        )


def run_so2_raw_expression_interactive(
    *,
    paths: ProjectPaths,
    source_analysis_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the completed offline expression-cluster viewer."""

    selected_analysis = (
        SOURCE_ANALYSIS_ID
        if source_analysis_id is None
        else str(source_analysis_id)
    )
    if selected_analysis != SOURCE_ANALYSIS_ID:
        raise SO2RawExpressionInteractiveError(
            "The viewer is locked to the completed SO2 raw-expression analysis."
        )
    source_root = (
        paths.report_root
        / "analyses"
        / DEFAULT_SOURCE_REPORT
        / selected_analysis
    )
    source_manifest_path = source_root / "manifest.json"
    source_table_path = source_root / "tables" / "cell_expression_clusters.parquet"
    source_palette_path = source_root / "clustering" / "expression_palette.json"
    source_manifest = _read_json(
        source_manifest_path, label="SO2 raw-expression clustering manifest"
    )
    source_configuration = source_manifest.get("configuration")
    if not isinstance(source_configuration, Mapping):
        raise SO2RawExpressionInteractiveError(
            "Source clustering manifest lacks its locked configuration."
        )
    _verify_source_manifest(
        output_root=source_root,
        manifest=source_manifest,
        configuration=source_configuration,
    )
    if any(
        (
            source_manifest.get("schema") != SOURCE_ANALYSIS_SCHEMA,
            source_manifest.get("analysis_id") != SOURCE_ANALYSIS_ID,
            source_manifest.get("analysis_scope")
            != "classical_raw_expression_only_joint_clustering",
            int(source_manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(source_manifest.get("cluster_count", -1))
            != EXPECTED_CLUSTER_COUNT,
            tuple(source_manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            float(source_configuration.get("leiden_resolution", math.nan))
            != DEFAULT_LEIDEN_RESOLUTION,
        )
    ):
        raise SO2RawExpressionInteractiveError(
            "Source report is not the completed raw-expression resolution-1.0 analysis."
        )
    palette_document = _read_json(
        source_palette_path, label="SO2 raw-expression palette"
    )
    palette_value = palette_document.get("colors")
    if not isinstance(palette_value, Mapping):
        raise SO2RawExpressionInteractiveError(
            "Expression palette lacks a colors mapping."
        )
    palette = {str(key): str(value) for key, value in palette_value.items()}

    # Read only browser-relevant fields. Stable identifiers and QC covariates
    # never enter the HTML rendering process.
    frame = pd.read_parquet(
        source_table_path,
        columns=["core_number", "x_um", "y_um", "expression_cluster"],
    )
    validated = validate_interactive_frame(frame, palette)
    if len(validated) != EXPECTED_TOTAL_CELLS:
        raise SO2RawExpressionInteractiveError(
            f"Expected {EXPECTED_TOTAL_CELLS} cells, found {len(validated)}."
        )
    observed_counts = validated.groupby("core_number", sort=False).size().to_dict()
    if observed_counts != EXPECTED_CELL_COUNTS_BY_CORE:
        raise SO2RawExpressionInteractiveError(
            "Interactive source per-core counts differ from the completed report."
        )

    if output_dir is None:
        output_root = (
            paths.report_root
            / "analyses"
            / DEFAULT_OUTPUT_REPORT
            / selected_analysis
        )
    else:
        output_root = Path(output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = paths.project_root / output_root
        output_root = output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    expected_sources = {
        "analysis_manifest": _file_record(source_manifest_path),
        "cluster_table": _file_record(source_table_path),
        "palette": _file_record(source_palette_path),
    }
    if manifest_path.is_file():
        manifest = _read_json(
            manifest_path, label="SO2 raw-expression interactive manifest"
        )
        _verify_interactive_manifest(output_root, manifest)
        if manifest.get("source_artifacts") != expected_sources:
            raise SO2RawExpressionInteractiveError(
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
            raise SO2RawExpressionInteractiveError(
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
        )
        zip_receipt = _write_transfer_zip(html_path=html_path, zip_path=zip_path)
        _atomic_write_text(
            output_root / "README.md",
            _render_readme(html_name=HTML_FILENAME, zip_name=ZIP_FILENAME),
        )
        manifest = _receipt_with_self_hash(
            {
                "schema": INTERACTIVE_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "source_analysis_id": selected_analysis,
                "analysis_scope": "raw_expression_resolution_1p0_visualization_only",
                "leiden_resolution": DEFAULT_LEIDEN_RESOLUTION,
                "core_order": list(SO2_CORE_NUMBERS),
                "core_cell_counts": {
                    str(number): EXPECTED_CELL_COUNTS_BY_CORE[number]
                    for number in SO2_CORE_NUMBERS
                },
                "point_count": EXPECTED_TOTAL_CELLS,
                "cluster_count": EXPECTED_CLUSTER_COUNT,
                "interaction": {
                    "cluster_click_highlight": True,
                    "nonselected_opacity": 0.09,
                    "selected_draw_order": "last",
                    "show_all_and_escape_reset": True,
                    "zoom": "wheel",
                    "pan": "pointer_drag",
                    "hover_fields": [
                        "numeric_core_number",
                        "expression_cluster",
                        "rounded_x_um",
                        "rounded_y_um",
                    ],
                    "combined_png_export": True,
                },
                "execution": {
                    "visualization_only": True,
                    "normalization_or_pca": False,
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
                    "pca_scores_included": False,
                    "clinical_fields_included": False,
                    "model_embeddings_included": False,
                },
                "interpretation": {
                    "expression_derived_clusters": True,
                    "cell_types_established": False,
                    "signaling_established": False,
                    "biological_influence_established": False,
                    "causality_established": False,
                    "e1_low_count_depth_warning_visible": True,
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
        "source_analysis_id": selected_analysis,
        "device": "none (visualization-only CPU workflow)",
        "point_count": int(manifest["point_count"]),
        "cluster_count": int(manifest["cluster_count"]),
        "leiden_resolution": float(manifest["leiden_resolution"]),
        "core_order": list(manifest["core_order"]),
        "output_root": output_root.as_posix(),
        "html": (output_root / HTML_FILENAME).as_posix(),
        "transfer_zip": (output_root / ZIP_FILENAME).as_posix(),
        "manifest": manifest_path.as_posix(),
    }


__all__ = [
    "SO2RawExpressionInteractiveError",
    "build_interactive_payload",
    "interactive_javascript",
    "render_interactive_html",
    "run_so2_raw_expression_interactive",
    "validate_interactive_frame",
]
