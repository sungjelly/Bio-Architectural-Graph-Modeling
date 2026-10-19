"""Offline interactive spatial viewer for the completed SO2 hL clustering.

The viewer is deliberately visualization-only: it reuses the checksum-verified
resolution-1.0 cluster table and never loads a checkpoint, embedding matrix, or
GPU runtime.  Its shareable HTML contains only numeric core labels, tissue
coordinates, cluster codes, and the established categorical palette.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import zipfile
from typing import Any, Mapping

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
from .so2_hl_clustering import (
    ANALYSIS_SCHEMA as SOURCE_ANALYSIS_SCHEMA,
    DEFAULT_LEIDEN_RESOLUTION,
    EXPECTED_RUN_ID,
    SO2HLClusteringError,
    _verify_final_manifest as _verify_source_manifest,
)
from .so2_pooled_full_core import (
    EXPECTED_CELL_COUNTS_BY_CORE,
    EXPECTED_TOTAL_CELLS,
    SO2_CORE_NUMBERS,
)


INTERACTIVE_SCHEMA = "so2_14core_hl_interactive_v1"
DEFAULT_SOURCE_REPORT = "so2_14core_contextual_embedding_clustering"
DEFAULT_OUTPUT_REPORT = "so2_14core_hl_interactive"
HTML_FILENAME = "contextual_leiden_resolution_1p0_spatial_14cores_interactive.html"
ZIP_FILENAME = "contextual_leiden_resolution_1p0_spatial_14cores_interactive.zip"
EXPECTED_CLUSTER_COUNT = 19


class SO2HLInteractiveError(SO2HLClusteringError):
    """Raised when the interactive-viewer contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cluster_sort_key(label: str) -> tuple[int, str]:
    match = re.fullmatch(r"C([0-9]+)", label)
    if match is None:
        raise SO2HLInteractiveError(
            f"Contextual cluster label must have the form C<number>: {label!r}"
        )
    return int(match.group(1)), label


def _validate_hex_color(value: object, *, label: str) -> str:
    color = str(value)
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", color) is None:
        raise SO2HLInteractiveError(
            f"Palette color for {label} must be a six-digit hexadecimal color."
        )
    return color


def validate_interactive_frame(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> pd.DataFrame:
    """Validate exact 14-core coverage and the joint contextual label space.

    Source identity columns are checked when present, but they are never copied
    into the browser payload.
    """

    required = {"core_number", "x_um", "y_um", "contextual_cluster"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SO2HLInteractiveError(
            f"Interactive source table lacks required columns: {missing}"
        )
    if frame.empty:
        raise SO2HLInteractiveError("Interactive source table is empty.")

    validated = frame.copy(deep=False)
    observed_core_order = tuple(
        int(value) for value in validated["core_number"].drop_duplicates().tolist()
    )
    if observed_core_order != SO2_CORE_NUMBERS:
        raise SO2HLInteractiveError(
            "Interactive map requires all 14 SO2 cores in the locked order "
            f"{SO2_CORE_NUMBERS}; observed {observed_core_order}."
        )

    core_values = validated["core_number"].to_numpy()
    if not np.isfinite(core_values.astype(np.float64, copy=False)).all():
        raise SO2HLInteractiveError("Core numbers contain non-finite values.")
    for coordinate in ("x_um", "y_um"):
        values = validated[coordinate].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all():
            raise SO2HLInteractiveError(
                f"Interactive coordinates contain non-finite {coordinate} values."
            )

    labels = validated["contextual_cluster"].astype(str)
    if labels.isna().any():
        raise SO2HLInteractiveError("Contextual cluster labels contain missing values.")
    observed_labels = sorted(set(labels.tolist()), key=_cluster_sort_key)
    expected_labels = [f"C{index}" for index in range(len(observed_labels))]
    if observed_labels != expected_labels:
        raise SO2HLInteractiveError(
            "Contextual cluster labels must be contiguous from C0."
        )

    palette_keys = sorted((str(key) for key in palette), key=_cluster_sort_key)
    if palette_keys != observed_labels:
        missing_palette = sorted(set(observed_labels).difference(palette_keys))
        extra_palette = sorted(set(palette_keys).difference(observed_labels))
        detail = f"missing={missing_palette}, extra={extra_palette}"
        raise SO2HLInteractiveError(
            f"Palette must cover the complete contextual cluster set ({detail})."
        )
    colors = [_validate_hex_color(palette[label], label=label) for label in palette_keys]
    if len({color.lower() for color in colors}) != len(colors):
        raise SO2HLInteractiveError("Contextual cluster palette colors must be unique.")

    if "global_cell_index" in validated:
        global_indices = validated["global_cell_index"]
        if global_indices.isna().any() or not global_indices.is_unique:
            raise SO2HLInteractiveError("Global source row indices must be unique.")
    if "cell_key" in validated:
        keys = validated["cell_key"]
        if keys.isna().any() or not keys.is_unique:
            raise SO2HLInteractiveError("Stable source keys must be unique.")

    return validated


def _encode_array(value: np.ndarray, *, dtype: str | np.dtype[Any]) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return base64.b64encode(memoryview(array).cast("B")).decode("ascii")


def build_interactive_payload(
    frame: pd.DataFrame,
    palette: Mapping[str, str],
) -> dict[str, Any]:
    """Encode only coordinates and cluster codes for the offline canvas viewer."""

    validated = validate_interactive_frame(frame, palette)
    clusters = sorted(
        set(validated["contextual_cluster"].astype(str)), key=_cluster_sort_key
    )
    canonical_palette = {label: str(palette[label]) for label in clusters}
    cluster_to_code = {label: index for index, label in enumerate(clusters)}
    if len(clusters) > np.iinfo(np.uint8).max + 1:
        raise SO2HLInteractiveError("Too many contextual clusters for Uint8 codes.")

    core_records: list[dict[str, Any]] = []
    observed_points = 0
    for core_number in SO2_CORE_NUMBERS:
        core = validated.loc[validated["core_number"] == core_number]
        x = np.ascontiguousarray(core["x_um"].to_numpy(dtype="<f8", copy=True))
        y = np.ascontiguousarray(core["y_um"].to_numpy(dtype="<f8", copy=True))
        codes = np.ascontiguousarray(
            core["contextual_cluster"]
            .astype(str)
            .map(cluster_to_code)
            .to_numpy(dtype=np.uint8, copy=True)
        )
        if not (len(x) == len(y) == len(codes) == len(core)):
            raise SO2HLInteractiveError(
                f"Coordinate/label ordering mismatch for SO2 core {core_number}."
            )
        x_b64 = _encode_array(x, dtype="<f8")
        y_b64 = _encode_array(y, dtype="<f8")
        codes_b64 = _encode_array(codes, dtype=np.uint8)

        # Fail closed if encoding ever changes dtype, length, or row order.
        x_round_trip = np.frombuffer(base64.b64decode(x_b64, validate=True), dtype="<f8")
        y_round_trip = np.frombuffer(base64.b64decode(y_b64, validate=True), dtype="<f8")
        codes_round_trip = np.frombuffer(
            base64.b64decode(codes_b64, validate=True), dtype=np.uint8
        )
        if not (
            np.array_equal(x_round_trip, x)
            and np.array_equal(y_round_trip, y)
            and np.array_equal(codes_round_trip, codes)
        ):
            raise SO2HLInteractiveError(
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
        raise SO2HLInteractiveError(
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
    """Return the dependency-free browser application."""

    return r"""
(() => {
  'use strict';
  const payloadNode = document.getElementById('bagm-data');
  const payload = JSON.parse(payloadNode.textContent);
  const clusters = payload.clusters;
  const palette = payload.palette;
  const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  let selectedCluster = null;
  const views = new Map();
  let hoverFrame = 0;

  function bytesFromBase64(encoded) {
    const binary = atob(encoded);
    const bytes = new Uint8Array(binary.length);
    for (let offset = 0; offset < binary.length; offset += 1) {
      bytes[offset] = binary.charCodeAt(offset);
    }
    return bytes;
  }

  function float64FromBase64(encoded) {
    const bytes = bytesFromBase64(encoded);
    if (bytes.byteLength % 8 !== 0) throw new Error('Invalid coordinate payload.');
    if (littleEndian && bytes.byteOffset % 8 === 0) {
      return new Float64Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 8);
    }
    const result = new Float64Array(bytes.byteLength / 8);
    const source = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let index = 0; index < result.length; index += 1) {
      result[index] = source.getFloat64(index * 8, true);
    }
    return result;
  }

  function hexToRgba(hex, opacity) {
    const number = Number.parseInt(hex.slice(1), 16);
    const red = (number >> 16) & 255;
    const green = (number >> 8) & 255;
    const blue = number & 255;
    return `rgba(${red},${green},${blue},${opacity})`;
  }

  function niceScaleLength(target) {
    if (!(target > 0)) return 1;
    const exponent = Math.floor(Math.log10(target));
    const fraction = target / (10 ** exponent);
    const nice = fraction >= 5 ? 5 : fraction >= 2 ? 2 : 1;
    return nice * (10 ** exponent);
  }

  function createBins(view) {
    const gridSize = 64;
    const bins = Array.from({length: gridSize * gridSize}, () => []);
    const rangeX = Math.max(view.bounds.x_max - view.bounds.x_min, Number.EPSILON);
    const rangeY = Math.max(view.bounds.y_max - view.bounds.y_min, Number.EPSILON);
    for (let index = 0; index < view.x.length; index += 1) {
      const bx = Math.max(0, Math.min(gridSize - 1,
        Math.floor((view.x[index] - view.bounds.x_min) / rangeX * gridSize)));
      const by = Math.max(0, Math.min(gridSize - 1,
        Math.floor((view.y[index] - view.bounds.y_min) / rangeY * gridSize)));
      bins[by * gridSize + bx].push(index);
    }
    view.gridSize = gridSize;
    view.bins = bins;
  }

  function resizeCanvas(view) {
    const rectangle = view.canvas.getBoundingClientRect();
    const width = Math.max(260, Math.round(rectangle.width));
    const height = Math.max(250, Math.round(rectangle.height));
    const ratio = Math.min(window.devicePixelRatio || 1, 2.5);
    if (view.canvas.width !== Math.round(width * ratio) ||
        view.canvas.height !== Math.round(height * ratio)) {
      view.canvas.width = Math.round(width * ratio);
      view.canvas.height = Math.round(height * ratio);
    }
    view.width = width;
    view.height = height;
    view.ratio = ratio;
    view.context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function calculateTransform(view) {
    const padding = 18;
    const rangeX = Math.max(view.bounds.x_max - view.bounds.x_min, 1);
    const rangeY = Math.max(view.bounds.y_max - view.bounds.y_min, 1);
    const baseScale = Math.min(
      (view.width - 2 * padding) / rangeX,
      (view.height - 2 * padding) / rangeY
    );
    return {
      centerX: (view.bounds.x_min + view.bounds.x_max) / 2,
      centerY: (view.bounds.y_min + view.bounds.y_max) / 2,
      baseScale,
      scale: baseScale * view.zoom
    };
  }

  function screenCoordinates(view, transform, index) {
    return [
      view.width / 2 + (view.x[index] - transform.centerX) * transform.scale + view.panX,
      view.height / 2 + (view.y[index] - transform.centerY) * transform.scale + view.panY
    ];
  }

  function drawCluster(view, transform, code, opacity, radius) {
    const context = view.context;
    context.fillStyle = hexToRgba(palette[clusters[code]], opacity);
    const indices = view.indicesByCluster[code];
    const diameter = radius * 2;
    for (let position = 0; position < indices.length; position += 1) {
      const index = indices[position];
      const [sx, sy] = screenCoordinates(view, transform, index);
      if (sx < -diameter || sy < -diameter ||
          sx > view.width + diameter || sy > view.height + diameter) continue;
      context.fillRect(sx - radius, sy - radius, diameter, diameter);
    }
  }

  function drawScaleBar(view, transform) {
    const context = view.context;
    const physical = niceScaleLength((view.width / 5) / transform.scale);
    const length = physical * transform.scale;
    if (!Number.isFinite(length) || length < 18 || length > view.width / 2) return;
    const right = view.width - 16;
    const y = view.height - 17;
    context.save();
    context.strokeStyle = '#111827';
    context.fillStyle = '#111827';
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(right - length, y);
    context.lineTo(right, y);
    context.moveTo(right - length, y - 4);
    context.lineTo(right - length, y + 4);
    context.moveTo(right, y - 4);
    context.lineTo(right, y + 4);
    context.stroke();
    context.font = '11px system-ui, sans-serif';
    context.textAlign = 'right';
    context.fillText(`${physical.toLocaleString()} µm`, right, y - 7);
    context.restore();
  }

  function drawView(view) {
    resizeCanvas(view);
    const context = view.context;
    context.clearRect(0, 0, view.width, view.height);
    context.fillStyle = '#f8fafc';
    context.fillRect(0, 0, view.width, view.height);
    const transform = calculateTransform(view);
    view.transform = transform;

    if (selectedCluster === null) {
      for (let code = 0; code < clusters.length; code += 1) {
        drawCluster(view, transform, code, 0.82, 0.9);
      }
    } else {
      const selectedCode = clusters.indexOf(selectedCluster);
      for (let code = 0; code < clusters.length; code += 1) {
        if (code !== selectedCode) drawCluster(view, transform, code, 0.09, 0.72);
      }
      // selected-last: keep the chosen population bright and visually on top.
      drawCluster(view, transform, selectedCode, 1.0, 1.65);
    }
    drawScaleBar(view, transform);
  }

  function drawAll() {
    views.forEach(drawView);
  }

  function updateSelection(nextCluster) {
    selectedCluster = nextCluster === selectedCluster ? null : nextCluster;
    document.querySelectorAll('.cluster-chip[data-cluster]').forEach((button) => {
      const active = button.dataset.cluster === selectedCluster;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    const status = document.getElementById('selection-status');
    status.textContent = selectedCluster === null
      ? 'Showing all contextual clusters.'
      : `Highlighting ${selectedCluster}; all other clusters are faded.`;
    drawAll();
  }

  function showAll() {
    if (selectedCluster !== null) updateSelection(selectedCluster);
    else drawAll();
  }

  function resetAllViews() {
    views.forEach((view) => {
      view.zoom = 1;
      view.panX = 0;
      view.panY = 0;
      drawView(view);
    });
  }

  function pointerPosition(event, canvas) {
    const rectangle = canvas.getBoundingClientRect();
    return [event.clientX - rectangle.left, event.clientY - rectangle.top];
  }

  function nearestPoint(view, mx, my) {
    const transform = view.transform || calculateTransform(view);
    const dataX = transform.centerX + (mx - view.width / 2 - view.panX) / transform.scale;
    const dataY = transform.centerY + (my - view.height / 2 - view.panY) / transform.scale;
    const rangeX = Math.max(view.bounds.x_max - view.bounds.x_min, Number.EPSILON);
    const rangeY = Math.max(view.bounds.y_max - view.bounds.y_min, Number.EPSILON);
    const bx = Math.max(0, Math.min(view.gridSize - 1,
      Math.floor((dataX - view.bounds.x_min) / rangeX * view.gridSize)));
    const by = Math.max(0, Math.min(view.gridSize - 1,
      Math.floor((dataY - view.bounds.y_min) / rangeY * view.gridSize)));
    let best = -1;
    let bestDistance = 9 * 9;
    for (let yOffset = -1; yOffset <= 1; yOffset += 1) {
      for (let xOffset = -1; xOffset <= 1; xOffset += 1) {
        const gx = bx + xOffset;
        const gy = by + yOffset;
        if (gx < 0 || gy < 0 || gx >= view.gridSize || gy >= view.gridSize) continue;
        const candidates = view.bins[gy * view.gridSize + gx];
        for (let position = 0; position < candidates.length; position += 1) {
          const index = candidates[position];
          if (selectedCluster !== null && clusters[view.codes[index]] !== selectedCluster) continue;
          const [sx, sy] = screenCoordinates(view, transform, index);
          const distance = (sx - mx) ** 2 + (sy - my) ** 2;
          if (distance < bestDistance) {
            bestDistance = distance;
            best = index;
          }
        }
      }
    }
    return best;
  }

  function updateHover(view, event) {
    const [mx, my] = pointerPosition(event, view.canvas);
    const index = nearestPoint(view, mx, my);
    if (index < 0) {
      view.tooltip.hidden = true;
      return;
    }
    const cluster = clusters[view.codes[index]];
    view.tooltip.textContent = `Core ${view.coreNumber} · ${cluster} · ` +
      `x ${Math.round(view.x[index]).toLocaleString()} µm · ` +
      `y ${Math.round(view.y[index]).toLocaleString()} µm`;
    view.tooltip.style.left = `${Math.min(mx + 12, view.width - 205)}px`;
    view.tooltip.style.top = `${Math.max(8, my - 34)}px`;
    view.tooltip.hidden = false;
  }

  function attachNavigation(view) {
    const canvas = view.canvas;
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      const [mx, my] = pointerPosition(event, canvas);
      const transform = calculateTransform(view);
      const worldX = (mx - view.width / 2 - view.panX) / transform.scale;
      const worldY = (my - view.height / 2 - view.panY) / transform.scale;
      const nextZoom = Math.max(0.5, Math.min(30, view.zoom * Math.exp(-event.deltaY * 0.0015)));
      view.panX = mx - view.width / 2 - worldX * transform.baseScale * nextZoom;
      view.panY = my - view.height / 2 - worldY * transform.baseScale * nextZoom;
      view.zoom = nextZoom;
      drawView(view);
    }, {passive: false});

    canvas.addEventListener('pointerdown', (event) => {
      view.dragging = true;
      view.dragStartX = event.clientX;
      view.dragStartY = event.clientY;
      view.panStartX = view.panX;
      view.panStartY = view.panY;
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add('dragging');
    });
    canvas.addEventListener('pointermove', (event) => {
      if (view.dragging) {
        view.panX = view.panStartX + event.clientX - view.dragStartX;
        view.panY = view.panStartY + event.clientY - view.dragStartY;
        drawView(view);
        return;
      }
      cancelAnimationFrame(hoverFrame);
      hoverFrame = requestAnimationFrame(() => updateHover(view, event));
    });
    canvas.addEventListener('mousemove', (event) => {
      if ('PointerEvent' in window || view.dragging) return;
      cancelAnimationFrame(hoverFrame);
      hoverFrame = requestAnimationFrame(() => updateHover(view, event));
    });
    const stopDrag = (event) => {
      view.dragging = false;
      canvas.classList.remove('dragging');
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    };
    canvas.addEventListener('pointerup', stopDrag);
    canvas.addEventListener('pointercancel', stopDrag);
    canvas.addEventListener('mouseleave', () => { view.tooltip.hidden = true; });
    canvas.addEventListener('dblclick', () => {
      view.zoom = 1;
      view.panX = 0;
      view.panY = 0;
      drawView(view);
    });
  }

  function initializeView(record) {
    const canvas = document.querySelector(`canvas[data-core="${record.core_number}"]`);
    const tooltip = canvas.parentElement.querySelector('.hover-card');
    const x = float64FromBase64(record.x_b64);
    const y = float64FromBase64(record.y_b64);
    const codes = bytesFromBase64(record.cluster_codes_b64);
    if (x.length !== record.point_count || y.length !== record.point_count ||
        codes.length !== record.point_count) throw new Error('Point count mismatch.');
    const indicesByCluster = Array.from({length: clusters.length}, () => []);
    for (let index = 0; index < codes.length; index += 1) {
      if (codes[index] >= clusters.length) throw new Error('Invalid cluster code.');
      indicesByCluster[codes[index]].push(index);
    }
    const view = {
      canvas, context: canvas.getContext('2d', {alpha: false}), tooltip,
      coreNumber: record.core_number, x, y, codes, bounds: record.bounds,
      indicesByCluster, zoom: 1, panX: 0, panY: 0, dragging: false
    };
    createBins(view);
    attachNavigation(view);
    views.set(record.core_number, view);
  }

  function exportCombinedPng() {
    const tileWidth = 800;
    const tileHeight = 610;
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = tileWidth * 3;
    exportCanvas.height = tileHeight * 5;
    const context = exportCanvas.getContext('2d');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
    context.textAlign = 'center';
    context.fillStyle = '#111827';
    context.font = '600 25px system-ui, sans-serif';
    payload.core_order.forEach((coreNumber, index) => {
      const column = index % 3;
      const row = Math.floor(index / 3);
      const left = column * tileWidth;
      const top = row * tileHeight;
      context.fillText(`SO2 Core ${coreNumber}`, left + tileWidth / 2, top + 31);
      context.drawImage(views.get(coreNumber).canvas, left + 12, top + 44,
        tileWidth - 24, tileHeight - 56);
    });
    const legendLeft = tileWidth * 2;
    const legendTop = tileHeight * 4;
    context.textAlign = 'left';
    context.font = '600 24px system-ui, sans-serif';
    context.fillText(selectedCluster === null ? 'All contextual clusters' :
      `Highlighted cluster: ${selectedCluster}`, legendLeft + 35, legendTop + 55);
    context.font = '18px system-ui, sans-serif';
    clusters.forEach((cluster, index) => {
      const column = index % 3;
      const row = Math.floor(index / 3);
      const x = legendLeft + 40 + column * 245;
      const y = legendTop + 105 + row * 57;
      context.fillStyle = palette[cluster];
      context.fillRect(x, y - 16, 22, 22);
      context.fillStyle = '#111827';
      context.fillText(cluster, x + 32, y + 2);
    });
    exportCanvas.toBlob((blob) => {
      if (!blob) return;
      const anchor = document.createElement('a');
      const suffix = selectedCluster === null ? 'all' : selectedCluster;
      anchor.download = `so2_hL_contextual_clusters_${suffix}.png`;
      anchor.href = URL.createObjectURL(blob);
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
    }, 'image/png');
  }

  document.querySelectorAll('.cluster-chip[data-cluster]').forEach((button) => {
    button.addEventListener('click', () => updateSelection(button.dataset.cluster));
  });
  document.getElementById('show-all').addEventListener('click', showAll);
  document.getElementById('reset-views').addEventListener('click', resetAllViews);
  document.getElementById('export-png').addEventListener('click', exportCombinedPng);
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') showAll();
  });

  payload.cores.forEach(initializeView);
  const observer = new ResizeObserver((entries) => {
    entries.forEach((entry) => {
      const coreNumber = Number(entry.target.dataset.core);
      if (views.has(coreNumber)) drawView(views.get(coreNumber));
    });
  });
  views.forEach((view) => observer.observe(view.canvas));
  drawAll();
  window.BAGM_SO2_VIEWER = Object.freeze({
    selectCluster: updateSelection,
    showAll,
    resetAllViews,
    exportCombinedPng
  });
})();
""".strip()


def _interactive_css() -> str:
    return """
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; color: #172033; background: #eef2f7; }
header { padding: 1.25rem clamp(1rem, 3vw, 2.5rem); background: #111827; color: #fff; }
header h1 { margin: 0 0 .35rem; font-size: clamp(1.35rem, 2.4vw, 2.1rem); }
header p { margin: .25rem 0; color: #dbe4f0; max-width: 80rem; line-height: 1.45; }
.workspace { display: grid; grid-template-columns: minmax(220px, 300px) minmax(0, 1fr); gap: 1rem; padding: 1rem; }
.controls { align-self: start; position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow: auto; padding: 1rem; border-radius: .8rem; background: #fff; box-shadow: 0 2px 14px rgba(15,23,42,.09); }
.controls h2 { margin: 0 0 .35rem; font-size: 1.05rem; }
.instructions { color: #526075; margin: .35rem 0 .8rem; font-size: .88rem; line-height: 1.4; }
.cluster-list { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .4rem; }
button { font: inherit; }
.cluster-chip { display: flex; align-items: center; gap: .45rem; min-width: 0; border: 1px solid #d7deea; border-radius: .45rem; background: #fff; color: #1f2937; padding: .42rem .5rem; cursor: pointer; transition: opacity .12s, border-color .12s, box-shadow .12s; }
.cluster-chip:hover, .cluster-chip:focus-visible { border-color: #475569; outline: none; }
.cluster-chip.active { border-color: #111827; box-shadow: 0 0 0 2px #111827; font-weight: 700; }
.swatch { width: .9rem; height: .9rem; border-radius: 50%; flex: 0 0 auto; background: var(--cluster-color); }
.cluster-count { margin-left: auto; color: #64748b; font-size: .75rem; }
.action-row { display: grid; gap: .45rem; margin-top: .8rem; }
.action { border: 0; border-radius: .45rem; padding: .55rem .65rem; cursor: pointer; background: #e7edf5; color: #162033; font-weight: 600; }
.action.primary { background: #1d4ed8; color: #fff; }
#selection-status { min-height: 2.4em; margin: .75rem 0 0; color: #334155; font-size: .84rem; }
.privacy-note { margin: .9rem 0 0; padding: .7rem; border-left: 3px solid #d97706; background: #fff7ed; color: #713f12; font-size: .78rem; line-height: 1.4; }
.core-grid { display: grid; grid-template-columns: repeat(3, minmax(250px, 1fr)); gap: .8rem; align-items: start; }
.core-panel { min-width: 0; overflow: hidden; border-radius: .7rem; background: #fff; box-shadow: 0 1px 10px rgba(15,23,42,.08); }
.core-panel h2 { margin: 0; padding: .65rem .8rem .1rem; text-align: center; font-size: 1rem; }
.point-count { text-align: center; color: #64748b; font-size: .72rem; padding-bottom: .25rem; }
.canvas-wrap { position: relative; }
canvas { display: block; width: 100%; height: clamp(280px, 31vw, 430px); cursor: grab; touch-action: none; }
canvas.dragging { cursor: grabbing; }
.hover-card { position: absolute; z-index: 3; pointer-events: none; max-width: 205px; border-radius: .35rem; padding: .34rem .48rem; color: #fff; background: rgba(15,23,42,.92); font-size: .73rem; white-space: nowrap; }
.legend-panel { display: flex; align-items: center; justify-content: center; min-height: 340px; padding: 1.5rem; color: #526075; text-align: center; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: .7rem; }
@media (max-width: 1100px) { .workspace { grid-template-columns: 1fr; } .controls { position: static; max-height: none; } .cluster-list { grid-template-columns: repeat(5, minmax(0, 1fr)); } }
@media (max-width: 820px) { .core-grid { grid-template-columns: repeat(2, minmax(240px, 1fr)); } .cluster-list { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 560px) { .core-grid { grid-template-columns: 1fr; } .cluster-list { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
""".strip()


def _cluster_buttons(frame: pd.DataFrame, palette: Mapping[str, str]) -> str:
    counts = frame["contextual_cluster"].astype(str).value_counts()
    labels = sorted(palette, key=_cluster_sort_key)
    return "\n".join(
        (
            f'<button type="button" class="cluster-chip" data-cluster="{label}" '
            f'aria-pressed="false" style="--cluster-color:{palette[label]}">'
            f'<span class="swatch" aria-hidden="true"></span>'
            f'<span>{label}</span><span class="cluster-count">{int(counts[label]):,}</span>'
            "</button>"
        )
        for label in labels
    )


def _core_panels(frame: pd.DataFrame) -> str:
    counts = frame.groupby("core_number", sort=False).size()
    panels = []
    for core_number in SO2_CORE_NUMBERS:
        panels.append(
            f'<article class="core-panel" data-panel-core="{core_number}">'
            f'<h2>SO2 Core {core_number}</h2>'
            f'<div class="point-count">{int(counts.loc[core_number]):,} cells</div>'
            '<div class="canvas-wrap">'
            f'<canvas data-core="{core_number}" role="img" '
            f'aria-label="Interactive contextual cluster map for SO2 Core {core_number}"></canvas>'
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
<title>SO2 14-core contextual hL clusters</title>
<style>{_interactive_css()}</style>
</head>
<body>
<header>
  <h1>SO2 14-core contextual hL clusters</h1>
  <p>Joint Leiden clustering at resolution 1.0. Click a cluster to keep it bright while fading the others; click it again, choose Show all, or press Escape to reset.</p>
  <p>Scroll to zoom, drag to pan, double-click a panel to reset it, and hover for core, cluster, and rounded tissue coordinates.</p>
</header>
<main class="workspace">
  <aside class="controls" aria-label="Cluster controls">
    <h2>Contextual clusters</h2>
    <p class="instructions">These are model-derived groups, not validated cell types. Counts are across all 14 cores.</p>
    <div class="cluster-list">{_cluster_buttons(frame, palette)}</div>
    <div class="action-row">
      <button type="button" class="action" id="show-all">Show all clusters</button>
      <button type="button" class="action" id="reset-views">Reset all spatial views</button>
      <button type="button" class="action primary" id="export-png">Export current view as PNG</button>
    </div>
    <p id="selection-status" aria-live="polite">Showing all contextual clusters.</p>
    <p class="privacy-note"><strong>Authorized sharing only.</strong> This offline file contains exact cell-level tissue coordinates. It contains no stable cell identifiers, expression values, learned vectors, clinical labels, or external network resources.</p>
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
        "contextual_cluster_number",
        "/workspace",
        "fetch(",
        "xmlhttprequest",
        "url(http",
    )
    lowered = html.lower()
    found = [token for token in prohibited if token.lower() in lowered]
    if found:
        raise SO2HLInteractiveError(
            f"Shareable HTML contains prohibited source or network fields: {found}"
        )
    if "cell_key" in frame and len(frame):
        first_key = str(frame["cell_key"].iloc[0])
        if first_key and first_key in html:
            raise SO2HLInteractiveError("Shareable HTML contains a stable source key.")
    if "connect-src 'none'" not in html:
        raise SO2HLInteractiveError("Shareable HTML lacks the offline network CSP.")
    if len(re.findall(r"<canvas\s+[^>]*data-core=", html)) != len(SO2_CORE_NUMBERS):
        raise SO2HLInteractiveError("Shareable HTML lacks exactly 14 core canvases.")
    external_asset = re.search(
        r"<(?:script|img)\b[^>]*\bsrc\s*=|<link\b[^>]*\bhref\s*=",
        html,
        flags=re.IGNORECASE,
    )
    if external_asset is not None:
        raise SO2HLInteractiveError("Shareable HTML contains an external DOM asset.")


def render_interactive_html(
    *,
    frame: pd.DataFrame,
    palette: Mapping[str, str],
    output_path: str | Path,
    source_table_path: str | Path,
    source_manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Write one self-contained HTML viewer and return its checksummed receipt."""

    output = Path(output_path)
    source_table = Path(source_table_path)
    source_manifest = Path(source_manifest_path)
    if not source_table.is_file() or not source_manifest.is_file():
        raise SO2HLInteractiveError("Interactive source artifacts are missing.")
    validated = validate_interactive_frame(frame, palette)
    html = _render_html(validated, palette)
    _validate_rendered_html(html, frame=validated)
    _atomic_write_text(output, html)
    if output.read_text(encoding="utf-8") != html:
        raise SO2HLInteractiveError("Interactive HTML failed its write/read check.")
    receipt = _receipt_with_self_hash(
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
                "contextual_cluster_code",
            ],
            "stable_identifiers_included": False,
            "clinical_fields_included": False,
            "embedding_values_included": False,
        }
    )
    return receipt


def _write_transfer_zip(*, html_path: Path, zip_path: Path) -> Mapping[str, Any]:
    """Create a deterministic, one-member transfer archive."""

    info = zipfile.ZipInfo(html_path.name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800
    with html_path.open("rb") as handle:
        html_bytes = handle.read()

    # Build in memory so the shared atomic writer owns the only filesystem write.
    from io import BytesIO

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(info, html_bytes)
    _atomic_write_bytes(zip_path, buffer.getvalue())

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        members = archive.infolist()
        if archive.testzip() is not None or len(members) != 1:
            raise SO2HLInteractiveError("Transfer ZIP integrity check failed.")
        member = members[0]
        member_path = Path(member.filename)
        if (
            member.filename != html_path.name
            or member_path.name != member.filename
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise SO2HLInteractiveError("Transfer ZIP member name is unsafe.")
        extracted = archive.read(member)
    if not hmac.compare_digest(
        hashlib.sha256(extracted).hexdigest(), sha256_file(html_path)
    ):
        raise SO2HLInteractiveError("Transfer ZIP member differs from the HTML file.")
    return {
        "member": html_path.name,
        "member_sha256": sha256_file(html_path),
        "zip_sha256": sha256_file(zip_path),
        "zip_size_bytes": int(zip_path.stat().st_size),
        "deterministic_timestamp": "1980-01-01T00:00:00",
    }


def _render_readme(*, source_manifest: Mapping[str, Any], html_name: str, zip_name: str) -> str:
    run_id = str(source_manifest["run_id"])
    return f"""# Interactive SO2 contextual-cluster map

Open `{html_name}` in a current desktop browser. The file is self-contained and
does not need a web server or internet connection. For transfer, `{zip_name}`
contains the same HTML as its only archive member.

## Interaction

- Click `C0` through `C18` to keep that joint contextual cluster bright and fade
  all other clusters across every core.
- Click the selected cluster again, click **Show all clusters**, or press Escape
  to restore all colors.
- Scroll over a tissue panel to zoom, drag to pan, and double-click to reset that
  panel. Hover reports only numeric core, cluster, and rounded tissue coordinates.
- **Export current view as PNG** saves the current combined 3 x 5 view.

This viewer reuses the completed hL Leiden labels at resolution 1.0 for run
`{run_id}`. It performs no model inference, retraining, or reclustering and does
not use a GPU. Cluster IDs are shared across the 14 panels because clustering was
performed jointly.

## Interpretation and sharing limits

These contextual clusters are model-derived patterns after graph-based
neighborhood processing. They are not independently validated cell types and do
not establish signaling, biological influence, or causality. Marker-based and
pathological validation remains separate.

The HTML omits stable cell identifiers, expression values, learned vectors,
clinical fields, donor mappings, and source filesystem paths. It does contain
exact cell-level tissue coordinates, so distribute it only to authorized
collaborators rather than publishing it openly.

## Reproduction

From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \\
  render-so2-hl-interactive \\
  --run-id {run_id}
```
"""


def _verify_interactive_manifest(
    output_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    _verify_self_hash(manifest, label="SO2 interactive manifest")
    if any(
        (
            manifest.get("schema") != INTERACTIVE_SCHEMA,
            manifest.get("status") != "complete",
            manifest.get("run_id") != EXPECTED_RUN_ID,
            tuple(manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            int(manifest.get("point_count", -1)) != EXPECTED_TOTAL_CELLS,
            int(manifest.get("cluster_count", -1)) != EXPECTED_CLUSTER_COUNT,
            float(manifest.get("leiden_resolution", math.nan))
            != DEFAULT_LEIDEN_RESOLUTION,
        )
    ):
        raise SO2HLInteractiveError("Interactive manifest identity is invalid.")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise SO2HLInteractiveError("Interactive manifest lacks output checksums.")
    required = {"README.md", HTML_FILENAME, ZIP_FILENAME}
    if set(files) != required:
        raise SO2HLInteractiveError(
            f"Interactive output set changed: {sorted(set(files).symmetric_difference(required))}"
        )
    for relative, record in files.items():
        path = output_root / str(relative)
        if not path.is_file() or _file_record(path) != dict(record):
            raise SO2HLInteractiveError(
                f"Interactive output checksum changed: {relative}"
            )
    html_path = output_root / HTML_FILENAME
    zip_path = output_root / ZIP_FILENAME
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        if archive.testzip() is not None or archive.namelist() != [HTML_FILENAME]:
            raise SO2HLInteractiveError("Interactive transfer ZIP is invalid.")
        archived = archive.read(HTML_FILENAME)
    if not hmac.compare_digest(
        hashlib.sha256(archived).hexdigest(), sha256_file(html_path)
    ):
        raise SO2HLInteractiveError("Archived and standalone HTML checksums differ.")


def run_so2_hl_interactive(
    *,
    paths: ProjectPaths,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create or verify the completed offline 14-core hL viewer bundle."""

    selected_run = EXPECTED_RUN_ID if run_id is None else str(run_id)
    if selected_run != EXPECTED_RUN_ID:
        raise SO2HLInteractiveError(
            "The interactive viewer is locked to the completed SO2 14-core run."
        )
    source_root = (
        paths.report_root / "analyses" / DEFAULT_SOURCE_REPORT / selected_run
    )
    source_manifest_path = source_root / "manifest.json"
    source_table_path = source_root / "tables" / "cell_contextual_clusters.parquet"
    source_palette_path = source_root / "clustering" / "contextual_palette.json"
    source_manifest = _read_json(
        source_manifest_path, label="SO2 contextual-clustering manifest"
    )
    _verify_source_manifest(source_root, source_manifest)
    if any(
        (
            source_manifest.get("schema") != SOURCE_ANALYSIS_SCHEMA,
            source_manifest.get("analysis_scope") != "contextual_hL_only",
            int(source_manifest.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS,
            int(source_manifest.get("cluster_count", -1)) != EXPECTED_CLUSTER_COUNT,
            tuple(source_manifest.get("core_order", ())) != SO2_CORE_NUMBERS,
            float(source_manifest.get("analysis_parameters", {}).get(
                "leiden_resolution", math.nan
            ))
            != DEFAULT_LEIDEN_RESOLUTION,
        )
    ):
        raise SO2HLInteractiveError(
            "Source report is not the completed SO2 hL resolution-1.0 analysis."
        )
    palette_document = _read_json(
        source_palette_path, label="SO2 contextual palette"
    )
    palette_value = palette_document.get("colors")
    if not isinstance(palette_value, Mapping):
        raise SO2HLInteractiveError("Contextual palette lacks a colors mapping.")
    palette = {str(key): str(value) for key, value in palette_value.items()}

    # Read only the four browser-relevant columns. Source identifiers never enter
    # this process's shareable payload construction.
    frame = pd.read_parquet(
        source_table_path,
        columns=["core_number", "x_um", "y_um", "contextual_cluster"],
    )
    validated = validate_interactive_frame(frame, palette)
    if len(validated) != EXPECTED_TOTAL_CELLS:
        raise SO2HLInteractiveError(
            f"Expected {EXPECTED_TOTAL_CELLS} cells, found {len(validated)}."
        )
    observed_counts = validated.groupby("core_number", sort=False).size().to_dict()
    if observed_counts != EXPECTED_CELL_COUNTS_BY_CORE:
        raise SO2HLInteractiveError(
            "Interactive source per-core counts differ from the completed report."
        )

    if output_dir is None:
        output_root = (
            paths.report_root / "analyses" / DEFAULT_OUTPUT_REPORT / selected_run
        )
    else:
        output_root = Path(output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = paths.project_root / output_root
        output_root = output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path, label="SO2 interactive manifest")
        _verify_interactive_manifest(output_root, manifest)
    else:
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
        readme_path = output_root / "README.md"
        _atomic_write_text(
            readme_path,
            _render_readme(
                source_manifest=source_manifest,
                html_name=HTML_FILENAME,
                zip_name=ZIP_FILENAME,
            ),
        )
        manifest = _receipt_with_self_hash(
            {
                "schema": INTERACTIVE_SCHEMA,
                "status": "complete",
                "created_at": _utc_now(),
                "run_id": selected_run,
                "analysis_scope": "contextual_hL_resolution_1p0_visualization_only",
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
                    "zoom": "wheel",
                    "pan": "pointer_drag",
                    "hover_fields": [
                        "numeric_core_number",
                        "contextual_cluster",
                        "rounded_x_um",
                        "rounded_y_um",
                    ],
                    "combined_png_export": True,
                },
                "execution": {
                    "visualization_only": True,
                    "model_inference": False,
                    "reclustering": False,
                    "gpu_used": False,
                },
                "sharing": {
                    "self_contained_offline_html": True,
                    "authorized_collaborators_only": True,
                    "exact_tissue_coordinates_included": True,
                    "stable_cell_identifiers_included": False,
                    "clinical_fields_included": False,
                    "embedding_values_included": False,
                },
                "source_artifacts": {
                    "analysis_manifest": _file_record(source_manifest_path),
                    "cluster_table": _file_record(source_table_path),
                    "palette": _file_record(source_palette_path),
                },
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
    "SO2HLInteractiveError",
    "build_interactive_payload",
    "interactive_javascript",
    "render_interactive_html",
    "run_so2_hl_interactive",
    "validate_interactive_frame",
]
