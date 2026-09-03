# Direct-hL interactive-map contract

## Status and scope

This is a shareable visualization extension of the CPU-only direct-hL SO2
14-core clustering report. It reuses its finalized resolution-1.0 cell
assignments and coordinates exactly. It performs no model inference,
retraining, reclustering, PCA, h0 analysis, or delta-h analysis.

## Interaction contract

Create one self-contained HTML document with the 14 spatial panels in ascending
core order 15 through 28. Clicking a cluster label highlights that cluster
across all cores in its saturated palette color while fading all other
clusters. Clicking it again, choosing `Show all clusters`, or pressing Escape
resets the view. Zoom, pan, per-panel reset, coordinate hover, and browser-side
PNG export remain available.

## Portability, privacy, and acceptance

The HTML must contain no CDN or remote assets and must open locally without a
server or internet connection. Also create a deterministic ZIP containing the
single HTML. Do not include expression values, learned vectors, metadata,
checkpoint weights, stable cell keys, cell indices, FOV identifiers, or source
paths. Exact tissue coordinates make the output a cell-level artifact for
authorized sharing.

Fail on source-report checksum drift, incomplete/reordered cores, a hard-coded
cluster count that differs from the direct-hL source report, palette drift,
non-finite coordinates, dropped/duplicated cells, external HTML dependencies,
missing highlight/reset behavior, GPU access, or output-checksum failure.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so2-hl-direct-interactive \
  --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6
```
