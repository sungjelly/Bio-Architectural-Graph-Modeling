# Contextual hL interactive-map contract

## Status and scope

This is a shareable visualization extension of the completed CPU-only SO2
14-core contextual hL clustering report for run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. It reuses the finalized
resolution-1.0 cell assignments and coordinates exactly. It performs no model
inference, retraining, reclustering, h0 analysis, or delta-h analysis.

## Interaction contract

Create one self-contained HTML document with the 14 spatial panels in ascending
core order 15 through 28. Every panel retains the repository's physical-coordinate
orientation and equal spatial aspect. The same contextual cluster uses the same
color in every panel.

Clicking either a cluster legend label or its cluster button must highlight that
cluster across all cores using its saturated palette color while fading all other
clusters. Clicking the selected cluster again, the `Show all clusters` button, or
Escape resets the view. Hover exposes only the core number, contextual cluster,
and rounded physical coordinates. Zoom, pan, reset-view, and browser-side PNG
export remain available.

## Portability, privacy, and resource constraints

The HTML must bundle its JavaScript and data, use no CDN or remote asset, and open
from a local file without a server or internet connection. Also create a ZIP
containing the single HTML for convenient transfer. Do not include raw source
identifiers, FOV identifiers, expression values, hL vectors, metadata values, or
checkpoint weights. Do not include even the stable per-cell keys or cell indices;
the exact tissue coordinates required for interactive rendering still make this
a cell-level artifact for authorized sharing. Generation is CPU-only and may not
access a GPU.

## Acceptance and provenance

Fail on source-report checksum drift, incomplete/reordered cores, missing or
noncontiguous cluster labels, palette drift, non-finite coordinates, dropped or
duplicated cells, external HTML dependencies, absent highlight/reset handlers,
or output-checksum failure. Save a README and final manifest containing source
report/table/palette checksums, HTML and ZIP checksums, row/trace counts, core
order, cluster count, and interaction settings in a separate run-specific report
directory. The model-derived clusters remain descriptive and are not validated
cell types or biological/causal conclusions.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so2-hl-interactive \
  --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6
```
