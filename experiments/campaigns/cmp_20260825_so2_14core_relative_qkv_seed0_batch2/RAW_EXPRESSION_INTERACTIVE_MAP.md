# SO2 raw-expression interactive-map contract

## Scope and status

Phase: complete. Outcome: all required outputs and source bindings are
checksum-verified.

Create one self-contained, offline HTML viewer for the completed classical
raw-expression Leiden analysis
`rawexpr_pca50_cosinek30_leiden_r1p0_s20260825`. The viewer must show SO2 cores
15 through 28 in their locked order and let a collaborator click `E0`, `E1`,
... to draw that joint cluster brightly while fading all others across every
core. It may support zoom, pan, hover, reset, and PNG export, matching the
completed contextual-hL viewer's established interaction pattern.

This is visualization only. It must not rerun preprocessing, PCA, kNN, Leiden,
model inference, or training, and it must not use a GPU. The source clustering
manifest, cell table, and palette must pass their existing checksum validation
before rendering.

## Permitted data and claims

The browser payload may contain only numeric core number, physical `x_um` and
`y_um` coordinates, and the raw-expression cluster code. It must omit stable
cell keys and indices, expression values, PCA scores, kNN edges, model
embeddings, metadata, clinical fields, donor mappings, and filesystem paths.
The file is therefore non-identifying at the cell-key level but still contains
exact tissue coordinates and is for authorized sharing only.

The clusters are exploratory expression-derived groups, not validated cell
types. The viewer establishes no signaling, biological influence, or causality.
The existing E1 low-count-depth warning must remain visible in the viewer or
its bundled README.

## Acceptance and falsification

Acceptance requires all 246,063 cells, the exact core counts and order, the
complete contiguous `E0`-through-`E11` palette, one canvas per core, cluster
click/highlight and show-all behavior, selected-last drawing, offline content
security policy, no external assets/network calls, a deterministic one-member
transfer ZIP, an explanatory README, and a self-checksummed manifest covering
all deliverables and source artifacts. A source checksum mismatch, missing or
reordered cell, prohibited payload field, non-finite coordinate, incomplete
palette, external dependency, GPU use, or checksum-invalid output is a stop
condition.

## Expected output and verification

```text
reports/analyses/so2_14core_raw_expression_interactive/
  rawexpr_pca50_cosinek30_leiden_r1p0_s20260825/
```

Reproduction command:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so2-raw-expression-interactive
```

Run the focused interactive and raw-clustering tests, reopen the completed
bundle through the CLI to exercise checksum-only resume, inspect the HTML in a
browser-compatible parser, and run `spatial_benchmark doctor`.

## Completed result (2026-08-26)

The viewer contains all 246,063 source rows, 14 ordered core canvases, and 12
cluster controls (`E0` through `E11`). The standalone HTML is 5,610,467 bytes;
the deterministic transfer ZIP is 3,532,406 bytes and contains that HTML as its
only member. JavaScript syntax, base64 coordinate/code round trips, point
counts, core order, offline CSP, prohibited-field exclusions, ZIP identity,
source checksums, output checksums, and checksum-only resume all passed.

Focused raw-expression clustering/viewer and existing hL-viewer regressions:
33 passed. Repository doctor: healthy. The output is visualization-only and
records no GPU use, reclustering, preprocessing, or model inference.
