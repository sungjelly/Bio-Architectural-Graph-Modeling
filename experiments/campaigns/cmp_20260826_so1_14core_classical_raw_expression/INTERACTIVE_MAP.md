# SO1 raw-expression interactive-map contract

## Status and scope

Phase: complete. Outcome: inconclusive for biological identity or mechanism;
the visualization and checksum acceptance criteria passed.

Create a self-contained offline HTML viewer for the checksum-verified SO1
classical raw-expression clustering analysis
`rawexpr_pca50_cosinek30_leiden_r1p0_s20260825`. This is visualization only: it
must not repeat normalization, PCA, neighbor construction, Leiden clustering,
model inference, or training, and it must not access a GPU.

## Question, estimand, and claim limit

The descriptive question is where each joint SO1 expression cluster occurs in
physical tissue coordinates across cores 1 through 14. The estimand is the
completed source analysis's fixed per-cell `S1E` assignment. The observational
unit is a cell for rendering; the 14 cores, not individual cells, are the
relevant sample units for cross-core interpretation.

The primary hypothesis is that the fixed joint clusters have spatial patterns
that authorized collaborators can inspect more effectively through selective
highlighting. A credible alternative is that visually striking partitions are
driven by count depth, vendor-QC failures, or core-specific technical effects.
The viewer therefore displays the completed source analysis's structured
low-count-depth warning and states that all 161,596 source cells—including
low-depth and vendor-QC-failed cells—are retained. An attractive spatial pattern
does not establish a cell type, signaling, biological influence, or causality.

## Input and label contract

The source is the completed report under:

```text
reports/analyses/so1_14core_raw_expression_clustering/
  rawexpr_pca50_cosinek30_leiden_r1p0_s20260825/
```

Before rendering, its public verifier must validate the final manifest and all
source checksums. The viewer reads only these four table columns:

```text
core_number, x_um, y_um, expression_cluster
```

The browser payload may contain only numeric core number, `x_um`, `y_um`, and a
compact expression-cluster code plus its categorical palette. It must omit
stable cell keys and indices, raw or transformed expression values, PCA scores,
kNN edges, model embeddings or checkpoints, vendor-QC fields, clinical fields,
donor mappings, and source filesystem paths.

SO1 labels are `S1E0`, `S1E1`, ... in descending source-cluster size. This is an
independent label namespace: `S1E3` does not correspond to SO2 `E3`, and matching
numeric suffixes do not imply matching populations.

## Interaction, privacy, and outputs

The file shows exactly one canvas for each core in order 1 through 14, with
unambiguous `SO1 Core N` titles. Clicking a cluster keeps it saturated and drawn
last while fading every other cluster across all cores. Clicking it again,
choosing **Show all clusters**, or pressing Escape resets selection. Wheel zoom,
pointer-drag pan, double-click reset, restricted hover, and combined PNG export
are supported.

The HTML is dependency-free, contains an offline content-security policy, and
makes no network requests. Hover exposes only numeric core, cluster, and rounded
coordinates. Exact tissue coordinates remain cell-level data, so the HTML and
its deterministic one-member transfer ZIP are for authorized sharing only.

Outputs use the separate namespace:

```text
reports/analyses/so1_14core_raw_expression_interactive/
  rawexpr_pca50_cosinek30_leiden_r1p0_s20260825/
```

The final self-checksummed manifest binds the source analysis manifest, cluster
table, palette, static PNG, standalone HTML, README, and transfer ZIP.

## Acceptance, falsification, and stop criteria

Acceptance requires all 161,596 cells; exact per-core counts and order; a
complete contiguous dynamic `S1E` palette; a source-derived low-count warning;
14 correctly titled canvases; consistent colors; highlight/fade/reset,
zoom/pan/hover/export behavior; no external assets; no prohibited payload
fields; identical standalone and archived HTML; checksum-only resume; and
focused plus existing-viewer regression tests passing.

Stop on source checksum drift, missing/reordered cores or cells, non-finite
coordinates, a noncontiguous or SO2-style label namespace, inconsistent
low-count warning values, palette drift, an external dependency, a prohibited
field, GPU use, or an invalid output checksum. No visual result can override a
failed contract check.

## Reproduction and verification

From the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so1-raw-expression-interactive

PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_so1_raw_expression_interactive.py \
  tests/unit/spatial_benchmark/test_so2_raw_expression_interactive.py \
  tests/unit/spatial_benchmark/test_so2_hl_interactive.py
```

## Verified result

The final viewer contains all 161,596 cells from all 14 ordered SO1 cores and
the fixed 14-cluster namespace `S1E0` through `S1E13`. It is visualization-only
and did not rerun PCA, neighbor construction, Leiden, training, or inference.
The embedded payload passed privacy/schema verification, the standalone HTML
and one-member ZIP agree, checksum-only resume verification passed, and visual
inspection passed.

The completed source clustering retained all cells, including 5,269 cells
below the 20-transcript floor. `S1E1` contains 4,996 of these cells (15.3955%
of `S1E1` and 94.82% of the low-count cohort), while `S1E10` is 91.53864910997318%
core 7. These cautions are not corrected by interactive highlighting. Cluster
selection can reveal spatial concentration, but it cannot establish cell type,
signaling, biological influence, or causality.

Verified SHA-256 values are:

- interactive manifest content:
  `7dfeb9e6cd78fea707d6989abe39acf98383e51242662bbf120751e34e62293a`;
- standalone HTML:
  `8cafbc7521ee3d3801ce99b82fcbc5d7a08f88a4e0a4ae5117ccc282c4be4fa3`;
- transfer ZIP:
  `ac6beb974143964a38813059230421f6b1e1f41fbd74cfc059cd5f986f56813a`;
- embedded browser payload:
  `a386fbf1f41a37e7e630a55bd0e7f97ca2f465d3b2d9b86725dda663ee6c6153`.
