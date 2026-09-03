# SO1 contextual direct-hL interactive map contract

## Status and scope

Phase: complete and verified. Outcome: success. Execution state:
`complete_verified`. This visualization-only extension reuses the
checksum-verified SO1 contextual direct-hL clustering report. The upstream run
and checkpoint were immutable, the downstream extraction/clustering report was
complete, and its public verifier passed before rendering.

The viewer reuses the finalized resolution-1.0 `S1C*` cell assignments,
palette, and tissue coordinates exactly. It performs no training, model
inference, extraction, PCA, neighbor construction, Leiden clustering, `h0`
analysis, or delta calculation. No interactive intrinsic or delta map is in
scope.

## Source and non-interference boundary

The source report is:

```text
reports/analyses/so1_14core_model_embedding_direct_knn_clustering/
  r_20260826T204925Z_06d12943_s000_f00_a01_94691119/
```

Before rendering, verify its final manifest, contextual cell table, contextual
palette, all 14 core counts/order, and all declared checksums. The renderer may
read only `core_number`, `x_um`, `y_um`, and `contextual_cluster` from the
verified table.

While upstream training is active, the isolated renderer and synthetic tests
may be implemented, but the renderer must not be executed against production
SO1 data. Never edit or access the active training bundle, mutable checkpoint,
training registry rows, worker/supervisor, process, environment, or GPUs.
Final rendering is CPU-only with `CUDA_VISIBLE_DEVICES=""` and
`automatic_enqueue: false`.

## Interaction and layout contract

Create one self-contained HTML document with exactly 14 canvases in ascending
core order 1 through 14. Each panel must have an unambiguous `SO1 Core N` title,
repository coordinate orientation, equal spatial aspect, and the same
contextual-cluster colors as the verified static report.

Clicking a contextual cluster label or button highlights that `S1C*` cluster
across all cores in its saturated palette color and draws it last while fading
all other clusters. Clicking the selected cluster again, choosing **Show all
clusters**, or pressing Escape resets the view. Provide wheel zoom,
pointer-drag pan, per-panel and global reset, restricted hover, and browser-side
combined PNG export. Cluster buttons, legend entries, panel titles, and core
numbers must remain legible for the dynamically observed cluster count.

## Privacy and portability

The browser payload may contain only numeric core number, `x_um`, `y_um`, a
compact contextual-cluster code, labels, and categorical palette. It must omit
stable cell keys and indices, FOV identifiers, expression values, `h0`, `hL`,
delta values, metadata, vendor annotations, clinical fields, donor mappings,
checkpoint/model bytes, and source filesystem paths.

The HTML must be dependency-free, include an offline content-security policy,
make no network requests, and open directly from a local file. Also create a
deterministic ZIP containing exactly that single byte-identical HTML member.
Exact coordinates make both artifacts cell-level data for authorized sharing
only; the viewer must state this visibly.

## Acceptance and stop criteria

Acceptance requires all 161,596 expected cells unless the later immutable
upstream manifest establishes a different checksum-bound count, all 14 cores
in exact order, a complete contiguous dynamic `S1C*` palette, one rendered dot
per cell, identical colors across panels, highlight/fade/reset, zoom/pan/hover
and export behavior, no external assets or prohibited payload fields,
byte-identical standalone/archived HTML, checksum-only resume, visual
inspection, and a self-checksummed final manifest.

Stop on unverified upstream or source-report state, checksum drift,
missing/reordered/duplicated cells or cores, nonfinite coordinates,
noncontiguous labels, palette drift, hard-coded cluster counts, an external
dependency, prohibited payload content, GPU visibility/use, or invalid output
checksums. A compelling visual pattern cannot override a failed contract.

## Outputs, claim limit, and future command

Outputs use the separate namespace:

```text
reports/analyses/so1_14core_hl_direct_knn_interactive/
  r_20260826T204925Z_06d12943_s000_f00_a01_94691119/
```

The final manifest must bind the source report manifest, contextual cell table,
palette, source static PNG, standalone HTML, README, browser payload, and ZIP by
path, size, and SHA-256. Cluster highlighting is descriptive and cannot
establish cell type, signaling, biological influence, mechanism, or causality.

The reproducible interface is:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so1-hl-direct-interactive \
  --run-id r_20260826T204925Z_06d12943_s000_f00_a01_94691119
```

The completed viewer contains 161,596 points and 26 contextual clusters across
all 14 cores. Clicking a cluster highlights it across every panel while other
clusters fade to opacity 0.09; reset, zoom, pan, restricted hover, and combined
PNG export are enabled. The standalone HTML SHA-256 is
`6c72a06ece8790857b968f65774ca61c4085c7298019bd1b3e6f9e7a8928dd4f` and
the deterministic ZIP SHA-256 is
`59193685f90276748825d335cba4462ef2c9123f631945800de4999b4e0cd921`.
Checksum-only replay and ZIP integrity verification passed.
