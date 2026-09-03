# Interactive SO1 direct-hL cluster map

Open `contextual_direct_hl_leiden_resolution_1p0_spatial_14cores_interactive.html` in a current desktop browser. It is a self-contained
offline file. `contextual_direct_hl_leiden_resolution_1p0_spatial_14cores_interactive.zip` contains byte-identical HTML as its only member.

## Method and interaction

The joint analysis used the direct 256-dimensional
final contextual representation (`hL`, after the final graph layer and before
the decoder), L2 normalization, a sparse cosine 30-nearest-
neighbor graph, and Leiden resolution 1. It used no
PCA, mean-centering, or feature projection. The cluster namespace is
`S1C0` through `S1C25`.

- Click a cluster to keep it bright and fade the others in every core.
- Click it again, choose **Show all clusters**, or press Escape to show all.
- Scroll to zoom, drag to pan, double-click to reset a panel, and hover for only
  numeric core, cluster, and rounded tissue coordinates.
- **Export current view as PNG** saves the combined 3 x 5 view.

This viewer reuses completed assignments for run `r_20260826T204925Z_06d12943_s000_f00_a01_94691119`. It performs no
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
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -c   'from spatial_benchmark.paths import current_paths; from spatial_benchmark.so1_hl_direct_interactive import run_so1_hl_direct_interactive; print(run_so1_hl_direct_interactive(paths=current_paths(), run_id="r_20260826T204925Z_06d12943_s000_f00_a01_94691119"))'
```
