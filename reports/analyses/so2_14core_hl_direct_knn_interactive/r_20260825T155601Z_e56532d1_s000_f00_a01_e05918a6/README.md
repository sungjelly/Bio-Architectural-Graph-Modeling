# Interactive SO2 direct-hL cluster map

Open `contextual_direct_hl_leiden_resolution_1p0_spatial_14cores_interactive.html` in a current desktop browser. It is a self-contained offline
file; `contextual_direct_hl_leiden_resolution_1p0_spatial_14cores_interactive.zip` contains the identical HTML as its only archive member.

## Method and interaction

The joint analysis used the direct 256-dimensional final
contextual representation (`hL`), L2 normalization, a sparse cosine
30-nearest-neighbor graph, and Leiden resolution
1. It used **no PCA and no mean-centering**. `D0` through
`D17` are a separate label space and must not be equated with the
earlier PCA-derived `C` clusters.

- Click a D cluster to keep it bright and fade all others across every core.
- Click it again, choose **Show all clusters**, or press Escape to reset.
- Scroll to zoom, drag to pan, double-click to reset a panel, and hover for only
  numeric core, D cluster, and rounded tissue coordinates.
- **Export current view as PNG** saves the current combined 3 x 5 view.

This viewer reuses completed assignments for run `r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. It performs no
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
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so2-hl-direct-interactive \
  --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6
```
