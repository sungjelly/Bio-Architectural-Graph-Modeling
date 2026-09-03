# Interactive SO2 contextual-cluster map

Open `contextual_leiden_resolution_1p0_spatial_14cores_interactive.html` in a current desktop browser. The file is self-contained and
does not need a web server or internet connection. For transfer, `contextual_leiden_resolution_1p0_spatial_14cores_interactive.zip`
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
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. It performs no model inference, retraining, or reclustering and does
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
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so2-hl-interactive \
  --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6
```
