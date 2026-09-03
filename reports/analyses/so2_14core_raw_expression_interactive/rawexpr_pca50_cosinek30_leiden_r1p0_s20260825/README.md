# Interactive SO2 raw-expression cluster map

Open `raw_expression_leiden_resolution_1p0_spatial_14cores_interactive.html` in a current desktop browser. The HTML is self-contained and
needs no server or internet connection. `raw_expression_leiden_resolution_1p0_spatial_14cores_interactive.zip` contains the identical HTML
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
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so2-raw-expression-interactive
```
