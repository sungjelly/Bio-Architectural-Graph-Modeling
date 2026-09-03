# Interactive SO1 raw-expression cluster map

Open `so1_raw_expression_leiden_resolution_1p0_spatial_14cores_interactive.html` in a current desktop browser. It is self-contained and needs
no server or internet connection. `so1_raw_expression_leiden_resolution_1p0_spatial_14cores_interactive.zip` contains the identical HTML as
its only archive member.

## Interaction and label scope

- Click `S1E0` through `S1E13` to keep one joint SO1 cluster bright while fading the others.
- Click the selected cluster again, choose **Show all clusters**, or press Escape
  to restore all colors.
- Scroll to zoom, drag to pan, double-click to reset a panel, and hover for only
  numeric core, cluster, and rounded tissue coordinates.
- **Export current view as PNG** saves the current combined 3 x 5 view.

The `S1E` namespace is specific to this independently clustered SO1 analysis.
For example, `S1E3` must not be interpreted as the same group as SO2 `E3`.
The viewer performs no normalization, PCA, neighbor construction, clustering,
model inference, training, or GPU work.

## Source-derived count-depth audit

- `S1E1`: 4,996 of 32,451 cells below 20 transcripts (15.4%).

Count depth can be associated with expression-cluster separation. These clusters are
expression-derived groups, not validated cell types, and they do not establish
signaling, biological influence, or causality. Marker-based and pathological
validation remains separate.

All 161,596 source cells are displayed. The source analysis retained low-depth
cells and cells that failed vendor QC; no vendor-QC field is included in the
browser payload.

The HTML omits stable cell identifiers, expression values, PCA scores, neighbor
edges, model data, clinical fields, and source filesystem paths. It contains
exact cell-level tissue coordinates, so share it only with authorized
collaborators.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so1-raw-expression-interactive
```
