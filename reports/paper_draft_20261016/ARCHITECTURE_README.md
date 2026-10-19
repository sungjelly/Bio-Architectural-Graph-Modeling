# BAGM architecture artifacts

These draft diagrams document the **original Relative-Geometric QKV** model
selected for the paper: the active joint-slide October 16 experiment. They do
not describe the September geometry-modulated cosine-QKV variant.

## Deliverables and scope

- `BAGM_architecture_editable.pptx`: three slides—English overview, block
  equations, and Korean overview. All visible content is native editable
  PowerPoint shapes, connectors, or text boxes. Sources and model distinctions
  are recorded in speaker notes.
- `BAGM_architecture_figure.pdf` and `.png`: compact English overview for either
  manuscript; 6.8 × 1.55 inches and 300-dpi PNG.
- `BAGM_architecture_preview.pdf` and `BAGM_architecture_slide_*_preview.png`:
  vector layout proofs and previews generated from the same drawing calls.
  These are not Microsoft PowerPoint or LibreOffice renderings.
- `build_architecture.py`: reproducible CPU-only build script.

The deliverable is an architecture description, not an experiment. No raw data,
clinical identifiers, empirical tissue images, invented results, or new model
performance claims are included. Leiden is a downstream exploratory readout;
attention/Jacobian interpretation is a planned analysis. A spatial graph can
encode regional smoothness and cell identity, so the diagram does not establish
graph-specific gain or a biological mechanism. Those alternatives require the
paper's proposed matched controls.

## Sources and exact distinctions

Read-only sources:

- `src/spatial_benchmark/relative_qkv_graph_transformer.py`
- `src/spatial_benchmark/models.py`
- `src/spatial_benchmark/so12_relative_qkv_data.py`
- `experiments/campaigns/cmp_20261016_so12_pearson_relative_qkv_train_validation/README.md`
- The same campaign's `configs/seed0_attempt2.yaml`

Four independent 256-wide blocks use eight 32-dimensional heads, an additive
70-D relative-geometry bias encoder (70→128→8), and 1,024-wide FFNs. Geometry
does not directly enter values. The model has no graph self-loops; residual
paths retain same-cell information. Residual/FFN dropout is 0.1; attention
dropout in the active configuration is **0.0**. The k200 graph is radial
stratified and symmetrized, so final degree is not necessarily 200.

For each cell, the masked-gene count is uniform on 0–1,000, and masked positions
are sampled without replacement. Input Pearson features are recomputed from
visible counts only; an explicit gene mask and exactly 22 permitted morphology
fields enter the node encoder. Fixed fully observed Pearson targets are for
supervision only. Training uses ten mask views per core per epoch. The 20-core
training and 8-core validation groups have disjoint donors; there is no
independent test partition. Historical clustering findings are not attributed
to this current experiment.

## Reproduction and verification

From `/workspace/BAGM`:

```bash
/tmp/bagm-paper-env/bin/python reports/paper_draft_20261016/build_architecture.py
```

Dependencies: `python-pptx`, `reportlab`, `pymupdf`; FreeSans fonts in
`/usr/share/fonts/truetype/freefont`; NanumGothic regular/bold fonts in `assets/`.
PowerPoint may substitute fonts if they are not installed; all text remains
editable. Korean slide labels use `h0`, `h4`, `um`, and `delta` to avoid missing
mathematical glyphs in the Korean typeface.

Acceptance checks passed: three PPTX slides and three note pages; no embedded
picture objects; 87 native text/shape objects and 14 connectors; all objects
within slide bounds; text widths within intended boxes. All three slide proofs
and the compact figure were visually inspected. The PDF figure is vector, and
the PNG is 2,040 × 465 pixels. No training source, active run, registry, or
protected input was modified. Unrelated repository changes were preserved.
