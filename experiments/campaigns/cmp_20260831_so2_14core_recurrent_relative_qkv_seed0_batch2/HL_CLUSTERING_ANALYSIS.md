# Recurrent contextual hL Leiden clustering contract

## Status and scope

- Phase: complete exploratory post-hoc analysis
- Outcome: inconclusive (descriptive visualization completed)
- Source run: `r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf`
- Source checkpoint: `checkpoints/last.ckpt` (SHA-256
  `01f3611897557add92b9733bff7e3940d6b19d22881bd8401340e255c9e355e2`)

The source model finished its 175 training epochs, and its checkpoint passed an
independent exact reload/replay check. The run bundle is nevertheless recorded
as failed because the post-training finalizer required an explicit canonical
fit-prediction protocol. This analysis therefore binds the checkpoint by path
and checksum, preserves that lifecycle caveat, and does not relabel the run as
completed or modify its immutable bundle.

This task extracts only the final contextual graph representation `hL` after
the fourth recurrent application of the single weight-tied graph block and
before the decoder. It does not retrain the model, inspect attention, assign
cell types, or test mechanisms. The requested visual deliverable is one static
PNG; interactive maps and PDF figures are prohibited.

## Question, alternatives, estimand, and claim

The exploratory question is whether a joint unsupervised partition of the
recurrent model's contextual representations produces spatially coherent
model-derived groups across SO2 cores 15--28. A credible alternative is that
apparent groups are diffuse, driven mainly by core/batch structure, or reflect
spatially smooth covariates rather than biologically meaningful states.

The estimand is each cell's deterministic joint Leiden assignment at resolution
1.0 on a sparse 30-nearest-neighbor graph of `hL`. The maximum defensible claim
is descriptive: a cluster is a group in this model's contextual representation,
not a cell type, communication event, predictive dependency, or causal effect.
This one-run visualization cannot establish seed stability, patient
replication, graph-specific predictive gain, faithfulness, or biological
validation.

## Locked inputs and method

Verify the checkpoint, prepared cohort, prepared graph, gene order, core order,
cell order, and input manifests against the source run. For each complete core,
use the source preprocessing and graph, an all-zero gene mask, `model.eval()`,
and `torch.inference_mode()`. Preserve full-core context and extract one row of
`hL` per cell.

Cluster all 246,063 cells jointly using the same method as the prior SO2 hL
analysis: mean-center `hL`, retain 50 exact PCA components, L2-normalize, build
a sparse cosine FAISS-HNSW 30-nearest-neighbor graph, and run Leiden at
resolution 1.0 with seed 20260825. Relabel clusters deterministically by
descending size with a stable minimum-cell-index tie-break. Never construct a
dense cell-by-cell distance matrix.

GPU extraction is permitted because all four RTX 3090 devices were idle at
task start. First run a representative pilot and record device visibility,
runtime, and peak memory. Parallel per-core extraction is allowed only when
each worker has isolated outputs and the resulting rows are reassembled in the
locked core order. PCA, graph construction, Leiden, tables, and plotting remain
deterministic CPU operations.

## Deliverables and acceptance/falsification criteria

The analysis directory must contain checksum-verified per-core embeddings,
joint labels, aligned cell assignments, cluster/core summaries, parameters,
receipts, provenance, and a final manifest. The sole rendered map must be:

`figures/contextual_leiden_resolution_1p0_spatial_14cores.png`

It must show all 14 cores in ascending order, with `SO2 Core N` titles, equal
spatial aspect, repository coordinate orientation, one borderless point per
cell, and one shared categorical legend. A cluster with more than 90% of its
cells from one core is retained and flagged rather than hidden.

Acceptance requires exact input checksums, 246,063 unique aligned cell rows,
finite arrays, the expected per-core counts, all-zero extraction masks, a
strict checkpoint load, deterministic labels on repeat, a valid sparse graph,
and a readable PNG. Stop and report failure on any checksum, shape, order,
mask, finite-value, determinism, or source-contract violation. Spatially
diffuse or core-dominated clusters are valid negative/inconclusive findings.

## Intended reproduction

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-so2-recurrent-hl-clusters \
  --run-id r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf \
  --checkpoint artifacts/runs/2026/08/r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf/checkpoints/last.ckpt \
  --n-neighbors 30 \
  --leiden-resolution 1.0 \
  --pca-components 50 \
  --random-seed 20260825 \
  --extract-devices cuda:0,cuda:1,cuda:2,cuda:3
```

## Observed result

The verified joint partition contains 21 clusters spanning 1,108--18,831 cells
and 5,950,064 undirected sparse kNN edges. Seeded Leiden reproduced all 246,063
labels exactly from the stored graph. Clusters `C0`, `C6`, `C9`, `C10`, `C14`,
and `C17` each receive more than 90% of their cells from one core; they were
retained and flagged. The map shows spatial structure, but the core-dominated
groups keep the biological interpretation inconclusive and support the stated
core/batch alternative.

The C21 repeat-forward gate differed by a maximum of `1.4305115e-6`, within the
prespecified CUDA roundoff tolerance (`rtol=1e-6`, `atol=1e-5`); C21 and the
largest core C23 used 0.69 GiB and 2.16 GiB peak reserved VRAM, respectively.
All source, shape, alignment, finite-value, sparse-graph, label-determinism, and
output-format checks passed.

The sole rendered output is
`reports/analyses/so2_14core_recurrent_contextual_embedding_clustering/`
`r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf/figures/`
`contextual_leiden_resolution_1p0_spatial_14cores.png`, with SHA-256
`9c6450d1acd3af015dd0731fe717854d40ce16947bc6315ae12421df857f691c`.
No PDF or interactive map was created.
