# Contextual hL clustering analysis contract

## Status and scope

This is an exploratory, visualization-only analysis of the completed locked SO2
14-core Relative-Geometric QKV run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. It does not retrain or
modify the model. The sole analyzed representation is the final contextual
graph embedding `hL`, after the fourth graph layer and before the expression
decoder. Intrinsic `h0` clustering and delta-h analysis are out of scope.

## Question, estimand, and maximum claim

The question is whether a joint unsupervised partition of contextual model
representations reveals spatially coherent model-derived groups across the 14
SO2 tissue cores. The estimand is each cell's deterministic Leiden assignment
on the joint hL embedding-space graph at resolution 1.0. The alternative is
that the partition is spatially diffuse or dominated by source-core effects.
The maximum claim is descriptive: these are model-derived contextual clusters,
not established cell types, signaling states, biological influence, or causal
effects. Marker-based and pathological validation remain separate work.

## Locked inputs and computation

Use the verified terminal checkpoint `checkpoints/last.ckpt` (SHA-256
`2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d`)
from the run bundle. Preserve the prepared expression transformation, all-zero
extraction mask, node metadata, gene order, fixed spatial graph, relative
geometry, cell order, and core order 15 through 28. Process one complete core
at a time under `model.eval()` and `torch.inference_mode()`.

All inference and analysis must run on CPU because GPUs are reserved for model
training. The reproducible command hard-disables CUDA visibility. Extraction is
resumable from checksum-verified per-core hL artifacts.

Cluster all 246,063 cells jointly. Mean-center hL, retain up to 50 exact PCA
components, L2-normalize, build a sparse cosine 30-nearest-neighbor graph, and
run seeded Leiden at resolution 1.0 with random seed 20260825. Never construct
a dense cell-by-cell distance matrix. Relabel clusters deterministically by
descending size with a stable tie-break.

## Outputs and acceptance criteria

Produce one combined tissue-coordinate map containing all cores in ascending
order (15--28), with large unambiguous `SO2 Core N` panel titles, equal spatial
aspect, the repository coordinate orientation, one borderless point per cell,
and a shared categorical legend. Save PNG and PDF plus aligned cell assignments,
cluster/core summaries, palettes, parameters, extraction and clustering
receipts, and a final checksum manifest under the run-specific report directory.

Stop and report failure on checkpoint or input drift, missing or reordered
cells/cores, non-finite hL values or coordinates, an extraction-mask violation,
metadata mutation, dense N-by-N graph construction, CPU-only policy violation,
non-deterministic labels, incomplete outputs, or checksum failure. A cluster
with more than 90% of its cells from one core is retained and flagged.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-so2-hl-clusters \
  --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6 \
  --n-neighbors 30 \
  --leiden-resolution 1.0 \
  --pca-components 50 \
  --random-seed 20260825 \
  --device cpu
```
