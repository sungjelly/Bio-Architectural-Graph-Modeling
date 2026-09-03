# Direct-hL clustering analysis contract

## Status and scope

This is an exploratory, visualization-only analysis of the completed SO2
14-core Relative-Geometric QKV run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. It does not retrain or
modify the model. The analyzed representation is the final contextual graph
embedding `hL`, after the fourth graph layer and before the expression decoder.
The failed fixed-300 continuation is not used because it did not serialize a
checkpoint.

## Question, estimand, and maximum claim

The question is how the joint Leiden partition changes when the learned
256-dimensional `hL` representation is clustered directly, without PCA or
mean-centering. The estimand is each cell's deterministic Leiden assignment on
the joint direct-hL embedding-space graph at resolution 1.0. The alternative is
that PCA removal materially changes cluster boundaries or source-core
concentration. The maximum claim is descriptive: these are model-derived
contextual clusters, not established cell types, signaling states, biological
influence, or causal effects.

## Locked inputs and computation

Reuse the checksum-verified per-core `hL` arrays extracted from the verified
terminal checkpoint `checkpoints/last.ckpt` (SHA-256
`2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d`).
Preserve all 246,063 cells and the core order 15 through 28.

The clustering pipeline is exactly:

```text
direct 256-dimensional hL
  -> L2 normalization required to evaluate cosine distance
  -> sparse 30-nearest-neighbor graph
  -> Leiden clustering at resolution 1.0
```

There is no PCA, feature projection, mean-centering, batch correction, spatial
graph reuse, or cell-by-cell dense distance matrix. The L2 normalization is
only the standard computation of cosine similarity; it is not a learned
encoder or dimensionality reduction. Cross-core neighbors are permitted in the
joint embedding graph. Use random seed 20260825 and run entirely on CPU.

## Outputs and acceptance criteria

Write to a method-specific report directory so the completed PCA-based report
remains immutable. Produce checksum-bound labels, cell and cluster summaries,
the sparse kNN graph, parameters, palettes, a combined spatial PNG and PDF, and
the portable interactive viewer specified separately. Core titles must identify
SO2 cores 15 through 28, colors must be consistent across panels, and clusters
with more than 90% of cells from one core must be retained and flagged. Use
independent labels `D0`, `D1`, ... and an independent palette so these direct-hL
clusters cannot be mistaken for the earlier PCA-based `C0`, `C1`, ...
partition.

Fail on checkpoint/source-report drift, missing or reordered cells/cores,
non-finite or zero-norm `hL`, any PCA or mean-centering step, a dense
cell-by-cell matrix, GPU visibility/use, non-deterministic labels, incomplete
figures/tables, or checksum failure.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-so2-hl-direct-clusters \
  --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6 \
  --n-neighbors 30 \
  --leiden-resolution 1.0 \
  --random-seed 20260825 \
  --device cpu \
  --cpu-threads 40 \
  --dpi 300
```
