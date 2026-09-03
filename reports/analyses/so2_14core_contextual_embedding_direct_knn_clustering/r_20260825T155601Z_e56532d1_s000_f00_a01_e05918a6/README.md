# SO2 14-core direct contextual-hL clustering

This CPU-only exploratory analysis reuses the checksum-verified epoch-175 hL
embeddings from `r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. It performs no model inference or
training and uses no GPU. The existing PCA-based analysis remains unchanged.

## Exact clustering method

All 246,063 raw 256-dimensional hL rows were concatenated in
locked core/cell order and clustered jointly. There was **no PCA**, **no
mean-centering**, no new encoder, and no batch correction. Each raw hL row was
L2-normalized only to calculate cosine similarity, then a sparse FAISS cosine
30-nearest-neighbor graph was constructed and Leiden was run at resolution 1.0
with seed 20260825. No dense cell-by-cell matrix was constructed. D-prefixed
labels and a separate palette distinguish this partition from the prior
PCA-based C clusters.

This direct-hL pipeline is a requested sensitivity analysis, not evidence that
removing PCA is generally preferable. Direct cosine geometry retains all 256
model dimensions, including any noisy, redundant, or anisotropic directions.

## Results

- Checkpoint: `/workspace/BAGM/artifacts/runs/2026/08/r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6/checkpoints/last.ckpt`
- Checkpoint SHA-256: `2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d`
- Completed epochs / model seed: 175 / 0
- hL shape: `[246063, 256]`
- Joint direct-hL cluster count: `18`
- Cluster-size range: `1,228` to `27,650` cells
- Clusters with >90% of cells from one core: D4, D6, D7, D8, D11, D17
- Sampled exact-neighbor recall@30 for FAISS HNSW (128 deterministic
  queries): mean `0.994792`, median
  `1.000000`, minimum
  `0.700000`; predefined mean-recall threshold
  `0.90` passed

## Core coverage

- SO2 Core 15: 38,145 cells
- SO2 Core 16: 16,229 cells
- SO2 Core 17: 9,041 cells
- SO2 Core 18: 11,696 cells
- SO2 Core 19: 19,119 cells
- SO2 Core 20: 12,155 cells
- SO2 Core 21: 4,897 cells
- SO2 Core 22: 11,443 cells
- SO2 Core 23: 43,462 cells
- SO2 Core 24: 14,506 cells
- SO2 Core 25: 13,147 cells
- SO2 Core 26: 14,856 cells
- SO2 Core 27: 24,245 cells
- SO2 Core 28: 13,122 cells

## Interpretation limits

These are model-derived contextual clusters, not established cell types,
signaling states, biological influence, or causal effects. Marker-based and
pathological validation remain separate. The cohort is fit-only/transductive,
and one trained seed does not establish representation stability.

## Reproduction

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark analyze-so2-hl-direct-clusters --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6 --n-neighbors 30 --leiden-resolution 1.0 --random-seed 20260825 --device cpu --cpu-threads 40
```

The workflow is resumable after checksum-verified clustering, so plotting can
be retried without rebuilding the direct neighbor graph.
