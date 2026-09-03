# SO2 14-core contextual hL clustering

This report is a CPU-only, post-training readout of the completed locked run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. No model retraining occurred, and no GPU was used by the
analysis command.

## Representation and method

`hL` is the final Relative-Geometric QKV graph-layer output immediately before
the expression decoder. All 246,063 cells were clustered
jointly, so a `C` cluster ID has the same model-derived meaning in every panel.
The workflow mean-centered hL, retained up to 50 exact PCA components,
L2-normalized the PCA scores, built a sparse cosine 30-nearest-neighbor graph,
and ran seeded Leiden at resolution 1.0 (seed 20260825). The original spatial
training graph was not reused as the clustering graph, and no dense cell-by-cell
distance matrix was constructed.

PCA + kNN + Leiden is a standard exploratory clustering workflow in single-cell
analysis. This particular input is a learned graph-contextual representation
rather than a conventional expression matrix, so its clusters are not cell-type
annotations without separate marker and pathological validation.

## Inputs

- Checkpoint: `/workspace/BAGM/artifacts/runs/2026/08/r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6/checkpoints/last.ckpt`
- Checkpoint SHA-256: `2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d`
- Model seed: `0`
- Hidden width / graph layers: `256` / `4`
- hL shape: `[246063, 256]`
- Joint cluster count: `19`
- Cluster-size range: `3,104` to `18,750` cells
- Clusters with >90% of cells from one core: C1, C4, C6, C8, C12

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

Contextual clusters represent patterns after graph-based neighborhood
processing by this trained model. They do not independently establish cell
type, signaling, biological influence, or causality. No biological names were
assigned. Marker-based and pathological validation will be conducted
separately. The SO2 cohort is fit-only/transductive, and its tissue context is
not asserted uniformly across all 14 cores.

## Reproduction

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark analyze-so2-hl-clusters --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6 --n-neighbors 30 --leiden-resolution 1.0 --pca-components 50 --random-seed 20260825 --device cpu --cpu-threads 40
```

Extraction is resumable per core through checksum-verified hL artifacts. A
plotting retry reuses the completed extraction and clustering receipts.
