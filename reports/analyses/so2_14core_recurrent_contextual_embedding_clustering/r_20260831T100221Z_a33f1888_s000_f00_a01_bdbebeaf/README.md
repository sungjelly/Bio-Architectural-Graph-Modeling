# SO2 recurrent one-block contextual hL clustering

This is an exploratory post-hoc readout of checkpoint `/workspace/BAGM/artifacts/runs/2026/08/r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf/checkpoints/last.ckpt`
(SHA-256 `01f3611897557add92b9733bff7e3940d6b19d22881bd8401340e255c9e355e2`). The model completed 175 training epochs
and passed exact reload/prediction replay, but the source run remains status
`failed` because its post-training artifact finalizer rejected the canonical
fit-prediction protocol. This report does not alter or promote that run.

## Result and method

All 246,063 cells from SO2 cores 15--28 were clustered jointly.
`hL` is the 256-dimensional output after the fourth recurrent application of
one fully weight-tied graph block, immediately before the decoder. Extraction
used complete-core graphs, an all-zero gene mask, eval/inference mode, FP32, and
empty decoder targets. It then used exact mean-centered PCA50, L2 normalization,
a sparse cosine FAISS-HNSW 30-nearest-neighbor graph, and Leiden resolution 1.0
with seed 20260825. The result has 21 clusters,
with sizes 1,108--
18,831.
Clusters with >90% of their cells from one core: C0, C6, C9, C10, C14, C17.

GPU pilot gates:

- Core 21: 1.42 s, peak reserved 0.69 GiB on cuda:0
- Core 23: 9.21 s, peak reserved 2.16 GiB on cuda:0

Only one static rendered map was created:
`figures/contextual_leiden_resolution_1p0_spatial_14cores.png`. No interactive
map or PDF was produced.

## Interpretation limit

These are descriptive model-derived groups from one seed and one transductively
fit cohort. Spatial coherence does not establish cell identity, predictive
dependency, faithfulness, patient replication, biological mechanism, or causal
influence. Core/batch structure and spatially smooth covariates remain credible
alternative explanations.

## Reproduction

Run from the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-so2-recurrent-hl-clusters \
  --run-id r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf \
  --checkpoint artifacts/runs/2026/08/r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf/checkpoints/last.ckpt \
  --n-neighbors 30 --leiden-resolution 1.0 \
  --pca-components 50 --random-seed 20260825 \
  --extract-devices cuda:0,cuda:1,cuda:2,cuda:3
```
