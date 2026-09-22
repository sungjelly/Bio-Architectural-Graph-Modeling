# SO2 four-block hidden-layer cluster progression

This CPU-only exploratory report uses the completed locked run `r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6` and
checkpoint SHA-256 `2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d`. It extracts the actual node-encoder
state `h0` and complete post-block states `h1`, `h2`, and `h3`. Captured block-4
output was required to match the existing verified `hL` artifact within the
locked float32 replay tolerance for every core, while matching the same-process
public final state exactly; the existing hL partition was then reused.

## Method

All 246,063 cells from SO2 cores 15--28 were clustered jointly
and independently at every layer. Each pipeline mean-centered the 256-D state,
performed exact PCA50, L2-normalized the scores, built a sparse cosine FAISS
HNSW 30-nearest-neighbor union graph, and ran seeded Leiden at resolution 1.0
with seed 20260825. Cluster counts are h0=16, h1=20, h2=19, h3=19, hL=19.

The cluster IDs and colors are layer-specific. For example, `H1C0` and `H2C0`
are not asserted to be the same group. Cell-aligned contingency heatmaps and
partition metrics describe splits and merges without treating IDs as a lineage.

## Adjacent-layer agreement

- `h0 -> h1`: ARI 0.5159; NMI 0.6622
- `h1 -> h2`: ARI 0.6569; NMI 0.7396
- `h2 -> h3`: ARI 0.5779; NMI 0.7371
- `h3 -> hL`: ARI 0.6700; NMI 0.7930

## Figures

The `figures/` directory contains static PNG spatial maps for h0, h1, h2, and
h3 plus a two-row split/merge transition heatmap for h0 -> h1 -> h2 -> h3 ->
hL. The previously verified hL spatial PNG is referenced as an immutable source
artifact rather than copied or modified. No PDF, HTML, or interactive map was
created by this workflow.

## Interpretation limits

These partitions describe this fitted model's representation geometry. Spatial
coherence, cluster continuity, or cluster splitting does not establish a cell
type, biological mechanism, predictive dependency, patient replication, or
causal influence. This fit-only SO2 model does not support a generalization
claim, and no cell-type or disease labels were used to construct the clusters.

## Reproduction

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark   analyze-so2-hidden-layer-progression   --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6   --n-neighbors 30 --leiden-resolution 1.0 --pca-components 50   --random-seed 20260825 --device cpu --cpu-threads 40   --cluster-workers 4 --dpi 300
```

Extraction and each layer's clustering are independently resumable from
checksum-verified receipts.
