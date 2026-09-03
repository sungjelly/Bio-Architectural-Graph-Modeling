# SO2 14-core contextual hL UMAP

This is a CPU-only two-dimensional UMAP of all 246,063 cells
from the completed joint contextual hL clustering for run `r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. The left
panel colors the immutable Leiden resolution-1.0 labels C0 through C18; the
right panel colors the exact same coordinates by numeric tissue core to expose
possible core-driven structure.

## Method

The workflow reused the checksum-verified 50-component, L2-normalized PCA
representation of contextual `hL`. It reconstructed the original directed
cosine 30-nearest-neighbor graph with single-threaded FAISS
HNSW, verified the directed-neighbor and similarity hashes against the source
clustering receipt, computed UMAP fuzzy weights with python-igraph, and ran
native UMAP for 200 epochs with `min_dist=0.3` and
seed 20260825. An immediate second layout produced the identical
coordinate hash `6430c609c9265fa5a30e9a29ecd5b986ea1386335fa09a4133579fb9b925ec53`. No model inference, retraining, or
reclustering occurred. The two native layouts took 6358.3 CPU
seconds in total; the PNG and rasterized PDF point layers were both rendered
at 300 DPI.

## Observed display

The fixed cluster labels occupy visually coherent local regions, but the most
prominent detached islands also track tissue-core identity. In the source
cluster table, C4 and C6 are respectively 99.76% and 99.87% core 15; C1 and C8
are 98.44% and 99.45% core 23; and C12 is 96.63% core 25. The central manifold
contains visibly more mixed core colors. This supports the narrow claim that
the fixed clusters have coherent displayed geometry while also flagging core
structure as a strong alternative explanation for several islands.

## Interpretation limits

UMAP preserves selected local graph relationships imperfectly. Its axes,
orientation, island area, and distances between separated islands are
visualization coordinates, not biological measurements. Apparent cluster
separation does not validate cell identities or mechanisms. Five source
clusters (C1, C4, C6, C8, and C12) contain more than 90% of their cells from one
core, so the core-colored panel is essential context. This single-seed,
transductive, all-fit analysis does not establish patient generalization,
signaling, biological influence, or causality.

The aligned Parquet table omits cell keys, tissue coordinates, expression,
learned vectors, clinical fields, and donor identifiers. `global_cell_index`
is only the zero-based locked source-report row position.

## Reproduction

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so2-hl-umap \
  --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6 \
  --n-neighbors 30 \
  --min-dist 0.3 \
  --epochs 200 \
  --random-seed 20260825 \
  --device cpu \
  --dpi 300
```
