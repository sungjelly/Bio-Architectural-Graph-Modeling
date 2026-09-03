# SO2 14-core classical raw-expression clustering

This report is an expression-only baseline for all 246,063
cells from SO2 cores 15 through 28. It does not load or use a checkpoint, trained
model, h0/hL representation, graph, relative geometry, metadata, core identity,
or spatial coordinate during clustering. Coordinates are joined only after the
Leiden labels are frozen, solely for the maps.

## Method

The source is the checksum-verified nonnegative integer `expression_counts`
array for the fixed, ordered 1,000-gene biological CosMx panel. Technical probes
were excluded upstream. Each profile was normalized as

```text
counts * 197 / max(cell total, 20)
```

and transformed with `log1p`. The 20-count denominator floor follows CosMx 1K
guidance and prevents extreme up-scaling of very low-count cells. No cell was
filtered: 8,157 cells below 20 transcripts are retained and explicitly
flagged in the cell and cluster-QC tables.

All nonconstant panel genes were jointly mean-centered, scaled to unit sample
variance, and clipped at +/-10. Exact feature-covariance PCA retained
50 components (10.25%
of clipped scaled variance). L2-normalized PCA scores formed a deterministic
sparse cosine 30-nearest-neighbor
graph. Seeded Leiden used resolution
1 and seed
20260825. The expression-space graph is
not the model's spatial graph, and no dense cell-by-cell matrix was constructed.

The result has 12 joint expression clusters,
named `E0`, `E1`, ... by descending size. The size range is
1,353 to
56,156 cells. Clusters with more than 90%
of cells from one core: None.

`E3` and an hL label such as `C3` are unrelated identifiers; matching numbers do
not imply matching populations.

The prespecified depth audit found that `E1` contains
7,742 of the 8,157 cells below 20 transcripts
(94.9%), and its median library size is
41. This is strong evidence that count depth contributes
to that partition; `E1` must not be treated as a cell type
without a QC sensitivity analysis and independent marker/pathology validation.

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

This PCA + kNN + Leiden workflow is a conventional exploratory single-cell
clustering approach, adapted to targeted CosMx counts. The numerical resolution
1.0 is a prespecified comparison setting, not a universal cell-type resolution.
Clusters may reflect biology, count depth, technical effects, or targeted-panel
composition. They do not independently establish cell type, signaling,
biological influence, or causality. Marker-based and pathological validation
will be conducted separately; no biological cluster names are assigned here.

Method references:

- Bruker Spatial Biology, CosMx RNA QC and normalization:
  https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/normalization/
- Scanpy preprocessing and clustering:
  https://scanpy.readthedocs.io/en/latest/tutorials/basics/clustering.html
- Traag, Waltman & van Eck (2019), Leiden community detection:
  https://www.nature.com/articles/s41598-019-41695-z

## Reproduction

From the repository root, with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-so2-raw-expression-clusters \
  --normalization-target 197 \
  --library-size-floor 20 \
  --scale-clip 10 \
  --pca-components 50 \
  --n-neighbors 30 \
  --leiden-resolution 1.0 \
  --random-seed 20260825 \
  --device cpu
```

Preprocessing, clustering, and plotting are separate checksum-verified stages,
so a plotting retry does not repeat PCA or Leiden.
