# SO1 14-core classical raw-expression clustering

This report is an expression-only baseline for all 161,596
cells from SO1 cores 1 through 14. It does not load or use a checkpoint, trained
model, h0/hL representation, graph, relative geometry, metadata, core identity,
or spatial coordinate as a clustering feature. Core/row provenance is retained
only for alignment and QC and is never supplied to PCA, kNN, or Leiden.
Coordinates are loaded only after the Leiden labels are frozen, solely for maps.

## Method

The source is the checksum-verified nonnegative integer `expression_counts`
array for the fixed, ordered 1,000-gene biological CosMx panel. Technical probes
were excluded upstream. Each profile was normalized as

```text
counts * 162 / max(cell total, 20)
```

and transformed with `log1p`. The 20-count denominator floor follows CosMx 1K
guidance and prevents extreme up-scaling of very low-count cells. No cell was
filtered: 5,269 cells below 20 transcripts are retained and explicitly
flagged in the cell and cluster-QC tables.

The source slide also contains 5,306 cells that failed the vendor's QC flag.
They are retained to preserve complete source coverage, and that vendor flag is
not loaded into clustering. A separate filtering sensitivity analysis is needed
before interpreting fragile or low-depth groups biologically.

All nonconstant panel genes were jointly mean-centered, scaled to unit sample
variance, and clipped at +/-10. Exact feature-covariance PCA retained
50 components (9.18%
of clipped scaled variance). L2-normalized PCA scores formed a deterministic
sparse cosine 30-nearest-neighbor
graph. To prevent core-blocked source order from influencing approximate HNSW
topology, rows were inserted in a deterministic seed-based permutation that did
not consult core labels, then neighbor indices were mapped back to canonical
cell order. Seeded Leiden used resolution
1 and seed
20260825. The expression-space graph is
not the model's spatial graph, and no dense cell-by-cell matrix was constructed.

The result has 14 joint expression clusters,
named `S1E0`, `S1E1`, ... by descending size. The size range is
1,734 to
44,633 cells. Clusters with more than 90%
of cells from one core: S1E10.

`S1E3`, an independently fitted SO2 label such as `E3`, and an hL label such as
`C3` are unrelated identifiers; matching numbers do not imply matching
populations.

The prespecified depth audit found that `S1E1` contains
4,996 of the 5,269 cells below 20 transcripts
(94.8%), and its median library size is
46. This is strong evidence that the partition is
associated with count depth; `S1E1` must not be treated as a cell type
without a QC sensitivity analysis and independent marker/pathology validation.

## Core coverage

- SO1 Core 1: 8,924 cells
- SO1 Core 2: 7,450 cells
- SO1 Core 3: 12,190 cells
- SO1 Core 4: 14,657 cells
- SO1 Core 5: 11,399 cells
- SO1 Core 6: 18,212 cells
- SO1 Core 7: 10,722 cells
- SO1 Core 8: 4,972 cells
- SO1 Core 9: 17,223 cells
- SO1 Core 10: 14,756 cells
- SO1 Core 11: 18,145 cells
- SO1 Core 12: 7,816 cells
- SO1 Core 13: 5,345 cells
- SO1 Core 14: 9,785 cells

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
  analyze-so1-raw-expression-clusters \
  --normalization-target 162 \
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
