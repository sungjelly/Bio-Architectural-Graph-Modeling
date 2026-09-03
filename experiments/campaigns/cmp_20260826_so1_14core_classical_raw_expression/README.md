# SO1 14-core classical raw-expression clustering

## Status and objective

Phase: complete exploratory post-hoc baseline analysis. Outcome: inconclusive
for biological identity or mechanism; the computational acceptance criteria
passed.

Create one joint classical expression-space clustering for all 161,596 cells in
SO1 cores 1 through 14, then render a combined tissue-coordinate PNG/PDF and 14
individual maps. This analysis is independent of every trained graph model. Its
estimand is a transductive partition of the selected cells under a conventional
targeted-panel expression pipeline, not a test of generalization or mechanism.

The primary hypothesis is that normalized expression from the fixed 1,000-gene
CosMx biological panel contains joint structure recoverable by Leiden. A
credible alternative is that the partition is substantially driven by library
depth, vendor-QC failures, core-specific technical effects, or targeted-panel
composition rather than cell identity. Cluster IDs are model-free analysis
labels, not biological annotations.

## Inputs and leakage boundary

Use only checksum-verified nonnegative integer `expression_counts` from
`data/processed/so1_14core_relative_qkv_v1`, in locked order `SO1-C01` through
`SO1-C14`. The prepared artifact contains all 161,596 SO1 cells, no zero-total
cells, and the same ordered 1,000 biological probes in every core. Technical
controls were excluded upstream.

No checkpoint, trained representation, graph, geometry, coordinate, metadata,
core label, vendor cell-type result, clinical field, or stable cell key is used
as a computational feature for normalization, PCA, kNN, or Leiden. Core/row
provenance is retained only to verify alignment and QC and is not supplied to
those computations. Physical coordinates are loaded only after labels are
frozen for spatial maps. No batch correction is applied. All cells remain in the primary analysis,
including 5,269 cells below 20 transcripts and the 5,306 source cells marked as
failed by vendor QC; these overlapping but nonidentical groups require separate
sensitivity analysis before biological interpretation.

## Locked primary method

Run one joint CPU-only Scanpy-style workflow:

1. Normalize each cell as `counts * 162 / max(cell_total, 20)`. The target 162
   is the audited SO1 median total among cells with at least 20 transcripts.
2. Apply elementwise `log1p`.
3. Retain every nonconstant gene in the fixed targeted panel; do not select HVGs.
4. Jointly mean-center and sample-standard-deviation-scale genes, clipping to
   `[-10, 10]`.
5. Compute exact feature-covariance PCA and retain 50 components.
6. L2-normalize PCA scores, apply a deterministic seed-based insertion
   permutation that does not consult core labels, and construct a sparse cosine
   kNN graph with `k=30` across all 14 cores; map neighbors back to canonical
   cell order before Leiden.
7. Run Leiden `RBConfigurationVertexPartition` at resolution 1.0 with seed
   20260825, then relabel by descending cluster size and stable tie-break.

Use labels `S1E0`, `S1E1`, ... so they cannot be mistaken for independently
fitted SO2 `E*` labels or contextual-model `C*` labels. Flag, but do not alter,
clusters with more than 90% of cells from one core. Save cluster/core composition
and raw-library/detected-gene QC summaries. The final manifest must expose a
structured low-count warning derived from the cluster QC table.

## Controls, acceptance, and interpretation

The structural negative controls are withholding all non-expression fields
until labels are frozen, verifying ordered source/component checksums, and using
only sparse neighbor arrays. A dense cell-by-cell matrix, GPU use, silent row
filtering, reordered cells, nonfinite values, or nondeterministic labels is a
stop condition.

Acceptance requires all 14 ordered cores and all 161,596 cells, finite PCA
scores, a sparse joint graph, contiguous deterministic `S1E*` labels, complete
cluster/QC tables, one dot per cell with a consistent palette, combined PNG/PDF,
14 per-core PNGs, resumable stage receipts, and a self-checksummed final
manifest. Visual separation alone does not establish cell type. These clusters
do not establish signaling, biological influence, or causality; marker-based,
pathological, and vendor-QC filtering validation remain separate.

## Outputs and reproduction

Outputs belong under:

```text
reports/analyses/so1_14core_raw_expression_clustering/
  rawexpr_pca50_cosinek30_leiden_r1p0_s20260825/
```

Run from the repository root with CUDA hidden:

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

Preprocessing, clustering, and plotting have separate checksum-bound receipts;
a plotting retry must reuse verified PCA and clustering outputs.

## Verified result

The CPU-only workflow retained all 161,596 cells and all 1,000 nonconstant
panel genes across the complete ordered set of SO1 cores 1 through 14; zero
cells were dropped. PCA retained 50 components with total explained-variance
ratio 0.09184099685466733. The joint sparse expression-analysis graph contained
4,105,256 undirected edges, had one connected component, and yielded 14
clusters (`S1E0` through `S1E13`) with sizes from 1,734 to 44,633 cells. Leiden
quality was 5,751,530.229056849 and modularity was 0.7005081082710611.

One cluster met the prespecified core-dominance flag: 91.53864910997318% of
`S1E10` came from core 7. This cluster was retained without merging or removal.
Count depth is also a material caution: 5,269 cells had fewer than 20
transcripts, and `S1E1` contained 4,996 of them. These cells were 15.3955% of
`S1E1` and 94.82% of the complete low-count cohort. Vendor-QC-failed cells were
also retained. Consequently, the observed partition does not by itself
distinguish cell identity from count-depth, vendor-QC, targeted-panel, or
core-specific effects. The outcome is scientifically inconclusive pending
marker, pathological, and filtering-sensitivity validation.

The verified primary artifacts are:

- manifest content SHA-256:
  `aeb3de639ad5bf5107f91a06ec01afc6b91145fd2d89ec0ad76fbc153502d620`;
- combined PNG SHA-256:
  `f52a8558b14684da97f95f1aa8859d8df9ae782f2d9d44eb0cd781d66a7471ea`;
- combined PDF SHA-256:
  `426dab1428446bd27137ff33765f4bc72f627f1b2bd88ed6efcdb139d3857d91`;
- cell-cluster table SHA-256:
  `013980dd32951d173adccce6351b8ae9fccfaced0c15ce683e8df830d13226e7`.

Both checksum-only resume verification and visual inspection of the combined
map passed. These are model-independent expression-derived clusters, not cell
types, signaling states, biological influences, or causal effects.
