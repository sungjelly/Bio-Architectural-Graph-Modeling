# SO2 raw-expression clustering baseline contract

## Status, question, and scope

Phase: complete exploratory post-hoc baseline analysis. Outcome: technically
complete and biologically inconclusive pending marker/pathology validation; all
required artifacts are checksum-verified.

The question is whether the 246,063 cells in SO2 cores 15 through 28 form
reproducible expression-space communities under a conventional single-cell
clustering pipeline that is completely independent of the trained graph model.
The primary hypothesis is that library-size-normalized expression from the
curated 1,000-gene CosMx panel contains joint structure that Leiden can recover
across cores. A credible alternative is that partitions are driven largely by
count depth, core-specific technical effects, or the preselected composition of
the targeted panel rather than cell identity.

This is an unsupervised, transductive description of the selected cells, not a
generalization test. Cells are observational units; cores are the tissue/sample
units used to audit composition. Cluster separation is not evidence of cell
type, signaling, biological influence, or causality. Marker and pathological
validation remain separate.

## Inputs and prohibited information

Use only the nonnegative integer `expression_counts` arrays from the immutable,
checksum-bound SO2 prepared cohort, in locked order `SO2-C15` through `SO2-C28`.
The arrays are direct copies of the raw biological-probe counts; technical
controls were excluded during the audited cohort preparation, leaving one
identically ordered 1,000-gene targeted panel.

The clustering transformation may use no checkpoint, trained-model output,
graph edge, relative geometry, spatial coordinate, morphology/imaging metadata,
core/slide/FOV label, clinical field, vendor cell-type label, or stable cell
identifier. Core number and physical coordinates are joined back only after the
partition is frozen, for composition summaries and spatial plotting. No batch
correction or cross-core integration is applied.

All 246,063 cells remain in the primary analysis to preserve one-to-one
comparability with the completed hL map. Raw total counts and detected-gene
counts are retained as QC covariates for auditing but do not enter clustering.
Cells are not silently filtered. Very low-depth cells may form a technical
cluster; that outcome must remain visible.

## Locked primary method

Run one joint, CPU-only, Scanpy-style classical pipeline:

1. Normalize every cell with the CosMx 1K safeguard
   `counts * 197 / max(cell_total, 20)`. The denominator floor is the platform
   recommendation for avoiding extreme up-scaling of cells below 20 counts;
   197 is the locked median total among this cohort's cells with at least 20
   counts.
2. Apply elementwise `log1p` to nonzero normalized values.
3. Use all nonconstant genes from the fixed 1,000-gene targeted panel. Do not
   select highly variable genes: this panel was already biologically selected,
   and an additional data-dependent filter would change the comparator.
4. Mean-center and sample-standard-deviation-scale each retained gene, then
   clip standardized values to the locked interval `[-10, 10]` so a very small
   number of extreme measurements cannot dominate the covariance.
5. Compute an exact feature-covariance PCA and retain 50 components.
6. L2-normalize PCA scores and build a deterministic sparse cosine 30-nearest-
   neighbor graph jointly across all cores.
7. Run seeded Leiden with `RBConfigurationVertexPartition`, resolution 1.0,
   seed 20260825, and relabel clusters by descending size with a stable tie-break.

The kNN/Leiden settings intentionally match the completed hL analysis so the
representation—not downstream clustering hyperparameters—is the main changed
factor. The graph is an expression-space analysis graph and is unrelated to the
model's spatial training graph. No dense cell-by-cell matrix may be created.
Raw-expression labels use `E0`, `E1`, ... and have no correspondence to hL
labels `C0`, `C1`, ... merely because numbers match.

scVI is not the primary method because it is another learned model, its latent
space adds training choices, and the official scvi-tools guide notes that GPU is
effectively required for fast inference. PCA provides the clearer classical and
CPU-only comparator requested here.

## Predictions, controls, and falsification

Support for genuine expression structure would include multiple clusters with
nontrivial membership across cores and cluster composition not explained solely
by total-count/detected-gene QC. Evidence for the technical alternative includes
clusters almost entirely restricted to one core or clusters whose count-depth
distribution is extreme. Clusters with more than 90% of cells from one core are
flagged, never automatically removed or merged.

Negative/leakage controls are structural: clustering receives expression only;
row order is checksum-locked; coordinates and core labels are withheld until
after labels are complete; no model or graph artifact is opened; and the kNN
implementation exposes only sparse neighbor arrays. The hL assignments do not
enter preprocessing, tuning, or label selection.

Acceptance requires exact source checksums, all 14 ordered cores, all 246,063
cells, 1,000 ordered biological genes, nonnegative integer counts, no zero-total
cell, finite transformed values and PCA scores, a sparse joint kNN graph,
deterministic contiguous labels, one dot per cell in every spatial panel,
consistent colors, a complete table and cluster/core/QC summaries, PNG/PDF
combined maps plus individual core maps, and a self-checksummed final manifest.
Failure or stop conditions include source drift, dropped/reordered cells,
non-finite values, a dense N-by-N allocation, nondeterministic labels, GPU use,
or incomplete/checksum-invalid outputs.

## Outputs and reproduction

Write the versioned report beneath:

```text
reports/analyses/so2_14core_raw_expression_clustering/
  rawexpr_pca50_cosinek30_leiden_r1p0_s20260825/
```

The workflow is resumable after preprocessing and after clustering. Run from
the repository root with CUDA hidden:

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

Method references: the Scanpy documentation for `normalize_total`, PCA,
neighbors, and Leiden; Traag, Waltman, and van Eck (2019), *From Louvain to
Leiden: guaranteeing well-connected communities*; and the scvi-tools model
guide for the alternative learned latent-space workflow.

## Completed result (2026-08-26)

The locked pipeline retained all 246,063 cells and all 1,000 nonconstant genes.
The first 50 PCs explain 10.25% of the clipped, scaled log-expression variance.
The sparse graph is connected and Leiden resolution 1.0 produced 12 clusters
(`E0` through `E11`) ranging from 1,353 to 56,156 cells, with modularity 0.7276.
No cluster crosses the prespecified >90%-single-core flag, although `E2`, `E3`,
and `E8` are strongly enriched for cores 15 (84.5%), 23 (80.5%), and 16 (83.5%),
respectively.

The count-depth alternative remains important: `E1` contains 7,742 of the 8,157
cells below 20 transcripts (94.9% of all such cells), and its median library
size is 41 versus the cohort median of 189. This makes `E1` especially unsafe to
interpret as a biological cell population without separate QC sensitivity,
marker, and pathology review. The completed clustering is therefore a valid
classical expression baseline, not a validated cell-type map.
