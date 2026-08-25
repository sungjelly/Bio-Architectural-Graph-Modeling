# Six-core embedding-cluster analysis contract

## Status and objective

Phase: post-hoc locked-readout analysis. Classification: exploratory. This
analysis was specified after model fitting, so it is not confirmatory and does
not alter or retrain the completed model.

The objective is to extract, for every manifested cell in Cancer cores 1, 9,
13, 15, 21, and 23, the output of the trained `NodeEncoder` (`h0`) and the
output of the final relative-geometric QKV graph block before decoding (`hL`),
then describe their joint six-core Leiden structure and the magnitude
`||hL - h0||2` in tissue coordinates.

Concrete deliverables are six checksum-bound compressed embedding files, one
aligned cell/cluster Parquet table, intrinsic and contextual cluster summaries,
independent clustering receipts and palettes, twelve per-core maps per
representation family, three combined PNG/PDF figure pairs, a checksummed
manifest, and a reproducible CLI command.

## Scientific question and permitted claim

The question is whether the fitted model's intrinsic and graph-contextualized
representations exhibit different descriptive joint structure across the six
cores, and where contextual representation-change magnitudes occur spatially.
The primary working hypothesis is that graph processing changes representation
geometry enough to produce a contextual clustering distinct from the intrinsic
clustering. A credible alternative is that `hL` largely preserves `h0`, or that
apparent clusters mainly track core-specific preprocessing, morphology,
expression distributions, or fitted-cohort batch structure.

Predictions that distinguish these explanations are independently constructed
intrinsic and contextual kNN graphs and Leiden labels, their cluster-size and
core-composition profiles, and the distribution of raw `||hL - h0||2` values.
Clusters with more than 90% of cells from one core are retained and flagged.

The estimand is descriptive model-representation structure in the transductive
fitted cohort. The maximum permitted claim is that cells are similar in a
specified trained representation or changed in representation by a specified
magnitude. The analysis does not independently establish cell type, signaling,
biological influence, mechanism, or causality.

## Units, inputs, and leakage scope

The observational units are all 117,996 manifested fitted cells; the six cores
are spatial samples and are not treated as independent population-level
replicates. There is no train/validation/test split and no generalization claim.
The selected immutable model unit is seed 0, run
`r_20260824T121803Z_16144620_s000_f00_a02_62498796`, because no analysis run or
campaign primary is configured and multiple seeds are complete.

Required inputs are the verified canonical `last` checkpoint, the prepared
six-core cohort manifest and per-core NPZ files, and the corresponding fixed
spatial graphs and relative-geometry caches. Extraction uses the stored gene
order, standardized `log1p` expression, transformed 22-field metadata, graph,
and 70-field relative geometry without refitting or batch correction. Raw
coordinates remain plotting/geometry inputs and never enter the `NodeEncoder`.
Every extraction mask is all-zero so every standardized expression feature is
supplied.

## Primary settings, controls, and metrics

Intrinsic and contextual embeddings are clustered in completely separate
pipelines. Each pipeline mean-centers its input, retains up to 50 PCA
components, L2-normalizes the scores, constructs a deterministic cosine 30-NN
graph without an N-by-N distance matrix, and runs Leiden at resolution 1.0 with
seed 20260825. Cluster IDs are sorted by descending size with a stable
cell-index tie-break.

Primary descriptive metrics are the number and size range of clusters, cluster
proportions, per-core composition, PCA variance explained, embedding kNN graph
components, Leiden quality/modularity, and global/per-core raw delta-norm
summaries. No metric is an optimization target. Controls are exact six-core
coverage, stable row keys, finite-value and nonzero-variance checks, all-zero
mask receipts, pre/post metadata checksums, separate kNN graph construction,
deterministic replay tests, prediction equality with intermediate outputs
enabled, and a single global p1-p99 plotting scale for delta norm.

## Acceptance, falsification, and stop criteria

Acceptance requires all six immutable input checksums to verify; every
manifested cell to appear exactly once and in identical order across `h0`,
`hL`, coordinates, delta values, and labels; equal finite embedding dimensions;
unchanged predictions and metadata; six embedding files; all requested tables;
all three combined PNG/PDF figure pairs and individual maps; and a manifest
whose declared checksums verify.

The working hypothesis is descriptively weakened if the two independently
constructed clusterings and their summaries are effectively indistinguishable
or if delta norms are uniformly negligible; that is a valid result, not a
reason to tune settings. Execution stops rather than silently repairing data if
the checkpoint/bundle is not verified, preprocessing or graph checksums drift,
any cell is dropped or duplicated, coordinates/embeddings are non-finite,
prediction invariance fails, metadata changes, deterministic graph construction
cannot be provided, or an output checksum cannot be verified.

Expected compute is one complete core at a time on an available GPU for exact
inference, followed by CPU PCA, sparse approximate-neighbor graph construction,
and Leiden clustering. The command is resumable from verified embedding files;
a plotting failure must not repeat model inference.

## Verification command

From the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-embedding-clusters \
  --run-id r_20260824T121803Z_16144620_s000_f00_a02_62498796 \
  --n-neighbors 30 \
  --leiden-resolution 1.0 \
  --pca-components 50 \
  --random-seed 20260825
```
