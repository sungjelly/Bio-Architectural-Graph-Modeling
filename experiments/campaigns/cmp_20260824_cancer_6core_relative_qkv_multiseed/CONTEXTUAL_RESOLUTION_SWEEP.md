# Contextual hL Leiden resolution-sweep contract

## Status and scope

This is an exploratory visualization-only extension of the completed locked
six-core embedding analysis for run
`r_20260824T121803Z_16144620_s000_f00_a02_62498796`. It does not retrain or
rerun the model. It analyzes only the final contextual graph embedding `hL`.
No intrinsic `h0` clustering or delta-h visualization is in scope.

## Fixed inputs and graph

The source is the checksum-valid bundle at
`reports/analyses/cancer_6core_embedding_clustering/<run_id>/`. All 117,996
cells remain in the locked core order 1, 9, 13, 15, 21, 23. The sweep must
recompute and then reuse one contextual embedding-space analysis graph with
the source analysis settings: mean centering, 50-component PCA, L2
normalization, cosine 30-NN FAISS-HNSW, and seed 20260825. The recomputed hL,
PCA-score, and kNN-edge checksums must match the source contextual clustering
receipt. Resolution 1.0 labels must exactly match the existing contextual
labels before any output is called complete.

## Resolution grid and outputs

The locked grid is `0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0`. Leiden is run
separately at each value on the same saved sparse graph. Each partition is
sorted deterministically by descending cluster size with the established
stable tie-break. Cluster identifiers are resolution-qualified so numbers at
different resolutions do not imply population identity.

For each value, create a PNG/PDF spatial figure with the six cores in the
repository-standard 2 x 3 order, original micrometre coordinates and
orientation, equal aspect, one borderless dot per cell, physical scale bars,
and an external categorical legend. Palettes are consistent within a figure
and independently namespaced across resolutions. Save aligned assignments,
cluster/core summaries, the shared sparse graph, stage receipts, README, and a
checksum-valid final manifest in a separate run-specific report directory.

## Interpretation and stop criteria

These are model-derived contextual partitions at alternative Leiden
granularities, not independently established cell types. Higher resolution is
only a clustering granularity control; marker-based and pathological
validation remain separate. Stop on source-checksum drift, hL/cell-order
misalignment, non-finite values, a contextual graph checksum mismatch,
resolution-1.0 label mismatch, missing cells/cores, dense cell-by-cell graph
construction, or output-checksum failure.

## Reproduction

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-contextual-resolution-sweep \
  --run-id r_20260824T121803Z_16144620_s000_f00_a02_62498796 \
  --resolutions 0.25 0.5 0.75 1.0 1.25 1.5 2.0 \
  --random-seed 20260825
```
