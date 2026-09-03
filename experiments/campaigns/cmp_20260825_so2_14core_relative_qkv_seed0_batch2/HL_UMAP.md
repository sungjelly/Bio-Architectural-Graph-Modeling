# Contextual hL UMAP visualization contract

## Status and scope

Phase: complete exploratory visualization. Outcome: supported with a major
core-structure caveat.

Create a deterministic two-dimensional UMAP of the completed joint contextual
`hL` clustering for locked run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. The visualization must
reuse the existing resolution-1.0 Leiden labels, their fixed palette, the
50-component L2-normalized PCA representation, and the locked global cell
order. It must not run model inference, retrain, recluster, rename clusters, or
modify the completed source report.

## Scientific question and claim boundary

The question is whether the local neighborhood structure used for the joint
contextual clustering can be displayed in two dimensions with visually
coherent fixed cluster labels. The primary hypothesis is that cells assigned
to the same fixed Leiden cluster occupy locally coherent UMAP regions. A
credible alternative is substantial overlap in the display, or apparent
separation driven primarily by tissue-core identity rather than a shared
contextual representation.

Predictions that distinguish these explanations are a cluster-colored panel
and a companion core-colored panel using the identical coordinates. Cluster
coherence accompanied by strong core separation remains evidence of possible
source-core structure, not a shared biological state.

The estimand is a seeded two-dimensional graph layout of all 246,063 fitted
cells. Cells are display units; the fourteen tissue cores are the relevant
observational units for assessing core structure. There is no inferential
effect estimate or direction of improvement. The maximum defensible claim is
that the fixed model-derived contextual clusters have the displayed local
geometry under the locked UMAP parameters. UMAP axes, global orientation,
island area, and distances between separated islands have no direct biological
meaning. The clusters are not validated cell types, signaling states,
mechanisms, or causal effects.

## Locked method and controls

Verify the completed source report and its checksum-bound PCA, label, palette,
and cell-table artifacts before use. Reconstruct the directed cosine
30-nearest-neighbor graph from `contextual_pca_l2_normalized.npy` with the same
single-threaded FAISS-HNSW settings and locked row order used by the Leiden
analysis. The directed-neighbor and cosine-similarity hashes must equal the
source receipt. Convert distances to UMAP fuzzy weights with python-igraph and
run its native two-dimensional UMAP layout with:

- `n_neighbors=30` and cosine distance;
- `min_dist=0.3`;
- `epochs=200`;
- deterministic PCA-based initialization;
- random seed `20260825`;
- CPU-only execution.

The existing Leiden labels at resolution 1.0 are an immutable overlay, not an
output of UMAP. The core-colored panel is the prespecified confounding control.
No null is needed for this visualization-only deliverable; it must not be used
as a statistical cluster-validity test.

## Acceptance, falsification, and stop criteria

Acceptance requires exactly 246,063 finite UMAP rows in locked order, exact
core counts for cores 15 through 28, all and only labels C0 through C18, the
unchanged source palette, matching directed-kNN hashes, a deterministic
coordinate hash on an immediate repeat, PNG and PDF figures, an aligned
privacy-minimized coordinate table without cell keys or expression values, a
parameter/version receipt, and a final checksum manifest. The cluster and core
panels must use identical point coordinates and include explicit interpretive
caveats.

Stop on source-manifest or file-checksum drift, incomplete or reordered rows,
neighbor-graph mismatch, non-finite coordinates, CUDA visibility or GPU use,
palette/label drift, nondeterministic coordinates, incomplete outputs, or final
checksum failure. A visually overlapping or core-driven layout is not a
workflow failure; it is a valid negative or cautionary visualization outcome.

## Resources, outputs, and verification

The workflow is CPU-only, uses no raw or clinical data, and reads only the
completed contextual clustering report. Write additive outputs under:

```text
reports/analyses/so2_14core_contextual_embedding_umap/
  r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6/
```

Expected artifacts are the two-coordinate NumPy array, a privacy-minimized
aligned Parquet table, PNG and PDF figures, a README, a UMAP receipt, and a
checksum-bound final manifest. Run and verify from the repository root:

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

PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_so2_hl_umap.py
```

## Completed result

The run completed on CPU for all 246,063 cells and retained all and only the
fixed labels C0 through C18 in the locked row and core order. The reconstructed
directed-neighbor and cosine hashes matched the source receipt. The positive
UMAP fuzzy graph contained 5,960,471 unique undirected edges with source-bound
logical checksum
`9959a807bb452bfa26d351929197e35950912d5c1e71b156d8a33e526115f641`.
Two independent native-igraph layout calls under the locked RNG state were
bitwise identical, with coordinate checksum
`6430c609c9265fa5a30e9a29ecd5b986ea1386335fa09a4133579fb9b925ec53`.
The two layouts required 6,358.3 CPU seconds in total. Process RSS observed
during the layout was approximately 1.67 GiB; peak RSS was not instrumented.
No GPU was visible or used.

Visual review supports local coherence of the fixed labels, but prominent
detached islands coincide with source-core structure. C4 and C6 are 99.76% and
99.87% core 15, C1 and C8 are 98.44% and 99.45% core 23, and C12 is 96.63%
core 25. The central manifold is more core-mixed. The result therefore meets
the visualization acceptance criteria while strengthening the prespecified
alternative explanation for several islands. It does not establish shared
cell types, patient generalization, biological mechanism, or causality.

The complete bundle is checksum-verified against the current source report.
Both panels use identical coordinates, the PNG and PDF point layers are locked
to 300 DPI, and the aligned table contains no cell keys, tissue coordinates,
expression values, learned vectors, clinical fields, or donor identifiers.
This visualization-only extension does not alter the source run or warrant a
new curated scientific result record.

Verification completed with 15 focused clustering/UMAP tests passing, all 7
infrastructure CLI smoke tests passing, successful idempotent CLI reopening of
the final bundle, and a clean `spatial_benchmark doctor` result. The broader
repository suite completed with 1,276 passing, 1 skipped, and 34 failures in
unrelated campaigns that require absent locked scratch materializations,
prepared geometry, or an audited sibling repository. No failure involved the
SO2 contextual clustering, UMAP implementation, report bundle, or new CLI
route.
