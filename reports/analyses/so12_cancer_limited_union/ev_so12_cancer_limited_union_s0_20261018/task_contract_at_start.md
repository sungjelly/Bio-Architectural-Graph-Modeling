# One- and two-cluster cancer-region agreement

Phase: execution. Outcome: pending. Exploratory follow-up requested after viewing
the unrestricted cluster-union comparison. This is a new evaluation; its parent
bundle and all clinical, model, preprocessing and clustering inputs are immutable.

## Contract

Question: how well can one or two complete Leiden clusters delineate the supplied
regions on cores 15 and 23? The hypothesis is that a small cluster group captures
useful regional structure. Alternatives include fragmented tumor-associated
states, broad clusters spanning the reference boundary, registration error and
annotation-selected overfitting. Lower agreement is a valid result.

Reuse all 56 verified parent partitions (h4,h3,h2,h1,h0,PCA,scVI; resolutions
0.25,0.5,1,2; seeds 0,1), all 81,607 cells, and the reviewed fixed owner/ROI rasters
from `ev_so12_cancer_comparison_s0_20261018`. Parent manifest SHA256:
`e9a345642c160602ddd6b4c5c377c415377a6489453b8d60c82f34108a60c2b0`.
No representation fitting, registration fitting or reclustering occurs. The
underlying train21/val5/test2 split and frozen baseline inference remain intact.

For each core and partition exhaustively enumerate every visible singleton and
every pair, maximizing footprint Dice/F1 = 2TP/(2TP+FP+FN). Report best singleton,
best exact pair and best union of at most two clusters. Break exact ties by fewer
clusters, then ascending cluster IDs. Clusters with no evaluable pixels in the
selection core are ineligible. Preserve all evaluable pixels: selected pixels
outside the outline are FP; unselected outlined pixels are FN. Restricting the
predicted positive group never removes other cells from evaluation. The primary
estimand is an annotation-selected descriptive oracle, not independent accuracy.
Also transfer the source-core choice unchanged to the opposite core, and report
secondary full-region scores (uncovered annotated pixels count as FN), all-positive
and empty controls, and the parent's unrestricted oracle as an upper bound.
Include fixed-union 50-µm registration sensitivity and within-core shuffled-label
controls where supported by the reused parent helpers. Cores are the biological
units; two cores do not justify population uncertainty or superiority claims.

Primary figures retain the previously fixed resolution 1, seed 0. Report every
resolution and seed; never select the primary setting by annotation score. Render
ground truth, h4, h0, PCA, scVI side by side for each core and a combined figure.
Use clearly contrasting colors for selected clusters and other cells, with the
same prediction color on both sides of a high-contrast annotation boundary so
spillover remains visible. Provide error maps for all seven representations,
showing TP, FP, FN and TN explicitly. Keep clinical imagery unmodified except
plot overlays and cropping already defined by the parent.

## Execution, controls and acceptance

Evaluation: `ev_so12_cancer_limited_union_s0_20261018`. Link to source run
`r_20261017T044837Z_caf76350_s000_f00_a01_12df41be` in the local registry. Active
writes belong to `scratch/active_runs/<evaluation>/`; final output belongs to
`reports/analyses/so12_cancer_limited_union/<evaluation>/`.

Use CPU only: exact enumeration over small integer cluster counts and raster
plotting require no GPU kernels or training. Budget 300 MiB incremental disk,
four CPU threads and a 5-GiB free-space floor; no deletion is currently needed.
Use a managed supervisor worker for the rendering job. Record exact commands,
source and input hashes, environment, git state, runtime and memory. Verify parent
inventory before/after; reference parent labels/rasters rather than copying them.

Require synthetic tests against independent brute-force subsets, deterministic
ties, absent/empty clusters and explicit FP/FN penalties. On real data require
single <= at-most-two <= unrestricted F1, consistent exact-pair selection, complete
56-partition coverage and independent native-pixel recomputation. Review every
new PNG and confirm plotted error counts match saved scores. Stop on altered
source bytes, invalid alignment/counts, insufficient storage or incomplete output.
Publish an immutable checksummed bundle only after checks and visual review,
complete registry records, and create a validated curated conclusion with current
catalogs. No external uploads. Existing unrelated workspace changes stay intact.

## Reproduction and outcome

Commands and observed results will be added after execution. The prior approximate
section registration, unknown annotation exhaustiveness and incomplete segmented
pixel coverage remain limitations. No retraining or biological performance claim
is authorized by this descriptive selection exercise.
