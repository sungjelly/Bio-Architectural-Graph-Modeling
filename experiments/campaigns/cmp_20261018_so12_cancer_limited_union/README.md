# One- and two-cluster cancer-region agreement

Phase: complete. Outcome: inconclusive. Exploratory follow-up requested after viewing
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

All commands run from `/workspace/BAGM` with `PYTHONPATH=.:src:scripts/analysis`.
The managed wrapper pins CPU numerical libraries to four threads and disables
GPU visibility. The execution entry point performs metrics, independent pixel
audit, plotting and report generation sequentially, with logs and source hashes.

```bash
/venv/main/bin/python scripts/analysis/run_so12_cancer_limited_union.py --stage initialize
supervisorctl start bagm_so12_cancer_limited_union
# Wrapper invokes: /venv/main/bin/python scripts/analysis/run_so12_cancer_limited_union.py --stage execute
# After tests and both visual reviews:
/venv/main/bin/python scripts/analysis/run_so12_cancer_limited_union.py --stage publish
/venv/main/bin/python scripts/analysis/promote_so12_cancer_limited_union_result.py
/venv/main/bin/python scripts/analysis/run_so12_cancer_limited_union.py --stage verify
```

Equivalent stage commands are saved in `provenance/commands.json`. The independent
auditor intentionally refuses to overwrite its own receipt. A completed published
evaluation cannot be rerun in place; reproductions need a new evaluation ID.

The prior approximate
section registration, unknown annotation exhaustiveness and incomplete segmented
pixel coverage remain limitations. No retraining or biological performance claim
is authorized by this descriptive selection exercise.

## Audited outcome

All 56 partitions completed. The managed CPU job took 147.35 seconds, with
4.67 GiB peak child RSS and no failures. Incremental output before final metadata
is 32.7 MiB; free storage remains approximately 6.0 GiB. No data or checkpoint
deletion was needed. All parent files and the original manifest hash are intact.

At the fixed primary resolution 1, seed 0, every representation/core selected
two clusters over one. Footprint Dice/F1 is:

| Method | Core 15 one | Core 15 ≤2 | Core 23 one | Core 23 ≤2 |
|---|---:|---:|---:|---:|
| h4 | 0.5000 | 0.6900 | 0.4768 | 0.6814 |
| h3 | 0.6998 | 0.7739 | 0.4216 | 0.6889 |
| h2 | 0.6379 | 0.7137 | 0.4500 | 0.6203 |
| h1 | 0.6361 | 0.6962 | 0.3414 | 0.5596 |
| h0 | 0.5535 | 0.6798 | 0.3214 | 0.5432 |
| PCA | 0.4528 | 0.6780 | 0.4307 | 0.6098 |
| scVI | 0.2704 | 0.4228 | 0.3163 | 0.4987 |

The fixed primary setting favors h3 on both cores. h4 selects clusters [5,6]
on core 15 and [0,7] on core 23. Core 23 h4 precision is 0.9837 but recall is
0.5212: compact grouping has little outside-outline spillover but misses much
of the annotated footprint. Restricting the positive group does not remove
those missed pixels from the denominator. Exact pair and at-most-two results
are separately retained even when identical at the primary setting.

Resolution strongly changes this compact-group estimand. At resolution 0.25,
seed 0, h4 reaches 0.7502/0.8997 (cores 15/23) and h3 reaches 0.7785/0.9064.
At resolution 0.5, seed 0, h4 reaches 0.7938/0.8409. These are supplementary
settings, not replacements for the fixed primary panels; all four resolutions
and both seeds appear in the score curves and complete tables. Broad annotation
coverage makes the all-positive baseline high: 0.6347/0.8502.

Twelve focused automated tests passed. The independent auditor recomputed
9,408 binary score records, checked 4,704 selected-union records and all 140
primary shuffled replicates without discrepancies. It independently certified
exact singleton/pair ties, whole-support penalties, opposite-core transfer and
fixed-union boundary sensitivities. The renderer checked 28 native TP/FP/FN/TN
maps against saved counts. Root and an independent reviewer visually inspected
all seven PNGs. The independent review also verified all 14 PNG/PDF hashes,
one-page PDF structure and cross-reference offsets, all seven embedded HTML image
payloads, all 28 map confusion records and all 14 current/archived source files.
No blocking findings remained before publication.

Outputs include high-contrast `figures/core_15_five_panel.{png,pdf}` and
`core_23_five_panel.{png,pdf}`, the combined five-panel figure, all seven group
maps, `limited_union_errors.{png,pdf}`, `best_single_errors.{png,pdf}` and
`resolution_agreement.{png,pdf}`. Orange/blue group colors depend solely on
predicted cluster membership. The black/white reference boundary never clips
predictions. Error colors distinguish TP, FP, FN and TN. `report/index.html`
embeds all figures; `metrics/limited_union.csv` retains every setting/support
and `metrics/primary_selection_table.csv` is the compact primary table.

This experiment supports a descriptive comparison of compact regional cluster
groups. It remains inconclusive for population superiority or independent
cancer-classifier accuracy. The parent unrestricted result is preserved as a
distinct estimand, not superseded.

## Publication completed

Final bundle:
`reports/analyses/so12_cancer_limited_union/ev_so12_cancer_limited_union_s0_20261018/`.
Manifest SHA256:
`457a36fcc0dfc2cc72824bdc184b1020515969f00c670af9f392246cf7c5e228`.
The evaluation and campaign are completed in the local registry. The new curated
record is `results/descriptive_analysis/representation_comparison/res_so12_test_cancer_limited_cluster_agreement/`.
Strict source verification passed; all 13 result records validate and the
deterministic catalogs were regenerated. The prior unrestricted bundle/result
and all pre-existing workspace changes remain intact.
