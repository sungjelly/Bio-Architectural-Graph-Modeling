# Pooled-core backward cluster removal

Phase: planned. Outcome: pending. Exploratory follow-up requested after the
per-core backward-removal figures were inspected.

## Contract

Recalculate a single pooled pixel F1 for each representation over cores 15 and
23 together. Use the original joint test-cell Leiden partitions and one shared
set of global cluster IDs applied to both cores. Sum TP, FP and FN before
computing F1 = 2TP / (2TP + FP + FN). Do not average the per-core F1 values or
reuse independently optimized per-core predictions. Larger evaluable regions
contribute more pixels; this is a micro-averaged overlap estimand.

Start with every cluster overlapping either annotation. At each step evaluate
every possible single-cluster deletion and choose the highest remaining pooled
F1; compare exact rational values, then break ties by lexicographic retained
IDs. Continue to empty. Report the highest path F1, breaking ties by fewer
clusters then IDs. Keep fixed resolution 1, seed 0 for the main figure/table;
retain all four resolutions (.25,.5,1,2) and both Leiden seeds (0,1), 56 paths.
The scientific question is how well one common cluster group matches both
annotations. The alternative is that per-core mappings exploit local annotation
prevalence or core-specific clusters. A common mapping can disagree with either
core independently; high pooled F1 alone cannot establish cross-core transfer.

The evaluable support is the unchanged valid segmented-cell footprint. Selected
outside pixels remain false positives in either core; omitted inside pixels
remain false negatives. Also preserve full-region scores including uncovered
annotated pixels. Sum disjoint per-cell areas and positive areas on the original
81,607 canonical rows. Cell retention is the unweighted number of cells with at
least one valid owned pixel inside either annotation retained / all such cells;
denominator fixed. Cluster retention is selected / initially overlapping global
clusters, counting each cluster once across the cores. Preserve majority-inside
cell retention, per-core components of the same pooled prediction, full paths,
Pareto flags and best path F1 at >=90% and >=95% pooled cell retention.

Experimental/observational units are two cores; cells and pixels are not
independent biological replicates. There is no retraining, normalization refit,
registration refit or reclustering. Annotations select the scored union, so
claims are descriptive agreement, not held-out cancer classification accuracy.
Approximate registration, incomplete segmentation, annotation completeness and
pooled weighting remain limitations. Do not infer a population-level winner.

## Inputs, controls and acceptance

Original immutable parent: `ev_so12_cancer_comparison_s0_20261018`, manifest
SHA256 `e9a345642c160602ddd6b4c5c377c415377a6489453b8d60c82f34108a60c2b0`.
Reuse the unchanged backward-deletion engine from the completed per-core
analysis with checksummed source binding. Parent BAGM run:
`r_20261017T044837Z_caf76350_s000_f00_a01_12df41be`; existing frozen encoders and
split `so12_paper_21train_5val_2test_v1`. No annotation is a model input.

Controls: zero-cluster endpoint, all-overlap starting point, independently
computed exact pooled unrestricted bound, all 56 fixed partitions, per-core
component reconciliation, and synthetic cases distinguishing pooled F1 from
macro averaging and independent per-core selection. No new biological null or
patient-level uncertainty is claimed. Exact bound checks the path maximum,
not every intermediate fixed-cardinality subset's optimality.

Acceptance: focused tests for pooling/disjointness/FP/cell retention; independent
native-raster verification of all steps, single removals/ties, best points and
fixed denominators; complete 56-path CSV/JSON outputs; legible reviewed bar and
retention table; source/output hashes, registry completion, immutable publication
and new curated record with verified sources. Stop on mismatched inputs, counts,
missing settings or <5 GiB storage. No outcome/ranking is required for success.

## Execution and resources

Register evaluation `ev_so12_cancer_pooled_union_s0_20261018` and campaign
`cmp_20261018_so12_cancer_pooled_union`. Active output:
`scratch/active_runs/ev_so12_cancer_pooled_union_s0_20261018/`. Final output:
`reports/analyses/so12_cancer_pooled_union/ev_so12_cancer_pooled_union_s0_20261018/`.
CPU-only integer aggregation and plotting; no GPU-capable training. Four CPU
threads, <100 MiB additional storage, 5 GiB reserve, no deletion expected.
Run the complete job via supervisor; preserve config, environment, source code,
append-only logs and execution resource receipt.

From the repository root with `PYTHONPATH=.:src:scripts/analysis`:

```bash
/venv/main/bin/python scripts/analysis/run_so12_cancer_pooled_union.py --stage initialize
supervisorctl start bagm_so12_cancer_pooled_union
# Managed execute runs metrics, independent audit, then rendering.
# After focused tests and direct visual review:
/venv/main/bin/python scripts/analysis/run_so12_cancer_pooled_union.py --stage publish
/venv/main/bin/python scripts/analysis/promote_so12_cancer_pooled_union_result.py
/venv/main/bin/python scripts/analysis/run_so12_cancer_pooled_union.py --stage verify
```

Deliver `figures/pooled_f1_bar.{png,pdf,svg}`, matching F1/retention table, and
numeric CSV with F1, retained cluster count/percentage and cell count/percentage.
The bar axis uses an explicitly labeled nonzero lower bound below every score;
exact four-decimal labels make the zoom auditable. All-setting outputs remain
available separately; the primary display is not chosen for the best seed.
