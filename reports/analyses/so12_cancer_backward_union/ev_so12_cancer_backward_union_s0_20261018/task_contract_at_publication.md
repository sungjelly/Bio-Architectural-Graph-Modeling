# Backward cluster removal and annotated-cell retention

Phase: complete. Outcome: inconclusive. Exploratory follow-up explicitly requested
after unrestricted and one/two-cluster results were inspected.

## Contract

For each core 15/23, representation h4,h3,h2,h1,h0,PCA,scVI, resolution
0.25,0.5,1,2 and Leiden seed 0/1, start with every cluster having at least one
evaluable segmented pixel inside the supplied annotation. Include each selected
cluster's entire evaluable footprint, including pixels outside the annotation.
Remove exactly one cluster per step: evaluate every possible removal, choose the
remaining union with highest pixel Dice/F1, and break exact ties lexicographically
by retained IDs. Continue to the empty union, retaining every step even after F1
falls. Report the highest F1 on that path, number/IDs of selected clusters, and
percentages with explicit denominators. Compare the path maximum with the parent's
exact unrestricted maximum; do not assume greedy deletion finds every optimal
fixed-cardinality subset. Select among equal path maxima by fewest clusters, then
IDs. Keep the previously fixed resolution 1 seed 0 as the primary table and show
every other resolution and both seeds separately.

The question is whether fewer clusters can preserve overlap and in-region cell
coverage. Alternatives include broad annotation prevalence, fragmented cell
states, spatial misregistration, and gains from annotation-selected overfitting.
All conclusions are descriptive. No retraining, preprocessing refit, registration
refit, reclustering or test-set predictive accuracy claims.

Primary metric: pixel F1 = 2TP/(2TP+FP+FN) on the unchanged owner raster's valid
segmented-cell footprint. Selected outside pixels remain FP; omitted inside
pixels remain FN. Secondary full-region F1 includes uncovered positive pixels.
An "in-region cell" for the count-based primary retention percentage means a
cell with at least one valid owned pixel inside the annotation. This definition
matches the starting cluster eligibility and is stated on every retention report.
Also report majority-inside-cell retention (at least half its valid owned pixels
inside) as boundary-assignment sensitivity. Cell counts are unweighted, distinct
from area-weighted pixel recall. The denominator never shrinks as clusters are
removed. Report retained clusters/initial overlapping clusters, retained in-region
cells/all in-region cells, and selected visible cells/all visible cells in the core.
Cells with no valid owned pixels are not inventively assigned an annotation label.

Do not combine F1 and retention into an arbitrary weighted score. Show the full
F1-versus-retained-cell-percent curve and its nondominated points. Alongside the
unconstrained path maximum, report best F1 subject to at least 90% and at least95%
cell retention as transparent supplementary operating points. These thresholds
illustrate the tradeoff and are not clinical or confirmatory cutoffs.

## Inputs, validation and resources

Reuse all 56 immutable parent partitions, the fixed geometry and all 81,607
canonical test cells from `ev_so12_cancer_comparison_s0_20261018`, manifest SHA256
`e9a345642c160602ddd6b4c5c377c415377a6489453b8d60c82f34108a60c2b0`.
The previous compact-union evaluation is also preserved unchanged. Biological
units are the two cores; pixels and cells are observations, not independent patient
replicates. The inherited approximate registration and uncertain annotation
exhaustiveness limit interpretation.

Register `ev_so12_cancer_backward_union_s0_20261018` under campaign
`cmp_20261018_so12_cancer_backward_union`, source run
`r_20261017T044837Z_caf76350_s000_f00_a01_12df41be`. Active writes belong to
`scratch/active_runs/<evaluation>/`; publish to
`reports/analyses/so12_cancer_backward_union/<evaluation>/` after checks. CPU-only
integer enumeration and compact plotting require no GPU. Four threads, <100MiB
new storage and a 5-GiB free-space reserve; no deletion expected. Managed supervisor
execution for the complete job, source/input/config/environment hashes and logs.

Acceptance: synthetic exact-removal/tie/FP/FN tests, constant denominators and
monotone cluster/cell retention, all 112 core-partition paths complete, independent
native-raster recomputation of every step and best operating point, comparison
against the exact unrestricted upper bound, reviewed curve/table figures, complete
CSV/JSON/report, verified immutable publication, registry completion and curated
result/catalog validation. Stop on altered inputs, incorrect counts, missing
settings or storage reserve failure. Negative results are valid outcomes.

## Commands and observed outcome

Run from the repository root with `PYTHONPATH=.:src:scripts/analysis`:

```bash
/venv/main/bin/python scripts/analysis/run_so12_cancer_backward_union.py --stage initialize
supervisorctl start bagm_so12_cancer_backward_union
# Managed wrapper calls run_so12_cancer_backward_union.py --stage execute
# After automated and visual checks:
/venv/main/bin/python scripts/analysis/run_so12_cancer_backward_union.py --stage publish
/venv/main/bin/python scripts/analysis/promote_so12_cancer_backward_union_result.py
/venv/main/bin/python scripts/analysis/run_so12_cancer_backward_union.py --stage verify
```

All 56 partitions and 112 core-specific paths completed, retaining all 1,529
deletion steps through empty. An independent native-raster audit verified every
step, 3,058 binary score records, exact deletion choices/ties, fixed denominators,
Pareto flags and constrained operating points. Nine focused tests passed. The
CPU job took 28.91 seconds and peaked at approximately 282 MiB child RSS. No
scientific data were deleted, and storage remains above the 5-GiB reserve.

Every path's maximum equals its parent exact unrestricted F1 (maximum gap 0).
This verifies the best F1 for these paths/data, not all intermediate groups'
fixed-cardinality optimality. The new reporting contribution is how many
clusters and unweighted in-region cells each optimum retains.

Fixed resolution 1, seed 0 results:

| Method | Core | Best F1 | Clusters kept / initial (%) | In-region cells kept (%) |
|---|---:|---:|---:|---:|
| h4 | 15 | 0.7836 | 10/15 (66.7%) | 17,083/19,470 (87.7%) |
| h3 | 15 | 0.7921 | 6/14 (42.9%) | 14,429/19,470 (74.1%) |
| h2 | 15 | 0.7797 | 9/16 (56.2%) | 17,023/19,470 (87.4%) |
| h1 | 15 | 0.7865 | 10/18 (55.6%) | 15,891/19,470 (81.6%) |
| h0 | 15 | 0.7804 | 9/14 (64.3%) | 16,135/19,470 (82.9%) |
| PCA | 15 | 0.7710 | 6/12 (50.0%) | 15,301/19,470 (78.6%) |
| scVI | 15 | 0.7292 | 9/13 (69.2%) | 16,203/19,470 (83.2%) |
| h4 | 23 | 0.9171 | 13/16 (81.2%) | 32,604/33,352 (97.8%) |
| h3 | 23 | 0.9143 | 13/16 (81.2%) | 32,480/33,352 (97.4%) |
| h2 | 23 | 0.9115 | 14/16 (87.5%) | 32,462/33,352 (97.3%) |
| h1 | 23 | 0.9011 | 15/18 (83.3%) | 32,000/33,352 (95.9%) |
| h0 | 23 | 0.8867 | 12/14 (85.7%) | 31,878/33,352 (95.6%) |
| PCA | 23 | 0.8648 | 11/12 (91.7%) | 32,901/33,352 (98.6%) |
| scVI | 23 | 0.9041 | 11/13 (84.6%) | 31,520/33,352 (94.5%) |

Core 15 h3 has the highest primary F1, but retains only 74.1% of annotation-
overlapping cells. h4 keeps 87.7% with a 0.0085 lower F1. This illustrates the
requested tradeoff without imposing a new weighted metric. At ≥95% cell retention
on this primary removal path, h2 gives 0.7720 and h4 gives 0.7654; core 23 h4's
unconstrained optimum already retains97.8%. Rankings depend on the chosen
coverage constraint and partition setting, not a single universally best layer.

The complete tables preserve ≥90% and ≥95% constrained points, actual cluster
IDs/counts, raw cell counts, majority-inside-cell sensitivity, pixel recall,
precision, full-region scores and all settings. Figures
`figures/f1_vs_cell_retention.{png,pdf}` and
`figures/all_settings_best_points.{png,pdf}` show the primary paths and all-setting
optima. `report/index.html` embeds both figures and the primary table.
`metrics/backward_steps.csv` contains every removal step;
`metrics/backward_best_summary.csv` contains the three operating points per path.

Annotation-selected grouping, two-core coverage, incomplete segmentation and
approximate-section alignment remain limitations. Outcome is inconclusive for
independent predictive or population superiority claims. Prior published bundles
are unchanged. Root and independent figure/report reviews passed: both PNGs were
directly inspected, both PDFs passed structural/checksum checks, HTML embeds
match the reviewed PNG bytes, and all14 primary HTML/CSV/summary rows match the
audited best and constrained steps. No blocking findings remained.
