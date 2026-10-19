# Backward cluster removal and annotated-cell retention

Phase: execution. Outcome: pending. Exploratory follow-up explicitly requested
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

Commands and observed results will be recorded after execution.
