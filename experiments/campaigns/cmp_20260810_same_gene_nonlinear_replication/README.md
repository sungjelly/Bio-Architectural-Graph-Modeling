# Nonlinear same-gene cross-cell replication

## Status

`complete / strict same-gene selectivity not supported (5/7 frozen gates)`

The frozen contract was not changed after outcomes.  The verified aggregate
result is in
[`reports/analyses/same_gene_nonlinear_replication_20260810/report.md`](../../../reports/analyses/same_gene_nonlinear_replication_20260810/report.md),
registered as `eval_same_gene_nonlinear_replication_20260810_v1`.

This campaign is a prespecified nonlinear replication of
`cmp_20260810_same_gene_cross_cell_jacobian`.  It asks whether the ridge
experiment's modest predictive signal and diagonal enrichment survive a
separately optimized nonlinear neighbor branch, and whether high values are
actually exclusive to matching RNA names.

## Estimand

For every evaluated cell, all 1,000 receiver RNA values are hidden.  The model
predicts them from 22 morphology variables and, depending on the arm, a frozen
mean of up to 12 other-cell RNA vectors in the 0--25 um or 25--50 um band.  The
primary sensitivity matrix is

`mean_cell d standardized_prediction[target_RNA] /
 d standardized_neighbor_mean[source_RNA]`.

Rows are targets and columns are sources.  Because the neighbor branch is
`Linear(1000,1000) + Linear(1000,64) -> exact GELU -> Linear(64,1000)`, this
component-equal population mean Jacobian is computed exactly from the full
linear skip plus the weighted mean hidden derivative rather than by a
finite-difference approximation.  The full skip nests the ridge RNA map as a
special case, so a low-rank bottleneck is not confounded with nonlinearity.
The receiver RNA is never an input.

## Leakage controls

- The original 27 opaque spatial components and four outer folds are reused
  without modification.
- Scaling, missing-value imputation, epoch selection, and all fitted weights
  use no outer-test cell.
- Epoch selection uses only the designated validation component fold.  The
  selected epoch count is then refit from a fresh initialization on all
  non-test components.
- Graph-derived input arrays were constructed within FOV with no self edge.
- The observed and frozen within-FOV source-state permutation arms use the
  same receiver population.
- Every canonical prediction row retains component cellwise MSE, so its equal
  component mean reproduces the registered primary metric.

## Frozen model and gates

The immutable contract is `frozen_task_contract.yaml` with SHA-256
`5b9f0058e011084cf28ea3e1fc78ebba787e0cf07dd35adde19381c5f6feb66e`.
The model uses 64
hidden units, AdamW at 0.001 with 0.0001 weight decay, batch size 4096, and
candidate epoch counts 1, 2, 4, 8, and 12.  The strict same-name conclusion
requires all frozen gates, including at least 25% row top-1 and 50% row top-1%
alignment.  Thresholds will not be relaxed after observing outcomes.

The analytical identity oracle must recover a 100% row top-1 diagonal.  The
nonlinear Jacobian implementation must also match autograd and centered finite
differences before GPU production is allowed.

## Interpretation boundary

Geometry components are spatial isolation units, not patients or biological
replicates.  Verified patient, core, cell-type, and clinical labels are absent.
At most this campaign can support an exploratory nonlinear model sensitivity
within the two slides.  It cannot establish communication, mechanism,
causality, or patient/clinical generalization.

## Planned execution

The user-allocated four devices are exposed inside this container as logical
CUDA devices 0--3.  The pilot uses logical device 0.  After the resource and
scientific controls pass, folds 0--3 run concurrently on logical devices 0--3.
