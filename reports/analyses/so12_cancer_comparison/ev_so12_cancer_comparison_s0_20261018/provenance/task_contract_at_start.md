# Test-core representation and annotated-region comparison

Phase: planned. Outcome: pending. This is exploratory: h0/h4 maps and the
annotated histology have already been inspected. User authorized execution,
baseline fitting-population decisions, and removal of unnecessary derived
artifacts if storage requires it; protected raw/clinical inputs remain immutable.

## Contract

Compare frozen epoch-194 BAGM h0,h1,h2,h3,h4 with PCA and scVI on the identical
81,607 cells of cores 15 (38,145) and 23 (43,462). Produce a five-column figure
per core (annotated histology, h4, h0, PCA, scVI), supplementary intermediate
layers, full resolution tables, registration diagnostics and a portable report.
Question: how well can unions of representation clusters delineate the supplied
hand-outlined regions? Greater h4 agreement is a hypothesis, not a requirement.
Alternatives include already sufficient cell-autonomous expression/morphology,
overpartitioning, core effects, annotation selection and section misregistration.

The latest source run is
`r_20261017T044837Z_caf76350_s000_f00_a01_12df41be`, selected best epoch 194.
It trained on 21 cores, validated on 5 and reserved cores 15/23 for testing.
Therefore fit PCA and scVI on the same 264,491 training cells, with validation
cores permitted only for scVI early stopping; do not fit either on the test
cores. PCA uses 50 components of the existing frozen train-standardized Pearson
features (centering fitted on training only). scVI uses raw counts of the same
1,000 biological targets, a 50-dimensional latent state, two hidden layers of
128 units, negative-binomial likelihood and slide batch covariate. Both slides
are represented in training. No clinical labels, core identifiers, coordinates
or morphology enter these expression-only baselines. scVI inference uses frozen
weights and known training batch categories, with no query adaptation. This is
not an input-matched ablation of BAGM's morphology information.

Reuse the verified h0/h4 embeddings and graphs from
`ev_so12_paper_test_h0_s0_20261018` and
`ev_so12_paper_test_hL_s0_20261017`. Extract h1-h3 with full-core, all-observed,
deterministic FP32 inference. Require exact h0/h4 reproduction and ordinary
prediction invariance. Preserve the selected checkpoint, fitted normalization,
gene order, cell order, sources and all existing immutable bundles.

For every representation construct a joint test-cell exact cosine k=30 graph,
row L2 normalization, undirected unweighted union, no spatial edges added.
Use RBConfiguration Leiden to convergence at resolutions 0.25, 0.5, 1, 2,
seeds 0 and 1. Reuse the already fixed resolution-1 h0/h4 labels. Primary
figures use resolution 1 seed 0 regardless of annotation agreement. Report all
resolutions and both seeds; report ARI stability without a post-hoc threshold.
Test-cell joint clustering is transductive descriptive partitioning, even
though representation fitting respects the held-out-core split.

Clinical inputs are `data/clinical/Core_15.png` and `Core_23.png`. Blue outlines
are interpreted as the user-designated cancer-region annotations; do not infer
diagnosis from image appearance. Registration must use tissue geometry and
image landmarks, never expression, cluster labels or overlap optimization
against cancer boundaries. Inspect matching, orientation, scale and coverage.
If trustworthy registration cannot be established, produce comparison maps
and explicitly withhold pixel scores rather than invent correspondence.

On a method-independent raster with one deterministic cell owner per pixel,
measure Dice/F1 = 2TP/(2TP+FP+FN), IoU, precision and recall. Multiple clusters
may be unioned; predicted pixels anywhere outside the outlined region within
the evaluable core count as FP. The primary support is the aligned segmented
cell footprint. Report coverage, overlap ownership and an additional full-region
score/coverage ceiling so intercellular gaps are visible. Pixels outside the
outlined regions are treated as reference-negative for this annotation-agreement
exercise, not proof of histologically cancer-free tissue. Uncertain boundaries
and section differences require a sensitivity audit.

Report two distinct estimands: (1) per-core best-union Dice, choosing the union
using that core's annotations, explicitly an ORACLE descriptive upper bound;
(2) choose the cluster union using the other core's annotation, freeze it, then
score the held-out core. Joint cluster IDs permit this two-direction transfer.
Never present an oracle or resolution-selected maximum as unbiased predictive
accuracy. Optimize union exactly by sorting cluster TP/pixel area and evaluating
all prefixes. Include all-positive and empty baselines, perfect synthetic and
spillover controls, and shuffled-label negative controls. Cores, not pixels,
are biological units; two cores and one encoder seed do not support population
confidence intervals or superiority claims.

## Execution and acceptance

Run synthetic metric/registration/feature invariance checks first. Register the
post-hoc evaluation locally. Write active files only under its scratch directory,
record configuration, input/code/environment hashes and append-only progress.
Use managed supervisor workers, GPU locks and bounded CPU concurrency. Budget
at most 1.5 GiB incremental disk and preserve a 5-GiB reserve; reuse references,
avoid duplicate datasets and save only compact baseline weights and test latent
arrays. Before any deletion, record exact reproducible derived targets and why
they are not dependencies of this comparison or the selected source. No deletion
is currently necessary. No external data uploads.

Acceptance requires correct split membership and cell/gene order, finite
embeddings, untouched source bindings, verified graph and Leiden settings,
all requested maps, complete resolution/seed metrics (or an evidenced external
registration blocker), reviewed registration and plotted figures, checksummed
publication and registry completion. Stop the affected branch on alignment,
source identity, numeric, storage or reproducibility failure and retain evidence.
An unfavorable PCA/scVI/BAGM comparison is a valid result. Code, exact commands,
timings, final paths and outcomes will be recorded below after execution.
