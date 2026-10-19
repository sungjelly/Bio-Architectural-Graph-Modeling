# Local gene residual count reconstruction

Phase: implementation. Outcome: pending. This is an exploratory design informed
by the previous BAGM results, not a confirmatory or independent-test study.

## Task contract

Objective: implement a simple new count reconstruction model and evaluate it
against the current SO2 BAGM endpoints on the identical donor split and fixed
target masks. The user prioritizes count reconstruction without restricting the
objective to MAE. Deliver model/data/training code, leakage and likelihood tests,
registered immutable runs, indexed replayed checkpoints, and an honest comparison.

Core idea: a shallow masked-expression MLP predicts each cell; a small gene-wise
residual uses same-gene expression and detection frequency among up to 16 nearest
non-self cells within 75 micrometers. Neighbor pooling uses only clean neighboring
observations. No attention, contextual source encoder, or target return path.
The readout is a Bernoulli hurdle with zero-truncated NB2 positive counts, matching
the current hurdle observation model. A self-only ablation disables local inputs.

Hypothesis: preserving local gene evidence while simplifying the encoder improves
count likelihood and/or point reconstruction over the existing graph model.
Alternatives: same-cell coexpression is sufficient; local neighbors add noise;
observed counts are largely unpredictable; or optimization/selection explains
apparent differences. Local-versus-self comparisons test the spatial contribution.
A failed superiority criterion is a valid negative result, never a reason to
silently change the primary metric or omit a seed.

Estimand: target-isolated partial-gene reconstruction using visible target genes,
22 permitted morphology/imaging covariates, and clean non-self neighbor expression.
Reuse the training-only SO2 NB overlay: 208,696 cells from six donor groups in 12
cores; validation 37,367 cells from one held-out donor in two cores. No test split.
No identifiers, hidden target totals, expression-derived cell labels, or hidden
target measurements may enter inputs. Data remain local and raw inputs immutable.

Use the exact inherited per-target Uniform{1,...,1000} masking namespaces/base
seed, one target visit per epoch, and fixed validation masks. Report coverage and
mask hashes. Fresh random target minibatches within shuffled cores are allowed
and recorded; they differ from old paired-core optimization. All variant/seed
comparisons share masks, optimizer schedule, target exposure, and evaluation.

Primary metric: pooled full-constant hurdle count NLL, lower better. Secondary:
marginal-mean raw/log1p MAE and RMSE, marginal predictions on true-zero and
true-positive subsets, conditional-positive mean errors (clearly separated),
detection BCE/Brier/AP, and fixed-threshold precision/recall/F1. Compare raw MAE
with always-zero, and likelihood with train-fitted per-gene hurdle constants.
Do not call a mostly-zero point predictor a reconstruction improvement solely
because it beats mean predictions on MAE. Retain per-core and mask-severity scores.

Historical references: NB selected epoch 193 (NLL .3476011875, MAE .2235333539,
RMSE 1.1815561763); hurdle selected epoch 164 (NLL .3282417527, MAE .2259919385,
RMSE 1.1684848057). Both are seed 0 with different stopping exposure. Historical
comparisons establish endpoint performance on this validation split, not a pure
architecture effect or independent generalization. Latest BCE is a detection-only
reference and cannot supply count reconstruction metrics.

Pilot: registered two-epoch resource/correctness execution after focused tests.
Then four independent fixed-budget runs: local/self-only crossed with model seeds
0/1, initially 32 epochs, batch 1024, AdamW LR .001, weight decay .00001, dropout
.1, clipping 1.0. Fixed cosine decay to .0001 avoids adapting LR to outcomes.
Select each checkpoint by strict lowest validation NLL; also report the mean of
epochs 29-32, all seeds and per-core directions. Do not extend an unfavorable
budget without an explicit prospective amendment. At fixed budget, require mean
selected NLL across both local seeds at least .002 below historical hurdle NLL
and favorable NLL in both seeds; raw RMSE must not worsen by more than 1% in
either seed. This is a practical exploratory criterion, not significance.
Graph/local gain additionally requires local to beat matched self-only in both
seeds. Improvement in other metrics is described separately if primary fails.

Correctness gates: hidden-value invariance, explicit self exclusion, no-neighbor
finite behavior, same-cell visible input usability, exact masks/coverage, proper
hurdle likelihood, finite gradients, positive-count/marginal metric semantics,
atomic checkpoints and selected-checkpoint reload/replay. Stop on failed gates,
OOM/nonfinite states, GPU free memory below 2 GiB, disk below 5 GiB, or host-memory
danger; preserve failed attempts rather than blindly retrying.

Resources: four currently idle RTX 3090s; pilot on one device, then one independent
variant/seed per available device. Inspect actual framework visibility and pilot
throughput/VRAM/RSS before full runs. New services use supervisor; existing jobs
remain untouched. Active files stay in shared BAGM scratch and final bundles in
shared BAGM artifacts through the project lifecycle. Source is the existing
analysis worktree. Store hashes/config/split/data/environment/command and changes.

Acceptance: implementation and tests pass, pilot gates pass, all four registered
runs complete or failures are disclosed, selected checkpoints replay, reports and
catalogs validate. Maximum claim: simpler-model validation performance on one
held-out donor, not independent patient replication, biology, or causality.

Exact implementation/reproduction commands and final outcomes will be recorded
below before task completion.
