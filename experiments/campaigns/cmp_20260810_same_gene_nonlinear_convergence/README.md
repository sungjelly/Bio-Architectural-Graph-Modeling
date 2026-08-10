# Nonlinear same-gene convergence audit

## Status

`refrozen after preflight correction, before any usable pilot or scientific outcome / production pending`

Frozen contract SHA-256:
`b8669ed2f0f60c45e248f1cee1dab781bbc50dd90ed97e1628050f5875a333d8`.

An initial resource-only attempt was interrupted before publication when a
concurrent static audit found incomplete wrapper source binding. Its outer test
was never evaluated and no scientific effect output was inspected; it remains
as a failed, excluded registry record. The refrozen contract now binds the
generic launcher, base runner, model, and convergence tests and enforces the
runtime gate fail-closed.

This is a post-outcome exploratory follow-up to
`cmp_20260810_same_gene_nonlinear_replication`. All 16 arm-by-fold validation
paths chose epoch 12, the previous upper boundary, and all 16 validation losses
still decreased from epoch 8 to 12. The only allowed scientific change here is
the optimization horizon: candidates become `[12, 24, 48, 96, 192]` at the
same constant learning rate.

The audit asks a narrow attribution question: did the 12-epoch budget explain
either prior frozen failure—near-versus-morphology prediction gain below 2%, or
row selectivity below 25%/50%? It is not an independent replication because the
same four outer geometry folds have already been observed.

## Identity and anchor

The frozen contract binds the four corrected attempt-2 run IDs, their
checkpoints, both antecedent aggregate files, the ridge aggregate files, the
1000-gene order, and the common 932-gene eligibility mask. Before any new result
is interpreted, the continuous long trajectory must reproduce:

- all 16 epoch-12 validation losses within `1e-7`;
- epoch-12 final-refit component metrics and predictions within `1e-7`; and
- epoch-12 Jacobians within maximum absolute error `1e-6`.

An anchor failure makes optimization attribution invalid. Outer-test results are
evaluated only after validation has selected the epoch.

## Frozen training and convergence

Data, 27 geometry groups, folds, architecture, four feature arms, AdamW,
learning rate `1e-3`, weight decay, batch size, seeds, standardization, loss,
gene eligibility, Jacobian estimand, and all seven scientific gates are
unchanged. Optimizer state remains continuous through epoch 192. A fresh model
is then refit on all non-test components for the selected epoch count, while an
epoch-12 snapshot is retained on that same refit trajectory.

For each morphology and observed-near fold, convergence passes if the selected
epoch is below 192 or `(MSE96 - MSE192) / MSE96 <= 0.001`. All eight trajectories
must pass. Otherwise the result is `optimization_inconclusive_at_192_epochs`;
the horizon is not extended after inspecting outcomes.

## Attribution and verdicts

Prediction-budget attribution requires the original 2%/3-fold gate to pass, a
paired 27-component bootstrap lower bound above zero for the selected-minus-
anchor12 gain, and selected near MSE improving over anchor12 in at least three
folds. Row-budget attribution requires the original row gate to pass and both
row fractions to improve in aggregate and in at least three fold matrices.

Verdicts follow the mutually exclusive precedence frozen in the contract:
anchor invalid, convergence inconclusive, strict selectivity (with or without
full budget attribution), both failures explained, one failure explained,
robust nonexclusive enrichment (only if every non-row gate passes), or evidence
against a 12-epoch budget explanation.

## Resource pilot

Before production, logical GPU 0 runs fold 0 with capped tuning/validation data,
all four arms, the complete 192-epoch path, and a forced 192-epoch capped refit.
It evaluates validation only; no outer-test scientific effect is inspected.
Finite outputs, analytical Jacobian controls, leakage checks, peak VRAM
`<=20.5 GiB`, and projected runtime `<=2 h/fold` are fixed gates. Pilot outcomes
cannot change the schedule, batch size, or architecture.

## Interpretation boundary

At most a positive result says that, within this fixed optimizer/model and the
already-observed held-out-geometry split, the 12-epoch budget materially limited
the prespecified metric(s). A well-converged negative result is also meaningful:
it can show that optimization budget does not explain the lack of strict
same-name selectivity in this model. Neither result establishes cell-cell
communication, mechanism, causality, patient generalization, or clinical
validity. Geometry components are leakage-control units, not patients.
