# Local Gene Residual training with validation early stopping

Phase: planned. Outcome: pending. Authorized by the user on 2026-10-05 after
review of the completed 32-epoch initial comparison. This is a prospective
follow-up informed by those results, not a revision of that experiment's outcome.

## Task contract

Objective: automatically run the unchanged Local Gene Residual count model to a
predeclared validation stopping criterion AFTER the entire active
`cmp_20261005_so2_genemae_masking` training/evaluation matrix has completed.
Do not compete with its GPUs or alter its jobs. Deliver a durable supervised
handoff, tested early-stopping trainer, registered/replayed/indexed checkpoints,
and automatic all-seed report. Preserve the original 32-epoch runs and result.

Question: does longer optimization improve the local model's selected validation
count reconstruction, and does local context remain useful against a matched
cell-only control? Alternatives: the initial checkpoint was near its useful
limit; further training overfits; or local context adds no reproducible gain.
Compare both model seeds 0/1 for local and self-only, never only the best seed.

Estimand and inputs are unchanged: target-isolated partial-gene reconstruction,
208,696 training cells from six donor groups / twelve cores; 37,367 validation
cells from one held-out donor / two cores; no test set. Reuse pinned SO2 count
overlay, training-only transforms, permitted 22 covariates, exact inherited
training/fixed-validation masking, k<=16 non-self neighbors within 75 micrometers.
No target-derived labels/totals or identifiers. Raw/clinical inputs immutable.
No data or results leave this instance. Independent unit is the donor; seeds and
cores do not create independent patient replication.

Architecture remains the 2,329,432-parameter two-layer 512-wide MLP with direct
same-gene local corrections and Bernoulli hurdle / zero-truncated NB2 outputs.
Use fresh runs from the same deterministic seed initializations, not a claim of
uninterrupted resumption of the old 32-epoch cosine schedule. Local and self-only
use identical masks, balanced within-core batches of at most 1024, ordering,
optimizer, and stopping rules. Every epoch visits all training cells once.

AdamW initial LR .001, weight decay 1e-5, dropout .1, gradient clipping1,
bfloat16 forward with FP32 likelihood. Replace fixed cosine with
ReduceLROnPlateau on validation count NLL: mode min, factor .5, patience7,
absolute threshold1e-4, minimum LR1e-5. Validate every epoch. Save the strict
lowest-NLL checkpoint independently of patience accounting. Early stopping:
30 consecutive validation epochs without an improvement of at least1e-4
relative to the last significant-improvement anchor, and at least50 epochs.
Small improvements can accumulate against that anchor. Maximum500 epochs is
an operational safety ceiling, not a convergence result; if reached first,
report `maximum_epochs` and do not claim early stopping occurred. Nonfinite,
OOM, provenance, resource, and replay failures are separate failure outcomes.

Primary: pooled full-constant validation hurdle count NLL, lower better.
Secondary: marginal-mean raw/log MAE and RMSE, positive/zero errors, explicitly
separate conditional-positive errors, detection BCE/Brier/AP, per-core and mask
severity scores. Preserve strict best, final metrics, recent-epoch summaries,
actual stop reason, optimizer/LR/stopping history, full coverage, all seeds and
any failed attempts. Old NLL improvement threshold .002 is retained for context;
do not retroactively promote the old experiment's failed margin. A meaningful
follow-up gain over historical hurdle uses the same .002 mean NLL margin,
favorable NLL in both seeds, and RMSE no more than1% worse in either seed.
Local benefit requires lower NLL than matched self-only in both seeds.

Maximum defensible claim: validation-selected performance on the same reused
held-out donor, not sealed-test performance, guaranteed global convergence,
independent patient replication, or biological causality. Architecture and
optimizer differ from historical BAGM. This protocol fixes early stopping
before the new runs' outcomes are inspected; selection still reuses validation.

Execution: CPU-only policy/trainer/handoff tests now; durable controller waits
for all nine successful full GeneMAE slots (three masks x seeds0/1/2), empty
active dependency queue, verified final bundles, and free GPU locks/devices.
Then registered two-epoch new-runner correctness/resource pilot on GPU0, followed
by four independent full runs on GPUs0..3. Pilot cannot satisfy early-stopping
completion and is excluded from scientific averages. Existing passed model/data
checks and pilot are reference evidence; new trainer source is frozen for its
pilot and full runs. No automatic retry of failed training attempts.

Resources: use existing four RTX3090s only after dependency completion. The
previous pilot measured <1GiB peak allocated VRAM; retain >=2GiB GPU headroom,
>=5GiB free disk, and <90% host memory. Keep one job per GPU, separate outputs,
atomic checkpoints, logs, RNG/optimizer/scheduler/stopping states and complete
source/config/data/split provenance. Active run files live in shared scratch;
only finalized bundles go to shared artifacts. Supervisor owns the controller
and its foreground child processes. Controller state is durable and visible.

Scheduling acceptance for this request: code and CPU tests pass; dependency
identity and full nine-slot coverage are pinned; installed supervisor controller
is demonstrably waiting without using a GPU; start/pilot/full/report actions are
concrete and automatic. Scientific completion occurs later only after every
required run stops or fails with its exact reason and all available evidence is
reported. Never label a waiting, running, or safety-capped job converged.

Verification commands and recorded handoff status will be appended below.
