# Local Gene Residual training with validation early stopping

Phase: queued behind the active GeneMAE matrix. Outcome: pending. Authorized by the user on 2026-10-05 after
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

## Installed handoff and verification

Installed and started at 2026-10-05 08:03 UTC as supervisor program
`bagm_local_gene_after_current` (PID 223132 at acceptance). Observed state:
`waiting_dependency`, three of nine production slots completed, four running,
two queued. No child process or new-campaign training run existed; the four GPU
process IDs belonged to the existing campaign and were unchanged.

Frozen execution plan:
`/workspace/BAGM/state/handoffs/local_gene_early_stopping/plan.json`.
SHA256: `2f184f81208689baf4e07bc3895a2631170480584ee890c19a16301d51e37161`.
The plan pins every dependency job/config and trainer, policy, data/model,
report, controller, wrapper, dataset configuration and contract source hashes.
The wrapper installed in `/opt/supervisor-scripts` was checked against source.
The immutable scientific contract is `frozen_task_contract.md`.

Live status (waiting is not scientific completion):

```bash
supervisorctl status bagm_local_gene_after_current
cat /workspace/BAGM/state/handoffs/local_gene_early_stopping/state.json
tail -n 20 /var/log/portal/bagm_local_gene_after_current.log
```

CPU verification: **145 tests passed**, including 34 stopping-policy tests,
18 runner tests, 29 orchestration tests, 11 report tests, and existing local
model/data/metrics/report regression checks. No test used a GPU. Exact command
and full output are retained in
`/workspace/BAGM/reports/local_gene_early_stopping/verification_v1/`.
Actual CPU `--check-only` admission reported `admitted=false`,
`dependency_verified=false`, and `gpu_probed=false`; controller check confirmed
the exact nine-slot dependency matrix. Original four local/self bundles and two
historical BAGM bundles passed the new report's read-only comparison loaders.
Shell syntax, source-plan validation, and `git diff --check` passed.

The general repository doctor was not clean: it reported one historically
unindexed checkpoint, two unavailable historical six-core paths, free disk
18.783 GB below its generic 25 GB threshold, and four process records from the
active matrix. SQLite integrity was `ok`. These are retained as limitations,
not represented as passed checks. The new protocol enforces its explicit
5 GiB free-disk floor during execution. Existing active jobs were not changed.

After dependency verification, the service runs a two-epoch pilot, then both
local seeds and both matched cell-only seeds. It invokes the following report
command with the four actual completed run IDs, in the shared BAGM-root
environment supplied by the wrapper:

```bash
PYTHONPATH=src:. /venv/main/bin/python scripts/analysis/report_local_gene_early_stopping.py \
  --run-ids LOCAL0 LOCAL1 SELF0 SELF1 --version v1 --publish-result
```

Expected scientific report destination:
`/workspace/BAGM/reports/local_gene_early_stopping/v1/report.md`.
That report does not yet exist. Any dependency, source, resource, pilot or
training failure holds subsequent stages and records `held_failure`; no retry
or source change is automatic. Supervisor does not automatically restart this
controller after execution begins, preventing duplicate training attempts.
