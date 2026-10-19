# SO2 binary expression with target-cell optimizer batches

Phase: planned. Outcome: pending. This is an exploratory optimization experiment,
specified after inspecting the completed paired-core BCE result and before
measuring the target-minibatch candidates.

## Task contract

Objective: select a practical target-cell optimizer batch size on one RTX 3090,
then start a fresh, registered BCE training run with that batch size. Deliver
measured memory and throughput, verified coverage and numerical checks, a
restartable supervised full run, and a concise status report. Full scientific
completion additionally requires terminal checkpoint replay, immutable archive
verification and catalog indexing; launch is not a completed training result.

Question: does making more frequent updates using target-cell batches improve
masked-expression occurrence prediction within the existing SO2 split?
Hypothesis: a nominal 1,024-target batch is a useful resource/optimization
starting point. Alternatives: repeated context encoding makes small batches
inefficient; core-local correlated targets produce unstable updates; or low
recall comes from prevalence, limited information or model design rather than
the number of targets per update. Resource pilots discriminate memory and
throughput alternatives; validation BCE and recorded optimization trajectories
can assess optimization behavior, but cannot isolate all these explanations.

Estimand: target-isolated partial-gene binary-expression reconstruction from
the target's permitted observed genes, morphology/imaging covariates and clean
neighbor expression. Maximum claim: exploratory optimization and prediction
within one held-out donor validation split. No test-set, patient-replication,
graph-specific gain, communication or causal claim is permitted.

The fixed upstream reference is completed BCE run
`r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863`. Resolve its immutable bundle
through `spatial_benchmark.paths`, verify checksums and preserve its resolved
configuration plus data/split/environment provenance in each new run. Do not
resume or modify the completed bundle. The input dataset contains 208,696
training cells in 12 cores / six donor groups and 37,367 validation cells in
two cores / one held-out donor group. There is no test cohort. Patients are
biological units; cells and cores are nested observations. One seed is an
exploratory optimization trial, not evidence of seed stability.

Preserve the 1,000-gene panel, train-only preprocessing, permitted 22 imaging
and morphology features, four target-isolated graph blocks, Bernoulli head,
training-only prevalence initialization, unweighted masked-entry BCE, AdamW
learning rate 0.0001 and weight decay 0.00001, gradient clipping at 1.0,
mixed precision and original deterministic train/validation masks. Raw counts
are loss-only. Hidden target totals, expression-derived labels, patient IDs
and held-out-derived preprocessing are prohibited model inputs. Preserve
non-self incoming edges and the lack of a route from withheld target values
back into that target.

Change the optimizer batch explicitly: at most B target cells from one core
per update. Process a seeded permutation of the cores and a seeded permutation
of contiguous target windows within each core. This preserves the audited
contiguous-target model API and complete spatial context. These are core-local
batches, not independent randomly mixed cells across patients. Divide each
core into `ceil(n_cells / B)` balanced contiguous windows whose sizes differ
by at most one; this avoids very small remainder updates while keeping every
batch at most B. Never drop or repeat target cells. Validate
disjoint complete coverage every epoch. Keep clean input tensors available
within a core, but recompute learned source embeddings after every parameter
update; cached learned embeddings would be stale and alter gradients.

Pool BCE over all masked entries within each optimizer batch. Aggregate epoch
metrics by summed loss and entry counts, never by an unweighted mean of batch
means. Every balanced window receives one optimizer update. Record
actual batch sizes and cumulative optimizer steps. This changes gradient
noise, per-step averaging, dropout draws and steps per epoch: comparison with
the old six-update epoch is not a pure memory or rank-count equivalence test.
The exact updates per epoch are 413 at B=512, 209 at B=1,024 and 108 at B=2,048,
compared with six before. Report learning curves against epoch/target visits,
optimizer steps and elapsed time.

Primary model-selection metric: fixed-mask validation BCE, lower is better.
Secondary: Brier score, precision/recall/F1 at fixed threshold 0.5, AUPRC,
AUROC and calibration; threshold and class weights remain untuned. Compare
with the prespecified historical BCE baseline (best BCE 0.2401795339072312,
F1 0.141795) as an exploratory reference. A lower best validation value alone
does not establish graph-specific benefit. Self/context baselines, spatial
nulls and independent donors are required before that separate scientific
claim; none is silently passed by this batching task.

Positive/correctness controls: deterministic coverage and masks, finite
gradients beyond the expected zero upstream gradients at the first update,
dropout-disabled weighted split-loss/gradient equivalence at fixed weights,
existing target-isolation leakage tests, and selected-checkpoint reload/replay.
Negative controls: reject duplicate/missing targets, changed fixed masks,
stale or incompatible checkpoint/configuration and nonfinite losses. A
biological null is not needed to establish runtime feasibility.

Resource pilot: fresh two-epoch registered runs at B=512, 1,024 and 2,048 on
separate safely available GPUs 0, 1 and 2. These are batch-size candidates,
not independent scientific replicates. Use four CPU threads per process and
retain an idle device for headroom. Measure peak allocated/reserved VRAM,
sampled physical free memory, utilization, epoch wall time, targets/second,
process RSS, cgroup memory and I/O. Hardware differences and concurrent I/O
limit fine-grained timing comparisons. Recheck inventory and actual framework
visibility before launch; never displace existing processes.

Runtime acceptance: every training/validation target is covered once, masks
and losses satisfy the contract, finite parameters/gradients, no OOM, at least
2 GiB measured GPU headroom and peak allocated/reserved memory <=22 GiB.
Host cgroup usage must remain below 160 GiB. Preserve the SO2 20 GiB free-disk
floor; do not delete old artifacts to satisfy a generic higher default.
Prefer 1,024 when it passes correctness/memory and its epoch time is within
25% of the fastest passing candidate. Otherwise choose the fastest passing
candidate; retain all measurements and failures. This rule uses resource
metrics, not the most favorable two-epoch validation score. Stop on failed
coverage, leakage, nonfinite values, resource limits or inconsistent artifacts
and diagnose before another attempt. If no candidate passes, report the
failure and test a stated correction before a costly full run.

After the gate passes, launch one fresh seed-0 full run on an available GPU
under a persistent supervisor. Use the original validation-driven scheduler
(factor 0.5, patience 8, absolute threshold 0.0001, minimum LR 0.000001) and
stopping policy (maximum 300 epochs, minimum 50, patience 25, minimum
improvement 0.0001); maintain strict best selection separately from patience.
Expected stopping time is unknown because each epoch now has many more
updates. Training may validly finish with no improvement over the reference.
Falsification of a claimed predictive improvement is a final BCE that does
not beat the reference under this exploratory protocol; one favorable seed
still cannot establish robustness.

Save every run in the authoritative registry with classification, immutable
resolved configuration, exact command, source snapshot/diff, dependency and
hardware versions, data/split fingerprints, append-only metrics, status and
resource telemetry. Active files belong under `scratch/active_runs/<run_id>`.
Use atomic latest and best checkpoints; final best must be reloaded, verified,
registered and indexed with role/metric/checksum/retention before publication
under `artifacts/runs/YYYY/MM/<run_id>`. Retain compact predictions/histograms
and scalar telemetry, not unrestricted row-level matrices. Pilot runs are
diagnostic and must not be promoted as curated biological results. A later
decision-relevant completed comparison requires its own verified result record.

