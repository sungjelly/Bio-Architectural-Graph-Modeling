# SO2 BCE optimization audit

Phase: planned. Outcome: pending. Exploratory study: the historical BCE result
was inspected before this design. New hyperparameter-screen outcomes have not
been inspected. The user expanded the target-batch task on 2026-10-01 to test
whether batch size and other hyperparameters limit training, and explicitly
lowered the safety floor for future runs to **5 GiB of free disk space**.

## Task contract

Objective: test plausible optimization and regularization explanations for the
current reconstruction performance; deliver registered controlled runs,
verified checkpoints, complete comparison tables/curves, a concise report,
and a conclusion that distinguishes supported, negative and unresolved
explanations. Choose a training configuration only from the declared evidence.

Question: do target-cell batch size, learning rate or regularization materially
improve fixed-mask BCE prediction under the existing SO2 split? Primary
hypothesis: a tested configuration improves validation BCE by at least 0.002
relative to the incumbent at matched target exposure and retains improvement
in a second optimization seed. This margin is a practical decision threshold,
not a biological effect threshold or statistical significance claim.

Alternatives: more updates only accelerate early fitting without improving
the eventual endpoint; correlated core-local batches or excessive learning
rate destabilize optimization; regularization changes fitting but not held-out
performance; class prevalence and a 0.5 decision threshold explain much of low
recall; or data/model limitations dominate the tested hyperparameters. Improved
validation BCE across seeds supports an optimization contribution. Lower
fixed-mask training loss without validation gain is compatible with a
generalization mismatch. No tested improvement does not prove an architectural
ceiling or global hyperparameter optimality.

Estimand: target-isolated partial-gene binary-expression reconstruction using
the target's visible genes, permitted imaging/morphology and clean neighbor
expression. Units: optimizer executions/seeds for optimization stability;
cells nested in cores/donors for prediction. The inherited split contains
208,696 training cells in 12 cores/six donor groups and 37,367 validation cells
in two cores/one held-out donor group. There is no test split. Maximum claim:
exploratory optimization behavior and validation-selected prediction in this
split, not independent patient replication, graph-specific gain or causality.

Inputs and provenance: completed BCE
`r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863`, its immutable resolved
configuration and checksummed data/graph/split/preprocessing manifests, and
the independently registered target-minibatch resource pilots. Preserve all
failed pilots. Access data under the repository's legacy data guide and
`spatial_benchmark.paths`; raw/clinical inputs remain immutable and local.

Keep the 1,000-gene panel, 22 permitted covariates, graph architecture,
target-isolated source/query routing, train-only preprocessing and prevalence
initialization, uniform per-cell masks, exact once-per-epoch coverage,
validation masks and unweighted masked-entry BCE fixed. No target-derived
labels, hidden totals or identifiers enter the model. Keep threshold 0.5 and
class weights fixed; threshold diagnostics cannot be counted as optimization
improvements. Preserve mask and data-order seeds across paired model seeds.

## Prespecified comparisons

The incumbent is target batch cap **1,024**, initial LR **0.0001**, dropout
**0.1**, weight decay **0.00001**, and model seed 0. This means balanced
contiguous windows, shuffled within shuffled cores, full clean context, fresh
learned source embeddings after every update, and 209 updates per epoch.

Screen A: nine fresh seed-0 runs, batch caps {512, 1,024, 2,048} crossed with
initial learning rates {0.00003, 0.0001, 0.0003}, each for eight complete epochs.
Keep dropout 0.1 and weight decay 0.00001. Include one fresh single-GPU paired-
core reference with the original six updates/epoch, LR 0.0001 and identical
other settings. Historical best BCE 0.2401795339072312 remains a separate
long-trained reference; it is not the matched eight-epoch incumbent.

All eight-epoch runs visit 1,669,568 training targets. They perform 3,304,
1,672, 864 or 48 optimizer steps for B=512, B=1,024, B=2,048 or paired cores.
Within one batch size, LR contrasts match both exposure and update count.
Between batch sizes, comparisons match exposure only. Show curves against
epochs/target visits, actual steps and elapsed time; do not infer exact
equal-step superiority from unequal-step epoch endpoints. The paired-core
reference controls rank count while exposing the update-frequency change.

Rank the grid by mean validation BCE in epochs 7–8, with ties within 0.0001
resolved by shorter measured training time and then proximity to incumbent
settings. Best-checkpoint scores remain separately reported. Selecting a
transient minimum is not the screening rule. Report all nine outcomes,
failures, per-core directions and secondary metrics. Do not hide a bad seed.

Screen B: at the grid-selected batch/LR, run three single-factor eight-epoch
controls: dropout 0, dropout 0.3, and weight decay 0.01. Compare with the same
selected grid run. The original weight decay is too small for a short
0-versus-0.00001 contrast to be informative about direct AdamW shrinkage.
Keep attention dropout 0 and gradient clip norm 1.0 unchanged.

Boundary rule: if the selected learning rate is at a grid boundary and clears
the 0.002 practical margin, run one additional eight-epoch adjacent LR at the
selected batch (0.00001 below or 0.001 above). Do not keep expanding a search
until a favorable outcome appears. Numerical failures at the boundary count
as evidence against that setting.

Confirmation: fresh incumbent and selected finalist runs at model seeds 0
and 1, each for 32 complete epochs, sharing fixed masks/order seeds. Both
variants receive the same epoch and validation schedule. Use the mean of
epochs 29–32 as the prespecified endpoint. Support a material optimization
gain only if the mean paired-seed BCE reduction is at least 0.002 and its
direction is favorable in both seeds. Report both seed effects and both
validation-core directions; the cores still represent only one donor.
If the incumbent wins the screen, use the best non-incumbent configuration as
the confirmation comparator; do not launch duplicate identical variants.

Convergence diagnostic: compare means from epochs 25–28 and 29–32. If either
variant improves by more than 0.0005, label its final-performance estimate
unconverged. The fixed-budget optimization comparison remains reportable,
but a negative claim about eventual performance is inconclusive. A longer
study requires a new declared budget and compatible continuation or fresh
matched runs; never alter a finalized successful run. This audit does not
automatically expand an unpromising search or infer an architectural ceiling.

If the user chooses an approximately one-hour budget instead, use a separately
declared bounded screen of the three learning rates at B=1,024 plus the
paired-core reference, followed by the two alternative batch sizes at the
selected LR when time permits. Its conclusion is provisional; seed and longer-
run confirmation remain explicitly unfinished. A budget preference does not
change any completed run or retrospectively change its selection criterion.

## Metrics, controls and acceptance

Primary: fixed-mask, natural-frequency validation BCE (lower better).
Secondary: Brier score, AUPRC/AUROC, calibration, predicted positive frequency,
precision/recall/F1 at the unchanged 0.5 threshold. A 0.01 F1 increase is a
secondary practical flag, not the selection criterion. Provide per-core
metrics and curves; do not treat cells as independent patient replications.

Diagnostic references: evaluate train-fitted per-gene prevalence probabilities
under the same fixed validation masks. Where feasible, evaluate the selected
checkpoint on fixed training masks in evaluation mode to distinguish fitting
from the online/dropout training objective. If that diagnostic is absent,
explicitly refrain from interpreting the raw train–validation gap. Historical
training and validation positive prevalences differ (about 11.94% and 8.12%).

Correctness controls: unchanged default paired-core behavior; deterministic,
disjoint exhaustive target windows; mask invariance across batch sizes;
fixed-weight split-forward and weighted-gradient equivalence; genuine
optimizer steps with fresh learned source embeddings; finite gradients and
parameters; checkpoint/resume transaction and selected-best replay. Upstream
gradients are expected to be zero on the very first update because the output
weights initialize to zero. Negative checks reject missing/duplicate targets,
stale code/config/checkpoints, changed validation masks and nonfinite states.

No biological null or self/context ablation is required for the narrow
optimization comparison. They remain mandatory before claiming graph-specific
predictive information or a biological dependency. Broad pooled predictive
metrics cannot pass those separate gates.

Stop a run on failed correctness, nonfinite loss, OOM, less than 2 GiB measured
GPU headroom, peak allocated/reserved VRAM over 22 GiB, cgroup memory at or
above 160 GiB, or less than 5 GiB free disk. Diagnose and preserve the failed
attempt before retrying. Do not stop merely because a candidate looks poor
before its prespecified budget ends. Use no automatic blind retries.

## Resources, artifacts and completion

Use the four safely available RTX 3090 GPUs for independent candidates/seeds,
one process per GPU, with four CPU threads each and separate locks, run IDs,
configurations and logs. Current framework is PyTorch 2.13.0+cu130/CUDA 13.0;
record actual versions and all-device occupancy for every launch. Perform a
new registered smoke using the new guarded protocol before broader screening.
Pilot compatibility must be audited, not assumed after executable edits.

Every run needs authoritative registry registration, classification, exact
command and source snapshot/diff, resolved immutable hyperparameters,
data/split/environment fingerprints, append-only metrics, resource use,
atomic checkpoints, status and completion marker. Active outputs remain in
`scratch/active_runs/<run_id>`; publish verified immutable bundles to
`artifacts/runs/YYYY/MM/<run_id>`. Index every retained best checkpoint in the
schema-v3 catalog. Keep scalar/histogram summaries, not unrestricted row-level
predictions. Reports belong in the owning local report directory and must
reference all expected/included/failed runs and source checksums.

Completion requires all runs demanded by the chosen budget branch, artifact
and checkpoint verification, focused tests and practical repository checks,
the resource and hyperparameter comparison report, explicit limitations and
negative outcomes, updated reproduction instructions, preserved unrelated Git
changes, and a validated curated result plus regenerated catalogs for any
decision-relevant completed conclusion. Appropriate negative wording is
“no tested setting produced a substantial gain under this budget,” not
“hyperparameters cannot matter.”

