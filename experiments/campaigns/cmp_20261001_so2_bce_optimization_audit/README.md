# SO2 BCE optimization audit

Phase: complete. Outcome: negative for a material tuning gain at the fixed
32-epoch budget; eventual-performance comparison remains inconclusive.
Exploratory study: the historical
BCE result was inspected before this design. The frozen task contract preceded
the new screen outcomes; all required stages are now complete. The user
expanded the target-batch task on 2026-10-01 to test
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

## Current status and commands

The immutable premeasurement plan is `frozen_task_contract.md`. The several-hour
branch completed without extending its frozen budgets or selection rules.

All three corrected target-batch resource pilots completed and passed checkpoint
replay and resource checks. Their report is
`/workspace/BAGM/reports/runtime/so2_binary_target_minibatch/v1/report.md`.
B=1,024 averaged 219.78 seconds/epoch, versus 206.05 for B=2,048 and 262.48
for B=512, so the prespecified runtime rule chose 1,024. The three initial
attempts failed because minimum_epochs equalled maximum_epochs; the actual
epoch-controller regression now passes, and the failed bundles are retained.
These resource pilots do not establish optimization superiority.

The optimization runner and guarded launcher remained frozen throughout testing.
Focused checks cover actual epoch transitions, mask/coverage invariance,
optimizer application, exact prevalence reference, fixed-training evaluation,
artifact publication and checkpoint catalog indexing. Repository-wide checks completed: 1,929 passed, 35 failed and four skipped.
The failures are the unchanged missing legacy-input cases. Doctor passed
SQLite integrity and the 5 GiB floor, with the preexisting catalog gap and
two missing six-core paths still exposed. Check logs and hashes are under
`/workspace/BAGM/reports/runtime/so2_bce_optimization_audit/prelaunch_v1/`. The new runs use the user-authorized
5 GiB disk floor. No changes are made to historical resolved configurations.

Run from the source worktree after setting the shared resource roots:

```bash
export BAGM_ROOT=/workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export BAGM_CACHE_ROOT=/workspace/BAGM/cache
export BAGM_EXPORT_ROOT=/workspace/BAGM/exports
export BAGM_REPORT_ROOT=/workspace/BAGM/reports
export BAGM_RESULT_ROOT=/workspace/BAGM/results
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

/venv/main/bin/python scripts/train/launch_so2_bce_optimization_audit.py --profile smoke --batch-mode target_window --target-batch-size 1024 --gpu 0
/venv/main/bin/python scripts/train/launch_so2_bce_optimization_audit.py --profile smoke --batch-mode core_pair --gpu 1
```

The two smokes ran concurrently under the supervisor programs
`bagm_so2_bce_audit_smoke_target_window` and `bagm_so2_bce_audit_smoke_core_pair`.
A smoke ID must reference a completed, verified run of the same batch mode and
identical executable fingerprints before a screen/confirmation can launch:

```bash
/venv/main/bin/python scripts/train/launch_so2_bce_optimization_audit.py --profile screen --batch-mode target_window --target-batch-size 1024 --learning-rate 0.0001 --dropout 0.1 --weight-decay 0.00001 --seed 0 --gpu 0 --smoke-run "$WINDOW_SMOKE_ID"
/venv/main/bin/python scripts/train/launch_so2_bce_optimization_audit.py --profile screen --batch-mode core_pair --gpu 1 --smoke-run "$PAIR_SMOKE_ID"
/venv/main/bin/python scripts/train/launch_so2_bce_optimization_audit.py --profile confirmation --batch-mode target_window --target-batch-size "$SELECTED_BATCH" --learning-rate "$SELECTED_LR" --dropout "$SELECTED_DROPOUT" --weight-decay "$SELECTED_DECAY" --seed 1 --gpu 3 --smoke-run "$WINDOW_SMOKE_ID"
```

The final settings and immutable stage JSON plans are recorded below.
The stage scheduler executes one launcher per GPU, holds
pending jobs after a failure, and never retries automatically:

```bash
/venv/main/bin/python scripts/operations/run_so2_bce_audit_stage.py --plan "$STAGE_PLAN"
```

Verification uses the focused audit tests, full repository tests where practical,
`PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor`, completed
bundle checksums, and verified schema-v3 checkpoint catalog entries. Known
missing historical dataset/legacy-script failures are reported separately.

### Recorded grid schedule

Stage plan: `/workspace/BAGM/state/logs/so2_bce_optimization_audit/plans/grid_a_v1.json`.
The stage cannot start until both registered smoke IDs in this plan pass.
GPU0: B512/LR0.00003, then B2048/LR0.0001.
GPU1: B1024/LR0.0001, then B2048/LR0.00003, then paired-core reference.
GPU2: B2048/LR0.0003, then B1024/LR0.00003, then B1024/LR0.0003.
GPU3: B512/LR0.0001, then B512/LR0.0003.
Every job is seed0/dropout0.1/weight-decay0.00001 and eight complete epochs.
The script records all child commands, status, exit codes and launcher logs.

```bash
/venv/main/bin/python scripts/operations/run_so2_bce_audit_stage.py --plan /workspace/BAGM/state/logs/so2_bce_optimization_audit/plans/grid_a_v1.json
```

The durable supervisor entry is `bagm_so2_bce_audit_grid_a_v1`, autostart and
retries disabled. Do not rerun a stage ID or overwrite a completed stage plan.

### Execution log

- Both registered smoke runs completed and passed immutable-bundle, checkpoint
  replay/catalog, mask/coverage and resource verification:
  `r_20261001T073730Z_b3feecba_s000_f00_a01_42b8b0fd` (target windows) and
  `r_20261001T073732Z_2bde0602_s000_f00_a01_e44fd1ae` (paired cores).
- Screen admission with each mode-matched smoke passed on the frozen executable
  fingerprint. The ten-run grid stage started around 07:48 UTC on 2026-10-01,
  under supervisor `bagm_so2_bce_audit_grid_a_v1`. Its operational status and
  immutable plan copy live in
  `/workspace/BAGM/state/logs/so2_bce_optimization_audit/grid_a_v1/`.
- Selection used the prespecified endpoints and complete grid accountability.

### Completed grid and follow-up allocation

The complete ten-run grid finished at 09:26:39 UTC on 2026-10-01. All ten
attempts succeeded. The verified report is
`/workspace/BAGM/reports/analyses/so2_bce_optimization_audit/grid_v1/report.md`;
its validation curves and endpoint heatmap were visually reviewed.

The prespecified terminal mean selected B512/LR0.0003/dropout0.1/WD0.00001:
BCE 0.23970766 versus 0.24099956 for the B1024/LR0.0001 incumbent, a reduction
of 0.00129191. This falls below the 0.002 practical margin, so the declared
boundary rule does not authorize an adjacent LR extension. It does not prove
that a larger learning rate could never help.

At matched eight-epoch target exposure, the incumbent target-window regimen
reduced BCE by 0.00719035 and increased F1 by 0.03931919 relative to the fresh
paired-core reference. The step counts were 1,672 versus 48. This supports an
early optimization benefit of the changed update regimen, without establishing
equal-step or eventual superiority.

Screen B used B512/LR0.0003 with dropout0 on GPU0, dropout0.3 on GPU1,
and WD0.01 (dropout0.1) on GPU2. Each received eight complete epochs. GPU3
ran the predetermined incumbent seed0 confirmation for 32 epochs concurrently.
The remaining confirmation jobs started after complete Screen B selection;
neither the primary endpoints nor selection/stop criteria changed.

The reviewed immutable plan is
`/workspace/BAGM/state/logs/so2_bce_optimization_audit/plans/regularization_b_v1.json`.
Its preparation receipt binds the verified grid report by checksum. Execute:

```bash
/venv/main/bin/python scripts/operations/run_so2_bce_audit_stage.py --plan /workspace/BAGM/state/logs/so2_bce_optimization_audit/plans/regularization_b_v1.json
```

Supervisor program: `bagm_so2_bce_audit_regularization_b_v1`. Generate the
regularization report after its three screen jobs complete; the concurrent
incumbent confirmation need not finish before that selection.

Independent completed-grid diagnostics are retained under
`/workspace/BAGM/reports/runtime/so2_bce_optimization_audit/`:
`gradient_grid_v1/README.md` verifies 17,568 updates and 480 block/head epoch
rows, with finite gradients, no later zero minima, and zero clipping activations
(maximum full-model norm 0.234644 versus clip 1). This does not show that every
subpath or spatial input contributes useful predictive information.
`grid_a_runtime_observation_v1/final_summary_v1.md` verifies all 460 bundle
files and records runtime/resource qualifications, including rare CPU quota
events, occasional software power-cap flags and one software thermal flag.
Mean worker-sampled GPU utilization was 23.3–35.3%; these point samples do not
measure a causal idle fraction or isolate a hardware timing effect.

The separately registered execution profile completed with exact agreement on
all checked first-epoch metrics/counts/masks. Its reviewed report is
`/workspace/BAGM/reports/runtime/so2_bce_execution_profile/r_20261001T090308Z_b64e2ba9_s000_f00_a01_286bb53e/v2/report.md`.
Many kernel launches, transfers, receiver-chunk computations and CPU mask work
motivate future controlled implementation experiments. Logging scopes were
small. Nested/overlapping timings and profiler overhead prevent summing them
into wall-time fractions or claiming a validated speedup. Version v1 is retained
and superseded by v2, which separates CUDA annotation events from kernel
candidates. Training sources and the audit criteria remained frozen.

### Completed regularization screen and confirmation

All three new regularization screens completed successfully on 2026-10-01.
The verified `regularization_v1` report is under the audit analysis report root;
its curves and table were reviewed. At B512/LR0.0003, terminal BCE was
0.23966941 for dropout0, 0.23970766 for dropout0.1, 0.24067777 for dropout0.3,
and 0.23970513 for WD0.01 with dropout0.1. The dropout0, dropout0.1 and WD0.01
settings are within the prespecified 0.0001 tie band. The shorter measured
epoch time selected dropout0; this does not establish a meaningful predictive
advantage of removing dropout or a causal runtime gain under shared-host load.

The distinct confirmation comparator is B512/LR0.0003/dropout0/WD0.00001.
Its fresh seed0 and seed1 runs used GPUs0 and1; fresh incumbent seed1 used GPU2.
The incumbent seed0 used GPU3. Every slot completed 32
epochs and selection used the mean of epochs29–32, with convergence checked
against epochs25–28. Both paired seed effects must be positive and their mean
BCE reduction at least 0.002 for the material-gain criterion to pass.

The reviewed plan and parent-report checksum receipt are
`/workspace/BAGM/state/logs/so2_bce_optimization_audit/plans/confirmation_c_v1.json`
and `confirmation_c_v1.preparation_receipt_v1.json` in that directory.
Supervisor program: `bagm_so2_bce_audit_confirmation_c_v1`, with autostart,
restart and retries disabled. Reproduction entry point:

```bash
/venv/main/bin/python scripts/operations/run_so2_bce_audit_stage.py --plan /workspace/BAGM/state/logs/so2_bce_optimization_audit/plans/confirmation_c_v1.json
```

The plan permitted only GPUs0/1/2 and did not manage or interrupt the existing
GPU3 incumbent. All four confirmation slots passed final verification.

### Final result and decision

All 19 main audit runs completed: two protocol smokes, ten batch/LR/reference
screens, three additional regularization screens and four fresh confirmations.
No main audit run failed. The three earlier resource-pilot controller failures
remain preserved in the separate target-minibatch campaign. The independent
execution profile also completed. No frozen epoch budget was extended.

The primary endpoint averages validation BCE over epochs 29–32, then compares
the two paired optimization seeds. F1 below is the mean per-epoch F1 at the
unchanged 0.5 cutoff, not F1 recomputed by pooling epochs or selected checkpoints.

| Configuration | Batch cap | LR | Dropout | Weight decay | Mean terminal BCE | Mean terminal F1 |
|---|---:|---:|---:|---:|---:|---:|
| Incumbent | 1,024 | 0.0001 | 0.1 | 0.00001 | 0.23894460 | 0.148459 |
| Selected comparator | 512 | 0.0003 | 0 | 0.00001 | 0.23876134 | 0.154255 |

The paired BCE reductions were 0.00005516 and 0.00031137, averaging
**0.00018326**, below the prespecified 0.002 material-gain margin. The F1
increase of 0.005797 also falls below the secondary 0.01 flag. One validation
core worsened slightly in seed0 (BCE reduction −0.00007243); the other three
seed/core contrasts improved. These two cores share one validation donor.
This is a negative result for a material gain from the tested tuning at this
budget, not evidence of equality, global optimality or an architectural ceiling.

Incumbent seed0 improved by 0.00059637 between the final four-epoch windows,
exceeding the frozen 0.0005 convergence flag. Comparator BCE worsened slightly
late in both seeds. Eventual long-training performance therefore remains
**inconclusive**. This audit does not authorize a claim that hyperparameters
cannot matter. Preserve B1024/LR0.0001/dropout0.1/WD0.00001 as the working
configuration: the comparator did not pass the declared criterion for a
material improvement. Its recorded trainer durations were 143–144 minutes,
versus about 130 minutes for the incumbent; device and concurrency differences
prevent interpreting this as a controlled runtime effect.

The positive early batching result remains separate: after eight matched
epochs, target-window B1024 reduced BCE by 0.00719035 and raised F1 by
0.03931919 versus paired-core updates. Both visited 1,669,568 targets, but
performed 1,672 versus 48 updates. Core-scale updates limited early learning
under this protocol; the longer comparison provides little evidence of a
material additional benefit from the tested smaller batches, LR and
regularization settings.

All four selected-best checkpoints also passed evaluation on fixed epoch-one
training masks. Training BCE was 0.298312/0.298023 for the incumbent and
0.295422/0.294915 for the comparator, versus the common train-fitted gene-prior
training BCE of 0.340518. All models beat the corresponding validation prior
(0.278080). The splits differ in prevalence and composition, so their raw BCE
difference does not establish a controlled generalization gap or prove
overfitting. Selected-best diagnostics are distinct from the terminal endpoint.

Ancillary cutoff analysis used only the four selected-best checkpoint
histograms, with predefined cutoffs and no threshold selection. At 0.5, F1
was 0.14729–0.15683, precision 68.70–70.32% and recall 8.23–8.85%. At the
histogram-aligned cutoff 0.2001953125, F1 was 0.30860–0.31548, precision
31.88–34.11% and recall 28.18–31.22%. Thus the low F1 at 0.5 is partly an
operating-point issue. Lowering the cutoff trades precision for recall; it does
not improve probability ranking or prove generalization. Always predicting
positive yields F1 0.150166, illustrating why F1 alone is insufficient;
model AP was 0.30566–0.30766 versus a prevalence reference of 0.081178.
Histogram discretization bounds are not statistical confidence intervals.

Independent confirmation verification checked all 188 immutable bundle files,
32-epoch mask agreement, within-variant ordered update agreement across seeds,
indexed checkpoint identity/replay and fixed-training diagnostic binding.
Each run visited 6,678,272 targets; updates totaled 6,688 for each incumbent and
13,216 for each comparator. Across 39,808 updates, all 768 block/head/full-model
epoch gradient summaries were finite with positive means, no persistent zero
minima and no clipping (maximum norm 0.265267 versus clip 1). This rules out
the observed persistent-zero/clipping failure modes at this reporting scale;
it does not demonstrate that every internal pathway or graph input is useful.

Confirmation maximum shared cgroup memory was 59.704 GiB; maximum allocated
and reserved GPU memory were 2.822 and 3.557 GiB. Recorded physical GPU
headroom stayed above 20 GiB. The 5 GiB free-disk guard passed throughout.
Low GPU utilization and the separate execution profile motivate controlled
chunk-size, staging and mask-generation benchmarks. Profiling alone establishes
neither a causal share of idle time nor a validated implementation speedup.

Verified outputs:

- Main comparison: `/workspace/BAGM/reports/analyses/so2_bce_optimization_audit/confirmation_v1/report.md`.
- Curated conclusion: `/workspace/BAGM/results/predictive_benchmark/relative_qkv/res_so2_bce_optimization_audit/README.md` and its checksum-bound `result.yaml`.
- Cutoff diagnostic: `/workspace/BAGM/reports/analyses/so2_bce_threshold_sensitivity/v1/report.html` (portable), with Markdown, tables and source hashes.
- Gradient diagnostic: `/workspace/BAGM/reports/runtime/so2_bce_optimization_audit/gradient_confirmation_v1/README.md`.
- Independent protocol receipt: `/workspace/BAGM/reports/runtime/so2_bce_optimization_audit/confirmation_c_verification_v1/final_protocol_runtime_verification_v1.json`.
- Final reporting tests, doctor and resource receipt: `/workspace/BAGM/reports/runtime/so2_bce_optimization_audit/final_verification_v1/`.

The main validation curves and paired endpoint figure were visually reviewed.
The curated result records the early batching benefit and negative fixed-budget
tuning outcome separately; eventual performance remains inconclusive. No
patient-level uncertainty, graph-specific gain, faithfulness, biological null
calibration or causal interpretation follows from this audit. A subsequent
study should distinguish residual fitting limits from lack of graph-specific
gain using a declared longer matched budget and appropriate self/context
baselines; execution optimizations need their own equivalence and timing tests.

Final verification: **69 focused reporting tests passed** in 4.21 seconds.
Doctor passed SQLite integrity and the 5 GiB disk floor (19.934 GiB free), but
returned exit 1 for the same prelaunch historical issues: one unindexed
checkpoint (69 artifacts, 68 indexed) and two missing registered six-core
dataset/split paths. All new audit checkpoints were independently verified and
indexed. The earlier full repository suite remains 1,929 passed, 35 missing-
legacy-input failures and four skipped; it was not redundantly rerun. All four
GPUs were idle with no compute processes after the completed studies.

After that immutable verification receipt was sealed, root separately ran
`manage_results.py validate --verify-payloads --verify-sources` and
`manage_results.py catalog --check`: both passed with four result records.
The catalog content SHA256 is
`6717d534f67805474775c8f236ed7b3bba7d147dc52db3f692eed95f5132ed90`.
`git diff --check` passed in both repository roots, and both final Git status
listings were reviewed with unrelated changes preserved. These publication
checks are recorded here rather than retroactively modifying the sealed receipt.

### Report reproduction

Use the shared-root environment above from the source worktree. Reports read
only verified immutable completed bundles and the registry status of other
attempts. They never read active training histories or choose a setting from an
incomplete matrix. Each command creates a new immutable report directory;
an existing version is rejected rather than overwritten.

After all nine grid candidates and the paired-core reference have resolved:

```bash
AUDIT_REPORT_DIR="$BAGM_REPORT_ROOT/analyses/so2_bce_optimization_audit"
/venv/main/bin/python scripts/analysis/report_so2_bce_optimization_audit.py --stage grid --version grid_v1
```

The grid summary declares its selected setting and whether the single adjacent
LR boundary test is required. Run the three prescribed regularization controls
and any required boundary test before producing the next stage:

```bash
/venv/main/bin/python scripts/analysis/report_so2_bce_optimization_audit.py --stage regularization --parent-report "$AUDIT_REPORT_DIR/grid_v1/summary.json" --version regularization_v1
```

The regularization summary identifies the distinct confirmation comparator.
After both variants finish the 32-epoch profile at model seeds 0 and 1:

```bash
/venv/main/bin/python scripts/analysis/report_so2_bce_optimization_audit.py --stage confirmation --parent-report "$AUDIT_REPORT_DIR/regularization_v1/summary.json" --version confirmation_v1
```

The following are the executed ancillary and curated publication commands.
For reproduction, choose new report versions and a new result ID; successful
published outputs are immutable.

```bash
/venv/main/bin/python scripts/analysis/report_so2_bce_threshold_sensitivity.py --parent-report "$AUDIT_REPORT_DIR/confirmation_v1/summary.json" --version v1
/venv/main/bin/python scripts/analysis/report_so2_bce_gradient_audit.py --stage confirmation --version confirmation_v1 --run-id r_20261001T092859Z_5cb93f7c_s000_f00_a01_0fb40980 --run-id r_20261001T101202Z_15baf65c_s001_f00_a01_6427735a --run-id r_20261001T101202Z_f8197b29_s000_f00_a01_ed00c19e --run-id r_20261001T101202Z_a32d61e8_s001_f00_a01_8332ee26
/venv/main/bin/python scripts/analysis/publish_so2_bce_optimization_audit_result.py --report "$AUDIT_REPORT_DIR/confirmation_v1/summary.json" --result-id res_so2_bce_optimization_audit
/venv/main/bin/python scripts/results/manage_results.py validate --verify-payloads --verify-sources
/venv/main/bin/python scripts/results/manage_results.py catalog --check
BAGM_MIN_FREE_DISK_GB=5 /venv/main/bin/python -m spatial_benchmark doctor
```

Parent reports are bound by summary and manifest hashes. Incomplete reports
cannot become stage parents or curated results. Report output includes
`report.md`, `summary.json`, run/attempt/history/secondary/per-core CSV tables,
standalone validation and endpoint figures, the producer snapshot, and a
checksum manifest. Epoch/update/time curves distinguish eight-epoch early
update-regimen effects from 32-epoch tuning comparisons. Historical 129-epoch
best performance remains a separate unequal-budget reference.

For reporting readiness only, the following dry run allows missing endpoints
and explicitly suppresses scientific selection:

```bash
/venv/main/bin/python scripts/analysis/report_so2_bce_optimization_audit.py --stage grid --version preflight_v2 --allow-incomplete
```

The original `preflight_v1` is preserved. Its empty-stage figures showed
default numerical axes; `preflight_v2` adds explicit pending-state panels and
separate readiness-smoke curves. Neither report supplies a selected candidate.
Use a new unused version name for any later provisional snapshot.

Only after diagnosis may `--accepted-failure-run RUN_ID` count a verified
nonfinite or resource failure as a resolved planned setting. Such a run receives
no ranked score. Resource failure establishes operational infeasibility,
not predictive inferiority; implementation, configuration, provenance and
unresolved infrastructure failures remain blockers. All attempts stay visible.

Focused reporting checks:

```bash
PYTHONPATH=src:. /venv/main/bin/python -m pytest -q tests/unit/infrastructure/test_so2_bce_optimization_report.py
```

### Interim report and execution-cost follow-up

`/workspace/BAGM/reports/analyses/so2_bce_optimization_audit/interim_grid_v1/`
contains the first four verified grid outcomes and is explicitly incomplete.
It is not admissible as a selection parent. Both plotted artifacts were visually
reviewed. The remaining grid jobs subsequently completed under the original
frozen criteria.

Low sampled GPU utilization prompted the separately contracted exploratory
runtime study `cmp_20261001_so2_bce_execution_profile`. Its one-epoch registered
profile ran after GPU0 finished its final grid job; it did not modify
any frozen training source or the optimization comparisons. The profiler's
aggregate timings cannot by themselves establish a validated speedup.

### Allocation refinement recorded before grid selection

The scientific comparisons, selection criteria and epoch budgets remain those
in the frozen task contract. If the grid does not trigger a boundary LR run,
Screen B uses GPUs0/1/2 for the three independent regularization controls, and
GPU3 starts the already-declared incumbent seed0 32-epoch confirmation in
parallel. That incumbent is fixed before any screen outcome. After the three
controls finish, the final comparator is selected from their complete evidence;
its seeds0/1 and incumbent seed1 start on GPUs0/1/2. The ongoing incumbent seed0
continues on GPU3. Report only after all four confirmation slots finish.

If a boundary LR run is triggered, all four GPUs execute Screen B/boundary
controls, then all four confirmation jobs start after selection. This allocation
change uses otherwise idle capacity and changes no model, mask, seed, exposure
budget, selection endpoint or failure policy. Final reports retain actual
per-run device/time/concurrency provenance; timings remain contention-sensitive.
