# SO1 + SO2 training with reserved paper cores

Phase: full, training active. Outcome: pending. User approved implementation on
2026-10-17 after reviewing the proposed cell-balanced split.

## Task contract

Prepare and launch one fresh seed-0, four-layer Relative-QKV model using both
slides and the approved whole-core split. Deliver a registered dataset/split,
training-only Pearson and morphology transforms, validated training and final
test paths, and a managed run. Operational delivery requires verified preparation,
focused tests, a bound GPU preflight, and a healthy training process (or an
explicitly reported managed wait for existing GPU work). Scientific completion
requires the selected checkpoint, final evaluation and verified archive.

Question: can this model predict masked standardized Pearson features in held-out
cores selected for later comparisons with manually segmented cancer regions?
Hypothesis: masked reconstruction improves on the fixed equal-core training mean.
Alternatives: intracellular co-expression, morphology or broad tissue structure
may account for performance without informative graph context. Beating the mean
supports prediction only; a later matched self-only/graph-null comparison is
required for any graph-specific claim. This single-seed experiment is exploratory.

The estimand retains the existing per-cell partial-gene masking recipe: independently
sample 0 through 1,000 hidden genes in every graph cell, ten views per training
core per epoch, and three fixed evaluation views. Inputs use only visible counts
to estimate exposure, frozen training gene frequencies and residual statistics;
full-count residuals are supervised targets only. Preserve the original four-layer,
256-wide, eight-head Relative-QKV model and complete-core graph receptive fields.
Do not introduce a new target-only masking or neighborhood-sampling experiment.

| Role | Cores | Cells | Fraction |
|---|---|---:|---:|
| Training | 1,3,4,6,7,8,9,10,11,12,14,16,17,18,19,20,21,22,24,25,26 | 264491 | 64.88% |
| Validation | 2,5,13,27,28 | 61561 | 15.10% |
| Test | 15,23 | 81607 | 20.02% |

The experimental unit is the model run; the held-out evaluation unit is a core.
Cells are correlated observations, not independent biological replicates. Per the
user's explicit instruction, donor overlap is permitted. This is a core/section
holdout, not new-donor validation. Both test cores belong to SO2. These samples
were examined and used in earlier experiments; this fresh fit provides a new-run
holdout, not a historically untouched confirmatory test. Start from new weights.
Priority cancer-region cores 1,9,21 remain in training, 13 is validation, and 15,23
are test. Manual cancer masks are not model inputs; their separate files have not
yet been located, so ROI agreement figures are a later evaluation deliverable.

Fit gene frequencies, Pearson residual means/sample SDs and morphology preprocessing
on the 21 training cores only. Reuse immutable raw counts and coordinate-only graph
caches with verified alignment. Materialize only train/validation targets before
training; load test data after freezing the validation-selected best checkpoint.
There is one shared normalization, fitted once on the entire new training set and
applied unchanged to all partitions. Refitting is necessary because the previous
training fit included newly held-out cores 15 and 23; validation and test never
receive independently fitted normalization, and no statistics update during training.
No cross-core or cross-slide graph edges. No slide IDs, clinical labels, cell
identifiers, absolute coordinates or full-count library sizes as model covariates.
Use plain standardized quasi-Poisson residuals (phi 1.01, upper clip 10), consistent
with the prior chosen model recipe; no second signed-log dataset is necessary.

Primary selection metric: equal-core validation masked Huber loss (lower is better).
AdamW lr 1e-4, weight decay 1e-5, gradient clipping 1; maximum 300 epochs, minimum
50, patience 25, min_delta 1e-4, strict best checkpoint. An epoch visits all 21
cores exactly once: ten pairs and one singleton, 11 updates, with mathematically
correct distributed view averaging and core-weighted epoch reporting. Preserve
best/latest checkpoints only. Test uses a distinct fixed mask namespace, selected
best weights and frozen training statistics, once after model selection; it does
not affect early stopping, hyperparameters, preflight, or preprocessing fitting.

Controls: equal-core training-mean predictor; hidden-count perturbation invariance;
train-only fit membership; deterministic fixed masks; distributed-versus-serial
singleton weighting; finite outputs and fresh FP32/BF16 numerical preflight.
Missing genes, changed source checksums, cell/key/graph mismatch, test use before
selection, invalid gradients, nonfinite loss, failed preflight, or disk below the
5 GiB safety floor stop launch. A model not beating the mean is a valid negative
predictive result, not grounds to change the split or select on test performance.

Inputs are the 28 verified CosMx core count arrays, protected slide metadata,
original graph caches and their provenance. Historical preprocessing run
`r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410` supplies alignment/source records,
not fitted parameters. Model recipe reference is
`cmp_20261016_so12_pearson_relative_qkv_train_validation`.

Compute: four RTX 3090 GPUs under existing global/per-device locks, CPU preparation
first, no interruption of the active hidden-state analysis. New train/validation
targets are approximately 1.215 GiB; retain raw inputs and old run bundles. Storage
cleanup is unnecessary at preparation time. Active run outputs belong under
`scratch/active_runs`; registered final bundles belong under `artifacts/runs`.

Expected artifacts: split/source/fitted-statistics manifest, resolved config,
registered dataset and run IDs, source/config/data-bound preflight receipt,
append-only training metrics, best/latest checksummed checkpoints, validation and
final test coverage/metrics/predictions, environment and timing provenance.

Verification and exact execution commands will be recorded below once the new
entry points are created. Required checks include focused new and old-protocol
tests and `PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor`.

## Execution

From the project root, prepare on CPU, then generate the registered configuration:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/train/prepare_so12_paper_data.py
PYTHONPATH=src /venv/main/bin/python scripts/train/prepare_so12_paper_split.py
```

The managed service `bagm_so12_paper_split` runs this one-shot controller:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/train/execute_so12_paper_split.py \
  --config experiments/campaigns/cmp_20261017_so12_paper_core_split/configs/seed0.yaml
```

It waits for the global and all four device locks, checks that GPU processes have
finished, runs the source/config/data-bound four-rank preflight, and only then
registers and claims its exact training job. It cannot consume another campaign's
queued job. A second controller cannot enqueue a duplicate run. No automatic retry
changes the protocol. The controller writes status/events and the eventual job ID
under `scratch/active_runs/ctrl_so12_paper_split_seed0`; preflight logs and receipt
are under `scratch/active_runs/pf_so12_paper_split_seed0`.

Initial doctor check found the existing catalog count mismatch (87 artifacts,
86 indexed checkpoints) and default 25-GiB reserve warning with 9.17 GiB free.
SQLite integrity is clean. This campaign explicitly uses the previously adopted
5-GiB disk floor, checked before preparation, preflight and run launch; historical
catalog issues are recorded separately from new protocol checks.

## Preparation and launch record

- Prepared and registered `so12_paper_core_split/v1`, split
  `so12_paper_21train_5val_2test_v1`. New target arrays contain 326,052
  train/validation cells; no test count, target or graph arrays were opened.
- Dataset fingerprint:
  `e976f5fa6efe76be97a31ce14e0c4d457502fe35a66216a8deffefb95b476035`.
  Split fingerprint:
  `cc641f1f9abba5c36b6ec89269c41527696b80a57546a2788ccffd12e6f24154`.
  Frozen fitted-parameter file SHA256:
  `2770f5bbc469ba17ef9c6399b4c4803619d99f15669e4a608573bc68a4a1fafc`.
- Full real-data load verified raw-count/metadata alignment, cached targets and
  graph checksums in 111 seconds. Receipt:
  `scratch/active_runs/ctrl_so12_paper_split_seed0/data_verification.json`.
- Focused verification: 22 data tests; 28 old/new trainer regression tests,
  including real four-process Gloo equivalence for pairs and the uneven singleton;
  132 infrastructure/new and legacy protocol/archive tests; 4 controller tests.
  These groups overlap and are not additive. Controller checks cover exact-job
  claiming, concurrent duplicate rejection, child exit before GPU-lock release,
  and durable launch intent preventing duplicate enqueue after interruption.
- Installed and started supervisor service `bagm_so12_paper_split` at
  2026-10-17 03:59 UTC. Verified status `waiting_for_existing_gpu_owner` while
  `bagm_so12_hidden_leiden` continues uninterrupted. No model training has begun
  and no passing GPU preflight is claimed yet. The controller automatically
  proceeds when locks and GPUs become available; there is no extra approval step.
- Approximately 7.9 GiB remained after preparation. No data, observations or
  historical checkpoints were deleted.

Broader CPU repository regression command (GPUs reserved by the existing job):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=src /venv/main/bin/python -m pytest tests -q -p no:cacheprovider
```

Its complete log is retained at
`scratch/active_runs/ctrl_so12_paper_split_seed0/repository_tests.log`.
Completed in 6m55s: **1,590 passed, 34 failed, 4 skipped**. All new paper-split
and existing SO12 data/trainer/protocol tests passed. The failures are in older
adjacency-ablation, multiscale and MyJjU workflows: absent locked configurations,
receipts or prepared geometry, plus a changed externally audited gradient-source
identity. Those files/workflows were not modified or recreated in this task.
This is not a claim that the whole repository test suite is green. GPU-dependent
tests were skipped because the existing analysis owns the devices; the mandatory
four-GPU preflight remains an automatic prerequisite for this run.

Doctor rerun with `BAGM_MIN_FREE_DISK_GB=5` confirms clean SQLite integrity and
7.825 GiB free. Its sole issue remains the historical checkpoint catalog count
mismatch; the current GPU-lock warning is expected. Full output is retained in
`scratch/active_runs/ctrl_so12_paper_split_seed0/doctor.json`.
The scientific outcome remains pending until the eventual registered training
run and final test evaluation finish.

## Storage audit for training plus subsequent Leiden

Read-only audit on 2026-10-17 found approximately **7.77 GiB free**, with the new
1.215-GiB training/validation target dataset already included in disk usage.
The following allowances are additional to that measured usage:

| Work | Conservative additional disk allowance |
|---|---:|
| Finish the currently running h0/hL Leiden analysis | 0.15 GiB |
| New training, including atomic checkpoint-save peak | 0.50 GiB |
| One subsequent h0/hL Leiden analysis over all 407,659 cells | 1.40 GiB |
| Total | 2.05 GiB |

Projected free space is approximately **5.72 GiB**, or **0.72 GiB above the
5-GiB safety floor**. The current workflow therefore fits within the available
capacity; additional large analyses or duplicate full exports need a new check.
This estimate assumes 256-dimensional FP32 h0 and hL representations, sparse
30-neighbor graphs, two Leiden seeds and the current six-core PNG/PDF/HTML scope.
Future hL-only storage would be smaller (budget 0.8 GiB), but the existing script
currently extracts both representations.

Measured evidence: the previous complete training bundle occupies 122,224,640
allocated bytes, with two 60,322,721-byte checkpoints. Its same-model checkpoint
sizes imply roughly 0.3 GiB during the worst atomic best/latest rewrite; the
0.5-GiB training budget includes logs, provenance and longer histories. Current
Leiden output occupies 1,226,301,440 allocated bytes before its final reports.
Each representation is 417,446,400 bytes; neighbor indices and distances each
occupy 48,919,208 bytes per representation. Neighbor-search temporary tensors
reside in RAM/GPU memory. Both training and Leiden publication use a directory
rename on this filesystem, so they do not require a second full output copy.

No deletion was needed or performed. Contingencies identified, but not applied:
0.871 GiB of idle browser/test temporary files; and 3.037 GiB of byte-identical
failed-preprocessing matrices verified against successful dataset copies. Matrix
deduplication would need a documented retention update preserving paths, checksums
and failure provenance. Raw/clinical inputs, current datasets and active outputs
remain protected. The existing graph aliases already share hardlinks, so removing
their apparent duplicate paths would not reclaim the apparent directory sizes.

Machine-readable measurements and assumptions:
`scratch/active_runs/ctrl_so12_paper_split_seed0/storage_audit.json`.
Recheck free space against the 1.4-GiB Leiden budget before its eventual launch;
the present estimate is not a reservation against other writers.

## Live loss-curve snapshot

The user-requested diagnostic at 2026-10-17 14:02 UTC contains all 177 completed
epochs from run `r_20261017T044837Z_caf76350_s000_f00_a01_12df41be`. It is saved
under `reports/runtime/so12_paper_loss_20261017T140220147075Z/` as PNG, PDF and
CSV, with the exact complete-line metric snapshot, plotting source, checksums
and registry artifact records. This is a transient training diagnostic, not a
completed scientific result or test-set evaluation.

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  /venv/main/bin/python scripts/analysis/plot_so12_paper_training_loss.py \
  --run-id r_20261017T044837Z_caf76350_s000_f00_a01_12df41be
```

Each invocation creates a separate timestamped snapshot. Both curves report
masked-expression Huber loss with equal weight per core. Training uses 21 cores,
fresh masks, dropout and online pre-update losses; validation uses five different
cores, fixed masks and dropout-free post-epoch evaluation. Their absolute levels
are therefore not directly matched. The right panel enlarges the last 100
validation epochs and shows the running minimum without smoothing. At this
snapshot the best validation loss is 0.195405253 at epoch 177; the early-stopping
patience anchor remains epoch 172 because this smaller new gain does not exceed
the configured 0.0001 reset threshold.

Verification: contiguous epoch coverage, finite values, exact agreement with
the recorded best checkpoint, valid PNG/PDF exports and visual layout inspection.
No model, normalization, training configuration, checkpoint or test data changed.
