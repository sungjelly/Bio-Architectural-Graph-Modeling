# Joint-slide four-layer Relative-Geometric QKV training

Phase: complete, final bundle and checkpoints verified. Outcome: supported for
Huber/MAE gain over constants; squared error is worse and graph-specific gain
remains untested. This is an exploratory, single-seed
training/validation experiment requested after the joint-slide preprocessing
comparison. It is not a sealed-test evaluation or a graph-mechanism claim.

## Objective and task contract

Train the original four-layer `relative-qkv-gat` on the integrated SO_1 + SO_2
cohort, retain independent random gene masking for every cell, and select a
checkpoint using donor-separated validation. Deliver a registered, managed run,
resolved configuration, provenance, training/validation curves, restartable
latest checkpoint, validation-selected best checkpoint, and verified final
metrics. The immediate operational deliverable is a healthy supervised training
process; the scientific experiment completes only after finalization.

The hypothesis is that observed expression, permitted morphology and spatial
context predict hidden integrated Pearson features in validation donors. A
credible alternative is that predictions mainly recover per-gene means or
cell-autonomous structure. A frozen training-only per-gene constant supplies a
cheap reference. This single-model experiment cannot establish graph-specific
gain, biological integration, or causal influence without matched controls.

Cells are observations; cores are repeated samples; donors are the independent
biological units. Seed 0 is fixed before training. No seed selection is allowed.

## Inputs and split

- Existing integrated dataset: `data/processed/so12_pearson_integration_v1`.
- Upstream verified preprocessing run:
  `r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410`.
- Plain Pearson features are the reconstruction targets; signed-log Pearson is
  not part of this run. Preserve all 1,000 biological genes and 407,659 mapped
  cells. The existing exclusion of 340 unmapped SO_2 FOV246 cells is unchanged.
- Reuse the existing donor split: training cores 1–10 and 15–24, 301,198 cells
  from ten donor groups; validation cores 11–14 and 25–28, 106,461 cells from
  four other donor groups. Both slides occur in both partitions. No donor or
  spatial graph crosses partitions. No independent test partition is present.
- The held-out cores were examined during preprocessing diagnostics, so their
  reuse is exploratory validation/model selection, not untouched test evidence.
- Reuse coordinate-only per-core radial-stratified k200/r500-um graphs and
  their 70-dimensional relative-geometry caches. No cross-core/slide edges.
- Read source raw-count arrays and original permitted 22 morphology/imaging
  fields. Fit morphology imputation/transforms/scales on pooled training cells
  only, then freeze them. Never supply identifiers, vendor labels, RNA QC,
  cell totals, disease labels, or absolute coordinates as node covariates.

## Masking and normalization contract

For each cell independently, draw an integer masked-gene count uniformly from
0 through 1,000 inclusive, then sample that many positions without replacement.
Use ten independent mask views per training core per epoch. Metadata stays
visible. Training and validation mask namespaces are disjoint; use three fixed
validation views per core, identical at every epoch.

The precomputed Pearson arrays used full-cell totals and must never be masked
and supplied directly as inputs. Instead, recompute each input view from raw
visible counts only. With frozen training gene frequencies p, estimate exposure
as sum(visible counts)/sum(p over visible genes), clamped to at least one.
For fully masked cells use exposure one; all expression outputs are masked
anyway. Set mu=estimated_exposure*p, form (count-mu)/sqrt(1.01*mu), apply the
upstream frozen training residual mean/SD and upper clip of ten, and finally
set masked entries exactly to zero. The explicit gene-mask channel remains.
Changing hidden raw counts must not change any input feature.

The immutable fully observed Pearson arrays are supervised targets only.
This estimand is masked residual-feature reconstruction, not raw-count
likelihood or absolute-count recovery. Full-cell totals are never inputs or
offsets. Train-fitted target transformations may use training outcomes; frozen
validation targets are used only for scoring and model selection.

## Architecture and training

- Original `relative_geometry_qkv_graph_transformer`: four independent graph
  blocks, hidden width 256, eight heads, 1,024-wide FFN/decoder, geometry as
  attention-logit bias, dropout 0.1, no implicit self edges.
- Masked-entry Huber loss with delta one; equal-core training aggregation.
- Four-GPU DDP with the established two-core/ten-view schedule: ranks 0/1
  share the first core's views and ranks 2/3 share the second core's views.
  Average gradients over both cores and all views before one optimizer step.
- AdamW, learning rate 1e-4, weight decay 1e-5, gradient norm clip one.
  Fixed learning rate retains the original optimizer strategy.
- Maximum 300 epochs. Validation early stopping after at least 50 epochs and
  25 consecutive validations without a decrease exceeding 1e-4. The best
  checkpoint tracks every strict improvement, independently of patience.
- Primary metric: equal-core validation masked Huber, lower is better.
  Also retain pooled masked Huber, MAE, MSE and Pearson diagnostics with
  explicit scales and aggregation. Report the fixed constant reference.
- Use BF16 autocast with FP32 loss/attention accumulation only after a bounded
  FP32 equivalence check. Preserve RNGs, optimizer state and stopping state.

## Resource and correctness gates

Observed allocation: all four RTX3090 24-GiB GPUs are idle. Never displace
another process. A representative four-rank pilot must exercise real complete
cores including the largest training/validation cores before production.
Require finite forward/backward/validation values, nonzero finite gradients,
rank agreement, complete mask coverage, successful checkpoint reload,
hidden-count input invariance and donor-split invariance. Record throughput,
peak allocated/reserved VRAM, host usage and estimated runtime.

Target peak allocated VRAM below 21 GiB per device. Preserve at least 5 GiB
free disk; current free space is approximately 8.5 GiB. Reuse large arrays via
read-only references/memmaps, keep only best/latest model checkpoints and
compact aggregate predictions, and estimate output growth before launch.
No deletion of historical inputs or outputs is authorized by this task.

Stop on split/alignment/checksum failure, nonfinite values, hidden-count input
dependence, resource-limit violation, or unsafe GPU occupancy. Preserve and
classify any failed attempt; do not silently change scientific settings.

## Execution and verification

New dedicated protocol: `so12_pearson_relative_qkv_train_validation_v1`.
The existing SO1/SO2 fit-only trainers and their immutable outputs remain
unchanged. New source files implement the dataset adapter and validation
trainer; narrowly scoped configuration/queue hooks select the new entry point.

Use the authoritative registry and one-shot supervisor worker. Active output
belongs in `scratch/active_runs/<run_id>`; verified finalized output belongs in
`artifacts/runs/YYYY/MM/<run_id>`. Register/index retained checkpoints with their
role, epoch, selected metric, checksums and verification status. Keep any
prediction export aggregated and free of direct identifiers.

Before production: run focused data/masking/configuration/runner tests,
representative GPU preflight, configuration validation and project doctor.
Record existing doctor findings separately from new failures. Add exact
commands, launch/run identifiers, pilot results and monitoring locations here
as they become available. Production acceptance requires successful best
checkpoint reload, final validation metrics and immutable bundle verification.

## Execution record

- Materialized and validated `configs/seed0.yaml` with
  `PYTHONPATH=src /venv/main/bin/python scripts/train/prepare_so12_relative_qkv.py`.
  The original integrated dataset and donor split registrations are reused;
  the training adapter has its own explicit preprocessing provenance.
- Data/protocol tests: 27 passed together. Trainer scientific/durability CPU
  tests: seven passed. Configuration/queue regression tests also passed.
- Full raw/target/graph checksum verification passed. Largest training core:
  SO2-C23, 43,462 cells; largest validation core: SO2-C27, 24,245 cells.
- Project doctor: SQLite integrity and configuration checks passed. Existing
  findings are incomplete historical checkpoint indexing and 8.66 GiB free
  disk versus its generic 25-GiB default. This campaign explicitly uses the
  prespecified 5-GiB floor. Expected retained output is below 1 GiB; large
  input matrices and graph caches are referenced, not copied.
- Supervised preflight: `bagm_so12_relative_qkv_preflight`, started after
  confirming all four GPUs were idle. It uses the production configuration
  and writes a source/config/data-bound receipt to
  `state/preflight/so12_relative_qkv_ddp4/preflight.json`.
- Preflight log: `/var/log/portal/bagm_so12_relative_qkv_preflight.log`.
- Production service prepared: `bagm_so12_relative_qkv_worker`; it uses a
  one-shot registered worker, four GPU locks, 30-second heartbeats and the
  5-GiB floor. Its active registered retry is recorded below.
- Early-stopping patience counts from the first validation; stopping requires
  both at least 50 completed epochs and 25 successive validations without a
  decrease larger than 1e-4. Strict best-checkpoint selection is independent.
- Full independent CPU integrity receipt: `state/preflight/so12_data_integrity.json`.
  All 92,667,112 directed edges passed bounds, canonical-order, no-self-loop,
  range and finite-geometry checks. Independent target/input formula replay
  agreed within 1.7e-6; training gene frequencies matched exactly.
- Synthetic run publication and checkpoint indexing passed with 24 aggregate
  validation rows and distinct best/last epochs and metric values. Combined
  protocol/archive/catalog regression suite: 58 passed, one existing skip.
- Recovery note: a new run resumed from an already-terminal latest checkpoint
  must reconstruct `metrics/events.jsonl` from saved history before publication
  if it executes no new training epochs. The normal run and nonterminal resume
  do not take this path. Do not modify source during the active bound run.
- Initial GPU preflight passed: peak allocated 15.50 GiB, peak reserved
  17.88 GiB, output mean absolute BF16/FP32 difference 0.00104, gradient
  relative L2 difference 0.00533 and gradient cosine 0.999986. All ranks
  agreed, hidden-count input invariance and checkpoint reload passed.
- First registered launch, `r_20261016T090140Z_00dadce7_s000_f00_a01_cdb171a3`,
  failed before model execution because the adapter version had been placed
  in the upstream dataset's `preprocessing_version` field. Its failure bundle
  is preserved. No optimizer steps executed and no scientific settings changed.
- Corrected config: `configs/seed0_attempt2.yaml`. It keeps the authoritative
  upstream preprocessing version and separately declares
  `training_adapter_version: observed_only_mask_adapter_v1`. Exact worker
  validation passed against a backup of the live registry. A fresh GPU
  receipt passed under
  `state/preflight/so12_relative_qkv_ddp4_attempt2/`; the original receipt is
  retained. Production is an explicit retry linked to the failed attempt.
- Corrected preflight passed with the same losses and numerical checks.
  Production started at 2026-10-16 09:08 UTC:
  `r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b`, queue job
  `q_25e498d175391f433b93`, explicitly linked to the failed first attempt.
  Registry status is running; supervisor service is
  `bagm_so12_relative_qkv_worker`.
- Active run directory:
  `scratch/active_runs/r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b/`.
  Read `logs/stdout.log` for update/epoch progress and
  `results/epoch_metrics.jsonl` for complete epoch metrics. Active checkpoints
  are `checkpoints/best.ckpt` and `checkpoints/latest.ckpt`; publication retains
  `best.ckpt` and `last.ckpt` with distinct selection metadata.
- First complete epoch verified: ten optimizer updates, all 20 training cores,
  all 24 fixed validation core/view combinations. Equal-core masked Huber:
  training 0.23421114, validation 0.20530324; constant-reference validation
  0.21024251. These are initial-epoch metrics, not final model performance.
  Epoch duration was 351.2 seconds; final runtime depends on early stopping.
- Best/latest checkpoints were read back successfully on CPU, with strict
  model-state loading, optimizer-state loading, four rank RNG states and exact
  source/configuration binding. Receipt:
  `state/preflight/so12_production_start_verification.json`. Epoch 2 is active.

## Exploratory zero/mean comparison contract

Requested during active training on 2026-10-16. Compare a snapshot of the
validation-selected checkpoint with (1) zero standardized-Pearson features,
(2) the existing equal-core training per-gene mean and (3) the cell-weighted
training per-gene mean. All constants are fitted on the 20 training cores only.
The objective is a verified numeric comparison on identical hidden entries;
the alternative is that the model mainly improves robust central-location
prediction rather than recovering useful cell-specific or spatial signal.

Reuse the exact three fixed masks on each of eight validation cores, the
configured seed and immutable target arrays. The primary score is equal-core
masked Huber, lower is better; secondary MAE/MSE/Pearson are pooled over masked
entries. Zero-predictor Pearson correlation is undefined. Preserve all metrics,
including any baseline advantages. Donors are independent units; this small
comparison is exploratory selection evidence, not a test or graph-benefit claim.

Use CPU only (two threads and bounded target chunks) while the four GPUs keep
training. No active trainer/model/data source modifications and no new training
are needed. Register a versioned evaluation associated with the source run;
snapshot its best-checkpoint metrics and checksum before comparing. Verify all
target checksums, complete mask coverage and exact mask-receipt digest, and
require the replayed existing mean baseline to agree with startup metrics to
1e-8. Stop on any mismatch or nonfinite score. Deliver comparison CSV/JSON,
per-core results, fitted constants, provenance/checksums and a completion marker.
Do not promote this transient checkpoint comparison as a final curated result.

Reproduce the versioned comparison (each invocation snapshots the then-current
best checkpoint and creates a new evaluation):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /venv/main/bin/python scripts/analysis/compare_so12_zero_mean.py --run-id r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_so12_zero_mean.py
```

The three focused checks passed: training-only mean fitting and weighting,
hand-calculated zero-prediction losses/undefined correlation, and chunked metric
equivalence to the trainer while ignoring visible targets.

Comparison completed at 2026-10-16 15:54 UTC and is registered as
`ev_so12_zero_mean_e086_20261016T155304Z`. Its immutable report is
`reports/analyses/so12_zero_mean/ev_so12_zero_mean_e086_20261016T155304Z/`.
The checkpoint snapshot is the best at epoch 86, SHA-256
`2c894bf62fc722e29e307b9a513a62d699f2c78d8c528612aa0ebd6f58308a58`.
All predictors use the same 159,816,944 masked entries across 24 views of
106,461 validation cells. The model scores come from that checkpoint's saved
validation record; the constants were independently scored on CPU.

| Predictor | Equal-core Huber | Pooled MAE | Pooled MSE | Pooled Pearson |
| --- | ---: | ---: | ---: | ---: |
| Zero | 0.20935822 | 0.42273230 | 0.99997359 | Undefined |
| Training mean, equal core weights | 0.21024251 | 0.42893796 | 0.99890831 | 0.03227131 |
| Training mean, equal cell weights | 0.20910161 | 0.42092624 | 1.00004250 | 0.00049447 |
| Four-layer model, epoch 86 | 0.19667494 | 0.30887353 | 1.00995025 | 0.12152992 |

The model lowers Huber by 6.06% versus zero, 6.45% versus the equal-core mean,
and 5.94% versus the equal-cell mean; it wins on Huber in all eight cores.
MAE falls by 26.6–28.0%, but MSE is 0.99–1.11% worse. Thus the robust-loss
improvement does not extend to squared error. This is a single-seed, interim,
validation-selected comparison across four validation donors. Preprocessing
diagnostics previously examined this validation set, so it is not an untouched
test. Gains over constants do not establish graph-specific information; a
matched cell-autonomous model would distinguish that explanation.

Independent audit verified all report hashes, the completion marker, registry
status, checkpoint ownership and full binding, exact mask coverage/digest,
saved model metrics, and reaggregation of the baseline sufficient statistics.
The replayed equal-core baseline agrees with the startup reference to 1e-8.
Training remained active at epoch 93 after 92 completed epochs at 15:56 UTC;
these numbers are not final training results. Subsequent reuse of the analysis
script stages each evaluation separately under `scratch/active_runs/<evaluation_id>`
to avoid writing inside the source run during publication or after completion.
Each report retains the exact analysis source used for its own execution.

## Completion notification

The user requested a message in the existing Codex conversation when training
finishes. A separate CPU-only supervisor service polls the read-only registry
every 60 seconds. Successful completion requires both registry `completed` and
the published `_SUCCESS` marker; failures or an unexpectedly stopped worker
also trigger a status report. The monitor uses the installed `codex queue` CLI
to start a reply in the original conversation when that conversation is
idle. It never changes training, its configuration, its source, or run outputs.
Operational receipts live under `state/notifications/`; no scientific result
is created by this monitoring task. The follow-up must verify final artifacts
and distinguish completion, failure, and interruption before notifying the user.
Acceptance requires a successful read-only app-server daemon check, focused
notification-condition checks, and a running supervisor monitor. Delivery needs
the local instance and Codex app server to remain available; transient connection
failures are retried. A deliberately interrupted or unloaded conversation needs
resuming before its queued message runs. Desktop/push delivery is governed by the
user's app settings. An ambiguous dispatch timeout is retained for inspection
instead of risking duplicate messages.

Installed at 2026-10-16 17:30 UTC as `bagm_so12_completion_notifier`, with
wrapper/config retained in `ops/supervisor/bagm-so12-completion-notifier.*`.
The monitor receipt is
`state/notifications/r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b.json`.
Both training and notifier services were verified running; training was in
epoch 112. The daemon check passed against app server 0.163.0, and three focused
marker/status checks passed. Independent review checked marker semantics,
read-only training access, and duplicate prevention after uncertain delivery.
Doctor retained its existing incomplete checkpoint catalog and generic 25-GiB
disk-floor findings (8.526 GiB free); database integrity and configs passed.
This campaign's 5-GiB floor remains satisfied. No full training tests were rerun
for this operational monitor; training code and configuration are unchanged.

```bash
PYTHONPATH=src /venv/main/bin/python scripts/diagnostics/watch_run_notification.py --run-id r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b --thread-id 01a1439e-b190-7740-869f-55e997248425 --worker bagm_so12_relative_qkv_worker --check
PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/infrastructure/test_run_notification.py
supervisorctl status bagm_so12_completion_notifier bagm_so12_relative_qkv_worker
```

## Final training result

Run `r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b` completed successfully
on 2026-10-16 at 20:12 UTC. It ran 151 epochs and 1,510 optimizer updates,
with trainer runtime 11h 00m 55s (about 11h 04m from launch to publication).
Four RTX 3090 GPUs were used; maximum allocated GPU memory was 15.55 GiB.
The final training Huber was 0.21386423. Both the best and last checkpoint
represent epoch 151; their roles are separately retained and indexed.

Early stopping followed the configured 25-validation patience and 1e-4
minimum decrease. Epoch 126 was the last qualifying patience reset.
Subsequent smaller decreases updated the strict best checkpoint, including
epoch 151, but did not reset patience; this is why the last epoch is also best.

| Predictor | Equal-core Huber | Pooled MAE | Pooled MSE | Pooled Pearson |
| --- | ---: | ---: | ---: | ---: |
| Zero | 0.20935822 | 0.42273230 | 0.99997359 | Undefined |
| Training mean, equal core weights | 0.21024251 | 0.42893796 | 0.99890831 | 0.03227131 |
| Training mean, equal cell weights | 0.20910161 | 0.42092624 | 1.00004250 | 0.00049447 |
| Final selected model, epoch 151 | 0.19611390 | 0.30815995 | 1.00588773 | 0.13449855 |

The existing baseline report remains directly comparable: target manifest,
donor split, configuration/source bindings, 24 masks and 159,816,944 masked
entries agree. Huber decreases by 6.33%, 6.72%, and 6.21% versus zero,
equal-core mean, and equal-cell mean, respectively; all eight validation cores
improve. MAE falls 26.8–28.2%, while MSE is 0.58–0.70% worse. The exploratory
validation and graph-specific limitations above still apply.

Final immutable bundle:
`artifacts/runs/2026/10/r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b/`.
Best checkpoint: `checkpoints/best.ckpt`, SHA-256
`6b4526508d223fb8a286084f57fc5f4b36129db6afeffaa328672d2c06226af4`.
All 30 bundle files and the completion marker passed verification. Both
checkpoints loaded strictly on CPU with finite model tensors and matching
run/configuration/data/source bindings. The trainer had already reloaded and
re-evaluated the best checkpoint with identical validation masks before
publication. Both catalog entries are verified. Operational audit receipt:
`state/notifications/r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b.verification.json`.
Training and notifier services exited normally; neither was restarted during
the completion audit.
The final mixed predictive result is indexed at
`results/predictive_benchmark/relative_qkv/res_so12_pearson_random_mask_validation/`;
its schema, payloads, and 17 immutable source checksums passed validation.
