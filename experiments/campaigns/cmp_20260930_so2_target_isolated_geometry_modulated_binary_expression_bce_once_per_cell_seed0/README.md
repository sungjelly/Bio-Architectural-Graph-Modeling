# SO2 target-isolated binary-expression BCE, once per cell

Campaign ID:
`cmp_20260930_so2_target_isolated_geometry_modulated_binary_expression_bce_once_per_cell_seed0`

Phase: complete. Outcome: supported for the frozen primary optimization
criterion. Training and artifact verification completed on 2026-09-30 at
19:07:56 UTC. Binary-only training did not improve detection BCE, Brier score,
precision, recall, or F1 over the prespecified hurdle endpoint; the comparison
remains exploratory and does not establish generalization or graph-specific gain.

## Experiment

This run asks only whether a masked target-cell gene is expressed (`raw count
> 0`). It keeps the completed hurdle-NB run's SO2-C15--C26 training set,
SO2-C27--C28 validation set, training-only preprocessing, graph encoder,
target-isolated routing, masks, optimizer, scheduler, early stopping, and model
seed. There is no test split.

The model has four independent 256-dimensional geometry-modulated QKV graph
blocks with eight heads, followed by a 1,024-dimensional decoder trunk and one
1,000-gene logit projection. It has exactly 5,134,088 trainable parameters.
There is no count-mean head and no inverse-dispersion parameter.

Every training cell is a target exactly once per epoch. Its mask size is drawn
uniformly from 1 through 1,000 genes, positions are sampled without
replacement, and only the designated target is masked. Neighbor expression is
fully observed. Validation uses one fixed matched mask per cell.

The primary loss and selection metric are the natural-frequency, unweighted
binary cross-entropy across all masked target entries. No positive class
weight, zero/nonzero balancing, focal term, count likelihood, or positive-count
regression is used. The fixed 0.5 threshold is not tuned on validation.

## Interpretation of initialization and gradients

The final detection weights start at zero and each bias starts at the logit of
the training-only Jeffreys-smoothed per-gene prevalence. Therefore the first
optimizer update has zero decoder-trunk and graph-encoder gradients by
construction: the zero projection weights block upstream gradient flow. This
is expected, not a failed preflight. The detection projection learns on that
update; subsequent updates must demonstrate finite, nonzero trunk and graph
gradients. Gradient direction is recorded each epoch for the full model, each
graph block, and the detection head, but gradient vectors are never retained.

## Validation and storage

Validation BCE controls ReduceLROnPlateau, early stopping, and exact-best
checkpoint selection. Each epoch also records Brier score, observed/predicted
positive rates, fixed-threshold confusion metrics, balanced accuracy, MCC, and
deterministic 4,096-bin histogram approximations of AUROC, AUPRC, and ECE. The
active run uses atomic latest plus best checkpoints; after verified best reload
and replay, latest is deleted. Only the best checkpoint, scalar histories, and
compact curve histograms remain. Epoch checkpoint archives, full prediction
matrices, and raw gradients are prohibited.

This is a single-seed validation-selected experiment. It can show whether
binary-only supervision is promising under this split, but it is not an
unbiased test estimate or a generalization or causal claim.

## Launch gate

From the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/diagnostics/preflight_so2_target_isolated_binary_expression_bce_ddp.py \
  --config configs/experiment/so2_target_isolated_geometry_modulated_binary_expression_bce_once_per_cell_seed0.yaml

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 enqueue-experiment \
  --campaign-id cmp_20260930_so2_target_isolated_geometry_modulated_binary_expression_bce_once_per_cell_seed0 \
  --config configs/experiment/so2_target_isolated_geometry_modulated_binary_expression_bce_once_per_cell_seed0.yaml \
  --priority 0 --max-attempts 1 --gpu 0,1,2,3
```

The authoritative design is
[`frozen_task_contract.yaml`](frozen_task_contract.yaml).

## Interrupted-run recovery, 2026-09-30

Operational phase: recovery verified; the same execution subsequently completed.
At the recovery inspection, the registered run
`r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863` stopped after epoch 17
(last metric 09:40:20 UTC); the observed host boot time was 09:43:01 UTC and
container startup was 10:00:19 UTC. Its recorded worker and launcher were absent.
The queue's `running` status was stale. These observations support interruption
by an environment restart; the initiating cause is not recorded.

Recovery contract: continue this same execution at epoch 18 from its atomic
`checkpoints/latest.ckpt`, preserving the run ID, fixed configuration, model,
split, masks, optimizer, scheduler, scaler, four-rank RNG states, and separate
checkpoint-selection and stopping counters. The alternative is an inconsistent
or incomplete checkpoint transaction; that would reject continuation and
require a separately registered fresh attempt retaining the interrupted run.
This operational recovery does not change the scientific estimand, units,
metrics, baselines, leakage restrictions, or claim limits in the frozen task
contract. Existing validation remains model-selection evidence only.

Acceptance requires matching configuration, source, preflight, data, and split
hashes; successful latest/best checkpoint loading and transaction checks;
complete scalar histories through epoch 17; absent old processes; adequate
disk space; and four available GPUs. After launch, require a renewed registry
heartbeat, four live ranks, epoch 18 and subsequent finite losses, no duplicated
epoch history, and continuing checkpoint updates. A mismatch, nonfinite loss,
missing rank, failed coverage check, or finalization failure stops this recovery
for diagnosis. Do not overwrite the original preflight receipt: the checkpoint
is bound to its checksum.

The four RTX 3090 devices retain the measured allocation (one rank per GPU,
about 5.1 GiB peak per rank) and the original 20-GiB disk floor. The expected
terminal artifact is the same immutable run bundle with the selected best
checkpoint, verified replay, indexed checksum, and `_SUCCESS`. Gradient
direction across the interruption is explicitly unavailable because gradient
vectors are not persisted; optimization state is restored independently.

The standard queue worker always creates a new execution. The administrative
recovery launcher instead attaches this active run, holds the normal project
lock, checks old PID/start-time records, maintains the existing job heartbeat,
and uses the normal archive and registry finalization APIs. Its script, launch
record, checksums, diagnostic results, and recovery events are retained under
the run's `provenance/` directory. Original logs are appended and original
configuration/provenance are preserved.

Verification from this source checkout:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_so2_target_isolated_nb_runner.py \
  tests/unit/spatial_benchmark/test_so2_target_isolated_binary_expression_bce_runner.py \
  tests/integration/spatial_benchmark/test_so2_target_isolated_binary_expression_bce_config_queue.py

# The recovery script supplies the shared-runtime BAGM_* overrides.
# Validation applies to the original stopped epoch-17 state, before launch.
bash /workspace/BAGM/scratch/active_runs/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/provenance/resume_after_restart_20260930.sh --validate-only

# Actual supervised launch (already issued once at 17:13 UTC):
supervisorctl start bagm_so2_target_isolated_bce_recovery_20260930

# Passive monitoring:
supervisorctl status bagm_so2_target_isolated_bce_recovery_20260930
tail -n 5 /workspace/BAGM/scratch/active_runs/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/metrics/events.jsonl
```

The launch command was issued once after 41 checkpoint/audit checks, 37 focused
trainer/configuration tests, and 61 archive/queue/checkpoint tests passed.
Supervisor owns the process independently of the client connection. Registry
heartbeat resumed at 17:13:13 UTC and four ranks were observed loading data.
The original worker service remains stopped. This dedicated recovery service
has no automatic restart policy; another interruption requires diagnosis.

`doctor` reports database integrity `ok`, with pre-existing unrelated missing
cancer-cohort paths and one unindexed failed recurrent-run checkpoint. Its
generic 25-GiB disk floor is above the 21.8 GiB available; this campaign's
frozen 20-GiB floor passes. The obsolete child PID record is reconciled by the
recovery launcher. Complete checks, checkpoint/code/config hashes, the registry
backup reference, launch code, and receipts are preserved in run provenance.
Recovery acceptance passed: epoch 18 finished at 17:18:51 UTC with training BCE
0.3073012192 and validation BCE 0.2454808887; epoch 19 finished at 17:19:48 UTC
with training BCE 0.3073055972 and validation BCE 0.2446667758. The epoch-18
checkpoint records 108 optimizer updates, scheduler epoch 18, unchanged
learning rate 0.0001, and four saved RNG states. All five CSV histories and the
event log preserve their pre-interruption prefixes byte for byte, with no
duplicate or missing epochs. The live heartbeat and four GPU ranks were
verified; peak allocated VRAM remains about 5.1 GiB. Evidence is in
`provenance/resumed_progress_verification_20260930.json`.

The recovery restored training; final completion and selected-checkpoint replay
are documented below. Registry duration combines observed pre-interruption epoch
durations and resumed-process runtime, with downtime reported separately;
pre-interruption setup and partial-epoch time are not fully recoverable. The
historical scratch paths in the recovery commands above moved into the immutable
bundle on publication and must not be used to relaunch this completed run.

## Completion monitoring and results report

The user requested continued inspection through the terminal state on
2026-09-30. The deliverables are a verified completion/failure notification in
the conversation and a concise local results report tied to the immutable run.
Operational monitoring checks advancing epochs, finite losses, checkpoint
updates, worker heartbeat, processes, GPU resources, and the unchanged stopping
rule. Training may stop before or after the earlier 180-epoch time estimate;
that estimate is not a new training limit.

Success requires registry and queue completion, `_SUCCESS`, valid bundle hashes,
verified selected-checkpoint reload, checkpoint catalog indexing, complete
epoch coverage, and preserved recovery provenance. Failure or stagnation is
diagnosed from live processes and timestamps rather than inferred from a
registry label. Monitoring must not alter model parameters, masks, validation,
or stopping thresholds. Operational observations belong under `state/logs/`.

The report will include the selected and final epochs, BCE and classification
metrics, zero-class imbalance, learning and resource history, original epoch-17
interruption, controls, negative findings, and the single-seed/single-validation-
donor limitations. Where recorded metrics and fixed-mask receipts match, it may
compare binary detection with the prespecified hurdle-NB run
`r_20260928T021117Z_9ea46874_s000_f00_a01_41b54097`; total hurdle likelihood and
binary BCE are different objectives and cannot be ranked as the same metric.
Any report designed after inspecting these runs is explicitly exploratory.
No additional model tuning or held-out test is authorized by this reporting
step, and no graph-specific, independent replication, or causal claim follows
from validation performance alone.

## Verified completion and results

Run `r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863` stopped normally at
epoch **129** after 25 validations without a greater-than-0.0001 improvement
in the patience reference. The independently maintained strict minimum selects
epoch **123**, with replayed validation BCE **0.2401795339072312**. BCE improved
by **0.03347819155 (12.23%)** from epoch 1, passing the frozen optimization
criterion with finite losses and required gradients. Epoch 1 already includes
six optimizer updates and is not an untrained baseline.

| Saved validation metric | Binary-only BCE | Hurdle comparator |
| --- | ---: | ---: |
| Detection BCE (lower better) | 0.240180 | 0.239984 |
| Brier score (lower better) | 0.065949 | 0.065885 |
| Precision at 0.5 | 68.68% | 68.82% |
| Recall at 0.5 | 7.91% | 8.04% |
| F1 at 0.5 | 0.141795 | 0.144029 |
| Specificity at 0.5 | 99.6815% | 99.6781% |

Binary AUROC is 0.770606, approximate average precision/AUPRC is 0.298425,
and ECE is 0.003369, all from the retained 4,096-bin histogram. Matching hurdle
AUROC/AUPRC/ECE are unavailable. Accuracy is 92.23%, versus 91.88% for predicting
every entry negative: only 8.12% of masked entries are positive, and the fixed
threshold misses most observed positives. Specificity improves slightly while
the other listed matched detection measures worsen. No threshold was tuned.
The endpoints have different prespecified selection objectives (binary BCE
versus joint hurdle likelihood) and resulting training exposure: hurdle selects
epoch 164 and stops at 171, versus binary epoch 123 and stopping at 129. This is
a comparison of the protocol's selected endpoints, not a matched-update ablation;
no model-family superiority follows.

All epochs contain every one of the 208,696 training cells exactly once as a
target, with a fresh mask; all 37,367 validation targets retain their fixed
masks. Histories are contiguous, and primary and secondary selected-checkpoint
metrics replay within 1e-6, with exact confusion counts. The same seed-0 execution
was resumed after epoch 17; it is neither a new seed nor a failed excluded run.
There are six training donor groups and only one validation donor group (two
cores), with no sealed test. Same-cell expression, morphology, gene prevalence,
segmentation spillover, and count-supervision regularization remain alternatives.
Graphless/self/context baselines, realistic nulls, repeated seeds/folds, patient
replication and explanation faithfulness remain untested. The next discriminating
test is a locked train-fitted prevalence and self/context comparison across
independent donor-held-out folds.

The immutable bundle is
`/workspace/BAGM/artifacts/runs/2026/09/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/`.
Both registry and queue are completed, `_SUCCESS` is present, all 39 bundled
files verify, and artifact **775** is the sole retained checkpoint, indexed as
verified `best`, epoch 123, with SHA-256
`2320932a771ab670ddb2d90420adffdfba09a4818d673936b2431c7e039e5316`.
The preferred semantic alias is
`run.exploratory-screen.so2-donor-grouped-ta.target-isolated.k200-r500-union.target-only.disabled.d256.s000.f00.a01.vf95a63d2f4e3.x7390515b8bce`.
The supervisor recovery process exited normally and all four GPUs were released.

Completed epoch durations total 2.04 hours across both segments; the resumed
training process took 1.91 hours. Registry execution duration is 2.19 hours under
the documented recovery policy. Original-launch-to-completion wall time was
9.81 hours, including the interruption. Four RTX 3090 GPUs used one DDP rank
each; peak allocated VRAM was 5.08 GiB per rank. Full framework/CUDA provenance
and recovery receipts are retained. The unrelated pre-existing `doctor` gaps
described above do not change the successful verification of this bundle.

The portable [HTML report](/workspace/BAGM/reports/analyses/so2_binary_expression_bce/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/v1/report.html)
and [Markdown report](/workspace/BAGM/reports/analyses/so2_binary_expression_bce/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/v1/report.md)
include figures, complete matched metrics, controls, limitations and checksummed
source records. The report preserves its generating code and environment.
The separate [completion receipt](/workspace/BAGM/reports/analyses/so2_binary_expression_bce/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/completion_verification_v1.json)
records registry/catalog verification and its exact verifier source.
The validated [curated comparison](/workspace/BAGM/results/descriptive_analysis/relative_qkv/res_so2_binary_occurrence_hurdle_comparison/README.md)
records the negative result for improvement in the five named shared detection
metrics, retains the small specificity advantage and unavailable AUPRC comparison,
and keeps all seven evidence dimensions separate. The result catalogs are current.

Reproduction and checks from `/workspace/BAGM` (the reporting checkout):

```bash
PYTHONPATH=src /venv/main/bin/python state/monitoring/verify_bce_completion.py --check-only
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_binary_expression_bce.py \
  --run-id r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863 --version v2
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_report_so2_binary_expression_bce.py
```

The completed `v1` report is preserved; use a new unused report version to
reproduce it. The focused report suite passed all 24 tests. The frozen design
and training artifacts remain unchanged by this report.
