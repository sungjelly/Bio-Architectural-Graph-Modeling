# SO_2 GeneMAE masking comparison

Phase: audited. Outcome: inconclusive under the full verification contract.
The narrower whole-cell reconstruction baseline hypothesis is supported.
Exploratory, transductive reconstruction.

## Task contract

Train the existing dual-path GeneMAE on SO_2 with 10% whole-target-cell
masking, and report reconstruction against alternative masking techniques.
The fixed comparison is whole-cell, individual-gene, and spatial-block
masking, each hiding approximately 10% of expression entries. Whole-cell
and block masks hide every gene in round-half-up(0.1 * core cells) cells;
individual-gene masks hide exactly 100 of 1,000 genes in every cell.
Masks are renewed each epoch. A spatial block consists of the nearest cells
to a uniformly sampled anchor, using measured coordinates only.
The training-mask sequence is shared across seeds within each method;
initialization, dropout and core/tile traversal vary with seed. Reported
seed variability is conditional on that shared masking schedule.

The question is whether neighborhood information supports reconstruction of
fully hidden cells. The primary hypothesis is at least 2% lower equal-core
masked log1p-count Huber loss than the visible-cell per-gene mean on common
whole-cell masks, favoring GeneMAE in at least 10/14 cores and all three seeds.
Alternatives are gene sparsity, regional expression means, local copying,
and intracellular co-expression explaining a partial-gene advantage.
All trained variants are evaluated on the SAME three fixed replicates of
EACH masking task. Ranking models within an evaluation task is meaningful;
comparing losses across tasks describes different task difficulties.

Required deliverables: registered two-epoch resource pilot; nine independent
200-epoch runs (three masking techniques x seeds 0,1,2); checksum-verified
final checkpoints; append-only training metrics; full core/seed/mask tables;
baseline and graph-control comparisons; reconstruction plots; concise report
and curated result record. No seed or checkpoint is selected after scoring.

## Inputs, units, and leakage

Use the checksum-bound `data/processed/so2_14core_relative_qkv_v1/manifest.json`
and its fourteen `cores/SO2-C15.npz` through `SO2-C28.npz` files. Verify all
input file checksums and record the manifest digest in every config/run.
The 246,063 cells exclude the explicitly unmapped SO_2 FOV 246 (340 cells).
The ordered 1,000 biological probes exclude technical controls; combined
probe symbols remain intact. No clinical labels or donor identifiers are used.
HGD/Normal replication is not re-estimated; no diagnosis-level claim is made.

Use `log1p(raw biological counts)` without cell library normalization or
fitted gene standardization. Huber delta 1 is a robust transformed-expression
regression objective, not a count likelihood. Only visible expression and
the mask token are node inputs; coordinates construct the graph. Morphology,
RNA totals, vendor cell types, disease labels, identifiers and target-derived
labels are prohibited. The model internally replaces ALL hidden entries.
Test prediction invariance to changing hidden targets. Graphs and spatial
tiles depend only on coordinates. Baselines estimate means from visible
entries separately for each evaluation mask.

All mapped cells belong to the fitting cohort; there is no patient-held-out
or validation split. Fixed evaluation masks differ from training masks, but
their target cells are observed during other training epochs. Cores are the
descriptive observational units; three seeds are technical replicates, not
independent patients. Maximum claim: fitted-cohort masked reconstruction.
No predictive benchmark gate, independent patient replication, interaction,
mechanism, or causal claim can pass from this experiment.

## Model and controls

Reuse the 6,888,016-parameter GeneMAE: four GATv2 layers, width 256,
six heads, graph embedding 192, self branch 512/512/192, MLP decoder,
dropout and DropEdge 0.2. Spatial symmetric-union k=15 graphs use the
Gaussian distance feature within deterministic median-split tiles of at
most 7,000 cells. No edges cross cores or tiles. No expression enters graph
construction. There are no tissue-barrier annotations or maximum edge-length
filter; this graph is computational context, not evidence of direct contact.

AdamW lr 0.0015, weight decay 0.0001, cosine schedule, clip norm 5,
FP32, 200 epochs, one tile per step, all tiles once per epoch. Loss is
computed only on masked entries. A block-masked tile with zero target
entries is visited but skips optimization, recorded explicitly. No early
stopping or tuning; only the final checkpoint is interpreted. Four idle
RTX3090 GPUs host independent jobs; pilot establishes throughput and VRAM.

Report Huber (primary, lower better), MSE, MAE, R2, pooled/gene/cell Pearson,
positive- and zero-entry errors, and count of masked entries. Undefined
correlations remain null. Summaries average replicates within core and then
cores equally; show all seeds and core heterogeneity. No calibrated
interval/NLL is claimed for this point-estimate model.

Controls: all zero, visible per-gene core mean, and visible-neighbor mean
spatial smoothing. Frozen-model graph controls remove all non-self edges or
permute node labels within each tile with one fixed permutation. These are
model-use diagnostics; permutation preserves topology and the degree
multiset but does not preserve cell-type, compartment or distance confounders.
They are insufficient as biological null calibration. The neighbor-mean
baseline recovers a synthetic smooth spatial signal and degrades under
expression randomization. A separate tiny GeneMAE learns a constant gene
profile, checking optimization rather than spatial interaction recovery.
Hidden-target perturbations test leakage. These are implementation checks;
they do not establish recovery of a planted biological interaction by GeneMAE.

## Gates and completion

Before full runs: focused masking/graph/leakage/recovery/queue tests pass;
pilot covers all cores/tiles twice; loss/gradients/parameters are finite;
checkpoint replay passes; peak VRAM <=21 GiB; host RAM <=40 GiB;
projected per-run 200-epoch training <=6 hours; disk free >=8 GiB after
estimated outputs (budget 3 GiB for ten bundles and reports).
Stop on input mismatch, nonfinite values, OOM, unsafe device contention,
missing masking coverage, failed artifact verification or disk gate.
Failures remain visible; no automatic scientific changes or blind retries.

Training completes with all nine successful, verified final bundles and
cataloged checkpoints, matched evaluations, report, tests and doctor audit.
The hypothesis may fail without invalidating a correctly completed run.
Only report graph sensitivity separately, and do not infer spatial biological
importance even if its reconstruction diagnostic improves. Next discriminating
test is patient-held-out training/evaluation with matched self/morphology and
regional baselines, segmentation sensitivity and confounder-preserving nulls.

## Commands (repository root)

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_so2_genemae_masking.py tests/unit/infrastructure/test_configuration.py tests/unit/infrastructure/test_queue_command.py
PYTHONPATH=src /venv/main/bin/python scripts/train/prepare_so2_genemae_masking.py
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark enqueue-experiment --campaign-id cmp_20261005_so2_genemae_masking --config experiments/campaigns/cmp_20261005_so2_genemae_masking/configs/pilot.yaml --gpu 0 --max-attempts 1
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark worker --gpu 0 --once --min-free-gb 8 --parallel-gpu-workers
# Full configs: configs/{whole_cell,partial_gene,spatial_block}_s{0,1,2}.yaml.
# Enqueue only after binding the verified pilot bundle with prepare --pilot-run.
# Run one worker per allocated GPU with --parallel-gpu-workers.
PYTHONPATH=src /venv/main/bin/python scripts/train/execute_so2_genemae_masking.py --pilot-run artifacts/runs/2026/10/r_20261005T070204Z_bd6488a7_s000_f00_a01_a5ca1063
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_genemae_masking.py
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

The matrix launcher executes the report command automatically after all nine
runs finalize. Do not rerun it into the existing report directory.
Allocation is seed/method order `(whole_cell, partial_gene, spatial_block)`
for seed 0, then seed 1, then seed 2, claimed by the next free GPU among
0,1,2,3. Each GPU runs one job at a time with no automatic retry. Dynamic
assignment avoids idle devices when block masks require fewer training steps;
actual device assignments are recorded in each run's hardware provenance.
After the first four jobs began, measured epoch times were about 13 seconds
for partial-gene, 7.5 seconds for whole-cell and 4.5 seconds for block masks.
Remaining queued jobs were given priorities partial-gene=2, whole-cell=1,
block=0 to reduce the final idle-device tail. This changes only independent
job ordering, not scientific configurations, seeds or budgets; the receipt is
`reports/analyses/so2_genemae_masking/verification/scheduling.json`.

Current findings: pilot `r_20261005T064812Z_79664b14_s000_f00_a01_e8cf9d38`
completed two training epochs (equal-core training Huber 0.052778 then
0.046227; 49 tiles and 24,609,000 masked entries each epoch); evaluation
finished all fourteen cores and exact checkpoint replay, but the run failed
while serializing its resource summary: a NumPy boolean was rejected by the
strict canonical JSON writer. The failed bundle and checkpoint remain intact.
The corrected implementation converts gate outcomes to Python booleans and
adds a direct serialization regression test. Config
`configs/pilot_serialization_fix.yaml` is a new registered attempt; the original
pilot config is preserved. These are diagnostic figures, not the final result.
The focused configuration/queue/masking suite passed 49 tests; the additional
archive/masking suite passed 33 tests, then 34 after the serialization fix.
The initial broad suite was interrupted
to isolate possible GPU contention (30 failures, 678 passes, one skip); CPU-isolated rerun
finished with 1,434 passed, 34 failed and four skipped. All 34 failures concern
missing historical adjacency/GeneMAE/geometry artifacts or an older external
gradient-audit source identity, outside this campaign. The pilot emitted a
CUDA allocator pressure warning and continued without a failed training step.
The same cache-allocation warning recurs in the revised pilot while it is the
only GPU process, so concurrent tests were not established as its cause. There
is no raised `torch.OutOfMemoryError`; peak allocated and reserved VRAM will
be recorded separately and checked against the pilot gate. GPU access for the
broad test suite was disabled to avoid possible contention regardless.

Historical ten-core inputs are absent; this campaign uses the available SO_2
cohort and a new protocol, preserving the historical 50% partial-gene
experiment unchanged. Initial doctor reports a pre-existing missing checkpoint
catalog entry and free disk below its generic 25-GiB threshold. This campaign
prespecifies an 8-GiB floor plus a 3-GiB output reserve; no data/history is deleted.

The revised pilot `r_20261005T070204Z_bd6488a7_s000_f00_a01_a5ca1063`
completed successfully, including exact CPU checkpoint replay and hidden-target
invariance. All resource gates passed: peak allocated 8.848 GiB, peak reserved
23.016 GiB (allocator cache), host 3.119 GiB, full evaluation 624 seconds,
projected 200-epoch total 0.618 hours. Allocator cache-reclaim warnings did not
raise an out-of-memory exception or invalidate any training step. The original
failed pilot checkpoint is checksum-verified and cataloged as diagnostic-only.

All nine full configurations are registered. The first four runs started at
07:14 UTC on 2026-10-05. All nine models completed 200 epochs and all fixed-mask
evaluations; seven original runs passed every execution check. Two individual-
gene runs and their recovery audits retain failed verification status below.
Verification logs are under `reports/analyses/so2_genemae_masking/verification/`.

## Checkpoint recovery audit

Production partial-gene seed 1, run
`r_20261005T074042Z_b35cac84_s001_f00_a01_95e91d2d`, completed all 200
epochs and all 588 evaluation rows, then failed the combined exact CPU
checkpoint-replay/hidden-target-invariance assertion. The failed bundle remains
immutable. Its saved weights pass twenty fresh-process CPU comparisons across
one and four threads, and a GPU-to-CPU round trip also passes exact replay,
hidden-target invariance and parameter equality. The original assertion did
not record the two errors separately, so its cause is not established.

Before recovery, fix these acceptance checks: preserve the original weights,
scientific configuration, training history and masks; use no further optimizer
steps; require exact CPU replay and hidden-target invariance for all three
masking methods; reevaluate every core/task/replicate and require all recorded
metrics to agree with the failed attempt at rtol=1e-5, atol=1e-7 (allowing the
declared CUDA scatter nondeterminism), with exactly matching mask receipts and
counts. Final checkpoint parameters must match the original checkpoint exactly.
Use a separately registered recovery attempt and preserve source checksums, failed
attempt identity and its unavailable peak-memory summary. This changes neither
the model nor the scientific acceptance criteria. A failed recovery is retained
and investigated, not retried blindly.

Partial-gene seed 2 (`r_20261005T075208Z_b35cac84_s002_f00_a01_e5347d9b`)
later failed the same assertion after finishing training/evaluation. A fresh
GPU-to-CPU probe reproduced a 4.7684e-7 maximum output difference despite exact
parameter equality and exact hidden-target invariance. Twenty additional probes
(ten each with one and four CPU threads) were bit-exact, so thread count alone
does not establish the cause. Both seeds undergo the same complete recovery
audit; no reconstruction threshold is changed.

The first seed-1 recovery launch, attempt 2
(`r_20261005T084509Z_b35cac84_s001_f00_a02_a66df110`), was rejected before
execution because `runtime` is not an allowed root configuration field. Its
bundle is retained. Recovery provenance now uses `metadata.runtime`, which the
schema accepts and scientific identifiers explicitly exclude. The focused test
now validates this configuration as well as its queue command and unchanged
scientific identity. All 52 focused tests pass. The broad CPU rerun reports
1,437 passed, the same 34 historical-artifact/audit-identity failures, and four
skipped; `verification/recovery_full_suite_cpu.log` preserves the details.

Recovery configuration commands, from the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/train/prepare_so2_genemae_recovery.py --source-run artifacts/runs/2026/10/r_20261005T074042Z_b35cac84_s001_f00_a01_95e91d2d --attempt 3
PYTHONPATH=src /venv/main/bin/python scripts/train/prepare_so2_genemae_recovery.py --source-run artifacts/runs/2026/10/r_20261005T075208Z_b35cac84_s002_f00_a01_e5347d9b --attempt 2
```

The configs are `configs/partial_gene_s1_recovery_a3.yaml` and
`configs/partial_gene_s2_recovery_a2.yaml`. They were enqueued through
`Registry.enqueue` with explicit `attempt_count=maximum_attempts` of 3 and 2,
respectively, and `retry_of` links to jobs `q_e15cacc027a1c14f7827` and
`q_b0d2781baadd1e76c37d`. Workers use the same command as the full matrix, with
`--gpu 1 --worker-id so2-genemae-recovery-gpu1-a3` and
`--gpu 2 --worker-id so2-genemae-recovery-gpu2-a2`, respectively. Each resolved
command and complete source provenance is retained in its new bundle.
The original matrix controller stops with the recorded failures. Its planned
fully verified report gate cannot pass. The audited report below explicitly
includes the failed original attempts as descriptive evidence and preserves
the failed gate, rather than claiming nine successful executions.

## Audited outcome and stopping decision

Both full recovery evaluations failed the prespecified agreement gate solely
for `gene_pearson_median`: 43 values for seed 1 (maximum difference
6.2371e-7), and 33 for seed 2 (maximum 7.3614e-7), out of 11,172 numeric
values per audit. Huber, MSE and MAE all agreed within the original tolerance,
with maximum absolute differences of 2.0433e-10 and 1.6190e-10. Masks and all
counts matched exactly. Fresh-process CPU checkpoint and hidden-target checks
passed before each replay. The exact cause of the intermittent original CPU
assertion is not established; a seed-2 probe reproduced a 4.7684e-7 output
difference with identical weights, while subsequent probes were bit-exact.

The full nine-run verification gate is **not met**. Its threshold is unchanged,
and these failures remain failures. Stop this recovery branch with an
inconclusive verification outcome; no further retries are queued. The requested
descriptive report includes every original 200-epoch model, clearly flags the
two failed runs, and shows the independent numerical audit. It does not claim
that all nine runs passed or that the matrix is fully validated. The next
numerical test would isolate CPU/GPU kernels and retain per-entry predictions
to explain intermittent replay and median-correlation sensitivity; a future
biological benchmark still requires patient-held-out splits and stronger controls.

On the common 10% whole-cell masks (equal-core means across all three seeds):

| Training mask | Huber | MSE | MAE | R² |
|---|---:|---:|---:|---:|
| Whole cell | 0.038705 | 0.079474 | 0.148733 | 0.2036 |
| Individual genes (two failed verification runs included descriptively) | 0.049533 | 0.108489 | 0.113272 | -0.0595 |
| Spatial block | 0.043528 | 0.092695 | 0.166251 | 0.0841 |

Whole-cell training improves Huber by 7.93% over the visible-gene-mean baseline,
favoring it in all 14 cores and all three seeds. Its Huber is 21.86% lower than
individual-gene training and 11.08% lower than spatial-block training on this
same task. The zero predictor has lower MAE (0.099643) but worse Huber/MSE;
positive-expression errors and other metrics remain visible. Spatial-block
models do not beat the visible-gene-mean baseline on their own block task.
No patient-generalization, interaction or causal conclusion follows.

The immutable analysis version is
[`reports/analyses/so2_genemae_masking/v1/README.md`](../../../reports/analyses/so2_genemae_masking/v1/README.md),
with all tables, four figures, failure receipts and a checksum manifest. The
curated record is
[`res_so2_genemae_ten_percent_masking`](../../../results/descriptive_analysis/genemae/res_so2_genemae_ten_percent_masking/README.md).
Its verified record status means schema/source integrity; its scientific
outcome remains inconclusive under the full execution contract.

Final validation: this result's sources and all 13 report payloads verify;
all result records pass schema validation and catalogs preserve the five
pre-existing records. Global source verification separately finds an unrelated
missing `cmp_20261004_local_gene_residual/frozen_task_contract.md`; it is not
reconstructed or silently ignored. Final doctor has no running/queued jobs.
It retains the pre-existing checkpoint-catalog completeness issue (85 registered
checkpoint artifacts versus 84 indexed) and, under its default setting, the
25-GiB disk warning (18.135 GiB free). With this campaign's documented 8-GiB
floor, only the pre-existing catalog issue remains. Full details are saved in
`verification/doctor_final_*.json` and `verification/final_audit.json`.
