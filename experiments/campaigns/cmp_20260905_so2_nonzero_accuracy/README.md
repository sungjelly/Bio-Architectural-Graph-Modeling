# SO2 nonzero reconstruction accuracy

Phase: complete. Outcome: negative for the proposed geometry nonzero-MSE
advantage at these frozen endpoints. Exploratory post-hoc evaluation requested after
overall and epoch-level losses were inspected. Existing model checkpoints are
frozen; there is no training, tuning, or new checkpoint selection.

## Task contract

Objective: compare the four SO2 final models on masked entries whose **observed
raw count is greater than zero**, while showing the corresponding zero-entry
errors. Deliver complete per-core and equal-core tables, count-stratified
metrics, a figure, checksummed evaluation bundles and a comparison report.

Question: do similar aggregate errors conceal different ability to reconstruct
nonzero expression? The working hypothesis is that geometry modulation reduces
nonzero error despite its mixed aggregate results. Alternatives are a tradeoff
between zero/nonzero errors, shrinkage of predictions, or training duration.
The comparison will show signed paired differences on every core, zero and
positive strata, and count bins 1, 2, 3, 4–7, and 8+. A consistent positive-only
gain with unchanged/worse zero error would support that tradeoff explanation;
no gain would fail to support the proposed nonzero advantage. No comparative
effect threshold or formal significance claim is imposed after seeing results.

The primary descriptive metric is equal-core mean positive-only MSE on the
shared gene-standardized log1p scale; lower is better. Required secondary
metrics are positive/zero/all MSE, MAE, Huber and signed error on that scale,
MSE and MAE on the inverse-standardized log1p scale, and fixed decoded count
accuracy. All inclusion masks are defined from raw observed counts, never
the sign of standardized targets or predictions. The existing 14 core graphs,
246,063 cells, ordered 1,000 biological probes, and one fixed evaluation mask
per core are reused unchanged across all four models. The seed and checksum
must equal the source run's held-in diagnostic exactly.

Count accuracy is supplementary for these continuous regressors. Predicted
counts are conceptually max(0, expm1(predicted_log1p)), rounded half-up to the
nearest integer. Exact matching can be computed in log space to avoid
overflow. A prediction is detected as nonzero at a decoded continuous count
of at least 0.5. Report positive recall, zero specificity, positive precision
and balanced accuracy together; positive recall alone rewards predicting
everything as expressed. Thresholds are fixed here and will not be tuned.
Negative unrounded log predictions remain unmodified for log-scale errors;
only count decoding clips at zero. Empty strata return null with support zero.

Original e175, its continued e300 model, recurrent e175, and geometry e200
are distinct frozen endpoints. All are seed 0; original and continuation are
one training lineage, not independent seeds. Recurrent training completed but
source artifact finalization failed; retain that status visibly. Different
training duration limits architecture attribution. The failed duplicate
continuation is recorded as excluded to avoid double counting.

The estimand is fitted-cohort transductive partial-gene reconstruction. Cells
and repeated masks are not biological replicates; cores are descriptive
strata, and no patient-level confidence interval or generalization claim is
made. Observed same-cell genes and permitted morphology remain available.
Raw count is observed measurement, not latent true expression. Zero counts can
reflect nondetection. There is no graph-specific, mechanistic or causal claim.

Controls: all-zero raw-count predictor and the prepared shared equal-core
per-gene mean log1p predictor (standardized zero). The latter is an all-fit,
target-derived descriptive reference, not a leakage-free held-out baseline.
Synthetic positive/zero fixtures verify recovery of perfect predictions,
threshold behavior and metric decomposition. No new spatial null is needed
for this accuracy-only question, and no graph-use claim will be made.

Data: immutable prepared cohort and graph manifests referenced by source
resolved configs. Load one complete core at a time; no clinical fields or
identifiers are exported. Source checkpoint, core NPZ, graph files and
normalization statistics must pass their recorded checksums. Current model
loaders must load state strictly and replay the original all-entry diagnostics
within a maximum absolute difference of 2e-6 before new metrics are accepted.
Use the source diagnostic's evaluation precision and complete receiver-wise
softmax; no neighbor sampling, graph truncation or new masks are permitted.

Acceptance: synthetic tests pass; pilot covers every model on core 21 and the
largest core 23; masks/targets match; original all-entry metrics replay;
finite predictions and source hashes pass; support partitions agree; weighted
zero/positive squared-error sums reconstruct all-entry squared error; all
14 cores finish for each model; outputs contain no row-level predictions;
each evaluation is registered, verified and marked completed. Stop on source,
shape, mask, replay, decomposition or finite-value mismatch. Diagnose rather
than blindly retry. Missing outputs or failures remain visible.

GPU plan: four safely idle RTX 3090 devices, one model per device, core 21 then
core 23 pilots before the remaining cores. PyTorch 2.13.0+cu130 / CUDA 13.0
visibility was verified on all four devices. Pilot measured peak memory must
remain below 21 GiB on each 23.56-GiB device, with finite outputs and practical
runtime. Record runtime, allocated/reserved VRAM, host-memory peak, source
precision and device for every core. This is full inference, not retraining.

Registered evaluation output is staged at
`scratch/active_runs/<source_run_id>/posthoc_reports/so2_nonzero_metrics/v1/`
and published immutably to
`reports/analyses/so2_nonzero_metrics/<source_run_id>/v1/`.
The cross-model report is under
`reports/analyses/so2_nonzero_metrics/comparison/v1/`.
Original training bundles, checkpoints and statuses remain unchanged.

## Entry points and verification

Run from the project root. Four workers use original/continued/recurrent/geometry
on CUDA devices 0/1/2/3 respectively, with four CPU threads per worker. Both
representative pilot cores passed for every model before full evaluation.

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_so2_nonzero_metrics.py
PYTHONPATH=src /venv/main/bin/python scripts/analysis/compare_so2_nonzero_accuracy.py --help
```

The exact preparation, pilot and full sequence used below preserves a barrier
between phases and propagates any worker failure. Existing completed v1
evaluations are verified rather than recomputed; reproducing inference with
changed code or inputs requires a separately registered, versioned evaluation.

```bash
PYTHONPATH=src /venv/main/bin/python - <<'PY'
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
models = ('original', 'continued', 'recurrent', 'geometry')
script = 'scripts/analysis/compare_so2_nonzero_accuracy.py'
def run(model, phase):
    subprocess.run([sys.executable, script, '--model', model, '--phase', phase], check=True)
for model in models:
    run(model, 'prepare')
for phase in ('pilot', 'full'):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda model: run(model, phase), models))
PY
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_nonzero_accuracy.py
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_nonzero_accuracy.py --check
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate --verify-sources --verify-payloads
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog --check
```

## Completed evaluation and interpretation

All four registered evaluations completed on 2026-09-05. The
[comparison report](../../../reports/analyses/so2_nonzero_metrics/comparison/v1/report.md)
contains the figure and links to complete equal-core, per-core, count-bin and
paired-difference tables. The frozen pre-evaluation task contract remains inside
each source evaluation bundle. No checkpoint selection or training occurred.

| Endpoint | Positive standardized MSE | Positive log1p MSE | Positive exact count | Positive detection recall |
|---|---:|---:|---:|---:|
| Original e175 | 8.862753 | 0.688236 | 3.63% | 13.15% |
| Continued e300 | 8.850479 | 0.681247 | 3.75% | 13.50% |
| Recurrent e175 | 8.969275 | 0.696982 | 3.51% | 12.85% |
| Geometry e200 | 8.915118 | 0.690519 | 3.61% | 13.09% |
| Shared gene mean | 8.780938 | 0.756558 | 1.97% | 7.44% |

Geometry positive standardized MSE is 0.591% above original and 0.730% above
continued, with higher error in 13/14 and 14/14 cores respectively. Geometry
improves zero MSE to 0.038964, from 0.042920 original and 0.042278 continued.
This supports a zero/nonzero tradeoff, rather than the proposed positive-error
advantage. The continued endpoint leads the four models on positive standardized
MSE, log1p MSE, exact count accuracy and detection recall. The differences are
small in error magnitude and are not a controlled architecture comparison.

Count-stratified standardized MSE is higher for geometry than both original
and continued in every aggregate bin (1, 2, 3, 4–7, 8+). The 8+ bin is 11.80%
worse than continued on standardized MSE and 17.27% worse on log1p MSE, whereas
its log1p MSE is 0.118% better than original. This is consistent with additional
training in the original lineage helping larger observed counts. It does not
isolate the effect of architecture or establish reproducibility across seeds.
Geometry improves positive standardized MSE over recurrent in 12/14 cores,
with a 0.604% lower equal-core mean.

All four models have worse positive standardized MSE than the gene-mean
reference, while improving its positive log1p MSE and count accuracy. These
metrics weight genes differently; aggregate reconstruction gain cannot be
assumed to imply better positive reconstruction on every scale. The mean is
target-derived from this fitted cohort, not an independently trained baseline.
Signed positive-entry errors are negative for all models, consistent with
underprediction. Detection recall is only 12.85–13.50%, with zero specificity
99.21–99.24% at the fixed count threshold. Always predicting zero achieves
89.59% overall exact accuracy, above the learned models' approximately 89.30%;
overall count accuracy alone is therefore misleading for these sparse targets.
Positive entries nevertheless contribute most standardized squared error,
despite being a minority of entries.

Alternative explanations remain training duration, the combined attention-score
change and the shared robust-loss objective. Core differences do not establish
patient replication. The maximum claim is a descriptive fitted-cohort
reconstruction comparison of these seed-0 endpoints. No graph-specific gain,
generalization, mechanism or causal interpretation is supported. The recurrent
source run remains failed for artifact finalization, while its registered
post-hoc evaluation completed successfully. The failed duplicate continuation
remains excluded, and no unfavorable model or core was omitted.

Acceptance checks passed: 15 metric tests; eight representative pilot receipts;
56/56 completed core/model receipts with identical masks and count supports;
strict checkpoint state/checksum and input verification; exact replay of all
six saved diagnostic quantities (maximum absolute difference 0); verified
zero/positive and count-bin decompositions; equal shared baseline receipts;
four completed registry evaluations and four registered manifests. The final
comparison check verifies 12 report files and 160 source files. All registered
evaluations have empty error logs and no failure marker. Full row-level
predictions were not exported.

Repository health check:
`PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor` reports the
pre-existing `checkpoint_catalog_incomplete` issue (41 checkpoint artifacts,
40 catalog entries). The missing entry is recurrent source artifact 562,
`checkpoints/last.ckpt`, registered on 2026-08-31 before this evaluation. This
task independently verified its file/state checksums and exact diagnostic
replay; the historical finalization/catalog defect remains visible and was not
silently repaired or relabeled. Registry integrity and configuration checks
passed. This task created no checkpoints.

Resource observations include pilots and full inference, without preparation
overhead. Runtime is the sum of per-core durations and includes metric
aggregation; independent models ran concurrently. All devices are RTX 3090;
PyTorch is 2.13.0+cu130 and CUDA is 13.0.

| Model / device | Core runtime (s) | Inference only (s) | Peak allocated VRAM (GiB) | Peak reserved VRAM (GiB) | Peak host RSS (GiB) |
|---|---:|---:|---:|---:|---:|
| Original / 0 | 346.90 | 59.64 | 1.105 | 2.449 | 9.108 |
| Continued / 1 | 346.16 | 58.46 | 1.105 | 2.449 | 9.104 |
| Recurrent / 2 | 441.82 | 67.12 | 1.092 | 2.436 | 9.079 |
| Geometry / 3 | 369.43 | 71.09 | 1.029 | 1.365 | 9.166 |

Decision: do not treat geometry as an accuracy improvement on these endpoints.
Before interpreting architecture differences, the next discriminating test is
a matched-duration, repeated-seed comparison on patient-held-out splits against
mean, cell-autonomous and context baselines, with both error scales and
zero/positive strata fixed in advance. Future training should preserve these
fixed-mask metrics per epoch; the historical checkpoints do not support a
retrospective nonzero curve for every epoch. That new experiment is not part of
this completed post-hoc task.

The curated conclusion is
[`res_so2_nonzero_reconstruction_comparison`](../../../results/descriptive_analysis/relative_qkv/res_so2_nonzero_reconstruction_comparison/README.md),
which references immutable source manifests and keeps evidence dimensions
separate.
