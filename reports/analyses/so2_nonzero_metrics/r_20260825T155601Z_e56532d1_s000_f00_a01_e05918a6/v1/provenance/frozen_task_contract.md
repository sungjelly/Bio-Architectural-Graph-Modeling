# SO2 nonzero reconstruction accuracy

Phase: pilot. Outcome: pending. Exploratory post-hoc evaluation requested after
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

Run from the project root. Exact evaluation commands and current results are
filled below when the pilot validates the implementation.

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_so2_nonzero_metrics.py
PYTHONPATH=src /venv/main/bin/python scripts/analysis/compare_so2_nonzero_accuracy.py --help
```
