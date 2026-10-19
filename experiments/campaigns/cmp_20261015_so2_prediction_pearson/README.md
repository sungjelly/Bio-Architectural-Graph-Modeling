# SO2 prediction Pearson replay

Phase: complete. Outcome: supported for the descriptive metric-extension task.
This is an exploratory, descriptive extension of existing frozen predictions.

The immutable [task contract](frozen_task_contract.md) fixes the scientific
question, source checkpoints, native masks, scales, verification tolerances,
resource gates and acceptance criteria before inference. It is authoritative;
this README records execution and outcomes without changing that contract.

## Question and scope

Measure how well existing masked-expression predictions covary with observed
expression. The primary presentation is mean cell-wise Pearson across masked
genes on the log1p-count scale. Gene-wise and pooled Pearson, MAE/RMSE,
per-core results and undefined-correlation counts expose alternative explanations
such as average gene abundance, normalization and unequal core sizes.

The Local Gene Residual family was chosen using existing aggregate raw-count
RMSE, and both original best-NLL epoch-30 checkpoints are retained. It was
trained with hurdle count likelihood, not a direct MAE/RMSE loss. Twelve cores
from six donors supplied training; 37,367 cells in two cores from one other donor
supplied validation and checkpoint selection. This reused validation population
is not a sealed test set. Each target retains clean, non-self neighbor summaries
and independently hides Uniform{1,...,1000} genes.

The historical continued Relative-QKV endpoint was Huber-trained for 300 epochs
on all fourteen cores, including the two later designated validation. It has one
available seed and uses simultaneous Uniform{0,...,1000} masking across graph
cells. Model fitting and normalization used the entire cohort. Its 246,063-cell
evaluation is fitted-cohort reconstruction. These are different native tasks;
their correlations do not establish architecture superiority or an effect of
adding validation cells to training.

Raw targets and metadata remain local and immutable. Technical control probes
are excluded by the verified, ordered 1,000-gene panel. Only aggregate outputs
are saved. No model was trained, tuned or selected using the new correlations.

## Sources and execution

| Endpoint | Seed / epoch | Immutable source run |
|---|---|---|
| Local Gene Residual | 0 / 30 | `r_20261004T072107Z_9d2b00cb_s000_f00_a01_f41786d5` |
| Local Gene Residual | 1 / 30 | `r_20261004T072107Z_9d2b00cb_s001_f00_a01_1caff983` |
| Historical Relative-QKV | 0 / 300 | `r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2` |

The later local early-stopping proposal has no completed replacement endpoint.
The GeneMAE masking experiment is a separate study and is not substituted here.

Run from `/workspace/BAGM`. The replay imports the original local-model modules
from the analysis worktree; both that package and the main package are snapshotted
and checksum-checked before and after execution. Existing successful run bundles
are never modified.

```bash
export BAGM_ROOT=/workspace/BAGM
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_REPORT_ROOT=/workspace/BAGM/reports
export BAGM_RESULT_ROOT=/workspace/BAGM/results
export PYTHONPATH=/workspace/BAGM-relative-qkv-analysis/src:/workspace/BAGM:/workspace/BAGM/scripts/analysis
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
/venv/main/bin/python scripts/analysis/run_so2_prediction_pearson.py --prepare
supervisorctl start bagm_pearson_local_s0 bagm_pearson_local_s1 bagm_pearson_historical_e300
supervisorctl status bagm_pearson_local_s0 bagm_pearson_local_s1 bagm_pearson_historical_e300
# Run only after all three replays have completed successfully:
/venv/main/bin/python scripts/analysis/run_so2_prediction_pearson.py --verify
/venv/main/bin/python scripts/analysis/report_so2_prediction_pearson.py
/venv/main/bin/python scripts/analysis/report_so2_prediction_pearson.py --check
```

The installed supervisor wrapper is
`/opt/supervisor-scripts/bagm-prediction-pearson.sh`, copied from
`ops/supervisor/bagm-prediction-pearson.sh`; the configuration is
`/etc/supervisor/conf.d/bagm-prediction-pearson.conf`. Seeds 0 and 1 use physical
GPUs 0 and 1, and the historical replay uses GPU 2. Each process exposes only its
assigned device and holds the existing shared-project/exclusive-device locks.
No other GPU process is displaced. There are no automatic retries.

Preparation registers `ev_so2_prediction_pearson_v1_<source_run_id>` against each
source run. Active output is isolated under
`scratch/active_runs/<source_run_id>/posthoc_reports/so2_prediction_pearson/v1`.
Successful bundles publish to
`reports/analyses/so2_prediction_pearson/v1/{local_s0,local_s1,historical_e300}`;
the cross-run report publishes to the sibling `comparison` directory. Existing
successful outputs are verified, not overwritten. The report builder refuses an
existing comparison directory; use `--check` for subsequent verification.

## Verification and interpretation

`MaskedRegressionAccumulator` is checked against independent SciPy Pearson
calculations on partial masks, empty/singleton rows and constant vectors.
Correlations use float64 sufficient statistics; fewer than two masked entries
or a centered sum of squares at most 1e-12 yields undefined correlation.
Such correlations are excluded from means and counted, not replaced with zero.

Local replays preserve the original batch size 1,024, BF16 inference, masks and
training-only transforms. Their marginal expected raw count supplies raw metrics;
the log prediction is log1p of that mean. Native per-core/global errors must replay
within 1e-5. The historical replay preserves full-core inference and inverse
standardization without clipping negative log predictions. It must reproduce
original per-core/global native metrics and prior log-scale errors within 2e-6;
the first core must pass before remaining cores proceed. Source, mask, finite
prediction, resource and replay failures stop execution without relaxing criteria.

Pearson measures co-variation and does not replace detection precision, recall,
F1 or AP. No baseline superiority, SOTA ranking, independent-patient replication,
biological true-zero recovery, interaction or causal claim follows. Stronger
comparisons require matched masking/allowed context, training-only preprocessing
and independent held-out cohorts.

## Final results

All three replays completed and the comparison passed checksum/completion
verification. Existing native metric replay differences were zero for the local
global metrics and the historical equal-core aggregate. No tolerance was changed,
checkpoint retrained or source run excluded. Both local seeds remain visible.

| Endpoint and scope | Log1p cell Pearson | Log1p gene Pearson | Log1p pooled Pearson | Log1p MAE | Log1p RMSE |
|---|---:|---:|---:|---:|---:|
| Local seed 0, reused validation | 0.315573 | 0.260078 | 0.440309 | 0.146622 | 0.252289 |
| Local seed 1, reused validation | 0.315539 | 0.259568 | 0.440313 | 0.146902 | 0.252327 |
| Mean of both local model scores | 0.315556 | 0.259823 | 0.440311 | 0.146762 | 0.252308 |
| Historical seed 0, all fitted cores | 0.421892 | 0.271327 | 0.561158 | 0.119613 | 0.285272 |

The local mean is an arithmetic mean of two model scores, not an ensemble.
Different populations, masks and neighbor information prevent interpreting these
rows as a matched ranking. Local raw-count cell/gene/pooled Pearson average
0.384223/0.280235/0.607006, with MAE 0.223391 and RMSE 1.100785. Scale changes
matter: historical native standardized cell Pearson is 0.195858, whereas its
log1p coefficient is 0.421892. Gene-specific standardization changes cell-wise
and pooled correlation.

Each local model scored 18,708,207 masked entries, with 36,474 valid and 893
undefined cell correlations. Historical reconstruction scored 122,903,254
entries, with 241,569 valid and 4,494 undefined log-scale cell correlations.
All 1,000 global gene correlations were defined. Equal-core sensitivity, native
scales, per-core support and unchanged local detection scores remain in the
[full report](../../../reports/analyses/so2_prediction_pearson/v1/comparison/report.md).

Hardware was NVIDIA GeForce RTX 3090, PyTorch 2.13.0+cu130 / CUDA 13.0, with one
process per allocated GPU and four CPU threads each. Local seed 0 took 155.95
seconds, seed 1 took 120.90 seconds, and the historical replay took 318.47
seconds; these include input loading and verification. Each local replay peaked
at 0.458 GiB allocated VRAM. Historical peak allocated VRAM was 1.105 GiB and
reserved VRAM 2.449 GiB. Detailed resource receipts are
retained in every evaluation bundle; no execution failure occurred.

The curated record is
[`res_so2_prediction_pearson`](../../../results/descriptive_analysis/spatial_reconstruction/res_so2_prediction_pearson/README.md).
Its supported outcome means the requested metric extension and replay checks
completed; evidence dimensions explicitly leave predictive superiority, patient
replication, biological mechanisms and perturbation support unestablished.

Final record/catalog checks, from the project root:

```bash
/venv/main/bin/python scripts/results/manage_results.py validate --verify-payloads
/venv/main/bin/python scripts/results/manage_results.py catalog
/venv/main/bin/python scripts/results/manage_results.py catalog --check
```

The new result and all eight source references pass
`validate_result_manifest` with both source and payload verification. All seven
result records pass schema/payload validation; regenerated catalogs pass the
deterministic check and preserve the six earlier records. Catalog content SHA256:
`4fc9030aa37e74e2096155edf273845d73a5eaa7220ea4eb0b9d40f5f394e3e4`.

Global `validate --verify-sources` still fails on the previously documented
unrelated missing file
`experiments/campaigns/cmp_20261004_local_gene_residual/frozen_task_contract.md`.
This analysis does not recreate or alter that source; it is separate from the
new record's successful source verification.
