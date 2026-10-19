# Masked-expression Pearson correlation of existing SO2 predictors

On log1p observed counts, the current Local Gene Residual model has mean cell-wise Pearson **0.3156** across its two seeds on the reused validation donor. The historical BAGM endpoint has **0.4219** on its fitted cohort. These values describe different populations and masking protocols; they are not a matched model ranking.

Both local seeds and the historical endpoint were fixed from existing error reports before this analysis. No model was retrained and no checkpoint, seed or threshold was selected using these correlations.

## Primary scale: log1p observed counts

| Model | Mean cell Pearson | Mean gene Pearson | Pooled Pearson | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Local residual, seed 0 | 0.3156 | 0.2601 | 0.4403 | 0.1466 | 0.2523 |
| Local residual, seed 1 | 0.3155 | 0.2596 | 0.4403 | 0.1469 | 0.2523 |
| Local residual, mean of both seeds | 0.3156 | 0.2598 | 0.4403 | 0.1468 | 0.2523 |
| Historical BAGM, seed 0 | 0.4219 | 0.2713 | 0.5612 | 0.1196 | 0.2853 |

Cell-wise Pearson correlates predictions and targets across the masked genes within each cell, then averages valid cells. Gene-wise Pearson correlates across masked cells for each gene, then averages valid genes. Pooled Pearson uses all masked entries together and can be influenced by differences in average abundance between genes. These three quantities answer different questions. All original masked zero and positive observations are included.

The local mean is an arithmetic mean of the two model scores, not an ensemble prediction. Its seed range for cell-wise Pearson is 0.3155–0.3156. Seeds are technical replicates and do not provide independent-donor uncertainty.

Local log-scale predictions are log1p of the marginal expected count, not expected log1p counts. Historical log predictions are inverse-standardized model outputs without clipping negative values.

## Native and additional output scales

Local model, raw observed counts:

| Model | Mean cell Pearson | Mean gene Pearson | Pooled Pearson | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Local residual, seed 0 | 0.3843 | 0.2810 | 0.6070 | 0.2232 | 1.1019 |
| Local residual, seed 1 | 0.3841 | 0.2794 | 0.6070 | 0.2236 | 1.0997 |
| Local residual, mean of both seeds | 0.3842 | 0.2802 | 0.6070 | 0.2234 | 1.1008 |

Historical BAGM, its native gene-standardized log1p scale:

| Model | Mean cell Pearson | Mean gene Pearson | Pooled Pearson | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Historical BAGM, seed 0 | 0.1959 | 0.2713 | 0.2970 | 0.4009 | 1.0017 |

MAE/RMSE units change with the output scale. Gene-specific standardization can also change cell-wise and pooled Pearson; a common log1p scale alone does not make these experiments comparable.

## Equal-core sensitivity analysis

Each core is scored separately, then core scores are averaged equally. The local row additionally averages both seeds. This differs from the global table's valid-cell weighting and global per-gene correlations.

| Model | Mean cell Pearson | Mean gene Pearson | Pooled Pearson | MAE | RMSE |
|---|---:|---:|---:|---:|---:|
| Local residual, seed 0 | 0.3145 | 0.2625 | 0.4403 | 0.1457 | 0.2519 |
| Local residual, seed 1 | 0.3146 | 0.2620 | 0.4403 | 0.1460 | 0.2519 |
| Local residual, mean of both seeds | 0.3146 | 0.2622 | 0.4403 | 0.1458 | 0.2519 |
| Historical BAGM, seed 0 | 0.3909 | 0.2505 | 0.5022 | 0.1114 | 0.2755 |

Equal-core RMSE is the arithmetic mean of per-core RMSE, not the square root of mean per-core MSE. The complete CSV keeps these aggregation labels explicit.

## Correlation support and undefined values

Fewer than two selected values or a centered sum of squares at most 1e-12 in either vector makes correlation undefined. Undefined correlations are excluded from the means and counted below; they are never replaced with zero.

| Model | Scale | Cells | Masked entries | Valid / undefined cell correlations | Valid / undefined gene correlations |
|---|---|---:|---:|---:|---:|
| Local residual, seed 0 | raw_count | 37,367 | 18,708,207 | 36,474 / 893 | 1,000 / 0 |
| Local residual, seed 0 | log1p | 37,367 | 18,708,207 | 36,474 / 893 | 1,000 / 0 |
| Local residual, seed 1 | raw_count | 37,367 | 18,708,207 | 36,474 / 893 | 1,000 / 0 |
| Local residual, seed 1 | log1p | 37,367 | 18,708,207 | 36,474 / 893 | 1,000 / 0 |
| Historical BAGM, seed 0 | standardized | 246,063 | 122,903,254 | 245,556 / 507 | 1,000 / 0 |
| Historical BAGM, seed 0 | log1p | 246,063 | 122,903,254 | 241,569 / 4,494 | 1,000 / 0 |

## Detection remains a separate question

Pearson measures co-variation and does not guarantee correct zero/nonzero decisions. Targets `[0, 1, 2]` and predictions `[1, 3, 5]` have Pearson 1, although the observed zero is predicted positive and every magnitude is wrong. The following existing local detection metrics were replayed unchanged, using the probability head and threshold 0.5. Average precision is approximated using 4,096 probability bins; no threshold or metric was optimized here.

| Model | Precision | Recall | F1 | Average precision |
|---|---:|---:|---:|---:|
| Local residual, seed 0 | 0.7336 | 0.0727 | 0.1324 | 0.3021 |
| Local residual, seed 1 | 0.7345 | 0.0722 | 0.1314 | 0.3020 |
| Local residual, mean of both seeds | 0.7340 | 0.0725 | 0.1319 | 0.3020 |

These detection labels mean observed count greater than zero in this assay; an observed zero does not establish biological RNA absence. Historical BAGM has no probability-head detection result in this replay.

## Selection, populations and permitted conclusions

The Local Gene Residual family was chosen because its completed two-seed aggregate had the lowest raw-count RMSE among the current count-model comparisons. Its training loss was hurdle count negative log-likelihood, and both saved epoch-30 checkpoints were originally selected by validation NLL. It was fitted on 208,696 cells from twelve cores/six donors and evaluated here on 37,367 cells from two cores of one different donor. That validation donor was repeatedly used for selection; this is not a sealed test estimate.

Historical Relative-QKV uses its fixed epoch-300 endpoint, chosen from the previous four-endpoint comparison by standardized MSE. It was Huber-trained on all fourteen cores, including the two cores later used for validation. All 246,063 replayed cells therefore belong to its fitted cohort; only one seed is available.

The local protocol masks Uniform{1,…,1000} target genes and allows clean non-self neighbor summaries. The historical protocol masks Uniform{0,…,1000} genes simultaneously in every graph cell. Different available neighbor information, populations, fitting scopes and normalization preclude attributing score differences to architecture or training-set size. These results add descriptive reconstruction metrics, without establishing external benchmark superiority, independent-test generalization, biological mechanism or causal effects.

## Verification and immutable sources

All three input bundles must pass checksum and completion verification before this report is written. The local replays match archived native metrics within absolute tolerance 1e-5; the historical replay uses 2e-6. Checkpoint, mask and data identities are verified by the input replays. Float64 correlation accumulation was checked against SciPy on a fixture with masking, singleton and constant vectors. No row-level prediction matrices are exported.

| Endpoint | Immutable source run | Replay bundle |
|---|---|---|
| Local residual, seed 0 | `r_20261004T072107Z_9d2b00cb_s000_f00_a01_f41786d5` | [Metrics](../local_s0/result.json), [manifest](../local_s0/manifest.json) |
| Local residual, seed 1 | `r_20261004T072107Z_9d2b00cb_s001_f00_a01_1caff983` | [Metrics](../local_s1/result.json), [manifest](../local_s1/manifest.json) |
| Historical BAGM, seed 0 | `r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2` | [Metrics](../historical_e300/result.json), [manifest](../historical_e300/manifest.json) |

[Complete summary CSV](summary.csv) retains all scales, seeds and aggregation rules. [Input provenance](input_provenance.json) records source manifest/result/checkpoint hashes; [manifest](manifest.json) binds this report and its source snapshot.

Reproduce from the project root with `BAGM_ROOT=/workspace/BAGM PYTHONPATH=/workspace/BAGM-relative-qkv-analysis/src:/workspace/BAGM/scripts/analysis /venv/main/bin/python scripts/analysis/report_so2_prediction_pearson.py`; use `--check` to verify the existing immutable report. An existing comparison is never overwritten.
