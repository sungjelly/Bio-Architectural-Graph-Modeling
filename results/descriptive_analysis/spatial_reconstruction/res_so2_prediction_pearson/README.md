# SO2 masked-expression Pearson

All three frozen checkpoint evaluations passed source/mask checks and replayed their archived native metrics. This record adds descriptive correlation scores; it does not compare the models on a matched benchmark.

| Endpoint and scope | Mean cell Pearson | Mean gene Pearson | Pooled Pearson |
|---|---:|---:|---:|
| Local Gene Residual, mean of seeds 0/1; reused validation donor | 0.3156 | 0.2598 | 0.4403 |
| Historical Relative-QKV, seed 0; all 14 fitted cores | 0.4219 | 0.2713 | 0.5612 |

Values use log1p observed counts. Cell-wise Pearson averages correlations across masked genes within valid cells. Gene-wise Pearson averages correlations across masked cells within valid genes. Pooled Pearson combines entries and can reflect differences in average gene abundance. The local row averages two model scores; it is not an ensemble prediction.

Local models were trained on 12 cores and evaluated on 37,367 cells from one held-out donor in two other cores, which were reused for checkpoint selection. Historical training and normalization used all 14 cores and all 246,063 replayed cells. The models also differ in masks and available neighbor information. The larger historical coefficient therefore does not establish architecture superiority, an effect of adding validation cells to training, or SOTA.

The local raw-count MAE/RMSE average 0.2234/1.1008. Pearson does not replace detection precision/recall/F1/AP. Undefined cell correlations are excluded and counted: 893 per local seed and 4,494 historical log-scale cells; every global gene correlation is defined. All seeds and native masks were retained, with no retraining or new selection.

[Full report and scale/aggregation tables](../../../../reports/analyses/so2_prediction_pearson/v1/comparison/report.md) · [Workflow and reproduction commands](../../../../experiments/campaigns/cmp_20261015_so2_prediction_pearson/README.md).
