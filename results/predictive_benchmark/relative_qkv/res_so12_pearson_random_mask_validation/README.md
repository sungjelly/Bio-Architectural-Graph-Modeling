# SO1 + SO2 random-mask QKV validation

The completed four-layer Relative-Geometric QKV model improves masked-feature Huber loss and absolute error over zero and training-mean predictions, while squared error remains slightly worse. The overall evidence is **mixed**; this verified record has schema outcome **inconclusive**.

| Predictor | Equal-core Huber ↓ | Pooled MAE ↓ | Pooled MSE ↓ | Pooled Pearson ↑ |
|---|---:|---:|---:|---:|
| Zero standardized Pearson feature | 0.209358 | 0.422732 | 0.999974 | undefined |
| Training mean, equal core weights | 0.210243 | 0.428938 | 0.998908 | 0.032271 |
| Training mean, equal cell weights | 0.209102 | 0.420926 | 1.000043 | 0.000494 |
| QKV, selected epoch 151 | **0.196114** | **0.308160** | 1.005888 | **0.134499** |

The model lowers Huber loss by **6.33%** versus zero, **6.72%** versus the equal-core mean and **6.21%** versus the equal-cell mean. Huber improves in **8/8 validation cores** against each baseline. Pooled MAE falls 26.79–28.16%, while MSE increases 0.58–0.70%; pooled correlation remains weak. Constant zero refers to the standardized Pearson feature, not zero RNA counts, and its correlation is undefined. Pooled Pearson combines masked entries across genes and cells; it is not a gene-wise recovery metric.

The prespecified seed 0 and fold 0 are complete. Fold 0 denotes one fixed donor split, not cross-validation. Training uses 301,198 cells from 20 cores and 10 donor groups; validation uses 106,461 cells from eight cores and four different donor groups. Each cell independently masks 0–1,000 genes, with ten fresh training views per core per epoch and three fixed validation views. The four-layer, 5,003,016-parameter model uses visible-count-only Pearson inputs, visible morphology and coordinate-only within-core graphs. Targets are the frozen integrated Pearson features. AdamW uses a fixed learning rate of 0.0001 and masked Huber loss with delta one.

Training stopped by patience after **151 epochs / 1,510 optimizer updates**. The best and final checkpoint are both epoch 151. The last improvement exceeding the 0.0001 patience threshold occurred at epoch 126; smaller later improvements still updated the strict best checkpoint. Trainer runtime was **39,655.22 seconds (11 h 00 min 55 s)** on four RTX 3090 GPUs, with peak allocated memory 15.55 GiB per device.

All constants are fitted solely on training targets. The existing baseline report was generated at epoch 86; this record uses only its frozen constant results and pairs them with the final immutable epoch-151 validation metrics. Exact target-manifest, split, configuration/source binding, mask receipts and 159,816,944 masked-entry counts agree. The primary metric averages the three view means within each core and then weights the eight cores equally. MAE, MSE and correlation pool all masked entries. The selected final checkpoint was reloaded and its validation scores replayed successfully.

The strongest alternative explanation is that gene-specific robust locations and observed same-cell expression explain the gains without useful spatial information. Huber optimization can favor different predictions from squared-error-optimal means. No matched cell-autonomous or graph-ablation control was run. Validation donors had already been used for preprocessing diagnostics and then for checkpoint selection, so selection bias and only four validation donors limit interpretation. One seed cannot establish stability. The 8/8 core result is descriptive within this split, not independent patient replication or a sealed-test generalization estimate. No biological integration, mechanism, graph-specific gain or causal claim is supported here. The next discriminating comparison is a matched cell-autonomous predictor with stronger train-fitted robust constant controls, followed by a locked independent test.

Completed contributing run: `r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b`. The initial launch `r_20261016T090140Z_00dadce7_s000_f00_a01_cdb171a3` failed registry preprocessing-version validation before model execution and performed zero optimizer updates. Its explicit retry corrected registration metadata without changing scientific settings; the failure is retained in the result provenance.

The [final run summary](../../../../artifacts/runs/2026/10/r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b/summary.json), [replayed final validation](../../../../artifacts/runs/2026/10/r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b/diagnostics/validation_per_core.json) and [frozen baseline report](../../../../reports/analyses/so12_zero_mean/ev_so12_zero_mean_e086_20261016T155304Z/metrics.json) are referenced by checksum in [result.yaml](result.yaml). No checkpoints or datasets are copied into this record.

Validation of this record:

```bash
PYTHONPATH=src /venv/main/bin/python -c "from spatial_benchmark.result_catalog import validate_result_manifest; validate_result_manifest('results/predictive_benchmark/relative_qkv/res_so12_pearson_random_mask_validation/result.yaml', verify_payloads=True, verify_sources=True)"
```
