# SO2 masked nonzero reconstruction

Exploratory evaluation of four frozen seed-0 endpoints on the same fixed masks across all 14 fitted cores (246,063 cells). Lower error is better. Each core receives equal weight.

![Nonzero accuracy, detection, error decomposition, and paired core differences](accuracy_comparison.png)

The figure shows positive-entry errors, supplementary rounded count accuracy, detection tradeoffs, and each stratum's contribution to total squared error. Grey bars are the shared descriptive references.

| Predictor | Positive standardized MSE | Positive standardized MAE | Positive exact count | Positive detection recall | Zero specificity | Detection balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Original e175 | 8.862753 | 2.645668 | 3.63% | 13.15% | 99.22% | 56.18% |
| Continued e300 | 8.850479 | 2.639939 | 3.75% | 13.50% | 99.21% | 56.36% |
| Recurrent e175* | 8.969275 | 2.665164 | 3.51% | 12.85% | 99.24% | 56.04% |
| Geometry e200 | 8.915118 | 2.654261 | 3.61% | 13.09% | 99.23% | 56.16% |
| Zero count | 10.718155 | 3.036763 | 0.00% | 0.00% | 100.00% | 50.00% |
| Gene mean | 8.780938 | 2.656020 | 1.97% | 7.44% | 99.08% | 53.26% |

**Continued e300 has the lowest positive-entry standardized MSE among these four endpoints.** Nonzero means observed raw count > 0. Exact count accuracy requires the decoded integer to match the observed count; detection only asks whether the decoded count is positive.

Observed zeros account for **89.59%** of masked entries on an equal-core basis. Consequently, always predicting zero achieves that same overall exact-count accuracy. The learned models' overall exact-count accuracy ranges from **89.30% to 89.30%**. This makes overall exact accuracy a poor standalone measure of reconstruction quality; the models were trained for continuous Huber loss, not integer classification.

## Sensitivity to the error scale

Gene standardization weights squared log errors by the inverse of each gene's normalization variance. The unstandardized log1p metric therefore answers a differently weighted reconstruction question. Keep both scales visible when comparing with the target-derived gene-mean reference.

| Predictor | Positive standardized MSE | Positive log1p MSE |
|---|---:|---:|
| Original e175 | 8.862753 | 0.688236 |
| Continued e300 | 8.850479 | 0.681247 |
| Recurrent e175* | 8.969275 | 0.696982 |
| Geometry e200 | 8.915118 | 0.690519 |
| Zero count | 10.718155 | 1.073846 |
| Gene mean | 8.780938 | 0.756558 |

## Paired positive-error comparisons

Geometry minus Original e175 positive standardized MSE is **+0.052365 (+0.591%)**; geometry has lower MSE in **1/14 cores**. These are descriptive paired differences, without a post-hoc superiority threshold or significance claim.

Geometry minus Continued e300 positive standardized MSE is **+0.064639 (+0.730%)**; geometry has lower MSE in **0/14 cores**. These are descriptive paired differences, without a post-hoc superiority threshold or significance claim.

## Zero versus positive errors

Each core's all-entry MSE equals its zero squared-error sum divided by all masked entries plus its positive squared-error sum divided by all masked entries. Averaging those contributions across cores preserves the exact decomposition; weighting equal-core stratum means by a pooled support fraction would not.

| Predictor | All MSE | Zero MSE | Positive contribution to all MSE | Zero contribution to all MSE | Positive signed error |
|---|---:|---:|---:|---:|---:|
| Original e175 | 0.948670 | 0.042920 | 0.910653 | 0.038017 | -2.604543 |
| Continued e300 | 0.947419 | 0.042278 | 0.909940 | 0.037479 | -2.594335 |
| Recurrent e175* | 0.955829 | 0.037155 | 0.922863 | 0.032966 | -2.628830 |
| Geometry e200 | 0.951623 | 0.038964 | 0.917064 | 0.034559 | -2.616656 |
| Zero count | 1.110121 | 0.000000 | 1.110121 | 0.000000 | -3.036763 |
| Gene mean | 0.999797 | 0.102293 | 0.908108 | 0.091689 | -2.651569 |

Signed error is prediction minus target; negative positive-entry bias describes underprediction on the standardized log1p scale. Detailed count bins (1, 2, 3, 4–7, 8+) and errors on the unstandardized log1p scale are in `countbins.csv` and `percore.csv`.

## Scope and limitations

- The estimand is partial-gene reconstruction of fitted cells. Observed same-cell genes and permitted morphology remain available. These results do not measure patient-held-out prediction or isolate graph use.
- Every inclusion mask uses observed raw count, not standardized target sign. Observed positive counts are not latent ground-truth expression; zero counts can reflect nondetection.
- Original e175 and continued e300 are one training lineage. Recurrent ends at e175 and geometry at e200. Training duration and the complete attention-score change limit attribution to geometry modulation.
- *Recurrent training and checkpoint reload completed, but its original bundle has failed artifact-finalization status. That source status is retained. The failed duplicate continuation is excluded rather than counted as another seed.
- Zero count and shared equal-core gene-mean log1p are all-fit descriptive references. The gene mean is target-derived and is not a leakage-free held-out baseline.
- Rounded-count diagnostics use fixed half-up decoding and a 0.5 continuous-count detection threshold. Positive recall, zero specificity, precision, and balanced accuracy must be interpreted together. Undefined precision is retained as null.
- Only seed 0 is available. Core variation is descriptive; no formal confidence interval, significance test, graph-specific gain, biological mechanism, or causal claim is supported.

## Files and verification

`modelsummary.csv` contains equal-core means and supported-core counts. `percore.csv` retains every core. `countbins.csv` separates observed count ranges. `paired_core_deltas.csv` preserves every contrast against original and continued models. `source_references.json`, `verification.json`, and `manifest.json` bind all source and report files by checksum.

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_nonzero_accuracy.py --check
```
