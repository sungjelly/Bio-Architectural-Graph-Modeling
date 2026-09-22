# SO2 nonzero reconstruction comparison

**Outcome: negative for the proposed geometry nonzero-MSE advantage at these frozen endpoints.** The continued original e300 model has the lowest positive-entry standardized MSE among the four evaluated models. This is an exploratory fitted-cohort comparison, not a model-family or generalization claim.

| Frozen endpoint | Positive standardized MSE | Positive log1p MSE | Positive exact count | Positive detection recall |
|---|---:|---:|---:|---:|
| Original e175 | 8.862753 | 0.688236 | 3.63% | 13.15% |
| Continued e300 | 8.850479 | 0.681247 | 3.75% | 13.50% |
| Recurrent e175 (source finalization failed) | 8.969275 | 0.696982 | 3.51% | 12.85% |
| Geometry e200 | 8.915118 | 0.690519 | 3.61% | 13.09% |
| Zero-count reference | 10.718155 | 1.073846 | 0.00% | 0.00% |
| All-fit gene-mean reference | 8.780938 | 0.756558 | 1.97% | 7.44% |

Geometry has 0.591% higher positive standardized MSE than original e175, worsening in 13/14 fitted cores, and 0.730% higher than continued e300, worsening in 14/14. Geometry has lower zero-entry MSE, so aggregate error hides a zero/nonzero tradeoff.

The shared gene-mean reference has lower positive standardized MSE than all four models, but higher positive unstandardized log1p MSE and poorer positive count accuracy and detection. Standardization weights genes differently; these outcomes must remain separate. The gene-mean reference uses all-fit target-derived preprocessing. Positive detection recall is supplementary and must be interpreted with precision, zero specificity and balanced accuracy in the complete report.

Observed zeros account for 89.59% of masked entries on an equal-core basis. Always predicting zero therefore scores higher overall exact-count accuracy than the models. Overall exact accuracy alone is not evidence of useful positive-expression reconstruction.

The same original fixed masks, 14 full-core graphs, 246,063 fitted cells and 1,000 biological probes were reused. Positive means observed raw count greater than zero. Each core receives equal weight. Continuous count predictions are clipped at zero only for supplementary half-up integer decoding; detection uses a fixed 0.5 count threshold. Primary standardized errors and secondary unstandardized log1p errors retain continuous predictions.

All models use seed 0. Original and continued share one lineage, and the four checkpoints have unequal training exposure. Attention normalization, scale, modulation and bias change together in geometry, so the result does not isolate modulation. Observed same-cell genes and always-visible morphology can explain prediction without spatial information. There is no held-out patient, significance test, stability audit, mechanism-breaking spatial null, faithfulness test, external support or perturbational evidence. Zero counts may reflect nondetection, and observed positive counts are measurements rather than latent biological truth.

The recurrent source bundle retains failed finalization status; its checkpoint was strictly loaded and its post-hoc evaluation completed. The checkpoint is registered as artifact 562 but also has a pre-existing missing checkpoint-catalog entry. Independent checksum, strict state and original-metric replay checks passed; this evaluation does not repair or upgrade the source lifecycle or catalog status. The failed duplicate continuation is excluded. All four evaluation bundles completed all 14 cores, replayed original all-entry diagnostics within 2e-6, verified source hashes and support/error decompositions, and preserved aggregate-only outputs.

[Complete report](../../../../reports/analyses/so2_nonzero_metrics/comparison/v1/report.md) · [Comparison manifest](../../../../reports/analyses/so2_nonzero_metrics/comparison/v1/manifest.json) · [Figure](../../../../reports/analyses/so2_nonzero_metrics/comparison/v1/accuracy_comparison.png)

The exact contributing run IDs, five checksummed source manifests and separate evidence statuses are recorded in `result.yaml`. The result owns no copied checkpoints, datasets or prediction rows.

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_nonzero_accuracy.py --check
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate --verify-sources --verify-payloads
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog --check
```
