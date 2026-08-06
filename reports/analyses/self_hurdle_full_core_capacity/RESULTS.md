# Self-only continuous-hurdle full-core capacity: results

Outcome: **NEGATIVE** for the frozen two-core capacity hypothesis.

This is an exploratory, graphless, held-in reconstruction result. It is not a validation/test estimate and does not test cell-cell interaction or biological mechanism.

## Prespecified whole-node gate

| Core | Detection BA model/ref | Gain (pp) | Positive Huber model/ref | Improvement | Positive-state MAE model/ref | Improvement | Core pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ANC-03 | 64.85% / 51.09% | 13.76 | 0.3642 / 0.4949 | 26.40% | 0.5653 / 0.5700 | 0.82% | no |
| ANC-05 | 65.63% / 50.35% | 15.29 | 0.3741 / 0.5266 | 28.95% | 0.5707 / 0.5800 | 1.60% | no |

The campaign passes only if all three criteria pass on both cores. Failure of one criterion on either core makes the frozen hypothesis negative.

## Estimands reported separately

Each percentage cell is `model / all-fit per-gene reference`. Loss/error columns report relative improvement; positive means lower error.

| Core | Mask estimand | Detection BA | Sensitivity | Specificity | Precision | Positive exact | Positive ±1 | State-8 exact | State-8 balanced | Huber impr. | State-MAE impr. | Reconstructed log1p-MAE impr. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ANC-03 | whole_node | 64.85% / 51.09% | 64.89% / 2.37% | 64.81% / 99.82% | 16.91% / 58.82% | 52.47% / 54.99% | 92.66% / 90.77% | 61.48% / 89.95% | 17.87% / 13.44% | 26.40% | 0.82% | -234.13% |
| ANC-03 | partial_gene | 66.25% / 51.08% | 71.95% / 2.34% | 60.54% / 99.82% | 16.71% / 58.39% | 49.95% / 55.12% | 93.40% / 90.83% | 57.79% / 89.97% | 18.59% / 13.42% | 24.96% | -2.09% | -279.94% |
| ANC-03 | spatial_block | 64.24% / 51.01% | 63.49% / 2.22% | 64.99% / 99.80% | 17.06% / 55.75% | 53.56% / 55.70% | 92.20% / 90.38% | 61.54% / 89.74% | 17.80% / 13.37% | 24.62% | 1.07% | -221.46% |
| ANC-05 | whole_node | 65.63% / 50.35% | 67.11% / 0.73% | 64.16% / 99.96% | 15.82% / 67.02% | 49.95% / 51.03% | 94.12% / 92.72% | 61.14% / 90.85% | 17.72% / 12.75% | 28.95% | 1.60% | -285.56% |
| ANC-05 | partial_gene | 66.79% / 50.34% | 71.01% / 0.71% | 62.57% / 99.96% | 16.00% / 66.15% | 48.27% / 51.10% | 94.82% / 92.65% | 59.76% / 90.86% | 18.27% / 12.75% | 28.10% | 0.45% | -321.00% |
| ANC-05 | spatial_block | 65.99% / 50.35% | 68.02% / 0.74% | 63.96% / 99.96% | 15.73% / 66.80% | 50.14% / 51.31% | 93.95% / 92.57% | 61.08% / 90.98% | 17.87% / 12.75% | 28.27% | 1.39% | -289.18% |

## Resource and provenance audit

| Core | Run ID | GPU | Registry runtime | Training runtime | Peak VRAM | Recorded disk | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| ANC-03 | `r_20260729T172739Z_0570cda7_s000_f00_a01_7449d530` | 2 | 2.44 min | 0.78 min | 1.645 GiB | 49.383 GB | `4b14d67ea935bf1fe397747dcc27507008be017b85409f6c71788ba4173ccb49` |
| ANC-05 | `r_20260729T172739Z_ea3ca4d1_s000_f00_a01_74a0b1a1` | 3 | 1.23 min | 0.44 min | 0.969 GiB | 49.334 GB | `95a1e1c4e2712f34263b2bd1b1bb55253e338b89e075216283bb382757f39170` |

Resource pilots:

| Core | Run ID | GPU | Registry runtime | Peak VRAM | AMP loss discrepancy | Recorded disk | Checkpoint SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| ANC-03 | `r_20260729T172556Z_47141900_s000_f00_a01_289503ce` | 0 | 0.99 min | 1.645 GiB | 0.00000191 | 49.165 GB | `99a543999c2714b0212b5340d15623eb889da0d77ee03f46964ddea9cbffc04d` |
| ANC-05 | `r_20260729T172557Z_de468a3e_s000_f00_a01_49f86de9` | 1 | 0.58 min | 0.968 GiB | 0.00000131 | 49.109 GB | `598b5b4f7b2a3b1b6b46ea04aa9b22cad8de5e708c0f07394e63e37351c2fbce` |

Conservative registered campaign GPU use was 0.0873 GPU-hours; the preferred and absolute limits were 12 and 24. Maximum recorded disk use was 49.383 GB, below the 55 GB hard stop.

Both resource-pilot and both science bundles, final-epoch checkpoints, configuration hashes, fixed masks, per-gene references, and checkpoint-catalog records verified. No best run or checkpoint was selected.

## Interpretation

The prespecified two-core representation-capacity hypothesis was not supported; individual held-in metric gains are descriptive.

The per-gene reference is not a strong learned control: at a fixed 0.5 detection threshold it predicts almost all rare genes as absent. Detection balanced-accuracy gains therefore do not by themselves establish biological signal. Likewise, high state-8 exact accuracy is zero-dominated; its reference can exceed the model while balanced accuracy moves in the opposite direction.

The strongest unresolved explanations are transductive memorization, common within-cell co-expression, and morphology or technical intensity. There is no unseen-core evaluation, morphology-only learned baseline, biological-label readout, graph input, interaction test, mechanism test, or causal test.

The complete machine-readable comparison is in `comparison.json`; all model/reference metric pairs are in `metrics.csv`.
