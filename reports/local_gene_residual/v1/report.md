# Local gene residual count reconstruction

The prespecified historical endpoint criterion **did not pass**. Mean selected local-model NLL across seeds 0/1 was **0.326651**, against the threshold **0.326242**. The local component **improved** NLL over the matched self-only model in both seeds.

This is an exploratory validation comparison on one previously used held-out donor. The registry uses `validation_confirmation` as a catalog bucket; this experiment is not an independent or confirmatory study. It does not establish independent patient generalization, biological mechanism, or causality.

![Validation count NLL and RMSE across all training epochs and all four runs](learning_curves.png)

## Selected checkpoints and controls

All point errors use the marginal predictive mean. Selection minimizes pooled count NLL; it does not select the best MAE or the best seed.

| Model | Seed | Selected / final epoch | Count NLL ↓ | Raw MAE ↓ | Raw RMSE ↓ | Epochs 29–32 NLL |
|---|---:|---:|---:|---:|---:|---:|
| BAGM NB | 0 | 193 / 196 | 0.347601 | 0.223533 | 1.181556 | — |
| BAGM hurdle | 0 | 164 / 171 | 0.328242 | 0.225992 | 1.168485 | — |
| Local residual | 0 | 30 / 32 | 0.326641 | 0.223224 | 1.101903 | 0.326765 |
| Local residual | 1 | 30 / 32 | 0.326661 | 0.223558 | 1.099667 | 0.326808 |
| Self only | 0 | 30 / 32 | 0.328384 | 0.229236 | 1.132959 | 0.328737 |
| Self only | 1 | 30 / 32 | 0.328370 | 0.229255 | 1.129915 | 0.328678 |
| Train-fitted gene constants, θ=1 | — | — | 0.371358 | 0.338528 | 1.364136 | — |
| Always zero | — | — | — | 0.151251 | — | — |

The constant hurdle baseline fits prevalence and conditional-positive mean on training cells, with dispersion fixed at θ=1. It is not a maximum-likelihood fit of all hurdle parameters. Always-zero MAE is the observed masked-count mean (approximately 0.15125); low MAE alone does not establish useful expression recovery.

## Prespecified decisions

- Local mean NLL ≤ 0.3262417527: **False**.
- Both local seeds improve historical hurdle NLL: **True**.
- Both local seeds have RMSE ≤ 1.1801696537: **True**.
- Local NLL beats matched self-only NLL in both seeds: **True**.

These are practical exploratory thresholds, not tests of statistical significance. Two optimizer seeds do not supply two independent biological replicates.

| Seed | Local − self NLL | Local − self RMSE | SO2-C27 NLL difference | SO2-C28 NLL difference |
|---:|---:|---:|---:|---:|
| 0 | -0.001742 | -0.031056 | -0.001886 | -0.001476 |
| 1 | -0.001709 | -0.030248 | -0.001869 | -0.001413 |

Negative differences favor local context. Both validation cores come from the same donor.

## Additional reconstruction metrics

| Model | Seed | Log1p MAE | Observed-positive marginal MAE | Conditional-positive MAE | Detection BCE | Detection AP |
|---|---:|---:|---:|---:|---:|---:|
| local | 0 | 0.146622 | 1.488801 | 0.846649 | 0.239753 | 0.302070 |
| local | 1 | 0.146902 | 1.489139 | 0.846545 | 0.239786 | 0.301951 |
| self | 0 | 0.149428 | 1.501084 | 0.889012 | 0.240964 | 0.292896 |
| self | 1 | 0.149196 | 1.502187 | 0.891713 | 0.240950 | 0.292853 |

Observed-positive marginal errors score the complete predictor on entries known to be positive. Conditional-positive errors score the count component after conditioning on positivity; these are different estimands. Detection uses the fixed threshold 0.5 for precision/recall/F1, and AP is threshold-free with histogram approximation.

## Runtime and model size

| Model | Seed | Parameters | Runtime (min) | Mean epoch (s) | Optimizer steps | Peak allocated GPU / RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| BAGM NB | 0 | 5,135,088 | 198.2 | — | — | — |
| BAGM hurdle | 0 | 6,160,088 | 169.3 | — | — | — |
| local | 0 | 2,329,432 | 23.9 | 38.4 | 6,688 | 0.84 / 5.79 |
| local | 1 | 2,329,432 | 21.8 | 34.5 | 6,688 | 0.84 / 5.79 |
| self | 0 | 2,329,432 | 20.7 | 34.0 | 6,688 | 0.85 / 5.79 |
| self | 1 | 2,329,432 | 20.2 | 33.3 | 6,688 | 0.85 / 5.80 |

Runtime includes data loading, baseline scoring, training and final replay. Epoch timing includes validation and checkpoint writes. Concurrent jobs and differing epoch budgets prevent a controlled speedup claim against historical BAGM runs.

## Evidence and limits

All four 32-epoch runs passed immutable-bundle verification, matched source/data/split/normalization fingerprints, identical validation and training masks/order, full target coverage, and selected-checkpoint replay. Validation comprises 37,367 cells and 18,708,207 masked entries. Training comprises 208,696 cells; each is visited once per epoch. The configured batch size 1,024 is the maximum: each core uses balanced batches formed with `numpy.array_split`. Initial and selected checkpoints passed the FP32/BF16 likelihood check on one fixed training minibatch with an absolute NLL tolerance of 0.001; this is a limited numerical check, not full-cohort precision equivalence. Biological generalization remains limited to one held-out donor and no sealed test set.

The local model is a two-hidden-layer masked-cell MLP plus four learned same-gene neighborhood weights per gene, with a Bernoulli hurdle/zero-truncated NB2 output. Up to 16 clean, non-self neighbors within 75 μm supply log1p mean counts and positive fractions. Raw physical density is recorded by the loader but is not an input to this model. The self-only ablation removes the local residual.

Historical NB and hurdle checkpoints used different stopping epochs and six updates per epoch. These new runs change architecture and optimization together. A favorable endpoint comparison cannot attribute the gain solely to architecture. Local-versus-self comparisons share masks, target order, optimizer schedule and budget and isolate the added local inputs within the new design.

Likelihood includes its normalization constants. Positive-marginal and conditional-positive errors are distinct in `comparison.csv`; per-core metrics and mask-severity metrics are retained in their own CSVs and `summary.json`. Detection AP/AUROC use 4,096 probability bins, so they are approximations. No gene-level or patient-population uncertainty is inferred from pooled cell counts.

Files: [full metrics](comparison.csv), [per-core metrics](per_core.csv), [mask severity](mask_strata.csv), [learning curves](learning_curves.csv), [machine-readable summary](summary.json), [checksums](manifest.json).

## Immutable sources

- local, seed 0: `r_20261004T072107Z_9d2b00cb_s000_f00_a01_f41786d5`
- local, seed 1: `r_20261004T072107Z_9d2b00cb_s001_f00_a01_1caff983`
- self, seed 0: `r_20261004T072106Z_dd0aa417_s000_f00_a01_e0116c2b`
- self, seed 1: `r_20261004T072107Z_dd0aa417_s001_f00_a01_ea627eca`
- BAGM NB: `r_20260926T104132Z_47d710fa_s000_f00_a01_b967dc7c`
- BAGM hurdle: `r_20260928T021117Z_9ea46874_s000_f00_a01_41b54097`

## Failed attempts

No failed attempts were registered in this campaign at report creation.
