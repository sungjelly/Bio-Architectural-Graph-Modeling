# Standalone audit: same-gene robustness multiverse v1

Overall: **PASS** — 42140/42140 checks passed; 0 failed.

This audit did not import the campaign analyzer or `spatial_benchmark`. It recomputed the identifiable numerical content from the four published aggregate files.

## Variant-level reconstructed results

| Variant | near/morph component gain | 95% component bootstrap | cell-weighted gain | near/permutation gain | diag/off | row top-1 | row top-1% |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0 | 1.5392% | [0.7111%, 2.4572%] | 2.1315% | 1.5201% | 3.4125 | 231/932 (0.2479) | 401/932 (0.4303) |
| V1 | 1.5429% | [0.7126%, 2.4612%] | 2.1372% | 1.5063% | 3.4676 | 242/932 (0.2597) | 406/932 (0.4356) |
| V2 | 1.4100% | [0.5834%, 2.3227%] | 2.0512% | 1.4300% | 3.4743 | 239/932 (0.2564) | 408/932 (0.4378) |
| V3 | 0.6513% | [0.1937%, 1.2192%] | 0.9363% | 0.8926% | 2.9581 | 241/932 (0.2586) | 374/932 (0.4013) |
| V4 | 1.3244% | [0.6421%, 2.0475%] | 1.8708% | 1.4385% | 3.4133 | 247/932 (0.2650) | 406/932 (0.4356) |
| V5 | 0.5255% | [0.0667%, 1.0855%] | 0.8438% | 0.8621% | 3.1345 | 249/932 (0.2672) | 368/932 (0.3948) |
| V6 | 0.3048% | [-0.0863%, 0.7101%] | 0.6739% | 0.8903% | 3.4696 | 261/932 (0.2800) | 419/932 (0.4496) |

## Budget attribution

Verdict: `budget_explanation_not_supported`; saturated trajectories: 40/40.
Selected-minus-anchor12 prediction-gain difference: -0.0616% (bootstrap [-0.0946%, -0.0328%]).

## Deterministic label null

| Family | Statistic | Observed | null 95% | upper-tail p |
|---|---|---:|---:|---:|
| full | median_absolute_diagonal | 0.016730663 | [0.0045449783, 0.0052783303] | 9.9990001e-05 |
| full | row_top1_fraction | 0.24785408 | [0, 0.0032188841] | 9.9990001e-05 |
| full | row_top1_percent_fraction | 0.43025751 | [0.0042918455, 0.017167382] | 9.9990001e-05 |
| prevalence_sd_decile_matched | median_absolute_diagonal | 0.016730663 | [0.0048382634, 0.005580782] | 9.9990001e-05 |
| prevalence_sd_decile_matched | row_top1_fraction | 0.24785408 | [0, 0.0032188841] | 9.9990001e-05 |
| prevalence_sd_decile_matched | row_top1_percent_fraction | 0.43025751 | [0.0096566524, 0.025751073] | 9.9990001e-05 |

The reported null 95% range is a reference interval for null draws, not a confidence interval. The label null is secondary and cannot rescue a frozen gate.

## Identifiability boundary of the four-file audit

- Seed/fold Jacobian matrices are not published, so seed-specific Jacobian gates, fold-ratio support, and fold stability cannot be reconstructed from these four files; only their encoded arithmetic and classification logic can be checked.
- Per-slide Jacobian matrices are not published, so the two per-slide summaries cannot be independently reconstructed here; the published slide-equal matrix is checked.
- The matched-null prevalence, target-standard-deviation covariates, and stratum memberships are not published, so its stored draw summaries are checked but its draw vector needs upstream prepared arrays/checkpoints for exact regeneration.
- Validation histories are not published, so each trajectory's 96-to-192 gain is checked for internal saturation logic but not recomputed from validation records.
- Seed/fold row matrices are not published, so the budget row-support counts are checked only for downstream logical consistency.
- The CSV has only the 932 eligible labels, so its index mask hash is verified but the complete frozen 1000-label gene-order hash requires an upstream checkpoint/prepared axis.
- Technical-control truth and component MAE source values require run bundles; this four-file audit checks only their published logical state, schemas, and finiteness.
