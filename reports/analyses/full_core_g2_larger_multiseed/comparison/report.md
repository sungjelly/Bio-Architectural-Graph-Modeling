# Full-core G2 width comparison

Six checksum-verified, fixed-200-epoch held-in runs were compared (two widths × seeds 0, 1, 2). Percent variance explained is `100 × masked R²`, not classification accuracy or a generalization estimate.

Protocol completion: **COMPLETE**.

## Per-seed evidence

| Variant | Seed | Attempt | Mean Huber | Mean PVE (%) | Total (s) | Summed epochs (s) | Peak VRAM (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| g2_width512_exact_k1000_full_core | 0 | 1 | 0.228873 | 0.146 | 1764.88 | 1536.82 | 7.975 |
| g2_width512_exact_k1000_full_core | 1 | 1 | 0.229089 | -0.513 | 1738.90 | 1524.48 | 7.975 |
| g2_width512_exact_k1000_full_core | 2 | 1 | 0.229145 | -0.404 | 1758.04 | 1532.04 | 7.975 |
| g2_width1024_exact_k1000_full_core | 0 | 2 | 0.230141 | 0.441 | 2859.70 | 2654.50 | 8.356 |
| g2_width1024_exact_k1000_full_core | 1 | 1 | 0.228203 | -1.156 | 2863.85 | 2642.46 | 8.356 |
| g2_width1024_exact_k1000_full_core | 2 | 1 | 0.228193 | -1.172 | 2840.08 | 2625.72 | 8.356 |

## Variant aggregates

| Variant | Huber mean ± SD | PVE mean ± SD (%) | Total mean ± SD (s) | Recorded training mean ± SD (s) | Summed epochs mean ± SD (s) | Peak VRAM mean ± SD (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| g2_width512_exact_k1000_full_core | 0.229036 ± 0.000143 | -0.257 ± 0.353 | 1753.94 ± 13.47 | 1531.94 ± 6.27 | 1531.12 ± 6.22 | 7.975 ± 0.000 |
| g2_width1024_exact_k1000_full_core | 0.228846 ± 0.001122 | -0.629 ± 0.927 | 2854.54 ± 12.69 | 2641.87 ± 14.40 | 2640.89 ± 14.45 | 8.356 ± 0.000 |

### Per-run convergence

| Variant | Seed | Final train loss | Minimum train loss | Last-20 slope |
|---|---:|---:|---:|---:|
| g2_width512_exact_k1000_full_core | 0 | 0.196729 | 0.190121 | -0.000820817 |
| g2_width512_exact_k1000_full_core | 1 | 0.197145 | 0.190155 | -0.000796095 |
| g2_width512_exact_k1000_full_core | 2 | 0.196060 | 0.191311 | -0.000798164 |
| g2_width1024_exact_k1000_full_core | 0 | 0.196567 | 0.190950 | -0.000794942 |
| g2_width1024_exact_k1000_full_core | 1 | 0.195511 | 0.190633 | -0.000832195 |
| g2_width1024_exact_k1000_full_core | 2 | 0.196470 | 0.189845 | -0.000818026 |

| Variant | Final train loss mean ± SD | Minimum loss mean ± SD | Last-20 slope mean ± SD |
|---|---:|---:|---:|
| g2_width512_exact_k1000_full_core | 0.196645 ± 0.000547 | 0.190529 ± 0.000677 | -0.000805025 ± 0.000013715 |
| g2_width1024_exact_k1000_full_core | 0.196183 ± 0.000584 | 0.190476 ± 0.000569 | -0.000815055 ± 0.000018803 |

## Paired seed deltas

| Seed | Baseline − wider Huber | Relative reduction | Wider − baseline PVE (pp) | Wider favored on both |
|---:|---:|---:|---:|---:|
| 0 | -0.001268 | -0.554% | 0.294 | no |
| 1 | 0.000886 | 0.387% | -0.643 | no |
| 2 | 0.000952 | 0.415% | -0.768 | no |

Across seeds, baseline-minus-wider Huber was 0.000190 ± 0.001263 (min -0.001268, max 0.000952); wider-minus-baseline PVE was -0.372 ± 0.581 percentage points (min -0.768, max 0.294).

## H1-width gate

The locked Huber statistic is `(mean baseline Huber across seeds − mean wider Huber across seeds) / mean baseline Huber across seeds`. It is the ratio of aggregate means, not the mean of the three per-seed ratios.

- Observed relative reduction: 0.083%.
- H1-width: **FAIL**. Passing also requires higher aggregate mean PVE and every seed favoring wider G2 on both metrics.

## Resource pilot

- Run `r_20260726T115503Z_1334b662_s000_f00_a01_334b2e85` (attempt 1): 12,687,784 parameters, 8.353 GiB peak VRAM, 27.53s recorded two-epoch training (26.74s summed epoch compute), 138.69s total.
- Projected 200-epoch training: 0.765h; resource gate **PASSED**.

## Failed-attempt ledger

| Run | Variant | Seed | Fold | Attempt | Retry of | Category |
|---|---|---:|---:|---:|---|---|
| `r_20260726T115820Z_04a51acf_s000_f00_a01_9409eae2` | g2_width1024_exact_k1000_full_core | 0 | 0 | 1 | none | interrupted |
- Failure count reconciliation: confirmed at 1.

## Jacobian branch

- Status: **SKIPPED**. At least one wider-G2 seed is not strictly above 95% mean whole-node masked percent variance explained; Jacobians must not be compared. Jacobians were not computed here.
- Comparison artifact status: `complete`.
