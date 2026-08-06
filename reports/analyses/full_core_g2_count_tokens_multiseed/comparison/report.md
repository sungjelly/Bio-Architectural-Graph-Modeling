# Full-core tokenized G2 width comparison

Six checksum-verified fixed-200-epoch runs were compared: two widths by seeds 0, 1, and 2. These percentages are held-in categorical reconstruction metrics, not validation/test accuracy.

Comparison status: `complete`.

## Whole-node results

| Variant | Exact % | Balanced % | Nonzero % | Cross-entropy | Parameters |
|---|---:|---:|---:|---:|---:|
| `g2_tokenized_width512_exact_k1000_full_core` | 91.7799 ± 0.0045 | 28.1677 ± 0.1606 | 1.5314 ± 0.0777 | 0.326539 | 7,062,880 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 91.7684 ± 0.0220 | 28.1051 ± 0.6304 | 1.5024 ± 0.3055 | 0.332731 | 18,834,784 |

### Per-seed whole-node means

| Variant | Seed | Exact % | Balanced % | Nonzero % | Cross-entropy | Train s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `g2_tokenized_width512_exact_k1000_full_core` | 0 | 91.7848 | 28.3532 | 1.6210 | 0.326392 | 1541.23 | 8.284 |
| `g2_tokenized_width512_exact_k1000_full_core` | 1 | 91.7786 | 28.0762 | 1.4869 | 0.326657 | 1530.27 | 8.284 |
| `g2_tokenized_width512_exact_k1000_full_core` | 2 | 91.7762 | 28.0738 | 1.4861 | 0.326569 | 1535.63 | 8.284 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 0 | 91.7887 | 28.7849 | 1.8308 | 0.326030 | 2651.15 | 8.711 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 1 | 91.7714 | 27.5398 | 1.2267 | 0.334866 | 2645.84 | 8.711 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 2 | 91.7450 | 27.9906 | 1.4497 | 0.337298 | 2631.45 | 8.711 |

### Per-mask whole-node results

| Variant | Seed | Mask | Exact % | Balanced % | Nonzero % | Cross-entropy |
|---|---:|---:|---:|---:|---:|---:|
| `g2_tokenized_width512_exact_k1000_full_core` | 0 | 0 | 91.8348 | 28.3501 | 1.6149 | 0.324413 |
| `g2_tokenized_width512_exact_k1000_full_core` | 0 | 1 | 91.7522 | 28.3984 | 1.6355 | 0.327940 |
| `g2_tokenized_width512_exact_k1000_full_core` | 0 | 2 | 91.7673 | 28.3111 | 1.6126 | 0.326824 |
| `g2_tokenized_width512_exact_k1000_full_core` | 1 | 0 | 91.8290 | 28.0747 | 1.4820 | 0.324696 |
| `g2_tokenized_width512_exact_k1000_full_core` | 1 | 1 | 91.7464 | 28.1295 | 1.5060 | 0.328163 |
| `g2_tokenized_width512_exact_k1000_full_core` | 1 | 2 | 91.7604 | 28.0243 | 1.4728 | 0.327112 |
| `g2_tokenized_width512_exact_k1000_full_core` | 2 | 0 | 91.8271 | 28.0824 | 1.4860 | 0.324731 |
| `g2_tokenized_width512_exact_k1000_full_core` | 2 | 1 | 91.7444 | 28.1249 | 1.5040 | 0.328008 |
| `g2_tokenized_width512_exact_k1000_full_core` | 2 | 2 | 91.7570 | 28.0143 | 1.4684 | 0.326968 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 0 | 0 | 91.8378 | 28.7742 | 1.8204 | 0.324095 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 0 | 1 | 91.7566 | 28.8209 | 1.8398 | 0.327569 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 0 | 2 | 91.7718 | 28.7598 | 1.8321 | 0.326427 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 1 | 0 | 91.8231 | 27.5489 | 1.2276 | 0.332671 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 1 | 1 | 91.7383 | 27.5733 | 1.2374 | 0.336485 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 1 | 2 | 91.7529 | 27.4970 | 1.2151 | 0.335442 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 2 | 0 | 91.7955 | 27.9986 | 1.4495 | 0.335398 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 2 | 1 | 91.7153 | 28.0445 | 1.4689 | 0.338395 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 2 | 2 | 91.7242 | 27.9286 | 1.4307 | 0.338101 |

## Baseline context

The all-zero exact baseline is 91.6968%; the per-gene modal exact baseline is 91.7253%; its balanced accuracy is 28.3004%. The modal baseline is an all-fit transductive reference. Exact accuracy must therefore be read with balanced and nonzero accuracy.

| Reference metric | Mean % | Seed SD |
|---|---:|---:|
| Uniform-class exact | 25.0000 | 0.0000 |
| Empirical-frequency random exact | 84.3779 | 0.0000 |
| All-zero exact | 91.6968 | 0.0000 |
| All-zero balanced | 25.0000 | 0.0000 |
| Per-gene modal exact | 91.7253 | 0.0000 |
| Per-gene modal balanced | 28.3004 | 0.0000 |
| Per-gene modal nonzero | 1.6031 | 0.0000 |

## Token recall and support

Recall is averaged across the three fixed masks and then across seeds. Support is the mean per-run sum across those masks.

| Variant | Token | Mean recall % | Seed SD | Mean support |
|---|---:|---:|---:|---:|
| `g2_tokenized_width512_exact_k1000_full_core` | 0 (`zero`) | 99.9519 | 0.0027 | 6670945 |
| `g2_tokenized_width512_exact_k1000_full_core` | 1 (`one`) | 0.0000 | 0.0000 | 334857 |
| `g2_tokenized_width512_exact_k1000_full_core` | 2 (`two`) | 0.0000 | 0.0000 | 196466 |
| `g2_tokenized_width512_exact_k1000_full_core` | 3 (`three_or_more`) | 12.7191 | 0.6448 | 72732 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 0 (`zero`) | 99.9420 | 0.0254 | 6670945 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 1 (`one`) | 0.0000 | 0.0000 | 334857 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 2 (`two`) | 0.0000 | 0.0000 | 196466 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 3 (`three_or_more`) | 12.4784 | 2.5367 | 72732 |

## Resources and convergence

| Variant | Total runtime s | Training s | Peak GiB | Final loss | Minimum loss | Last-20 slope |
|---|---:|---:|---:|---:|---:|---:|
| `g2_tokenized_width512_exact_k1000_full_core` | 1639.91 | 1535.71 | 8.284 | 0.290680 | 0.289667 | -0.00104659 |
| `g2_tokenized_width1024_exact_k1000_full_core` | 2755.99 | 2642.81 | 8.711 | 0.294938 | 0.293370 | -0.00110388 |

## Locked gates

- H1-width: **FAIL**; mean exact gain -0.0115 pp, mean balanced gain -0.0626 pp.
- Resource pilot: **PASSED**; 8.703 GiB peak and 0.768 h projected.
- Relaxed Jacobian: **SKIPPED**. At least one wider-model seed is not strictly above 95% exact accuracy, so the relaxed categorical Jacobian is skipped.

## Limits

All cells, graph construction, preprocessing summaries, and baselines come from one held-in core. Evaluation masks use a separate seed derivation but are not an entry-wise holdout and may overlap epoch training masks. Seeds and masks are technical repeats, not independent biological replication or evidence of patient generalization.
The exact comparison contract is the resolved configuration saved inside each launched bundle; the checked source experiment definitions compose to those snapshots. The legacy mask-namespace label denotes separate seed derivation, not entry-wise holdout.
Registry reconciliation: **PASSED**. The campaign contains exactly the supplied pilot plus six science runs; all seven jobs are completed with no failure category or last error.
