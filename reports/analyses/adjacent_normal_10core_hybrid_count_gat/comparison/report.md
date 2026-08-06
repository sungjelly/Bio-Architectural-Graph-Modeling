# Ten-core hybrid raw-count GAT comparison

Status: **complete**. This is an exploratory, held-in masked-expression study of ten pathology-confirmed adjacent-normal cores. Three fixed masks are averaged within mode/core; the ten core aliases receive equal weight.

## Frozen model and objective

Counts use eight output states: `0`, `1`, `2`, `3`, `4–7`, `8–15`, `16–31`, and `>=32`; token 8 is input-only `MASK`. The encoder combines gene-specific token projections, exact per-gene standardized `log1p(raw count)`, and the permitted morphology/imaging covariates. Masking replaces the token and sets the continuous input to standardized zero.

The graph arm is an exact receiver-partitioned edge-conditioned GATv2 (width 512, four heads, two graph layers, FFN 512, decoder 512) over the exact mutual `k=1000` graph and 17 geometry attributes. The matched-self arm has the same encoder, decoder, and trainable parameter count but receives no graph, neighbor, coordinate, or edge input.

The fixed objective is the equal mean of balanced detection BCE, balanced six-threshold positive ordinal BCE, and positive-only standardized-`log1p` Huber loss (`delta=1.0`). Whole-node masking is the primary estimand.

## Evidence coverage

Registry slots: 20/20 uniquely completed; pilot receipt verified: yes; analysis ready: yes.
Execution audit: 0 failed/stale run or queue records; 0 duplicate completed slots; 0 invalid production attempts. Attempt-2 recoveries remain eligible when they are the sole valid completed run for a slot; earlier failures remain visible in the audit tables.

## Frozen gate decisions

| Gate | Assessable | Decision |
|---|:---:|:---:|
| Graph gain | yes | FAIL |
| Representation | yes | FAIL |

### Graph gate

| Criterion | Observed | Threshold | Pass |
|---|---:|---:|:---:|
| all twenty production runs complete and verified | yes | yes | yes |
| mean paired relative hybrid loss improvement at least 2 percent | -0.0000 | 0.0200 | no |
| at least 8 of 10 cores favor gat | 5 | 8 | no |
| all three holm adjusted component tests reject | 0 | 3 | no |
| mean positive ordinal mae not worse | 0.0607 | 0.0000 | no |
| mean positive continuous huber not worse | -0.0041 | 0.0000 | yes |

### Representation gate

| Criterion | Observed | Threshold | Pass |
|---|---:|---:|:---:|
| mean positive ordinal mae relative improvement at least 2 percent | -0.6920 | 0.0200 | no |
| mean positive continuous huber relative improvement at least 2 percent | 0.2457 | 0.0200 | yes |
| at least 8 of 10 cores favor gat on positive ordinal mae | 0 | 8 | no |
| at least 8 of 10 cores favor gat on positive continuous huber | 10 | 8 | yes |
| mean detection balanced accuracy exceeds reference | 0.1431 | 0.0000 | yes |

## Whole-node paired core results

| Alias | GAT loss | Self loss | Relative gain (%) | GAT favored |
|---|---:|---:|---:|:---:|
| ANC-01 | 0.421910 | 0.423290 | 0.3260 | yes |
| ANC-02 | 0.471293 | 0.471369 | 0.0162 | yes |
| ANC-03 | 0.445911 | 0.451495 | 1.2368 | yes |
| ANC-04 | 0.463698 | 0.465170 | 0.3166 | yes |
| ANC-05 | 0.452506 | 0.449099 | -0.7587 | no |
| ANC-06 | 0.451333 | 0.463496 | 2.6243 | yes |
| ANC-07 | 0.425817 | 0.419612 | -1.4787 | no |
| ANC-08 | 0.422982 | 0.421210 | -0.4207 | no |
| ANC-09 | 0.457034 | 0.449610 | -1.6512 | no |
| ANC-10 | 0.463993 | 0.462919 | -0.2320 | no |

## Whole-node representation results

| Alias | Ordinal MAE gain vs reference (%) | Continuous Huber gain vs reference (%) | Detection balanced-accuracy difference |
|---|---:|---:|---:|
| ANC-01 | -67.0110 | 22.7117 | 0.153447 |
| ANC-02 | -44.4748 | 24.1921 | 0.118718 |
| ANC-03 | -63.1342 | 27.2414 | 0.139259 |
| ANC-04 | -69.7446 | 27.3071 | 0.144653 |
| ANC-05 | -91.1052 | 29.1049 | 0.137222 |
| ANC-06 | -61.5661 | 24.5865 | 0.165896 |
| ANC-07 | -63.5144 | 20.9227 | 0.146499 |
| ANC-08 | -58.8514 | 21.5491 | 0.148170 |
| ANC-09 | -101.8081 | 21.9223 | 0.126781 |
| ANC-10 | -70.7599 | 26.1998 | 0.150282 |

## Exact paired component inference

Raw differences are matched-self minus GAT. Each test enumerates all `2^10` sign flips; Holm correction covers exactly the three frozen loss components.

| Component | Mean raw difference | Raw p | Holm p | Reject |
|---|---:|---:|---:|:---:|
| detection_bce | -0.002951 | 0.846680 | 1.000000 | no |
| ordinal_bce | -0.000884 | 0.661133 | 1.000000 | no |
| positive_continuous_huber | 0.004073 | 0.035156 | 0.105469 | no |

## Descriptive prior categorical comparison

The prior `0/1/2/>=3` study fit three model seeds on one true-Normal core, whereas the current study uses ten pathology-confirmed adjacent-normal cores. It is shown only for descriptive context; the observations are not exchangeable.
The current eight-state hybrid outputs are collapsed to the common four states only for this display. The prior model directly predicted four categorical states and lacked the current exact-value continuous channel and hurdle/ordinal objective, so numerical differences cannot be attributed to tokenization alone.

| Source | Variant | Exact (%) | Balanced (%) | Positive exact (%) | All-zero exact (%) |
|---|---|---:|---:|---:|---:|
| current_ten_core_hybrid_gat | hybrid-gat-k1000 | 59.0687 | 40.9428 | 29.1359 | 90.4214 |
| prior_one_core_categorical | prior-categorical-gat-width1024 | 91.7684 | 28.1051 | 1.5024 | 91.6968 |
| prior_one_core_categorical | prior-categorical-gat-width512 | 91.7799 | 28.1677 | 1.5314 | 91.6968 |

## Runtime and convergence

| Alias | Arm | Parameters | Runtime (h) | Peak VRAM (GiB) | Final train loss | Last-20 slope | Complete/finite |
|---|---|---:|---:|---:|---:|---:|:---:|
| ANC-01 | hybrid-gat-k1000 | 11674880 | 0.277 | 7.341 | 0.429915 | -0.00014191 | yes |
| ANC-01 | hybrid-matched-self | 11674880 | 0.030 | 2.213 | 0.430688 | 0.00000856 | yes |
| ANC-02 | hybrid-gat-k1000 | 11674880 | 0.181 | 6.585 | 0.469526 | -0.00048723 | yes |
| ANC-02 | hybrid-matched-self | 11674880 | 0.020 | 1.538 | 0.458854 | -0.00027310 | yes |
| ANC-03 | hybrid-gat-k1000 | 11674880 | 0.277 | 7.324 | 0.431716 | -0.00037532 | yes |
| ANC-03 | hybrid-matched-self | 11674880 | 0.031 | 2.225 | 0.436350 | -0.00011901 | yes |
| ANC-04 | hybrid-gat-k1000 | 11674880 | 0.145 | 6.170 | 0.467472 | -0.00084783 | yes |
| ANC-04 | hybrid-matched-self | 11674880 | 0.018 | 1.263 | 0.465428 | -0.00076503 | yes |
| ANC-05 | hybrid-gat-k1000 | 11674880 | 0.138 | 6.197 | 0.454359 | -0.00040078 | yes |
| ANC-05 | hybrid-matched-self | 11674880 | 0.016 | 1.210 | 0.447759 | -0.00010269 | yes |
| ANC-06 | hybrid-gat-k1000 | 11674880 | 0.251 | 7.119 | 0.446322 | -0.00026018 | yes |
| ANC-06 | hybrid-matched-self | 11674880 | 0.026 | 1.999 | 0.444202 | -0.00068992 | yes |
| ANC-07 | hybrid-gat-k1000 | 11674880 | 0.226 | 6.931 | 0.431823 | -0.00019969 | yes |
| ANC-07 | hybrid-matched-self | 11674880 | 0.026 | 1.866 | 0.427737 | -0.00024422 | yes |
| ANC-08 | hybrid-gat-k1000 | 11674880 | 0.274 | 7.314 | 0.432954 | 0.00010289 | yes |
| ANC-08 | hybrid-matched-self | 11674880 | 0.030 | 2.194 | 0.427993 | -0.00005792 | yes |
| ANC-09 | hybrid-gat-k1000 | 11674880 | 0.214 | 6.803 | 0.455026 | -0.00037383 | yes |
| ANC-09 | hybrid-matched-self | 11674880 | 0.025 | 1.805 | 0.457244 | 0.00001453 | yes |
| ANC-10 | hybrid-gat-k1000 | 11674880 | 0.211 | 6.843 | 0.449376 | -0.00054603 | yes |
| ANC-10 | hybrid-matched-self | 11674880 | 0.024 | 1.770 | 0.452833 | -0.00018675 | yes |

## Negative evidence

- The prior one-core categorical width-increase gate failed; this is nonexchangeable descriptive context only.
- The frozen graph gate failed.
- The frozen representation gate failed.

## Limitations

- The ten opaque aliases denote pathology-confirmed adjacent-normal tissue, not true Normal tissue.
- All fitting, preprocessing, and reference estimation occur within each core; this is not patient-held-out generalization.
- The k=1000 graph represents broad regional context and cannot identify direct cell-cell communication.
- The prior categorical experiment fit three model seeds on one true-Normal core, whereas this study uses ten pathology-confirmed adjacent-normal cores; the designs are not exchangeable.
- Predictive reconstruction and graph gain do not establish biological mechanism or causality.

Maximum defensible conclusion: held-in masked-expression representation capacity and possible broad-context graph gain across ten adjacent-normal cores.
