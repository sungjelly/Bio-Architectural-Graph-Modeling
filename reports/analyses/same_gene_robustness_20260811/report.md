# Same-gene robustness multiverse v1

This immutable report includes all prespecified V0--V6 core jobs: 140/140 verified, with no missing, failed, or duplicate selected coverage slot; 0 earlier failed/pruned attempts are retained in attempt_history.

V0 design boundary: antecedent-exact observed near/annular inputs and primary cohort; corrected degree-preserving receiver-collision-free permutation control, so not a fully exact antecedent replication.

## Gate classifications

| Gate | V0 consensus | V0 seed classification | Across V0–V5 |
|---|---:|---|---|
| near_vs_morphology_prediction | false | robust_gate_failure (0/5 pass) | robust_across_preprocessing |
| near_vs_permutation_prediction | true | robust_pass (5/5 pass) | preprocessing_sensitive |
| same_name_diagonal_enrichment | true | robust_pass (5/5 pass) | robust_across_preprocessing |
| strict_row_selectivity | false | robust_gate_failure (0/5 pass) | robust_across_preprocessing |
| near_vs_permutation_diagonal | true | robust_pass (5/5 pass) | robust_across_preprocessing |
| fold_stability | true | robust_pass (5/5 pass) | robust_across_preprocessing |
| technical_validity | true | robust_pass (5/5 pass) | robust_across_preprocessing |

## Budget attribution

Frozen verdict: `budget_explanation_not_supported`. Saturated trajectories: 40/40.

## Deterministic gene-label null (secondary)

Computed only after all 140 core jobs passed verification, on the frozen 932-gene V0 selected observed-near consensus matrix.

| Family | Statistic | Observed | Upper-tail p |
|---|---|---:|---:|
| full | median_absolute_diagonal | 0.016730663 | 9.9990001e-05 |
| full | row_top1_fraction | 0.24785408 | 9.9990001e-05 |
| full | row_top1_percent_fraction | 0.43025751 | 9.9990001e-05 |
| prevalence_sd_decile_matched | median_absolute_diagonal | 0.016730663 | 9.9990001e-05 |
| prevalence_sd_decile_matched | row_top1_fraction | 0.24785408 | 9.9990001e-05 |
| prevalence_sd_decile_matched | row_top1_percent_fraction | 0.43025751 | 9.9990001e-05 |

This secondary null cannot change or rescue a frozen strict gate.

## Prepared permutation audit

| Variant | Fixed source states | Near cross-FOV edges | Annular cross-FOV edges |
|---|---:|---:|---:|
| V0 | 0 | 0 | 0 |
| V1 | 0 | 165821 | 371933 |
| V2 | 1 | 0 | 0 |
| V3 | 0 | 0 | 0 |
| V4 | 0 | 0 | 0 |
| V5 | 1 | 152854 | 349395 |
| V6 | 0 | 0 | 0 |

## Interpretation boundary

- post_hoc robustness analysis, not independent replication
- conditional on 27 geometry components from two observed slides
- not patient-level inference, mechanism, causality, or cell-cell communication
- prediction losses average all 1000 outputs; Jacobian name-alignment uses only the frozen common 932-gene axis
- V0 observed arms and primary cohort are antecedent-exact, but its corrected permutation control is not the antecedent permutation
- V6 is mechanistic-secondary and cannot rescue a mixed V0--V5 classification
- the deterministic V0 gene-label null is secondary and cannot change or rescue a frozen strict gate
- no best-variant, best-seed, or favorable-attempt selection is permitted

All ratios and classifications in this document are machine-derived from the float64 aggregate NPZ and checked across JSON/CSV/NPZ before publication.
