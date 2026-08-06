# MyJJu GeneMAE versus current BAGM

## Outcome

The reproduced GeneMAE ensemble met the frozen descriptive common-task comparison gate against the current BAGM GAT ensemble on held-in 20% partial-gene reconstruction across ten adjacent-normal core aliases. This supports only a comparison of the two end-to-end trained systems, not an isolated architecture effect. Non-primary frozen gates failed: graph_use. The graph-null result does not support graph-specific predictive gain.

This is exploratory held-in partial-gene reconstruction on ten pathology-confirmed adjacent-normal core aliases. It is not a generalization, interaction, mechanism, or causal result.

## Common 20% task

| Model | Graph | Huber ↓ | MAE ↓ | Pooled Pearson ↑ | Mean gene Pearson ↑ |
|---|---|---:|---:|---:|---:|
| myjju-genemae | observed | 0.3482 | 0.5471 | 0.3129 | 0.1845 |
| myjju-genemae | node_label_permuted | 0.3520 | 0.5521 | 0.2974 | 0.1795 |
| current-bagm-pooled-gat | observed | 1.2864 | 1.4815 | 0.1992 | 0.1270 |
| current-bagm-matched-self | observed | 1.1286 | 1.3034 | 0.2019 | 0.1206 |
| per_core_gene_mean | observed | 0.3901 | 0.7009 | 0.2227 | -0.0000 |
| all_zero | observed | 0.3588 | 0.4066 | undefined | undefined |

BAGM probabilities and continuous heads were combined before decoding; its decoded counts were then transformed with the true full-cell library size. The latter makes the common scale oracle and descriptive.
This is an end-to-end trained-system comparison, not a controlled architecture ablation. GeneMAE and BAGM differ in training objective, masking, normalization, permitted covariates, and graph construction.
The per-core gene-mean control is a target-derived oracle: it uses all fit cells, including the hidden target entries.

## Frozen gates

| Gate | Result | Observed | Frozen threshold |
|---|---|---|---|
| primary_comparison | PASS | mean_huber_relative_improvement=0.7293, genemae_favoring_cores=10, genemae_favoring_seed_pairs=7, masked_mae_nonworse=pass, mean_gene_pearson_nonworse=pass | mean_huber_relative_improvement_minimum=0.0200, minimum_genemae_favoring_cores=8, minimum_genemae_favoring_seed_pairs=5, masked_mae_nonworse=pass, mean_gene_pearson_nonworse=pass |
| baseline | PASS | mean_huber_relative_improvement=0.1073, genemae_favoring_cores=10 | mean_huber_relative_improvement_minimum=0.0200, minimum_genemae_favoring_cores=8 |
| graph_use | FAIL | mean_huber_relative_improvement=0.0109, unpermuted_favoring_cores=10 | mean_huber_relative_improvement_minimum=0.0200, minimum_unpermuted_favoring_cores=8 |

## Descriptive across-core effects

Positive relative Huber improvement means the candidate has strictly lower loss. These are descriptive distributions only; no population confidence interval or bootstrap is reported because all cores are held in and coupled by shared fitted weights.

| Contrast | Cores | Median | Minimum | Maximum | Favoring cores |
|---|---:|---:|---:|---:|---:|
| genemae_vs_bagm_gat | 10 | 0.7330 | 0.6835 | 0.7627 | 10 |
| genemae_vs_target_derived_per_core_gene_mean | 10 | 0.1081 | 0.0926 | 0.1186 | 10 |
| genemae_observed_vs_node_label_permuted | 10 | 0.0089 | 0.0058 | 0.0193 | 10 |

## Current BAGM whole-node result (separate context)

The immutable current BAGM campaign reported whole-node hybrid loss 0.431869 for the GAT ensemble versus 0.445254 for matched self: 3.01% graph gain, favoring GAT in 10/10 cores and 7/7 seed pairs.
That graph gate passed, but the current campaign outcome was negative because its pooled-data and representation gates failed. Whole-node hybrid loss is a different estimand and is not ranked against GeneMAE partial-gene metrics.

## Source-native 50% GeneMAE task

- Observed-graph Huber: 0.3328
- Permuted-graph Huber: 0.3366
- These 50% results are not ranked against BAGM.

## Historical source report (non-comparable context)

- SO1 sb50: Pearson 0.3579, Spearman 0.2689, R2 0.0914, MSE 1.4733, mean per-gene Pearson 0.1943, shuffled-feature Pearson 0.3427.
- SO2 sb50: Pearson 0.3081, Spearman 0.2271, R2 0.0552, MSE 1.5256, mean per-gene Pearson 0.1588, shuffled-feature Pearson 0.2934.
No historical weights are available. These source-reported values used held-out data for checkpoint selection/scoring and are checksum-bound context only, not an independent or common-task comparison.
The source constructor was minimally repaired by assigning `self.self_hidden` from its constructor argument, then seven models were retrained from scratch. This report does not evaluate the missing original weights.

## Variability, failures, and resources

All 7 GeneMAE, 7 BAGM GAT, and 7 BAGM matched-self final checkpoints were audited. Registered failed attempts retained in the report: 2.
Complete per-core, per-seed, convergence, runtime, peak VRAM, peak host-memory, mask, checksum, and failure tables accompany this report.

## Limitations

- all ten cores and all cells were used for fitting
- held-in transductive reconstruction is not generalization
- partial-gene masking can be dominated by same-cell co-expression
- full-cell CP10k normalization uses hidden entries in its denominator
- BAGM common-scale conversion uses the true full-cell library size
- the per-core gene-mean control is target-derived and uses all fit cells, including hidden target entries
- this compares end-to-end trained systems that differ in objective, masking, normalization, covariates, and graph; it does not isolate an architecture effect
- adjacent-normal tissue is not true Normal
- cores coupled by shared fitted weights are not independent model fits
- no population confidence interval is reported because the held-in coupled cores do not support population inference
- model seeds and mask replicates are not biological replicates
- the fixed graph null tests graph use but not a biological mechanism
- the targeted panel limits biological interpretation

## Maximum claim

The reproduced GeneMAE ensemble met the frozen descriptive common-task comparison gate against the current BAGM GAT ensemble on held-in 20% partial-gene reconstruction across ten adjacent-normal core aliases. This supports only a comparison of the two end-to-end trained systems, not an isolated architecture effect. Non-primary frozen gates failed: graph_use. The graph-null result does not support graph-specific predictive gain.
