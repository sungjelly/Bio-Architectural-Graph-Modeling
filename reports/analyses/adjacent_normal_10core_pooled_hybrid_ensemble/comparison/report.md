# Pooled ten-core hybrid-count ensemble

Status: complete; outcome: **negative under at least one frozen gate**.

This exploratory, transductive study fitted each ensemble member as one shared model over all 117,386 cells in ten pathology-confirmed adjacent-normal cores. These samples are not true Normal. Each core remained a separate full-core graph batch; no cross-core edges were created.

## Architecture and objective

Both arms used the hybrid raw-count encoder with gene-specific count-token projections, exact within-bin standardized `log1p(count)`, and 22 permitted morphology/imaging covariates. Tokens were fixed as `0`, `1`, `2`, `3`, `4–7`, `8–15`, `16–31`, and `32+`; token 8 was input-only MASK. The GAT used two 512-wide four-head exact receiver-partitioned edge-conditioned layers on the mutual k=1,000 graph with 17 measured geometry features. The self arm had the same 11,674,880 trainable parameters and no graph, neighbors, coordinates, or edge attributes.

Every member ran 200 global epochs (2,000 optimizer steps) with AdamW, learning rate `3e-4`, weight decay `1e-4`, clipping `1.0`, fixed masks, no validation selection, no early stopping, no neighbor sampling, and no edge dropout. The loss was the equal mean of balanced detection BCE, balanced six-threshold ordinal BCE, and positive standardized-log1p Huber.

The seven-member ensembles averaged detection and ordinal probabilities and continuous standardized predictions before decoding. Member logits were streamed and not retained.

## Frozen descriptive gates

| Gate | Passed |
|---|---:|
| `pooled_data_gate` | false |
| `graph_gate` | true |
| `representation_gate` | false |
| `ensemble_gate` | true |

Failed gates are a valid negative result and were not rescued by overall exact accuracy: `pooled_data_gate`, `representation_gate`.

Pooled seed-0 versus prior independent seed-0 GAT:

| Metric | Mean relative gain | Favoring aliases | Passed |
|---|---:|---:|---:|
| `detection_bce` | 2.32% | 10/10 | true |
| `positive_ordinal_mae` | 3.76% | 7/10 | false |
| `reconstructed_count_log1p_mae` | -8.57% | 4/10 | false |

Representation checks against the per-gene references:

| Metric | Mean relative gain | Favoring aliases | Passed |
|---|---:|---:|---:|
| `positive_ordinal_mae` | -61.16% | 0/10 | false |
| `positive_continuous_huber` | 26.16% | 10/10 | true |

Detection balanced accuracy was 0.6723 for the GAT ensemble, 0.5110 for the per-core reference, and 0.5090 for the equal-core pooled reference.

Prediction-level ensemble versus the mean individual GAT member:

| Metric | Ensemble | Mean member | Passed non-worsening |
|---|---:|---:|---:|
| `hybrid_loss` | 0.4319 | 0.4354 | true |
| `positive_ordinal_mae` | 0.9331 | 0.9420 | true |
| `positive_continuous_huber` | 0.3686 | 0.3730 | true |

## Whole-node graph comparison

The equal-core mean relative GAT ensemble hybrid-loss improvement was 3.01%; 10/10 aliases and 7/7 paired model seeds favored the GAT. These are descriptive outcomes, not independent core-level inference, because all aliases share fitted weights.

Positive ordinal MAE non-worsening: true; positive continuous Huber non-worsening: true.

| Alias | GAT loss | Self loss | GAT gain | GAT ordinal | Self ordinal | GAT cont. | Self cont. |
|---|---:|---:|---:|---:|---:|---:|---:|
| ANC-01 | 0.4186 | 0.4295 | 2.54% | 0.8646 | 0.9253 | 0.3475 | 0.3564 |
| ANC-02 | 0.4338 | 0.4445 | 2.39% | 0.9801 | 0.9782 | 0.3668 | 0.3721 |
| ANC-03 | 0.4395 | 0.4516 | 2.69% | 0.9442 | 0.9360 | 0.3705 | 0.3754 |
| ANC-04 | 0.4388 | 0.4491 | 2.28% | 0.9889 | 0.9742 | 0.3748 | 0.3811 |
| ANC-05 | 0.4405 | 0.4511 | 2.34% | 1.0008 | 0.9190 | 0.3716 | 0.3766 |
| ANC-06 | 0.4253 | 0.4358 | 2.41% | 0.9394 | 0.9714 | 0.3706 | 0.3750 |
| ANC-07 | 0.4176 | 0.4430 | 5.73% | 0.8615 | 0.9302 | 0.3540 | 0.3794 |
| ANC-08 | 0.4204 | 0.4357 | 3.52% | 0.8233 | 0.8939 | 0.3450 | 0.3577 |
| ANC-09 | 0.4276 | 0.4404 | 2.90% | 0.8835 | 0.9362 | 0.3586 | 0.3660 |
| ANC-10 | 0.4564 | 0.4718 | 3.25% | 1.0449 | 0.9816 | 0.4262 | 0.4351 |

Paired seeds are technical replicates:

| Seed | GAT loss | Self loss | GAT gain |
|---:|---:|---:|---:|
| 0 | 0.4350 | 0.4461 | 2.49% |
| 1 | 0.4378 | 0.4447 | 1.56% |
| 2 | 0.4342 | 0.4483 | 3.13% |
| 3 | 0.4362 | 0.4465 | 2.31% |
| 4 | 0.4354 | 0.4451 | 2.19% |
| 5 | 0.4348 | 0.4490 | 3.15% |
| 6 | 0.4344 | 0.4449 | 2.35% |

## Descriptive comparison with the previous tokenizer

The previous experiment used one legacy true-Normal core, three seeds, and categorical `0/1/2/>=3` targets. The current values use ten adjacent-normal cores, collapse eight-state ensemble predictions after decoding, and average ten coupled core outcomes. Differences therefore mix tissue context, data, pooling, objective, architecture, and ensemble effects; they are descriptive, not an attribution to tokenizer choice.

| Source | Exact | Balanced | Positive exact | All-zero exact |
|---|---:|---:|---:|---:|
| current_ten_core_pooled_hybrid_gat_ensemble / pooled-hybrid-gat-k1000 | 62.32% | 42.64% | 30.97% | 90.42% |
| prior_one_core_categorical_gat / g2_tokenized_width512_exact_k1000_full_core | 91.78% | 28.17% | 1.53% | 91.70% |
| prior_one_core_categorical_gat / g2_tokenized_width1024_exact_k1000_full_core | 91.77% | 28.11% | 1.50% | 91.70% |

## Runtime and verification

All 14 prespecified production members completed, every bundle and single final checkpoint verified, and all exact graph, data, split, mask, materialization, pilot-gate, and checkpoint identities reconciled. There were 2 failed or stale attempts before selected completions: 2 pilot and 0 production; none were hidden. The two attempt-1 pilot failures were operational `invalid_configuration` events before model construction or GPU work: long-lived workers retained a stale protocol validator. Safely reloaded workers and immutable attempt-2 retries completed.

| Arm | Seed | Attempt | GPU | Device | Runtime h | Peak VRAM GiB |
|---|---:|---:|---:|---|---:|---:|
| pooled-hybrid-gat-k1000 | 0 | 1 | 0 | NVIDIA GeForce RTX 3090 | 2.31 | 7.34 |
| pooled-hybrid-gat-k1000 | 1 | 1 | 1 | NVIDIA GeForce RTX 3090 | 2.29 | 7.34 |
| pooled-hybrid-gat-k1000 | 2 | 1 | 2 | NVIDIA GeForce RTX 3090 | 2.29 | 7.34 |
| pooled-hybrid-gat-k1000 | 3 | 1 | 3 | NVIDIA GeForce RTX 3090 | 2.29 | 7.34 |
| pooled-hybrid-gat-k1000 | 4 | 1 | 5 | NVIDIA GeForce RTX 3090 | 2.29 | 7.34 |
| pooled-hybrid-gat-k1000 | 5 | 1 | 6 | NVIDIA GeForce RTX 3090 | 2.27 | 7.34 |
| pooled-hybrid-gat-k1000 | 6 | 1 | 7 | NVIDIA GeForce RTX 3090 | 2.28 | 7.34 |
| pooled-hybrid-matched-self | 0 | 1 | 0 | NVIDIA GeForce RTX 3090 | 0.27 | 2.22 |
| pooled-hybrid-matched-self | 1 | 1 | 1 | NVIDIA GeForce RTX 3090 | 0.27 | 2.22 |
| pooled-hybrid-matched-self | 2 | 1 | 2 | NVIDIA GeForce RTX 3090 | 0.27 | 2.22 |
| pooled-hybrid-matched-self | 3 | 1 | 3 | NVIDIA GeForce RTX 3090 | 0.28 | 2.22 |
| pooled-hybrid-matched-self | 4 | 1 | 5 | NVIDIA GeForce RTX 3090 | 0.27 | 2.22 |
| pooled-hybrid-matched-self | 5 | 1 | 6 | NVIDIA GeForce RTX 3090 | 0.27 | 2.22 |
| pooled-hybrid-matched-self | 6 | 1 | 7 | NVIDIA GeForce RTX 3090 | 0.27 | 2.22 |

## Limitations and maximum conclusion

- This is held-in masked-expression reconstruction with transductive preprocessing and references; it is not held-out-core or patient-held-out generalization.
- Cells, mask repeats, and model seeds are not biological replicates. Shared fitted weights couple the ten core outcomes, so no formal core-level inference is reported.
- Whole-node prediction may infer cell state and return learned means. Partial-gene masking can be dominated by same-cell co-expression.
- Mutual k=1,000 is broad regional context, not direct cell-cell communication. The design does not establish a biological mechanism or causality.
- Adjacent-normal tissue is not true Normal, and the targeted panel limits lineage interpretation.
- Pooled seed-0 versus prior independent fits also changes cross-core weight sharing, optimizer-step budget, and expression normalization. It cannot isolate a causal effect of larger training data or sharing.
- The prior categorical experiment used a legacy true-Normal core, whereas the current ten cores are adjacent-normal; its collapsed-state comparison is descriptive only.

Maximum defensible conclusion: The complete exploratory held-in experiment was technically valid, but the frozen pooled_data_gate, representation_gate failed; no graph, representation, or ensemble success is claimed beyond the specific gates that passed.

Complete machine-readable tables and their checksums are listed in `manifest.json`.
