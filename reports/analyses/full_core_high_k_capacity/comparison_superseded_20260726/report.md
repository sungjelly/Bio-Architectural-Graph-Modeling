# Full-core G2 capacity comparison

## Facts

- G2 run: `r_20260725T083929Z_8364b333_s000_f00_a01_f5c47601`.
- G2-matched cell-autonomous run: `r_20260725T090752Z_e05d042b_s000_f00_a01_4ff98df7`.
- Both checksum-verified runs completed 200 fixed epochs with 3,987,880 trainable parameters.
- Mean held-in whole-node masked Huber: G2 0.228873; matched self 0.229975.
- Mean self-minus-G2 difference: 0.001102; relative G2 gain 0.479%.

| Locked criterion | Pass |
|---|---:|
| Relative mean G2 gain ≥2% | no |
| All three paired replicates favor G2 | yes |
| Both runs complete 200 finite epochs | yes |
| No unresolved divergence | yes |

## Inference

**Locked gate: FAIL.** The mean G2 gain is positive but below the locked 2% capacity threshold.

## Limits

- All fitted transforms, graph construction, and evaluation cells come from the same single core.
- The comparison is held-in and is not an estimate of generalization.
- The three mask replicates are technical repeats, not independent biological replicates.
- This analysis does not verify the earlier held-out predictive hypothesis.
- A capacity result does not establish cell-cell communication, a biological mechanism, or causality.
