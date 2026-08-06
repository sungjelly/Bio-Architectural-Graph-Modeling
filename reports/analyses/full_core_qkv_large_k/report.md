# Full-core QKV-GAT large-k comparison

This is a locked, held-in regression-capacity comparison. Percent variance explained is `100 * masked R²`; it is not classification accuracy. This is not an estimate of generalization.

## Runs

| Role | Run ID | k | Parameters | Huber | R² | PVE (%) |
|---|---|---:|---:|---:|---:|---:|
| k1000 | `r_20260726T151303Z_e6052ff9_s000_f00_a01_59aaad67` | 1000 | 36,749,480 | 0.228182 | -0.008443 | -0.844278 |
| k5000 | `r_20260726T151303Z_712ec9be_s000_f00_a01_0b694448` | 5000 | 36,749,480 | 0.227444 | 0.008082 | 0.808240 |
| matched_self | `r_20260726T151303Z_4ca683a4_s000_f00_a01_dfe67899` | 5000 | 36,749,480 | 0.227723 | -0.005064 | -0.506404 |

## Paired comparisons

| Candidate vs reference | Relative Huber gain (%) | PVE difference (points) | All 3 Huber masks favor candidate |
|---|---:|---:|:---:|
| k1000_vs_matched_self | -0.201485 | -0.337874 | no |
| k5000_vs_matched_self | 0.122298 | 1.314644 | yes |
| k5000_vs_k1000 | 0.323132 | 1.652518 | yes |

## Locked gates

- Evaluated graph run IDs: `r_20260726T151303Z_e6052ff9_s000_f00_a01_59aaad67`, `r_20260726T151303Z_712ec9be_s000_f00_a01_0b694448`.
- Source/environment provenance: identical; Git commit `6081a49496f3848594649bc3705573e7818df545`.
- Representation gate: **FAIL**; eligible graph run IDs: none.
- k=5,000 vs k=1,000 gate: **FAIL**.

A failed gate is a valid negative result; thresholds were not changed after observing these metrics.

## Limitations

- All cells and preprocessing statistics were fitted; this is held-in transductive reconstruction, not generalization.
- The three masks are technical repeats in one spatial core, not independent biological replicates.
- No validation, test, patient-held-out, or independent-cohort result is included.
- The k=5,000 graph represents broad regional context and cannot by itself establish direct cell-cell interaction.
- Predictive capacity and attention weights do not establish biological mechanism or causality.
- The checkpoint catalog's internal `validation_selection` evidence-tier label means exploratory candidate retention for later validation; no validation set or validation-based checkpoint selection was used here.

Maximum defensible claim: one-core transductive held-in masked-expression representation capacity under the locked QKV architecture and 300-epoch budget.
