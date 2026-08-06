# Post-hoc tokenized G2 relaxed categorical sensitivity

The prespecified 95% accuracy gate failed for all wider seeds. This analysis was run only after an explicit user request and is post-hoc exploratory; it does not revise the negative width or failed-gate conclusions.

## Numerical outcome

All 95% intervals below quantify only technical variation from resampling the three masks and Rademacher trace probes. They are not biological, patient, or training-seed uncertainty intervals.

- Identical-checkpoint numerical control: **PASS**.
- Exploratory operational cross-width match: **FAIL**.

| Seed | Cosine [95% technical CI] | Relative discrepancy [95% technical CI] | Norm ratio wider/current [95% technical CI] | All criteria |
|---:|---:|---:|---:|:---:|
| 0 | 0.5522 [0.5436, 0.5606] | 0.9638 [0.9547, 0.9728] | 0.8332 [0.8275, 0.8385] | no |
| 1 | 0.5249 [0.5198, 0.5296] | 1.0424 [1.0375, 1.0475] | 1.4437 [1.4353, 1.4522] | no |
| 2 | 0.5374 [0.5314, 0.5432] | 1.1113 [1.1016, 1.1204] | 0.5772 [0.5703, 0.5841] | no |

## Limitations

- Gradients are local, model- and scale-dependent sensitivities on one transductive core. They are not biological interactions or causal effects.
- Random current/wider controls use the same paired seed. Aligned parameter shapes may share RNG prefixes, raising null similarity and making separation conservative.
- Poor class-balanced and nonzero token accuracy remains the strongest limitation; functional similarity cannot rescue an uninformative predictor.
