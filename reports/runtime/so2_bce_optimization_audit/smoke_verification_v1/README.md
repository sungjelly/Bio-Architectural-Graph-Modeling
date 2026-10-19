# SO2 BCE audit smoke verification

Both two-epoch smoke bundles passed independent checksum, registry, checkpoint, coverage, finite-gradient, endpoint and resource checks. This is runtime evidence; no scientific score selection was performed.

| Mode | Run ID | Updates | Peak allocated / reserved GiB | Minimum sampled GPU headroom GiB |
|---|---|---:|---:|---:|
| target_window | `r_20261001T073730Z_b3feecba_s000_f00_a01_42b8b0fd` | 418 | 2.237 / 2.521 | 20.220 |
| core_pair | `r_20261001T073732Z_2bde0602_s000_f00_a01_e44fd1ae` | 12 | 14.880 / 17.973 | 4.768 |

Both runs covered 208,696 training targets and 37,367 validation targets per epoch. Training masks match across modes, validation masks remain fixed, and the training-fitted prevalence reference and source-code provenance are byte-identical. Each best checkpoint is separately stored, checksum-verified and indexed; replay preserved validation coverage.

Full run IDs, immutable source checksums, exact resource measurements and limitations are in [verification.json](verification.json). Physical GPU sampling occurs every five seconds. The runs do not determine a hyperparameter winner, eventual convergence or patient-level generalization.
