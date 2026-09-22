# SO2 per-epoch comparison

Exploratory audit of recorded training masked Huber; lower is better.

![Loss trajectories and paired epoch differences](epoch_comparison.png)

[Every epoch (CSV)](per_epoch_comparison.csv) · [Every core/epoch (CSV)](per_core_epoch_losses.csv) · [25-epoch blocks](blocks_25_epochs.csv) · [PDF figure](epoch_comparison.pdf)

| Epoch | Original + continuation | Recurrent* | Geometry | Geometry − original |
|---:|---:|---:|---:|---:|
| 1 | 0.275392608 | 0.286406829 | 0.274927556 | -0.000465051 |
| 10 | 0.249187371 | 0.251354252 | 0.249319921 | 0.000132550 |
| 25 | 0.246101777 | 0.247039993 | 0.246198710 | 0.000096933 |
| 50 | 0.243679308 | 0.244611401 | 0.244088519 | 0.000409212 |
| 75 | 0.242669745 | 0.243078527 | 0.243132500 | 0.000462755 |
| 100 | 0.241925727 | 0.242525720 | 0.242103169 | 0.000177442 |
| 125 | 0.241537604 | 0.242018620 | 0.241556667 | 0.000019063 |
| 150 | 0.241604707 | 0.241842561 | 0.241342603 | -0.000262104 |
| 175 | 0.240848190 | 0.241221837 | 0.241710118 | 0.000861927 |
| 200 | 0.240678813 | — | 0.240708410 | 0.000029597 |
| 250 | 0.240242460 | — | — | — |
| 300 | 0.239695210 | — | — | — |

Geometry has lower loss in 71/200 matched epochs against the original trajectory and 167/175 against recurrent. Against the original, the direction reverses repeatedly; the 25-epoch block means favor geometry in blocks 1–25, 76–100 and 126–150, and favor the original in the other five shared blocks.

At epochs 176–200, mean training Huber is 0.240917320 for geometry and 0.240760973 for the original (geometry 0.06494% higher). There is no sustained geometry advantage over the original in these logs.

*Recurrent training completed 175 epochs, but artifact finalization failed. Its recorded losses are included descriptively with that status. The failed duplicate original continuation is listed in provenance and not counted as an independent model. Original epochs 1–175 are an exact inherited prefix of the successful continuation; the table uses that training trajectory once.

MAE, MSE and R² exist only for final fixed-mask evaluations, not every epoch. Here each loss was measured during training, with dropout and changing model weights. Curves stop where each model stopped; no missing epochs are filled. The fixed 25-epoch smoothing window is a post-hoc descriptive aid.

All models fit the same cells; observed same-cell genes and morphology remain available. One seed, repeated epochs and cores provide no patient-level uncertainty, held-out generalization estimate, graph-specific gain or biological mechanism evidence. Joint attention changes and execution differences prevent attribution to geometry modulation alone. These training diagnostics do not change a project scientific gate.

Verification: source hashes, epoch coverage, all equal-core means, shared configs, exact original-prefix identity, mask checksums/core order/RNG seeds, missing endpoints, and output checksums pass. See [verification.json](verification.json) and [provenance.json](provenance.json).
