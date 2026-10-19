# SO2 BCE target-cell batch resource selection

Selected **1,024 target cells per optimizer update**, using the prespecified resource rule.
All three registered two-epoch pilots passed coverage, numerical, memory and best-checkpoint replay checks.

| Targets/update | Updates/epoch | GPU | Epoch 1 / 2 (s) | Mean (s) | Peak allocated / reserved (GiB) | Min free (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 413 | 0 | 269.7 / 255.2 | 262.5 | 2.06 / 2.42 | 20.32 |
| 1024 | 209 | 1 | 220.5 / 219.0 | 219.8 | 2.24 / 2.52 | 20.22 |
| 2048 | 108 | 2 | 207.2 / 204.9 | 206.1 | 2.49 / 2.78 | 19.96 |

Every epoch targets all 208,696 training cells once and evaluates all 37,367 validation cells with fixed masks. Balanced contiguous windows are shuffled within each shuffled core. The context graph remains complete; learned source embeddings are recomputed after each update.

The historical paired-core run used six optimizer steps per epoch. These candidates change the optimization budget and stochastic trajectory. An eventual comparison must show epochs, optimizer steps and elapsed time; these pilot scores do not establish predictive improvement.

Timing limitations: Two epochs only; timings include validation but exclude startup, checkpoint I/O and final replay. Candidates ran on different cards with concurrent host and I/O load. Runtime pilots are not independent scientific replicates or evidence of predictive superiority. ETA scales observed epoch time and does not predict early stopping or future contention.

Immutable source runs:

- B=512: `r_20261001T071512Z_ea5598e5_s000_f00_a02_2b3f9d91`
- B=1024: `r_20261001T071514Z_b891ab91_s000_f00_a02_3f586be8`
- B=2048: `r_20261001T071516Z_37baa123_s000_f00_a02_0e765ab7`

Failed attempts (preserved, excluded from successful timing comparisons):

- `r_20261001T070447Z_141a30b5_s000_f00_a01_710b2c74`: Pilot minimum_epochs equaled maximum_epochs; the epoch controller requires a strict inequality. The attempt completed its first training and validation pass, then failed before checkpoint commit.
- `r_20261001T070449Z_8f1694f7_s000_f00_a01_9060ae12`: Pilot minimum_epochs equaled maximum_epochs; the epoch controller requires a strict inequality. The attempt completed its first training and validation pass, then failed before checkpoint commit.
- `r_20261001T070451Z_7921e6b8_s000_f00_a01_aca29682`: Pilot minimum_epochs equaled maximum_epochs; the epoch controller requires a strict inequality. The attempt completed its first training and validation pass, then failed before checkpoint commit.
