# SO2 BCE optimization audit: confirmation

Required completed coverage: **4/4**. Endpoint selection: **complete**.

The audit tests binary expression presence under target-isolated partial-gene masking. It separates the early effect of changing the update regimen from longer fixed-budget hyperparameter tuning; all validation comes from one held-out donor.

![Validation BCE against epochs, optimizer steps and epoch time](validation_curves.png)

| Mode / batch | LR | Dropout | WD | Seed | Terminal BCE | Best BCE (epoch) | Steps | Total epoch time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target_window / 1024 | 0.0001 | 0.1 | 1e-05 | 0 | 0.23888354 | 0.23877644 (32) | 6688 | 125.3 |
| target_window / 1024 | 0.0001 | 0.1 | 1e-05 | 1 | 0.23900566 | 0.23870651 (31) | 6688 | 125.2 |
| target_window / 512 | 0.0003 | 0 | 1e-05 | 0 | 0.23882838 | 0.23841607 (25) | 13216 | 138.4 |
| target_window / 512 | 0.0003 | 0 | 1e-05 | 1 | 0.23869429 | 0.23840900 (28) | 13216 | 139.6 |

Recorded runtime and resources for included stage runs:

| Run | Physical GPU | Total epoch time (min) | Elapsed run (min) | Peak allocated / reserved (GiB) | Min observed free (GiB) |
|---|---:|---:|---:|---:|---:|
| `r_20261001T092859Z_5cb93f7c_s000_f00_a01_0fb40980` | 3 | 125.3 | 129.9 | 2.819 / 3.557 | 20.054 |
| `r_20261001T101202Z_15baf65c_s001_f00_a01_6427735a` | 2 | 125.2 | 130.1 | 2.819 / 3.557 | 20.054 |
| `r_20261001T101202Z_f8197b29_s000_f00_a01_ed00c19e` | 0 | 138.4 | 143.0 | 2.822 / 3.484 | 20.232 |
| `r_20261001T101202Z_a32d61e8_s001_f00_a01_8332ee26` | 1 | 139.6 | 144.2 | 2.822 / 3.484 | 20.232 |

Total epoch time sums the recorded training, validation and control sections, before each epoch's checkpoint/history persistence. Elapsed run is the trainer clock from configuration/data loading through summary construction, including setup, checkpoint writes, selected-best replay and final diagnostics. It excludes earlier process/distributed startup, subsequent final artifact verification/retention, and worker publication/checkpoint indexing; it is not end-to-end launcher wall time or GPU kernel time.

Minimum observed free memory is the saved conservative minimum across sampled physical GPU memory and CUDA free-memory queries; it is not a continuous physical-memory measurement.

Maximum observed shared cgroup memory across these runs: **59.704 GiB**. This is aggregate shared-host memory, including concurrent jobs, not per-job RAM.

![Prespecified terminal endpoint comparison](endpoint_comparison.png)

Mean paired-seed BCE reduction: **0.00018326**; material-gain criterion: **False**.
Any endpoint still improving by >0.0005: **True**. Eventual-performance conclusion: **inconclusive**.

- Seed 0: BCE reduction 0.00005516; core effects {"SO2-C27": 0.0001242400378090036, "SO2-C28": -7.242537722829123e-05}.
- Seed 1: BCE reduction 0.00031137; core effects {"SO2-C27": 0.00028122144013720263, "SO2-C28": 0.00036703925742673227}.

Prespecified endpoints are means of epochs 7–8 for screens and 29–32 for confirmation. Best-checkpoint metrics are reported separately. Positive BCE reductions favor the comparator. Historical selected-best BCE 0.24017953 follows 129 epochs and is not a matched-budget control.

Train-fitted per-gene prevalence metrics, fixed-training diagnostics when required, per-core directions, all attempts and source checksums are in summary.json and the companion tables.

Other attempts (including preserved failures and unfinished runs): 0.

- Exploratory validation tuning on one held-out donor; no independent test cohort or patient confidence interval.
- Seeds measure optimization variability. The two validation cores do not establish patient replication.
- Across batch sizes equal epochs match target exposure, while optimizer updates differ; within batch size LR comparisons match both.
- Selected-best replay metrics and fixed-training diagnostics are separate from the prespecified terminal endpoint.
- Online training BCE uses changing masks and dropout; the raw train-validation gap is not a controlled fitting estimate.
- Model AP/AUROC use 4096-bin histograms; the train-fitted per-gene reference has exact score-tie handling.
- Timing includes validation in epoch histories; startup, checkpoint I/O, final replay and diagnostic work additionally contribute to run duration.
- No graph-specific predictive gain, faithfulness, biological null, independent replication, mechanism or causality is tested.

Train-fitted per-gene prior BCE on the same validation masks: **0.27807952**.

At eight matched epochs, B=1,024 target windows reduce terminal BCE by 0.00719035 relative to paired-core updates; F1 change is 0.03931919. Each run sees 1,669,568 targets, with 1,672 versus 48 optimizer steps.
This is evidence about early optimization under different update regimens. It is separate from the 32-epoch tuned-versus-incumbent contrast. A negative tuning result does not erase an observed early update-regimen benefit; early benefit alone does not establish eventual BCE or F1 superiority.

Selected-best fixed-training diagnostics (evaluation mode, epoch-one training masks; diagnostic only):

| Run | Fixed training BCE | Training BCE reduction from gene prior | Selected-best validation BCE |
|---|---:|---:|---:|
| `r_20261001T092859Z_5cb93f7c_s000_f00_a01_0fb40980` | 0.29831199 | 0.04220601 | 0.23877644 |
| `r_20261001T101202Z_15baf65c_s001_f00_a01_6427735a` | 0.29802339 | 0.04249461 | 0.23870651 |
| `r_20261001T101202Z_f8197b29_s000_f00_a01_ed00c19e` | 0.29542224 | 0.04509576 | 0.23841607 |
| `r_20261001T101202Z_a32d61e8_s001_f00_a01_8332ee26` | 0.29491455 | 0.04560345 | 0.23840900 |
