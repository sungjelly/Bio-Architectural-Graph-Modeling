# SO2 BCE optimization audit: regularization

Required completed coverage: **4/4**. Endpoint selection: **complete**.

The audit tests binary expression presence under target-isolated partial-gene masking. It separates the early effect of changing the update regimen from longer fixed-budget hyperparameter tuning; all validation comes from one held-out donor.

![Validation BCE against epochs, optimizer steps and epoch time](validation_curves.png)

| Mode / batch | LR | Dropout | WD | Seed | Terminal BCE | Best BCE (epoch) | Steps | Epoch min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target_window / 512 | 0.0003 | 0.1 | 1e-05 | 0 | 0.23970766 | 0.23958019 (8) | 3304 | 37.6 |
| target_window / 512 | 0.0003 | 0 | 1e-05 | 0 | 0.23966941 | 0.23919835 (8) | 3304 | 36.3 |
| target_window / 512 | 0.0003 | 0.3 | 1e-05 | 0 | 0.24067777 | 0.24058097 (8) | 3304 | 36.8 |
| target_window / 512 | 0.0003 | 0.1 | 0.01 | 0 | 0.23970513 | 0.23956997 (8) | 3304 | 37.9 |

Finalist: `r_20261001T092859Z_583aa3f1_s000_f00_a01_5a4a79ea`. Distinct confirmation comparator: `r_20261001T092859Z_583aa3f1_s000_f00_a01_5a4a79ea`.

Prespecified endpoints are means of epochs 7–8 for screens and 29–32 for confirmation. Best-checkpoint metrics are reported separately. Positive BCE reductions favor the comparator. Historical selected-best BCE 0.24017953 follows 129 epochs and is not a matched-budget control.

Train-fitted per-gene prevalence metrics, fixed-training diagnostics when required, per-core directions, all attempts and source checksums are in summary.json and the companion tables.

Other attempts (including preserved failures and unfinished runs): 1.

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

Preserved failed or unfinished attempts:

- `r_20261001T092859Z_5cb93f7c_s000_f00_a01_0fb40980`: running; category none.
