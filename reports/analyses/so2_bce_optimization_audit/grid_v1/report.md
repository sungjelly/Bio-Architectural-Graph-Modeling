# SO2 BCE optimization audit: grid

Required completed coverage: **10/10**. Endpoint selection: **complete**.

The audit tests binary expression presence under target-isolated partial-gene masking. It separates the early effect of changing the update regimen from longer fixed-budget hyperparameter tuning; all validation comes from one held-out donor.

![Validation BCE against epochs, optimizer steps and epoch time](validation_curves.png)

| Mode / batch | LR | Dropout | WD | Seed | Terminal BCE | Best BCE (epoch) | Steps | Epoch min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target_window / 512 | 3e-05 | 0.1 | 1e-05 | 0 | 0.24156873 | 0.24147724 (8) | 3304 | 37.9 |
| target_window / 512 | 0.0001 | 0.1 | 1e-05 | 0 | 0.24036931 | 0.24018002 (8) | 3304 | 37.3 |
| target_window / 512 | 0.0003 | 0.1 | 1e-05 | 0 | 0.23970766 | 0.23958019 (8) | 3304 | 37.6 |
| target_window / 1024 | 3e-05 | 0.1 | 1e-05 | 0 | 0.24280128 | 0.24279287 (8) | 1672 | 31.0 |
| target_window / 1024 | 0.0001 | 0.1 | 1e-05 | 0 | 0.24099956 | 0.24093517 (8) | 1672 | 30.7 |
| target_window / 1024 | 0.0003 | 0.1 | 1e-05 | 0 | 0.24049435 | 0.24020361 (8) | 1672 | 29.0 |
| target_window / 2048 | 3e-05 | 0.1 | 1e-05 | 0 | 0.24409312 | 0.24371907 (8) | 864 | 27.9 |
| target_window / 2048 | 0.0001 | 0.1 | 1e-05 | 0 | 0.24187322 | 0.24170479 (7) | 864 | 29.2 |
| target_window / 2048 | 0.0003 | 0.1 | 1e-05 | 0 | 0.24135273 | 0.24084219 (8) | 864 | 28.8 |
| core_pair / None | 0.0001 | 0.1 | 1e-05 | 0 | 0.24818992 | 0.24769855 (8) | 48 | 24.9 |

![Prespecified terminal endpoint comparison](endpoint_comparison.png)

Grid selection: `r_20261001T082850Z_052d767b_s000_f00_a01_35ab7f19`. BCE reduction from matched incumbent: 0.00129191.

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
