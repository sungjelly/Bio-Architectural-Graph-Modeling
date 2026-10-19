# SO2 BCE optimization audit: grid

Required completed coverage: **4/10**. Endpoint selection: **incomplete_no_selection**.

The audit tests binary expression presence under target-isolated partial-gene masking. It separates the early effect of changing the update regimen from longer fixed-budget hyperparameter tuning; all validation comes from one held-out donor.

![Validation BCE against epochs, optimizer steps and epoch time](validation_curves.png)

| Mode / batch | LR | Dropout | WD | Seed | Terminal BCE | Best BCE (epoch) | Steps | Epoch min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| target_window / 512 | 3e-05 | 0.1 | 1e-05 | 0 | 0.24156873 | 0.24147724 (8) | 3304 | 37.9 |
| target_window / 512 | 0.0001 | 0.1 | 1e-05 | 0 | 0.24036931 | 0.24018002 (8) | 3304 | 37.3 |
| target_window / 1024 | 0.0001 | 0.1 | 1e-05 | 0 | 0.24099956 | 0.24093517 (8) | 1672 | 30.7 |
| target_window / 2048 | 0.0003 | 0.1 | 1e-05 | 0 | 0.24135273 | 0.24084219 (8) | 864 | 28.8 |

![Prespecified terminal endpoint comparison](endpoint_comparison.png)

Prespecified endpoints are means of epochs 7–8 for screens and 29–32 for confirmation. Best-checkpoint metrics are reported separately. Positive BCE reductions favor the comparator. Historical selected-best BCE 0.24017953 follows 129 epochs and is not a matched-budget control.

Train-fitted per-gene prevalence metrics, fixed-training diagnostics when required, per-core directions, all attempts and source checksums are in summary.json and the companion tables.

Other attempts (including preserved failures and unfinished runs): 4.

- Exploratory validation tuning on one held-out donor; no independent test cohort or patient confidence interval.
- Seeds measure optimization variability. The two validation cores do not establish patient replication.
- Across batch sizes equal epochs match target exposure, while optimizer updates differ; within batch size LR comparisons match both.
- Selected-best replay metrics and fixed-training diagnostics are separate from the prespecified terminal endpoint.
- Online training BCE uses changing masks and dropout; the raw train-validation gap is not a controlled fitting estimate.
- Model AP/AUROC use 4096-bin histograms; the train-fitted per-gene reference has exact score-tie handling.
- Timing includes validation in epoch histories; startup, checkpoint I/O, final replay and diagnostic work additionally contribute to run duration.
- No graph-specific predictive gain, faithfulness, biological null, independent replication, mechanism or causality is tested.

Train-fitted per-gene prior BCE on the same validation masks: **0.27807952**.

Preserved failed or unfinished attempts:

- `r_20261001T082016Z_95cce2a4_s000_f00_a01_df1ebd49`: running; category none.
- `r_20261001T082215Z_22f0027c_s000_f00_a01_54281998`: running; category none.
- `r_20261001T082850Z_052d767b_s000_f00_a01_35ab7f19`: running; category none.
- `r_20261001T082930Z_d0f806f6_s000_f00_a01_57f4d373`: running; category none.
