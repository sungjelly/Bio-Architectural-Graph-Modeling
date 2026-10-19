# SO2 BCE optimization audit: grid

Required completed coverage: **0/10**. Endpoint selection: **incomplete_no_selection**.

The audit tests binary expression presence under target-isolated partial-gene masking. It separates the early effect of changing the update regimen from longer fixed-budget hyperparameter tuning; all validation comes from one held-out donor.

![Validation BCE against epochs, optimizer steps and epoch time](validation_curves.png)

| Mode / batch | LR | Dropout | WD | Seed | Terminal BCE | Best BCE (epoch) | Steps | Epoch min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

![Prespecified terminal endpoint comparison](endpoint_comparison.png)

Completed smoke runs verify readiness and are excluded from screen ranking:

![Completed two-epoch readiness smoke curves](smoke_validation_curves.png)

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

Preserved failed or unfinished attempts:

- `r_20261001T074811Z_06acf7a5_s000_f00_a01_2aa32391`: running; category none.
- `r_20261001T074811Z_10be86dd_s000_f00_a01_ec605202`: running; category none.
- `r_20261001T074811Z_425c1faf_s000_f00_a01_65533cf1`: running; category none.
- `r_20261001T074811Z_97513f17_s000_f00_a01_bfa65cb7`: running; category none.
