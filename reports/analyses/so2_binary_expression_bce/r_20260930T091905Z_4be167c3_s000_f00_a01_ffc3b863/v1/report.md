# SO2 binary-expression BCE: completed training report

## Training and selected checkpoint

Run r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863 completed 129 epochs; the selected checkpoint is epoch 123 (stop reason: patience). Replayed validation BCE is 0.240180. The frozen optimization criterion passed: BCE improved by 0.033478 from epoch 1. All report integrity checks passed.

This report is exploratory and was designed after viewing the learning curve. It summarizes masked-gene occurrence prediction (raw count > 0) with observed target genes, morphology and fully observed neighboring expression available. Training: 208,696 cells from 12 cores/six donor groups. Validation: 37,367 cells from two cores/one donor group; 18,708,207 masked entries, including 1,518,702 positives. Each target is visited once per epoch; validation masks are fixed. Seed 0 only. Validation controls scheduling, stopping and checkpoint selection.

![Training history. Dotted vertical line: selected BCE checkpoint; red dashed line: interruption after epoch 17. Training and validation have different core distributions and masking realizations; their gap is not an overfitting estimate.](learning.png)

Training history. Dotted vertical line: selected BCE checkpoint; red dashed line: interruption after epoch 17. Training and validation have different core distributions and masking realizations; their gap is not an overfitting estimate.


## Occurrence prediction

Observed positive prevalence is 8.12%. At the fixed 0.5 threshold, precision is 68.68%, recall 7.91%, F1 0.1418, and balanced accuracy 53.79%. Accuracy is 92.23%, compared with 91.88% for predicting every entry negative. Most observed positive entries are missed at this threshold; high accuracy largely reflects class imbalance. No threshold was tuned for this report.

![Selected-checkpoint occurrence curves reconstructed from the retained 4,096-bin aggregate histogram. AUPRC means descending-threshold average precision, not trapezoidal PR area. The calibration display combines those bins into 20 groups; reported ECE uses all 4,096 bins. No per-entry predictions or confidence intervals are inferred.](detection.png)

Selected-checkpoint occurrence curves reconstructed from the retained 4,096-bin aggregate histogram. AUPRC means descending-threshold average precision, not trapezoidal PR area. The calibration display combines those bins into 20 groups; reported ECE uses all 4,096 bins. No per-entry predictions or confidence intervals are inferred.


## Matched hurdle comparison

Binary BCE minus hurdle detector BCE is +0.000195 (negative favors binary-only training). The split, preprocessing, initialization prevalence and fixed validation mask receipts match. The hurdle checkpoint was selected on joint hurdle-NB likelihood, while the binary checkpoint was selected on BCE; this comparison retains that selection asymmetry. Joint hurdle-NB NLL and binary BCE are different objectives and are not compared numerically. Hurdle AUROC, AUPRC and ECE are unavailable in its saved final artifacts. Hurdle accuracy, balanced accuracy, MCC and predicted-positive fraction are derived from saved confusion counts.

| Metric | Binary BCE | Hurdle NB | Binary − hurdle | Preferred |
| --- | --- | --- | --- | --- |
| Detection BCE | 0.240180 | 0.239984 | +0.000195 | lower |
| Brier score | 0.065949 | 0.065885 | +0.000065 | lower |
| Precision at 0.5 | 0.686825 | 0.688231 | -0.001406 | higher |
| Recall at 0.5 | 0.079058 | 0.080431 | -0.001372 | higher |
| F1 at 0.5 | 0.141795 | 0.144029 | -0.002234 | higher |
| Specificity at 0.5 | 0.996815 | 0.996781 | +0.000034 | higher |
| Accuracy at 0.5 | 0.922313 | 0.922393 | -0.000080 | higher |
| Balanced accuracy at 0.5 | 0.537937 | 0.538606 | -0.000669 | higher |
| MCC at 0.5 | 0.215374 | 0.217532 | -0.002158 | higher |
| Mean predicted probability | 0.082576 | 0.082808 | -0.000232 | descriptive |
| Predicted positive fraction at 0.5 | 0.009344 | 0.009487 | -0.000143 | descriptive |


## Descriptive references

Constant-prevalence BCE, Brier and AUPRC references below use observed validation prevalence. They are retrospective descriptive references, not training-fitted independent baselines. Epoch 1 already contains six optimizer updates; it is not an untrained prevalence baseline.

| Metric/reference | Value |
| --- | --- |
| AUROC (4,096 bins) | 0.770606 |
| AUPRC / average precision (4,096 bins) | 0.298425 |
| ECE (4,096 bins) | 0.003369 |
| Validation-prevalence BCE reference | 0.281638 |
| Validation-prevalence Brier reference | 0.074588 |
| Validation-prevalence AUPRC reference | 0.081178 |


## Interruption and resources

The original execution was interrupted after epoch 17 around 09:40 UTC on September 30; recorded host and container restart observations support an environment interruption, whose initiating cause is unknown. The same run resumed from its atomic checkpoint under supervisor at 17:13 UTC with model, optimizer, scheduler and RNG state restored. Gradient-direction continuity at epoch 18 is unavailable. No epochs are missing or duplicated. Runner duration is 1.91 hours for the resumed process; the sum of completed epoch durations is 2.04 hours across both segments. These are not total elapsed wall time: downtime and unrecoverable initial/partial-epoch work are excluded. Four RTX 3090 GPUs ran one DDP rank each; peak allocated VRAM was 5.08 GiB. Framework/CUDA and recovery receipts are retained in summary.json. Elapsed time from original launch through training completion, including the interruption, was 9.81 hours.


## Interpretation and limits

The strongest alternatives are that observed same-cell genes, morphology and gene prevalence explain performance without useful spatial context, that segmentation spillover mimics neighborhood information, or that count supervision regularizes the hurdle detector. Target-isolated routing, training-only preprocessing, donor-grouped splitting, matching masks, and replay/coverage checks address specific leakage and comparability risks. They do not demonstrate useful graph-specific information. Graphless/self/context baselines, realistic spatial nulls, segmentation robustness, multiple seeds/folds, independent-patient replication and a sealed test remain absent. Cells and the two validation cores are not independent patient replicates; no patient-level uncertainty interval is justified. Maximum claim: this validation-selected run learns masked occurrence prediction and supports an exploratory matched comparison. No mechanism or causal claim follows. The next discriminating test is a locked comparison against train-fitted gene-prevalence and self/context models across independent donor-held-out folds.


## Verification and provenance

Both immutable input bundles passed verify_run_bundle. The best checkpoint checksum, strict minimum epoch, replayed primary and secondary metrics (tolerance 1e-6), contiguous histories, finite gradients, once-per-target coverage and fixed mask receipts were checked without re-running a model. Registry/checkpoint-catalog lifecycle verification is performed separately by the monitoring task. Source SHA-256 records and all artifact hashes are available in summary.json and report_manifest.json. No raw data or patient identifiers are exported.

## Reproduction

Run from the repository root; choose a new version if the report already exists.

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_binary_expression_bce.py --run-id r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863 --comparator-id r_20260928T021117Z_9ea46874_s000_f00_a01_41b54097 --version v1
```
