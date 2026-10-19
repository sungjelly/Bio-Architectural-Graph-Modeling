# SO2 binary occurrence versus hurdle detection

**Outcome: negative for improvement in detection BCE, Brier score, precision,
recall and F1 at the prespecified selected endpoints.** Specificity is slightly
higher for binary-only training. This is a verified exploratory descriptive
comparison; it does not establish model-family superiority.

The question is whether removing positive-count supervision improves occurrence
prediction on the same fixed masked target entries. Both prespecified seed-0,
fold-0, attempt-1 runs are included. Binary-only training completed 129 epochs
and selected epoch 123; hurdle-NB completed 171 and selected epoch 164. Binary
selection used BCE, while hurdle selection used joint hurdle-NB NLL. Only the
hurdle detector's BCE is compared with binary BCE.
Training exposure differs because each objective drives scheduling and
stopping. This remains an additional limit on attributing endpoint differences
to removal of count supervision.

| Metric | Binary-only | Hurdle detector | Binary minus hurdle |
| --- | ---: | ---: | ---: |
| Detection BCE | 0.240179534 | 0.239984150 | +0.000195384 |
| Brier score | 0.065949287 | 0.065884707 | +0.000064580 |
| Precision at 0.5 | 0.686825351 | 0.688231052 | -0.001405701 |
| Recall at 0.5 | 0.079058301 | 0.080430526 | -0.001372224 |
| F1 at 0.5 | 0.141795024 | 0.144029016 | -0.002233992 |
| Specificity at 0.5 | 0.996815092 | 0.996780943 | +0.000034149 |

Lower BCE and Brier are better; higher threshold metrics are better. Threshold
0.5 was fixed. Binary AUROC is 0.770606, binned average precision (reported as
AUPRC) is 0.298425, and ECE is 0.003369, each from 4,096 probability bins.
Corresponding hurdle AUROC/AUPRC/ECE were not retained, so their comparative
improvement is unknown. The small specificity improvement and missing AUPRC
comparison prevent a blanket claim that every occurrence metric worsens or
that the entire binary-specialization hypothesis was falsified.

The separate frozen optimization criterion **passed**: selected validation BCE
0.2401795339072312 improved by 0.033478191549836916 over epoch-1 BCE
0.2736577254570681, exceeding the 0.0001 threshold; required losses and gradients
were finite. Epoch 1 already included six optimizer updates and is not an
untrained prevalence baseline.

Training used 208,696 cells from 12 cores/six donor groups. Validation used
37,367 cells from two cores belonging to one donor group, with 18,708,207 fixed
masked entries and 1,518,702 positives (8.12%). Each target has one uniform
1-through-1,000-gene mask; neighbors are fully observed. The pooled masked-entry
estimand mixes partial and occasional whole-target masks and does not provide a
separate whole-node result. Validation controlled scheduling, stopping and
checkpoint selection; there is no sealed test estimate or patient-level
uncertainty interval.

The strongest explanations remain count-supervision regularization, selection
objective differences, intracellular co-expression, gene prevalence and
morphology; segmentation spillover can mimic spatial information. Identical
dataset/split/preprocessing/graph fingerprints, exact matched training masks
and core order through all 129 shared epochs, fixed validation mask receipts,
target-isolated routing and checkpoint replay address specific comparability
and leakage risks. They do not establish graph-specific predictive gain.
Train-fitted prevalence/self/context baselines, realistic spatial nulls,
segmentation robustness, multiple seeds/folds, independent-patient replication,
faithfulness, external support and perturbational evidence remain untested.
Validation-prevalence references in the detailed report are retrospective
class-imbalance descriptions, not independent fitted baselines.

The binary run was interrupted after epoch 17 and recovered within the same
immutable run ID and attempt. It finalized on September 30, 2026 at approximately
19:07:54 UTC. No planned scientific run was excluded and the interruption is not
a failed seed. All epochs are present; gradient-direction continuity at epoch 18
is unavailable. Both immutable bundles and the selected checkpoint were
verified, and the separate completion receipt confirms the terminal registry,
queue and checkpoint-catalog state. Record status `verified` denotes this audit;
it does not upgrade the scientific claim.

The decision supported here is to retain the hurdle detector as the reference
for these five measured metrics on this split, without promoting binary-only
supervision as an improvement. The next discriminating test is a locked
comparison with train-fitted prevalence and self/context baselines over
independent donor-held-out folds, retaining matched discrimination metrics.

The [portable report](../../../../reports/analyses/so2_binary_expression_bce/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/v1/report.html),
[complete numeric summary](../../../../reports/analyses/so2_binary_expression_bce/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/v1/summary.json),
and [completion receipt](../../../../reports/analyses/so2_binary_expression_bce/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/completion_verification_v1.json)
are referenced by checksum in `result.yaml`. Only these two small result files
are curated here; datasets, checkpoints and full run bundles remain at their
canonical locations.

From the repository root, verify the completed lifecycle without writing:

```bash
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_binary_expression_bce/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/verify_bce_completion_v1.py --check-only
```

Rebuild using the exact saved report implementation and a new report version:

```bash
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_binary_expression_bce/r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863/v1/report_builder.py --run-id r_20260930T091905Z_4be167c3_s000_f00_a01_ffc3b863 --comparator-id r_20260928T021117Z_9ea46874_s000_f00_a01_41b54097 --version v2
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate --verify-payloads --verify-sources
```
