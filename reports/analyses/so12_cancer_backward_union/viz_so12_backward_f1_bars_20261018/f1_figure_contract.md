# F1 bar figure: presentation derivative

Status: planned. This is a visual reformat of the completed exploratory
`ev_so12_cancer_backward_union_s0_20261018` evaluation, not a new experiment,
selection, or scientific conclusion. Preserve its immutable bundle and curated
record. Source manifest SHA256:
`88f966f74307485e1c66c2728f41b4efe9bb51860823c63133976605c0483ffa`.

Deliver a two-panel bar figure, vector PDF/SVG, readable table, and numeric CSV.
Use all seven methods, fixed resolution 1 and Leiden seed 0, and each path's
best F1. Use y ranges 0.72–0.80 and 0.86–0.94: both spans are 0.08, all values
remain visible, and truncation must be explicit. Print scores to four decimals.
The table also reports clusters retained / initially overlapping clusters and
unweighted in-region cell retention at that same operating point.

Inherited estimand: pixel Dice/F1 on the valid segmented footprint; selected
outside pixels count as false positives. Core is the observational biological
unit. Annotation-selected groups and approximate alignment limit these to
descriptive agreement. Cell retention uses all cells with any valid owned pixel
inside the annotation as its fixed denominator. No inference/error bars are
invented. A plausible display failure is distorted comparison from truncation;
explicit limits, equal spans, exact labels, and the companion table address it.

Acceptance: source inventory verification; all 14 F1 values and retention counts
agree with the audited path steps and existing CSV; numerical limits include
all bars; inspect plot/table for clipping and legibility. Stop on mismatches.
CPU-only rendering needs seconds and <10 MiB storage; no raw data access,
GPU, retraining, reclustering, or deletion is required.

Run from the repository root:

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/plot_so12_backward_f1_bars.py
```

The script renders into `scratch/active_runs/viz_so12_backward_f1_bars_20261018/`.
After direct review, publish without overwriting to
`reports/analyses/so12_cancer_backward_union/viz_so12_backward_f1_bars_20261018/`.
Preserve source/script hashes and the reviewed file checksums in the derivative.
