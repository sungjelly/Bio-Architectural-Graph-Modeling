# Backward-removal F1 figures

Presentation derivative of ev_so12_cancer_backward_union_s0_20261018.
Fixed resolution 1, Leiden seed 0. No metric recomputation or new selection.
The two y axes start at 0.72 / 0.86 and have the same 0.08 F1 span.
F1 values are descriptive, annotation-selected overlap with approximate registration.

- f1_bar_comparison.png / .pdf / .svg: two-core bar figure.
- f1_retention_table.png / .pdf / .svg: F1, cluster count/percentage and cell retention.
- f1_table.csv: seven-method wide table, full precision F1.
- f1_and_retention_full_precision.csv: unchanged source table with counts and operating points.

Reproduction from the repository root:
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/plot_so12_backward_f1_bars.py

Publication is immutable. To rerender, use a new FIGURE_ID in a versioned copy of
this script. The script copied here and its source hashes identify the exact render.
All 14 values match the audited source path steps and CSV. Root and independent
visual reviews passed both figures. The original evaluation remains unchanged.
