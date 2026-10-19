# SO1 + SO2 Pearson preprocessing is plausible in a donor-separated pilot

Vendor-style quasi-Poisson Pearson residuals reduce the held-out-core depth-association
score from 0.3721 to 0.0331, with improvement in all eight diagnostic cores. Signed-log
Pearson reaches 0.0307; both pass the fixed provisional annotation guardrail. Plain
Pearson is the preferred next training baseline because the extra signed-log step
has inconsistent benefits and lower SO2 annotation recovery than plain Pearson.

This is shared preprocessing, not demonstrated batch correction or model-training
improvement. The twelve vendor letters are expression-derived and not independent
biological truth. Four diagnostic donors, one fixed split and one seed support only
an exploratory conclusion. Full-cell totals mean these exported features cannot be
masked afterward for a leakage-safe hidden-count benchmark.

The full report, plots, exact methods and derived dataset pointers are in
`reports/analyses/so12_pearson_preprocessing/`. Both transforms cover all 407,659 mapped
cells and 1,000 biological genes. Source observations and checkpoints were retained.
One CUDA initialization failure preceded the successful, scientifically unchanged retry.
