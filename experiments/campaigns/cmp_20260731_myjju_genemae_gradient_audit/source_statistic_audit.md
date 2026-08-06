# Source statistic audit

## Scope and provenance

This is a code-level audit of
`/workspace/Gastric-Cancer-Analysis-by-MyJJu/scripts/gene_mae_coexpr.py`
at repository commit
`f9ef61071c7e9b2751bbd59d154c13de534e7f2f`. The audited file has SHA-256
`0ae143b5957e9275882ba595702d6eacd033545beda306f0f0f82905a8174680`.
The source checkpoints and exact sampled tile identities are not present, so
its reported numerical matrix cannot be independently replayed. This campaign
reproduces the procedure on the seven checksum-verified ten-core checkpoints;
it is not a numerical reproduction of the missing historical weights.

## What the code computes

For each selected output gene \(i\), the source differentiates the sum of that
gene's reconstruction over every output cell with respect to every input entry
of selected gene \(j\), sums those derivatives over input cells, and divides by
the number of input cells:

\[
J_{ij} =
\frac{1}{N}\sum_{u=1}^{N}\sum_{v=1}^{N}
\frac{\partial \widehat{x}_{v i}}{\partial x_{u j}}.
\]

The implementation is at lines 79--103 of `gene_mae_coexpr.py`. It supplies an
all-false mask, so the target gene remains visible to the model. In a graph
model this is not simply the advertised cell-wise diagonal average
\(N^{-1}\sum_u \partial\widehat{x}_{ui}/\partial x_{uj}\): it also includes all
cross-cell derivatives. Because output and input cell identities are summed
before saving, the statistic cannot recover which receiver depended on which
source cell, distinguish direct from multihop paths, or separate the explicit
self branch from graph self-loops.

The saved statistic is not \(J\). Lines 163--169 replace it with

\[
J^\mathrm{published}_{ij}
= \frac{1}{2}\left(|J_{ij}|+|J_{ji}|\right).
\]

This discards sign and direction. Receiver-specific positive and negative
effects can also cancel inside \(J\) before the absolute value is taken.

## Selection and controls

The ranking universe is a hand-curated set of 39 known lineage and
gastric-cancer-associated genes. The reported top partners are therefore
conditional on that selected universe. Agreement with the same biological
knowledge used to choose and annotate the genes is circular annotation, not
independent validation.

The source analysis does not include:

- masked-target gradients;
- same-cell versus other-cell decomposition;
- self-branch, graph, or hop ablations;
- node-label-permuted, rewired, or parameter-randomized gradients;
- finite-difference faithfulness tests;
- seed stability or uncertainty;
- matched expression/co-expression nulls; or
- independent perturbational validation.

Its full-slide computation is also not the same cohort estimand as this
campaign's ten pathology-confirmed adjacent-normal cores. A negative result
here is evidence against support in the adjacent-normal cohort, not proof that
a tumor-specific dependency cannot exist.

## Claim audit

The source uses “causal coupling” in a heading and print label, but its own
report later states that the quantity is a local input-gradient sensitivity
and “not a mechanistic causal claim.” The latter statement is correct.

Even a stable, graph-dependent, finite-difference-faithful gradient would show
only that the frozen predictor locally uses an input. It would not establish a
specific molecular interaction, a sender-to-receiver mechanism, direction in
the biological system, or causality. Those conclusions require independent
biological units and controlled perturbation with relevant cell-autonomous,
off-target, pathway-blockade, and rescue controls.

The maximum claim available to this campaign is therefore a stable, faithful,
graph-dependent, null-calibrated **model-implied predictive sensitivity**.
