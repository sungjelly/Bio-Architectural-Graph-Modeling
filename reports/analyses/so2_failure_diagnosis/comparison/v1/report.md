# Why SO2 reconstruction gains remain limited

Exploratory diagnostic and literature comparison · 2026-09-06. All local numerical
comparisons below use the same 14 fitted cores and original hidden-entry masks.

The evidence supports an **objective/metric mismatch**, and leaves several
architectural explanations unresolved. Geometry is active, but has not improved
the main positive-error comparison over the original lineage. This is not evidence
that graph models in general fail or that changing to NB will fix the problem.

![Errors on four scales for all four endpoints and six fixed baselines. Dots show individual fitted cores; diamonds show equal-core means.](diagnostic_comparison.png)

## Matched performance: the missing simple baselines

All errors below are lower-is-better. Positive means observed raw count >0,
not positive standardized expression. These are distinct metrics, not accuracy
percentages. Full numerical tables and per-core paired differences are included.

| Predictor | All-entry Huber | All-entry z-MSE | Positive z-MSE | Positive log1p MSE |
|---|---:|---:|---:|---:|
| Gene mean (all fit) | 0.2710 | 0.9998 | 8.7809 | 0.7566 |
| Always zero | 0.2636 | 1.1101 | 10.7182 | 1.0738 |
| Huber constant (all fit) | 0.2537 | 1.0381 | 9.8634 | 0.8681 |
| Local 16: mean log1p | 0.3154 | 1.0856 | 8.5311 | 0.6870 |
| Local 16: mean count | 0.3904 | 1.2426 | 7.8989 | 0.6194 |
| Full graph: mean log1p | 0.2705 | 0.9828 | 8.5894 | 0.7010 |
| Original e175 | 0.2404 | 0.9487 | 8.8628 | 0.6882 |
| Continued e300 | 0.2395 | 0.9474 | 8.8505 | 0.6812 |
| Recurrent e175 | 0.2404 | 0.9558 | 8.9693 | 0.6970 |
| Geometry e200 | 0.2400 | 0.9516 | 8.9151 | 0.6905 |

The spatial baselines average **only observed neighboring values**, including
observed zeros. Local baselines use up to 16 nearest cells within 75 µm. Full-graph
averaging uses every incoming edge of the exact original graph, about 227.5
neighbors/cell globally. A missing neighborhood falls back to the observed core
gene mean, then zero; hidden entries never enter either fallback. The log-mean
and raw-count-mean baselines differ by averaging scale, so their metric tradeoff
must not be attributed solely to spatial scale. They are fixed, untuned references.

**The model beats naive averaging overall, but not on every positive-only metric.**
Geometry's all-entry standardized MSE is 3.17% lower than the original-full-graph
mean, with lower error in 14/14 cores. Its positive standardized MSE is 3.79%
higher than that mean, with higher error in 12/14 cores; its positive log1p MSE
is nevertheless 1.49% lower. The local raw-count average lowers positive
standardized MSE by 11.40% and positive log1p MSE by 10.30% relative to geometry,
in both cases with lower error in all 14 cores. However, its all-entry standardized
MSE rises to 1.2426 versus geometry's 0.9516, and zero-entry MSE rises to 0.4886
versus 0.0390. Thus it exchanges better positive predictions for much worse zero
predictions. These results do not support an unqualified "averaging is better"
claim, or establish that learned attention is oversmoothing.
Per-baseline fallback support is recorded in [fallback_frequency.csv](fallback_frequency.csv).

The constant gene mean and Huber optimum use the all-fit target distribution.
They are descriptive target-derived references, not leakage-free held-out models.
The four neural predictions were replayed and verified in the earlier evaluation;
this analysis reuses those immutable metrics and introduces no new neural fitting.

Geometry improves all-entry Huber over the Huber constant by
5.43%, compared with
11.46% over the gene mean.
Its relative error differences against each spatial reference, including signs
and all 14 paired core results, are in [paired_differences.csv](paired_differences.csv).
No single metric establishes overall superiority: lower positive error can be
accompanied by worse zero error and worse all-entry performance.

## What the loss test establishes

For each gene, the constant that minimizes Huber solves
`E[clip(prediction - standardized_log_count, -1, 1)] = 0`.
The MSE-optimal constant is the mean. On our equal-core fitted distributions,
**999/1,000 Huber-optimal constants fall below their means**;
their average standardized shift is -0.1949. The largest
stationarity error is 1e-13; every solution has
Huber risk no worse than either reference constant. This is direct mathematical
and empirical evidence of loss-induced shrinkage in the constant predictor class.
The actual geometry endpoint also has a mean signed error of -0.1715
over **all masked entries** on the standardized-log scale. This documents a downward
mean bias on that scale, beyond the outcome-conditioned positive-only diagnostic;
it is consistent with, but does not causally isolate, the objective's preference.

It does not prove that the trained graph model has reached its conditional Huber
optimum, or that Huber is wrong for every scientific objective. Huber is deliberately
robust to large residuals. Our implementation applies it to gene-standardized
log1p counts, so three choices affect the target: logarithmic compression,
inverse-gene-scale weighting, and clipping of residual derivatives beyond one
standardized unit. That objective is not designed to estimate expected raw counts
or classify whether a particular noisy count observation will be nonzero.

Even before fitting the new constants, always-zero prediction had lower Huber
than the gene mean (0.263637 versus 0.270998), while having worse standardized
MSE (1.110121 versus 0.999797). Thus the disagreement exists without a decoder,
attention, or graph. Geometry's observed positives contribute about 96.37% of
standardized squared error and 93.47% of Huber loss. **Zeros dominate entry counts,
not the measured error sums.** Neither loss share measures parameter-gradient
share; the required per-entry Jacobians and clipped residuals were not collected.

Negative bias conditional on observed positives is not alone proof of bad
calibration. For example, a correct conditional mean for a count that is one with
probability 0.1 and zero otherwise is 0.1; rounding yields zero even on the positive
outcomes. Conversely, positive-only MSE can favor overprediction on zeros. This is
why rounded-count recall and exact count matching cannot serve as the sole gate.

## Architecture, data exposure and optimization

| Endpoint | Parameters | Backbone difference | Epochs / optimizer updates |
|---|---:|---|---:|
| Original | 5,003,016 | Four independent relative-QKV blocks | 175 / 1,225 |
| Continued | 5,003,016 | Same original checkpoint lineage resumed | 300 total / 2,100 total |
| Recurrent | 2,605,680 | One entire block reused four times | 175 / 1,225 |
| Geometry | 5,134,088 | Normalized Q/K, learned scale, dimension modulation and bounded bias | 200 / 1,400 |

All use 246,063 cells, 1,000 genes, 22 morphology/imaging covariates, a 256-wide
cell state, eight heads, 1,024-wide FFNs and a **256→1,024→1,000 nonlinear decoder**.
The decoder has ordinary learned linear outputs, not a count distribution or
zero/nonzero head. Its two linear layers contain 1,288,168 parameters, about one
quarter of the geometry model. It is not visibly a tiny output bottleneck: the cell representation
is narrower, but a 256-dimensional embedding is not by itself evidence of insufficient
capacity. These sizes neither prove nor rule out a decoder bottleneck. A successful
trained probe can reveal predictive information unused by the current decoder;
an unsuccessful probe cannot establish that the embedding contains no useful information.

The graph has 55,980,536 directed edges, no self/cross-core edges, and mixed radial
shells out to 500 µm. Four message passes can mix local and broad context. Attention
averages learned value vectors, followed by residual and nonlinear updates; it is
not identical to averaging raw neighboring gene measurements. Values derive from
the whole node state (expression, mask, morphology and earlier context); edge
geometry does not enter the value projection directly.

Mask counts are uniform from 0 through 1,000 per cell, with ten masks/core/epoch.
Because the loss scores entries, heavily masked cells contribute more targets.
Measured across the fixed masks, a scored entry belongs to a cell with
**66.67% of genes hidden on average**; **19.14%**
of scored entries come from cells hiding at least 900 genes. Both senders and
receivers are masked. This is a more difficult information budget than a flat
"50% masked" description implies, and differs from whole-gene panel completion
or fully observed autoencoder reconstruction in much of the literature. Reducing
evaluation masking changes the available information; it cannot be counted as an architecture
improvement. A training-mask curriculum must be tested on the same fixed endpoints.

There are only seven optimizer updates per epoch, each averaging 20 full-core
mask-view losses. Epoch counts therefore do not match minibatch epochs in other
papers. Training losses decrease and all graph blocks receive gradients. Geometry
and recurrent logged preclip gradient maxima remain below clip norm 1, so clipping
is not restricting the recorded updates. Continued training gives small gains,
which prevents treating the early plateau as a proven optimum. Near-plateau gradient
oscillation is a possible optimization issue, not an established cause.

Existing geometry audits show across-edge modulation RMS of 0.139–0.165 and
geometry-bias variation comparable to content-score variation. This contradicts
"geometry never activated," but does not show useful routing. Existing hidden-layer
maps used fully observed inputs, so they cannot establish or exclude oversmoothing
under these heavy reconstruction masks. Attention entropy, effective neighbor count,
message/residual ratios and masked-layer contraction remain unmeasured.

## Ranked diagnosis and the experiment that could change it

| Candidate explanation | Assessment from current evidence | Discriminating test |
|---|---|---|
| Loss/scale differs from desired count reconstruction | Strongly supported mismatch; neural causal effect remains untested | Hold backbone/data/budget fixed; compare Huber-log, MSE-log, Poisson and NB count heads on common metrics |
| Masking leaves too little information for rare counts | Directly documented difficult information budget; information sufficiency unknown | Fixed 15%, 40%, 70%, 90% masks plus whole-node masking; report abundance and mask strata separately |
| Broad graph dilutes local information | Plausible from degree/range; raw averaging results alone cannot establish oversmoothing | Train cell-only, local-graph and multiscale/self-plus-context controls; then measure masked-layer contraction |
| Decoder discards useful latent information | Possible, currently weak direct evidence | Compare frozen-embedding linear/ridge and modest MLP probes with the current decoder on disjoint masks/samples |
| Insufficient optimization or update budget | Small continued gains and possible plateau oscillation; no evidence of catastrophic gradient failure | Small-subset overfit test and matched optimizer-update/LR schedules, before another long full run |
| Too little data or biological diversity | Far less atlas breadth than foundation models; no controlled scaling evidence | Within-task learning curves and grouped patient validation; do not substitute cell count for independent patients |
| Implementation/masking bug | Tested core masking, target transform, checksums and metric replay pass; no identified fault | Retain synthetic recovery, leakage tests and identity/overfit controls; do not claim every possible bug excluded |

NB would alter the output head and likelihood: decode a positive expected count
µ and a positive dispersion θ, optimize raw-count negative log likelihood, and
evaluate its mean/probabilities/intervals. The graph encoder can initially stay
unchanged. NB already assigns probability to zero; add a hurdle or extra-zero
component only if held-out calibration supports it. Library size must be known
independently of hidden targets or inferred solely from permitted observations;
the true total contains hidden-target information even under partial masking.
Log/normalized encoder inputs can coexist with raw-count
likelihood targets. NB can improve likelihood without improving positive-only MSE,
and its dispersion can absorb misspecification, so common point metrics and
calibration must accompany NLL. Raw Huber and NLL values are not comparable scores.

The highest-value next neural experiment is a matched **cell-only versus current
graph × Huber versus MSE** pilot, retaining the same standardized-log targets and
decoder, fixed inputs, equal optimizer updates, repeat seeds and patient-grouped
validation. This isolates the robust-loss choice from graph inclusion. Follow with
NB/Poisson count-head conditions on the same encoders: those jointly change the
output scale, head and likelihood, so their effect cannot be attributed to loss alone.
Use a fixed Poisson deviance on decoded count predictions as a common point metric,
NLL for probabilistic predictions on identical counts/support with full likelihood
constants, per-gene correlations, zero/positive error strata,
abundance strata, count-probability calibration and patient-level uncertainty.
If the cell-only model matches the graph, prioritize proving usable spatial signal
before adding more geometric attention parameters. A decoder probe and small-subset
overfit check cheaply separate representation limitations from optimizer/readout issues.
These proposed neural experiments were not executed as part of this CPU diagnostic.

## External reconstruction-model comparison

Exploratory literature audit, as of **2026-09-06**. This compares task designs, not a leaderboard. BAGM's supplied context is **246,063 cells, 1,000 biological probes, 14 fitted cores**, partial-entry masking, and a **5,134,088-parameter** geometry model. Local cohort/split source: `experiments/campaigns/cmp_20260905_so2_nonzero_accuracy/README.md`; parameter count comes from the source-run audit. Geometry source run: `r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf`. No external model was executed.

| Model | Scale and approach | What its evidence measures |
|---|---|---|
| scVI / DCA | Dataset-specific count models. scVI currently defaults to ZINB; DCA supports NB/ZINB and other objectives. | Distributional reconstruction and denoising; neither requires atlas-scale pretraining. [scVI API](https://docs.scvi-tools.org/en/stable/api/reference/scvi.model.SCVI.html), [DCA paper](https://www.nature.com/articles/s41467-018-07931-2) |
| scGPT | 33M normal human cells; **53M parameters reported by the CZI model card**. Cited integration example uses binned-expression masked MSE. | Representation and downstream-task scores do not quantify our reconstruction accuracy. [Author inventory](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo), [CZI card](https://virtualcellmodels.cziscience.com/model/scgpt), [example](https://github.com/bowang-lab/scGPT/blob/main/examples/finetune_integration.py) |
| scFoundation | Over 50M cells; nominal 100M parameters; asymmetric transformer with read-depth-aware masked MSE. | Downsampling/continuous-expression recovery with source/target depth indicators. Detailed preprocessing was verified in the original manuscript; published Methods were inaccessible. [Manuscript](https://yiheng-zhu.github.io/Yiheng/papers/5/scFoundation_bioRxiv_2023.pdf), [published article](https://www.nature.com/articles/s41592-024-02305-7) |
| xVERSE | April 2026 preprint; over 89M cells; dual Poisson decoder with auxiliary supervision. | Whole-gene imputation. Lung5 PCC: .4130 zero-shot, .4785 fine-tuned. Pretraining exclusion for these imputation samples was not established; no overlap finding is asserted. [Preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC13104837/) |
| STAGATE | Dataset-specific spatial graph autoencoder; 512→30 encoder. | Normalized-expression reconstruction and spatial domains, without our hidden-entry test. [Paper](https://www.nature.com/articles/s41467-022-29439-6), [MSE code](https://github.com/QIFEIDKN/STAGATE_pyG/blob/main/STAGATE_pyG/Train_STAGATE.py) |
| Bi-channel masked graph autoencoder | Preliminary 2022 workshop; 100,149 CosMx cells/960 markers; gated spatial/similarity graphs. | Closest task: at 30% corruption, log-normalized RMSE .284 versus scVI .318. At 10%, scGNN's RMSE .294 beats its .317. Different scale/split prevents comparison with BAGM. [Paper](https://openreview.net/references/pdf?id=EPYjWg1hsQ) |
| SpaIM | Dataset-specific MLP style transfer; 53 spatial datasets with scRNA references. | Whole-gene imputation: breast PCC .70±.02 versus Tangram .62±.02. Its ACC combines rankings, not exact-count matches. [Paper](https://www.nature.com/articles/s41467-025-63185-9) |

The supported inference is that **NB is one defensible count-modeling choice, not a universal requirement**. MSE and Poisson examples contradict a claim that all capable methods must use NB. They do not show that changing BAGM's loss alone would improve its predictions.

Likewise, successful dataset-specific methods show that atlas-scale pretraining is not necessary for every reconstruction task. Large foundation corpora differ in biological diversity, gene coverage, supervision and optimization, so their results cannot isolate a cell-count effect. Our cell count does not establish adequate independent-patient replication; the current fitted-core evaluation cannot answer that generalization question.

The most informative next comparison holds data, permitted inputs, masks and compute fixed while changing one component: cell-only versus spatial context, or Huber/MSE versus a count decoder. Evaluate continuous errors, positive/zero strata and detection together. Score count distributions with likelihood/calibration separately from rounded point predictions. A masked-entry benchmark also requires preprocessing and expression-derived graph construction to exclude hidden targets.

Full specifications and qualifications are in `literature.csv`; `source_ledger.json` records access dates, claim scope and source type. Published numbers are illustrative within their original studies, not thresholds for accepting BAGM or evidence of biological mechanism.


## Evidence limits and verification

This is an all-fit transductive comparison of one trained seed and one fixed mask
per core. Original and continued share a lineage; training durations differ.
All cores are shown, without selecting favorable seeds, metrics or regions. There
is no patient generalization estimate, graph-specific neural gain, uncertainty
calibration, attribution faithfulness, realistic spatial null, independent biological
support or perturbational evidence. Fourteen cores are not assumed to be fourteen
independent patients. Observed counts are noisy measurements, not latent ground truth.

The recurrent source completed training but failed artifact finalization; its frozen
checkpoint and subsequent metric replay were independently verified. Its pre-existing
checkpoint-catalog gap remains recorded. No failed run was silently relabeled.

Four GPUs were occupied by independent training, which was left undisturbed. CPU
baselines use exact original masks and checksum-bound counts/graphs. The helper and
existing metric suites passed 29 tests. Numerical acceptance includes both pilot cores,
all 14 complete receipts, risk/stationarity checks, baseline replay, support partitions,
finite values, source hashes, independent summary reconstruction and registry audit.
See [independent_verification.json](independent_verification.json), the baseline
[manifest](../../baselines/v1/manifest.json), and [source_ledger.json](source_ledger.json).

Maximum defensible conclusion: current fitted-cohort endpoints show modest,
metric-dependent reconstruction gains. The training objective demonstrably favors
lower constant predictions than MSE, and the new spatial references expose the
remaining tradeoffs. The evidence does not yet isolate a single neural failure
cause or justify calling NB, a larger decoder, or a different graph a guaranteed fix.
