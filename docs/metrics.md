# Metrics

The versioned source of truth is `configs/schema/metrics_v1.yaml`. Metric names
use an explicit namespace such as `val/masked_huber`,
`resource/peak_vram_gb`, `graph/isolated_node_rate`, or
`mask/effective_rate`. Model family, edge-feature use, embedding dimension,
neighbor count, masking method, and dataset are parameters or tags, not
metrics.

## Current regression task

The existing benchmark is masked-expression regression. Its primary metric is
Huber loss over finite masked entries (`val/masked_huber`, lower is better).
Secondary metrics include masked MSE and MAE plus gene- and cell-wise Pearson
and Spearman summaries. Negative log likelihood, deviance, and predictive
interval calibration apply only when the model declares a valid likelihood or
interval construction; they are not fabricated to fill a schema.

Masked \(R^2\) is the percentage-style regression summary:

```text
masked_r2 = 1 - sum((prediction - target)^2)
                  / sum((target - mean(masked finite targets))^2)
masked_percent_variance_explained = 100 * masked_r2
```

Both sums use exactly the same finite masked prediction/target pairs. The
calculation is flat across those entries and remains on the declared modeled
expression scale. It is undefined when the selected targets have zero
variance. Negative values are retained: for example, `-25%` means 25
percentage points below the constant masked-target-mean reference in this
variance-explained convention. It is not a classification accuracy and is not
bounded below.

For fixed technical mask replicates, compute \(R^2\) separately for every
replicate and take the unweighted arithmetic mean within each mask mode. Derive
the reported percentage from that mean (`100 * mean(masked_r2)`); do not clip
replicate values or pool entries across replicates. The replicate table
retains each value so the final summary can be reconciled. Model-seed
predictions are still ensembled before scoring under protocols that declare
that policy. Technical masks and model seeds are not biological replicates.

No within-tolerance percentage is currently defined. Such a rate requires an
absolute-error tolerance with a scientific or assay-based rationale, expressed
on a declared scale and locked before outcomes are inspected. Introducing an
arbitrary threshold after seeing results would create a tunable percentage
that can be made favorable and must not be called generic accuracy.

An explicitly transductive, no-holdout capacity run uses
`fit/whole_node/masked_huber`. `fit/*` means that every evaluated cell belongs
to the fitted core; it is never an alias for validation accuracy and cannot be
used as generalization evidence. Such a run must not emit `val/*`, `test/*`, or
`external/*` outcomes.

Keep partial-gene, whole-node, and spatial-block results separate. For the
legacy whole-node contrast, predictions are first averaged across the locked
model seeds and the score is computed within held-out spatial blocks. Blocks,
not cells or seeds, are the resampling units. Report the complete block
distribution, paired contrast, uncertainty interval, effect size, and all seed
and failure information.

## Hybrid raw-count hurdle task

The exploratory hybrid-count task uses raw biological-probe counts as targets
and keeps zero detection distinct from positive count level. On masked entries
its primary loss is the equal mean of three terms:

```text
hybrid_loss = (balanced_detection_bce
               + balanced_positive_ordinal_bce
               + positive_standardized_log1p_huber) / 3
```

Balanced detection BCE gives equal weight to the mean loss among zero and
positive targets. Each of the six cumulative positive-state thresholds gives
equal weight to its below-or-equal and above-threshold strata before the six
threshold losses are averaged. Positive continuous Huber uses delta 1 on the
recorded per-gene standardized `log1p(count)` scale and excludes zero targets.
An absent required stratum is an error; it is not assigned a zero loss.

Eight-state exact accuracy is descriptive because the zero state dominates.
Detection balanced accuracy, positive ordinal MAE, and positive continuous
Huber are required to assess whether positive expression levels were learned.
The all-zero and all-fit per-gene references are transductive references, not
generalization estimates. Fixed mask replicates are averaged within a tissue
core before cores are aggregated with equal weight; neither masked entries nor
cells are independent biological replicates.

## Continuous hurdle-count pilot

The multiscale hurdle-count pilot removes the conflicting balanced ordinal
term and predicts one detection logit plus one positive standardized
`log1p(count)` value per gene:

```text
hurdle_loss = (balanced_detection_bce
               + positive_standardized_log1p_huber) / 2
```

The detection and positive-target strata must both be present. Positive count
states are derived with the fixed inverse transform and half-up rounded count
bins declared in the campaign contract; no fitted state threshold is allowed.
`fit/whole_node/hurdle_loss` is the primary held-in metric. Detection balanced
accuracy, positive count-state MAE, positive continuous Huber, and per-state
recall must accompany it. Overall exact accuracy remains descriptive because
the zero state dominates.

## Optional task families

Classification metrics become active only for a declared classification
outcome. The registry supports AUROC, AUPRC, sensitivity, specificity,
balanced accuracy, macro-F1, per-class precision/recall/F1, Brier score, and
expected calibration error. Threshold metrics require a threshold locked
without test leakage; AUPRC is reported with prevalence; calibration metadata
records binning.

Survival metrics become active only when the endpoint, censoring policy, and
time grid are declared. Supported fields are concordance index, integrated
Brier score, and time-dependent AUROC at predetermined time points. Do not
calculate classification or survival metrics for expression reconstruction.

## Diagnostics

Every applicable run records efficiency: training and inference duration,
samples per second, peak VRAM, parameter count, and checkpoint size.

Graph diagnostics include node/edge count distributions, isolated-node and
disconnected-component rates, mean and high-percentile degree, effective
neighbor count, and construction time. Feature diagnostics include missing
node/edge-feature rates and embedding-norm mean/standard deviation. Masking
diagnostics include requested/effective rate, masked node and edge counts, and
class-distribution shift only when a class label is relevant.

Generalization gaps must keep a consistent metric direction:
train-to-validation is validation minus train loss for loss metrics, and
validation-to-external is external minus validation loss. Define an analogous
signed convention before applying it to a maximized metric. Graph-specific
gain is reported against the prespecified self/context baseline, not inferred
from aggregate reconstruction alone.

## Variant aggregation and selection

Aggregate by `repro_id` across the prespecified seeds and folds. The standard
summary contains expected/completed/failed counts; primary mean, standard
deviation, minimum, and maximum; relevant AUPRC and Brier means when
applicable; median duration; maximum peak VRAM; and best/worst run IDs for
audit. Null values are correct for irrelevant task-family fields.

Do not rank experiments by their best seed. Compare full aggregates under the
same split, controls, and tuning budget, and show heterogeneity across
patients/samples, folds, seeds, graph choices, and relevant preprocessing.
Report effect sizes and uncertainty based on independent biological units.
Correct the prespecified multiple-testing family for exploratory gene/program
screens.

An operational composite may be named `selection_score` only when its formula,
weights, metric directions, missing-data behavior, and version are explicit.
It must not be presented as the scientific outcome. Stable, faithful,
null-calibrated, patient-replicated, independently supported, and
perturbational evidence are reported separately.
