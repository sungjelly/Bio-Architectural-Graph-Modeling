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

Keep partial-gene, whole-node, and spatial-block results separate. For the
legacy whole-node contrast, predictions are first averaged across the locked
model seeds and the score is computed within held-out spatial blocks. Blocks,
not cells or seeds, are the resampling units. Report the complete block
distribution, paired contrast, uncertainty interval, effect size, and all seed
and failure information.

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
