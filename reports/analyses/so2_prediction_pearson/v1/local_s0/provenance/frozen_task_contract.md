# Post-hoc Pearson evaluation of existing SO2 predictions

Frozen before inference on 2026-10-15. No retraining or new checkpoint selection.

Objective: compute masked-entry Pearson correlation for the strongest completed
current count reconstruction endpoint by aggregate validation RMSE and the earlier
BAGM endpoint fitted on all 14 cores. Deliver registered checkpoint replays,
per-core and pooled metrics, a versioned descriptive report and integrity checks.

Hypothesis: existing point predictions capture variation in observed expression.
Alternatives: correlations primarily reflect average gene abundance, vary with
normalization, or do not imply accurate detection or calibrated counts. This is
a descriptive post-hoc evaluation, not a new architecture superiority test.

Endpoints fixed from existing error reports before computing correlations:

- Local Gene Residual, both seeds 0 and 1, original best-NLL epoch-30 checkpoints:
  r_20261004T072107Z_9d2b00cb_s000_f00_a01_f41786d5 and
  r_20261004T072107Z_9d2b00cb_s001_f00_a01_1caff983.
  The family has the lowest current aggregate raw-count RMSE; its training loss
  was hurdle count likelihood. No direct MAE/RMSE-trained replacement is implied.
- Historical continued Relative-QKV, seed 0, epoch-300 last checkpoint:
  r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2.
  Lowest standardized MSE among its four previously matched historical endpoints;
  Huber-trained on standardized log1p counts. One seed, not independent repeats.

The proposed later local early-stopping campaign has no completed endpoints and
does not supersede these runs. The latest GeneMAE all-cohort experiment is a
different masking study and is not silently substituted for historical BAGM.

Estimands and units: local model predicts masked observed counts in 37,367 cells
from two validation cores of one held-out donor; 12 other cores were used to fit
the model and preprocessing. Validation was used for checkpoint selection and is
not a sealed test. Historical model and normalization were fitted on all 14 cores
(246,063 cells), including those later designated validation. Its replay is
fitted-cohort reconstruction. Both use 1,000 biological genes.

Preserve each native mask and input contract. Local masks hide Uniform{1,...,1000}
genes independently in each target while clean non-self neighbor summaries remain
available. Historical masks hide Uniform{0,...,1000} genes simultaneously across
all graph cells. Preserve original seeds and checksums. These protocols do not
constitute a matched comparison, even when output scales are shared.

Report Pearson across masked genes within each cell (mean over valid cells),
across cells within each gene (mean over valid genes), and pooled across entries.
Report counts of valid/undefined correlations. Fewer than two values or variance
at most 1e-12 yields undefined correlation, not zero. Include observed zeros and
positives under the original masks. Preserve all model seeds; mean and range are
descriptive, not independent-donor uncertainty. Primary presentation is log1p
observed counts; also report local raw counts and historical standardized values.
Local log prediction is log1p of the marginal expected count, not expected log
count. Historical inverse-standardized predictions retain native negative values.

MAE/RMSE measure magnitude error; Pearson measures co-variation; neither replaces
detection precision/recall/F1/AP. No threshold is tuned in this analysis. Existing
constant/gene-mean baselines may be replayed descriptively but are not required
for this metric-addition task. No SOTA claim or direct HEIST ranking is warranted
across different data, gene panels, masks, preprocessing or evaluation splits.

Verification: reuse MaskedRegressionAccumulator and validate against independent
SciPy calculations on a fixture including masking and constant vectors. Verify
checkpoint hashes and source identity, data manifests and original mask receipts.
Replay local archived raw/log MAE and RMSE (absolute tolerance 1e-5); replay native
historical metrics within the original tolerance (2e-6). Correlations use float64
sufficient statistics. Test a first core before continuing the full historical
evaluation, aborting on verification failure or nonfinite selected predictions.

Resources: inference only, assigned idle RTX 3090 devices, one process/device,
four CPU threads each; no hyperparameter search. Require at least 2 GiB GPU and
5 GiB disk free, check a CUDA operation, preserve failed attempts. Store only
aggregate metrics and provenance, not duplicate expression matrices. Run through
supervisor; do not change existing services. Raw data and completed runs remain
immutable. Snapshot code, contract, checkpoint hashes, environment and command.

Register each evaluation against its source run before inference. Active output:
scratch/active_runs/<run>/posthoc_reports/so2_prediction_pearson/v1.
Final output: reports/analyses/so2_prediction_pearson/v1/<endpoint>, with file
checksums and _SUCCESS written after verification. Publish a cross-run report
only once all intended replays complete or failures are disclosed explicitly.

Acceptance: all three selected checkpoints have verified replayed metrics,
reproducible artifact bundles and a clear report separating held-out validation
from fitted-cohort reconstruction. Stop on failed data/mask/source/replay checks;
do not loosen a tolerance after observing an unfavorable result. No new training
and no claims of independent-test generalization or biological true-zero recovery.
