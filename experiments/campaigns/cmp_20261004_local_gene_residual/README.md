# Local gene residual count reconstruction

Phase: complete. Outcome: negative for the prespecified material NLL gain;
measured reconstruction improvements are supported on this validation split. This is an exploratory design informed
by the previous BAGM results, not a confirmatory or independent-test study.

## Task contract

Objective: implement a simple new count reconstruction model and evaluate it
against the current SO2 BAGM endpoints on the identical donor split and fixed
target masks. The user prioritizes count reconstruction without restricting the
objective to MAE. Deliver model/data/training code, leakage and likelihood tests,
registered immutable runs, indexed replayed checkpoints, and an honest comparison.

Core idea: a shallow masked-expression MLP predicts each cell; a small gene-wise
residual uses same-gene expression and detection frequency among up to 16 nearest
non-self cells within 75 micrometers. Neighbor pooling uses only clean neighboring
observations. No attention, contextual source encoder, or target return path.
The readout is a Bernoulli hurdle with zero-truncated NB2 positive counts, matching
the current hurdle observation model. A self-only ablation disables local inputs.

Hypothesis: preserving local gene evidence while simplifying the encoder improves
count likelihood and/or point reconstruction over the existing graph model.
Alternatives: same-cell coexpression is sufficient; local neighbors add noise;
observed counts are largely unpredictable; or optimization/selection explains
apparent differences. Local-versus-self comparisons test the spatial contribution.
A failed superiority criterion is a valid negative result, never a reason to
silently change the primary metric or omit a seed.

Estimand: target-isolated partial-gene reconstruction using visible target genes,
22 permitted morphology/imaging covariates, and clean non-self neighbor expression.
Reuse the training-only SO2 NB overlay: 208,696 cells from six donor groups in 12
cores; validation 37,367 cells from one held-out donor in two cores. No test split.
No identifiers, hidden target totals, expression-derived cell labels, or hidden
target measurements may enter inputs. Data remain local and raw inputs immutable.

Use the exact inherited per-target Uniform{1,...,1000} masking namespaces/base
seed, one target visit per epoch, and fixed validation masks. Report coverage and
mask hashes. Fresh random target minibatches within shuffled cores are allowed
and recorded; they differ from old paired-core optimization. All variant/seed
comparisons share masks, optimizer schedule, target exposure, and evaluation.

Primary metric: pooled full-constant hurdle count NLL, lower better. Secondary:
marginal-mean raw/log1p MAE and RMSE, marginal predictions on true-zero and
true-positive subsets, conditional-positive mean errors (clearly separated),
detection BCE/Brier/AP, and fixed-threshold precision/recall/F1. Compare raw MAE
with always-zero, and likelihood with train-fitted per-gene hurdle constants.
Do not call a mostly-zero point predictor a reconstruction improvement solely
because it beats mean predictions on MAE. Retain per-core and mask-severity scores.

Historical references: NB selected epoch 193 (NLL .3476011875, MAE .2235333539,
RMSE 1.1815561763); hurdle selected epoch 164 (NLL .3282417527, MAE .2259919385,
RMSE 1.1684848057). Both are seed 0 with different stopping exposure. Historical
comparisons establish endpoint performance on this validation split, not a pure
architecture effect or independent generalization. Latest BCE is a detection-only
reference and cannot supply count reconstruction metrics.

Pilot: registered two-epoch resource/correctness execution after focused tests.
Then four independent fixed-budget runs: local/self-only crossed with model seeds
0/1, initially 32 epochs, batch 1024, AdamW LR .001, weight decay .00001, dropout
.1, clipping 1.0. Fixed cosine decay to .0001 avoids adapting LR to outcomes.
Select each checkpoint by strict lowest validation NLL; also report the mean of
epochs 29-32, all seeds and per-core directions. Do not extend an unfavorable
budget without an explicit prospective amendment. At fixed budget, require mean
selected NLL across both local seeds at least .002 below historical hurdle NLL
and favorable NLL in both seeds; raw RMSE must not worsen by more than 1% in
either seed. This is a practical exploratory criterion, not significance.
Graph/local gain additionally requires local to beat matched self-only in both
seeds. Improvement in other metrics is described separately if primary fails.

Correctness gates: hidden-value invariance, explicit self exclusion, no-neighbor
finite behavior, same-cell visible input usability, exact masks/coverage, proper
hurdle likelihood, finite gradients, positive-count/marginal metric semantics,
atomic checkpoints and selected-checkpoint reload/replay. Stop on failed gates,
OOM/nonfinite states, GPU free memory below 2 GiB, disk below 5 GiB, or host-memory
danger; preserve failed attempts rather than blindly retrying.

Resources: four currently idle RTX 3090s; pilot on one device, then one independent
variant/seed per available device. Inspect actual framework visibility and pilot
throughput/VRAM/RSS before full runs. New services use supervisor; existing jobs
remain untouched. Active files stay in shared BAGM scratch and final bundles in
shared BAGM artifacts through the project lifecycle. Source is the existing
analysis worktree. Store hashes/config/split/data/environment/command and changes.

Acceptance: implementation and tests pass, pilot gates pass, all four registered
runs complete or failures are disclosed, selected checkpoints replay, reports and
catalogs validate. Maximum claim: simpler-model validation performance on one
held-out donor, not independent patient replication, biology, or causality.

Exact implementation/reproduction commands and final outcomes will be recorded
below before task completion.

## Implementation and reproduction

Source worktree: `/workspace/BAGM-relative-qkv-analysis`; shared inputs and run
storage: `/workspace/BAGM`. `ops/supervisor/bagm-local-gene-residual.sh` sets the
project path overrides, activates the existing Python executable, and assigns
one physical GPU. The deployed identical wrapper is under
`/opt/supervisor-scripts/`. Run long jobs with the supervisor configurations
`/etc/supervisor/conf.d/bagm-local-gene-*.conf`; they do not auto-retry failures.

The independent model is `src/spatial_benchmark/local_gene_residual.py`.
`local_gene_data.py` builds verified non-self summaries and inherited masks;
`local_gene_metrics.py` pools count/detection metrics with explicit denominators.
`run_local_gene_residual.py` manages training, selection, replay, and publication.
The trunk is 2022 → 512 → 512 with GELU/dropout 0.1. Two 1000-gene heads predict
detection logits and NB means; a 1000-gene dispersion vector is learned. Each
head receives two trainable per-gene local coefficients (4000 in total). All
parameters including frozen local coefficients in the self control: 2,329,432;
trainable self-only parameters: 2,325,432. No position or density channel is used.

Exact run commands (supervisor executes the corresponding commands):

```bash
/opt/supervisor-scripts/bagm-local-gene-residual.sh 0 --profile smoke --variant local --seed 0
/opt/supervisor-scripts/bagm-local-gene-residual.sh 0 --profile full --variant local --seed 0 --smoke-run r_20261004T071603Z_178fc589_s000_f00_a01_254a98de
/opt/supervisor-scripts/bagm-local-gene-residual.sh 1 --profile full --variant local --seed 1 --smoke-run r_20261004T071603Z_178fc589_s000_f00_a01_254a98de
/opt/supervisor-scripts/bagm-local-gene-residual.sh 2 --profile full --variant self --seed 0 --smoke-run r_20261004T071603Z_178fc589_s000_f00_a01_254a98de
/opt/supervisor-scripts/bagm-local-gene-residual.sh 3 --profile full --variant self --seed 1 --smoke-run r_20261004T071603Z_178fc589_s000_f00_a01_254a98de
```

The immutable protocol is `frozen_task_contract.md`; updating this README does
not change the frozen acceptance criteria. The pilot completed two epochs in
195 seconds including input loading, with 0.843 GiB peak allocated VRAM and
successful selected-checkpoint replay. Its NLL is not a full-run outcome.
Initial likelihood constants use training prevalence and positive means with
fixed theta=1; they are not maximum-likelihood fitted dispersion estimates.

Verification:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /venv/main/bin/python -m pytest -q -p no:cacheprovider tests/unit/spatial_benchmark/test_local_gene_residual.py tests/unit/spatial_benchmark/test_local_gene_data.py tests/unit/spatial_benchmark/test_local_gene_metrics.py tests/unit/spatial_benchmark/test_hurdle_negative_binomial.py
```

All 40 initial focused tests passed. The maximum batch size is 1024: each core is partitioned into balanced batches
with `numpy.array_split`, yielding 209 updates per complete epoch. Full runs are
paired by masks, target order,
initial seed, optimizer, and exposure; historical BAGM comparisons have different
optimization and stopping exposure. Float32-versus-bfloat16 likelihood checks
run before training and on the selected checkpoint. Detection AP/AUROC use 4096
probability bins. Exact individual row predictions are intentionally not exported;
the prediction table contains aggregate per-core sufficient summaries only.


## Loading a selected model

Checkpoints are trusted local experiment artifacts and include optimizer/RNG
state. The archive checksum and checkpoint catalog identify each selected file.
For inference, load its architecture and model state:

```python
import torch
from spatial_benchmark.local_gene_residual import LocalGeneResidualCountModel

saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
settings = saved["config"]["model"]
model = LocalGeneResidualCountModel(
    1000, covariate_dim=22, hidden_dim=settings["hidden_dim"],
    dropout=settings["dropout"], use_local=settings["use_local"],
)
model.load_state_dict(saved["model"])
model.eval()
with torch.no_grad():
    output = model(expression, hidden_mask, covariates,
                   local_log_mean, local_positive_fraction)
count_estimate = output.marginal_mean
probability_nonzero = output.nonzero_probability
positive_count_estimate = output.positive_conditional_mean
```

`expression` and the 22 covariates must use the stored training-only transforms;
`hidden_mask=True` marks values to reconstruct. Neighbor features must exclude
the target and use the same panel order, radius, and nearest-neighbor cap. The
marginal mean is the overall count prediction; the positive conditional mean is
only conditional on a count being nonzero. `load_local_gene_data` reconstructs
all these inputs from the pinned local SO2 artifacts. Hidden values are removed
inside the model before any arithmetic. The target-isolated contract permits
clean neighboring cells; this checkpoint is not validated for simultaneous
missing genes across an entire spatial block.

Full-run archive classification uses the catalog bucket `validation_confirmation`.
This is a storage/lifecycle label; the study remains exploratory on a reused
validation donor and provides no independent confirmation or patient replication.
The curated conclusion uses the explicit stage `exploratory_validation`.


## Infrastructure checks during execution

The independent trainer review added seven passing tests in
`test_local_gene_runner_review.py` for pooled/stratified evaluation, atomic
checkpoint failure handling, paired variant identity, and pilot provenance gates.
All active source hashes remained identical to the pilot.

`spatial_benchmark doctor` confirms SQLite integrity and configuration validity,
but its global status is not clean: two older registered six-core dataset/split
paths point into this worktree instead of shared data, one historical checkpoint
is unindexed, and its generic 25 GiB disk floor exceeds the approximately 18 GiB
available. These pre-existing conditions are distinct from the campaign's frozen
5 GiB free-disk gate. Active full-run data are individually checksum verified;
new checkpoints are explicitly indexed. No legacy records or files were altered.
The raw doctor output is retained in shared scratch as
`local_gene_doctor_20261004.json` and will be included in the final verification
record. The held global worker lock is expected during these supervised jobs.


## Full-run allocation

| Variant | Seed | GPU | Immutable run ID |
|---|---:|---:|---|
| Local | 0 | 0 | `r_20261004T072107Z_9d2b00cb_s000_f00_a01_f41786d5` |
| Local | 1 | 1 | `r_20261004T072107Z_9d2b00cb_s001_f00_a01_1caff983` |
| Self only | 0 | 2 | `r_20261004T072106Z_dd0aa417_s000_f00_a01_e0116c2b` |
| Self only | 1 | 3 | `r_20261004T072107Z_dd0aa417_s001_f00_a01_ea627eca` |

Report command, requiring all four completed verified bundles:

```bash
export BAGM_ROOT=/workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export BAGM_REPORT_ROOT=/workspace/BAGM/reports
export BAGM_RESULT_ROOT=/workspace/BAGM/results
export BAGM_CACHE_ROOT=/workspace/BAGM/cache
export BAGM_EXPORT_ROOT=/workspace/BAGM/exports
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_local_gene_residual.py --run-ids r_20261004T072107Z_9d2b00cb_s000_f00_a01_f41786d5 r_20261004T072107Z_9d2b00cb_s001_f00_a01_1caff983 r_20261004T072106Z_dd0aa417_s000_f00_a01_e0116c2b r_20261004T072107Z_dd0aa417_s001_f00_a01_ea627eca --version v1
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate --verify-payloads --verify-sources
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog --check
```

The report version is immutable; use a fresh version name to regenerate it.
The detailed report retains all four runs, selected and terminal-window scores,
per-core directions, severity strata, and the actual historical NB/hurdle sources.


The full CPU repository suite completed in 467 seconds: **2109 passed, 35 failed,
4 skipped**. All 35 failures concern unavailable pre-existing inputs: locked
adjacency/MyJJu campaign materializations, the six-core clinical mapping,
stage-zero prepared geometry, and the absent audited MyJJu external script.
The failures are preserved in `local_gene_full_suite_20261004.log`; they are not
reported as a clean repository test run. They do not involve this model's source
or data artifacts. No historical input or test expectation was changed to make
these checks pass.


## BAGM design observations motivating this comparison

These are code-supported limitations or hypotheses, not proof of the cause of
BAGM's validation error.

- Geometry changes attention scores but never message values
  (`geometry_modulated_relative_qkv_graph_transformer.py`,
  `_GeometryModulatedAttentionMixin`). If all incoming values are identical, a
  normalized weighted average returns the same value regardless of routing.
  Geometry therefore cannot express every geometry-dependent count effect.
- All four target-isolated blocks read the same clean, cell-autonomous source
  bank (`target_isolated_negative_binomial.py`, `source_bank_evolves=False`).
  They refine the target repeatedly; they do not provide four-hop propagation.
  This protects hidden-target isolation but constrains contextual interactions.
- Early compression into the 256-dimensional BAGM latent state may discard
  gene-specific evidence. This is a capacity hypothesis requiring ablation,
  not an established bug. The new direct same-gene residual specifically avoids
  encoding all local genes through a shared latent bottleneck.
- Count-likelihood training and reporting MAE of a predicted mean optimize
  different decisions. Only 8.1178% of scored validation entries are positive;
  an always-zero predictor has MAE 0.151251, below the historical count models.
  That does not make zero a useful probabilistic count reconstruction model.
- Historical count models use six optimizer updates per epoch; this experiment
  uses 209. Architecture, minibatch ordering, and learning-rate schedule can all
  explain an endpoint difference. The matched self-only controls isolate the
  local correction, not every change relative to BAGM.

The new model also has deliberate limits: one fixed neighborhood scale, linear
same-gene spatial corrections, and no learned cross-gene spatial message mixing.
It cannot establish multi-hop signaling or biological causality. A useful next
validation after a favorable endpoint is a locked comparison on additional
held-out donors, including spatial-block masking when nearby clean counts would
not be available at deployment. If the local gain disappears there, local
copying or segmentation effects remain plausible explanations.


Consolidated focused verification: **60 passed** in 6.96 seconds, with two
upstream TorchScript deprecation warnings. This includes the four initial test
modules, `test_local_gene_runner_review.py`, and
`test_report_local_gene_residual.py`. Checksummed logs and the doctor receipt
are finalized under `/workspace/BAGM/reports/local_gene_residual/verification_v1`.
The recorded supervisor definitions are also available in
`ops/supervisor/bagm-local-gene-residual.conf`.


## Final outcome

All four 32-epoch runs completed successfully; the lowest-NLL checkpoint was
at epoch 30 for every run. All selected checkpoints passed replay and artifact
verification, and all four are indexed with checksums and retention metadata.
No failed attempts occurred. Both seeds and both cores show a local-versus-self
NLL improvement. The full practical superiority criterion **did not pass**:
mean NLL improved by 0.001591, less than the frozen 0.002 margin.

| Model | Mean selected NLL | Mean raw MAE | Mean raw RMSE | Parameters |
|---|---:|---:|---:|---:|
| Historical BAGM NB, seed 0 | 0.347601 | 0.223533 | 1.181556 | 5,135,088 |
| Historical BAGM hurdle, seed 0 | 0.328242 | 0.225992 | 1.168485 | 6,160,088 |
| New local model, seeds 0/1 | 0.326651 | 0.223391 | 1.100785 | 2,329,432 |
| New self-only control, seeds 0/1 | 0.328377 | 0.229246 | 1.131437 | 2,329,432 total; 2,325,432 trainable |

Against the historical hurdle endpoint, the local model reduces RMSE by 5.79%,
MAE by 1.15%, and NLL by 0.48%, with 62.19% fewer parameters. Both local seeds
improve all three of these metrics versus hurdle. The NB reference has lower
MAE than local seed 1 by 0.000024; do not describe universal superiority over
every historical metric and seed. Mean terminal epochs 29–32 NLL is 0.326765
and 0.326808 for local seeds 0/1, respectively, also below historical hurdle.
The local model took 23.9 and 21.8 minutes per seed including loading; these
are observed runtimes, not a controlled architecture-only speedup estimate.

The appropriate conclusion is a smaller model with a meaningful RMSE gain and
a modest likelihood/MAE improvement on the current validation split. The more
ambitious predeclared NLL improvement hypothesis is negative at this budget.
There is no independent test or patient-replication claim. Do not automatically
extend the budget or change the criterion after seeing these outcomes.

Final artifacts:

- Detailed figure-led report: `/workspace/BAGM/reports/local_gene_residual/v1/report.md`.
- Full metrics, per-core/mask-severity CSVs and manifest: the same version directory.
- Focused/full test logs and doctor receipt: `/workspace/BAGM/reports/local_gene_residual/verification_v1`.
- Curated result: `/workspace/BAGM/results/predictive_benchmark/local_gene_residual/res_so2_local_gene_count_reconstruction/result.yaml`.
- Selected checkpoints: `checkpoints/best.ckpt` in the immutable run bundles
  listed above, under `/workspace/BAGM/artifacts/runs/2026/10/`.

The learning-curve figure was visually checked. It shows shared epoch-to-epoch
variation across both paired seeds, and consistent local gain relative to self.
All existing user changes remain preserved; no historical successful run bundle
was altered. The final result records the unsuccessful primary-margin criterion
and the favorable secondary reconstruction outcomes together.
