# SO2 target-isolated geometry-modulated hurdle-NB2, once per cell

Campaign ID:
`cmp_20260928_so2_target_isolated_geometry_modulated_hurdle_nb_once_per_cell_seed0`

Phase: planned. Outcome: pending. The design is locked, but launch remains
blocked until the checksum-bound implementation and four-GPU preflight pass
every gate in the frozen task contract.

## Question and comparison

Does replacing the ordinary NB2 observation model with a proper Bernoulli
hurdle plus zero-truncated NB2 count model improve fixed-mask validation
likelihood or zero calibration when all data exposure, routing, masking,
optimization, and stopping rules are matched?

The paired comparator is campaign
`cmp_20260926_so2_target_isolated_geometry_modulated_nb_once_per_cell_seed0`,
completed run `r_20260926T104132Z_47d710fa_s000_f00_a01_b967dc7c`. This
campaign intentionally reuses its split and preprocessing fingerprints, graph,
model seed, mask base seed and namespaces, core-order seed, target-shard seed,
and one-mask-per-target schedule. Thus the scientific variant is the
observation head and likelihood, plus the parameters necessarily added by that
head. The earlier checkpoint is not used to initialize this model.

This is still a single validation-selected seed. It can provide a matched
exploratory comparison, not an unbiased test estimate or a generalization or
causal claim.

## Data, estimand, and leakage boundary

- Training: SO2-C15 through SO2-C26, 208,696 cells in six donor groups.
- Validation: SO2-C27 and SO2-C28, 37,367 cells in one held-out donor group.
- Test: none.
- Estimand: fixed-mask, validation-selected target-isolated partial-gene raw
  count reconstruction.
- Model inputs: train-only standardized `log1p` expression and the 22 allowed
  morphology/imaging covariates.
- Loss-only targets: raw nonnegative integer biological-probe counts.
- Prohibited inputs include raw targets, target-derived totals or annotations,
  coordinates, sample identities, and expression-derived QC totals.

The existing immutable SO2 overlay is reused because its counts, split, and
training-only transformations do not depend on the observation likelihood.
The hurdle detection prevalence is computed from the twelve training cores
only; validation counts cannot influence initialization.

## Model and hurdle likelihood

The graph encoder remains four independent 256-dimensional geometry-modulated
Relative-QKV blocks with eight 32-dimensional heads, 1,024-dimensional FFNs,
and no geometry injection into values. The target-isolated two-stream routing
is unchanged:

1. A pre-masked target-query row evolves through all four graph blocks.
2. A fully observed, cell-autonomous neighbor K/V bank stays static.
3. Only incoming non-self neighbor-to-target edges are evaluated.
4. Target states are never K/V sources and context states never receive
   messages, preventing a hidden target value from leaving and returning.

The decoder has one shared 1,024-dimensional trunk and two independent
1,000-gene final projections: an NB2 mean-logit projection and a nonzero-logit
projection. One inverse-dispersion parameter per gene remains shared across
cells. The exact trainable parameter count is 6,160,088.

For a masked count `y`, let `pi` be the sigmoid nonzero probability and let
`NB2(mu, theta)` be the underlying count distribution. The normalized hurdle
model is

```text
P(y = 0) = 1 - pi
P(y > 0) = pi * NB2(y; mu, theta) / (1 - NB2(0; mu, theta)).
```

Detection BCE is evaluated on every masked entry. Zero-truncated full-constant
NB2 NLL is evaluated only on positive masked entries. Their sums form one joint
NLL, divided by the total number of masked entries; positive observations are
not reweighted to equal the zero stratum.

Detection weights start at zero. Each gene's detection bias is the logit of
the training-only Jeffreys-smoothed prevalence
`(positive_count + 0.5) / (208696 + 1)`, clamped to
`[1e-4, 1 - 1e-4]`. Likelihood arithmetic, including the zero-truncation
normalizer, runs in FP32 outside autocast.

## Matched masks and distributed optimization

Every training cell is targeted exactly once per global epoch. Each target
gets one independently generated mask with gene count uniformly sampled from
`{1,...,1000}` and positions selected without replacement. Neighbors remain
fully observed. This campaign deliberately uses the comparator's exact
training namespace `bagm.so2.target_isolated_nb.training_masks.v1`, validation
namespace `bagm.so2.target_isolated_nb.fixed_validation_masks.v1`, and base
seed `2026092601`. Equal core, epoch, and target indexes therefore produce the
same masks in the two campaigns.

The twelve training cores are shuffled without replacement using core-order
seed `2026092602` and paired into six updates. All four DDP ranks process a
disjoint contiguous target shard of each core. The two cores are staged
sequentially; the first backward is under `no_sync`, the second synchronizes,
and one optimizer step follows. For a pair, each rank scales its local joint
NLL sum by `world_size / globally_all_reduced_masked_entry_count`, so DDP's
gradient mean is exactly the pooled per-masked-entry objective.

## Validation, stopping, and retention

Every validation cell is visited once with its one fixed comparator-matched
mask. The primary metric is
`val/target_only_masked_hurdle_negative_binomial_nll`, pooled by masked entries
across both validation cores. Detection and positive-only count components are
reported separately but do not independently control model selection.

Training uses AdamW at `1e-4`, weight decay `1e-5`, global gradient clipping at
1.0, and mixed precision with FP32 likelihoods. ReduceLROnPlateau uses factor
0.5, patience 8, absolute threshold `1e-4`, and minimum LR `1e-6`. Maximum
training is 300 epochs; stopping patience starts after epoch 50 and is 25
validations at minimum delta `1e-4`.

Checkpoint selection and stopping control are separate. Any strict decrease in
the primary metric selects a new exact best checkpoint, while the `1e-4`
meaningful-improvement state controls patience. The active run keeps atomic
latest and best checkpoints. After best reload and fixed-mask replay, latest is
deleted and only the verified best remains. Epoch archives, full prediction
matrices, test artifacts, and gradient vectors are prohibited.

## Hypotheses, alternatives, and decisions

Primary hypothesis: the hurdle likelihood improves its best fixed-mask joint
validation NLL by more than `1e-4` relative to its own epoch 1 while all loss
and gradient diagnostics remain finite.

Matched-comparison hypothesis: separating zero occurrence from positive count
magnitude improves zero Brier score and/or joint validation NLL relative to the
paired ordinary-NB2 run without changing data exposure or graph routing.

Credible alternatives are that ordinary NB2 already models the zero mass
adequately, the hurdle gate overfits the zero-heavy validation objective, the
positive subset is too sparse for stable count-head learning, the truncated
normalizer destabilizes small means, or neighbor context adds no useful signal.

Implementation acceptance requires exact likelihood agreement with an
independent high-precision oracle, exact paired-mask replay, leakage invariance,
pooled DDP-gradient equivalence under unequal zero/positive counts, finite
nonzero gradients for every graph block and both output heads, checkpoint
round-trip/replay, no test artifacts, at least 2 GiB VRAM headroom per GPU, and
at least 20 GiB free disk.

The implementation is falsified by any missing/duplicate target, empty target
mask, masked neighbor, target return path, validation-dependent detection
initialization, non-normalized hurdle likelihood, denominator mismatch,
non-finite likelihood or gradient, mask drift, DDP mismatch, or failed best
checkpoint replay. A favorable single-seed validation result remains
exploratory and must be replicated before a stronger claim.

## Observability and expected outputs

Each epoch records pooled joint training and validation NLL, separately pooled
detection and positive-count components with their denominators, raw/log1p
errors for all masked entries, conditional-positive raw/log1p errors, observed
and predicted positive rates, detection Brier score, fixed-threshold confusion
counts, recall, specificity, precision and F1, theta summaries, coverage
receipts, LR, duration, throughput, and peak VRAM. The detection threshold is
fixed at 0.5 and is not validation-tuned. Preclip gradient norms and direction
cosines are recorded for the full model, each graph block, and raw theta.
Preflight separately verifies finite nonzero gradients in both final
projections. No gradient tensor is saved.

Expected final files include resolved configuration and provenance, scalar
history, per-core diagnostics, gradient-direction tables, fixed-validation
summary predictions, final/reload receipts, one best checkpoint, checksum
manifest, and `_SUCCESS`.

## Entry points and launch gate

From the repository root, the intended preflight and runner are:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/diagnostics/preflight_so2_target_isolated_hurdle_nb_ddp.py \
  --config configs/experiment/so2_target_isolated_geometry_modulated_hurdle_nb_once_per_cell_seed0.yaml

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 enqueue-experiment \
  --campaign-id cmp_20260928_so2_target_isolated_geometry_modulated_hurdle_nb_once_per_cell_seed0 \
  --config configs/experiment/so2_target_isolated_geometry_modulated_hurdle_nb_once_per_cell_seed0.yaml \
  --priority 0 --max-attempts 1 --gpu 0,1,2,3
```

The queue derives the four-rank `torchrun` command; direct unregistered launch
is not an accepted production path. The authoritative immutable design is
[`frozen_task_contract.yaml`](frozen_task_contract.yaml).
