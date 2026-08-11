# Same-gene robustness multiverse v1

Status: `frozen_preoutcome`

Phase: `planned`

Outcome: `pending`

This campaign is a preregistered, post-hoc robustness study of the existing
same-gene nonlinear result. It is not an independent replication: the two
slides and the antecedent outer-test outcomes have already been observed. No
run from this new campaign is authorized until `frozen_task_contract.yaml` is
filled with every prepared-data identity, reviewed, and changed from
`draft_preoutcome` to `frozen_preoutcome`. Its immutable SHA-256 is recorded
in the launch manifest; the contract does not contain an impossible hash of
itself.

This document authorizes the V0-V6 core phase (seven technical pilots followed
by 140 production fold processes) and one zero-training post-core secondary:
the exact deterministic 10,000-draw gene-label null declared below. Null banks,
leave-one-slide-out, distance, native-eligibility, and optimization-extension
work require their own frozen authority and matching pilots.

## Questions

The campaign separates three questions that must not be conflated:

1. Is same-name Jacobian-magnitude enrichment stable to graph construction,
   cell QC, expression normalization, and optimizer seed?
2. Does the original strict row-exclusivity gate continue to fail under those
   prespecified changes?
3. Is the conclusion that a 12-epoch budget did not explain the original
   failures stable across optimizer seeds?

The campaign must not select the variant that gives the largest effect. Every
listed core variant, seed, fold, and failed run is reportable.

## Shared training protocol

- Model: the existing additive neighbor MLP, unchanged.
- Optimizer: AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Batch size: 4096.
- Epoch candidates: 12, 24, 48, 96, 192, using one continuous tuning
  optimizer trajectory.
- Selection: validation component-equal MSE, with the earlier epoch winning a
  tie.
- Refit: reinitialize and train on every non-test component for the selected
  epoch count.
- Anchor: save epoch 12 from the same final-refit trajectory.
- Seed bases: 20260810, 20261810, 20262810, 20263810, and 20264810.
  Fold `f` uses `seed_base + f`, yielding 20 distinct execution seeds rather
  than reusing an RNG stream across nominal seed-by-fold replicates.
- Outer folds: the existing four frozen geometry-component folds.
- Validation fold: `(outer_fold + 1) mod 4`.
- The canonical prediction split for production is `test`, never
  `validation`.
- The receiver's RNA vector, RNA total, cell type, or any quantity calculated
  from them is never a model input. Conditional-residual variants change the
  response estimand; they do not expose those quantities as predictors.

All models predict all 1,000 panel probes. Primary Jacobian summaries use the
already frozen common 932-gene axis and its recorded mask hash so a variant
cannot gain or lose genes after its result is seen.

## Core variant matrix

Each row uses five seeds and four outer folds: 20 four-arm fold processes per
variant. The four result keys are `morphology_only`, `observed_near`,
`observed_annular`, and `within_fov_permuted_near`; the last is the
preprocessing-matched permuted-near negative control.

| ID | Tier | Frozen change | Processes |
|---|---|---|---:|
| V0 | primary | Antecedent-exact observed within-FOV arms and primary cohort, with a corrected degree-preserving permuted control | 20 |
| V1 | primary | Permit edges across FOVs only within the same frozen geometry component | 20 |
| V2 | primary | Keep QC-pass cells as both receivers and graph source nodes; rebuild graph | 20 |
| V3 | primary | Replace log1p counts with log1p CP10k using the 1,000-gene panel total | 20 |
| V4 | primary | Remove train-fitted panel-library-size effects from target and source profiles | 20 |
| V5 | primary | Cross-FOV-within-component graph + full QC-pass + log1p CP10k | 20 |
| V6 | mechanistic secondary | Remove train-fitted vendor cell-type and library-size effects | 20 |

The core contains 140 scientific fold slots, an initial 35 four-GPU waves,
and 560 selected successful arm fits. A technical failure may add a separately
materialized immutable recovery attempt; it does not add a scientific slot,
and every such attempt must be declared, registry-matched, and reported.
The prepared root also contains `A0`, a 4.7 GB component-graph/CP10k/all-cell
array set created before the phase split. Its bytes are bound by the root
manifest, but this contract does not authorize an A0 model fit or include A0
in any core classification.

V0 is not described as a fully exact replication of every antecedent input.
Data-only comparison found expression, morphology, FOV, geometry group, fold,
QC, observed near/annular degrees and means, and the primary eligible mask to
be exact. The old permuted graph, however, differed from observed-near degree
at 1,779 SO1 nodes and 2,407 SO2 nodes. V0 deliberately rebuilds that negative
control so permuted degree equals observed-near degree and receiver collisions
are absent. The rebuilt native graph eligibility adds 36 SO1 and 17 SO2
receivers, but those additions are not used: V0's primary receiver cohort stays
exactly the antecedent cohort. Consequently, observed-near comparisons retain
the antecedent inputs while near-versus-permutation is a corrected-control
comparison.

### Cross-FOV graph rule

V1 and V5 construct distance edges among cells in the same frozen geometry
component, regardless of original FOV. They retain the original distance
bands, `k=12`, deterministic distance/row tie break, zero-distance exclusion,
and minimum degree 4. Any edge crossing a geometry component is fatal. The
permutation mapping remains within the source's original FOV to retain FOV
batch composition. It is a receiver-collision-free bijection that uses exact
matching to minimize fixed source states, uses a perfect derangement whenever
one exists, and never removes an observed edge slot; therefore permuted and
observed near degrees are exactly equal. Fixed-state counts and affected FOVs
are reportable. The pre-outcome feasibility audit found a Hall violation among
the 11 active QC source states in SO2 FOV 245: one fixed state is unavoidable
there, while all receiver collisions remain avoidable (237,945 of 237,946 SO2
QC source states change).

### Full QC-pass rule

V2 and V5 start from raw cell-ID-matched cells satisfying the vendor
`qcCellsPassed` field. QC-failed cells are excluded as receivers and source
nodes before graph construction. Geometry components and their folds remain
fixed. The primary comparison retains the antecedent eligible receivers and
intersects them with QC pass where applicable. Native eligibility recomputed
from each rebuilt graph is a separately authorized follow-up, not part of the
core production gate.

### CP10k rule

For raw counts `c_ig`, V3 and V5 use

`log1p(10000 * c_ig / max(sum_g(c_ig), 1))`.

The sum is over the same ordered 1,000-probe panel. The same transform is
applied to receiver targets and source profiles before neighbor aggregation.
Training-only target scale statistics are then applied exactly as in V0.

### Library-size residual rule

V4 defines `x_ig = log1p(c_ig)` and `l_i = log1p(sum_g(c_ig))`. For every
gene, a component-equal weighted regression `x_ig = alpha_g + beta_g*l_i +
epsilon_ig` is fitted using tuning-train only for validation selection and
final-train only for final test evaluation. The target is `epsilon_ig`; the
neighbor feature is the mean source residual. Receiver library size is not an
input. This is a conditional-expression estimand, not absolute-RNA
reconstruction.

### Cell-state residual rule

V6 fits a component-equal weighted, training-only regression containing the
12-level vendor InSitu cell-type fixed effect and `log1p` panel total. Target
and source profiles are the resulting residuals. Every frozen level must occur
in each training fit; a missing training level fails closed. The preparation
audit found all 12 levels in every relevant split. The vendor type is derived
from RNA, so V6 is a conservative mechanistic diagnostic and not an
independent covariate adjustment.

## Frozen primary gates

The original seven-gate vector is applied without changing a threshold:

1. near versus morphology relative component-equal MSE gain at least 0.02 and
   near favored in at least 3/4 folds;
2. near versus matched permutation gain at least 0.01 and near favored in at
   least 3/4 folds;
3. absolute same-name diagonal/off-diagonal ratio at least 2.0;
4. row top-1 fraction at least 0.25 and row top-1% fraction at least 0.50;
5. observed/permuted median absolute diagonal ratio at least 1.25, with at
   least 3/4 fold ratios at least 1.25;
6. fold signed-diagonal median Spearman at least 0.70 and at least 0.75 of
   eligible genes having a consistent sign in at least 3/4 folds;
7. all technical controls pass.

For a variant, the signed consensus Jacobian is the equal mean of its 20
seed-by-fold matrices before absolute values are taken. Prediction consensus
first averages per-component losses over seeds. A gate is a `robust_pass` only
when the consensus and at least 4/5 seed-specific four-fold aggregates pass. A
gate is a `robust_gate_failure` only when the consensus and at least 4/5 seeds
fail. Anything else is `seed_sensitive`.

The preprocessing-level conclusion covers V0-V5 only. It is
`robust_across_preprocessing` only if every one of V0-V5 has the same
three-way classification. No majority vote and no best-variant selection are
allowed. V6 is reported as a mechanistic sensitivity.

## Budget attribution

Budget attribution is primary only in V0. All 40 morphology/near
seed-by-fold tuning trajectories must be saturated. A trajectory is saturated
when its selected epoch is below 192, or when
`(MSE_96 - MSE_192) / MSE_96 <= 0.001`. Any unsaturated trajectory makes the
budget result inconclusive and triggers a separately frozen 384-epoch
extension only for affected trajectories.

The selected model explains the prediction failure only if it passes the
original 2% gate, the lower 95% paired component-bootstrap bound for
`gain_selected - gain_anchor12` is above zero, and at least 4/5 seeds have
selected near better than anchor near in at least 3/4 folds. Row failure is
explained only if the selected consensus passes both strict row thresholds,
both exceed anchor 12, and at least 4/5 seeds improve in at least 3/4 fold
matrices. Strong budget explanation requires both attribution gates.

## Authorized deterministic post-core secondary

Only after all 140 core jobs have passed verification, the same immutable
aggregate computes a two-family, 10,000-draw gene-label null from the V0
selected observed-near total Jacobian. The matrix is the signed equal mean over
five seeds and four folds, restricted on both axes to the frozen common 932
genes. Source-column labels are deranged relative to fixed target rows using
base seed 20261019. One family deranges all 932 labels; the second deranges
within strata formed by crossing stable marginal rank deciles of V0 final-train
nonzero prevalence and final-train target standard deviation. Singleton strata
are deterministically attached to the nearest nonsingleton stratum before
derangement.

The fixed statistics are median absolute diagonal, row top-1 fraction, and row
top-1% fraction (10 of 932), with add-one upper-tail p-values. This is a
deterministic relabeling of already verified core aggregates, creates no model
fit or GPU resource uncertainty, and therefore needs no training pilot. It is
secondary enrichment evidence only: it cannot change or rescue any frozen
strict gate and supports no per-gene inference.

## Separately authorized follow-ups

The remaining scientific ideas remain prespecified, but this contract does not
launch them. Native graph eligibility, two 99-member permutation-null banks,
two-direction/two-graph leave-one-slide-out, three distance bands, and any
384-epoch optimization extension each require a separate phase contract. That
authority must bind its exact prepared arrays or mappings, sources, seeds, job
inventory, and matching technical pilots before that phase can run. A core
pilot receipt cannot authorize a training follow-up.

## Fail-closed pilots

Seven pilots are required before core production: exactly one for each V0-V6.
Pilots use seed 20260810, fold 0, capped tuning train 24,000, validation
12,000, generated-but-never-evaluated test 12,000, and a forced 192-epoch
capped refit for resource measurement.

Pilots must not evaluate, serialize, print, or register an outer-test
scientific effect. All pilots must pass finite-output, split isolation,
receiver-input exclusion, exact Jacobian/autograd/finite-difference controls,
an actually executed identity oracle, graph-specific invariants, checkpoint
GPU replay, canonical split labeling, source/config/data hash verification,
frozen live-environment verification, peak VRAM at most 20.5 GB, and projected
full-fold runtime at most 0.25 hours.
All 140 core scientific slots are forbidden until all seven immutable pilot
receipts pass.
Those receipts authorize no other phase.

The source-bound `environment_lock.json` is an execution gate, not descriptive
metadata. Materialization and analysis require four visible homogeneous RTX
3090 GPUs; each pilot/full child and marker replay requires exactly one visible
GPU plus a CUDA smoke test. CPython, Python, CUDA, and every declared package
version must match exactly; GPU name and compute capability must match and each
device must have at least 25,000,000,000 bytes. Missing packages, version drift,
GPU-count/identity/capability/memory drift, a failed smoke test, or a lock/source
hash change blocks execution. Each run archives the full one-GPU receipt and
binds its receipt SHA and lock SHA in technical controls. Four-GPU offline
analysis validates that archived observation against the lock without rerunning
the one-GPU gate in its incompatible visibility context.

## Execution and publication

1. Correct the canonical outer-test metadata path and hard-crash recovery.
2. Build and hash every core prepared variant using only data/shape/graph audits;
   do not train or expose scientific outcomes.
3. Replace every prepared-identity placeholder, verify manifest, integrity,
   processed, variant-spec, raw, split, base, and root bindings, then freeze
   the contract and record its hash in the launch manifest.
4. Run and verify all seven core pilots without viewing test effects.
5. Fill exactly 140 V0-V6 scientific slots with verified successful runs; do
   not compare outcomes until every slot succeeds. A failed process may be
   replaced only by the next contiguous, explicitly materialized recovery
   attempt, and every attempt remains in the registry and final report.
6. Recompute JSON/NPZ/CSV summaries from checkpoints and component records.
7. After complete core verification, compute the declared deterministic
   gene-label null without additional training.
8. Publish one immutable aggregate report containing every attempt, every
   prespecified variant, and that secondary null.

### Exact entry points

Run every command from `/workspace/Bio-Architectural-Graph-Modeling`. The
materialized paths below are operational state; the frozen contract, source
authorities, run bundles, registry records, and final report are the durable
scientific record.

Build the source/contract launch authority:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/train/materialize_same_gene_robustness.py \
  build-launch \
  --contract experiments/campaigns/cmp_20260810_same_gene_robustness_multiverse_v1/frozen_task_contract.yaml \
  --source-list experiments/campaigns/cmp_20260810_same_gene_robustness_multiverse_v1/source_authorities.json \
  --output state/materialized/same_gene_robustness_v1/launch_manifest.json
```

Materialize and run the seven technical pilots:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/train/materialize_same_gene_robustness.py \
  build-plan --profile pilot \
  --variant-root V0=data/processed/same_gene_robustness_v1/variants/v0_within_fov_log1p_all \
  --variant-root V1=data/processed/same_gene_robustness_v1/variants/v1_component_log1p_all \
  --variant-root V2=data/processed/same_gene_robustness_v1/variants/v2_within_fov_log1p_qc_induced \
  --variant-root V3=data/processed/same_gene_robustness_v1/variants/v3_within_fov_cp10k_all \
  --variant-root V4=data/processed/same_gene_robustness_v1/variants/v4_train_only_library_residual \
  --variant-root V5=data/processed/same_gene_robustness_v1/variants/v5_component_cp10k_qc_induced \
  --variant-root V6=data/processed/same_gene_robustness_v1/variants/v6_train_only_cell_type_library_residual \
  --launch-manifest state/materialized/same_gene_robustness_v1/launch_manifest.json \
  --output-dir state/materialized/same_gene_robustness_v1/pilot/jobs \
  --plan state/materialized/same_gene_robustness_v1/pilot/plan.json

PYTHONPATH=src /venv/main/bin/python scripts/train/launch_same_gene_robustness.py \
  --plan state/materialized/same_gene_robustness_v1/pilot/plan.json \
  --ledger state/materialized/same_gene_robustness_v1/pilot/ledger.json

PYTHONPATH=src /venv/main/bin/python scripts/train/materialize_same_gene_robustness.py \
  build-receipts \
  --pilot-plan state/materialized/same_gene_robustness_v1/pilot/plan.json \
  --output-dir state/materialized/same_gene_robustness_v1/pilot/receipts
```

Only after all seven receipts verify, materialize and run attempt 1 for the
140 production slots:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/train/materialize_same_gene_robustness.py \
  build-plan --profile full --attempt 1 \
  --variant-root V0=data/processed/same_gene_robustness_v1/variants/v0_within_fov_log1p_all \
  --variant-root V1=data/processed/same_gene_robustness_v1/variants/v1_component_log1p_all \
  --variant-root V2=data/processed/same_gene_robustness_v1/variants/v2_within_fov_log1p_qc_induced \
  --variant-root V3=data/processed/same_gene_robustness_v1/variants/v3_within_fov_cp10k_all \
  --variant-root V4=data/processed/same_gene_robustness_v1/variants/v4_train_only_library_residual \
  --variant-root V5=data/processed/same_gene_robustness_v1/variants/v5_component_cp10k_qc_induced \
  --variant-root V6=data/processed/same_gene_robustness_v1/variants/v6_train_only_cell_type_library_residual \
  --variant-receipt V0=state/materialized/same_gene_robustness_v1/pilot/receipts/V0.pilot-receipt.json \
  --variant-receipt V1=state/materialized/same_gene_robustness_v1/pilot/receipts/V1.pilot-receipt.json \
  --variant-receipt V2=state/materialized/same_gene_robustness_v1/pilot/receipts/V2.pilot-receipt.json \
  --variant-receipt V3=state/materialized/same_gene_robustness_v1/pilot/receipts/V3.pilot-receipt.json \
  --variant-receipt V4=state/materialized/same_gene_robustness_v1/pilot/receipts/V4.pilot-receipt.json \
  --variant-receipt V5=state/materialized/same_gene_robustness_v1/pilot/receipts/V5.pilot-receipt.json \
  --variant-receipt V6=state/materialized/same_gene_robustness_v1/pilot/receipts/V6.pilot-receipt.json \
  --launch-manifest state/materialized/same_gene_robustness_v1/launch_manifest.json \
  --output-dir state/materialized/same_gene_robustness_v1/full/attempt1/jobs \
  --plan state/materialized/same_gene_robustness_v1/full/attempt1/plan.json

PYTHONPATH=src /venv/main/bin/python scripts/train/launch_same_gene_robustness.py \
  --plan state/materialized/same_gene_robustness_v1/full/attempt1/plan.json \
  --ledger state/materialized/same_gene_robustness_v1/full/attempt1/ledger.json
```

Any recovery plan must name only failed slots with `--retry-slot`, use the next
contiguous `--attempt`, and be supplied as an additional `--plan` to analysis.
After all 140 slots succeed, publish and independently verify the aggregate:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/analyze_same_gene_robustness.py \
  --contract experiments/campaigns/cmp_20260810_same_gene_robustness_multiverse_v1/frozen_task_contract.yaml \
  --launch-manifest state/materialized/same_gene_robustness_v1/launch_manifest.json \
  --plan state/materialized/same_gene_robustness_v1/full/attempt1/plan.json \
  --output reports/analyses/same_gene_robustness_20260811

PYTHONPATH=src /venv/main/bin/python scripts/analysis/analyze_same_gene_robustness.py \
  --contract experiments/campaigns/cmp_20260810_same_gene_robustness_multiverse_v1/frozen_task_contract.yaml \
  --launch-manifest state/materialized/same_gene_robustness_v1/launch_manifest.json \
  --plan state/materialized/same_gene_robustness_v1/full/attempt1/plan.json \
  --output reports/analyses/same_gene_robustness_20260811 \
  --verify-only
```

The authorized scientific total is exactly 140 selected successful runs and
560 selected model-arm fits. The attempt-one plan has 35 four-GPU waves;
technical recovery attempts can add process invocations but cannot add or
replace scientific slots silently. At least 40 GB of free space must remain
throughout. This workspace is not assumed persistent: the contract, code, prepared
fingerprints, registry, reports, and irreducible run artifacts must be
committed or synchronized off-instance before recycle or destroy.

## Claim limits

- This is post-hoc robustness analysis on already observed slides, not new
  confirmation.
- Geometry components prevent direct training/test overlap but are not
  patients or biological replicates.
- No result establishes cell-cell communication, causality, signaling, or a
  per-sender-cell effect.
- The Jacobian is with respect to a standardized neighbor mean. Individual
  sender effects additionally depend on degree and scale.
- CP10k and residual variants change the response estimand.
- Vendor cell type is RNA-derived and therefore not an independent control.
- A failed strict gate is not evidence of zero name alignment; label-null
  enrichment is a separate secondary claim.
- Probe labels such as slash-combined probes remain indivisible labels.
- The available data have no independent assay-round annotation; "same name"
  means target/source probe-label equality.
- No gene-level inferential claim or patient-generalization claim is allowed.
- Independent confirmation requires new slides or patients frozen before
  outcome inspection.
