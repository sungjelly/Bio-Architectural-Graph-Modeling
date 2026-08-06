# Signed-program additive Stage-0 pilot

## Status

- Phase: complete
- Outcome: negative
- Campaign: `cmp_20260729_signed_program_stage0_pilot`
- Design: exploratory architecture diagnostic on the already locked synthetic
  fixture; no registry, queue, or real-expression run is authorized

All three prespecified variants failed. This branch is stopped; see
`RESULTS.md`.

This campaign was defined after the additive QKV model failed the locked
Stage-0 synthetic recovery gate. The failed result is evidence against that
architecture on this fixture, not evidence that local sender state is absent
from the generated data. One plausible architectural failure is especially
relevant: the planted target depends on the **number** of active local senders,
whereas receiver-wise softmax attention normalizes over incoming neighbors and
can discard count information.

`frozen_task_contract.yaml` was written and checksum-locked before any outcome
from the three models below was run. The source fixture, seed, effect, masks,
optimizer, epoch budget, and gates remain unchanged.

## Objective and deliverables

Determine whether a bounded, biologically structured additive model can recover
the locked positive one-hop sender-state effect without producing the analogous
discovery under the locked null injection.

Deliverables are:

1. a frozen task contract and checksum;
2. a model with exact self, regional, and local decomposition;
3. exact signed source-program-to-receiver-program edge contributions;
4. parameter-matched self+regional, true-local, permuted-sender, and
   null-injection arms within each architecture;
5. focused unit tests and one small CPU or GPU diagnostic; and
6. an outcome receipt that reports every prespecified variant, including
   failures.

No production registration, queue mutation, real-data graph run, or biological
interpretation is allowed by this campaign.

## Scientific question and hypotheses

Question: can an additive local branch that preserves sender counts recover a
known signed local dependency that the QKV-softmax branch missed?

Primary hypothesis: an unnormalized sum of fixed nonnegative sender programs,
mapped through signed source-program-to-receiver-program coefficients, will
recover the planted effect and distinguish true from spatially displaced sender
states.

Credible alternatives are:

- the mixed masking/training objective does not identify the planted dependency
  within 96 epochs;
- the regional decoy remains sufficient to dominate optimization;
- the fixed evaluation set is too small for the deletion diagnostic;
- apparent recovery is a capacity or optimization artifact that also appears
  under the null injection; or
- the existing deletion gate is incompatible with a useful fitted predictor.

The discriminating predictions are the existing locked Stage-0 checks: true
local must beat both self+regional and permuted-local, the planted contribution
must have the correct sign, deletion of the top planted contributions must
worsen target loss more than distance-matched deletion, and the analogous
discovery must not occur under the null injection.

## Estimand and permitted claim

The estimand is recovery of a generated, positive, direct one-hop
sender-program dependency on one fixed observed ANC-05 geometry. Expression is
entirely synthetic. The experimental unit is one deterministic synthetic
fixture; mask observations and edges are not independent biological
replicates.

The maximum defensible claim after a pass is that a particular constrained
architecture can recover this planted dependency under the frozen diagnostic.
A pass does not establish accuracy on real expression, unseen-core
generalization, a biological interaction, a mechanism, or causality. A failure
is evidence that the tested architecture is not validated for real-data graph
interpretation under this gate.

## Fixed inputs and leakage controls

- Source configuration:
  `cmp_20260729_multiscale_hurdle_count_pilot/stage0_synthetic_recovery_config.yaml`
  with SHA-256
  `61a2fcf80f9f7e0ae4bae58448682fe6c286c12689f1fda526bbd9268eba7e00`.
- Prepared geometry is addressed only by opaque alias `ANC-05`.
- Expected selected geometry: 160 nodes; expected fixture checksum
  `f53076d07e81842b4e73a4bb6ec66b795d2899d207a620c35a136377b2aa1cd7`.
- Generated counts, active-sender pattern, local/regional graphs, sender
  permutation, evaluation nodes, masks, and null target are constructed by the
  existing checksum-bound fixture implementation.
- Hidden entries are masked inside both the self encoder and fixed program
  projection. Coordinates and identifiers are not model covariates.
- No target expression, result, cell label, pathway annotation, or planted
  sender identity is used to define a program or architecture variant.

This reuses a fixture whose QKV outcome is already known, so the campaign is
exploratory architecture development rather than independent confirmation.

## Prespecified architecture

All variants use the exact decomposition

```text
prediction = self prediction
           + regional program prediction
           + local signed program prediction
```

The self branch is the existing mask-aware count encoder and unrestricted
decoder. Regional context is a separate degree-normalized mean of sender
programs over the `(75, 300] µm` graph. Local context uses the `<=75 µm` graph
and never mixes regional edges.

Because the synthetic genes have no biological pathway labels, the only
non-circular program definition is fixed identity: each synthetic gene is one
nonnegative sender and receiver program. The model API accepts externally
prespecified nonnegative program matrices for later work, but this diagnostic
does not learn or select programs from its target. Signed coefficient tensors
map sender programs to receiver programs and two hurdle channels. Thus each
selected edge has an exact signed additive contribution; there is no attention
score to reinterpret.

The following three variants are exhaustive for this campaign:

| Variant | Fixed local basis | Purpose |
|---|---|---|
| `count_sum` | one unnormalized weight `1` per edge | preserve active-sender count |
| `count_plus_concentration` | unnormalized `1` plus receiver-degree-normalized `1/degree` | separate count from concentration |
| `radial_count` | unnormalized `1` plus three fixed triangular near/mid/far functions of distance/75 µm | preserve count while resolving coarse spatial scale |

All regional and local coefficient tensors start at zero, so initial prediction
is exactly self-only. Variants may differ in parameter count, but the four
scientific arms within each variant must have identical trainable parameter
names, shapes, values, seed, masks, and optimizer.

## Arms, controls, and nulls

Each variant runs exactly four arms:

| Arm | Regional | Local |
|---|---|---|
| `self_regional` | true regional mean | disabled zero contribution |
| `true_local` | true regional mean | correctly aligned local sender programs |
| `permuted_local` | true regional mean | identical edge slots/attributes with locked spatial-antipode sender states |
| `null_true_local` | true regional mean | true local routing with the locked marginal-preserving null target |

The permuted arm tests correct sender-state alignment conditional on fixed
topology. It does not test topology and cannot identify molecular signaling.
The null-injection arm is the negative control for attribution/deletion
discovery.

## Frozen training and gate

Every arm uses seed `2718`, data seed `0`, null seed `104729`, effect size
`4.0`, 96 fixed epochs, AdamW learning rate `0.003`, no weight decay, gradient
clip `1.0`, mixed partial/whole-node masking rates `0.35/0.35`, mask seed
`314159`, FP32, no early stopping, and no validation or test selection.

A variant passes only if every existing Stage-0 relation is true:

- true-local whole-node hurdle loss is strictly below self+regional;
- true-local loss is strictly below permuted-local;
- mean selected planted continuous contribution is strictly positive;
- top-contribution deletion loss increase is strictly positive and greater
  than the distance-matched deletion increase; and
- no corresponding positive-contribution plus deletion discovery occurs under
  the null injection.

The existing sender-permutation QC gates must also pass. Thresholds, seed,
effect, epoch count, top-edge count (`16`), fixture membership, and loss are not
changed. All three variants are reported; no favorable variant may stand in for
the complete result. If none passes, this branch ends negative. A pass
authorizes only a separately contracted real-data pilot; it does not authorize
one here.

## Resource and failure plan

This is a 160-node, four-gene diagnostic. It may use CPU or one safely available
GPU, runs serially, and must write only a compact JSON receipt. It must not
enqueue work, mutate the registry, publish a run bundle, or consume material
GPU time. Any non-finite value, fixture/config checksum mismatch,
parameter-matching failure, mask failure, or missing arm is an execution
failure rather than a scientific negative.

## Verification

Run from the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_signed_program_synthetic.py

PYTHONPATH=src /venv/main/bin/python \
  scripts/diagnostics/run_signed_program_synthetic.py \
  --device cpu \
  --output scratch/diagnostics/signed_program_stage0_result.json
```

The second command is the only outcome-bearing diagnostic for this campaign.
Its receipt must bind the task-contract checksum, source configuration checksum,
fixture checksum, model parameter structures, all arm metrics, gate decisions,
runtime, and exact command.
