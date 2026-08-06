# Multiscale hurdle-count biological-signal pilot

## Status

- Phase: complete
- Outcome: negative
- Campaign: `cmp_20260729_multiscale_hurdle_count_pilot`
- Design: exploratory held-in representation and graph-decomposition pilot

The registered Stage-0 run failed the locked synthetic recovery gate. Real-data
graph arms and graph interpretation are therefore stopped. The separate
self-only representation question is not part of this campaign's graph result;
see `RESULTS.md` for the observed metrics and verification record.

This campaign is exploratory. Results from the earlier continuous, categorical,
large-neighborhood QKV, and ten-core hybrid-count studies were inspected before
this contract was written. The ten biological units are pathology-confirmed
adjacent-normal cores, not true-Normal tissue. Cells and fixed mask replicates
are technical observations.

Before any GPU training, real-core graph QC found that counting successful
2-switches overstated the strength of the rewired null: only `6.33%` to
`8.20%` of final local relations differed because later swaps could restore
original edges. `contract_amendment_001.yaml` strengthened the gate to prohibit
reintroduction and require at least `50%` final replacement. A subsequent
degree-relaxed feasibility analysis showed that no distance-bin choice could
meet that replacement gate and the frozen distance-fidelity gate on all five
cores. `contract_amendment_002.yaml` therefore retains the failed rewiring
design as a negative record and replaces it, before training, with a
macroblock-stratified spatial-antipode sender-state permutation. No run may use
either superseded rewiring materialization. `contract_amendment_003.yaml`
clarifies that this fixed-topology null tests correct sender-state alignment,
not topology itself, and adds a mandatory masking noninterference check for
the small number of effective sender-equals-receiver slots.

## Objective and deliverables

Test whether:

1. a detection-plus-positive-continuous hurdle objective avoids the failing
   balanced ordinal decoder while retaining useful nonzero-count prediction;
2. a parameter-matched additive architecture can separate cell-autonomous,
   regional-context, and sparse local-message contributions; and
3. the local contribution is predictive on true local topology but not on a
   topology-matched, spatially displaced sender-state permutation.

Required deliverables are a frozen configuration/graph receipt, focused and
synthetic-recovery tests, registered resource pilots, verified run bundles,
paired core-level comparisons, resource accounting, and a concise report with
negative evidence and exact reproduction commands.

## Evidence motivating the change

The completed ten-core hybrid model had useful detection and positive
continuous performance but failed its ordered-state gate:

- detection balanced accuracy: `65.41%` versus `51.10%` reference;
- positive continuous Huber: `24.57%` better than the per-gene reference in
  `10/10` cores;
- positive ordinal MAE: `69.20%` worse than its reference in `0/10` cores.

An exploratory read of its protected local predictions showed that decoding
positive count states from the continuous channel gave approximately `0.571`
MAE versus `0.579` for the per-gene positive-median reference, with `8/10`
cores favorable. This is a post-outcome diagnostic, not a new confirmatory
result. It motivates a clean objective pilot; it does not establish that the
new model will pass.

The prior k=1,000 and k=5,000 studies failed their practical graph-gain gates.
Those graphs represent regional/global context, not direct interaction.
Another large-k sweep is therefore not justified by the current evidence.

## Questions, hypotheses, alternatives, and predictions

### H1: count representation

A balanced detection loss plus positive-only robust continuous loss will
predict detection and positive count level better than transductive per-gene
references without relying on the dominant zero state.

Predictions:

- detection balanced accuracy exceeds the prevalence reference;
- positive standardized-log1p Huber improves by at least 2%;
- positive count-state MAE, obtained only from the continuous count estimate
  and fixed count bins, improves by at least 2%; and
- the direction holds in at least four of five prespecified pilot cores.

Alternative explanations include insufficient observed information, a
per-gene reference already being near-optimal, and optimization failure.
Overall exact accuracy cannot distinguish these explanations because about
90% of targets are zero.

### H2: separated spatial signal

After accounting for the unrestricted self and regional branches, the sparse
local signed-message branch reduces whole-node hurdle loss on true topology
and does not do so when local sender states are spatially displaced within
their macroblock while topology and edge geometry remain fixed.

Predictions:

- `S+R+L` improves at least 2% over `S+R` in at least four of five pilot
  cores;
- `S+R+L` beats `S+R+L-permuted` in at least four of five cores;
- spatial-block direction agrees with whole-node direction; and
- signed local contributions pass a planted-signal deletion/recovery check
  before any real-data biological interpretation.

Credible alternatives are that the usable information is cell-autonomous,
that regional smoothing explains any graph gain, that segmentation spillover
creates a short-range shortcut, or that flexible branches improve optimization
without using true topology. A true-local advantage over the permutation may
also reflect local cell-type composition; it is not specific evidence for
molecular signaling.

The contrasts answer different questions. `S+R+L` versus `S+R` measures
graph-specific gain from adding the observed local branch. `S+R+L` versus
`S+R+L-permuted` holds topology and geometry fixed and tests whether prediction
depends on the correct alignment of sender state to local source position.
Neither contrast alone supports a communication mechanism.

## Estimands and maximum claim

The primary estimand is held-in whole-node masked raw-count reconstruction.
Partial-gene and spatial-block masking are secondary and remain separately
reported. Every core is trained independently and all within-core
preprocessing is transductive.

The maximum permitted claim is exploratory adjacent-normal, held-in
representation capacity and a possible topology-specific predictive
dependency. The campaign cannot establish unseen-patient generalization,
direct cell-cell communication, biological mechanism, or causality.

## Inputs and leakage controls

- Reuse the ten checksum-bound prepared adjacent-normal artifacts from
  `cmp_20260729_adjacent_normal_10core_hybrid_count_gat`.
- Retain 1,000 biological raw-count targets and the same 22 permitted
  morphology/imaging covariates.
- Exclude technical controls, identifiers, coordinates as node covariates,
  RNA-derived QC/library size, vendor cell types/clusters/neighborhoods, and
  hidden target expression.
- Apply the mask authoritatively inside the encoder to both discrete and
  continuous expression inputs.
- Use coordinates only for graph construction and spatial masks.
- Fit transformations independently within the current core. This
  resubstitution is a named limitation, not a validation design.

## Fixed count objective and metrics

The input encoder retains the fixed eight observed count states plus the
input-only mask token and exact standardized `log1p(raw count)`.

The new primary prediction has:

1. one detection logit per gene; and
2. one standardized positive `log1p(count)` estimate per gene.

The masked-entry loss is the equal mean of:

- balanced detection BCE, defined as the mean of the zero and positive mean
  BCE values; and
- positive-only Huber loss with delta `1.0`.

Positive counts are reconstructed by inverting the recorded per-gene
standardization, applying `expm1`, clipping at zero, and applying explicit
nonnegative half-up rounding as `floor(value + 0.5)`. Fixed states are `0`,
`1`, `2`, `3`, `4-7`, `8-15`, `16-31`, and `>=32`. Positive-state metrics
condition on a true positive target and therefore ignore the detection
decision; full-state metrics apply the predicted detection decision. No fitted
state threshold or post-outcome calibration is allowed.

Primary metrics are whole-node hurdle loss, detection balanced accuracy, and
positive count-state MAE. Also report detection sensitivity/specificity,
positive exact/within-one accuracy, positive continuous Huber/MAE,
reconstructed-count log1p MAE, eight-state balanced and exact accuracy, and
per-state recall. Exact accuracy is descriptive only.

## Parameter-matched architecture arms

Every arm has the same encoder, decoder, branch modules, trainable parameter
count, seed, masks, optimizer, and epoch budget.

| Arm | Regional routing | Local routing |
|---|---|---|
| `S` | within-cell surrogate | within-cell surrogate |
| `S+R` | true regional context | within-cell surrogate |
| `S+R+L` | true regional context | true local topology |
| `S+R+L-permuted` | true regional context | true local topology with macroblock-stratified, spatial-antipode sender states |

Predictions decompose exactly as:

```text
self prediction + regional prediction + local signed prediction
```

The self branch is unrestricted. Regional and local output projections start
at zero so the initial predictor is exactly self-only. Surrogate routing feeds
the receiver representation through the same branch parameters without
neighbor information, rather than leaving graph-only parameters unused.

Local messages are one-hop, signed, and low-rank so selected
sender-program-to-receiver-target contributions can be decoded without
materializing an edge-by-gene tensor. Attention remains computational routing,
not biological importance.

## Graph contract

- Local graph: exact mutual kNN, `k` cap `64`, maximum distance `75 µm`, no
  self-loops. The radius defines the candidate biological scale; the cap is a
  safety bound.
- Regional context: exact mutual kNN with cap `256`, restricted to the annulus
  `(75, 300] µm`, summarized separately from local messages.
- The local and regional edge sets must be disjoint.
- The permutation null keeps the true local edge index, receiver identities,
  and edge attributes bit-identical. It changes only which node state supplies
  each local source message. Self and regional branches remain unpermuted.
- Within each pre-existing FOV-qualified `300 µm` macroblock, nodes are ordered
  along a deterministic principal coordinate axis and circularly shifted by
  half the group size. Construction uses only coordinates, macroblock IDs, and
  stable row indices—not expression or expression-derived annotations.
- Before GPU use, the mapping must be a bijection globally and in every
  macroblock, change at least `99%` of node and local-edge-slot sender
  identities, and move at least `90%` of nodes beyond the `75 µm` local radius.
  Its checksum and the fraction of effective sender-equals-receiver messages
  must be recorded.
- Pre-training QC found only `0.0345%` to `0.0461%` of directed slots have an
  effective permuted source equal to the receiver, but those slots touch
  `1.98%` to `2.70%` of receivers. A regression test must therefore show
  bit-exact invariance to every masked receiver value for whole-node masks and
  to masked entries for partial-gene masks before GPU training. These slots
  are retained as a disclosed, slightly self-like conservative feature of the
  null.
- Long-range regional edges are context only and cannot be interpreted as
  direct interaction.

Graph checksums, edge counts, distance summaries, component counts, isolated
nodes, permutation checksums, displacement summaries, sender replacement, and
true-versus-permuted graph-identity checks must be recorded.

## Units, stages, and gates

Core-level values are the inferential observations. Technical mask replicates
are averaged within core before any comparison.

### Stage 0: synthetic recovery

On a fixed `160`-node ANC-05 geometry/macroblock fixture, plant a known signed
sender-program to receiver-target effect on local edges plus a co-localized
regional decoy. The initially proposed `240`-node coordinate-only crop was
rejected before outcome modeling because only `85.83%` of its permuted nodes
moved beyond `75 µm`; the fixed `160`-node crop passes at `96.25%`. Require:

- lower true-local loss than self+regional and permuted local;
- correct contribution sign;
- top-contribution deletion worsens the planted receiver target more than a
  distance-matched null deletion; and
- no analogous discovery under a null injection.

Failure blocks real-data graph interpretation.

### Stage 1: objective/resource pilot

Use `ANC-03` (largest selected core) and `ANC-05` (smallest selected core),
chosen by cell count rather than outcome. Run the `S` arm for two epochs for
memory/precision checks, then the fixed science budget only if the resource
gate passes.

Require finite losses/gradients, FP32-versus-AMP absolute loss discrepancy at
most `1e-3`, peak allocated VRAM at most `12 GiB`, and projected runtime at
most `0.5 GPU-hours` per 200-epoch core. The representation gate must pass on
both cores before Stage 2.

### Stage 2: five-core architecture pilot

The prespecified size-stratified core aliases are `ANC-02`, `ANC-03`,
`ANC-05`, `ANC-06`, and `ANC-09`. Run all four arms with seed `0`, 200 epochs,
and three fixed mask replicates per mode.

Advance to the remaining five cores only if the H1 representation criteria
pass and both H2 effect/direction criteria pass in this five-core pilot.
Outcome-driven substitution of cores is prohibited.

## Resource and stop contract

- Preferred aggregate GPU budget: at most `12 GPU-hours`.
- Absolute aggregate GPU budget: at most `24 GPU-hours`.
- Safe devices: GPUs `0,1,2,3,5,6,7`; GPU `4` is excluded for prior thermal
  throttling.
- Maximum concurrency: six independent jobs.
- Per-device peak allocated VRAM: at most `20.5 GiB`.
- Aggregate observed process VRAM target: at most `50 GiB`.
- Filesystem used-space hard stop: `55.0 GB` in decimal `df` units.
- New finalized campaign output target: at most `3.5 GiB`, with last-only
  checkpoints.

Stop before scaled execution on a failed synthetic, mask, graph, permutation,
parameter, precision, memory, runtime, checksum, or artifact gate. Diagnose
isolated execution failures; do not silently change scientific parameters or
omit failed cores.

## Current result and decision

Registered run
`r_20260729T172139Z_1833137a_s000_f00_a01_df2d8e14` completed all four
prespecified Stage-0 arms on the locked 160-node ANC-05 observed-geometry
fixture. Execution and artifact verification passed, but the scientific gate
failed:

- true-local improved over self+regional by `3.38%`;
- true-local was `0.22%` worse than permuted-local;
- the planted contribution sign was positive, but deleting its top edges
  increased target loss by only `0.000417`, versus `0.002199` for the
  distance-matched deletion; and
- an analogous positive-contribution/deletion discovery occurred under the
  null injection.

This is distinct from
`stage0_prerun_negative_diagnostic_v1.json`, an unregistered code-path
diagnostic on generated unit-test coordinates. That earlier result was retained
as pre-run evidence but was explicitly prohibited from standing in for the
registered observed-geometry result.

The registered run used GPU 0 (RTX 3090) for `39.58` seconds
(`0.0110 GPU-hours`), peaked at `0.0715 GiB` allocated VRAM, and published a
`4,028,316`-byte verified bundle. Because Stage 0 was a prespecified stop gate,
Stages 1 and 2 were not run in this campaign. The maximum defensible conclusion
is that this additive QKV local branch did not reliably recover the planted
correctly aligned sender-state dependency on the locked synthetic fixture. It
provides no evidence about real-cell graph gain, biological communication,
mechanism, or causality.

## Expected artifacts and verification

Active output belongs only under `scratch/active_runs/<run_id>/`; verified
immutable bundles belong under `artifacts/runs/YYYY/MM/<run_id>/`. Every run
must be registered and preserve resolved configuration, command, code/data/
split/environment provenance, append-only metrics, fixed masks, resource use,
last checkpoint, failure records, checksums, and schema-v3 checkpoint index.

Before enqueue:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_hurdle_continuous.py \
  tests/unit/spatial_benchmark/test_local_source_permutation.py \
  tests/unit/spatial_benchmark/test_multiscale_graphs.py \
  tests/unit/spatial_benchmark/test_multiscale_hybrid.py \
  tests/unit/spatial_benchmark/test_multiscale_hurdle_training.py \
  tests/unit/spatial_benchmark/test_multiscale_synthetic.py \
  tests/unit/spatial_benchmark/test_materialize_multiscale_hurdle_campaign.py \
  tests/unit/spatial_benchmark/test_enqueue_multiscale_hurdle_campaign.py \
  tests/unit/spatial_benchmark/test_multiscale_hurdle_capacity_runner.py \
  tests/unit/spatial_benchmark/test_verify_multiscale_hurdle_gates.py

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

After completion, run focused tests, practical repository tests,
`verify-artifacts`, checkpoint indexing verification, the locked comparator,
and `doctor`. Update this README with exact commands, run IDs, resource use,
gate decisions, failures, limitations, and the maximum defensible conclusion.

The completed Stage-0 result can be checked without rerunning training:

```bash
cd experiments/campaigns/cmp_20260729_multiscale_hurdle_count_pilot
sha256sum -c frozen_task_contract.sha256 \
  contract_amendment_001.sha256 \
  contract_amendment_002.sha256 \
  contract_amendment_003.sha256 \
  stage0_prerun_negative_diagnostic_v1.sha256 \
  stage0_synthetic_recovery_config.sha256

cd ../../..
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts \
  --run-id r_20260729T172139Z_1833137a_s000_f00_a01_df2d8e14

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 resolve-checkpoint \
  r_20260729T172139Z_1833137a_s000_f00_a01_df2d8e14 --role last
```
