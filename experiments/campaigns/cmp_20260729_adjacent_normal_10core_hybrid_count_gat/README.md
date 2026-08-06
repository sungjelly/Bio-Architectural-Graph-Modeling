# Ten-core adjacent-normal hybrid raw-count GAT

## Status

- Phase: complete
- Outcome: negative; both frozen primary gates failed
- Campaign: `cmp_20260729_adjacent_normal_10core_hybrid_count_gat`
- Design: exploratory held-in masked-expression reconstruction

This campaign is explicitly **exploratory** because the ten-core count
distribution was inspected before the criteria below were fixed. The samples
are ten pathology-confirmed adjacent-normal spatial tissue cores, referred to
only as `ANC-01` through `ANC-10`; they are not true-Normal samples. Each core
is fit independently and remains the biological unit for aggregation. Cells
and mask replicates are technical observations, not biological replicates.

The frozen machine-readable contract is `frozen_task_contract.yaml`. Its
scientific parameters and gates must not be changed after any pilot or
production outcome is observed. Resource-only changes such as receiver chunk
size require a separately named and documented resource variant.
The frozen contract SHA-256 is
`2c4db92868ab37b806b82274f20f31e402b737714ff0a946f4954386db9eca1c`.

## Frozen task contract

### Objective and deliverables

Determine whether a hybrid raw-count representation learns detected expression
and ordered positive levels rather than exploiting the dominant zero state,
and whether exact broad-context GATv2 improves on an exactly
parameter-matched cell-autonomous control. Compare the result descriptively
with the previous one-core `0/1/2/>=3` categorical G2 campaign without treating
the studies as exchangeable.

Required deliverables are:

1. frozen campaign and configuration records;
2. implementation plus focused and synthetic-recovery tests;
3. two registered two-epoch `ANC-01` resource pilots, one per arm, and a
   same-batch FP32-versus-AMP diagnostic;
4. twenty verified 200-epoch production bundles if and only if the pilot gate
   passes;
5. per-mask, per-core, and unweighted ten-core metric tables, exact paired
   inference, gate decisions, failures, and provenance;
6. a concise Markdown report and portable single-file HTML report.

### Question, hypotheses, alternatives, and discriminating predictions

Primary questions:

1. Does the hybrid representation predict detection and positive count level
   beyond transductive per-gene references?
2. Does exact mutual-`k=1000` broad-context GATv2 reduce the frozen hurdle loss
   relative to an exactly parameter-matched self-only network?
3. Do positive-state results change the interpretation of the previous
   zero-dominated four-state categorical experiment?

Primary hypotheses:

- H1-representation: the GAT hybrid representation passes the frozen positive
  ordinal, positive continuous, and detection criteria.
- H1-graph: the GAT passes the frozen paired graph-gain criteria against the
  matched self arm.

Credible alternatives:

- A1-zero shortcut: high eight-state accuracy is driven by the roughly 90%
  zero state while positive-level metrics remain poor.
- A2-self information: same-cell observed genes and permitted morphology or
  imaging covariates contain most usable signal, so matched self performs as
  well as GAT.
- A3-regional smoothing: any GAT advantage reflects broad co-localized tissue
  context, not direct cellular communication.
- A4-optimization: the richer hurdle objective or fixed 200-epoch budget fails
  to optimize even though the implementation is capable of recovery.
- A5-reference sufficiency: per-gene prevalence, positive ordinal frequencies,
  and positive medians explain the observed positive-state performance.

Discriminating predictions:

- A zero shortcut may yield high overall exact accuracy but cannot pass
  detection balanced accuracy, positive ordinal MAE, and positive continuous
  Huber gates.
- A graph-specific advantage must survive exact parameter matching, occur in
  at least eight cores, reach at least 2% mean relative hybrid-loss gain, and
  have corrected exact paired support in all three component losses.
- A representation advantage must beat per-gene references by at least 2% on
  both positive metrics, with at least eight cores favoring the model on each.
- The synthetic planted-signal recovery check must improve detection and
  ordered positive prediction; failure supports an implementation or
  optimization defect and blocks the resource pilot.

### Estimands and maximum permitted claim

The primary estimand is held-in whole-node masked-expression reconstruction.
Partial-gene and spatial-block modes are secondary estimands and remain
separate in all tables. Every cell, graph, and fitted preprocessing statistic
within a core is available during fitting; fixed evaluation masks do not make
this a validation or test design.

The maximum defensible conclusion is held-in masked-expression representation
capacity and possible broad-context graph gain across ten adjacent-normal
cores. This campaign cannot support patient-held-out generalization,
true-Normal performance, direct cellular interaction, biological mechanism,
or causality.

### Inputs, allowed covariates, graph, and leakage policy

- Inputs are the ten verified artifacts under
  `data/processed/adjacent_normal_10core_qkv_large_k_v1/anc-##/prepared_v1`,
  routed publicly only through aliases `ANC-01` through `ANC-10`.
- Raw biological-probe counts are targets and expression inputs. Counts must be
  finite, nonnegative, and integer-valued; validation fails closed otherwise.
- Technical controls are excluded.
- The exact 22 permitted morphology and imaging fields are inherited from
  `configs/features/cosmx_full_core_morphology_edge_geometry.yaml`.
- Prohibited node inputs are identifiers, absolute or local coordinates,
  target-cell library size, all RNA-derived QC, vendor annotations, hidden
  target values, and measured edge attributes in the self arm.
- The GAT uses each core's existing checksum-bound exact mutual `k=1000`
  graph and the existing 17 measured geometry edge features. It uses exact
  receiver partitioning, no sampling, no edge dropout, and no self loops.
- This graph is broad regional context. It is not interpreted as direct
  cell-cell communication.
- Coordinates are permitted only to identify the already frozen graph and
  spatial evaluation blocks; they are never node features.
- The explicit mask is applied inside the model and is authoritative even if
  an unmasked count or continuous value is passed accidentally.

The principal leakage limitation is resubstitution: all core-level
preprocessing and reference statistics are all-fit and transductive. The mask
implementation, graph inputs, and feature lists must be audited before pilot
execution.

### Fixed count representation

Biological raw counts map to eight output states:

| Token | Raw count |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4-7 |
| 5 | 8-15 |
| 6 | 16-31 |
| 7 | 32 or greater |

Token `8` is input-only `MASK` and is never a target. No fitted quantiles,
gene medians, or other data-dependent token boundaries are permitted.

The encoder concatenates or otherwise jointly projects:

1. gene-specific gene-by-token projections, including a distinct per-gene
   mask projection;
2. the exact within-bin, per-gene standardized `log1p(raw count)` value; and
3. the permitted morphology and imaging covariates.

For a masked entry the discrete branch is forced to token `8` and the
continuous branch is forced to standardized zero, irrespective of the supplied
raw token or continuous value. Per-gene continuous location and scale are fit
on all cells of the current core and recorded as transductive preprocessing.

### Fixed architectures and parameter matching

`hybrid-gat-k1000` uses the existing exact receiver-partitioned,
edge-conditioned GATv2 design with hidden width 512, four heads, two graph
layers, FFN width 512, decoder width 512, exact mutual `k=1000` topology, and
17 geometry edge features.

`hybrid-matched-self` uses the identical hybrid encoder, FFN/decoder widths,
depth, outputs, seed, masks, and optimizer. It receives no topology,
neighboring-cell values, coordinates, or edge attributes. Graph-only parameter
budgets are repurposed into strictly within-cell transformations. Constructed
trainable parameter counts must be exactly equal; mismatch blocks all training.

Each gene decodes one detection logit, six positive cumulative ordinal logits,
and one continuous standardized-`log1p` prediction. The mask token is excluded
from the output support. Positive state is decoded as
`1 + sum(sigmoid(cumulative_logit_t) >= 0.5)` over six thresholds; detection is
decoded at probability `0.5`.

### Fixed loss

Loss is evaluated only at masked entries:

1. balanced detection BCE is the mean of the zero-target mean BCE and the
   positive-target mean BCE;
2. for each of six positive ordinal thresholds, balanced BCE is the mean of
   its below-or-equal and above-threshold mean BCE, then thresholds are
   averaged;
3. positive continuous Huber uses delta `1.0` on per-gene standardized
   `log1p(count)` for positive targets only.

Total hybrid loss is exactly
`(detection_loss + ordinal_loss + positive_continuous_huber) / 3`. No weights
may be retuned. An absent expected detection, ordinal-threshold, or positive
continuous stratum is an error, not a zero contribution.

### Training and masks

- optimizer: AdamW;
- learning rate: `3e-4`;
- weight decay: `1e-4`;
- gradient clipping: `1.0`;
- seed: `0` independently in every core and arm;
- fixed budget: 200 epochs;
- deterministic execution;
- mixed precision only after the frozen equivalence diagnostic passes;
- no validation selection, early stopping, neighbor sampling, or edge dropout;
- final epoch is canonical and the registered retained checkpoint;
- existing paired partial-gene, whole-node, and spatial-block epoch-mask
  schedule;
- three fixed, disjoint evaluation-mask replicates per mode in production.

### Frozen references and metrics

All references are fitted independently within a core and labeled
transductive:

- all-zero prediction;
- per-gene detection prevalence with Jeffreys smoothing
  `(positive_count + 0.5) / (total_count + 1)`;
- per-gene positive cumulative probabilities with the same add-half smoothing
  for every threshold;
- per-gene median positive `log1p(count)` prediction.

For every core, arm, mask mode, and fixed mask replicate, record total hybrid
loss; all three component losses; detection balanced accuracy, sensitivity,
specificity, and precision; eight-state exact and balanced accuracy; per-state
support and recall; positive exact accuracy, ordinal MAE, and within-one-state
accuracy; positive standardized-log1p Huber and MAE; reconstructed-count
`log1p` MAE; and metrics collapsed to `0/1/2/>=3`. Also record runtime, peak
allocated VRAM, trainable parameter count, convergence summaries, and all
failures. Overall exact accuracy is descriptive only.

Reconstructed count uses the continuous prediction inverted through the
recorded per-gene standardization and `expm1`, clipped at zero; detection below
`0.5` reconstructs zero. The collapsed comparison maps states 0, 1, and 2
directly and states 3-7 to `>=3`.

Core-level summaries first average technical mask replicates, then aggregate
the ten core values with equal core weights. Cells and masks never determine
the inferential sample size.

### Frozen gates and exact inference

Define paired improvement as `matched_self - GAT`, so positive values favor
GAT. Relative total-loss improvement uses matched-self loss as denominator.

The graph gate passes only if all conditions hold:

1. all twenty production runs finish 200 finite epochs and verify;
2. unweighted mean paired relative whole-node hybrid-loss improvement is at
   least 2%;
3. at least eight of ten cores favor GAT on whole-node hybrid loss;
4. one-sided exact paired sign-flip tests on the ten core-level improvements
   support positive gain separately for detection BCE, ordinal BCE, and
   positive continuous Huber after Holm family-wise correction at `0.05`;
5. mean GAT positive ordinal MAE and positive continuous Huber are each no
   worse than matched self.

Each exact test enumerates all `2^10` sign assignments of the paired
core-level differences and uses the mean signed improvement as statistic.
Ties remain in the enumeration. All three Holm-adjusted hypotheses must reject
in the GAT-favorable direction.

The representation gate is applied to `hybrid-gat-k1000` and passes only if:

1. mean positive ordinal MAE improves by at least 2% over the per-gene
   positive ordinal reference;
2. mean positive continuous Huber improves by at least 2% over the per-gene
   positive median-log-count reference;
3. at least eight of ten cores favor GAT over the corresponding reference for
   each positive metric; and
4. mean detection balanced accuracy exceeds the per-gene prevalence
   reference.

A failed gate is a valid negative result. Overall exact accuracy cannot rescue
either gate.

### Pilot gate, compute policy, and stop criteria

After focused and synthetic tests pass, run registered two-epoch pilots on
`ANC-01` for both arms. On one frozen batch, compare FP32 and AMP forward loss
using identical weights and masks. Production is authorized only if:

- losses and gradients are finite;
- trainable parameter counts are exactly equal;
- each pilot's peak allocated VRAM is at most `20.5 GiB`;
- absolute FP32-versus-AMP total-loss discrepancy is at most `1e-3`; and
- projected 200-epoch GAT runtime is at most six hours per core.

Before pilots and production, inspect every GPU, active process, framework
visibility, memory, utilization, temperature, and repository queue state. Do
not kill, preempt, or collide with existing work. If the pilot passes, use all
safely available GPUs or the repository queue with isolated output, locks, and
logs. A receiver-chunk change is execution-only but must be a separately
documented resource variant; tokenization, losses, masks, graph, seed, epoch
budget, metrics, and gates remain frozen.

Stop before production on a failed pilot gate, count/token/mask leakage,
parameter mismatch, graph or checksum mismatch, non-finite loss or gradient,
failed synthetic recovery, or artifact/registry verification failure. During
production, diagnose and recover isolated execution failures without changing
scientific parameters. Missing permissions, unavailable required inputs, or
unsafe occupied compute are external blockers; an unfavorable scientific
result is not.

### Registration, provenance, outputs, and verification

Every pilot and production run must be registered and classified, write only
under its isolated `scratch/active_runs/<run_id>/` directory while active, and
publish a verified immutable bundle under `artifacts/runs/YYYY/MM/<run_id>/`.
Each bundle must preserve resolved configuration; command and working
directory; code and dirty-worktree provenance; input, graph, split, and mask
checksums; environment and hardware; append-only epoch and evaluation metrics;
final checkpoint and schema-v3 checkpoint-catalog record; logs; failures;
runtime and peak memory; status; completion marker; and bundle checksums.

Required verification order:

1. focused unit tests for token boundaries, count validation, authoritative
   masking, continuous neutral masking, loss balancing/fail-closed strata,
   ordinal decoding, shapes, and parameter matching;
2. small synthetic detection/ordered-level recovery test;
3. focused and relevant regression tests;
4. both two-epoch `ANC-01` pilots and same-batch precision diagnostic;
5. twenty production runs only after a recorded pilot pass;
6. strict bundle/checkpoint verification, registry reconciliation, practical
   repository tests, and
   `PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor`;
7. Markdown and portable embedded-asset HTML reports with per-core tables,
   paired comparisons, negative evidence, limitations, prior-campaign
   comparison, and maximum defensible conclusion.

### Locked configuration materialization

The preparation-only materializer audits the frozen contract, the prior
ten-core materialization receipt, all ten prepared artifacts, exact mutual
`k=1000` graph identities, the 22-node/17-edge feature contracts, and exact
trainable parameter equality before its first write. It atomically publishes
two `ANC-01` pilot configurations, twenty production configurations, and a
checksum-bound receipt. It does not register, enqueue, or train any run.

Run from the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/materialize_hybrid_count_campaign.py
```

Canonical outputs are under
`scratch/locked_campaigns/cmp_20260729_adjacent_normal_10core_hybrid_count_gat/`:

- `resource_pilot_configs/`: the two two-epoch `ANC-01` arm configs;
- `production_configs/`: the twenty alias-by-arm 200-epoch configs;
- `locked_config_materialization.json`: source, component, parameter-count,
  graph, split, config, GPU-assignment, and no-mutation evidence;
- `pilot_gate_receipt.json`: intentionally absent until the resource-pilot
  workflow independently writes a passing or failing gate decision.

The materializer is byte-idempotent and fails closed if an existing locked
output differs. Its focused tests are:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_materialize_hybrid_count_campaign.py
```

### Frozen operational sequence

Run every command below from the repository root. Source, configurations, and
the report implementation must remain unchanged from the worker restart until
all queued production work finishes, because the queue records the dirty-tree
fingerprint when each run is claimed.

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_hybrid_count.py \
  tests/unit/spatial_benchmark/test_hybrid_count_capacity_runner.py \
  tests/unit/spatial_benchmark/test_materialize_hybrid_count_campaign.py \
  tests/unit/spatial_benchmark/test_enqueue_hybrid_count_campaign.py \
  tests/unit/spatial_benchmark/test_verify_hybrid_count_pilot_gate.py \
  tests/unit/spatial_benchmark/test_compare_hybrid_count_multicore.py \
  tests/unit/spatial_benchmark/test_paired_inference.py \
  tests/unit/infrastructure/test_configuration.py \
  tests/unit/infrastructure/test_queue_command.py \
  tests/unit/infrastructure/test_run_archive.py \
  tests/test_checkpoint_catalog.py \
  tests/integration/infrastructure/test_checkpoint_semantics.py \
  tests/integration/infrastructure/test_registry_queue.py

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 create-campaign \
  --campaign-id cmp_20260729_adjacent_normal_10core_hybrid_count_gat \
  --name "Ten-core adjacent-normal hybrid raw-count GAT" \
  --scientific-question "Does the hybrid representation recover positive expression levels, and does broad-context k=1000 GATv2 improve over an exactly parameter-matched self-only control across ten adjacent-normal cores?" \
  --plan experiments/campaigns/cmp_20260729_adjacent_normal_10core_hybrid_count_gat/campaign.yaml \
  --status pilot

supervisorctl restart \
  bagm-multicore-qkv-gpu0 bagm-multicore-qkv-gpu1 \
  bagm-multicore-qkv-gpu2 bagm-multicore-qkv-gpu3 \
  bagm-multicore-qkv-gpu4 bagm-multicore-qkv-gpu5 \
  bagm-multicore-qkv-gpu6 bagm-multicore-qkv-gpu7

PYTHONPATH=src /venv/main/bin/python \
  scripts/train/enqueue_hybrid_count_campaign.py --stage pilot

PYTHONPATH=src /venv/main/bin/python \
  scripts/train/verify_hybrid_count_pilot_gate.py

# Execute only when the preceding command returns a passing gate receipt.
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/enqueue_hybrid_count_campaign.py --stage production

PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/compare_hybrid_count_multicore.py \
  --output-dir reports/analyses/adjacent_normal_10core_hybrid_count_gat/comparison

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

The pilot enqueue receipt always identifies the original attempt-one queue
jobs. After a failed, stale, or pruned execution has been diagnosed, the gate
verifier accepts at most one exact, unbranched attempt-two descendant. The
retry must preserve the locked configuration, command, config reference, GPU,
priority, and two-attempt ceiling. The gate receipt retains the original job
ID, identifies the completed descendant separately, and includes the failed
attempt evidence. It also requires the runner's encoder and decoder
initial-state audits to be bit-identical across the paired arms. A branch,
duplicate completion, extra campaign root, changed retry semantics, or
incomplete terminal attempt fails closed.

Immediately before each enqueue, inspect `nvidia-smi`, active GPU processes,
temperatures, framework device visibility, queue state, worker PIDs, host RAM,
and free disk. GPU 4 remains excluded from this campaign's locked assignments;
its worker is restarted only to refresh imports and will not claim these jobs.

## Execution record

On 2026-07-29, the preparation-only materializer verified and published the
locked configuration set. The canonical materialization receipt checksum is
`ff859d8d5358a6f855d38f428e225537022af7d746bde4d277012292087d8b35`.
It records 11,674,880 trainable parameters in each arm, two pilot configs,
twenty production configs, and an LPT assignment restricted to GPUs
`0,1,2,3,5,6,7`. Independent read-only preflight through both the worker's
hybrid contract validator and the enqueue validator accepted all 22 configs.
The focused materializer/configuration/queue-command tests passed (16 tests).
The published directory remained byte-identical on an idempotent rerun.

An earlier pre-registration materialization was invalidated because its
resolved trainer configs omitted the explicit `optimizer: AdamW` provenance
field, although the implementation already used AdamW. It was preserved under
`scratch/diagnostics/hybrid_count_materialization_invalidated_pre_optimizer_20260729/`
with an invalidation note. No registry, queue, training, or scientific-outcome
mutation occurred before this correction.

Implementation preflight completed before registry or queue mutation. The
integrated campaign suite passed 170 tests. The complete practical repository
suite passed 490 tests with one unrelated optional-`pyarrow` test skipped and
no failures. The synthetic planted-covariate recovery test reduced the frozen
hurdle loss below 15% of initialization, achieved at least 0.99 detection
balanced accuracy, and positive ordinal MAE at most 0.10. Full-size paired
construction verified 11,674,880 trainable parameters per arm and bit-identical
initial encoder and decoder states. `spatial_benchmark doctor` reported schema
version 3, registry integrity `ok`, no issues or warnings, an empty active
queue, and 51.537 GiB free disk. These are implementation and infrastructure
checks, not scientific outcomes.

The two registered `ANC-01` pilots completed without failure. The independent
pilot-gate receipt checksum is
`63981dfda6f6cda1e17e717142507757599b951584fd21fbb34842a2fc328f28`.
FP32-versus-AMP total-loss discrepancies were `1.19e-6` for GAT and `9.54e-7`
for matched self. Peak allocated VRAM was `7.340 GiB` and `2.213 GiB`,
respectively. The measured GAT projection was `0.258 h` for 200 epochs. Both
checkpoints, all finite-loss/gradient checks, the 11,674,880-parameter match,
and bit-identical encoder/decoder initialization checks passed, so production
was authorized without a resource variant.

All twenty production runs completed at attempt 1 with no registered failure,
missing slot, duplicate completion, or invalid production attempt. Every run
completed 200 epochs, retained epoch 199 as the canonical `last` checkpoint,
and passed immutable-bundle and schema-v3 checkpoint verification.

### Frozen gate results

The graph gate failed:

- unweighted mean paired relative whole-node hybrid-loss improvement was
  `-0.0021%`, rather than the required `>=2%`;
- GAT had lower hybrid loss in `5/10` cores, rather than at least `8/10`;
- none of the three exact paired component tests rejected after Holm
  correction: detection BCE mean self-minus-GAT difference `-0.002951`
  (`p_Holm=1.0`), ordinal BCE `-0.000884` (`p_Holm=1.0`), and positive
  continuous Huber `+0.004073` (`p_Holm=0.105469`);
- GAT positive ordinal MAE was worse by `0.060706` on average, although its
  positive continuous Huber was better by `0.004073`.

The representation gate also failed. The evidence is mixed rather than a
simple zero shortcut:

- GAT detection balanced accuracy averaged `0.6541`, exceeding the per-gene
  prevalence reference `0.5110` by `0.1431`;
- positive continuous Huber averaged `0.3765` versus reference `0.5005`, a
  `24.57%` relative improvement, and all `10/10` cores favored GAT;
- positive ordinal MAE averaged `0.9789` versus reference `0.5789`, a
  `69.20%` relative worsening, and `0/10` cores favored GAT.

Thus the model learned useful detection and robust-loss continuous positive
value information, but it did not pass the prespecified ordered-positive-level
criterion. Lower balanced cumulative ordinal BCE did not translate into lower
decoded ordinal MAE under the frozen threshold decoder. This objective/decoder
disagreement is a result, not grounds to change the locked metric or gate.

Overall eight-state exact accuracy was `58.62%`, below the `90.42%` all-zero
reference, while eight-state balanced accuracy was `32.97%` versus `12.5%` for
all-zero. Positive-state exact accuracy was `43.30%` and within-one-state
accuracy was `77.17%`. These descriptive values cannot rescue the failed
representation gate.

For the descriptive common `0/1/2/>=3` collapse, current GAT averaged `59.07%`
exact accuracy, `40.94%` balanced accuracy, and `29.14%` positive exact
accuracy. The prior one-core true-Normal categorical experiment reported about
`91.77%`, `28.1%`, and `1.5%`, respectively, against an all-zero exact
reference of `91.70%`. The studies differ in tissue context, biological-unit
count, model, objective, output support, and continuous channel, so the
difference cannot be attributed to tokenization alone.

The maximum supported outcome-specific conclusion is: in held-in masked
expression across these ten adjacent-normal cores, the hybrid model shows
partial representation capacity for detection and robust continuous positive
values, but not for the frozen ordered positive-state criterion; the data do
not support broad-context graph gain over the exactly parameter-matched
self-only control. No patient-held-out, true-Normal, direct-interaction,
mechanistic, or causal claim is supported.

Canonical reports and complete tables are under
`reports/analyses/adjacent_normal_10core_hybrid_count_gat/comparison/`:
`report.md`, portable single-file `report.html`, `comparison.json`, and the
per-mask, per-core, inference, resource, failure, and verification CSVs.

Final reconciliation found 22 completed attempt-one runs and queue jobs
(two pilot and twenty production), zero campaign failures, and 22 present,
verified canonical checkpoints. All 21 files covered by the report manifest
matched their recorded SHA-256 checksums. The repository-wide artifact
verifier returned `valid: true`; a post-finalization campaign-scoped
re-verification also returned `valid: true` with no registry or bundle issues.
The focused campaign suite passed 170 tests. The final project doctor reported
schema version 3, database integrity `ok`, no issues or warnings, and no
claimed, queued, or running jobs. The campaign registry status is `complete`;
all eight GPUs were idle after execution.
