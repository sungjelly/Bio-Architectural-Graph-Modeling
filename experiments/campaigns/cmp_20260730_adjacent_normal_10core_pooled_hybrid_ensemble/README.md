# Pooled ten-core adjacent-normal hybrid-count ensemble

## Status

- Phase: complete
- Outcome: negative
- Campaign: `cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble`
- Design: exploratory held-in pooled masked-expression reconstruction
- Pilot gate: passed
- Production: all fourteen prespecified runs completed; zero production
  failures
- Analysis completed: `2026-07-30T09:00:57Z`
- Registry finalized: `2026-07-30T09:08:42.737296Z`

This campaign trains one shared parameter set on all ten pathology-confirmed
adjacent-normal cores, referred to only as `ANC-01` through `ANC-10`. They are
not true-Normal samples. The analysis is exploratory because the ten-core
count distribution and the prior independently fitted hybrid-count results
were inspected before this contract was written.

The machine-readable contract is `frozen_task_contract.yaml`. It was frozen
before pooled-model implementation. Scientific fields, seeds, gates, masks,
graphs, loss weights, and epoch budget must not change after pilot or
production outcomes are observed. A resource-only variant must be separately
named and documented.
The frozen contract SHA-256 is
`c6af3dc756155ee502506f08304a7436ae99da36ad2b4ed8fae48672a312f6e2`.

## Task contract

### Objective and hypotheses

The primary question is whether sharing one model across all 117,386 cells
improves held-in masked-expression reconstruction relative to fitting one
model independently per core. Secondary questions are whether a
prediction-level ensemble improves over its seven individual members and
whether the broad-context GAT improves over an exactly parameter-matched
self-only ensemble.

Primary hypotheses:

1. pooled seed-0 GAT improves detection BCE, positive ordinal MAE, and
   reconstructed-count `log1p` MAE by at least 2% over the corresponding prior
   independently fitted seed-0 GAT, with at least eight cores favoring pooling
   on each metric;
2. the seven-member GAT prediction ensemble improves whole-node hybrid loss
   by at least 2% over the seven-member matched-self ensemble, at least eight
   cores and five seed pairs favor GAT, and positive ordinal and continuous
   metrics are not worse;
3. the GAT ensemble improves both positive metrics by at least 2% over the
   prespecified per-gene references in at least eight cores and exceeds both
   detection references;
4. prediction-level ensembling is no worse than the mean individual GAT
   member on whole-node hybrid loss and both positive metrics.

Credible alternatives are that core-specific distributions conflict, the
larger update budget overfits held-in masks, cell-autonomous covariates explain
the usable signal, broad `k=1000` context adds no information, or the frozen
ordinal objective/decoder mismatch persists. A failed gate is a valid negative
result.

### Model, pooling, and controls

Each ensemble member is one model object with one optimizer and one shared
state dictionary. A global epoch contains the ten complete core graphs as ten
sequential graph batches in deterministic shuffled order, producing ten
optimizer steps. Every core is visited exactly once per epoch for 200 epochs,
or 2,000 optimizer steps per member.

The graphs remain ten disconnected exact mutual-`k=1000` graphs. No
cross-core edge is created, and cores are never concatenated on GPU. The graph
is broad regional context and cannot be interpreted as direct communication.
Core aliases, coordinates, identifiers, RNA-derived QC, target-cell library
size, vendor annotations, and hidden targets are prohibited node inputs.

The GAT and self arms retain the previous 512-wide, four-head, two-layer
hybrid architecture and must each contain exactly 11,674,880 trainable
parameters. Within every seed, common encoder and decoder initialization must
be bit-identical. The self arm receives no topology, neighboring-cell values,
coordinates, or measured edge attributes.

Seven paired production seeds (`0` through `6`) are fixed. Seeds are technical
model replicates, not biological replicates, and no member may be selected or
dropped because of its result.

### Representation, preprocessing, and objective

The raw-count token boundaries remain:

| Token | Raw count |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 2 |
| 3 | 3 |
| 4 | 4–7 |
| 5 | 8–15 |
| 6 | 16–31 |
| 7 | 32 or greater |

Token `8` is input-only `MASK`. Counts must be finite, nonnegative integers.
The encoder combines gene-specific token projections, exact within-bin
standardized `log1p(count)`, and the same 22 permitted morphology/imaging
covariates. The explicit mask is authoritative and forces the continuous
input to zero.

One immutable expression mean and scale is fitted as an equal-core mixture of
within-core first and second `log1p(count)` moments. This prevents the largest
core from receiving almost twice the normalization weight of the smallest.
Existing per-core all-fit morphology/imaging and edge standardization remains
unchanged and is recorded as transductive preprocessing.

The loss is computed and balanced within each core batch:

1. equal zero/positive detection BCE;
2. equal below/above cumulative BCE at each of six positive thresholds;
3. positive-only standardized-`log1p` Huber with delta 1;
4. equal mean of the three components.

An absent expected stratum fails closed. AdamW, learning rate `3e-4`, weight
decay `1e-4`, gradient clipping `1.0`, deterministic execution, no neighbor
sampling, no edge dropout, no early stopping, and final-epoch checkpointing
are fixed.

### Estimands, evaluation, and ensemble

Whole-node masking is primary. Partial-gene and spatial-block masking remain
separate secondary estimands. The exact prior fixed per-core evaluation masks
are regenerated and checksum-verified so the pooled seed-0 comparison uses
the same targets.

Metrics are first averaged across mask replicates within a core, then across
the ten cores without cell-count weighting. The seven model seeds do not
increase the biological sample size.

The ensemble averages detection probabilities, ordinal-threshold
probabilities, and shared-standardized continuous predictions before decoding
and recomputing every metric. Averaging nonlinear member metrics is not an
ensemble evaluation.

Required references are all-zero, per-core all-fit per-gene references,
equal-core pooled per-gene references, the exactly parameter-matched
self-only ensemble, and the prior independently fitted seed-0 models.

All previous hybrid metrics remain required, including loss components,
detection operating characteristics, eight-state and positive-state metrics,
per-state support/recall, reconstructed-count error, the descriptive
`0/1/2/>=3` collapse, convergence, runtime, peak VRAM/host memory, parameter
count, and failures.

Because all ten outcomes share fitted weights, no core-level independence
test is treated as formal inference. The frozen graph thresholds are
descriptive decision gates. A held-out-core claim would require separate
leave-one-core-out fits and is outside this all-ten-core training estimand.

### Pilot and execution gates

Two registered seed-0 pilots, one per arm, run two global epochs over all ten
cores. Production is blocked unless:

- both pilots visit every core in each epoch and complete 20 finite optimizer
  steps;
- parameter counts and paired initialization match exactly;
- the maximum same-weight FP32-versus-AMP loss discrepancy on every core is
  at most `1e-3`;
- peak allocated VRAM is at most 20.5 GiB;
- peak host memory per process is at most 40 GiB;
- projected 200-epoch GAT runtime is at most six hours per member; and
- projected post-production free disk is at least 27.5 GiB.

Safe GPUs are `0,1,2,3,5,6,7`; GPU 4 is excluded. Existing supervisor workers
must be used, and no process may be killed, preempted, or collided with.

### Pilot outcome and operational lineage

The first queued attempt for each pilot failed during archive validation,
before model construction or GPU training. The long-lived workers had retained
the stale validator that accepted only the prior full-core protocol, even
though the source validator had already been extended for the pooled protocol.
The failure category was `invalid_configuration`; it did not measure either
model and did not alter the locked configuration or scientific contract.

The workers reloaded the current validator and separate immutable attempt-2
runs completed successfully. The queue lineage is:

| Arm | Attempt 1 | Attempt-1 result | Attempt 2 | Attempt-2 result |
|---|---|---|---|---|
| `pooled-hybrid-gat-k1000` | `q_5d7dbbe20729a16147a9` / `r_20260730T045810Z_8337b1ce_s000_f00_a01_a16200b5` | Failed before GPU work; `invalid_configuration` | `q_68d645fff082e9ae2035` / `r_20260730T050028Z_8337b1ce_s000_f00_a02_5c2657d3` | Completed; immutable bundle verified |
| `pooled-hybrid-matched-self` | `q_bdef091a4114daacf9a3` / `r_20260730T045810Z_435c46b1_s000_f00_a01_7ecfe24f` | Failed before GPU work; `invalid_configuration` | `q_4e07bb5b05624467e2ea` / `r_20260730T050028Z_435c46b1_s000_f00_a02_150d438e` | Completed; immutable bundle verified |

The measured pilot outcomes were:

| Arm | Completed run | Final training hybrid loss | Training time (s) | Total time (s) | Maximum FP32–AMP loss difference | Peak VRAM (GiB) | Peak host memory (GiB) | Projected 200-epoch runtime (h) | Projected post-campaign free disk (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pooled-hybrid-gat-k1000` | `r_20260730T050028Z_8337b1ce_s000_f00_a02_5c2657d3` | 0.6387317478656769 | 77.21716112014838 | 628.3885920189787 | 0.0000021457672119140625 | 7.339542388916016 | 13.048526763916016 | 2.1378835658090085 | 38.92326736450195 |
| `pooled-hybrid-matched-self` | `r_20260730T050028Z_435c46b1_s000_f00_a02_150d438e` | 0.6234620988368988 | 6.222977976081893 | 417.62978342501447 | 0.0000019073486328125 | 2.223557472229004 | 5.066135406494141 | 0.16634004232603022 | 39.16045379638672 |

Both completed runs covered all ten aliases exactly once in each of two global
epochs, completed 20 optimizer steps, had finite losses and gradients, used
11,674,880 trainable parameters, and preserved bit-identical paired encoder
and decoder initialization. Their exact evaluation masks and verified graph
bundle also matched. Every measured resource value passed its frozen
threshold. The two-epoch training losses above are pilot diagnostics and are
not scientific results.

The verified pilot-gate receipt is
`scratch/locked_campaigns/cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble/pilot_gate_receipt.json`,
with embedded canonical-payload SHA-256
`549a7de5684279492c43de774d546226161cd8f8e14393d0ac92d4696a7cd068`.
It is linked to pilot enqueue receipt
`84794b3c72e284720d899fe31dfb5e9b195d27c3c8f7c5d32b2ec883e4f21811`
and materialization receipt
`d1a49b0428b1f90ec17f84581264fe4dc4ac3eca56526a0e29bcc761af02fdaa`.

### Production outcome

The passing pilot gate authorized all fourteen locked production jobs. Their
enqueue receipt is
`scratch/locked_campaigns/cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble/production_enqueue_receipt.json`,
with embedded canonical-payload SHA-256
`76ab088c9e26e970cea99827114471e1dda83b450a3423bb7c2e2e77d80aac58`.

All fourteen production slots completed on their first attempt: seven GAT and
seven matched-self runs for seeds `0` through `6`. Each run completed 200
global epochs, 2,000 finite optimizer steps, all 90 fixed-mask evaluations,
and one verified final-epoch checkpoint. Every epoch visited all ten aliases
exactly once. Both arms contained exactly 11,674,880 trainable parameters,
and paired encoder/decoder initialization matched for all seven seeds.

There were zero failed, stale, or retried production attempts. The only
campaign failures were the two pre-GPU pilot attempt-1
`invalid_configuration` events documented above; both immutable attempt-2
pilot retries succeeded.

Production GAT members took 2.27–2.31 hours each and used 7.3414 GiB peak
allocated VRAM. Matched-self members took 0.27–0.28 hours and used 2.2236 GiB.
The fixed GPU assignment was `0,1,2,3,5,6,7`; GPU 4 was not used.

The prespecified convergence diagnostic was unfavorable for the GAT: all
seven last-20-epoch training-loss slopes were positive
(`7.42e-06` to `8.61e-04`), and final training losses
(`0.43925`–`0.44387`) were above their observed minima
(`0.42555`–`0.42676`). All seven matched-self slopes were negative
(`-3.50e-04` to `-5.07e-06`). The fixed final epoch remains canonical because
the contract prohibited early stopping and post-hoc checkpoint selection.
This does not invalidate the final-epoch gates, but it is evidence of late GAT
training degradation or incomplete convergence and weakens any favorable
interpretation.

### Retention decision

Prior canonical bundles occupy disk, not training memory. The prelaunch
snapshot had 47.828 GiB free, while this campaign was projected to require
approximately 6–8 GiB.
Manual deletion would invalidate the registry and checkpoint provenance.
Accordingly, prior canonical evidence is retained and the new campaign uses
fresh immutable run IDs. The reviewed decision and its reconsideration
threshold are in `retention_decision.md`.

After production, comparison, and registry reconciliation, final project
doctor reported 39.596 GiB free, above its 25 GiB minimum and the campaign's
frozen 27.5 GiB projected-free-space gate. No prior canonical bundle or
failed-run record was deleted.

### Final frozen-gate results

The complete comparison is a valid negative result because two of four frozen
gates failed:

| Gate | Result | Locked evidence |
|---|---:|---|
| Pooled-data gate | Failed | Detection BCE improved 2.32% with 10/10 aliases favoring pooled fitting. Positive ordinal MAE improved 3.76% on average but only 7/10 aliases favored pooling. Reconstructed-count `log1p` MAE worsened 8.57%, with only 4/10 aliases favoring pooling. |
| Graph gate | Passed | The GAT ensemble improved whole-node hybrid loss by 3.01% over matched self; 10/10 aliases and 7/7 paired technical seeds favored GAT. Equal-core positive ordinal MAE and positive continuous Huber were not worse. |
| Representation gate | Failed | Detection balanced accuracy was 0.6723 versus 0.5110 for the per-core reference and 0.5090 for the equal-core pooled reference. Positive continuous Huber improved 26.16% in 10/10 aliases, but positive ordinal MAE was 61.16% worse than the per-gene reference and 0/10 aliases favored the model. |
| Ensemble gate | Passed | Prediction-level ensembling improved hybrid loss from the mean-member 0.4354 to 0.4319, positive ordinal MAE from 0.9420 to 0.9331, and positive continuous Huber from 0.3730 to 0.3686. |

The graph result is evidence only for descriptive broad-regional-context
predictive gain in this held-in task. It is not evidence of direct cellular
communication. The representation did learn detection and continuous positive
levels, but it failed the locked ordered-positive-level criterion. High overall
or collapsed exact accuracy cannot rescue that failure.

Pooled seed-0 versus prior independent seed-0 fitting changes more than the
amount of training data: it also changes cross-core weight sharing,
optimizer-step budget, and expression normalization. The failed pooled-data
gate therefore cannot isolate a causal effect of data size or shared fitting.

For descriptive comparison only, collapsing the current eight-state GAT
ensemble to `0/1/2/>=3` gave 62.32% exact accuracy, 42.64% balanced accuracy,
and 30.97% positive exact accuracy. The previous categorical experiment gave
about 91.77% exact, 28.1% balanced, and 1.5% positive exact accuracy. The prior
experiment used one legacy true-Normal core; this campaign used ten
adjacent-normal cores and changed the objective, pooling, architecture, and
ensemble. The difference cannot be attributed to tokenizer choice.

### Final reports and provenance

- Markdown report:
  `reports/analyses/adjacent_normal_10core_pooled_hybrid_ensemble/comparison/report.md`
  (`045650071cd98ee4df27dd14d558715f816768f8e74ac9dd268cf466e8c3de5d`)
- Portable single-file HTML:
  `reports/analyses/adjacent_normal_10core_pooled_hybrid_ensemble/comparison/report.html`
  (`a3024b1dc84eb3ff4f42cbf483926e5b4824306fd95578ea6939cbdb19735432`)
- Machine-readable comparison:
  `reports/analyses/adjacent_normal_10core_pooled_hybrid_ensemble/comparison/comparison.json`
  (`1e128b05453f67730c6c57621815a495e06b21ee3088ae042706fb770540b3e7`)
- Forty-file report manifest:
  `reports/analyses/adjacent_normal_10core_pooled_hybrid_ensemble/comparison/manifest.json`
  (`1ef63f7b9ae739d512a4cc1a8d4e756f782d373b0847b341eb2f84aa00467a6e`)

All 40 manifest entries verify by byte size and SHA-256. The HTML has no
external assets. The report contains the complete per-member, per-alias,
per-mask-mode, per-replicate metrics; per-state support and recall; paired
comparisons; runtimes; convergence; resources; and failure inventory.
Evaluator source, runtime environment, frozen inputs, receipts, run IDs, and
checkpoint hashes are preserved in `provenance.json`.

Final verification completed with 784 tests passed, one expected skip because
the optional `pyarrow` package is absent, and three upstream deprecation
warnings. Repository-wide artifact verification reported zero bundle and
registry issues. Project doctor reported `ok: true`, no issues or warnings,
schema version 3 integrity `ok`, zero queued/running/claimed jobs, and 39.596
GiB free.

Peak-VRAM registry reconciliation changed nine null metadata fields only,
after a checksum-bound plan, online database backup, conditional transaction,
artifact verification, and doctor check. Its application receipt is
`reports/analyses/adjacent_normal_10core_pooled_hybrid_ensemble/registry_reconciliation/peak_vram_application_20260730T075908338478Z.json`
with canonical payload checksum
`cad5d223e1d6b66136eec7e06ae265a323e3d6c36c8b0facc12b77abaeebd809`.

The authoritative campaign registry status was then finalized from `planned`
to `complete` with a config-preserving compare-and-swap after revalidating the
approved report manifest and terminal queue/run inventory. The append-only
receipt is
`reports/analyses/adjacent_normal_10core_pooled_hybrid_ensemble/registry_reconciliation/campaign_finalization_20260730T090842737296Z.json`;
its canonical payload checksum is
`693ca02dd689a62f3e5fc12178e5df690b051ac5e074338eac8fb28cf874894c`.
The transaction changed one campaign row, no queue or run row, and preserved
the registered frozen configuration exactly.

### Maximum claim and completion

The complete exploratory held-in experiment was technically valid. It found
a descriptive broad-context graph gain and a useful prediction-level ensemble,
but failed the locked pooled-data and representation gates because ordered
positive-level recovery was inadequate and reconstructed-count error did not
improve over prior independent fitting. The campaign cannot establish
held-out-core or patient generalization, true-Normal performance, direct
cellular interaction, biological mechanism, or causality.

Completion requires focused and synthetic recovery tests, a passing
two-arm pilot, all fourteen production runs if authorized, member- and
ensemble-level evaluation, immutable verified bundles and checkpoints,
registry reconciliation, practical repository tests, project doctor, and
concise Markdown plus portable single-file HTML reports. An unfavorable
scientific result completes the campaign if execution and verification are
valid. Final verification commands are:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_pooled_ensemble.py \
  tests/unit/spatial_benchmark/test_compare_pooled_hybrid_ensemble.py
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/compare_pooled_hybrid_ensemble.py --device cuda:0
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 doctor
```
