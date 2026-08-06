# MyJJu GeneMAE gradient claim audit: CPU replay successor

## Status

- Phase: complete
- Outcome: negative
- Design: exploratory, held-in, post-hoc model-behaviour audit
- Campaign: `cmp_20260731_myjju_genemae_gradient_audit_cpu_replay`
- Predecessor: `cmp_20260731_myjju_genemae_gradient_audit`
- Failed predecessor run:
  `r_20260731T081950Z_b2203286_s000_f00_a01_35181af2`

## Why this is a separate campaign

The predecessor's frozen CUDA pilot failed only its strict `1e-6`
deterministic-replay and checkpoint-reload controls: the maximum absolute
difference was `2.86102294921875e-6`. Every other pilot control passed,
including finite gradients, exact decomposition, masking and receptive-field
checks, VRAM, disk, and the conservative runtime gate. That failure is
registered and immutable.

The upstream GeneMAE workflow explicitly documents that PyG GATv2 CUDA
scatter/reduction kernels can vary in the last float32 bits and performs its
strict checkpoint replay on CPU. This successor tests the stated corrective
hypothesis that the identical checkpoint passes the unchanged `1e-6` replay
gate on CPU. Production gradient extraction remains float32 CUDA. The observed
CUDA discrepancy remains a limitation.

This is a post-pilot amendment, not a confirmatory reset. No gradient matrix,
target eligibility result, stability result, faithfulness result, or null
result was inspected before freezing this successor contract.

## Scientific contract

The complete predecessor estimands, 39 genes, 19 source-selected pairs, five
targets, seven checkpoints, three masks, ten cores, sampling rules,
aggregation rules, perturbation scales, gate thresholds, nulls, and claim
ceiling are inherited without change from the predecessor contract (SHA-256
`5e5b1321f5e9b9c0acfa98d22f1a3348eb856b76fc74ad18d4cc8df835f3a4b5`).
The successor contract is
`frozen_task_contract.yaml` (SHA-256
`bcdd6d7223f9b97f26833f77935c75645110aaba1669d11c0a7180c957514cc1`).

The question remains:

> Do the source-selected GeneMAE gradients define stable, faithful,
> graph-dependent, and null-calibrated model-implied sensitivities for masked
> targets?

The primary hypothesis is that at least one target passes both predictive and
graph-use eligibility, with stable signed gradients that pass both bounded
faithfulness scales and all three null families. Alternatives are dominance
by same-cell co-expression, weak graph dependence, seed/mask/core
instability, nonlinear finite changes, expression-matched null explanation,
or numerical non-reproducibility.

The observational unit is the adjacent-normal tissue core. Seeds and masks are
technical repeats. All 117,386 cells were used for fitting; full-cell CP10k
normalization uses hidden entries in the denominator; core-to-patient
independence is unverified; and no held-out cohort exists.

The primary registry metric is the mean Boolean outcome over the exact 40 gate
rows. It is bookkeeping only, not an evidence score. Eligibility requires at
least 2% Huber improvement and 8/10 favoring cores against both the per-core
gene mean and the node-label-permuted graph. Stability, faithfulness, graph
structure, parameter randomization, and 10,000-draw matched-null thresholds
remain exactly those in the frozen contract.

The maximum possible claim is a stable, faithful, graph-dependent,
null-calibrated model-implied predictive sensitivity. This design cannot
validate a biological mechanism. Controlled perturbation with pathway or
receptor blockade, rescue, appropriate cell-autonomous/off-target controls,
and independent biological replication would be required.

## Acceptance, falsification, and stopping

The successor pilot must pass strict CPU replay at `1e-6`, all inherited
numerical controls, peak allocated VRAM no greater than 20.5 GiB, and projected
runtime no greater than two hours per seed. Seed 0 is then the full-shard
largest-tile memory/correctness sentinel. Production stops on any identity,
finite-value, numerical, resource, coverage, artifact, or registry failure.

Passing the pilot does not support the biological claim. Passing all
computational gates would support only the candidate-set sensitivity claim
above. Any failed gate is retained as negative evidence.

## Inputs, outputs, and verification

Inputs are the seven immutable epoch-199 source-fidelity checkpoints, exact
ten-core cohort fingerprint
`b3c06228ce7fda4d4c5da09c3281de46e67b8069dfadb14276ba6402af87767e`,
and graph fingerprint
`f860f84f96e9daf4c9c4e4ddd5fc40aeacd3f6362c9e97d07248ac2575b462d5`.
The GPU map remains seeds `0,1,2,3,4,5,6` to physical devices
`0,1,2,3,5,6,7`.

Expected outputs are the pilot and seven checksum-bound shards, exact 40 gate
rows, retained directed and published-symmetric source matrices, resource and
failure summaries, portable reports, canonical fit prediction summaries, and
one immutable registered run bundle. Verification uses the four focused
gradient-audit test modules, the production identity checks, repository
doctor, registry artifact checks, and checkpoint-catalog verification.

## Execution

The registered aggregate run is
`r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a`. Its owned work root is:

```text
scratch/active_runs/r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a/diagnostics/audit_work
```

Prepare command (completed):

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/manage_myjju_gradient_audit_run.py \
  --database state/tracking/bagm.sqlite3 prepare
```

Successor pilot:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/audit_myjju_genemae_gradients.py \
  --database state/tracking/bagm.sqlite3 \
  --work-root scratch/active_runs/r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a/diagnostics/audit_work \
  pilot --device cuda:0
```

On a passing pilot, each production command uses:

```text
PYTHONPATH=src /venv/main/bin/python scripts/analysis/audit_myjju_genemae_gradients.py
  --database state/tracking/bagm.sqlite3
  --work-root scratch/active_runs/r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a/diagnostics/audit_work
  seed-shard --seed <SEED> --device cuda:<DEVICE>
  --reviewed-pilot-sha256 <PILOT_SHA256>
```

with exact assignments `0:0, 1:1, 2:2, 3:3, 4:5, 5:6, 6:7`. Seed 0
runs first as the full-shard sentinel. Aggregation writes to
`diagnostics/audit_work/aggregate`, records the pilot, seven shard commands,
and aggregate command as nine JSON argv lists, and is followed by:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/manage_myjju_gradient_audit_run.py \
  --database state/tracking/bagm.sqlite3 finalize \
  --run-id r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a
```

The reviewed pilot checksum and exact expanded pilot, shard, and aggregate
argv lists are retained in the final bundle under
`provenance/audit_execution_commands.json`.

## Current results

The successor pilot passed. Strict CPU deterministic replay and checkpoint
reload both had maximum absolute error `0.0`. All numerical checks passed.
Peak allocated VRAM was `4.6910 GiB`; projected full-shard runtime was
`0.2010 hours/seed` using the maximum node/edge workload ratio, measured
random-control time, context-load time, and a 1.25 safety factor. The reviewed
pilot SHA-256 is
`7e0032163a98f2b5f006fe5b04ee475e5ea5cba553af2d804d72bcde32a8806c`.

The exact seed-0 sentinel command is:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/audit_myjju_genemae_gradients.py \
  --database state/tracking/bagm.sqlite3 \
  --work-root scratch/active_runs/r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a/diagnostics/audit_work \
  seed-shard --seed 0 --device cuda:0 \
  --reviewed-pilot-sha256 7e0032163a98f2b5f006fe5b04ee475e5ea5cba553af2d804d72bcde32a8806c
```

All seven production shards completed. Each passed all ten numerical and
coverage diagnostics. Per-seed runtime was `568.4`--`595.5` seconds and peak
allocated VRAM was `8.329 GiB` per seed on NVIDIA GeForce RTX 3090 devices.

The locked aggregate result is **negative**:

> `biological_mechanism_not_validated`

The stronger computational candidate claim also failed. The exact gate
summary was:

| Gate | Passed | Total |
|---|---:|---:|
| target predictive eligibility | 5 | 5 |
| target graph-use eligibility | 1 | 5 |
| seed-rank stability | 1 | 1 |
| mask-rank stability | 5 | 5 |
| signed-pair stability | 18 | 19 |
| bounded faithfulness | 2 | 2 |
| graph-gradient structure null | 0 | 1 |
| parameter randomization | 1 | 1 |
| matched-pair null | 1 | 1 |

Only `KRT8` met target graph-use eligibility: its equal-core Huber loss was
`31.93%` better than the gene mean and `7.47%` better than the node-label
permuted graph, with all ten cores favoring the observed graph. The other
targets' equal-core graph gains were `0.74%` (`COL1A1`), `-0.03%` (`EPCAM`),
`0.14%` (`OLFM4`), and `0.09%` (`CEACAM6`), below the frozen `2%` gate.

The decisive negative control was the graph-gradient structure test. In every
core, the locked-pair enrichment statistic `S` was lower on the observed graph
than after node-label permutation; therefore `0/10` cores passed the frozen
requirement of at least `25%` improvement in at least `8/10` cores. This
falsifies the prespecified computational candidate-set claim even though
gradient ranks were seed- and mask-stable, both local finite-difference tests
were numerically faithful, the trained model exceeded all seven
parameter-randomized controls, and the selected pairs exceeded the
expression/co-expression-matched null (`p=9.999e-5`). `OLFM4<-KRT19` was also
directionally unstable across seeds and cores.

The published-style statistic remains biologically non-validating regardless
of those controls. It is observational and held-in; combines same-cell and
up-to-four-hop paths; uses source-selected genes and pairs; has no verified
patient replication, independent cohort, orthogonal assay, or controlled
perturbation; and its absolute symmetrization removes direction and sign. The
maximum defensible conclusion from this run is
`no_claim_beyond_reported_model_behavior`.

The immutable result is run
`r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a` under
`artifacts/runs/2026/07/r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a/`.
The concise report is `interpretation/report.md` (SHA-256
`9dbcd5b2840c92bf9fdd3e1105377c1bd7fe4625e525e182528d14ef37573a50`);
all 40 locked gate rows are in `interpretation/gate_results.csv`.

Final verification passed: the finalizer verified all 60 run-bundle files and
was idempotent on replay; registry and bundle artifact checks reported no
issues; the analysis-state checkpoint is indexed with verification status
`verified`; all 49 focused tests passed; and `spatial_benchmark doctor`
reported database integrity `ok` with no issues or warnings.
