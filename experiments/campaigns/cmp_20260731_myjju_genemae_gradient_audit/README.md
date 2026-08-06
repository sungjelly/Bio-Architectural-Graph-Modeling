# MyJJu GeneMAE gradient claim audit

## Status

- Phase: complete
- Outcome: blocked
- Campaign: `cmp_20260731_myjju_genemae_gradient_audit`
- Design: exploratory, held-in, post-hoc model-behaviour audit
- Upstream campaign:
  `cmp_20260730_myjju_genemae_10core_comparison`
- Frozen contract:
  `experiments/campaigns/cmp_20260731_myjju_genemae_gradient_audit/frozen_task_contract.yaml`

The claim under test is that learned GeneMAE gradients validate biological
mechanisms. That wording is not a valid conclusion from an observational
gradient. This campaign tests narrower prerequisites: whether the published
statistic is reproducible, whether a task-relevant masked-target gradient is
faithful to the frozen model, whether it is stable across seeds, masks, and
cores, whether it depends on the graph, and whether it exceeds prespecified
nulls. Even if all computational gates pass, the maximum claim is a stable,
faithful, null-calibrated model-implied sensitivity. Mechanism validation
requires independent controlled perturbation.

The analysis is exploratory because the source report, its selected marker
pairs, the upstream reconstruction results, and the failed global graph-use
gate were inspected before this contract was written. Thresholds below are
frozen before inspecting any gradients from the seven reproduced checkpoints.

## Task contract

### Objective and deliverables

Audit the source gradient implementation and evaluate all seven frozen
GeneMAE checkpoints on all ten adjacent-normal cores without retraining or
checkpoint selection.

Required deliverables are:

1. an equation- and provenance-level audit of the source statistic;
2. an exact reproduction of that statistic on the frozen ensemble, retaining
   its signed directed form before the source's absolute symmetrisation;
3. task-relevant gradients of masked receiver targets with respect to visible
   inputs, split into same-cell and cross-cell/up-to-four-hop components;
4. seed, mask, and core stability summaries;
5. bounded finite-difference faithfulness checks;
6. observed-graph, node-label-permuted, parameter-randomised, co-expression,
   and matched-pair null controls;
7. target-specific predictive and graph-use eligibility checks;
8. a checksum-bound aggregate run bundle and concise Markdown, HTML, CSV, and
   JSON reports with negative results and failures retained.

### Scientific question, hypotheses, and alternatives

Primary computational question:

> Do the source-selected GeneMAE gradients define stable and faithful
> model-implied sensitivities for masked targets, beyond graph and
> expression-matched nulls?

The primary hypothesis is that at least one locked target passes predictive
eligibility and graph-use eligibility, then its signed gradient passes the
stability, bounded-faithfulness, and null-calibration gates below.

Credible alternatives are:

- the source result is an artefact of using unmasked targets, including the
  target gene itself as input;
- same-cell co-expression dominates and the graph contributes little;
- absolute-value symmetrisation creates plausible undirected modules while
  hiding sign cancellation and directional instability;
- gradient rank is explained by expression prevalence or raw co-expression;
- local derivatives do not predict bounded changes because of curvature,
  saturation, or clipping;
- results vary across training seeds, masks, cores, or graph assignments;
- tiling, self-loops, degree, or one-to-four-hop message paths are mistaken for
  a direct biological interaction; and
- literature agreement is circular because the 39-gene universe and reported
  pairs were selected using those same modules and source gradients.

### Frozen models, cohort, genes, and masks

- Checkpoints: the immutable final epoch-199 checkpoints for seeds `0` through
  `6` from the upstream campaign; every checksum and registry record must
  verify.
- Cohort: aliases `ANC-01` through `ANC-10`, 117,386 held-in cells, 1,000
  ordered biological probes.
- Biological/observational unit: tissue core.
- Technical repeats: seven training seeds and three fixed 20% masks; neither
  is a biological replicate.
- Preprocessing: upstream full-cell `log1p(CP10k)` before masking. Gradients
  are with respect to this compositional normalized input, not raw molecules.
- Graph: the exact canonical tiled symmetric k=15 graph; no cross-core or
  cross-tile edges.
- No row-level identifiers may be written. Cell-level temporary arrays, if
  required for ensemble calculation, remain in the owned active scratch run
  and are excluded from the finalized report bundle.

The 39 marker genes and seven modules are copied verbatim from the external
source before inspecting new gradients:

- epithelial: `EPCAM, KRT8, KRT18, KRT19, CDH1, KRT7, KRT17, KRT5`
- proliferation: `MKI67, PCNA, TOP2A, BIRC5`
- immune-T: `PTPRC, CD3D, CD3E, CD8A, CD4, FOXP3, NKG7`
- immune-myeloid/B: `CD68, CD163, MS4A1, CD79A`
- stroma: `COL1A1, COL1A2, COL3A1, ACTA2, DCN, LUM, PDGFRB`
- endothelial: `PECAM1, VWF`
- cancer-candidate: `PSCA, OLFM4, CEACAM6, CLDN4, LGR5, SOX9, MYC`

The locked source-selected positive-control pairs are:

- `KRT8 <- KRT19, KRT18, KRT7`
- `COL1A1 <- COL3A1, COL1A2, DCN`
- `EPCAM <- PSCA, OLFM4, KRT19, CLDN4, CDH1`
- `OLFM4 <- PSCA, EPCAM, KRT19, CDH1`
- `CEACAM6 <- KRT19, KRT8, CLDN4, KRT18`

These pairs test reproducibility of the source-selected conclusion. They are
not an independent biological validation set.

### Estimands

The reproduced source statistic is

```text
J_source[target, source] =
  (1 / N) sum_input_cells d(sum_output_cells reconstruction[target])
                            / d input[source]
```

It uses an all-false entry mask. It is a global uniform-input directional
derivative over the model's explicit same-cell branch and one-to-four-hop graph
paths. The source publishes
`0.5 * (abs(J_source) + abs(J_source.T))`, which is unsigned and
undirected.

The primary task-relevant statistic uses a fixed 20% entry mask:

```text
J_masked[target, source] =
  (1 / number_masked_target_receivers)
  d(sum reconstruction[masked target receivers, target])
    / d visible input[source]
```

For every backward pass, signed sums and L1 gradient mass are retained
separately for the same receiver cell and for all other input cells. The
cross-cell term includes all paths up to four GAT layers and is not a direct
one-hop interaction.

Bounded faithfulness uses source-gene shifts of `+/-0.10` and `+/-0.25`
within-core standard deviations on visible entries, clipped to the empirical
1st--99th percentile. Gradient dot-products use the exact clipped direction
and are compared with the frozen model's centered finite changes. These
single-gene shifts are numerical faithfulness tests, not biologically complete
counterfactuals.

### Eligibility and gates

A locked target is eligible for gradient interpretation only if, on the exact
three 20% masks and prediction-level seven-model ensemble:

1. it improves equal-core masked Huber by at least 2% over the per-core
   all-fit gene mean and favors the model in at least 8/10 cores; and
2. the observed graph improves equal-core masked Huber by at least 2% over the
   node-label-permuted graph and favors the observed graph in at least 8/10
   cores.

The global upstream graph-use result was already negative (1.09%, below 2%).
That is prior adverse evidence, not a substitute for the new target-specific
check. A target failing either eligibility condition is not biologically
interpreted.

The gradient stability gate requires:

- median pairwise Spearman correlation of off-diagonal absolute masked
  39-by-39 gradient ranks across seven seeds of at least 0.70 in at least 8/10
  cores;
- median pairwise Spearman correlation across the three masks for each of the
  five locked target rows of at least 0.70 in at least 8/10 cores; and
- any signed pair claim to retain one sign in at least 6/7 seeds and at least
  8/10 cores.

The faithfulness gate requires, separately at both perturbation scales:

- Spearman correlation at least 0.70 between gradient-predicted and actual
  centered changes;
- sign agreement at least 80% for non-negligible changes; and
- median absolute linearisation error divided by median absolute actual change
  at most 0.50.

The null-calibration gate requires:

- the locked pair statistic to exceed the 95th percentile of 10,000
  target-preserving random pair sets matched on source prevalence, expression
  magnitude, and absolute raw co-expression;
- the observed-graph statistic to exceed its node-label-permuted counterpart
  by at least 25% in at least 8/10 cores; and
- trained-model structure to exceed all seven deterministically initialised
  parameter-randomised controls on the locked reference core.

All thresholds are fixed before new gradient inspection. A failed gate remains
negative; seeds, targets, masks, thresholds, and modules will not be changed
post hoc.

### Controls and verification

Positive/numerical controls:

- an analytical tiny-graph model with planted signed same-cell and cross-cell
  effects;
- autograd versus centered finite differences;
- identical checkpoint reload equality.

Negative/null controls:

- masked input entries must have zero input gradient;
- nodes outside the receptive field and across tiles must have zero gradient;
- deterministic node-label-permuted graphs;
- seven parameter-randomised models on the locked reference core;
- target-preserving expression/co-expression-matched random pairs;
- raw co-expression concordance and source-style unmasked versus
  task-relevant masked-gradient concordance.

The pilot must verify finite outputs/gradients, exact gradient decomposition,
masked-entry zeros, deterministic replay, maximum allocated VRAM at most
20.5 GiB, projected runtime at most two hours per seed, and sufficient disk.
Production maps seeds `0,1,2,3,4,5,6` to GPUs `0,1,2,3,5,6,7`.

Stop on a checkpoint, cohort, graph, gene-order, or mask checksum mismatch;
nonfinite output or gradient; failed numerical control; unsafe GPU collision;
resource-gate failure; incomplete seed/core/mask coverage; or artifact/registry
verification failure.

### Split, leakage, circularity, and maximum claims

All ten cores were used to fit every checkpoint. Full-cell normalization uses
hidden entries in the denominator. The analysis is transductive and
held-in—there is no patient-, donor-, core-, or spatial-block-held-out
generalization. The source-selected modules and pairs are circular as
biological validation. Cells are not independent replicates.

The claim ladder is:

- faithfulness only: `faithful local model sensitivity`;
- plus stability: `stable model-implied sensitivity`;
- plus target eligibility and null calibration:
  `stable, faithful, graph-dependent, null-calibrated predictive sensitivity`;
- source/literature agreement only:
  `biologically annotated candidate`, not validation.

This design cannot validate a biological mechanism. That would require an
independent, powered intervention with a prespecified receiver endpoint,
appropriate vehicle/off-target and cell-autonomous controls, pathway or
receptor blockade, rescue, and replication in independent biological units.
`Causal`, `validated mechanism`, `direct interaction`, and patient-generalized
language are prohibited for this campaign.

## Planned execution

The conclusion-bearing aggregate run was prepared as
`r_20260731T081950Z_b2203286_s000_f00_a01_35181af2`. Its owned work directory
is:

```text
scratch/active_runs/r_20260731T081950Z_b2203286_s000_f00_a01_35181af2/diagnostics/audit_work
```

Prepare command (completed):

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/manage_myjju_gradient_audit_run.py \
  --database state/tracking/bagm.sqlite3 prepare
```

Locked resource pilot:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/audit_myjju_genemae_gradients.py \
  --database state/tracking/bagm.sqlite3 \
  --work-root scratch/active_runs/r_20260731T081950Z_b2203286_s000_f00_a01_35181af2/diagnostics/audit_work \
  pilot --device cuda:0
```

If and only if the pilot passes, replace `<PILOT_SHA256>` below with the
reported artifact checksum. Seed 0 is the full-shard memory and correctness
sentinel because production contains larger tiles than the frozen ANC-01
pilot. Seeds 1--6 launch in parallel only after that sentinel verifies.

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/audit_myjju_genemae_gradients.py \
  --database state/tracking/bagm.sqlite3 \
  --work-root scratch/active_runs/r_20260731T081950Z_b2203286_s000_f00_a01_35181af2/diagnostics/audit_work \
  seed-shard --seed 0 --device cuda:0 \
  --reviewed-pilot-sha256 <PILOT_SHA256>
```

The remaining exact seed/device assignments are:

```text
seed 1 -> cuda:1
seed 2 -> cuda:2
seed 3 -> cuda:3
seed 4 -> cuda:5
seed 5 -> cuda:6
seed 6 -> cuda:7
```

Each uses the same `seed-shard` command with its listed `--seed`, `--device`,
and the reviewed pilot checksum. Aggregation writes only to:

```text
scratch/active_runs/r_20260731T081950Z_b2203286_s000_f00_a01_35181af2/diagnostics/audit_work/aggregate
```

The aggregate invocation supplies nine JSON argv records: the pilot, seven
seed shards, and aggregate command itself. Finalization is:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/manage_myjju_gradient_audit_run.py \
  --database state/tracking/bagm.sqlite3 finalize \
  --run-id r_20260731T081950Z_b2203286_s000_f00_a01_35181af2
```

Verification commands are the four focused gradient-audit test modules,
`PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor`, registry
artifact verification, and immutable run-bundle verification. The aggregate
must preserve resolved configuration, upstream run/checkpoint checksums, Git
and dirty-tree provenance, environment/hardware, exact commands, append-only
metrics, resource use, failures, and a completion marker.

## Outcome

The locked CUDA pilot failed only the strict `1e-6` deterministic-replay and
checkpoint-reload controls. Its maximum absolute replay difference was
`2.86102294921875e-6`; all other numerical, resource, disk, and runtime checks
passed. The failed attempt is preserved immutably as
`r_20260731T081950Z_b2203286_s000_f00_a01_35181af2`. This frozen campaign did
not proceed to production and is closed as technically blocked.

The separate, explicitly post-pilot CPU-replay successor preserved every
scientific estimand, target, pair, seed, mask, core, threshold, null, and claim
ceiling. It completed as
`r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a` and returned the negative
verdict `biological_mechanism_not_validated`; the computational candidate-set
claim also failed its graph-gradient structure null.
