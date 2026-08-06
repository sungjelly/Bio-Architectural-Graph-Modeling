# MyJJu GeneMAE ten-core comparison

## Status

- Phase: complete
- Outcome: supported for the frozen primary common-task gate; graph-use result
  negative
- Registry status: complete
- Campaign: `cmp_20260730_myjju_genemae_10core_comparison`
- Design: exploratory held-in reproduction and end-to-end partial-gene system
  comparison
- Current BAGM comparator:
  `cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble`
- Authoritative report:
  `reports/analyses/myjju_genemae_10core_comparison/comparison_v2/report.md`
- Authoritative manifest SHA-256:
  `463617420a1ab671d7794e5c88c0c7319c5960ca28fb0e26a83d3408b9be2e41`
- Registry reconciliation:
  `reports/analyses/myjju_genemae_10core_comparison/registry_reconciliation/campaign_finalization_20260730T191024047253Z.json`

This campaign addresses the request to run the latest and best task-compatible
model from `/workspace/Gastric-Cancer-Analysis-by-MyJJu` on the ten
pathology-confirmed adjacent-normal cores and compare it with the current BAGM
model.

The external repository has no checkpoint. Its latest dual-path GeneMAE source
also cannot instantiate because `self.self_hidden` is read before assignment.
Accordingly, this campaign reproduces and retrains the documented architecture
with one explicit source-fidelity repair:

```python
self.self_hidden = int(self_hidden)
```

It does not claim to evaluate the missing historical weights. The external
supervised lesion classifier is excluded because the ten-core cohort contains
only adjacent-normal tissue, so lesion-versus-normal AUC is undefined and is
not comparable with masked-expression reconstruction.

The analysis is exploratory. The external metrics, the current BAGM outcome,
and the ten-core count distribution were inspected before this contract was
written.

## Task contract

### Objective and deliverables

Reproduce the report-selected dual-path GeneMAE architecture, train seven
prespecified pooled seeds on all ten adjacent-normal cores, evaluate it on
fixed partial-gene masks, and compare it with the immutable seven-member BAGM
GAT and matched-self ensembles on a common prediction scale.

Required deliverables are:

1. a checksum-bound external-source and source-repair audit;
2. deterministic alias-safe k=15 graph and log1p(CP10k) input preparation;
3. focused unit tests and a two-epoch resource pilot;
4. seven registered 200-epoch production runs with final checkpoints;
5. prediction-level GeneMAE ensemble evaluation on the exact three BAGM 20%
   partial-gene masks per core;
6. a source-native 50% partial-gene evaluation and a node-label-permuted graph
   null;
7. a common-scale comparison with the current BAGM GAT ensemble, current
   matched-self ensemble, all-zero reference, and per-gene mean reference; and
8. concise Markdown and portable single-file HTML reports, complete tables,
   machine-readable results, provenance, failures, and a checksum manifest.

### Scientific question, hypothesis, and alternatives

The primary question is whether the reproduced GeneMAE ensemble predicts held-in
masked gene entries more accurately than the current pooled BAGM GAT ensemble
when both are scored on identical 20% partial-gene masks in full-cell
log1p(CP10k) space.

The primary hypothesis is that GeneMAE reduces equal-core masked Huber loss by
at least 2%, at least eight of ten cores favor it, and at least five of seven
same-numbered technical seed comparisons favor it.

Credible alternatives are:

- GeneMAE's apparent advantage in its source report came from easier masking,
  one-seed variation, checkpoint selection on the evaluation donor, or
  full-cell library-size normalization;
- BAGM's raw-count objective, morphology inputs, and broad-context graph transfer
  poorly to the normalized partial-gene metric even if they help whole-node
  reconstruction;
- both models mainly learn intracellular gene means or co-expression, with
  little graph-specific information; and
- spatial tiling removes cross-tile information and limits GeneMAE.

Predictions that distinguish these explanations are fixed before the new
outcome is inspected:

- a genuine GeneMAE reconstruction advantage must survive all seven seeds,
  equal-core aggregation, the same 20% masks, and the per-gene mean reference;
- graph-specific information must degrade under a fixed node-label permutation
  that preserves each tile's topology and degree sequence; and
- a gain visible only at the source-native 50% mask rate, or only in pooled
  correlation but not Huber/MAE and per-gene correlation, is not a common-task
  win.

### Estimand and maximum claim

The primary estimand is held-in partial-gene reconstruction: 20% of genes are
masked in every cell and predicted from the remaining genes plus each model's
permitted context. The secondary native estimand uses 50% masks for GeneMAE
only.

All ten cores and all their cells are used for fitting. This is transductive
capacity evidence, not held-out-core or patient generalization. The maximum
claim is a descriptive comparison of retrained model architectures on
held-in partial-gene reconstruction across ten adjacent-normal cores.

Whole-node, spatial-block, lesion classification, interaction, mechanism, and
causal claims are prohibited. The existing BAGM whole-node result is reported
separately and is not ranked against GeneMAE.

### Experimental units, models, and inputs

- Biological/observational unit: tissue core, aliases `ANC-01` through
  `ANC-10`.
- Total cells: 117,386.
- Ordered biological probes: 1,000; exact equality is required.
- Technical model seeds: `0` through `6`; seeds are not biological replicates.
- External model: dual-path GeneMAE, 6,888,016 expected trainable parameters.
- Current comparators: all seven final pooled BAGM GAT checkpoints and all
  seven final pooled matched-self checkpoints. No favorable BAGM seed is
  selected.

GeneMAE receives only biological expression and its explicit mask. It does not
receive identifiers, alias, slide, disease label, vendor annotations,
morphology, RNA-derived QC, or absolute coordinates as node covariates.
Coordinates are used only to construct the spatial graph.

Full-cell log1p(CP10k) is computed before entry masking to reproduce the source
model. Hidden entries therefore contribute to the library-size denominator.
This is recorded as target-derived transductive preprocessing and prevents a
strict target-hidden or whole-node claim.

### Architecture and graph

The reproduced architecture is fixed to the source report:

- symmetric spatial kNN with `k=15`;
- recursive median spatial tiling at at most 7,000 cells;
- one Gaussian distance-kernel edge feature using the tile median edge length;
- four GATv2 layers, hidden width 256, six heads with `concat=False`,
  LayerNorm, residuals after the first layer, ELU, dropout 0.2, DropEdge 0.2,
  and concatenated jumping knowledge projected to width 192;
- a node-wise `1000 -> 512 -> 512 -> 192` self branch;
- concatenated graph/self embeddings projected through width 256 to 1,000
  outputs;
- one learned mask value per gene; and
- Huber delta 1 on masked entries only, with no SCE term.

Edges never cross cores or tiles. The graph is local spatial context, not a
direct communication graph. The fixed graph null applies one deterministic
node-label permutation per tile to both rows of `edge_index`, preserving
topology and degree while breaking the graph-to-cell assignment.

### Training and evaluation

Production uses AdamW, learning rate `1.5e-3`, weight decay `1e-4`, cosine
schedule, gradient clipping 5, FP32, 50% Bernoulli training masks, no early
stopping, no validation selection, and exactly 200 global epochs. Every
spatial tile is visited once per epoch in deterministic seed-specific order.
The final epoch is the only checkpoint.

Common evaluation regenerates the current campaign's exact three 20%
partial-gene mask replicates for every core and verifies their checksums.
GeneMAE native evaluation uses three separate fixed 50% masks. Metrics are
computed on masked entries, averaged across mask replicates within core, then
across the ten cores without cell-count weighting.

Prediction ensembles average GeneMAE normalized-log predictions and recompute
metrics. The current BAGM ensembles retain their canonical combination of
detection and ordinal probabilities plus continuous predictions before
decoding. For the common comparison, decoded BAGM counts are transformed to
log1p(CP10k) with the same true full-cell library size used by GeneMAE. This is
an oracle-scale descriptive comparison and not a strict masking result.

Primary metric: masked log1p(CP10k) Huber, lower is better.

Required secondary metrics are masked MSE, MAE, pooled Pearson, R2, mean and
median per-gene Pearson, mean cell-wise Pearson, seed variability, core
favoring count, runtime, peak VRAM, peak host memory, convergence, parameter
count, and failures.

### Controls and decision gates

Required controls are:

- all-zero normalized expression;
- per-core all-fit per-gene normalized-expression mean;
- current pooled BAGM matched-self ensemble;
- current pooled BAGM GAT ensemble;
- GeneMAE node-label-permuted graph evaluation; and
- the source-reported metrics as historical context only.

The primary comparison gate passes only if GeneMAE improves mean equal-core
20% masked Huber by at least 2%, at least eight cores favor GeneMAE, at least
five of seven same-numbered seed comparisons favor GeneMAE, and masked MAE and
mean per-gene Pearson are not worse.

The baseline gate passes only if GeneMAE improves masked Huber by at least 2%
over the per-gene mean reference in at least eight cores.

The graph-use gate passes only if the unpermuted GeneMAE ensemble improves
masked Huber by at least 2% over its node-label-permuted graph evaluation in at
least eight cores. Failure means the experiment does not support
graph-specific predictive gain.

Gate failure is a valid negative result and does not authorize post-hoc
threshold, seed, mask, or checkpoint changes.

### Pilot, resources, and stop criteria

A registered seed-0 pilot runs two epochs. Production is blocked unless:

- the repaired model instantiates with exactly 6,888,016 parameters;
- every expected tile is visited once per epoch;
- all losses, gradients, parameters, and evaluation metrics are finite;
- peak allocated VRAM is at most 20.5 GiB;
- peak host memory is at most 40 GiB;
- projected 200-epoch runtime is at most six hours per seed; and
- projected final free disk is at least 27.5 GiB.

Initial production devices are GPUs `0,1,2,3,5,6,7`; GPU 4 remains excluded
until the worker guard explicitly supports this runner. Existing workers must
be safely reloaded after the new command mapping is verified.

Stop on a source checksum mismatch, gene-order mismatch, core/mask checksum
mismatch, cross-core edge, parameter-count mismatch, nonfinite value, failed
pilot gate, unsafe GPU collision, insufficient disk, or artifact/registry
verification failure.

## Results and decision

### Execution evidence

The successful resource pilot was
`r_20260730T162022Z_5bdedc78_s000_f00_a01_78c5fb9b`. It covered every tile
twice, produced finite losses, gradients, parameters, and metrics, and passed
an exact CPU checkpoint replay. Peak allocated VRAM was 8.47 GiB, peak host
memory was 4.90 GiB, and the projected 200-epoch runtime was 0.33 hours.

All seven prespecified production seeds completed exactly 200 epochs with the
final epoch-199 checkpoint:

| Seed | Run ID |
|---:|---|
| 0 | `r_20260730T162653Z_bbf1b999_s000_f00_a01_48b2356a` |
| 1 | `r_20260730T162705Z_bbf1b999_s001_f00_a01_d1250f8e` |
| 2 | `r_20260730T162704Z_bbf1b999_s002_f00_a01_76a04093` |
| 3 | `r_20260730T162705Z_bbf1b999_s003_f00_a01_a4365c93` |
| 4 | `r_20260730T162705Z_bbf1b999_s004_f00_a01_cf97a923` |
| 5 | `r_20260730T162705Z_bbf1b999_s005_f00_a01_00bfa079` |
| 6 | `r_20260730T162705Z_bbf1b999_s006_f00_a01_afbbd08f` |

There were no production failures or retries. Mean production runtime was
20.26 minutes per seed; the maxima were 8.50 GiB allocated VRAM and 4.91 GiB
host memory. All seven successful bundles, registered artifacts, and
schema-v3 final-checkpoint catalog entries verified.

Three failed resource-pilot attempts remain registered:

- `r_20260730T160114Z_5bdedc78_s000_f00_a01_25eea03e`: an index-less CUDA
  device was passed to `torch.cuda.set_device`;
- `r_20260730T160518Z_5bdedc78_s000_f00_a01_ce31dcef`: a mask-source mapping
  was accessed with the wrong interface; and
- `r_20260730T160920Z_5bdedc78_s000_f00_a01_365ddd7f`: a strict CUDA replay
  tolerance exposed PyG/CUDA last-bit nondeterminism. The replacement gate
  used deterministic CPU replay, which was exact across 3,665,000 outputs.

Each error changed the implementation approach before the next attempt; no
blind retry or favorable result omission was used.

### Common 20% partial-gene result

The prediction-level equal-core ensembles produced:

| Model or control | Huber ↓ | MSE ↓ | MAE ↓ | Pooled Pearson ↑ | Mean gene Pearson ↑ |
|---|---:|---:|---:|---:|---:|
| GeneMAE, observed graph | 0.348210 | 1.493567 | 0.547090 | 0.312950 | 0.184500 |
| GeneMAE, node-label-permuted graph | 0.352045 | 1.510642 | 0.552088 | 0.297431 | 0.179459 |
| Current BAGM pooled GAT | 1.286428 | 6.175874 | 1.481547 | 0.199152 | 0.127020 |
| Current BAGM matched self | 1.128644 | 5.357008 | 1.303384 | 0.201860 | 0.120578 |
| Target-derived per-core gene mean | 0.390082 | 1.542720 | 0.700893 | 0.222682 | numerically degenerate |
| All zero | 0.358824 | 1.792379 | 0.406596 | undefined | undefined |

The frozen decisions were:

| Gate | Result | Evidence |
|---|---|---|
| Primary GeneMAE versus BAGM GAT | pass | 72.93% lower equal-core Huber; GeneMAE favored in 10/10 cores and 7/7 same-numbered technical seed pairs; MAE and mean gene Pearson non-worse |
| Gene-mean baseline | pass | 10.73% lower equal-core Huber; GeneMAE favored in 10/10 cores |
| GeneMAE graph use | fail | observed graph was only 1.09% better than the node-label-permuted graph, below the frozen 2% threshold, despite the same direction in 10/10 cores |

The primary outcome is therefore `supported`, but only in the narrow
prespecified sense: the reproduced GeneMAE end-to-end system performed better
than the current BAGM GAT end-to-end system on this held-in partial-gene task.
It does not show that the GeneMAE architecture is intrinsically superior.

Contradictory and metric-dependent evidence remains material:

- GeneMAE was only 2.96% better than all zero by Huber and was 34.55% worse
  than all zero by MAE. The result is not uniformly best across controls or
  metrics.
- BAGM GAT Huber was 13.98% higher than matched self on this partial-gene
  task, although the separate BAGM whole-node graph gate passed.
- The GeneMAE graph-null gate failed, so this experiment does not support
  graph-specific predictive gain or a spatial biological mechanism.
- The per-core gene-mean predictor is constant within core for each gene, so
  its near-zero gene Pearson is undefined or numerically degenerate rather
  than evidence of no association.

The current BAGM whole-node result remains a distinct estimand: GAT hybrid loss
was 0.431869 versus 0.445254 for matched self, a 3.01% graph gain favoring GAT
in 10/10 cores and 7/7 seed pairs. That campaign was nevertheless negative
because its pooled-data and representation gates failed. These whole-node
losses are not ranked against GeneMAE.

The source-native GeneMAE 50% result was Huber 0.332818 on the observed graph
and 0.336632 under the graph null. It is not compared with BAGM.

### Audit, supersession, and maximum claim

The authoritative version-2 report contains all 180 evaluation batches, 1,260
member-batch metric rows, all 21 final comparison checkpoints, eight pilot
attempts across the two campaigns, and all five registered pilot failures.
Its 40-file manifest and all target, mask, gene-order, checkpoint, and input
checksums verified. The registry finalizer reconciled all four GeneMAE pilots
and all seven production slots, then performed a verified no-op because the
campaign was already complete.

The initial immutable report under
`reports/analyses/myjju_genemae_10core_comparison/comparison/` is superseded:
it omitted the four GeneMAE resource-pilot rows from its failure tables.
Its predictive metrics and gate decisions were valid. Replaying for version 2
changed the largest member-level metric by `1.89e-8` and equal-core Huber by
less than `6e-12`, consistent with the already diagnosed CUDA/PyG last-bit
nondeterminism; masks, favor counts, and gate decisions were unchanged.

One BAGM replay emitted a warning when a read-only NumPy coordinate view was
wrapped as a tensor. The downstream path only reads coordinates and the two
replays agreed within the tolerances above; the warning remains documented
rather than silently suppressed.

The maximum defensible claim is that the reproduced, retrained GeneMAE
ensemble met the frozen descriptive common-task comparison gate against the
current BAGM GAT ensemble on held-in 20% partial-gene reconstruction across
these ten adjacent-normal cores. This is not held-out generalization, a
controlled architecture ablation, true-Normal performance, graph-specific
evidence, a biological mechanism, or a causal result.

The next discriminating experiment is a patient- or core-held-out comparison
that trains both systems under the same partial-gene objective, normalization,
permitted inputs, graph, tuning budget, and non-oracle library-size contract.
A spatial-block test is also required before excluding local copying.

## Execution and reproduction commands

The focused freeze suite and successful materialization/pilot commands were:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/infrastructure/test_configuration.py \
  tests/unit/infrastructure/test_queue_command.py \
  tests/unit/spatial_benchmark/test_myjju_genemae.py \
  tests/unit/spatial_benchmark/test_myjju_genemae_pooled_runner.py \
  tests/unit/spatial_benchmark/test_myjju_genemae_comparison.py
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/materialize_myjju_genemae_campaign.py materialize
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 enqueue-experiment \
  --campaign-id cmp_20260730_myjju_genemae_10core_comparison \
  --config scratch/locked_campaigns/cmp_20260730_myjju_genemae_10core_comparison/pilot_configs/seed-00_myjju_genemae_resource_pilot.yaml \
  --priority 30 --max-attempts 2 --gpu 0
supervisorctl start bagm-multicore-qkv-gpu0
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/materialize_myjju_genemae_campaign.py verify-pilot \
  --pilot-run artifacts/runs/2026/07/r_20260730T162022Z_5bdedc78_s000_f00_a01_78c5fb9b
```

The independently verified pilot gate receipt checksum is
`1792ba1323cdfa63c2954ae2dbb30246ba1ea505228169ec3b2b3a4b3d4957cb`.
Production is authorized on the prespecified GPU map
`seed 0,1,2,3,4,5,6 -> GPU 0,1,2,3,5,6,7`. The exact production enqueue
commands are:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark --database state/tracking/bagm.sqlite3 enqueue-experiment --campaign-id cmp_20260730_myjju_genemae_10core_comparison --config scratch/locked_campaigns/cmp_20260730_myjju_genemae_10core_comparison/production_configs/seed-00_myjju_genemae.yaml --priority 30 --max-attempts 2 --gpu 0
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark --database state/tracking/bagm.sqlite3 enqueue-experiment --campaign-id cmp_20260730_myjju_genemae_10core_comparison --config scratch/locked_campaigns/cmp_20260730_myjju_genemae_10core_comparison/production_configs/seed-01_myjju_genemae.yaml --priority 30 --max-attempts 2 --gpu 1
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark --database state/tracking/bagm.sqlite3 enqueue-experiment --campaign-id cmp_20260730_myjju_genemae_10core_comparison --config scratch/locked_campaigns/cmp_20260730_myjju_genemae_10core_comparison/production_configs/seed-02_myjju_genemae.yaml --priority 30 --max-attempts 2 --gpu 2
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark --database state/tracking/bagm.sqlite3 enqueue-experiment --campaign-id cmp_20260730_myjju_genemae_10core_comparison --config scratch/locked_campaigns/cmp_20260730_myjju_genemae_10core_comparison/production_configs/seed-03_myjju_genemae.yaml --priority 30 --max-attempts 2 --gpu 3
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark --database state/tracking/bagm.sqlite3 enqueue-experiment --campaign-id cmp_20260730_myjju_genemae_10core_comparison --config scratch/locked_campaigns/cmp_20260730_myjju_genemae_10core_comparison/production_configs/seed-04_myjju_genemae.yaml --priority 30 --max-attempts 2 --gpu 5
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark --database state/tracking/bagm.sqlite3 enqueue-experiment --campaign-id cmp_20260730_myjju_genemae_10core_comparison --config scratch/locked_campaigns/cmp_20260730_myjju_genemae_10core_comparison/production_configs/seed-05_myjju_genemae.yaml --priority 30 --max-attempts 2 --gpu 6
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark --database state/tracking/bagm.sqlite3 enqueue-experiment --campaign-id cmp_20260730_myjju_genemae_10core_comparison --config scratch/locked_campaigns/cmp_20260730_myjju_genemae_10core_comparison/production_configs/seed-06_myjju_genemae.yaml --priority 30 --max-attempts 2 --gpu 7
supervisorctl start bagm-multicore-qkv-gpu0 bagm-multicore-qkv-gpu1 bagm-multicore-qkv-gpu2 bagm-multicore-qkv-gpu3 bagm-multicore-qkv-gpu5 bagm-multicore-qkv-gpu6 bagm-multicore-qkv-gpu7
```

After every production slot is terminal, index any retained failed-attempt
checkpoint that was published before the worker's success-only indexing hook,
verify all artifacts, run the locked comparison, and finalize the campaign
registry only from the explicitly reviewed comparison-manifest checksum:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  index-checkpoints \
  --run-id r_20260730T160920Z_5bdedc78_s000_f00_a01_365ddd7f
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/compare_myjju_genemae_10core.py \
  --database state/tracking/bagm.sqlite3 \
  --device cuda:0 \
  --output reports/analyses/myjju_genemae_10core_comparison/comparison_v2
MYJJU_COMPARISON_MANIFEST=reports/analyses/myjju_genemae_10core_comparison/comparison_v2/manifest.json
MYJJU_COMPARISON_MANIFEST_SHA256=$(sha256sum "$MYJJU_COMPARISON_MANIFEST" | cut -d' ' -f1)
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/finalize_myjju_genemae_campaign_registry.py \
  --database state/tracking/bagm.sqlite3 \
  --comparison-manifest "$MYJJU_COMPARISON_MANIFEST" \
  --expected-manifest-sha256 "$MYJJU_COMPARISON_MANIFEST_SHA256"
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 doctor
```

The finalizer is versioned and campaign-specific. It verifies the complete
report inventory, the outcome implied by the frozen primary gate, exact seeds
0 through 6, final epoch-199 checkpoints, terminal queue/run state, and the
preserved failed-pilot inventory. It takes an online SQLite backup and an
append-only checksum-bound receipt before changing only the campaign registry
status from `planned` to `complete`; it never updates run, queue, artifact, or
checkpoint rows.
