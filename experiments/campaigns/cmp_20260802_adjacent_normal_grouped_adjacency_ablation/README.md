# Grouped adjacent-normal adjacency ablation

## Status

- Phase: planned
- Outcome: pending
- Campaign: `cmp_20260802_adjacent_normal_grouped_adjacency_ablation`
- Design: exploratory donor/core-held-out partial-gene reconstruction

This campaign asks one bounded question: does exchanging masked transcript
information across fixed local spatial edges improve masked-transcript
prediction over the literal identity adjacency? It does not test attention,
edge features, cell labels, cancer tissue, signaling mechanisms, or causality.
The analysis is exploratory because earlier held-in outcomes on this selected
cohort were inspected before this contract was fixed.

The machine-readable contract is `frozen_task_contract.yaml`. Scientific
fields, folds, masks, topology, model capacity, optimizer, checkpoint rule,
metrics, and gates must not change after a smoke or pilot model is evaluated.
Its frozen SHA-256 is
`08e4040ce8b0a3535c7bef1cbf896c68bc5e11693cb13bbdfb3b992467742eff`.

## Task contract

### Objective, hypothesis, and alternatives

The required deliverables are a geometry/preprocessing artifact, focused
tests, paired smoke and resource-pilot runs, the complete paired five-fold by
five-seed comparison, a conditional topology-null comparison, immutable run
bundles, and a core-equal report with tables and plots.

Primary hypothesis: a fixed local spatial adjacency provides information that
reduces donor/core-held-out masked `log1p(count)` Huber loss relative to the
identity adjacency. Credible alternatives are intracellular co-expression
alone, generic neighborhood smoothing or composition, slide/core batch,
segmentation spillover, or unstable optimization. If a real-graph advantage is
observed, a fixed within-FOV random assignment of cells to graph positions
tests actual topology separately from access to generic neighboring cells.

Predictions fixed before training are:

1. a genuine graph contribution lowers Huber and MAE in most held-out cores
   and paired seeds, including the 25--50% and 50--75% target-mask strata;
2. graph benefit should be larger when more true-neighbor values for the target
   gene are observed;
3. if generic smoothing explains the gain, the real and position-permuted
   graphs should perform similarly.

### Estimand and maximum claim

The estimand is partial-gene reconstruction on previously unseen donor/core
graphs. Every cell independently masks an exact uniformly sampled number of
the 1,000 biological probes. Observed genes in the same cell and, only in the
graph arms, masked inputs from local neighboring cells are permitted.

The maximum possible claim is: "Spatial neighboring-cell information provides
predictive value for masked transcript reconstruction in Adjacent Normal
gastric tissue." No communication, mechanism, causal, cancer-specific, or
cross-tissue claim is permitted.

### Cohort, units, and splits

The immutable upstream selection manifest verifies ten pathology-confirmed
Adjacent Normal cores from ten distinct donors, five per slide. Opaque aliases
`ANC-01` through `ANC-10` are used; restricted donor/core identifiers are not
written. Donor and core are one-to-one here, so each alias is one independent
unit. The cohort contains 117,386 cells in 139 raw FOVs.

Five outer folds are fixed by position within the two five-core slide blocks.
Fold `j` tests the `j`th alias from each slide; validation uses the next first-
slide alias for even `j` and the next second-slide alias for odd `j`; the other
seven aliases train. Thus every fold is 7 train / 1 validation / 2 test, every
test fold contains both slides, and every donor/core is tested exactly once.
No cell, FOV, core, or donor crosses a split within a fold.

### Inputs, QC, and leakage controls

Only `expression_counts`, ordered gene names, `coordinates_um`, raw FOV labels,
and the vendor-QC flag are reused from the ten checksum-verified upstream
artifacts. Existing normalized targets, morphology, internal split labels,
graphs, and fixed masks are prohibited. All 1,000 targets are biological
probes; `Negative*` and `SystemControl*` controls are absent. Slash-combined
probe symbols remain indivisible.

The established retain-all cell policy is retained (112,815/117,386 cells pass
vendor QC); QC status is reported but is not a model input. Coordinates are
used only for graph construction. Cell/FOV/core/donor/slide/tissue labels,
annotations, morphology, RNA-derived QC, and target-derived library size are
not model inputs. SO_2 FOV 246 is outside every selected route and remains
explicitly excluded without an inferred core assignment.

For each fold, gene-wise `log1p(count)` mean and variance are fitted as an
equal-core mixture over the seven training cores only. No feature selection,
imputation, dimensionality reduction, or library-size normalization is used.
Validation and test targets do not contribute to fitted preprocessing.

### Geometry and adjacency

Geometry-only inspection preceded the contract: the median cell-to-12th-
neighbor distance is 24.55 um, the 95th percentile is 38.72 um, and 98.46% of
cells have 12 neighbors within 50 um. Rare sparse cells would otherwise connect
over 500 um. The fixed graph is therefore `k=12`, radius 50 um, symmetric
union, constructed separately within every raw FOV. It uses coordinates only,
has no cross-FOV/core edges, and is materialized once before training.

The model adjacency adds one explicit diagonal self-loop to the fixed graph.
The isolated arm contains exactly those diagonal self-loops. The conditional
null applies one fixed random cell-to-position assignment independently within
each FOV to both endpoints of every off-diagonal edge, then adds the same
self-loops. It preserves graph size and the degree distribution exactly and
never creates a cross-FOV edge. No edge attributes enter the model.

### Masking

For every cell and every training presentation, sample
`m_i ~ Uniform{0,...,1000}` and then select exactly `m_i` gene positions without
replacement. Masks are independent across cells. Dynamic training-mask seeds
derive from a separate fixed base plus fold, paired model seed, epoch, and core;
paired arms use identical schedules and core order.

Three deterministic evaluation replicates per core are materialized from a
model-seed-independent base seed. Their arrays and checksums are shared across
all arms, folds, and seeds. The model receives only the masked standardized
expression plus the full binary mask indicator. Masking is re-applied inside
the encoder, including to all neighbor cells. Huber loss is computed only on
masked entries; a zero-mask cell contributes context but no loss.

### Shared model and training budget

Every arm uses one identical one-hop GraphSAGE-style network:

- separate linear projections of 1,000 masked values and 1,000 mask flags to
  width 128;
- one mean-adjacency update with distinct self and aggregate projections;
- LayerNorm, GELU, dropout 0.1, a width-256 residual feed-forward block;
- a width-256 decoder to 1,000 standardized-log1p outputs.

No parameters depend on nodes, edges, folds, cores, or conditions. Paired runs
must have identical parameter counts and initial-state checksums. Training uses
AdamW (`lr=1e-3`, weight decay `1e-4`), gradient clipping 1, FP32, one complete
core graph per optimizer step, and 80 epochs. Every training core is visited
once per epoch in the same paired seed-specific order. There is no early
stopping. Every five epochs, the mean validation Huber over all three fixed
masks is evaluated; the lowest value selects `best.ckpt`. All arms still run
the same 560 optimizer updates per full run.

Five paired model seeds (`0` through `4`) are required. A one-epoch paired
smoke and a five-epoch paired seed-0 fold-0 resource pilot precede production.
The pilot must cover every expected core, remain finite, preserve paired masks
and initialization, use at most 20.5 GiB VRAM and 40 GiB host memory per process,
and project at most 45 minutes per full run. Failure blocks production until
the cause is diagnosed; scientific parameters are not retuned from pilot
performance.

### Metrics and aggregation

The scientific primary metric is held-out masked Huber on unclipped
`log1p(count)` predictions; masked MAE is co-primary. Secondary metrics are
RMSE, MSE, gene-wise Pearson/Spearman means, cell-wise Pearson/Spearman means,
and per-seed variation. The registry/checkpoint metric is standardized-scale
`val/masked_huber`; it is not the scientific test effect.

Each run reports every test core and fixed mask separately. Replicates are
averaged within core and seed. For inference, paired graph-minus-isolated
differences are averaged across seeds within each core, then across ten cores
with equal weight. A deterministic 10,000-resample core bootstrap supplies the
95% interval. Cells, masked entries, mask replicates, and seeds are not treated
as independent units.

Nonoverlapping target-mask strata are `(0,25%]`, `(25,50%]`, `(50,75%]`,
`(75,100%)`, and exactly `100%`; zero-mask cells are counted but have no error.
Neighbor availability is computed per masked target gene as the fraction of
off-diagonal true spatial neighbors where that same gene is observed, using
`[0,25%]`, `(25,50%]`, `(50,75%]`, and `(75,100%]`. The same strata are applied
to every arm.

For each core, define `d = graph Huber - isolated Huber`; negative favors the
graph. Also report `100*(isolated-graph)/isolated`, favoring-core count, all
five seed-level equal-core differences, and analogous MAE results.

### Frozen decisions

`GRAPH CONTEXT SUPPORTED` requires all of:

1. mean core-level relative Huber improvement at least 2%;
2. the 95% core-bootstrap interval for graph-minus-isolated Huber lies below 0;
3. at least 8/10 cores and 4/5 seed-level aggregates favor the graph;
4. mean MAE is lower; and
5. Huber differences favor the graph in both moderate target-mask strata.

`GRAPH CONTEXT NOT SUPPORTED` requires the upper 95% bootstrap bound for the
core-level relative improvement to be below 2%, with valid leakage, pairing,
and execution audits. Any other valid mixed result is `INCONCLUSIVE`; failed
pairing/preprocessing/leakage checks or inadequate independent units are also
inconclusive.

If the real graph has any favorable aggregate Huber point estimate, the full
25-run position-permuted null is required. Actual topology is supported only
by the analogous 2%/interval/8-core/4-seed criteria for real versus null;
otherwise it is not supported or inconclusive under the same narrow-interval
rule. A null result does not alter the primary adjacency conclusion.

### Failure, resources, artifacts, and commands

Stop on changed input/gene/split/graph/mask checksums, a cross-FOV edge,
non-identical paired initialization or mask schedules, parameter mismatch,
nonfinite values, incomplete core visits/updates, unsafe GPU collision, or
artifact/registry verification failure. Ordinary code, dependency, layout, or
resource errors must be diagnosed and repaired without silent result omission.

All active run output belongs under `scratch/active_runs/<run_id>/`; verified
runs publish under `artifacts/runs/YYYY/MM/<run_id>/`. Shared preparation is
under `data/processed/adjacent_normal_grouped_adjacency_ablation_v1/`; reports
belong under `reports/analyses/adjacent_normal_grouped_adjacency_ablation/`.
Every run records config, command, Git/data/split/environment provenance,
append-only metrics, best checkpoint and checksum, resources, failures, and a
completion marker in the authoritative SQLite registry.

Planned verification and entry points (filled with exact materialized paths by
the implementation) are:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_adjacency_ablation.py
PYTHONPATH=src /venv/main/bin/python scripts/train/prepare_adjacency_ablation.py
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/audit_adjacency_preparation_semantics.py
PYTHONPATH=src /venv/main/bin/python scripts/train/enqueue_adjacency_ablation.py smoke
PYTHONPATH=src /venv/main/bin/python scripts/train/enqueue_adjacency_ablation.py pilot
PYTHONPATH=src /venv/main/bin/python scripts/train/enqueue_adjacency_ablation.py primary
PYTHONPATH=src /venv/main/bin/python scripts/analysis/analyze_adjacency_ablation.py
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 doctor
```
