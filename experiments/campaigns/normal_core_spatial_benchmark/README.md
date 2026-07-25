# Normal-Core Spatial Masked-Expression Benchmark

## Status

- Phase: complete
- Outcome: not supported by the prespecified gate
- Scope: exploratory within-core prediction

This workflow tests whether local spatial context improves masked-expression
prediction beyond cell-autonomous baselines in the one legacy true-Normal CosMx
core. The selector is resolved locally from the clinical workbook; the
restricted donor mapping and exact core label are not written to reports.

The selected core has 24,245 matched expression/metadata cells across 24 FOVs,
not approximately 150,000 cells. Pooling adjacent-normal cores would change the
experimental unit and is outside this workflow.

The sealed five-seed result showed a small whole-node advantage for true-graph
G1 over B0: Huber loss 0.224367 versus 0.224994, a paired gain of 0.000627
(0.279%; spatial-block bootstrap 95% interval 0.000433 to 0.000823). The
interval excluded zero, true G1 slightly beat rewired G1, and the block-mask
point estimate favored G1, but the primary gain was far below the locked 2%
threshold. See the [archived final report](../../../artifacts/legacy_runs/lr_spatial_benchmark_batch/original/report/final_report.md) and
[machine-readable summary](../../../artifacts/legacy_runs/lr_spatial_benchmark_batch/original/final_analysis_v1/summary.json).

## Task Contract

### Objective and deliverables

1. Build leakage-resistant physical cell graphs and deterministic masks.
2. Train the nested B0, B1, G1, G2, and (only after graph utility) G3 ladder
   with the same expression, metadata, splits, masks, and tuning budget.
3. Compare graph topology, radius, masking curriculum, node hidden dimension,
   depth, and edge-embedding dimension without opening the final test split.
4. Produce graph-QC figures, fixed split/mask artifacts, run manifests,
   per-block predictive metrics, uncertainty estimates, and a concise report.
5. Lock a recommended graph/model standard for later multi-core experiments.

### Scientific question and estimand

Primary question:

> Conditional on independently measured, always-visible cell metadata, do true
> local spatial neighbors reduce whole-node masked Huber loss relative to a
> cell-autonomous model in held-out spatial regions of this core?

For held-out spatial block \(b\), the primary contrast is

```text
delta_b = loss_b(B0 + metadata) - loss_b(G1 true graph + metadata)
```

after averaging predictions across five locked model seeds. Positive values
favor the spatial model. Whole-node masking is primary. Partial-gene and
contiguous-block masking are separate secondary estimands and are not pooled
into one claim.

The maximum defensible claim is within-core spatial predictive dependency.
Cells and seeds are not biological replicates. This design cannot establish
patient-level generalization, communication, mechanism, or causality.

### Hypotheses, alternatives, and discriminating predictions

- H1: true local spatial context adds predictive information. G1 on the true
  graph should improve whole-node and block-mask loss over B0 and over the same
  G1 trained on a mechanism-breaking rewired graph.
- A1: cell morphology/imaging and intracellular co-expression explain the
  result. Then B0 should match graph models, especially for partial-gene masks.
- A2: uniform spatial smoothing explains the result. Then B1 should match G1.
- A3: broad location or FOV batch, not local neighborhoods, explains the gain.
  A broad spatial-field baseline or rewired/matched graph should match G1.
- A4: segmentation spillover or nearest-neighbor copying explains the gain.
  Gain should weaken under minimum-distance/edge perturbations and block masks.
- A5: edge geometry adds information. G2 should beat G1 and degrade when edge
  attributes are zeroed or permuted within distance bins.

### Inputs and feature policy

Immutable inputs are the nested raw CosMx expression and metadata CSVs,
`data/clinical/fov_core_map.csv`, and the two local pathology workbooks described
in `docs/legacy_data_guide.md`. Selection uses the unique legacy tissue label `정상`; the
new pathology-review row is present and has no correction note. The legacy
clinical policy remains provisional pending project-wide reconciliation.

All 1,000 biological panel targets are modeled. `Negative*` and
`SystemControl*` probes are excluded from targets. Slash-combined probe symbols
remain indivisible probe-level targets.

The following independently measured morphology/imaging values are available,
unmasked, to every model and every expression-mask mode:

```text
Area, Area.um2, AspectRatio, Width, Height,
Mean/Max PanCK, G, Membrane, CD45, DAPI,
SplitRatioToLocal, NucArea, NucAspectRatio,
Circularity, Eccentricity, Perimeter, Solidity
```

Transforms, imputation, and scaling are fitted on training regions only.
Missingness indicators are added when needed.

Prohibited primary inputs include cell/FOV/slide/core identifiers; absolute or
local coordinates as node covariates; RNA count, library-size, complexity,
negative/false-code and RNA-derived QC summaries; vendor cell type, posterior,
cluster, expression-space neighborhood, and niche annotations. Coordinates are
used only for splitting, graph construction, edge geometry, and an explicitly
labeled broad-field control. This prevents hidden expression from re-entering
through metadata.

### Splits, preprocessing, and leakage controls

- Joins use `(slide, fov, cell_ID)`.
- Spatial macroblocks are assigned before preprocessing.
- Train, validation, and sealed test regions are disjoint, with no cross-split
  graph edges. Held-out graphs are constructed independently.
- Gene log1p means/standard deviations, metadata transforms, and edge scaling
  are fitted using training regions only.
- The final test split is opened once after graph, masking, and dimension
  standards are locked.
- Fixed validation/test masks and multiple mask replicates are saved and reused
  across architectures and seeds. Training mask schedules are paired.
- Random-cell splits are permitted only in synthetic/smoke tests.

### Masking

Expression is transformed gene-wise as train-fitted standardized `log1p`.
Masked expression values are zero and a full explicit gene-mask channel is
supplied. Metadata is never masked. Loss is computed only on masked expression.

Candidate curricula use a ten-epoch partial-mask warm-up, followed by
deterministic epoch-level mask draws over the complete split:

```text
P-only       = 100% partial-gene epochs
P+N          = 70% partial, 30% whole-node
P+N+B        = 60% partial, 30% whole-node, 10% spatial-block
```

Partial masks hide 20% of genes. Whole-node masks hide all genes in 10% of
eligible cells. Block masks hide all genes in a contiguous physical region.
The conclusion-bearing fixed bundles use one prespecified primary specification
per mode, with three validation and five test replicates.

### Models, graphs, and controls

Shared models:

- B0: self-only masked-expression MLP with metadata.
- Broad-Field: B0-like cell-autonomous control augmented with the locked global
  quadratic basis `[x, y, x², xy, y²]`. Coordinates are centered and scaled
  using training nodes only; the fitted micrometer center/scale and exclusions
  of IDs, Fourier features, knots, graphs, and neighbor features are recorded
  in every run manifest.
- B1: uniform mean-neighbor message passing.
- G1: topology-only one-hop GATv2.
- G2: edge-conditioned GATv2.
- G3: additive self plus signed edge-message decoder, staged from B0.

Required controls include a constant/global-mean predictor, B0, B1, a
parameter-matched self-only model, true-versus-rewired G1, a broad spatial-field
baseline, and nearest-neighbor copying. G2 controls are zero edge attributes,
distance only, and within-distance-bin permutation. G4 is out of scope until
the first-wave graph passes.

Candidate symmetric graphs use global physical coordinates, two directed edges
per undirected relation, no self-loops, and no cross-split edges:

```text
k in {8, 12, 16}
radius_um in {30, 50, 75}
symmetry in {union, mutual}
```

A geometry-only QC pass precedes training. It records degree, edge-length,
components, isolated nodes, cap-hit rates, FOV seams, and split-boundary
handling. Exploratory screening may use one seed; the top three validation
candidates require three paired seeds. Rewired controls preserve degree and
approximately preserve distance. After graph lock, node hidden dimensions
`{128, 256, 512}`, graph depths `{1, 2}`, and G2 edge dimensions
`{16, 32, 64}` are tested sequentially rather than as a full Cartesian grid.

Selection prioritizes validation whole-node loss, then block-mask loss,
stability, and graph QC. Biological-looking patterns are not a selection
criterion. A small graph-by-mask interaction check precedes lock.

The locked standards are k=16, radius 75 µm, mutual symmetry, no minimum
distance, P+N+B masking, 512 hidden units, two G1 layers, and 64-dimensional G2
edge embeddings. G3 remained disabled because its validation-only utility gate
was not met.

### Acceptance and falsification criteria

H1 is supported within this core only if the locked true-graph G1:

1. improves the primary whole-node loss over B0 by at least 2% relative;
2. has a spatial-block-resampled 95% interval for `B0 - G1` above zero;
3. also beats its degree/distance-matched rewired-graph control; and
4. improves contiguous block-mask loss in the same direction.

Metrics are aggregated within held-out spatial blocks before uncertainty
estimation. Mask replicates and model seeds are technical variation, not
independent samples. If spatial blocks are too few for calibrated inference,
the interval is labeled descriptive. Failure of any criterion is a valid
negative or inconclusive result and is reported without weakening the gate.

G1 must beat B1 to claim attention utility. G2 must beat G1 or improve stability
and pass edge-attribute controls. G3 may proceed only after graph utility, must
be within 2% of G2 primary loss, and must pass matched high-contribution edge
deletion. Gene-wise gains are exploratory, use block resampling and BH-FDR, and
do not proceed to attribution unless reproducibly positive.

### Compute and artifacts

The host has eight 24-GB RTX 3090 GPUs. Independent candidates/seeds run one per
GPU. Each epoch uses the exact split graph and all split nodes; there is no
neighbor sampling or stochastic target-node minibatching. Mixed precision was
enabled only after an FP32 equivalence audit passed.

Every conclusion-bearing run records configuration, command, Git state, input
checksums, split/mask IDs, seeds, dependency/CUDA/GPU versions, runtime, peak
VRAM, failures, checkpoints, metrics, and artifact paths. The workspace is not
persistent, so irreplaceable final outputs require an off-instance copy.

The original checksum-bound artifacts are preserved, unchanged, under
`artifacts/legacy_runs/lr_spatial_benchmark_batch/original/`:

```text
artifacts/legacy_runs/lr_spatial_benchmark_batch/original/prepared_full_v1/
artifacts/legacy_runs/lr_spatial_benchmark_batch/original/graph_grid_trainval_v2/
artifacts/legacy_runs/lr_spatial_benchmark_batch/original/selection_{mask,graph,hidden,depth,edge_embedding}_v1/
artifacts/legacy_runs/lr_spatial_benchmark_batch/original/standards_lock_v1/
artifacts/legacy_runs/lr_spatial_benchmark_batch/original/final_locked_v1/runs/<locked-job-id>/
artifacts/legacy_runs/lr_spatial_benchmark_batch/original/final_analysis_v1/
artifacts/legacy_runs/lr_spatial_benchmark_batch/original/report/final_report.md
```

## Run Order

Completed sequence:

1. synthetic tests, CPU/GPU smoke profiles, determinism, and AMP equivalence;
2. immutable data/split/mask preparation and train+validation-only graph QC;
3. validation-only masking and graph screens;
4. sequential hidden-width, depth, and G2 edge-width screens;
5. validation-only graph-by-mask interaction and top-candidate confirmation;
6. immutable standards lock;
7. exact B0/B1/G1/G2 five-seed sealed ladder with controls;
8. full artifact/provenance audit, spatial-block bootstrap analysis, figure
   inspection, and report.

The final matrix deliberately omitted G3 because the conditional validation
gate was false.

## Completion Gate

Completion criteria were met: 119 automated tests passed; all 50 authorized
locked runs completed; every declared prediction/checkpoint/metric hash and
2,700 stored array headers were independently audited; canonical prepared
targets, rows, blocks, and masks were reverified during analysis; and all final
figures were inspected.

Reproduce the conclusion-bearing aggregation with:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/analyze_spatial_benchmark.py \
  --runs-dir artifacts/legacy_runs/lr_spatial_benchmark_batch/original/final_locked_v1/runs \
  --standards-lock artifacts/legacy_runs/lr_spatial_benchmark_batch/original/standards_lock_v1 \
  --output-dir reports/analyses/normal_core_spatial_benchmark_reproduction \
  --locked-graph-id k16_r75_mutual_rbf8 \
  --expected-seeds 5 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 2026
```
