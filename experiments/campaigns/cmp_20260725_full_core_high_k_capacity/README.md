# Full-Core High-k Representation-Capacity Pilot

## Status

- Phase: completed
- Outcome: positive but sub-threshold; locked capacity gate failed
- Scope: exploratory, held-in reconstruction within one legacy true-Normal core
- Campaign: `cmp_20260725_full_core_high_k_capacity`

This campaign is a deliberately transductive capacity test. It fits models on
all 24,245 cells from the one protected legacy true-Normal CosMx core and
reports reconstruction on fresh fixed masks over those same cells. There is no
validation or test set. Consequently, the result cannot estimate
generalization and must not be described as test accuracy.

## Results

Both conclusion-bearing runs completed 200 finite epochs at seed 0 with
3,987,880 trainable parameters and the same preprocessing, graph, and fixed
evaluation masks:

| Model | Held-in whole-node Huber | Training time | Peak allocated VRAM |
|---|---:|---:|---:|
| Exact k=1000 G2 | 0.228873 | 1,509.9 s | 7.975 GiB |
| G2-parameter-matched cell-autonomous control | 0.229975 | 63.4 s | 1.973 GiB |

The self-minus-G2 difference was 0.001102, a 0.479% relative G2 gain.
All three paired technical mask replicates favored G2, both histories were
finite, and neither run met the post-specified divergence definition. The
prespecified primary criterion nevertheless failed because 0.479% is below
the required 2%. This is not evidence that
small k caused the earlier low predictive accuracy, and the no-holdout design
cannot verify the original predictive hypothesis.

Secondary metrics were mixed rather than consistently G2-favorable:

| Held-in metric | G2 | Matched self |
|---|---:|---:|
| Partial-gene Huber | 0.225619 | 0.225471 |
| Whole-node MAE | 0.360737 | 0.356999 |
| Whole-node MSE | 1.003334 | 1.006448 |
| Whole-node gene Pearson | 0.180659 | 0.182428 |
| Whole-node cell Pearson | 0.091392 | 0.079892 |
| Spatial-block Huber | 0.226019 | 0.226272 |

The exploratory routing analysis found broad attention: mean effective
neighbor count 713.2 of mean incoming degree 858.5, with mean
attention-weighted distance 147.8 µm. Top-routed senders were descriptively
enriched for the TLS organizer program, but neither repeated top-edge deletion
nor bounded organizer-channel sender ablation worsened the receiver-program
Huber loss relative to eight distance-matched nulls. The joint TLS dependency
criterion failed.

Artifacts:

- G2 bundle:
  `artifacts/runs/2026/07/r_20260725T083929Z_8364b333_s000_f00_a01_f5c47601`
- Matched control bundle:
  `artifacts/runs/2026/07/r_20260725T090752Z_e05d042b_s000_f00_a01_4ff98df7`
- Locked comparison:
  `reports/analyses/full_core_high_k_capacity/comparison`
- Toy interpretability analysis:
  `reports/analyses/full_core_high_k_capacity/interpretability`

## Task Contract

### Objective and deliverables

1. Implement a reproducible full-core training path that does not silently
   relabel held-in cells as validation or test data.
2. Fit one exact high-neighbor G2 model and one parameter-matched
   cell-autonomous model with paired seeds, masks, preprocessing, and epoch
   budget.
3. Report held-in masked-expression regression metrics, convergence, graph
   geometry, runtime, and peak memory.
4. After both runs complete, run a small exploratory routing/faithfulness
   analysis and a prespecified TLS-program toy analysis without allowing it to
   override the capacity-gate outcome.
5. Preserve configuration, code/data provenance, checkpoints, metrics, graph
   checksums, logs, and a concise comparison report.

### Scientific question, hypotheses, and alternatives

Capacity question:

> When every cell in this core is available for fitting, can a literal
> high-neighbor edge-conditioned GAT representation reconstruct freshly masked
> whole-cell expression better than a parameter-matched cell-autonomous
> representation?

- H1-capacity: the high-k graph representation lowers held-in whole-node masked
  Huber loss by at least 2% relative to the matched cell-autonomous model.
- A1-self: intracellular expression plus morphology/imaging already contains
  the usable information, so the matched self-only model performs similarly.
- A2-smoothing: any G2 gain is broad regional smoothing or cell-state
  co-localization rather than direct interaction. This two-run pilot cannot
  rule that out because it omits broad-field, mean-neighbor, and rewired
  controls.
- A3-dense-noise: the dense graph dilutes local signal or oversmooths node
  states, so G2 matches or underperforms the self-only model.
- A4-capacity: extra G2 parameters, not topology, explain a gain. The primary
  comparator is therefore the parameter-matched self-only model rather than
  the smaller B0.

The earlier predictive hypothesis was a held-out spatial prediction claim.
This no-holdout design cannot verify that hypothesis. It can only show whether
the chosen model has held-in reconstruction capacity. A future
split-preserving high-k experiment with B0/B1/rewired controls would be needed
to revisit predictive graph utility.

### Estimand and maximum permitted claim

The primary estimand is the paired difference in held-in whole-node masked
Huber loss:

```text
delta_fit = fit_loss(parameter-matched self) - fit_loss(high-k G2)
```

Predictions use fresh deterministic masks not used for gradient updates, but
the cells, graph, preprocessing statistics, and tissue sample were all used
for fitting. The maximum claim is that the trained representation has
held-in masked-expression reconstruction capacity in this one core. It is not
evidence of patient generalization, independent prediction, communication,
mechanism, or causality.

### Data, preprocessing, and leakage policy

- Input artifact:
  `artifacts/legacy_runs/lr_spatial_benchmark_batch/original/prepared_full_v1`.
- All 24,245 cells and all 1,000 biological probes are fit data.
- Technical controls remain excluded.
- The 22 permitted morphology/imaging covariates remain always visible.
- Identifiers, coordinates, RNA-derived QC, vendor cell types, clusters,
  expression-derived neighborhoods, and niches remain prohibited model
  covariates.
- Gene and metadata transforms are refit on the complete core and labeled
  `full_core_fit`; no train-only preprocessing claim is made.
- The verified materialized full-core preprocessing checksum is
  `a112fbb2bdf929197759c82913fc379c4df797f75676457bcd10419fe7a969d5`.
  The all-fit role-assignment checksum is
  `2c8c59fb659401cc202126cb14154f064d2e3380819aff647f06a30db354d84c`.
- Coordinates are used only for graph construction, macroblock diagnostics,
  and distance-aware interpretation.
- Held-in evaluation masks use a seed namespace disjoint from epoch masks.

The principal leakage risk is not hidden leakage but resubstitution: all
evaluation cells influenced model fitting. That limitation is intrinsic to
the requested estimand and will remain visible in every report.

### Graph decision

The original graph jointly used a neighbor cap and a 75 µm radius. On the full
core, the number of neighbors within 75 µm has mean 131.81, median 131, 95th
percentile 218.8, and maximum 306. Therefore `k=1000` and `k=5000` are
identical when the 75 µm radius is retained.

This pilot instead uses a literal 1,000-nearest-neighbor candidate graph:

```text
k = 1000
symmetry = mutual
self loops = false
full-core graph = true
radius guard = 650 µm
```

The 650 µm guard exceeds the observed maximum 1,000th-neighbor distance
(609.72 µm), so it does not truncate the directed kNN candidates. The mutual
graph has 21,029,944 directed edges, mean degree 867.39, median degree 917,
and mean edge length 137.66 µm. It is a regional-context graph, not a
direct-contact graph. Production construction with eight CPU workers took
47.2 seconds, peaked at approximately 2.36 GiB resident host memory, produced
one connected component with no loops, duplicates, or isolated nodes, and
yielded graph checksum
`2469064e2fe14b48f642fca09a546d9e420d9fd851b9668996da62ee8246d060`.

The `k=5000` candidate is omitted. Its mutual graph would have 101,237,016
directed edges, mean degree 4,175.58, and mean edge length 319.10 µm. It would
connect about one fifth of the core to each cell and would test a different,
largely global estimand at roughly five times the edge cost.

### Models and training

- Graph model: G2 edge-conditioned GATv2, 512 hidden units, two graph layers,
  four attention heads, 64-dimensional shared edge embeddings, and the locked
  nonlinear decoder.
- Comparator: a strictly cell-autonomous model with the same encoder,
  hidden/decoder widths, residual depth, seed, and optimization budget. Its G2
  edge encoder and edge-projection parameter budget is repurposed on
  within-cell embeddings; it never receives topology, neighboring cells,
  coordinates, or measured edge attributes. Both models have exactly
  3,987,880 trainable parameters.
- Seed: one paired model seed (`0`), as requested for this large pilot.
- Objective: the locked `P+N+B` curriculum with paired epoch masks.
- Epoch budget: 200 fixed epochs with no validation-based early stopping or
  checkpoint selection.
- Primary checkpoint: the final epoch. Training diagnostics may identify an
  earlier lower held-in loss but cannot replace the final checkpoint.
- Precision: mixed precision only after an equivalence smoke check.
- Graph execution: exact full-neighbor attention. Neighbor sampling is
  prohibited because it would make the effective k smaller than 1,000.

The existing one-GPU full-graph implementation cannot materialize this graph's
attention activations on a 24 GB GPU. The implementation must use exact
receiver partitioning, activation checkpointing, or an equivalently verified
method. Any reduction in k, hidden width, depth, or edge features creates a new
variant and is not an automatic retry.

### Metrics and decision criteria

The task is regression; ambiguous classification `accuracy` is not reported.

Primary metric:

- `fit/whole_node/masked_huber`, lower is better.

Secondary metrics:

- held-in partial-gene and spatial-block masked Huber;
- masked MSE and MAE;
- gene-wise and cell-wise Pearson and Spearman summaries;
- final and best observed epoch loss, last-20-epoch slope, and gradient norms;
- attention effective-neighbor count and attended-distance summaries for G2;
- runtime, throughput, peak VRAM, host memory, and checkpoint size.

Three paired fixed-mask replicates per mask mode are technical repeats. They
are not independent biological replicates.

H1-capacity passes only if:

1. the paired relative G2 gain in mean whole-node held-in Huber is at least 2%;
2. all three mask-replicate differences favor G2;
3. both runs complete 200 finite epochs; and
4. neither run has unresolved divergence. The qualitative requirement was
   declared before training, but the following numerical thresholds were
   finalized after G2 started and are therefore a post-specified audit: any
   non-finite loss or gradient in the last 20 epochs; a last-20 gradient
   maximum greater than 10 times the positive median gradient over all
   epochs; or, for any mask mode with at least three observations, its last
   loss greater than 1.25 times the median of its previous up-to-five
   same-mode losses.

A non-positive gain falsifies H1-capacity for this configuration. A positive
gain below 2% is sub-threshold. Passing this gate still does not verify the
held-out predictive hypothesis.

### Toy interpretation gate

Interpretation proceeds only after both runs complete. A failed predictive
gate remains visible; the following analyses are exploratory diagnostics, not
independent biological validation:

1. Compute last-layer incoming-attention entropy, effective-neighbor count,
   and attention-weighted distance under a fixed whole-node mask. This tests
   whether the model actually uses a broad portion of the 1,000-neighbor
   candidate set.
2. Prespecify a panel-supported TLS organizer program
   `{CCL19, CCL21, CXCL13}` and lymphoid receiver program
   `{CCR7, CXCR5, MS4A1, CD3D, CD3E, CD79A, CD74, HLA-DRA}`.
3. Compare receiver-program reconstruction and bounded sender-program
   perturbations with distance-matched null edges.
4. Delete top-routed edges for a deterministic receiver subset and compare
   the loss change with an equal number of distance-matched random deletions.

Attention is computational routing, not biological importance. A candidate is
called faithful only if deletion changes the relevant prediction more than
the matched deletion null. Even then, the maximum label is a
model-implied TLS-related predictive dependency in this one transductive core.

### Compute, stop criteria, and artifacts

- Hardware: eight available RTX 3090 GPUs (24 GB each), 377 GiB host RAM.
- A representative 1-3 epoch pilot must record throughput and peak memory
  before the 200-epoch run.
- Stop on non-finite loss/gradients, a verified implementation mismatch,
  inability to execute exact k=1000, repeated out-of-memory after bounded
  receiver-chunk tuning, or insufficient disk for required artifacts.
- Do not silently substitute `k=1000, radius=75 µm`; that graph has at most 306
  neighbors and answers a different capless-local question.
- Active output belongs under `scratch/active_runs/`; verified bundles belong
  under `artifacts/runs/YYYY/MM/`; cross-run analysis belongs under
  `reports/analyses/full_core_high_k_capacity/`.
- Required provenance includes exact command, resolved config, Git/dirty
  fingerprint, prepared-artifact checksum, full-core preprocessing checksum,
  graph checksum/QC, seeds, environment, GPU allocation, logs, final
  checkpoint, metrics, failures, and interpretation artifacts.

`/workspace` is not backed by a persistent volume. Local completion does not
constitute an off-instance backup.

## Completed Run Order

1. Unit-test exact chunked/partitioned attention against ordinary GATv2 on a
   small graph.
2. Verify full-core preprocessing and exact mutual-kNN graph invariants.
3. Run the registered two-epoch G2 resource diagnostic
   (`configs/experiment/full_core_high_k_g2_resource_pilot.yaml`). The resource
   diagnostic was not conclusion-bearing and did not enter the comparison.
4. Run the 200-epoch G2 and parameter-matched self-only variants with seed 0.
5. Evaluate paired fresh held-in masks and write the comparison report.
6. Run the gated toy interpretation and faithfulness diagnostics.

## Verification Commands

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/compare_full_core_capacity.py \
  --g2-run artifacts/runs/2026/07/r_20260725T083929Z_8364b333_s000_f00_a01_f5c47601 \
  --b0-g2-matched-run artifacts/runs/2026/07/r_20260725T090752Z_e05d042b_s000_f00_a01_4ff98df7 \
  --output reports/analyses/full_core_high_k_capacity/comparison

PYTHONPATH=src /venv/main/bin/python scripts/analysis/analyze_full_core_gat_interpretability.py \
  --run artifacts/runs/2026/07/r_20260725T083929Z_8364b333_s000_f00_a01_f5c47601 \
  --output reports/analyses/full_core_high_k_capacity/interpretability \
  --device cuda:0
```

Both run bundles passed the success-contract checksum verifier. Focused tests,
the complete applicable test suite, the infrastructure doctor, and generated
analysis inspection are recorded at completion.
