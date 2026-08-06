# Full-Core Large-Capacity QKV-GAT k Stress Test

## Status

- Phase: complete
- Outcome: locked representation and large-k gates failed
- Campaign: `cmp_20260726_full_core_qkv_large_k`
- Scope: exploratory held-in reconstruction in one legacy true-Normal core

This campaign deliberately fits all 24,245 cells and evaluates fresh masks on
those same cells. It has no validation or test set. The percentage result is
masked variance explained (`100 * R²`), not classification accuracy or an
estimate of generalization.

## Task contract

### Objective and deliverables

1. Implement and verify an edge-aware sparse QKV graph Transformer with exact
   receiver-wise attention over every retained neighbor.
2. Select the largest common model that safely executes both exact k=1,000 and
   k=5,000 graphs on a 24 GiB RTX 3090.
3. Run one fixed seed for k=1,000, k=5,000, and an exactly parameter-count-
   matched cell-only control, each for 300 epochs without validation selection.
4. Report held-in Huber, MSE, MAE, masked R² and percent variance explained,
   paired technical-mask differences, convergence, runtime, and resources.
5. Preserve verified immutable run bundles and a locked comparison report.

### Scientific question and hypotheses

Question:

> With all cells available for fitting, does a much larger genuine QKV graph
> Transformer encode held-in masked-expression information, and does increasing
> the exact neighborhood from k=1,000 to k=5,000 improve that capacity?

- H1-representation: at least one graph model has positive whole-node masked
  variance explained and at least 2% lower whole-node Huber than the matched
  cell-only model, with all three fixed technical masks favoring the graph.
- H1-k: k=5,000 has at least 2% lower whole-node Huber and higher masked R²
  than k=1,000, with all three fixed masks favoring k=5,000.
- A1-self: intracellular expression and permitted morphology/imaging explain
  the reconstructable signal; the matched cell-only model performs similarly.
- A2-global smoothing: k=5,000 helps by broad field/state smoothing, not direct
  cellular interaction. This is especially credible because each receiver is
  connected to about 20.6% of the core.
- A3-dense noise/oversmoothing: k=5,000 dilutes local signal and underperforms
  k=1,000.
- A4-optimization: the larger graph is harder to optimize in 300 epochs. Loss
  curves can diagnose under-training but cannot retroactively change the
  fixed budget.

### Estimand and maximum claim

The primary estimand is held-in whole-node masked Huber on three deterministic
fresh mask replicates. Secondary percentage accuracy is:

```text
masked_percent_variance_explained = 100 * (
    1 - masked_squared_error / masked_target_total_sum_of_squares
)
```

R² is computed separately per replicate and then averaged. It is undefined at
zero target variance and may be negative; values are never clipped. Relative
Huber gain is `100 * (reference - candidate) / reference`.

All cells, graph topology, and preprocessing statistics are fitted. The maximum
claim is one-core transductive representation capacity. These runs cannot
verify patient-held-out prediction, biological replication, direct cell-cell
interaction, mechanism, or causality.

### Units, data, inputs, and leakage

- Observational and experimental unit: one spatial core.
- Technical repeat: three deterministic masks per mode; these are not
  biological replicates.
- Input:
  `artifacts/legacy_runs/lr_spatial_benchmark_batch/original/prepared_full_v1`.
- Materialized preprocessing checksum:
  `a112fbb2bdf929197759c82913fc379c4df797f75676457bcd10419fe7a969d5`.
- All-fit role checksum:
  `2c8c59fb659401cc202126cb14154f064d2e3380819aff647f06a30db354d84c`.
- Targets: 1,000 biological probes on the full-core fitted standardized log1p
  scale; technical controls are excluded.
- Always-visible inputs: 22 permitted morphology/imaging covariates.
- Edge inputs: 17 standardized geometric/distance features.
- Prohibited node inputs: identifiers, coordinates, RNA-derived QC/library
  size, and vendor cell type/cluster/neighborhood/niche annotations.
- Primary limitation: resubstitution. Fresh masks do not make fitted cells a
  validation or test set.

### Exact graph lock

Both variants use exact nearest neighbors, mutual symmetry, no self loops,
zero edge dropout, and a common 1,200 µm post-kNN radius guard. The guard
exceeds the observed maximum 5,000th-neighbor distance (1,115.08 µm), so it
does not filter either graph. Using the previous 650 µm guard for k=5,000
would be invalid.

| k | Directed edges | Mean degree | Mean edge distance | Graph checksum |
|---:|---:|---:|---:|---|
| 1,000 | 21,029,944 | 867.39 | 137.66 µm | `23d9b09af45fca21e4f30ef04a6921765b19399eff3e3ef78e371f6b019004e6` |
| 5,000 | 101,237,016 | 4,175.58 | 319.10 µm | `50f293972c443011a80abfbd81bf7cc7f44a35ccb39794e900b67dc4d2ca8d85` |

k=5,000 is a regional/global-context graph, not a direct-contact graph.

### Architecture and resource selection

The graph backbone uses receiver queries, sender keys and values, scaled
dot-product attention, per-head edge logit bias, bounded per-head value gates,
pre-LayerNorm residual blocks, four-times-width GELU feed-forward layers, and
an expression decoder. Attention is normalized over every incoming edge.
Receiver chunks and activation checkpointing are exact execution, not neighbor
sampling. FP16/BF16 logits, weights, messages, and aggregation are accumulated
in FP32 before the learned output projection.

The following diagnostic candidates were added sequentially, with each smaller
fallback locked before its outcome was observed:

| Candidate | Width | Layers | Heads | FFN/decoder | Parameters |
|---|---:|---:|---:|---:|---:|
| extra-large target | 2,048 | 8 | 32 × 64 | 8,192 | 432,030,056 |
| large fallback | 1,536 | 8 | 24 × 64 | 6,144 | 245,389,160 |
| resource fallback | 1,024 | 8 | 16 × 64 | 4,096 | 111,177,064 |
| resource fallback | 896 | 8 | 14 × 64 | 3,584 | 85,816,040 |
| boundary fallback | 832 | 8 | 13 × 64 | 3,328 | 74,364,328 |
| runtime-bound fallback | 768 | 8 | 12 × 64 | 3,072 | 63,731,816 |
| runtime-bound fallback | 704 | 8 | 11 × 64 | 2,816 | 53,918,504 |
| runtime-bound fallback | 640 | 8 | 10 × 64 | 2,560 | 44,924,392 |
| runtime-bound fallback | 576 | 8 | 9 × 64 | 2,304 | 36,749,480 |

The extra-large model is selected only if two complete k=5,000 optimizer steps
finish with finite losses/gradients, peak allocated VRAM no greater than
20.5 GiB, no implementation/checksum failure, and projected 300-epoch training
time no greater than 24 hours per graph run. Otherwise the same rule is applied
to the large fallback, width 1,024, width 896, width 832, width 768, width
704, width 640, and then width 576. Each
contingency was added only after the next-larger diagnostic exceeded the
runtime gate and before that fallback's outcome was observed. A failed
diagnostic is retained and never enters the scientific comparison. No width is
changed between conclusion-bearing runs.

After width reductions failed to materially clear the runtime gate while
leaving several GiB of memory headroom, the execution bottleneck was localized
to the 200,000-edge cap: a typical 64-receiver k=5,000 shard has about 267,000
edges and was therefore split. A width-896 execution-tuning diagnostic with a
300,000-edge cap was locked before its outcome. This cap changes only the exact
receiver partition schedule; it does not change parameters, edges, attention
normalization, masks, or numerical accumulation. It may be selected only under
the unchanged memory and runtime criteria above.

The 20.5 GiB ceiling reserves about 15% of the 24 GiB card for allocator and
step-to-step variation. Width 2,048 is the predeclared ceiling because its
parameters, gradients, and Adam states consume about 6.44 GiB before the
k=5,000 graph (about 7.92 GiB) and activations; larger widths do not have a
credible safety margin on this hardware.

The selected common architecture is width 576, eight layers, nine
64-dimensional heads, a 2,304-wide FFN and decoder, and 36,749,480 trainable
parameters. Its 320,000-edge cap is nonbinding for a 64-receiver shard at
k=5,000, so it executes the pure receiver-shard schedule while retaining and
normalizing over every edge. The matched self-only model has the same 130
state tensors, tensor shapes, and parameter count.

### Training and comparison lock

- Conclusion-bearing runs: QKV k=1,000, QKV k=5,000, and QKV parameter-matched
  self-only; seed 0 only.
- Same selected dimensions, initialization seed, epoch masks, optimizer,
  learning rate, weight decay, gradient clipping, decoder, and epoch budget.
- Fixed 300 epochs; final epoch retained; no early stopping, validation, test,
  or best-checkpoint selection.
- AdamW, learning rate `1e-4`, weight decay `1e-5`, gradient clip `1.0`.
- AMP after numerical equivalence tests; dropout and attention dropout zero.
- Full exact graph; no neighbor sampling or edge dropout.

The matched self-only control consumes no topology, neighboring expression,
coordinates, or edge attributes. Its QKV/edge parameter budgets are repurposed
as within-cell transformations; parameter count and state shapes must match
the graph model exactly and every trainable parameter must receive a finite
gradient in a synthetic audit.

Primary decisions:

1. H1-representation passes only if the graph-vs-self Huber gain is at least
   2%, all three whole-node mask replicates favor the graph, whole-node masked
   R² is positive, and all runs finish 300 finite epochs.
2. H1-k passes only if the k=5,000-vs-k=1,000 Huber gain is at least 2%, all
   three whole-node masks favor k=5,000, k=5,000 has higher masked R², and both
   graph runs finish 300 finite epochs.
3. Positive but sub-2% differences are explicitly sub-threshold. A nonpositive
   difference falsifies that hypothesis for this architecture and budget.

No rewired, mean-neighbor, broad-field, patient-held-out, or independent-cohort
control is included because the user requested only a few large runs. Passing
the capacity gate would therefore motivate—not replace—those tests.

### Failure and stop criteria

Stop a variant on a graph identity mismatch, non-finite loss or gradient,
unresolved exact-attention mismatch, repeated OOM after lowering only the
execution chunk bound, peak VRAM above the pilot ceiling, projected runtime
above the declared bound, or insufficient disk for complete provenance and
checkpoints. Do not silently reduce k, width, depth, edge features, epoch
budget, or evaluation masks.

### Compute and artifacts

- Hardware: RTX 3090 24 GiB; one GPU per run.
- Safely independent conclusion runs may execute on separate idle GPUs after
  the common model is locked.
- `/workspace` is not a persistent volume; local artifacts are not an
  off-instance backup.
- Active output: `scratch/active_runs/<run_id>/`.
- Final bundles: `artifacts/runs/YYYY/MM/<run_id>/`.
- Comparison: `reports/analyses/full_core_qkv_large_k/`.

Every run must preserve exact config, code/dirty provenance, data and graph
checksums, masks, environment, hardware/device, runtime, peak VRAM, history,
last checkpoint, metrics, predictions, failures, and completion marker.

## Run order and commands

1. Run focused QKV equivalence, high-degree FP32-accumulation, checkpoint, and
   matched-self tests.
2. Register the campaign and run the k=5,000 extra-large two-epoch diagnostic.
3. If it fails the predeclared resource gate, test the next prespecified
   smaller candidate until one passes.
4. Lock one common architecture in this README and the three full configs.
5. Enqueue the three 300-epoch runs on separate idle GPUs.
6. Verify bundles and generate one comparison report.

Registration and diagnostic enqueue:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  create-campaign \
  --campaign-id cmp_20260726_full_core_qkv_large_k \
  --name "Full-core large-capacity QKV-GAT k stress test" \
  --scientific-question "Does exact k=5000 improve one-core held-in QKV-GAT reconstruction over k=1000 and a matched self-only control?" \
  --plan experiments/campaigns/cmp_20260726_full_core_qkv_large_k/campaign.yaml \
  --status pilot

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  enqueue-experiment \
  --campaign-id cmp_20260726_full_core_qkv_large_k \
  --config configs/experiment/full_core_qkv_k5000_resource_pilot_xlarge.yaml \
  --priority 20 --max-attempts 1 --gpu 0
```

Workers are managed by supervisor, not loose shell processes. Exact full-run
enqueue commands are:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 enqueue-experiment \
  --campaign-id cmp_20260726_full_core_qkv_large_k \
  --config configs/experiment/full_core_qkv_k5000.yaml \
  --priority 20 --max-attempts 1 --gpu 1

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 enqueue-experiment \
  --campaign-id cmp_20260726_full_core_qkv_large_k \
  --config configs/experiment/full_core_qkv_k1000.yaml \
  --priority 20 --max-attempts 1 --gpu 2

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 enqueue-experiment \
  --campaign-id cmp_20260726_full_core_qkv_large_k \
  --config configs/experiment/full_core_qkv_matched_self.yaml \
  --priority 20 --max-attempts 1 --gpu 3
```

Immutable run and job IDs are recorded after queue claim.

## Current results

The extra-large k=5,000 resource diagnostic
`r_20260726T113751Z_53b4a989_s000_f00_a01_95bbec74` was stopped after the
predeclared runtime gate failed. It spent more than 10.5 minutes in the
two-epoch training section without completing, while observed process GPU
residency rose to 22,878 MiB. The registered failed bundle records termination
status `-15`; it is diagnostic only and has no scientific outcome. The
245,389,160-parameter fallback will use an 80,000-edge exact execution chunk,
which is twice the diagnostic target chunk but retains all neighbors and uses
the memory released by the smaller parameter/optimizer state.

The large fallback
`r_20260726T115403Z_5ea2c7a3_s000_f00_a01_d0374bb6` was likewise stopped
after its two-epoch section exceeded 10 minutes without completing. Its
observed process GPU residency peaked at 20,646 MiB, but it failed the runtime
criterion; its registered bundle records termination status `-15`. Before
running another model outcome, the 111,177,064-parameter fallback was added
with a 200,000-edge exact execution chunk. This is still about 28 times the
parameter count of the prior G2 model and preserves eight QKV/FFN blocks.

That 1,024-width diagnostic
`r_20260726T121004Z_f3b9aa12_s000_f00_a01_c7cf3399` was stopped when its
two-epoch section also exceeded 9.6 minutes. Observed process GPU residency
peaked at 21,272 MiB, and the registered bundle records termination status
`-15`. Before inspecting another outcome, width 896 was added with the same
eight blocks, head dimension, and exact 200,000-edge execution cap. At
85,816,040 parameters it remains more than 21 times larger than the prior G2.

Width 896
`r_20260726T122719Z_5256b939_s000_f00_a01_6ad3a62a` narrowly exceeded the
same runtime boundary and was stopped with registered status `-15`; observed
GPU residency peaked at 19,812 MiB. Before observing another outcome, width
832 was added as the next boundary candidate. It retains eight blocks and
64-dimensional heads and has 74,364,328 parameters.

Width 832
`r_20260726T124250Z_b80cb338_s000_f00_a01_4a840d59` also exceeded 9.6
minutes after graph preparation without completing two epochs. Observed GPU
residency was about 19,006 MiB. The diagnostic was stopped, reconciled, and
preserved as a verified `_FAILED` bundle with no scientific metric. Before
observing another outcome, width 768 was added with eight blocks, twelve
64-dimensional heads, a 3,072-wide FFN/decoder, and 63,731,816 parameters.

Width 768
`r_20260726T130338Z_099347b6_s000_f00_a01_5c1b8aba` remained
memory-safe at about 18,242 MiB but exceeded 10.5 minutes after graph
preparation without completing two epochs. It was stopped through the child
process so the managed worker could preserve a canonical `_FAILED` bundle.
Before observing another outcome, width 704 was added with eight blocks,
eleven 64-dimensional heads, a 2,816-wide FFN/decoder, and 53,918,504
parameters.

Width 704
`r_20260726T132000Z_bb0a39ca_s000_f00_a01_ee9200d5` was stable at
about 17,518 MiB but also crossed the runtime boundary without completing
two epochs. Its managed `_FAILED` bundle contains no scientific result.
Before observing another outcome, width 640 was added with eight blocks, ten
64-dimensional heads, a 2,560-wide FFN/decoder, and 44,924,392 parameters.

The first width-640 attempt
`r_20260726T133510Z_17a5caa5_s000_f00_a01_8a5005c9` failed before
preprocessing because a concurrently managed untracked source module was
temporarily absent. The exact retry
`r_20260726T133700Z_17a5caa5_s000_f00_a02_743f1602` trained normally
but likewise crossed the time boundary at a stable observed residency of
about 16,788 MiB. Neither attempt yields a scientific metric. Because the
weak scaling with width implicated execution-partition overhead, the next
diagnostic restores width 896 and raises only the exact edge-chunk cap from
200,000 to 300,000.

The tuned width-896 diagnostic
`r_20260726T135333Z_fd60b28c_s000_f00_a01_799686e4` reduced
partition overhead but still missed the runtime gate; observed device
residency reached about 21,122 MiB. It was stopped and preserved as
diagnostic failure evidence. Before observing another outcome, the same
300,000-edge exact schedule was paired with width 640. This retains
44,924,392 parameters while combining the two independent resource savings.

The combined width-640/300k diagnostic
`r_20260726T140934Z_c8788fd8_s000_f00_a01_42731c52` was
memory-safe at about 18,878 MiB but still crossed the runtime gate. This
showed that retained-edge computation, rather than partition launch overhead,
dominates the remaining wall time. Before observing another outcome, width
576 was added with nine 64-dimensional heads, eight layers, a 2,304-wide
FFN/decoder, the same 300,000-edge cap, and 36,749,480 parameters.

Width 576 with a 300,000-edge cap completed two finite epochs in
288.45 and 288.77 seconds, with a 15.26 GiB peak allocation. The resulting
24.051-hour projection missed the unchanged limit by about three minutes.
Before another outcome was observed, a final execution-only diagnostic raised
the soft edge cap to 320,000, the maximum possible edges for 64 receivers at
k=5,000. Width, depth, heads, parameters, graph, masks, and attention remain
unchanged.

The final boundary diagnostic
`r_20260726T144512Z_a501c9ef_s000_f00_a01_980e110a` completed two finite
epochs in 286.95 and 287.46 seconds. Its mean projects to 23.934 hours for
300 epochs, leaving 238.9 seconds under the locked runtime limit, and its peak
allocated VRAM was 15.55 GiB. The immutable bundle and checkpoint both passed
checksum verification. This is therefore the largest candidate that passed
all predeclared resource criteria and is locked for all three conclusion runs.

## Conclusion results

All three conclusion runs completed exactly 300 finite epochs and passed
immutable-bundle, checkpoint, graph, mask, prediction, source-provenance, and
environment-provenance verification.

| Role | Run ID | Whole-node Huber | Whole-node R² | PVE |
|---|---|---:|---:|---:|
| k=1,000 QKV-GAT | `r_20260726T151303Z_e6052ff9_s000_f00_a01_59aaad67` | 0.228182 | -0.008443 | -0.844% |
| k=5,000 QKV-GAT | `r_20260726T151303Z_712ec9be_s000_f00_a01_0b694448` | 0.227444 | 0.008082 | 0.808% |
| matched cell-only | `r_20260726T151303Z_4ca683a4_s000_f00_a01_dfe67899` | 0.227723 | -0.005064 | -0.506% |

k=5,000 beat k=1,000 and matched self on all three paired whole-node
technical masks. Its aggregate relative Huber gains were only 0.323% and
0.122%, respectively, below the locked 2% threshold. k=1,000 was 0.201%
worse than matched self. Therefore:

- H1-representation failed: neither graph candidate passed the complete gate.
- H1-k failed: k=5,000 was directionally better but sub-threshold.
- The results do not support low k as the sole or primary explanation for the
  earlier low accuracy. This one core and seed do not establish that k is
  irrelevant in other regimes.

The percentage metric is percent variance explained, not classification
accuracy. These values are held-in transductive reconstruction capacity, not
generalization.

The locked machine-readable comparison and paired rows are in
`reports/analyses/full_core_qkv_large_k/`.

### Diagnostic-only interpretability

Because the representation gate failed, biological/TLS annotation was
suppressed. A prespecified negative-gate override evaluated eight
expression-independent receiver cells as a computational diagnostic only.
Final-layer effective routing was diffuse (mean 3,497.6 effective neighbors
per receiver/head; mean routing-weighted distance 331.6 µm). Deleting the top
5% routed edges changed predictions 7.63 times more than distance-matched null
deletions, but changed Huber by -0.000021, a slight improvement. Thus the
ranking is locally influential to final-layer predictions but is not evidence
of beneficial biological interaction, mechanism, or causality.
