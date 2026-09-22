# SO2 four-block hidden-layer progression clustering

## Task contract

Phase: complete exploratory post-hoc analysis. Outcome: supported as a
descriptive model-behavior result; no biological or generalization claim is
made.

### Objective and scientific question

Use the completed locked four-block SO2 model run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6` to extract the actual
post-block states `h1`, `h2`, and `h3`, then cluster every state jointly across
SO2 cores 15--28. The concrete deliverables are static PNG spatial maps for
`h1`, `h2`, and `h3`, plus a static summary of how the independently fitted
partitions change from `h0` through `hL`. No interactive maps are required.

The primary question is how the model-derived joint Leiden partition changes
as information passes through the four independently parameterized graph
blocks. The working hypothesis is that successive graph steps progressively
change the partition while retaining measurable continuity between adjacent
states. A credible alternative is that most apparent structure is already
present in the intrinsic encoder state or reflects core-specific structure,
with graph blocks producing either little change or unstable repartitioning.

The discriminating predictions are adjacent-layer partition agreement and
the cell-aligned split/merge pattern, reported together with the spatial maps.
Smooth progression predicts stronger agreement for adjacent than distant
states. Abrupt or core-dominated behavior predicts sharp agreement drops or
clusters overwhelmingly restricted to one core.

### Estimand and permitted claim

For each state independently, the estimand is a cell's deterministic Leiden
assignment on the joint 246,063-cell representation graph at resolution 1.0.
Here `h0` is the all-genes-observed node-encoder output; `h1`, `h2`, and `h3`
are the complete post-attention-residual and post-FFN-residual outputs of graph
blocks 1--3; and `hL` is the corresponding output of block 4 immediately
before the decoder.

Cluster identifiers are arbitrary within a layer and do not represent a
lineage across layers. Partition agreement and cell-level contingency, rather
than equal numeric labels, define progression. The maximum defensible claim is
descriptive model behavior: these are representation-space clusters, not cell
types, biological mechanisms, predictive dependencies, or causal effects.

### Units, locked inputs, and computation

The observational unit for clustering is a cell; the fourteen tissue cores
are retained as spatial and composition strata, not treated as independent
replication for a biological claim. Use all 246,063 mapped cells in invariant
core and cell order. Use the verified terminal checkpoint
`checkpoints/last.ckpt` with SHA-256
`2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d`,
the checkpoint-bound prepared expression/covariates, fixed spatial graphs and
relative geometry, an all-zero extraction mask, `model.eval()`, and
`torch.inference_mode()`.

The GPUs are currently occupied by a separate active SO2 training run, so this
post-hoc analysis must use CPU without displacing that work. Extract one full
core at a time. Preserve immutable run artifacts; write resumable work below
`scratch/active_runs/<run_id>/posthoc_reports/` and publish only verified
outputs below `reports/analyses/`.

### Matched clustering and controls

Apply the established hL settings independently to each state: global
mean-centering, exact PCA to 50 components, L2 normalization, a sparse cosine
30-nearest-neighbor graph, and seeded Leiden at resolution 1.0 with seed
20260825. Relabel each partition deterministically by descending size with
stable tie-breaks. Never construct a dense cell-by-cell distance matrix.

Controls and audits are:

- compare the newly captured block-4 tensor against the existing verified hL
  tensor for every core with a prespecified float32 replay tolerance
  (`rtol=1e-6`, `atol=2e-6`), while requiring exact same-process equality to
  the forward result's final full-node state;
- verify prediction invariance when intermediate-state capture is enabled;
- verify exact full-edge/chunked agreement on synthetic graphs;
- verify core/cell order, shapes, finite values, input non-mutation, and file
  checksums;
- repeat clustering from the saved graph or state and require identical labels;
- report cluster count, core concentration, adjacent/distant adjusted Rand
  index and normalized mutual information without interpreting them as
  biological validation.

No external annotation, cell label, disease label, or target-derived metadata
is used to create the clusters. Visual coherence is not validation.

### Acceptance, falsification, and stop criteria

Accept the deliverable when all requested layer tensors are exact post-block
states, all 246,063 cells occur once per layer in the locked order, h4 is
numerically equivalent to the verified hL artifact within the locked replay
tolerance, matched deterministic clustering completes for each
required state, the PNGs are legible and checksum-verified, and focused tests
pass. The scientific hypothesis is considered unsupported if the expected
continuity pattern is absent; that negative result still completes the task.

Stop on checkpoint/input drift, an h4/hL tolerance failure, reordered or missing cells,
non-finite values, mutation of protected inputs, non-deterministic partitions,
dense N-by-N allocation, or insufficient compute/storage. Record the exact
failure rather than silently retrying with changed analysis settings.

### Expected artifacts and verification

Expected artifacts are per-core captured `h0`--`h4` states, persisted `h0`--`h3`
arrays and receipts, an h4-to-verified-hL replay receipt without duplicating h4,
per-layer PCA/kNN/Leiden outputs and parameters, aligned cell assignments,
cluster/core summaries, four layer-specific spatial PNGs, a static
progression PNG, and a checksum manifest. The existing hL clustering is reused
only after its manifest verifies; if no compatible SO2 h0 partition exists,
`h0` is extracted and clustered under the same locked settings rather than
substituting a different model or raw-expression clustering.

The final report must record the exact command, run/checkpoint identity, CPU
environment, timing, tests, and limitations. This exploratory visualization is
not promoted to the curated `results/` evidence layer.

## Completed result

The verified report is
`reports/analyses/so2_14core_hidden_layer_progression/r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6/`.
All 246,063 cells were present once in every layer. The independently clustered
partitions contained 16 clusters at h0, 20 at h1, and 19 each at h2, h3, and
hL. Cluster size ranges were 957--35,277 at h0, 965--27,503 at h1,
1,031--29,446 at h2, 1,048--28,908 at h3, and 3,104--18,750 in the reused hL
partition.

Adjacent-layer agreement was:

- h0 -> h1: ARI 0.515866; NMI 0.662202;
- h1 -> h2: ARI 0.656866; NMI 0.739635;
- h2 -> h3: ARI 0.577895; NMI 0.737059;
- h3 -> hL: ARI 0.670017; NMI 0.792980.

All four new layer graphs were connected, and a second seeded Leiden run
reproduced each label vector exactly. The captured fourth-block output equaled
the same-process public final state exactly. It matched the archived hL array
within the prespecified tolerance in all 14 cores: 13 were bitwise equal, and
the sole nonzero replay difference had maximum absolute magnitude
`9.5367431640625e-07`, below `atol=2e-6` with `rtol=1e-6`. The 14 core forwards
took 390.94 seconds in aggregate. On the provenance-bound full invocation,
parallel clustering completed between 58 and 117 minutes by layer; the report
was published after approximately 118 minutes. All four spatial maps and the
transition heatmap passed visual inspection. All 75 manifest-recorded files
matched their recorded sizes and SHA-256 checksums, and a second identical
invocation verified and reopened the immutable report in 22 seconds.

Focused validation completed with 55 passed tests and two upstream Torch JIT
deprecation warnings. The full repository suite completed with 1,365 passed,
one skipped, and 34 failures. All 34 failures were unchanged environment-bound
tests requiring unavailable locked adjacency-ablation materializations,
prepared multiscale geometry, MyJJu locked configs, or the audited external
MyJJu source tree; none exercised this workflow or its modified model API. The
repository doctor verified database integrity and the checked configs, but
returned nonzero because the existing checkpoint catalog contains 40
checkpoint artifacts and only 39 indexed checkpoints. It also reported the
expected worker lock held by the separate active run. These repository-state
issues predate and are independent of this post-hoc report.

Observed adjacency agreement is moderate rather than identity, with the
highest adjacent agreement between h3 and hL. This supports the narrow
descriptive statement that the fitted model changes its representation-space
partition across blocks while retaining measurable continuity. It does not
show that the clusters are cell types, that the changes are biologically
meaningful, or that graph-specific prediction generalizes.

The exact reproduction command is:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-so2-hidden-layer-progression \
  --run-id r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6 \
  --n-neighbors 30 --leiden-resolution 1.0 --pca-components 50 \
  --random-seed 20260825 --device cpu --cpu-threads 40 \
  --cluster-workers 4 --dpi 300
```
