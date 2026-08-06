# MyJJu GeneMAE full-panel k=1000 Jacobian replay

## Status

- Phase: pilot
- Outcome: pending
- Campaign: `cmp_20260802_myjju_genemae_full_panel_k1000_jacobian`
- Design: exploratory, held-in, frozen-model topology stress test
- Upstream model campaign: `cmp_20260730_myjju_genemae_10core_comparison`
- Upstream gradient audit: `cmp_20260731_myjju_genemae_gradient_audit_cpu_replay`
- Frozen implementation protocol:
  `experiments/campaigns/cmp_20260802_myjju_genemae_full_panel_k1000_jacobian/implementation_protocol.yaml`

This campaign recomputes the source-style GeneMAE Jacobian for all 1,000
ordered biological genes across the ten adjacent-normal cores. It replays all
seven retained epoch-199 checkpoints on newly constructed symmetric-union
k-nearest-neighbour graphs with `k=1000` and does not retrain the models.

That distinction is substantive. The checkpoints were trained on k=15 graphs;
therefore, the result is an out-of-training-topology sensitivity stress test.
It is not the Jacobian of a model trained at k=1000, and it cannot establish a
biological or causal mechanism.

## Task contract

### Objective and deliverables

Produce a complete, provenance-bound 1,000-by-1,000 Jacobian for the frozen
seven-checkpoint ensemble, with equal weighting across the ten tissue cores.
Required outputs are:

1. the signed directed matrix before any absolute-value transform;
2. the source-style symmetric absolute display matrix;
3. ordered row and column gene names and checksums;
4. per-seed, per-core coverage and finite-value checks;
5. graph identities plus node, edge, and realized degree summaries;
6. numerical equivalence controls for the memory-bounded Jacobian algorithm;
7. a zoomable all-label image and a portable overview image; and
8. a registered immutable run bundle with configuration, code/data/model
   provenance, append-only metrics, status, and completion marker.

### Scientific question and alternatives

Question: what full-panel source-style local sensitivity does the frozen
GeneMAE ensemble exhibit when replayed on a dense k=1000 spatial graph?

Primary computational hypothesis: the exact memory-bounded estimator produces
a finite, deterministic 1,000-by-1,000 matrix and reproduces the existing k=15
39-gene source-style statistic when run under the original topology.

Credible alternatives are that dense topology causes GPU memory or numerical
failure; that full-panel ranks are dominated by the self/cell-autonomous path;
that k=1000 replay substantially changes behavior because it is out of the
training topology; or that apparent hotspots arise from absolute-value
symmetrisation and diagonal effects rather than stable directed sensitivities.

Predictions that distinguish these alternatives:

- an algebraically equivalent small-model implementation and a k=15 replay
  must agree with the established estimator within numerical tolerance;
- all seven seeds and ten cores must yield finite matrices without selecting a
  favorable seed or core;
- signed directed, diagonal, off-diagonal, and symmetric absolute summaries
  must be retained separately; and
- k=1000 graph manifests must show the requested minimum neighbor count for
  every eligible tile and report the higher degrees created by symmetric union.

### Estimand and maximum claim

For target gene `t` and source gene `s`, the retained source-style estimand is

```text
J[t, s] = (1 / N) * d sum_i reconstruction[i, t]
                       / d uniform_shift[s]

where input[i, s] is replaced by input[i, s] + uniform_shift[s]
for every input cell i in the same graph tile.
```

By the chain rule, this equals the established sum over all output-cell and
input-cell derivatives, divided by the number of input cells. Tile numerators
are weighted by their cell counts, cores are averaged equally, and seeds are
averaged equally. The display transform is
`0.5 * (abs(J) + abs(J.T))`; it is unsigned and undirected. The signed directed
matrix remains canonical.

The maximum defensible claim is a frozen-model, held-in, topology-stress
`model-implied sensitivity` on normalized compositional inputs. It is not a
correlation, a direct cell-cell interaction, a patient-generalized result, a
raw-molecule effect, or a causal influence.

### Inputs, units, preprocessing, and graph

- Models: all seven checksum-verified seed 0--6 final checkpoints; no best-seed
  selection.
- Cohort: `ANC-01` through `ANC-10`, 117,386 held-in cells and 1,000 ordered
  biological genes. Tissue core is the observational unit; seeds are technical
  repeats.
- Input scale: upstream full-cell `log1p(CP10k)`. Gradients are with respect to
  this normalized compositional representation, not counts.
- Graph: deterministic spatial kNN, `k=1000`, symmetric union, Gaussian distance
  feature, no constructed self-edges, convolution self-loops enabled, no
  cross-core or cross-tile edges.
- Degree semantics: each node has at least `min(1000, tile_nodes - 1)` outgoing
  constructed edges; symmetric union can increase degree above 1,000.
- Tiling: retain the upstream maximum of 7,000 cells when feasible. Every tile
  used for the requested graph must contain at least 1,001 cells. A resource-
  driven tiling change requires a documented new variant and is not silent.
- Split/generalization: none. All cells were previously used for model fitting.
  The analysis is transductive and held-in.

No raw or clinical input is modified. No row-level cell identifiers or patient
identifiers are exported. Cell-level tensors remain in owned active scratch and
are excluded from the final report bundle.

### Controls, acceptance, falsification, and stop criteria

The cheapest discriminating controls run before the full computation:

1. compare the uniform-shift Jacobian with the explicit all-cell derivative sum
   on a deterministic analytical model;
2. reproduce the stored 39-by-39 k=15 source-style block within absolute and
   relative numerical tolerances fixed in the implementation protocol;
3. benchmark one worst-case k=1000 tile for finite outputs, determinism, peak
   VRAM, and projected runtime; and
4. verify every k=1000 tile's node count, edge alignment, symmetry, checksums,
   and realized degree minimum.

Acceptance requires all controls to pass, complete 7-seed by 10-core coverage,
exact 1,000-by-1,000 shapes, ordered-gene equality, finite values, deterministic
aggregate checksums, and successful artifact and registry verification. The
heatmap alone is not completion.

Stop before production on checkpoint/cohort/gene-order mismatch, a tile below
1,001 cells, graph identity or degree failure, nonfinite output/gradient,
numerical-equivalence failure, unsafe GPU collision, pilot peak allocation over
20.5 GiB, projected production runtime over 48 GPU-hours without an explicit
revised decision, insufficient disk, or inability to finalize a verified run.
A negative or resource-infeasible pilot is retained and reported rather than
silently changing the estimand.

### Resource plan

Before the pilot, inspect every GPU, its processes, framework visibility, and
free memory. Existing jobs have priority and are not preempted. Benchmark the
largest tile on one safely free RTX 3090, recording CUDA/PyTorch/PyG versions,
device ID, runtime, throughput, host memory, and peak allocated/reserved VRAM.
Only after the pilot passes may independent seed shards be distributed across
safely free devices. Each shard gets an isolated scratch directory, config,
log, metrics stream, and completion marker.

### Expected artifacts and verification

The final bundle must contain compact matrices (`npz`), gene-order tables,
graph manifests, seed/core metrics, the zoomable all-label image, an overview
PNG, a concise Markdown/HTML report, resolved configuration, provenance, logs,
checksums, status, and completion marker. Exact commands will be added here
when the implementation is frozen; the expected entry points are:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/recompute_myjju_full_panel_jacobian.py --help
PYTHONPATH=src /venv/main/bin/python -m pytest tests/test_full_panel_jacobian.py -q
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

## Current evidence

- The retained source-style result covers only 39 selected genes, not the full
  panel.
- The frozen models use 1,000 ordered biological genes and were trained at
  k=15.
- All eight GPUs were occupied by a separate registered adjacent-normal
  ablation campaign when this contract was written; no device was preempted.
- No full-panel k=1000 value has yet been inspected. This remains exploratory
  because the earlier 39-gene result and upstream model evaluation are known.
