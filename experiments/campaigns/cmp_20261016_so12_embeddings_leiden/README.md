# SO1 + SO2 normalized embeddings and Leiden clustering

Phase: complete. Outcome: inconclusive for biological interpretation; computational outputs verified. This is exploratory, fully observed cell
representation and clustering. It is separate from the active relative-QKV
training run and does not modify that run, its model, or its analysis evidence.

## Task contract

- Objective: export three aligned 32-dimensional representations and Leiden
  labels for all 407,659 mapped cells from both slides, with compact reusable
  arrays, visual summaries, diagnostics, and an immutable registered run bundle.
- Question: how do a linear expression representation, a nonlinear expression
  representation, and a representation including local tissue context change
  the resulting descriptive cell partitions?
- Working hypothesis: expression and local context produce partly shared and
  partly different partitions whose reproducibility can be characterized.
  Alternatives include algorithmic instability, slide/depth confounding, tissue
  composition, and spatial smoothing that creates apparent domains. A visually
  separated cluster is not evidence of a distinct or validated cell type.
- Distinguishing predictions: compare cluster assignments across encoders and
  repeat Leiden at fixed resolution with a second seed. Report slide/core
  composition and spatial agreement alongside stability. Low agreement across
  seeds weakens partition reproducibility; spatial continuity alone cannot
  establish biology because spatial context is an input.
- Estimand and maximum claim: exploratory, transductive clustering of these
  fully observed cells. Embedding parameters are fitted on training donors and
  frozen, but the expression-neighbor graphs and Leiden partitions use all
  cells. This is not unseen-donor generalization, validated batch correction,
  cell-cell communication, improved masked prediction, or causal inference.
- Observations are cells; cores are repeated tissue samples; the 14 verified
  anonymous donor groups are the biological units. Cells are not independent
  biological replicates. No patient-level significance testing is planned.
- Primary numerical diagnostic: adjusted Rand index (ARI) between Leiden seeds
  0 and 1 at resolution 1.0, higher indicating greater algorithmic agreement.
  It is not a ranking criterion for biological validity. Report cluster counts,
  sizes, and cross-method agreement without selecting a favorable resolution.
- Baseline: frozen expression PCA32 from the upstream Pearson preprocessing
  run. Nonlinear RBF kernel PCA32 provides a second non-spatial representation.
  Spatial context PCA32 is a simplified neighborhood-feature comparator.
- Controls: synthetic smoke checks must establish finite aligned outputs,
  recoverable planted structure, valid graph labels, and spatial boundary
  isolation. Seed 1 is the algorithmic stability control. Spatial permutation,
  independent annotations, and biological perturbation controls are not claimed.
  Compare same-cluster spatial-edge agreement with 20 label permutations within
  each core; this preserves core cluster composition, but does not establish
  biological validity because spatial smoothing is part of the encoder.
- Acceptance: all three finite arrays have shape (407659, 32); every cell has
  one label at every prespecified Leiden setting; observation order is verified;
  spatial neighbors never cross a core or slide or exceed 50 micrometers;
  registered data/split identities and upstream checksums match; all outputs
  are verified and published with metrics and a completion marker. Negative or
  unstable partitions remain reportable outcomes, not reasons to tune silently.
- Stop on input identity/order/checksum mismatch, nonfinite results, invalid
  spatial topology, smoke failure, output size exceeding 1 GiB, or disk free
  space falling below the 5 GiB reserve. Record failed attempts and tracebacks.
- Resources: CPU only, at most eight CPU threads, sequential encoder/graph analysis
  with four concurrent independent Leiden setting/seed tasks on each graph,
  no GPUs or neural training. This deliberate exception to the repository GPU
  preference protects the active GPU training run; these bounded classical
  calculations use existing fitted PCA and small kernel matrices. No duplicate
  expression matrices or checkpoints are written.

## Inputs and split

The dataset is `so12_pearson_integration` version `v1`, located through
`spatial_benchmark.paths` at
`data/processed/so12_pearson_integration_v1/`. Its immutable upstream run is
`r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410` from
`cmp_20261016_so12_pearson_preprocessing`. The runner checks the registered
manifest fingerprint, records the upstream identity and checksums, and retains
the existing dataset and split registrations.

The split is `so12_10train_4diagnostic_donors_v1`: training cores 1–10 and 15–24
(301,198 cells, ten donors), diagnostic cores 11–14 and 25–28 (106,461 cells,
four donors). All encoder fitting and kernel-landmark selection use training
cells. The all-cell clustering remains explicitly transductive. The upstream
340 unmapped SO2 FOV246 cells remain excluded; there is no new cell exclusion.
All 1,000 biological genes are retained; technical probes are excluded upstream.
Vendor annotations, donor, slide, disease, and core are not expression inputs.
Core/slide restrict spatial geometry and provide descriptive stratification.
Coordinates use the verified micrometer frame; global coordinates are not
appended to the expression feature vectors.

Plain Pearson residuals can be negative. scVI models count observations, so a
count-based scVI analysis would require a separate raw-count input contract;
these residuals are not valid scVI count inputs. See the
[official scvi-tools input documentation](https://docs.scvi-tools.org/en/1.0.0/tutorials/notebooks/api_overview.html).
The normalized data also use full observed cell totals and therefore cannot
be masked afterward to create a leakage-safe hidden-count benchmark.

## Fixed representations and clustering

1. `pca`: apply the frozen training-fitted Pearson PCA mean/components from the
   upstream run; retain 32 components.
2. `kernel_pca`: RBF kernel PCA with 512 reproducibly selected training
   landmarks balanced across training cores and 32 output components. Set gamma
   to the reciprocal of the median positive squared landmark-pair distance,
   center the landmark kernel, and retain its leading 32 eigenvectors. Record
   the realized parameters, landmark indices, and fitted transform. This
   landmark approximation need not preserve rare states absent from landmarks.
3. `spatial_pca`: combine each cell's expression PCA features with mean
   neighboring PCA features, using spatial weight 0.3. Center each block and
   divide by its scalar RMS estimated on training cells; concatenate the blocks
   multiplied by sqrt(0.7) and sqrt(0.3), respectively, then fit PCA32 using
   training cells. Eligible
   neighbors are the nearest 15 cells within 50 micrometers in the same core
   and slide. Gaussian weights use the farthest eligible neighbor distance as
   sigma. Cells without eligible neighbors use their own expression features;
   record this fallback count and realized neighbor counts.

The spatial construction is inspired by the use of cell and neighborhood
features in [BANKSY](https://www.nature.com/articles/s41588-024-01664-3).
It is a simplified spatial-context PCA comparator, not BANKSY: no azimuthal
Gabor features or complete BANKSY implementation is claimed.

Build a 30-neighbor representation graph for each encoder and run Leiden at
resolutions 0.5, 1.0, and 1.5 with seed 0. Repeat resolution 1.0 with seed 1.
Use Euclidean HNSW neighbors (M=32, efConstruction=120, efSearch=128). Measure
recall against exact neighbors for 256 query cells; require recall >=0.90,
otherwise retry the search at efSearch=256 then 512 and fail if recall remains
below the gate. Directed weights are exp(-squared_distance/(kth_squared_distance+eps));
symmetrize by arithmetic averaging. Leiden uses
`RBConfigurationVertexPartition`, two iterations, and the fixed resolutions.
Parallel HNSW insertion can be nondeterministic even at fixed seed; retain the
actual directed neighbor indices/distances for exact fixed-graph replay.
Representation-neighbor edges may join cells across slides/cores because they
express feature similarity; spatial-context edges may not. Do not force slide
mixing or optimize an embedding for apparent batch removal: slide and tissue
composition are confounded. Display plots are projections, not clustering
inputs or independent validation. Plot the first two representation dimensions
with that label; no UMAP optimization or biological interpretation is implied.

## Run order and commands

Run from the repository root after the separately documented storage retention
pass. Never delete current training artifacts to satisfy this workflow's budget.

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/run_so12_embeddings.py --smoke --threads 8
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/run_so12_embeddings.py --threads 8
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

The full command is launched by a one-shot supervisor service. The standalone
smoke is synthetic and writes no dataset or full experiment. Full execution
repeats that smoke before registration, then writes only to
`scratch/active_runs/<run_id>/`. It publishes verified output under
`artifacts/runs/YYYY/MM/<run_id>/` and records every artifact in the registry.
Retry with `--attempt 2 --retry-of <failed_run_id>` only after diagnosing the
failure. A new primary run ID is always created; successful bundles are immutable.

Attempt 1 (`r_20261016T181056Z_e4e8f852_s000_f00_a01_4a3df5b3`) was stopped
operationally during the first serial Leiden calculation after approximately
five minutes. Its partial arrays, numerical audit, logs, configuration, and
failed status are retained. The retry changes only CPU scheduling: four
independent Leiden settings/seeds can execute concurrently within the existing
eight-thread budget, while encoders and representation graphs remain sequential.
`runtime.leiden_parallel_workers=4` is execution metadata excluded from the
scientific identifier; dimensions, graphs, seeds, resolutions, and all other
scientific settings remain fixed. The exact retry is:

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/run_so12_embeddings.py --threads 8 --attempt 2 --retry-of r_20261016T181056Z_e4e8f852_s000_f00_a01_4a3df5b3
```

Expected artifacts include the three embeddings, Leiden labels keyed to the
existing cell order, fitted transformation parameters, fixed-setting metrics,
plots, runtime log, resolved configuration, source snapshot, environment,
data/split/upstream provenance, checksums, verification, and `_SUCCESS`.
No expression features are copied into the bundle. No neural checkpoint exists
for this analysis, so checkpoint catalog registration is not applicable.

## Completed exploratory analysis

Successful run: `r_20261016T181735Z_e4e8f852_s000_f00_a02_d8caf6c4`. Scientific variant:
`sci_e4e8f8523d0fdae9`. All three arrays have shape (407659, 32), all
values are finite, and every observation has all four aligned integer Leiden
labels. Input/core-coordinate checks, exact-query neighbor-recall gates,
synthetic controls, saved-output verification and portable report checks passed.

| Representation | Clusters r=0.5 | r=1.0 | r=1.5 | r=1.0, seed 1 | Seed ARI at r=1.0 | Mean neighbor recall |
|---|---:|---:|---:|---:|---:|---:|
| Pearson PCA | 9 | 19 | 32 | 19 | 0.7061 | 0.9880 |
| RBF kernel PCA | 10 | 21 | 33 | 23 | 0.6977 | 0.9977 |
| Spatial context PCA | 10 | 17 | 29 | 18 | 0.8253 | 0.9909 |

The primary setting is resolution 1.0, seed 0. These are exploratory partitions,
not validated cell identities. Seed ARI shows moderate sensitivity; the spatial
method's higher value is not evidence of biological superiority. Cross-method
ARI is 0.650047 (PCA/kernel), 0.514149
(PCA/spatial) and 0.499264 (kernel/spatial).

Spatial same-cluster edge agreement is 0.476371 versus within-core permutation
mean 0.241628; the corresponding PCA and kernel values are 0.294657/0.171974 and
0.295648/0.161363. Spatial continuity is partly imposed by the representation.
There are 6,073,375 eligible directed spatial edges, no cross-core edges,
maximum distance 49.9967 micrometers and 266 isolated-cell self fallbacks.

The successful CPU-only run took 22.91 minutes and retained
approximately 542.2 MiB. Peak parent RSS was
1.61 GiB; the largest completed child-process RSS was
2.56 GiB, not the aggregate concurrent worker memory.
Free storage was 10.39 GiB at verification. No GPU or neural
checkpoint was used. The interrupted first attempt is preserved as documented
above; all scientific settings were unchanged on retry.

The human-facing report is `reports/analyses/so12_embeddings_leiden/index.with_legends.html`.
The immutable report and all cell arrays are in
`artifacts/runs/2026/10/r_20261016T181735Z_e4e8f852_s000_f00_a02_d8caf6c4/`.
The curated evidence record is
`results/descriptive_analysis/classical_encoders/res_so12_normalized_embeddings_leiden/`.

### Load embeddings and labels

Run from the project root. Change `method` to `kernel_pca` or `spatial_pca` as
needed. All 32 dimensions and every cell are available; plots display only two
dimensions. Cluster numbers are local to each method and setting.

```python
import numpy as np
import pandas as pd
from spatial_benchmark.paths import current_paths

bundle = current_paths().artifact_root / "runs/2026/10/r_20261016T181735Z_e4e8f852_s000_f00_a02_d8caf6c4"
method = "pca"
embedding = np.load(bundle / "methods" / method / "embedding.npy", mmap_mode="r")
cells = pd.read_csv(bundle / "cells.csv.gz")
clusters = pd.read_csv(bundle / "methods" / method / "clusters.csv.gz")
assert np.array_equal(cells.row_index.to_numpy(), clusters.row_index.to_numpy())
assert embedding.shape == (len(cells), 32)
primary_labels = clusters["leiden_r1.0_s0"].to_numpy()
```

### Limitations and next discriminating test

All three encoders share PCA32 retaining only 8.22% of Pearson-feature variance;
the kernel method additionally uses 512 training landmarks. HNSW construction
can vary across threaded reruns, so actual neighbor lists and distances are
preserved. Mean recall passes the gate but some individual queries have lower
recall. The two Leiden seeds and resolution sweep expose algorithmic sensitivity,
not independent biological replication.

All-cell clustering is transductive despite training-only encoder fitting.
The targeted panel, slide/tissue confounding, segmentation spillover, missing
tissue-barrier modeling, and built-in spatial smoothing limit interpretation.
No method is declared biologically superior, and no batch-removal, predictive,
communication or causal claim is supported. The next discriminating test
requires independent cell-type or orthogonal molecular annotations with
independent-donor replication.
