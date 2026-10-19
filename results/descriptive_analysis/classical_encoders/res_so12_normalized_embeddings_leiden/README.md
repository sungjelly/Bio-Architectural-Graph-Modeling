# SO1/SO2 normalized representations and Leiden partitions

All 407,659 mapped cells now have three 32-dimensional embeddings and four
Leiden label sets per embedding. The computational outputs are verified;
biological interpretation remains **inconclusive**.

| Representation | Clusters r=0.5 | r=1.0 | r=1.5 | r=1.0, seed 1 | Seed ARI at r=1.0 | Mean neighbor recall |
|---|---:|---:|---:|---:|---:|---:|
| Pearson PCA | 9 | 19 | 32 | 19 | 0.7061 | 0.9880 |
| RBF kernel PCA | 10 | 21 | 33 | 23 | 0.6977 | 0.9977 |
| Spatial context PCA | 10 | 17 | 29 | 18 | 0.8253 | 0.9909 |

Seed 0 is used except where stated. The two seed partitions are nonidentical;
ARI measures algorithmic agreement and does not rank biological validity.
Cross-method ARI at resolution 1.0, seed 0 is 0.6500 for
PCA/kernel, 0.5141 for PCA/spatial and
0.4993 for kernel/spatial.

Spatial same-cluster edge agreement is 0.4764 versus a within-core permutation
mean of 0.2416; local continuity is partly built into the spatial representation.
The expression-only comparisons are 0.2947 versus 0.1720 (PCA) and 0.2956 versus
0.1614 (kernel). The spatial graph has 6,073,375 directed edges, no cross-core
edges, maximum distance 49.9967 micrometers, and 266 isolated-cell self fallbacks.
These observations do not establish biological domains or cell-cell interactions.

All encoders use the same training-fitted Pearson PCA32, retaining 8.22% of
normalized-feature variance. The nonlinear encoder adds a 512-landmark
approximation. Encoders are fitted using training donors, but representation
graphs and clusters include all cells: this is transductive analysis. Slide and
tissue composition are confounded; segmentation spillover, tissue barriers,
independent cell-type validation and biological replication remain unresolved.
No trained model or checkpoint was changed.

The [portable report](../../../../reports/analyses/so12_embeddings_leiden/index.with_legends.html)
contains the figures. The [immutable run bundle](../../../../artifacts/runs/2026/10/r_20261016T181735Z_e4e8f852_s000_f00_a02_d8caf6c4/)
contains `cells.csv.gz` and, under `methods/pca`, `methods/kernel_pca` and
`methods/spatial_pca`, `embedding.npy`, `clusters.csv.gz`, fitted parameters and
saved neighbor graphs. Join labels to observations by `row_index`; cluster IDs
are local to each method/setting. The primary column is `leiden_r1.0_s0`.

Successful run: `r_20261016T181735Z_e4e8f852_s000_f00_a02_d8caf6c4`. All source/configuration/output checks and portable
report checks passed. The runner completed in 22.91 minutes,
using CPU only; its bundle is approximately 542.2 MiB.
The first attempt `r_20261016T181056Z_e4e8f852_s000_f00_a01_4a3df5b3` was intentionally interrupted during serial
Leiden, retained with failed status, and retried with four concurrent independent
Leiden tasks and unchanged scientific settings. Its partial results are not used.

The result manifest records exact immutable-source checksums, both attempts,
Leiden seeds 0/1 and fixed preprocessing fold 0. Encoder seed is 0; fold 0 is not
a sealed clustering test. The next discriminating biological test requires
independent annotations or orthogonal measurements and donor-level replication.
