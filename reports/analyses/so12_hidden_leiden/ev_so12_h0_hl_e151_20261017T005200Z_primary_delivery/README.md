# SO1/SO2 primary cell-representation maps

Primary seed-0 results are complete; secondary seed-1 stability checks remain
pending as of 2026-10-17T03:36:32.433125+00:00. This isolated delivery does not replace or complete
parent evaluation `ev_so12_h0_hl_e151_20261017T005200Z`.

Open [the portable report](index.html) for all six five-method segmentation maps,
full color keys, cluster counts, pairwise agreement and pipeline definitions.
The [overview](figures/six_core_primary_preview.png) retains exact verified image
bytes. PNG filenames/titles retain “primary preview”; the primary partitions are
complete, while additional stability validation is pending.

The epoch-151 model uses all observed genes. h0 is before graph processing and
hL follows four graph blocks. All 407,659 cells from 28 cores enter each primary
partition; the displayed six cores contain 117,996 original cell polygons.

| Method | Global primary clusters |
|---|---:|
| h0 | 16 |
| hL | 22 |
| Pearson PCA | 19 |
| RBF kernel PCA | 21 |
| Spatial PCA | 17 |

All-cell h0–hL ARI is 0.370156. hL agreement with Pearson PCA,
RBF kernel PCA and spatial PCA is 0.284910, 0.277367 and
0.393237, respectively. These measure partition agreement and do not rank
biological validity. Exact tables for all cells, the displayed cores pooled and
each core are under `tables/`.

Hidden-state clustering uses direct 256-dimensional cosine k30 exhaustive
neighbors, unweighted union graphs and Leiden to convergence. The prior PCA
methods use 32 dimensions, Euclidean HNSW k30 neighbors, exponential edge weights,
arithmetic-mean symmetrization and two Leiden iterations. All primary partitions
use resolution 1.0 and seed 0; no subset was reclustered. Representation and
clustering-pipeline differences are therefore not separated.

All five columns share the same original cell polygons and physical geometry.
Cluster colors/IDs are independent across methods; global palettes remain fixed
across cores. Core 13 is model validation; the other five displayed cores are
training. All-cell clustering is transductive, without independent biological
annotations or causal evidence.

`metadata.json` records primary status and the secondary status at creation,
checkpoint and label provenance, image-byte preservation and replayed metrics.
`audits/` retains the verified extraction, graph, label and visual checks.
`provenance/failed_attempts/` retains two operational failures: logger argument
collision before GPU work (004300Z), and an exact-invariance pilot detecting an
omitted deterministic flag before any saved embeddings (004600Z). Their outputs
were not used. No checkpoint, embedding array or unrestricted cell table is
duplicated in this delivery.
