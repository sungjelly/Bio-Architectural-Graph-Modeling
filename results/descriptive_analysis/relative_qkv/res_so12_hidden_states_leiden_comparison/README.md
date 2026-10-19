# SO1/SO2 hidden-state and PCA cell partitions

**The full evaluation is complete and this computational result is verified.**
Both prespecified Leiden seeds completed; biological interpretation remains
inconclusive.

[Open the final portable report](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/report/index.html) ·
[Download maps, legends and tables](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/so12_all28cores_h0_hL_PCA_six_core_images.zip) ·
[View the six-core overview](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/figures/six_core_five_method_segmentation.png)

The epoch-151 checkpoint of the completed four-layer SO1/SO2 model receives
**all observed genes**. h0 is the trained cell encoder output before graph
blocks; hL follows four graph blocks and precedes the expression decoder. Each
partition includes all **407,659 mapped cells across 28 cores**. The three prior
PCA partitions and their palettes are reused without refitting.

| Representation | Global primary clusters |
|---|---:|
| h0, initial embedding | 16 |
| hL, final hidden state | 22 |
| Pearson PCA | 19 |
| RBF kernel PCA | 21 |
| Spatial PCA | 17 |

Primary Leiden uses resolution 1.0 and seed 0. The same **117,996 original cell
polygons** from cores **1, 9, 13, 15, 21 and 23** appear in five matched panels per
core. Core 13 is model validation; the other five are training. Global palettes
are consistent across cores within each method, but colors and cluster IDs are
independent across methods. The complete legend contains 95 method-specific
entries. Reported global cluster counts refer to all 28 cores. Individual maps,
overview and legends are available as PNG and PDF.

| Fixed-partition comparison | ARI, all cells | ARI, six displayed cores pooled |
|---|---:|---:|
| h0 versus hL | 0.370156 | 0.307399 |
| hL versus Pearson PCA | 0.284910 | 0.322338 |
| hL versus RBF kernel PCA | 0.277367 | 0.360973 |
| hL versus spatial PCA | 0.393237 | 0.601928 |

Full five-method matrices and cluster counts for all cells, the displayed cores
pooled and each displayed core are in the [comparison tables](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/report/comparison_metrics.json).
Subsets retain the original all-cell assignments; no subset was reclustered.
ARI measures agreement, not biological validity, and the higher hL/spatial-PCA
agreement in the selected six cores does not establish a graph benefit.

| Fixed-graph algorithmic repeat | Seed-0 clusters | Seed-1 clusters | Seed agreement, ARI |
|---|---:|---:|---:|
| h0 | 16 | 16 | 0.8069532157597211 |
| hL | 22 | 21 | 0.8837090007869288 |

Both repetitions converged on their respective fixed graphs. Their nonidentical
memberships, and the hL cluster-count change, show algorithmic seed sensitivity.
These are two Leiden seeds, not two model-training seeds or biological replicates.

No primary h0 cluster has more than 90% of its cells in one core. Five primary hL
clusters do, as recorded in the [final clustering summary](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/clustering/summary.json):

| hL cluster | Largest contributing core | Fraction of cluster from that core |
|---|---:|---:|
| hL-03 | 23 | 98.8161% |
| hL-10 | 15 | 99.9017% |
| hL-12 | 15 | 99.8426% |
| hL-17 | 3 | 94.3886% |
| hL-21 | 21 | 98.9613% |

Joint clustering across all 28 cores therefore does not establish batch removal
or integration quality. Core concentration alone does not identify its cause.

The comparison retains material pipeline differences:

| Setting | h0 and hL | Previous PCA methods |
|---|---|---|
| Dimensions | 256 direct hidden features | 32 encoder features |
| Representation neighbors | Exhaustive FP32 cosine k30 | Approximate Euclidean HNSW k30 |
| Edge weights | Unweighted undirected union | Exponential distance weights, arithmetic-mean symmetrization |
| Primary Leiden iterations | Until convergence | Two iterations |
| Spatial context | Model graph context enters hL; morphology enters h0 | Spatial PCA includes up to 15 neighbors within 50 µm, weight 0.3 |

Differences combine representation and clustering-pipeline effects. All-cell
clustering is transductive despite the original donor-separated model and
encoder fitting. One trained model seed is used. Independent cell-type
annotations, patient replication, geometry ablation, graphless controls, new
spatial nulls, segmentation perturbation and causal validation remain untested.

All 117,996 displayed polygons passed keyed coverage and geometry checks, with
zero repairs and zero alignment mismatches. One polygon-centroid distance above
5 µm passed because the supplied cell center lies inside its polygon; eight
centers outside their polygons remained within the 5 µm tolerance. Every panel
uses matched physical bounds, equal aspect, orientation and 100 µm scale bars.
All six individual final maps, overview and legend were visually reviewed;
overview, legend and cores 13/23 had an additional independent review. All eight
PNGs decoded and eight PDFs passed header checks. The six HTML previews exactly
match independent thumbnail generation from their source PNGs; all 57 original
PCA colors and all 95 legend entries were verified. Independent numerical replay,
seed-agreement checks and download-archive CRC passed.

The [final manifest](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/manifest.json)
binds **249 files**. The `_SUCCESS` marker records full completion at
**2026-10-17 04:54:25 UTC**, and the evaluation registry is completed. The curated
record includes Leiden seeds `[0, 1]` and 40 checksum-verified immutable sources.

| Final artifact | SHA-256 |
|---|---|
| Manifest | `0a1343215834e85dfee2a364e91b3fe6094623dcd9c7107c1f36f247cacbd81b` |
| Download ZIP, 18,619,413 bytes | `638260bbc8e464c84a838b8102c799fedc907423c1e79241479db55b4025b3e8` |
| Root final review | `fc3163d467152ffd7fcecebd55f84bc69e6a136bdd772e94ce751e4a810678fe` |
| Independent final render review | `3bd2f9c8ba0a2fa5cefaae844f6e2a072daa2737ad5eb2398fd9d098ed3620fb` |

The [root final review](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/diagnostics/final_review.json),
[independent numerical audit](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/diagnostics/independent_numeric_audit.json)
and [independent render review](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/diagnostics/final_render_review.json)
record the verification scope and source hashes.

The earlier [primary-only delivery](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z_primary_delivery/index.html)
was published at **03:38:23 UTC**, before secondary clustering finished. Its
`_PRIMARY_SUCCESS` marker, original PNG bytes and pending-at-creation labels
remain unchanged as a historical snapshot; they do not describe the current
completed evaluation. Its manifest SHA-256 is
`1ef870678568f68ff6f4bd9da8707fbe1fe0a429a821616f9dd1d2410e167762`.

Two earlier evaluations contribute no scientific outputs and remain visible in
[immutable failure provenance](../../../../reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/provenance/failed_attempts/index.json):

- `ev_so12_h0_hl_e151_20261017T004300Z`: logger argument collision before GPU
  computation.
- `ev_so12_h0_hl_e151_20261017T004600Z`: an exact-invariance pilot detected an
  omitted deterministic-mode flag; no embeddings were saved. The successful
  retry restored the training deterministic mode and passed exact invariance.

Source model run: `r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b`.
Source PCA run: `r_20261016T181735Z_e4e8f852_s000_f00_a02_d8caf6c4`.
No training, checkpoint or published source was modified.
