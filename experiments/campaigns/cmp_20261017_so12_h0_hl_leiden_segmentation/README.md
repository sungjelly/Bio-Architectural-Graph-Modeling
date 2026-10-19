# SO1 + SO2 hidden-state Leiden and cell-segmentation comparison

Phase: complete. Outcome: inconclusive. Exploratory, fully observed, transductive
representation analysis requested on 2026-10-17. The user explicitly selected
all observed genes for inference and requested comparison with the three
previous PCA methods on the same six tissue cores.

## Task contract

Extract h0 (trained node encoder output, before graph processing) and hL
(fourth graph-block output, before expression decoding) from the completed
SO12 four-layer model. Independently cluster all 407,659 mapped cells across
28 cores for each representation. Produce actual cell-boundary segmentation
maps for cores 1, 9, 13, 15, 21 and 23, with five matched panels per core:
h0, hL, PCA, RBF kernel PCA and spatial-context PCA. Reuse the previous PCA
methods' saved resolution-1.0, seed-0 labels; do not refit or relabel them.
Deliver six high-resolution PNG/PDF comparisons, a combined overview, complete
color keys, a portable HTML report, all-cell embeddings/labels, numerical
diagnostics and a registered, checksummed versioned evaluation.

The descriptive question is how partitions and tissue patterns differ before
and after graph processing and across the prior classical representations.
The working hypothesis is that hL differs from h0; credible alternatives are
preserved cell-autonomous structure, slide/core composition, spatial smoothing,
technical effects, and clustering settings. Cross-partition ARI and seed
stability characterize these alternatives without selecting favorable settings.
No cluster number, visual appearance or spatial continuity is an optimization
target. This analysis cannot establish cell identities, graph-specific predictive
gain, biological integration, cell communication, mechanism or causality.
hL-h0 also includes feedforward/residual transformations, not an isolated
neighbor contribution.

Cells are observations; donors are independent biological units. All 28 cores
are clustered jointly, including training and validation cells, so partitions
are transductive. Of the six displayed cores, core 13 was validation and the
other five were training; they contain 117,996 cells in total. No patient-level
significance claims or clinical-label analyses are planned. The existing 340
unmapped SO2 FOV246 cells remain excluded. No new cell filtering is allowed.

## Immutable inputs and extraction

- Model source run: `r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b`,
  final bundle under `artifacts/runs/2026/10/`, best epoch 151.
- Best checkpoint SHA-256:
  `6b4526508d223fb8a286084f57fc5f4b36129db6afeffaa328672d2c06226af4`.
- Integrated data: `data/processed/so12_pearson_integration_v1`, all 1,000
  biological genes, fixed training-only preprocessing and allowed 22 morphology
  fields. Check loaded preprocessing provenance against the training binding.
- PCA source run: `r_20261016T181735Z_e4e8f852_s000_f00_a02_d8caf6c4`.
  Use `methods/{pca,kernel_pca,spatial_pca}/clusters.csv.gz`,
  column `leiden_r1.0_s0`, joined by exact slide-qualified cell keys.
- Segmentation: immutable raw SO1/SO2 polygon CSVs, streamed by the existing
  canonical polygon loader. Preserve row-ordered vertices, source coordinate
  conversion 0.120281 micrometres/pixel, and recorded alignment checks.

Use the original complete within-core spatial graphs and 70-D relative geometry.
Inference uses `eval()`, `inference_mode()`, an all-zero gene mask, the canonical
visible-count Pearson input function and FP32 arithmetic with TF32 disabled.
Match training's `torch.use_deterministic_algorithms(True)` execution setting
with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
No weights, training settings, data, graph or prior analysis are changed.
Save aligned float32 h0 and hL arrays with 256 dimensions and compact raw
delta-L2 diagnostics; do not duplicate the full delta vector or checkpoint.
Pilot core 23 must establish finite outputs, unchanged predictions when
intermediate outputs are requested, correct h0/hL definitions, throughput
and GPU peak-memory feasibility before the full extraction.

## Fixed clustering and comparison settings

For h0 and hL separately, retain all 256 dimensions, normalize each row to unit
L2 norm solely for cosine distances, and construct an exact sparse 30-nearest-
neighbor graph using bounded GPU query blocks. Never materialize an all-cell
square distance matrix. Remove self edges; resolve equal-score neighbor ties
by row index. Symmetrize by undirected edge union and use an unweighted graph,
following the historical direct QKV representation protocol. Run Leiden
RBConfiguration at resolution 1.0 to convergence, with seed 0 for the displayed
partition and seed 1 as an algorithmic stability diagnostic. Save the actual
neighbor lists, distances, graph provenance, labels and cluster sizes.
Validate exact GPU neighbors against a bounded CPU reference before scaling.

The prior PCA-family partitions remain unchanged: they use 32-D representations,
Euclidean HNSW neighbors, weighted graphs and two Leiden iterations. Therefore
their image comparison also includes metric/graph/algorithm differences; it
cannot isolate encoder quality. Cross-method agreement is descriptive only.
Cluster identifiers and colors are independent across methods. Keep each
method's color mapping fixed across the six cores, display complete legends,
and do not imply matching numbers/colors are matching cell identities.

Render true cell polygons with equal physical aspect, identical bounds and
orientation across methods for each core, a micrometre scale bar and visible
tissue gaps. Label h0 as initial encoder embedding and hL as final hidden state.
Avoid replacing segmentation boundaries with centroid dots.

## Resources, acceptance and failure criteria

At inspection all four RTX 3090 GPUs were idle and roughly 11 GiB disk was free.
Use a supervisor-managed analysis, four independent GPU workers balanced by
core size for extraction, then independent h0/hL GPU neighbor jobs and bounded
CPU Leiden/rendering. Reserve at least 5 GiB disk; expected retained output is
below 2 GiB. Respect existing GPU locks/processes and never displace other work.
Do not install dependencies or alter a running training/inference environment.

Register the evaluation against the exact best checkpoint. Active output goes
under `scratch/active_runs/<evaluation_id>/`; publish the verified final report
under `reports/analyses/so12_hidden_leiden/<evaluation_id>/`. Preserve failed
attempts and error logs. Stage receipts permit resume only when their source
binding and file hashes match.

Acceptance requires all-cell key uniqueness/order/coverage, finite [407659,256]
representations, one integer label per cell in each setting, exact-neighbor
pilot agreement, correct polygon joins for all 117,996 displayed cells, six
complete five-panel images, complete color keys and visual inspection, frozen
PCA-label equality, validated provenance/checksums and successful registry
publication. Stop on source mismatch, missing/reordered cells, nonfinite or
zero-norm embeddings, invalid topology/labels, insufficient memory/storage,
failed pilot or incomplete segmentation. A low ARI or visually similar map is
a reportable result, not permission to tune. Record exact commands and outcomes
below as implementation and execution finish.

## Execution record

The user reiterated that clustering must include all 28 cores; this was already
the locked cohort. Only image selection is restricted to six cores. The existing
PCA-family partitions also cover the full 407,659-cell cohort.

Completed versioned evaluation: `ev_so12_h0_hl_e151_20261017T005200Z`, managed by
supervisor program `bagm_so12_hidden_leiden`. Entry point:

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python \
  scripts/analysis/run_so12_hidden_leiden.py \
  --evaluation-id ev_so12_h0_hl_e151_20261017T005200Z \
  --retry-of ev_so12_h0_hl_e151_20261017T004600Z
```

Prelaunch checks: 5 extraction tests, 4 exact-neighbor/Leiden tests and 6 renderer/
comparison tests passed. The training checkpoint and 68-file prior PCA bundle
passed their native checksum/completion contracts. The training model reloads
strictly with 5,003,016 finite parameters and four graph blocks. The new analysis
has a source snapshot, resolved configuration, environment record, append-only
events and per-worker logs. Final publication was gated on numerical and visual
verification; training was not changed.

Attempt `ev_so12_h0_hl_e151_20261017T004300Z` passed full input preparation, then
failed before launching any GPU job because the event logger's positional `name`
argument collided with its job-name payload. The failed evaluation and complete
error are retained. The logger parameter was corrected, and a real child-process
smoke test now verifies start/finish events and log capture (one test passed).
The explicit linked retry keeps every scientific setting unchanged and repeats
input verification before proceeding. No embedding or cluster was generated in
the failed attempt.

Attempt `ev_so12_h0_hl_e151_20261017T004600Z` passed preparation and executed
core 23, but the exact prediction-invariance pilot rejected nondeterministic
CUDA accumulation. The extraction worker had omitted the deterministic
algorithm setting explicitly used by the original training entry point. The
worker now restores that setting; the exact equality gate is unchanged. No
embedding receipt or cluster was saved in this failed attempt. The third linked
attempt uses unchanged weights, inputs and scientific settings, with corrected
execution reproducibility and a bounded multi-neighbor CUDA regression check.

The third attempt passed the complete core-23 pilot at 00:55 UTC. Exact
prediction equality, direct-encoder h0 equality and final-state hL equality all
passed. Two full forwards plus audit took 5.095 seconds, with 4.321 GiB peak
allocated and 5.736 GiB peak reserved GPU memory. Four independent workers then
started the other 27 cores. All 17 focused tests pass in the corrected code.

## Primary partition milestone, before stability completion

All 28 cores and 407,659 cells completed fully observed extraction. Both exact
cosine neighbor graphs passed their CPU reference and independent topology
audits. The global graphs allow cross-core neighbors; all 378 pairs of distinct
cores are represented in each graph.

The hL primary partition completed at 2026-10-17 01:54:13 UTC with 22 clusters;
h0 completed at 02:11:18 UTC with 16 clusters. Independent reconstruction of
the graphs, label coverage/order/checksums and modularity/RB quality all pass.
Primary h0–hL adjusted Rand agreement is 0.3701562886. On all cells, h0 agreement
with PCA, RBF kernel PCA and spatial PCA is respectively 0.246089, 0.222434 and
0.236825; hL agreement is respectively 0.284910, 0.277367 and 0.393237. Existing
PCA-family cluster counts remain 19, 21 and 17. These are descriptive partition
comparisons across the explicitly different pipelines above.

Seven primary preview PNGs were generated separately while both fixed-graph
seed-1 jobs continued. They cover the six requested cores, preserve their exact
117,996 polygons, and explicitly mark stability as pending. Root reviewed all
six individual images; a second reviewer checked the overview and cores 13/23.
All preview file/source checksums pass, with zero polygon repairs or alignment
failures. The preview helper and eight passing focused checks are captured in
supplemental provenance; frozen execution sources remain unchanged.

The primary previews did not satisfy the final publication gate. Full seed-1
stability results, final PNG/PDF/HTML output, independent numeric audit and final
visual review were still required at this milestone. They subsequently passed
as recorded in the final completion section below.

## Permanent primary-map delivery while stability runs continue

A separate, immutable seed-0 delivery was published on 2026-10-17 under
`reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z_primary_delivery/`.
It contains the exact seven previously reviewed PNGs, complete legends, a
portable HTML comparison, all-cell and per-core ARI/count tables, source and
failure provenance, and an 8,497,748-byte ZIP. All copied source hashes, PNG
decoding, six embedded HTML images and ZIP CRCs passed verification. Its
manifest SHA-256 is
`1ef870678568f68ff6f4bd9da8707fbe1fe0a429a821616f9dd1d2410e167762`;
registry artifact 2804 binds it to the existing evaluation and training run.

The `_PRIMARY_SUCCESS` marker applies only to these completed seed-0
deliverables. Both seed-1 jobs were still computing at that publication. The parent
evaluation remained `running` at that point; its original acceptance gates were unchanged.
This delivery neither substitutes a favorable seed nor asserts completed
stability validation. The fixed seed-0 image selection was specified before
execution. All 28 cores remain in each partition; only displayed tissue is
restricted to the six requested cores. The package builder is retained inside
the delivery as `provenance/package_so12_primary_delivery.py`.

## Verified final completion

Both prespecified Leiden seeds completed on both full-cohort graphs. hL seed 1
finished at 03:59:23 UTC; h0 seed 1 finished at 04:45:38 UTC. The completed
seed-0/seed-1 cluster counts are 16/16 for h0 and 22/21 for hL. Their respective
fixed-graph adjusted Rand agreements are **0.8069532158** and
**0.8837090008**. The displayed primary partitions remain seed 0, as specified
before execution. Their h0–hL agreement is **0.3701562886**. The three previous
PCA partitions remain unchanged at 19/21/17 clusters.

All 407,659 mapped cells across all 28 cores were jointly clustered for each
representation. Inference supplied all observed genes to the frozen epoch-151
checkpoint. The six displayed cores retain all 117,996 original cell polygons,
with no repairs, no failed alignment checks and exact slide-qualified joins.
The documented alignment rule permits centroid distance at most 5 micrometres
or polygon containment: one cell exceeded the distance threshold but passed
containment, and eight coordinates outside their polygons passed the distance
criterion. No claim that every coordinate lies inside its polygon is made.

The complete V2 independent audit passed in 57.35 seconds. It checked source,
checkpoint and data hashes, all embeddings and delta arrays, exact graph
reconstruction, all four label vectors and their objective values, both seed
ARIs, unchanged PCA labels, 80 independently recomputed pairwise ARIs, complete
legends and every renderer artifact. Root visually inspected all six final
images, the overview and legend. A second review verified all eight PNGs and
eight PDFs, six embedded HTML thumbnails, 95 legend entries, 57 preserved PCA
colors, polygon coverage, and all comparison populations against the earlier
primary delivery. All 17 prelaunch/regression tests and eight supplemental
primary-preview checks passed. Numerical and visual audits are retained in
`diagnostics/independent_numeric_audit.json`, `diagnostics/final_render_review.json`
and `diagnostics/final_review.json`.

Final publication completed at **2026-10-17 04:54:25 UTC** under
`reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z/`.
The 249-file manifest and its `_SUCCESS` binding were verified before and after
atomic publication; the registry records the evaluation as `completed` and
indexes its final manifest artifact. Manifest SHA-256:
`0a1343215834e85dfee2a364e91b3fe6094623dcd9c7107c1f36f247cacbd81b`.
The final 18,619,413-byte download
`so12_all28cores_h0_hL_PCA_six_core_images.zip` includes PNG/PDF maps, overview,
legends, portable HTML, numerical tables and seed summaries; every ZIP member
matches its source and CRC validation passed. Heavy embeddings remain in the
full bundle. Earlier pending-preview receipts remain historical provenance,
and are excluded from the final download.

Verification and publication commands, from the repository root:

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python \
  scripts/analysis/audit_so12_hidden_leiden_v2.py \
  --output reports/analyses/so12_hidden_leiden/ev_so12_h0_hl_e151_20261017T005200Z
PYTHONPATH=src:scripts/analysis /venv/main/bin/python \
  scripts/analysis/run_so12_hidden_leiden.py \
  --evaluation-id ev_so12_h0_hl_e151_20261017T005200Z --finalize
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog --check
```

The finalize command was executed once against the active directory and must
not be rerun against this already published evaluation. The V2 auditor can
verify the moved bundle without rewriting its creation-time provenance paths.

The analysis supports a reproducible descriptive difference between h0 and hL,
with the stated seed sensitivity; biological validity remains inconclusive.
Five primary hL clusters contain over 90% cells from a single core (clusters
3, 10, 12, 17 and 21; dominant cores 23, 15, 15, 3 and 21). h0 has none under
this diagnostic threshold. These are observed composition differences, not
proof of biological specificity or a criterion for tuning. Joint all-core
clustering does not establish batch removal. Encoder effects remain confounded
with clustering-pipeline differences relative to PCA. A matched graph/metric
comparison and independent cell annotations would distinguish those effects.
The curated result is
`results/descriptive_analysis/relative_qkv/res_so12_hidden_states_leiden_comparison/`.
