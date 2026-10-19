# SO1 + SO2 hidden-state Leiden and cell-segmentation comparison

Phase: pilot. Outcome: pending. Exploratory, fully observed, transductive
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

Current versioned evaluation: `ev_so12_h0_hl_e151_20261017T005200Z`, managed by
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
events and per-worker logs. Final publication waits for numerical and visual
verification; training is not changed.

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
