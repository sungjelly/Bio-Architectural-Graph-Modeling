# SO1 14-core direct embedding clustering

## Status and objective

Phase: complete, verified, run-bound exploratory post-hoc analysis. Outcome:
success. Execution state: `complete_verified`. The upstream trainer stopped itself
at epoch 650 after two consecutive strict plateau audits, finalized run
`r_20260826T204925Z_06d12943_s000_f00_a01_94691119`, and produced a locked,
reload-verified `last.ckpt`. Full bundle, registry, catalog, checkpoint,
configuration, source, plateau, and input-manifest verification passed before
this downstream campaign was bound. Production analysis is CPU-only with CUDA
hidden.

The completed CPU-only analysis extracted two representations for every manifested cell
in SO1 cores 1 through 14 from one immutable Relative-Geometric QKV model:

- intrinsic `h0`: the output of the trained `NodeEncoder`, before graph layers;
- contextual `hL`: the output of the fourth and final graph layer, before the
  expression decoder.

Cluster `h0` and `hL` jointly across all 14 cores in two independent direct
embedding pipelines with no PCA. Also calculate `delta_h = hL - h0` and the raw
per-cell norm `delta_h_l2 = ||hL - h0||2`. Produce static tissue-coordinate
maps for the intrinsic clusters, contextual clusters, and delta norm, plus a
self-contained interactive HTML viewer for the contextual clusters only.

## Scientific question and claim limit

The descriptive question is whether direct clustering of the fitted model's
intrinsic and contextual representations produces different joint structure
across the 14 SO1 cores, and where the magnitude of representation change is
located in tissue coordinates. The working hypothesis is that graph processing
changes representation geometry enough for the independently constructed `hL`
partition to differ from the independently constructed `h0` partition. A
credible alternative is that `hL` largely preserves `h0`, or that either
partition mainly reflects expression depth, visible morphology/metadata,
segmentation, core-specific effects, or fitted-cohort technical structure.

The estimands are each cell's deterministic Leiden assignment on the direct
`h0` graph, its deterministic Leiden assignment on the direct `hL` graph, and
its raw `||hL - h0||2`. These are transductive, single-model descriptive
quantities. Intrinsic clusters describe patterns in the cell's own expression
and permitted metadata embedding. Contextual clusters describe patterns after
graph-based neighborhood processing. Delta norm is only a contextual
representation-change magnitude. None independently establishes cell type,
signaling, biological influence, mechanism, generalization, or causality.
Marker-based and pathological validation remain separate work.

## Upstream completion and immutable-binding gate

The sole upstream campaign is
`cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150`. The selected
run and checkpoint are deliberately unbound while training is active. A
mutable `scratch/active_runs/.../checkpoints/latest.ckpt` is never eligible.

Before any production extraction, clustering, or rendering against the SO1
run, require all of the following. Isolated downstream implementation and
synthetic unit testing may be prepared while training continues, provided the
non-interference boundary below is maintained:

1. the unique eligible seed-0 upstream attempt has registry status
   `completed` and an end time;
2. its finalized bundle exists under `artifacts/runs/YYYY/MM/<run_id>/` and has
   a valid `_SUCCESS` marker;
3. full run-bundle artifact verification passes;
4. the run has the locked-final SO1 study classification and exact intended
   scientific variant;
5. `checkpoints/last.ckpt` is a registered present artifact and its checkpoint
   catalog record has role `last`, verification status `verified`, and
   retention class `retain_locked_final_model`;
6. the final checkpoint reload receipt and terminal plateau receipt verify;
7. the checkpoint, model state, resolved configuration, run manifest, source
   commit, cohort manifest, graph manifest, prepared-core components, gene
   order, and all 14 graph artifacts are checksum-bound;
8. the resolved model has embedding/hidden width 256 and four graph layers;
9. no other completed attempt creates an ambiguous selection. If multiple
   eligible completed runs unexpectedly exist, stop instead of selecting by a
   favorable metric.

If the current attempt fails and one authorized retry completes, the unique
verified completed retry may be bound and every failed/excluded attempt must be
recorded. Raw weights or embeddings from different seeds or attempts must not
be averaged. Binding is a separate reviewed revision of this downstream
campaign; it must not modify the upstream campaign or run.

The binding record must contain the upstream campaign configuration checksum,
selected immutable run ID and preferred alias, scientific/reproduction IDs,
seed/fold/attempt, source commit, artifact root, run-manifest checksum,
resolved-config file and canonical-content checksums, `_SUCCESS` checksum,
artifact-checksum manifest, reload/plateau receipts, and checkpoint artifact
ID, checkpoint ID, role, epoch, path, SHA-256, size, model-state checksum,
monitored metric/value, retention class, and verification status.

## Non-interference boundary

While upstream training is active, this campaign may contain only its planning
files plus isolated downstream source code and synthetic tests. It must not:

- edit any upstream training campaign file, resolved configuration, source
  implementation used by the active process, active scratch bundle, log,
  metric, receipt, checkpoint, or finalized run path;
- create, update, reconcile, or otherwise mutate the upstream campaign, queue,
  run, artifact, checkpoint, or evaluation rows in the authoritative registry;
- signal, pause, stop, resume, restart, attach a debugger to, reprioritize, or
  replace the training process or its supervisor/worker;
- read, copy, load, hash, or analyze the mutable `latest.ckpt` as if it were a
  final checkpoint;
- install or change shared dependencies or environment settings used by the
  active run;
- reserve, query through a compute framework, or execute work on any GPU.

Production inference, clustering, and rendering start only after immutable
upstream verification. The post-hoc workflow is CPU-only with
`CUDA_VISIBLE_DEVICES=""`; it may neither displace training nor opportunistically
use an idle-looking GPU. `automatic_enqueue` remains false.

## Representation extraction contract

Use the exact transformation, gene order, permitted standardized 22-field
metadata, fixed spatial graph, relative-geometric encoding, cell order, and
core order stored in the finalized run and its immutable input manifests. Do
not refit preprocessing, add batch correction, or use core identity, vendor
annotations, clinical fields, stable identifiers, or expression-derived QC
totals as model inputs.

For each complete core, execute on CPU under `model.eval()` and
`torch.inference_mode()` with dropout disabled, attention dropout disabled, no
neighbor sampling, and an all-zero binary gene mask. Supply the complete
standardized expression vector and unchanged permitted metadata. Coordinates
are used only for the existing graph/relative geometry and subsequent maps;
they must not enter the `NodeEncoder`.

Use the backward-compatible intermediate-output path so that one forward pass
returns `node_encoder_embedding=h0`, `final_graph_embedding=hL`, and the normal
prediction. Verify predictions are numerically unchanged when intermediate
outputs are enabled and that `final_graph_embedding` equals the model's normal
final node embedding. Decoder activations are not `hL`.

For every cell save aligned core number, stable non-identifying row key or
index, `x_um`, `y_um`, `h0`, `hL`, `delta_h`, and raw `delta_h_l2`. Store large
arrays in compressed NPZ files and labels/coordinates in Parquet. Per-core
receipts must bind source inputs, checkpoint, array shapes/dtypes/checksums,
all-zero masks, unchanged metadata, prediction invariance, device, and elapsed
time. Extraction is resumable only from fully verified receipts.

Acceptance requires all 161,596 manifested cells and cores 1 through 14 in
identical order, no dropped or duplicated cells, finite coordinates and
embeddings, equal nonzero `h0`/`hL` dimensions, and exact recomputation of both
`delta_h` and `delta_h_l2` from the saved arrays.

## Direct no-PCA clustering contract

Construct two completely independent joint clustering pipelines:

```text
intrinsic:  direct 256-dimensional h0
            -> L2 normalization for cosine only
            -> sparse cosine 30-NN graph
            -> Leiden resolution 1.0

contextual: direct 256-dimensional hL
            -> L2 normalization for cosine only
            -> independently built sparse cosine 30-NN graph
            -> independently run Leiden resolution 1.0
```

Use random seed 20260825. There is no PCA, mean-centering, feature projection,
HVG selection, learned encoder, integration, or reuse of the other
representation's graph or labels. L2 normalization is only the standard
calculation of cosine similarity. The embedding-space graph is distinct from
the spatial training graph and may connect cells from different cores.

Use a deterministic scalable sparse neighbor implementation such as
single-threaded FAISS HNSW with a seed-derived insertion permutation that does
not consult core labels. Recompute the h0 and hL graphs independently while
using the same locked insertion permutation, so representation comparisons do
not inherit different approximate-index randomizations. Record HNSW parameters
and the permutation checksum, and require a deterministic sampled exact-recall
audit with a prespecified minimum mean recall@30 of 0.90. Never construct an
N-by-N distance matrix.

Sort labels by descending cluster size with minimum global row index and raw
cluster ID as stable tie-breaks. Use independent namespaces and palettes:
`S1I0`, `S1I1`, ... for intrinsic clusters and `S1C0`, `S1C1`, ... for
contextual clusters. Matching suffixes do not imply matching populations.
Report cluster counts and size ranges, proportions, core composition, graph
components, Leiden quality/modularity, and clusters with more than 90% of
their cells from one core. Retain all flagged clusters without integration,
merging, or removal.

Primary computational acceptance is finite aligned cell coverage equal to
1.0 and deterministic replay of both label arrays. Cluster number,
modularity, visual separation, and apparent biological plausibility are
descriptive, not optimization objectives.

## Static figures and delta norm

Create three combined PNG/PDF figure pairs and one PNG per core per figure
family under a run-specific report directory:

```text
intrinsic_direct_h0_leiden_resolution_1p0_spatial_14cores.png/.pdf
contextual_direct_hl_leiden_resolution_1p0_spatial_14cores.png/.pdf
delta_h_l2_spatial_14cores.png/.pdf
```

Combined figures use the repository's established `3 x 5` SO1 grid in exact
ascending row-major order, with cores 1--14 in the first fourteen panels and
the legend or shared colorbar in the final panel. Every tissue panel must say
`SO1 Core N`, use one borderless dot per cell, equal physical aspect,
repository coordinate orientation, no connecting lines, a reliable
micrometre scale bar, and a small rasterized point layer for PDF where
necessary.

Intrinsic and contextual palettes are deterministic, distinguishable beyond
20 clusters, and independent. A cluster's color is constant across all panels
of its own representation. The delta map uses raw, unclipped
`||h_contextual - h_intrinsic||2` values with one shared sequential color scale
across all cores. Plotting alone may clip to the global 1st and 99th
percentiles; record both thresholds and never rescale cores independently.

The final report must include per-core embedding files, independent clustering
graphs and receipts, aligned cell assignments, summary/core-composition
tables, palettes, static figures, stage manifests, code/input provenance, and
one self-checksummed final manifest. A plotting retry must reuse verified
extraction and clustering stages.

## Interactive contextual map

The contextual `hL` HTML deliverable is governed by `INTERACTIVE_MAP.md`. It
must reuse the finalized `S1C*` assignments and coordinates exactly and must
not repeat inference or clustering. No interactive `h0` or delta viewer is in
scope for this campaign.

## Stop criteria and interpretation

Stop without silent repair if the upstream model remains active or unverified,
any immutable checksum drifts, CPU-only isolation is not assured, any GPU is
accessed, preprocessing is refit, metadata changes, coordinates enter the
`NodeEncoder`, masks are nonzero, prediction invariance fails, cells or cores
are missing/reordered, embeddings or coordinates are nonfinite, zero-norm rows
prevent cosine analysis, either kNN graph is reused, PCA or mean-centering is
introduced, recall/determinism fails, an N-by-N array is attempted, or any
required artifact/checksum is absent.

The working hypothesis is descriptively weakened if the independently
constructed partitions are effectively indistinguishable or delta norms are
uniformly negligible. That is a valid result and is not permission to tune the
locked settings after inspection.

## Completed outputs and verification

Static outputs are under:

```text
reports/analyses/so1_14core_model_embedding_direct_knn_clustering/
  r_20260826T204925Z_06d12943_s000_f00_a01_94691119/
```

Interactive outputs are under:

```text
reports/analyses/so1_14core_hl_direct_knn_interactive/
  r_20260826T204925Z_06d12943_s000_f00_a01_94691119/
```

The exact repository entry points are:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  analyze-so1-model-embedding-clusters \
  --run-id r_20260826T204925Z_06d12943_s000_f00_a01_94691119 \
  --n-neighbors 30 \
  --leiden-resolution 1.0 \
  --random-seed 20260825 \
  --device cpu \
  --cpu-threads 40 \
  --dpi 300

CUDA_VISIBLE_DEVICES="" PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  render-so1-hl-direct-interactive \
  --run-id r_20260826T204925Z_06d12943_s000_f00_a01_94691119
```

The static manifest binds 95 files and passed the public verifier. It records
161,596 cells; h0 and hL shapes `[161596, 256]`; 17 intrinsic and 26 contextual
clusters; and raw, unclipped delta norms with shared plotting limits
10.85126167005577--23.448148354487014. All three combined maps passed visual
inspection. The interactive manifest binds the standalone HTML and its
byte-identical single-member ZIP; a second invocation passed checksum-only
replay, and `unzip -t` passed.

Final validation passed with 60 focused unit/smoke tests, repository doctor,
Python compilation, and `git diff --check`. Production inference and analysis
were CPU-only with CUDA hidden. No training process, upstream campaign/run,
checkpoint, finalized bundle, or GPU was modified or used by this analysis.
