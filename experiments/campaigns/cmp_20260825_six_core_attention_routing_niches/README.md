# Six-core attention-routing niche map

## Status and scope

Phase: pilot. Outcome: pending full four-model analysis.

This post-training analysis constructs a parameter-locked spatial partition from
the completed Relative-Geometric QKV Graph Transformer fits for Cancer cores
`1, 9, 13, 15, 21, 23`. It never trains, resumes, resets, or modifies a model.
Only catalog-verified `last.ckpt` files from successful immutable run bundles
are eligible. All eligible completed model seeds available when extraction
starts are included; an in-progress periodic checkpoint is never promoted by
this workflow.

The analysis is exploratory with respect to biology because the fitted cohort
and model behavior were already observed. Its primary computational parameters
were fixed before the niche maps were inspected. A visually attractive map is
not an acceptance criterion.

> These regions are model-defined attention-routing niches. They describe stable spatial patterns in how the trained model routes information during masked-expression reconstruction. They do not by themselves establish direct molecular signaling or biological causality.

## Task contract

### Objective and deliverables

Create one combined 2×3 map, six high-resolution per-core maps, an optional
strongest-edge overlay, cell/directed-edge/mutual-edge tables, dissolved region
geometry, deterministic colors, sensitivity summaries, a manifest, and an
auditable QC report. The canonical bundle is written first under
`scratch/active_runs/<run_id>/` and published only after verification to
`artifacts/runs/YYYY/MM/<run_id>/`.

### Scientific question, hypothesis, and alternatives

Question: does final-layer information routing define mutually above-uniform,
mask-stable, model-stable, and spatially connected regions within each fitted
core?

Primary hypothesis: reciprocal degree-adjusted final-layer routing contains
enough stable structure to yield deterministic connected components under the
locked retention and clustering parameters.

Credible alternatives are that apparent regions are driven by receiver degree,
cell density, distance, one-directional routing, analysis masks, one model
initialization, long graph edges that bridge empty tissue, segmentation
spillover, or broad spatial fields. The degree adjustment, reciprocal minimum,
mask/seed aggregation, all-genes-visible sensitivity, parameter sensitivity,
local connected-component split, and explicit uncertainty summaries address
some of these alternatives. They do not establish faithfulness, null
calibration, independent patient replication, signaling, or causality.

Predictions that distinguish these alternatives are: retained pairs exceed
uniform routing in both directions with support at least 0.60; attention sums
to one within each complete receiver neighborhood and head; the primary
partition is reproducible from the recorded seeds; final IDs are connected in
the independently constructed local graph; and seed/mask disagreement remains
visible rather than being hidden. Material instability at top-neighbor values
`5, 8, 10` or resolutions `0.5, 1.0, 1.5` weakens the routing-niche claim but
does not authorize changing the primary parameters.

### Estimand and maximum claim

The estimand is a within-core partition of cells induced by the median across
eligible model seeds and ten newly generated analysis-mask views of

`min(d_j * alpha[i->j], d_i * alpha[j->i])`

for reciprocal final-layer edges, followed by locked weighted Leiden
clustering and spatial connected-component splitting. The permitted claim is
only a stable model-defined attention-routing niche within the fitted cohort.
Attention is normalized computational routing, not biological importance.

### Units and uncertainty

Cells and directed/reciprocal graph edges are observational objects. The six
cores are the biological specimens. Model seeds and analysis masks are
computational perturbations, not independent biological replicates. Seed
spread is reported only when at least two completed models exist; otherwise
the output is labeled `single-model, mask-consensus map` and no seed
uncertainty is fabricated.

### Locked confidence definition

Confidence is a descriptive computational-stability quantity, not a posterior
probability.  For cell `i`, let `A_i` be its assignment agreement after
maximum-Jaccard matching of seed-specific partitions to the consensus when at
least two models are available.  With one model only, `A_i` is instead the
agreement of the ten single-view partitions with the mask-consensus partition;
the seed-agreement field remains unavailable rather than being set to perfect.
Let `P_i` be the median support fraction of retained mutual edges incident to
the cell (zero for a cell with no retained edge), and let
`V_i = 1 / (1 + sigma_seed_i + sigma_mask_i)`, where each sigma is the median
incident-edge standard deviation of the reciprocal routing score after
aggregation over the other computational axis.  The seed term is omitted,
not imputed, for a single-model analysis.  The locked cell confidence is

`confidence_i = (A_i * P_i * V_i) ** (1/3)`.

All three components and the seed/mask spreads remain separate columns so the
composite cannot hide why confidence is low.  Niche confidence is the median
cell confidence within that connected niche.  This definition was fixed
before rendering or inspecting the maps.

For the niche summary, `boundary_edge_score` is the median consensus mutual
score of retained edges with exactly one endpoint in the niche; it is missing
when no such retained edge exists and is never interpreted as signaling.
Descriptive marker summaries use the five largest within-niche minus
rest-of-core differences in mean `log1p(raw count)` and are explicitly labeled
post-hoc annotations, not validation or biological niche names.  Metadata
summaries are within-niche medians of the measured, allow-listed morphology and
intensity fields, also descriptive only.

### Locked primary analysis

- Cores and order: `CAN-01, CAN-09, CAN-13, CAN-15, CAN-21, CAN-23`.
- Analysis mask base seed: `2026082501`.
- Views: `10`, indexed `0..9`; every cell independently masks an exact count
  sampled from `Uniform{0,...,G}`, with positions uniform without replacement.
- Mask derivation: base seed + core alias + view index, never model seed.
- Model mode: `eval`, inference mode/no gradients, dropout disabled.
- Numerical mode: deterministic float32 inference with TF32 and autocast
  disabled; receiver chunking may change memory use but not neighborhoods.
- Layer: final graph layer immediately before the reconstruction decoder.
- Extraction: all edges, all heads, exact complete-receiver normalization; no
  neighbor sampling or edge truncation.
- Directed enrichment: receiver in-degree times head-mean attention.
- Directed export: exact degree-adjusted head-mean routing for every
  model-seed/mask-view/edge combination, all-visible routing per seed, and
  per-seed/per-head mask means for attention, content QK score, positional
  bias, and combined logit. Per-view/per-head channel tensors are not retained
  because they would add roughly 138 GB before encoding; this does not affect
  the primary head-averaged estimand.
- Reciprocal score: minimum of the two directed enrichments.
- Consensus score: median over all eligible seed/view combinations.
- Support: fraction of seed/view combinations with reciprocal score `>1.0`.
- Eligibility: consensus score `>1.0` and support `>=0.60`.
- Per-cell retention: top `8` eligible mutual edges, undirected union and
  deduplication.
- Community detection: weighted Leiden, resolution `1.0`, seed `2026082502`,
  independently within each core.
- Micro-niche threshold: fewer than `20` cells; never auto-merged.
- Primary region geometry: original segmentation polygons, aligned by the
  slide-qualified original cell key and dissolved within final niche. Polygon
  alignment accepts a prepared vendor coordinate when it is within `5 um` of
  the keyed polygon centroid or is covered by the valid keyed polygon; this
  avoids rejecting elongated cells whose vendor center is not their geometric
  centroid. Centroid exceptions and containment acceptances are recorded, and
  a coordinate that is both outside its polygon and beyond `5 um` remains a
  hard failure. If the polygon adjacency audit shows that segmentation
  boundaries do not encode usable cell adjacency, spatial connectedness uses
  Delaunay adjacency capped at `75 um`; the reason and coverage are recorded.
  Local `k=6` capped at `75 um` is the last fallback.
- Color seed: `2026082503`; categorical, deterministic, core-scoped, and
  adjacency-aware.

Sensitivity analyses are top-neighbor values `5, 8, 10`, Leiden resolutions
`0.5, 1.0, 1.5`, and an all-genes-visible replay. They cannot replace the
primary top-8/resolution-1.0 masked-consensus result.

### Controls, leakage, and limitations

Positive computational controls are strict checkpoint reload, per-receiver
attention normalization, reciprocal-pair construction, and deterministic
rerun checksums. Negative controls are rejection of one-directional pairs,
cross-core edges, smoke checkpoints, incomplete runs, checksum drift, and
disconnected components sharing an ID. The all-visible replay is a masking
sensitivity analysis, not an independent null.

The exact training normalization, metadata transform, ordered genes, prepared
node order, graph, and relative geometry are reloaded and checksum verified;
nothing is refit. Cell type/vendor annotations are descriptive only and never
become model inputs. No hidden-expression-derived label is used in masking or
graph construction. Because this is a transductive fit-only cohort, it does not
support patient-generalization or biological-mechanism claims.

### Acceptance, falsification, and stop criteria

Acceptance requires all requested static artifacts; exact six-core coverage;
one assignment per eligible prepared cell; successful strict loading of every
eligible final checkpoint; unchanged checkpoint/input checksums; zero
cross-core edges; aligned edge diagnostics; per-receiver/head attention sums
within numerical tolerance; correct reciprocal/degree calculations; connected
final niches; deterministic colors; micrometre coordinates and scale bars;
and identical primary assignment/checksums on a deterministic verification
rerun. The production check regenerates and re-extracts all ten mask views for
one fixed reference model, requires exact routing-array equality, verifies the
same mask receipts across every model, and independently rebuilds the complete
consensus graph, assignment, and color mapping from the retained routing
arrays.

The primary hypothesis is weakened or computationally falsified if no retained
mutual graph remains, if seed/mask support is poor, or if spatial splitting
reduces communities largely to isolated micro-components. Such a result is
reported without relaxing thresholds. Extraction stops before publication on
checkpoint/input drift, architecture mismatch, irreconcilable node order,
attention normalization failure, insufficient disk/VRAM safety margin, or an
incomplete required output. Exact evidence and the needed external state
change must then be recorded.

### Inputs, compute, and expected artifacts

Immutable upstream inputs are the verified completed run bundles for
`cmp_20260824_cancer_6core_relative_qkv_multiseed`,
`data/processed/cancer_6core_relative_qkv_v1`,
`data/processed/cancer_6core_relative_qkv_graphs_v1`, and the source
segmentation polygons used only after slide-qualified alignment. One complete
core is staged per GPU at a time; all model seeds for that core use the same
locked masks in sequence, while independent cores are scheduled across the
safely available GPUs. GPU model, IDs, CUDA/PyTorch versions, runtime, and peak
memory are recorded. The run requires at least 40 GiB free
at launch, records a seed-aware table-size estimate and sampled filesystem
high-water receipts, streams each core into the canonical Parquet files, and
removes a verified staging shard immediately after it is appended.

Required bundle paths are:

```text
six_core_attention_niche_map.{png,pdf,svg}
six_core_mutual_attention_network_overlay.{png,pdf}
core_{01,09,13,15,21,23}_attention_niche_map.png
cell_attention_niche_assignments.parquet
mutual_attention_edges.parquet
directed_attention_edges.parquet
attention_niche_summary.csv
attention_niche_colors.json
attention_niche_regions.geojson
analysis_manifest.yaml
analysis_qc_report.md
README.md
```

The manifest may additionally index receiver-sharded directed diagnostics and
sensitivity tables when retaining every per-head/per-view scalar in one flat
file would exceed the recorded disk budget. Direction is never discarded from
the canonical directed export.

## Pilot verification

A full-resolution technical pilot on Core 21 used the completed seed-0 final
checkpoint and all ten locked analysis masks. This was not used to change any
primary threshold. It covered 4,897 cells, 1,137,942 directed edges, and
568,971 reciprocal pairs; retained 28,169 mutual edges; found 31 preliminary
Leiden communities and 883 final spatially connected components, of which 833
contained fewer than 20 cells. Median single-model mask-agreement confidence
was 0.9136. The all-genes-visible partition had adjusted Rand index 0.7860 and
maximum-Jaccard cell agreement 0.8569 with the masked consensus. The largest
attention normalization error was `1.55e-6`; final-layer inference and the
downstream assignment replay were exactly deterministic.

The high number of spatially split micro-components is a substantive pilot
finding under exact segmentation-polygon adjacency, not a reason to relax the
locked contiguity or clustering parameters. The temporary 211 MB pilot staging
directory was removed after recording this receipt because it was not a
registered conclusion-bearing run.

The first registered full attempt,
`r_20260825T051807Z_825125bb_s000_f00_a01_058e2607`, failed before attention
extraction for Core 13. One valid, unrepaired, elongated polygon covered its
prepared coordinate but had a geometric centroid distance of `6.798546 um`,
exceeding the original centroid-only `5 um` gate. Counts, keys, prepared
coordinates, expression/metadata replay, and graph checks had all passed. No
result map was available or inspected. The source cell key is intentionally
omitted from this tracked report.
The gate was changed before the next attempt to the containment-aware rule
above; it does not alter a trained checkpoint, prepared coordinate, graph, or
locked niche-analysis threshold. The failed bundle and marker remain in the
immutable run archive.

The replacement full attempt,
`r_20260825T053454Z_0da9fbf2_s000_f00_a01_69a07db0`, passed Core 13 input and
attention extraction and wrote its directed, mutual, assignment, summary,
sensitivity, color, and region shards. It then failed before writing the core
receipt because receipt construction attempted to hash the all-visible routing
array after that large array had deliberately been released. The receipt now
reuses the identical SHA-256 already computed and stored by the extraction
audit before release. This lifecycle-only repair changes neither the computed
routing values nor any scientific parameter. The incomplete attempt remains a
failed immutable bundle and was not used as a scientific result.

The subsequent full attempt,
`r_20260825T055100Z_0da9fbf2_s000_f00_a01_ead1d2c2`, completed all six cores,
canonical consolidation, sensitivity analysis, summaries, colors, and region
geometry. It then failed before figures and final QC because the renderer
incorrectly treated the composite `assignment_confidence` and the distinct
`niche_assignment_agreement` diagnostic as aliases. Only 0.722% of assignment
rows have numerically equal values in those two fields, so they must remain
separate; the renderer now consumes the canonical composite confidence.

This failed bundle is immutable and checksum-bound by `_FAILED` content digest
`99b34e17d2fd1e0d4c52ef564b7ea119242e0fa332f5b40e6a3ee6de008e7b3f`.
The later registry-audited retention decision
`cleanup_20260825_relative_qkv_final_only_v2` tombstoned its failed-run directed
and mutual Parquet files. The remaining receipts retain their checksums and the
audit result, but the missing values cannot support a continuation. A fresh
registered full run from the unchanged catalog-verified checkpoints is
therefore required; the tombstones are not bypassed or reconstructed from
summaries.

An independent post-failure audit also found that six-decimal GeoJSON rounding
made 234 of 10,063 serialized region features invalid even though the dissolved
pre-serialization geometries were valid. GeoJSON generation now validates the
serialized coordinates, applies `make_valid` only when needed, retains
Polygon/MultiPolygon parts, and requires every feature to be valid and nonempty.
Maximum area change in the failed output audit was `1.12e-6 um2`, and repaired
geometry remained within `1e-4 um2` of the locked summary area. This
serialization repair does not change niche membership or any routing result.

## Verification and full commands

Focused tests:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_attention_routing_niches.py \
  tests/unit/spatial_benchmark/test_attention_niche_geometry.py \
  tests/unit/spatial_benchmark/test_attention_niche_visualization.py \
  tests/unit/spatial_benchmark/test_attention_niche_pipeline.py
```

The full registered execution uses the four catalog-verified completed model
seeds available at launch and the normal worker-owned run scratch directory:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 enqueue-experiment \
  --campaign-id cmp_20260825_six_core_attention_routing_niches \
  --config experiments/campaigns/cmp_20260825_six_core_attention_routing_niches/analysis_config.yaml \
  --priority 0 --max-attempts 1 --gpu 0,2,3

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 worker \
  --worker-id attention-niche-multigpu --gpu 0,2,3 --once
```
