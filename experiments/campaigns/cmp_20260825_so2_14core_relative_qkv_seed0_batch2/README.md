# SO2 cores 15–28 Relative-Geometric QKV training

## Task contract

Baseline phase: completed on 2026-08-26 at epoch 175 under the original
plateau contract. Additive fixed epoch-300 continuation: configured, pending.

The completed baseline run and its epoch-175 checkpoint remain immutable. A
later user-authorized successor resumes at human epoch 176 and trains through
the fixed final epoch 300 under a checksum-bound amendment. Its stricter
plateau calculation is explicitly post-hoc and exploratory and cannot change
training duration or checkpoint choice. See
[Fixed epoch-176-to-300 continuation](CONTINUATION_EPOCH176_TO300.md).

This campaign fits one shared Relative-Geometric QKV Graph Transformer (model
seed 0) across the fourteen disconnected SO2 tissue-core graphs numbered 15
through 28. It is a fit-only, transductive masked-expression experiment: every
eligible cell is used for fitting, there is no validation/test split, no
generalization claim, and checkpoint duration is governed only by a
prespecified training-loss plateau audit after a minimum of 150 global epochs.

The cohort is defined by core routing, not by a single disease label. Existing
reviewed evidence is mixed and the prior Cancer rediagnosis attestation covers
only cores 15, 21, and 23 within this range. Tissue diagnosis, core identity,
slide, FOV, donor, and cell identifiers are never model inputs. The maximum
defensible claim is masked-expression reconstruction behavior within these
fourteen fitted cores.

The primary hypothesis is that the unchanged relative-geometric QKV model can
learn a stable decreasing masked-Huber training objective when shared across
all fourteen cores. A credible alternative is that apparent improvement is
dominated by same-cell co-expression/morphology and that core heterogeneity
causes noisy or non-convergent optimization. The primary operational metric is
the equal-core arithmetic mean of the ten-view masked Huber loss; lower is
better. Per-core losses and gradient norms are retained so instability cannot
be hidden by the mean.

The locked training unit is a complete core graph. Each global epoch visits all
fourteen cores exactly once in deterministic shuffled order. Consecutive cores
form seven deterministic gradient-accumulation pairs. For each pair, ten
independent exact 0–100% masks are run per core and every loss contributes
`loss / 20`; gradients are clipped and AdamW steps once after both cores. Only
one graph is resident on each GPU at a time. The production execution uses four
DDP ranks: ranks 0 and 1 receive views 0–4 and 5–9 of the first paired core,
while ranks 2 and 3 receive the corresponding views of the second paired core.
Each rank averages five local view losses and DDP averages the four rank
gradients, which is exactly the arithmetic mean of all 20 core-view losses.
This preserves equal-core weighting and yields 20 complete-graph mask-view
gradients per optimizer update without neighbor sampling or cross-core message
passing.

The architecture, graph policy, masking distribution, optimizer, learning
rate, and permitted node inputs match the six-core Relative-QKV campaign:
width 256, four graph layers, eight 32-dimensional heads, 1,024-wide FFN and
decoder, geometry as a per-head logit bias only, radial-stratified nominal
`k=200` within 500 µm, ten mask views per cell/core/epoch, AdamW at `1e-4`, and
gradient clipping at 1.0.

Acceptance requires:

- exactly cores 15–28 and 246,063 mapped cells, with SO2 FOV 246 explicitly
  excluded as unmapped;
- one identical ordered 1,000-gene biological panel and permitted metadata
  schema across all cores;
- graph coverage/QC, zero cross-core/self edges, and checksum-bound artifacts;
- exactly fourteen core visits, 140 mask-view forwards/backwards, and seven
  optimizer updates per global epoch;
- four-rank DDP gradient-mean equivalence, deterministic order/masks, one-core
  peak staging per GPU, and deterministic epoch-boundary resume;
- an fsynced `results/epoch_metrics.csv` row after every completed epoch;
- atomic rolling checkpoint replacement with only the newest resumable
  checkpoint retained during training and one final `last.ckpt` at completion;
- largest-core AMP/FP32, finite-gradient, VRAM, and save/load preflight;
- an initial and minimum budget of 150 epochs—not a maximum—followed by
  25-epoch continuation blocks until two consecutive prespecified plateau
  audits pass; there is no scientific maximum epoch cap.

Failure criteria include schema/routing drift, failed graph coverage, numerical
non-equivalence, non-finite loss/gradient, peak-memory failure that cannot be
resolved by exact receiver chunking, missing live metrics, non-deterministic
resume, or insufficient disk safety margin. A plateau is convergence evidence
for the fitted training objective, not validation or biological causality.

## Fit cohort and node-input policy

The immutable cohort aliases are `SO2-C15` through `SO2-C28`. Each alias maps
to one disconnected complete-core graph, and every eligible mapped cell has
the single role `fit`. The role contract is:

```text
fit_scope: all_cells_so2_cores_15_through_28_transductive
validation_or_test_partition_present: false
generalization_claim_supported: false
```

Raw counts are the source expression representation. Technical probes
(`Negative*`, `SystemControl*`, and other controls) are excluded, leaving one
identically ordered 1,000-gene biological panel across all cores. The target is
gene-wise standardized `log1p(count)`. One shared mean and scale are fitted
with equal-core weighting, so a larger core does not dominate preprocessing.
The gene order, transformation parameters, schemas, and source/preprocessing
checksums are persisted with the prepared cohort. Library size and RNA-derived
QC totals are not covariates.

For cell `i`, the existing `NodeEncoder` receives standardized expression
`x_i`, the explicit binary gene mask `m_i`, and permitted morphology/imaging
metadata `c_i`. It reapplies the mask internally and uses three separate
learned projections:

\[
\widetilde{x}_i=x_i\odot(1-m_i),\qquad
h_i^{(0)}=
\operatorname{Dropout}\!\left(
\operatorname{GELU}\!\left(
\operatorname{LayerNorm}\!\left[
W_x\widetilde{x}_i+W_m m_i+W_c c_i+b
\right]\right)\right).
\]

Metadata is always visible and is never masked. Permitted fields are the
locked morphology/imaging measurements: area and area in µm², aspect ratio,
width, height, PanCK/G/Membrane/CD45/DAPI mean and maximum intensities,
split ratio to local, nuclear area and aspect ratio, circularity,
eccentricity, perimeter, and solidity. The established imputation,
transformation, missing-indicator, and standardization contract is reused.

The node token never contains raw or local coordinates, core number/alias,
slide or FOV identity, cell or patient identity, vendor cell type/cluster/
neighborhood/niche, RNA-derived totals, or target-derived annotations. Patient
identifiers are also excluded from ordinary interpretation exports. The binary
mask prevents the encoder from confusing an observed zero with a hidden gene
and prevents masked target values from leaking through the expression path.

## Graph construction and coverage QC

Each core is constructed independently in its audited physical coordinate
system. There are no cross-core edges, no core-identity embeddings, and no
implicit or explicit self-loops; self information is carried only by residual
connections. Full-core coordinates may connect spatially continuous adjacent
vendor FOVs according to the repository's audited full-core convention.

The deterministic pre-symmetrization policy is
`radial_stratified_knn`, with maximum range 500 µm and these per-receiver
quotas:

| Radial shell | Nearest sources retained |
| --- | ---: |
| 0–50 µm | 48 |
| 50–150 µm | 64 |
| 150–300 µm | 48 |
| 300–500 µm | 40 |
| **Nominal total** | **200** |

Unused shell capacity is filled by the nearest remaining cells from other
shells without exceeding 500 µm. Ties are deterministic. The reverse of every
retained relationship is added, then directed edges are deduplicated,
self-edges removed, canonically sorted, and checksummed. Consequently, final
in-degree may exceed the nominal `k=200`. No neighbors are sampled or truncated
during training, and exact receiver-wise softmax always sees every incoming
edge.

The graph collection completed under manifest content checksum
`e41a4c92868bac96d05984b5a070d9984cda6ac04e3ad30458e5920259c573d5`.
The distance column reports p05/p50/p95/p99 in micrometers. Values below are
copied from that immutable manifest rather than estimated from the nominal
policy.

| Core alias | Cells | Directed edges | In-degree min/mean/median/max | Edge-distance quantiles | Eligible cells with a 300–500 µm edge | Duplicates removed / self removed / final cross-core | Graph SHA-256 |
| --- | ---: | ---: | --- | --- | ---: | --- | --- |
| SO2-C15 | 38,145 | 8,607,462 | 7/225.7/225/288 | 18.7/73.9/303.0/305.8 | 38,145/38,145 (100%) | 6,650,152 / 38,145 / 0 | `47da2e3d0a3c28c50dd331748ddb726c441687b7c6b1bb68d2ce67483bf4550b` |
| SO2-C16 | 16,229 | 3,671,048 | 200/226.2/226/335 | 20.3/76.6/303.5/306.1 | 16,229/16,229 (100%) | 2,820,552 / 16,229 / 0 | `bfb2f9098eaf7c5211bd9673d3572c77e4d574cc7aedfb2ab7243069bfc09616` |
| SO2-C17 | 9,041 | 2,070,630 | 44/229.0/229/301 | 21.2/84.1/304.2/307.6 | 9,041/9,041 (100%) | 1,543,574 / 9,041 / 0 | `801814a4295c8ffb98e553f44359b52f962b531a2efcde4e2db252023ed62e82` |
| SO2-C18 | 11,696 | 2,674,636 | 8/228.7/228/303 | 21.4/87.4/304.1/308.6 | 11,696/11,696 (100%) | 2,003,226 / 11,696 / 0 | `ada86a21f4727bafd79bcf7e2577c1bdb7ac02ee083ee9ee11205adc0b47abed` |
| SO2-C19 | 19,119 | 4,403,562 | 31/230.3/229/320 | 20.1/85.0/303.9/307.7 | 19,119/19,119 (100%) | 3,243,444 / 19,119 / 0 | `90046fa5bb34d140fda43eead9a7163dfa6dccf0e57c7c6b9b014315cc898888` |
| SO2-C20 | 12,155 | 2,746,180 | 22/225.9/226/278 | 21.9/83.9/304.1/307.2 | 12,155/12,155 (100%) | 2,107,802 / 12,155 / 0 | `dc59307864177722c5570238176c2686eae2a7fd6bb009e84e3c165cdc34dc2b` |
| SO2-C21 | 4,897 | 1,137,942 | 200/232.4/232/304 | 22.6/95.4/308.1/325.5 | 4,897/4,897 (100%) | 820,858 / 4,897 / 0 | `8ec23d683ff445afcb342c4e6f0168fe628b915623e99a350a00cee43cbe2eaa` |
| SO2-C22 | 11,443 | 2,598,742 | 61/227.1/227/327 | 23.6/91.9/304.8/309.3 | 11,443/11,443 (100%) | 1,978,180 / 11,443 / 0 | `3cdbcf44651950a236c80e16d6031ed73597b55f3b5f3128a1e099cd50ebb7b6` |
| SO2-C23 | 43,462 | 9,972,300 | 58/229.4/228/326 | 15.7/67.7/302.6/305.9 | 43,462/43,462 (100%) | 7,412,064 / 43,462 / 0 | `4d38c97e7dcab83f3d4b2cf6c0f94ecdd7e73042f3baea2decc697983a04578b` |
| SO2-C24 | 14,506 | 3,281,446 | 44/226.2/227/274 | 18.9/71.7/303.3/305.2 | 14,506/14,506 (100%) | 2,510,560 / 14,506 / 0 | `9ad2250c61b31912c0cfd84246be0c7283de730fedf47520af3c7107061abe33` |
| SO2-C25 | 13,147 | 3,049,388 | 200/231.9/231/318 | 23.2/103.0/305.7/310.7 | 13,147/13,147 (100%) | 2,209,412 / 13,147 / 0 | `5301b437c1813579f85c3bafa45b847d817d33de47dc1386104ac33ba59b634d` |
| SO2-C26 | 14,856 | 3,290,872 | 200/221.5/221/283 | 23.6/80.5/304.0/306.5 | 14,856/14,856 (100%) | 2,651,528 / 14,856 / 0 | `4caf6aa24f6e67ba985cdbb4583f244de69fa9e9e9396bf1a086d5c1476e3f13` |
| SO2-C27 | 24,245 | 5,558,646 | 200/229.3/228/301 | 19.7/81.6/303.6/307.7 | 24,245/24,245 (100%) | 4,139,354 / 24,245 / 0 | `0344e36234b96b21e83cfd82efe55198c16383fa752e740fd02a5545e350fef2` |
| SO2-C28 | 13,122 | 2,917,682 | 13/222.4/221/289 | 25.2/86.6/304.9/308.3 | 13,122/13,122 (100%) | 2,328,966 / 13,122 / 0 | `d2aa02a24edfe203cb4e398e22b1d279a09371c4b5adfe40d2f3d3671253eb87` |

Coverage eligibility means that another physical cell exists in the
receiver's 300–500 µm annulus. Every eligible receiver is expected to retain at
least one edge in that shell; systematic violation fails preparation. The
manifest must also report zero final self-edges and zero cross-core edges.

## Transformation-invariant relative geometry

Coordinates are used only to deterministically construct graph relationships,
local orientation tensors, and pairwise relative positional encodings. For an
edge `j → i`,

\[
\Delta p_{ij}=p_j-p_i,\qquad
r_{ij}=\lVert\Delta p_{ij}\rVert,\qquad
\widehat d_{ij}=\frac{\Delta p_{ij}}{r_{ij}+\epsilon}.
\]

Each cell's coordinate-only local covariance uses up to the nearest 64 cells
within approximately 150 µm and Gaussian weights with 75 µm scale. It is
converted to the normalized traceless tensor and anisotropy

\[
T_i=\frac{C_i}{\operatorname{tr}(C_i)+\epsilon}-\frac12 I,
\qquad
a_i=\sqrt{2\operatorname{tr}(T_i^2)}.
\]

Cells without enough valid neighbors receive `T_i = 0` and `a_i = 0`, and their
count is recorded. Distance is encoded with 64 overlapping Gaussian radial
basis functions spanning 0–500 µm and a smooth cutoff envelope approaching
zero at 500 µm. Distances outside the graph contract are rejected rather than
silently clipped. The complete 70-dimensional pair vector is

\[
\rho_{ij}=\left[
\operatorname{RBF}_{1:64}(r_{ij}),
\widehat d_{ij}^{\mathsf T}T_i\widehat d_{ij},
\widehat d_{ij}^{\mathsf T}T_j\widehat d_{ij},
\operatorname{tr}(T_iT_j),a_i,a_j,r_{ij}/500
\right].
\]

These scalar contractions are invariant when the whole core is consistently
translated, rotated, or reflected. The orientation is axial: it can represent
parallel-versus-perpendicular preference but cannot distinguish opposite
directions along the same axis. No front/back polarity is claimed or fabricated
without an independently measured directed biological signal. Raw displacement
vectors and global angles never enter node, Q/K/V, decoder, or value content.

## Relative-Geometric QKV architecture

The model has hidden width 256, four graph Transformer blocks, eight heads of
dimension 32, a 1,024-wide FFN, dropout 0.10, zero attention dropout, and
activation checkpointing. Each pre-normalized cell state determines separate
queries, keys, and values:

\[
q_{ih}=W_{Q,h}\operatorname{LN}(h_i),\quad
k_{jh}=W_{K,h}\operatorname{LN}(h_j),\quad
v_{jh}=W_{V,h}\operatorname{LN}(h_j).
\]

For each layer, a positional-bias network maps
`70 → 128 → LayerNorm → GELU → 8`; its final projection is zero-initialized.
Relative geometry changes only the attention logit:

\[
s_{ijh}^{\mathrm{content}}=\frac{q_{ih}^{\mathsf T}k_{jh}}{\sqrt{32}},
\qquad
s_{ijh}=s_{ijh}^{\mathrm{content}}+
b_{ijh}^{\mathrm{position}},
\qquad
b_{ijh}^{\mathrm{position}}=f_h(\rho_{ij}).
\]

For each receiver and head, exact sparse attention normalizes jointly over all
incoming sources and aggregates sender values:

\[
\alpha_{ijh}=\operatorname{softmax}_{j\in\mathcal N(i)}(s_{ijh}),
\qquad
z_{ih}=\sum_{j\in\mathcal N(i)}\alpha_{ijh}v_{jh}.
\]

Geometry therefore controls routing, while cell state controls message
content. There are no ordinary edge attributes, edge-key/value projections,
value gates, position-conditioned values, or `SharedEdgeEncoder`. Concatenated
heads pass through the output projection and residual connection, followed by
pre-LayerNorm `256 → 1024 → 256` FFN and a second residual. Receiver chunking is
exact: a receiver's incoming normalization is never split, approximated,
sampled, or truncated, and mixed-precision logits, softmax, messages, and
reductions use FP32 accumulation.

After four blocks, the existing expression decoder maps
`256 → 1024 → 1000` and predicts every biological gene for every cell.

## Masking, objective, and fixed training role

For every core, epoch, view, and cell, the masked-gene count is independently
drawn from the full discrete distribution

\[
m_i\sim\operatorname{Uniform}\{0,1,\ldots,1000\},
\]

then exactly `m_i` distinct gene positions are sampled uniformly without
replacement. Zero-percent and 100-percent masking are both valid, views are
not ratio-binned, neighboring and receiver cells follow the same rule, and
metadata remains visible. Mask seeds depend on base mask seed, core alias,
global epoch, and view index—not model seed. Every cell receives ten independent
views per epoch.

For mask indicator `M_ig = 1` at hidden positions, each view uses FP32 masked
Huber loss with delta 1.0:

\[
\mathcal L=
\frac{\sum_{i,g}M_{ig}\operatorname{Huber}(\widehat x_{ig}-x_{ig})}
{\sum_{i,g}M_{ig}}.
\]

A whole-view zero mask is deterministically resampled, so loss never divides
by zero and never creates a silent gradient-free step. AdamW uses learning rate
`1e-4`, weight decay `1e-5`, no scheduler, and gradient clipping at norm 1.0.
There is no early stopping based on validation, no restore-best behavior, and
no best-epoch selection. The prespecified plateau rule observes only held-in
training loss after the 150-epoch minimum and determines whether another fixed
25-epoch training block is required. It is not a validation metric.

## Interpretation readiness and limitations

The trained checkpoint supports receiver-chunked, selected-layer explanation
extraction with edge-aligned source/receiver indices, per-head attention,
content logits, positional biases, combined logits, embeddings, and relative
geometry summaries. Post-training sharded cell–cell exports may include source
and receiver coordinates solely for plotting; those coordinates were never raw
model inputs. Reciprocal, degree-adjusted summaries are descriptive mutual
attention-routing scores, not direct interaction measurements.

Selected-cell, selected-edge, and selected-gene analysis hooks can evaluate
attention gradients and source-gene-to-target-gene prediction Jacobians without
materializing an exhaustive edge × source-gene × target-gene tensor. Attention
alone does not identify which source RNA produced a target RNA prediction, and
neither attention, gradients, Jacobians, embeddings, nor positional preference
establishes signaling direction or biological causality.

Key limitations are the transductive all-fit role, mixed and incompletely
attested tissue context across cores 15–28, an axial orientation representation
without polarity, dependence on the measured panel and morphology channels,
and seed uncertainty from training only one model. Held-in diagnostics describe
reconstruction on fitted cells and cannot support out-of-sample performance or
clinical generalization claims.

## Live metrics and latest-only checkpoint semantics

Rank zero appends one flushed and fsynced row to
`results/epoch_metrics.csv` after every completed global epoch. The row records
the equal-core training loss, all fourteen per-core losses, mean/max paired
gradient norms before clipping, learning rate, epoch and cumulative optimizer
updates, complete-graph view and masked-entry counts, epoch duration and ETA,
maximum peak VRAM across the four ranks, and the prespecified training-loss
plateau audit. The CSV and structured metric event are written before the epoch
checkpoint. On resume, any CSV row newer than the durable checkpoint is
atomically trimmed, so monitoring never changes model selection or training
duration.

During training, `checkpoints/latest.ckpt` is atomically replaced at every
epoch boundary; prior checkpoint bytes are removed after the replacement is
durable. It includes model, optimizer, AMP scaler, completed epoch, cumulative
histories, model/mask/order derivations, cohort/core/update contracts, and
checksums sufficient for deterministic continuation. A successful terminal
bundle contains only independently reloadable `checkpoints/last.ckpt`. The
latest/final checkpoint is always the last completed epoch, never a
metric-selected epoch.

## Planned artifacts and commands

The campaign will use additive configs and source modules; it must not mutate
the frozen six-core campaign or its checkpoint schema. Prepared arrays and
graphs belong under `data/processed/so2_14core_relative_qkv_v1/` and
`data/processed/so2_14core_relative_qkv_graphs_v1/`. Active run output belongs
under `scratch/active_runs/<run_id>/`, then the verified bundle is published to
`artifacts/runs/YYYY/MM/<run_id>/` through the normal worker.

The source and runtime trees are intentionally separate. Run the following
from the clean source worktree; all large data, registry, scratch, and artifact
paths remain under `/workspace/BAGM`:

```bash
cd /workspace/BAGM-relative-qkv-analysis
export BAGM_ROOT=/workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export BAGM_CACHE_ROOT=/workspace/BAGM/cache
export BAGM_EXPORT_ROOT=/workspace/BAGM/exports
export BAGM_REPORT_ROOT=/workspace/BAGM/reports
export PYTHONPATH=/workspace/BAGM-relative-qkv-analysis/src
```

Preparation is idempotently gated on each immutable manifest:

```bash
test -f "$BAGM_DATA_ROOT/processed/so2_14core_relative_qkv_v1/manifest.json" || \
  /venv/main/bin/python -u scripts/data/prepare_so2_14core_relative_qkv.py
test -f "$BAGM_DATA_ROOT/processed/so2_14core_relative_qkv_graphs_v1/manifest.json" || \
  /venv/main/bin/python -u scripts/data/materialize_so2_14core_relative_graphs.py
```

After GPUs 0--3 are idle, run the bounded four-rank preflight. It performs one
real paired-core optimizer update (20 complete-graph mask views), checks finite
loss/gradients and checkpoint reload, reports maximum rank VRAM, and binds the
receipt to the resolved config plus cohort/graph manifests. It is a diagnostic,
not a completed experiment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /venv/main/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  scripts/diagnostics/preflight_so2_14core_relative_qkv_ddp.py \
  --config configs/experiment/so2_14core_relative_qkv_seed0_batch2.yaml \
  --prior-c23-receipt \
  /workspace/BAGM/artifacts/runs/2026/08/r_20260824T121803Z_16144620_s000_f00_a02_62498796/diagnostics/hardware_preflight.json
```

Register the immutable dataset/split/campaign with the completed cohort and
graph checksums locked below:

The dataset fingerprint is the prepared cohort manifest's internal
`manifest_content_sha256`, which excludes its self-hash field. The split
fingerprint is SHA-256 of compact, sorted-key JSON over exactly the six fields
recorded in `dataset.split_fingerprint_basis`: ordered aliases, dataset ID,
fit scope, total cell count, the false partition-presence flag, and dataset
version. It therefore commits to the complete all-fit role assignment without
pretending that a validation/test partition exists. The graph manifest file
checksum is
`c4a632473e458ae6e6eeab92c026ebf6e4d18ae49b568723f0e66a67db4fc90f`;
its canonical content checksum is
`e41a4c92868bac96d05984b5a070d9984cda6ac04e3ad30458e5920259c573d5`.
The completed cohort-with-graphs manifest checksum is
`d7e41ab22cf5ba340775160daf403205783c522d4ce573dda0d1fe2323c58b51`.

```bash
/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" register-dataset \
  --dataset-id cosmx_so2_14core_pooled_fit_v1 \
  --version so2_14core_pooled_fit_v1 \
  --display-name "CosMx SO2 cores 15-28 pooled transductive fit" \
  --protected-source-path "$BAGM_DATA_ROOT/processed/so2_14core_relative_qkv_v1/manifest.json" \
  --raw-fingerprint c7abbeddd8ed018d09163bb290224d3a92fd38ab3f3696d012b731f2f05f9bf0 \
  --preprocessing-version so2_14core_equal_core_log1p_metadata_v1 \
  --processed-fingerprint e41a4c92868bac96d05984b5a070d9984cda6ac04e3ad30458e5920259c573d5 \
  --sample-count 246063 --graph-count 14 \
  --node-feature-schema expression_mask_permitted_metadata_v1 \
  --edge-feature-schema relative_geometry_logit_bias_70d_v1 \
  --status available --verification-status verified

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" register-split \
  --split-id fit_all_so2_cores_15_through_28_transductive_v1 \
  --dataset-id cosmx_so2_14core_pooled_fit_v1 \
  --dataset-version so2_14core_pooled_fit_v1 \
  --method all_cells_fit_only_transductive --unit spatial_core \
  --fold-count 1 \
  --fingerprint 385f27e29cc8e598eb6d6545dbc5d01337ed30661254e5848b88876aa9771abd \
  --protected-path "$BAGM_DATA_ROOT/processed/so2_14core_relative_qkv_graphs_v1/cohort_manifest_with_graphs.json" \
  --verification-status verified

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" create-campaign \
  --campaign-id cmp_20260825_so2_14core_relative_qkv_seed0_batch2 \
  --name "SO2 cores 15-28 Relative-Geometric QKV seed-0 fit" \
  --scientific-question "Does the shared relative-QKV model reach a stable fitted training-loss plateau across SO2 cores 15 through 28?" \
  --plan experiments/campaigns/cmp_20260825_so2_14core_relative_qkv_seed0_batch2/campaign.yaml \
  --status planned
```

Enqueue exactly one four-GPU registry job. The derived child command is
`torch.distributed.run --nproc-per-node=4 --max-restarts=0`; do not replace it
with four independent workers or a loose shell process:

```bash
/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" enqueue-experiment \
  --campaign-id cmp_20260825_so2_14core_relative_qkv_seed0_batch2 \
  --config configs/experiment/so2_14core_relative_qkv_seed0_batch2.yaml \
  --priority 100 --max-attempts 2 --gpu 0,1,2,3

install -m 0755 ops/supervisor/bagm-so2-14core-ddp4-worker.sh \
  /opt/supervisor-scripts/bagm-so2-14core-ddp4-worker.sh
install -m 0644 ops/supervisor/bagm-so2-14core-ddp4-worker.conf \
  /etc/supervisor/conf.d/bagm-so2-14core-ddp4-worker.conf
supervisorctl reread
supervisorctl update
supervisorctl start bagm_so2_14core_ddp4_worker
```

Once the queue reports a run ID, the live CSV is safe to copy or download after
every completed epoch because each append is flushed and fsynced. The structured
event log and queue-owned stdout remain available in parallel:

```bash
export RUN_ID=r_YYYYMMDDTHHMMSSZ_...
tail -f "$BAGM_SCRATCH_ROOT/active_runs/$RUN_ID/results/epoch_metrics.csv"
tail -f "$BAGM_SCRATCH_ROOT/active_runs/$RUN_ID/metrics/events.jsonl"
tail -f "$BAGM_SCRATCH_ROOT/active_runs/$RUN_ID/logs/stdout.log"
```

During training, `checkpoints/latest.ckpt` is atomically replaced after every
epoch. If an attempt fails, set `launcher.resume_checkpoint` in an otherwise
identical additive retry config to the failed bundle's absolute
`checkpoints/latest.ckpt` and enqueue that config through the same command. The
runner reconciles the copied CSV to the checkpoint epoch, rejects duplicates or
gaps, and preserves the exact future order/mask sequence. Torchrun never restarts
mid-epoch. Successful completion atomically leaves only
`checkpoints/last.ckpt`; reload and bundle verification are:

```bash
/venv/main/bin/python - <<'PY'
from pathlib import Path
import torch
checkpoint = Path("/workspace/BAGM/artifacts/runs/YYYY/MM/RUN_ID/checkpoints/last.ckpt")
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
print(payload["completed_global_epochs"], payload["model_state_checksum"])
PY

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" verify-artifacts
```

> The trained model captures predictive dependencies useful for
> masked-expression reconstruction. Attention, gradients, and Jacobians are
> model-derived quantities and do not by themselves establish direct signaling
> or causality.

## Completed contextual hL UMAP extension

The locked joint Leiden resolution-1.0 labels now have a deterministic,
CPU-only UMAP visualization for all 246,063 cells. The two-panel figure shows
the immutable clusters and the same coordinates colored by tissue core. The
visualization contract, exact reproduction command, checksums, observed
core-associated islands, and claim limits are documented in
[`HL_UMAP.md`](HL_UMAP.md). The verified additive report is under
`reports/analyses/so2_14core_contextual_embedding_umap/` and does not modify
the completed model run or its source clustering report.
