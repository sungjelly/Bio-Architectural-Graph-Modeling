# Six-core Cancer relative-geometric QKV Transformer

## Status and scientific scope

Phase: implementation. Outcome: pending.

The active first phase fits one seed-0 shared masked-expression model across
six disconnected Cancer-core graphs: `CAN-01`, `CAN-09`, `CAN-13`, `CAN-15`, `CAN-21`, and
`CAN-23`. All eligible cells are fit data. There is no validation or test
partition, no early stopping, no best-epoch selection, and no generalization
claim. Fixed-mask computations after training are held-in fit diagnostics.

Protected historical sources contain conflicting earlier labels for part of
the requested cohort. The user explicitly attested on 2026-08-24 that all six
cores were rediagnosed as Cancer. Seeds 1 through 4 and the five-seed ensemble
report are explicitly deferred by Amendment 004 and are not part of the
current completion claim. The versioned
`cancer_6core_user_rediagnosis_attestation_v1` policy preserves both the
earlier non-identifying labels and the newer attestation without rewriting
protected inputs. Tissue labels, core identity, slide, FOV, patient identity,
and cell identity are never model inputs.

The scientific question is whether a shared relative-geometric QKV graph
Transformer can reconstruct independently masked expression in these six
fitted cores while exposing stable, transformation-invariant model-routing
summaries. A credible alternative is that reconstruction is dominated by
same-cell co-expression and always-visible morphology, with spatial attention
reflecting density, distance, segmentation spillover, or unstable routing.

The maximum defensible result is a model-derived predictive dependency within
the fitted cohort. This is not independent biological replication or causal
evidence.

## Input contract

Each cell token combines three separately learned projections:

\[
h_i^{(0)}=\operatorname{Dropout}\!\left(\operatorname{GELU}\!\left(
\operatorname{LN}\left[W_x(x_i\odot(1-m_i))+W_m m_i+W_c c_i+b\right]
\right)\right).
\]

The expression schema contains the same ordered 1,000 biological CosMx probes
for every core. `Negative*`, `SystemControl*`, and all technical controls are
excluded. Targets are raw-count `log1p` values standardized by shared
equal-core-weighted first and second moments. No library-size normalization or
RNA-derived total enters the model.

The 22 permitted metadata fields are morphology and imaging measurements:
`Area`, `Area.um2`, `AspectRatio`, `Width`, `Height`, mean/max PanCK, G,
Membrane, CD45, and DAPI, `SplitRatioToLocal`, `NucArea`, `NucAspectRatio`,
`Circularity`, `Eccentricity`, `Perimeter`, and `Solidity`. The preparation
receipt records imputation, transformation, missing indicators,
standardization, names, and checksums. Metadata is always visible.

## Graph and physical coverage

Each core has an independent graph in the full-core global micrometre frame.
There are no cross-core edges and no self-loops. Directed selection uses a
nominal budget of 200 sources per receiver with quotas 48, 64, 48, and 40 in
the `(0,50]`, `(50,150]`, `(150,300]`, and `(300,500]` µm shells. Unused quota
is filled by the nearest remaining cells within 500 µm. Distance and source
index provide deterministic tie-breaking.

Every retained relation is made bidirectional, deduplicated, stripped of self
edges, receiver-major sorted, and checksummed. Final in-degree can exceed 200.
Graph QC records cell and edge counts, degree and distance distributions,
deduplication counts, zero cross-core edges, and the fraction of physically
eligible receivers retaining a 300–500 µm source. The graph gate requires all
eligible receivers to have long-range coverage.

## Relative positional encoding

Coordinates never enter a node projection, Q/K/V, decoder, or metadata.
Coordinates deterministically produce only pairwise relative positional
features. For edge \(j\to i\):

\[
\Delta p_{ij}=p_j-p_i,\quad r_{ij}=\lVert\Delta p_{ij}\rVert,\quad
\hat d_{ij}=\Delta p_{ij}/(r_{ij}+\epsilon).
\]

Distance uses 64 overlapping Gaussian RBFs from 0 to 500 µm and a smooth
cutoff approaching zero at 500 µm. A coordinate-only local covariance within
150 µm, capped at 64 cells and weighted with a 75 µm Gaussian, yields

\[
T_i=\frac{C_i}{\operatorname{tr}(C_i)+\epsilon}-\frac12I,\qquad
a_i=\sqrt{2\operatorname{tr}(T_i^2)}.
\]

The 70-dimensional relative vector is

\[
\rho_{ij}=\left[\operatorname{RBF}_{1:64}(r_{ij}),
\hat d^\top T_i\hat d,\hat d^\top T_j\hat d,
\operatorname{tr}(T_iT_j),a_i,a_j,r_{ij}/500\right].
\]

These contractions are invariant to consistent translation, rotation, and
reflection. The orientation tensor is axial: it distinguishes along versus
across a local axis but cannot distinguish opposite directions along that axis
without an independently measured polarity signal.

For production, each core has one shared, read-only, checksum-bound memory-map
cache of these 70 values in canonical edge order. The cache is not duplicated
inside seed run bundles and can be reproduced exactly from coordinates,
orientation tensors, edges, and locked RBF parameters. Caching once is needed
to avoid recomputing the same geometry in all ten mask views and all five
seeds. On the measured RTX 3090 profile, one current core's cache is staged
once in AMP float16 and reused across its ten views; attention intermediates
remain exactly receiver-chunked. Only one core is resident at a time.

## QKV architecture

The production model has width 256, four pre-LayerNorm graph blocks, eight
32-dimensional heads, a 1,024-wide FFN, and a `256 → 1024 → 1000` expression
decoder. Residual/FFN dropout is 0.10 and attention dropout is zero.

For layer \(\ell\) and head \(h\):

\[
q_{ih}=W_{Q,h}\operatorname{LN}(h_i),\quad
k_{jh}=W_{K,h}\operatorname{LN}(h_j),\quad
v_{jh}=W_{V,h}\operatorname{LN}(h_j),
\]

\[
b_{ijh}=f_h(\rho_{ij}),\quad
s_{ijh}=q_{ih}^{\top}k_{jh}/\sqrt{32}+b_{ijh},
\]

\[
\alpha_{ijh}=\operatorname{softmax}_{j\in\mathcal N(i)}s_{ijh},\qquad
z_{ih}=\sum_{j\in\mathcal N(i)}\alpha_{ijh}v_{jh}.
\]

The positional network is `70 → 128 → LayerNorm → GELU → 8`, with its final
projection zero-initialized. Geometry modifies attention logits only. It does
not create edge keys, edge values, value gates, raw-coordinate value content,
or position-conditioned messages. Heads are concatenated, output-projected,
added residually, and followed by a pre-LayerNorm `256 → 1024 → 256` FFN.

Receiver chunking is exact: every receiver's complete incoming neighborhood is
normalized together, without sampling, truncation, approximate softmax, or
split normalization. Half-precision logits, softmax, messages, and reductions
accumulate in FP32. Activation checkpointing is enabled. Initial execution
limits are 512 receivers and 200,000 edges per chunk; only these execution
limits may change after measured preflight.

## Masking, loss, and training

For every cell, mask view, and global epoch, the masked count is sampled
uniformly from the integers 0 through 1,000, then exactly that many distinct
gene positions are selected. Both endpoints are valid and there are no mask
ratio bins or strata. Every core optimizer step uses ten independent views of
this same full distribution. Seeds derive only from base mask seed
`2026082401`, core alias, global epoch, and view index; the model seed is
excluded, so all five models see identical paired mask schedules. The encoder
reapplies every mask internally.

Loss is mean masked Huber on standardized `log1p` targets with delta 1.0.
Optimization uses AdamW, learning rate `1e-4`, weight decay `1e-5`, gradient
clip norm 1.0, no scheduler, and mixed precision only after an FP32-equivalence
preflight.

One global epoch contains one complete-core optimizer step for each of the six
cores in deterministically shuffled order. Only one complete core is staged on
the GPU at a time. For one core step, gradients are cleared once; ten complete
graph forwards use ten independent masks; each masked Huber loss contributes
`loss / 10` to backpropagation; gradients are clipped once; and the optimizer
steps once. Thus the ten mask-specific gradients are averaged without changing
core weighting or optimizer-step count.

Seed 0 runs at least 150 global epochs and 900 optimizer steps, executing at
least 9,000 complete-core masking views. Each cell therefore receives at least
1,500 mask realizations. Amendments 003 and 004 supersede Amendment 002's
fixed-300 stopping clause: epoch 150 is the first seed-0 plateau audit,
not a fixed endpoint. The prespecified equal-core mean training-loss rule is
audited every 25 epochs over a 50-epoch window. Seed 0 must pass at two
consecutive audits; otherwise it continues for another 25-epoch block. The
two-audit confirmation makes epoch 175 the earliest
possible final epoch. There is no scientific maximum epoch cap. Epoch-boundary
resume checkpoints are retained every 25 epochs; the canonical model is the
last checkpoint at the confirmed plateau epoch, never a selected best
epoch. This is training-loss convergence monitoring, not validation or model
selection.

## Hardware preflight

The binding seed-0 preflight used the complete largest graph, `CAN-23`
(43,462 cells and 9,972,300 directed edges), on one RTX 3090. With receiver
chunks of 512 and at most 200,000 edges per chunk, the complete-core AMP
forward/backward pass retained every edge, used 15.10 GiB peak allocated VRAM,
and took 7.29 seconds after one-time graph staging. The full-edge reference and
chunked predictions differed by at most `3.73e-7`; attention differed by at
most `1.12e-8`; AMP and FP32 predictions differed by `5.64e-4` maximum and
`9.38e-5` mean. Loss and gradients were finite, and save/load predictions were
bit-identical. The diagnostic receipt SHA-256 is
`e142f29b9a9aba9da405204ad5804a146731bcae6614d1954e1d58bc4919afb3`.
This is a resource and numerical diagnostic, not a completed experiment.

## Interpretation and ensemble outputs

Post-training, receiver-chunked extraction can return aligned edge indices,
per-head attention, content logits, positional bias, combined logits, node
embeddings, relative summaries, source/receiver indices, and layer number.
Cell-cell Parquet shards add plotting-only coordinates, distance, degree
adjustment, reciprocal keys, and descriptive mutual routing scores. Coordinates
in exports did not enter node inputs.

Selected autograd hooks compute
\(\partial\alpha_{j\to i}/\partial x_{j,g}\) and
\(\partial\hat x_{i,g_t}/\partial x_{j,g_s}\) without materializing exhaustive
edge-by-gene Jacobians.

The five-seed report uses fixed held-in masks shared across seeds, linear CKA,
optional Procrustes alignment, Hungarian head matching, matched attention
correlations and overlaps, positional/content contribution agreement, mutual
pair stability, and selected-gradient stability. Reported uncertainty is
ensemble spread or seed uncertainty, not a calibrated biological confidence
interval.

## Completion gates and limitations

Production begins only after graph, invariance, full/chunk equivalence,
AMP/FP32, finite-gradient, save/load, resume, and largest-core VRAM checks pass.
This first phase requires one loadable seed-0 checkpoint at its confirmed
training-loss plateau, a verified immutable bundle and completion marker, and
a registered checkpoint-catalog record. It must not be described as a
completed five-seed campaign or an ensemble stability result. Current local storage is
not volume-backed; this operational risk must remain visible and the instance
must not be recycled before outputs are secured.

The study is transductive, uses a targeted 1,000-probe panel, and has only six
fitted cores. It cannot establish patient generalization, distinguish direct
from multihop effects by attention alone, rule out segmentation spillover or
broad spatial fields, or establish a biological mechanism.

> The trained model captures predictive dependencies useful for masked-expression reconstruction. Attention, gradients, and Jacobians are model-derived quantities and do not by themselves establish direct signaling or causality.
