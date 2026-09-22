# SO1 four-block geometry-modulated Relative-QKV strict-plateau fit

Campaign ID:
`cmp_20260905_so1_14core_geometry_modulated_relative_qkv_seed0_batch2_plateau_min150`.

Phase: completed. The production run passed its integration and four-rank
preflight gates, trained from scratch, and stopped under the frozen strict
training-loss plateau rule.

This campaign trains one fresh seed-0 model on every routed cell in SO_1 cores
1 through 14. It reuses the exact four-block geometry-modulated architecture
successfully trained in SO2 run
`r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf`, but it does not load that
run's checkpoint, optimizer state, metrics, masks, or stopping decision. SO1's
immutable preprocessing, graph artifacts, fit-only diagnostic evaluation, and
strict training-loss plateau protocol remain authoritative.

## Scientific question and scope

The question is whether the geometry-modulated cosine-QKV score produces a
lower held-in fixed-mask Huber loss than the completed four-block SO1
Relative-QKV reference while remaining numerically and operationally valid.
The primary hypothesis predicts a lower final
`fit/uniform_per_cell/masked_huber`; the main alternative is no improvement or
worse fit because the SO2 result does not transfer across slide-specific tissue
composition and optimization. Capacity, logit scaling, and different
optimization dynamics are competing explanations for any difference; this is
not an isolated test of dimension-wise modulation.

The estimand is partial-gene masked-expression reconstruction within the fitted
SO1 cohort. There is no validation/test partition, held-out core, held-out slide,
or patient-level replication. A positive result supports only an exploratory
single-seed fitted-cohort architecture comparison. Attention and gradients are
computational routing and local sensitivity diagnostics, not biological or
causal evidence.

The within-SO1 reference is completed run
`r_20260826T204925Z_06d12943_s000_f00_a01_94691119`: 5,003,016 parameters,
strict plateau at epoch 650, and fixed-mask held-in Huber
`0.23698082992008754`. The new model has 5,134,088 parameters, so any difference
may reflect the complete score family and added capacity. The completed SO2
geometry run is architecture/resource provenance only; its weights are not an
input.

## Locked cohort

| Alias | Original core | Cells |
| --- | ---: | ---: |
| SO1-C01 | 1 | 8,924 |
| SO1-C02 | 2 | 7,450 |
| SO1-C03 | 3 | 12,190 |
| SO1-C04 | 4 | 14,657 |
| SO1-C05 | 5 | 11,399 |
| SO1-C06 | 6 | 18,212 |
| SO1-C07 | 7 | 10,722 |
| SO1-C08 | 8 | 4,972 |
| SO1-C09 | 9 | 17,223 |
| SO1-C10 | 10 | 14,756 |
| SO1-C11 | 11 | 18,145 |
| SO1-C12 | 12 | 7,816 |
| SO1-C13 | 13 | 5,345 |
| SO1-C14 | 14 | 9,785 |
| **Total** | **14 cores** | **161,596** |

The cohort contains 205 routed FOVs and no excluded unmapped SO1 FOVs or cells.
Its tissue context is mixed or unresolved; core-number routing is not a diagnosis
label. The immutable hashes are:

- dataset content: `e006316e0f04afa645191544bcac8aa64f58c423e755f2f8d79bfd9db68a233d`;
- split: `a0d2c008ff02471010585a023724f75d6f029d1a81f39f1f2dbba1531091b9a4`;
- cohort manifest file: `15f9da492959c35d89020b3956047ec163a5f3537eaaefd5b8a011cb9279b440`;
- graph content: `5262453fc631c15a66f00f960de2a766a6142a4f1f7644b8b77a8784ec43d3b4`;
- graph manifest file: `754e98fa1b2d8b488892c4effbf095cf26da6d99c927ee45e9ee01c2d64b978f`;
- completed cohort-with-graphs manifest:
  `e078588668d9b2285db27da6cda8b7f1d144aa055065c3022e43048fd1951596`.

## Locked model and protocol

The model has four independently parameterized graph blocks, hidden dimension
256, eight 32-dimensional heads, 1,024-wide FFN and decoder, and expression-only
values. Queries and keys are L2-normalized per head. A geometry MLP produces
mean-one dimension-wise modulation and bounded geometry bias; a learned bounded
per-head logit scale starts at `1.8856180831641267`. Geometry projection heads
start at zero, giving unit modulation and zero bias. FP32 scoring/accumulation,
128-receiver chunks, at most 50,000 edges per chunk, and activation checkpointing
match the completed SO2 geometry implementation.

Everything slide-specific is inherited literally from the completed SO1
reference: 1,000 transformed expression targets; the same 22 permitted metadata
fields; the 70-dimensional relative geometry cache; radial-stratified `k=200`,
500-um union graph with no self/cross-core edges; ten uniform integer-count mask
views; batch-two/equal-core updates; AdamW at `1e-4`; weight decay `1e-5`; Huber
delta 1; gradient clipping at 1; deterministic four-rank NCCL; and no elastic
restarts.

The active strict stopping rule begins auditing at epoch 150 in 25-epoch blocks,
uses the last 50 equal-core training losses, and requires two consecutive audits
with absolute relative half-window change at most `0.0005` and normalized
absolute slope at most `0.000025` per epoch. Earliest stop is epoch 175; there is
no scientific maximum epoch cap and no validation or test selection.

## Observability, storage, and controls

Rank zero must fsync one loss row per completed epoch. Full-model and four-block
gradient norm/direction summaries are read from finite FP32 DDP-averaged
gradients after AMP unscale and before clipping. Only scalar epoch summaries are
persisted; gradient tensors, per-step files, and gradient vectors in checkpoints
are prohibited. The existing observer schema IDs retain their legacy `so2_`
prefix, but their row semantics are slide-agnostic and unchanged.

Training replaces one atomic `checkpoints/latest.ckpt` each epoch. Successful
finalization leaves one verified `checkpoints/last.ckpt`; epoch archives and best
checkpoint selection are prohibited. Loss and gradient plots are descriptive and
cannot affect optimization or stopping.

The locked SO1 Relative-QKV run is the primary baseline. The unchanged cohort,
graph, masking, trainer, and evaluation semantics control slide-specific data
differences. No spatial null, randomization, or perturbation is included, so the
campaign cannot support an interaction-mechanism claim.

## Run order and gates

1. Validate the composed configuration and frozen section hashes.
2. Run focused model/config/runner tests.
3. Run the distinct four-rank preflight on the two largest cores, C06 and C11.
4. Require exactly 5,134,088 parameters, four disjoint blocks, neutral geometry
   initialization, finite loss/gradients, checkpoint reload, peak VRAM at most
   22 GiB, and at least 2 GiB headroom on every 24-GiB GPU.
5. Register and enqueue exactly one fresh seed-0 attempt only after the preflight
   receipt passes and all four GPUs are idle.
6. Train until the frozen SO1 strict plateau rule passes, verify scalar rows and
   the sole final checkpoint, then finalize the immutable run bundle.

After shared production integration, the preflight command is:

```bash
cd /workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_STATE_ROOT=/workspace/BAGM/state
export PYTHONPATH=/workspace/BAGM-relative-qkv-analysis/src
CUDA_VISIBLE_DEVICES=0,1,2,3 /venv/main/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  scripts/diagnostics/preflight_so1_14core_relative_qkv_ddp.py \
  --config configs/experiment/so1_14core_geometry_modulated_relative_qkv_seed0_batch2_plateau_min150.yaml \
  --output state/preflight/so1_14core_geometry_modulated_relative_qkv_ddp4_plateau_min150.json
```

Automatic enqueue is false. A config/hash mismatch, unexpected resume source,
non-finite value, missing block gradient, DDP mismatch, CSV gap, checkpoint
retention violation, failed reload, or VRAM gate failure blocks launch. A valid
run with no improvement is a negative scientific result, not a reason to alter
the frozen thresholds.

## Completed outcome

Production run `r_20260905T065805Z_b13f5b24_s000_f00_a01_b1658a6c`
completed at epoch 475 after the strict plateau criterion passed. Its final
fixed-mask held-in Huber loss was `0.23747088760137558`, compared with
`0.23698082992008754` for the locked SO1 Relative-QKV reference. The primary
hypothesis was therefore not supported: the geometry-modulated variant was
slightly worse on this fitted-cohort diagnostic. This is not a generalization
estimate because the campaign has no validation or test partition.

The run retained only `checkpoints/last.ckpt` (SHA-256
`ba5999ab030a3a512447f1b5135d5413ac7f19490d501edf0d318a6857c2f93b`).
Loss and gradient-direction summaries were recorded for every completed epoch,
and measured peak VRAM across the four ranks was 6.756 GiB.
