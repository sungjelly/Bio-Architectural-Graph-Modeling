# SO2 cores 15–28 untied eight-block Relative-QKV fit

## Frozen task contract

Phase: planned full exploratory run; implementation and preflight pending.
Outcome: pending. This is an exploratory single-seed depth comparison because
the four-block baseline outcomes were inspected before this contract was
written. It cannot establish architectural equivalence, generalization, or a
biological mechanism.

Campaign: `cmp_20260903_so2_14core_untied8_relative_qkv_seed0_batch2`.
Exactly one fresh model-seed-0 production run is planned. It starts from epoch
0 and imports no model or optimizer state.

### Question, hypothesis, and alternatives

Question: under the literal August 25 SO2 training protocol, does increasing
the Relative-QKV graph depth from four independently parameterized blocks to
eight independently parameterized blocks improve the fitted masked-expression
objective without unacceptable optimization instability?

Primary exploratory hypothesis: the untied eight-block model will train with
finite gradients, complete the unchanged August 25 plateau rule, and have an
equal-core mean training masked-Huber loss at global epoch 150 no higher than
the four-block reference value `0.2416047074965068`. A result above that value
is negative for the directional depth hypothesis, not an infrastructure
failure.

Credible alternatives are that four blocks already provide adequate effective
depth; extra depth causes oversmoothing, redundant transformations, or poorer
optimization; or the held-in partial-gene objective is dominated by same-cell
expression and morphology and therefore is insensitive to graph depth.
Layerwise gradient norms and directions discriminate dead/unstable blocks from
a merely neutral loss result, but gradient cosine alone is never convergence
evidence and is interpreted jointly with loss and gradient norm.

### Estimand and maximum claim

The estimand is held-in, all-cell, transductive partial-gene masked-expression
reconstruction across fourteen disconnected SO2 complete-core graphs,
`SO2-C15` through `SO2-C28`. There are 246,063 fitted cells and 1,000 biological
genes. SO2 FOV 246 and its 340 cells remain excluded because the FOV is
unmapped. There is no validation/test split.

The maximum claim is a seed-0 exploratory comparison of fitted reconstruction
and optimization behavior on these exact cores. Cells are not independent
biological replicates. No patient-held-out, spatial-block, clinical,
communication, mechanistic, or causal claim is supported.

### Architecture and control

The model is the ordinary, non-recurrent
`ReceiverChunkedRelativeGeometryQKVGraphTransformer`. It has eight separately
constructed graph blocks, each applied once in sequence:

```text
NodeEncoder -> B0 -> B1 -> B2 -> B3 -> B4 -> B5 -> B6 -> B7 -> Decoder
```

`graph_layers=8`, `unique_graph_blocks=8`, `effective_graph_depth=8`, and
`graph_block_weight_tying=none`. There is no recurrent unroll field. Every
block has independent LayerNorm, Q/K/V, positional-bias, output-projection, and
FFN parameters. With 1,000 genes and 22 covariates, the locked expected count
is 8,199,464 trainable parameters: 518,400 encoder + eight times 799,112 per
block + 1,288,168 decoder.

The immutable reference control is four-block run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`, with 5,003,016 parameters,
epoch-150 training loss `0.2416047074965068`, and original plateau completion
at epoch 175. Its later fixed epoch-300 continuation is post-hoc context, not
the primary control.

All non-depth factors reuse the August 25 configs without overrides:

- dataset `so2_14core_pooled_fit_v1`;
- features `cosmx_so2_14core_metadata_relative_geometry`;
- graph `so2_radial_stratified_k200_r500_bidirectional`;
- masking `uniform_per_cell_0_100`, including mask seed `2026082401`;
- trainer `pooled_relative_qkv_14core_batch2_plateau_min150`, including core
  order seed `2026082402`;
- width 256, eight 32-dimensional heads, FFN/decoder width 1,024, geometry only
  as attention-logit bias, exact receiver chunking, and activation checkpointing;
- four-rank DDP on GPUs 0–3, 14 core visits, 140 mask-view gradients, and seven
  optimizer updates per global epoch.

No graphless, rewired, additional-seed, or held-out arm is added. Those absent
controls prevent a general depth or biological conclusion.

### Stopping, operational review, and checkpoints

The literal August 25 rule remains scientific source of truth: minimum 150
global epochs; audits every 25 epochs over a 50-epoch training-loss window;
relative mean improvement at most `0.002`; normalized absolute slope per epoch
at most `0.0001`; stop after two consecutive passing audits; earliest stop 175;
and `maximum_scientific_epoch_cap=null`.

If epoch 300 completes without a confirmed plateau, the depth-eight worker
deliberately exits non-success after durably writing the epoch-300 metrics and
atomically replacing `checkpoints/latest.ckpt`. The queue bundle is then
failed/inconclusive, not scientifically converged. It may be resumed only
after an explicit campaign amendment and operational review; the one-attempt
queue policy prevents an automatic duplicate approximately 20-hour retrain.
This gate is not a scientific convergence criterion, does not change the
plateau calculation, and is not a scientific maximum epoch cap.

Loss is flushed and fsynced every epoch. During training, exactly one rolling
`checkpoints/latest.ckpt` is atomically replaced at each epoch boundary. No
epoch archive or best checkpoint is retained. A successful final bundle keeps
only one independently reloadable `checkpoints/last.ckpt`.

### Scalar-only gradient diagnostics

After AMP unscale and finite checking, before clipping or stepping, rank zero
reads the full FP32 DDP-averaged gradient in trainable parameter registration
order. The existing full-model scalar summary is written once per completed
epoch to `results/gradient_direction_metrics.csv`, schema
`so2_full_gradient_direction_metrics_v1`.

A separate long-form block-only file,
`results/gradient_direction_by_block.csv`, schema
`so2_block_gradient_direction_metrics_v1`, contains exactly eight ordered rows
per completed epoch for `blocks.0` through `blocks.7`. Each row records block
index/name, block trainable-parameter count, observed update count, pre-clip
gradient-norm mean/min/max, consecutive-update cosine mean/median/min/max and
valid-pair count, prior-epoch aggregate-gradient cosine, and resume-boundary
availability. Encoder and decoder are excluded from the layerwise file.

Fresh epoch 1 has six possible consecutive pairs and no prior-epoch aggregate
cosine. Later uninterrupted epochs have seven possible pairs; the first epoch
after process/checkpoint resume has six and flags the unavailable boundary.
Zero-norm cosine pairs are invalid and omitted. No gradient tensor,
per-optimizer-step file, or gradient vector in a checkpoint may persist. These
diagnostics never affect optimization, stopping, or checkpoint choice.

### Acceptance and failure criteria

Technical acceptance requires configuration/queue tests; exactly eight unique
block parameter sets and state prefixes `blocks.0`–`blocks.7`; no cross-block
parameter identity; measured count 8,199,464; finite gradients reaching every
block; AMP/FP32 and receiver-chunk/full-edge equivalence; a passing distinct
four-rank largest-core preflight whose peak VRAM is at most 22.0 GiB on every
24-GiB rank (at least 2 GiB headroom); immutable input checksums; exactly one
loss row, one global-gradient row, and eight block-gradient rows per completed epoch;
valid literal plateau completion; a single reloadable final checkpoint; and
verified final bundle/catalog artifacts.

The directional hypothesis is supported for this seed only if technical
acceptance passes and epoch-150 equal-core training loss is at most the locked
four-block reference. It is negative if a valid run exceeds that value and
inconclusive if it cannot reach epoch 150 or comparison integrity is lost.

Stop/failure conditions include architecture or untied-identity drift, missing
block gradients, input checksum drift, invalid DDP aggregation, non-finite loss
or gradient, unrecoverable OOM, nondeterministic resume, missing/gapped scalar
rows, any persisted gradient vector/tensor or per-step diagnostic file,
multiple rolling/final checkpoints, or checkpoint/finalization verification
failure. Reaching epoch 300 without plateau triggers the deliberate resumable
non-success review outcome above, not a scientific failure and not automatic
scientific convergence.

### Resources and artifacts

Production requests one four-process NCCL job on four 24-GiB RTX 3090 GPUs,
with one complete core staged per rank and at least 25 GiB free disk. The
four-block control peaked near 15.21 GiB. Eight blocks add 3,196,448 parameters
and double graph-block applications, so runtime and peak-memory assumptions
must be established by the new preflight rather than copied from the baseline.
Production is accepted only if every preflight rank peaks at or below 22.0 GiB,
leaving at least 2 GiB headroom on each 24-GiB card.

Expected artifacts include resolved config/provenance, per-epoch loss metrics,
global and blockwise scalar-gradient CSVs, structured metric events, logs,
resource telemetry, the distinct preflight receipt, one rolling checkpoint
during training, and one verified final checkpoint/catalog entry after
success. Final reporting also includes `figures/loss_vs_epoch.png` and
`figures/gradient_direction_vs_epoch.png`; both are derived only from durable
scalar CSVs after training and are never used for stopping, checkpoint
selection, or any training decision.

### Exact commands

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

/venv/main/bin/python -m pytest -q \
  tests/unit/infrastructure/test_configuration.py \
  tests/unit/infrastructure/test_queue_command.py

nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 /venv/main/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  scripts/diagnostics/preflight_so2_14core_relative_qkv_ddp.py \
  --config configs/experiment/so2_14core_untied8_relative_qkv_seed0_batch2.yaml

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" create-campaign \
  --campaign-id cmp_20260903_so2_14core_untied8_relative_qkv_seed0_batch2 \
  --name "SO2 untied 8-block Relative-QKV seed-0 exploratory fit" \
  --scientific-question "Does doubling untied Relative-QKV depth improve the fitted SO2 masked-expression objective?" \
  --plan experiments/campaigns/cmp_20260903_so2_14core_untied8_relative_qkv_seed0_batch2/campaign.yaml \
  --status planned

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" enqueue-experiment \
  --campaign-id cmp_20260903_so2_14core_untied8_relative_qkv_seed0_batch2 \
  --config configs/experiment/so2_14core_untied8_relative_qkv_seed0_batch2.yaml \
  --priority 100 --max-attempts 1 --gpu 0,1,2,3
```

`--max-attempts 1` is an enqueue-time operational limit, not a scientific
config field. It is locked here to prevent an automatic full retrain after the
intentional epoch-300 review exit or another non-success. The derived child
command must be four-rank `torch.distributed.run` against
`scripts/train/run_so2_14core_relative_qkv.py`.
