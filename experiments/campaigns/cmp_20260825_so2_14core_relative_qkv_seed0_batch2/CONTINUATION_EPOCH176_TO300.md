# Fixed epoch-176-to-300 continuation

## Status and non-retroactivity

The original seed-0 production run completed successfully on 2026-08-26 at
175 global epochs under its original two-audit training-loss plateau rule. Its
run bundle, completion marker, plateau decision, epoch-175 `last.ckpt`, and
registry record remain valid and immutable. This continuation neither retracts
that result nor changes any byte in the completed bundle.

The later user instruction authorizes a distinct successor run that resumes
the verified epoch-175 state and trains through the fixed final epoch 300. The
successor is governed by
`task_contract_amendment_001_fixed_epoch300_continuation.yaml` and the additive
experiment config
`configs/experiment/so2_14core_relative_qkv_seed0_batch2_resume175_fixed300.yaml`.
It receives a new run ID and its own archive bundle; it is not a retry attempt
or replacement for the completed source run.

The amendment file SHA-256 is
`4acf79c598be8c4155f6d519bf446ed32ef1a5628b8cc18960f05017981863d6`;
the adjacent sidecar and `campaign.yaml` bind that exact value. The original
`frozen_task_contract.yaml` and its SHA sidecar are unchanged.

## Immutable source lineage

| Field | Locked value |
| --- | --- |
| Source run ID | `r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6` |
| Source artifact | `artifacts/runs/2026/08/r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6` |
| Completed epochs | 175 |
| Completed optimizer updates | 1,225 |
| Source checkpoint | `checkpoints/last.ckpt` |
| Checkpoint SHA-256 | `2e0f9d837fdbb78673f9788e355ce7a6f6843ffa1fc7a46a6b0c126d53fe7d8d` |
| Resume-payload SHA-256 | `fd7fa5031a6eaebbfe279cc75d8c4c2821a8619916e6221b02a10123d3d36724` |
| Model-state SHA-256 | `594a8a27beec40902febffcd2f3682824bb89117bcd330400ba8eb2ec87f46ad` |
| Optimizer-state SHA-256 | `32d84e63307d1711c50b8bc0835bc23a863bd02ec67f18e2f87a4c9fa824be3c` |
| AMP-scaler SHA-256 | `9c0cef8b853609a7e02dae61329567280f61370ad375d457051f73799f2f7048` |
| History SHA-256 | `c1d911cd39ed22b765aa43c9e68718739d4cf1ba13cfbcf3c02fc5b7e5bd15ca` |
| Resolved-config file SHA-256 | `47bdadef687a84d0a9212bb1325d2ed2ac49bea2e596a08e171858a4125db3bb` |
| Epoch-metrics CSV SHA-256 | `970bb65bf2bef092838844109c04437d35c89cc2a85424baf5e8acc1bf747e46` |
| Run-manifest file SHA-256 | `36d15da9ff13c7e6365c4eb560c2093c6ebee22a560e469445a1a0a843b84b00` |
| `_SUCCESS` file SHA-256 | `3199501e6b19c36537743982816c5ceb2208a86dc7366e4cebf81490e5f8a5ec` |

The continuation loader must verify these identities before restoring the
model, optimizer, AMP scaler, histories, and deterministic schedule state. The
source checkpoint's historical plateau payload describes why the source run
completed; it is lineage evidence, not a stopping instruction for the fixed
successor.

## Fixed training arithmetic

The successor begins with human-readable epoch 176 (internal zero-based epoch
index 175) and ends after exactly 300 completed global epochs. It therefore
adds 125 epochs. At seven paired-core optimizer updates and 140 complete-core
mask views per epoch, it adds:

```text
125 epochs × 7 updates/epoch = 875 additional optimizer updates
1,225 + 875 = 2,100 cumulative optimizer updates

125 epochs × 140 views/epoch = 17,500 additional complete-core mask views
125 epochs × 10 masks/cell/epoch = 1,250 additional masks per cell
1750 + 1250 = 3,000 cumulative masks per cell
```

All scientific and execution settings otherwise remain unchanged: the exact
246,063-cell SO2-C15-through-SO2-C28 cohort, fit-only transductive role,
1,000-gene and metadata schemas, radial-stratified nominal `k=200`/500-µm
graphs, model architecture and seed 0, ten independent masks per cell/core/
epoch, AdamW parameters, exact four-rank batch-2 gradient mean, no neighbor
sampling, and deterministic schedule derivations.

Training duration is fixed. Plateau behavior, held-in diagnostics, or any
other metric cannot stop, extend, or select a checkpoint. The successor writes
one final independently reloadable `checkpoints/last.ckpt` at epoch 300 and no
intermediate checkpoint. The source epoch-175 checkpoint remains available in
its immutable source bundle.

## Strict exploratory plateau diagnostic

The continuation additionally records a stricter stationarity diagnostic over
the equal-core mean masked-Huber training history. These thresholds were chosen
after inspecting the completed epoch-175 history, so they are explicitly
**post-hoc and exploratory**, not a prespecified convergence test, validation
criterion, or calibrated statistical confidence statement.

At every 25-epoch boundary, a 50-epoch window is divided into adjacent 25-epoch
halves. The diagnostic passes an audit only when all losses are finite and both
conditions hold:

\[
\frac{|\bar L_{\mathrm{recent\ 25}}-\bar L_{\mathrm{previous\ 25}}|}
{|\bar L_{\mathrm{previous\ 25}}|}
\le 0.0005,
\]

\[
\frac{|\operatorname{slope}(L_{t-49:t})|}
{|\operatorname{mean}(L_{t-49:t})|}
\le 0.000025\quad\text{per epoch}.
\]

Both the absolute half-window change and the absolute normalized slope are
fourfold tighter than the original signed-improvement/slope limits of `0.002`
and `0.0001`. The absolute change also prevents worsening loss from satisfying
the mean-change condition merely because its signed improvement is negative.
Two consecutive audits are required; the final epoch-300 status is called
`strict exploratory plateau observed` only if the epoch-275 and epoch-300
audits both pass.

For context, the source epoch-175 audit had a relative adjacent-window change
of `0.0013815334550942554` and normalized absolute slope
`0.00003688024113199939`. It passed the original rule but fails both stricter
exploratory limits. This descriptive comparison must not be presented as an
independent validation of the new thresholds. If the epoch-300 diagnostic
fails, the fixed epoch-300 checkpoint is still finalized and the failure is
reported without extending training.

## Additive configs and execution provenance

The successor uses:

- trainer: `pooled_relative_qkv_14core_batch2_resume175_fixed300`;
- evaluation: `held_in_so2_14core_fit_diagnostics_fixed300`;
- launcher: `local_four_gpu_3090_ddp_resume175_fixed300`;
- evaluation protocol:
  `held_in_pooled_14core_relative_qkv_fixed_continuation_epoch300`;
- new preflight receipt:
  `state/preflight/so2_14core_relative_qkv_ddp4_resume175_fixed300.json`.

The new preflight receipt must checksum-bind the resolved successor config,
source checkpoint identity, immutable cohort and graph manifests, exact DDP
layout, AMP/FP32 equivalence gates, finite gradients, peak VRAM, and a save/load
round trip. The original preflight receipt is not silently reused.

After the additive continuation protocol is available in the normal config,
queue, runner, and archive validators, create the new receipt and enqueue the
new experiment through the existing supervisor-owned four-GPU worker:

```bash
cd /workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export PYTHONPATH=/workspace/BAGM-relative-qkv-analysis/src

CUDA_VISIBLE_DEVICES=0,1,2,3 /venv/main/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  scripts/diagnostics/preflight_so2_14core_relative_qkv_ddp.py \
  --config configs/experiment/so2_14core_relative_qkv_seed0_batch2_resume175_fixed300.yaml \
  --output "$BAGM_STATE_ROOT/preflight/so2_14core_relative_qkv_ddp4_resume175_fixed300.json"

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" enqueue-experiment \
  --campaign-id cmp_20260825_so2_14core_relative_qkv_seed0_batch2 \
  --config configs/experiment/so2_14core_relative_qkv_seed0_batch2_resume175_fixed300.yaml \
  --priority 100 --max-attempts 2 --gpu 0,1,2,3
```

The successor bundle must contain a checksum-bound continuation-lineage receipt
and source-checkpoint verification receipt. Its cumulative metrics history must
retain the origin of epochs 1–175 and append epochs 176–300 without rewriting
the immutable source CSV. The completed successor is a new archived run, while
the source bundle remains independently verifiable at its original path.

> The trained model captures predictive dependencies useful for
> masked-expression reconstruction. Attention, gradients, and Jacobians are
> model-derived quantities and do not by themselves establish direct signaling
> or causality.
