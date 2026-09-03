# SO2 cores 15–28 recurrent Relative-QKV seed-0 fit

## Status and scope

- Phase: active production training. Queue job
  `q_c050c006f112aadad43c` launched run
  `r_20260831T100221Z_a33f1888_s000_f00_a01_bdbebeaf` on four GPUs at
  `2026-08-31T10:02:21.590222Z`. Epoch 1 completed with equal-core masked-Huber
  `0.28640682942100926` and mean consecutive-gradient cosine
  `0.4629239911834399`. The preflight receipt content SHA-256 is
  `3ee624d94a9474a07251a5db304d62140921bc42156d37972d6e530f50f6a03a`.
- Outcome: pending.
- Analysis status: **exploratory single-seed architecture comparison**. The
  August 25 baseline and its outcomes were inspected before this contract was
  written, so this is not confirmatory and cannot establish architectural
  equivalence, generalization, or biological mechanism.
- Campaign: `cmp_20260831_so2_14core_recurrent_relative_qkv_seed0_batch2`.
- Production runs: exactly one fresh seed-0 run; no checkpoint initialization
  or optimizer state is imported from the untied baseline.

The experiment changes one architectural factor. The existing model has four
independently parameterized graph Transformer blocks. The proposed model owns
one complete block and calls that same block four times:

```text
h0 = NodeEncoder(expression, mask, permitted metadata)
h1 = B_theta(h0, graph, relative geometry)
h2 = B_theta(h1, graph, relative geometry)
h3 = B_theta(h2, graph, relative geometry)
h4 = B_theta(h3, graph, relative geometry)
prediction = ExpressionDecoder(h4)
```

`B_theta` includes both LayerNorms, Q/K/V projections, positional-bias MLP,
attention output projection, and the `256 -> 1024 -> 256` FFN. All four calls
share every block parameter. The encoder and decoder each run once. Thus the
model has one unique graph block, four message-passing applications, and an
effective graph depth of four. The expected trainable parameter count is
2,605,680 versus 5,003,016 for the untied control (47.92% fewer); the preflight
and final summary must measure rather than assume this count.

## Scientific question, hypothesis, and alternatives

Question: on the same fourteen fitted SO2 core graphs and under the literal
August 25 training protocol, can a single recurrently reused Relative-QKV block
retain useful masked-expression fitting behavior while reducing unique model
parameters?

Primary exploratory hypothesis: full block tying will produce finite,
decreasing training loss, pass the unchanged August 25 plateau rule, and keep
the equal-core mean training masked-Huber loss at global epoch 150 within 5%
of the paired untied seed-0 reference. The reference epoch-150 value is locked
at `0.2416047074965068`, so the prespecified non-inferiority-style descriptive
margin is `0.2536849428713321`. This 5% margin is a pragmatic exploratory
threshold, not a statistical equivalence margin.

Credible alternative: depth-specific parameters are needed. Reusing one block
may cause recurrent-state collapse, unstable or highly aligned updates, or an
epoch-150 loss above the margin even though the nominal message-passing depth
and compute are similar. Another explanation for a favorable result is that
the task is dominated by same-cell expression and morphology, so the fitted
objective may be insensitive to graph-block capacity.

Predictions that distinguish these explanations:

- If tying is adequate for this fitted objective, loss should decrease without
  non-finite gradients, reach the August 25 plateau stop, and remain at or
  below the epoch-150 margin.
- If independent depth-specific transformations matter, the recurrent model
  should show a reproducible deficit in the paired epoch trajectory or fail the
  margin/plateau criteria despite identical masks, core order, graph, and
  optimizer settings.
- Gradient cosine is diagnostic only. Persistent near-one or oscillating
  cosine may flag redundant or unstable recurrent updates, but cosine alone is
  never treated as convergence; it is interpreted jointly with gradient norm
  and loss.

## Estimand and maximum claim

The estimand is held-in, all-cell, transductive partial-gene masked-expression
reconstruction over SO2 cores 15–28. There is no validation or test partition.
All 246,063 mapped cells are fitting observations in fourteen disconnected
complete-core graphs; SO2 FOV 246 and its 340 cells remain explicitly excluded
because that FOV is unmapped.

The maximum defensible claim is an exploratory seed-0 comparison of fitted
masked-expression behavior and optimization diagnostics between tied and
untied four-application architectures on these exact cores. Cells are not
independent biological replicates. This run provides no patient-held-out,
spatial-block, clinical-generalization, signaling, mechanistic, or causal
evidence.

## Locked control and matched factors

The positive/reference control is the immutable August 25 untied run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`, which used four unique
blocks, completed the original plateau policy at epoch 175, and recorded
5,003,016 parameters. Its equal-core training masked-Huber was
`0.2416047074965068` at epoch 150 and `0.24084819033741953` at epoch 175.
The later fixed epoch-300 continuation is post-hoc context, not the stopping
policy or primary control for this campaign.

The recurrent run reuses, without overriding, the baseline dataset, features,
graph, masking, and trainer configs. Therefore the following are paired
controls:

- ordered cohort `SO2-C15` through `SO2-C28`, 246,063 cells and 1,000 genes;
- the same equal-core standardized `log1p` targets and 22 permitted
  morphology/imaging covariates;
- fourteen disconnected radial-stratified graphs, nominal `k=200`, 500 µm
  maximum range, bidirectional union, no self-loops or cross-core edges;
- geometry used only as a 70-dimensional per-head attention-logit bias;
- ten exact uniform 0–100% gene-mask views per core and epoch, mask base seed
  `2026082401`, excluding model seed from mask derivation;
- deterministic core order seed `2026082402`;
- width 256, eight 32-dimensional heads, FFN/decoder width 1,024, dropout 0.10,
  exact receiver chunking, activation checkpointing, and FP32 attention
  accumulation;
- masked Huber loss (delta 1.0), AdamW (`1e-4`, weight decay `1e-5`), no
  scheduler, AMP, and gradient clipping at 1.0;
- four-rank DDP, seven paired-core optimizer updates and 140 complete-graph
  mask views per global epoch.

No graphless, rewired, additional-seed, or patient-held-out arm is introduced
in this one-run campaign. Those absent controls limit interpretation and are
required before any broader architectural or biological conclusion.

## Literal August 25 stopping and checkpoint policy

The run starts from epoch 0. The minimum budget is 150 global epochs, not a
maximum. Beginning at epoch 150, the training-only equal-core mean masked-Huber
history is audited every 25 epochs over the previous 50 epochs. An audit passes
only under the existing August 25 implementation when relative mean
improvement is at most `0.002` and normalized absolute slope per epoch is at
most `0.0001`. Training stops after two consecutive passing audits. The
earliest possible stop is epoch 175, continuation occurs in fixed 25-epoch
blocks, and there is no scientific maximum epoch cap.

Loss is recorded after every completed global epoch. During training there is
strictly one rolling `checkpoints/latest.ckpt`, atomically replaced at every
epoch boundary. No epoch-numbered archive or best checkpoint is retained. On
successful finalization, the only retained checkpoint is
`checkpoints/last.ckpt`; it represents the last completed plateau-confirmed
epoch and must reload independently.

## Scalar-only gradient diagnostics

At each of the seven optimizer updates, after AMP unscale and the finite check
but before gradient clipping or the optimizer step, rank zero reads the full
FP32 DDP-averaged gradient over all trainable parameters in parameter
registration order. A parameter with `grad=None` contributes a zero-filled
slice so vector alignment cannot drift. Only one-based global-epoch scalar
rows are persisted to `results/gradient_direction_metrics.csv` under schema
`so2_full_gradient_direction_metrics_v1`:

- schema, run ID, model seed, and one-based global epoch;
- trainable parameter count and optimizer updates observed;
- mean, minimum, and maximum pre-clip full-gradient norm;
- mean, median, minimum, maximum, and valid-pair count for cosine between
  consecutive optimizer-step full gradients;
- `epoch_aggregate_gradient_cosine_to_previous_epoch`, comparing the aggregate
  full gradient with the prior uninterrupted epoch; and
- a boolean `resume_boundary_unavailable` flag.

Fresh epoch 1 has six within-epoch consecutive pairs, a blank prior-epoch
aggregate cosine, and `resume_boundary_unavailable=false`. Later uninterrupted
epochs have seven pairs because the first update is compared with the preceding
epoch's last update, plus an available prior-epoch aggregate cosine. The first
completed epoch after a process/checkpoint resume has six pairs, a blank
aggregate cosine, and `resume_boundary_unavailable=true`; the following
uninterrupted epoch returns to seven pairs and an available aggregate cosine.
A zero-norm vector makes its cosine invalid and omitted, so the recorded valid
pair count can be below these expected counts.

The implementation may hold only the bounded in-memory vectors needed for the
current calculation. It must not write gradient tensors, parameter-level
gradients, per-optimizer-step diagnostic files, or gradient-vector checkpoint
payloads. The direction CSV remains separate from the legacy
`results/epoch_metrics.csv` loss/plateau schema. Diagnostics do not affect
loss, gradients, clipping, optimizer steps, plateau decisions, or checkpoint
selection.

## Acceptance, falsification, and stop criteria

Technical acceptance requires all of the following:

- composition and registry validation lock one unique graph block, four calls,
  and all-step weight tying under the dedicated recurrent model family;
- parameter identity, four-call count, and gradient accumulation from every
  call into the same parameters are verified;
- the tied implementation matches a four-block reference with forcibly shared
  weights under FP32/AMP and full-edge/receiver-chunked execution tolerances;
- immutable cohort and graph checksums match the August 25 inputs;
- the four-rank largest-core preflight reports finite loss/gradients, exact DDP
  averaging, measured VRAM, checkpoint reload, and the new config-bound receipt;
- every global epoch has one fsynced loss row and one scalar direction row,
  exactly 14 core visits, 140 views, and seven optimizer updates;
- two consecutive literal August 25 plateau audits pass; and
- the final bundle contains one reloadable `last.ckpt`, no `latest.ckpt` and no
  epoch checkpoint archive, with artifact verification passing.

The exploratory hypothesis is supported for this seed only if technical
acceptance passes, the measured parameter count is below 5,003,016, and the
epoch-150 equal-core loss is at most `0.2536849428713321`. It is descriptively
negative if the valid epoch-150 loss is above that threshold. A single-seed
result inside the margin is not evidence of statistical equivalence.

Immediate failure/stop conditions are architecture/config drift, more than one
unique block, fewer or more than four applications, incomplete shared-gradient
flow, dataset/graph checksum drift, cross-core/self edges, invalid DDP
aggregation, non-finite loss or full gradient, unrecoverable OOM, non-
deterministic epoch-boundary resume, missing/gapped per-epoch records, retained
gradient tensors, vector checkpoint payloads, or per-step files, multiple
rolling/epoch checkpoints, checkpoint reload failure, or less than the
required disk margin. Failure to
plateau is not silently converted to a fixed epoch cap; it requires an explicit
contract amendment or termination recorded as inconclusive/negative.

## Resources and expected artifacts

Production allocation is one four-process NCCL job on physical GPUs 0–3, each
an RTX 3090 with 24 GiB. One complete core is staged per rank. The untied
reference peaked at 15.21 GiB. The recurrent model should reduce parameter and
optimizer-state memory but still executes graph attention four times, so a
large compute-speed improvement is not assumed. Expected wall time is roughly
8.3 hours through epoch 150 plus about 1.4 hours per 25-epoch continuation
block. At least 25 GiB free disk is required.

Expected durable artifacts are:

- resolved config and code/data/environment provenance;
- `results/epoch_metrics.csv` with loss, per-core loss, throughput, VRAM, and
  plateau fields for every epoch;
- `results/gradient_direction_metrics.csv` with one scalar-only gradient
  direction row for every epoch;
- append-only structured metric/status events and stdout/stderr logs;
- preflight receipt
  `state/preflight/so2_14core_recurrent_relative_qkv_ddp4.json`;
- during training only, one atomic `checkpoints/latest.ckpt`;
- after success only, `checkpoints/last.ckpt`, summary, completion marker,
  checksums, and schema-v3 checkpoint catalog entry.

## Reproduction and launch commands

Run from the additive source worktree while keeping runtime state under
`/workspace/BAGM`:

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

Validate the additive config and focused infrastructure:

```bash
/venv/main/bin/python -m pytest -q \
  tests/unit/infrastructure/test_configuration.py \
  tests/unit/infrastructure/test_queue_command.py

/venv/main/bin/python - <<'PY'
from pathlib import Path
from spatial_benchmark.configuration import compose_config
from spatial_benchmark.queueing import command_for_config
config = compose_config(
    Path("configs/experiment/so2_14core_recurrent_relative_qkv_seed0_batch2.yaml"),
    config_root=Path("configs"),
)
print(config["model"])
print(command_for_config(config))
PY
```

Preparation is immutable and should only be run if the checksum-bound
manifests are absent:

```bash
test -f "$BAGM_DATA_ROOT/processed/so2_14core_relative_qkv_v1/manifest.json" || \
  /venv/main/bin/python -u scripts/data/prepare_so2_14core_relative_qkv.py
test -f "$BAGM_DATA_ROOT/processed/so2_14core_relative_qkv_graphs_v1/manifest.json" || \
  /venv/main/bin/python -u scripts/data/materialize_so2_14core_relative_graphs.py
```

After read-only GPU inspection confirms GPUs 0–3 are available, create the
dedicated config-bound preflight receipt:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1,2,3 /venv/main/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  scripts/diagnostics/preflight_so2_14core_relative_qkv_ddp.py \
  --config configs/experiment/so2_14core_recurrent_relative_qkv_seed0_batch2.yaml
```

Register the campaign if it is absent, then enqueue exactly one four-GPU job:

```bash
/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" create-campaign \
  --campaign-id cmp_20260831_so2_14core_recurrent_relative_qkv_seed0_batch2 \
  --name "SO2 recurrent Relative-QKV seed-0 exploratory fit" \
  --scientific-question "Does one Relative-QKV block reused four times retain the fitted SO2 masked-expression objective under the August 25 protocol?" \
  --plan experiments/campaigns/cmp_20260831_so2_14core_recurrent_relative_qkv_seed0_batch2/campaign.yaml \
  --status planned

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" enqueue-experiment \
  --campaign-id cmp_20260831_so2_14core_recurrent_relative_qkv_seed0_batch2 \
  --config configs/experiment/so2_14core_recurrent_relative_qkv_seed0_batch2.yaml \
  --priority 100 --max-attempts 2 --gpu 0,1,2,3
```

The queue-derived child command must remain the tracked four-rank invocation of
`scripts/train/run_so2_14core_relative_qkv.py`; do not launch four independent
jobs. Monitor the fsynced epoch record and the single rolling checkpoint:

```bash
export RUN_ID=r_YYYYMMDDTHHMMSSZ_...
tail -f "$BAGM_SCRATCH_ROOT/active_runs/$RUN_ID/results/epoch_metrics.csv"
tail -f "$BAGM_SCRATCH_ROOT/active_runs/$RUN_ID/results/gradient_direction_metrics.csv"
tail -f "$BAGM_SCRATCH_ROOT/active_runs/$RUN_ID/logs/stdout.log"
find "$BAGM_SCRATCH_ROOT/active_runs/$RUN_ID/checkpoints" -maxdepth 1 -type f -print
```

After completion, verify the final bundle and checkpoint catalog:

```bash
/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" verify-artifacts
/venv/main/bin/python -m spatial_benchmark doctor
```
