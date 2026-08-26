# SO1 cores 1-14 Relative-QKV strict-plateau campaign

Campaign ID:
`cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150`.

This campaign fits one fresh seed-0 Relative-Geometric QKV model to every
routed cell in SO_1 cores 1 through 14. It reproduces the SO2 architecture,
features, graph recipe, ten-view masking, batch-two update semantics, and
four-GPU DDP execution, but it does not copy weights, optimizer state, metrics,
or stopping decisions from an SO2 run.

The campaign is implementation-complete and its immutable SO1 cohort and graph
artifacts have been prepared and independently checksum-verified. Registry
registration is allowed. The GPU preflight, enqueue, and production start remain
gated on the active SO2 recovery releasing all four GPUs.

## Scientific scope

The estimand is held-in masked-expression reconstruction within the fitted
SO1 cohort. All 161,596 routed cells are in the fit role. There is no
validation, test, held-out core, held-out slide, or external cohort. The
stopping signal is training loss, not an estimate of out-of-sample performance.

Permitted claims are restricted to fitted-cohort reconstruction behavior and
predictive dependencies learned within these 14 cores. Core-number routing is
diagnosis-neutral. The cohort can contain mixed or unresolved tissue contexts,
so this campaign does not assert that all cores are normal, cancer, one tissue
state, representative of a patient population, or suitable for clinical or
causal inference.

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

The preparation audit found 205 routed FOVs and zero unmapped SO1 FOVs or
cells. Raw and clinical source assets remain protected; only deidentified
aliases appear in ordinary configuration and result artifacts.

The canonical fit-only split fingerprint is
`a0d2c008ff02471010585a023724f75d6f029d1a81f39f1f2dbba1531091b9a4`.
It is SHA-256 over compact, sorted-key JSON containing the ordered aliases,
dataset ID, fit scope, total cell count, false validation/test-partition flag,
and dataset version. The verified cohort canonical fingerprint is
`e006316e0f04afa645191544bcac8aa64f58c423e755f2f8d79bfd9db68a233d`,
and the verified graph canonical fingerprint is
`5262453fc631c15a66f00f960de2a766a6142a4f1f7644b8b77a8784ec43d3b4`.

## Locked model and data recipe

- Model: Relative-Geometric QKV graph transformer, hidden dimension 256, four
  graph layers, eight attention heads, 32 dimensions per head, 1,024-wide FFN
  and decoder.
- Targets: 1,000 raw biological probe counts transformed with `log1p`, then
  gene-wise standardized using pooled equal-core moments.
- Always-visible covariates: the same 22 morphology/image metadata fields used
  by SO2, with pooled median imputation, missingness indicators, `log1p` where
  defined, and standardization.
- Geometry: 70-dimensional relative positional encoding affects attention
  logits only. Absolute coordinates, FOV/slide/core identity, cell/patient
  identifiers, vendor annotations, and expression-derived QC totals are not
  node inputs.
- Graph: complete per-core radial-stratified kNN, nominal `k=200`, maximum
  radius 500 um, shell quotas 48/64/48/40, deterministic bidirectional union,
  no self loops, no cross-core edges, and no neighbor sampling.
- Masking: ten independently derived uniform-per-cell mask views for every core
  and global epoch. Mask seeds exclude model seed.

## Locked optimization and DDP semantics

One global epoch visits every core once. Each of seven optimizer updates pairs
two complete core graphs. Four ranks divide the 20 pair-view losses as five
views on each rank, and DDP averaging yields the exact equal mean of all 20
losses. Only one complete core graph is staged on each GPU at a time.

Training uses seed 0, AdamW, learning rate `0.0001`, weight decay `0.00001`,
Huber delta 1, gradient clipping at 1, mixed precision after the AMP/FP32
preflight, deterministic execution, NCCL world size four, and physical GPUs
0, 1, 2, and 3. Elastic restarts are disabled.

This is a fresh initialization. No resume checkpoint is configured, and no
SO2 checkpoint is a permitted source.

## Strict training-loss stopping rule

The minimum is 150 global epochs. The first audit occurs after epoch 150, and
subsequent audits occur after fixed 25-epoch continuation blocks. Each audit
uses the last 50 equal-core mean training masked-Huber losses and passes only
when both conditions hold:

1. absolute relative change between the first and second 25-epoch half-window
   means is at most `0.0005`; and
2. the absolute linear slope over the 50 epochs, normalized by the absolute
   50-epoch mean loss, is at most `0.000025` per epoch.

Training stops only after two consecutive passing eligible audits. The earliest
possible stop is epoch 175. A failed audit adds exactly 25 epochs. There is no
fixed epoch-300 target and no maximum scientific epoch cap. Non-finite values
fail the audit. These thresholds are the active stopping rule, not post-hoc or
diagnostic-only annotations. They never select a best checkpoint, and no
validation or test metric participates.

## Checkpoints and live observability

Rank zero appends one row per completed global epoch to
`results/epoch_metrics.csv`, flushes it, and calls `fsync`. The append-only event
log, per-core losses, gradient norms, throughput, ETA fields, peak VRAM, and
plateau fields remain available while training is active.

During training, exactly one atomic latest checkpoint slot is replaced after
each completed epoch; historical periodic checkpoints are not retained. When
the strict rule is confirmed, finalization leaves only
`checkpoints/last.ckpt`, representing the final confirmed plateau epoch.
Final checkpoint reload, checksum, and deterministic replay checks are required
before the run bundle can be considered successful.

## Immutable preparation and readiness gate

The source tree is `/workspace/BAGM-relative-qkv-analysis`; generated runtime
state remains under `/workspace/BAGM`. Preparation writes only additive
processed artifacts:

- cohort: `data/processed/so1_14core_relative_qkv_v1`;
- graphs: `data/processed/so1_14core_relative_qkv_graphs_v1`.

Preparation is idempotently gated on the two manifests:

```bash
cd /workspace/BAGM-relative-qkv-analysis
export BAGM_ROOT=/workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export PYTHONPATH=/workspace/BAGM-relative-qkv-analysis/src

test -f "$BAGM_DATA_ROOT/processed/so1_14core_relative_qkv_v1/manifest.json" || \
  /venv/main/bin/python -u scripts/data/prepare_so1_14core_relative_qkv.py
test -f "$BAGM_DATA_ROOT/processed/so1_14core_relative_qkv_graphs_v1/manifest.json" || \
  /venv/main/bin/python -u scripts/data/materialize_so1_14core_relative_graphs.py
```

Both scripts have finished. All declared 14 aliases, 161,596 cells, zero
unmapped routing, every listed artifact checksum, graph QC, and the completed
cohort-with-graphs manifest were independently verified. The bound SHA-256
values are:

- cohort canonical content:
  `e006316e0f04afa645191544bcac8aa64f58c423e755f2f8d79bfd9db68a233d`;
- cohort manifest file:
  `15f9da492959c35d89020b3956047ec163a5f3537eaaefd5b8a011cb9279b440`;
- graph canonical content:
  `5262453fc631c15a66f00f960de2a766a6142a4f1f7644b8b77a8784ec43d3b4`;
- graph manifest file:
  `754e98fa1b2d8b488892c4effbf095cf26da6d99c927ee45e9ee01c2d64b978f`;
- completed cohort-with-graphs manifest file:
  `e078588668d9b2285db27da6cda8b7f1d144aa055065c3022e43048fd1951596`.

These are SO1-generated values; none is substituted from SO2. Production
enqueue and start remain fail-closed until the four-rank preflight passes.

Register the verified dataset, split, and campaign with the repository CLI:

```bash
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_DATA_ROOT=/workspace/BAGM/data

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" register-dataset \
  --dataset-id cosmx_so1_14core_pooled_fit_v1 \
  --version so1_14core_pooled_fit_v1 \
  --display-name "CosMx SO1 cores 1-14 pooled transductive fit" \
  --protected-source-path "$BAGM_DATA_ROOT/processed/so1_14core_relative_qkv_v1/manifest.json" \
  --raw-fingerprint e006316e0f04afa645191544bcac8aa64f58c423e755f2f8d79bfd9db68a233d \
  --preprocessing-version so1_14core_equal_core_log1p_metadata_v1 \
  --processed-fingerprint 5262453fc631c15a66f00f960de2a766a6142a4f1f7644b8b77a8784ec43d3b4 \
  --sample-count 161596 --graph-count 14 \
  --node-feature-schema expression_mask_permitted_metadata_v1 \
  --edge-feature-schema relative_geometry_logit_bias_70d_v1 \
  --status available --verification-status verified

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" register-split \
  --split-id fit_all_so1_cores_1_through_14_transductive_v1 \
  --dataset-id cosmx_so1_14core_pooled_fit_v1 \
  --dataset-version so1_14core_pooled_fit_v1 \
  --method all_cells_fit_only_transductive --unit spatial_core \
  --fold-count 1 \
  --fingerprint a0d2c008ff02471010585a023724f75d6f029d1a81f39f1f2dbba1531091b9a4 \
  --protected-path "$BAGM_DATA_ROOT/processed/so1_14core_relative_qkv_graphs_v1/cohort_manifest_with_graphs.json" \
  --verification-status verified

/venv/main/bin/python -m spatial_benchmark \
  --database "$BAGM_STATE_ROOT/tracking/bagm.sqlite3" create-campaign \
  --campaign-id cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150 \
  --name "SO1 cores 1-14 Relative-Geometric QKV seed-0 strict-plateau fit" \
  --scientific-question "Does one fresh shared relative-QKV model reach the locked strict fitted training-loss plateau across SO1 cores 1 through 14 after at least 150 global epochs?" \
  --plan experiments/campaigns/cmp_20260826_so1_14core_relative_qkv_seed0_batch2_plateau_min150/campaign.yaml \
  --status planned
```

## Four-rank preflight

Only after all four GPUs are idle and the immutable hashes are bound, run the
bounded DDP preflight:

```bash
cd /workspace/BAGM-relative-qkv-analysis
CUDA_VISIBLE_DEVICES=0,1,2,3 /venv/main/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=4 --max-restarts=0 \
  scripts/diagnostics/preflight_so1_14core_relative_qkv_ddp.py \
  --config configs/experiment/so1_14core_relative_qkv_seed0_batch2_plateau_min150.yaml \
  --output state/preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json
```

The preflight must exercise the real paired-core update, all 20 mask-view
losses, finite forward/backward values, checkpoint save/load, and largest-core
memory safety. It also checksum-verifies the prior shared-implementation
full/chunked and AMP/FP32 equivalence gates; it does not claim SO1 graph byte
identity with that prior core. It is not a production epoch and does not
authorize a scientific claim.

## Supervisor installation and queue safety

The supplied prepare, preflight, and DDP worker services all use
`autostart=false`. Installing or updating them does not prepare data, reserve a
GPU, register a campaign, enqueue a run, or start training. The queue is passive:
an SO1 job may be enqueued after readiness checks while no SO1 worker is
running. The worker wrapper does not provide a GPU-idleness scheduler; start it
manually only after confirming GPUs 0-3 are free and disk free space remains at
least 25 GiB.

```bash
install -m 0755 ops/supervisor/bagm-so1-14core-prepare.sh \
  /opt/supervisor-scripts/bagm-so1-14core-prepare.sh
install -m 0755 ops/supervisor/bagm-so1-14core-preflight.sh \
  /opt/supervisor-scripts/bagm-so1-14core-preflight.sh
install -m 0755 ops/supervisor/bagm-so1-14core-ddp4-worker.sh \
  /opt/supervisor-scripts/bagm-so1-14core-ddp4-worker.sh
install -m 0644 ops/supervisor/bagm-so1-14core-prepare.conf \
  /etc/supervisor/conf.d/bagm-so1-14core-prepare.conf
install -m 0644 ops/supervisor/bagm-so1-14core-preflight.conf \
  /etc/supervisor/conf.d/bagm-so1-14core-preflight.conf
install -m 0644 ops/supervisor/bagm-so1-14core-ddp4-worker.conf \
  /etc/supervisor/conf.d/bagm-so1-14core-ddp4-worker.conf
supervisorctl reread
supervisorctl update
```

Do not issue any `supervisorctl start` command as part of installation. The
explicit order after readiness approval is prepare, hash binding and registry
registration, preflight, enqueue exactly one experiment, then start the
one-shot four-GPU worker.

For the current host, SO2 recovery run
`r_20260826T122252Z_52d16093_s000_f00_a01_a48fd9e2` owns all four GPUs first.
The one-shot handoff coordinator preserves the same order without polling from
an unmanaged shell. It blocks on the repository's global worker lock, then
requires the registered SO2 job and run to be completed, an exact contiguous
epoch 1--300 CSV, a sole loadable `last.ckpt`, successful fixed-budget and
prediction-replay receipts, a clean artifact-verification result, two idle-GPU
samples, and at least 25 GiB free. Only then does it run the SO1 preflight,
enqueue exactly one matching job, release the global lock, start the normal
SO1 worker, and verify epoch 1 in the live CSV plus `latest.ckpt`.

```bash
install -m 0755 ops/supervisor/bagm-so1-after-so2-handoff.sh \
  /opt/supervisor-scripts/bagm-so1-after-so2-handoff.sh
install -m 0644 ops/supervisor/bagm-so1-after-so2-handoff.conf \
  /etc/supervisor/conf.d/bagm-so1-after-so2-handoff.conf
supervisorctl reread
supervisorctl update
supervisorctl start bagm_so1_after_so2_handoff
```

Its durable state is
`/workspace/BAGM/state/handoffs/so2_epoch300_to_so1_plateau_min150.json`.
Any failed scientific, provenance, GPU-idleness, disk, preflight, or queue
gate stops the coordinator without starting SO1.

Once a run ID exists, the live files are:

```bash
export RUN_ID=r_YYYYMMDDTHHMMSSZ_...
tail -f "/workspace/BAGM/scratch/active_runs/$RUN_ID/results/epoch_metrics.csv"
tail -f "/workspace/BAGM/scratch/active_runs/$RUN_ID/metrics/events.jsonl"
tail -f "/workspace/BAGM/scratch/active_runs/$RUN_ID/logs/stdout.log"
```

## Acceptance and failure conditions

Acceptance requires immutable manifest verification, exact resolved-config
validation, focused and regression tests, graph QC, four-rank hardware
preflight, live CSV verification, two consecutive strict passing audits, one
loadable final `last.ckpt`, and a finalized run bundle whose checksums verify.

Any alias/count mismatch, unmapped routing, hash mismatch, cross-core edge,
non-finite loss or gradient, DDP/rank mismatch, AMP equivalence failure, CSV gap
or duplicate, checkpoint-layout violation, or final reload mismatch fails the
attempt. A failed attempt does not justify a claim and must not silently relax
the stopping thresholds or convert the run to a fixed epoch budget.

Materializing the 14 graphs is estimated to require about 10.1 GiB total, or
about 8.1 GiB incremental when reusable C01/C09/C13 caches are valid. Runtime
should be treated as an operational estimate only: approximately 131 seconds
per global epoch on the current four-RTX-3090 host implies about 5.5 hours to
epoch 150, with each additional 25-epoch block adding about 55 minutes.
