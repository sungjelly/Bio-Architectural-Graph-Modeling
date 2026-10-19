# SO2 BCE execution profile

Phase: complete. Outcome: inconclusive. Exploratory runtime study, declared after
observing roughly 25–32% GPU utilization during the frozen optimization grid.
This campaign does not modify or select that grid's scientific settings.

## Frozen task contract

Objective: obtain a short, reproducible CPU/CUDA execution profile of actual
single-GPU target-window BCE training. Deliver one registered diagnostic run,
aggregate operator and named-range timings, adjacent unprofiled update times,
resource measurements, provenance, ordinary verified prediction artifacts and
an indexed checkpoint. No GPU run starts until the owning grid lane has ended
and the parent explicitly allocates its idle device.

Question: which observed costs plausibly explain low useful GPU utilization?
Hypothesis: serial host checks, synchronization, receiver-chunk transfers and
launches contribute materially. Alternatives: repeated full-core encoder/KV
work, backward recomputation, small kernel arithmetic, or shared-host effects
dominate instead. CPU/CUDA operator/range measurements discriminate among
these explanations; sampled nvidia-smi utilization alone does not.

The experimental unit is one canonical optimizer update, nested in a core and
one diagnostic execution. This is an execution-cost estimand, not patient
replication or a biological effect. Maximum claim: measured operator/range
costs for the sampled early updates and the observed one-epoch run. Three
windows do not characterize every core, later epoch or future optimization.
CPU and CUDA work can overlap; inclusive nested ranges and exclusive operator
totals must never be added into a false wall-time percentage.

Use completed target-window smoke
`r_20261001T073730Z_b3feecba_s000_f00_a01_42b8b0fd` as the immutable parent.
Require its successful bundle, resolved scientific configuration, checkpoint
catalog, own source checksums and the unchanged frozen training fingerprint.
Capture and verify the new profiler and launcher SHA-256 values separately in
the resolved configuration. Preserve data/graph/split/environment provenance.
Input handling follows `docs/legacy_data_guide.md`; raw and clinical inputs are
immutable and remain local.

Keep target cap 1,024, receiver chunks 128, seed 0, LR 0.0001, dropout 0.1,
weight decay 0.00001, AdamW, clipping, mixed precision, masks, target order,
fresh learned source embeddings, graph checks, gradient checks/diagnostics,
resource checks and durable progress writes exactly as in the parent.
Run one complete epoch: 208,696 training targets, 209 optimizer updates, then
the unchanged 37,367-target fixed validation evaluation and checkpoint replay.
Minimum epochs zero, maximum one and patience two are a separately tested
diagnostic stopping contract. Keep normal predictive archive requirements;
do not masquerade as the narrow existing HNB or niche analysis protocols.
The retained best checkpoint is the sole one-epoch candidate, not a selected
scientific result. Monitored validation BCE is archival metadata only.

After two warmup updates, profile updates 3–5 with torch.profiler CPU+CUDA.
Record adjacent unprofiled update timings. Use static named ranges around
existing callables for encoder, source projection, receiver chunks, CPU graph
and target checks, gradient diagnostics, staging/transfers, progress/fsync,
validation and checkpoint operations. Wrappers preserve arguments, returned
objects, exceptions and RNG behavior; no arithmetic or optimizer behavior is
changed. Do not globally monkeypatch tensor scalar conversion. Its operators
remain visible in the aggregate profiler table.

Disable shape, stack, module and tensor-memory tracing. Save aggregate
operator/count/self-CPU/self-device measurements and aggregate named ranges
only; never export raw Chrome traces, tensor values or cell/patient IDs.
Full-epoch perf_counter ranges add no new CUDA synchronization and measure
observed blocking wall time, not isolated GPU duration. Record profiler
startup/finalization time separately. Adjacent updates have different inputs
and weights; their timing ratio is descriptive, not an exact causal overhead
estimate. No reset-state A/B is attempted because restoring every optimizer,
tracker, scaler and RNG side effect would add unnecessary correctness risk.

Primary descriptive measures: observed wall milliseconds per update and
operator/range self CPU/device microseconds; lower execution time would be
better in a future controlled implementation comparison. No optimization
candidate is being compared here. Controls: unchanged source fingerprint and
update body, CPU wrapper-equivalence tests, real min0/max1 controller test,
exact coverage/mask receipts, finite gradients, selected-state replay and
comparison with the parent's epoch-one scalar evidence. Scalar comparison
uses absolute tolerance 0.000001; integer counts and mask fingerprints must
match exactly. A mismatch invalidates equivalence and requires diagnosis.

Bound profiling to three completed updates and at most 90 seconds between
profile start and the next update boundary. If the time cap is reached, stop
profiling at that boundary, retain the observed partial profile and mark its
coverage limited; continue the canonical epoch. Expected total duration is
5–7 minutes, with a 10-minute soft budget reported explicitly if exceeded.
Do not remove validation/checks or change batches to meet a timing target.
Fail on OOM/nonfinite state/coverage drift, fewer than 5 GiB free disk,
GPU headroom below 2 GiB, allocated/reserved VRAM over 22 GiB, shared cgroup
memory at or above 160 GiB, stale parent/source hashes, or unavailable CUDA
profiling. Preserve failures rather than retrying blindly.

Use one safely idle GPU with four CPU threads. Respect the existing shared
global lease, exclusive per-GPU lock, child ownership and heartbeat rules.
All active output lives in registered `scratch/active_runs/<run_id>/`; publish
only verified immutable bundles under `artifacts/runs/YYYY/MM/<run_id>/`.
No edits to `src/spatial_benchmark`, `scripts/train` or `scripts/diagnostics`
are permitted. New instrumentation and launcher code live under
`scripts/analysis`, outside the frozen grid fingerprint.

Completion requires focused tests, aggregate-only output checks, unchanged
training fingerprints, a complete canonical epoch and validation/replay,
normal archive/checkpoint verification, reviewed timing interpretation and an
updated execution log. A null or inconclusive profile is a valid result; no
speedup is claimed without a subsequent declared equivalent A/B experiment.

## Execution and verification

The frozen contract is `frozen_task_contract.md`, created before implementation
from the text above this section. The guarded parent configuration is
`12bc44ce0d4e56a3864266c90c4233276967587c84627893f4e2fd6edaff9b31`;
its unchanged training-source fingerprint is
`314f9007ce06b11131616b9ac2919e33ca8c29745bfbffa0971be273f4fadb1f`.
The launcher records and verifies all three new analysis script checksums and
physically preserves them under `provenance/source_snapshot/scripts/analysis/`.

Run from the source worktree using the established shared roots:

```bash
export BAGM_ROOT=/workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_REPORT_ROOT=/workspace/BAGM/reports
export BAGM_RESULT_ROOT=/workspace/BAGM/results
export BAGM_CACHE_ROOT=/workspace/BAGM/cache
export BAGM_EXPORT_ROOT=/workspace/BAGM/exports
export PYTHONPATH=src:.
/venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_so2_bce_execution_profile.py tests/unit/infrastructure/test_so2_bce_execution_profile_launcher.py
/venv/main/bin/python scripts/analysis/launch_so2_bce_execution_profile.py --gpu 0 --check-only
/venv/main/bin/python scripts/analysis/launch_so2_bce_execution_profile.py --gpu 0 --attempt 1
```

The final launch command is for the parent-allocated idle GPU 0 only, after its
grid lane ends. `--check-only` verifies parent/configuration/provenance and
device admission without registering or launching a run. The registered
launcher supervises one-rank torchrun, holds the existing shared global and
exclusive device locks, preserves failures, publishes a verified immutable
bundle and indexes its best checkpoint. There is no independent full-training
profile in this bounded diagnostic campaign.

Expected new outputs inside the ordinary predictive bundle:

- `diagnostics/execution_operator_profile.json`: aggregate CPU/CUDA operators
  and named ranges for updates 3–5, or explicitly limited capture.
- `diagnostics/execution_profile_summary.json`: full-run named blocking wall
  times, per-update aggregate timing, adjacent-window comparison and parent
  epoch-one equivalence checks; no target row ranges or identifiers.
- `diagnostics/execution_profile_preflight.json` and
  `diagnostics/execution_profile_resource_samples.jsonl`: correctness and
  measured resource gates through validation and selected-state replay.
- `results/epoch_metrics.csv`, `results/execution_profile_endpoint.json`,
  validation predictions, `checkpoints/best.ckpt`, normal provenance and
  `_SUCCESS`: one-epoch predictive lifecycle evidence. The endpoint helper's
  reusable audit schema is retained; the resolved campaign, runtime profile
  classification and one-epoch budget distinguish this diagnostic execution.

Current status: the registered one-epoch profile is complete and independently
verified. The reviewed human report is version `v2`; `v1` is preserved and
superseded as explained below. Observed costs narrow follow-up priorities but
do not identify a dominant causal bottleneck or establish an achievable
speedup or a biological conclusion.

Timing interpretation: each core's first update interval starts after
`core_started`, excluding its mask realization and full-core staging; those
have separate named wall ranges. Profiler startup and finalization are also
excluded from update intervals and reported separately. The summary's
`elapsed_through_final_artifacts_seconds` ends before subsequent archive
publication and checkpoint indexing. None of these measures alone is a full
end-to-end uninstrumented runtime.

## Human report reproduction

The analysis-only builder accepts a registered completed profile bundle; it
does not read active metrics or perform training. With the same path overrides,
substitute the immutable profiling run ID after successful publication:

```bash
/venv/main/bin/python scripts/analysis/report_so2_bce_execution_profile.py --run-id <PROFILE_RUN_ID> --version v1
/venv/main/bin/python -m pytest -q tests/unit/infrastructure/test_so2_bce_execution_profile_report.py
```

Output is exclusively created under
`reports/runtime/so2_bce_execution_profile/<PROFILE_RUN_ID>/v1/`, with Markdown,
JSON, complete aggregate CSV tables, PNG/SVG figures, producer snapshots and
checksums. Existing versions fail closed. The builder rechecks both successful
bundles, registry/catalog/checkpoint/replay consistency, preserved source
snapshots, parent-epoch equivalence, capture windows and physical resource
samples. It validates stored source evidence without comparing against live
training source files.

Operator exports retain `device_type`. CPU self time, device time attributed
to CPU operators, and CUDA kernel/copy candidates have separate tables and plots.
The latter two can describe overlapping work; neither is added to the other or
converted into a GPU wall-time fraction. Full-run named wall ranges remain
nested and nonadditive. The report makes no validated speedup claim.
Long static template kernel names use a bounded display label and retain the
SHA-256 of the full name, with explicit truncation/normalization flags. Figure
labels keep a short checksum suffix when shortened; no tensor metadata is
retained by this label handling.

## Execution log

At 09:03 UTC on 2026-10-01, GPU0 had completed both of its planned grid jobs
and was idle. The profiler admission check verified that device, the registered
immutable parent, unchanged training sources, all three instrumentation hashes,
and the 5 GiB free-disk floor. Supervisor `bagm_so2_bce_execution_profile`
launched registered run `r_20261001T090308Z_b64e2ba9_s000_f00_a01_286bb53e`.
The other three GPUs continue the optimization grid; their shared-host load is
part of the recorded runtime context. No existing job was displaced.

The run completed successfully at 09:11 UTC. Independent post-run verification
checked the successful immutable bundle, registry lifecycle, one verified best
checkpoint catalog entry, checkpoint checksum and fixed-validation replay.
The single epoch contains all 208,696 training targets exactly once in 209
updates and all 37,367 validation targets. Parent epoch-one mask/order receipts
and integer counts match exactly. All six checked scalar differences are
**zero**, including validation BCE `0.24684311676688203`. Weight-state equality
against the parent was not separately tested; no scientific setting is selected
from this diagnostic result.

The complete operator capture covers updates 3–5 and 3.930396 s. Observed
profiled update intervals average 1.310120 s; adjacent unprofiled updates
average 0.953679 s (descriptive ratio 1.373754). Different windows and changing
weights prevent a causal overhead estimate. Profiler startup took 0.043183 s;
stopping and aggregate processing took **56.299073 s** outside those update
intervals. The instrumented epoch took 320.029 s and the trainer reported
465.950 s through its run lifecycle. These are not uninstrumented throughput
estimates.

Selected measurements from the three-update capture:

| Recorded event | Calls | Observed time and scope |
|---|---:|---|
| `cudaLaunchKernel` | 57,423 | 657.894 ms self CPU |
| `cudaMemcpyAsync` | 1,875 | 265.504 ms self CPU |
| Pageable host-to-device copy | 291 | 202.993 ms CUDA event duration |
| `aten::item` | 1,338 | 22.950 ms inclusive CPU, overlapping scalar-conversion children |
| `cudaStreamSynchronize` | 1,122 | 17.533 ms self CPU |

These are separate attribution views and cannot be added into a wall-time
percentage. Large kernel-launch counts and transfer activity motivate a
controlled investigation of receiver-chunk granularity and immutable graph
staging. They do not prove how much idle time either causes or how much a
change would save.

Full-epoch named blocking wall observations include target mask realization
27.882 s, CPU graph-order checks 8.245 s, finite-gradient checks 3.715 s,
receiver-chunk forward/recomputation 84.100 s and graph-layer scopes 68.438 s.
The latter scopes overlap each other and enclosing forward execution.
Progress write/stdout scopes measured 0.751 s and their nested fsync scopes
0.725 s; this run does not support those writes as the dominant runtime cost.
Asynchronous encoder/projection wall ranges are not isolated GPU durations.
Data loading took 123.140 s before training and is a separate startup cost.

Resource gates passed: peak allocated/reserved memory 2.203/2.521 GiB;
maximum sampled physical GPU use 3.352 GiB; conservative headroom 20.208 GiB;
maximum shared cgroup memory 55.735 GiB. The physical measurements are sampled,
not continuous peak observations. Hardware was GPU0, NVIDIA RTX 3090, Torch
2.13.0+cu130, CUDA 13.0, with four CPU threads.

## Verified reports and correction

The first published report `v1` was preserved after visual inspection revealed
that `DistributedDataParallel.forward` and `Optimizer.step#AdamW.step` can be
CUDA-typed annotations rather than native kernels. The analysis reporter was
corrected without changing any training/instrumentation source or finalized
bundle. Its 13 focused tests pass, including the annotation regression.

The reviewed report is
`reports/runtime/so2_bce_execution_profile/r_20261001T090308Z_b64e2ba9_s000_f00_a01_286bb53e/v2/report.md`.
All three figures were inspected and report/source/checkpoint checksums
reverified. Version `v2` separates known annotation-name families from other
CUDA activity candidates and explicitly records the limitation: the original
aggregate export did not save `is_user_annotation`, so name-based grouping
does not establish kernel occupancy. No CUDA annotation duration is treated
as native execution or utilization.

Version `v2` manifest SHA-256:
`ba1b30f82531276d52d1ae3dae91de6b0698bed2e71f25d8235b5a571dfa8fe0`.
Summary SHA-256:
`5cd79e3500333b2d9dbfa28882ebe335b7aafcc18df144e1085cd5b994dda1ad`.
The retained best checkpoint SHA-256 is
`ade29ac02a69291fa607ee141850b6b2a1fb07574ad80e12431b16803820eaa1`.

Exact completed-report command, using the path overrides above:

```bash
/venv/main/bin/python scripts/analysis/report_so2_bce_execution_profile.py --run-id r_20261001T090308Z_b64e2ba9_s000_f00_a01_286bb53e --version v2
```

This exact version now exists and is immutable; later reproductions must use
a new report version. The next discriminating test would compare one declared
implementation change with a matched starting model/optimizer/RNG state,
exact mask/coverage and numerical-equivalence checks, and measured throughput.
Potential receiver-chunk or static-graph staging changes require their own
pilot. No such change was implemented or benchmarked here, and no runtime
speedup or biological result is promoted from this diagnostic run.
