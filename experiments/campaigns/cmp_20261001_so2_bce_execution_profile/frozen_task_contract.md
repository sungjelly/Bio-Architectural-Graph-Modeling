# SO2 BCE execution profile

Phase: planned. Outcome: pending. Exploratory runtime study, declared after
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

