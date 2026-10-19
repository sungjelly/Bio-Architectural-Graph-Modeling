# SO2 hurdle count model: GPU resource trial

Phase: complete. Outcome: supported (bounded runtime feasibility only).
This is an exploratory runtime diagnostic,
specified before measuring one-GPU execution. It does not select a new
biological model or establish graph-specific predictive gain.

## Task contract

Objective: run the existing SO2 hurdle count model on one RTX 3090, measure
whether it fits and its useful throughput, and determine whether four
independent experiments can use the four available devices. If one GPU fails
the memory gate, evaluate two GPUs after diagnosing the failure.

Hypothesis: the unchanged 6,160,088-parameter hurdle model, CPU-streamed
geometry, receiver chunking, and sequential two-core gradient accumulation
fit one 24 GiB RTX 3090 with at least 2 GiB measured headroom. Alternatives:
the larger target shard exceeds VRAM, or CPU memory / I/O / thread contention
prevents useful independent-job concurrency. A full training and validation
epoch distinguishes these alternatives better than extrapolating four-rank
memory usage.

Estimand: runtime and numerical correctness of target-isolated partial-gene
raw-count reconstruction. Maximum claim: feasibility and measured speed for
this configuration and dataset. Runtime trials are not scientific replicates;
validation scores here do not support a biological or generalization claim.

Experimental unit: an isolated process/run and device allocation. Observational
units for the unchanged reconstruction task remain cells nested within cores
and donor groups. The inherited split has 208,696 training cells in twelve
cores / six donor groups and 37,367 validation cells in two cores / one held-out
donor group. There is no test cohort.

Baseline and inputs: inherit the resolved scientific configuration of
`r_20260928T021117Z_9ea46874_s000_f00_a01_41b54097`, with its immutable data,
split, graph and train-only preprocessing manifests. Training starts fresh.
Keep raw targets loss-only; no hidden target totals, labels or identifiers may
enter the model. Preserve the two-stream isolation and incoming non-self
edges. Read `docs/legacy_data_guide.md` before local data access.

Keep the original model, seeds, masks, count likelihood, AdamW, mixed precision,
two complete cores per update, and six optimizer updates per epoch. All
training cells must be targeted once per epoch. Use pooled masked-entry loss
across both cores, never an unweighted mean of core means. Changing rank count
changes dropout draws: objective equivalence does not imply identical training
trajectories or permit an identical resume from the four-rank checkpoint.

Primary resource metrics: peak allocated and reserved VRAM (lower is better),
seconds per complete epoch including validation (lower is better), and
completed target cells / second (higher is better). Record process RSS, cgroup
memory, I/O, device utilization, CUDA/framework versions and thread budgets.

Acceptance: finite loss and gradients; exhaustive disjoint target coverage;
same deterministic target masks; correct pooled objective; full fixed-mask
validation; peak allocated and reserved VRAM no greater than 22 GiB and at
least 2 GiB device headroom; no OOM or other-job displacement. Numerical
controls include existing hurdle likelihood/leakage tests and synthetic
unequal-count pooled-gradient equivalence for one, two and four partitions.
No new biological null is needed for this resource-only claim; existing
biological limitations and the need for self/context baselines remain.

Pilot order: fresh single-GPU two-epoch training trial, including every core
and both validation cores. First inspect all devices and actual framework
visibility. Use separate registered run IDs and output directories. If memory
or correctness fails, stop and preserve the failure; diagnose before any
two-GPU fallback or microbatch change. Only consider concurrent or full runs
after this gate passes. Do not choose unspecified scientific variants merely
to occupy GPUs.

If the one-GPU gate passes and measured host memory permits it, run four
isolated two-epoch replicas on physical GPUs 0, 1, 2 and 3. These repeat the
same seed solely to measure concurrent execution; they are not independent
scientific seeds. Each must pass the same correctness and VRAM gates. Record
aggregate cgroup memory and CPU / I/O contention. Stop this task's concurrent
jobs if aggregate cgroup memory exceeds 160 GiB (the limit is 170.9 GiB), and
retain all failures. Compare completed epochs per wall-clock hour and each
job's epoch duration with the isolated pilot; report slowdown even if all
jobs technically fit. The resulting claim is limited to the tested model and
resource configuration, not arbitrary future larger variants.

Resources: four idle RTX 3090 GPUs are available at initial inspection;
PyTorch 2.13.0+cu130 / CUDA 13.0. Begin with physical GPU 0. Host cgroup memory
limit is approximately 170.9 GiB and CPU quota 38.4 cores. Start each independent
process with four intra-op CPU threads. Free disk is approximately 22 GiB;
preserve the original 20 GiB minimum and record actual usage. Pilot outputs
should be small scalar metrics, receipts, logs and provenance; no prediction
matrices, dataset copies or diagnostic checkpoints are required.

Every run must use the local registry and RunArchive, save a resolved config,
code/data/split/environment provenance, append-only metrics, resource receipts,
status and a verified completion marker. Frozen historical contracts, receipts
and successful bundles remain immutable. A bounded pilot is classified as
diagnostic, with no promotion to the curated biological results catalog.

## Execution and verification

The scientific source is the existing `/workspace/BAGM-relative-qkv-analysis`
worktree; the authoritative registry, data and output roots remain in
`/workspace/BAGM`. Run from the source worktree root:

```bash
export BAGM_ROOT=/workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export BAGM_CACHE_ROOT=/workspace/BAGM/cache
export BAGM_EXPORT_ROOT=/workspace/BAGM/exports
export BAGM_REPORT_ROOT=/workspace/BAGM/reports
export BAGM_RESULT_ROOT=/workspace/BAGM/results
export PYTHONPATH=src:.

/venv/main/bin/python scripts/diagnostics/run_so2_hurdle_gpu_trial.py \
  --gpu 0 --epochs 2 --check-only
/venv/main/bin/python scripts/diagnostics/run_so2_hurdle_gpu_trial.py \
  --gpu 0 --epochs 2
```

The launcher holds the shared global-worker lease and an exclusive lock for
the selected physical GPU, checks GPU occupancy and disk headroom, and creates
one registered job/run. It uses the regular worker heartbeat, provenance,
failure and immutable-archive lifecycle. The new diagnostic artifact contract
is restricted to this pinned source configuration and two-epoch protocol.
Concurrent pilots use the same command in separate supervised processes with
`--gpu 0`, `--gpu 1`, `--gpu 2` and `--gpu 3`.

Before launch run focused masking, hurdle model/loss and new runtime-contract
tests. Full repository verification uses:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /venv/main/bin/python \
  -m pytest tests -q -p no:cacheprovider
/venv/main/bin/python -m spatial_benchmark doctor
```

Initial numerical validation: 57 existing hurdle/masking/runner/config tests
passed, plus three independent model prediction and pooled-gradient
equivalence cases for one, two and four target partitions. Dropout was disabled
in the equivalence controls. The read-only launcher admission check passed
against the real immutable source bundle and shared registry.

The pre-launch `doctor` check reports SQLite integrity `ok` and valid example
configs, but exits nonzero for three existing operational issues: 46 checkpoint
artifacts versus 45 indexed checkpoints; two missing registered paths belonging
to the older six-core workflow; and 21.462 GiB free disk versus the generic
25 GiB floor. This pilot independently verifies its SO2 source bundle and
inputs and enforces the prespecified 20 GiB floor. It does not repair unrelated
registry records or delete historical outputs to satisfy the generic floor.

Completion requires a registered, verified bounded training run (or a
documented resource/correctness failure), measured memory and throughput,
an honest concurrency recommendation, exact reproduction commands, and a
review of final Git status. A full scientific run is a separate lifecycle
stage and cannot be called complete while training remains active.

## Completed measurements and decision

All five registered trials completed successfully on 2026-10-01. Each trained
two full epochs, with six two-core updates per epoch and every one of the
208,696 training cells and 37,367 validation cells visited once. All four graph
blocks, the whole model and dispersion parameters had finite positive gradient
norms. Fixed validation receipts were identical across epochs; training masks
changed by epoch. The isolated and parallel trials have exactly equal recorded
losses, final validation metrics and gradient summaries at corresponding
epochs. These are repeated seed-0 resource trials, not five scientific seeds.

| Allocation | GPU | Epoch 1 / 2 (seconds) | Mean epoch | Estimated 180 epochs |
|---|---:|---:|---:|---:|
| Isolated | 0 | 195.3 / 189.1 | 192.2 s | 9.61 h |
| Four concurrent jobs | 0 | 202.8 / 193.0 | 197.9 s | 9.90 h |
| Four concurrent jobs | 1 | 221.0 / 216.7 | 218.8 s | 10.94 h |
| Four concurrent jobs | 2 | 192.3 / 184.2 | 188.2 s | 9.41 h |
| Four concurrent jobs | 3 | 192.9 / 189.6 | 191.2 s | 9.56 h |

Every trial peaked at 16.470 GiB allocated and 19.377 GiB reserved. Five-second
device samples reached 20.195 GiB physical usage with 3.365 GiB minimum free.
The conservative allocator/physical-memory gate passed. The concurrent jobs'
shared cgroup samples peaked at 54.53 GiB; the independent scheduler monitor
observed 54.54 GiB. Both are below the 160 GiB stop threshold and 170.9 GiB
limit. All four processes exited zero, with no resource-stop event or OOM.
Each completed bundle has 33 checksum-verified files and an `_SUCCESS` marker;
no trial retained checkpoints or prediction matrices. All trial GPU processes
have exited.

Decision: one RTX 3090 per copy is sufficient for this unchanged hurdle model,
and four such copies can train concurrently. Two GPUs per copy are not required
for this measured memory requirement. Reserve one whole card per experiment;
the measured headroom does not support packing a second copy on the same card.
Larger architectures or different graph/core sizes need a new resource pilot.

The mean parallel epoch-time ratio was 1.04 relative to isolated GPU 0. Only
GPU 0 was tested in isolation: other cards' ratios also include possible
device differences. Aggregate throughput was 3.37 times the isolated trial
over the combined training/validation window, including staggered starts.
Two epochs do not measure converged accuracy or guarantee a full-run duration.
The 180-epoch estimates exclude initialization, checkpoints and finalization;
the isolated worker spent approximately 129 seconds outside its two epoch
timers. Dropout trajectories differ from historical four-rank training.

The verified cross-run report is
`BAGM_REPORT_ROOT/runtime/so2_hurdle_gpu_scaling/v1/report.md`, with complete
scalar tables, source checksums, limitations and a producer snapshot. Reproduce
it into a fresh report root using the environment above and:

```bash
/venv/main/bin/python scripts/analysis/report_so2_hurdle_gpu_trial.py \
  --isolated-run r_20261001T053158Z_94b9eb4c_s000_f00_a01_80f17c59 \
  --parallel-runs \
    r_20261001T054148Z_94b9eb4c_s000_f00_a01_74c220a4 \
    r_20261001T054148Z_94b9eb4c_s000_f00_a01_fa18c372 \
    r_20261001T054148Z_94b9eb4c_s000_f00_a01_0c056643 \
    r_20261001T054148Z_94b9eb4c_s000_f00_a01_3a90d363
```

The launcher deliberately supports this two-epoch diagnostic only. A full
training campaign requires a separately documented runtime configuration,
checkpoint/resume and selection contract; the historical four-rank launcher
and its frozen preflight have not been redefined. No long training campaign
was started or declared complete by this resource study.

## Final verification and remaining limitations

Focused checks passed: 57 existing hurdle/masking/runner tests, three new
one/two/four-partition model-equivalence cases, 20 worker tests, and 111
launcher/archive/registry/config tests (these groups may overlap). The full
repository run finished with 1,793 passed, 35 failed, four skipped and two
deprecation warnings in 472.47 seconds. Every failure was traced to unavailable
legacy local inputs, with no failure in the new runtime workflow: 29 adjacency
runner cases, one adjacency analysis case, one clinical-routing-map case, one
multiscale geometry case, two MyJju locked-config cases and one missing external
MyJju source script. Three representative failures reproduced independently.
Those unrelated inputs were not rewritten or fabricated. The pre-launch
`doctor` issues described above remain separate from the successful SO2 input,
runtime and archive checks.

This bounded resource result is not promoted to the curated biological results
catalog. It supplies no evidence for graph-specific predictive gain, independent
patient replication, biological mechanisms or causality. The next scientific
step is to specify the independent full experiments and their controls; the
next infrastructure step for those runs is a full single-GPU runtime profile
bound to this measured resource evidence.
