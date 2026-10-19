# SO2 hurdle model: isolated and concurrent GPU trials

All five registered two-epoch trials passed coverage, finite-gradient and GPU-memory checks. Four concurrent jobs had an observed mean epoch-time ratio of 1.04× relative to the isolated GPU 0 baseline; aggregate measured epoch throughput was 3.37× the isolated trial.

| Trial | GPU | Epoch 1 / 2 (s) | Mean (s) | Ratio vs isolated GPU 0 | 180 epochs (h) | Allocated / reserved peak (GiB) | Sampled physical used peak (GiB) | Conservative free (GiB) | Peak RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| isolated | 0 | 195.3 / 189.1 | 192.2 | 1.00× | 9.61 | 16.47 / 19.38 | 20.20 | 3.37 | 19.72 |
| parallel | 0 | 202.8 / 193.0 | 197.9 | 1.03× | 9.90 | 16.47 / 19.38 | 20.20 | 3.37 | 19.71 |
| parallel | 1 | 221.0 / 216.7 | 218.8 | 1.14× | 10.94 | 16.47 / 19.38 | 20.20 | 3.37 | 19.71 |
| parallel | 2 | 192.3 / 184.2 | 188.2 | 0.98× | 9.41 | 16.47 / 19.38 | 20.20 | 3.37 | 19.72 |
| parallel | 3 | 192.9 / 189.6 | 191.2 | 0.99× | 9.56 | 16.47 / 19.38 | 20.20 | 3.37 | 19.71 |

Concurrent shared cgroup memory reached a sampled maximum of **54.53 GiB** against the prespecified 160 GiB stop threshold (gate: passed). This aggregate includes shared caches and other processes; per-run cgroup values are not additive.

The four-way training windows overlap for 344.6 seconds. Across the 456.0-second combined window, throughput was 63.2 completed epochs/hour versus 18.7 in isolation.

If later epoch durations remain similar, four 180-epoch jobs would finish in about **10.94 hours** from their training starts. This is an extrapolation from two epochs, not an observed full-training duration; startup, checkpoint writing and finalization add time.

The tested allocation passed the measured two-epoch resource gates. The claim is limited to runtime feasibility. All four concurrent replicas use seed 0; they are not a multi-seed performance result.

## Limitations

- ETA is 180 times the arithmetic mean of two train+validation epoch durations; excludes startup, final publication, checkpoints and unmeasured later-epoch variation or early stopping.
- The aggregate window runs from the earliest first training update to the latest second-epoch completion; it includes inter-job staggering and validation but excludes earlier data loading.
- GPU physical memory and utilization are sampled every five seconds, with additional CUDA free-memory measurements at phase boundaries; unsampled peaks can be missed.
- Mean sampled GPU utilization uses the entire resource-sampling period, including data loading and initialization; it is not a training-window-only utilization estimate.
- Only GPU 0 has an isolated baseline. Ratios for GPUs 1–3 conflate concurrency with possible device differences, so their full differences cannot be attributed to contention.
- Cgroup memory is a shared container measurement including caches and other processes; values across runs are never summed. Historical memory.peak is not used.
- These fresh, same-seed resource replicas have no checkpoints or prediction matrices and establish neither converged accuracy nor independent biological replication.
- One-rank and historical four-rank training preserve the pooled objective but change dropout draws; identical trajectories or checkpoint-resume equivalence are not claimed.
- No historical four-GPU timing is included in the controlled isolated-versus-concurrent comparison.

## Immutable sources

Every bundle was checksum-verified before reporting. Full source hashes, windows and measurements are in `summary.json` and `table.csv`.

- isolated, GPU 0: `r_20261001T053158Z_94b9eb4c_s000_f00_a01_80f17c59`
- parallel, GPU 0: `r_20261001T054148Z_94b9eb4c_s000_f00_a01_74c220a4`
- parallel, GPU 1: `r_20261001T054148Z_94b9eb4c_s000_f00_a01_fa18c372`
- parallel, GPU 2: `r_20261001T054148Z_94b9eb4c_s000_f00_a01_0c056643`
- parallel, GPU 3: `r_20261001T054148Z_94b9eb4c_s000_f00_a01_3a90d363`
