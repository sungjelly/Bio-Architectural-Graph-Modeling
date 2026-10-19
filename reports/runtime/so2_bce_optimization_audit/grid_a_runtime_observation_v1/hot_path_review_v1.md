# Read-only training hot-path review

GPU utilization and predictive optimization are separate questions. These
observations identify possible execution overhead; they do not establish that
hyperparameters or the predictive model are adequate or inadequate. No profiler
was attached, and no runtime setting or training source was changed.

The observed B512 device ordering reverses between epochs one and two. Physical
disk read counters did not grow during the initial sampled training interval.
The first supplemental cgroup intervals used about 5.5–6.1 CPU cores in total,
with no quota-throttled periods or microseconds. Observed clocks were stable and
thermal, power-cap and hardware-slowdown flags were inactive. This evidence
argues against sustained physical disk reads, shared CPU quota throttling, or
thermal throttling during those intervals. It does not exclude CPU serial work,
synchronization, memory bandwidth, launch overhead or transient contention.
The longer observation log remains the source for later intervals.

The optimizer batch cap (512, 1,024 or 2,048 target cells) differs from the
receiver execution cap (128 receivers and at most 50,000 edges per chunk).
Increasing the optimizer batch does not turn each graph layer into one large
kernel. The four blocks still execute a Python loop over receiver chunks.

| Plausible cost | Frozen source location | What the code establishes |
|---|---|---|
| Chunk staging and kernel launches | `target_isolated_negative_binomial.py:192`, `:278`, `:285`, `:315`, `:423` | Each of four blocks iterates receiver chunks, transfers CPU source/receiver indices and geometry to GPU, and uses activation checkpointing during training. Recompute adds arithmetic; staging/launches may limit overlap. |
| Repeated complete-core source work | `target_isolated_negative_binomial.py:229`, `:401` | Every optimizer window recomputes the full clean source encoder and each block's source K/V projections. Learned embeddings must be refreshed after parameter updates. Smaller optimizer windows incur this work more often. |
| Repeated CPU graph checks | `target_isolated_negative_binomial.py:169`, `:373`; `relative_qkv_graph_transformer.py:755` | Full edge bounds/self/sortedness checks execute per forward. Receiver pointers are cached at `relative_qkv_graph_transformer.py:841`; geometry is shape-checked, not fully scanned for finiteness in this path. |
| Scalar synchronization and diagnostics | `run_so2_geometry_modulated_nb.py:1128`; `gradient_direction_observability.py:347`, `:396`, `:511`, `:673`; `run_so2_target_minibatch_binary_expression_bce.py:192`, `:204`, `:210`, `:212`, `:222` | Per-parameter and full-vector checks, full/block/head norms and cosines, scalar conversion and explicit CUDA synchronization require completed GPU results. Gradient vectors remain on GPU; scalar reads cross to the host. |
| Target masks and repeated correctness checks | `run_so2_target_isolated_nb.py:1141`; `target_isolated_negative_binomial.py:150`; `binary_expression.py:66`, `:115` | CPU mask copies transfer per optimizer window; masked/visible equality is checked in both runner and model. BCE checks and positive-count `.item()` add scalar dependencies. Per-target CPU mask generation also occurs once per core/epoch (`target_cell_masking.py:330`). |
| Durable progress and resource observation | `run_so2_target_minibatch_binary_expression_bce.py:112`, `:224`; `so2_hurdle_gpu_trial_worker.py:72`, `:126`, `:152` | Each update reads CUDA/proc/cgroup counters and writes/fsyncs progress, then flushes stdout. `nvidia-smi` runs on a separate sampler thread every five seconds; it is not invoked once per optimizer update. |

All source filenames are under `src/spatial_benchmark/`, `scripts/train/`, or
`scripts/diagnostics/` as recorded in `hot_path_source_receipt_v1.json`. The
reviewed files match the immutable smoke run's source checksums.

These are code-established operations, not measured shares of runtime. A later,
separately declared profiling experiment could measure staging, host checks,
kernel launches, forward/backward, optimizer and diagnostics at fixed weights
and inputs. Any proposed change would need correctness/equivalence tests and a
new timing baseline. The current screen must finish with its frozen execution
settings so its comparisons remain interpretable.
