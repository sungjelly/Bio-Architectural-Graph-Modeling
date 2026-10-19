# BCE execution-cost profile

Run `r_20261001T090308Z_b64e2ba9_s000_f00_a01_286bb53e` completed the unchanged one-epoch workload (209 updates), fixed validation and selected-checkpoint replay. Parent epoch-one counts and masks match exactly; six scalar metrics match within 0.000001. This is an exploratory execution-cost observation, not an optimization comparison or a biological result.

The study asks which measured costs plausibly explain low useful GPU utilization. Alternatives include serial host checks and synchronization, receiver-chunk transfers and launches, repeated full-core encoder/projection work, and backward recomputation. Profiler overhead, these particular early windows and concurrent shared-host load also affect observed timing; this study does not isolate their causal contributions.

![Separate CPU and device attribution views](operators.png)

Capture covered updates [3, 4, 5] (complete three-update capture). Mean profiled interval: **1.310 s**; adjacent unprofiled interval: **0.954 s**; descriptive ratio **1.374×**. Profiler startup took 0.043 s and stopping/aggregation took 56.299 s, separately from update intervals. Different windows and changing weights prevent a causal overhead estimate.

## CPU operator self time

| Operator | Calls | Milliseconds |
|---|---:|---:|
| `cudaLaunchKernel` | 57423 | 657.894 |
| `cudaMemcpyAsync` | 1875 | 265.504 |
| `aten::fill_` | 22806 | 191.673 |
| `aten::mul` | 5700 | 156.733 |
| `aten::empty` | 11748 | 131.467 |
| `aten::mm` | 1788 | 130.610 |
| `aten::_index_put_impl_` | 960 | 94.074 |
| `NativeDropoutBackward0` | 297 | 94.038 |

## Device time attributed to CPU operators

| Operator | Calls | Milliseconds |
|---|---:|---:|
| `aten::copy_` | 5367 | 274.739 |
| `aten::mul` | 5700 | 180.842 |
| `aten::_index_put_impl_` | 960 | 102.813 |
| `aten::fill_` | 22806 | 83.082 |
| `aten::div` | 1314 | 43.685 |
| `aten::sum` | 822 | 43.220 |
| `aten::gather` | 1344 | 39.930 |
| `aten::mm` | 1788 | 33.232 |

## CUDA activity candidates: kernels and copies

| Operator | Calls | Milliseconds |
|---|---:|---:|
| `Memcpy HtoD (Pageable -> Device)` | 291 | 202.993 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)` | 1089 | 84.675 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)` | 10878 | 59.546 |
| `void (anonymous namespace)::indexing_backward_kernel<float, 4>(long const*, long const*, float const*, float*, long, long, long, long, bool)` | 384 | 55.943 |
| `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 2ul>)` | 1893 | 51.500 |
| `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` | 1344 | 39.930 |
| `void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::… [sha256:a87ba9565acecec8]` | 636 | 38.267 |
| `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4, 4>)` | 414 | 37.638 |

## CUDA-typed annotation candidates: separate, overlapping scopes

| Operator | Calls | Milliseconds |
|---|---:|---:|
| `execution/receiver_chunk_and_checkpoint_recompute` | 192 | 1567.208 |
| `DistributedDataParallel.forward` | 3 | 1348.118 |
| `execution/graph_layer_with_chunk_staging` | 12 | 1139.292 |
| `execution/finite_gradient_checks` | 3 | 72.581 |
| `execution/gradient_block_diagnostics` | 3 | 37.546 |

Device type alone does not identify a native kernel. In this retained capture, `DistributedDataParallel.forward` and `Optimizer.step#AdamW.step` are CUDA-typed annotations, alongside the explicit `execution/` ranges. The report separates those known annotation-name families from remaining CUDA kernel/copy candidates. The original aggregate export did not retain the profiler's `is_user_annotation` field, so this classification uses static names and does not establish kernel occupancy.

These tables are separate attribution views. CPU operators can carry device time also represented by CUDA activity rows; nested annotations overlap; CUDA activity may overlap CPU work or other CUDA activity. No rows are summed into a GPU wall-time fraction or pie chart. The scalar/synchronization/copy candidate table is an operator-name filter for follow-up inspection, not a proven bottleneck classification.

Long static kernel names use bounded display labels with full-name SHA-256 values in the aggregate table. Shortened figure labels retain a checksum suffix to distinguish otherwise similar template names. No tensor values, shapes or raw traces are included.

![Named blocking wall ranges](wall_ranges.png)

Named ranges cover initialization, training, validation and final artifact preparation. They measure observed blocking wall duration without adding CUDA synchronization. Parent and child ranges overlap. Receiver-chunk calls include forward execution and checkpoint recomputation. Core-first update intervals exclude mask realization and staging, which have separate named ranges.

![Observed update intervals](update_times.png)

The epoch took 320.03 s; the trainer's recorded run duration was 465.95 s. The observation clock reached 466.26 s through final artifact preparation; this excludes subsequent archive publication/indexing. The 600 s soft observation budget was not exceeded. These clocks cover different scopes and do not estimate an uninstrumented epoch.

Resource and correctness gates passed: allocated/reserved peaks 2.203/2.521 GiB; sampled physical GPU use 3.352 GiB; sampled physical free minimum 20.209 GiB; conservative headroom 20.208 GiB. Maximum observed shared cgroup memory was 55.735 GiB. Physical samples are not continuous peak measurements. Device: NVIDIA GeForce RTX 3090; physical GPU 0; Torch 2.13.0+cu130, CUDA 13.0.

Maximum defensible claim: these are measured costs of a few early updates and named scopes in one instrumented execution. They do not establish every core's typical cost, the fraction of GPU idle time explained, an attainable speedup, eventual predictive superiority, graph-specific gain or biological mechanism. A proposed implementation change needs a separate state-controlled equivalence and throughput comparison. Complete aggregate tables and checksum-bound sources accompany this report; no raw traces or cell/patient identifiers are exported.
