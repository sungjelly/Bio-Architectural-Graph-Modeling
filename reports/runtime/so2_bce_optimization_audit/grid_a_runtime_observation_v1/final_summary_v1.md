# Completed grid: independent protocol and runtime audit

All ten registered grid jobs completed successfully. Independent verification passed for all 460 immutable bundle files, eight epochs per run, exact target coverage, the mean validation BCE from epochs 7–8, indexed best-checkpoint checksums, and the resource gates. The terminal endpoint was checked separately from best-checkpoint selection.

Training and validation mask receipts match across all ten jobs at each epoch, including the paired-core reference. Executable sources, frozen task contract, upstream configuration, training-fitted prevalence prior, model seed and parameter count match. All 480 gradient-report rows are finite with positive epoch-mean norm. Epoch-1 block minimum norms include zero because the initial decoder output weight is deliberately zero; later block gradients are nonzero. Final checkpoint files have distinct filesystem inodes.

Worker samples show a shared-cgroup peak of 58.763 GiB and at least 4.768 GiB physical GPU free memory. The independent 30-second observer recorded 57.219 GiB cgroup peak, 20.022 GiB minimum free disk and a 67°C maximum GPU core temperature. Every recorded resource gate passed.

The longer hardware record qualifies the early point observations: 57 of 52113 CPU quota periods were throttled (0.109%), totaling 47746581 aggregate throttled microseconds. This shared-cgroup counter is not elapsed slowdown. Software power-cap flags were active in 24/13/6/23 of 174 samples on GPUs 0/1/2/3. One GPU-2 sample reported software thermal slowdown at 08:22:52 UTC, alongside a 48°C core temperature and 1965 MHz SM clock; hardware thermal-slowdown flags remained inactive. These observations do not establish sustained thermal throttling or exclude brief events.

Mean worker-sampled GPU utilization ranged from 23.3% to 35.3%. Those point samples include all phases while each worker sampler was active. GPU utilization is an execution-efficiency observation, separate from the predictive optimization findings. Runtime differs with optimizer update count, physical device and concurrent work; this grid does not isolate a causal hardware or hyperparameter timing effect.

The separate registered execution profiler occupied GPU 0 from 09:03:08.270564 to 09:11:10.945803 UTC after that grid lane completed. Shared observer data include that occupancy; the grid running-lane field does not. Its dedicated verification belongs to the execution-profile campaign.

Machine-readable details: [final protocol/runtime verification](final_protocol_runtime_verification_v1.json), [per-bundle verification receipts](completed_bundle_verifications.jsonl), [hardware samples](samples.jsonl), [observer completion](completion.json), and [source-backed hot-path review](hot_path_review_v1.md). The hot-path review describes plausible costs; no profiler was attached or training setting changed by this observer. No predictive candidate was selected by this audit.
