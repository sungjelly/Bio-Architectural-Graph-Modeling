# Grid A runtime observations

Read-only 30-second hardware and shared-cgroup snapshots supplement each worker's existing five-second resource log. Observation began after first-wave epoch one. No affinity, thread count, GPU clock, allocation or training setting was changed.

`metadata.json` records scope and limitations; `samples.jsonl` records device utilization/clocks/power/temperature/throttle flags, shared cgroup CPU counter deltas and disk/memory. Timing remains confounded by physical device and concurrent jobs. This report does not rank scientific performance.
