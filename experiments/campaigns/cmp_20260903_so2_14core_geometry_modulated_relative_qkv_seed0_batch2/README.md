# SO2 four-block geometry-modulated Relative-QKV fit

This campaign trains one fresh seed-0 model on SO2 cores 15–28 using the
August 25 data, graph, masking, optimizer, and plateau protocol. The model has
four independent graph blocks. Its only scientific change is the attention
score family: per-head L2-normalized Q/K, bounded learned logit scale,
geometry-conditioned dimension-wise Q–K modulation, and bounded geometry
bias. Values remain expression-only.

This is an exploratory fitted-cohort comparison against August 25 run
`r_20260825T155601Z_e56532d1_s000_f00_a01_e05918a6`. Because cosine Q/K and
geometry modulation change together, results identify the complete score
mechanism rather than modulation alone. No held-out, biological-mechanism, or
generalization claim is supported.

The run records an fsynced loss row every completed epoch plus global and
four-block gradient norm/direction summaries. Gradient tensors and per-step
files are never persisted. During training one atomic rolling
`checkpoints/latest.ckpt` is replaced each epoch; successful finalization keeps
only `checkpoints/last.ckpt`.

Training may launch only after the distinct four-rank preflight verifies the
5,134,088-parameter architecture, neutral geometry initialization, finite
loss/gradients, all four block gradients, checkpoint reload, and at most
22 GiB peak VRAM on every 24-GiB GPU. The production run then follows the
literal August 25 minimum-150/25-epoch-block training-loss plateau rule.
