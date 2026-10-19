# Terminal aggregate gradient audit

Stage: confirmation; 4 complete runs and 39,808 optimizer updates.

Finite gradient evidence passed. All scope/epoch mean norms positive: True. Zero minima beyond expected bootstrap: False. Adjacent-pair counts match expected bootstrap: True.

Preclip norms exceeded the configured clipping threshold in 0/39,808 updates (0.000000%).

| Run | Updates | Largest preclip norm | Clip activations | Later zero minimum |
|---|---:|---:|---:|---|
| r_20261001T092859Z_5cb93f7c_s000_f00_a01_0fb40980 | 6688 | 0.207798 | 0 | False |
| r_20261001T101202Z_15baf65c_s001_f00_a01_6427735a | 6688 | 0.174232 | 0 | False |
| r_20261001T101202Z_a32d61e8_s001_f00_a01_8332ee26 | 13216 | 0.265267 | 0 | False |
| r_20261001T101202Z_f8197b29_s000_f00_a01_ed00c19e | 13216 | 0.231511 | 0 | False |

- Four graph-block parameter aggregates and the detection head are observed; internal encoder/geometry/feed-forward paths are not separately identified.
- Finite nonzero block gradients establish aggregate training activity, not useful spatial context, successful optimization, graph-specific gain or biology.
- The first upstream zero is expected from the zero-initialized output head. Retained epoch minima and valid-pair counts support this interpretation; individual block vectors are not exported.
- Clipping activation counts use the norm returned by clip_grad_norm_ before clipping, not the postclip parameter-gradient magnitude.
- Negative gradient cosines can arise from changing masks/core-local batches; they do not by themselves establish an optimization flaw.

All source hashes and scope/epoch values are retained in summary.json and scope_epochs.csv. This is a post-hoc diagnostic and does not alter model selection.
