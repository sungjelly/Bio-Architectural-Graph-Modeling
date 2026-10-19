# Terminal aggregate gradient audit

Stage: grid; 10 complete runs and 17,568 optimizer updates.

Finite gradient evidence passed. All scope/epoch mean norms positive: True. Zero minima beyond expected bootstrap: False. Adjacent-pair counts match expected bootstrap: True.

Preclip norms exceeded the configured clipping threshold in 0/17,568 updates (0.000000%).

| Run | Updates | Largest preclip norm | Clip activations | Later zero minimum |
|---|---:|---:|---:|---|
| r_20261001T074811Z_06acf7a5_s000_f00_a01_2aa32391 | 864 | 0.173383 | 0 | False |
| r_20261001T074811Z_10be86dd_s000_f00_a01_ec605202 | 1672 | 0.207798 | 0 | False |
| r_20261001T074811Z_425c1faf_s000_f00_a01_65533cf1 | 3304 | 0.166053 | 0 | False |
| r_20261001T074811Z_97513f17_s000_f00_a01_bfa65cb7 | 3304 | 0.187477 | 0 | False |
| r_20261001T082016Z_95cce2a4_s000_f00_a01_df1ebd49 | 1672 | 0.155800 | 0 | False |
| r_20261001T082215Z_22f0027c_s000_f00_a01_54281998 | 864 | 0.181787 | 0 | False |
| r_20261001T082850Z_052d767b_s000_f00_a01_35ab7f19 | 3304 | 0.234644 | 0 | False |
| r_20261001T082930Z_d0f806f6_s000_f00_a01_57f4d373 | 864 | 0.165054 | 0 | False |
| r_20261001T085324Z_32178e25_s000_f00_a01_2dfb5a15 | 48 | 0.132272 | 0 | False |
| r_20261001T085435Z_c393af7c_s000_f00_a01_459ab1fb | 1672 | 0.228895 | 0 | False |

- Four graph-block parameter aggregates and the detection head are observed; internal encoder/geometry/feed-forward paths are not separately identified.
- Finite nonzero block gradients establish aggregate training activity, not useful spatial context, successful optimization, graph-specific gain or biology.
- The first upstream zero is expected from the zero-initialized output head. Retained epoch minima and valid-pair counts support this interpretation; individual block vectors are not exported.
- Clipping activation counts use the norm returned by clip_grad_norm_ before clipping, not the postclip parameter-gradient magnitude.
- Negative gradient cosines can arise from changing masks/core-local batches; they do not by themselves establish an optimization flaw.

All source hashes and scope/epoch values are retained in summary.json and scope_epochs.csv. This is a post-hoc diagnostic and does not alter model selection.
