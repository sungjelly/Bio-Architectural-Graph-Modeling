# Four-seed Relative-QKV stability report

This is a **four-seed ensemble-spread** analysis over model seeds 0, 1, 2, and 3. Seed 4 and five-seed campaign completion remain deferred. The spread is not a calibrated biological confidence interval.

All computations are held-in and transductive. Attention, gradients, and Jacobians are model-derived quantities and do not establish direct signaling, biological mechanism, or causality.

## Members and plateau epochs

| Seed | Run ID | Final epoch | Checkpoint SHA-256 |
| ---: | --- | ---: | --- |
| 0 | `r_20260824T121803Z_16144620_s000_f00_a02_62498796` | 200 | `c5b7fdd6e3192c3146d415f1c77474f9e4483bba090c4fc9c2ac7d79ab554c0c` |
| 1 | `r_20260824T124852Z_95591978_s001_f00_a01_266e6db4` | 225 | `3ce4a0e7983a14cba34a32e182800c62ed0184377247443da6bbec72531d0d3c` |
| 2 | `r_20260824T124855Z_95591978_s002_f00_a01_3d4cbf7d` | 175 | `91f3f8bfe03ee1624c6ac954a612c65fd6cabc4bf0cce03dc2debf1c44de6411` |
| 3 | `r_20260824T125052Z_95591978_s003_f00_a01_c02388f5` | 200 | `b8943dcb21bac2ea7b3077f1c51e7b6b24b31effe3d23ce471aa05201a3f36d4` |

Different final epochs are permitted because every seed independently satisfied the locked training-loss plateau rule. No validation or test metric selected a checkpoint.

## Final training loss

Metric: equal-core mean masked Huber over the training masks.

| Mean | Sample SD | Minimum | Maximum | Range |
| ---: | ---: | ---: | ---: | ---: |
| 0.2419572 | 0.00034530362 | 0.24159237 | 0.24231865 | 0.00072628483 |

Full per-seed numeric curves are in `tables/training_curves.parquet`.

## Fixed held-in fit diagnostics

These are diagnostics on identical fixed masks, not validation or test evaluation.

| Metric | Mean | Sample SD | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: |
| `fit/uniform_per_cell/masked_huber` | 0.24062536 | 0.00033089984 | 0.24039614 | 0.24111572 |
| `fit/uniform_per_cell/masked_mae` | 0.38237464 | 0.0039329112 | 0.3779648 | 0.38724542 |
| `fit/uniform_per_cell/masked_mse` | 0.94743447 | 0.0035114305 | 0.94308967 | 0.95155181 |
| `fit/uniform_per_cell/masked_r2` | 0.048145086 | 0.0039979222 | 0.043460935 | 0.053132276 |

## Representation and attention stability

- Full aligned final-node embeddings: 117996 cells.
- Linear CKA matrix: `[[1.0, 0.9750044593145472, 0.9719836742014208, 0.9812195975700853], [0.9750044593145472, 1.0, 0.9702235200332088, 0.9750540836669226], [0.9719836742014208, 0.9702235200332088, 1.0, 0.9712864787362357], [0.9812195975700853, 0.9750540836669226, 0.9712864787362357, 1.0]]`.
- Orthogonal Procrustes similarities are recorded in `arrays/embedding_stability.npz`.
- Attention heads were aligned to seed 0 using Hungarian maximum fixed-signature Spearman.
- Matched top-edge overlap uses fraction 0.05 (4397 fixed edges per head).
- Matched attention Spearman/Jaccard, positional-bias correlations, and content-versus-bias rank/sign/magnitude statistics are preserved in `arrays/attention_head_stability.npz` and `report.json`.
- Per-edge four-seed mean-head attention and receiver-centered content/bias/combined summaries are in `tables/fixed_edge_summary.parquet`.

## Mutual routing and selected gradients

- Mutual routing selection: union of each seed's top 100 pairs within each core.
- The mutual score is descriptive reciprocal degree-adjusted attention, not causal influence.
- Selected gradient requests: 24; selection was independent of model outputs.
- Both attention and prediction derivatives have mean, sample SD, median, min/max, empirical quantiles, support, sign consistency, and pairwise seed Spearman statistics.
- No exhaustive Jacobian was materialized.

## Limitations

- Four model seeds quantify ensemble spread, not a calibrated confidence interval.
- All diagnostics are held-in and transductive; they do not estimate generalization.
- Attention is computational routing and is not direct signaling or causality.
- Gradients are local, scale-dependent model sensitivities and are not causality.
- Head alignment is derived from fixed held-in probes and may not be unique.
- The axial orientation representation cannot distinguish opposite polarity.
- The 24 selected gradient probes are sparse and are not representative of all edges or genes.
- Any finite nonzero gradient, however small, counts as support under the locked protocol.
- Models stopped at different plateau epochs, so exposure duration can contribute to observed seed spread.
