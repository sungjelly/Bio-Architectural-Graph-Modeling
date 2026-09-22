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

## Completed model and hL visualization

The authoritative registry records run
`r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf` as completed on
2026-09-04 at 14:27 UTC, with the final epoch-200 checkpoint indexed and verified.
The campaign-level registry label remains the historical `active_training`
label; it is not the run's terminal status. This README and the frozen training
contract were recovered from the checksum-bound training provenance because
they were absent from the current checkout.

The exploratory PNG task contract and exact reproduction commands are in
[`HL_CLUSTERING_ANALYSIS.md`](HL_CLUSTERING_ANALYSIS.md). This report extracts
the final fully observed hL representation and does not alter the training
bundle or establish biological cluster identities.

The PNG analysis is complete: 20 joint clusters across all 246,063 cells,
with identical seeded Leiden replay and independently verified cell alignment.
The 300-DPI map and checksummed report are published under
`reports/analyses/so2_14core_geometry_modulated_hl/r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf/v1/`.
See the analysis document for reproduction, the seven core-dominated clusters,
verification evidence, and interpretation limits.

## Epoch-by-epoch training comparison

The completed [epoch comparison report](../../../reports/analyses/so2_epoch_comparison/v1/README.md)
aligns this run with the original four-block trajectory through its successful
epoch-300 continuation and the recurrent epoch-175 model. Geometry has lower
training Huber in 71/200 matched epochs against the original and 167/175 against
recurrent. The original comparison shows small differences that repeatedly
reverse direction. These are exploratory training diagnostics from seed 0;
recurrent finalization failed, and no held-out or graph-specific gain is claimed.
The report includes all epoch/core tables, figures, provenance, verification,
and exact reproduction commands. No training bundle is modified.

## Nonzero reconstruction comparison

The completed exploratory
[nonzero evaluation](../cmp_20260905_so2_nonzero_accuracy/README.md) replays the
same fixed masks across all 14 cores for four frozen endpoints. Geometry e200
has positive standardized MSE 8.915118 versus 8.862753 original e175 and
8.850479 continued e300; it is worse in 13/14 and 14/14 cores respectively,
while improving zero-entry MSE. Positive exact count accuracy is 3.61%, compared
with 3.63% and 3.75%. These results do not support a nonzero accuracy advantage
for the geometry endpoint. Both error scales, the gene-mean and zero baselines,
complete tables, verification and limitations are in the linked campaign.
