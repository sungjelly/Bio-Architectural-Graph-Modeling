# SO2 four-block geometry-modulated Relative-QKV with NB validation

This campaign trains one fresh seed-0 model on SO2 cores 15–28 with a
donor-grouped **training/validation split only**. Six donor pairs (12 cores) are
used for optimization and one entirely unseen donor pair (2 cores) is used for
validation, checkpoint selection, learning-rate scheduling, and early stopping.
There is no test split, no test evaluation, and no test artifact.

The graph encoder is the existing four-independent-block geometry-modulated
Relative-QKV model. Each block projects Q, K, and V separately from the same
pre-attention LayerNorm node-content state `h`. Initially, `h` combines masked
train-standardized `log1p` expression, the explicit gene-mask channel, and the
allowed morphology metadata normalized from training cores only; later blocks
also contain prior node-content messages and residual feed-forward state. The
relative vector `rho` changes attention routing through Q-K modulation and bias,
but is never injected into, projected into, or used to gate V. This is the
`shared_node_content_state_without_direct_relative_geometry` value-stream
contract—not an expression-only value stream. The count head reinterprets the
existing decoder logits as positive NB means and adds one learned positive
inverse-dispersion per gene. Training minimizes full-constant masked NB2
negative log likelihood against raw integer counts, and no total computed from
hidden target counts is supplied to the model.

The held-out donor pair is selected before outcome inspection by the blinded
SHA-256 rule frozen in `frozen_task_contract.yaml`. Expression and metadata
preprocessing statistics are fitted on training cores only and applied unchanged
to validation. Geometry graph caches are immutable, core-local, and referenced
in place so the 15+ GiB cache is not duplicated.

Every epoch records training and fixed-mask validation NLL, count errors and
zero calibration, dispersion summaries, learning rate, runtime/VRAM, plus global,
four-block, and dispersion-gradient norm/direction diagnostics. Validation NB
NLL is the sole model-selection signal. The run may stop after epoch 50 when it
has failed to improve by at least `1e-4` for 25 consecutive validations, with a
hard ceiling of 300 epochs.

During training, atomic rolling recovery and validation-best checkpoints are
overwritten in place. After successful reload verification, only the single
validation-best checkpoint is retained. Full prediction matrices and gradient
tensors are not persisted.

Because the sole validation donor is repeatedly used for scheduler, stopping,
and checkpoint selection, its metrics are exploratory model-selection estimates,
not an unbiased test of generalization. A matched Huber run on exactly this split
would be required before claiming that NB improves accuracy over the prior loss.

Production launch is gated on unit tests, a synthetic NB recovery/overfit check,
a tiny real-data loss-decrease check, a four-rank numerical/checkpoint preflight,
and a measured peak below 22 GiB on every 24-GiB GPU.

## Completed outcome

Production run `r_20260907T144628Z_489a9b42_s000_f00_a01_188c8511`
completed successfully after epoch 192. Validation NB NLL reached its best value,
`0.34690783321857455`, at epoch 167; training then stopped after the frozen
25-validation patience window elapsed without the required improvement. Loss,
validation metrics, and gradient-direction summaries were recorded for every
completed epoch.

Finalization retained only `checkpoints/best.ckpt` (SHA-256
`2b5790213477a1adb47633a5f3a8d56d0b325ab14ebbd9c7b67ba849a2a7cc2e`). No
test artifacts were produced. The reported validation result remains a
model-selection estimate from one repeatedly evaluated donor pair, not an
unbiased test estimate and not by itself evidence that NB improves over Huber.
