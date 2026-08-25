# Locked four-seed stability analysis

This additive post-training protocol compares completed, strictly verified
plateau checkpoints for model seeds 0, 1, 2, and 3. Seed 4 is deferred. The
result must be described as **four-seed ensemble spread** or **seed
uncertainty**, never as five-seed campaign completion or a calibrated biological
confidence interval.

The analysis is held-in and transductive. Attention is computational routing;
attention, gradients, and Jacobians are model-derived quantities and do not by
themselves establish direct signaling, a biological mechanism, or causality.
The local orientation tensor is axial: it cannot distinguish opposite
directions along an axis without an independently measured polarity signal.

The four checkpoints may stop at different independently audited plateau
epochs. Consequently, training-exposure duration can contribute to the
observed seed spread; the analysis does not attribute every difference solely
to initialization.

The locked protocol is
`analysis_protocol_selected_gradient_stability_v1.yaml` (SHA-256
`ec7c57df87083a54a81a661e50319fdb614993992701f48a620efaa80622696e`).
The 24 model-independent selected-gradient requests are in
`selected_gradient_requests_v1.csv` (SHA-256
`2ebe0d5fbf59f268d8b60d9b634a4ecf5d2621083ee05257eabb568af0bec4d2`).
The request verifier regenerates every row from the locked graph, plotting-only
coordinates, ordered gene schema, and fixed inference masks before CUDA is
initialized.

For logit comparisons, content logits and relative positional biases are each
centered by subtracting their mean over all incoming edges for every receiver
and head. This removes receiver-wise softmax-null offsets. Attention
probabilities themselves are not centered. Head signatures, positional-bias
correlations, content-versus-position diagnostics, and fixed-edge logit
summaries all use this locked gauge.

All ensemble summaries use sample standard deviation (`ddof=1`) and NumPy
linear empirical quantiles at 0.05, 0.25, 0.75, and 0.95. A selected gradient is
counted as supported when its absolute value is strictly greater than zero;
therefore even a tiny finite nonzero value counts. Mutual-pair support means
membership in that seed's per-core top 100, not merely a nonzero mutual score.

The selected derivatives perturb one gene-wise standardized `log1p(raw count)`
source-expression unit. Prediction derivatives are reported in standardized
`log1p(raw count)` target-prediction units per standardized source-expression
unit; attention derivatives are attention probability per standardized
source-expression unit. The 24 locked probes (four per core) are deliberately
sparse, model-independent diagnostics and are not representative of all edges
or source-gene/target-gene relationships.

Run the complete compact analysis on one explicitly bound GPU:

```bash
set -euo pipefail

# SOURCE_ROOT must be a clean, committed worktree containing the analysis code.
# RUNTIME_ROOT is the external worktree holding immutable data/run artifacts.
readonly SOURCE_ROOT=/workspace/BAGM-relative-qkv-analysis
readonly RUNTIME_ROOT=/workspace/BAGM
readonly PHYSICAL_GPU_INDEX=1
readonly CAMPAIGN_ROOT="${SOURCE_ROOT}/experiments/campaigns/cmp_20260824_cancer_6core_relative_qkv_multiseed"
readonly RECEIPT_ROOT="${RUNTIME_ROOT}/reports/analyses/cancer_6core_relative_qkv/checkpoint_verification"
readonly OUTPUT="${RUNTIME_ROOT}/reports/analyses/cancer_6core_relative_qkv/four_seed_stability_v1"

readonly RUN0="${RUNTIME_ROOT}/artifacts/runs/2026/08/r_20260824T121803Z_16144620_s000_f00_a02_62498796"
readonly RUN1="${RUNTIME_ROOT}/artifacts/runs/2026/08/r_20260824T124852Z_95591978_s001_f00_a01_266e6db4"
readonly RUN2="${RUNTIME_ROOT}/artifacts/runs/2026/08/r_20260824T124855Z_95591978_s002_f00_a01_3d4cbf7d"
readonly RUN3="${RUNTIME_ROOT}/artifacts/runs/2026/08/r_20260824T125052Z_95591978_s003_f00_a01_c02388f5"

readonly RECEIPT0="${RECEIPT_ROOT}/r_20260824T121803Z_16144620_s000_f00_a02_62498796.json"
readonly RECEIPT1="${RECEIPT_ROOT}/r_20260824T124852Z_95591978_s001_f00_a01_266e6db4.json"
readonly RECEIPT2="${RECEIPT_ROOT}/r_20260824T124855Z_95591978_s002_f00_a01_3d4cbf7d.json"
readonly RECEIPT3="${RECEIPT_ROOT}/r_20260824T125052Z_95591978_s003_f00_a01_c02388f5.json"

test -z "$(git -C "${SOURCE_ROOT}" status --porcelain)"
test ! -e "${OUTPUT}"
cd "${SOURCE_ROOT}"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_INDEX}" \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
PYTHONPATH="${SOURCE_ROOT}/src" \
/venv/main/bin/python \
"${SOURCE_ROOT}/scripts/analysis/run_relative_qkv_four_seed_stability.py" \
  --run "0=${RUN0}" \
  --run "1=${RUN1}" \
  --run "2=${RUN2}" \
  --run "3=${RUN3}" \
  --checkpoint "0=${RUN0}/checkpoints/last.ckpt" \
  --checkpoint "1=${RUN1}/checkpoints/last.ckpt" \
  --checkpoint "2=${RUN2}/checkpoints/last.ckpt" \
  --checkpoint "3=${RUN3}/checkpoints/last.ckpt" \
  --receipt "0=${RECEIPT0}" \
  --receipt "1=${RECEIPT1}" \
  --receipt "2=${RECEIPT2}" \
  --receipt "3=${RECEIPT3}" \
  --cohort-dir "${RUNTIME_ROOT}/data/processed/cancer_6core_relative_qkv_v1" \
  --graph-dir "${RUNTIME_ROOT}/data/processed/cancer_6core_relative_qkv_graphs_v1" \
  --protocol "${CAMPAIGN_ROOT}/analysis_protocol_selected_gradient_stability_v1.yaml" \
  --protocol-sha256 "${CAMPAIGN_ROOT}/analysis_protocol_selected_gradient_stability_v1.sha256" \
  --gradient-requests "${CAMPAIGN_ROOT}/selected_gradient_requests_v1.csv" \
  --gradient-requests-sha256 "${CAMPAIGN_ROOT}/selected_gradient_requests_v1.sha256" \
  --device cuda:0 \
  --receiver-chunk-size 512 \
  --max-edges-per-chunk 200000 \
  --output "${OUTPUT}"
```

Recheck `nvidia-smi` immediately before launch and change only
`PHYSICAL_GPU_INDEX` if GPU 1 is no longer idle. The program independently
rejects a selected GPU that has another compute owner. Keeping data, archived
runs, receipts, and output under `RUNTIME_ROOT` prevents those generated files
from dirtying the source-provenance worktree.

Execution is locked to deterministic seed `2026082497`, deterministic
algorithms with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, AMP float16 replay for
attention and node-embedding extraction with FP32 attention accumulation, and
FP32 without AMP for selected derivatives. Receiver chunks remain exact: all
incoming edges of a receiver are normalized together, with no sampling or
truncation. The chunk settings above are part of the protocol, not tuning
options.

The command refuses an existing output path. It processes one checkpoint and
one complete core at a time, preserves exact receiver-wise attention, retains
only fixed receiver explanations and compact mutual-pair candidates, and never
materializes a full edge-by-gene Jacobian. Publication is atomic and includes a
checksum manifest, exact source/runtime provenance, Parquet tables, compressed
NumPy arrays, `report.json`, and `report.md`. The required output schema also
preserves aligned per-seed fixed-edge scores, per-seed mutual scores and top-100
support indicators, and all 96 selected-gradient rows before deriving the
four-seed summaries.
