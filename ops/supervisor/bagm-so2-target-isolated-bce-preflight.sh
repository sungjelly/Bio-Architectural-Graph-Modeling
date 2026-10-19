#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

readonly source_root="${BAGM_BCE_SOURCE_ROOT:-/workspace/BAGM-relative-qkv-analysis}"
readonly runtime_root="${BAGM_BCE_RUNTIME_ROOT:-/workspace/BAGM}"

export BAGM_ROOT="${source_root}"
export BAGM_DATA_ROOT="${runtime_root}/data"
export BAGM_STATE_ROOT="${runtime_root}/state"
export BAGM_ARTIFACT_ROOT="${runtime_root}/artifacts"
export BAGM_SCRATCH_ROOT="${runtime_root}/scratch"
export BAGM_CACHE_ROOT="${runtime_root}/cache"
export BAGM_EXPORT_ROOT="${runtime_root}/exports"
export BAGM_REPORT_ROOT="${runtime_root}/reports"
export BAGM_RESULT_ROOT="${runtime_root}/results"
export PYTHONPATH="${source_root}/src:${source_root}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

source /venv/main/bin/activate
cd "${source_root}"

exec /venv/main/bin/python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --max-restarts=0 \
  scripts/diagnostics/preflight_so2_target_isolated_binary_expression_bce_ddp.py \
  --config configs/experiment/so2_target_isolated_geometry_modulated_binary_expression_bce_once_per_cell_seed0.yaml \
  --output "${runtime_root}/state/preflight/so2_target_isolated_binary_expression_bce_once_per_cell_ddp4.json"
