#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

readonly source_root="${BAGM_SO1_SOURCE_ROOT:-/workspace/BAGM-relative-qkv-analysis}"
readonly runtime_root="${BAGM_SO1_RUNTIME_ROOT:-/workspace/BAGM}"

export BAGM_ROOT="${source_root}"
export BAGM_DATA_ROOT="${runtime_root}/data"
export BAGM_STATE_ROOT="${runtime_root}/state"
export BAGM_ARTIFACT_ROOT="${runtime_root}/artifacts"
export BAGM_SCRATCH_ROOT="${runtime_root}/scratch"
export BAGM_CACHE_ROOT="${runtime_root}/cache"
export BAGM_EXPORT_ROOT="${runtime_root}/exports"
export BAGM_REPORT_ROOT="${runtime_root}/reports"
export PYTHONPATH="${source_root}/src"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source /venv/main/bin/activate
cd "${source_root}"

exec /venv/main/bin/python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=4 \
  --max-restarts=0 \
  scripts/diagnostics/preflight_so1_14core_relative_qkv_ddp.py \
  --config configs/experiment/so1_14core_relative_qkv_seed0_batch2_plateau_min150.yaml \
  --output "${runtime_root}/state/preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json"
