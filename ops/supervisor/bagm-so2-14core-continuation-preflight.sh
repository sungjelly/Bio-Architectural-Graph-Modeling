#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

readonly source_root=/workspace/BAGM-relative-qkv-analysis
readonly runtime_root=/workspace/BAGM

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
  scripts/diagnostics/preflight_so2_14core_relative_qkv_ddp.py \
  --config configs/experiment/so2_14core_relative_qkv_seed0_batch2_resume175_fixed300.yaml \
  --output "${runtime_root}/state/preflight/so2_14core_relative_qkv_ddp4_resume175_fixed300.json" \
  --prior-c23-receipt \
  "${runtime_root}/artifacts/runs/2026/08/r_20260824T121803Z_16144620_s000_f00_a02_62498796/diagnostics/hardware_preflight.json"
