#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

readonly source_root="${BAGM_HURDLE_SOURCE_ROOT:-/workspace/BAGM-relative-qkv-analysis}"
readonly runtime_root="${BAGM_HURDLE_RUNTIME_ROOT:-/workspace/BAGM}"

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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=2000

source /venv/main/bin/activate
cd "${source_root}"

exec /venv/main/bin/python -u -m spatial_benchmark \
  --database "${runtime_root}/state/tracking/bagm.sqlite3" \
  worker \
  --worker-id so2-target-isolated-hurdle-nb-once-ddp4 \
  --gpu 0,1,2,3 \
  --once \
  --poll-seconds 5 \
  --heartbeat-seconds 30 \
  --stale-after-seconds 900 \
  --min-free-gb 20
