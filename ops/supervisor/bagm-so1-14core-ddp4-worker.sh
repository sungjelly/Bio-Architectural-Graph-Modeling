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
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source /venv/main/bin/activate
cd "${source_root}"

# This one-shot queue worker owns GPUs 0-3 as one indivisible DDP resource.
# The service is intentionally not an idleness scheduler: confirm all four
# GPUs are free before manually starting it.
exec /venv/main/bin/python -u -m spatial_benchmark \
  --database "${runtime_root}/state/tracking/bagm.sqlite3" \
  worker \
  --worker-id so1-relative-qkv-plateau-min150-ddp4 \
  --gpu 0,1,2,3 \
  --once \
  --poll-seconds 5 \
  --heartbeat-seconds 30 \
  --stale-after-seconds 900 \
  --min-free-gb 25
