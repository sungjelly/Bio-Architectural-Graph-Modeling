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

source /venv/main/bin/activate
cd "${source_root}"

exec /venv/main/bin/python -u \
  scripts/train/handoff_so2_to_so1_relative_qkv.py \
  --database "${runtime_root}/state/tracking/bagm.sqlite3" \
  --state "${runtime_root}/state/handoffs/so2_epoch300_to_so1_plateau_min150.json"
