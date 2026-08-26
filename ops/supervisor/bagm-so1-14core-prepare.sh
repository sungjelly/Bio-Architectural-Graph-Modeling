#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

readonly source_root="${BAGM_SO1_SOURCE_ROOT:-/workspace/BAGM-relative-qkv-analysis}"
readonly runtime_root="${BAGM_SO1_RUNTIME_ROOT:-/workspace/BAGM}"
readonly cohort_dir="${runtime_root}/data/processed/so1_14core_relative_qkv_v1"
readonly graph_dir="${runtime_root}/data/processed/so1_14core_relative_qkv_graphs_v1"

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

if [[ ! -f "${cohort_dir}/manifest.json" ]]; then
  /venv/main/bin/python -u scripts/data/prepare_so1_14core_relative_qkv.py
fi

if [[ ! -f "${graph_dir}/manifest.json" ]]; then
  /venv/main/bin/python -u scripts/data/materialize_so1_14core_relative_graphs.py
fi
