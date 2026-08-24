#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

task_root="${BAGM_ROOT:-/workspace/BAGM}"
cohort_dir="${task_root}/data/processed/cancer_6core_relative_qkv_v1"
graph_dir="${task_root}/data/processed/cancer_6core_relative_qkv_graphs_v1"

source /venv/main/bin/activate
cd "${task_root}"

if [[ ! -f "${cohort_dir}/manifest.json" ]]; then
  /venv/main/bin/python scripts/data/prepare_cancer_6core_relative_qkv.py
fi

if [[ ! -f "${graph_dir}/manifest.json" ]]; then
  /venv/main/bin/python scripts/data/materialize_cancer_6core_relative_graphs.py
fi
