#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

task_root="${BAGM_ROOT:-/workspace/BAGM}"

source /venv/main/bin/activate
cd "${task_root}"

exec /venv/main/bin/python \
  scripts/diagnostics/preflight_cancer_6core_relative_qkv.py \
  --config configs/experiment/cancer_6core_relative_qkv_seed0.yaml \
  --output state/preflight/cancer_6core_relative_qkv_seed0.json
