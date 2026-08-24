#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

gpu_id="${1:?one physical GPU index is required}"
task_root="${BAGM_ROOT:-/workspace/BAGM}"

source /venv/main/bin/activate
cd "${task_root}"

exec /venv/main/bin/python -u -m spatial_benchmark \
  --database "${task_root}/state/tracking/bagm.sqlite3" \
  worker \
  --worker-id "relative-qkv-gpu-${gpu_id}" \
  --gpu "${gpu_id}" \
  --parallel-gpu-workers \
  --poll-seconds 5 \
  --heartbeat-seconds 30 \
  --stale-after-seconds 900 \
  --min-free-gb 25
