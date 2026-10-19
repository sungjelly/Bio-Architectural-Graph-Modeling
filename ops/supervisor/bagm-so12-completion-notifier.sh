#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
export BAGM_ROOT=/workspace/BAGM
export PYTHONPATH="$BAGM_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=''
cd "$BAGM_ROOT"
exec /venv/main/bin/python scripts/diagnostics/watch_run_notification.py \
  --run-id r_20261016T090820Z_ad267085_s000_f00_a02_451ef28b \
  --thread-id 01a1439e-b190-7740-869f-55e997248425 \
  --worker bagm_so12_relative_qkv_worker
