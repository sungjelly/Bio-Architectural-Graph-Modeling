#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
export BAGM_ROOT=/workspace/BAGM
export PYTHONPATH="$BAGM_ROOT/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BAGM_ROOT"
exec /venv/main/bin/python -u -m spatial_benchmark worker \
  --worker-id bagm-so12-relative-qkv-ddp4 --gpu 0,1,2,3 --once \
  --heartbeat-seconds 30 --stale-after-seconds 900 --min-free-gb 5
