#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
export BAGM_ROOT=/workspace/BAGM
export PYTHONPATH="$BAGM_ROOT/src:$BAGM_ROOT/scripts/analysis"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
cd "$BAGM_ROOT"
exec flock --shared --nonblock "$BAGM_ROOT/state/locks/bagm-worker.lock" \
  flock --exclusive --nonblock "$BAGM_ROOT/state/locks/bagm-worker-gpu-0.lock" \
  flock --exclusive --nonblock "$BAGM_ROOT/state/locks/bagm-worker-gpu-1.lock" \
  flock --exclusive --nonblock "$BAGM_ROOT/state/locks/bagm-worker-gpu-2.lock" \
  flock --exclusive --nonblock "$BAGM_ROOT/state/locks/bagm-worker-gpu-3.lock" \
  /venv/main/bin/python -u scripts/analysis/run_so12_pearson_preprocessing.py "$@"
