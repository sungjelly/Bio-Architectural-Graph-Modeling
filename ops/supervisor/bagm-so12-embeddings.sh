#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
export BAGM_ROOT=/workspace/BAGM
export PYTHONPATH="$BAGM_ROOT/src:$BAGM_ROOT/scripts/analysis"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cd "$BAGM_ROOT"
exec flock --exclusive --nonblock "$BAGM_ROOT/state/locks/so12-embeddings.lock" \
  nice -n 10 /venv/main/bin/python -u scripts/analysis/run_so12_embeddings.py --threads 8 "$@"
