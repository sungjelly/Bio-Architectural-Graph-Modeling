#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
export BAGM_ROOT=/workspace/BAGM
export PYTHONPATH="$BAGM_ROOT/src:$BAGM_ROOT/scripts/analysis"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$BAGM_ROOT"
exec /venv/main/bin/python -u scripts/analysis/run_so12_hidden_leiden.py "$@"
