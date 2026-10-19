#!/bin/bash
set -eo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"
export BAGM_ROOT=/workspace/BAGM
export PYTHONPATH="$BAGM_ROOT:$BAGM_ROOT/src:$BAGM_ROOT/scripts/analysis"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd "$BAGM_ROOT"
exec /venv/main/bin/python -u scripts/analysis/run_so12_cancer_baseline_leiden.py
