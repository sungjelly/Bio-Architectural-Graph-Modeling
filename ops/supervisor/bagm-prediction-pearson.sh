#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
pearson_gpu="$1"
pearson_model="$2"
export CUDA_VISIBLE_DEVICES="$pearson_gpu"
export BAGM_ROOT=/workspace/BAGM
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export BAGM_REPORT_ROOT=/workspace/BAGM/reports
export BAGM_RESULT_ROOT=/workspace/BAGM/results
export BAGM_CACHE_ROOT=/workspace/BAGM/cache
export BAGM_EXPORT_ROOT=/workspace/BAGM/exports
export PYTHONPATH=/workspace/BAGM-relative-qkv-analysis/src:/workspace/BAGM:/workspace/BAGM/scripts/analysis
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
cd "$BAGM_ROOT"
exec flock --shared --nonblock /workspace/BAGM/state/locks/bagm-worker.lock \
    flock --exclusive --nonblock "/workspace/BAGM/state/locks/bagm-worker-gpu-${pearson_gpu}.lock" \
    /venv/main/bin/python -u scripts/analysis/run_so2_prediction_pearson.py --model "$pearson_model"
