#!/bin/bash
set -euo pipefail
utils=/opt/supervisor-scripts/utils
if [[ "${1:-}" == "--handoff" ]]; then
    . "${utils}/logging.sh" "/var/log/portal/bagm_local_gene_after_current.log"
else
    . "${utils}/logging.sh" "/var/log/portal/bagm_local_gene_early_stopping_gpu${1:?GPU argument required}.log"
fi
. "${utils}/environment.sh"
export BAGM_ROOT=/workspace/BAGM-relative-qkv-analysis
export BAGM_DATA_ROOT=/workspace/BAGM/data
export BAGM_STATE_ROOT=/workspace/BAGM/state
export BAGM_ARTIFACT_ROOT=/workspace/BAGM/artifacts
export BAGM_SCRATCH_ROOT=/workspace/BAGM/scratch
export BAGM_REPORT_ROOT=/workspace/BAGM/reports
export BAGM_RESULT_ROOT=/workspace/BAGM/results
export BAGM_CACHE_ROOT=/workspace/BAGM/cache
export BAGM_EXPORT_ROOT=/workspace/BAGM/exports
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${BAGM_ROOT}/src:${BAGM_ROOT}"
cd "$BAGM_ROOT"
if [[ "${1:-}" == "--handoff" ]]; then
    shift
    export CUDA_VISIBLE_DEVICES=''
    exec /venv/main/bin/python -u scripts/operations/run_local_gene_after_current.py "$@"
fi
gpu="$1"
shift
export CUDA_VISIBLE_DEVICES="$gpu"
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE MASTER_ADDR MASTER_PORT
exec /venv/main/bin/python -u scripts/train/run_local_gene_early_stopping.py --gpu "$gpu" "$@"
