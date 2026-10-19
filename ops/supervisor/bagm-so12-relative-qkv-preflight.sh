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
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd "$BAGM_ROOT"
exec flock --shared --nonblock "$BAGM_ROOT/state/locks/bagm-worker.lock" \
  flock --exclusive --nonblock "$BAGM_ROOT/state/locks/bagm-worker-gpu-0.lock" \
  flock --exclusive --nonblock "$BAGM_ROOT/state/locks/bagm-worker-gpu-1.lock" \
  flock --exclusive --nonblock "$BAGM_ROOT/state/locks/bagm-worker-gpu-2.lock" \
  flock --exclusive --nonblock "$BAGM_ROOT/state/locks/bagm-worker-gpu-3.lock" \
  /venv/main/bin/python -u -m torch.distributed.run --standalone --nnodes=1 \
  --nproc-per-node=4 --max-restarts=0 scripts/train/run_so12_relative_qkv.py \
  --config experiments/campaigns/cmp_20261016_so12_pearson_relative_qkv_train_validation/configs/seed0_attempt2.yaml \
  --run-scratch state/preflight/so12_relative_qkv_ddp4_attempt2 --preflight
