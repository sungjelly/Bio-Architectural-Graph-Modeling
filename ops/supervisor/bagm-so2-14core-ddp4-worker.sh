#!/bin/bash
set -euo pipefail

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh" ""
. "${utils}/environment.sh"

# Source code is isolated in the clean worktree; large runtime state remains in
# the established BAGM tree.  The queue-owned child inherits every override.
readonly source_root=/workspace/BAGM-relative-qkv-analysis
readonly runtime_root=/workspace/BAGM

export BAGM_ROOT="${source_root}"
export BAGM_DATA_ROOT="${runtime_root}/data"
export BAGM_STATE_ROOT="${runtime_root}/state"
export BAGM_ARTIFACT_ROOT="${runtime_root}/artifacts"
export BAGM_SCRATCH_ROOT="${runtime_root}/scratch"
export BAGM_CACHE_ROOT="${runtime_root}/cache"
export BAGM_EXPORT_ROOT="${runtime_root}/exports"
export BAGM_REPORT_ROOT="${runtime_root}/reports"
export PYTHONPATH="${source_root}/src"
export PYTHONUNBUFFERED=1

source /venv/main/bin/activate
cd "${source_root}"

# This one queue worker owns all four GPUs as one indivisible resource.  The
# queued child is torchrun itself, so TERM propagation and rank failure are
# tracked by the normal registry attempt.  Elastic restarts are disabled by
# the command derived from the locked experiment configuration.
exec /venv/main/bin/python -u -m spatial_benchmark \
  --database "${runtime_root}/state/tracking/bagm.sqlite3" \
  worker \
  --worker-id so2-relative-qkv-ddp4 \
  --gpu 0,1,2,3 \
  --once \
  --poll-seconds 5 \
  --heartbeat-seconds 30 \
  --stale-after-seconds 900 \
  --min-free-gb 25
