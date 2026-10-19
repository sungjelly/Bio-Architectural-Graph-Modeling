# Final reporting verification

Focused tests: **69 passed** in 4.21 seconds; no test failures.

Doctor exited **1**. SQLite integrity is `ok`; the user-authorized 5 GiB floor passes with about19.934 GiB free. Doctor still reports the same one-checkpoint catalog gap (69 artifacts,68 indexed) and the same two missing registered six-core dataset/split paths as prelaunch. Exact issues are retained in doctor.json; no global clean-health claim is made.

All four RTX3090 GPUs had0%utilization and no compute processes at the recorded observation. Each reported24124 MiB free. This is a point-in-time occupancy snapshot, not a training throughput result.

Only the five reporting suites ran; the full repository suite was not repeated. No training/code changes or curated-result publication were performed. The analyzed scripts/tests retained their source hashes, and the Git status listings were unchanged.

Exact environment overrides, command argument arrays, timestamps, exit codes and durations are in execution.json. Reproduction from the source root uses those BAGM_* shared roots and BAGM_MIN_FREE_DISK_GB=5:

```bash
PYTHONPATH=src:. /venv/main/bin/python -m pytest -q tests/unit/infrastructure/test_so2_bce_optimization_report.py tests/unit/spatial_benchmark/test_publish_so2_bce_optimization_audit_result.py tests/unit/spatial_benchmark/test_so2_bce_threshold_sensitivity.py tests/unit/spatial_benchmark/test_so2_bce_gradient_audit.py tests/unit/infrastructure/test_so2_bce_execution_profile_report.py
BAGM_MIN_FREE_DISK_GB=5 PYTHONPATH=src:. /venv/main/bin/python -m spatial_benchmark doctor
```

All retained logs, snapshots and this receipt are checksum-bound in manifest.json.
