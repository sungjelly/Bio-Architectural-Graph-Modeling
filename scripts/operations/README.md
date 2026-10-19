# SO2 BCE audit stage execution

This operational runner executes a prespecified audit plan through the existing
registered launcher. It makes no model-selection decisions. The scientific
contract is [the optimization-audit campaign](../../experiments/campaigns/cmp_20261001_so2_bce_optimization_audit/README.md).

Run from the repository root with the same BAGM path overrides as the registered
launcher:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/operations/run_so2_bce_audit_stage.py --plan /path/to/declared_stage.json
```

The JSON object contains `stage_id`, `allowed_gpus` (unique strings from `0` to
`3`), and an ordered `jobs` list. Each job contains exactly `label`, `gpu`,
`profile`, `batch_mode`, `batch_size`, `learning_rate`, `dropout`, `weight_decay`,
`seed`, and `smoke_run`. Use `batch_size: null` for paired cores and
`smoke_run: null` only for the smoke profile. Other profiles require the
appropriate verified smoke run ID. Labels and stage IDs use letters, digits,
underscores or hyphens. Duplicate scientific jobs are rejected.

Jobs retain their listed order within each GPU lane, with one active launcher
per lane. The launcher owns GPU availability checks, locks, registration,
training and immutable output verification. Operational `plan.json`, atomic
`state.json`, and individual launcher logs are written only under
`BAGM_STATE_ROOT/logs/so2_bce_optimization_audit/<stage_id>/`. Existing stage
directories are never reused; there are no automatic retries or resumes.
Keep the input plan outside that output directory: the runner creates the
directory exclusively and saves its own immutable plan snapshot inside it.

A failed launcher holds every pending job while already running launchers
finish. The process returns nonzero and records every exit code. SIGINT or
SIGTERM holds pending jobs and forwards the signal only to this runner's own
launcher process groups; the registered launcher handles its training children.
An uncatchable process/host loss can leave a recorded running state: inspect the
registered jobs and device locks before declaring a new stage. A supervisor
should use `autorestart=false` so it does not repeatedly attempt the same stage.

Acceptance checks cover argument derivation, strict plan validation, lane order,
failure holds, signal ownership, persisted states and existing-stage rejection:

```bash
PYTHONPATH=src:. /venv/main/bin/python -m pytest -q tests/unit/infrastructure/test_so2_bce_audit_stage.py
```

## Local Gene Residual after the current GeneMAE batch

`run_local_gene_after_current.py` implements the authorized deferred follow-up
in `experiments/campaigns/cmp_20261005_local_gene_early_stopping/README.md`.
Its frozen plan pins all nine production queue jobs in the active GeneMAE
campaign and the complete source hashes for the follow-up. Diagnostic pilot
failures are not production slots. Any production failure, source drift,
configuration change or changed slot set holds the follow-up for inspection.

The supervisor program `bagm_local_gene_after_current` waits without acquiring
CUDA or taking GPU worker locks. After all nine production jobs are successful,
it verifies their immutable bundles and waits for all four GPUs/worker locks.
It then runs a registered two-epoch pilot, four early-stopped local/self seed
runs, and the result publisher. The trainer owns its actual GPU lock and final
admission check, so concurrent claims fail closed. No existing job is stopped.

Plan, atomically updated status, dependency verification receipt and separate
child-launcher logs live in
`/workspace/BAGM/state/handoffs/local_gene_early_stopping/`.
The source wrapper and supervisor configuration are in `ops/supervisor/`.
Use `--check-only` with the plan for CPU-only dependency inspection. Supervisor
uses `autorestart=false`: once any new execution begins, a failed/interrupted
handoff requires explicit inspection, and cannot silently create duplicate runs.
