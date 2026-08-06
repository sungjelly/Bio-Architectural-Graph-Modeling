# Experiment payload retention decision: 2026-08-02

## Decision and scope

This is an explicit, user-authorized retention decision for local disk cleanup.
It supersedes the no-deletion decision recorded on 2026-07-30 only for the
payload classes listed below. It does not change any scientific result,
campaign outcome, run status, metric, or claim.

The objective is to recover at least 15 GiB while retaining enough compact
evidence to audit every run and every favorable, negative, or failed result.
The cleanup is storage retention, not scientific selection.

Eligible registered artifacts are restricted to unpromoted modern runs
(`r_*`) classified as `diagnostic` or `exploratory_screen`:

- every checkpoint or prediction artifact from a failed run;
- completed-run prediction artifacts at least 8 MiB; and
- completed-run checkpoint artifacts at least 128 MiB.

The following are protected from this decision:

- `data/raw/`, `data/clinical/`, root `data.tar.gz`, and all source data;
- all `data/processed/` inputs and split/dataset registries;
- source, configuration, tests, environments, campaign contracts, and queue or
  run records;
- all summaries, metrics, logs, provenance, diagnostics, reports, and report
  tables;
- every promoted run;
- every `locked_final` and `validation_confirmation` artifact;
- the complete imported legacy artifact tree, including its diagnostic and
  exploratory runs, because it is one checksum-bound historical import; and
- modern checkpoints below 128 MiB and predictions below 8 MiB, which retain
  the current GeneMAE checkpoints and compact prediction summaries.

Rebuildable local caches and clearly inactive scratch work may also be deleted:

- `cache/legacy_package_cache/`, `cache/python/`, and `cache/pytest/`;
- `.pytest_cache/` and repository `__pycache__/` directories;
- `scratch/temporary/`, the abandoned `.hybrid-materializer-check-*` tree,
  the completed post-hoc token-sensitivity active workspace, and the empty
  `scratch/active_runs/legacy_outputs/` directory.

Locked campaign receipts, materialized configurations, preprocessing receipts,
state, SQLite backups, reports, and exports remain intact.

## Rationale and alternatives

The initial inventory found approximately 47 GiB in the repository:
approximately 37 GiB of artifacts, 8 GiB of data, and 1.6 GiB of rebuildable
cache. Failed bundles alone occupy only about 0.12 GiB, so deleting only failed
runs would not materially address disk use. Conversely, deleting the 15.2 GiB
legacy locked-final payload would discard the strongest retained evidence and
is outside this decision.

The selected thresholds remove large reproducible payloads without choosing a
favorable seed or result. Compact aggregate evidence and small operationally
useful checkpoints remain. The strongest alternative is to preserve every
successful immutable bundle; that was the previous decision, but it does not
meet the user's current storage objective.

## Safety and acceptance criteria

Before deletion, the workflow must:

1. observe no queued, claimed, running, or finalizing experiment;
2. create an SQLite-consistent registry backup;
3. write a complete deletion plan containing run ID, artifact ID, lifecycle
   stage, kind, path, size, and registered SHA-256;
4. verify every selected file is a regular non-symlink beneath the configured
   modern artifact root and matches its registered size and SHA-256;
5. abort on a missing checksum, path escape, marker mismatch, promotion, or
   changed registry row; and
6. record the plan checksum before unlinking any payload.

After deletion, the workflow must:

1. retain original paths, sizes, and checksums in the registry while changing
   their artifact status to `deleted_by_retention`;
2. change affected checkpoint-catalog verification status to
   `deleted_by_retention` without deleting the catalog row;
3. write a checksum-bound completion receipt and preserve the pre-cleanup
   database backup;
4. verify selected paths are absent and retained registered artifacts still
   match their checksums;
5. run focused retention tests and the project doctor;
6. confirm protected paths and compact reports still exist; and
7. record final disk use and `git status --short`.

Deletion is irreversible for the large payload bytes: the SQLite backup and
manifests preserve metadata, not model weights or predictions. A failure during
application must leave registry rows visibly pending or deleted rather than
claiming the files are present.

## Reviewed plan

The dry run selected 149 registered files from 96 modern runs: 53
`last.ckpt` files and 96 `fit.jsonl` prediction files, totaling
18,028,054,661 bytes (16.79 GiB). Of those rows, 147 belong to completed
diagnostic or exploratory runs and two belong to one failed diagnostic run.
The plan checksum is
`bdf294570e060520d850a497dc23dc41adf728a21f91fb92751f6357cb3bae04`.

Review confirmed that the plan contains no `lr_*` run, legacy path,
`locked_final` run, `validation_confirmation` run, or completed payload below
its threshold. The two MyJJu entries are from one failed diagnostic attempt;
all eight completed MyJJu checkpoints (one diagnostic and seven production
exploratory checkpoints) remain. The pre-application project doctor passed
with no issues, SQLite integrity `ok`, no live queue or run state, and
38.603 GiB free.

## Outcome and verification

Application completed at `2026-08-02T14:24:34.962562Z`. All 149 payloads
matched their registered SHA-256 before deletion. The workflow deleted
18,028,054,661 registered bytes (16.79 GiB), marked all 149 artifact rows
`deleted_by_retention`, marked the 53 corresponding checkpoint-catalog rows
the same way, and left zero pending deletions. No planned path remains. The
application receipt checksum is
`aee977ba415c6621ff05b5e0cf3157ff33c5a341ef561817d2eb078e21b016e0`.

The named rebuildable cache, pytest cache, bytecode cache, and inactive scratch
paths were also removed. Final allocated project size is 30,387,785,728 bytes
(28.3 GiB), including 21,663,092,736 artifact bytes (20.2 GiB); `cache/` is
empty and active scratch plus retained receipts use about 5.2 MiB. Available
filesystem space increased from 38.603 GiB in the pre-application doctor to
56.992 GiB in the final doctor, a measured gain of 18.389 GiB.

Protected content remained intact:

- all 209 `locked_final` or `validation_confirmation` registered artifacts are
  present;
- all eight completed MyJJu checkpoints are present (661,900,984 bytes);
- the full 18,509,012,992-byte allocated legacy artifact tree remains;
- `data/raw/`, `data/processed/`, and `data/clinical/` remain with 11, 91, and
  four files respectively; and
- reports, metrics, logs, provenance, run markers, summaries, and registry
  records remain.

The first post-application `verify-artifacts` run exposed an infrastructure
gap: the registry understood tombstones, but canonical bundle verification
still reported the intentionally absent manifest entries as ordinary missing
files. The verifier was changed to accept only registry-supplied checkpoint or
prediction tombstones whose relative path, size, and SHA-256 exactly match the
immutable bundle manifest. Any unregistered absence, changed byte, path escape,
or tombstoned path that still exists remains an error. Focused production spot
checks and the complete registry/bundle verification then passed with no
issues.

Final verification results:

- current and backup SQLite integrity: `ok`;
- deletion-plan and application-receipt checksums: valid;
- `verify-artifacts`: valid, no registry or bundle issues;
- `doctor`: `ok`, no issues or warnings, no live/claimed/queued run;
- full repository tests: 891 passed, one skipped because optional `pyarrow` is
  not installed, and three upstream deprecation warnings; and
- `git diff --check`: passed; unrelated pre-existing worktree changes were
  preserved.

The deleted weights and prediction tables are not recoverable from the report
or registry metadata. The pre-deletion SQLite snapshot preserves the former
registry state, not the deleted payload bytes.

## Commands

The reviewed plan and application use the repository-root entry point:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/maintenance/compact_experiment_artifacts.py \
  --database state/tracking/bagm.sqlite3 \
  --decision-id cleanup_20260802 \
  --output-dir reports/retention/cleanup_20260802

PYTHONPATH=src /venv/main/bin/python \
  scripts/maintenance/compact_experiment_artifacts.py \
  --database state/tracking/bagm.sqlite3 \
  --decision-id cleanup_20260802 \
  --output-dir reports/retention/cleanup_20260802 \
  --apply
```

Verification commands are:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/infrastructure/test_artifact_retention.py
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 doctor
```
