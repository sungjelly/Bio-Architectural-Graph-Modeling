# Operations

The local CLI is:

```bash
PYTHONPATH=src python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 <command>
```

Use `--help` on the root command and every subcommand. The database under
`state/tracking/` is the one authoritative registry. SQLite uses WAL mode, a
busy timeout, foreign keys, versioned schema upgrades, parameterized SQL, and
transactions for state transitions. Run `doctor` after infrastructure changes;
it includes database integrity and local contract checks.

## Setup and inspection

```bash
PYTHONPATH=src python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 doctor

PYTHONPATH=src python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 list-queue
```

Supported commands are `doctor`, `register-dataset`, `register-split`,
`create-campaign`, `enqueue-experiment`, `enqueue-sweep`, `worker`,
`list-queue`, `show-run`, `index-checkpoints`, `list-checkpoints`,
`show-checkpoint`, `resolve-checkpoint`, `export-checkpoint-catalog`,
`summarize-variants`, `export-leaderboard`, `promote-run`,
`verify-artifacts`, and `import-legacy`.

Register a campaign before enqueueing its experiments. Review the fully
resolved configuration, scientific/reproduction identifiers, dataset/split
registrations, test policy, and disk estimate. An enqueue example is:

```bash
PYTHONPATH=src python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  enqueue-experiment \
  --campaign-id cmp_20260724_edge_feature_ablation \
  --config configs/experiment/edge_feature_ablation_g1.yaml \
  --priority 0 \
  --max-attempts 1 \
  --gpu 0
```

When no explicit subprocess command is supplied, the CLI derives the existing
spatial-benchmark training entry point. Inspect
`enqueue-experiment --help` before activation. The example campaign and sweep
are deliberately not enqueued by repository setup or validation.
Arbitrary `--command` and `launcher.command` vectors are restricted to gated
test-only fixtures; scientific argv is derived from resolved configuration so
it cannot evade `scientific_id`/`repro_id`.

## Checkpoint indexing and discovery

Primary run IDs are immutable. Historical runs retain `lr_*` IDs; future
workers create `r_*` IDs. Both may have a preferred date-free semantic alias.
The `artifacts/runs/YYYY/MM/` hierarchy is physical storage only. Browse using:

```text
campaign → lifecycle stage → study axis → scientific variant
         → seed / fold / attempt
```

Schema v3 stores this view in `run_aliases`, `run_categories`, and
`checkpoint_catalog`. Index all registered checkpoint artifacts idempotently:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  index-checkpoints
```

Use `--run-id <primary-or-alias>` to limit indexing. Verification is on by
default. `--no-verify` is for an explicit diagnostic only; it must not support
promotion or interpretation. `--update` is required to replace conflicting
derived semantics and must follow a reviewed classification change.

List by scientific category:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  list-checkpoints \
  --stage locked_final \
  --study-axis locked_ladder \
  --model g2 \
  --edge-features enabled \
  --limit 50
```

Show or resolve one verified best checkpoint using either primary ID or
preferred alias:

```bash
RUN_REF=hist.diagnostic.runtime-smoke.b0.self.p-n-b.disabled.d128.s000.fna.ana.vaf897cf26410.x1cd0a0ff3c83

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  show-checkpoint "$RUN_REF" --role best

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  resolve-checkpoint "$RUN_REF" --role best
```

The exact command for the reviewed generated catalog is:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  export-checkpoint-catalog \
  --output exports/runs/checkpoints/catalog_20260724T180010Z \
  --links
```

That existing directory contains `catalog.jsonl`, `catalog.csv`,
`summary.json`, and a relative `by-stage/` symlink view. It is generated and
non-authoritative; do not overwrite it merely to refresh a display. Use a new
timestamped export path for a later snapshot.

The current index has 158 verified best checkpoints totaling 1,499,185,266
bytes: 9 diagnostic, 96 exploratory-screen, 3 validation-confirmation, and 50
locked-final. Nine exact-content groups cover 20 records and 84,324,336
redundant bytes. The files were not copied, moved, loaded, rewritten, deleted,
or hard-linked during indexing. Duplicate records can have different
provenance, so physical deduplication requires a separate explicit retention
decision.

Interpretation guardrails:

- `diagnostic` checkpoints validate runtime behavior only.
- `exploratory_screen` checkpoints support selection, not confirmatory claims.
- `validation_confirmation` checkpoints support only their locked validation
  question.
- `locked_final` checkpoints are conclusion-bearing only within the owning
  campaign's units, controls, and prespecified claim.
- Fold and attempt are unknown for all 158 legacy runs. Visible `fna`/`ana` and
  false knownness flags override registry placeholders.
- Never choose a checkpoint from its single-run metric or best seed. Use
  `summarize-variants` and the prespecified complete seed/fold aggregate first.

## One-GPU worker

Before a real run, inspect GPU processes and memory, framework visibility,
CUDA, free disk, and the registered inputs. Never displace an existing process.
Start one worker manually with:

```bash
PYTHONPATH=src python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  worker --gpu 0 --min-free-gb 25
```

`--once` processes at most one claim. `--auto-retry` permits only declared
attempt retries; it never changes batch size or scientific parameters.
Heartbeat, polling, and stale thresholds are configurable. Test-only jobs are
refused unless the explicit `--allow-test-jobs` gate is used.

The worker atomically claims one queued job, obtains
`state/locks/bagm-worker.lock`, sets `CUDA_VISIBLE_DEVICES`, creates a unique
`r_*` run ID, derives a preferred semantic alias and category from the resolved
top-level `classification` mapping, and launches a fresh subprocess. Missing
classification stays `unknown`; it is never guessed from the run timestamp.
Output starts in
`scratch/active_runs/<run_id>/logs/` and is preserved in the finalized or
failed run bundle. One worker permits one GPU-training subprocess by default.
GPU-specific claims are matched to the worker GPU before claiming. A durable
PID/start-tick record prevents a replacement worker from launching while a
child from a crashed parent may still be alive.

Queue states are `queued`, `claimed`, `running`, `completed`, `failed`,
`pruned`, `cancelled`, and `stale`. The worker heartbeats while a subprocess
runs and handles SIGTERM/SIGINT without claiming another job. A missing
heartbeat is marked stale only after the configured conservative threshold;
inspect the process, lock, logs, disk, and artifact state before retrying.
Stale scratch is registered as `deferred_stale_scratch` and is not moved while
a detached writer cannot be ruled out.

## Failure and recovery

Failures are classified as CUDA out-of-memory, NaN/infinite loss, nonzero exit,
missing heartbeat, insufficient disk, missing dataset, invalid configuration,
or artifact-finalization failure where applicable. Every failed attempt keeps
its resolved configuration, provenance, logs, exception, and partial metrics.

A retry creates a new queue job and run ID with `retry_of` links. It uses the
same canonical configuration and command. Never silently reduce batch size,
change a graph, substitute data, open a test split, or omit a failed seed. If
the failure suggests a scientific or resource change, create a new reviewed
variant instead of calling it a retry.

If a worker stops:

1. Confirm the old process is absent; do not kill or restart unrelated work.
2. Inspect `list-queue`, the lock, heartbeat, active run, stdout/stderr, and
   free disk.
3. Run `doctor` and `verify-artifacts` for any possibly finalized run.
4. Classify a genuinely stale job and preserve its attempt before retrying.
5. Restart one worker only after the lock and GPU are safely available.

No run is successful because its process returned zero. Finalization validates
the bundle, declared hashes, summary, final metrics, best checkpoint, and
canonical validation predictions. It commits a recoverable `finalizing` run,
writes the checkpoint catalog entry, writes a checksum-bound `_SUCCESS`, and
then atomically completes run and job.
A restarted worker reconciles either an unmarked or marked `finalizing` bundle
before claiming new work; `doctor` reports unresolved finalization records.

Production prediction finalization requires a stable, untracked
`BAGM_SAMPLE_KEY_SALT` of at least 16 bytes. The adapter stores one declared
validation-mask evaluation with HMAC sample keys and keeps the complete native
NPZ as restricted provenance. Never put the salt in Git or a report.

## Multi-GPU matrix launcher

`scripts/sweeps/launch_matrix.py` schedules independent jobs across the GPUs
allocated to the process. Its default selection uses `BAGM_GPU_IDS` when set,
then `CUDA_VISIBLE_DEVICES`, and otherwise discovers the indices reported by
`nvidia-smi`. Override the selection explicitly with `--gpus` for an individual
invocation:

```bash
export BAGM_GPU_IDS=0,1,2,3
PYTHONPATH=src /venv/main/bin/python scripts/sweeps/launch_matrix.py \
  --prepared <prepared-artifact> \
  --matrix <matrix.yaml> \
  --output-root <output-root>
```

Host-level allocation does not rewrite immutable historical campaign
contracts or receipts that recorded the hardware on which they ran.

## Disk, retention, and backup

The worker refuses new work below `--min-free-gb`. Monitor canonical artifacts,
active scratch, caches, SQLite WAL files, and logs. Do not delete history
ad hoc. Apply the versioned retention policy in
`configs/schema/run_archive_v1.yaml`: successful runs keep core provenance,
metrics, validation predictions, best checkpoint, and logs; promoted runs may
keep larger interpretation/test artifacts; failed runs keep diagnosis but no
unnecessary checkpoints. Checkpoint catalog duplicate annotations do not
authorize deletion. Retention and physical deduplication remain separate,
reviewed decisions.

An approved payload deletion must use a checksum-bound plan and retain the
artifact registry row. Mark an artifact `retention_deletion_pending` before
unlinking it, then `deleted_by_retention` only after its registered path is
absent. Apply the same tombstone to checkpoint-catalog verification status.
`doctor` and `verify-artifacts` treat an absent tombstoned payload as expected,
but report pending deletions or a tombstoned path that still exists. Preserve
the decision, plan, receipt, compact run evidence, and an SQLite-consistent
pre-deletion snapshot together.

Back up the SQLite database together with canonical artifacts using an
SQLite-consistent snapshot and an approved protected destination. A database
without artifact bundles, or bundles without their registry/provenance, is not
a complete backup. The project performs no remote upload automatically.

## Service template

`ops/systemd/bagm-worker.service` is reviewable only. It uses
`/workspace/Bio-Architectural-Graph-Modeling`, `/venv/main/bin/python`, one GPU
worker, clean SIGTERM shutdown, restart delay,
and logs under `state/logs/`. Do not install, enable, or start it during
unattended setup. Review paths, environment, GPU allocation, and persistence with the
operator before use.
