# Peak-VRAM Registry Reconciliation

## Cause and scope

The pooled runner originally wrote the unit-explicit summary field
`peak_vram_gib`. Schema-v3 queue finalization read only the historical
`peak_vram_gb` spelling. The immutable bundle, final metrics, and resource
diagnostic therefore retained the measurement, while the corresponding run
column could remain null.

The compatibility fix has two parts:

1. pooled summaries now write both spellings with the same binary-GiB value;
2. newly started queue workers accept either spelling and reject conflicting
   dual values.

This is execution metadata only. It does not change model weights, masks,
graphs, objectives, metrics, seeds, or scientific gates.

## Worker behavior during the active campaign

Python workers and already-running experiment subprocesses do not reload source
files. Consequently:

- a pooled subprocess that started before the fix will still emit only
  `peak_vram_gib`, and its already-running worker will leave the registry field
  null;
- a pooled subprocess started after the fix emits both fields, so even an older
  long-lived worker can populate the historical registry column;
- a worker restarted after the fix can populate the column from either field.

Workers must not be restarted merely to repair this metadata. Reconcile after
all production jobs reach a terminal state.

## Frozen reconciliation procedure

The reconciliation is a two-phase, fail-closed metadata repair.

### 1. Build a read-only signed plan

Do not begin while any job in this campaign is queued, claimed, running, or
finalizing. Resolve retry chains and select only the single completed attempt
for each expected pilot or production slot. There must be exactly two completed
pilots and fourteen completed production runs.

For every selected run:

1. Require registry status `completed`, a single `_SUCCESS` marker, and a
   campaign-matching immutable artifact path.
2. Run `spatial_benchmark.run_archive.verify_run_bundle` and
   `Registry.verify_artifacts(run_id=...)`.
3. Verify that the registered `summary.json` path, byte size, and SHA-256 match
   both the file and the bundle checksum manifest.
4. Require summary `run_id`, campaign, pooled evaluation protocol, and alias set
   `ANC-01` through `ANC-10` to match the registered run.
5. Read the binary-GiB value independently from:
   - `summary.json`: `peak_vram_gib`, or the compatibility
     `peak_vram_gb`;
   - `metrics/final.json`: `resource/peak_vram_gib`;
   - `diagnostics/resource_usage.json`:
     `peak_allocated_vram_gib`.
6. Require all present values to be finite, nonnegative, and equal within
   `1e-12` absolute or relative tolerance. A missing source or disagreement is
   a blocker, not an invitation to choose one value.
7. Classify the registry action:
   - null registry value: `backfill`;
   - matching non-null value: `already_consistent`;
   - differing non-null value: `conflict` and abort.

Save an append-only JSON plan containing the campaign and receipt checksums,
run IDs, selected attempts, bundle-content hashes, source-file hashes, all
three measured values, current registry values, proposed values, and actions.
Compute its checksum as canonical SHA-256 with the top-level `checksum` field
omitted.

### 2. Apply the reviewed plan once

Application requires the exact reviewed plan checksum. Before writing, repeat
all bundle, artifact, source-agreement, retry-lineage, and registry-current-value
checks. Create a SQLite online backup under `state/tracking/backups/`.

Use one `BEGIN IMMEDIATE` transaction. For each `backfill` item, execute a
conditional update equivalent to:

```sql
UPDATE runs
SET peak_vram_gb = :verified_binary_gib,
    updated_at = :applied_at
WHERE run_id = :run_id
  AND campaign_id = :campaign_id
  AND status = 'completed'
  AND peak_vram_gb IS NULL;
```

Require one changed row per planned backfill. Do not overwrite a non-null
value. Roll back the complete transaction on any row-count or state mismatch.

After commit, rebuild the read-only plan. Every selected run must now be
`already_consistent`. Preserve a checksum-bound application receipt containing
the reviewed plan checksum, database-backup path and checksum, transaction
time, changed run IDs, old/new values, and post-verification checksum.

Finally run:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 doctor
```

The plan, application receipt, and post-verification output belong with the
campaign report. The immutable run bundles must not be edited.

## Entry point

Planning is the default. It audits the signed receipts, complete retry
lineages, immutable bundles, registered artifacts, and all three independent
VRAM sources before creating an append-only report file:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/reconcile_pooled_peak_vram_registry.py
```

Application is intentionally separate and requires the exact checksum printed
by the reviewed plan:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/reconcile_pooled_peak_vram_registry.py \
  --apply-plan reports/analyses/adjacent_normal_10core_pooled_hybrid_ensemble/registry_reconciliation/<reviewed-plan>.json \
  --expected-plan-checksum <exact-canonical-sha256>
```

The application path re-runs the complete audit, creates an online SQLite
backup under `state/tracking/backups/`, performs only conditional null-value
updates in one `BEGIN IMMEDIATE` transaction, rebuilds the plan, runs artifact
verification and project doctor, and writes a checksum-bound append-only
receipt. It never modifies an immutable run bundle.
