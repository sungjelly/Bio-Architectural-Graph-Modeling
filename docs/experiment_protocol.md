# Experiment Protocol

## Scientific objects

A **campaign** is one scientific question, ablation, sweep, or related batch.
A **variant** is one fixed scientific configuration and excludes seed, fold,
attempt, timestamps, host/GPU identity, runtime status, and output paths. A
**run** is one execution of a variant for one seed, fold, and attempt. An
**evaluation** applies one checkpoint to one named registered dataset and
split. An **artifact** is a declared file or directory produced by a run or
evaluation.

Keep estimands separate. Partial-gene masking tests reconstruction using
visible genes in the same cell; whole-node masking tests neighborhood and
permitted-covariate prediction; spatial-block masking tests extrapolation into
contiguous regions; cross-cell sensitivity is a model perturbation estimand.
None alone establishes a biological mechanism or causality.

## Identifier contract

All hashes use strict, key-sorted, whitespace-free canonical JSON and SHA-256.
NaN and infinity are invalid.

`campaign_id` is human-readable:

```text
cmp_<YYYYMMDD>_<slug>
```

`scientific_id` is `sci_` plus the configured leading SHA-256 characters of
the recursively filtered scientific configuration. The implementation
includes model, masking regime, dataset and split identities, features, graph,
trainer, and evaluation choices. It excludes these exact execution or
organizational field names wherever they occur:

```text
seed, random_seed, model_seed, mask_seed, fold, fold_id,
attempt, attempt_number, retry, retry_of, campaign, campaign_id,
variant_label, display_name,
timestamp, created_at, updated_at, start_time, started_at,
end_time, ended_at, finished_at, duration, duration_seconds,
hostname, host, gpu, gpu_id, gpu_model, requested_gpu, device,
cuda_visible_devices, worker_id, job_id, run_id, status, runtime,
launcher, heartbeat_seconds, stale_after_seconds,
disk_safety_min_free_gb, capture_stdout, capture_stderr,
output_dir, output_path, artifact_dir, artifact_path,
log_dir, log_path, state_root, scratch_root, cache_root, export_root
```

The exported `SCIENTIFIC_EXCLUDED_FIELDS` constant is the source of truth. Any
change to that set is an identifier-schema change and requires tests and
documentation. `launcher` remains operational rather than scientific only
because production argv is deterministically derived from the included model,
masking, dataset, feature, graph, trainer, and evaluation fields; arbitrary
launcher commands are rejected outside explicitly gated test fixtures.

`repro_id` is `rep_` plus a SHA-256 prefix over:

- the canonical filtered scientific configuration;
- Git commit;
- the dirty-working-tree fingerprint, explicitly `null` for a clean tree;
- dataset fingerprint;
- split fingerprint;
- preprocessing version;
- environment fingerprint and/or container fingerprint.

Different code, dirty changes, data, split assignments, preprocessing, or
environment therefore cannot silently share a reproduction ID.

`run_id` has the form:

```text
r_<UTC YYYYMMDDTHHMMSSZ>_<scientific-8>_s<seed-3>_f<fold-2>_a<attempt-2>_<unique-suffix>
```

The suffix makes retries and simultaneous launches unique. A retry receives a
new run ID and records `retry_of`; it does not change scientific parameters.

That `r_*` form applies to future worker-created runs. The 158 imported
historical executions retain their immutable `lr_*` primary IDs. Never rewrite
a primary ID to make it more readable: registry foreign keys, manifests,
metrics, and artifact checksums depend on it.

Each indexed run may also have a preferred, date-free semantic alias. Its
visible components encode lifecycle stage, study axis, model, graph, masking,
edge-feature state, embedding dimension, and the known execution coordinates;
short hashes keep the alias collision-safe. For example:

```text
hist.diagnostic.runtime-smoke.b0.self.p-n-b.disabled.d128.s000.fna.ana.vaf897cf26410.x1cd0a0ff3c83
```

`fna` and `ana` mean the source did not record fold or attempt. They must not
be replaced with the importer placeholders `f00` and `a01`. Schema-v3
`run_categories.seed_known`, `fold_known`, and `attempt_known` are the source
of truth for whether an execution coordinate is evidence. Aliases are lookup
keys only; canonical paths and primary IDs do not change.

The semantic hierarchy is:

```text
campaign → lifecycle stage → study axis → scientific variant
         → seed / fold / attempt
```

The physical `artifacts/runs/YYYY/MM/<run_id>/` partition is only a scalable
storage layout. Dates are provenance and operational coordinates, not the
primary analysis categories.

## Seeds, folds, and units

Define the biological generalization unit before splitting. Patients or
independent samples normally define replication. Cells, spatial blocks, mask
replicates, and model seeds quantify different sources of variation and must
not be promoted to patient-level replication. Use grouped patient/sample
splits for generalization and spatial blocks where local copying is plausible.
Fit data-dependent preprocessing using training units only.

Prespecify all seeds and folds. Report every completed, failed, pruned, and
excluded attempt with reasons. Aggregate all expected runs and never present
the best seed as the scientific result. The protected legacy normal-core
benchmark is within-core and descriptive; it cannot support patient-level
generalization. Its seed is known, but fold and attempt are unknown for all
158 imported runs.

## Controls, lock, and test isolation

Each campaign must declare its primary hypothesis, credible alternatives,
discriminating predictions, estimand, metric and direction, controls, nulls,
acceptance/falsification criteria, leakage risks, compute, stop criteria, and
artifacts before conclusion-bearing execution. If outcomes were examined
first, label the analysis exploratory.

For graph utility, compare against cell-autonomous, morphology, broad-field,
uniform-neighbor, non-spatial, fixed-edge, and applicable established
baselines under the same split and budget. Use mechanism-breaking nulls that
preserve relevant confounders. Lock preprocessing, graph, mask bundles,
selection metric, model family, seed/fold set, and analysis before opening a
sealed test split. Test data must not select hyperparameters, interpretations,
or thresholds.

## Run lifecycle

An attempt starts in `scratch/active_runs/<run_id>/`. It records resolved
configuration and provenance before training, appends metric events during
training, saves the explicitly monitored best checkpoint, writes predictions
and diagnostics, and validates the bundle. The canonical destination is:

```text
artifacts/runs/YYYY/MM/<run_id>/
```

The required and optional contract is versioned in
`configs/schema/run_archive_v1.yaml`. `_SUCCESS` is written only after
configuration, provenance, summary, final metrics, best checkpoint, and
canonical validation predictions and declared checksums validate. A durable
`finalizing` state makes marker/registry completion recoverable after a hard
crash. Exceptions preserve logs, traceback, partial
metrics, and configuration under `_FAILED`. Successful bundles are immutable;
post-hoc evaluations are separately versioned.

Per-sample prediction tables follow
`configs/schema/predictions_v1.yaml`. Sample keys must be stable and
non-identifying. Save numeric curve data, not only rendered figures. Never
copy a dataset into a run directory.

## Checkpoint catalog and interpretation

Every checkpoint must first be an immutable registered artifact. Schema v3
then records its searchable semantics in `checkpoint_catalog` and joins it to
`run_aliases` and `run_categories`. The entry includes role, best epoch,
monitored metric/mode/value, checksum-backed artifact identity, retention
class, duplicate-content group, and verification status. This metadata makes
a checkpoint discoverable; it does not make the run scientifically eligible.

The historical backfill contains one verified best checkpoint for each of 158
runs:

| Lifecycle stage | Runs | Interpretation boundary |
|---|---:|---|
| diagnostic | 9 | runtime/smoke validation only |
| exploratory_screen | 96 | model or hyperparameter selection only |
| validation_confirmation | 3 | locked validation confirmation |
| locked_final | 50 | conclusion-bearing only within campaign limits |

The files total 1,499,185,266 bytes. Nine exact-content duplicate groups cover
20 records; 84,324,336 bytes are redundant copies. They remain physically
preserved because different run/stage provenance can point to identical model
state. Duplicate annotation is not a retention decision, and neither hard-link
replacement nor deletion is automatic.

Checkpoint interpretation must proceed from a prespecified variant aggregate,
not the best per-run metric. First select the eligible campaign, stage, study
axis, scientific/reproduction ID, expected seed/fold set, controls, and metric
direction. Then inspect the run-level checkpoint and predictions. Diagnostic
or exploratory checkpoints cannot be promoted into confirmatory evidence
because they look favorable.

Future resolved configurations declare `classification.lifecycle_stage`,
`classification.study_axis`, `classification.retention_class`,
`classification.classification_confidence`, and optionally
`classification.source_batch`. The worker derives a semantic alias and
category record before training, and indexes the verified checkpoint during
finalization and crash reconciliation. Missing classification remains visibly
`unknown`; the worker does not infer it from a date or filename.

## Aggregation and promotion

Variant summaries group by `repro_id` and include expected, completed, and
failed counts; mean, standard deviation, minimum, and maximum of the primary
metric; relevant secondary/calibration metrics; duration and peak VRAM;
and best/worst run IDs for audit. Ranking uses complete prespecified variant
aggregates, not a favorable execution.

The generated catalog at
`exports/runs/checkpoints/catalog_20260724T180010Z/` is a non-authoritative
discovery/export view. Its relative links do not replace registered artifact
paths or immutable manifests.

Promotion is an explicit retention and evaluation decision. It does not alter
the original run, erase failures, or transform exploratory evidence into
confirmation. Promoted runs may retain test/external predictions, selected
additional checkpoints, interpretation arrays, or larger embeddings. The
formula of any operational `selection_score` must be versioned and explicit;
it is not the scientific outcome.

For every major result report the question, observed result, strongest
alternative, controls, remaining uncertainty, and maximum defensible claim.
Stable, faithful, null-calibrated, patient-replicated, independently supported,
and perturbational evidence remain separate dimensions.
