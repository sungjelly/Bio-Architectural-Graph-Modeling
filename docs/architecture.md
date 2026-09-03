# Project Architecture

Bio-Architectural Graph Modeling is one project rooted at
`/workspace/BAGM`. The enclosing `/workspace` may
contain other projects; do not add another wrapper or multi-project hierarchy
inside this repository. Reusable code lives in the existing
`spatial_benchmark` package; scientific campaigns compose that code rather
than copying model implementations.

## Top-level ownership

| Path | Owner and contract |
|---|---|
| `AGENTS.md`, `README.md`, project metadata | Repository-wide policy, scientific context, packaging, and developer entry points. |
| `configs/` | Composable, reviewable model, masking, dataset, feature, graph, trainer, evaluation, experiment, sweep, launcher, and schema definitions. |
| `src/spatial_benchmark/` | Reusable models, data/graph logic, training, tracking, orchestration, paths, and identifier APIs. |
| `scripts/` | Thin entry points grouped by train, evaluate, data, sweeps, analysis, maintenance, or legacy purpose. |
| `tests/` | Unit, integration, and smoke checks using non-identifying synthetic fixtures where possible. |
| `notebooks/` | Exploration or analysis only; notebooks are not canonical pipelines. |
| `data/raw/` | Immutable local source data. Never rewrite it. |
| `data/clinical/` | Immutable restricted clinical inputs. Never track or expose row-level contents. |
| `data/external/` | Immutable third-party or externally supplied inputs. |
| `data/interim/` | Rebuildable partial preprocessing output. |
| `data/processed/` | Versioned reusable processed arrays or tables. |
| `data/graphs/` | Versioned reusable graph objects and graph-construction manifests. |
| `data/splits/` | Protected row-level split assignments; keep untracked. |
| `data/registry/` | Tracked non-identifying dataset and split metadata and fingerprints. |
| `experiments/definitions/` | Bounded scientific questions and variant definitions. |
| `experiments/campaigns/` | Campaign task contracts, not mutable run payloads. |
| `experiments/templates/` | Starting points that must be reviewed before activation. |
| `artifacts/runs/YYYY/MM/<run_id>/` | Canonical immutable run bundles after finalization. `YYYY/MM` is a physical storage partition, not scientific categorization. |
| `artifacts/legacy_runs/` | Preserved historical batches indexed without rewriting their native formats. |
| `artifacts/legacy_tracking/` | Preserved historical tracker state that is not authoritative for new runs. |
| `artifacts/promoted/` | Versioned references or material explicitly retained after a promotion decision. |
| `artifacts/quarantine/` | Invalid or unverifiable output retained for audit, never silently treated as a run. |
| `state/tracking/` | The one authoritative local SQLite registry and its schema state. |
| `state/queue/`, `state/locks/`, `state/pids/`, `state/logs/` | Mutable local orchestration state. |
| `scratch/active_runs/` | In-progress attempt output; a successful run is finalized from here. |
| `scratch/preprocessing/`, `scratch/temporary/` | Rebuildable, non-canonical working data. |
| `cache/` | Rebuildable dataset, graph, model, package, and test caches. |
| `exports/` | Derived run exports, variants, leaderboards, failure tables, and semantic checkpoint views; never a second registry. |
| `reports/` | Cross-run figures, tables, analyses, and campaign reports with numeric source data. |
| `results/` | Curated conclusion records organized by experiment type and method family; references immutable runs/reports and is never a second payload archive. |
| `docs/` | Architecture, protocol, configuration, operations, data, and metrics contracts. |
| `ops/` | Reviewable service, scheduler, container, and maintenance templates; nothing is installed automatically. |

Campaign-specific summaries may live with the owning campaign, but registry-run
payloads have one canonical location under `artifacts/runs/`. A campaign or
workflow result must refer to a canonical run by immutable ID and checksum; it
must not copy datasets or silently choose a favorable upstream run.

The semantic run hierarchy is orthogonal to storage:

```text
campaign
└── lifecycle stage
    └── study axis
        └── scientific variant
            └── seed / fold / attempt
```

Historical primary keys remain `lr_*`; future worker-created primary keys use
`r_*`. A preferred semantic alias makes the hierarchy readable without
changing immutable IDs, foreign keys, manifests, or paths.

## Path boundary

All reusable path resolution goes through `spatial_benchmark.paths`. The module
exports `PROJECT_ROOT`, `CONFIG_ROOT`, `DATA_ROOT`, `ARTIFACT_ROOT`,
`STATE_ROOT`, `SCRATCH_ROOT`, `CACHE_ROOT`, `EXPORT_ROOT`, `REPORT_ROOT`, and
`RESULT_ROOT`.
`BAGM_ROOT` selects the project root. The corresponding overrides are
`BAGM_CONFIG_ROOT`, `BAGM_DATA_ROOT`, `BAGM_ARTIFACT_ROOT`,
`BAGM_STATE_ROOT`, `BAGM_SCRATCH_ROOT`, `BAGM_CACHE_ROOT`,
`BAGM_EXPORT_ROOT`, `BAGM_REPORT_ROOT`, and `BAGM_RESULT_ROOT`. Relative
overrides resolve under the selected project root. Reusable code must not
depend on the caller's working directory or embed a server home directory.

## Runtime ownership

One SQLite database at `state/tracking/bagm.sqlite3` is authoritative for new
campaigns, global variants, campaign-variant memberships, runs, evaluations,
metrics, artifacts, datasets, splits, queue jobs, and failures. Schema v3 adds:

- `run_aliases`, for globally unique canonical/semantic aliases and one
  preferred display alias per run;
- `run_categories`, for lifecycle stage, study axis, variant keys, explicit
  seed/fold/attempt knownness, retention, classification rules, and confidence;
- `checkpoint_catalog`, for checkpoint role, best epoch, monitored metric,
  retention, duplicate-content annotation, and verification state.

Canonical artifact directories remain independently verifiable: the database
is an index and state machine, not the only copy of scientific output.

The historical backfill indexes 158 verified best checkpoints without moving
or loading them: 9 diagnostic, 96 exploratory-screen, 3
validation-confirmation, and 50 locked-final. The checkpoints total
1,499,185,266 bytes. Nine exact-content groups cover 20 records and 84,324,336
redundant bytes; the catalog records these relationships but performs no
physical deduplication. The generated view at
`exports/runs/checkpoints/catalog_20260724T180010Z/` contains JSONL, CSV,
summary, and optional relative links. SQLite and immutable manifests remain
authoritative.

A worker atomically claims one job, obtains the project lock, and creates a
fresh subprocess. The attempt writes to `scratch/active_runs/<run_id>/`.
Finalization validates required files and hashes, moves the completed bundle
to the year/month archive, commits a recoverable `finalizing` transition,
writes checksum-bound `_SUCCESS`, indexes checkpoint metadata, and atomically
completes run and job. The worker derives category semantics from explicit
resolved-config `classification` fields; missing classification stays
`unknown` rather than being guessed from a timestamp or filename.
`_FAILED` and `_PRUNED` preserve partial evidence. A zero exit
code alone never establishes completion.

Successful run directories are immutable. A post-hoc evaluation must have its
own versioned evaluation record and artifact location. Generated runtime state,
active scratch output, caches, and canonical run payloads are ignored by Git;
source, configs, schemas, registries, tests, documentation, and small metadata
remain reviewable.
