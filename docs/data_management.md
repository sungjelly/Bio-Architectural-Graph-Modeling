# Data Management

Read `docs/legacy_data_guide.md` before accessing the current CosMx snapshot.
It defines keys, joins, streaming rules, panel controls, clinical-label
ambiguity, and known integrity exceptions. Raw, clinical, and external source
data are immutable local inputs.

## Data zones

- `data/raw/`: original source exports. Never modify, normalize, rename, or
  deduplicate in place.
- `data/clinical/`: restricted clinical inputs. Never expose row-level
  contents, direct identifiers, or donor mappings.
- `data/external/`: immutable third-party, pretrained, or independently
  supplied data.
- `data/interim/`: rebuildable partial outputs and conversion staging.
- `data/processed/`: versioned reusable feature tables or arrays with manifests.
- `data/graphs/`: reusable graph objects with construction configuration and
  QC.
- `data/splits/`: protected assignment files; keep them untracked.
- `data/registry/`: tracked non-identifying metadata, aggregate counts,
  schemas, protected path references, and fingerprints.

The raw CosMx resolver must support
`data/raw/<filename>/<filename>`. Cell joins use `(slide, fov, cell_ID)` and
FOV joins use `(slide, fov)`; keys without slide are not globally unique.
Stream large transcript and polygon tables rather than loading them eagerly.
Use global physical coordinates for graph geometry and record the coordinate
frame.

## Feature and leakage policy

Exclude `Negative*` and `SystemControl*` probes from biological features while
retaining them for declared technical QC. Keep slash-combined probe names as
probe-level targets. Vendor cell type, clusters, neighborhoods, niches,
disease stage, core, donor, posterior probabilities, and target-derived
annotations are interpretation or stratification metadata unless a workflow
justifies them as permitted inputs.

Create splits before data-dependent preprocessing. Fit normalization,
imputation, feature selection, scaling, batch correction, graph choices, and
hyperparameters on training units only when making an inductive claim.
Construct held-out graphs independently and prohibit cross-split edges. A
strictly masked target's hidden expression, expression-derived label, or
unavailable library size must not re-enter through node features, graph
construction, or preprocessing.

Patient/sample grouping defines biological generalization. Random cell splits
are debugging or transductive tests. The current protected normal-core subset
contains one independent sample and is limited to within-core claims. HGD and
true Normal have one documented core/sample each in the full snapshot and
remain descriptive until a new verified manifest establishes replication.

## Registries and fingerprints

`data/registry/datasets.yaml` and `data/registry/splits.yaml` contain no row
assignments. Each dataset version records a protected source path, schema,
aggregate counts, preprocessing version, status, and deterministic fingerprint.
Each split records the method, grouping/uncertainty unit, seed, fold count,
constraints, protected assignment path, and fingerprint.

The fingerprint algorithm sorts normalized path/role, byte-size, and SHA-256
records, serializes strict canonical JSON, and hashes it with SHA-256. Split
fingerprints additionally bind the ordered protected assignments, grouping
unit, method, seed, and schema version. A changed source, assignment,
preprocessing version, or fingerprint requires a new version/ID. Do not update
an existing identifier to point at different content.

Raw fingerprinting is read-only. It must not rewrite metadata or create backup
copies of large datasets. Conclusion-bearing preprocessing also records input
hashes, exclusions, schema, conversion code, and output hashes in an immutable
manifest.

## Privacy and storage

Tracked files, logs, registries, predictions, reports, and curated result
records must not contain direct patient/donor identifiers or restricted
row-level metadata. Prediction
tables use opaque registry keys or an HMAC with an untracked secret; an
unsalted hash of a direct identifier is not sufficient protection. Do not
upload source data, artifacts, metadata, or fingerprints to an external
service.

Runs reference dataset and split versions; they never copy full datasets.
Parquet is preferred for large tables, NPZ for small dense arrays, and Zarr
only for suitable chunked arrays when supported. Every rendered curve retains
its numeric source data.

Check whether the workspace is backed by a persistent volume before relying on
local storage. Stop/start persistence is not protection against recycle or
destruction. Back up irreplaceable data and finalized artifacts only to an
approved protected destination; this repository does not initiate remote sync.
