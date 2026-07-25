# Checkpoint Catalog Task Contract

## Objective

Create a non-destructive semantic catalog for every registered BAGM checkpoint.
Keep the existing immutable run IDs and artifact paths, add readable run aliases
and evidence-backed categories, and make checkpoints easy to query and resolve
for later interpretation.

## Scientific and operational question

Can historical and future checkpoints be selected by their scientific role and
configuration without treating archive dates or hand-built paths as scientific
metadata?

The primary hypothesis is that a normalized catalog keyed by immutable run and
artifact identifiers is sufficient. The main alternative is a category-based
physical reorganization of checkpoint files. That alternative would duplicate
or mutate immutable artifacts and is rejected unless catalog validation shows
that references cannot be resolved reliably.

## Estimand and permitted claim

This task measures catalog completeness, identity stability, and artifact
verification. It does not compare model quality or support a biological claim.
The unit is one registered checkpoint artifact associated with one run.

## Inputs and controls

- Authoritative registry: `state/tracking/bagm.sqlite3`
- Audited legacy index:
  `artifacts/legacy_runs/lr_spatial_benchmark_batch/index.jsonl`
- Immutable manifests and checkpoint files under `artifacts/legacy_runs/`
- Positive control: every registered historical checkpoint is cataloged once
- Negative control: a missing or checksum-mismatched fixture must not resolve
- Duplicate control: identical SHA-256 content is annotated, never assumed to
  represent the same logical run

## Acceptance and falsification criteria

The work passes when:

- all 158 historical checkpoint records are cataloged;
- their primary run IDs and checkpoint bytes remain unchanged;
- unknown historical fold and attempt values remain explicitly unknown;
- stage totals are 9 diagnostic, 96 exploratory screen, 3 validation
  confirmation, and 50 locked final;
- semantic aliases are deterministic and unique;
- exact-content duplicate groups are reported without deletion or relinking;
- filter, resolution, export, schema-upgrade, and future-run integration tests
  pass;
- the registry integrity check, artifact verification, and project doctor pass.

The approach is falsified if any original checkpoint changes, an alias is
ambiguous, a category relies on an unsupported inference, or a verified
checkpoint cannot be resolved from its catalog record.

## Failure and stop criteria

Do not load model weights, start training, rewrite manifests, replace primary
run IDs, delete duplicate content, or overwrite an existing export. Stop a
resolution when path, size, or checksum verification fails.

## Expected artifacts and verification

Expected changes are a versioned registry schema, semantic catalog module and
CLI, controlled schema configuration, generated local catalog/export, tests,
and documentation. Validation uses focused unit/integration tests, the complete
test suite, `verify-artifacts`, and `doctor`; no GPU training is authorized.
