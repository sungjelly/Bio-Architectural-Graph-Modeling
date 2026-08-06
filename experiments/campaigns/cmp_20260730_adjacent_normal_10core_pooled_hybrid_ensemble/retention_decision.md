# Retention decision for the pooled ten-core campaign

Date: 2026-07-30

The request to start over included clearing previous experiments to make room
for the new pooled ensemble. A read-only inventory established:

- 47.828 GiB is already free;
- canonical experiment artifacts occupy 28.088 GiB;
- the ten required prepared inputs occupy 0.350 GiB;
- state, reports, and scratch together occupy only 0.023 GiB;
- prior artifacts consume disk space but no host RAM or GPU VRAM;
- the projected paired seven-seed ensemble requires roughly 6–8 GiB, subject
  to the resource pilot.

The repository has no supported archive relocation or completed-run tombstone
operation. Manual deletion of successful bundles would invalidate the
authoritative registry, checkpoint catalog, checksums, and reproducibility.
Negative experiments are still evidence and are not equivalent to disposable
caches.

Decision: preserve canonical prior bundles, the registry, reports, frozen
contracts, and all prepared `ANC-01` through `ANC-10` inputs. Do not use
`git clean` or delete dirty-worktree files. Retain only one final checkpoint
per new ensemble member and no per-epoch checkpoints or large interpretation
arrays.

Revisit physical archival only if the pooled pilot projects less than
27.5 GiB free after production. Such archival requires an approved protected
destination, an SQLite-consistent snapshot paired with the bundles, checksum
and restore verification, and a versioned registry-aware relocation or
tombstone workflow. No previous canonical artifact was deleted by this
decision.

## Post-pilot review

Both attempt-2 resource pilots passed. The more conservative projected
post-production free-space estimate was 38.92326736450195 GiB for the GAT
pilot, compared with the frozen 27.5 GiB minimum. The matched-self estimate
was 39.16045379638672 GiB. The production campaign was therefore enqueued
without deleting prior canonical artifacts.

The failed attempt-1 pilot jobs did not consume a training allocation: they
stopped during validation because long-lived workers retained a stale
protocol validator. Their failure records remain in the registry, and the
successful attempt-2 runs are separate immutable bundles. This operational
retry does not change the storage decision or authorize deletion of negative
or failed-run provenance.

## Final review

All fourteen production runs and the final comparison completed. Project
doctor reported 39.754 GiB free after peak-VRAM registry reconciliation,
above both its 25 GiB minimum and the campaign's frozen 27.5 GiB
projected-free-space gate. The final comparison occupies approximately 16 MiB.

The scientific outcome was negative because the pooled-data and
representation gates failed. That result is conclusion-bearing evidence, not
disposable scratch output. No prior bundle, production bundle, failed pilot
record, registry row, or prepared `ANC-01` through `ANC-10` input was deleted.
