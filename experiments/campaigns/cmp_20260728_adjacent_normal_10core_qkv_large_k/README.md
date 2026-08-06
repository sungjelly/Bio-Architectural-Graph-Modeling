# Ten-core adjacent-normal QKV-GAT large-k replication

## Scope

This campaign repeats the locked three-arm full-core QKV experiment on ten
independent pathology-confirmed **adjacent-normal** gastric tissue cores. It is
not a ten-core true-Normal study: the available snapshot contains only one
documented true-Normal core.

The protected selection reconciles the legacy donor/core workbook with the
current pathology review. Eligible rows must use one of the two explicit
legacy adjacent-tissue labels, be reviewed as `Normal` with normal tissue
confirmed, have no correction note, contain at least 5,001 cells, and represent
distinct donors. Five cores per slide are chosen by distance from the
slide-specific eligible median cell count. Protected donor/core identifiers
are never model inputs or public outputs.

The protected selection manifest is externally pinned before artifact
preparation as SHA-256
`a460de90ba9998e6817cdf3fbbcd2416eec048ec3fe6c825db8677b70b8b297f`.

## Locked experiment

Each core is fitted independently. The three arms are:

1. exact mutual-neighbor QKV graph Transformer at `k=1,000`;
2. the same QKV graph Transformer at `k=5,000`;
3. an exactly parameter-matched cell-autonomous control.

The architecture remains the previously verified 36,749,480-parameter model:
width 576, eight pre-LayerNorm QKV/FFN blocks, nine 64-dimensional heads,
2,304-wide FFN and decoder, edge bias plus bounded value gating, exact receiver
partitioning, and activation checkpointing. All arms use seed 0, the same
300-epoch fixed budget, optimizer, epoch masks, and three fixed evaluation
masks. There is no validation or test partition and no checkpoint selection.

The ten-core preflight observed 5,000th-neighbor maxima from 1,175.43 to
1,901.91 µm. The earlier one-core 1,200 µm guard would therefore truncate this
cohort. A common 2,000 µm post-kNN guard is locked for every arm and core,
leaving at least 98.09 µm margin. It changes no exact neighbor selection; it
does set the common distance-normalization scale for edge features.
Because each graph fits its 17 edge-feature standardization statistics on its
own retained edges, the `k=5,000` versus `k=1,000` contrast changes topology
and the fitted geometry representation together. It is not a pure causal
effect of neighbor count alone.

The primary outcome is whole-node masked Huber averaged over the three
technical masks within each core. Percentage accuracy is reported as masked
percent variance explained (`100 * R²`), which may be negative. Biological
replication and inference use the ten cores, never cells or mask repeats.

## Contrasts and decision rule

For each core:

```text
relative Huber gain (%) = 100 * (reference - k5000) / reference
```

The two paired contrasts are `k5000` versus matched self and `k5000` versus
`k1000`. A contrast passes only if all 30 production runs complete 300 finite
epochs, its unweighted mean core-level gain is at least 2%, at least 9 of 10
cores favor `k5000`, and multiplicity-adjusted paired evidence supports a
positive mean. The representation contrast additionally requires positive mean
`k5000` PVE and a positive mean PVE difference versus self.

The maximum defensible claim is average held-in masked-expression
representation capacity across these ten adjacent-normal cores. Because every
cell and preprocessing statistic is fitted, the result is not patient-held-out
prediction. `k=5,000` is broad regional context (roughly one third to two
thirds of each selected core), not a direct-contact or causal interaction
graph.

## Operational contract

Prepared core routing remains under ignored protected storage. Public configs
contain only opaque `ANC-##` aliases and verified checksums. Exact graph
identities are materialized before enqueue. One job occupies one RTX 3090; the
eight GPU workers use distinct state/lock roots and the shared canonical
registry. Jobs are balanced by exact directed-edge workload, and the workspace
must retain the queue's 25 GiB free-space floor.

Production enqueue is receipt-backed and resumable: a canonical-config digest
may have at most one queue job in this campaign. Jobs allow two attempts for
diagnosed recovery, but workers do not auto-retry or change scientific
parameters.

After materialization and before enqueue, run the separate offline finalizer:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/finalize_multicore_qkv_materialization.py \
  --selection-manifest \
  scratch/preprocessing/adjacent_normal_10_core_selection/selection.json
```

It re-verifies the externally pinned selection, all ten protected artifacts,
20 graph receipts, and 30 production configs without rebuilding any graph. It
then migrates only the checksum-bound legacy pre-finalization state, recomputes
the self-aware eight-GPU assignment, writes each launcher GPU and canonical
config digest, and atomically replaces the materialization checksum. The
utility has no registry, enqueue, worker, or training operation.

Focused verification:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_finalize_multicore_qkv_materialization.py
```
