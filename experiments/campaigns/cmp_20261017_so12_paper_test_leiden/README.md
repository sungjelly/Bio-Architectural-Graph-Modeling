# Leiden clustering of the reserved paper test cores

Phase: scheduled. Outcome: awaiting source training/final test completion.
The managed waiting service was started on 2026-10-17 at 08:37 UTC to run
automatically after the approved training experiment finishes.

## Task contract

Use the validation-selected best checkpoint from training run
`r_20261017T044837Z_caf76350_s000_f00_a01_12df41be` to extract final-layer hL
representations for test cores 15 and 23 only (38,145 and 43,462 cells;
81,607 total), then cluster those two cores jointly with Leiden. Deliver a
registered, checksummed post-hoc analysis with cell-aligned embeddings, cluster
labels, cluster sizes, spatial cell-polygon maps, and a compact report. Set up
a supervisor-managed waiting job now; computation must start only after the
source run has completed final test evaluation and immutable archive publication.

Question: which descriptive expression/context groups occur in the two reserved
test cores under the selected model? The working hypothesis is that final hidden
states yield coherent and seed-stable groups. Alternatives include broad tissue
structure, technical effects, cell-autonomous expression and unstable clustering.
Report two fixed seeds' adjusted Rand agreement and core composition, including
weak stability or core segregation. No parameter is selected on attractive maps
or agreement with manually segmented cancer regions. This is exploratory
transductive clustering of test embeddings; it is not an inductive cluster-label
prediction benchmark, cancer validation, or a causal/communication claim.

The model and all normalization/morphology statistics remain frozen. Use all
observed genes for inference (zero gene mask), deterministic FP32 arithmetic,
the training-only Pearson input adapter and original complete within-core graphs.
Extract the fourth graph block's 256-dimensional output before decoding, not
predicted expression or attention. Do not load or cluster training/validation
cells, use historical model weights, refit normalization, or change the active
training code/configuration. Cell keys must be slide-qualified; direct patient
identifiers and clinical labels are excluded.

Prespecified clustering: L2-normalize hL for cosine distance only; exact sparse
30-nearest-neighbor graph with bounded GPU query blocks, self edges removed,
stable index tie-breaking, undirected union, unweighted edges. Run Leiden
RBConfiguration at resolution 1.0 to convergence. Seed 0 is the displayed
partition; seed 1 is a stability diagnostic, never a replacement chosen for
appearance. The embedding-space clustering graph may link the two test cores;
the model's spatial graphs cannot cross cores. Shared labels/palette across
cores permit descriptive figure comparison. No all-cell square distance matrix
is stored. No additional normalization or PCA is fitted on test data.

Observational units are cells; core-level reports describe two sections and
cannot establish donor-independent replication. Source training explicitly
allows donor overlap and historical sample exposure. The primary descriptive
outputs are seed-0 labels, cluster counts/composition and seed-0/seed-1 ARI.
No minimum ARI is imposed after seeing the data. Extraction checks against the
canonical model prediction and direct last-layer hook are positive controls;
CPU exhaustive neighbor checks, complete coverage and source binding guard
against implementation errors. Biological interpretation requires later labels
or orthogonal evidence; manual cancer ROI inputs have not yet been located.

Acceptance: finalized registry source status and verified bundle, selected-best
checkpoint/test receipt identity, unchanged training preprocessing fingerprint,
exact ordered test-cell coverage, finite nonzero [81607,256] embeddings, no
train/validation cells, exact-neighbor CPU-reference agreement, complete labels
for both seeds, maps with all original cell polygons and shared color keys,
checksummed report and registered artifacts. Numeric failures, source mismatch,
nonfinite output, graph/label/alignment errors, failed training, or disk below
the 5-GiB reserve stop the analysis with an explicit retained failure record.

Compute/storage: wait for the training job and its GPU locks to release; use
one allocated GPU for extraction and exact neighbor search, CPU Leiden/rendering.
Do not preempt existing jobs. Budget 0.5 GiB of additional disk for this test-only
hL workflow, well below the previously budgeted 1.4-GiB full-cohort h0/hL analysis.
Write active outputs under `scratch/active_runs/<evaluation_id>` and publish by
same-filesystem rename into a versioned analysis report directory. Keep model,
graph and raw input references rather than copying them. No deletion is needed.

Required verification: synthetic source/partition/finalization guards; meaningful
extraction, exact-neighbor and label-coverage tests; rendering coverage; existing
clustering helper regressions; clean imports without altering training source
bindings; managed waiting-state verification. A real GPU pilot on the selected
checkpoint is part of the deferred job and must pass before full clustering.

Immediate operational completion means the validated managed job is waiting on
the exact source run; scientific completion requires the final verified
clustering artifacts. No additional user approval is needed to launch.

## Execution and verification

Evaluation: `ev_so12_paper_test_hL_s0_20261017`, registered with checkpoint `best`
and status `pending`. Supervisor service: `bagm_so12_paper_test_leiden`.
Controller: `scripts/analysis/run_so12_paper_test_leiden.py`. The installed
wrapper and supervisor configuration match
`ops/supervisor/bagm-so12-paper-test-leiden.{sh,conf}`.

The controller polls the exact source registry row every 30 seconds. It waits
for completed publication, then obtains the shared worker lock and exclusive
GPU-0 lock, verifies the selected checkpoint and final test receipts, and runs
extraction, clustering and rendering. It does not restart a failed or partial
computation automatically. A clean waiting-service restart requires identical
source hashes, configuration and owned output path. Immutable completed output
can reconcile its registry record without recomputation.

Status: `scratch/active_runs/ctrl_so12_paper_test_leiden/status.json`.
Stage logs and active results:
`scratch/active_runs/ev_so12_paper_test_hL_s0_20261017/`.
Published results, after completion:
`reports/analyses/so12_paper_test_leiden/ev_so12_paper_test_hL_s0_20261017/`.
Primary deliverables are `clustering/clusters.csv.gz`, `embeddings/hL.npy`,
`figures/core_15_hL_leiden_segmentation.{png,pdf}`,
`figures/core_23_hL_leiden_segmentation.{png,pdf}`,
`figures/test_cores_hL_leiden.{png,pdf}`, a shared color key and
`report/index.html`.

Focused verification: 70 tests passed in 12.51 seconds across new controller,
extraction and clustering tests plus existing extraction/clustering/map
regressions. This includes real synthetic extraction-to-clustering handoff,
CPU exact-neighbor/Leiden recovery, synthetic raw-polygon export, complete
publication inventories, checkpoint/source/core guards, lock contention and
actual child-process termination. Direct CLI imports and wrapper shell syntax
pass. Synthetic map layout was inspected; real-data manual visual review has
not occurred and will not be falsely recorded by the automated job.

At launch, the training run was still running; all 23 training-bound source
files matched its startup receipt. The waiting evaluation had only its frozen
configuration, no computation-start marker and no extracted test cells.
Approximately 7.4 GiB remained available; no cleanup was performed. Real test
extraction and its GPU pilot remain deferred until the source completion gate.

```bash
supervisorctl status bagm_so12_paper_test_leiden
cat scratch/active_runs/ctrl_so12_paper_test_leiden/status.json
```
