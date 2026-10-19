# h0 Leiden maps of the reserved paper test cores

Phase: complete. Outcome: descriptive artifacts verified; biological interpretation remains inconclusive.
User requested h0 maps on 2026-10-18 after
viewing the final hL maps. This is an exploratory descriptive companion analysis.

## Task contract

Extract the trained node encoder output h0 before graph blocks from the selected
best checkpoint (epoch 194) of `r_20261017T044837Z_caf76350_s000_f00_a01_12df41be`.
Use exactly test cores 15 and 23, with 38,145 and 43,462 cells respectively.
Save joint Leiden labels, individual and combined cell-polygon PNG/PDF maps,
shared h0 color key, seed-stability statistics, and a portable HTML report.

Question: what descriptive groups are visible before graph message passing?
The working hypothesis is that trained cell-level expression/morphology features
already organize much of the tissue. Alternatives include technical/core effects
and unstable graph partitions. h0 is a trained representation, not raw expression
or an untrained baseline. Comparing h0 with hL describes representations; it
does not establish causal communication or a graph-specific predictive benefit.

Keep the checkpoint, normalization, all-observed input mask, FP32 deterministic
inference and test cell order identical to the published hL evaluation
`ev_so12_paper_test_hL_s0_20261017`. Use the actual full within-core graphs for
the consistency audit. Require exact agreement of the new forward pass's hL
with the published hL, h0 with a direct encoder call, and ordinary predictions
with the intermediate-output forward. The largest core is the real-data pilot.
Never load training/validation arrays or refit preprocessing. Existing successful
bundles and their bound source files remain unchanged.

Use identical clustering settings: row L2 normalization, exact cosine k=30,
bounded query chunks, unweighted undirected union, Leiden RBConfiguration
resolution 1.0, convergence, seed 0 for display and seed 1 for stability ARI.
No PCA or parameter tuning. A CPU exact-neighbor audit must pass before full
search. Report all clusters and cells; neither seed nor resolution is selected
for appearance. h0 labels/colors are shared across the two test cores, but their
numeric IDs do not correspond to independently clustered hL labels/colors.

Observational units are cells nested within two cores. Donor overlap is permitted
by the approved split; no new-donor, cancer-segmentation or biological identity
claim is justified. Manual cancer ROI labels are not used. Seed agreement and
core composition are descriptive controls, without a post-hoc pass threshold.

Acceptance requires verified finalized source/reference bundles, checkpoint and
frozen-preprocessing identity, exact canonical 81,607-cell coverage, finite
nonzero [81607,256] h0, exact forward/reference audits, exact-neighbor CPU
agreement, complete labels and aligned original polygons, checksummed publication,
registry completion and visual inspection before showing the figure. Failure of
any identity/numeric/coverage check stops with a retained failure record.

Resources: one free RTX 3090 under shared global and exclusive GPU-0 locks;
managed supervisor service, no preemption. Use CPU for Leiden/rendering. The
observed free disk is about 7.2 GiB; budget 0.5 GiB and enforce a 5-GiB reserve.
No cleanup or duplicate checkpoint/hL matrix copy is needed. Write active output
under scratch and publish by same-filesystem rename into a separate h0 analysis
directory. Record code/configuration/environment/reference hashes and evaluation
identity. Synthetic extraction/representation/schema/neighbor/render checks
precede real computation. Exact execution and result paths will be added below.

## Execution

Evaluation `ev_so12_paper_test_h0_s0_20261018` is registered separately from hL.
Supervisor service `bagm_so12_paper_test_h0_leiden` started at 03:15 UTC on
2026-10-18. Its wrapper/configuration are retained under `ops/supervisor/`.

```bash
PYTHONPATH=.:src:scripts/analysis /venv/main/bin/python \
  scripts/analysis/run_so12_paper_test_h0_leiden.py
```

The controller obtains the existing shared worker/exclusive GPU-0 locks, verifies
the exact finalized training and hL reference, and runs the new h0 extraction,
clustering and rendering entry points. The first full-core forward is core 23;
it must pass the h0/direct-encoder, published-hL and prediction invariance audits
before proceeding. Existing source bindings and successful outputs are preserved.

Control/status/logs: `scratch/active_runs/ctrl_so12_paper_test_h0_leiden/`.
Active computation: `scratch/active_runs/ev_so12_paper_test_h0_s0_20261018/`.
Final destination, after verified publication:
`reports/analyses/so12_paper_test_h0_leiden/ev_so12_paper_test_h0_s0_20261018/`.
Main files are `embeddings/h0.npy`, `clustering/clusters.csv.gz`,
`figures/test_cores_h0_leiden.{png,pdf}`, individual core maps, shared color key,
and `report/index.html`.

Combined focused checks: **75 passed** in 13.95 seconds. These cover managed
controller/resource/publication guards, complete source and reference identity,
exact encoder/reference agreement and deliberate one-ULP mismatch rejection,
synthetic extraction-to-clustering handoff with exact cells/coordinates,
exhaustive-neighbor/Leiden recovery and full polygon exports. Direct CLI imports
and supervisor shell syntax pass. All 23 training-bound source files match the
training startup receipt. Synthetic maps were visually inspected; real output
inspection remains pending. Full log: `focused_tests.log` in the control folder.

Doctor reports clean SQLite integrity and 7.112 GiB free at launch. Its sole
issue is the previously documented one-checkpoint historical catalog gap
(89 checkpoint artifacts, 88 indexed); the active worker-lock warning is
expected. No catalog or unrelated run data were changed. Extraction completed
in 79.89 seconds with 4,727,748,096 peak allocated GPU bytes. All 81,607 h0 rows
match direct encoder output, both full forwards reproduce published hL exactly,
and prediction invariance passes. The exact-neighbor pilot passed before full
graph construction; Leiden optimization then started normally.

## Completed results

Published successfully at **2026-10-18 03:23:57 UTC**. Registry status is
`completed`; the final report path is the destination listed above. Independent
verification checked the full manifest inventory, file hashes, h0/reference
identity, exact per-core audits and complete 81,607-cell coverage. The bundle
contains 60 files and occupies 127,915,607 bytes (about 122 MiB). No deletion was
needed.

Both Leiden seeds produced **14 clusters**; their adjusted Rand agreement is
**0.731728** (core 15: 0.738700; core 23: 0.735420). The displayed seed-0
partition has four clusters with more than 90% of their cells from one core.
The fraction of neighbor-graph edges between cores is 0.149802. These are
descriptive stability/composition summaries, not cell-type or cancer-region
validation and not evidence of graph-specific benefit.

The overview and shared color key were visually inspected after publication:
both core panels, original tissue outlines, cell colors and 100-µm scale bars
are readable. This later visual inspection is recorded here; the immutable
bundle correctly retains its original automated-publication
`manual_visual_review: not_performed` field. Cluster IDs/colors are shared
within the h0 figures and remain independent of hL cluster IDs/colors.

Primary delivery: `figures/test_cores_h0_leiden.png`,
`figures/shared_cluster_color_key.png`, corresponding PDFs and
`report/index.html`. All requested deliverables and computational acceptance
checks passed. Further biological interpretation would require independent
annotations or validation, outside this image request.
