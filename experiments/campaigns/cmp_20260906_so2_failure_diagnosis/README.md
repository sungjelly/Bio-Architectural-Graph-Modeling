# SO2 reconstruction diagnostic and external model comparison

Phase: complete. Outcome: inconclusive about the neural failure cause; supported
for a loss/metric mismatch in the constant-predictor class. This is an exploratory post-hoc analysis;
the four model outcomes were examined before this contract was fixed.

Objective: produce an auditable comparison of data size, architecture, training
objective, masking, and performance against primary published sources; distinguish
established limitations from possible causes. The numerical deliverable is a
fixed-mask comparison with constant and observed-neighbor predictors, followed by
a ranked diagnostic report. No new neural training or checkpoint selection occurs.

Question: does the current standardized-log Huber objective favor conservative
predictions even without a graph, and does the learned model outperform simple
spatial averaging? Primary hypothesis: part of the apparent failure reflects an
objective/metric mismatch. Alternatives: useful context is absent or unused;
aggregation dilutes local signals; decoder/optimization limits reconstruction.
Huber-optimal constants below MSE-optimal constants support loss-induced shrinkage.
Better neighborhood mean predictions would show an available spatial baseline
the trained endpoints fail to exploit on that metric. Neither observation alone
identifies the causal effect of changing loss, decoder, or graph architecture.

Estimand: reconstruction of the original hidden entries in the fitted SO2 cores
15–28, retaining observed same-cell genes and neighboring measurements. All 14
cores and all four seed-0 endpoints are included. Cores are descriptive strata;
there is no patient split or biological replication estimate. The continuing
original model is the same training lineage. No generalization, graph attribution,
biological mechanism, or causal claim is permitted.

Primary metrics (lower is better): equal-core all-entry standardized Huber and
MSE, positive-entry standardized and log1p MSE. Keep metrics separate; no composite
score or success threshold selected after observing results. Report zero strata
and source-model differences by core. Constants use the equal-core fitted target
distribution, as does the earlier gene mean; label them target-derived descriptive
references. Huber constants solve sum p(y)*clip(a-y,-1,1)=0 per gene. Include the
MSE-optimal log mean, zero-count reference, and Huber constant.

Spatial baselines use only unmasked neighboring entries, including observed zeros:
uniform mean log1p on the original full graph; uniform mean log1p on up to 16
nearest neighbors within 75 micrometers; uniform mean raw count on that local
graph, converted to log1p for common-scale evaluation. These are separate fixed
baselines, without tuning. For a receiver/gene with no observed neighbors, use
the observed-core per-gene mean, then zero if no observed value exists. Exclude
self edges. No hidden target enters a numerator, denominator, fallback, or local
graph construction. Coordinates alone construct the local graph. Report fallback
frequency. No target-derived cell labels or expression-derived library size.

Inputs: original source diagnostic masks, prepared cohort and graph manifests,
and four immutable nonzero-evaluation reports. Verify checksum/shape/alignment,
mask identity, target transformation, and existing gene-mean/zero metrics before
acceptance. Stream cores and genes; do not export unrestricted predictions.
Synthetic controls test a sparse zero-dominated Huber optimum, squared-loss mean,
directional averaging, inclusion of observed zeros, masked-value leakage and
fallback behavior. Any failed invariant stops execution. No blind retries.

Acceptance: tested math; small core 21 and largest core 23 pilot pass within host
memory and runtime budget; all 14 fixed masks complete; baseline replay within
2e-6; finite metrics and disjoint support decomposition; source/provenance
checksums; registered evaluation and verified immutable report. Fail if the
Huber root residual exceeds 1e-9 or its fitted risk exceeds either constant
reference beyond numerical tolerance. The loss hypothesis is falsified for these
constants if the fitted Huber optima do not shift downward; neural causal claims
remain unresolved regardless of this result.

Resources: CPU only, two numerical threads, cores sequential. Four RTX 3090s are
occupied by an existing independent SO1 training job (observed 2026-09-06); do not
displace it. Budget under 12 GiB host RSS and 5 minutes per core in the pilot;
stop and diagnose resource violations. This is sparse baseline computation, not
a GPU-capable full neural training experiment. No additional patient data or
external upload. Active files live beneath the geometry source run's
`scratch/active_runs/<run_id>/posthoc_reports/so2_failure_diagnosis/v1/`; publish to
`reports/analyses/so2_failure_diagnosis/baselines/v1/`. The synthesis is a
separate checksummed bundle at `reports/analyses/so2_failure_diagnosis/comparison/v1/`.
Preserve all older artifacts.

Verification and reproduction, from repository root:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_so2_diagnostic_baselines.py
PYTHONPATH=src /venv/main/bin/python scripts/analysis/diagnose_so2_baselines.py --phase prepare
PYTHONPATH=src /venv/main/bin/python scripts/analysis/diagnose_so2_baselines.py --phase pilot
PYTHONPATH=src /venv/main/bin/python scripts/analysis/diagnose_so2_baselines.py --phase full
PYTHONPATH=src /venv/main/bin/python scripts/analysis/diagnose_so2_baselines.py --phase verify
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_failure_diagnosis/comparison/v1/independent_audit.py
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_failure_diagnosis.py
PYTHONPATH=src /venv/main/bin/python scripts/analysis/finalize_so2_failure_diagnosis.py
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate --verify-sources --verify-payloads
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog --check
```

The comparison report will distinguish peer-reviewed work from preprints,
pretraining from per-dataset fitting, independent units from cells, masked genes
from masked entries, and rank aggregates from actual correlations. A controlled
decoder/loss/graph ablation remains necessary to identify a neural failure cause.

## Completed findings and verification

The [full report](../../../reports/analyses/so2_failure_diagnosis/comparison/v1/report.md)
compares all four SO2 endpoints, six fixed references and eight external methods.
No external model was executed. All 14 cores completed on CPU, taking 570.82 seconds
of baseline evaluation plus 6.37 seconds of histogram fitting; peak host RSS was
1.721 GiB. Existing GPU training was left undisturbed. Both pilot cores passed.

The fitted Huber constant is below the MSE-optimal gene mean for 999/1000 genes,
with average standardized shift -0.194874. Maximum stationarity error is below
1e-13, and every fitted Huber risk is no worse than either reference constant.
Geometry improves fixed-mask Huber from 0.253725 for this constant reference to
0.239953, a 5.43% improvement. This demonstrates learned predictive signal and
an objective-dependent preference; it does not identify the neural cause.

| Predictor | All-entry standardized MSE | Positive standardized MSE | Positive log1p MSE |
|---|---:|---:|---:|
| Geometry e200 | 0.951623 | 8.915118 | 0.690519 |
| Original full-graph log mean | 0.982751 | 8.589392 | 0.700991 |
| Local k16/r75 log mean | 1.085565 | 8.531141 | 0.686985 |
| Local k16/r75 raw-count mean | 1.242584 | 7.898872 | 0.619399 |

Geometry beats full-graph averaging on all-entry MSE in all 14 cores (3.17%
lower), but has higher positive standardized MSE in 12/14 cores (3.79% higher).
Local raw-count averaging lowers positive standardized MSE by 11.40% relative to
geometry, with lower error in all 14 cores, while greatly worsening zero and
all-entry errors. These tradeoffs do not establish oversmoothing or an overall
advantage of simple averaging. Fixed-mask scored entries come from cells masking
66.67% of genes on average; 19.14% come from cells masking at least 900 genes.

Acceptance checks passed: 29 focused helper/metric tests; all 14 source masks,
counts, transformations and graph identities; original mean/zero baseline replay
within 3.55e-15; finite values and support partitions; all constant-risk/root
checks; source manifests; registered completed evaluation and manifest artifact.
An independent audit verified four prior bundles/152 pinned payloads and
reconstructed 4,032 metric values. The source recurrent finalization failure and
pre-existing checkpoint catalog gap remain visible. `doctor` still reports 41
checkpoint artifacts versus 40 catalog entries; database integrity and config
checks pass. This task did not create or modify a checkpoint.

Decision: prioritize a matched cell-only/current-graph × Huber/MSE experiment
with the same target scale and decoder, followed by separate NB/Poisson count-head
conditions. Decoder probes and masked-layer diagnostics can discriminate unused
latent information from graph mixing. The proposed neural experiments are future
work, not requirements left unfinished in this completed diagnostic. One seed,
all-fit data, different endpoint durations and missing patient-held-out controls
prevent a generalization or neural-causal conclusion. Literature supplies no
directly comparable SOTA score and no evidence that more cells or NB alone fixes BAGM.

Finalized v1 bundles refuse overwrite. Existing results should be checked using
the verification commands; changing source code, inputs or protocol requires a
new registered version. The report builder writes a mutable synthesis stage;
the finalizer requires the independently generated audit receipt and recorded
visual inspection before publication. The audit implementation is preserved in
the synthesis bundle. All executed code and the pre-result task contract remain
in the immutable bundles. The curated result is
`results/descriptive_analysis/relative_qkv/res_so2_reconstruction_failure_diagnosis/`.

Publication metadata initially failed the result schema's strict UTC format check
(`+00:00` instead of required `Z`). The report and numerical bundles were already
verified and were not altered. The narrow recovery command
`PYTHONPATH=src /venv/main/bin/python scripts/analysis/finalize_so2_failure_diagnosis.py --repair-utc-metadata`
normalizes only those previously unvalidated metadata timestamps, verifies both
immutable bundles, then completes schema/catalog validation and campaign status.
The finalizer now emits the required format for future first-time publication;
its original executed version remains in the immutable synthesis provenance.
