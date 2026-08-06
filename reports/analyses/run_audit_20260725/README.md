# Registered Run Audit, 2026-07-25

## Status

- Phase: complete
- Outcome: prespecified graph-utility hypothesis not supported
- Scope: retrospective audit of all runs registered in the local BAGM registry
- Classification: exploratory retrospective analysis; no criteria were locked
  before the historical outcomes were observed

## Task contract

### Objective and deliverables

Produce a portable, single-file HTML report that:

1. inventories every registered run and its lifecycle category;
2. documents each model family, model size, data, graph, masking, training, and
   evaluation configuration to the extent preserved by the artifacts;
3. compares performance using the metric and aggregation appropriate to each
   experimental stage, with failures and duplicate checkpoints visible;
4. distinguishes observed performance from prespecified gate outcomes and
   defensible scientific conclusions;
5. recommends the next discriminating experiment; and
6. includes a complete machine-readable run table and explicit provenance.

Expected outputs are `run_report.html`, `runs.csv`, `summary.json`, and the
reproducible local generator used to create them.

### Scientific question and hypotheses

The retrospective question is whether the completed normal-core benchmark
provides evidence that local spatial graph context improves masked-expression
prediction beyond cell-autonomous and simpler spatial controls.

The campaign hypothesis predicts that locked true-graph G1 improves whole-node
masked Huber loss over B0 by at least 2%, has a spatial-block interval above
zero, beats degree/distance-matched rewiring, and improves block masking in the
same direction.

Credible alternatives are:

- morphology and intracellular expression dominate, so B0 is adequate;
- uniform smoothing is adequate, so B1 matches learned graph attention;
- broad spatial field or FOV structure explains apparent graph gain;
- segmentation spillover or local copying explains a small gain;
- edge geometry adds no useful signal beyond topology, so G2 does not
  consistently beat G1.

The report will compare predictions from these alternatives against preserved
locked and exploratory evidence; it will not create a post-hoc success rule.

### Estimand, units, and permitted claim

The primary preserved estimand is the held-out spatial-block contrast in masked
Huber loss between the paired B0 and true-graph G1 seed ensembles. Spatial
blocks are uncertainty units. Cells, masks, and model seeds are technical
variation, not independent biological replicates.

The maximum permitted claim is a within-one-core model-implied spatial
predictive dependency. The audit cannot establish patient generalization,
cell-cell communication, mechanism, or causality.

### Metrics, controls, and leakage risks

- Primary metric: whole-node masked Huber loss, lower is better.
- Prespecified primary comparison: locked G1 versus B0, including the original
  2% relative-gain gate.
- Controls: global mean, B0, B1, parameter-matched self model, broad field,
  nearest-neighbor copy, rewired G1, and preserved G2 edge controls where
  available.
- Secondary estimands: partial-gene and contiguous-block masking, kept
  separate.
- Leakage audit: train-only fitting; spatially disjoint splits; independently
  built split graphs; excluded identifiers, coordinates as node covariates,
  RNA-derived QC, and vendor-derived annotations.

The historical dataset and split are already fixed and outcomes have already
been inspected. This report therefore summarizes evidence; it is not a new
confirmatory test.

### Acceptance, falsification, and stop criteria

The audit is complete only if:

- every registry run is represented exactly once in `runs.csv`;
- registry, checkpoint catalog, manifests, and preserved analysis summaries
  reconcile or every discrepancy is documented;
- model-size and metric claims identify their source and missing fields remain
  explicitly unknown;
- model rankings are not based on one favorable seed where aggregate evidence
  exists;
- the original campaign gate is reported without modification;
- failed, diagnostic, exploratory, confirmation, and locked-final runs remain
  visibly distinct;
- the HTML is a portable single file and passes structural/link checks; and
- `verify-artifacts` and the project doctor pass, or failures are reported.

Stop without a scientific conclusion if canonical artifacts or the registry
cannot be reconciled. No training, sealed-test reopening, artifact mutation,
or raw/clinical data modification is authorized by this audit.

### Inputs, compute, and verification

Inputs are the local SQLite registry, immutable legacy run bundles and
manifests, checkpoint catalog, standards lock, final analysis, prior final
report, campaign README, and tracked configuration/code. Only CPU analysis is
required.

Planned verification:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark verify-artifacts
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
/venv/main/bin/python reports/analyses/run_audit_20260725/generate_report.py --check
```

## Results

The audit reconciled all 158 registered runs: 9 diagnostic, 96 exploratory
screen, 3 validation-confirmation, and 50 locked-final runs. All runs are
completed and all 158 best checkpoints are verified. The legacy fold and
attempt placeholders remain reported as unknown.

The locked true-graph G1 reduced whole-node masked Huber loss from 0.224994 to
0.224367 relative to B0. The paired gain was 0.000627 (0.279%; spatial-block
95% interval 0.000433 to 0.000823). This reproducible within-core difference
failed the prespecified 2% minimum. G2 and its edge controls showed no robust
added utility, so the configuration-only edge-feature campaign should not be
run unchanged.

The maximum defensible conclusion is weak, sub-threshold evidence for a
topology-specific predictive signal within one core. There is no
independent-patient replication and no evidence establishing communication,
mechanism, or causality.

## Deliverables

- `run_report.html`: portable single-file report with embedded figures and the
  complete searchable run table;
- `runs.csv`: one machine-readable row for each registered run;
- `summary.json`: aggregate results and source checksums; and
- `generate_report.py`: reproducible local generator and strict audit checks.

## Verification outcome

The generator check passed with exactly 158 unique runs, 10 locked conditions
with five paired seeds each, three embedded figures, and no external HTML
dependencies. `verify-artifacts` returned `valid: true` with no bundle or
registry issues. The project doctor returned `ok: true` with no issues or
warnings.
