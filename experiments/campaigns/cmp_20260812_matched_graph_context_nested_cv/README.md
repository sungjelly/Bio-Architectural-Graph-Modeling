# Matched graph-context nested-CV experiment

Status: `running`

Phase: `stage_a_nested_tuning`

Outcome: `pending`

This exploratory campaign asks whether a locally aligned, one-hop spatial
expression context improves whole-node expression prediction beyond a strong,
exactly parameter-matched no-graph model after both arms receive an independent
and equal hyperparameter-selection budget. Existing outcomes on these two
slides have already been inspected, so this is not an independent confirmation.

## Task contract

### Objective and deliverables

1. Audit and reuse the checksum-bound V0 all-cell/log1p/within-FOV arrays.
2. Implement an exact-parameter-matched graph/no-graph model and synthetic
   positive/null recovery gate.
3. Within each outer fold independently, tune each arm using only its paired
   validation geometry components, first with one seed and then with three
   seeds, without evaluating that outer-test component.
4. Freeze one selected configuration per outer-fold/arm in a checksum-bound
   selection receipt before any new outer-test evaluation.
5. Fit five paired seeds across all four outer folds for no-graph, true near,
   source-state-permuted near, and annular contexts.
6. Report component-equal effects, slide-stratified uncertainty, seed and
   component heterogeneity, feature-deletion faithfulness, target-gene and
   prespecified program summaries, failures, and claim limits.

### Scientific question and estimand

Primary question:

> Conditional on 22 independently measured morphology/imaging variables and a
> train-fitted quadratic broad-spatial-field control, does the observed mean
> RNA profile of other cells within 0--25 um improve prediction of a receiver's
> completely hidden 1,000-probe RNA profile?

The estimand is whole-node prediction. Receiver RNA, receiver library size,
RNA-derived labels, technical probes, cell/FOV/slide identifiers, and hidden
target-derived quantities are prohibited inputs. Source-cell RNA is permitted
only through a predefined graph-context mean. The primary metric is held-out
geometry-component-equal standardized MSE; MAE and per-gene metrics are
secondary.

The maximum claim is a graph-alignment-dependent predictive dependency within
the two already-observed slides. Geometry components are leakage-control units,
not patients or independent biological replicates. This campaign cannot
establish patient generalization, cell-cell communication, mechanism, or
causality.

### Hypotheses and alternatives

- H1: observed local source-cell RNA provides at least 2% graph-specific
  predictive gain beyond the independently tuned no-graph model.
- A1: morphology and broad spatial field are sufficient; no-graph matches the
  local graph.
- A2: generic source-state distribution or extra model capacity explains the
  result; the degree-matched source-state permutation matches the true graph.
- A3: broad regional co-localization explains the result; the 25--50 um
  annular context matches the 0--25 um context.
- A4: cell-state or library-size spatial autocorrelation explains the result;
  existing V4/V6 sensitivity results attenuate the gain.
- A5: a small number of large geometry components or one slide drives the
  result; equal-component and slide-specific effects disagree.
- A6: segmentation spillover dominates the shortest distances; a separately
  reported 10--25 um sensitivity attenuates or removes the signal.

The positive synthetic control must recover a planted context effect and the
null control must not create one. Failure blocks interpretation of real-data
model effects.

### Data, split, and leakage controls

The primary input is
`data/processed/same_gene_robustness_v1/variants/v0_within_fov_log1p_all/`,
bound by its manifest and integrity manifest. It contains 407,999 cells,
396,622 primary-eligible receivers, 1,000 biological probes, 22 permitted
morphology/imaging features, two slides, and 27 opaque geometry components.

Outer folds remain the frozen four-fold geometry partition. For outer fold
`f`, validation is `(f + 1) mod 4`, tuning-train is the other two folds, and
final-train is every non-test fold. All target, morphology, broad-field, and
context statistics are fit on the applicable training population only.
Graph edges remain within FOV, contain no self edge, and never cross a split
because complete geometry components are assigned to folds.

The test folds were observed in earlier campaigns. New tuning code must still
avoid reading their outcomes before the selection receipt is frozen, but the
final estimate remains explicitly exploratory.

### Matched model and arms

All learned arms use the same architecture and exact trainable-parameter count:

```text
prediction = Linear(27 broad-field+morphology features, 1000)
           + Linear(1000 context, 1000; no bias)
           + Linear(GELU(Linear(1000 context, h)), 1000; no bias)
```

The 27 base features are the 22 permitted morphology/imaging fields plus the
within-slide train-fitted basis `[x, y, x^2, xy, y^2]`. The no-graph context is
a frozen rank-27 random projection of those base features into 1,000 values.
It contains no graph, neighbor, identifier, receiver RNA, or target-derived
quantity, but gives the no-graph arm the identical nonlinear parameter budget.

Arms:

- `no_graph`: fixed base-feature projection, no topology or neighbor values;
- `observed_near`: observed 0--25 um within-FOV source-cell mean;
- `permuted_near`: receiver-collision-free, degree-preserving within-FOV
  source-state permutation using the frozen V0 mapping;
- `observed_annular`: observed 25--50 um within-FOV source-cell mean.

The true-near model is also evaluated after replacing its context by zeros,
permuted-near context, annular context, and a 10--25 um context without
refitting. The last is reconstructed by filtering only the frozen near-CSR
edge endpoints using their stored coordinates; no new edge is introduced, and
a receiver with no retained edge receives a zero context. These are
feature-deletion/substitution faithfulness diagnostics, not new fitted arms.

### Hyperparameter selection

The deterministic 16-candidate design spans context hidden width
`{32, 64, 128}`, learning rate `{3e-4, 1e-3, 3e-3}`, weight decay
`{0, 1e-4, 1e-3}`, and dropout `{0, 0.1, 0.2}` without taking the full Cartesian
product. The exact pre-outcome tuples are:

| ID | hidden | learning rate | weight decay | dropout |
|---|---:|---:|---:|---:|
| c00 | 32 | 0.0003 | 0 | 0 |
| c01 | 32 | 0.0003 | 0.001 | 0.2 |
| c02 | 32 | 0.001 | 0.0001 | 0.1 |
| c03 | 32 | 0.003 | 0 | 0.2 |
| c04 | 32 | 0.003 | 0.001 | 0 |
| c05 | 64 | 0.0003 | 0 | 0.1 |
| c06 | 64 | 0.0003 | 0.001 | 0 |
| c07 | 64 | 0.001 | 0 | 0.2 |
| c08 | 64 | 0.001 | 0.0001 | 0.1 |
| c09 | 64 | 0.003 | 0.0001 | 0 |
| c10 | 64 | 0.003 | 0.001 | 0.2 |
| c11 | 128 | 0.0003 | 0.0001 | 0.2 |
| c12 | 128 | 0.0003 | 0.001 | 0 |
| c13 | 128 | 0.001 | 0 | 0 |
| c14 | 128 | 0.001 | 0.001 | 0.1 |
| c15 | 128 | 0.003 | 0.0001 | 0.1 |

Batch size is fixed at 4096. Every candidate follows one continuous AdamW
trajectory and is scored at epochs `[12, 24, 48, 96, 192]`.

Stage A evaluates all candidates with seed 20260812 for all four outer-fold
jobs, but selection never pools across outer folds. Within each outer fold,
each arm and each hidden-width stratum advances the best candidate using only
that outer fold's paired validation components. This gives three candidates
per outer-fold/arm while preserving all widths. Stage B evaluates those
outer-specific candidates with seeds 20261812 and 20262812.

The confirmation architecture must remain exactly parameter matched within
each outer fold. Separately for each outer fold, Stage B first identifies each
arm's best optimizer/dropout candidate at each width. For width `h`, the
outer-fold-specific arm relative regret is
`best_mse(arm,h) / best_mse(arm,any_h) - 1`. The shared width minimizes the
maximum regret across all four arms, then mean regret, then width, using only
that outer fold's validation components. Within that shared width, one
learning-rate/weight-decay/dropout/epoch configuration is selected per arm
from its three-seed component-equal validation mean. Within 0.25% of the
minimum, the tie-break order is lower seed standard deviation, larger weight
decay, lower dropout, lower learning rate, and candidate ID. No validation
outcome is pooled across outer folds for selection: a component serving as
outer test in one fold may be validation elsewhere, but can never influence
the configuration used to predict itself. Four outer-specific selections are
checksum-bound and frozen before confirmation.

Confirmation uses seeds 20260812, 20261812, 20262812, 20263812, and 20264812,
all four outer folds, fresh final-train fits, and the independently selected
outer-fold/arm optimizer configurations with one shared hidden width per outer
fold. No best-seed or best-fold selection is permitted.

### Metrics, inference, and decisions

For each seed, fold, component, arm, and gene, save MSE and MAE. The primary
contrast is `no_graph - observed_near`, positive in the graph-favorable
direction. Seeds are averaged within component before components receive equal
weight. A deterministic 10,000-resample bootstrap stratified by slide supplies
a descriptive 95% interval. Cells and seeds are technical observations, and
27 components nested in two slides are not treated as patient replication.

`GRAPH CONTEXT SUPPORTED` requires all of:

1. at least 2% relative component-equal MSE improvement over independently
   tuned no-graph;
2. the descriptive 95% interval for improvement has lower bound above zero;
3. both slides have positive mean improvement;
4. at least 20/27 components and 4/5 seed aggregates favor true near;
5. component-equal MAE does not worsen; and
6. true near improves at least 1% over independently tuned permuted near, with
   its descriptive interval lower bound above zero.

If the upper interval for the no-graph-relative improvement is below 2%, the
valid verdict is `GRAPH CONTEXT NOT SUPPORTED`. Other valid mixed outcomes are
`INCONCLUSIVE`. A graph-context result does not establish GAT/attention utility;
the model uses a fixed uniform mean. Near-versus-annular, deletion faithfulness,
V0--V6 prior sensitivities, and gene/program results remain separate evidence
dimensions and cannot rescue the primary gate.

### Additional scientific analyses

- locality: observed near versus annular and a 10--25 um short-edge-removal
  sensitivity reconstructed only from frozen near-graph endpoints;
- topology: observed near versus matched permutation;
- faithfulness: zero, permuted, and annular substitution in the locked true
  model;
- heterogeneity: effect versus slide, component size, degree, and vendor-QC
  fraction without treating these as independent causal covariates;
- programs: prespecified TLS/immune, myeloid, stromal/ECM, epithelial, and
  endothelial target families, followed by exploratory BH-FDR gene summaries;
- interpretation: locked mean Jacobians are model sensitivities only and are
  reported separately from predictive gain and null calibration.

The fixed target-program labels are panel-constrained descriptive families,
not independent validation or directional cell-cell mechanisms:

- TLS/immune: `CXCL13, CCL19, CCL21, LTB, MS4A1, CD79A, CD74, HLA-DRA,
  CD3D, CD3E, CXCR5, CCR7`;
- myeloid: `LYZ, FCER1G, TYROBP, C1QA, C1QB, C1QC, APOE, SPP1, IL1B,
  CXCL8`;
- stromal/ECM: `COL1A1, COL1A2, COL3A1, COL6A1, COL6A2, DCN, LUM,
  COL4A1, COL4A2, FN1, FAP, PDGFRA`;
- epithelial: `EPCAM, KRT8, KRT18, KRT19, KRT7, KRT17, CEACAM6, KRT20,
  TACSTD2`;
- endothelial: `PECAM1, VWF, KDR, ENG, RAMP2, ESAM, RGCC`.

### Compute, artifacts, and stop criteria

Four safely available RTX 3090 GPUs run independent fold/config/seed jobs. Each
job writes active output only below `scratch/active_runs/`, then publishes a
checksum-bound immutable bundle below `artifacts/runs/YYYY/MM/` and registers
it in `state/tracking/bagm.sqlite3`. Tuning bundles omit checkpoints and test
outputs. Confirmation bundles retain the selected checkpoint, per-component
and per-gene metrics, exact configuration, input/selection hashes, environment,
GPU/runtime/VRAM, logs, and completion marker.

Stop on a synthetic-gate failure, changed input/split/selection hash,
train/validation/test component overlap, nonfinite value, missing fold/config/
seed/arm coverage, unsafe GPU collision, peak VRAM above 20.5 GiB, less than
25 GiB free disk, registry/artifact inconsistency, or a failed checkpoint
replay. An unfavorable scientific result is not a blocker.

### Exact execution sequence

Run from the repository root with `PYTHONPATH=src`. The coordinator writes its
checksum-bound authorities below
`state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/`:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/train/launch_matched_graph_context.py materialize
PYTHONPATH=src /venv/main/bin/python scripts/train/launch_matched_graph_context.py launch-stage-a --plan state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/stage_a_plan.json
PYTHONPATH=src /venv/main/bin/python scripts/train/launch_matched_graph_context.py select-stage-a --plan state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/stage_a_plan.json
PYTHONPATH=src /venv/main/bin/python scripts/train/launch_matched_graph_context.py materialize-stage-b --stage-a-selection state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/stage_a_selection.json
PYTHONPATH=src /venv/main/bin/python scripts/train/launch_matched_graph_context.py launch-stage-b --plan state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/stage_b_plan.json
PYTHONPATH=src /venv/main/bin/python scripts/train/launch_matched_graph_context.py lock-selection --stage-a-plan state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/stage_a_plan.json --stage-a-selection state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/stage_a_selection.json --stage-b-plan state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/stage_b_plan.json
PYTHONPATH=src /venv/main/bin/python scripts/train/launch_matched_graph_context.py materialize-confirmation --selection-receipt state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/selection_receipt.json
PYTHONPATH=src /venv/main/bin/python scripts/train/launch_matched_graph_context.py launch-confirmation --plan state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/confirmation_plan.json
PYTHONPATH=src /venv/main/bin/python scripts/analysis/analyze_matched_graph_context.py --selection-receipt state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/selection_receipt.json --confirmation state/matched_graph_context/cmp_20260812_matched_graph_context_nested_cv/confirmation_plan.json --output-root reports/analyses/matched_graph_context_nested_cv
```

The preflight synthetic receipt is `synthetic_gate.json`; the irreversible
outcome firewall is `selection_receipt.json`. Final verification includes the
focused tests, all immutable run-bundle checksums, the analysis `_SUCCESS`
receipt, registry consistency, and `PYTHONPATH=src /venv/main/bin/python -m
spatial_benchmark doctor`.
