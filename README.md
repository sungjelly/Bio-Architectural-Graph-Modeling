# Bio-Architectural Graph Modeling

## Abstract

This project develops and rigorously evaluates graph models of spatially resolved gastric cancer tissue. It asks whether cellular neighborhoods provide predictive information about expression beyond cell identity, morphology, broad tissue context, and simpler non-graph baselines.

The main objective is not merely to build another graph-attention or masked-expression model. It is to identify model-derived biological hypotheses that are stable across training runs, faithful to the model's predictions, reproducible across patients, calibrated against realistic spatial nulls, and supported by independent evidence. Causal claims require perturbational validation.

## Objectives

1. Demonstrate patient-held-out spatial predictive gain beyond cell-autonomous, morphology, regional-context, neighborhood-composition, and non-spatial baselines.
2. Build leakage-resistant, multiscale tissue graphs that distinguish direct contact, local paracrine context, and broader regional architecture.
3. Separate cell-autonomous, broad spatial-context, and local interaction signals in the model.
4. Extract predefined, interpretable readouts such as signed message contributions, bounded finite-difference effects, and distance-specific gene-program interactions.
5. Test stability across seeds, masks, graph definitions, segmentation variants, and patient bootstraps.
6. Test faithfulness through ablation, insertion/deletion, counterfactual, and model-randomization experiments.
7. Calibrate discoveries with realistic null models and planted semi-synthetic interactions.
8. Replicate conclusions across independent patients and cohorts, then seek supporting evidence from orthogonal modalities where possible.

The intended conclusion is an evidence record of the form:

```text
source cell type and program
    -> receiver cell type and target program
    + spatial scale, direction, effect size, uncertainty,
      faithfulness, null calibration, and patient prevalence
```

Until a controlled perturbation supports the proposed direction and mechanism, these conclusions are predictive dependencies or candidate mechanisms, not causal effects.

## Scientific Position

The project treats the following as separate hypotheses:

1. A graph model can predict masked expression.
2. Its representations contain biologically relevant information.
3. A predefined model-derived statistic faithfully identifies information used by the model.
4. The inferred dependency reflects a reproducible biological mechanism.
5. The dependency is causal.

Success at one level does not establish the next. Attention is not automatically an explanation, a gradient is a local model sensitivity rather than a correlation, and observational prediction does not establish causality.

## Project Flow

The research program is organized into gated phases:

0. **Estimands and ground truth:** distinguish partial-gene, whole-node, spatial-block, cross-cell, direct, multihop, and niche questions; test recovery on semi-synthetic ground truth.
1. **Data and graph quality control:** audit segmentation, tissue geometry, graph construction, technical effects, and target leakage.
2. **Predictive benchmark:** compare the graph model with strong simple and spatial baselines using leave-patient-out and spatial-block evaluation.
3. **Locked readout extraction:** freeze successful models and extract predefined signed contributions, bounded counterfactual effects, and distance-response profiles.
4. **Stability and faithfulness audit:** repeat across seeds, masks, graph choices, segmentation perturbations, and patient bootstraps.
5. **External validation:** evaluate a locked candidate network in untouched patients or cohorts and with independent molecular evidence.
6. **Perturbational validation:** test a small number of high-confidence candidates experimentally before using causal language.

A failed gate is an informative scientific result. Acceptance criteria must not be weakened after results are observed simply to advance the pipeline.

## Repository Organization

```text
configs/               composed scientific, trainer, evaluation, and sweep settings
src/spatial_benchmark/ canonical model code and experiment infrastructure
scripts/               thin train, evaluation, data, sweep, and analysis entry points
tests/                 unit, integration, and smoke checks
data/                  immutable raw/clinical inputs, derived data, splits, and registries
experiments/           campaign and variant definitions
scratch/active_runs/   mutable output while a run is executing
artifacts/runs/         immutable finalized bundles; year/month is storage only
artifacts/legacy_runs/ preserved historical campaigns
state/                 local SQLite registry, locks, queue state, and operational logs
exports/runs/checkpoints/ generated semantic checkpoint catalogs and link views
reports/               cross-run analyses, figures, and tables
```

The existing `spatial_benchmark` package remains canonical. A campaign is one
bounded scientific question; a variant is one fixed scientific configuration;
a run is one seed/fold/attempt execution. New runs are registered locally and
published from scratch only after artifact verification succeeds.

Run discovery follows the scientific hierarchy
campaign → lifecycle stage → study axis → variant → seed/fold/attempt. Primary
IDs are immutable: the imported runs retain `lr_*` IDs and future workers issue
`r_*` IDs. Preferred semantic aliases are deterministic, date-free display
identifiers. The physical `artifacts/runs/YYYY/MM/` partition supports storage
and lifecycle operations; it is not how runs should be selected or
interpreted.

Useful commands from the project root:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark --help
make test
```

See `docs/experiment_protocol.md` for identifiers and lifecycle rules and
`docs/operations.md` for the one-GPU queue.

## Checkpoint Discovery

SQLite schema v3 adds `run_aliases`, `run_categories`, and
`checkpoint_catalog` to the authoritative registry. The 158 historical best
checkpoints are indexed and checksum-verified without changing their native
files: 9 diagnostic, 96 exploratory-screen, 3 validation-confirmation, and 50
locked-final records. Their total size is 1,499,185,266 bytes. Nine
exact-content groups contain 20 records and 84,324,336 redundant bytes; these
are annotations only, not authorization to delete or hard-link immutable
history.

Browse by scientific category or resolve an exact checkpoint with either a
primary ID or preferred alias:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  list-checkpoints --stage locked_final --model g2 --limit 20

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  resolve-checkpoint \
  hist.diagnostic.runtime-smoke.b0.self.p-n-b.disabled.d128.s000.fna.ana.vaf897cf26410.x1cd0a0ff3c83
```

The reviewed generated catalog is
`exports/runs/checkpoints/catalog_20260724T180010Z/`. It is a semantic view,
not a second registry. Interpret diagnostic runs only as diagnostics,
exploratory screens only as selection evidence, confirmations within their
locked validation scope, and locked-final runs within the campaign's stated
limits. Never choose a scientific conclusion from one favorable seed; use
prespecified variant aggregates.

## Data and Outputs

Local data and generated artifacts are intentionally untracked. Do not modify raw or clinical inputs in place. The root `data.tar.gz` archive, when present, is an input artifact and should be preserved unless its removal is explicitly requested.

Read [`docs/legacy_data_guide.md`](docs/legacy_data_guide.md) before accessing the current CosMx
snapshot. It documents file roles, dimensions, canonical keys, panel controls,
clinical-label ambiguity, split requirements, and known integrity exceptions.

Run Python entry points from the repository root unless a campaign README
states otherwise. The legacy core-assignment utility currently requires
clinical-schema reconciliation before its outputs can be treated as canonical;
see the data guide and `scripts/core_assignment/README.md`.

Computational workflows should provide a cheap smoke or pilot profile and, when a distinct scaled experiment is meaningful, a documented full profile. Pilot outputs are diagnostic; locked, fully validated outputs are canonical. If pilot artifacts are removed, retain the configuration, metrics, logs or summary, and the decision they informed.

## Current Status

The completed normal-core masked-expression benchmark is preserved under
`artifacts/legacy_runs/lr_spatial_benchmark_batch/` and indexed without altering
its checksum-bound outputs. Its legacy fold and attempt were not recorded, so
catalog records expose them as unknown rather than treating registry
placeholders as evidence. The authoritative experiment registry is local
SQLite under `state/tracking/`; no remote tracking service is initialized.
