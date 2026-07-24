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
data/
  raw/                 local immutable source data
  processed/           generated or curated intermediate data
  clinical/            local clinical metadata, when available
scripts/
  core_assignment/     shared FOV-to-core assignment utility
results/
  core_assignment/     shared core-assignment outputs
workflows/
  <experiment>/        one bounded idea, hypothesis, or experiment
```

Each workflow is an independent, reviewable experiment rather than an extension of one monolithic pipeline. A workflow may test an architecture, objective, graph construction, baseline, ablation, null model, interpretation method, or validation concept.

A typical workflow contains:

```text
workflows/<experiment>/
  README.md
  configs/
  scripts/
  src/                 optional reusable implementation
  tests/               workflow-local tests
  results/             ignored generated analyses
  outputs/             optional ignored model artifacts
```

Every workflow README must define its scientific question, hypothesis, estimand, permitted claim, inputs, leakage risks, baselines and controls, split strategy, commands, outputs, go/no-go criteria, compute plan, and current conclusion. Cross-workflow dependencies must use declared artifacts identified by an immutable run ID, configuration, checksum or manifest, and provenance record; they need not be tracked by Git.

## Data and Outputs

Local data and generated artifacts are intentionally untracked. Do not modify raw or clinical inputs in place. The root `data.tar.gz` archive, when present, is an input artifact and should be preserved unless its removal is explicitly requested.

Run Python entry points from the repository root unless a workflow README states otherwise. Shared core assignments are generated with:

```bash
python scripts/core_assignment/plot_fov_core_layout.py
```

Computational workflows should provide a cheap smoke or pilot profile and, when a distinct scaled experiment is meaningful, a documented full profile. Pilot outputs are diagnostic; locked, fully validated outputs are canonical. If pilot artifacts are removed, retain the configuration, metrics, logs or summary, and the decision they informed.

## Current Status

The previous clustering experiments have been removed, and `workflows/` is ready for new Bio-Architectural Graph Modeling work. Project-wide dependencies and model entry points will be defined with the first workflow.
