# Jacobian biological-validation audit

## Status

- Phase: audited
- Outcome: entrywise biological correspondence is not testable from the
  current artifacts; the available evidence does not support a biological
  Jacobian claim
- Analysis date: 2026-07-26 UTC
- Scope: post-hoc audit of the tokenized full-core G2 models and the related
  prior TLS-focused G2 interpretability control

This analysis was requested after model outcomes and the relaxed-sensitivity
protocol had already been examined. It is therefore exploratory. It does not
modify the immutable run bundles or the active post-hoc sensitivity workflow.

## Audit result

The requested correspondence is not testable from the current artifacts.
There is no materialized Jacobian, input-gene by output-gene matrix, or ranked
gene/cell/edge list. The active workflow uses Rademacher vector-Jacobian
products and retains only `A2`, `B2`, and `AB`, which estimate global
whole-Jacobian cosine, relative discrepancy, and norm ratio. These summaries
do not identify individual high entries.

For one mask, the represented output has 9,700,000 coordinates and the
effective observed input has 87,280,000 coordinates. Their dense product is
846,616,000,000,000 entries, or 3.386 PB in FP32. The implementation
deliberately avoids that object.

The model-quality prerequisite is also weak. Mean balanced and nonzero
accuracies are below the all-fit per-gene modal reference for both widths, and
both models have exactly 0% recall for tokens 1 and 2. The identical-checkpoint
numerical control passed exactly across three masks and 96 probes (cosine 1,
relative discrepancy 0, norm ratio 1), validating repeatability but not
biology.

A related earlier continuous-G2 TLS control provides useful negative evidence:
top-routed senders were enriched for the predeclared CCL19/CCL21/CXCL13
organizer score, but matched edge deletion and organizer-channel intervention
did not jointly support a faithful TLS-related dependency. This is a
counterexample to equating biological plausibility with model reliance.

The maximum conclusion is:

> The current evidence does not establish that high Jacobian entries
> correspond to verified biology. It also does not establish absence of such
> correspondence; the required entrywise statistic was never produced.

Outputs:

- `report.html`: portable self-contained report
- `evidence.json`: aggregate machine-readable evidence and claim register
- `sources.json`: primary literature and resource provenance
- `verification.json`: report, checksum, citation, portability, and claim
  checks

Verification completed with all report checks passing and
`spatial_benchmark doctor` reporting `ok: true`, SQLite integrity `ok`, and no
issues.

## Task contract

### Objective and deliverables

Determine whether the current artifacts permit a defensible test of whether
large Jacobian values correspond to independently established biological
relationships. Deliver:

1. a provenance-checked audit of the Jacobian estimand and stored outputs;
2. a biological-evidence rubric that keeps model sensitivity, database
   annotation, observational replication, orthogonal support, and causal
   perturbation separate;
3. an audit of the strongest related biological positive-control analysis
   already present in the repository;
4. a portable, self-contained HTML report; and
5. a concrete next experiment with acceptance and falsification criteria.

### Scientific question, hypotheses, and distinguishing predictions

Question:

> Do high, stable, model-dependent input-to-output sensitivity blocks from the
> current tokenized G2 models show more independent biological support than
> matched null blocks?

- H1-biological-correspondence: a prespecified high-sensitivity set is stable
  across seeds and masks, separates from randomized models, and is enriched
  for independently curated signaling relationships after matching for panel
  coverage, expression prevalence, and spatial/cell-type structure.
- A1-no-entrywise-statistic: the current workflow estimates only whole-Jacobian
  similarity and therefore cannot rank genes, gene pairs, cells, or edges.
- A2-class-imbalance shortcut: apparent sensitivities arise from models that
  mostly reproduce the dominant zero token and do not support biological
  interpretation.
- A3-coexpression-or-identity: high sensitivity reflects intracellular
  coexpression, cell identity, regional fields, or graph smoothing rather than
  cross-cell signaling.
- A4-parameterization: ranks are driven by token projection scale or
  initialization and do not survive model-randomization or input-scale
  controls.
- A5-biological-plausibility-without-faithfulness: marker or database overlap
  is present, but interventions show that the model does not depend on the
  proposed signal.

H1 requires all of the following predictions: an axis-labelled entry or block
statistic exists; its ranking is stable; randomized controls are lower; the
enrichment exceeds matched nulls; and a targeted deletion or bounded
intervention changes the relevant predictions in the expected direction.
Failure of any prerequisite limits the result to annotation or an
inconclusive/negative audit.

### Estimand and maximum claim

The audited sensitivity estimand is the local derivative of centered
four-class masked-receiver logits with respect to relaxed observed
four-channel token indicators, projected onto each token simplex. Under the
whole-node mask, differentiated inputs are other observed nodes; the statistic
combines one-hop and two-layer multihop paths. It is not a raw-count
derivative, a correlation coefficient, a direct ligand-receptor edge, or a
causal effect.

The maximum permitted claim for a successful future analysis is:

> a stable, null-calibrated, model-implied predictive sensitivity with
> literature/database annotation in one transductively fitted core.

Patient generalization, independent biological replication, a biological
mechanism, and causality are outside this design.

### Units, primary metric, baselines, controls, and nulls

- Biological/observational unit: one true-Normal spatial core.
- Training repeats: three seeds per width; these are not biological
  replicates.
- Technical repeats: three whole-node masks and stochastic trace probes; these
  are not biological replicates.
- Future primary metric: enrichment odds ratio for a locked top sensitivity
  family versus an external reference, with a 95% interval from the
  biological unit once independent units exist. In this single-core audit,
  only descriptive enrichment and technical stability are permitted.
- Required baselines: all-zero and per-gene modal prediction; same-cell-only,
  morphology/regional-context, and graph-smoothing predictors.
- Required controls: identical checkpoint, parameter-randomized model, label
  or target permutation, input-scale/token-projection normalization, and a
  planted semi-synthetic positive control.
- Required biological null: degree-, distance-, cell-type-, compartment-, and
  prevalence-matched candidate pairs; global shuffling alone is insufficient.

### Split, leakage, multiplicity, and circularity

The current results are held-in and transductive. All cells, graph topology,
token prevalence, and per-gene modes come from the same core. No result may be
described as validation or generalization. Candidate selection and biological
lookup must be separated: reference sets, top-set thresholds, matching
variables, and the multiple-testing family must be locked before inspecting
the ranked signals. Knowledge used to choose a program is a positive control,
not independent validation.

### Acceptance, falsification, and stop criteria

The requested correspondence is accepted only if:

1. a signed or norm-based gene/program block statistic with explicit input and
   output axes is stored;
2. the statistic is reproducible across all three model seeds and masks;
3. it passes identical-model recovery and separates from randomized models;
4. the underlying target has graph-specific predictive gain over appropriate
   non-graph baselines;
5. enrichment survives matched nulls and multiplicity correction; and
6. a bounded intervention supports faithfulness.

The current audit stops without an enrichment test if no entry/block ranking
exists, because inventing ranks from trace-level sufficient statistics is
mathematically invalid. A failed predictive gate, failed randomization,
unstable ranks, null-level enrichment, or failed intervention falsifies the
corresponding claim rather than being silently omitted.

### Inputs, resources, artifacts, and verification

Audited inputs:

- `reports/analyses/full_core_g2_count_tokens_multiseed/comparison/comparison.json`
- `scratch/active_runs/posthoc_g2_token_relaxed_categorical_sensitivity_v1/analysis_manifest.json`
- `scratch/active_runs/posthoc_g2_token_relaxed_categorical_sensitivity_v1/protocol.json`
- `scratch/active_runs/posthoc_g2_token_relaxed_categorical_sensitivity_v1/resource_pilot.json`
- `scratch/active_runs/posthoc_g2_token_relaxed_categorical_sensitivity_v1/identical_control_review.json`
- `src/spatial_benchmark/categorical_sensitivity.py`
- `scripts/analysis/run_g2_token_categorical_sensitivity.py`
- `reports/analyses/full_core_high_k_capacity/interpretability/analysis.json`

This audit is CPU-only and read-only with respect to models and data. Expected
outputs are `evidence.json`, `sources.json`, `report.html`, and
`verification.json`.

Verification commands:

```bash
PYTHONPATH=src /venv/main/bin/python \
  reports/analyses/jacobian_biological_validation_20260726/generate_report.py

/venv/main/bin/python \
  reports/analyses/jacobian_biological_validation_20260726/verify_report.py

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```
