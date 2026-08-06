# Current Model Architecture Audit, 2026-07-26

## Status

- Phase: complete
- Outcome: verified
- Scope: current full-core high-k G2 and parameter-matched self-only models,
  with the implemented BAGM model ladder documented for context
- Classification: descriptive architecture and training-method audit; no new
  model fitting or sealed-test evaluation

## Task contract

### Objective and deliverables

Produce a portable, single-file HTML report that documents:

1. the current model inputs, tensor shapes, graph representation, forward
   paths, layer structure, outputs, and experimental role;
2. exact trainable-parameter totals and per-module/per-tensor breakdowns
   reconciled between code, resolved configurations, and finalized
   checkpoints;
3. preprocessing, masking, optimization, exact dense-attention execution,
   precision, checkpointing, evaluation, and reproducibility methods;
4. runtime, peak-memory, convergence, and checkpoint facts for the current
   completed model pair;
5. differences from the earlier sparse held-out benchmark architecture; and
6. implemented but inactive models or branches, clearly separated from models
   that have finalized checkpoints.

Expected outputs are `model_architecture_report.html`,
`parameter_breakdown.csv`, `model_specs.json`, and the reproducible generator.

### Question, hypothesis, and alternatives

This is a descriptive audit rather than a new scientific experiment. Its
central verification question is whether the current high-k G2 and
cell-autonomous comparator actually satisfy their declared architectural
contract.

The declared contract predicts that both models have exactly 3,987,880
trainable parameters; share the same 1,000-gene/22-covariate encoder, 512-wide
representation, two residual mixing layers, and nonlinear decoder; and differ
only in whether the G2-specific parameter budget processes measured
intercellular edges or is repurposed for within-cell computation.

Credible alternatives are a stale configuration, a code/checkpoint mismatch,
hidden parameter-count inequality, or a comparator that accidentally consumes
topology or edge information. These alternatives are rejected only if source,
resolved configs, checkpoint tensor names/shapes, checksums, and recorded run
manifests reconcile.

### Estimand and permitted claim

There is no new predictive estimand. The report may claim only that the
documented architecture and training procedure match the preserved current
artifacts. Existing performance results remain held-in, single-core capacity
evidence and are not reinterpreted as generalization, communication,
mechanism, or causality.

### Inputs and protected-data policy

Inputs are tracked source/configuration files, the current campaign README,
the local registry, finalized run manifests/checkpoints/metrics, and the
existing comparison output. Raw expression, transcript, polygon, and clinical
tables are not required and will not be read. No row-level predictions,
restricted identifiers, or protected metadata will be emitted.

### Acceptance and stop criteria

The audit is complete only if:

- both finalized current runs and their declared checksums verify;
- code/config/checkpoint parameter totals reconcile exactly;
- the two current models have equal total parameters and their architectural
  differences are localized explicitly;
- every checkpoint tensor is assigned to a documented module group;
- training and evaluation methods distinguish held-in fitting from held-out
  prediction;
- missing or inactive components remain labeled as such;
- the HTML is self-contained, semantic, accessible, and print-safe; and
- the report generator, artifact verification, and project doctor pass.

Stop and report a mismatch rather than silently inferring architecture from
one source if code, config, manifest, or checkpoint evidence conflicts.

## Results

- Both final checkpoints load strictly into the current implementation and
  contain 42 FP32 tensors, 3,987,880 trainable parameters, and 15,951,520 raw
  parameter bytes.
- The parameter budget reconciles exactly as 1,036,800 node-encoder
  parameters, 5,568 edge/surrogate-encoder parameters, 1,084,928 parameters
  in each of two mixing blocks, and 775,656 decoder parameters.
- The G2 checkpoint consumes the verified 21,029,944-edge mutual-k1000 graph
  and 17 measured geometric edge attributes through exact receiver-chunked
  GATv2 execution. The matched checkpoint is behaviorally invariant to graph
  and edge arguments and repurposes the same parameter budget within each
  cell.
- Training used exactly paired epoch masks, AdamW, 200 fixed epochs, mixed
  precision, deterministic algorithms, no edge or neighbor sampling, and the
  final epoch without validation selection.
- The existing held-in whole-node result remains positive but subthreshold:
  G2 improves mean masked Huber by 0.479% and all three technical masks favor
  G2, but the locked 2% capacity gate fails.
- A fresh seed-0 constructor reconstruction shows that corresponding
  post-encoder initial tensors are not identical because module creation and
  initialization policies differ. The comparison is parameter-matched, not
  initialization-, operator-, activation-, or compute-matched.

## Deliverables

- `model_architecture_report.html` — portable, self-contained report
- `model_specs.json` — machine-readable model, data, graph, training, run,
  performance, and verification facts
- `parameter_breakdown.csv` — all 84 state tensors with shapes, roles, counts,
  byte sizes, and model fractions
- `generate_report.py` — deterministic report generator and checker

### Verification

```bash
/venv/main/bin/python \
  reports/analyses/current_model_architecture_20260726/generate_report.py \
  --check
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark verify-artifacts
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

Verified on 2026-07-26:

- generator check: 32/32 architecture and artifact assertions passed;
- finalized artifact verification: passed for the 161-run registry;
- project doctor: `ok: true`, no issues or warnings; and
- targeted model/dense-GAT/full-core-training tests: 27 passed.
