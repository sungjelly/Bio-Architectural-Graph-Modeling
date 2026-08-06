# Self-only continuous-hurdle full-core capacity

## Status

- Phase: complete
- Outcome: negative
- Campaign: `cmp_20260729_self_hurdle_full_core_capacity`
- Design: exploratory, held-in, graphless representation-capacity experiment

This campaign is separate from
`cmp_20260729_multiscale_hurdle_count_pilot`. The latter's synthetic Stage-0
gate was negative, so no real-data graph arm is authorized. That negative gate
motivates narrowing the question; it is not weakened or reused as an input
gate here.

## Objective and deliverables

Train a large self-only model for the full fixed 200-epoch budget on exactly
two pathology-confirmed adjacent-normal cores, with no validation or test
partition and no checkpoint selection. Report percentage accuracies alongside
losses and compare every primary readout with a per-gene all-fit reference.

Required deliverables are:

1. two 2-epoch resource pilots (`ANC-03`, `ANC-05`);
2. a checksum-bound resource-gate receipt;
3. exactly two 200-epoch science runs on the same cores if that gate passes;
4. immutable final checkpoints, fixed held-in mask predictions, provenance,
   resource accounting, and a two-core report that exposes both results.

## Scientific question, hypotheses, and alternatives

Question: can within-cell observed expression plus permitted morphology and
imaging covariates encode masked raw-count structure better than a per-gene
reference when model capacity and overfitting are deliberately favored?

Primary hypothesis: on both cores, the fully trained representation improves
whole-node detection balanced accuracy and positive-count prediction over the
all-fit per-gene reference.

Support requires, on both cores:

- positive detection balanced-accuracy gain;
- at least `2%` lower positive standardized-log1p Huber loss; and
- at least `2%` lower positive count-state MAE.

Failure on either core is negative for this two-core capacity hypothesis. The
experiment still completes if the prespecified result is negative.

Credible alternatives are that per-gene prevalence/median statistics are
already adequate, the mask hides too much within-cell information, the model
primarily encodes morphology or technical intensity rather than biology, or
optimization fails. High held-in accuracy can also result from transductive
memorization and common cell-state structure; it does not demonstrate
generalization or cell-cell interaction.

## Estimand and maximum claim

The estimand is held-in masked raw-count reconstruction. Partial-gene,
whole-node, and spatial-block masks remain separately reported. All
preprocessing and per-gene references are fitted within the same core.

The maximum permitted claim is exploratory within-core representation
capacity for two adjacent-normal cores. This design cannot support
unseen-patient generalization, stable population biology, a spatial predictive
dependency, cell-cell communication, a mechanism, or causality.

`ANC-03` and `ANC-05` are two independent spatial cores, but they were selected
before this campaign as the largest and smallest resource-pilot cores. They are
not a random population sample. Mask replicates are technical repeats, not
biological replicates.

## Inputs, leakage, and graph prohibition

- Reuse only the checksum-bound prepared artifacts for `ANC-03` and `ANC-05`
  from `cmp_20260729_adjacent_normal_10core_hybrid_count_gat`.
- Targets are the 1,000 biological probe counts; `Negative*` and
  `SystemControl*` probes remain excluded.
- Permitted always-visible covariates are the same 22 morphology/imaging
  fields.
- Direct identifiers, coordinates as node inputs, RNA-derived library size/QC,
  vendor cell types/clusters/neighborhoods/niches, and hidden target values are
  prohibited.
- Coordinates are used only to construct fixed spatial-block masks.
- The model and runner expose no edge-index or edge-attribute input. The
  required configuration `graph` section is an explicitly disabled schema
  placeholder; no graph is built, loaded, or passed to the model.

The fully masked target cell's expression-derived library size is unavailable.
The mask is applied inside both the discrete-count and continuous-expression
encoder branches.

## Architecture and objective

The graphless model has:

- gene-specific embeddings for eight fixed raw-count states plus an input-only
  mask token;
- exact standardized `log1p(raw count)` input;
- a `768`-dimensional encoder;
- three `768 -> 1536 -> 768` residual feed-forward blocks;
- a `768`-dimensional decoder; and
- two outputs per gene: detection logit and positive standardized-log1p count.

The fixed parameter count is `16,917,200`. There are no trainable node IDs,
sample IDs, graph parameters, or coordinate features.

Loss is the equal mean of balanced detection BCE and positive-only Huber loss
with delta `1.0`. Counts use fixed nonnegative half-up decoding and the fixed
states `0`, `1`, `2`, `3`, `4-7`, `8-15`, `16-31`, `>=32`.

Primary reporting includes hurdle loss, detection balanced accuracy,
detection sensitivity/specificity/precision, positive exact and within-one
accuracy, positive count-state MAE, positive continuous Huber/MAE, eight-state
exact/balanced accuracy, and every corresponding available per-gene reference.
All accuracy proportions are also stored as percentages.

## Stages, resource gates, and stop criteria

### Resource pilots

Run one 2-epoch pilot on each core. Both must pass:

- FP32/AMP absolute hurdle-loss discrepancy `<= 0.001`;
- peak allocated VRAM `<= 12 GiB`;
- finite loss and gradients;
- projected 200-epoch time per core `<= 6 GPU-hours`;
- projected sum for the two science runs `<= 12 GPU-hours`;
- graph input/construction count exactly zero; and
- filesystem used space strictly below `55 GB` decimal.

The gate must read completed registered bundles and bind their run IDs,
configuration hashes, completion markers, and diagnostics. A failed gate
blocks both science jobs; no automatic resizing or retry under a changed
configuration is allowed.

### Science

After the gate passes, run exactly one fixed 200-epoch job on each core.
Production peak allocated VRAM must remain `<= 20.5 GiB`, disk must remain
below `55 GB`, and the total campaign GPU time must remain below the absolute
`24 GPU-hour` ceiling. The preferred budget is `12 GPU-hours`.

There is no early stopping, validation, test set, best-checkpoint selection,
or hyperparameter search. Only the final epoch is checkpointed. A hardware or
numerical failure may use one identical retry; a changed model is a new
campaign, not a retry.

Stop on disk hard-stop, non-finite values, OOM, failed AMP equivalence, graph
construction/use, missing provenance, or a projected resource-gate failure.

## Registration and exact run order

From the repository root, after focused verification:

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 \
  create-campaign \
  --campaign-id cmp_20260729_self_hurdle_full_core_capacity \
  --name "Self-only continuous-hurdle full-core capacity"

PYTHONPATH=src /venv/main/bin/python \
  scripts/train/materialize_self_hurdle_campaign.py

PYTHONPATH=src /venv/main/bin/python \
  scripts/train/enqueue_self_hurdle_campaign.py --stage resource
```

After both resource jobs finalize:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/train/verify_self_hurdle_resource_gate.py

PYTHONPATH=src /venv/main/bin/python \
  scripts/train/enqueue_self_hurdle_campaign.py --stage science
```

The enqueue command only queues registered jobs. Existing safe GPU workers run
them automatically. GPU `4` is excluded. Workers must be started or restarted
only after the normal GPU/process preflight.

## Verification

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_analyze_self_hurdle_capacity.py \
  tests/unit/spatial_benchmark/test_self_hurdle.py \
  tests/unit/spatial_benchmark/test_self_hurdle_capacity_runner.py \
  tests/unit/spatial_benchmark/test_materialize_self_hurdle_campaign.py \
  tests/unit/spatial_benchmark/test_enqueue_self_hurdle_campaign.py \
  tests/unit/spatial_benchmark/test_verify_self_hurdle_resource_gate.py \
  tests/unit/infrastructure/test_queue_command.py \
  tests/unit/infrastructure/test_configuration.py

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor

PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/analyze_self_hurdle_capacity.py
```

## Current result

The two resource pilots and two fixed 200-epoch science runs completed. All
four immutable bundles and indexed final-epoch checkpoints verify.

| Stage | Core | Run ID |
|---|---|---|
| resource | `ANC-03` | `r_20260729T172556Z_47141900_s000_f00_a01_289503ce` |
| resource | `ANC-05` | `r_20260729T172557Z_de468a3e_s000_f00_a01_49f86de9` |
| science | `ANC-03` | `r_20260729T172739Z_0570cda7_s000_f00_a01_7449d530` |
| science | `ANC-05` | `r_20260729T172739Z_ea3ca4d1_s000_f00_a01_74a0b1a1` |

The prespecified whole-node gate was negative on both cores. Detection
balanced accuracy was `64.85%` versus `51.09%` for the all-fit per-gene
reference on `ANC-03`, and `65.63%` versus `50.35%` on `ANC-05`. Positive
continuous Huber loss improved by `26.40%` and `28.95%`. However, positive
count-state MAE improved by only `0.82%` and `1.60%`, below the frozen `2%`
criterion on both cores. The two-core capacity hypothesis is therefore not
supported.

Conservative GPU accounting over the full registered start-to-end duration of
all four runs is `0.0872757 GPU-hours`. Maximum science peak allocated VRAM was
`1.6453 GiB`; maximum recorded filesystem use was `49.3834 GB`, below the
`55 GB` hard stop.

The concise report is
[`reports/analyses/self_hurdle_full_core_capacity/RESULTS.md`](../../../reports/analyses/self_hurdle_full_core_capacity/RESULTS.md).
The checksum-bound machine-readable comparison, complete model/reference
metric table, and report manifest are in the same directory. This is a
graphless, held-in, exploratory negative result; it provides no evidence of
unseen-core generalization, cell-cell interaction, biological mechanism, or
causality.
