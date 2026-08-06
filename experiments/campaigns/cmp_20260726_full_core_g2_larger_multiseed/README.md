# Full-Core G2 Width Scaling Across Seeds

## Status

- Phase: completed
- Outcome: larger-width improvement gate failed; Jacobian branch skipped
- Campaign: `cmp_20260726_full_core_g2_larger_multiseed`
- Scope: exploratory held-in reconstruction in one legacy true-Normal core

This campaign compares the current exact-k=1,000 G2 with a width-doubled G2.
Every cell is used for fitting, so results measure transductive representation
capacity, not validation/test accuracy or patient generalization.

## Results

All six conclusion-bearing runs completed 200 finite epochs and passed the
finalized-bundle checksum audit. The wider model increased trainable
parameters from 3,987,880 to 12,687,784 (3.18-fold), but it did not produce a
meaningful reconstruction improvement:

| Variant | Mean Huber ± seed SD | Mean PVE ± seed SD | Training time ± seed SD | Peak VRAM |
|---|---:|---:|---:|---:|
| Current G2 | 0.229036 ± 0.000143 | −0.257% ± 0.353 | 1,531.9 ± 6.3 s | 7.975 GiB |
| Wider G2 | 0.228846 ± 0.001122 | −0.629% ± 0.927 | 2,641.9 ± 14.4 s | 8.356 GiB |

The locked aggregate Huber reduction was 0.083%, below the required 2%.
Wider G2 improved Huber for seeds 1 and 2 by 0.387% and 0.415%, respectively,
but regressed by 0.554% for seed 0. Mean PVE worsened by 0.372 percentage
points, and no seed favored wider G2 on both Huber and PVE. H1-width therefore
failed. Negative last-20 loss slopes remained similar at both widths, so the
fixed 200-epoch runs were still improving; this does not rescue the failed
prespecified comparison.

The three wider PVE values were 0.441%, −1.156%, and −1.172%. None approached
95%, so the conditional Jacobian branch was skipped exactly as specified. No
Jacobian was computed or interpreted.

The two-epoch resource pilot passed. One wider seed-0 attempt was interrupted
after GPU 4 sustained severe thermal throttling; it contributed no metrics.
The identical scientific configuration completed as attempt 2 on GPU 7. No
conclusion-bearing run reported thermal throttling.

Artifacts:

- Comparison report:
  `reports/analyses/full_core_g2_larger_multiseed/comparison/report.md`
- Machine-readable comparison:
  `reports/analyses/full_core_g2_larger_multiseed/comparison/comparison.json`
- Six successful run bundles:
  `artifacts/runs/2026/07/r_20260726T115820Z_e0ce7822_s000_f00_a01_79f0edd6`,
  `...3850b579`, `...499ddf9f`, `...aaecc923`, `...e9445de2`, and
  `r_20260726T120217Z_04a51acf_s000_f00_a02_72e3f371`
- Interrupted attempt:
  `artifacts/runs/2026/07/r_20260726T115820Z_04a51acf_s000_f00_a01_9409eae2`

## Task contract

### Objective and scientific question

Train the current and larger G2 under three paired model seeds, test whether
the larger parameter budget improves fixed-mask reconstruction, and apply the
requested 95% gate before any Jacobian comparison.

Question:

> Holding data, graph, masking, optimizer, and epoch budget fixed, does
> doubling G2 representation width materially improve held-in masked-expression
> reconstruction across seeds?

- H1-width: the larger G2 has at least 2% lower mean whole-node masked Huber,
  higher mean whole-node masked variance explained, and all three paired seeds
  favor it on both metrics.
- A1-capacity-saturated: width does not help because the usable signal or
  objective, rather than parameter count, limits reconstruction.
- A2-optimization: the larger model is harder to optimize under the unchanged
  200-epoch and learning-rate budget.
- A3-seed-instability: an apparent aggregate gain is driven by one favorable
  initialization.

### Estimand, metric, and 95% gate

The primary estimand is the paired seed difference in mean held-in whole-node
masked Huber over three fixed technical masks. Lower is better. The
percentage-style secondary metric is:

```text
masked_percent_variance_explained = 100 * (
    1 - masked_squared_error / masked_target_total_sum_of_squares
)
```

It is exactly `100 * R²`, is computed within each fixed mask before averaging,
may be negative, and is not classification accuracy. No within-tolerance
accuracy is introduced after seeing outcomes.

The Jacobian branch opens only if every larger-G2 seed has mean whole-node
masked percent variance explained strictly above 95%. If it opens, compare
functional input-to-output Jacobian sketches for all six models under the same
three fixed masks. For 32 common Rademacher output probes per mask, seeded by
`(20260726, mask entry ID, probe index)`, compute FP32 vector-Jacobian products
`Jᵀu` with dropout, AMP, and parameter gradients disabled. Probe support is
restricted to masked receiver predictions; derivatives are taken only with
respect to observed input-expression entries. With two graph layers, this
estimand includes direct and multihop model-implied sensitivity.

The full Jacobian is not materialized: it would contain roughly
`(24,245,000)^2` entries. Common-probe Hutchinson estimates must report
Frobenius cosine, relative Frobenius discrepancy, and Jacobian norm ratio.
Uncertainty uses a fixed 2,000-replicate hierarchical bootstrap that resamples
the three masks with equal weight and then the 32 probes within each selected
mask; all intervals are two-sided percentile 95% intervals.

The primary Jacobian comparisons are the three paired cross-width contrasts
(current seed `s` versus wider seed `s`). All three within-current and all
three within-wider seed pairs are calibration contrasts. Eight independently
reinitialized cross-width pairs, with model seeds `9100` through `9107`, are
negative controls. The identical-checkpoint control must have cosine lower
bound at least `0.9999`, relative-discrepancy upper bound at most `0.001`, and
norm-ratio interval within `[0.999, 1.001]`; otherwise the analysis is
numerically invalid.

An operational match requires every paired cross-width contrast to have
cosine lower bound at least `0.95`, relative-discrepancy upper bound at most
`0.25`, and norm-ratio interval within `[0.80, 1.25]`. It must also be no worse
than the full range of within-current seed-pair point estimates on all three
metrics and be separated from randomized controls: its cosine lower bound
must exceed the randomized-control cosine upper bound and its discrepancy
upper bound must be below the randomized-control discrepancy lower bound.
High cosine alone is insufficient. If the gate fails, record the failed gate
and do not compute or interpret Jacobians.

### Models and controlled change

| Variant | Hidden/FFN/decoder | Layers | Heads | Edge MLP | Parameters |
|---|---:|---:|---:|---:|---:|
| Current G2 | 512 | 2 | 4 | 17→64→64 | 3,987,880 |
| Wider G2 | 1,024 | 2 | 8 | 17→64→64 | 12,687,784 |

Heads increase only to keep the per-head dimension at 128. Graph depth, edge
encoder, dropout, inputs, decoder output, and all scientific settings remain
fixed. This is a 3.18-fold parameter increase and a width intervention, not a
graph-density or depth intervention.

### Units, data, graph, and leakage

- Experimental and observational unit: one spatial core.
- Execution repeats: model seeds `0, 1, 2`.
- Technical repeats: three deterministic masks per mode within each seed.
- Input: the verified `prepared_full_v1` artifact with 24,245 cells and 1,000
  biological probes; technical controls remain excluded.
- Always-visible covariates: the same 22 morphology/imaging measurements.
- Graph: the same exact mutual k=1,000 graph with 21,029,944 directed edges,
  650 µm non-truncating guard, 17 geometry features, and checksum
  `2469064e2fe14b48f642fca09a546d9e420d9fd851b9668996da62ee8246d060`.
- Prohibited node inputs remain identifiers, coordinates, RNA-derived QC,
  library size, and vendor annotations.
- All preprocessing statistics, topology, and evaluated cells are fitted.
  Fresh masks do not create a validation or test set.

### Training, resource pilot, and stop criteria

Both variants use paired epoch masks, AdamW at `3e-4`, weight decay `1e-4`,
gradient clipping at `1.0`, mixed precision, deterministic algorithms, and
200 fixed epochs. The final epoch is canonical; there is no early stopping or
best-seed selection.

Before full execution, the wider model must finish the two-epoch diagnostic
with finite losses and gradients, peak allocated VRAM no greater than 20.5
GiB, and projected 200-epoch training time no greater than six hours. Stop a
run on graph/checksum mismatch, non-finite loss or gradient, repeated OOM after
lowering only receiver chunk size, insufficient disk, or artifact
verification failure. Do not silently alter width, graph, masks, seed set, or
epoch budget.

### Execution record

Execution began on 2026-07-26 UTC on eight NVIDIA GeForce RTX 3090 GPUs
(24,576 MiB each, compute capability 8.6, driver 580.126.09). The recorded
software stack is Python 3.12.13, PyTorch 2.11.0+cu130, CUDA build 13.0,
cuDNN 9.19, and torch-geometric 2.7.0. GPU 0 remained assigned to an unrelated
registered campaign.

The first pilot enqueue, job `q_1f5d7ab0d9d7aba856ea`, was cancelled before a
run began because GPU 0 was occupied. The replacement on GPU 7 completed as
run `r_20260726T115503Z_1334b662_s000_f00_a01_334b2e85`: 12,687,784
parameters, 8.353 GiB peak allocated VRAM, two finite epochs, and 138.686 s
end-to-end. Its two training epochs took 13.744 and 12.995 s, projecting
roughly 44.6 minutes for 200 training epochs; both resource gates passed. Its
two-epoch reconstruction metrics are diagnostic and cannot answer H1.

The six conclusion-bearing jobs were scheduled in parallel:

| Variant | Seed | GPU | Job | Run | State |
|---|---:|---:|---|---|---|
| Current | 0 | 1 | `q_636c841d3ab3d840fca3` | `r_20260726T115820Z_e0ce7822_s000_f00_a01_79f0edd6` | completed |
| Current | 1 | 2 | `q_c3daca7ebd415fb370a5` | `r_20260726T115820Z_e0ce7822_s001_f00_a01_3850b579` | completed |
| Current | 2 | 3 | `q_8383330ba980207cd584` | `r_20260726T115820Z_e0ce7822_s002_f00_a01_499ddf9f` | completed |
| Wider | 0 | 4 | `q_0cc55ff72768a7abd1b8` | `r_20260726T115820Z_04a51acf_s000_f00_a01_9409eae2` | failed: thermal relocation |
| Wider | 0 | 7 | `q_thermalretry7_0cc55ff7` | `r_20260726T120217Z_04a51acf_s000_f00_a02_72e3f371` | completed retry |
| Wider | 1 | 5 | `q_b68e78e6b6417a2366ef` | `r_20260726T115820Z_04a51acf_s001_f00_a01_aaecc923` | completed |
| Wider | 2 | 6 | `q_818b3819f426df5051f3` | `r_20260726T115820Z_04a51acf_s002_f00_a01_e9445de2` | completed |

The first wider seed-0 attempt was stopped after GPU 4 sustained 90 °C
software thermal throttling. No scientific result from that attempt is used.
Its failed bundle records category `interrupted`; attempt 2 uses the identical
canonical configuration on GPU 7. The authoritative SQLite registry retains
the exact resolved configuration, command vector, GPU assignment, heartbeat,
failure, and retry lineage for every row above. Every science job executes:

```text
/venv/main/bin/python scripts/train/run_full_core_capacity.py \
  --config {run_scratch}/config.resolved.yaml \
  --run-scratch {run_scratch}
```

The exact comparison command was:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/compare_g2_width_multiseed.py \
  --run artifacts/runs/2026/07/r_20260726T115820Z_e0ce7822_s000_f00_a01_79f0edd6 \
  --run artifacts/runs/2026/07/r_20260726T115820Z_e0ce7822_s001_f00_a01_3850b579 \
  --run artifacts/runs/2026/07/r_20260726T115820Z_e0ce7822_s002_f00_a01_499ddf9f \
  --run artifacts/runs/2026/07/r_20260726T120217Z_04a51acf_s000_f00_a02_72e3f371 \
  --run artifacts/runs/2026/07/r_20260726T115820Z_04a51acf_s001_f00_a01_aaecc923 \
  --run artifacts/runs/2026/07/r_20260726T115820Z_04a51acf_s002_f00_a01_e9445de2 \
  --failed-run artifacts/runs/2026/07/r_20260726T115820Z_04a51acf_s000_f00_a01_9409eae2 \
  --resource-pilot artifacts/runs/2026/07/r_20260726T115503Z_1334b662_s000_f00_a01_334b2e85 \
  --expected-failed-run-count 1 \
  --output reports/analyses/full_core_g2_larger_multiseed/comparison
```

### Required outputs and decisions

1. One verified resource-pilot bundle.
2. Six verified conclusion-eligible bundles: two variants × three seeds.
3. Per-seed and aggregate Huber/PVE table, paired differences, convergence,
   runtime, peak VRAM, parameter counts, and all failures.
4. A machine-readable gate record and concise comparison report.
5. Jacobian comparison artifacts only if the locked >95% PVE gate opens.

The result is supported only when the full H1-width criterion passes.
Positive but sub-2% Huber improvement is sub-threshold. A mixed seed direction
is unstable. Failure to reach 95% PVE closes only the Jacobian branch; it does
not by itself decide whether modest width scaling helped reconstruction.

### Verification commands

```bash
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor

PYTHONPATH=src /venv/main/bin/python - <<'PY'
from spatial_benchmark.configuration import compose_config
for path in (
    "configs/experiment/full_core_g2_multiseed_baseline.yaml",
    "configs/experiment/full_core_g2_multiseed_width1024.yaml",
    "configs/experiment/full_core_g2_multiseed_width1024_resource_pilot.yaml",
):
    compose_config(path)
print("configs valid")
PY

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark \
  --database state/tracking/bagm.sqlite3 verify-artifacts
```

Completion verification produced:

- infrastructure doctor: `ok: true`, SQLite integrity `ok`;
- artifact verifier: `valid: true`, no bundle or registry issues;
- full test suite: 297 passed, one optional PyArrow execution-path test
  skipped because PyArrow is not installed, and no failures;
- strict comparison protocol: complete, resource pilot passed, one failed run
  reconciled, H1-width failed, and Jacobian branch skipped.

`/workspace` is not backed by a persistent volume. Completion on this instance
is not an off-instance backup.
