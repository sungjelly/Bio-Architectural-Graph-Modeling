# SO_1 + SO_2 Pearson preprocessing trial

Phase: complete. Outcome: supported for preprocessing plausibility only. This is an exploratory preprocessing comparison,
not a trained integration model or confirmatory biological validation.

## Task contract (fixed before examining transformed results)

- Objective: produce aligned, reusable Pearson and signed-log Pearson features
  for both slides and report whether they merit a subsequent training comparison.
- Hypothesis: vendor-style quasi-Poisson residuals reduce representation dependence
  on total RNA counts relative to the existing log1p-count feature recipe without
  materially degrading provisional annotation consistency.
- Alternatives: library depth contains real biology; apparent improvement erases
  biological signal, reflects donor composition, or just compresses feature scale.
- Estimand: descriptive held-out-donor preprocessing diagnostics, not predictive
  model gain, causal biology, or proof of technical batch removal.
- Units: cells are observations; cores are repeated samples; 14 anonymous donor
  groups are biological units. SO_1/SO_2 have confounded tissue composition.
- Split: training cores 1–10 and 15–24 (10 donors, 301,198 cells); diagnostics
  cores 11–14 and 25–28 (4 donors, 106,461 cells). Consecutive odd/even pairs are
  verified from the protected workbook. No donor crosses partitions or slides.
- All 407,659 mapped cells and 1,000 biological genes retained; SO_2 FOV246's
  340 unmapped cells remain excluded by the existing source contract. Vendor
  QC flags retained for sensitivity diagnostics. No new observation deletion.
- Fit gene frequencies, residual means/sample SDs, baseline gene scales and
  32-component PCA on training cells only, pooled across both slides. Transform
  held-out cells with frozen parameters. Cell totals use all observed panel genes.
- Comparators: (1) log1p(raw counts), gene standardized; (2) log1p(counts per
  10,000), gene standardized; (3) quasi-Poisson Pearson, phi=1.01, train gene
  standardization, upper clip=10; (4) sign(z)*log1p(abs(z)) after (3).
- Vendor input formula follows scPearsonPCA commit
  41c499026d789ff35288de2ddc62f506a9d51cf0. Exact built-in AtoMx implementation is
  not publicly established. Signed log is a nonstandard ablation, not a claimed
  AtoMx option. Ordinary log1p(residual) is checked for invalid values.
- Common analytical PCA centers each transformed training matrix. This differs
  from vendor PCA's uncentered second moment after upper clipping; the exported
  residual features themselves use the documented vendor-style input formula.
- Primary diagnostic: median across held-out cores of variance-weighted squared
  correlations between 32 PCs and log1p(cell total), lower is the depth-reduction
  hypothesis. Report all cores, not cell-independent significance tests.
- Guardrail: within-slide held-out vendor-label balanced accuracy should not fall
  by >0.05 versus log1p(raw) on either slide. The labels a–l are opaque, derived
  from expression, and not verified as aligned biological types across slides.
  This guardrail measures consistency only, not independent biological validity.
- Slide-classification AUC and cross-slide nearest-neighbor mixing are descriptive
  confounded diagnostics, never optimization targets. No forced slide alignment.
- Controls: independent numerical formula checks; synthetic planted type and
  library-depth effects; training-label permutation null; baseline comparisons;
  all-cell versus QC-pass diagnostics; frozen fit and donor-separated evaluation.
- Plausibility requires finite outputs, verified alignment, correct formula, lower
  primary depth diagnostic than raw-log baseline and passing annotation guardrail.
  Failure is reported as negative/inconclusive without relaxing these criteria.
- Stop on ordering/provenance mismatch, nonfinite features, zero-frequency genes,
  zero training variance, unavailable GPUs, or projected disk free space <5 GiB.
  No automatic historical deletion. Current ~15 GiB is sufficient for ~3.3 GiB
  derived features plus small diagnostics. User permits cleanup if storage fills.
- Compute: four idle RTX3090 GPUs, one independent transform/PCA per GPU; CPU
  loader and diagnostic classifiers use bounded threads. No neural training,
  checkpoints, cross-slide spatial edges, or registration of tissue coordinates.
- Output: registered preprocessing run with config/code/environment/data/split
  provenance, append-only metrics, verification and completion marker; derived
  per-core matrices under data/processed/so12_pearson_integration_v1; diagnostic
  figures/tables and report. Raw counts and existing graph caches are referenced.
- This workflow uses an explicitly non-training bundle contract, because the
  repository's model RunArchive only supports a different analysis-only protocol.
- Masking limitation: full-cell totals depend on every gene. Exported features
  are suitable for fully observed representation experiments; applying a mask
  after this transform is not leakage-safe masked-count prediction. That requires
  a separate observed-only normalization design and raw-count target handling.

## Commands

From the project root:

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/run_so12_pearson_preprocessing.py --smoke
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/run_so12_pearson_preprocessing.py
```

The full run is launched through the accompanying one-shot supervisor service.
The runner verifies dimensions, all-core metadata alignment, numerical controls,
checksums, finite output, and frozen fit before publishing a successful bundle.
Final conclusions and the next discriminating test will be appended after review.

## Completed pilot

Successful run: `r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410`.
Scientific variant: `sci_660b87cbf38fb2fc`. Seed 0, fixed fold 0, attempt 2.
The first registered attempt `r_20261016T082320Z_660b87cb_s000_f00_a01_cd1c1544`
failed at concurrent CUDA eigensolver lazy initialization. Serial initialization
resolved this operational problem; scientific settings were unchanged. Its failed
bundle, partial arrays, traceback and completion marker remain archived.

| Transform | Held-out median depth R² | SO1 balanced annotation accuracy | SO2 balanced annotation accuracy |
|---|---:|---:|---:|
| Log1p raw, standardized | 0.37208 | 0.70622 | 0.73492 |
| Log1p CP10k, standardized | 0.32606 | 0.67875 | 0.72962 |
| Pearson | 0.03308 | 0.71663 | 0.76012 |
| Signed-log Pearson | 0.03072 | 0.73676 | 0.72669 |

All three alternatives pass the fixed pilot gate. Pearson reduces depth
association by 91.1% relative to raw log and improves all eight held-out cores.
QC-pass depth diagnostics agree. Signed log improves depth in only four cores
relative to plain Pearson and reduces SO2 annotation agreement by 0.0334 relative
to Pearson. Prefer plain Pearson for the next controlled training comparison;
retain signed log as an ablation. Neither result establishes batch correction,
independent biological preservation, or better masked-expression prediction.

All 56 exported matrices passed full checksums, shape and finite-value checks.
Independent float64 recomputation on 84 rows per transform had maximum error
1.41e-6 (Pearson) and 2.71e-7 (signed log), below the fixed 2e-5 tolerance.
All 28 cores match original metadata coordinates and count totals. No source
observations, old checkpoints or historical results were deleted.

Derived dataset: `data/processed/so12_pearson_integration_v1/`, 3.046 GiB.
Run runtime 63.3 seconds; one transform per GPU 0–3, peak allocated VRAM
4.23–5.03 GiB. Free storage after completion: 8.49 GiB. The independent report,
portable HTML, execution logs and plot links are in
`reports/analyses/so12_pearson_preprocessing/`. The data and split are registered
as `so12_pearson_integration` version `v1` and
`so12_10train_4diagnostic_donors_v1`.

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so12_pearson_preprocessing.py --run-id r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410
PYTHONPATH=src /venv/main/bin/python scripts/analysis/curate_so12_pearson_preprocessing.py
```

The original full-run command refuses to overwrite the existing derived dataset.
Reproduction requires a separate BAGM_DATA_ROOT with the same verified inputs and
an empty processed/so12_pearson_integration_v1 destination. All executed source,
parameters and provenance are retained in the successful immutable bundle.

The project doctor confirms SQLite integrity and all new completion markers.
It still reports the pre-existing checkpoint index gap (85 artifacts / 84 indexed)
and its global 25-GiB free-space threshold. This bounded preprocessing trial used
the documented 5-GiB reserve and finished with 8.49 GiB free. The diagnostic receipt
is `reports/analyses/so12_pearson_preprocessing/doctor.json`; no unrelated checkpoint
catalog migration or global disk-threshold change was performed.

Next discriminating experiment: train the same model with raw-log, CP10k-log,
Pearson and signed-log inputs under a separately locked observed-only normalization
contract, raw-count targets and donor-held-out evaluation. Preserve per-core
spatial graphs and slide IDs. Independent annotations or orthogonal measurements
are needed before asserting biological integration.
