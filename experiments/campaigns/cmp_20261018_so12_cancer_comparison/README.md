# Test-core representation and annotated-region comparison

Phase: complete. Outcome: inconclusive. This is exploratory: h0/h4 maps and the
annotated histology have already been inspected. User authorized execution,
baseline fitting-population decisions, and removal of unnecessary derived
artifacts if storage requires it; protected raw/clinical inputs remain immutable.

## Contract

Compare frozen epoch-194 BAGM h0,h1,h2,h3,h4 with PCA and scVI on the identical
81,607 cells of cores 15 (38,145) and 23 (43,462). Produce a five-column figure
per core (annotated histology, h4, h0, PCA, scVI), supplementary intermediate
layers, full resolution tables, registration diagnostics and a portable report.
Question: how well can unions of representation clusters delineate the supplied
hand-outlined regions? Greater h4 agreement is a hypothesis, not a requirement.
Alternatives include already sufficient cell-autonomous expression/morphology,
overpartitioning, core effects, annotation selection and section misregistration.

The latest source run is
`r_20261017T044837Z_caf76350_s000_f00_a01_12df41be`, selected best epoch 194.
It trained on 21 cores, validated on 5 and reserved cores 15/23 for testing.
Therefore fit PCA and scVI on the same 264,491 training cells, with validation
cores permitted only for scVI early stopping; do not fit either on the test
cores. PCA uses 50 components of the existing frozen train-standardized Pearson
features (centering fitted on training only). scVI uses raw counts of the same
1,000 biological targets, a 50-dimensional latent state, two hidden layers of
128 units, negative-binomial likelihood and slide batch covariate. Both slides
are represented in training. No clinical labels, core identifiers, coordinates
or morphology enter these expression-only baselines. scVI inference uses frozen
weights and known training batch categories, with no query adaptation. This is
not an input-matched ablation of BAGM's morphology information.

Before baseline execution, scVI settings were completed as follows: seed 0,
batch size 1,024, maximum 100 epochs, validation-ELBO patience 15, learning rate
0.001, KL warmup 20 epochs and early-stopping warmup 20 epochs. Restore the
strict minimum full validation ELBO, with no test-based choice. The shorter KL
warmup is explicit because a 400-epoch default would exceed this run's budget.

Reuse the verified h0/h4 embeddings and graphs from
`ev_so12_paper_test_h0_s0_20261018` and
`ev_so12_paper_test_hL_s0_20261017`. Extract h1-h3 with full-core, all-observed,
deterministic FP32 inference. Require exact h0/h4 reproduction and ordinary
prediction invariance. Preserve the selected checkpoint, fitted normalization,
gene order, cell order, sources and all existing immutable bundles.

For every representation construct a joint test-cell exact cosine k=30 graph,
row L2 normalization, undirected unweighted union, no spatial edges added.
Use RBConfiguration Leiden to convergence at resolutions 0.25, 0.5, 1, 2,
seeds 0 and 1. Reuse the already fixed resolution-1 h0/h4 labels. Primary
figures use resolution 1 seed 0 regardless of annotation agreement. Report all
resolutions and both seeds; report ARI stability without a post-hoc threshold.
Test-cell joint clustering is transductive descriptive partitioning, even
though representation fitting respects the held-out-core split.

Clinical inputs are `data/clinical/Core_15.png` and `Core_23.png`. Blue outlines
are interpreted as the user-designated cancer-region annotations; do not infer
diagnosis from image appearance. Registration must use tissue geometry and
image landmarks, never expression, cluster labels or overlap optimization
against cancer boundaries. Inspect matching, orientation, scale and coverage.
If trustworthy registration cannot be established, produce comparison maps
and explicitly withhold pixel scores rather than invent correspondence.

On a method-independent raster with one deterministic cell owner per pixel,
measure Dice/F1 = 2TP/(2TP+FP+FN), IoU, precision and recall. Multiple clusters
may be unioned; predicted pixels anywhere outside the outlined region within
the evaluable core count as FP. The primary support is the aligned segmented
cell footprint. Report coverage, overlap ownership and an additional full-region
score/coverage ceiling so intercellular gaps are visible. Pixels outside the
outlined regions are treated as reference-negative for this annotation-agreement
exercise, not proof of histologically cancer-free tissue. Uncertain boundaries
and section differences require a sensitivity audit.

Report two distinct estimands: (1) per-core best-union Dice, choosing the union
using that core's annotations, explicitly an ORACLE descriptive upper bound;
(2) choose the cluster union using the other core's annotation, freeze it, then
score the held-out core. Joint cluster IDs permit this two-direction transfer.
Never present an oracle or resolution-selected maximum as unbiased predictive
accuracy. Optimize union exactly by sorting cluster TP/pixel area and evaluating
all prefixes. Include all-positive and empty baselines, perfect synthetic and
spillover controls, and shuffled-label negative controls. Cores, not pixels,
are biological units; two cores and one encoder seed do not support population
confidence intervals or superiority claims.

## Execution and acceptance

Run synthetic metric/registration/feature invariance checks first. Register the
post-hoc evaluation locally. Write active files only under its scratch directory,
record configuration, input/code/environment hashes and append-only progress.
Use managed supervisor workers, GPU locks and bounded CPU concurrency. Budget
at most 1.5 GiB incremental disk and preserve a 5-GiB reserve; reuse references,
avoid duplicate datasets and save only compact baseline weights and test latent
arrays. Before any deletion, record exact reproducible derived targets and why
they are not dependencies of this comparison or the selected source. No deletion
is currently necessary. No external data uploads.

Acceptance requires correct split membership and cell/gene order, finite
embeddings, untouched source bindings, verified graph and Leiden settings,
all requested maps, complete resolution/seed metrics (or an evidenced external
registration blocker), reviewed registration and plotted figures, checksummed
publication and registry completion. Stop the affected branch on alignment,
source identity, numeric, storage or reproducibility failure and retain evidence.
An unfavorable PCA/scVI/BAGM comparison is a valid result. Code, exact commands,
timings, final paths and outcomes will be recorded below after execution.

## Progress and registration decision

The comparison is registered as `ev_so12_cancer_comparison_s0_20261018`.
The managed layers stage completed at 05:16 UTC in 102 seconds. All 81,607
rows of h1-h3 are finite; h0/h4 reproduction and ordinary predictions are
bitwise exact for both cores. Three focused synthetic extraction checks passed.

Geometry-only similarity fitting (tissue silhouette, all rotations and both
reflection parities) supports approximately minus-90-degree alignment. The
root and annotation agent independently reviewed overlays before metric
calculation. Tissue-envelope Dice is 0.9762 for core 15 and 0.9332 for core 23;
these are registration diagnostics, not cancer scores. Median boundary errors
are about 9 and 17 micrometres. Core 23 has substantial internal/acellular
coverage disagreement and a 95th-percentile boundary discrepancy near 308
micrometres. Proceed with explicitly approximate section-level agreement,
fixed transform for all methods, coverage reporting and 50-micrometre boundary
sensitivity. This does not establish pixel-exact histology registration.

An isolated scVI dependency installation initially attempted redundant Torch
downloads and was interrupted before installation. Only its identified
temporary download cache (695,818,694 file bytes) was removed; input data and
model artifacts were not deleted. The corrected installation explicitly omits
dependency resolution and adds only the separately resolved missing packages.
The base environment remains unchanged. Installation provenance is retained
with the isolated environment and will be copied to the final report.

PCA completed at 05:22 UTC: 50 components retain 9.4074% of frozen training
Pearson-feature variance. scVI completed at 05:30 UTC after 77 epochs, restoring
epoch 61 (zero-based; epoch 62 one-based), full validation negative ELBO
373.8721008300781. Its fixed test posterior means were extracted without any
weight update. The independent scVI checkpoint is accurately indexed under
`r_20261018T052052Z_04078a70_s000_f00_a01_14d78fb9`, rather than mislabeled as
the parent BAGM model. Its small native archive additionally includes actual
frozen, unmasked validation count-reconstruction aggregates required by the
existing archive contract; those are not comparable to BAGM's masked loss and
did not affect checkpoint selection. An archive-only CPU device-argument error
was corrected with the first helper snapshot and retry reason preserved.

All seven exact-neighbor graph checks passed. To reduce wall time, the
execution-only CPU budget was increased to eight concurrent workers: six for
the disjoint h0-h4 sweep and two for PCA/scVI. Available logical CPUs are 80
and the measured cgroup quota is 38.4 cores. All graph recipes, partition seeds
and convergence settings remain unchanged. The final aggregation waits for
both partial sweeps and reuses their checked results.

## Commands and artifact paths

The isolated scVI interpreter is `/venv/so12-scvi/bin/python`, inheriting the
unchanged base packages; exact added versions and installation audit accompany
the final report. All commands run from the repository root with
`PYTHONPATH=.:src:scripts/analysis`. Long stages use the matching managed wrappers
and supervisor configurations retained in `ops/supervisor/bagm-so12-cancer-*`.

```bash
/venv/main/bin/python scripts/analysis/run_so12_cancer_comparison.py --stage initialize
/venv/main/bin/python scripts/analysis/run_so12_cancer_comparison.py --stage layers
/venv/main/bin/python scripts/analysis/run_so12_cancer_comparison.py --stage pca
/venv/main/bin/python scripts/analysis/run_so12_cancer_comparison.py --stage scvi
/venv/main/bin/python scripts/analysis/run_so12_cancer_comparison.py --stage graphs
/venv/main/bin/python scripts/analysis/run_so12_cancer_comparison.py --stage leiden
```

The optional disjoint parallel sweeps use `run_so12_cancer_neural.py` and
`run_so12_cancer_baseline_leiden.py`. Frozen source and resource-lock receipts
are retained in `stages/`. Registration was performed and visually reviewed
before scoring; do not rerun its exploratory fitter over the reviewed result.
After clustering, metrics, figures and reporting use their respective
`so12_cancer_metrics.py`, `so12_cancer_figures.py` and `so12_cancer_report.py`
entry points with `--output scratch/active_runs/ev_so12_cancer_comparison_s0_20261018`.
Manual review precedes `publish_so12_cancer_comparison.py`, which verifies and
renames the bundle. `promote_so12_cancer_comparison_result.py` records the
qualified outcome and regenerates the result catalogs.

Expected final destination:
`reports/analyses/so12_cancer_comparison/ev_so12_cancer_comparison_s0_20261018/`.
Primary figures are `figures/core_{15,23}_five_panel.{png,pdf}` and
`combined_five_panel.{png,pdf}`; supplementary figures include all seven
representations, all-resolution score curves and oracle-union error maps.
`metrics/pixel_agreement.{csv,json}` retains complete results, and
`report/index.html` is portable with embedded figures.

Combined focused scientific checks: 30 passed, covering extraction, baseline
fit/inference, exact neighbors/Leiden, multi-cluster metrics, geometry recovery,
mask extraction and figure rendering. Separate publication/report and scVI
archive tests are recorded alongside their source snapshots. Existing
unrelated repository changes are preserved.

## Audited outcome

All 56 partitions completed by 06:40 UTC. Full consolidation passed at
06:41:25; metrics, core mixing, figures and the portable report completed at
06:45:43 UTC. The long baseline sweep reflects convergence: the final scVI
partition took about 20 minutes. No iterations, seeds or scientific settings
were relaxed.

Fixed resolution 1, seed 0 footprint Dice/F1:

| Method | Oracle core 15 | Oracle core 23 | Transfer core 15 | Transfer core 23 |
|---|---:|---:|---:|---:|
| h4 | 0.7836 | 0.9171 | 0.7457 | 0.8780 |
| h3 | 0.7921 | 0.9143 | 0.7332 | 0.8550 |
| h2 | 0.7797 | 0.9115 | 0.6589 | 0.8614 |
| h1 | 0.7865 | 0.9011 | 0.7400 | 0.8114 |
| h0 | 0.7804 | 0.8867 | 0.6705 | 0.8152 |
| PCA | 0.7710 | 0.8648 | 0.6492 | 0.7781 |
| scVI | 0.7292 | 0.9041 | 0.6687 | 0.8497 |

Oracle unions select and score on the same core. Transfer columns refer to
the scored core, with union selection on the opposite core. All-positive
footprint F1 is 0.6347/0.8502 for cores 15/23. At this primary setting h4 has
the highest transfer score on both cores, while h3 has the highest core 15
oracle and h4 the highest core 23 oracle. This is not a uniform ranking across
settings: h4 macro transfer changes from 0.81185 to 0.77172 with seed 1, while
h2 seed 1 reaches 0.80576. Primary h4 seed ARI is 0.5373. The h0 macro transfer
of 0.74282 is very close to the all-positive baseline of 0.74248; PCA transfer
of 0.71368 is lower. The 20 within-core shuffled-label controls do not preserve
spatial structure and do not justify population-level significance.

Only 69.5%/65.0% of outlined pixels have segmented-cell coverage. Full-region
ideal coverage ceilings are 0.8200/0.7880; h4 full-region oracle F1 is
0.6549/0.7310 and transfer is 0.6385/0.6912. Fixed-union 50-micrometre sensitivity
changes h4 macro transfer from nominal 0.8118 to 0.7487 (erosion), 0.8249
(boundary exclusion), or 0.8235 (dilation). Core 23 internal tissue mismatch
and unknown annotation exhaustiveness remain limitations. Cross-core graph
edges fall from 14.98% at h0 to 4.32% at h4, versus 24.59% for PCA and 32.36%
for scVI; this does not distinguish tissue biology from sample effects.

An independent audit recomputed 1,064 binary score records and certified 112
core-wise optimal unions without discrepancies. Root and an independent
reviewer inspected all six PNG figures; the independent review also verified
PDF structure and the report's embedded image bytes. Focused scientific,
report and publication tests passed; their logs and test sources are retained.
The 5-GiB-reserve doctor check passes storage and SQLite integrity, but still
reports the pre-existing historical checkpoint-catalog gap (90 artifacts,
89 indexed). The new scVI archive/catalog entry independently verifies.

No raw/clinical inputs, processed datasets or scientific checkpoints were
deleted. Only the documented failed-install temporary download cache was
removed. Free storage before publication is about 6.06 GiB. All original
source/checkpoint/normalization bindings remain unchanged.

Next discriminating test for a stronger paper claim: verify the histology-to-cell
transform and annotation exhaustiveness, fix the mapping/setting on separate
annotated development cores, then assess additional independent cores/patients.
The current comparison is complete as a descriptive analysis and inconclusive
for population superiority. Atomic publication and curated-result registration
follow this audited contract snapshot.

## Publication completed

The verified local bundle is now at
`reports/analyses/so12_cancer_comparison/ev_so12_cancer_comparison_s0_20261018/`.
Manifest SHA256: `e9a345642c160602ddd6b4c5c377c415377a6489453b8d60c82f34108a60c2b0`.
The evaluation and campaign are completed in the local registry. The curated
result is
`results/descriptive_analysis/representation_comparison/res_so12_test_cancer_representation_agreement/`;
its strict source verification passed, all 12 result records validate, and
the deterministic result catalogs were regenerated. The immutable bundle
contains the audited pre-publication contract snapshot.

## Presentation revision: categorical contrast

The user requested more distinguishable cluster colors and a thicker black
region outline on the five-panel figures. This is a presentation-only revision:
fixed resolution 1/seed 0 labels, all cell footprints, reviewed ROI geometry,
histology, scores, and conclusions stay unchanged. The same restrained categorical
lookup is applied by cluster ID across both cores and every method; it does not
depend on region membership or annotation agreement. Acceptance requires distinct
colors for every cluster, exact preservation of the owner-to-label raster mapping,
unchanged input hashes, and visual review of both core panels. This revision makes
no new scientific claim and requires only CPU rendering.

The original publication remains immutable. Render the revised core 15, core 23,
and combined five-panel PNG/PDF pairs in scratch, then publish them separately at
`reports/analyses/so12_cancer_comparison/ev_so12_cancer_comparison_s0_20261018_presentation_v2/`.
The revision includes its color key, source hashes, and rendering parameters.

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/restyle_so12_cancer_figures.py \
  --source reports/analyses/so12_cancer_comparison/ev_so12_cancer_comparison_s0_20261018 \
  --output scratch/active_runs/ev_so12_cancer_comparison_s0_20261018_presentation_v2
PYTHONPATH=src /venv/main/bin/python -m pytest -q tests/unit/spatial_benchmark/test_so12_cancer_figures.py
```

Completed: all six revised image files passed checksum/format checks and visual
review; all cluster IDs/counts and raster mappings are preserved, with distinct
colors for every cluster. The black outline is 1.45 pt (previously 0.5 pt), with
a narrow 2.15-pt white under-stroke. Four focused tests passed. All 326 files in
the original publication still match its immutable manifest. The presentation
package preserves the renderer and shared source snapshots plus verification.

### Boundary-only correction

The user's follow-up requests the original colors and layout with only thicker
black boundaries in the core 15 and 23 PNGs. The `--outline-only` mode preserves
the original color key exactly, uses black 1.5-pt lines (three times the original
width), and renders only those two PNGs. The source images and earlier revisions
remain intact. Verify original color-key equality, image dimensions, and unchanged
pixels away from the outlines and the necessary Blue-to-Black caption correction.

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/restyle_so12_cancer_figures.py \
  --source reports/analyses/so12_cancer_comparison/ev_so12_cancer_comparison_s0_20261018 \
  --output scratch/active_runs/ev_so12_cancer_comparison_s0_20261018_presentation_boundary_v3 \
  --outline-only
```

Completed and verified in the sibling
`ev_so12_cancer_comparison_s0_20261018_presentation_boundary_v3/figures/`.
Both original 3113×1430 PNG dimensions and the complete original color key are
identical. Pixel comparison found zero changes outside the boundary neighborhoods
and caption; the original PNG checksums still match their publication manifest.

### Combined figure with external caption

The next presentation revision combines both cores into one two-row/five-column
image (core 15 above core 23). It preserves the original palette and thick black
outlines, enlarges method headings to 20 pt and scale-bar labels to 14 pt, and
removes per-core/settings titles and the footer. Those details are supplied as
an external figure caption. Acceptance requires unchanged palette/inputs, both
complete rows, legible annotations, and no clipped headings or scale bars.

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/restyle_so12_cancer_figures.py \
  --source reports/analyses/so12_cancer_comparison/ev_so12_cancer_comparison_s0_20261018 \
  --output scratch/active_runs/ev_so12_cancer_comparison_s0_20261018_presentation_combined_v4 \
  --combined-publication
```

Completed: `ev_so12_cancer_comparison_s0_20261018_presentation_combined_v4/figures/cores_15_23_five_panel.{png,pdf}`
contains both rows at 300 dpi. Original palette and scientific input hashes are
preserved. Root and independent visual review confirmed readable enlarged labels,
complete tissue/region boundaries, and removal of the core/settings/footer text.
The external caption is retained in the presentation delivery manifest.

### Wider horizontal panel spacing

The user requested more horizontal space between the five panels in both rows.
The spacing revision changes `wspace` from 0.025 to 0.15 and expands the canvas
from 22 to 24.1569 inches wide, preserving the physical panel slots and 12.6-inch
height. Colors, labels, fonts, outlines, row order, scale bars, and scientific
inputs remain unchanged. Verify panel width conservation, original palette and
input hashes, both complete rows, and unclipped annotations.

```bash
PYTHONPATH=src:scripts/analysis /venv/main/bin/python scripts/analysis/restyle_so12_cancer_figures.py \
  --source reports/analyses/so12_cancer_comparison/ev_so12_cancer_comparison_s0_20261018 \
  --output scratch/active_runs/ev_so12_cancer_comparison_s0_20261018_presentation_spaced_v5 \
  --combined-publication --column-spacing .15
```

Completed in the sibling `ev_so12_cancer_comparison_s0_20261018_presentation_spaced_v5/`.
PNG/PDF visual review passed for both rows. PNG width increases from 6528 to
7162 pixels while height remains 3721 pixels at 300 dpi; panel slots retain
their physical size. Palette and input hashes match the prior combined revision.
