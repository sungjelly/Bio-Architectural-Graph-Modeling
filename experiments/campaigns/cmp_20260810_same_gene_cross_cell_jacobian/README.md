# Same-gene cross-cell Jacobian pilot

## Status

- Phase: complete
- Outcome: negative for strict same-gene selectivity; partial predictive and
  diagonal-enrichment evidence retained
- Campaign: `cmp_20260810_same_gene_cross_cell_jacobian`
- Design: exploratory, label-free, geometry-group-held-out
- Input snapshot: public Drive export under
  `data/raw/Gastric_Cancer_Analysis/`
- Frozen contract:
  `experiments/campaigns/cmp_20260810_same_gene_cross_cell_jacobian/frozen_task_contract.yaml`

This campaign begins after the repository, raw schema, FOV geometry, and prior
GeneMAE gradient results were inspected.  It is therefore exploratory and
cannot be relabelled confirmatory.  Clinical donor/core mappings are absent,
so geometry components are deliberately opaque spatial groups rather than
patients, cores, diagnoses, or biological replicates.

## Task contract

### Objective and deliverables

Train a simple whole-node expression predictor on the currently available two
CosMx slides and test whether the cross-cell input-output Jacobian is
selectively large when source and target probe names are identical.

Required deliverables are:

1. a checksum-bound processed-data manifest and deterministic four-fold split;
2. held-out predictions for morphology-only, observed-near, annular-neighbor,
   and within-FOV source-permutation models;
3. one signed 1,000-by-1,000 standardized Jacobian per outer fold and arm;
4. same-name diagonal enrichment, row-rank, sign, and fold-stability summaries;
5. a planted analytical recovery test and a same-cell oracle positive control;
6. resource, provenance, failure, and registry records; and
7. a concise result report that retains negative and adverse evidence.

### Question, hypothesis, and alternatives

Question:

> In a receiver-whole-node-masked task, is the prediction of probe `g` more
> sensitive to nearby cells' visible copy of `g` than to other probes, and is
> that selectivity stronger than morphology, broader spatial autocorrelation,
> and an alignment-breaking null explain?

The working hypothesis is that the observed 0--25 um neighbor arm has
held-out predictive value and a stable same-name diagonal excess beyond the
controls below.

Credible alternatives are:

- the diagonal is ordinary spatial autocorrelation or shared cell state;
- the shortest-distance signal is segmentation spillover or transcript
  misassignment;
- FOV- or slide-level technical fields explain the result;
- a flexible coefficient matrix produces diagonal-looking structure without
  improving held-out prediction; or
- fold instability or gene scale creates an attribution artefact.

The 25--50 um arm and within-FOV source permutation distinguish a strictly
short-range pattern from broader/FOV-level context.  They do not eliminate
unmeasured compartments or prove communication.

### Data, units, and split

- Use all available SO_1 and SO_2 cell-expression and metadata rows after the
  exact slide-qualified `(slide, fov, cell_ID)` join.
- Exclude `Negative*` and `SystemControl*`; retain the 1,000 ordered biological
  probes, including slash-combined probes as indivisible measurements.
- Allowed non-expression inputs are the repository's fixed 22 independently
  measured morphology/imaging variables.  Vendor cell types, clusters,
  neighborhoods, niches, diagnoses, and target-derived labels are prohibited.
- Default QC policy is all cells; vendor-QC status is reported and a passed-cell
  sensitivity may be added only as a separately labelled variant.
- Build connected components of FOV origins separately by slide using a
  prespecified 0.75 mm threshold.  This includes the observed horizontal,
  vertical, and diagonal FOV pitch and lies on the stable 0.60--0.75 mm
  component plateau.  These are opaque geometry groups only.
- Assign whole components to four folds with the frozen component-ordinal map
  in the YAML contract, chosen using only slide, FOV count, and cell count.
  No FOV or component may cross folds.
- Each outer fold is the test set once.  The next fold cyclically is validation,
  the remaining two are tuning-train, and the selected ridge penalty is then
  refit on all three non-test folds.
- Learned imputation, centering, scaling, and penalty selection use no outer
  test values.  All metrics are first computed per opaque geometry component
  and then equally averaged; cells are not treated as biological replicates.

Because the clinical map is unavailable, the intended generalization unit is
an unseen geometry component within these two slides.  Patient-, donor-,
core-, disease-, and population-generalized language is prohibited.

### Graphs, model, and estimand

Graphs are constructed independently inside each FOV from measured cell-center
coordinates; self edges and cross-FOV edges are forbidden.

- near arm: up to the 12 closest cells at `(0, 25]` um;
- annular arm: up to the 12 closest cells at `(25, 50]` um;
- a receiver is in the matched primary evaluation only when both arms contain
  at least four sources;
- neighbor features are the degree-normalized mean of source-cell
  `log1p(count)` values.

The deliberately simple additive model is fitted by multivariate ridge:

```text
standardized log1p(receiver RNA)
    = intercept + morphology coefficients
    + neighbor-mean-RNA coefficients.
```

The penalty grid on normalized sufficient statistics is
`[1e-4, 1e-3, 1e-2, 1e-1, 1]`.  The intercept is not penalized.  The
morphology-only model omits the neighbor block.  The null model applies a
deterministic within-FOV derangement to source-cell states in the fixed near
edge slots, preserving the receiver, edge geometry, FOV-level expression
distribution, and source feature covariance while breaking
receiver/source-state alignment.  Any rare remapped source equal to its
receiver is dropped before degree renormalization and audited explicitly.

Rows of each cross-cell Jacobian are receiver targets and columns are source
probes.  For a receiver `v` and one included source `u`, on the standardized
scale,

```text
d prediction[v, target] / d input[u, source]
    = B[target, source] / degree(v),
```

where `B` is the fitted neighbor-mean coefficient block.  The primary matrix
is `B`, the response to a uniform one-standard-deviation shift of every
included neighbor.  Per-source derivatives and realized degrees are reported
separately.  This is a one-hop model-implied sensitivity, not a count-scale,
molecular, or causal effect.

The receiver expression is never a model input in the primary arms.  A
separate same-cell oracle arm supplies receiver expression and must recover a
strong diagonal; it is a numerical positive control and is excluded from all
scientific comparisons.

### Metrics and frozen gates

Predictive metrics are component-equal held-out MSE (primary), Huber loss,
MAE, gene-wise correlation, and relative gains.  Jacobian summaries retain
the signed matrix and report:

- median absolute diagonal divided by the median absolute off-diagonal;
- each diagonal entry's absolute rank within its target row;
- fraction of genes whose same-name source is row top-1, top-10, and top-1%;
- diagonal sign consistency and Spearman stability across outer folds;
- near-minus-annular and near-minus-permutation diagonal excess; and
- expression prevalence/variance strata.

Primary diagonal summaries require a probe to have at least 5% nonzero
prevalence and nonzero train variance in every outer-fold final training set.
All 1,000 probes remain in the matrices and descriptive outputs; this rule
only prevents unstable rare-probe entries from driving the frozen gates.

The narrow computational hypothesis is supported only if all of the following
hold without changing the thresholds after inspection:

1. near versus morphology component-equal MSE gain is at least 2% overall and
   favors near in at least three of four outer folds;
2. near versus within-FOV-permuted MSE gain is at least 1% overall and favors
   near in at least three of four folds;
3. the near median absolute diagonal/off-diagonal ratio is at least 2.0;
4. the same-name source is row top-1 for at least 25% of eligible genes and
   row top-1% for at least 50%;
5. the near median absolute diagonal is at least 1.25 times the permuted-arm
   value overall and in at least three of four folds;
6. the median pairwise fold Spearman correlation of signed diagonal values is
   at least 0.70 and at least 75% of eligible genes retain one sign in at least
   three of four folds; and
7. all numerical, leakage, finiteness, coverage, and artifact checks pass.

The segmentation-spillover alternative is flagged when near diagonal
enrichment is at least 1.5 times the annular enrichment or when the predictive
gain is confined to the near arm.  This flag does not prove spillover; it
limits interpretation.

### Controls, acceptance, and stop rules

Required controls are:

- analytical planted coefficient recovery and centered finite difference;
- exact zero self contribution in every primary cross-cell model;
- same-cell oracle diagonal recovery;
- morphology-only prediction;
- 25--50 um annular neighbors;
- deterministic within-FOV source-state derangement with 100% changed source
  mappings and audited target-self collision removal; and
- deterministic random-coefficient diagonal-enrichment reference.

Stop before the full run on a raw/schema/key/fold checksum mismatch, a
cross-FOV/self edge, test-informed preprocessing, nonfinite input/coefficient/
metric, analytical recovery failure, fewer than 80% matched eligible cells,
unsafe GPU collision, peak VRAM above 20.5 GiB, or insufficient disk.  A null
or unfavorable biological result is not a technical failure.

The pilot opens production only if one complete fold finishes, all controls
pass, projected wall time is at most two hours per fold, and the fixed design
fits one RTX 3090 with the stated margin.  Production maps folds 0--3 to CUDA
devices 0--3 and writes isolated run directories.

### Maximum defensible claim

At most this campaign can support a stable, held-out-geometry,
graph-alignment-dependent same-gene model sensitivity in the current two
slides.  It cannot establish direct cell communication, signaling, a molecular
mechanism, causality, patient replication, or clinical generalization.

### Planned commands and outputs

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/data/prepare_same_gene_jacobian.py --profile full

PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_same_gene_jacobian.py

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /venv/main/bin/python \
  scripts/train/run_same_gene_jacobian.py --fold 0 --profile pilot

PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/analyze_same_gene_jacobian.py --verify-only

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

Active outputs belong under `scratch/active_runs/`; verified fold bundles are
published under `artifacts/runs/YYYY/MM/`, and the aggregate report belongs
under `reports/analyses/same_gene_cross_cell_jacobian_20260810_v2/`.

## Results

All preparation, pilot, four-fold production, aggregation, registry, and
artifact checks completed.  The raw 18-file snapshot fingerprint is
`e1513d598d4ea910386842cdf4a6d9bd58e21318d484bdb34dfb3962f1490ea5`;
the derived-data fingerprint is
`6304132b4a57699c81b8616324dbeb2faee24b58ce70490be552595d84af34ce`;
and the frozen split fingerprint is
`12c0d46244ed443a482586fc85422672f9f04132c7def49a741c40ba48bf4264`.

The data preparation retained 407,999 cells and all 1,000 biological probes.
Matched near/annular/permutation coverage was 97.21%.  It produced 27 opaque
geometry components and the frozen four folds without using clinical,
patient, core, or cell-type labels.

The one-GPU pilot passed the analytical, oracle, finiteness, leakage, disk,
runtime, and memory controls.  Production then mapped folds 0--3 to four
separate RTX 3090 devices.  All four runs completed in 6.49--6.68 seconds with
peak allocated VRAM 5.741 GiB and no failed fold:

- `r_20260810T165340Z_718d3b86_s810_f00_a02_8ea7da1d`
- `r_20260810T165340Z_718d3b86_s810_f01_a02_a2a7fb9e`
- `r_20260810T165340Z_718d3b86_s810_f02_a02_d80bceee`
- `r_20260810T165340Z_718d3b86_s810_f03_a02_10eba65f`

These canonical attempt-2 runs preserve the preliminary native bundles as
immutable legacy evidence and link each new
run with `retry_of`.  Re-execution reproduced every stored arm result and all
20 shared checkpoint fields exactly (maximum numerical difference 0), while
adding canonical component predictions, 35 full fitted-state fields, checksums,
source snapshots, and `_SUCCESS` markers.  The native-bundle audit is
`reports/audits/same_gene_cross_cell_jacobian_preliminary_native_20260810.json`.

On the 27 held-out geometry components, the observed 0--25 um arm improved
component-equal MSE by 2.231% versus morphology alone (spatial-component
bootstrap interval 1.420--3.136%) and by 1.297% versus the within-FOV
source-state permutation (0.916--1.705%).  Both comparisons favored observed
near neighbors in all four folds.  The 25--50 um annular arm also improved on
morphology by 1.661%, showing that broader spatial autocorrelation/context is
part of the signal.

For the 932 probes eligible in every fold, the equal-fold signed near matrix
had median absolute diagonal/off-diagonal ratio 2.852.  Its diagonal was 1.824
times the permuted-arm diagonal.  Signed diagonal ranks were stable across
folds (median pairwise Spearman 0.892), and 91.4% of probes retained one sign
in at least three folds.  The analytical same-cell oracle recovered row top-1
for 100% of probes, confirming that the pipeline can detect a truly exclusive
diagonal.

The strict hypothesis nevertheless failed its frozen row-selectivity gate:
the same-name source was row top-1 for 24.46% of eligible targets (minimum
25%) and row top-1% for 41.09% (minimum 50%).  Six of seven aggregate gates
passed; thresholds were not relaxed.  Thus same-name sensitivities are
enriched and predictive, but high coefficients are not confined to matching
RNA names.  The verdict is
`strict_same_gene_selectivity_not_supported`.

The near/annular enrichment ratio was 1.353, below the frozen 1.5
short-distance spillover flag.  This does not exclude segmentation spillover,
shared cell state, technical fields, compartment structure, or ordinary
spatial autocorrelation.

The conclusion-bearing evaluation is registered as
`eval_same_gene_cross_cell_jacobian_20260810_v2`.  The complete report,
figure, matrices, per-gene table, provenance, and verification record are in
`reports/analyses/same_gene_cross_cell_jacobian_20260810_v2/`.

Completed verification:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_same_gene_jacobian.py

PYTHONPATH=src /venv/main/bin/python \
  scripts/data/prepare_same_gene_jacobian.py --verify-only --verify-raw

PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/analyze_same_gene_jacobian.py

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

The focused experiment suite passed 7/7 tests.  The repository-wide suite
finished with 941 passed, 34 failed, and 1 skipped; all 34 failures are the
pre-existing clean-clone integration tests that require ignored locked-campaign
materializations, prepared geometry, or the external MyJJu sibling repository.
No same-gene Jacobian test failed.

The maximum claim remains an exploratory, held-out-geometry,
graph-alignment-dependent model-implied sensitivity.  This campaign does not
establish communication, a molecular mechanism, causality, clinical meaning,
or patient generalization.
