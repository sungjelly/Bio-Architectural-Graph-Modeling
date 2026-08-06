# GeneMAE Jacobian-pair annotation

## Status

- Phase: complete
- Outcome: inconclusive for biological correspondence; descriptive annotation complete
- Campaign: `cmp_20260802_genemae_jacobian_pair_annotation`
- Design: exploratory post-hoc model-behaviour analysis
- Upstream gradient run:
  `r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a`

This analysis begins after the upstream model metrics, 39-gene matrix, and
prior gradient-audit outcome were inspected. It is therefore exploratory and
cannot be reframed as confirmatory. It reuses the immutable upstream matrix;
it does not retrain a model or modify a run bundle.

## Task contract

### Objective and deliverables

Extract the already computed gene-to-gene sensitivity matrix from the
seven-seed GeneMAE ensemble, rank ten gene pairs reproducibly, and determine
whether primary literature or authoritative biological resources annotate
those pairs.

Deliverables are:

1. axis-labelled signed directed and unsigned symmetric 39-by-39 matrices;
2. the ten largest unique off-diagonal mutual-sensitivity pairs, retaining
   both directed signed entries;
3. separate receiver-row and source-column Jacobian-profile correlations so
   derivative magnitude is not mislabeled as correlation;
4. a pair-level evidence table with evidence type, tissue context,
   perturbational status, citation, and limitations;
5. machine-readable provenance and verification records; and
6. a concise report stating the maximum defensible claim and adverse evidence.

### Scientific question, hypothesis, and alternatives

Question:

> Do the largest off-diagonal sensitivities in the stored GeneMAE marker-gene
> matrix correspond to independently documented gene relationships?

Exploratory hypothesis: the leading pairs will be enriched for documented
shared complexes, extracellular-matrix assembly, or cell-state programs.

Credible alternatives are:

- the biology-selected 39-gene universe makes agreement circular;
- large entries primarily reflect same-cell co-expression, cell identity, or
  normalization rather than graph-dependent information;
- the unsigned symmetrisation hides direction and sign;
- aggregate rankings hide core- or seed-specific instability; and
- literature search preferentially finds plausible stories for familiar
  markers while unsupported pairs remain underreported.

The distinguishing output is not a binary biological-validation label. Each
pair is assigned the strongest supported evidence class, and negative or
indirect evidence remains visible.

### Model choice, input, and provenance

The selected model is the complete seven-checkpoint GeneMAE ensemble from
`cmp_20260730_myjju_genemae_10core_comparison`, not the best individual seed.
It had the lowest common-task held-in 20% partial-gene Huber among the models
compared (`0.348210`) but did not pass the graph-use gate (`1.09%` gain versus
the node-label-permuted graph, below the frozen `2%` threshold).

The sole matrix source is:

```text
artifacts/runs/2026/07/
  r_20260731T083020Z_8f363415_s000_f00_a01_5375b35a/
  interpretation/report.json
```

Its required SHA-256 is
`6a5bdbcc17c397a84696062a249cd56f83b17be3a6cb3b80e8772606c1b9cf33`.
The analysis must fail closed on checksum, schema, axis, dimension, symmetry,
finite-value, or transform mismatch.

### Estimands and ranking

Rows of the signed matrix are target genes and columns are source genes:

```text
J[target, source] = (1 / N) sum_input_cells
  d(sum_output_cells reconstruction[target]) / d input[source]
```

The input is unmasked `log1p(CP10k)` expression. The derivative combines the
explicit node-wise branch with same-cell graph self-loops and cross-cell paths
up to four GAT layers. It is a local model sensitivity, not an expression
correlation, direct molecular interaction, or causal effect.

The primary unordered-pair score is the upstream published transform:

```text
M[i, j] = 0.5 * (abs(J[i, j]) + abs(J[j, i]))
```

The diagonal is excluded. The top ten are ranked by descending `M`, with the
fixed upstream gene order as the deterministic tie-break. No effect-size
threshold or pair is changed after ranking.

For the literal phrase "correlated in the Jacobian", two secondary statistics
are reported separately for each unordered pair, excluding the two pair genes
from the compared coordinates:

```text
receiver-profile rho = Spearman(J[i, other sources], J[j, other sources])
source-profile rho   = Spearman(J[other targets, i], J[other targets, j])
```

These quantify similarity of model-sensitivity profiles. They do not imply a
direct gene-gene interaction. P-values are not used because the analysis is
post-hoc and the 741-pair family was not prospectively powered.

### Units, controls, nulls, and limitations

- Observational unit: adjacent-normal tissue core (`10` held-in cores).
- Technical repeats: seven model seeds; these are not biological replicates.
- Gene universe: 39 source-selected markers, not the full 1,000-gene panel.
- Split: all 117,386 cells were used in fitting; there is no held-out-core or
  patient-generalization estimate.
- Preprocessing leakage: the full-cell CP10k denominator uses masked entries
  in the upstream partial-gene task.
- Required adverse controls from the upstream audit remain part of the result:
  the graph-gradient structure null failed in `0/10` cores, and only `KRT8`
  passed target-specific graph-use eligibility.
- The source-selected gene universe and several source-selected pairs make
  literature agreement circular annotation, not independent validation.
- No additional raw or patient-level data are read or exported.

### Biological-evidence rubric

Evidence is recorded on separate axes:

1. direct physical or biochemical relationship;
2. shared curated complex, pathway, or structural program;
3. independent tissue- or cell-state co-expression;
4. perturbational evidence linking one member to the other;
5. gastric-specific evidence; and
6. no convincing pair-specific evidence found in the documented search.

Primary studies and authoritative curated resources are preferred. A shared
marker program is weaker than a direct interaction. A database or literature
match is annotation, not validation of the model or its direction. Failure to
find evidence is reported as `not found`, not proof of absence.

### Acceptance, falsification, and stop criteria

The analysis is complete only if:

- the upstream checksum and all matrix invariants pass;
- all 39 row and column labels are preserved exactly;
- the ten pair ranks reproduce deterministically;
- both directed signed entries are retained for every selected pair;
- every pair has at least one auditable source or an explicit documented
  no-evidence result;
- unsupported and indirect pairs are not promoted to verified biology;
- focused tests and the analysis verifier pass; and
- repository doctor and final `git status --short` are reviewed.

The biological-correspondence hypothesis is not supported merely by a high
fraction of literature-annotated pairs because the panel was selected using
biological knowledge. A checksum/schema failure, nonfinite matrix, or
unresolvable source stops the workflow. An unsupported pair or negative
result does not stop the workflow; it is part of the result.

### Maximum defensible claim

At most, a selected pair may be described as a literature-annotated,
held-in, aggregate GeneMAE model-implied sensitivity. This campaign cannot
establish graph dependence, direct cell-cell communication, patient
replication, a biological mechanism, or causality.

### Resources, outputs, and verification

This is a CPU-only, read-only analysis of one immutable JSON artifact.
Outputs belong under:

```text
reports/analyses/genemae_jacobian_gene_pairs_20260802/
```

Planned verification commands:

```bash
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/spatial_benchmark/test_jacobian_pair_analysis.py

PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/analyze_genemae_jacobian_pairs.py

PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

## Results

The required upstream checksum matched and both 39-by-39 matrices reproduced
exactly. The top ten unique off-diagonal mutual sensitivities were:

| Rank | Pair | Mutual magnitude | Dominant directed entry |
|---:|---|---:|---:|
| 1 | `COL1A1-COL3A1` | 0.07119 | `COL1A1<-COL3A1`, 0.09508 |
| 2 | `KRT8-KRT19` | 0.06298 | `KRT8<-KRT19`, 0.07221 |
| 3 | `KRT8-KRT18` | 0.03766 | `KRT8<-KRT18`, 0.04068 |
| 4 | `KRT18-KRT19` | 0.03459 | `KRT19<-KRT18`, 0.03617 |
| 5 | `COL1A1-COL1A2` | 0.03423 | `COL1A1<-COL1A2`, 0.06002 |
| 6 | `COL1A1-DCN` | 0.03195 | `COL1A1<-DCN`, 0.04708 |
| 7 | `COL3A1-DCN` | 0.02903 | `COL3A1<-DCN`, 0.03288 |
| 8 | `COL1A2-COL3A1` | 0.02730 | `COL3A1<-COL1A2`, 0.04227 |
| 9 | `COL1A1-ACTA2` | 0.02269 | `COL1A1<-ACTA2`, 0.03880 |
| 10 | `ACTA2-PSCA` | 0.02143 | `PSCA<-ACTA2`, -0.03456 |

Six pairs have direct structural or biochemical support: type-I/type-III
collagen crosslinking, KRT8 partnerships with KRT18 or KRT19, the type-I
collagen heterotrimer, and decorin binding to type-I or type-III collagen.
Three have weaker shared-program or indirect functional support:
`KRT18-KRT19`, `COL1A2-COL3A1`, and `COL1A1-ACTA2`.

`ACTA2-PSCA` did not have a verified direct relationship. One PSCA-FLAG
co-immunoprecipitation/mass-spectrometry table listed ACTA2 with only two
peptides, alongside several homologous actins, but the paper validated a
different interactor and did not follow up ACTA2. Current IntAct and STRING
queries returned no curated pair edge. Gastric studies instead localize PSCA
primarily to epithelium and ACTA2 primarily to stromal/CAF compartments. The
two negative aggregate Jacobian directions are therefore more parsimoniously
explained by compartment or cell-state contrast than molecular antagonism.

The literal profile-correlation analysis provided an additional warning.
Only `COL1A1-COL3A1` appeared in both the top receiver-row and source-column
lists; several other high correlations crossed canonical immune, epithelial,
and stromal lineages. Aggregate scale, cell mixtures, or low-rank model
structure can therefore generate high profile correlation without a direct
gene relationship.

The annotation hypothesis remains inconclusive because favorable matches are
circular within this biology-selected 39-gene panel, the analysis is held-in,
and the upstream graph-gradient structure gate passed in `0/10` cores. The
maximum claim remains a literature-annotated model-implied sensitivity.

## Outputs and completed verification

- concise report:
  `reports/analyses/genemae_jacobian_gene_pairs_20260802/report.md`
- signed matrix:
  `reports/analyses/genemae_jacobian_gene_pairs_20260802/signed_directed_jacobian.csv`
- primary ranking:
  `reports/analyses/genemae_jacobian_gene_pairs_20260802/top_mutual_sensitivity_pairs.csv`
- complete evidence table and sources:
  `reports/analyses/genemae_jacobian_gene_pairs_20260802/biological_evidence.csv`
  and `sources.json`
- provenance and verification:
  `reports/analyses/genemae_jacobian_gene_pairs_20260802/provenance.json`
  and `verification.json`

The extractor and independent `--verify-only` replay passed all checksum,
shape, finiteness, transform, rank, evidence-coverage, and adverse-result
checks. The focused Jacobian and upstream-gradient suite passed `24` tests.
The complete repository suite passed `888` tests with one expected skip for
the absent optional `pyarrow` dependency and three upstream deprecation
warnings. Repository artifact verification returned no bundle or registry
issues. Project doctor reported `ok: true`, SQLite integrity `ok`, no issues or
warnings, no queued/running/claimed jobs, and `38.627` GiB free. The campaign
is registered locally with status `complete`.
