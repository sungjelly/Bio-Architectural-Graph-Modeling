# Curated Result Management

## Task contract

Objective: establish `results/` as the ordered, human-facing index of important
scientific conclusions. Each result must identify the experiment type, method,
scientific question, outcome, evidence limits, and immutable source runs. The
deliverables are a versioned result schema, validated directory layout,
scaffolding command, deterministic catalog, repository policy, and tests.

The infrastructure question is whether a conclusion can be found and audited
without confusing one run, a generated report, or a favorable seed with the
scientific result. The primary hypothesis is that a small required record plus
a generated catalog makes important outcomes discoverable and provenance-bound.
The credible alternative is that free-form folders are sufficient; this would
predict that experiment type, method, run coverage, outcome, and evidence
limits remain consistently recoverable without validation. The acceptance
test is structural rather than biological: valid records pass deterministic
validation and catalog generation, while missing provenance, path drift,
duplicate IDs, unsafe paths, and incomplete evidence statements fail closed.

The unit is one curated conclusion record, which may aggregate multiple runs,
seeds, folds, patients, controls, or post-hoc evaluations. The primary metric
is the fraction of cataloged records passing the complete schema and checksum
contract; the required value is 1.0. Negative and inconclusive outcomes are
valid results. Smoke runs, arbitrary intermediate files, and a best seed alone
are not. This infrastructure makes no biological claim and does not replace a
campaign's scientific acceptance or falsification criteria.

Verification commands:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog --check
PYTHONPATH=src /venv/main/bin/python -m pytest -q \
  tests/unit/infrastructure/test_result_catalog.py \
  tests/unit/infrastructure/test_paths_identifiers.py
PYTHONPATH=src /venv/main/bin/python -m spatial_benchmark doctor
```

## Three storage layers

These paths answer different questions and must not be collapsed:

| Layer | Purpose | Mutability |
|---|---|---|
| `artifacts/runs/YYYY/MM/<run_id>/` | Complete canonical execution bundle | Immutable after successful finalization |
| `reports/` | Detailed analyses, figures, tables, and numeric source data | Versioned analysis output |
| `results/` | Curated conclusion records and a browseable catalog | Additive; corrections create a new revision or explicit supersession |

A result references immutable run IDs and artifact/report checksums. It does
not copy checkpoints, datasets, registry databases, unrestricted row-level
tables, or complete run directories. Large local figures and tables may live
beside a result record, but the record must checksum and describe them; Git
tracks the small metadata and summaries by default, not bulky payloads.

## Directory layout

```text
results/
├── README.md
├── CATALOG.md
├── catalog.json
├── <experiment_type>/
│   └── <method_family>/
│       └── <result_id>/
│           ├── result.yaml
│           ├── README.md
│           ├── figures/
│           ├── tables/
│           └── attachments/
└── core_assignment/          # preserved legacy output location
```

`experiment_type`, `method_family`, and `result_id` are lowercase ASCII slugs.
The directory path is checked against `result.yaml`; moving a record without
updating and revalidating it is an error. Result IDs are date-free semantic
identifiers of the form `res_<slug>`. Dates belong in provenance fields, not
the scientific browsing hierarchy.

Controlled experiment types are:

- `data_qc`
- `synthetic_recovery`
- `predictive_benchmark`
- `ablation`
- `stability_audit`
- `faithfulness_audit`
- `null_calibration`
- `niche_discovery`
- `external_validation`
- `perturbation`
- `descriptive_analysis`
- `infrastructure_validation`

`method_family` is a concise stable family such as `relative_qkv`,
`edge_conditioned_gat`, `hybrid_count`, or `spatial_baseline`. Exact model,
graph, masking, estimator, and software versions remain explicit in the
manifest rather than being compressed into the folder name.

## Required record content

Every `result.yaml` follows `configs/schema/result_record_v1.yaml` and records:

- identity: schema version, result ID, title, experiment type, method family,
  lifecycle stage, study axis, campaign IDs, and result status;
- method: model/estimator, graph or spatial context, masking or perturbation,
  evaluation design, and implementation version;
- conclusion: question, estimand, outcome, observed result, strongest
  alternative explanation, controls, remaining uncertainty, and maximum
  defensible claim;
- evidence dimensions kept separate: predictive gain, stability,
  faithfulness, null calibration, patient replication, external support, and
  perturbation support;
- provenance: all contributing run IDs, expected and included seeds/folds,
  failed or excluded runs with reasons, immutable source manifests or reports,
  Git commit, and creation time;
- files: relative path, role, SHA-256, and size for each curated attachment.

Allowed outcomes are `supported`, `negative`, `inconclusive`, and `blocked`.
Allowed record statuses are `draft`, `verified`, and `superseded`. A verified
record requires at least one immutable source, a nonempty run set when the
source type is a run artifact, complete evidence dimensions, checksum-verified
declared files, and a human-readable `README.md`.

## Promotion and correction

Promote a result only when it changes the evidence record or a documented
scientific/operational decision. Examples include a completed benchmark gate,
a rigorous negative result, a locked stability or null audit, an independently
replicated candidate set, or a validated method comparison. Do not promote a
smoke test, transient training curve, attractive visualization, single best
seed, or unverified exploratory observation as a verified result.

Promotion never upgrades the underlying claim. A transductive held-in analysis
remains transductive; attention remains routing; gradients remain local model
sensitivity; patient replication, faithfulness, null calibration, external
support, and perturbation evidence stay separate.

Do not edit a verified record in place when its conclusion or source set
changes materially. Create a new result ID and mark the old record
`superseded`, with `superseded_by` pointing to the replacement. Typographical
or link-only corrections may retain the result ID but must update checksums and
revision metadata.

## Workflow

Create a draft scaffold from the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py create \
  --result-id res_example_graph_gain \
  --title "Example graph-specific predictive gain" \
  --experiment-type predictive_benchmark \
  --method-family relative_qkv \
  --lifecycle-stage validation_confirmation \
  --study-axis graph_specific_gain \
  --campaign-id cmp_example
```

Fill the explicit placeholders, add only curated attachments, then validate and
regenerate both catalog files:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py validate
PYTHONPATH=src /venv/main/bin/python scripts/results/manage_results.py catalog
```

`catalog --check` is the drift check for tests and review. Catalog ordering is
deterministic by experiment type, method family, and result ID. The catalog is
a discovery view; each `result.yaml` remains the authoritative conclusion
record, and immutable run artifacts remain the execution source of truth.
