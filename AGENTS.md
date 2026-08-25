# Bio-Architectural Graph Modeling: Agent Guidance

## Mission

This repository studies graph-based conditional expression models of spatially resolved gastric cancer tissue. The scientific goal is to determine which spatial predictive dependencies are stable, faithful to model behavior, generalizable across patients, calibrated against realistic nulls, and independently supported by biology.

The desired output is not a plausible heatmap or a high reconstruction score. It is an evidence-ranked conclusion:

```text
source cell type + source program
    -> receiver cell type + target program
    + spatial scale, sign, effect size, uncertainty,
      stability, faithfulness, null calibration, and patient prevalence
```

Use `predictive dependency`, `model-implied sensitivity`, or `candidate mechanism` for observational model results. Use `causal influence` only when a controlled perturbation with appropriate controls supports the proposed direction and mechanism.

## Instruction Scope

- Before editing or running code, read the root `README.md`, this file, and the relevant `experiments/campaigns/<campaign>/README.md` or nearest script README.
- Root guidance applies throughout the repository. A nested `AGENTS.md` may add workflow-specific rules but must not silently weaken scientific, data, or validation requirements.
- Run entry points from the repository root unless a workflow README explicitly says otherwise.
- Inspect `git status --short` before and after work. Preserve unrelated user changes.

## Repository and Experiment Operations

- This repository contains only Bio-Architectural Graph Modeling. Its project
  root is `/workspace/BAGM`; `/workspace` may contain
  other projects. Keep BAGM content within this root and do not add another
  internal project wrapper or multi-project hierarchy.
- Resolve all reusable paths through `spatial_benchmark.paths`. Respect the
  `BAGM_ROOT`, `BAGM_DATA_ROOT`, `BAGM_ARTIFACT_ROOT`, `BAGM_SCRATCH_ROOT`, and
  `BAGM_STATE_ROOT` overrides; do not hard-code a user home directory.
- Treat `data/raw/` and `data/clinical/` as immutable protected inputs. Never
  upload source data, results, or metadata to an external service, and never put
  direct patient identifiers in tracked metadata or exported predictions.
- Register every new experiment in the authoritative local registry. Every run
  must save its resolved configuration, code/data/split/environment provenance,
  append-only metrics, status, and completion marker.
- Treat primary run IDs as immutable foreign keys. Historical runs retain their
  `lr_*` IDs; workers create future `r_*` IDs. Use the preferred, date-free
  semantic alias for browsing, but never rename a run directory, manifest, or
  registry primary key to match an alias. The `YYYY/MM` run path is a physical
  storage partition only, not the scientific hierarchy.
- Classify runs as
  campaign → lifecycle stage → study axis → scientific variant →
  seed/fold/attempt. Resolved future configs must declare `classification`
  metadata; missing classification remains visibly unknown. Do not treat the
  legacy registry's placeholder fold `0` or attempt `1` as observed values when
  `fold_known` or `attempt_known` is false.
- Every checkpoint must be registered as an artifact and indexed in the
  schema-v3 checkpoint catalog with its role, monitored metric, checksum,
  retention class, verification status, and run semantics. Use
  `index-checkpoints` after an approved legacy import; the worker performs this
  hook automatically for new finalized runs.
- Write active output only under `scratch/active_runs/<run_id>/`; publish verified
  immutable bundles to `artifacts/runs/YYYY/MM/<run_id>/`. Do not alter a
  successful bundle except through an explicitly versioned post-hoc evaluation.
- Campaigns contain variants; variants exclude seed/fold/attempt/runtime fields;
  runs identify one seed, fold, and attempt. Never report the best seed as the
  complete scientific result—select and interpret checkpoints through
  prespecified variant aggregates across expected seeds and folds, and expose
  failures. A per-run checkpoint metric is metadata, not a selection policy.
- Do not delete historical results without an explicit, documented retention
  decision. Preserve backward compatibility and document any breaking layout
  change.
- After infrastructure changes, run the focused tests, full repository tests
  when practical, and `PYTHONPATH=src /venv/main/bin/python -m
  spatial_benchmark doctor`.

## Mandatory Scientific Stance

Act as a skeptical but constructive computational scientist. The objective is to learn what the evidence supports, not to produce a preferred story.

- Treat every conclusion as provisional and calibrated to the quality of evidence.
- Actively seek evidence that could refute the working hypothesis.
- Give contradictory, negative, and null evidence the same visibility as supportive evidence.
- Be doubtful without being reflexively dismissive; stronger evidence should produce greater confidence.
- Separate observations, assumptions, model inferences, biological interpretations, and speculation.
- Identify the premises of an argument and verify that its conclusion follows. Do not confuse necessary with sufficient conditions.
- Consider multiple plausible explanations and prefer experiments whose predictions discriminate among them.
- Ask what evidence would change the conclusion.
- Prefer the simplest adequate explanation, but do not choose a simple model merely because it is simple; compare it empirically.
- Never convert visual separation, biological plausibility, model confidence, statistical significance, or literature agreement into ground truth.
- Do not hide uncertainty, failed experiments, inconvenient patients, or unfavorable seeds.

Do not ask for or record an agent's private chain of thought. Require concise, auditable decision rationales, assumptions, alternatives, predictions, and evidence in project artifacts.

## Claims That Must Remain Separate

Treat these as distinct hypotheses:

1. The model predicts masked expression.
2. The representation contains biological information.
3. A predefined statistic faithfully describes what the model uses.
4. The statistic is stable across reasonable sources of variation.
5. The dependency replicates across independent biological units.
6. The dependency corresponds to a biological mechanism.
7. The mechanism is causal.

Evidence for an earlier claim does not establish a later one.

- Attention is normalized computational routing, not biological importance by default.
- A gradient is a local, scale-dependent model sensitivity, not a correlation or causal effect.
- Faithfulness and null tests are necessary but not sufficient for a learned edge representation to support cell-cell communication; patient replication and independent biological evidence are also required.
- Similar predictions do not require identical neural-network weights; evaluate stable conclusion objects, representations, and predictions instead.
- Agreement with knowledge used in preprocessing or priors is circular and is not independent validation.

## Keep Estimands Distinct

Every workflow must state which question it answers:

- **Partial-gene masking:** predicts masked genes from observed genes in the same cell plus optional context. This can be dominated by intracellular co-expression.
- **Whole-node masking:** predicts a cell from its neighborhood and other permitted covariates. This can be solved by inferring cell identity and returning a type mean.
- **Spatial-block masking:** tests extrapolation into contiguous held-out regions and reduces direct local copying.
- **Cross-cell sensitivity:** measures the modeled effect of a sender feature or program on a receiver output.
- **Direct versus multihop effects:** distinguishes a one-edge contribution from a dependency transmitted through intermediate nodes.
- **Niche discovery:** groups reproducible local interaction profiles, not scalar raw edge weights.

Do not mix these estimands in one biological claim. Only interpret graph attributions for targets that show reproducible graph-specific predictive gain over the appropriate self/context baseline.

## Goal-Oriented Task Contract

Before substantial implementation or experimentation, write a compact task contract in the workflow README, an experiment plan, or another durable artifact:

- objective and concrete deliverables;
- scientific question and primary hypothesis;
- at least one credible alternative explanation;
- predictions that distinguish the alternatives;
- estimand and permitted claim;
- experimental and observational units;
- primary metric and direction of improvement;
- baselines, positive controls, negative controls, and nulls;
- split strategy and leakage risks;
- acceptance and falsification criteria;
- failure or stop criteria;
- required inputs, compute resources, and expected artifacts;
- verification commands.

If the data or outcome was examined before criteria were fixed, label the analysis exploratory. Do not present it as confirmatory.

### Execution Loop

For each assigned task:

1. Inspect the repository, data availability, prior artifacts, and current compute state.
2. Define completion criteria and identify the cheapest experiment that can discriminate among the hypotheses.
3. Establish or reproduce the baseline.
4. Make the smallest scientifically interpretable change.
5. Run unit or smoke checks before expensive experiments.
6. Diagnose the result against the task contract; do not perform blind retries.
7. Challenge promising results with baselines, ablations, nulls, leakage checks, and alternative explanations.
8. Scale to full, multi-seed or multi-fold experiments only after the pilot is valid.
9. Verify every artifact, summarize the evidence, and update the workflow README.
10. Continue until the acceptance criteria pass, the hypothesis is validly falsified, or a genuine external blocker prevents further progress.

A command exiting successfully is not task completion. Partial code, an unvalidated artifact, or one favorable run is not completion. A rigorous null or negative result can complete a scientific task.

When blocked, record the exact blocker, evidence, attempted solutions, partial artifacts, and the specific external input or state change needed. Difficulty, a long runtime, or an unfavorable result is not a blocker.

## Project Gates

The abstract program is:

0. define estimands and demonstrate recovery on synthetic or semi-synthetic ground truth;
1. complete data, segmentation, geometry, and graph QC;
2. demonstrate patient-held-out graph-specific predictive gain over strong baselines;
3. freeze the successful model and extract predefined readouts;
4. establish stability, faithfulness, and realistic-null calibration;
5. replicate a locked candidate set in independent patients or cohorts, then assess orthogonal molecular support;
6. seek perturbational evidence for a small set of actionable candidates.

Each gate needs documented go/no-go criteria. Do not weaken a gate after viewing the result. If a gate fails, report the failure and either test a stated new hypothesis or stop that branch.

## Workflow Contract

Each `experiments/campaigns/<name>/` represents one bounded hypothesis, model concept, ablation, benchmark, null, or validation experiment. Avoid monolithic campaigns that accumulate unrelated ideas.

Each campaign README must include:

- phase: planned, pilot, full, audited, or complete;
- outcome: pending, supported, negative, inconclusive, or blocked;
- scientific question, falsifiable hypothesis, and alternatives;
- precise estimand and maximum defensible claim;
- input paths, versions, and upstream dependencies;
- allowed model inputs and prohibited leakage fields;
- experimental unit and split definition;
- baselines, controls, nulls, and ablations;
- entry points and exact smoke/full commands when those profiles are meaningful;
- expected outputs and run order;
- go/no-go and completion criteria;
- GPU and resource plan;
- current results, negative findings, limitations, and next discriminating test.

Cross-workflow dependencies must use declared artifacts identified by an immutable run ID, configuration, checksum or manifest, and provenance record; Git tracking is not required. Do not import undocumented intermediate files or manually choose a favorable upstream run.

Keep active generated files under `scratch/active_runs/` and finalized run bundles
under `artifacts/runs/`. Cross-run reports belong under `reports/`; historical
workflow outputs remain immutable under `artifacts/legacy_runs/`.

## Experimental Design Requirements

### Splits and Independent Units

- Define the intended generalization unit before creating a split.
- Patients or independent samples are normally the biological replicates; cells from one patient are not independent patient-level replication.
- Use patient- or sample-grouped splits for generalization claims. Random cell splits are debugging or transductive reconstruction tests only.
- Use spatial-block and leave-region-out tests where local copying is a plausible shortcut.
- Make splits before data-dependent preprocessing. Fit normalization, imputation, feature selection, dimensionality reduction, batch correction, graph construction choices, and hyperparameter selection using training data only when the claim is inductive.
- Use nested or group-aware validation for tuning. Keep the final test cohort sealed until modeling and interpretation choices are locked.
- State whether evaluation is inductive or transductive. Held-out graphs must not influence training topology or learned summaries in an inductive claim.

### Baselines, Controls, and Nulls

Compare complex graph methods under the same split and a fair compute or tuning budget with:

- global, patient, and cell-type means;
- cell-autonomous and morphology models;
- broad spatial-field and spatial-smoothing models;
- neighborhood-composition models;
- non-spatial learned models;
- fixed-edge graph models without attention or learned edge features;
- appropriate established spatial communication methods when making communication claims.

Use mechanism-breaking controls appropriate to the claim:

- cell-type-, patient-, and compartment-stratified expression permutation;
- distant same-type neighbor substitution;
- degree- and distance-preserving graph rewiring;
- coordinate permutation within compartments;
- segmentation and coordinate perturbation;
- parameter or model randomization;
- planted semi-synthetic interactions and co-localization-only decoys.

Include positive controls where feasible to show the pipeline can recover a signal. A null should preserve relevant confounders while breaking the proposed mechanism; an unrealistic global shuffle is rarely sufficient.

Where panel coverage and independence permit, prespecify gastric control families:

- TLS-associated immune signaling as a positive-control family, without supplying the held-out relationship as a model prior;
- macrophage-fibroblast and epithelial-stromal relationships as discovery families;
- broad tumor-versus-normal metabolic fields as confounders that regional context should absorb;
- candidate direct edges across lumens, tissue tears, or anatomical barriers as negative controls.

### Prediction Evaluation

- Match the likelihood and loss to the platform and modeled scale; justify count versus transformed-expression assumptions.
- Report negative log-likelihood or deviance, gene- and cell-wise Pearson and Spearman correlation where appropriate, calibration or predictive intervals, and patient-level uncertainty.
- Stratify results by abundance and dispersion, context-sensitive signaling genes, rare states, spatial boundaries, and spatial autocorrelation.
- Compare every metric with the prespecified baselines and show the distribution across patients, folds, and seeds.
- Aggregate reconstruction performance alone cannot pass the predictive benchmark gate.

### Statistical Reasoning

- Base power and uncertainty on independent biological units, not raw cell count.
- Prespecify a scientifically meaningful effect where possible.
- Report effect sizes, uncertainty intervals, patient prevalence, and heterogeneity, not only p-values.
- Correct the relevant multiple-testing family and document how it was defined.
- Never report only the best seed, fold, graph, threshold, patient subset, or metric.
- Show variation across patients or cores, folds, seeds, graph choices, and relevant preprocessing choices.
- Do not interpret failure to reject a null as proof of no effect, especially with few biological replicates.
- Prefer program-level screening before gene-level follow-up to reduce multiplicity and instability among correlated genes.

HGD and true Normal currently have one documented core or sample each. Verify this after data import. Until a new data manifest shows adequate replication, treat them as descriptive groups rather than the main basis for model selection, candidate ranking, progression claims, or population-level inference.

## Graph and Model Guardrails

- The preferred primary model explicitly separates cell-autonomous, broad regional-context, and local-interaction contributions.
- Use sparse, multiscale, mechanism-specific graphs: contact, local paracrine, regional context, and optional structural relations.
- Primary edge attributes should include measured or computed distance, contact, geometry, and tissue-barrier information where available.
- Do not interpret long-range or barrier-crossing edges as direct interaction without mechanism-specific evidence.
- Do not densify a graph merely to represent long-range context; use explicit context channels or multiscale structures and audit oversmoothing, oversquashing, and boundary mixing.
- Candidate proximity edges may be symmetric, but sender-to-receiver messages must be directionally interpretable.
- Separate direct one-hop effects, multihop relays, and regional context.
- Use shared functions of measured biological, morphological, and geometric edge attributes.
- Do not use unrestricted trainable per-node or per-edge identifiers in the primary inductive model. If studied, label them as transductive sample-specific variables and test on unseen graphs.
- Prefer signed, gene- or program-specific message contributions and biologically bounded finite differences over raw attention.
- Keep perturbations within plausible source-cell or program distributions. Avoid unrealistic one-gene counterfactuals for strongly correlated programs.
- Treat segmentation spillover as a primary alternative explanation for short-range cross-cell dependencies.
- A flexible predictor and a constrained interpretable model may be trained in parallel; agreement is stronger evidence than interpretation from one unconstrained model.

## Data and Leakage Policy

- Read `docs/legacy_data_guide.md` before accessing local data. Treat its key, join,
  clinical-label, streaming, and known-exception guidance as part of the input
  contract.
- Expected local layout is `data/raw/`, `data/processed/`, and `data/clinical/`.
- Treat `data/raw/`, `data/clinical/`, and `data.tar.gz` as immutable shared inputs. Never rewrite or delete them unless the user explicitly requests the exact action.
- Raw CosMx inputs may use the nested form `data/raw/<filename>/<filename>`; preserve support for it.
- Keep local data untracked. Do not expose patient-level or restricted data in logs, reports, commits, or external services.
- Exclude technical control probes such as `Negative*` and `SystemControl*` from biological expression features, while retaining them when needed for technical QC.
- CosMx is a targeted panel: subtype inference is limited by panel coverage, classic gastric MUC/TFF lineage markers may be incomplete or absent, and marker-derived labels are not ground truth.
- Vendor cell types, clusters, neighborhoods, niches, disease stage, core, donor, slide, posterior probabilities, and target-derived annotations are interpretation or stratification metadata unless a workflow explicitly justifies them as model inputs.
- A target-cell label inferred from its hidden expression is leakage in strict masking experiments.
- Do not construct a strict masking graph with the hidden target expression.
- Do not provide a fully masked target's expression-derived library size unless the estimand explicitly treats it as available.
- Audit preprocessing, graph construction, feature selection, batch correction, segmentation, and model selection for target or test-set leakage.
- Never validate a discovery using the same features, labels, priors, or annotations that created it without calling the analysis circular.

## GPU and Compute Policy

Use GPU capacity aggressively for useful, scientifically independent work.

Obey scheduler, allocation, and other-user boundaries. Use only devices that are safely available to this task; never kill, preempt, or displace an existing GPU process merely to increase utilization.

Before a long run:

1. Inspect all devices with `nvidia-smi`, including free memory, active processes, utilization, and topology where relevant.
2. Verify framework visibility, device names, CUDA versions, and memory from the actual environment.
3. Run a representative pilot to measure peak VRAM, throughput, host-memory use, and I/O behavior.
4. Record the proposed allocation of candidates, seeds, folds, nulls, patients, or compartments to devices.

For full experiments:

- Keep all safely available GPUs busy when independent jobs can run without changing scientific results.
- Size batches, graph partitions, precision, caching, and job concurrency to use most available VRAM while retaining a measured safety margin for peak variation.
- If one job cannot use a device efficiently, pack independent jobs on that GPU when measured memory allows it.
- Parallelize seeds, folds, graph variants, candidates, and null repetitions before inventing scientifically unnecessary model complexity.
- Use mixed precision, compilation, gradient accumulation, or distributed training only after verifying numerical equivalence appropriate to the experiment.
- Separate every job's output directory, config snapshot, seed, and log. Never allow concurrent writes to one artifact.
- Monitor useful compute utilization, throughput, thermals or throttling, OOMs, and peak allocated and reserved VRAM. Memory reservation alone is not useful utilization.
- Do not sacrifice controls, deterministic splits, required seeds, or result validity merely to increase utilization.
- Recover failed jobs from isolated checkpoints when valid; do not silently omit them.

Every full run must report GPU model, device IDs, CUDA and framework versions, parallel schedule, runtime, peak VRAM, and failures. CPU-only execution of a GPU-capable full experiment requires a documented reason.

## Reproducibility and Provenance

Every full or conclusion-bearing run must preserve:

- immutable primary run ID, preferred semantic alias, category, start and end
  time, and status;
- exact command and working directory;
- Git commit plus a record of relevant uncommitted changes;
- immutable configuration and all hyperparameters;
- input paths, versions, and checksums or a data manifest;
- split IDs and experimental-unit definition;
- all random seeds and known nondeterministic operations;
- dependency, operating-system, framework, CUDA, and hardware versions;
- logs, runtime, resource use, metrics, and artifact paths;
- indexed best-checkpoint identity, checksum, monitored metric, epoch,
  verification result, and retention class;
- failures, excluded runs, and reasons;
- a concise conclusion tied to the prespecified criteria.

A fresh process should be able to reproduce the result from documented inputs. Prefer configuration-driven entry points and restartable workflows. Full, locked outputs are canonical. Bulky pilot outputs may be removed only after retaining their configuration, metrics, logs or summary, and the decision they informed.

## Code, Tests, and Validation

- Start with the smallest relevant test or smoke profile.
- Add or update tests for changed parsing, graph construction, masking, splitting, metrics, and artifact contracts.
- Use synthetic fixtures for leakage, permutation, graph, and recovery tests where possible.
- Validate shapes, node and edge alignment, split disjointness, finite values, output schemas, and deterministic behavior where promised.
- After changing an entry point, config, input contract, output location, or run order, update the workflow README in the same task.
- Do not launch a costly full run when a pilot already violates correctness or acceptance checks.
- Do not overwrite canonical outputs without preserving their configuration and provenance.

## Interpretation and Reporting

Use this evidence hierarchy:

1. Database, pathway, literature, or known-marker overlap is annotation or a positive control, not independent validation.
2. Replication in independent patients or cohorts establishes observational reproducibility.
3. Orthogonal protein, imaging, metabolomic, or related measurements provide separate molecular support.
4. A controlled perturbation of the proposed sender, receptor, edge, or receiver pathway is required before a causal claim.

For every major result, state:

1. the question;
2. the observed result;
3. the strongest alternative explanation;
4. the controls that address it;
5. the remaining uncertainty;
6. the maximum defensible claim.

Every reported interaction should include source and receiver types/programs, spatial scale, predictive effect, graph-specific gain, seed stability, graph and segmentation robustness, patient replication, faithfulness, null calibration, external evidence, perturbation status, and final evidence designation.

Report stable, faithful, and biologically supported evidence as separate dimensions. Do not collapse them into one opaque score.

Keep reports concise and figure-led. Do not dump full dataframes into the narrative. Store complete cross-run tables under `reports/tables/` or the owning campaign report directory, with run IDs and provenance. If HTML reports are produced, make them portable single files with embedded `data:` images, semantic structure, readable contrast, captions, alt text, and print-safe styling.

## Definition of Done

Do not declare a task complete until:

- the requested deliverables exist;
- the task contract's acceptance checks pass, or its hypothesis has been validly falsified;
- relevant automated and smoke checks pass;
- full runs required by the task have completed across the prescribed seeds, folds, patients, graphs, or nulls;
- outputs and logs are complete, isolated, and reproducible;
- leakage, pseudoreplication, confounding, and circularity have been audited;
- claims match the study design and evidence;
- uncertainty, limitations, negative results, and failed runs are reported;
- the relevant README contains exact reproduction commands and current status;
- `git status --short` has been reviewed and unrelated changes remain untouched.

Finish the scientific task, not merely the code.
