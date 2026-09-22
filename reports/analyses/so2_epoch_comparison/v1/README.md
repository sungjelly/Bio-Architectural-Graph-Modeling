# SO2 epoch-by-epoch training comparison

Phase: complete. Outcome: inconclusive for model superiority. This is an exploratory descriptive comparison
of existing training records requested after the final-metric comparison was
examined. No new training, checkpoint selection, or scientific gate is proposed.

## Task contract

Objective: align every recorded global epoch of the geometry-modulated,
original four-block, and recurrent shared-block SO2 models; deliver complete
CSV tables, an annotated loss figure, and checksummed provenance. The question
is whether geometry has consistently lower training loss at matched epochs.
The working hypothesis predicts sustained negative geometry-minus-original
differences. Credible alternatives are stochastic training fluctuations and
unequal training duration; reversals and differences near zero support those
alternatives. These descriptive patterns have no prespecified significance or
minimum effect threshold and cannot establish model superiority.

The estimand is the recorded equal-core mean masked Huber training objective
on standardized log1p expression, averaged over ten mask views per core and
fourteen fitted SO2 cores per global epoch. Lower is better. Each epoch has
seven optimizer updates. Epochs and cores are repeated observations, not
independent patient replicates. All models use seed 0, the same fitted cohort,
graph, preprocessing, masking protocol, and optimizer. There is no test split.
Same-cell observed expression and morphology remain available. No
generalization, graph-specific gain, mechanism, or causal claim is supported.

Comparator handling: original epochs 1–175 come from the original run;
epochs 176–300 come from its successful continuation. Their overlapping
histories must match exactly in scientific metrics and are not counted as
independent runs. Recurrent epochs 1–175 are included with a visible failed
finalization status. The failed duplicate continuation is inventoried but
excluded from aggregate comparisons to avoid double counting. There are no
new positive controls, negative controls, or nulls in this reporting task.

Per-epoch fixed-mask evaluation MAE, MSE, and R² do not exist in these logs.
Training losses are collected with training-time masks and dropout while
parameters update; they are not post-epoch fixed-checkpoint evaluations.
Compare all raw epochs and fixed nonoverlapping 25-epoch blocks. Plot a
trailing 25-epoch mean with a full-window requirement solely as a visual aid.
Do not treat epoch win counts as a significance test or smooth missing epochs.

Inputs: immutable run bundles listed in `provenance.json`, their registered
statuses, resolved configs, artifact checksum manifests, and epoch CSVs.
No raw, clinical, or row-level cell data are read. Resource plan: CPU-only
small-table analysis and standard Matplotlib rendering; no GPU computation
is needed. Output stays in this versioned cross-run report directory. These
training diagnostics are not promoted as a verified curated scientific result.

Acceptance: verify source hashes; contiguous epochs; finite losses; equal-core
means; identical overlapping original history; shared training mask entry
counts, configs and update counts; correct missingness beyond each run's
endpoint; and output checksums. Stop on any mismatch rather than silently
dropping records. Retain run IDs and status in numeric exports. Completion is
the full epoch table and checked figure, regardless of comparative direction.

## Reproduction

From the repository root:

```bash
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_epoch_comparison/v1/compare.py
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_epoch_comparison/v1/compare.py --check
```

Generation refuses to overwrite a completed report. Use `--output-dir` with
a new directory beneath the configured report root for a new version.
The generated `comparison.md` contains the findings and links to all outputs.

## Completed comparison

The report contains all 300 original/continuation epochs, 200 geometry epochs,
175 recurrent epochs, and 9,450 per-core/model/epoch loss records. Geometry has
lower training Huber in 71/200 shared epochs against the original trajectory
and 167/175 against recurrent. Source hashes, exact inherited original history,
all 28,000 geometry and 24,500 recurrent paired mask checksums, core scheduling,
equal-core means, missing endpoints and output hashes pass. An independent
read-only audit reproduced the comparison counts and checked the methodology.
No source bundle was changed or new model trained. See
[the figure and comparison](comparison.md) and [verification](verification.json).
