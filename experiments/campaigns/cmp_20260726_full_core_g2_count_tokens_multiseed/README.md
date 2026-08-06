# Full-Core G2 Raw-Count Token Prediction Across Seeds

## Status

- Phase: completed
- Outcome: larger width did not improve token accuracy, the original
  Jacobian gate failed, and the user-requested post-hoc relaxed categorical
  sensitivities did not meet the operational match criteria
- Campaign: `cmp_20260726_full_core_g2_count_tokens_multiseed`
- Scope: exploratory held-in token reconstruction in one legacy true-Normal
  core

This campaign changes the earlier continuous-expression regression task into
categorical raw-count prediction so that exact accuracy has a valid percentage
interpretation. Every cell remains part of fitting. Results are transductive
held-in reconstruction, not validation/test accuracy or patient
generalization.

## Results

The resource pilot and all six conclusion-bearing runs completed successfully.
Every science run finished 200 finite epochs, and every finalized bundle
passed checksum and registry verification. Increasing parameters from
7,062,880 to 18,834,784 (2.67-fold) did not improve any aggregate accuracy
metric:

| Variant | Exact % ± seed SD | Balanced % ± seed SD | Nonzero % ± seed SD | Cross-entropy | Training s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Current width | 91.7799 ± 0.0045 | 28.1677 ± 0.1606 | 1.5314 ± 0.0777 | 0.326539 | 1,535.7 | 8.284 GiB |
| Wider | 91.7684 ± 0.0220 | 28.1051 ± 0.6304 | 1.5024 ± 0.3055 | 0.332731 | 2,642.8 | 8.711 GiB |

Mean wider-minus-current differences were −0.0115 percentage points exact,
−0.0626 points balanced, and −0.0290 points nonzero. Cross-entropy increased
by 0.006192. Only seed 0 improved exact and balanced accuracy; seeds 1 and 2
regressed on both. Every locked H1-width criterion therefore failed.

The high exact percentage is a class-imbalance shortcut, not successful token
reconstruction. The all-zero reference is 91.6968% exact, while the all-fit
per-gene modal reference is 91.7253% exact, 28.3004% balanced, and 1.6031%
nonzero. Both trained variants recalled token 0 at about 99.95%, recalled
tokens 1 and 2 at exactly 0%, and recalled token 3 at only about 12.5%. The
wider model was worse than the modal reference on balanced and nonzero
accuracy.

The wider seed accuracies were 91.7887%, 91.7714%, and 91.7450%. None was
strictly above 95%, so the relaxed categorical Jacobian branch was skipped
exactly as prespecified. No prespecified Jacobian was computed or interpreted;
the separate analysis below was performed only after the user explicitly
overrode that stopping rule.

## Post-hoc Jacobian override

After the completed accuracy comparison and failed 95% gate were reported, the
user explicitly requested that the models still be compared. This reopened
only the relaxed categorical-sensitivity analysis. It is a post-hoc,
exploratory override of the stopping rule, not the prespecified conditional
branch, and cannot rescue H1-width or convert the held-in accuracy into
generalization evidence.

The analysis used the numerical protocol locked below: centered output
logits, derivatives with respect to relaxed observed one-hot channels,
simplex-tangent projection, the same three whole-node masks and common
Rademacher probes, identical-checkpoint and paired random-initialization
controls, and fail-closed numerical checks. Integer token IDs themselves are
not treated as differentiable quantities.

The post-hoc execution is versioned separately from the immutable training
bundles. Mutable shard output belongs under
`scratch/active_runs/posthoc_g2_token_relaxed_categorical_sensitivity_v1/`;
the verified, self-checksummed cross-run result belongs under
`reports/analyses/full_core_g2_count_tokens_multiseed/posthoc_relaxed_categorical_sensitivity_v1/`.
This is one joint six-checkpoint estimand, so it is not represented as six
independent single-run registry evaluations.

The one-probe wider-model FP32 resource pilot passed and was explicitly
checksum-reviewed. The independently recomputed identical-checkpoint control
also passed exactly: cosine 1, relative discrepancy 0, and norm ratio 1. The
complete workload then finished without numerical, checksum, isolation, or
thermal failures. Strict aggregation verified 30 shards and 2,304 full-graph
VJPs.

The trained cross-width sensitivities did not operationally match:

| Seed | Cosine [95% technical CI] | Relative discrepancy [95% technical CI] | Norm ratio wider/current [95% technical CI] | Match |
|---:|---:|---:|---:|:---:|
| 0 | 0.5522 [0.5436, 0.5606] | 0.9638 [0.9547, 0.9728] | 0.8332 [0.8275, 0.8385] | no |
| 1 | 0.5249 [0.5198, 0.5296] | 1.0424 [1.0375, 1.0475] | 1.4437 [1.4353, 1.4522] | no |
| 2 | 0.5374 [0.5314, 0.5432] | 1.1113 [1.1016, 1.1204] | 0.5772 [0.5703, 0.5841] | no |

All seeds failed the locked cosine and discrepancy thresholds; seeds 1 and 2
also failed the scale criteria. The eight paired random-initialization
controls had near-zero cosine, with a maximum 95% interval upper bound of
0.000691, so the trained models share more local sensitivity structure than
untrained models. That moderate alignment is still far below a match and is
not evidence of useful prediction. The 95% intervals quantify only technical
mask and randomized-probe variability, not biological, patient, or
training-seed uncertainty.

Artifacts:

- Comparison report:
  `reports/analyses/full_core_g2_count_tokens_multiseed/comparison/report.md`
- Machine-readable comparison and CSV tables:
  `reports/analyses/full_core_g2_count_tokens_multiseed/comparison/`
- Post-hoc sensitivity report and self-checksummed result:
  `reports/analyses/full_core_g2_count_tokens_multiseed/posthoc_relaxed_categorical_sensitivity_v1/`
- Pilot:
  `artifacts/runs/2026/07/r_20260726T134226Z_d86b9849_s000_f00_a01_371b97a1`
- Current width, seeds 0-2:
  `r_20260726T134609Z_6dda5587_s000_f00_a01_df92a5e0`,
  `r_20260726T134614Z_6dda5587_s001_f00_a01_e6fc4986`, and
  `r_20260726T134619Z_6dda5587_s002_f00_a01_0156cb89`
- Wider, seeds 0-2:
  `r_20260726T134635Z_56a25fc4_s000_f00_a01_c647ac60`,
  `r_20260726T134625Z_56a25fc4_s001_f00_a01_a6ab3ca5`, and
  `r_20260726T134630Z_56a25fc4_s002_f00_a01_9833c850`

The authoritative registry contains exactly these seven campaign jobs: one
pilot and six science runs, all completed, with no failure, cancellation, or
retry.

## Task contract

### Objective, question, and hypotheses

Train the current-width and wider G2 token models under three paired seeds,
report exact and class-balanced masked-token accuracy in percent, and apply the
requested 95% gate before any Jacobian comparison.

Question:

> Holding the raw-count tokens, graph, masks, optimizer, and fixed epoch budget
> constant, does the larger G2 improve held-in masked-token prediction across
> seeds?

- H1-width: the wider model improves mean whole-node exact accuracy by at
  least 0.5 percentage points and mean balanced accuracy by at least 1.0
  percentage point, with no paired seed decreasing on either metric.
- A1-zero shortcut: high exact accuracy is largely obtained by predicting the
  dominant zero token.
- A2-information limit: whole-node token prediction is limited by available
  neighborhood and morphology information rather than model width.
- A3-optimization: the wider categorical model is harder to optimize under the
  unchanged learning rate and 200-epoch budget.
- A4-seed instability: an aggregate difference is driven by one initialization.

Predictions distinguishing these explanations:

- Genuine token prediction must improve over the all-zero and per-gene modal
  baselines and raise balanced and nonzero-only accuracy, not only exact
  accuracy.
- A pure zero shortcut can exceed 90% exact accuracy while remaining near
  chance on balanced accuracy and near 0% on nonzero tokens.
- A width effect must have the locked paired-seed direction above; a mixed
  direction is treated as unstable.

### Tokenization and model

Each biological probe count is mapped without fitted thresholds:

| Output token | Raw count | Label |
|---:|---:|---|
| 0 | 0 | `zero` |
| 1 | 1 | `one` |
| 2 | 2 | `two` |
| 3 | 3 or more | `three_or_more` |

The input has an additional mask-only token that is never an output class.
The encoder uses gene-by-token indicator projections. Token IDs are not
treated as continuous scalar values. Masked values are replaced by the
mask-only channel inside the model, and cross-entropy is computed only at
masked entries.

The vocabulary is fixed before training, requires no quantile fitting, retains
the main distinctions supported by this sparse targeted-panel snapshot, and
places the long high-count tail in one class. In the verified
24,245-by-1,000 matrix, token prevalence is 91.7889%, 4.5360%, 2.6792%, and
0.9959%. Every output token occurs for every gene.

Both models retain the exact receiver-partitioned two-layer edge-conditioned
GATv2 backbone. The controlled change remains representation width:

| Variant | Hidden/FFN/decoder | Heads | Output classes |
|---|---:|---:|---:|
| Current-width token G2 | 512 | 4 | 4 per gene |
| Wider token G2 | 1,024 | 8 | 4 per gene |

Parameter counts are measured from the constructed models and recorded rather
than inferred in this plan.

### Estimand, metrics, baselines, and 95% gate

The primary estimand is the paired seed difference in mean held-in whole-node
exact masked-token accuracy across three fixed technical masks. Report all
percentage metrics on a 0-100 scale:

- exact masked-token accuracy;
- balanced accuracy, defined as the unweighted mean recall of output tokens
  0-3;
- nonzero-only exact accuracy over targets in tokens 1-3;
- recall and support for each token;
- unweighted masked categorical cross-entropy.

Locked references are 25% uniform-class chance, empirical-frequency random
chance, an all-zero predictor, and the all-fit per-gene modal predictor. On the
three fixed whole-node masks, the last baseline is approximately
91.7253% exact but only 28.3004% balanced accuracy. These values will be
recomputed from each materialized mask and checksum-verified in every run.

The requested Jacobian branch opens only if every wider-model seed has mean
whole-node exact accuracy strictly above 95.0%. The literal gate is retained,
but passing it would not establish useful prediction because the per-gene
modal baseline is already near 92%.

Token IDs are discrete, so a derivative with respect to an integer ID is not a
mathematically meaningful input Jacobian. If the gate opens, the predefined
comparison is therefore a **relaxed categorical sensitivity**, not a
continuous-expression Jacobian: differentiate centered output logits for the
masked receiver targets with respect to the observed four-channel one-hot
token indicators, and project input gradients onto the per-gene simplex
tangent space. Use the same fixed masks and common Rademacher output probes for
all models. Report paired cross-width Frobenius cosine, relative Frobenius
discrepancy, and norm ratio, with identical-model and reinitialized controls.
The numerical and operational-match thresholds are those locked in
`cmp_20260726_full_core_g2_larger_multiseed`. If the 95% gate fails, record the
failure and do not compute or interpret this relaxed Jacobian.

The following numerical clarification was locked while all six 200-epoch runs
were still training and before any conclusion-bearing accuracy was available:

- Use exactly the three fixed whole-node masks and 32 probes per mask.
- For probe-aggregated sufficient statistics
  `A2 = ||J_A^T u||²`, `B2 = ||J_B^T u||²`, and
  `AB = <J_A^T u, J_B^T u>`, compute cosine as
  `AB / sqrt(A2*B2)`, relative discrepancy as
  `sqrt(A2+B2-2*AB) / (A2*B2)^(1/4)`, and directed norm ratio as
  `sqrt(B2/A2)`. Non-finite or zero norms invalidate the analysis.
- Orient primary pairs as wider/current. Orient within-width seed pairs from
  lower to higher seed, but compare norm matching through
  `abs(log(norm_ratio))`: a primary pair must be no worse than the maximum
  within-current value. Its cosine must be at least the minimum
  within-current cosine and its discrepancy at most the maximum
  within-current discrepancy.
- The identical control is current-width seed 0 loaded into two independent
  model instances, with all VJPs recomputed. For each random-control seed
  9100-9107, separately instantiate one current-width and one wider model with
  that same paired seed. Their matching RNG prefixes can conservatively raise
  cross-width similarity; these are paired controls, not statistically
  independent width draws. Separation uses the maximum of the eight
  random-control cosine interval upper bounds and the minimum of their
  discrepancy interval lower bounds.
- Derive each probe seed from the first eight little-endian bytes of SHA-256
  over compact canonical JSON
  `[20260726, mask_entry_id, probe_index]`, masked to 63 bits. Generate CPU
  Rademacher values with NumPy `Generator(PCG64(seed))` and record their
  checksums.
- For each of 2,000 deterministic bootstrap replicates, resample the three
  masks with replacement and then resample 32 probes independently within
  each selected mask. Average `A2`, `B2`, and `AB` before deriving each
  statistic. The bootstrap seed uses the same rule with
  `[20260726, "bootstrap"]`.
- Before launching the full conditional workload, run one FP32 probe on wider
  seed 0 and whole-node mask replicate 0. Require finite outputs and at most
  20.5 GiB peak allocated VRAM, record wall time, project the complete 2,304
  VJP workload, and require an explicit launch review; no time cutoff is
  inferred after seeing that pilot.

### Units, data, graph, and leakage

- Experimental and observational unit: one spatial core.
- Execution repeats: paired model seeds `0`, `1`, and `2`.
- Technical repeats: three fixed masks per masking mode within each seed.
- Input: verified `prepared_full_v1`, 24,245 cells and 1,000 biological probes;
  technical controls remain excluded.
- Always-visible covariates: the same 22 morphology/imaging measurements.
- Graph: the same exact mutual k=1,000 graph, 650 µm non-truncating guard,
  21,029,944 directed edges, and 17 geometry features.
- Prohibited inputs: identifiers, coordinates, RNA-derived QC, library size,
  vendor labels, and the hidden target token.

All nodes, topology, preprocessing statistics, token prevalences, and
per-gene baseline modes are fitted or measured on the same core. Fixed fresh
masks do not create a validation/test set. Cells and masks are technical units,
not independent biological replicates.

### Training, resource pilot, and stopping

Both variants use paired epoch masks, unweighted categorical cross-entropy,
AdamW at `3e-4`, weight decay `1e-4`, gradient clipping at `1.0`, mixed
precision, deterministic algorithms, and 200 fixed epochs. The final epoch is
canonical; there is no validation selection, early stopping, or best-seed
selection.

Before the six full runs, the wider model must finish a two-epoch resource
pilot with finite loss and gradients, peak allocated VRAM no greater than
20.5 GiB, and projected 200-epoch time no greater than six hours. Full runs use
GPUs 1, 2, and 3 for current width and GPUs 7, 5, and 6 for wider seeds 0, 1,
and 2. GPU 0 belongs to another campaign; GPU 4 is quarantined after prior
thermal throttling.

Stop on tokenizer, graph, or checksum mismatch; target-token leakage;
non-finite loss or gradients; repeated OOM after changing only execution chunk
size in a separately reviewed resource variant; insufficient disk; or artifact
verification failure. Do not silently change vocabulary, loss weighting,
width, graph, masks, seeds, or epoch budget.

### Required outputs and verification

1. Tokenizer distribution, checksum, mask-token contract, and baseline audit.
2. One verified two-epoch wider resource-pilot bundle.
3. Six verified conclusion-eligible bundles: two widths by three seeds.
4. Per-mask, per-seed, and aggregate exact/balanced/nonzero accuracy,
   cross-entropy, token recalls, baselines, runtime, peak VRAM, parameter
   counts, convergence, and failures.
5. A concise comparison report plus machine-readable tables and gate record.
6. Relaxed categorical Jacobian artifacts only if the locked 95% gate opens.

Verification must include focused tests, the full practical test suite,
`python -m spatial_benchmark doctor`, and finalized-bundle verification.

### Execution and verification record

Execution completed on 2026-07-26 UTC:

| Role | Seed | GPU | Job | Run | State |
|---|---:|---:|---|---|---|
| Wider resource pilot | 0 | 7 | `q_a18ca3596e5d1e6803a1` | `r_20260726T134226Z_d86b9849_s000_f00_a01_371b97a1` | completed |
| Current width | 0 | 1 | `q_d299768b790f6ea47877` | `r_20260726T134609Z_6dda5587_s000_f00_a01_df92a5e0` | completed |
| Current width | 1 | 2 | `q_6862e0288fbd26a7d336` | `r_20260726T134614Z_6dda5587_s001_f00_a01_e6fc4986` | completed |
| Current width | 2 | 3 | `q_c03a0b3ee53bf1a6cc40` | `r_20260726T134619Z_6dda5587_s002_f00_a01_0156cb89` | completed |
| Wider | 0 | 7 | `q_d85ccae7f2416d26ffb0` | `r_20260726T134635Z_56a25fc4_s000_f00_a01_c647ac60` | completed |
| Wider | 1 | 5 | `q_53a356b10dfefcb19b3d` | `r_20260726T134625Z_56a25fc4_s001_f00_a01_a6ab3ca5` | completed |
| Wider | 2 | 6 | `q_eb3e0aa60c2b64cb9db8` | `r_20260726T134630Z_56a25fc4_s002_f00_a01_9833c850` | completed |

The pilot used 8.703 GiB peak allocated VRAM and projected 0.768 hours of
training, passing both resource gates. All six conclusion-bearing jobs then
completed without retry, OOM, non-finite loss, or thermal slowdown; the
maximum observed job temperature was 81 °C. Temporary supervisor workers were
removed after completion.

Completion verification produced:

- strict comparison status `complete`;
- registry reconciliation `passed` for exactly 7/7 completed campaign jobs
  with no failure category or last error;
- resource pilot `passed`, H1-width `failed`, and relaxed Jacobian
  `skipped`;
- full practical suite: 348 passed and two expected optional skips;
- infrastructure doctor: `ok: true`, SQLite integrity `ok`, no issues;
- global artifact verifier: `valid: true`, with no bundle or registry issues;
- source experiment definitions compose exactly to all launched resolved
  configuration snapshots.

The comparison command used all six run paths plus the pilot and read-only
registry:

```bash
PYTHONPATH=src /venv/main/bin/python \
  scripts/analysis/compare_g2_token_multiseed.py \
  --run <current-s0> --run <current-s1> --run <current-s2> \
  --run <wider-s0> --run <wider-s1> --run <wider-s2> \
  --resource-pilot <pilot> \
  --registry state/tracking/bagm.sqlite3 \
  --output reports/analyses/full_core_g2_count_tokens_multiseed/comparison
```

`/workspace` is not backed by a persistent volume on this instance. These
artifacts survive stop/start but not instance recycle or destruction unless
they are copied off-box.

### Post-hoc sensitivity execution record

The frozen post-hoc implementation passed 356 repository tests with one
expected optional dependency skip, the infrastructure doctor, actual
six-bundle provenance preflight, and an independent read-only audit. The
workflow source SHA-256 is
`49f6715f34f86ad3574fac3dcaab1187b57b4d639e6dab78f05e807544fe4459`.

The mandatory wider-seed-0, mask-0, probe-0 FP32 pilot passed on isolated
physical GPU 7. Its reviewed artifact SHA-256 is
`a545ddcdfcfc421bb20b2c39b206c486dd452efa496bcf74cf8c24fea6a46b2b`.
The tangent VJP squared norm was finite and nonzero (`66.3771`), peak allocated
VRAM was 9.321 GiB, and one VJP plus tangent projection took 16.011 seconds.
The locked 2,304-VJP projection is 10.247 total GPU-hours, approximately 3.416
hours across three GPUs before graph/input setup. The pilot leaves 11.179 GiB
of headroom to the 20.5-GiB gate; even the audited 0.6-0.8-GiB extra static
footprint of a six-model trained shard leaves a large margin. The explicit
review therefore authorizes only the next fail-fast stage: the three
identical-checkpoint shards.

All three identical-checkpoint shards then completed 32 common probes without
OOM, non-finite output, or thermal slowdown. Their locked combined review
passed exactly: cosine `1.0` with interval `[1.0, 1.0]`, relative discrepancy
`0.0` with interval `[0.0, 0.0]`, and norm ratio `1.0` with interval
`[1.0, 1.0]`. The explicitly reviewed control artifact SHA-256 is
`02f496eb4e9e126a211f31dc9064b3bdba968d8833a1d81d6d63265bbf213158`.
This numerical-control pass authorizes the remaining trained-model and
randomized-control shards; it does not alter the failed accuracy gate.
