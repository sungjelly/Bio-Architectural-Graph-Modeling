# Attention τ and β diagnostic

Objective: show learned per-head τ and geometry-dependent β in the completed
seed-0 run `r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf`.
This is an exploratory, descriptive checkpoint inspection, not new training.

Question: how large are τ and β, and how does β's variation across each receiver's
neighbors compare with the actual modulated Q–K content score? A larger τ alone
does not imply a larger realized content score. Receiver-constant offsets are
another alternative explanation for large raw scores; subtracting each
receiver/head's mean removes offsets that cancel under softmax.

Protocol fixed before extraction: inspect all 32 learned scales; execute all
14 original full-core graphs in eval mode, FP32 without AMP, with one fixed
uniform-per-cell masking realization per core from the established diagnostic
mask helper. No neighbor sampling, retraining, checkpoint selection, or
clinical-label input. Stream every edge at every block. Report β mean/RMS and
actual content RMS, and their receiver-centered RMS. Weight edges equally within
each core and cores equally in cohort summaries. Cores are descriptive strata,
not asserted independent biological replicates. Seeds: trained model seed 0;
one documented mask seed per core. No generalization or biological claims.

Validation: source checkpoint and input file hashes, strict state loading,
parameter count, finite scores, content + β identity, softmax shift invariance,
exact edge coverage, and pilot public-forward agreement. Stop on any mismatch.
One small core is the resource pilot; then partition remaining cores over free
GPUs, at most one core per GPU. Source model and data stay immutable.

Deliverables: a PNG/PDF figure, per-core/head and aggregate CSV/JSON statistics,
source provenance and runtime/VRAM receipts, and this reproduction script.

Run from the project root with `PYTHONPATH=src /venv/main/bin/python
reports/analyses/so2_attention_tau_beta/v1/analyze.py --cores 21 --device cuda:0
--pilot`. Remaining cores can be split between disjoint GPU workers using
`--cores`. Run `--summarize` only after every core receipt exists.

## Completed results

Score: `s_ijh = tau_h * sum_l(g_ijhl * qhat_ihl * khat_jhl) + beta_ijh`.

Inspected all 246,063 cells and 55,980,536 directed edges in 14 cores, at every one of four blocks. No edges were sampled. The following RMS values average squared scores equally over edges within each core, then equally over cores and heads. τ is a multiplier; β is an additive, geometry-dependent score bounded in [-1, 1].

| Block | Mean τ | β RMS | Actual content RMS | Centered β RMS | Centered content RMS | Centered β/content |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.9133 | 0.8927 | 0.7698 | 0.4581 | 0.5483 | 0.8354 |
| 2 | 1.9201 | 0.8506 | 0.7526 | 0.6191 | 0.5862 | 1.0561 |
| 3 | 1.9241 | 0.8095 | 0.7306 | 0.4794 | 0.5825 | 0.8230 |
| 4 | 1.9321 | 0.8477 | 0.7681 | 0.5544 | 0.6506 | 0.8521 |

τ ranges from 1.8857 to 1.9834 across all 32 heads (initial value 1.8856). β contributes score variation of comparable magnitude to the realized content term. Its mean signed value, extrema, saturation fraction, and channel covariance/correlation are available in the full CSVs. Geometry also enters the modulation g, so this decomposition does not isolate all geometry effects. These magnitudes describe model scores and do not establish predictive necessity or biological importance.

Centering subtracts the incoming-neighbor mean separately for each receiver and head. These offsets cancel in softmax. RMS ratios measure channel spread, not a causal contribution or a fraction of attention explained. Content depends on the fixed expression mask; τ and β do not. This one-seed, one-mask-per-core diagnostic does not quantify mask/seed variability.

All 14 extractions passed finite-score, edge-coverage, channel-identity, softmax-shift and source-checksum checks. Three public-forward replays (cores 23, 20, 21) passed at atol=rtol=1e-6. Maximum replay difference: 4.77e-07. Peak device allocation: 0.896 GiB; sum of per-core elapsed times across four concurrent GPUs: 464.7 seconds. Hardware/software and masks are recorded per core. The initial pre-extraction inference-mode failure and correction are retained in execution_plan.json.

Source audit confirms the attention implementation matches training exactly; base-model changes add only optional output collection. Executed-source snapshots preserve the versions used for extraction. Figure rendering subsequently changed only plot contrast/limits and adds separate render provenance.

### Reproduction commands

Each extraction refuses to overwrite an existing completed core receipt. Use a new version directory for a rerun. Commands originally executed (the four workers run concurrently):

```bash
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_attention_tau_beta/v1/analyze.py --cores 23 20 21 --device cuda:0 --pilot
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_attention_tau_beta/v1/analyze.py --cores 15 25 22 --device cuda:1
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_attention_tau_beta/v1/analyze.py --cores 27 26 28 17 --device cuda:2
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_attention_tau_beta/v1/analyze.py --cores 19 16 24 18 --device cuda:3
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_attention_tau_beta/v1/analyze.py --summarize
```

![Attention scales and bias](tau_beta.png)

Files: tau_beta.png, tau_beta.pdf, per_head.csv, per_core_head.csv, summary.json, core_*.json, source_semantics_audit.json, verification.json, and figure_provenance.json.
