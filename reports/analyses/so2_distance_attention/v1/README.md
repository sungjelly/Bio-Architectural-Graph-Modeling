# Distance versus attention

Exploratory diagnostic of epoch-200 SO2 model
`r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf` (seed 0), continuing the
τ/β and g inspections. User hypothesis: attention falls as inverse-square
physical distance. Objective: plot the observed relationship and quantify
agreement with that hypothesis without treating attention as molecular flux.

Protocol fixed before examining distance-conditioned outputs: full 14-core
graphs, every edge and all 4 blocks × 8 heads, FP32 eval, same fixed per-cell
mask per core as the earlier τ/β diagnostic. Distances are Euclidean distances
between prepared two-dimensional cell coordinates in μm, checked against
geometry column 69 × 500. This is projected tissue distance, not measured 3D
transport length. Graph domain is (0,500] μm; graph shell quotas and bidirectional
union make the available edge-distance distribution nonuniform.

The raw score is s = content + β and α = softmax(s) over incoming edges.
For α proportional to r^-p, s = receiver/head intercept − p log(r). Regress
receiver-centered s on receiver-centered log(r), giving p = −slope and R².
Edges receive weight 1/number of included neighbors per receiver, then receivers
are equally weighted within each core and cores equally weighted. This is a
pooled within-receiver regression, not the average of individual slopes.
Primary fit: 10–450 μm; sensitivity fit: all positive observed distances. Fit
all heads, report core heterogeneity. No seed or favorable head selection.

Alternative descriptions: uniform weights (zero centered score), freely fit
power exponent, and an exponential distance kernel (centered score linear in
distance). Compare their in-sample centered-score MSE/R² with fixed p=2. These
are descriptive fits, not held-out model selection or a mechanistic law.
The positive control constructs r^-2 weights normalized on the same complete
incoming neighborhoods and must recover p=2 and R²=1. Plot this matched reference
beside degree-scaled attention n_in*α (uniform reference 1), raw α, combined
score, content and β. Fixed 10-μm bins; equal-edge means within each core/bin,
then equal-core means. Binned curves and regression have distinct declared
weighting. Across-core ranges are descriptive, not independent-patient CIs.

Acceptance: checkpoint/source/input hashes; finite scores; exact edge coverage;
positive distances ≤500 μm; coordinate/cache consistency; normalized attention;
content+β identity; synthetic power/exponential recovery; pilot replay agreement
with public model forward. Stop on any failed check. Source files stay immutable.
All GPUs are occupied by an existing four-rank training run, so this bounded
inference diagnostic uses available CPUs; pilot one small core before parallel
CPU workers. No training is stopped, preempted, or modified.

Deliverables: figures (PNG/PDF), complete per-core/head fits, binned curves,
summary, verification, provenance and reproduction code. Allowed conclusion:
descriptive distance association within this trained model/cohort/mask only.
Correlation cannot isolate distance from expression, orientation, graph selection
or segmentation effects and does not establish communication or causality.

## Completed results

Analyzed all 55,980,536 directed edges across 246,063 cells, 14 cores, four blocks and eight heads. Source: frozen SO2 epoch-200 seed-0 checkpoint. The model shows a sharp short-range preference and a broad tail; the inverse-square hypothesis is a poor description of its attention.

| Block | Effective power exponent p | Pooled power-law score R² | Exponential score R² | Fixed inverse-square MSE / flat MSE |
|---|---:|---:|---:|---:|
| 1 | 0.4274 | 0.2348 | 0.1052 | 3.944 |
| 2 | 0.6535 | 0.3794 | 0.2179 | 2.232 |
| 3 | 0.3521 | 0.1426 | 0.0751 | 3.980 |
| 4 | 0.3224 | 0.0905 | 0.0349 | 3.359 |

These exponents summarize centered-score fits over 10–450 μm, pooling cores and heads as specified above. An inverse-square attention kernel would have p=2. The fitted exponent ranges across the 14 block-level core summaries are 0.362–0.495 (block 1), 0.594–0.695 (block 2), 0.300–0.392 (block 3), and 0.244–0.379 (block 4). These ranges are descriptive core heterogeneity, not independent-patient confidence intervals.

Including every positive graph distance gives block exponents 0.4587, 0.6550, 0.3678, and 0.3446, preserving the conclusion. At the individual-head level, primary exponents range from −0.7011 to 0.7757. Four head summaries have negative exponents (attention increasing with distance after receiver centering): B3H4, B3H7, B4H3, and B4H6. Fixed p=2 fits worse than the constant centered-score baseline in every head.

The fitted free power law captures more centered-score variance than a fitted exponential in all four block summaries, but neither provides a complete description. The observed mean curves are not smooth power laws: attention is high in the first roughly 20–30 μm, falls steeply through approximately 50 μm, and then has a broad, relatively flat tail with bumps. The raw score/β component curves have a similar saturating distance profile; these component associations do not isolate geometry causally.

At 15 μm, mean degree-scaled attention is approximately 4.1–4.9 across blocks (uniform attention=1). At 55 μm it is 0.60–0.79; at 105–495 μm the illustrated bin values largely remain around 0.46–0.93. These are pooled bin summaries on the selected graph, not the distance ratio for one receiver. Graph shell quotas and changing receiver normalizers cause even the matched inverse-square reference to bend after pooling. This is why the centered-logit slope test is the primary law diagnostic.

### Physics interpretation

For an ideal steady isotropic 3D point source with diffusion and degradation, concentration is c(r)=Q exp(−r/λ)/(4πDr). Without degradation it scales as 1/r. Differentiating this no-degradation solution using Fick’s law gives net outward radial flux per unit area Q/(4πr²). Thus inverse-square is a flux result under specified assumptions, not a generic diffusion concentration law. [Primary derivation source, Appendix A](https://arxiv.org/html/2507.19341v1#A1). Additional verified sources and derivations are in physics_sources.md.

Neural attention is normalized over incoming neighbors of each receiver and is not a mass-conserving outward molecular flux. Distances here are projected 2D cell-centroid separations, not measured 3D transport paths. Orientation, expression-dependent content, graph selection and segmentation all remain alternative contributors to the observed association. These results characterize one trained model and one fixed mask per core; they do not establish biological transport, a communication mechanism, causal effects, patient generalization, or seed/mask robustness.

### Validation and execution

The independently implemented verifier passed the 896-fit grid, 14-core coverage, input receipt-to-manifest hashes, preserved/current source checks, same-graph p=2 positive controls, attention normalization, distance/cache consistency, bin sums, all seven curve channels, CSVs and reaggregated fits. The pilot public-forward replay difference was zero. Maximum attention-sum error was 1.43e−6; maximum coordinate/cache distance difference was 1.49e−5 μm; independently recomputed fits agreed within 1.14e−12. Large inputs were hashed and checked during extraction; the independent final audit checked those receipts against frozen manifests without another full-byte reread.

Execution used four low-priority CPU workers, eight threads each, because all GPUs were allocated to the ongoing SO1 training. Sum of per-core elapsed times was 2076.9 seconds (concurrent worker times overlap). Source training and artifacts were not modified. No extraction failures or omitted cores.

### Reproduction

From project root, run the pilot, then the four worker commands concurrently on available CPUs. Extraction refuses to overwrite completed core receipts.

```bash
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_distance_attention/v1/analyze.py --cores 21 --threads 8 --pilot
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_distance_attention/v1/analyze.py --cores 23 20 --threads 8
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_distance_attention/v1/analyze.py --cores 15 25 22 --threads 8
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_distance_attention/v1/analyze.py --cores 27 26 28 17 --threads 8
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_distance_attention/v1/analyze.py --cores 19 16 24 18 --threads 8
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_distance_attention/v1/analyze.py --summarize
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_distance_attention/v1/render.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_distance_attention/v1/verify.py
```

The final verifier refuses to overwrite verification.json. Completed reports preserve source snapshots and input/figure/verification hashes.

![Distance and attention](distance_attention.png)

![Raw scores and components](score_components.png)

Full data: per_core_head.csv, per_head.csv, per_block.csv, curves.npz, core_*.json/npz, summary.json. Figures have PNG and PDF versions; metadata is in figure_provenance.json and verification.json.
