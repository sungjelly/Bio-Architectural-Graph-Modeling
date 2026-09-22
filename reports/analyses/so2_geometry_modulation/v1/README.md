# Geometry modulation variation

Descriptive frozen-checkpoint inspection requested after the τ/β diagnostic.
Source run: `r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf`, epoch 200,
seed 0. Objective: determine how far the multiplicative geometry term g departs
from its neutral value 1, and how much of that departure changes across edges.
This is an exploratory model diagnostic, with no retraining or biological claim.

The mean over 32 dimensions of each edge/head is exactly 1 by construction.
An alternative to adaptive geometry dependence is nearly fixed dimension
reweighting. Distinguish these by decomposing mean squared departure from 1
into fixed average dimension weighting, within-core edge variance, and
between-core variance of dimension means. Cores are descriptive strata, not
asserted independent biological replicates. All 14 cores and 32 block/head
combinations are included. Edges are equally weighted within each core; cores
and head dimensions are equally weighted in aggregation.

Protocol fixed before extraction: stream every geometry edge through each
trained block's geometry encoder, FP32, eval mode, without expression inputs,
masks, or full graph forwarding. g is a deterministic function of geometry.
Use stable variance/mean reduction to avoid cancellation for near-constant g.
Report departure RMS, across-edge RMS, fixed RMS, dynamic fraction, extrema,
and fractions within 10% of 1 and beyond 25% of 1. These thresholds describe
magnitudes, not scientific go/no-go gates. Quantile/density visualization uses
4096 uniformly sampled edges per core, seed 2026090500 + core number, and all
head dimensions; sample the same edges in every block. Full moments and
threshold fractions use every edge, not the display sample.

Verify strict checkpoint and training source identity, geometry file/manifest
hashes, full edge coverage, finite positive g, per-edge/head dimensional mean
one, and variance decomposition. Pilot core 21 before independent core workers
on available GPUs. Stop on any failed check. Source artifacts remain immutable.
Deliver PNG/PDF, per-head/core metrics, summary/provenance, and reproduction code.

Run from project root using `PYTHONPATH=src /venv/main/bin/python
reports/analyses/so2_geometry_modulation/v1/analyze.py --cores 21 --device cuda:0`.
Remaining cores use disjoint `--cores` lists/GPU devices; then `--summarize` and
`render.py`. No rerun may overwrite a completed core receipt.

## Completed diagnostic

All 55,980,536 directed edges in 246,063 cells across 14 cores were measured in each of four blocks. There was no expression/masking dependence, neighbor sampling, training, or checkpoint selection. Full reductions covered 448 core/block/head combinations.

| Block | RMS departure from 1 | Fixed dimension RMS | Across-edge RMS | Dynamic share of squared departure | Within [0.9,1.1] | Approximate central 90% of g |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.2923 | 0.2570 | 0.1393 | 22.7% | 22.2% | approximately 0.65–1.55 |
| 2 | 0.2565 | 0.2053 | 0.1538 | 36.0% | 29.0% | approximately 0.67–1.52 |
| 3 | 0.2236 | 0.1627 | 0.1534 | 47.1% | 37.2% | approximately 0.68–1.42 |
| 4 | 0.1980 | 0.1095 | 0.1650 | 69.4% | 44.6% | approximately 0.67–1.30 |

g varies meaningfully across geometry edges, with a typical standard deviation of 0.139–0.165 for a fixed head/dimension. Departures from 1 are larger (0.198–0.292 RMS), because they also include a stable average pattern of dimension weights. The share attributable to edge dependence rises from 23% in block 1 to 69% in block 4. Thus early blocks contain more fixed dimension reweighting, while later blocks have a larger adaptive share despite smaller total departures from 1.

The global observed coefficient range was 0.425–2.279; these extrema are not typical ranges. Values remain positive and every edge/head averages to 1 over dimensions. The raw pre-normalization range [0.5,1.5] is not a bound on normalized g. The central 90% intervals above are approximate, based on 4096 sampled edges per core and histogram bins of width 0.01. Full moments, threshold fractions and extrema used every edge.

The decomposition is T = F + W + B: T is average squared departure from 1; F is the squared departure of the cohort-average dimension weights; W is within-core edge variance; B is variance of core means. Dynamic share = (W+B)/T. Edges are equally weighted within cores; cores, dimensions and heads are equally weighted. These fractions describe squared coefficient variation, not a fraction of attention or predictive effect. The latter would also depend on normalized Q/K and softmax. This is a single-checkpoint descriptive result, not evidence for generalization or biological mechanisms.

Checks: strict source checkpoint loading and matching geometry implementation; every geometry file hash bound to the training graph manifest; finite positive outputs; maximum dimensional-mean error 2.38e-07; full edge counts and variance decomposition verified. FP32, AMP and TF32 disabled. Peak GPU allocation 0.339 GiB. Sum of per-core elapsed times 118.3 seconds across independent workers. No failures. Source snapshots, masks-free scope, histogram sampling seeds, software/hardware and execution assignments are retained locally.

### Reproduction

Run the pilot first, then the four independent worker commands concurrently on free GPUs. Completed core files are never overwritten by the extraction script.

```bash
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_geometry_modulation/v1/analyze.py --cores 21 --device cuda:0
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_geometry_modulation/v1/analyze.py --cores 23 20 --device cuda:0
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_geometry_modulation/v1/analyze.py --cores 15 25 22 --device cuda:1
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_geometry_modulation/v1/analyze.py --cores 27 26 28 17 --device cuda:2
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_geometry_modulation/v1/analyze.py --cores 19 16 24 18 --device cuda:3
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_geometry_modulation/v1/analyze.py --summarize
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_geometry_modulation/v1/render.py
```

![Geometry modulation variation](g_variation.png)

Main files: g_variation.png/.pdf, per_head.csv, per_block.csv, summary.json, distributions.npz, core_*.json, verification.json, and figure_provenance.json.
