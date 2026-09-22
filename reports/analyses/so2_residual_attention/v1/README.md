# Residual versus attention update size

Exploratory descriptive analysis of frozen SO2 seed-0 epoch-200 run
`r_20260903T073353Z_ed491664_s000_f00_a01_c6362abf`, checkpoint SHA256
`aba6a1d910e60f6a30bb4b27b15dbd1c30199da41a15460c23d994079b67543b`.
Question: how large is each block's projected attention update relative to
its incoming residual, and how much does this vary across cells and cores?

Protocol, fixed before inspecting activations: all 14 frozen SO2 core graphs,
246,063 cells, 4 blocks, FP32 eval and the same fixed masks as the prior tau/beta
and distance analyses. At each block capture h (incoming residual), u (actual
attention output projection), a=h+u, f (FFN output), y=a+f. No dropout in eval.
Primary metric is ||u||2/||h||2 per cell; report core means, exact core quantiles,
equal-core aggregates and histograms. Also report RMS norm ratio, cosine(h,u),
||h+u||/||h||, fraction u larger than h, and ||f||/||a|| separately. A ratio
of one is equal magnitude; zero is no update. There is no preferred outcome.

Alternative explanations: large common vectors can inflate norms without
large between-cell variation; an update can reinforce, cancel, or rotate the
residual. Sensitivities measure feature-centered norms (subtract each vector's
feature mean) and between-cell centered RMS norms (subtract each core's mean
vector). Neither is an information-theoretic quantity or a predictive ablation.
Later residuals already contain previous attention and FFN updates, and sender
representations can carry multihop information. No self-versus-neighbor percentage,
causal mechanism, held-out predictive gain, or independent-patient CI is inferred.

Observational units are cells; core summaries show descriptive heterogeneity,
not independent patient replication. No fitting or model selection, no new split,
no clinical annotations and no new expression/graph inputs. One checkpoint/mask
per core; seed/mask robustness and predictive ablation remain unmeasured.

Acceptance: strict checkpoint/source/input/mask binding; all cells and edges;
finite summaries; exact h+u+f reconstruction within FP32 tolerance; norm energy
identity; synthetic aligned/opposed/orthogonal/zero-update recovery; pilot public
forward replay; independent aggregate verification. Stop on failed checks.
Deliverables: code, core receipts, distributions, summary tables, PNG/PDF figure,
provenance, verification and reproduction commands. This is a post-hoc report,
not a new trained experiment or a change in biological evidence status.

Compute: all four RTX 3090 GPUs are allocated to existing SO1 training PIDs
442492–442495. Use CPUs at nice 10, eight threads per worker, pilot core 21
before four disjoint workers; do not alter existing GPU workloads. Preserve
raw/clinical inputs, completed training bundles and unrelated working changes.

Reproduction from repository root (extraction refuses overwrite):

```bash
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_residual_attention/v1/analyze.py --cores 21 --threads 8 --pilot
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_residual_attention/v1/analyze.py --cores 23 20 --threads 8
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_residual_attention/v1/analyze.py --cores 15 25 22 --threads 8
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_residual_attention/v1/analyze.py --cores 27 26 28 17 --threads 8
nice -n 10 env PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_residual_attention/v1/analyze.py --cores 19 16 24 18 --threads 8
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_residual_attention/v1/analyze.py --summarize
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_residual_attention/v1/render.py
PYTHONPATH=src /venv/main/bin/python reports/analyses/so2_residual_attention/v1/verify.py
```

## Completed findings

All 14 cores, 246,063 cells and 55,980,536 directed edges were included. This is a descriptive diagnostic of the specified frozen endpoint and masks. The actual attention update is smaller than the incoming residual at every measured cell/block. Its absolute norm grows through the network while the residual grows faster.

| Block | Residual RMS norm | Attention RMS norm | Attention/residual RMS | Between-cell-centered RMS ratio | Core RMS ratio range | Mean cosine(h,u) | FFN/post-attention RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.3656 | 1.8636 | 0.2530 | 0.2090 | 0.2277–0.2680 | 0.0987 | 0.3949 |
| 2 | 8.6624 | 2.1742 | 0.2510 | 0.2103 | 0.2286–0.2745 | 0.1591 | 0.3002 |
| 3 | 10.0824 | 2.4600 | 0.2440 | 0.2278 | 0.2063–0.3092 | 0.1672 | 0.2446 |
| 4 | 11.6427 | 2.6030 | 0.2236 | 0.2085 | 0.1965–0.2603 | 0.1910 | 0.2058 |

RMS ratio means sqrt(mean_core(mean_cell(||u||²))) / sqrt(mean_core(mean_cell(||h||²))). It is distinct from averaging the cellwise ratio. Equal-core means of cellwise u/h are 0.2927, 0.2802, 0.2598, 0.2315. There is no attention/update mixing coefficient that sums these values into a percentage of information.

| Block | Mean core q10 of u/h | Mean core median | Mean core q90 | Largest observed u/h |
|---|---:|---:|---:|---:|
| 1 | 0.1827 | 0.2747 | 0.4309 | 0.8502 |
| 2 | 0.1820 | 0.2627 | 0.4061 | 0.7765 |
| 3 | 0.1728 | 0.2480 | 0.3670 | 0.6840 |
| 4 | 0.1601 | 0.2239 | 0.3155 | 0.5068 |

Core quantiles are exact; their averages are not pooled quantiles and are not patient-level confidence intervals. All individual core summaries and complete scalar histograms are retained. Across-core RMS heterogeneity is modest, but within-core cell ratios vary appreciably. The largest observed ratios are still below one in every block.

Mean cosine(h,u) is only 0.10–0.19, so the update has low alignment with the residual. The attention addition raises aggregate RMS norm by 5.58%, 6.65%, 6.69%, 6.77%, respectively; averaging cellwise norm changes instead gives 7.28%, 8.31%, 8.02%, 7.51%. A nonzero update can change direction as well as magnitude, so neither increase equals its norm ratio. The separate FFN update declines from 39.5% to 20.6% of its actual input RMS magnitude.

Subtracting each core’s mean vector from each branch gives attention/residual RMS ratios of 20.9%, 21.0%, 22.8%, 20.9%; common across-cell offsets therefore do not explain away the observed update size. This sensitivity also removes potentially useful core-wide signals. The feature-centered per-cell ratios are in the full tables; subtracting a vector’s feature mean is a different operation. LayerNorm acts on feature-centered/scaled states locally, while the final decoder can still use the full embedding.

The residual from block 2 onward already includes earlier attention and FFN updates. These magnitudes therefore compare accumulated state to a new update, and cannot partition cell-autonomous versus neighbor information. No decoder ablation, predictive necessity, information entropy, biological communication, held-out performance, or seed/mask robustness was tested.

## Verification and resource record

Independent saved-statistic audit passed all 56 core/block rows, all five full-count histogram metrics, seven source snapshots, 42 input bindings, mask identity against the prior distance analysis, checkpoint/configuration identity, norm-energy identities, RMS reaggregation and CSV agreement. Public-forward pilot replay, exact block reconstruction and independently reconstructed weighted-message updates all had maximum absolute error zero. Independent aggregate RMS ratios agreed within 5.56e-17, and norm moments/energy within 2.85e-14. Histogram bin bounds also agree with saved moments/quantiles.

No raw activations were persisted. Exact within-core centered variances and quantiles are extracted from all cells; the independent audit validates their bounds and histogram consistency rather than redoing full inference. Input files were fully hashed during extraction; the final audit compares receipts to frozen manifests and previously audited masks/inputs without rereading every large input byte. Post-attention norm metrics recompute h+u in FP64, while the executed model addition is FP32; this only introduces rounding-scale differences.

CPU inference used four disjoint low-priority workers with eight PyTorch threads each because all GPUs remained allocated to the existing SO1 training. The pilot including public-forward replay took 37.8 seconds. Sum of overlapping core elapsed times was 1247.2 seconds; maximum recorded process RSS was 8.34 GiB. No extraction failures or omitted cores. Versions, allocation and all runtimes are in execution_plan.json; independent checks are in verification.json.

![Residual and attention comparison](residual_attention.png)

Figures: [PDF](residual_attention.pdf). Data: summary.json, per_core_block.csv, core_*.json and core_*_histograms.npz. Figure hashes and rendering versions are in figure_provenance.json. The original model and training artifacts were not modified.
