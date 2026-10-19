# SO_2 GeneMAE: 10% masking comparison

Whole-cell-trained GeneMAE has equal-core Huber 0.038705 on fixed 10% whole-cell masks, 7.93% improvement over visible per-gene means; 14/14 cores and 3/3 seeds favor it. The prespecified reconstruction baseline gate is supported. On the same whole-cell evaluation masks, whole-cell training has 21.86% lower Huber than training with individual genes masks; 11.08% lower Huber than training with spatial block masks. The all-zero predictor has lower MAE (0.099643 versus 0.148733); its Huber is 0.050397 and MSE 0.112680, compared with model MSE 0.079474. This metric disagreement makes the primary loss and positive-expression errors important to interpretation. All nine models completed 200 training epochs. Seven original runs passed all execution checks; individual-gene seeds 1 and 2 failed the final exact CPU replay/invariance assertion. Independent fresh CPU replay and hidden-target checks passed, but full evaluation replays failed the prespecified numerical agreement gate only for median gene-wise Pearson (maximum differences 6.24e-7 and 7.36e-7). The strict gate remains failed; the nine-run verification criterion is not met. All seeds remain in this explicitly descriptive comparison, using their original recorded evaluations. Reconstruction loss replay differences and every secondary metric discrepancy are retained in replay_audit.json. The overall campaign outcome is inconclusive under its full verification contract; the narrower whole-cell baseline result is supported.

All nine 200-epoch trained models are included. Values average three mask replicates per core, fourteen cores equally, then three seeds. This is fitted-cohort reconstruction; evaluation cells were visible in other training epochs.

![Masked Huber comparisons within each evaluation task](reconstruction_comparison.png)

## Matched whole-cell reconstruction

| Training mask | Huber ± seed SD ↓ | MSE ↓ | MAE ↓ | R² ↑ | Pooled Pearson ↑ | Mean gene Pearson ↑ | Positive MSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whole cell | 0.038705 ± 0.000026 | 0.079474 | 0.148733 | 0.2036 | 0.4437 | 0.1479 | 0.671150 |
| Individual genes | 0.049533 ± 0.000119 | 0.108489 | 0.113272 | -0.0595 | 0.2333 | 0.0173 | 1.024260 |
| Spatial block | 0.043528 ± 0.000001 | 0.092695 | 0.166251 | 0.0841 | 0.3077 | -0.0008 | 0.761830 |

| Training mask | Seed 0 Huber ↓ | Seed 1 Huber ↓ | Seed 2 Huber ↓ |
|---|---:|---:|---:|
| Whole cell | 0.038698 | 0.038733 | 0.038683 |
| Individual genes | 0.049669 | 0.049479 | 0.049449 |
| Spatial block | 0.043527 | 0.043530 | 0.043528 |

![Whole-cell reconstruction in each fitted core](core_reconstruction.png)


## Cross-task evaluation

| Training mask | Whole-cell Huber ↓ | Gene-entry Huber ↓ | Spatial-block Huber ↓ |
|---|---:|---:|---:|
| Whole cell | 0.038705 | 0.039467 | 0.044230 |
| Individual genes | 0.049533 | 0.035050 | 0.050155 |
| Spatial block | 0.043528 | 0.208019 | 0.043937 |

Compare models within a column. Differences between columns include task difficulty and available information; they do not establish model superiority.

## Controls on the requested whole-cell task

| Condition | Huber ↓ | MAE ↓ | Positive MSE ↓ |
|---|---:|---:|---:|
| neighbor mean | 0.041761 | 0.154495 | 0.675239 |
| node label permuted | 0.042799 | 0.158570 | 0.742054 |
| observed | 0.038705 | 0.148733 | 0.671150 |
| self edges only | 0.044251 | 0.141746 | 0.861772 |
| visible gene mean | 0.042037 | 0.160914 | 0.726177 |
| zero | 0.050397 | 0.099643 | 1.072599 |

![Positive and zero errors on common whole-cell masks](positive_zero_errors.png)

![All nine training trajectories](training_curves.png)

## Interpretation and limitations

Sparsity, broad spatial fields, cell identity and segmentation spillover remain alternative explanations. Visible means, neighbor smoothing and frozen-model graph ablations address some shortcuts but do not replace trained matched self/morphology/regional controls. Node-label permutation is not a confounder-preserving biological null. No patient-held-out split, population interval, calibration, biological mechanism or causal effect is established. All uncertainty bars describe technical seed variability only. Training masks are shared across seeds within each method, while initialization, dropout and traversal order vary with seed. Tiling removes cross-tile edges and may especially limit block reconstruction. Equal epoch budgets visit every tile; spatial-block tiles without targets skip optimization, so optimizer-step counts differ. Pooled Pearson can reflect differences in average abundance between genes; mean gene-wise Pearson is reported separately. The full tables expose positive and zero errors, undefined correlations and core heterogeneity.

## Execution and provenance

| Training mask | Seed | Immutable run | Status | GPU | Optimizer steps | Runtime min | Peak VRAM GiB |
|---|---:|---|---|---:|---:|---:|---:|
| Whole cell | 0 | `r_20261005T071413Z_eb2b8bb8_s000_f00_a01_cd04af8b` | completed | 2 | 9800 | 37.51 | 8.88 |
| Individual genes | 0 | `r_20261005T071413Z_b35cac84_s000_f00_a01_76060a5a` | completed | 3 | 9800 | 54.81 | 8.88 |
| Spatial block | 0 | `r_20261005T071413Z_eb047e27_s000_f00_a01_22a79608` | completed | 1 | 5003 | 26.15 | 8.88 |
| Whole cell | 1 | `r_20261005T071413Z_eb2b8bb8_s001_f00_a01_c7db4275` | completed | 0 | 9800 | 37.85 | 8.88 |
| Individual genes | 1 | `r_20261005T074042Z_b35cac84_s001_f00_a01_95e91d2d` | failed | 1 | 9800 | 55.91 | not saved |
| Spatial block | 1 | `r_20261005T080924Z_eb047e27_s001_f00_a01_8934164f` | completed | 3 | 5003 | 26.72 | 8.88 |
| Whole cell | 2 | `r_20261005T075224Z_eb2b8bb8_s002_f00_a01_836c985d` | completed | 0 | 9800 | 37.22 | 8.88 |
| Individual genes | 2 | `r_20261005T075208Z_b35cac84_s002_f00_a01_e5347d9b` | failed | 2 | 9800 | 55.55 | not saved |
| Spatial block | 2 | `r_20261005T082957Z_eb047e27_s002_f00_a01_3601f656` | completed | 0 | 5003 | 27.31 | 8.88 |

Hardware: NVIDIA GeForce RTX 3090; PyTorch 2.13.0+cu130; CUDA 13.0; FP32. One independent job per GPU on devices 0–3, with the next queued job claimed as a device becomes free. Longer pending gene-entry runs received priority after initial throughput measurements.

Diagnostic and failed runs are retained in `verification.json`; no production seed is silently omitted.

Focused masking/configuration/queue tests passed (49 initially), and archive/masking checks passed (34 after the serialization fix). Recovery routing and strict replay-audit checks passed with the focused suite (52 tests). The latest CPU-isolated full repository suite returned 1,437 passed, 34 failed and four skipped. Failures concern unavailable historical adjacency/GeneMAE/geometry inputs or an older external gradient-audit source identity. Logs are retained in `../verification/`. Initial doctor also reported a pre-existing incomplete checkpoint catalog and its generic 25-GiB disk threshold; this campaign uses the documented 8-GiB floor plus a 3-GiB output reserve. These exceptions are not represented as a clean repository-wide audit.

Reproduce from repository root with `PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so2_genemae_masking.py` after all prespecified runs finalize. The script refuses an existing report directory. Training commands are in the campaign README.
