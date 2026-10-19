# SO_1 + SO_2 preprocessing comparison

The locked preprocessing plausibility gate passes for Log1p counts per 10,000, Quasi-Poisson Pearson, Signed-log Pearson.

**Maximum claim: exploratory preprocessing plausibility only.**

The dataset retains 407,659 mapped cells × 1,000 biological probes across 28 cores. Training: 301,198 cells from 10 donors; held-out diagnostics: 106,461 cells from 4 donors. Transform parameters and 32-PC projections were fitted on all training cells pooled across slides; diagnostics use the same 28,000 sampled cells for every method, up to 1,000 per core. Paired cores remain in the same partition. The existing exclusion of 340 unmapped SO_2 FOV246 cells is retained.

## Comparison

| Transform | Depth R² | SO_1 BA | SO_2 BA | SO_1 null BA | SO_2 null BA | Slide AUROC | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Log1p raw counts | 0.372 | 0.706 | 0.735 | 0.266 | 0.302 | 0.559 | reference |
| Log1p counts per 10,000 | 0.326 | 0.679 | 0.730 | 0.277 | 0.309 | 0.564 | pass |
| Quasi-Poisson Pearson | 0.033 | 0.717 | 0.760 | 0.230 | 0.341 | 0.553 | pass |
| Signed-log Pearson | 0.031 | 0.737 | 0.727 | 0.285 | 0.318 | 0.515 | pass |

Depth R² is the median held-out-core, variance-weighted squared correlation of PCs with log1p biological-probe depth. BA is balanced accuracy. Lower depth R² supports depth reduction; slide AUROC is confounded.

A candidate must have lower median held-out-core depth R² than log1p(raw), and its held-out vendor-label balanced accuracy must drop by no more than 0.05 on each slide. Both conditions are required. Slide AUROC and label-null accuracy do not select a winner.

Use plain quasi-Poisson Pearson as the primary candidate in a subsequent controlled training comparison, with signed-log Pearson retained as an ablation. This prioritizes the documented residual transform without claiming that its training performance is established. Passing this gate does not establish an advantage in neural-network training. Signed log trades annotation agreement between slides and does not demonstrate a clear overall advantage.

![Four preprocessing comparisons: PCA colored by slide and depth association for every held-out core.](../../../artifacts/runs/2026/10/r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410/diagnostics/preprocessing_diagnostics.png)

Top: sampled first two PCs, colored by slide; this descriptive separation is confounded. Bottom: variance-weighted depth R² for every held-out core; dashed line marks the median. The primary metric uses all 32 PCs, not just the plotted two.

## Transform definitions

Pearson residuals are defined from raw counts x, not log counts: nᵢ = Σg xᵢg; p̂g = Σtraining xᵢg / Σtraining,g xᵢg; μᵢg = nᵢp̂g; rᵢg = (xᵢg − μᵢg) / √(1.01 μᵢg). Use training gene means and sample SDs to obtain z, then cap only the upper tail at 10. The signed-log ablation applies sign(z)·log1p(|z|) afterward. It is nonstandard and is not claimed to be an AtoMx option. Ordinary log1p(z) is invalid when z ≤ −1. Baselines also use frozen training gene standardization. PCA re-centers each transformed training matrix; this is not a bit-for-bit reproduction of vendor PCA. In this dataset, 0.09% of plain-Pearson feature entries are ≤ −1, so ordinary log1p would be nonfinite.

The residual formula follows the pinned vendor scPearsonPCA implementation. The exact built-in AtoMx residual implementation is not publicly established. The vendor tutorial discusses Pearson-residual PCA and batch-correction alternatives; neither source establishes that this trial removes technical slide effects.

Sources: [Pinned vendor R implementation](https://github.com/Nanostring-Biostats/CosMx-Analysis-Scratch-Space/blob/41c499026d789ff35288de2ddc62f506a9d51cf0/_code/scPearsonPCA/R/sparse_quasipoisson_pca_seurat.R); [Vendor Pearson-PCA tutorial](https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/pearsonpca/).

## Reusable dataset

Reusable matrices are under /workspace/BAGM/data/processed/so12_pearson_integration_v1. Each of pearson/ and signed_log_pearson/ contains 28 float32 per-core .npy matrices totaling 407,659 × 1,000 values, plus fitted_parameters.npz. observations.csv.gz preserves row order within each core; genes.json supplies column order; manifest.json records shapes and checksums. Raw counts remain in the source per-core expression_counts arrays. Preserve (slide, fov, cell_ID) keys and core IDs; reuse independent core spatial graphs without cross-core or cross-slide edges.

```python
import json
import numpy as np
import pandas as pd
from spatial_benchmark.paths import current_paths

root = current_paths().data_root / "processed/so12_pearson_integration_v1"
core = "SO1-C01"
features = np.load(root / "pearson" / f"{core}.npy", mmap_mode="r")
metadata = pd.read_csv(root / "observations.csv.gz",
                      usecols=["core", "slide", "fov", "cell_ID", "train", "qc_pass", "x_um", "y_um"])
cells = metadata.loc[metadata["core"] == core].reset_index(drop=True)
genes = json.loads((root / "genes.json").read_text())
assert features.shape == (len(cells), len(genes))
assert not cells.duplicated(["slide", "fov", "cell_ID"]).any()
# Row i in features belongs to row i in cells; preserve this alignment.
```

## Interpretation and limits

- Maximum defensible claim: exploratory preprocessing plausibility on one donor-held-out split. No neural network was trained; masked-expression gain, biological integration, and causal mechanisms were not tested.
- Opaque vendor labels are expression-derived and their namespaces are evaluated separately by slide. Accuracy is circular annotation agreement, not independent cell-type or biological validation.
- Donors do not cross slides, and slide is confounded with tissue composition and disease. Lower slide AUROC or apparent mixing is not proof of technical batch removal and can erase biology.
- Lower depth association can reflect removal of real RNA-content biology or mere scale compression. Four held-out donors and one split do not establish robustness or population-level replication.
- The label null permutes training annotations within each core, preserving composition. It is one diagnostic permutation, not a significance test; no cell-independent p-values are reported.
- Cell totals use every measured gene. Masking after these transforms leaks hidden-count information into visible features. These files are suitable for fully observed representation experiments; strict masked-count prediction needs observed-only preprocessing and raw integer targets.
- Vendor QC flags are retained. All mapped cells are primary; QC-pass depth association is a sensitivity check. No new cells or old user files were deleted.

Next discriminating test: compare the passing candidate(s), if any, against both baselines with a locked, mask-aware preprocessing protocol, raw-count prediction targets, donor-held-out splits, and per-slide/core results. Independent labels or orthogonal measurements are needed before a biological integration claim.

## Compute and verification

| Transform | GPU | Transform/PCA seconds | Peak allocated VRAM (GiB) | QC-pass depth R² |
| --- | --- | --- | --- | --- |
| Log1p raw counts | 0 | 2.1 | 4.23 | 0.379 |
| Log1p counts per 10,000 | 1 | 2.1 | 4.56 | 0.311 |
| Quasi-Poisson Pearson | 2 | 4.4 | 5.03 | 0.031 |
| Signed-log Pearson | 3 | 4.6 | 5.03 | 0.027 |

Verified run runtime: 63.3 seconds; parallel schedule: one transform per GPU. GPU 0: NVIDIA GeForce RTX 3090, GPU 1: NVIDIA GeForce RTX 3090, GPU 2: NVIDIA GeForce RTX 3090, GPU 3: NVIDIA GeForce RTX 3090. PyTorch 2.13.0+cu130; CUDA 13.0. Derived dataset: 3.046 GiB; free disk at verification: 8.49 GiB; free disk when reporting: 8.49 GiB. Peak disk usage was not sampled; these are footprint/free-space measurements. No cleanup was performed by this report.

Prior failed attempt(s): r_20261016T082320Z_660b87cb_s000_f00_a01_cd1c1544. The first attempt encountered a CUDA lazy linear-algebra initialization race during parallel setup; it did not supply scientific results to this report. The successful retry initialized libraries serially with unchanged scientific settings. The partial failed-run bundle is retained for audit.

Diagnostic warnings: none recorded.

Run: r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410; seed 0; fold 0; attempt 2. Metrics SHA256: f94d2c2edd7a1a3620a1ed4bbd4d87a274fb4e4b0d4c2007480ed35ddddbe173. Dataset-manifest SHA256: c5c10869268cb5e0ab9787705c41ab4a3be613062b80bc85352d6cc7fcd5a611. Consumed evidence files were checked against the immutable bundle checksums before rendering.

- [Immutable run](../../../artifacts/runs/2026/10/r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410)
- [Dataset manifest](../../../data/processed/so12_pearson_integration_v1/manifest.json)
- [Per-core depth diagnostics](../../../artifacts/runs/2026/10/r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410/diagnostics/per_core_depth_association.csv)
- [Per-donor annotation agreement](../../../artifacts/runs/2026/10/r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410/diagnostics/per_donor_vendor_label_recovery.csv)

Rebuild the report:

```bash
PYTHONPATH=src /venv/main/bin/python scripts/analysis/report_so12_pearson_preprocessing.py --run-id r_20261016T082445Z_660b87cb_s000_f00_a02_4a2f3410
```
