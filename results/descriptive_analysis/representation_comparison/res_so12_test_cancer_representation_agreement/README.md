# Frozen encoder cluster unions and outlined cancer regions in two SO12 test cores

Fixed resolution1 seed0 footprint Dice (core15/core23): h4: oracle 0.7836/0.9171, other-core transfer 0.7457/0.8780; h3: oracle 0.7921/0.9143, other-core transfer 0.7332/0.8550; h2: oracle 0.7797/0.9115, other-core transfer 0.6589/0.8614; h1: oracle 0.7865/0.9011, other-core transfer 0.7400/0.8114; h0: oracle 0.7804/0.8867, other-core transfer 0.6705/0.8152; PCA: oracle 0.7710/0.8648, other-core transfer 0.6492/0.7781; scVI: oracle 0.7292/0.9041, other-core transfer 0.6687/0.8497. All56 partitions (7 methods ×4 resolutions ×2 Leiden seeds) and both pixel supports retained. PCA50 retains 9.407% of training feature variance. All-positive footprint baselines are 0.6347/0.8502.

These numbers are exploratory annotation agreement. Oracle unions are selected on the scored core; transfer unions are selected on the other jointly clustered core. Neither is unbiased cancer-classifier accuracy. Outside-outline predictions are penalized.

Only two cores, donor overlap and historical exposure; one fitted encoder seed. H&E screenshots lack an authoritative transform, exhaustive-label certification and verified same-section identity. Approximate alignment is strong in outer silhouette but core23 has substantial internal/acellular coverage discrepancy. Native-resolution raster omits641cells with no valid owned pixel and assigns overlaps deterministically. PCA/scVI lack BAGM morphology/graph inputs; scVI also models slide batches. Oracle is optimistic; transfer is between jointly clustered cores; no population significance or superiority claim.

Complete figures, all resolutions/seeds and machine-readable scores: `reports/analyses/so12_cancer_comparison/ev_so12_cancer_comparison_s0_20261018`.

BAGM source `r_20261017T044837Z_caf76350_s000_f00_a01_12df41be`; separately cataloged scVI source `r_20261018T052052Z_04078a70_s000_f00_a01_14d78fb9`. No raw/clinical data or scientific model artifacts were deleted for this task.
