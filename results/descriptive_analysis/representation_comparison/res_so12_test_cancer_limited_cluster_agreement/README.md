# One or two complete Leiden clusters against outlined regions on SO12 test cores

Fixed resolution 1, seed 0 footprint Dice/F1: h4: core 15 one=0.5000, ≤2=0.6900, IDs [5,6]; core 23 one=0.4768, ≤2=0.6814, IDs [0,7] | h3: core 15 one=0.6998, ≤2=0.7739, IDs [0,8]; core 23 one=0.4216, ≤2=0.6889, IDs [1,4] | h2: core 15 one=0.6379, ≤2=0.7137, IDs [0,3]; core 23 one=0.4500, ≤2=0.6203, IDs [1,10] | h1: core 15 one=0.6361, ≤2=0.6962, IDs [0,11]; core 23 one=0.3414, ≤2=0.5596, IDs [6,8] | h0: core 15 one=0.5535, ≤2=0.6798, IDs [1,10]; core 23 one=0.3214, ≤2=0.5432, IDs [6,8] | pca: core 15 one=0.4528, ≤2=0.6780, IDs [3,5]; core 23 one=0.4307, ≤2=0.6098, IDs [2,4] | scvi: core 15 one=0.2704, ≤2=0.4228, IDs [2,4]; core 23 one=0.3163, ≤2=0.4987, IDs [0,2]. All 56 partitions retained.

These are annotation-selected descriptive F1 scores, not independently evaluated cancer-classifier accuracy. Region selection uses the scored core. The opposite-core transfer results are separately reported and remain exploratory because both cores were jointly clustered. Registration is approximate; section identity and annotation exhaustiveness are unverified. Only 69.5% and 65.0% of annotated pixels have segmented-cell coverage on cores 15 and 23. Two cores and one fitted encoder seed do not support population superiority claims.

The restricted positive group contains at most two whole clusters. All other evaluable pixels remain in the FP/FN denominators. Selected clusters retain their color outside the outline; no error pixels are hidden. This is a separate follow-up and does not replace the unrestricted result.

Full report, source-bound images and metrics: `reports/analyses/so12_cancer_limited_union/ev_so12_cancer_limited_union_s0_20261018`.
