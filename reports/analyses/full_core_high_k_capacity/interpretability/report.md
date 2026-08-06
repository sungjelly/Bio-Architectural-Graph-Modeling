# Full-core G2 toy interpretability analysis

- Run: `r_20260725T083929Z_8364b333_s000_f00_a01_f5c47601`
- Protocol: `held_in_full_core_fixed_budget`
- Scope: one held-in, transductively fitted spatial core; this is not validation, test, patient-level, or causal evidence.

## Attention routing

The expression-independent hash sample contained 128 masked receivers. Mean receiver-level attention entropy was 6.54774 nats, mean effective neighbor count was 713.204, and mean attention-weighted distance was 147.81 µm.

## Outcome-conditioned deletion analysis

The high-lymphoid-score set contained 64 masked receivers. Deleting the top 10% of incoming edges by last-layer mean-head attention was compared with 8 equal-count, deterministic, exactly rank-distance-bin-matched deletions. This removes edges from both GAT layers and renormalizes attention; it is not an isolated last-layer intervention.

- Mean across-draw top-minus-null receiver-program Huber-change: -0.00080256
- Mean across-draw top-minus-null receiver-program prediction-MAE: 0.00342178
- Minimum-effect deletion support: `false`

## Organizer sender-program perturbation

The three organizer channels were mean-ablated in 256 top-routed unmasked senders and compared with 8 equal-count, disjoint, distance-matched sender sets.

- Mean organizer-score top-minus-null enrichment: 0.199156
- Mean sender-ablation top-minus-null receiver-program Huber-change: -4.2748e-07
- Minimum-effect sender-program support: `false`
- Organizer-enrichment support: `true`
- Joint TLS dependency support: `false`
- Evidence label: **no joint deletion-, organizer-enrichment-, and sender-program-supported TLS-related dependency detected**

## Interpretation boundary

The strongest possible label is: **model-implied TLS-related predictive dependency in one transductively fitted core**. Attention is not importance. Even a positive deletion contrast shows model routing sensitivity, not biological importance or causality. Receiver selection used observed outcomes and is explicitly exploratory.
