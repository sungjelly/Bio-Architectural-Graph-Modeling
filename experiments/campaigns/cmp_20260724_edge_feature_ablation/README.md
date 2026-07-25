# Edge-Feature Ablation Example

## Status

- Phase: planned
- Outcome: pending
- Execution: configuration example only; not enqueued

## Task contract

The objective is to compare the existing topology-only G1 model with the
existing edge-conditioned G2 model while holding the protected normal-core
dataset, spatial split, P+N+B masking, k=16/radius=75 µm mutual graph, 512
hidden units, two graph layers, optimization, masks, and seed set fixed.

The scientific question is whether measured edge geometry adds predictive
information beyond topology. The primary hypothesis predicts lower held-out
whole-node masked Huber loss for G2. Credible alternatives are that topology is
sufficient, that a gain is only distance smoothing, or that short-range
segmentation spillover creates the apparent signal. G2 must therefore be
compared with zero-edge and within-distance-bin edge-permutation controls, and
the graph models must remain paired on masks and spatial blocks.

The estimand is the spatial-block distribution of
`loss(G1) - loss(G2)` after the seed-ensemble policy is applied. Spatial blocks
are the uncertainty units; cells and seeds are not biological replicates. The
primary metric is `val/masked_huber` (lower is better). The maximum claim is a
within-core model-implied predictive dependency. This example cannot establish
patient generalization, communication, mechanism, or causality.

The G2 branch passes only if the prespecified paired contrast favors G2, its
spatial-block uncertainty interval excludes zero in that direction, and G2
also beats both edge controls without a result driven by one favorable seed.
A null or reversed result supports the simpler topology-only explanation. Stop
before sealed-test evaluation if configuration, leakage, graph-QC, mask
pairing, or artifact-validation checks fail.

Inputs are the registry entries `cosmx_normal_core_legacy_v1` and split
`127b17da70688537`; protected identifiers must remain outside tracked files.
Train-only fitting, independently built split graphs, no cross-split edges, and
the existing prohibited-input policy are mandatory. Expected outputs are
resolved configurations, provenance, block-level metrics, predictions,
checkpoints, graph/mask diagnostics, and one immutable run bundle per attempt.

The example uses one GPU per worker and one training subprocess at a time. It
must not be run merely as part of repository validation. Before activation,
review the campaign definition, register the campaign, enqueue the two
configuration families explicitly, and preserve the sealed test policy.

Configuration-only verification:

```bash
PYTHONPATH=src python -m spatial_benchmark doctor
PYTHONPATH=src python -m spatial_benchmark enqueue-sweep --help
```

The exact variants are
`configs/experiment/edge_feature_ablation_g1.yaml` and
`configs/experiment/edge_feature_ablation_g2.yaml`; the declarative sweep is
`configs/sweep/edge_feature_ablation_example.yaml`.
