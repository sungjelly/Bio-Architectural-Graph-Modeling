# QKV-GAT bounded routing diagnostic

Status: **negative-capacity-gate diagnostic only**.

The analysis used 8 deterministically hash-selected whole-node receivers. Selection did not use expression or TLS scores.

## Routing

Normalized attention is a computational routing quantity. The production value gate rescales that routing, so the reported effective coefficient is attention × value gate. Neither quantity is biological importance.

Mean effective-neighbor count per receiver/head: 3497.588; mean effective-routing-weighted distance: 331.564 µm.

## Prediction faithfulness

Deleting the top final-layer effective-routing edges changed predictions by mean absolute 0.00113145. The ratio to the mean deterministic distance-matched null change was 7.632.

This is an isolated final-layer intervention. It does not test earlier-layer routes, graph-specific generalization, cell-cell communication, mechanism, or causality.

## TLS annotation

Status: suppressed_negative_capacity_gate. TLS results are exploratory annotation only and are not independent validation.
