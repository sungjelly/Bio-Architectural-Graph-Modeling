# SO2 reconstruction diagnostic

Outcome: **inconclusive about the neural failure cause**; objective mismatch is supported in the constant-predictor class.

999/1000 per-gene Huber constants are below the fitted log mean; mean standardized shift -0.194874. Geometry all-entry Huber 0.239952905 versus Huber constant 0.253724528. Geometry positive standardized MSE 8.915117984; local log mean 8.531140629, local count mean 7.898871660, original-full-graph log mean 8.589392370. Full matched tables expose all-entry and zero/positive tradeoffs. Published methods differ in data scale, objective and task, and supply no directly comparable SOTA threshold.

The objective demonstrably favors lower constants than MSE, and matched spatial baselines reveal metric-dependent performance. Current evidence cannot identify one neural failure cause, establish graph-specific predictive gain, or support a biological/causal claim.

See the [full diagnostic report](../../../../reports/analyses/so2_failure_diagnosis/comparison/v1/report.md) for matched tables, literature, alternative explanations, tests and source provenance. This supplements the earlier nonzero comparison; it does not supersede its endpoint results.
