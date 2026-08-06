# Source-statistic audit inheritance

This corrective campaign inherits the equation- and provenance-level audit at
`experiments/campaigns/cmp_20260731_myjju_genemae_gradient_audit/source_statistic_audit.md`
without reinterpretation.

The audited external repository is
`/workspace/Gastric-Cancer-Analysis-by-MyJJu` at commit
`f9ef61071c7e9b2751bbd59d154c13de534e7f2f`. The audited
`scripts/gene_mae_coexpr.py` file has SHA-256
`0ae143b5957e9275882ba595702d6eacd033545beda306f0f0f82905a8174680`.
The predecessor audit has SHA-256
`11f6bfb0af489c533f6f036c4269d12665245e2870dc8bcec8f79270f75fc846`.

The reproduced source statistic remains a global signed directed derivative:

```text
J[target, source] =
  (1 / N) d(sum_output_cells reconstruction[target])
            / d(uniform input[source] shift over input cells)
```

The source publishes `0.5 * (abs(J) + abs(J.T))`, which removes sign and
direction. Historical source checkpoints and exact tile identities remain
unavailable, so this is a procedure-faithful adaptation to the frozen
ten-core adjacent-normal ensemble, not a numerical replay of the source
slide. Source-selected genes and pairs remain circular positive controls, not
independent biological validation.

The only successor amendment is the device used for the strict checkpoint
replay control. It does not change this statistic, any gradient value,
scientific gate, or claim ceiling.
