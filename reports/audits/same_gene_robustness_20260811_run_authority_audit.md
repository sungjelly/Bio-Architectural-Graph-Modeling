# Run-authority audit: same-gene robustness multiverse v1

Overall: **PASS** — 67768/67768 checks passed; 0 failed.

This audit independently read the 140 selected immutable run bundles and V0 prepared arrays. It imported neither the campaign analyzer nor `spatial_benchmark`.

## Authority-complete checks

- Run-derived consensus near/permuted matrices reproduced the published eligible 932×932 matrices with maximum absolute error `0`.
- The matched prevalence/target-SD label-null reproduced all 30,000 stored values with maximum absolute error `0`; strata=29, minimum size=3.
- Complete 1,000-gene order SHA-256: `046eb86c7ea8f1fe6977598a0190132340400fc61802fcde63ab5ac0e9502b03`.
- All selected and anchor component MSE/MAE/cell-count records were reconciled to run `results.json`.
- All seed/fold Jacobian diagonal, row-rank, observed/permuted ratio, fold-Spearman, and sign-consistency gates were reconstructed from run NPZ matrices.
- All 40 V0 96→192 validation gains and selected epochs were reconstructed from validation histories.
- The five seed-level row-budget fold-support counts were reconstructed from selected and anchor fold matrices.

## Technical maxima across 140 runs

- peak VRAM: 16.789804 GiB (gate 20.5 GiB)
- analytical/autograd error: 5.55e-17
- analytical/finite-difference error: 4.97e-12
- checkpoint replay metric/prediction errors: 0 / 0

## Remaining boundary

- This run-authority audit still does not turn geometry components into biological or patient replicates.
- Per-slide component-weighted Jacobian summaries require checkpoint-weight reconstruction; the separately published slide-equal aggregate was already checked by the four-file audit, while per-slide summaries remain a provenance-level rather than independent numerical check here.
- Artifact manifests, registry identities, and source/environment hashes are covered by the publication verifier/provenance audit; this file focuses on numerical scientific authority.
