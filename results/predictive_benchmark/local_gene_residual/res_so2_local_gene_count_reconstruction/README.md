# Local gene residual count reconstruction

Exploratory endpoint comparison; primary practical criterion outcome: **negative**.

Across both local seeds, selected NLL=0.326651015, MAE=0.223390653, RMSE=1.100784904; historical hurdle NLL=0.328241753, MAE=0.225991939, RMSE=1.168484806. NLL improved by 0.001590738, below the prespecified 0.002 margin; the full practical criterion did not pass. RMSE improved 5.794% and MAE 1.151%. Local context improves NLL over matched self-only controls in both seeds and both cores. All four runs completed and replayed.

The independent model uses a two-hidden-layer masked-cell MLP plus same-gene
local corrections, with Bernoulli hurdle/zero-truncated NB2 count outputs. It
has 2,329,432 parameters, compared with 6,160,088 in the historical hurdle model.
Both variants were trained for 32 epochs with model seeds 0/1, identical masks,
ordering and optimizer schedules. Best epochs were selected by count NLL.

The failed 0.002 NLL margin is retained as a negative result for the prespecified
material-gain hypothesis. The measured RMSE and smaller NLL improvements remain
useful partial evidence; they are not an independent-test result. Historical
optimization differs, and one held-out donor cannot establish patient replication.

[Detailed comparison and all seeds](../../../../reports/local_gene_residual/v1/report.md)

[Verification record](../../../../reports/local_gene_residual/verification_v1/README.md)

All 60 focused tests pass. The broad repository suite has 2109 passes, 35 failures
on missing historical inputs, and four skips; the preserved logs expose these
limits. Checkpoint replay and immutable-bundle validation passed for all four
full runs. No failed campaign attempts were omitted.
