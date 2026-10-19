# SO2 BCE optimization audit at matched target exposure

Outcome: **negative** at the fixed32-epoch budget.

Across model seeds 0 and 1, the incumbent-minus-comparator reduction in mean validation BCE at epochs 29–32 was 0.00005516 and 0.00031137; mean reduction 0.00018326. The prespecified material-gain criterion was not met. Endpoint convergence diagnostic: still improving in at least one run. Separately, the eight-epoch target-window incumbent reduced BCE versus paired cores by 0.00719035, with F1 change +0.03931919, at equal 1,669,568 target visits and 1,672 versus 48 optimizer updates. This early update-regimen result is distinct from the tuning comparison and does not establish eventual superiority.

Only one held-out validation donor, two optimization seeds, repeated validation selection and no sealed test. Exposure is matched; optimizer updates differ across batch sizes. At least one endpoint remains unconverged, so eventual-performance comparison is inconclusive. Graph-specific gain, attribution stability/faithfulness, biological nulls, patient replication and mechanism remain untested.

Exploratory validation-selected optimization behavior at the tested32-epoch budget on this split; no claim of global optimality, graph-specific information, independent patient replication or causality.

Detailed report and all immutable run sources are checksum-bound in result.yaml; tables/confirmation.json contains the four run-level endpoints.

Storage retention note (2026-10-16): the user-authorized cleanup removed 19 checkpoint files from the 19 historical BCE optimization runs referenced here. Recorded metrics, reports, configurations and provenance remain available; fresh inference or checkpoint replay requires regenerating the removed checkpoints. The exact files and registry tombstones are documented by the [retention receipt](../../../../reports/retention/so12_embeddings_cleanup_20261016/application_receipt.json) at `reports/retention/so12_embeddings_cleanup_20261016/application_receipt.json`. This storage change does not revise the scientific findings or their verification limitations.
