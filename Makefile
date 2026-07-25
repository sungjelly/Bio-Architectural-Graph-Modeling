PYTHON ?= /venv/main/bin/python
BAGM := PYTHONPATH=src $(PYTHON) -m spatial_benchmark
CONFIG ?= configs/experiment/edge_feature_ablation_g2.yaml
CAMPAIGN_ID ?= cmp_20260724_edge_feature_ablation
CHECKPOINT_CATALOG_OUTPUT ?= exports/runs/checkpoints/catalog_manual

.PHONY: doctor test enqueue worker queue-status summarize leaderboard verify-artifacts import-legacy index-checkpoints checkpoints checkpoint-catalog

doctor:
	$(BAGM) doctor

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src $(PYTHON) -m pytest tests -q -p no:cacheprovider

enqueue:
	$(BAGM) enqueue-experiment --campaign-id $(CAMPAIGN_ID) --config $(CONFIG)

worker:
	$(BAGM) worker --gpu 0

queue-status:
	$(BAGM) list-queue

summarize:
	$(BAGM) summarize-variants

leaderboard:
	$(BAGM) export-leaderboard --output exports/leaderboards/leaderboard.csv

verify-artifacts:
	$(BAGM) verify-artifacts

index-checkpoints:
	$(BAGM) index-checkpoints

checkpoints:
	$(BAGM) list-checkpoints --limit 100

checkpoint-catalog:
	$(BAGM) export-checkpoint-catalog --output $(CHECKPOINT_CATALOG_OUTPUT) --links

import-legacy:
	$(BAGM) import-legacy --path artifacts/legacy_runs/lr_spatial_benchmark_batch --campaign-id cmp_legacy_normal_core_spatial_benchmark
