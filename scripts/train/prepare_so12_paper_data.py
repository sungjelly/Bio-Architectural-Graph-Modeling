#!/usr/bin/env python3
"""Prepare and register the approved SO1/SO2 paper core split, CPU only."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from spatial_benchmark.paths import current_paths
from spatial_benchmark.registry import Registry
from spatial_benchmark.so12_paper_split_data import (
    CACHE_RELATIVE_PATH, DATASET_ID, DATASET_VERSION, PREPROCESSING_VERSION,
    SPLIT_ID, prepare_so12_paper_data,
)
from spatial_benchmark.so12_relative_qkv_data import _canonical_sha256


def main() -> None:
    paths = current_paths()
    manifest = prepare_so12_paper_data(paths)
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    reference = paths.data_root / CACHE_RELATIVE_PATH / "manifest.json"
    dataset = registry.register_dataset(
        DATASET_ID, DATASET_VERSION,
        display_name="SO1+SO2 train-fitted Pearson paper core split",
        protected_source_path=reference,
        raw_fingerprint=_canonical_sha256(manifest["source_files"]),
        preprocessing_version=PREPROCESSING_VERSION,
        processed_fingerprint=manifest["dataset_fingerprint"],
        aggregate_sample_count=407659, graph_count=28,
        node_feature_schema="1000 visible-count Pearson genes + 1000 mask indicators + 22 training-fitted morphology",
        edge_feature_schema="immutable coordinate-only relative geometry, 70 dimensions",
        creation_date="2026-10-17", verification_status="verified",
        metadata={"manifest": str(reference), "split_id": SPLIT_ID,
                  "materialized_cells": manifest["materialized_cells"],
                  "test_target_policy": manifest["test_target_policy"],
                  "preprocessing_fingerprint": manifest["preprocessing_fingerprint"],
                  "generalization_scope": "exploratory_core_section_holdout",
                  "donor_overlap_authorized": True},
    )
    split = registry.register_split(
        SPLIT_ID, dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
        method="user_approved_whole_core_three_way", unit="core_section",
        seed=None, fold_count=1, fingerprint=manifest["split_fingerprint"],
        protected_path=reference, verification_status="verified",
        metadata=manifest["split"],
    )
    print(json.dumps({"dataset_id": dataset["dataset_id"], "version": DATASET_VERSION,
                      "split_id": split["split_id"], "manifest": str(reference),
                      "dataset_fingerprint": manifest["dataset_fingerprint"],
                      "split_fingerprint": manifest["split_fingerprint"],
                      "materialized_cells": manifest["materialized_cells"],
                      "test_arrays_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
