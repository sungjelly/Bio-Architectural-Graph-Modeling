import itertools
import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import so12_cancer_metrics as metrics


def test_exact_union_matches_exhaustive_subsets_and_penalizes_spill():
    rng = np.random.default_rng(41)
    for _ in range(50):
        area = rng.integers(1, 50, 7)
        positive = np.array([rng.integers(0, n + 1) for n in area])
        reference = int(positive.sum() + 5)
        actual = metrics.optimal_union(area, positive, positive_pixels=reference)
        candidates = [0.] + [2 * positive[list(indices)].sum() / (reference + area[list(indices)].sum())
            for size in range(1, 8) for indices in itertools.combinations(range(7), size)]
        assert actual["scores"]["dice_f1"] == pytest.approx(max(candidates))
    result = metrics.optimal_union(np.array([10, 10, 100]), np.array([10, 10, 1]))
    assert result["selected_clusters"] == [0, 1]
    assert result["scores"]["fp"] == 0
    assert result["scores"]["dice_f1"] > metrics.binary_scores(21, 120, 21)["dice_f1"]


def _counts():
    # Core15: cluster0 is cancer; core23: cluster1 is cancer. Transfer must fail,
    # although each core separately has a perfect annotation-selected oracle.
    owner15 = np.array([[0, 0, 1, 1, -1]], dtype=np.int32)
    owner23 = np.array([[2, 2, 3, 3, -1]], dtype=np.int32)
    valid = np.ones_like(owner15, dtype=bool)
    return {
        15: metrics.build_pixel_counts(owner15, np.array([[1, 1, 0, 0, 1]], bool), valid, n_cells=4),
        23: metrics.build_pixel_counts(owner23, np.array([[0, 0, 1, 1, 1]], bool), valid, n_cells=4),
    }


def test_oracle_and_transfer_remain_distinct_and_full_region_counts_gaps():
    counts = _counts()
    result = metrics.evaluate_partition(np.array([0, 1, 0, 1]), counts, null_repeats=0)
    assert result["equal_core_oracle_dice"] == 1
    assert result["equal_core_transfer_dice"] == 0
    assert result["transfer"]["15_to_23"]["selected_clusters"] == [0]
    assert not result["transfer"]["15_to_23"]["selection_uses_target_annotation"]
    row = result["cores"]["15"]
    assert row["oracle"]["full_region"]["dice_f1"] == .8
    assert row["oracle"]["full_region"]["fn"] == 1
    assert row["coverage"]["full_region_pixel_coverage_dice_ceiling"] == .8
    assert row["empty"]["footprint"]["dice_f1"] == 0
    assert row["all_positive"]["footprint"]["fp"] == 2


def test_negative_controls_reselect_oracle_and_are_reproducible():
    labels, counts = np.array([0, 1, 0, 1]), _counts()
    result = metrics.evaluate_partition(labels, counts, null_repeats=20, null_seed=19)
    assert result == metrics.evaluate_partition(labels, counts, null_repeats=20, null_seed=19)
    null = result["shuffled_label_control"]
    assert null["oracle_reselected_each_repeat"] and len(null["results"]) == 20
    assert not null["population_p_value_claimed"]


def test_registration_withholding_and_support_errors_are_explicit():
    blocked = metrics.build_pixel_counts(None, None, None, n_cells=4,
        registration_status="blocked", registration_reason="section mismatch")
    assert blocked["status"] == "withheld"
    counts = _counts()
    counts[23] = blocked
    result = metrics.evaluate_partition(np.array([0, 1, 0, 1]), counts, null_repeats=0)
    assert result["cores"]["23"]["reason"] == "section mismatch"
    assert result["transfer"]["15_to_23"]["status"] == "withheld"
    assert result["equal_core_oracle_dice"] is None
    with pytest.raises(ValueError, match="Invalid global cell"):
        metrics.build_pixel_counts(np.array([[4]], np.int32), np.ones((1,1),bool), np.ones((1,1),bool), n_cells=4)
    with pytest.raises(ValueError, match="both core"):
        metrics.evaluate_partition(np.array([0, 1, 0, 1]), {15:_counts()[15],23:_counts()[15]}, null_repeats=0)
    assert metrics.binary_scores(0, 0, 0)["dice_f1"] is None


def test_metric_runner_covers_seeds_keeps_nominal_selection_and_withholds(tmp_path, monkeypatch):
    monkeypatch.setenv("BAGM_SCRATCH_ROOT", str(tmp_path))
    monkeypatch.setattr(metrics, "EXPECTED_CELLS", 4)
    monkeypatch.setattr(metrics, "METHODS", ("h0",))
    monkeypatch.setattr(metrics, "RESOLUTIONS", (1.,))
    output = tmp_path / "active_runs/synthetic"
    (output / "registration").mkdir(parents=True)
    (output / "clustering").mkdir()
    pd.DataFrame({"row_index":range(4),"core_number":[15,15,23,23]}).to_csv(output / "cells.csv.gz", index=False)
    parts = []
    for seed in (0,1):
        path = output / f"clustering/labels_s{seed}.npy"
        np.save(path, np.array([0,1,0,1], np.int32))
        parts.append({"method":"h0","resolution":1.,"seed":seed,
                      "labels":{"path":str(path.relative_to(output)),"relative":True,"sha256":metrics._sha(path)}})
    metrics._dump(output / "clustering_summary.json", {"passed":True,"status":"complete","n_cells":4,"partitions":parts})
    records = []
    for core, owner, roi in ((15,[0,0,1,1,-1],[1,1,0,0,1]),(23,[2,2,3,3,-1],[0,0,1,1,1])):
        owner, roi = np.array([owner],np.int32), np.array([roi],bool)
        path = output / f"registration/core_{core}.npz"
        np.savez_compressed(path, owner=owner, roi=roi, valid=np.ones_like(roi), valid50=np.ones_like(roi),
                            roi_eroded50=np.zeros_like(roi),roi_dilated50=np.ones_like(roi))
        records.append({"core_number":core,"passed":True,"status":"verified_approximate_section",
                        "raster_path":str(path.relative_to(output)),"raster_sha256":metrics._sha(path)})
    metrics._dump(output / "registration/registration_manifest.json", {"cores":records})
    result = metrics.run_metrics(output)
    assert result["pixel_scores_available"] and len(result["results"]) == 2
    sensitivity = result["results"][0]["evaluation"]["fixed_union_boundary_sensitivity"]["15"]
    assert sensitivity["dilate_roi_50um"]["selection_frozen_from_nominal_masks"]
    assert sensitivity["dilate_roi_50um"]["nominal_oracle_union"]["footprint"]["predicted_pixels"] == 2
    assert (output / "metrics/pixel_agreement.csv").is_file()
    records[1].update(passed=False,status="blocked",reason="section mismatch")
    metrics._dump(output / "registration/registration_manifest.json", {"cores":records})
    withheld = metrics.run_metrics(output)
    assert not withheld["pixel_scores_available"]
    assert withheld["results"][0]["evaluation"]["cores"]["23"]["status"] == "withheld"
