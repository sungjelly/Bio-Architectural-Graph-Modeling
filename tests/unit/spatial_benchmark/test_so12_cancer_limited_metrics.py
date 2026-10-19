import itertools
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import so12_cancer_limited_metrics as metrics


def test_selector_matches_independent_exhaustive_search_with_exact_ties():
    rng = np.random.default_rng(20261018)
    for _ in range(100):
        area = rng.integers(0, 100, 8)
        positive = np.array([rng.integers(0, a+1) for a in area])
        eligible = [int(i) for i in np.flatnonzero(area)]
        result = metrics.exact_limited_unions(area, positive)
        def ranked(candidate):
            p = sum(int(positive[i]) for i in candidate)
            denominator = sum(int(v) for v in positive)+sum(int(area[i]) for i in candidate)
            return (-Fraction(2*p, denominator or 1), len(candidate), candidate)
        for name, sizes in (("best_single", (1,)), ("best_pair", (2,)), ("best_at_most_two", (0,1,2))):
            candidates = [ids for size in sizes for ids in itertools.combinations(eligible, size)]
            expected = list(min(candidates, key=ranked)) if candidates else None
            assert result[name] == expected


def test_rational_comparison_avoids_float_tie_and_excludes_absent_clusters():
    # This one-pixel improvement disappears in double precision.
    n = 2**54
    result = metrics.exact_limited_unions(np.array([0,n,n], np.int64), np.array([0,n-1,n], np.int64))
    assert result["best_single"] == [2]
    assert result["best_pair"] == [1,2]
    assert result["eligible_cluster_ids"] == [1,2]
    tied = metrics.exact_limited_unions(np.array([2,8,0]), np.array([2,2,0]))
    assert tied["best_at_most_two"] == [0]  # singleton and pair both 2/3
    zero = metrics.exact_limited_unions(np.array([3,3,0]), np.zeros(3, dtype=int))
    assert zero["best_single"] == [0] and zero["best_pair"] == [0,1]
    assert zero["best_at_most_two"] == []
    absent = metrics.exact_limited_unions(np.array([0,3,0]), np.array([0,1,0]))
    assert absent["best_pair"] is None
    empty = metrics.exact_limited_unions(np.zeros(3, dtype=int), np.zeros(3, dtype=int))
    assert empty["best_single"] is None and empty["best_pair"] is None and empty["best_at_most_two"] == []


def counts():
    # Three visible labels per core. Cancer labels differ between cores.
    result = {}
    for core, owner, roi in ((15,[0,0,1,1,2,2,-1],[1,1,1,0,0,0,1]),
                             (23,[3,3,4,4,5,5,-1],[0,0,1,0,1,1,1])):
        owner, roi = np.array([owner],np.int32), np.array([roi],bool)
        result[core] = metrics.base.build_pixel_counts(owner,roi,np.ones_like(roi),n_cells=6)
    return result


def test_all_pixels_remain_in_denominator_and_transfer_is_source_selected():
    result = metrics.evaluate_partition(np.array([0,1,2,0,1,2]), counts())
    row = result["cores"]["15"]
    single, pair = row["best_single"], row["best_pair"]
    assert single["selected_clusters"] == [0]
    assert single["footprint"]["fn"] == 1  # unselected positive cell remains FN
    assert pair["selected_clusters"] == [0,1]
    assert pair["footprint"]["fp"] == 1  # selected spill outside ROI remains FP
    assert pair["footprint"]["evaluable_pixels"] == 6
    assert pair["full_region"]["fn"] == 1  # intercellular gap remains FN
    assert pair["full_region"]["evaluable_pixels"] == 7
    for support in ("footprint", "full_region"):
        s = pair[support]
        assert sum(s[k] for k in ("tp","fp","fn","tn")) == s["evaluable_pixels"]
    transferred = result["transfer"]["15_to_23"]
    assert not transferred["selection_uses_target_annotation"]
    assert transferred["best_single"]["selected_clusters"] == [0]
    assert transferred["best_single"]["footprint"]["dice_f1"] == 0
    assert result["cores"]["23"]["best_single"]["selected_clusters"] == [2]


def test_absent_pair_is_explicit_and_unknown_source_labels_stay_negative():
    c = counts()
    labels = np.array([4,4,4,0,1,4])
    result = metrics.evaluate_partition(labels,c)
    assert result["cores"]["15"]["best_pair"]["status"] == "unavailable"
    assert result["transfer"]["15_to_23"]["best_pair"]["footprint"] is None
    assert result["transfer"]["15_to_23"]["best_at_most_two"]["selected_clusters"] == [4]
    assert result["transfer"]["15_to_23"]["unseen_target_pixel_fraction"] == pytest.approx(4/6)
    with pytest.raises(ValueError,match="both core"):
        metrics.evaluate_partition(labels,{15:c[15],23:c[15]})


def test_sensitivity_freezes_selection_and_nulls_are_reproducible():
    labels,c = np.array([0,1,2,0,1,2]),counts()
    result = metrics.evaluate_partition(labels,c,null_repeats=20,null_seed=5)
    assert result == metrics.evaluate_partition(labels,c,null_repeats=20,null_seed=5)
    assert len(result["shuffled_label_control"]["results"]) == 20
    altered = {**c[15],"positive_per_cell":np.array([0,0,2,0,0,0]),
               "footprint_positive_pixels":2,"full_region_positive_pixels":2}
    changed = metrics.fixed_union_sensitivity(labels,result,{15:{"changed":altered}})["15"]["changed"]
    assert changed["selection_frozen_from_nominal_masks"]
    for kind in metrics.SELECTIONS:
        assert changed["oracle"][kind]["selected_clusters"] == result["cores"]["15"][kind]["selected_clusters"]
    assert changed["oracle"]["best_at_most_two"]["footprint"]["dice_f1"] == 0
    assert not result["shuffled_label_control"]["population_p_value_claimed"]


def synthetic_parent(tmp_path, monkeypatch):
    parent = tmp_path/"parent";parent.mkdir()
    (parent/"registration").mkdir();(parent/"clustering").mkdir();(parent/"metrics").mkdir()
    cells = pd.DataFrame({"row_index":range(6),"core_number":[15]*3+[23]*3})
    cells.to_csv(parent/"cells.csv.gz",index=False)
    cells_sha = metrics._sha(parent/"cells.csv.gz")
    records = []
    for core,owner,roi in ((15,[0,0,1,1,2,2,-1],[1,1,1,0,0,0,1]),
                            (23,[3,3,4,4,5,5,-1],[0,0,1,0,1,1,1])):
        owner,roi = np.array([owner],np.int32),np.array([roi],bool)
        path=parent/f"registration/core_{core}.npz"
        np.savez_compressed(path,owner=owner,roi=roi,valid=np.ones_like(roi),valid50=np.ones_like(roi),
                            roi_eroded50=np.zeros_like(roi),roi_dilated50=np.ones_like(roi))
        records.append({"core_number":core,"passed":True,"status":"verified_approximate_section",
                        "raster_path":str(path.relative_to(parent)),"raster_sha256":metrics._sha(path)})
    metrics._dump(parent/"registration/registration_manifest.json",{"canonical_cells":{"sha256":cells_sha},"cores":records})
    parts,old = [],[]
    labels = np.array([0,1,2,0,1,2],np.int32)
    for seed in (0,1):
        path=parent/f"clustering/s{seed}.npy";np.save(path,labels)
        identity={"method":"h0","resolution":1.,"seed":seed}
        parts.append({**identity,"labels":{"relative":True,"path":str(path.relative_to(parent)),"sha256":metrics._sha(path)}})
        old.append({**identity,"evaluation":metrics.base.evaluate_partition(labels,counts(),null_repeats=0)})
    metrics._dump(parent/"clustering_summary.json",{"passed":True,"all_methods_complete":True,"n_cells":6,"partitions":parts})
    metrics._dump(parent/"metrics/pixel_agreement.json",{"passed":True,"results":old,
        "code_sha256":metrics._sha(metrics.base.__file__),"input_sha256":{"cells.csv.gz":cells_sha}})
    files={str(f.relative_to(parent)):{"sha256":metrics._sha(f),"size_bytes":f.stat().st_size} for f in parent.rglob("*") if f.is_file()}
    metrics._dump(parent/"manifest.json",{"passed":True,"status":"complete","evaluation_id":"synthetic","files":files})
    digest=metrics._sha(parent/"manifest.json")
    metrics._dump(parent/"_SUCCESS",{"manifest_sha256":digest})
    monkeypatch.setattr(metrics,"PARENT_MANIFEST_SHA256",digest)
    monkeypatch.setattr(metrics,"EXPECTED_CELLS",6)
    monkeypatch.setattr(metrics,"METHODS",("h0",))
    monkeypatch.setattr(metrics,"RESOLUTIONS",(1.,))
    monkeypatch.setenv("BAGM_SCRATCH_ROOT",str(tmp_path/"scratch"))
    return parent,tmp_path/"scratch/active_runs/synthetic"


def test_runner_checks_frozen_parent_and_emits_complete_schema(tmp_path,monkeypatch):
    parent,output=synthetic_parent(tmp_path,monkeypatch)
    result=metrics.run(parent,output)
    assert result["passed"] and result["parent_inventory_verified_before_after"]
    assert len(result["results"]) == 2
    assert len(result["results"][0]["shuffled_label_control"]["results"]) == 20
    receipt=json.loads((output/"limited_metrics_summary.json").read_text())
    assert receipt["partition_count"] == 2
    assert len(pd.read_csv(output/"metrics/primary_summary.csv")) == 36
    assert len(pd.read_csv(output/"metrics/limited_union.csv")) == 72
    assert result["results"][0]["cores"]["15"]["unrestricted"]["selection"].startswith("frozen_parent")
    (parent/"clustering/s0.npy").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Parent file changed"):
        metrics.run(parent,output)


def test_canonical_row_binding_mismatch_is_rejected(tmp_path,monkeypatch):
    parent,output=synthetic_parent(tmp_path,monkeypatch)
    path=parent/"registration/registration_manifest.json"
    d=json.loads(path.read_text());d["canonical_cells"]["sha256"]="bad"
    metrics._dump(path,d)
    # Test the semantic guard independently of the preceding inventory guard.
    monkeypatch.setattr(metrics,"verify_parent",lambda p:{"evaluation_id":"synthetic"})
    with pytest.raises(ValueError,match="different canonical cells"):
        metrics.run(parent,output)
