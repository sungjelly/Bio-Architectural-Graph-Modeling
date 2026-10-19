from fractions import Fraction
import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import so12_cancer_backward_metrics as metrics


def native_counts(areas, positives, *, gap_positive=0):
    owner=[];roi=[]
    for cell,(area,positive) in enumerate(zip(areas,positives)):
        owner.extend([cell]*area);roi.extend([True]*positive+[False]*(area-positive))
    owner.extend([-1]*gap_positive);roi.extend([True]*gap_positive)
    owner=np.array([owner],np.int32);roi=np.array([roi],bool)
    return metrics.base.build_pixel_counts(owner,roi,np.ones_like(roi),n_cells=len(areas))


def test_every_removal_matches_independent_fraction_enumeration():
    rng=np.random.default_rng(32)
    for _ in range(60):
        areas=rng.integers(1,30,8)
        positives=np.array([rng.integers(0,a+1) for a in areas])
        labels=rng.integers(0,6,8)
        counts=native_counts(areas,positives)
        path=metrics.backward_path(labels,counts)
        current=tuple(sorted(set(labels[positives>0])))
        total=int(positives.sum())
        def score(ids):
            selected=np.isin(labels,ids)
            return Fraction(2*int(positives[selected].sum()),max(1,total+int(areas[selected].sum())))
        expected=[current]
        while current:
            candidates=[tuple(i for i in current if i!=removed) for removed in current]
            current=min(candidates,key=lambda ids:(-score(ids),ids))
            expected.append(current)
        assert [tuple(s['selected_clusters']) for s in path['steps']]==expected
        best=min(range(len(expected)),key=lambda i:(-score(expected[i]),len(expected[i]),expected[i]))
        assert path['best_step_index']==best
        for step in path['steps']:
            assert step['footprint']['evaluable_pixels']==int(areas.sum())
            assert step['roi_cells_total']==int((positives>0).sum())


def test_deletion_ties_and_best_path_ties_have_distinct_rules():
    path=metrics.backward_path(np.arange(3),native_counts([2,2,2],[1,1,1]))
    assert [s['selected_clusters'] for s in path['steps']]==[[0,1,2],[0,1],[0],[]]
    assert [s['removed_cluster'] for s in path['steps']]==[None,2,1,0]
    tied=metrics.backward_path(np.arange(2),native_counts([2,6],[2,2]))
    assert tied['steps'][0]['footprint']['dice_f1']==tied['steps'][1]['footprint']['dice_f1']
    assert tied['best_step_index']==1
    assert tied['steps'][1]['selected_clusters']==[0]


def test_cell_retention_is_unweighted_and_fixed_support_penalizes_fp_fn():
    path=metrics.backward_path(np.arange(3),native_counts([10,100,10],[10,1,0],gap_positive=2))
    assert path['initial_overlap_cluster_ids']==[0,1]
    assert path['best_step_index']==1
    first,best,empty=path['steps']
    assert first['footprint']['fp']==99
    assert best['roi_cells_retained']==1 and best['roi_cells_total']==2
    assert best['roi_cells_retained_pct']==50
    assert best['footprint']['recall']==pytest.approx(10/11)
    assert best['majority_cells_retained_pct']==100
    assert best['selected_visible_cells_pct']==pytest.approx(100/3)
    assert best['cluster_retained_pct']==50
    assert best['footprint']['fn']==1 and best['full_region']['fn']==3
    assert all(s['footprint']['evaluable_pixels']==120 and s['full_region']['evaluable_pixels']==122 for s in path['steps'])
    assert path['best_retention90_step_index']==path['best_retention95_step_index']==0
    assert [s['pareto_nondominated'] for s in path['steps']]==[True,True,False]
    assert empty['roi_cells_retained_pct']==0
    assert path['greedy_gap_to_parent_unrestricted_f1']==0


def test_zero_overlap_and_exact_retention_threshold_edges():
    path=metrics.backward_path(np.array([0,1]),native_counts([2,2],[0,0]))
    assert len(path['steps'])==1 and path['best_step_index']==0
    assert path['best_retention90_step_index'] is None and path['best_retention95_step_index'] is None
    for key in ['cluster_retained_pct','roi_cells_retained_pct','majority_cells_retained_pct']:
        assert path['steps'][0][key] is None
    steps=[]
    for retained in (100,95,90,89):
        steps.append({'k':retained,'selected_clusters':list(range(retained)),
            'roi_cells_retained':retained,'roi_cells_total':100,
            'footprint':{'tp':100,'predicted_pixels':retained+100,'positive_pixels':100}})
    assert metrics.best_step_index(steps,90)==2
    assert metrics.best_step_index(steps,95)==1
    n=2**54
    assert metrics._compare({'footprint':{'tp':n,'predicted_pixels':n,'positive_pixels':2*n}},
                            {'footprint':{'tp':n-1,'predicted_pixels':n,'positive_pixels':2*n}})>0


def test_parent_unrestricted_replay_rejects_changed_counts():
    counts=native_counts([2,6],[2,2])
    parent=metrics.limited.evaluate_partition(np.array([0,1,0,1]),{
        15:metrics.base.build_pixel_counts(np.array([[0,0,1,1]],np.int32),np.array([[1,1,0,0]],bool),np.ones((1,4),bool),n_cells=4),
        23:metrics.base.build_pixel_counts(np.array([[2,2,3,3]],np.int32),np.array([[1,1,0,0]],bool),np.ones((1,4),bool),n_cells=4)})['cores']['15']['unrestricted']
    with pytest.raises(ValueError,match='Parent unrestricted score'):
        metrics.backward_path(np.arange(2),counts,parent_oracle=parent)


def test_runner_emits_all_paths_and_parent_bindings(tmp_path,monkeypatch):
    # Reuse the prior synthetic parent builder only; no real-data execution.
    import importlib.util
    from pathlib import Path
    fixture_path=Path(__file__).with_name('test_so12_cancer_limited_metrics.py')
    spec=importlib.util.spec_from_file_location('limited_synthetic_fixture',fixture_path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    parent,output=fixture.synthetic_parent(tmp_path,monkeypatch)
    monkeypatch.setattr(metrics,'EXPECTED_CELLS',6)
    monkeypatch.setattr(metrics,'METHODS',('h0',))
    monkeypatch.setattr(metrics,'RESOLUTIONS',(1.,))
    monkeypatch.setattr(metrics,'PARENT_MANIFEST_SHA256',metrics.limited.PARENT_MANIFEST_SHA256)
    result=metrics.run(parent,output)
    assert result['partition_count']==2 and result['core_path_count']==4
    assert result['parent_inventory_verified_before_after']
    assert result['cell_retention_unweighted'] and result['cell_retention_denominators_fixed']
    assert len(pd.read_csv(output/'metrics/backward_best_summary.csv'))==12
    assert len(pd.read_csv(output/'metrics/backward_primary_summary.csv'))==6
    receipt=json.loads((output/'backward_metrics_summary.json').read_text())
    assert len(receipt['output_sha256'])==4
    assert result['results'][0]['cores']['15']['parent_unrestricted_origin'].startswith('frozen_parent')
