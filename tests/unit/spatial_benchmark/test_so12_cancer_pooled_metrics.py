from fractions import Fraction
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import so12_cancer_pooled_metrics as metrics


def core_counts(cell_ids,areas,positives,*,n_cells=4,gap_positive=0):
    owner,roi = [],[]
    for cell,a,p in zip(cell_ids,areas,positives):
        owner.extend([cell]*a);roi.extend([True]*p+[False]*(a-p))
    owner.extend([-1]*gap_positive);roi.extend([True]*gap_positive)
    owner,roi=np.array([owner],np.int32),np.array([roi],bool)
    return metrics.base.build_pixel_counts(owner,roi,np.ones_like(roi),n_cells=n_cells)


def counterexample():
    return {15:core_counts([0,1],[100,100],[90,10]),23:core_counts([2,3],[10,10],[0,10])}


def test_shared_micro_selection_differs_from_macro_and_independent_masks():
    labels=np.array([0,1,0,1])
    c=counterexample()
    path=metrics.pooled_path(labels,c)
    best=path['steps'][path['best_step_index']]
    assert best['selected_clusters']==[0]
    assert best['footprint']['dice_f1']==pytest.approx(180/220)
    assert best['diagnostic_macro_footprint_f1']==pytest.approx(.45)
    assert path['steps'][0]['diagnostic_macro_footprint_f1']==pytest.approx(2/3)
    assert best['core_components']['15']['selected_clusters']==best['core_components']['23']['selected_clusters']==[0]
    assert best['core_components']['23']['footprint']['fp']==10
    assert best['footprint']['fp']==20  # spills from both cores remain
    # Independent best masks select A on15 and B on23, an unauthorized higher score.
    independent=[metrics.backward.backward_path(labels,c[core]) for core in (15,23)]
    chosen=[p['steps'][p['best_step_index']] for p in independent]
    assert [s['selected_clusters'] for s in chosen]==[[0],[1]]
    tp=sum(s['footprint']['tp'] for s in chosen)
    pred=sum(s['footprint']['predicted_pixels'] for s in chosen)
    positive=sum(s['footprint']['positive_pixels'] for s in chosen)
    assert 2*tp/(pred+positive)==pytest.approx(200/220)
    assert 2*tp/(pred+positive)>best['footprint']['dice_f1']
    assert not path['per_core_optimized_masks_reused'] and not path['per_core_f1_averaged_for_selection']


def test_fixed_disjoint_cell_retention_and_full_region_reconcile():
    c=counterexample()
    c[23]=core_counts([2,3],[10,10],[0,10],gap_positive=3)
    path=metrics.pooled_path(np.array([0,1,0,1]),c)
    first,best,empty=path['steps']
    assert first['initial_overlap_clusters']==2  # shared IDs counted once
    assert best['cluster_retained_pct']==50
    assert best['roi_cells_total']==3 and best['roi_cells_retained']==1
    assert best['roi_cells_retained_pct']==pytest.approx(100/3)
    assert best['footprint']['recall']==pytest.approx(90/110)
    assert best['majority_cells_total']==2 and best['majority_cells_retained']==1
    assert best['selected_visible_cells']==2 and best['visible_cells_total']==4
    assert best['full_region']['fn']==best['footprint']['fn']+3
    assert empty['roi_cells_total']==3 and empty['roi_cells_retained']==0
    for step in path['steps']:
        for support in ('footprint','full_region'):
            for key in ('tp','fp','fn','tn','predicted_pixels','positive_pixels','evaluable_pixels'):
                assert step[support][key]==sum(v[support][key] for v in step['core_components'].values())
    assert path['pooled_unrestricted_f1']==best['footprint']['dice_f1']
    assert path['parent_unrestricted_origin']==metrics.ORACLE_ORIGIN
    assert path['greedy_gap_to_pooled_unrestricted_f1']==0


def test_pooling_rejects_overlapping_rows_and_inconsistent_support():
    c=counterexample()
    with pytest.raises(ValueError,match='disjoint'):
        metrics.pool_counts({15:c[15],23:c[15]})
    bad={**c[23],'footprint_pixels':21}
    with pytest.raises(ValueError,match='Footprint support'):
        metrics.pool_counts({15:c[15],23:bad})
    bad={**c[23],'full_region_positive_pixels':21,'uncovered_positive_pixels':11}
    with pytest.raises(ValueError,match='Full-region support'):
        metrics.pool_counts({15:c[15],23:bad})
    bad={**c[23],'n_cells':3}
    with pytest.raises(ValueError,match='global count rows'):
        metrics.pool_counts({15:c[15],23:bad})
    with pytest.raises(ValueError,match='Exactly cores'):
        metrics.pool_counts({15:c[15]})


def test_exact_pooled_unrestricted_oracle_matches_all_subsets_and_ties():
    rng=np.random.default_rng(314)
    for _ in range(60):
        area=rng.integers(0,30,7)
        positive=np.array([rng.integers(0,a+1) for a in area])
        eligible=tuple(map(int,np.flatnonzero(area)))
        total=sum(map(int,positive))
        def rank(ids):
            score=Fraction(2*sum(int(positive[i]) for i in ids),max(1,total+sum(int(area[i]) for i in ids)))
            return -score,len(ids),ids
        candidates=[ids for size in range(len(eligible)+1) for ids in itertools.combinations(eligible,size)]
        expected=list(min(candidates,key=rank))
        assert metrics.exact_unrestricted_ids(area,positive)==expected
    assert metrics.exact_unrestricted_ids(np.array([2,6]),np.array([2,2]))==[0]
    n=2**54
    assert metrics.exact_unrestricted_ids(np.array([n,n]),np.array([n,0]))==[0]


def test_zero_positive_pooled_support_keeps_empty_path_and_undefined_retention():
    c={15:core_counts([0,1],[2,2],[0,0]),23:core_counts([2,3],[2,2],[0,0])}
    path=metrics.pooled_path(np.array([0,1,0,1]),c)
    assert len(path['steps'])==1 and path['steps'][0]['selected_clusters']==[]
    assert path['best_step_index']==0 and path['best_retention90_step_index'] is None
    assert path['steps'][0]['roi_cells_retained_pct'] is None
    assert path['steps'][0]['diagnostic_macro_footprint_f1'] is None
    assert path['pooled_unrestricted_f1'] is None


def test_synthetic_runner_uses_one_path_per_partition_and_verifies_parent(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    fixture_path=Path(__file__).with_name('test_so12_cancer_limited_metrics.py')
    spec=importlib.util.spec_from_file_location('limited_parent_fixture',fixture_path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    parent,output=fixture.synthetic_parent(tmp_path,monkeypatch)
    monkeypatch.setattr(metrics,'EXPECTED_CELLS',6)
    monkeypatch.setattr(metrics,'METHODS',('h0',))
    monkeypatch.setattr(metrics,'RESOLUTIONS',(1.,))
    monkeypatch.setattr(metrics,'PARENT_MANIFEST_SHA256',metrics.limited.PARENT_MANIFEST_SHA256)
    result=metrics.run(parent,output)
    assert result['partition_count']==result['pooled_path_count']==2
    assert result['all_core_component_counts_reconciled']
    assert all('pooled' in row and 'cores' not in row for row in result['results'])
    assert len(pd.read_csv(output/'metrics/pooled_best_summary.csv'))==6
    assert len(pd.read_csv(output/'metrics/pooled_primary_summary.csv'))==3
    receipt=json.loads((output/'pooled_metrics_summary.json').read_text())
    assert len(receipt['output_sha256'])==4
    assert receipt['parent_inventory_verified_before_after']
    (parent/'clustering/s0.npy').write_bytes(b'altered')
    with pytest.raises(ValueError,match='Parent file changed'):
        metrics.run(parent,output)
