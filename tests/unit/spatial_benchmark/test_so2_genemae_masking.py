import copy
import numpy as np
import pytest
import torch

from spatial_benchmark.myjju_genemae import make_source_model, build_symmetric_knn_graph
from spatial_benchmark.queueing import command_for_config
from spatial_benchmark.so2_genemae_masking import (
    METHODS, make_mask, neighbor_mean, visible_mean, ReconstructionMetrics, validate_protocol,
    resource_gates,
)
from scripts.train.prepare_so2_genemae_masking import make_config


def test_resource_receipt_serializes_numpy_comparison_values():
    from spatial_benchmark.identifiers import canonical_json
    gates = resource_gates(np.float64(9), np.float64(5), np.float64(.5), np.float64(18), np.bool_(True))
    assert all(type(v) is bool for v in gates.values())
    assert '"runtime":true' in canonical_json(gates)
    assert resource_gates(22,5,1,18,True)["vram"] is False


@pytest.mark.parametrize("method", METHODS)
def test_exact_mask_semantics_and_reproducibility(method):
    coords = np.column_stack([np.arange(103), np.zeros(103)])
    mask = make_mask(coords, 1000, method, 123)
    assert np.array_equal(mask, make_mask(coords, 1000, method, 123))
    assert not np.array_equal(mask, make_mask(coords, 1000, method, 124))
    if method == "partial_gene":
        assert np.all(mask.sum(1) == 100)
    else:
        assert np.all((mask.sum(1) == 0) | (mask.sum(1) == 1000))
        assert mask[:,0].sum() == 10
        if method == "spatial_block":
            assert np.all(np.diff(np.flatnonzero(mask[:,0])) == 1)


@pytest.mark.parametrize("method", METHODS)
def test_target_invariance_model_and_baselines(method):
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(50,2))
    x = rng.uniform(size=(50,20)).astype(np.float32)
    mask = make_mask(coords, 20, method, 44)
    e,a = build_symmetric_knn_graph(coords, 5)
    changed = x.copy(); changed[mask] = 999
    baseline = visible_mean(x,mask)
    np.testing.assert_array_equal(baseline, visible_mean(changed,mask))
    np.testing.assert_array_equal(neighbor_mean(x,mask,e,baseline), neighbor_mean(changed,mask,e,baseline))
    model = make_source_model(20, hidden=8, embed=8, heads=2, layers=2, self_hidden=16).eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x), torch.from_numpy(e), torch.from_numpy(a), torch.from_numpy(mask))[0]
        other = model(torch.from_numpy(changed), torch.from_numpy(e), torch.from_numpy(a), torch.from_numpy(mask))[0]
    assert torch.equal(pred,other)


def test_spatial_positive_control_and_expression_randomization():
    coords = np.column_stack([np.linspace(0,10,200), np.zeros(200)])
    x = np.column_stack([1+np.sin(coords[:,0]), 1+np.cos(coords[:,0])]).astype(np.float32)
    m = make_mask(coords,2,"whole_cell",12)
    edges,_ = build_symmetric_knn_graph(coords,5)
    baseline = visible_mean(x,m)
    smooth = neighbor_mean(x,m,edges,baseline)
    error = np.square(smooth[m]-x[m]).mean()
    assert error < .01 * np.square(np.broadcast_to(baseline,x.shape)[m]-x[m]).mean()
    shuffled = x[np.random.default_rng(99).permutation(len(x))]
    pred_null = neighbor_mean(shuffled,m,edges,visible_mean(shuffled,m))
    assert np.square(pred_null[m]-shuffled[m]).mean() > 10*error


def test_masked_loss_learns_constant_gene_reference():
    torch.set_num_threads(1); torch.manual_seed(21)
    coords = np.column_stack([np.arange(40), np.zeros(40)])
    x = torch.tensor(np.tile([[0.,1.]], (40,1)), dtype=torch.float32)
    e,a = build_symmetric_knn_graph(coords,4)
    e,a = torch.from_numpy(e), torch.from_numpy(a)
    m = torch.from_numpy(make_mask(coords,2,"whole_cell",123))
    model = make_source_model(2, hidden=8, embed=8, heads=2, layers=2, self_hidden=16, dropout=0., drop_edge_p=0.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    first = float(model.loss(x,e,a,m).detach())
    for _ in range(40):
        optimizer.zero_grad(); loss=model.loss(x,e,a,m); loss.backward(); optimizer.step()
    assert float(model.loss(x,e,a,m).detach()) < .1*first


def test_stratified_metrics_do_not_hide_positive_errors():
    y=np.array([[0.,0.,2.],[0.,1.,0.]],dtype=np.float32)
    metrics=ReconstructionMetrics(3)
    metrics.update(y,np.zeros_like(y),np.ones_like(y,dtype=bool))
    result=metrics.finalize()
    assert result["positive_count"] == 2
    assert result["zero_count"] == 4
    assert result["positive_mse"] == 2.5
    assert result["zero_mse"] == 0
    assert result["pooled_pearson"] is None


def test_config_queue_and_historical_protocol_separation():
    config=make_config("whole_cell",0,"a"*64,"b"*64,pilot=True)
    assert validate_protocol(config)
    assert "run_so2_genemae_masking.py" in command_for_config(config)[1]
    changed=copy.deepcopy(config); changed["masking"]["rate"]=.2
    with pytest.raises(ValueError,match="Protocol drift"):
        validate_protocol(changed)


def test_checkpoint_recovery_keeps_scientific_identity():
    from spatial_benchmark.identifiers import scientific_id
    config=make_config("partial_gene",1,"a"*64,"b"*64,pilot=False)
    recovered=copy.deepcopy(config)
    recovered["metadata"]["runtime"]={"checkpoint_recovery":{"source_run_id":"r_example"}}
    recovered["attempt"]=2
    assert validate_protocol(recovered) is False
    assert scientific_id(config)==scientific_id(recovered)
    assert "recover_so2_genemae_masking.py" in command_for_config(recovered)[1]
    assert "run_so2_genemae_masking.py" in command_for_config(config)[1]


def test_recovery_rejects_changed_scores_counts_and_missing_rows():
    import pandas as pd
    from scripts.train.recover_so2_genemae_masking import compare_evaluations
    rows=[{"train_method":"partial_gene","model_seed":1,"core":str(i),
           "evaluation_method":"whole_cell","replicate":0,"condition":"observed",
           "n_masked":100,"masked_huber":.04} for i in range(588)]
    original=pd.DataFrame(rows)
    valid=original.copy();valid.loc[0,"masked_huber"]+=1e-8
    assert compare_evaluations(original,valid)["all_metrics_reproduced"]
    for column,delta in (("masked_huber",.001),("n_masked",1)):
        changed=original.copy();changed.loc[0,column]+=delta
        with pytest.raises(AssertionError):compare_evaluations(original,changed)
    with pytest.raises(AssertionError):compare_evaluations(original,original.iloc[:-1])
