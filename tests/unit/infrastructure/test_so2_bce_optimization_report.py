"""Synthetic tests of prespecified endpoint selection and reporting scope."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def report():
    path = Path(__file__).resolve().parents[3]/"scripts/analysis/report_so2_bce_optimization_audit.py"
    spec = importlib.util.spec_from_file_location("audit_report_test",path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def row(report,run_id,bce,*,batch=1024,lr=1e-4,dropout=.1,wd=1e-5,seed=0,seconds=100,profile="screen",unconverged=False):
    return {"run_id":run_id,"profile":profile,**report.INCUMBENT,"target_batch_size":batch,"learning_rate":lr,
        "dropout":dropout,"weight_decay":wd,"seed":seed,"endpoint_bce":bce,"epoch_time_seconds":seconds,
        "selected_best":{"bce":.01},"endpoint_secondary":{"detection_f1":.2},
        "endpoint_per_core":{"SO2-C27":{"bce":bce+.001},"SO2-C28":{"bce":bce-.001}},
        "endpoint_unconverged":unconverged}


def test_grid_expects_all_nine_cells_and_paired_reference(report):
    expected = report.expected_rows("grid",None)
    assert len(expected)==10
    assert len({report.run_key(x) for x in expected})==10
    assert sum(x["batch_mode"]=="core_pair" for x in expected)==1
    assert {x["seed"] for x in expected}=={0}


def test_rank_uses_terminal_metric_then_declared_tolerance_and_time(report):
    best = row(report,"best",.25,seconds=200)
    tied = row(report,"tied",.25009,seconds=100,batch=512)
    outside = row(report,"outside",.25011,seconds=1,batch=2048)
    assert report.rank([best,tied,outside])["run_id"]=="tied"
    outside["selected_best"]["bce"] = 0
    assert report.rank([best,outside])["run_id"]=="best"


@pytest.mark.parametrize("gain,lr,boundary",[(.0019,3e-5,None),(.0021,3e-5,1e-5),(.0021,3e-4,1e-3),(.0021,1e-4,None)])
def test_boundary_expansion_requires_declared_material_gain(report,gain,lr,boundary):
    incumbent = row(report,"incumbent",.26)
    winner = row(report,"winner",.26-gain,batch=512,lr=lr)
    result = report.stage_decision("grid",[incumbent,winner],None,True,[incumbent,winner])
    assert result["grid_selected_run_id"]=="winner"
    assert (result["boundary_setting"] or {}).get("learning_rate")==boundary


def test_incomplete_matrix_produces_no_selection(report):
    result = report.stage_decision("grid",[row(report,"incumbent",.26)],None,False,[])
    assert result=={"selection_status":"incomplete_no_selection"}


def test_incumbent_win_uses_distinct_best_non_incumbent_from_complete_pool(report):
    incumbent = row(report,"incumbent",.25)
    alternative = row(report,"grid_alternative",.251,batch=512)
    dropout = row(report,"dropout",.252,dropout=0.)
    parent = {"stage_run_ids":["incumbent","grid_alternative"]}
    result = report.stage_decision("regularization",[incumbent,dropout],parent,True,[incumbent,alternative,dropout])
    assert result["incumbent_won_screen"] is True
    assert result["selected_finalist_run_id"]=="incumbent"
    assert result["confirmation_comparator_run_id"]=="grid_alternative"
    assert report.expected_rows("confirmation",result)==[
        {"profile":"confirmation",**report.INCUMBENT,"seed":0,"fold":0},
        {"profile":"confirmation",**report.INCUMBENT,"seed":1,"fold":0},
        {"profile":"confirmation",**report.setting(alternative),"seed":0,"fold":0},
        {"profile":"confirmation",**report.setting(alternative),"seed":1,"fold":0}]


@pytest.mark.parametrize("effects,supported",[([.003,.002],True),([.004,-.0001],False),([.001,.001],False)])
def test_confirmation_uses_both_seed_directions_and_mean_margin(report,effects,supported):
    rows = []
    for seed,effect in enumerate(effects):
        rows.extend([row(report,"inc"+str(seed),.25,seed=seed,profile="confirmation"),
                     row(report,"alt"+str(seed),.25-effect,seed=seed,batch=512,profile="confirmation",unconverged=seed==1)])
    result = report.compare_confirmation(rows)
    assert result["mean_paired_bce_reduction"]==pytest.approx(sum(effects)/2)
    assert result["material_gain_supported"] is supported
    assert result["paired_seed_effect_range"]==pytest.approx([min(effects),max(effects)])
    assert result["any_endpoint_unconverged"] is True
    assert result["eventual_performance_outcome"]=="inconclusive"
    assert result["confidence_interval"] is None
    assert len(result["paired_seed_effects"])==2


def test_failed_confirmation_does_not_claim_numeric_gain(report):
    rows = [row(report,"inc",.25,profile="confirmation")]
    result = report.stage_decision("confirmation",rows,{},True,rows)
    assert result["fixed_budget_outcome"]=="inconclusive"
    assert result["material_gain_supported"] is False
    assert result["mean_paired_bce_reduction"] is None


def test_parent_report_requires_complete_matching_stage_and_hashes(report,tmp_path):
    path = tmp_path/"summary.json"
    path.write_text(json.dumps({"schema":"so2_bce_optimization_audit_report_v1","campaign_id":report.CAMPAIGN,
                                "stage":"grid","required_coverage_complete":True}))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"files":{"summary.json":report.sha256_file(path)}}))
    loaded,reference = report.read_parent(path,"grid")
    assert loaded["required_coverage_complete"] is True
    assert reference["sha256"]==report.sha256_file(path)
    with pytest.raises(ValueError,match="incomplete or incompatible"):
        report.read_parent(path,"regularization")
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match="checksum"):
        report.read_parent(path,"grid")


def test_empty_stage_figures_mark_pending_without_implying_numeric_scores(report,tmp_path,monkeypatch):
    from matplotlib.figure import Figure
    figures = []
    monkeypatch.setattr(Figure,"savefig",lambda self,*args,**kwargs:figures.append(self))
    report.plot_curves(tmp_path,[],[])
    assert all(not axis.axison for axis in figures[-1].axes)
    assert all(any("Awaiting completed stage runs" in label.get_text() for label in axis.texts)
               for axis in figures[-1].axes)
    report.plot_endpoint_comparison(tmp_path,"grid",[])
    assert len(figures[-1].axes)==1, "An empty heatmap must not invent a numerical colorbar"
    assert len(figures[-1].axes[0].texts)==9
    assert all(label.get_text()=="No finalized\nscore" for label in figures[-1].axes[0].texts)


def test_resource_table_distinguishes_timing_and_shared_memory_scopes(report):
    item={"run_id":"synthetic_completed","physical_gpu":"3","epoch_time_seconds":120.,
        "elapsed_run_seconds":180.,"peak_allocated_gib":1.234,"peak_reserved_gib":2.345,
        "minimum_physical_free_gib":20.123,"peak_shared_cgroup_gib":55.678}
    before=dict(item)
    rendered="\n".join(report.resource_table([item]))
    assert "Total epoch time (min)" in rendered and "Elapsed run (min)" in rendered
    assert "| `synthetic_completed` | 3 | 2.0 | 3.0 | 1.234 / 2.345 | 20.123 |" in rendered
    assert "55.678 GiB" in rendered and "not per-job RAM" in rendered
    assert "worker publication/checkpoint indexing" in rendered
    assert "not a continuous physical-memory measurement" in rendered
    assert item==before and report.resource_table([])==[]
