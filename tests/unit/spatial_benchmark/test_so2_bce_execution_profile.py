"""CPU-only checks of additive observation, stopping and evidence boundaries."""
from copy import deepcopy
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from scripts.analysis import profile_so2_bce_execution as profile
from scripts.analysis import so2_bce_execution_profile_contract as contract
from spatial_benchmark.identifiers import canonical_sha256


@pytest.fixture
def configuration(monkeypatch):
    parent={"seed":0,"fold":0,"attempt":1,"campaign":{},"classification":{},"experiment":{},
        "source_configuration":{"synthetic":True},"evaluation":{},"metadata":{},
        "model":{"dropout":.1},"trainer":{"optimizer_batch_mode":"target_window",
            "target_cells_per_optimizer_update":1024,"optimizer_updates_per_global_epoch":209,
            "learning_rate":1e-4,"weight_decay":1e-5,"early_stopping_min_delta":1e-4},
        "launcher":{"cpu_threads":4,"disk_safety_min_free_gb":5}}
    monkeypatch.setattr(contract,"PARENT_CONFIG_SHA",canonical_sha256(parent))
    monkeypatch.setattr(contract,"validate_audit",lambda config:None)
    reference={"run_id":contract.PARENT_RUN,"configuration_sha256":canonical_sha256(parent),
        "success_marker_sha256":contract.PARENT_SUCCESS_SHA,"artifact_path":"/synthetic/parent",
        "receipt_path":"/synthetic/parent/diagnostics/optimization_audit_preflight.json",
        **{key:"a"*64 for key in ("config_file_sha256","manifest_sha256","checksum_manifest_sha256",
                                 "epoch_metrics_sha256","receipt_sha256","code_fingerprint_sha256")}}
    return contract.build_configuration(parent,reference,gpu="0",task_contract_sha256="b"*64,
        instrumentation_hashes={path:"c"*64 for path in contract.SCRIPT_PATHS})


@pytest.mark.parametrize("group,key,value",[
    ("trainer","learning_rate",3e-4),("trainer","target_cells_per_optimizer_update",2048),
    ("trainer","max_epochs",2),("trainer","minimum_global_epochs",1),
    ("model","dropout",0.),("launcher","disk_safety_min_free_gb",1),
    ("metadata","execution_profile","smoke"),
])
def test_guard_rejects_scientific_resource_and_semantic_drift(configuration,group,key,value):
    contract.validate_configuration(configuration)
    changed=deepcopy(configuration); changed[group][key]=value
    with pytest.raises(ValueError,match="changed its pinned"):
        contract.validate_configuration(changed)


def test_installed_one_epoch_controller_and_required_predictive_outputs(configuration):
    modules=(profile,profile.audit,profile.base,profile.bce,profile.minibatch)
    snapshots=[(module,dict(vars(module))) for module in modules]
    try:
        profile.install_contract(configuration)
        base=profile.base
        state=base.update_early_stopping(base.EarlyStoppingState(),.3,completed_epoch=1,
            minimum_epochs=base.MINIMUM_EPOCHS,maximum_epochs=base.MAXIMUM_EPOCHS,
            patience=base.EARLY_STOPPING_PATIENCE,min_delta=base.EARLY_STOPPING_MIN_DELTA)
        assert state.should_stop and state.stop_reason=="maximum_epochs"
        assert base.WORLD_SIZE==1 and base.OPTIMIZER_UPDATES_PER_EPOCH==209
        assert base.MINIMUM_EPOCHS==0 and base.MAXIMUM_EPOCHS==1
        assert base.MINIMUM_FREE_DISK_GIB==5
        assert profile.PROFILE_OUTPUT in base.ADDITIONAL_REQUIRED_FINAL_OUTPUTS
        assert profile.SUMMARY_OUTPUT in base.ADDITIONAL_REQUIRED_FINAL_OUTPUTS
        assert profile.PREFLIGHT_PATH in base.ADDITIONAL_REQUIRED_FINAL_OUTPUTS
        assert "results/execution_profile_endpoint.json" in base.ADDITIONAL_REQUIRED_FINAL_OUTPUTS
    finally:
        if profile._OBSERVER is not None:
            profile._OBSERVER.restore()
        for module,snapshot in snapshots:
            for key in set(vars(module))-set(snapshot):
                delattr(module,key)
            vars(module).update(snapshot)


def test_wrappers_preserve_optimizer_rng_values_and_exceptions(configuration):
    def execute(observed):
        torch.manual_seed(891)
        model=torch.nn.Sequential(torch.nn.Linear(5,4),torch.nn.Dropout(.1),torch.nn.Linear(4,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
        observer=profile.Observer(configuration)
        original=model.forward
        if observed:
            observer.wrap(model,"forward","synthetic_forward")
        losses=[]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            observer.active=observed
            for _ in range(3):
                x=torch.randn(7,5); y=torch.randn(7,1)
                optimizer.zero_grad(set_to_none=True)
                loss=(model(x)-y).square().mean();loss.backward();optimizer.step()
                losses.append(loss.item())
        observer.restore()
        assert model.forward==original
        if observed:
            rows=profile.aggregate_events(profiler.key_averages())
            assert any(row["operator"]=="execution/synthetic_forward" for row in rows)
            assert all(row["device_type"]=="CPU" for row in rows)
        return losses,[parameter.detach().clone() for parameter in model.parameters()],torch.get_rng_state()
    plain,observed=execute(False),execute(True)
    assert plain[0]==observed[0]
    assert all(torch.equal(a,b) for a,b in zip(plain[1],observed[1]))
    assert torch.equal(plain[2],observed[2])
    error=RuntimeError("synthetic expected error")
    def fail(): raise error
    obj=SimpleNamespace(call=fail);observer=profile.Observer(configuration)
    observer.wrap(obj,"call","failure",phase="validation")
    with pytest.raises(RuntimeError) as caught: obj.call()
    assert caught.value is error and observer.phase=="initialization"
    assert observer.ranges["validation/failure"]["calls"]==1
    observer.restore();assert obj.call is fail


@pytest.mark.parametrize("limited",[False,True])
def test_capture_closes_named_scope_before_stopping_and_exports_only_aggregates(configuration,monkeypatch,limited):
    counter=[0.]
    def clock(): counter[0]+=.01;return counter[0]
    observer=profile.Observer(configuration,clock=clock)
    scopes=[];written={};events=[]
    @contextmanager
    def scope(name):
        scopes.append(name)
        try:yield
        finally:scopes.pop()
    class FakeProfiler:
        def start(self): assert not scopes
        def stop(self): assert not scopes
        def key_averages(self):
            return [SimpleNamespace(key="aten::add",device_type="DeviceType.CPU",count=3,self_cpu_time_total=2.,cpu_time_total=3.,
                                    self_device_time_total=4.,device_time_total=5.)]
    monkeypatch.setattr(profile.torch.profiler,"profile",lambda **kwargs:FakeProfiler())
    monkeypatch.setattr(profile.torch.profiler,"record_function",scope)
    monkeypatch.setattr(profile,"_ORIGINAL_PROGRESS",lambda event,**kw:events.append(event))
    monkeypatch.setattr(profile.RunArchive,"attach_active",lambda *a,**kw:SimpleNamespace(write_json=lambda p,v:written.setdefault(p,v)))
    monkeypatch.setenv("BAGM_RUN_ID","synthetic")
    observer.progress("epoch_training_started");observer.progress("core_started")
    for update in range(1,9):
        if limited and update==3:
            counter[0]+=91.
        observer.progress("target_update_completed",update=update,target_cells=1024,
                          core_alias="not_exported",target_start=100,target_stop=1124)
    expected=[3] if limited else [3,4,5]
    assert observer.profiled_updates==expected and observer.profiler is None
    assert [row["update"] for row in observer.updates if row["profiled"]]==expected
    payload=written[profile.PROFILE_OUTPUT]
    assert payload["profile_coverage_complete"] is not limited
    assert payload["capture_limited_by_budget"] is limited
    assert not payload["raw_events_or_tensor_metadata_exported"]
    assert "target_start" not in json.dumps(payload) and "not_exported" not in json.dumps(observer.updates)
    assert len(events)==10


def test_parent_comparison_fails_on_scalar_count_or_mask_drift():
    scalars=("train_pooled_masked_binary_cross_entropy","validation_pooled_masked_binary_cross_entropy",
        "validation_detection_brier_score","validation_detection_f1","validation_detection_auprc_histogram_4096",
        "validation_detection_auroc_histogram_4096")
    counts=("train_target_cells","validation_target_cells","train_masked_entries","validation_masked_entries",
        "train_positive_entries","validation_positive_entries","optimizer_updates","cumulative_optimizer_updates",
        "validation_detection_true_positive","validation_detection_true_negative",
        "validation_detection_false_positive","validation_detection_false_negative")
    row={**{key:".3" for key in scalars},**{key:"5" for key in counts},
        **{key:'{"synthetic":"receipt"}' for key in ("core_order_json","training_coverage_receipts_json","validation_coverage_receipts_json")}}
    assert profile.parent_epoch_comparison(row,row,1e-6)["passed"]
    for key,value in ((scalars[0],".300002"),(counts[0],"6"),("validation_coverage_receipts_json",'{}')):
        with pytest.raises(profile.base.SO2NBTrainingError,match="mismatch"):
            profile.parent_epoch_comparison({**row,key:value},row,1e-6)


def test_long_templated_kernel_names_are_bounded_and_distinguished_by_full_name_sha():
    prefix="legitimate_cuda_kernel<"+"nested_template_parameter,"*80
    events=[SimpleNamespace(key=prefix+suffix,device_type="DeviceType.CUDA",count=1,
        self_cpu_time_total=0.,cpu_time_total=0.,self_device_time_total=3.,device_time_total=3.)
        for suffix in ("float>","double>")]
    rows=profile.aggregate_events(events)
    assert all(len(row["operator"])<=500 and row["operator_label_truncated"] for row in rows)
    assert len({row["operator"] for row in rows})==2
    assert {row["operator_full_label_sha256"] for row in rows}=={hashlib.sha256(event.key.encode()).hexdigest() for event in events}
    assert all(not row["operator_label_control_characters_normalized"] for row in rows)
