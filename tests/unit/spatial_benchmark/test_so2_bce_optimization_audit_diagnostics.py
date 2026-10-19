"""Synthetic execution of selected-best training diagnostics and final wiring."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from spatial_benchmark.binary_expression import TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer


@pytest.fixture
def audit():
    module = importlib.import_module("scripts.train.run_so2_bce_optimization_audit")
    snapshots = [(item, dict(vars(item))) for item in (module, module.minibatch, module.bce, module.base)]
    try:
        yield module
    finally:
        for item, snapshot in snapshots:
            for name in tuple(vars(item)):
                if name not in snapshot:
                    delattr(item, name)
            vars(item).update(snapshot)


def _core(n, seed):
    generator = torch.Generator().manual_seed(seed)
    receiver = torch.arange(n).repeat_interleave(n-1)
    sender = torch.tensor([s for r in range(n) for s in range(n) if s != r])
    return SimpleNamespace(alias=f"SYNTHETIC-{seed}", n_nodes=n, n_genes=4,
        input_expression=torch.randn(n,4,generator=generator),
        raw_count_target=(torch.arange(n*4).reshape(n,4)%3).to(torch.int32),
        node_covariates=torch.randn(n,2,generator=generator),
        edge_index=torch.stack((sender,receiver)),
        relative_geometry=torch.randn(len(sender),70,generator=generator))


def test_fixed_training_diagnostic_executes_real_eval_without_changing_model(audit, monkeypatch, tmp_path):
    torch.manual_seed(101)
    model = TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer(
        num_genes=4,node_covariate_dim=2,hidden_dim=8,attention_heads=2,attention_head_dim=4,
        graph_layers=4,ffn_dim=16,decoder_dim=12,geometry_hidden_dim=6,dropout=.3,
        attention_dropout=0,receiver_chunk_size=4,max_edges_per_chunk=32,
        activation_checkpointing=True,detection_initial_nonzero_probability=torch.tensor([.2,.4,.6,.8]))
    with torch.no_grad():
        model.detection_head.weight.normal_(0,.2)
    model.train()
    cores = (_core(7,31),_core(11,47))
    audit._MODEL = model
    audit._BUNDLE = SimpleNamespace(training_batches=cores)
    audit._CONFIG = {"trainer":{"amp":False,"amp_dtype":"auto"}}
    audit.minibatch._SCRATCH = tmp_path
    (tmp_path/"checkpoints").mkdir()
    (tmp_path/"checkpoints/best.ckpt").write_bytes(b"synthetic checksum fixture; model supplied separately")
    monkeypatch.setattr(audit.base,"WORLD_SIZE",1)
    resources = []
    monkeypatch.setattr(audit.minibatch,"_resource_snapshot",lambda device: resources.append(str(device)))
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    before = {key:value.clone() for key,value in model.state_dict().items()}
    modes = []
    hook = model.register_forward_pre_hook(lambda m,args:modes.append((m.training,torch.is_grad_enabled())))
    result = audit._fixed_training_evaluation()
    hook.remove()
    assert modes == [(False,False),(False,False)]
    assert model.training
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert result["checkpoint_sha256"] == audit.minibatch.sha256_file(tmp_path/"checkpoints/best.ckpt")
    assert result["used_for_model_selection"] is False
    assert result["metrics"]["target_cells"] == 18
    assert set(result["per_core"]) == {c.alias for c in cores}
    assert len(resources) == len(cores)
    assert result["mask_contract"] == "fixed_original_training_epoch_one_masks"
    expected_sum = 0.; masked_entries = 0
    model.eval()
    with torch.no_grad():
        for core in cores:
            partition, realization = audit.base._partition_and_mask(core,rank=0,global_epoch=1)
            assert result["mask_receipts"][core.alias] == realization.to_receipt()
            mask = torch.from_numpy(realization.mask.copy())
            output = model(input_expression=core.input_expression,
                target_input_expression=core.input_expression.masked_fill(mask,0),target_gene_mask=mask,
                edge_index=core.edge_index,relative_geometry=core.relative_geometry,node_covariates=core.node_covariates,
                target_start=0,target_stop=core.n_nodes)
            expected_sum += float(F.binary_cross_entropy_with_logits(output.nonzero_logits[mask],
                (core.raw_count_target[mask]>0).float(),reduction="sum"))
            masked_entries += int(mask.sum())
    assert result["metrics"]["masked_entries"] == masked_entries
    assert result["metrics"]["pooled_masked_binary_cross_entropy"] == pytest.approx(expected_sum/masked_entries,abs=1e-7)
    reference = result["train_fitted_gene_prevalence_reference"]
    assert reference["masked_entries"] == masked_entries
    assert result["bce_reduction_from_train_fitted_gene_prevalence"] == pytest.approx(
        reference["pooled_masked_binary_cross_entropy"]-expected_sum/masked_entries,abs=1e-7)
    json.dumps(result,allow_nan=False)


def test_final_artifacts_keep_terminal_endpoint_prior_and_training_diagnostic_separate(audit, monkeypatch, tmp_path):
    G = audit.minibatch.GIB
    audit.minibatch._SCRATCH = tmp_path
    audit.minibatch._DEVICE_TOTAL = 24*G
    audit.minibatch._MAX_CGROUP_MEMORY = 40*G
    audit.minibatch._SAMPLER = None
    audit.minibatch._INITIALIZATION_RECEIPT = {"code_hashes":{"synthetic.py":"a"*64}}
    audit.base.OPTIMIZER_UPDATES_PER_EPOCH = 209
    audit._CONFIG = {"metadata":{"execution_profile":"confirmation","evaluate_fixed_training_at_best":True,
                                 "frozen_task_contract_sha256":"b"*64},
        "trainer":{"max_epochs":32,"optimizer_batch_mode":"target_window","target_cells_per_optimizer_update":1024},
        "source_configuration":{"synthetic":True},"experiment":{"screening_endpoint_epochs":[29,30,31,32]}}
    receipts = {"SYNTHETIC":{"mask":"fixed"}}
    audit._BASELINE = {"validation_coverage_receipts_sha256":audit.base._canonical_sha256(receipts),
                      "metrics":{"pooled_masked_binary_cross_entropy":.4}}
    validation = SimpleNamespace(coverage_receipts=receipts,pooled_binary_cross_entropy=.1)
    monkeypatch.setenv("BAGM_RUN_ID","r_synthetic")
    monkeypatch.setattr(audit.bce,"_write_additional_final_artifacts",lambda *_:None)
    monkeypatch.setattr(audit,"_fixed_training_evaluation",lambda:{"synthetic_probe":True,"used_for_model_selection":False})
    resources = {"peak_allocated_bytes":8*G,"peak_reserved_bytes":10*G,
                 "minimum_sampled_physical_free_bytes":13*G,"maximum_sampled_cgroup_memory_bytes":40*G}
    monkeypatch.setattr(audit.minibatch,"_resource_snapshot",lambda _:dict(resources))
    (tmp_path/"results").mkdir();(tmp_path/"metrics").mkdir();(tmp_path/"diagnostics").mkdir()
    with (tmp_path/"results/epoch_metrics.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["global_epoch","validation_pooled_masked_binary_cross_entropy",
            "cumulative_optimizer_updates","train_target_cells"]);writer.writeheader()
        for epoch in range(1,33):
            writer.writerow({"global_epoch":epoch,"validation_pooled_masked_binary_cross_entropy":.1 if epoch==1 else .2+epoch*.001,
                "cumulative_optimizer_updates":epoch*209,"train_target_cells":208696})
    (tmp_path/"metrics/target_minibatch_resources.jsonl").write_text("".join(json.dumps({**resources,
        "epoch":epoch,"optimizer_updates":209})+"\n" for epoch in range(1,33)))
    (tmp_path/audit.RESOURCE_SAMPLES_PATH).write_text(json.dumps({"physical_memory_free_bytes":13*G,
        "cgroup":{"memory.current":40*G}})+"\n")
    written={}
    archive=SimpleNamespace(write_json=lambda name,payload:written.__setitem__(name,payload))
    audit._write_final_artifacts(archive,validation)
    assert written[audit.BASELINE_PATH] == audit._BASELINE
    assert written[audit.TRAIN_EVALUATION_PATH]["synthetic_probe"]
    endpoint=written[audit.ENDPOINT_PATH]
    assert endpoint["mean_validation_bce"] == pytest.approx(.2305)
    assert endpoint["selected_best_checkpoint_bce"] == .1
    assert endpoint["training_target_visits"] == 32*208696
    assert endpoint["optimizer_updates_at_final_epoch"] == 32*209
    gate=written[audit.PREFLIGHT_PATH]
    assert gate["completed_epochs"] == 32 and gate["passed"]
    assert gate["resource_gates"]["resource_gate_passed"]
    assert gate["optimizer_batch_mode"] == "target_window"
    json.dumps(written,allow_nan=False)
