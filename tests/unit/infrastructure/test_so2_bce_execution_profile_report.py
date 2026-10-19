"""Aggregate reporting must not confuse attribution with elapsed CUDA time."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.analysis import report_so2_bce_execution_profile as report
from spatial_benchmark.paths import ProjectPaths


def operator(name,device,*,cpu=0.,cuda=0.,kind="operator"):
    return {"operator":name,"device_type":device,"kind":kind,"calls":3,
        "operator_full_label_sha256":hashlib.sha256(name.encode()).hexdigest(),
        "operator_label_truncated":False,"operator_label_control_characters_normalized":False,
        "self_cpu_microseconds":cpu,"inclusive_cpu_microseconds":cpu,
        "self_device_microseconds":cuda,"inclusive_device_microseconds":cuda}


def observation():
    updates=[{"update":i,"target_cells":1000 if i<209 else 696,"wall_seconds":2. if i in (3,4,5) else 1.,
              "profiled":i in (3,4,5)} for i in range(1,210)]
    profile={"schema":"so2_bce_execution_operator_profile_v1","profiled_update_indices":[3,4,5],
        "profile_coverage_complete":True,"capture_limited_by_budget":False,"raw_events_or_tensor_metadata_exported":False,
        "capture_wall_seconds":6.,"startup_wall_seconds":.5,"stop_and_aggregate_wall_seconds":.8,
        "aggregate_operator_rows":[operator("aten::linear","CPU",cpu=1.,cuda=100.),operator("kernel","CUDA",cuda=100.)]}
    summary={"schema":"so2_bce_execution_profile_summary_v1","all_update_timings":updates,
        "profiled_updates":updates[2:5],"adjacent_unprofiled_updates":updates[5:8],
        "profiled_mean_update_seconds":2.,"adjacent_unprofiled_mean_update_seconds":1.,
        "descriptive_profiled_to_adjacent_wall_ratio":2.,"profile_startup_seconds":.5,
        "profile_stop_and_aggregate_seconds":.8,"named_wall_ranges":[{"range":"training/parent","calls":209,
            "inclusive_blocking_wall_seconds":200.,"maximum_call_wall_seconds":2.}]}
    return profile,summary


def test_overlapping_cpu_attribution_and_cuda_kernel_rows_remain_separate():
    rows=[operator("aten::linear","CPU",cpu=8.,cuda=100.),operator("kernel","CUDA",cuda=100.),
          operator("execution/parent","CPU",cpu=1.,cuda=200.,kind="named_range"),
          operator("aten::item","CPU",cpu=3.)]
    groups=report.operator_groups(rows)
    assert [r["operator"] for r in groups["cpu_self"]]==["aten::linear","aten::item"]
    assert groups["cpu_attributed_device"]==[rows[0]]
    assert groups["device_activity_candidates"]==[rows[1]] and groups["named_ranges"]==[rows[2]]
    assert groups["scalar_sync_copy_candidates"]==[rows[3]]
    assert not any("total" in key or "fraction" in key for key in groups)


def test_recomputes_capture_windows_without_using_stored_ratio_as_authority():
    profile,summary=observation()
    assert report.validate_observation(profile,summary)["descriptive_profiled_to_adjacent_ratio"]==2.
    summary["descriptive_profiled_to_adjacent_wall_ratio"]=.2
    with pytest.raises(ValueError,match="calculation"):
        report.validate_observation(profile,summary)


@pytest.mark.parametrize("mutation",["missing_update","raw_export","missing_device_type","unexplained_partial"])
def test_rejects_incomplete_or_ambiguous_profile_evidence(mutation):
    profile,summary=observation()
    if mutation=="missing_update":summary["all_update_timings"].pop()
    if mutation=="raw_export":profile["raw_events_or_tensor_metadata_exported"]=True
    if mutation=="missing_device_type":profile["aggregate_operator_rows"][0]["device_type"]="unknown"
    if mutation=="unexplained_partial":
        profile["profiled_update_indices"]=[3];profile["profile_coverage_complete"]=False
    with pytest.raises(ValueError):report.validate_observation(profile,summary)


def test_parent_equivalence_is_recomputed_from_immutable_numeric_counts_and_masks():
    row={**{k:".25" for k in report.SCALARS},**{k:"10" for k in report.COUNTS},**{k:'{"mask":"abc"}' for k in report.MASKS}}
    stored={"passed":True,"absolute_scalar_tolerance":1e-6,"integer_counts_and_mask_receipts_exact":True,
        "scalar_comparisons":{k:{"observed":.25,"parent":.25,"absolute_difference":0.} for k in report.SCALARS}}
    report.check_parent_epoch(row,row,stored)
    changed={**row,"train_target_cells":"9"}
    with pytest.raises(ValueError,match="integer"):report.check_parent_epoch(changed,row,stored)
    changed={**row,"validation_coverage_receipts_json":'{}'}
    with pytest.raises(ValueError,match="mask/order"):report.check_parent_epoch(changed,row,stored)


def test_actual_sample_cannot_be_hidden_by_passed_resource_boolean():
    gate={"resource_gate_passed":True,"peak_allocated_bytes":2*report.GIB,"peak_reserved_bytes":3*report.GIB,
        "headroom_bytes":3*report.GIB,"minimum_sampled_physical_free_bytes":3*report.GIB,
        "maximum_sampled_cgroup_memory_bytes":50*report.GIB}
    receipt={"schema":"so2_bce_execution_profile_preflight_v1","passed":True,"completed_epochs":1,
        "coverage_verified":True,"finite_gradients":True,"optimizer_updates_per_global_epoch":209,
        "resource_gates":gate,"minimum_free_disk_gib":5}
    samples=[{"physical_memory_used_bytes":4*report.GIB,"physical_memory_free_bytes":4*report.GIB,
        "cgroup":{"memory.current":45*report.GIB},"process_peak_rss_bytes":2*report.GIB}]
    assert report.resource_summary(receipt,samples)["resource_gate_passed"]
    samples[0]["physical_memory_free_bytes"]=report.GIB
    with pytest.raises(ValueError,match="Physical headroom"):report.resource_summary(receipt,samples)


def test_figures_render_separate_views_and_full_tables(tmp_path):
    profile,summary=observation()
    groups=report.operator_groups(profile["aggregate_operator_rows"])
    report.figures(tmp_path,{"observation":summary},groups)
    assert {p.name for p in tmp_path.iterdir()}=={name+suffix for name in ("operators","wall_ranges","update_times") for suffix in (".png",".svg")}
    report.save_csv(tmp_path/"operators.csv",profile["aggregate_operator_rows"])
    assert "device_type" in (tmp_path/"operators.csv").read_text()


def test_active_run_is_rejected_before_bundle_or_history_access(tmp_path,monkeypatch):
    paths=ProjectPaths.from_environment({"BAGM_ROOT":str(tmp_path)})
    registry=SimpleNamespace(get_run=lambda _: {"status":"running"})
    def forbidden(*args,**kwargs):raise AssertionError("Active evidence was read")
    monkeypatch.setattr(report,"verify_run_bundle",forbidden)
    with pytest.raises(ValueError,match="registered completed"):
        report.checked_bundle(report.PARENT_RUN,paths,registry)


def test_replay_uses_existing_controller_tolerance():
    assert report.replay_close(.25,.2500009)
    assert not report.replay_close(.25,.250002)


def test_plot_labels_keep_long_kernel_names_distinguishable():
    rows=[operator("kernel<"+"template,"*60+suffix,"CUDA",cuda=10.) for suffix in ("float>","double>")]
    assert len({report.plot_label(row) for row in rows})==2
    assert all(len(report.plot_label(row))<=65 for row in rows)
    assert all(row["operator_full_label_sha256"][:10] in report.plot_label(row) for row in rows)


def test_cuda_typed_framework_annotations_are_not_reported_as_native_kernel_events():
    rows=[operator("DistributedDataParallel.forward","CUDA",cuda=1300.),
          operator("Optimizer.step#AdamW.step","CUDA",cuda=10.),
          operator("execution/receiver_chunk","CUDA",cuda=1500.,kind="named_range"),
          operator("Memcpy HtoD (Pageable -> Device)","CUDA",cuda=200.),
          operator("void templated_kernel<float>()","CUDA",cuda=80.)]
    groups=report.operator_groups(rows)
    assert groups["device_activity_candidates"]==rows[3:]
    assert {row["operator"] for row in groups["device_annotation_candidates"]}=={row["operator"] for row in rows[:3]}
