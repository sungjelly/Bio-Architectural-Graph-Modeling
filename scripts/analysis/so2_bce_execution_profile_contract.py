"""Strict additive one-epoch runtime-observation contract; no training edits."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Mapping

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.so2_bce_optimization_audit import validate_configuration as validate_audit

CAMPAIGN = "cmp_20261001_so2_bce_execution_profile"
PROTOCOL = "donor_grouped_so2_bce_execution_profile_v1"
PARENT_RUN = "r_20261001T073730Z_b3feecba_s000_f00_a01_42b8b0fd"
PARENT_CONFIG_SHA = "12bc44ce0d4e56a3864266c90c4233276967587c84627893f4e2fd6edaff9b31"
PARENT_SUCCESS_SHA = "ab824f4bf545c54f744cc788e59d6a4b570532efb1f1169792d0868a1db71724"
SCRIPT_PATHS = (
    "scripts/analysis/so2_bce_execution_profile_contract.py",
    "scripts/analysis/profile_so2_bce_execution.py",
    "scripts/analysis/launch_so2_bce_execution_profile.py",
)
PROFILE_SETTINGS = {"warmup_updates":2,"profiled_updates":3,"adjacent_unprofiled_updates":3,
    "max_profile_seconds_at_update_boundary":90,"soft_total_budget_seconds":600,
    "activities":["CPU","CUDA"],"record_shapes":False,"with_stack":False,
    "profile_memory":False,"with_flops":False,"with_modules":False,"export_raw_trace":False,
    "reset_state_ab_test":False,"parent_scalar_absolute_tolerance":1e-6}


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha(value: Any) -> bool:
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


def build_configuration(parent: Mapping[str,Any], reference: Mapping[str,Any], *, gpu: str,
                        task_contract_sha256: str, instrumentation_hashes: Mapping[str,str]) -> dict[str,Any]:
    config = deepcopy(dict(parent))
    config["source_profile_configuration"] = deepcopy(dict(parent))
    config["campaign"] = {"campaign_id":CAMPAIGN,"display_name":"SO2 BCE execution profile"}
    config["classification"].update(lifecycle_stage="diagnostic",study_axis="execution_profile",
        scientific_variant="bce_target_window1024_one_epoch_observation",retention_class="retain_runtime_profile_evidence")
    config["experiment"] = {"variant_label":"bce_target_window1024_one_epoch_observation",
        "estimand":"observed_runtime_costs_of_frozen_target_window_bce_updates",
        "permitted_claim":"exploratory_execution_costs_only", "fixed_budget_epochs":1,
        "screening_endpoint_epochs":[1],"screening_endpoint_aggregation":"single_diagnostic_epoch_validation",
        "validation_metric_is_checkpoint_metadata_only":True,
        "primary_runtime_measure":"observed_wall_milliseconds_per_update_and_aggregate_operator_times"}
    config["evaluation"]["protocol"] = PROTOCOL
    config["metadata"].update(execution_profile="runtime_profile", frozen_task_contract_sha256=task_contract_sha256,
        source_profile_configuration_sha256=PARENT_CONFIG_SHA, smoke_preflight=None,
        profile_parent=deepcopy(dict(reference)), instrumentation_hashes=dict(instrumentation_hashes),
        profiler=deepcopy(PROFILE_SETTINGS), evaluate_fixed_training_at_best=False)
    config["trainer"].update(max_epochs=1,minimum_global_epochs=0,early_stopping_patience=2)
    config["launcher"].update(requested_gpu=str(gpu),require_exact_visible_devices=str(gpu),
        hardware_preflight_receipt=reference.get("receipt_path"))
    config["attempt"] = 1
    return config


def validate_configuration(config: Mapping[str,Any]) -> None:
    parent = config.get("source_profile_configuration")
    _require(isinstance(parent,Mapping) and canonical_sha256(parent)==PARENT_CONFIG_SHA,
             "Execution profiling requires the pinned completed target-window smoke configuration")
    validate_audit(parent)
    metadata = config.get("metadata",{})
    _require(isinstance(metadata,Mapping),"Profiling metadata must be a mapping")
    reference = metadata.get("profile_parent")
    _require(isinstance(reference,Mapping) and reference.get("run_id")==PARENT_RUN
             and reference.get("configuration_sha256")==PARENT_CONFIG_SHA
             and reference.get("success_marker_sha256")==PARENT_SUCCESS_SHA,
             "Profiling parent successful bundle identity differs")
    for key in ("config_file_sha256","manifest_sha256","checksum_manifest_sha256","epoch_metrics_sha256",
                "receipt_sha256","code_fingerprint_sha256"):
        _require(_sha(reference.get(key)),"Profiling parent lacks "+key)
    _require(str(reference.get("receipt_path","")).endswith("/diagnostics/optimization_audit_preflight.json")
             and bool(reference.get("artifact_path")),"Profiling parent paths are incomplete")
    hashes = metadata.get("instrumentation_hashes")
    _require(isinstance(hashes,Mapping) and set(hashes)==set(SCRIPT_PATHS) and all(_sha(v) for v in hashes.values()),
             "All three profiling analysis scripts require exact checksums")
    contract_sha = metadata.get("frozen_task_contract_sha256")
    _require(_sha(contract_sha),"Profiling requires its own frozen contract checksum")
    gpu = config.get("launcher",{}).get("requested_gpu")
    _require(gpu in {"0","1","2","3"},"Profiling requires one declared physical GPU")
    attempt = config.get("attempt")
    _require(type(attempt) is int and attempt>0,"Execution attempt must be a positive integer")
    expected = build_configuration(parent,reference,gpu=gpu,task_contract_sha256=contract_sha,
                                   instrumentation_hashes=hashes)
    expected["attempt"] = attempt
    _require(config==expected,"Profiling changed its pinned scientific, resource or instrumentation contract")


def command_for_configuration(config: Mapping[str,Any], project_root: Path) -> list[str]:
    validate_configuration(config)
    return [sys.executable,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=1",
            "--max-restarts=0",str(project_root/"scripts/analysis/profile_so2_bce_execution.py"),
            "--config","{run_scratch}/config.resolved.yaml","--run-scratch","{run_scratch}"]
