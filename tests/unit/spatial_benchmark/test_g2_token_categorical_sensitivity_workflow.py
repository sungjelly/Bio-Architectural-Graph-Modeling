from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "analysis"
    / "run_g2_token_categorical_sensitivity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "g2_token_categorical_sensitivity_workflow_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_WORKFLOW = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _WORKFLOW
_SPEC.loader.exec_module(_WORKFLOW)
_GPU_UUIDS = {
    5: "GPU-0c5fd41d-c91d-6bea-a500-37fa0bc09589",
    6: "GPU-ca481ad2-1214-9b8a-0b51-7a3fa2bc0ac8",
    7: "GPU-64985f62-0b1c-ee82-d6d2-bbb6b972f96c",
}
_GPU_UUID = _GPU_UUIDS[7]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _run_row(width: str, seed: int) -> dict[str, object]:
    label = f"{width}_s{seed}"
    variant = (
        _WORKFLOW._CURRENT_VARIANT
        if width == "current"
        else _WORKFLOW._WIDER_VARIANT
    )
    return {
        "run_id": f"r_synthetic_{label}",
        "run_path": f"/synthetic/immutable/{label}",
        "variant_label": variant,
        "width_label": width,
        "seed": seed,
        "constructor_arguments": {},
        "state_dict_sha256": _sha(f"state-{label}"),
        "checkpoint_sha256": _sha(f"checkpoint-{label}"),
        "run_bundle_checksum_manifest_sha256": _sha(
            f"bundle-{label}"
        ),
        "completion_marker_sha256": _sha(f"success-{label}"),
        "config_resolved_sha256": _sha(f"config-{label}"),
        "final_metrics_sha256": _sha(f"metrics-{label}"),
        "fixed_mask_provenance_sha256": _sha(f"masks-{label}"),
        "tokenization_provenance_sha256": _sha(f"tokens-{label}"),
        "graph_sha256": _sha("graph"),
        "preprocessing_sha256": _sha("preprocessing"),
        "mask_bundle_sha256": _sha("mask-bundle"),
        "token_matrix_sha256": _sha("token-matrix"),
        "whole_node_exact_accuracy_percent": 92.0,
        "parameter_count": 11 if width == "current" else 17,
        "standard_bundle_verification": {
            "valid": True,
            "status": "success",
            "file_count": 20,
        },
    }


def _manifest_core() -> dict[str, object]:
    package_root = _ROOT / "src" / "spatial_benchmark"
    masks = [
        {
            "replicate": replicate,
            "entry_id": f"whole-node-{replicate}",
            "mask_checksum": _sha(f"mask-{replicate}"),
            "shape": [2, 3],
            "n_masked": 3,
        }
        for replicate in range(3)
    ]
    return {
        "schema_version": _WORKFLOW._ANALYSIS_SCHEMA_VERSION,
        "artifact_kind": (
            "g2_token_categorical_sensitivity_analysis_inputs"
        ),
        "active_analysis_layout_version": (
            _WORKFLOW._ACTIVE_ANALYSIS_LAYOUT_VERSION
        ),
        "protocol": _WORKFLOW.PROTOCOL_VERSION,
        "analysis_mode": _WORKFLOW._ANALYSIS_MODE,
        "campaign_id": _WORKFLOW._CAMPAIGN_ID,
        "prespecified_gate": {
            "eligible": False,
            "status": "skipped",
            "jacobians_computed": False,
        },
        "comparison_path": "/synthetic/comparison.json",
        "comparison_sha256": _sha("comparison"),
        "runs": [
            *(_run_row("current", seed) for seed in range(3)),
            *(_run_row("wider", seed) for seed in range(3)),
        ],
        "common_identity": {
            "graph_sha256": _sha("graph"),
            "preprocessing_sha256": _sha("preprocessing"),
            "mask_bundle_sha256": _sha("mask-bundle"),
            "token_matrix_sha256": _sha("token-matrix"),
            "n_nodes": 2,
            "n_genes": 3,
            "directed_edges": 4,
            "whole_node_masks": masks,
        },
        "execution_contract": {
            "expected_full_graph_vjps": 2_304,
        },
        "source_checksums": {
            "workflow_script_sha256": _WORKFLOW._sha256_file(_SCRIPT),
            "spatial_benchmark_package_sha256": {
                path.name: _WORKFLOW._sha256_file(path)
                for path in sorted(package_root.glob("*.py"))
            },
        },
        "environment": {"synthetic_cpu_test": True},
        "maximum_claim": "synthetic test only",
    }


def _gpu_resource(
    *,
    full_graph_vjp_count: int | None = None,
    physical_index: int = 7,
) -> dict[str, object]:
    gpu_uuid = _GPU_UUIDS[physical_index]
    value: dict[str, object] = {
        "device": "cuda:0",
        "isolated_gpu": {
            "cuda_visible_devices": str(physical_index),
            "visible_device_count": 1,
            "logical_device": "cuda:0",
            "logical_device_index": 0,
            "supervisor_assigned_physical_gpu_uuid": (
                gpu_uuid
            ),
            "physical_gpu_index": physical_index,
            "physical_gpu_uuid": gpu_uuid,
            "physical_pci_bus_id": "00000000:0F:00.0",
            "driver_version": "580.126.09",
            "device_name": "synthetic GPU",
            "compute_capability": [8, 6],
            "total_memory_bytes": 24 * 1024**3,
        },
        "floating_point": "fp32_no_amp_no_tf32",
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    if full_graph_vjp_count is not None:
        value.update(
            {
                "duration_seconds": 1.0,
                "peak_cuda_memory_allocated_bytes": 1024,
                "full_graph_vjp_count": full_graph_vjp_count,
            }
        )
    return value


def _data_provenance(
    common: Mapping[str, Any],
    mask: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "preprocessing_sha256": common["preprocessing_sha256"],
        "token_matrix_sha256": common["token_matrix_sha256"],
        "graph_sha256": common["graph_sha256"],
        "directed_edges": common["directed_edges"],
        "mask_bundle_sha256": common["mask_bundle_sha256"],
        "mask_replicate": mask["replicate"],
        "mask_entry_id": mask["entry_id"],
        "mask_checksum": mask["mask_checksum"],
        "n_target_nodes": 1,
        "shape": [2, 3],
    }


def _write_pilot(
    work: Path,
    manifest: Mapping[str, Any],
) -> str:
    analysis_sha = str(manifest["analysis_input_sha256"])
    common = manifest["common_identity"]
    mask = common["whole_node_masks"][0]
    _, probe = _WORKFLOW.make_rademacher_probe(
        (1, 3, 4),
        mask_entry_id=mask["entry_id"],
        probe_index=0,
    )
    peak = 1024
    total_seconds = 2.0
    wider = {
        f"{row['width_label']}_s{row['seed']}": row
        for row in manifest["runs"]
    }["wider_s0"]
    payload = {
        "schema_version": 1,
        "artifact_kind": (
            "g2_token_categorical_sensitivity_resource_pilot"
        ),
        "protocol": _WORKFLOW.PROTOCOL_VERSION,
        "analysis_mode": _WORKFLOW._ANALYSIS_MODE,
        "analysis_input_sha256": analysis_sha,
        "model_label": "wider_s0",
        "model_run_id": wider["run_id"],
        "model_state": {
            "label": "wider_s0",
            "source": wider["run_id"],
            "paired_seed": None,
            "state_dict_sha256": wider["state_dict_sha256"],
            "parameter_count": wider["parameter_count"],
        },
        "mask_replicate": 0,
        "mask_entry_id": mask["entry_id"],
        "probe_index": 0,
        "probe": probe,
        "data_provenance": _data_provenance(common, mask),
        "vjp_wall_seconds": 1.5,
        "tangent_projection_wall_seconds": 0.5,
        "total_model_equivalent_wall_seconds": total_seconds,
        "peak_cuda_memory_allocated_bytes": peak,
        "peak_cuda_memory_allocated_gib": peak / float(1024**3),
        "tangent_vjp_squared_norm": 1.0,
        "projected_2304_vjp_seconds": total_seconds * 2_304,
        "projected_2304_vjp_hours": total_seconds * 2_304 / 3600.0,
        "resource": _gpu_resource(),
        "projection_is_not_a_time_gate": True,
        "criteria": {
            "finite_nonzero_vjp": True,
            "peak_allocated_vram_at_most_20_5_gib": True,
        },
        "passes": True,
        "explicit_review_required_before_full_shards": True,
    }
    path = _WORKFLOW._pilot_path(work)
    _WORKFLOW._write_bound_json(
        path,
        payload,
        analysis_input_sha256=analysis_sha,
    )
    return _WORKFLOW._sha256_file(path)


def _pair_cross(pair: tuple[str, str], shard_type: str) -> float:
    if shard_type == "identical":
        return 1.0
    if shard_type == "random":
        return 0.1
    if pair[0].split("_s")[0] == pair[1].split("_s")[0]:
        return 0.97
    return 0.99


def _write_shard(
    work: Path,
    manifest: Mapping[str, Any],
    *,
    shard_type: str,
    mask_replicate: int,
    pilot_sha: str,
    review_sha: str | None,
    control_seed: int | None = None,
) -> Path:
    analysis_sha = str(manifest["analysis_input_sha256"])
    common = manifest["common_identity"]
    mask = common["whole_node_masks"][mask_replicate]
    probes = _WORKFLOW._expected_probe_records(mask, common=common)
    pairs = _WORKFLOW._expected_pairs(shard_type, control_seed)
    models = [
        dict(record)
        for record in _WORKFLOW._expected_model_records(
            manifest,
            shard_type=shard_type,
            control_seed=control_seed,
        )
    ]
    for record in models:
        if record["state_dict_sha256"] is None:
            record["state_dict_sha256"] = _sha(
                f"random-state-{record['label']}"
            )
    statistics = {}
    for pair in pairs:
        cross = _pair_cross(pair, shard_type)
        statistics[_WORKFLOW._pair_key(pair)] = [
            {
                "mask_entry_id": mask["entry_id"],
                "probe_index": probe_index,
                "reference_squared_norm": 1.0,
                "candidate_squared_norm": 1.0,
                "cross_inner_product": cross,
            }
            for probe_index in range(_WORKFLOW.PROBES_PER_MASK)
        ]
    model_count = 6 if shard_type == "trained" else 2
    payload = {
        "schema_version": _WORKFLOW._SHARD_SCHEMA_VERSION,
        "artifact_kind": "g2_token_categorical_sensitivity_shard",
        "protocol": _WORKFLOW.PROTOCOL_VERSION,
        "analysis_mode": _WORKFLOW._ANALYSIS_MODE,
        "analysis_input_sha256": analysis_sha,
        "shard_type": shard_type,
        "mask_replicate": mask_replicate,
        "mask_entry_id": mask["entry_id"],
        "mask_checksum": mask["mask_checksum"],
        "control_seed": control_seed,
        "reviewed_resource_pilot_sha256": pilot_sha,
        "reviewed_identical_control_sha256": review_sha,
        "models": models,
        "pairs": [
            {"reference": pair[0], "candidate": pair[1]}
            for pair in pairs
        ],
        "probe_records": list(probes),
        "statistics": statistics,
        "data_provenance": _data_provenance(common, mask),
        "resource": _gpu_resource(
            full_graph_vjp_count=(
                model_count * _WORKFLOW.PROBES_PER_MASK
            ),
            physical_index=5 + mask_replicate,
        ),
    }
    path = _WORKFLOW._shard_path(
        work,
        shard_type=shard_type,
        mask_replicate=mask_replicate,
        control_seed=control_seed,
    )
    _WORKFLOW._write_bound_json(
        path,
        payload,
        analysis_input_sha256=analysis_sha,
    )
    return path


def _initialize_synthetic_analysis(
    tmp_path: Path,
) -> tuple[Path, Mapping[str, Any], str]:
    work = tmp_path / "active-analysis"
    manifest, _ = _WORKFLOW._initialize_work_root(
        work,
        _manifest_core(),
    )
    pilot_sha = _write_pilot(work, manifest)
    return work, manifest, pilot_sha


def _rebind_payload(
    path: Path,
    payload: Mapping[str, Any],
    *,
    analysis_sha: str,
) -> None:
    _WORKFLOW._atomic_json(path, payload)
    _WORKFLOW._atomic_json(
        _WORKFLOW._sidecar_path(path),
        {
            "schema_version": 1,
            "analysis_input_sha256": analysis_sha,
            "artifact_sha256": _WORKFLOW._sha256_file(path),
        },
    )


def test_failed_gate_validation_binds_all_six_run_rows(tmp_path: Path) -> None:
    runs = []
    rows = []
    wider_values = {}
    for width, variant in (
        ("current", _WORKFLOW._CURRENT_VARIANT),
        ("wider", _WORKFLOW._WIDER_VARIANT),
    ):
        for seed in range(3):
            accuracy = 91.0 + seed / 10 + (0.5 if width == "wider" else 0)
            run_id = f"r_{width}_{seed}"
            runs.append(
                SimpleNamespace(
                    variant_label=variant,
                    seed=seed,
                    run_id=run_id,
                    exact_accuracy_percent=accuracy,
                )
            )
            rows.append(
                {
                    "variant_label": variant,
                    "seed": seed,
                    "run_id": run_id,
                    "whole_node_exact_accuracy_percent": accuracy,
                }
            )
            if width == "wider":
                wider_values[str(seed)] = accuracy
    comparison = {
        "schema_version": 1,
        "artifact_kind": "g2_token_multiseed_comparison",
        "campaign_id": _WORKFLOW._CAMPAIGN_ID,
        "status": "complete",
        "verified_identity": {
            "all_six_runs_match": True,
            "fixed_epochs_per_run": 200,
            "seeds_per_variant": [0, 1, 2],
        },
        "runs": rows,
        "relaxed_jacobian_gate": {
            "eligible": False,
            "status": "skipped",
            "jacobians_computed": False,
            "operator": "strictly_greater_than",
            "threshold_percent": 95.0,
            "wider_seed_values_percent": wider_values,
        },
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison), encoding="utf-8")

    _WORKFLOW._validate_failed_gate_comparison(path, runs)
    comparison["runs"][0]["whole_node_exact_accuracy_percent"] += 0.01
    path.write_text(json.dumps(comparison), encoding="utf-8")
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="disagrees",
    ):
        _WORKFLOW._validate_failed_gate_comparison(path, runs)


def test_bound_json_detects_tampering_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    analysis_sha = _sha("analysis")
    path = tmp_path / "artifact.json"
    payload = {
        "analysis_input_sha256": analysis_sha,
        "value": 1,
    }
    _WORKFLOW._write_bound_json(
        path,
        payload,
        analysis_input_sha256=analysis_sha,
    )
    assert _WORKFLOW._verify_bound_json(
        path,
        analysis_input_sha256=analysis_sha,
    )["value"] == 1
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="overwrite",
    ):
        _WORKFLOW._write_bound_json(
            path,
            payload,
            analysis_input_sha256=analysis_sha,
        )

    path.write_text(
        json.dumps({**payload, "value": 2}),
        encoding="utf-8",
    )
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="checksum binding",
    ):
        _WORKFLOW._verify_bound_json(
            path,
            analysis_input_sha256=analysis_sha,
        )

    json_only = tmp_path / "json-only.json"
    _WORKFLOW._atomic_json(json_only, payload)
    assert not _WORKFLOW._sidecar_path(json_only).exists()
    _WORKFLOW._verify_bound_json(
        json_only,
        analysis_input_sha256=analysis_sha,
    )
    assert _WORKFLOW._sidecar_path(json_only).is_file()

    sidecar_only = tmp_path / "sidecar-only.json"
    _WORKFLOW._atomic_json(
        _WORKFLOW._sidecar_path(sidecar_only),
        {
            "schema_version": 1,
            "analysis_input_sha256": analysis_sha,
            "artifact_sha256": _sha("interrupted-publication"),
        },
    )
    _WORKFLOW._write_bound_json(
        sidecar_only,
        payload,
        analysis_input_sha256=analysis_sha,
    )
    assert _WORKFLOW._verify_bound_json(
        sidecar_only,
        analysis_input_sha256=analysis_sha,
    ) == payload


def test_gpu_configuration_rejects_unisolated_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="process-level isolation",
    ):
        _WORKFLOW._configure_device(
            "cuda:0",
            expected_physical_gpu_uuid=_GPU_UUID,
        )


def test_gpu_preflight_rejects_wrong_cublas_and_forbidden_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="CUBLAS_WORKSPACE_CONFIG",
    ):
        _WORKFLOW._configure_device(
            "cuda:0",
            expected_physical_gpu_uuid=_GPU_UUID,
        )

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(
        _WORKFLOW.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            uuid=_GPU_UUID.removeprefix("GPU-"),
            name="synthetic GPU",
            major=8,
            minor=6,
            total_memory=24 * 1024**3,
        ),
    )
    monkeypatch.setattr(
        _WORKFLOW.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                f"4, {_GPU_UUID}, 00000000:0C:00.0, 580.126.09\n"
            )
        ),
    )
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="forbidden",
    ):
        _WORKFLOW._gpu_identity_record(
            _WORKFLOW.torch.device("cuda:0"),
            expected_physical_gpu_uuid=_GPU_UUID,
        )
    monkeypatch.setattr(
        _WORKFLOW.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                f"7, {_GPU_UUID}, 00000000:0F:00.0, 580.126.09\n"
            )
        ),
    )
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="supervisor assignment",
    ):
        _WORKFLOW._gpu_identity_record(
            _WORKFLOW.torch.device("cuda:0"),
            expected_physical_gpu_uuid=(
                "GPU-0c5fd41d-c91d-6bea-a500-37fa0bc09589"
            ),
        )


def test_resumed_pilot_and_shard_are_semantically_revalidated(
    tmp_path: Path,
) -> None:
    work, manifest, pilot_sha = _initialize_synthetic_analysis(tmp_path)
    analysis_sha = str(manifest["analysis_input_sha256"])
    shard_path = _write_shard(
        work,
        manifest,
        shard_type="identical",
        mask_replicate=0,
        pilot_sha=pilot_sha,
        review_sha=None,
    )
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["resource"]["full_graph_vjp_count"] = 63
    _rebind_payload(
        shard_path,
        shard,
        analysis_sha=analysis_sha,
    )
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="resource/execution",
    ):
        _WORKFLOW._run_shard(
            work_root=work,
            analysis_input_sha256=analysis_sha,
            runs=(),
            shard_type="identical",
            mask_replicate=0,
            control_seed=None,
            reviewed_pilot_sha256=pilot_sha,
            reviewed_identical_sha256=None,
            device=_WORKFLOW.torch.device("cpu"),
            gpu_identity={},
        )

    pilot_path = _WORKFLOW._pilot_path(work)
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    pilot["model_state"]["state_dict_sha256"] = _sha("wrong-state")
    _rebind_payload(
        pilot_path,
        pilot,
        analysis_sha=analysis_sha,
    )
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="checkpoint state",
    ):
        _WORKFLOW._run_pilot(
            work_root=work,
            analysis_input_sha256=analysis_sha,
            runs=(),
            device=_WORKFLOW.torch.device("cpu"),
            gpu_identity={},
        )


def test_identical_review_fails_closed_and_revalidates_on_resume(
    tmp_path: Path,
) -> None:
    work, manifest, pilot_sha = _initialize_synthetic_analysis(tmp_path)
    analysis_sha = str(manifest["analysis_input_sha256"])
    for replicate in range(3):
        path = _write_shard(
            work,
            manifest,
            shard_type="identical",
            mask_replicate=replicate,
            pilot_sha=pilot_sha,
            review_sha=None,
        )
        shard = json.loads(path.read_text(encoding="utf-8"))
        rows = shard["statistics"][
            _WORKFLOW._pair_key(_WORKFLOW._IDENTICAL_PAIR)
        ]
        for row in rows:
            row["cross_inner_product"] = 0.9
        _rebind_payload(path, shard, analysis_sha=analysis_sha)

    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="numerical review failed",
    ):
        _WORKFLOW.review_identical_shards(
            work,
            reviewed_pilot_sha256=pilot_sha,
        )
    review_path = _WORKFLOW._identical_review_path(work)
    review = _WORKFLOW._verify_bound_json(
        review_path,
        analysis_input_sha256=analysis_sha,
    )
    assert review["passes"] is False
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="numerical review failed",
    ):
        _WORKFLOW.review_identical_shards(
            work,
            reviewed_pilot_sha256=pilot_sha,
        )


def test_identical_review_and_complete_synthetic_aggregation(
    tmp_path: Path,
) -> None:
    work, manifest, pilot_sha = _initialize_synthetic_analysis(tmp_path)
    for replicate in range(3):
        _write_shard(
            work,
            manifest,
            shard_type="identical",
            mask_replicate=replicate,
            pilot_sha=pilot_sha,
            review_sha=None,
        )

    review = _WORKFLOW.review_identical_shards(
        work,
        reviewed_pilot_sha256=pilot_sha,
    )
    assert review["passes"] is True
    assert review["full_graph_vjp_count"] == 192
    review_path = _WORKFLOW._identical_review_path(work)
    review_sha = _WORKFLOW._sha256_file(review_path)

    for replicate in range(3):
        _write_shard(
            work,
            manifest,
            shard_type="trained",
            mask_replicate=replicate,
            pilot_sha=pilot_sha,
            review_sha=review_sha,
        )
        for seed in _WORKFLOW._RANDOM_SEEDS:
            _write_shard(
                work,
                manifest,
                shard_type="random",
                mask_replicate=replicate,
                control_seed=seed,
                pilot_sha=pilot_sha,
                review_sha=review_sha,
            )

    output = tmp_path / "comparison-output"
    result = _WORKFLOW.aggregate_verified_shards(work, output)

    assert result["status"] == "post_hoc_operational_match"
    assert result["verified_shard_count"] == 30
    assert result["verified_full_graph_vjp_count"] == 2_304
    assert result["decision"]["analysis_numerically_valid"] is True
    assert result["decision"]["operational_match"] is True
    assert (output / "analysis_manifest.json").is_file()
    assert (output / "resource_pilot.sha256.json").is_file()
    assert (output / "identical_control_review.sha256.json").is_file()
    assert (output / "_SUCCESS").is_file()
    _WORKFLOW._verify_output(output)

    trained_path = _WORKFLOW._shard_path(
        work,
        shard_type="trained",
        mask_replicate=0,
    )
    shard = deepcopy(json.loads(trained_path.read_text(encoding="utf-8")))
    shard["statistics"]["unexpected"] = []
    common = manifest["common_identity"]
    mask = common["whole_node_masks"][0]
    with pytest.raises(
        _WORKFLOW.SensitivityWorkflowError,
        match="missing or unexpected",
    ):
        _WORKFLOW._validate_shard(
            shard,
            manifest=manifest,
            analysis_input_sha256=manifest["analysis_input_sha256"],
            reviewed_resource_pilot_sha256=pilot_sha,
            reviewed_identical_control_sha256=review_sha,
            shard_type="trained",
            mask_row=mask,
            control_seed=None,
            expected_probe_records=(
                _WORKFLOW._expected_probe_records(mask, common=common)
            ),
        )
