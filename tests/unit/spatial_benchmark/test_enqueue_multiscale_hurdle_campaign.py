"""Focused tests for locked multiscale hurdle staged enqueueing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.registry import Registry


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "enqueue_multiscale_hurdle_campaign.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "enqueue_multiscale_hurdle_campaign_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_MATERIALIZER_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "materialize_multiscale_hurdle_campaign.py"
)
_MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "materialize_multiscale_hurdle_for_enqueue_tests",
    _MATERIALIZER_SCRIPT,
)
assert (
    _MATERIALIZER_SPEC is not None
    and _MATERIALIZER_SPEC.loader is not None
)
_MATERIALIZER = importlib.util.module_from_spec(_MATERIALIZER_SPEC)
sys.modules[_MATERIALIZER_SPEC.name] = _MATERIALIZER
_MATERIALIZER_SPEC.loader.exec_module(_MATERIALIZER)

EnqueueError = _MODULE.MultiscaleHurdleEnqueueError
enqueue_stage = _MODULE.enqueue_stage


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _component(reference: str) -> dict[str, Any]:
    return deepcopy(
        yaml.safe_load((_ROOT / reference).read_text(encoding="utf-8"))
    )


def _source_config(alias: str) -> dict[str, Any]:
    key = alias.lower().replace("-", "")
    dataset_fingerprint = _sha(f"{alias}-dataset")
    split_fingerprint = _sha(f"{alias}-split")
    return {
        "dataset": {
            "dataset_id": f"cosmx_{key}_enqueue_test",
            "version": "adjacent_normal_full_core_fit_v1",
            "split_id": f"split_{key}",
            "dataset_fingerprint": dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "preprocessing_version": "synthetic_preprocessing_v1",
            "prepared_artifact_reference": f"prepared/{alias.lower()}",
            "biological_target_count": 1000,
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "preprocessing_fit_scope": "all_nodes_transductive",
            "validation_or_test_partition_present": False,
            "patient_generalization_supported": False,
        },
        "features": {
            "use_edge_features": True,
            "fit_scope": "all_nodes_transductive",
            "node_expression": {
                "biological_targets": 1000,
                "source_scale": "raw_biological_probe_counts",
                "explicit_mask_authoritative_inside_model": True,
            },
            "node_metadata": {
                "transformed_with": "synthetic_verified_transform",
                "fields": list(_MATERIALIZER.ALLOWED_METADATA_COLUMNS),
            },
            "edge_features": {
                "fit_scope": "all_retained_directed_edges_transductive",
                "standardization": "full_core_edge_wise",
                "fields": list(_MATERIALIZER.EDGE_ATTRIBUTE_NAMES),
            },
            "prohibited_node_inputs": [
                "direct_identifiers",
                "absolute_or_local_coordinates",
                "expression_derived_library_size",
                "rna_derived_qc",
                "vendor_cell_type_cluster_neighborhood_or_niche",
                "hidden_target_values",
            ],
        },
    }


def _graph_receipt(alias: str) -> dict[str, Any]:
    n_nodes = 100
    x = np.arange(n_nodes, dtype=np.float64) * 20.0
    coordinates = np.stack(
        (
            x,
            np.full(n_nodes, float(int(alias[-2:])) * 10_000.0),
        ),
        axis=1,
    )
    source = np.arange(n_nodes, dtype=np.int64)
    receiver = (source + 1) % n_nodes
    local_edge_index = np.stack((source, receiver), axis=0)
    local_edge_attributes = np.zeros(
        (n_nodes, len(_MATERIALIZER.EDGE_ATTRIBUTE_NAMES)),
        dtype=np.float32,
    )
    permutation = (
        _MATERIALIZER.build_macroblock_spatial_antipode_permutation(
            coordinates,
            np.asarray([f"block-{alias}"] * n_nodes),
            local_edge_index=local_edge_index,
            local_edge_attributes=local_edge_attributes,
        )
    )
    scales: dict[str, Any] = {}
    for offset, scale in enumerate(("local", "regional")):
        scales[scale] = {
            "qc": {
                "n_directed_edges": n_nodes + offset,
                "n_components": 2 + offset,
                "n_isolated_nodes": offset,
            },
            "checksums": {
                "graph_sha256": _sha(f"{alias}-{scale}-graph"),
            },
        }
    return {
        **scales,
        "bundle_checksums": {
            "bundle_sha256": _sha(f"{alias}-graph-bundle"),
        },
        "bundle_qc": {
            "local_regional_disjoint": True,
            "all_graphs_symmetric": True,
            "all_graphs_receiver_sorted": True,
            "all_graphs_loop_and_duplicate_free": True,
        },
        "local_source_permutation": dict(permutation.receipt),
    }


def _register_input(
    registry: Registry,
    *,
    project_root: Path,
    source: Mapping[str, Any],
) -> None:
    dataset = source["dataset"]
    prepared = Path(str(dataset["prepared_artifact_reference"]))
    (project_root / prepared).mkdir(parents=True, exist_ok=True)
    registry.register_dataset(
        str(dataset["dataset_id"]),
        str(dataset["version"]),
        display_name=f"Synthetic {dataset['biological_unit_alias']}",
        protected_source_path=prepared,
        preprocessing_version=str(dataset["preprocessing_version"]),
        processed_fingerprint=str(dataset["dataset_fingerprint"]),
        verification_status="verified",
    )
    registry.register_split(
        str(dataset["split_id"]),
        dataset_id=str(dataset["dataset_id"]),
        dataset_version=str(dataset["version"]),
        method="all-fit",
        unit="single_adjacent_normal_spatial_core",
        fingerprint=str(dataset["split_fingerprint"]),
        protected_path=prepared,
        verification_status="verified",
    )


def _write_config(
    *,
    project_root: Path,
    locked: Path,
    source: Mapping[str, Any],
    alias: str,
    arm: str,
    pilot: bool,
    gpu: int,
    graph_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    output_reference = locked.relative_to(project_root).as_posix()
    contract_reference = (
        Path("experiments")
        / "campaigns"
        / _MODULE.CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    ).as_posix()
    role = "pilot" if pilot else "science"
    config = _MATERIALIZER._build_config(
        source_config=source,
        alias=alias,
        arm=arm,
        pilot=pilot,
        requested_gpu=gpu,
        model_component=_component(_MATERIALIZER._MODEL_COMPONENTS[arm]),
        graph_component=_component(_MATERIALIZER._GRAPH_COMPONENT),
        graph_record=graph_receipt,
        trainer_component=_component(
            _MATERIALIZER._TRAINER_COMPONENTS[role]
        ),
        evaluation_component=_component(
            _MATERIALIZER._EVALUATION_COMPONENTS[role]
        ),
        launcher_component=_component(_MATERIALIZER._LAUNCHER_COMPONENT),
        contract_reference=contract_reference,
        contract_sha256=_MODULE.FROZEN_CONTRACT_SHA256,
        contract_amendment={
            "reference": (
                _MATERIALIZER.CONTRACT_AMENDMENT_RELATIVE.as_posix()
            ),
            "sha256": _MATERIALIZER.CONTRACT_AMENDMENT_SHA256,
            "required_supplement": {
                "reference": (
                    _MATERIALIZER.REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE
                    .as_posix()
                ),
                "sha256": (
                    _MATERIALIZER.REQUIRED_CONTRACT_SUPPLEMENT_SHA256
                ),
                (
                    "mask_noninterference_gate_required_before_"
                    "gpu_training"
                ): True,
            },
            "supersedes": {
                "reference": (
                    _MATERIALIZER.SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE
                    .as_posix()
                ),
                "sha256": (
                    _MATERIALIZER.SUPERSEDED_CONTRACT_AMENDMENT_SHA256
                ),
                "retained_as_negative_design_record": True,
            },
        },
        output_reference=output_reference,
    )
    directory = "resource_pilot_configs" if pilot else "science_configs"
    filename = _MATERIALIZER._config_filename(
        alias, arm, pilot=pilot
    )
    path = locked / directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "alias": alias,
        "arm": arm,
        "stage": 1 if pilot or arm == "self" else 2,
        "config": path.relative_to(project_root).as_posix(),
        "config_sha256": canonical_sha256(config),
        "file_sha256": _sha256_file(path),
        "requested_gpu": gpu,
        "n_nodes": 100,
        "graph_bundle_sha256": graph_receipt["bundle_checksums"][
            "bundle_sha256"
        ],
        "local_source_permutation_sha256": graph_receipt[
            "local_source_permutation"
        ]["checksum"],
    }


def _run_id(index: int) -> str:
    return (
        f"r_20260729T12{index:04d}Z_1234abcd_s000_f00_a01_"
        f"gate{index:02d}"
    )


def _gate_job(
    *,
    project_root: Path,
    record: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    run_id = _run_id(index)
    bundle = project_root / "artifacts" / "runs" / "2026" / "07" / run_id
    bundle.mkdir(parents=True)
    marker = bundle / "_SUCCESS"
    content_sha256 = _sha(f"{run_id}-bundle-content")
    marker.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "success",
                "content_sha256": content_sha256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "alias": record["alias"],
        "arm": record["arm"],
        "config_sha256": record["config_sha256"],
        "run_id": run_id,
        "bundle_reference": bundle.relative_to(project_root).as_posix(),
        "success_marker_content_sha256": content_sha256,
        "verified_bundle": True,
    }


def _resource_gate(
    *,
    project_root: Path,
    materialization: Mapping[str, Any],
) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for index, record in enumerate(materialization["pilot_jobs"], start=1):
        job = _gate_job(
            project_root=project_root,
            record=record,
            index=index,
        )
        job.update(
            {
                "finite_losses_and_gradients": True,
                "parameter_match": True,
                "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
                "fp32_amp_absolute_loss_discrepancy": 5e-4,
                "precision_equivalence_passed": True,
                "peak_allocated_vram_gib": 10.0,
                "peak_vram_passed": True,
                "projected_gpu_hours_per_200_epochs": 0.4,
                "projected_runtime_passed": True,
                "filesystem_used_decimal_gb": 42.0,
                "disk_safety_passed": True,
                "runner_pilot_gate_passed": True,
            }
        )
        jobs.append(job)
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.RESOURCE_GATE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "stage": "resource",
            "materialization_checksum": materialization["checksum"],
            "frozen_contract_sha256": (
                materialization["frozen_contract"]["sha256"]
            ),
            "thresholds": dict(_MODULE.RESOURCE_THRESHOLDS),
            "complete": True,
            "gate_passed": True,
            "production_authorized": True,
            "failure_reasons": [],
            "jobs": jobs,
        }
    )


def _representation_gate(
    *,
    project_root: Path,
    materialization: Mapping[str, Any],
    resource_gate: Mapping[str, Any],
) -> dict[str, Any]:
    records = [
        record
        for record in materialization["science_jobs"]
        if record["alias"] in _MODULE.STAGE1_ALIASES
        and record["arm"] == "self"
    ]
    jobs: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=3):
        job = _gate_job(
            project_root=project_root,
            record=record,
            index=index,
        )
        job.update(
            {
                "h1_gate_passed": True,
                (
                    "detection_balanced_accuracy_above_"
                    "prevalence_reference"
                ): True,
                "detection_balanced_accuracy": 0.65,
                "prevalence_reference_balanced_accuracy": 0.51,
                "positive_continuous_huber_relative_improvement": 0.03,
                "positive_count_state_mae_relative_improvement": 0.04,
            }
        )
        jobs.append(job)
    return _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.REPRESENTATION_GATE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "stage": "stage1",
            "materialization_checksum": materialization["checksum"],
            "frozen_contract_sha256": (
                materialization["frozen_contract"]["sha256"]
            ),
            "resource_gate_checksum": resource_gate["checksum"],
            "thresholds": dict(_MODULE.REPRESENTATION_THRESHOLDS),
            "complete": True,
            "gate_passed": True,
            "both_h1_gates_passed": True,
            "stage2_authorized": True,
            "failure_reasons": [],
            "jobs": jobs,
        }
    )


@pytest.fixture
def locked_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(_MODULE, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        _MODULE,
        "_disk_used_decimal_gb",
        lambda _path: 42.0,
    )
    locked = (
        project_root
        / "scratch"
        / "locked_campaigns"
        / _MODULE.CAMPAIGN_ID
    )
    locked.mkdir(parents=True)
    contract = (
        project_root
        / "experiments"
        / "campaigns"
        / _MODULE.CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    contract.parent.mkdir(parents=True)
    contract.write_bytes(
        (
            _ROOT
            / "experiments"
            / "campaigns"
            / _MODULE.CAMPAIGN_ID
            / "frozen_task_contract.yaml"
        ).read_bytes()
    )
    for relative in (
        _MODULE.CONTRACT_AMENDMENT_RELATIVE,
        _MODULE.SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE,
        _MODULE.REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE,
    ):
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((_ROOT / relative).read_bytes())
    database = project_root / "state" / "tracking.sqlite3"
    registry = Registry(database)
    registry.create_campaign(
        _MODULE.CAMPAIGN_ID,
        name="Synthetic multiscale hurdle enqueue campaign",
    )

    sources = {
        alias: _source_config(alias) for alias in _MODULE.STAGE2_ALIASES
    }
    graph_receipts = {
        alias: _graph_receipt(alias) for alias in _MODULE.STAGE2_ALIASES
    }
    for source in sources.values():
        _register_input(
            registry, project_root=project_root, source=source
        )
    safe = sorted(_MODULE.SAFE_GPU_IDS)
    pilot_records = [
        _write_config(
            project_root=project_root,
            locked=locked,
            source=sources[alias],
            alias=alias,
            arm="self",
            pilot=True,
            gpu=safe[index % len(safe)],
            graph_receipt=graph_receipts[alias],
        )
        for index, alias in enumerate(_MODULE.STAGE1_ALIASES)
    ]
    science_records: list[dict[str, Any]] = []
    for index, (alias, arm) in enumerate(
        (alias, arm)
        for alias in _MODULE.STAGE2_ALIASES
        for arm in _MODULE.ARMS
    ):
        science_records.append(
            _write_config(
                project_root=project_root,
                locked=locked,
                source=sources[alias],
                alias=alias,
                arm=arm,
                pilot=False,
                gpu=safe[index % len(safe)],
                graph_receipt=graph_receipts[alias],
            )
        )
    materialization = _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.MATERIALIZATION_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "frozen_contract": {
                "reference": contract.relative_to(project_root).as_posix(),
                "sha256": _MODULE.FROZEN_CONTRACT_SHA256,
            },
            "contract_amendment": {
                "reference": (
                    _MODULE.CONTRACT_AMENDMENT_RELATIVE.as_posix()
                ),
                "sha256": _MODULE.CONTRACT_AMENDMENT_SHA256,
                "required_supplement": {
                    "reference": (
                        _MODULE.REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE
                        .as_posix()
                    ),
                    "sha256": (
                        _MODULE.REQUIRED_CONTRACT_SUPPLEMENT_SHA256
                    ),
                    (
                        "mask_noninterference_gate_required_before_"
                        "gpu_training"
                    ): True,
                },
                "supersedes": {
                    "reference": (
                        _MODULE.SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE
                        .as_posix()
                    ),
                    "sha256": (
                        _MODULE.SUPERSEDED_CONTRACT_AMENDMENT_SHA256
                    ),
                    "retained_as_negative_design_record": True,
                },
            },
            "parameter_audit": {
                "trainable_parameter_count": (
                    _MODULE.EXPECTED_PARAMETER_COUNT
                ),
                "named_parameter_shapes_sha256": _sha(
                    "named-parameter-shapes"
                ),
                "matched_arms": list(_MODULE.ARMS),
            },
            "cores": [
                {
                    "alias": alias,
                    "n_nodes": 100,
                    "n_genes": 1000,
                    "prepared_artifact": sources[alias]["dataset"][
                        "prepared_artifact_reference"
                    ],
                    "source_prepared_data_sha256": _sha(
                        f"{alias}-prepared"
                    ),
                    "preprocessing_sha256": sources[alias]["dataset"][
                        "dataset_fingerprint"
                    ],
                    "split_fingerprint": sources[alias]["dataset"][
                        "split_fingerprint"
                    ],
                    "graph_receipt": graph_receipts[alias],
                    "graph_receipt_sha256": canonical_sha256(
                        graph_receipts[alias]
                    ),
                }
                for alias in _MODULE.STAGE2_ALIASES
            ],
            "allowed_gpu_ids": sorted(_MODULE.SAFE_GPU_IDS),
            "resource_limits": {
                "preferred_aggregate_gpu_hours": 12.0,
                "absolute_aggregate_gpu_hours": 24.0,
                "stage1_peak_allocated_vram_gib": 12.0,
                "per_device_peak_allocated_vram_gib": 20.5,
                "aggregate_observed_process_vram_gib": 50.0,
                "filesystem_used_decimal_gb_hard_stop": 55.0,
            },
            "fixed_graph_contract": {
                "local_k_cap": 64,
                "local_maximum_distance_um": 75.0,
                "regional_k_cap": 256,
                "regional_minimum_distance_exclusive_um": 75.0,
                "regional_maximum_distance_um": 300.0,
                "original_rewired_arm_authorized": False,
                "local_source_permutation": {
                    "schema": (
                        _MATERIALIZER.LOCAL_SOURCE_PERMUTATION_SCHEMA
                    ),
                    "uses_random_seed": False,
                    "minimum_node_mapping_changed_fraction": 0.99,
                    "displacement_threshold_um": 75.0,
                    (
                        "minimum_node_displacement_above_threshold_"
                        "fraction"
                    ): 0.90,
                    (
                        "minimum_local_edge_slot_sender_identity_"
                        "changed_fraction"
                    ): 0.99,
                    (
                        "observed_local_topology_and_attributes_"
                        "unchanged"
                    ): True,
                },
            },
            "pilot_jobs": pilot_records,
            "science_jobs": science_records,
            "resource_gate_receipt_reference": (
                locked / "resource_gate_receipt.json"
            ).relative_to(project_root).as_posix(),
            "representation_gate_receipt_reference": (
                locked / "representation_gate_receipt.json"
            ).relative_to(project_root).as_posix(),
            "counts": {
                "cores": 5,
                "pilot_configs": 2,
                "science_configs": 20,
            },
            "registry_mutation_performed": False,
            "queue_mutation_performed": False,
            "training_performed": False,
        }
    )
    materialization_path = locked / "locked_config_materialization.json"
    _write_json(materialization_path, materialization)
    resource_gate = _resource_gate(
        project_root=project_root,
        materialization=materialization,
    )
    resource_gate_path = locked / "resource_gate_receipt.json"
    _write_json(resource_gate_path, resource_gate)
    representation_gate = _representation_gate(
        project_root=project_root,
        materialization=materialization,
        resource_gate=resource_gate,
    )
    representation_gate_path = locked / "representation_gate_receipt.json"
    _write_json(representation_gate_path, representation_gate)
    return {
        "project_root": project_root,
        "locked": locked,
        "database": database,
        "materialization": materialization,
        "materialization_path": materialization_path,
        "resource_gate": resource_gate,
        "resource_gate_path": resource_gate_path,
        "representation_gate": representation_gate,
        "representation_gate_path": representation_gate_path,
    }


def _call(
    fixture: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    return enqueue_stage(
        stage=stage,
        materialization_path=fixture["materialization_path"],
        resource_gate_path=fixture["resource_gate_path"],
        representation_gate_path=fixture["representation_gate_path"],
        receipt_path=fixture["locked"] / f"{stage}_enqueue_receipt.json",
        database_path=fixture["database"],
    )


def _job_identity(row: Mapping[str, Any]) -> tuple[str, str, bool]:
    experiment = row["canonical_config"]["experiment"]
    return (
        str(experiment["biological_unit_alias"]),
        str(experiment["arm"]),
        bool(experiment["resource_pilot"]),
    )


def test_three_stages_enqueue_exact_matrix_idempotently(
    locked_fixture: dict[str, Any],
) -> None:
    first = {
        stage: _call(locked_fixture, stage)
        for stage in ("resource", "stage1", "stage2")
    }
    second = {
        stage: _call(locked_fixture, stage)
        for stage in ("resource", "stage1", "stage2")
    }

    assert first == second
    assert {stage: len(value["jobs"]) for stage, value in first.items()} == {
        "resource": 2,
        "stage1": 2,
        "stage2": 18,
    }
    expected_resource = {
        (alias, "self", True) for alias in _MODULE.STAGE1_ALIASES
    }
    expected_stage1 = {
        (alias, "self", False) for alias in _MODULE.STAGE1_ALIASES
    }
    expected_stage2 = {
        (alias, arm, False)
        for alias in _MODULE.STAGE2_ALIASES
        for arm in _MODULE.ARMS
    } - expected_stage1
    registry = Registry(locked_fixture["database"])
    rows = registry.list_queue(limit=100)
    assert len(rows) == 22
    observed = {_job_identity(row) for row in rows}
    assert observed == expected_resource | expected_stage1 | expected_stage2
    for row in rows:
        identity = _job_identity(row)
        expected_stage = (
            "resource"
            if identity in expected_resource
            else "stage1"
            if identity in expected_stage1
            else "stage2"
        )
        assert int(row["maximum_attempts"]) == 2
        assert int(row["attempt_count"]) == 1
        assert int(row["priority"]) == _MODULE._STAGE_PRIORITY[
            expected_stage
        ]
        assert int(row["requested_gpu"]) in _MODULE.SAFE_GPU_IDS
        assert int(row["requested_gpu"]) != 4
        assert row["command"] == _MODULE._runner_command()
        assert row["command"][1].endswith(
            "scripts/train/run_multiscale_hurdle_capacity.py"
        )
        assert row["command"][2:] == [
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-scratch",
            "{run_scratch}",
        ]
    with registry.connect() as connection:
        membership_count = connection.execute(
            """
            SELECT count(*) FROM campaign_variants
            WHERE campaign_id = ?
            """,
            (_MODULE.CAMPAIGN_ID,),
        ).fetchone()[0]
    assert membership_count == 22

    prohibited = _MODULE._PROHIBITED_IDENTIFIER_KEYS
    for stage, receipt in first.items():
        assert receipt["complete"] is True
        assert receipt["maximum_attempts"] == 2
        assert receipt["disk_used_decimal_gb_at_enqueue"] == 42.0
        persisted = json.loads(
            (
                locked_fixture["locked"]
                / f"{stage}_enqueue_receipt.json"
            ).read_text(encoding="utf-8")
        )
        checksum = persisted.pop("checksum")
        assert checksum == canonical_sha256(persisted)
        serialized = json.dumps(receipt).lower()
        for key in prohibited:
            assert f'"{key}"' not in serialized
        assert not list(
            locked_fixture["locked"].glob(
                f".{stage}_enqueue_receipt.json.*.tmp"
            )
        )
    assert first["resource"]["resource_gate_checksum"] is None
    assert first["stage1"]["resource_gate_checksum"] == (
        locked_fixture["resource_gate"]["checksum"]
    )
    assert first["stage2"]["representation_gate_checksum"] == (
        locked_fixture["representation_gate"]["checksum"]
    )


def test_concurrent_resource_invocations_create_only_two_root_jobs(
    locked_fixture: dict[str, Any],
) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _index: _call(locked_fixture, "resource"),
                range(2),
            )
        )

    assert all(result["complete"] is True for result in results)
    rows = Registry(locked_fixture["database"]).list_queue(limit=100)
    assert len(rows) == 2
    assert {_job_identity(row) for row in rows} == {
        (alias, "self", True) for alias in _MODULE.STAGE1_ALIASES
    }


@pytest.mark.parametrize(
    "raw",
    [
        '{"schema_version":1,"schema_version":1}\n',
        '{"value":NaN}\n',
        '{"value":Infinity}\n',
    ],
)
def test_materialization_requires_strict_json(
    locked_fixture: dict[str, Any],
    raw: str,
) -> None:
    locked_fixture["materialization_path"].write_text(
        raw, encoding="utf-8"
    )

    with pytest.raises(EnqueueError):
        _call(locked_fixture, "resource")

    assert Registry(locked_fixture["database"]).list_queue(limit=100) == []


@pytest.mark.parametrize(
    ("stage", "gate_name", "mutation", "message"),
    [
        ("stage1", "resource", "checksum_binding", "resource gate"),
        ("stage1", "resource", "incomplete", "resource gate"),
        ("stage1", "resource", "unsafe_evidence", "resource gate"),
        ("stage1", "resource", "marker_drift", "_SUCCESS marker"),
        (
            "stage2",
            "representation",
            "resource_binding",
            "representation gate",
        ),
        (
            "stage2",
            "representation",
            "one_h1_failure",
            "both cores",
        ),
        (
            "stage2",
            "representation",
            "duplicate_run",
            "repeats an immutable run ID",
        ),
        (
            "stage2",
            "representation",
            "identifier",
            "prohibited identifier",
        ),
    ],
)
def test_gate_tampering_fails_before_queue_mutation(
    locked_fixture: dict[str, Any],
    stage: str,
    gate_name: str,
    mutation: str,
    message: str,
) -> None:
    gate_path = locked_fixture[f"{gate_name}_gate_path"]
    gate = deepcopy(locked_fixture[f"{gate_name}_gate"])
    gate.pop("checksum")
    if mutation == "checksum_binding":
        gate["materialization_checksum"] = "0" * 64
    elif mutation == "incomplete":
        gate["complete"] = False
    elif mutation == "unsafe_evidence":
        gate["jobs"][0]["peak_allocated_vram_gib"] = 12.1
    elif mutation == "marker_drift":
        bundle = (
            locked_fixture["project_root"]
            / gate["jobs"][0]["bundle_reference"]
        )
        (bundle / "_SUCCESS").write_text(
            "changed\n", encoding="utf-8"
        )
    elif mutation == "resource_binding":
        gate["resource_gate_checksum"] = "1" * 64
    elif mutation == "one_h1_failure":
        gate["jobs"][0]["h1_gate_passed"] = False
    elif mutation == "identifier":
        gate["jobs"][0]["patient_id"] = "restricted"
    else:
        gate["jobs"][1]["run_id"] = gate["jobs"][0]["run_id"]
        gate["jobs"][1]["bundle_reference"] = gate["jobs"][0][
            "bundle_reference"
        ]
        gate["jobs"][1]["success_marker_content_sha256"] = gate["jobs"][0][
            "success_marker_content_sha256"
        ]
    _write_json(gate_path, _signed(gate))

    with pytest.raises(EnqueueError, match=message):
        _call(locked_fixture, stage)

    assert Registry(locked_fixture["database"]).list_queue(limit=100) == []
    assert not (
        locked_fixture["locked"] / f"{stage}_enqueue_receipt.json"
    ).exists()


def test_atomic_receipt_refuses_direct_identifier_fields(
    locked_fixture: dict[str, Any],
) -> None:
    path = locked_fixture["locked"] / "unsafe_enqueue_receipt.json"

    with pytest.raises(EnqueueError, match="prohibited identifier"):
        _MODULE._atomic_receipt(
            path,
            {
                "schema_version": 1,
                "patient_id": "restricted",
            },
        )

    assert not path.exists()


def test_config_file_drift_preflights_all_jobs_before_registry_writes(
    locked_fixture: dict[str, Any],
) -> None:
    last = locked_fixture["materialization"]["science_jobs"][-1]
    config_path = locked_fixture["project_root"] / last["config"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["trainer"]["max_epochs"] = 199
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(EnqueueError, match="file checksum changed"):
        _call(locked_fixture, "resource")

    assert Registry(locked_fixture["database"]).list_queue(limit=100) == []


@pytest.mark.parametrize("mutation", ["unsafe_gpu", "duplicate_matrix"])
def test_materialization_matrix_and_safe_gpu_are_fail_closed(
    locked_fixture: dict[str, Any],
    mutation: str,
) -> None:
    materialization = deepcopy(locked_fixture["materialization"])
    materialization.pop("checksum")
    if mutation == "unsafe_gpu":
        materialization["science_jobs"][0]["requested_gpu"] = 4
    else:
        materialization["science_jobs"][1]["alias"] = (
            materialization["science_jobs"][0]["alias"]
        )
        materialization["science_jobs"][1]["arm"] = (
            materialization["science_jobs"][0]["arm"]
        )
    _write_json(
        locked_fixture["materialization_path"],
        _signed(materialization),
    )

    with pytest.raises(EnqueueError, match="GPU|exact job matrix"):
        _call(locked_fixture, "resource")

    assert Registry(locked_fixture["database"]).list_queue(limit=100) == []


def test_disk_used_at_or_above_decimal_limit_blocks_enqueue(
    locked_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _MODULE,
        "_disk_used_decimal_gb",
        lambda _path: 55.0,
    )

    with pytest.raises(EnqueueError, match="below 55.0 decimal GB"):
        _call(locked_fixture, "resource")

    assert Registry(locked_fixture["database"]).list_queue(limit=100) == []


def test_unregistered_campaign_blocks_enqueue(
    locked_fixture: dict[str, Any],
) -> None:
    database = locked_fixture["project_root"] / "state" / "empty.sqlite3"

    with pytest.raises(EnqueueError, match="not registered"):
        enqueue_stage(
            stage="resource",
            materialization_path=locked_fixture["materialization_path"],
            receipt_path=locked_fixture["locked"] / "empty-receipt.json",
            database_path=database,
        )

    assert Registry(database).list_queue(limit=100) == []


@pytest.mark.parametrize("mutation", ["extra", "duplicate"])
def test_extra_or_duplicate_campaign_root_jobs_are_rejected(
    locked_fixture: dict[str, Any],
    mutation: str,
) -> None:
    _call(locked_fixture, "resource")
    registry = Registry(locked_fixture["database"])
    root = registry.list_queue(limit=100)[0]
    config = deepcopy(root["canonical_config"])
    if mutation == "extra":
        config["experiment"]["variant_label"] = "unexpected_extra"
    registry.enqueue(
        campaign_id=_MODULE.CAMPAIGN_ID,
        configuration=config,
        command=_MODULE._runner_command(),
        experiment_config_reference=Path("unexpected.yaml"),
        priority=0,
        maximum_attempts=2,
        requested_gpu="0",
        job_id=f"job_{mutation}_fixture",
    )

    with pytest.raises(
        EnqueueError,
        match="unexpected root|duplicate root",
    ):
        _call(locked_fixture, "resource")


def test_existing_job_command_and_attempt_budget_are_revalidated(
    locked_fixture: dict[str, Any],
) -> None:
    _call(locked_fixture, "resource")
    registry = Registry(locked_fixture["database"])
    job = registry.list_queue(limit=100)[0]
    with registry.transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE queue_jobs
            SET command_json = ?, maximum_attempts = ?
            WHERE job_id = ?
            """,
            (json.dumps(["wrong-command"]), 3, job["job_id"]),
        )

    with pytest.raises(EnqueueError, match="wrong command"):
        _call(locked_fixture, "resource")
