from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from spatial_benchmark.registry import Registry


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/enqueue_adjacency_ablation.py"
_SPEC = importlib.util.spec_from_file_location(
    "enqueue_adjacency_ablation_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_RUNNER_SCRIPT = _ROOT / "scripts/train/run_adjacency_ablation.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_adjacency_ablation_for_enqueue_tests", _RUNNER_SCRIPT
)
assert _RUNNER_SPEC is not None and _RUNNER_SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = _RUNNER
_RUNNER_SPEC.loader.exec_module(_RUNNER)


def _folds() -> dict[str, dict[str, object]]:
    aliases = list(_MODULE.ALIASES)
    result: dict[str, dict[str, object]] = {}
    for fold in _MODULE.FOLDS:
        test = [aliases[fold], aliases[5 + fold]]
        validation = [
            (
                aliases[(fold + 1) % 5]
                if fold % 2 == 0
                else aliases[5 + ((fold + 1) % 5)]
            )
        ]
        train = [
            alias
            for alias in aliases
            if alias not in set(test + validation)
        ]
        result[str(fold)] = {
            "train_aliases": train,
            "validation_aliases": validation,
            "test_aliases": test,
            "preprocessing_reference": f"fold_{fold}.npz",
        }
    return result


def _manifest() -> dict[str, object]:
    content_sha = "a" * 64
    return {
        "schema_version": 1,
        "artifact_kind": _MODULE.PREPARED_ARTIFACT_KIND,
        "artifact_id": content_sha[:16],
        "content_sha256": content_sha,
        "dataset": {
            "dataset_id": _MODULE.DATASET_ID,
            "dataset_version": _MODULE.DATASET_VERSION,
            "source_dataset_id": "protected_fixture",
            "tissue_context": "AdjacentNormal",
            "core_count": 10,
            "cell_count": 117_386,
            "fov_group_count": 139,
            "n_genes": 1000,
            "qc_policy": "retain_all_vendor_qc_flag_not_input",
            "qc_passed_count": 112_815,
            "technical_control_prefixes_excluded": [
                "Negative",
                "SystemControl",
            ],
            "selection_sha256": "b" * 64,
            "dataset_fingerprint": "c" * 64,
        },
        "split": {
            "split_id": _MODULE.SPLIT_ID,
            "method": "slide_balanced_grouped_five_fold",
            "unit": "donor_core_one_to_one",
            "fold_count": 5,
            "train_groups": 7,
            "validation_groups": 1,
            "test_groups": 2,
            "test_slide_balance": True,
            "assignment_fingerprint": "d" * 64,
        },
        "features": {"gene_names": [f"g{i}" for i in range(1000)]},
        "graph": {
            "kind": "fixed_coordinate_knn_radius",
            "k": 12,
            "radius_um": 50.0,
            "symmetry": "union",
            "grouping": "raw_fov_within_core",
            "self_loops": "one_per_cell_in_all_arms",
            "edge_features": False,
            "arms": ["spatial", "isolated", "position_permuted_null"],
            "null_kind": "fixed_within_fov_position_permutation",
            "null_seed": 2_026_080_202,
            "graph_fingerprint": "e" * 64,
        },
        "masking": {
            "distribution": "exact_uniform_integer_0_through_G_per_cell",
            "positions": "without_replacement",
            "repetitions": 3,
            "base_seed": 20_260_802,
            "seed_derivation": "fixture",
            "bitorder": "little",
            "pack_axis": 1,
            "packed_gene_bytes": 125,
            "mask_fingerprint": "f" * 64,
        },
        "preprocessing": {
            "transform": "gene_wise_standardized_log1p_raw_count",
            "fit_scope": "seven_training_cores_only",
            "weighting": "equal_core",
            "scale_floor": 1.0e-6,
            "dtype": "float32",
            "preprocessing_fingerprint": "1" * 64,
        },
        "cores": {alias: {} for alias in _MODULE.ALIASES},
        "folds": _folds(),
        "files": {},
        "input_provenance": {},
        "provenance": {},
        "registry_contract": {
            "explicit_register_flag_required": True,
            "dataset_id": _MODULE.DATASET_ID,
            "dataset_version": _MODULE.DATASET_VERSION,
            "split_id": _MODULE.SPLIT_ID,
            "default_database_reference": "state/tracking/bagm.sqlite3",
        },
    }


def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path, Path, Path]:
    project_root = tmp_path / "project"
    contract = (
        project_root
        / "experiments/campaigns"
        / _MODULE.CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    contract.parent.mkdir(parents=True)
    contract.write_bytes(
        (
            _ROOT
            / "experiments/campaigns"
            / _MODULE.CAMPAIGN_ID
            / "frozen_task_contract.yaml"
        ).read_bytes()
    )
    prepared = (
        project_root
        / "data/processed/adjacent_normal_grouped_adjacency_ablation_v1"
    )
    prepared.mkdir(parents=True)
    manifest_path = prepared / "manifest.json"
    manifest_path.write_text('{"fixture": true}\n', encoding="utf-8")
    manifest = _manifest()
    monkeypatch.setattr(
        _RUNNER,
        "EXPECTED_PREPARED_MANIFEST",
        manifest_path.relative_to(project_root).as_posix(),
    )
    monkeypatch.setattr(
        _RUNNER,
        "EXPECTED_PREPARED_MANIFEST_SHA256",
        _MODULE._sha256_file(manifest_path),
    )
    monkeypatch.setattr(
        _RUNNER,
        "EXPECTED_PREPARED_CONTENT_SHA256",
        manifest["content_sha256"],
    )
    monkeypatch.setattr(
        _RUNNER,
        "EXPECTED_DATASET_FINGERPRINT",
        manifest["dataset"]["dataset_fingerprint"],
    )
    monkeypatch.setattr(
        _RUNNER,
        "EXPECTED_SPLIT_FINGERPRINT",
        manifest["split"]["assignment_fingerprint"],
    )
    monkeypatch.setattr(
        _RUNNER,
        "EXPECTED_PREPROCESSING_FINGERPRINT",
        manifest["preprocessing"]["preprocessing_fingerprint"],
    )
    fake_preparer = types.ModuleType("prepare_adjacency_ablation")
    fake_preparer.verify_prepared_artifact = lambda _: deepcopy(manifest)
    monkeypatch.setitem(sys.modules, "prepare_adjacency_ablation", fake_preparer)
    monkeypatch.setattr(_MODULE, "_PROJECT_ROOT", project_root)

    def command_for_config(config: object) -> list[str]:
        assert isinstance(config, dict)
        assert config["evaluation"]["protocol"] == (
            "grouped_core_adjacency_ablation_v1"
        )
        assert config["model"]["name"] == _MODULE.MODEL_NAME
        return [
            sys.executable,
            str(project_root / "scripts/train/run_adjacency_ablation.py"),
            "--config",
            "{run_scratch}/config.resolved.yaml",
            "--run-scratch",
            "{run_scratch}",
        ]

    monkeypatch.setattr(_MODULE, "command_for_config", command_for_config)
    database = project_root / "state/tracking/bagm.sqlite3"
    registry = Registry(database)
    registry.initialize()
    registry.register_dataset(
        _MODULE.DATASET_ID,
        _MODULE.DATASET_VERSION,
        display_name="fixture",
        protected_source_path=prepared,
        processed_fingerprint=manifest["dataset"]["dataset_fingerprint"],
        preprocessing_version=manifest["preprocessing"][
            "preprocessing_fingerprint"
        ],
        aggregate_sample_count=117_386,
        graph_count=30,
        verification_status="verified",
    )
    registry.register_split(
        _MODULE.SPLIT_ID,
        dataset_id=_MODULE.DATASET_ID,
        dataset_version=_MODULE.DATASET_VERSION,
        method="slide_balanced_grouped_five_fold",
        unit="donor_core_one_to_one",
        fold_count=5,
        fingerprint=manifest["split"]["assignment_fingerprint"],
        protected_path=prepared,
        verification_status="verified",
    )
    locked = project_root / "scratch/locked_campaigns" / _MODULE.CAMPAIGN_ID
    return manifest, manifest_path, database, locked


def test_materialize_is_immutable_idempotent_and_exactly_paired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest_path, database, locked = _setup(tmp_path, monkeypatch)

    first = _MODULE.materialize(
        manifest_path=manifest_path,
        locked_root=locked,
        database_path=database,
    )
    second = _MODULE.materialize(
        manifest_path=manifest_path,
        locked_root=locked,
        database_path=database,
    )

    assert first == second
    assert first["counts"] == {
        "smoke": 2,
        "pilot": 2,
        "primary": 50,
        "null": 25,
    }
    assert len(first["jobs"]) == 79
    assert {job["requested_gpu"] for job in first["jobs"]} == set(range(8))
    assert all(
        job["config_file_sha256"]
        == _MODULE._sha256_file(
            _MODULE._PROJECT_ROOT / job["config_reference"]
        )
        for job in first["jobs"]
    )

    primary_fold0_seed0 = [
        job
        for job in first["jobs"]
        if job["stage"] == "primary"
        and job["fold"] == 0
        and job["seed"] == 0
    ]
    assert {job["condition"] for job in primary_fold0_seed0} == {
        "spatial",
        "isolated",
    }
    configs = {
        job["condition"]: _MODULE.load_yaml_mapping(
            _MODULE._PROJECT_ROOT / job["config_reference"]
        )
        for job in primary_fold0_seed0
    }
    assert _MODULE._paired_payload(configs["spatial"]) == (
        _MODULE._paired_payload(configs["isolated"])
    )
    assert configs["spatial"]["model"]["expected_parameter_count"] == 645_736
    assert configs["spatial"]["trainer"]["max_epochs"] == 80
    assert configs["spatial"]["trainer"][
        "expected_total_optimizer_updates"
    ] == 560
    assert _RUNNER._validate_config(configs["spatial"]) == (
        "primary",
        "spatial",
        0,
        0,
    )
    registry = Registry(database)
    with registry.connect() as connection:
        variant_count = connection.execute(
            "SELECT COUNT(*) FROM campaign_variants WHERE campaign_id = ?",
            (_MODULE.CAMPAIGN_ID,),
        ).fetchone()[0]
    assert variant_count == 7


def test_smoke_enqueue_is_idempotent_and_has_one_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest_path, database, locked = _setup(tmp_path, monkeypatch)
    materialization = _MODULE.materialize(
        manifest_path=manifest_path,
        locked_root=locked,
        database_path=database,
    )
    receipt_path = locked / "smoke_enqueue_receipt.json"
    kwargs = {
        "stage": "smoke",
        "materialization_path": locked / "materialization_receipt.json",
        "database_path": database,
        "receipt_path": receipt_path,
        "pilot_gate_path": locked / "missing-pilot.json",
        "null_trigger_path": locked / "missing-null.json",
    }

    first = _MODULE.enqueue_stage(**kwargs)
    second = _MODULE.enqueue_stage(**kwargs)

    assert first == second
    assert first["materialization_checksum"] == materialization["checksum"]
    assert len(first["jobs"]) == 2
    assert "recovery_plan_checksum" not in first
    assert "recovery_enqueue_checksum" not in first
    assert "execution_device" not in first
    registry = Registry(database)
    with registry.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM queue_jobs ORDER BY job_id"
        ).fetchall()
    assert len(rows) == 2
    assert {int(row["maximum_attempts"]) for row in rows} == {1}
    assert {int(row["attempt_count"]) for row in rows} == {1}
    assert {str(row["requested_gpu"]) for row in rows} == {"0"}


def test_pilot_and_null_gates_are_checksum_bound_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest_path, database, locked = _setup(tmp_path, monkeypatch)
    materialization = _MODULE.materialize(
        manifest_path=manifest_path,
        locked_root=locked,
        database_path=database,
    )
    pilot_jobs = [
        {
            "condition": item["condition"],
            "run_id": f"pilot-{item['condition']}",
            "config_sha256": item["config_sha256"],
            "initial_state_sha256": "2" * 64,
            "training_mask_schedule_sha256": "3" * 64,
            "finite_metrics": True,
            "coverage_complete": True,
            "update_count_verified": True,
            "resource_limits_passed": True,
        }
        for item in materialization["jobs"]
        if item["stage"] == "pilot"
    ]
    gate = _MODULE._signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.PILOT_GATE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "materialization_checksum": materialization["checksum"],
            "frozen_contract_sha256": _MODULE.CONTRACT_SHA256,
            "passed": True,
            "fold": 0,
            "seed": 0,
            "jobs": pilot_jobs,
        }
    )
    gate_path = locked / "pilot_gate_receipt.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert _MODULE._load_pilot_gate(
        gate_path, materialization=materialization
    )["passed"] is True

    mismatched = deepcopy(gate)
    mismatched["jobs"][1]["training_mask_schedule_sha256"] = "4" * 64
    mismatched.pop("checksum")
    mismatched = _MODULE._signed(mismatched)
    gate_path.write_text(json.dumps(mismatched), encoding="utf-8")
    with pytest.raises(
        _MODULE.AdjacencyAblationEnqueueError,
        match="exact passing paired pilot gate",
    ):
        _MODULE._load_pilot_gate(
            gate_path, materialization=materialization
        )

    def attempt_two_sha(item: dict[str, object]) -> str:
        config = dict(
            _MODULE.load_yaml_mapping(
                _MODULE._PROJECT_ROOT / str(item["config_reference"])
            )
        )
        config["attempt"] = 2
        return _MODULE.canonical_sha256(config)

    primary_jobs = [
        {
            "fold": item["fold"],
            "seed": item["seed"],
            "condition": item["condition"],
            "run_id": (
                f"primary-{item['fold']}-{item['seed']}-{item['condition']}"
            ),
            "job_id": f"retry-{item['fold']}-{item['seed']}-{item['condition']}",
            "attempt": 2,
            "config_sha256": attempt_two_sha(item),
            "materialized_config_sha256": item["config_sha256"],
        }
        for item in materialization["jobs"]
        if item["stage"] == "primary"
    ]
    recovery = {
        "plan": {"checksum": "5" * 64},
        "enqueue": {"checksum": "6" * 64},
        "primary": {
            (job["fold"], job["seed"], job["condition"]): {
                "resolved_config_sha256": job["config_sha256"],
                "enqueue": {"retry_job_id": job["job_id"]},
                "materialized_job": {
                    "config_sha256": job["materialized_config_sha256"]
                },
            }
            for job in primary_jobs
        },
    }
    trigger = _MODULE._signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.NULL_TRIGGER_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "materialization_checksum": materialization["checksum"],
            "recovery_plan_checksum": recovery["plan"]["checksum"],
            "recovery_enqueue_checksum": recovery["enqueue"]["checksum"],
            "frozen_contract_sha256": _MODULE.CONTRACT_SHA256,
            "triggered": True,
            "graph_huber_lower_than_isolated": True,
            "graph_minus_isolated_huber": -0.01,
            "scientific_audit_passed": True,
            "scientific_audit_errors": [],
            "primary_jobs": primary_jobs,
        }
    )
    trigger_path = locked / "null_trigger_receipt.json"
    trigger_path.write_text(json.dumps(trigger), encoding="utf-8")
    assert len(
        _MODULE._load_null_trigger(
            trigger_path, materialization=materialization, recovery=recovery
        )["primary_jobs"]
    ) == 50

    unfavorable = deepcopy(trigger)
    unfavorable["graph_minus_isolated_huber"] = 0.0
    unfavorable.pop("checksum")
    unfavorable = _MODULE._signed(unfavorable)
    trigger_path.write_text(json.dumps(unfavorable), encoding="utf-8")
    with pytest.raises(
        _MODULE.AdjacencyAblationEnqueueError,
        match="favorable aggregate",
    ):
        _MODULE._load_null_trigger(
            trigger_path, materialization=materialization, recovery=recovery
        )

    with pytest.raises(_MODULE.AdjacencyAblationEnqueueError):
        _MODULE.enqueue_stage(
            stage="primary",
            materialization_path=locked / "materialization_receipt.json",
            database_path=database,
            receipt_path=locked / "primary_enqueue_receipt.json",
            pilot_gate_path=locked / "missing-pilot-gate.json",
            null_trigger_path=trigger_path,
        )
    with Registry(database).connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM queue_jobs").fetchone()[0] == 0


def test_null_gate_verifies_exact_completed_retry_queue_inventory() -> None:
    class Rows:
        def __init__(self, identifiers: set[str]) -> None:
            self.identifiers = identifiers

        def fetchall(self) -> list[dict[str, str]]:
            return [{"job_id": identifier} for identifier in self.identifiers]

    class Connection:
        def __init__(self, registry: "FakeRegistry") -> None:
            self.registry = registry

        def __enter__(self) -> "Connection":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def execute(self, *_: object) -> Rows:
            return Rows(self.registry.retry_ids)

    class FakeRegistry:
        def __init__(self) -> None:
            self.jobs: dict[str, dict[str, object]] = {}
            self.runs: dict[str, dict[str, object]] = {}
            self.retry_ids: set[str] = set()

        def connect(self) -> Connection:
            return Connection(self)

        def get_job(self, identifier: str) -> dict[str, object] | None:
            return self.jobs.get(identifier)

        def get_run(self, identifier: str) -> dict[str, object] | None:
            return self.runs.get(identifier)

    registry = FakeRegistry()
    authorized: dict[tuple[int, int, str], dict[str, object]] = {}
    gate_jobs: list[dict[str, object]] = []
    for fold in _MODULE.FOLDS:
        for seed in _MODULE.SEEDS:
            for condition in _MODULE.CONDITIONS:
                slot = (fold, seed, condition)
                worker_slot = (fold * len(_MODULE.SEEDS) + seed) % 8
                root_job_id = f"q_root_{fold}_{seed}_{condition}"
                root_run_id = f"r_root_{fold}_{seed}_{condition}"
                retry_job_id = f"q_retry_{fold}_{seed}_{condition}"
                retry_run_id = f"r_retry_{fold}_{seed}_{condition}"
                root_config = {
                    "fold": fold,
                    "seed": seed,
                    "condition": condition,
                    "attempt": 1,
                }
                resolved_config = {**root_config, "attempt": 2}
                resolved_sha = _MODULE.canonical_sha256(resolved_config)
                registry.jobs[root_job_id] = {
                    "job_id": root_job_id,
                    "run_id": root_run_id,
                    "canonical_config": root_config,
                }
                registry.jobs[retry_job_id] = {
                    "job_id": retry_job_id,
                    "status": "completed",
                    "run_id": retry_run_id,
                    "retry_of": root_job_id,
                    "attempt_count": 2,
                    "maximum_attempts": 2,
                    "requested_gpu": str(worker_slot),
                    "canonical_config": root_config,
                }
                registry.runs[retry_run_id] = {
                    "run_id": retry_run_id,
                    "status": "completed",
                    "attempt": 2,
                    "retry_of": root_run_id,
                    "config": resolved_config,
                }
                registry.retry_ids.add(retry_job_id)
                authorized[slot] = {
                    "plan": {
                        "root_job_id": root_job_id,
                        "worker_slot": worker_slot,
                    },
                    "enqueue": {"retry_job_id": retry_job_id},
                    "resolved_config_sha256": resolved_sha,
                }
                gate_jobs.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "condition": condition,
                        "run_id": retry_run_id,
                        "job_id": retry_job_id,
                        "config_sha256": resolved_sha,
                    }
                )
    recovery = {"primary": authorized}
    _MODULE._verify_completed_recovery_gate_runs(
        registry=registry, recovery=recovery, gate_jobs=gate_jobs
    )

    registry.retry_ids.add("q_unauthorized")
    with pytest.raises(
        _MODULE.AdjacencyAblationEnqueueError,
        match="missing or unauthorized primary retry rows",
    ):
        _MODULE._verify_completed_recovery_gate_runs(
            registry=registry, recovery=recovery, gate_jobs=gate_jobs
        )
    registry.retry_ids.remove("q_unauthorized")
    first_retry = str(gate_jobs[0]["job_id"])
    registry.jobs[first_retry]["maximum_attempts"] = 3
    with pytest.raises(
        _MODULE.AdjacencyAblationEnqueueError,
        match="not the authorized completed retry",
    ):
        _MODULE._verify_completed_recovery_gate_runs(
            registry=registry, recovery=recovery, gate_jobs=gate_jobs
        )
