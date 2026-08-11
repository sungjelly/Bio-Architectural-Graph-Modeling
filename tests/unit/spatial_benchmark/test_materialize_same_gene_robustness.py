from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import sys
from typing import Any

import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.identifiers import create_run_id, scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import RunArchive


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/materialize_same_gene_robustness.py"
_SPEC = importlib.util.spec_from_file_location(
    "materialize_same_gene_robustness_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_LAUNCHER_SCRIPT = _ROOT / "scripts/train/launch_same_gene_robustness.py"
_LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "launch_same_gene_robustness_for_materializer_tests", _LAUNCHER_SCRIPT
)
assert _LAUNCHER_SPEC is not None and _LAUNCHER_SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_LAUNCHER_SPEC)
sys.modules[_LAUNCHER_SPEC.name] = _LAUNCHER
_LAUNCHER_SPEC.loader.exec_module(_LAUNCHER)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _synthetic_launcher_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def verified(path: Path, *, visibility_mode: str) -> dict[str, Any]:
        assert visibility_mode == "launcher"
        return {
            "schema_version": 1,
            "verified": True,
            "visibility_mode": "launcher",
            "environment_lock_sha256": _sha(Path(path)),
            "observation": {"synthetic": True},
            "verification_sha256": "e" * 64,
        }

    monkeypatch.setattr(_MODULE, "verify_live_environment", verified)


def test_seed_by_fold_execution_streams_are_globally_unique() -> None:
    execution_seeds = {
        seed_base + fold
        for seed_base in _MODULE.SEEDS
        for fold in _MODULE.FOLDS
    }

    assert len(_MODULE.SEEDS) == 5
    assert len(execution_seeds) == len(_MODULE.SEEDS) * len(_MODULE.FOLDS) == 20
    assert len({seed % 1_000_000 for seed in _MODULE.SEEDS}) == len(_MODULE.SEEDS)


@pytest.mark.parametrize(
    "value",
    (None, "", "2026-08-10T23:59:59+00:00", "2026-02-30T00:00:00Z"),
)
def test_contract_freeze_timestamp_is_strict_and_non_null(value: object) -> None:
    with pytest.raises(_MODULE.SameGeneMaterializationError, match="frozen_at"):
        _MODULE._require_frozen_at({"frozen_at": value})
    assert _MODULE._require_frozen_at(
        {"frozen_at": "2026-08-10T23:59:59Z"}
    ) == "2026-08-10T23:59:59Z"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _authority_fixture(root: Path, *, authorized: bool = True) -> dict[str, Any]:
    for relative in sorted(_MODULE.REQUIRED_SOURCES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative != _MODULE.CONTRACT_RELATIVE_PATH:
            path.write_text(
                f"# synthetic source for {relative}\n", encoding="utf-8"
            )

    variant_roots: dict[str, Path] = {}
    variant_authorities: dict[str, dict[str, str]] = {}
    robustness_root = root / "prepared/same_gene_robustness_v1"
    for variant in _MODULE.VARIANTS:
        variant_root = robustness_root / f"variants/{variant.lower()}"
        variant_spec = {
            "graph": {"partition": "within_fov"},
            "normalization": {"name": "log1p"},
            "node_policy": {"name": "all"},
            "permutation_seed": 20260810,
        }
        integrity = variant_root / "integrity_manifest.json"
        _write_json(
            integrity,
            {
                "contract_variant_id": variant,
                "variant_fingerprint": "2" * 64,
                "variant_spec": variant_spec,
            },
        )
        manifest = variant_root / "manifest.json"
        _write_json(
            manifest,
            {
                "variant_id": variant,
                "preprocessing_version": f"synthetic_{variant.lower()}_v1",
                "raw_snapshot": {"fingerprint": "1" * 64},
                "processed_fingerprint": "2" * 64,
                "split_fingerprint": "3" * 64,
                "variant_spec": variant_spec,
                "base_prepared_manifest_sha256": "4" * 64,
            },
        )
        variant_authorities[variant] = {
            "manifest_sha256": _sha(manifest),
            "integrity_manifest_sha256": _sha(integrity),
            "processed_fingerprint": "2" * 64,
            "variant_spec_sha256": canonical_sha256(variant_spec),
        }
        variant_roots[variant] = variant_root
    root_payload = {
        "manifest_schema_version": 1,
        "variants": {
            variant.lower(): {
                "path": f"variants/{variant.lower()}",
                "contract_variant_id": variant,
            }
            for variant in _MODULE.VARIANTS
        },
    }
    root_payload["processed_fingerprint"] = canonical_sha256(root_payload)
    root_manifest = robustness_root / "manifest.json"
    _write_json(root_manifest, root_payload)

    contract = root / _MODULE.CONTRACT_RELATIVE_PATH
    contract.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign_id": _MODULE.CAMPAIGN_ID,
                "status": (
                    "frozen_preoutcome" if authorized else "draft_preoutcome"
                ),
                "launch_authorized": authorized,
                "frozen_at": "2026-08-10T23:59:59Z",
                "authorization": {
                    "required_status_to_launch": "frozen_preoutcome"
                },
                "dataset": {
                    "raw_fingerprint": "1" * 64,
                    "split_fingerprint": "3" * 64,
                    "base_prepared_manifest_sha256": "4" * 64,
                    "prepared_variant_fingerprints": {
                        "status": "complete_preoutcome_identity_binding",
                        "robustness_root_manifest_sha256": _sha(root_manifest),
                        "robustness_root_processed_fingerprint": root_payload[
                            "processed_fingerprint"
                        ],
                        "variants": variant_authorities,
                    },
                },
                "pilots": {
                    "fail_closed_gates": {
                        "maximum_projected_hours_per_full_fold": 0.25
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_list = root / "authority/source-list.json"
    _write_json(source_list, sorted(_MODULE.REQUIRED_SOURCES))
    launch = root / "authority/launch.json"
    if authorized:
        _MODULE.build_launch_manifest(
            contract=contract,
            source_list=source_list,
            output=launch,
            project_root=root,
        )
    return {
        "contract": contract,
        "source_list": source_list,
        "launch": launch,
        "variant_roots": variant_roots,
    }


def _pilot_plan(root: Path, fixture: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = root / "authority/pilot-plan.json"
    payload = _MODULE.build_job_plan(
        profile="pilot",
        variant_roots=fixture["variant_roots"],
        launch_manifest=fixture["launch"],
        output_dir=root / "state/materialized",
        plan=path,
        project_root=root,
    )
    return path, payload


def _make_pilot_bundles(root: Path, plan: dict[str, Any]) -> None:
    for job in plan["jobs"]:
        config_path = root / job["argv"][3]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        variant = Path(config["variant_root"]).name.upper()
        run_id = f"pilot-{variant.lower()}"
        artifact = root / f"artifacts/{run_id}"
        success = artifact / "_SUCCESS"
        success.parent.mkdir(parents=True, exist_ok=True)
        success.write_text('{"status":"success"}\n', encoding="utf-8")
        results = artifact / "results.json"
        _write_json(
            results,
            {
                "run_id": run_id,
                "status": "completed",
                "profile": "pilot",
                "statistical_evaluation_role": "resource_validation",
                "controls": {
                    "all_outputs_finite": True,
                    "train_validation_test_component_overlap": False,
                    "receiver_rna_or_derived_covariate_model_input": False,
                    "identity_oracle_row_top1_fraction": 1.0,
                    "identity_oracle_actually_executed": True,
                    "analytical_nonlinear_jacobian": {
                        "maximum_autograd_error": 1e-12,
                        "maximum_finite_difference_error": 1e-10,
                        "passed": True,
                    },
                    "graph_specific_invariants": True,
                    "checkpoint_gpu_replay_max_abs_metric_error": 1e-9,
                    "checkpoint_gpu_replay_max_abs_prediction_error": 1e-9,
                    "checkpoint_replay_device_type": "cuda",
                    "canonical_production_split_label": "test",
                    "source_config_data_hashes_verified": True,
                    "outer_test_untouched": True,
                    "environment_lock_verified": True,
                    "environment_lock_sha256": _sha(
                        root / _MODULE.ENVIRONMENT_LOCK_RELATIVE_PATH
                    ),
                    "environment_verification_sha256": "e" * 64,
                    "environment_visibility_mode": "job",
                    "peak_vram_gb": 8.0,
                    "projected_full_hours_per_fold": 0.2,
                },
                # This sentinel must never enter a receipt or console output.
                "arms": {"observed_near": {"scientific_effect": 987654321}},
            },
        )
        bundle_manifest = artifact / "provenance/artifact_checksums.json"
        _write_json(
            bundle_manifest,
            {
                "version": 1,
                "files": {
                    "results.json": {
                        "type": "file",
                        "size": results.stat().st_size,
                        "sha256": _sha(results),
                    }
                },
            },
        )
        marker_payload = {
            "schema_version": 1,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "config_sha256": _sha(config_path),
            "run_id": run_id,
            "artifact_path": artifact.relative_to(root).as_posix(),
            "artifact_success_sha256": _sha(success),
        }
        _write_json(
            root / job["expected_success_marker"],
            {
                "payload": marker_payload,
                "marker_sha256": canonical_sha256(marker_payload),
            },
        )


def test_build_launch_is_wrapper_compatible_and_binds_transitive_sources(
    tmp_path: Path,
) -> None:
    fixture = _authority_fixture(tmp_path)
    payload = json.loads(fixture["launch"].read_text(encoding="utf-8"))

    assert set(payload) == {"campaign_id", "contract", "sources"}
    assert payload["campaign_id"] == _MODULE.CAMPAIGN_ID
    source_paths = {row["path"] for row in payload["sources"]}
    assert _MODULE.REQUIRED_SOURCES.issubset(source_paths)
    assert _MODULE.PHASE_TRANSFORM_RELATIVE_PATH in source_paths
    assert _MODULE.RESIDUALIZATION_RELATIVE_PATH in source_paths
    assert _MODULE.BASE_ANALYZER_RELATIVE_PATH in source_paths
    assert payload["sources"] == sorted(
        payload["sources"], key=lambda row: row["path"]
    )
    assert all(
        row == {
            "path": row["path"],
            "size": (tmp_path / row["path"]).stat().st_size,
            "sha": _sha(tmp_path / row["path"]),
        }
        for row in payload["sources"]
    )


def test_plan_materialization_fails_closed_on_live_environment_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authority_fixture(tmp_path)

    def drifted(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise _MODULE.EnvironmentLockError("synthetic missing package")

    monkeypatch.setattr(_MODULE, "verify_live_environment", drifted)
    with pytest.raises(
        _MODULE.SameGeneMaterializationError, match="live environment"
    ):
        _MODULE.build_job_plan(
            profile="pilot",
            variant_roots=fixture["variant_roots"],
            launch_manifest=fixture["launch"],
            output_dir=tmp_path / "state/materialized",
            plan=tmp_path / "authority/drifted-plan.json",
            project_root=tmp_path,
        )


def test_build_launch_rejects_draft_missing_and_escaping_sources(
    tmp_path: Path,
) -> None:
    draft = _authority_fixture(tmp_path / "draft", authorized=False)
    with pytest.raises(_MODULE.SameGeneMaterializationError, match="frozen"):
        _MODULE.build_launch_manifest(
            contract=draft["contract"],
            source_list=draft["source_list"],
            output=tmp_path / "draft/authority/launch.json",
            project_root=tmp_path / "draft",
        )

    fixture = _authority_fixture(tmp_path / "missing")
    sources = json.loads(fixture["source_list"].read_text(encoding="utf-8"))
    sources.remove(_MODULE.PHASE_TRANSFORM_RELATIVE_PATH)
    _write_json(fixture["source_list"], sources)
    with pytest.raises(_MODULE.SameGeneMaterializationError, match="omits"):
        _MODULE.build_launch_manifest(
            contract=fixture["contract"],
            source_list=fixture["source_list"],
            output=tmp_path / "missing/authority/other-launch.json",
            project_root=tmp_path / "missing",
        )

    sources.append("../escape.py")
    _write_json(fixture["source_list"], sources)
    with pytest.raises(_MODULE.SameGeneMaterializationError, match="relative"):
        _MODULE.build_launch_manifest(
            contract=fixture["contract"],
            source_list=fixture["source_list"],
            output=tmp_path / "missing/authority/escape-launch.json",
            project_root=tmp_path / "missing",
        )


def test_frozen_launch_rejects_incomplete_prepared_identity_binding(
    tmp_path: Path,
) -> None:
    fixture = _authority_fixture(tmp_path)
    contract = yaml.safe_load(fixture["contract"].read_text(encoding="utf-8"))
    contract["dataset"]["prepared_variant_fingerprints"]["status"] = (
        "pending_preoutcome_identity_binding"
    )
    fixture["contract"].write_text(
        yaml.safe_dump(contract, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(
        _MODULE.SameGeneMaterializationError,
        match="not complete_preoutcome_identity_binding",
    ):
        _MODULE.build_launch_manifest(
            contract=fixture["contract"],
            source_list=fixture["source_list"],
            output=tmp_path / "authority/incomplete-launch.json",
            project_root=tmp_path,
        )


def test_build_launch_rejects_an_omitted_transitive_local_import(
    tmp_path: Path,
) -> None:
    fixture = _authority_fixture(tmp_path)
    imported = tmp_path / "src/spatial_benchmark/synthetic_dependency.py"
    imported.parent.mkdir(parents=True, exist_ok=True)
    imported.write_text("VALUE = 1\n", encoding="utf-8")
    wrapper = tmp_path / _MODULE.WRAPPER_RELATIVE_PATH
    wrapper.write_text(
        "from spatial_benchmark.synthetic_dependency import VALUE\n",
        encoding="utf-8",
    )

    with pytest.raises(_MODULE.SameGeneMaterializationError, match="omits"):
        _MODULE.build_launch_manifest(
            contract=fixture["contract"],
            source_list=fixture["source_list"],
            output=tmp_path / "authority/transitive-launch.json",
            project_root=tmp_path,
        )


def test_build_plan_materializes_exact_pilot_slots_and_wrapper_argv(
    tmp_path: Path,
) -> None:
    fixture = _authority_fixture(tmp_path)
    plan_path, plan = _pilot_plan(tmp_path, fixture)

    assert plan_path.is_file()
    assert plan["minimum_free_disk_gb"] == 40
    assert plan["source_manifest"] == {
        "path": fixture["launch"].relative_to(tmp_path).as_posix(),
        "sha256": _sha(fixture["launch"]),
    }
    assert plan["environment_lock"] == {
        "path": _MODULE.ENVIRONMENT_LOCK_RELATIVE_PATH,
        "sha256": _sha(tmp_path / _MODULE.ENVIRONMENT_LOCK_RELATIVE_PATH),
    }
    assert len(plan["jobs"]) == 7
    observed = set()
    for job in plan["jobs"]:
        assert job["gpu"] == "auto"
        assert job["argv"][:3] == [
            sys.executable,
            _MODULE.WRAPPER_RELATIVE_PATH,
            "--config",
        ]
        assert job["verify_argv"] == [*job["argv"], "--verify-job-marker"]
        assert sum(value == "--config" for value in job["argv"]) == 1
        config_path = tmp_path / job["argv"][3]
        assert job["expected_config_sha256"] == _sha(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        variant = Path(config["variant_root"]).name.upper()
        observed.add((variant, config["model_seed"], config["fold"]))
        assert config["profile"] == "pilot"
        assert config["pilot_receipt"] is None
        assert config["job_success_marker"] == job["expected_success_marker"]
    assert observed == {
        (variant, _MODULE.PILOT_SEED, _MODULE.PILOT_FOLD)
        for variant in _MODULE.VARIANTS
    }
    coordinator_plan = _LAUNCHER.load_plan(plan_path, project_root=tmp_path)
    assert len(coordinator_plan.jobs) == 7
    assert coordinator_plan.minimum_free_disk_gb == 40


def test_retry_plan_requires_explicit_slots_and_never_reuses_attempt_one_paths(
    tmp_path: Path,
) -> None:
    fixture = _authority_fixture(tmp_path)
    with pytest.raises(_MODULE.SameGeneMaterializationError, match="explicit retry"):
        _MODULE.build_job_plan(
            profile="pilot",
            variant_roots=fixture["variant_roots"],
            launch_manifest=fixture["launch"],
            output_dir=tmp_path / "state/materialized",
            plan=tmp_path / "authority/retry-plan.json",
            project_root=tmp_path,
            attempt=2,
        )

    retry = _MODULE.build_job_plan(
        profile="pilot",
        variant_roots=fixture["variant_roots"],
        launch_manifest=fixture["launch"],
        output_dir=tmp_path / "state/materialized",
        plan=tmp_path / "authority/retry-plan.json",
        project_root=tmp_path,
        attempt=2,
        retry_slots=(("V3", _MODULE.PILOT_SEED, _MODULE.PILOT_FOLD),),
    )
    assert retry["plan_id"].endswith("pilot-retry-a2-v1")
    assert len(retry["jobs"]) == 1
    job = retry["jobs"][0]
    assert job["job_id"].endswith("-a2")
    config = json.loads((tmp_path / job["argv"][3]).read_text(encoding="utf-8"))
    assert config["attempt"] == 2
    assert "-a2.json" in job["expected_success_marker"]


@pytest.mark.parametrize(
    "field",
    [
        "manifest_sha256",
        "integrity_manifest_sha256",
        "processed_fingerprint",
        "variant_spec_sha256",
        "raw_fingerprint",
        "split_fingerprint",
        "base_prepared_manifest_sha256",
        "robustness_root_manifest_sha256",
        "robustness_root_processed_fingerprint",
    ],
)
def test_materializer_rejects_every_contract_prepared_identity_mismatch(
    tmp_path: Path, field: str
) -> None:
    fixture = _authority_fixture(tmp_path)
    launch = _MODULE._verify_launch(fixture["launch"], project_root=tmp_path)
    binding = launch.prepared_binding
    if field in binding.variants["V0"]:
        variants = {
            key: dict(value) for key, value in binding.variants.items()
        }
        variants["V0"][field] = "f" * 64
        binding = replace(binding, variants=variants)
    else:
        binding = replace(binding, **{field: "f" * 64})

    with pytest.raises(_MODULE.SameGeneMaterializationError, match="frozen contract"):
        _MODULE._variant_identity(
            "V0",
            fixture["variant_roots"]["V0"],
            project_root=tmp_path,
            binding=binding,
        )


def test_receipts_reverify_all_markers_and_full_plan_has_140_bound_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _authority_fixture(tmp_path)
    pilot_path, pilot = _pilot_plan(tmp_path, fixture)
    _make_pilot_bundles(tmp_path, pilot)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def verified(command: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(_MODULE.subprocess, "run", verified)
    receipts = _MODULE.build_pilot_receipts(
        pilot_plan=pilot_path,
        output_dir=tmp_path / "authority/receipts",
        project_root=tmp_path,
    )

    assert capsys.readouterr() == ("", "")
    assert set(receipts) == set(_MODULE.VARIANTS)
    assert len(calls) == 7
    expected_verify = {tuple(job["verify_argv"]) for job in pilot["jobs"]}
    assert {tuple(command) for command, _ in calls} == expected_verify
    assert all(
        kwargs["shell"] is False
        and kwargs["stdout"] is _MODULE.subprocess.PIPE
        and kwargs["stderr"] is _MODULE.subprocess.PIPE
        and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        and kwargs["env"]["PYTHONPATH"] == str(tmp_path / "src")
        for _, kwargs in calls
    )
    for variant, receipt_path in receipts.items():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["receipt_sha256"] == canonical_sha256(receipt["payload"])
        assert receipt["payload"]["variant"] == variant
        assert set(receipt["payload"]["bundle"]) == set(_MODULE._BUNDLE_KEYS)
        assert set(receipt["payload"]["controls"]) == set(_MODULE._CONTROL_KEYS)
        assert "arms" not in json.dumps(receipt)
        assert "987654321" not in json.dumps(receipt)

    full_path = tmp_path / "authority/full-plan.json"
    full = _MODULE.build_job_plan(
        profile="full",
        variant_roots=fixture["variant_roots"],
        variant_receipts=receipts,
        launch_manifest=fixture["launch"],
        output_dir=tmp_path / "state/materialized",
        plan=full_path,
        project_root=tmp_path,
    )
    assert len(full["jobs"]) == 7 * 5 * 4
    assert full["projected_output_bytes"] >= 5 * 1024**3
    full_slots = {
        (
            Path(
                json.loads((tmp_path / job["argv"][3]).read_text())["variant_root"]
            ).name.upper(),
            json.loads((tmp_path / job["argv"][3]).read_text())["model_seed"],
            json.loads((tmp_path / job["argv"][3]).read_text())["fold"],
        )
        for job in full["jobs"]
    }
    assert full_slots == {
        (variant, seed, fold)
        for variant in _MODULE.VARIANTS
        for seed in _MODULE.SEEDS
        for fold in _MODULE.FOLDS
    }
    floor = _MODULE.MINIMUM_FREE_DISK_GB * 1024**3
    free = floor + full["projected_output_bytes"] - 1
    monkeypatch.setattr(
        _MODULE.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=free, used=0, free=free),
    )
    with pytest.raises(_MODULE.SameGeneMaterializationError, match="ending-free"):
        _MODULE.build_job_plan(
            profile="full",
            variant_roots=fixture["variant_roots"],
            variant_receipts=receipts,
            launch_manifest=fixture["launch"],
            output_dir=tmp_path / "state/materialized",
            plan=tmp_path / "authority/full-plan-low-disk.json",
            project_root=tmp_path,
        )
    assert not (tmp_path / "authority/full-plan-low-disk.json").exists()


def test_pilot_receipts_select_latest_retry_and_bind_terminal_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _authority_fixture(tmp_path)
    first_path, first = _pilot_plan(tmp_path, fixture)
    _make_pilot_bundles(tmp_path, first)
    first_v3 = next(job for job in first["jobs"] if "-V3-" in job["job_id"])
    (tmp_path / first_v3["expected_success_marker"]).unlink()

    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})
    failed_id = create_run_id(
        seed=_MODULE.PILOT_SEED % 1_000_000,
        fold=0,
        attempt=1,
        scientific_id_value="sci_12345678",
        unique_suffix="failedv3",
    )
    failed = RunArchive.create(failed_id, paths=paths)
    failed.finalize_failure("synthetic pilot failure")
    registry_path = tmp_path / "state/tracking/bagm.sqlite3"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(registry_path)
    try:
        connection.execute(
            """
            CREATE TABLE runs(
                run_id TEXT, campaign_id TEXT, seed INTEGER, fold INTEGER,
                attempt INTEGER, status TEXT, artifact_path TEXT, config_json TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                failed_id,
                _MODULE.CAMPAIGN_ID,
                _MODULE.PILOT_SEED % 1_000_000,
                0,
                1,
                "failed",
                str(failed.artifact_path),
                json.dumps(
                    {
                        "robustness_variant": {"variant_id": "V3"},
                        "classification": {
                            "variant_label": "same_gene_robustness_V3_pilot"
                        },
                    }
                ),
            ),
        )
        connection.commit()
    finally:
        connection.close()

    retry_path = tmp_path / "authority/pilot-retry-a2.json"
    retry = _MODULE.build_job_plan(
        profile="pilot",
        variant_roots=fixture["variant_roots"],
        launch_manifest=fixture["launch"],
        output_dir=tmp_path / "state/materialized",
        plan=retry_path,
        project_root=tmp_path,
        attempt=2,
        retry_slots=(("V3", _MODULE.PILOT_SEED, _MODULE.PILOT_FOLD),),
    )
    _make_pilot_bundles(tmp_path, retry)
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )
    receipts = _MODULE.build_pilot_receipts(
        pilot_plan=(first_path, retry_path),
        output_dir=tmp_path / "authority/retry-receipts",
        project_root=tmp_path,
    )
    v3 = json.loads(receipts["V3"].read_text(encoding="utf-8"))["payload"]
    assert v3["selected_attempt"] == 2
    assert [row["attempt"] for row in v3["attempt_history"]] == [1, 2]
    assert [row["registry_status"] for row in v3["attempt_history"]] == [
        "failed",
        "completed",
    ]
    assert v3["attempt_history_sha256"] == canonical_sha256(
        v3["attempt_history"]
    )
    assert all(
        json.loads(receipts[variant].read_text(encoding="utf-8"))["payload"][
            "selected_attempt"
        ]
        == 1
        for variant in _MODULE.VARIANTS
        if variant != "V3"
    )


def test_full_plan_requires_exact_receipt_mapping(tmp_path: Path) -> None:
    fixture = _authority_fixture(tmp_path)
    with pytest.raises(_MODULE.SameGeneMaterializationError, match="requires"):
        _MODULE.build_job_plan(
            profile="full",
            variant_roots=fixture["variant_roots"],
            launch_manifest=fixture["launch"],
            output_dir=tmp_path / "state/materialized",
            plan=tmp_path / "authority/full-plan.json",
            project_root=tmp_path,
        )

    incomplete = {variant: tmp_path / f"{variant}.json" for variant in _MODULE.VARIANTS[:-1]}
    with pytest.raises(_MODULE.SameGeneMaterializationError, match="exactly"):
        _MODULE.build_job_plan(
            profile="full",
            variant_roots=fixture["variant_roots"],
            variant_receipts=incomplete,
            launch_manifest=fixture["launch"],
            output_dir=tmp_path / "state/materialized",
            plan=tmp_path / "authority/full-plan.json",
            project_root=tmp_path,
        )


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "config_sha"])
def test_receipt_builder_fails_on_missing_duplicate_or_tampered_slots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    fixture = _authority_fixture(tmp_path)
    pilot_path, pilot = _pilot_plan(tmp_path, fixture)
    _make_pilot_bundles(tmp_path, pilot)
    changed = deepcopy(pilot)
    if tamper == "missing":
        changed["jobs"].pop()
    elif tamper == "duplicate":
        changed["jobs"][-1] = deepcopy(changed["jobs"][0])
    else:
        changed["jobs"][0]["expected_config_sha256"] = "0" * 64
    _write_json(pilot_path, changed)
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    with pytest.raises(_MODULE.SameGeneMaterializationError):
        _MODULE.build_pilot_receipts(
            pilot_plan=pilot_path,
            output_dir=tmp_path / "authority/receipts",
            project_root=tmp_path,
        )


def test_receipt_builder_rejects_failed_controls_without_exposing_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _authority_fixture(tmp_path)
    pilot_path, pilot = _pilot_plan(tmp_path, fixture)
    _make_pilot_bundles(tmp_path, pilot)
    first = pilot["jobs"][0]
    config = json.loads((tmp_path / first["argv"][3]).read_text())
    run_id = f"pilot-{Path(config['variant_root']).name}"
    results = tmp_path / f"artifacts/{run_id}/results.json"
    payload = json.loads(results.read_text(encoding="utf-8"))
    payload["controls"]["all_outputs_finite"] = False
    _write_json(results, payload)
    # Rebind the checksum manifest so the failure reaches the control gate.
    manifest = results.parent / "provenance/artifact_checksums.json"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload["files"]["results.json"] = {
        "type": "file",
        "size": results.stat().st_size,
        "sha256": _sha(results),
    }
    _write_json(manifest, manifest_payload)
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    with pytest.raises(_MODULE.SameGeneMaterializationError, match="finite-output"):
        _MODULE.build_pilot_receipts(
            pilot_plan=pilot_path,
            output_dir=tmp_path / "authority/receipts",
            project_root=tmp_path,
        )
    captured = capsys.readouterr()
    assert "987654321" not in captured.out
    assert "987654321" not in captured.err


def _cross_launch_receipt_fixture(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mock_selected_run: bool = True,
) -> dict[str, Any]:
    parent_launch_path = root / "authority/parent-launch.json"
    child_launch_path = root / "authority/child-launch.json"
    parent_plan_path = root / "authority/parent-plan.json"
    child_plan_path = root / "authority/child-plan.json"
    amendment_path = root / _MODULE.TECHNICAL_AMENDMENT_RELATIVE_PATH
    source_list_path = root / "authority/recovery-source-list.json"
    parent_ledger_path = root / "authority/parent-ledger.json"
    environment_lock_path = root / _MODULE.ENVIRONMENT_LOCK_RELATIVE_PATH
    for path, payload in (
        (parent_launch_path, {"launch": "parent"}),
        (child_launch_path, {"launch": "child"}),
        (parent_plan_path, {"plan": "attempt-one"}),
        (child_plan_path, {"plan": "attempt-two"}),
        (amendment_path, {"amendment": "recovery"}),
        (source_list_path, ["synthetic"]),
        (parent_ledger_path, {"ledger": "failed-parent"}),
        (environment_lock_path, {"environment": "synthetic"}),
    ):
        _write_json(path, payload)
    binding = _MODULE.PreparedContractBinding(
        raw_fingerprint="1" * 64,
        split_fingerprint="2" * 64,
        base_prepared_manifest_sha256="3" * 64,
        robustness_root_manifest_sha256="4" * 64,
        robustness_root_processed_fingerprint="5" * 64,
        variants={
            variant: {
                "manifest_sha256": "6" * 64,
                "integrity_manifest_sha256": "7" * 64,
                "processed_fingerprint": "8" * 64,
                "variant_spec_sha256": "9" * 64,
            }
            for variant in _MODULE.VARIANTS
        },
    )
    failed_attempts = {
        variant: {
            "variant": variant,
            "attempt": 1,
            "model_seed": _MODULE.PILOT_SEED,
            "fold": _MODULE.PILOT_FOLD,
            "scientific_id": f"sci_{index:016x}",
            "run_id": f"parent-failed-{variant.lower()}",
            "materialized_config_sha256": f"{index + 1:064x}",
            "artifact_path": f"artifacts/parent-failed-{variant.lower()}",
            "failed_marker_sha256": f"{index + 11:064x}",
            "exception_sha256": f"{index + 21:064x}",
            "registry_status": "failed",
            "artifact_status": "failed",
            "failure_category": "same_gene_nonlinear_run_failure",
        }
        for index, variant in enumerate(_MODULE.VARIANTS)
    }
    amendment = _MODULE.TechnicalAmendment(
        path=amendment_path,
        sha256=_sha(amendment_path),
        payload={"synthetic": True},
        contract_sha256="a" * 64,
        parent_launch_path=parent_launch_path,
        parent_launch_sha256=_sha(parent_launch_path),
        parent_source_manifest_sha256="b" * 64,
        parent_plan_path=parent_plan_path,
        parent_plan_sha256=_sha(parent_plan_path),
        parent_ledger_path=parent_ledger_path,
        parent_ledger_sha256=_sha(parent_ledger_path),
        parent_git_commit="c" * 40,
        child_launch_path=child_launch_path,
        child_launch_core_sha256="d" * 64,
        child_source_list_path=source_list_path,
        child_source_list_sha256=_sha(source_list_path),
        failed_attempts=failed_attempts,
    )
    parent_launch = _MODULE.LaunchIdentity(
        path=parent_launch_path,
        sha256=_sha(parent_launch_path),
        contract_sha256="a" * 64,
        source_manifest_sha="b" * 64,
        payload={"synthetic": "parent"},
        prepared_binding=binding,
        technical_amendment=None,
    )
    child_launch = _MODULE.LaunchIdentity(
        path=child_launch_path,
        sha256=_sha(child_launch_path),
        contract_sha256="a" * 64,
        source_manifest_sha="e" * 64,
        payload={"synthetic": "child"},
        prepared_binding=binding,
        technical_amendment=amendment,
    )
    variants: dict[str, Any] = {}
    parent_slots: list[Any] = []
    child_slots: list[Any] = []
    artifacts: dict[str, Path] = {}
    for index, variant_id in enumerate(_MODULE.VARIANTS):
        variant_root = root / f"prepared/{variant_id.lower()}"
        manifest = variant_root / "manifest.json"
        integrity = variant_root / "integrity_manifest.json"
        _write_json(manifest, {"variant": variant_id})
        _write_json(integrity, {"variant": variant_id})
        variant = _MODULE.VariantIdentity(
            variant_id=variant_id,
            root=variant_root,
            manifest_path=manifest,
            manifest_sha256="6" * 64,
            integrity_manifest_sha256="7" * 64,
            raw_fingerprint="1" * 64,
            processed_fingerprint="8" * 64,
            split_fingerprint="2" * 64,
            base_prepared_manifest_sha256="3" * 64,
            variant_spec_sha256="9" * 64,
            robustness_root_manifest_sha256="4" * 64,
            robustness_root_processed_fingerprint="5" * 64,
        )
        variants[variant_id] = variant
        artifact = root / f"artifacts/child-{variant_id.lower()}"
        artifact.mkdir(parents=True)
        manifest_path = artifact / "provenance/artifact_checksums.json"
        _write_json(manifest_path, {"variant": variant_id})
        artifacts[variant_id] = artifact
        slot_values: list[Any] = []
        for attempt, launch, plan in (
            (1, parent_launch, parent_plan_path),
            (2, child_launch, child_plan_path),
        ):
            config = root / f"authority/{variant_id.lower()}-a{attempt}.json"
            marker = root / f"authority/{variant_id.lower()}-a{attempt}.marker.json"
            _write_json(config, {"variant": variant_id, "attempt": attempt})
            _write_json(marker, {"marker": True})
            slot_values.append(
                _MODULE.PilotSlot(
                    variant=variant,
                    job_id=(
                        f"same-gene-robustness-pilot-{variant_id}-"
                        f"s{_MODULE.PILOT_SEED}-f0-a{attempt}"
                    ),
                    config_path=config,
                    config_sha256=_sha(config),
                    marker_path=marker,
                    verify_argv=("synthetic", variant_id, str(attempt)),
                    launch=launch,
                    attempt=attempt,
                    plan_path=plan,
                )
            )
        parent_slots.append(slot_values[0])
        child_slots.append(slot_values[1])

    def plan_reference(
        value: str | Path, *, project_root: Path
    ) -> tuple[Path, Path, str]:
        del project_root
        path = Path(value).resolve()
        if path == parent_plan_path:
            return path, parent_launch_path, parent_launch.sha256
        assert path == child_plan_path
        return path, child_launch_path, child_launch.sha256

    def verify_launch(path: str | Path, *, project_root: Path) -> Any:
        del project_root
        if Path(path).resolve() == child_launch_path:
            return child_launch
        raise _MODULE.SameGeneMaterializationError(
            "historical parent live sources changed"
        )

    def pilot_slots(
        path: Path,
        *,
        project_root: Path,
        historical_amendment: Any = None,
    ) -> tuple[Any, ...]:
        del project_root
        assert historical_amendment == amendment
        return tuple(parent_slots if path.resolve() == parent_plan_path else child_slots)

    controls = {
        "all_outputs_finite": True,
        "train_validation_test_component_overlap": False,
        "receiver_rna_or_derived_covariate_model_input": False,
        "identity_oracle_row_top1_fraction": 1.0,
        "identity_oracle_actually_executed": True,
        "analytical_autograd_max_abs_error": 0.0,
        "analytical_finite_difference_max_abs_error": 0.0,
        "graph_specific_invariants": True,
        "checkpoint_gpu_replay_max_abs_metric_error": 0.0,
        "checkpoint_gpu_replay_max_abs_prediction_error": 0.0,
        "checkpoint_replay_device_type": "cuda",
        "canonical_production_split_label": "test",
        "source_config_data_hashes_verified": True,
        "outer_test_untouched": True,
        "peak_vram_gb": 1.0,
        "projected_full_hours_per_fold": 0.1,
        "gate_passed": True,
        "production_authorized": True,
        "environment_lock_verified": True,
        "environment_lock_sha256": _sha(environment_lock_path),
        "environment_verification_sha256": "0" * 64,
        "environment_visibility_mode": "job",
    }
    monkeypatch.setattr(_MODULE, "_pilot_plan_launch_reference", plan_reference)
    monkeypatch.setattr(_MODULE, "_verify_launch", verify_launch)
    monkeypatch.setattr(_MODULE, "_pilot_slots", pilot_slots)
    monkeypatch.setattr(
        _MODULE,
        "_verify_amended_parent_failures",
        lambda *_args, **_kwargs: {
            variant: {"science": variant, "campaign": {"launch": "parent"}}
            for variant in _MODULE.VARIANTS
        },
    )
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=b"", stderr=b""
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_marker_bundle",
        lambda slot, **_kwargs: (
            {"run_id": f"child-success-{slot.variant.variant_id.lower()}"},
            artifacts[slot.variant.variant_id],
            "1" * 64,
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_verify_checksum_manifest",
        lambda artifact: artifact / "provenance/artifact_checksums.json",
    )
    monkeypatch.setattr(
        _MODULE, "_technical_controls", lambda *_args, **_kwargs: dict(controls)
    )
    if mock_selected_run:
        monkeypatch.setattr(
            _MODULE,
            "_verify_amended_selected_run",
            lambda **_kwargs: {"status": "completed"},
        )
    monkeypatch.setattr(
        _MODULE,
        "_terminal_unsuccessful_attempt",
        lambda slot, **_kwargs: {
            "registry_status": "failed",
            "run_id": failed_attempts[slot.variant.variant_id]["run_id"],
            "artifact_path": failed_attempts[slot.variant.variant_id][
                "artifact_path"
            ],
            "artifact_status": "failed",
        },
    )
    return {
        "parent_plan": parent_plan_path,
        "child_plan": child_plan_path,
        "parent_launch": parent_launch,
        "child_launch": child_launch,
        "amendment": amendment,
        "variants": variants,
        "parent_slots": parent_slots,
        "child_slots": child_slots,
        "controls": controls,
    }


def test_cross_launch_receipts_accept_exact_all_seven_a1_to_a2_chain_and_bind_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cross_launch_receipt_fixture(tmp_path, monkeypatch)
    receipts = _MODULE.build_pilot_receipts(
        pilot_plan=(fixture["parent_plan"], fixture["child_plan"]),
        output_dir=tmp_path / "authority/recovery-receipts",
        project_root=tmp_path,
    )

    assert set(receipts) == set(_MODULE.VARIANTS)
    for variant, receipt_path in receipts.items():
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))["payload"]
        assert payload["selected_attempt"] == 2
        assert payload["launch_manifest_sha"] == fixture["child_launch"].sha256
        assert payload["technical_amendment"] == {
            "path": fixture["amendment"].path.relative_to(tmp_path).as_posix(),
            "sha256": fixture["amendment"].sha256,
            "parent_launch_sha256": fixture["parent_launch"].sha256,
            "child_launch_sha256": fixture["child_launch"].sha256,
        }
        assert [row["attempt"] for row in payload["attempt_history"]] == [1, 2]
        assert payload["attempt_history"][1]["retry_of"] == (
            fixture["amendment"].failed_attempts[variant]["run_id"]
        )
        _MODULE._validate_receipt_identity(
            receipt_path,
            launch=fixture["child_launch"],
            variant=fixture["variants"][variant],
            project_root=tmp_path,
        )

    first = receipts["V0"]
    changed = json.loads(first.read_text(encoding="utf-8"))
    changed["payload"]["launch_manifest_sha"] = "0" * 64
    changed["receipt_sha256"] = canonical_sha256(changed["payload"])
    _write_json(first, changed)
    with pytest.raises(
        _MODULE.SameGeneMaterializationError,
        match="technical-amendment binding mismatch",
    ):
        _MODULE._validate_receipt_identity(
            first,
            launch=fixture["child_launch"],
            variant=fixture["variants"]["V0"],
            project_root=tmp_path,
        )


def test_cross_launch_receipts_reject_missing_amendment_before_artifact_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cross_launch_receipt_fixture(
        tmp_path, monkeypatch, mock_selected_run=False
    )
    child_without_amendment = replace(
        fixture["child_launch"], technical_amendment=None
    )

    def no_amendment(path: str | Path, *, project_root: Path) -> Any:
        del project_root
        if Path(path).resolve() == fixture["child_launch"].path:
            return child_without_amendment
        return fixture["parent_launch"]

    monkeypatch.setattr(_MODULE, "_verify_launch", no_amendment)
    with pytest.raises(
        _MODULE.SameGeneMaterializationError,
        match="source-bound child amendment",
    ):
        _MODULE.build_pilot_receipts(
            pilot_plan=(fixture["parent_plan"], fixture["child_plan"]),
            output_dir=tmp_path / "authority/rejected-receipts",
            project_root=tmp_path,
        )


def test_cross_launch_receipts_reject_partial_all_seven_attempt_two_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cross_launch_receipt_fixture(tmp_path, monkeypatch)

    def partial_slots(
        path: Path,
        *,
        project_root: Path,
        historical_amendment: Any = None,
    ) -> tuple[Any, ...]:
        del project_root
        assert historical_amendment == fixture["amendment"]
        if path.resolve() == fixture["parent_plan"]:
            return tuple(fixture["parent_slots"])
        return tuple(fixture["child_slots"][:-1])

    monkeypatch.setattr(_MODULE, "_pilot_slots", partial_slots)
    with pytest.raises(
        _MODULE.SameGeneMaterializationError,
        match="exactly seven shared-child a2 pilots",
    ):
        _MODULE.build_pilot_receipts(
            pilot_plan=(fixture["parent_plan"], fixture["child_plan"]),
            output_dir=tmp_path / "authority/partial-receipts",
            project_root=tmp_path,
        )


def test_amended_child_launch_cannot_replace_the_declared_parent_attempt_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cross_launch_receipt_fixture(tmp_path, monkeypatch)

    def child_only_reference(
        value: str | Path, *, project_root: Path
    ) -> tuple[Path, Path, str]:
        del project_root
        return (
            Path(value).resolve(),
            fixture["child_launch"].path,
            fixture["child_launch"].sha256,
        )

    monkeypatch.setattr(
        _MODULE, "_pilot_plan_launch_reference", child_only_reference
    )
    with pytest.raises(
        _MODULE.SameGeneMaterializationError,
        match="declared parent-to-child amendment",
    ):
        _MODULE.build_pilot_receipts(
            pilot_plan=(fixture["parent_plan"], fixture["child_plan"]),
            output_dir=tmp_path / "authority/child-only-receipts",
            project_root=tmp_path,
        )


def test_amended_child_materializes_only_one_all_seven_attempt_two_pilot_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _cross_launch_receipt_fixture(tmp_path, monkeypatch)
    required = tuple(
        (variant, _MODULE.PILOT_SEED, _MODULE.PILOT_FOLD)
        for variant in _MODULE.VARIANTS
    )

    _MODULE._validate_amended_plan_request(
        fixture["child_launch"],
        profile="pilot",
        attempt=2,
        retry_slots=tuple(reversed(required)),
    )
    _MODULE._validate_amended_plan_request(
        fixture["child_launch"],
        profile="full",
        attempt=1,
        retry_slots=None,
    )
    for attempt, slots in ((1, None), (2, required[:-1]), (3, required)):
        with pytest.raises(
            _MODULE.SameGeneMaterializationError,
            match="one all-seven pilot attempt-two plan",
        ):
            _MODULE._validate_amended_plan_request(
                fixture["child_launch"],
                profile="pilot",
                attempt=attempt,
                retry_slots=slots,
            )


@pytest.mark.parametrize("tamper", ("retry_of", "scientific_payload"))
def test_selected_amended_registry_run_requires_exact_retry_and_scientific_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    fixture = _cross_launch_receipt_fixture(
        tmp_path, monkeypatch, mock_selected_run=False
    )
    slot = fixture["child_slots"][0]
    variant = slot.variant.variant_id
    parent_configuration = {
        "profile": "pilot",
        "science": {"estimand": "same-gene"},
        "robustness_variant": {"variant_id": variant},
        "campaign": {"launch": "parent"},
    }
    child_configuration = deepcopy(parent_configuration)
    child_configuration["campaign"] = {
        "frozen_contract_sha256": fixture["amendment"].contract_sha256,
        "launch_manifest_sha256": fixture["child_launch"].sha256,
        "source_manifest_sha256": fixture["child_launch"].source_manifest_sha,
        "materialized_job_config_sha256": slot.config_sha256,
        "technical_amendment_sha256": fixture["amendment"].sha256,
    }
    previous_run_id = fixture["amendment"].failed_attempts[variant]["run_id"]
    artifact = tmp_path / "artifacts/selected-registry-run"
    row = {
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "seed": _MODULE.PILOT_SEED % 1_000_000,
        "fold": _MODULE.PILOT_FOLD,
        "attempt": 2,
        "status": "completed",
        "retry_of": previous_run_id,
        "artifact_path": str(artifact),
        "scientific_id": scientific_id(child_configuration),
        "configuration": child_configuration,
    }
    if tamper == "retry_of":
        row["retry_of"] = "different-parent"
    else:
        child_configuration["science"]["estimand"] = "changed"
    monkeypatch.setattr(_MODULE, "_registry_run", lambda *_args, **_kwargs: row)

    with pytest.raises(
        _MODULE.SameGeneMaterializationError,
        match="selected amended pilot registry lineage differs",
    ):
        _MODULE._verify_amended_selected_run(
            slot=slot,
            run_id="selected-run",
            artifact=artifact,
            previous_run_id=previous_run_id,
            parent_configuration=parent_configuration,
            amendment=fixture["amendment"],
            project_root=tmp_path,
        )
