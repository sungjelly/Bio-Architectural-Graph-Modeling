from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256, scientific_id


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_PATH = PROJECT_ROOT / "scripts/train/run_same_gene_robustness.py"
_SPEC = importlib.util.spec_from_file_location(
    "same_gene_robustness_runner_tests", WRAPPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
wrapper = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = wrapper
_SPEC.loader.exec_module(wrapper)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _source_row(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "size": path.stat().st_size, "sha": _sha(path)}


@pytest.fixture(autouse=True)
def _synthetic_live_job_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def verified(path: Path, *, visibility_mode: str) -> dict[str, Any]:
        assert visibility_mode == "job"
        return {
            "schema_version": 1,
            "verified": True,
            "visibility_mode": "job",
            "environment_lock_sha256": _sha(Path(path)),
            "observation": {"synthetic": True},
            "verification_sha256": "e" * 64,
        }

    monkeypatch.setattr(wrapper, "verify_live_environment", verified)


def _fixture(
    root: Path,
    *,
    variant_id: str = "V0",
    profile: str = "full",
    residualization: dict[str, Any] | None = None,
) -> dict[str, Path]:
    base_runner = root / wrapper.BASE_RUNNER_RELATIVE_PATH
    robustness_runner = root / wrapper.WRAPPER_RELATIVE_PATH
    phase_transform = root / wrapper.PHASE_TRANSFORM_RELATIVE_PATH
    residualization_source = root / wrapper.RESIDUALIZATION_RELATIVE_PATH
    environment_lock = root / wrapper.ENVIRONMENT_LOCK_RELATIVE_PATH
    environment_verifier = root / wrapper.ENVIRONMENT_VERIFIER_RELATIVE_PATH
    base_runner.parent.mkdir(parents=True, exist_ok=True)
    phase_transform.parent.mkdir(parents=True, exist_ok=True)
    base_runner.write_text("base runner\n", encoding="utf-8")
    robustness_runner.write_text("robust wrapper\n", encoding="utf-8")
    phase_transform.write_text("phase transform\n", encoding="utf-8")
    residualization_source.write_text("residualization\n", encoding="utf-8")
    _write_json(environment_lock, {"synthetic_environment_lock": True})
    environment_verifier.write_text(
        "synthetic environment verifier\n", encoding="utf-8"
    )

    contract = root / wrapper.CONTRACT_RELATIVE_PATH
    contract.parent.mkdir(parents=True, exist_ok=True)
    launch = root / "state/launch.json"

    robustness_root = root / f"data/processed/robustness_{variant_id.lower()}"
    variant_root = robustness_root / f"variants/{variant_id.lower()}"
    variant_manifest = variant_root / "manifest.json"
    variant_spec = {
        "graph": {"partition": "within_fov", "k": 12},
        "normalization": {"name": "log1p_raw"},
        "node_policy": {"name": "all"},
        "permutation_seed": 20260810,
        "primary_eligibility_file": f"eligible_{variant_id.lower()}.npy",
        "feature_files": dict(wrapper.DEFAULT_FEATURE_FILES),
        "residualization": (
            {"kind": "none"}
            if residualization is None
            else residualization
        ),
    }
    variant_root.mkdir(parents=True, exist_ok=True)
    np.save(
        variant_root / wrapper.FROZEN_GENE_ELIGIBILITY_FILE,
        np.ones(1000, dtype=bool),
        allow_pickle=False,
    )
    node_count = 6
    degree = np.full(node_count, 4, dtype=np.int64)
    indptr = np.arange(0, node_count * 4 + 1, 4, dtype=np.int64)
    indices = np.asarray(
        [
            (receiver + offset) % node_count
            for receiver in range(node_count)
            for offset in (1, 2, 3, 4)
        ],
        dtype=np.int64,
    )
    source_permutation = (
        np.arange(node_count, dtype=np.int64) + 1
    ) % node_count
    for slide in ("SO_1", "SO_2"):
        slide_root = variant_root / slide
        slide_root.mkdir()
        shared_slide = robustness_root / "shared" / slide
        shared_slide.mkdir(parents=True, exist_ok=True)
        np.save(
            shared_slide / "base_matched_eligible.npy",
            np.ones(node_count, dtype=bool),
            allow_pickle=False,
        )
        arrays = {
            "fold.npy": np.zeros(node_count, dtype=np.int8),
            "geometry_group.npy": np.zeros(node_count, dtype=np.int16),
            "fov.npy": np.zeros(node_count, dtype=np.int16),
            "qc_passed.npy": np.ones(node_count, dtype=bool),
            "matched_eligible.npy": np.ones(node_count, dtype=bool),
            f"eligible_{variant_id.lower()}.npy": np.ones(
                node_count, dtype=bool
            ),
            "near_degree.npy": degree,
            "annular_degree.npy": degree,
            "permuted_near_degree.npy": degree,
            "near_indptr.npy": indptr,
            "annular_indptr.npy": indptr,
            "near_indices.npy": indices,
            "annular_indices.npy": indices,
            "source_permutation.npy": source_permutation,
            "expression_log1p.npy": np.zeros(
                (node_count, 3), dtype=np.float32
            ),
            "metadata.npy": np.zeros((node_count, 2), dtype=np.float32),
            **{
                filename: np.zeros((node_count, 3), dtype=np.float32)
                for filename in wrapper.DEFAULT_FEATURE_FILES.values()
            },
        }
        for filename, value in arrays.items():
            np.save(slide_root / filename, value, allow_pickle=False)
    content: dict[str, dict[str, Any]] = {}
    for path in sorted(variant_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(variant_root).as_posix()
        content[relative] = {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha(path),
            "storage": "regular_file",
            "shape": list(np.load(path, allow_pickle=False).shape),
            "dtype": str(np.load(path, allow_pickle=False).dtype),
        }
    integrity_payload: dict[str, Any] = {
        "manifest_schema_version": 1,
        "variant_path_id": variant_id.lower(),
        "contract_variant_id": variant_id,
        "preprocessing_version": f"robustness_{variant_id.lower()}_v1",
        "raw_snapshot": {"fingerprint": "1" * 64},
        "split_fingerprint": "3" * 64,
        "base_prepared_manifest_sha256": "4" * 64,
        "variant_spec": variant_spec,
        "content": content,
    }
    processed_fingerprint = canonical_sha256(integrity_payload)
    integrity = {
        **integrity_payload,
        "variant_fingerprint": processed_fingerprint,
    }
    integrity_path = variant_root / "integrity_manifest.json"
    _write_json(integrity_path, integrity)
    _write_json(
        variant_manifest,
        {
            "variant_id": variant_id,
            "preprocessing_version": f"robustness_{variant_id.lower()}_v1",
            "raw_snapshot": {"fingerprint": "1" * 64},
            "processed_fingerprint": processed_fingerprint,
            "split_fingerprint": "3" * 64,
            "variant_spec": variant_spec,
            "base_prepared_manifest_sha256": "4" * 64,
        },
    )
    root_payload: dict[str, Any] = {
        "manifest_schema_version": 1,
        "created_at": "2026-08-10T00:00:00Z",
        "variants": {
            variant_id.lower(): {
                "path": f"variants/{variant_id.lower()}",
                "manifest_sha256": _sha(variant_manifest),
                "integrity_manifest_sha256": _sha(integrity_path),
                "variant_fingerprint": processed_fingerprint,
                "contract_variant_id": variant_id,
            }
        },
    }
    root_fingerprint_payload = dict(root_payload)
    root_fingerprint_payload.pop("created_at")
    root_payload["processed_fingerprint"] = canonical_sha256(
        root_fingerprint_payload
    )
    root_manifest = robustness_root / "manifest.json"
    _write_json(root_manifest, root_payload)

    placeholder = {
        "manifest_sha256": "a" * 64,
        "integrity_manifest_sha256": "b" * 64,
        "processed_fingerprint": "c" * 64,
        "variant_spec_sha256": "d" * 64,
    }
    variant_authorities = {
        value: dict(placeholder) for value in sorted(wrapper.ALLOWED_VARIANTS)
    }
    variant_authorities[variant_id] = {
        "manifest_sha256": _sha(variant_manifest),
        "integrity_manifest_sha256": _sha(integrity_path),
        "processed_fingerprint": processed_fingerprint,
        "variant_spec_sha256": canonical_sha256(variant_spec),
    }
    contract.write_text(
        yaml.safe_dump(
            {
                "campaign_id": wrapper.CAMPAIGN_ID,
                "status": "frozen_preoutcome",
                "launch_authorized": True,
                "frozen_at": "2026-08-10T23:59:59Z",
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
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    sources = [
        _source_row(root, wrapper.BASE_RUNNER_RELATIVE_PATH),
        _source_row(root, wrapper.WRAPPER_RELATIVE_PATH),
        _source_row(root, wrapper.PHASE_TRANSFORM_RELATIVE_PATH),
        _source_row(root, wrapper.RESIDUALIZATION_RELATIVE_PATH),
        _source_row(root, wrapper.ENVIRONMENT_LOCK_RELATIVE_PATH),
        _source_row(root, wrapper.ENVIRONMENT_VERIFIER_RELATIVE_PATH),
    ]
    _write_json(
        launch,
        {
            "campaign_id": wrapper.CAMPAIGN_ID,
            "contract": {
                "path": wrapper.CONTRACT_RELATIVE_PATH,
                "sha": _sha(contract),
            },
            "sources": sources,
        },
    )

    verified_launch = wrapper._verify_launch_manifest(launch, project_root=root)
    verified_variant = wrapper._verify_variant_root(variant_root, project_root=root)
    pilot_artifact = root / f"artifacts/pilot-{variant_id.lower()}"
    pilot_payload = pilot_artifact / "summary.json"
    pilot_success = pilot_artifact / "_SUCCESS"
    pilot_checksums = pilot_artifact / "provenance/artifact_checksums.json"
    pilot_payload.parent.mkdir(parents=True, exist_ok=True)
    pilot_success.write_text("pilot-success\n", encoding="utf-8")
    pilot_payload.write_text('{"pilot":true}\n', encoding="utf-8")
    _write_json(
        pilot_checksums,
        {
            "version": 1,
            "files": {
                "summary.json": {
                    "type": "file",
                    "size": pilot_payload.stat().st_size,
                    "sha256": _sha(pilot_payload),
                }
            },
        },
    )
    receipt_payload = {
        "campaign": wrapper.CAMPAIGN_ID,
        "contract": verified_launch.contract_sha256,
        "source_manifest_sha": verified_launch.source_manifest_sha,
        "variant": variant_id,
        "prepared_manifest_sha": verified_variant.manifest_sha256,
        "selected_attempt": 1,
        "attempt_history": [
            {
                "attempt": 1,
                "job_id": (
                    f"same-gene-robustness-pilot-{variant_id}-s20260810-f0-a1"
                ),
                "plan_sha256": "b" * 64,
                "config_sha256": "5" * 64,
                "selected": True,
                "registry_status": "completed",
                "run_id": f"pilot-{variant_id.lower()}",
                "artifact_path": pilot_artifact.relative_to(root).as_posix(),
                "artifact_status": "success",
            }
        ],
        "bundle": {
            "verified": True,
            "run_id": f"pilot-{variant_id.lower()}",
            "artifact_path": pilot_artifact.relative_to(root).as_posix(),
            "success_sha256": _sha(pilot_success),
            "config_sha256": "5" * 64,
            "bundle_manifest_sha256": _sha(pilot_checksums),
        },
        "controls": {
            "all_outputs_finite": True,
            "train_validation_test_component_overlap": False,
            "receiver_rna_or_derived_covariate_model_input": False,
            "identity_oracle_row_top1_fraction": 1.0,
            "identity_oracle_actually_executed": True,
            "analytical_autograd_max_abs_error": 1e-12,
            "analytical_finite_difference_max_abs_error": 1e-10,
            "graph_specific_invariants": True,
            "checkpoint_gpu_replay_max_abs_metric_error": 1e-9,
            "checkpoint_gpu_replay_max_abs_prediction_error": 1e-9,
            "checkpoint_replay_device_type": "cuda",
            "canonical_production_split_label": "test",
            "source_config_data_hashes_verified": True,
            "outer_test_untouched": True,
            "environment_lock_verified": True,
            "environment_lock_sha256": _sha(environment_lock),
            "environment_verification_sha256": "e" * 64,
            "environment_visibility_mode": "job",
            "peak_vram_gb": 7.5,
            "projected_full_hours_per_fold": 0.1,
            "gate_passed": True,
            "production_authorized": True,
        },
    }
    receipt_payload["attempt_history_sha256"] = canonical_sha256(
        receipt_payload["attempt_history"]
    )
    receipt = root / f"state/{variant_id.lower()}_pilot_receipt.json"
    _write_json(
        receipt,
        {
            "payload": receipt_payload,
            "receipt_sha256": canonical_sha256(receipt_payload),
        },
    )
    marker = root / f"state/markers/{variant_id.lower()}_s20260810_f0_a1.json"
    config = root / f"state/jobs/{variant_id.lower()}_s20260810_f0_a1.json"
    _write_json(
        config,
        {
            "schema_version": 1,
            "variant_root": variant_root.relative_to(root).as_posix(),
            "model_seed": 20260810,
            "fold": 0,
            "profile": profile,
            "attempt": 1,
            "launch_manifest": launch.relative_to(root).as_posix(),
            "pilot_receipt": (
                receipt.relative_to(root).as_posix() if profile == "full" else None
            ),
            "job_success_marker": marker.relative_to(root).as_posix(),
        },
    )
    return {
        "contract": contract,
        "launch": launch,
        "variant_root": variant_root,
        "variant_manifest": variant_manifest,
        "receipt": receipt,
        "pilot_payload": pilot_payload,
        "marker": marker,
        "config": config,
    }


def _materialized(paths: dict[str, Path], root: Path, *extra: str) -> argparse.Namespace:
    cli = wrapper.parse_args(["--config", str(paths["config"]), *extra])
    return wrapper._materialize_cli_arguments(cli, project_root=root)


def test_wrapper_rejects_a_missing_or_noncanonical_contract_freeze_time() -> None:
    for value in (None, "2026-08-10T23:59:59+00:00", "2026-02-30T00:00:00Z"):
        with pytest.raises(wrapper.RobustnessRunError, match="frozen_at"):
            wrapper._require_frozen_at({"frozen_at": value})


def test_materialized_config_supplies_all_cli_values_and_exact_matches(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    arguments = _materialized(
        paths,
        tmp_path,
        "--variant-root",
        str(paths["variant_root"]),
        "--model-seed",
        "20260810",
        "--fold",
        "0",
        "--profile",
        "full",
        "--attempt",
        "1",
        "--launch-manifest",
        str(paths["launch"]),
        "--pilot-receipt",
        str(paths["receipt"]),
    )

    assert arguments.variant_root == paths["variant_root"]
    assert arguments.model_seed == 20260810
    assert arguments.profile == "full"
    assert arguments.job_success_marker == paths["marker"]
    assert arguments.config_sha256 == _sha(paths["config"])


def test_cli_value_must_exactly_match_materialized_config(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    cli = wrapper.parse_args(
        ["--config", str(paths["config"]), "--fold", "1"]
    )
    with pytest.raises(wrapper.RobustnessRunError, match="--fold"):
        wrapper._materialize_cli_arguments(cli, project_root=tmp_path)


def test_parser_rejects_duplicate_config_options(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(SystemExit):
        wrapper.parse_args(
            ["--config", str(paths["config"]), "--config", str(paths["config"])]
        )


def test_full_requires_pilot_receipt_before_base_runner(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, profile="pilot")
    job = json.loads(paths["config"].read_text(encoding="utf-8"))
    job["profile"] = "full"
    _write_json(paths["config"], job)
    arguments = _materialized(paths, tmp_path)
    with pytest.raises(wrapper.RobustnessRunError, match="pilot-receipt"):
        wrapper._verify_inputs(arguments, project_root=tmp_path)


def test_training_fails_closed_on_live_environment_drift_before_data_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, profile="pilot")
    arguments = _materialized(paths, tmp_path)

    def drifted(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise wrapper.EnvironmentLockError("synthetic package drift")

    monkeypatch.setattr(wrapper, "verify_live_environment", drifted)
    with pytest.raises(wrapper.RobustnessRunError, match="live environment"):
        wrapper._verify_inputs(arguments, project_root=tmp_path)


def test_offline_analysis_reconstructs_lock_bound_config_without_job_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    arguments = _materialized(paths, tmp_path)

    def forbidden(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("offline analysis must not probe one-GPU job CUDA")

    monkeypatch.setattr(wrapper, "verify_live_environment", forbidden)
    verified = wrapper._verify_inputs(
        arguments,
        project_root=tmp_path,
        verify_live_job_environment=False,
    )
    assert verified.environment_verification is None
    wrapper._patch_base_runner(
        verified,
        project_root=tmp_path,
        require_live_job_environment=False,
    )
    configuration = wrapper.runner._configuration(
        profile="full", fold=0, attempt=1
    )

    assert wrapper.runner.ENVIRONMENT_LOCK_REQUIRED is False
    assert wrapper.runner.ENVIRONMENT_LOCK_REPORT is None
    assert configuration["campaign"]["environment_lock_sha256"] == _sha(
        tmp_path / wrapper.ENVIRONMENT_LOCK_RELATIVE_PATH
    )
    assert "environment_verification_sha256" not in configuration["campaign"]


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
def test_every_prepared_identity_is_bound_to_frozen_contract(
    tmp_path: Path, field: str
) -> None:
    paths = _fixture(tmp_path, profile="pilot")
    launch = wrapper._verify_launch_manifest(paths["launch"], project_root=tmp_path)
    variant = wrapper._verify_variant_root(
        paths["variant_root"], project_root=tmp_path
    )
    binding = launch.prepared_binding
    if field in binding.variants[variant.variant_id]:
        variants = {
            key: dict(value) for key, value in binding.variants.items()
        }
        variants[variant.variant_id][field] = "f" * 64
        binding = replace(binding, variants=variants)
    else:
        binding = replace(binding, **{field: "f" * 64})

    with pytest.raises(wrapper.RobustnessRunError, match="frozen contract"):
        wrapper._verify_variant_contract_binding(
            replace(launch, prepared_binding=binding), variant
        )


def test_tampered_source_fails_before_base_runner_is_called(
    tmp_path: Path, monkeypatch: Any
) -> None:
    paths = _fixture(tmp_path)
    arguments = _materialized(paths, tmp_path)
    (tmp_path / wrapper.BASE_RUNNER_RELATIVE_PATH).write_text(
        "tampered\n", encoding="utf-8"
    )
    called = False

    def forbidden(_: argparse.Namespace) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("base runner must not be entered")

    monkeypatch.setattr(wrapper.runner, "run", forbidden)
    with pytest.raises(wrapper.RobustnessRunError, match="source identity"):
        wrapper.run(arguments, project_root=tmp_path)
    assert called is False


def test_tampered_pilot_bundle_fails_before_production_data_load(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths["pilot_payload"].write_text("tampered\n", encoding="utf-8")
    arguments = _materialized(paths, tmp_path)

    with pytest.raises(wrapper.RobustnessRunError, match="payload mismatch"):
        wrapper._verify_inputs(arguments, project_root=tmp_path)


def test_graph_invariant_recomputation_rejects_receiver_collision(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, profile="pilot")
    variant_root = paths["variant_root"]
    manifest = json.loads(paths["variant_manifest"].read_text(encoding="utf-8"))
    permutation_path = variant_root / "SO_1/source_permutation.npy"
    permutation = (
        np.arange(6, dtype=np.int64) - 1
    ) % 6
    np.save(permutation_path, permutation, allow_pickle=False)

    with pytest.raises(wrapper.RobustnessRunError, match="collides with its receiver"):
        wrapper._verify_graph_specific_invariants(
            variant_root,
            robustness_root=variant_root.parents[1],
            variant_spec=manifest["variant_spec"],
            eligibility_file=manifest["variant_spec"][
                "primary_eligibility_file"
            ],
        )


@pytest.mark.parametrize(
    ("control", "bad_value", "message"),
    [
        ("all_outputs_finite", False, "finite-output"),
        ("train_validation_test_component_overlap", True, "split-overlap"),
        (
            "receiver_rna_or_derived_covariate_model_input",
            True,
            "receiver-input",
        ),
        ("identity_oracle_row_top1_fraction", 0.99, "identity-oracle"),
        ("identity_oracle_actually_executed", False, "identity-oracle execution"),
        ("analytical_autograd_max_abs_error", 2e-10, "autograd"),
        ("analytical_finite_difference_max_abs_error", 2e-8, "finite-difference"),
        ("graph_specific_invariants", False, "graph-invariant"),
        (
            "checkpoint_gpu_replay_max_abs_metric_error",
            2e-7,
            "metric-replay",
        ),
        (
            "checkpoint_gpu_replay_max_abs_prediction_error",
            2e-7,
            "prediction-replay",
        ),
        ("checkpoint_replay_device_type", "cpu", "not executed on GPU"),
        ("canonical_production_split_label", "validation", "production-split"),
        ("source_config_data_hashes_verified", False, "hash control"),
        ("outer_test_untouched", False, "outer-test"),
        ("environment_lock_verified", False, "environment-lock control"),
        ("environment_visibility_mode", "launcher", "visibility mode"),
        ("environment_lock_sha256", "0" * 64, "environment-lock identity"),
        ("environment_verification_sha256", "bad", "verification SHA"),
        ("peak_vram_gb", 20.6, "VRAM"),
        ("projected_full_hours_per_fold", 0.251, "runtime"),
        ("gate_passed", False, "aggregate gate"),
        ("production_authorized", False, "authorize production"),
    ],
)
def test_full_receipt_rejects_every_failed_control(
    tmp_path: Path, control: str, bad_value: Any, message: str
) -> None:
    paths = _fixture(tmp_path)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["payload"]["controls"][control] = bad_value
    receipt["receipt_sha256"] = canonical_sha256(receipt["payload"])
    _write_json(paths["receipt"], receipt)
    arguments = _materialized(paths, tmp_path)
    with pytest.raises(wrapper.RobustnessRunError, match=message):
        wrapper._verify_inputs(arguments, project_root=tmp_path)


def test_configuration_and_base_constants_bind_variant_seed_and_eligibility(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    arguments = _materialized(paths, tmp_path)
    verified = wrapper._verify_inputs(arguments, project_root=tmp_path)
    wrapper._patch_base_runner(verified, project_root=tmp_path)
    configuration = wrapper.runner._configuration(
        profile="full", fold=0, attempt=1
    )

    assert wrapper.runner.SEED_BASE == 20260810
    assert wrapper.runner.TRACKING_SEED == 260810
    assert wrapper.runner.MAXIMUM_PROJECTED_HOURS_PER_FOLD == 0.25
    assert wrapper.runner.SOURCE_CONFIG_DATA_HASHES_VERIFIED is True
    assert wrapper.runner.GRAPH_SPECIFIC_INVARIANTS_VERIFIED is True
    assert wrapper.runner.ENVIRONMENT_LOCK_REQUIRED is True
    assert wrapper.runner.ENVIRONMENT_LOCK_REPORT is not None
    assert wrapper.runner.ENVIRONMENT_LOCK_REPORT["visibility_mode"] == "job"
    assert wrapper.runner.FEATURE_FILE == wrapper.DEFAULT_FEATURE_FILES
    assert wrapper.runner.ELIGIBILITY_FILE == "eligible_v0.npy"
    assert wrapper.runner.PHASE_INPUT_BUILDER is None
    assert configuration["robustness_variant"]["variant_id"] == "V0"
    assert configuration["robustness_variant"]["permutation_seed"] == 20260810
    assert configuration["normalization"] == {"name": "log1p_raw"}
    assert configuration["cohort"]["node_policy"] == {"name": "all"}
    assert configuration["evaluation"]["primary_eligibility_file"] == "eligible_v0.npy"
    assert configuration["evaluation"]["frozen_gene_eligibility_count"] == 932
    assert (
        wrapper.runner.FROZEN_GENE_ELIGIBILITY_SHA256
        == wrapper.FROZEN_GENE_ELIGIBILITY_SHA256
    )
    assert configuration["campaign"]["pilot_receipt_sha256"] is not None
    assert configuration["campaign"]["environment_lock_sha256"] == _sha(
        tmp_path / wrapper.ENVIRONMENT_LOCK_RELATIVE_PATH
    )
    assert paths["config"].relative_to(tmp_path).as_posix() in set(
        wrapper.runner.PROVENANCE_SOURCE_PATHS
    )


def test_residual_variant_installs_train_only_phase_builder(tmp_path: Path) -> None:
    paths = _fixture(
        tmp_path,
        variant_id="V4",
        profile="pilot",
        residualization={
            "kind": "library",
            "panel_log_total_file": "panel_log_total.npy",
            "neighbor_panel_log_total_files": {
                arm: f"{arm}_panel_log_total.npy"
                for arm in wrapper.DEFAULT_FEATURE_FILES
            },
        },
    )
    arguments = _materialized(paths, tmp_path)
    verified = wrapper._verify_inputs(arguments, project_root=tmp_path)
    wrapper._patch_base_runner(verified, project_root=tmp_path)
    assert isinstance(
        wrapper.runner.PHASE_INPUT_BUILDER,
        wrapper.TrainOnlyPhaseTransformBuilder,
    )


def test_scientific_identity_is_reproducible_and_variant_sensitive(tmp_path: Path) -> None:
    first_paths = _fixture(tmp_path, variant_id="V0")
    first_args = _materialized(first_paths, tmp_path)
    first_verified = wrapper._verify_inputs(first_args, project_root=tmp_path)
    wrapper._patch_base_runner(first_verified, project_root=tmp_path)
    first = wrapper.runner._configuration(profile="full", fold=0, attempt=1)
    replay = wrapper.runner._configuration(profile="full", fold=0, attempt=1)
    assert scientific_id(first) == scientific_id(replay)

    second_paths = _fixture(tmp_path, variant_id="V1")
    second_args = _materialized(second_paths, tmp_path)
    second_verified = wrapper._verify_inputs(second_args, project_root=tmp_path)
    wrapper._patch_base_runner(second_verified, project_root=tmp_path)
    changed = wrapper.runner._configuration(profile="full", fold=0, attempt=1)
    assert scientific_id(first) != scientific_id(changed)


def test_atomic_marker_is_exclusive_and_verify_mode_rechecks_binding(
    tmp_path: Path, monkeypatch: Any
) -> None:
    paths = _fixture(tmp_path)
    arguments = _materialized(paths, tmp_path)
    verified = wrapper._verify_inputs(arguments, project_root=tmp_path)
    artifact = tmp_path / "artifacts/run-a"
    artifact.mkdir(parents=True)
    (artifact / "_SUCCESS").write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(
        wrapper,
        "_verify_completed_run",
        lambda **_: (artifact, _sha(artifact / "_SUCCESS")),
    )
    marker = wrapper._publish_job_marker(
        verified=verified,
        run_id="run-a",
        artifact=artifact,
        success_sha256=_sha(artifact / "_SUCCESS"),
        project_root=tmp_path,
    )
    assert paths["marker"].is_file()
    assert marker["payload"]["config_sha256"] == _sha(paths["config"])
    assert wrapper.verify_job_marker(arguments, project_root=tmp_path)["verified"]
    with pytest.raises(wrapper.RobustnessRunError, match="already exists"):
        wrapper._publish_job_marker(
            verified=verified,
            run_id="run-a",
            artifact=artifact,
            success_sha256=_sha(artifact / "_SUCCESS"),
            project_root=tmp_path,
        )


def test_job_marker_rejects_config_tampering(tmp_path: Path, monkeypatch: Any) -> None:
    paths = _fixture(tmp_path)
    arguments = _materialized(paths, tmp_path)
    verified = wrapper._verify_inputs(arguments, project_root=tmp_path)
    artifact = tmp_path / "artifacts/run-b"
    artifact.mkdir(parents=True)
    (artifact / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(
        wrapper,
        "_verify_completed_run",
        lambda **_: (artifact, _sha(artifact / "_SUCCESS")),
    )
    wrapper._publish_job_marker(
        verified=verified,
        run_id="run-b",
        artifact=artifact,
        success_sha256=_sha(artifact / "_SUCCESS"),
        project_root=tmp_path,
    )
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["attempt"] = 2
    _write_json(paths["config"], config)
    changed_arguments = _materialized(paths, tmp_path)
    with pytest.raises(wrapper.RobustnessRunError, match="config SHA-256"):
        wrapper.verify_job_marker(changed_arguments, project_root=tmp_path)


def test_run_publishes_known_marker_only_after_completed_run_verification(
    tmp_path: Path, monkeypatch: Any
) -> None:
    paths = _fixture(tmp_path)
    arguments = _materialized(paths, tmp_path)
    artifact = tmp_path / "artifacts/run-c"
    artifact.mkdir(parents=True)
    (artifact / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(
        wrapper.runner,
        "run",
        lambda _: {"run_id": "run-c", "artifact_path": str(artifact)},
    )
    monkeypatch.setattr(
        wrapper,
        "_verify_completed_run",
        lambda **_: (artifact, _sha(artifact / "_SUCCESS")),
    )

    result = wrapper.run(arguments, project_root=tmp_path)

    assert result["job_success_marker"] == str(paths["marker"])
    marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
    assert marker["payload"]["run_id"] == "run-c"
    assert marker["payload"]["artifact_path"] == "artifacts/run-c"


def test_failed_post_run_verification_never_creates_job_marker(
    tmp_path: Path, monkeypatch: Any
) -> None:
    paths = _fixture(tmp_path)
    arguments = _materialized(paths, tmp_path)
    monkeypatch.setattr(
        wrapper.runner,
        "run",
        lambda _: {"run_id": "run-d", "artifact_path": str(tmp_path / "bad")},
    )

    def fail_verification(**_: Any) -> tuple[Path, str]:
        raise wrapper.RobustnessRunError("registry status is not completed")

    monkeypatch.setattr(wrapper, "_verify_completed_run", fail_verification)
    with pytest.raises(wrapper.RobustnessRunError, match="registry status"):
        wrapper.run(arguments, project_root=tmp_path)
    assert not paths["marker"].exists()
