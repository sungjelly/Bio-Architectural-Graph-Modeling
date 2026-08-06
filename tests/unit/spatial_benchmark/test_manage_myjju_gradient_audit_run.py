from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry
from spatial_benchmark.run_archive import verify_run_bundle


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "analysis"
    / "manage_myjju_gradient_audit_run.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "manage_myjju_gradient_audit_run_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

GradientAuditRunError = _MODULE.GradientAuditRunError


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=root,
        config_root=root / "configs",
        data_root=root / "data",
        artifact_root=root / "artifacts",
        state_root=root / "state",
        scratch_root=root / "scratch",
        cache_root=root / "cache",
        export_root=root / "exports",
        report_root=root / "reports",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _descriptor(path: Path, *, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture
def audit_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Registry, ProjectPaths]:
    project = tmp_path / "project"
    project.mkdir()
    paths = _paths(project)
    campaign_dir = (
        project
        / "experiments"
        / "campaigns"
        / _MODULE.CAMPAIGN_ID
    )
    campaign_dir.mkdir(parents=True)
    for name in (
        "campaign.yaml",
        "frozen_task_contract.yaml",
        "implementation_protocol.yaml",
    ):
        source = (
            _ROOT
            / "experiments"
            / "campaigns"
            / _MODULE.CAMPAIGN_ID
            / name
        )
        (campaign_dir / name).write_bytes(source.read_bytes())
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    campaign_config = json.loads(
        json.dumps(
            __import__("yaml").safe_load(
                (campaign_dir / "campaign.yaml").read_text(encoding="utf-8")
            )
        )
    )
    registry.create_campaign(
        _MODULE.CAMPAIGN_ID,
        name="MyJJu GeneMAE gradient claim audit",
        scientific_question="synthetic registry bundle test",
        config=campaign_config,
        status="planned",
    )
    monkeypatch.setattr(
        _MODULE,
        "_capture_provenance",
        lambda _configuration, _paths: {
            "git_commit": "a" * 40,
            "dirty_fingerprint": "b" * 64,
            "dataset_fingerprint": _MODULE.COHORT_FINGERPRINT_SHA256,
            "split_fingerprint": _MODULE.SPLIT_FINGERPRINT,
            "preprocessing_version": "full_cell_log1p_cp10k_v1",
            "environment_fingerprint": "c" * 64,
            "environment_text": "synthetic-environment\n",
            "uncommitted_changes_patch": "",
            "untracked_files": [],
            "gpu_model": None,
            "hardware": {"cpu": "synthetic"},
        },
    )
    return registry, paths


def _prepare(
    fixture: tuple[Registry, ProjectPaths],
) -> dict[str, Any]:
    registry, paths = fixture
    return _MODULE.prepare_run(
        registry=registry,
        paths=paths,
        invocation=["python", str(_SCRIPT), "prepare"],
    )


def _write_sidecar(path: Path, descriptor: dict[str, Any]) -> None:
    _write_json(
        path.with_name(f"{path.name}.sha256.json"),
        {
            "path": path.name,
            "sha256": descriptor["sha256"],
            "size_bytes": descriptor["size_bytes"],
        },
    )


def _materialize_valid_outputs(
    prepared: dict[str, Any],
) -> Path:
    active = Path(prepared["active_run_root"])
    work = Path(prepared["work_root"])
    scientific_input_sha256 = "d" * 64
    pilot_path = work / "pilot/resource_pilot.json"
    _write_json(
        pilot_path,
        {
            "kind": "synthetic_resource_pilot",
            "passed": True,
            "peak_allocated_vram_gib": 1.0,
            "analysis_input_sha256": scientific_input_sha256,
        },
    )
    pilot = _descriptor(pilot_path, relative="pilot/resource_pilot.json")
    shards: list[dict[str, Any]] = []
    for seed in _MODULE.EXPECTED_SEEDS:
        directory = work / f"shards/seed-{seed:02d}"
        metadata_path = directory / f"seed-{seed:02d}.metadata.json"
        _write_json(
            metadata_path,
            {
                "kind": "synthetic_gradient_seed_shard",
                "seed": seed,
                "analysis_input_sha256": scientific_input_sha256,
                "cores": list(_MODULE.EXPECTED_CORES),
                "mask_replicates": list(_MODULE.EXPECTED_MASK_REPLICATES),
                "row_identifiers_stored": False,
            },
        )
        metadata = _descriptor(
            metadata_path,
            relative=(
                f"shards/seed-{seed:02d}/"
                f"seed-{seed:02d}.metadata.json"
            ),
        )
        _write_sidecar(metadata_path, metadata)
        arrays_path = directory / f"seed-{seed:02d}.arrays.npz"
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            arrays_path,
            selected_truth=np.asarray([0.0, 1.0], dtype=np.float32),
            selected_mask=np.asarray([True, False]),
            selected_prediction=np.asarray(
                [seed / 10.0, 1.0 + seed / 10.0],
                dtype=np.float32,
            ),
            core_offsets=np.asarray([0, 2], dtype=np.int64),
        )
        arrays = {
            **_descriptor(
                arrays_path,
                relative=(
                    f"shards/seed-{seed:02d}/"
                    f"seed-{seed:02d}.arrays.npz"
                ),
            ),
            "retention": "transient_deleted_after_verified_aggregation",
        }
        _write_sidecar(arrays_path, arrays)
        shards.append(
            {
                "seed": seed,
                "metadata": metadata,
                "arrays": arrays,
            }
        )

    coverage = {
        "seeds": list(_MODULE.EXPECTED_SEEDS),
        "cores": list(_MODULE.EXPECTED_CORES),
        "mask_replicates": list(_MODULE.EXPECTED_MASK_REPLICATES),
    }
    transient = _MODULE._transient_core(
        coverage=coverage,
        pilot=pilot,
        shards=shards,
    )
    aggregate_input_sha256 = canonical_sha256(transient)
    aggregate = work / "aggregate"
    report_path = aggregate / "report.json"
    gate_rows: list[dict[str, Any]] = []
    for gate in (
        "target_predictive_eligibility",
        "target_graph_use_eligibility",
        "mask_rank_stability",
    ):
        gate_rows.extend(
            {
                "gate": gate,
                "scope": target,
                "threshold": {"minimum": 0.0},
                "observed": {"value": 1.0},
                "pass": not (
                    gate == "target_graph_use_eligibility" and target == "KRT8"
                ),
            }
            for target in sorted(_MODULE.EXPECTED_TARGET_SCOPES)
        )
    gate_rows.append(
        {
            "gate": "seed_rank_stability",
            "scope": "all_cores",
            "threshold": {"minimum": 0.70},
            "observed": {"value": 0.80},
            "pass": True,
        }
    )
    gate_rows.extend(
        {
            "gate": "signed_pair_stability",
            "scope": f"{target}<-{source}",
            "target": target,
            "source": source,
            "threshold": {"minimum_seed_prevalence": 6},
            "observed": {"seed_prevalence": 7},
            "pass": True,
        }
        for target, source in sorted(_MODULE.EXPECTED_LOCKED_PAIRS)
    )
    gate_rows.extend(
        {
            "gate": "bounded_faithfulness",
            "scope": scope,
            "threshold": {"minimum_spearman": 0.70},
            "observed": {"spearman": 0.80},
            "pass": True,
        }
        for scope in sorted(
            _MODULE.EXPECTED_GATE_SCOPES["bounded_faithfulness"]
        )
    )
    for gate in (
        "graph_gradient_structure_null",
        "parameter_randomization",
        "matched_pair_null",
    ):
        gate_rows.append(
            {
                "gate": gate,
                "scope": "aggregate",
                "threshold": {"minimum": 0.0},
                "observed": {"value": 1.0},
                "pass": True,
            }
        )
    assert len(gate_rows) == 40
    gate_fraction = sum(int(row["pass"]) for row in gate_rows) / len(gate_rows)
    _write_json(
        report_path,
        {
            "kind": "synthetic_gradient_audit_report",
            "analysis_status": "complete",
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "frozen_contract_sha256": _MODULE.FROZEN_CONTRACT_SHA256,
            "implementation_protocol_sha256": (
                _MODULE.IMPLEMENTATION_PROTOCOL_SHA256
            ),
            "analysis_input_sha256": aggregate_input_sha256,
            "scientific_input_sha256": scientific_input_sha256,
            "coverage": coverage,
            "gate_rows": gate_rows,
            "final_metrics": {
                _MODULE.PRIMARY_METRIC: gate_fraction,
                "audit/eligible_target_fraction": 0.2,
            },
            "claim_verdict": "biological_mechanism_not_supported_by_this_design",
            "candidate_set_computational_precursors_supported": False,
            "mechanism_validation_available": False,
            "mechanism_claim_supported": False,
            "maximum_defensible_claim": (
                "no_claim_beyond_reported_model_behavior"
            ),
        },
    )
    gate_path = aggregate / "gate_results.csv"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    with gate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = __import__("csv").DictWriter(
            handle,
            fieldnames=("gate", "scope", "threshold", "observed", "pass"),
        )
        writer.writeheader()
        for row in gate_rows:
            writer.writerow(
                {
                    "gate": row["gate"],
                    "scope": row["scope"],
                    "threshold": json.dumps(
                        row["threshold"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "observed": json.dumps(
                        row["observed"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "pass": str(row["pass"]).lower(),
                }
            )
    markdown_path = aggregate / "report.md"
    markdown_path.write_text(
        "# Synthetic gradient audit\n\nMechanism validation is unavailable.\n",
        encoding="utf-8",
    )
    html_path = aggregate / "report.html"
    html_path.write_text(
        "<!doctype html><html><body><main>Mechanism validation is "
        "unavailable.</main></body></html>\n",
        encoding="utf-8",
    )
    prediction_path = aggregate / "canonical_fit_predictions.jsonl"
    rows = [
        {
            "run_id": "pending-registration",
            "sample_key": f"{core}/{target}/mask-{replicate}",
            "dataset_id": _MODULE.DATASET_ID,
            "split": "fit",
            "y_true": 0.25 + target_index,
            "y_pred": 0.30 + target_index,
            "graph_id": "myjju-k15-tiled-union",
            "effective_mask_rate": 0.2,
        }
        for core in _MODULE.EXPECTED_CORES
        for replicate in _MODULE.EXPECTED_MASK_REPLICATES
        for target_index, target in enumerate(_MODULE.EXPECTED_TARGETS)
    ]
    prediction_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    path_by_role = {
        "audit_report_json": report_path,
        "gate_results_csv": gate_path,
        "audit_report_markdown": markdown_path,
        "audit_report_html": html_path,
        "canonical_fit_predictions_jsonl": prediction_path,
    }
    files = [
        {
            "role": role,
            **_descriptor(path_by_role[role], relative=filename),
        }
        for role, filename in _MODULE.REQUIRED_REPORT_FILES.items()
    ]
    manifest_core = {
        "kind": _MODULE.REPORT_MANIFEST_KIND,
        "schema_version": _MODULE.REPORT_MANIFEST_SCHEMA_VERSION,
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "frozen_contract_sha256": _MODULE.FROZEN_CONTRACT_SHA256,
        "implementation_protocol_sha256": (
            _MODULE.IMPLEMENTATION_PROTOCOL_SHA256
        ),
        "analysis_input_sha256": aggregate_input_sha256,
        "coverage": coverage,
        "pilot": pilot,
        "shards": shards,
        "files": files,
        "execution_commands": [
            ["python", "audit_myjju_genemae_gradients.py", "pilot"],
            *[
                [
                    "python",
                    "audit_myjju_genemae_gradients.py",
                    "seed-shard",
                    "--seed",
                    str(seed),
                ]
                for seed in _MODULE.EXPECTED_SEEDS
            ],
            ["python", "audit_myjju_genemae_gradients.py", "aggregate"],
        ],
    }
    manifest = {
        **manifest_core,
        "checksum": canonical_sha256(manifest_core),
    }
    manifest_path = aggregate / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def test_prepare_registers_one_aggregate_with_unknown_seed_and_fold(
    audit_fixture: tuple[Registry, ProjectPaths],
) -> None:
    registry, paths = audit_fixture
    first = _prepare(audit_fixture)
    second = _prepare(audit_fixture)

    assert first["created"] is True
    assert second["created"] is False
    assert second["run_id"] == first["run_id"]
    active = Path(first["active_run_root"])
    assert active == paths.scratch_root / "active_runs" / first["run_id"]
    assert active.is_dir()
    assert Path(first["work_root"]) == active / _MODULE.WORK_RELATIVE
    run = registry.get_run(first["run_id"])
    assert run is not None and run["status"] == "running"
    assert run["seed"] == 0 and run["fold"] == 0 and run["attempt"] == 1
    assert run["artifact_path"] == first["artifact_path"]
    category = registry.get_run_category(first["run_id"])
    assert category is not None
    assert category["lifecycle_stage"] == "exploratory_screen"
    assert category["study_axis"] == "gradient_attribution_audit"
    assert category["seed_known"] is False
    assert category["fold_known"] is False
    assert category["attempt_known"] is True
    aliases = registry.get_run_aliases(first["run_id"])
    assert len(aliases) == 1
    assert aliases[0]["preferred"] is True
    assert ".sna.fna.a01." in aliases[0]["alias_id"]
    receipt = json.loads(
        (active / _MODULE.PREPARE_RECEIPT_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    checksum = receipt.pop("checksum")
    assert checksum == canonical_sha256(receipt)
    with registry.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS count FROM runs WHERE campaign_id = ?",
            (_MODULE.CAMPAIGN_ID,),
        ).fetchone()["count"]
    assert count == 1


def test_finalize_rejects_report_tamper_before_publish_or_registry_commit(
    audit_fixture: tuple[Registry, ProjectPaths],
) -> None:
    registry, paths = audit_fixture
    prepared = _prepare(audit_fixture)
    _materialize_valid_outputs(prepared)
    report = Path(prepared["work_root"]) / "aggregate/report.md"
    report.write_text(report.read_text(encoding="utf-8") + "tamper\n")

    with pytest.raises(GradientAuditRunError, match="size or checksum mismatch"):
        _MODULE.finalize_run(
            registry=registry,
            paths=paths,
            run_reference=prepared["run_id"],
        )

    run = registry.get_run(prepared["run_id"])
    assert run is not None and run["status"] == "running"
    assert Path(prepared["active_run_root"]).is_dir()
    assert not Path(prepared["artifact_path"]).exists()
    with registry.connect() as connection:
        artifact_count = connection.execute(
            "SELECT COUNT(*) AS count FROM artifacts WHERE run_id = ?",
            (prepared["run_id"],),
        ).fetchone()["count"]
    assert artifact_count == 0


def test_finalize_publishes_verified_bundle_without_transient_vectors(
    audit_fixture: tuple[Registry, ProjectPaths],
) -> None:
    registry, paths = audit_fixture
    prepared = _prepare(audit_fixture)
    _materialize_valid_outputs(prepared)

    result = _MODULE.finalize_run(
        registry=registry,
        paths=paths,
        run_reference=prepared["run_id"],
    )

    assert result["completed"] is True
    assert result["idempotent"] is False
    artifact = Path(result["artifact_path"])
    assert artifact.is_dir()
    assert not Path(prepared["active_run_root"]).exists()
    assert not list(artifact.rglob("*.npz"))
    assert (artifact / "_SUCCESS").is_file()
    assert (artifact / "checkpoints/last.ckpt").is_file()
    canonical_rows = [
        json.loads(line)
        for line in (artifact / "predictions/fit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert {row["run_id"] for row in canonical_rows} == {prepared["run_id"]}
    state = json.loads(
        (artifact / "checkpoints/last.ckpt").read_text(encoding="utf-8")
    )
    assert state["model_weights"] is False
    assert state["mechanism_validation_available"] is False
    cleanup = json.loads(
        (artifact / _MODULE.TRANSIENT_CLEANUP_RECEIPT_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    assert len(cleanup["deleted_files"]) == 14
    assert verify_run_bundle(artifact)["valid"] is True
    run = registry.get_run(prepared["run_id"])
    assert run is not None and run["status"] == "completed"
    assert run["primary_metric_name"] == _MODULE.PRIMARY_METRIC
    assert run["primary_metric_value"] == pytest.approx(39 / 40)
    assert registry.verify_artifacts(run_id=prepared["run_id"]) == []
    checkpoints = registry.list_checkpoint_catalog(
        run_id=prepared["run_id"],
        role="last",
        verification_status="verified",
    )
    assert len(checkpoints) == 1
    assert checkpoints[0]["monitored_metric"] == _MODULE.PRIMARY_METRIC
    assert checkpoints[0]["metadata"]["model_weights"] is False
    assert (
        checkpoints[0]["metadata"]["artifact_semantics"]
        == "analysis_state_only_not_model_weights"
    )

    repeated = _MODULE.finalize_run(
        registry=registry,
        paths=paths,
        run_reference=prepared["run_id"],
    )
    assert repeated["completed"] is True
    assert repeated["idempotent"] is True
    assert repeated["artifact_path"] == artifact.as_posix()


def test_finalize_rejects_mechanism_available_reframing(
    audit_fixture: tuple[Registry, ProjectPaths],
) -> None:
    registry, paths = audit_fixture
    prepared = _prepare(audit_fixture)
    manifest_path = _materialize_valid_outputs(prepared)
    report_path = manifest_path.parent / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["mechanism_validation_available"] = True
    _write_json(report_path, report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["role"] == "audit_report_json":
            entry.update(_descriptor(report_path, relative="report.json"))
    manifest.pop("checksum")
    manifest["checksum"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        GradientAuditRunError,
        match="mechanism_validation_available=false",
    ):
        _MODULE.finalize_run(
            registry=registry,
            paths=paths,
            run_reference=prepared["run_id"],
        )

    run = registry.get_run(prepared["run_id"])
    assert run is not None and run["status"] == "running"


def test_finalize_rejects_pending_gate_reduction(
    audit_fixture: tuple[Registry, ProjectPaths],
) -> None:
    registry, paths = audit_fixture
    prepared = _prepare(audit_fixture)
    manifest_path = _materialize_valid_outputs(prepared)
    report_path = manifest_path.parent / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["analysis_status"] = "aggregate_inputs_verified_gate_reduction_pending"
    report["claim_verdict"] = "inconclusive_gate_reduction_pending"
    report["final_metrics"][_MODULE.PRIMARY_METRIC] = 0.0
    _write_json(report_path, report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["role"] == "audit_report_json":
            entry.update(_descriptor(report_path, relative="report.json"))
    manifest.pop("checksum")
    manifest["checksum"] = canonical_sha256(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(
        GradientAuditRunError,
        match="pending gate reduction cannot be finalized",
    ):
        _MODULE.finalize_run(
            registry=registry,
            paths=paths,
            run_reference=prepared["run_id"],
        )

    run = registry.get_run(prepared["run_id"])
    assert run is not None and run["status"] == "running"
