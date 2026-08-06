from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping

import pytest

from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "finalize_myjju_genemae_campaign_registry.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "finalize_myjju_genemae_campaign_registry_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FinalizationError = _MODULE.GeneMAECampaignFinalizationError


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(*, role: str, seed: int) -> dict[str, Any]:
    production = role == "production"
    return {
        "campaign": {"campaign_id": _MODULE.CAMPAIGN_ID},
        "classification": {
            "lifecycle_stage": (
                "exploratory_screen" if production else "diagnostic"
            ),
        },
        "metadata": {"execution_role": role},
        "model": {
            "name": _MODULE.GENEMAE_MODEL_KEY,
            "family": "myjju_dual_path_genemae",
            "expected_trainable_parameters": _MODULE.EXPECTED_PARAMETER_COUNT,
        },
        "seed": seed,
        "trainer": {
            "diagnostic_resource_pilot": not production,
            "fixed_epoch_budget": production,
            "max_epochs": (
                _MODULE.EXPECTED_COMPLETED_EPOCHS if production else 2
            ),
            "checkpoint_policy": "last_only",
            "primary_checkpoint_role": "last",
        },
        "experiment": {
            "resource_pilot": not production,
            "conclusion_eligible": production,
            "variant_label": (
                "myjju_genemae_ensemble_member"
                if production
                else "myjju_genemae_resource_pilot"
            ),
        },
    }


def _set_job_terminal(
    registry: Registry,
    *,
    job_id: str,
    run_id: str,
    status: str,
) -> None:
    with registry.transaction(immediate=True) as connection:
        connection.execute(
            """
            UPDATE queue_jobs
            SET status = ?, run_id = ?, claimed_at = created_at,
                started_at = created_at, finished_at = created_at
            WHERE job_id = ?
            """,
            (status, run_id, job_id),
        )


def _add_attempt(
    registry: Registry,
    paths: ProjectPaths,
    *,
    role: str,
    seed: int,
    status: str,
    run_id: str,
    job_id: str,
    attempt: int = 1,
    retry_run_id: str | None = None,
    retry_job_id: str | None = None,
) -> dict[str, Any]:
    config = _config(role=role, seed=seed)
    scientific_id = f"s_{role}"
    registry.register_variant(
        scientific_id,
        campaign_id=_MODULE.CAMPAIGN_ID,
        configuration=config,
    )
    registry.enqueue(
        campaign_id=_MODULE.CAMPAIGN_ID,
        configuration=config,
        command=["python", "train.py"],
        job_id=job_id,
        maximum_attempts=2,
        attempt_count=attempt,
        retry_of=retry_job_id,
    )
    artifact_root = paths.artifact_root / "runs" / run_id
    artifact_root.mkdir(parents=True, exist_ok=True)
    run = registry.create_run(
        run_id,
        campaign_id=_MODULE.CAMPAIGN_ID,
        scientific_id=scientific_id,
        repro_id=f"repro_{role}_{seed}_{attempt}",
        seed=seed,
        fold=0,
        attempt=attempt,
        configuration=config,
        status=status,
        artifact_path=artifact_root,
        retry_of=retry_run_id,
        parameter_count=(
            _MODULE.EXPECTED_PARAMETER_COUNT
            if role == "production" and status == "completed"
            else None
        ),
        primary_metric_name=_MODULE.PRIMARY_METRIC,
        primary_metric_value=(
            0.2 + seed / 100.0
            if role == "production" and status == "completed"
            else None
        ),
        failure_category=None if status == "completed" else "nonzero_exit",
    )
    queue_status = "stale" if status == "failed" and role == "stale" else status
    _set_job_terminal(
        registry,
        job_id=job_id,
        run_id=run_id,
        status=queue_status,
    )
    result = {
        "run_id": run_id,
        "job_id": job_id,
        "seed": seed,
        "attempt": attempt,
        "status": status,
    }
    if role == "production" and status == "completed":
        checkpoint = artifact_root / "checkpoints" / "last.ckpt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-seed-{seed}".encode("utf-8"))
        checkpoint_sha = _sha256(checkpoint)
        config_path = artifact_root / "config.resolved.yaml"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifact_root / "_SUCCESS").write_text(
            json.dumps({"run_id": run_id}) + "\n",
            encoding="utf-8",
        )
        artifact_id = registry.record_artifact(
            run_id,
            kind="checkpoints",
            path=checkpoint,
            sha256=checkpoint_sha,
            size_bytes=checkpoint.stat().st_size,
            status="present",
        )
        registry.register_checkpoint_metadata(
            artifact_id,
            run_id=run_id,
            role="last",
            best_epoch=_MODULE.EXPECTED_FINAL_EPOCH,
            monitored_metric=_MODULE.PRIMARY_METRIC,
            monitored_mode="min",
            monitored_value=0.2 + seed / 100.0,
            retention_class="retain_exploratory_evidence",
            verification_status="verified",
        )
        result.update(
            {
                "checkpoint_sha256": checkpoint_sha,
                "state_dict_sha256": hashlib.sha256(
                    f"state-{seed}".encode("utf-8")
                ).hexdigest(),
                "config_sha256": hashlib.sha256(
                    config_path.read_bytes()
                ).hexdigest(),
            }
        )
    return result


def _registry_fixture(
    paths: ProjectPaths, *, campaign_status: str = "planned"
) -> tuple[Registry, dict[int, dict[str, Any]]]:
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    registry.create_campaign(
        _MODULE.CAMPAIGN_ID,
        name="MyJJu finalizer test",
        scientific_question="test",
        config={"frozen": {"contract": _MODULE.FROZEN_CONTRACT_SHA256}},
        status=campaign_status,
    )
    production = {
        seed: _add_attempt(
            registry,
            paths,
            role="production",
            seed=seed,
            status="completed",
            run_id=f"r_production_seed_{seed}",
            job_id=f"q_production_seed_{seed}",
        )
        for seed in _MODULE.PRODUCTION_SEEDS
    }
    _add_attempt(
        registry,
        paths,
        role="resource_pilot",
        seed=0,
        status="failed",
        run_id="r_pilot_failed",
        job_id="q_pilot_failed",
    )
    _add_attempt(
        registry,
        paths,
        role="resource_pilot",
        seed=0,
        status="completed",
        run_id="r_pilot_completed",
        job_id="q_pilot_completed",
    )
    return registry, production


def _report(
    paths: ProjectPaths,
    production: Mapping[int, Mapping[str, Any]],
    *,
    primary_passed: bool = True,
    outcome: str | None = None,
    run_id_override: tuple[int, str] | None = None,
    omit_pilot_failure: bool = False,
    report_directory_name: str = "comparison",
) -> tuple[Path, str]:
    directory = (
        _MODULE._comparison_report_root(paths) / report_directory_name
    )
    gate_status = {
        "primary_comparison": primary_passed,
        "baseline": False,
        "graph_use": True,
    }
    failed = [
        name for name in _MODULE.GATE_NAMES if not gate_status[name]
    ]
    comparison = {
        "schema_version": 1,
        "artifact_kind": _MODULE.COMPARISON_KIND,
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "status": "complete",
        "outcome": (
            outcome
            if outcome is not None
            else ("supported" if primary_passed else "negative")
        ),
        "exploratory": True,
        "estimand": "held_in_partial_gene_reconstruction",
        "biological_unit": "tissue_core",
        "core_count": 10,
        "model_seeds": list(_MODULE.PRODUCTION_SEEDS),
        "seeds_are_biological_replicates": False,
        "protected_identifiers_emitted": False,
        "frozen_gates": [
            {"gate": name, "passed": gate_status[name]}
            for name in _MODULE.GATE_NAMES
        ],
        "failed_gates": failed,
        "coverage": {
            "expected_batches": 180,
            "observed_batches": 180,
            "genemae_members": 7,
            "bagm_gat_members": 7,
            "bagm_self_members": 7,
            "registered_failures_before_completion": (
                0 if omit_pilot_failure else 1
            ),
        },
    }
    run_audit: list[dict[str, Any]] = []
    ordered_run_ids: list[str] = []
    checkpoint_sha: dict[str, str] = {}
    for seed in _MODULE.PRODUCTION_SEEDS:
        item = production[seed]
        run_id = str(item["run_id"])
        if run_id_override is not None and run_id_override[0] == seed:
            run_id = run_id_override[1]
        ordered_run_ids.append(run_id)
        checkpoint_sha[f"{_MODULE.GENEMAE_MODEL_KEY}:seed-{seed}"] = str(
            item["checkpoint_sha256"]
        )
        run_audit.append(
            {
                "model_key": _MODULE.GENEMAE_MODEL_KEY,
                "seed": seed,
                "run_id": run_id,
                "attempt": 1,
                "bundle_verified": True,
                "registry_artifacts_verified": True,
                "checkpoint_catalog_verified": True,
                "checkpoint_role": "last",
                "checkpoint_file_sha256": item["checkpoint_sha256"],
                "state_dict_sha256": item["state_dict_sha256"],
                "config_sha256": item["config_sha256"],
                "parameter_count": _MODULE.EXPECTED_PARAMETER_COUNT,
                "completed_epochs": _MODULE.EXPECTED_COMPLETED_EPOCHS,
                "final_epoch": _MODULE.EXPECTED_FINAL_EPOCH,
            }
        )
    provenance = {
        "schema_version": 1,
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "frozen_contract_sha256": _MODULE.FROZEN_CONTRACT_SHA256,
        "read_only_registry_audit": True,
        "protected_identifiers_emitted": False,
        "run_ids": {_MODULE.GENEMAE_MODEL_KEY: ordered_run_ids},
        "checkpoint_sha256": checkpoint_sha,
    }
    _write_json(directory / "comparison.json", comparison)
    _write_json(directory / "provenance.json", provenance)
    _write_json(directory / "run_audit.json", run_audit)
    pilot_rows = [
        {
            "stage": "pilot",
            "model_key": _MODULE.GENEMAE_MODEL_KEY,
            "seed": 0,
            "run_id": run_id,
            "attempt": 1,
            "status": status,
            "retry_of": None,
            "failure_category": (
                None if status == "completed" else "nonzero_exit"
            ),
            "selected_completed_attempt": False,
        }
        for run_id, status in (
            ("r_pilot_failed", "failed"),
            ("r_pilot_completed", "completed"),
        )
    ]
    _write_json(directory / "attempt_inventory.json", run_audit + pilot_rows)
    _write_json(
        directory / "registered_failures.json",
        [] if omit_pilot_failure else [pilot_rows[0]],
    )
    _write_json(directory / "pilot_inventory.json", pilot_rows)
    (directory / "report.md").write_text("# Test report\n", encoding="utf-8")
    (directory / "report.html").write_text(
        "<!doctype html><html><body>Test</body></html>\n",
        encoding="utf-8",
    )
    files = {
        path.relative_to(directory).as_posix(): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": _MODULE.COMPARISON_KIND,
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "portable_single_file_html": True,
        "protected_identifiers_emitted": False,
        "files": files,
        "comparison_sha256": files["comparison.json"]["sha256"],
        "provenance_sha256": files["provenance.json"]["sha256"],
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, _sha256(manifest_path)


def _campaign_row(database: Path) -> dict[str, Any]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT status, config_json, created_at, updated_at
            FROM campaigns WHERE campaign_id = ?
            """,
            (_MODULE.CAMPAIGN_ID,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _campaign_table_checksums(database: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        for table in ("queue_jobs", "runs", "artifacts", "checkpoint_catalog"):
            rows = [
                dict(row)
                for row in connection.execute(
                    f"SELECT * FROM {table} ORDER BY 1"
                ).fetchall()
            ]
            result[table] = canonical_sha256(rows)
    return result


def _finalize(
    paths: ProjectPaths,
    manifest_path: Path,
    manifest_sha256: str,
    *,
    suffix: str = "one",
) -> tuple[dict[str, Any], Path, Path]:
    receipt = (
        _MODULE._reconciliation_directory(paths) / f"receipt_{suffix}.json"
    )
    backup = (
        paths.state_root
        / "tracking"
        / "backups"
        / f"backup_{suffix}.sqlite3"
    )
    result = _MODULE.finalize_campaign(
        paths=paths,
        database_path=paths.state_root / "tracking" / "bagm.sqlite3",
        comparison_manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha256,
        receipt_output_path=receipt,
        backup_path=backup,
        finalized_at=f"2026-07-30T17:00:0{0 if suffix == 'one' else 1}.000000Z",
    )
    return result, receipt, backup


def test_finalization_backs_up_and_only_completes_campaign(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(paths, production)
    before_campaign = _campaign_row(registry.path)
    before_tables = _campaign_table_checksums(registry.path)

    result, receipt_path, backup_path = _finalize(
        paths, manifest_path, manifest_sha
    )

    after_campaign = _campaign_row(registry.path)
    assert before_campaign["status"] == "planned"
    assert after_campaign["status"] == "complete"
    assert after_campaign["config_json"] == before_campaign["config_json"]
    assert after_campaign["created_at"] == before_campaign["created_at"]
    assert _campaign_table_checksums(registry.path) == before_tables
    assert _campaign_row(backup_path) == before_campaign
    assert result["comparison"]["outcome"] == "supported"
    assert result["comparison"]["failed_gates"] == ["baseline"]
    assert result["transaction"]["changed_count"] == 1
    assert result["transaction"]["run_or_queue_rows_modified"] is False
    assert result["transaction"]["failed_pilot_rows_preserved"] is True
    assert result["registry_evidence"]["resource_pilot"]["failed_run_ids"] == [
        "r_pilot_failed"
    ]
    assert receipt_path.is_file()
    receipt = dict(json.loads(receipt_path.read_text(encoding="utf-8")))
    checksum = receipt.pop("checksum")
    assert checksum == canonical_sha256(receipt)


def test_already_complete_is_a_verified_noop(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths, campaign_status="complete")
    manifest_path, manifest_sha = _report(paths, production)
    before = _campaign_row(registry.path)

    result, _, backup_path = _finalize(
        paths, manifest_path, manifest_sha, suffix="two"
    )

    assert result["transaction"]["changed_count"] == 0
    assert result["transaction"]["idempotent_already_complete"] is True
    assert _campaign_row(registry.path) == before
    assert _campaign_row(backup_path) == before


def test_versioned_report_is_accepted(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(
        paths,
        production,
        report_directory_name="comparison_v2",
    )

    result, _, _ = _finalize(
        paths, manifest_path, manifest_sha, suffix="versioned"
    )

    assert result["comparison"]["manifest_path"] == manifest_path.as_posix()
    assert result["transaction"]["changed_count"] == 1
    assert _campaign_row(registry.path)["status"] == "complete"


def test_noncanonical_report_directory_is_rejected(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(
        paths,
        production,
        report_directory_name="comparison_repaired",
    )
    backup = (
        paths.state_root
        / "tracking"
        / "backups"
        / "backup_noncanonical.sqlite3"
    )

    with pytest.raises(
        FinalizationError,
        match="canonical versioned MyJJu campaign report",
    ):
        _MODULE.finalize_campaign(
            paths=paths,
            database_path=registry.path,
            comparison_manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha,
            receipt_output_path=(
                _MODULE._reconciliation_directory(paths)
                / "receipt_noncanonical.json"
            ),
            backup_path=backup,
            finalized_at="2026-07-30T19:00:00.000000Z",
        )

    assert _campaign_row(registry.path)["status"] == "planned"
    assert not backup.exists()


def test_manifested_file_tamper_fails_before_backup(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(paths, production)
    (manifest_path.parent / "report.md").write_text(
        "# Mutated\n", encoding="utf-8"
    )
    backup = paths.state_root / "tracking" / "backups" / "backup_one.sqlite3"

    with pytest.raises(FinalizationError, match="size or checksum mismatch"):
        _finalize(paths, manifest_path, manifest_sha)

    assert _campaign_row(registry.path)["status"] == "planned"
    assert not backup.exists()


def test_active_campaign_state_blocks_finalization(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(paths, production)
    with registry.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE queue_jobs SET status = 'running' "
            "WHERE job_id = 'q_production_seed_0'"
        )
        connection.execute(
            "UPDATE runs SET status = 'running' "
            "WHERE run_id = 'r_production_seed_0'"
        )

    with pytest.raises(FinalizationError, match="active or unknown"):
        _finalize(paths, manifest_path, manifest_sha)

    assert _campaign_row(registry.path)["status"] == "planned"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verification_status", "declared"),
        ("best_epoch", 198),
        ("role", "best"),
    ],
)
def test_invalid_production_checkpoint_blocks_finalization(
    tmp_path: Path, field: str, value: Any
) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(paths, production)
    with registry.transaction(immediate=True) as connection:
        connection.execute(
            f"UPDATE checkpoint_catalog SET {field} = ? "
            "WHERE run_id = 'r_production_seed_6'",
            (value,),
        )

    with pytest.raises(FinalizationError, match="checkpoint"):
        _finalize(paths, manifest_path, manifest_sha)

    assert _campaign_row(registry.path)["status"] == "planned"


def test_report_and_registry_run_identity_mismatch_blocks(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(
        paths,
        production,
        run_id_override=(3, "r_unregistered_replacement"),
    )

    with pytest.raises(FinalizationError, match="registry and comparison differ"):
        _finalize(paths, manifest_path, manifest_sha)

    assert _campaign_row(registry.path)["status"] == "planned"


def test_report_cannot_omit_registered_genemae_pilot_failure(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(
        paths,
        production,
        omit_pilot_failure=True,
    )

    with pytest.raises(
        FinalizationError,
        match="pilot failure inventories differ",
    ):
        _finalize(paths, manifest_path, manifest_sha)

    assert _campaign_row(registry.path)["status"] == "planned"


def test_outcome_must_follow_primary_gate_only(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry, production = _registry_fixture(paths)
    manifest_path, manifest_sha = _report(
        paths,
        production,
        primary_passed=False,
        outcome="supported",
    )

    with pytest.raises(FinalizationError, match="primary frozen gate"):
        _finalize(paths, manifest_path, manifest_sha)

    assert _campaign_row(registry.path)["status"] == "planned"
