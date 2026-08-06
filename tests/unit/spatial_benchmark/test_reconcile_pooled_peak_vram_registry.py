from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256, scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry, utc_now


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts" / "train" / "reconcile_pooled_peak_vram_registry.py"
_SPEC = importlib.util.spec_from_file_location(
    "reconcile_pooled_peak_vram_registry_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ReconciliationError = _MODULE.PeakVramReconciliationError


def _signed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _config(*, stage: str, arm: str, seed: int) -> dict[str, Any]:
    pilot = stage == "pilot"
    return {
        "version": 1,
        "campaign": {
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "frozen_contract_sha256": _MODULE.EXPECTED_CONTRACT_SHA256,
        },
        "dataset": {"core_aliases": list(_MODULE.ALIASES)},
        "experiment": {
            "arm": arm,
            "core_aliases": list(_MODULE.ALIASES),
            "resource_pilot": pilot,
        },
        "trainer": {"diagnostic_resource_pilot": pilot},
        "evaluation": {"protocol": _MODULE.EXPECTED_PROTOCOL},
        "classification": {
            "scientific_variant": arm,
            "lifecycle_stage": "diagnostic" if pilot else "exploratory_screen",
        },
        "metadata": {
            "execution_role": "resource_pilot" if pilot else "production"
        },
        "seed": seed,
        "fold": 0,
        "attempt": 1,
    }


def _bundle(
    *,
    paths: ProjectPaths,
    run_id: str,
    stage: str,
    arm: str,
    seed: int,
    materialization_checksum: str,
    peak: float,
) -> Path:
    root = paths.artifact_root / "runs" / "2026" / "07" / run_id
    summary = {
        "run_id": run_id,
        "status": "success",
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "evaluation_protocol": _MODULE.EXPECTED_PROTOCOL,
        "aliases": list(_MODULE.ALIASES),
        "public_variant": arm,
        "model_seed": seed,
        "diagnostic_resource_pilot": stage == "pilot",
        "run_attempt": 1,
        "materialization_checksum": materialization_checksum,
        "peak_vram_gib": peak,
    }
    final_metrics = {"resource/peak_vram_gib": peak}
    resource = {
        "schema": "pooled_hybrid_count_resource_diagnostic_v1",
        "aliases": list(_MODULE.ALIASES),
        "public_variant": arm,
        "diagnostic_resource_pilot": stage == "pilot",
        "peak_allocated_vram_gib": peak,
    }
    source_payloads = {
        "summary.json": summary,
        "metrics/final.json": final_metrics,
        "diagnostics/resource_usage.json": resource,
    }
    for relative, payload in source_payloads.items():
        _write_json(root / relative, payload)
    files = {
        relative: {
            "type": "file",
            "sha256": _sha256(root / relative),
            "size": (root / relative).stat().st_size,
        }
        for relative in source_payloads
    }
    _write_json(root / "provenance/artifact_checksums.json", {"files": files})
    _write_json(
        root / "_SUCCESS",
        {
            "status": "success",
            "run_id": run_id,
            "content_sha256": canonical_sha256(files),
        },
    )
    return root


def _refresh_bundle_manifest(root: Path, relative: str) -> None:
    manifest_path = root / "provenance/artifact_checksums.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative] = {
        "type": "file",
        "sha256": _sha256(root / relative),
        "size": (root / relative).stat().st_size,
    }
    _write_json(manifest_path, manifest)
    _write_json(
        root / "_SUCCESS",
        {
            "status": "success",
            "run_id": root.name,
            "content_sha256": canonical_sha256(manifest["files"]),
        },
    )


def _make_campaign(tmp_path: Path) -> dict[str, Any]:
    paths = _paths(tmp_path)
    contract = (
        tmp_path
        / "experiments"
        / "campaigns"
        / _MODULE.CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    contract.parent.mkdir(parents=True, exist_ok=True)
    # The production constant is deliberately reused as content so the fixture
    # exercises the exact immutable-contract checksum check.
    contract.write_bytes(b"")
    # Finding a preimage is neither intended nor useful in a unit fixture.
    # Patch the file hasher only for this one contract path in _build().

    locked = paths.scratch_root / "locked_campaigns" / _MODULE.CAMPAIGN_ID
    materialized: dict[tuple[str, str, int], dict[str, Any]] = {}
    for stage, seeds in (("pilot", (0,)), ("production", _MODULE.SEEDS)):
        for seed in seeds:
            for arm in _MODULE.ARMS:
                config = _config(stage=stage, arm=arm, seed=seed)
                relative = (
                    Path("scratch")
                    / "locked_campaigns"
                    / _MODULE.CAMPAIGN_ID
                    / "configs"
                    / f"{stage}_{arm}_{seed}.yaml"
                )
                config_path = tmp_path / relative
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                )
                materialized[(stage, arm, seed)] = {
                    "arm": arm,
                    "seed": seed,
                    "config": relative.as_posix(),
                    "config_sha256": canonical_sha256(config),
                    "file_sha256": _sha256(config_path),
                    "requested_gpu": (
                        0
                        if stage == "pilot" and arm == _MODULE.ARMS[0]
                        else 1
                        if stage == "pilot"
                        else (0, 1, 2, 3, 5, 6, 7)[seed]
                    ),
                    "config_payload": config,
                }
    materialization = _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.MATERIALIZATION_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "counts": {
                "aliases": 10,
                "pilot_configs": 2,
                "production_configs": 14,
                "production_seeds": 7,
            },
            "cohort": {"aliases": list(_MODULE.ALIASES)},
            "frozen_contract": {
                "reference": contract.relative_to(tmp_path).as_posix(),
                "sha256": _MODULE.EXPECTED_CONTRACT_SHA256,
            },
            "pilot_jobs": [
                {
                    key: value
                    for key, value in materialized[("pilot", arm, 0)].items()
                    if key != "config_payload"
                }
                for arm in _MODULE.ARMS
            ],
            "production_jobs": [
                {
                    key: value
                    for key, value in materialized[("production", arm, seed)].items()
                    if key != "config_payload"
                }
                for seed in _MODULE.SEEDS
                for arm in _MODULE.ARMS
            ],
            "queue_mutation_performed": False,
            "registry_mutation_performed": False,
            "training_performed": False,
        }
    )
    materialization_path = locked / "locked_config_materialization.json"
    _write_json(materialization_path, materialization)

    def root_id(stage: str, arm: str, seed: int) -> str:
        arm_key = "gat" if arm == _MODULE.ARMS[0] else "self"
        return f"q-{stage}-{arm_key}-{seed}"

    pilot_enqueue = _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.PILOT_ENQUEUE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "stage": "pilot",
            "materialization_checksum": materialization["checksum"],
            "pilot_gate_checksum": None,
            "complete": True,
            "jobs": [
                {
                    "arm": arm,
                    "seed": 0,
                    "job_id": root_id("pilot", arm, 0),
                    "config_sha256": materialized[
                        ("pilot", arm, 0)
                    ]["config_sha256"],
                    "requested_gpu": materialized[
                        ("pilot", arm, 0)
                    ]["requested_gpu"],
                    "maximum_attempts": 2,
                }
                for arm in _MODULE.ARMS
            ],
        }
    )
    pilot_enqueue_path = locked / "pilot_enqueue_receipt.json"
    _write_json(pilot_enqueue_path, pilot_enqueue)

    pilot_run_ids = {
        arm: f"r-pilot-{'gat' if arm == _MODULE.ARMS[0] else 'self'}"
        for arm in _MODULE.ARMS
    }
    pilot_gate = _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.PILOT_GATE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "frozen_contract_sha256": _MODULE.EXPECTED_CONTRACT_SHA256,
            "materialization_checksum": materialization["checksum"],
            "pilot_enqueue_receipt_checksum": pilot_enqueue["checksum"],
            "gate_passed": True,
            "production_authorized": True,
            "failure_reasons": [],
            "jobs": [
                {
                    "arm": arm,
                    "seed": 0,
                    "original_enqueue_job_id": root_id("pilot", arm, 0),
                    "completed_job_id": root_id("pilot", arm, 0),
                    "completed_attempt": 1,
                    "run_id": pilot_run_ids[arm],
                    "config_sha256": materialized[
                        ("pilot", arm, 0)
                    ]["config_sha256"],
                    "verified_bundle": True,
                    "checkpoint_verified": True,
                    "attempts": [
                        {
                            "job_id": root_id("pilot", arm, 0),
                            "attempt": 1,
                            "maximum_attempts": 2,
                            "status": "completed",
                            "run_id": pilot_run_ids[arm],
                            "failure_category": None,
                            "retry_of": None,
                            "is_original_enqueue_job": True,
                            "selected_completed_attempt": True,
                        }
                    ],
                }
                for arm in _MODULE.ARMS
            ],
        }
    )
    pilot_gate_path = locked / "pilot_gate_receipt.json"
    _write_json(pilot_gate_path, pilot_gate)

    production_enqueue = _signed(
        {
            "schema_version": 1,
            "receipt_kind": _MODULE.PRODUCTION_ENQUEUE_KIND,
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "stage": "production",
            "materialization_checksum": materialization["checksum"],
            "pilot_gate_checksum": pilot_gate["checksum"],
            "complete": True,
            "jobs": [
                {
                    "arm": arm,
                    "seed": seed,
                    "job_id": root_id("production", arm, seed),
                    "config_sha256": materialized[
                        ("production", arm, seed)
                    ]["config_sha256"],
                    "requested_gpu": materialized[
                        ("production", arm, seed)
                    ]["requested_gpu"],
                    "maximum_attempts": 2,
                }
                for seed in _MODULE.SEEDS
                for arm in _MODULE.ARMS
            ],
        }
    )
    production_enqueue_path = locked / "production_enqueue_receipt.json"
    _write_json(production_enqueue_path, production_enqueue)

    database = paths.state_root / "tracking" / "bagm.sqlite3"
    registry = Registry(database)
    registry.create_campaign(
        _MODULE.CAMPAIGN_ID,
        name="fixture pooled campaign",
        scientific_question="fixture",
        config={
            "frozen_contract_sha256": _MODULE.EXPECTED_CONTRACT_SHA256,
            "materialization_checksum": materialization["checksum"],
            "production_run_count": 14,
            "production_seeds": list(_MODULE.SEEDS),
        },
    )
    bundles: dict[tuple[str, str, int], Path] = {}
    for slot, planned in materialized.items():
        stage, arm, seed = slot
        arm_key = "gat" if arm == _MODULE.ARMS[0] else "self"
        run_id = (
            pilot_run_ids[arm]
            if stage == "pilot"
            else f"r-production-{arm_key}-{seed}"
        )
        job_id = root_id(stage, arm, seed)
        config = planned["config_payload"]
        scientific_identifier = scientific_id(config)
        registry.register_variant(
            scientific_identifier,
            campaign_id=_MODULE.CAMPAIGN_ID,
            configuration=config,
        )
        registry.enqueue(
            campaign_id=_MODULE.CAMPAIGN_ID,
            configuration=config,
            command=["python", "fixture.py"],
            experiment_config_reference=planned["config"],
            priority=10,
            maximum_attempts=2,
            requested_gpu=str(planned["requested_gpu"]),
            job_id=job_id,
        )
        root = _bundle(
            paths=paths,
            run_id=run_id,
            stage=stage,
            arm=arm,
            seed=seed,
            materialization_checksum=materialization["checksum"],
            peak=2.5 + seed / 10,
        )
        bundles[slot] = root
        registry.create_run(
            run_id,
            campaign_id=_MODULE.CAMPAIGN_ID,
            scientific_id=scientific_identifier,
            repro_id=f"repro-{stage}-{arm_key}-{seed}",
            seed=seed,
            fold=0,
            attempt=1,
            configuration=config,
            status="completed",
            artifact_path=root,
        )
        summary = root / "summary.json"
        registry.record_artifact(
            run_id,
            kind="metadata",
            path=summary,
            sha256=_sha256(summary),
            size_bytes=summary.stat().st_size,
        )
        with registry.transaction(immediate=True) as connection:
            connection.execute(
                """
                UPDATE queue_jobs
                SET status = 'completed', run_id = ?, finished_at = ?
                WHERE job_id = ?
                """,
                (run_id, utc_now(), job_id),
            )
    return {
        "paths": paths,
        "database": database,
        "registry": registry,
        "materialization": materialization_path,
        "pilot_enqueue": pilot_enqueue_path,
        "pilot_gate": pilot_gate_path,
        "production_enqueue": production_enqueue_path,
        "contract": contract,
        "bundles": bundles,
    }


def _build(
    fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    original_hash = _MODULE._sha256_file
    contract = fixture["contract"].resolve(strict=False)

    def hash_with_frozen_fixture(path: Path) -> str:
        if path.resolve(strict=False) == contract:
            return _MODULE.EXPECTED_CONTRACT_SHA256
        return original_hash(path)

    monkeypatch.setattr(_MODULE, "_sha256_file", hash_with_frozen_fixture)
    return _MODULE.build_reconciliation_plan(
        paths=fixture["paths"],
        database_path=fixture["database"],
        materialization_path=fixture["materialization"],
        pilot_enqueue_path=fixture["pilot_enqueue"],
        pilot_gate_path=fixture["pilot_gate"],
        production_enqueue_path=fixture["production_enqueue"],
        created_at="2026-07-30T12:00:00Z",
        bundle_verifier=lambda _root: {"valid": True, "status": "success"},
    )


def test_default_plan_is_signed_complete_and_registry_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_campaign(tmp_path)
    plan = _build(fixture, monkeypatch)
    unsigned = dict(plan)
    checksum = unsigned.pop("checksum")
    assert canonical_sha256(unsigned) == checksum
    assert plan["mode"] == "read_only_plan"
    assert plan["actions"] == {
        "backfill": 16,
        "already_consistent": 0,
        "conflict": 0,
    }
    assert len(plan["items"]) == 16
    assert {item["stage"] for item in plan["items"]} == {"pilot", "production"}
    assert all(item["registry_value_binary_gib"] is None for item in plan["items"])
    with fixture["registry"].connect() as connection:
        values = connection.execute(
            "SELECT peak_vram_gb FROM runs WHERE campaign_id = ?",
            (_MODULE.CAMPAIGN_ID,),
        ).fetchall()
    assert len(values) == 16
    assert all(row["peak_vram_gb"] is None for row in values)

    output = tmp_path / "reports" / "plan.json"
    _MODULE.write_reconciliation_plan(output, plan)
    with pytest.raises(FileExistsError):
        _MODULE.write_reconciliation_plan(output, plan)


def test_plan_fails_before_bundle_audit_when_campaign_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_campaign(tmp_path)
    with fixture["registry"].transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE queue_jobs SET status = 'running' WHERE job_id = ?",
            ("q-production-gat-0",),
        )
        connection.execute(
            "UPDATE runs SET status = 'running' WHERE run_id = ?",
            ("r-production-gat-0",),
        )
    calls = 0

    def forbidden(_root: Path) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("active campaign reached bundle verification")

    original_hash = _MODULE._sha256_file

    def fixture_hash(path: Path) -> str:
        if path.resolve(strict=False) == fixture["contract"].resolve(strict=False):
            return _MODULE.EXPECTED_CONTRACT_SHA256
        return original_hash(path)

    monkeypatch.setattr(_MODULE, "_sha256_file", fixture_hash)
    with pytest.raises(ReconciliationError, match="active or incomplete"):
        _MODULE.build_reconciliation_plan(
            paths=fixture["paths"],
            database_path=fixture["database"],
            materialization_path=fixture["materialization"],
            pilot_enqueue_path=fixture["pilot_enqueue"],
            pilot_gate_path=fixture["pilot_gate"],
            production_enqueue_path=fixture["production_enqueue"],
            bundle_verifier=forbidden,
        )
    assert calls == 0


def test_plan_rejects_metric_disagreement_and_registry_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_campaign(tmp_path)
    root = fixture["bundles"][("production", _MODULE.ARMS[0], 0)]
    final_path = root / "metrics/final.json"
    _write_json(final_path, {"resource/peak_vram_gib": 9.0})
    _refresh_bundle_manifest(root, "metrics/final.json")
    with pytest.raises(ReconciliationError, match="values disagree"):
        _build(fixture, monkeypatch)

    _write_json(final_path, {"resource/peak_vram_gib": 2.5})
    _refresh_bundle_manifest(root, "metrics/final.json")
    with fixture["registry"].transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE runs SET peak_vram_gb = 8.0 WHERE run_id = ?",
            ("r-production-gat-0",),
        )
    with pytest.raises(ReconciliationError, match="conflicting non-null"):
        _build(fixture, monkeypatch)


def test_plan_rejects_unrelated_queue_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_campaign(tmp_path)
    config = _config(stage="production", arm=_MODULE.ARMS[0], seed=0)
    fixture["registry"].enqueue(
        campaign_id=_MODULE.CAMPAIGN_ID,
        configuration=config,
        command=["python", "fixture.py"],
        experiment_config_reference="unexpected.yaml",
        priority=10,
        maximum_attempts=2,
        requested_gpu="0",
        job_id="q-unrelated-root",
    )
    with fixture["registry"].transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE queue_jobs SET status = 'cancelled' WHERE job_id = ?",
            ("q-unrelated-root",),
        )
    with pytest.raises(ReconciliationError, match="unrelated root"):
        _build(fixture, monkeypatch)


def test_apply_requires_exact_reviewed_checksum_backs_up_and_reconciles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_campaign(tmp_path)
    plan = _build(fixture, monkeypatch)
    plan_path = tmp_path / "reports" / "reviewed-plan.json"
    _MODULE.write_reconciliation_plan(plan_path, plan)
    report_root = _MODULE._reconciliation_report_root(fixture["paths"])
    receipt_path = report_root / "application.json"
    backup_path = (
        fixture["paths"].state_root
        / "tracking"
        / "backups"
        / "before.sqlite3"
    )

    with pytest.raises(ReconciliationError, match="expected checksum"):
        _MODULE.apply_reconciliation_plan(
            paths=fixture["paths"],
            database_path=fixture["database"],
            plan_path=plan_path,
            expected_plan_checksum="0" * 64,
            receipt_output_path=receipt_path,
            materialization_path=fixture["materialization"],
            pilot_enqueue_path=fixture["pilot_enqueue"],
            pilot_gate_path=fixture["pilot_gate"],
            production_enqueue_path=fixture["production_enqueue"],
            bundle_verifier=lambda _root: {"valid": True, "status": "success"},
            post_apply_verifier=lambda **_kwargs: {"all_passed": True},
            backup_path=backup_path,
        )
    assert not backup_path.exists()
    receipt = _MODULE.apply_reconciliation_plan(
        paths=fixture["paths"],
        database_path=fixture["database"],
        plan_path=plan_path,
        expected_plan_checksum=plan["checksum"],
        receipt_output_path=receipt_path,
        materialization_path=fixture["materialization"],
        pilot_enqueue_path=fixture["pilot_enqueue"],
        pilot_gate_path=fixture["pilot_gate"],
        production_enqueue_path=fixture["production_enqueue"],
        bundle_verifier=lambda _root: {"valid": True, "status": "success"},
        post_apply_verifier=lambda **_kwargs: {"all_passed": True},
        backup_path=backup_path,
    )
    assert backup_path.is_file()
    assert _sha256(backup_path) == receipt["database_backup"]["sha256"]
    assert receipt["transaction"]["changed_count"] == 16
    assert len(receipt["transaction"]["changed_run_ids"]) == 16
    assert receipt_path.is_file()
    unsigned = dict(receipt)
    assert canonical_sha256({k: v for k, v in unsigned.items() if k != "checksum"}) == (
        receipt["checksum"]
    )
    with fixture["registry"].connect() as connection:
        values = connection.execute(
            "SELECT peak_vram_gb FROM runs WHERE campaign_id = ?",
            (_MODULE.CAMPAIGN_ID,),
        ).fetchall()
    assert all(row["peak_vram_gb"] is not None for row in values)
    with pytest.raises(ReconciliationError, match="already been applied"):
        _MODULE.apply_reconciliation_plan(
            paths=fixture["paths"],
            database_path=fixture["database"],
            plan_path=plan_path,
            expected_plan_checksum=plan["checksum"],
            receipt_output_path=report_root / "second-application.json",
            materialization_path=fixture["materialization"],
            pilot_enqueue_path=fixture["pilot_enqueue"],
            pilot_gate_path=fixture["pilot_gate"],
            production_enqueue_path=fixture["production_enqueue"],
            bundle_verifier=lambda _root: {"valid": True, "status": "success"},
            post_apply_verifier=lambda **_kwargs: {"all_passed": True},
            backup_path=(
                fixture["paths"].state_root
                / "tracking"
                / "backups"
                / "second.sqlite3"
            ),
        )


def test_conditional_transaction_rolls_back_all_rows_on_state_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _make_campaign(tmp_path)
    plan = _build(fixture, monkeypatch)
    first = plan["items"][0]["run_id"]
    second = plan["items"][1]["run_id"]
    second_value = plan["items"][1]["proposed_value_binary_gib"]
    with fixture["registry"].transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE runs SET peak_vram_gb = ? WHERE run_id = ?",
            (second_value, second),
        )
    # Make the lineage snapshot current while deliberately retaining the stale
    # null-only backfill action. The second conditional update must fail after
    # the first update and roll the whole transaction back.
    stale = deepcopy(plan)
    for row in stale["lineage"]["run_attempts"]:
        if row["run_id"] == second:
            row["registry_peak_vram_gb"] = second_value
    with pytest.raises(ReconciliationError, match="conditional backfill"):
        _MODULE._transactional_backfill(
            registry=fixture["registry"],
            plan=stale,
            applied_at="2026-07-30T12:30:00Z",
        )
    assert fixture["registry"].get_run(first)["peak_vram_gb"] is None
    assert fixture["registry"].get_run(second)["peak_vram_gb"] == second_value
