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
_SCRIPT = _ROOT / "scripts" / "train" / "finalize_pooled_campaign_registry.py"
_SPEC = importlib.util.spec_from_file_location(
    "finalize_pooled_campaign_registry_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FinalizationError = _MODULE.PooledCampaignFinalizationError


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison(
    paths: ProjectPaths,
    *,
    campaign_id: str = _MODULE.CAMPAIGN_ID,
    status: str = "complete",
    outcome: str = "negative",
) -> tuple[Path, str]:
    directory = _MODULE._expected_comparison_directory(paths)
    gates = {
        "pooled_data_gate": {"passed": False},
        "graph_gate": {"passed": True},
        "representation_gate": {"passed": False},
        "ensemble_gate": {"passed": True},
        "formal_core_level_inference": "not_performed",
    }
    failed = ["pooled_data_gate", "representation_gate"]
    comparison = {
        "schema_version": 1,
        "artifact_kind": _MODULE.COMPARISON_KIND,
        "campaign_id": campaign_id,
        "status": status,
        "outcome": outcome,
        "exploratory": True,
        "tissue_context": "pathology_confirmed_adjacent_normal",
        "tissue_context_is_true_normal": False,
        "frozen_gates": gates,
        "failed_gates": failed,
        "coverage": {
            "production_slots_expected": 14,
            "production_slots_completed": 14,
            "failed_production_attempts_before_completion": 0,
        },
        "training_scope": {
            "one_shared_model_per_member": True,
            "cross_core_edges": False,
            "production_run_count": 14,
            "core_aliases": list(_MODULE.ALIASES),
        },
    }
    _write_json(directory / "comparison.json", comparison)
    _write_json(
        directory / "provenance.json",
        {
            "campaign_id": _MODULE.CAMPAIGN_ID,
            "protected_identifiers_emitted": False,
        },
    )
    (directory / "report.md").write_text("# Test report\n", encoding="utf-8")
    files = {
        path.relative_to(directory).as_posix(): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "artifact_kind": _MODULE.COMPARISON_KIND,
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "portable_html": True,
        "protected_identifiers_emitted": False,
        "files": files,
        "comparison_sha256": files["comparison.json"]["sha256"],
        "provenance_sha256": files["provenance.json"]["sha256"],
    }
    manifest_path = directory / "manifest.json"
    _write_json(manifest_path, manifest)
    return directory, _sha256(manifest_path)


def _registry(paths: ProjectPaths, *, status: str = "planned") -> Registry:
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    registry.create_campaign(
        _MODULE.CAMPAIGN_ID,
        name="pooled test",
        scientific_question="test",
        config={"frozen": {"value": 7}},
        status=status,
    )
    return registry


def _campaign_raw(database: Path) -> dict[str, Any]:
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT status, config_json, created_at, updated_at "
            "FROM campaigns WHERE campaign_id = ?",
            (_MODULE.CAMPAIGN_ID,),
        ).fetchone()
    assert row is not None
    return dict(row)


def _finalize(
    paths: ProjectPaths,
    directory: Path,
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
        comparison_directory=directory,
        expected_manifest_sha256=manifest_sha256,
        receipt_output_path=receipt,
        backup_path=backup,
        finalized_at=f"2026-07-30T10:00:0{0 if suffix == 'one' else 1}.000000Z",
    )
    return result, receipt, backup


def test_finalization_cas_backs_up_and_preserves_config(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    registry = _registry(paths)
    directory, manifest_sha256 = _comparison(paths)
    database = registry.path
    before = _campaign_raw(database)

    result, receipt_path, backup_path = _finalize(
        paths, directory, manifest_sha256
    )

    after = _campaign_raw(database)
    backup_row = _campaign_raw(backup_path)
    assert before["status"] == "planned"
    assert after["status"] == "complete"
    assert backup_row == before
    assert after["config_json"] == before["config_json"]
    assert after["created_at"] == before["created_at"]
    assert result["transaction"]["changed_count"] == 1
    assert result["transaction"]["config_json_unchanged"] is True
    assert result["database"]["backup"]["sha256"] == _sha256(backup_path)
    assert receipt_path.is_file()
    unsigned = dict(json.loads(receipt_path.read_text(encoding="utf-8")))
    checksum = unsigned.pop("checksum")
    assert checksum == canonical_sha256(unsigned)


def test_already_complete_is_verified_idempotently(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _registry(paths, status="complete")
    directory, manifest_sha256 = _comparison(paths)
    before = _campaign_raw(paths.state_root / "tracking" / "bagm.sqlite3")

    result, _, backup_path = _finalize(
        paths, directory, manifest_sha256, suffix="two"
    )

    assert result["transaction"]["changed_count"] == 0
    assert result["transaction"]["idempotent_already_complete"] is True
    assert _campaign_raw(paths.state_root / "tracking" / "bagm.sqlite3") == before
    assert _campaign_raw(backup_path) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign_id", "cmp_wrong"),
        ("status", "running"),
        ("outcome", "supported"),
    ],
)
def test_invalid_comparison_identity_status_or_outcome_fails_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    paths = _paths(tmp_path)
    registry = _registry(paths)
    arguments = {field: value}
    directory, manifest_sha256 = _comparison(paths, **arguments)
    receipt = _MODULE._reconciliation_directory(paths) / "receipt.json"
    backup = paths.state_root / "tracking" / "backups" / "backup.sqlite3"

    with pytest.raises(FinalizationError):
        _MODULE.finalize_campaign(
            paths=paths,
            database_path=registry.path,
            comparison_directory=directory,
            expected_manifest_sha256=manifest_sha256,
            receipt_output_path=receipt,
            backup_path=backup,
        )

    assert _campaign_raw(registry.path)["status"] == "planned"
    assert not receipt.exists()
    assert not backup.exists()


def test_manifested_file_tamper_fails_before_registry_mutation(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    registry = _registry(paths)
    directory, manifest_sha256 = _comparison(paths)
    (directory / "report.md").write_text("# Changed\n", encoding="utf-8")
    receipt = _MODULE._reconciliation_directory(paths) / "receipt.json"
    backup = paths.state_root / "tracking" / "backups" / "backup.sqlite3"

    with pytest.raises(FinalizationError, match="size or checksum mismatch"):
        _MODULE.finalize_campaign(
            paths=paths,
            database_path=registry.path,
            comparison_directory=directory,
            expected_manifest_sha256=manifest_sha256,
            receipt_output_path=receipt,
            backup_path=backup,
        )

    assert _campaign_raw(registry.path)["status"] == "planned"
    assert not backup.exists()


@pytest.mark.parametrize("active_kind", ["queue", "run"])
def test_active_queue_or_run_state_blocks_finalization(
    tmp_path: Path, active_kind: str
) -> None:
    paths = _paths(tmp_path)
    registry = _registry(paths)
    directory, manifest_sha256 = _comparison(paths)
    if active_kind == "queue":
        registry.enqueue(
            campaign_id=_MODULE.CAMPAIGN_ID,
            configuration={"test": active_kind},
            command=["python", "train.py"],
            job_id="q_active",
        )
    else:
        registry.register_variant(
            "s_active",
            campaign_id=_MODULE.CAMPAIGN_ID,
            configuration={"model": {"family": "test"}},
        )
        registry.create_run(
            "r_active",
            campaign_id=_MODULE.CAMPAIGN_ID,
            scientific_id="s_active",
            repro_id="repro",
            seed=0,
            fold=0,
            attempt=1,
            configuration={"model": {"family": "test"}},
            status="pending",
        )

    with pytest.raises(FinalizationError, match="active or unknown"):
        _finalize(paths, directory, manifest_sha256)

    assert _campaign_raw(registry.path)["status"] == "planned"
