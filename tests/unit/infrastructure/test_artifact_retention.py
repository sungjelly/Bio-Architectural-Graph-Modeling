from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spatial_benchmark.artifact_retention import (
    ArtifactRetentionError,
    DELETED_BY_RETENTION,
    compact_registered_artifacts,
    retire_registered_run_bundles,
    select_retention_candidates,
)
from spatial_benchmark.cli import _doctor, _verify_bundles
from spatial_benchmark.identifiers import canonical_json
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import FULL_RUN_RETENTION_RECEIPT_KIND, Registry


def _paths(root: Path) -> ProjectPaths:
    paths = ProjectPaths(
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
    paths.ensure_runtime_directories()
    (paths.artifact_root / "runs" / "2026" / "08").mkdir(parents=True)
    return paths


def _register_run(
    registry: Registry,
    paths: ProjectPaths,
    *,
    run_id: str,
    status: str,
    stage: str,
    promoted: bool = False,
) -> Path:
    campaign_id = f"cmp_{run_id}"
    scientific_id = f"sci_{run_id}"
    registry.create_campaign(campaign_id, name=campaign_id)
    registry.register_variant(
        scientific_id,
        campaign_id=campaign_id,
        configuration={},
    )
    root = paths.artifact_root / "runs" / "2026" / "08" / run_id
    root.mkdir()
    (root / {"completed": "_SUCCESS", "failed": "_FAILED"}[status]).write_text(
        "terminal\n", encoding="utf-8"
    )
    registry.create_run(
        run_id,
        campaign_id=campaign_id,
        scientific_id=scientific_id,
        repro_id=f"rep_{run_id}",
        seed=0,
        fold=0,
        attempt=1,
        configuration={},
        status=status,
        artifact_path=root,
    )
    registry.register_run_semantics(
        run_id,
        lifecycle_stage=stage,
        study_axis="retention-test",
        source_batch="test",
        seed_known=True,
        fold_known=True,
        attempt_known=True,
        retention_class=f"retain_{stage}",
        category_key=f"{stage}/{run_id}",
        classification_confidence="high",
        timestamp_basis="test",
    )
    if promoted:
        with registry.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE runs SET promoted_at = ? WHERE run_id = ?",
                ("2026-08-02T00:00:00Z", run_id),
            )
    return root


def _record_file(
    registry: Registry,
    run_id: str,
    root: Path,
    *,
    name: str,
    kind: str,
    content: bytes,
    checkpoint: bool = False,
) -> tuple[int, Path]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    artifact_id = registry.record_artifact(
        run_id,
        kind=kind,
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )
    if checkpoint:
        registry.register_checkpoint_metadata(
            artifact_id,
            run_id=run_id,
            role="last",
            retention_class="retain_exploratory_evidence",
            verification_status="verified",
        )
    return artifact_id, path


def _register_full_bundle_run(
    registry: Registry,
    paths: ProjectPaths,
    *,
    campaign_id: str,
    run_id: str,
    status: str,
) -> tuple[Path, int, Path]:
    scientific_id = f"sci_{run_id}"
    registry.register_variant(
        scientific_id,
        campaign_id=campaign_id,
        configuration={},
    )
    root = paths.artifact_root / "runs" / "2026" / "08" / run_id
    root.mkdir()
    registry.create_run(
        run_id,
        campaign_id=campaign_id,
        scientific_id=scientific_id,
        repro_id=f"rep_{run_id}",
        seed=0,
        fold=0,
        attempt=1,
        configuration={},
        status=status,
        artifact_path=root,
    )
    registry.register_run_semantics(
        run_id,
        lifecycle_stage="posthoc_evaluation",
        study_axis="retirement-test",
        source_batch="test",
        seed_known=True,
        fold_known=True,
        attempt_known=True,
        retention_class="retain_conclusion_bearing_analysis",
        category_key=f"posthoc/{run_id}",
        classification_confidence="high",
        timestamp_basis="test",
    )
    payload_id, payload = _record_file(
        registry,
        run_id,
        root,
        name="result.bin",
        kind="metadata",
        content=b"result-payload",
    )
    checksums = {
        "result.bin": {
            "type": "file",
            "size": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        }
    }
    manifest_content = (
        json.dumps({"version": 1, "files": checksums}, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    _record_file(
        registry,
        run_id,
        root,
        name="provenance/artifact_checksums.json",
        kind="provenance",
        content=manifest_content,
    )
    marker_name = {"completed": "_SUCCESS", "failed": "_FAILED"}[status]
    marker_status = marker_name.removeprefix("_").lower()
    marker = root / marker_name
    marker.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": marker_status,
                "content_sha256": hashlib.sha256(
                    canonical_json(checksums).encode("utf-8")
                ).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, payload_id, payload


def test_retention_selection_excludes_small_locked_and_promoted_payloads(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    exploratory = _register_run(
        registry,
        paths,
        run_id="r_exploratory",
        status="completed",
        stage="exploratory_screen",
    )
    _record_file(
        registry,
        "r_exploratory",
        exploratory,
        name="checkpoints/large.ckpt",
        kind="checkpoints",
        content=b"x" * 16,
        checkpoint=True,
    )
    _record_file(
        registry,
        "r_exploratory",
        exploratory,
        name="checkpoints/small.ckpt",
        kind="checkpoints",
        content=b"x" * 4,
        checkpoint=True,
    )
    locked = _register_run(
        registry,
        paths,
        run_id="r_locked",
        status="completed",
        stage="locked_final",
    )
    _record_file(
        registry,
        "r_locked",
        locked,
        name="checkpoints/large.ckpt",
        kind="checkpoints",
        content=b"l" * 16,
        checkpoint=True,
    )
    promoted = _register_run(
        registry,
        paths,
        run_id="r_promoted",
        status="completed",
        stage="exploratory_screen",
        promoted=True,
    )
    _record_file(
        registry,
        "r_promoted",
        promoted,
        name="predictions/large.jsonl",
        kind="predictions",
        content=b"p" * 16,
    )

    selected = select_retention_candidates(
        registry,
        checkpoint_min_bytes=8,
        prediction_min_bytes=8,
    )

    assert [(item.run_id, Path(item.path).name) for item in selected] == [
        ("r_exploratory", "large.ckpt")
    ]


def test_retention_apply_tombstones_payloads_and_preserves_compact_evidence(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    completed = _register_run(
        registry,
        paths,
        run_id="r_completed",
        status="completed",
        stage="exploratory_screen",
    )
    checkpoint_id, checkpoint = _record_file(
        registry,
        "r_completed",
        completed,
        name="checkpoints/large.ckpt",
        kind="checkpoints",
        content=b"c" * 16,
        checkpoint=True,
    )
    _, prediction = _record_file(
        registry,
        "r_completed",
        completed,
        name="predictions/large.jsonl",
        kind="predictions",
        content=b"p" * 16,
    )
    _, compact = _record_file(
        registry,
        "r_completed",
        completed,
        name="summary.json",
        kind="metadata",
        content=b"{}\n",
    )
    failed = _register_run(
        registry,
        paths,
        run_id="r_failed",
        status="failed",
        stage="diagnostic",
    )
    failed_checkpoint_id, failed_checkpoint = _record_file(
        registry,
        "r_failed",
        failed,
        name="checkpoints/small.ckpt",
        kind="checkpoints",
        content=b"f" * 2,
        checkpoint=True,
    )
    output = paths.report_root / "retention" / "test_cleanup"

    plan = compact_registered_artifacts(
        registry,
        decision_id="test_cleanup",
        output_dir=output,
        paths=paths,
        checkpoint_min_bytes=8,
        prediction_min_bytes=8,
        apply=False,
    )
    assert plan["artifact_count"] == 3
    assert plan["total_bytes"] == 34
    assert checkpoint.exists() and prediction.exists() and failed_checkpoint.exists()

    applied = compact_registered_artifacts(
        registry,
        decision_id="test_cleanup",
        output_dir=output,
        paths=paths,
        checkpoint_min_bytes=8,
        prediction_min_bytes=8,
        apply=True,
    )

    assert applied["deleted_bytes"] == 34
    assert not checkpoint.exists()
    assert not prediction.exists()
    assert not failed_checkpoint.exists()
    assert compact.read_bytes() == b"{}\n"
    assert (completed / "_SUCCESS").is_file()
    assert (failed / "_FAILED").is_file()
    assert (output / "deletion_plan.jsonl").is_file()
    assert (output / "application_receipt.json").is_file()
    assert (paths.state_root / "backups" / "bagm_pre_test_cleanup.sqlite3").is_file()
    with registry.connect() as connection:
        statuses = {
            int(row["artifact_id"]): str(row["status"])
            for row in connection.execute(
                "SELECT artifact_id, status FROM artifacts"
            )
        }
        checkpoint_statuses = {
            int(row["artifact_id"]): str(row["verification_status"])
            for row in connection.execute(
                "SELECT artifact_id, verification_status FROM checkpoint_catalog"
            )
        }
    assert statuses[checkpoint_id] == DELETED_BY_RETENTION
    assert statuses[failed_checkpoint_id] == DELETED_BY_RETENTION
    assert checkpoint_statuses[checkpoint_id] == DELETED_BY_RETENTION
    assert checkpoint_statuses[failed_checkpoint_id] == DELETED_BY_RETENTION
    assert registry.verify_artifacts() == []


def test_full_run_retirement_requires_complete_campaign_and_verifies_receipt(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    campaign_id = "cmp_full_retirement"
    registry.create_campaign(campaign_id, name=campaign_id)
    first_root, prior_id, prior_payload = _register_full_bundle_run(
        registry,
        paths,
        campaign_id=campaign_id,
        run_id="r_retire_first",
        status="failed",
    )
    second_root, _, _ = _register_full_bundle_run(
        registry,
        paths,
        campaign_id=campaign_id,
        run_id="r_retire_second",
        status="completed",
    )
    prior_payload.unlink()
    with registry.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE artifacts SET status = ? WHERE artifact_id = ?",
            (DELETED_BY_RETENTION, prior_id),
        )
    output = paths.report_root / "retention" / "retire_complete_campaign"

    with pytest.raises(ArtifactRetentionError, match="complete run set"):
        retire_registered_run_bundles(
            registry,
            run_ids=["r_retire_first"],
            expected_campaign_id=campaign_id,
            decision_id="retire_complete_campaign",
            output_dir=output,
            paths=paths,
            apply=False,
        )

    planned = retire_registered_run_bundles(
        registry,
        run_ids=["r_retire_first", "r_retire_second"],
        expected_campaign_id=campaign_id,
        decision_id="retire_complete_campaign",
        output_dir=output,
        prior_retention_decision_ids=["earlier_cleanup"],
        paths=paths,
        apply=False,
    )
    assert planned["original_artifact_row_count"] == 4
    assert planned["present_artifact_count"] == 3
    assert planned["prior_tombstone_count"] == 1
    assert first_root.is_dir() and second_root.is_dir()

    applied = retire_registered_run_bundles(
        registry,
        run_ids=["r_retire_second", "r_retire_first"],
        expected_campaign_id=campaign_id,
        decision_id="retire_complete_campaign",
        output_dir=output,
        prior_retention_decision_ids=["earlier_cleanup"],
        paths=paths,
        apply=True,
    )

    assert applied["deleted_registered_artifact_count"] == 3
    assert applied["deleted_terminal_marker_count"] == 2
    assert not first_root.exists() and not second_root.exists()
    verified, issues = registry.verify_full_run_retirements()
    assert verified == {"r_retire_first", "r_retire_second"}
    assert issues == []
    assert registry.verify_artifacts() == []
    assert _verify_bundles(registry, None, paths) == []
    doctor = _doctor(registry, paths)
    assert not any(
        issue.get("kind") == "missing_canonical_markers"
        for issue in doctor["issues"]
    )
    with registry.connect() as connection:
        original_statuses = {
            str(row["status"])
            for row in connection.execute(
                """
                SELECT status FROM artifacts
                WHERE kind != ?
                """,
                (FULL_RUN_RETENTION_RECEIPT_KIND,),
            )
        }
        receipt_rows = connection.execute(
            "SELECT run_id, status FROM artifacts WHERE kind = ?",
            (FULL_RUN_RETENTION_RECEIPT_KIND,),
        ).fetchall()
    assert original_statuses == {DELETED_BY_RETENTION}
    assert {(row["run_id"], row["status"]) for row in receipt_rows} == {
        ("r_retire_first", "present"),
        ("r_retire_second", "present"),
    }

    first_root.mkdir()
    verified, issues = registry.verify_full_run_retirements()
    assert verified == {"r_retire_second"}
    assert any(
        issue["issue"] == "full_retirement_root_still_exists" for issue in issues
    )
    assert _verify_bundles(registry, None, paths)
