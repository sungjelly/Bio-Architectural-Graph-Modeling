from __future__ import annotations

import hashlib
from pathlib import Path

from spatial_benchmark.artifact_retention import (
    DELETED_BY_RETENTION,
    compact_registered_artifacts,
    select_retention_candidates,
)
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry


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
