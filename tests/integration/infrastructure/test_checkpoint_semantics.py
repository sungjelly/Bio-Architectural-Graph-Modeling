from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from spatial_benchmark.checkpoint_catalog import (
    CheckpointCatalogError,
    CheckpointNotFoundError,
    CheckpointVerificationError,
    build_checkpoint_catalog,
    export_checkpoint_catalog,
    index_checkpoint_catalog,
    resolve_checkpoint,
    run_semantics_from_configuration,
    show_checkpoint,
)
from spatial_benchmark.cli import main
from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import QueueWorker, WorkerSettings
from spatial_benchmark.registry import Registry


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


def _configuration(*, classified: bool = True) -> dict[str, object]:
    configuration: dict[str, object] = {
        "version": 1,
        "campaign": {"campaign_id": "cmp_semantic_test"},
        "model": {
            "name": "g1",
            "family": "topology_only",
            "embedding_dim": 32,
        },
        "masking": {"type": "whole_node", "rate": 0.1},
        "dataset": {
            "dataset_id": "synthetic",
            "version": "v1",
            "split_id": "split_1",
            "dataset_fingerprint": "dataset-fingerprint",
            "split_fingerprint": "split-fingerprint",
            "preprocessing_version": "test-v1",
        },
        "features": {"use_edge_features": False, "edge_features": []},
        "graph": {
            "graph_id": "k4_r10_mutual",
            "neighbor_k": 4,
            "radius_um": 10.0,
            "symmetry": "mutual",
        },
        "trainer": {"learning_rate": 0.001, "batch_size": 1},
        "evaluation": {
            "primary_metric": "val/masked_huber",
            "primary_direction": "minimize",
        },
        "seed": 3,
        "fold": 2,
        "attempt": 1,
        "metadata": {"test_only_dummy": True},
    }
    if classified:
        configuration["classification"] = {
            "schema_version": 1,
            "lifecycle_stage": "exploratory_screen",
            "study_axis": "graph_screen",
            "retention_class": "retain_exploratory_evidence",
            "classification_confidence": "high",
            "source_batch": "future_graph_screen_v1",
        }
    return configuration


def _registry(root: Path) -> tuple[Registry, ProjectPaths]:
    paths = _paths(root)
    registry = Registry(paths.state_root / "tracking" / "bagm.sqlite3")
    registry.create_campaign(
        "cmp_semantic_test",
        name="Semantic checkpoint fixture",
    )
    registry.register_dataset(
        "synthetic",
        "v1",
        display_name="Temporary non-clinical semantic fixture",
        raw_fingerprint="dataset-fingerprint",
        processed_fingerprint="dataset-fingerprint",
        preprocessing_version="test-v1",
        verification_status="verified",
    )
    registry.register_split(
        "split_1",
        dataset_id="synthetic",
        dataset_version="v1",
        method="fixed",
        unit="sample",
        fingerprint="split-fingerprint",
        verification_status="verified",
    )
    return registry, paths


def _register_completed_checkpoint(
    registry: Registry,
    paths: ProjectPaths,
    *,
    run_id: str = "primary_semantic_run",
) -> Path:
    configuration = _configuration()
    variant = scientific_id(configuration)
    registry.register_variant(
        variant,
        campaign_id="cmp_semantic_test",
        configuration=configuration,
    )
    run_root = paths.artifact_root / "runs" / "2026" / "07" / run_id
    checkpoint = run_root / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    payload = b"temporary semantic checkpoint fixture"
    checkpoint.write_bytes(payload)
    registry.create_run(
        run_id,
        campaign_id="cmp_semantic_test",
        scientific_id=variant,
        repro_id="rep_semantic_fixture",
        seed=3,
        fold=2,
        attempt=1,
        configuration=configuration,
        status="completed",
        artifact_path=run_root,
        primary_metric_name="val/masked_huber",
        primary_metric_value=0.125,
    )
    registry.record_artifacts(
        run_id,
        [
            {
                "kind": "checkpoint_best",
                "path": checkpoint,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
    )
    return checkpoint


def _register_checkpoint_fixture(
    registry: Registry,
    paths: ProjectPaths,
    *,
    run_id: str,
    filename: str = "best.ckpt",
    kind: str = "checkpoint_best",
    payload: bytes | None = None,
    configuration: dict[str, object] | None = None,
    status: str = "completed",
    fold: int | None = None,
    attempt: int | None = None,
    primary_metric_name: str | None = None,
    primary_metric_value: float = 0.125,
    summary: dict[str, object] | None = None,
    as_symlink: bool = False,
) -> Path:
    resolved = deepcopy(configuration if configuration is not None else _configuration())
    if fold is not None:
        resolved["fold"] = fold
    if attempt is not None:
        resolved["attempt"] = attempt
    variant = scientific_id(resolved)
    registry.register_variant(
        variant,
        campaign_id="cmp_semantic_test",
        configuration=resolved,
    )

    run_root = paths.artifact_root / "runs" / "2026" / "07" / run_id
    checkpoint = run_root / "checkpoints" / filename
    checkpoint.parent.mkdir(parents=True)
    content = payload if payload is not None else f"checkpoint:{run_id}".encode()
    if as_symlink:
        source = checkpoint.with_name(f"source-{checkpoint.name}")
        source.write_bytes(content)
        checkpoint.symlink_to(source.name)
    else:
        checkpoint.write_bytes(content)
    if summary is not None:
        (run_root / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    evaluation = resolved.get("evaluation")
    inferred_metric = (
        str(evaluation.get("primary_metric"))
        if isinstance(evaluation, dict) and evaluation.get("primary_metric")
        else None
    )
    registry.create_run(
        run_id,
        campaign_id="cmp_semantic_test",
        scientific_id=variant,
        repro_id=f"rep_{run_id}",
        seed=int(resolved["seed"]),
        fold=int(resolved["fold"]),
        attempt=int(resolved["attempt"]),
        configuration=resolved,
        status=status,
        artifact_path=run_root,
        primary_metric_name=primary_metric_name or inferred_metric,
        primary_metric_value=primary_metric_value,
    )
    registry.record_artifacts(
        run_id,
        [
            {
                "kind": kind,
                "path": checkpoint,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        ],
    )
    return checkpoint


def _success_command() -> list[str]:
    code = (
        "import os,pathlib;"
        "p=pathlib.Path(os.environ['BAGM_RUN_SCRATCH']);"
        "(p/'checkpoints').mkdir(parents=True,exist_ok=True);"
        "(p/'checkpoints'/'best.ckpt').write_bytes(b'dummy-checkpoint');"
        "print('semantic dummy complete')"
    )
    return [sys.executable, "-c", code]


def _cli_payload(
    arguments: list[str],
    *,
    capsys: object,
) -> dict[str, object]:
    assert main(arguments) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    return json.loads(captured.out)


def test_run_semantics_uses_explicit_classification_or_visible_unknowns() -> None:
    explicit = _configuration(classified=True)
    missing = _configuration(classified=False)

    explicit_semantics = run_semantics_from_configuration(
        primary_run_id="future_explicit",
        configuration=explicit,
    )
    missing_semantics = run_semantics_from_configuration(
        primary_run_id="future_unclassified",
        configuration=missing,
    )

    assert explicit_semantics["lifecycle_stage"] == "exploratory_screen"
    assert explicit_semantics["study_axis"] == "graph_screen"
    assert explicit_semantics["source_batch"] == "future_graph_screen_v1"
    assert explicit_semantics["retention_class"] == (
        "retain_exploratory_evidence"
    )
    assert explicit_semantics["classification_confidence"] == "high"
    assert explicit_semantics["rules"]["classification_source"] == (
        "resolved_config"
    )
    assert explicit_semantics["semantic_alias"].startswith(
        "run.exploratory-screen."
    )
    assert ".graph-screen." in explicit_semantics["semantic_alias"]

    assert missing_semantics["lifecycle_stage"] == "unknown"
    assert missing_semantics["study_axis"] == "unclassified"
    assert missing_semantics["retention_class"] == "retain_pending_review"
    assert missing_semantics["classification_confidence"] == "unknown"
    assert missing_semantics["rules"]["classification_source"] == (
        "missing_explicit_classification"
    )
    assert missing_semantics["semantic_alias"].startswith(
        "run.unknown.unclassified."
    )


def test_classification_is_not_part_of_scientific_identity() -> None:
    explicit = _configuration(classified=True)
    changed = {
        **explicit,
        "classification": {
            "schema_version": 1,
            "lifecycle_stage": "locked_final",
            "study_axis": "locked_ladder",
            "retention_class": "retain_locked_final",
            "classification_confidence": "medium",
            "source_batch": "different_browsing_bucket",
        },
    }
    absent = _configuration(classified=False)

    assert scientific_id(explicit) == scientific_id(changed)
    assert scientific_id(explicit) == scientific_id(absent)


def test_checkpoint_index_backfills_metadata_and_is_idempotent(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    checkpoint = _register_completed_checkpoint(registry, paths)

    first = index_checkpoint_catalog(registry, paths, verify=True)
    second = index_checkpoint_catalog(registry, paths, verify=True)

    assert first["indexed_runs"] == 1
    assert first["indexed_checkpoints"] == 1
    assert second["indexed_runs"] == 1
    assert second["indexed_checkpoints"] == 1
    with registry.connect() as connection:
        counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in (
                "run_categories",
                "run_aliases",
                "checkpoint_catalog",
            )
        }
    assert counts == {
        "run_categories": 1,
        "run_aliases": 1,
        "checkpoint_catalog": 1,
    }

    aliases = registry.get_run_aliases("primary_semantic_run")
    assert len(aliases) == 1
    alias = str(aliases[0]["alias_id"])
    assert alias.startswith("run.exploratory-screen.")
    assert ".graph-screen." in alias
    assert aliases[0]["preferred"] is True
    assert registry.resolve_run_id(alias) == "primary_semantic_run"

    category = registry.get_run_category(alias)
    assert category is not None
    assert category["lifecycle_stage"] == "exploratory_screen"
    assert category["study_axis"] == "graph_screen"
    assert category["classification_confidence"] == "high"

    catalog = registry.list_checkpoint_catalog(run_id=alias)
    assert len(catalog) == 1
    assert catalog[0]["role"] == "best"
    assert catalog[0]["path"] == str(checkpoint)
    assert catalog[0]["verification_status"] == "verified"
    assert catalog[0]["preferred_alias"] == alias

    shown = registry.show_run(alias)
    assert shown is not None
    assert shown["run_id"] == "primary_semantic_run"
    assert shown["category"]["study_axis"] == "graph_screen"
    assert shown["checkpoints"][0]["role"] == "best"
    assert shown["checkpoints"][0]["verification_status"] == "verified"


def test_checkpoint_cli_indexes_browses_resolves_and_exports(
    tmp_path: Path,
    capsys: object,
) -> None:
    registry, paths = _registry(tmp_path)
    checkpoint = _register_completed_checkpoint(registry, paths)
    common = [
        "--root",
        str(tmp_path),
        "--database",
        str(registry.path),
    ]

    indexed = _cli_payload(
        [*common, "index-checkpoints", "--run-id", "primary_semantic_run"],
        capsys=capsys,
    )
    assert indexed["indexed_checkpoints"] == 1
    alias = str(registry.get_run_aliases("primary_semantic_run")[0]["alias_id"])

    listed = _cli_payload(
        [
            *common,
            "list-checkpoints",
            "--stage",
            "exploratory_screen",
            "--study-axis",
            "graph_screen",
            "--edge-features",
            "disabled",
        ],
        capsys=capsys,
    )
    assert listed["count"] == 1
    assert listed["checkpoints"][0]["semantic_alias"] == alias

    shown = _cli_payload(
        [*common, "show-checkpoint", alias, "--no-verify"],
        capsys=capsys,
    )
    assert shown["run_id"] == "primary_semantic_run"
    assert shown["checkpoint_role"] == "best"

    resolved = _cli_payload(
        [*common, "resolve-checkpoint", alias],
        capsys=capsys,
    )
    assert resolved == {
        "path": str(checkpoint),
        "role": "best",
        "run_id": "primary_semantic_run",
        "verified": True,
    }

    export = tmp_path / "exports" / "semantic-catalog"
    exported = _cli_payload(
        [
            *common,
            "export-checkpoint-catalog",
            "--output",
            str(export),
            "--stage",
            "exploratory_screen",
        ],
        capsys=capsys,
    )
    assert exported["records"] == 1
    assert (export / "catalog.jsonl").is_file()
    assert (export / "catalog.csv").is_file()
    assert (export / "summary.json").is_file()
    record = json.loads((export / "catalog.jsonl").read_text().strip())
    assert record["semantic_alias"] == alias

    reindexed = _cli_payload(
        [*common, "index-checkpoints", "--run-id", alias],
        capsys=capsys,
    )
    assert reindexed["indexed_checkpoints"] == 1


def test_future_queue_run_registers_semantics_and_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    configuration = _configuration(classified=True)
    job = registry.enqueue(
        campaign_id="cmp_semantic_test",
        configuration=configuration,
        command=_success_command(),
        maximum_attempts=1,
        requested_gpu=None,
    )
    worker = QueueWorker(
        registry,
        settings=WorkerSettings(
            worker_id="semantic-test-worker",
            gpu=None,
            once=True,
            min_free_gb=0,
            heartbeat_seconds=0.05,
            allow_test_jobs=True,
        ),
        paths=paths,
    )

    assert worker.run() == 1
    completed = registry.get_job(str(job["job_id"]))
    assert completed is not None
    assert completed["status"] == "completed"
    run_id = str(completed["run_id"])
    shown = registry.show_run(run_id)
    assert shown is not None
    assert shown["status"] == "completed"
    assert shown["category"]["lifecycle_stage"] == "exploratory_screen"
    assert shown["category"]["study_axis"] == "graph_screen"
    assert len(shown["aliases"]) == 1
    alias = str(shown["aliases"][0]["alias_id"])
    assert alias.startswith("run.exploratory-screen.")
    assert ".graph-screen." in alias
    assert registry.resolve_run_id(alias) == run_id
    assert len(shown["checkpoints"]) == 1
    assert shown["checkpoints"][0]["role"] == "best"
    assert shown["checkpoints"][0]["verification_status"] == "verified"

    catalog = registry.list_checkpoint_catalog(run_id=alias)
    assert len(catalog) == 1
    assert catalog[0]["preferred_alias"] == alias
    assert Path(str(catalog[0]["path"])).is_file()


def test_registered_checkpoint_symlink_is_rejected_without_dereferencing(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    checkpoint = _register_checkpoint_fixture(
        registry,
        paths,
        run_id="symlink_checkpoint_run",
        as_symlink=True,
    )

    assert checkpoint.is_symlink()
    with pytest.raises(
        CheckpointVerificationError,
        match="may not be a symlink",
    ):
        show_checkpoint(
            registry,
            "symlink_checkpoint_run",
            paths,
            role="best",
            verify=True,
        )
    assert checkpoint.is_symlink()


def test_requested_best_does_not_fall_back_to_only_last_checkpoint(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    last = _register_checkpoint_fixture(
        registry,
        paths,
        run_id="last_only_run",
        filename="last.ckpt",
        kind="checkpoint_last",
    )

    with pytest.raises(CheckpointNotFoundError, match="no 'best' checkpoint"):
        show_checkpoint(registry, "last_only_run", paths, role="best")
    with pytest.raises(CheckpointNotFoundError, match="no 'best' checkpoint"):
        resolve_checkpoint(registry, "last_only_run", paths, role="best")
    assert show_checkpoint(
        registry,
        "last_only_run",
        paths,
        role="last",
    )["checkpoint_path"] == str(last)


def test_checkpoint_catalog_export_is_confined_below_export_root(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    _register_checkpoint_fixture(
        registry,
        paths,
        run_id="export_boundary_run",
    )

    forbidden = (
        paths.report_root / "checkpoint-catalog",
        paths.artifact_root / "runs" / "checkpoint-catalog",
        paths.export_root,
    )
    for destination in forbidden:
        with pytest.raises(
            CheckpointCatalogError,
            match="configured export root",
        ):
            export_checkpoint_catalog(registry, destination, paths)
        assert not destination.exists()


def test_checkpoint_catalog_export_refuses_dangling_destination_symlink(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    _register_checkpoint_fixture(
        registry,
        paths,
        run_id="export_dangling_link_run",
    )
    paths.export_root.mkdir(parents=True)
    destination = paths.export_root / "dangling-catalog"
    destination.symlink_to(paths.export_root / "missing-target")

    with pytest.raises(FileExistsError, match="symlink destination"):
        export_checkpoint_catalog(registry, destination, paths)
    assert destination.is_symlink()
    assert not (paths.export_root / "missing-target").exists()


def test_checkpoint_verification_status_upgrades_and_never_downgrades(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    run_id = "verification_monotonic_run"
    _register_checkpoint_fixture(registry, paths, run_id=run_id)

    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=run_id,
        verify=False,
    )
    assert registry.list_checkpoint_catalog(run_id=run_id)[0][
        "verification_status"
    ] == "declared"

    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=run_id,
        verify=True,
    )
    assert registry.list_checkpoint_catalog(run_id=run_id)[0][
        "verification_status"
    ] == "verified"

    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=run_id,
        verify=False,
    )
    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=run_id,
        verify=False,
        update=True,
    )
    assert registry.list_checkpoint_catalog(run_id=run_id)[0][
        "verification_status"
    ] == "verified"


def test_sequential_duplicate_indexing_reconciles_all_peers_idempotently(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    shared = b"identical checkpoint payload"
    first_run = "duplicate_first_run"
    second_run = "duplicate_second_run"
    _register_checkpoint_fixture(
        registry,
        paths,
        run_id=first_run,
        payload=shared,
    )
    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=first_run,
        verify=True,
    )
    assert registry.list_checkpoint_catalog(run_id=first_run)[0][
        "duplicate_count"
    ] == 1

    _register_checkpoint_fixture(
        registry,
        paths,
        run_id=second_run,
        payload=shared,
    )
    result = index_checkpoint_catalog(
        registry,
        paths,
        run_reference=second_run,
        verify=True,
    )
    assert result["duplicate_peer_runs_reconciled"] == 1

    stored = registry.list_checkpoint_catalog(limit=None)
    assert {row["run_id"] for row in stored} == {first_run, second_run}
    assert {row["duplicate_count"] for row in stored} == {2}
    assert len({row["duplicate_group"] for row in stored}) == 1

    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=second_run,
        verify=True,
    )
    index_checkpoint_catalog(registry, paths, verify=True)
    rerun = registry.list_checkpoint_catalog(limit=None)
    assert [(row["run_id"], row["duplicate_count"]) for row in rerun] == [
        (first_run, 2),
        (second_run, 2),
    ]


def test_checkpoints_directory_kind_rejects_non_model_extensions(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    run_id = "checkpoint_sidecar_run"
    sidecar = _register_checkpoint_fixture(
        registry,
        paths,
        run_id=run_id,
        filename="retention-metadata.json",
        kind="checkpoints",
    )

    assert build_checkpoint_catalog(registry, paths, run_ids=[run_id]) == []
    with pytest.raises(
        CheckpointNotFoundError,
        match="No checkpoint is registered",
    ):
        index_checkpoint_catalog(
            registry,
            paths,
            run_reference=run_id,
        )
    assert sidecar.is_file()
    assert registry.show_run(run_id)["artifacts"][0]["path"] == str(sidecar)


def test_canonical_summary_best_epoch_and_configured_direction_are_indexed(
    tmp_path: Path,
) -> None:
    registry, paths = _registry(tmp_path)
    configuration = _configuration()
    configuration["evaluation"] = {
        "primary_metric": "val/auroc",
        "primary_direction": "maximize",
    }
    run_id = "canonical_summary_run"
    _register_checkpoint_fixture(
        registry,
        paths,
        run_id=run_id,
        configuration=configuration,
        primary_metric_name="val/auroc",
        primary_metric_value=0.875,
        summary={
            "best_epoch": 17,
            "primary_metric_name": "val/auroc",
            "primary_metric_value": 0.875,
        },
    )

    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=run_id,
        verify=True,
    )
    checkpoint = registry.list_checkpoint_catalog(run_id=run_id)[0]
    assert checkpoint["best_epoch"] == 17
    assert checkpoint["monitored_metric"] == "val/auroc"
    assert checkpoint["monitored_mode"] == "max"
    assert checkpoint["monitored_value"] == pytest.approx(0.875)


@pytest.mark.parametrize("malformation", ["index", "manifest"])
def test_targeted_canonical_index_ignores_unrelated_malformed_legacy_metadata(
    tmp_path: Path,
    malformation: str,
) -> None:
    registry, paths = _registry(tmp_path)
    target_run = "targeted_canonical_run"
    _register_checkpoint_fixture(
        registry,
        paths,
        run_id=target_run,
        payload=b"canonical target",
    )
    legacy_run = f"unrelated_legacy_{malformation}"
    _register_checkpoint_fixture(
        registry,
        paths,
        run_id=legacy_run,
        filename="model.pt",
        kind="legacy_checkpoint",
        payload=b"unrelated legacy checkpoint",
    )

    legacy_root = paths.artifact_root / "legacy_runs" / legacy_run
    legacy_root.mkdir(parents=True)
    index_path = legacy_root / "index.jsonl"
    if malformation == "index":
        index_path.write_text("{invalid json\n", encoding="utf-8")
        expected = "Invalid JSON"
    else:
        manifest = legacy_root / "manifest.json"
        manifest.write_text("{invalid json\n", encoding="utf-8")
        index_path.write_text(
            json.dumps(
                {
                    "legacy_run_id": legacy_run,
                    "manifest_path": manifest.relative_to(
                        paths.project_root
                    ).as_posix(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        expected = "Cannot parse legacy run manifest"

    with pytest.raises(CheckpointCatalogError, match=expected):
        build_checkpoint_catalog(registry, paths)

    result = index_checkpoint_catalog(
        registry,
        paths,
        run_reference=target_run,
        verify=True,
    )
    assert result["requested_run_id"] == target_run
    assert result["indexed_checkpoints"] == 1
    assert registry.list_checkpoint_catalog(run_id=target_run)[0][
        "verification_status"
    ] == "verified"


def test_checkpoint_cli_filters_run_role_execution_status_and_retention(
    tmp_path: Path,
    capsys: object,
) -> None:
    registry, paths = _registry(tmp_path)
    target_run = "cli_filter_target"
    clone_run = "cli_filter_clone"
    _register_checkpoint_fixture(
        registry,
        paths,
        run_id=target_run,
        payload=b"target checkpoint",
    )
    _register_checkpoint_fixture(
        registry,
        paths,
        run_id=clone_run,
        payload=b"clone checkpoint",
    )
    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=target_run,
        verify=True,
    )
    index_checkpoint_catalog(
        registry,
        paths,
        run_reference=clone_run,
        verify=True,
    )
    alias = str(registry.get_run_aliases(target_run)[0]["alias_id"])
    common = [
        "--root",
        str(tmp_path),
        "--database",
        str(registry.path),
        "list-checkpoints",
    ]

    unfiltered = _cli_payload(common, capsys=capsys)
    assert unfiltered["count"] == 2
    selected = _cli_payload(
        [
            *common,
            "--run-id",
            alias,
            "--role",
            "best",
            "--fold",
            "2",
            "--attempt",
            "1",
            "--status",
            "completed",
            "--retention-tier",
            "retain_exploratory_evidence",
        ],
        capsys=capsys,
    )
    assert selected["count"] == 1
    assert selected["checkpoints"][0]["run_id"] == target_run

    mismatches = (
        ["--role", "last"],
        ["--fold", "99"],
        ["--attempt", "99"],
        ["--status", "failed", "--include-failed"],
        ["--retention-tier", "retain_locked_final"],
    )
    for arguments in mismatches:
        payload = _cli_payload([*common, *arguments], capsys=capsys)
        assert payload["count"] == 0
