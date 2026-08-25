"""Command-line interface for BAGM experiment infrastructure."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
from typing import Any, Mapping, Sequence

import yaml

from .checkpoint_catalog import (
    CheckpointCatalogError,
    export_checkpoint_catalog,
    index_checkpoint_catalog,
    list_checkpoints,
    resolve_checkpoint,
    show_checkpoint,
)
from .configuration import (
    ConfigurationError,
    compose_config,
    load_yaml_mapping,
    validate_experiment_config,
)
from .identifiers import (
    canonical_sha256,
    create_campaign_id,
    scientific_id,
    scientific_payload,
)
from .paths import ProjectPaths
from .queueing import (
    ProjectWorkerLock,
    QueueWorker,
    WorkerLockError,
    WorkerSettings,
    command_for_config,
)
from .registry import (
    ARTIFACT_STATUS_DELETED_BY_RETENTION,
    ARTIFACT_STATUS_RETENTION_PENDING,
    QUEUE_STATUSES,
    Registry,
    RegistryError,
)
from .run_archive import RunValidationError, verify_run_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bagm",
        description="Local experiment registry and one-GPU queue for BAGM.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Project root override (defaults to BAGM_ROOT or repository discovery).",
    )
    parser.add_argument(
        "--database",
        type=Path,
        help="SQLite registry (default: state/tracking/bagm.sqlite3).",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    subparsers.add_parser("doctor", help="Initialize and validate local infrastructure.")

    dataset = subparsers.add_parser("register-dataset")
    dataset.add_argument("--dataset-id", required=True)
    dataset.add_argument("--version", required=True)
    dataset.add_argument("--display-name", required=True)
    dataset.add_argument("--protected-source-path", type=Path)
    dataset.add_argument("--raw-fingerprint")
    dataset.add_argument("--preprocessing-version")
    dataset.add_argument("--processed-fingerprint")
    dataset.add_argument("--sample-count", type=int)
    dataset.add_argument("--graph-count", type=int)
    dataset.add_argument("--node-feature-schema")
    dataset.add_argument("--edge-feature-schema")
    dataset.add_argument("--creation-date")
    dataset.add_argument("--status", default="available")
    dataset.add_argument("--verification-status", default="unverified")

    split = subparsers.add_parser("register-split")
    split.add_argument("--split-id", required=True)
    split.add_argument("--dataset-id", required=True)
    split.add_argument("--dataset-version", required=True)
    split.add_argument("--method", required=True)
    split.add_argument("--unit", required=True)
    split.add_argument("--seed", type=int)
    split.add_argument("--fold-count", type=int)
    split.add_argument("--stratification", action="append", default=[])
    split.add_argument("--fingerprint")
    split.add_argument("--protected-path", type=Path)
    split.add_argument("--verification-status", default="unverified")

    campaign = subparsers.add_parser("create-campaign")
    campaign.add_argument("--campaign-id")
    campaign.add_argument("--name", required=True)
    campaign.add_argument("--scientific-question")
    campaign.add_argument("--plan", type=Path)
    campaign.add_argument("--status", default="planned")

    show_campaign = subparsers.add_parser("show-campaign")
    show_campaign.add_argument("campaign_id")

    update_campaign = subparsers.add_parser("update-campaign")
    update_campaign.add_argument("--campaign-id", required=True)
    update_campaign.add_argument("--plan", required=True, type=Path)
    update_campaign.add_argument("--expected-config-sha256", required=True)
    update_campaign.add_argument("--reason", required=True)
    update_campaign.add_argument("--actor", default="bagm-cli")
    update_campaign.add_argument("--status")
    update_campaign.add_argument("--name")
    update_campaign.add_argument("--scientific-question")

    campaign_revisions = subparsers.add_parser("list-campaign-revisions")
    campaign_revisions.add_argument("campaign_id")
    campaign_revisions.add_argument("--limit", type=int, default=100)

    enqueue = subparsers.add_parser("enqueue-experiment")
    _enqueue_arguments(enqueue)

    sweep = subparsers.add_parser("enqueue-sweep")
    sweep.add_argument("--campaign-id", required=True)
    sweep.add_argument("--sweep", required=True, type=Path)
    sweep.add_argument("--priority", type=int, default=0)
    sweep.add_argument("--max-attempts", type=int, default=1)
    sweep.add_argument("--gpu", default="0")

    worker = subparsers.add_parser("worker")
    worker.add_argument(
        "--worker-id",
        default=f"{socket.gethostname()}-{os.getpid()}",
    )
    worker.add_argument("--gpu", default=os.environ.get("BAGM_GPU_DEVICE", "0"))
    worker.add_argument("--poll-seconds", type=float, default=5.0)
    worker.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=_environment_float("BAGM_QUEUE_HEARTBEAT_SECONDS", 30.0),
    )
    worker.add_argument("--stale-after-seconds", type=float, default=900.0)
    worker.add_argument(
        "--min-free-gb",
        type=float,
        default=_environment_float("BAGM_MIN_FREE_DISK_GB", 25.0),
    )
    worker.add_argument("--once", action="store_true")
    worker.add_argument("--auto-retry", action="store_true")
    worker.add_argument(
        "--parallel-gpu-workers",
        action="store_true",
        help=(
            "Use one advisory worker lock per explicit GPU instead of the "
            "project-global lock. Start at most one such worker per GPU."
        ),
    )
    worker.add_argument(
        "--allow-test-jobs",
        action="store_true",
        help="Permit explicitly marked CPU-only dummy jobs; never use for science.",
    )
    worker.add_argument(
        "--cpu",
        action="store_true",
        help="Set CUDA_VISIBLE_DEVICES empty (intended for smoke tests only).",
    )

    queue = subparsers.add_parser("list-queue")
    queue.add_argument(
        "--status", action="append", choices=sorted(QUEUE_STATUSES), default=[]
    )
    queue.add_argument("--limit", type=int, default=100)
    queue.add_argument("--verbose", action="store_true")

    show = subparsers.add_parser("show-run")
    show.add_argument("run_id")

    index_checkpoints = subparsers.add_parser(
        "index-checkpoints",
        help="Persist semantic aliases/categories for registered checkpoints.",
    )
    index_checkpoints.add_argument("--run-id")
    index_checkpoints.add_argument("--no-verify", action="store_true")
    index_checkpoints.add_argument(
        "--update",
        action="store_true",
        help="Explicitly replace conflicting derived catalog metadata.",
    )

    checkpoint_list = subparsers.add_parser(
        "list-checkpoints",
        help="Browse checkpoints by scientific category instead of archive date.",
    )
    _checkpoint_filter_arguments(checkpoint_list)
    checkpoint_list.add_argument("--limit", type=int, default=100)

    checkpoint_show = subparsers.add_parser("show-checkpoint")
    checkpoint_show.add_argument("run_id")
    checkpoint_show.add_argument("--role", default="best")
    checkpoint_show.add_argument("--no-verify", action="store_true")

    checkpoint_resolve = subparsers.add_parser("resolve-checkpoint")
    checkpoint_resolve.add_argument("run_id")
    checkpoint_resolve.add_argument("--role", default="best")

    checkpoint_export = subparsers.add_parser("export-checkpoint-catalog")
    checkpoint_export.add_argument("--output", required=True, type=Path)
    checkpoint_export.add_argument("--links", action="store_true")
    _checkpoint_filter_arguments(checkpoint_export)

    embedding_analysis = subparsers.add_parser(
        "analyze-embedding-clusters",
        help=(
            "Extract intrinsic/contextual node embeddings and run joint "
            "six-core Leiden clustering."
        ),
    )
    embedding_analysis.add_argument("--run-id")
    embedding_analysis.add_argument("--checkpoint", type=Path)
    embedding_analysis.add_argument("--n-neighbors", type=int, default=30)
    embedding_analysis.add_argument(
        "--leiden-resolution", type=float, default=1.0
    )
    embedding_analysis.add_argument("--pca-components", type=int, default=50)
    embedding_analysis.add_argument(
        "--random-seed", type=int, default=20260825
    )
    embedding_analysis.add_argument("--device", default="cuda:0")
    embedding_analysis.add_argument("--output-dir", type=Path)

    summarize = subparsers.add_parser("summarize-variants")
    summarize.add_argument("--campaign-id")

    leaderboard = subparsers.add_parser("export-leaderboard")
    leaderboard.add_argument("--output", required=True, type=Path)
    leaderboard.add_argument("--campaign-id")

    promote = subparsers.add_parser("promote-run")
    promote.add_argument("run_id")

    verify = subparsers.add_parser("verify-artifacts")
    verify.add_argument("--run-id")

    legacy = subparsers.add_parser("import-legacy")
    legacy.add_argument("--path", required=True, type=Path)
    legacy.add_argument("--campaign-id", default="cmp_legacy_import")
    return parser


def _environment_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}.") from error


def _enqueue_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--priority", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--command",
        nargs=argparse.REMAINDER,
        help=(
            "Explicit argument vector. Supports {run_id}, {run_scratch}, "
            "{artifact_dir}, and {project_root}; no shell is used."
        ),
    )


def _checkpoint_filter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id")
    parser.add_argument("--campaign-id")
    parser.add_argument("--stage", dest="lifecycle_stage")
    parser.add_argument("--study-axis")
    parser.add_argument("--source-batch")
    parser.add_argument("--condition")
    parser.add_argument("--model", dest="model_name")
    parser.add_argument("--masking", dest="masking_type")
    parser.add_argument("--dataset-id")
    parser.add_argument("--scientific-id")
    parser.add_argument("--graph-id")
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--role", dest="checkpoint_role")
    parser.add_argument("--status")
    parser.add_argument("--retention-tier")
    parser.add_argument(
        "--edge-features",
        choices=("enabled", "disabled"),
    )
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--promoted-only", action="store_true")
    parser.add_argument("--duplicates-only", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        paths = _paths(arguments.root)
        database = arguments.database
        if database is None:
            database = paths.state_root / "tracking" / "bagm.sqlite3"
        elif not database.is_absolute():
            database = paths.project_root / database
        registry = Registry(database)
        result = _dispatch(arguments, registry=registry, paths=paths)
        if result is not None:
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        if arguments.command_name == "doctor" and isinstance(result, Mapping):
            return 0 if result.get("ok") is True else 1
        if arguments.command_name == "verify-artifacts" and isinstance(
            result, Mapping
        ):
            return 0 if result.get("valid") is True else 1
        return 0
    except (
        CheckpointCatalogError,
        ConfigurationError,
        FileExistsError,
        FileNotFoundError,
        RegistryError,
        RunValidationError,
        ValueError,
        WorkerLockError,
    ) as error:
        print(f"bagm: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


def _paths(root: Path | None) -> ProjectPaths:
    environment: dict[str, str] = {}
    if root is not None:
        environment["BAGM_ROOT"] = str(root.resolve(strict=False))
    else:
        import os

        environment.update(os.environ)
    return ProjectPaths.from_environment(environment)


def _dispatch(
    arguments: argparse.Namespace,
    *,
    registry: Registry,
    paths: ProjectPaths,
) -> Any:
    command = arguments.command_name
    if command == "doctor":
        return _doctor(registry, paths)
    if command == "register-dataset":
        record = registry.register_dataset(
            arguments.dataset_id,
            arguments.version,
            display_name=arguments.display_name,
            protected_source_path=arguments.protected_source_path,
            raw_fingerprint=arguments.raw_fingerprint,
            preprocessing_version=arguments.preprocessing_version,
            processed_fingerprint=arguments.processed_fingerprint,
            aggregate_sample_count=arguments.sample_count,
            graph_count=arguments.graph_count,
            node_feature_schema=arguments.node_feature_schema,
            edge_feature_schema=arguments.edge_feature_schema,
            creation_date=arguments.creation_date,
            status=arguments.status,
            verification_status=arguments.verification_status,
        )
        return _select(record, "dataset_id", "dataset_version", "status", "verification_status")
    if command == "register-split":
        record = registry.register_split(
            arguments.split_id,
            dataset_id=arguments.dataset_id,
            dataset_version=arguments.dataset_version,
            method=arguments.method,
            unit=arguments.unit,
            seed=arguments.seed,
            fold_count=arguments.fold_count,
            stratification=arguments.stratification,
            fingerprint=arguments.fingerprint,
            protected_path=arguments.protected_path,
            verification_status=arguments.verification_status,
        )
        return _select(record, "split_id", "dataset_id", "dataset_version", "verification_status")
    if command == "create-campaign":
        plan = load_yaml_mapping(arguments.plan) if arguments.plan else {}
        campaign_id = arguments.campaign_id or create_campaign_id(arguments.name)
        record = registry.create_campaign(
            campaign_id,
            name=arguments.name,
            scientific_question=arguments.scientific_question,
            config=plan,
            status=arguments.status,
        )
        return _select(record, "campaign_id", "name", "status")
    if command == "show-campaign":
        record = registry.get_campaign(arguments.campaign_id)
        if record is None:
            raise RegistryError(
                f"Campaign {arguments.campaign_id!r} does not exist."
            )
        result = _select(
            record,
            "campaign_id",
            "name",
            "scientific_question",
            "status",
            "config",
            "created_at",
            "updated_at",
        )
        result["config_sha256"] = canonical_sha256(record["config"])
        return result
    if command == "update-campaign":
        plan = load_yaml_mapping(arguments.plan)
        record = registry.update_campaign(
            arguments.campaign_id,
            config=plan,
            expected_config_sha256=arguments.expected_config_sha256,
            reason=arguments.reason,
            actor=arguments.actor,
            status=arguments.status,
            name=arguments.name,
            scientific_question=arguments.scientific_question,
        )
        return _select(
            record,
            "campaign_id",
            "name",
            "scientific_question",
            "status",
            "changed",
            "revision_id",
            "previous_config_sha256",
            "config_sha256",
            "updated_at",
        )
    if command == "list-campaign-revisions":
        records = registry.list_campaign_revisions(
            arguments.campaign_id, limit=arguments.limit
        )
        return [
            _select(
                record,
                "revision_id",
                "campaign_id",
                "previous_config_sha256",
                "new_config_sha256",
                "previous_name",
                "new_name",
                "previous_scientific_question",
                "new_scientific_question",
                "previous_status",
                "new_status",
                "reason",
                "actor",
                "created_at",
            )
            for record in records
        ]
    if command == "enqueue-experiment":
        config = _load_experiment(arguments.config, paths)
        _campaign_matches(config, arguments.campaign_id)
        _validate_registered_references(config, registry=registry, paths=paths)
        metadata = config.get("metadata", {})
        if arguments.command and not (
            isinstance(metadata, Mapping)
            and bool(metadata.get("test_only_dummy"))
        ):
            raise ConfigurationError(
                "--command is restricted to test_only_dummy fixtures; "
                "scientific argv is derived from the resolved configuration."
            )
        command_vector = (
            list(arguments.command)
            if arguments.command
            else command_for_config(config, paths=paths)
        )
        registry.register_variant(
            scientific_id(config),
            campaign_id=arguments.campaign_id,
            configuration=config,
        )
        record = registry.enqueue(
            campaign_id=arguments.campaign_id,
            configuration=config,
            command=command_vector,
            experiment_config_reference=arguments.config,
            priority=arguments.priority,
            maximum_attempts=arguments.max_attempts,
            requested_gpu=arguments.gpu,
        )
        return _select(record, "job_id", "campaign_id", "status", "attempt_count", "maximum_attempts")
    if command == "enqueue-sweep":
        return _enqueue_sweep(arguments, registry=registry, paths=paths)
    if command == "worker":
        settings = WorkerSettings(
            worker_id=arguments.worker_id,
            gpu=None if arguments.cpu else arguments.gpu,
            poll_seconds=arguments.poll_seconds,
            heartbeat_seconds=arguments.heartbeat_seconds,
            stale_after_seconds=arguments.stale_after_seconds,
            min_free_gb=arguments.min_free_gb,
            once=arguments.once,
            auto_retry=arguments.auto_retry,
            allow_test_jobs=arguments.allow_test_jobs,
            parallel_gpu_workers=arguments.parallel_gpu_workers,
        )
        processed = QueueWorker(
            registry, settings=settings, paths=paths
        ).run()
        return {"processed_jobs": processed, "worker_id": settings.worker_id}
    if command == "list-queue":
        rows = registry.list_queue(
            statuses=arguments.status or None, limit=arguments.limit
        )
        if not arguments.verbose:
            fields = (
                "job_id",
                "campaign_id",
                "priority",
                "status",
                "attempt_count",
                "maximum_attempts",
                "created_at",
                "run_id",
                "worker_id",
                "failure_category",
                "retry_of",
            )
            rows = [_select(row, *fields) for row in rows]
        return {"jobs": rows}
    if command == "show-run":
        record = registry.show_run(arguments.run_id)
        if record is None:
            raise RegistryError(f"Unknown run: {arguments.run_id}")
        return record
    if command == "index-checkpoints":
        return index_checkpoint_catalog(
            registry,
            paths,
            run_reference=arguments.run_id,
            verify=not arguments.no_verify,
            update=arguments.update,
        )
    if command == "list-checkpoints":
        if arguments.limit < 1:
            raise ValueError("--limit must be positive.")
        rows = list_checkpoints(
            registry,
            paths,
            filters=_checkpoint_filters(arguments, registry=registry),
            include_failed=arguments.include_failed,
            promoted_only=arguments.promoted_only,
            duplicates_only=arguments.duplicates_only,
        )
        return {
            "count": len(rows),
            "returned": min(len(rows), arguments.limit),
            "truncated": len(rows) > arguments.limit,
            "checkpoints": rows[: arguments.limit],
        }
    if command == "show-checkpoint":
        return show_checkpoint(
            registry,
            arguments.run_id,
            paths,
            role=arguments.role,
            verify=not arguments.no_verify,
        )
    if command == "resolve-checkpoint":
        run_id = registry.resolve_run_id(arguments.run_id)
        checkpoint_path = resolve_checkpoint(
            registry,
            run_id,
            paths,
            role=arguments.role,
        )
        return {
            "run_id": run_id,
            "role": arguments.role,
            "path": str(checkpoint_path),
            "verified": True,
        }
    if command == "export-checkpoint-catalog":
        return export_checkpoint_catalog(
            registry,
            arguments.output,
            paths,
            filters=_checkpoint_filters(arguments, registry=registry),
            include_failed=arguments.include_failed,
            promoted_only=arguments.promoted_only,
            duplicates_only=arguments.duplicates_only,
            symlink_view=arguments.links,
        )
    if command == "analyze-embedding-clusters":
        from spatial_benchmark.relative_qkv_embedding_clustering import (
            run_embedding_cluster_analysis,
        )

        return run_embedding_cluster_analysis(
            registry=registry,
            paths=paths,
            run_id=arguments.run_id,
            checkpoint=arguments.checkpoint,
            n_neighbors=arguments.n_neighbors,
            leiden_resolution=arguments.leiden_resolution,
            pca_components=arguments.pca_components,
            random_seed=arguments.random_seed,
            device=arguments.device,
            output_dir=arguments.output_dir,
        )
    if command == "summarize-variants":
        return {
            "variants": registry.summarize_variants(
                campaign_id=arguments.campaign_id
            )
        }
    if command == "export-leaderboard":
        rows = registry.summarize_variants(campaign_id=arguments.campaign_id)
        _write_leaderboard(arguments.output, rows, paths)
        return {"output": str(_project_path(arguments.output, paths)), "rows": len(rows)}
    if command == "promote-run":
        record = registry.promote_run(registry.resolve_run_id(arguments.run_id))
        return _select(record, "run_id", "status", "promoted_at", "artifact_path")
    if command == "verify-artifacts":
        run_id = (
            registry.resolve_run_id(arguments.run_id)
            if arguments.run_id
            else None
        )
        issues = registry.verify_artifacts(run_id=run_id)
        bundle_issues = _verify_bundles(registry, run_id, paths)
        return {
            "valid": not issues and not bundle_issues,
            "registry_issues": issues,
            "bundle_issues": bundle_issues,
        }
    if command == "import-legacy":
        return _import_legacy(
            registry,
            path=_project_path(arguments.path, paths),
            campaign_id=arguments.campaign_id,
        )
    raise AssertionError(f"Unhandled command: {command}")


def _checkpoint_filters(
    arguments: argparse.Namespace,
    *,
    registry: Registry,
) -> dict[str, Any]:
    names = (
        "run_id",
        "campaign_id",
        "lifecycle_stage",
        "study_axis",
        "source_batch",
        "condition",
        "model_name",
        "masking_type",
        "dataset_id",
        "scientific_id",
        "graph_id",
        "embedding_dim",
        "seed",
        "fold",
        "attempt",
        "checkpoint_role",
        "status",
        "retention_tier",
    )
    filters = {
        name: getattr(arguments, name)
        for name in names
        if getattr(arguments, name, None) is not None
    }
    edge_features = getattr(arguments, "edge_features", None)
    if edge_features is not None:
        filters["use_edge_features"] = edge_features == "enabled"
    if "run_id" in filters:
        filters["run_id"] = registry.resolve_run_id(str(filters["run_id"]))
    return filters


def _doctor(registry: Registry, paths: ProjectPaths) -> dict[str, Any]:
    paths.validate()
    runtime = (
        paths.state_root,
        paths.state_root / "tracking",
        paths.state_root / "locks",
        paths.state_root / "logs",
        paths.scratch_root,
        paths.artifact_root,
        paths.export_root,
    )
    for path in runtime:
        path.mkdir(parents=True, exist_ok=True)
    lock_available = True
    try:
        with ProjectWorkerLock(paths.state_root / "locks" / "bagm-worker.lock"):
            pass
    except WorkerLockError:
        lock_available = False
    integrity = registry.integrity_check()
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    required_tables = {
        "campaigns",
        "campaign_revisions",
        "variants",
        "campaign_variants",
        "runs",
        "evaluations",
        "metrics",
        "artifacts",
        "datasets",
        "splits",
        "queue_jobs",
        "failures",
        "run_aliases",
        "run_categories",
        "checkpoint_catalog",
    }
    missing_tables = sorted(required_tables - registry.table_names())
    if missing_tables:
        issues.append({"kind": "missing_registry_tables", "tables": missing_tables})
    checkpoint_catalog_status = {
        "checkpoint_artifacts": 0,
        "indexed_checkpoints": 0,
        "categorized_runs": 0,
        "semantic_aliases": 0,
    }
    if not {
        "run_aliases",
        "run_categories",
        "checkpoint_catalog",
    }.intersection(missing_tables):
        with registry.connect() as connection:
            checkpoint_catalog_status = {
                "checkpoint_artifacts": int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM artifacts
                        WHERE instr(lower(kind), 'checkpoint') > 0
                        """
                    ).fetchone()[0]
                ),
                "indexed_checkpoints": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM checkpoint_catalog"
                    ).fetchone()[0]
                ),
                "categorized_runs": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM run_categories"
                    ).fetchone()[0]
                ),
                "semantic_aliases": int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM run_aliases
                        WHERE alias_type = 'semantic'
                        """
                    ).fetchone()[0]
                ),
            }
        if (
            checkpoint_catalog_status["checkpoint_artifacts"]
            != checkpoint_catalog_status["indexed_checkpoints"]
        ):
            issues.append(
                {
                    "kind": "checkpoint_catalog_incomplete",
                    **checkpoint_catalog_status,
                }
            )
    finalizing = registry.list_finalizing_runs()
    if finalizing:
        issues.append(
            {
                "kind": "runs_require_finalization_reconciliation",
                "run_ids": [str(item["run_id"]) for item in finalizing[:20]],
            }
        )
    for name, path in (
        ("project_root", paths.project_root),
        ("config_root", paths.config_root),
        ("data_root", paths.data_root),
        ("artifact_root", paths.artifact_root),
        ("state_root", paths.state_root),
        ("scratch_root", paths.scratch_root),
    ):
        if not path.is_dir():
            issues.append({"kind": "missing_path_root", "name": name, "path": str(path)})
    if not (paths.project_root / ".git").exists():
        issues.append({"kind": "missing_git_root", "path": str(paths.project_root)})

    config_results: list[dict[str, Any]] = []
    for relative in (
        "configs/base.yaml",
        "configs/experiment/edge_feature_ablation_g1.yaml",
        "configs/experiment/edge_feature_ablation_g2.yaml",
    ):
        candidate = paths.project_root / relative
        if not candidate.is_file():
            issues.append({"kind": "missing_example_config", "path": relative})
            continue
        try:
            compose_config(candidate, config_root=paths.config_root)
            config_results.append({"path": relative, "valid": True})
        except (ConfigurationError, OSError) as error:
            issues.append(
                {"kind": "invalid_example_config", "path": relative, "reason": str(error)}
            )

    missing_registered_paths: list[dict[str, str]] = []
    missing_artifact_paths: list[str] = []
    retention_state_issues: list[dict[str, str]] = []
    missing_canonical_markers: list[str] = []
    with registry.connect() as connection:
        for row in connection.execute(
            "SELECT dataset_id, dataset_version, protected_source_path FROM datasets"
        ):
            if not row["protected_source_path"]:
                continue
            source = Path(str(row["protected_source_path"]))
            if not source.is_absolute():
                source = paths.project_root / source
            if not source.exists():
                missing_registered_paths.append(
                    {
                        "kind": "dataset",
                        "id": f"{row['dataset_id']}/{row['dataset_version']}",
                        "path": str(source),
                    }
                )
        for row in connection.execute(
            "SELECT split_id, protected_path FROM splits"
        ):
            if not row["protected_path"]:
                continue
            source = Path(str(row["protected_path"]))
            if not source.is_absolute():
                source = paths.project_root / source
            if not source.exists():
                missing_registered_paths.append(
                    {"kind": "split", "id": str(row["split_id"]), "path": str(source)}
                )
        for row in connection.execute(
            "SELECT DISTINCT path, status FROM artifacts"
        ):
            artifact = Path(str(row["path"]))
            if not artifact.is_absolute():
                artifact = paths.project_root / artifact
            artifact_status = str(row["status"])
            if artifact_status == ARTIFACT_STATUS_DELETED_BY_RETENTION:
                if artifact.exists() or artifact.is_symlink():
                    retention_state_issues.append(
                        {
                            "path": str(artifact),
                            "issue": "retention_tombstone_path_exists",
                        }
                    )
                continue
            if artifact_status == ARTIFACT_STATUS_RETENTION_PENDING:
                retention_state_issues.append(
                    {
                        "path": str(artifact),
                        "issue": "retention_deletion_pending",
                    }
                )
                continue
            if not artifact.exists():
                missing_artifact_paths.append(str(artifact))
        for row in connection.execute(
            """
            SELECT r.run_id, r.status, r.artifact_path
            FROM runs r
            WHERE r.status IN ('completed', 'failed', 'pruned')
              AND NOT EXISTS (
                  SELECT 1 FROM artifacts a
                  WHERE a.run_id = r.run_id AND a.kind = 'legacy_manifest'
              )
            """
        ):
            marker = {
                "completed": "_SUCCESS",
                "failed": "_FAILED",
                "pruned": "_PRUNED",
            }[str(row["status"])]
            root = Path(str(row["artifact_path"]))
            if not root.is_absolute():
                root = paths.project_root / root
            if not (root / marker).is_file():
                missing_canonical_markers.append(str(row["run_id"]))
    if missing_registered_paths:
        issues.append({"kind": "missing_registered_paths", "items": missing_registered_paths})
    if missing_artifact_paths:
        issues.append(
            {
                "kind": "missing_artifact_paths",
                "count": len(missing_artifact_paths),
                "paths": missing_artifact_paths[:20],
            }
        )
    if retention_state_issues:
        issues.append(
            {
                "kind": "artifact_retention_state_issues",
                "count": len(retention_state_issues),
                "items": retention_state_issues[:20],
            }
        )
    if missing_canonical_markers:
        issues.append(
            {
                "kind": "missing_canonical_markers",
                "run_ids": missing_canonical_markers[:20],
            }
        )
    broken_links = [
        str(path.relative_to(paths.project_root))
        for path in paths.project_root.rglob("*")
        if path.is_symlink() and not path.exists()
    ]
    if broken_links:
        issues.append({"kind": "broken_symlinks", "paths": broken_links})

    minimum_free_gb = _environment_float("BAGM_MIN_FREE_DISK_GB", 25.0)
    free_gb = shutil.disk_usage(paths.scratch_root).free / (1024**3)
    if free_gb < minimum_free_gb:
        issues.append(
            {
                "kind": "insufficient_disk_space",
                "free_gb": round(free_gb, 3),
                "minimum_free_gb": minimum_free_gb,
            }
        )
    if not lock_available:
        warnings.append(
            {
                "kind": "worker_lock_held",
                "note": "Expected when the one-GPU worker is currently active.",
            }
        )
    process_records = sorted((paths.state_root / "pids").glob("*.json"))
    if process_records and lock_available:
        issues.append(
            {
                "kind": "orphan_process_records_require_worker_reconciliation",
                "paths": [str(path) for path in process_records[:20]],
            }
        )
    return {
        "ok": integrity == ["ok"] and not issues,
        "project_root": str(paths.project_root),
        "database": str(registry.path),
        "schema_version": registry.schema_version(),
        "integrity_check": integrity,
        "journal_mode": "wal",
        "busy_timeout_ms": registry.busy_timeout_ms,
        "worker_lock_available": lock_available,
        "free_disk_gb": round(free_gb, 3),
        "minimum_free_disk_gb": minimum_free_gb,
        "config_checks": config_results,
        "issues": issues,
        "warnings": warnings,
        "checkpoint_catalog": checkpoint_catalog_status,
        "queue_counts": {
            status: len(registry.list_queue(statuses=[status], limit=100_000))
            for status in sorted(QUEUE_STATUSES)
        },
    }


def _load_experiment(path: Path, paths: ProjectPaths) -> dict[str, Any]:
    source = _project_path(path, paths)
    return compose_config(source, config_root=paths.config_root, validate=True)


def _campaign_matches(configuration: Mapping[str, Any], campaign_id: str) -> None:
    campaign = configuration.get("campaign", configuration.get("campaign_id"))
    configured = (
        campaign.get("campaign_id") if isinstance(campaign, Mapping) else campaign
    )
    if configured and str(configured) != campaign_id:
        raise ConfigurationError(
            f"Configuration campaign {configured!r} does not match {campaign_id!r}."
        )


def _enqueue_sweep(
    arguments: argparse.Namespace,
    *,
    registry: Registry,
    paths: ProjectPaths,
) -> dict[str, Any]:
    sweep_path = _project_path(arguments.sweep, paths)
    sweep = load_yaml_mapping(sweep_path)
    references = sweep.get("experiment_configs")
    if not isinstance(references, list) or not references:
        base = sweep.get("base_config")
        references = [base] if isinstance(base, str) else []
    if not references:
        raise ConfigurationError(
            "Sweep requires experiment_configs or a base_config."
        )
    factors = sweep.get("factors", {})
    if not isinstance(factors, Mapping):
        raise ConfigurationError("Sweep factors must be a mapping.")
    names = list(factors)
    values: list[list[Any]] = []
    for name in names:
        options = factors[name]
        if not isinstance(options, list) or not options:
            raise ConfigurationError(f"Sweep factor {name!r} must be a non-empty list.")
        values.append(options)
    planned: list[tuple[dict[str, Any], Path]] = []
    for reference in references:
        if not isinstance(reference, str):
            raise ConfigurationError("Sweep config references must be strings.")
        config_path = _project_path(Path(reference), paths)
        base_config = _load_experiment(config_path, paths)
        for combination in itertools.product(*values) if values else [()]:
            candidate = deepcopy(base_config)
            for name, value in zip(names, combination, strict=True):
                _set_config_value(candidate, str(name), value)
            candidate["campaign"] = _campaign_value(
                candidate.get("campaign"), arguments.campaign_id
            )
            validate_experiment_config(candidate)
            planned.append((candidate, config_path))
    jobs = []
    for config, config_path in planned:
        _validate_registered_references(config, registry=registry, paths=paths)
        scientific_identifier = scientific_id(config)
        registry.register_variant(
            scientific_identifier,
            campaign_id=arguments.campaign_id,
            configuration=config,
        )
        jobs.append(
            registry.enqueue(
                campaign_id=arguments.campaign_id,
                configuration=config,
                command=command_for_config(config, paths=paths),
                experiment_config_reference=config_path,
                priority=arguments.priority,
                maximum_attempts=arguments.max_attempts,
                requested_gpu=arguments.gpu,
            )
        )
    return {"enqueued": len(jobs), "job_ids": [job["job_id"] for job in jobs]}


def _set_config_value(config: dict[str, Any], dotted_name: str, value: Any) -> None:
    parts = dotted_name.split(".")
    target: dict[str, Any] = config
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            raise ConfigurationError(
                f"Sweep factor path does not resolve to a mapping: {dotted_name}"
            )
        target = nested
    if parts[-1] not in target:
        raise ConfigurationError(
            f"Sweep factor is not a declared config field: {dotted_name}"
        )
    target[parts[-1]] = value


def _validate_registered_references(
    config: Mapping[str, Any], *, registry: Registry, paths: ProjectPaths
) -> None:
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ConfigurationError("dataset must be a mapping.")
    dataset_id = str(dataset.get("dataset_id", ""))
    version = str(dataset.get("version", ""))
    split_id = str(dataset.get("split_id", ""))
    if registry.get_dataset(dataset_id, version) is None:
        raise ConfigurationError(
            f"Dataset is not registered: {dataset_id}/{version}."
        )
    split = registry.get_split(split_id)
    if split is None:
        raise ConfigurationError(f"Split is not registered: {split_id}.")
    if (
        split["dataset_id"] != dataset_id
        or split["dataset_version"] != version
    ):
        raise ConfigurationError(
            f"Split {split_id} does not belong to {dataset_id}/{version}."
        )
    reference = dataset.get("prepared_artifact_reference")
    if reference:
        prepared = Path(str(reference))
        if not prepared.is_absolute():
            prepared = paths.project_root / prepared
        if not prepared.exists():
            raise ConfigurationError(
                f"Prepared dataset artifact does not exist: {prepared}."
            )


def _campaign_value(existing: Any, campaign_id: str) -> Any:
    if isinstance(existing, Mapping):
        value = dict(existing)
        value["campaign_id"] = campaign_id
        return value
    return campaign_id


def _write_leaderboard(
    output: Path, rows: list[dict[str, Any]], paths: ProjectPaths
) -> None:
    target = _project_path(output, paths)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite leaderboard: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
    if target.suffix.lower() not in {".csv", ".tsv"}:
        raise ValueError("Leaderboard output must end in .json, .csv, or .tsv.")
    fields = sorted({key for row in rows for key in row})
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t" if target.suffix.lower() == ".tsv" else ",",
        )
        writer.writeheader()
        writer.writerows(rows)


def _verify_bundles(
    registry: Registry,
    run_id: str | None,
    paths: ProjectPaths,
) -> list[dict[str, Any]]:
    parameters: tuple[str, ...] = (run_id,) if run_id else ()
    with registry.connect() as connection:
        records = [
            dict(row)
            for row in connection.execute(
                """
                SELECT r.run_id, r.status, r.artifact_path,
                       EXISTS(
                           SELECT 1 FROM artifacts a
                           WHERE a.run_id = r.run_id
                             AND a.kind = 'legacy_manifest'
                       ) AS is_legacy_native
                FROM runs r
                WHERE r.status IN ('completed', 'failed', 'pruned')
                """
                + (" AND r.run_id = ?" if run_id else ""),
                parameters,
            ).fetchall()
        ]
        tombstone_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT a.run_id, a.path, a.sha256, a.size_bytes,
                       r.artifact_path
                FROM artifacts a
                JOIN runs r ON r.run_id = a.run_id
                WHERE a.status = ?
                """
                + (" AND a.run_id = ?" if run_id else "")
                + " ORDER BY a.run_id, a.artifact_id",
                (
                    (ARTIFACT_STATUS_DELETED_BY_RETENTION, run_id)
                    if run_id
                    else (ARTIFACT_STATUS_DELETED_BY_RETENTION,)
                ),
            ).fetchall()
        ]
    issues: list[dict[str, Any]] = []
    tombstones_by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for tombstone in tombstone_rows:
        tombstone_run_id = str(tombstone["run_id"])
        run_root = Path(str(tombstone["artifact_path"]))
        payload = Path(str(tombstone["path"]))
        if not run_root.is_absolute():
            run_root = paths.project_root / run_root
        if not payload.is_absolute():
            payload = paths.project_root / payload
        try:
            relative = payload.resolve(strict=False).relative_to(
                run_root.resolve(strict=False)
            ).as_posix()
        except ValueError:
            issues.append(
                {
                    "run_id": tombstone_run_id,
                    "issue": (
                        "Retention tombstone path is outside its run bundle: "
                        f"{payload}"
                    ),
                }
            )
            continue
        tombstones_by_run.setdefault(tombstone_run_id, {})[relative] = {
            "type": "file",
            "size": tombstone["size_bytes"],
            "sha256": tombstone["sha256"],
        }
    for record in records:
        if not record or not record.get("artifact_path"):
            continue
        if record.get("is_legacy_native"):
            # Registry.verify_artifacts validates the native files against the
            # audited legacy manifest; do not demand canonical BAGM markers.
            continue
        try:
            verify_run_bundle(
                record["artifact_path"],
                require_success_contract=record.get("status") == "completed",
                tombstoned_artifacts=tombstones_by_run.get(
                    str(record["run_id"]),
                    {},
                ),
            )
        except (FileNotFoundError, RunValidationError) as error:
            issues.append({"run_id": record["run_id"], "issue": str(error)})
    return issues


def _import_legacy(
    registry: Registry, *, path: Path, campaign_id: str
) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"Legacy import directory was not found: {path}")
    registry.create_campaign(
        campaign_id,
        name="Imported legacy results",
        scientific_question="Historical result preservation; metadata may be incomplete.",
        config={"source_kind": "legacy"},
        status="archived",
    )
    indexed: list[tuple[Path, dict[str, Any]]] = []
    index_paths = sorted(path.rglob("index.jsonl"), key=lambda item: item.as_posix())
    index_errors: list[dict[str, str]] = []
    for index_path in index_paths:
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, Mapping):
                        raise ValueError("index record is not a mapping")
                    manifest_reference = record.get("manifest_path")
                    legacy_run_id = record.get("legacy_run_id")
                    if not isinstance(manifest_reference, str) or not isinstance(
                        legacy_run_id, str
                    ):
                        raise ValueError(
                            "index record requires manifest_path and legacy_run_id"
                        )
                    indexed.append(
                        (
                            _resolve_index_reference(
                                manifest_reference,
                                index_path=index_path,
                                import_root=path,
                            ),
                            dict(record),
                        )
                    )
                except (ValueError, json.JSONDecodeError) as error:
                    index_errors.append(
                        {
                            "path": str(index_path),
                            "line": str(line_number),
                            "reason": str(error),
                        }
                    )
    if indexed:
        candidates = indexed
        source_mode = "audited_index"
    else:
        manifests = sorted(
            {
                *path.rglob("manifest.yaml"),
                *path.rglob("manifest.json"),
            },
            key=lambda item: item.as_posix(),
        )
        candidates = [(manifest, {}) for manifest in manifests]
        source_mode = "manifest_fallback"
    imported = []
    skipped: list[dict[str, str]] = list(index_errors)
    for manifest_path, index_record in candidates:
        try:
            payload = (
                yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.suffix in {".yaml", ".yml"}
                else json.loads(manifest_path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, yaml.YAMLError) as error:
            skipped.append({"path": str(manifest_path), "reason": str(error)})
            continue
        if not isinstance(payload, Mapping):
            skipped.append({"path": str(manifest_path), "reason": "not a mapping"})
            continue
        config = (
            _legacy_index_configuration(index_record)
            if index_record
            else payload.get("config", {})
        )
        if not isinstance(config, Mapping):
            config = {}
        scientific_identifier = scientific_id(dict(config))
        registry.register_variant(
            scientific_identifier,
            campaign_id=campaign_id,
            configuration=dict(config),
        )
        digest = __import__("hashlib").sha256(
            str(manifest_path.resolve(strict=False)).encode("utf-8")
        ).hexdigest()[:20]
        run_id = str(index_record.get("legacy_run_id") or f"legacy_{digest}")
        if registry.get_run(run_id) is not None:
            skipped.append({"path": str(manifest_path), "reason": "already imported"})
            continue
        seed = _integer(
            index_record.get(
                "seed", payload.get("seed", payload.get("model_seed"))
            ),
            0,
        )
        fold = _integer(index_record.get("fold", payload.get("fold")), 0)
        artifact_reference = index_record.get("archived_run_path")
        artifact_path = (
            _resolve_index_reference(
                str(artifact_reference),
                index_path=manifest_path,
                import_root=path,
            )
            if artifact_reference
            else manifest_path.parent
        )
        training = payload.get("training", {})
        if not isinstance(training, Mapping):
            training = {}
        timing = payload.get("timing", {})
        if not isinstance(timing, Mapping):
            timing = {}
        resources = payload.get("resources", {})
        if not isinstance(resources, Mapping):
            resources = {}
        provenance = payload.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        git = provenance.get("git", {})
        if not isinstance(git, Mapping):
            git = {}
        prepared = payload.get("prepared_artifact", {})
        if not isinstance(prepared, Mapping):
            prepared = {}
        dependencies = resources.get("dependencies", {})
        if not isinstance(dependencies, Mapping):
            dependencies = {}
        visible_devices = dependencies.get("visible_devices", [])
        gpu_names = [
            str(device.get("name"))
            for device in visible_devices
            if isinstance(device, Mapping) and device.get("name")
        ] if isinstance(visible_devices, list) else []
        primary_name = "val/masked_huber"
        primary_value = _finite_float(
            training.get("best_validation_loss", payload.get("primary_metric_value"))
        )
        peak_bytes = _finite_float(resources.get("peak_cuda_memory_bytes"))
        legacy_repro = {
            "scientific_configuration": scientific_payload(dict(config)),
            "git": {
                "commit": git.get("commit", "unknown"),
                "dirty": git.get("dirty"),
                "status_sha256": git.get("status_sha256"),
            },
            "data": {
                "dataset_id": index_record.get("dataset_id"),
                "dataset_version": index_record.get("dataset_version"),
                "prepared_manifest_sha256": prepared.get("manifest_sha256"),
                "split_id": index_record.get("split_id"),
                "preprocessing_version": "spatial_benchmark_preparation_v1",
            },
            "environment": dependencies,
        }
        registry.create_run(
            run_id,
            campaign_id=campaign_id,
            scientific_id=scientific_identifier,
            repro_id=f"legacy_rep_{canonical_sha256(legacy_repro)[:20]}",
            seed=seed,
            fold=fold,
            attempt=max(1, _integer(index_record.get("attempt"), 1)),
            configuration=dict(config),
            status=(
                "completed"
                if str(index_record.get("status", payload.get("status", ""))).lower()
                in {"complete", "completed", "success", "successful"}
                else "failed"
            ),
            artifact_path=artifact_path,
            model_family=index_record.get("model_family"),
            masking_type=index_record.get("masking_type"),
            dataset_id=index_record.get("dataset_id"),
            dataset_version=index_record.get("dataset_version"),
            split_id=index_record.get("split_id"),
            use_edge_features=index_record.get("use_edge_features"),
            git_commit=git.get("commit"),
            dirty_status=git.get("dirty"),
            preprocessing_version="spatial_benchmark_preparation_v1",
            dataset_fingerprint=prepared.get("manifest_sha256"),
            split_fingerprint=index_record.get("split_id"),
            start_time=timing.get("started_at"),
            end_time=timing.get("completed_at"),
            duration_seconds=_finite_float(timing.get("runtime_seconds")),
            gpu_model="; ".join(gpu_names) or None,
            peak_vram_gb=(peak_bytes / (1024**3) if peak_bytes is not None else None),
            primary_metric_name=primary_name if primary_value is not None else None,
            primary_metric_value=primary_value,
        )
        registry.record_artifact(
            run_id,
            kind="legacy_manifest",
            path=manifest_path,
            sha256=index_record.get("manifest_sha256"),
            size_bytes=manifest_path.stat().st_size,
        )
        if primary_value is not None:
            registry.record_metric(
                run_id,
                primary_name,
                primary_value,
                split="validation",
            )
        declared_files = payload.get("files", {})
        if not isinstance(declared_files, Mapping):
            declared_files = {}
        for field, kind, artifact_status in (
            ("checkpoint_files", "legacy_checkpoint", "present"),
            ("metric_files", "legacy_metrics", "present"),
            (
                "prediction_files",
                "legacy_predictions_restricted",
                "restricted_identifiers",
            ),
        ):
            references = index_record.get(field, [])
            if not isinstance(references, list):
                continue
            for reference in references:
                if not isinstance(reference, str):
                    continue
                artifact = _resolve_index_reference(
                    reference,
                    index_path=manifest_path,
                    import_root=path,
                )
                registry.record_artifact(
                    run_id,
                    kind=kind,
                    path=artifact,
                    sha256=declared_files.get(artifact.name),
                    size_bytes=(artifact.stat().st_size if artifact.is_file() else None),
                    status=(artifact_status if artifact.exists() else "missing"),
                )
        imported.append(run_id)
    return {
        "source_mode": source_mode,
        "index_files": [str(index_path) for index_path in index_paths],
        "imported": len(imported),
        "run_ids": imported,
        "skipped": skipped,
    }


def _resolve_index_reference(
    reference: str,
    *,
    index_path: Path,
    import_root: Path,
) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate
    search_roots = [
        Path.cwd(),
        import_root / "original",
        import_root,
        *import_root.parents,
        index_path.parent,
        *index_path.parents,
    ]
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved
    # Preserve a deterministic unresolved location for the ensuing clear
    # FileNotFoundError; never guess from basename alone.
    return Path.cwd() / candidate


def _legacy_index_configuration(record: Mapping[str, Any]) -> dict[str, Any]:
    hyperparameters = record.get("hyperparameters", {})
    graph = record.get("graph", {})
    return {
        "model": {
            "name": record.get("model_family", "unknown"),
            "family": record.get("model_family", "unknown"),
            "embedding_dim": (
                hyperparameters.get("embedding_dim")
                if isinstance(hyperparameters, Mapping)
                else None
            ),
        },
        "masking": {"type": record.get("masking_type", "unknown")},
        "dataset": {
            "dataset_id": record.get("dataset_id", "unknown"),
            "version": record.get("dataset_version", "unknown"),
            "split_id": record.get("split_id", "unknown"),
        },
        "features": {
            "use_edge_features": record.get("use_edge_features"),
        },
        "graph": dict(graph) if isinstance(graph, Mapping) else {},
        "trainer": (
            dict(hyperparameters) if isinstance(hyperparameters, Mapping) else {}
        ),
        "seed": _integer(record.get("seed"), 0),
        "fold": _integer(record.get("fold"), 0),
    }


def _integer(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _project_path(path: Path, paths: ProjectPaths) -> Path:
    return path if path.is_absolute() else paths.project_root / path


def _select(mapping: Mapping[str, Any], *fields: str) -> dict[str, Any]:
    return {field: mapping.get(field) for field in fields}


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
