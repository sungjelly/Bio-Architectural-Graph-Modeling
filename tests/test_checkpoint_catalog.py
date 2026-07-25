from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pytest

from spatial_benchmark.checkpoint_catalog import (
    CheckpointVerificationError,
    build_checkpoint_catalog,
    checkpoint_catalog_summary,
    export_checkpoint_catalog,
    filter_checkpoint_records,
    resolve_checkpoint,
    show_checkpoint,
)
from spatial_benchmark.identifiers import scientific_id, semantic_run_alias
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DATABASE = PROJECT_ROOT / "state/tracking/bagm.sqlite3"
HISTORICAL_INDEX = (
    PROJECT_ROOT
    / "artifacts/legacy_runs/lr_spatial_benchmark_batch/index.jsonl"
)


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


def _config(*, seed: int = 3) -> dict[str, object]:
    return {
        "model": {
            "name": "g1",
            "family": "g1",
            "embedding_dim": 32,
            "graph_layers": 2,
        },
        "masking": {"type": "P+N+B", "rate": 0.2},
        "dataset": {
            "dataset_id": "dataset",
            "version": "v1",
            "split_id": "split",
        },
        "features": {"use_edge_features": False},
        "graph": {
            "graph_id": "k16_r75_mutual_rbf8",
            "neighbor_k": 16,
            "radius_um": 75.0,
            "symmetry": "mutual",
            "min_distance_um": 0.0,
        },
        "trainer": {"learning_rate": 0.001, "batch_size": 1},
        "evaluation": {
            "primary_metric": "val/masked_huber",
            "primary_direction": "minimize",
        },
        "campaign": {"campaign_id": "cmp_catalog"},
        "seed": seed,
        "fold": 2,
        "attempt": 1,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


@pytest.fixture
def catalog_fixture(tmp_path: Path) -> tuple[Registry, ProjectPaths, dict[str, Path]]:
    paths = _paths(tmp_path)
    registry = Registry(paths.state_root / "tracking/catalog.sqlite3")
    registry.create_campaign(
        "cmp_catalog",
        name="Checkpoint catalog fixture",
        config={
            "classification": {
                "stage": "locked_final",
                "study_axis": "locked_ladder",
            }
        },
    )

    future_config = _config()
    future_scientific_id = scientific_id(future_config)
    registry.register_variant(
        future_scientific_id,
        campaign_id="cmp_catalog",
        configuration=future_config,
    )
    future_run_id = "r_20260724T120000Z_12345678_s003_f02_a01_fixture"
    future_checkpoint = (
        paths.artifact_root
        / "runs/2026/07"
        / future_run_id
        / "checkpoints/best.ckpt"
    )
    future_checkpoint.parent.mkdir(parents=True)
    future_checkpoint.write_bytes(b"same-checkpoint-content")
    registry.create_run(
        future_run_id,
        campaign_id="cmp_catalog",
        scientific_id=future_scientific_id,
        repro_id="rep_future",
        seed=3,
        fold=2,
        attempt=1,
        configuration=future_config,
        status="completed",
        artifact_path=future_checkpoint.parents[1],
        start_time="2026-07-24T12:00:00Z",
        primary_metric_name="val/masked_huber",
        primary_metric_value=0.20,
    )
    future_artifact_id = registry.record_artifact(
        future_run_id,
        kind="checkpoint_best",
        path=future_checkpoint,
        sha256=_sha(future_checkpoint),
        size_bytes=future_checkpoint.stat().st_size,
    )
    future_alias = semantic_run_alias(
        primary_run_id=future_run_id,
        historical=False,
        categories={
            "lifecycle_stage": "locked_final",
            "study_axis": "locked_ladder",
            "model": "g1",
            "graph": "k16_r75_mutual_rbf8",
            "mask": "P+N+B",
            "edge_feature_state": "disabled",
            "embedding_dim": 32,
            "seed": 3,
            "fold": 2,
            "attempt": 1,
            "variant_evidence": {"dataset_id": "dataset", "split_id": "split"},
        },
    )
    registry.register_run_semantics(
        future_run_id,
        lifecycle_stage="locked_final",
        study_axis="locked_ladder",
        source_batch="new_locked_final",
        seed_known=True,
        fold_known=True,
        attempt_known=True,
        retention_class="retain_locked_final",
        category_key="locked-final/g1/g1-true",
        classification_confidence="high",
        timestamp_basis="worker_utc",
        model_key="g1",
        dataset_key="dataset-v1",
        masking_key="p-n-b",
        graph_key="k16-r75-mutual-rbf8",
        feature_key="edge-disabled",
        embedding_key="d32",
        semantic_alias=future_alias,
    )

    legacy_config = _config(seed=7)
    legacy_scientific_id = scientific_id(legacy_config)
    registry.register_variant(
        legacy_scientific_id,
        campaign_id="cmp_catalog",
        configuration=legacy_config,
    )
    legacy_run_id = "lr_g1_fixture"
    legacy_root = (
        paths.artifact_root
        / "legacy_runs/batch/original/final_locked_v1/runs/native-g1"
    )
    legacy_root.mkdir(parents=True)
    legacy_checkpoint = legacy_root / "model_state.pt"
    legacy_checkpoint.write_bytes(future_checkpoint.read_bytes())
    legacy_manifest = legacy_root / "manifest.json"
    manifest = {
        "format_version": 1,
        "artifact_kind": "normal_true_tissue_spatial_benchmark_run",
        "status": "complete",
        "run_id": "native-run-id",
        "model_name": "g1",
        "model_seed": 7,
        "sealed_test_opened": True,
        "config": {
            "model": {
                "name": "g1",
                "hidden_dim": 32,
                "graph_layers": 2,
            },
            "training": {"curriculum": "P+N+B", "restore_best": True},
            "run": {"model_seed": 7, "rewired": False},
        },
        "graph": {
            "graph_id": "k16_r75_mutual_rbf8",
            "edge_control": "none",
            "config": {"k": 16, "radius_um": 75.0, "symmetry": "mutual"},
        },
        "training": {
            "best_epoch": 4,
            "best_validation_loss": 0.19,
        },
        "standards_lock": {"condition": "g1_true"},
        "artifacts": {"checkpoint": "model_state.pt"},
        "timing": {
            "started_at": "2026-07-24T12:01:00Z",
            "completed_at": "2026-07-24T12:02:00Z",
        },
    }
    legacy_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    registry.create_run(
        legacy_run_id,
        campaign_id="cmp_catalog",
        scientific_id=legacy_scientific_id,
        repro_id="legacy_rep",
        seed=7,
        fold=0,
        attempt=1,
        configuration=legacy_config,
        status="completed",
        artifact_path=legacy_root,
        start_time="2026-07-24T12:01:00Z",
        primary_metric_name="val/masked_huber",
        primary_metric_value=0.19,
    )
    registry.record_artifact(
        legacy_run_id,
        kind="legacy_checkpoint",
        path=legacy_checkpoint,
        sha256=_sha(legacy_checkpoint),
        size_bytes=legacy_checkpoint.stat().st_size,
    )
    index_path = paths.artifact_root / "legacy_runs/batch/index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_record = {
        "legacy_run_id": legacy_run_id,
        "legacy_native_run_id": "native-run-id",
        "batch": "final_locked_v1",
        "manifest_path": _relative(legacy_manifest, paths.project_root),
        "checkpoint_files": [
            "final_locked_v1/runs/native-g1/model_state.pt"
        ],
        "model_family": "g1",
        "masking_type": "P+N+B",
        "dataset_id": "dataset",
        "dataset_version": "v1",
        "split_id": "split",
        "use_edge_features": False,
        "seed": 7,
        "fold": None,
        "attempt": None,
        "status": "complete",
        "graph": {
            "graph_id": "k16_r75_mutual_rbf8",
            "neighbor_k": 16,
            "radius_um": 75.0,
            "symmetry": "mutual",
            "min_distance_um": 0.0,
            "rewired": False,
            "edge_control": "none",
        },
        "hyperparameters": {
            "embedding_dim": 32,
            "edge_embedding_dim": None,
            "graph_layers": 2,
        },
        "inference_confidence": {
            "run_boundary": "high",
            "seed": "high",
            "fold": "unknown",
            "attempt": "unknown",
        },
    }
    index_path.write_text(json.dumps(index_record) + "\n", encoding="utf-8")

    return registry, paths, {
        "future_checkpoint": future_checkpoint,
        "future_run_id": Path(future_run_id),
        "future_alias": Path(future_alias),
        "future_artifact_id": Path(str(future_artifact_id)),
        "legacy_checkpoint": legacy_checkpoint,
        "legacy_run_id": Path(legacy_run_id),
    }


def test_catalog_enriches_exact_legacy_identity_and_preserves_unknown_dimensions(
    catalog_fixture: tuple[Registry, ProjectPaths, dict[str, Path]],
) -> None:
    registry, paths, values = catalog_fixture
    records = build_checkpoint_catalog(registry, paths)

    assert len(records) == 2
    legacy = next(
        item for item in records if item["run_id"] == str(values["legacy_run_id"])
    )
    assert legacy["native_run_id"] == "native-run-id"
    assert legacy["lifecycle_stage"] == "locked_final"
    assert legacy["study_axis"] == "locked_ladder"
    assert legacy["source_batch"] == "final_locked_v1"
    assert legacy["evidence_tier"] == "sealed_conclusion_bearing"
    assert legacy["interpretation_tier"] == (
        "conclusion_bearing_with_campaign_limits"
    )
    assert legacy["retention_tier"] == "retain_locked_final"
    assert legacy["retention_action"] == "retain"
    assert legacy["condition"] == "g1_true"
    assert legacy["model_name"] == "g1"
    assert legacy["masking_type"] == "P+N+B"
    assert legacy["graph_id"] == "k16_r75_mutual_rbf8"
    assert legacy["neighbor_k"] == 16
    assert legacy["radius_um"] == 75.0
    assert legacy["embedding_dim"] == 32
    assert legacy["edge_embedding_dim"] is None
    assert legacy["seed"] == 7
    assert legacy["seed_known"] is True
    # Registry placeholders are deliberately not presented as source evidence.
    assert legacy["fold"] is None
    assert legacy["fold_known"] is False
    assert legacy["attempt"] is None
    assert legacy["attempt_known"] is False
    assert ".fna.ana." in legacy["semantic_alias"]
    assert legacy["legacy_checkpoint_declared_in_index"] is True

    future = next(
        item for item in records if item["run_id"] == str(values["future_run_id"])
    )
    assert future["semantic_alias"] == str(values["future_alias"])
    assert future["fold"] == 2
    assert future["attempt"] == 1
    assert future["checkpoint_role"] == "best"


def test_exact_sha_duplicates_are_annotated_not_collapsed(
    catalog_fixture: tuple[Registry, ProjectPaths, dict[str, Path]],
) -> None:
    registry, paths, _ = catalog_fixture
    records = build_checkpoint_catalog(registry, paths)

    assert len(records) == 2
    assert {item["content_duplicate_count"] for item in records} == {2}
    assert len({item["content_duplicate_group"] for item in records}) == 1
    assert sum(item["duplicate_of_checkpoint_id"] is not None for item in records) == 1
    expected_redundant = len(b"same-checkpoint-content")
    assert {
        item["duplicate_group_redundant_bytes"] for item in records
    } == {expected_redundant}
    summary = checkpoint_catalog_summary(records)
    assert summary["exact_duplicate_group_count"] == 1
    assert summary["exact_duplicate_record_count"] == 2
    assert summary["exact_duplicate_redundant_bytes"] == expected_redundant


def test_filter_show_resolve_and_checksum_refusal(
    catalog_fixture: tuple[Registry, ProjectPaths, dict[str, Path]],
) -> None:
    registry, paths, values = catalog_fixture
    records = build_checkpoint_catalog(registry, paths)
    selected = filter_checkpoint_records(
        records,
        filters={"lifecycle_stage": "locked_final", "seed": 7},
    )
    assert [item["run_id"] for item in selected] == [
        str(values["legacy_run_id"])
    ]

    shown = show_checkpoint(
        registry, str(values["future_alias"]), paths, role="best"
    )
    assert shown["run_id"] == str(values["future_run_id"])
    assert resolve_checkpoint(
        registry, str(values["future_alias"]), paths
    ) == values["future_checkpoint"].resolve()

    values["future_checkpoint"].write_bytes(b"tampered-checkpoint")
    with pytest.raises(CheckpointVerificationError, match="mismatch"):
        resolve_checkpoint(registry, str(values["future_alias"]), paths)


def test_exclusive_export_and_relative_verified_symlink_view(
    catalog_fixture: tuple[Registry, ProjectPaths, dict[str, Path]],
) -> None:
    registry, paths, values = catalog_fixture
    output = paths.export_root / "runs/checkpoint-catalog-test"
    result = export_checkpoint_catalog(
        registry, output, paths, symlink_view=True
    )

    assert result["records"] == 2
    assert (output / "catalog.jsonl").is_file()
    assert (output / "catalog.csv").is_file()
    assert (output / "summary.json").is_file()
    assert (output / "README.md").is_file()
    links = sorted((output / "by-stage").rglob("*"))
    links = [path for path in links if path.is_symlink()]
    assert len(links) == 2
    assert all(not Path(os.readlink(path)).is_absolute() for path in links)
    assert {path.resolve() for path in links} == {
        values["future_checkpoint"].resolve(),
        values["legacy_checkpoint"].resolve(),
    }
    # Exporting never replaces the source with a link or changes its bytes.
    assert not values["future_checkpoint"].is_symlink()
    assert values["future_checkpoint"].read_bytes() == b"same-checkpoint-content"

    with pytest.raises(FileExistsError, match="overwrite"):
        export_checkpoint_catalog(registry, output, paths)


def test_catalog_works_when_optional_schema_v3_tables_are_absent(
    catalog_fixture: tuple[Registry, ProjectPaths, dict[str, Path]],
) -> None:
    registry, paths, _ = catalog_fixture
    with registry.transaction(immediate=True) as connection:
        connection.execute("DROP TABLE checkpoint_catalog")
        connection.execute("DROP TABLE run_aliases")
        connection.execute("DROP TABLE run_categories")

    records = build_checkpoint_catalog(registry, paths)
    assert len(records) == 2
    assert all(item["semantic_alias"] for item in records)
    legacy = next(item for item in records if item["historical"])
    assert legacy["fold"] is None
    assert legacy["attempt"] is None


class _ReadOnlyRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.path.resolve()}?mode=ro&immutable=1",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        return connection


@pytest.mark.skipif(
    not HISTORICAL_DATABASE.is_file() or not HISTORICAL_INDEX.is_file(),
    reason="local historical registry/artifacts are not available",
)
def test_local_historical_catalog_regression() -> None:
    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(PROJECT_ROOT)})
    records = build_checkpoint_catalog(  # type: ignore[arg-type]
        _ReadOnlyRegistry(HISTORICAL_DATABASE),
        paths,
    )

    assert len(records) == 158
    assert Counter(item["lifecycle_stage"] for item in records) == {
        "diagnostic": 9,
        "exploratory_screen": 96,
        "validation_confirmation": 3,
        "locked_final": 50,
    }
    assert all(item["fold"] is None for item in records)
    assert all(item["attempt"] is None for item in records)
    final_conditions = Counter(
        item["condition"]
        for item in records
        if item["lifecycle_stage"] == "locked_final"
    )
    assert final_conditions == {
        "b0": 5,
        "b0_parameter_matched": 5,
        "b1": 5,
        "broad_field": 5,
        "g1_rewired": 5,
        "g1_true": 5,
        "g2_distance_only": 5,
        "g2_permuted": 5,
        "g2_true": 5,
        "g2_zero": 5,
    }
    summary = checkpoint_catalog_summary(records)
    assert summary["total_checkpoint_bytes"] == 1_499_185_266
    assert summary["exact_duplicate_group_count"] == 9
    assert summary["exact_duplicate_record_count"] == 20
    assert summary["exact_duplicate_redundant_bytes"] == 84_324_336
