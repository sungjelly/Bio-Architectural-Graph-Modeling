from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.result_catalog import (
    CATALOG_JSON_NAME,
    CATALOG_MARKDOWN_NAME,
    EVIDENCE_DIMENSIONS,
    RESULT_MANIFEST_NAME,
    ResultCatalogError,
    create_result_scaffold,
    sha256_file,
    validate_result_manifest,
    validate_result_tree,
    write_or_check_catalog,
)


def _paths(tmp_path: Path) -> ProjectPaths:
    return ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})


def _load_manifest(directory: Path) -> dict[str, object]:
    loaded = yaml.safe_load((directory / RESULT_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_manifest(directory: Path, payload: dict[str, object]) -> None:
    (directory / RESULT_MANIFEST_NAME).write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _draft(
    tmp_path: Path,
    *,
    result_id: str = "res_graph_gain",
    method_family: str = "relative_qkv",
) -> tuple[ProjectPaths, Path]:
    paths = _paths(tmp_path)
    directory = create_result_scaffold(
        result_id=result_id,
        title="Graph-specific predictive gain",
        experiment_type="predictive_benchmark",
        method_family=method_family,
        lifecycle_stage="validation_confirmation",
        study_axis="graph_specific_gain",
        campaign_ids=("cmp_example",),
        paths=paths,
        git_commit="a" * 40,
        timestamp_utc="2026-08-25T12:00:00Z",
    )
    return paths, directory


def _promote_verified(paths: ProjectPaths, directory: Path) -> None:
    payload = _load_manifest(directory)
    payload["status"] = "verified"
    payload["outcome"] = "negative"
    payload["method"] = {
        "name": "Relative-geometric QKV graph transformer",
        "model_family": "relative_geometry_qkv_graph_transformer",
        "graph_context": "Bidirectional radial-stratified spatial graph",
        "masking_or_perturbation": "Uniform per-cell partial-gene masking",
        "evaluation_design": "Prespecified four-seed held-in stability audit",
        "implementation_version": "relative_qkv_v1",
    }
    payload["conclusion"] = {
        "question": "Does the graph channel provide stable predictive gain?",
        "estimand": "Four-seed aggregate masked-expression gain over self control.",
        "observed_result": "The locked gain criterion was not met.",
        "strongest_alternative_explanation": "The audit may be underpowered.",
        "controls": ["Parameter-matched self-only control", "Seed-complete audit"],
        "remaining_uncertainty": "No independent patient test was performed.",
        "maximum_defensible_claim": "No stable held-in graph gain was established.",
    }
    payload["evidence"] = {
        dimension: {
            "status": "failed" if dimension == "predictive_gain" else "not_tested",
            "summary": (
                "The prespecified gain gate failed."
                if dimension == "predictive_gain"
                else "This evidence dimension was not tested."
            ),
        }
        for dimension in EVIDENCE_DIMENSIONS
    }
    source = paths.artifact_root / "runs/2026/08/r_example/manifest.yaml"
    source.parent.mkdir(parents=True)
    source.write_text("schema: synthetic\n", encoding="utf-8")
    curated = directory / "tables/summary.txt"
    curated.write_text("metric,value\ngain,-0.01\n", encoding="utf-8")
    payload["provenance"] = {
        "created_at_utc": "2026-08-25T12:00:00Z",
        "updated_at_utc": "2026-08-25T13:00:00Z",
        "git_commit": "b" * 40,
        "run_ids": ["r_example"],
        "expected_seeds": [0, 1],
        "included_seeds": [0, 1],
        "expected_folds": [0],
        "included_folds": [0],
        "failed_or_excluded_runs": [],
        "sources": [
            {
                "kind": "run_artifact",
                "root": "artifact",
                "path": "runs/2026/08/r_example/manifest.yaml",
                "sha256": sha256_file(source),
                "verification_status": "verified",
                "run_id": "r_example",
            }
        ],
    }
    payload["files"] = [
        {
            "path": "tables/summary.txt",
            "role": "numeric_source",
            "sha256": sha256_file(curated),
            "size_bytes": curated.stat().st_size,
        }
    ]
    _write_manifest(directory, payload)
    (directory / "README.md").write_text(
        "# Graph-specific predictive gain\n\n"
        "The prespecified gain criterion was not met. This is a negative result.\n",
        encoding="utf-8",
    )


def test_scaffold_uses_semantic_hierarchy_and_validates_as_draft(
    tmp_path: Path,
) -> None:
    paths, directory = _draft(tmp_path)

    assert directory == (
        paths.result_root
        / "predictive_benchmark"
        / "relative_qkv"
        / "res_graph_gain"
    )
    assert (directory / "figures").is_dir()
    assert (directory / "tables").is_dir()
    assert (directory / "attachments").is_dir()
    record = validate_result_manifest(directory / RESULT_MANIFEST_NAME, paths=paths)
    assert record.payload["status"] == "draft"
    assert record.payload["experiment_type"] == "predictive_benchmark"


def test_verified_record_checks_sources_and_curated_payloads(tmp_path: Path) -> None:
    paths, directory = _draft(tmp_path)
    _promote_verified(paths, directory)

    record = validate_result_manifest(
        directory / RESULT_MANIFEST_NAME,
        paths=paths,
        verify_payloads=True,
        verify_sources=True,
    )

    assert record.payload["outcome"] == "negative"
    assert record.payload["evidence"]["predictive_gain"]["status"] == "failed"


def test_verified_record_rejects_pending_evidence_and_todo_readme(
    tmp_path: Path,
) -> None:
    paths, directory = _draft(tmp_path)
    payload = _load_manifest(directory)
    payload["status"] = "verified"
    _write_manifest(directory, payload)

    with pytest.raises(ResultCatalogError, match="README still contains TODO"):
        validate_result_manifest(directory / RESULT_MANIFEST_NAME, paths=paths)


def test_manifest_path_must_match_record_classification(tmp_path: Path) -> None:
    paths, directory = _draft(tmp_path)
    payload = _load_manifest(directory)
    payload["method_family"] = "different_method"
    _write_manifest(directory, payload)

    with pytest.raises(ResultCatalogError, match="directory must match"):
        validate_result_manifest(directory / RESULT_MANIFEST_NAME, paths=paths)


def test_unsafe_payload_path_and_identifier_fields_fail_closed(tmp_path: Path) -> None:
    paths, directory = _draft(tmp_path)
    payload = _load_manifest(directory)
    payload["files"] = [
        {
            "path": "../escape.txt",
            "role": "attachment",
            "sha256": "a" * 64,
            "size_bytes": 1,
        }
    ]
    _write_manifest(directory, payload)
    with pytest.raises(ResultCatalogError, match="unsafe path component"):
        validate_result_manifest(directory / RESULT_MANIFEST_NAME, paths=paths)

    payload["files"] = []
    payload["patient_id"] = "restricted"
    _write_manifest(directory, payload)
    with pytest.raises(ResultCatalogError, match="prohibited identifier"):
        validate_result_manifest(directory / RESULT_MANIFEST_NAME, paths=paths)


def test_checksum_verification_detects_source_and_payload_drift(tmp_path: Path) -> None:
    paths, directory = _draft(tmp_path)
    _promote_verified(paths, directory)
    (directory / "tables/summary.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ResultCatalogError, match="size drifted"):
        validate_result_manifest(
            directory / RESULT_MANIFEST_NAME,
            paths=paths,
            verify_payloads=True,
        )

    source = paths.artifact_root / "runs/2026/08/r_example/manifest.yaml"
    source.write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(ResultCatalogError, match="source checksum drifted"):
        validate_result_manifest(
            directory / RESULT_MANIFEST_NAME,
            paths=paths,
            verify_sources=True,
        )


def test_duplicate_result_ids_are_rejected_across_method_families(
    tmp_path: Path,
) -> None:
    paths, _ = _draft(tmp_path)
    _draft(tmp_path, method_family="spatial_baseline")

    with pytest.raises(ResultCatalogError, match="Duplicate result_id"):
        validate_result_tree(paths=paths)


def test_catalog_generation_is_deterministic_and_check_detects_drift(
    tmp_path: Path,
) -> None:
    paths, directory = _draft(tmp_path)
    _promote_verified(paths, directory)

    first = write_or_check_catalog(paths=paths)
    first_json = (paths.result_root / CATALOG_JSON_NAME).read_bytes()
    first_markdown = (paths.result_root / CATALOG_MARKDOWN_NAME).read_bytes()
    second = write_or_check_catalog(paths=paths)

    assert first == second
    assert first_json == (paths.result_root / CATALOG_JSON_NAME).read_bytes()
    assert first_markdown == (paths.result_root / CATALOG_MARKDOWN_NAME).read_bytes()
    assert first["results"][0]["directory"] == (
        "predictive_benchmark/relative_qkv/res_graph_gain"
    )
    assert json.loads(first_json)["result_count"] == 1
    write_or_check_catalog(paths=paths, check=True)

    (paths.result_root / CATALOG_MARKDOWN_NAME).write_text("stale\n", encoding="utf-8")
    with pytest.raises(ResultCatalogError, match="missing or stale"):
        write_or_check_catalog(paths=paths, check=True)
