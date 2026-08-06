#!/usr/bin/env python3
"""Audit and finalize an existing ten-core QKV campaign materialization.

This utility deliberately does not rebuild exact graphs, touch the experiment
registry, enqueue jobs, or start workers. It verifies the completed protected
artifacts, graph receipts, production configurations, and external cohort
selection; recomputes the deterministic self-aware eight-GPU assignment; then
atomically rewrites each affected config and the checksum-bound
materialization receipt.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from spatial_benchmark.adjacent_normal_selection import (  # noqa: E402
    load_adjacent_normal_route,
)
from spatial_benchmark.artifacts import load_prepared_artifact  # noqa: E402
from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.paths import current_paths  # noqa: E402


CAMPAIGN_ID = "cmp_20260728_adjacent_normal_10core_qkv_large_k"
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
ARMS = ("k1000", "k5000", "matched_self")
GRAPH_K_VALUES = (1000, 5000)
EXPECTED_SELECTION_SHA256 = (
    "a460de90ba9998e6817cdf3fbbcd2416eec048ec3fe6c825db8677b70b8b297f"
)
EXPECTED_ARTIFACT_KIND = (
    "adjacent_normal_tissue_spatial_benchmark_preparation"
)
RADIUS_GUARD_UM = 2000.0
GPU_COUNT = 8
MODEL_PARAMETER_COUNT = 36_749_480
PRODUCTION_EPOCHS = 300
SELF_NODE_WORK_FACTOR = 100.0
SELF_GRAPH_AUDIT_FRACTION = 0.01
FINALIZATION_SCHEMA_VERSION = 1


class MulticoreMaterializationFinalizationError(ValueError):
    """Raised when an existing campaign materialization cannot be trusted."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MulticoreMaterializationFinalizationError(
            f"{label} must be a mapping."
        )
    return value


def _strict_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MulticoreMaterializationFinalizationError(
            f"{label} must be a positive integer."
        )
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise MulticoreMaterializationFinalizationError(
            f"{label} must be finite."
        )
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MulticoreMaterializationFinalizationError(
            f"{label} must be finite."
        ) from exc
    if not math.isfinite(result):
        raise MulticoreMaterializationFinalizationError(
            f"{label} must be finite."
        )
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_json_constant(value: str) -> None:
    raise MulticoreMaterializationFinalizationError(
        "Campaign materialization contains a non-finite JSON constant."
    )


def _unique_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MulticoreMaterializationFinalizationError(
                "Campaign materialization contains a duplicate JSON key."
            )
        result[key] = value
    return result


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except MulticoreMaterializationFinalizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MulticoreMaterializationFinalizationError(
            "Campaign materialization is not readable strict JSON."
        ) from exc
    return dict(_mapping(value, "campaign materialization"))


def _verify_materialization_checksum(
    materialization: Mapping[str, Any],
) -> None:
    checksum = materialization.get("checksum")
    if not _is_sha256(checksum):
        raise MulticoreMaterializationFinalizationError(
            "Campaign materialization checksum is malformed."
        )
    payload = {
        key: value
        for key, value in materialization.items()
        if key != "checksum"
    }
    try:
        expected = canonical_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise MulticoreMaterializationFinalizationError(
            "Campaign materialization cannot be canonicalized."
        ) from exc
    if checksum != expected:
        raise MulticoreMaterializationFinalizationError(
            "Campaign materialization checksum does not verify."
        )


def _resolve_project_reference(
    project_root: Path,
    value: object,
    *,
    label: str,
    require_directory: bool = False,
    require_file: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise MulticoreMaterializationFinalizationError(
            f"{label} must be a nonempty project-relative path."
        )
    reference = Path(value)
    if reference.is_absolute():
        raise MulticoreMaterializationFinalizationError(
            f"{label} must remain project-relative."
        )
    unresolved = project_root / reference
    if unresolved.is_symlink():
        raise MulticoreMaterializationFinalizationError(
            f"{label} cannot be a symbolic link."
        )
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(project_root.resolve())
    except ValueError as exc:
        raise MulticoreMaterializationFinalizationError(
            f"{label} escapes the project root."
        ) from exc
    if require_directory and not candidate.is_dir():
        raise MulticoreMaterializationFinalizationError(
            f"{label} is not an available directory."
        )
    if require_file and not candidate.is_file():
        raise MulticoreMaterializationFinalizationError(
            f"{label} is not an available file."
        )
    return candidate


def _validate_external_selection(selection_manifest: Path) -> str:
    observed = sha256_file(selection_manifest)
    if observed != EXPECTED_SELECTION_SHA256:
        raise MulticoreMaterializationFinalizationError(
            "Protected cohort selection differs from its locked external SHA-256."
        )
    routes = tuple(
        load_adjacent_normal_route(selection_manifest, alias)
        for alias in ALIASES
    )
    if tuple(route.alias for route in routes) != ALIASES:
        raise MulticoreMaterializationFinalizationError(
            "Protected cohort selection does not contain the exact ten aliases."
        )
    if {
        slide: sum(route.slide == slide for route in routes)
        for slide in ("SO_1", "SO_2")
    } != {"SO_1": 5, "SO_2": 5}:
        raise MulticoreMaterializationFinalizationError(
            "Protected cohort selection is not balanced five per slide."
        )
    return observed


def _validate_artifact(
    *,
    project_root: Path,
    core_record: Mapping[str, Any],
    alias: str,
    selection_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    artifact = _resolve_project_reference(
        project_root,
        core_record.get("prepared_artifact"),
        label="prepared artifact reference",
        require_directory=True,
    )
    try:
        manifest, _, _ = load_prepared_artifact(
            artifact,
            load_arrays=False,
        )
    except Exception as exc:
        raise MulticoreMaterializationFinalizationError(
            "A prepared artifact failed checksum or schema verification."
        ) from exc
    selection = _mapping(
        manifest.get("selection"),
        "prepared artifact selection",
    )
    inputs = _mapping(
        manifest.get("inputs"),
        "prepared artifact inputs",
    )
    protected_input = _mapping(
        inputs.get("protected_selection_manifest"),
        "prepared artifact protected-selection input",
    )
    features = _mapping(
        manifest.get("features"),
        "prepared artifact features",
    )
    if (
        manifest.get("artifact_kind") != EXPECTED_ARTIFACT_KIND
        or selection.get("opaque_alias") != alias
        or selection.get("restricted_identifiers_emitted") is not False
        or selection.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or protected_input.get("sha256") != selection_sha256
    ):
        raise MulticoreMaterializationFinalizationError(
            "A prepared artifact disagrees with its alias, tissue context, "
            "privacy declaration, or protected-selection input SHA."
        )
    n_nodes = _strict_positive_int(
        core_record.get("n_nodes"),
        "materialized node count",
    )
    n_genes = _strict_positive_int(
        core_record.get("n_genes"),
        "materialized gene count",
    )
    if (
        selection.get("n_cells") != n_nodes
        or features.get("n_biological_probes") != n_genes
        or n_nodes <= max(GRAPH_K_VALUES)
        or n_genes != 1000
    ):
        raise MulticoreMaterializationFinalizationError(
            "A materialized core disagrees with its prepared artifact dimensions."
        )
    if not _is_sha256(core_record.get("preprocessing_sha256")):
        raise MulticoreMaterializationFinalizationError(
            "A materialized core preprocessing checksum is malformed."
        )
    return artifact, manifest


def _validate_graph_record(
    *,
    value: object,
    expected_k: int,
    n_nodes: int,
) -> dict[str, Any]:
    record = dict(_mapping(value, "materialized graph record"))
    if record.get("k") != expected_k:
        raise MulticoreMaterializationFinalizationError(
            "A materialized graph has the wrong literal k."
        )
    edge_count = _strict_positive_int(
        record.get("n_directed_edges"),
        "materialized directed-edge count",
    )
    if (
        edge_count % 2
        or edge_count > n_nodes * expected_k
    ):
        raise MulticoreMaterializationFinalizationError(
            "A materialized mutual graph has an impossible directed-edge count."
        )
    graph_sha256 = record.get("graph_sha256")
    if not _is_sha256(graph_sha256):
        raise MulticoreMaterializationFinalizationError(
            "A materialized graph checksum is malformed."
        )
    kth_distance = _finite_float(
        record.get("candidate_kth_distance_max_um"),
        "candidate kth-distance maximum",
    )
    margin = _finite_float(
        record.get("radius_guard_margin_um"),
        "radius-guard margin",
    )
    if (
        kth_distance <= 0
        or margin <= 0
        or not math.isclose(
            kth_distance + margin,
            RADIUS_GUARD_UM,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise MulticoreMaterializationFinalizationError(
            "A materialized graph does not certify a positive nontruncating "
            "2,000-um radius-guard margin."
        )
    mean_degree = _finite_float(
        record.get("mean_degree"),
        "materialized mean degree",
    )
    if not math.isclose(
        mean_degree,
        edge_count / n_nodes,
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        raise MulticoreMaterializationFinalizationError(
            "Materialized graph mean degree disagrees with its edge and node counts."
        )
    components = _strict_positive_int(
        record.get("n_components"),
        "materialized component count",
    )
    isolated = record.get("n_isolated_nodes")
    if (
        components > n_nodes
        or isinstance(isolated, bool)
        or not isinstance(isolated, int)
        or isolated != 0
    ):
        raise MulticoreMaterializationFinalizationError(
            "A materialized graph has invalid component or isolation QC."
        )
    return record


def _validate_core_records(
    *,
    materialization: Mapping[str, Any],
    project_root: Path,
    selection_sha256: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Path],
    dict[tuple[str, str], str],
]:
    raw_records = materialization.get("materialized_cores")
    if not isinstance(raw_records, list) or len(raw_records) != len(ALIASES):
        raise MulticoreMaterializationFinalizationError(
            "Materialization must contain exactly ten core records."
        )
    records: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Path] = {}
    config_references: dict[tuple[str, str], str] = {}
    graph_hashes: set[str] = set()
    for value in raw_records:
        record = dict(_mapping(value, "materialized core record"))
        alias = record.get("alias")
        if not isinstance(alias, str) or alias not in ALIASES or alias in records:
            raise MulticoreMaterializationFinalizationError(
                "Materialized core aliases are incomplete or duplicated."
            )
        artifact, _ = _validate_artifact(
            project_root=project_root,
            core_record=record,
            alias=alias,
            selection_sha256=selection_sha256,
        )
        n_nodes = int(record["n_nodes"])
        raw_graphs = _mapping(
            record.get("graphs"),
            "materialized core graphs",
        )
        if set(raw_graphs) != {"1000", "5000"}:
            raise MulticoreMaterializationFinalizationError(
                "Each core must contain exactly k=1,000 and k=5,000 graph records."
            )
        graphs = {
            k: _validate_graph_record(
                value=raw_graphs[str(k)],
                expected_k=k,
                n_nodes=n_nodes,
            )
            for k in GRAPH_K_VALUES
        }
        if (
            int(graphs[5000]["n_directed_edges"])
            <= int(graphs[1000]["n_directed_edges"])
            or graphs[5000]["graph_sha256"] == graphs[1000]["graph_sha256"]
        ):
            raise MulticoreMaterializationFinalizationError(
                "A core's k=5,000 graph is not distinct and denser than k=1,000."
            )
        for graph in graphs.values():
            graph_sha = str(graph["graph_sha256"])
            if graph_sha in graph_hashes:
                raise MulticoreMaterializationFinalizationError(
                    "Materialized graph identities are duplicated across core/k records."
                )
            graph_hashes.add(graph_sha)

        configs = _mapping(
            record.get("configs"),
            "materialized core config references",
        )
        if set(configs) != set(ARMS):
            raise MulticoreMaterializationFinalizationError(
                "Each core must reference exactly the three locked arm configs."
            )
        for arm in ARMS:
            reference = configs[arm]
            _resolve_project_reference(
                project_root,
                reference,
                label="production config reference",
                require_file=True,
            )
            if not isinstance(reference, str):
                raise MulticoreMaterializationFinalizationError(
                    "Production config reference must be a string."
                )
            config_references[(alias, arm)] = reference
        records[alias] = {**record, "graphs": graphs}
        artifacts[alias] = artifact
    if tuple(sorted(records)) != ALIASES:
        raise MulticoreMaterializationFinalizationError(
            "Materialized core records do not cover the locked ten aliases."
        )
    if len(set(artifacts.values())) != len(ALIASES):
        raise MulticoreMaterializationFinalizationError(
            "Prepared artifact references are not unique by alias."
        )
    if len(set(config_references.values())) != len(ALIASES) * len(ARMS):
        raise MulticoreMaterializationFinalizationError(
            "Production config references are not unique by alias and arm."
        )
    return records, artifacts, config_references


def _expected_job_work(
    *,
    core: Mapping[str, Any],
    arm: str,
) -> float:
    graphs = _mapping(core.get("graphs"), "materialized core graphs")
    if arm == "k1000":
        return float(
            _mapping(graphs[1000], "k1000 graph")[
                "n_directed_edges"
            ]
        )
    if arm == "k5000":
        return float(
            _mapping(graphs[5000], "k5000 graph")[
                "n_directed_edges"
            ]
        )
    return (
        float(core["n_nodes"]) * SELF_NODE_WORK_FACTOR
        + SELF_GRAPH_AUDIT_FRACTION
        * float(
            _mapping(graphs[5000], "k5000 graph")[
                "n_directed_edges"
            ]
        )
    )


def _expected_legacy_job_work(
    *,
    core: Mapping[str, Any],
    arm: str,
) -> float:
    """Return the checksum-bound pre-finalizer work rule."""

    if arm == "matched_self":
        return float(core["n_nodes"]) * SELF_NODE_WORK_FACTOR
    return _expected_job_work(core=core, arm=arm)


def _input_contract(materialization: Mapping[str, Any]) -> str:
    finalization = materialization.get("finalization")
    if finalization is None:
        return "legacy_unfinalized_v1"
    record = _mapping(finalization, "materialization finalization")
    if (
        record.get("schema_version") != FINALIZATION_SCHEMA_VERSION
        or record.get("utility")
        != "finalize_multicore_qkv_materialization"
        or record.get("graph_recomputation_performed") is not False
        or record.get("registry_or_queue_mutation_performed") is not False
    ):
        raise MulticoreMaterializationFinalizationError(
            "Materialization has an unknown or unsafe prior finalization record."
        )
    return "strict_finalized_v1"


def _validate_config(
    *,
    config: Mapping[str, Any],
    alias: str,
    arm: str,
    core: Mapping[str, Any],
    artifact_reference: str,
) -> None:
    try:
        validate_experiment_config(config)
    except Exception as exc:
        raise MulticoreMaterializationFinalizationError(
            "A production config fails the experiment schema."
        ) from exc
    campaign = _mapping(config.get("campaign"), "config campaign")
    experiment = _mapping(config.get("experiment"), "config experiment")
    dataset = _mapping(config.get("dataset"), "config dataset")
    model = _mapping(config.get("model"), "config model")
    graph = _mapping(config.get("graph"), "config graph")
    features = _mapping(config.get("features"), "config features")
    trainer = _mapping(config.get("trainer"), "config trainer")
    launcher = _mapping(config.get("launcher"), "config launcher")

    if (
        campaign.get("campaign_id") != CAMPAIGN_ID
        or experiment.get("biological_unit_alias") != alias
        or experiment.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or dataset.get("biological_unit_alias") != alias
        or dataset.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or dataset.get("prepared_artifact_reference")
        != artifact_reference
        or dataset.get("dataset_fingerprint")
        != core.get("preprocessing_sha256")
        or dataset.get("split_id") != core.get("split_id")
        or dataset.get("split_fingerprint")
        != core.get("split_fingerprint")
        or dataset.get("validation_or_test_partition_present") is not False
        or config.get("seed") != 0
        or config.get("fold") != 0
        or config.get("attempt") != 1
        or trainer.get("max_epochs") != PRODUCTION_EPOCHS
        or model.get("hidden_dim") != 576
        or model.get("graph_layers") != 8
        or model.get("attention_heads") != 9
        or launcher.get("requested_gpu_count") != 1
        or launcher.get("distributed") is not False
    ):
        raise MulticoreMaterializationFinalizationError(
            "A production config disagrees with its locked campaign/core contract."
        )

    expected_k = 1000 if arm == "k1000" else 5000
    expected_graph = _mapping(
        _mapping(core.get("graphs"), "materialized core graphs")[
            expected_k
        ],
        "expected graph",
    )
    if (
        graph.get("k") != expected_k
        or graph.get("neighbor_k") != expected_k
        or graph.get("radius_um") != RADIUS_GUARD_UM
        or graph.get("radius_guard_um") != RADIUS_GUARD_UM
        or graph.get("expected_directed_edges")
        != expected_graph.get("n_directed_edges")
        or graph.get("expected_materialized_graph_sha256")
        != expected_graph.get("graph_sha256")
    ):
        raise MulticoreMaterializationFinalizationError(
            "A production config's graph k/count/hash/guard differs from "
            "materialization."
        )
    if arm == "matched_self":
        if (
            model.get("name") != "qkv-gat-matched-self"
            or model.get("family") != "qkv_parameter_matched_self_control"
            or features.get("use_edge_features") is not False
        ):
            raise MulticoreMaterializationFinalizationError(
                "The matched-self production config is not the locked "
                "cell-autonomous control."
            )
    elif (
        model.get("name") != "qkv-gat"
        or model.get("family") != "edge_aware_qkv_graph_transformer"
        or features.get("use_edge_features") is not True
    ):
        raise MulticoreMaterializationFinalizationError(
            "A graph-arm production config is not the locked QKV model."
        )


def _validate_jobs_and_configs(
    *,
    materialization: Mapping[str, Any],
    project_root: Path,
    core_records: Mapping[str, Mapping[str, Any]],
    config_references: Mapping[tuple[str, str], str],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], Path],
    dict[tuple[str, str], dict[str, Any]],
    str,
]:
    input_contract = _input_contract(materialization)
    raw_jobs = materialization.get("jobs")
    if (
        materialization.get("job_count") != len(ALIASES) * len(ARMS)
        or not isinstance(raw_jobs, list)
        or len(raw_jobs) != len(ALIASES) * len(ARMS)
    ):
        raise MulticoreMaterializationFinalizationError(
            "Materialization must declare exactly 30 jobs."
        )
    expected_coverage = {
        (alias, arm) for alias in ALIASES for arm in ARMS
    }
    jobs: dict[tuple[str, str], dict[str, Any]] = {}
    config_paths: dict[tuple[str, str], Path] = {}
    configs: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_jobs:
        job = dict(_mapping(raw, "materialized job"))
        alias = job.get("alias")
        arm = job.get("arm")
        key = (alias, arm)
        if (
            not isinstance(alias, str)
            or not isinstance(arm, str)
            or key not in expected_coverage
            or key in jobs
        ):
            raise MulticoreMaterializationFinalizationError(
                "Materialized jobs do not provide unique alias/arm coverage."
            )
        expected_reference = config_references[(alias, arm)]
        if job.get("config") != expected_reference:
            raise MulticoreMaterializationFinalizationError(
                "A job config reference differs from its core arm reference."
            )
        config_path = _resolve_project_reference(
            project_root,
            expected_reference,
            label="production config reference",
            require_file=True,
        )
        config = load_yaml_mapping(config_path)
        _validate_config(
            config=config,
            alias=alias,
            arm=arm,
            core=core_records[alias],
            artifact_reference=str(
                core_records[alias]["prepared_artifact"]
            ),
        )
        old_digest = canonical_sha256(config)
        recorded_digest = job.get("config_sha256")
        if (
            input_contract == "strict_finalized_v1"
            and recorded_digest != old_digest
        ):
            raise MulticoreMaterializationFinalizationError(
                "A production config differs from its existing materialization digest."
            )
        if (
            input_contract == "legacy_unfinalized_v1"
            and recorded_digest is not None
        ):
            raise MulticoreMaterializationFinalizationError(
                "The unfinalized legacy receipt unexpectedly contains config digests."
            )
        requested_gpu = job.get("requested_gpu")
        launcher = _mapping(config.get("launcher"), "config launcher")
        if (
            isinstance(requested_gpu, bool)
            or not isinstance(requested_gpu, int)
            or not 0 <= requested_gpu < GPU_COUNT
        ):
            raise MulticoreMaterializationFinalizationError(
                "An existing job GPU assignment is invalid."
            )
        launcher_gpu = launcher.get("requested_gpu")
        if input_contract == "strict_finalized_v1":
            if launcher_gpu != str(requested_gpu):
                raise MulticoreMaterializationFinalizationError(
                    "A finalized job/config GPU assignment is inconsistent."
                )
            expected_work = _expected_job_work(
                core=core_records[alias],
                arm=arm,
            )
        else:
            if launcher_gpu != "0":
                raise MulticoreMaterializationFinalizationError(
                    "An unfinalized config differs from the audited legacy "
                    "launcher-GPU-zero state."
                )
            expected_work = _expected_legacy_job_work(
                core=core_records[alias],
                arm=arm,
            )
        observed_work = _finite_float(
            job.get("estimated_work"),
            "job estimated work",
        )
        if not math.isclose(
            observed_work,
            expected_work,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise MulticoreMaterializationFinalizationError(
                "A job's estimated work differs from its accepted locked formula."
            )
        jobs[key] = job
        config_paths[key] = config_path
        configs[key] = config
    if set(jobs) != expected_coverage:
        raise MulticoreMaterializationFinalizationError(
            "Materialized jobs do not cover all ten aliases and three arms."
        )
    if input_contract == "legacy_unfinalized_v1":
        expected_assignment, _ = _assign_gpus(jobs)
        expected_keys = [
            (str(job["alias"]), str(job["arm"]))
            for job in expected_assignment
        ]
        observed_keys = [
            (str(job["alias"]), str(job["arm"]))
            for job in raw_jobs
        ]
        expected_gpus = {
            (str(job["alias"]), str(job["arm"])): int(
                job["requested_gpu"]
            )
            for job in expected_assignment
        }
        if expected_keys != observed_keys or any(
            int(job["requested_gpu"])
            != expected_gpus[(str(job["alias"]), str(job["arm"]))]
            for job in raw_jobs
        ):
            raise MulticoreMaterializationFinalizationError(
                "The unfinalized receipt differs from its deterministic legacy "
                "eight-GPU assignment."
            )
    return jobs, config_paths, configs, input_contract


def _assign_gpus(
    jobs: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[float]]:
    loads = [0.0] * GPU_COUNT
    ordered = sorted(
        jobs.values(),
        key=lambda item: (
            -float(item["estimated_work"]),
            str(item["alias"]),
            str(item["arm"]),
        ),
    )
    assigned: list[dict[str, Any]] = []
    for job in ordered:
        gpu = min(
            range(GPU_COUNT),
            key=lambda index: (loads[index], index),
        )
        assigned.append({**dict(job), "requested_gpu": gpu})
        loads[gpu] += float(job["estimated_work"])
    return assigned, loads


def _serialize_yaml(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(payload),
        sort_keys=False,
        allow_unicode=True,
    )


def _stage_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".finalize.tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        os.chmod(temporary, path.stat().st_mode & 0o777)
    return temporary


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    temporary = _stage_text(path, serialized)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_configs_atomically(
    rewrites: Sequence[tuple[Path, Mapping[str, Any]]],
) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, payload in rewrites:
            staged.append(
                (destination, _stage_text(destination, _serialize_yaml(payload)))
            )
        for destination, temporary in staged:
            temporary.replace(destination)
    finally:
        for _, temporary in staged:
            if temporary.exists():
                temporary.unlink()


def finalize_materialization(
    *,
    materialization_path: Path,
    selection_manifest: Path,
    project_root: Path = _PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate and finalize a completed materialization without graph rebuilds."""

    project_root = project_root.resolve()
    materialization = _load_json_mapping(materialization_path)
    _verify_materialization_checksum(materialization)
    if (
        materialization.get("schema_version") != 1
        or materialization.get("campaign_id") != CAMPAIGN_ID
    ):
        raise MulticoreMaterializationFinalizationError(
            "Materialization is not the locked ten-core QKV campaign."
        )
    model = _mapping(materialization.get("model"), "materialized model")
    if (
        model.get("parameter_count") != MODEL_PARAMETER_COUNT
        or model.get("hidden_dim") != 576
        or model.get("graph_layers") != 8
        or model.get("attention_heads") != 9
        or model.get("attention_head_dim") != 64
        or model.get("fixed_epochs") != PRODUCTION_EPOCHS
    ):
        raise MulticoreMaterializationFinalizationError(
            "Materialized model differs from the locked 36,749,480-parameter "
            "QKV architecture."
        )

    selection_sha256 = _validate_external_selection(selection_manifest)
    core_records, _, config_references = _validate_core_records(
        materialization=materialization,
        project_root=project_root,
        selection_sha256=selection_sha256,
    )
    jobs, config_paths, configs, _ = _validate_jobs_and_configs(
        materialization=materialization,
        project_root=project_root,
        core_records=core_records,
        config_references=config_references,
    )
    assignment_jobs = {
        key: {
            **job,
            "estimated_work": _expected_job_work(
                core=core_records[key[0]],
                arm=key[1],
            ),
        }
        for key, job in jobs.items()
    }
    assigned_jobs, gpu_loads = _assign_gpus(assignment_jobs)

    rewritten_configs: dict[tuple[str, str], dict[str, Any]] = {}
    final_jobs: list[dict[str, Any]] = []
    for assigned in assigned_jobs:
        key = (str(assigned["alias"]), str(assigned["arm"]))
        config = deepcopy(configs[key])
        launcher = dict(_mapping(config.get("launcher"), "config launcher"))
        launcher["requested_gpu"] = str(assigned["requested_gpu"])
        config["launcher"] = launcher
        try:
            validate_experiment_config(config)
        except Exception as exc:
            raise MulticoreMaterializationFinalizationError(
                "A GPU-rewritten config fails experiment validation."
            ) from exc
        digest = canonical_sha256(config)
        rewritten_configs[key] = config
        final_jobs.append(
            {
                **assigned,
                "estimated_work": _expected_job_work(
                    core=core_records[key[0]],
                    arm=key[1],
                ),
                "config_sha256": digest,
            }
        )

    result = deepcopy(materialization)
    cohort = dict(_mapping(result.get("cohort"), "materialized cohort"))
    if (
        cohort.get("tissue_context")
        != "pathology_confirmed_adjacent_normal"
        or cohort.get("core_count") != 10
        or cohort.get("donor_count") != 10
        or cohort.get("slide_balance") != {"SO_1": 5, "SO_2": 5}
        or cohort.get("true_normal_core_count") != 0
    ):
        raise MulticoreMaterializationFinalizationError(
            "Materialized cohort declaration is not the locked ten-core "
            "adjacent-normal cohort."
        )
    cohort["protected_selection_manifest_sha256"] = selection_sha256
    result["cohort"] = cohort
    result["jobs"] = final_jobs
    result["job_count"] = len(final_jobs)
    result["finalization"] = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "utility": "finalize_multicore_qkv_materialization",
        "external_selection_sha256": selection_sha256,
        "validated_alias_count": len(core_records),
        "validated_artifact_count": len(core_records),
        "validated_graph_record_count": (
            len(core_records) * len(GRAPH_K_VALUES)
        ),
        "validated_config_count": len(final_jobs),
        "gpu_assignment": {
            "algorithm": "self_aware_lpt_v1",
            "gpu_count": GPU_COUNT,
            "self_work_formula": (
                "n_nodes*100 + 0.01*k5000_n_directed_edges"
            ),
            "loads": {
                str(index): load
                for index, load in enumerate(gpu_loads)
            },
            "job_counts": {
                str(index): sum(
                    int(job["requested_gpu"]) == index
                    for job in final_jobs
                )
                for index in range(GPU_COUNT)
            },
        },
        "graph_recomputation_performed": False,
        "registry_or_queue_mutation_performed": False,
        "accepted_input_contract": (
            "legacy_unfinalized_v1_or_strict_finalized_v1"
        ),
    }
    result.pop("checksum", None)
    result["checksum"] = canonical_sha256(result)

    config_rewrites = [
        (config_paths[key], rewritten_configs[key])
        for key in sorted(rewritten_configs)
    ]
    _write_configs_atomically(config_rewrites)
    _atomic_write_json(materialization_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    paths = current_paths()
    state_dir = (
        paths.scratch_root / "locked_campaigns" / CAMPAIGN_ID
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--materialization",
        type=Path,
        default=state_dir / "campaign_materialization.json",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = finalize_materialization(
        materialization_path=arguments.materialization.resolve(),
        selection_manifest=arguments.selection_manifest.resolve(),
    )
    print(
        json.dumps(
            {
                "campaign_id": result["campaign_id"],
                "alias_count": result["finalization"][
                    "validated_alias_count"
                ],
                "config_count": result["finalization"][
                    "validated_config_count"
                ],
                "gpu_count": result["finalization"]["gpu_assignment"][
                    "gpu_count"
                ],
                "status": "audited_and_finalized_not_enqueued",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
