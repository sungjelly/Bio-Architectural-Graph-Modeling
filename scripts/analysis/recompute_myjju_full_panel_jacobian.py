#!/usr/bin/env python3
"""Recompute the frozen MyJJu GeneMAE full-panel source-style Jacobian.

The workflow is deliberately split into graph preparation, a worst-case GPU
pilot, independent seed shards, and aggregation.  Generated state is accepted
only under one registered active run; raw and clinical inputs are read-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.full_panel_jacobian import (  # noqa: E402
    FullPanelJacobianError,
    summarize_symmetric_directed_degree,
    uniform_shift_jacobian,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.myjju_genemae import (  # noqa: E402
    build_symmetric_knn_graph,
    recursive_spatial_tiles,
)
from spatial_benchmark.myjju_genemae_comparison import (  # noqa: E402
    RegisteredRunEvidence,
    discover_registered_genemae_production,
)
from spatial_benchmark.paths import ProjectPaths, current_paths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402


CAMPAIGN_ID = "cmp_20260802_myjju_genemae_full_panel_k1000_jacobian"
UPSTREAM_CAMPAIGN_ID = "cmp_20260730_myjju_genemae_10core_comparison"
CONTRACT_SHA256 = (
    "cb335f12532e2f37a021d28687411b94245d70f223169b82e89a9971abf67fde"
)
PROTOCOL_SHA256 = (
    "ef86d6c5bd850faf9677b11285f2c0d75dc2adddbbf5ba80cf81921165c5388c"
)
EXPECTED_COHORT_SHA256 = (
    "b3c06228ce7fda4d4c5da09c3281de46e67b8069dfadb14276ba6402af87767e"
)
ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
SEEDS = tuple(range(7))
EXPECTED_TOTAL_NODES = 117_386
EXPECTED_GENES = 1_000
GRAPH_K = 1_000
MAXIMUM_TILE_NODES = 7_000
EXPECTED_TILE_COUNT = 26
PILOT_TARGET_COUNT = 8
PILOT_MAX_ALLOCATED_GIB = 20.5
PILOT_MAX_PROJECTED_GPU_HOURS = 48.0
PILOT_SAFETY_FACTOR = 1.25
GRAPH_MANIFEST_KIND = "myjju_full_panel_k1000_graph_cache_v1"
SEED_SHARD_KIND = "myjju_full_panel_k1000_jacobian_seed_v1"


class WorkflowError(RuntimeError):
    """Raised when execution would violate the frozen workflow contract."""


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    paths: ProjectPaths
    database_path: Path
    evidence: tuple[RegisteredRunEvidence, ...]
    config: Mapping[str, Any]
    cohort: Any
    gene_names: tuple[str, ...]
    analysis_input_sha256: str
    input_provenance: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(label: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise WorkflowError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"{label} must be a JSON mapping")
    return value


def _runner() -> Any:
    return importlib.import_module("scripts.train.run_myjju_genemae_pooled")


def _verify_design_files(paths: ProjectPaths) -> None:
    campaign = paths.project_root / "experiments" / "campaigns" / CAMPAIGN_ID
    expected = {
        campaign / "frozen_task_contract.yaml": CONTRACT_SHA256,
        campaign / "implementation_protocol.yaml": PROTOCOL_SHA256,
    }
    changed = [
        path.relative_to(paths.project_root).as_posix()
        for path, checksum in expected.items()
        if not path.is_file() or _sha256_file(path) != checksum
    ]
    if changed:
        raise WorkflowError(f"frozen design file identity changed: {changed}")


def _load_context(*, paths: ProjectPaths, database_path: Path) -> ExecutionContext:
    _verify_design_files(paths)
    if not database_path.is_file():
        raise WorkflowError(f"authoritative registry is absent: {database_path}")
    registry = Registry(database_path, initialize=False)
    if registry.get_campaign(CAMPAIGN_ID) is None:
        raise WorkflowError("full-panel campaign is not registered")
    evidence, _, _, _ = discover_registered_genemae_production(
        registry=registry,
        paths=paths,
    )
    evidence = tuple(sorted(evidence, key=lambda item: item.seed))
    if tuple(item.seed for item in evidence) != SEEDS:
        raise WorkflowError("upstream checkpoint discovery lacks seeds 0 through 6")
    runner = _runner()
    configs = tuple(
        runner.load_yaml_mapping(item.artifact_root / "config.resolved.yaml")
        for item in evidence
    )
    common_sections = ("dataset", "graph", "evaluation", "model")
    common_identity = canonical_sha256(
        {name: configs[0].get(name) for name in common_sections}
    )
    if any(
        canonical_sha256({name: config.get(name) for name in common_sections})
        != common_identity
        for config in configs[1:]
    ):
        raise WorkflowError("upstream checkpoints do not share one input identity")
    cohort = runner.load_verified_ten_core_cohort(configs[0])
    gene_names = tuple(str(name) for name in cohort.gene_names)
    if (
        tuple(cohort.aliases) != ALIASES
        or cohort.total_nodes != EXPECTED_TOTAL_NODES
        or cohort.n_genes != EXPECTED_GENES
        or len(gene_names) != EXPECTED_GENES
        or len(set(gene_names)) != EXPECTED_GENES
        or cohort.fingerprint_sha256 != EXPECTED_COHORT_SHA256
    ):
        raise WorkflowError("loaded cohort or ordered gene identity changed")
    provenance = {
        "campaign_id": CAMPAIGN_ID,
        "upstream_campaign_id": UPSTREAM_CAMPAIGN_ID,
        "frozen_contract_sha256": CONTRACT_SHA256,
        "implementation_protocol_sha256": PROTOCOL_SHA256,
        "cohort_fingerprint_sha256": cohort.fingerprint_sha256,
        "cohort_checksums": cohort.checksums.to_dict(),
        "ordered_gene_names_sha256": canonical_sha256(list(gene_names)),
        "checkpoint_training_graph_k": 15,
        "replay_graph_k": GRAPH_K,
        "members": [
            {
                "seed": item.seed,
                "run_id": item.run_id,
                "checkpoint_sha256": item.checkpoint_sha256,
                "state_dict_sha256": item.state_dict_sha256,
                "config_sha256": item.config_sha256,
            }
            for item in evidence
        ],
    }
    return ExecutionContext(
        paths=paths,
        database_path=database_path,
        evidence=evidence,
        config=configs[0],
        cohort=cohort,
        gene_names=gene_names,
        analysis_input_sha256=canonical_sha256(provenance),
        input_provenance=provenance,
    )


def _owned_work_root(paths: ProjectPaths, value: Path) -> Path:
    root = value.resolve(strict=False)
    active_root = (paths.scratch_root / "active_runs").resolve(strict=True)
    if not root.is_relative_to(active_root) or root.parent != active_root:
        raise WorkflowError(
            "work-root must be one direct registered run under scratch/active_runs"
        )
    owner = _load_json(root / ".bagm-run-owner.json", label="run owner marker")
    if owner.get("run_id") != root.name or owner.get("format_version") != 1:
        raise WorkflowError("work-root ownership marker does not match its run ID")
    if any((root / marker).exists() for marker in ("_SUCCESS", "_FAILED", "_PRUNED")):
        raise WorkflowError("work-root is already finalized")
    return root


def _tile_cache_paths(work_root: Path, alias: str, tile_index: int) -> tuple[Path, Path]:
    base = work_root / "graph_cache" / f"k{GRAPH_K}" / alias
    stem = f"tile-{tile_index:02d}"
    return base / f"{stem}.npz", base / f"{stem}.json"


def _tile_partitions(context: ExecutionContext) -> dict[str, tuple[np.ndarray, ...]]:
    result: dict[str, tuple[np.ndarray, ...]] = {}
    total = 0
    for core in context.cohort.cores:
        partitions = recursive_spatial_tiles(
            core.coordinates_um,
            max_nodes=MAXIMUM_TILE_NODES,
        )
        if not partitions:
            raise WorkflowError(f"{core.alias} produced no spatial tiles")
        joined = np.concatenate(partitions)
        if (
            joined.size != core.n_nodes
            or np.unique(joined).size != core.n_nodes
            or not np.array_equal(
                np.sort(joined), np.arange(core.n_nodes, dtype=np.int64)
            )
            or min(int(item.size) for item in partitions) <= GRAPH_K
        ):
            raise WorkflowError(
                f"{core.alias} does not have an exact partition with every tile "
                f"larger than k={GRAPH_K}"
            )
        result[core.alias] = partitions
        total += len(partitions)
    if total != EXPECTED_TILE_COUNT:
        raise WorkflowError(
            f"tile inventory changed: observed {total}, expected {EXPECTED_TILE_COUNT}"
        )
    return result


def _tile_record(
    *,
    alias: str,
    tile_index: int,
    node_indices: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    cache_path: Path,
) -> dict[str, Any]:
    if edge_index.shape[1] != edge_attr.shape[0] or not np.isfinite(edge_attr).all():
        raise WorkflowError(f"{alias} tile {tile_index} edge arrays are invalid")
    degree = summarize_symmetric_directed_degree(
        edge_index,
        node_count=int(node_indices.size),
        required_minimum=GRAPH_K,
    )
    record = {
        "alias": alias,
        "tile_index": int(tile_index),
        "n_nodes": int(node_indices.size),
        "n_directed_edges": int(edge_index.shape[1]),
        "node_indices_sha256": _array_sha256("node_indices", node_indices),
        "edge_index_sha256": _array_sha256("edge_index", edge_index),
        "edge_attr_sha256": _array_sha256("edge_attr", edge_attr),
        "degree": asdict(degree),
        "cache_relative_path": cache_path.as_posix(),
        "graph_semantics": {
            "k": GRAPH_K,
            "symmetry": "union",
            "constructed_self_loops": False,
            "convolution_add_self_loops": True,
            "edge_feature": "gaussian_distance_kernel",
        },
    }
    record["tile_identity_sha256"] = canonical_sha256(record)
    return record


def _load_cached_tile(
    work_root: Path,
    record: Mapping[str, Any],
    *,
    audit_degree: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative = Path(str(record.get("cache_relative_path", "")))
    path = (work_root / relative).resolve(strict=False)
    if relative.is_absolute() or not path.is_relative_to(work_root.resolve()):
        raise WorkflowError("graph cache path escapes the owned active run")
    if not path.is_file() or path.is_symlink():
        raise WorkflowError(f"graph cache file is missing or unsafe: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != {"node_indices", "edge_index", "edge_attr"}:
            raise WorkflowError(f"graph cache array inventory changed: {path}")
        node_indices = np.asarray(bundle["node_indices"])
        edge_index = np.asarray(bundle["edge_index"])
        edge_attr = np.asarray(bundle["edge_attr"])
    if (
        node_indices.dtype != np.int64
        or edge_index.dtype != np.int64
        or edge_attr.dtype != np.float32
    ):
        raise WorkflowError(f"graph cache array dtype changed: {path}")
    expected = {
        "node_indices_sha256": _array_sha256("node_indices", node_indices),
        "edge_index_sha256": _array_sha256("edge_index", edge_index),
        "edge_attr_sha256": _array_sha256("edge_attr", edge_attr),
    }
    if any(record.get(name) != value for name, value in expected.items()):
        raise WorkflowError(f"graph cache checksum changed: {path}")
    if (
        node_indices.shape != (int(record["n_nodes"]),)
        or edge_index.shape != (2, int(record["n_directed_edges"]))
        or edge_attr.shape != (int(record["n_directed_edges"]), 1)
        or not np.isfinite(edge_attr).all()
    ):
        raise WorkflowError(f"graph cache shape or finiteness changed: {path}")
    if audit_degree:
        observed = summarize_symmetric_directed_degree(
            edge_index,
            node_count=int(node_indices.size),
            required_minimum=GRAPH_K,
        )
        if canonical_sha256(asdict(observed)) != canonical_sha256(
            record.get("degree")
        ):
            raise WorkflowError(f"graph degree audit changed: {path}")
    return node_indices, edge_index, edge_attr


def _prepare_graphs(context: ExecutionContext, work_root: Path) -> dict[str, Any]:
    manifest_path = work_root / "graph_cache" / f"k{GRAPH_K}" / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path, label="graph cache manifest")
        _validate_graph_manifest(context, work_root, manifest, audit_degree=True)
        return manifest
    partitions = _tile_partitions(context)
    records: list[dict[str, Any]] = []
    for alias in ALIASES:
        core = context.cohort.core(alias)
        for tile_index, raw_indices in enumerate(partitions[alias]):
            cache_path, record_path = _tile_cache_paths(
                work_root, alias, tile_index
            )
            relative_cache = cache_path.relative_to(work_root)
            if cache_path.exists() or record_path.exists():
                record = _load_json(record_path, label="partial graph tile record")
                node_indices, edge_index, edge_attr = _load_cached_tile(
                    work_root, record, audit_degree=True
                )
                expected_indices = np.asarray(raw_indices, dtype=np.int64)
                if not np.array_equal(node_indices, expected_indices):
                    raise WorkflowError(
                        f"{alias} tile {tile_index} cached partition changed"
                    )
                records.append(record)
                del node_indices, edge_index, edge_attr
                continue
            node_indices = np.asarray(raw_indices, dtype=np.int64)
            coordinates = np.asarray(
                core.coordinates_um[node_indices], dtype=np.float64
            )
            started = time.monotonic()
            edge_index, edge_attr = build_symmetric_knn_graph(
                coordinates,
                k=GRAPH_K,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("xb") as handle:
                np.savez(
                    handle,
                    node_indices=node_indices,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
                handle.flush()
                os.fsync(handle.fileno())
            record = _tile_record(
                alias=alias,
                tile_index=tile_index,
                node_indices=node_indices,
                edge_index=edge_index,
                edge_attr=edge_attr,
                cache_path=relative_cache,
            )
            record["construction_runtime_seconds"] = time.monotonic() - started
            record["cache_size_bytes"] = cache_path.stat().st_size
            record["tile_identity_sha256"] = canonical_sha256(
                {key: value for key, value in record.items() if key != "tile_identity_sha256"}
            )
            _write_json_exclusive(record_path, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "event": "graph_tile_complete",
                        "alias": alias,
                        "tile_index": tile_index,
                        "nodes": int(node_indices.size),
                        "edges": int(edge_index.shape[1]),
                        "minimum_degree": record["degree"]["minimum"],
                        "runtime_seconds": record["construction_runtime_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del coordinates, edge_index, edge_attr
            gc.collect()
    manifest = {
        "schema_version": 1,
        "artifact_kind": GRAPH_MANIFEST_KIND,
        "campaign_id": CAMPAIGN_ID,
        "analysis_input_sha256": context.analysis_input_sha256,
        "graph_k": GRAPH_K,
        "maximum_tile_nodes": MAXIMUM_TILE_NODES,
        "tile_count": len(records),
        "core_aliases": list(ALIASES),
        "tiles": records,
    }
    manifest["graph_bundle_sha256"] = canonical_sha256(manifest)
    _write_json_exclusive(manifest_path, manifest)
    _validate_graph_manifest(context, work_root, manifest, audit_degree=True)
    return manifest


def _validate_graph_manifest(
    context: ExecutionContext,
    work_root: Path,
    manifest: Mapping[str, Any],
    *,
    audit_degree: bool,
) -> tuple[Mapping[str, Any], ...]:
    core = dict(manifest)
    checksum = core.pop("graph_bundle_sha256", None)
    rows = manifest.get("tiles")
    if (
        manifest.get("artifact_kind") != GRAPH_MANIFEST_KIND
        or manifest.get("campaign_id") != CAMPAIGN_ID
        or manifest.get("analysis_input_sha256") != context.analysis_input_sha256
        or manifest.get("graph_k") != GRAPH_K
        or manifest.get("maximum_tile_nodes") != MAXIMUM_TILE_NODES
        or manifest.get("tile_count") != EXPECTED_TILE_COUNT
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_TILE_COUNT
        or checksum != canonical_sha256(core)
    ):
        raise WorkflowError("graph cache manifest identity or checksum is invalid")
    expected_keys = {
        (alias, tile_index)
        for alias, partitions in _tile_partitions(context).items()
        for tile_index in range(len(partitions))
    }
    observed_keys: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise WorkflowError("graph cache tile record must be a mapping")
        key = (str(row.get("alias")), int(row.get("tile_index", -1)))
        if key in observed_keys:
            raise WorkflowError(f"duplicate graph cache tile: {key}")
        observed_keys.add(key)
        tile_core = dict(row)
        declared_tile_sha256 = tile_core.pop("tile_identity_sha256", None)
        if declared_tile_sha256 != canonical_sha256(tile_core):
            raise WorkflowError(f"graph cache tile identity checksum failed: {key}")
        _load_cached_tile(work_root, row, audit_degree=audit_degree)
    if observed_keys != expected_keys:
        raise WorkflowError("graph cache does not cover the exact tile inventory")
    return tuple(rows)


def _freeze_model(model: Any, device: str) -> Any:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.to(device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise WorkflowError("model evaluation/parameter-gradient policy failed")
    return model


def _load_model(context: ExecutionContext, *, seed: int, device: str) -> Any:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    return _freeze_model(
        _runner().reconstruct_model_from_checkpoint(
            context.evidence[seed].checkpoint_path,
            device=device,
        ),
        device,
    )


def _graph_rows_by_key(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    return {
        (str(row["alias"]), int(row["tile_index"])): row
        for row in rows
    }


def _cuda_environment(torch: Any, device: str) -> dict[str, Any]:
    resolved = torch.device(device)
    result: dict[str, Any] = {
        "device": str(resolved),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "host": socket.gethostname(),
    }
    if resolved.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(resolved)
        result["gpu_capability"] = list(torch.cuda.get_device_capability(resolved))
    return result


def _time_jacobian(
    *,
    torch: Any,
    model: Any,
    x: Any,
    edge_index: Any,
    edge_attr: Any,
    targets: tuple[int, ...],
) -> tuple[Any, float]:
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    started = time.monotonic()
    result = uniform_shift_jacobian(
        model,
        x,
        edge_index,
        edge_attr=edge_attr,
        target_gene_indices=targets,
        source_gene_indices=tuple(range(EXPECTED_GENES)),
        target_batch_size=1,
    )
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    return result, time.monotonic() - started


def _run_pilot(
    context: ExecutionContext,
    work_root: Path,
    manifest: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    import torch

    output_path = work_root / "pilot" / "resource_pilot.json"
    if output_path.exists():
        result = _load_json(output_path, label="resource pilot")
        if (
            result.get("campaign_id") != CAMPAIGN_ID
            or result.get("analysis_input_sha256") != context.analysis_input_sha256
            or result.get("graph_bundle_sha256")
            != manifest.get("graph_bundle_sha256")
        ):
            raise WorkflowError("existing resource pilot identity changed")
        return result
    rows = _validate_graph_manifest(
        context, work_root, manifest, audit_degree=False
    )
    largest = max(
        rows,
        key=lambda row: (
            int(row["n_nodes"]),
            str(row["alias"]),
            int(row["tile_index"]),
        ),
    )
    if (
        largest.get("alias") != "ANC-06"
        or largest.get("n_nodes") != 6561
    ):
        raise WorkflowError("prespecified largest pilot tile identity changed")
    node_indices, edge_index_np, edge_attr_np = _load_cached_tile(
        work_root, largest, audit_degree=True
    )
    core = context.cohort.core(str(largest["alias"]))
    normalized = _runner().log1p_cp10k(core.expression_counts)
    target_device = torch.device(device)
    model = _load_model(context, seed=0, device=device)
    x = torch.from_numpy(np.asarray(normalized[node_indices], dtype=np.float32)).to(
        target_device
    )
    edge_index = torch.from_numpy(edge_index_np).to(target_device)
    edge_attr = torch.from_numpy(edge_attr_np).to(target_device)
    targets = tuple(
        int(item)
        for item in np.linspace(
            0,
            EXPECTED_GENES - 1,
            PILOT_TARGET_COUNT,
            dtype=np.int64,
        )
    )
    if len(set(targets)) != PILOT_TARGET_COUNT:
        raise WorkflowError("pilot target selection is not unique")
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)
    one, one_runtime = _time_jacobian(
        torch=torch,
        model=model,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        targets=(targets[0],),
    )
    many, many_runtime = _time_jacobian(
        torch=torch,
        model=model,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        targets=targets,
    )
    replay, replay_runtime = _time_jacobian(
        torch=torch,
        model=model,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        targets=targets,
    )
    if target_device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(target_device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(target_device) / 1024**3
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0
    first_target_error = float(
        np.max(np.abs(one.signed_jacobian[0] - many.signed_jacobian[0]))
    )
    replay_error = float(
        np.max(np.abs(many.signed_jacobian - replay.signed_jacobian))
    )
    live_tolerance_passed = bool(
        np.allclose(
            one.signed_jacobian[0],
            many.signed_jacobian[0],
            atol=1.0e-6,
            rtol=1.0e-5,
        )
    )
    deterministic_replay = bool(
        np.allclose(
            many.signed_jacobian,
            replay.signed_jacobian,
            atol=1.0e-6,
            rtol=1.0e-5,
        )
    )
    backward_seconds = max(
        0.0,
        (many_runtime - one_runtime) / float(PILOT_TARGET_COUNT - 1),
    )
    forward_seconds = max(0.0, one_runtime - backward_seconds)
    if backward_seconds == 0.0:
        # Timing noise must not create an implausibly optimistic projection.
        backward_seconds = many_runtime / float(PILOT_TARGET_COUNT)
        forward_seconds = 0.0
    projected_tile_seconds = forward_seconds + EXPECTED_GENES * backward_seconds
    projected_total_gpu_hours = (
        projected_tile_seconds
        * EXPECTED_TILE_COUNT
        * len(SEEDS)
        * PILOT_SAFETY_FACTOR
        / 3600.0
    )
    finite = bool(
        np.isfinite(one.signed_jacobian).all()
        and np.isfinite(many.signed_jacobian).all()
        and np.isfinite(replay.signed_jacobian).all()
    )
    failures: list[str] = []
    if not finite:
        failures.append("nonfinite_jacobian")
    if not live_tolerance_passed:
        failures.append("one_row_vs_multirow_mismatch")
    if not deterministic_replay:
        failures.append("deterministic_replay_mismatch")
    if peak_allocated > PILOT_MAX_ALLOCATED_GIB:
        failures.append("peak_allocated_vram_gate")
    if projected_total_gpu_hours > PILOT_MAX_PROJECTED_GPU_HOURS:
        failures.append("projected_gpu_hours_gate")
    result = {
        "schema_version": 1,
        "artifact_kind": "myjju_full_panel_k1000_resource_pilot_v1",
        "campaign_id": CAMPAIGN_ID,
        "analysis_input_sha256": context.analysis_input_sha256,
        "graph_bundle_sha256": manifest["graph_bundle_sha256"],
        "checkpoint": {
            "seed": 0,
            "run_id": context.evidence[0].run_id,
            "checkpoint_sha256": context.evidence[0].checkpoint_sha256,
        },
        "tile": {
            "alias": largest["alias"],
            "tile_index": largest["tile_index"],
            "n_nodes": largest["n_nodes"],
            "n_directed_edges": largest["n_directed_edges"],
            "minimum_realized_degree": largest["degree"]["minimum"],
            "tile_identity_sha256": largest["tile_identity_sha256"],
        },
        "target_indices": list(targets),
        "resources": {
            **_cuda_environment(torch, device),
            "one_target_runtime_seconds": one_runtime,
            "eight_target_runtime_seconds": many_runtime,
            "replay_runtime_seconds": replay_runtime,
            "estimated_forward_seconds": forward_seconds,
            "estimated_backward_seconds_per_target": backward_seconds,
            "projected_seconds_per_largest_tile": projected_tile_seconds,
            "projected_total_gpu_hours": projected_total_gpu_hours,
            "projection_safety_factor": PILOT_SAFETY_FACTOR,
            "peak_allocated_vram_gib": peak_allocated,
            "peak_reserved_vram_gib": peak_reserved,
        },
        "diagnostics": {
            "finite": finite,
            "one_row_vs_multirow_maximum_absolute_error": first_target_error,
            "one_row_vs_multirow_tolerance_passed": live_tolerance_passed,
            "deterministic_replay_maximum_absolute_error": replay_error,
            "deterministic_replay": deterministic_replay,
            "model_eval_mode": not model.training,
            "model_parameter_gradients_disabled": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
        },
        "failed_checks": failures,
        "passed": not failures,
        "maximum_defensible_claim": "resource_and_numerical_preflight_only",
    }
    _write_json_exclusive(output_path, result)
    return result


def _seed_paths(work_root: Path, seed: int) -> tuple[Path, Path]:
    root = work_root / "shards" / f"seed-{seed:02d}"
    return root / "arrays.npz", root / "metadata.json"


def _seed_tile_paths(
    work_root: Path, seed: int, alias: str, tile_index: int
) -> tuple[Path, Path]:
    root = work_root / "shards" / f"seed-{seed:02d}" / "tiles" / alias
    stem = f"tile-{tile_index:02d}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _run_seed(
    context: ExecutionContext,
    work_root: Path,
    manifest: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    target_batch_size: int,
) -> dict[str, Any]:
    import torch

    pilot = _load_json(
        work_root / "pilot" / "resource_pilot.json", label="resource pilot"
    )
    if pilot.get("passed") is not True:
        raise WorkflowError("production is prohibited because the pilot did not pass")
    arrays_path, metadata_path = _seed_paths(work_root, seed)
    if arrays_path.exists() or metadata_path.exists():
        metadata = _load_json(metadata_path, label=f"seed {seed} metadata")
        if (
            metadata.get("status") != "success"
            or metadata.get("seed") != seed
            or metadata.get("analysis_input_sha256") != context.analysis_input_sha256
            or metadata.get("graph_bundle_sha256")
            != manifest.get("graph_bundle_sha256")
            or not arrays_path.is_file()
            or _sha256_file(arrays_path) != metadata.get("arrays_sha256")
        ):
            raise WorkflowError(f"existing seed {seed} shard is invalid")
        return metadata
    rows = _validate_graph_manifest(
        context, work_root, manifest, audit_degree=False
    )
    by_key = _graph_rows_by_key(rows)
    partitions = _tile_partitions(context)
    model = _load_model(context, seed=seed, device=device)
    target_device = torch.device(device)
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)
    started = time.monotonic()
    core_matrices = np.zeros(
        (len(ALIASES), EXPECTED_GENES, EXPECTED_GENES), dtype=np.float32
    )
    tile_records: list[dict[str, Any]] = []
    for core_index, alias in enumerate(ALIASES):
        core = context.cohort.core(alias)
        normalized = _runner().log1p_cp10k(core.expression_counts)
        numerator = np.zeros((EXPECTED_GENES, EXPECTED_GENES), dtype=np.float64)
        denominator = 0
        for tile_index in range(len(partitions[alias])):
            graph_record = by_key[(alias, tile_index)]
            tile_array_path, tile_metadata_path = _seed_tile_paths(
                work_root, seed, alias, tile_index
            )
            if tile_array_path.exists() or tile_metadata_path.exists():
                tile_meta = _load_json(
                    tile_metadata_path, label=f"seed {seed} tile metadata"
                )
                if (
                    tile_meta.get("tile_identity_sha256")
                    != graph_record.get("tile_identity_sha256")
                    or tile_meta.get("checkpoint_sha256")
                    != context.evidence[seed].checkpoint_sha256
                    or tile_meta.get("target_batch_size") != target_batch_size
                    or not tile_array_path.is_file()
                    or _sha256_file(tile_array_path)
                    != tile_meta.get("arrays_sha256")
                ):
                    raise WorkflowError(
                        f"seed {seed} {alias} tile {tile_index} partial result is invalid"
                    )
                with np.load(tile_array_path, allow_pickle=False) as bundle:
                    matrix = np.asarray(bundle["signed_jacobian"], dtype=np.float32)
                if matrix.shape != (EXPECTED_GENES, EXPECTED_GENES) or not np.isfinite(
                    matrix
                ).all():
                    raise WorkflowError("cached tile Jacobian is invalid")
                tile_records.append(tile_meta)
            else:
                node_indices, edge_index_np, edge_attr_np = _load_cached_tile(
                    work_root, graph_record, audit_degree=False
                )
                x = torch.from_numpy(
                    np.asarray(normalized[node_indices], dtype=np.float32)
                ).to(target_device)
                edge_index = torch.from_numpy(edge_index_np).to(target_device)
                edge_attr = torch.from_numpy(edge_attr_np).to(target_device)
                tile_started = time.monotonic()
                result = uniform_shift_jacobian(
                    model,
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                    target_gene_indices=tuple(range(EXPECTED_GENES)),
                    source_gene_indices=tuple(range(EXPECTED_GENES)),
                    target_batch_size=target_batch_size,
                )
                if target_device.type == "cuda":
                    torch.cuda.synchronize(target_device)
                matrix = result.signed_jacobian.astype(np.float32)
                if matrix.shape != (EXPECTED_GENES, EXPECTED_GENES) or not np.isfinite(
                    matrix
                ).all():
                    raise WorkflowError("tile Jacobian is invalid")
                tile_array_path.parent.mkdir(parents=True, exist_ok=True)
                with tile_array_path.open("xb") as handle:
                    np.savez(handle, signed_jacobian=matrix)
                    handle.flush()
                    os.fsync(handle.fileno())
                tile_meta = {
                    "schema_version": 1,
                    "artifact_kind": "myjju_full_panel_k1000_tile_jacobian_v1",
                    "campaign_id": CAMPAIGN_ID,
                    "analysis_input_sha256": context.analysis_input_sha256,
                    "graph_bundle_sha256": manifest["graph_bundle_sha256"],
                    "tile_identity_sha256": graph_record["tile_identity_sha256"],
                    "seed": seed,
                    "checkpoint_sha256": context.evidence[seed].checkpoint_sha256,
                    "alias": alias,
                    "tile_index": tile_index,
                    "n_nodes": int(node_indices.size),
                    "target_batch_size": target_batch_size,
                    "runtime_seconds": time.monotonic() - tile_started,
                    "matrix_shape": [EXPECTED_GENES, EXPECTED_GENES],
                    "matrix_dtype": "float32",
                    "matrix_sha256": _array_sha256("signed_jacobian", matrix),
                    "arrays_sha256": _sha256_file(tile_array_path),
                }
                _write_json_exclusive(tile_metadata_path, tile_meta)
                tile_records.append(tile_meta)
                print(
                    json.dumps(
                        {
                            "event": "jacobian_tile_complete",
                            "seed": seed,
                            "alias": alias,
                            "tile_index": tile_index,
                            "runtime_seconds": tile_meta["runtime_seconds"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                del x, edge_index, edge_attr, result
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            node_count = int(graph_record["n_nodes"])
            numerator += matrix.astype(np.float64) * float(node_count)
            denominator += node_count
            del matrix
        if denominator != core.n_nodes:
            raise WorkflowError(f"{alias} tile node denominator changed")
        core_matrices[core_index] = (numerator / float(denominator)).astype(
            np.float32
        )
        del normalized, numerator
        gc.collect()
    if not np.isfinite(core_matrices).all():
        raise WorkflowError(f"seed {seed} core matrices contain nonfinite values")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    with arrays_path.open("xb") as handle:
        np.savez(
            handle,
            core_signed_jacobian=core_matrices,
            gene_names=np.asarray(context.gene_names, dtype="U"),
            core_aliases=np.asarray(ALIASES, dtype="U"),
        )
        handle.flush()
        os.fsync(handle.fileno())
    peak_allocated = (
        torch.cuda.max_memory_allocated(target_device) / 1024**3
        if target_device.type == "cuda"
        else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(target_device) / 1024**3
        if target_device.type == "cuda"
        else 0.0
    )
    metadata = {
        "schema_version": 1,
        "artifact_kind": SEED_SHARD_KIND,
        "campaign_id": CAMPAIGN_ID,
        "status": "success",
        "analysis_input_sha256": context.analysis_input_sha256,
        "graph_bundle_sha256": manifest["graph_bundle_sha256"],
        "seed": seed,
        "checkpoint_run_id": context.evidence[seed].run_id,
        "checkpoint_sha256": context.evidence[seed].checkpoint_sha256,
        "state_dict_sha256": context.evidence[seed].state_dict_sha256,
        "target_batch_size": target_batch_size,
        "coverage": {"cores": list(ALIASES), "tiles": len(tile_records)},
        "matrix_shape": [len(ALIASES), EXPECTED_GENES, EXPECTED_GENES],
        "matrix_dtype": "float32",
        "matrix_sha256": _array_sha256("core_signed_jacobian", core_matrices),
        "arrays_sha256": _sha256_file(arrays_path),
        "runtime_seconds": time.monotonic() - started,
        "resources": {
            **_cuda_environment(torch, device),
            "peak_allocated_vram_gib": peak_allocated,
            "peak_reserved_vram_gib": peak_reserved,
        },
        "tile_records": tile_records,
        "maximum_defensible_claim": (
            "per_seed_held_in_out_of_training_topology_model_implied_sensitivity"
        ),
    }
    _write_json_exclusive(metadata_path, metadata)
    return metadata


def _inspect(context: ExecutionContext) -> dict[str, Any]:
    partitions = _tile_partitions(context)
    return {
        "campaign_id": CAMPAIGN_ID,
        "analysis_input_sha256": context.analysis_input_sha256,
        "cohort_fingerprint_sha256": context.cohort.fingerprint_sha256,
        "total_nodes": context.cohort.total_nodes,
        "genes": len(context.gene_names),
        "seeds": [item.seed for item in context.evidence],
        "checkpoint_run_ids": [item.run_id for item in context.evidence],
        "graph": {
            "k": GRAPH_K,
            "maximum_tile_nodes": MAXIMUM_TILE_NODES,
            "tile_count": sum(len(value) for value in partitions.values()),
            "tile_nodes": {
                alias: [int(item.size) for item in partitions[alias]]
                for alias in ALIASES
            },
        },
        "claim": (
            "frozen_model_held_in_out_of_training_topology_model_implied_sensitivity"
        ),
    }


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("state/tracking/bagm.sqlite3"),
    )
    parser.add_argument("--work-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect")
    subparsers.add_parser("prepare-graphs")
    pilot = subparsers.add_parser("pilot")
    pilot.add_argument("--device", required=True)
    seed = subparsers.add_parser("run-seed")
    seed.add_argument("--seed", type=int, required=True, choices=SEEDS)
    seed.add_argument("--device", required=True)
    seed.add_argument("--target-batch-size", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    paths = current_paths()
    database = arguments.database
    if not database.is_absolute():
        database = paths.project_root / database
    context = _load_context(paths=paths, database_path=database.resolve())
    if arguments.command == "inspect":
        print(json.dumps(_inspect(context), indent=2, sort_keys=True))
        return 0
    if arguments.work_root is None:
        raise WorkflowError("--work-root is required for this command")
    work_root = _owned_work_root(paths, arguments.work_root)
    if arguments.command == "prepare-graphs":
        result = _prepare_graphs(context, work_root)
    else:
        manifest = _load_json(
            work_root / "graph_cache" / f"k{GRAPH_K}" / "manifest.json",
            label="graph cache manifest",
        )
        if arguments.command == "pilot":
            result = _run_pilot(
                context,
                work_root,
                manifest,
                device=arguments.device,
            )
        elif arguments.command == "run-seed":
            result = _run_seed(
                context,
                work_root,
                manifest,
                seed=arguments.seed,
                device=arguments.device,
                target_batch_size=arguments.target_batch_size,
            )
        else:  # pragma: no cover - argparse enforces the command set.
            raise AssertionError(arguments.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
