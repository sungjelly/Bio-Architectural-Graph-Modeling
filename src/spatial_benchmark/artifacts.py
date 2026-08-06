"""Immutable preparation artifacts for the normal-core benchmark.

Preparation is intentionally separate from model fitting.  It selects the
single approved tissue unit, creates the spatial split, fits node and edge
transforms on training regions only, builds split-restricted graph topology,
and generates fixed validation/test masks.  It never evaluates a test target.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .paths import PROJECT_ROOT as DEFAULT_PROJECT_ROOT

from .data import (
    ALLOWED_METADATA_COLUMNS,
    CoreSelection,
    discover_slide_raw_path,
    load_selected_core,
    select_unique_legacy_true_normal_core,
)
from .graphs import EdgeAttributeStandardizer, build_spatial_graph
from .masking import (
    FixedMaskBundle,
    MaskSpec,
    create_fixed_mask_bundle,
    load_fixed_mask_bundle,
    save_fixed_mask_bundle,
)
from .splits import TrainOnlyPreprocessor, make_spatial_split


PREPARED_ARTIFACT_FORMAT_VERSION = 1
_DATA_FILENAME = "prepared_data.npz"
_MANIFEST_FILENAME = "manifest.json"
_CHECKSUM_FILENAME = "checksums.sha256"
_MASK_DIRECTORY = "fixed_masks"

DEFAULT_PREPARE_CONFIG: dict[str, Any] = {
    "version": 1,
    "project_root": str(DEFAULT_PROJECT_ROOT),
    "data": {
        "raw_dir": "data/raw",
        "legacy_workbook": "data/clinical/Gastric Study_Old.xlsx",
        "pathology_review_workbook": "data/clinical/Gastric Study.xlsx",
        "core_map_csv": "data/clinical/fov_core_map.csv",
        "chunksize": 2048,
        "expected_biological_probes": 1000,
        "pixel_size_um": 0.120281,
        "qc_policy": "all",
        "selection_source": "legacy_true_normal",
        "selection_manifest": None,
        "selection_alias": None,
    },
    "split": {
        "block_size_um": 300.0,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "seed": 20260724,
        "fov_aware": True,
    },
    "preprocessing": {
        "log1p_metadata": True,
        "epsilon": 1e-8,
    },
    "graph": {
        "k": 12,
        "radius_um": 50.0,
        "symmetry": "union",
        "min_distance_um": 0.0,
        "rbf_bins": 8,
        "edge_dropout": 0.1,
    },
    "masking": {
        "mask_seed": 20260724,
        "curriculum": "P+N+B",
        "warmup_epochs": 10,
        "partial_gene_rate": 0.2,
        "whole_node_rate": 0.1,
        "block_node_rate": 0.1,
        "block_width_um": None,
        "block_shape": "disk",
        "validation_replicates": 3,
        "test_replicates": 5,
    },
}

_ALLOWED_KEYS: dict[str, set[str]] = {
    "root": set(DEFAULT_PREPARE_CONFIG).union({"model", "optimization", "run"}),
    "data": set(DEFAULT_PREPARE_CONFIG["data"]),
    "split": set(DEFAULT_PREPARE_CONFIG["split"]),
    "preprocessing": set(DEFAULT_PREPARE_CONFIG["preprocessing"]),
    "graph": set(DEFAULT_PREPARE_CONFIG["graph"]),
    "masking": set(DEFAULT_PREPARE_CONFIG["masking"]),
    "model": {
        "name",
        "hidden_dim",
        "attention_heads",
        "graph_layers",
        "ffn_dim",
        "decoder_dim",
        "edge_hidden_dim",
        "edge_embedding_dim",
        "message_head_dim",
        "message_dim",
        "dropout",
        "attention_dropout",
    },
    "optimization": {
        "learning_rate",
        "joint_learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "max_epochs",
        "early_stopping_patience",
        "validation_every",
        "amp",
    },
    "run": {
        "model_seed",
        "deterministic",
        "device",
        "save_predictions",
    },
}


class ArtifactContractError(ValueError):
    """Raised when preparation configuration or output violates its contract."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _check_unknown_keys(
    mapping: Mapping[str, Any],
    allowed: set[str],
    *,
    section: str,
) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise ArtifactContractError(
            f"Unknown preparation keys in {section}: {', '.join(unknown)}"
        )


def _validate_config(config: Mapping[str, Any]) -> None:
    _check_unknown_keys(config, _ALLOWED_KEYS["root"], section="root")
    if int(config["version"]) != 1:
        raise ArtifactContractError("Only preparation config version 1 is supported.")
    for section in ("data", "split", "preprocessing", "graph", "masking"):
        value = config.get(section)
        if not isinstance(value, Mapping):
            raise ArtifactContractError(f"Config section {section} must be a mapping.")
        _check_unknown_keys(value, _ALLOWED_KEYS[section], section=section)
    for section in ("model", "optimization", "run"):
        if section in config:
            value = config[section]
            if not isinstance(value, Mapping):
                raise ArtifactContractError(
                    f"Config section {section} must be a mapping."
                )
            _check_unknown_keys(value, _ALLOWED_KEYS[section], section=section)

    data = config["data"]
    if int(data["chunksize"]) <= 0:
        raise ArtifactContractError("data.chunksize must be positive.")
    expected = data["expected_biological_probes"]
    if expected is not None and int(expected) <= 0:
        raise ArtifactContractError(
            "data.expected_biological_probes must be positive or null."
        )
    if data["qc_policy"] not in {"all", "passed"}:
        raise ArtifactContractError("data.qc_policy must be 'all' or 'passed'.")
    selection_source = data["selection_source"]
    if selection_source not in {
        "legacy_true_normal",
        "protected_adjacent_normal_manifest",
    }:
        raise ArtifactContractError(
            "data.selection_source must be 'legacy_true_normal' or "
            "'protected_adjacent_normal_manifest'."
        )
    selection_manifest = data.get("selection_manifest")
    selection_alias = data.get("selection_alias")
    if selection_source == "legacy_true_normal":
        if selection_manifest is not None or selection_alias is not None:
            raise ArtifactContractError(
                "Legacy true-Normal selection forbids a selection manifest or alias."
            )
    elif (
        not isinstance(selection_manifest, str)
        or not selection_manifest.strip()
        or not isinstance(selection_alias, str)
        or not selection_alias.strip()
    ):
        raise ArtifactContractError(
            "Protected adjacent-normal selection requires a manifest path and alias."
        )
    if not bool(config["split"]["fov_aware"]):
        raise ArtifactContractError(
            "This workflow requires split.fov_aware=true for preparation."
        )
    masking = config["masking"]
    for name in ("validation_replicates", "test_replicates"):
        if int(masking[name]) <= 0:
            raise ArtifactContractError(f"masking.{name} must be positive.")
    _mask_specs(config)


def load_prepare_config(path: str | Path) -> dict[str, Any]:
    """Load a strict YAML configuration and fill declared defaults."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Preparation config was not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ArtifactContractError("Preparation YAML root must be a mapping.")
    _check_unknown_keys(loaded, _ALLOWED_KEYS["root"], section="root")
    for section in ("data", "split", "preprocessing", "graph", "masking"):
        if section in loaded:
            if not isinstance(loaded[section], Mapping):
                raise ArtifactContractError(
                    f"Config section {section} must be a mapping."
                )
            _check_unknown_keys(
                loaded[section], _ALLOWED_KEYS[section], section=section
            )
    for section in ("model", "optimization", "run"):
        if section in loaded:
            if not isinstance(loaded[section], Mapping):
                raise ArtifactContractError(
                    f"Config section {section} must be a mapping."
                )
            _check_unknown_keys(
                loaded[section], _ALLOWED_KEYS[section], section=section
            )
    config = _deep_merge(DEFAULT_PREPARE_CONFIG, loaded)
    _validate_config(config)
    return config


def _resolve_project_root(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> Path:
    configured = Path(str(config.get("project_root") or DEFAULT_PROJECT_ROOT))
    if configured.is_absolute():
        root = configured
    else:
        root = DEFAULT_PROJECT_ROOT / configured
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Configured project root is not a directory: {root}")
    return root


def _resolve_input_path(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    path = path if path.is_absolute() else project_root / path
    path = path.resolve()
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(f"Configured input path was not found: {path}")
    return path


def _relative_or_absolute(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _input_record(path: Path, project_root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _relative_or_absolute(path, project_root),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def _assert_inputs_unchanged(
    paths: Mapping[str, Path],
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    for name, path in paths.items():
        stat = path.stat()
        record = records[name]
        if (
            int(stat.st_size) != int(record["size_bytes"])
            or int(stat.st_mtime_ns) != int(record["mtime_ns"])
        ):
            raise ArtifactContractError(
                f"Input {name} changed while the prepared artifact was being built."
            )


def _run_command(
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout: float = 10.0,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 127, ""
    return int(completed.returncode), completed.stdout.strip()


def _git_provenance(project_root: Path) -> dict[str, Any]:
    commit_code, commit = _run_command(
        ["git", "rev-parse", "HEAD"], cwd=project_root
    )
    status_code, status = _run_command(
        ["git", "status", "--short", "--untracked-files=all"], cwd=project_root
    )
    if commit_code != 0:
        return {"available": False}
    status_bytes = status.encode("utf-8")
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status) if status_code == 0 else None,
        # Paths are not embedded in the artifact; the digest still fixes the
        # exact local status used for this preparation.
        "status_sha256": hashlib.sha256(status_bytes).hexdigest()
        if status_code == 0
        else None,
    }


def _package_versions() -> dict[str, str | None]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "PyYAML")
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _system_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size


def _gpu_hardware(project_root: Path) -> list[dict[str, Any]]:
    code, output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        cwd=project_root,
    )
    if code != 0 or not output:
        return []
    devices: list[dict[str, Any]] = []
    for index, line in enumerate(output.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        name, memory_mib, driver = parts
        try:
            memory_value: int | None = int(memory_mib)
        except ValueError:
            memory_value = None
        devices.append(
            {
                "device_index": index,
                "name": name,
                "memory_mib": memory_value,
                "driver_version": driver,
            }
        )
    return devices


def _hardware_provenance(project_root: Path) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": _system_memory_bytes(),
        "gpus": _gpu_hardware(project_root),
    }


def _provenance(
    project_root: Path,
    *,
    command: Sequence[str] | None,
    config_path: Path,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd().resolve()),
        "command": list(command) if command is not None else None,
        "config_path": _relative_or_absolute(config_path.resolve(), project_root),
        "config_sha256": sha256_file(config_path),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": _package_versions(),
        "git": _git_provenance(project_root),
        "hardware": _hardware_provenance(project_root),
    }


def _manifest_checksum(manifest: Mapping[str, Any]) -> str:
    core = deepcopy(dict(manifest))
    core.pop("artifact_id", None)
    core.pop("manifest_content_sha256", None)
    return hashlib.sha256(_canonical_json(core)).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _all_artifact_files(root: Path, *, include_checksums: bool = False) -> list[Path]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not include_checksums:
        files = [
            path
            for path in files
            if path.relative_to(root).as_posix() != _CHECKSUM_FILENAME
        ]
    return files


def _write_checksum_file(root: Path) -> None:
    records = []
    for path in _all_artifact_files(root):
        relative = path.relative_to(root).as_posix()
        records.append((relative, sha256_file(path)))
    (root / _CHECKSUM_FILENAME).write_text(
        "".join(
            f"{checksum}  {relative}\n"
            for relative, checksum in sorted(records)
        ),
        encoding="ascii",
    )


def _read_checksum_file(root: Path) -> dict[str, str]:
    path = root / _CHECKSUM_FILENAME
    if not path.is_file():
        raise FileNotFoundError("Prepared artifact lacks checksums.sha256.")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        try:
            checksum, relative = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise ArtifactContractError("Invalid prepared checksum line.") from exc
        if (
            len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise ArtifactContractError("Invalid prepared checksum record.")
        if relative in result:
            raise ArtifactContractError("Duplicate prepared checksum record.")
        result[relative] = checksum
    return result


def _verify_artifact_files(root: Path) -> None:
    expected = _read_checksum_file(root)
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in _all_artifact_files(root)
    }
    if set(expected) != set(actual_paths):
        raise ArtifactContractError("Prepared artifact file set differs from checksums.")
    for relative, path in actual_paths.items():
        if sha256_file(path) != expected[relative]:
            raise ArtifactContractError(
                f"Prepared artifact checksum mismatch for {relative}."
            )


def _mask_specs(config: Mapping[str, Any]) -> list[MaskSpec]:
    masking = config["masking"]
    common_block = {
        "block_node_rate": float(masking["block_node_rate"]),
        "block_width_um": masking.get("block_width_um"),
        "block_shape": str(masking.get("block_shape", "disk")),
    }
    return [
        MaskSpec(
            "partial",
            partial_gene_rate=float(masking["partial_gene_rate"]),
            label="partial-primary",
        ),
        MaskSpec(
            "node",
            node_rate=float(masking["whole_node_rate"]),
            label="whole-node-primary",
        ),
        MaskSpec(
            "block",
            label="spatial-block-primary",
            **common_block,
        ),
    ]


def _prepare_arrays_and_manifest(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    project_root: Path,
    command: Sequence[str] | None,
    temporary_output: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data_config = config["data"]
    raw_dir = _resolve_input_path(project_root, data_config["raw_dir"])
    selection_source = str(data_config["selection_source"])
    selection_context: dict[str, Any]
    if selection_source == "legacy_true_normal":
        legacy_workbook = _resolve_input_path(
            project_root, data_config["legacy_workbook"]
        )
        review_workbook = _resolve_input_path(
            project_root, data_config["pathology_review_workbook"]
        )
        core_map_csv = _resolve_input_path(
            project_root, data_config["core_map_csv"]
        )
        selection = select_unique_legacy_true_normal_core(
            legacy_workbook,
            core_map_csv,
            pathology_review_workbook=review_workbook,
        )
        selection_input_paths: dict[str, Path] = {
            "legacy_workbook": legacy_workbook,
            "pathology_review_workbook": review_workbook,
            "fov_to_tissue_map": core_map_csv,
        }
        artifact_kind = "normal_true_tissue_spatial_benchmark_preparation"
        selection_context = {
            "tissue_context": "legacy_true_normal",
            "selection_source": selection_source,
        }
    else:
        from .adjacent_normal_selection import load_adjacent_normal_route

        selection_manifest = _resolve_input_path(
            project_root, data_config["selection_manifest"]
        )
        route = load_adjacent_normal_route(
            selection_manifest,
            str(data_config["selection_alias"]),
        )
        selection = CoreSelection(
            slide=route.slide,
            fovs=route.fovs,
            label_policy=(
                "pathology-confirmed adjacent-normal protected manifest"
            ),
        )
        selection_input_paths = {
            "protected_selection_manifest": selection_manifest,
        }
        artifact_kind = (
            "adjacent_normal_tissue_spatial_benchmark_preparation"
        )
        selection_context = {
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "selection_source": selection_source,
            "opaque_alias": route.alias,
        }
    expression_path = discover_slide_raw_path(
        raw_dir, selection.slide, "expression"
    )
    metadata_path = discover_slide_raw_path(raw_dir, selection.slide, "metadata")
    input_paths = {
        **selection_input_paths,
        "expression_csv": expression_path,
        "metadata_csv": metadata_path,
    }
    input_records = {
        name: _input_record(path, project_root)
        for name, path in input_paths.items()
    }

    dataset = load_selected_core(
        raw_dir,
        selection,
        chunksize=int(data_config["chunksize"]),
        expected_biological_probes=(
            None
            if data_config["expected_biological_probes"] is None
            else int(data_config["expected_biological_probes"])
        ),
        pixel_size_um=float(data_config["pixel_size_um"]),
        qc_policy=str(data_config["qc_policy"]),
    )

    split_config = config["split"]
    fov = dataset.keys["fov"].to_numpy(dtype=np.int32, copy=True)
    spatial_split = make_spatial_split(
        dataset.coordinates_um,
        fov=fov,
        block_size_um=float(split_config["block_size_um"]),
        val_fraction=float(split_config["val_fraction"]),
        test_fraction=float(split_config["test_fraction"]),
        seed=int(split_config["seed"]),
        fov_aware=True,
    )
    spatial_split.assert_fovs_disjoint(fov)

    preprocessing_config = config["preprocessing"]
    node_preprocessor = TrainOnlyPreprocessor(
        log1p_metadata=bool(preprocessing_config["log1p_metadata"]),
        epsilon=float(preprocessing_config["epsilon"]),
    )
    node_preprocessor.fit_dataset(dataset, spatial_split)
    nodes = node_preprocessor.transform(dataset.expression, dataset.metadata)

    graph_config = config["graph"]
    graph = build_spatial_graph(
        dataset.coordinates_um,
        k=int(graph_config["k"]),
        radius_um=float(graph_config["radius_um"]),
        symmetry=str(graph_config["symmetry"]),
        group_labels=spatial_split.labels,
        fov=fov,
        min_distance_um=float(graph_config["min_distance_um"]),
        rbf_bins=int(graph_config["rbf_bins"]),
    )
    source_split = spatial_split.labels[graph.edge_index[0]]
    target_split = spatial_split.labels[graph.edge_index[1]]
    training_edges = (source_split == "train") & (target_split == "train")
    if not training_edges.any():
        raise ArtifactContractError("The training graph contains no edges.")
    if np.any(source_split != target_split) or graph.qc.cross_group_edges:
        raise ArtifactContractError("A prepared graph edge crosses split boundaries.")
    edge_preprocessor = EdgeAttributeStandardizer(
        epsilon=float(preprocessing_config["epsilon"])
    )
    edge_preprocessor.fit(graph.edge_attr[training_edges])
    standardized_edge_attr = edge_preprocessor.transform(graph.edge_attr)

    evaluation_indices = {
        "validation": np.flatnonzero(spatial_split.labels == "val"),
        "test": np.flatnonzero(spatial_split.labels == "test"),
    }
    masking_config = config["masking"]
    fixed_masks: dict[str, FixedMaskBundle] = {}
    for split_name, node_index in evaluation_indices.items():
        replicate_key = (
            "validation_replicates"
            if split_name == "validation"
            else "test_replicates"
        )
        bundle = create_fixed_mask_bundle(
            {split_name: dataset.coordinates_um[node_index]},
            dataset.n_genes,
            specs=_mask_specs(config),
            replicates=int(masking_config[replicate_key]),
            base_seed=int(masking_config["mask_seed"]),
        )
        fixed_masks[split_name] = bundle
        save_fixed_mask_bundle(
            bundle,
            temporary_output / _MASK_DIRECTORY / split_name,
        )

    arrays: dict[str, np.ndarray] = {
        "expression_counts": dataset.expression.astype(np.int32, copy=False),
        "target_expression": nodes.expression.astype(np.float32, copy=False),
        "node_covariates": nodes.metadata.astype(np.float32, copy=False),
        "coordinates_px": dataset.coordinates_px.astype(np.float64, copy=False),
        "coordinates_um": dataset.coordinates_um.astype(np.float64, copy=False),
        "fov": fov,
        "cell_ID": dataset.keys["cell_ID"].to_numpy(dtype=np.int32, copy=True),
        "qc_passed": dataset.qc_passed.astype(bool, copy=False),
        "split_labels": spatial_split.labels.astype("U5", copy=False),
        "macroblock_ids": spatial_split.macroblock_ids.astype("U96", copy=False),
        "train_node_index": np.flatnonzero(spatial_split.train_mask).astype(np.int64),
        "validation_node_index": evaluation_indices["validation"].astype(np.int64),
        "test_node_index": evaluation_indices["test"].astype(np.int64),
        "edge_index": graph.edge_index.astype(np.int64, copy=False),
        "edge_attributes_raw": graph.edge_attr.astype(np.float32, copy=False),
        "edge_attributes": standardized_edge_attr.astype(np.float32, copy=False),
        "train_edge_mask": training_edges.astype(bool, copy=False),
        "expression_mean": node_preprocessor.expression_mean_.astype(np.float64),
        "expression_scale": node_preprocessor.expression_scale_.astype(np.float64),
        "metadata_median": node_preprocessor.metadata_median_.astype(np.float64),
        "metadata_mean": node_preprocessor.metadata_mean_.astype(np.float64),
        "metadata_scale": node_preprocessor.metadata_scale_.astype(np.float64),
        "metadata_missing_indicator_indices": (
            node_preprocessor.missing_indicator_indices_.astype(np.int64)
        ),
        "edge_attribute_mean": edge_preprocessor.mean_.astype(np.float64),
        "edge_attribute_scale": edge_preprocessor.scale_.astype(np.float64),
    }
    split_graph_records: dict[str, Any] = {}
    for split_name, split_label, node_index in (
        ("train", "train", arrays["train_node_index"]),
        ("validation", "val", arrays["validation_node_index"]),
        ("test", "test", arrays["test_node_index"]),
    ):
        edge_mask = source_split == split_label
        global_to_local = np.full(dataset.n_cells, -1, dtype=np.int64)
        global_to_local[node_index] = np.arange(len(node_index), dtype=np.int64)
        local_edge_index = global_to_local[graph.edge_index[:, edge_mask]]
        if local_edge_index.size and (
            local_edge_index.min() < 0 or local_edge_index.max() >= len(node_index)
        ):
            raise ArtifactContractError(
                f"Failed to remap the independent {split_name} graph."
            )
        arrays[f"{split_name}_edge_index"] = local_edge_index.astype(
            np.int64, copy=False
        )
        arrays[f"{split_name}_edge_attributes_raw"] = graph.edge_attr[
            edge_mask
        ].astype(np.float32, copy=False)
        arrays[f"{split_name}_edge_attributes"] = standardized_edge_attr[
            edge_mask
        ].astype(np.float32, copy=False)
        split_graph_records[split_name] = {
            "n_nodes": int(len(node_index)),
            "n_directed_edges": int(edge_mask.sum()),
            "uses_local_node_indices": True,
        }

    split_counts = {
        name: int(np.sum(spatial_split.labels == label))
        for name, label in (("train", "train"), ("validation", "val"), ("test", "test"))
    }
    manifest: dict[str, Any] = {
        "format_version": PREPARED_ARTIFACT_FORMAT_VERSION,
        "artifact_kind": artifact_kind,
        "configuration": deepcopy(dict(config)),
        "selection": {
            "policy": selection.label_policy,
            "slide": selection.slide,
            "n_fovs": len(selection.fovs),
            "n_cells": dataset.n_cells,
            **selection_context,
            # Donor and tissue-unit numeric identifiers are intentionally absent.
            "restricted_identifiers_emitted": False,
        },
        "inputs": input_records,
        "features": {
            "n_biological_probes": dataset.n_genes,
            "gene_names": list(dataset.gene_names),
            "technical_control_prefixes_excluded": [
                "Negative",
                "SystemControl",
            ],
            "measured_metadata_names": list(ALLOWED_METADATA_COLUMNS),
            "model_covariate_names": list(nodes.metadata_names),
            "coordinates_are_model_covariates": False,
            "routing_keys_are_model_covariates": False,
            "qc_indicator_is_model_covariate": False,
            "qc_policy": data_config["qc_policy"],
            "n_qc_passed": int(dataset.qc_passed.sum()),
            "n_qc_not_passed": int((~dataset.qc_passed).sum()),
        },
        "split": {
            "split_id": spatial_split.split_id,
            "counts": split_counts,
            "n_macroblocks": int(np.unique(spatial_split.macroblock_ids).size),
            "fov_disjoint": True,
            "macroblock_disjoint": True,
            "assignment_precedes_fitted_preprocessing": True,
            "test_usage_during_preparation": "fixed mask generation only",
        },
        "preprocessing": {
            "node_fit_scope": "train nodes only",
            "edge_fit_scope": "train-to-train edges only",
            "n_training_nodes": node_preprocessor.n_train_,
            "n_training_edges": int(training_edges.sum()),
            "expression_transform": "gene-wise standardized log1p counts",
            "metadata_transform": (
                "median imputation, log1p, standardization"
                if node_preprocessor.log1p_metadata
                else "median imputation, standardization"
            ),
            "statistics_location": _DATA_FILENAME,
        },
        "graph": {
            "config": dict(graph.config),
            "edge_attribute_names": list(graph.edge_attr_names),
            "qc": graph.qc.to_dict(),
            "two_directed_edges_per_relation": True,
            "cross_split_edges": graph.qc.cross_group_edges,
            "split_graphs_constructed_independently": True,
            "split_graphs": split_graph_records,
        },
        "fixed_masks": {
            "directory": _MASK_DIRECTORY,
            "bundles": {
                name: {
                    "directory": f"{_MASK_DIRECTORY}/{name}",
                    "bundle_id": bundle.bundle_id,
                    "bundle_checksum": bundle.checksum,
                    "replicates": int(bundle.manifest["replicates"]),
                }
                for name, bundle in sorted(fixed_masks.items())
            },
            "splits": ["validation", "test"],
            "test_targets_evaluated": False,
        },
        "arrays": {
            name: {"shape": list(value.shape), "dtype": value.dtype.str}
            for name, value in sorted(arrays.items())
        },
        "provenance": _provenance(
            project_root,
            command=command,
            config_path=config_path,
        ),
    }
    _assert_inputs_unchanged(input_paths, input_records)
    return arrays, manifest


def prepare_artifact(
    config_path: str | Path,
    output_path: str | Path,
    *,
    command: Sequence[str] | None = None,
) -> Path:
    """Prepare and atomically publish one immutable artifact directory."""

    source = Path(config_path).resolve()
    config = load_prepare_config(source)
    project_root = _resolve_project_root(config, config_path=source)
    destination = Path(output_path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite immutable prepared artifact: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        arrays, manifest = _prepare_arrays_and_manifest(
            config,
            config_path=source,
            project_root=project_root,
            command=command,
            temporary_output=temporary,
        )
        data_path = temporary / _DATA_FILENAME
        np.savez_compressed(
            data_path,
            **{name: arrays[name] for name in sorted(arrays)},
        )
        non_manifest_files = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in _all_artifact_files(temporary)
        }
        manifest["files"] = non_manifest_files
        checksum = _manifest_checksum(manifest)
        manifest["manifest_content_sha256"] = checksum
        manifest["artifact_id"] = checksum[:16]
        _write_json(temporary / _MANIFEST_FILENAME, manifest)
        _write_checksum_file(temporary)
        _verify_artifact_files(temporary)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_prepared_artifact(
    path: str | Path,
    *,
    load_arrays: bool = True,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray] | None,
    dict[str, FixedMaskBundle],
]:
    """Verify and load a prepared artifact without pickle-enabled arrays."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared artifact directory was not found: {root}")
    _verify_artifact_files(root)
    manifest_path = root / _MANIFEST_FILENAME
    data_path = root / _DATA_FILENAME
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError("Prepared artifact lacks its NPZ or JSON manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != PREPARED_ARTIFACT_FORMAT_VERSION:
        raise ArtifactContractError("Unsupported prepared artifact format.")
    checksum = _manifest_checksum(manifest)
    if manifest.get("manifest_content_sha256") != checksum:
        raise ArtifactContractError("Prepared manifest content checksum mismatch.")
    if manifest.get("artifact_id") != checksum[:16]:
        raise ArtifactContractError("Prepared artifact ID does not match its content.")
    for relative, expected in manifest.get("files", {}).items():
        candidate = root / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ArtifactContractError(
                f"Prepared manifest file checksum mismatch for {relative}."
            )

    arrays: dict[str, np.ndarray] | None
    if load_arrays:
        with np.load(data_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        described = manifest.get("arrays", {})
        if set(arrays) != set(described):
            raise ArtifactContractError("Prepared NPZ arrays differ from the manifest.")
        for name, array in arrays.items():
            record = described[name]
            if list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]:
                raise ArtifactContractError(
                    f"Prepared NPZ array schema mismatch for {name}."
                )
    else:
        arrays = None
    masks: dict[str, FixedMaskBundle] = {}
    bundle_records = manifest["fixed_masks"].get("bundles", {})
    if set(bundle_records) != {"validation", "test"}:
        raise ArtifactContractError(
            "Prepared manifest must contain validation and test mask bundles."
        )
    for name, record in bundle_records.items():
        bundle = load_fixed_mask_bundle(root / record["directory"])
        if bundle.bundle_id != record["bundle_id"]:
            raise ArtifactContractError(
                f"Fixed-mask bundle ID differs for {name}."
            )
        masks[name] = bundle
    return manifest, arrays, masks
