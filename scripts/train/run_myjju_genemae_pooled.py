#!/usr/bin/env python3
"""Run one queue-owned pooled MyJJu GeneMAE campaign member.

The worker creates and owns the active :class:`RunArchive`; this subprocess
attaches to it and writes only scientific outputs.  Ten verified adjacent-
normal cores remain separate.  Each core is recursively tiled, and every tile
is visited exactly once per global epoch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.fingerprints import sha256_file  # noqa: E402
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.masking import (  # noqa: E402
    MaskSpec,
    create_fixed_mask_bundle,
)
from spatial_benchmark.myjju_genemae import (  # noqa: E402
    SOURCE_COMMIT,
    SOURCE_FILE_SHA256,
    build_symmetric_knn_graph,
    count_parameters,
    log1p_cp10k,
    make_source_model,
    masked_regression_metrics,
    permute_graph_node_labels,
    recursive_spatial_tiles,
    sample_entry_mask,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.pooled_full_core import (  # noqa: E402
    ANC_ALIASES,
    EXPECTED_N_GENES,
    EXPECTED_TOTAL_NODES,
    PooledCoreData,
    PooledFullCoreCohort,
    load_pooled_full_core_cohort,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    RunValidationError,
    deidentify_prediction_rows,
    verify_run_bundle,
)


# Bootstrap discovery is script-relative, while all runtime project paths honor
# BAGM_ROOT through the canonical resolver.
PROJECT_ROOT = current_paths().project_root


CAMPAIGN_ID = "cmp_20260730_myjju_genemae_10core_comparison"
CONTRACT_SHA256 = (
    "6f171b5bece63df943dd8fe679121fbdd41baf290339d64576dc6bfec847eada"
)
SOURCE_AUDIT_SHA256 = (
    "6f4eb7a557d5b488fe84d7b4f0d26d07f97dd6349f9cdd3203b12a426d6def91"
)
EXPECTED_PARAMETER_COUNT = 6_888_016
CONTRACT_RELATIVE = (
    Path("experiments/campaigns") / CAMPAIGN_ID / "frozen_task_contract.yaml"
)
CAMPAIGN_RELATIVE = Path("experiments/campaigns") / CAMPAIGN_ID / "campaign.yaml"
SOURCE_AUDIT_RELATIVE = (
    Path("experiments/campaigns") / CAMPAIGN_ID / "external_source_audit.yaml"
)
LOCKED_RELATIVE = Path("scratch/locked_campaigns") / CAMPAIGN_ID
MATERIALIZATION_RELATIVE = LOCKED_RELATIVE / "locked_config_materialization.json"
PILOT_GATE_RELATIVE = LOCKED_RELATIVE / "pilot_gate_receipt.json"
COMPARATOR_RECEIPT_RELATIVE = (
    Path("scratch/locked_campaigns")
    / "cmp_20260730_adjacent_normal_10core_pooled_hybrid_ensemble"
    / "locked_config_materialization.json"
)
COMPARATOR_RECEIPT_FILE_SHA256 = (
    "723ada09229530cf7f3b8f05e031173b4cc316c50324b041b56a03213c76aba1"
)
COMPARATOR_RECEIPT_CHECKSUM = (
    "d1a49b0428b1f90ec17f84581264fe4dc4ac3eca56526a0e29bcc761af02fdaa"
)
MATERIALIZATION_KIND = "myjju_genemae_locked_config_materialization_v1"
PILOT_GATE_KIND = "myjju_genemae_resource_pilot_gate_v1"
PRIMARY_METRIC = "fit/partial_gene/log1p_cp10k_masked_huber"
PROJECTED_CAMPAIGN_OUTPUT_BYTES = 8 * 1024**3
PILOT_MAX_VRAM_GIB = 20.5
PILOT_MAX_HOST_GIB = 40.0
PILOT_MAX_RUNTIME_HOURS = 6.0
PILOT_MIN_FINAL_FREE_DISK_GIB = 27.5
GRAPH_NULL_NAMESPACE = "myjju-genemae-node-label-permutation-v1"
NATIVE_MASK_NAMESPACE = "myjju-genemae-native-50-v1"
IMPLEMENTATION_FILES = {
    "model_module": "src/spatial_benchmark/myjju_genemae.py",
    "runner": "scripts/train/run_myjju_genemae_pooled.py",
    "materializer": "scripts/train/materialize_myjju_genemae_campaign.py",
}
PINNED_MODEL_MODULE_SHA256 = (
    "3348fdfb989fb85ad3598aae2b0bd3d9faf0b62aebdfa30aa6b4b1dc16d8b6a7"
)
REQUIRED_EVALUATION_METRIC_FIELDS = (
    "masked_huber",
    "masked_mse",
    "masked_mae",
    "pooled_pearson",
    "masked_r2",
    "gene_pearson_mean",
    "gene_pearson_median",
    "cell_pearson_mean",
    "n_masked",
)


class MyJJuRunnerError(RuntimeError):
    """Raised before execution can violate the frozen runner contract."""


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise MyJJuRunnerError(f"config.{name} must be a mapping")
    return value


def _mapping_value(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MyJJuRunnerError(f"{label} must be a mapping")
    return value


def _configured_mask_source(
    mask_sources: Mapping[str, Any],
    alias: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    configured = _mapping_value(
        mask_sources.get(alias), f"mask source {alias}"
    )
    common = _mapping_value(
        configured.get("common"), f"{alias} common mask identity"
    )
    native = _mapping_value(
        configured.get("native"), f"{alias} native mask identity"
    )
    base_seed = native.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise MyJJuRunnerError(
            f"{alias} native mask base_seed must be an integer"
        )
    return configured, common, base_seed


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63 - 1
    )


def seed_all(model_seed: int) -> None:
    """Seed Python, NumPy, CPU torch, and every visible CUDA generator."""

    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)


def _resolve_runtime_device(
    device: str | torch.device | None,
) -> torch.device:
    """Resolve a queue-visible CUDA device to an explicitly indexed device."""

    if device is None:
        if not torch.cuda.is_available():
            raise MyJJuRunnerError(
                "queue-owned MyJJu runs require a visible CUDA device"
            )
        return torch.device("cuda:0")
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda:0")
    return resolved


def final_epoch_index(completed_epochs: int) -> int:
    """Convert a positive completed-epoch count to its zero-based final index."""

    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or completed_epochs < 1
    ):
        raise ValueError("completed_epochs must be a positive integer")
    return completed_epochs - 1


def _array_sha256(label: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(label.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Return the canonical checkpoint digest used by replay and comparison."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    return digest.hexdigest()


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    return value


def _signed_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise MyJJuRunnerError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MyJJuRunnerError(f"{label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise MyJJuRunnerError(f"{label} must be a JSON mapping")
    core = dict(payload)
    checksum = core.pop("checksum", None)
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or canonical_sha256(core) != checksum
    ):
        raise MyJJuRunnerError(f"{label} checksum does not verify")
    return dict(payload), checksum


@dataclass(frozen=True, slots=True)
class SourceTile:
    alias: str
    tile_index: int
    node_indices: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    permuted_edge_index: np.ndarray
    node_indices_sha256: str
    edge_index_sha256: str
    edge_attr_sha256: str
    permuted_edge_index_sha256: str
    graph_sha256: str
    graph_null_seed: int

    @property
    def n_nodes(self) -> int:
        return int(self.node_indices.size)

    @property
    def n_directed_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def identity(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "tile_index": self.tile_index,
            "n_nodes": self.n_nodes,
            "n_directed_edges": self.n_directed_edges,
            "node_indices_sha256": self.node_indices_sha256,
            "edge_index_sha256": self.edge_index_sha256,
            "edge_attr_sha256": self.edge_attr_sha256,
            "graph_sha256": self.graph_sha256,
            "graph_null_seed": self.graph_null_seed,
            "permuted_edge_index_sha256": self.permuted_edge_index_sha256,
        }


@dataclass(frozen=True, slots=True)
class TiledCohort:
    aliases: tuple[str, ...]
    tiles: tuple[SourceTile, ...]
    k: int
    max_nodes: int

    def for_alias(self, alias: str) -> tuple[SourceTile, ...]:
        return tuple(tile for tile in self.tiles if tile.alias == alias)

    def identity(self) -> dict[str, Any]:
        cores: dict[str, Any] = {}
        for alias in self.aliases:
            selected = self.for_alias(alias)
            core = {
                "alias": alias,
                "tile_count": len(selected),
                "n_nodes": sum(tile.n_nodes for tile in selected),
                "n_directed_edges": sum(
                    tile.n_directed_edges for tile in selected
                ),
                "tiles": [tile.identity() for tile in selected],
            }
            core["graph_bundle_sha256"] = canonical_sha256(core)
            cores[alias] = core
        result = {
            "schema": "myjju_source_tiled_symmetric_knn_v1",
            "aliases": list(self.aliases),
            "k": self.k,
            "maximum_tile_nodes": self.max_nodes,
            "tiling": "recursive_median_longest_axis",
            "symmetry": "union",
            "edge_feature": "gaussian_distance_kernel",
            "edge_scale": "within_tile_median_edge_length",
            "constructed_graph_self_loops": False,
            "convolution_add_self_loops": True,
            "cross_core_edges": False,
            "cross_tile_edges": False,
            "cores": cores,
        }
        result["graph_bundle_sha256"] = canonical_sha256(result)
        return result


@dataclass(frozen=True, slots=True)
class EvaluationMask:
    label: str
    mask_rate: float
    replicate: int
    seed: int
    checksum: str
    mask: np.ndarray
    source_entry_id: str

    def identity(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mask_rate": self.mask_rate,
            "replicate": self.replicate,
            "seed": self.seed,
            "mask_checksum": self.checksum,
            "source_entry_id": self.source_entry_id,
            "shape": list(self.mask.shape),
            "n_masked": int(self.mask.sum(dtype=np.int64)),
        }


@dataclass(frozen=True, slots=True)
class CoreEvaluationMasks:
    alias: str
    common_source_bundle_checksum: str
    common_source_base_seed: int
    common: tuple[EvaluationMask, ...]
    native_bundle_checksum: str
    native_base_seed: int
    native: tuple[EvaluationMask, ...]

    @property
    def all_masks(self) -> tuple[EvaluationMask, ...]:
        return self.common + self.native

    def identity(self) -> dict[str, Any]:
        value = {
            "alias": self.alias,
            "common": {
                "mask_rate": 0.2,
                "source": (
                    "exact_regeneration_of_current_bagm_partial_gene_masks"
                ),
                "source_bundle_checksum": (
                    self.common_source_bundle_checksum
                ),
                "base_seed": self.common_source_base_seed,
                "entries": [entry.identity() for entry in self.common],
            },
            "native": {
                "mask_rate": 0.5,
                "source": NATIVE_MASK_NAMESPACE,
                "bundle_checksum": self.native_bundle_checksum,
                "base_seed": self.native_base_seed,
                "entries": [entry.identity() for entry in self.native],
            },
        }
        value["combined_checksum"] = canonical_sha256(value)
        return value


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Any
    tile_history: tuple[dict[str, Any], ...]
    epoch_history: tuple[dict[str, Any], ...]
    completed_epochs: int
    optimizer_steps: int
    all_gradients_finite: bool
    all_parameters_finite: bool
    duration_seconds: float
    state_dict_sha256: str


@dataclass(frozen=True, slots=True)
class MyJJuRunResult:
    run_id: str
    primary_metric_name: str
    primary_metric_value: float
    checkpoint_path: Path
    prediction_path: Path
    summary: Mapping[str, Any]


def prepare_source_tiles(
    cohort: PooledFullCoreCohort,
    *,
    k: int = 15,
    max_nodes: int = 7000,
) -> TiledCohort:
    """Build deterministic source-style graphs for each core and spatial tile."""

    if tuple(cohort.aliases) != ANC_ALIASES:
        raise MyJJuRunnerError("cohort aliases are not exact and ordered")
    result: list[SourceTile] = []
    for core in cohort.cores:
        partitions = recursive_spatial_tiles(
            core.coordinates_um, max_nodes=max_nodes
        )
        if not partitions:
            raise MyJJuRunnerError(f"{core.alias} produced no spatial tile")
        joined = np.concatenate(partitions)
        if (
            joined.size != core.n_nodes
            or np.unique(joined).size != core.n_nodes
            or not np.array_equal(
                np.sort(joined), np.arange(core.n_nodes, dtype=np.int64)
            )
        ):
            raise MyJJuRunnerError(
                f"{core.alias} spatial tiles are not an exact partition"
            )
        for tile_index, raw_indices in enumerate(partitions):
            indices = np.asarray(raw_indices, dtype=np.int64)
            local_coordinates = np.asarray(
                core.coordinates_um[indices], dtype=np.float64
            )
            edge_index, edge_attr = build_symmetric_knn_graph(
                local_coordinates, k=k
            )
            if edge_index.size and (
                int(edge_index.min()) < 0
                or int(edge_index.max()) >= indices.size
            ):
                raise MyJJuRunnerError(
                    f"{core.alias} tile {tile_index} has an invalid edge"
                )
            if edge_index.shape[1] != edge_attr.shape[0] or np.any(
                edge_index[0] == edge_index[1]
            ):
                raise MyJJuRunnerError(
                    f"{core.alias} tile {tile_index} graph is misaligned"
                )
            pairs = {tuple(pair) for pair in edge_index.T.tolist()}
            if any((target, source) not in pairs for source, target in pairs):
                raise MyJJuRunnerError(
                    f"{core.alias} tile {tile_index} graph is not symmetric"
                )
            null_seed = _stable_seed(
                GRAPH_NULL_NAMESPACE, core.alias, tile_index
            )
            permuted = np.asarray(
                permute_graph_node_labels(
                    edge_index, num_nodes=indices.size, seed=null_seed
                ),
                dtype=np.int64,
            )
            indices_sha = _array_sha256("node_indices", indices)
            edge_sha = _array_sha256("edge_index", edge_index)
            attr_sha = _array_sha256("edge_attr", edge_attr)
            permuted_sha = _array_sha256(
                "permuted_edge_index", permuted
            )
            graph_sha = canonical_sha256(
                {
                    "alias": core.alias,
                    "tile_index": tile_index,
                    "node_indices_sha256": indices_sha,
                    "edge_index_sha256": edge_sha,
                    "edge_attr_sha256": attr_sha,
                    "constructed_graph_self_loops": False,
                    "convolution_add_self_loops": True,
                }
            )
            result.append(
                SourceTile(
                    alias=core.alias,
                    tile_index=tile_index,
                    node_indices=indices,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    permuted_edge_index=permuted,
                    node_indices_sha256=indices_sha,
                    edge_index_sha256=edge_sha,
                    edge_attr_sha256=attr_sha,
                    permuted_edge_index_sha256=permuted_sha,
                    graph_sha256=graph_sha,
                    graph_null_seed=null_seed,
                )
            )
    tiled = TiledCohort(
        aliases=tuple(cohort.aliases),
        tiles=tuple(result),
        k=int(k),
        max_nodes=int(max_nodes),
    )
    if sum(tile.n_nodes for tile in tiled.tiles) != EXPECTED_TOTAL_NODES:
        raise MyJJuRunnerError("tiled graph cohort cell count changed")
    return tiled


def _common_specs() -> list[MaskSpec]:
    return [
        MaskSpec(
            mode="partial",
            partial_gene_rate=0.2,
            node_rate=0.1,
            block_node_rate=0.1,
            block_width_um=None,
            block_shape="disk",
            label="partial_gene",
        ),
        MaskSpec(
            mode="node",
            partial_gene_rate=0.2,
            node_rate=0.1,
            block_node_rate=0.1,
            block_width_um=None,
            block_shape="disk",
            label="whole_node",
        ),
        MaskSpec(
            mode="block",
            partial_gene_rate=0.2,
            node_rate=0.1,
            block_node_rate=0.1,
            block_width_um=None,
            block_shape="disk",
            label="spatial_block",
        ),
    ]


def regenerate_evaluation_masks(
    core: PooledCoreData,
    *,
    comparator_source: Mapping[str, Any],
    native_base_seed: int | None = None,
) -> CoreEvaluationMasks:
    """Regenerate exact BAGM 20% masks and fixed source-native 50% masks."""

    try:
        common_base_seed = int(comparator_source["base_seed"])
        expected_bundle = str(
            comparator_source.get(
                "source_bundle_checksum",
                comparator_source.get("bundle_checksum"),
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MyJJuRunnerError(
            f"{core.alias} comparator mask source is malformed"
        ) from exc
    common_bundle = create_fixed_mask_bundle(
        {"fit": core.coordinates_um},
        core.n_genes,
        _common_specs(),
        replicates=3,
        base_seed=common_base_seed,
    )
    if common_bundle.checksum != expected_bundle:
        raise MyJJuRunnerError(
            f"{core.alias} common BAGM mask bundle checksum changed"
        )
    expected_entries: dict[int, Mapping[str, Any]] = {}
    for item in comparator_source.get("entries", ()):
        mode = str(item.get("mode", ""))
        label = str(item.get("label", ""))
        if mode == "partial_gene" or label == "common_20":
            expected_entries[int(item.get("replicate"))] = item
    common: list[EvaluationMask] = []
    for entry in common_bundle.manifest["entries"]:
        spec = entry["spec"]
        if spec["label"] != "partial_gene":
            continue
        replicate = int(entry["replicate"])
        expected = expected_entries.get(replicate)
        if (
            expected is None
            or expected.get(
                "entry_id", expected.get("source_entry_id")
            )
            != entry["entry_id"]
            or expected.get("seed") != entry["seed"]
            or expected.get("mask_checksum") != entry["mask_checksum"]
        ):
            raise MyJJuRunnerError(
                f"{core.alias} BAGM partial mask identity changed"
            )
        mask = np.asarray(
            common_bundle.masks[str(entry["entry_id"])], dtype=bool
        )
        common.append(
            EvaluationMask(
                label="common_20",
                mask_rate=0.2,
                replicate=replicate,
                seed=int(entry["seed"]),
                checksum=str(entry["mask_checksum"]),
                mask=mask,
                source_entry_id=str(entry["entry_id"]),
            )
        )
    if {entry.replicate for entry in common} != {0, 1, 2}:
        raise MyJJuRunnerError(
            f"{core.alias} common partial masks are incomplete"
        )

    base_seed = (
        _stable_seed(NATIVE_MASK_NAMESPACE, core.alias)
        if native_base_seed is None
        else int(native_base_seed)
    )
    native_spec = MaskSpec(
        mode="partial",
        partial_gene_rate=0.5,
        node_rate=0.1,
        block_node_rate=0.1,
        block_width_um=None,
        block_shape="disk",
        label="native_partial_gene_50",
    )
    native_bundle = create_fixed_mask_bundle(
        {"fit": core.coordinates_um},
        core.n_genes,
        [native_spec],
        replicates=3,
        base_seed=base_seed,
    )
    native: list[EvaluationMask] = []
    for entry in native_bundle.manifest["entries"]:
        mask = np.asarray(
            native_bundle.masks[str(entry["entry_id"])], dtype=bool
        )
        native.append(
            EvaluationMask(
                label="native_50",
                mask_rate=0.5,
                replicate=int(entry["replicate"]),
                seed=int(entry["seed"]),
                checksum=str(entry["mask_checksum"]),
                mask=mask,
                source_entry_id=str(entry["entry_id"]),
            )
        )
    if {entry.replicate for entry in native} != {0, 1, 2}:
        raise MyJJuRunnerError(
            f"{core.alias} native partial masks are incomplete"
        )
    return CoreEvaluationMasks(
        alias=core.alias,
        common_source_bundle_checksum=common_bundle.checksum,
        common_source_base_seed=common_base_seed,
        common=tuple(sorted(common, key=lambda item: item.replicate)),
        native_bundle_checksum=native_bundle.checksum,
        native_base_seed=base_seed,
        native=tuple(sorted(native, key=lambda item: item.replicate)),
    )


def load_verified_ten_core_cohort(
    config: Mapping[str, Any],
) -> PooledFullCoreCohort:
    """Load only the alias-safe, checksum-verified ten-core prepared cohort."""

    dataset = _section(config, "dataset")
    raw = dataset.get("prepared_artifacts")
    if not isinstance(raw, Mapping) or tuple(raw) != ANC_ALIASES:
        raise MyJJuRunnerError(
            "dataset.prepared_artifacts must contain ordered ANC aliases"
        )
    inputs: list[tuple[str, Path]] = []
    root = current_paths().project_root.resolve()
    for alias in ANC_ALIASES:
        path = Path(str(raw[alias]))
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise MyJJuRunnerError(
                f"{alias} prepared artifact escapes project root"
            ) from exc
        inputs.append((alias, resolved))
    cohort = load_pooled_full_core_cohort(tuple(inputs))
    if (
        cohort.total_nodes != EXPECTED_TOTAL_NODES
        or cohort.n_genes != EXPECTED_N_GENES
        or cohort.fingerprint_sha256 != dataset.get("dataset_fingerprint")
        or cohort.checksums.ordered_gene_schema_sha256
        != dataset.get("ordered_gene_schema_sha256")
        or any(
            core.preprocessing_qc.protected_identifier_arrays_returned
            for core in cohort.cores
        )
    ):
        raise MyJJuRunnerError("loaded ten-core cohort identity changed")
    return cohort


def _verify_frozen_files() -> dict[str, Any]:
    contract = PROJECT_ROOT / CONTRACT_RELATIVE
    campaign = PROJECT_ROOT / CAMPAIGN_RELATIVE
    source_audit = PROJECT_ROOT / SOURCE_AUDIT_RELATIVE
    if (
        sha256_file(contract) != CONTRACT_SHA256
        or sha256_file(source_audit) != SOURCE_AUDIT_SHA256
    ):
        raise MyJJuRunnerError(
            "frozen task contract or source audit checksum changed"
        )
    declaration = load_yaml_mapping(campaign)
    audit = load_yaml_mapping(source_audit)
    if (
        declaration.get("campaign_id") != CAMPAIGN_ID
        or declaration.get("frozen_contract_sha256") != CONTRACT_SHA256
    ):
        raise MyJJuRunnerError("campaign declaration checksum binding changed")
    source_files = {
        str(record["relative_path"]): str(record["sha256"])
        for record in _section(audit, "source_files").values()
    }
    if (
        audit.get("repository", {}).get("commit") != SOURCE_COMMIT
        or any(
            source_files.get(relative) != checksum
            for relative, checksum in SOURCE_FILE_SHA256.items()
        )
    ):
        raise MyJJuRunnerError("source audit disagrees with reproduced module")
    return {
        "contract_reference": CONTRACT_RELATIVE.as_posix(),
        "contract_sha256": CONTRACT_SHA256,
        "source_audit_reference": SOURCE_AUDIT_RELATIVE.as_posix(),
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "external_source_commit": SOURCE_COMMIT,
        "external_source_file_sha256": dict(SOURCE_FILE_SHA256),
        "implementation_file_sha256": {
            label: {
                "reference": reference,
                "sha256": sha256_file(PROJECT_ROOT / reference),
            }
            for label, reference in IMPLEMENTATION_FILES.items()
        },
        "verified": True,
    }


def _materialization_and_config(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    metadata = _section(config, "metadata")
    if metadata.get("locked_config_materialization_receipt") != (
        MATERIALIZATION_RELATIVE.as_posix()
    ):
        raise MyJJuRunnerError("materialization receipt reference changed")
    payload, checksum = _signed_json(
        PROJECT_ROOT / MATERIALIZATION_RELATIVE,
        label="MyJJu materialization receipt",
    )
    if (
        payload.get("receipt_kind") != MATERIALIZATION_KIND
        or payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("frozen_contract", {}).get("sha256")
        != CONTRACT_SHA256
        or payload.get("parameter_audit", {}).get(
            "trainable_parameter_count"
        )
        != EXPECTED_PARAMETER_COUNT
    ):
        raise MyJJuRunnerError("materialization receipt identity changed")
    implementation = payload.get("implementation_file_sha256")
    if not isinstance(implementation, Mapping):
        raise MyJJuRunnerError(
            "materialization omits implementation semantic checksums"
        )
    for label, reference in IMPLEMENTATION_FILES.items():
        record = implementation.get(label)
        if (
            not isinstance(record, Mapping)
            or record.get("reference") != reference
            or record.get("sha256")
            != sha256_file(PROJECT_ROOT / reference)
        ):
            raise MyJJuRunnerError(
                f"materialized implementation checksum changed for {label}"
            )
    if (
        implementation["model_module"].get("sha256")
        != PINNED_MODEL_MODULE_SHA256
    ):
        raise MyJJuRunnerError(
            "pinned MyJJu compatibility module checksum changed"
        )
    comparator = payload.get("comparator_materialization")
    comparator_path = PROJECT_ROOT / COMPARATOR_RECEIPT_RELATIVE
    _, comparator_checksum = _signed_json(
        comparator_path, label="pinned BAGM comparator materialization"
    )
    if (
        not isinstance(comparator, Mapping)
        or comparator.get("reference")
        != COMPARATOR_RECEIPT_RELATIVE.as_posix()
        or comparator.get("file_sha256")
        != COMPARATOR_RECEIPT_FILE_SHA256
        or comparator.get("canonical_checksum")
        != COMPARATOR_RECEIPT_CHECKSUM
        or sha256_file(comparator_path)
        != COMPARATOR_RECEIPT_FILE_SHA256
        or comparator_checksum != COMPARATOR_RECEIPT_CHECKSUM
    ):
        raise MyJJuRunnerError(
            "pinned BAGM comparator materialization changed"
        )
    role = str(metadata.get("execution_role"))
    seed = int(config.get("seed", -1))
    matches = [
        record
        for record in payload.get("jobs", ())
        if record.get("role")
        == ("pilot" if role == "resource_pilot" else role)
        and record.get("seed") == seed
    ]
    if len(matches) != 1:
        raise MyJJuRunnerError("materialized config slot is not unique")
    source_path = PROJECT_ROOT / str(matches[0]["config"])
    if (
        not source_path.is_file()
        or sha256_file(source_path) != matches[0].get("file_sha256")
    ):
        raise MyJJuRunnerError("materialized source config checksum changed")
    source_config = load_yaml_mapping(source_path)
    runtime = dict(config)
    source = dict(source_config)
    # Retry attempts are an execution dimension inserted by the worker.
    runtime["attempt"] = 1
    source["attempt"] = 1
    if canonical_sha256(runtime) != canonical_sha256(source):
        raise MyJJuRunnerError(
            "worker configuration differs from the locked materialization"
        )
    return payload, checksum


def _verify_production_gate(
    config: Mapping[str, Any],
    *,
    materialization_checksum: str,
) -> Mapping[str, Any] | None:
    role = _section(config, "metadata").get("execution_role")
    if role == "resource_pilot":
        return None
    payload, _ = _signed_json(
        PROJECT_ROOT / PILOT_GATE_RELATIVE,
        label="MyJJu pilot gate receipt",
    )
    if (
        payload.get("receipt_kind") != PILOT_GATE_KIND
        or payload.get("campaign_id") != CAMPAIGN_ID
        or payload.get("materialization_checksum")
        != materialization_checksum
        or payload.get("gate_passed") is not True
    ):
        raise MyJJuRunnerError("production pilot gate is absent or invalid")
    artifact = PROJECT_ROOT / str(payload.get("pilot_artifact_reference", ""))
    verification = verify_run_bundle(artifact, require_success_contract=True)
    if (
        verification.get("status") != "success"
        or sha256_file(artifact / "_SUCCESS")
        != payload.get("success_marker_sha256")
        or sha256_file(artifact / "diagnostics/resource.json")
        != payload.get("resource_diagnostic_sha256")
        or sha256_file(artifact / "checkpoints/last.ckpt")
        != payload.get("checkpoint_sha256")
    ):
        raise MyJJuRunnerError("pilot gate source artifact changed")
    return payload


def _validate_config(config: Mapping[str, Any]) -> tuple[bool, int]:
    validate_experiment_config(config)
    campaign = _section(config, "campaign")
    model = _section(config, "model")
    graph = _section(config, "graph")
    trainer = _section(config, "trainer")
    evaluation = _section(config, "evaluation")
    metadata = _section(config, "metadata")
    seed = int(config.get("seed", -1))
    diagnostic = metadata.get("execution_role") == "resource_pilot"
    expected_epochs = 2 if diagnostic else 200
    expected = {
        "campaign.campaign_id": (
            campaign.get("campaign_id"),
            CAMPAIGN_ID,
        ),
        "campaign.frozen_contract_sha256": (
            campaign.get("frozen_contract_sha256"),
            CONTRACT_SHA256,
        ),
        "model.name": (model.get("name"), "myjju-genemae"),
        "model.family": (
            model.get("family"),
            "myjju_dual_path_genemae",
        ),
        "model.expected_trainable_parameters": (
            model.get("expected_trainable_parameters"),
            EXPECTED_PARAMETER_COUNT,
        ),
        "graph.kind": (
            graph.get("kind"),
            "symmetric_spatial_knn_tiled",
        ),
        "graph.neighbor_k": (graph.get("neighbor_k"), 15),
        "graph.symmetry": (graph.get("symmetry"), "union"),
        "graph.maximum_tile_nodes": (
            graph.get("maximum_tile_nodes"),
            7000,
        ),
        "graph.constructed_graph_self_loops": (
            graph.get("constructed_graph_self_loops"),
            False,
        ),
        "graph.convolution_add_self_loops": (
            graph.get("convolution_add_self_loops"),
            True,
        ),
        "trainer.optimizer": (trainer.get("optimizer"), "AdamW"),
        "trainer.learning_rate": (trainer.get("learning_rate"), 0.0015),
        "trainer.weight_decay": (trainer.get("weight_decay"), 0.0001),
        "trainer.gradient_clip_norm": (
            trainer.get("gradient_clip_norm"),
            5.0,
        ),
        "trainer.max_epochs": (trainer.get("max_epochs"), expected_epochs),
        "trainer.train_mask_rate": (
            trainer.get("train_mask_rate"),
            0.5,
        ),
        "trainer.precision": (trainer.get("precision"), "fp32"),
        "trainer.amp": (trainer.get("amp"), False),
        "trainer.restore_best": (trainer.get("restore_best"), False),
        "trainer.primary_checkpoint_role": (
            trainer.get("primary_checkpoint_role"),
            "last",
        ),
        "trainer.checkpoint_policy": (
            trainer.get("checkpoint_policy"),
            "last_only",
        ),
        "evaluation.primary_metric": (
            evaluation.get("primary_metric"),
            PRIMARY_METRIC,
        ),
        "evaluation.common_mask_rate": (
            evaluation.get("common_mask_rate"),
            0.2,
        ),
        "evaluation.native_mask_rate": (
            evaluation.get("native_mask_rate"),
            0.5,
        ),
        "evaluation.mask_replicates_per_rate": (
            evaluation.get("mask_replicates_per_rate"),
            3,
        ),
        "evaluation.graph_conditions": (
            evaluation.get("graph_conditions"),
            ["observed", "node_label_permuted"],
        ),
        "evaluation.diagnostic_only": (
            evaluation.get("diagnostic_only"),
            diagnostic,
        ),
    }
    drift = [
        f"{field}: observed={observed!r}, expected={required!r}"
        for field, (observed, required) in expected.items()
        if observed != required
    ]
    if drift:
        raise MyJJuRunnerError("locked configuration drift: " + "; ".join(drift))
    if seed not in range(7) or (diagnostic and seed != 0):
        raise MyJJuRunnerError("model seed is outside the frozen slots")
    return diagnostic, seed


def _normalised_expression(
    cohort: PooledFullCoreCohort,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for core in cohort.cores:
        value = np.asarray(log1p_cp10k(core.expression_counts), dtype=np.float32)
        if value.shape != (core.n_nodes, EXPECTED_N_GENES) or not np.isfinite(
            value
        ).all():
            raise MyJJuRunnerError(
                f"{core.alias} CP10k transformation is invalid"
            )
        result[core.alias] = value
    return result


def train_source_model(
    *,
    model: torch.nn.Module,
    cohort: PooledFullCoreCohort,
    tiled: TiledCohort,
    normalized_expression: Mapping[str, np.ndarray],
    trainer: Mapping[str, Any],
    model_seed: int,
    device: str | torch.device,
) -> TrainingResult:
    """Train the shared model with one deterministic visit per tile and epoch."""

    seed_all(model_seed)
    target_device = torch.device(device)
    model = model.to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trainer["learning_rate"]),
        weight_decay=float(trainer["weight_decay"]),
    )
    epochs = int(trainer["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    tile_history: list[dict[str, Any]] = []
    epoch_history: list[dict[str, Any]] = []
    step = 0
    gradients_finite = True
    parameters_finite = True
    started = time.monotonic()
    alias_to_core = {core.alias: core for core in cohort.cores}
    for epoch_index in range(epochs):
        model.train()
        order_rng = np.random.default_rng(
            _stable_seed("tile-order-v1", model_seed, epoch_index)
        )
        order = order_rng.permutation(len(tiled.tiles))
        seen: list[int] = []
        epoch_losses: list[float] = []
        epoch_started = time.monotonic()
        for order_index, tile_position in enumerate(order.tolist()):
            tile = tiled.tiles[int(tile_position)]
            seen.append(int(tile_position))
            core = alias_to_core[tile.alias]
            values = torch.from_numpy(
                normalized_expression[tile.alias][tile.node_indices]
            ).to(target_device)
            edges = torch.from_numpy(tile.edge_index).to(target_device)
            edge_attr = torch.from_numpy(tile.edge_attr).to(target_device)
            mask_seed = _stable_seed(
                "training-mask-v1",
                model_seed,
                epoch_index,
                tile.alias,
                tile.tile_index,
            )
            mask = sample_entry_mask(
                values.shape,
                rate=float(trainer["train_mask_rate"]),
                seed=mask_seed,
                device=target_device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(
                values,
                edges,
                edge_attr=edge_attr,
                entry_mask=mask,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch_index + 1}, "
                    f"tile {tile.alias}/{tile.tile_index}"
                )
            loss.backward()
            current_gradients_finite = all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            gradients_finite &= current_gradients_finite
            if not current_gradients_finite:
                raise FloatingPointError(
                    f"non-finite gradient at epoch {epoch_index + 1}"
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(trainer["gradient_clip_norm"]),
            )
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(
                    f"non-finite gradient norm at epoch {epoch_index + 1}"
                )
            optimizer.step()
            current_parameters_finite = all(
                bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()
            )
            parameters_finite &= current_parameters_finite
            if not current_parameters_finite:
                raise FloatingPointError(
                    f"non-finite parameter at epoch {epoch_index + 1}"
                )
            step += 1
            loss_value = float(loss.detach().cpu())
            epoch_losses.append(loss_value)
            tile_history.append(
                {
                    "epoch_number": epoch_index + 1,
                    "global_step": step,
                    "epoch_order_index": order_index,
                    "tile_position": int(tile_position),
                    "biological_unit_alias": tile.alias,
                    "tile_index": tile.tile_index,
                    "n_nodes": tile.n_nodes,
                    "n_directed_edges": tile.n_directed_edges,
                    "training_mask_seed": mask_seed,
                    "training_mask_rate_observed": float(
                        mask.float().mean().cpu()
                    ),
                    "masked_huber": loss_value,
                    "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            del values, edges, edge_attr, mask, loss
        if (
            len(seen) != len(tiled.tiles)
            or len(set(seen)) != len(tiled.tiles)
            or set(seen) != set(range(len(tiled.tiles)))
        ):
            raise MyJJuRunnerError(
                f"epoch {epoch_index + 1} did not visit every tile once"
            )
        scheduler.step()
        epoch_history.append(
            {
                "epoch_number": epoch_index + 1,
                "tiles_visited": len(seen),
                "optimizer_steps": len(seen),
                "masked_huber_mean": float(np.mean(epoch_losses)),
                "masked_huber_min": float(np.min(epoch_losses)),
                "masked_huber_max": float(np.max(epoch_losses)),
                "duration_seconds": time.monotonic() - epoch_started,
                "ordered_tile_positions_sha256": canonical_sha256(seen),
                "every_tile_once": True,
            }
        )
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    return TrainingResult(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        tile_history=tuple(tile_history),
        epoch_history=tuple(epoch_history),
        completed_epochs=epochs,
        optimizer_steps=step,
        all_gradients_finite=gradients_finite,
        all_parameters_finite=parameters_finite,
        duration_seconds=time.monotonic() - started,
        state_dict_sha256=state_dict_sha256(state),
    )


@torch.inference_mode()
def predict_core_mask(
    model: torch.nn.Module,
    core: PooledCoreData,
    tiles: Sequence[SourceTile],
    entry_mask: np.ndarray,
    *,
    device: str | torch.device,
    graph_condition: str = "observed",
    normalized_expression: np.ndarray | None = None,
) -> np.ndarray:
    """Return raw full ``[nodes, genes]`` normalized-log predictions.

    This is the built-in prediction provider used by ensemble comparison.  The
    returned row order is the verified prepared-core row order.  Callers must
    retain the supplied mask and target separately; unmasked predictions are
    not part of the partial-gene estimand.
    """

    if graph_condition not in {"observed", "node_label_permuted"}:
        raise ValueError("graph_condition must be observed or node_label_permuted")
    mask = np.asarray(entry_mask)
    if mask.dtype != np.bool_ or mask.shape != (
        core.n_nodes,
        core.n_genes,
    ):
        raise ValueError("entry_mask must be a full-core boolean matrix")
    target = (
        np.asarray(log1p_cp10k(core.expression_counts), dtype=np.float32)
        if normalized_expression is None
        else np.asarray(normalized_expression, dtype=np.float32)
    )
    if target.shape != mask.shape or not np.isfinite(target).all():
        raise ValueError("normalized_expression is invalid")
    selected_tiles = tuple(tiles)
    joined = np.concatenate([tile.node_indices for tile in selected_tiles])
    if (
        not selected_tiles
        or joined.size != core.n_nodes
        or np.unique(joined).size != core.n_nodes
        or not np.array_equal(np.sort(joined), np.arange(core.n_nodes))
    ):
        raise ValueError("tiles are not an exact core partition")
    output = np.empty_like(target, dtype=np.float32)
    target_device = torch.device(device)
    model.eval()
    for tile in selected_tiles:
        if tile.alias != core.alias:
            raise ValueError("tile alias does not match the core")
        indices = tile.node_indices
        values = torch.from_numpy(target[indices]).to(target_device)
        local_mask = torch.from_numpy(mask[indices]).to(target_device)
        edge_array = (
            tile.edge_index
            if graph_condition == "observed"
            else tile.permuted_edge_index
        )
        edges = torch.from_numpy(edge_array).to(target_device)
        edge_attr = torch.from_numpy(tile.edge_attr).to(target_device)
        reconstruction, returned_mask = model(
            values,
            edges,
            edge_attr=edge_attr,
            entry_mask=local_mask,
        )
        if not torch.equal(returned_mask, local_mask) or not bool(
            torch.isfinite(reconstruction).all()
        ):
            raise FloatingPointError(
                f"{core.alias} {graph_condition} inference is invalid"
            )
        output[indices] = reconstruction.detach().cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError(
            f"{core.alias} {graph_condition} output is non-finite"
        )
    return output


def evaluate_model_on_masks(
    model: torch.nn.Module,
    core: PooledCoreData,
    tiles: Sequence[SourceTile],
    masks: CoreEvaluationMasks,
    *,
    device: str | torch.device,
    normalized_expression: np.ndarray | None = None,
    prediction_callback: (
        Any | None
    ) = None,
) -> list[dict[str, Any]]:
    """Evaluate both graph conditions on all fixed 20% and 50% masks.

    ``prediction_callback`` receives ``(entry, condition, target, mask,
    prediction)`` before the temporary full prediction is released.  This
    supports exact seven-model prediction ensembling without requiring each
    member run to persist hundreds of millions of row-level values.
    """

    target = (
        np.asarray(log1p_cp10k(core.expression_counts), dtype=np.float32)
        if normalized_expression is None
        else np.asarray(normalized_expression, dtype=np.float32)
    )
    rows: list[dict[str, Any]] = []
    for entry in masks.all_masks:
        for condition in ("observed", "node_label_permuted"):
            started = time.monotonic()
            prediction = predict_core_mask(
                model,
                core,
                tiles,
                entry.mask,
                device=device,
                graph_condition=condition,
                normalized_expression=target,
            )
            metrics = masked_regression_metrics(
                target, prediction, entry.mask
            )
            invalid = [
                name
                for name in REQUIRED_EVALUATION_METRIC_FIELDS
                if metrics.get(name) is None
                or not math.isfinite(float(metrics[name]))
            ]
            if invalid:
                raise FloatingPointError(
                    f"{core.alias} {entry.label} replicate "
                    f"{entry.replicate} {condition} has non-finite required "
                    f"metrics: {', '.join(invalid)}"
                )
            if prediction_callback is not None:
                prediction_callback(
                    entry,
                    condition,
                    target,
                    entry.mask,
                    prediction,
                )
            rows.append(
                {
                    "biological_unit_alias": core.alias,
                    "mask_label": entry.label,
                    "mask_rate": entry.mask_rate,
                    "mask_replicate": entry.replicate,
                    "mask_seed": entry.seed,
                    "mask_checksum": entry.checksum,
                    "graph_condition": condition,
                    "duration_seconds": time.monotonic() - started,
                    **metrics,
                }
            )
            del prediction
    return rows


def reconstruct_model_from_checkpoint(
    checkpoint: str | Path | Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Strictly reconstruct a source-fidelity model from a final checkpoint."""

    if isinstance(checkpoint, Mapping):
        payload = dict(checkpoint)
    else:
        payload = torch.load(
            Path(checkpoint), map_location="cpu", weights_only=False
        )
    if (
        not isinstance(payload, Mapping)
        or payload.get("model_name") != "myjju-genemae"
        or payload.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or payload.get("external_source_commit") != SOURCE_COMMIT
    ):
        raise MyJJuRunnerError("checkpoint model/source identity is invalid")
    implementation = payload.get("implementation_file_sha256")
    if (
        not isinstance(implementation, Mapping)
        or not isinstance(implementation.get("model_module"), Mapping)
        or implementation["model_module"].get("reference")
        != IMPLEMENTATION_FILES["model_module"]
        or implementation["model_module"].get("sha256")
        != PINNED_MODEL_MODULE_SHA256
        or sha256_file(PROJECT_ROOT / IMPLEMENTATION_FILES["model_module"])
        != PINNED_MODEL_MODULE_SHA256
    ):
        raise MyJJuRunnerError(
            "checkpoint compatibility-module identity is invalid"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise MyJJuRunnerError("checkpoint model_state_dict is missing")
    if state_dict_sha256(state) != payload.get("state_dict_sha256"):
        raise MyJJuRunnerError("checkpoint state_dict checksum changed")
    model = make_source_model(num_genes=EXPECTED_N_GENES, mask_rate=0.5)
    model.load_state_dict(state, strict=True)
    if count_parameters(model) != EXPECTED_PARAMETER_COUNT:
        raise MyJJuRunnerError("reconstructed model parameter count changed")
    model.to(torch.device(device))
    model.eval()
    return model


@torch.inference_mode()
def verify_checkpoint_replay(
    *,
    original_model: torch.nn.Module,
    reloaded_model: torch.nn.Module,
    core: PooledCoreData,
    tile: SourceTile,
    normalized_expression: np.ndarray,
    device: str | torch.device,
    absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare one prespecified fixed-mask forward pass after strict reload."""

    target_device = torch.device(device)
    values = torch.from_numpy(
        np.asarray(
            normalized_expression[tile.node_indices], dtype=np.float32
        )
    ).to(target_device)
    edges = torch.from_numpy(tile.edge_index).to(target_device)
    edge_attr = torch.from_numpy(tile.edge_attr).to(target_device)
    seed = _stable_seed(
        "checkpoint-replay-mask-v1", core.alias, tile.tile_index
    )
    mask = sample_entry_mask(
        values.shape, rate=0.2, seed=seed, device=target_device
    )
    original_model.eval()
    reloaded_model.eval()
    original_state_sha256 = state_dict_sha256(original_model.state_dict())
    reloaded_state_sha256 = state_dict_sha256(reloaded_model.state_dict())
    exact_state_dict = original_state_sha256 == reloaded_state_sha256
    original, original_mask = original_model(
        values, edges, edge_attr=edge_attr, entry_mask=mask
    )
    reloaded, reloaded_mask = reloaded_model(
        values, edges, edge_attr=edge_attr, entry_mask=mask
    )
    maximum_difference = float(
        torch.max(torch.abs(original - reloaded)).detach().cpu()
    )
    exact_masks = torch.equal(original_mask, reloaded_mask) and torch.equal(
        original_mask, mask
    )
    finite = bool(torch.isfinite(original).all()) and bool(
        torch.isfinite(reloaded).all()
    )
    passed = (
        exact_state_dict
        and target_device.type == "cpu"
        and exact_masks
        and finite
        and maximum_difference <= float(absolute_tolerance)
    )
    payload = {
        "schema_version": 1,
        "alias": core.alias,
        "tile_index": tile.tile_index,
        "mask_seed": seed,
        "mask_sha256": _array_sha256(
            "checkpoint_replay_mask", mask.detach().cpu().numpy()
        ),
        "original_prediction_sha256": _array_sha256(
            "checkpoint_replay_prediction",
            original.detach().cpu().numpy(),
        ),
        "reloaded_prediction_sha256": _array_sha256(
            "checkpoint_replay_prediction",
            reloaded.detach().cpu().numpy(),
        ),
        "maximum_absolute_difference": maximum_difference,
        "absolute_tolerance": float(absolute_tolerance),
        "execution_device": str(target_device),
        "deterministic_cpu_replay": target_device.type == "cpu",
        "original_state_dict_sha256": original_state_sha256,
        "reloaded_state_dict_sha256": reloaded_state_sha256,
        "state_dict_exact": exact_state_dict,
        "returned_masks_exact": exact_masks,
        "predictions_finite": finite,
        "replay_passed": passed,
    }
    if not passed:
        raise MyJJuRunnerError(
            "strictly reloaded final checkpoint failed fixed replay"
        )
    return payload


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_fields = REQUIRED_EVALUATION_METRIC_FIELDS
    prefixes = {
        ("common_20", "observed"): "fit/partial_gene",
        ("native_50", "observed"): "fit/native_partial_gene",
        (
            "common_20",
            "node_label_permuted",
        ): "fit/partial_gene_node_label_permuted",
        (
            "native_50",
            "node_label_permuted",
        ): "fit/native_partial_gene_node_label_permuted",
    }
    result: dict[str, Any] = {}
    for (label, condition), prefix in prefixes.items():
        selected = [
            row
            for row in rows
            if row["mask_label"] == label
            and row["graph_condition"] == condition
        ]
        for alias in ANC_ALIASES:
            alias_key = alias.lower().replace("-", "_")
            core_rows = [
                row
                for row in selected
                if row["biological_unit_alias"] == alias
            ]
            if len(core_rows) != 3:
                raise MyJJuRunnerError(
                    f"expected three rows for {alias}/{label}/{condition}"
                )
            for field in metric_fields:
                values = [
                    float(row[field])
                    for row in core_rows
                    if row[field] is not None
                    and math.isfinite(float(row[field]))
                ]
                result[
                    f"{prefix}/{alias_key}/log1p_cp10k_{field}"
                ] = (
                    float(np.mean(values)) if values else None
                )
        for field in metric_fields:
            values = [
                result[
                    f"{prefix}/{alias.lower().replace('-', '_')}/"
                    f"log1p_cp10k_{field}"
                ]
                for alias in ANC_ALIASES
            ]
            finite = [
                float(value)
                for value in values
                if value is not None and math.isfinite(float(value))
            ]
            result[f"{prefix}/log1p_cp10k_{field}"] = (
                float(np.mean(finite)) if finite else None
            )
    return result


def evaluation_metric_audit_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize exact field order and numeric types for coverage checksums."""

    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "biological_unit_alias": str(
                    row["biological_unit_alias"]
                ),
                "mask_label": str(row["mask_label"]),
                "mask_replicate": int(row["mask_replicate"]),
                "graph_condition": str(row["graph_condition"]),
                **{
                    field: (
                        int(row[field])
                        if field == "n_masked"
                        else float(row[field])
                    )
                    for field in REQUIRED_EVALUATION_METRIC_FIELDS
                },
            }
        )
    return result


def _peak_host_memory_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.  This repository executes Linux.
    return value * 1024 if sys.platform.startswith("linux") else value


def _determinism_facts(
    *, model_seed: int, device: torch.device
) -> dict[str, Any]:
    return {
        "python_random_seed": model_seed,
        "numpy_seed": model_seed,
        "torch_cpu_seed": model_seed,
        "torch_visible_cuda_generators_seeded": device.type == "cuda",
        "tile_order": "deterministic_seed_specific",
        "training_masks": "deterministic_cpu_generator_per_tile_epoch",
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "known_nondeterministic_operations": (
            [
                (
                    "PyG GATv2 CUDA scatter/reduction kernels may be "
                    "nondeterministic; deterministic algorithms are recorded "
                    "but not forced because source-compatible kernels may not "
                    "support them."
                )
            ]
            if device.type == "cuda"
            else []
        ),
        "exact_bitwise_reproducibility_promised": False,
    }


def _checkpoint_bytes(
    *,
    archive: RunArchive,
    config: Mapping[str, Any],
    training: TrainingResult,
    cohort: PooledFullCoreCohort,
    tiled_identity: Mapping[str, Any],
    mask_identities: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    materialization_checksum: str,
) -> bytes:
    state = {
        name: value.detach().cpu()
        for name, value in training.model.state_dict().items()
    }
    payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "model_name": "myjju-genemae",
        "model_family": "myjju_dual_path_genemae",
        "checkpoint_role": "last",
        "checkpoint_policy": "final_global_epoch_no_validation_selection",
        "selection_policy": "last_epoch_without_validation_selection",
        "seed": int(config["seed"]),
        "model_seed": int(config["seed"]),
        "final_epoch": final_epoch_index(training.completed_epochs),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "model_state_dict": state,
        "optimizer_state_dict": _cpu_tree(training.optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tree(training.scheduler.state_dict()),
        "state_dict_sha256": training.state_dict_sha256,
        "external_source_commit": SOURCE_COMMIT,
        "external_source_file_sha256": dict(SOURCE_FILE_SHA256),
        "source_audit_sha256": SOURCE_AUDIT_SHA256,
        "implementation_file_sha256": source_identity[
            "implementation_file_sha256"
        ],
        "source_fidelity_repair": "assign_self_hidden_constructor_attribute",
        "historical_weights_used": False,
        "cohort_fingerprint_sha256": cohort.fingerprint_sha256,
        "cohort_checksums": cohort.checksums.to_dict(),
        "per_core_preprocessing_sha256": {
            core.alias: core.checksums.preprocessing_sha256
            for core in cohort.cores
        },
        "ordered_gene_schema_sha256": (
            cohort.checksums.ordered_gene_schema_sha256
        ),
        "graph_bundle_sha256": tiled_identity["graph_bundle_sha256"],
        "per_core_graph_bundle_sha256": {
            alias: tiled_identity["cores"][alias]["graph_bundle_sha256"]
            for alias in ANC_ALIASES
        },
        "evaluation_mask_identities": dict(mask_identities),
        "materialization_checksum": materialization_checksum,
        "runtime_config_sha256": canonical_sha256(config),
        "target_scale": "full_cell_log1p_cp10k",
        "training_mask_rate": 0.5,
        "constructed_graph_self_loops": False,
        "convolution_add_self_loops": True,
        "monitored_metric": None,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _worker_archive_and_config(
    args: argparse.Namespace,
) -> tuple[RunArchive, dict[str, Any]]:
    run_id = os.environ.get("BAGM_RUN_ID", "").strip()
    environment_scratch = os.environ.get("BAGM_RUN_SCRATCH", "").strip()
    if not run_id or not environment_scratch:
        raise MyJJuRunnerError(
            "BAGM_RUN_ID and BAGM_RUN_SCRATCH are required; this runner "
            "must execute under the queue worker"
        )
    supplied = args.run_scratch.resolve(strict=False)
    if supplied != Path(environment_scratch).resolve(strict=False):
        raise MyJJuRunnerError("--run-scratch does not match BAGM_RUN_SCRATCH")
    expected_config = supplied / "config.resolved.yaml"
    if args.config.resolve(strict=False) != expected_config.resolve(strict=False):
        raise MyJJuRunnerError(
            "--config must be the worker-owned resolved configuration"
        )
    environment_config = os.environ.get("BAGM_CONFIG_PATH")
    if environment_config and Path(environment_config).resolve(
        strict=False
    ) != expected_config.resolve(strict=False):
        raise MyJJuRunnerError(
            "BAGM_CONFIG_PATH does not match the worker-owned configuration"
        )
    archive = RunArchive.attach_active(
        run_id,
        paths=current_paths(),
        scratch_path=supplied,
    )
    config = dict(load_yaml_mapping(expected_config))
    validate_experiment_config(config)
    return archive, config


def run_myjju_genemae_pooled(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    sample_key_salt: str,
    device: str | torch.device | None = None,
) -> MyJJuRunResult:
    """Execute one complete pilot or production member in an active archive."""

    diagnostic, model_seed = _validate_config(config)
    source_identity = _verify_frozen_files()
    materialization, materialization_checksum = _materialization_and_config(
        config
    )
    pilot_gate = _verify_production_gate(
        config, materialization_checksum=materialization_checksum
    )
    if len(sample_key_salt.encode("utf-8")) < 16:
        raise RunValidationError(
            "BAGM_SAMPLE_KEY_SALT must contain at least 16 UTF-8 bytes"
        )
    target_device = _resolve_runtime_device(device)
    if target_device.type == "cuda":
        torch.cuda.set_device(target_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(target_device)
    total_started = time.monotonic()
    data_started = time.monotonic()
    cohort = load_verified_ten_core_cohort(config)
    normalized = _normalised_expression(cohort)
    data_duration = time.monotonic() - data_started
    graph_started = time.monotonic()
    graph = _section(config, "graph")
    tiled = prepare_source_tiles(
        cohort,
        k=int(graph["neighbor_k"]),
        max_nodes=int(graph["maximum_tile_nodes"]),
    )
    tiled_identity = tiled.identity()
    if tiled_identity != graph.get("expected_tiled_graphs"):
        raise MyJJuRunnerError(
            "reconstructed tiled graphs differ from materialization"
        )
    if tiled_identity != materialization.get("tiled_graphs"):
        raise MyJJuRunnerError(
            "tiled graph identity differs from signed receipt"
        )
    graph_duration = time.monotonic() - graph_started
    # Initialization is part of the seed-controlled scientific variant and
    # must happen after seeding, not merely before the first training mask.
    seed_all(model_seed)
    model = make_source_model(num_genes=EXPECTED_N_GENES, mask_rate=0.5)
    parameter_count = count_parameters(model)
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise MyJJuRunnerError(
            f"source model parameter count changed: {parameter_count}"
        )
    training = train_source_model(
        model=model,
        cohort=cohort,
        tiled=tiled,
        normalized_expression=normalized,
        trainer=_section(config, "trainer"),
        model_seed=model_seed,
        device=target_device,
    )
    history_rows = [
        {
            "run_id": archive.run_id,
            "split": "fit",
            "model_seed": model_seed,
            **row,
        }
        for row in training.tile_history
    ]
    archive.write_table("metrics/history", history_rows, fallback="jsonl")
    archive.write_table(
        "metrics/global_epoch_history",
        [
            {
                "run_id": archive.run_id,
                "split": "fit",
                "model_seed": model_seed,
                **row,
            }
            for row in training.epoch_history
        ],
        fallback="jsonl",
    )
    for row in training.epoch_history:
        archive.append_metric_event(
            {
                "name": "fit/training/masked_huber",
                "value": row["masked_huber_mean"],
                "step": row["epoch_number"],
            }
        )

    mask_sources = _section(config, "evaluation").get("prior_mask_sources")
    if not isinstance(mask_sources, Mapping) or tuple(mask_sources) != ANC_ALIASES:
        raise MyJJuRunnerError("evaluation mask sources are incomplete")
    evaluation_rows: list[dict[str, Any]] = []
    mask_identities: dict[str, Any] = {}
    canonical_rows: list[dict[str, Any]] = []
    evaluation_started = time.monotonic()
    graph_identity_by_alias = tiled_identity["cores"]
    for core in cohort.cores:
        configured_identity, common_identity, native_base_seed = (
            _configured_mask_source(mask_sources, core.alias)
        )
        masks = regenerate_evaluation_masks(
            core,
            comparator_source=common_identity,
            native_base_seed=native_base_seed,
        )
        identity = masks.identity()
        if identity != configured_identity:
            raise MyJJuRunnerError(
                f"{core.alias} evaluation masks differ from materialization"
            )
        mask_identities[core.alias] = identity
        canonical_capture: dict[str, Any] = {}

        def capture(
            entry: EvaluationMask,
            condition: str,
            target: np.ndarray,
            mask: np.ndarray,
            prediction: np.ndarray,
        ) -> None:
            if (
                entry.label != "common_20"
                or entry.replicate != 0
                or condition != "observed"
            ):
                return
            counts = mask.sum(axis=0, dtype=np.int64)
            if np.any(counts == 0):
                raise MyJJuRunnerError(
                    f"{core.alias} canonical mask leaves a gene unsupported"
                )
            canonical_capture["y_true"] = (
                np.where(mask, target, 0.0).sum(axis=0, dtype=np.float64)
                / counts
            ).astype(float).tolist()
            canonical_capture["y_pred"] = (
                np.where(mask, prediction, 0.0).sum(
                    axis=0, dtype=np.float64
                )
                / counts
            ).astype(float).tolist()
            canonical_capture["effective_mask_rate"] = float(mask.mean())

        evaluation_rows.extend(
            evaluate_model_on_masks(
                training.model,
                core,
                tiled.for_alias(core.alias),
                masks,
                device=target_device,
                normalized_expression=normalized[core.alias],
                prediction_callback=capture,
            )
        )
        if set(canonical_capture) != {
            "y_true",
            "y_pred",
            "effective_mask_rate",
        }:
            raise MyJJuRunnerError(
                f"{core.alias} canonical prediction was not captured"
            )
        raw_row = {
            "_protected_alias": core.alias,
            "run_id": archive.run_id,
            "biological_unit_alias": core.alias,
            "graph_id": graph_identity_by_alias[core.alias][
                "graph_bundle_sha256"
            ],
            "dataset_id": str(_section(config, "dataset")["dataset_id"]),
            "split": "fit",
            "fold": int(config["fold"]),
            "y_true": canonical_capture["y_true"],
            "y_pred": canonical_capture["y_pred"],
            "node_count": core.n_nodes,
            "edge_count": graph_identity_by_alias[core.alias][
                "n_directed_edges"
            ],
            "effective_mask_rate": canonical_capture[
                "effective_mask_rate"
            ],
            "masking_type": "partial_gene",
            "mask_replicate": 0,
            "prediction_granularity": (
                "per_gene_mean_over_prespecified_masked_entries"
            ),
        }
        canonical_rows.extend(
            deidentify_prediction_rows(
                [raw_row],
                identifier_fields=["_protected_alias"],
                salt=sample_key_salt,
                namespace=(
                    "bagm:myjju-genemae:ten-core:"
                    f"{_section(config, 'dataset')['dataset_id']}"
                ),
            )
        )
        del masks
        gc.collect()
    evaluation_duration = time.monotonic() - evaluation_started
    if len(evaluation_rows) != 10 * 3 * 2 * 2:
        raise MyJJuRunnerError(
            "evaluation did not produce 120 core/replicate/condition rows"
        )
    if mask_identities != materialization.get("evaluation_masks"):
        raise MyJJuRunnerError(
            "evaluation mask identity differs from signed receipt"
        )
    normalized_evaluation_metric_rows = evaluation_metric_audit_rows(
        evaluation_rows
    )
    all_evaluation_metrics_finite = (
        len(normalized_evaluation_metric_rows) == 120
        and all(
            row.get(field) is not None
            and math.isfinite(float(row[field]))
            for row in normalized_evaluation_metric_rows
            for field in REQUIRED_EVALUATION_METRIC_FIELDS
        )
    )
    evaluation_metric_audit_sha256 = canonical_sha256(
        normalized_evaluation_metric_rows
    )
    if not all_evaluation_metrics_finite:
        raise FloatingPointError(
            "one or more required values in the 120 evaluation rows is non-finite"
        )
    archive.write_table(
        "metrics/per_core_replicate",
        [
            {
                "run_id": archive.run_id,
                "split": "fit",
                "model_seed": model_seed,
                **row,
            }
            for row in evaluation_rows
        ],
        fallback="jsonl",
    )
    archive.write_json(
        "diagnostics/evaluation_coverage.json",
        {
            "schema_version": 1,
            "row_count": len(normalized_evaluation_metric_rows),
            "required_metric_fields": list(
                REQUIRED_EVALUATION_METRIC_FIELDS
            ),
            "required_metric_value_count": (
                len(normalized_evaluation_metric_rows)
                * len(REQUIRED_EVALUATION_METRIC_FIELDS)
            ),
            "all_required_metrics_finite": (
                all_evaluation_metrics_finite
            ),
            "metric_audit_sha256": evaluation_metric_audit_sha256,
        },
    )
    final_metrics = _aggregate_metrics(evaluation_rows)
    if (
        final_metrics.get(PRIMARY_METRIC) is None
        or not math.isfinite(float(final_metrics[PRIMARY_METRIC]))
    ):
        raise FloatingPointError("primary evaluation metric is non-finite")
    archive.write_json("metrics/final.json", final_metrics)
    for name, value in final_metrics.items():
        if value is not None and math.isfinite(float(value)):
            archive.append_metric_event({"name": name, "value": value})
    prediction_path = archive.write_predictions(
        "fit", canonical_rows, fallback="jsonl"
    )

    checkpoint_and_replay_started = time.monotonic()
    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt",
        _checkpoint_bytes(
            archive=archive,
            config=config,
            training=training,
            cohort=cohort,
            tiled_identity=tiled_identity,
            mask_identities=mask_identities,
            source_identity=source_identity,
            materialization_checksum=materialization_checksum,
        ),
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    # PyG CUDA scatter/reduction can vary at the last few float32 bits even
    # for two forwards through the same model.  Replay on CPU preserves the
    # strict 1e-6 numerical gate and complements the exact state-dict digest.
    training.model.to(torch.device("cpu"))
    reloaded_model = reconstruct_model_from_checkpoint(
        checkpoint_path, device="cpu"
    )
    first_core = cohort.cores[0]
    first_tile = tiled.for_alias(first_core.alias)[0]
    replay_payload = verify_checkpoint_replay(
        original_model=training.model,
        reloaded_model=reloaded_model,
        core=first_core,
        tile=first_tile,
        normalized_expression=normalized[first_core.alias],
        device="cpu",
    )
    replay_payload.update(
        {
            "checkpoint_sha256": checkpoint_sha,
            "state_dict_sha256": training.state_dict_sha256,
            "strict_state_dict_load": True,
        }
    )
    archive.write_json("diagnostics/checkpoint_replay.json", replay_payload)
    del reloaded_model
    gc.collect()
    checkpoint_and_replay_duration = (
        time.monotonic() - checkpoint_and_replay_started
    )
    epoch_losses = [
        float(row["masked_huber_mean"]) for row in training.epoch_history
    ]
    convergence = {
        "fixed_budget_no_validation_selection": True,
        "completed_epochs": training.completed_epochs,
        "expected_epochs": int(_section(config, "trainer")["max_epochs"]),
        "optimizer_steps": training.optimizer_steps,
        "expected_optimizer_steps": (
            int(_section(config, "trainer")["max_epochs"])
            * len(tiled.tiles)
        ),
        "every_tile_once_each_epoch": all(
            row["every_tile_once"] for row in training.epoch_history
        ),
        "initial_epoch_masked_huber": epoch_losses[0],
        "final_epoch_masked_huber": epoch_losses[-1],
        "minimum_epoch_masked_huber": min(epoch_losses),
        "minimum_epoch": int(np.argmin(epoch_losses)) + 1,
        "final_over_initial_ratio": (
            epoch_losses[-1] / epoch_losses[0]
            if epoch_losses[0] != 0
            else None
        ),
        "all_losses_finite": all(math.isfinite(value) for value in epoch_losses),
        "all_gradients_finite": training.all_gradients_finite,
        "all_parameters_finite": training.all_parameters_finite,
        "all_global_epochs_completed": (
            training.completed_epochs
            == int(_section(config, "trainer")["max_epochs"])
        ),
        "all_losses_and_gradients_finite": (
            all(math.isfinite(value) for value in epoch_losses)
            and training.all_gradients_finite
        ),
        "checkpoint_is_final_epoch_only": True,
    }
    archive.write_json("diagnostics/convergence.json", convergence)
    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated(target_device))
        if target_device.type == "cuda"
        else 0
    )
    peak_host_bytes = _peak_host_memory_bytes()
    mean_epoch_seconds = float(
        np.mean(
            [row["duration_seconds"] for row in training.epoch_history]
        )
    )
    projected_runtime_seconds = (
        data_duration
        + graph_duration
        + mean_epoch_seconds * 200.0
        + evaluation_duration
        + checkpoint_and_replay_duration
    )
    projected_hours = projected_runtime_seconds / 3600.0
    current_free_bytes = shutil.disk_usage(archive.paths.artifact_root).free
    projected_final_free_bytes = max(
        0, current_free_bytes - PROJECTED_CAMPAIGN_OUTPUT_BYTES
    )
    all_metrics_finite = all(
        value is not None and math.isfinite(float(value))
        for value in final_metrics.values()
    )
    pilot_checks = {
        "parameter_count_match": parameter_count == EXPECTED_PARAMETER_COUNT,
        "every_tile_once_each_epoch": convergence[
            "every_tile_once_each_epoch"
        ],
        "completed_exact_epoch_budget": (
            training.completed_epochs
            == int(_section(config, "trainer")["max_epochs"])
        ),
        "finite_losses_gradients_parameters_metrics": (
            convergence["all_losses_finite"]
            and training.all_gradients_finite
            and training.all_parameters_finite
            and all_metrics_finite
            and all_evaluation_metrics_finite
        ),
        "peak_vram_passed": (
            peak_vram_bytes / 1024**3 <= PILOT_MAX_VRAM_GIB
        ),
        "peak_host_memory_passed": (
            peak_host_bytes / 1024**3 <= PILOT_MAX_HOST_GIB
        ),
        "projected_runtime_passed": projected_hours
        <= PILOT_MAX_RUNTIME_HOURS,
        "projected_disk_passed": projected_final_free_bytes / 1024**3
        >= PILOT_MIN_FINAL_FREE_DISK_GIB,
        "full_evaluation_coverage": len(evaluation_rows) == 120,
        "all_120_row_required_metrics_finite": (
            all_evaluation_metrics_finite
        ),
    }
    resource_payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "diagnostic_resource_pilot": diagnostic,
        "device": str(target_device),
        "parameter_count": parameter_count,
        "tile_count": len(tiled.tiles),
        "optimizer_steps": training.optimizer_steps,
        "data_duration_seconds": data_duration,
        "graph_duration_seconds": graph_duration,
        "training_duration_seconds": training.duration_seconds,
        "evaluation_duration_seconds": evaluation_duration,
        "measured_checkpoint_and_replay_finalization_allowance_seconds": (
            checkpoint_and_replay_duration
        ),
        "total_duration_seconds": time.monotonic() - total_started,
        "mean_epoch_duration_seconds": mean_epoch_seconds,
        "peak_allocated_vram_bytes": peak_vram_bytes,
        "peak_allocated_vram_gib": peak_vram_bytes / 1024**3,
        "peak_host_memory_bytes": peak_host_bytes,
        "peak_host_memory_gib": peak_host_bytes / 1024**3,
        "current_free_disk_gib": current_free_bytes / 1024**3,
        "projected_campaign_output_gib": (
            PROJECTED_CAMPAIGN_OUTPUT_BYTES / 1024**3
        ),
        "projected_final_free_disk_gib": (
            projected_final_free_bytes / 1024**3
        ),
        "projected_200_epoch_runtime_hours": projected_hours,
        "projected_200_epoch_runtime_formula": (
            "data_duration + graph_duration + 200 * mean_epoch_duration + "
            "full_evaluation_duration + measured_checkpoint_and_replay_"
            "finalization_allowance"
        ),
        "evaluation_metric_audit_sha256": (
            evaluation_metric_audit_sha256
        ),
        "thresholds": {
            "peak_allocated_vram_gib_maximum": PILOT_MAX_VRAM_GIB,
            "peak_host_memory_gib_maximum": PILOT_MAX_HOST_GIB,
            "projected_200_epoch_runtime_hours_maximum": (
                PILOT_MAX_RUNTIME_HOURS
            ),
            "projected_final_free_disk_gib_minimum": (
                PILOT_MIN_FINAL_FREE_DISK_GIB
            ),
        },
        "checks": pilot_checks,
        "pilot_gate_passed": all(pilot_checks.values()),
    }
    archive.write_json("diagnostics/resource.json", resource_payload)
    if diagnostic and resource_payload["pilot_gate_passed"] is not True:
        raise MyJJuRunnerError(
            "resource pilot completed but failed one or more frozen gates"
        )
    archive.write_json("provenance/external_source_audit.json", source_identity)
    archive.write_json(
        "provenance/cohort.json",
        {
            "aliases": list(cohort.aliases),
            "total_nodes": cohort.total_nodes,
            "n_genes": cohort.n_genes,
            "fingerprint_sha256": cohort.fingerprint_sha256,
            "checksums": cohort.checksums.to_dict(),
            "target_scale": "full_cell_log1p_cp10k",
            "hidden_entries_contribute_to_library_denominator": True,
            "protected_identifier_arrays_returned": False,
        },
    )
    archive.write_json("provenance/tiled_graphs.json", tiled_identity)
    archive.write_json(
        "provenance/fixed_evaluation_masks.json",
        {
            "schema_version": 1,
            "aliases": list(ANC_ALIASES),
            "masks": mask_identities,
            "common_mask_rate": 0.2,
            "native_mask_rate": 0.5,
            "replicates_per_rate": 3,
        },
    )
    archive.write_json(
        "provenance/training.json",
        {
            "model_seed": model_seed,
            "final_epoch": final_epoch_index(training.completed_epochs),
            "optimizer_steps": training.optimizer_steps,
            "state_dict_sha256": training.state_dict_sha256,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_policy": "last_only",
            "validation_selection": False,
            "early_stopping": False,
            "materialization_checksum": materialization_checksum,
            "production_pilot_gate": pilot_gate,
            "constructed_graph_self_loops": False,
            "convolution_add_self_loops": True,
            "determinism": _determinism_facts(
                model_seed=model_seed, device=target_device
            ),
        },
    )
    primary_value = float(final_metrics[PRIMARY_METRIC])
    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "campaign_id": CAMPAIGN_ID,
        "model_name": "myjju-genemae",
        "model_family": "myjju_dual_path_genemae",
        "model_seed": model_seed,
        "primary_metric_name": PRIMARY_METRIC,
        "primary_metric_value": primary_value,
        "parameter_count": parameter_count,
        "final_epoch": final_epoch_index(training.completed_epochs),
        "completed_global_epochs": training.completed_epochs,
        "optimizer_steps": training.optimizer_steps,
        "checkpoint_role": "last",
        "checkpoint_path": "checkpoints/last.ckpt",
        "checkpoint_sha256": checkpoint_sha,
        "state_dict_sha256": training.state_dict_sha256,
        "prediction_path": prediction_path.relative_to(
            archive.scratch_path
        ).as_posix(),
        "prediction_granularity": (
            "compact_per_core_per_gene_masked_means"
        ),
        "per_core_replicate_metrics": (
            "metrics/per_core_replicate.parquet_or_jsonl"
        ),
        "evaluation_row_count": len(evaluation_rows),
        "evaluation_masks": {
            "common_rate": 0.2,
            "native_rate": 0.5,
            "replicates_per_rate": 3,
            "graph_conditions": [
                "observed",
                "node_label_permuted",
            ],
        },
        "core_count": len(cohort.cores),
        "tile_count": len(tiled.tiles),
        "total_cells": cohort.total_nodes,
        "gene_count": cohort.n_genes,
        "cohort_fingerprint_sha256": cohort.fingerprint_sha256,
        "graph_bundle_sha256": tiled_identity["graph_bundle_sha256"],
        "duration_seconds": time.monotonic() - total_started,
        "peak_vram_gb": peak_vram_bytes / 1024**3,
        "peak_host_memory_bytes": peak_host_bytes,
        "diagnostic_resource_pilot": diagnostic,
        "conclusion_eligible": not diagnostic,
        "generalization_estimate": False,
        "exploratory": True,
        "maximum_claim": (
            "diagnostic runtime, memory, and numerical feasibility only"
            if diagnostic
            else (
                "descriptive held-in partial-gene reconstruction by a "
                "retrained source-fidelity architecture"
            )
        ),
        "historical_weights_evaluated": False,
        "source_fidelity_repair": "assign_self_hidden_constructor_attribute",
        "materialization_checksum": materialization_checksum,
        "pilot_gate_passed": (
            resource_payload["pilot_gate_passed"] if diagnostic else True
        ),
        "failures": [],
    }
    archive.write_summary(summary)
    return MyJJuRunResult(
        run_id=archive.run_id,
        primary_metric_name=PRIMARY_METRIC,
        primary_metric_value=primary_value,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    result = run_myjju_genemae_pooled(
        config,
        archive,
        sample_key_salt=os.environ.get("BAGM_SAMPLE_KEY_SALT", ""),
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "primary_metric_name": result.primary_metric_name,
                "primary_metric_value": result.primary_metric_value,
                "checkpoint": str(result.checkpoint_path),
                "predictions": str(result.prediction_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
