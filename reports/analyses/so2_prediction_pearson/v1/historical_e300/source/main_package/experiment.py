"""Immutable single-run orchestration for the normal-core benchmark."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .artifacts import load_prepared_artifact, sha256_file
from .graphs import (
    EdgeAttributeStandardizer,
    GraphQC,
    SpatialGraph,
    _assemble_graph,
    build_spatial_graph,
    rewire_spatial_graph,
)
from .spatial_field import (
    BROAD_SPATIAL_BASIS_NAME,
    BROAD_SPATIAL_FEATURE_NAMES,
)
from .standards_lock import (
    StandardsLockError,
    authorize_locked_test_job,
    canonical_job_hash,
    condition_name,
)
from .training import (
    GraphSplitView,
    TrainingConfig,
    build_model,
    evaluate_fixed_mask,
    fit_model,
    fit_staged_g3,
)


class ExperimentContractError(ValueError):
    """Raised when a run would violate a locked experiment contract."""


_GRAPH_OVERRIDE_KEYS = {
    "k",
    "radius_um",
    "symmetry",
    "min_distance_um",
    "rbf_bins",
    "edge_dropout",
    "rewire_distance_bins",
    "min_rewire_success_fraction",
    "max_rewire_distance_mean_change",
}
_MODEL_OVERRIDE_KEYS = {
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
}
_TRAINING_OVERRIDE_KEYS = {
    "max_epochs",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "patience",
    "curriculum",
    "edge_dropout",
    "amp",
    "amp_dtype",
    "deterministic",
    "device",
}
_SELF_MODEL_KEYS = {
    "b0",
    "self",
    "selfmlp",
    "b0matched",
    "matchedself",
    "parametermatchedself",
}
_BROAD_FIELD_MODEL_KEYS = {
    "broadfield",
    "broadspatialfield",
    "spatialfield",
}
_EDGE_MODEL_KEYS = {
    "g2",
    "edgegat",
    "edgeconditionedgatv2",
    "g3",
    "additive",
    "additiveedgemessage",
}
_B1_MODEL_KEYS = {"b1", "mean", "meanneighbor"}
_G1_MODEL_KEYS = {"g1", "topologygat", "topologygatv2"}
_KNOWN_MODEL_KEYS = (
    _SELF_MODEL_KEYS
    | _BROAD_FIELD_MODEL_KEYS
    | _B1_MODEL_KEYS
    | _G1_MODEL_KEYS
    | _EDGE_MODEL_KEYS
)
_REWIRE_ONLY_GRAPH_OVERRIDES = {
    "rewire_distance_bins",
    "min_rewire_success_fraction",
    "max_rewire_distance_mean_change",
}
_GRAPH_CONSTRUCTION_KEYS = {
    "k",
    "radius_um",
    "symmetry",
    "min_distance_um",
    "rbf_bins",
}
_EDGE_CONTROL_DISTANCE_BINS = 8


def _model_key(value: object) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )


def _canonical_matrix_model_name(model_key: str) -> str:
    if model_key in {"b0", "self", "selfmlp"}:
        return "b0"
    if model_key in {"b0matched", "matchedself", "parametermatchedself"}:
        return "b0-matched"
    if model_key in _BROAD_FIELD_MODEL_KEYS:
        return "broad-field"
    if model_key in _B1_MODEL_KEYS:
        return "b1"
    if model_key in _G1_MODEL_KEYS:
        return "g1"
    if model_key in {"g2", "edgegat", "edgeconditionedgatv2"}:
        return "g2"
    if model_key in {"g3", "additive", "additiveedgemessage"}:
        return "g3"
    raise ExperimentContractError(f"Unknown model key for lock: {model_key}")


def _locked_job_request(
    *,
    model_key: str,
    model_seed: int,
    graph_overrides: Mapping[str, Any],
    model_overrides: Mapping[str, Any],
    training_overrides: Mapping[str, Any],
    rewired: bool,
    rewire_seed: int,
    swaps_per_edge: float,
    edge_control: str,
    pretrained_b0_checkpoint: str | Path | None,
    g3_frozen_epochs: int,
    g3_joint_learning_rate: float,
) -> dict[str, Any]:
    """Build the exact launcher-matrix job represented by run arguments."""

    job: dict[str, Any] = {
        "model": _canonical_matrix_model_name(model_key),
        "seed": int(model_seed),
    }

    def merge(values: Mapping[str, Any], *, source: str) -> None:
        for name, value in values.items():
            if name in job and job[name] != value:
                raise ExperimentContractError(
                    f"Conflicting locked-job value for {name!r} from {source}."
                )
            job[name] = value

    merge(graph_overrides, source="graph overrides")
    merge(model_overrides, source="model overrides")
    merge(
        {
            name: value
            for name, value in training_overrides.items()
            if name != "device"
        },
        source="training overrides",
    )
    if rewired:
        merge(
            {
                "rewired": True,
                "rewire_seed": int(rewire_seed),
                "swaps_per_edge": float(swaps_per_edge),
            },
            source="rewiring controls",
        )
    if model_key in {"g2", "edgegat", "edgeconditionedgatv2"}:
        merge({"edge_control": edge_control}, source="edge control")
    elif edge_control != "none":
        merge({"edge_control": edge_control}, source="edge control")
    if edge_control == "permuted" and not rewired:
        merge({"rewire_seed": int(rewire_seed)}, source="edge-control seed")
    if model_key in {"g3", "additive", "additiveedgemessage"}:
        assert pretrained_b0_checkpoint is not None
        merge(
            {
                "pretrained_b0_checkpoint": str(pretrained_b0_checkpoint),
                "g3_frozen_epochs": int(g3_frozen_epochs),
                "g3_joint_learning_rate": float(g3_joint_learning_rate),
            },
            source="G3 controls",
        )
    return job


def _validate_override_keys(
    values: Mapping[str, Any] | None,
    allowed: set[str],
    *,
    name: str,
) -> dict[str, Any]:
    result = dict(values or {})
    unknown = sorted(set(result).difference(allowed))
    if unknown:
        raise ExperimentContractError(
            f"Unknown {name} override keys: {', '.join(unknown)}"
        )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_content_hash(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("manifest_content_sha256", None)
    return _canonical_hash(core)


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        # Do not embed untracked paths, which may contain restricted filenames.
        "status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if status is not None
            else None
        ),
    }


def _dependency_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import torch_geometric

        state["torch_geometric"] = torch_geometric.__version__
    except Exception:
        state["torch_geometric"] = None
    if torch.cuda.is_available():
        try:
            state["cudnn"] = torch.backends.cudnn.version()
            state["cudnn_probe_error"] = None
        except RuntimeError as exc:
            # Dependency provenance must still be writable for CPU-only
            # diagnostics on a GPU host with a mismatched optional cuDNN.
            # GPU training will surface an actual cuDNN use-site failure.
            state["cudnn"] = None
            state["cudnn_probe_error"] = str(exc)
        state["visible_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": torch.cuda.get_device_properties(index).total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return state


def _graph_id(config: Mapping[str, Any], *, rewired: bool) -> str:
    base = (
        f"k{int(config['k'])}_r{float(config['radius_um']):g}_"
        f"{str(config['symmetry']).lower()}_rbf{int(config.get('rbf_bins', 8))}"
    )
    minimum = float(config.get("min_distance_um", 0.0))
    if minimum:
        base += f"_min{minimum:g}"
    return f"{base}__rewired" if rewired else base


def _derived_group_seed(seed: int, group: str, *, purpose: str) -> int:
    payload = f"{purpose}|{int(seed)}|{group}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _rewire_split_graphs(
    coordinates: np.ndarray,
    split_labels: np.ndarray,
    fov: np.ndarray,
    graph_config: Mapping[str, Any],
    *,
    seed: int,
    swaps_per_edge: float,
    base_graph: SpatialGraph | None = None,
) -> SpatialGraph:
    """Build and rewire each split independently, then assemble global indices."""

    canonical_splits = ("train", "val", "test")
    split_values = np.asarray(split_labels).astype(str)
    observed = set(split_values.tolist())
    if observed != set(canonical_splits):
        raise ExperimentContractError(
            "Rewiring requires exactly train, val, and test split labels."
        )
    if base_graph is not None and base_graph.n_nodes != len(split_values):
        raise ExperimentContractError(
            "The base graph does not align to prepared nodes."
        )

    global_pairs: list[np.ndarray] = []
    split_records: dict[str, Any] = {}
    original_distance_sum = 0.0
    rewired_distance_sum = 0.0
    total_relations = 0
    total_target = 0
    total_successful = 0
    total_attempts = 0
    for split_name in canonical_splits:
        display_name = "validation" if split_name == "val" else split_name
        node_index = np.flatnonzero(split_values == split_name).astype(np.int64)
        local_coordinates = coordinates[node_index]
        local_fov = fov[node_index]
        if base_graph is None:
            local_graph = build_spatial_graph(
                local_coordinates,
                k=int(graph_config["k"]),
                radius_um=float(graph_config["radius_um"]),
                symmetry=str(graph_config["symmetry"]),
                fov=local_fov,
                min_distance_um=float(
                    graph_config.get("min_distance_um", 0.0)
                ),
                rbf_bins=int(graph_config.get("rbf_bins", 8)),
            )
        else:
            edge_mask = split_values[base_graph.edge_index[0]] == split_name
            if np.any(
                split_values[base_graph.edge_index[1, edge_mask]]
                != split_name
            ):
                raise ExperimentContractError(
                    "The base graph crosses a split boundary."
                )
            inverse = np.full(len(split_values), -1, dtype=np.int64)
            inverse[node_index] = np.arange(len(node_index), dtype=np.int64)
            local_edges = inverse[base_graph.edge_index[:, edge_mask]]
            pair_mask = local_edges[0] < local_edges[1]
            local_pairs = local_edges[:, pair_mask].T
            local_graph = _assemble_graph(
                local_pairs,
                local_coordinates,
                radius_um=float(graph_config["radius_um"]),
                rbf_bins=int(graph_config.get("rbf_bins", 8)),
                group_labels=np.full(len(node_index), "__all__", dtype="U8"),
                fov_labels=np.asarray(local_fov).astype(str),
                candidate_counts=None,
                k_for_qc=None,
                config={
                    "kind": "spatial_knn_radius",
                    "k": int(graph_config["k"]),
                    "radius_um": float(graph_config["radius_um"]),
                    "min_distance_um": float(
                        graph_config.get("min_distance_um", 0.0)
                    ),
                    "symmetry": str(graph_config["symmetry"]),
                    "rbf_bins": int(graph_config.get("rbf_bins", 8)),
                    "group_restricted": False,
                },
            )
        split_seed = _derived_group_seed(
            int(seed), split_name, purpose="rewire"
        )
        local_rewired = rewire_spatial_graph(
            local_graph,
            local_coordinates,
            seed=split_seed,
            swaps_per_edge=float(swaps_per_edge),
            distance_bins=int(graph_config.get("rewire_distance_bins", 8)),
            fov=local_fov,
        )
        local_pair_mask = local_rewired.edge_index[0] < local_rewired.edge_index[1]
        local_pairs = local_rewired.edge_index[:, local_pair_mask].T
        if len(local_pairs):
            global_pairs.append(node_index[local_pairs])

        metadata = dict(local_rewired.metadata)
        relations = int(local_rewired.qc.n_undirected_edges)
        if relations:
            original_pair_mask = local_graph.edge_index[0] < local_graph.edge_index[1]
            rewired_pair_mask = (
                local_rewired.edge_index[0] < local_rewired.edge_index[1]
            )
            original_mean = float(
                local_graph.edge_attr[original_pair_mask, 0].mean()
            )
            rewired_mean = float(
                local_rewired.edge_attr[rewired_pair_mask, 0].mean()
            )
            original_distance_sum += original_mean * relations
            rewired_distance_sum += rewired_mean * relations
            total_relations += relations
        total_target += int(metadata.get("rewire_target_swaps", 0))
        total_successful += int(metadata.get("rewire_successful_swaps", 0))
        total_attempts += int(metadata.get("rewire_attempts", 0))
        split_records[display_name] = {
            "seed": split_seed,
            "n_nodes": int(len(node_index)),
            "n_undirected_edges": relations,
            **metadata,
        }

    pairs = (
        np.concatenate(global_pairs, axis=0)
        if global_pairs
        else np.empty((0, 2), dtype=np.int64)
    )
    original_mean = (
        original_distance_sum / total_relations if total_relations else 0.0
    )
    rewired_mean = (
        rewired_distance_sum / total_relations if total_relations else 0.0
    )
    metadata = {
        "rewire_seed": int(seed),
        "rewire_seed_derivation": "sha256(purpose|seed|split)",
        "rewire_target_swaps": int(total_target),
        "rewire_successful_swaps": int(total_successful),
        "rewire_attempts": int(total_attempts),
        "rewire_success_fraction": (
            float(total_successful / total_target) if total_target else 1.0
        ),
        "degree_preserved_exactly": True,
        "distance_bins_fitted_per_split": True,
        "original_edge_distance_mean_um": float(original_mean),
        "rewired_edge_distance_mean_um": float(rewired_mean),
        "relative_edge_distance_mean_change": float(
            abs(rewired_mean - original_mean)
            / max(original_mean, np.finfo(np.float64).eps)
        ),
        "split_rewires": split_records,
    }
    return _assemble_graph(
        pairs,
        coordinates,
        radius_um=float(graph_config["radius_um"]),
        rbf_bins=int(graph_config.get("rbf_bins", 8)),
        group_labels=split_values,
        fov_labels=np.asarray(fov).astype(str),
        candidate_counts=None,
        k_for_qc=None,
        config={
            "kind": "degree_distance_rewired",
            "k": int(graph_config["k"]),
            "radius_um": float(graph_config["radius_um"]),
            "min_distance_um": float(graph_config.get("min_distance_um", 0.0)),
            "symmetry": str(graph_config["symmetry"]),
            "rbf_bins": int(graph_config.get("rbf_bins", 8)),
            "group_restricted": True,
        },
        metadata=metadata,
    )


def _make_graph(
    arrays: Mapping[str, np.ndarray],
    graph_config: Mapping[str, Any],
    *,
    rewired: bool,
    rewire_seed: int,
    swaps_per_edge: float,
    base_graph: SpatialGraph | None = None,
) -> tuple[SpatialGraph, np.ndarray]:
    coordinates = arrays["coordinates_um"]
    split_labels = arrays["split_labels"]
    fov = arrays["fov"]
    if rewired:
        if not np.isfinite(swaps_per_edge) or swaps_per_edge <= 0:
            raise ExperimentContractError(
                "A rewired control requires swaps_per_edge > 0."
            )
        graph = _rewire_split_graphs(
            coordinates,
            split_labels,
            fov,
            graph_config,
            seed=int(rewire_seed),
            swaps_per_edge=float(swaps_per_edge),
            base_graph=base_graph,
        )
        target_swaps = int(graph.metadata.get("rewire_target_swaps", 0))
        successful_swaps = int(
            graph.metadata.get("rewire_successful_swaps", 0)
        )
        success_fraction = float(
            graph.metadata.get("rewire_success_fraction", 0.0)
        )
        minimum_success = float(
            graph_config.get("min_rewire_success_fraction", 0.5)
        )
        maximum_distance_change = float(
            graph_config.get("max_rewire_distance_mean_change", 0.2)
        )
        distance_change = float(
            graph.metadata.get("relative_edge_distance_mean_change", np.inf)
        )
        if target_swaps <= 0 or successful_swaps <= 0:
            raise ExperimentContractError(
                "The requested rewired control produced no valid edge swaps."
            )
        if success_fraction < minimum_success:
            raise ExperimentContractError(
                "The rewired control did not achieve its prespecified minimum "
                f"swap fraction ({success_fraction:.3f} < {minimum_success:.3f})."
            )
        if distance_change > maximum_distance_change:
            raise ExperimentContractError(
                "The rewired control exceeded its prespecified mean-distance "
                f"change ({distance_change:.3f} > {maximum_distance_change:.3f})."
            )
    else:
        graph = build_spatial_graph(
            coordinates,
            k=int(graph_config["k"]),
            radius_um=float(graph_config["radius_um"]),
            symmetry=str(graph_config["symmetry"]),
            group_labels=split_labels,
            fov=fov,
            min_distance_um=float(graph_config.get("min_distance_um", 0.0)),
            rbf_bins=int(graph_config.get("rbf_bins", 8)),
        )
    edge_splits = split_labels[graph.edge_index[0]]
    if np.any(edge_splits != split_labels[graph.edge_index[1]]):
        raise ExperimentContractError("Run graph contains cross-split edges.")
    train_edges = edge_splits == "train"
    if not train_edges.any():
        raise ExperimentContractError("The run training graph contains no edges.")
    standardizer = EdgeAttributeStandardizer().fit(graph.edge_attr[train_edges])
    return graph, standardizer.transform(graph.edge_attr)


def _prepared_graph(
    prepared_manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> tuple[SpatialGraph, np.ndarray]:
    """Restore the immutable prepared topology without recomputing it."""

    graph_record = prepared_manifest.get("graph")
    if not isinstance(graph_record, Mapping):
        raise ExperimentContractError("Prepared graph declaration is missing.")
    try:
        qc = GraphQC(**dict(graph_record["qc"]))
        graph = SpatialGraph(
            edge_index=np.asarray(arrays["edge_index"], dtype=np.int64),
            edge_attr=np.asarray(arrays["edge_attributes_raw"], dtype=np.float32),
            edge_attr_names=tuple(graph_record["edge_attribute_names"]),
            n_nodes=int(len(arrays["split_labels"])),
            qc=qc,
            config=dict(graph_record["config"]),
        )
        attributes = np.asarray(
            arrays["edge_attributes"], dtype=np.float32
        ).copy()
    except (KeyError, TypeError, ValueError) as exc:
        raise ExperimentContractError(
            "Prepared graph arrays do not match their manifest."
        ) from exc
    if attributes.shape != graph.edge_attr.shape:
        raise ExperimentContractError(
            "Prepared standardized edge attributes are misaligned."
        )
    if (
        graph.qc.n_nodes != graph.n_nodes
        or graph.qc.n_directed_edges != graph.n_directed_edges
        or graph.qc.self_loops
        or graph.qc.duplicate_directed_edges
        or not graph.qc.directed_edge_pairs_are_symmetric
    ):
        raise ExperimentContractError("Prepared graph QC is internally inconsistent.")
    split_values = np.asarray(arrays["split_labels"]).astype(str)
    source, target = graph.edge_index
    if (
        graph.qc.cross_group_edges
        or np.any(split_values[source] != split_values[target])
    ):
        raise ExperimentContractError("Prepared graph contains cross-split edges.")
    if not np.any(split_values[source] == "train"):
        raise ExperimentContractError("Prepared training graph contains no edges.")
    return graph, attributes


def _edge_control(
    edge_attributes: np.ndarray,
    edge_attr_names: Sequence[str],
    edge_index: np.ndarray,
    edge_splits: Sequence[object] | np.ndarray,
    control: str,
    *,
    seed: int,
) -> np.ndarray:
    control = str(control).lower().replace("-", "_")
    attributes = np.asarray(edge_attributes, dtype=np.float32).copy()
    edges = np.asarray(edge_index)
    split_values = np.asarray(edge_splits).astype(str)
    if (
        edges.shape != (2, len(attributes))
        or split_values.shape != (len(attributes),)
    ):
        raise ExperimentContractError(
            "Edge controls require aligned attributes, edge indices, and splits."
        )
    if control in {"none", "full"}:
        return attributes
    if control == "zero":
        return np.zeros_like(attributes)
    if control == "distance_only":
        keep = np.asarray(
            [
                ("distance" in name.lower()) or ("rbf" in name.lower())
                for name in edge_attr_names
            ],
            dtype=bool,
        )
        attributes[:, ~keep] = 0.0
        return attributes
    if control == "permuted":
        try:
            distance_column = tuple(edge_attr_names).index(
                "distance_over_radius"
            )
        except ValueError as exc:
            raise ExperimentContractError(
                "Permuted edge control requires distance_over_radius."
            ) from exc
        distance = attributes[:, distance_column]
        training_positions = np.flatnonzero(split_values == "train")
        if len(training_positions) == 0:
            raise ExperimentContractError(
                "Permuted edge bins require training edges."
            )
        # Bin boundaries are fitted only on training-edge geometry. Held-out
        # distributions cannot influence the mechanism-breaking control.
        boundaries = np.unique(
            np.quantile(
                distance[training_positions],
                np.linspace(0, 1, _EDGE_CONTROL_DISTANCE_BINS + 1),
            )
        )
        bins = np.digitize(distance, boundaries[1:-1], right=True)
        # Never move attributes between train, validation, and test graphs.
        # Directed reverse edges are intentionally permuted edge-wise.
        for split_name in np.unique(split_values):
            for distance_bin in np.unique(bins[split_values == split_name]):
                positions = np.flatnonzero(
                    (split_values == split_name) & (bins == distance_bin)
                )
                if len(positions) > 1:
                    seed_payload = (
                        f"{int(seed)}|{split_name}|{int(distance_bin)}"
                    ).encode("utf-8")
                    group_seed = int.from_bytes(
                        hashlib.sha256(seed_payload).digest()[:8], "little"
                    )
                    rng = np.random.default_rng(group_seed)
                    attributes[positions] = attributes[
                        rng.permutation(positions)
                    ]
        return attributes
    raise ExperimentContractError(
        "edge_control must be one of none, zero, distance_only, or permuted"
    )


def _split_view(
    name: str,
    arrays: Mapping[str, np.ndarray],
    graph: SpatialGraph,
    edge_attributes: np.ndarray,
) -> tuple[GraphSplitView, np.ndarray]:
    label = {"train": "train", "validation": "val", "test": "test"}[name]
    node_index = np.flatnonzero(arrays["split_labels"] == label).astype(np.int64)
    edge_mask = arrays["split_labels"][graph.edge_index[0]] == label
    if np.any(arrays["split_labels"][graph.edge_index[1, edge_mask]] != label):
        raise ExperimentContractError(f"{name} graph crosses a split boundary.")
    inverse = np.full(len(arrays["split_labels"]), -1, dtype=np.int64)
    inverse[node_index] = np.arange(len(node_index), dtype=np.int64)
    local_edges = inverse[graph.edge_index[:, edge_mask]]
    if local_edges.size and local_edges.min() < 0:
        raise ExperimentContractError(f"Failed to localize {name} graph.")
    view = GraphSplitView(
        expression=torch.from_numpy(
            np.asarray(arrays["target_expression"][node_index], dtype=np.float32)
        ),
        coordinates_um=torch.from_numpy(
            np.asarray(arrays["coordinates_um"][node_index], dtype=np.float64)
        ),
        edge_index=torch.from_numpy(local_edges),
        node_covariates=torch.from_numpy(
            np.asarray(arrays["node_covariates"][node_index], dtype=np.float32)
        ),
        edge_attributes=torch.from_numpy(
            np.asarray(edge_attributes[edge_mask], dtype=np.float32)
        ),
        block_ids=np.asarray(arrays["macroblock_ids"][node_index]),
        name=name,
    )
    return view, node_index


def _model_kwargs(
    model_name: str,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    common = {
        "hidden_dim": int(model_config.get("hidden_dim", 256)),
        "ffn_dim": int(model_config.get("ffn_dim", 512)),
        "decoder_dim": int(model_config.get("decoder_dim", 512)),
        "dropout": float(model_config.get("dropout", 0.1)),
    }
    key = _model_key(model_name)
    if key in (
        {"b0", "self", "selfmlp", "b1", "mean", "meanneighbor"}
        | _BROAD_FIELD_MODEL_KEYS
    ):
        return common
    common.update(
        {
            "attention_heads": int(model_config.get("attention_heads", 4)),
            "graph_layers": int(model_config.get("graph_layers", 1)),
            "attention_dropout": float(
                model_config.get("attention_dropout", 0.1)
            ),
        }
    )
    if key in _EDGE_MODEL_KEYS:
        common.update(
            {
                "edge_hidden_dim": int(model_config.get("edge_hidden_dim", 64)),
                "edge_embedding_dim": int(
                    model_config.get("edge_embedding_dim", 32)
                ),
            }
        )
    if key in {"g3", "additive", "additiveedgemessage"}:
        common.pop("graph_layers", None)
        common.update(
            {
                "message_head_dim": int(
                    model_config.get("message_head_dim", 16)
                ),
                "message_dim": int(model_config.get("message_dim", 64)),
            }
        )
    return common


def _training_config(
    prepared_config: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> TrainingConfig:
    masking = prepared_config["masking"]
    optimization = prepared_config.get("optimization", {})
    return TrainingConfig(
        max_epochs=int(
            run_config.get("max_epochs", optimization.get("max_epochs", 200))
        ),
        learning_rate=float(
            run_config.get(
                "learning_rate", optimization.get("learning_rate", 3e-4)
            )
        ),
        weight_decay=float(
            run_config.get(
                "weight_decay", optimization.get("weight_decay", 1e-4)
            )
        ),
        gradient_clip_norm=float(
            run_config.get(
                "gradient_clip_norm",
                optimization.get("gradient_clip_norm", 1.0),
            )
        ),
        patience=int(
            run_config.get(
                "patience",
                optimization.get("early_stopping_patience", 25),
            )
        ),
        curriculum=str(
            run_config.get("curriculum", masking.get("curriculum", "P+N+B"))
        ),
        warmup_epochs=int(masking.get("warmup_epochs", 10)),
        partial_gene_rate=float(masking.get("partial_gene_rate", 0.2)),
        node_rate=float(masking.get("whole_node_rate", 0.1)),
        block_node_rate=float(masking.get("block_node_rate", 0.1)),
        block_width_um=masking.get("block_width_um"),
        block_shape=str(masking.get("block_shape", "disk")),
        mask_seed=int(masking.get("mask_seed", 0)),
        model_seed=int(run_config["model_seed"]),
        edge_dropout=float(
            run_config.get(
                "edge_dropout", prepared_config["graph"].get("edge_dropout", 0.1)
            )
        ),
        amp=bool(run_config.get("amp", optimization.get("amp", False))),
        amp_dtype=str(run_config.get("amp_dtype", "auto")),
        deterministic=bool(run_config.get("deterministic", True)),
        deterministic_warn_only=False,
        device=str(run_config.get("device", "cuda")),
    )


def _evaluation_records(
    model: torch.nn.Module,
    view: GraphSplitView,
    split_name: str,
    bundle: Any,
    *,
    device: str,
    amp: bool,
    amp_dtype: str,
    save_predictions: bool,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    declarations: list[dict[str, Any]] = []
    if save_predictions:
        arrays[f"{split_name}__y_true"] = view.expression.numpy()
        arrays[f"{split_name}__block_ids"] = np.asarray(view.block_ids)
        arrays[f"{split_name}__cell_ids"] = np.arange(view.num_nodes, dtype=np.int64)
    for entry in bundle.manifest["entries"]:
        mode = str(entry["spec"]["mode"])
        replicate = int(entry["replicate"])
        mask = bundle.masks[str(entry["entry_id"])]
        result = evaluate_fixed_mask(
            model,
            view,
            mask,
            device=device,
            amp=amp,
            amp_dtype=amp_dtype,
        )
        prefix = f"{split_name}__{mode}__r{replicate}"
        metrics.append(
            {
                "split": split_name,
                "mask_mode": mode,
                "mask_replicate": replicate,
                "mask_entry_id": entry["entry_id"],
                "metrics": result.metrics,
            }
        )
        if save_predictions:
            prediction_key = f"{prefix}__prediction"
            mask_key = f"{prefix}__mask"
            arrays[prediction_key] = result.predictions.numpy()
            arrays[mask_key] = result.mask.numpy().astype(bool, copy=False)
            if result.self_prediction is not None:
                arrays[f"{prefix}__self_prediction"] = (
                    result.self_prediction.numpy()
                )
            if result.neighbor_prediction is not None:
                arrays[f"{prefix}__neighbor_prediction"] = (
                    result.neighbor_prediction.numpy()
                )
            declarations.append(
                {
                    "split": split_name,
                    "mask_mode": mode,
                    "mask_replicate": replicate,
                    "prefix": prefix,
                    "prediction_key": prediction_key,
                    "y_true_key": f"{split_name}__y_true",
                    "mask_key": mask_key,
                    "block_ids_key": f"{split_name}__block_ids",
                    "cell_ids_key": f"{split_name}__cell_ids",
                }
            )
    return metrics, arrays, declarations


def load_run_manifest(
    path: str | Path,
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Load an immutable run manifest and verify its declared artifacts."""

    root = Path(path)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Completed run manifest was not found: {root}")
    if manifest_path.is_symlink():
        raise ExperimentContractError("Run manifest cannot be a symbolic link.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ExperimentContractError("Run manifest root must be a mapping.")
    if (
        manifest.get("format_version") != 1
        or manifest.get("artifact_kind")
        != "normal_true_tissue_spatial_benchmark_run"
        or manifest.get("status") != "complete"
    ):
        raise ExperimentContractError("Unsupported or incomplete run manifest.")
    expected_content_hash = _manifest_content_hash(manifest)
    if manifest.get("manifest_content_sha256") != expected_content_hash:
        raise ExperimentContractError("Run manifest content checksum mismatch.")
    run_id = manifest.get("run_id")
    if (
        not isinstance(run_id, str)
        or len(run_id) != 20
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        raise ExperimentContractError("Run ID schema is invalid.")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ExperimentContractError("Run manifest has no artifact checksums.")
    if any(
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or name == "manifest.json"
        or not isinstance(checksum, str)
        or len(checksum) != 64
        or any(
            character not in "0123456789abcdef"
            for character in checksum
        )
        for name, checksum in files.items()
    ):
        raise ExperimentContractError("Run artifact checksum schema is invalid.")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ExperimentContractError("Run artifact declarations are missing.")
    metrics_name = manifest.get("metrics_file")
    checkpoint_name = artifacts.get("checkpoint")
    prediction_name = artifacts.get("predictions")
    declared_names = [metrics_name, checkpoint_name]
    if prediction_name is not None:
        declared_names.append(prediction_name)
    if any(
        not isinstance(name, str) or not name or Path(name).name != name
        for name in declared_names
    ):
        raise ExperimentContractError("Run artifact filename schema is invalid.")
    if len(set(declared_names)) != len(declared_names):
        raise ExperimentContractError("Run artifact filenames must be distinct.")
    if set(files) != set(declared_names):
        raise ExperimentContractError(
            "Run artifact declarations differ from their checksums."
        )
    if verify_files:
        entries = list(root.iterdir())
        if any(
            candidate.name != "manifest.json"
            and (candidate.is_symlink() or not candidate.is_file())
            for candidate in entries
        ):
            raise ExperimentContractError(
                "Run output contains an undeclared directory or symbolic link."
            )
        actual_files = {
            candidate.name: candidate
            for candidate in entries
            if candidate.name != "manifest.json"
        }
        if set(actual_files) != set(files):
            raise ExperimentContractError(
                "Run artifact file set differs from its manifest."
            )
        for name, candidate in actual_files.items():
            if sha256_file(candidate) != str(files[name]):
                raise ExperimentContractError(
                    f"Run artifact checksum mismatch for {name}."
                )
    metrics_path = root / str(metrics_name)
    if not metrics_path.is_file():
        raise ExperimentContractError("Run metrics artifact is missing.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping):
        raise ExperimentContractError("Run metrics root must be a mapping.")
    validation_metrics = metrics.get("validation")
    test_metrics = metrics.get("test")
    if not isinstance(validation_metrics, list) or not validation_metrics:
        raise ExperimentContractError(
            "Run metrics must contain validation evaluations."
        )
    if not isinstance(test_metrics, list):
        raise ExperimentContractError("Run test metrics must be a list.")
    if any(
        not isinstance(record, Mapping)
        or record.get("split") != "validation"
        for record in validation_metrics
    ) or any(
        not isinstance(record, Mapping) or record.get("split") != "test"
        for record in test_metrics
    ):
        raise ExperimentContractError("Run metric split schema is invalid.")
    sealed_value = manifest.get("sealed_test_opened")
    metrics_opened_value = metrics.get("test_targets_evaluated")
    if not isinstance(sealed_value, bool) or not isinstance(
        metrics_opened_value, bool
    ):
        raise ExperimentContractError(
            "Run test-opening declarations must be boolean."
        )
    sealed_opened = sealed_value
    if metrics_opened_value != sealed_opened:
        raise ExperimentContractError(
            "Run test-opening declarations disagree between manifest and metrics."
        )
    if not sealed_opened and test_metrics:
        raise ExperimentContractError(
            "A sealed run contains undeclared test evaluation metrics."
        )
    if sealed_opened and not test_metrics:
        raise ExperimentContractError(
            "An opened-test run contains no test evaluation metrics."
        )
    standards_lock = manifest.get("standards_lock")
    if not sealed_opened and standards_lock is not None:
        raise ExperimentContractError(
            "A sealed run contains an unexpected standards-lock authorization."
        )
    if sealed_opened:
        if not isinstance(standards_lock, Mapping):
            raise ExperimentContractError(
                "An opened-test run lacks standards-lock authorization."
            )
        locked_job = standards_lock.get("canonical_job")
        job_hash = standards_lock.get("canonical_job_hash")
        matrix_file = standards_lock.get("final_matrix_file")
        required_hex_lengths = {
            "lock_id": 20,
            "artifact_id": 16,
            "lock_manifest_sha256": 64,
            "final_matrix_sha256": 64,
            "canonical_job_hash": 64,
        }
        if (
            standards_lock.get("authorization_version") != 1
            or standards_lock.get("matrix_role")
            not in {"final_execution", "g3"}
            or not isinstance(matrix_file, str)
            or not matrix_file.endswith(".yaml")
            or Path(matrix_file).name != matrix_file
            or not isinstance(locked_job, Mapping)
            or not isinstance(standards_lock.get("condition"), str)
            or any(
                not isinstance(standards_lock.get(name), str)
                or len(str(standards_lock.get(name))) != length
                or any(
                    character not in "0123456789abcdef"
                    for character in str(standards_lock.get(name))
                )
                for name, length in required_hex_lengths.items()
            )
        ):
            raise ExperimentContractError(
                "Run standards-lock authorization schema is invalid."
            )
        try:
            observed_job_hash = canonical_job_hash(locked_job)
        except (StandardsLockError, TypeError, ValueError) as exc:
            raise ExperimentContractError(
                "Run standards-lock canonical job is invalid."
            ) from exc
        if observed_job_hash != job_hash:
            raise ExperimentContractError(
                "Run standards-lock canonical job hash is invalid."
            )
        try:
            locked_condition = condition_name(locked_job)
            locked_seed = int(locked_job.get("seed"))
            manifest_seed = int(manifest.get("model_seed"))
        except (StandardsLockError, TypeError, ValueError) as exc:
            raise ExperimentContractError(
                "Run standards-lock job condition is invalid."
            ) from exc
        if (
            locked_condition != standards_lock.get("condition")
            or locked_seed != manifest_seed
            or _model_key(locked_job.get("model"))
            != _model_key(manifest.get("model_name"))
        ):
            raise ExperimentContractError(
                "Run and standards-lock job declarations disagree."
            )
    evaluations = manifest.get("evaluations")
    if not isinstance(evaluations, list):
        raise ExperimentContractError("Run evaluation declarations must be a list.")
    required_evaluation_keys = {
        "split",
        "mask_mode",
        "mask_replicate",
        "prefix",
        "prediction_key",
        "y_true_key",
        "mask_key",
        "block_ids_key",
        "cell_ids_key",
    }
    if any(
        not isinstance(record, Mapping)
        or not required_evaluation_keys.issubset(record)
        for record in evaluations
    ):
        raise ExperimentContractError("Run evaluation record schema is invalid.")
    declared_splits = {
        str(record.get("split"))
        for record in evaluations
    }
    if len(declared_splits) > 2 or not declared_splits.issubset(
        {"validation", "test"}
    ):
        raise ExperimentContractError("Run evaluation split schema is invalid.")
    if not sealed_opened and "test" in declared_splits:
        raise ExperimentContractError(
            "A sealed run declares test prediction artifacts."
        )
    if prediction_name is None and evaluations:
        raise ExperimentContractError(
            "A run without predictions cannot declare prediction arrays."
        )
    expected_evaluations = len(validation_metrics) + len(test_metrics)
    if prediction_name is not None and len(evaluations) != expected_evaluations:
        raise ExperimentContractError(
            "Run metric and prediction evaluation counts disagree."
        )
    graph_record = manifest.get("graph")
    graph_qc = (
        graph_record.get("qc")
        if isinstance(graph_record, Mapping)
        else None
    )
    if not isinstance(graph_record, Mapping):
        raise ExperimentContractError("Run graph declaration is missing.")
    graph_kind = graph_record.get("kind")
    graph_control = graph_record.get("edge_control")
    control_contract = graph_record.get("edge_control_contract")
    declared_model = manifest.get("model_name")
    declared_model_key = _model_key(declared_model)
    config_record = manifest.get("config")
    configured_model = (
        config_record.get("model")
        if isinstance(config_record, Mapping)
        else None
    )
    if (
        declared_model_key not in _KNOWN_MODEL_KEYS
        or not isinstance(configured_model, Mapping)
        or configured_model.get("name") != declared_model
        or graph_kind not in {"self", "broad_field", "true", "rewired"}
        or graph_control
        not in {"none", "zero", "distance_only", "permuted"}
        or not isinstance(control_contract, Mapping)
        or control_contract.get("name") != graph_control
        or control_contract.get("seed")
        != graph_record.get("edge_control_seed")
    ):
        raise ExperimentContractError("Run graph/control schema is invalid.")
    expected_cell_autonomous_kind = (
        "self"
        if declared_model_key in _SELF_MODEL_KEYS
        else (
            "broad_field"
            if declared_model_key in _BROAD_FIELD_MODEL_KEYS
            else None
        )
    )
    if expected_cell_autonomous_kind is not None:
        kind_matches = graph_kind == expected_cell_autonomous_kind
    else:
        kind_matches = graph_kind in {"true", "rewired"}
    if not kind_matches:
        raise ExperimentContractError(
            "Run model and graph-kind declarations disagree."
        )
    if graph_control != "none" and declared_model_key not in _EDGE_MODEL_KEYS:
        raise ExperimentContractError(
            "Run edge controls are attached to an inapplicable model."
        )
    if graph_control == "permuted":
        if (
            not isinstance(graph_record.get("edge_control_seed"), int)
            or control_contract.get("distance_bin_count")
            != _EDGE_CONTROL_DISTANCE_BINS
            or control_contract.get("distance_bin_fit_scope")
            != "training edges only"
        ):
            raise ExperimentContractError(
                "Permuted-edge control contract is incomplete."
            )
    elif graph_record.get("edge_control_seed") is not None:
        raise ExperimentContractError(
            "A deterministic edge control declares an unused seed."
        )
    rewire_record = graph_record.get("rewire")
    if graph_kind in {"self", "broad_field"}:
        expected_graph_id = (
            "self" if graph_kind == "self" else "broad-spatial-field"
        )
        if (
            graph_record.get("graph_id") != expected_graph_id
            or graph_record.get("base_graph_id") is not None
            or rewire_record is not None
            or graph_control != "none"
        ):
            raise ExperimentContractError(
                "Cell-autonomous run graph declarations are inconsistent."
            )
    elif graph_kind == "rewired":
        achieved = (
            rewire_record.get("achieved")
            if isinstance(rewire_record, Mapping)
            else None
        )
        if (
            not isinstance(achieved, Mapping)
            or achieved.get("degree_preserved_exactly") is not True
            or achieved.get("distance_bins_fitted_per_split") is not True
        ):
            raise ExperimentContractError(
                "Rewired run lacks its degree/distance contract."
            )
    elif rewire_record is not None:
        raise ExperimentContractError(
            "A non-rewired graph declares a rewiring record."
        )
    spatial_control = manifest.get("spatial_control")
    training_record = manifest.get("training")
    training_spatial_control = (
        training_record.get("spatial_control")
        if isinstance(training_record, Mapping)
        else None
    )
    if declared_model_key in _BROAD_FIELD_MODEL_KEYS:
        control_core = (
            dict(spatial_control)
            if isinstance(spatial_control, Mapping)
            else {}
        )
        control_checksum = control_core.pop("basis_fit_checksum", None)
        center = np.asarray(control_core.get("center_um", []), dtype=np.float64)
        scale = np.asarray(control_core.get("scale_um", []), dtype=np.float64)
        constant_axes = control_core.get("constant_axes")
        expected_control_keys = {
            "basis",
            "center_um",
            "constant_axes",
            "constant_axis_fallback_scale_um",
            "contains_cell_or_region_ids",
            "contains_graph_or_neighbor_features",
            "contains_knots_or_spatial_lookup_embeddings",
            "contains_periodic_or_fourier_features",
            "control_name",
            "coordinate_units",
            "feature_names",
            "fit_scope",
            "held_out_refit",
            "maximum_polynomial_degree",
            "n_features",
            "scale_estimator",
            "scale_um",
        }
        if (
            not isinstance(spatial_control, Mapping)
            or set(control_core) != expected_control_keys
            or control_core.get("control_name") != "broad_spatial_field"
            or control_core.get("basis") != BROAD_SPATIAL_BASIS_NAME
            or control_core.get("maximum_polynomial_degree") != 2
            or control_core.get("feature_names")
            != list(BROAD_SPATIAL_FEATURE_NAMES)
            or control_core.get("n_features")
            != len(BROAD_SPATIAL_FEATURE_NAMES)
            or control_core.get("coordinate_units") != "micrometers"
            or control_core.get("scale_estimator")
            != "training_population_standard_deviation"
            or control_core.get("constant_axis_fallback_scale_um") != 1.0
            or control_core.get("fit_scope")
            != "training split coordinates only"
            or control_core.get("held_out_refit") is not False
            or control_core.get("contains_cell_or_region_ids") is not False
            or control_core.get("contains_periodic_or_fourier_features")
            is not False
            or control_core.get(
                "contains_knots_or_spatial_lookup_embeddings"
            )
            is not False
            or control_core.get("contains_graph_or_neighbor_features")
            is not False
            or not isinstance(constant_axes, list)
            or len(constant_axes) != 2
            or any(type(value) is not bool for value in constant_axes)
            or center.shape != (2,)
            or scale.shape != (2,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
            or (
                isinstance(constant_axes, list)
                and len(constant_axes) == 2
                and any(
                    is_constant and scale[index] != 1.0
                    for index, is_constant in enumerate(constant_axes)
                )
            )
            or control_checksum != _canonical_hash(control_core)
            or training_spatial_control != spatial_control
        ):
            raise ExperimentContractError(
                "Broad spatial-field provenance is missing or invalid."
            )
    elif spatial_control is not None or training_spatial_control is not None:
        raise ExperimentContractError(
            "A non-spatial-field model declares coordinate control provenance."
        )
    cross_group_edges = (
        graph_qc.get("cross_group_edges")
        if isinstance(graph_qc, Mapping)
        else None
    )
    if not isinstance(cross_group_edges, int) or cross_group_edges != 0:
        raise ExperimentContractError(
            "Run graph manifest does not certify zero cross-split edges."
        )
    split_counts = manifest.get("split_node_counts")
    if not isinstance(split_counts, Mapping):
        raise ExperimentContractError("Run split node counts are missing.")
    if any(
        not isinstance(split_counts.get(name), int)
        or int(split_counts[name]) <= 0
        for name in ("train", "validation")
    ):
        raise ExperimentContractError(
            "Run train/validation node counts must be positive integers."
        )
    if sealed_opened:
        if (
            not isinstance(split_counts.get("test"), int)
            or int(split_counts["test"]) <= 0
        ):
            raise ExperimentContractError(
                "An opened-test run must record its test node count."
            )
    elif split_counts.get("test") is not None:
        raise ExperimentContractError(
            "A sealed run must not materialize a test split view."
        )
    if prediction_name is not None and verify_files:
        prediction_path = root / str(prediction_name)
        try:
            with np.load(prediction_path, allow_pickle=False) as archive:
                prediction_keys = set(archive.files)
        except (OSError, ValueError) as exc:
            raise ExperimentContractError(
                "Run prediction artifact is not a safe NPZ archive."
            ) from exc
        declared_prediction_keys = {
            str(record[key])
            for record in evaluations
            for key in (
                "prediction_key",
                "y_true_key",
                "mask_key",
                "block_ids_key",
                "cell_ids_key",
            )
            if key in record
        }
        if not declared_prediction_keys.issubset(prediction_keys):
            raise ExperimentContractError(
                "Run prediction declarations reference missing arrays."
            )
        if not sealed_opened and any(
            key.startswith("test__") for key in prediction_keys
        ):
            raise ExperimentContractError(
                "A sealed run contains test prediction arrays."
            )
    return manifest


def run_experiment(
    prepared_path: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path,
    model_name: str,
    model_seed: int,
    graph_overrides: Mapping[str, Any] | None = None,
    model_overrides: Mapping[str, Any] | None = None,
    training_overrides: Mapping[str, Any] | None = None,
    rewired: bool = False,
    rewire_seed: int = 0,
    swaps_per_edge: float = 1.0,
    edge_control: str = "none",
    pretrained_b0_checkpoint: str | Path | None = None,
    g3_frozen_epochs: int = 10,
    g3_joint_learning_rate: float = 1e-4,
    evaluate_test: bool = False,
    save_predictions: bool = True,
    standards_lock_path: str | Path | None = None,
    command: Sequence[str] | None = None,
) -> Path:
    """Train, evaluate, and atomically publish one isolated model seed."""

    prepared_root = Path(prepared_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite run output: {destination}")
    if prepared_root == destination or prepared_root in destination.parents:
        raise ExperimentContractError(
            "Run output cannot be published inside the immutable prepared artifact."
        )
    project = Path(project_root).resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"Project root was not found: {project}")
    graph_override_values = _validate_override_keys(
        graph_overrides, _GRAPH_OVERRIDE_KEYS, name="graph"
    )
    model_override_values = _validate_override_keys(
        model_overrides, _MODEL_OVERRIDE_KEYS, name="model"
    )
    training_override_values = _validate_override_keys(
        training_overrides, _TRAINING_OVERRIDE_KEYS, name="training"
    )
    model_key = _model_key(model_name)
    if model_key not in _KNOWN_MODEL_KEYS:
        raise ExperimentContractError(
            f"Unknown model {model_name!r}; expected B0, B0-matched, "
            "Broad-Field, B1, G1, G2, or G3."
        )
    self_model = model_key in _SELF_MODEL_KEYS
    broad_field_model = model_key in _BROAD_FIELD_MODEL_KEYS
    cell_autonomous_model = self_model or broad_field_model
    edge_model = model_key in _EDGE_MODEL_KEYS
    g3_model = model_key in {"g3", "additive", "additiveedgemessage"}
    applicable_model_keys = set(_model_kwargs(model_name, {}))
    inapplicable_model_keys = sorted(
        set(model_override_values).difference(applicable_model_keys)
    )
    if inapplicable_model_keys:
        raise ExperimentContractError(
            f"Model {model_name!r} does not consume overrides: "
            f"{', '.join(inapplicable_model_keys)}"
        )
    if not rewired:
        inactive_rewire_keys = sorted(
            set(graph_override_values).intersection(
                _REWIRE_ONLY_GRAPH_OVERRIDES
            )
        )
        if inactive_rewire_keys:
            raise ExperimentContractError(
                "Rewiring-only graph overrides require rewired=True: "
                f"{', '.join(inactive_rewire_keys)}"
            )
    normalized_edge_control = str(edge_control).lower().replace("-", "_")
    if normalized_edge_control == "full":
        normalized_edge_control = "none"
    if normalized_edge_control not in {
        "none",
        "zero",
        "distance_only",
        "permuted",
    }:
        raise ExperimentContractError(
            "edge_control must be one of none, zero, distance_only, or permuted"
        )
    if cell_autonomous_model and graph_override_values:
        raise ExperimentContractError(
            "Cell-autonomous controls cannot vary graph construction."
        )
    if cell_autonomous_model and rewired:
        raise ExperimentContractError(
            "A cell-autonomous control cannot be labeled as graph-rewired."
        )
    if normalized_edge_control != "none" and not edge_model:
        raise ExperimentContractError(
            "Edge-attribute controls are defined only for G2/G3 models."
        )
    if g3_model and pretrained_b0_checkpoint is None:
        raise ExperimentContractError(
            "G3 requires a verified, same-seed pretrained B0 checkpoint."
        )
    if not g3_model and pretrained_b0_checkpoint is not None:
        raise ExperimentContractError(
            "A pretrained B0 checkpoint is accepted only for staged G3."
        )
    if evaluate_test and not save_predictions:
        raise ExperimentContractError(
            "A sealed-test run must preserve predictions for audit."
        )
    if not evaluate_test and standards_lock_path is not None:
        raise ExperimentContractError(
            "A standards lock may be attached only to an open-test run."
        )
    lock_authorization: dict[str, Any] | None = None
    if evaluate_test:
        if standards_lock_path is None:
            raise ExperimentContractError(
                "Opening the sealed test requires a verified standards lock."
            )
        canonical_model_name = _canonical_matrix_model_name(model_key)
        if str(model_name).strip().lower() != canonical_model_name:
            raise ExperimentContractError(
                "An open-test run must use the exact model name declared "
                "by its locked final matrix."
            )
        requested_locked_job = _locked_job_request(
            model_key=model_key,
            model_seed=int(model_seed),
            graph_overrides=graph_override_values,
            model_overrides=model_override_values,
            training_overrides=training_override_values,
            rewired=bool(rewired),
            rewire_seed=int(rewire_seed),
            swaps_per_edge=float(swaps_per_edge),
            edge_control=normalized_edge_control,
            pretrained_b0_checkpoint=pretrained_b0_checkpoint,
            g3_frozen_epochs=int(g3_frozen_epochs),
            g3_joint_learning_rate=float(g3_joint_learning_rate),
        )
        try:
            lock_authorization = authorize_locked_test_job(
                standards_lock_path,
                requested_locked_job,
            )
        except (FileNotFoundError, StandardsLockError) as exc:
            raise ExperimentContractError(
                f"Open-test standards-lock authorization failed: {exc}"
            ) from exc

    manifest, arrays, bundles = load_prepared_artifact(prepared_root)
    if arrays is None:
        raise RuntimeError("Prepared arrays were not loaded.")
    prepared_artifact_id = str(manifest["artifact_id"])
    prepared_manifest_sha256 = sha256_file(
        prepared_root / "manifest.json"
    )
    prepared_config = manifest["configuration"]
    graph_config = {
        **dict(prepared_config["graph"]),
        **graph_override_values,
    }
    minimum_rewire_fraction = float(
        graph_config.get("min_rewire_success_fraction", 0.5)
    )
    maximum_rewire_distance_change = float(
        graph_config.get("max_rewire_distance_mean_change", 0.2)
    )
    if not 0.0 <= minimum_rewire_fraction <= 1.0:
        raise ExperimentContractError(
            "min_rewire_success_fraction must lie in [0, 1]."
        )
    if (
        not np.isfinite(maximum_rewire_distance_change)
        or maximum_rewire_distance_change < 0
    ):
        raise ExperimentContractError(
            "max_rewire_distance_mean_change must be finite and nonnegative."
        )
    if int(graph_config.get("rewire_distance_bins", 8)) <= 0:
        raise ExperimentContractError("rewire_distance_bins must be positive.")
    model_config = dict(prepared_config.get("model", {}))
    # The prepared model name is a planning default, not permission to
    # relabel a run that explicitly selects another ladder member.
    model_config.pop("name", None)
    model_config.update(model_override_values)
    effective_model_config = _model_kwargs(model_name, model_config)
    training_config_values = {
        "model_seed": int(model_seed),
        "edge_dropout": float(graph_config.get("edge_dropout", 0.1)),
        **training_override_values,
    }
    started = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()
    topology_overridden = bool(
        set(graph_override_values).intersection(_GRAPH_CONSTRUCTION_KEYS)
    )
    prepared_topology = (
        _prepared_graph(manifest, arrays)
        if not topology_overridden
        else None
    )
    if not rewired and prepared_topology is not None:
        graph, edge_attributes = prepared_topology
    else:
        graph, edge_attributes = _make_graph(
            arrays,
            graph_config,
            rewired=bool(rewired),
            rewire_seed=int(rewire_seed),
            swaps_per_edge=float(swaps_per_edge),
            base_graph=(
                prepared_topology[0]
                if rewired and prepared_topology is not None
                else None
            ),
        )
    edge_splits = arrays["split_labels"][graph.edge_index[0]]
    edge_attributes = _edge_control(
        edge_attributes,
        graph.edge_attr_names,
        graph.edge_index,
        edge_splits,
        normalized_edge_control,
        seed=int(rewire_seed),
    )
    train_view, train_index = _split_view(
        "train", arrays, graph, edge_attributes
    )
    validation_view, validation_index = _split_view(
        "validation", arrays, graph, edge_attributes
    )
    test_view: GraphSplitView | None = None
    test_index: np.ndarray | None = None
    if evaluate_test:
        test_view, test_index = _split_view("test", arrays, graph, edge_attributes)

    model = build_model(
        model_name,
        num_genes=train_view.num_genes,
        node_covariate_dim=train_view.node_covariate_dim,
        edge_attribute_dim=train_view.edge_attribute_dim,
        seed=int(model_seed),
        deterministic=bool(training_config_values.get("deterministic", True)),
        **effective_model_config,
    )
    fit_config = _training_config(
        prepared_config,
        training_config_values,
    )
    validation_mask = bundles["validation"].get(
        "validation", "node", replicate=0
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pretrained_b0_record: dict[str, Any] | None = None
    if g3_model:
        assert pretrained_b0_checkpoint is not None
        checkpoint_path = Path(pretrained_b0_checkpoint).resolve()
        if not checkpoint_path.is_file() or checkpoint_path.name != "model_state.pt":
            raise FileNotFoundError(
                "G3 pretrained checkpoint must be a model_state.pt run artifact."
            )
        source_manifest = load_run_manifest(checkpoint_path.parent)
        if _model_key(source_manifest["model_name"]) not in {
            "b0",
            "self",
            "selfmlp",
        }:
            raise ExperimentContractError("G3 source checkpoint is not a B0 run.")
        if int(source_manifest["model_seed"]) != int(model_seed):
            raise ExperimentContractError(
                "G3 and its pretrained B0 must use the same model seed."
            )
        if (
            source_manifest["prepared_artifact"]["artifact_id"]
            != prepared_artifact_id
        ):
            raise ExperimentContractError(
                "G3 and its pretrained B0 use different prepared artifacts."
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        pretrained_b0_record = {
            "run_id": source_manifest["run_id"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_seed": int(source_manifest["model_seed"]),
        }
        training_result = fit_staged_g3(
            model,
            checkpoint,
            train_view,
            validation_view,
            fit_config,
            frozen_epochs=int(g3_frozen_epochs),
            joint_learning_rate=float(g3_joint_learning_rate),
            validation_mask=validation_mask,
            validation_mode="node",
        )
    else:
        training_result = fit_model(
            model,
            train_view,
            validation_view,
            fit_config,
            validation_mask=validation_mask,
            validation_mode="node",
        )

    validation_metrics, prediction_arrays, declarations = _evaluation_records(
        model,
        validation_view,
        "validation",
        bundles["validation"],
        device=str(fit_config.device),
        amp=fit_config.amp,
        amp_dtype=fit_config.amp_dtype,
        save_predictions=save_predictions,
    )
    test_metrics: list[dict[str, Any]] = []
    if evaluate_test:
        assert test_view is not None
        test_metrics, test_arrays, test_declarations = _evaluation_records(
            model,
            test_view,
            "test",
            bundles["test"],
            device=str(fit_config.device),
            amp=fit_config.amp,
            amp_dtype=fit_config.amp_dtype,
            save_predictions=True,
        )
        prediction_arrays.update(test_arrays)
        declarations.extend(test_declarations)

    graph_id = _graph_id(graph_config, rewired=rewired)
    graph_kind = (
        "self"
        if self_model
        else (
            "broad_field"
            if broad_field_model
            else ("rewired" if rewired else "true")
        )
    )
    declared_graph_id = (
        "self"
        if graph_kind == "self"
        else (
            "broad-spatial-field"
            if graph_kind == "broad_field"
            else graph_id
        )
    )
    base_graph_id = _graph_id(graph_config, rewired=False)
    declared_base_graph_id = (
        None
        if graph_kind in {"self", "broad_field"}
        else base_graph_id
    )
    spatial_control_record = training_result.spatial_control_provenance
    if broad_field_model and not isinstance(spatial_control_record, Mapping):
        raise RuntimeError(
            "broad spatial-field training did not return basis provenance"
        )
    if not broad_field_model and spatial_control_record is not None:
        raise RuntimeError(
            "ordinary model training returned unexpected coordinate provenance"
        )
    rewire_contract = (
        {
            "seed": int(rewire_seed),
            "swaps_per_edge": float(swaps_per_edge),
            "distance_bins": int(graph_config.get("rewire_distance_bins", 8)),
            "minimum_success_fraction": float(
                graph_config.get("min_rewire_success_fraction", 0.5)
            ),
            "maximum_distance_mean_change": float(
                graph_config.get("max_rewire_distance_mean_change", 0.2)
            ),
            "achieved": dict(graph.metadata),
        }
        if rewired
        else None
    )
    edge_control_seed = (
        int(rewire_seed) if normalized_edge_control == "permuted" else None
    )
    edge_control_contract = {
        "name": normalized_edge_control,
        "seed": edge_control_seed,
        "distance_bin_count": (
            _EDGE_CONTROL_DISTANCE_BINS
            if normalized_edge_control == "permuted"
            else None
        ),
        "distance_bin_fit_scope": (
            "training edges only"
            if normalized_edge_control == "permuted"
            else None
        ),
        "application_scope": (
            "independently within each split and distance bin"
            if normalized_edge_control == "permuted"
            else "all split-restricted edges"
        ),
    }
    completed = datetime.now(timezone.utc)
    run_contract = {
        "prepared_artifact_id": prepared_artifact_id,
        "model_name": str(model_name).lower(),
        "model_seed": int(model_seed),
        "graph_id": declared_graph_id,
        "graph_kind": graph_kind,
        "base_graph_id": declared_base_graph_id,
        "graph_config": graph_config,
        "model_config": effective_model_config,
        "training_config": asdict(fit_config),
        "rewire": rewire_contract,
        "edge_control": edge_control_contract,
        "evaluate_test": bool(evaluate_test),
        "save_predictions": bool(save_predictions),
        "pretrained_b0": pretrained_b0_record,
        "g3_frozen_epochs": int(g3_frozen_epochs) if g3_model else None,
        "g3_joint_learning_rate": (
            float(g3_joint_learning_rate) if g3_model else None
        ),
        "spatial_control": spatial_control_record,
        "standards_lock": lock_authorization,
    }
    run_id = _canonical_hash(run_contract)[:20]
    run_manifest: dict[str, Any] = {
        "format_version": 1,
        "artifact_kind": "normal_true_tissue_spatial_benchmark_run",
        "run_id": run_id,
        "status": "complete",
        "model_name": str(model_name).lower(),
        "model_seed": int(model_seed),
        "standards_lock": lock_authorization,
        "prepared_artifact": {
            "path": str(prepared_root),
            "artifact_id": prepared_artifact_id,
            "manifest_sha256": prepared_manifest_sha256,
            "split_id": manifest["split"]["split_id"],
            "validation_mask_bundle_id": bundles["validation"].bundle_id,
            "test_mask_bundle_id": bundles["test"].bundle_id,
        },
        "graph": {
            "graph_id": declared_graph_id,
            "kind": graph_kind,
            "base_graph_id": declared_base_graph_id,
            "config": graph_config,
            "edge_control": normalized_edge_control,
            "edge_control_seed": edge_control_seed,
            "edge_control_contract": edge_control_contract,
            "qc": graph.qc.to_dict(),
            "rewire": rewire_contract,
        },
        "spatial_control": spatial_control_record,
        "config": {
            "model": {
                "name": str(model_name).lower(),
                **effective_model_config,
            },
            "run": {
                "model_seed": int(model_seed),
                "evaluate_test": bool(evaluate_test),
                "save_predictions": bool(save_predictions),
                "edge_control": normalized_edge_control,
                "edge_control_seed": edge_control_seed,
                "rewired": bool(rewired),
            },
            "graph": graph_config,
            "training": asdict(fit_config),
        },
        "training": {
            "best_epoch": training_result.best_epoch,
            "best_validation_loss": training_result.best_validation_loss,
            "stopped_early": training_result.stopped_early,
            "validation_mask_checksum": (
                training_result.validation_mask_checksum
            ),
            "graph_execution": training_result.graph_execution,
            "history": [asdict(record) for record in training_result.history],
            "protocol": training_result.training_protocol,
            "stages": [
                asdict(record) for record in training_result.stage_provenance
            ],
            "pretrained_self_source": training_result.pretrained_self_source,
            "pretrained_self_checksum": (
                training_result.pretrained_self_checksum
            ),
            "pretrained_b0": pretrained_b0_record,
            "spatial_control": spatial_control_record,
        },
        "evaluations": declarations,
        "metrics_file": "metrics.json",
        "artifacts": {
            "checkpoint": "model_state.pt",
            "predictions": "predictions.npz" if save_predictions else None,
        },
        "split_node_counts": {
            "train": int(len(train_index)),
            "validation": int(len(validation_index)),
            "test": int(len(test_index)) if test_index is not None else None,
        },
        "sealed_test_opened": bool(evaluate_test),
        "timing": {
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "runtime_seconds": time.monotonic() - start_monotonic,
        },
        "resources": {
            "dependencies": _dependency_state(),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
        },
        "provenance": {
            "command": list(command or []),
            "working_directory": str(Path.cwd()),
            "git": _git_state(project),
        },
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        torch.save(
            {
                "run_id": run_id,
                "prepared_artifact_id": prepared_artifact_id,
                "model_name": str(model_name).lower(),
                "model_seed": int(model_seed),
                "model_config": effective_model_config,
                "spatial_control": spatial_control_record,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
            },
            temporary / "model_state.pt",
        )
        _write_json(
            temporary / "metrics.json",
            {
                "validation": validation_metrics,
                "test": test_metrics,
                "test_targets_evaluated": bool(evaluate_test),
            },
        )
        if save_predictions:
            np.savez_compressed(
                temporary / "predictions.npz",
                **{
                    key: prediction_arrays[key]
                    for key in sorted(prediction_arrays)
                },
            )
        file_records = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        run_manifest["files"] = file_records
        run_manifest["manifest_content_sha256"] = _manifest_content_hash(
            run_manifest
        )
        _write_json(temporary / "manifest.json", run_manifest)
        load_run_manifest(temporary)
        current_prepared, _, _ = load_prepared_artifact(
            prepared_root, load_arrays=False
        )
        if (
            str(current_prepared["artifact_id"]) != prepared_artifact_id
            or sha256_file(prepared_root / "manifest.json")
            != prepared_manifest_sha256
        ):
            raise ExperimentContractError(
                "Prepared artifact changed during the run."
            )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Run output appeared before atomic publication: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


__all__ = [
    "ExperimentContractError",
    "load_run_manifest",
    "run_experiment",
]
