#!/usr/bin/env python3
"""Run a privacy-safe toy interpretability analysis on the final full-core G2.

This is an exploratory, outcome-conditioned analysis of one transductively
fitted core.  It verifies every upstream identity before loading the final
checkpoint and emits aggregate JSON/Markdown only.  Attention is treated as a
routing measurement, not biological importance; repeated matched global-edge
and organizer-channel interventions provide bounded model-sensitivity checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
import yaml


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.dense_gat import (  # noqa: E402
    ReceiverChunkedEdgeConditionedGATv2,
)
from spatial_benchmark.full_core import (  # noqa: E402
    FullCoreData,
    ReceiverSortedGraph,
    build_exact_mutual_knn_graph,
    load_and_refit_full_core,
)
from spatial_benchmark.masking import (  # noqa: E402
    FixedMaskBundle,
    MaskSpec,
    create_fixed_mask_bundle,
    derive_mask_seed,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.run_archive import verify_run_bundle  # noqa: E402


PROTOCOL = "held_in_full_core_fixed_budget"
CAMPAIGN_ID = "cmp_20260725_full_core_high_k_capacity"
EXPECTED_DATASET_SHA256 = (
    "a112fbb2bdf929197759c82913fc379c4df797f75676457bcd10419fe7a969d5"
)
EXPECTED_GRAPH_SHA256 = (
    "2469064e2fe14b48f642fca09a546d9e420d9fd851b9668996da62ee8246d060"
)
EXPECTED_NODES = 24_245
EXPECTED_GENES = 1_000
EXPECTED_DIRECTED_EDGES = 21_029_944
HASH_SAMPLE_SIZE = 128
OUTCOME_SAMPLE_SIZE = 64
DELETION_FRACTION = 0.10
MATCHED_NULL_REPLICATES = 8
SENDER_PERTURBATION_SOURCES = 256
MIN_RELATIVE_HUBER_CONTRAST = 0.02
MIN_PREDICTION_MAE_CONTRAST = 0.005
MIN_ORGANIZER_SCORE_ENRICHMENT = 0.10
ORGANIZER_GENES = ("CCL19", "CCL21", "CXCL13")
RECEIVER_PROGRAM_GENES = (
    "CCR7",
    "CXCR5",
    "MS4A1",
    "CD3D",
    "CD3E",
    "CD79A",
    "CD74",
    "HLA-DRA",
)
MAXIMUM_INTERPRETATION = (
    "model-implied TLS-related predictive dependency in one "
    "transductively fitted core"
)


class InterpretabilityContractError(RuntimeError):
    """Raised when analysis inputs violate the locked campaign contract."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Finalized successful full-core G2 run bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory for analysis.json and report.md.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device; the production graph is intended for CUDA.",
    )
    return parser


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpretabilityContractError(
            f"{location} must contain a mapping"
        )
    return value


def _expect(observed: Any, expected: Any, location: str) -> None:
    if observed != expected:
        raise InterpretabilityContractError(
            f"{location} must be {expected!r}, found {observed!r}"
        )


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise InterpretabilityContractError(
            f"Cannot parse required JSON artifact: {path.name}"
        ) from error
    return dict(_mapping(value, path.name))


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise InterpretabilityContractError(
            f"Cannot parse required YAML artifact: {path.name}"
        ) from error
    return dict(_mapping(value, path.name))


def _reject_bundle_symlinks(root: Path) -> None:
    symlinks = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise InterpretabilityContractError(
            "Interpretability input bundle may not contain symlinks: "
            + ", ".join(sorted(symlinks))
        )


def _validate_locked_protocol(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    """Reject anything except the prespecified final 200-epoch G2 run."""

    campaign = _mapping(config.get("campaign"), "config.campaign")
    model = _mapping(config.get("model"), "config.model")
    graph = _mapping(config.get("graph"), "config.graph")
    trainer = _mapping(config.get("trainer"), "config.trainer")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    dataset = _mapping(config.get("dataset"), "config.dataset")
    features = _mapping(config.get("features"), "config.features")

    _expect(campaign.get("campaign_id"), CAMPAIGN_ID, "campaign.campaign_id")
    _expect(manifest.get("campaign_id"), CAMPAIGN_ID, "manifest.campaign_id")
    _expect(config.get("seed"), 0, "config.seed")
    _expect(config.get("fold"), 0, "config.fold")

    locked_model = {
        "name": "g2",
        "family": "edge_conditioned_gatv2",
        "embedding_dim": 512,
        "hidden_dim": 512,
        "graph_layers": 2,
        "attention_heads": 4,
        "ffn_dim": 512,
        "decoder_dim": 512,
        "edge_hidden_dim": 64,
        "edge_embedding_dim": 64,
        "dropout": 0.10,
        "attention_dropout": 0.10,
        "receiver_chunk_size": 512,
        "activation_checkpointing": True,
        "exact_receiver_partitioning": True,
        "implicit_self_loops": False,
        "trainable_node_identifiers": False,
        "trainable_edge_identifiers": False,
    }
    for field, expected in locked_model.items():
        _expect(model.get(field), expected, f"model.{field}")
    if model.get("attention_head_dim") not in {None, 128}:
        raise InterpretabilityContractError(
            "model.attention_head_dim must be absent or 128"
        )
    _expect(features.get("use_edge_features"), True, "features.use_edge_features")

    locked_graph = {
        "kind": "exact_spatial_knn_radius_guard",
        "k": 1000,
        "neighbor_k": 1000,
        "radius_guard_um": 650.0,
        "radius_um": 650.0,
        "symmetry": "mutual",
        "edge_dropout": 0.0,
        "self_loops": False,
        "full_core_graph": True,
        "expected_materialized_graph_sha256": EXPECTED_GRAPH_SHA256,
        "expected_directed_edges": EXPECTED_DIRECTED_EDGES,
    }
    for field, expected in locked_graph.items():
        _expect(graph.get(field), expected, f"graph.{field}")
    _expect(
        graph.get("candidate_selection"),
        "exact_knn_before_radius_guard_validation",
        "graph.candidate_selection",
    )
    if bool(graph.get("neighbor_sampling", False)):
        raise InterpretabilityContractError(
            "graph.neighbor_sampling is prohibited"
        )

    locked_trainer = {
        "max_epochs": 200,
        "fixed_epoch_budget": True,
        "early_stopping": False,
        "validation_every": None,
        "neighbor_sampling": False,
        "restore_best": False,
        "primary_checkpoint_role": "last",
        "checkpoint_policy": "last_only",
        "graph_execution": "full_core_exact_no_neighbor_sampling",
    }
    for field, expected in locked_trainer.items():
        _expect(trainer.get(field), expected, f"trainer.{field}")
    if trainer.get("diagnostic_resource_pilot") is True:
        raise InterpretabilityContractError(
            "Resource-pilot checkpoints are not interpretable outcomes"
        )

    _expect(evaluation.get("protocol"), PROTOCOL, "evaluation.protocol")
    _expect(evaluation.get("splits"), ["fit"], "evaluation.splits")
    _expect(
        evaluation.get("canonical_prediction_split"),
        "fit",
        "evaluation.canonical_prediction_split",
    )
    _expect(
        evaluation.get("primary_metric"),
        "fit/whole_node/masked_huber",
        "evaluation.primary_metric",
    )
    _expect(
        evaluation.get("mask_modes"),
        ["partial_gene", "whole_node", "spatial_block"],
        "evaluation.mask_modes",
    )
    _expect(
        evaluation.get("mask_replicates_per_mode"),
        3,
        "evaluation.mask_replicates_per_mode",
    )
    _expect(
        evaluation.get("generalization_estimate"),
        False,
        "evaluation.generalization_estimate",
    )
    _expect(
        evaluation.get("validation_or_test_selection"),
        False,
        "evaluation.validation_or_test_selection",
    )

    _expect(
        dataset.get("dataset_fingerprint"),
        EXPECTED_DATASET_SHA256,
        "dataset.dataset_fingerprint",
    )
    _expect(
        dataset.get("biological_target_count"),
        EXPECTED_GENES,
        "dataset.biological_target_count",
    )
    _expect(
        dataset.get("preprocessing_fit_scope"),
        "all_nodes_transductive",
        "dataset.preprocessing_fit_scope",
    )
    _expect(
        dataset.get("validation_or_test_partition_present"),
        False,
        "dataset.validation_or_test_partition_present",
    )

    _expect(summary.get("status"), "success", "summary.status")
    _expect(
        summary.get("training_exit_status"),
        "success",
        "summary.training_exit_status",
    )
    _expect(
        summary.get("evaluation_protocol"),
        PROTOCOL,
        "summary.evaluation_protocol",
    )
    _expect(summary.get("model_name"), "g2", "summary.model_name")
    _expect(summary.get("model_seed"), 0, "summary.model_seed")
    _expect(summary.get("final_epoch"), 199, "summary.final_epoch")
    _expect(
        summary.get("fixed_epoch_budget"),
        200,
        "summary.fixed_epoch_budget",
    )
    _expect(summary.get("checkpoint_role"), "last", "summary.checkpoint_role")
    _expect(
        summary.get("diagnostic_resource_pilot"),
        False,
        "summary.diagnostic_resource_pilot",
    )
    _expect(
        summary.get("conclusion_eligible"),
        True,
        "summary.conclusion_eligible",
    )
    _expect(
        summary.get("generalization_estimate"),
        False,
        "summary.generalization_estimate",
    )
    _expect(
        summary.get("graph_sha256"),
        EXPECTED_GRAPH_SHA256,
        "summary.graph_sha256",
    )
    _expect(
        summary.get("graph_directed_edges"),
        EXPECTED_DIRECTED_EDGES,
        "summary.graph_directed_edges",
    )
    selection = _mapping(
        summary.get("canonical_prediction_selection"),
        "summary.canonical_prediction_selection",
    )
    _expect(selection.get("split"), "fit", "canonical selection split")
    _expect(
        selection.get("mask_mode"),
        "whole_node",
        "canonical selection mask_mode",
    )
    _expect(
        selection.get("mask_replicate"),
        0,
        "canonical selection mask_replicate",
    )


def _state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    """Reproduce the training checkpoint's exact state checksum."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = torch.as_tensor(state_dict[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _load_verified_checkpoint(
    root: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    training_provenance: Mapping[str, Any],
) -> tuple[Mapping[str, Tensor], Mapping[str, Any]]:
    path = root / "checkpoints" / "last.ckpt"
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise InterpretabilityContractError(
            "Final checkpoint must be a non-empty regular file"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise InterpretabilityContractError(
            "Final checkpoint failed safe weights-only loading"
        ) from error
    checkpoint = _mapping(payload, "checkpoint")
    state = _mapping(
        checkpoint.get("model_state_dict"), "checkpoint.model_state_dict"
    )
    if not all(isinstance(value, Tensor) for value in state.values()):
        raise InterpretabilityContractError(
            "Checkpoint state dict contains non-tensor values"
        )

    expected_fields = {
        "schema_version": 1,
        "run_id": root.name,
        "checkpoint_role": "last",
        "checkpoint_policy": "final_epoch_no_validation_selection",
        "training_protocol": PROTOCOL,
        "model_name": "g2",
        "epoch": 199,
        "fixed_epoch_budget": 200,
        "full_core_preprocessing_sha256": EXPECTED_DATASET_SHA256,
        "graph_sha256": EXPECTED_GRAPH_SHA256,
        "evaluation_mask_bundle_sha256": summary.get(
            "evaluation_mask_bundle_sha256"
        ),
    }
    for field, expected in expected_fields.items():
        _expect(checkpoint.get(field), expected, f"checkpoint.{field}")
    _expect(
        dict(_mapping(checkpoint.get("model_config"), "checkpoint.model_config")),
        dict(_mapping(config.get("model"), "config.model")),
        "checkpoint.model_config",
    )

    observed_state_sha = _state_dict_sha256(state)
    _expect(
        checkpoint.get("state_dict_sha256"),
        observed_state_sha,
        "checkpoint.state_dict_sha256",
    )
    _expect(
        training_provenance.get("state_dict_sha256"),
        observed_state_sha,
        "training provenance state_dict_sha256",
    )
    _expect(
        training_provenance.get("training_protocol"),
        PROTOCOL,
        "training provenance protocol",
    )
    _expect(
        training_provenance.get("checkpoint_policy"),
        "final_epoch_no_validation_selection",
        "training provenance checkpoint policy",
    )
    _expect(
        training_provenance.get("graph_execution"),
        "full_core_exact_no_neighbor_sampling",
        "training provenance graph execution",
    )
    _expect(
        training_provenance.get("final_epoch"),
        199,
        "training provenance final_epoch",
    )
    _expect(
        training_provenance.get("fixed_epoch_budget"),
        200,
        "training provenance fixed_epoch_budget",
    )
    return state, checkpoint


def _resolve_prepared_artifact(config: Mapping[str, Any]) -> Path:
    dataset = _mapping(config.get("dataset"), "config.dataset")
    reference = dataset.get("prepared_artifact_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise InterpretabilityContractError(
            "dataset.prepared_artifact_reference is required"
        )
    path = Path(reference)
    if not path.is_absolute():
        project_root = current_paths(anchor=_BOOTSTRAP_ROOT).project_root
        path = project_root / path
    return path


def _rebuild_verified_core(
    config: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
) -> FullCoreData:
    core = load_and_refit_full_core(_resolve_prepared_artifact(config))
    if core.n_nodes != EXPECTED_NODES or core.n_genes != EXPECTED_GENES:
        raise InterpretabilityContractError(
            "Materialized full core has an unexpected shape"
        )
    _expect(
        core.checksums.preprocessing_sha256,
        EXPECTED_DATASET_SHA256,
        "materialized preprocessing checksum",
    )
    _expect(
        dict(
            _mapping(
                input_provenance.get("preprocessing_checksums"),
                "input provenance preprocessing_checksums",
            )
        ),
        core.checksums.to_dict(),
        "input provenance preprocessing checksums",
    )
    if core.preprocessing_qc.protected_identifier_arrays_returned:
        raise InterpretabilityContractError(
            "Full-core loader unexpectedly returned protected identifiers"
        )
    return core


def _rebuild_verified_graph(
    core: FullCoreData,
    config: Mapping[str, Any],
    input_provenance: Mapping[str, Any],
) -> ReceiverSortedGraph:
    graph_config = _mapping(config.get("graph"), "config.graph")
    graph = build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=int(graph_config["k"]),
        radius_guard_um=float(graph_config["radius_guard_um"]),
        query_chunk_size=int(graph_config.get("query_chunk_size", 2048)),
        receiver_chunk_size=int(
            graph_config.get("receiver_shard_size", 512)
        ),
        mutual_search_chunk_size=int(
            graph_config.get("mutual_search_chunk_size", 4_000_000)
        ),
        workers=int(graph_config.get("construction_workers", 1)),
        epsilon=float(
            graph_config.get("edge_standardizer_epsilon", 1e-8)
        ),
    )
    _expect(graph.n_nodes, core.n_nodes, "materialized graph node count")
    _expect(graph.k, 1000, "materialized graph k")
    _expect(
        graph.qc.n_directed_edges,
        EXPECTED_DIRECTED_EDGES,
        "materialized graph directed edge count",
    )
    _expect(
        graph.checksums.graph_sha256,
        EXPECTED_GRAPH_SHA256,
        "materialized graph checksum",
    )
    if (
        not graph.qc.receiver_sorted
        or graph.qc.self_loops
        or graph.qc.duplicate_directed_edges
        or not graph.qc.directed_edge_pairs_are_symmetric
    ):
        raise InterpretabilityContractError(
            "Materialized graph violates exact receiver-sorted mutual invariants"
        )
    _expect(
        dict(
            _mapping(
                input_provenance.get("graph_checksums"),
                "input provenance graph_checksums",
            )
        ),
        graph.checksums.to_dict(),
        "input provenance graph checksums",
    )
    _expect(
        dict(
            _mapping(
                input_provenance.get("graph_config"),
                "input provenance graph_config",
            )
        ),
        dict(graph_config),
        "input provenance graph config",
    )
    return graph


def _mask_rates(masking: Mapping[str, Any]) -> Mapping[str, Any]:
    for field in ("rates", "rate"):
        value = masking.get(field)
        if isinstance(value, Mapping):
            return value
    raise InterpretabilityContractError(
        "masking.rates or masking.rate mapping is required"
    )


def _rebuild_verified_whole_node_mask(
    core: FullCoreData,
    config: Mapping[str, Any],
    mask_provenance: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[np.ndarray, FixedMaskBundle]:
    dataset = _mapping(config.get("dataset"), "config.dataset")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    masking = _mapping(config.get("masking"), "config.masking")
    rates = _mask_rates(masking)
    partial_rate = float(
        masking.get("partial_gene_rate", rates.get("partial_gene", 0.2))
    )
    node_rate = float(
        masking.get("whole_node_rate", rates.get("whole_node", 0.1))
    )
    block_rate = float(
        masking.get("block_node_rate", rates.get("spatial_block", 0.1))
    )
    width = masking.get("block_width_um")
    shape = str(masking.get("block_shape", "disk"))
    specs = [
        MaskSpec(
            mode="partial",
            partial_gene_rate=partial_rate,
            node_rate=node_rate,
            block_node_rate=block_rate,
            block_width_um=width,
            block_shape=shape,
            label="partial_gene",
        ),
        MaskSpec(
            mode="node",
            partial_gene_rate=partial_rate,
            node_rate=node_rate,
            block_node_rate=block_rate,
            block_width_um=width,
            block_shape=shape,
            label="whole_node",
        ),
        MaskSpec(
            mode="block",
            partial_gene_rate=partial_rate,
            node_rate=node_rate,
            block_node_rate=block_rate,
            block_width_um=width,
            block_shape=shape,
            label="spatial_block",
        ),
    ]
    base_seed = derive_mask_seed(
        int(masking.get("mask_seed", 0)),
        "held-in-full-core-fixed-evaluation",
        dataset.get("dataset_id"),
        dataset.get("version"),
        dataset.get("split_fingerprint"),
    )
    bundle = create_fixed_mask_bundle(
        {"fit": core.coordinates_um},
        core.n_genes,
        specs,
        replicates=int(evaluation["mask_replicates_per_mode"]),
        base_seed=base_seed,
    )
    recorded_manifest = _mapping(
        mask_provenance.get("bundle_manifest"),
        "mask provenance bundle_manifest",
    )
    _expect(
        dict(recorded_manifest),
        dict(bundle.manifest),
        "reconstructed fixed-mask manifest",
    )
    for location, value in (
        ("checkpoint", checkpoint.get("evaluation_mask_bundle_sha256")),
        ("summary", summary.get("evaluation_mask_bundle_sha256")),
    ):
        _expect(value, bundle.checksum, f"{location} mask bundle checksum")

    mask = np.asarray(bundle.get("fit", "whole_node", 0), dtype=np.bool_)
    selected = _whole_node_selected_rows(mask)
    if int(selected.sum()) < max(HASH_SAMPLE_SIZE, OUTCOME_SAMPLE_SIZE):
        raise InterpretabilityContractError(
            "Canonical whole-node mask selects too few receivers"
        )
    return mask, bundle


def _whole_node_selected_rows(mask: np.ndarray) -> np.ndarray:
    """Validate an all-genes-or-no-genes mask and return selected rows."""

    values = np.asarray(mask)
    if values.ndim != 2 or values.dtype != np.bool_:
        raise TypeError("whole-node mask must be a boolean matrix")
    selected = values.any(axis=1)
    # ``array_equal`` does not broadcast; the explicit elementwise comparison
    # is required for an [N, G] mask against its [N, 1] row indicator.
    if not bool(np.all(values == selected[:, None])):
        raise InterpretabilityContractError(
            "Canonical whole-node mask contains partial rows"
        )
    return selected


def _hash_key(namespace: str, *values: int) -> bytes:
    payload = json.dumps(
        [str(namespace), *(int(value) for value in values)],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).digest()


def _deterministic_hash_select(
    candidates: Sequence[int] | np.ndarray,
    count: int,
    *,
    namespace: str,
) -> np.ndarray:
    """Select a set using only candidate indices and a fixed hash namespace."""

    values = np.asarray(candidates)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise TypeError("candidates must be a one-dimensional integer array")
    values = values.astype(np.int64, copy=False)
    if len(np.unique(values)) != len(values):
        raise ValueError("candidates must be unique")
    count = int(count)
    if count <= 0 or count > len(values):
        raise ValueError("count must be in [1, len(candidates)]")
    ranked = sorted(
        (value.item() for value in values),
        key=lambda value: (_hash_key(namespace, value), value),
    )
    return np.sort(np.asarray(ranked[:count], dtype=np.int64))


def _resolve_panel_indices(
    gene_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    lookup: dict[str, int] = {}
    for index, name in enumerate(gene_names):
        if name in lookup:
            raise InterpretabilityContractError(
                f"Gene panel cannot resolve duplicate probe {name!r}"
            )
        lookup[str(name)] = index
    required = (*ORGANIZER_GENES, *RECEIVER_PROGRAM_GENES)
    missing = [name for name in required if name not in lookup]
    if missing:
        raise InterpretabilityContractError(
            "Required TLS toy-analysis genes are absent: " + ", ".join(missing)
        )
    return (
        np.asarray([lookup[name] for name in ORGANIZER_GENES], dtype=np.int64),
        np.asarray(
            [lookup[name] for name in RECEIVER_PROGRAM_GENES],
            dtype=np.int64,
        ),
    )


def _top_outcome_score_select(
    masked_receivers: Sequence[int] | np.ndarray,
    target_expression: np.ndarray,
    program_gene_indices: Sequence[int] | np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select high-score receivers; this is explicitly outcome-conditioned."""

    receivers = np.asarray(masked_receivers, dtype=np.int64)
    expression = np.asarray(target_expression)
    genes = np.asarray(program_gene_indices, dtype=np.int64)
    if (
        receivers.ndim != 1
        or genes.ndim != 1
        or expression.ndim != 2
        or not len(genes)
    ):
        raise ValueError("Outcome-score inputs have invalid shapes")
    count = int(count)
    if count <= 0 or count > len(receivers):
        raise ValueError("count must be in [1, len(masked_receivers)]")
    scores = expression[np.ix_(receivers, genes)].mean(axis=1, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Outcome scores must be finite")
    order = sorted(
        range(len(receivers)),
        key=lambda offset: (
            -float(scores[offset]),
            _hash_key("tls-outcome-score-tie-v1", int(receivers[offset])),
        ),
    )
    chosen_offsets = np.asarray(order[:count], dtype=np.int64)
    chosen = receivers[chosen_offsets]
    chosen_scores = scores[chosen_offsets]
    sorted_order = np.argsort(chosen, kind="stable")
    return chosen[sorted_order], chosen_scores[sorted_order]


def _describe(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Summary values must be a non-empty finite vector")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _receiver_attention_arrays(
    receiver: Sequence[int] | np.ndarray,
    attention: np.ndarray,
    raw_distance_um: Sequence[float] | np.ndarray,
    selected_receivers: Sequence[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    receiver_array = np.asarray(receiver, dtype=np.int64)
    weights = np.asarray(attention, dtype=np.float64)
    distance = np.asarray(raw_distance_um, dtype=np.float64)
    selected = np.asarray(selected_receivers, dtype=np.int64)
    if weights.ndim == 1:
        weights = weights[:, None]
    if (
        receiver_array.ndim != 1
        or weights.ndim != 2
        or distance.ndim != 1
        or len(receiver_array) != len(weights)
        or len(distance) != len(weights)
        or not np.isfinite(weights).all()
        or not np.isfinite(distance).all()
    ):
        raise ValueError("Attention arrays are invalid or misaligned")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("selected_receivers must be unique")

    degrees: list[float] = []
    entropy: list[float] = []
    effective: list[float] = []
    weighted_distance: list[float] = []
    for node in selected.tolist():
        incoming = receiver_array == node
        if not bool(incoming.any()):
            raise ValueError("A selected receiver has no incoming edges")
        probability = weights[incoming].mean(axis=1)
        total = float(probability.sum())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Mean-head attention is not normalizable")
        probability = probability / total
        node_entropy = float(
            -(probability * np.log(np.clip(probability, 1e-300, None))).sum()
        )
        degrees.append(float(incoming.sum()))
        entropy.append(node_entropy)
        effective.append(math.exp(node_entropy))
        weighted_distance.append(
            float(np.dot(probability, distance[incoming]))
        )
    return {
        "incoming_degree": np.asarray(degrees, dtype=np.float64),
        "attention_entropy_nats": np.asarray(entropy, dtype=np.float64),
        "effective_neighbor_count": np.asarray(effective, dtype=np.float64),
        "attention_weighted_distance_um": np.asarray(
            weighted_distance, dtype=np.float64
        ),
    }


def _aggregate_attention(
    receiver: Sequence[int] | np.ndarray,
    attention: np.ndarray,
    raw_distance_um: Sequence[float] | np.ndarray,
    selected_receivers: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    arrays = _receiver_attention_arrays(
        receiver,
        attention,
        raw_distance_um,
        selected_receivers,
    )
    return {
        "receiver_count": int(len(np.asarray(selected_receivers))),
        **{name: _describe(values) for name, values in arrays.items()},
    }


def _rank_distance_bins(
    distances: np.ndarray,
    edge_ids: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    order = np.lexsort((edge_ids, distances))
    bins = np.empty(len(distances), dtype=np.int64)
    bins[order] = (
        np.arange(len(distances), dtype=np.int64) * int(bin_count)
    ) // len(distances)
    return bins


def _build_distance_matched_deletion_plan(
    edge_ids: Sequence[int] | np.ndarray,
    receiver: Sequence[int] | np.ndarray,
    mean_head_attention: Sequence[float] | np.ndarray,
    raw_distance_um: Sequence[float] | np.ndarray,
    selected_receivers: Sequence[int] | np.ndarray,
    *,
    fraction: float = DELETION_FRACTION,
    namespace: str = "tls-distance-matched-deletion-v1",
) -> dict[str, np.ndarray]:
    """Choose top-attention and exact rank-distance-bin-matched control edges."""

    ids = np.asarray(edge_ids, dtype=np.int64)
    receivers = np.asarray(receiver, dtype=np.int64)
    attention = np.asarray(mean_head_attention, dtype=np.float64)
    distance = np.asarray(raw_distance_um, dtype=np.float64)
    selected = np.asarray(selected_receivers, dtype=np.int64)
    if (
        ids.ndim != 1
        or receivers.shape != ids.shape
        or attention.shape != ids.shape
        or distance.shape != ids.shape
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(attention).all()
        or not np.isfinite(distance).all()
    ):
        raise ValueError("Deletion-plan edge arrays are invalid")
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("fraction must be in (0, 0.5)")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("selected_receivers must be unique")

    top_ids: list[int] = []
    random_ids: list[int] = []
    deleted_counts: list[int] = []
    degree_values: list[int] = []
    bin_counts: list[int] = []
    for node in selected.tolist():
        positions = np.flatnonzero(receivers == node)
        degree = len(positions)
        if degree == 0:
            raise ValueError("A deletion receiver has no incoming edges")
        delete_count = max(1, int(math.ceil(degree * float(fraction))))
        if delete_count * 2 > degree:
            raise ValueError("Too few edges for a non-overlapping control")
        local_ids = ids[positions]
        local_attention = attention[positions]
        local_distance = distance[positions]
        attention_order = np.lexsort((local_ids, -local_attention))
        top_local = attention_order[:delete_count]
        top_mask = np.zeros(degree, dtype=np.bool_)
        top_mask[top_local] = True

        chosen_bins: np.ndarray | None = None
        chosen_bin_count = 0
        for proposed_bins in range(min(10, degree), 0, -1):
            bins = _rank_distance_bins(
                local_distance, local_ids, proposed_bins
            )
            feasible = True
            for bin_number in range(proposed_bins):
                top_in_bin = int(np.sum(top_mask & (bins == bin_number)))
                available = int(np.sum((~top_mask) & (bins == bin_number)))
                if top_in_bin > available:
                    feasible = False
                    break
            if feasible:
                chosen_bins = bins
                chosen_bin_count = proposed_bins
                break
        if chosen_bins is None:
            raise RuntimeError("Could not construct a matched deletion control")

        matched_local: list[int] = []
        for bin_number in range(chosen_bin_count):
            needed = int(np.sum(top_mask & (chosen_bins == bin_number)))
            if not needed:
                continue
            candidates = np.flatnonzero(
                (~top_mask) & (chosen_bins == bin_number)
            )
            ranked_candidates = sorted(
                candidates.tolist(),
                key=lambda offset: (
                    _hash_key(
                        namespace,
                        node,
                        int(local_ids[offset]),
                    ),
                    int(local_ids[offset]),
                ),
            )
            matched_local.extend(ranked_candidates[:needed])
        if len(matched_local) != delete_count:
            raise RuntimeError("Matched deletion count is not exact")

        top_for_node = local_ids[top_local]
        random_for_node = local_ids[np.asarray(matched_local, dtype=np.int64)]
        if np.intersect1d(top_for_node, random_for_node).size:
            raise RuntimeError("Top and matched deletion edges overlap")
        top_bin_hist = np.bincount(
            chosen_bins[top_local], minlength=chosen_bin_count
        )
        random_bin_hist = np.bincount(
            chosen_bins[np.asarray(matched_local, dtype=np.int64)],
            minlength=chosen_bin_count,
        )
        if not np.array_equal(top_bin_hist, random_bin_hist):
            raise RuntimeError("Distance-bin histograms are not exactly matched")

        top_ids.extend(top_for_node.tolist())
        random_ids.extend(random_for_node.tolist())
        deleted_counts.append(delete_count)
        degree_values.append(degree)
        bin_counts.append(chosen_bin_count)

    return {
        "top_edge_ids": np.sort(np.asarray(top_ids, dtype=np.int64)),
        "matched_edge_ids": np.sort(
            np.asarray(random_ids, dtype=np.int64)
        ),
        "deleted_count_per_receiver": np.asarray(
            deleted_counts, dtype=np.int64
        ),
        "degree_per_receiver": np.asarray(degree_values, dtype=np.int64),
        "distance_bin_count_per_receiver": np.asarray(
            bin_counts, dtype=np.int64
        ),
    }


def _aligned_edge_positions(
    ordered_edge_ids: np.ndarray,
    selected_edge_ids: np.ndarray,
) -> np.ndarray:
    ordered = np.asarray(ordered_edge_ids, dtype=np.int64)
    selected = np.asarray(selected_edge_ids, dtype=np.int64)
    if (
        ordered.ndim != 1
        or selected.ndim != 1
        or len(np.unique(ordered)) != len(ordered)
        or len(np.unique(selected)) != len(selected)
        or (len(ordered) > 1 and np.any(ordered[1:] <= ordered[:-1]))
    ):
        raise ValueError("Edge ID vectors must be sorted and unique")
    positions = np.searchsorted(ordered, selected)
    if (
        len(selected)
        and (
            bool(np.any(positions >= len(ordered)))
            or not np.array_equal(ordered[positions], selected)
        )
    ):
        raise ValueError("Selected edge IDs are not in the aligned edge set")
    return positions


def _source_level_statistics(
    source: np.ndarray,
    distance_um: np.ndarray,
    routing_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources = np.asarray(source, dtype=np.int64)
    distance = np.asarray(distance_um, dtype=np.float64)
    routing = np.asarray(routing_weight, dtype=np.float64)
    if (
        sources.ndim != 1
        or distance.shape != sources.shape
        or routing.shape != sources.shape
        or not np.isfinite(distance).all()
        or not np.isfinite(routing).all()
    ):
        raise ValueError("Source-level edge arrays are invalid")
    unique, inverse = np.unique(sources, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    mean_distance = np.bincount(
        inverse, weights=distance, minlength=len(unique)
    ) / counts
    maximum_routing = np.full(len(unique), -np.inf, dtype=np.float64)
    np.maximum.at(maximum_routing, inverse, routing)
    return unique, mean_distance, maximum_routing


def _select_bounded_top_sender_sources(
    aligned_edge_ids: np.ndarray,
    edge_source: np.ndarray,
    mean_head_attention: np.ndarray,
    raw_distance_um: np.ndarray,
    top_edge_ids: np.ndarray,
    eligible_source_nodes: np.ndarray,
    *,
    count: int = SENDER_PERTURBATION_SOURCES,
) -> dict[str, np.ndarray]:
    """Select a bounded top-routed sender set without organizer-score ranking."""

    positions = _aligned_edge_positions(aligned_edge_ids, top_edge_ids)
    source = np.asarray(edge_source, dtype=np.int64)[positions]
    attention = np.asarray(mean_head_attention, dtype=np.float64)[positions]
    distance = np.asarray(raw_distance_um, dtype=np.float64)[positions]
    eligible = np.asarray(eligible_source_nodes, dtype=np.bool_)
    if eligible.ndim != 1 or (
        len(source) and int(source.max()) >= len(eligible)
    ):
        raise ValueError("eligible_source_nodes is not aligned")
    keep = eligible[source]
    source = source[keep]
    attention = attention[keep]
    distance = distance[keep]
    unique, mean_distance, maximum_attention = _source_level_statistics(
        source,
        distance,
        attention,
    )
    count = int(count)
    if count <= 0 or len(unique) < count:
        raise InterpretabilityContractError(
            "Too few unmasked top-routed senders for the bounded "
            f"organizer perturbation: need {count}, found {len(unique)}"
        )
    order = sorted(
        range(len(unique)),
        key=lambda offset: (
            -float(maximum_attention[offset]),
            _hash_key(
                "tls-top-sender-routing-rank-v1", int(unique[offset])
            ),
        ),
    )
    selected_offsets = np.asarray(order[:count], dtype=np.int64)
    return {
        "source_nodes": unique[selected_offsets],
        "representative_distance_um": mean_distance[selected_offsets],
        "maximum_routing_weight": maximum_attention[selected_offsets],
    }


def _select_distance_matched_null_senders(
    aligned_edge_ids: np.ndarray,
    edge_source: np.ndarray,
    mean_head_attention: np.ndarray,
    raw_distance_um: np.ndarray,
    matched_edge_ids: np.ndarray,
    eligible_source_nodes: np.ndarray,
    top_sender_plan: Mapping[str, np.ndarray],
    *,
    namespace: str,
) -> dict[str, np.ndarray]:
    """Select equal-count null senders from distance-matched null edges."""

    positions = _aligned_edge_positions(aligned_edge_ids, matched_edge_ids)
    source = np.asarray(edge_source, dtype=np.int64)[positions]
    attention = np.asarray(mean_head_attention, dtype=np.float64)[positions]
    distance = np.asarray(raw_distance_um, dtype=np.float64)[positions]
    eligible = np.asarray(eligible_source_nodes, dtype=np.bool_)
    top_sources = np.asarray(
        top_sender_plan["source_nodes"], dtype=np.int64
    )
    target_distance = np.asarray(
        top_sender_plan["representative_distance_um"], dtype=np.float64
    )
    if eligible.ndim != 1 or (
        len(source) and int(source.max()) >= len(eligible)
    ):
        raise ValueError("eligible_source_nodes is not aligned")
    keep = eligible[source] & ~np.isin(source, top_sources)
    source = source[keep]
    attention = attention[keep]
    distance = distance[keep]
    unique, candidate_distance, maximum_attention = (
        _source_level_statistics(source, distance, attention)
    )
    if len(unique) < len(top_sources):
        raise InterpretabilityContractError(
            "Too few distinct eligible null senders for organizer perturbation"
        )

    selected_offsets: list[int] = []
    selected_bin_count = 0
    selected_target_bins: np.ndarray | None = None
    selected_candidate_bins: np.ndarray | None = None
    for bin_count in range(min(10, len(top_sources)), 0, -1):
        if bin_count == 1:
            boundaries = np.empty(0, dtype=np.float64)
        else:
            boundaries = np.quantile(
                target_distance,
                np.arange(1, bin_count, dtype=np.float64) / bin_count,
            )
        target_bins = np.searchsorted(
            boundaries, target_distance, side="right"
        )
        candidate_bins = np.searchsorted(
            boundaries, candidate_distance, side="right"
        )
        feasible = all(
            int(np.sum(candidate_bins == bin_number))
            >= int(np.sum(target_bins == bin_number))
            for bin_number in range(bin_count)
        )
        if not feasible:
            continue

        proposed: list[int] = []
        for bin_number in range(bin_count):
            target_in_bin = target_distance[target_bins == bin_number]
            needed = len(target_in_bin)
            if not needed:
                continue
            candidates = np.flatnonzero(candidate_bins == bin_number)
            centre = float(np.median(target_in_bin))
            ranked = sorted(
                candidates.tolist(),
                key=lambda offset: (
                    _hash_key(namespace, int(unique[offset])),
                    abs(float(candidate_distance[offset]) - centre),
                ),
            )
            proposed.extend(ranked[:needed])
        if len(proposed) == len(top_sources):
            selected_offsets = proposed
            selected_bin_count = bin_count
            selected_target_bins = target_bins
            selected_candidate_bins = candidate_bins
            break
    if (
        not selected_offsets
        or selected_target_bins is None
        or selected_candidate_bins is None
    ):
        raise RuntimeError("Could not distance-match null sender sources")

    selected = np.asarray(selected_offsets, dtype=np.int64)
    selected_distance = candidate_distance[selected]
    absolute_mismatch: list[float] = []
    for bin_number in range(selected_bin_count):
        left = np.sort(
            target_distance[selected_target_bins == bin_number]
        )
        in_bin = (
            selected_candidate_bins[selected] == bin_number
        )
        right = np.sort(selected_distance[in_bin])
        if len(left) != len(right):
            raise RuntimeError("Sender distance-bin counts are not exact")
        absolute_mismatch.extend(np.abs(left - right).tolist())
    null_sources = unique[selected]
    if (
        len(np.unique(null_sources)) != len(top_sources)
        or np.intersect1d(null_sources, top_sources).size
    ):
        raise RuntimeError("Sender perturbation sets are not disjoint/equal")
    return {
        "source_nodes": null_sources,
        "representative_distance_um": selected_distance,
        "maximum_routing_weight": maximum_attention[selected],
        "distance_bin_count": np.asarray(
            [selected_bin_count], dtype=np.int64
        ),
        "absolute_distance_mismatch_um": np.asarray(
            absolute_mismatch, dtype=np.float64
        ),
    }


def _mean_ablate_sender_program(
    masked_expression: Tensor,
    source_nodes: np.ndarray,
    gene_indices: np.ndarray,
) -> Tensor:
    """Set three visible organizer channels to their fitted mean (zero)."""

    sources = torch.as_tensor(
        np.asarray(source_nodes, dtype=np.int64),
        dtype=torch.long,
        device=masked_expression.device,
    )
    genes = torch.as_tensor(
        np.asarray(gene_indices, dtype=np.int64),
        dtype=torch.long,
        device=masked_expression.device,
    )
    if (
        sources.ndim != 1
        or genes.ndim != 1
        or not sources.numel()
        or not genes.numel()
        or bool((sources < 0).any())
        or bool((sources >= masked_expression.shape[0]).any())
        or bool((genes < 0).any())
        or bool((genes >= masked_expression.shape[1]).any())
    ):
        raise ValueError("Sender-program perturbation indices are invalid")
    perturbed = masked_expression.clone()
    perturbed[sources[:, None], genes[None, :]] = 0.0
    return perturbed


def _repeated_program_effect_metrics(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    targeted_prediction: np.ndarray,
    matched_null_predictions: Sequence[np.ndarray],
    *,
    huber_delta: float,
) -> dict[str, Any]:
    """Aggregate one targeted perturbation against repeated matched nulls."""

    target_array = np.asarray(target, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    targeted = np.asarray(targeted_prediction, dtype=np.float64)
    null = np.asarray(matched_null_predictions, dtype=np.float64)
    if (
        target_array.ndim != 2
        or target_array.shape != baseline.shape
        or targeted.shape != baseline.shape
        or null.ndim != 3
        or null.shape[1:] != baseline.shape
        or not len(null)
        or not all(
            np.isfinite(value).all()
            for value in (target_array, baseline, targeted, null)
        )
    ):
        raise ValueError("Repeated program predictions are invalid")
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")

    def huber(prediction: np.ndarray) -> np.ndarray:
        residual = np.abs(prediction - target_array)
        elementwise = np.where(
            residual <= delta,
            0.5 * residual**2,
            delta * (residual - 0.5 * delta),
        )
        return elementwise.mean(axis=-1)

    baseline_loss = huber(baseline)
    targeted_loss_change = huber(targeted) - baseline_loss
    null_loss_change = huber(null) - baseline_loss[None, :]
    targeted_prediction_change = np.abs(targeted - baseline).mean(axis=1)
    null_prediction_change = np.abs(
        null - baseline[None, :, :]
    ).mean(axis=2)
    loss_contrast = targeted_loss_change[None, :] - null_loss_change
    prediction_contrast = (
        targeted_prediction_change[None, :] - null_prediction_change
    )
    loss_draw_means = loss_contrast.mean(axis=1)
    prediction_draw_means = prediction_contrast.mean(axis=1)
    baseline_mean = float(baseline_loss.mean())
    relative_loss_draw_means = loss_draw_means / max(
        baseline_mean, np.finfo(np.float64).eps
    )
    return {
        "receiver_count": int(target_array.shape[0]),
        "program_gene_count": int(target_array.shape[1]),
        "matched_null_replicates": int(len(null)),
        "baseline_program_huber": _describe(baseline_loss),
        "targeted_condition": {
            "program_huber_change_from_baseline": _describe(
                targeted_loss_change
            ),
            "program_prediction_mae_from_baseline": _describe(
                targeted_prediction_change
            ),
        },
        "matched_null_conditions": {
            "program_huber_change_from_baseline_pooled": _describe(
                null_loss_change.ravel()
            ),
            "program_huber_change_draw_means": _describe(
                null_loss_change.mean(axis=1)
            ),
            "program_prediction_mae_from_baseline_pooled": _describe(
                null_prediction_change.ravel()
            ),
            "program_prediction_mae_draw_means": _describe(
                null_prediction_change.mean(axis=1)
            ),
        },
        "paired_targeted_minus_null": {
            "program_huber_change_pooled": _describe(
                loss_contrast.ravel()
            ),
            "program_huber_change_draw_means": _describe(loss_draw_means),
            "relative_huber_contrast_draw_means": _describe(
                relative_loss_draw_means
            ),
            "minimum_relative_huber_contrast_across_draws": float(
                relative_loss_draw_means.min()
            ),
            "program_prediction_mae_pooled": _describe(
                prediction_contrast.ravel()
            ),
            "program_prediction_mae_draw_means": _describe(
                prediction_draw_means
            ),
            "minimum_prediction_mae_contrast_across_draws": float(
                prediction_draw_means.min()
            ),
            "fraction_receiver_draw_huber_contrasts_positive": float(
                np.mean(loss_contrast > 0.0)
            ),
            "fraction_receiver_draw_prediction_contrasts_positive": float(
                np.mean(prediction_contrast > 0.0)
            ),
        },
    }


def _minimum_effect_support(metrics: Mapping[str, Any]) -> bool:
    paired = _mapping(
        metrics.get("paired_targeted_minus_null"),
        "paired targeted-minus-null metrics",
    )
    targeted = _mapping(
        metrics.get("targeted_condition"), "targeted condition metrics"
    )
    targeted_huber = _mapping(
        targeted.get("program_huber_change_from_baseline"),
        "targeted Huber change",
    )
    return bool(
        float(targeted_huber["mean"]) > 0.0
        and float(
            paired["minimum_relative_huber_contrast_across_draws"]
        )
        >= MIN_RELATIVE_HUBER_CONTRAST
        and float(
            paired["minimum_prediction_mae_contrast_across_draws"]
        )
        >= MIN_PREDICTION_MAE_CONTRAST
    )


def _program_deletion_metrics(
    target: np.ndarray,
    baseline_prediction: np.ndarray,
    top_deleted_prediction: np.ndarray,
    matched_deleted_prediction: np.ndarray,
    *,
    huber_delta: float,
) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=np.float64)
    baseline = np.asarray(baseline_prediction, dtype=np.float64)
    top = np.asarray(top_deleted_prediction, dtype=np.float64)
    matched = np.asarray(matched_deleted_prediction, dtype=np.float64)
    if (
        target_array.ndim != 2
        or target_array.shape != baseline.shape
        or top.shape != baseline.shape
        or matched.shape != baseline.shape
        or not all(
            np.isfinite(value).all()
            for value in (target_array, baseline, top, matched)
        )
    ):
        raise ValueError("Program prediction arrays are invalid or misaligned")
    delta = float(huber_delta)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")

    def huber(prediction: np.ndarray) -> np.ndarray:
        residual = np.abs(prediction - target_array)
        elementwise = np.where(
            residual <= delta,
            0.5 * residual**2,
            delta * (residual - 0.5 * delta),
        )
        return elementwise.mean(axis=1)

    baseline_loss = huber(baseline)
    top_loss_change = huber(top) - baseline_loss
    matched_loss_change = huber(matched) - baseline_loss
    top_prediction_change = np.abs(top - baseline).mean(axis=1)
    matched_prediction_change = np.abs(matched - baseline).mean(axis=1)
    loss_contrast = top_loss_change - matched_loss_change
    prediction_contrast = top_prediction_change - matched_prediction_change
    return {
        "receiver_count": int(target_array.shape[0]),
        "program_gene_count": int(target_array.shape[1]),
        "baseline_program_huber": _describe(baseline_loss),
        "top_attention_deletion": {
            "program_huber_change_from_baseline": _describe(top_loss_change),
            "program_prediction_mae_from_baseline": _describe(
                top_prediction_change
            ),
        },
        "distance_matched_random_deletion": {
            "program_huber_change_from_baseline": _describe(
                matched_loss_change
            ),
            "program_prediction_mae_from_baseline": _describe(
                matched_prediction_change
            ),
        },
        "paired_top_minus_matched": {
            "program_huber_change": _describe(loss_contrast),
            "program_prediction_mae": _describe(prediction_contrast),
            "fraction_huber_contrast_positive": float(
                np.mean(loss_contrast > 0.0)
            ),
            "fraction_prediction_contrast_positive": float(
                np.mean(prediction_contrast > 0.0)
            ),
        },
    }


def _model_from_config(
    core: FullCoreData,
    graph: ReceiverSortedGraph,
    model_config: Mapping[str, Any],
    state_dict: Mapping[str, Tensor],
) -> ReceiverChunkedEdgeConditionedGATv2:
    arguments: dict[str, Any] = {
        "num_genes": core.n_genes,
        "edge_attribute_dim": len(graph.edge_attribute_names),
        "node_covariate_dim": int(core.node_covariates.shape[1]),
        "hidden_dim": int(model_config["hidden_dim"]),
        "attention_heads": int(model_config["attention_heads"]),
        "graph_layers": int(model_config["graph_layers"]),
        "ffn_dim": int(model_config["ffn_dim"]),
        "decoder_dim": int(model_config["decoder_dim"]),
        "edge_hidden_dim": int(model_config["edge_hidden_dim"]),
        "edge_embedding_dim": int(model_config["edge_embedding_dim"]),
        "dropout": float(model_config["dropout"]),
        "attention_dropout": float(model_config["attention_dropout"]),
        "receiver_chunk_size": int(model_config["receiver_chunk_size"]),
        "activation_checkpointing": bool(
            model_config["activation_checkpointing"]
        ),
    }
    if model_config.get("attention_head_dim") is not None:
        arguments["attention_head_dim"] = int(
            model_config["attention_head_dim"]
        )
    model = ReceiverChunkedEdgeConditionedGATv2(**arguments)
    model.load_state_dict(state_dict, strict=True)
    if _state_dict_sha256(model.state_dict()) != _state_dict_sha256(
        state_dict
    ):
        raise InterpretabilityContractError(
            "Reconstructed DenseG2 state differs from the checkpoint"
        )
    return model


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise InterpretabilityContractError(
            f"Invalid inference device: {value!r}"
        ) from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InterpretabilityContractError(
            "CUDA was requested but is not available"
        )
    return device


def _scenario_forward(
    model: ReceiverChunkedEdgeConditionedGATv2,
    *,
    masked_expression: Tensor,
    gene_mask: Tensor,
    node_covariates: Tensor,
    edge_index: np.ndarray,
    edge_attributes: np.ndarray,
    target_nodes: np.ndarray,
    device: torch.device,
    deleted_edge_ids: np.ndarray | None = None,
    attention_receivers: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Execute one read-only graph scenario and release its edge tensors."""

    if deleted_edge_ids is None:
        scenario_edges = edge_index
        scenario_attributes = edge_attributes
    else:
        deleted = np.asarray(deleted_edge_ids, dtype=np.int64)
        if (
            deleted.ndim != 1
            or len(np.unique(deleted)) != len(deleted)
            or bool(np.any(deleted < 0))
            or bool(np.any(deleted >= edge_index.shape[1]))
        ):
            raise ValueError("deleted_edge_ids are invalid")
        keep = np.ones(edge_index.shape[1], dtype=np.bool_)
        keep[deleted] = False
        scenario_edges = edge_index[:, keep]
        scenario_attributes = edge_attributes[keep]
        if len(scenario_edges[1]) > 1 and np.any(
            scenario_edges[1, 1:] < scenario_edges[1, :-1]
        ):
            raise RuntimeError("Deletion broke receiver-sorted graph order")

    edge_tensor = torch.from_numpy(
        np.asarray(scenario_edges, dtype=np.int64)
    ).to(device=device)
    attribute_tensor = torch.from_numpy(
        np.asarray(scenario_attributes, dtype=np.float32)
    ).to(device=device)
    targets = torch.from_numpy(
        np.asarray(target_nodes, dtype=np.int64)
    ).to(device=device)
    explanation_targets = (
        None
        if attention_receivers is None
        else torch.from_numpy(
            np.asarray(attention_receivers, dtype=np.int64)
        ).to(device=device)
    )
    model.clear_edge_layout_cache()
    try:
        with torch.inference_mode():
            output = model(
                masked_expression,
                gene_mask,
                edge_index=edge_tensor,
                edge_attributes=attribute_tensor,
                node_covariates=node_covariates,
                target_nodes=targets,
                return_explanations=attention_receivers is not None,
                attention_receivers=explanation_targets,
            )
        result = {
            "prediction": output.prediction.detach().float().cpu().numpy()
        }
        if attention_receivers is not None:
            if output.attention_weights is None or output.edge_index is None:
                raise RuntimeError("DenseG2 did not return requested attention")
            result["attention"] = (
                output.attention_weights.detach().float().cpu().numpy()
            )
            result["explanation_edge_index"] = (
                output.edge_index.detach().cpu().numpy()
            )
        return result
    finally:
        model.clear_edge_layout_cache()
        del edge_tensor, attribute_tensor, targets, explanation_targets
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _deleted_sender_score_summary(
    edge_index: np.ndarray,
    edge_ids: np.ndarray,
    target_expression: np.ndarray,
    organizer_gene_indices: np.ndarray,
) -> dict[str, float | int]:
    source = edge_index[0, np.asarray(edge_ids, dtype=np.int64)]
    scores = target_expression[
        np.ix_(source, organizer_gene_indices)
    ].mean(axis=1, dtype=np.float64)
    return _describe(scores)


def _sender_organizer_enrichment(
    top_sender_plan: Mapping[str, np.ndarray],
    null_sender_plans: Sequence[Mapping[str, np.ndarray]],
    target_expression: np.ndarray,
    organizer_gene_indices: np.ndarray,
) -> dict[str, Any]:
    top_sources = np.asarray(
        top_sender_plan["source_nodes"], dtype=np.int64
    )
    top_scores = target_expression[
        np.ix_(top_sources, organizer_gene_indices)
    ].mean(axis=1, dtype=np.float64)
    null_scores = [
        target_expression[
            np.ix_(
                np.asarray(plan["source_nodes"], dtype=np.int64),
                organizer_gene_indices,
            )
        ].mean(axis=1, dtype=np.float64)
        for plan in null_sender_plans
    ]
    null_array = np.stack(null_scores, axis=0)
    enrichment_draw_means = top_scores.mean() - null_array.mean(axis=1)
    return {
        "score_definition": (
            "mean full-core-standardized observed expression across "
            "CCL19, CCL21, and CXCL13"
        ),
        "top_routed_sender_score": _describe(top_scores),
        "distance_matched_null_sender_score_pooled": _describe(
            null_array.ravel()
        ),
        "top_minus_null_score_draw_means": _describe(
            enrichment_draw_means
        ),
        "minimum_top_minus_null_score_across_draws": float(
            enrichment_draw_means.min()
        ),
    }


def _assert_aggregate_only(payload: Mapping[str, Any]) -> None:
    """Defensively reject output keys that imply row-level disclosure."""

    forbidden_key_tokens = (
        "cell_id",
        "barcode",
        "patient",
        "donor",
        "local_index",
        "receiver_indices",
        "receiver_ids",
        "source_indices",
        "source_ids",
        "edge_ids",
        "sample_key",
    )

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                lowered = str(key).lower()
                if any(token in lowered for token in forbidden_key_tokens):
                    raise InterpretabilityContractError(
                        "Aggregate output contains a forbidden row-level key: "
                        + ".".join((*path, str(key)))
                    )
                visit(child, (*path, str(key)))
        elif isinstance(value, (list, tuple, np.ndarray)):
            sequence = list(value)
            if any(
                isinstance(child, (Mapping, list, tuple, np.ndarray))
                or isinstance(child, (int, float, np.integer, np.floating))
                or not isinstance(child, str)
                for child in sequence
            ):
                raise InterpretabilityContractError(
                    "Aggregate output contains a row-like or numeric sequence: "
                    + ".".join(path)
                )
        elif isinstance(value, (float, np.floating)) and not math.isfinite(
            float(value)
        ):
            raise InterpretabilityContractError(
                "Aggregate output contains a non-finite value"
            )

    visit(payload, ())


def _render_markdown(report: Mapping[str, Any]) -> str:
    hashed = _mapping(
        report["attention_routing"]["expression_independent_hash_sample"],
        "attention hash sample",
    )
    deletion = _mapping(report["deletion_analysis"], "deletion analysis")
    program = _mapping(deletion["receiver_program_effect"], "program effect")
    contrast = _mapping(
        program["paired_targeted_minus_null"], "paired deletion contrast"
    )
    loss = _mapping(
        contrast["program_huber_change_draw_means"],
        "paired Huber contrast",
    )
    prediction = _mapping(
        contrast["program_prediction_mae_draw_means"],
        "paired prediction contrast",
    )
    sender = _mapping(
        report["sender_program_perturbation"],
        "sender-program perturbation",
    )
    sender_program = _mapping(
        sender["receiver_program_effect"], "sender-program effect"
    )
    sender_contrast = _mapping(
        sender_program["paired_targeted_minus_null"],
        "sender-program paired contrast",
    )
    sender_loss = _mapping(
        sender_contrast["program_huber_change_draw_means"],
        "sender-program Huber contrast",
    )
    enrichment = _mapping(
        sender["organizer_score_enrichment"],
        "organizer enrichment",
    )
    enrichment_draws = _mapping(
        enrichment["top_minus_null_score_draw_means"],
        "organizer enrichment draws",
    )
    entropy = _mapping(
        hashed["attention_entropy_nats"], "attention entropy"
    )
    effective = _mapping(
        hashed["effective_neighbor_count"], "effective neighbors"
    )
    distance = _mapping(
        hashed["attention_weighted_distance_um"], "weighted distance"
    )
    evidence = _mapping(report["interpretation"], "interpretation")
    return "\n".join(
        [
            "# Full-core G2 toy interpretability analysis",
            "",
            f"- Run: `{report['run_id']}`",
            f"- Protocol: `{report['protocol']}`",
            "- Scope: one held-in, transductively fitted spatial core; this is "
            "not validation, test, patient-level, or causal evidence.",
            "",
            "## Attention routing",
            "",
            f"The expression-independent hash sample contained "
            f"{hashed['receiver_count']} masked receivers. Mean receiver-level "
            f"attention entropy was {entropy['mean']:.6g} nats, mean effective "
            f"neighbor count was {effective['mean']:.6g}, and mean "
            f"attention-weighted distance was {distance['mean']:.6g} µm.",
            "",
            "## Outcome-conditioned deletion analysis",
            "",
            f"The high-lymphoid-score set contained "
            f"{deletion['receiver_count']} masked receivers. Deleting the "
            f"top 10% of incoming edges by last-layer mean-head attention was "
            f"compared with {deletion['matched_null_replicates']} equal-count, "
            f"deterministic, exactly rank-distance-bin-matched deletions. "
            "This removes edges from both GAT layers and renormalizes "
            "attention; it is not an isolated last-layer intervention.",
            "",
            f"- Mean across-draw top-minus-null receiver-program "
            f"Huber-change: "
            f"{loss['mean']:.6g}",
            f"- Mean across-draw top-minus-null receiver-program "
            f"prediction-MAE: "
            f"{prediction['mean']:.6g}",
            f"- Minimum-effect deletion support: "
            f"`{str(evidence['descriptive_deletion_support']).lower()}`",
            "",
            "## Organizer sender-program perturbation",
            "",
            f"The three organizer channels were mean-ablated in "
            f"{sender['source_count_each_condition']} top-routed unmasked "
            f"senders and compared with "
            f"{sender['matched_null_replicates']} equal-count, disjoint, "
            "distance-matched sender sets.",
            "",
            f"- Mean organizer-score top-minus-null enrichment: "
            f"{enrichment_draws['mean']:.6g}",
            f"- Mean sender-ablation top-minus-null receiver-program "
            f"Huber-change: {sender_loss['mean']:.6g}",
            f"- Minimum-effect sender-program support: "
            f"`{str(evidence['descriptive_sender_program_support']).lower()}`",
            f"- Organizer-enrichment support: "
            f"`{str(evidence['descriptive_organizer_enrichment_support']).lower()}`",
            f"- Joint TLS dependency support: "
            f"`{str(evidence['joint_tls_dependency_support']).lower()}`",
            f"- Evidence label: **{evidence['evidence_label']}**",
            "",
            "## Interpretation boundary",
            "",
            f"The strongest possible label is: **{MAXIMUM_INTERPRETATION}**. "
            "Attention is not importance. Even a positive deletion contrast "
            "shows model routing sensitivity, not biological importance or "
            "causality. Receiver selection used observed outcomes and is "
            "explicitly exploratory.",
            "",
        ]
    )


def _write_exclusive_output(
    destination: Path,
    report: Mapping[str, Any],
    markdown: str,
) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite interpretability output: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(mode=0o750)
    try:
        with (destination / "analysis.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        with (destination / "report.md").open("x", encoding="utf-8") as handle:
            handle.write(markdown)
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _run_analysis(
    root: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    _reject_bundle_symlinks(root)
    verification = verify_run_bundle(root, require_success_contract=True)
    _expect(verification.get("status"), "success", "run bundle status")

    config = load_yaml_mapping(root / "config.resolved.yaml")
    validate_experiment_config(config)
    manifest = _read_yaml_mapping(root / "manifest.yaml")
    summary = _read_json_mapping(root / "summary.json")
    input_provenance = _read_json_mapping(
        root / "provenance" / "full_core_inputs.json"
    )
    mask_provenance = _read_json_mapping(
        root / "provenance" / "fixed_evaluation_masks.json"
    )
    training_provenance = _read_json_mapping(
        root / "provenance" / "full_core_training.json"
    )
    _validate_locked_protocol(config, manifest, summary)
    state_dict, checkpoint = _load_verified_checkpoint(
        root,
        config,
        summary,
        training_provenance,
    )

    core = _rebuild_verified_core(config, input_provenance)
    graph = _rebuild_verified_graph(core, config, input_provenance)
    whole_node_mask, mask_bundle = _rebuild_verified_whole_node_mask(
        core,
        config,
        mask_provenance,
        checkpoint,
        summary,
    )
    organizer_indices, receiver_program_indices = _resolve_panel_indices(
        core.gene_names
    )
    masked_receivers = np.flatnonzero(whole_node_mask.any(axis=1)).astype(
        np.int64
    )
    hash_receivers = _deterministic_hash_select(
        masked_receivers,
        HASH_SAMPLE_SIZE,
        namespace="full-core-g2-attention-receivers-v1",
    )
    outcome_receivers, outcome_scores = _top_outcome_score_select(
        masked_receivers,
        core.target_expression,
        receiver_program_indices,
        OUTCOME_SAMPLE_SIZE,
    )
    all_receivers = np.union1d(hash_receivers, outcome_receivers).astype(
        np.int64
    )

    edge_index, edge_attributes = graph.concatenate()
    if (
        edge_index.shape != (2, EXPECTED_DIRECTED_EDGES)
        or edge_attributes.shape[0] != EXPECTED_DIRECTED_EDGES
    ):
        raise InterpretabilityContractError(
            "Concatenated graph does not match its verified edge count"
        )
    model_config = _mapping(config.get("model"), "config.model")
    model = _model_from_config(core, graph, model_config, state_dict)
    model.to(device=device)
    model.eval()

    expression_device = torch.from_numpy(
        np.asarray(core.target_expression, dtype=np.float32)
    ).to(device=device)
    mask_device = torch.from_numpy(
        np.array(whole_node_mask, dtype=np.bool_, copy=True)
    ).to(device=device)
    covariates_device = torch.from_numpy(
        np.asarray(core.node_covariates, dtype=np.float32)
    ).to(device=device)
    masked_expression = expression_device.masked_fill(mask_device, 0.0)

    baseline = _scenario_forward(
        model,
        masked_expression=masked_expression,
        gene_mask=mask_device,
        node_covariates=covariates_device,
        edge_index=edge_index,
        edge_attributes=edge_attributes,
        target_nodes=all_receivers,
        device=device,
        attention_receivers=all_receivers,
    )
    explanation_receiver_lookup = np.zeros(core.n_nodes, dtype=np.bool_)
    explanation_receiver_lookup[all_receivers] = True
    explanation_mask = explanation_receiver_lookup[edge_index[1]]
    explanation_edge_ids = np.flatnonzero(explanation_mask).astype(np.int64)
    expected_explanation_edges = edge_index[:, explanation_mask]
    if not np.array_equal(
        baseline["explanation_edge_index"], expected_explanation_edges
    ):
        raise InterpretabilityContractError(
            "Returned attention edges do not align with the verified graph"
        )
    selected_receiver = expected_explanation_edges[1]
    selected_distance = (
        edge_attributes[explanation_mask, 0].astype(np.float64)
        * float(graph.edge_attribute_scale[0])
        + float(graph.edge_attribute_mean[0])
    )
    selected_attention = baseline["attention"]
    if selected_attention.shape[0] != len(explanation_edge_ids):
        raise InterpretabilityContractError(
            "Returned attention weights do not align with selected edges"
        )
    hash_attention = _aggregate_attention(
        selected_receiver,
        selected_attention,
        selected_distance,
        hash_receivers,
    )
    outcome_attention = _aggregate_attention(
        selected_receiver,
        selected_attention,
        selected_distance,
        outcome_receivers,
    )

    outcome_receiver_lookup = np.zeros(core.n_nodes, dtype=np.bool_)
    outcome_receiver_lookup[outcome_receivers] = True
    outcome_edge_mask = outcome_receiver_lookup[selected_receiver]
    outcome_edge_ids = explanation_edge_ids[outcome_edge_mask]
    outcome_edge_receiver = selected_receiver[outcome_edge_mask]
    outcome_edge_attention = selected_attention[outcome_edge_mask].mean(axis=1)
    outcome_edge_distance = selected_distance[outcome_edge_mask]
    deletion_plans = [
        _build_distance_matched_deletion_plan(
            outcome_edge_ids,
            outcome_edge_receiver,
            outcome_edge_attention,
            outcome_edge_distance,
            outcome_receivers,
            namespace=(
                "tls-distance-matched-deletion-v1"
                f"-r{replicate:02d}"
            ),
        )
        for replicate in range(MATCHED_NULL_REPLICATES)
    ]
    top_deleted_ids = deletion_plans[0]["top_edge_ids"]
    if any(
        not np.array_equal(plan["top_edge_ids"], top_deleted_ids)
        for plan in deletion_plans[1:]
    ):
        raise RuntimeError("Top-routed edge set changed across null draws")
    deletion_null_digests = {
        hashlib.sha256(
            np.ascontiguousarray(plan["matched_edge_ids"]).tobytes()
        ).hexdigest()
        for plan in deletion_plans
    }
    if len(deletion_null_digests) != MATCHED_NULL_REPLICATES:
        raise InterpretabilityContractError(
            "Deterministic edge-null replicates are not distinct"
        )
    top_deleted = _scenario_forward(
        model,
        masked_expression=masked_expression,
        gene_mask=mask_device,
        node_covariates=covariates_device,
        edge_index=edge_index,
        edge_attributes=edge_attributes,
        target_nodes=outcome_receivers,
        device=device,
        deleted_edge_ids=top_deleted_ids,
    )

    baseline_positions = np.searchsorted(all_receivers, outcome_receivers)
    if not np.array_equal(
        all_receivers[baseline_positions], outcome_receivers
    ):
        raise RuntimeError("Outcome receivers are not aligned to baseline targets")
    baseline_program = baseline["prediction"][
        np.ix_(baseline_positions, receiver_program_indices)
    ]
    top_program = top_deleted["prediction"][:, receiver_program_indices]
    matched_programs: list[np.ndarray] = []
    for plan in deletion_plans:
        matched_deleted = _scenario_forward(
            model,
            masked_expression=masked_expression,
            gene_mask=mask_device,
            node_covariates=covariates_device,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            target_nodes=outcome_receivers,
            device=device,
            deleted_edge_ids=plan["matched_edge_ids"],
        )
        matched_programs.append(
            matched_deleted["prediction"][:, receiver_program_indices]
        )
    target_program = core.target_expression[
        np.ix_(outcome_receivers, receiver_program_indices)
    ]
    trainer = _mapping(config.get("trainer"), "config.trainer")
    huber_delta = float(trainer.get("huber_delta", 1.0))
    deletion_program_metrics = _repeated_program_effect_metrics(
        target_program,
        baseline_program,
        top_program,
        matched_programs,
        huber_delta=huber_delta,
    )
    deletion_support = _minimum_effect_support(deletion_program_metrics)

    eligible_source_nodes = ~whole_node_mask.any(axis=1)
    outcome_edge_source = expected_explanation_edges[
        0, outcome_edge_mask
    ]
    top_sender_plan = _select_bounded_top_sender_sources(
        outcome_edge_ids,
        outcome_edge_source,
        outcome_edge_attention,
        outcome_edge_distance,
        top_deleted_ids,
        eligible_source_nodes,
    )
    null_sender_plans = [
        _select_distance_matched_null_senders(
            outcome_edge_ids,
            outcome_edge_source,
            outcome_edge_attention,
            outcome_edge_distance,
            plan["matched_edge_ids"],
            eligible_source_nodes,
            top_sender_plan,
            namespace=f"tls-null-sender-source-v1-r{replicate:02d}",
        )
        for replicate, plan in enumerate(deletion_plans)
    ]
    sender_null_digests = {
        hashlib.sha256(
            np.sort(
                np.asarray(plan["source_nodes"], dtype=np.int64)
            ).tobytes()
        ).hexdigest()
        for plan in null_sender_plans
    }
    if len(sender_null_digests) != MATCHED_NULL_REPLICATES:
        raise InterpretabilityContractError(
            "Deterministic sender-null replicates are not distinct"
        )
    organizer_enrichment = _sender_organizer_enrichment(
        top_sender_plan,
        null_sender_plans,
        core.target_expression,
        organizer_indices,
    )
    enrichment_support = bool(
        float(
            organizer_enrichment[
                "minimum_top_minus_null_score_across_draws"
            ]
        )
        >= MIN_ORGANIZER_SCORE_ENRICHMENT
    )

    top_sender_expression = _mean_ablate_sender_program(
        masked_expression,
        top_sender_plan["source_nodes"],
        organizer_indices,
    )
    top_sender_output = _scenario_forward(
        model,
        masked_expression=top_sender_expression,
        gene_mask=mask_device,
        node_covariates=covariates_device,
        edge_index=edge_index,
        edge_attributes=edge_attributes,
        target_nodes=outcome_receivers,
        device=device,
    )
    top_sender_program = top_sender_output["prediction"][
        :, receiver_program_indices
    ]
    del top_sender_expression, top_sender_output

    null_sender_programs: list[np.ndarray] = []
    for plan in null_sender_plans:
        null_sender_expression = _mean_ablate_sender_program(
            masked_expression,
            plan["source_nodes"],
            organizer_indices,
        )
        null_sender_output = _scenario_forward(
            model,
            masked_expression=null_sender_expression,
            gene_mask=mask_device,
            node_covariates=covariates_device,
            edge_index=edge_index,
            edge_attributes=edge_attributes,
            target_nodes=outcome_receivers,
            device=device,
        )
        null_sender_programs.append(
            null_sender_output["prediction"][:, receiver_program_indices]
        )
        del null_sender_expression, null_sender_output
    sender_program_metrics = _repeated_program_effect_metrics(
        target_program,
        baseline_program,
        top_sender_program,
        null_sender_programs,
        huber_delta=huber_delta,
    )
    sender_support = _minimum_effect_support(sender_program_metrics)
    joint_support = bool(
        deletion_support and sender_support and enrichment_support
    )

    matched_deleted_ids = np.concatenate(
        [plan["matched_edge_ids"] for plan in deletion_plans]
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "full_core_g2_toy_interpretability",
        "status": "complete",
        "run_id": root.name,
        "campaign_id": CAMPAIGN_ID,
        "protocol": PROTOCOL,
        "analysis_precision": "float32",
        "upstream_verification": {
            "run_bundle_checksums_verified": True,
            "final_checkpoint_role": "last",
            "final_epoch": 199,
            "fixed_epoch_budget": 200,
            "checkpoint_state_sha256": checkpoint["state_dict_sha256"],
            "materialized_preprocessing_sha256": (
                core.checksums.preprocessing_sha256
            ),
            "materialized_graph_sha256": graph.checksums.graph_sha256,
            "fixed_mask_bundle_sha256": mask_bundle.checksum,
            "whole_node_mask_replicate": 0,
            "validation_or_test_used": False,
        },
        "scope": {
            "experimental_units": 1,
            "fit_scope": "all nodes transductive",
            "generalization_estimate": False,
            "outcome_conditioned_exploration": True,
            "causal_interpretation_supported": False,
        },
        "gene_panels": {
            "organizer": list(ORGANIZER_GENES),
            "receiver_program": list(RECEIVER_PROGRAM_GENES),
        },
        "selection": {
            "masked_receiver_population_count": int(len(masked_receivers)),
            "expression_independent_hash_sample_count": int(
                len(hash_receivers)
            ),
            "hash_sample_rule": (
                "SHA-256 rank over canonical whole-node-masked receiver "
                "positions; no expression values used"
            ),
            "outcome_conditioned_high_lymphoid_score_count": int(
                len(outcome_receivers)
            ),
            "outcome_score_definition": (
                "mean full-core-standardized observed expression across the "
                "eight receiver-program genes"
            ),
            "outcome_score_selected_summary": _describe(outcome_scores),
            "selection_overlap_count": int(
                np.intersect1d(hash_receivers, outcome_receivers).size
            ),
        },
        "attention_routing": {
            "layer": "last",
            "head_aggregation": "arithmetic mean across four heads",
            "expression_independent_hash_sample": hash_attention,
            "outcome_conditioned_high_score_sample": outcome_attention,
            "attention_is_importance": False,
        },
        "deletion_analysis": {
            "receiver_count": int(len(outcome_receivers)),
            "deletion_fraction": DELETION_FRACTION,
            "matched_null_replicates": MATCHED_NULL_REPLICATES,
            "rounding": "ceil per receiver, minimum one",
            "top_edge_rule": "last-layer mean-head attention descending",
            "perturbation_scope": (
                "global graph ablation: selected incoming edges are removed "
                "from the graph consumed by both GAT layers, followed by "
                "attention renormalization; this is not an isolated "
                "last-layer intervention"
            ),
            "matched_control": (
                "equal count per receiver with exact adaptive rank-distance-"
                "bin counts and eight deterministic SHA-256 draws"
            ),
            "deleted_edges_total_each_condition": int(len(top_deleted_ids)),
            "deleted_edges_per_receiver": _describe(
                deletion_plans[0]["deleted_count_per_receiver"]
            ),
            "incoming_degree_per_receiver": _describe(
                deletion_plans[0]["degree_per_receiver"]
            ),
            "adaptive_distance_bin_count": _describe(
                np.concatenate(
                    [
                        plan["distance_bin_count_per_receiver"]
                        for plan in deletion_plans
                    ]
                )
            ),
            "top_deleted_edge_distance_um": _describe(
                (
                    edge_attributes[top_deleted_ids, 0].astype(np.float64)
                    * float(graph.edge_attribute_scale[0])
                    + float(graph.edge_attribute_mean[0])
                )
            ),
            "matched_deleted_edge_distance_um": _describe(
                (
                    edge_attributes[matched_deleted_ids, 0].astype(np.float64)
                    * float(graph.edge_attribute_scale[0])
                    + float(graph.edge_attribute_mean[0])
                )
            ),
            "top_deleted_sender_organizer_score": (
                _deleted_sender_score_summary(
                    edge_index,
                    top_deleted_ids,
                    core.target_expression,
                    organizer_indices,
                )
            ),
            "matched_deleted_sender_organizer_score_pooled": (
                _deleted_sender_score_summary(
                    edge_index,
                    matched_deleted_ids,
                    core.target_expression,
                    organizer_indices,
                )
            ),
            "receiver_program_effect": deletion_program_metrics,
            "isolated_last_layer_faithfulness_test": False,
        },
        "sender_program_perturbation": {
            "performed": True,
            "receiver_count": int(len(outcome_receivers)),
            "source_count_each_condition": int(
                len(top_sender_plan["source_nodes"])
            ),
            "organizer_gene_count": int(len(organizer_indices)),
            "matched_null_replicates": MATCHED_NULL_REPLICATES,
            "intervention": (
                "replace the three visible organizer-gene inputs with zero, "
                "their full-core fitted standardized mean; gene-mask channels "
                "remain unchanged"
            ),
            "bound": (
                f"exactly {SENDER_PERTURBATION_SOURCES} distinct unmasked "
                "sender nodes and three organizer channels per condition"
            ),
            "top_sender_selection": (
                "highest maximum baseline last-layer mean-head attention "
                "among senders on top-routed edges; organizer expression is "
                "not separately thresholded or used to rank senders"
            ),
            "matched_sender_control": (
                "equal-count disjoint senders drawn from each edge-deletion "
                "null, with exact adaptive representative-distance-bin counts"
            ),
            "top_sender_representative_distance_um": _describe(
                top_sender_plan["representative_distance_um"]
            ),
            "matched_sender_representative_distance_um_pooled": _describe(
                np.concatenate(
                    [
                        plan["representative_distance_um"]
                        for plan in null_sender_plans
                    ]
                )
            ),
            "absolute_distance_mismatch_um_pooled": _describe(
                np.concatenate(
                    [
                        plan["absolute_distance_mismatch_um"]
                        for plan in null_sender_plans
                    ]
                )
            ),
            "adaptive_distance_bin_count": _describe(
                np.concatenate(
                    [
                        plan["distance_bin_count"]
                        for plan in null_sender_plans
                    ]
                )
            ),
            "organizer_score_enrichment": organizer_enrichment,
            "receiver_program_effect": sender_program_metrics,
        },
        "interpretation": {
            "descriptive_deletion_support": deletion_support,
            "descriptive_sender_program_support": sender_support,
            "descriptive_organizer_enrichment_support": enrichment_support,
            "joint_tls_dependency_support": joint_support,
            "evidence_label": (
                MAXIMUM_INTERPRETATION
                if joint_support
                else (
                    "no joint deletion-, organizer-enrichment-, and "
                    "sender-program-supported TLS-related dependency detected"
                )
            ),
            "decision_rule": (
                "both targeted interventions must have a positive mean "
                "receiver-program Huber change; every one of eight "
                "deterministic matched-null draws must "
                f"show at least {MIN_RELATIVE_HUBER_CONTRAST:.1%} relative "
                "receiver-program Huber contrast and at least "
                f"{MIN_PREDICTION_MAE_CONTRAST:.3f} standardized-expression "
                "prediction-MAE contrast for both global edge deletion and "
                "organizer-channel sender ablation; the minimum organizer "
                "score enrichment must also be at least "
                f"{MIN_ORGANIZER_SCORE_ENRICHMENT:.2f}"
            ),
            "maximum_allowed_label": MAXIMUM_INTERPRETATION,
            "attention_caveat": (
                "Attention is a normalized model-routing quantity, not "
                "biological importance. Repeated matched interventions "
                "provide model-sensitivity evidence only."
            ),
            "edge_ablation_caveat": (
                "Deleting baseline top-attention edges alters the graph for "
                "both GAT layers and renormalizes attention; it is a global "
                "edge ablation, not isolated last-layer faithfulness."
            ),
            "biological_caveat": (
                "Intervention sensitivity is model-implied within one "
                "outcome-selected, transductively fitted core; it is not "
                "causal, independently replicated, or generalization evidence."
            ),
            "organizer_score_caveat": (
                "Organizer enrichment uses observed expression and the "
                "attention ranking can itself depend on those inputs; the "
                "enrichment is not independent. Mean-ablation is an in-model "
                "input intervention, not a biological perturbation."
            ),
        },
        "held_in_fit_metric": {
            "name": summary["primary_metric_name"],
            "value": float(summary["primary_metric_value"]),
            "is_test_accuracy": False,
        },
        "privacy": {
            "aggregate_only": True,
            "protected_identifiers_emitted": False,
            "local_node_positions_emitted": False,
            "raw_edge_positions_emitted": False,
        },
    }
    _assert_aggregate_only(report)

    del (
        expression_device,
        mask_device,
        covariates_device,
        masked_expression,
        model,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.run.is_symlink():
        raise InterpretabilityContractError(
            "Run bundle path may not be a symlink"
        )
    root = args.run.resolve(strict=True)
    if not root.is_dir():
        raise InterpretabilityContractError(
            f"Run bundle is not a directory: {root}"
        )
    raw_destination = args.output.expanduser().absolute()
    if raw_destination.exists() or raw_destination.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite interpretability output: "
            f"{raw_destination}"
        )
    destination = raw_destination.resolve(strict=False)
    if destination == root or root in destination.parents:
        raise InterpretabilityContractError(
            "Output must be outside the immutable input run bundle"
        )
    report = _run_analysis(root, device=_resolve_device(args.device))
    markdown = _render_markdown(report)
    _write_exclusive_output(destination, report, markdown)
    print(
        json.dumps(
            {
                "status": "complete",
                "run_id": report["run_id"],
                "output": destination.as_posix(),
                "joint_tls_dependency_support": report["interpretation"][
                    "joint_tls_dependency_support"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
