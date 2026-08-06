#!/usr/bin/env python3
"""Bounded, gate-conditioned QKV-GAT routing and faithfulness analysis.

This analysis is exploratory and transductive.  It verifies a finalized
full-core QKV-GAT bundle, its final checkpoint, the refitted data, the exact
graph, the fixed whole-node mask, and a separate predictive-capacity report.
It then evaluates a small receiver subset chosen by an expression-independent
hash.

For the production ``bias_gate`` operator, the final-layer message coefficient
is normalized attention multiplied by the bounded edge value gate.  Both are
reported separately.  The product is called an effective routing coefficient,
not biological importance.  Its prediction faithfulness is tested by deleting
top-ranked final-layer incoming edges and comparing the prediction change with
deterministic distance-matched deletion nulls.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
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
from spatial_benchmark.qkv_graph_transformer import (  # noqa: E402
    ReceiverChunkedEdgeAwareQKVGraphTransformer,
    _incoming_softmax,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunValidationError,
    verify_run_bundle,
)


PROTOCOL = "held_in_full_core_fixed_budget"
CAMPAIGN_ID = "cmp_20260726_full_core_qkv_large_k"
IMPLEMENTATION_CLASS = (
    "spatial_benchmark.qkv_graph_transformer."
    "ReceiverChunkedEdgeAwareQKVGraphTransformer"
)
DEFAULT_RECEIVERS = 8
MAX_RECEIVERS = 32
DEFAULT_DELETION_FRACTION = 0.05
DEFAULT_NULL_REPLICATES = 4
MAX_NULL_REPLICATES = 32
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


class QKVInterpretabilityError(RuntimeError):
    """Raised when an input violates the bounded analysis contract."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Finalized successful full-core QKV-GAT run bundle.",
    )
    parser.add_argument(
        "--capacity-report",
        type=Path,
        required=True,
        help=(
            "Comparison JSON containing representation_gate.passes and "
            "representation_gate.eligible_graph_run_ids."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New caller-selected output directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device. Production high-k graphs require CUDA.",
    )
    parser.add_argument(
        "--receiver-count",
        type=int,
        default=DEFAULT_RECEIVERS,
        help="Expression-independent hash-selected whole-node receivers.",
    )
    parser.add_argument(
        "--deletion-fraction",
        type=float,
        default=DEFAULT_DELETION_FRACTION,
        help="Fraction of final-layer incoming edges deleted per receiver.",
    )
    parser.add_argument(
        "--null-replicates",
        type=int,
        default=DEFAULT_NULL_REPLICATES,
        help="Number of deterministic distance-matched deletion nulls.",
    )
    parser.add_argument(
        "--allow-negative-gate-diagnostic",
        action="store_true",
        help=(
            "Permit a failed representation gate for routing diagnostics only. "
            "The output is forcibly labeled diagnostic-only."
        ),
    )
    return parser


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QKVInterpretabilityError(f"{location} must be a mapping")
    return value


def _expect(observed: Any, expected: Any, location: str) -> None:
    if observed != expected:
        raise QKVInterpretabilityError(
            f"{location} must be {expected!r}, found {observed!r}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise QKVInterpretabilityError(
            f"Cannot parse required JSON mapping: {path}"
        ) from error
    return dict(_mapping(value, path.as_posix()))


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise QKVInterpretabilityError(
            f"Cannot parse required YAML mapping: {path}"
        ) from error
    return dict(_mapping(value, path.as_posix()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise QKVInterpretabilityError(f"Cannot checksum {path}") from error
    return digest.hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = torch.as_tensor(state_dict[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _capacity_gate(
    path: Path,
    *,
    run_id: str,
    allow_negative: bool,
) -> dict[str, Any]:
    """Validate and bind an explicit representation gate to this run."""

    report = _read_json(path)
    _expect(
        report.get("campaign_id"),
        CAMPAIGN_ID,
        "capacity_report.campaign_id",
    )
    gate = _mapping(
        report.get("representation_gate"),
        "capacity_report.representation_gate",
    )
    passes = gate.get("passes")
    if not isinstance(passes, bool):
        raise QKVInterpretabilityError(
            "capacity_report.representation_gate.passes must be boolean"
        )
    raw_run_ids = gate.get("eligible_graph_run_ids")
    if (
        not isinstance(raw_run_ids, Sequence)
        or isinstance(raw_run_ids, (str, bytes))
        or any(not isinstance(value, str) or not value for value in raw_run_ids)
    ):
        raise QKVInterpretabilityError(
            "capacity_report.representation_gate.eligible_graph_run_ids "
            "must be a string list"
        )
    run_ids = list(raw_run_ids)
    if len(set(run_ids)) != len(run_ids):
        raise QKVInterpretabilityError(
            "representation gate run IDs must be unique"
        )
    raw_evaluated = report.get("evaluated_graph_run_ids")
    if (
        not isinstance(raw_evaluated, Sequence)
        or isinstance(raw_evaluated, (str, bytes))
        or not raw_evaluated
        or any(not isinstance(value, str) or not value for value in raw_evaluated)
    ):
        raise QKVInterpretabilityError(
            "capacity_report.evaluated_graph_run_ids must be a non-empty "
            "string list"
        )
    evaluated = list(raw_evaluated)
    if len(set(evaluated)) != len(evaluated):
        raise QKVInterpretabilityError(
            "evaluated graph run IDs must be unique"
        )
    if not set(run_ids).issubset(evaluated):
        raise QKVInterpretabilityError(
            "Eligible graph runs must be a subset of evaluated graph runs"
        )
    if passes != bool(run_ids):
        raise QKVInterpretabilityError(
            "representation_gate.passes must agree with whether any graph "
            "run is individually eligible"
        )
    if run_id not in evaluated:
        raise QKVInterpretabilityError(
            "representation gate is not bound to the supplied run"
        )
    if passes and run_id not in run_ids:
        raise QKVInterpretabilityError(
            "The supplied graph run did not pass its individual "
            "graph-vs-self representation gate"
        )
    if not passes and not allow_negative:
        raise QKVInterpretabilityError(
            "Predictive representation gate failed; refuse interpretation. "
            "Use --allow-negative-gate-diagnostic only for a clearly labeled "
            "negative-gate routing diagnostic."
        )
    return {
        "passes": passes,
        "analysis_permitted": bool(passes or allow_negative),
        "diagnostic_only": not passes,
        "negative_gate_override_used": bool(not passes and allow_negative),
        "eligible_graph_run_count": len(run_ids),
        "evaluated_graph_run_count": len(evaluated),
        "report_sha256": _file_sha256(path),
    }


def _model_key(value: object) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )


def _validate_run_contract(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    training: Mapping[str, Any],
) -> None:
    campaign = _mapping(config.get("campaign"), "config.campaign")
    model = _mapping(config.get("model"), "config.model")
    graph = _mapping(config.get("graph"), "config.graph")
    trainer = _mapping(config.get("trainer"), "config.trainer")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation")
    dataset = _mapping(config.get("dataset"), "config.dataset")
    features = _mapping(config.get("features"), "config.features")

    _expect(campaign.get("campaign_id"), CAMPAIGN_ID, "campaign.campaign_id")
    _expect(manifest.get("campaign_id"), CAMPAIGN_ID, "manifest.campaign_id")
    _expect(summary.get("run_id"), root.name, "summary.run_id")
    _expect(summary.get("status"), "success", "summary.status")
    _expect(
        summary.get("training_exit_status"),
        "success",
        "summary.training_exit_status",
    )
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
    _expect(config.get("seed"), 0, "config.seed")
    _expect(config.get("fold"), 0, "config.fold")

    if _model_key(model.get("name")) != "qkvgat":
        raise QKVInterpretabilityError("model must be QKV-GAT")
    _expect(
        model.get("family"),
        "edge_aware_qkv_graph_transformer",
        "model.family",
    )
    _expect(
        model.get("edge_conditioning_mode"),
        "bias_gate",
        "model.edge_conditioning_mode",
    )
    _expect(
        model.get("exact_receiver_partitioning"),
        True,
        "model.exact_receiver_partitioning",
    )
    _expect(model.get("implicit_self_loops"), False, "model.implicit_self_loops")
    _expect(
        model.get("trainable_node_identifiers"),
        False,
        "model.trainable_node_identifiers",
    )
    _expect(
        model.get("trainable_edge_identifiers"),
        False,
        "model.trainable_edge_identifiers",
    )
    _expect(model.get("uses_graph_inputs"), True, "model.uses_graph_inputs")
    _expect(model.get("uses_edge_inputs"), True, "model.uses_edge_inputs")
    _expect(features.get("use_edge_features"), True, "features.use_edge_features")
    _expect(summary.get("model_name"), model.get("name"), "summary.model_name")

    k = int(graph.get("k", -1))
    if k not in {1000, 5000}:
        raise QKVInterpretabilityError("graph.k must be 1000 or 5000")
    _expect(graph.get("neighbor_k"), k, "graph.neighbor_k")
    _expect(
        graph.get("kind"),
        "exact_spatial_knn_radius_guard",
        "graph.kind",
    )
    _expect(graph.get("symmetry"), "mutual", "graph.symmetry")
    _expect(graph.get("self_loops"), False, "graph.self_loops")
    _expect(graph.get("edge_dropout"), 0.0, "graph.edge_dropout")
    _expect(graph.get("full_core_graph"), True, "graph.full_core_graph")
    if bool(graph.get("neighbor_sampling", False)):
        raise QKVInterpretabilityError("graph.neighbor_sampling is prohibited")

    _expect(trainer.get("max_epochs"), 300, "trainer.max_epochs")
    _expect(trainer.get("fixed_epoch_budget"), True, "trainer.fixed_epoch_budget")
    _expect(trainer.get("early_stopping"), False, "trainer.early_stopping")
    _expect(trainer.get("restore_best"), False, "trainer.restore_best")
    _expect(
        trainer.get("primary_checkpoint_role"),
        "last",
        "trainer.primary_checkpoint_role",
    )
    _expect(trainer.get("neighbor_sampling"), False, "trainer.neighbor_sampling")
    _expect(
        trainer.get("graph_execution"),
        "full_core_exact_no_neighbor_sampling",
        "trainer.graph_execution",
    )
    _expect(
        summary.get("fixed_epoch_budget"),
        300,
        "summary.fixed_epoch_budget",
    )
    _expect(summary.get("final_epoch"), 299, "summary.final_epoch")
    _expect(summary.get("checkpoint_role"), "last", "summary.checkpoint_role")

    _expect(evaluation.get("protocol"), PROTOCOL, "evaluation.protocol")
    _expect(summary.get("evaluation_protocol"), PROTOCOL, "summary.protocol")
    _expect(evaluation.get("splits"), ["fit"], "evaluation.splits")
    _expect(
        evaluation.get("canonical_prediction_split"),
        "fit",
        "evaluation.canonical_prediction_split",
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
        evaluation.get("fixed_mask_bundle"),
        True,
        "evaluation.fixed_mask_bundle",
    )
    _expect(
        evaluation.get("mask_replicates_per_mode"),
        3,
        "evaluation.mask_replicates_per_mode",
    )
    _expect(
        dataset.get("validation_or_test_partition_present"),
        False,
        "dataset.validation_or_test_partition_present",
    )
    _expect(
        dataset.get("experimental_unit"),
        "single_spatial_core",
        "dataset.experimental_unit",
    )
    selection = _mapping(
        summary.get("canonical_prediction_selection"),
        "summary.canonical_prediction_selection",
    )
    _expect(selection.get("split"), "fit", "canonical selection split")
    _expect(
        selection.get("mask_mode"),
        "whole_node",
        "canonical selection mask mode",
    )
    _expect(
        selection.get("mask_replicate"),
        0,
        "canonical selection mask replicate",
    )

    construction = _mapping(
        training.get("model_construction"),
        "training.model_construction",
    )
    _expect(
        construction.get("canonical_model_key"),
        "qkvgat",
        "model_construction.canonical_model_key",
    )
    _expect(
        construction.get("implementation_class"),
        IMPLEMENTATION_CLASS,
        "model_construction.implementation_class",
    )
    _expect(
        training.get("training_protocol"),
        PROTOCOL,
        "training.training_protocol",
    )
    _expect(training.get("final_epoch"), 299, "training.final_epoch")
    _expect(
        training.get("fixed_epoch_budget"),
        300,
        "training.fixed_epoch_budget",
    )


def _load_verified_checkpoint(
    root: Path,
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    training: Mapping[str, Any],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    path = root / "checkpoints" / "last.ckpt"
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise QKVInterpretabilityError(
            "Final checkpoint must be a non-empty regular file"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise QKVInterpretabilityError(
            "Final checkpoint failed safe weights-only loading"
        ) from error
    checkpoint = dict(_mapping(payload, "checkpoint"))
    state_mapping = _mapping(
        checkpoint.get("model_state_dict"),
        "checkpoint.model_state_dict",
    )
    if not all(isinstance(value, Tensor) for value in state_mapping.values()):
        raise QKVInterpretabilityError(
            "Checkpoint state dict contains non-tensor values"
        )
    state = dict(state_mapping)
    expected = {
        "schema_version": 1,
        "run_id": root.name,
        "checkpoint_role": "last",
        "checkpoint_policy": "final_epoch_no_validation_selection",
        "training_protocol": PROTOCOL,
        "model_name": _mapping(config.get("model"), "config.model").get("name"),
        "epoch": 299,
        "fixed_epoch_budget": 300,
        "full_core_preprocessing_sha256": _mapping(
            config.get("dataset"), "config.dataset"
        ).get("dataset_fingerprint"),
        "graph_sha256": summary.get("graph_sha256"),
        "evaluation_mask_bundle_sha256": summary.get(
            "evaluation_mask_bundle_sha256"
        ),
    }
    for field, value in expected.items():
        _expect(checkpoint.get(field), value, f"checkpoint.{field}")
    _expect(
        dict(_mapping(checkpoint.get("model_config"), "checkpoint.model_config")),
        dict(_mapping(config.get("model"), "config.model")),
        "checkpoint.model_config",
    )
    checkpoint_construction = dict(
        _mapping(
            checkpoint.get("model_construction"),
            "checkpoint.model_construction",
        )
    )
    training_construction = dict(
        _mapping(
            training.get("model_construction"),
            "training.model_construction",
        )
    )
    _expect(
        checkpoint_construction,
        training_construction,
        "checkpoint.model_construction",
    )
    checksum = _state_dict_sha256(state)
    _expect(
        checkpoint.get("state_dict_sha256"),
        checksum,
        "checkpoint.state_dict_sha256",
    )
    _expect(
        training.get("state_dict_sha256"),
        checksum,
        "training.state_dict_sha256",
    )
    return state, checkpoint


def _resolve_prepared_artifact(config: Mapping[str, Any]) -> Path:
    dataset = _mapping(config.get("dataset"), "config.dataset")
    reference = dataset.get("prepared_artifact_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise QKVInterpretabilityError(
            "dataset.prepared_artifact_reference is required"
        )
    path = Path(reference)
    if not path.is_absolute():
        path = current_paths(anchor=_BOOTSTRAP_ROOT).project_root / path
    return path


def _rebuild_verified_core(
    config: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> FullCoreData:
    core = load_and_refit_full_core(_resolve_prepared_artifact(config))
    dataset = _mapping(config.get("dataset"), "config.dataset")
    _expect(
        core.checksums.preprocessing_sha256,
        dataset.get("dataset_fingerprint"),
        "materialized preprocessing checksum",
    )
    _expect(
        core.checksums.to_dict(),
        dict(
            _mapping(
                inputs.get("preprocessing_checksums"),
                "inputs.preprocessing_checksums",
            )
        ),
        "input preprocessing checksums",
    )
    if core.preprocessing_qc.protected_identifier_arrays_returned:
        raise QKVInterpretabilityError(
            "Full-core loader unexpectedly returned protected identifiers"
        )
    return core


def _rebuild_verified_graph(
    core: FullCoreData,
    config: Mapping[str, Any],
    inputs: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> ReceiverSortedGraph:
    graph_config = _mapping(config.get("graph"), "config.graph")
    graph = build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=int(graph_config["k"]),
        radius_guard_um=float(graph_config["radius_guard_um"]),
        query_chunk_size=int(graph_config.get("query_chunk_size", 2048)),
        receiver_chunk_size=int(graph_config.get("receiver_shard_size", 512)),
        mutual_search_chunk_size=int(
            graph_config.get("mutual_search_chunk_size", 4_000_000)
        ),
        workers=int(graph_config.get("construction_workers", 1)),
        epsilon=float(graph_config.get("edge_standardizer_epsilon", 1e-8)),
    )
    _expect(graph.n_nodes, core.n_nodes, "graph node count")
    _expect(
        graph.qc.n_directed_edges,
        int(graph_config["expected_directed_edges"]),
        "graph directed edge count",
    )
    _expect(
        graph.checksums.graph_sha256,
        graph_config.get("expected_materialized_graph_sha256"),
        "graph configured checksum",
    )
    _expect(
        graph.checksums.graph_sha256,
        summary.get("graph_sha256"),
        "graph summary checksum",
    )
    _expect(
        graph.qc.n_directed_edges,
        summary.get("graph_directed_edges"),
        "graph summary edge count",
    )
    _expect(
        graph.checksums.to_dict(),
        dict(
            _mapping(
                inputs.get("graph_checksums"),
                "inputs.graph_checksums",
            )
        ),
        "input graph checksums",
    )
    _expect(
        dict(graph_config),
        dict(_mapping(inputs.get("graph_config"), "inputs.graph_config")),
        "input graph config",
    )
    if (
        not graph.qc.receiver_sorted
        or graph.qc.self_loops
        or graph.qc.duplicate_directed_edges
        or not graph.qc.directed_edge_pairs_are_symmetric
    ):
        raise QKVInterpretabilityError(
            "Materialized graph violates exact mutual graph invariants"
        )
    return graph


def _mask_rates(masking: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("rates", "rate"):
        value = masking.get(name)
        if isinstance(value, Mapping):
            return value
    raise QKVInterpretabilityError("masking.rates or masking.rate is required")


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
    specs = [
        MaskSpec(
            mode=mode,
            partial_gene_rate=partial_rate,
            node_rate=node_rate,
            block_node_rate=block_rate,
            block_width_um=masking.get("block_width_um"),
            block_shape=str(masking.get("block_shape", "disk")),
            label=label,
        )
        for mode, label in (
            ("partial", "partial_gene"),
            ("node", "whole_node"),
            ("block", "spatial_block"),
        )
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
    _expect(
        dict(
            _mapping(
                mask_provenance.get("bundle_manifest"),
                "mask_provenance.bundle_manifest",
            )
        ),
        dict(bundle.manifest),
        "reconstructed fixed-mask manifest",
    )
    for location, checksum in (
        ("checkpoint", checkpoint.get("evaluation_mask_bundle_sha256")),
        ("summary", summary.get("evaluation_mask_bundle_sha256")),
    ):
        _expect(checksum, bundle.checksum, f"{location} mask checksum")
    mask = np.asarray(bundle.get("fit", "whole_node", 0), dtype=np.bool_)
    selected = _whole_node_selected_rows(mask)
    if not selected.any():
        raise QKVInterpretabilityError("Whole-node mask selected no receivers")
    return mask, bundle


def _whole_node_selected_rows(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim != 2 or values.dtype != np.bool_:
        raise TypeError("whole-node mask must be a boolean matrix")
    selected = values.any(axis=1)
    if not bool(np.all(values == selected[:, None])):
        raise QKVInterpretabilityError(
            "Canonical whole-node mask contains partial rows"
        )
    return selected


def _hash_key(namespace: str, *values: int) -> bytes:
    payload = json.dumps(
        [namespace, *(int(value) for value in values)],
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
        values.tolist(),
        key=lambda value: (_hash_key(namespace, value), value),
    )
    return np.sort(np.asarray(ranked[:count], dtype=np.int64))


def _instantiate_verified_model(
    state: Mapping[str, Tensor],
    training: Mapping[str, Any],
) -> ReceiverChunkedEdgeAwareQKVGraphTransformer:
    construction = _mapping(
        training.get("model_construction"),
        "training.model_construction",
    )
    _expect(
        construction.get("implementation_class"),
        IMPLEMENTATION_CLASS,
        "model_construction.implementation_class",
    )
    arguments = dict(
        _mapping(
            construction.get("constructor_arguments"),
            "model_construction.constructor_arguments",
        )
    )
    allowed = {
        "num_genes",
        "edge_attribute_dim",
        "node_covariate_dim",
        "hidden_dim",
        "attention_heads",
        "attention_head_dim",
        "graph_layers",
        "ffn_dim",
        "decoder_dim",
        "edge_hidden_dim",
        "edge_embedding_dim",
        "edge_conditioning_mode",
        "dropout",
        "attention_dropout",
        "receiver_chunk_size",
        "max_edges_per_chunk",
        "activation_checkpointing",
    }
    if set(arguments) != allowed:
        raise QKVInterpretabilityError(
            "Recorded constructor arguments are incomplete or contain "
            f"unexpected fields: {sorted(set(arguments) ^ allowed)}"
        )
    _expect(
        arguments.get("edge_conditioning_mode"),
        "bias_gate",
        "constructor edge_conditioning_mode",
    )
    model = ReceiverChunkedEdgeAwareQKVGraphTransformer(**arguments)
    try:
        incompatibility = model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise QKVInterpretabilityError(
            "Checkpoint does not match the recorded QKV implementation"
        ) from error
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise QKVInterpretabilityError("Checkpoint state loading was not strict")
    observed_parameters = sum(parameter.numel() for parameter in model.parameters())
    _expect(
        observed_parameters,
        int(training["parameter_count"]),
        "reconstructed model parameter count",
    )
    model.eval()
    return model


def _selected_edge_arrays(
    edge_index: np.ndarray,
    edge_attributes: np.ndarray,
    selected_receivers: np.ndarray,
    coordinates_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    receivers = edge_index[1]
    if len(receivers) > 1 and np.any(receivers[1:] < receivers[:-1]):
        raise QKVInterpretabilityError("Graph is not receiver-sorted")
    pieces: list[np.ndarray] = []
    for receiver in selected_receivers.tolist():
        start = int(np.searchsorted(receivers, receiver, side="left"))
        stop = int(np.searchsorted(receivers, receiver, side="right"))
        if start == stop:
            raise QKVInterpretabilityError(
                "A selected receiver has no incoming edges"
            )
        pieces.append(np.arange(start, stop, dtype=np.int64))
    selected_ids = np.concatenate(pieces)
    selected_edges = np.ascontiguousarray(edge_index[:, selected_ids])
    selected_attributes = np.ascontiguousarray(edge_attributes[selected_ids])
    source = selected_edges[0]
    receiver = selected_edges[1]
    distance = np.linalg.norm(
        coordinates_um[source] - coordinates_um[receiver],
        axis=1,
    )
    return selected_edges, selected_attributes, distance.astype(np.float64)


def _autocast(
    *,
    enabled: bool,
    device: torch.device,
    dtype_name: str,
):
    if not enabled:
        return nullcontext()
    name = str(dtype_name)
    if name not in {"auto", "float16", "bfloat16"}:
        raise QKVInterpretabilityError(
            "AMP dtype must be 'auto', 'float16', or 'bfloat16'"
        )
    if device.type not in {"cuda", "cpu"}:
        raise QKVInterpretabilityError(
            "AMP is supported only on CUDA or CPU"
        )
    if name == "auto":
        dtype = (
            torch.float16
            if device.type == "cuda"
            else torch.bfloat16
        )
    elif name == "float16":
        if device.type == "cpu":
            raise QKVInterpretabilityError(
                "CPU AMP requires bfloat16 or auto"
            )
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _compute_penultimate(
    model: ReceiverChunkedEdgeAwareQKVGraphTransformer,
    expression: Tensor,
    gene_mask: Tensor,
    node_covariates: Tensor,
    edge_index: Tensor,
    edge_attributes: Tensor,
    *,
    amp: bool,
    amp_dtype: str,
) -> Tensor:
    """Execute the encoder and all graph blocks except the final block."""

    device = expression.device
    with torch.no_grad(), _autocast(
        enabled=amp,
        device=device,
        dtype_name=amp_dtype,
    ):
        embedding, _ = model._encode_and_targets(
            expression,
            gene_mask,
            node_covariates,
            None,
        )
        layout = model._receiver_layout(
            edge_index,
            num_nodes=embedding.shape[0],
        )
        for block in model.blocks[:-1]:
            embedding = model._chunked_block_forward(
                block=block,
                node_embedding=embedding,
                edge_index=edge_index,
                edge_attributes=edge_attributes,
                layout=layout,
                explanation_receivers=None,
            )[0]
    return embedding


def _local_receivers(
    global_receivers: Tensor,
    selected_receivers: Tensor,
) -> Tensor:
    positions = torch.searchsorted(selected_receivers, global_receivers)
    if (
        global_receivers.numel()
        and (
            bool((positions >= selected_receivers.numel()).any())
            or not torch.equal(
                selected_receivers.index_select(0, positions),
                global_receivers,
            )
        )
    ):
        raise QKVInterpretabilityError(
            "Selected edges contain an unselected receiver"
        )
    return positions


def _final_layer_components(
    model: ReceiverChunkedEdgeAwareQKVGraphTransformer,
    penultimate: Tensor,
    edge_index: Tensor,
    edge_attributes: Tensor,
    selected_receivers: Tensor,
) -> dict[str, Tensor]:
    """Recompute exact final-layer QKV attention and bounded value gates."""

    block = model.blocks[-1]
    if block.edge_conditioning_mode != "bias_gate":
        raise QKVInterpretabilityError(
            "Effective routing is implemented only for production bias_gate"
        )
    assert block.edge_value_gate is not None
    query, key, value = block.project_nodes(penultimate)
    source = edge_index[0]
    receiver = edge_index[1]
    local_receiver = _local_receivers(receiver, selected_receivers)
    encoded_edges = model.edge_encoder(edge_attributes)
    scores = (
        torch.einsum(
            "ehd,ehd->eh",
            query.index_select(0, receiver),
            key.index_select(0, source),
        )
        * block.attention_scale
        + block.edge_attention_bias(encoded_edges)
    )
    attention = _incoming_softmax(
        scores,
        local_receiver,
        num_receivers=selected_receivers.numel(),
    )
    gate = 1.0 + torch.tanh(
        block.edge_value_gate(encoded_edges)
    ).to(dtype=attention.dtype)
    return {
        "source": source,
        "receiver": receiver,
        "local_receiver": local_receiver,
        "selected_receivers": selected_receivers,
        "residual": penultimate.index_select(0, selected_receivers),
        "values": value.index_select(0, source).to(dtype=attention.dtype),
        "scores": scores,
        "attention": attention,
        "value_gate": gate,
    }


def _readout_with_deleted_edges(
    model: ReceiverChunkedEdgeAwareQKVGraphTransformer,
    components: Mapping[str, Tensor],
    deleted_edge_positions: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode selected receivers after an isolated final-layer deletion."""

    attention = components["attention"]
    value_gate = components["value_gate"]
    local_receiver = components["local_receiver"]
    values = components["values"]
    keep = torch.ones(
        attention.shape[0],
        dtype=torch.bool,
        device=attention.device,
    )
    if deleted_edge_positions is not None and deleted_edge_positions.numel():
        keep[deleted_edge_positions] = False
    retained_receiver = local_receiver[keep]
    if bool(
        (
            torch.bincount(
                retained_receiver,
                minlength=components["selected_receivers"].numel(),
            )
            == 0
        ).any()
    ):
        raise QKVInterpretabilityError(
            "Deletion removed every incoming edge for a receiver"
        )
    renormalized_attention = _incoming_softmax(
        components["scores"][keep],
        retained_receiver,
        num_receivers=components["selected_receivers"].numel(),
    )
    effective = renormalized_attention * value_gate[keep]
    messages = values[keep] * effective.unsqueeze(-1)
    aggregated = attention.new_zeros(
        (
            components["selected_receivers"].numel(),
            attention.shape[1],
            values.shape[2],
        )
    )
    aggregated.index_add_(0, retained_receiver, messages)
    final_embedding = model.blocks[-1].finish_partition(
        components["residual"],
        aggregated,
    )
    prediction = model.decoder(final_embedding)
    return prediction, renormalized_attention, effective


def _normalize_effective(
    effective: np.ndarray,
    local_receiver: np.ndarray,
    receiver_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(effective, dtype=np.float64)
    receiver = np.asarray(local_receiver, dtype=np.int64)
    if (
        values.ndim != 2
        or receiver.shape != (values.shape[0],)
        or np.any(values < 0)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Effective routing arrays are invalid")
    mass = np.zeros((receiver_count, values.shape[1]), dtype=np.float64)
    np.add.at(mass, receiver, values)
    normalized = np.divide(
        values,
        mass[receiver],
        out=np.zeros_like(values),
        where=mass[receiver] > 0,
    )
    return normalized, mass


def _describe(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Summary values must be non-empty and finite")
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _routing_summary(
    local_receiver: np.ndarray,
    normalized_weights: np.ndarray,
    distance_um: np.ndarray,
    *,
    receiver_count: int,
) -> dict[str, Any]:
    receiver = np.asarray(local_receiver, dtype=np.int64)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    distance = np.asarray(distance_um, dtype=np.float64)
    if (
        weights.ndim != 2
        or receiver.shape != (len(weights),)
        or distance.shape != (len(weights),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("Routing summary arrays are invalid")
    entropy: list[float] = []
    effective_count: list[float] = []
    weighted_distance: list[float] = []
    normalization_error: list[float] = []
    degrees: list[int] = []
    zero_mass_receiver_heads = 0
    for node in range(receiver_count):
        incoming = np.flatnonzero(receiver == node)
        if not len(incoming):
            raise ValueError("A routing receiver has no incoming edges")
        degrees.append(len(incoming))
        node_weights = weights[incoming]
        sums = node_weights.sum(axis=0)
        for head in range(weights.shape[1]):
            if sums[head] <= 0:
                zero_mass_receiver_heads += 1
                continue
            normalization_error.append(float(abs(sums[head] - 1.0)))
            probability = node_weights[:, head]
            positive = probability > 0
            value = -float(
                np.sum(probability[positive] * np.log(probability[positive]))
            )
            entropy.append(value)
            effective_count.append(math.exp(value))
            weighted_distance.append(
                float(np.dot(probability, distance[incoming]))
            )
    return {
        "receiver_count": receiver_count,
        "head_count": int(weights.shape[1]),
        "incoming_degree": _describe(degrees),
        "nonzero_mass_receiver_heads": len(entropy),
        "zero_mass_receiver_heads": zero_mass_receiver_heads,
        "entropy_nats_per_receiver_head": (
            _describe(entropy) if entropy else None
        ),
        "effective_neighbor_count_per_receiver_head": (
            _describe(effective_count) if effective_count else None
        ),
        "weighted_distance_um_per_receiver_head": (
            _describe(weighted_distance) if weighted_distance else None
        ),
        "maximum_normalization_error": (
            float(max(normalization_error))
            if normalization_error
            else None
        ),
    }


def _rank_distance_bins(
    distances: np.ndarray,
    edge_positions: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    order = np.lexsort((edge_positions, distances))
    bins = np.empty(len(distances), dtype=np.int64)
    bins[order] = (
        np.arange(len(distances), dtype=np.int64) * int(bin_count)
    ) // len(distances)
    return bins


def _distance_matched_deletion_plan(
    local_receiver: np.ndarray,
    routing_score: np.ndarray,
    distance_um: np.ndarray,
    *,
    receiver_count: int,
    fraction: float,
    null_replicates: int,
    namespace: str,
) -> dict[str, Any]:
    receiver = np.asarray(local_receiver, dtype=np.int64)
    routing = np.asarray(routing_score, dtype=np.float64)
    distance = np.asarray(distance_um, dtype=np.float64)
    if (
        receiver.ndim != 1
        or routing.shape != receiver.shape
        or distance.shape != receiver.shape
        or not np.isfinite(routing).all()
        or not np.isfinite(distance).all()
    ):
        raise ValueError("Deletion-plan arrays are invalid")
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("fraction must be in (0, 0.5)")
    if int(null_replicates) <= 0:
        raise ValueError("null_replicates must be positive")

    top_by_receiver: list[np.ndarray] = []
    bins_by_receiver: list[np.ndarray] = []
    positions_by_receiver: list[np.ndarray] = []
    delete_counts: list[int] = []
    bin_counts: list[int] = []
    for node in range(receiver_count):
        positions = np.flatnonzero(receiver == node)
        degree = len(positions)
        if not degree:
            raise ValueError("A deletion receiver has no incoming edges")
        delete_count = max(1, int(math.ceil(degree * float(fraction))))
        if delete_count * 2 > degree:
            raise ValueError("Too few edges for non-overlapping null deletion")
        order = np.lexsort((positions, -routing[positions]))
        top_local = order[:delete_count]
        top_mask = np.zeros(degree, dtype=np.bool_)
        top_mask[top_local] = True
        chosen_bins: np.ndarray | None = None
        chosen_count = 0
        for proposed in range(min(10, degree), 0, -1):
            bins = _rank_distance_bins(
                distance[positions],
                positions,
                proposed,
            )
            if all(
                int(np.sum(top_mask & (bins == number)))
                <= int(np.sum((~top_mask) & (bins == number)))
                for number in range(proposed)
            ):
                chosen_bins = bins
                chosen_count = proposed
                break
        if chosen_bins is None:
            raise RuntimeError("Could not construct distance-matched nulls")
        positions_by_receiver.append(positions)
        top_by_receiver.append(top_local)
        bins_by_receiver.append(chosen_bins)
        delete_counts.append(delete_count)
        bin_counts.append(chosen_count)

    top = np.sort(
        np.concatenate(
            [
                positions[top_local]
                for positions, top_local in zip(
                    positions_by_receiver,
                    top_by_receiver,
                    strict=True,
                )
            ]
        )
    )
    nulls: list[np.ndarray] = []
    for replicate in range(int(null_replicates)):
        selected_for_replicate: list[np.ndarray] = []
        for node, (positions, top_local, bins) in enumerate(
            zip(
                positions_by_receiver,
                top_by_receiver,
                bins_by_receiver,
                strict=True,
            )
        ):
            top_mask = np.zeros(len(positions), dtype=np.bool_)
            top_mask[top_local] = True
            matched: list[int] = []
            for bin_number in range(int(bins.max(initial=-1)) + 1):
                needed = int(np.sum(top_mask & (bins == bin_number)))
                candidates = np.flatnonzero(
                    (~top_mask) & (bins == bin_number)
                )
                ranked = sorted(
                    candidates.tolist(),
                    key=lambda offset: (
                        _hash_key(
                            namespace,
                            replicate,
                            node,
                            int(positions[offset]),
                        ),
                        int(positions[offset]),
                    ),
                )
                matched.extend(ranked[:needed])
            if len(matched) != len(top_local):
                raise RuntimeError("Distance-matched deletion count drifted")
            selected_for_replicate.append(
                positions[np.asarray(matched, dtype=np.int64)]
            )
        null = np.sort(np.concatenate(selected_for_replicate))
        if np.intersect1d(top, null).size:
            raise RuntimeError("Top and null deletion edges overlap")
        nulls.append(null)

    top_distance = distance[top]
    null_distance_delta = [
        float(abs(distance[null].mean() - top_distance.mean()))
        for null in nulls
    ]
    return {
        "top_positions": top,
        "null_positions": tuple(nulls),
        "deleted_count_per_receiver": np.asarray(
            delete_counts,
            dtype=np.int64,
        ),
        "distance_bin_count_per_receiver": np.asarray(
            bin_counts,
            dtype=np.int64,
        ),
        "top_mean_distance_um": float(top_distance.mean()),
        "null_mean_distance_absolute_difference_um": np.asarray(
            null_distance_delta,
            dtype=np.float64,
        ),
    }


def _huber(values: np.ndarray, *, delta: float) -> float:
    absolute = np.abs(values)
    losses = np.where(
        absolute <= delta,
        0.5 * values**2,
        delta * (absolute - 0.5 * delta),
    )
    return float(losses.mean())


def _prediction_change(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    target: np.ndarray,
    *,
    huber_delta: float,
) -> dict[str, float]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    perturbed_values = np.asarray(perturbed, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if (
        baseline_values.shape != perturbed_values.shape
        or truth.shape != baseline_values.shape
        or not np.isfinite(baseline_values).all()
        or not np.isfinite(perturbed_values).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("Prediction arrays are invalid")
    change = perturbed_values - baseline_values
    baseline_error = baseline_values - truth
    perturbed_error = perturbed_values - truth
    return {
        "mean_absolute_prediction_change": float(np.mean(np.abs(change))),
        "root_mean_squared_prediction_change": float(
            np.sqrt(np.mean(change**2))
        ),
        "baseline_huber": _huber(baseline_error, delta=huber_delta),
        "perturbed_huber": _huber(perturbed_error, delta=huber_delta),
        "huber_change": (
            _huber(perturbed_error, delta=huber_delta)
            - _huber(baseline_error, delta=huber_delta)
        ),
        "baseline_mae": float(np.mean(np.abs(baseline_error))),
        "perturbed_mae": float(np.mean(np.abs(perturbed_error))),
        "mae_change": float(
            np.mean(np.abs(perturbed_error))
            - np.mean(np.abs(baseline_error))
        ),
    }


def _tls_annotation(
    gene_names: Sequence[str],
    target_expression: np.ndarray,
    selected_receivers: np.ndarray,
    source: np.ndarray,
    local_receiver: np.ndarray,
    effective_distribution: np.ndarray,
) -> dict[str, Any]:
    lookup = {str(name): index for index, name in enumerate(gene_names)}
    organizers = [name for name in ORGANIZER_GENES if name in lookup]
    receiver_genes = [
        name for name in RECEIVER_PROGRAM_GENES if name in lookup
    ]
    result: dict[str, Any] = {
        "status": "unavailable_due_panel_coverage",
        "exploratory_annotation_only": True,
        "selection_used_tls_expression": False,
        "uses_posthoc_hidden_target_expression": True,
        "organizer_genes_present": organizers,
        "organizer_genes_missing": [
            name for name in ORGANIZER_GENES if name not in lookup
        ],
        "receiver_program_genes_present": receiver_genes,
        "receiver_program_genes_missing": [
            name for name in RECEIVER_PROGRAM_GENES if name not in lookup
        ],
        "caution": (
            "TLS agreement is annotation, not independent validation, "
            "prediction faithfulness, mechanism, or causality. Organizer and "
            "receiver scores use posthoc target expression, never model input "
            "or receiver selection."
        ),
    }
    if not organizers or not receiver_genes:
        return result

    organizer_indices = np.asarray(
        [lookup[name] for name in organizers],
        dtype=np.int64,
    )
    receiver_indices = np.asarray(
        [lookup[name] for name in receiver_genes],
        dtype=np.int64,
    )
    sender_scores = target_expression[source][:, organizer_indices].mean(axis=1)
    receiver_scores = target_expression[selected_receivers][
        :, receiver_indices
    ].mean(axis=1)
    mean_head_routing = effective_distribution.mean(axis=1)
    weighted_sender_scores: list[float] = []
    unweighted_sender_scores: list[float] = []
    for node in range(len(selected_receivers)):
        incoming = np.flatnonzero(local_receiver == node)
        probability = mean_head_routing[incoming]
        if float(probability.sum()) <= 0:
            result["status"] = "unavailable_zero_effective_routing_mass"
            return result
        probability = probability / probability.sum()
        weighted_sender_scores.append(
            float(np.dot(probability, sender_scores[incoming]))
        )
        unweighted_sender_scores.append(float(sender_scores[incoming].mean()))
    enrichment = (
        np.asarray(weighted_sender_scores)
        - np.asarray(unweighted_sender_scores)
    )
    result.update(
        {
            "status": (
                "computed_complete_panel"
                if len(organizers) == len(ORGANIZER_GENES)
                and len(receiver_genes) == len(RECEIVER_PROGRAM_GENES)
                else "computed_partial_panel"
            ),
            "receiver_program_standardized_score": _describe(receiver_scores),
            "effective_routing_weighted_sender_organizer_score": _describe(
                weighted_sender_scores
            ),
            "unweighted_incoming_sender_organizer_score": _describe(
                unweighted_sender_scores
            ),
            "routing_weighted_minus_unweighted_sender_score": _describe(
                enrichment
            ),
        }
    )
    return result


def _assert_aggregate_only(payload: Mapping[str, Any]) -> None:
    forbidden = {
        "receiver_ids",
        "receiver_indices",
        "source_ids",
        "source_indices",
        "edge_ids",
        "edge_positions",
        "predictions",
        "expression",
        "coordinates",
    }

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                normalized = str(key).lower()
                if normalized in forbidden:
                    raise QKVInterpretabilityError(
                        f"Aggregate output contains forbidden key {path}.{key}"
                    )
                visit(child, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            if len(value) > 32:
                raise QKVInterpretabilityError(
                    f"Aggregate output contains a row-like sequence at {path}"
                )
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise QKVInterpretabilityError(
                f"Aggregate output contains non-finite value at {path}"
            )

    visit(payload, "report")


def _markdown_report(report: Mapping[str, Any]) -> str:
    gate = _mapping(report["representation_gate"], "report.gate")
    routing = _mapping(report["routing"], "report.routing")
    faithfulness = _mapping(report["faithfulness"], "report.faithfulness")
    top = _mapping(faithfulness["top_effective_routing_deletion"], "top")
    comparison = _mapping(faithfulness["top_vs_matched_null"], "comparison")
    ratio = comparison["prediction_change_ratio_to_null_mean"]
    ratio_text = (
        f"{float(ratio):.4g}"
        if ratio is not None
        else "undefined because the matched-null mean was zero"
    )
    status = (
        "capacity-gate-passed exploratory analysis"
        if gate["passes"]
        else "negative-capacity-gate diagnostic only"
    )
    effective = _mapping(routing["effective_routing"], "effective")
    effective_count_value = effective[
        "effective_neighbor_count_per_receiver_head"
    ]
    distance_value = effective["weighted_distance_um_per_receiver_head"]
    if effective_count_value is None or distance_value is None:
        routing_sentence = (
            "Every selected receiver/head had zero effective routing mass; "
            "effective-neighbor and weighted-distance summaries are undefined."
        )
    else:
        effective_count = _mapping(
            effective_count_value,
            "effective_count",
        )
        distance = _mapping(distance_value, "distance")
        routing_sentence = (
            "Mean effective-neighbor count per receiver/head: "
            f"{effective_count['mean']:.3f}; mean effective-routing-"
            f"weighted distance: {distance['mean']:.3f} µm."
        )
    return "\n".join(
        [
            "# QKV-GAT bounded routing diagnostic",
            "",
            f"Status: **{status}**.",
            "",
            (
                f"The analysis used {report['selection']['receiver_count']} "
                "deterministically hash-selected whole-node receivers. "
                "Selection did not use expression or TLS scores."
            ),
            "",
            "## Routing",
            "",
            (
                "Normalized attention is a computational routing quantity. "
                "The production value gate rescales that routing, so the "
                "reported effective coefficient is attention × value gate. "
                "Neither quantity is biological importance."
            ),
            "",
            routing_sentence,
            "",
            "## Prediction faithfulness",
            "",
            (
                "Deleting the top final-layer effective-routing edges changed "
                "predictions by mean absolute "
                f"{top['mean_absolute_prediction_change']:.6g}. The ratio to "
                "the mean deterministic distance-matched null change was "
                f"{ratio_text}."
            ),
            "",
            (
                "This is an isolated final-layer intervention. It does not "
                "test earlier-layer routes, graph-specific generalization, "
                "cell-cell communication, mechanism, or causality."
            ),
            "",
            "## TLS annotation",
            "",
            (
                f"Status: {report['tls_annotation']['status']}. TLS results "
                "are exploratory annotation only and are not independent "
                "validation."
            ),
            "",
        ]
    )


def analyze_qkv_interpretability(
    *,
    run: Path,
    capacity_report: Path,
    output: Path,
    device_name: str,
    receiver_count: int,
    deletion_fraction: float,
    null_replicates: int,
    allow_negative_gate_diagnostic: bool,
) -> dict[str, Any]:
    receiver_count = int(receiver_count)
    null_replicates = int(null_replicates)
    if not 1 <= receiver_count <= MAX_RECEIVERS:
        raise QKVInterpretabilityError(
            f"receiver_count must be in [1, {MAX_RECEIVERS}]"
        )
    if not 1 <= null_replicates <= MAX_NULL_REPLICATES:
        raise QKVInterpretabilityError(
            f"null_replicates must be in [1, {MAX_NULL_REPLICATES}]"
        )
    if not 0.0 < float(deletion_fraction) < 0.5:
        raise QKVInterpretabilityError(
            "deletion_fraction must be in (0, 0.5)"
        )
    root = run.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise QKVInterpretabilityError("Run must be a regular directory")
    symlinks = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise QKVInterpretabilityError(
            "Run bundle may not contain symlinks: " + ", ".join(symlinks)
        )
    try:
        verification = verify_run_bundle(root)
    except (RunValidationError, OSError, ValueError) as error:
        raise QKVInterpretabilityError(
            "Run bundle failed canonical verification"
        ) from error
    _expect(verification.get("status"), "success", "bundle verification status")

    config = _read_yaml(root / "config.resolved.yaml")
    manifest = _read_yaml(root / "manifest.yaml")
    summary = _read_json(root / "summary.json")
    training = _read_json(root / "provenance/full_core_training.json")
    inputs = _read_json(root / "provenance/full_core_inputs.json")
    mask_provenance = _read_json(
        root / "provenance/fixed_evaluation_masks.json"
    )
    _expect(
        mask_provenance.get("used_for_gradient_updates"),
        False,
        "mask provenance gradient use",
    )
    _expect(
        mask_provenance.get("used_for_checkpoint_selection"),
        False,
        "mask provenance checkpoint-selection use",
    )
    _validate_run_contract(root, config, manifest, summary, training)
    gate = _capacity_gate(
        capacity_report.resolve(strict=True),
        run_id=root.name,
        allow_negative=allow_negative_gate_diagnostic,
    )
    state, checkpoint = _load_verified_checkpoint(
        root,
        config,
        summary,
        training,
    )
    core = _rebuild_verified_core(config, inputs)
    graph = _rebuild_verified_graph(core, config, inputs, summary)
    whole_node_mask, mask_bundle = _rebuild_verified_whole_node_mask(
        core,
        config,
        mask_provenance,
        checkpoint,
        summary,
    )
    masked_receivers = np.flatnonzero(
        _whole_node_selected_rows(whole_node_mask)
    )
    selected_receivers = _deterministic_hash_select(
        masked_receivers,
        receiver_count,
        namespace=(
            "full-core-qkv-final-routing-v1:"
            f"{root.name}:{graph.checksums.graph_sha256}"
        ),
    )

    model = _instantiate_verified_model(state, training)
    del state
    graph_edges, graph_attributes = graph.concatenate()
    (
        selected_edges,
        selected_attributes,
        selected_distance_um,
    ) = _selected_edge_arrays(
        graph_edges,
        graph_attributes,
        selected_receivers,
        core.coordinates_um,
    )

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise QKVInterpretabilityError("CUDA was requested but is unavailable")
    model.to(device)
    expression = torch.from_numpy(
        np.asarray(core.target_expression, dtype=np.float32)
    ).to(device)
    gene_mask = torch.from_numpy(whole_node_mask).to(device)
    node_covariates = torch.from_numpy(
        np.asarray(core.node_covariates, dtype=np.float32)
    ).to(device)
    edge_index = torch.from_numpy(graph_edges).to(device)
    edge_attributes = torch.from_numpy(graph_attributes).to(device)
    trainer = _mapping(config.get("trainer"), "config.trainer")
    amp = bool(trainer.get("amp", False))
    amp_dtype = str(trainer.get("amp_dtype", "auto"))

    penultimate = _compute_penultimate(
        model,
        expression,
        gene_mask,
        node_covariates,
        edge_index,
        edge_attributes,
        amp=amp,
        amp_dtype=amp_dtype,
    )
    model.clear_edge_layout_cache()
    del expression, gene_mask, node_covariates, edge_index, edge_attributes
    del graph_edges, graph_attributes
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    selected_edges_device = torch.from_numpy(selected_edges).to(device)
    selected_attributes_device = torch.from_numpy(selected_attributes).to(
        device
    )
    selected_receivers_device = torch.from_numpy(selected_receivers).to(device)
    with torch.no_grad(), _autocast(
        enabled=amp,
        device=device,
        dtype_name=amp_dtype,
    ):
        components = _final_layer_components(
            model,
            penultimate,
            selected_edges_device,
            selected_attributes_device,
            selected_receivers_device,
        )
        baseline_prediction, _, _ = (
            _readout_with_deleted_edges(model, components)
        )

    attention = components["attention"].detach().float().cpu().numpy()
    gate_values = components["value_gate"].detach().float().cpu().numpy()
    local_receiver = (
        components["local_receiver"].detach().cpu().numpy().astype(np.int64)
    )
    effective = attention * gate_values
    effective_distribution, effective_mass = _normalize_effective(
        effective,
        local_receiver,
        receiver_count,
    )
    # The intervention ranks the literal production coefficient.  The
    # per-head-renormalized product is used only to calculate entropy and
    # distance summaries.
    routing_score = effective.mean(axis=1)
    plan = _distance_matched_deletion_plan(
        local_receiver,
        routing_score,
        selected_distance_um,
        receiver_count=receiver_count,
        fraction=float(deletion_fraction),
        null_replicates=int(null_replicates),
        namespace=(
            "full-core-qkv-distance-null-v1:"
            f"{root.name}:{graph.checksums.graph_sha256}"
        ),
    )

    truth = np.asarray(
        core.target_expression[selected_receivers],
        dtype=np.float64,
    )
    baseline = baseline_prediction.detach().float().cpu().numpy()
    huber_delta = float(trainer.get("huber_delta", 1.0))
    with torch.no_grad(), _autocast(
        enabled=amp,
        device=device,
        dtype_name=amp_dtype,
    ):
        top_prediction = _readout_with_deleted_edges(
            model,
            components,
            torch.from_numpy(plan["top_positions"]).to(device),
        )[0]
        null_predictions = [
            _readout_with_deleted_edges(
                model,
                components,
                torch.from_numpy(positions).to(device),
            )[0]
            for positions in plan["null_positions"]
        ]
    top_change = _prediction_change(
        baseline,
        top_prediction.detach().float().cpu().numpy(),
        truth,
        huber_delta=huber_delta,
    )
    null_changes = [
        _prediction_change(
            baseline,
            prediction.detach().float().cpu().numpy(),
            truth,
            huber_delta=huber_delta,
        )
        for prediction in null_predictions
    ]
    null_prediction_changes = np.asarray(
        [
            row["mean_absolute_prediction_change"]
            for row in null_changes
        ],
        dtype=np.float64,
    )
    null_huber_changes = np.asarray(
        [row["huber_change"] for row in null_changes],
        dtype=np.float64,
    )
    top_value = top_change["mean_absolute_prediction_change"]
    null_mean = float(null_prediction_changes.mean())
    ratio = float(top_value / null_mean) if null_mean > 0 else None
    top_exceeds_null_mean = bool(top_value > null_mean)
    tls_annotation = (
        _tls_annotation(
            core.gene_names,
            core.target_expression,
            selected_receivers,
            selected_edges[0],
            local_receiver,
            effective_distribution,
        )
        if gate["passes"]
        else {
            "status": "suppressed_negative_capacity_gate",
            "exploratory_annotation_only": True,
            "selection_used_tls_expression": False,
            "caution": (
                "Biological annotation is suppressed because the predictive "
                "representation gate failed. The override permits only a "
                "computational routing diagnostic."
            ),
        }
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "full_core_qkv_bounded_interpretability",
        "analysis_status": (
            "exploratory_capacity_gate_passed"
            if gate["passes"]
            else "negative_capacity_gate_diagnostic_only"
        ),
        "representation_gate": gate,
        "provenance": {
            "run_id": root.name,
            "campaign_id": CAMPAIGN_ID,
            "model_name": _mapping(config["model"], "config.model")["name"],
            "model_state_dict_sha256": training["state_dict_sha256"],
            "materialized_preprocessing_sha256": (
                core.checksums.preprocessing_sha256
            ),
            "materialized_graph_sha256": graph.checksums.graph_sha256,
            "fixed_mask_bundle_sha256": mask_bundle.checksum,
            "bundle_checksums_verified": True,
            "checkpoint_reconstructed_from_recorded_implementation_args": True,
            "device": str(device),
        },
        "scope": {
            "estimand": (
                "final-layer routing and isolated final-layer prediction "
                "faithfulness under the canonical held-in whole-node mask"
            ),
            "generalization_estimate": False,
            "single_transductively_fitted_core": True,
            "attention_is_biological_importance": False,
            "maximum_claim": (
                (
                    "descriptive final-layer computational routing and "
                    "prediction response within this fitted model"
                )
                if gate["passes"]
                else "negative-capacity-gate computational diagnostic only"
            ),
        },
        "selection": {
            "receiver_count": receiver_count,
            "eligible_whole_node_receiver_count": int(len(masked_receivers)),
            "selection_method": (
                "deterministic SHA-256 rank over node positions; independent "
                "of expression, routing, error, phenotype, and TLS scores"
            ),
            "mask_mode": "whole_node",
            "mask_replicate": 0,
        },
        "routing": {
            "coefficient_definition": (
                "effective_routing_coefficient = "
                "normalized_attention * bounded_value_gate"
            ),
            "normalized_attention": _routing_summary(
                local_receiver,
                attention,
                selected_distance_um,
                receiver_count=receiver_count,
            ),
            "bounded_value_gate": {
                "theoretical_range": "(0, 2)",
                "neutral_value": 1.0,
                "values": _describe(gate_values),
            },
            "effective_routing": {
                **_routing_summary(
                    local_receiver,
                    effective_distribution,
                    selected_distance_um,
                    receiver_count=receiver_count,
                ),
                "pre_normalization_mass_per_receiver_head": _describe(
                    effective_mass
                ),
                "summary_distribution": (
                    "attention*gate normalized over incoming edges separately "
                    "for each receiver and head"
                ),
            },
            "caution": (
                "Attention and attention×gate are computational routing "
                "coefficients, not signed value contributions, biological "
                "importance, interaction strength, mechanism, or causality."
            ),
        },
        "faithfulness": {
            "intervention": (
                "delete incoming edges only in the final QKV block, "
                "recompute receiver-wise softmax on retained logits, and "
                "decode predictions"
            ),
            "edge_ranking": (
                "descending mean across heads of the literal "
                "attention*value_gate coefficient"
            ),
            "isolated_final_layer_test": True,
            "full_network_edge_deletion_test": False,
            "deletion_fraction_per_receiver": float(deletion_fraction),
            "deleted_edge_count_per_receiver": _describe(
                plan["deleted_count_per_receiver"]
            ),
            "distance_matching": {
                "method": (
                    "exact per-receiver rank-distance-bin count matching"
                ),
                "bin_count_per_receiver": _describe(
                    plan["distance_bin_count_per_receiver"]
                ),
                "top_mean_distance_um": plan["top_mean_distance_um"],
                "null_mean_distance_absolute_difference_um": _describe(
                    plan["null_mean_distance_absolute_difference_um"]
                ),
            },
            "top_effective_routing_deletion": top_change,
            "distance_matched_null_deletions": {
                "replicate_count": len(null_changes),
                "mean_absolute_prediction_change": _describe(
                    null_prediction_changes
                ),
                "huber_change": _describe(null_huber_changes),
            },
            "top_vs_matched_null": {
                "prediction_change_ratio_to_null_mean": ratio,
                "prediction_change_ratio_status": (
                    "finite"
                    if ratio is not None
                    else "undefined_zero_null_mean"
                ),
                "prediction_change_difference_from_null_mean": float(
                    top_value - null_mean
                ),
                "top_change_exceeds_null_mean": top_exceeds_null_mean,
                "null_replicates_at_least_as_large_as_top": int(
                    np.sum(null_prediction_changes >= top_value)
                ),
                "top_exceeds_every_null": bool(
                    np.all(top_value > null_prediction_changes)
                ),
                "descriptive_only": True,
            },
            "caution": (
                "This tests whether a final-layer ranking is locally faithful "
                "to final-layer predictions. It does not test routes in "
                "earlier layers or establish a biological dependency."
            ),
        },
        "tls_annotation": tls_annotation,
        "limitations": [
            "one fitted core with no validation or test partition",
            "hash-selected cells are technical observations, not replicates",
            "only the final graph layer is intervened on",
            "distance matching does not control cell state or segmentation",
            "no patient replication, graph randomization, or external evidence",
        ],
    }
    _assert_aggregate_only(report)

    destination = output.resolve(strict=False)
    if destination.exists():
        raise QKVInterpretabilityError(
            f"Output path already exists and will not be replaced: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    try:
        (destination / "analysis.json").write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "report.md").write_text(
            _markdown_report(report),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(destination)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = analyze_qkv_interpretability(
        run=arguments.run,
        capacity_report=arguments.capacity_report,
        output=arguments.output,
        device_name=arguments.device,
        receiver_count=arguments.receiver_count,
        deletion_fraction=arguments.deletion_fraction,
        null_replicates=arguments.null_replicates,
        allow_negative_gate_diagnostic=(
            arguments.allow_negative_gate_diagnostic
        ),
    )
    print(
        json.dumps(
            {
                "status": result["analysis_status"],
                "run_id": result["provenance"]["run_id"],
                "output": str(arguments.output),
                "representation_gate_passes": result[
                    "representation_gate"
                ]["passes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
