#!/usr/bin/env python3
"""Run one worker-owned, no-holdout full-core capacity experiment.

The queue worker owns the run root, resolved configuration, generic
provenance, logs, manifest, registry transitions, and finalization.  This
subprocess attaches to that active archive and writes only scientific outputs:
the final checkpoint, fit metrics and predictions, full-core/graph QC, and
run-specific provenance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BOOTSTRAP_ROOT / "src"))

from spatial_benchmark.configuration import (  # noqa: E402
    load_yaml_mapping,
    validate_experiment_config,
)
from spatial_benchmark.dense_gat import (  # noqa: E402
    ReceiverChunkedEdgeConditionedGATv2,
)
from spatial_benchmark.expression_tokens import (  # noqa: E402
    DEFAULT_COUNT_TOKEN_SPEC,
    audit_expression_tokens,
    fit_per_gene_modal_tokens,
    tokenize_expression_counts,
)
from spatial_benchmark.full_core import (  # noqa: E402
    FullCoreData,
    ReceiverSortedGraph,
    build_exact_mutual_knn_graph,
    load_and_refit_full_core,
)
from spatial_benchmark.full_core_training import (  # noqa: E402
    FullCoreTrainingResult,
    fit_full_core_model,
)
from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.masking import (  # noqa: E402
    FixedMaskBundle,
    MaskSpec,
    create_fixed_mask_bundle,
    derive_mask_seed,
)
from spatial_benchmark.models import (  # noqa: E402
    EdgeParameterMatchedSelfControl,
)
from spatial_benchmark.paths import current_paths  # noqa: E402
from spatial_benchmark.qkv_graph_transformer import (  # noqa: E402
    ReceiverChunkedEdgeAwareQKVGraphTransformer,
)
from spatial_benchmark.qkv_self_control import (  # noqa: E402
    QKVParameterMatchedSelfControl,
)
from spatial_benchmark.run_archive import (  # noqa: E402
    RunArchive,
    RunValidationError,
    deidentify_prediction_rows,
)
from spatial_benchmark.training import (  # noqa: E402
    GraphSplitView,
    TrainingConfig,
    evaluate_fixed_mask,
    evaluate_fixed_token_mask,
    set_deterministic_seed,
)
from spatial_benchmark.tokenized_g2 import (  # noqa: E402
    TokenizedReceiverChunkedEdgeConditionedGATv2,
)


_PROTOCOL = "held_in_full_core_fixed_budget"
_REGRESSION_TASK = "masked_expression_regression"
_TOKEN_TASK = "masked_expression_token_classification"
_TOKEN_PRIMARY_METRIC = "fit/whole_node/masked_token_accuracy_percent"
_TOKENIZER_SCHEMA = "raw_count_tokens_0_1_2_3plus_v1"
_TOKEN_MODAL_BASELINE_SCOPE = "all_fit_transductive"
_PUBLIC_MASK_NAMES = {
    "partial": "partial_gene",
    "node": "whole_node",
    "block": "spatial_block",
}
_REQUIRED_PUBLIC_MASKS = (
    "partial_gene",
    "whole_node",
    "spatial_block",
)
_MATCHED_SELF_KEYS = {
    "b0g2matched",
    "g2matchedself",
    "edgeparametermatchedself",
}
_G2_KEYS = {"g2", "edgegat", "edgeconditionedgatv2"}
_TOKEN_G2_KEYS = {"g2tokenized", "tokenizededgeconditionedgatv2"}
_QKV_GAT_KEYS = {"qkvgat"}
_QKV_MATCHED_SELF_KEYS = {"qkvgatmatchedself"}
_GRAPH_MODEL_KEYS = _G2_KEYS | _TOKEN_G2_KEYS | _QKV_GAT_KEYS


class FullCoreRunnerError(RuntimeError):
    """Raised when a run would violate the held-in capacity contract."""


@dataclass(frozen=True)
class CapacityRunResult:
    """Small in-memory handoff after all runner-owned artifacts are written."""

    run_id: str
    model_name: str
    primary_metric_name: str
    primary_metric_value: float
    final_epoch: int
    checkpoint_path: Path
    prediction_path: Path
    summary: Mapping[str, Any]


def _section(
    config: Mapping[str, Any], name: str
) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise FullCoreRunnerError(
            f"resolved configuration requires a {name!r} mapping"
        )
    return value


def _model_key(value: object) -> str:
    return "".join(
        character
        for character in str(value).strip().lower()
        if character.isalnum()
    )


def _mask_rates(masking: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("rates", "rate"):
        value = masking.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _validate_capacity_contract(config: Mapping[str, Any]) -> str:
    evaluation = _section(config, "evaluation")
    trainer = _section(config, "trainer")
    graph = _section(config, "graph")
    model = _section(config, "model")
    dataset = _section(config, "dataset")
    features = _section(config, "features")
    task_family = str(evaluation.get("task_family", "")).strip().lower()
    if task_family not in {_REGRESSION_TASK, _TOKEN_TASK}:
        raise FullCoreRunnerError(
            "evaluation.task_family must select masked expression regression "
            "or token classification"
        )
    token_task = task_family == _TOKEN_TASK
    key = _model_key(model.get("name"))
    if token_task:
        if key not in _TOKEN_G2_KEYS:
            raise FullCoreRunnerError(
                "token classification requires model.name='g2-tokenized'"
            )
        if str(dataset.get("task", "")).strip().lower() != _TOKEN_TASK:
            raise FullCoreRunnerError(
                "token classification requires a matching dataset.task"
            )
        if model.get("tokenizer_schema") != _TOKENIZER_SCHEMA:
            raise FullCoreRunnerError(
                f"model.tokenizer_schema must be {_TOKENIZER_SCHEMA!r}"
            )
        if int(model.get("num_expression_tokens", 0)) != 4:
            raise FullCoreRunnerError(
                "the locked raw-count vocabulary has four output tokens"
            )
        tokenization = dataset.get("tokenization")
        if not isinstance(tokenization, Mapping):
            raise FullCoreRunnerError(
                "token classification requires dataset.tokenization"
            )
        if (
            tokenization.get("schema") != _TOKENIZER_SCHEMA
            or int(tokenization.get("num_output_tokens", 0)) != 4
            or int(tokenization.get("mask_token_id", -1)) != 4
        ):
            raise FullCoreRunnerError(
                "dataset.tokenization does not match the locked 0/1/2/3+ "
                "vocabulary with input-only MASK token 4"
            )
    else:
        if key in _TOKEN_G2_KEYS:
            raise FullCoreRunnerError(
                "g2-tokenized requires token-classification evaluation"
            )
        if str(dataset.get("task", "")).strip().lower() != _REGRESSION_TASK:
            raise FullCoreRunnerError(
                "regression evaluation requires a matching dataset.task"
            )

    if str(evaluation.get("protocol", "")).strip().lower() != _PROTOCOL:
        raise FullCoreRunnerError(
            f"evaluation.protocol must be {_PROTOCOL!r}"
        )
    if str(evaluation.get("canonical_prediction_split", "")).lower() != "fit":
        raise FullCoreRunnerError(
            "held-in full-core evaluation requires canonical split 'fit'"
        )
    if list(evaluation.get("splits", ())) != ["fit"]:
        raise FullCoreRunnerError(
            "held-in full-core evaluation must declare only the fit role"
        )
    if list(evaluation.get("mask_modes", ())) != list(
        _REQUIRED_PUBLIC_MASKS
    ):
        raise FullCoreRunnerError(
            "evaluation.mask_modes must be partial_gene, whole_node, "
            "spatial_block in that order"
        )
    mask_replicates = int(
        evaluation.get("mask_replicates_per_mode", 0)
    )
    diagnostic_resource_pilot = (
        trainer.get("diagnostic_resource_pilot") is True
    )
    if diagnostic_resource_pilot:
        if (
            mask_replicates != 1
            or evaluation.get("diagnostic_only") is not True
            or evaluation.get("conclusion_bearing") is not False
        ):
            raise FullCoreRunnerError(
                "a diagnostic resource pilot requires exactly one mask "
                "replicate, evaluation.diagnostic_only=true, and "
                "evaluation.conclusion_bearing=false"
            )
    elif mask_replicates != 3:
        raise FullCoreRunnerError(
            "conclusion-bearing full-core evaluation requires exactly three "
            "mask replicates"
        )
    expected_primary = (
        _TOKEN_PRIMARY_METRIC
        if token_task
        else "fit/whole_node/masked_huber"
    )
    if evaluation.get("primary_metric") != expected_primary:
        raise FullCoreRunnerError(
            "full-core capacity primary metric must be "
            f"{expected_primary}"
        )
    if bool(dataset.get("validation_or_test_partition_present", True)):
        raise FullCoreRunnerError(
            "full-core capacity data must explicitly contain no validation "
            "or test partition"
        )
    if str(dataset.get("preprocessing_fit_scope", "")).lower() != (
        "all_nodes_transductive"
    ):
        raise FullCoreRunnerError(
            "dataset preprocessing_fit_scope must be all_nodes_transductive"
        )

    if trainer.get("fixed_epoch_budget") is not True:
        raise FullCoreRunnerError("trainer.fixed_epoch_budget must be true")
    if trainer.get("restore_best") is not False:
        raise FullCoreRunnerError("trainer.restore_best must be false")
    if str(trainer.get("primary_checkpoint_role", "")).lower() != "last":
        raise FullCoreRunnerError(
            "trainer.primary_checkpoint_role must be last"
        )
    if str(trainer.get("checkpoint_policy", "")).lower() != "last_only":
        raise FullCoreRunnerError("trainer.checkpoint_policy must be last_only")
    if int(trainer.get("max_epochs", 0)) <= 0:
        raise FullCoreRunnerError("trainer.max_epochs must be positive")
    if bool(trainer.get("neighbor_sampling", True)):
        raise FullCoreRunnerError("neighbor sampling is prohibited")
    if token_task and trainer.get("objective") != (
        "unweighted_masked_categorical_cross_entropy"
    ):
        raise FullCoreRunnerError(
            "token training requires unweighted masked categorical "
            "cross-entropy"
        )
    if not token_task and trainer.get("objective") not in {
        None,
        "masked_huber",
    }:
        raise FullCoreRunnerError(
            "regression training does not accept a categorical objective"
        )

    if str(graph.get("kind", "")).lower() != (
        "exact_spatial_knn_radius_guard"
    ):
        raise FullCoreRunnerError(
            "graph.kind must request exact kNN with a radius guard"
        )
    if str(graph.get("symmetry", "")).lower() != "mutual":
        raise FullCoreRunnerError("the high-k graph must use mutual symmetry")
    if graph.get("full_core_graph") is not True:
        raise FullCoreRunnerError("graph.full_core_graph must be true")
    if bool(graph.get("neighbor_sampling", False)):
        raise FullCoreRunnerError("graph neighbor sampling is prohibited")
    if float(graph.get("edge_dropout", -1.0)) != 0.0:
        raise FullCoreRunnerError(
            "exact high-k capacity training requires edge_dropout=0"
        )
    if int(graph.get("k", graph.get("neighbor_k", 0))) <= 0:
        raise FullCoreRunnerError("graph k must be positive")

    if key not in (
        _GRAPH_MODEL_KEYS | _MATCHED_SELF_KEYS | _QKV_MATCHED_SELF_KEYS
    ):
        raise FullCoreRunnerError(
            "runner supports only exact G2, exact QKV-GAT, or the "
            "corresponding parameter-matched single-cell controls"
        )
    if key in _GRAPH_MODEL_KEYS:
        if features.get("use_edge_features") is not True:
            raise FullCoreRunnerError(
                f"{model.get('name')} requires measured edge features"
            )
        if model.get("exact_receiver_partitioning") is not True:
            raise FullCoreRunnerError(
                f"{model.get('name')} must explicitly enable exact "
                "receiver partitioning"
            )
        if int(model.get("receiver_chunk_size", 0)) <= 0:
            raise FullCoreRunnerError(
                "graph models require a positive receiver_chunk_size"
            )
        if key in _QKV_GAT_KEYS:
            edge_conditioning_mode = str(
                model.get("edge_conditioning_mode", "")
            ).strip()
            if edge_conditioning_mode not in {"bias_gate", "vector"}:
                raise FullCoreRunnerError(
                    "qkv-gat edge_conditioning_mode must be 'bias_gate' "
                    "or 'vector'"
                )
            if model.get("activation_checkpointing") is not True:
                raise FullCoreRunnerError(
                    "qkv-gat full-core training requires "
                    "activation_checkpointing=true"
                )
            max_edges_per_chunk = model.get("max_edges_per_chunk")
            if (
                max_edges_per_chunk is None
                or isinstance(max_edges_per_chunk, bool)
                or not isinstance(max_edges_per_chunk, int)
                or max_edges_per_chunk <= 0
            ):
                raise FullCoreRunnerError(
                    "qkv-gat full-core training requires a positive "
                    "max_edges_per_chunk"
                )
    else:
        if features.get("use_edge_features") is not False:
            raise FullCoreRunnerError(
                "a matched single-cell control must not consume edge features"
            )
        if key in _QKV_MATCHED_SELF_KEYS:
            edge_conditioning_mode = str(
                model.get("edge_conditioning_mode", "")
            ).strip()
            if edge_conditioning_mode not in {"bias_gate", "vector"}:
                raise FullCoreRunnerError(
                    "qkv-gat-matched-self edge_conditioning_mode must be "
                    "'bias_gate' or 'vector'"
                )
    return key


def _resolve_prepared_artifact(
    config: Mapping[str, Any], archive: RunArchive
) -> Path:
    dataset = _section(config, "dataset")
    value = dataset.get("prepared_artifact_reference")
    if not isinstance(value, str) or not value.strip():
        raise FullCoreRunnerError(
            "dataset.prepared_artifact_reference is required"
        )
    path = Path(value)
    if not path.is_absolute():
        path = archive.paths.project_root / path
    return path


def _build_graph(
    core: FullCoreData,
    graph_config: Mapping[str, Any],
) -> ReceiverSortedGraph:
    return build_exact_mutual_knn_graph(
        core.coordinates_um,
        k=int(graph_config.get("k", graph_config.get("neighbor_k", 0))),
        radius_guard_um=float(
            graph_config.get(
                "radius_guard_um", graph_config.get("radius_um", 0.0)
            )
        ),
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


def _verify_materialized_identity(
    core: FullCoreData,
    dataset: Mapping[str, Any],
) -> Mapping[str, Any]:
    configured_fingerprint = str(
        dataset.get("dataset_fingerprint", "")
    )
    if configured_fingerprint != core.checksums.preprocessing_sha256:
        raise FullCoreRunnerError(
            "dataset.dataset_fingerprint does not match the materialized "
            "full-core preprocessing checksum"
        )

    fingerprint_basis = dataset.get("dataset_fingerprint_basis")
    if not isinstance(fingerprint_basis, Mapping):
        raise FullCoreRunnerError(
            "dataset.dataset_fingerprint_basis is required"
        )
    declared_source_fields = {
        "source_artifact_id": core.checksums.source_artifact_id,
        "source_prepared_data_sha256": (
            core.checksums.source_prepared_data_sha256
        ),
        "materialized_preprocessing_sha256": (
            core.checksums.preprocessing_sha256
        ),
    }
    for field, expected in declared_source_fields.items():
        if field in fingerprint_basis and str(
            fingerprint_basis[field]
        ) != str(expected):
            raise FullCoreRunnerError(
                f"dataset.dataset_fingerprint_basis.{field} does not "
                "match the materialized full core"
            )

    experimental_unit = str(dataset.get("experimental_unit", ""))
    if not experimental_unit:
        raise FullCoreRunnerError(
            "dataset.experimental_unit is required for split identity"
        )
    expected_split_basis = {
        "schema": "full_core_no_holdout_roles_v1",
        "dataset_fingerprint": core.checksums.preprocessing_sha256,
        "n_nodes": core.n_nodes,
        "role_assignment": (
            "every row in verified prepared artifact order assigned fit"
        ),
        "role_counts": {
            "fit": core.n_nodes,
            "validation": 0,
            "test": 0,
        },
        "experimental_unit": experimental_unit,
    }
    configured_split_basis = dataset.get("split_fingerprint_basis")
    if not isinstance(configured_split_basis, Mapping) or dict(
        configured_split_basis
    ) != expected_split_basis:
        raise FullCoreRunnerError(
            "dataset.split_fingerprint_basis does not match the materialized "
            "all-fit role assignment"
        )
    materialized_split_fingerprint = canonical_sha256(
        expected_split_basis
    )
    if str(dataset.get("split_fingerprint", "")) != (
        materialized_split_fingerprint
    ):
        raise FullCoreRunnerError(
            "dataset.split_fingerprint does not match the recomputed "
            "materialized all-fit split"
        )
    split_id = str(dataset.get("split_id", ""))
    if split_id != materialized_split_fingerprint[:16]:
        raise FullCoreRunnerError(
            "dataset.split_id is not the canonical split fingerprint prefix"
        )
    return {
        "dataset_fingerprint": configured_fingerprint,
        "split_fingerprint": materialized_split_fingerprint,
        "split_fingerprint_basis": expected_split_basis,
        "source_identity_fields_checked": sorted(
            field
            for field in declared_source_fields
            if field in fingerprint_basis
        ),
    }


def _verify_graph_identity(
    graph: ReceiverSortedGraph,
    graph_config: Mapping[str, Any],
) -> None:
    expected_checksum = graph_config.get(
        "expected_materialized_graph_sha256"
    )
    expected_edge_count = graph_config.get("expected_directed_edges")
    if not isinstance(expected_checksum, str) or len(
        expected_checksum
    ) != 64:
        raise FullCoreRunnerError(
            "graph.expected_materialized_graph_sha256 is required"
        )
    if (
        isinstance(expected_edge_count, bool)
        or not isinstance(expected_edge_count, int)
        or expected_edge_count < 0
    ):
        raise FullCoreRunnerError(
            "graph.expected_directed_edges must be a non-negative integer"
        )
    if graph.checksums.graph_sha256 != expected_checksum:
        raise FullCoreRunnerError(
            "constructed graph checksum does not match the verified "
            "materialized graph"
        )
    if graph.qc.n_directed_edges != expected_edge_count:
        raise FullCoreRunnerError(
            "constructed graph edge count does not match the verified "
            "materialized graph"
        )


def _model_dimensions(
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    dimensions: dict[str, Any] = {
        "hidden_dim": int(model_config.get("hidden_dim", 512)),
        "attention_heads": int(model_config.get("attention_heads", 4)),
        "graph_layers": int(model_config.get("graph_layers", 2)),
        "ffn_dim": int(model_config.get("ffn_dim", 512)),
        "decoder_dim": int(model_config.get("decoder_dim", 512)),
        "edge_hidden_dim": int(model_config.get("edge_hidden_dim", 64)),
        "edge_embedding_dim": int(
            model_config.get("edge_embedding_dim", 64)
        ),
        "dropout": float(model_config.get("dropout", 0.1)),
        "attention_dropout": float(
            model_config.get("attention_dropout", 0.1)
        ),
    }
    dimensions["attention_head_dim"] = (
        None
        if model_config.get("attention_head_dim") is None
        else int(model_config["attention_head_dim"])
    )
    return dimensions


def _model_constructor(
    *,
    model_key: str,
    core: FullCoreData,
    graph: ReceiverSortedGraph,
    model_config: Mapping[str, Any],
) -> tuple[type[torch.nn.Module], dict[str, Any]]:
    """Resolve an explicit implementation class and complete constructor."""

    common = {
        "num_genes": core.n_genes,
        "node_covariate_dim": int(core.node_covariates.shape[1]),
        "edge_attribute_dim": len(graph.edge_attribute_names),
        **_model_dimensions(model_config),
    }
    if model_key in _TOKEN_G2_KEYS:
        return TokenizedReceiverChunkedEdgeConditionedGATv2, {
            **common,
            "num_expression_tokens": int(
                model_config.get("num_expression_tokens", 0)
            ),
            "receiver_chunk_size": int(
                model_config.get("receiver_chunk_size", 512)
            ),
            "activation_checkpointing": bool(
                model_config.get("activation_checkpointing", True)
            ),
        }
    if model_key in _G2_KEYS:
        return ReceiverChunkedEdgeConditionedGATv2, {
            **common,
            "receiver_chunk_size": int(
                model_config.get("receiver_chunk_size", 512)
            ),
            "activation_checkpointing": bool(
                model_config.get("activation_checkpointing", True)
            ),
        }
    if model_key in _QKV_GAT_KEYS:
        max_edges_per_chunk = model_config.get("max_edges_per_chunk")
        return ReceiverChunkedEdgeAwareQKVGraphTransformer, {
            **common,
            "edge_conditioning_mode": str(
                model_config.get("edge_conditioning_mode", "bias_gate")
            ),
            "receiver_chunk_size": int(
                model_config.get("receiver_chunk_size", 256)
            ),
            "max_edges_per_chunk": (
                None
                if max_edges_per_chunk is None
                else int(max_edges_per_chunk)
            ),
            "activation_checkpointing": bool(
                model_config.get("activation_checkpointing", True)
            ),
        }
    if model_key in _QKV_MATCHED_SELF_KEYS:
        # Receiver partitioning, edge chunk limits, and activation
        # checkpointing are graph-execution choices.  They do not define the
        # cell-autonomous matched control and are intentionally not forwarded.
        return QKVParameterMatchedSelfControl, {
            **common,
            "edge_conditioning_mode": str(
                model_config.get("edge_conditioning_mode", "bias_gate")
            ),
        }
    return EdgeParameterMatchedSelfControl, common


def _model_construction_record(
    *,
    model_key: str,
    core: FullCoreData,
    graph: ReceiverSortedGraph,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return sufficient implementation metadata to recreate the model."""

    implementation, constructor_arguments = _model_constructor(
        model_key=model_key,
        core=core,
        graph=graph,
        model_config=model_config,
    )
    return {
        "canonical_model_key": model_key,
        "implementation_class": (
            f"{implementation.__module__}.{implementation.__qualname__}"
        ),
        "constructor_arguments": constructor_arguments,
    }


def _make_model(
    *,
    model_key: str,
    core: FullCoreData,
    graph: ReceiverSortedGraph,
    model_config: Mapping[str, Any],
) -> torch.nn.Module:
    implementation, constructor_arguments = _model_constructor(
        model_key=model_key,
        core=core,
        graph=graph,
        model_config=model_config,
    )
    return implementation(**constructor_arguments)


def _fit_view(
    *,
    core: FullCoreData,
    graph: ReceiverSortedGraph,
    uses_graph: bool,
    expression: np.ndarray | None = None,
) -> GraphSplitView:
    if uses_graph:
        edge_index, edge_attributes = graph.concatenate()
        torch_edges = torch.from_numpy(
            np.asarray(edge_index, dtype=np.int64)
        )
        torch_attributes: torch.Tensor | None = torch.from_numpy(
            np.asarray(edge_attributes, dtype=np.float32)
        )
    else:
        # The paired graph is still constructed and audited, but a strict
        # single-cell control neither consumes nor transfers it to the GPU.
        torch_edges = torch.empty((2, 0), dtype=torch.long)
        torch_attributes = None
    return GraphSplitView(
        expression=torch.from_numpy(
            np.asarray(
                core.target_expression if expression is None else expression,
                dtype=np.float32,
            )
        ),
        coordinates_um=torch.from_numpy(
            np.asarray(core.coordinates_um, dtype=np.float64)
        ),
        edge_index=torch_edges,
        edge_attributes=torch_attributes,
        node_covariates=torch.from_numpy(
            np.asarray(core.node_covariates, dtype=np.float32)
        ),
        block_ids=np.asarray(core.macroblock_ids),
        name="fit",
    )


def _training_config(
    config: Mapping[str, Any],
) -> TrainingConfig:
    trainer = _section(config, "trainer")
    graph = _section(config, "graph")
    masking = _section(config, "masking")
    rates = _mask_rates(masking)
    requested_device = trainer.get("device")
    if requested_device is None:
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    return TrainingConfig(
        max_epochs=int(trainer["max_epochs"]),
        learning_rate=float(trainer.get("learning_rate", 3e-4)),
        weight_decay=float(trainer.get("weight_decay", 1e-4)),
        gradient_clip_norm=float(
            trainer.get("gradient_clip_norm", 1.0)
        ),
        huber_delta=float(trainer.get("huber_delta", 1.0)),
        patience=0,
        min_delta=0.0,
        curriculum=str(masking.get("curriculum", "P+N+B")),
        warmup_epochs=int(masking.get("warmup_epochs", 10)),
        partial_gene_rate=float(
            masking.get(
                "partial_gene_rate", rates.get("partial_gene", 0.2)
            )
        ),
        node_rate=float(
            masking.get("whole_node_rate", rates.get("whole_node", 0.1))
        ),
        block_node_rate=float(
            masking.get(
                "block_node_rate", rates.get("spatial_block", 0.1)
            )
        ),
        block_width_um=masking.get("block_width_um"),
        block_shape=str(masking.get("block_shape", "disk")),
        mask_seed=int(masking.get("mask_seed", 0)),
        model_seed=int(config.get("seed", 0)),
        edge_dropout=float(graph.get("edge_dropout", 0.0)),
        amp=bool(trainer.get("amp", False)),
        amp_dtype=str(trainer.get("amp_dtype", "auto")),
        deterministic=bool(trainer.get("deterministic", True)),
        deterministic_warn_only=bool(
            trainer.get("deterministic_warn_only", False)
        ),
        device=str(requested_device),
        restore_best=False,
    )


def _evaluation_masks(
    config: Mapping[str, Any],
    core: FullCoreData,
    training: TrainingConfig,
) -> FixedMaskBundle:
    evaluation = _section(config, "evaluation")
    dataset = _section(config, "dataset")
    seed = derive_mask_seed(
        training.mask_seed,
        "held-in-full-core-fixed-evaluation",
        dataset.get("dataset_id"),
        dataset.get("version"),
        dataset.get("split_fingerprint"),
    )
    specs = [
        MaskSpec(
            mode="partial",
            partial_gene_rate=training.partial_gene_rate,
            node_rate=training.node_rate,
            block_node_rate=training.block_node_rate,
            block_width_um=training.block_width_um,
            block_shape=training.block_shape,
            label="partial_gene",
        ),
        MaskSpec(
            mode="node",
            partial_gene_rate=training.partial_gene_rate,
            node_rate=training.node_rate,
            block_node_rate=training.block_node_rate,
            block_width_um=training.block_width_um,
            block_shape=training.block_shape,
            label="whole_node",
        ),
        MaskSpec(
            mode="block",
            partial_gene_rate=training.partial_gene_rate,
            node_rate=training.node_rate,
            block_node_rate=training.block_node_rate,
            block_width_um=training.block_width_um,
            block_shape=training.block_shape,
            label="spatial_block",
        ),
    ]
    bundle = create_fixed_mask_bundle(
        {"fit": core.coordinates_um},
        core.n_genes,
        specs,
        replicates=int(evaluation["mask_replicates_per_mode"]),
        base_seed=seed,
    )
    expected_entries = 3 * int(
        evaluation["mask_replicates_per_mode"]
    )
    if len(bundle.manifest["entries"]) != expected_entries:
        raise FullCoreRunnerError(
            "fixed held-in evaluation produced the wrong number of masks"
        )
    return bundle


def _checkpoint_bytes(
    *,
    archive: RunArchive,
    model_name: str,
    model_config: Mapping[str, Any],
    model_construction: Mapping[str, Any],
    training: FullCoreTrainingResult,
    graph: ReceiverSortedGraph,
    core: FullCoreData,
    masks: FixedMaskBundle,
    task_family: str = _REGRESSION_TASK,
    tokenization_provenance: Mapping[str, Any] | None = None,
) -> bytes:
    payload = {
        "schema_version": 1,
        "run_id": archive.run_id,
        "checkpoint_role": "last",
        "checkpoint_policy": "final_epoch_no_validation_selection",
        "training_protocol": _PROTOCOL,
        "model_name": model_name,
        "model_config": dict(model_config),
        "model_construction": dict(model_construction),
        "epoch": training.final_epoch,
        "fixed_epoch_budget": training.fixed_epoch_budget,
        "model_state_dict": dict(training.final_state_dict),
        "state_dict_sha256": training.final_state_checksum,
        "full_core_preprocessing_sha256": (
            core.checksums.preprocessing_sha256
        ),
        "graph_sha256": graph.checksums.graph_sha256,
        "evaluation_mask_bundle_sha256": masks.checksum,
        "task_family": task_family,
    }
    if tokenization_provenance is not None:
        payload["tokenization"] = dict(tokenization_provenance)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _replicate_metric_row(
    *,
    entry: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    metrics = result.metrics
    gene = metrics["gene"]
    cell = metrics["cell"]
    internal_mode = str(entry["spec"]["mode"])
    return {
        "split": "fit",
        "mask_mode": _PUBLIC_MASK_NAMES[internal_mode],
        "mask_replicate": int(entry["replicate"]),
        "mask_entry_id": str(entry["entry_id"]),
        "mask_seed": int(entry["seed"]),
        "mask_checksum": str(entry["mask_checksum"]),
        "n_masked": int(metrics["n_masked"]),
        "masked_huber": float(metrics["huber"]),
        "masked_mse": float(metrics["mse"]),
        "masked_mae": float(metrics["mae"]),
        "masked_r2": _finite_or_none(metrics["r2"]),
        "masked_percent_variance_explained": _finite_or_none(
            metrics["percent_variance_explained"]
        ),
        "gene_mean_pearson": _finite_or_none(gene["mean_pearson"]),
        "gene_median_pearson": _finite_or_none(
            gene["median_pearson"]
        ),
        "gene_mean_spearman": _finite_or_none(
            gene["mean_spearman"]
        ),
        "gene_median_spearman": _finite_or_none(
            gene["median_spearman"]
        ),
        "cell_mean_pearson": _finite_or_none(cell["mean_pearson"]),
        "cell_median_pearson": _finite_or_none(
            cell["median_pearson"]
        ),
        "cell_mean_spearman": _finite_or_none(
            cell["mean_spearman"]
        ),
        "cell_median_spearman": _finite_or_none(
            cell["median_spearman"]
        ),
    }


def _token_replicate_metric_row(
    *,
    entry: Mapping[str, Any],
    result: Any,
) -> dict[str, Any]:
    metrics = result.metrics
    internal_mode = str(entry["spec"]["mode"])
    supports = list(metrics["per_token_support"])
    recalls = list(metrics["per_token_recall_percent"])
    confusion = list(metrics["confusion_matrix"])
    if len(supports) != 4 or len(recalls) != 4 or len(confusion) != 4:
        raise FullCoreRunnerError(
            "token metrics do not match the locked four-class vocabulary"
        )
    if any(len(list(row)) != 4 for row in confusion):
        raise FullCoreRunnerError("token confusion matrix must be 4 by 4")
    row: dict[str, Any] = {
        "split": "fit",
        "mask_mode": _PUBLIC_MASK_NAMES[internal_mode],
        "mask_replicate": int(entry["replicate"]),
        "mask_entry_id": str(entry["entry_id"]),
        "mask_seed": int(entry["seed"]),
        "mask_checksum": str(entry["mask_checksum"]),
        "n_masked": int(metrics["n_masked"]),
        "masked_token_cross_entropy": float(metrics["cross_entropy"]),
        "masked_token_accuracy_percent": float(
            metrics["accuracy_percent"]
        ),
        "masked_token_balanced_accuracy_percent": float(
            metrics["balanced_accuracy_percent"]
        ),
        "masked_nonzero_token_accuracy_percent": _finite_or_none(
            metrics["nonzero_accuracy_percent"]
        ),
        "baseline_uniform_accuracy_percent": float(
            metrics["uniform_chance_accuracy_percent"]
        ),
        "baseline_empirical_frequency_accuracy_percent": float(
            metrics["empirical_frequency_random_accuracy_percent"]
        ),
        "baseline_always_zero_accuracy_percent": float(
            metrics["always_zero_accuracy_percent"]
        ),
        "baseline_always_zero_balanced_accuracy_percent": float(
            metrics["always_zero_balanced_accuracy_percent"]
        ),
        "baseline_per_gene_modal_accuracy_percent": _finite_or_none(
            metrics["per_gene_modal_accuracy_percent"]
        ),
        "baseline_per_gene_modal_balanced_accuracy_percent": _finite_or_none(
            metrics["per_gene_modal_balanced_accuracy_percent"]
        ),
        "baseline_per_gene_modal_nonzero_accuracy_percent": _finite_or_none(
            metrics["per_gene_modal_nonzero_accuracy_percent"]
        ),
    }
    for token_id in range(4):
        row[f"token_{token_id}_support"] = int(supports[token_id])
        row[f"token_{token_id}_recall_percent"] = _finite_or_none(
            recalls[token_id]
        )
        for predicted_id in range(4):
            row[f"confusion_{token_id}_{predicted_id}"] = int(
                confusion[token_id][predicted_id]
            )
    return row


def _block_metric_rows(
    *,
    entry: Mapping[str, Any],
    result: Any,
) -> Iterator[dict[str, Any]]:
    public_mode = _PUBLIC_MASK_NAMES[str(entry["spec"]["mode"])]
    for block_index, block in enumerate(result.metrics["blocks"]):
        yield {
            "split": "fit",
            "mask_mode": public_mode,
            "mask_replicate": int(entry["replicate"]),
            # Do not export the artifact's routing label.  Its deterministic
            # order is enough for within-run descriptive inspection.
            "macroblock_index": block_index,
            "n_cells": int(block["n_cells"]),
            "n_masked": int(block["n_masked"]),
            "masked_huber": _finite_or_none(block["huber"]),
            "masked_mse": _finite_or_none(block["mse"]),
            "masked_mae": _finite_or_none(block["mae"]),
            "masked_r2": _finite_or_none(block["r2"]),
            "masked_percent_variance_explained": _finite_or_none(
                block["percent_variance_explained"]
            ),
            "pearson_flat": _finite_or_none(block["pearson_flat"]),
            "spearman_flat": _finite_or_none(block["spearman_flat"]),
            "n_correlation_pairs": int(block["n_correlation_pairs"]),
        }


def _protected_prediction_rows(
    *,
    archive: RunArchive,
    dataset: Mapping[str, Any],
    graph_id: str,
    edge_count: int,
    fold: int,
    entry: Mapping[str, Any],
    result: Any,
    target: np.ndarray,
    huber_delta: float,
    sample_key_salt: str,
) -> Iterator[dict[str, Any]]:
    prediction = result.predictions.numpy()
    mask = result.mask.numpy().astype(bool, copy=False)
    public_mode = _PUBLIC_MASK_NAMES[str(entry["spec"]["mode"])]
    namespace = (
        f"bagm:{dataset['dataset_id']}:{dataset['version']}:full-core-fit"
    )
    batch: list[dict[str, Any]] = []
    for node_index in np.flatnonzero(mask.any(axis=1)):
        target_indices = np.flatnonzero(mask[node_index])
        truth = target[node_index, target_indices].astype(
            np.float64, copy=False
        )
        estimate = prediction[node_index, target_indices].astype(
            np.float64, copy=False
        )
        absolute = np.abs(estimate - truth)
        huber = np.where(
            absolute <= huber_delta,
            0.5 * absolute**2,
            huber_delta * (absolute - 0.5 * huber_delta),
        )
        batch.append(
            {
                "_protected_local_index": int(node_index),
                "run_id": archive.run_id,
                "graph_id": graph_id,
                "dataset_id": str(dataset["dataset_id"]),
                "split": "fit",
                "fold": fold,
                "y_true": truth.tolist(),
                "y_pred": estimate.tolist(),
                "target_indices": target_indices.astype(int).tolist(),
                "sample_loss": float(huber.mean()),
                "node_count": int(target.shape[0]),
                "edge_count": edge_count,
                "effective_mask_rate": float(
                    target_indices.size / target.shape[1]
                ),
                "masking_type": public_mode,
                "mask_replicate": int(entry["replicate"]),
            }
        )
        if len(batch) >= 128:
            yield from deidentify_prediction_rows(
                batch,
                identifier_fields=["_protected_local_index"],
                salt=sample_key_salt,
                namespace=namespace,
            )
            batch.clear()
    if batch:
        yield from deidentify_prediction_rows(
            batch,
            identifier_fields=["_protected_local_index"],
            salt=sample_key_salt,
            namespace=namespace,
        )


def _protected_token_prediction_rows(
    *,
    archive: RunArchive,
    dataset: Mapping[str, Any],
    graph_id: str,
    edge_count: int,
    fold: int,
    entry: Mapping[str, Any],
    result: Any,
    target: np.ndarray,
    sample_key_salt: str,
) -> Iterator[dict[str, Any]]:
    prediction = np.asarray(result.predictions)
    mask = result.mask.numpy().astype(bool, copy=False)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise FullCoreRunnerError(
            "token prediction, target, and mask shapes are not aligned"
        )
    public_mode = _PUBLIC_MASK_NAMES[str(entry["spec"]["mode"])]
    namespace = (
        f"bagm:{dataset['dataset_id']}:{dataset['version']}:full-core-fit"
    )
    batch: list[dict[str, Any]] = []
    for node_index in np.flatnonzero(mask.any(axis=1)):
        target_indices = np.flatnonzero(mask[node_index])
        truth = target[node_index, target_indices].astype(
            np.int32, copy=False
        )
        estimate = prediction[node_index, target_indices].astype(
            np.int32, copy=False
        )
        batch.append(
            {
                "_protected_local_index": int(node_index),
                "run_id": archive.run_id,
                "graph_id": graph_id,
                "dataset_id": str(dataset["dataset_id"]),
                "split": "fit",
                "fold": fold,
                "y_true": truth.tolist(),
                "y_pred": estimate.tolist(),
                "target_indices": target_indices.astype(int).tolist(),
                "node_count": int(target.shape[0]),
                "edge_count": edge_count,
                "effective_mask_rate": float(
                    target_indices.size / target.shape[1]
                ),
                "masking_type": public_mode,
                "mask_replicate": int(entry["replicate"]),
            }
        )
        if len(batch) >= 128:
            yield from deidentify_prediction_rows(
                batch,
                identifier_fields=["_protected_local_index"],
                salt=sample_key_salt,
                namespace=namespace,
            )
            batch.clear()
    if batch:
        yield from deidentify_prediction_rows(
            batch,
            identifier_fields=["_protected_local_index"],
            salt=sample_key_salt,
            namespace=namespace,
        )


def _mean_finite(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def _final_metrics(
    *,
    replicate_rows: Sequence[Mapping[str, Any]],
    training: FullCoreTrainingResult,
    training_duration: float,
    evaluation_duration: float,
    total_duration: float,
    parameter_count: int,
    checkpoint_size: int,
    replicates_per_mode: int,
    graph_construction_duration: float,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    summary_fields = {
        "masked_huber": "masked_huber",
        "masked_mse": "masked_mse",
        "masked_mae": "masked_mae",
        "masked_r2": "masked_r2",
        "gene_pearson_mean": "gene_mean_pearson",
        "gene_spearman_mean": "gene_mean_spearman",
        "cell_pearson_mean": "cell_mean_pearson",
        "cell_spearman_mean": "cell_mean_spearman",
    }
    for mode in _REQUIRED_PUBLIC_MASKS:
        selected = [
            row for row in replicate_rows if row["mask_mode"] == mode
        ]
        if len(selected) != replicates_per_mode:
            raise FullCoreRunnerError(
                f"expected {replicates_per_mode} held-in evaluation rows "
                f"for {mode}"
            )
        for metric_name, source_name in summary_fields.items():
            value = _mean_finite(selected, source_name)
            if value is not None:
                metrics[f"fit/{mode}/{metric_name}"] = value
        r2_name = f"fit/{mode}/masked_r2"
        if r2_name in metrics:
            # Percentage variance explained is a display transform of the
            # replicate-mean R², not an independently aggregated metric.
            metrics[f"fit/{mode}/masked_percent_variance_explained"] = (
                100.0 * metrics[r2_name]
            )

    metrics.update(
        {
            "resource/training_duration_seconds": training_duration,
            "resource/inference_duration_seconds": evaluation_duration,
            "resource/total_duration_seconds": total_duration,
            "resource/parameter_count": parameter_count,
            "resource/checkpoint_size_bytes": checkpoint_size,
            "resource/peak_vram_gb": (
                max(
                    record.peak_cuda_memory_bytes
                    for record in training.history
                )
                / (1024**3)
            ),
            "graph/construction_duration_seconds": (
                graph_construction_duration
            ),
        }
    )
    return metrics


def _final_token_metrics(
    *,
    replicate_rows: Sequence[Mapping[str, Any]],
    training: FullCoreTrainingResult,
    training_duration: float,
    evaluation_duration: float,
    total_duration: float,
    parameter_count: int,
    checkpoint_size: int,
    replicates_per_mode: int,
    graph_construction_duration: float,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    summary_fields = (
        "masked_token_cross_entropy",
        "masked_token_accuracy_percent",
        "masked_token_balanced_accuracy_percent",
        "masked_nonzero_token_accuracy_percent",
        "baseline_uniform_accuracy_percent",
        "baseline_empirical_frequency_accuracy_percent",
        "baseline_always_zero_accuracy_percent",
        "baseline_always_zero_balanced_accuracy_percent",
        "baseline_per_gene_modal_accuracy_percent",
        "baseline_per_gene_modal_balanced_accuracy_percent",
        "baseline_per_gene_modal_nonzero_accuracy_percent",
        *(f"token_{token_id}_recall_percent" for token_id in range(4)),
    )
    for mode in _REQUIRED_PUBLIC_MASKS:
        selected = [
            row for row in replicate_rows if row["mask_mode"] == mode
        ]
        if len(selected) != replicates_per_mode:
            raise FullCoreRunnerError(
                f"expected {replicates_per_mode} held-in token rows for {mode}"
            )
        for name in summary_fields:
            value = _mean_finite(selected, name)
            if value is not None:
                metrics[f"fit/{mode}/{name}"] = value

    metrics.update(
        {
            "resource/training_duration_seconds": training_duration,
            "resource/inference_duration_seconds": evaluation_duration,
            "resource/total_duration_seconds": total_duration,
            "resource/parameter_count": parameter_count,
            "resource/checkpoint_size_bytes": checkpoint_size,
            "resource/peak_vram_gb": (
                max(
                    record.peak_cuda_memory_bytes
                    for record in training.history
                )
                / (1024**3)
            ),
            "graph/construction_duration_seconds": (
                graph_construction_duration
            ),
        }
    )
    return metrics


def _peak_host_memory_bytes() -> int:
    # Linux ru_maxrss is KiB; this project executes in a Linux container.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def run_full_core_capacity(
    config: Mapping[str, Any],
    archive: RunArchive,
    *,
    sample_key_salt: str,
    full_core_data: FullCoreData | None = None,
    receiver_graph: ReceiverSortedGraph | None = None,
) -> CapacityRunResult:
    """Execute one fixed-budget run inside an existing worker-owned archive.

    ``full_core_data`` and ``receiver_graph`` are dependency-injection seams
    for synthetic tests.  Production calls omit them and always verify/refit
    the declared prepared artifact and rebuild the exact mutual-kNN graph.
    """

    started = time.monotonic()
    if len(sample_key_salt.encode("utf-8")) < 16:
        raise RunValidationError(
            "BAGM_SAMPLE_KEY_SALT must contain at least 16 bytes"
        )
    model_key = _validate_capacity_contract(config)
    model_config = _section(config, "model")
    graph_config = _section(config, "graph")
    dataset = _section(config, "dataset")
    evaluation_config = _section(config, "evaluation")
    task_family = str(evaluation_config["task_family"])
    token_task = model_key in _TOKEN_G2_KEYS

    data_started = time.monotonic()
    core = (
        full_core_data
        if full_core_data is not None
        else load_and_refit_full_core(
            _resolve_prepared_artifact(config, archive)
        )
    )
    data_duration = time.monotonic() - data_started
    materialized_identity = _verify_materialized_identity(core, dataset)
    tokenization_audit: Mapping[str, Any] | None = None
    per_gene_modal_tokens: np.ndarray | None = None
    model_expression: np.ndarray | None = None
    if token_task:
        model_expression = tokenize_expression_counts(
            core.expression_counts,
            spec=DEFAULT_COUNT_TOKEN_SPEC,
        )
        per_gene_modal_tokens = fit_per_gene_modal_tokens(
            model_expression,
            num_tokens=4,
        )
        tokenization_audit = {
            **audit_expression_tokens(
                model_expression,
                spec=DEFAULT_COUNT_TOKEN_SPEC,
            ),
            "fit_scope": "all_nodes_transductive",
            "thresholds_fitted": False,
            "input_mask_token_is_output_class": False,
            "per_gene_modal_baseline_scope": (
                _TOKEN_MODAL_BASELINE_SCOPE
            ),
            "per_gene_modal_baseline_entrywise_holdout": False,
        }

    graph_started = time.monotonic()
    graph = (
        receiver_graph
        if receiver_graph is not None
        else _build_graph(core, graph_config)
    )
    graph_duration = time.monotonic() - graph_started
    if graph.n_nodes != core.n_nodes or not graph.qc.receiver_sorted:
        raise FullCoreRunnerError(
            "exact graph is not receiver-sorted and aligned to the full core"
        )
    if graph.qc.self_loops or graph.qc.duplicate_directed_edges:
        raise FullCoreRunnerError(
            "exact graph violates no-loop or uniqueness invariants"
        )
    _verify_graph_identity(graph, graph_config)

    uses_graph = model_key in _GRAPH_MODEL_KEYS
    view = _fit_view(
        core=core,
        graph=graph,
        uses_graph=uses_graph,
        expression=model_expression,
    )
    training_config = _training_config(config)
    evaluation_masks = _evaluation_masks(config, core, training_config)

    archive.write_json(
        "diagnostics/full_core_preprocessing.json",
        core.preprocessing_qc.to_dict(),
    )
    archive.write_json(
        "diagnostics/graph_statistics.json",
        graph.qc.to_dict(),
    )
    archive.write_json(
        "diagnostics/mask_statistics.json",
        {
            "role": "held_in_fit_technical_replicates",
            "independent_biological_replicates": 1,
            "manifest": evaluation_masks.manifest,
        },
    )
    if tokenization_audit is not None:
        archive.write_json(
            "diagnostics/expression_tokenization.json",
            dict(tokenization_audit),
        )
    archive.write_json(
        "provenance/full_core_inputs.json",
        {
            "fit_scope": "all_nodes_transductive",
            "generalization_estimate": False,
            "prepared_artifact_reference": dataset.get(
                "prepared_artifact_reference"
            ),
            "preprocessing_checksums": core.checksums.to_dict(),
            "materialized_identity_verification": materialized_identity,
            "graph_checksums": graph.checksums.to_dict(),
            "graph_config": dict(graph_config),
            "task_family": task_family,
            "expression_representation": (
                _TOKENIZER_SCHEMA
                if token_task
                else "full_core_fitted_gene_wise_standardized_log1p_counts"
            ),
            "data_preparation_duration_seconds": data_duration,
            "graph_construction_duration_seconds": graph_duration,
        },
    )
    archive.write_json(
        "provenance/fixed_evaluation_masks.json",
        {
            "bundle_manifest": evaluation_masks.manifest,
            "seed_namespace": "held-in-full-core-fixed-evaluation",
            "seed_derivation_relationship": (
                "separate_from_epoch_mask_seed_derivation"
            ),
            "entrywise_holdout": False,
            "entries_may_overlap_training_masks": True,
            "used_for_gradient_updates": False,
            "used_for_gradient_updates_scope": (
                "exact_fixed_evaluation_mask_realizations_only"
            ),
            "evaluation_mask_realizations_used_directly_for_gradient_updates": (
                False
            ),
            "used_for_checkpoint_selection": False,
        },
    )

    set_deterministic_seed(
        training_config.model_seed,
        deterministic=training_config.deterministic,
        warn_only=training_config.deterministic_warn_only,
    )
    model_construction = _model_construction_record(
        model_key=model_key,
        core=core,
        graph=graph,
        model_config=model_config,
    )
    model = _make_model(
        model_key=model_key,
        core=core,
        graph=graph,
        model_config=model_config,
    )
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    training_started = time.monotonic()
    training_result = fit_full_core_model(
        model,
        view,
        training_config,
        objective=(
            "masked_token_cross_entropy"
            if token_task
            else "masked_huber"
        ),
        num_expression_tokens=4 if token_task else None,
    )
    training_duration = time.monotonic() - training_started
    history_rows = [
        {
            "run_id": archive.run_id,
            "split": "fit",
            "training_protocol": training_result.training_protocol,
            **row,
        }
        for row in training_result.history_rows()
    ]
    archive.write_table("metrics/history", history_rows, fallback="jsonl")
    model_name = str(model_config["name"])
    checkpoint_path = archive.write_bytes(
        "checkpoints/last.ckpt",
        _checkpoint_bytes(
            archive=archive,
            model_name=model_name,
            model_config=model_config,
            model_construction=model_construction,
            training=training_result,
            graph=graph,
            core=core,
            masks=evaluation_masks,
            task_family=task_family,
            tokenization_provenance=tokenization_audit,
        ),
    )
    checkpoint_size = checkpoint_path.stat().st_size
    archive.write_json(
        "provenance/full_core_training.json",
        {
            "training_protocol": training_result.training_protocol,
            "graph_execution": training_result.graph_execution,
            "checkpoint_policy": training_result.checkpoint_policy,
            "final_epoch": training_result.final_epoch,
            "fixed_epoch_budget": training_result.fixed_epoch_budget,
            "state_dict_sha256": training_result.final_state_checksum,
            "model_seed": training_config.model_seed,
            "epoch_mask_seed": training_config.mask_seed,
            "parameter_count": parameter_count,
            "device": training_result.device,
            "model_construction": model_construction,
            "objective": (
                "masked_token_cross_entropy"
                if token_task
                else "masked_huber"
            ),
            "num_expression_tokens": 4 if token_task else None,
        },
    )

    replicate_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    target = view.expression.numpy()
    fold = int(config.get("fold", 0))
    if uses_graph:
        graph_id = f"exact_mutual_k{graph.k}_{graph.checksums.graph_sha256[:16]}"
        prediction_edge_count = graph.qc.n_directed_edges
    else:
        graph_id = (
            "self_only_paired_graph_"
            f"{graph.checksums.graph_sha256[:16]}"
        )
        prediction_edge_count = 0

    evaluation_started = time.monotonic()

    def prediction_rows() -> Iterator[dict[str, Any]]:
        for entry in evaluation_masks.manifest["entries"]:
            mask = evaluation_masks.masks[str(entry["entry_id"])]
            if token_task:
                if per_gene_modal_tokens is None:
                    raise FullCoreRunnerError(
                        "token baseline modes were not materialized"
                    )
                result = evaluate_fixed_token_mask(
                    model,
                    view,
                    mask,
                    num_expression_tokens=4,
                    per_gene_modal_tokens=per_gene_modal_tokens,
                    device=training_config.device,
                    amp=training_config.amp,
                    amp_dtype=training_config.amp_dtype,
                )
                metric_row = _token_replicate_metric_row(
                    entry=entry, result=result
                )
            else:
                result = evaluate_fixed_mask(
                    model,
                    view,
                    mask,
                    device=training_config.device,
                    huber_delta=training_config.huber_delta,
                    amp=training_config.amp,
                    amp_dtype=training_config.amp_dtype,
                )
                metric_row = _replicate_metric_row(
                    entry=entry, result=result
                )
            replicate_rows.append(metric_row)
            if not token_task:
                block_rows.extend(
                    _block_metric_rows(entry=entry, result=result)
                )
            public_mode = str(metric_row["mask_mode"])
            event_metric_names = (
                (
                    "masked_token_cross_entropy",
                    "masked_token_accuracy_percent",
                    "masked_token_balanced_accuracy_percent",
                    "masked_nonzero_token_accuracy_percent",
                    "baseline_uniform_accuracy_percent",
                    "baseline_empirical_frequency_accuracy_percent",
                    "baseline_always_zero_accuracy_percent",
                    "baseline_always_zero_balanced_accuracy_percent",
                    "baseline_per_gene_modal_accuracy_percent",
                    "baseline_per_gene_modal_balanced_accuracy_percent",
                    "baseline_per_gene_modal_nonzero_accuracy_percent",
                    *(f"token_{token_id}_recall_percent" for token_id in range(4)),
                )
                if token_task
                else (
                    "masked_huber",
                    "masked_mse",
                    "masked_mae",
                    "masked_r2",
                    "masked_percent_variance_explained",
                )
            )
            for metric_name in event_metric_names:
                metric_value = metric_row[metric_name]
                if metric_value is None:
                    continue
                archive.append_metric_event(
                    {
                        "name": f"fit/{public_mode}/{metric_name}",
                        "value": metric_value,
                        "mask_replicate": int(entry["replicate"]),
                        "mask_seed": int(entry["seed"]),
                    }
                )
            if (
                public_mode == "whole_node"
                and int(entry["replicate"]) == 0
            ):
                if token_task:
                    yield from _protected_token_prediction_rows(
                        archive=archive,
                        dataset=dataset,
                        graph_id=graph_id,
                        edge_count=prediction_edge_count,
                        fold=fold,
                        entry=entry,
                        result=result,
                        target=target,
                        sample_key_salt=sample_key_salt,
                    )
                else:
                    yield from _protected_prediction_rows(
                        archive=archive,
                        dataset=dataset,
                        graph_id=graph_id,
                        edge_count=prediction_edge_count,
                        fold=fold,
                        entry=entry,
                        result=result,
                        target=target,
                        huber_delta=training_config.huber_delta,
                        sample_key_salt=sample_key_salt,
                    )

    prediction_path = archive.write_prediction_jsonl_stream(
        "fit", prediction_rows()
    )
    evaluation_duration = time.monotonic() - evaluation_started
    archive.write_table(
        "metrics/evaluation_replicates",
        replicate_rows,
        fallback="jsonl",
    )
    if block_rows:
        archive.write_table(
            "metrics/macroblock_descriptives",
            block_rows,
            fallback="jsonl",
        )

    total_duration = time.monotonic() - started
    final_metrics_function = (
        _final_token_metrics if token_task else _final_metrics
    )
    final_metrics = final_metrics_function(
        replicate_rows=replicate_rows,
        training=training_result,
        training_duration=training_duration,
        evaluation_duration=evaluation_duration,
        total_duration=total_duration,
        parameter_count=parameter_count,
        checkpoint_size=checkpoint_size,
        replicates_per_mode=int(
            _section(config, "evaluation")[
                "mask_replicates_per_mode"
            ]
        ),
        graph_construction_duration=graph_duration,
    )
    primary_name = (
        _TOKEN_PRIMARY_METRIC
        if token_task
        else "fit/whole_node/masked_huber"
    )
    primary_value = float(final_metrics[primary_name])
    for name, value in final_metrics.items():
        archive.append_metric_event(
            {
                "name": name,
                "value": value,
                "phase": "final_aggregate",
            }
        )
    archive.write_json("metrics/final.json", final_metrics)
    training_losses = np.asarray(
        [
            record.train_loss
            for record in training_result.history
        ],
        dtype=np.float64,
    )
    convergence_tail = training_losses[
        -min(20, len(training_losses)) :
    ]
    convergence_slope = (
        float(
            np.polyfit(
                np.arange(len(convergence_tail), dtype=np.float64),
                convergence_tail,
                1,
            )[0]
        )
        if len(convergence_tail) >= 2
        else None
    )
    archive.write_json(
        "diagnostics/training_convergence.json",
        {
            "objective": (
                "masked_token_cross_entropy"
                if token_task
                else "masked_huber"
            ),
            "final_epoch": training_result.final_epoch,
            "final_train_loss": training_result.final_train_loss,
            "minimum_observed_train_loss": min(
                record.train_loss for record in training_result.history
            ),
            "last_20_epoch_loss_slope": convergence_slope,
            "all_epochs_completed": (
                len(training_result.history)
                == training_result.fixed_epoch_budget
            ),
            "all_losses_and_gradients_finite": True,
        },
    )
    peak_vram_bytes = max(
        record.peak_cuda_memory_bytes
        for record in training_result.history
    )
    replicates_per_mode = int(
        _section(config, "evaluation")["mask_replicates_per_mode"]
    )
    diagnostic_resource_pilot = (
        _section(config, "trainer").get("diagnostic_resource_pilot")
        is True
    )
    summary = {
        "run_id": archive.run_id,
        "status": "success",
        "training_exit_status": "success",
        "evaluation_protocol": _PROTOCOL,
        "task_family": task_family,
        "canonical_prediction_split": "fit",
        "model_name": model_name,
        "model_seed": training_config.model_seed,
        "final_epoch": training_result.final_epoch,
        "fixed_epoch_budget": training_result.fixed_epoch_budget,
        "checkpoint_role": "last",
        "primary_metric_name": primary_name,
        "primary_metric_value": primary_value,
        "metrics": final_metrics,
        "parameter_count": parameter_count,
        "duration_seconds": total_duration,
        "peak_vram_gb": peak_vram_bytes / (1024**3),
        "peak_host_memory_bytes": _peak_host_memory_bytes(),
        "graph_sha256": graph.checksums.graph_sha256,
        "graph_directed_edges": graph.qc.n_directed_edges,
        "evaluation_mask_bundle_sha256": evaluation_masks.checksum,
        "evaluation_mask_replicates_per_mode": replicates_per_mode,
        "evaluation_metrics_include_all_configured_replicates_per_mode": True,
        "canonical_prediction_selection": {
            "split": "fit",
            "mask_mode": "whole_node",
            "mask_replicate": 0,
            "selection_status": "prespecified",
            "available_replicates_per_mode": replicates_per_mode,
        },
        "diagnostic_resource_pilot": diagnostic_resource_pilot,
        "conclusion_eligible": not diagnostic_resource_pilot,
        "generalization_estimate": False,
        "maximum_claim": (
            "diagnostic runtime and memory feasibility only"
            if diagnostic_resource_pilot
            else (
                (
                    "held-in masked-token reconstruction capacity in one "
                    "transductively fitted core"
                )
                if token_task
                else (
                    "held-in masked-expression reconstruction capacity in "
                    "one transductively fitted core"
                )
            )
        ),
    }
    archive.write_summary(summary)
    return CapacityRunResult(
        run_id=archive.run_id,
        model_name=model_name,
        primary_metric_name=primary_name,
        primary_metric_value=primary_value,
        final_epoch=training_result.final_epoch,
        checkpoint_path=checkpoint_path,
        prediction_path=prediction_path,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one worker-owned held-in full-core fixed-budget capacity job."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-scratch", required=True, type=Path)
    return parser


def _worker_archive_and_config(
    args: argparse.Namespace,
) -> tuple[RunArchive, dict[str, Any]]:
    run_id = os.environ.get("BAGM_RUN_ID", "").strip()
    environment_scratch = os.environ.get("BAGM_RUN_SCRATCH", "").strip()
    if not run_id or not environment_scratch:
        raise FullCoreRunnerError(
            "BAGM_RUN_ID and BAGM_RUN_SCRATCH are required; this runner "
            "must execute under the queue worker"
        )
    supplied_scratch = args.run_scratch.resolve(strict=False)
    if supplied_scratch != Path(environment_scratch).resolve(strict=False):
        raise FullCoreRunnerError(
            "--run-scratch does not match BAGM_RUN_SCRATCH"
        )
    expected_config = supplied_scratch / "config.resolved.yaml"
    if args.config.resolve(strict=False) != expected_config.resolve(
        strict=False
    ):
        raise FullCoreRunnerError(
            "--config must be the worker-owned resolved configuration"
        )
    environment_config = os.environ.get("BAGM_CONFIG_PATH")
    if environment_config and Path(environment_config).resolve(
        strict=False
    ) != expected_config.resolve(strict=False):
        raise FullCoreRunnerError(
            "BAGM_CONFIG_PATH does not match the worker-owned configuration"
        )
    paths = current_paths()
    archive = RunArchive.attach_active(
        run_id,
        paths=paths,
        scratch_path=supplied_scratch,
    )
    configuration = load_yaml_mapping(expected_config)
    validate_experiment_config(configuration)
    return archive, configuration


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    archive, config = _worker_archive_and_config(args)
    salt = os.environ.get("BAGM_SAMPLE_KEY_SALT", "")
    result = run_full_core_capacity(
        config,
        archive,
        sample_key_salt=salt,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "model_name": result.model_name,
                "primary_metric_name": result.primary_metric_name,
                "primary_metric_value": result.primary_metric_value,
                "final_epoch": result.final_epoch,
                "checkpoint": str(result.checkpoint_path),
                "predictions": str(result.prediction_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
