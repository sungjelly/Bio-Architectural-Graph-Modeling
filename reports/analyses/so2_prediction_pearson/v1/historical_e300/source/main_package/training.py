"""Deterministic full-graph training for the spatial model ladder.

This module intentionally has no CLI or configuration-file dependency.  Each
``GraphSplitView`` is already a split-restricted graph with local node indices
and no cross-split edges.  Training evaluates the complete supplied topology;
there is no stochastic neighbor sampling.  For node/block masks, only masked
target nodes are decoded, while their full source-node union still participates
in message passing.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import math
import os
import random
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .expression_tokens import evaluate_masked_token_predictions
from .masking import (
    MaskBatch,
    MaskSpec,
    curriculum_mode,
    derive_mask_seed,
    generate_mask,
    paired_epoch_seed,
)
from .metrics import evaluate_masked_predictions, masked_huber_loss
from .models import (
    AdditiveEdgeMessageModel,
    BroadSpatialFieldControl,
    EdgeParameterMatchedSelfControl,
    EdgeConditionedGATv2,
    MeanNeighborModel,
    ModelOutput,
    ParameterMatchedSelfControl,
    SelfOnlyMLP,
    TopologyGATv2,
)


@dataclass(frozen=True)
class GraphSplitView:
    """One independently constructed split graph.

    Coordinates and block IDs are routing/evaluation data.  Coordinates are
    passed only through the explicitly labeled broad spatial-field control
    path; ordinary ladder models never receive them.  Node covariates are the
    explicit always-visible morphology/imaging matrix.
    """

    expression: Tensor
    coordinates_um: Tensor
    edge_index: Tensor
    node_covariates: Optional[Tensor] = None
    edge_attributes: Optional[Tensor] = None
    block_ids: Optional[Any] = None
    name: str = "split"

    def __post_init__(self) -> None:
        expression = torch.as_tensor(self.expression)
        if expression.ndim != 2 or not expression.is_floating_point():
            raise TypeError(
                "expression must be a floating [num_nodes, num_genes] tensor"
            )
        if not bool(torch.isfinite(expression).all()):
            raise ValueError("expression must contain only finite values")
        num_nodes = expression.shape[0]
        if num_nodes == 0 or expression.shape[1] == 0:
            raise ValueError("expression dimensions must be non-empty")

        coordinates = torch.as_tensor(self.coordinates_um)
        if coordinates.shape != (num_nodes, 2):
            raise ValueError("coordinates_um must have shape [num_nodes, 2]")
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("coordinates_um must contain only finite values")
        coordinates = coordinates.detach().to(device="cpu", dtype=torch.float64)

        edge_index = torch.as_tensor(self.edge_index)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if (
            edge_index.dtype == torch.bool
            or edge_index.is_floating_point()
            or edge_index.is_complex()
        ):
            raise TypeError("edge_index must use an integer dtype")
        edge_index = edge_index.to(dtype=torch.long)
        if edge_index.numel():
            if bool((edge_index < 0).any()) or bool(
                (edge_index >= num_nodes).any()
            ):
                raise ValueError("edge_index contains an out-of-range node")
            if bool((edge_index[0] == edge_index[1]).any()):
                raise ValueError("self loops are prohibited")

        node_covariates = self.node_covariates
        if node_covariates is not None:
            node_covariates = torch.as_tensor(node_covariates)
            if (
                node_covariates.ndim != 2
                or node_covariates.shape[0] != num_nodes
                or not node_covariates.is_floating_point()
            ):
                raise TypeError(
                    "node_covariates must be a floating "
                    "[num_nodes, num_covariates] tensor"
                )
            if not bool(torch.isfinite(node_covariates).all()):
                raise ValueError("node_covariates must contain only finite values")

        edge_attributes = self.edge_attributes
        if edge_attributes is not None:
            edge_attributes = torch.as_tensor(edge_attributes)
            if (
                edge_attributes.ndim != 2
                or edge_attributes.shape[0] != edge_index.shape[1]
                or not edge_attributes.is_floating_point()
            ):
                raise TypeError(
                    "edge_attributes must be a floating "
                    "[num_edges, edge_attribute_dim] tensor"
                )
            if edge_attributes.shape[1] == 0:
                raise ValueError("edge_attributes must have a positive width")
            if not bool(torch.isfinite(edge_attributes).all()):
                raise ValueError("edge_attributes must contain only finite values")

        block_ids = self.block_ids
        if block_ids is not None:
            if torch.is_tensor(block_ids):
                block_ids = block_ids.detach().cpu().numpy()
            block_ids = np.asarray(block_ids)
            if block_ids.shape != (num_nodes,):
                raise ValueError("block_ids must have shape [num_nodes]")
            block_ids = np.array(block_ids, copy=True)

        name = str(self.name).strip()
        if not name:
            raise ValueError("split name must be non-empty")
        object.__setattr__(self, "expression", expression)
        object.__setattr__(self, "coordinates_um", coordinates)
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "node_covariates", node_covariates)
        object.__setattr__(self, "edge_attributes", edge_attributes)
        object.__setattr__(self, "block_ids", block_ids)
        object.__setattr__(self, "name", name)

    @property
    def num_nodes(self) -> int:
        return int(self.expression.shape[0])

    @property
    def num_genes(self) -> int:
        return int(self.expression.shape[1])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def node_covariate_dim(self) -> int:
        return (
            0
            if self.node_covariates is None
            else int(self.node_covariates.shape[1])
        )

    @property
    def edge_attribute_dim(self) -> int:
        return (
            0
            if self.edge_attributes is None
            else int(self.edge_attributes.shape[1])
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Optimizer, mask-curriculum, and early-stopping settings."""

    max_epochs: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    huber_delta: float = 1.0
    patience: int = 25
    min_delta: float = 0.0
    curriculum: str = "P+N+B"
    warmup_epochs: int = 10
    partial_gene_rate: float = 0.20
    node_rate: float = 0.10
    block_node_rate: float = 0.10
    block_width_um: Optional[float] = None
    block_shape: str = "disk"
    mask_seed: int = 0
    model_seed: int = 0
    edge_dropout: float = 0.10
    amp: bool = False
    amp_dtype: str = "auto"
    deterministic: bool = True
    deterministic_warn_only: bool = True
    device: Optional[str] = None
    restore_best: bool = True

    def __post_init__(self) -> None:
        if int(self.max_epochs) <= 0:
            raise ValueError("max_epochs must be positive")
        for name, allow_zero in (
            ("learning_rate", True),
            ("weight_decay", True),
            ("gradient_clip_norm", False),
            ("huber_delta", False),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (
                value < 0 if allow_zero else value <= 0
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if int(self.patience) < 0 or int(self.warmup_epochs) < 0:
            raise ValueError("patience and warmup_epochs cannot be negative")
        if not math.isfinite(float(self.min_delta)) or float(self.min_delta) < 0:
            raise ValueError("min_delta must be finite and non-negative")
        for name in (
            "partial_gene_rate",
            "node_rate",
            "block_node_rate",
            "edge_dropout",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        if self.partial_gene_rate == 0:
            raise ValueError("partial_gene_rate must be positive")
        if self.node_rate == 0:
            raise ValueError("node_rate must be positive")
        if self.block_node_rate == 0 and self.block_width_um is None:
            raise ValueError(
                "block_node_rate must be positive without a fixed block width"
            )
        if self.block_width_um is not None and (
            not math.isfinite(float(self.block_width_um))
            or float(self.block_width_um) <= 0
        ):
            raise ValueError("block_width_um must be finite and positive")
        # Validate the schedule even when all requested epochs fall in warm-up.
        curriculum_mode(
            epoch=int(self.warmup_epochs),
            seed=int(self.mask_seed),
            curriculum=self.curriculum,
            warmup_epochs=int(self.warmup_epochs),
        )
        MaskSpec(
            mode="block",
            partial_gene_rate=self.partial_gene_rate,
            node_rate=self.node_rate,
            block_node_rate=self.block_node_rate,
            block_width_um=self.block_width_um,
            block_shape=self.block_shape,
        )
        dtype = str(self.amp_dtype).lower()
        if dtype not in {"auto", "float16", "bfloat16"}:
            raise ValueError(
                "amp_dtype must be 'auto', 'float16', or 'bfloat16'"
            )
        object.__setattr__(self, "amp_dtype", dtype)


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    mask_mode: str
    mask_seed: int
    mask_checksum: str
    edge_dropout_seed: int
    edge_checksum: str
    n_masked_entries: int
    n_target_nodes: int
    n_edges_used: int
    train_loss: float
    validation_loss: float
    gradient_norm: float
    improved: bool
    stage: str = "single"


@dataclass(frozen=True)
class TrainingStageRecord:
    """Auditable provenance for one optimizer stage."""

    name: str
    start_epoch: int
    requested_epochs: int
    completed_epochs: int
    learning_rate: float
    self_branch_trainable: bool
    early_stopping: bool


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    validation_mask: Tensor
    validation_mask_checksum: str
    device: str
    graph_execution: str = "full_split_exact_no_neighbor_sampling"
    training_protocol: str = "single_stage"
    stage_provenance: tuple[TrainingStageRecord, ...] = ()
    pretrained_self_source: Optional[str] = None
    pretrained_self_checksum: Optional[str] = None
    spatial_control_provenance: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class EvaluationResult:
    predictions: Tensor
    metrics: Mapping[str, Any]
    mask: Tensor
    self_prediction: Optional[Tensor] = None
    neighbor_prediction: Optional[Tensor] = None
    attention_weights: Optional[Tensor] = None
    edge_embedding: Optional[Tensor] = None
    edge_message: Optional[Tensor] = None
    edge_index: Optional[Tensor] = None


@dataclass(frozen=True)
class TokenEvaluationResult:
    """Categorical predictions and metrics for one immutable expression mask."""

    predictions: Tensor
    metrics: Mapping[str, Any]
    mask: Tensor


@dataclass(frozen=True)
class _DeviceView:
    expression: Tensor
    node_covariates: Optional[Tensor]
    edge_index: Tensor
    edge_attributes: Optional[Tensor]
    broad_spatial_coordinates_um: Optional[Tensor] = None


def set_deterministic_seed(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = False,
) -> None:
    """Seed Python, NumPy, CPU, and all visible CUDA generators."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def build_model(
    model_id: str,
    *,
    num_genes: int,
    node_covariate_dim: int = 0,
    edge_attribute_dim: Optional[int] = None,
    seed: int = 0,
    deterministic: bool = True,
    **model_kwargs: Any,
) -> nn.Module:
    """Construct one ladder model under a deterministic initialization seed."""

    key = "".join(
        character
        for character in str(model_id).strip().lower()
        if character.isalnum()
    )
    classes: dict[str, type[nn.Module]] = {
        "b0": SelfOnlyMLP,
        "self": SelfOnlyMLP,
        "selfmlp": SelfOnlyMLP,
        "b0matched": ParameterMatchedSelfControl,
        "matchedself": ParameterMatchedSelfControl,
        "parametermatchedself": ParameterMatchedSelfControl,
        "b0g2matched": EdgeParameterMatchedSelfControl,
        "g2matchedself": EdgeParameterMatchedSelfControl,
        "edgeparametermatchedself": EdgeParameterMatchedSelfControl,
        "broadfield": BroadSpatialFieldControl,
        "broadspatialfield": BroadSpatialFieldControl,
        "spatialfield": BroadSpatialFieldControl,
        "b1": MeanNeighborModel,
        "mean": MeanNeighborModel,
        "meanneighbor": MeanNeighborModel,
        "g1": TopologyGATv2,
        "topologygat": TopologyGATv2,
        "topologygatv2": TopologyGATv2,
        "g2": EdgeConditionedGATv2,
        "edgegat": EdgeConditionedGATv2,
        "edgeconditionedgatv2": EdgeConditionedGATv2,
        "g3": AdditiveEdgeMessageModel,
        "additive": AdditiveEdgeMessageModel,
        "additiveedgemessage": AdditiveEdgeMessageModel,
    }
    if key not in classes:
        raise ValueError(
            f"unknown model_id {model_id!r}; expected B0, B0-matched, "
            "B0-G2-matched, Broad-Field, B1, G1, G2, or G3"
        )
    model_class = classes[key]
    set_deterministic_seed(seed, deterministic=deterministic)
    kwargs = {
        "num_genes": int(num_genes),
        "node_covariate_dim": int(node_covariate_dim),
        **model_kwargs,
    }
    if model_class in (
        EdgeParameterMatchedSelfControl,
        EdgeConditionedGATv2,
        AdditiveEdgeMessageModel,
    ):
        if edge_attribute_dim is None or int(edge_attribute_dim) <= 0:
            raise ValueError(f"{model_id} requires edge_attribute_dim > 0")
        kwargs["edge_attribute_dim"] = int(edge_attribute_dim)
    return model_class(**kwargs)


def _resolve_device(model: nn.Module, requested: Optional[str]) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    try:
        current = next(model.parameters()).device
    except StopIteration:
        current = torch.device("cpu")
    if current.type != "cpu":
        return current
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_dtype(model: nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _to_device_view(
    view: GraphSplitView,
    *,
    device: torch.device,
    dtype: torch.dtype,
    include_broad_spatial_coordinates: bool = False,
) -> _DeviceView:
    return _DeviceView(
        expression=view.expression.to(device=device, dtype=dtype),
        node_covariates=(
            None
            if view.node_covariates is None
            else view.node_covariates.to(device=device, dtype=dtype)
        ),
        edge_index=view.edge_index.to(device=device, dtype=torch.long),
        edge_attributes=(
            None
            if view.edge_attributes is None
            else view.edge_attributes.to(device=device, dtype=dtype)
        ),
        broad_spatial_coordinates_um=(
            view.coordinates_um.to(device=device, dtype=torch.float64)
            if include_broad_spatial_coordinates
            else None
        ),
    )


def _array_checksum(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _state_dict_checksum(state_dict: Mapping[str, Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and values in a state dictionary."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = torch.as_tensor(state_dict[name]).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _copy_pretrained_b0_branch(
    model: AdditiveEdgeMessageModel,
    pretrained_b0: SelfOnlyMLP | Mapping[str, Any],
) -> tuple[str, str]:
    """Load the three B0 self modules into G3 and return source provenance."""

    if isinstance(pretrained_b0, SelfOnlyMLP):
        model.copy_self_branch_from(pretrained_b0)
        source_state = pretrained_b0.state_dict()
        source = "SelfOnlyMLP"
    elif isinstance(pretrained_b0, Mapping):
        nested_state = pretrained_b0.get("state_dict")
        source_state = (
            nested_state
            if isinstance(nested_state, Mapping)
            else pretrained_b0
        )
        source = (
            "checkpoint[state_dict]"
            if isinstance(nested_state, Mapping)
            else "state_dict"
        )
    else:
        raise TypeError(
            "pretrained_b0 must be a SelfOnlyMLP or a B0 checkpoint mapping"
        )

    if not isinstance(source_state, Mapping):
        raise TypeError("the B0 checkpoint state_dict must be a mapping")

    canonical: dict[str, Tensor] = {}
    module_specs = (
        ("encoder", model.encoder, "encoder"),
        ("self_block", model.self_block, "self_block"),
        ("decoder", model.self_decoder, "decoder"),
    )
    for source_prefix, destination, checksum_prefix in module_specs:
        prefix = f"{source_prefix}."
        module_state = {
            str(name)[len(prefix) :]: value
            for name, value in source_state.items()
            if str(name).startswith(prefix)
        }
        if not module_state:
            raise ValueError(
                f"B0 checkpoint is missing the {source_prefix!r} module"
            )
        if not all(torch.is_tensor(value) for value in module_state.values()):
            raise TypeError("all B0 state_dict values must be tensors")
        try:
            destination.load_state_dict(module_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "B0 checkpoint self-branch dimensions do not match G3"
            ) from exc
        for name, value in module_state.items():
            canonical[f"{checksum_prefix}.{name}"] = torch.as_tensor(value)

    return source, _state_dict_checksum(canonical)


def apply_edge_dropout(
    edge_index: Tensor,
    edge_attributes: Optional[Tensor],
    *,
    probability: float,
    seed: int,
) -> tuple[Tensor, Optional[Tensor], Tensor]:
    """Drop directed edges with an isolated, reproducible NumPy generator."""

    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("edge dropout probability must lie in [0, 1]")
    num_edges = int(edge_index.shape[1])
    if edge_attributes is not None and edge_attributes.shape[0] != num_edges:
        raise ValueError("edge attributes are not aligned with edge_index")
    if probability == 0.0:
        keep = torch.ones(num_edges, dtype=torch.bool, device=edge_index.device)
    elif probability == 1.0:
        keep = torch.zeros(num_edges, dtype=torch.bool, device=edge_index.device)
    else:
        draw = np.random.default_rng(int(seed)).random(num_edges)
        keep = torch.as_tensor(
            draw >= probability,
            dtype=torch.bool,
            device=edge_index.device,
        )
    kept_edges = edge_index[:, keep]
    kept_attributes = (
        None if edge_attributes is None else edge_attributes[keep]
    )
    return kept_edges, kept_attributes, keep


def _mask_spec_for_mode(
    mode: str,
    config: TrainingConfig,
) -> MaskSpec:
    return MaskSpec(
        mode=mode,
        partial_gene_rate=config.partial_gene_rate,
        node_rate=config.node_rate,
        block_node_rate=config.block_node_rate,
        block_width_um=config.block_width_um,
        block_shape=config.block_shape,
    )


def make_epoch_mask(
    view: GraphSplitView,
    config: TrainingConfig,
    epoch: int,
) -> MaskBatch:
    """Create the architecture-independent paired mask for one epoch."""

    mode = curriculum_mode(
        epoch=epoch,
        seed=config.mask_seed,
        curriculum=config.curriculum,
        warmup_epochs=config.warmup_epochs,
    )
    seed = paired_epoch_seed(config.mask_seed, epoch)
    return generate_mask(
        _mask_spec_for_mode(mode, config),
        n_genes=view.num_genes,
        coordinates_um=view.coordinates_um,
        seed=seed,
    )


def _prepare_fixed_mask(
    view: GraphSplitView,
    mask: Any,
) -> Tensor:
    if torch.is_tensor(mask):
        prepared = mask.detach().to(device="cpu", dtype=torch.bool).clone()
    else:
        prepared = torch.tensor(mask, dtype=torch.bool, device="cpu")
    if prepared.shape != view.expression.shape:
        raise ValueError(
            "fixed mask must have the same [nodes, genes] shape as expression"
        )
    if not bool(prepared.any()):
        raise ValueError("fixed mask must select at least one expression entry")
    return prepared.contiguous()


def _make_validation_mask(
    view: GraphSplitView,
    config: TrainingConfig,
    validation_mode: str | MaskSpec,
) -> Tensor:
    spec = (
        validation_mode
        if isinstance(validation_mode, MaskSpec)
        else _mask_spec_for_mode(validation_mode, config)
    )
    seed = derive_mask_seed(
        config.mask_seed, "fixed-validation", view.name, spec.spec_id
    )
    batch = generate_mask(
        spec,
        n_genes=view.num_genes,
        coordinates_um=view.coordinates_um,
        seed=seed,
    )
    return torch.from_numpy(np.array(batch.mask, copy=True))


def _autocast_context(
    *,
    enabled: bool,
    device: torch.device,
    dtype_name: str,
) -> Any:
    if not enabled:
        return nullcontext()
    if dtype_name not in {"auto", "float16", "bfloat16"}:
        raise ValueError(
            "amp_dtype must be 'auto', 'float16', or 'bfloat16'"
        )
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("AMP is supported only on CUDA or CPU devices")
    if dtype_name == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    elif dtype_name == "float16":
        if device.type == "cpu":
            raise ValueError("CPU AMP requires bfloat16 or auto")
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _forward_model(
    model: nn.Module,
    view: _DeviceView,
    input_expression: Tensor,
    gene_mask: Tensor,
    *,
    edge_index: Tensor,
    edge_attributes: Optional[Tensor],
    target_nodes: Optional[Tensor] = None,
    return_explanations: bool = False,
) -> ModelOutput:
    """Dispatch coordinates exclusively to the labeled broad-field control."""

    kwargs: dict[str, Any] = {
        "edge_index": edge_index,
        "edge_attributes": edge_attributes,
        "node_covariates": view.node_covariates,
        "target_nodes": target_nodes,
        "return_explanations": return_explanations,
    }
    if isinstance(model, BroadSpatialFieldControl):
        if view.broad_spatial_coordinates_um is None:
            raise RuntimeError(
                "broad spatial coordinates were not enabled for this control"
            )
        kwargs["coordinates_um"] = view.broad_spatial_coordinates_um
    elif view.broad_spatial_coordinates_um is not None:
        raise RuntimeError(
            "ordinary models cannot receive broad spatial coordinates"
        )
    return model(input_expression, gene_mask, **kwargs)


def _forward_masked_targets(
    model: nn.Module,
    view: _DeviceView,
    mask: Tensor,
    *,
    edge_index: Tensor,
    edge_attributes: Optional[Tensor],
) -> tuple[ModelOutput, Tensor, Tensor]:
    target_nodes = mask.any(dim=1).nonzero(as_tuple=False).flatten()
    if target_nodes.numel() == 0:
        raise ValueError("training mask contains no target nodes")
    masked_expression = view.expression.masked_fill(mask, 0.0)
    output = _forward_model(
        model,
        view,
        masked_expression,
        mask,
        edge_index=edge_index,
        edge_attributes=edge_attributes,
        target_nodes=target_nodes,
    )
    target = view.expression.index_select(0, target_nodes)
    target_mask = mask.index_select(0, target_nodes)
    return output, target, target_mask


def _validation_loss(
    model: nn.Module,
    view: _DeviceView,
    fixed_mask_cpu: Tensor,
    *,
    config: TrainingConfig,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    mask = fixed_mask_cpu.to(device=device)
    with torch.no_grad(), _autocast_context(
        enabled=config.amp,
        device=device,
        dtype_name=config.amp_dtype,
    ):
        output, target, target_mask = _forward_masked_targets(
            model,
            view,
            mask,
            edge_index=view.edge_index,
            edge_attributes=view.edge_attributes,
        )
        loss = masked_huber_loss(
            target,
            output.prediction,
            target_mask,
            delta=config.huber_delta,
        )
    model.train(was_training)
    return float(loss.detach().float().cpu())


def _make_grad_scaler(device: torch.device, enabled: bool) -> Any:
    scaler_enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older Torch.
        return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def fit_model(
    model: nn.Module,
    train_view: GraphSplitView,
    validation_view: GraphSplitView,
    config: TrainingConfig,
    *,
    validation_mask: Optional[Any] = None,
    validation_mode: str | MaskSpec = "node",
) -> TrainingResult:
    """Fit one model on complete split graphs with paired epoch masks."""

    if train_view.num_genes != validation_view.num_genes:
        raise ValueError("training and validation gene dimensions differ")
    if train_view.node_covariate_dim != validation_view.node_covariate_dim:
        raise ValueError("training and validation metadata dimensions differ")
    if train_view.edge_attribute_dim != validation_view.edge_attribute_dim:
        raise ValueError("training and validation edge dimensions differ")

    broad_spatial_control = isinstance(model, BroadSpatialFieldControl)
    spatial_control_provenance: Optional[Mapping[str, Any]] = None
    if broad_spatial_control:
        spatial_control_provenance = model.fit_coordinate_basis(
            train_view.coordinates_um
        )
    set_deterministic_seed(
        config.model_seed,
        deterministic=config.deterministic,
        warn_only=config.deterministic_warn_only,
    )
    device = _resolve_device(model, config.device)
    model.to(device)
    dtype = _model_dtype(model)
    train_device = _to_device_view(
        train_view,
        device=device,
        dtype=dtype,
        include_broad_spatial_coordinates=broad_spatial_control,
    )
    validation_device = _to_device_view(
        validation_view,
        device=device,
        dtype=dtype,
        include_broad_spatial_coordinates=broad_spatial_control,
    )
    fixed_validation_mask = (
        _make_validation_mask(validation_view, config, validation_mode)
        if validation_mask is None
        else _prepare_fixed_mask(validation_view, validation_mask)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = _make_grad_scaler(device, config.amp)
    history: list[EpochRecord] = []
    best_state: Optional[dict[str, Tensor]] = None
    best_validation_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(config.max_epochs):
        model.train()
        mask_batch = make_epoch_mask(train_view, config, epoch)
        mask = torch.from_numpy(np.array(mask_batch.mask, copy=True)).to(
            device=device
        )
        edge_dropout_seed = derive_mask_seed(
            config.mask_seed, "edge-dropout", epoch
        )
        edges, edge_attributes, keep = apply_edge_dropout(
            train_device.edge_index,
            train_device.edge_attributes,
            probability=config.edge_dropout,
            seed=edge_dropout_seed,
        )

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(
            enabled=config.amp,
            device=device,
            dtype_name=config.amp_dtype,
        ):
            output, target, target_mask = _forward_masked_targets(
                model,
                train_device,
                mask,
                edge_index=edges,
                edge_attributes=edge_attributes,
            )
            loss = masked_huber_loss(
                target,
                output.prediction,
                target_mask,
                delta=config.huber_delta,
            )
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError(
                f"non-finite training loss at epoch {epoch}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm).detach().cpu()):
            raise FloatingPointError(
                f"non-finite gradient norm at epoch {epoch}"
            )
        scaler.step(optimizer)
        scaler.update()

        validation_loss = _validation_loss(
            model,
            validation_device,
            fixed_validation_mask,
            config=config,
            device=device,
        )
        if not math.isfinite(validation_loss):
            raise FloatingPointError(
                f"non-finite validation loss at epoch {epoch}"
            )
        improved = (
            validation_loss
            < best_validation_loss - float(config.min_delta)
        )
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        history.append(
            EpochRecord(
                epoch=epoch,
                mask_mode=mask_batch.spec.mode,
                mask_seed=mask_batch.seed,
                mask_checksum=_array_checksum(mask_batch.mask),
                edge_dropout_seed=edge_dropout_seed,
                edge_checksum=_array_checksum(keep),
                n_masked_entries=mask_batch.n_masked_entries,
                n_target_nodes=mask_batch.n_selected_nodes,
                n_edges_used=int(edges.shape[1]),
                train_loss=float(loss.detach().float().cpu()),
                validation_loss=validation_loss,
                gradient_norm=float(gradient_norm.detach().float().cpu()),
                improved=improved,
            )
        )
        if (
            not improved
            and epochs_without_improvement >= config.patience
        ):
            break

    if best_state is None or best_epoch < 0:
        raise RuntimeError("training produced no finite validation checkpoint")
    if config.restore_best:
        model.load_state_dict(best_state)

    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=len(history) < config.max_epochs,
        validation_mask=fixed_validation_mask.clone(),
        validation_mask_checksum=_array_checksum(fixed_validation_mask),
        device=str(device),
        graph_execution=(
            "cell_autonomous_broad_spatial_field_no_graph"
            if broad_spatial_control
            else "full_split_exact_no_neighbor_sampling"
        ),
        stage_provenance=(
            TrainingStageRecord(
                name="single",
                start_epoch=0,
                requested_epochs=config.max_epochs,
                completed_epochs=len(history),
                learning_rate=config.learning_rate,
                self_branch_trainable=True,
                early_stopping=True,
            ),
        ),
        spatial_control_provenance=spatial_control_provenance,
    )


def fit_staged_g3(
    model: AdditiveEdgeMessageModel,
    pretrained_b0: SelfOnlyMLP | Mapping[str, Any],
    train_view: GraphSplitView,
    validation_view: GraphSplitView,
    config: TrainingConfig,
    *,
    frozen_epochs: int = 10,
    joint_learning_rate: float = 1e-4,
    validation_mask: Optional[Any] = None,
    validation_mode: str | MaskSpec = "node",
) -> TrainingResult:
    """Train G3 from a B0 self branch using the locked staged protocol.

    ``config.max_epochs`` is the total epoch budget.  The first
    ``frozen_epochs`` update only the edge/attention/message branch at
    ``config.learning_rate``.  The remaining budget jointly fine-tunes all
    parameters at ``joint_learning_rate`` and uses validation early stopping.
    Epoch numbers do not reset at the stage boundary, so paired masks and edge
    dropout draws remain aligned to the same global epoch schedule.
    """

    if not isinstance(model, AdditiveEdgeMessageModel):
        raise TypeError("fit_staged_g3 requires an AdditiveEdgeMessageModel")
    frozen_epochs = int(frozen_epochs)
    if frozen_epochs < 0:
        raise ValueError("frozen_epochs cannot be negative")
    if frozen_epochs >= config.max_epochs:
        raise ValueError(
            "frozen_epochs must be smaller than the total max_epochs budget"
        )
    joint_learning_rate = float(joint_learning_rate)
    if (
        not math.isfinite(joint_learning_rate)
        or joint_learning_rate < 0
    ):
        raise ValueError(
            "joint_learning_rate must be finite and non-negative"
        )
    if train_view.num_genes != validation_view.num_genes:
        raise ValueError("training and validation gene dimensions differ")
    if train_view.node_covariate_dim != validation_view.node_covariate_dim:
        raise ValueError("training and validation metadata dimensions differ")
    if train_view.edge_attribute_dim != validation_view.edge_attribute_dim:
        raise ValueError("training and validation edge dimensions differ")
    if train_view.num_genes != model.num_genes:
        raise ValueError("training genes do not match the G3 model")
    if train_view.node_covariate_dim != model.node_covariate_dim:
        raise ValueError("training metadata do not match the G3 model")
    if train_view.edge_attribute_dim != model.edge_attribute_dim:
        raise ValueError("training edge attributes do not match the G3 model")

    pretrained_source, pretrained_checksum = _copy_pretrained_b0_branch(
        model, pretrained_b0
    )
    set_deterministic_seed(
        config.model_seed,
        deterministic=config.deterministic,
        warn_only=config.deterministic_warn_only,
    )
    device = _resolve_device(model, config.device)
    model.to(device)
    dtype = _model_dtype(model)
    train_device = _to_device_view(train_view, device=device, dtype=dtype)
    validation_device = _to_device_view(
        validation_view, device=device, dtype=dtype
    )
    fixed_validation_mask = (
        _make_validation_mask(validation_view, config, validation_mode)
        if validation_mask is None
        else _prepare_fixed_mask(validation_view, validation_mask)
    )
    scaler = _make_grad_scaler(device, config.amp)

    history: list[EpochRecord] = []
    stage_provenance: list[TrainingStageRecord] = []
    best_state: Optional[dict[str, Tensor]] = None
    best_validation_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False

    stage_specs = (
        (
            "frozen_neighbor",
            frozen_epochs,
            config.learning_rate,
            False,
            False,
        ),
        (
            "joint_finetune",
            config.max_epochs - frozen_epochs,
            joint_learning_rate,
            True,
            True,
        ),
    )

    try:
        for (
            stage_name,
            requested_epochs,
            learning_rate,
            self_branch_trainable,
            early_stopping,
        ) in stage_specs:
            stage_start = len(history)
            model.set_self_branch_trainable(self_branch_trainable)
            if requested_epochs == 0:
                stage_provenance.append(
                    TrainingStageRecord(
                        name=stage_name,
                        start_epoch=stage_start,
                        requested_epochs=0,
                        completed_epochs=0,
                        learning_rate=learning_rate,
                        self_branch_trainable=self_branch_trainable,
                        early_stopping=early_stopping,
                    )
                )
                continue

            trainable_parameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            if not trainable_parameters:
                raise RuntimeError(
                    f"stage {stage_name!r} has no trainable parameters"
                )
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=learning_rate,
                weight_decay=config.weight_decay,
            )
            if early_stopping:
                epochs_without_improvement = 0

            for stage_epoch in range(requested_epochs):
                epoch = len(history)
                model.train()
                if not self_branch_trainable:
                    # Keep the frozen B0 predictor itself deterministic so the
                    # graph branch learns a residual against a fixed function,
                    # while graph/attention dropout remains active.
                    model.encoder.eval()
                    model.self_block.eval()
                    model.self_decoder.eval()
                mask_batch = make_epoch_mask(train_view, config, epoch)
                mask = torch.from_numpy(
                    np.array(mask_batch.mask, copy=True)
                ).to(device=device)
                edge_dropout_seed = derive_mask_seed(
                    config.mask_seed, "edge-dropout", epoch
                )
                edges, edge_attributes, keep = apply_edge_dropout(
                    train_device.edge_index,
                    train_device.edge_attributes,
                    probability=config.edge_dropout,
                    seed=edge_dropout_seed,
                )

                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(
                    enabled=config.amp,
                    device=device,
                    dtype_name=config.amp_dtype,
                ):
                    output, target, target_mask = _forward_masked_targets(
                        model,
                        train_device,
                        mask,
                        edge_index=edges,
                        edge_attributes=edge_attributes,
                    )
                    loss = masked_huber_loss(
                        target,
                        output.prediction,
                        target_mask,
                        delta=config.huber_delta,
                    )
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch}"
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, config.gradient_clip_norm
                )
                if not bool(
                    torch.isfinite(gradient_norm).detach().cpu()
                ):
                    raise FloatingPointError(
                        f"non-finite gradient norm at epoch {epoch}"
                    )
                scaler.step(optimizer)
                scaler.update()

                validation_loss = _validation_loss(
                    model,
                    validation_device,
                    fixed_validation_mask,
                    config=config,
                    device=device,
                )
                if not math.isfinite(validation_loss):
                    raise FloatingPointError(
                        f"non-finite validation loss at epoch {epoch}"
                    )
                improved = (
                    validation_loss
                    < best_validation_loss - float(config.min_delta)
                )
                if improved:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    best_state = {
                        name: parameter.detach().cpu().clone()
                        for name, parameter in model.state_dict().items()
                    }
                elif early_stopping:
                    epochs_without_improvement += 1

                history.append(
                    EpochRecord(
                        epoch=epoch,
                        mask_mode=mask_batch.spec.mode,
                        mask_seed=mask_batch.seed,
                        mask_checksum=_array_checksum(mask_batch.mask),
                        edge_dropout_seed=edge_dropout_seed,
                        edge_checksum=_array_checksum(keep),
                        n_masked_entries=mask_batch.n_masked_entries,
                        n_target_nodes=mask_batch.n_selected_nodes,
                        n_edges_used=int(edges.shape[1]),
                        train_loss=float(loss.detach().float().cpu()),
                        validation_loss=validation_loss,
                        gradient_norm=float(
                            gradient_norm.detach().float().cpu()
                        ),
                        improved=improved,
                        stage=stage_name,
                    )
                )
                if (
                    early_stopping
                    and not improved
                    and epochs_without_improvement >= config.patience
                    and stage_epoch + 1 < requested_epochs
                ):
                    stopped_early = True
                    break

            stage_provenance.append(
                TrainingStageRecord(
                    name=stage_name,
                    start_epoch=stage_start,
                    requested_epochs=requested_epochs,
                    completed_epochs=len(history) - stage_start,
                    learning_rate=learning_rate,
                    self_branch_trainable=self_branch_trainable,
                    early_stopping=early_stopping,
                )
            )
            if stopped_early:
                break
    finally:
        # A returned G3 is always ready for evaluation or later fine-tuning.
        model.set_self_branch_trainable(True)

    if best_state is None or best_epoch < 0:
        raise RuntimeError("staged G3 training produced no finite checkpoint")
    if config.restore_best:
        model.load_state_dict(best_state)

    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
        validation_mask=fixed_validation_mask.clone(),
        validation_mask_checksum=_array_checksum(fixed_validation_mask),
        device=str(device),
        training_protocol="g3_b0_frozen_neighbor_then_joint",
        stage_provenance=tuple(stage_provenance),
        pretrained_self_source=pretrained_source,
        pretrained_self_checksum=pretrained_checksum,
    )


def evaluate_fixed_mask(
    model: nn.Module,
    view: GraphSplitView,
    mask: Any,
    *,
    device: Optional[str] = None,
    huber_delta: float = 1.0,
    amp: bool = False,
    amp_dtype: str = "auto",
    return_explanations: bool = False,
) -> EvaluationResult:
    """Evaluate a supplied immutable mask and return predictions plus metrics."""

    fixed_mask = _prepare_fixed_mask(view, mask)
    resolved_device = _resolve_device(model, device)
    model.to(resolved_device)
    broad_spatial_control = isinstance(model, BroadSpatialFieldControl)
    if broad_spatial_control and not model.coordinate_basis_is_fitted:
        raise RuntimeError(
            "the broad spatial basis must be fitted on training coordinates "
            "before held-out evaluation"
        )
    device_view = _to_device_view(
        view,
        device=resolved_device,
        dtype=_model_dtype(model),
        include_broad_spatial_coordinates=broad_spatial_control,
    )
    was_training = model.training
    model.eval()
    mask_device = fixed_mask.to(device=resolved_device)
    masked_expression = device_view.expression.masked_fill(
        mask_device, 0.0
    )
    with torch.no_grad(), _autocast_context(
        enabled=amp,
        device=resolved_device,
        dtype_name=str(amp_dtype).lower(),
    ):
        output = _forward_model(
            model,
            device_view,
            masked_expression,
            mask_device,
            edge_index=device_view.edge_index,
            edge_attributes=device_view.edge_attributes,
            return_explanations=return_explanations,
        )
    model.train(was_training)
    predictions = output.prediction.detach().float().cpu()
    metrics = evaluate_masked_predictions(
        view.expression.detach().cpu(),
        predictions,
        fixed_mask,
        block_ids=view.block_ids,
        huber_delta=huber_delta,
    )
    return EvaluationResult(
        predictions=predictions,
        metrics=metrics,
        mask=fixed_mask.clone(),
        self_prediction=(
            None
            if output.self_prediction is None
            else output.self_prediction.detach().float().cpu()
        ),
        neighbor_prediction=(
            None
            if output.neighbor_prediction is None
            else output.neighbor_prediction.detach().float().cpu()
        ),
        attention_weights=(
            None
            if output.attention_weights is None
            else output.attention_weights.detach().float().cpu()
        ),
        edge_embedding=(
            None
            if output.edge_embedding is None
            else output.edge_embedding.detach().float().cpu()
        ),
        edge_message=(
            None
            if output.edge_message is None
            else output.edge_message.detach().float().cpu()
        ),
        edge_index=(
            None
            if output.edge_index is None
            else output.edge_index.detach().cpu()
        ),
    )


def evaluate_fixed_token_mask(
    model: nn.Module,
    view: GraphSplitView,
    mask: Any,
    *,
    num_expression_tokens: int,
    per_gene_modal_tokens: Any,
    device: Optional[str] = None,
    amp: bool = False,
    amp_dtype: str = "auto",
) -> TokenEvaluationResult:
    """Evaluate masked categorical tokens without retaining dense logits."""

    if (
        isinstance(num_expression_tokens, bool)
        or not isinstance(num_expression_tokens, int)
        or num_expression_tokens < 2
    ):
        raise ValueError("num_expression_tokens must be an integer >= 2")
    fixed_mask = _prepare_fixed_mask(view, mask)
    resolved_device = _resolve_device(model, device)
    model.to(resolved_device)
    device_view = _to_device_view(
        view,
        device=resolved_device,
        dtype=_model_dtype(model),
    )
    was_training = model.training
    model.eval()
    mask_device = fixed_mask.to(device=resolved_device)
    with torch.no_grad(), _autocast_context(
        enabled=amp,
        device=resolved_device,
        dtype_name=str(amp_dtype).lower(),
    ):
        output, target, target_mask = _forward_masked_targets(
            model,
            device_view,
            mask_device,
            edge_index=device_view.edge_index,
            edge_attributes=device_view.edge_attributes,
        )
        expected_shape = (*target.shape, num_expression_tokens)
        if tuple(output.prediction.shape) != expected_shape:
            raise ValueError(
                "token logits shape mismatch: expected "
                f"{expected_shape}, got {tuple(output.prediction.shape)}"
            )
        rounded = target.round()
        if not bool(torch.equal(target, rounded)):
            raise ValueError("token targets must contain integral IDs")
        token_target = rounded.to(dtype=torch.long)
        if bool((token_target < 0).any()) or bool(
            (token_target >= num_expression_tokens).any()
        ):
            raise ValueError("token target lies outside the output vocabulary")
        selected_logits = output.prediction[target_mask]
        selected_targets = token_target[target_mask]
        cross_entropy = F.cross_entropy(
            selected_logits.float(), selected_targets
        )
        target_predictions = output.prediction.argmax(dim=-1).to(
            dtype=torch.int16
        )
    model.train(was_training)

    target_nodes = mask_device.any(dim=1).nonzero(
        as_tuple=False
    ).flatten()
    predictions = torch.zeros(
        view.expression.shape, dtype=torch.int16, device="cpu"
    )
    predictions.index_copy_(
        0,
        target_nodes.detach().cpu(),
        target_predictions.detach().cpu(),
    )
    metrics = evaluate_masked_token_predictions(
        view.expression.detach().cpu().numpy().astype(
            np.int64, copy=False
        ),
        predictions.numpy().astype(np.int64, copy=False),
        fixed_mask.numpy(),
        cross_entropy=float(cross_entropy.detach().cpu()),
        per_gene_modal_tokens=per_gene_modal_tokens,
        num_tokens=num_expression_tokens,
    )
    return TokenEvaluationResult(
        predictions=predictions,
        metrics=metrics,
        mask=fixed_mask.clone(),
    )


train_model = fit_model
evaluate_model = evaluate_fixed_mask


__all__ = [
    "EpochRecord",
    "EvaluationResult",
    "GraphSplitView",
    "TokenEvaluationResult",
    "TrainingConfig",
    "TrainingResult",
    "TrainingStageRecord",
    "apply_edge_dropout",
    "build_model",
    "evaluate_fixed_mask",
    "evaluate_fixed_token_mask",
    "evaluate_model",
    "fit_model",
    "fit_staged_g3",
    "make_epoch_mask",
    "set_deterministic_seed",
    "train_model",
]
