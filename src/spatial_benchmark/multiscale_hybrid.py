"""Parameter-matched additive multiscale hurdle-count model.

The model keeps cell-autonomous, regional-context, and sparse local-message
predictions separate and combines them only by exact addition::

    prediction = self_prediction + regional_prediction + local_prediction

Every routing arm constructs exactly the same modules. ``surrogate`` routing
uses two deterministic within-cell pseudo-messages, ``true`` uses observed
source states, and ``permuted`` uses the same observed edge slots and
attributes while gathering local source states through a caller-supplied node
permutation. Consequently, changing an arm label changes neither parameter
names, shapes, nor counts.

Both graph branches partition complete incoming neighborhoods by receiver.
Edge-sized hidden activations are recomputed during backward when activation
checkpointing is enabled, so the implementation never requires a full
``edges x hidden`` tensor.  Stable receiver ordering and CSR-style segment
reductions preserve exact per-receiver softmax and aggregation.

The local branch exposes signed contributions only for explicitly selected
edges (and optionally selected genes).  It never creates an unavoidable
``all_edges x all_genes`` tensor.  Attention remains computational routing,
not biological importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .hurdle_continuous import NUM_HURDLE_CONTINUOUS_CHANNELS
from .hybrid_count import HybridCountNodeEncoder
from .models import (
    ResidualFeedForward,
    SharedEdgeEncoder,
    _normalize_target_nodes,
    _prepare_edge_attributes,
    _prepare_edge_index,
    _select_targets,
)


RoutingMode = Literal["surrogate", "true", "permuted"]
ROUTING_MODES = frozenset({"surrogate", "true", "permuted"})


@dataclass
class MultiscaleHybridOutput:
    """Exact decomposed prediction and optional selected local evidence.

    ``prediction``, each component prediction, and ``node_embedding`` contain
    only ``target_nodes`` when a target selection was supplied.  Selected
    local edge tensors preserve the order requested through
    ``local_contribution_edge_indices``.
    """

    prediction: Tensor
    node_embedding: Tensor
    self_prediction: Tensor
    regional_prediction: Tensor
    local_prediction: Tensor
    regional_routing: str
    local_routing: str
    local_contribution_edge_indices: Optional[Tensor] = None
    local_contribution_edge_index: Optional[Tensor] = None
    local_contribution_effective_source_index: Optional[Tensor] = None
    local_attention_weights: Optional[Tensor] = None
    local_edge_messages: Optional[Tensor] = None
    local_contributions: Optional[Tensor] = None
    local_contribution_gene_indices: Optional[Tensor] = None


@dataclass
class _ReceiverLayout:
    """Cached stable receiver ordering for one caller-owned graph tensor."""

    edge_index: Tensor
    tensor_version: int
    num_nodes: int
    edge_order: Optional[Tensor]
    receiver_ptr: tuple[int, ...]


@dataclass
class _BranchResult:
    """Internal branch aggregate and bounded selected-edge explanation."""

    node_message: Tensor
    selected_edge_ids: Optional[Tensor] = None
    selected_attention: Optional[Tensor] = None
    selected_edge_messages: Optional[Tensor] = None


def _validated_routing_mode(value: str, *, name: str) -> str:
    mode = str(value).strip().lower()
    if mode not in ROUTING_MODES:
        raise ValueError(
            f"{name} must be one of {sorted(ROUTING_MODES)}, got {value!r}"
        )
    return mode


def _normalize_index_selection(
    value: Tensor | Sequence[int],
    *,
    size: int,
    device: torch.device,
    name: str,
) -> Tensor:
    """Return a unique one-dimensional long selection in caller order."""

    selection = torch.as_tensor(value, device=device)
    if selection.dtype == torch.bool:
        if selection.ndim != 1 or selection.numel() != size:
            raise ValueError(f"a boolean {name} must have shape [{size}]")
        selection = selection.nonzero(as_tuple=False).flatten()
    else:
        if selection.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if selection.is_floating_point() or selection.is_complex():
            raise TypeError(f"{name} must use an integer dtype")
        selection = selection.to(dtype=torch.long)
    if selection.numel():
        if bool((selection < 0).any()) or bool((selection >= size).any()):
            raise ValueError(f"{name} contains an out-of-range index")
        if torch.unique(selection).numel() != selection.numel():
            raise ValueError(f"{name} must not contain duplicate indices")
    return selection


def _segment_softmax_and_sum(
    attention_logits: Tensor,
    edge_values: Tensor,
    lengths: Tensor,
    *,
    attention_dropout: float,
    training: bool,
) -> tuple[Tensor, Tensor]:
    """Exact softmax and signed aggregation for receiver-sorted edges."""

    if attention_logits.ndim != 2:
        raise ValueError("attention_logits must have shape [edges, heads]")
    if edge_values.ndim != 3:
        raise ValueError(
            "edge_values must have shape [edges, heads, value_dim]"
        )
    if (
        edge_values.shape[:2] != attention_logits.shape
        or lengths.ndim != 1
    ):
        raise ValueError("segment inputs are not aligned")
    num_receivers = lengths.numel()
    heads = attention_logits.shape[1]
    value_dim = edge_values.shape[2]
    if attention_logits.shape[0] == 0:
        return (
            attention_logits,
            edge_values.new_zeros((num_receivers, heads, value_dim)),
        )

    # Attention normalization, weighted values, and receiver aggregation use
    # FP32 under mixed precision.  This matches the high-degree graph contract
    # and avoids degree-dependent FP16/BF16 accumulation error.
    accumulation_logits = (
        attention_logits.float()
        if attention_logits.dtype in (torch.float16, torch.bfloat16)
        else attention_logits
    )
    accumulation_values = (
        edge_values.float()
        if edge_values.dtype in (torch.float16, torch.bfloat16)
        else edge_values
    )
    maxima = torch.segment_reduce(
        accumulation_logits,
        reduce="max",
        lengths=lengths,
    )
    centered = accumulation_logits - torch.repeat_interleave(
        maxima, lengths, dim=0
    )
    exponentiated = torch.exp(centered)
    denominators = torch.segment_reduce(
        exponentiated,
        reduce="sum",
        lengths=lengths,
    )
    attention = exponentiated / torch.repeat_interleave(
        denominators.clamp_min(torch.finfo(exponentiated.dtype).tiny),
        lengths,
        dim=0,
    )
    routed_attention = F.dropout(
        attention,
        p=attention_dropout,
        training=training,
    )
    aggregate = torch.segment_reduce(
        routed_attention.unsqueeze(-1) * accumulation_values,
        reduce="sum",
        lengths=lengths,
    )
    return routed_attention, aggregate


class _HurdlePredictionDecoder(nn.Module):
    """Unrestricted node decoder with a configurable channel count."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_genes: int,
        output_channels: int,
        decoder_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if decoder_dim <= 0:
            raise ValueError("decoder_dim must be positive")
        self.num_genes = int(num_genes)
        self.output_channels = int(output_channels)
        self.linear_in = nn.Linear(hidden_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(
            decoder_dim,
            num_genes * output_channels,
        )

    def forward(self, node_embedding: Tensor) -> Tensor:
        flat = self.linear_out(
            self.dropout(F.gelu(self.linear_in(node_embedding)))
        )
        return flat.reshape(
            node_embedding.shape[0],
            self.num_genes,
            self.output_channels,
        )


class _ExactReceiverChunkedRoutingBranch(nn.Module):
    """One exact, bounded-memory edge-conditioned routing branch."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        edge_attribute_dim: int,
        attention_heads: int,
        attention_head_dim: int,
        value_head_dim: int,
        message_dim: int,
        edge_hidden_dim: int,
        edge_embedding_dim: int,
        attention_dropout: float,
        receiver_chunk_size: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        for name, value in (
            ("hidden_dim", hidden_dim),
            ("edge_attribute_dim", edge_attribute_dim),
            ("attention_heads", attention_heads),
            ("attention_head_dim", attention_head_dim),
            ("value_head_dim", value_head_dim),
            ("message_dim", message_dim),
            ("edge_hidden_dim", edge_hidden_dim),
            ("edge_embedding_dim", edge_embedding_dim),
            ("receiver_chunk_size", receiver_chunk_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(attention_dropout) < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not isinstance(activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be boolean")

        self.hidden_dim = int(hidden_dim)
        self.edge_attribute_dim = int(edge_attribute_dim)
        self.attention_heads = int(attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.value_head_dim = int(value_head_dim)
        self.message_dim = int(message_dim)
        self.edge_embedding_dim = int(edge_embedding_dim)
        self.attention_dropout = float(attention_dropout)
        self.receiver_chunk_size = int(receiver_chunk_size)
        self.activation_checkpointing = activation_checkpointing

        attention_width = attention_heads * attention_head_dim
        value_width = attention_heads * value_head_dim
        self.query_projection = nn.Linear(
            hidden_dim, attention_width, bias=False
        )
        self.key_projection = nn.Linear(
            hidden_dim, attention_width, bias=False
        )
        self.value_projection = nn.Linear(
            hidden_dim, value_width, bias=False
        )
        self.edge_encoder = SharedEdgeEncoder(
            edge_attribute_dim=edge_attribute_dim,
            edge_embedding_dim=edge_embedding_dim,
            edge_hidden_dim=edge_hidden_dim,
        )
        self.edge_attention_projection = nn.Linear(
            edge_embedding_dim, attention_width, bias=False
        )
        self.attention_vector = nn.Parameter(
            torch.empty(attention_heads, attention_head_dim)
        )
        self.edge_value_gate = nn.Linear(
            edge_embedding_dim, value_width, bias=True
        )
        self.message_projection = nn.Linear(
            value_width, message_dim, bias=False
        )
        nn.init.xavier_uniform_(self.attention_vector)
        self._receiver_layout_cache: Optional[_ReceiverLayout] = None

    def clear_edge_layout_cache(self) -> None:
        """Discard derived ordering after a caller replaces or mutates a graph."""

        self._receiver_layout_cache = None

    def _apply(self, fn, recurse: bool = True):  # type: ignore[no-untyped-def]
        self.clear_edge_layout_cache()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self) -> dict[str, object]:
        state = super().__getstate__()
        state["_receiver_layout_cache"] = None
        return state

    def _receiver_layout(
        self,
        edge_index: Tensor,
        *,
        num_nodes: int,
    ) -> _ReceiverLayout:
        version = int(edge_index._version)
        cached = self._receiver_layout_cache
        if (
            cached is not None
            and cached.edge_index is edge_index
            and cached.tensor_version == version
            and cached.num_nodes == num_nodes
        ):
            return cached

        receiver = edge_index[1]
        counts = torch.bincount(receiver, minlength=num_nodes)
        cumulative = torch.cumsum(counts, dim=0).cpu().tolist()
        receiver_ptr = (0, *(int(value) for value in cumulative))
        if receiver.numel() < 2 or not bool(
            (receiver[1:] < receiver[:-1]).any()
        ):
            edge_order = None
        else:
            edge_order = torch.argsort(receiver, stable=True)
        layout = _ReceiverLayout(
            edge_index=edge_index,
            tensor_version=version,
            num_nodes=num_nodes,
            edge_order=edge_order,
            receiver_ptr=receiver_ptr,
        )
        self._receiver_layout_cache = layout
        return layout

    def _edge_values(
        self,
        *,
        source_values: Tensor,
        edge_embedding: Tensor,
    ) -> Tensor:
        gates = 1.0 + torch.tanh(
            self.edge_value_gate(edge_embedding).view(
                -1,
                self.attention_heads,
                self.value_head_dim,
            )
        )
        return source_values * gates

    def _true_chunk_routing(
        self,
        all_keys: Tensor,
        all_values: Tensor,
        receiver_queries: Tensor,
        source: Tensor,
        edge_attributes: Tensor,
        lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        edge_embedding = self.edge_encoder(edge_attributes)
        source_keys = all_keys.index_select(0, source)
        source_values = all_values.index_select(0, source)
        receiver_queries_by_edge = torch.repeat_interleave(
            receiver_queries,
            lengths,
            dim=0,
        )
        edge_attention = self.edge_attention_projection(
            edge_embedding
        ).view(
            -1,
            self.attention_heads,
            self.attention_head_dim,
        )
        attention_input = F.leaky_relu(
            receiver_queries_by_edge + source_keys + edge_attention,
            negative_slope=0.2,
        )
        attention_logits = (
            attention_input * self.attention_vector
        ).sum(dim=-1)
        edge_values = self._edge_values(
            source_values=source_values,
            edge_embedding=edge_embedding,
        )
        attention, aggregate = _segment_softmax_and_sum(
            attention_logits,
            edge_values,
            lengths,
            attention_dropout=self.attention_dropout,
            training=self.training,
        )
        return attention, edge_values, aggregate

    def _true_chunk_output(
        self,
        all_keys: Tensor,
        all_values: Tensor,
        receiver_queries: Tensor,
        source: Tensor,
        edge_attributes: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        _, _, aggregate = self._true_chunk_routing(
            all_keys,
            all_values,
            receiver_queries,
            source,
            edge_attributes,
            lengths,
        )
        return self.message_projection(
            aggregate.reshape(
                aggregate.shape[0],
                self.attention_heads * self.value_head_dim,
            )
        )

    def _true_chunk_selected(
        self,
        all_keys: Tensor,
        all_values: Tensor,
        receiver_queries: Tensor,
        source: Tensor,
        edge_attributes: Tensor,
        lengths: Tensor,
        selected: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        attention, edge_values, aggregate = self._true_chunk_routing(
            all_keys,
            all_values,
            receiver_queries,
            source,
            edge_attributes,
            lengths,
        )
        selected_attention = attention[selected]
        selected_values = edge_values[selected]
        selected_messages = self.message_projection(
            (
                selected_attention.unsqueeze(-1) * selected_values
            ).reshape(
                -1,
                self.attention_heads * self.value_head_dim,
            )
        )
        aggregate_message = self.message_projection(
            aggregate.reshape(
                aggregate.shape[0],
                self.attention_heads * self.value_head_dim,
            )
        )
        return aggregate_message, selected_attention, selected_messages

    def _surrogate_chunk(
        self,
        receiver_embedding: Tensor,
    ) -> Tensor:
        """Route two deterministic pseudo-messages from the same cell only."""

        num_receivers = receiver_embedding.shape[0]
        query = self.query_projection(receiver_embedding).view(
            num_receivers,
            self.attention_heads,
            self.attention_head_dim,
        )
        positive_key = self.key_projection(receiver_embedding).view(
            num_receivers,
            self.attention_heads,
            self.attention_head_dim,
        )
        positive_value = self.value_projection(receiver_embedding).view(
            num_receivers,
            self.attention_heads,
            self.value_head_dim,
        )
        keys = torch.stack((positive_key, -positive_key), dim=1).reshape(
            2 * num_receivers,
            self.attention_heads,
            self.attention_head_dim,
        )
        values = torch.stack(
            (positive_value, -positive_value), dim=1
        ).reshape(
            2 * num_receivers,
            self.attention_heads,
            self.value_head_dim,
        )

        indices = torch.arange(
            self.edge_attribute_dim,
            device=receiver_embedding.device,
            dtype=torch.long,
        ).remainder(self.hidden_dim)
        positive_attributes = receiver_embedding.index_select(1, indices)
        attributes = torch.stack(
            (positive_attributes, -positive_attributes), dim=1
        ).reshape(2 * num_receivers, self.edge_attribute_dim)
        edge_embedding = self.edge_encoder(attributes)
        edge_attention = self.edge_attention_projection(
            edge_embedding
        ).view(
            2 * num_receivers,
            self.attention_heads,
            self.attention_head_dim,
        )
        queries = query.unsqueeze(1).expand(-1, 2, -1, -1).reshape(
            2 * num_receivers,
            self.attention_heads,
            self.attention_head_dim,
        )
        logits = (
            F.leaky_relu(
                queries + keys + edge_attention,
                negative_slope=0.2,
            )
            * self.attention_vector
        ).sum(dim=-1)
        gated_values = self._edge_values(
            source_values=values,
            edge_embedding=edge_embedding,
        )
        lengths = torch.full(
            (num_receivers,),
            2,
            dtype=torch.long,
            device=receiver_embedding.device,
        )
        _, aggregate = _segment_softmax_and_sum(
            logits,
            gated_values,
            lengths,
            attention_dropout=self.attention_dropout,
            training=self.training,
        )
        return self.message_projection(
            aggregate.reshape(
                num_receivers,
                self.attention_heads * self.value_head_dim,
            )
        )

    def _run_surrogate(self, node_embedding: Tensor) -> _BranchResult:
        outputs: list[Tensor] = []
        use_checkpoint = (
            self.activation_checkpointing and torch.is_grad_enabled()
        )
        for start in range(0, node_embedding.shape[0], self.receiver_chunk_size):
            end = min(start + self.receiver_chunk_size, node_embedding.shape[0])
            receiver_chunk = node_embedding[start:end]
            if use_checkpoint:
                output = checkpoint(
                    self._surrogate_chunk,
                    receiver_chunk,
                    use_reentrant=False,
                )
            else:
                output = self._surrogate_chunk(receiver_chunk)
            outputs.append(output)
        node_message = (
            torch.cat(outputs, dim=0)
            if outputs
            else node_embedding.new_empty((0, self.message_dim))
        )
        return _BranchResult(node_message=node_message)

    def _run_true(
        self,
        node_embedding: Tensor,
        edge_index: Tensor,
        edge_attributes: Tensor,
        selected_edge_ids: Optional[Tensor],
        source_index_by_node: Optional[Tensor],
    ) -> _BranchResult:
        num_nodes = node_embedding.shape[0]
        layout = self._receiver_layout(edge_index, num_nodes=num_nodes)
        all_keys = self.key_projection(node_embedding).view(
            num_nodes,
            self.attention_heads,
            self.attention_head_dim,
        )
        all_values = self.value_projection(node_embedding).view(
            num_nodes,
            self.attention_heads,
            self.value_head_dim,
        )
        all_queries = self.query_projection(node_embedding).view(
            num_nodes,
            self.attention_heads,
            self.attention_head_dim,
        )

        selected_mask: Optional[Tensor] = None
        if selected_edge_ids is not None:
            selected_mask = torch.zeros(
                edge_index.shape[1],
                dtype=torch.bool,
                device=edge_index.device,
            )
            selected_mask[selected_edge_ids] = True

        outputs: list[Tensor] = []
        returned_ids: list[Tensor] = []
        selected_attention: list[Tensor] = []
        selected_messages: list[Tensor] = []
        use_checkpoint = (
            self.activation_checkpointing and torch.is_grad_enabled()
        )
        for receiver_start in range(
            0, num_nodes, self.receiver_chunk_size
        ):
            receiver_end = min(
                receiver_start + self.receiver_chunk_size,
                num_nodes,
            )
            edge_start = layout.receiver_ptr[receiver_start]
            edge_end = layout.receiver_ptr[receiver_end]
            if layout.edge_order is None:
                source = edge_index[0, edge_start:edge_end]
                chunk_attributes = edge_attributes[edge_start:edge_end]
                original_ids = torch.arange(
                    edge_start,
                    edge_end,
                    dtype=torch.long,
                    device=edge_index.device,
                )
            else:
                original_ids = layout.edge_order[edge_start:edge_end]
                source = edge_index[0].index_select(0, original_ids)
                chunk_attributes = edge_attributes.index_select(
                    0, original_ids
                )
            if source_index_by_node is not None:
                source = source_index_by_node.index_select(0, source)
            receiver_queries = all_queries[receiver_start:receiver_end]
            lengths = torch.tensor(
                [
                    layout.receiver_ptr[index + 1]
                    - layout.receiver_ptr[index]
                    for index in range(receiver_start, receiver_end)
                ],
                dtype=torch.long,
                device=edge_index.device,
            )
            common = (
                all_keys,
                all_values,
                receiver_queries,
                source,
                chunk_attributes,
                lengths,
            )
            if selected_mask is None:
                function = self._true_chunk_output
                if use_checkpoint:
                    output = checkpoint(
                        function,
                        *common,
                        use_reentrant=False,
                    )
                else:
                    output = function(*common)
                outputs.append(output)
                continue

            chunk_selected = selected_mask.index_select(0, original_ids)
            function = self._true_chunk_selected
            arguments = (*common, chunk_selected)
            if use_checkpoint:
                output, attention, messages = checkpoint(
                    function,
                    *arguments,
                    use_reentrant=False,
                )
            else:
                output, attention, messages = function(*arguments)
            outputs.append(output)
            returned_ids.append(original_ids[chunk_selected])
            selected_attention.append(attention)
            selected_messages.append(messages)

        node_message = (
            torch.cat(outputs, dim=0)
            if outputs
            else node_embedding.new_empty((0, self.message_dim))
        )
        if selected_edge_ids is None:
            return _BranchResult(node_message=node_message)

        edge_ids = (
            torch.cat(returned_ids, dim=0)
            if returned_ids
            else edge_index.new_empty((0,))
        )
        attention = (
            torch.cat(selected_attention, dim=0)
            if selected_attention
            else node_embedding.new_empty((0, self.attention_heads))
        )
        messages = (
            torch.cat(selected_messages, dim=0)
            if selected_messages
            else node_embedding.new_empty((0, self.message_dim))
        )
        returned_sort = torch.argsort(edge_ids)
        requested_sort = torch.argsort(selected_edge_ids)
        if not torch.equal(
            edge_ids.index_select(0, returned_sort),
            selected_edge_ids.index_select(0, requested_sort),
        ):
            raise RuntimeError("selected local edges were not restored exactly")
        restore = torch.empty_like(selected_edge_ids)
        restore[requested_sort] = returned_sort
        return _BranchResult(
            node_message=node_message,
            selected_edge_ids=edge_ids.index_select(0, restore),
            selected_attention=attention.index_select(0, restore),
            selected_edge_messages=messages.index_select(0, restore),
        )

    def forward(
        self,
        node_embedding: Tensor,
        *,
        routing: str,
        edge_index: Optional[Tensor],
        edge_attributes: Optional[Tensor],
        selected_edge_ids: Optional[Tensor] = None,
        source_index_by_node: Optional[Tensor] = None,
    ) -> _BranchResult:
        mode = _validated_routing_mode(routing, name="routing")
        if mode == "surrogate":
            if (
                edge_index is not None
                or edge_attributes is not None
                or source_index_by_node is not None
            ):
                raise ValueError(
                    "surrogate routing prohibits graph, edge, and source-map "
                    "inputs"
                )
            if selected_edge_ids is not None:
                raise ValueError(
                    "surrogate routing has no graph edges to explain"
                )
            return self._run_surrogate(node_embedding)

        if edge_index is None or edge_attributes is None:
            raise ValueError(
                f"{mode} routing requires edge_index and edge_attributes"
            )
        prepared_edges = _prepare_edge_index(
            edge_index,
            num_nodes=node_embedding.shape[0],
            device=node_embedding.device,
        )
        prepared_attributes = _prepare_edge_attributes(
            edge_attributes,
            num_edges=prepared_edges.shape[1],
            edge_attribute_dim=self.edge_attribute_dim,
            reference=node_embedding,
        )
        prepared_source_index: Optional[Tensor] = None
        if mode == "permuted":
            if source_index_by_node is None:
                raise ValueError(
                    "permuted routing requires source_index_by_node"
                )
            source_map = torch.as_tensor(
                source_index_by_node,
                device=node_embedding.device,
            )
            if (
                source_map.ndim != 1
                or source_map.numel() != node_embedding.shape[0]
                or source_map.dtype == torch.bool
                or source_map.is_floating_point()
                or source_map.is_complex()
            ):
                raise TypeError(
                    "source_index_by_node must be an integer node permutation"
                )
            prepared_source_index = source_map.to(
                dtype=torch.long,
            ).contiguous()
            if (
                bool((prepared_source_index < 0).any())
                or bool(
                    (
                        prepared_source_index
                        >= node_embedding.shape[0]
                    ).any()
                )
                or int(torch.unique(prepared_source_index).numel())
                != node_embedding.shape[0]
            ):
                raise ValueError(
                    "source_index_by_node must be a complete node permutation"
                )
        elif source_index_by_node is not None:
            raise ValueError(
                "only permuted routing accepts source_index_by_node"
            )
        return self._run_true(
            node_embedding,
            prepared_edges,
            prepared_attributes,
            selected_edge_ids,
            prepared_source_index,
        )


class MultiscaleAdditiveHybridModel(nn.Module):
    """Exact self + regional + signed-local hurdle-count predictor."""

    def __init__(
        self,
        *,
        num_genes: int,
        local_edge_attribute_dim: int,
        regional_edge_attribute_dim: int,
        expression_mean: Tensor | Sequence[float],
        expression_scale: Tensor | Sequence[float],
        node_covariate_dim: int = 0,
        hidden_dim: int = 512,
        decoder_dim: int = 512,
        ffn_dim: int = 512,
        attention_heads: int = 4,
        attention_head_dim: int = 32,
        value_head_dim: int = 16,
        message_dim: int = 64,
        edge_hidden_dim: int = 64,
        edge_embedding_dim: int = 32,
        output_channels: int = NUM_HURDLE_CONTINUOUS_CHANNELS,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        receiver_chunk_size: int = 128,
        activation_checkpointing: bool = True,
        regional_routing: RoutingMode = "surrogate",
        local_routing: RoutingMode = "surrogate",
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_genes", num_genes),
            ("hidden_dim", hidden_dim),
            ("decoder_dim", decoder_dim),
            ("ffn_dim", ffn_dim),
            ("output_channels", output_channels),
        ):
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(node_covariate_dim) < 0:
            raise ValueError("node_covariate_dim cannot be negative")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.num_genes = int(num_genes)
        self.node_covariate_dim = int(node_covariate_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_channels = int(output_channels)
        self.regional_routing = _validated_routing_mode(
            regional_routing,
            name="regional_routing",
        )
        if self.regional_routing == "permuted":
            raise ValueError(
                "regional_routing cannot use the local source permutation"
            )
        self.local_routing = _validated_routing_mode(
            local_routing,
            name="local_routing",
        )

        self.encoder = HybridCountNodeEncoder(
            num_genes=num_genes,
            node_covariate_dim=node_covariate_dim,
            hidden_dim=hidden_dim,
            expression_mean=expression_mean,
            expression_scale=expression_scale,
            dropout=dropout,
        )
        self.self_block = ResidualFeedForward(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.self_decoder = _HurdlePredictionDecoder(
            hidden_dim=hidden_dim,
            num_genes=num_genes,
            output_channels=output_channels,
            decoder_dim=decoder_dim,
            dropout=dropout,
        )

        common_branch = {
            "hidden_dim": hidden_dim,
            "attention_heads": attention_heads,
            "attention_head_dim": attention_head_dim,
            "value_head_dim": value_head_dim,
            "message_dim": message_dim,
            "edge_hidden_dim": edge_hidden_dim,
            "edge_embedding_dim": edge_embedding_dim,
            "attention_dropout": attention_dropout,
            "receiver_chunk_size": receiver_chunk_size,
            "activation_checkpointing": activation_checkpointing,
        }
        self.regional_branch = _ExactReceiverChunkedRoutingBranch(
            edge_attribute_dim=regional_edge_attribute_dim,
            **common_branch,
        )
        self.local_branch = _ExactReceiverChunkedRoutingBranch(
            edge_attribute_dim=local_edge_attribute_dim,
            **common_branch,
        )
        self.regional_output_projection = nn.Linear(
            message_dim,
            num_genes * output_channels,
            bias=False,
        )
        self.local_output_projection = nn.Linear(
            message_dim,
            num_genes * output_channels,
            bias=False,
        )
        nn.init.zeros_(self.regional_output_projection.weight)
        nn.init.zeros_(self.local_output_projection.weight)

    def _decode_branch(
        self,
        node_message: Tensor,
        projection: nn.Linear,
        targets: Optional[Tensor],
    ) -> Tensor:
        selected = _select_targets(node_message, targets)
        return projection(selected).reshape(
            selected.shape[0],
            self.num_genes,
            self.output_channels,
        )

    def decode_selected_local_contributions(
        self,
        edge_messages: Tensor,
        *,
        gene_indices: Optional[Tensor | Sequence[int]] = None,
    ) -> tuple[Tensor, Tensor]:
        """Decode selected low-rank messages without allocating all-edge output."""

        if (
            edge_messages.ndim != 2
            or edge_messages.shape[1] != self.local_branch.message_dim
        ):
            raise ValueError(
                "edge_messages must have shape "
                f"[selected_edges, {self.local_branch.message_dim}]"
            )
        if gene_indices is None:
            genes = torch.arange(
                self.num_genes,
                dtype=torch.long,
                device=edge_messages.device,
            )
        else:
            genes = _normalize_index_selection(
                gene_indices,
                size=self.num_genes,
                device=edge_messages.device,
                name="local_contribution_gene_indices",
            )
        channels = torch.arange(
            self.output_channels,
            dtype=torch.long,
            device=edge_messages.device,
        )
        rows = (
            genes.unsqueeze(1) * self.output_channels
            + channels.unsqueeze(0)
        ).reshape(-1)
        weight = self.local_output_projection.weight.index_select(0, rows)
        decoded = F.linear(edge_messages, weight, bias=None).reshape(
            edge_messages.shape[0],
            genes.numel(),
            self.output_channels,
        )
        return decoded, genes

    def forward(
        self,
        input_expression: Tensor,
        gene_mask: Tensor,
        *,
        node_covariates: Optional[Tensor] = None,
        regional_edge_index: Optional[Tensor] = None,
        regional_edge_attributes: Optional[Tensor] = None,
        local_edge_index: Optional[Tensor] = None,
        local_edge_attributes: Optional[Tensor] = None,
        local_source_index_by_node: Optional[Tensor] = None,
        target_nodes: Optional[Tensor | Sequence[int]] = None,
        local_contribution_edge_indices: Optional[
            Tensor | Sequence[int]
        ] = None,
        local_contribution_gene_indices: Optional[
            Tensor | Sequence[int]
        ] = None,
    ) -> MultiscaleHybridOutput:
        """Evaluate one routing arm on separately supplied regional/local graphs."""

        # HybridCountNodeEncoder applies both token and continuous masks inside
        # the model, so caller-supplied hidden values cannot enter any branch.
        node_embedding = self.encoder(
            input_expression,
            gene_mask,
            node_covariates,
        )
        targets = _normalize_target_nodes(
            target_nodes,
            num_nodes=node_embedding.shape[0],
            device=node_embedding.device,
        )

        selected_local_edges: Optional[Tensor] = None
        if local_contribution_edge_indices is not None:
            if local_edge_index is None:
                raise ValueError(
                    "local contribution selection requires local_edge_index"
                )
            selected_local_edges = _normalize_index_selection(
                local_contribution_edge_indices,
                size=local_edge_index.shape[1],
                device=node_embedding.device,
                name="local_contribution_edge_indices",
            )
        elif local_contribution_gene_indices is not None:
            raise ValueError(
                "local contribution genes require selected local edges"
            )

        regional_result = self.regional_branch(
            node_embedding,
            routing=self.regional_routing,
            edge_index=regional_edge_index,
            edge_attributes=regional_edge_attributes,
            source_index_by_node=None,
        )
        local_result = self.local_branch(
            node_embedding,
            routing=self.local_routing,
            edge_index=local_edge_index,
            edge_attributes=local_edge_attributes,
            selected_edge_ids=selected_local_edges,
            source_index_by_node=local_source_index_by_node,
        )

        selected_embedding = _select_targets(node_embedding, targets)
        self_prediction = self.self_decoder(
            self.self_block(selected_embedding)
        )
        regional_prediction = self._decode_branch(
            regional_result.node_message,
            self.regional_output_projection,
            targets,
        )
        local_prediction = self._decode_branch(
            local_result.node_message,
            self.local_output_projection,
            targets,
        )
        prediction = (
            self_prediction + regional_prediction + local_prediction
        )

        contribution_edge_index = None
        contribution_effective_source = None
        contribution_values = None
        contribution_genes = None
        if selected_local_edges is not None:
            if (
                local_result.selected_edge_ids is None
                or local_result.selected_edge_messages is None
                or local_result.selected_attention is None
                or local_edge_index is None
            ):
                raise RuntimeError(
                    "local branch did not return requested contributions"
                )
            contribution_edge_index = local_edge_index.to(
                device=node_embedding.device,
                dtype=torch.long,
            ).index_select(1, local_result.selected_edge_ids)
            contribution_effective_source = contribution_edge_index[0]
            if local_source_index_by_node is not None:
                source_map = torch.as_tensor(
                    local_source_index_by_node,
                    device=node_embedding.device,
                    dtype=torch.long,
                )
                contribution_effective_source = source_map.index_select(
                    0,
                    contribution_effective_source,
                )
            contribution_values, contribution_genes = (
                self.decode_selected_local_contributions(
                    local_result.selected_edge_messages,
                    gene_indices=local_contribution_gene_indices,
                )
            )

        return MultiscaleHybridOutput(
            prediction=prediction,
            node_embedding=selected_embedding,
            self_prediction=self_prediction,
            regional_prediction=regional_prediction,
            local_prediction=local_prediction,
            regional_routing=self.regional_routing,
            local_routing=self.local_routing,
            local_contribution_edge_indices=(
                local_result.selected_edge_ids
            ),
            local_contribution_edge_index=contribution_edge_index,
            local_contribution_effective_source_index=(
                contribution_effective_source
            ),
            local_attention_weights=local_result.selected_attention,
            local_edge_messages=local_result.selected_edge_messages,
            local_contributions=contribution_values,
            local_contribution_gene_indices=contribution_genes,
        )


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the exact number of trainable scalar parameters."""

    return sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "MultiscaleAdditiveHybridModel",
    "MultiscaleHybridOutput",
    "ROUTING_MODES",
    "RoutingMode",
    "trainable_parameter_count",
]
