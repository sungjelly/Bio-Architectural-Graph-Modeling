r"""Target-isolated four-block geometry-modulated NB2 attention.

This module implements the target-wise conditioning contract used by the SO2
one-visit-per-cell experiment without duplicating a full core for every target.
For a contiguous target range ``[target_start, target_stop)`` it builds two
streams with the *same* shared node encoder:

* a clean, cell-autonomous source bank from every cell with an all-false mask;
* one separately supplied, pre-zeroed input row for each masked target.

The target state evolves through all four independently parameterized graph
blocks.  At every block, queries and residual/feed-forward state come from the
evolving target stream, while keys and values come from the immutable clean
source bank.  Only edges whose receiver is in the target range are evaluated.
Consequently simultaneous targets see one another's original clean expression,
but a target's withheld values have no computational route back to that target:
the target state is never used as a source and explicit self-loops are rejected.

The class subclasses the production receiver-chunked geometry-modulated NB2
model and registers no additional parameters.  Its encoder, four attention
blocks, decoder, and gene-shared inverse dispersion therefore have exactly the
same names, shapes, and parameter count as the production model.  Geometry is
used only for FP32 attention-score modulation/bias, never for values.
"""

from __future__ import annotations

from bisect import bisect_right
from functools import partial
from typing import Iterator, Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .geometry_modulated_relative_qkv_graph_transformer import (
    GeometryModulatedRelativeQKVGraphTransformerBlock,
)
from .negative_binomial import (
    NegativeBinomialModelOutput,
    ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer,
)
from .relative_qkv_graph_transformer import (
    _ReceiverLayout,
    _prepare_chunked_edge_index,
    _prepare_relative_geometry,
)


def _validate_target_range(
    target_start: int,
    target_stop: int,
    *,
    num_nodes: int,
) -> tuple[int, int]:
    for value, name in (
        (target_start, "target_start"),
        (target_stop, "target_stop"),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 0 <= target_start < target_stop <= num_nodes:
        raise ValueError(
            "target range must satisfy "
            "0 <= target_start < target_stop <= num_nodes"
        )
    return target_start, target_stop


class TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer(
    ReceiverChunkedGeometryModulatedNegativeBinomialGraphTransformer
):
    """Vectorize independent target tasks against one static clean source bank.

    ``target_input_expression`` and ``target_gene_mask`` both have shape
    ``[target_stop - target_start, num_genes]``.  Every masked target entry must
    already be exactly zero; every visible target entry must exactly match its
    clean row; and every target row must withhold at least one gene.  In contrast,
    ``input_expression`` and ``node_covariates`` contain the complete clean core,
    including clean source rows for simultaneously predicted targets.
    ``edge_index`` and ``relative_geometry`` must be CPU-resident, edge-aligned,
    and receiver-major.  Edges use ``edge_index[0] = source`` and
    ``edge_index[1] = receiver``.

    The returned NB2 mean and node embeddings contain only the contiguous target
    rows.  When requested, ``node_encoder_embedding`` is the masked target state
    before graph execution and ``graph_step_embeddings`` contains the four
    target-only states after the four independent graph blocks.  The inherited
    ``full_node_embedding`` field is target-only as well; the clean source bank
    is intentionally not exposed as a graph-evolved state.
    """

    target_conditioning = "masked_target_queries_static_clean_source_keys_values"
    source_bank_role = "clean_cell_autonomous_encoder_state"
    source_bank_evolves = False

    @staticmethod
    def _validate_target_mask(
        target_gene_mask: Tensor,
        *,
        num_targets: int,
        num_genes: int,
    ) -> None:
        if not isinstance(target_gene_mask, Tensor):
            raise TypeError("target_gene_mask must be a torch.Tensor")
        if target_gene_mask.dtype != torch.bool:
            raise TypeError("target_gene_mask must be boolean")
        expected = (num_targets, num_genes)
        if target_gene_mask.ndim != 2 or tuple(target_gene_mask.shape) != expected:
            raise ValueError(
                "target_gene_mask must have shape "
                f"{expected}, got {tuple(target_gene_mask.shape)}"
            )
        if not bool(target_gene_mask.any(dim=1).all()):
            raise ValueError("every target must withhold at least one gene")

    @staticmethod
    def _validate_target_input_expression(
        target_input_expression: Tensor,
        target_gene_mask: Tensor,
        *,
        input_expression: Tensor,
        num_targets: int,
        num_genes: int,
        target_start: int,
        target_stop: int,
    ) -> None:
        if not isinstance(target_input_expression, Tensor):
            raise TypeError("target_input_expression must be a torch.Tensor")
        expected = (num_targets, num_genes)
        if (
            target_input_expression.ndim != 2
            or tuple(target_input_expression.shape) != expected
        ):
            raise ValueError(
                "target_input_expression must have shape "
                f"{expected}, got {tuple(target_input_expression.shape)}"
            )
        if not target_input_expression.is_floating_point():
            raise TypeError("target_input_expression must be floating point")
        if (
            target_input_expression.device != input_expression.device
            or target_input_expression.dtype != input_expression.dtype
        ):
            raise ValueError(
                "target_input_expression must share input_expression's "
                "device and dtype"
            )
        device_mask = target_gene_mask.to(device=target_input_expression.device)
        withheld_values = target_input_expression.masked_select(device_mask)
        if not bool((withheld_values == 0).all()):
            raise ValueError(
                "target_input_expression must be exactly zero at every "
                "masked position"
            )
        clean_target_rows = input_expression[target_start:target_stop]
        visible = ~device_mask
        if not torch.equal(
            target_input_expression.masked_select(visible),
            clean_target_rows.masked_select(visible),
        ):
            raise ValueError(
                "every visible target_input_expression entry must exactly "
                "match its clean input_expression target row"
            )

    @staticmethod
    def _validate_cpu_receiver_major_graph(
        edge_index: Tensor,
        relative_geometry: Tensor,
    ) -> None:
        if not isinstance(edge_index, Tensor):
            raise TypeError("edge_index must be a torch.Tensor")
        if edge_index.device.type != "cpu":
            raise ValueError("edge_index must remain CPU-resident")
        if not isinstance(relative_geometry, Tensor):
            raise TypeError("relative_geometry must be a torch.Tensor")
        if relative_geometry.device.type != "cpu":
            raise ValueError("relative_geometry must remain CPU-resident")
        if edge_index.ndim == 2 and edge_index.shape[0] == 2:
            if edge_index.shape[1] and bool(
                (edge_index[0] == edge_index[1]).any()
            ):
                raise ValueError("explicit self loops are not allowed")
            receiver = edge_index[1]
            if receiver.numel() > 1 and bool(
                (receiver[1:] < receiver[:-1]).any()
            ):
                raise ValueError("edge_index must be receiver-major sorted")

    def _target_receiver_ranges(
        self,
        layout: _ReceiverLayout,
        *,
        target_start: int,
        target_stop: int,
    ) -> Iterator[tuple[int, int]]:
        """Yield exact target receiver shards under both configured limits."""

        receiver_start = target_start
        while receiver_start < target_stop:
            hard_stop = min(
                receiver_start + self.receiver_chunk_size,
                target_stop,
            )
            receiver_stop = hard_stop
            if self.max_edges_per_chunk is not None:
                edge_start = layout.receiver_ptr[receiver_start]
                edge_limit = edge_start + self.max_edges_per_chunk
                receiver_stop = min(
                    hard_stop,
                    max(
                        receiver_start + 1,
                        bisect_right(layout.receiver_ptr, edge_limit) - 1,
                    ),
                )
            yield receiver_start, receiver_stop
            receiver_start = receiver_stop

    @staticmethod
    def _project_target_queries_and_clean_sources(
        block: GeometryModulatedRelativeQKVGraphTransformerBlock,
        target_embedding: Tensor,
        clean_source_embedding: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project only the Q target stream and K/V clean source stream."""

        target_normalized = block.attention_normalization(target_embedding)
        source_normalized = block.attention_normalization(clean_source_embedding)
        target_shape = (
            target_embedding.shape[0],
            block.attention_heads,
            block.attention_head_dim,
        )
        source_shape = (
            clean_source_embedding.shape[0],
            block.attention_heads,
            block.attention_head_dim,
        )
        queries = block.query_projection(target_normalized).view(target_shape)
        keys = block.key_projection(source_normalized).view(source_shape)
        values = block.value_projection(source_normalized).view(source_shape)
        queries = F.normalize(
            queries.float(),
            p=2.0,
            dim=-1,
            eps=block.qk_normalization_epsilon,
        )
        keys = F.normalize(
            keys.float(),
            p=2.0,
            dim=-1,
            eps=block.qk_normalization_epsilon,
        )
        return queries, keys, values

    def _target_isolated_layer(
        self,
        block: GeometryModulatedRelativeQKVGraphTransformerBlock,
        target_embedding: Tensor,
        clean_source_embedding: Tensor,
        edge_index: Tensor,
        relative_geometry: Tensor,
        layout: _ReceiverLayout,
        *,
        target_start: int,
        target_stop: int,
    ) -> Tensor:
        queries, keys, values = self._project_target_queries_and_clean_sources(
            block,
            target_embedding,
            clean_source_embedding,
        )
        outputs: list[Tensor] = []
        use_checkpoint = self.activation_checkpointing and torch.is_grad_enabled()

        for receiver_start, receiver_stop in self._target_receiver_ranges(
            layout,
            target_start=target_start,
            target_stop=target_stop,
        ):
            edge_start = layout.receiver_ptr[receiver_start]
            edge_stop = layout.receiver_ptr[receiver_stop]
            chunk_edges_host = edge_index[:, edge_start:edge_stop]
            chunk_geometry_host = relative_geometry[edge_start:edge_stop]
            source = chunk_edges_host[0].to(
                device=target_embedding.device,
                dtype=torch.long,
            )
            receiver = chunk_edges_host[1].to(
                device=target_embedding.device,
                dtype=torch.long,
            )
            chunk_geometry = chunk_geometry_host.to(
                device=target_embedding.device,
                dtype=target_embedding.dtype,
            )
            local_start = receiver_start - target_start
            local_stop = receiver_stop - target_start
            common = (
                queries[local_start:local_stop],
                keys,
                values,
                target_embedding[local_start:local_stop],
                source,
                receiver,
                chunk_geometry,
            )
            chunk_function = partial(
                self._output_partition,
                block,
                receiver_start,
            )
            if use_checkpoint:
                output = checkpoint(
                    chunk_function,
                    *common,
                    use_reentrant=False,
                )
            else:
                output = chunk_function(*common)
            outputs.append(output)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        input_expression: Tensor,
        target_input_expression: Tensor,
        target_gene_mask: Tensor,
        edge_index: Tensor,
        relative_geometry: Tensor,
        node_covariates: Optional[Tensor] = None,
        *,
        target_start: int,
        target_stop: int,
        return_intermediate_embeddings: bool = False,
        return_graph_step_embeddings: bool = False,
    ) -> NegativeBinomialModelOutput:
        """Predict only ``[target_start, target_stop)`` under isolated masking."""

        if not isinstance(input_expression, Tensor):
            raise TypeError("input_expression must be a torch.Tensor")
        if input_expression.ndim != 2:
            raise ValueError(
                "input_expression must have shape [num_nodes, num_genes]"
            )
        num_nodes, num_genes = input_expression.shape
        if num_genes != self.num_genes:
            raise ValueError(
                f"expected {self.num_genes} genes, got {num_genes}"
            )
        target_start, target_stop = _validate_target_range(
            target_start,
            target_stop,
            num_nodes=num_nodes,
        )
        self._validate_target_mask(
            target_gene_mask,
            num_targets=target_stop - target_start,
            num_genes=num_genes,
        )
        self._validate_target_input_expression(
            target_input_expression,
            target_gene_mask,
            input_expression=input_expression,
            num_targets=target_stop - target_start,
            num_genes=num_genes,
            target_start=target_start,
            target_stop=target_stop,
        )
        self._validate_cpu_receiver_major_graph(edge_index, relative_geometry)

        prepared_edges = _prepare_chunked_edge_index(
            edge_index,
            num_nodes=num_nodes,
            default_device=input_expression.device,
        )
        prepared_geometry = _prepare_relative_geometry(
            relative_geometry,
            num_edges=prepared_edges.shape[1],
            reference=input_expression,
            move_to_reference=False,
        )
        if prepared_geometry.device != prepared_edges.device:
            raise ValueError(
                "edge_index and relative_geometry must share their CPU device"
            )
        # Sorting is checked before preparation, so receiver pointers address
        # contiguous, edge-aligned slices without allocating an edge permutation.
        layout = self._receiver_layout(prepared_edges, num_nodes=num_nodes)
        if layout.edge_order is not None:  # Defensive against future prep changes.
            raise RuntimeError("receiver-major graph unexpectedly required sorting")

        clean_mask = torch.zeros(
            input_expression.shape,
            dtype=torch.bool,
            device=input_expression.device,
        )
        clean_source_embedding = self.encoder(
            input_expression=input_expression,
            gene_mask=clean_mask,
            node_covariates=node_covariates,
        )
        target_covariates = (
            None
            if node_covariates is None
            else node_covariates[target_start:target_stop]
        )
        target_embedding = self.encoder(
            input_expression=target_input_expression,
            gene_mask=target_gene_mask,
            node_covariates=target_covariates,
        )
        node_encoder_embedding = (
            target_embedding if return_intermediate_embeddings else None
        )
        graph_step_embeddings: Optional[list[Tensor]] = (
            [] if return_graph_step_embeddings else None
        )

        for block in self.blocks:
            target_embedding = self._target_isolated_layer(
                block,
                target_embedding,
                clean_source_embedding,
                prepared_edges,
                prepared_geometry,
                layout,
                target_start=target_start,
                target_stop=target_stop,
            )
            if graph_step_embeddings is not None:
                graph_step_embeddings.append(target_embedding)

        prediction = self.decoder(target_embedding)
        return self._make_output(
            prediction=prediction,
            selected_embedding=target_embedding,
            full_embedding=target_embedding,
            node_encoder_embedding=node_encoder_embedding,
            graph_step_embeddings=(
                tuple(graph_step_embeddings)
                if graph_step_embeddings is not None
                else None
            ),
            edge_index=None,
            attention=None,
            content_logits=None,
            positional_bias=None,
            combined_logits=None,
            layer_number=None,
        )


# Short and receiver-explicit aliases for configuration/integration callers.
TargetIsolatedNegativeBinomialGraphTransformer = (
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer
)
ReceiverChunkedTargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer = (
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer
)


__all__ = [
    "ReceiverChunkedTargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer",
    "TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer",
    "TargetIsolatedNegativeBinomialGraphTransformer",
]
