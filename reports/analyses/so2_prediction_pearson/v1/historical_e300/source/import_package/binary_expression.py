"""Bernoulli-only expression detection for target-isolated SO2 training.

The model predicts whether each masked raw transcript count is positive.  Its
visible inputs remain the training-standardized continuous ``log1p`` values;
raw counts are accepted only by the loss helpers to construct binary labels.

The target-isolated geometry encoder and decoder trunk are identical to the
matched hurdle-NB2 model.  The existing gene-wise decoder projection is used
directly as the detection-logit head, and no count mean or dispersion
parameters are retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from .negative_binomial import _fp32_autocast_disabled
from .relative_qkv_graph_transformer import (
    RelativeGeometryQKVGraphTransformer,
    RelativeGeometryQKVModelOutput,
)
from .target_isolated_negative_binomial import (
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer,
)


BINARY_EXPRESSION_PROBABILITY_EPSILON = 1e-4


def _validate_probability_epsilon(value: float) -> float:
    value = float(value)
    if not 0.0 < value < 0.5:
        raise ValueError("detection probability epsilon must be in (0, 0.5)")
    return value


def _masked_binary_expression_inputs(
    nonzero_logits: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Validate and select logits and binary labels for masked entries."""

    if not all(isinstance(value, Tensor) for value in (nonzero_logits, raw_target, mask)):
        raise TypeError("nonzero_logits, raw_target, and mask must be tensors")
    if nonzero_logits.ndim != 2 or not all(nonzero_logits.shape):
        raise ValueError("nonzero_logits must be nonempty with shape [cells, genes]")
    if raw_target.shape != nonzero_logits.shape or mask.shape != nonzero_logits.shape:
        raise ValueError("raw_target and mask must have the same shape as nonzero_logits")
    if (
        raw_target.device != nonzero_logits.device
        or mask.device != nonzero_logits.device
    ):
        raise ValueError("nonzero_logits, raw_target, and mask must share a device")
    if not nonzero_logits.is_floating_point() or nonzero_logits.is_complex():
        raise TypeError("nonzero_logits must be a real floating tensor")
    if raw_target.dtype == torch.bool or raw_target.is_complex():
        raise TypeError("raw_target must contain real numeric counts")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be boolean")
    if not bool(mask.any()):
        raise ValueError("mask must select at least one entry")

    with _fp32_autocast_disabled(nonzero_logits):
        selected_logits = nonzero_logits.masked_select(mask).float()
        selected_target = raw_target.masked_select(mask).float()
        if not bool(torch.isfinite(selected_logits).all()):
            raise ValueError("masked nonzero logits must be finite")
        if not bool(torch.isfinite(selected_target).all()) or bool(
            (selected_target < 0.0).any()
        ):
            raise ValueError("masked raw targets must be finite and nonnegative")
        if bool((selected_target != selected_target.round()).any()):
            raise ValueError("masked raw targets must be integer-valued")
        binary_target = (selected_target > 0.0).float()
    return selected_logits, binary_target


@dataclass(frozen=True)
class MaskedBinaryExpressionLossSums:
    """Additive FP32 BCE statistics over one masked target shard."""

    masked_entries: int
    positive_entries: int
    detection_nll_sum: Tensor


def masked_binary_expression_bce_sums(
    nonzero_logits: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> MaskedBinaryExpressionLossSums:
    """Return summed Bernoulli NLL over exactly the masked entries.

    Every selected raw count contributes once.  A count greater than zero is a
    positive label; count magnitude has no effect on this objective.
    """

    selected_logits, binary_target = _masked_binary_expression_inputs(
        nonzero_logits,
        raw_target,
        mask,
    )
    with _fp32_autocast_disabled(nonzero_logits):
        detection_sum = F.binary_cross_entropy_with_logits(
            selected_logits,
            binary_target,
            reduction="sum",
        )
    if detection_sum.dtype != torch.float32 or not bool(torch.isfinite(detection_sum)):
        raise FloatingPointError("masked binary-expression BCE sum is non-finite")
    return MaskedBinaryExpressionLossSums(
        masked_entries=int(binary_target.numel()),
        positive_entries=int(binary_target.sum().item()),
        detection_nll_sum=detection_sum,
    )


def masked_binary_expression_bce(
    nonzero_logits: Tensor,
    raw_target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Return entry-pooled Bernoulli NLL over exactly the masked entries."""

    sums = masked_binary_expression_bce_sums(nonzero_logits, raw_target, mask)
    loss = sums.detection_nll_sum / float(sums.masked_entries)
    if loss.dtype != torch.float32 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("masked binary-expression BCE is non-finite")
    return loss


@dataclass(kw_only=True)
class BinaryExpressionModelOutput(RelativeGeometryQKVModelOutput):
    """Target output for Bernoulli expression-presence prediction."""

    nonzero_logits: Tensor
    nonzero_probability: Tensor
    predicted_zero_probability: Tensor


class TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer(
    TargetIsolatedGeometryModulatedNegativeBinomialGraphTransformer
):
    """Four-block target-isolated geometry model with a Bernoulli-only head.

    The inherited class supplies the audited target-isolated graph execution.
    Initialization immediately removes its otherwise-unused NB2 dispersion
    parameter and reinitializes the sole output projection as the detection
    head, so the resulting state dict contains no count-model parameters.
    """

    observation_model = "bernoulli_expression_presence"

    def __init__(
        self,
        *args: object,
        detection_initial_nonzero_probability: Optional[Tensor] = None,
        detection_probability_epsilon: float = (
            BINARY_EXPRESSION_PROBABILITY_EPSILON
        ),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        epsilon = _validate_probability_epsilon(detection_probability_epsilon)
        if detection_initial_nonzero_probability is None:
            probability = torch.full(
                (self.num_genes,),
                0.5,
                dtype=torch.float32,
            )
        else:
            probability = detection_initial_nonzero_probability.detach().float().clone()
        if probability.shape != (self.num_genes,):
            raise ValueError("initial nonzero probability must have shape [genes]")
        if not bool(torch.isfinite(probability).all()):
            raise ValueError("initial nonzero probabilities must be finite")
        probability.clamp_(epsilon, 1.0 - epsilon)

        # The inherited route registers one per-gene NB2 dispersion parameter.
        # It is not part of this observation model and must not enter the state
        # dict, optimizer, gradient telemetry, or parameter count.
        del self.raw_theta

        self.register_buffer(
            "detection_initial_nonzero_probability",
            probability.clone(),
            persistent=True,
        )
        self.detection_probability_epsilon = epsilon
        self.observation_model = "bernoulli_expression_presence"
        torch.nn.init.zeros_(self.decoder.linear_out.weight)
        with torch.no_grad():
            self.decoder.linear_out.bias.copy_(torch.logit(probability))

    @property
    def detection_head(self) -> torch.nn.Linear:
        """Return the sole gene-wise output projection for diagnostics."""

        return self.decoder.linear_out

    def _make_output(
        self,
        *,
        prediction: Tensor,
        selected_embedding: Tensor,
        full_embedding: Tensor,
        node_encoder_embedding: Optional[Tensor],
        graph_step_embeddings: Optional[tuple[Tensor, ...]],
        edge_index: Optional[Tensor],
        attention: Optional[Tensor],
        content_logits: Optional[Tensor],
        positional_bias: Optional[Tensor],
        combined_logits: Optional[Tensor],
        layer_number: Optional[int],
    ) -> BinaryExpressionModelOutput:
        expected_shape = (selected_embedding.shape[0], self.num_genes)
        if prediction.shape != expected_shape:
            raise RuntimeError(
                "binary-expression decoder must return "
                f"{expected_shape}, got {tuple(prediction.shape)}"
            )
        with _fp32_autocast_disabled(prediction):
            nonzero_logits = prediction.float()
            if not bool(torch.isfinite(nonzero_logits).all()):
                raise FloatingPointError("binary-expression logits are non-finite")
            nonzero_probability = torch.sigmoid(nonzero_logits)
            predicted_zero_probability = 1.0 - nonzero_probability
        if not bool(torch.isfinite(nonzero_probability).all()):
            raise FloatingPointError("binary-expression probabilities are non-finite")

        base_output = RelativeGeometryQKVGraphTransformer._make_output(
            self,
            prediction=nonzero_probability,
            selected_embedding=selected_embedding,
            full_embedding=full_embedding,
            node_encoder_embedding=node_encoder_embedding,
            graph_step_embeddings=graph_step_embeddings,
            edge_index=edge_index,
            attention=attention,
            content_logits=content_logits,
            positional_bias=positional_bias,
            combined_logits=combined_logits,
            layer_number=layer_number,
        )
        return BinaryExpressionModelOutput(
            **vars(base_output),
            nonzero_logits=nonzero_logits,
            nonzero_probability=nonzero_probability,
            predicted_zero_probability=predicted_zero_probability,
        )


__all__ = [
    "BINARY_EXPRESSION_PROBABILITY_EPSILON",
    "BinaryExpressionModelOutput",
    "MaskedBinaryExpressionLossSums",
    "TargetIsolatedGeometryModulatedBinaryExpressionGraphTransformer",
    "masked_binary_expression_bce",
    "masked_binary_expression_bce_sums",
]
