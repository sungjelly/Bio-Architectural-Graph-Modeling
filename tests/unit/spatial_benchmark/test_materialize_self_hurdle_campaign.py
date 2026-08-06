from __future__ import annotations

from scripts.train.materialize_self_hurdle_campaign import (
    EXPECTED_PARAMETER_COUNT,
    _parameter_audit,
)


def test_large_model_component_has_frozen_parameter_count() -> None:
    audit = _parameter_audit(
        {
            "hidden_dim": 768,
            "decoder_dim": 768,
            "ffn_dim": 1536,
            "residual_blocks": 3,
            "dropout": 0.1,
        }
    )
    assert audit["trainable_parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert audit["uses_graph_inputs"] is False
    assert audit["uses_edge_inputs"] is False
    assert len(audit["named_parameter_shapes_sha256"]) == 64

