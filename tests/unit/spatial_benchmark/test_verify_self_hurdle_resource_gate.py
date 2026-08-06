from __future__ import annotations

import math

import pytest

from scripts.train.verify_self_hurdle_resource_gate import (
    SelfHurdleGateError,
    _finite,
)


@pytest.mark.parametrize("value", [True, math.inf, math.nan, "1.0"])
def test_resource_gate_rejects_nonfinite_or_non_numeric_values(
    value: object,
) -> None:
    with pytest.raises(SelfHurdleGateError, match="finite number"):
        _finite(value, "resource field")

