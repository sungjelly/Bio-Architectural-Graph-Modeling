"""Small exact-inference utilities for paired biological-unit effects.

The functions in this module deliberately operate on one value per
independent unit.  They do not accept cell-level weights or technical-repeat
weights, which helps keep campaign comparators from accidentally treating
cells or fixed mask replicates as biological replicates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExactSignFlipResult:
    """Result of an exhaustive one-sided paired sign-flip test."""

    alternative: str
    statistic: float
    p_value: float
    unit_count: int
    permutation_count: int
    favorable_units: int
    unfavorable_units: int
    tied_units: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_vector(values: Sequence[float], *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{label} must be a nonempty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def exact_one_sided_paired_sign_flip(
    differences: Sequence[float],
) -> ExactSignFlipResult:
    """Test whether the mean paired difference is greater than zero.

    All ``2**n`` sign assignments are enumerated.  Zero differences remain
    in the enumeration and therefore do not spuriously reduce the nominal
    sample size.  The p-value is the exact tail probability including the
    observed assignment; no asymptotic or Monte Carlo approximation is used.
    """

    values = _finite_vector(differences, label="differences")
    if values.size > 20:
        raise ValueError(
            "exhaustive sign-flip enumeration is limited to 20 paired units"
        )
    observed = float(np.mean(values))
    extreme = 0
    permutations = 1 << int(values.size)
    # A scale-aware tolerance prevents round-off from excluding an exactly
    # tied signed mean while leaving scientifically meaningful differences
    # unchanged.
    tolerance = 32.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(values)))
    )
    for signs in itertools.product((-1.0, 1.0), repeat=int(values.size)):
        statistic = float(np.mean(values * np.asarray(signs)))
        if statistic >= observed - tolerance:
            extreme += 1
    return ExactSignFlipResult(
        alternative="mean_difference_greater_than_zero",
        statistic=observed,
        p_value=extreme / permutations,
        unit_count=int(values.size),
        permutation_count=permutations,
        favorable_units=int(np.count_nonzero(values > 0)),
        unfavorable_units=int(np.count_nonzero(values < 0)),
        tied_units=int(np.count_nonzero(values == 0)),
    )


def holm_adjust(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, dict[str, float | bool | int]]:
    """Apply Holm's step-down family-wise adjustment to named hypotheses."""

    if not p_values:
        raise ValueError("p_values cannot be empty")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    validated: list[tuple[str, float]] = []
    for name, raw in p_values.items():
        value = float(raw)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"p-value for {name!r} must lie in [0, 1]")
        validated.append((str(name), value))
    validated.sort(key=lambda item: (item[1], item[0]))
    family_size = len(validated)
    adjusted_running = 0.0
    result: dict[str, dict[str, float | bool | int]] = {}
    continue_rejecting = True
    for rank, (name, raw) in enumerate(validated, start=1):
        multiplier = family_size - rank + 1
        adjusted_running = max(adjusted_running, multiplier * raw)
        adjusted = min(1.0, adjusted_running)
        threshold = alpha / multiplier
        rejected = continue_rejecting and raw <= threshold
        if not rejected:
            continue_rejecting = False
        result[name] = {
            "raw_p_value": raw,
            "holm_adjusted_p_value": adjusted,
            "holm_rank": rank,
            "holm_threshold": threshold,
            "reject": rejected,
        }
    return result


def paired_relative_improvements(
    reference: Sequence[float],
    candidate: Sequence[float],
) -> np.ndarray:
    """Return ``(reference - candidate) / reference`` per paired unit."""

    baseline = _finite_vector(reference, label="reference")
    proposed = _finite_vector(candidate, label="candidate")
    if baseline.shape != proposed.shape:
        raise ValueError("reference and candidate must have identical shape")
    if np.any(baseline <= 0):
        raise ValueError("relative loss improvement requires positive references")
    return (baseline - proposed) / baseline


__all__ = [
    "ExactSignFlipResult",
    "exact_one_sided_paired_sign_flip",
    "holm_adjust",
    "paired_relative_improvements",
]
