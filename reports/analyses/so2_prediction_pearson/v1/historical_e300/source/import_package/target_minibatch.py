"""Deterministic, balanced optimizer windows over canonical target-cell rows.

Window order has its own seed domain, independent of gene-mask sampling.
Windows preserve contiguous target indices for the target-isolated model;
they do not subsample or truncate the clean source-cell context.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from numbers import Integral
import random
from typing import Any


TARGET_WINDOW_PLAN_SCHEMA = "target_cell_balanced_window_plan_v1"
WINDOW_ORDER_SEED_DOMAIN = "bagm.target_cell_minibatch.window_order.v1"


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _canonical_alias(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("core_alias must be a string")
    alias = value.strip().upper()
    if not alias:
        raise ValueError("core_alias must be non-empty")
    return alias


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _shuffle_seed(*, seed: int, core_alias: str, global_epoch: int) -> int:
    payload = {
        "domain": WINDOW_ORDER_SEED_DOMAIN,
        "seed": _integer(seed, name="seed", minimum=0),
        "core_alias": _canonical_alias(core_alias),
        "global_epoch": _integer(global_epoch, name="global_epoch", minimum=0),
    }
    return int.from_bytes(hashlib.sha256(_canonical_bytes(payload)).digest(), "big")


def target_window_plan(
    n_cells: int,
    batch_size: int,
    *,
    seed: int,
    core_alias: str,
    global_epoch: int,
) -> tuple[tuple[int, int], ...]:
    """Return a shuffled exhaustive plan of half-open target-row windows.

There are exactly ``ceil(n_cells / batch_size)`` nonempty windows. Their
sizes differ by at most one and never exceed the nominal batch size. This
balances a core's final remainder across all its windows, avoiding a tiny
tail update while retaining every target exactly once. The integer epoch
is used as supplied, so callers must consistently use their epoch convention.
The local RNG neither reads nor changes global Python or mask RNG state.
"""

    n_cells = _integer(n_cells, name="n_cells", minimum=1)
    batch_size = _integer(batch_size, name="batch_size", minimum=1)
    shuffle_seed = _shuffle_seed(
        seed=seed, core_alias=core_alias, global_epoch=global_epoch,
    )
    window_count = (n_cells + batch_size - 1) // batch_size
    quotient, remainder = divmod(n_cells, window_count)
    windows: list[tuple[int, int]] = []
    start = 0
    for index in range(window_count):
        stop = start + quotient + int(index < remainder)
        windows.append((start, stop))
        start = stop
    random.Random(shuffle_seed).shuffle(windows)
    return tuple(windows)


def expected_updates_per_epoch(core_counts: Sequence[int], batch_size: int) -> int:
    """Count one optimizer update per nonempty window in every core."""

    batch_size = _integer(batch_size, name="batch_size", minimum=1)
    if not isinstance(core_counts, Sequence) or isinstance(core_counts, (str, bytes)):
        raise TypeError("core_counts must be a sequence")
    if not core_counts:
        raise ValueError("core_counts must be non-empty")
    return sum(
        (_integer(count, name="core count", minimum=1) + batch_size - 1)
        // batch_size
        for count in core_counts
    )


def validate_target_windows(windows: Sequence[tuple[int, int]], n_cells: int) -> None:
    """Reject empty, overlapping, repeated, missing or out-of-bounds rows.

Execution order may be shuffled. Validation sorts the intervals, without
materializing a cell-sized coverage vector, and requires exact adjacency.
This validates exhaustive coverage; it does not require balanced sizes.
"""

    n_cells = _integer(n_cells, name="n_cells", minimum=1)
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise TypeError("windows must be a sequence")
    if not windows:
        raise ValueError("windows must be non-empty")
    checked: list[tuple[int, int]] = []
    for window in windows:
        if (
            not isinstance(window, Sequence)
            or isinstance(window, (str, bytes))
            or len(window) != 2
        ):
            raise TypeError("each window must contain exactly two integer bounds")
        start = _integer(window[0], name="window start", minimum=0)
        stop = _integer(window[1], name="window stop", minimum=0)
        if start >= stop or stop > n_cells:
            raise ValueError("window is empty, reversed or out of bounds")
        checked.append((start, stop))
    cursor = 0
    for start, stop in sorted(checked):
        if start != cursor:
            raise ValueError("windows contain a gap, duplicate or overlap")
        cursor = stop
    if cursor != n_cells:
        raise ValueError("windows do not cover all target cells")


def window_plan_receipt(
    n_cells: int,
    batch_size: int,
    *,
    seed: int,
    core_alias: str,
    global_epoch: int,
) -> dict[str, Any]:
    """Return the deterministic plan and its canonical SHA-256 receipt.

``plan_sha256`` hashes the complete receipt except that field, using UTF-8
JSON with sorted keys, compact separators, ASCII escaping and no NaNs.
The hash binds execution order as well as coverage and seed provenance.
"""

    windows = target_window_plan(
        n_cells, batch_size, seed=seed, core_alias=core_alias,
        global_epoch=global_epoch,
    )
    validate_target_windows(windows, n_cells)
    receipt: dict[str, Any] = {
        "schema": TARGET_WINDOW_PLAN_SCHEMA,
        "n_cells": int(n_cells),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "core_alias": _canonical_alias(core_alias),
        "global_epoch": int(global_epoch),
        "shuffle_seed_domain": WINDOW_ORDER_SEED_DOMAIN,
        "shuffle_seed": _shuffle_seed(
            seed=seed, core_alias=core_alias, global_epoch=global_epoch,
        ),
        "shuffle_algorithm": "python_random_Random_shuffle",
        "window_count": len(windows),
        "windows": [list(window) for window in windows],
    }
    receipt["plan_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    return receipt
