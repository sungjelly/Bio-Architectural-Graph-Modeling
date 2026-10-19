"""Additive metrics and durable CSVs for target-cell-once NB training.

The objects in this module contain no model or distributed-runtime code.  A
rank records additive negative-binomial statistics for its canonical,
contiguous target interval, then rank zero validates and pools the completed
payloads.  CSV publication is an atomic, idempotent projection that can be
trimmed back to the last durable checkpoint before deterministic replay.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import numbers
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


RANK_METRIC_SCHEMA = "target_cell_rank_core_metrics_v1"
EPOCH_AGGREGATE_SCHEMA = "target_cell_epoch_metrics_v1"


class TargetCellTrainingError(RuntimeError):
    """Raised when target coverage, metric, or persistence invariants fail."""


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TargetCellTrainingError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise TargetCellTrainingError(f"{name} must be at least {minimum}.")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TargetCellTrainingError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise TargetCellTrainingError(f"{name} must be finite and non-negative.")
    return result


def _alias(value: object, *, name: str = "core_alias") -> str:
    if not isinstance(value, str):
        raise TargetCellTrainingError(f"{name} must be a string.")
    result = value.strip().upper()
    if not result:
        raise TargetCellTrainingError(f"{name} cannot be empty.")
    return result


def _canonical_interval(
    n_cells: int,
    world_size: int,
    rank: int,
) -> tuple[int, int]:
    quotient, remainder = divmod(n_cells, world_size)
    start = rank * quotient + min(rank, remainder)
    return start, start + quotient + int(rank < remainder)


@dataclass(frozen=True, slots=True)
class RankCoreMetrics:
    """Completed additive metrics for one core's canonical rank interval."""

    core_alias: str
    rank: int
    world_size: int
    n_cells: int
    target_start: int
    target_stop: int
    target_visit_counts: tuple[int, ...]
    negative_binomial_nll_sum: float
    masked_entry_count: int

    def __post_init__(self) -> None:
        alias = _alias(self.core_alias)
        world_size = _integer(
            self.world_size, name="world_size", minimum=1
        )
        rank = _integer(self.rank, name="rank")
        n_cells = _integer(self.n_cells, name="n_cells", minimum=1)
        start = _integer(self.target_start, name="target_start")
        stop = _integer(self.target_stop, name="target_stop")
        if rank >= world_size:
            raise TargetCellTrainingError("rank must be less than world_size.")
        expected = _canonical_interval(n_cells, world_size, rank)
        if (start, stop) != expected:
            raise TargetCellTrainingError(
                "Target interval is not the canonical contiguous rank split."
            )
        if isinstance(self.target_visit_counts, (str, bytes)):
            raise TargetCellTrainingError(
                "target_visit_counts must be an integer sequence."
            )
        try:
            visits = tuple(
                _integer(value, name="target_visit_count")
                for value in self.target_visit_counts
            )
        except TypeError as exc:
            raise TargetCellTrainingError(
                "target_visit_counts must be an integer sequence."
            ) from exc
        if len(visits) != stop - start:
            raise TargetCellTrainingError(
                "Target visit counts do not match the rank interval."
            )
        if any(count != 1 for count in visits):
            raise TargetCellTrainingError(
                "Every target cell must be visited exactly once."
            )
        nll_sum = _finite_nonnegative(
            self.negative_binomial_nll_sum,
            name="negative_binomial_nll_sum",
        )
        masked_entries = _integer(
            self.masked_entry_count, name="masked_entry_count"
        )
        if visits and masked_entries < len(visits):
            raise TargetCellTrainingError(
                "Each visited target must contribute a nonempty mask."
            )
        if not visits and (masked_entries != 0 or nll_sum != 0.0):
            raise TargetCellTrainingError(
                "An empty rank interval cannot contribute loss statistics."
            )

        object.__setattr__(self, "core_alias", alias)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "n_cells", n_cells)
        object.__setattr__(self, "target_start", start)
        object.__setattr__(self, "target_stop", stop)
        object.__setattr__(self, "target_visit_counts", visits)
        object.__setattr__(self, "negative_binomial_nll_sum", nll_sum)
        object.__setattr__(self, "masked_entry_count", masked_entries)

    @property
    def target_visit_count(self) -> int:
        return int(sum(self.target_visit_counts))

    @property
    def pooled_negative_binomial_nll(self) -> float | None:
        if self.masked_entry_count == 0:
            return None
        return self.negative_binomial_nll_sum / self.masked_entry_count

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": RANK_METRIC_SCHEMA,
            "core_alias": self.core_alias,
            "rank": self.rank,
            "world_size": self.world_size,
            "n_cells": self.n_cells,
            "target_start": self.target_start,
            "target_stop": self.target_stop,
            "target_visit_counts": list(self.target_visit_counts),
            "negative_binomial_nll_sum": self.negative_binomial_nll_sum,
            "masked_entry_count": self.masked_entry_count,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "RankCoreMetrics":
        if not isinstance(payload, Mapping):
            raise TargetCellTrainingError("Rank metrics payload must be a mapping.")
        fields = {
            "schema",
            "core_alias",
            "rank",
            "world_size",
            "n_cells",
            "target_start",
            "target_stop",
            "target_visit_counts",
            "negative_binomial_nll_sum",
            "masked_entry_count",
        }
        if set(payload) != fields or payload.get("schema") != RANK_METRIC_SCHEMA:
            raise TargetCellTrainingError("Rank metrics payload schema mismatch.")
        return cls(
            core_alias=payload["core_alias"],
            rank=payload["rank"],
            world_size=payload["world_size"],
            n_cells=payload["n_cells"],
            target_start=payload["target_start"],
            target_stop=payload["target_stop"],
            target_visit_counts=payload["target_visit_counts"],
            negative_binomial_nll_sum=payload[
                "negative_binomial_nll_sum"
            ],
            masked_entry_count=payload["masked_entry_count"],
        )


class TargetRankMetricAccumulator:
    """Accumulate batches for one rank/core and finalize exact target coverage."""

    def __init__(
        self,
        *,
        core_alias: str,
        n_cells: int,
        world_size: int,
        rank: int,
        target_start: int | None = None,
        target_stop: int | None = None,
    ) -> None:
        self.core_alias = _alias(core_alias)
        self.n_cells = _integer(n_cells, name="n_cells", minimum=1)
        self.world_size = _integer(
            world_size, name="world_size", minimum=1
        )
        self.rank = _integer(rank, name="rank")
        if self.rank >= self.world_size:
            raise TargetCellTrainingError("rank must be less than world_size.")
        expected_start, expected_stop = _canonical_interval(
            self.n_cells, self.world_size, self.rank
        )
        start = (
            expected_start
            if target_start is None
            else _integer(target_start, name="target_start")
        )
        stop = (
            expected_stop
            if target_stop is None
            else _integer(target_stop, name="target_stop")
        )
        if (start, stop) != (expected_start, expected_stop):
            raise TargetCellTrainingError(
                "Target interval is not the canonical contiguous rank split."
            )
        self.target_start = start
        self.target_stop = stop
        self._target_visit_counts = [0] * (stop - start)
        self._negative_binomial_nll_terms: list[float] = []
        self._masked_entry_count = 0

    def add(
        self,
        target_indices: Sequence[int],
        *,
        negative_binomial_nll_sum: float,
        masked_entry_count: int,
    ) -> None:
        """Add one target batch after checking it cannot duplicate a visit."""

        if isinstance(target_indices, (str, bytes)):
            raise TargetCellTrainingError(
                "target_indices must be a nonempty integer sequence."
            )
        try:
            targets = tuple(
                _integer(value, name="target_index")
                for value in target_indices
            )
        except TypeError as exc:
            raise TargetCellTrainingError(
                "target_indices must be a nonempty integer sequence."
            ) from exc
        if not targets:
            raise TargetCellTrainingError("A metric batch must contain targets.")
        if len(set(targets)) != len(targets):
            raise TargetCellTrainingError(
                "A target cannot be repeated within a metric batch."
            )
        offsets = []
        for target in targets:
            if not self.target_start <= target < self.target_stop:
                raise TargetCellTrainingError(
                    "Target is outside this rank's contiguous interval."
                )
            offset = target - self.target_start
            if self._target_visit_counts[offset] != 0:
                raise TargetCellTrainingError(
                    "A target cell cannot be visited more than once."
                )
            offsets.append(offset)
        nll_sum = _finite_nonnegative(
            negative_binomial_nll_sum,
            name="negative_binomial_nll_sum",
        )
        masked_entries = _integer(
            masked_entry_count, name="masked_entry_count", minimum=1
        )
        if masked_entries < len(targets):
            raise TargetCellTrainingError(
                "Each target must contribute at least one masked entry."
            )
        for offset in offsets:
            self._target_visit_counts[offset] = 1
        self._negative_binomial_nll_terms.append(nll_sum)
        self._masked_entry_count += masked_entries

    def finalize(self) -> RankCoreMetrics:
        """Return a transport payload only after every local target was seen."""

        if any(count != 1 for count in self._target_visit_counts):
            raise TargetCellTrainingError(
                "Cannot finalize before every local target is visited exactly once."
            )
        return RankCoreMetrics(
            core_alias=self.core_alias,
            rank=self.rank,
            world_size=self.world_size,
            n_cells=self.n_cells,
            target_start=self.target_start,
            target_stop=self.target_stop,
            target_visit_counts=tuple(self._target_visit_counts),
            negative_binomial_nll_sum=math.fsum(
                self._negative_binomial_nll_terms
            ),
            masked_entry_count=self._masked_entry_count,
        )


@dataclass(frozen=True, slots=True)
class CoreMetrics:
    """Deterministically pooled rank statistics for one fully covered core."""

    core_alias: str
    n_cells: int
    world_size: int
    negative_binomial_nll_sum: float
    masked_entry_count: int
    target_visit_count: int

    def __post_init__(self) -> None:
        alias = _alias(self.core_alias)
        n_cells = _integer(self.n_cells, name="n_cells", minimum=1)
        world_size = _integer(
            self.world_size, name="world_size", minimum=1
        )
        nll_sum = _finite_nonnegative(
            self.negative_binomial_nll_sum,
            name="negative_binomial_nll_sum",
        )
        masked_entries = _integer(
            self.masked_entry_count, name="masked_entry_count", minimum=1
        )
        visits = _integer(
            self.target_visit_count, name="target_visit_count", minimum=1
        )
        if visits != n_cells:
            raise TargetCellTrainingError(
                "Core target coverage must be exact and exhaustive."
            )
        if masked_entries < visits:
            raise TargetCellTrainingError(
                "Each core target must contribute a nonempty mask."
            )
        object.__setattr__(self, "core_alias", alias)
        object.__setattr__(self, "n_cells", n_cells)
        object.__setattr__(self, "world_size", world_size)
        object.__setattr__(self, "negative_binomial_nll_sum", nll_sum)
        object.__setattr__(self, "masked_entry_count", masked_entries)
        object.__setattr__(self, "target_visit_count", visits)

    @property
    def pooled_negative_binomial_nll(self) -> float:
        return self.negative_binomial_nll_sum / self.masked_entry_count

    def to_payload(self) -> dict[str, Any]:
        return {
            "core_alias": self.core_alias,
            "n_cells": self.n_cells,
            "world_size": self.world_size,
            "negative_binomial_nll_sum": self.negative_binomial_nll_sum,
            "masked_entry_count": self.masked_entry_count,
            "target_visit_count": self.target_visit_count,
            "pooled_negative_binomial_nll": (
                self.pooled_negative_binomial_nll
            ),
            "coverage_exact_disjoint_exhaustive": True,
        }


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Pooled epoch statistics in deterministic core order."""

    per_core: tuple[CoreMetrics, ...]

    def __post_init__(self) -> None:
        if not self.per_core or any(
            not isinstance(item, CoreMetrics) for item in self.per_core
        ):
            raise TargetCellTrainingError(
                "Epoch metrics require at least one CoreMetrics value."
            )
        aliases = tuple(item.core_alias for item in self.per_core)
        if aliases != tuple(sorted(aliases)) or len(set(aliases)) != len(aliases):
            raise TargetCellTrainingError(
                "Per-core metrics must be unique and canonically ordered."
            )
        if len({item.world_size for item in self.per_core}) != 1:
            raise TargetCellTrainingError(
                "All core metrics must use the same world_size."
            )

    @property
    def negative_binomial_nll_sum(self) -> float:
        return math.fsum(
            item.negative_binomial_nll_sum for item in self.per_core
        )

    @property
    def masked_entry_count(self) -> int:
        return int(sum(item.masked_entry_count for item in self.per_core))

    @property
    def target_visit_count(self) -> int:
        return int(sum(item.target_visit_count for item in self.per_core))

    @property
    def pooled_negative_binomial_nll(self) -> float:
        return self.negative_binomial_nll_sum / self.masked_entry_count

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": EPOCH_AGGREGATE_SCHEMA,
            "core_aliases": [item.core_alias for item in self.per_core],
            "negative_binomial_nll_sum": self.negative_binomial_nll_sum,
            "masked_entry_count": self.masked_entry_count,
            "target_visit_count": self.target_visit_count,
            "pooled_negative_binomial_nll": (
                self.pooled_negative_binomial_nll
            ),
            "coverage_exact_disjoint_exhaustive": True,
            "per_core": [item.to_payload() for item in self.per_core],
        }


def aggregate_rank_core_metrics(
    payloads: Sequence[RankCoreMetrics | Mapping[str, Any]],
    *,
    expected_core_aliases: Sequence[str] | None = None,
) -> EpochMetrics:
    """Validate and pool completed rank payloads in ``(alias, rank)`` order."""

    if isinstance(payloads, (str, bytes)):
        raise TargetCellTrainingError("Rank payloads must be a sequence.")
    materialized: list[RankCoreMetrics] = []
    try:
        for payload in payloads:
            materialized.append(
                payload
                if isinstance(payload, RankCoreMetrics)
                else RankCoreMetrics.from_payload(payload)
            )
    except TypeError as exc:
        raise TargetCellTrainingError("Rank payloads must be a sequence.") from exc
    if not materialized:
        raise TargetCellTrainingError("At least one rank payload is required.")

    observed_aliases = {item.core_alias for item in materialized}
    if expected_core_aliases is not None:
        if isinstance(expected_core_aliases, (str, bytes)):
            raise TargetCellTrainingError(
                "expected_core_aliases must be a sequence."
            )
        try:
            expected = tuple(
                _alias(value, name="expected_core_alias")
                for value in expected_core_aliases
            )
        except TypeError as exc:
            raise TargetCellTrainingError(
                "expected_core_aliases must be a sequence."
            ) from exc
        if not expected or len(set(expected)) != len(expected):
            raise TargetCellTrainingError(
                "Expected core aliases must be nonempty and unique."
            )
        if observed_aliases != set(expected):
            raise TargetCellTrainingError(
                "Rank payloads do not cover the expected cores exactly."
            )
    aliases = tuple(sorted(observed_aliases))
    world_sizes = {item.world_size for item in materialized}
    if len(world_sizes) != 1:
        raise TargetCellTrainingError(
            "Every rank/core payload must use the same world_size."
        )
    world_size = next(iter(world_sizes))
    pooled_cores: list[CoreMetrics] = []
    for alias in aliases:
        parts = sorted(
            (item for item in materialized if item.core_alias == alias),
            key=lambda item: item.rank,
        )
        if len(parts) != world_size or [part.rank for part in parts] != list(
            range(world_size)
        ):
            raise TargetCellTrainingError(
                f"Core {alias} must contain every rank exactly once."
            )
        n_cells = parts[0].n_cells
        if any(part.n_cells != n_cells for part in parts):
            raise TargetCellTrainingError(
                f"Core {alias} has inconsistent cell counts."
            )
        cursor = 0
        for part in parts:
            if part.target_start != cursor:
                raise TargetCellTrainingError(
                    f"Core {alias} rank intervals are not contiguous."
                )
            cursor = part.target_stop
        if cursor != n_cells:
            raise TargetCellTrainingError(
                f"Core {alias} rank coverage is not exhaustive."
            )
        visits = sum(part.target_visit_count for part in parts)
        if visits != n_cells:
            raise TargetCellTrainingError(
                f"Core {alias} target visits are not exact."
            )
        pooled_cores.append(
            CoreMetrics(
                core_alias=alias,
                n_cells=n_cells,
                world_size=world_size,
                negative_binomial_nll_sum=math.fsum(
                    part.negative_binomial_nll_sum for part in parts
                ),
                masked_entry_count=sum(
                    part.masked_entry_count for part in parts
                ),
                target_visit_count=visits,
            )
        )
    return EpochMetrics(per_core=tuple(pooled_cores))


def _normalize_csv_value(value: object, *, column: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise TargetCellTrainingError(
                f"CSV column {column} must be finite."
            )
        return repr(numeric)
    raise TargetCellTrainingError(
        f"CSV column {column} must contain a scalar value."
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _AtomicCSV:
    def __init__(
        self,
        path: str | Path,
        columns: Sequence[str],
        *,
        rank: int,
    ) -> None:
        if _integer(rank, name="writer_rank") != 0:
            raise TargetCellTrainingError("Only rank zero may write metric CSVs.")
        if isinstance(columns, (str, bytes)):
            raise TargetCellTrainingError("CSV columns must be a sequence.")
        try:
            materialized = tuple(columns)
        except TypeError as exc:
            raise TargetCellTrainingError(
                "CSV columns must be a sequence."
            ) from exc
        if (
            not materialized
            or any(not isinstance(column, str) or not column for column in materialized)
            or len(set(materialized)) != len(materialized)
        ):
            raise TargetCellTrainingError(
                "CSV columns must be nonempty, unique strings."
            )
        self.path = Path(path)
        self.columns = materialized
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink() or (
            self.path.exists() and not self.path.is_file()
        ):
            raise TargetCellTrainingError("Metric CSV path must be a regular file.")

    def _raw_rows(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink() or not self.path.is_file():
            raise TargetCellTrainingError("Metric CSV path must be a regular file.")
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != self.columns:
                raise TargetCellTrainingError("Metric CSV schema mismatch.")
            rows = list(reader)
        if any(
            set(row) != set(self.columns)
            or any(value is None for value in row.values())
            for row in rows
        ):
            raise TargetCellTrainingError("Metric CSV contains malformed rows.")
        return rows

    def _normalize_row(self, row: Mapping[str, object]) -> dict[str, str]:
        if not isinstance(row, Mapping) or set(row) != set(self.columns):
            raise TargetCellTrainingError("Metric CSV row has the wrong columns.")
        return {
            column: _normalize_csv_value(row[column], column=column)
            for column in self.columns
        }

    def _rewrite(self, rows: Sequence[Mapping[str, object]]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".writing",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=self.columns,
                    extrasaction="raise",
                )
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            temporary.unlink(missing_ok=True)


class AtomicEpochMetricsCSV(_AtomicCSV):
    """Atomic one-row-per-epoch CSV with idempotent resume semantics."""

    def __init__(
        self,
        path: str | Path,
        columns: Sequence[str],
        *,
        epoch_column: str = "global_epoch",
        rank: int = 0,
    ) -> None:
        super().__init__(path, columns, rank=rank)
        if epoch_column not in self.columns:
            raise TargetCellTrainingError("Epoch column is absent from CSV columns.")
        self.epoch_column = epoch_column

    def rows(self) -> tuple[dict[str, str], ...]:
        rows = self._raw_rows()
        try:
            epochs = [int(row[self.epoch_column]) for row in rows]
        except (TypeError, ValueError) as exc:
            raise TargetCellTrainingError(
                "Metric CSV contains an invalid epoch."
            ) from exc
        if epochs != list(range(1, len(rows) + 1)):
            raise TargetCellTrainingError(
                "Epoch metric rows must be unique and contiguous from one."
            )
        return tuple(rows)

    @property
    def completed_epochs(self) -> int:
        return len(self.rows())

    def append(self, row: Mapping[str, object]) -> bool:
        normalized = self._normalize_row(row)
        epoch = _integer(row[self.epoch_column], name=self.epoch_column, minimum=1)
        normalized[self.epoch_column] = str(epoch)
        existing = list(self.rows())
        if epoch <= len(existing):
            if existing[epoch - 1] == normalized:
                return False
            raise TargetCellTrainingError(
                "Refusing a divergent duplicate epoch metric row."
            )
        if epoch != len(existing) + 1:
            raise TargetCellTrainingError(
                "Epoch metrics must append contiguously."
            )
        self._rewrite([*existing, normalized])
        return True

    def reconcile(self, *, checkpoint_epoch: int) -> int:
        epoch = _integer(checkpoint_epoch, name="checkpoint_epoch")
        rows = list(self.rows())
        if len(rows) < epoch:
            raise TargetCellTrainingError(
                "Epoch metrics end before the checkpoint epoch."
            )
        if len(rows) > epoch:
            self._rewrite(rows[:epoch])
        return epoch


class AtomicPerCoreMetricsCSV(_AtomicCSV):
    """Atomic fixed-core rows per epoch with complete-block reconciliation."""

    def __init__(
        self,
        path: str | Path,
        columns: Sequence[str],
        core_aliases: Sequence[str],
        *,
        epoch_column: str = "global_epoch",
        core_column: str = "core_alias",
        rank: int = 0,
    ) -> None:
        super().__init__(path, columns, rank=rank)
        if epoch_column not in self.columns or core_column not in self.columns:
            raise TargetCellTrainingError(
                "Epoch and core columns must be present in CSV columns."
            )
        if isinstance(core_aliases, (str, bytes)):
            raise TargetCellTrainingError("Core aliases must be a sequence.")
        try:
            aliases = tuple(_alias(value) for value in core_aliases)
        except TypeError as exc:
            raise TargetCellTrainingError("Core aliases must be a sequence.") from exc
        if not aliases or len(set(aliases)) != len(aliases):
            raise TargetCellTrainingError(
                "Core aliases must be nonempty and unique."
            )
        self.epoch_column = epoch_column
        self.core_column = core_column
        self.core_aliases = aliases

    def rows(self) -> tuple[dict[str, str], ...]:
        rows = self._raw_rows()
        width = len(self.core_aliases)
        if len(rows) % width:
            raise TargetCellTrainingError(
                "Per-core CSV ends with an incomplete epoch block."
            )
        for offset in range(0, len(rows), width):
            epoch = offset // width + 1
            block = rows[offset : offset + width]
            try:
                observed_epochs = [
                    int(row[self.epoch_column]) for row in block
                ]
            except (TypeError, ValueError) as exc:
                raise TargetCellTrainingError(
                    "Per-core CSV contains an invalid epoch."
                ) from exc
            if observed_epochs != [epoch] * width:
                raise TargetCellTrainingError(
                    "Per-core epoch blocks must be contiguous from one."
                )
            if tuple(row[self.core_column] for row in block) != self.core_aliases:
                raise TargetCellTrainingError(
                    "Per-core CSV aliases are incomplete or out of order."
                )
        return tuple(rows)

    @property
    def completed_epochs(self) -> int:
        return len(self.rows()) // len(self.core_aliases)

    def append_epoch(
        self,
        rows: Sequence[Mapping[str, object]],
        *,
        global_epoch: int,
    ) -> bool:
        epoch = _integer(global_epoch, name="global_epoch", minimum=1)
        if isinstance(rows, (str, bytes)):
            raise TargetCellTrainingError("Per-core rows must be a sequence.")
        normalized_by_alias: dict[str, dict[str, str]] = {}
        try:
            materialized = tuple(rows)
        except TypeError as exc:
            raise TargetCellTrainingError(
                "Per-core rows must be a sequence."
            ) from exc
        for row in materialized:
            normalized = self._normalize_row(row)
            row_epoch = _integer(
                row[self.epoch_column],
                name=self.epoch_column,
                minimum=1,
            )
            if row_epoch != epoch:
                raise TargetCellTrainingError("Per-core row epoch mismatch.")
            alias = _alias(row[self.core_column])
            if alias in normalized_by_alias:
                raise TargetCellTrainingError("Duplicate per-core metric row.")
            normalized[self.epoch_column] = str(epoch)
            normalized[self.core_column] = alias
            normalized_by_alias[alias] = normalized
        if set(normalized_by_alias) != set(self.core_aliases):
            raise TargetCellTrainingError(
                "Per-core rows must cover every configured core exactly once."
            )
        block = [normalized_by_alias[alias] for alias in self.core_aliases]
        existing = list(self.rows())
        completed = len(existing) // len(self.core_aliases)
        if epoch <= completed:
            start = (epoch - 1) * len(self.core_aliases)
            if existing[start : start + len(block)] == block:
                return False
            raise TargetCellTrainingError(
                "Refusing divergent duplicate per-core epoch metrics."
            )
        if epoch != completed + 1:
            raise TargetCellTrainingError(
                "Per-core metrics must append complete contiguous epochs."
            )
        self._rewrite([*existing, *block])
        return True

    def reconcile(self, *, checkpoint_epoch: int) -> int:
        epoch = _integer(checkpoint_epoch, name="checkpoint_epoch")
        rows = list(self.rows())
        completed = len(rows) // len(self.core_aliases)
        if completed < epoch:
            raise TargetCellTrainingError(
                "Per-core metrics end before the checkpoint epoch."
            )
        retained = epoch * len(self.core_aliases)
        if len(rows) > retained:
            self._rewrite(rows[:retained])
        return epoch


__all__ = [
    "EPOCH_AGGREGATE_SCHEMA",
    "RANK_METRIC_SCHEMA",
    "AtomicEpochMetricsCSV",
    "AtomicPerCoreMetricsCSV",
    "CoreMetrics",
    "EpochMetrics",
    "RankCoreMetrics",
    "TargetCellTrainingError",
    "TargetRankMetricAccumulator",
    "aggregate_rank_core_metrics",
]
