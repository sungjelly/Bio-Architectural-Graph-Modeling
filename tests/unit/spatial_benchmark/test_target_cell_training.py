from __future__ import annotations

import csv
from pathlib import Path
import random

import pytest

from spatial_benchmark.target_cell_training import (
    RANK_METRIC_SCHEMA,
    AtomicEpochMetricsCSV,
    AtomicPerCoreMetricsCSV,
    RankCoreMetrics,
    TargetCellTrainingError,
    TargetRankMetricAccumulator,
    aggregate_rank_core_metrics,
)


def _completed_rank(
    alias: str,
    *,
    n_cells: int,
    world_size: int,
    rank: int,
    scale: float = 1.0,
) -> RankCoreMetrics:
    accumulator = TargetRankMetricAccumulator(
        core_alias=alias,
        n_cells=n_cells,
        world_size=world_size,
        rank=rank,
    )
    start = accumulator.target_start
    stop = accumulator.target_stop
    targets = list(range(start, stop))
    if targets:
        accumulator.add(
            targets,
            negative_binomial_nll_sum=scale * (rank + 1),
            masked_entry_count=2 * len(targets),
        )
    return accumulator.finalize()


def test_rank_accumulator_adds_batches_and_requires_each_target_once() -> None:
    accumulator = TargetRankMetricAccumulator(
        core_alias="so2-c15",
        n_cells=7,
        world_size=2,
        rank=0,
        target_start=0,
        target_stop=4,
    )
    accumulator.add(
        [2, 0],
        negative_binomial_nll_sum=1.25,
        masked_entry_count=5,
    )
    accumulator.add(
        [3, 1],
        negative_binomial_nll_sum=2.75,
        masked_entry_count=7,
    )
    result = accumulator.finalize()

    assert result.core_alias == "SO2-C15"
    assert result.target_visit_counts == (1, 1, 1, 1)
    assert result.target_visit_count == 4
    assert result.masked_entry_count == 12
    assert result.negative_binomial_nll_sum == pytest.approx(4.0)
    assert result.pooled_negative_binomial_nll == pytest.approx(1.0 / 3.0)
    assert result.to_payload()["schema"] == RANK_METRIC_SCHEMA
    assert RankCoreMetrics.from_payload(result.to_payload()) == result


def test_rank_accumulator_fails_closed_on_duplicate_gap_and_bad_statistics() -> None:
    accumulator = TargetRankMetricAccumulator(
        core_alias="SO2-C16",
        n_cells=3,
        world_size=1,
        rank=0,
    )
    accumulator.add(
        [0], negative_binomial_nll_sum=1.0, masked_entry_count=1
    )
    with pytest.raises(TargetCellTrainingError, match="more than once"):
        accumulator.add(
            [0], negative_binomial_nll_sum=1.0, masked_entry_count=1
        )
    with pytest.raises(TargetCellTrainingError, match="outside"):
        accumulator.add(
            [3], negative_binomial_nll_sum=1.0, masked_entry_count=1
        )
    with pytest.raises(TargetCellTrainingError, match="at least one"):
        accumulator.add(
            [1, 2], negative_binomial_nll_sum=1.0, masked_entry_count=1
        )
    with pytest.raises(TargetCellTrainingError, match="every local target"):
        accumulator.finalize()


def test_empty_rank_interval_finalizes_without_fake_loss_entries() -> None:
    result = _completed_rank(
        "SO2-C17", n_cells=2, world_size=4, rank=3
    )
    assert result.target_start == result.target_stop == 2
    assert result.target_visit_counts == ()
    assert result.target_visit_count == 0
    assert result.masked_entry_count == 0
    assert result.pooled_negative_binomial_nll is None


def test_rank_payload_rejects_noncanonical_or_inexact_coverage() -> None:
    payload = _completed_rank(
        "SO2-C18", n_cells=5, world_size=2, rank=1
    ).to_payload()
    payload["target_start"] = 2
    with pytest.raises(TargetCellTrainingError, match="canonical contiguous"):
        RankCoreMetrics.from_payload(payload)

    payload = _completed_rank(
        "SO2-C18", n_cells=5, world_size=2, rank=1
    ).to_payload()
    payload["target_visit_counts"][0] = 2
    with pytest.raises(TargetCellTrainingError, match="exactly once"):
        RankCoreMetrics.from_payload(payload)


def test_rank_aggregation_is_order_independent_and_entry_weighted() -> None:
    rank_metrics = [
        _completed_rank(
            alias,
            n_cells=5 if alias == "SO2-C15" else 4,
            world_size=3,
            rank=rank,
            scale=1.0 if alias == "SO2-C15" else 10.0,
        )
        for alias in ("SO2-C16", "SO2-C15")
        for rank in range(3)
    ]
    payloads = [item.to_payload() for item in rank_metrics]
    random.Random(1234).shuffle(payloads)
    aggregate = aggregate_rank_core_metrics(
        payloads,
        expected_core_aliases=("SO2-C16", "SO2-C15"),
    )

    assert [item.core_alias for item in aggregate.per_core] == [
        "SO2-C15",
        "SO2-C16",
    ]
    assert [item.target_visit_count for item in aggregate.per_core] == [5, 4]
    assert aggregate.target_visit_count == 9
    assert aggregate.masked_entry_count == 18
    assert aggregate.negative_binomial_nll_sum == pytest.approx(66.0)
    assert aggregate.pooled_negative_binomial_nll == pytest.approx(66.0 / 18.0)
    receipt = aggregate.to_payload()
    assert receipt["coverage_exact_disjoint_exhaustive"] is True
    assert all(
        row["coverage_exact_disjoint_exhaustive"]
        for row in receipt["per_core"]
    )


def test_rank_aggregation_rejects_missing_duplicate_or_unexpected_cores() -> None:
    ranks = [
        _completed_rank("SO2-C15", n_cells=3, world_size=2, rank=rank)
        for rank in range(2)
    ]
    with pytest.raises(TargetCellTrainingError, match="every rank exactly once"):
        aggregate_rank_core_metrics(ranks[:1])
    with pytest.raises(TargetCellTrainingError, match="every rank exactly once"):
        aggregate_rank_core_metrics([ranks[0], ranks[0]])
    with pytest.raises(TargetCellTrainingError, match="expected cores exactly"):
        aggregate_rank_core_metrics(
            ranks, expected_core_aliases=("SO2-C15", "SO2-C16")
        )


EPOCH_COLUMNS = (
    "global_epoch",
    "negative_binomial_nll_sum",
    "masked_entry_count",
    "pooled_negative_binomial_nll",
)


def _epoch_row(epoch: int, value: float = 2.5) -> dict[str, object]:
    return {
        "global_epoch": epoch,
        "negative_binomial_nll_sum": value * 4,
        "masked_entry_count": 4,
        "pooled_negative_binomial_nll": value,
    }


def test_epoch_csv_is_atomic_idempotent_and_reconcilable(tmp_path: Path) -> None:
    path = tmp_path / "results" / "epoch_metrics.csv"
    writer = AtomicEpochMetricsCSV(path, EPOCH_COLUMNS)
    assert writer.append(_epoch_row(1)) is True
    assert writer.append(_epoch_row(1)) is False
    assert writer.append(_epoch_row(2, 2.0)) is True
    assert writer.append(_epoch_row(3, 1.5)) is True
    assert writer.completed_epochs == 3

    assert writer.reconcile(checkpoint_epoch=2) == 2
    assert writer.completed_epochs == 2
    assert writer.append(_epoch_row(3, 1.25)) is True
    assert [row["global_epoch"] for row in writer.rows()] == ["1", "2", "3"]
    with pytest.raises(TargetCellTrainingError, match="divergent duplicate"):
        writer.append(_epoch_row(3, 999.0))
    with pytest.raises(TargetCellTrainingError, match="before the checkpoint"):
        writer.reconcile(checkpoint_epoch=4)


def test_epoch_csv_rejects_gaps_nonfinite_values_and_nonzero_writer_rank(
    tmp_path: Path,
) -> None:
    path = tmp_path / "epoch.csv"
    with pytest.raises(TargetCellTrainingError, match="Only rank zero"):
        AtomicEpochMetricsCSV(path, EPOCH_COLUMNS, rank=1)
    writer = AtomicEpochMetricsCSV(path, EPOCH_COLUMNS)
    with pytest.raises(TargetCellTrainingError, match="contiguously"):
        writer.append(_epoch_row(2))
    nonfinite = _epoch_row(1)
    nonfinite["pooled_negative_binomial_nll"] = float("nan")
    with pytest.raises(TargetCellTrainingError, match="finite"):
        writer.append(nonfinite)


PER_CORE_COLUMNS = (
    "global_epoch",
    "core_alias",
    "negative_binomial_nll_sum",
    "masked_entry_count",
    "pooled_negative_binomial_nll",
)


def _core_rows(epoch: int, offset: float = 0.0) -> list[dict[str, object]]:
    return [
        {
            "global_epoch": epoch,
            "core_alias": alias,
            "negative_binomial_nll_sum": 10.0 + index + offset,
            "masked_entry_count": 5,
            "pooled_negative_binomial_nll": (
                10.0 + index + offset
            )
            / 5.0,
        }
        for index, alias in enumerate(("SO2-C16", "SO2-C15"))
    ]


def test_per_core_csv_orders_complete_blocks_and_reconciles(tmp_path: Path) -> None:
    writer = AtomicPerCoreMetricsCSV(
        tmp_path / "results" / "per_core_epoch_metrics.csv",
        PER_CORE_COLUMNS,
        ("SO2-C15", "SO2-C16"),
    )
    first = _core_rows(1)
    assert writer.append_epoch(first, global_epoch=1) is True
    assert writer.append_epoch(first, global_epoch=1) is False
    assert writer.append_epoch(_core_rows(2), global_epoch=2) is True
    assert writer.append_epoch(_core_rows(3), global_epoch=3) is True
    assert writer.completed_epochs == 3
    assert [row["core_alias"] for row in writer.rows()] == [
        "SO2-C15",
        "SO2-C16",
        "SO2-C15",
        "SO2-C16",
        "SO2-C15",
        "SO2-C16",
    ]

    assert writer.reconcile(checkpoint_epoch=1) == 1
    assert writer.completed_epochs == 1
    assert writer.append_epoch(_core_rows(2, 1.0), global_epoch=2) is True


def test_per_core_csv_rejects_partial_duplicate_and_corrupt_blocks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "per_core.csv"
    writer = AtomicPerCoreMetricsCSV(
        path,
        PER_CORE_COLUMNS,
        ("SO2-C15", "SO2-C16"),
    )
    with pytest.raises(TargetCellTrainingError, match="every configured core"):
        writer.append_epoch(_core_rows(1)[:1], global_epoch=1)
    duplicate = _core_rows(1)
    duplicate[1]["core_alias"] = duplicate[0]["core_alias"]
    with pytest.raises(TargetCellTrainingError, match="Duplicate"):
        writer.append_epoch(duplicate, global_epoch=1)

    writer.append_epoch(_core_rows(1), global_epoch=1)
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=PER_CORE_COLUMNS).writerow(
            _core_rows(2)[0]
        )
    with pytest.raises(TargetCellTrainingError, match="incomplete epoch"):
        writer.rows()


def test_atomic_replace_failure_keeps_previous_complete_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = AtomicEpochMetricsCSV(tmp_path / "epoch.csv", EPOCH_COLUMNS)
    writer.append(_epoch_row(1))
    prior = writer.path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(
        "spatial_benchmark.target_cell_training.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="injected replace failure"):
        writer.append(_epoch_row(2))

    assert writer.path.read_bytes() == prior
    assert not list(tmp_path.glob(".epoch.csv.*.writing"))
