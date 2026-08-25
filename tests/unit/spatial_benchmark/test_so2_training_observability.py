from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from spatial_benchmark.so2_training_observability import (
    AtomicLatestCheckpointStore,
    CHECKPOINT_SCHEMA,
    DurableEpochMetricsCSV,
    EPOCH_METRICS_COLUMNS,
    SO2_CORE_ALIASES,
    SO2ObservabilityError,
    build_epoch_metrics_row,
    plateau_monitor_fields,
    require_latest_only_checkpoint_layout,
)


def _row(epoch: int, *, durations: tuple[float, ...] = ()) -> dict[str, object]:
    losses = [1.0 / (index + 1) for index in range(epoch)]
    return build_epoch_metrics_row(
        run_id="r_20260825T000000Z_deadbeef_s000_f00_a01_deadbeef",
        model_seed=0,
        global_epoch=epoch,
        equal_core_mean_masked_huber=losses[-1],
        per_core_masked_huber={
            alias: losses[-1] + index / 1000.0
            for index, alias in enumerate(SO2_CORE_ALIASES)
        },
        gradient_norms_before_clip=tuple(index / 10.0 for index in range(1, 8)),
        learning_rate=1e-4,
        optimizer_updates_this_epoch=7,
        cumulative_optimizer_updates=epoch * 7,
        epoch_duration_seconds=10.0 + epoch,
        prior_epoch_durations=durations,
        complete_graph_mask_views=140,
        masked_entries_across_views=1_400_000,
        peak_vram_gib_all_ranks=12.5,
        loss_history=losses,
    )


def _payload(epoch: int, model: torch.nn.Module) -> dict[str, object]:
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "completed_global_epochs": epoch,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": {},
        "amp_scaler_state_dict": {},
    }


def test_epoch_row_contains_all_core_losses_and_operational_metrics() -> None:
    row = _row(1)
    assert tuple(row) == EPOCH_METRICS_COLUMNS
    assert row["equal_core_mean_masked_huber"] == pytest.approx(1.0)
    assert row["loss_so2_c15"] == pytest.approx(1.0)
    assert row["loss_so2_c28"] == pytest.approx(1.013)
    assert row["gradient_norm_mean_before_clip"] == pytest.approx(0.4)
    assert row["gradient_norm_max_before_clip"] == pytest.approx(0.7)
    assert row["complete_graph_mask_views"] == 140
    assert row["complete_graph_views_per_second"] == pytest.approx(140 / 11)
    assert row["eta_to_epoch_200_seconds"] == pytest.approx(199 * 11)
    assert row["plateau_audit_performed"] is False


def test_epoch_row_rejects_incomplete_ddp_epoch() -> None:
    values = dict(_row(1))
    values["complete_graph_mask_views"] = 139
    with pytest.raises(SO2ObservabilityError, match="exactly 140"):
        build_epoch_metrics_row(
            run_id=str(values["run_id"]),
            model_seed=0,
            global_epoch=1,
            equal_core_mean_masked_huber=1.0,
            per_core_masked_huber={alias: 1.0 for alias in SO2_CORE_ALIASES},
            gradient_norms_before_clip=[1.0] * 7,
            learning_rate=1e-4,
            optimizer_updates_this_epoch=7,
            cumulative_optimizer_updates=7,
            epoch_duration_seconds=1.0,
            prior_epoch_durations=(),
            complete_graph_mask_views=139,
            masked_entries_across_views=1,
            peak_vram_gib_all_ranks=1.0,
            loss_history=[1.0],
        )


def test_csv_appends_fsyncable_rows_and_duplicate_is_idempotent(
    tmp_path: Path,
) -> None:
    writer = DurableEpochMetricsCSV(tmp_path)
    first = _row(1)
    assert writer.append(first) is True
    assert writer.append(first) is False
    second = _row(2, durations=(11.0,))
    assert writer.append(second) is True
    assert writer.completed_epochs == 2
    assert writer.prior_durations == pytest.approx((11.0, 12.0))

    with writer.path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert list(rows[0]) == list(EPOCH_METRICS_COLUMNS)
    assert rows[1]["global_epoch"] == "2"


def test_csv_resume_trims_uncheckpointed_row_without_creating_duplicates(
    tmp_path: Path,
) -> None:
    writer = DurableEpochMetricsCSV(tmp_path)
    durations: tuple[float, ...] = ()
    for epoch in range(1, 4):
        writer.append(_row(epoch, durations=durations))
        durations += (10.0 + epoch,)

    assert writer.reconcile(checkpoint_epoch=2) == 2
    assert writer.completed_epochs == 2
    assert writer.append(_row(3, durations=(11.0, 12.0))) is True
    assert writer.completed_epochs == 3
    with writer.path.open("r", encoding="utf-8", newline="") as handle:
        epochs = [int(row["global_epoch"]) for row in csv.DictReader(handle)]
    assert epochs == [1, 2, 3]


def test_csv_resume_rejects_history_gap(tmp_path: Path) -> None:
    writer = DurableEpochMetricsCSV(tmp_path)
    writer.append(_row(1))
    with pytest.raises(SO2ObservabilityError, match="end before"):
        writer.reconcile(checkpoint_epoch=2)


def test_failed_csv_replace_preserves_previous_complete_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = DurableEpochMetricsCSV(tmp_path)
    writer.append(_row(1))
    first_bytes = writer.path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected CSV replacement failure")

    monkeypatch.setattr(
        "spatial_benchmark.so2_training_observability.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="injected CSV replacement failure"):
        writer.append(_row(2, durations=(11.0,)))

    assert writer.path.read_bytes() == first_bytes
    assert writer.completed_epochs == 1
    assert not list((tmp_path / "results").glob(".epoch_metrics.csv.*.writing"))


def test_failed_first_csv_replace_leaves_no_torn_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = DurableEpochMetricsCSV(tmp_path)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected first CSV replacement failure")

    monkeypatch.setattr(
        "spatial_benchmark.so2_training_observability.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="injected first CSV replacement failure"):
        writer.append(_row(1))

    assert not writer.path.exists()
    assert not list((tmp_path / "results").glob(".epoch_metrics.csv.*.writing"))


def test_plateau_columns_record_two_consecutive_training_audits() -> None:
    first = plateau_monitor_fields([1.0] * 150, model_seed=0)
    assert first["plateau_audit_performed"] is True
    assert first["plateau_qualifying_passed"] is True
    assert first["plateau_consecutive_passing_audits"] == 1
    assert first["plateau_should_stop"] is False

    second = plateau_monitor_fields([1.0] * 175, model_seed=0)
    assert second["plateau_consecutive_passing_audits"] == 2
    assert second["plateau_should_stop"] is True


def test_rolling_checkpoint_replaces_previous_and_model_reloads(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(3, 2)
    first_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    store = AtomicLatestCheckpointStore(tmp_path)
    first = store.save(_payload(1, model), completed_global_epochs=1)
    assert first.role == "latest"
    assert require_latest_only_checkpoint_layout(tmp_path, final=False) == first.path

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    second = store.save(_payload(2, model), completed_global_epochs=2)
    assert second.path == first.path
    assert second.sha256 != first.sha256
    assert require_latest_only_checkpoint_layout(tmp_path, final=False) == second.path
    loaded = store.load_latest()
    reloaded = torch.nn.Linear(3, 2)
    reloaded.load_state_dict(loaded["model_state_dict"], strict=True)
    assert all(
        torch.equal(reloaded.state_dict()[name], value)
        for name, value in model.state_dict().items()
    )
    assert any(
        not torch.equal(reloaded.state_dict()[name], first_state[name])
        for name in first_state
    )


def test_failed_atomic_replace_preserves_previous_latest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = torch.nn.Linear(2, 1)
    store = AtomicLatestCheckpointStore(tmp_path)
    first = store.save(_payload(1, model), completed_global_epochs=1)
    first_bytes = first.path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replacement failure")

    monkeypatch.setattr(
        "spatial_benchmark.so2_training_observability.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="injected"):
        store.save(_payload(2, model), completed_global_epochs=2)
    assert first.path.read_bytes() == first_bytes
    assert not list((tmp_path / "checkpoints").glob("*.writing"))
    assert store.completed_global_epochs == 1


def test_finalize_retains_only_loadable_last_checkpoint(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    store = AtomicLatestCheckpointStore(tmp_path)
    store.save(_payload(1, model), completed_global_epochs=1)
    final_payload = _payload(1, model)
    final_payload["plateau"] = {"should_stop": True}
    receipt = store.finalize(final_payload, completed_global_epochs=1)

    assert receipt.role == "last"
    assert require_latest_only_checkpoint_layout(tmp_path, final=True) == receipt.path
    loaded = torch.load(receipt.path, map_location="cpu", weights_only=True)
    assert loaded["completed_global_epochs"] == 1
    assert loaded["plateau"]["should_stop"] is True
    with pytest.raises(SO2ObservabilityError, match="immutable|already exists"):
        store.save(_payload(2, model), completed_global_epochs=2)


def test_nonzero_rank_cannot_write_shared_outputs(tmp_path: Path) -> None:
    with pytest.raises(SO2ObservabilityError, match="rank zero"):
        DurableEpochMetricsCSV(tmp_path, rank=1)
    with pytest.raises(SO2ObservabilityError, match="rank zero"):
        AtomicLatestCheckpointStore(tmp_path, rank=1)
