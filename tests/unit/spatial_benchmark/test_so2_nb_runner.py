from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as torch_mp

from scripts.train import run_so2_geometry_modulated_nb as runner
from spatial_benchmark.gradient_direction_observability import (
    BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
    GRADIENT_DIRECTION_METRICS_COLUMNS,
    BlockGradientDirectionEpochSummary,
    GradientDirectionEpochSummary,
)
from spatial_benchmark.identifiers import create_run_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.run_archive import RunArchive, validate_prediction_rows
from spatial_benchmark.so2_nb_training import (
    AtomicBestLatestCheckpointStore,
    CHECKPOINT_SCHEMA,
    DurableNBPerCoreMetricsCSV,
    DurableScalarCSV,
    EPOCH_METRIC_COLUMNS,
    PROTOCOL,
    EarlyStoppingState,
    DurableNBEpochMetricsCSV,
    SO2NBTrainingError,
    ValidationAggregate,
    ValidationViewStatistics,
    aggregate_validation_statistics,
    sha256_file,
    update_early_stopping,
)


def _view(alias: str, view: int, *, n: int, nll: float) -> ValidationViewStatistics:
    return ValidationViewStatistics(
        alias=alias,
        view_index=view,
        n_masked_entries=n,
        negative_binomial_nll_sum=nll * n,
        raw_absolute_error_sum=2.0 * n,
        raw_squared_error_sum=9.0 * n,
        log1p_absolute_error_sum=0.5 * n,
        log1p_squared_error_sum=0.25 * n,
        poisson_deviance_sum=1.5 * n,
        observed_zero_sum=0.25 * n,
        predicted_zero_probability_sum=0.4 * n,
        zero_brier_sum=0.1 * n,
        observed_count_sum=3.0 * n,
        predicted_count_sum=4.0 * n,
    )


def _rank_zero_failure_gloo_worker(
    rank: int,
    rendezvous_path: str,
    result_queue: object,
) -> None:
    """Prove a rank-zero Python failure leaves the Gloo group usable."""

    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    callback_calls = 0
    try:
        def fail_on_rank_zero() -> None:
            nonlocal callback_calls
            callback_calls += 1
            raise ValueError("injected rank-zero persistence failure")

        try:
            runner._broadcast_rank_zero_result(
                rank=rank,
                control_group=dist.group.WORLD,
                phase="two-rank checkpoint persistence",
                operation=fail_on_rank_zero,
            )
        except SO2NBTrainingError as error:
            error_message = str(error)
        else:
            raise AssertionError("Rank-zero failure was not propagated.")

        live_ranks = torch.tensor(1, dtype=torch.int64)
        dist.all_reduce(live_ranks)
        result_queue.put(  # type: ignore[attr-defined]
            (rank, callback_calls, error_message, int(live_ranks.item()))
        )
    finally:
        dist.destroy_process_group()


def test_early_stop_tracks_best_from_epoch_one_but_patience_starts_after_50() -> None:
    state = EarlyStoppingState()
    for epoch in range(1, 76):
        state = update_early_stopping(state, 1.0, completed_epoch=epoch)
        if epoch == 1:
            assert state.improved and state.best_epoch == 1
        elif epoch <= 50:
            assert state.bad_validations == 0
        elif epoch < 75:
            assert state.bad_validations == epoch - 50
            assert not state.should_stop

    assert state.should_stop
    assert state.stop_reason == "patience"
    assert state.completed_epoch == 75
    assert state.best_epoch == 1


def test_best_update_uses_strict_absolute_min_delta() -> None:
    state = update_early_stopping(EarlyStoppingState(), 1.0, completed_epoch=1)
    state = update_early_stopping(state, 0.99995, completed_epoch=2)
    assert not state.improved and state.best_value == 1.0
    state = update_early_stopping(state, 0.9998, completed_epoch=3)
    assert state.improved and state.best_value == pytest.approx(0.9998)


def test_validation_primary_is_equal_view_then_equal_core_not_pooled() -> None:
    records = (
        _view("A", 0, n=1, nll=1.0),
        _view("A", 1, n=9, nll=3.0),
        _view("B", 0, n=100, nll=10.0),
        _view("B", 1, n=100, nll=14.0),
    )
    result = aggregate_validation_statistics(
        records, aliases=("A", "B"), views_per_core=2
    )

    assert result.per_core["A"]["masked_negative_binomial_nll"] == 2.0
    assert result.per_core["B"]["masked_negative_binomial_nll"] == 12.0
    assert result.primary_equal_core_nll == 7.0
    assert result.pooled_nll == pytest.approx((1 + 27 + 1000 + 1400) / 210)
    assert result.raw_count_rmse == 3.0
    assert result.per_view["A:0"]["observed_count_mean"] == 3.0
    assert result.per_view["B:1"]["predicted_count_mean"] == 4.0


def _best_payload(epoch: int, value: float, tensor_value: float) -> dict[str, object]:
    early = EarlyStoppingState(
        best_value=value,
        best_epoch=epoch,
        completed_epoch=epoch,
        improved=True,
    )
    return {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "protocol": PROTOCOL,
        "checkpoint_role": "best",
        "completed_epoch": epoch,
        "early_stopping_state": asdict(early),
        "preprocessing_fingerprint": "1" * 64,
        "split_fingerprint": "2" * 64,
        "best_validation_metric": value,
        "best_checkpoint_sha256": None,
        "embedded_best_checkpoint": None,
        "embedded_best_checkpoint_tree_sha256": None,
        "model_state_dict": {"weight": torch.tensor([tensor_value])},
    }


def test_latest_is_commit_point_and_recovers_best_after_two_file_crash(
    tmp_path: Path,
) -> None:
    store = AtomicBestLatestCheckpointStore(tmp_path)
    committed_best = _best_payload(1, 2.0, 1.0)
    best_receipt = store.save_best(committed_best)
    latest_early = EarlyStoppingState(
        best_value=2.0,
        best_epoch=1,
        bad_validations=1,
        completed_epoch=2,
    )
    latest = {
        **_best_payload(2, 2.0, 2.0),
        "checkpoint_role": "latest",
        "early_stopping_state": asdict(latest_early),
        "best_checkpoint_sha256": best_receipt.sha256,
        "embedded_best_checkpoint": committed_best,
        "embedded_best_checkpoint_tree_sha256": runner._tree_sha256(committed_best),
    }
    store.save_latest(latest)

    # Simulate a crash after a future best replacement but before latest commits.
    store.save_best(_best_payload(3, 1.5, 3.0))
    assert runner._reconcile_checkpoint_transaction(store) is True

    restored = store.load("best")
    torch.testing.assert_close(restored["model_state_dict"]["weight"], torch.tensor([1.0]))
    repaired_latest = store.load("latest")
    assert repaired_latest["best_checkpoint_sha256"] == sha256_file(store.best_path)
    store.finalize_best_only()
    assert sorted(path.name for path in store.directory.iterdir()) == ["best.ckpt"]


def test_checkpoint_roles_are_built_from_one_cpu_state_snapshot(
    tmp_path: Path,
) -> None:
    original_best = _best_payload(1, 2.0, 1.0)
    role_fields = {
        "checkpoint_role",
        "best_checkpoint_sha256",
        "embedded_best_checkpoint",
        "embedded_best_checkpoint_tree_sha256",
    }
    checkpoint_state = {
        key: value for key, value in original_best.items() if key not in role_fields
    }
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min"
    )
    checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()
    best = runner._checkpoint_payload_from_state(
        checkpoint_state,
        role="best",
        best_checkpoint_sha256=None,
    )
    store = AtomicBestLatestCheckpointStore(tmp_path)
    best_receipt = store.save_best(best)
    latest = runner._checkpoint_payload_from_state(
        checkpoint_state,
        role="latest",
        best_checkpoint_sha256=best_receipt.sha256,
        embedded_best_checkpoint=best,
    )
    store.save_latest(latest)
    reloaded_best = store.load("best")
    reloaded_latest = store.load("latest")

    assert best["checkpoint_role"] == "best"
    assert latest["checkpoint_role"] == "latest"
    assert latest["model_state_dict"]["weight"].device.type == "cpu"
    assert latest["embedded_best_checkpoint"]["checkpoint_role"] == "best"
    assert latest["embedded_best_checkpoint_tree_sha256"] == runner._tree_sha256(
        latest["embedded_best_checkpoint"]
    )
    assert latest["embedded_best_checkpoint"]["scheduler_state_dict"][
        "mode_worse"
    ] == float("inf")
    assert reloaded_best["scheduler_state_dict"]["best"] == float("inf")
    assert reloaded_latest["embedded_best_checkpoint"]["scheduler_state_dict"][
        "mode_worse"
    ] == float("inf")
    assert role_fields.isdisjoint(checkpoint_state)

    with pytest.raises(SO2NBTrainingError, match="already contains role fields"):
        runner._checkpoint_payload_from_state(
            best,
            role="latest",
            best_checkpoint_sha256="a" * 64,
            embedded_best_checkpoint=best,
        )


def test_checkpoint_tree_hash_domain_separates_infinities() -> None:
    positive = runner._tree_sha256(float("inf"))
    negative = runner._tree_sha256(float("-inf"))
    ordinary_tuple = runner._tree_sha256(
        ("__bagm_checkpoint_nonfinite_float_v1__", "positive_infinity")
    )

    assert positive != negative
    assert positive != ordinary_tuple
    assert positive != runner._tree_sha256(np.float64("inf"))
    with pytest.raises(SO2NBTrainingError, match="NaN scalar"):
        runner._tree_sha256(float("nan"))
    with pytest.raises(SO2NBTrainingError, match="NaN NumPy scalar"):
        runner._tree_sha256(np.float64("nan"))


def test_rank_zero_phase_broadcasts_failure_before_all_ranks_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broadcasts: list[dict[str, object]] = []

    def rank_zero_broadcast(values: list[object], **_: object) -> None:
        broadcasts.append(copy.deepcopy(values[0]))

    monkeypatch.setattr(
        runner.torch.distributed,
        "broadcast_object_list",
        rank_zero_broadcast,
    )
    with pytest.raises(SO2NBTrainingError, match="checkpoint write.*ValueError.*boom"):
        with runner._synchronized_rank_zero_phase(
            rank=0,
            control_group=object(),
            phase="checkpoint write",
        ):
            raise ValueError("boom")

    assert broadcasts == [
        {"ok": False, "error_type": "ValueError", "error": "boom"}
    ]

    def peer_broadcast(values: list[object], **_: object) -> None:
        values[0] = broadcasts[0]

    monkeypatch.setattr(
        runner.torch.distributed,
        "broadcast_object_list",
        peer_broadcast,
    )
    with pytest.raises(SO2NBTrainingError, match="checkpoint write.*ValueError.*boom"):
        with runner._synchronized_rank_zero_phase(
            rank=1,
            control_group=object(),
            phase="checkpoint write",
        ):
            pass


def test_rank_zero_result_success_broadcasts_the_operation_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    broadcasts: list[object] = []
    control_group = object()

    def operation() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"completed_epoch": 7}

    def broadcast(
        values: list[object], *, src: int, group: object
    ) -> None:
        assert src == 0
        assert group is control_group
        broadcasts.append(copy.deepcopy(values[0]))

    monkeypatch.setattr(
        runner.torch.distributed,
        "broadcast_object_list",
        broadcast,
    )
    result = runner._broadcast_rank_zero_result(
        rank=0,
        control_group=control_group,
        phase="checkpoint persistence",
        operation=operation,
    )

    assert calls == 1
    assert result == {"completed_epoch": 7}
    assert broadcasts == [
        {"ok": True, "result": {"completed_epoch": 7}}
    ]


def test_rank_zero_result_propagates_failure_and_excludes_follower_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank_zero_envelope: dict[str, object] | None = None

    def rank_zero_broadcast(values: list[object], **_: object) -> None:
        nonlocal rank_zero_envelope
        rank_zero_envelope = copy.deepcopy(values[0])

    monkeypatch.setattr(
        runner.torch.distributed,
        "broadcast_object_list",
        rank_zero_broadcast,
    )
    with pytest.raises(
        SO2NBTrainingError,
        match="checkpoint persistence.*OSError.*disk full",
    ):
        runner._broadcast_rank_zero_result(
            rank=0,
            control_group=object(),
            phase="checkpoint persistence",
            operation=lambda: (_ for _ in ()).throw(OSError("disk full")),
        )

    assert rank_zero_envelope == {
        "ok": False,
        "error_type": "OSError",
        "error": "disk full",
    }
    follower_calls = 0

    def follower_operation() -> None:
        nonlocal follower_calls
        follower_calls += 1
        raise AssertionError("Follower callback must never run.")

    def follower_broadcast(values: list[object], **_: object) -> None:
        values[0] = copy.deepcopy(rank_zero_envelope)

    monkeypatch.setattr(
        runner.torch.distributed,
        "broadcast_object_list",
        follower_broadcast,
    )
    with pytest.raises(
        SO2NBTrainingError,
        match="checkpoint persistence.*OSError.*disk full",
    ):
        runner._broadcast_rank_zero_result(
            rank=3,
            control_group=object(),
            phase="checkpoint persistence",
            operation=follower_operation,
        )
    assert follower_calls == 0


@pytest.mark.parametrize("malformed", [None, {}, {"ok": "yes"}, {"ok": True}])
def test_rank_zero_result_rejects_malformed_envelopes(
    monkeypatch: pytest.MonkeyPatch,
    malformed: object,
) -> None:
    def malformed_broadcast(values: list[object], **_: object) -> None:
        values[0] = malformed

    monkeypatch.setattr(
        runner.torch.distributed,
        "broadcast_object_list",
        malformed_broadcast,
    )
    with pytest.raises(SO2NBTrainingError, match="malformed"):
        runner._broadcast_rank_zero_result(
            rank=1,
            control_group=object(),
            phase="malformed-test",
            operation=lambda: pytest.fail("Follower callback was invoked."),
        )


def test_two_rank_gloo_propagates_rank_zero_failure_without_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    context = torch_mp.get_context("spawn")
    result_queue = context.SimpleQueue()
    rendezvous = tmp_path / "rank-zero-failure-rendezvous"
    torch_mp.spawn(
        _rank_zero_failure_gloo_worker,
        args=(str(rendezvous), result_queue),
        nprocs=2,
        join=True,
    )
    rows = sorted(result_queue.get() for _ in range(2))

    assert [row[0] for row in rows] == [0, 1]
    assert [row[1] for row in rows] == [1, 0]
    assert all(
        "two-rank checkpoint persistence" in row[2]
        and "ValueError" in row[2]
        and "injected rank-zero persistence failure" in row[2]
        for row in rows
    )
    assert [row[3] for row in rows] == [2, 2]


def test_first_epoch_crash_discards_orphan_best_and_truncates_metrics(
    tmp_path: Path,
) -> None:
    store = AtomicBestLatestCheckpointStore(tmp_path)
    store.save_best(_best_payload(1, 2.0, 1.0))
    writer = DurableNBEpochMetricsCSV(tmp_path)
    row = {field: 0 for field in EPOCH_METRIC_COLUMNS}
    row.update(
        {
            "schema": "ignored-and-normalized",
            "global_epoch": 1,
            "stop_reason": "",
            "global_gradient_json": {},
            "block_gradients_json": [],
            "dispersion_gradient_json": {},
            "training_per_core_json": {},
            "validation_per_core_json": {},
            "training_mask_receipts_json": [],
            "validation_mask_checksums_json": {},
        }
    )
    writer.append(row)
    events = tmp_path / "metrics/events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        '{"name":"train/masked_negative_binomial_nll","step":1,"value":2.0}\n'
        '{"name":"val/unseen_donor/masked_negative_binomial_nll","step":1,"value":3.0}\n',
        encoding="utf-8",
    )

    assert store.discard_best_without_latest() is True
    writer.reconcile(checkpoint_epoch=0)
    runner._reconcile_metric_events(events, checkpoint_epoch=0)

    assert not store.best_path.exists()
    assert writer.rows() == []
    assert events.read_text(encoding="utf-8") == ""


def test_terminal_latest_has_no_additional_training_epochs() -> None:
    terminal = EarlyStoppingState(
        best_value=1.0,
        best_epoch=40,
        bad_validations=25,
        completed_epoch=75,
        should_stop=True,
        stop_reason="patience",
    )
    assert list(runner._remaining_training_epochs(terminal)) == []
    active = EarlyStoppingState(
        best_value=1.0,
        best_epoch=40,
        bad_validations=24,
        completed_epoch=74,
    )
    remaining = runner._remaining_training_epochs(active)
    assert remaining.start == 74 and remaining.stop == 300


def test_interrupted_atomic_temps_are_removed_without_touching_checkpoints(
    tmp_path: Path,
) -> None:
    store = AtomicBestLatestCheckpointStore(tmp_path)
    store.save_best(_best_payload(1, 2.0, 1.0))
    stale_checkpoint = store.directory / ".latest.ckpt.crash.writing"
    stale_checkpoint.write_bytes(b"partial")
    stale_table = tmp_path / "metrics/.history.parquet.writing"
    stale_table.parent.mkdir(parents=True, exist_ok=True)
    stale_table.write_bytes(b"partial")

    runner._remove_interrupted_atomic_temps(tmp_path)

    assert store.best_path.is_file()
    assert not stale_checkpoint.exists()
    assert not stale_table.exists()


def test_bundle_fingerprints_are_bound_to_config() -> None:
    dataset = {
        "dataset_fingerprint": "1" * 64,
        "dataset_fingerprint_role": "immutable_overlay_manifest_content_sha256",
        "overlay_manifest_file_sha256": "2" * 64,
        "split_fingerprint": "3" * 64,
        "preprocessing_fingerprint": "4" * 64,
    }
    bundle = SimpleNamespace(
        manifest_content_sha256="1" * 64,
        manifest_sha256="2" * 64,
        split_fingerprint="3" * 64,
        preprocessing_fingerprint="4" * 64,
    )
    runner._validate_bundle_binding({"dataset": dataset}, bundle)
    mutated = SimpleNamespace(**{**vars(bundle), "manifest_sha256": "9" * 64})
    with pytest.raises(SO2NBTrainingError, match="overlay_manifest_file_sha256"):
        runner._validate_bundle_binding({"dataset": dataset}, mutated)


def test_preflight_gpu_identities_bind_inventory_vram_and_live_rank() -> None:
    identities = [
        {
            "rank": rank,
            "local_rank": rank,
            "name": "NVIDIA GeForce RTX 3090",
            "total_memory_bytes": 25_295_011_840,
            "total_memory_gib": 25_295_011_840 / float(1024**3),
            "compute_capability": [8, 6],
            "torch_version": str(torch.__version__),
            "cuda_runtime": torch.version.cuda or "12.0",
        }
        for rank in range(4)
    ]
    vram = [
        {
            "rank": rank,
            "local_rank": rank,
            "total_memory_bytes": 25_295_011_840,
            "peak_allocated_vram_gib": 10.0,
            "peak_reserved_vram_gib": 11.0,
        }
        for rank in range(4)
    ]
    receipt = {"gpu_identities": identities}
    gates = {"gpu_inventory": {"devices": identities}}
    runner._validate_preflight_gpu_identities(
        receipt, gates, vram, current_identity=identities[2]
    )

    stale_live = {**identities[2], "total_memory_bytes": 1}
    with pytest.raises(SO2NBTrainingError, match="live_gpu_identity"):
        runner._validate_preflight_gpu_identities(
            receipt, gates, vram, current_identity=stale_live
        )

    mismatched_receipt = copy.deepcopy(receipt)
    mismatched_receipt["gpu_identities"][1]["name"] = "stale GPU"
    with pytest.raises(SO2NBTrainingError, match="gpu_inventory.devices"):
        runner._validate_preflight_gpu_identities(
            mismatched_receipt, gates, vram, current_identity=identities[2]
        )

    stale_vram = copy.deepcopy(vram)
    stale_vram[3]["total_memory_bytes"] -= 1
    with pytest.raises(SO2NBTrainingError, match="gpu_vram_identity"):
        runner._validate_preflight_gpu_identities(
            receipt, gates, stale_vram, current_identity=identities[2]
        )


def test_gradient_rows_add_run_and_seed_for_exact_csv_schemas() -> None:
    common = dict(
        global_epoch=1,
        trainable_parameter_count=7,
        optimizer_updates_observed=6,
        gradient_norm_mean_before_clip=2.0,
        gradient_norm_min_before_clip=1.0,
        gradient_norm_max_before_clip=3.0,
        consecutive_optimizer_step_cosine_mean=0.2,
        consecutive_optimizer_step_cosine_median=0.2,
        consecutive_optimizer_step_cosine_min=-0.1,
        consecutive_optimizer_step_cosine_max=0.5,
        consecutive_optimizer_step_cosine_valid_pairs=5,
        epoch_aggregate_gradient_cosine_to_previous_epoch=None,
        resume_boundary_unavailable=False,
    )
    global_row = runner._gradient_rows(
        GradientDirectionEpochSummary(**common), run_id="run"
    )
    block_row = runner._gradient_rows(
        BlockGradientDirectionEpochSummary(
            **common, block_index=0, block_name="blocks.0"
        ),
        run_id="run",
    )

    assert set(global_row) == set(GRADIENT_DIRECTION_METRICS_COLUMNS)
    assert set(block_row) == set(BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS)
    assert global_row["model_seed"] == 0
    assert block_row["run_id"] == "run"


def test_compact_validation_summaries_satisfy_predictive_archive_contract(
    tmp_path: Path,
) -> None:
    validation = aggregate_validation_statistics(
        [
            _view(alias, view, n=10 + view, nll=2.0 + view / 10)
            for alias in ("SO2-C27", "SO2-C28")
            for view in range(10)
        ]
    )
    batches = {
        alias: SimpleNamespace(n_nodes=20, n_genes=1_000, n_edges=40)
        for alias in ("SO2-C27", "SO2-C28")
    }
    bundle = SimpleNamespace(batches_by_alias=batches)
    run_id = create_run_id(
        seed=0,
        fold=0,
        attempt=1,
        scientific_id_value="sci_7a91c6e212345678",
        timestamp=datetime(2026, 9, 7, tzinfo=timezone.utc),
        unique_suffix="so2nb001",
    )
    config = {
        "dataset": {"dataset_id": "cosmx_so2_nb_train12_val2_v1"},
        "evaluation": {
            "artifact_contract": "predictive",
            "canonical_prediction_split": "validation",
            "primary_metric": "val/unseen_donor/masked_negative_binomial_nll",
        },
        "trainer": {"primary_checkpoint_role": "best", "restore_best": True},
    }
    predictions = runner._validation_prediction_rows(
        run_id=run_id, config=config, bundle=bundle, validation=validation
    )
    assert len(predictions) == 20
    validate_prediction_rows(
        predictions, expected_run_id=run_id, expected_split="validation"
    )
    assert all(len(row["sample_key"]) == 64 for row in predictions)

    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})
    archive = RunArchive.create(
        run_id,
        paths=paths,
        resolved_config=config,
    )
    primary = validation.primary_equal_core_nll
    store = AtomicBestLatestCheckpointStore(archive.scratch_path)
    best_payload = _best_payload(1, primary, 1.0)
    best_receipt = store.save_best(best_payload)
    latest_state = EarlyStoppingState(
        best_value=primary,
        best_epoch=1,
        completed_epoch=1,
        improved=True,
    )
    store.save_latest(
        {
            **_best_payload(1, primary, 1.0),
            "checkpoint_role": "latest",
            "early_stopping_state": asdict(latest_state),
            "best_checkpoint_sha256": best_receipt.sha256,
            "embedded_best_checkpoint": best_payload,
            "embedded_best_checkpoint_tree_sha256": runner._tree_sha256(best_payload),
        }
    )
    archive.write_predictions("validation", predictions, fallback="jsonl")
    archive.write_table(
        "metrics/history",
        [{"run_id": run_id, "global_epoch": 1, "train/loss": 2.5, "val/loss": primary}],
        fallback="jsonl",
    )
    archive.write_summary(
        {
            "run_id": run_id,
            "status": "success",
            "final_epoch": 1,
            "best_epoch": 1,
            "epoch_metrics_rows": 1,
            "primary_metric_name": "val/unseen_donor/masked_negative_binomial_nll",
            "primary_metric_value": primary,
        }
    )
    archive.append_metric_event(
        {
            "name": "train/masked_negative_binomial_nll",
            "value": 2.5,
            "step": 1,
        }
    )
    archive.append_metric_event(
        {
            "name": "val/unseen_donor/masked_negative_binomial_nll",
            "value": primary,
            "step": 1,
        }
    )
    archive.write_json(
        "metrics/final.json",
        {"val/unseen_donor/masked_negative_binomial_nll": primary},
    )

    epoch_row = {field: 0 for field in EPOCH_METRIC_COLUMNS}
    epoch_row.update(
        {
            "global_epoch": 1,
            "best_epoch": 1,
            "best_validation_metric": primary,
            "training_equal_core_masked_nb_nll": 2.5,
            "validation_equal_core_masked_nb_nll": primary,
            "global_gradient_json": {},
            "block_gradients_json": [],
            "dispersion_gradient_json": {},
            "training_per_core_json": {},
            "validation_per_core_json": {},
            "training_mask_receipts_json": [],
            "validation_mask_checksums_json": {},
            "stop_reason": "maximum_epochs",
        }
    )
    DurableNBEpochMetricsCSV(archive.scratch_path).append(epoch_row)
    DurableNBPerCoreMetricsCSV(archive.scratch_path).append_epoch(
        global_epoch=1,
        training_per_core={alias: 2.5 for alias in runner.TRAINING_ALIASES},
        validation_per_core=validation.per_core,
    )
    common = dict(
        global_epoch=1,
        trainable_parameter_count=7,
        optimizer_updates_observed=6,
        gradient_norm_mean_before_clip=2.0,
        gradient_norm_min_before_clip=1.0,
        gradient_norm_max_before_clip=3.0,
        consecutive_optimizer_step_cosine_mean=0.2,
        consecutive_optimizer_step_cosine_median=0.2,
        consecutive_optimizer_step_cosine_min=-0.1,
        consecutive_optimizer_step_cosine_max=0.5,
        consecutive_optimizer_step_cosine_valid_pairs=5,
        epoch_aggregate_gradient_cosine_to_previous_epoch=None,
        resume_boundary_unavailable=False,
    )
    global_row = runner._gradient_rows(
        GradientDirectionEpochSummary(**common), run_id=run_id
    )
    block_rows = [
        runner._gradient_rows(
            BlockGradientDirectionEpochSummary(
                **common, block_index=index, block_name=f"blocks.{index}"
            ),
            run_id=run_id,
        )
        for index in range(4)
    ]
    DurableScalarCSV(
        archive.scratch_path,
        "results/gradient_direction_metrics.csv",
        GRADIENT_DIRECTION_METRICS_COLUMNS,
        rows_per_epoch=1,
    ).append_epoch([global_row], global_epoch=1)
    DurableScalarCSV(
        archive.scratch_path,
        "results/gradient_direction_by_block.csv",
        BLOCK_GRADIENT_DIRECTION_METRICS_COLUMNS,
        rows_per_epoch=4,
    ).append_epoch(block_rows, global_epoch=1)
    DurableScalarCSV(
        archive.scratch_path,
        "results/gradient_direction_dispersion.csv",
        GRADIENT_DIRECTION_METRICS_COLUMNS,
        rows_per_epoch=1,
    ).append_epoch([global_row], global_epoch=1)
    figure = archive.scratch_path / "figures/loss_vs_epoch.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"plot")
    archive.write_json(
        "diagnostics/final_checkpoint_reload_verification.json",
        {
            "verified": True,
            "checkpoint_sha256": best_receipt.sha256,
        },
    )
    archive.write_json("diagnostics/hardware_preflight.json", {"passed": True})
    archive.write_json(
        "provenance/so2_geometry_modulated_nb_training.json",
        {
            "validation_prediction_summary_rows": 20,
            "plots": {"loss_vs_epoch": "figures/loss_vs_epoch.png"},
        },
    )
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "test", "dirty": True})
    archive.write_text("provenance/uncommitted_changes.patch", "new runner\n")
    archive.write_text("provenance/environment.txt", "python=test\n")
    archive.write_json("provenance/hardware.json", {"gpus": 4})
    archive.write_json("provenance/data_fingerprints.json", {"dataset": "test"})
    archive.write_json("provenance/split_fingerprint.json", {"split": "test"})
    archive.write_text("provenance/command.txt", "torchrun runner\n")

    # This is the real queue lifecycle boundary: the runner must finish before
    # the queue owns and writes manifest.yaml.
    assert not (archive.scratch_path / "manifest.yaml").exists()
    runner._validate_runner_outputs(
        archive.scratch_path,
        run_id=run_id,
        checkpoint_store=store,
        expect_latest=True,
    )
    store.finalize_best_only()
    runner._validate_runner_outputs(
        archive.scratch_path,
        run_id=run_id,
        checkpoint_store=store,
        expect_latest=False,
    )
