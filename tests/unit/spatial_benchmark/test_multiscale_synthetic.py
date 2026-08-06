"""Focused Stage-0 multiscale synthetic-recovery contracts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from spatial_benchmark.identifiers import canonical_sha256  # noqa: E402
from spatial_benchmark.multiscale_synthetic import (  # noqa: E402
    ARM_NULL_TRUE_LOCAL,
    ARM_PERMUTED_LOCAL,
    ARM_SELF_REGIONAL,
    ARM_TRUE_LOCAL,
    AliasSafeObservedGeometry,
    MultiscaleSyntheticError,
    SyntheticDeletionDiagnostic,
    SyntheticRecoveryConfig,
    assert_alias_safe_configuration,
    build_multiscale_synthetic_fixture,
    evaluate_synthetic_recovery_gate,
    load_alias_safe_observed_geometry,
    run_multiscale_synthetic_recovery,
)
from spatial_benchmark.multiscale_hurdle_contract import (  # noqa: E402
    ACTIVE_CONTRACT_AMENDMENT_SHA256,
    REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
)
from spatial_benchmark.paths import ProjectPaths  # noqa: E402
from spatial_benchmark.registry import Registry  # noqa: E402
from spatial_benchmark.run_archive import RunArchive  # noqa: E402


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/diagnostics/run_multiscale_synthetic_recovery.py"
_SPEC = importlib.util.spec_from_file_location(
    "test_multiscale_synthetic_runner_module",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)

_ENQUEUE_SCRIPT = (
    _ROOT / "scripts/train/enqueue_multiscale_synthetic_recovery.py"
)
_ENQUEUE_SPEC = importlib.util.spec_from_file_location(
    "test_multiscale_synthetic_enqueue_module",
    _ENQUEUE_SCRIPT,
)
assert _ENQUEUE_SPEC is not None and _ENQUEUE_SPEC.loader is not None
_ENQUEUER = importlib.util.module_from_spec(_ENQUEUE_SPEC)
sys.modules[_ENQUEUE_SPEC.name] = _ENQUEUER
_ENQUEUE_SPEC.loader.exec_module(_ENQUEUER)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        project_root=root,
        config_root=root / "configs",
        data_root=root / "data",
        artifact_root=root / "artifacts",
        state_root=root / "state",
        scratch_root=root / "scratch",
        cache_root=root / "cache",
        export_root=root / "exports",
        report_root=root / "reports",
    )


@pytest.fixture(scope="module")
def observed_geometry() -> AliasSafeObservedGeometry:
    generator = np.random.default_rng(401)
    node_count = 160
    coordinates = generator.uniform(0.0, 200.0, size=(node_count, 2))
    coordinates -= np.median(coordinates, axis=0)
    coordinates = np.ascontiguousarray(coordinates, dtype=np.float64)
    stable_rows = np.arange(node_count, dtype=np.int64)
    return AliasSafeObservedGeometry(
        biological_unit_alias="ANC-05",
        coordinates_um=coordinates,
        macroblock_ids=np.full(
            node_count,
            "opaque-block-0",
            dtype="U32",
        ),
        stable_node_row_index=stable_rows,
        full_node_count=7450,
        selected_node_count=node_count,
        prepared_data_sha256="a" * 64,
        prepared_manifest_sha256="b" * 64,
        selection_index_sha256=_array_sha256(stable_rows),
        geometry_sha256=_array_sha256(coordinates),
        prepared_artifact_reference="data/processed/synthetic/prepared_v1",
    )


@pytest.fixture(scope="module")
def recovery_config() -> SyntheticRecoveryConfig:
    return SyntheticRecoveryConfig(
        seed=0,
        max_epochs=1,
        device="cpu",
        hidden_dim=8,
        decoder_dim=8,
        ffn_dim=12,
        attention_head_dim=3,
        value_head_dim=2,
        message_dim=5,
        edge_hidden_dim=7,
        edge_embedding_dim=4,
        receiver_chunk_size=64,
        target_node_batch_size=512,
    )


@pytest.fixture(scope="module")
def synthetic_fixture(observed_geometry, recovery_config):
    return build_multiscale_synthetic_fixture(
        observed_geometry,
        recovery_config,
    )


@pytest.fixture(scope="module")
def recovery_result(
    observed_geometry,
    recovery_config,
    synthetic_fixture,
):
    return run_multiscale_synthetic_recovery(
        observed_geometry,
        recovery_config,
        fixture=synthetic_fixture,
    )


def test_alias_safe_loader_reads_only_verified_geometry(tmp_path: Path) -> None:
    artifact = tmp_path / "data/processed/opaque/prepared_v1"
    artifact.mkdir(parents=True)
    generator = np.random.default_rng(88)
    coordinates = generator.uniform(1000.0, 2000.0, size=(220, 2))
    macroblocks = np.repeat(
        np.asarray(["opaque-a", "opaque-b"], dtype="U16"),
        110,
    )
    np.savez(
        artifact / "prepared_data.npz",
        coordinates_um=coordinates,
        macroblock_ids=macroblocks,
    )
    data_sha = hashlib.sha256(
        (artifact / "prepared_data.npz").read_bytes()
    ).hexdigest()
    manifest = {
        "format_version": 1,
        "selection": {
            "opaque_alias": "ANC-05",
            "restricted_identifiers_emitted": False,
        },
        "files": {"prepared_data.npz": data_sha},
        "arrays": {
            "coordinates_um": {
                "shape": [220, 2],
                "dtype": "<f8",
            },
            "macroblock_ids": {
                "shape": [220],
                "dtype": "<U16",
            },
        },
    }
    (artifact / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    first = load_alias_safe_observed_geometry(
        "data/processed/opaque/prepared_v1",
        project_root=tmp_path,
        biological_unit_alias="ANC-05",
        expected_prepared_data_sha256=data_sha,
        selected_node_count=160,
    )
    second = load_alias_safe_observed_geometry(
        "data/processed/opaque/prepared_v1",
        project_root=tmp_path,
        biological_unit_alias="ANC-05",
        expected_prepared_data_sha256=data_sha,
        selected_node_count=160,
    )

    np.testing.assert_array_equal(first.coordinates_um, second.coordinates_um)
    assert first.geometry_sha256 == second.geometry_sha256
    assert first.selection_index_sha256 == second.selection_index_sha256
    assert first.selected_node_count == 160
    assert first.macroblock_ids.shape == (160,)
    assert first.stable_node_row_index.shape == (160,)
    np.testing.assert_allclose(
        np.median(first.coordinates_um, axis=0),
        np.zeros(2),
        atol=1e-12,
    )
    receipt_text = json.dumps(first.receipt(), sort_keys=True)
    assert "cell_ID" not in receipt_text
    assert "coordinates_um" not in receipt_text
    assert first.receipt()["row_identifiers_loaded"] is False
    assert first.receipt()["macroblock_ids_exported"] is False
    assert first.receipt()["stable_node_row_indices_exported"] is False

    with pytest.raises(
        MultiscaleSyntheticError,
        match="checksum",
    ):
        load_alias_safe_observed_geometry(
            artifact,
            project_root=tmp_path,
            biological_unit_alias="ANC-05",
            expected_prepared_data_sha256="0" * 64,
            selected_node_count=160,
        )


def test_alias_only_contract_rejects_direct_identifier_fields() -> None:
    assert_alias_safe_configuration(
        {"dataset": {"biological_unit_alias": "ANC-05"}}
    )
    with pytest.raises(
        MultiscaleSyntheticError,
        match="patient_id",
    ):
        assert_alias_safe_configuration(
            {"dataset": {"patient_id": "prohibited"}}
        )
    with pytest.raises(
        MultiscaleSyntheticError,
        match="cell_ID",
    ):
        assert_alias_safe_configuration(
            {"nested": [{"cell_ID": 7}]}
        )
    with pytest.raises(ValueError, match="selected_node_count is frozen at 160"):
        SyntheticRecoveryConfig(selected_node_count=240)


def test_fixture_plants_local_signal_regional_decoy_and_bounded_null(
    synthetic_fixture,
    recovery_config,
) -> None:
    audit = synthetic_fixture.audit
    assert audit["effect_sign"] == "positive"
    assert audit["active_sender_count"] == recovery_config.active_sender_count
    assert audit["planted_directed_edge_count"] >= (
        recovery_config.top_edge_count
    )
    assert audit["planted_local_target_correlation"] > 0.75
    assert 0.15 <= audit["regional_decoy_target_correlation"] <= 0.30
    assert audit["regional_decoy_target_correlation"] < (
        audit["planted_local_target_correlation"]
    )
    assert abs(audit["null_local_target_correlation"]) <= (
        recovery_config.maximum_null_absolute_correlation
    )
    assert abs(audit["null_regional_target_correlation"]) <= (
        recovery_config.maximum_null_absolute_correlation
    )
    assert audit["null_preserves_target_marginal_exactly"] is True
    assert audit["observed_permuted_signal_correlation"] <= (
        recovery_config.maximum_permuted_signal_correlation
    )
    assert audit["pre_outcome_geometry_selection"]["rejected_candidate"][
        "node_displacement_above_75um_fraction"
    ] == pytest.approx(0.8583333333333333)
    assert audit["pre_outcome_geometry_selection"]["selected_candidate"][
        "node_displacement_above_75um_fraction"
    ] == pytest.approx(0.9625)
    assert synthetic_fixture.graph_audit["fixed_contract"][
        "rewired_graph_constructed"
    ] is False
    permutation_qc = synthetic_fixture.sender_state_permutation_audit["qc"]
    assert permutation_qc["gate_passed"] is True
    assert permutation_qc["node_mapping_changed_fraction"] >= 0.99
    assert (
        permutation_qc[
            "node_displacement_above_threshold_fraction"
        ]
        >= 0.90
    )
    assert (
        permutation_qc[
            "effective_source_edge_slot_identity_changed_fraction"
        ]
        >= 0.99
    )
    assert torch.equal(
        synthetic_fixture.true_view.local_edge_index,
        synthetic_fixture.permuted_view.local_edge_index,
    )
    assert torch.equal(
        synthetic_fixture.true_view.local_edge_attributes,
        synthetic_fixture.permuted_view.local_edge_attributes,
    )
    assert synthetic_fixture.true_view.local_source_index_by_node is None
    assert (
        synthetic_fixture.permuted_view.local_source_index_by_node is not None
    )
    assert synthetic_fixture.true_view.num_genes == 4
    node_count = recovery_config.selected_node_count
    assert synthetic_fixture.evaluation_mask.shape == (node_count, 4)
    assert torch.equal(
        synthetic_fixture.evaluation_mask,
        synthetic_fixture.evaluation_mask[:, :1].expand(-1, 4),
    )
    local_codes = (
        synthetic_fixture.true_view.local_edge_index[0] * node_count
        + synthetic_fixture.true_view.local_edge_index[1]
    )
    regional_codes = (
        synthetic_fixture.true_view.regional_edge_index[0] * node_count
        + synthetic_fixture.true_view.regional_edge_index[1]
    )
    assert not bool(
        torch.isin(local_codes, regional_codes).any()
    )


def _diagnostic(
    *,
    contribution: float,
    top_delta: float,
    null_delta: float,
) -> SyntheticDeletionDiagnostic:
    return SyntheticDeletionDiagnostic(
        selected_contribution_mean=contribution,
        selected_contribution_median=contribution,
        selected_contribution_positive_fraction=float(contribution > 0),
        selected_edge_count=20,
        top_edge_count=10,
        selected_edge_checksum="a" * 64,
        matched_null_edge_checksum="b" * 64,
        mean_top_edge_distance_um=20.0,
        mean_matched_null_edge_distance_um=20.1,
        maximum_absolute_match_distance_difference_um=0.5,
        baseline_target_loss=0.5,
        top_deleted_target_loss=0.5 + top_delta,
        matched_null_deleted_target_loss=0.5 + null_delta,
        top_deletion_loss_delta=top_delta,
        matched_null_deletion_loss_delta=null_delta,
    )


def test_gate_requires_all_recovery_checks_and_rejects_null_discovery() -> None:
    passed = evaluate_synthetic_recovery_gate(
        node_mapping_changed_fraction=0.99,
        node_displacement_above_threshold_fraction=0.90,
        effective_source_edge_slot_identity_changed_fraction=0.99,
        sender_state_permutation_qc_passed=True,
        self_regional_loss=0.60,
        true_local_loss=0.40,
        permuted_local_loss=0.55,
        planted_diagnostic=_diagnostic(
            contribution=0.10,
            top_delta=0.12,
            null_delta=0.02,
        ),
        null_diagnostic=_diagnostic(
            contribution=-0.01,
            top_delta=0.01,
            null_delta=0.01,
        ),
    )
    assert passed.gate_passed is True
    assert passed.failure_reasons == ()

    failed = evaluate_synthetic_recovery_gate(
        node_mapping_changed_fraction=0.99,
        node_displacement_above_threshold_fraction=0.90,
        effective_source_edge_slot_identity_changed_fraction=0.99,
        sender_state_permutation_qc_passed=True,
        self_regional_loss=0.60,
        true_local_loss=0.40,
        permuted_local_loss=0.55,
        planted_diagnostic=_diagnostic(
            contribution=0.10,
            top_delta=0.12,
            null_delta=0.02,
        ),
        null_diagnostic=_diagnostic(
            contribution=0.02,
            top_delta=0.04,
            null_delta=0.01,
        ),
    )
    assert failed.analogous_null_discovery is True
    assert failed.gate_passed is False
    assert (
        "analogous_discovery_occurred_under_null_injection"
        in failed.failure_reasons
    )

    weak_permutation = evaluate_synthetic_recovery_gate(
        node_mapping_changed_fraction=0.98,
        node_displacement_above_threshold_fraction=0.89,
        effective_source_edge_slot_identity_changed_fraction=0.98,
        sender_state_permutation_qc_passed=False,
        self_regional_loss=0.60,
        true_local_loss=0.40,
        permuted_local_loss=0.55,
        planted_diagnostic=_diagnostic(
            contribution=0.10,
            top_delta=0.12,
            null_delta=0.02,
        ),
        null_diagnostic=_diagnostic(
            contribution=-0.01,
            top_delta=0.01,
            null_delta=0.01,
        ),
    )
    assert weak_permutation.node_mapping_changed_sufficient is False
    assert weak_permutation.node_displacement_sufficient is False
    assert (
        weak_permutation.edge_slot_sender_identity_change_sufficient
        is False
    )
    assert weak_permutation.gate_passed is False
    assert (
        "sender_state_permutation_qc_failed"
        in weak_permutation.failure_reasons
    )


def test_real_models_train_parameter_matched_and_run_deletion_diagnostics(
    recovery_result,
) -> None:
    assert set(recovery_result.arms) == {
        ARM_SELF_REGIONAL,
        ARM_TRUE_LOCAL,
        ARM_PERMUTED_LOCAL,
        ARM_NULL_TRUE_LOCAL,
    }
    assert recovery_result.parameter_match_verified is True
    counts = {
        outcome.parameter_count
        for outcome in recovery_result.arms.values()
    }
    structures = {
        outcome.parameter_structure_sha256
        for outcome in recovery_result.arms.values()
    }
    initial_states = {
        outcome.initial_parameter_sha256
        for outcome in recovery_result.arms.values()
    }
    assert len(counts) == len(structures) == len(initial_states) == 1
    for outcome in recovery_result.arms.values():
        assert len(outcome.training.history) == 1
        assert np.isfinite(outcome.whole_node_loss)
    for diagnostic in (
        recovery_result.planted_diagnostic,
        recovery_result.null_diagnostic,
    ):
        assert diagnostic.selected_edge_count >= diagnostic.top_edge_count
        assert len(diagnostic.selected_edge_checksum) == 64
        assert len(diagnostic.matched_null_edge_checksum) == 64
        assert diagnostic.maximum_absolute_match_distance_difference_um >= 0
    metrics = recovery_result.final_metrics()
    assert np.isfinite(metrics["fit/whole_node/hurdle_loss"])
    assert metrics[
        "synthetic/sender_state_node_mapping_changed_fraction"
    ] == recovery_result.fixture.sender_state_permutation_audit["qc"][
        "node_mapping_changed_fraction"
    ]
    # One epoch is an execution smoke, not expected to pass the scientific gate.
    assert isinstance(recovery_result.gate.gate_passed, bool)


def _runner_config(
    geometry: AliasSafeObservedGeometry,
    settings: SyntheticRecoveryConfig,
    fixture,
) -> dict[str, object]:
    bundle = fixture.graph_audit["bundle_checksums"]
    permutation = fixture.sender_state_permutation_audit
    permutation_qc = permutation["qc"]
    geometry_receipt = geometry.receipt()
    return {
        "model": {
            "name": "multiscale-hurdle-count",
            "family": "additive_multiscale_hurdle_count",
            "embedding_dim": settings.hidden_dim,
        },
        "masking": {"type": "mixed_expression_masking", "rate": 0.35},
        "dataset": {
            "dataset_id": "opaque_adjacent_normal_geometry_v1",
            "version": "v1",
            "split_id": "synthetic-fit",
            "dataset_fingerprint": geometry.prepared_data_sha256,
            "split_fingerprint": geometry.geometry_sha256,
            "preprocessing_version": "observed_geometry_only_v1",
            "biological_unit_alias": geometry.biological_unit_alias,
            "prepared_artifact_reference": (
                geometry.prepared_artifact_reference
            ),
            "prepared_data_sha256": geometry.prepared_data_sha256,
        },
        "features": {
            "use_edge_features": True,
            "edge_features": ["distance_um"],
        },
        "graph": {
            "neighbor_k": 64,
            "k": 64,
            "radius_um": 75.0,
            "symmetry": "mutual",
        },
        "trainer": {
            "learning_rate": settings.learning_rate,
            "batch_size": 1,
            "primary_checkpoint_role": "last",
            "restore_best": False,
        },
        "evaluation": {
            "task_family": "masked_expression_hurdle_count",
            "protocol": "held_in_full_core_fixed_budget",
            "canonical_prediction_split": "fit",
            "primary_metric": "fit/whole_node/hurdle_loss",
            "primary_direction": "minimize",
        },
        "campaign": {
            "campaign_id": (
                "cmp_20260729_multiscale_hurdle_count_pilot"
            ),
            "active_contract_amendment_sha256": (
                ACTIVE_CONTRACT_AMENDMENT_SHA256
            ),
            "required_contract_supplement_sha256": (
                REQUIRED_CONTRACT_SUPPLEMENT_SHA256
            ),
        },
        "metadata": {
            "execution_role": "stage0_synthetic_recovery",
            "no_post_outcome_tuning": True,
            "pre_run_negative_diagnostic": {
                "reference": _RUNNER.PRERUN_NEGATIVE_DIAGNOSTIC_REFERENCE,
                "file_sha256": (
                    _RUNNER.PRERUN_NEGATIVE_DIAGNOSTIC_FILE_SHA256
                ),
                "payload_checksum": (
                    _RUNNER.PRERUN_NEGATIVE_DIAGNOSTIC_PAYLOAD_CHECKSUM
                ),
                "registered_stage0_result_inferred": False,
            },
            "expected_fixture_preflight": {
                "selected_node_count": geometry_receipt[
                    "selected_node_count"
                ],
                "full_node_count": geometry_receipt["full_node_count"],
                "prepared_manifest_sha256": geometry_receipt[
                    "prepared_manifest_sha256"
                ],
                "selection_index_sha256": geometry_receipt[
                    "selection_index_sha256"
                ],
                "geometry_sha256": geometry_receipt["geometry_sha256"],
                "macroblock_ids_sha256": geometry_receipt[
                    "macroblock_ids_sha256"
                ],
                "macroblock_count": geometry_receipt["macroblock_count"],
                "true_graph_bundle_sha256": bundle["bundle_sha256"],
                "true_local_graph_sha256": bundle["local_graph_sha256"],
                "true_regional_graph_sha256": bundle[
                    "regional_graph_sha256"
                ],
                "sender_state_permutation_receipt_checksum": permutation[
                    "checksum"
                ],
                "sender_state_permutation_mapping_sha256": permutation[
                    "source_index_by_node_sha256"
                ],
                "effective_permuted_source_equals_receiver_count": (
                    permutation_qc[
                        "effective_permuted_source_equals_receiver_count"
                    ]
                ),
                "effective_permuted_source_equals_receiver_fraction": (
                    permutation_qc[
                        "effective_permuted_source_equals_receiver_fraction"
                    ]
                ),
                "affected_receiver_count": permutation_qc[
                    "effective_permuted_source_equals_receiver_affected_receiver_count"
                ],
                "affected_receiver_fraction": permutation_qc[
                    "effective_permuted_source_equals_receiver_affected_receiver_fraction"
                ],
                "fixture_checksum": fixture.fixture_checksum,
            },
            "synthetic_recovery": asdict(settings),
        },
        "seed": settings.seed,
        "fold": 0,
    }


def _write_worker_support(archive: RunArchive) -> None:
    archive.write_json(
        "provenance/git.json",
        {"commit": "test", "dirty": False, "dirty_fingerprint": None},
    )
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "test\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json(
        "provenance/data_fingerprints.json",
        {"dataset_fingerprint": "a" * 64},
    )
    archive.write_json(
        "provenance/split_fingerprint.json",
        {"split_fingerprint": "b" * 64},
    )
    archive.write_text(
        "provenance/command.txt",
        '["run_multiscale_synthetic_recovery.py"]\n',
    )
    archive.prepare_log_files()
    archive.write_manifest(
        {
            "run_id": archive.run_id,
            "status": "success",
            "schema_version": 1,
        }
    )


def test_archive_runner_writes_verifiable_identifier_free_bundle(
    tmp_path: Path,
    observed_geometry,
    recovery_config,
    recovery_result,
) -> None:
    paths = _paths(tmp_path)
    run_id = "r_20260729T120000Z_12345678_s000_f00_a01_synthetic"
    config = _runner_config(
        observed_geometry,
        recovery_config,
        recovery_result.fixture,
    )
    archive = RunArchive.create(
        run_id,
        paths=paths,
        resolved_config=config,
    )

    result = _RUNNER.run_synthetic_recovery_archive(
        config,
        archive,
        observed_geometry=observed_geometry,
        recovery_result=recovery_result,
    )
    _write_worker_support(archive)
    archive._validate_success_ready()

    assert result.primary_metric_name == "fit/whole_node/hurdle_loss"
    assert result.checkpoint_path.is_file()
    assert result.prediction_path.is_file()
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    checksum = receipt.pop("checksum")
    assert checksum == canonical_sha256(receipt)
    assert receipt["registered_execution"] is True
    assert receipt["direct_identifiers_emitted"] is False
    assert receipt["active_contract_amendment_sha256"] == (
        ACTIVE_CONTRACT_AMENDMENT_SHA256
    )
    assert receipt["required_contract_supplement_sha256"] == (
        REQUIRED_CONTRACT_SUPPLEMENT_SHA256
    )
    assert receipt["sender_state_permutation_receipt_checksum"] == (
        recovery_result.fixture.sender_state_permutation_audit["checksum"]
    )
    assert receipt["real_data_graph_interpretation_authorized"] == (
        recovery_result.gate.gate_passed
    )
    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert set(checkpoint["arm_state_dicts"]) == {
        ARM_SELF_REGIONAL,
        ARM_TRUE_LOCAL,
        ARM_PERMUTED_LOCAL,
        ARM_NULL_TRUE_LOCAL,
    }
    for path in archive.scratch_path.rglob("*"):
        if path.suffix not in {".json", ".jsonl", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in (
            '"patient_id"',
            '"donor_id"',
            '"cell_ID"',
            '"core_id"',
        ):
            assert forbidden not in text


def test_cli_requires_exact_queue_worker_owned_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    observed_geometry,
    recovery_config,
    synthetic_fixture,
) -> None:
    paths = _paths(tmp_path)
    run_id = "r_20260729T120001Z_12345678_s000_f00_a01_synthetic"
    config = _runner_config(
        observed_geometry,
        recovery_config,
        synthetic_fixture,
    )
    archive = RunArchive.create(
        run_id,
        paths=paths,
        resolved_config=config,
    )
    config_path = archive.scratch_path / "config.resolved.yaml"
    monkeypatch.setenv("BAGM_RUN_ID", run_id)
    monkeypatch.setenv("BAGM_RUN_SCRATCH", str(archive.scratch_path))
    monkeypatch.setenv("BAGM_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(_RUNNER, "current_paths", lambda: paths)
    monkeypatch.setattr(
        _RUNNER,
        "validate_experiment_config",
        lambda _: None,
    )
    args = _RUNNER.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--run-scratch",
            str(archive.scratch_path),
        ]
    )

    attached, loaded = _RUNNER._worker_archive_and_config(args)

    assert attached.run_id == run_id
    assert loaded["campaign"]["campaign_id"] == (
        "cmp_20260729_multiscale_hurdle_count_pilot"
    )
    bad_args = _RUNNER.build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--run-scratch",
            str(tmp_path / "other"),
        ]
    )
    with pytest.raises(
        MultiscaleSyntheticError,
        match="BAGM_RUN_SCRATCH",
    ):
        _RUNNER._worker_archive_and_config(bad_args)


def test_locked_stage0_enqueue_is_checkable_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "state/tracking/bagm.sqlite3"
    receipt_path = tmp_path / "stage0_enqueue_receipt.json"
    registry = Registry(database)
    registry.create_campaign(
        _ENQUEUER.CAMPAIGN_ID,
        name="Synthetic Stage-0 test campaign",
    )
    registry.register_dataset(
        "cosmx_anc05_adjacent_normal_full_core_fit_v1",
        "adjacent_normal_full_core_fit_v1",
        display_name="Opaque geometry fixture",
        protected_source_path=(
            "data/processed/adjacent_normal_10core_qkv_large_k_v1/"
            "anc-05/prepared_v1"
        ),
        raw_fingerprint=(
            "c0713d5d70fc9597a54b30c1d5264a175e353cdd9d9f7d9c0e47f08c10d0dd3e"
        ),
        processed_fingerprint=(
            "b12c1d688f07824ae3b029cdfaaf59990be48d1fbf4290ea6ee67a5c4d1688f7"
        ),
        verification_status="verified_materialized_full_core_preprocessing",
    )
    registry.register_split(
        "3a01fab74089e721",
        dataset_id="cosmx_anc05_adjacent_normal_full_core_fit_v1",
        dataset_version="adjacent_normal_full_core_fit_v1",
        method="deterministic_all_cells_full_core_fit_no_holdout",
        unit="single_adjacent_normal_spatial_core",
        fingerprint=(
            "3a01fab74089e721fadd893480d87b99ccbedb84140bfa9204aa4fe1347c6166"
        ),
        verification_status="verified_all_materialized_nodes_assigned_fit",
    )
    monkeypatch.setattr(
        _ENQUEUER,
        "_disk_used_decimal_gb",
        lambda: 42.0,
    )

    checked = _ENQUEUER.enqueue_stage0(
        database_path=database,
        receipt_path=receipt_path,
        check_only=True,
    )
    assert checked["complete"] is True
    assert checked["queue_mutation_performed"] is False
    assert checked["job_id"] is None
    assert receipt_path.exists() is False

    first = _ENQUEUER.enqueue_stage0(
        database_path=database,
        receipt_path=receipt_path,
    )
    second = _ENQUEUER.enqueue_stage0(
        database_path=database,
        receipt_path=receipt_path,
    )

    assert first["queue_mutation_performed"] is True
    assert second["queue_mutation_performed"] is False
    assert first["job_id"] == second["job_id"]
    with registry.connect() as connection:
        rows = connection.execute(
            "SELECT * FROM queue_jobs WHERE campaign_id = ?",
            (_ENQUEUER.CAMPAIGN_ID,),
        ).fetchall()
    assert len(rows) == 1
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    checksum = stored.pop("checksum")
    assert checksum == canonical_sha256(stored)
    assert stored["training_started"] is False
