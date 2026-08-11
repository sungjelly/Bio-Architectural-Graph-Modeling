from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.registry import Registry, RegistryConflictError
from spatial_benchmark.run_archive import RunArchive, verify_run_bundle
from spatial_benchmark.same_gene_nonlinear import AdditiveNeighborMLP


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER_PATH = PROJECT_ROOT / "scripts/train/run_same_gene_nonlinear.py"
_SPEC = importlib.util.spec_from_file_location(
    "same_gene_nonlinear_runner_convergence_tests", RUNNER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runner
_SPEC.loader.exec_module(runner)


def _lifecycle_fixture(root: Path) -> tuple[Any, Registry, RunArchive, dict[str, Any]]:
    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(root)})
    configuration = runner._configuration(profile="pilot", fold=0, attempt=1)
    identifier = scientific_id(configuration)
    run_id = runner.create_run_id(
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        scientific_id_value=identifier,
        unique_suffix="life0001",
    )
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    registry.create_campaign(
        runner.CAMPAIGN_ID,
        name=runner.CAMPAIGN_DISPLAY_NAME,
        scientific_question=runner.CAMPAIGN_SCIENTIFIC_QUESTION,
        config={"synthetic": True},
    )
    registry.register_variant(
        identifier,
        campaign_id=runner.CAMPAIGN_ID,
        configuration=configuration,
    )
    archive = RunArchive.create(
        run_id,
        paths=paths,
        manifest={"lifecycle_status_source": "registry_and_completion_marker"},
        resolved_config=configuration,
    )
    registry.create_run(
        run_id,
        campaign_id=runner.CAMPAIGN_ID,
        scientific_id=identifier,
        repro_id="rep_synthetic_lifecycle",
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        configuration=configuration,
        status="running",
        artifact_path=archive.artifact_path,
    )
    result = {
        "run_id": run_id,
        "campaign_id": runner.CAMPAIGN_ID,
        "profile": "pilot",
        "outer_fold": 0,
        "status": "completed",
        "duration_seconds": 1.25,
        "peak_vram_gb": 2.5,
        "parameter_count": 123,
        "controls": {"projected_full_hours_per_fold": 0.1},
        "arms": {
            "observed_near": {
                "evaluation": {"component_equal_mse": 0.75},
                "jacobian_summary": {"row_top1_fraction": 1.0},
            }
        },
    }
    archive.write_summary(
        {"status": "completed", "primary_metric": "validation/test", "primary_value": 0.75}
    )
    archive.append_metric_event({"name": "validation/test", "value": 0.75})
    archive.write_json("metrics/final.json", {"validation/test": 0.75})
    archive.write_table(
        "metrics/history",
        [{"epoch": 1, "validation/test": 0.75}],
    )
    archive.prepare_log_files()
    archive.write_json("provenance/git.json", {"commit": "synthetic", "dirty": False})
    archive.write_text("provenance/uncommitted_changes.patch", "")
    archive.write_text("provenance/environment.txt", "synthetic\n")
    archive.write_json("provenance/hardware.json", {"device": "cpu"})
    archive.write_json("provenance/data_fingerprints.json", {"dataset": "synthetic"})
    archive.write_json("provenance/split_fingerprint.json", {"split": "synthetic"})
    archive.write_text("provenance/command.txt", "synthetic\n")
    archive.write_bytes("checkpoints/last.ckpt", b"checkpoint")
    archive.write_predictions(
        "validation",
        [
            {
                "run_id": run_id,
                "sample_key": "sk_synthetic",
                "dataset_id": "synthetic",
                "split": "validation",
                "y_true": 1.0,
                "y_pred": 0.5,
            }
        ],
    )
    archive.write_json("results.json", result)
    archive.write_json("resolved_configuration.json", configuration)
    archive.publish_success_pending()
    return paths, registry, archive, configuration


@pytest.mark.parametrize(
    "boundary",
    (
        "registry_finalizing",
        "artifacts_recorded",
        "checkpoint_recorded",
        "metrics_recorded",
        "success_marked",
        "registry_completed",
    ),
)
def test_success_finalization_replays_every_durable_crash_boundary(
    tmp_path: Path, boundary: str
) -> None:
    _paths, registry, archive, configuration = _lifecycle_fixture(tmp_path)

    def crash(name: str) -> None:
        if name == boundary:
            raise RuntimeError(f"synthetic failpoint {name}")

    with pytest.raises(RuntimeError, match="synthetic failpoint"):
        runner._finish_published_success(
            registry=registry,
            archive=archive,
            configuration=configuration,
            profile="pilot",
            fold=0,
            failpoint=crash,
        )
    recovered = runner._finish_published_success(
        registry=registry,
        archive=archive,
        configuration=configuration,
        profile="pilot",
        fold=0,
    )
    assert recovered["lifecycle_reconciled"] is True
    assert registry.get_run(archive.run_id)["status"] == "completed"
    assert verify_run_bundle(archive.artifact_path)["status"] == "success"
    with registry.connect() as connection:
        duplicate_artifacts = connection.execute(
            "SELECT path, count(*) AS n FROM artifacts WHERE run_id = ? "
            "GROUP BY path HAVING n > 1",
            (archive.run_id,),
        ).fetchall()
        duplicate_metrics = connection.execute(
            "SELECT name, count(*) AS n FROM metrics WHERE run_id = ? "
            "GROUP BY name HAVING n > 1",
            (archive.run_id,),
        ).fetchall()
    assert duplicate_artifacts == []
    assert duplicate_metrics == []


def test_registry_attempt_claim_is_opt_in_and_atomic(tmp_path: Path) -> None:
    paths, registry, archive, configuration = _lifecycle_fixture(tmp_path)
    identifier = scientific_id(configuration)
    second = runner.create_run_id(
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        scientific_id_value=identifier,
        unique_suffix="default2",
    )
    # Existing registry clients retain their historical repeated-run behavior.
    registry.create_run(
        second,
        campaign_id=runner.CAMPAIGN_ID,
        scientific_id=identifier,
        repro_id="rep_default_duplicate",
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        configuration=configuration,
    )
    third = runner.create_run_id(
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        scientific_id_value=identifier,
        unique_suffix="claimed3",
    )
    with pytest.raises(RegistryConflictError, match="already claimed"):
        registry.create_run(
            third,
            campaign_id=runner.CAMPAIGN_ID,
            scientific_id=identifier,
            repro_id="rep_claimed_duplicate",
            seed=runner.TRACKING_SEED,
            fold=0,
            attempt=1,
            configuration=configuration,
            enforce_unique_attempt=True,
        )
    assert registry.get_run(third) is None


def _partial_attempt_fixture(
    root: Path, *, status: str = "running", live: bool = False
) -> tuple[ProjectPaths, Registry, RunArchive, Any]:
    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(root)})
    configuration = runner._configuration(profile="pilot", fold=0, attempt=1)
    identifier = scientific_id(configuration)
    run_id = runner.create_run_id(
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        scientific_id_value=identifier,
        unique_suffix="partial1",
    )
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    registry.create_campaign(
        runner.CAMPAIGN_ID,
        name=runner.CAMPAIGN_DISPLAY_NAME,
        scientific_question=runner.CAMPAIGN_SCIENTIFIC_QUESTION,
        config={"synthetic": True},
    )
    registry.register_variant(
        identifier,
        campaign_id=runner.CAMPAIGN_ID,
        configuration=configuration,
    )
    archive = RunArchive.create(run_id, paths=paths)
    pid = runner.os.getpid() if live else 999_999_999
    ticks = runner._process_start_ticks(pid) if live else 1
    assert ticks is not None
    archive.write_json(
        "diagnostics/attempt_process.json",
        {
            "run_id": run_id,
            "host": runner.socket.gethostname(),
            "pid": pid,
            "proc_start_ticks": ticks,
        },
    )
    registry.create_run(
        run_id,
        campaign_id=runner.CAMPAIGN_ID,
        scientific_id=identifier,
        repro_id="rep_partial",
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        configuration=configuration,
        status=status,
        artifact_path=archive.artifact_path,
    )
    arguments = runner.argparse.Namespace(profile="pilot", fold=0, attempt=1)
    return paths, registry, archive, arguments


@pytest.mark.parametrize("initial_status", ("pending", "running", "failed"))
def test_dead_or_terminal_partial_attempt_is_sealed_before_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initial_status: str
) -> None:
    paths, registry, archive, arguments = _partial_attempt_fixture(
        tmp_path, status=initial_status
    )
    monkeypatch.setattr(runner, "current_paths", lambda: paths)
    result = runner.abandon_incomplete_attempt(arguments)
    assert (archive.artifact_path / "_FAILED").is_file()
    assert not archive.scratch_path.exists()
    assert registry.get_run(archive.run_id)["status"] == "failed"
    assert result["already_terminal"] is (initial_status == "failed")
    assert verify_run_bundle(
        archive.artifact_path, require_success_contract=False
    )["status"] == "failed"


def test_explicit_abandonment_refuses_live_owner_and_artifact_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_root = tmp_path / "live"
    paths, _registry, archive, arguments = _partial_attempt_fixture(
        live_root, live=True
    )
    monkeypatch.setattr(runner, "current_paths", lambda: paths)
    with pytest.raises(runner.NonlinearRunError, match="still live"):
        runner.abandon_incomplete_attempt(arguments)
    assert archive.scratch_path.is_dir()

    link_root = tmp_path / "link"
    paths2, _registry2, archive2, arguments2 = _partial_attempt_fixture(link_root)
    # Remove only the unused synthetic scratch in this isolated fixture, then
    # place a foreign directory behind the canonical artifact symlink.
    for child in sorted(archive2.scratch_path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            child.rmdir()
    archive2.scratch_path.rmdir()
    foreign = link_root / "foreign"
    foreign.mkdir()
    archive2.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    archive2.artifact_path.symlink_to(foreign, target_is_directory=True)
    monkeypatch.setattr(runner, "current_paths", lambda: paths2)
    with pytest.raises(runner.NonlinearRunError, match="symlink"):
        runner.abandon_incomplete_attempt(arguments2)


def test_retry_parent_requires_contiguous_terminal_predecessor(tmp_path: Path) -> None:
    paths = ProjectPaths.from_environment({"BAGM_ROOT": str(tmp_path)})
    registry = Registry(paths.state_root / "tracking/bagm.sqlite3")
    first_config = runner._configuration(profile="pilot", fold=0, attempt=1)
    second_config = runner._configuration(profile="pilot", fold=0, attempt=2)
    first_config["campaign"] = {"materialized_job_config_sha256": "a" * 64}
    second_config["campaign"] = {"materialized_job_config_sha256": "b" * 64}
    identifier = scientific_id(first_config)
    registry.create_campaign(
        runner.CAMPAIGN_ID,
        name=runner.CAMPAIGN_DISPLAY_NAME,
        config={"synthetic": True},
    )
    registry.register_variant(
        identifier,
        campaign_id=runner.CAMPAIGN_ID,
        configuration=first_config,
    )
    first_id = runner.create_run_id(
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        scientific_id_value=identifier,
        unique_suffix="retry001",
    )
    registry.create_run(
        first_id,
        campaign_id=runner.CAMPAIGN_ID,
        scientific_id=identifier,
        repro_id="rep_retry_1",
        seed=runner.TRACKING_SEED,
        fold=0,
        attempt=1,
        configuration=first_config,
        status="failed",
    )
    assert runner._retry_parent(
        registry, configuration=second_config, fold=0, attempt=2
    ) == first_id
    with pytest.raises(runner.NonlinearRunError, match="preceding attempt"):
        runner._retry_parent(
            registry,
            configuration=runner._configuration(profile="pilot", fold=0, attempt=3),
            fold=0,
            attempt=3,
        )


def test_frozen_gene_eligibility_is_shape_dtype_and_hash_bound(
    tmp_path: Path, monkeypatch: Any
) -> None:
    mask = np.zeros(1000, dtype=bool)
    mask[:932] = True
    path = tmp_path / "eligible_genes.npy"
    np.save(path, mask, allow_pickle=False)
    digest = hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes(order="C")
    ).hexdigest()
    monkeypatch.setattr(
        runner, "FROZEN_GENE_ELIGIBILITY_FILE", path.name
    )
    monkeypatch.setattr(
        runner, "FROZEN_GENE_ELIGIBILITY_SHA256", digest
    )
    loaded = runner._load_frozen_gene_eligibility(tmp_path)
    assert loaded is not None and np.array_equal(loaded, mask)

    changed = mask.copy()
    changed[999] = True
    np.save(path, changed, allow_pickle=False)
    with pytest.raises(runner.NonlinearRunError, match="SHA-256 mismatch"):
        runner._load_frozen_gene_eligibility(tmp_path)


def _small_training_problem() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(91)
    target = torch.rand(12, 4, generator=generator)
    morphology = torch.randn(12, 3, generator=generator)
    feature = torch.rand(12, 4, generator=generator)
    mask = np.ones(12, dtype=bool)
    groups = np.repeat(np.asarray([11, 22, 33], dtype=np.int16), 4)
    device = torch.device("cpu")
    return {
        "target": target,
        "morphology": morphology,
        "feature": feature,
        "train_mask": mask,
        "groups": groups,
        "target_stats": runner._target_statistics(target, mask, device),
        "morphology_stats": runner._morphology_statistics(
            morphology, mask, device
        ),
        "seed": 707,
        "device": device,
    }


def _initial_small_model_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    model = AdditiveNeighborMLP(
        gene_count=4,
        morphology_count=3,
        hidden_count=3,
        use_neighbor=True,
    )
    return copy.deepcopy(model.state_dict())


def _model_from_state(state: dict[str, torch.Tensor]) -> AdditiveNeighborMLP:
    model = AdditiveNeighborMLP(
        gene_count=4,
        morphology_count=3,
        hidden_count=3,
        use_neighbor=True,
    )
    model.load_state_dict(state, strict=True)
    return model


def _run_training_segments(
    initial_state: dict[str, torch.Tensor],
    segments: list[int],
    *,
    reset_optimizer: bool,
) -> tuple[AdditiveNeighborMLP, list[dict[str, float]]]:
    model = _model_from_state(initial_state)
    problem = _small_training_problem()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    completed = 0
    for length in segments:
        if reset_optimizer and completed:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-4
            )
        history.extend(
            runner._train_epochs(
                model,
                optimizer=optimizer,
                epochs=length,
                start_epoch=completed,
                **problem,
            )
        )
        completed += length
    return model, history


def test_segmented_training_preserves_exact_optimizer_trajectory() -> None:
    initial_state = _initial_small_model_state()
    single, single_history = _run_training_segments(
        initial_state, [12], reset_optimizer=False
    )
    segmented, segmented_history = _run_training_segments(
        initial_state, [1, 1, 2, 4, 4], reset_optimizer=False
    )

    assert single_history == segmented_history
    assert set(single.state_dict()) == set(segmented.state_dict())
    for name, value in single.state_dict().items():
        assert torch.equal(value, segmented.state_dict()[name]), name


def test_resetting_adamw_between_candidates_changes_the_trajectory() -> None:
    initial_state = _initial_small_model_state()
    continuous, _ = _run_training_segments(
        initial_state, [1, 1, 2, 4, 4], reset_optimizer=False
    )
    reset, _ = _run_training_segments(
        initial_state, [1, 1, 2, 4, 4], reset_optimizer=True
    )

    maximum_difference = max(
        float(torch.max(torch.abs(continuous.state_dict()[name] - value)).item())
        for name, value in reset.state_dict().items()
    )
    assert maximum_difference > 1e-8


def test_pilot_masks_keep_resource_validation_distinct_from_outer_test() -> None:
    folds = np.repeat(np.arange(4, dtype=np.int8), 6)
    groups = np.repeat(np.asarray([101, 202, 303, 404], dtype=np.int16), 6)
    eligible = np.ones(len(folds), dtype=bool)

    masks = runner._split_masks(
        folds, groups, eligible, outer_fold=0, profile="pilot"
    )

    assert np.array_equal(np.flatnonzero(masks["test"]), np.arange(0, 6))
    assert np.array_equal(np.flatnonzero(masks["validation"]), np.arange(6, 12))
    assert np.array_equal(np.flatnonzero(masks["tuning_train"]), np.arange(12, 24))
    assert not np.any(masks["test"] & masks["validation"])
    assert not np.any(masks["test"] & masks["tuning_train"])
    assert not np.any(masks["test"] & masks["final_train"])
    assert np.all(masks["tuning_train"] <= masks["final_train"])
    assert np.all(masks["validation"] <= masks["final_train"])


def test_pilot_refit_uses_validation_and_serializes_anchor_jacobians(
    monkeypatch: Any, tmp_path: Path
) -> None:
    generator = torch.Generator().manual_seed(123)
    target = torch.rand(16, 4, generator=generator)
    morphology = torch.randn(16, 3, generator=generator)
    feature = torch.rand(16, 4, generator=generator)
    groups = np.repeat(np.asarray([10, 20, 30, 40], dtype=np.int16), 4)
    masks = {
        "tuning_train": np.arange(16) < 8,
        "validation": (np.arange(16) >= 8) & (np.arange(16) < 12),
        "final_train": np.arange(16) < 12,
        "test": np.arange(16) >= 12,
    }
    device = torch.device("cpu")
    observed_masks: list[np.ndarray] = []
    detailed_masks: list[np.ndarray] = []
    tuning_evaluations = 0

    def new_model(
        *, use_neighbor: bool, seed: int, device: torch.device
    ) -> AdditiveNeighborMLP:
        torch.manual_seed(seed)
        return AdditiveNeighborMLP(
            gene_count=4,
            morphology_count=3,
            hidden_count=2,
            use_neighbor=use_neighbor,
        ).to(device)

    def evaluate(
        model: AdditiveNeighborMLP, *, mask: np.ndarray, **_: Any
    ) -> dict[str, float]:
        nonlocal tuning_evaluations
        del model
        observed_masks.append(mask.copy())
        tuning_evaluations += 1
        return {"component_equal_mse": float(3 - tuning_evaluations)}

    def evaluate_with_jacobian(
        model: AdditiveNeighborMLP,
        *,
        mask: np.ndarray,
        eligible_genes: np.ndarray,
        **_: Any,
    ) -> tuple[
        dict[str, Any],
        np.ndarray,
        np.ndarray,
        list[dict[str, Any]],
        dict[str, np.ndarray],
        dict[str, float],
    ]:
        detailed_masks.append(mask.copy())
        gene_count = int(eligible_genes.shape[0])
        marker = float(next(model.parameters()).detach().sum().item())
        component = int(groups[np.flatnonzero(mask)[0]])
        evaluation = {
            "cell_count": int(np.sum(mask)),
            "component_count": 1,
            "component_equal_mse": abs(marker),
            "component_equal_mae": abs(marker),
            "per_component": [
                {
                    "geometry_group": component,
                    "cell_count": int(np.sum(mask)),
                    "mse": abs(marker),
                    "mae": abs(marker),
                }
            ],
        }
        component_predictions = [
            {
                "geometry_group": component,
                "cell_count": int(np.sum(mask)),
                "y_true": [0.0] * gene_count,
                "y_pred": [marker] * gene_count,
            }
        ]
        identity = np.eye(gene_count, dtype=np.float64)
        parts = {
            "total": identity * marker,
            "linear": identity * (marker / 2),
            "nonlinear": identity * (marker / 2),
            "mean_hidden_derivative": np.full(2, marker, dtype=np.float64),
            "component_geometry_groups": np.asarray([component], dtype=np.int16),
            "component_mean_hidden_derivative": np.full(
                (1, 2), marker, dtype=np.float64
            ),
        }
        return (
            evaluation,
            np.full(gene_count, abs(marker), dtype=np.float64),
            np.zeros(gene_count, dtype=np.float64),
            component_predictions,
            parts,
            {"diagonal_offdiagonal_ratio": abs(marker)},
        )

    monkeypatch.setattr(runner, "_new_model", new_model)
    monkeypatch.setattr(runner, "_evaluate", evaluate)
    monkeypatch.setattr(runner, "_evaluate_with_jacobian", evaluate_with_jacobian)
    monkeypatch.setattr(runner, "PILOT_EPOCH_CANDIDATES", (1, 2))
    monkeypatch.setattr(runner, "PILOT_REFIT_EPOCH_OVERRIDE", 2)
    monkeypatch.setattr(runner, "ANCHOR_EPOCH", 1)
    monkeypatch.setattr(runner, "BATCH_SIZE", 4)

    result, state, jacobians = runner._fit_arm(
        "observed_near",
        target=target,
        morphology=morphology,
        feature=feature,
        masks=masks,
        groups=groups,
        profile="pilot",
        fold=0,
        device=device,
    )

    assert result["selected_epoch"] == 2
    assert result["refit_epoch"] == 2
    assert state["fit_mask"] == "tuning_train"
    assert state["evaluation_mask"] == "validation"
    assert state["anchor_epoch"] == 1
    assert state["anchor_state_dict"] is not None
    assert result["anchor"] is not None
    assert result["anchor"]["epoch"] == 1
    assert result["checkpoint_replay"]["passed"] is True
    assert result["checkpoint_replay"]["maximum_metric_abs_error"] == 0.0
    assert result["checkpoint_replay"]["maximum_prediction_abs_error"] == 0.0
    assert state["checkpoint_replay"] == result["checkpoint_replay"]
    assert all(np.array_equal(mask, masks["validation"]) for mask in observed_masks)
    assert all(np.array_equal(mask, masks["validation"]) for mask in detailed_masks)
    assert not any(np.array_equal(mask, masks["test"]) for mask in observed_masks)
    assert not any(np.array_equal(mask, masks["test"]) for mask in detailed_masks)

    assert jacobians is not None
    base_keys = {
        "total",
        "linear",
        "nonlinear",
        "mean_hidden_derivative",
        "component_geometry_groups",
        "component_mean_hidden_derivative",
    }
    assert set(jacobians) == base_keys | {f"anchor1_{name}" for name in base_keys}
    assert any(
        not torch.equal(value, state["state_dict"][name])
        for name, value in state["anchor_state_dict"].items()
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"arms": {"observed_near": state}}, checkpoint_path)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert restored["arms"]["observed_near"]["anchor_epoch"] == 1
    assert restored["arms"]["observed_near"]["anchor_state_dict"] is not None

    matrix_path = tmp_path / "jacobians.npz"
    expected_matrix_keys = {f"observed_near_{name}" for name in jacobians}
    np.savez_compressed(
        matrix_path,
        genes=np.asarray(["g0", "g1", "g2", "g3"]),
        **{f"observed_near_{name}": value for name, value in jacobians.items()},
        eligible_observed_near=np.asarray(result["eligible_genes"], dtype=bool),
    )
    with np.load(matrix_path, allow_pickle=False) as archive:
        assert set(archive.files) == expected_matrix_keys | {
            "genes",
            "eligible_observed_near",
        }
        for name, value in jacobians.items():
            assert np.array_equal(archive[f"observed_near_{name}"], value)


def test_checkpoint_replay_fails_on_serialized_weight_drift() -> None:
    problem = _small_training_problem()
    live = _model_from_state(_initial_small_model_state())
    changed_state = copy.deepcopy(live.state_dict())
    changed_state["morphology.weight"][0, 0] += 1.0
    state = {
        "model_kwargs": {
            "gene_count": 4,
            "morphology_count": 3,
            "hidden_count": 3,
            "use_neighbor": True,
        },
        "state_dict": changed_state,
    }

    with pytest.raises(runner.NonlinearRunError, match="checkpoint GPU replay"):
        runner._checkpoint_replay_control(
            live,
            state=state,
            target=problem["target"],
            morphology=problem["morphology"],
            feature=problem["feature"],
            mask=problem["train_mask"],
            groups=problem["groups"],
            target_stats=problem["target_stats"],
            morphology_stats=problem["morphology_stats"],
            device=problem["device"],
        )


def test_convergence_configuration_has_fold_invariant_full_scientific_id(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(runner, "EXPERIMENT_FLAVOR", "convergence")
    monkeypatch.setattr(runner, "EPOCH_CANDIDATES", (12, 24, 48, 96, 192))
    monkeypatch.setattr(
        runner, "PILOT_EPOCH_CANDIDATES", (12, 24, 48, 96, 192)
    )
    monkeypatch.setattr(runner, "PILOT_ARMS", runner.FULL_ARMS)
    monkeypatch.setattr(runner, "PILOT_REFIT_EPOCH_OVERRIDE", 192)
    monkeypatch.setattr(runner, "ANCHOR_EPOCH", 12)

    full_configs = [
        runner._configuration(profile="full", fold=fold, attempt=fold + 1)
        for fold in range(4)
    ]
    full_ids = {scientific_id(configuration) for configuration in full_configs}
    pilot = runner._configuration(profile="pilot", fold=0, attempt=1)

    assert len(full_ids) == 1
    assert scientific_id(pilot) not in full_ids
    assert pilot["trainer"]["learning_rate_schedule"] == "constant"
    assert pilot["trainer"]["effective_epoch_candidates"] == [12, 24, 48, 96, 192]
    assert pilot["trainer"]["pilot_refit_epoch_override"] == 192
    assert pilot["trainer"]["anchor_epoch"] == 12
    assert pilot["trainer"]["maximum_projected_hours_per_fold"] == 2.0
    assert pilot["evaluation"]["statistical_partition"] == "resource_validation"
    assert full_configs[0]["evaluation"]["statistical_partition"] == "outer_geometry_test"
    assert pilot["evaluation"]["canonical_prediction_split"] == "validation"
    assert pilot["evaluation"]["protocol"] == "resource_validation"
    assert full_configs[0]["evaluation"]["canonical_prediction_split"] == "test"
    assert (
        full_configs[0]["evaluation"]["protocol"]
        == "held_out_geometry_masked_reconstruction"
    )
    assert pilot["evaluation"]["primary_metric"].startswith("validation/")
    assert full_configs[0]["evaluation"]["primary_metric"].startswith("test/")
    assert full_configs[0]["dataset"]["version"] == runner.DATASET_VERSION
    assert full_configs[0]["model"]["embedding_dim"] == runner.HIDDEN_COUNT
    assert full_configs[0]["graph"]["neighbor_k"] == 12
    assert full_configs[0]["features"]["use_edge_features"] is False
    assert full_configs[0]["seed"] == runner.TRACKING_SEED
    assert full_configs[0]["fold"] == 0
    assert full_configs[0]["attempt"] == 1
    assert "seed_base" not in full_configs[0]["trainer"]
    assert pilot["masking"]["receiver_expression_input"] is False

    changed_schedule = copy.deepcopy(full_configs[0])
    changed_schedule["trainer"]["learning_rate_schedule"] = "cosine"
    assert scientific_id(changed_schedule) not in full_ids


def test_model_replicate_seed_is_execution_identity_not_scientific_variant(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(runner, "SEED_BASE", 20260810)
    monkeypatch.setattr(runner, "TRACKING_SEED", 260810)
    first = runner._configuration(profile="full", fold=0, attempt=1)
    monkeypatch.setattr(runner, "SEED_BASE", 20260814)
    monkeypatch.setattr(runner, "TRACKING_SEED", 260814)
    second = runner._configuration(profile="full", fold=3, attempt=2)

    assert first["trainer"]["model_seed"] == 20260810
    assert second["trainer"]["model_seed"] == 20260817
    assert scientific_id(first) == scientific_id(second)
