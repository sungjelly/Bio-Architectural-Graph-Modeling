from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from spatial_benchmark.identifiers import scientific_id
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
    assert all(np.array_equal(mask, masks["validation"]) for mask in observed_masks)
    assert all(np.array_equal(mask, masks["validation"]) for mask in detailed_masks)
    assert not any(np.array_equal(mask, masks["test"]) for mask in observed_masks)
    assert not any(np.array_equal(mask, masks["test"]) for mask in detailed_masks)

    assert jacobians is not None
    base_keys = {"total", "linear", "nonlinear", "mean_hidden_derivative"}
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
    assert pilot["masking"]["receiver_expression_input"] is False

    changed_schedule = copy.deepcopy(full_configs[0])
    changed_schedule["trainer"]["learning_rate_schedule"] = "cosine"
    assert scientific_id(changed_schedule) not in full_ids
