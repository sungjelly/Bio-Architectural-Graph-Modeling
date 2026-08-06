from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.masking import (  # noqa: E402
    MaskSpec,
    apply_expression_mask,
    create_fixed_mask_bundle,
    curriculum_mode,
    generate_mask,
    load_fixed_mask_bundle,
    paired_epoch_seed,
    save_fixed_mask_bundle,
)
from spatial_benchmark.metrics import (  # noqa: E402
    benjamini_hochberg,
    bh_fdr,
    block_bootstrap_ci,
    ensemble_predictions,
    evaluate_masked_predictions,
    masked_huber_loss,
    masked_mae_loss,
    masked_mse_loss,
    masked_r2_score,
    paired_sign_flip_test,
    paired_spatial_gain,
)


def grid_coordinates(n_x: int = 8, n_y: int = 6, spacing: float = 10.0) -> np.ndarray:
    x, y = np.meshgrid(np.arange(n_x), np.arange(n_y), indexing="xy")
    return np.column_stack([x.ravel(), y.ravel()]).astype(float) * spacing


def test_partial_mask_is_exact_paired_and_expression_only() -> None:
    coordinates = grid_coordinates()
    eligible = np.ones(coordinates.shape[0], dtype=bool)
    eligible[[0, 3, 9]] = False
    first = generate_mask(
        "partial",
        n_genes=10,
        coordinates_um=coordinates,
        eligible_nodes=eligible,
        seed=481,
        partial_gene_rate=0.30,
    )
    repeated = generate_mask(
        MaskSpec("partial", partial_gene_rate=0.30),
        n_genes=10,
        coordinates_um=coordinates,
        eligible_nodes=np.flatnonzero(eligible),
        seed=481,
    )
    different = generate_mask(
        "partial",
        n_genes=10,
        coordinates_um=coordinates,
        eligible_nodes=eligible,
        seed=482,
        partial_gene_rate=0.30,
    )

    assert first.mask.dtype == np.bool_
    assert first.mask.shape == (coordinates.shape[0], 10)
    np.testing.assert_array_equal(first.mask, repeated.mask)
    assert not np.array_equal(first.mask, different.mask)
    np.testing.assert_array_equal(first.mask[eligible].sum(axis=1), 3)
    assert not first.mask[~eligible].any()
    assert first.achieved_entry_rate == pytest.approx(0.30)

    expression = np.arange(first.mask.size, dtype=np.float32).reshape(first.mask.shape)
    metadata = np.arange(coordinates.shape[0] * 4, dtype=np.float32).reshape(-1, 4)
    original_expression = expression.copy()
    original_metadata = metadata.copy()
    inputs = first.apply(expression, metadata)
    assert np.all(inputs.expression[first.mask] == 0)
    np.testing.assert_array_equal(
        inputs.expression[~first.mask],
        original_expression[~first.mask],
    )
    np.testing.assert_array_equal(inputs.gene_mask, first.mask.astype(np.float32))
    np.testing.assert_array_equal(inputs.metadata, original_metadata)
    np.testing.assert_array_equal(metadata, original_metadata)
    np.testing.assert_array_equal(expression, original_expression)
    assert inputs.metadata is not metadata


def test_whole_node_mask_and_curriculum_use_a_separate_mask_seed_stream() -> None:
    coordinates = grid_coordinates(10, 5)
    eligible = np.arange(5, 50)
    batch = generate_mask(
        "whole-node",
        7,
        coordinates,
        eligible_nodes=eligible,
        seed=99,
        node_rate=0.20,
    )
    assert batch.n_eligible_nodes == 45
    assert batch.n_selected_nodes == 9
    assert np.all(batch.mask[batch.selected_nodes])
    assert not batch.mask[~batch.selected_nodes].any()

    # The pairing key contains the mask seed, epoch, and batch only.  A model
    # initialisation seed is intentionally not accepted by either function.
    assert paired_epoch_seed(77, 12, 3) == paired_epoch_seed(77, 12, 3)
    assert paired_epoch_seed(77, 12, 3) != paired_epoch_seed(77, 12, 4)
    assert curriculum_mode(0, 77, "P+N+B") == "partial"
    assert curriculum_mode(9, 77, "P+N+B") == "partial"
    schedule_a = [
        curriculum_mode(12, 77, "P+N+B", batch_index=index)
        for index in range(40)
    ]
    schedule_b = [
        curriculum_mode(12, 77, "P+N+B", batch_index=index)
        for index in range(40)
    ]
    assert schedule_a == schedule_b
    assert set(schedule_a) <= {"partial", "node", "block"}
    assert curriculum_mode(100, 1, "P-only", batch_index=10) == "partial"


def test_spatial_block_is_one_contiguous_physical_region_and_reports_rate() -> None:
    coordinates = grid_coordinates(11, 11, spacing=10.0)
    disk = generate_mask(
        MaskSpec("block", block_width_um=45.0, block_shape="disk"),
        n_genes=6,
        coordinates_um=coordinates,
        seed=123,
    )
    center = np.asarray(disk.details["center_um"])
    selected_coordinates = coordinates[disk.selected_nodes]
    distances = np.linalg.norm(selected_coordinates - center, axis=1)
    assert np.all(distances <= 22.5 + 1e-12)
    assert disk.n_selected_nodes > 0
    assert np.all(disk.mask[disk.selected_nodes])
    assert disk.details["achieved_node_rate"] == pytest.approx(
        disk.n_selected_nodes / coordinates.shape[0]
    )

    nearest_disk = generate_mask(
        MaskSpec("block", block_node_rate=0.20),
        n_genes=3,
        coordinates_um=coordinates,
        seed=5,
    )
    center = np.asarray(nearest_disk.details["center_um"])
    radius = nearest_disk.details["block_width_um"] / 2
    selected_distance = np.linalg.norm(
        coordinates[nearest_disk.selected_nodes] - center,
        axis=1,
    )
    unselected_distance = np.linalg.norm(
        coordinates[~nearest_disk.selected_nodes] - center,
        axis=1,
    )
    assert np.all(selected_distance <= radius + 1e-12)
    assert np.all(unselected_distance >= radius - 1e-12)
    assert nearest_disk.achieved_node_rate >= 0.20


def test_fixed_validation_test_replicates_roundtrip_and_detect_tampering(
    tmp_path: Path,
) -> None:
    coordinates = {
        "validation": grid_coordinates(7, 5),
        "test": grid_coordinates(6, 4) + np.array([200.0, 300.0]),
    }
    specs = [
        MaskSpec("partial", partial_gene_rate=0.25),
        MaskSpec("node", node_rate=0.20),
        MaskSpec("block", block_node_rate=0.25),
    ]
    first = create_fixed_mask_bundle(
        coordinates,
        n_genes=12,
        specs=specs,
        replicates=2,
        base_seed=2026,
    )
    repeated = create_fixed_mask_bundle(
        coordinates,
        n_genes=12,
        specs=specs,
        replicates=2,
        base_seed=2026,
    )
    assert first.checksum == repeated.checksum
    assert first.bundle_id == repeated.bundle_id
    assert len(first.masks) == 2 * 3 * 2
    for key in first.masks:
        np.testing.assert_array_equal(first.masks[key], repeated.masks[key])

    output = save_fixed_mask_bundle(first, tmp_path / "fixed_masks")
    assert (output / "manifest.json").is_file()
    assert (output / "masks.npz").is_file()
    assert (output / "checksums.sha256").is_file()
    loaded = load_fixed_mask_bundle(output)
    assert loaded.checksum == first.checksum
    np.testing.assert_array_equal(
        loaded.get("validation", "node", replicate=1),
        first.get("validation", "node", replicate=1),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_fixed_mask_bundle(first, output)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["seed_contract"].endswith("model seeds are excluded.")
    with (output / "masks.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="file checksum mismatch"):
        load_fixed_mask_bundle(output)


def test_masked_numpy_losses_ignore_unmasked_and_nonfinite_values() -> None:
    target = np.array([[0.0, 1.0, np.nan], [2.0, 3.0, 4.0]])
    prediction = np.array([[2.0, 2.0, 99.0], [1.0, -100.0, 4.5]])
    mask = np.array([[True, False, True], [True, False, True]])
    # Valid masked errors are [2, -1, .5].
    assert masked_mse_loss(target, prediction, mask) == pytest.approx(1.75)
    assert masked_mae_loss(target, prediction, mask) == pytest.approx(3.5 / 3)
    expected_huber = (1.5 + 0.5 + 0.125) / 3
    assert masked_huber_loss(target, prediction, mask) == pytest.approx(expected_huber)
    none = masked_mse_loss(target, prediction, mask, reduction="none")
    assert none[0, 2] == 0.0
    assert none[1, 1] == 0.0
    with pytest.raises(ValueError, match="selects no finite"):
        masked_mse_loss(target, prediction, np.zeros_like(mask))


def test_masked_torch_loss_has_gradients_only_on_masked_entries() -> None:
    torch = pytest.importorskip("torch")
    target = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    prediction = torch.tensor([[1.0, 9.0], [4.0, 2.0]], requires_grad=True)
    mask = torch.tensor([[True, False], [False, True]])
    loss = masked_huber_loss(target, prediction, mask)
    loss.backward()
    assert loss.item() == pytest.approx(0.5)
    assert prediction.grad[0, 1].item() == 0.0
    assert prediction.grad[1, 0].item() == 0.0
    assert prediction.grad[0, 0].item() != 0.0
    assert prediction.grad[1, 1].item() != 0.0

    metadata = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    expression = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    applied = apply_expression_mask(expression, mask.repeat(2, 1), metadata)
    assert torch.equal(applied.metadata, metadata)
    assert applied.metadata.data_ptr() != metadata.data_ptr()


def test_masked_r2_is_unclipped_and_zero_variance_is_undefined() -> None:
    target = np.array([[0.0, 2.0, 99.0], [np.nan, 7.0, 7.0]])
    mask = np.array([[True, True, False], [True, False, False]])

    perfect = target.copy()
    assert masked_r2_score(target, perfect, mask) == pytest.approx(1.0)

    mean_prediction = np.array([[1.0, 1.0, -999.0], [0.0, 0.0, 0.0]])
    assert masked_r2_score(target, mean_prediction, mask) == pytest.approx(0.0)

    worse_than_mean = np.array([[0.0, 4.0, -999.0], [0.0, 0.0, 0.0]])
    assert masked_r2_score(target, worse_than_mean, mask) == pytest.approx(-1.0)
    evaluated = evaluate_masked_predictions(
        target,
        worse_than_mean,
        mask,
    )
    assert evaluated["r2"] == pytest.approx(-1.0)
    assert evaluated["percent_variance_explained"] == pytest.approx(-100.0)

    constant_target = np.array([[3.0, 3.0], [3.0, 3.0]])
    constant_mask = np.ones_like(constant_target, dtype=bool)
    assert np.isnan(
        masked_r2_score(
            constant_target,
            np.zeros_like(constant_target),
            constant_mask,
        )
    )
    constant_metrics = evaluate_masked_predictions(
        constant_target,
        np.zeros_like(constant_target),
        constant_mask,
    )
    assert np.isnan(constant_metrics["r2"])
    assert np.isnan(constant_metrics["percent_variance_explained"])

    with pytest.raises(ValueError, match="selects no finite"):
        masked_r2_score(target, perfect, np.zeros_like(mask))


def test_metric_registry_declares_r2_percentage_as_an_exact_transform() -> None:
    registry = yaml.safe_load(
        (PROJECT_ROOT / "configs/schema/metrics_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    family = registry["task_families"]["masked_expression_regression"]
    contract = family["masked_r2_contract"]
    assert contract["negative_values"] == "retained_without_clipping"
    assert contract["zero_target_variance"] == "undefined"
    assert contract["percentage_reconciliation"] == (
        "100 * aggregate_masked_r2"
    )
    metrics = family["metrics"]
    for split_and_mode in (
        "fit/partial_gene",
        "fit/whole_node",
        "fit/spatial_block",
        "val",
        "test",
        "external",
    ):
        r2_name = f"{split_and_mode}/masked_r2"
        percentage_name = (
            f"{split_and_mode}/masked_percent_variance_explained"
        )
        assert metrics[r2_name]["direction"] == "maximize"
        assert metrics[percentage_name]["direction"] == "maximize"
        assert metrics[percentage_name]["reconciliation"] == (
            f"100 * {r2_name}"
        )


def test_evaluation_ensembles_seeds_and_guards_invalid_correlations() -> None:
    target = np.array(
        [
            [0.0, 5.0, 1.0],
            [1.0, 5.0, 2.0],
            [2.0, 5.0, 3.0],
            [3.0, 5.0, 4.0],
            [4.0, 5.0, 5.0],
            [5.0, 5.0, 6.0],
        ]
    )
    first_seed = target - 1.0
    second_seed = target + 1.0
    predictions = np.stack([first_seed, second_seed])
    mask = np.ones_like(target, dtype=bool)
    block_ids = np.array(["a", "a", "a", "b", "b", "b"])

    ensemble = ensemble_predictions(predictions)
    np.testing.assert_allclose(ensemble, target)
    metrics = evaluate_masked_predictions(target, predictions, mask, block_ids)
    assert metrics["n_prediction_seeds"] == 2
    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["huber"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["percent_variance_explained"] == pytest.approx(100.0)
    assert len(metrics["per_seed"]) == 2
    assert all(item["technical_only"] for item in metrics["per_seed"])
    assert len(metrics["blocks"]) == 2
    assert sum(block["n_masked"] for block in metrics["blocks"]) == mask.size
    assert metrics["gene"]["pearson"][0] == pytest.approx(1.0)
    assert np.isnan(metrics["gene"]["pearson"][1])  # constant target guard
    assert metrics["gene"]["spearman"][2] == pytest.approx(1.0)

    # Whole-node masks have enough genes for per-cell correlations, whereas a
    # one-gene partial mask correctly returns NaN instead of an invalid value.
    one_gene_mask = np.zeros_like(mask)
    one_gene_mask[:, 0] = True
    one_gene = evaluate_masked_predictions(target, target, one_gene_mask)
    assert np.isnan(one_gene["cell"]["pearson"]).all()


def test_block_bootstrap_sign_flip_and_primary_paired_gain_are_deterministic() -> None:
    target = np.zeros((16, 2), dtype=float)
    mask = np.ones_like(target, dtype=bool)
    block_ids = np.repeat(np.arange(8), 2)
    # The baseline has error 2, the graph error 1 in every block.  Two seeds
    # straddle each ensemble value so seed count cannot inflate n_blocks.
    baseline = np.stack(
        [np.full_like(target, 1.5), np.full_like(target, 2.5)]
    )
    spatial = np.stack(
        [np.full_like(target, 0.5), np.full_like(target, 1.5)]
    )
    first = paired_spatial_gain(
        target,
        baseline,
        spatial,
        mask,
        block_ids,
        loss="mse",
        n_bootstrap=500,
        bootstrap_seed=91,
    )
    second = paired_spatial_gain(
        target,
        baseline,
        spatial,
        mask,
        block_ids,
        loss="mse",
        n_bootstrap=500,
        bootstrap_seed=91,
    )
    assert first["n_blocks"] == 8
    assert first["n_baseline_prediction_seeds"] == 2
    assert first["n_spatial_prediction_seeds"] == 2
    assert first["delta"] == pytest.approx(3.0)
    assert first["relative_gain"] == pytest.approx(0.75)
    assert first["delta_ci"] == second["delta_ci"]
    assert first["delta_ci"]["lower"] == pytest.approx(3.0)
    assert first["sign_flip"]["method"] == "exact_sign_flip"
    assert first["sign_flip"]["p_value"] == pytest.approx(1 / 256)
    assert first["inference_unit"] == "spatial_block"
    assert "model_seed" in first["technical_units_not_replicates"]

    ci_a = block_bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_resamples=500, seed=9)
    ci_b = block_bootstrap_ci([1.0, 2.0, 3.0, 4.0], n_resamples=500, seed=9)
    assert ci_a == ci_b
    exact = paired_sign_flip_test([1, 1, 1, 1], alternative="greater")
    assert exact["method"] == "exact_sign_flip"
    assert exact["p_value"] == pytest.approx(1 / 16)
    mc_a = paired_sign_flip_test(
        np.linspace(-1, 2, 25),
        exact_max_blocks=10,
        n_resamples=1_000,
        seed=44,
    )
    mc_b = paired_sign_flip_test(
        np.linspace(-1, 2, 25),
        exact_max_blocks=10,
        n_resamples=1_000,
        seed=44,
    )
    assert mc_a == mc_b
    assert mc_a["method"] == "monte_carlo_sign_flip"


def test_benjamini_hochberg_is_monotone_shape_preserving_and_nan_safe() -> None:
    p_values = np.array([[0.01, 0.04, np.nan], [0.03, 0.002, 1.0]])
    adjusted = benjamini_hochberg(p_values)
    assert adjusted.shape == p_values.shape
    assert np.isnan(adjusted[0, 2])
    finite_p = p_values[np.isfinite(p_values)]
    finite_q = adjusted[np.isfinite(p_values)]
    order = np.argsort(finite_p, kind="mergesort")
    assert np.all(np.diff(finite_q[order]) >= -1e-15)
    assert np.all(finite_q >= finite_p)
    decision = bh_fdr(p_values, alpha=0.05)
    np.testing.assert_allclose(
        decision["adjusted_p"],
        adjusted,
        equal_nan=True,
    )
    assert decision["reject"].dtype == bool
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        benjamini_hochberg([0.1, 1.1])
