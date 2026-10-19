import copy
import json

import pytest

from scripts.analysis.report_so2_bce_gradient_audit import summarize_gradients, validate_matrix


def fixture():
    rows = []
    for epoch in (1, 2):
        def tracker(scope):
            bootstrap = epoch == 1 and scope.startswith("blocks.")
            value = {"run_id": "synthetic", "model_seed": 0, "global_epoch": epoch,
                "optimizer_updates_observed": 6, "resume_boundary_unavailable": False,
                "trainable_parameter_count": 10, "gradient_norm_min_before_clip": 0. if bootstrap else .5,
                "gradient_norm_mean_before_clip": 1., "gradient_norm_max_before_clip": 1.5,
                "consecutive_optimizer_step_cosine_valid_pairs": (4 if bootstrap else 5) if epoch == 1 else 6,
                "consecutive_optimizer_step_cosine_mean": .1,
                "epoch_aggregate_gradient_cosine_to_previous_epoch": None if epoch == 1 else -.2}
            if scope.startswith("blocks."):
                value.update(block_name=scope, block_index=int(scope[-1]))
            return value
        rows.append({"global_epoch": epoch, "optimizer_updates": 6,
            "global_gradient_json": json.dumps(tracker("global")),
            "block_gradients_json": json.dumps([tracker(f"blocks.{i}") for i in range(4)]),
            "detection_head_gradient_json": json.dumps(tracker("head"))})
    updates = [{"run_id": "synthetic", "epoch": epoch, "update": step,
        "cumulative_optimizer_update": (epoch-1)*6+step, "gradient_norm": (.5, .5, 1., 1., 1.5, 1.5)[step-1]}
        for epoch in (1, 2) for step in range(1, 7)]
    return rows, updates


def audit(rows, updates):
    return summarize_gradients(rows, updates, run_id="synthetic", seed=0, steps=6, clip_norm=1.)


def test_clipping_bootstrap_and_later_zero_are_distinct():
    rows, updates = fixture()
    observed = audit(rows, updates)
    assert observed["total_updates"] == 12
    assert observed["updates_above_clip_threshold"] == 4
    assert observed["clipping_activation_fraction"] == 1/3
    assert observed["all_scope_epoch_means_positive"]
    assert observed["adjacent_pair_counts_match_bootstrap_expectation"]
    assert not observed["later_zero_minimum_detected"]
    blocks = json.loads(rows[1]["block_gradients_json"])
    blocks[2]["gradient_norm_min_before_clip"] = 0.
    rows[1]["block_gradients_json"] = json.dumps(blocks)
    assert audit(rows, updates)["later_zero_minimum_detected"]


@pytest.mark.parametrize("mutation", ("duplicate", "missing", "nonfinite", "wrong_counter"))
def test_rejects_incomplete_or_nonfinite_update_evidence(mutation):
    rows, updates = fixture()
    if mutation == "duplicate": updates[1] = dict(updates[0])
    elif mutation == "missing": updates.pop()
    elif mutation == "nonfinite": updates[0]["gradient_norm"] = float("nan")
    else: updates[2]["cumulative_optimizer_update"] = 99
    with pytest.raises(ValueError):
        audit(rows, updates)


def test_rejects_tracker_update_count_drift():
    rows, updates = fixture()
    blocks = json.loads(rows[1]["block_gradients_json"])
    blocks[0]["optimizer_updates_observed"] = 5
    rows[1]["block_gradients_json"] = json.dumps(blocks)
    with pytest.raises(ValueError, match="coverage"):
        audit(rows, updates)


def test_grid_requires_every_setting_and_excludes_regularization_substitutes():
    def config(mode, batch, lr):
        return {"metadata": {"execution_profile": "screen"}, "seed": 0, "fold": 0,
            "model": {"dropout": .1}, "trainer": {"max_epochs": 8, "optimizer_batch_mode": mode,
                "target_cells_per_optimizer_update": batch, "learning_rate": lr, "weight_decay": 1e-5}}
    configs = [config("target_window", b, lr) for b in (512, 1024, 2048) for lr in (3e-5, 1e-4, 3e-4)]
    configs.append(config("core_pair", None, 1e-4))
    validate_matrix(configs, "grid")
    wrong = copy.deepcopy(configs); wrong[0]["model"]["dropout"] = 0.
    with pytest.raises(ValueError, match="matrix"):
        validate_matrix(wrong, "grid")
    with pytest.raises(ValueError, match="Entire"):
        validate_matrix(configs[:-1], "grid")
