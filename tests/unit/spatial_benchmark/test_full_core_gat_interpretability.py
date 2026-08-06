from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from spatial_benchmark.configuration import compose_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "analysis"
    / "analyze_full_core_gat_interpretability.py"
)


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "analyze_full_core_gat_interpretability",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_hash_receiver_selection_is_deterministic_and_order_independent() -> None:
    module = _load_script()
    candidates = np.asarray([29, 3, 17, 8, 41, 11], dtype=np.int64)

    selected = module._deterministic_hash_select(
        candidates,
        3,
        namespace="unit-test",
    )
    permuted = module._deterministic_hash_select(
        candidates[::-1],
        3,
        namespace="unit-test",
    )

    assert np.array_equal(selected, permuted)
    assert len(selected) == 3
    assert set(selected).issubset(set(candidates))
    # Expression cannot affect this selection because it is not an input.
    assert "expression" not in inspect.signature(
        module._deterministic_hash_select
    ).parameters
    assert np.array_equal(
        selected,
        module._deterministic_hash_select(
            candidates,
            3,
            namespace="unit-test",
        ),
    )


def test_whole_node_mask_validation_broadcasts_rows_and_rejects_partial() -> None:
    module = _load_script()
    valid = np.asarray(
        [
            [False, False, False],
            [True, True, True],
            [False, False, False],
        ],
        dtype=np.bool_,
    )
    assert np.array_equal(
        module._whole_node_selected_rows(valid),
        [False, True, False],
    )

    invalid = valid.copy()
    invalid[0, 1] = True
    with pytest.raises(
        module.InterpretabilityContractError,
        match="partial rows",
    ):
        module._whole_node_selected_rows(invalid)


def test_fixed_whole_node_mask_reconstruction_matches_manifest() -> None:
    module = _load_script()
    coordinates = np.asarray(
        [(x, y) for y in range(16) for x in range(16)],
        dtype=np.float64,
    )
    core = SimpleNamespace(
        coordinates_um=coordinates,
        n_genes=6,
    )
    config = {
        "dataset": {
            "dataset_id": "synthetic-full-core",
            "version": "v1",
            "split_fingerprint": "synthetic-split",
        },
        "evaluation": {"mask_replicates_per_mode": 3},
        "masking": {
            "rates": {
                "partial_gene": 0.2,
                "whole_node": 0.75,
                "spatial_block": 0.1,
            },
            "mask_seed": 314159,
            "block_shape": "disk",
            "block_width_um": None,
        },
    }
    dataset = config["dataset"]
    seed = module.derive_mask_seed(
        314159,
        "held-in-full-core-fixed-evaluation",
        dataset["dataset_id"],
        dataset["version"],
        dataset["split_fingerprint"],
    )
    specs = [
        module.MaskSpec(
            mode=mode,
            partial_gene_rate=0.2,
            node_rate=0.75,
            block_node_rate=0.1,
            block_width_um=None,
            block_shape="disk",
            label=label,
        )
        for mode, label in (
            ("partial", "partial_gene"),
            ("node", "whole_node"),
            ("block", "spatial_block"),
        )
    ]
    expected = module.create_fixed_mask_bundle(
        {"fit": coordinates},
        6,
        specs,
        replicates=3,
        base_seed=seed,
    )

    mask, rebuilt = module._rebuild_verified_whole_node_mask(
        core,
        config,
        {"bundle_manifest": expected.manifest},
        {"evaluation_mask_bundle_sha256": expected.checksum},
        {"evaluation_mask_bundle_sha256": expected.checksum},
    )

    assert rebuilt.checksum == expected.checksum
    assert np.array_equal(mask, expected.get("fit", "whole_node", 0))
    selected = module._whole_node_selected_rows(mask)
    assert int(selected.sum()) >= module.HASH_SAMPLE_SIZE


def test_locked_protocol_accepts_repository_g2_and_rejects_pilot() -> None:
    module = _load_script()
    config = compose_config(
        PROJECT_ROOT / "configs/experiment/full_core_high_k_g2.yaml",
        config_root=PROJECT_ROOT / "configs",
    )
    manifest = {"campaign_id": module.CAMPAIGN_ID}
    summary = {
        "status": "success",
        "training_exit_status": "success",
        "evaluation_protocol": module.PROTOCOL,
        "model_name": "g2",
        "model_seed": 0,
        "final_epoch": 199,
        "fixed_epoch_budget": 200,
        "checkpoint_role": "last",
        "diagnostic_resource_pilot": False,
        "conclusion_eligible": True,
        "generalization_estimate": False,
        "graph_sha256": module.EXPECTED_GRAPH_SHA256,
        "graph_directed_edges": module.EXPECTED_DIRECTED_EDGES,
        "canonical_prediction_selection": {
            "split": "fit",
            "mask_mode": "whole_node",
            "mask_replicate": 0,
        },
    }

    module._validate_locked_protocol(config, manifest, summary)
    pilot = deepcopy(summary)
    pilot["diagnostic_resource_pilot"] = True
    with pytest.raises(
        module.InterpretabilityContractError,
        match="diagnostic_resource_pilot",
    ):
        module._validate_locked_protocol(config, manifest, pilot)


def test_attention_helpers_compute_entropy_effective_count_and_distance() -> None:
    module = _load_script()
    receiver = np.asarray([0, 0, 1, 1], dtype=np.int64)
    attention = np.asarray(
        [
            [0.8, 0.6],
            [0.2, 0.4],
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    distance = np.asarray([10.0, 30.0, 20.0, 40.0])

    arrays = module._receiver_attention_arrays(
        receiver,
        attention,
        distance,
        np.asarray([0, 1], dtype=np.int64),
    )
    expected_entropy_0 = -(0.7 * np.log(0.7) + 0.3 * np.log(0.3))
    expected_entropy_1 = np.log(2.0)

    assert np.array_equal(arrays["incoming_degree"], [2.0, 2.0])
    assert arrays["attention_entropy_nats"] == pytest.approx(
        [expected_entropy_0, expected_entropy_1]
    )
    assert arrays["effective_neighbor_count"] == pytest.approx(
        [np.exp(expected_entropy_0), 2.0]
    )
    assert arrays["attention_weighted_distance_um"] == pytest.approx(
        [16.0, 30.0]
    )
    aggregate = module._aggregate_attention(
        receiver,
        attention,
        distance,
        np.asarray([0, 1], dtype=np.int64),
    )
    assert aggregate["receiver_count"] == 2
    assert aggregate["incoming_degree"]["mean"] == 2.0


def test_deletion_plan_is_exact_deterministic_and_distance_bin_matched() -> None:
    module = _load_script()
    edge_ids = np.arange(40, dtype=np.int64)
    receiver = np.repeat(np.asarray([5, 9], dtype=np.int64), 20)
    distance = np.tile(np.arange(1, 21, dtype=np.float64), 2)
    attention = np.tile(np.arange(1, 21, dtype=np.float64), 2)
    selected = np.asarray([5, 9], dtype=np.int64)

    first = module._build_distance_matched_deletion_plan(
        edge_ids,
        receiver,
        attention,
        distance,
        selected,
        fraction=0.10,
        namespace="unit-test",
    )
    second = module._build_distance_matched_deletion_plan(
        edge_ids,
        receiver,
        attention,
        distance,
        selected,
        fraction=0.10,
        namespace="unit-test",
    )

    for key in first:
        assert np.array_equal(first[key], second[key])
    assert np.array_equal(first["top_edge_ids"], [18, 19, 38, 39])
    assert len(first["matched_edge_ids"]) == len(first["top_edge_ids"]) == 4
    assert not np.intersect1d(
        first["top_edge_ids"], first["matched_edge_ids"]
    ).size
    assert np.array_equal(first["deleted_count_per_receiver"], [2, 2])

    for offset, node in enumerate(selected):
        positions = np.flatnonzero(receiver == node)
        local_ids = edge_ids[positions]
        bins = module._rank_distance_bins(
            distance[positions],
            local_ids,
            int(first["distance_bin_count_per_receiver"][offset]),
        )
        top_local = np.isin(local_ids, first["top_edge_ids"])
        matched_local = np.isin(local_ids, first["matched_edge_ids"])
        assert np.array_equal(
            np.bincount(
                bins[top_local],
                minlength=int(
                    first["distance_bin_count_per_receiver"][offset]
                ),
            ),
            np.bincount(
                bins[matched_local],
                minlength=int(
                    first["distance_bin_count_per_receiver"][offset]
                ),
            ),
        )


def test_program_deletion_metrics_report_paired_model_sensitivity() -> None:
    module = _load_script()
    target = np.zeros((3, 2), dtype=np.float64)
    baseline = np.zeros_like(target)
    top_deleted = np.ones_like(target)
    matched_deleted = np.full_like(target, 0.5)

    metrics = module._program_deletion_metrics(
        target,
        baseline,
        top_deleted,
        matched_deleted,
        huber_delta=1.0,
    )

    assert metrics["baseline_program_huber"]["mean"] == 0.0
    assert (
        metrics["top_attention_deletion"][
            "program_huber_change_from_baseline"
        ]["mean"]
        == pytest.approx(0.5)
    )
    assert (
        metrics["distance_matched_random_deletion"][
            "program_huber_change_from_baseline"
        ]["mean"]
        == pytest.approx(0.125)
    )
    paired = metrics["paired_top_minus_matched"]
    assert paired["program_huber_change"]["mean"] == pytest.approx(0.375)
    assert paired["program_prediction_mae"]["mean"] == pytest.approx(0.5)
    assert paired["fraction_huber_contrast_positive"] == 1.0


def test_bounded_sender_program_plan_and_mean_ablation_are_paired() -> None:
    module = _load_script()
    edge_ids = np.arange(12, dtype=np.int64)
    source = np.arange(12, dtype=np.int64)
    attention = np.asarray(
        [0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.3, 0.2]
    )
    distance = np.asarray(
        [10, 20, 30, 40, 11, 19, 31, 41, 15, 25, 35, 45],
        dtype=np.float64,
    )
    eligible = np.ones(12, dtype=np.bool_)
    top = module._select_bounded_top_sender_sources(
        edge_ids,
        source,
        attention,
        distance,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        eligible,
        count=3,
    )
    null = module._select_distance_matched_null_senders(
        edge_ids,
        source,
        attention,
        distance,
        np.asarray([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64),
        eligible,
        top,
        namespace="unit-test-null-senders",
    )

    assert len(top["source_nodes"]) == len(null["source_nodes"]) == 3
    assert not np.intersect1d(
        top["source_nodes"], null["source_nodes"]
    ).size
    assert len(null["absolute_distance_mismatch_um"]) == 3
    expression = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    original = expression.clone()
    perturbed = module._mean_ablate_sender_program(
        expression,
        top["source_nodes"],
        np.asarray([1, 3], dtype=np.int64),
    )
    assert torch.equal(expression, original)
    assert bool(
        (
            perturbed[
                torch.as_tensor(top["source_nodes"])[:, None],
                torch.tensor([1, 3])[None, :],
            ]
            == 0.0
        ).all()
    )


def test_repeated_null_metrics_enforce_minimum_effect_across_draws() -> None:
    module = _load_script()
    target = np.zeros((4, 2), dtype=np.float64)
    baseline = np.zeros_like(target)
    targeted = np.ones_like(target)
    nulls = [
        np.full_like(target, 0.10),
        np.full_like(target, 0.20),
        np.full_like(target, 0.25),
    ]
    metrics = module._repeated_program_effect_metrics(
        target,
        baseline,
        targeted,
        nulls,
        huber_delta=1.0,
    )

    assert metrics["matched_null_replicates"] == 3
    paired = metrics["paired_targeted_minus_null"]
    assert (
        paired["minimum_prediction_mae_contrast_across_draws"]
        == pytest.approx(0.75)
    )
    assert module._minimum_effect_support(metrics) is True

    weak = module._repeated_program_effect_metrics(
        target,
        baseline,
        targeted,
        [np.full_like(target, 0.999)],
        huber_delta=1.0,
    )
    assert (
        weak["paired_targeted_minus_null"][
            "minimum_prediction_mae_contrast_across_draws"
        ]
        > 0.0
    )
    assert module._minimum_effect_support(weak) is False


def test_state_checksum_matches_training_protocol_algorithm() -> None:
    module = _load_script()
    state = {
        "beta": torch.tensor([1.0, -2.0], dtype=torch.float32),
        "alpha": torch.tensor([[3, 4]], dtype=torch.int64),
    }
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))

    assert module._state_dict_sha256(state) == digest.hexdigest()


def test_aggregate_output_is_exclusive_json_and_markdown(
    tmp_path: Path,
) -> None:
    module = _load_script()
    report = {
        "run_id": "synthetic-run",
        "attention_routing": {
            "expression_independent_hash_sample": {
                "receiver_count": 128,
                "attention_entropy_nats": {"mean": 1.0},
                "effective_neighbor_count": {"mean": 2.0},
                "attention_weighted_distance_um": {"mean": 3.0},
            }
        },
        "deletion_analysis": {
            "receiver_count": 64,
            "matched_null_replicates": 8,
            "receiver_program_effect": {
                "paired_targeted_minus_null": {
                    "program_huber_change_draw_means": {"mean": 0.2},
                    "program_prediction_mae_draw_means": {"mean": 0.1},
                }
            },
        },
        "sender_program_perturbation": {
            "source_count_each_condition": 16,
            "matched_null_replicates": 8,
            "organizer_score_enrichment": {
                "top_minus_null_score_draw_means": {"mean": 0.4}
            },
            "receiver_program_effect": {
                "paired_targeted_minus_null": {
                    "program_huber_change_draw_means": {"mean": 0.05}
                }
            },
        },
        "interpretation": {
            "descriptive_deletion_support": True,
            "descriptive_sender_program_support": True,
            "descriptive_organizer_enrichment_support": True,
            "joint_tls_dependency_support": True,
            "evidence_label": module.MAXIMUM_INTERPRETATION,
        },
        "privacy": {
            "aggregate_only": True,
            "protected_identifiers_emitted": False,
            "local_node_positions_emitted": False,
            "raw_edge_positions_emitted": False,
        },
    }
    module._assert_aggregate_only(report)
    markdown = module._render_markdown(
        {
            **report,
            "protocol": module.PROTOCOL,
        }
    )
    destination = tmp_path / "analysis"
    module._write_exclusive_output(destination, report, markdown)

    assert sorted(path.name for path in destination.iterdir()) == [
        "analysis.json",
        "report.md",
    ]
    assert (
        json.loads(
            (destination / "analysis.json").read_text(encoding="utf-8")
        )
        == report
    )
    assert "Attention is not importance" in (
        destination / "report.md"
    ).read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        module._write_exclusive_output(destination, report, markdown)
    with pytest.raises(
        module.InterpretabilityContractError,
        match="forbidden row-level key",
    ):
        module._assert_aggregate_only(
            {"receiver_indices": [1, 2, 3]}
        )
    with pytest.raises(
        module.InterpretabilityContractError,
        match="row-like or numeric sequence",
    ):
        module._assert_aggregate_only({"quantiles": [0.1, 0.2]})
    with pytest.raises(
        module.InterpretabilityContractError,
        match="row-like or numeric sequence",
    ):
        module._assert_aggregate_only(
            {"records": [{"mean": 0.1}, {"mean": 0.2}]}
        )
