"""Focused CPU contracts for the multiscale continuous-hurdle runner."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from spatial_benchmark.identifiers import canonical_sha256


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/run_multiscale_hurdle_capacity.py"
_SPEC = importlib.util.spec_from_file_location(
    "test_multiscale_hurdle_capacity_runner_module", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _yaml(relative: str) -> dict[str, Any]:
    return dict(
        yaml.safe_load((_ROOT / relative).read_text(encoding="utf-8"))
    )


def _sha(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _permutation_receipt(n_nodes: int = 100) -> dict[str, Any]:
    x = np.arange(n_nodes, dtype=np.float64) * 20.0
    coordinates = np.stack((x, np.zeros_like(x)), axis=1)
    source = np.arange(n_nodes, dtype=np.int64)
    receiver = (source + 1) % n_nodes
    edge_index = np.stack((source, receiver), axis=0)
    edge_attributes = np.zeros(
        (n_nodes, len(_RUNNER.EDGE_ATTRIBUTE_NAMES)),
        dtype=np.float32,
    )
    result = _RUNNER.build_macroblock_spatial_antipode_permutation(
        coordinates,
        np.asarray(["fixture-block"] * n_nodes),
        local_edge_index=edge_index,
        local_edge_attributes=edge_attributes,
    )
    return dict(result.receipt)


def _config(
    arm: str = "self",
    *,
    pilot: bool = True,
    alias: str | None = None,
) -> dict[str, Any]:
    alias = alias or ("ANC-03" if pilot else "ANC-02")
    model_paths = {
        "self": "configs/model/multiscale_hurdle_self.yaml",
        "self-regional": (
            "configs/model/multiscale_hurdle_self_regional.yaml"
        ),
        "self-regional-local": (
            "configs/model/multiscale_hurdle_self_regional_local.yaml"
        ),
        "self-regional-local-permuted": (
            "configs/model/"
            "multiscale_hurdle_self_regional_local_permuted.yaml"
        ),
    }
    model = _yaml(model_paths[arm])
    trainer = _yaml(
        "configs/trainer/multiscale_hurdle_resource_pilot_2.yaml"
        if pilot
        else "configs/trainer/multiscale_hurdle_fixed_200.yaml"
    )
    evaluation = _yaml(
        "configs/evaluation/held_in_multiscale_hurdle_resource_pilot_v1.yaml"
        if pilot
        else "configs/evaluation/held_in_multiscale_hurdle_v1.yaml"
    )
    trainer["amp_authorization"] = (
        {
            "mode": "runner_internal_same_batch_equivalence",
            "maximum_absolute_loss_discrepancy": 0.001,
        }
        if pilot
        else {
            "mode": "require_external_resource_gate_receipt",
            "receipt_schema": _RUNNER._RESOURCE_GATE_KIND,
            "receipt_reference": _RUNNER._RESOURCE_GATE_RELATIVE.as_posix(),
            "frozen_contract_sha256": _RUNNER._FROZEN_CONTRACT_SHA256,
        }
    )
    graph = _yaml("configs/graph/multiscale_local64_regional256.yaml")
    graph.update(
        {
            "expected_materialized_graph_sha256": _sha("bundle"),
            "expected_graph_receipt_sha256": _sha("receipt"),
            "expected_directed_edges": 20,
            "expected_bundle_qc": {"verified": True},
            "local_source_permutation": _permutation_receipt(),
        }
    )
    for index, scale in enumerate(("local", "regional")):
        graph[scale].update(
            {
                "expected_graph_sha256": _sha(scale),
                "expected_directed_edges": 20 + index,
                "expected_components": 1,
                "expected_isolated_nodes": 0,
            }
        )
    fingerprint = _sha(f"{alias}-data")
    split = _sha(f"{alias}-split")
    representation = {
        "schema": _RUNNER._REPRESENTATION_SCHEMA,
        "source_scale": "raw_biological_probe_counts",
        "input_states": {
            "schema": _RUNNER._INPUT_STATE_SCHEMA,
            "num_states": 8,
            "mask_token_id": 8,
            "mask_token_is_output": False,
        },
        "output_channels_per_gene": 2,
        "output_channels": [
            "detection_logit",
            "positive_standardized_log1p",
        ],
        "detection_threshold": 0.5,
        "count_rounding": "nonnegative_half_up_floor_x_plus_0_5",
        "fixed_count_states": [
            "0",
            "1",
            "2",
            "3",
            "4-7",
            "8-15",
            "16-31",
            "32+",
        ],
        "continuous_transform": "per_gene_all_fit_standardized_log1p",
        "fit_required": False,
    }
    rates = {
        "partial_gene": 0.2,
        "whole_node": 0.1,
        "spatial_block": 0.1,
    }
    role = "resource_pilot" if pilot else "science"
    return {
        "model": model,
        "dataset": {
            "dataset_id": f"cosmx_{alias.lower().replace('-', '')}_fit",
            "version": "fit_v1",
            "split_id": split[:16],
            "dataset_fingerprint": fingerprint,
            "split_fingerprint": split,
            "prepared_artifact_reference": (
                f"data/processed/fixture/{alias.lower()}/prepared_v1"
            ),
            "task": _RUNNER._TASK_FAMILY,
            "target_scale": (
                "raw_biological_probe_counts_with_per_gene_all_fit_"
                "standardized_log1p"
            ),
            "count_representation": representation,
            "biological_target_count": 1000,
            "preprocessing_fit_scope": "all_nodes_transductive",
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "validation_or_test_partition_present": False,
            "patient_generalization_supported": False,
            "frozen_task_contract_sha256": (
                _RUNNER._FROZEN_CONTRACT_SHA256
            ),
        },
        "features": {
            "use_edge_features": True,
            "fit_scope": "all_nodes_transductive",
            "node_expression": {
                "biological_targets": 1000,
                "source_scale": "raw_biological_probe_counts",
                "discrete_transform": "fixed_hybrid_count_states",
                "continuous_transform": (
                    "per_gene_all_fit_standardized_log1p"
                ),
                "masked_discrete_value": "input_only_mask_token_8",
                "masked_continuous_value": 0.0,
                "explicit_mask_authoritative_inside_model": True,
                "prediction_schema": (
                    "detection_plus_positive_standardized_log1p"
                ),
            },
            "node_metadata": {
                "fields": list(_RUNNER.ALLOWED_METADATA_COLUMNS)
            },
            "edge_features": {
                "fit_scope": "true_local_and_regional_edges_transductive",
                "standardization": (
                    "per_scale_true_graph_edge_wise"
                ),
                "fields": list(_RUNNER.EDGE_ATTRIBUTE_NAMES),
            },
            "prohibited_node_inputs": [
                "direct_identifiers",
                "absolute_or_local_coordinates",
                "expression_derived_library_size",
                "rna_derived_qc",
                "vendor_cell_type_cluster_neighborhood_or_niche",
                "hidden_target_values",
            ],
        },
        "graph": graph,
        "masking": {
            "type": "mixed_expression_masking",
            "curriculum": "P+N+B",
            "rate": deepcopy(rates),
            "rates": deepcopy(rates),
            "post_warmup_probabilities": {
                "partial_gene": 0.6,
                "whole_node": 0.3,
                "spatial_block": 0.1,
            },
            "warmup_epochs": 10,
            "block_shape": "disk",
            "block_width_um": None,
            "mask_seed": 314159,
            "validation_replicates": 0,
            "test_replicates": 0,
            "fit_replicates": 1 if pilot else 3,
            "mask_expression_only": True,
            "explicit_gene_mask_channel": True,
        },
        "trainer": trainer,
        "evaluation": evaluation,
        "launcher": {
            "requested_gpu": "0",
            "requested_gpu_count": 1,
            "set_cuda_visible_devices": True,
            "concurrency": 1,
            "disk_safety_max_used_decimal_gb": 55.0,
        },
        "campaign": {
            "campaign_id": _RUNNER._CAMPAIGN_ID,
            "exploratory": True,
            "frozen_contract": (
                _RUNNER._FROZEN_CONTRACT_RELATIVE.as_posix()
            ),
            "frozen_contract_sha256": (
                _RUNNER._FROZEN_CONTRACT_SHA256
            ),
            "contract_amendment": (
                _RUNNER._CONTRACT_AMENDMENT_RELATIVE.as_posix()
            ),
            "contract_amendment_sha256": (
                _RUNNER._CONTRACT_AMENDMENT_SHA256
            ),
            "contract_supplement": (
                _RUNNER._CONTRACT_SUPPLEMENT_RELATIVE.as_posix()
            ),
            "contract_supplement_sha256": (
                _RUNNER._CONTRACT_SUPPLEMENT_SHA256
            ),
            "superseded_contract_amendment": (
                _RUNNER.SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.as_posix()
            ),
            "superseded_contract_amendment_sha256": (
                _RUNNER.SUPERSEDED_CONTRACT_AMENDMENT_SHA256
            ),
            "superseded_amendment_retained_as_negative_record": True,
        },
        "experiment": {
            "variant_label": (
                f"{alias.lower().replace('-', '')}_"
                f"{arm.replace('-', '_')}_{role}"
            ),
            "arm": arm,
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "estimand": "held_in_full_core_whole_node_masked_raw_count",
            "permitted_claim": (
                "exploratory_held_in_graph_specific_and_correctly_aligned_"
                "local_sender_state_predictive_dependency"
            ),
            "paired_within_core": True,
            "conclusion_eligible": not pilot,
            "excluded_from_primary_comparison": pilot,
            "resource_pilot": pilot,
            "stage": 1 if pilot or arm == "self" else 2,
        },
        "metadata": {
            "locked_config_materialization_receipt": (
                _RUNNER._MATERIALIZATION_RELATIVE.as_posix()
            ),
            "frozen_scientific_contract": True,
            "sender_state_permutation_amendment_enforced": True,
            "mask_noninterference_supplement_enforced": True,
            "original_rewired_arm_authorized": False,
            "execution_role": role,
            "production_requires_resource_gate": not pilot,
            "stage2_requires_representation_gate": (
                not pilot and arm != "self"
            ),
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }


@pytest.mark.parametrize(
    ("arm", "routing"),
    list(_RUNNER._ROUTING.items()),
)
def test_contract_accepts_exact_routes_and_stage(
    arm: str,
    routing: tuple[str, str],
) -> None:
    pilot = arm == "self"
    config = _config(arm, pilot=pilot)
    contract = _RUNNER._validate_multiscale_contract(config)
    assert (contract.regional_routing, contract.local_routing) == routing
    assert contract.uses_permuted_local is (arm.endswith("permuted"))

    drifted = deepcopy(config)
    drifted["model"]["local_routing"] = "true"
    if routing[1] != "true":
        with pytest.raises(
            _RUNNER.MultiscaleHurdleRunnerError,
            match="model.local_routing",
        ):
            _RUNNER._validate_multiscale_contract(drifted)


def test_contract_rejects_batch_epoch_graph_and_identifier_drift() -> None:
    config = _config()
    mutations = [
        ("trainer", "target_node_batch_size", 512, "target_node_batch_size"),
        ("trainer", "max_epochs", 3, "max_epochs"),
        ("graph", "k", 63, "graph.k"),
    ]
    for section, field, value, match in mutations:
        changed = deepcopy(config)
        changed[section][field] = value
        with pytest.raises(
            _RUNNER.MultiscaleHurdleRunnerError, match=match
        ):
            _RUNNER._validate_multiscale_contract(changed)
    direct = deepcopy(config)
    direct["dataset"]["patient_id"] = "prohibited"
    with pytest.raises(
        _RUNNER.MultiscaleHurdleRunnerError, match="alias-only"
    ):
        _RUNNER._validate_multiscale_contract(direct)


def test_per_gene_references_and_metric_row_include_percent_accuracy() -> None:
    counts = np.asarray(
        [[0, 1, 4], [1, 0, 8], [2, 3, 0], [0, 2, 16]],
        dtype=np.float32,
    )
    mean = np.log1p(counts).mean(axis=0)
    scale = np.log1p(counts).std(axis=0)
    references = _RUNNER._fit_per_gene_references(
        counts, expression_mean=mean, expression_scale=scale
    )
    target = torch.from_numpy(counts)
    mask = torch.ones_like(target, dtype=torch.bool)
    result = SimpleNamespace(
        target=target,
        target_mask=mask,
    )
    reference_metrics = _RUNNER._reference_metrics(
        references,
        result,
        expression_mean=mean,
        expression_scale=scale,
    )
    model_metrics = {
        key.removeprefix("reference_per_gene_"): value
        for key, value in reference_metrics.items()
    }
    entry = {
        "spec": {"mode": "node"},
        "replicate": 0,
        "entry_id": "whole-node-r0",
        "seed": 7,
        "mask_checksum": "a" * 64,
    }
    row = _RUNNER._replicate_metric_row(
        entry=entry,
        model_metrics=model_metrics,
        reference_metrics=reference_metrics,
    )
    assert row["detection_balanced_accuracy_percent"] == pytest.approx(
        100.0 * row["detection_balanced_accuracy"]
    )
    assert row[
        "positive_continuous_huber_relative_improvement_over_per_gene_reference"
    ] == pytest.approx(0.0)
    assert len(row["state8_recall"]) == 8


def test_disk_gate_uses_decimal_gb_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    usage = shutil_usage = SimpleNamespace(
        total=100_000_000_000,
        used=55_000_000_000,
        free=45_000_000_000,
    )
    monkeypatch.setattr(_RUNNER.shutil, "disk_usage", lambda _path: usage)
    assert _RUNNER._enforce_disk_safety(tmp_path)["used_decimal_gb"] == 55.0
    shutil_usage.used = 55_000_000_001
    with pytest.raises(
        _RUNNER.MultiscaleHurdleRunnerError, match="55.0 GB"
    ):
        _RUNNER._enforce_disk_safety(tmp_path)


def test_receipt_checksum_rejects_resigned_content_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    payload = {"receipt_kind": "example", "complete": True}
    payload["checksum"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    observed, checksum = _RUNNER._verified_json_receipt(
        path, kind="example"
    )
    assert observed["complete"] is True
    assert checksum == payload["checksum"]

    payload["complete"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        _RUNNER.MultiscaleHurdleRunnerError, match="checksum"
    ):
        _RUNNER._verified_json_receipt(path, kind="example")


def test_frozen_contract_verification_requires_amendment003(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    relative_paths = (
        _RUNNER._FROZEN_CONTRACT_RELATIVE,
        _RUNNER._FROZEN_CONTRACT_HASH_RELATIVE,
        _RUNNER._CONTRACT_AMENDMENT_RELATIVE,
        _RUNNER._CONTRACT_AMENDMENT_RELATIVE.with_suffix(".sha256"),
        _RUNNER._CONTRACT_SUPPLEMENT_RELATIVE,
        _RUNNER._CONTRACT_SUPPLEMENT_RELATIVE.with_suffix(".sha256"),
        _RUNNER.SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE,
        _RUNNER.SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE.with_suffix(
            ".sha256"
        ),
    )
    for relative in relative_paths:
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((_ROOT / relative).read_bytes())

    verified = _RUNNER._verify_frozen_contract(project_root)
    supplement = verified["amendment"]["required_supplement"]
    assert supplement == {
        "path": _RUNNER._CONTRACT_SUPPLEMENT_RELATIVE.as_posix(),
        "sha256": _RUNNER._CONTRACT_SUPPLEMENT_SHA256,
        "verified": True,
        "mask_noninterference_gate_required_before_gpu_training": True,
    }

    (project_root / _RUNNER._CONTRACT_SUPPLEMENT_RELATIVE).write_text(
        "campaign_id: drifted\n",
        encoding="utf-8",
    )
    with pytest.raises(
        _RUNNER.MultiscaleHurdleRunnerError,
        match="amendment003",
    ):
        _RUNNER._verify_frozen_contract(project_root)


def test_runner_rebuilds_exact_true_graph_and_source_permutation() -> None:
    n_nodes = 100
    coordinates = np.stack(
        (
            np.arange(n_nodes, dtype=np.float64) * 20.0,
            np.zeros(n_nodes, dtype=np.float64),
        ),
        axis=1,
    )
    core = SimpleNamespace(
        coordinates_um=coordinates,
        macroblock_ids=np.asarray(["fixture-block"] * n_nodes),
    )
    graph_config = _config()["graph"]
    graphs = _RUNNER.build_true_multiscale_graphs(
        coordinates,
        query_chunk_size=int(graph_config["query_chunk_size"]),
        receiver_chunk_size=int(graph_config["receiver_shard_size"]),
        mutual_search_chunk_size=int(
            graph_config["mutual_search_chunk_size"]
        ),
        workers=int(graph_config["construction_workers"]),
        epsilon=1e-8,
    )
    local_index, local_attributes = graphs.local.concatenate()
    permutation = (
        _RUNNER.build_macroblock_spatial_antipode_permutation(
            coordinates,
            core.macroblock_ids,
            local_edge_index=local_index,
            local_edge_attributes=local_attributes,
        )
    )
    expected = _RUNNER.true_graph_receipt(graphs)
    expected["local_source_permutation"] = dict(permutation.receipt)
    graph_config["local_source_permutation"] = dict(permutation.receipt)
    graph_config["expected_graph_receipt_sha256"] = canonical_sha256(
        expected
    )
    graph_config["expected_materialized_graph_sha256"] = (
        graphs.checksums.bundle_sha256
    )
    graph_config["expected_bundle_qc"] = expected["bundle_qc"]
    for scale in ("local", "regional"):
        scale_graph = getattr(graphs, scale)
        graph_config[scale].update(
            {
                "expected_graph_sha256": (
                    scale_graph.checksums.graph_sha256
                ),
                "expected_directed_edges": (
                    scale_graph.qc.n_directed_edges
                ),
                "expected_components": scale_graph.qc.n_components,
                "expected_isolated_nodes": (
                    scale_graph.qc.n_isolated_nodes
                ),
            }
        )

    rebuilt_graphs, rebuilt, rebuilt_permutation = (
        _RUNNER._build_verified_graphs(
            core,
            graph_config,
            expected,
        )
    )
    assert rebuilt == expected
    assert (
        rebuilt_permutation.receipt["checksum"]
        == permutation.receipt["checksum"]
    )
    assert rebuilt_graphs.local.checksums == graphs.local.checksums
    assert rebuilt_graphs.regional.checksums == graphs.regional.checksums

    tampered = deepcopy(expected)
    tampered["local_source_permutation"]["qc"][
        "effective_permuted_source_equals_receiver_"
        "affected_receiver_count"
    ] += 1
    with pytest.raises(
        _RUNNER.MultiscaleHurdleRunnerError,
        match="differs from materialization",
    ):
        _RUNNER._build_verified_graphs(
            core,
            graph_config,
            tampered,
        )


def test_real_model_audit_enforces_exact_budget_and_shape_receipt() -> None:
    config = _config()
    core = SimpleNamespace(
        n_genes=1000,
        expression_mean=np.zeros(1000, dtype=np.float32),
        expression_scale=np.ones(1000, dtype=np.float32),
        node_covariates=np.zeros((2, 22), dtype=np.float32),
    )
    expected_model = _RUNNER.MultiscaleAdditiveHybridModel(
        **_RUNNER._model_arguments(
            core,
            config["model"],
            regional_routing="surrogate",
            local_routing="surrogate",
        )
    )
    shape_sha = canonical_sha256(_RUNNER._named_shapes(expected_model))
    del expected_model
    model, construction, audit = _RUNNER._paired_models(
        core=core,
        model_config=config["model"],
        selected_arm="self",
        seed=0,
        expected_shape_sha256=shape_sha,
    )
    assert _RUNNER.trainable_parameter_count(model) == 7_559_184
    assert audit["all_arm_parameter_shapes_identical"] is True
    assert audit["all_arm_initial_states_bit_identical"] is True
    json.dumps(construction, allow_nan=False)

    with pytest.raises(
        _RUNNER.MultiscaleHurdleRunnerError,
        match="materialization",
    ):
        _RUNNER._paired_models(
            core=core,
            model_config=config["model"],
            selected_arm="self",
            seed=0,
            expected_shape_sha256="0" * 64,
        )
