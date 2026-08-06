"""Focused contracts for the shared-model pooled hybrid-count runner."""

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

from spatial_benchmark.hybrid_count_metrics import HybridCountReferences
from spatial_benchmark.identifiers import canonical_sha256
from spatial_benchmark.masking import MaskSpec, create_fixed_mask_bundle
from spatial_benchmark.training import TrainingConfig


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "scripts/train/run_pooled_hybrid_count_capacity.py"
_SPEC = importlib.util.spec_from_file_location(
    "test_pooled_hybrid_count_capacity_runner_module", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)


def _component(reference: str, section: str) -> dict[str, Any]:
    payload = yaml.safe_load((_ROOT / reference).read_text(encoding="utf-8"))
    return dict(payload[section])


def _prior_mask_record(alias: str) -> dict[str, Any]:
    return {
        "source_run_id": f"r_alias_safe_{alias.lower()}",
        "reference": f"artifacts/runs/alias-safe/{alias}/masks.json",
        "file_sha256": "1" * 64,
        "bundle_checksum": "2" * 64,
        "base_seed": 7,
        "entries": [
            {
                "entry_id": f"{mode}-{replicate}",
                "mode": mode,
                "replicate": replicate,
                "seed": replicate + 10,
                "mask_checksum": "3" * 64,
            }
            for mode in ("partial_gene", "whole_node", "spatial_block")
            for replicate in range(3)
        ],
    }


def _config(
    model_name: str = "hybrid-count-gat",
    *,
    pilot: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    uses_graph = model_name == "hybrid-count-gat"
    arm = (
        "pooled-hybrid-gat-k1000"
        if uses_graph
        else "pooled-hybrid-matched-self"
    )
    model = _component(
        (
            "configs/model/hybrid_count_gat.yaml"
            if uses_graph
            else "configs/model/hybrid_count_matched_self.yaml"
        ),
        "model",
    )
    trainer = _component(
        (
            "configs/trainer/pooled_hybrid_resource_pilot_2.yaml"
            if pilot
            else "configs/trainer/pooled_hybrid_fixed_200.yaml"
        ),
        "trainer",
    )
    trainer["amp_authorization"] = {
        "mode": (
            "same_weight_fp32_amp_each_core_diagnostic"
            if pilot
            else "require_external_pilot_gate_receipt"
        ),
        "receipt_schema": _RUNNER._PILOT_RECEIPT_SCHEMA,
        "receipt_reference": _RUNNER._PILOT_RECEIPT_RELATIVE.as_posix(),
        "frozen_contract_sha256": _RUNNER._FROZEN_CONTRACT_SHA256,
    }
    evaluation = _component(
        (
            "configs/evaluation/"
            "held_in_pooled_10core_hybrid_count_pilot_v1.yaml"
            if pilot
            else "configs/evaluation/"
            "held_in_pooled_10core_hybrid_count_v1.yaml"
        ),
        "evaluation",
    )
    evaluation["prior_mask_sources"] = {
        alias: _prior_mask_record(alias) for alias in _RUNNER.ANC_ALIASES
    }
    graph = _component(
        "configs/graph/pooled_k1000_r2000_mutual.yaml", "graph"
    )
    graph["expected_core_graphs"] = {
        alias: {
            "n_nodes": 8,
            "graph_sha256": f"{index + 1:x}".zfill(64),
            "n_directed_edges": 16,
        }
        for index, alias in enumerate(_RUNNER.ANC_ALIASES)
    }
    features = _component(
        "configs/features/cosmx_full_core_morphology_edge_geometry.yaml",
        "features",
    )
    features["use_edge_features"] = uses_graph
    features["node_expression"] = {
        "biological_targets": 1000,
        "source_scale": "raw_biological_probe_counts",
        "discrete_transform": "fixed_hybrid_count_states",
        "continuous_transform": "shared_equal_core_standardized_log1p",
        "masked_discrete_value": "input_only_mask_token_8",
        "masked_continuous_value": 0.0,
        "explicit_mask_authoritative_inside_model": True,
    }
    if not uses_graph:
        features["edge_features"] = []
    features["prohibited_node_inputs"] = [
        "direct_identifiers",
        "absolute_or_local_coordinates",
        "expression_derived_library_size",
        "rna_derived_qc",
        "vendor_cell_type_cluster_neighborhood_or_niche",
        "hidden_target_values",
        "core_alias",
    ]
    dataset = _component(
        "configs/dataset/adjacent_normal_10core_pooled_fit_v1.yaml",
        "dataset",
    )
    dataset["dataset_fingerprint"] = "a" * 64
    dataset["split_id"] = "b" * 16
    dataset["count_representation"] = {
        "schema": _RUNNER._REPRESENTATION_SCHEMA,
        "source_scale": "raw_biological_probe_counts",
        "num_output_states": 8,
        "mask_token_id": 8,
        "mask_token_is_output": False,
        "fixed_boundaries": True,
        "fit_required": False,
        "count_mapping": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4-7": 4,
            "8-15": 5,
            "16-31": 6,
            "32+": 7,
        },
        "continuous_channel": {
            "source": "raw_biological_probe_counts",
            "transform": "shared_equal_core_standardized_log1p",
            "preserves_exact_within_bin_value": True,
            "masked_value": 0.0,
        },
    }
    return {
        "campaign": {
            "campaign_id": _RUNNER._CAMPAIGN_ID,
            "exploratory": True,
            "frozen_contract_sha256": _RUNNER._FROZEN_CONTRACT_SHA256,
        },
        "experiment": {
            "arm": arm,
            "core_aliases": list(_RUNNER.ANC_ALIASES),
            "one_shared_model_state": True,
            "cross_core_edges": False,
            "resource_pilot": pilot,
        },
        "dataset": dataset,
        "features": features,
        "graph": graph,
        "model": model,
        "masking": {
            "type": "mixed_expression_masking",
            "curriculum": "P+N+B",
            "rate": {
                "partial_gene": 0.2,
                "whole_node": 0.1,
                "spatial_block": 0.1,
            },
            "rates": {
                "partial_gene": 0.2,
                "whole_node": 0.1,
                "spatial_block": 0.1,
            },
            "post_warmup_probabilities": {
                "partial_gene": 0.6,
                "whole_node": 0.3,
                "spatial_block": 0.1,
            },
            "warmup_epochs": 10,
            "block_shape": "disk",
            "block_width_um": None,
            "mask_seed": 314159,
            "mask_expression_only": True,
            "explicit_gene_mask_channel": True,
        },
        "trainer": trainer,
        "evaluation": evaluation,
        "metadata": {
            "locked_config_materialization_receipt": (
                _RUNNER._MATERIALIZATION_RECEIPT_RELATIVE.as_posix()
            )
        },
        "seed": seed,
        "fold": 0,
        "attempt": 1,
    }


@pytest.mark.parametrize(
    "model_name",
    ("hybrid-count-gat", "hybrid-count-matched-self"),
)
def test_contract_accepts_exact_two_pooled_pilot_arms(
    model_name: str,
) -> None:
    contract = _RUNNER._validate_pooled_contract(_config(model_name))
    assert contract.seed == 0
    assert contract.diagnostic_resource_pilot is True
    assert contract.uses_graph is (model_name == "hybrid-count-gat")
    assert contract.public_variant.startswith("pooled-hybrid-")


@pytest.mark.parametrize("seed", range(7))
def test_contract_accepts_all_seven_production_seeds(seed: int) -> None:
    contract = _RUNNER._validate_pooled_contract(
        _config(pilot=False, seed=seed)
    )
    assert contract.seed == seed
    assert contract.diagnostic_resource_pilot is False


def test_contract_rejects_single_core_cross_core_and_identifier_drift() -> None:
    config = _config()

    missing = deepcopy(config)
    missing["dataset"]["core_aliases"].pop()
    with pytest.raises(
        _RUNNER.PooledHybridCountRunnerError, match="core_aliases"
    ):
        _RUNNER._validate_pooled_contract(missing)

    cross_core = deepcopy(config)
    cross_core["graph"]["cross_core_edges"] = True
    with pytest.raises(
        _RUNNER.PooledHybridCountRunnerError, match="cross_core_edges"
    ):
        _RUNNER._validate_pooled_contract(cross_core)

    identifier = deepcopy(config)
    identifier["dataset"]["patient_id"] = "prohibited"
    with pytest.raises(
        _RUNNER.PooledHybridCountRunnerError, match="alias-only"
    ):
        _RUNNER._validate_pooled_contract(identifier)

    wrong_seed = _config(pilot=False, seed=7)
    with pytest.raises(
        _RUNNER.PooledHybridCountRunnerError,
        match="production requires seed 0 through 6",
    ):
        _RUNNER._validate_pooled_contract(wrong_seed)


def _mask_training_config() -> TrainingConfig:
    return TrainingConfig(
        max_epochs=2,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip_norm=1.0,
        huber_delta=1.0,
        patience=0,
        min_delta=0.0,
        curriculum="P+N+B",
        warmup_epochs=10,
        partial_gene_rate=0.25,
        node_rate=0.25,
        block_node_rate=0.25,
        block_width_um=None,
        block_shape="disk",
        mask_seed=314159,
        model_seed=0,
        edge_dropout=0.0,
        amp=False,
        deterministic=True,
        deterministic_warn_only=False,
        device="cpu",
        restore_best=False,
    )


def test_prior_fixed_masks_reconstruct_from_frozen_base_seed() -> None:
    config = _config()
    training = _mask_training_config()
    coordinates = np.stack(
        [np.arange(8, dtype=np.float64), np.zeros(8)], axis=1
    )
    core = SimpleNamespace(
        alias="ANC-01",
        coordinates_um=coordinates,
        n_genes=8,
    )
    specs = [
        MaskSpec(
            mode=mode,
            partial_gene_rate=training.partial_gene_rate,
            node_rate=training.node_rate,
            block_node_rate=training.block_node_rate,
            block_width_um=training.block_width_um,
            block_shape=training.block_shape,
            label=label,
        )
        for mode, label in (
            ("partial", "partial_gene"),
            ("node", "whole_node"),
            ("block", "spatial_block"),
        )
    ]
    expected = create_fixed_mask_bundle(
        {"fit": coordinates},
        8,
        specs,
        replicates=3,
        base_seed=123,
    )
    record = config["evaluation"]["prior_mask_sources"]["ANC-01"]
    record["base_seed"] = 123
    record["bundle_checksum"] = expected.checksum
    record["entries"] = [
        {
            "entry_id": entry["entry_id"],
            "mode": entry["spec"]["label"],
            "replicate": entry["replicate"],
            "seed": entry["seed"],
            "mask_checksum": entry["mask_checksum"],
        }
        for entry in expected.manifest["entries"]
    ]
    observed = _RUNNER._mask_bundle_for_core(
        config=config, core=core, training=training
    )
    assert observed.checksum == expected.checksum

    config["evaluation"]["prior_mask_sources"]["ANC-01"]["entries"][0][
        "mask_checksum"
    ] = "f" * 64
    with pytest.raises(
        _RUNNER.PooledHybridCountRunnerError,
        match="entry identities changed",
    ):
        _RUNNER._mask_bundle_for_core(
            config=config, core=core, training=training
        )


def test_metric_aggregation_is_replicate_then_equal_core_not_cell_weighted() -> None:
    rows: list[dict[str, Any]] = []
    for alias_index, alias in enumerate(_RUNNER.ANC_ALIASES):
        for mode in _RUNNER._REQUIRED_PUBLIC_MASKS:
            for replicate in range(3):
                rows.append(
                    {
                        "biological_unit_alias": alias,
                        "split": "fit",
                        "mask_mode": mode,
                        "mask_replicate": replicate,
                        "mask_entry_id": f"{alias}-{mode}-{replicate}",
                        "mask_seed": replicate,
                        "mask_checksum": "a" * 64,
                        "hybrid_loss": float(alias_index),
                    }
                )
    metrics = _RUNNER._aggregate_metrics(rows, replicates_per_mode=3)
    assert metrics["fit/ANC-01/whole_node/hybrid_loss"] == 0.0
    assert metrics["fit/ANC-10/whole_node/hybrid_loss"] == 9.0
    assert metrics["fit/whole_node/hybrid_loss"] == pytest.approx(4.5)


def test_equal_core_reference_metrics_are_finite_and_fail_closed() -> None:
    counts = np.asarray([[0, 1, 2, 3, 4, 8, 16, 32]], dtype=np.float32)
    mask = np.ones_like(counts, dtype=bool)
    references = HybridCountReferences(
        detection_probability=np.full(8, 0.5),
        positive_ordinal_probability=np.full((8, 6), 0.5),
        positive_continuous_standardized=np.zeros(8),
        detected_state=np.ones(8, dtype=bool),
        positive_state=np.full(8, 4, dtype=np.int64),
        count_state=np.full(8, 4, dtype=np.int64),
        audit={"reference_sha256": "a" * 64},
    )
    result = SimpleNamespace(
        target=torch.from_numpy(counts),
        target_mask=torch.from_numpy(mask),
    )
    metrics = _RUNNER._reference_metrics(
        result,
        references,
        expression_mean=np.zeros(8),
        expression_scale=np.ones(8),
        prefix="reference_equal_core",
    )
    assert metrics["reference_equal_core_detection_bce"] == pytest.approx(
        np.log(2)
    )
    assert metrics["reference_equal_core_ordinal_bce"] == pytest.approx(
        np.log(2)
    )
    assert all(np.isfinite(value) for value in metrics.values())

    no_zero = SimpleNamespace(
        target=torch.from_numpy(counts[:, 1:]),
        target_mask=torch.ones((1, 7), dtype=torch.bool),
    )
    with pytest.raises(
        _RUNNER.PooledHybridCountRunnerError, match="missing a stratum"
    ):
        _RUNNER._reference_metrics(
            no_zero,
            HybridCountReferences(
                detection_probability=references.detection_probability[1:],
                positive_ordinal_probability=(
                    references.positive_ordinal_probability[1:]
                ),
                positive_continuous_standardized=(
                    references.positive_continuous_standardized[1:]
                ),
                detected_state=references.detected_state[1:],
                positive_state=references.positive_state[1:],
                count_state=references.count_state[1:],
                audit=references.audit,
            ),
            expression_mean=np.zeros(7),
            expression_scale=np.ones(7),
            prefix="reference_equal_core",
        )


def _write_signed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    signed = deepcopy(payload)
    signed["checksum"] = canonical_sha256(signed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(signed), encoding="utf-8")
    return signed


def test_production_pilot_receipt_matches_queue_contract(tmp_path: Path) -> None:
    config = _config(pilot=False)
    contract = _RUNNER._validate_pooled_contract(config)
    with pytest.raises(
        _RUNNER.PooledHybridCountRunnerError, match="missing"
    ):
        _RUNNER._validate_production_pilot_receipt(
            tmp_path,
            config,
            contract,
            materialization_checksum="a" * 64,
        )
    gate = {
        "schema_version": 1,
        "receipt_kind": _RUNNER._PILOT_RECEIPT_SCHEMA,
        "campaign_id": _RUNNER._CAMPAIGN_ID,
        "materialization_checksum": "a" * 64,
        "frozen_contract_sha256": _RUNNER._FROZEN_CONTRACT_SHA256,
        "thresholds": dict(_RUNNER._PILOT_GATE_THRESHOLDS),
        "failure_reasons": [],
        "gate_passed": True,
        "production_authorized": True,
        "same_frozen_precision_batches_all_cores": True,
        "same_evaluation_masks": True,
        "same_verified_graph_bundle": True,
        "paired_initialization_digests_match": True,
        "jobs": [
            {
                "arm": arm,
                "seed": 0,
                "parameter_count": _RUNNER._EXPECTED_PARAMETER_COUNT,
                "checkpoint_epoch": 1,
                "checkpoint_role": "last",
                **{
                    field: True
                    for field in _RUNNER._PILOT_JOB_REQUIRED_TRUE_FIELDS
                },
            }
            for arm in _RUNNER._PUBLIC_VARIANTS.values()
        ],
    }
    signed = _write_signed(
        tmp_path / _RUNNER._PILOT_RECEIPT_RELATIVE, gate
    )
    record = _RUNNER._validate_production_pilot_receipt(
        tmp_path,
        config,
        contract,
        materialization_checksum="a" * 64,
    )
    assert record is not None
    assert record["checksum"] == signed["checksum"]

    gate["jobs"][0]["precision_equivalence_passed"] = False
    _write_signed(tmp_path / _RUNNER._PILOT_RECEIPT_RELATIVE, gate)
    with pytest.raises(
        _RUNNER.PooledHybridCountRunnerError, match="pilot receipt arm"
    ):
        _RUNNER._validate_production_pilot_receipt(
            tmp_path,
            config,
            contract,
            materialization_checksum="a" * 64,
        )


def test_materialization_binding_normalizes_only_retry_attempt(
    tmp_path: Path,
) -> None:
    root_config = _config()
    retry_config = deepcopy(root_config)
    retry_config["attempt"] = 2
    receipt = {
        "schema_version": 1,
        "receipt_kind": "pooled_hybrid_count_locked_config_materialization_v1",
        "campaign_id": _RUNNER._CAMPAIGN_ID,
        "frozen_contract": {
            "sha256": _RUNNER._FROZEN_CONTRACT_SHA256,
        },
        "pilot_jobs": [
            {
                "arm": "pooled-hybrid-gat-k1000",
                "seed": 0,
                "config_sha256": canonical_sha256(root_config),
            }
        ],
        "production_jobs": [],
    }
    _write_signed(
        tmp_path / _RUNNER._MATERIALIZATION_RECEIPT_RELATIVE, receipt
    )
    record = _RUNNER._validate_materialization_receipt(
        tmp_path,
        retry_config,
        _RUNNER._validate_pooled_contract(retry_config),
    )
    assert record["config_sha256"] == canonical_sha256(root_config)
    assert record["runtime_config_sha256"] == canonical_sha256(retry_config)
    assert record["run_attempt"] == 2


def test_initial_state_digest_is_deterministic_and_content_sensitive() -> None:
    first = {"weight": torch.tensor([[1.0, 2.0]])}
    second = {"weight": torch.tensor([[1.0, 2.0]])}
    changed = {"weight": torch.tensor([[1.0, 3.0]])}
    assert _RUNNER._state_dict_sha256(first) == _RUNNER._state_dict_sha256(
        second
    )
    assert _RUNNER._state_dict_sha256(first) != _RUNNER._state_dict_sha256(
        changed
    )


def test_summary_peak_vram_fields_keep_registry_compatibility() -> None:
    fields = _RUNNER._summary_peak_vram_fields(3 * 1024**3)
    assert fields == {"peak_vram_gib": 3.0, "peak_vram_gb": 3.0}
