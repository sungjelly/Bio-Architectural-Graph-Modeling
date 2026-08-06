"""Focused contract tests for multiscale hurdle campaign materialization."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Mapping

import numpy as np
import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "materialize_multiscale_hurdle_campaign.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "materialize_multiscale_hurdle_campaign_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MaterializationError = _MODULE.MultiscaleHurdleMaterializationError
materialize_campaign = _MODULE.materialize_campaign


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _component(reference: str) -> dict[str, Any]:
    return deepcopy(
        yaml.safe_load((_ROOT / reference).read_text(encoding="utf-8"))
    )


def _source_config(alias: str) -> dict[str, Any]:
    index = int(alias[-2:])
    fingerprint = _sha(f"{alias}-preprocessing")
    split = _sha(f"{alias}-split")
    return {
        "campaign": {"campaign_id": _MODULE.SOURCE_CAMPAIGN_ID},
        "experiment": {"biological_unit_alias": alias},
        "dataset": {
            "dataset_id": f"cosmx_{alias.lower().replace('-', '')}_test",
            "version": "adjacent_normal_full_core_fit_v1",
            "split_id": split[:16],
            "dataset_fingerprint": fingerprint,
            "split_fingerprint": split,
            "prepared_artifact_reference": (
                f"prepared/{alias.lower()}/prepared_v1"
            ),
            "biological_target_count": 1000,
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "preprocessing_fit_scope": "all_nodes_transductive",
            "validation_or_test_partition_present": False,
            "patient_generalization_supported": False,
            "synthetic_node_count": 7_000 + index,
        },
        "features": {
            "use_edge_features": True,
            "fit_scope": "all_nodes_transductive",
            "node_expression": {
                "biological_targets": 1000,
                "source_scale": "raw_biological_probe_counts",
                "explicit_mask_authoritative_inside_model": True,
            },
            "node_metadata": {
                "transformed_with": "synthetic_verified_transform",
                "fields": list(_MODULE.ALLOWED_METADATA_COLUMNS),
            },
            "edge_features": {
                "fit_scope": "all_retained_directed_edges_transductive",
                "standardization": "full_core_edge_wise",
                "fields": list(_MODULE.EDGE_ATTRIBUTE_NAMES),
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
    }


def _core(alias: str) -> SimpleNamespace:
    index = int(alias[-2:])
    n_nodes = 100
    x = np.arange(n_nodes, dtype=np.float64) * 20.0
    return SimpleNamespace(
        coordinates_um=np.stack(
            (x, np.full(n_nodes, float(index) * 10_000.0)),
            axis=1,
        ),
        macroblock_ids=np.asarray([f"block-{index}"] * n_nodes),
        n_nodes=n_nodes,
        n_genes=1000,
        expression_mean=np.zeros(1000, dtype=np.float32),
        expression_scale=np.ones(1000, dtype=np.float32),
        metadata_names=tuple(_MODULE.ALLOWED_METADATA_COLUMNS),
        checksums=SimpleNamespace(
            preprocessing_sha256=_sha(f"{alias}-preprocessing"),
            source_prepared_data_sha256=_sha(f"{alias}-prepared"),
        ),
    )


def _graph_receipt(index: int) -> dict[str, Any]:
    scales: dict[str, Any] = {}
    for offset, scale in enumerate(("local", "regional")):
        scales[scale] = {
            "qc": {
                "n_directed_edges": 1_000 + 10 * index + offset,
                "n_components": 2 + offset,
                "n_isolated_nodes": offset,
            },
            "checksums": {
                "graph_sha256": _sha(f"{index}-{scale}-graph"),
            },
        }
    return {
        **scales,
        "bundle_checksums": {
            "bundle_sha256": _sha(f"{index}-graph-bundle"),
        },
        "bundle_qc": {
            "local_regional_disjoint": True,
            "all_graphs_symmetric": True,
            "all_graphs_receiver_sorted": True,
            "all_graphs_loop_and_duplicate_free": True,
        },
    }


@pytest.fixture
def materializer_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    project_root = tmp_path / "project"
    contract_path = (
        project_root
        / "experiments"
        / "campaigns"
        / _MODULE.CAMPAIGN_ID
        / "frozen_task_contract.yaml"
    )
    contract_path.parent.mkdir(parents=True)
    contract_path.write_bytes(
        (
            _ROOT
            / "experiments"
            / "campaigns"
            / _MODULE.CAMPAIGN_ID
            / "frozen_task_contract.yaml"
        ).read_bytes()
    )
    amendment_path = (
        project_root
        / _MODULE.CONTRACT_AMENDMENT_RELATIVE
    )
    amendment_path.parent.mkdir(parents=True, exist_ok=True)
    amendment_path.write_bytes(
        (
            _ROOT
            / _MODULE.CONTRACT_AMENDMENT_RELATIVE
        ).read_bytes()
    )
    for relative in (
        _MODULE.SUPERSEDED_CONTRACT_AMENDMENT_RELATIVE,
        _MODULE.REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE,
    ):
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((_ROOT / relative).read_bytes())
    source_receipt_path = project_root / "source_materialization.json"
    source_receipt_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    output_dir = (
        project_root
        / "scratch"
        / "locked_campaigns"
        / _MODULE.CAMPAIGN_ID
    )

    source_configs = {
        alias: _source_config(alias) for alias in _MODULE.STAGE2_ALIASES
    }
    source_jobs = {
        (alias, "hybrid-gat-k1000"): {
            "alias": alias,
            "arm": "hybrid-gat-k1000",
        }
        for alias in _MODULE.STAGE2_ALIASES
    }
    source_receipt = {
        "campaign_id": _MODULE.SOURCE_CAMPAIGN_ID,
        "checksum": _sha("source-receipt-canonical"),
    }
    monkeypatch.setattr(
        _MODULE,
        "_load_source_receipt",
        lambda *_args, **_kwargs: (
            deepcopy(source_receipt),
            deepcopy(source_jobs),
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_source_identity",
        lambda *, alias, **_kwargs: (
            deepcopy(source_configs[alias]),
            _core(alias),
        ),
    )

    def fake_component(
        _project_root: Path,
        reference: str,
    ) -> tuple[dict[str, Any], str]:
        path = _ROOT / reference
        return _component(reference), _MODULE.sha256_file(path)

    monkeypatch.setattr(_MODULE, "_component", fake_component)
    def fake_graphs(coordinates: np.ndarray, **_kwargs: Any) -> Any:
        n_nodes = int(np.asarray(coordinates).shape[0])
        source = np.arange(n_nodes, dtype=np.int64)
        receiver = (source + 1) % n_nodes
        edge_index = np.stack((source, receiver), axis=0)
        edge_attributes = np.zeros(
            (n_nodes, len(_MODULE.EDGE_ATTRIBUTE_NAMES)),
            dtype=np.float32,
        )
        return SimpleNamespace(
            index=int(round(float(np.asarray(coordinates)[0, 1]) / 10_000)),
            local=SimpleNamespace(
                concatenate=lambda: (edge_index, edge_attributes)
            ),
        )

    monkeypatch.setattr(
        _MODULE,
        "build_true_multiscale_graphs",
        fake_graphs,
    )
    monkeypatch.setattr(
        _MODULE,
        "true_graph_receipt",
        lambda graphs: _graph_receipt(int(graphs.index)),
    )

    expected_parameter_audit = {
        "trainable_parameter_count": 7_559_184,
        "named_parameter_shapes_sha256": _sha("parameter-shapes"),
        "matched_arms": list(_MODULE.ARMS),
    }
    monkeypatch.setattr(
        _MODULE,
        "_parameter_audit",
        lambda **_kwargs: deepcopy(expected_parameter_audit),
    )
    return {
        "project_root": project_root,
        "source_receipt_path": source_receipt_path,
        "contract_path": contract_path,
        "output_dir": output_dir,
        "source_configs": source_configs,
        "expected_parameter_audit": expected_parameter_audit,
    }


def _call(paths: Mapping[str, Any]) -> dict[str, Any]:
    return materialize_campaign(
        project_root=paths["project_root"],
        source_receipt_path=paths["source_receipt_path"],
        contract_path=paths["contract_path"],
        output_dir=paths["output_dir"],
        workers=3,
    )


def _load_configs(directory: Path) -> list[dict[str, Any]]:
    return [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.yaml"))
    ]


def test_materializer_publishes_exact_locked_matrix_idempotently(
    materializer_fixture: dict[str, Any],
) -> None:
    first = _call(materializer_fixture)
    output = materializer_fixture["output_dir"]
    original_bytes = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    second = _call(materializer_fixture)

    assert second == first
    assert {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    } == original_bytes
    assert len(original_bytes) == 23

    receipt = json.loads(
        (output / _MODULE.RECEIPT_NAME).read_text(encoding="utf-8")
    )
    checksum = receipt.pop("checksum")
    assert checksum == canonical_sha256(receipt)
    assert receipt["counts"] == {
        "cores": 5,
        "pilot_configs": 2,
        "science_configs": 20,
    }
    assert receipt["parameter_audit"] == materializer_fixture[
        "expected_parameter_audit"
    ]
    assert receipt["frozen_contract"]["sha256"] == (
        _MODULE.FROZEN_CONTRACT_SHA256
    )
    assert receipt["allowed_gpu_ids"] == [0, 1, 2, 3, 5, 6, 7]
    assert 4 not in {
        job["requested_gpu"]
        for job in receipt["pilot_jobs"] + receipt["science_jobs"]
    }

    pilots = _load_configs(output / "resource_pilot_configs")
    science = _load_configs(output / "science_configs")
    assert {
        (
            config["experiment"]["biological_unit_alias"],
            config["experiment"]["arm"],
        )
        for config in pilots
    } == {("ANC-03", "self"), ("ANC-05", "self")}
    assert {
        (
            config["experiment"]["biological_unit_alias"],
            config["experiment"]["arm"],
        )
        for config in science
    } == {
        (alias, arm)
        for alias in _MODULE.STAGE2_ALIASES
        for arm in _MODULE.ARMS
    }
    expected_routing = {
        "self": ("surrogate", "surrogate"),
        "self-regional": ("true", "surrogate"),
        "self-regional-local": ("true", "true"),
        "self-regional-local-permuted": ("true", "permuted"),
    }
    for config in pilots + science:
        arm = config["experiment"]["arm"]
        assert (
            config["model"]["regional_routing"],
            config["model"]["local_routing"],
        ) == expected_routing[arm]
        assert config["model"]["output_channels_per_gene"] == 2
        assert config["model"]["parameter_match_group"] == (
            "multiscale_hurdle_v1"
        )
        assert config["model"]["exact_additive_decomposition"] is True
        assert config["model"]["regional_output_zero_initialized"] is True
        assert config["model"]["local_output_zero_initialized"] is True
        assert config["launcher"]["requested_gpu"] != "4"
        assert config["launcher"]["disk_safety_max_used_decimal_gb"] == 55.0
        assert config["campaign"]["frozen_contract_sha256"] == (
            _MODULE.FROZEN_CONTRACT_SHA256
        )
        assert config["campaign"]["contract_supplement_sha256"] == (
            _MODULE.REQUIRED_CONTRACT_SUPPLEMENT_SHA256
        )
        assert (
            config["metadata"][
                "mask_noninterference_supplement_enforced"
            ]
            is True
        )
        assert "rewired_local" not in config["graph"]
        assert config["graph"]["expected_graph_receipt_sha256"] == (
            canonical_sha256(
                next(
                    core["graph_receipt"]
                    for core in receipt["cores"]
                    if core["alias"]
                    == config["experiment"]["biological_unit_alias"]
                )
            )
        )
        serialized = yaml.safe_dump(config).lower()
        for prohibited in _MODULE._PROHIBITED_IDENTIFIER_KEYS:
            assert f"{prohibited}:" not in serialized

    for record in receipt["pilot_jobs"] + receipt["science_jobs"]:
        config_path = materializer_fixture["project_root"] / record["config"]
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert record["config_sha256"] == canonical_sha256(config)
        assert record["file_sha256"] == _MODULE.sha256_file(config_path)
        assert (
            record["local_source_permutation_sha256"]
            == config["graph"]["local_source_permutation"]["checksum"]
        )

    assert receipt["contract_amendment"]["required_supplement"] == {
        "reference": (
            _MODULE.REQUIRED_CONTRACT_SUPPLEMENT_RELATIVE.as_posix()
        ),
        "sha256": _MODULE.REQUIRED_CONTRACT_SUPPLEMENT_SHA256,
        "mask_noninterference_gate_required_before_gpu_training": True,
    }


def test_real_parameter_audit_matches_all_routing_arms_and_rejects_drift() -> None:
    components = {
        arm: _component(reference)
        for arm, reference in _MODULE._MODEL_COMPONENTS.items()
    }
    audit = _MODULE._parameter_audit(
        expression_mean=np.zeros(1000, dtype=np.float32),
        expression_scale=np.ones(1000, dtype=np.float32),
        node_covariate_dim=22,
        model_components=components,
    )
    assert audit["trainable_parameter_count"] == 7_559_184
    assert audit["matched_arms"] == list(_MODULE.ARMS)
    assert len(audit["named_parameter_shapes_sha256"]) == 64

    drifted = deepcopy(components)
    drifted["self-regional"]["message_dim"] += 1
    with pytest.raises(
        MaterializationError,
        match="exact parameter-shape match",
    ):
        _MODULE._parameter_audit(
            expression_mean=np.zeros(1000, dtype=np.float32),
            expression_scale=np.ones(1000, dtype=np.float32),
            node_covariate_dim=22,
            model_components=drifted,
        )


def test_frozen_contract_hash_and_authoritative_path_fail_closed(
    materializer_fixture: dict[str, Any],
) -> None:
    contract = materializer_fixture["contract_path"]
    assert _MODULE._verify_frozen_contract(
        materializer_fixture["project_root"],
        contract,
    ) == _MODULE.FROZEN_CONTRACT_SHA256

    other = materializer_fixture["project_root"] / "copied_contract.yaml"
    other.write_bytes(contract.read_bytes())
    with pytest.raises(MaterializationError, match="authoritative"):
        _MODULE._verify_frozen_contract(
            materializer_fixture["project_root"],
            other,
        )

    contract.write_text("campaign_id: drifted\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="checksum drifted"):
        _MODULE._verify_frozen_contract(
            materializer_fixture["project_root"],
            contract,
        )


def test_materializer_rejects_protected_identifier_before_publication(
    materializer_fixture: dict[str, Any],
) -> None:
    materializer_fixture["source_configs"]["ANC-03"]["dataset"][
        "patient_id"
    ] = "restricted-value"

    with pytest.raises(MaterializationError, match="prohibited identifier"):
        _call(materializer_fixture)
    assert not materializer_fixture["output_dir"].exists()


def test_materializer_rejects_unsafe_gpu_assignment_before_publication(
    materializer_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsafe(
        jobs: list[Mapping[str, Any]],
        *,
        node_counts: Mapping[str, int],
    ) -> dict[tuple[str, str], int]:
        del node_counts
        return {
            (str(job["alias"]), str(job["arm"])): 4
            for job in jobs
        }

    monkeypatch.setattr(_MODULE, "_gpu_assignments", unsafe)
    with pytest.raises(MaterializationError, match="GPU"):
        _call(materializer_fixture)
    assert not materializer_fixture["output_dir"].exists()


def test_atomic_publication_rejects_drift_without_overwrite(
    materializer_fixture: dict[str, Any],
) -> None:
    _call(materializer_fixture)
    output = materializer_fixture["output_dir"]
    target = (
        output
        / "science_configs"
        / "anc-09_self_regional_local_permuted_science.yaml"
    )
    target.write_text("tampered: true\n", encoding="utf-8")

    with pytest.raises(MaterializationError, match="Existing locked output"):
        _call(materializer_fixture)
    assert target.read_text(encoding="utf-8") == "tampered: true\n"
