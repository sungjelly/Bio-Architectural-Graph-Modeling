from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "materialize_hybrid_count_campaign.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "materialize_hybrid_count_campaign_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MaterializationError = _MODULE.HybridCountMaterializationError
materialize_campaign = _MODULE.materialize_campaign


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _component(reference: str, section: str) -> dict[str, Any]:
    payload = yaml.safe_load((_ROOT / reference).read_text(encoding="utf-8"))
    return deepcopy(payload[section])


def _identity(alias: str) -> dict[str, Any]:
    alias_index = int(alias[-2:])
    n_nodes = 6_000 + alias_index
    preprocessing = _sha(f"{alias}-preprocessing")
    split = _sha(f"{alias}-split")
    graph = _sha(f"{alias}-k1000-graph")
    return {
        "alias": alias,
        "n_nodes": n_nodes,
        "n_genes": 1_000,
        "prepared_artifact": f"prepared/{alias.lower()}/prepared_v1",
        "source_artifact_id": _sha(f"{alias}-artifact")[:16],
        "source_prepared_data_sha256": _sha(f"{alias}-prepared"),
        "preprocessing_sha256": preprocessing,
        "split_id": split[:16],
        "split_fingerprint": split,
        "split_fingerprint_basis": {
            "schema": "full_core_no_holdout_roles_v1",
            "dataset_fingerprint": preprocessing,
            "n_nodes": n_nodes,
            "role_assignment": "all rows assigned fit",
            "role_counts": {"fit": n_nodes, "validation": 0, "test": 0},
            "experimental_unit": "single_adjacent_normal_spatial_core",
        },
        "dataset_id": f"cosmx_{alias.lower().replace('-', '')}_test",
        "dataset_version": "adjacent_normal_full_core_fit_v1",
        "preprocessing_version": "adjacent_normal_full_core_fit_v1",
        "experimental_unit": "single_adjacent_normal_spatial_core",
        "source_recipe_schema": "full_core_fit_v1",
        "source_metadata_transform": "synthetic_verified_transform",
        "graph": {
            "kind": "exact_spatial_knn_radius_guard",
            "neighbor_k": 1_000,
            "k": 1_000,
            "radius_um": 2_000.0,
            "radius_guard_um": 2_000.0,
            "symmetry": "mutual",
            "edge_dropout": 0.0,
            "self_loops": False,
            "expected_materialized_graph_sha256": graph,
            "expected_directed_edges": n_nodes * 800,
        },
        "graph_sha256": graph,
        "n_directed_edges": n_nodes * 800,
        "node_metadata": {
            "transformed_with": "synthetic_verified_transform",
            "fields": list(_MODULE.ALLOWED_METADATA_COLUMNS),
        },
        "edge_features": {
            "fit_scope": "all_retained_directed_edges_transductive",
            "standardization": "full_core_edge_wise",
            "fields": list(_MODULE.EDGE_ATTRIBUTE_NAMES),
        },
    }


@pytest.fixture
def materializer_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    source_path = project_root / "source_materialization.json"
    source_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    contract_path = project_root / "frozen_task_contract.yaml"
    contract_path.write_text("synthetic: true\n", encoding="utf-8")
    output_dir = project_root / "scratch" / "locked" / _MODULE.CAMPAIGN_ID

    source_payload = {
        "campaign_id": _MODULE.SOURCE_CAMPAIGN_ID,
        "materialized_cores": [
            {"alias": alias} for alias in _MODULE.ALIASES
        ],
        "jobs": [],
    }
    identities = {alias: _identity(alias) for alias in _MODULE.ALIASES}
    monkeypatch.setattr(
        _MODULE,
        "_validate_frozen_contract",
        lambda _path: ({"campaign_id": _MODULE.CAMPAIGN_ID}, "a" * 64),
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_source_materialization",
        lambda _path: (deepcopy(source_payload), "b" * 64),
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_source_core",
        lambda *, alias, **_kwargs: deepcopy(identities[alias]),
    )

    def fake_component(
        _project_root: Path,
        reference: str,
        section: str,
    ) -> tuple[dict[str, Any], str]:
        return _component(reference, section), _sha(reference)

    monkeypatch.setattr(_MODULE, "_component_section", fake_component)
    monkeypatch.setattr(
        _MODULE,
        "_validate_model_components",
        lambda _models: 11_674_880,
    )
    return {
        "project_root": project_root,
        "source_path": source_path,
        "contract_path": contract_path,
        "output_dir": output_dir,
    }


def _call(paths: Mapping[str, Path]) -> dict[str, Any]:
    return materialize_campaign(
        project_root=paths["project_root"],
        source_materialization=paths["source_path"],
        contract_path=paths["contract_path"],
        output_dir=paths["output_dir"],
    )


def test_materializer_publishes_locked_alias_safe_configs_idempotently(
    materializer_fixture: dict[str, Path],
) -> None:
    first = _call(materializer_fixture)
    output = materializer_fixture["output_dir"]
    first_bytes = {
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
    } == first_bytes
    receipt = json.loads(
        (output / _MODULE.RECEIPT_NAME).read_text(encoding="utf-8")
    )
    checksum = receipt.pop("checksum")
    assert checksum == canonical_sha256(receipt)
    assert receipt["counts"] == {
        "aliases": 10,
        "pilot_configs": 2,
        "production_configs": 20,
    }
    assert receipt["parameter_count"] == 11_674_880
    assert receipt["allowed_gpu_ids"] == [0, 1, 2, 3, 5, 6, 7]
    assert sum(receipt["assignment"]["production_job_counts"].values()) == 20
    assert 4 not in {job["requested_gpu"] for job in receipt["production_jobs"]}
    assert len(first_bytes) == 23

    configs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((output / "production_configs").glob("*.yaml"))
    ]
    assert {
        (config["dataset"]["biological_unit_alias"], config["experiment"]["arm"])
        for config in configs
    } == {
        (alias, arm) for alias in _MODULE.ALIASES for arm in _MODULE.ARMS
    }
    for config in configs:
        assert config["campaign"]["exploratory"] is True
        assert config["seed"] == 0
        assert config["trainer"]["optimizer"] == "AdamW"
        assert config["trainer"]["max_epochs"] == 200
        assert config["trainer"]["early_stopping"] is False
        assert config["trainer"]["neighbor_sampling"] is False
        assert config["masking"]["fit_replicates"] == 3
        assert config["dataset"]["count_representation"] == (
            _MODULE._COUNT_REPRESENTATION
        )
        assert len(config["features"]["node_metadata"]["fields"]) == 22
        if config["experiment"]["arm"] == "hybrid-gat-k1000":
            assert config["features"]["use_edge_features"] is True
            assert len(config["features"]["edge_features"]["fields"]) == 17
        else:
            assert config["features"]["use_edge_features"] is False
            assert config["features"]["edge_features"] == []
        serialized = yaml.safe_dump(config)
        assert "SO_" not in serialized
        assert "donor_id" not in serialized.lower()
        assert "patient_id" not in serialized.lower()


def test_materializer_rejects_existing_drift_without_overwrite(
    materializer_fixture: dict[str, Path],
) -> None:
    _call(materializer_fixture)
    path = (
        materializer_fixture["output_dir"]
        / "production_configs"
        / "anc-10_hybrid_matched_self.yaml"
    )
    path.write_text("tampered: true\n", encoding="utf-8")

    with pytest.raises(MaterializationError, match="different content"):
        _call(materializer_fixture)

    assert path.read_text(encoding="utf-8") == "tampered: true\n"


def test_materializer_rejects_unsafe_assignment_before_publication(
    materializer_fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unsafe_assignment(
        jobs: list[Mapping[str, Any]],
    ) -> tuple[dict[tuple[str, str], int], dict[str, float], dict[str, int]]:
        return (
            {
                (str(job["alias"]), str(job["arm"])): 4
                for job in jobs
            },
            {"4": 0.0},
            {"4": len(jobs)},
        )

    monkeypatch.setattr(_MODULE, "_lpt_assign", unsafe_assignment)

    with pytest.raises(MaterializationError, match="prohibited GPU"):
        _call(materializer_fixture)

    assert not materializer_fixture["output_dir"].exists()
