from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Mapping

import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT / "scripts" / "train" / "materialize_pooled_hybrid_campaign.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "materialize_pooled_hybrid_campaign_for_tests", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MaterializationError = _MODULE.PooledHybridMaterializationError
materialize_campaign = _MODULE.materialize_campaign


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _component(reference: str, section: str) -> dict[str, Any]:
    payload = yaml.safe_load((_ROOT / reference).read_text(encoding="utf-8"))
    return deepcopy(payload[section])


def _graph_sources() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for receipt in _MODULE.EXPECTED_PREPARED_CORES:
        result[receipt.alias] = {
            "n_nodes": receipt.n_nodes,
            "n_genes": 1_000,
            "graph_sha256": _sha(f"{receipt.alias}-graph"),
            "n_directed_edges": receipt.n_nodes * 700,
            "source_prepared_data_sha256": receipt.prepared_data_sha256,
            "source_full_core_preprocessing_sha256": _sha(
                f"{receipt.alias}-full-core"
            ),
        }
    return result


def _synthetic_cohort(
    graph_sources: Mapping[str, Mapping[str, Any]],
) -> SimpleNamespace:
    cores = []
    for receipt in _MODULE.EXPECTED_PREPARED_CORES:
        source = graph_sources[receipt.alias]
        cores.append(
            SimpleNamespace(
                alias=receipt.alias,
                n_nodes=receipt.n_nodes,
                n_genes=1_000,
                checksums=SimpleNamespace(
                    source_manifest_sha256=receipt.manifest_sha256,
                    source_prepared_data_sha256=(
                        receipt.prepared_data_sha256
                    ),
                    source_full_core_preprocessing_sha256=source[
                        "source_full_core_preprocessing_sha256"
                    ],
                    preprocessing_sha256=_sha(
                        f"{receipt.alias}-pooled-preprocessing"
                    ),
                ),
                preprocessing_qc=SimpleNamespace(
                    protected_identifier_arrays_returned=False
                ),
            )
        )
    checksum_values = {
        "ordered_sources_sha256": _sha("ordered-sources"),
        "ordered_gene_schema_sha256": _sha("genes"),
        "ordered_metadata_schema_sha256": _sha("metadata"),
        "expression_mean_sha256": _sha("mean"),
        "expression_scale_sha256": _sha("scale"),
        "combined_fingerprint_sha256": _sha("cohort"),
    }
    return SimpleNamespace(
        aliases=_MODULE.ALIASES,
        total_nodes=117_386,
        n_genes=1_000,
        metadata_names=tuple(_MODULE.ALLOWED_METADATA_COLUMNS),
        gene_names=tuple(f"GENE-{index:04d}" for index in range(1_000)),
        fingerprint_sha256=_sha("cohort"),
        checksums=SimpleNamespace(
            to_dict=lambda: deepcopy(checksum_values)
        ),
        cores=tuple(cores),
    )


def _mask_sources() -> dict[str, dict[str, Any]]:
    result = {}
    for alias in _MODULE.ALIASES:
        entries = []
        for mode in ("partial_gene", "whole_node", "spatial_block"):
            for replicate in range(3):
                entries.append(
                    {
                        "entry_id": f"{mode}-{replicate}",
                        "mode": mode,
                        "replicate": replicate,
                        "seed": 10_000 + replicate,
                        "mask_checksum": _sha(
                            f"{alias}-{mode}-{replicate}"
                        ),
                    }
                )
        result[alias] = {
            "source_run_id": f"r_safe_{alias.lower()}",
            "reference": f"prior/{alias.lower()}/fixed_masks.json",
            "file_sha256": _sha(f"{alias}-mask-file"),
            "bundle_checksum": _sha(f"{alias}-mask-bundle"),
            "base_seed": 9_999,
            "entries": entries,
        }
    return result


@pytest.fixture
def materializer_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    contract_path = project_root / "frozen_task_contract.yaml"
    contract_path.write_text("synthetic: true\n", encoding="utf-8")
    prior_path = project_root / "prior_materialization.json"
    prior_path.write_text('{"synthetic":true}\n', encoding="utf-8")
    output_dir = (
        project_root
        / "scratch"
        / "locked_campaigns"
        / _MODULE.CAMPAIGN_ID
    )
    for alias in _MODULE.ALIASES:
        (
            project_root
            / "data"
            / "processed"
            / "adjacent_normal_10core_qkv_large_k_v1"
            / alias.lower()
            / "prepared_v1"
        ).mkdir(parents=True)

    graphs = _graph_sources()
    cohort = _synthetic_cohort(graphs)
    monkeypatch.setattr(
        _MODULE,
        "_validate_frozen_contract",
        lambda _path: (
            {"campaign_id": _MODULE.CAMPAIGN_ID},
            _MODULE.EXPECTED_FROZEN_CONTRACT_SHA256,
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_prior_materialization",
        lambda _path: (
            {"campaign_id": _MODULE.PRIOR_CAMPAIGN_ID},
            _MODULE.EXPECTED_PRIOR_MATERIALIZATION_CHECKSUM,
            deepcopy(graphs),
        ),
    )

    def fake_component(
        _project_root: Path, reference: str, section: str
    ) -> tuple[dict[str, Any], str]:
        return _component(reference, section), _sha(reference)

    monkeypatch.setattr(_MODULE, "_component_section", fake_component)
    monkeypatch.setattr(
        _MODULE, "load_pooled_full_core_cohort", lambda _paths: cohort
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_prior_mask_sources",
        lambda **_kwargs: _mask_sources(),
    )
    monkeypatch.setattr(
        _MODULE,
        "_validate_model_components",
        lambda _models: _MODULE.EXPECTED_PARAMETER_COUNT,
    )
    return {
        "project_root": project_root,
        "contract_path": contract_path,
        "prior_path": prior_path,
        "output_dir": output_dir,
        "graphs": graphs,
        "cohort": cohort,
    }


def _call(paths: Mapping[str, Any]) -> dict[str, Any]:
    return materialize_campaign(
        project_root=paths["project_root"],
        prior_materialization=paths["prior_path"],
        contract_path=paths["contract_path"],
        output_dir=paths["output_dir"],
    )


def test_materializer_writes_exact_pooled_plan_byte_idempotently(
    materializer_fixture: dict[str, Any],
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
    assert len(first_bytes) == 17
    receipt = json.loads(
        (output / _MODULE.RECEIPT_NAME).read_text(encoding="utf-8")
    )
    unsigned = dict(receipt)
    checksum = unsigned.pop("checksum")
    assert checksum == canonical_sha256(unsigned)
    assert receipt["counts"] == {
        "aliases": 10,
        "pilot_configs": 2,
        "production_configs": 14,
        "production_seeds": 7,
    }
    assert receipt["parameter_count"] == 11_674_880
    assert receipt["parameter_counts"] == {
        arm: 11_674_880 for arm in _MODULE.ARMS
    }
    assert receipt["paired_common_initialization_required"] is True
    assert receipt["registry_mutation_performed"] is False
    assert receipt["queue_mutation_performed"] is False
    assert receipt["training_performed"] is False
    assert receipt["cohort"]["total_nodes"] == 117_386
    assert tuple(receipt["cohort"]["aliases"]) == _MODULE.ALIASES
    assert len(receipt["cores"]) == 10
    assert len(receipt["graph_sources"]) == 10
    assert len(receipt["evaluation_mask_sources"]) == 10

    pilot = receipt["pilot_jobs"]
    assert {
        job["arm"]: job["requested_gpu"] for job in pilot
    } == _MODULE.PILOT_GPU_MAP
    production = receipt["production_jobs"]
    assert {(job["arm"], job["seed"]) for job in production} == {
        (arm, seed)
        for arm in _MODULE.ARMS
        for seed in _MODULE.MODEL_SEEDS
    }
    assert all(
        job["requested_gpu"] == _MODULE.SEED_GPU_MAP[job["seed"]]
        for job in production
    )

    configs = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((output / "production_configs").glob("*.yaml"))
    ]
    assert len(configs) == 14
    for config in configs:
        assert config["evaluation"]["protocol"] == (
            "held_in_pooled_10core_fixed_budget"
        )
        assert tuple(config["dataset"]["core_aliases"]) == _MODULE.ALIASES
        assert len(config["dataset"]["prepared_artifacts"]) == 10
        assert len(config["dataset"]["prepared_artifact_sha256"]) == 10
        assert config["dataset"]["total_fit_cells"] == 117_386
        assert len(config["graph"]["expected_core_graphs"]) == 10
        assert len(config["evaluation"]["prior_mask_sources"]) == 10
        assert config["trainer"]["max_epochs"] == 200
        assert config["trainer"]["total_optimizer_steps"] == 2_000
        assert config["metadata"]["paired_common_initialization_required"]
        uses_graph = (
            config["experiment"]["arm"]
            == "pooled-hybrid-gat-k1000"
        )
        assert config["features"]["use_edge_features"] is uses_graph
        assert config["graph"]["model_graph_input_enabled"] is uses_graph
        serialized = yaml.safe_dump(config)
        assert "SO_" not in serialized
        assert "donor_id" not in serialized.lower()
        assert "patient_id" not in serialized.lower()


def test_materializer_rejects_existing_drift_without_overwrite(
    materializer_fixture: dict[str, Any],
) -> None:
    _call(materializer_fixture)
    path = (
        materializer_fixture["output_dir"]
        / "production_configs"
        / "seed-06_pooled_hybrid_matched_self.yaml"
    )
    path.write_text("tampered: true\n", encoding="utf-8")

    with pytest.raises(MaterializationError, match="different content"):
        _call(materializer_fixture)

    assert path.read_text(encoding="utf-8") == "tampered: true\n"


def test_pooled_loader_total_mismatch_fails_before_publication(
    materializer_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = deepcopy(materializer_fixture["cohort"])
    invalid.total_nodes = 117_385
    monkeypatch.setattr(
        _MODULE, "load_pooled_full_core_cohort", lambda _paths: invalid
    )

    with pytest.raises(MaterializationError, match="pooled cohort"):
        _call(materializer_fixture)

    assert not materializer_fixture["output_dir"].exists()


def test_prior_mask_failure_occurs_before_any_output(
    materializer_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_masks(**_kwargs: Any) -> dict[str, Any]:
        raise MaterializationError("prior mask seed identity changed")

    monkeypatch.setattr(_MODULE, "_validate_prior_mask_sources", reject_masks)

    with pytest.raises(MaterializationError, match="mask seed identity"):
        _call(materializer_fixture)

    assert not materializer_fixture["output_dir"].exists()


def test_real_model_components_have_exact_parameter_match() -> None:
    models = {
        arm: _component(reference, "model")
        for arm, reference in _MODULE._MODEL_COMPONENTS.items()
    }

    assert _MODULE._validate_model_components(models) == 11_674_880
