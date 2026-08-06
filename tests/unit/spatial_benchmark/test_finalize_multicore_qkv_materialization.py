from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import pytest
import yaml

from spatial_benchmark.identifiers import canonical_sha256


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "train"
    / "finalize_multicore_qkv_materialization.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "finalize_multicore_qkv_materialization_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

FinalizationError = _MODULE.MulticoreMaterializationFinalizationError
finalize_materialization = _MODULE.finalize_materialization


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _graph(alias: str, *, k: int, n_nodes: int) -> dict[str, Any]:
    degree = 800 if k == 1000 else 4000
    edge_count = n_nodes * degree
    kth_distance = 900.0 + int(alias[-2:]) * 3.0 + k / 10_000
    return {
        "k": k,
        "n_directed_edges": edge_count,
        "graph_sha256": _sha(f"{alias}-graph-{k}"),
        "candidate_kth_distance_max_um": kth_distance,
        "radius_guard_margin_um": 2000.0 - kth_distance,
        "mean_degree": edge_count / n_nodes,
        "n_components": 1,
        "n_isolated_nodes": 0,
    }


def _config(
    *,
    alias: str,
    arm: str,
    artifact_reference: str,
    preprocessing_sha256: str,
    split_id: str,
    split_fingerprint: str,
    graphs: dict[int, dict[str, Any]],
    requested_gpu: int,
) -> dict[str, Any]:
    is_self = arm == "matched_self"
    k = 1000 if arm == "k1000" else 5000
    graph = graphs[k]
    return {
        "campaign": {"campaign_id": _MODULE.CAMPAIGN_ID},
        "experiment": {
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
        },
        "dataset": {
            "biological_unit_alias": alias,
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "prepared_artifact_reference": artifact_reference,
            "dataset_fingerprint": preprocessing_sha256,
            "split_id": split_id,
            "split_fingerprint": split_fingerprint,
            "validation_or_test_partition_present": False,
        },
        "model": {
            "name": "qkv-gat-matched-self" if is_self else "qkv-gat",
            "family": (
                "qkv_parameter_matched_self_control"
                if is_self
                else "edge_aware_qkv_graph_transformer"
            ),
            "hidden_dim": 576,
            "graph_layers": 8,
            "attention_heads": 9,
        },
        "graph": {
            "k": k,
            "neighbor_k": k,
            "radius_um": 2000.0,
            "radius_guard_um": 2000.0,
            "expected_directed_edges": graph["n_directed_edges"],
            "expected_materialized_graph_sha256": graph["graph_sha256"],
        },
        "features": {"use_edge_features": not is_self},
        "trainer": {"max_epochs": 300},
        "launcher": {
            "requested_gpu": str(requested_gpu),
            "requested_gpu_count": 1,
            "distributed": False,
        },
        "seed": 0,
        "fold": 0,
        "attempt": 1,
    }


def _write_materialization(
    path: Path,
    payload: dict[str, Any],
) -> None:
    value = deepcopy(payload)
    value.pop("checksum", None)
    value["checksum"] = canonical_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def campaign_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    selection = project_root / "selection.json"
    selection.write_text('{"locked":"selection"}\n', encoding="utf-8")
    selection_sha = _sha('{"locked":"selection"}\n')
    monkeypatch.setattr(
        _MODULE,
        "EXPECTED_SELECTION_SHA256",
        selection_sha,
    )

    aliases_by_artifact: dict[Path, tuple[str, int]] = {}
    materialized_cores = []
    jobs = []
    config_paths: dict[tuple[str, str], Path] = {}
    for alias_index, alias in enumerate(_MODULE.ALIASES, start=1):
        n_nodes = 6000 + alias_index
        artifact_reference = (
            f"protected/{alias.lower()}/prepared_v1"
        )
        artifact = project_root / artifact_reference
        artifact.mkdir(parents=True)
        aliases_by_artifact[artifact.resolve()] = (alias, n_nodes)
        preprocessing_sha = _sha(f"{alias}-preprocessing")
        split_id = _sha(f"{alias}-split")[:16]
        split_fingerprint = _sha(f"{alias}-split")
        graphs = {
            k: _graph(alias, k=k, n_nodes=n_nodes)
            for k in _MODULE.GRAPH_K_VALUES
        }
        config_refs = {}
        for arm in _MODULE.ARMS:
            config_reference = f"configs/{alias.lower()}_{arm}.yaml"
            config_path = project_root / config_reference
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = _config(
                alias=alias,
                arm=arm,
                artifact_reference=artifact_reference,
                preprocessing_sha256=preprocessing_sha,
                split_id=split_id,
                split_fingerprint=split_fingerprint,
                graphs=graphs,
                requested_gpu=0,
            )
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False),
                encoding="utf-8",
            )
            if arm == "k1000":
                estimated_work = float(
                    graphs[1000]["n_directed_edges"]
                )
            elif arm == "k5000":
                estimated_work = float(
                    graphs[5000]["n_directed_edges"]
                )
            else:
                estimated_work = n_nodes * 100.0
            jobs.append(
                {
                    "alias": alias,
                    "arm": arm,
                    "config": config_reference,
                    "estimated_work": estimated_work,
                }
            )
            config_refs[arm] = config_reference
            config_paths[(alias, arm)] = config_path
        materialized_cores.append(
            {
                "alias": alias,
                "prepared_artifact": artifact_reference,
                "n_nodes": n_nodes,
                "n_genes": 1000,
                "preprocessing_sha256": preprocessing_sha,
                "split_id": split_id,
                "split_fingerprint": split_fingerprint,
                "graphs": {
                    str(k): graph for k, graph in graphs.items()
                },
                "configs": config_refs,
            }
        )

    legacy_jobs, _ = _MODULE._assign_gpus(
        {
            (job["alias"], job["arm"]): job
            for job in jobs
        }
    )
    materialization = {
        "schema_version": 1,
        "campaign_id": _MODULE.CAMPAIGN_ID,
        "cohort": {
            "tissue_context": "pathology_confirmed_adjacent_normal",
            "core_count": 10,
            "donor_count": 10,
            "slide_balance": {"SO_1": 5, "SO_2": 5},
            "true_normal_core_count": 0,
            # The finalizer must replace this with the external SHA.
            "protected_selection_manifest_sha256": _sha("stale"),
        },
        "model": {
            "parameter_count": 36_749_480,
            "hidden_dim": 576,
            "graph_layers": 8,
            "attention_heads": 9,
            "attention_head_dim": 64,
            "fixed_epochs": 300,
        },
        "materialized_cores": materialized_cores,
        "jobs": legacy_jobs,
        "job_count": 30,
    }
    materialization_path = project_root / "campaign_materialization.json"
    _write_materialization(materialization_path, materialization)

    artifact_calls: list[tuple[Path, bool]] = []

    def fake_load_prepared_artifact(
        path: Path,
        *,
        load_arrays: bool,
    ) -> tuple[dict[str, Any], None, dict[str, Any]]:
        resolved = Path(path).resolve()
        artifact_calls.append((resolved, load_arrays))
        alias, n_nodes = aliases_by_artifact[resolved]
        return (
            {
                "artifact_kind": (
                    "adjacent_normal_tissue_spatial_benchmark_preparation"
                ),
                "selection": {
                    "opaque_alias": alias,
                    "restricted_identifiers_emitted": False,
                    "tissue_context": (
                        "pathology_confirmed_adjacent_normal"
                    ),
                    "n_cells": n_nodes,
                },
                "inputs": {
                    "protected_selection_manifest": {
                        "sha256": selection_sha,
                    }
                },
                "features": {"n_biological_probes": 1000},
            },
            None,
            {},
        )

    def fake_route(path: Path, alias: str) -> SimpleNamespace:
        index = int(alias[-2:])
        return SimpleNamespace(
            alias=alias,
            slide="SO_1" if index <= 5 else "SO_2",
            fovs=(index,),
        )

    monkeypatch.setattr(
        _MODULE,
        "load_prepared_artifact",
        fake_load_prepared_artifact,
    )
    monkeypatch.setattr(
        _MODULE,
        "load_adjacent_normal_route",
        fake_route,
    )
    monkeypatch.setattr(
        _MODULE,
        "validate_experiment_config",
        lambda config: None,
    )
    return {
        "project_root": project_root,
        "selection": selection,
        "selection_sha": selection_sha,
        "materialization": materialization_path,
        "config_paths": config_paths,
        "artifact_calls": artifact_calls,
    }


def test_finalizer_audits_and_rewrites_self_aware_assignment_atomically(
    campaign_fixture: dict[str, Any],
) -> None:
    result = finalize_materialization(
        materialization_path=campaign_fixture["materialization"],
        selection_manifest=campaign_fixture["selection"],
        project_root=campaign_fixture["project_root"],
    )

    assert result["cohort"]["protected_selection_manifest_sha256"] == (
        campaign_fixture["selection_sha"]
    )
    assert result["job_count"] == 30
    assert result["finalization"] == {
        "schema_version": 1,
        "utility": "finalize_multicore_qkv_materialization",
        "external_selection_sha256": campaign_fixture["selection_sha"],
        "validated_alias_count": 10,
        "validated_artifact_count": 10,
        "validated_graph_record_count": 20,
        "validated_config_count": 30,
        "gpu_assignment": result["finalization"]["gpu_assignment"],
        "graph_recomputation_performed": False,
        "registry_or_queue_mutation_performed": False,
        "accepted_input_contract": (
            "legacy_unfinalized_v1_or_strict_finalized_v1"
        ),
    }
    gpu_audit = result["finalization"]["gpu_assignment"]
    assert gpu_audit["algorithm"] == "self_aware_lpt_v1"
    assert gpu_audit["gpu_count"] == 8
    assert set(gpu_audit["loads"]) == {str(index) for index in range(8)}
    assert sum(gpu_audit["job_counts"].values()) == 30
    assert len(campaign_fixture["artifact_calls"]) == 10
    assert all(
        load_arrays is False
        for _, load_arrays in campaign_fixture["artifact_calls"]
    )

    materialized = json.loads(
        campaign_fixture["materialization"].read_text(encoding="utf-8")
    )
    checksum = materialized.pop("checksum")
    assert checksum == canonical_sha256(materialized)
    observed_coverage = {
        (job["alias"], job["arm"]) for job in materialized["jobs"]
    }
    assert observed_coverage == {
        (alias, arm)
        for alias in _MODULE.ALIASES
        for arm in _MODULE.ARMS
    }
    for job in result["jobs"]:
        key = (job["alias"], job["arm"])
        config = yaml.safe_load(
            campaign_fixture["config_paths"][key].read_text(
                encoding="utf-8"
            )
        )
        assert config["launcher"]["requested_gpu"] == str(
            job["requested_gpu"]
        )
        assert job["config_sha256"] == canonical_sha256(config)
        assert 0 <= job["requested_gpu"] < 8
        if job["arm"] == "matched_self":
            core = next(
                record
                for record in result["materialized_cores"]
                if record["alias"] == job["alias"]
            )
            expected = (
                core["n_nodes"] * 100.0
                + 0.01
                * core["graphs"]["5000"]["n_directed_edges"]
            )
            assert job["estimated_work"] == expected


def test_finalizer_is_idempotent(
    campaign_fixture: dict[str, Any],
) -> None:
    first = finalize_materialization(
        materialization_path=campaign_fixture["materialization"],
        selection_manifest=campaign_fixture["selection"],
        project_root=campaign_fixture["project_root"],
    )
    first_bytes = campaign_fixture["materialization"].read_bytes()
    first_configs = {
        key: path.read_bytes()
        for key, path in campaign_fixture["config_paths"].items()
    }
    second = finalize_materialization(
        materialization_path=campaign_fixture["materialization"],
        selection_manifest=campaign_fixture["selection"],
        project_root=campaign_fixture["project_root"],
    )
    assert first == second
    assert campaign_fixture["materialization"].read_bytes() == first_bytes
    assert {
        key: path.read_bytes()
        for key, path in campaign_fixture["config_paths"].items()
    } == first_configs


def test_finalizer_rejects_external_selection_change_without_writes(
    campaign_fixture: dict[str, Any],
) -> None:
    materialization_before = campaign_fixture["materialization"].read_bytes()
    configs_before = {
        key: path.read_bytes()
        for key, path in campaign_fixture["config_paths"].items()
    }
    campaign_fixture["selection"].write_text(
        '{"changed":"selection"}\n',
        encoding="utf-8",
    )
    with pytest.raises(FinalizationError, match="external SHA"):
        finalize_materialization(
            materialization_path=campaign_fixture["materialization"],
            selection_manifest=campaign_fixture["selection"],
            project_root=campaign_fixture["project_root"],
        )
    assert campaign_fixture["materialization"].read_bytes() == (
        materialization_before
    )
    assert {
        key: path.read_bytes()
        for key, path in campaign_fixture["config_paths"].items()
    } == configs_before


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("graph_margin", "radius-guard"),
        ("graph_hash", "checksum"),
        ("job_coverage", "unique alias/arm"),
    ],
)
def test_finalizer_rejects_receipt_tampering_before_rewrites(
    campaign_fixture: dict[str, Any],
    tamper: str,
    message: str,
) -> None:
    path = campaign_fixture["materialization"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("checksum")
    if tamper == "graph_margin":
        payload["materialized_cores"][0]["graphs"]["5000"][
            "radius_guard_margin_um"
        ] += 1.0
    elif tamper == "graph_hash":
        payload["materialized_cores"][0]["graphs"]["1000"][
            "graph_sha256"
        ] = "not-a-sha"
    else:
        payload["jobs"][0] = deepcopy(payload["jobs"][1])
    _write_materialization(path, payload)
    materialization_before = path.read_bytes()
    configs_before = {
        key: config_path.read_bytes()
        for key, config_path in campaign_fixture["config_paths"].items()
    }
    with pytest.raises(FinalizationError, match=message):
        finalize_materialization(
            materialization_path=path,
            selection_manifest=campaign_fixture["selection"],
            project_root=campaign_fixture["project_root"],
        )
    assert path.read_bytes() == materialization_before
    assert {
        key: config_path.read_bytes()
        for key, config_path in campaign_fixture["config_paths"].items()
    } == configs_before


def test_finalizer_rejects_artifact_input_sha_and_config_digest_mismatch(
    campaign_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_loader = _MODULE.load_prepared_artifact

    def bad_artifact_loader(
        path: Path,
        *,
        load_arrays: bool,
    ) -> tuple[dict[str, Any], None, dict[str, Any]]:
        manifest, arrays, masks = original_loader(
            path,
            load_arrays=load_arrays,
        )
        value = deepcopy(manifest)
        value["inputs"]["protected_selection_manifest"]["sha256"] = _sha(
            "wrong-selection"
        )
        return value, arrays, masks

    monkeypatch.setattr(
        _MODULE,
        "load_prepared_artifact",
        bad_artifact_loader,
    )
    with pytest.raises(FinalizationError, match="input SHA"):
        finalize_materialization(
            materialization_path=campaign_fixture["materialization"],
            selection_manifest=campaign_fixture["selection"],
            project_root=campaign_fixture["project_root"],
        )

    monkeypatch.setattr(
        _MODULE,
        "load_prepared_artifact",
        original_loader,
    )
    config_path = campaign_fixture["config_paths"][("ANC-01", "k1000")]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["trainer"]["max_epochs"] = 299
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(FinalizationError, match="campaign/core contract"):
        finalize_materialization(
            materialization_path=campaign_fixture["materialization"],
            selection_manifest=campaign_fixture["selection"],
            project_root=campaign_fixture["project_root"],
        )


def test_legacy_migration_rejects_unknown_gpu_state(
    campaign_fixture: dict[str, Any],
) -> None:
    materialization_path = campaign_fixture["materialization"]
    payload = json.loads(materialization_path.read_text(encoding="utf-8"))
    payload.pop("checksum")
    payload["jobs"][0]["requested_gpu"] = (
        int(payload["jobs"][0]["requested_gpu"]) + 1
    ) % 8
    _write_materialization(materialization_path, payload)
    with pytest.raises(FinalizationError, match="deterministic legacy"):
        finalize_materialization(
            materialization_path=materialization_path,
            selection_manifest=campaign_fixture["selection"],
            project_root=campaign_fixture["project_root"],
        )

    # The only accepted stale launcher state is the externally audited
    # all-zero legacy state.
    payload = json.loads(materialization_path.read_text(encoding="utf-8"))
    payload.pop("checksum")
    first = payload["jobs"][0]
    first["requested_gpu"] = 0
    # Restore the exact deterministic assignment from all current work rows.
    job_map = {
        (job["alias"], job["arm"]): job for job in payload["jobs"]
    }
    restored, _ = _MODULE._assign_gpus(job_map)
    payload["jobs"] = restored
    _write_materialization(materialization_path, payload)
    config_path = campaign_fixture["config_paths"][("ANC-01", "k1000")]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["launcher"]["requested_gpu"] = "1"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(FinalizationError, match="launcher-GPU-zero"):
        finalize_materialization(
            materialization_path=materialization_path,
            selection_manifest=campaign_fixture["selection"],
            project_root=campaign_fixture["project_root"],
        )
