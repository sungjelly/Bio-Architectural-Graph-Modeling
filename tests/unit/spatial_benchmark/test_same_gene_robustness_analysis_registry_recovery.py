from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from spatial_benchmark.identifiers import scientific_id
from spatial_benchmark.registry import Registry


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_PATH = (
    PROJECT_ROOT
    / "scripts/analysis/run_same_gene_robustness_registry_recovery.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "same_gene_robustness_analysis_registry_recovery_tests", WRAPPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
recovery = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = recovery
_SPEC.loader.exec_module(recovery)


class _AdapterError(RuntimeError):
    pass


def _amendment_payload() -> dict[str, object]:
    return json.loads(recovery.DEFAULT_AMENDMENT.read_text(encoding="utf-8"))


def test_frozen_recovery_authority_verifies_every_bound_file() -> None:
    authority = recovery._verify_authority()
    assert authority.payload["amendment_id"] == recovery.AMENDMENT_ID
    assert authority.amendment_sha256 == recovery._sha256_file(
        recovery.DEFAULT_AMENDMENT
    )
    assert {row["path"] for row in authority.recovery_sources} == {
        recovery.WRAPPER_PATH,
        recovery.SCHEMA_PATH,
        recovery.TEST_PATH,
    }


def test_decoded_registry_configuration_matches_registry_public_api(
    tmp_path: Path,
) -> None:
    registry = Registry(tmp_path / "registry.sqlite3")
    registry.create_campaign(recovery.CAMPAIGN_ID, name="recovery fixture")
    configuration = {
        "evaluation": {"protocol": "held_out_geometry_masked_reconstruction"},
        "robustness_variant": {"variant_id": "V0"},
    }
    identifier = scientific_id(configuration)
    registry.register_variant(
        identifier,
        campaign_id=recovery.CAMPAIGN_ID,
        configuration=configuration,
    )
    registry.create_run(
        "fixture-run",
        campaign_id=recovery.CAMPAIGN_ID,
        scientific_id=identifier,
        repro_id="fixture-repro",
        seed=260810,
        fold=0,
        attempt=1,
        configuration=configuration,
    )
    row = registry.get_run("fixture-run")
    assert row is not None
    assert "config_json" not in row
    assert isinstance(row["config"], dict)
    assert recovery._decoded_registry_configuration(
        row, error_type=_AdapterError
    ) == configuration


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"config": "not-decoded"},
        {"config_json": "{}"},
        {"configuration_json": "{}"},
        {"config": {}, "config_json": "{}"},
        {"config": {"invalid": float("nan")}},
    ],
)
def test_decoded_registry_configuration_rejects_every_nonpublic_shape(
    row: dict[str, object],
) -> None:
    with pytest.raises(_AdapterError):
        recovery._decoded_registry_configuration(row, error_type=_AdapterError)


def test_amendment_validation_is_exact_and_rejects_authority_drift() -> None:
    payload = _amendment_payload()
    assert recovery._validate_amendment(payload) == payload

    extra = copy.deepcopy(payload)
    extra["unbound"] = True
    with pytest.raises(recovery.AnalysisRecoveryError, match="keys differ"):
        recovery._validate_amendment(extra)

    changed = copy.deepcopy(payload)
    changed["authorities"][0]["sha256"] = "0" * 64
    with pytest.raises(recovery.AnalysisRecoveryError, match="authority differs"):
        recovery._validate_amendment(changed)

    changed_access = copy.deepcopy(payload)
    changed_access["outcome_access_record"]["aggregate_computed"] = True
    with pytest.raises(recovery.AnalysisRecoveryError, match="outcome_access_record"):
        recovery._validate_amendment(changed_access)


def test_file_verifier_rejects_same_size_byte_tamper(tmp_path: Path) -> None:
    path = tmp_path / "bound.txt"
    path.write_bytes(b"authority")
    row = {
        "path": "bound.txt",
        "size_bytes": path.stat().st_size,
        "sha256": recovery._sha256_file(path),
    }
    assert recovery._verify_file_row(
        row, project_root=tmp_path, label="fixture"
    ) == path
    path.write_bytes(b"AuthoritY")
    with pytest.raises(recovery.AnalysisRecoveryError, match="SHA-256 differs"):
        recovery._verify_file_row(row, project_root=tmp_path, label="fixture")


def test_provenance_extension_binds_amendment_wrapper_schema_and_tests() -> None:
    authority = recovery._verify_authority()
    result = recovery._extend_provenance(
        {
            "schema_version": 1,
            "sources": [],
            "float_policy": {
                "aggregate_dtype": "float64",
                "storage_dtype": "float64",
            },
        },
        authority=authority,
    )
    paths = [row["path"] for row in result["sources"]]
    amendment_relative = authority.amendment_path.relative_to(PROJECT_ROOT).as_posix()
    assert paths == sorted(paths)
    assert set(paths) == {
        amendment_relative,
        recovery.WRAPPER_PATH,
        recovery.SCHEMA_PATH,
        recovery.TEST_PATH,
        *{
            row["path"]
            for row in authority.payload["authorities"]
        },
    }
    metadata = result["analysis_recovery"]
    assert metadata["amendment_sha256"] == authority.amendment_sha256
    assert metadata["entrypoint_sha256"] == recovery._sha256_file(WRAPPER_PATH)
    assert metadata["patched_symbols"] == [
        "_registry_configuration",
        "_source_provenance",
    ]
    assert metadata["outcome_access_record"] == recovery.EXPECTED_OUTCOME_ACCESS


def test_bound_registry_diagnosis_rechecks_only_configuration_metadata() -> None:
    database = PROJECT_ROOT / recovery.EXPECTED_AUTHORITIES["registry_database"]["path"]
    recovery._verify_registry_diagnosis(database)


def test_installed_adapter_changes_only_the_two_authorized_symbols() -> None:
    authority = recovery._verify_authority()

    class CanonicalError(RuntimeError):
        pass

    def original_provenance(**_kwargs: object) -> dict[str, object]:
        return {"schema_version": 1, "sources": []}

    sentinel = object()
    analyzer = SimpleNamespace(
        _registry_configuration=sentinel,
        _source_provenance=original_provenance,
        RobustnessAnalysisError=CanonicalError,
        untouched=sentinel,
    )
    recovery._install_patches(analyzer, authority=authority)
    assert analyzer._registry_configuration({"config": {"a": 1}}) == {"a": 1}
    assert "analysis_recovery" in analyzer._source_provenance()
    assert analyzer.untouched is sentinel


def test_main_fails_authority_before_loading_canonical_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = False

    def reject(*_args: object, **_kwargs: object) -> object:
        raise recovery.AnalysisRecoveryError("synthetic hash mismatch")

    def load(*_args: object, **_kwargs: object) -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError("must not load")

    monkeypatch.setattr(recovery, "_verify_authority", reject)
    monkeypatch.setattr(recovery, "_load_analyzer", load)
    assert recovery.main(["--verify-only"]) == 2
    assert loaded is False


def test_publish_and_verify_only_arguments_use_the_identical_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = object()
    observed: list[list[str]] = []
    analyzer = SimpleNamespace(main=lambda argv: observed.append(list(argv)) or 0)
    monkeypatch.setattr(recovery, "_verify_authority", lambda *_a, **_k: authority)
    monkeypatch.setattr(recovery, "_load_analyzer", lambda *_a, **_k: analyzer)
    monkeypatch.setattr(recovery, "_install_patches", lambda *_a, **_k: None)

    common = ["--contract", "contract.yaml", "--output", "report"]
    assert recovery.main(common) == 0
    assert recovery.main([*common, "--verify-only"]) == 0
    assert observed == [common, [*common, "--verify-only"]]


def test_authority_check_only_does_not_import_analyzer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        recovery,
        "_load_analyzer",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not import")),
    )
    assert recovery.main(["--authority-check-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verified"] is True
    assert payload["scientific_outcomes_read"] is False
