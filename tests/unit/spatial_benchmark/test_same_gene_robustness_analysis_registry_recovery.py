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


def _analyzer_argv(*, verify_only: bool = False) -> list[str]:
    values = [
        "--contract",
        recovery.EXPECTED_EXECUTION_CONTRACT["contract_path"],
        "--launch-manifest",
        recovery.EXPECTED_EXECUTION_CONTRACT["launch_manifest_path"],
        "--plan",
        recovery.EXPECTED_EXECUTION_CONTRACT["full_plan_path"],
        "--output",
        recovery.EXPECTED_EXECUTION_CONTRACT["output_path"],
    ]
    if verify_only:
        values.append("--verify-only")
    return values


def test_frozen_recovery_authority_verifies_every_bound_file() -> None:
    authority = recovery._verify_authority()
    assert authority.payload["amendment_id"] == recovery.AMENDMENT_ID
    assert authority.amendment_sha256 == recovery._sha256_file(
        recovery.DEFAULT_AMENDMENT
    )
    assert authority.amendment_size_bytes == recovery.DEFAULT_AMENDMENT.stat().st_size
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

    changed_execution = copy.deepcopy(payload)
    changed_execution["execution_contract"]["output_path"] = "reports/wrong"
    with pytest.raises(recovery.AnalysisRecoveryError, match="execution_contract"):
        recovery._validate_amendment(changed_execution)


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


def test_initial_amendment_identity_rejects_same_size_toctou(tmp_path: Path) -> None:
    amendment = tmp_path / "amendment.json"
    amendment.write_bytes(b'{"frozen":true}')
    authority = recovery.RecoveryAuthority(
        amendment_path=amendment,
        amendment_size_bytes=amendment.stat().st_size,
        amendment_sha256=recovery._sha256_file(amendment),
        payload={},
        recovery_sources=(),
    )
    recovery._verify_authority_unchanged(authority, project_root=tmp_path)
    amendment.write_bytes(b'{"frozen":fals}')
    assert amendment.stat().st_size == authority.amendment_size_bytes
    with pytest.raises(recovery.AnalysisRecoveryError, match="SHA-256 changed"):
        recovery._verify_authority_unchanged(authority, project_root=tmp_path)
    with pytest.raises(recovery.AnalysisRecoveryError, match="SHA-256 changed"):
        recovery._extend_provenance(
            {"schema_version": 1, "sources": []},
            authority=authority,
            project_root=tmp_path,
        )


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
    assert metadata["amendment_size_bytes"] == authority.amendment_size_bytes
    assert metadata["entrypoint_sha256"] == recovery._sha256_file(WRAPPER_PATH)
    assert metadata["patched_symbols"] == [
        "_registry_configuration",
        "_source_provenance",
    ]
    assert metadata["outcome_access_record"] == recovery.EXPECTED_OUTCOME_ACCESS
    assert metadata["execution_contract"] == recovery.EXPECTED_EXECUTION_CONTRACT


def test_bound_registry_diagnosis_rechecks_only_configuration_metadata() -> None:
    database = PROJECT_ROOT / recovery.EXPECTED_AUTHORITIES["registry_database"]["path"]
    recovery._verify_registry_diagnosis(database)


@pytest.mark.parametrize(
    "extra",
    [
        ["--plan", recovery.EXPECTED_EXECUTION_CONTRACT["full_plan_path"]],
        ["--contract", recovery.EXPECTED_EXECUTION_CONTRACT["contract_path"]],
        ["--verify-only", "--verify-only"],
        ["--unknown", "value"],
        ["positional"],
    ],
)
def test_analyzer_parser_rejects_duplicate_or_unrecognized_authority(
    extra: list[str],
) -> None:
    with pytest.raises(recovery.AnalysisRecoveryError):
        recovery._parse_analyzer_invocation([*_analyzer_argv(), *extra])


def test_verify_only_in_middle_never_skips_the_following_argument() -> None:
    common = _analyzer_argv()
    middle = [*common[:2], "--verify-only", *common[2:]]
    parsed = recovery._parse_analyzer_invocation(middle)
    assert parsed.verify_only is True
    assert parsed.launch_manifest_path == recovery.EXPECTED_EXECUTION_CONTRACT[
        "launch_manifest_path"
    ]
    assert parsed.plan_path == recovery.EXPECTED_EXECUTION_CONTRACT["full_plan_path"]

    with pytest.raises(recovery.AnalysisRecoveryError, match="unrecognized"):
        recovery._parse_analyzer_invocation(
            [*common[:2], "--verify-only", "--unknown", "value", *common[2:]]
        )
    with pytest.raises(recovery.AnalysisRecoveryError, match="duplicate analyzer authority"):
        recovery._parse_analyzer_invocation(
            [
                *middle,
                "--plan",
                recovery.EXPECTED_EXECUTION_CONTRACT["full_plan_path"],
            ]
        )


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("contract_path", "experiments/wrong.yaml"),
        ("launch_manifest_path", "state/wrong-launch.json"),
        ("plan_path", "state/wrong-plan.json"),
        ("output_path", "reports/analyses/wrong"),
    ],
)
def test_invocation_must_match_every_amendment_path(
    field: str, wrong: str
) -> None:
    authority = recovery._verify_authority()
    values = _analyzer_argv()
    option = {
        "contract_path": "--contract",
        "launch_manifest_path": "--launch-manifest",
        "plan_path": "--plan",
        "output_path": "--output",
    }[field]
    values[values.index(option) + 1] = wrong
    invocation = recovery._parse_analyzer_invocation(values)
    with pytest.raises(recovery.AnalysisRecoveryError, match=field):
        recovery._verify_invocation_authority(invocation, authority)


@pytest.mark.parametrize(
    "arguments",
    [
        [
            *_analyzer_argv(),
            "--plan",
            recovery.EXPECTED_EXECUTION_CONTRACT["full_plan_path"],
        ],
        [
            *_analyzer_argv()[:-1],
            "reports/analyses/wrong",
        ],
    ],
)
def test_main_rejects_wrong_or_duplicate_cli_before_analyzer_import(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = False

    def load(*_args: object, **_kwargs: object) -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError("must fail before analyzer import")

    monkeypatch.setattr(recovery, "_load_analyzer", load)
    assert recovery.main(arguments) == 2
    assert loaded is False


def test_effective_registry_rejects_bagm_state_root_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from spatial_benchmark.paths import current_paths

    authority = recovery._verify_authority()
    monkeypatch.setenv("BAGM_STATE_ROOT", str(tmp_path / "unbound-state"))
    analyzer = SimpleNamespace(current_paths=current_paths)
    with pytest.raises(recovery.AnalysisRecoveryError, match="BAGM_STATE_ROOT"):
        recovery._verify_effective_registry_path(analyzer, authority)


def test_actual_registry_public_lifecycle_accepts_all_154_decoded_rows() -> None:
    authority = recovery._verify_authority()
    before = recovery._sha256_file(
        PROJECT_ROOT / recovery.EXPECTED_AUTHORITIES["registry_database"]["path"]
    )
    analyzer = recovery._load_analyzer(authority)
    recovery._install_patches(analyzer, authority=authority)
    bound = recovery._verify_effective_registry_path(analyzer, authority)
    recovery._verify_public_registry_lifecycle(
        analyzer,
        authority,
        bound_database=bound,
    )
    assert recovery._sha256_file(bound) == before


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
    assert recovery.main(_analyzer_argv()) == 2
    assert loaded is False


def test_publish_and_verify_only_arguments_use_the_identical_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = object()
    events: list[str] = []
    observed: list[list[str]] = []

    def analyzer_main(argv: list[str]) -> int:
        events.append("outcome_read")
        observed.append(list(argv))
        return 0

    analyzer = SimpleNamespace(main=analyzer_main)
    original_parse = recovery._parse_analyzer_invocation
    monkeypatch.setattr(
        recovery,
        "_parse_analyzer_invocation",
        lambda argv: events.append("parse") or original_parse(argv),
    )
    monkeypatch.setattr(
        recovery,
        "_verify_authority",
        lambda *_a, **_k: events.append("authority") or authority,
    )
    monkeypatch.setattr(
        recovery,
        "_verify_invocation_authority",
        lambda *_a, **_k: events.append("invocation"),
    )
    monkeypatch.setattr(
        recovery,
        "_verify_authority_unchanged",
        lambda *_a, **_k: events.append("amendment_recheck"),
    )
    monkeypatch.setattr(
        recovery,
        "_load_analyzer",
        lambda *_a, **_k: events.append("load") or analyzer,
    )
    monkeypatch.setattr(
        recovery,
        "_install_patches",
        lambda *_a, **_k: events.append("install"),
    )
    monkeypatch.setattr(
        recovery,
        "_verify_effective_registry_path",
        lambda *_a, **_k: events.append("registry_path") or Path("registry"),
    )
    monkeypatch.setattr(
        recovery,
        "_verify_public_registry_lifecycle",
        lambda *_a, **_k: events.append("registry_lifecycle"),
    )

    for verify_only in (False, True):
        events.clear()
        arguments = _analyzer_argv(verify_only=verify_only)
        assert recovery.main(arguments) == 0
        assert events == [
            "parse",
            "authority",
            "invocation",
            "amendment_recheck",
            "load",
            "install",
            "amendment_recheck",
            "registry_path",
            "registry_lifecycle",
            "amendment_recheck",
            "outcome_read",
            "amendment_recheck",
        ]
    assert observed == [_analyzer_argv(), _analyzer_argv(verify_only=True)]


def test_recovery_arguments_reject_duplicates_and_check_only_mixing() -> None:
    with pytest.raises(recovery.AnalysisRecoveryError, match="only once"):
        recovery._arguments(
            [
                "--recovery-amendment",
                str(recovery.DEFAULT_AMENDMENT),
                "--recovery-amendment",
                str(recovery.DEFAULT_AMENDMENT),
            ]
        )
    with pytest.raises(recovery.AnalysisRecoveryError, match="does not accept"):
        recovery._arguments(["--authority-check-only", *_analyzer_argv()])


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
