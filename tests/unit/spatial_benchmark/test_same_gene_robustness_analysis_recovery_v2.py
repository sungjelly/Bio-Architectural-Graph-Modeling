from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_PATH = (
    PROJECT_ROOT
    / "scripts/analysis/run_same_gene_robustness_analysis_recovery_v2.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "same_gene_robustness_analysis_recovery_v2_tests", WRAPPER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
recovery = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = recovery
_SPEC.loader.exec_module(recovery)


class _CoverageError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _amendment_payload() -> dict[str, Any]:
    return json.loads(recovery.DEFAULT_AMENDMENT.read_text(encoding="utf-8"))


def _analyzer_argv(
    *, verify_only: bool = False, verify_only_in_middle: bool = False
) -> list[str]:
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
    if verify_only_in_middle:
        values[2:2] = ["--verify-only"]
    elif verify_only:
        values.append("--verify-only")
    return values


def _fake_analyzer(*, expected_components: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        ARMS=(
            "morphology_only",
            "observed_near",
            "observed_annular",
            "within_fov_permuted_near",
        ),
        EXPECTED_COMPONENTS=expected_components,
        RobustnessAnalysisError=_CoverageError,
    )


def _component_rows(
    *, variants: tuple[str, ...] = ("V0",), seeds: tuple[int, ...] = (17,)
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    arms = _fake_analyzer().ARMS
    for variant in variants:
        for seed in seeds:
            for arm_index, arm in enumerate(arms):
                groups = (2, 1) if arm_index % 2 else (1, 2)
                for group in groups:
                    rows.append(
                        {
                            "variant_id": variant,
                            "model_seed": seed,
                            "fold": group - 1,
                            "arm": arm,
                            "geometry_group": group,
                            "cell_count": 100 + group,
                        }
                    )
    return rows


def test_frozen_v2_authority_verifies_v1_and_every_v2_source() -> None:
    authority = recovery._verify_authority()
    assert authority.payload["amendment_id"] == recovery.AMENDMENT_ID
    assert authority.amendment_size_bytes == recovery.DEFAULT_AMENDMENT.stat().st_size
    assert authority.amendment_sha256 == _sha256(recovery.DEFAULT_AMENDMENT)
    assert {row["role"] for row in authority.parent_files} == set(
        recovery.EXPECTED_V1_FILES
    )
    assert {row["role"] for row in authority.recovery_sources} == set(
        recovery.EXPECTED_V2_PATHS
    )
    v1 = recovery._load_v1_module(authority)
    v1_authority = recovery._verify_v1_layer(v1, authority)
    assert v1_authority.amendment_sha256 == recovery.EXPECTED_V1_FILES[
        "v1_amendment"
    ]["sha256"]


def test_original_launch_source_inventory_remains_exact_48_of_48() -> None:
    manifest_path = (
        PROJECT_ROOT
        / recovery.EXPECTED_EXECUTION_CONTRACT["launch_manifest_path"]
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    assert len(sources) == 48
    assert len({row["path"] for row in sources}) == 48
    for row in sources:
        path = PROJECT_ROOT / row["path"]
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == row["size"]
        assert _sha256(path) == row["sha"]


def test_component_adapter_accepts_equal_integer_maps_with_different_row_order() -> None:
    adapter = recovery._component_coverage_adapter(_fake_analyzer())
    assert adapter(_component_rows()) == (1, 2)
    assert recovery._integer_item_tuple({2: 4, 1: 3}) == ((1, 3), (2, 4))
    assert recovery._integer_item_tuple({1: 3, 2: 4}) == ((1, 3), (2, 4))


def test_component_adapter_does_not_coerce_invalid_fold_values() -> None:
    rows = _component_rows()
    rows[0]["fold"] = object()
    adapter = recovery._component_coverage_adapter(_fake_analyzer())
    with pytest.raises(TypeError):
        adapter(rows)


def test_component_adapter_rejects_planted_arm_fold_mismatch() -> None:
    rows = _component_rows()
    changed = next(
        row
        for row in rows
        if row["arm"] == "observed_annular" and row["geometry_group"] == 2
    )
    changed["fold"] = 9
    adapter = recovery._component_coverage_adapter(_fake_analyzer())
    with pytest.raises(_CoverageError, match="arm component-to-fold assignments differ"):
        adapter(rows)


def test_component_adapter_rejects_planted_arm_count_mismatch() -> None:
    rows = _component_rows()
    changed = next(
        row
        for row in rows
        if row["arm"] == "within_fov_permuted_near"
        and row["geometry_group"] == 1
    )
    changed["cell_count"] += 1
    adapter = recovery._component_coverage_adapter(_fake_analyzer())
    with pytest.raises(_CoverageError, match="arm component cell counts differ"):
        adapter(rows)


def test_component_adapter_preserves_cross_seed_count_check() -> None:
    rows = _component_rows(seeds=(17, 18))
    for row in rows:
        if row["model_seed"] == 18 and row["geometry_group"] == 1:
            row["cell_count"] += 1
    adapter = recovery._component_coverage_adapter(_fake_analyzer())
    with pytest.raises(_CoverageError, match="component cell counts differ across seeds"):
        adapter(rows)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unbound": True}), "keys differ"),
        (
            lambda value: value["parent_layer"]["files"][0].update(
                {"sha256": "0" * 64}
            ),
            "parent identity differs",
        ),
        (
            lambda value: value["failed_analysis_attempts"][1].update(
                {"gates_computed": True}
            ),
            "failed_analysis_attempts differs",
        ),
        (
            lambda value: value["recovery_sources"][0].update(
                {"path": "scripts/wrong.py"}
            ),
            "source path differs",
        ),
    ],
)
def test_amendment_validation_rejects_authority_or_failure_record_drift(
    mutation: Any, message: str
) -> None:
    payload = _amendment_payload()
    assert recovery._validate_amendment(payload) == payload
    changed = copy.deepcopy(payload)
    mutation(changed)
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match=message):
        recovery._validate_amendment(changed)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="invalid"):
        recovery._strict_json_bytes(b'{"a":1,"a":2}', label="fixture")
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="invalid"):
        recovery._strict_json_bytes(b'{"a":NaN}', label="fixture")


def test_same_size_amendment_toctou_is_rejected(tmp_path: Path) -> None:
    relative = recovery.DEFAULT_AMENDMENT.relative_to(recovery.PROJECT_ROOT)
    amendment = tmp_path / relative
    amendment.parent.mkdir(parents=True)
    amendment.write_bytes(b'{"frozen":true}')
    authority = recovery.RecoveryV2Authority(
        amendment_path=amendment,
        amendment_size_bytes=amendment.stat().st_size,
        amendment_sha256=_sha256(amendment),
        payload={},
        parent_files=(),
        recovery_sources=(),
    )
    recovery._verify_authority_unchanged(authority, project_root=tmp_path)
    amendment.write_bytes(b'{"frozen":fals}')
    assert amendment.stat().st_size == authority.amendment_size_bytes
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="SHA-256 changed"):
        recovery._verify_authority_unchanged(authority, project_root=tmp_path)


def test_same_size_bound_source_toctou_is_rejected(tmp_path: Path) -> None:
    relative = recovery.DEFAULT_AMENDMENT.relative_to(recovery.PROJECT_ROOT)
    amendment = tmp_path / relative
    amendment.parent.mkdir(parents=True)
    amendment.write_bytes(b'{"frozen":true}')
    source = tmp_path / "bound.txt"
    source.write_bytes(b"authority")
    row = {
        "role": "fixture",
        "path": "bound.txt",
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }
    authority = recovery.RecoveryV2Authority(
        amendment_path=amendment,
        amendment_size_bytes=amendment.stat().st_size,
        amendment_sha256=_sha256(amendment),
        payload={},
        parent_files=(),
        recovery_sources=(row,),
    )
    recovery._verify_authority_unchanged(authority, project_root=tmp_path)
    source.write_bytes(b"AuthoritY")
    assert source.stat().st_size == row["size_bytes"]
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="SHA-256 differs"):
        recovery._verify_authority_unchanged(authority, project_root=tmp_path)


def test_provenance_preserves_v1_and_binds_v2_with_both_failures() -> None:
    authority = recovery._verify_authority()
    v1 = recovery._load_v1_module(authority)
    v1_authority = recovery._verify_v1_layer(v1, authority)
    base = v1._extend_provenance(
        {
            "schema_version": 1,
            "sources": [],
            "float_policy": {
                "aggregate_dtype": "float64",
                "storage_dtype": "float64",
            },
        },
        authority=v1_authority,
    )
    result = recovery._extend_v2_provenance(
        base,
        authority=authority,
        v1_authority=v1_authority,
    )
    assert result["analysis_recovery"] == base["analysis_recovery"]
    metadata = result["analysis_recovery_v2"]
    assert metadata["amendment_sha256"] == authority.amendment_sha256
    assert metadata["parent_amendment_sha256"] == v1_authority.amendment_sha256
    assert metadata["failed_analysis_attempts"] == recovery.EXPECTED_ATTEMPTS
    assert metadata["newly_patched_symbols"] == [
        "_verify_component_coverage",
        "_source_provenance",
    ]
    source_paths = [row["path"] for row in result["sources"]]
    assert source_paths == sorted(source_paths)
    assert authority.amendment_path.relative_to(PROJECT_ROOT).as_posix() in source_paths
    assert set(recovery.EXPECTED_V2_PATHS.values()).issubset(source_paths)
    assert set(recovery.EXPECTED_V1_FILES[role]["path"] for role in recovery.EXPECTED_V1_FILES).issubset(source_paths)


def test_installed_v2_layer_changes_only_two_new_symbols() -> None:
    authority = recovery._verify_authority()
    v1 = recovery._load_v1_module(authority)
    v1_authority = recovery._verify_v1_layer(v1, authority)
    base = v1._extend_provenance(
        {"schema_version": 1, "sources": []}, authority=v1_authority
    )
    sentinel = object()

    def original_component(_rows: object) -> tuple[int, ...]:
        return (99,)

    def parent_provenance(**_kwargs: object) -> dict[str, Any]:
        return copy.deepcopy(base)

    analyzer = _fake_analyzer()
    analyzer._verify_component_coverage = original_component
    analyzer._source_provenance = parent_provenance
    analyzer.untouched = sentinel
    recovery._install_v2_patches(
        analyzer,
        authority=authority,
        v1=v1,
        v1_authority=v1_authority,
    )
    assert analyzer._verify_component_coverage is not original_component
    assert analyzer._verify_component_coverage(_component_rows()) == (1, 2)
    assert "analysis_recovery_v2" in analyzer._source_provenance()
    assert analyzer.untouched is sentinel


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
def test_v2_parser_rejects_duplicate_or_unrecognized_authority(
    extra: list[str],
) -> None:
    with pytest.raises(recovery.AnalysisRecoveryV2Error):
        recovery._parse_analyzer_invocation([*_analyzer_argv(), *extra])


@pytest.mark.parametrize(
    ("option", "wrong"),
    [
        ("--contract", "experiments/wrong.yaml"),
        ("--launch-manifest", "state/wrong-launch.json"),
        ("--plan", "state/wrong-plan.json"),
        ("--output", "reports/wrong"),
    ],
)
def test_v2_parser_requires_every_exact_execution_path(
    option: str, wrong: str
) -> None:
    values = _analyzer_argv()
    values[values.index(option) + 1] = wrong
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="differs"):
        recovery._parse_analyzer_invocation(values)


def test_verify_only_in_middle_cannot_skip_following_authority_or_unknown() -> None:
    values = _analyzer_argv(verify_only_in_middle=True)
    parsed = recovery._parse_analyzer_invocation(values)
    assert parsed.verify_only is True
    assert parsed.plan_path == recovery.EXPECTED_EXECUTION_CONTRACT["full_plan_path"]
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="unrecognized"):
        recovery._parse_analyzer_invocation(
            [*values[:3], "--unknown", "value", *values[3:]]
        )
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="duplicate"):
        recovery._parse_analyzer_invocation(
            [
                *values,
                "--plan",
                recovery.EXPECTED_EXECUTION_CONTRACT["full_plan_path"],
            ]
        )


def test_recovery_arguments_reject_duplicate_or_mixed_authority() -> None:
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="only once"):
        recovery._arguments(
            [
                "--recovery-v2-amendment",
                str(recovery.DEFAULT_AMENDMENT),
                "--recovery-v2-amendment",
                str(recovery.DEFAULT_AMENDMENT),
            ]
        )
    with pytest.raises(recovery.AnalysisRecoveryV2Error, match="does not accept"):
        recovery._arguments(["--authority-check-only", *_analyzer_argv()])


def test_wrong_cli_fails_before_any_authority_or_outcome_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal reached
        reached = True
        raise AssertionError("authority/outcome access must not occur")

    monkeypatch.setattr(recovery, "_verify_authority", forbidden)
    assert recovery.main([*_analyzer_argv(), "--unknown", "value"]) == 2
    assert reached is False


def test_effective_registry_rejects_bagm_state_root_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = recovery._verify_authority()
    v1 = recovery._load_v1_module(authority)
    v1_authority = recovery._verify_v1_layer(v1, authority)
    analyzer = v1._load_analyzer(v1_authority)
    v1._install_patches(analyzer, authority=v1_authority)
    monkeypatch.setenv("BAGM_STATE_ROOT", str(tmp_path / "unbound-state"))
    with pytest.raises(v1.AnalysisRecoveryError, match="BAGM_STATE_ROOT"):
        v1._verify_effective_registry_path(analyzer, v1_authority)


def test_actual_bound_registry_public_lifecycle_accepts_140_plus_14_rows() -> None:
    authority = recovery._verify_authority()
    v1 = recovery._load_v1_module(authority)
    v1_authority = recovery._verify_v1_layer(v1, authority)
    analyzer = v1._load_analyzer(v1_authority)
    v1._install_patches(analyzer, authority=v1_authority)
    bound = v1._verify_effective_registry_path(analyzer, v1_authority)
    before = _sha256(bound)
    v1._verify_public_registry_lifecycle(
        analyzer,
        v1_authority,
        bound_database=bound,
    )
    assert _sha256(bound) == before


def test_publish_and_verify_only_share_exact_preflight_and_patch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeV1Error(RuntimeError):
        pass

    events: list[str] = []
    observed: list[tuple[str, ...]] = []
    v2_authority = SimpleNamespace(payload={"amendment_id": recovery.AMENDMENT_ID})
    v1_authority = SimpleNamespace(
        payload={"amendment_id": recovery.V1_AMENDMENT_ID},
        amendment_sha256="1" * 64,
    )

    def parse(argv: list[str]) -> Any:
        events.append("v1_parse")
        return original_parse(argv)

    def analyzer_main(argv: tuple[str, ...]) -> int:
        events.append("outcome_read")
        observed.append(tuple(argv))
        return 0

    analyzer = SimpleNamespace(main=analyzer_main)
    fake_v1 = SimpleNamespace(
        AnalysisRecoveryError=FakeV1Error,
        _parse_analyzer_invocation=parse,
        _verify_invocation_authority=lambda *_a, **_k: events.append("v1_invocation"),
        _load_analyzer=lambda *_a, **_k: events.append("load_analyzer") or analyzer,
        _install_patches=lambda *_a, **_k: events.append("install_v1"),
        _verify_effective_registry_path=lambda *_a, **_k: events.append(
            "state_db"
        )
        or Path("registry"),
        _verify_public_registry_lifecycle=lambda *_a, **_k: events.append(
            "public_154"
        ),
    )
    original_parse = recovery._parse_analyzer_invocation
    monkeypatch.setattr(
        recovery,
        "_parse_analyzer_invocation",
        lambda argv: events.append("v2_parse") or original_parse(argv),
    )
    monkeypatch.setattr(
        recovery,
        "_verify_authority",
        lambda *_a, **_k: events.append("v2_authority") or v2_authority,
    )
    monkeypatch.setattr(
        recovery,
        "_load_v1_module",
        lambda *_a, **_k: events.append("load_v1") or fake_v1,
    )
    monkeypatch.setattr(
        recovery,
        "_verify_v1_layer",
        lambda *_a, **_k: events.append("v1_authority") or v1_authority,
    )
    monkeypatch.setattr(
        recovery,
        "_recheck_layers",
        lambda *_a, **_k: events.append("layer_recheck"),
    )
    monkeypatch.setattr(
        recovery,
        "_install_v2_patches",
        lambda *_a, **_k: events.append("install_v2"),
    )

    expected = [
        "v2_parse",
        "v2_authority",
        "load_v1",
        "v1_authority",
        "v1_parse",
        "v1_invocation",
        "layer_recheck",
        "load_analyzer",
        "install_v1",
        "layer_recheck",
        "state_db",
        "public_154",
        "layer_recheck",
        "install_v2",
        "layer_recheck",
        "outcome_read",
        "layer_recheck",
    ]
    for verify_only in (False, True):
        events.clear()
        arguments = _analyzer_argv(verify_only=verify_only)
        assert recovery.main(arguments) == 0
        assert events == expected
    assert observed == [tuple(_analyzer_argv()), tuple(_analyzer_argv(verify_only=True))]


def test_authority_check_only_reads_no_analyzer_or_outcome(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        recovery,
        "_install_v2_patches",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not patch")),
    )
    assert recovery.main(["--authority-check-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is True
    assert result["scientific_outcomes_read"] is False
    assert result["parent_amendment_sha256"] == recovery.EXPECTED_V1_FILES[
        "v1_amendment"
    ]["sha256"]
