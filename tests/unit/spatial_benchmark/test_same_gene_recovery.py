from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from spatial_benchmark.identifiers import canonical_json, canonical_sha256
from spatial_benchmark import same_gene_recovery as recovery


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_row(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha": _sha(path),
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _authority_fixture(root: Path) -> dict[str, Any]:
    contract = root / recovery.CONTRACT_RELATIVE_PATH
    environment = root / recovery.ENVIRONMENT_LOCK_RELATIVE_PATH
    tracked = root / "src/synthetic_parent_source.py"
    for path, content in (
        (contract, "frozen contract\n"),
        (environment, '{"environment":"frozen"}\n'),
        (tracked, "PARENT_VALUE = 1\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    _git(root, "init")
    _git(root, "config", "user.email", "recovery-tests@example.invalid")
    _git(root, "config", "user.name", "Recovery Tests")
    _git(
        root,
        "add",
        recovery.CONTRACT_RELATIVE_PATH,
        recovery.ENVIRONMENT_LOCK_RELATIVE_PATH,
        "src/synthetic_parent_source.py",
    )
    _git(root, "commit", "-m", "frozen parent sources")
    parent_commit = _git(root, "rev-parse", "HEAD")

    parent_sources = sorted(
        (
            _source_row(root, recovery.CONTRACT_RELATIVE_PATH),
            _source_row(root, recovery.ENVIRONMENT_LOCK_RELATIVE_PATH),
            _source_row(root, "src/synthetic_parent_source.py"),
        ),
        key=lambda row: row["path"],
    )
    parent_launch_path = root / "state/materialized/parent/launch.json"
    parent_launch = {
        "campaign_id": recovery.CAMPAIGN_ID,
        "contract": {
            "path": recovery.CONTRACT_RELATIVE_PATH,
            "sha": _sha(contract),
        },
        "sources": parent_sources,
    }
    _write_json(parent_launch_path, parent_launch)
    parent_plan_path = root / "state/materialized/parent/pilot/plan.json"
    parent_ledger_path = root / "state/materialized/parent/pilot/ledger.json"
    _write_json(parent_plan_path, {"authority": "parent-plan"})
    _write_json(parent_ledger_path, {"authority": "parent-ledger"})

    source_list_relative = (
        "experiments/campaigns/"
        f"{recovery.CAMPAIGN_ID}/source_authorities_recovery_r1.json"
    )
    source_list_path = root / source_list_relative
    _write_json(
        source_list_path,
        [
            source_list_relative,
            recovery.TECHNICAL_AMENDMENT_RELATIVE_PATH,
        ],
    )
    child_launch_relative = "state/materialized/recovery_r1/launch.json"
    child_launch_path = root / child_launch_relative
    child_core_sources = sorted(
        (
            _source_row(root, recovery.ENVIRONMENT_LOCK_RELATIVE_PATH),
            _source_row(root, source_list_relative),
            _source_row(root, "src/synthetic_parent_source.py"),
        ),
        key=lambda row: row["path"],
    )
    child_core = {
        "campaign_id": recovery.CAMPAIGN_ID,
        "contract": {
            "path": recovery.CONTRACT_RELATIVE_PATH,
            "sha": _sha(contract),
        },
        "sources": child_core_sources,
    }
    failed_attempts = [
        {
            "variant": f"V{index}",
            "attempt": 1,
            "model_seed": 20260810,
            "fold": 0,
            "scientific_id": f"sci_{index:016x}",
            "run_id": f"failed-run-v{index}",
            "materialized_config_sha256": f"{index + 1:064x}",
            "artifact_path": f"artifacts/failed-v{index}",
            "failed_marker_sha256": f"{index + 11:064x}",
            "exception_sha256": f"{index + 21:064x}",
            "registry_status": "failed",
            "artifact_status": "failed",
            "failure_category": "same_gene_nonlinear_run_failure",
        }
        for index in range(7)
    ]
    amendment_payload = {
        "schema_version": 1,
        "amendment_id": recovery.TECHNICAL_AMENDMENT_ID,
        "campaign_id": recovery.CAMPAIGN_ID,
        "status": "frozen_technical_recovery",
        "frozen_at": "2026-08-11T01:00:00Z",
        "scope": "execution_only_no_scientific_change",
        "scientific_effects_accessed": False,
        "contract": {
            "path": recovery.CONTRACT_RELATIVE_PATH,
            "sha256": _sha(contract),
        },
        "parent_authority": {
            "launch_path": parent_launch_path.relative_to(root).as_posix(),
            "launch_sha256": _sha(parent_launch_path),
            "source_manifest_sha256": recovery.source_manifest_sha256(
                parent_sources
            ),
            "pilot_plan_path": parent_plan_path.relative_to(root).as_posix(),
            "pilot_plan_sha256": _sha(parent_plan_path),
            "pilot_ledger_path": parent_ledger_path.relative_to(root).as_posix(),
            "pilot_ledger_sha256": _sha(parent_ledger_path),
            "git_commit": parent_commit,
        },
        "child_authority": {
            "launch_path": child_launch_relative,
            "launch_core_sha256": canonical_sha256(child_core),
            "source_list_path": source_list_relative,
            "source_list_sha256": _sha(source_list_path),
            "profile": "pilot",
            "first_recovery_attempt": 2,
            "required_variants": [f"V{index}" for index in range(7)],
        },
        "authorized_corrections": list(recovery.AUTHORIZED_CORRECTIONS),
        "scientific_invariants": {
            "campaign_id_unchanged": True,
            "contract_sha256_unchanged": True,
            "prepared_data_fingerprints_unchanged": True,
            "environment_lock_sha256_unchanged": True,
            "scientific_payload_must_match_parent": True,
            "seeds_folds_hyperparameters_gates_unchanged": True,
            "all_latest_pilot_attempts_share_child_launch": True,
            "production_blocked_until_all_seven_receipts_verify": True,
        },
        "failed_attempts": failed_attempts,
    }
    amendment_path = root / recovery.TECHNICAL_AMENDMENT_RELATIVE_PATH
    _write_json(amendment_path, amendment_payload)
    amendment = recovery.load_technical_amendment(
        amendment_path, project_root=root
    )
    child_launch = {
        **child_core,
        "sources": sorted(
            (
                *child_core_sources,
                _source_row(root, recovery.TECHNICAL_AMENDMENT_RELATIVE_PATH),
            ),
            key=lambda row: row["path"],
        ),
    }
    _write_json(child_launch_path, child_launch)
    return {
        "amendment": amendment,
        "amendment_path": amendment_path,
        "amendment_payload": amendment_payload,
        "child_launch": child_launch,
        "child_launch_path": child_launch_path,
        "tracked": tracked,
    }


def test_recovery_authority_verifies_parent_commit_and_child_core(
    tmp_path: Path,
) -> None:
    fixture = _authority_fixture(tmp_path)

    parent = recovery.verify_parent_git_sources(
        fixture["amendment"], project_root=tmp_path
    )
    recovery.verify_child_launch_binding(
        fixture["amendment"],
        fixture["child_launch"],
        launch_path=fixture["child_launch_path"],
        project_root=tmp_path,
    )

    assert parent["campaign_id"] == recovery.CAMPAIGN_ID
    assert recovery.launch_core_sha256(fixture["child_launch"]) == (
        fixture["amendment"].child_launch_core_sha256
    )
    assert set(fixture["amendment"].failed_attempts) == {
        f"V{index}" for index in range(7)
    }


@pytest.mark.parametrize("tamper", ("core", "environment", "amendment"))
def test_child_launch_binding_rejects_every_bound_identity_tamper(
    tmp_path: Path, tamper: str
) -> None:
    fixture = _authority_fixture(tmp_path)
    changed = deepcopy(fixture["child_launch"])
    if tamper == "core":
        next(
            row
            for row in changed["sources"]
            if row["path"] == "src/synthetic_parent_source.py"
        )["sha"] = "f" * 64
    elif tamper == "environment":
        next(
            row
            for row in changed["sources"]
            if row["path"] == recovery.ENVIRONMENT_LOCK_RELATIVE_PATH
        )["sha"] = "e" * 64
    else:
        next(
            row
            for row in changed["sources"]
            if row["path"] == recovery.TECHNICAL_AMENDMENT_RELATIVE_PATH
        )["sha"] = "d" * 64

    with pytest.raises(recovery.TechnicalAmendmentError):
        recovery.verify_child_launch_binding(
            fixture["amendment"],
            changed,
            launch_path=fixture["child_launch_path"],
            project_root=tmp_path,
        )


def test_parent_git_commit_cannot_authorize_different_source_bytes(
    tmp_path: Path,
) -> None:
    fixture = _authority_fixture(tmp_path)
    fixture["tracked"].write_text("PARENT_VALUE = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "src/synthetic_parent_source.py")
    _git(tmp_path, "commit", "-m", "different source")
    changed = deepcopy(fixture["amendment_payload"])
    changed["parent_authority"]["git_commit"] = _git(tmp_path, "rev-parse", "HEAD")
    _write_json(fixture["amendment_path"], changed)
    amendment = recovery.load_technical_amendment(
        fixture["amendment_path"], project_root=tmp_path
    )

    with pytest.raises(
        recovery.TechnicalAmendmentError, match="parent Git source identity changed"
    ):
        recovery.verify_parent_git_sources(amendment, project_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["authorized_corrections"].pop(),
            "authorized technical corrections changed",
        ),
        (
            lambda payload: payload["failed_attempts"].reverse(),
            "failed-attempt order changed",
        ),
        (
            lambda payload: payload["parent_authority"].__setitem__(
                "launch_path", "/tmp/parent-launch.json"
            ),
            "project-relative",
        ),
    ),
)
def test_amendment_loader_rejects_scope_inventory_and_path_tamper(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    fixture = _authority_fixture(tmp_path)
    changed = deepcopy(fixture["amendment_payload"])
    mutation(changed)
    _write_json(fixture["amendment_path"], changed)

    with pytest.raises(recovery.TechnicalAmendmentError, match=message):
        recovery.load_technical_amendment(
            fixture["amendment_path"], project_root=tmp_path
        )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"key":1,"key":2}\n', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"key":Infinity}\n', encoding="utf-8")

    with pytest.raises(recovery.TechnicalAmendmentError, match="duplicate"):
        recovery.strict_json(duplicate, label="synthetic")
    with pytest.raises(recovery.TechnicalAmendmentError, match="strict synthetic"):
        recovery.strict_json(nonfinite, label="synthetic")


def test_launch_core_requires_exactly_one_amendment_source(tmp_path: Path) -> None:
    fixture = _authority_fixture(tmp_path)
    without = deepcopy(fixture["child_launch"])
    without["sources"] = [
        row
        for row in without["sources"]
        if row["path"] != recovery.TECHNICAL_AMENDMENT_RELATIVE_PATH
    ]

    with pytest.raises(
        recovery.TechnicalAmendmentError, match="exactly one technical-amendment"
    ):
        recovery.launch_core_sha256(without)
