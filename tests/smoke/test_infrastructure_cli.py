from __future__ import annotations

import json
from pathlib import Path

import yaml

from spatial_benchmark.cli import build_parser, main
from spatial_benchmark.registry import Registry


def test_cli_exposes_required_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "doctor",
        "register-dataset",
        "register-split",
        "create-campaign",
        "show-campaign",
        "update-campaign",
        "list-campaign-revisions",
        "enqueue-experiment",
        "enqueue-sweep",
        "worker",
        "list-queue",
        "show-run",
        "index-checkpoints",
        "list-checkpoints",
        "show-checkpoint",
        "resolve-checkpoint",
        "export-checkpoint-catalog",
        "summarize-variants",
        "export-leaderboard",
        "promote-run",
        "verify-artifacts",
        "import-legacy",
    ):
        assert command in help_text


def test_doctor_initializes_temp_project(tmp_path: Path, capsys) -> None:
    # Doctor creates generated registry state, but must not bless an arbitrary
    # empty directory as a valid BAGM checkout.
    assert main(["--root", str(tmp_path), "doctor"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["schema_version"] == 4
    assert payload["integrity_check"] == ["ok"]
    assert Path(payload["database"]).is_file()


def test_cli_campaign_update_requires_expected_hash_and_records_revision(
    tmp_path: Path, capsys
) -> None:
    database = tmp_path / "state" / "tracking" / "bagm.sqlite3"
    original = {"active_model_seeds": [0], "deferred_model_seeds": [1, 2, 3, 4]}
    replacement = {
        "active_model_seeds": [0, 1, 2, 3],
        "deferred_model_seeds": [4],
    }
    Registry(database).create_campaign(
        "cmp_parallel", name="Parallel", config=original, status="planned"
    )
    plan = tmp_path / "campaign.yaml"
    plan.write_text(yaml.safe_dump(replacement, sort_keys=False), encoding="utf-8")

    assert main(["--root", str(tmp_path), "show-campaign", "cmp_parallel"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["config"] == original

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "update-campaign",
                "--campaign-id",
                "cmp_parallel",
                "--plan",
                str(plan),
                "--expected-config-sha256",
                shown["config_sha256"],
                "--status",
                "running",
                "--name",
                "Four-seed parallel campaign",
                "--scientific-question",
                "Do four initializations plateau?",
                "--reason",
                "Activate seeds 1-3.",
                "--actor",
                "test-suite",
            ]
        )
        == 0
    )
    updated = json.loads(capsys.readouterr().out)
    assert updated["changed"] is True
    assert updated["revision_id"] == 1
    assert updated["status"] == "running"
    assert updated["name"] == "Four-seed parallel campaign"
    assert updated["scientific_question"] == "Do four initializations plateau?"

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "list-campaign-revisions",
                "cmp_parallel",
            ]
        )
        == 0
    )
    revisions = json.loads(capsys.readouterr().out)
    assert revisions == [
        {
            "actor": "test-suite",
            "campaign_id": "cmp_parallel",
            "created_at": revisions[0]["created_at"],
            "new_config_sha256": updated["config_sha256"],
            "new_name": "Four-seed parallel campaign",
            "new_scientific_question": "Do four initializations plateau?",
            "new_status": "running",
            "previous_config_sha256": shown["config_sha256"],
            "previous_name": "Parallel",
            "previous_scientific_question": None,
            "previous_status": "planned",
            "reason": "Activate seeds 1-3.",
            "revision_id": 1,
        }
    ]
