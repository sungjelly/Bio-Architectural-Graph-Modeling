from __future__ import annotations

from pathlib import Path

from spatial_benchmark.cli import build_parser, main


def test_cli_exposes_required_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "doctor",
        "register-dataset",
        "register-split",
        "create-campaign",
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
    payload = __import__("json").loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["schema_version"] == 3
    assert payload["integrity_check"] == ["ok"]
    assert Path(payload["database"]).is_file()
