from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import spatial_benchmark.cli as cli
import spatial_benchmark.so1_hl_direct_interactive as so1_hl_interactive
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
        "analyze-embedding-clusters",
        "analyze-contextual-resolution-sweep",
        "analyze-so1-model-embedding-clusters",
        "render-so1-hl-direct-interactive",
        "summarize-variants",
        "export-leaderboard",
        "promote-run",
        "verify-artifacts",
        "import-legacy",
    ):
        assert command in help_text


def test_embedding_analysis_parser_defaults_and_help() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "analyze-embedding-clusters",
            "--run-id",
            "r_test_embedding_analysis",
        ]
    )
    assert arguments.command_name == "analyze-embedding-clusters"
    assert arguments.run_id == "r_test_embedding_analysis"
    assert arguments.checkpoint is None
    assert arguments.n_neighbors == 30
    assert arguments.leiden_resolution == 1.0
    assert arguments.pca_components == 50
    assert arguments.random_seed == 20260825
    assert arguments.device == "cuda:0"
    assert arguments.output_dir is None

    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    help_text = subparsers.choices["analyze-embedding-clusters"].format_help()
    for option in (
        "--run-id",
        "--checkpoint",
        "--n-neighbors",
        "--leiden-resolution",
        "--pca-components",
        "--random-seed",
        "--device",
        "--output-dir",
    ):
        assert option in help_text


def test_embedding_analysis_parser_accepts_explicit_overrides(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.ckpt"
    output = tmp_path / "embedding-analysis"
    arguments = build_parser().parse_args(
        [
            "analyze-embedding-clusters",
            "--checkpoint",
            str(checkpoint),
            "--n-neighbors",
            "41",
            "--leiden-resolution",
            "1.25",
            "--pca-components",
            "32",
            "--random-seed",
            "19",
            "--device",
            "cpu",
            "--output-dir",
            str(output),
        ]
    )
    assert arguments.run_id is None
    assert arguments.checkpoint == checkpoint
    assert arguments.n_neighbors == 41
    assert arguments.leiden_resolution == 1.25
    assert arguments.pca_components == 32
    assert arguments.random_seed == 19
    assert arguments.device == "cpu"
    assert arguments.output_dir == output


def test_so1_post_training_parser_defaults_and_registry_free_viewer(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    parser = build_parser()
    analysis = parser.parse_args(["analyze-so1-model-embedding-clusters"])
    assert analysis.device == "cpu"
    assert analysis.cpu_threads == 40
    assert analysis.n_neighbors == 30
    assert analysis.leiden_resolution == 1.0
    assert analysis.random_seed == 20260825

    viewer = parser.parse_args(
        ["render-so1-hl-direct-interactive", "--run-id", "r_final"]
    )
    assert viewer.run_id == "r_final"

    def fail_registry(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("viewer CLI must not open the experiment registry")

    monkeypatch.setattr(cli, "Registry", fail_registry)
    monkeypatch.setattr(
        so1_hl_interactive,
        "run_so1_hl_direct_interactive",
        lambda **kwargs: {"status": "complete", "run_id": kwargs["run_id"]},
    )
    assert cli.main(
        [
            "--root",
            str(tmp_path),
            "render-so1-hl-direct-interactive",
            "--run-id",
            "r_final",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "r_final"


def test_so1_model_analysis_cli_opens_registry_without_initialization(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    observed: dict[str, object] = {}

    class ReadOnlyRegistrySentinel:
        def __init__(self, path: Path, *, initialize: bool = True) -> None:
            observed["path"] = path
            observed["initialize"] = initialize

    monkeypatch.setattr(cli, "Registry", ReadOnlyRegistrySentinel)
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda arguments, *, registry, paths: {"status": "gated"},
    )
    assert cli.main(
        ["--root", str(tmp_path), "analyze-so1-model-embedding-clusters"]
    ) == 0
    assert observed["initialize"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "gated"


def test_contextual_resolution_sweep_parser_defaults_and_help() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "analyze-contextual-resolution-sweep",
            "--run-id",
            "r_test_contextual_sweep",
        ]
    )
    assert arguments.command_name == "analyze-contextual-resolution-sweep"
    assert arguments.run_id == "r_test_contextual_sweep"
    assert arguments.resolutions == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    assert arguments.random_seed == 20260825
    assert arguments.output_dir is None

    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    help_text = subparsers.choices[
        "analyze-contextual-resolution-sweep"
    ].format_help()
    for option in (
        "--run-id",
        "--resolutions",
        "--random-seed",
        "--output-dir",
    ):
        assert option in help_text


def test_contextual_resolution_sweep_parser_accepts_overrides(
    tmp_path: Path,
) -> None:
    output = tmp_path / "contextual-resolution-sweep"
    arguments = build_parser().parse_args(
        [
            "analyze-contextual-resolution-sweep",
            "--resolutions",
            "0.4",
            "0.8",
            "1.6",
            "--random-seed",
            "73",
            "--output-dir",
            str(output),
        ]
    )
    assert arguments.run_id is None
    assert arguments.resolutions == [0.4, 0.8, 1.6]
    assert arguments.random_seed == 73
    assert arguments.output_dir == output


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
