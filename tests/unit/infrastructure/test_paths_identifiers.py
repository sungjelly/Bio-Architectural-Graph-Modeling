from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from spatial_benchmark.identifiers import (
    IdentifierError,
    canonical_json,
    create_run_id,
    repro_id,
    semantic_run_alias,
    scientific_id,
)
from spatial_benchmark.paths import ProjectPaths, discover_project_root


def test_project_paths_honor_relative_and_absolute_overrides(tmp_path: Path) -> None:
    external_state = tmp_path.parent / f"{tmp_path.name}-state"
    paths = ProjectPaths.from_environment(
        {
            "BAGM_ROOT": str(tmp_path),
            "BAGM_DATA_ROOT": "protected-data",
            "BAGM_STATE_ROOT": str(external_state),
        }
    )

    assert paths.project_root == tmp_path.resolve()
    assert paths.data_root == (tmp_path / "protected-data").resolve()
    assert paths.artifact_root == (tmp_path / "artifacts").resolve()
    assert paths.state_root == external_state.resolve()
    assert paths.scratch_root == (tmp_path / "scratch").resolve()
    assert paths.config_root == (tmp_path / "configs").resolve()
    assert paths.export_root == (tmp_path / "exports").resolve()


def test_explicit_anchor_wins_over_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intended = tmp_path / "intended"
    unrelated = tmp_path / "unrelated"
    (intended / ".git").mkdir(parents=True)
    (unrelated / ".git").mkdir(parents=True)
    anchor = intended / "src" / "package" / "module.py"
    anchor.parent.mkdir(parents=True)
    anchor.touch()
    monkeypatch.chdir(unrelated)

    assert discover_project_root(anchor) == intended.resolve()


def test_canonical_json_and_scientific_exclusions_are_deterministic() -> None:
    first = {
        "model": {"name": "g1", "embedding_dim": 64},
        "graph": {"neighbor_k": 16},
        "seed": 1,
        "fold": 0,
        "launcher": {"kind": "local", "heartbeat_seconds": 30},
        "artifact_path": "/one",
    }
    second = {
        "artifact_path": "/different",
        "launcher": {"kind": "slurm", "heartbeat_seconds": 90},
        "fold": 4,
        "seed": 99,
        "graph": {"neighbor_k": 16},
        "model": {"embedding_dim": 64, "name": "g1"},
    }

    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert scientific_id(first) == scientific_id(second)
    changed = {**second, "graph": {"neighbor_k": 32}}
    assert scientific_id(first) != scientific_id(changed)


def test_repro_id_changes_for_reproducibility_inputs() -> None:
    config = {"model": {"name": "g1"}, "seed": 3}
    base = {
        "git_commit": "abc123",
        "dirty_fingerprint": None,
        "dataset_fingerprint": "data-a",
        "split_fingerprint": "split-a",
        "preprocessing_version": "prep-1",
        "environment_fingerprint": "env-a",
    }
    identifier = repro_id(config, **base)

    for key in (
        "git_commit",
        "dataset_fingerprint",
        "split_fingerprint",
        "preprocessing_version",
        "environment_fingerprint",
    ):
        changed = dict(base)
        changed[key] = f"{base[key]}-changed"
        assert repro_id(config, **changed) != identifier

    with pytest.raises(IdentifierError, match="environment"):
        repro_id(config, **{**base, "environment_fingerprint": None})


def test_run_id_is_auditable_and_deterministic_when_inputs_are_fixed() -> None:
    timestamp = datetime(2026, 7, 24, 4, 12, 33, tzinfo=timezone.utc)
    run_id = create_run_id(
        seed=3,
        fold=2,
        attempt=1,
        scientific_id_value="sci_7a91c6e212345678",
        timestamp=timestamp,
        unique_suffix="deadbeef",
    )

    assert run_id == "r_20260724T041233Z_7a91c6e2_s003_f02_a01_deadbeef"


def _semantic_categories() -> dict[str, object]:
    return {
        "lifecycle_stage": "Sealed Final",
        "study_axis": "Locked Ladder",
        "model": "G1",
        "graph": "k16_r75_mutual_rbf8",
        "mask": "P+N+B",
        "edge_feature_state": "none",
        "embedding_dim": 512,
        "seed": 3,
        "fold": None,
        "attempt": None,
        "variant_evidence": {
            "dataset_id": "cosmx_normal_core_legacy_v1",
            "split_id": "127b17da70688537",
        },
    }


def test_semantic_run_alias_is_date_free_safe_and_deterministic() -> None:
    categories = _semantic_categories()
    reordered = dict(reversed(list(categories.items())))

    alias = semantic_run_alias(
        primary_run_id="lr_g1_example",
        categories=categories,
        historical=True,
    )

    assert alias == semantic_run_alias(
        primary_run_id="lr_g1_example",
        categories=reordered,
        historical=True,
    )
    assert alias.startswith(
        "hist.sealed-final.locked-ladder.g1.k16-r75-mutual-rbf8."
        "p-n-b.none.d512.s003.fna.ana."
    )
    assert "2026" not in alias
    assert len(alias) <= 180
    assert alias.replace(".", "").replace("-", "").isalnum()


def test_semantic_run_alias_uses_known_execution_coordinates() -> None:
    categories = {
        **_semantic_categories(),
        "fold": 2,
        "attempt": 1,
    }
    alias = semantic_run_alias(
        primary_run_id="r_primary",
        categories=categories,
        historical=False,
    )

    assert alias.startswith("run.")
    assert ".s003.f02.a01." in alias


def test_semantic_run_alias_hashes_prevent_normalized_display_collisions() -> None:
    first_categories = _semantic_categories()
    second_categories = {**first_categories, "graph": "k16 r75 mutual rbf8"}

    first = semantic_run_alias(
        primary_run_id="legacy-a",
        categories=first_categories,
        historical=True,
    )
    normalized_collision = semantic_run_alias(
        primary_run_id="legacy-a",
        categories=second_categories,
        historical=True,
    )
    different_execution = semantic_run_alias(
        primary_run_id="legacy-b",
        categories=first_categories,
        historical=True,
    )

    assert first.split(".")[:-2] == normalized_collision.split(".")[:-2]
    assert first.split(".")[-2] != normalized_collision.split(".")[-2]
    assert first.split(".")[-1] != different_execution.split(".")[-1]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"fold": -1}, "fold"),
        ({"attempt": 0}, "attempt"),
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"model": ""}, "model"),
        ({"variant_evidence": "not-a-mapping"}, "variant_evidence"),
    ],
)
def test_semantic_run_alias_rejects_invalid_explicit_evidence(
    change: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(IdentifierError, match=message):
        semantic_run_alias(
            primary_run_id="legacy-a",
            categories={**_semantic_categories(), **change},
            historical=True,
        )


def test_semantic_run_alias_requires_every_visible_category() -> None:
    categories = _semantic_categories()
    del categories["study_axis"]

    with pytest.raises(IdentifierError, match="study_axis"):
        semantic_run_alias(
            primary_run_id="legacy-a",
            categories=categories,
            historical=True,
        )
