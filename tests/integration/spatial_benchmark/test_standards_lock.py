from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spatial_benchmark.standards_lock import (  # noqa: E402
    REQUIRED_FINAL_CONDITIONS,
    StandardsLockError,
    authorize_locked_test_job,
    canonical_job_hash,
    condition_name,
    create_standards_lock,
    expand_matrix,
    load_standards_lock,
    verify_locked_test_matrix,
)


def _write_recommendation(
    path: Path,
    *,
    selection: str,
    standard: dict[str, object],
    required_seeds: int = 3,
    test_metrics_used: bool = False,
    extra: dict[str, object] | None = None,
) -> Path:
    value = {
        "locked": True,
        "selection": selection,
        "candidate_id": f"{selection}-candidate",
        "required_seeds": required_seeds,
        "standard": standard,
        "test_metrics_used": test_metrics_used,
        **dict(extra or {}),
    }
    path.write_text(
        json.dumps(value, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _recommendations(root: Path) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    return [
        _write_recommendation(
            root / "graph.json",
            selection="graph",
            standard={
                "k": 12,
                "radius_um": 50.0,
                "symmetry": "union",
                "min_distance_um": 0.0,
            },
        ),
        _write_recommendation(
            root / "mask.json",
            selection="mask",
            standard={"curriculum": "P+N+B"},
        ),
        _write_recommendation(
            root / "hidden.json",
            selection="hidden",
            standard={"hidden_dim": 256},
        ),
        _write_recommendation(
            root / "depth.json",
            selection="depth",
            standard={"graph_layers": 2},
        ),
        _write_recommendation(
            root / "edge.json",
            selection="edge_embedding",
            standard={"edge_embedding_dim": 32},
        ),
    ]


def _read_matrix(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_validation_matrix_templates_are_sealed_and_sequential() -> None:
    config = PROJECT_ROOT / "configs" / "sweep"
    hidden = _read_matrix(
        config / "normal_core_representation_hidden_screen.yaml"
    )
    depth = _read_matrix(
        config / "normal_core_representation_depth_screen.yaml"
    )
    edge = _read_matrix(
        config / "normal_core_g2_edge_embedding_screen.yaml"
    )
    interaction = _read_matrix(
        config / "normal_core_graph_mask_interaction.yaml"
    )
    confirmation = _read_matrix(
        config / "normal_core_top_candidate_seed_confirmation.yaml"
    )

    assert hidden["factors"]["hidden_dim"] == [128, 256, 512]
    assert hidden["factors"]["model"] == ["g1"]
    assert depth["factors"]["graph_layers"] == [1, 2]
    assert depth["fixed"]["hidden_dim"] == 256
    assert edge["factors"]["edge_embedding_dim"] == [16, 32, 64]
    assert edge["factors"]["model"] == ["g2"]
    assert len(expand_matrix(interaction)) == 4
    assert confirmation["factors"]["seed"] == [0, 1, 2]
    assert len(expand_matrix(confirmation)) == 3
    for matrix in (hidden, depth, edge, interaction, confirmation):
        assert matrix["open_test"] is False
        assert matrix["save_predictions"] is False


def test_lock_is_atomic_checksummed_and_has_all_five_seed_controls(
    tmp_path: Path,
) -> None:
    recommendations = _recommendations(tmp_path / "recommendations")
    output = tmp_path / "standards_lock"
    created = create_standards_lock(recommendations, output)
    assert created == output.resolve()
    manifest, lock, matrices = load_standards_lock(output)

    assert manifest["status"] == "complete"
    assert lock["selection_scope"] == "validation_only"
    assert lock["test_metrics_used_for_selection"] is False
    assert lock["final_execution"]["seeds"] == [0, 1, 2, 3, 4]
    assert lock["final_execution"]["required_seed_count"] == 5
    assert lock["final_execution"]["execution_started_by_utility"] is False
    assert lock["final_execution"]["required_conditions"] == list(
        REQUIRED_FINAL_CONDITIONS
    )
    assert lock["g3"]["conditional"] is True
    assert lock["g3"]["enabled_by_lock"] is False
    assert set(matrices) == {
        "matrix_final_locked_ladder.yaml",
        "matrix_top_candidate_confirmation.yaml",
    }

    confirmation = matrices["matrix_top_candidate_confirmation.yaml"]
    assert confirmation["open_test"] is False
    assert confirmation["save_predictions"] is False
    assert confirmation["factors"]["seed"] == [0, 1, 2]
    assert confirmation["fixed"]["k"] == 12
    assert confirmation["fixed"]["hidden_dim"] == 256
    assert confirmation["fixed"]["graph_layers"] == 2

    final = matrices["matrix_final_locked_ladder.yaml"]
    assert final["open_test"] is True
    assert final["save_predictions"] is True
    jobs = expand_matrix(final)
    assert len(jobs) == 5 * len(REQUIRED_FINAL_CONDITIONS)
    observed = {
        condition: {
            int(job["seed"])
            for job in jobs
            if condition_name(job) == condition
        }
        for condition in REQUIRED_FINAL_CONDITIONS
    }
    assert observed == {
        condition: {0, 1, 2, 3, 4}
        for condition in REQUIRED_FINAL_CONDITIONS
    }
    b0_jobs = [job for job in jobs if condition_name(job) == "b0"]
    assert all("k" not in job for job in b0_jobs)
    matched_jobs = [
        job
        for job in jobs
        if condition_name(job) == "b0_parameter_matched"
    ]
    assert all(job["graph_layers"] == 2 for job in matched_jobs)
    rewired_jobs = [
        job for job in jobs if condition_name(job) == "g1_rewired"
    ]
    assert all(job["rewired"] is True for job in rewired_jobs)
    g2_controls = {
        job["edge_control"]
        for job in jobs
        if str(job["model"]) == "g2"
    }
    assert g2_controls == {
        "none",
        "zero",
        "distance_only",
        "permuted",
    }
    assert not (output / "runs").exists()
    assert not list(tmp_path.glob(".standards_lock.tmp-*"))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_standards_lock(recommendations, output)

    final_path = output / "matrix_final_locked_ladder.yaml"
    final_path.write_text(
        final_path.read_text(encoding="utf-8") + "\n# tamper\n",
        encoding="utf-8",
    )
    with pytest.raises(StandardsLockError, match="checksum mismatch"):
        load_standards_lock(output)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda value: value.update({"sealed_test_opened": True}),
            "not validation-only",
        ),
        (
            lambda value: value.update({"test_metrics_used": True}),
            "not validation-only",
        ),
        (
            lambda value: value.update({"required_seeds": 2}),
            "confirmation seeds",
        ),
    ],
)
def test_lock_refuses_unsealed_or_unconfirmed_recommendations(
    tmp_path: Path,
    mutation,
    error: str,
) -> None:
    recommendations = _recommendations(tmp_path / "recommendations")
    path = recommendations[0]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "forbidden_lock"
    with pytest.raises(StandardsLockError, match=error):
        create_standards_lock(recommendations, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".forbidden_lock.tmp-*"))


def test_five_final_seeds_and_g3_condition_are_enforced(
    tmp_path: Path,
) -> None:
    recommendations = _recommendations(tmp_path / "recommendations")
    with pytest.raises(StandardsLockError, match="exactly 5"):
        create_standards_lock(
            recommendations,
            tmp_path / "four_seed_lock",
            final_seeds=[0, 1, 2, 3],
        )
    with pytest.raises(StandardsLockError, match="positive validation-only"):
        create_standards_lock(
            recommendations,
            tmp_path / "ineligible_g3",
            enable_g3=True,
            g3_checkpoint_template=(
                "/locked/b0_seed_{seed}/model_state.pt"
            ),
        )
    recommendations.append(
        _write_recommendation(
            tmp_path / "recommendations" / "g3.json",
            selection="g3_eligibility",
            standard={"eligible": True},
        )
    )
    output = tmp_path / "g3_lock"
    create_standards_lock(
        recommendations,
        output,
        enable_g3=True,
        g3_checkpoint_template="/locked/b0_seed_{seed}/model_state.pt",
    )
    _, lock, matrices = load_standards_lock(output)
    assert lock["g3"]["enabled_by_lock"] is True
    assert lock["g3"]["validation_eligible"] is True
    g3 = matrices["matrix_final_g3_conditional.yaml"]
    assert g3["open_test"] is True
    jobs = expand_matrix(g3)
    assert len(jobs) == 5
    assert {condition_name(job) for job in jobs} == {"g3_conditional"}
    assert {
        job["pretrained_b0_checkpoint"]
        for job in jobs
    } == {
        f"/locked/b0_seed_{seed}/model_state.pt"
        for seed in range(5)
    }


def test_final_matrix_and_each_expanded_job_require_exact_lock_authorization(
    tmp_path: Path,
) -> None:
    lock_path = create_standards_lock(
        _recommendations(tmp_path / "recommendations"),
        tmp_path / "standards_lock",
    )
    manifest, lock, matrices = load_standards_lock(lock_path)
    final_name = lock["final_execution"]["matrix_file"]
    final_path = lock_path / final_name

    verified = verify_locked_test_matrix(lock_path, final_path)
    assert verified["lock_id"] == lock["lock_id"]
    assert verified["artifact_id"] == manifest["artifact_id"]
    assert verified["final_matrix_file"] == final_name
    assert (
        verified["final_matrix_sha256"]
        == manifest["files"][final_name]
    )

    copied_matrix = tmp_path / "byte-identical-copy.yaml"
    shutil.copyfile(final_path, copied_matrix)
    assert (
        verify_locked_test_matrix(lock_path, copied_matrix)[
            "final_matrix_sha256"
        ]
        == verified["final_matrix_sha256"]
    )

    jobs = expand_matrix(matrices[final_name])
    included_job = next(
        job
        for job in jobs
        if condition_name(job) == "g1_rewired" and job["seed"] == 2
    )
    authorization = authorize_locked_test_job(lock_path, included_job)
    assert authorization["condition"] == "g1_rewired"
    assert authorization["canonical_job_hash"] == canonical_job_hash(
        included_job
    )
    assert authorization["canonical_job"] == included_job

    # Include entries are complete standalone jobs. They do not inherit the
    # factor branch's fixed B0 values during either expansion or authorization.
    incomplete_include = dict(included_job)
    incomplete_include.pop("curriculum")
    with pytest.raises(StandardsLockError, match="not exactly authorized"):
        authorize_locked_test_job(lock_path, incomplete_include)

    changed_job = dict(included_job)
    changed_job["hidden_dim"] = 512
    with pytest.raises(StandardsLockError, match="not exactly authorized"):
        authorize_locked_test_job(lock_path, changed_job)

    changed_matrix = yaml.safe_load(copied_matrix.read_text(encoding="utf-8"))
    changed_matrix["include"][0]["hidden_dim"] = 512
    copied_matrix.write_text(
        yaml.safe_dump(changed_matrix, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(StandardsLockError, match="not exactly one"):
        verify_locked_test_matrix(lock_path, copied_matrix)


def test_launcher_enforces_lock_if_and_only_if_matrix_opens_test(
    tmp_path: Path,
) -> None:
    lock_path = create_standards_lock(
        _recommendations(tmp_path / "recommendations"),
        tmp_path / "standards_lock",
    )
    _, lock, _ = load_standards_lock(lock_path)
    launcher = PROJECT_ROOT / "scripts" / "sweeps" / "launch_matrix.py"
    base = [
        sys.executable,
        str(launcher),
        "--prepared",
        str(tmp_path / "missing-prepared"),
        "--output-root",
        str(tmp_path / "runs"),
    ]
    final_path = lock_path / lock["final_execution"]["matrix_file"]
    missing_lock = subprocess.run(
        [*base, "--matrix", str(final_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_lock.returncode != 0
    assert "--standards-lock is required" in missing_lock.stderr
    assert not (tmp_path / "runs").exists()

    confirmation = lock_path / lock["confirmation"]["matrix_file"]
    unnecessary_lock = subprocess.run(
        [
            *base,
            "--matrix",
            str(confirmation),
            "--standards-lock",
            str(lock_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert unnecessary_lock.returncode != 0
    assert "--standards-lock is accepted only" in unnecessary_lock.stderr
    assert not (tmp_path / "runs").exists()

    changed_matrix = tmp_path / "changed-final.yaml"
    value = yaml.safe_load(final_path.read_text(encoding="utf-8"))
    value["include"][0]["seed"] = 99
    changed_matrix.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )
    mismatched = subprocess.run(
        [
            *base,
            "--matrix",
            str(changed_matrix),
            "--standards-lock",
            str(lock_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatched.returncode != 0
    assert "not exactly one checksum-bound final matrix" in mismatched.stderr
    assert not (tmp_path / "runs").exists()


def test_standards_lock_cli_help() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "sweeps" / "lock_standards.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--recommendation" in result.stdout
    assert "--final-seeds" in result.stdout
    assert "never launches jobs" in " ".join(result.stdout.split())
