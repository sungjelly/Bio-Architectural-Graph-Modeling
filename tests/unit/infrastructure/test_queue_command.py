from __future__ import annotations

from pathlib import Path
import sys

import pytest

from spatial_benchmark.configuration import ConfigurationError
from spatial_benchmark.paths import ProjectPaths
from spatial_benchmark.queueing import (
    ArtifactFinalizationError,
    _peak_vram_from_summary,
    _primary_metric,
    command_for_config,
)


def _paths(root: Path) -> ProjectPaths:
    return ProjectPaths.from_environment({"BAGM_ROOT": str(root)})


def test_full_core_protocol_uses_dedicated_worker_owned_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "held_in_full_core_fixed_budget",
            }
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_full_core_capacity.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]
    assert "--output" not in command


def test_grouped_core_adjacency_protocol_uses_adjacency_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "grouped_core_adjacency_ablation_v1",
            },
            "model": {"name": "mean-adjacency-sage"},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_adjacency_ablation.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_grouped_core_adjacency_protocol_rejects_other_models(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ConfigurationError,
        match="requires model.name=mean-adjacency-sage",
    ):
        command_for_config(
            {
                "evaluation": {
                    "protocol": "grouped_core_adjacency_ablation_v1",
                },
                "model": {"name": "g2"},
            },
            paths=_paths(tmp_path),
        )


@pytest.mark.parametrize(
    "model_name",
    ["hybrid-count-gat", "hybrid-count-matched-self"],
)
def test_hybrid_count_protocol_uses_hybrid_runner(
    tmp_path: Path,
    model_name: str,
) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "held_in_full_core_fixed_budget",
            },
            "model": {"name": model_name},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_hybrid_count_capacity.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


@pytest.mark.parametrize(
    "model_name",
    ["hybrid-count-gat", "hybrid-count-matched-self"],
)
def test_pooled_hybrid_protocol_uses_pooled_runner(
    tmp_path: Path,
    model_name: str,
) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "held_in_pooled_10core_fixed_budget",
            },
            "model": {"name": model_name},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_pooled_hybrid_count_capacity.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_pooled_myjju_protocol_uses_genemae_runner(tmp_path: Path) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "held_in_pooled_10core_fixed_budget",
            },
            "model": {"name": "myjju-genemae"},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_myjju_genemae_pooled.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_pooled_protocol_rejects_non_hybrid_model(tmp_path: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match="requires a supported pooled ten-core model",
    ):
        command_for_config(
            {
                "evaluation": {
                    "protocol": "held_in_pooled_10core_fixed_budget",
                },
                "model": {"name": "g2"},
            },
            paths=_paths(tmp_path),
        )


def test_relative_six_core_protocol_uses_dedicated_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "held_in_pooled_6core_relative_qkv_fixed_budget",
            },
            "model": {"name": "relative-qkv-gat"},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_pooled_relative_qkv.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_relative_six_core_joint_plateau_protocol_uses_dedicated_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "held_in_pooled_6core_relative_qkv_joint_plateau",
            },
            "model": {"name": "relative-qkv-gat"},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_pooled_relative_qkv.py"
    )


def test_relative_six_core_protocol_rejects_other_models(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="relative-qkv-gat"):
        command_for_config(
            {
                "evaluation": {
                    "protocol": (
                        "held_in_pooled_6core_relative_qkv_fixed_budget"
                    ),
                },
                "model": {"name": "qkv-gat"},
            },
            paths=_paths(tmp_path),
        )


def test_attention_niche_posthoc_protocol_uses_analysis_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "posthoc_attention_routing_niche_v1",
                "artifact_contract": "analysis_only",
            },
            "model": {"name": "relative-qkv-gat"},
            "campaign": {
                "campaign_id": "cmp_20260825_six_core_attention_routing_niches"
            },
        },
        paths=_paths(tmp_path),
    )

    assert command == [
        sys.executable,
        str(tmp_path / "scripts/analysis/run_attention_routing_niches.py"),
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-id",
        "{run_id}",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_self_hurdle_protocol_uses_graphless_hurdle_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        {
            "evaluation": {
                "protocol": "held_in_full_core_fixed_budget",
            },
            "model": {"name": "self-hurdle-count"},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_self_hurdle_capacity.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_stage0_multiscale_recovery_uses_registered_synthetic_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        {
            "campaign": {
                "campaign_id": "cmp_20260729_multiscale_hurdle_count_pilot",
            },
            "evaluation": {
                "protocol": "held_in_full_core_fixed_budget",
            },
            "metadata": {"execution_role": "stage0_synthetic_recovery"},
            "model": {"name": "multiscale-hurdle-count"},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path
        / "scripts/diagnostics/run_multiscale_synthetic_recovery.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_ordinary_multiscale_held_in_config_uses_multiscale_runner(
    tmp_path: Path,
) -> None:
    command = command_for_config(
        {
            "campaign": {
                "campaign_id": "cmp_20260729_multiscale_hurdle_count_pilot",
            },
            "evaluation": {
                "protocol": "held_in_full_core_fixed_budget",
            },
            "metadata": {"execution_role": "science"},
            "model": {"name": "multiscale-hurdle-count"},
        },
        paths=_paths(tmp_path),
    )

    assert command[1] == str(
        tmp_path / "scripts/train/run_multiscale_hurdle_capacity.py"
    )
    assert command[2:] == [
        "--config",
        "{run_scratch}/config.resolved.yaml",
        "--run-scratch",
        "{run_scratch}",
    ]


def test_primary_metric_rejects_summary_name_drift() -> None:
    with pytest.raises(ArtifactFinalizationError, match="does not match"):
        _primary_metric(
            {
                "primary_metric_name": "val/masked_huber",
                "primary_metric_value": 0.2,
            },
            {
                "evaluation": {
                    "primary_metric": "fit/whole_node/masked_huber",
                }
            },
        )


def test_peak_vram_accepts_explicit_gib_and_rejects_disagreement() -> None:
    assert _peak_vram_from_summary({"peak_vram_gib": 7.25}) == 7.25
    assert _peak_vram_from_summary({"peak_vram_gb": 7.25}) == 7.25
    assert (
        _peak_vram_from_summary(
            {"peak_vram_gb": 7.25, "peak_vram_gib": 7.25}
        )
        == 7.25
    )
    with pytest.raises(ArtifactFinalizationError, match="disagree"):
        _peak_vram_from_summary(
            {"peak_vram_gb": 7.25, "peak_vram_gib": 7.5}
        )
