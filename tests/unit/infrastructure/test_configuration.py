from __future__ import annotations

from pathlib import Path

import pytest

from spatial_benchmark.configuration import (
    ConfigurationError,
    compose_config,
    load_yaml_mapping,
    validate_experiment_config,
)


GROUPS = {
    "model": (
        "model:\n  name: g2\n  family: edge_conditioned_gatv2\n"
        "  embedding_dim: 64\n"
    ),
    "masking": "masking:\n  type: partial_gene\n  rate: 0.2\n",
    "dataset": (
        "dataset:\n  dataset_id: gastric_cosmx\n  version: v1\n"
        "  split_id: patient_holdout_v1\n"
    ),
    "features": (
        "features:\n  use_edge_features: true\n"
        "  edge_features: [distance_um, contact]\n"
    ),
    "graph": (
        "graph:\n  neighbor_k: 16\n  radius_um: 75.0\n"
        "  symmetry: mutual\n"
    ),
    "trainer": "trainer:\n  learning_rate: 0.001\n  batch_size: 8\n",
    "evaluation": (
        "evaluation:\n  primary_metric: val/masked_huber\n"
        "  primary_direction: minimize\n"
    ),
}


def _configuration_tree(root: Path) -> Path:
    defaults = []
    for group, content in GROUPS.items():
        directory = root / group
        directory.mkdir(parents=True)
        (directory / "default.yaml").write_text(content, encoding="utf-8")
        defaults.append(f"  - {group}: default")
    base = root / "base.yaml"
    base.write_text(
        "defaults:\n"
        + "\n".join(defaults)
        + "\nmodel:\n  embedding_dim: 96\n"
        "seed: 3\nfold: 1\nattempt: 1\n"
        "campaign: cmp_20260724_edge_feature_ablation\n",
        encoding="utf-8",
    )
    return base


def test_strict_composition_merges_namespaced_groups(tmp_path: Path) -> None:
    base = _configuration_tree(tmp_path)

    resolved = compose_config(base, config_root=tmp_path)

    assert "defaults" not in resolved
    assert resolved["model"] == {
        "name": "g2",
        "family": "edge_conditioned_gatv2",
        "embedding_dim": 96,
    }
    assert resolved["dataset"]["split_id"] == "patient_holdout_v1"
    assert resolved["evaluation"]["primary_metric"] == "val/masked_huber"


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "duplicate.yaml"
    config.write_text("seed: 1\nseed: 2\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Duplicate"):
        load_yaml_mapping(config)


def test_group_traversal_and_wrong_parameter_ownership_are_rejected(
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text("defaults:\n  - model: ../secret\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Unsafe"):
        compose_config(unsafe, config_root=tmp_path, validate=False)

    base = _configuration_tree(tmp_path / "valid")
    resolved = compose_config(base, config_root=tmp_path / "valid")
    resolved["trainer"]["embedding_dim"] = 64
    with pytest.raises(ConfigurationError, match="belongs in model"):
        validate_experiment_config(resolved)
