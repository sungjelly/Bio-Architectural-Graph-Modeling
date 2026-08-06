from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import yaml

from spatial_benchmark.identifiers import create_run_id


_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _ROOT
    / "scripts"
    / "analysis"
    / "compare_multicore_qkv_large_k.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "compare_multicore_qkv_large_k_for_tests",
    _SCRIPT,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

MultiCoreQKVComparisonError = _MODULE.MultiCoreQKVComparisonError
compare_multicore_qkv_large_k = _MODULE.compare_multicore_qkv_large_k
write_comparison = _MODULE.write_comparison

_CAMPAIGN_ID = "cmp_20260728_adjacent_normal_10core_qkv_large_k"
_ALIASES = tuple(f"ANC-{index:02d}" for index in range(1, 11))
_ROLES = ("k1000", "k5000", "matched_self")


def _load_single_core_fixture_module() -> Any:
    path = (
        _ROOT
        / "tests"
        / "unit"
        / "spatial_benchmark"
        / "test_compare_full_core_qkv_large_k.py"
    )
    spec = importlib.util.spec_from_file_location(
        "single_core_qkv_fixture_for_multicore_tests",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def synthetic_campaign(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, dict[str, Path]]]:
    root = tmp_path_factory.mktemp("multicore-qkv-campaign")
    fixture = _load_single_core_fixture_module()
    original_config = fixture._config
    current = {"alias": "ANC-01", "index": 1}

    def config_with_alias(role: str, **kwargs: Any) -> dict[str, Any]:
        config = original_config(role, **kwargs)
        config["campaign"]["campaign_id"] = _CAMPAIGN_ID
        config["dataset"]["biological_unit_alias"] = current["alias"]
        prefix = str(current["alias"]).lower().replace("-", "")
        config["experiment"]["variant_label"] = f"{prefix}_{role}"
        return config

    def unique_run_id(role: str) -> str:
        role_index = _ROLES.index(role)
        moment = datetime(
            2026, 7, 28, 12, 0, tzinfo=timezone.utc
        ) + timedelta(seconds=3 * int(current["index"]) + role_index)
        token = {
            "k1000": "k1",
            "k5000": "k5",
            "matched_self": "ms",
        }[role]
        return create_run_id(
            seed=0,
            fold=0,
            attempt=1,
            scientific_id_value=(
                f"sci_{int(current['index']):02x}{role_index:02x}"
                "abcdef123456"
            ),
            timestamp=moment,
            unique_suffix=(
                f"anc{int(current['index']):02d}{token}"
            ),
        )

    fixture._config = config_with_alias
    fixture._run_id = unique_run_id

    paths: dict[str, dict[str, Path]] = {}
    for index, alias in enumerate(_ALIASES, start=1):
        current.update(alias=alias, index=index)
        offset = 0.001 * index
        paths[alias] = {
            "k1000": fixture._make_bundle(
                root,
                role="k1000",
                whole_huber=(
                    0.949 + offset,
                    0.950 + offset,
                    0.951 + offset,
                ),
                whole_pve=(8.0 + index, 9.0 + index, 10.0 + index),
            ),
            "k5000": fixture._make_bundle(
                root,
                role="k5000",
                whole_huber=(
                    0.899 + offset,
                    0.900 + offset,
                    0.901 + offset,
                ),
                whole_pve=(12.0 + index, 13.0 + index, 14.0 + index),
            ),
            "matched_self": fixture._make_bundle(
                root,
                role="matched_self",
                whole_huber=(
                    0.999 + offset,
                    1.000 + offset,
                    1.001 + offset,
                ),
                whole_pve=(5.0 + index, 6.0 + index, 7.0 + index),
            ),
        }

    manifest = {
        "schema_version": 1,
        "campaign_id": _CAMPAIGN_ID,
        "cores": {
            alias: {
                role: path.as_posix()
                for role, path in role_paths.items()
            }
            for alias, role_paths in paths.items()
        },
    }
    manifest_path = root / "campaign_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return manifest_path, paths


def test_locked_multicore_comparison_uses_ten_core_level_pairs(
    synthetic_campaign: tuple[Path, dict[str, dict[str, Path]]],
    tmp_path: Path,
) -> None:
    manifest, _paths = synthetic_campaign

    result = compare_multicore_qkv_large_k(manifest)

    assert result["status"] == "complete"
    assert result["compatibility"]["run_count"] == 30
    assert result["compatibility"]["core_aliases"] == list(_ALIASES)
    assert result["representation_gate"]["passes"] is True
    assert result["large_k_gate"]["passes"] is True
    representation = result["aggregate_contrasts"][
        "k5000_vs_matched_self"
    ]
    assert representation["n_independent_core_units"] == 10
    assert representation["technical_masks_averaged_per_core"] == 3
    assert len(representation["core_level_values"]) == 10
    gain = representation["relative_huber_gain_percent"]
    assert gain["positive_core_count"] == 10
    assert gain["exact_sign_flip_one_sided_p"] == pytest.approx(
        1.0 / 1024.0
    )
    assert gain["paired_t_95_percent_ci"]["lower"] > 0.0
    assert representation["confirmatory_inference"][
        "holm_adjusted_exact_sign_flip_one_sided_p"
    ] < 0.05
    assert representation["pve_percent"][
        "candidate_core_mean"
    ] > 0.0
    assert representation["pve_percent"][
        "candidate_minus_reference_mean_points"
    ] > 0.0

    output = write_comparison(result, tmp_path / "comparison")
    assert sorted(path.name for path in output.iterdir()) == [
        "comparison.json",
        "report.md",
    ]
    persisted = json.loads(
        (output / "comparison.json").read_text(encoding="utf-8")
    )
    assert len(persisted["cores"]) == 10
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "inference uses 10 core-level pairs" in report
    assert "pathology-adjacent normal tissue" in report
    assert "not biological replicates" in report
    assert "do not establish a biological mechanism" in report
    with pytest.raises(
        MultiCoreQKVComparisonError, match="already exists"
    ):
        write_comparison(result, output)


def test_manifest_rejects_non_distinct_run_directories(
    synthetic_campaign: tuple[Path, dict[str, dict[str, Path]]],
    tmp_path: Path,
) -> None:
    _manifest, paths = synthetic_campaign
    manifest = {
        "schema_version": 1,
        "campaign_id": _CAMPAIGN_ID,
        "cores": {
            alias: {
                role: path.as_posix()
                for role, path in role_paths.items()
            }
            for alias, role_paths in paths.items()
        },
    }
    manifest["cores"]["ANC-10"]["matched_self"] = manifest["cores"][
        "ANC-10"
    ]["k5000"]
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(
        MultiCoreQKVComparisonError,
        match="30 distinct run directories",
    ):
        compare_multicore_qkv_large_k(path)


def test_exact_statistics_and_holm_are_deterministic() -> None:
    assert _MODULE._exact_sign_flip_p([1.0] * 10) == pytest.approx(
        1.0 / 1024.0
    )
    assert _MODULE._exact_sign_flip_p([-1.0] * 10) == pytest.approx(1.0)
    adjusted = _MODULE._holm_adjust({"a": 0.01, "b": 0.04})
    assert adjusted == pytest.approx({"a": 0.02, "b": 0.04})
    interval = _MODULE._paired_t_ci([2.0] * 10)
    assert interval["lower"] == pytest.approx(2.0)
    assert interval["upper"] == pytest.approx(2.0)
