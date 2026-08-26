from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from spatial_benchmark.configuration import compose_config
from spatial_benchmark.fingerprints import sha256_file
from spatial_benchmark.so1_pooled_full_core import EXPECTED_CELL_COUNTS_BY_CORE


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiment/so1_14core_relative_qkv_seed0_batch2_plateau_min150.yaml"
)
_SPEC = importlib.util.spec_from_file_location(
    "preflight_so1_14core_relative_qkv_ddp",
    PROJECT_ROOT / "scripts/diagnostics/preflight_so1_14core_relative_qkv_ddp.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_PREFLIGHT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREFLIGHT
_SPEC.loader.exec_module(_PREFLIGHT)


def _resolved_config() -> dict[str, object]:
    return compose_config(EXPERIMENT_CONFIG, config_root=PROJECT_ROOT / "configs")


def _prior_receipt(path: Path, *, full_chunk_passed: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "cancer_6core_relative_qkv_preflight_v1",
                "status": "passed",
                "all_required_gates_passed": True,
                "largest_core": {"alias": "CAN-23"},
                "gates": {
                    "full_chunk_exactness": {
                        "passed": full_chunk_passed,
                        "maximum_absolute_difference": 0.0,
                    },
                    "amp_fp32_equivalence": {
                        "passed": True,
                        "maximum_absolute_difference": 1e-4,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_preflight_import_defaults_and_locked_largest_pair() -> None:
    assert _PREFLIGHT.PREFLIGHT_SCHEMA == (
        "so1_14core_relative_qkv_ddp4_preflight_v1"
    )
    assert _PREFLIGHT.PREFLIGHT_ALIASES == ("SO1-C06", "SO1-C11")
    assert EXPECTED_CELL_COUNTS_BY_CORE[6] == 18_212
    assert EXPECTED_CELL_COUNTS_BY_CORE[11] == 18_145
    assert sorted(EXPECTED_CELL_COUNTS_BY_CORE.values(), reverse=True)[:2] == [
        18_212,
        18_145,
    ]
    assert _PREFLIGHT.DEFAULT_OUTPUT == Path(
        "state/preflight/so1_14core_relative_qkv_ddp4_plateau_min150.json"
    )
    parsed = _PREFLIGHT.build_parser().parse_args(["--config", "experiment.yaml"])
    assert parsed.config == Path("experiment.yaml")
    assert parsed.output is None
    assert parsed.prior_equivalence_receipt is None


def test_preflight_training_config_is_one_ddp4_paired_update() -> None:
    config = _PREFLIGHT._preflight_config(
        _resolved_config(), rank=2, local_rank=2
    )
    assert config.model_seed == 0
    assert config.cohort_aliases == ("SO1-C06", "SO1-C11")
    assert config.segment_start_global_epoch == 0
    assert config.segment_end_global_epoch == 1
    assert config.cores_per_optimizer_update == 2
    assert config.optimizer_updates_per_global_epoch == 1
    assert config.mask_views_per_core == 10
    assert config.losses_per_optimizer_update == 20
    assert config.distributed_world_size == 4
    assert config.distributed_rank == 2
    assert config.device == "cuda:2"
    assert config.checkpoint_interval_global_epochs == 1


def test_prior_relative_qkv_equivalence_is_scoped_and_checksum_bound(
    tmp_path: Path,
) -> None:
    prior = _prior_receipt(tmp_path / "prior.json")
    verified = _PREFLIGHT._prior_relative_qkv_equivalence(
        prior_receipt=prior
    )
    assert verified["verified"] is True
    assert verified["scope"] == "shared_relative_qkv_implementation_only"
    assert verified["so1_graph_identity_claimed"] is False
    assert verified["prior_largest_core_alias"] == "CAN-23"
    assert verified["prior_receipt_sha256"] == sha256_file(prior)
    assert verified["prior_full_chunk_exactness"]["passed"] is True
    assert verified["prior_amp_fp32_equivalence"]["passed"] is True


def test_prior_relative_qkv_equivalence_fails_closed_on_gate_drift(
    tmp_path: Path,
) -> None:
    prior = _prior_receipt(tmp_path / "prior.json", full_chunk_passed=False)
    with pytest.raises(RuntimeError, match="lacks the passing"):
        _PREFLIGHT._prior_relative_qkv_equivalence(prior_receipt=prior)

    prior.write_text(
        json.dumps(
            {
                "status": "passed",
                "all_required_gates_passed": True,
                "largest_core": {"alias": "CAN-22"},
                "gates": {
                    "full_chunk_exactness": {"passed": True},
                    "amp_fp32_equivalence": {"passed": True},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacks the passing"):
        _PREFLIGHT._prior_relative_qkv_equivalence(prior_receipt=prior)


def test_atomic_receipt_write_replaces_without_temporary_files(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    first = {"schema": _PREFLIGHT.PREFLIGHT_SCHEMA, "status": "passed"}
    _PREFLIGHT._atomic_json(output, first)
    assert json.loads(output.read_text(encoding="utf-8")) == first

    second = {**first, "prior_relative_qkv_equivalence_verified": True}
    _PREFLIGHT._atomic_json(output, second)
    assert json.loads(output.read_text(encoding="utf-8")) == second
    assert not list(tmp_path.glob(".receipt.json.*.writing"))
